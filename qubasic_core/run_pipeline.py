"""QUBASIC run pipeline: backend options, method selection, RUN, SAMPLE, ESTIMATE, STEP."""

from __future__ import annotations

import sys
import time
import numpy as np
from typing import Any
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator

from qubasic_core.engine import (
    ExecResult,
)
from qubasic_core.errors import QBasicRangeError
from qubasic_core.host import HostState


def _has_channel(qc) -> bool:
    """Whether a circuit applies a non-unitary channel (APPLYCHANNEL)."""
    return any(inst.operation.name in ('kraus', 'superop', 'quantum_channel')
               for inst in qc.data)

def _circuit_fingerprint(qc) -> str:
    """Stable digest of a circuit's registers and instructions.

    Covers operation names, parameters (arrays by content), qubit and clbit
    indices, nested control-flow blocks, if_test conditions, and the operator
    and step count of Hamiltonian evolution gates.
    """
    import hashlib
    from qiskit import QuantumCircuit as _QC
    h = hashlib.blake2b(digest_size=16)

    def feed(obj) -> None:
        if isinstance(obj, np.ndarray):
            h.update(obj.tobytes())
        elif isinstance(obj, _QC):
            h.update(_circuit_fingerprint(obj).encode())
        elif hasattr(obj, 'data') and isinstance(getattr(obj, 'data', None), np.ndarray):
            h.update(obj.data.tobytes())
        elif isinstance(obj, (list, tuple)):
            h.update(b'[')
            for x in obj:
                feed(x)
            h.update(b']')
        else:
            h.update(repr(obj).encode())

    h.update(repr([(r.name, r.size) for r in qc.qregs]).encode())
    h.update(repr([(r.name, r.size) for r in qc.cregs]).encode())
    for inst in qc.data:
        op = inst.operation
        h.update(op.name.encode())
        feed(list(op.params))
        h.update(repr([qc.find_bit(q).index for q in inst.qubits]).encode())
        h.update(repr([qc.find_bit(c).index for c in inst.clbits]).encode())
        cond = getattr(op, 'condition', None)
        if cond is not None:
            target, value = cond
            h.update(repr((getattr(target, 'name', target), value)).encode())
        oper = getattr(op, 'operator', None)
        if oper is not None:
            h.update(repr(oper).encode())
            synth = getattr(op, 'synthesis', None)
            h.update(repr((type(synth).__name__, getattr(synth, 'reps', None),
                           getattr(synth, 'order', None))).encode())
    return h.hexdigest()


class RunPipelineMixin(HostState):
    """Running programs for QBasicTerminal: Aer backend options, memory-aware
    method selection, the RUN pipeline (circuit build, live fallback, shots,
    state capture, result display), SAMPLE, ESTIMATE and STEP."""

    _PRECISION_METHODS = ('automatic', 'statevector', 'density_matrix', 'unitary', 'superop')

    def _aer_options(self, method: str = 'statevector', *, include_noise: bool = True,
                     precision: str | None = None) -> dict:
        """AerSimulator options for one run: method, device, precision, noise,
        and the tuning knobs set through the memory map ($D008-$D00B)."""
        opts: dict = {'method': method}
        if self.sim_device == 'GPU':
            opts['device'] = 'GPU'
        prec = precision or self.sim_precision
        if prec == 'single' and method in self._PRECISION_METHODS:
            opts['precision'] = 'single'
        if include_noise:
            noise = self._noise_model
            if not noise and self._qubit_noise:
                noise = self._build_qubit_noise()
            if noise:
                opts['noise_model'] = noise
        for attr, key in (('_fusion_enable', 'fusion_enable'),
                          ('_mps_truncation', 'matrix_product_state_truncation_threshold'),
                          ('_sv_parallel_threshold', 'statevector_parallel_threshold'),
                          ('_es_approx_error', 'extended_stabilizer_approximation_error')):
            if getattr(self, attr, None) is not None:
                opts[key] = getattr(self, attr)
        return opts

    def _make_backend(self, method: str = 'statevector', *,
                      include_noise: bool = False,
                      precision: str | None = None) -> 'AerSimulator':
        """Build an AerSimulator respecting device, precision, tuning and
        (optionally) noise, so every execution path is configured alike."""
        return AerSimulator(**self._aer_options(method, include_noise=include_noise,
                                                precision=precision))

    def _build_qubit_noise(self):
        """Build a NoiseModel from per-qubit memory-mapped noise settings."""
        try:
            from qiskit_aer.noise import (
                NoiseModel, depolarizing_error, amplitude_damping_error,
                phase_damping_error,
            )
            nm = NoiseModel()
            _type_map = {1: depolarizing_error, 2: amplitude_damping_error,
                         3: phase_damping_error}
            _1q_gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'p', 'u',
                         'id', 's', 't', 'sdg', 'tdg', 'sx']
            _2q_gates = ['cx', 'cy', 'cz', 'ch', 'swap', 'dcx', 'iswap',
                         'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz']
            id1 = depolarizing_error(0.0, 1)  # identity channel for tensoring
            for q, (ntype, nparam) in self._qubit_noise.items():
                if ntype in _type_map and nparam > 0 and q < self.num_qubits:
                    if ntype == 1:
                        err = _type_map[ntype](nparam, 1)
                    else:
                        err = _type_map[ntype](nparam)
                    nm.add_quantum_error(err, _1q_gates, [q])
                    # Also fire the per-qubit error after any 2-qubit gate that
                    # touches q (identity on the partner qubit).
                    e_first = id1.tensor(err)   # err on qubit 0 of the pair
                    e_second = err.tensor(id1)  # err on qubit 1 of the pair
                    for j in range(self.num_qubits):
                        if j == q:
                            continue
                        nm.add_quantum_error(e_first, _2q_gates, [q, j])
                        nm.add_quantum_error(e_second, _2q_gates, [j, q])
            return nm
        except ImportError:
            return None

    @property
    def _transpile_opt_level(self) -> int:
        """Optimization level for transpilation.

        When a noise model is active, use level 0 so the transpiler does not
        collapse gate sequences (e.g. H-H -> identity) before noise channels
        can attach to them.  Without noise, use default (1) for performance.
        """
        if self._noise_model is not None:
            return 0
        if hasattr(self, '_qubit_noise') and self._qubit_noise:
            return 0
        return 1

    def _transpile_kwargs(self) -> dict:
        """coupling_map / basis_gates kwargs for transpile() (device model)."""
        kw: dict = {}
        if self._coupling_map:
            kw['coupling_map'] = self._coupling_map
        if self._basis_gates:
            kw['basis_gates'] = self._basis_gates
        return kw

    def _transpile_routed(self, qc, backend, level=None):
        """transpile() honoring the device model. Passing coupling_map or
        basis_gates alongside a backend is exactly how the offline device
        model works, so qiskit's advisory warning about it is silenced."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message=r'Providing `coupling_map`', category=UserWarning)
            return transpile(
                qc, backend,
                optimization_level=self._transpile_opt_level if level is None else level,
                **self._transpile_kwargs())

    def _run_kwargs(self) -> dict:
        """Build keyword args for backend.run() with optional seed."""
        kw: dict = {'shots': self.shots}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        return kw

    def _select_method(self, qc) -> tuple[str, str]:
        """Choose (method, precision) for a run.

        Under METHOD automatic: noiseless Clifford circuits use the stabilizer
        simulator at any width; otherwise the statevector family runs while it
        fits in available memory, at double precision when that fits and at
        single precision when only that fits; past that, matrix_product_state.
        Other methods run as set.
        """
        from qubasic_core import capacity as cap
        method, prec = self.sim_method, self.sim_precision
        if method != 'automatic':
            return method, prec
        noiseless = not self._noise_model and not self._qubit_noise
        if noiseless and self._is_clifford(qc):
            return 'stabilizer', prec
        n = self.num_qubits
        if cap.fits(cap.aer_peak_bytes(n, 'statevector', prec)):
            return 'automatic', prec
        if prec == 'double' and cap.fits(cap.aer_peak_bytes(n, 'statevector', 'single')):
            return 'statevector', 'single'
        return 'matrix_product_state', prec

    def _build_backend_opts(self, method: str, precision: str | None = None) -> dict:
        """AerSimulator options for a measured RUN (noise included)."""
        return self._aer_options(method, include_noise=True, precision=precision)

    def _check_fits(self, method: str, precision: str) -> None:
        """Refuse a width-bound run that cannot fit in available memory."""
        from qubasic_core import capacity as cap
        n = self.num_qubits
        peak = cap.aer_peak_bytes(n, 'statevector' if method == 'automatic' else method, precision)
        if cap.fits(peak):
            return
        hint = []
        if precision == 'double' and method in self._PRECISION_METHODS:
            hint.append("PRECISION SINGLE")
        if method in ('statevector', 'automatic'):
            hint.append("METHOD matrix_product_state")
        tail = f"; try {' or '.join(hint)}" if hint else ""
        raise QBasicRangeError(
            f"MEMORY: METHOD {method} at {n} qubits needs ~{cap.fmt_gb(peak)}, "
            f"{cap.fmt_gb(cap.available_bytes())} available{tail}")

    def _capture_mode(self, method: str, precision: str) -> str:
        """How a measured run keeps its final state for STATE, BLOCH and the
        other inspection commands: 'mid' saves it inside the run (no second
        simulation; needs a copy's worth of memory), 'lazy' recomputes it with
        a measure-free run on first use (1.04x the state), 'none' when even
        that cannot fit."""
        from qubasic_core import capacity as cap
        n = self.num_qubits
        if method in ('automatic', 'statevector') and not self._coupling_map:
            if cap.fits(cap.aer_peak_bytes(n, 'statevector', precision, 'mid')):
                return 'mid'
        if cap.fits(cap.aer_peak_bytes(n, 'statevector', precision, 'end')):
            return 'lazy'
        return 'none'

    def _defer_state_extraction(self, qc_sv, precision: str) -> None:
        """Leave a source that computes the measure-free circuit's state on
        first read of last_sv (one noisy trajectory when noise is active)."""
        seed = self._seed
        self._defer_state(lambda: self._extract_state_now(qc_sv, precision, seed))

    def _extract_state_now(self, qc_sv, precision: str, seed=None):
        """Simulate the measure-free circuit and return its final state, or
        None (with a note) when it cannot fit in available memory."""
        from qubasic_core import capacity as cap
        n = qc_sv.num_qubits
        peak = cap.aer_peak_bytes(n, 'statevector', precision, 'end')
        if not cap.fits(peak):
            hint = " (PRECISION SINGLE halves it)" if precision == 'double' else ""
            self.io.writeln(f"  (no statevector: {n} qubits needs ~{cap.fmt_gb(peak)}, "
                            f"{cap.fmt_gb(cap.available_bytes())} available{hint})")
            return None
        if peak is not None and peak > 1e9:
            self.io.writeln(f"  (computing the {n}-qubit state, ~{cap.fmt_gb(peak)})")
        try:
            q2 = qc_sv.copy()
            q2.save_statevector(label='qb_state')
            backend = self._make_backend('statevector', include_noise=True, precision=precision)
            kw: dict = {'shots': 1}
            if seed is not None:
                kw['seed_simulator'] = seed
            res = backend.run(transpile(q2, backend,
                                        optimization_level=self._transpile_opt_level),
                              **kw).result()
            return np.asarray(res.data()['qb_state'].data)
        except Exception as e:
            self.io.writeln(f"  (statevector unavailable: {e})")
            return None

    def _extract_save_results(self, result) -> None:
        """Extract SAVE_EXPECT/SAVE_PROBS/SAVE_AMPS into BASIC variables."""
        data = result.data()
        for key, val in data.items():
            if key.startswith('exp_'):
                self.variables[key[4:]] = float(np.real(val))
            elif key.startswith('prob_'):
                var = key[5:]
                self.variables[var] = val
                if isinstance(val, np.ndarray):
                    self.arrays[var] = val.tolist()
            elif key.startswith('amp_'):
                var = key[4:]
                if isinstance(val, np.ndarray):
                    self.arrays[var] = [complex(v) for v in val]
                self.variables[var] = val

    def _density_from_qc(self, qc_sv):
        """Solve the density matrix of a measure-free circuit, for DENSITY.

        Computed on demand (not on every run) from the circuit captured when a
        SET_DENSITY state was prepared.
        """
        try:
            q2 = qc_sv.copy()
            q2.save_density_matrix()
            dm_backend = self._make_backend('density_matrix', include_noise=True)
            _kw = {}
            if self._seed is not None:
                _kw['seed_simulator'] = self._seed
            res = dm_backend.run(
                transpile(q2, dm_backend, optimization_level=self._transpile_opt_level),
                **_kw).result()
            data = res.data(0)
            dm = data.get('density_matrix') if hasattr(data, 'get') else None
            return np.array(dm) if dm is not None else None
        except Exception:
            return None

    def _finalize_run(self, qc, method: str, t0: float, precision: str | None = None) -> None:
        """Update status, manifest, metrics after a successful run."""
        dt = time.time() - t0
        depth = qc.depth()
        n_gates = qc.size()
        self._update_status(gate_count=n_gates, circuit_depth=depth,
                           run_time_ms=dt * 1000)
        self.variables['_DEPTH'] = depth
        self.variables['_GATES'] = n_gates
        self.variables['_TIME'] = dt
        self._run_manifest = {
            'program': dict(self.program),
            'num_qubits': self.num_qubits,
            'shots': self.shots,
            'method': method,
            'precision': precision or self.sim_precision,
            'device': self.sim_device,
            'seed': self._seed,
            'noise_depol_p': self._noise_depol_p,
            'depth': depth,
            'gates': n_gates,
            'time_s': dt,
        }

    def _run_no_measure(self, qc, qc_sv, t0: float, precision: str | None = None) -> None:
        """Execute the no-MEASURE path: one measure-free statevector run that
        keeps the final state (1.04x the state; Aer hands the buffer over), or
        the lazily solved density matrix for a SET_DENSITY state."""
        from qubasic_core import capacity as cap
        prec = precision or self.sim_precision
        n = self.num_qubits
        peak = cap.aer_peak_bytes(n, 'statevector', prec, 'end')
        fits = cap.fits(peak)
        if self._pending_set_density is not None or _has_channel(qc_sv):
            # A mixed state has no statevector: DENSITY solves it on demand
            # from the measure-free circuit.
            self.last_sv = None
            self._last_density = None
            self._last_density_qc = qc_sv
        elif qc_sv.size() == 0:
            # Nothing to simulate: the state is |0...0>, built when first read.
            dtype = np.complex64 if prec == 'single' else np.complex128

            def _zero_state(_n=n, _dtype=dtype):
                sv = np.zeros(1 << _n, dtype=_dtype)
                sv[0] = 1.0
                return sv
            self._defer_state(_zero_state)
        elif not fits:
            self.last_sv = None
        else:
            try:
                q2 = qc_sv.copy()
                q2.save_statevector(label='qb_state')
                sv_backend = self._make_backend('statevector', include_noise=True, precision=prec)
                kw: dict = {'shots': 1}
                if self._seed is not None:
                    kw['seed_simulator'] = self._seed
                sv_result = sv_backend.run(
                    transpile(q2, sv_backend, optimization_level=self._transpile_opt_level),
                    **kw).result()
                self.last_sv = np.asarray(sv_result.data()['qb_state'].data)
                self._extract_save_results(sv_result)
            except (RuntimeError, ValueError, TypeError, KeyError):
                self.last_sv = None
        self.last_counts = None
        self._finalize_run(qc, 'statevector', t0, prec)
        depth = qc.depth()
        n_gates = qc.size()
        if n_gates == 0 and self._pending_set_density is None:
            return      # a classical program: its own output is the result
        dt = time.time() - t0
        tag = ", precision=single" if prec == 'single' else ""
        self.io.writeln(f"\nRAN {len(self.program)} lines, {n} qubits "
                        f"in {dt:.2f}s  [depth={depth}, gates={n_gates}{tag}]")
        if self._last_density_qc is not None:
            self.io.writeln("(no MEASURE; the state is mixed — use DENSITY to inspect)")
        elif not fits:
            hint = "; PRECISION SINGLE halves it" if prec == 'double' else ""
            self.io.writeln(f"(no MEASURE; a {n}-qubit statevector needs ~{cap.fmt_gb(peak)}, "
                            f"{cap.fmt_gb(cap.available_bytes())} available{hint})")
        else:
            self.io.writeln("(no MEASURE — use STATE, PROBS, or BLOCH to inspect)")

    def _add_final_measurement(self, qc) -> None:
        """Append the program's final measurement: its MEASURE subset into a
        'sub' register, or every qubit."""
        subset = self._measure_subset
        if subset:
            from qiskit.circuit import ClassicalRegister
            cr = ClassicalRegister(len(subset), 'sub')
            qc.add_register(cr)
            for i, q in enumerate(subset):
                qc.measure(q, cr[i])
        else:
            qc.measure_all()

    def _keep_state(self, result, qc_sv, method: str, precision: str, capture: str) -> None:
        """Keep the final state of a measured run: from the run itself when it
        saved one, otherwise as a deferred extraction from the measure-free
        circuit (computed on first use)."""
        if method in ('unitary', 'superop'):
            self.last_sv = None
            return
        if self._pending_set_density is not None or _has_channel(qc_sv):
            # Mixed state: DENSITY solves the density matrix on demand.
            self.last_sv = None
            self._last_density = None
            self._last_density_qc = qc_sv
            return
        if capture == 'mid':
            try:
                data = result.data()
                if 'qb_state' in data:
                    self.last_sv = np.asarray(data['qb_state'].data)
                    return
            except Exception:
                pass
        self._defer_state_extraction(qc_sv, precision)

    def _run_with_fallback(self, qc, backend_opts: dict, method: str,
                           capture: str = 'none') -> 'Any':
        """Run the circuit with shots, handling GPU and stabilizer fallbacks."""
        # str(NoiseModel) does not vary with channel parameters (amplitude_damping
        # 0.1 and 0.9 stringify identically), so fingerprint noise by the exact
        # NOISE command and the per-qubit memory-mapped noise instead — otherwise
        # changing noise strength between runs reuses the first circuit.
        _noise_key = (
            self._noise_spec,
            id(self._noise_model) if self._noise_model is not None else None,
            tuple(sorted(self._qubit_noise.items())),
        )
        # The built circuit itself is the key: every input that shapes it
        # (program text, DEF/UNITARY/HAMILTONIAN/CHANNEL definitions, variable
        # and array values, SET_STATE, POKE state prep) lands in its
        # instructions, so a fingerprint of the instructions cannot go stale.
        _dev_key = (
            tuple(map(tuple, self._coupling_map)) if self._coupling_map else None,
            tuple(self._basis_gates) if self._basis_gates else None,
        )
        cache_key = (
            _circuit_fingerprint(qc),
            method, self.sim_device,
            backend_opts.get('precision', 'double'), capture,
            _noise_key, _dev_key,
            getattr(self, '_fusion_enable', None),
            getattr(self, '_mps_truncation', None),
            getattr(self, '_sv_parallel_threshold', None),
            getattr(self, '_es_approx_error', None),
        )
        if self._circuit_cache_key == cache_key and self._circuit_cache is not None:
            qc_t, backend = self._circuit_cache
        else:
            backend = AerSimulator(**backend_opts)
            qc_t = self._transpile_routed(qc, backend)
            self._circuit_cache_key = cache_key
            self._circuit_cache = (qc_t, backend)
            self._last_transpiled = qc_t
        try:
            return backend.run(qc_t, **self._run_kwargs()).result()
        except KeyboardInterrupt:
            self.io.writeln("\n?INTERRUPTED")
            raise
        except Exception as _sim_err:
            _err_msg = str(_sim_err).lower()
            if 'gpu' in _err_msg and 'not supported' in _err_msg:
                self.io.writeln("?GPU EXECUTION FAILED — falling back to CPU")
                self.sim_device = 'CPU'
                self._circuit_cache_key = None
                self._circuit_cache = None
                backend_opts.pop('device', None)
                backend = AerSimulator(**backend_opts)
                qc_t = self._transpile_routed(qc, backend)
                return backend.run(qc_t, **self._run_kwargs()).result()
            elif 'stabilizer' in _err_msg or 'invalid parameters' in _err_msg:
                self._circuit_cache_key = None
                self._circuit_cache = None
                sv_opts = {k: v for k, v in backend_opts.items() if k != 'method'}
                sv_opts['method'] = 'statevector'
                self.io.writeln("  (stabilizer failed — falling back to statevector)")
                backend = AerSimulator(**sv_opts)
                qc_t = self._transpile_routed(qc, backend)
                return backend.run(qc_t, **self._run_kwargs()).result()
            raise

    def cmd_run(self) -> None:
        """RUN — execute the stored program: build and simulate its circuit, or run
        it live (shot by shot) when it reads measurement results, or on the LOCC
        or encoded engines in those modes, then show the histogram."""
        if self.locc_mode:
            return self._locc_run()
        if not self.program:
            self.io.writeln("NOTHING TO RUN")
            return

        _unsupported_gpu = {'stabilizer', 'extended_stabilizer', 'unitary', 'superop'}
        if self.sim_device == 'GPU' and self.sim_method in _unsupported_gpu:
            self.io.writeln(f"?METHOD '{self.sim_method}' does not support GPU — "
                           f"use METHOD statevector or density_matrix")
            return

        t0 = time.time()
        self._gosub_stack = []
        self._collect_data()
        sorted_lines = sorted(self.program.keys())
        # Compiling checks block structure and GOTO/GOSUB targets up front.
        self._scan_subs(self._compiled())

        # LQUBITS runs a program of transversal gates encoded (encoded.py).
        if self._logical_mode and self._logical_mode.get('encoded'):
            if self._run_encoded(t0):
                return

        # A program that reads measurement results as numbers runs live.
        self._live_needed = False
        if self._needs_live(sorted_lines):
            return self._run_live(t0, sorted_lines)

        # Build circuit; its output is held so a build that turns out to need
        # a live run prints nothing twice.
        from qubasic_core.live import BufferedIO
        held = BufferedIO(self.io)
        real_io, self.io = self.io, held
        try:
            qc, has_measure = self.build_circuit()
        except KeyboardInterrupt:
            self.io = real_io
            held.flush()
            self.io.writeln("\n?INTERRUPTED")
            return
        except Exception as e:
            self.io = real_io
            if self._live_needed:
                held.discard()
                return self._run_live(t0, sorted_lines, carry=held)
            held.flush()
            self.last_circuit = None
            # ON ERROR traps errors inside the build loop (ExecutorMixin
            # ._build_lines); an error that reaches here is untrapped.
            self.io.writeln(f"?BUILD ERROR: {e}")
            return
        finally:
            self.io = real_io
        if self._live_needed:
            held.discard()
            return self._run_live(t0, sorted_lines, carry=held)
        held.flush()

        qc_sv = qc.copy()
        method, precision = self._select_method(qc_sv)
        try:
            self._check_fits(method, precision)
        except QBasicRangeError as e:
            self.io.writeln(f"?{e.message}")
            return

        capture = 'none'
        run_qc = qc
        if self.sim_method in ('unitary', 'superop'):
            qc.save_unitary(label='unitary') if self.sim_method == 'unitary' else qc.save_superop(label='superop')
        elif has_measure:
            capture = self._capture_mode(method, precision)
            if capture == 'mid':
                run_qc = qc.copy()
                run_qc.save_statevector(label='qb_state')
                self._add_final_measurement(run_qc)
            self._add_final_measurement(qc)

        self.last_circuit = qc
        self._last_transpiled = None

        # No-MEASURE path
        if not has_measure and self.sim_method not in ('unitary', 'superop'):
            self._run_no_measure(qc, qc_sv, t0, precision)
            return

        # Run with shots
        backend_opts = self._build_backend_opts(method, precision)
        try:
            result = self._run_with_fallback(run_qc, backend_opts, method, capture)
        except KeyboardInterrupt:
            return
        except Exception as e:
            _err = str(e).lower()
            if 'gpu' in _err:
                subsys = 'device/GPU'
            elif 'noise' in _err or 'kraus' in _err:
                subsys = 'noise'
            elif 'stabilizer' in _err:
                subsys = 'backend/stabilizer'
            elif 'mps' in _err or 'matrix_product' in _err:
                subsys = 'backend/MPS'
            else:
                subsys = f'backend/{method}'
            self.io.writeln(f"?RUNTIME ERROR [{subsys}]: {e}")
            return

        # Extract results
        if method in ('unitary', 'superop'):
            self.last_counts = None
            data = result.data()
            label = method
            mat = data.get(label)
            if mat is not None:
                mat_np = np.asarray(mat)
                self.variables[label] = mat_np
                dim = mat_np.shape[0]
                self.io.writeln(f"\n  {label.upper()} ({dim}x{dim}):")
                if dim <= 16:
                    for i in range(dim):
                        row = '  '.join(f"{v.real:+.3f}{v.imag:+.3f}j" for v in mat_np[i])
                        self.io.writeln(f"    {row}")
                else:
                    self.io.writeln(f"    (too large — stored in variable '{label}')")
        else:
            try:
                self.last_counts = dict(result.get_counts())
            except Exception:
                self.io.writeln(f"  (method '{method}' produced no counts — falling back)")
                self._circuit_cache_key = None
                sv_opts = {k: v for k, v in backend_opts.items() if k != 'method'}
                sv_opts['method'] = 'statevector'
                sv_backend = AerSimulator(**sv_opts)
                sv_qc = transpile(qc, sv_backend, optimization_level=self._transpile_opt_level)
                result = sv_backend.run(sv_qc, **self._run_kwargs()).result()
                self.last_counts = dict(result.get_counts())
            # Dynamic-circuit runs carry extra mid-circuit MEAS registers, so
            # get_counts returns space-separated per-register keys. The final
            # MEASURE register is added last (leftmost token); keep only it so
            # the reported histogram is the actual end-of-circuit outcome.
            if self._classical_bits and self.last_counts and \
                    any(' ' in k for k in self.last_counts):
                collapsed: dict[str, int] = {}
                for key, cnt in self.last_counts.items():
                    outcome = key.split()[0]
                    collapsed[outcome] = collapsed.get(outcome, 0) + cnt
                self.last_counts = collapsed

        self._extract_save_results(result)
        self._keep_state(result, qc_sv, method, precision, capture)
        self._finalize_run(qc, method, t0, precision)

        # Display results with execution metadata
        m = self._run_manifest
        _meta_parts = [f"method={m['method']}"]
        if m.get('precision') == 'single':
            _meta_parts.append("precision=single")
        if m['device'] != 'CPU':
            _meta_parts.append(f"device={m['device']}")
        if self._noise_model is not None:
            _noise_tag = f"noise=depol({m['noise_depol_p']})" if m['noise_depol_p'] > 0 else "noise=on"
            _meta_parts.append(_noise_tag)
        _meta = ', '.join(_meta_parts)
        _throughput = f", {m['gates']/m['time_s']:.0f} gates/s" if m['time_s'] > 0.001 and m['gates'] > 0 else ""
        _complexity = self._circuit_complexity()
        _cx_str = f", {_complexity}" if _complexity else ""
        self.io.writeln(f"\nRAN {len(self.program)} lines, {self.num_qubits} qubits, "
                        f"{self.shots} shots in {m['time_s']:.2f}s  "
                        f"[depth={m['depth']}, gates={m['gates']}{_throughput}, {_meta}{_cx_str}]")
        if (self._coupling_map or self._basis_gates) and getattr(self, '_last_transpiled', None) is not None:
            qt = self._last_transpiled
            swaps = qt.count_ops().get('swap', 0)
            _basis = f", basis={'+'.join(self._basis_gates)}" if self._basis_gates else ""
            _topo = f", swaps={swaps}" if self._coupling_map else ""
            self.io.writeln(f"  routed onto device: depth={qt.depth()}, gates={qt.size()}{_topo}{_basis}")
        if getattr(self, '_logical_mode', None):
            self._logical_trailer()
        if method in ('unitary', 'superop'):
            pass  # matrix already displayed above
        elif has_measure and self.last_counts:
            self.print_histogram(self.last_counts)
            self._auto_display()
        else:
            self.io.writeln("(no MEASURE in program \u2014 use STATE or PROBS to inspect)")
        # Sound on completion for long runs (#25)
        if m['time_s'] > 2.0 and sys.stdout.isatty():
            self.io.write('\a')

    def result(self) -> dict:
        """Structured result of the last run, for headless/agent/JSON callers.

        JSON-serializable: counts, qubit/shot config, user variables, the
        statevector (when small enough), and key run-manifest fields.
        """
        _big = getattr(self, '_endian_big', False)
        _counts = self.last_counts or {}
        if _big and _counts:
            _counts = {k[::-1]: v for k, v in _counts.items()}
        out: dict = {
            'counts': _counts,
            'num_qubits': self.num_qubits,
            # Bitstring keys follow the active OPTION ENDIAN convention; the
            # bit_order field is the self-describing record of which one.
            'shots': self.shots,
            'bit_order': ('big-endian (qubit 0 = leftmost bit)' if _big
                          else 'little-endian (qubit 0 = rightmost bit)'),
        }
        uvars = {k: v for k, v in self.variables.items()
                 if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
        if uvars:
            out['variables'] = uvars
        # Only small states go into JSON; reading a deferred state that would
        # be dropped anyway would cost a simulation for nothing.
        sv = self.last_sv if self.num_qubits <= 8 else None
        if sv is not None:
            svf = np.ascontiguousarray(sv).ravel()
            if svf.size <= 256:  # keep JSON output bounded
                out['statevector'] = [[float(a.real), float(a.imag)] for a in svf]
        manifest = getattr(self, '_run_manifest', None)
        if manifest:
            out['method'] = manifest.get('method')
            out['depth'] = manifest.get('depth')
            out['gates'] = manifest.get('gates')
        return out

    def cmd_sample(self, rest: str = '') -> None:
        """SAMPLE [shots] — sample the program through Aer's SamplerV2 primitive:
        the program's MEASURE subset (or every qubit), with the active noise
        model, precision and SEED."""
        if not self.program:
            self.io.writeln("NOTHING TO SAMPLE")
            return
        shots = int(rest.strip()) if rest.strip() else self.shots
        try:
            qc, _ = self.build_circuit()
            method, precision = self._select_method(qc)
            self._add_final_measurement(qc)
            opts = self._aer_options(method, include_noise=True, precision=precision)
            qc_t = self._transpile_routed(qc, AerSimulator(**opts))
            from qiskit_aer.primitives import SamplerV2
            sampler = SamplerV2(default_shots=shots, seed=self._seed,
                                options={'backend_options': opts})
            pub_result = sampler.run([qc_t]).result()[0]
            register = 'sub' if self._measure_subset else 'meas'
            counts = dict(getattr(pub_result.data, register).get_counts())
            self.last_counts = counts
            self.io.writeln(f"SAMPLED {shots} shots ({len(counts)} unique outcomes)")
            if counts:
                self.print_histogram(counts)
        except Exception as e:
            self.io.writeln(f"?SAMPLE ERROR: {e}")

    def cmd_estimate(self, rest: str) -> None:
        """ESTIMATE <pauli> [qubits] — estimate observable expectation via EstimatorV2."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: ESTIMATE <Z|ZZ|XY|...> [qubits]")
            return
        pauli_str = parts[0].upper()
        qubits = [int(q) for q in parts[1:]] if len(parts) > 1 else list(range(len(pauli_str)))
        try:
            from qiskit.quantum_info import SparsePauliOp
            from qiskit_aer.primitives import EstimatorV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            qc, _ = self.build_circuit()
            full_pauli = ['I'] * self.num_qubits
            for i, p in enumerate(pauli_str):
                if i < len(qubits):
                    full_pauli[self.num_qubits - 1 - qubits[i]] = p
            op = SparsePauliOp(''.join(full_pauli))
            estimator = EstimatorV2()
            pm = generate_preset_pass_manager(optimization_level=0)
            qc_t = pm.run(qc)
            result = estimator.run([(qc_t, op)]).result()
            val = result[0].data.evs
            self.io.writeln(f"  <{pauli_str}> on qubits {qubits} = {float(val):.6f}")
        except Exception as e:
            self.io.writeln(f"?ESTIMATE ERROR: {e}")

    def cmd_step(self) -> None:
        """STEP — step through the program a line at a time, showing the state after each.

        It runs as one live shot: MEAS collapses the state and IF, PRINT and
        LET see the outcome."""
        if not self.program:
            self.io.writeln("NOTHING TO STEP")
            return

        from qubasic_core.exec_context import ExecContext
        from qubasic_core.scope import Scope
        from qubasic_core.live import LiveState, LiveBackend
        from qubasic_core import capacity as cap
        peak = cap.numpy_peak_bytes(self.num_qubits, self.sim_precision)
        if not cap.fits(peak):
            self.io.writeln(f"?STEP: a {self.num_qubits}-qubit state needs ~{cap.fmt_gb(peak)}, "
                            f"{cap.fmt_gb(cap.available_bytes())} available")
            return
        prog = self._compiled()
        self._gosub_stack = []
        self._collect_data()
        self._scan_subs(prog)
        state = LiveState(self.num_qubits, self.sim_precision, self._live_noise())
        self._prepare_run(state.qc)
        ctx = ExecContext(
            sorted_lines=prog, ip=0,
            run_vars=Scope(self.variables),
            max_iterations=self._max_iterations, qc=state.qc,
            backend=LiveBackend(state, self._custom_gates),
        )
        self._live = state
        prev_strict, self._strict_names = self._strict_names, False
        try:
            self._step_lines(ctx, state)
        finally:
            self._live = None
            self._strict_names = prev_strict
            self._step_auto = False

    def _step_lines(self, ctx, state) -> None:
        prog = sorted_lines = ctx.sorted_lines

        self.io.writeln(f"STEP MODE \u2014 {len(self.program)} lines, {self.num_qubits} qubits")
        self.io.writeln("Press ENTER to advance, A for auto-play, Q to quit\n")

        while ctx.ip < len(sorted_lines):
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            line_num = prog.lines[ctx.ip]
            stmt = prog.texts[ctx.ip]
            parsed = prog.stmts[ctx.ip]

            # Display current statement
            self.io.writeln(f">> {line_num:5d}  {stmt}")

            # Execute it, then show the state it leaves.
            try:
                result = self._exec_line(stmt, parsed=parsed, ctx=ctx)
                sv = state.sv
            except Exception as e:
                self.io.writeln(f"?LINE {line_num}: {e}")
                return
            self.last_sv = sv
            self._checkpoint_sv(line_num)
            self._print_sv_compact(sv)

            # Wait for input (or auto-advance)
            if not getattr(self, '_step_auto', False):
                try:
                    user = self.io.read_line("   [ENTER/A/Q] ").strip().upper()
                    if user == 'Q':
                        self.io.writeln("STOPPED")
                        return
                    if user == 'A' or user.startswith('A'):
                        self._step_auto = True
                        delay = 0.5
                        if len(user) > 1:
                            try:
                                delay = float(user[1:]) / 1000
                            except ValueError:
                                pass
                        self._step_delay = delay
                except (KeyboardInterrupt, EOFError):
                    self.io.writeln("\nSTOPPED")
                    return
            if getattr(self, '_step_auto', False):
                import time as _time
                _time.sleep(getattr(self, '_step_delay', 0.5))

            if result is ExecResult.END:
                break
            self._advance(ctx, result)

        self.io.writeln("DONE")

    def _is_clifford(self, qc: 'QuantumCircuit') -> bool:
        """Check if circuit contains only Clifford gates.

        Custom/unitary gates are conservatively treated as non-Clifford.
        Verifying that an arbitrary unitary normalizes the Pauli group is
        expensive, and the consequence of a false positive (stabilizer sim
        producing wrong results) is worse than a false negative (falling
        back to statevector sim, which always works).
        """
        clifford_gates = {'h', 'x', 'y', 'z', 's', 'sdg', 'cx', 'cz', 'cy',
                          'swap', 'id', 'sx', 'barrier', 'measure', 'reset'}
        for inst in qc.data:
            name = inst.operation.name.lower()
            if name == 'unitary':
                return False  # custom gates: conservatively non-Clifford
            if name not in clifford_gates:
                return False
        return True
