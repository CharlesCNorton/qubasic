"""QUBASIC runs: Aer options and method selection, the RUN pipeline,
the circuit builder, and live (shot-by-shot) execution for programs that read
measurement results.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import MutableMapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qubasic_core.engine import ExecResult, QBasicBuildError, QBasicRangeError
from qubasic_core.host import HostState
from qubasic_core.parser import (
    GATE_ALIASES,
    RE_AT_REG_LINE,
    RE_CTRL,
    RE_INV,
    RE_REG_INDEX,
    RE_SYNDROME,
    CompiledProgram,
)
from qubasic_core.simtools import _GATE_BUILDERS, _apply_gate_inplace, _measure_inplace, _np_gate_matrix, _sample_counts

if TYPE_CHECKING:
    from collections.abc import Container, Iterable

    from qiskit import ClassicalRegister
    from qiskit.circuit import Gate
    from qiskit.result import Result
    from qiskit_aer.noise import NoiseModel

    from qubasic_core.engine import CircuitBackend, ExecContext, ExecOutcome
    from qubasic_core.parser import IfJumpStmt, Stmt


# ---------------------------------------------------------------------------
# QUBASIC run pipeline: backend options, method selection, RUN, SAMPLE, ESTIMATE, STEP.
# ---------------------------------------------------------------------------

def _has_channel(qc: QuantumCircuit) -> bool:
    """Whether a circuit applies a non-unitary channel (APPLYCHANNEL)."""
    return any(inst.operation.name in ('kraus', 'superop', 'quantum_channel')
               for inst in qc.data)


def _expectation(val: Any) -> float:
    """An expectation value as a float, with rounding residue under 1e-12 read as 0."""
    v = float(np.real(val))
    return 0.0 if abs(v) < 1e-12 else v


def _circuit_fingerprint(qc: QuantumCircuit) -> str:
    """Stable digest of a circuit's registers and instructions.

    Covers operation names, parameters (arrays by content), qubit and clbit
    indices, nested control-flow blocks, if_test conditions, and the operator
    and step count of Hamiltonian evolution gates.
    """
    import hashlib

    from qiskit import QuantumCircuit as _QC
    h = hashlib.blake2b(digest_size=16)

    def feed(obj: Any) -> None:
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

    def _build_qubit_noise(self) -> NoiseModel | None:
        """Build a NoiseModel from per-qubit memory-mapped noise settings."""
        try:
            from qiskit_aer.noise import (
                NoiseModel,
                amplitude_damping_error,
                depolarizing_error,
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

    def _transpile_routed(self, qc: QuantumCircuit, backend: AerSimulator, level: int | None = None) -> QuantumCircuit:
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

    def _select_method(self, qc: QuantumCircuit) -> tuple[str, str]:
        """Choose (method, precision) for a run.

        Under METHOD automatic: noiseless Clifford circuits use the stabilizer
        simulator at any width; otherwise the statevector family runs while it
        fits in available memory, at double precision when that fits and at
        single precision when only that fits; past that, matrix_product_state.
        Other methods run as set.
        """
        from qubasic_core import simtools as cap
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
        from qubasic_core import simtools as cap
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
        from qubasic_core import simtools as cap
        n = self.num_qubits
        if method in ('automatic', 'statevector') and not self._coupling_map:
            if cap.fits(cap.aer_peak_bytes(n, 'statevector', precision, 'mid')):
                return 'mid'
        if cap.fits(cap.aer_peak_bytes(n, 'statevector', precision, 'end')):
            return 'lazy'
        return 'none'

    def _defer_state_extraction(self, qc_sv: QuantumCircuit, precision: str) -> None:
        """Leave a source that computes the measure-free circuit's state on
        first read of last_sv (one noisy trajectory when noise is active)."""
        seed = self._seed
        self._defer_state(lambda: self._extract_state_now(qc_sv, precision, seed))

    def _extract_state_now(self, qc_sv: QuantumCircuit, precision: str, seed: int | None = None) -> np.ndarray | None:
        """Simulate the measure-free circuit and return its final state, or
        None (with a note) when it cannot fit in available memory."""
        from qubasic_core import simtools as cap
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

    def _extract_save_results(self, result: Result) -> None:
        """Extract SAVE_EXPECT/SAVE_PROBS/SAVE_AMPS into BASIC variables."""
        data = result.data()
        for key, val in data.items():
            if key.startswith('exp_'):
                self.variables[key[4:]] = _expectation(val)
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

    def _density_from_qc(self, qc_sv: QuantumCircuit) -> np.ndarray | None:
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

    def _finalize_run(self, qc: QuantumCircuit, method: str, t0: float, precision: str | None = None) -> None:
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

    def _run_no_measure(self, qc: QuantumCircuit, qc_sv: QuantumCircuit, t0: float, precision: str | None = None) -> None:
        """Execute the no-MEASURE path: one measure-free statevector run that
        keeps the final state (1.04x the state; Aer hands the buffer over), or
        the lazily solved density matrix for a SET_DENSITY state."""
        from qubasic_core import simtools as cap
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
        elif not qc_sv.data:
            # Nothing to simulate (size() would also skip SET_STATE and SAVE_*
            # directives): the state is |0...0>, built when first read.
            dtype = np.complex64 if prec == 'single' else np.complex128

            def _zero_state(_n: int = n, _dtype: Any = dtype) -> np.ndarray:
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
        if not qc.data:
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

    def _add_final_measurement(self, qc: QuantumCircuit) -> None:
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

    def _keep_state(self, result: Result, qc_sv: QuantumCircuit, method: str, precision: str, capture: str) -> None:
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

    def _run_with_fallback(self, qc: QuantumCircuit, backend_opts: dict, method: str,
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
        or encoded engines in those modes, then show the histogram (and any
        picture the program drew on a graphics SCREEN)."""
        self._gfx_begin_run()
        self._print_col = 0
        self._sound_events.clear()
        self._play_state = type(self._play_state)()
        try:
            self._run_program()
        finally:
            for handle in list(self._file_handles):     # a program's end closes its files
                self._close_file(handle)
            self._gfx_present()
            if self._sound_events and self._on_terminal():
                self.io.writeln(f"({self._sound_seconds():.2f} s of sound: SAVEWAV <file> saves it)")

    def _run_program(self) -> None:
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
        assert m is not None     # _finalize_run records it
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
        qt = getattr(self, '_last_transpiled', None)
        if (self._coupling_map or self._basis_gates) and qt is not None:
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
            self._mitigate_after_run()
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
        """ESTIMATE <pauli> [qubits] — expectation value through Aer's EstimatorV2 primitive, with the active noise model, precision and SEED."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: ESTIMATE <Z|ZZ|XY|...> [qubits]")
            return
        pauli_str = parts[0].upper()
        qubits = [int(q) for q in parts[1:]] if len(parts) > 1 else list(range(len(pauli_str)))
        try:
            from qiskit.quantum_info import SparsePauliOp
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit_aer.primitives import EstimatorV2
            qc, _ = self.build_circuit()
            full_pauli = ['I'] * self.num_qubits
            for i, p in enumerate(pauli_str):
                if i < len(qubits):
                    full_pauli[self.num_qubits - 1 - qubits[i]] = p
            op = SparsePauliOp(''.join(full_pauli))
            method, precision = self._select_method(qc)
            options: dict[str, Any] = {
                'backend_options': self._aer_options(method, include_noise=True, precision=precision)}
            if self._seed is not None:
                options['run_options'] = {'seed': self._seed}
            estimator = EstimatorV2(options=options)
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

        from qubasic_core import simtools as cap
        from qubasic_core.engine import ExecContext
        from qubasic_core.expression import Scope
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

    def _step_lines(self, ctx: ExecContext, state: LiveState) -> None:
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


# ---------------------------------------------------------------------------
# QUBASIC executor mixin — circuit building and line execution.
# ---------------------------------------------------------------------------

def _feedforward_test(condition: str, names: Container[str]) -> tuple[str, int] | None:
    """(bit name, value it must read) when an IF condition tests one
    measurement bit in a form a dynamic circuit can branch on, else None."""
    s = condition.strip()
    m = re.match(r'NOT\s+(\w+)$', s, re.IGNORECASE)
    if m and m.group(1) in names:
        return m.group(1), 0
    if re.match(r'\w+$', s) and s in names:
        return s, 1
    m = re.match(r'(\w+)\s*(==|=|!=|<>)\s*([01])$', s)
    if m and m.group(1) in names:
        val = int(m.group(3))
        return m.group(1), (val if m.group(2) in ('==', '=') else 1 - val)
    return None


class ExecutorMixin(HostState):
    """Circuit building and per-line execution for QBasicTerminal.

    Provides the core execution pipeline: circuit compilation from stored
    programs, single-line execution, gate tokenization, qubit resolution,
    subroutine expansion, and immediate-mode gate application.

    All methods access shared terminal state (variables, arrays, subroutines,
    registers, etc.) through ``self``.
    """

    # Gate dispatch: name -> (method_name, arg_pattern)
    # arg_pattern: 'q' = qubits only, 'pq' = params then qubits, 'ppq' = 3-param then qubit
    _GATE_DISPATCH = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'T': 't', 'SDG': 'sdg', 'TDG': 'tdg',
        'SX': 'sx', 'ID': 'id',
        'CX': 'cx', 'CZ': 'cz', 'CY': 'cy', 'CH': 'ch',
        'SWAP': 'swap', 'DCX': 'dcx', 'ISWAP': 'iswap',
        'CCX': 'ccx', 'CSWAP': 'cswap',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz', 'P': 'p',
        'CRX': 'crx', 'CRY': 'cry', 'CRZ': 'crz', 'CP': 'cp',
        'RXX': 'rxx', 'RYY': 'ryy', 'RZZ': 'rzz',
        'U': 'u',
    }

    # ── Circuit Building ──────────────────────────────────────────────

    def _program_has_measure(self, sorted_lines: Iterable[int]) -> bool:
        """True if any reachable statement measures.

        The plain per-line scan misses a MEASURE that only appears inside a DEF
        subroutine body or an IF/THEN-ELSE clause, which left the run on the
        no-measure path with no counts. This recursion-guarded walk follows
        colon compounds, IF clauses, and subroutine bodies so the shots path
        fires whenever a measurement is actually reachable.
        """
        from qubasic_core.parser import CompoundStmt, IfThenStmt, MeasureStmt, parse_stmt

        def scan(text: str, seen: frozenset) -> bool:
            p = parse_stmt(text)
            if isinstance(p, MeasureStmt):
                return True
            if isinstance(p, CompoundStmt):
                return any(scan(part, seen) for part in p.parts)
            if isinstance(p, IfThenStmt):
                if p.then_clause and scan(p.then_clause, seen):
                    return True
                if p.else_clause and scan(p.else_clause, seen):
                    return True
                return False
            words = text.strip().split()
            if words:
                word = words[0].upper().split('(')[0]
                if word in self.subroutines and word not in seen:
                    sub = self.subroutines[word]
                    body = sub['body'] if isinstance(sub, dict) else sub
                    return any(scan(b, seen | {word}) for b in body)
            return False

        return any(scan(self.program[ln], frozenset()) for ln in sorted_lines)

    def _prepare_run(self, qc: QuantumCircuit) -> None:
        """Reset per-run state and emit pending state preparation into qc."""
        # Fresh mid-circuit measurement registry for this build (dynamic IF).
        self._classical_bits: dict[str, Any] = {}
        # Partial-measurement subset (None = measure all at the end).
        self._measure_subset: list[int] | None = None
        # Density matrix (and circuit) captured by a SET_DENSITY run; the
        # matrix is solved lazily by DENSITY from the stored circuit.
        self._last_density = None
        self._last_density_qc = None
        # Apply any qubit state preparation requested via POKE to $0100.
        if getattr(self, '_poke_state_prep', None):
            self._emit_poke_state_prep(qc)
        # Apply a pending immediate SET_STATE so it persists into this RUN.
        _pend = getattr(self, '_pending_set_state', None)
        if _pend is not None and len(_pend) == 2 ** self.num_qubits:
            from qiskit.quantum_info import Statevector
            from qiskit_aer.library import SetStatevector
            qc.append(SetStatevector(Statevector(_pend)), list(range(self.num_qubits)))
        # Inject a pending mixed state (density matrix) — needs density_matrix method.
        _pend_d = getattr(self, '_pending_set_density', None)
        if _pend_d is not None and _pend_d.shape[0] == 2 ** self.num_qubits:
            from qiskit.quantum_info import DensityMatrix
            from qiskit_aer.library import SetDensityMatrix
            qc.append(SetDensityMatrix(DensityMatrix(_pend_d)), list(range(self.num_qubits)))
        self._on_measure_fired = False
        # Error trapping arms when this run executes ON ERROR GOTO, never from
        # a previous run's state.
        self._error_target = None
        self._in_error_handler = False
        # Declarations (DIM AS, CONST, DIM SHARED) start over; DEFtype ranges
        # apply to the whole program.
        self._reset_declarations(self._compiled())

    def build_circuit(self, backend: CircuitBackend | None = None, *, start_ip: int = 0,
                      stop_ip: int | None = None) -> tuple['QuantumCircuit', bool]:
        """Compile program lines into a QuantumCircuit. Returns (circuit, has_measure).

        With a live backend (see qubasic_core.simulator) the lines execute against
        its per-shot state instead, from line index start_ip up to stop_ip."""
        from qubasic_core.engine import ExecContext
        from qubasic_core.expression import Scope
        from qubasic_core.simtools import QiskitBackend

        live = backend is not None
        qc = backend.qc if backend is not None else QuantumCircuit(self.num_qubits)
        if start_ip == 0:
            self._prepare_run(qc)
        if backend is None:
            backend = QiskitBackend(qc, self._apply_gate)
        prog = self._compiled()
        ctx = ExecContext(
            sorted_lines=prog,
            ip=start_ip,
            run_vars=Scope(self.variables),
            max_iterations=self._max_iterations,
            qc=qc,
            backend=backend,
        )
        # Reachability scan catches MEASURE inside subroutines and IF clauses,
        # not just literal top-level MEASURE lines.
        has_measure = False if live else self._program_has_measure(sorted(self.program))
        # The circuit under construction, so PRINT STATE / QUBIT(q) inside a
        # program can read the state at that line (a live run reads its own).
        self._build_qc = None if live else qc
        try:
            self._build_loop(ctx, qc, stop_ip)
        finally:
            self._build_qc = None
        return qc, has_measure or bool(self._build_saw_measure)

    def _build_loop(self, ctx: ExecContext, qc: QuantumCircuit, stop_ip: int | None = None) -> None:
        """Execute program lines into qc (the body of build_circuit)."""
        prev, self._strict_names = self._strict_names, False
        try:
            self._build_lines(ctx, qc, stop_ip)
        finally:
            self._strict_names = prev

    def _build_lines(self, ctx: ExecContext, qc: QuantumCircuit, stop_ip: int | None) -> None:
        from qubasic_core.parser import MeasureStmt, line_index
        handler_lines_run = 0
        self._build_saw_measure = False
        prog = ctx.sorted_lines
        end = len(prog) if stop_ip is None else stop_ip
        while ctx.ip < end:
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            line_num = prog.lines[ctx.ip]
            stmt = prog.texts[ctx.ip]
            parsed = prog.stmts[ctx.ip]
            first = prog.first[ctx.ip]

            # Per-activation budget for ON ERROR handlers (RESUME resets it via
            # _in_error_handler), so a handler that never resumes cannot spin.
            if self._in_error_handler:
                handler_lines_run += 1
                if handler_lines_run > 200:
                    self.io.writeln("?ERROR HANDLER LIMIT (200 lines)")
                    break
            else:
                handler_lines_run = 0

            # Debug instrumentation (line-level), each a no-op unless enabled.
            if first:
                if self._trace_mode:
                    self._trace_line(line_num)
                if self._breakpoints and self._check_breakpoint(line_num, prog, ctx.ip):
                    break
            if self._on_timer_target is not None:
                _timer_tgt = self._check_timer_callback(prog, ctx.ip)
                if _timer_tgt is not None:
                    ctx.ip = _timer_tgt
                    continue

            if isinstance(parsed, MeasureStmt):
                self._build_saw_measure = True
                if parsed.qubits:
                    self._measure_subset = [
                        self._resolve_qubit(q)
                        for q in parsed.qubits.replace(',', ' ').split()]
                if self._on_measure_target is not None and not self._on_measure_fired:
                    self._on_measure_fired = True
                    target = line_index(prog, self._on_measure_target)
                    if target is not None:
                        self._gosub_stack.append(ctx.ip + 1)
                        ctx.ip = target
                        continue
                ctx.ip += 1
                continue

            self._profile_line_start(line_num)
            gates_before = qc.size()
            try:
                result = self._exec_line(stmt, parsed=parsed, ctx=ctx)
            except QBasicBuildError as e:
                if self._live_needed:
                    raise
                _handler_ip = self._handle_error(e, line_num, ctx.sorted_lines, ctx.ip)
                if _handler_ip is None:
                    raise QBasicBuildError(
                        f"LINE {line_num}: {e}"
                    ) from None
                ctx.ip = _handler_ip
                continue
            except Exception as e:
                # A measurement bit read as a number makes the run go live;
                # no ON ERROR handler may take that.
                if self._live_needed:
                    raise
                _handler_ip = self._handle_error(e, line_num, ctx.sorted_lines, ctx.ip)
                if _handler_ip is None:
                    raise RuntimeError(f"LINE {line_num}: {e}") from None
                ctx.ip = _handler_ip
                continue
            finally:
                self._profile_line_end(line_num, qc.size() - gates_before)

            if result is ExecResult.END:
                break
            self._advance(ctx, result)

    @staticmethod
    def _advance(ctx: ExecContext, result: ExecOutcome) -> None:
        """Move ctx to a jump target, or to the next statement."""
        if isinstance(result, int) and not isinstance(result, bool):
            ctx.ip = result
        else:
            ctx.ip += 1

    def _compiled(self) -> CompiledProgram:
        """The compiled form of the current program (cached until it changes)."""
        from qubasic_core.parser import compile_program
        key = tuple(sorted(self.program.items()))
        cached = getattr(self, '_compiled_cache', None)
        if cached is None or cached[0] != key:
            cached = (key, compile_program(self.program))
            self._compiled_cache = cached
        return cached[1]

    def _exec_line(self, stmt: str, qc: QuantumCircuit | None = None, loop_stack: list[dict[str, Any]] | None = None, sorted_lines: Sequence[int] | None = None,
                   ip: int = 0, run_vars: MutableMapping[str, Any] | None = None, parsed: Stmt | None = None, *, ctx: ExecContext | None = None) -> ExecOutcome:
        """Execute one statement of a compiled program (see program.py).

        Accepts either individual parameters or ctx (ExecContext). When ctx is
        provided, qc/loop_stack/sorted_lines/ip/run_vars are read from ctx and
        the individual params are ignored.

        Evaluation order (deterministic, first match wins):
          1. Typed fast-path (parsed Stmt): BARRIER, REM, MEASURE, END, RETURN, @REG
          2. Feedforward: an IF on a mid-circuit measurement bit
          3. Control flow: LET, PRINT, GOTO, GOSUB, FOR/NEXT, WHILE/WEND, IF,
             DATA/READ, ON GOTO/GOSUB, SELECT CASE, DO/LOOP, EXIT, SUB/FUNCTION,
             ON ERROR, ASSERT, STOP, SWAP, DEF FN, OPTION BASE
          4. Keyword statements: MEAS, RESET, MEASURE_X/Y/Z, SYNDROME, UNITARY,
             DIM, REDIM, ERASE, GET, INPUT, POKE, SYS, file I/O, PRINT USING, ...
          5. Gate application (subroutine expansion + gate dispatch)

        Returns: int (jump target ip), ExecResult.ADVANCE, or ExecResult.END.
        """
        if ctx is not None:
            qc = ctx.backend.qc if ctx.backend and hasattr(ctx.backend, 'qc') else ctx.qc
            loop_stack = ctx.loop_stack
            sorted_lines = ctx.sorted_lines
            ip = ctx.ip
            run_vars = ctx.run_vars
        if loop_stack is None or sorted_lines is None or run_vars is None:
            raise TypeError("_exec_line needs ctx, or loop_stack, sorted_lines and run_vars")
        from qubasic_core.parser import (
            AtRegStmt,
            BarrierStmt,
            EndStmt,
            GateStmt,
            IfJumpStmt,
            MeasureStmt,
            RemStmt,
            ReturnStmt,
        )

        # 1. Typed fast-path: terminals that don't need control-flow dispatch
        if parsed is None:
            from qubasic_core.parser import parse_stmt
            parsed = parse_stmt(stmt)
        if isinstance(parsed, BarrierStmt):
            if qc is not None and hasattr(qc, 'barrier'):
                qc.barrier()
            return ExecResult.ADVANCE
        if isinstance(parsed, (RemStmt, MeasureStmt)):
            return ExecResult.ADVANCE
        if isinstance(parsed, EndStmt):
            return ExecResult.END
        if isinstance(parsed, ReturnStmt):
            return self._cf_return(sorted_lines, parsed)[1]
        if isinstance(parsed, AtRegStmt) and not self.locc_mode:
            raise ValueError("@register syntax requires LOCC mode (try: LOCC <n1> <n2>)")

        # 2. Dynamic circuit: an IF on a mid-circuit measurement bit (set by
        #    MEAS) compiles to a Qiskit if_test, so the conditional gates run at
        #    simulation time on the real measured value (feedforward).
        if (isinstance(parsed, IfJumpStmt) and parsed.then_end >= 0 and qc is not None
                and isinstance(sorted_lines, CompiledProgram)
                and hasattr(qc, 'if_test') and getattr(self, '_classical_bits', None)):
            tgt = self._classical_if_target(parsed.condition)
            if tgt is not None:
                self._feedforward_if(qc, tgt, parsed, loop_stack, sorted_lines, run_vars)
                return parsed.end

        # 3. Control-flow dispatch (GOTO, GOSUB, FOR/NEXT, WHILE/WEND, IF, LET,
        #    PRINT, DATA/READ, ON GOTO, SELECT CASE, DO/LOOP, EXIT, SUB/FUNCTION,
        #    ON ERROR, ...)
        handled, result = self._exec_control_flow(
            stmt, loop_stack, sorted_lines, ip, run_vars,
            lambda s, ls, sl, i, rv: self._exec_line(
                s, qc=qc, loop_stack=ls, sorted_lines=sl, ip=i, run_vars=rv),
            parsed=parsed)
        if handled:
            assert result is not None          # a handled statement has an outcome
            return result

        from qubasic_core.parser import RestoreStmt
        if isinstance(parsed, RestoreStmt):
            if isinstance(parsed.target, str):      # labels resolve when the program compiles
                raise ValueError(f"LABEL {parsed.target} NOT FOUND")
            self._restore_to(parsed.target)
            return ExecResult.ADVANCE

        # 4. Keyword statements
        _backend = ctx.backend if ctx else None
        if self._try_stmt_handlers(stmt, qc, run_vars, backend=_backend):
            return ExecResult.ADVANCE

        # 5. Gate application

        # Fast path: GateStmt already parsed by the parser
        if isinstance(parsed, GateStmt):
            info = self._gate_info(parsed.name)
            if info is not None:
                n_params, n_qubits = info
                args = list(parsed.args)
                params = [self._eval_operand(a, run_vars) for a in args[:n_params]]
                qubits = [self._resolve_qubit(a) for a in args[n_params:n_params + n_qubits]]
                try:
                    if _backend:
                        _backend.apply_gate(parsed.name, tuple(params), qubits)
                    else:
                        self._apply_gate(qc, parsed.name, params, qubits)
                except Exception as _gate_err:
                    if 'duplicate' in str(_gate_err).lower():
                        raise QBasicBuildError(
                            f"duplicate qubit arguments in {parsed.name}"
                        ) from None
                    raise
                return ExecResult.ADVANCE
            # Fall through to _apply_gate_str for custom gates not in GATE_TABLE

        # A REPL/config command (QUBITS, SHOTS, METHOD, LOCC, ...) used as a
        # numbered program line reaches here only because it is neither a gate
        # nor a statement. Explain it instead of failing with "UNKNOWN GATE".
        # SEND/SHARE are LOCC statements (handled on the LOCC path), so leave
        # them to their own error.
        _first = stmt.split(None, 1)[0].upper() if stmt.split() else ''
        if ((_first in self._CMD_WITH_ARG or _first in self._CMD_NO_ARG)
                and _first not in ('SEND', 'SHARE')):
            raise QBasicBuildError(
                f"{_first} is a configuration/session command, not a program "
                f"statement; run it without a line number (before RUN), e.g. "
                f"QUBITS 3 / SHOTS 1024 / METHOD statevector / LOCC 2 2")

        # Slow path: subroutine expansion + gate dispatch
        expanded = self._expand_statement(stmt)
        for gate_str in expanded:
            self._apply_gate_str(gate_str, qc, backend=_backend)

        return ExecResult.ADVANCE

    def _classical_if_target(self, condition: str) -> tuple[ClassicalRegister, int] | None:
        """If `condition` tests a mid-circuit measurement bit, return
        (classical_register, target_value); else None.

        Handled forms: ``c``, ``NOT c``, ``c = 0|1``, ``c <> 0|1`` (and the
        ``==`` / ``!=`` spellings).
        """
        cb = getattr(self, '_classical_bits', None)
        if not cb:
            return None
        hit = _feedforward_test(condition, cb)
        if hit is None:
            return None
        name, val = hit
        return cb[name], val

    def _feedforward_if(self, qc: QuantumCircuit, tgt: tuple[ClassicalRegister, int], parsed: IfJumpStmt, loop_stack: list[dict[str, Any]], prog: CompiledProgram, run_vars: MutableMapping[str, Any]) -> None:
        """Emit a Qiskit if_test (with else) for an IF on a measured bit: the
        compiled THEN and ELSE statements run inside the conditional blocks,
        so their gates apply at simulation time on the real outcome."""
        creg, val = tgt

        def _run(start: int, end: int) -> None:
            for i in range(start, end):
                self._exec_line(prog.texts[i], qc=qc, loop_stack=loop_stack,
                                sorted_lines=prog, ip=i, run_vars=run_vars,
                                parsed=prog.stmts[i])

        if parsed.else_end > parsed.else_start:
            with qc.if_test((creg, val)) as else_block:
                _run(parsed.then_start, parsed.then_end)
            with else_block:
                _run(parsed.else_start, parsed.else_end)
        else:
            with qc.if_test((creg, val)):
                _run(parsed.then_start, parsed.then_end)

    def _parse_syndrome(self, stmt: str, run_vars: MutableMapping[str, Any]) -> tuple[str, list[int], str] | None:
        """Parse SYNDROME statement. Returns (pauli_str, qubits, var) or None."""
        m = RE_SYNDROME.match(stmt)
        if not m:
            return None
        rest = m.group(1).strip()
        parts = rest.split('->')
        if len(parts) != 2:
            raise ValueError("SYNDROME syntax: SYNDROME <paulis> <qubits> -> <var>")
        var = parts[1].strip()
        tokens = parts[0].split()
        if len(tokens) < 2:
            raise ValueError("SYNDROME needs a Pauli string and qubit list")
        pauli_str = tokens[0].upper()
        qubit_args = tokens[1:]
        if len(pauli_str) != len(qubit_args):
            raise ValueError(
                f"Pauli string length ({len(pauli_str)}) must match "
                f"qubit count ({len(qubit_args)})")
        for p in pauli_str:
            if p not in 'IXYZ':
                raise ValueError(f"Unknown Pauli: {p} (use I, X, Y, Z)")
        qubits = [int(self._eval_with_vars(q, run_vars)) for q in qubit_args]
        return pauli_str, qubits, var

    @staticmethod
    def _split_colon_stmts(stmt: str) -> list[str]:
        """Split colon-separated statements, inheriting @register prefixes."""
        from qubasic_core.parser import _split_colon_stmts
        return _split_colon_stmts(stmt)

    def _expand_statement(self, stmt: str, _call_stack: set[str] | None = None) -> list[str]:
        """Expand subroutines. Returns list of gate strings.

        Uses explicit call-stack tracking to detect recursion instead of
        an arbitrary depth counter.
        """
        if _call_stack is None:
            _call_stack = set()

        # Handle parenthesized subroutine calls: NAME(arg1, arg2) -> NAME arg1, arg2
        m_call = re.match(r'(\w+)\(([^)]*)\)', stmt)
        if m_call:
            call_name = m_call.group(1).upper()
            if call_name in self.subroutines:
                call_args = m_call.group(2)
                stmt = f"{call_name} {call_args}"

        parts = stmt.split()
        word = parts[0].upper() if parts else ''

        if word not in self.subroutines:
            return [stmt]

        if word in _call_stack:
            raise RuntimeError(f"RECURSION: {word} calls itself")
        _call_stack = _call_stack | {word}

        sub = self.subroutines[word]
        # Handle both legacy (list) and new (dict with params) format
        if isinstance(sub, list):
            body = sub
            param_names = []
        else:
            body = sub['body']
            param_names = sub['params']

        # Parse arguments: NAME arg1, arg2 @offset
        rest = stmt[len(word):].strip()
        offset = 0
        m_off = re.search(r'@(\d+)', rest)
        if m_off:
            offset = int(m_off.group(1))
            rest = rest[:m_off.start()].strip()

        # Parse call arguments
        call_args = [a.strip() for a in rest.split(',') if a.strip()] if rest else []

        # Build param map for single-pass substitution
        param_map = {}
        for i, pname in enumerate(param_names):
            if i < len(call_args):
                param_map[pname] = call_args[i]
        if param_map:
            pattern = re.compile(r'\b(' + '|'.join(re.escape(p) for p in param_map) + r')\b')
            def _sub(m: re.Match[str]) -> str:
                return param_map[m.group(1)]

        result = []
        for gate_str in body:
            gs = pattern.sub(_sub, gate_str) if param_map else gate_str
            if offset:
                gs = self._offset_qubits(gs, offset)
            result.append(gs)
        return result

    def _offset_qubits(self, gate_str: str, offset: int) -> str:
        """Add offset to qubit indices in a gate string, preserving parameters.

        Handles both plain integers and register[index] notation.
        """
        tokens = self._tokenize_gate(gate_str)
        if len(tokens) < 2:
            return gate_str
        gate_name = tokens[0].upper()
        gate_key = GATE_ALIASES.get(gate_name, gate_name)
        info = self._gate_info(gate_key)
        n_params = info[0] if info else 0
        result = [tokens[0]]
        for i, tok in enumerate(tokens[1:]):
            if i < n_params:
                result.append(tok)
            else:
                # Skip register[index] notation -- it resolves its own offset
                m = RE_REG_INDEX.match(tok)
                if m:
                    result.append(tok)
                else:
                    try:
                        result.append(str(int(tok) + offset))
                    except ValueError:
                        result.append(tok)
        return ' '.join(result)

    def _apply_gate_str(self, stmt: str, qc: QuantumCircuit, _call_stack: set[str] | None = None, *, backend: CircuitBackend | None = None) -> None:
        """Parse and apply a single gate string to the circuit.

        When backend is provided, standard gates are dispatched through
        backend.apply_gate() instead of self._apply_gate(qc, ...).
        """
        stmt = stmt.strip()
        if not stmt:
            return

        # Expand subroutines with call-stack tracking for recursion detection
        if _call_stack is None:
            _call_stack = set()
        word = stmt.split()[0].upper() if stmt.split() else ''
        # Also check for parenthesized call syntax: NAME(args)
        m_paren = re.match(r'(\w+)\(', stmt)
        if m_paren:
            paren_name = m_paren.group(1).upper()
            if paren_name in self.subroutines:
                word = paren_name
        if word in self.subroutines:
            if word in _call_stack:
                raise QBasicBuildError(f"RECURSION: {word} calls itself")
            for sub_stmt in self._expand_statement(stmt, _call_stack):
                self._apply_gate_str(sub_stmt, qc, _call_stack | {word}, backend=backend)
            return

        upper = stmt.upper()
        if upper.startswith('REM') or upper.startswith("'") or upper == 'BARRIER':
            if upper == 'BARRIER':
                if backend:
                    backend.barrier()
                else:
                    qc.barrier()
            return
        if upper == 'MEASURE':
            return  # handled at run level

        # CTRL <gate> [params,] control, target(s) -- one control on any gate
        m_ctrl = RE_CTRL.match(stmt)
        if m_ctrl:
            canonical, params, qubits = self._modifier_operands(
                'CTRL', m_ctrl.group(1), m_ctrl.group(2))
            gate = self._gate_object(canonical, params).control(1)
            if backend and hasattr(backend, 'append_controlled'):
                backend.append_controlled(gate, qubits)
            else:
                qc.append(gate, qubits)
            return

        # INV <gate> [params,] qubit(s) -- inverse (dagger) of one gate
        m_inv = RE_INV.match(stmt)
        if m_inv:
            canonical, params, qubits = self._modifier_operands(
                'INV', m_inv.group(1), m_inv.group(2))
            gate = self._gate_object(canonical, params).inverse()
            if backend and hasattr(backend, 'append_controlled'):
                backend.append_controlled(gate, qubits)
            else:
                qc.append(gate, qubits)
            return

        # Parse: GATE [params] qubits
        # Tokenize
        tokens = self._tokenize_gate(stmt)
        if not tokens:
            return

        gate_name = tokens[0].upper()
        gate_name = GATE_ALIASES.get(gate_name, gate_name)

        info = self._gate_info(gate_name)
        if info is None:
            raise ValueError(f"UNKNOWN GATE: {gate_name}")

        n_params, n_qubits = info
        args = tokens[1:]

        # Parse arguments: first n_params are parameters, rest are qubits
        if len(args) < n_params + n_qubits:
            raise ValueError(
                f"{gate_name} needs {n_params} param(s) and {n_qubits} qubit(s), "
                f"got {len(args)} arg(s)")

        params = [self._eval_operand(a) for a in args[:n_params]]
        qubits = [self._resolve_qubit(a) for a in args[n_params:n_params+n_qubits]]

        # Apply gate through backend when available
        try:
            if backend:
                backend.apply_gate(gate_name, tuple(params), qubits)
            else:
                self._apply_gate(qc, gate_name, params, qubits)
        except Exception as _gate_err:
            if 'duplicate' in str(_gate_err).lower():
                raise QBasicBuildError(
                    f"duplicate qubit arguments in {gate_name}"
                ) from None
            raise

    def _modifier_operands(self, keyword: str, gate_name: str, arg_text: str,
                           limit: int | None = None) -> tuple[str, list, list[int]]:
        """Operands of CTRL/INV: (gate, params, qubits). For CTRL the control
        qubit comes first in `qubits`, then the gate's own qubits."""
        canonical = GATE_ALIASES.get(gate_name.upper(), gate_name.upper())
        info = self._gate_info(canonical)
        if info is None:
            raise ValueError(f"{keyword} {gate_name.upper()}: gate not found")
        n_params, n_q = info
        args = self._tokenize_gate(f"{canonical} {arg_text}")[1:]
        controls = 1 if keyword == 'CTRL' else 0
        if len(args) != n_params + controls + n_q:
            ctl = "1 control and " if controls else ""
            raise ValueError(
                f"{keyword} {canonical} needs {n_params} parameter(s), {ctl}"
                f"{n_q} target qubit(s); got {len(args)} argument(s)")
        params = [self._eval_operand(a) for a in args[:n_params]]
        qubits = [self._resolve_qubit(a, n_qubits=limit) for a in args[n_params:]]
        if len(set(qubits)) != len(qubits):
            raise QBasicBuildError(f"duplicate qubit arguments in {keyword} {canonical}")
        return canonical, params, qubits

    def _gate_object(self, canonical: str, params: list) -> Gate:
        """A Qiskit gate instance for a built-in or UNITARY-defined gate."""
        if canonical in self._custom_gates:
            from qiskit.circuit.library import UnitaryGate
            return UnitaryGate(self._custom_gates[canonical], label=canonical)
        from qiskit.circuit.library import get_standard_gate_name_mapping
        lib = get_standard_gate_name_mapping().get(canonical.lower())
        if lib is None:
            raise ValueError(f"GATE {canonical} NOT IMPLEMENTED")
        return type(lib)(*params)

    def _tokenize_gate(self, stmt: str) -> list[str]:
        """Split gate statement into tokens, handling commas and register notation.

        When arguments are comma-separated, splits on commas only so that
        compound expressions like ``I + 1`` are preserved as single tokens.
        Falls back to whitespace splitting when no commas are present
        (legacy format like ``CX 0 1``).
        """
        stmt = RE_REG_INDEX.sub(r'\1[\2]', stmt)
        parts = stmt.strip().split(None, 1)
        if len(parts) < 2:
            return [parts[0]] if parts else []
        gate = parts[0]
        args_str = parts[1]
        if ',' in args_str:
            # Comma-separated: split on commas, preserving expressions
            args = [a.strip() for a in args_str.split(',') if a.strip()]
        else:
            # Space-separated: split on whitespace (legacy)
            args = [a.strip() for a in args_str.split() if a.strip()]
        return [gate] + args

    def _resolve_qubit(self, arg: str, *, n_qubits: int | None = None) -> int:
        """Resolve a qubit argument and validate range.

        Accepts: integer literal, register[index], or expression.
        Validates against n_qubits (defaults to self.num_qubits).
        """
        limit = n_qubits if n_qubits is not None else self.num_qubits
        m = RE_REG_INDEX.match(arg)
        if m:
            name = m.group(1).lower()
            idx = int(m.group(2))
            if name not in self.registers:
                raise QBasicRangeError(f"UNKNOWN REGISTER: {name}")
            start, size = self.registers[name]
            if idx >= size:
                raise QBasicRangeError(f"{name}[{idx}] OUT OF RANGE (size={size})")
            q = start + idx
        else:
            try:
                q = int(self._eval_operand(arg))
            except Exception:
                raise ValueError(f"CANNOT RESOLVE QUBIT: {arg}")
        if q < 0 or q >= limit:
            raise QBasicRangeError(f"QUBIT {q} OUT OF RANGE (0-{limit-1})")
        return q

    def _apply_gate(self, qc: QuantumCircuit, gate_name: str, params: list[float], qubits: list[int]) -> None:
        """Apply a gate to the quantum circuit."""
        n = qc.num_qubits
        for q in qubits:
            if q < 0 or q >= n:
                raise QBasicRangeError(
                    f"QUBIT {q} OUT OF RANGE (0-{n-1}) in {gate_name}"
                )
        method_name = self._GATE_DISPATCH.get(gate_name)
        if method_name:
            method = getattr(qc, method_name)
            method(*params, *qubits)
        elif gate_name in self._custom_gates:
            from qiskit.circuit.library import UnitaryGate
            qc.append(UnitaryGate(self._custom_gates[gate_name]), qubits)
        else:
            raise ValueError(f"GATE {gate_name} NOT IMPLEMENTED")

    def run_immediate(self, line: str) -> None:
        """Execute a single gate command immediately.

        Uses the same _exec_line pipeline as cmd_run for consistency.
        """
        # In LOCC mode, handle @register prefix via the numpy engine. Split
        # colon-compound @REG lines so each gate is applied (a single @REG line
        # is just the one-part case); a bare @REG line keeps prior behavior.
        if self.locc_mode and self.locc and line.strip().startswith('@'):
            parts = self._split_colon_stmts(line) if ':' in line else [line.strip()]
            for part in parts:
                m = RE_AT_REG_LINE.match(part)
                if not m:
                    self.io.writeln(f"?BAD @register statement: {part}")
                    continue
                reg = m.group(1).upper()
                gate_stmt = m.group(2).strip()
                if reg not in self.locc.names:
                    self.io.writeln(f"?UNKNOWN REGISTER: {reg} (have {', '.join(self.locc.names)})")
                    return
                self._locc_apply_gate(reg, gate_stmt)
            self._locc_state()
            return
        if line.strip().startswith('@'):
            self.io.writeln("?@register syntax requires LOCC mode (try: LOCC <n1> <n2>)")
            return
        # Compile the line as a one-line program and run it through the same
        # statement pipeline as RUN.
        from qubasic_core.engine import ExecContext
        from qubasic_core.parser import compile_program
        qc = QuantumCircuit(self.num_qubits)
        prog = compile_program({0: line})
        imm_ctx = ExecContext(sorted_lines=prog, ip=0,
                              run_vars=dict(self.variables), qc=qc)
        while imm_ctx.ip < len(prog):
            imm_ctx.iteration_count += 1
            if imm_ctx.iteration_count > self._max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({self._max_iterations}) — possible infinite loop")
            result = self._exec_line(prog.texts[imm_ctx.ip], parsed=prog.stmts[imm_ctx.ip],
                                     ctx=imm_ctx)
            if result is ExecResult.END:
                break
            self._advance(imm_ctx, result)
        # A classical statement typed at the prompt (PRINT, MEASURE, an IF whose
        # clause printed, ...) adds no circuit operations and has already emitted
        # its own output via _exec_line. Simulating an empty circuit just to print
        # an unchanged |psi> would be spurious, so skip the statevector dump.
        if qc.size() == 0:
            return
        from qubasic_core import simtools as cap
        peak = cap.aer_peak_bytes(self.num_qubits, 'statevector', self.sim_precision, 'end')
        if not cap.fits(peak):
            self.last_circuit = qc
            self.io.writeln(f"  (applied; a {self.num_qubits}-qubit state needs "
                            f"~{cap.fmt_gb(peak)}, so it is not shown)")
            return
        qc.save_statevector(label='qb_state')
        backend = self._make_backend('statevector')
        result = backend.run(transpile(qc, backend), shots=1).result()
        sv = np.asarray(result.data()['qb_state'].data)
        self.last_sv = sv
        self.last_circuit = qc
        self._print_sv_compact(sv)


# ---------------------------------------------------------------------------
# QUBASIC live execution: programs that read measurement results as numbers.
#
# A program whose classical code reads a mid-circuit measurement bit (PRINT m,
# LET y = m + 1, a GOTO or loop chosen by it) cannot be one Qiskit circuit,
# because the value differs from shot to shot. Such a program runs live: each
# shot executes the program from the top on a numpy statevector, MEAS
# collapses the state and returns its outcome, and IF, PRINT and LET see real
# bits. Gates the executor appends to the shot's circuit are applied to the
# state when a measurement or a state read needs them. An active noise model
# runs as quantum trajectories (gate errors as Kraus jumps, readout error as
# classical flips).
#
# Only the first shot prints. Later shots replay the first shot's INPUT
# answers and skip file output, so a run reads as one execution of the
# program followed by the histogram over every shot.
# ---------------------------------------------------------------------------

# ── IO wrappers ─────────────────────────────────────────────────────────

class ShotIO:
    """IO for shot-by-shot execution. The first shot writes through and
    records every line read; after start_replay() output is dropped and the
    recorded lines are returned in order. `answers` and `skip` carry what an
    abandoned circuit build already read and showed."""

    def __init__(self, io: Any, answers: list[str] | None = None, skip: int = 0) -> None:
        self._io = io
        self.answers: list[str] = list(answers or [])
        self._pos = 0
        self._skip = skip
        self.replay = False

    def start_replay(self) -> None:
        self.replay = True
        self._pos = 0

    def _emit(self, text: str, newline: bool) -> None:
        if self.replay:
            return
        if self._skip:
            full = text + ('\n' if newline else '')
            if len(full) <= self._skip:
                self._skip -= len(full)
                return
            text, newline = full[self._skip:], False
            self._skip = 0
        if newline:
            self._io.writeln(text)
        else:
            self._io.write(text)

    def write(self, text: str) -> None:
        self._emit(text, False)

    def writeln(self, text: str) -> None:
        self._emit(text, True)

    def read_line(self, prompt: str) -> str:
        if self._pos < len(self.answers):
            self._pos += 1
            return self.answers[self._pos - 1]
        if self.replay:
            raise EOFError
        answer = self._io.read_line(prompt)
        self.record(answer)
        return answer

    def record(self, answer: str) -> None:
        """Record input read past this IO (a keypress from GET)."""
        self.answers.append(answer)
        self._pos += 1


class BufferedIO:
    """Holds a circuit build's output until the build finishes, so a build
    abandoned for a live run prints nothing twice. Reading input shows what
    is held first; `shown` and `answers` record what reached the user."""

    def __init__(self, io: Any) -> None:
        self._io = io
        self._held: list[tuple[str, bool]] = []
        self.shown = 0
        self.answers: list[str] = []

    def write(self, text: str) -> None:
        self._held.append((text, False))

    def writeln(self, text: str) -> None:
        self._held.append((text, True))

    def flush(self) -> None:
        for text, newline in self._held:
            if newline:
                self._io.writeln(text)
            else:
                self._io.write(text)
            self.shown += len(text) + newline
        self._held.clear()

    def discard(self) -> None:
        self._held.clear()

    def read_line(self, prompt: str) -> str:
        self.flush()
        answer = self._io.read_line(prompt)
        self.answers.append(answer)
        return answer

    def record(self, answer: str) -> None:
        self.flush()
        self.answers.append(answer)


# ── Noise as trajectories ───────────────────────────────────────────────

class _Channel:
    """A quantum channel ready for trajectory sampling. A mixed-unitary
    channel (Pauli, depolarizing) draws its branch with fixed probabilities;
    a general one weighs each Kraus operator by its probability on the
    current state. Operators are Qiskit-ordered over the channel's qubits."""

    def __init__(self, error: Any) -> None:
        from qiskit.quantum_info import Kraus
        ops = [np.asarray(k, dtype=complex) for k in Kraus(error).data]
        self.n = int(round(np.log2(ops[0].shape[0])))
        d = ops[0].shape[0]
        eye = np.eye(d)
        weights, unitaries = [], []
        mixed = True
        for k in ops:
            g = k.conj().T @ k
            c = float(g[0, 0].real)
            if c < 1e-15:
                continue
            if not np.allclose(g, c * eye, atol=1e-10):
                mixed = False
                break
            weights.append(c)
            unitaries.append(k / np.sqrt(c))
        self.mixed = mixed
        if mixed:
            w = np.array(weights) / sum(weights)
            self.cum = np.cumsum(w)
            self.ops = unitaries
            self.trivial = [bool(np.allclose(u, u[0, 0] * eye)) for u in unitaries]
        else:
            self.ops = ops


class LiveNoise:
    """An Aer noise model as trajectory channels: per instruction name and
    qubits (qubit-specific errors override all-qubit ones, as in Aer), plus
    single-qubit readout errors."""

    def __init__(self, model: Any) -> None:
        self._default = {name: _Channel(err) for name, err in
                         getattr(model, '_default_quantum_errors', {}).items()}
        self._local = {name: {tuple(q): _Channel(err) for q, err in per.items()}
                       for name, per in getattr(model, '_local_quantum_errors', {}).items()}
        self._ro_default = self._readout(getattr(model, '_default_readout_error', None))
        self._ro_local = {tuple(q): self._readout(e) for q, e in
                          getattr(model, '_local_readout_errors', {}).items()}

    @staticmethod
    def _readout(err: Any) -> np.ndarray | None:
        if err is None:
            return None
        p = np.asarray(err.probabilities, dtype=float)
        return p if p.shape == (2, 2) else None

    def channel(self, name: str, qubits: list[int]) -> _Channel | None:
        local = self._local.get(name)
        if local:
            ch = local.get(tuple(qubits))
            if ch is not None:
                return ch
        ch = self._default.get(name)
        if ch is not None and ch.n == len(qubits):
            return ch
        return None

    def readout(self, qubit: int) -> np.ndarray | None:
        return self._ro_local.get((qubit,), self._ro_default)


# ── The per-shot state ──────────────────────────────────────────────────

_CONTROL_FLOW_OPS = frozenset({'if_else', 'while_loop', 'for_loop', 'switch_case', 'box'})


def _numeric(params: Sequence[Any]) -> bool:
    try:
        for p in params:
            float(p)
        return True
    except (TypeError, ValueError):
        return False


class LiveState:
    """The statevector of one live shot plus the circuit fragment the
    executor is appending to. Built-in gates apply at once; everything else
    queues in `qc` until flush(). Qubit q is bit q of the amplitude index."""

    def __init__(self, n: int, precision: str = 'double', noise: LiveNoise | None = None) -> None:
        from qiskit import QuantumCircuit
        self.n = n
        self.dtype = np.complex64 if precision == 'single' else np.complex128
        self._sv: np.ndarray = np.zeros(1 << n, dtype=self.dtype)
        self._sv[0] = 1.0
        self.qc = QuantumCircuit(n)
        self.noise = noise
        self.ops = 0

    def reset_state(self, init: np.ndarray | None = None) -> None:
        if init is None:
            self._sv[...] = 0
            self._sv[0] = 1.0
        else:
            np.copyto(self._sv, init)
        self.qc.clear()

    @property
    def sv(self) -> np.ndarray:
        self.flush()
        return self._sv

    def flush(self) -> None:
        """Apply the instructions queued in qc, in order, and clear it."""
        qc = self.qc
        if len(qc.data) == 0:
            return
        index = {bit: i for i, bit in enumerate(qc.qubits)}
        for inst in list(qc.data):
            self.apply_operation(inst.operation, [index[q] for q in inst.qubits])
        qc.clear()

    # Gates

    def apply_named(self, name: str, params: tuple, qubits: list[int]) -> None:
        """A built-in gate (GATE_TABLE name), MSB-first over `qubits`."""
        canonical = GATE_ALIASES.get(name, name)
        _apply_gate_inplace(self._sv, _np_gate_matrix(canonical, tuple(params)),
                            list(qubits), self.n)
        self.ops += 1
        self._noise_after(canonical.lower(), qubits)

    def apply_matrix(self, matrix: np.ndarray, qubits: list[int], name: str = 'unitary') -> None:
        """A unitary in Qiskit order over `qubits` (qubits[0] least significant)."""
        _apply_gate_inplace(self._sv, np.asarray(matrix, dtype=complex), list(qubits)[::-1], self.n)
        self.ops += 1
        self._noise_after(name, qubits)

    def apply_operation(self, op: Any, qubits: list[int]) -> None:
        name = op.name
        if name in ('barrier', 'delay'):
            return
        if name == 'measure':
            self.measure(qubits[0])
            return
        if name == 'reset':
            self.reset(qubits[0])
            return
        if name == 'set_statevector':
            if list(qubits) != list(range(self.n)):
                raise ValueError("SET_STATE in a live run must cover every qubit")
            np.copyto(self._sv, np.asarray(op.params[0], dtype=complex).reshape(-1))
            return
        if name == 'kraus':
            self._kraus([np.asarray(k, dtype=complex) for k in op.params], qubits)
            return
        if name in _CONTROL_FLOW_OPS or name.startswith('save_') or name == 'set_density_matrix':
            raise ValueError(f"'{name}' is not available in a live run")
        upper = name.upper()
        if upper in _GATE_BUILDERS and _numeric(op.params):
            self.apply_named(upper, tuple(float(p) for p in op.params), qubits)
            return
        if name == 'unitary' or (op.num_qubits <= 6 and hasattr(op, '__array__')):
            try:
                self.apply_matrix(op.to_matrix(), qubits, name)
                return
            except Exception:
                pass
        definition = getattr(op, 'definition', None)
        if definition is not None:
            sub = {bit: i for i, bit in enumerate(definition.qubits)}
            for inst in definition.data:
                self.apply_operation(inst.operation, [qubits[sub[q]] for q in inst.qubits])
            return
        from qiskit.quantum_info import Operator
        self.apply_matrix(Operator(op).data, qubits, name)

    # Non-unitary steps

    def _noise_after(self, name: str, qubits: list[int]) -> None:
        if self.noise is not None:
            ch = self.noise.channel(name, list(qubits))
            if ch is not None:
                self._apply_channel(ch, list(qubits))

    def _apply_channel(self, ch: _Channel, qubits: list[int]) -> None:
        if ch.mixed:
            i = int(np.searchsorted(ch.cum, np.random.random(), side='right'))
            i = min(i, len(ch.ops) - 1)
            if not ch.trivial[i]:
                _apply_gate_inplace(self._sv, ch.ops[i], qubits[::-1], self.n)
        else:
            self._kraus(ch.ops, qubits)

    def _kraus(self, ops: list[np.ndarray], qubits: list[int]) -> None:
        """One trajectory step: pick K_i with probability ||K_i psi||^2."""
        from qubasic_core.simtools import subsystem_rdm
        msb = list(qubits)[::-1]
        if len(ops) == 1:
            pick = 0
        else:
            rho = subsystem_rdm(self._sv, msb, self.n)
            probs = np.array([max(float(np.real(np.trace(k @ rho @ k.conj().T))), 0.0)
                              for k in ops])
            probs /= probs.sum()
            pick = int(np.searchsorted(np.cumsum(probs), np.random.random(), side='right'))
            pick = min(pick, len(ops) - 1)
        _apply_gate_inplace(self._sv, ops[pick], msb, self.n)
        norm = float(np.sqrt(np.vdot(self._sv, self._sv).real))
        if norm > 0:
            self._sv *= 1.0 / norm

    def measure(self, qubit: int) -> int:
        """Projective Z measurement; returns the recorded bit (readout error
        included)."""
        self.flush()
        self._noise_after('measure', [qubit])
        bit = _measure_inplace(self._sv, qubit, self.n)
        if self.noise is not None:
            p = self.noise.readout(qubit)
            if p is not None and np.random.random() < p[bit][1 - bit]:
                bit = 1 - bit
        return bit

    def reset(self, qubit: int) -> None:
        self.flush()
        if _measure_inplace(self._sv, qubit, self.n):
            _apply_gate_inplace(self._sv, _np_gate_matrix('X'), [qubit], self.n)
        self._noise_after('reset', [qubit])

    def sample_final(self, subset: list[int] | None) -> str:
        """The final MEASURE of a shot: one outcome over `subset` (or every
        qubit), as a bitstring with the last measured qubit leftmost."""
        self.flush()
        qubits = list(subset) if subset else list(range(self.n))
        if self.noise is not None:
            for q in qubits:
                self._noise_after('measure', [q])
        key = next(iter(_sample_counts(self._sv, self.n, 1)))
        bits = {q: int(key[self.n - 1 - q]) for q in qubits}
        if self.noise is not None:
            for q in qubits:
                p = self.noise.readout(q)
                if p is not None and np.random.random() < p[bits[q]][1 - bits[q]]:
                    bits[q] = 1 - bits[q]
        return ''.join(str(bits[q]) for q in reversed(qubits))


class LiveBackend:
    """Executor backend for a live shot: built-in gates go straight to the
    state, custom gates as their matrices, and other circuit operations
    queue in the shot's circuit."""

    def __init__(self, state: LiveState, custom_gates: dict[str, np.ndarray]) -> None:
        self.state = state
        self._custom = custom_gates

    @property
    def qc(self) -> Any:
        return self.state.qc

    def apply_gate(self, gate_name: str, params: tuple, qubits: list[int]) -> None:
        if len(set(qubits)) != len(qubits):
            raise QBasicBuildError(f"duplicate qubit arguments in {gate_name}")
        self.state.flush()
        canonical = GATE_ALIASES.get(gate_name, gate_name)
        if canonical in self._custom:
            self.state.apply_matrix(self._custom[canonical], list(qubits))
        else:
            self.state.apply_named(canonical, tuple(params), list(qubits))

    def barrier(self) -> None:
        pass

    def reset(self, qubit: int) -> None:
        self.state.reset(qubit)

    def measure(self, qubit: int, classical_bit: Any = None) -> int:
        return self.state.measure(qubit)

    def h(self, qubit: int) -> None:
        self.apply_gate('H', (), [qubit])

    def append_inverse(self, sub_qc: Any, qubits: list[int]) -> None:
        self.state.qc.append(sub_qc.inverse(), qubits)

    def append_controlled(self, gate: Any, qubits: list[int]) -> None:
        self.state.qc.append(gate, qubits)


# ── Deciding when a program needs to run live ───────────────────────────

_RE_BIT_TARGET = re.compile(r'\b(?:MEAS\s+\S+|SYNDROME\b[^:]*?)\s*->\s*(\w+)', re.IGNORECASE)
_RE_IDENT = re.compile(r'[A-Za-z_]\w*')
_RE_BASIS_BIT = re.compile(r'm[xyz]_\d+$')
_QUANTUM_CLAUSE_WORDS = frozenset({'CTRL', 'INV', 'RESET', 'BARRIER'})


def _identifiers(text: str) -> set[str]:
    """Identifiers outside string literals, without MEAS/SYNDROME targets."""
    text = _RE_BIT_TARGET.sub('', text)
    out: set[str] = set()
    for i, seg in enumerate(text.split('"')):
        if i % 2 == 0:
            out.update(_RE_IDENT.findall(seg))
    return out


class LiveMixin(HostState):
    """Live (shot-by-shot) execution for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.variables,
    self.arrays, self.shots, self.num_qubits, self.io, self.build_circuit(),
    self._program_has_measure(), self._noise_model, self._custom_gates.
    """

    _gosub_stack: list

    def _init_live(self) -> None:
        self._live: LiveState | None = None
        self._live_needed = False
        self._live_saves: dict[str, tuple[str, list]] = {}

    # Static check

    def _bit_names(self, texts: list[str]) -> tuple[set[str], bool]:
        names: set[str] = set()
        basis = False
        for t in texts:
            names.update(m.group(1) for m in _RE_BIT_TARGET.finditer(t))
            if re.search(r'\bMEASURE_[XYZ]\b', t, re.IGNORECASE):
                basis = True
        return names, basis

    def _needs_live(self, sorted_lines: Sequence[int]) -> bool:
        """Whether the program reads a measurement bit anywhere a compiled
        dynamic circuit cannot: anything but IF <bit> THEN <gates> [ELSE
        <gates>] on one bit."""
        from qubasic_core.parser import CompoundStmt, IfThenStmt, parse_stmt, strip_comment
        texts = [strip_comment(self.program[ln]) for ln in sorted_lines]
        names, basis = self._bit_names(texts)
        if not names and not basis:
            return False

        def is_bit(ident: str) -> bool:
            return ident in names or (basis and bool(_RE_BASIS_BIT.match(ident)))

        def reads(text: str) -> bool:
            return any(is_bit(i) for i in _identifiers(text))

        def quantum_only(clause: str | None) -> bool:
            if not clause:
                return True
            for part in self._split_colon_stmts(clause):
                word = part.split(None, 1)[0].upper().split('(')[0] if part.strip() else ''
                if not (word in _QUANTUM_CLAUSE_WORDS or self._gate_info(word) is not None
                        or word in self.subroutines):
                    return False
            return not reads(clause)

        def check(text: str) -> bool:
            p = parse_stmt(text)
            if isinstance(p, CompoundStmt):
                return any(check(part) for part in p.parts)
            if isinstance(p, IfThenStmt):
                if not reads(p.condition):
                    return (bool(p.then_clause) and check(p.then_clause)) or (
                        p.else_clause is not None and check(p.else_clause))
                bit_names = {i for i in _identifiers(p.condition) if is_bit(i)}
                return not (p.then_clause and _feedforward_test(p.condition, bit_names)
                            and quantum_only(p.then_clause)
                            and quantum_only(p.else_clause))
            return reads(text)

        return any(check(t) for t in texts)

    # Running

    def _live_noise(self) -> LiveNoise | None:
        model = self._noise_model
        if model is None and self._qubit_noise:
            model = self._build_qubit_noise()
        return LiveNoise(model) if model is not None else None

    def _live_prefix(self, sorted_lines: Any = None) -> int:
        """Number of leading statements that run the same in every shot:
        plain gates, assignments and PRINTs with no jumps, input, randomness
        or measurement. They run once and each shot starts after them."""
        from qubasic_core.parser import BarrierStmt, GateStmt, LetStmt, PrintStmt, RemStmt
        prog = self._compiled()
        k = 0
        for p, text in zip(prog.stmts, prog.texts):
            if not isinstance(p, (GateStmt, LetStmt, PrintStmt, RemStmt, BarrierStmt)):
                break
            if re.search(r'\b(RND|TIMER|INKEY|PEEK|INPUT|FN)\b', text, re.IGNORECASE):
                break
            k += 1
        return k

    def _record_save(self, kind: str, var: str, value: Any) -> None:
        self._live_saves.setdefault(var, (kind, []))[1].append(value)

    def _run_live(self, t0: float, sorted_lines: Sequence[int], carry: BufferedIO | None = None) -> None:
        """Run the program shot by shot (see the module docstring)."""
        from qubasic_core import simtools as cap
        n = self.num_qubits
        prec = self.sim_precision
        if self._pending_set_density is not None:
            self.io.writeln("?A program that reads measurement results runs shot by shot "
                            "on a pure state; use SET_STATE rather than SET_DENSITY")
            return
        peak = cap.numpy_peak_bytes(n, prec)
        if not cap.fits(peak):
            self.io.writeln(f"?MEMORY: a live run at {n} qubits needs ~{cap.fmt_gb(peak)}, "
                            f"{cap.fmt_gb(cap.available_bytes())} available")
            return
        has_measure = self._program_has_measure(sorted_lines)
        shots = self.shots if has_measure else 1
        noise = self._live_noise()
        state = LiveState(n, prec, noise)
        backend = LiveBackend(state, self._custom_gates)
        shot_io = ShotIO(self.io, carry.answers if carry else None, carry.shown if carry else 0)
        real_io = self.io
        base_vars = dict(self.variables)
        base_arrays = {k: list(v) for k, v in self.arrays.items()}
        keep_copy = shots > 1 and cap.fits(cap.numpy_peak_bytes(n, prec, True))
        if self._seed is not None:
            np.random.seed(self._seed)
        self._live_saves = {}
        counts: dict[str, int] = {}
        first_vars: dict[str, Any] = {}
        first_arrays: dict[str, Any] = {}
        first_sv = None
        first_ops = 0
        start, snapshot, start_vars, start_arrays = 0, None, base_vars, base_arrays
        prefix_ops = 0
        progress = shots > 50 and sys.stdout.isatty()
        self._live = state
        self.io = shot_io
        try:
            prefix = self._live_prefix(sorted_lines) if noise is None and keep_copy else 0
            if prefix:
                self._gosub_stack = []
                self.build_circuit(backend=backend, stop_ip=prefix)
                snapshot = state.sv.copy()
                prefix_ops = state.ops
                start, start_vars = prefix, dict(self.variables)
                start_arrays = {k: list(v) for k, v in self.arrays.items()}
            for shot in range(shots):
                if shot:
                    shot_io.start_replay()
                if shot or snapshot is None:
                    state.reset_state(snapshot)
                    self.variables.clear()
                    self.variables.update(start_vars)
                    self.arrays.clear()
                    self.arrays.update({k: list(v) for k, v in start_arrays.items()})
                state.ops = 0
                self._gosub_stack = []
                self._data_ptr = 0
                self.build_circuit(backend=backend, start_ip=start)
                if has_measure:
                    key = state.sample_final(self._measure_subset)
                    counts[key] = counts.get(key, 0) + 1
                if shot == 0:
                    first_vars = dict(self.variables)
                    first_arrays = {k: list(v) for k, v in self.arrays.items()}
                    first_ops = prefix_ops + state.ops
                    if shots == 1:
                        first_sv = state.sv
                    elif keep_copy:
                        first_sv = state.sv.copy()
                if progress and (shot + 1) % max(1, shots // 10) == 0:
                    real_io.write(f"  {100 * (shot + 1) // shots}% ({shot + 1}/{shots} shots)\r")
        except KeyboardInterrupt:
            self.io = real_io
            self.io.writeln("\n?INTERRUPTED")
            return
        except Exception as e:
            self.io = real_io
            self.io.writeln(f"?RUNTIME ERROR: {e}")
            return
        finally:
            self.io = real_io
            self._live = None
        if progress:
            real_io.write(" " * 40 + "\r")

        self.variables.clear()
        self.variables.update(first_vars)
        self.arrays.clear()
        self.arrays.update(first_arrays)
        self._finish_live_saves()
        self.last_sv = first_sv
        self.last_counts = counts if has_measure else None
        self.last_circuit = None
        dt = time.time() - t0
        self._update_status(gate_count=first_ops, circuit_depth=0, run_time_ms=dt * 1000)
        self.variables['_TIME'] = dt
        self._run_manifest = {
            'program': dict(self.program), 'num_qubits': n, 'shots': shots,
            'method': 'live', 'precision': prec, 'device': 'CPU', 'seed': self._seed,
            'noise_depol_p': self._noise_depol_p, 'depth': 0, 'gates': first_ops,
            'time_s': dt,
        }
        tags = ["method=live"]
        if prec == 'single':
            tags.append("precision=single")
        if noise is not None:
            tags.append("noise=trajectories")
        shot_word = f"{shots} shot{'s' if shots != 1 else ''}"
        self.io.writeln(f"\nRAN {len(self.program)} lines, {n} qubits, {shot_word} in {dt:.2f}s  "
                        f"[{first_ops} gates/shot, {', '.join(tags)}]")
        if has_measure and counts:
            self.print_histogram(counts)
            self._mitigate_after_run()
        elif not has_measure:
            self.io.writeln("(no MEASURE — one shot; use STATE, PROBS, or BLOCH to inspect)")

    def _finish_live_saves(self) -> None:
        """SAVE_EXPECT / SAVE_PROBS hold their mean over the shots after a
        live run; SAVE_AMPS holds the first shot's amplitudes."""
        for var, (kind, values) in self._live_saves.items():
            if kind == 'amp':
                val = values[0]
                self.arrays[var] = [complex(v) for v in val]
            else:
                val = np.mean(np.array(values), axis=0)
                if kind == 'prob':
                    self.arrays[var] = list(np.asarray(val, dtype=float))
                else:
                    val = _expectation(val)
            self.variables[var] = val
