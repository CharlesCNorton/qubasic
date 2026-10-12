"""QUBASIC configuration commands: QUBITS, SHOTS, METHOD, PRECISION, COUPLING, BASIS, STATUS, SEED, VERSION."""

from __future__ import annotations

import sys
import re
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qubasic_core.engine import (
    MAX_QUBITS, RAM_BUDGET_FRACTION,
)
from qubasic_core.errors import QBasicRangeError
from qubasic_core.host import HostState


class ConfigMixin(HostState):
    """Session configuration commands for QBasicTerminal."""

    def cmd_qubits(self, rest: str) -> None:
        """QUBITS [n] — set or show the qubit count; the ceiling is per METHOD (statevector 32, stabilizer 4096, MPS and automatic 1024)."""
        if not rest:
            self.io.writeln(f"QUBITS = {self.num_qubits}")
            return
        n = int(rest)
        from qubasic_core.engine import METHOD_MAX_QUBITS
        cap = METHOD_MAX_QUBITS.get(self.sim_method, MAX_QUBITS)
        if n < 1 or n > cap:
            from qubasic_core.errors import QBasicRangeError
            hint = ("" if n <= MAX_QUBITS or self.sim_method != 'statevector'
                    else " (METHOD stabilizer reaches 4096, MPS/automatic 1024)")
            raise QBasicRangeError(f"RANGE: 1-{cap} for METHOD {self.sim_method}{hint}")

        if n != self.num_qubits:
            self._invalidate_run_state()
        self.num_qubits = n
        self.registers.clear()
        self._report_width(n)

    def _report_width(self, n: int) -> None:
        """Describe what an n-qubit register costs under the current METHOD
        and PRECISION, and what RUN will do if it does not fit."""
        from qubasic_core import capacity as cap
        method, prec = self.sim_method, self.sim_precision
        if method == 'stabilizer':
            self.io.writeln(f"{n} QUBITS ALLOCATED  (stabilizer tableau; polynomial memory)")
            return
        if method in ('matrix_product_state', 'extended_stabilizer'):
            self.io.writeln(f"{n} QUBITS ALLOCATED  ({method}: memory set by the circuit's "
                            f"structure, not 2^n)")
            return
        avail = cap.available_bytes()
        sv_method = 'statevector' if method == 'automatic' else method
        peak = cap.aer_peak_bytes(n, sv_method, prec)
        label = f"~{cap.fmt_gb(peak)} per run" + (", single precision" if prec == 'single' else "")
        self.io.writeln(f"{n} QUBITS ALLOCATED  ({sv_method} {label})")
        if avail is None or cap.fits(peak, avail):
            total = cap.total_bytes()
            if total and peak and n >= 16:
                par = int(total * RAM_BUDGET_FRACTION // peak)
                if par > 0:
                    self.io.writeln(f"  Max parallel instances in 80% budget: ~{par}")
            return
        free = cap.fmt_gb(avail)
        if method == 'automatic':
            single = cap.aer_peak_bytes(n, 'statevector', 'single')
            if prec == 'double' and cap.fits(single, avail):
                self.io.writeln(f"  exceeds {free} available; RUN uses single precision "
                                f"(~{cap.fmt_gb(single)}) for non-Clifford circuits")
            else:
                self.io.writeln(f"  exceeds {free} available; RUN uses stabilizer for "
                                f"Clifford circuits and matrix_product_state otherwise")
        else:
            hint = " (try PRECISION SINGLE)" if prec == 'double' and method in self._PRECISION_METHODS else ""
            self.io.writeln(f"  WARNING: exceeds {free} available; RUN will refuse{hint}")

    def cmd_shots(self, rest: str) -> None:
        """SHOTS [n] — set or show the number of measurement shots."""
        if not rest:
            self.io.writeln(f"SHOTS = {self.shots}")
            return
        self.shots = max(1, int(rest))
        self.io.writeln(f"SHOTS = {self.shots}")

    def cmd_method(self, rest: str) -> None:
        """METHOD [name | CPU | GPU] — set or show the simulation method or device (no argument probes what is available)."""
        if not rest:
            self.io.writeln(f"METHOD = {self.sim_method}  DEVICE = {self.sim_device}")
            methods = ['automatic', 'statevector', 'density_matrix',
                       'stabilizer', 'matrix_product_state', 'extended_stabilizer',
                       'unitary', 'superop']
            # Probe each method for actual availability
            from qiskit import QuantumCircuit as _QC
            import logging as _logging
            _pqc_m = _QC(1); _pqc_m.h(0); _pqc_m.measure_all()
            _pqc_u = _QC(1); _pqc_u.h(0)  # no measure for unitary/superop
            _aer_log = _logging.getLogger('qiskit_aer')
            _old_level = _aer_log.level
            _aer_log.setLevel(_logging.CRITICAL)
            avail = []
            for m in methods:
                try:
                    _b = AerSimulator(method=m)
                    _probe_qc = _pqc_u if m in ('unitary', 'superop') else _pqc_m
                    _b.run(transpile(_probe_qc, _b), shots=1).result()
                    avail.append(m)
                except Exception:
                    avail.append(f"{m} (unavailable)")
            _aer_log.setLevel(_old_level)
            self.io.writeln(f"  methods: {', '.join(avail)}")
            # GPU probe
            gpu_ok = False
            try:
                _gb = AerSimulator(method='statevector', device='GPU')
                _gb.run(transpile(_pqc_m, _gb), shots=1).result()
                gpu_ok = True
            except Exception:
                pass
            self.io.writeln(f"  devices: CPU{', GPU' if gpu_ok else ''}")
            if self._noise_model:
                self.io.writeln("  noise: active")
            return
        val = rest.strip().upper()
        if val in ('GPU', 'CPU'):
            if val == 'GPU':
                # Probe whether the Aer build actually supports GPU
                try:
                    probe = AerSimulator(method='statevector', device='GPU')
                    from qiskit import QuantumCircuit as _QC
                    _pqc = _QC(1); _pqc.h(0); _pqc.measure_all()
                    probe.run(transpile(_pqc, probe), shots=1).result()
                except Exception as _gpu_err:
                    self.io.writeln(f"?GPU NOT AVAILABLE: {_gpu_err}")
                    self.io.writeln("  Install qiskit-aer-gpu for CUDA support")
                    return
            self.sim_device = val
            self.io.writeln(f"DEVICE = {self.sim_device}")
        else:
            from qubasic_core.engine import METHOD_MAX_QUBITS
            name = rest.strip().lower()
            aliases = {'sv': 'statevector', 'mps': 'matrix_product_state',
                       'dm': 'density_matrix', 'density': 'density_matrix',
                       'auto': 'automatic', 'clifford': 'stabilizer'}
            name = aliases.get(name, name)
            if name not in METHOD_MAX_QUBITS:
                import difflib
                close = difflib.get_close_matches(name, list(METHOD_MAX_QUBITS), n=1)
                hint = f" (did you mean {close[0]}?)" if close else ""
                raise QBasicRangeError(
                    f"UNKNOWN METHOD '{rest.strip()}'{hint}; one of "
                    f"{', '.join(METHOD_MAX_QUBITS)}, or CPU/GPU")
            self.sim_method = name
            self._circuit_cache_key = None
            self.io.writeln(f"METHOD = {self.sim_method}")
            cap = METHOD_MAX_QUBITS.get(self.sim_method, MAX_QUBITS)
            if self.num_qubits > cap:
                self.num_qubits = cap
                self._invalidate_run_state()
                self.registers.clear()
                self.io.writeln(f"  (QUBITS clamped to {cap}, this method's ceiling)")

    def cmd_precision(self, rest: str = '') -> None:
        """PRECISION [SINGLE|DOUBLE] — amplitude precision: SINGLE halves the
        memory of statevector, density-matrix and numpy-engine runs, which buys
        one more qubit at the statevector wall."""
        arg = rest.strip().upper()
        if not arg:
            self.io.writeln(f"PRECISION = {self.sim_precision.upper()}")
            return
        if arg not in ('SINGLE', 'DOUBLE'):
            raise QBasicRangeError("PRECISION takes SINGLE or DOUBLE")
        new = arg.lower()
        if new != self.sim_precision:
            self.sim_precision = new
            self._circuit_cache_key = None
            self._circuit_cache = None
            if self.locc is not None:
                self.io.writeln("  (LOCC registers keep their precision until the next LOCC command)")
        self.io.writeln(f"PRECISION = {arg}")
        from qubasic_core import capacity as cap
        widest = cap.max_qubits('statevector', new)
        if widest is not None:
            self.io.writeln(f"  widest statevector run that fits now: {widest} qubits")

    def cmd_coupling(self, rest: str) -> None:
        """COUPLING linear|ring|heavyhex|full|OFF|<edges> — constrain 2-qubit connectivity.

        Sets the device coupling map used when transpiling for RUN, so circuits
        are routed (SWAPs inserted) onto a topology. 'linear' is a 1D chain,
        'ring' closes it, 'heavyhex' is a connected piece of IBM's heavy-hex
        lattice, 'full'/'OFF' removes the constraint, and an explicit list like
        '0-1, 1-2, 0-2' gives custom edges. All offline."""
        arg = rest.strip()
        self._circuit_cache_key = None
        if not arg:
            cm = self._coupling_map
            self.io.writeln(f"COUPLING = {'all-to-all' if not cm else cm}")
            return
        low = arg.lower()
        n = self.num_qubits
        if low in ('off', 'full', 'all', 'none'):
            self._coupling_map = None
            self.io.writeln("COUPLING = all-to-all (unconstrained)")
            return
        if low == 'linear':
            cm = [[i, i + 1] for i in range(n - 1)]
        elif low == 'ring':
            cm = [[i, (i + 1) % n] for i in range(n)] if n > 2 else [[0, 1]]
        elif low in ('heavyhex', 'heavy-hex'):
            from qubasic_core.resources import heavy_hex_edges
            cm = heavy_hex_edges(n)
        else:
            cm = []
            for tok in arg.replace(',', ' ').split():
                m = re.match(r'(\d+)-(\d+)$', tok)
                if not m:
                    self.io.writeln(f"?bad edge '{tok}' (use a-b)")
                    return
                cm.append([int(m.group(1)), int(m.group(2))])
            if not cm:
                self.io.writeln("?USAGE: COUPLING linear|ring|full|OFF|<a-b, ...>")
                return
        # Bidirectional edges so routing can use either direction.
        edges = []
        for a, b in cm:
            edges.append([a, b])
            if [b, a] not in cm:
                edges.append([b, a])
        self._coupling_map = edges
        named = low if low in ('linear', 'ring', 'heavyhex', 'heavy-hex') else 'custom'
        self.io.writeln(f"COUPLING = {named} ({len(cm)} edge(s))")

    def cmd_basis(self, rest: str) -> None:
        """BASIS <gate list>|OFF — restrict transpilation to a native gate set.

        E.g. BASIS rz, sx, x, cx targets a typical superconducting basis; the
        transpiler then decomposes the program into those gates for RUN. BASIS
        OFF removes the restriction."""
        arg = rest.strip()
        self._circuit_cache_key = None
        if not arg:
            self.io.writeln(f"BASIS = {self._basis_gates or 'default (all)'}")
            return
        if arg.lower() in ('off', 'none', 'default'):
            self._basis_gates = None
            self.io.writeln("BASIS = default (all gates)")
            return
        self._basis_gates = [g.strip().lower() for g in arg.replace(',', ' ').split() if g.strip()]
        self.io.writeln(f"BASIS = {self._basis_gates}")

    def _status_dict(self) -> dict:
        """Snapshot every active mode that silently changes how a line behaves.

        Returned as plain JSON-serializable values so a caller can read context
        instead of inferring it from prior commands.
        """
        from qubasic_core import capacity as cap
        cmap = self._coupling_map
        return {
            'qubits': self.num_qubits,
            'shots': self.shots,
            'method': self.sim_method,
            'device': self.sim_device,
            'precision': self.sim_precision,
            'max_statevector_qubits': cap.max_qubits('statevector', self.sim_precision),
            'seed': self._seed,
            'bit_order': ('big-endian (qubit 0 = leftmost bit)'
                          if getattr(self, '_endian_big', False)
                          else 'little-endian (qubit 0 = rightmost bit)'),
            'option_base': getattr(self, '_option_base', 0),
            'locc_mode': bool(getattr(self, 'locc_mode', False)),
            'locc_registers': (list(self.locc.names)
                               if getattr(self, 'locc_mode', False) and getattr(self, 'locc', None)
                               else []),
            'noise_active': getattr(self, '_noise_model', None) is not None,
            'noise_spec': getattr(self, '_noise_spec', None),
            'coupling_map': ([list(e) for e in cmap] if cmap else None),
            'basis_gates': getattr(self, '_basis_gates', None),
            'pending_set_state': getattr(self, '_pending_set_state', None) is not None,
            'pending_set_density': getattr(self, '_pending_set_density', None) is not None,
            'screen_mode': getattr(self, '_screen_mode', 0),
            'trace_mode': bool(getattr(self, '_trace_mode', False)),
            'bank': getattr(self, '_current_slot', 0),
            'program_lines': len(self.program),
            'variables': sum(1 for k in self.variables if not k.startswith('_')),
            'arrays': len(self.arrays),
            'subroutines': len(self.subroutines),
            'custom_gates': sorted(self._custom_gates.keys()),
        }

    def cmd_status(self, rest: str = '') -> None:
        """STATUS [JSON] — show every active mode (qubits, method, LOCC, noise,
        OPTION BASE, pending state injections, bank, ...) so behavior is readable
        rather than inferred. STATUS JSON emits the same data as JSON."""
        d = self._status_dict()
        if rest.strip().upper() == 'JSON':
            import json
            self.io.writeln(json.dumps(d, indent=2, default=str))
            return
        w = self.io.writeln
        w("  QUBASIC status")
        w(f"    qubits       {d['qubits']}    shots {d['shots']}    "
          f"method {d['method']}    device {d['device']}")
        widest = d['max_statevector_qubits']
        w(f"    precision    {d['precision']}"
          + (f"    widest statevector run that fits now: {widest} qubits" if widest else ""))
        w(f"    seed         {d['seed']}")
        w(f"    bit order    {d['bit_order']}")
        w(f"    option base  {d['option_base']}")
        if d['locc_mode']:
            w(f"    LOCC mode    on    registers {d['locc_registers']}")
        w(f"    noise        {('on  ' + str(d['noise_spec'])) if d['noise_active'] else 'off'}")
        if d['coupling_map']:
            w(f"    coupling     {d['coupling_map']}")
        if d['basis_gates']:
            w(f"    basis gates  {d['basis_gates']}")
        if d['pending_set_state']:
            w("    pending      SET_STATE applies on next RUN")
        if d['pending_set_density']:
            w("    pending      SET_DENSITY applies on next RUN")
        if d['screen_mode']:
            w(f"    screen mode  {d['screen_mode']}")
        if d['trace_mode']:
            w("    trace        on")
        w(f"    bank         {d['bank']}")
        w(f"    program      {d['program_lines']} lines, {d['variables']} vars, "
          f"{d['arrays']} arrays, {d['subroutines']} subs")
        if d['custom_gates']:
            w(f"    custom gates {d['custom_gates']}")

    def cmd_seed(self, rest: str) -> None:
        """SEED <n> — set deterministic random seed for reproducible results.
        SEED OFF — return to non-deterministic mode."""
        if not rest.strip() or rest.strip().upper() == 'OFF':
            self._seed = None
            self.io.writeln("SEED OFF (non-deterministic)")
            return
        s = int(rest.strip())
        self._seed = s
        np.random.seed(s)
        self._rng.seed(s)
        self.io.writeln(f"SEED = {s}")

    def cmd_version(self, rest: str = '') -> None:
        """VERSION — print build ID, simulator versions, and feature flags."""
        from qubasic_core import __version__
        import qiskit
        import qiskit_aer
        self.io.writeln(f"  QUBASIC {__version__}")
        self.io.writeln(f"  Qiskit {qiskit.__version__}")
        self.io.writeln(f"  Qiskit Aer {qiskit_aer.__version__}")
        try:
            import numpy as _np
            self.io.writeln(f"  NumPy {_np.__version__}")
        except ImportError:
            pass
        self.io.writeln(f"  Python {sys.version.split()[0]}")
        self.io.writeln(f"  Platform {sys.platform}")
        # Feature flags
        flags = []
        # GPU
        try:
            _probe = AerSimulator(method='statevector', device='GPU')
            from qiskit import QuantumCircuit as _QC
            _pqc = _QC(1); _pqc.h(0); _pqc.measure_all()
            _probe.run(transpile(_pqc, _probe), shots=1).result()
            flags.append('GPU')
        except Exception:
            pass
        try:
            import plotille  # noqa: F401
            flags.append('charts')
        except ImportError:
            pass
        try:
            import pyreadline3  # noqa: F401
            flags.append('readline')
        except ImportError:
            try:
                import readline  # noqa: F401
                flags.append('readline')
            except ImportError:
                pass
        self.io.writeln(f"  Features: {', '.join(flags) if flags else 'base'}")
        self.io.writeln(f"  Device: {self.sim_device}  Method: {self.sim_method}")
        if self._noise_model:
            self.io.writeln(f"  Noise: active (depol_p={self._noise_depol_p})")
