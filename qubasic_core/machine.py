"""QUBASIC machine settings: the configuration commands (QUBITS,
SHOTS, METHOD, PRECISION, COUPLING, BASIS, STATUS, SEED), the memory map
(PEEK, POKE, SYS, MAP) and profiling (PROFILE, STATS).
"""

from __future__ import annotations

import math
import os
import re
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qubasic_core.engine import MAX_QUBITS, RAM_BUDGET_FRACTION, QBasicRangeError
from qubasic_core.host import HostState

# ---------------------------------------------------------------------------
# QUBASIC configuration commands: QUBITS, SHOTS, METHOD, PRECISION, COUPLING, BASIS, STATUS, SEED, VERSION.
# ---------------------------------------------------------------------------

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
            from qubasic_core.engine import QBasicRangeError
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
        from qubasic_core import simtools as cap
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
            import logging as _logging

            from qiskit import QuantumCircuit as _QC
            _pqc_m = _QC(1)
            _pqc_m.h(0)
            _pqc_m.measure_all()
            _pqc_u = _QC(1)
            _pqc_u.h(0)  # no measure for unitary/superop
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
                    _pqc = _QC(1)
                    _pqc.h(0)
                    _pqc.measure_all()
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
        from qubasic_core import simtools as cap
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
            from qubasic_core.simtools import heavy_hex_edges
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
        from qubasic_core import simtools as cap
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
                               if self.locc_mode and self.locc is not None else []),
            'noise_active': getattr(self, '_noise_model', None) is not None,
            'noise_spec': getattr(self, '_noise_spec', None),
            'coupling_map': ([list(e) for e in cmap] if cmap else None),
            'basis_gates': getattr(self, '_basis_gates', None),
            'pending_set_state': getattr(self, '_pending_set_state', None) is not None,
            'pending_set_density': getattr(self, '_pending_set_density', None) is not None,
            'screen_mode': getattr(self, '_screen_mode', 0),
            'graphics_screen': self._canvas.mode if self._canvas is not None else 0,
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
            w(f"    display      {self._DISPLAY_NAMES[d['screen_mode']]}")
        if d['graphics_screen']:
            w(f"    graphics     SCREEN {d['graphics_screen']}")
        if d['trace_mode']:
            w("    trace        on")
        w(f"    bank         {d['bank']}")
        w(f"    program      {d['program_lines']} lines, {d['variables']} vars, "
          f"{d['arrays']} arrays, {d['subroutines']} subs")
        if d['custom_gates']:
            w(f"    custom gates {d['custom_gates']}")

    def cmd_seed(self, rest: str) -> None:
        """SEED n | OFF — fix the random seed so runs repeat, or release it."""
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
        import qiskit
        import qiskit_aer

        from qubasic_core import __version__
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
            _pqc = _QC(1)
            _pqc.h(0)
            _pqc.measure_all()
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


# ---------------------------------------------------------------------------
# QUBASIC memory map — PEEK, POKE, SYS, USR, WAIT, DUMP, MAP, MONITOR, CATALOG.
# ---------------------------------------------------------------------------

# ═══════════════════════════════════════════════════════════════════════
# Memory Map
# ═══════════════════════════════════════════════════════════════════════
#   $0000-$003F  Zero Page (64 slots, SYS parameter passing)
#   $0100-$01FF  Qubit State, 8 addresses for qubit N at $0100+8*N (qubits 0-31):
#                  +0 P(|1⟩)   +1 Bloch X  +2 Bloch Y  +3 Bloch Z
#                  +4 Re(α)    +5 Im(α)    +6 Re(β)    +7 Im(β)
#   $D000-$D00C  QPU Config (read/write)
#   $D010-$D014  QPU Status (read-only)
#   $D100+2*N    Per-qubit noise for qubit N: +0 type, +1 parameter
#   $E000-$E0B0  SYS Routine Table (built-in algorithms)
#   $F000-$FFFF  User SYS Routines
# ═══════════════════════════════════════════════════════════════════════

QUBIT_BLOCK = 8

CFG_NAMES = {
    0xD000: 'num_qubits', 0xD001: 'shots', 0xD002: 'sim_method',
    0xD003: 'sim_device', 0xD004: 'noise_type', 0xD005: 'noise_param',
    0xD006: 'max_iterations', 0xD007: 'screen_mode',
    # Performance tuning (Aer backend options)
    0xD008: 'fusion_enable',        # 0/1
    0xD009: 'mps_truncation',       # float threshold
    0xD00A: 'sv_parallel_threshold', # int
    0xD00B: 'es_approx_error',      # float (extended stabilizer)
    0xD00C: 'precision',            # 0 = double, 1 = single
}

STS_NAMES = {
    0xD010: 'gate_count', 0xD011: 'circuit_depth', 0xD012: 'run_time_ms',
    0xD013: 'total_probability', 0xD014: 'entanglement_entropy',
}

SIM_METHODS_FWD = {0: 'automatic', 1: 'statevector', 2: 'stabilizer',
                   3: 'matrix_product_state', 4: 'density_matrix'}
SIM_METHODS_REV = {v: k for k, v in SIM_METHODS_FWD.items()}
SIM_DEVICES_FWD = {0: 'CPU', 1: 'GPU'}
SIM_DEVICES_REV = {v: k for k, v in SIM_DEVICES_FWD.items()}

SYS_ROUTINES = {
    0xE000: 'BELL',      0xE010: 'GHZ',       0xE020: 'QFT',
    0xE030: 'GROVER',    0xE040: 'TELEPORT',   0xE050: 'DEUTSCH',
    0xE060: 'BERNSTEIN', 0xE070: 'SUPERDENSE', 0xE080: 'RANDOM',
    0xE090: 'STRESS',    0xE0A0: 'LOCC-TELEPORT', 0xE0B0: 'LOCC-COORD',
}


class MemoryMixin(HostState):
    """Memory-mapped PEEK/POKE, SYS, USR, WAIT, DUMP, MAP, MONITOR, CATALOG.

    Requires: TerminalProtocol — uses self.num_qubits, self.shots,
    self.sim_method, self.sim_device, self._noise_model, self._max_iterations,
    self.last_sv, self.variables, self.cmd_demo(), self.eval_expr().
    """

    def _init_memory(self) -> None:
        self._zero_page: list[float] = [0.0] * 64
        self._status: dict[int, float] = {a: 0.0 for a in STS_NAMES}
        self._status[0xD013] = 1.0
        self._user_sys: dict[int, str] = {}
        self._screen_mode: int = 0
        # Per-qubit noise: $D100 + qubit*2 = noise_type, +1 = noise_param
        # noise_type: 0=none, 1=depolarizing, 2=amplitude_damping, 3=phase_flip
        self._qubit_noise: dict[int, tuple[int, float]] = {}  # qubit -> (type, param)

    def _update_status(self, **kw: float) -> None:
        for addr, name in STS_NAMES.items():
            if name in kw:
                self._status[addr] = float(kw[name])

    def _state_status(self, addr: int) -> float:
        """$D013 (total probability) and $D014 (entanglement entropy of qubit
        0 against the rest, in bits, the partition ENTROPY 0 reports), read
        from the current state on demand."""
        sv = self._active_sv
        if sv is None:
            return 1.0 if addr == 0xD013 else 0.0
        flat = np.asarray(sv).reshape(-1)
        if addr == 0xD013:
            return float(np.vdot(flat, flat).real)
        n = int(flat.size).bit_length() - 1
        if n < 2:
            return 0.0
        from qubasic_core.simtools import entropy_bits, qubit_rdm
        lam = np.linalg.eigvalsh(qubit_rdm(flat, 0, n))
        return entropy_bits(lam)

    # ── PEEK ───────────────────────────────────────────────────────────

    def _peek(self, addr: float) -> float:
        a = int(addr)
        if 0x0000 <= a <= 0x003F:
            return self._zero_page[a]
        if 0x0100 <= a <= 0x01FF:
            return self._peek_qubit(a)
        if 0xD100 <= a <= 0xD1FF:
            qubit = (a - 0xD100) // 2
            field = (a - 0xD100) % 2
            ntype, nparam = self._qubit_noise.get(qubit, (0, 0.0))
            return float(ntype) if field == 0 else nparam
        if a in CFG_NAMES:
            return self._peek_config(a)
        if a in (0xD013, 0xD014):
            return self._state_status(a)
        if a in STS_NAMES:
            return self._status.get(a, 0.0)
        return 0.0

    def _peek_qubit(self, addr: int) -> float:
        off = addr - 0x0100
        qubit, field = off // QUBIT_BLOCK, off % QUBIT_BLOCK
        if qubit >= self.num_qubits:
            return 0.0
        sv = self.last_sv
        if sv is None:
            return 0.0
        sv_flat = np.asarray(sv).reshape(-1)
        if sv_flat.size != 2 ** self.num_qubits:
            return 0.0  # stale statevector from a different qubit count
        from qubasic_core.simtools import qubit_rdm
        r = qubit_rdm(sv_flat, qubit, self.num_qubits)
        rho_00 = float(r[0, 0].real)
        rho_11 = float(r[1, 1].real)
        # Bloch X = 2 Re(rho_10) and Bloch Y = +2 Im(rho_10).
        rho_10 = complex(r[1, 0])
        # Recover amplitudes with alpha as the real phase reference; beta then
        # carries the relative phase (real and imaginary parts both exposed).
        alpha = np.sqrt(max(0.0, rho_00))
        if alpha > 1e-12:
            beta = rho_10 / alpha
        else:
            beta = complex(np.sqrt(max(0.0, rho_11)))
        return [rho_11,
                float(2 * rho_10.real),
                float(2 * rho_10.imag),
                float(rho_00 - rho_11),
                float(alpha),
                0.0,
                float(beta.real),
                float(beta.imag)][field] if field < 8 else 0.0

    def _peek_config(self, addr: int) -> float:
        n = CFG_NAMES[addr]
        if n == 'num_qubits':
            return float(self.num_qubits)
        if n == 'shots':
            return float(self.shots)
        if n == 'sim_method':
            return float(SIM_METHODS_REV.get(self.sim_method, 0))
        if n == 'sim_device':
            return float(SIM_DEVICES_REV.get(self.sim_device, 0))
        if n == 'noise_type':
            return 0.0 if self._noise_model is None else 1.0
        if n == 'noise_param':
            return float(self._noise_depol_p) if self._noise_model is not None else 0.0
        if n == 'max_iterations':
            return float(self._max_iterations)
        if n == 'screen_mode':
            return float(self._canvas.mode if self._canvas is not None else self._screen_mode)
        if n == 'fusion_enable':
            return float(getattr(self, '_fusion_enable', 1))
        if n == 'mps_truncation':
            return float(getattr(self, '_mps_truncation', 1e-16))
        if n == 'sv_parallel_threshold':
            return float(getattr(self, '_sv_parallel_threshold', 14))
        if n == 'es_approx_error':
            return float(getattr(self, '_es_approx_error', 0.05))
        if n == 'precision':
            return 1.0 if self.sim_precision == 'single' else 0.0
        return 0.0

    # ── POKE ───────────────────────────────────────────────────────────

    def _poke(self, addr: float, value: float) -> None:
        a, v = int(addr), float(value)
        if 0x0000 <= a <= 0x003F:
            if not isinstance(value, (int, float)):
                self.io.writeln(f"?TYPE: zero page expects numeric value, got {type(value).__name__}")
                return
            if abs(v) > 1e15:
                self.io.writeln(f"?WARNING: suspiciously large zero-page value: {v}")
            self._zero_page[a] = v
            return
        if a == 0xD000:
            from qubasic_core.engine import METHOD_MAX_QUBITS
            ceiling = METHOD_MAX_QUBITS.get(self.sim_method, 32)
            new_n = max(1, min(ceiling, int(v)))
            if new_n != self.num_qubits:
                self._invalidate_run_state()
                self.registers.clear()
            self.num_qubits = new_n
        elif a == 0xD00C:
            self.sim_precision = 'single' if int(v) else 'double'
            self._circuit_cache_key = None
        elif a == 0xD001:
            self.shots = max(1, int(v))
        elif a == 0xD002:
            self.sim_method = SIM_METHODS_FWD.get(int(v), 'automatic')
        elif a == 0xD003:
            self.sim_device = SIM_DEVICES_FWD.get(int(v), 'CPU')
        elif a == 0xD006:
            self._max_iterations = max(1, int(v))
        elif a == 0xD007:
            self.cmd_screen(str(int(v)))          # as SCREEN n: graphics, text or a display mode
        elif a == 0xD008:
            self._fusion_enable = bool(int(v))
        elif a == 0xD009:
            self._mps_truncation = v
        elif a == 0xD00A:
            self._sv_parallel_threshold = int(v)
        elif a == 0xD00B:
            self._es_approx_error = v
        elif 0xD100 <= a <= 0xD1FF:
            qubit = (a - 0xD100) // 2
            field = (a - 0xD100) % 2
            ntype, nparam = self._qubit_noise.get(qubit, (0, 0.0))
            if field == 0:
                self._qubit_noise[qubit] = (int(v), nparam)
            else:
                self._qubit_noise[qubit] = (ntype, v)
        elif 0xD010 <= a <= 0xD01F:
            self.io.writeln(f"?READ-ONLY: ${a:04X}")
        elif 0x0100 <= a <= 0x01FF:
            # Bloch sphere POKE — prepare qubit state via memory map
            import math
            off = a - 0x0100
            qubit, field = off // QUBIT_BLOCK, off % QUBIT_BLOCK
            if qubit >= self.num_qubits:
                self.io.writeln(f"?QUBIT {qubit} OUT OF RANGE")
                return
            if field == 0:
                # POKE P(|1⟩) — set probability via RY rotation
                p1 = max(0.0, min(1.0, v))
                theta = 2 * math.asin(math.sqrt(p1))
                # This requires a circuit context — store for next RUN
                self._poke_state_prep[qubit] = ('RY', theta)
                self.io.writeln(f"POKE q{qubit}: P(1)={p1:.4f} -> RY({theta:.4f})")
            elif field in (1, 2, 3):
                # Bloch x, y, z — store target vector
                labels = {1: 'Bx', 2: 'By', 3: 'Bz'}
                self._poke_state_prep[(qubit, labels[field])] = v
                self.io.writeln(f"POKE q{qubit}.{labels[field]} = {v:.4f}")
            else:
                self.io.writeln(f"?FIELD {field} NOT WRITABLE (use 0=P(1), 1-3=Bloch)")

    # ── Commands ───────────────────────────────────────────────────────

    # Address ranges that PEEK/POKE actually back with state.
    _MAPPED_RANGES = ((0x0000, 0x003F), (0x0100, 0x01FF), (0xD000, 0xD01F),
                      (0xD100, 0xD1FF), (0xE000, 0xFFFF))

    def _is_mapped(self, addr: float) -> bool:
        a = int(addr)
        return any(lo <= a <= hi for lo, hi in self._MAPPED_RANGES)

    def cmd_peek(self, rest: str) -> None:
        """PEEK addr — read from memory-mapped address."""
        if not rest.strip():
            self.io.writeln("?USAGE: PEEK <addr>")
            return
        addr = self._eval_int(rest.strip())
        val = self._peek(addr)
        self.io.writeln(f"  ${int(addr):04X} = {val}")
        if not self._is_mapped(addr):
            self.io.writeln(f"  (note: ${int(addr):04X} is unmapped — reads as 0)")

    def cmd_poke(self, rest: str) -> None:
        """POKE addr, value — write to memory-mapped address."""
        from qubasic_core.parser import RE_POKE
        m = RE_POKE.match(f"POKE {rest}")
        if not m:
            self.io.writeln("?USAGE: POKE <addr>, <value>")
            return
        addr = self._eval_int(m.group(1))
        val = self.eval_expr(m.group(2))
        if not self._is_mapped(addr):
            self.io.writeln(f"  (note: ${int(addr):04X} is unmapped — write ignored)")
            return
        self._poke(addr, val)

    def _emit_poke_state_prep(self, qc: QuantumCircuit) -> None:
        """Apply qubit state preparation requested via POKE to the $0100 block.

        Field 0 (P(|1>)) stored an RY angle; Bloch fields stored a target
        (x, y, z), prepared here as the closest pure single-qubit state. Called
        at the start of circuit build so a poked qubit starts in that state.
        """
        prep = getattr(self, '_poke_state_prep', None)
        if not prep:
            return
        import numpy as np
        bloch: dict = {}
        for key, val in prep.items():
            if isinstance(key, int):
                if key < qc.num_qubits and isinstance(val, tuple) and val[0] == 'RY':
                    qc.ry(float(val[1]), key)
            elif isinstance(key, tuple):
                q, axis = key
                bloch.setdefault(q, {})[axis] = float(val)
        for q, comps in bloch.items():
            if q >= qc.num_qubits:
                continue
            x = comps.get('Bx', 0.0)
            y = comps.get('By', 0.0)
            z = comps.get('Bz', 0.0)
            r = (x * x + y * y + z * z) ** 0.5
            if r < 1e-9:
                continue
            theta = float(np.arccos(max(-1.0, min(1.0, z / r))))
            phi = float(np.arctan2(y, x))
            qc.ry(theta, q)
            if abs(phi) > 1e-12:
                qc.rz(phi, q)

    def cmd_sys(self, rest: str) -> None:
        """SYS addr — execute a built-in or user routine."""
        rest = rest.strip().upper()
        if rest.startswith('INSTALL'):
            parts = rest[7:].strip().split(',', 1)
            if len(parts) != 2:
                self.io.writeln("?USAGE: SYS INSTALL <addr>, <subroutine_name>")
                return
            addr = int(self.eval_expr(parts[0].strip()))
            name = parts[1].strip()
            if not (0xF000 <= addr <= 0xFFFF):
                self.io.writeln(f"?USER SYS RANGE: $F000-$FFFF (got ${addr:04X})")
                return
            self._user_sys[addr] = name
            self.io.writeln(f"INSTALLED {name} AT ${addr:04X}")
            return
        addr = self._eval_int(rest)
        if addr in SYS_ROUTINES:
            if SYS_ROUTINES[addr] == 'STRESS':
                self.io.writeln("  (note: STRESS runs a 20-qubit circuit; this can take a while)")
            self.cmd_demo(SYS_ROUTINES[addr])
        elif addr in self._user_sys:
            name = self._user_sys[addr]
            if name in self.subroutines:
                sub = self.subroutines[name]
                body = sub['body'] if isinstance(sub, dict) else sub
                for stmt in body:
                    self.process(stmt)
            else:
                self.io.writeln(f"?UNDEFINED ROUTINE: {name}")
        else:
            self.io.writeln(f"?NO ROUTINE AT ${addr:04X}")

    # File-writing commands blocked from USR execution to prevent
    # user-defined routines from performing uncontrolled I/O.
    _USR_BLOCKED_CMDS = frozenset({'SAVE', 'EXPORT', 'CSV', 'OPEN'})

    def _usr_fn(self, addr: float) -> float:
        """USR(addr) — execute routine, return last measurement result."""
        a = int(addr)
        if a in SYS_ROUTINES:
            self.cmd_demo(SYS_ROUTINES[a])
        elif a in self._user_sys:
            name = self._user_sys[a]
            if name in self.subroutines:
                sub = self.subroutines[name]
                body = sub['body'] if isinstance(sub, dict) else sub
                for stmt in body:
                    first_word = stmt.split(None, 1)[0].upper() if stmt.strip() else ''
                    if first_word in self._USR_BLOCKED_CMDS:
                        self.io.writeln(f"  ?BLOCKED IN USR: {first_word}")
                        continue
                    self.process(stmt)
        if self.last_counts:
            counts = self.last_counts
            top = max(counts, key=lambda k: counts[k])
            return float(int(top, 2))
        return 0.0

    def cmd_wait(self, rest: str) -> None:
        """WAIT addr, mask[, value[, timeout]] — block until (PEEK(addr) AND mask) == value."""
        parts = [p.strip() for p in rest.split(',')]
        if len(parts) < 2:
            self.io.writeln("?USAGE: WAIT <addr>, <mask>[, <value>[, <timeout>]]")
            return
        addr = int(self.eval_expr(parts[0]))
        mask = int(self.eval_expr(parts[1]))
        target = int(self.eval_expr(parts[2])) if len(parts) > 2 else mask
        timeout = float(self.eval_expr(parts[3])) if len(parts) > 3 else 30.0
        t0 = time.time()
        try:
            while time.time() - t0 < timeout:
                val = int(self._peek(addr))
                if (val & mask) == target:
                    return
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.io.writeln("?WAIT INTERRUPTED")
            return
        self.io.writeln("?WAIT TIMEOUT")

    _SYS_BACKENDS = {
        'BELL': 'Aer/statevector', 'GHZ': 'Aer/statevector',
        'QFT': 'Aer/statevector', 'GROVER': 'Aer/statevector',
        'TELEPORT': 'Aer/statevector', 'DEUTSCH': 'Aer/statevector',
        'BERNSTEIN': 'Aer/statevector', 'SUPERDENSE': 'Aer/statevector',
        'RANDOM': 'Aer/statevector', 'STRESS': 'Aer/automatic',
        'LOCC-TELEPORT': 'LOCC/numpy-joint', 'LOCC-COORD': 'LOCC/numpy-split',
    }

    def cmd_catalog(self) -> None:
        """CATALOG — list all SYS routines with addresses and backends."""
        self.io.writeln("\n  Built-in SYS Routines:")
        for addr, name in sorted(SYS_ROUTINES.items()):
            backend = self._SYS_BACKENDS.get(name, 'Aer')
            self.io.writeln(f"    ${addr:04X}  {name:20s} [{backend}]")
        if self._user_sys:
            self.io.writeln("\n  User SYS Routines:")
            for addr, name in sorted(self._user_sys.items()):
                self.io.writeln(f"    ${addr:04X}  {name}")
        self.io.writeln('')

    def cmd_dump(self, rest: str = '') -> None:
        """DUMP [start] [end] — hex dump of memory map."""
        parts = rest.split()
        start = int(self.eval_expr(parts[0])) if parts else 0x0000
        end = int(self.eval_expr(parts[1])) if len(parts) > 1 else start + 0x3F
        self.io.writeln('')
        for row_start in range(start, end + 1, 16):
            vals = [self._peek(row_start + i) for i in range(16) if row_start + i <= end]
            hex_part = ' '.join(f'{int(v) & 0xFF:02X}' if abs(v) < 256 else f'{v:.4g}'
                               for v in vals)
            self.io.writeln(f"  ${row_start:04X}: {hex_part}")
        self.io.writeln('')

    def cmd_map(self) -> None:
        """MAP — print the full memory map with current values."""
        self.io.writeln("\n  Memory Map:")
        self.io.writeln("  $0000-$003F  Zero Page        (64 slots)")
        nz = sum(1 for v in self._zero_page if v != 0)
        self.io.writeln(f"               {nz} non-zero values")
        self.io.writeln(f"  $0100-$01FF  Qubit State      ({self.num_qubits} qubits, 8 addr/qubit)")
        if self.last_sv is not None:
            for q in range(min(self.num_qubits, 4)):
                p1 = self._peek(0x0100 + q * 8)
                bz = self._peek(0x0100 + q * 8 + 3)
                self.io.writeln(f"               q{q}: P(1)={p1:.3f} Bz={bz:.3f}")
            if self.num_qubits > 4:
                self.io.writeln(f"               ... ({self.num_qubits - 4} more)")
        self.io.writeln("  $D000-$D00C  QPU Config")
        for addr, name in sorted(CFG_NAMES.items()):
            val = self._peek_config(addr)
            self.io.writeln(f"               ${addr:04X} {name:16s} = {val}")
        self.io.writeln("  $D010-$D014  QPU Status (read-only)")
        for addr, name in sorted(STS_NAMES.items()):
            val = self._peek(addr)
            self.io.writeln(f"               ${addr:04X} {name:20s} = {val}")
        self.io.writeln(f"  $E000-$E0B0  SYS Routines     ({len(SYS_ROUTINES)} built-in)")
        self.io.writeln(f"  $F000-$FFFF  User SYS         ({len(self._user_sys)} installed)")
        self.io.writeln('')

    def cmd_monitor(self) -> None:
        """MONITOR — interactive hex monitor for PEEK/POKE."""
        self.io.writeln("MONITOR — type address to PEEK, addr=val to POKE, Q to quit")
        while True:
            try:
                line = self.io.read_line('* ').strip()
            except (KeyboardInterrupt, EOFError):
                self.io.writeln('')
                break
            if not line or line.upper() == 'Q':
                break
            if '=' in line:
                parts = line.split('=', 1)
                try:
                    addr = int(self.eval_expr(parts[0].strip()))
                    val = self.eval_expr(parts[1].strip())
                    self._poke(addr, val)
                    self.io.writeln(f"  ${addr:04X} <- {val}")
                except Exception as e:
                    self.io.writeln(f"?{e}")
            else:
                try:
                    addr = int(self.eval_expr(line))
                    val = self._peek(addr)
                    self.io.writeln(f"  ${addr:04X} = {val}")
                except Exception as e:
                    self.io.writeln(f"?{e}")
        self.io.writeln("MONITOR OFF")


# ---------------------------------------------------------------------------
# QUBASIC profiler — profile mode, gate count, depth tracking, statistics accumulator.
# ---------------------------------------------------------------------------

MAX_STATS_RUNS = 10000


class ProfilerMixin(HostState):
    """Profiling and statistics for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.last_counts,
    self.shots, self.cmd_run().
    """

    def _init_profiler(self) -> None:
        self._profile_mode: bool = False
        self._profile_data: dict[int, dict[str, float]] = {}
        self._profile_start: float = 0.0
        self._stats_runs: list[dict[str, int]] = []

    # ── Profile mode ───────────────────────────────────────────────────

    def cmd_profile(self, rest: str = '') -> None:
        """PROFILE [ON|OFF|SHOW] — toggle profiling or show results."""
        arg = rest.strip().upper()
        if arg == 'ON':
            self._profile_mode = True
            self._profile_data.clear()
            self.io.writeln("PROFILE ON")
        elif arg == 'OFF':
            self._profile_mode = False
            self.io.writeln("PROFILE OFF")
        elif arg == 'SHOW' or not arg:
            self._show_profile()
        else:
            self.io.writeln("?USAGE: PROFILE [ON|OFF|SHOW]")

    def _profile_line_start(self, line_num: int) -> None:
        if self._profile_mode:
            self._profile_start = time.perf_counter()

    def _profile_line_end(self, line_num: int, gates: int = 0) -> None:
        if self._profile_mode:
            dt = (time.perf_counter() - self._profile_start) * 1000  # ms
            if line_num not in self._profile_data:
                self._profile_data[line_num] = {'time_ms': 0.0, 'calls': 0, 'gates': 0}
            entry = self._profile_data[line_num]
            entry['time_ms'] += dt
            entry['calls'] += 1
            entry['gates'] += gates

    def _show_profile(self) -> None:
        if not self._profile_data:
            self.io.writeln("  No profile data (PROFILE ON, then RUN)")
            return
        self.io.writeln("\n  Profile Results:")
        self.io.writeln(f"  {'Line':>6}  {'Time(ms)':>10}  {'%':>5}  {'Calls':>6}  {'Gates':>6}  Source")
        total_time = sum(d['time_ms'] for d in self._profile_data.values())
        for ln in sorted(self._profile_data.keys()):
            d = self._profile_data[ln]
            src = self.program.get(ln, '')[:40]
            pct = 100 * d['time_ms'] / total_time if total_time > 0 else 0
            self.io.writeln(f"  {ln:>6}  {d['time_ms']:>10.2f}  {pct:>5.1f}  {d['calls']:>6}  "
                            f"{d['gates']:>6}  {src}")
        self.io.writeln(f"\n  Total: {total_time:.2f} ms")
        self.io.writeln('')

    # ── Statistics accumulator ─────────────────────────────────────────

    def cmd_stats(self, rest: str = '') -> None:
        """STATS [N|SHOW|CLEAR|CSV <file>] — run the program N times and summarize the counts."""
        arg = rest.strip().upper()
        if arg == 'CLEAR':
            self._stats_runs.clear()
            self.io.writeln("STATS CLEARED")
            return
        if arg == 'SHOW' or not arg:
            self._show_stats()
            return
        if arg.startswith('CSV'):
            self._stats_export_csv(rest[3:].strip())
            return
        # STATS N — run N trials
        try:
            n = int(arg)
        except ValueError:
            self.io.writeln("?USAGE: STATS [N|SHOW|CLEAR|CSV <file>]")
            return
        if n < 1:
            self.io.writeln("?STATS needs at least 1 run")
            return
        self.io.writeln(f"\nRunning {n} trials...")
        old_io = self.io
        old_stdout = sys.stdout
        devnull = open(os.devnull, 'w')  # opened once for all trials
        base_seed = self._seed
        try:
            for trial in range(n):
                # A fixed SEED would make every trial identical, so vary it per
                # trial (and invalidate the cache) while keeping reproducibility.
                if base_seed is not None:
                    self._seed = base_seed + trial
                    np.random.seed(base_seed + trial)
                    self._circuit_cache_key = None
                sys.stdout = devnull
                try:
                    with self._silenced():
                        self.cmd_run()
                finally:
                    sys.stdout = old_stdout
                if self.last_counts:
                    if len(self._stats_runs) >= MAX_STATS_RUNS:
                        self.io.writeln(f"?STATS: run limit ({MAX_STATS_RUNS}) reached, stopping collection")
                        break
                    self._stats_runs.append(dict(self.last_counts))
                # Spinner only in an interactive terminal: carriage-return
                # animation turns into junk in piped or captured output.
                if n > 10 and (trial + 1) % (n // 10) == 0 and sys.stdout.isatty():
                    from qubasic_core.engine import quantum_spin
                    spin = quantum_spin(trial)
                    self.io.write(f"  {spin} {100 * (trial + 1) // n}%..." + '\r')
        finally:
            sys.stdout = old_stdout
            self.io = old_io
            devnull.close()
            if base_seed is not None:
                self._seed = base_seed
        if n > 10 and sys.stdout.isatty():
            self.io.write(" " * 30 + '\r')
        self.io.writeln(f"Collected {len(self._stats_runs)} runs ({n} trials)")

    def _show_stats(self) -> None:
        if not self._stats_runs:
            self.io.writeln("  No statistics collected (STATS N to run N trials)")
            return
        n = len(self._stats_runs)
        # Aggregate: count how often each state appears across runs
        state_totals: dict[str, list[int]] = {}
        for run in self._stats_runs:
            for state, count in run.items():
                if state not in state_totals:
                    state_totals[state] = []
                state_totals[state].append(count)
            # States not seen in this run get 0
            for state in state_totals:
                if state not in run:
                    state_totals[state].append(0)
        # Pad lists to same length.  This is O(states * n) but n is bounded
        # by MAX_STATS_RUNS (enforced during collection), so it stays tractable.
        for state in state_totals:
            while len(state_totals[state]) < n:
                state_totals[state].append(0)
        # Compute per-run shot totals for probability conversion
        shot_totals = [sum(run.values()) for run in self._stats_runs]
        avg_shots = sum(shot_totals) / len(shot_totals) if shot_totals else 1
        self.io.writeln(f"\n  Statistics over {n} runs ({avg_shots:.0f} shots/run):")
        self.io.writeln(f"  {'State':>10}  {'Mean':>8}  {'StdDev':>8}  {'Min':>6}  {'Max':>6}  {'P(mean)':>8}")
        for state in sorted(state_totals.keys()):
            vals = state_totals[state]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(variance)
            prob = mean / avg_shots if avg_shots > 0 else 0
            self.io.writeln(f"  |{self._bits(state)}\u27E9  {mean:>8.1f}  {std:>8.2f}  {min(vals):>6}  {max(vals):>6}  {prob:>7.4f}")
        self.io.writeln('')

    def _stats_export_csv(self, path: str) -> None:
        """Export accumulated stats to CSV."""
        if not self._stats_runs:
            self.io.writeln("  No statistics collected")
            return
        if not path:
            self.io.writeln("?USAGE: STATS CSV <filename>")
            return
        try:
            path = self._sanitize_path(path)
        except (ValueError, AttributeError) as e:
            self.io.writeln(f"?STATS CSV ERROR: {e}")
            return
        n = len(self._stats_runs)
        state_totals: dict[str, list[int]] = {}
        for run in self._stats_runs:
            for state, count in run.items():
                if state not in state_totals:
                    state_totals[state] = []
                state_totals[state].append(count)
            for state in state_totals:
                if state not in run:
                    state_totals[state].append(0)
        for state in state_totals:
            while len(state_totals[state]) < n:
                state_totals[state].append(0)
        shot_totals = [sum(run.values()) for run in self._stats_runs]
        avg_shots = sum(shot_totals) / len(shot_totals) if shot_totals else 1
        lines = ['state,mean,stddev,min,max,probability']
        for state in sorted(state_totals.keys()):
            vals = state_totals[state]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(variance)
            prob = mean / avg_shots if avg_shots > 0 else 0
            lines.append(f"{state},{mean:.2f},{std:.4f},{min(vals)},{max(vals)},{prob:.6f}")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        self.io.writeln(f"STATS EXPORTED to {path} ({n} runs, {len(state_totals)} states)")
