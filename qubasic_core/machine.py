"""QUBASIC machine settings: the configuration commands (QUBITS,
SHOTS, METHOD, PRECISION, COUPLING, BASIS, STATUS, SEED), the memory map
(PEEK, POKE, SYS, MAP), profiling (PROFILE, STATS) and the built-in demos
that DEMO and the SYS routines run.
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qubasic_core.engine import MAX_QUBITS, RAM_BUDGET_FRACTION, QBasicRangeError
from qubasic_core.host import SimulationHost
from qubasic_core.parser import RE_POKE

# ---------------------------------------------------------------------------
# QUBASIC configuration commands: QUBITS, SHOTS, METHOD, PRECISION, COUPLING, BASIS, STATUS, SEED, VERSION.
# ---------------------------------------------------------------------------

class ConfigMixin(SimulationHost):
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
            if self.noise.model:
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
            'option_base': self.symbols.option_base,
            'locc_mode': bool(getattr(self, 'locc_mode', False)),
            'locc_registers': (list(self.locc.names)
                               if self.locc_mode and self.locc is not None else []),
            'noise_active': self.noise.model is not None,
            'noise_spec': self.noise.spec,
            'coupling_map': ([list(e) for e in cmap] if cmap else None),
            'basis_gates': getattr(self, '_basis_gates', None),
            'pending_set_state': getattr(self, '_pending_set_state', None) is not None,
            'pending_set_density': getattr(self, '_pending_set_density', None) is not None,
            'screen_mode': getattr(self, '_screen_mode', 0),
            'graphics_screen': self.graphics.canvas.mode if self.graphics.canvas is not None else 0,
            'trace_mode': bool(self.debug.trace),
            'bank': self.editor.slot,
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
        if self.noise.model:
            self.io.writeln(f"  Noise: active (depol_p={self.noise.depol_p})")


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


@dataclass
class AerTuning:
    """Aer's tuning options as POKE sets them at $D008-$D00B; None leaves
    Aer's own default, which PEEK reports."""

    fusion_enable: bool | None = None
    mps_truncation: float | None = None
    sv_parallel_threshold: int | None = None
    es_approx_error: float | None = None

    DEFAULTS = {'fusion_enable': 1.0, 'mps_truncation': 1e-16, 'sv_parallel_threshold': 14.0,
                'es_approx_error': 0.05}
    _AER_NAMES = {'fusion_enable': 'fusion_enable',
                  'mps_truncation': 'matrix_product_state_truncation_threshold',
                  'sv_parallel_threshold': 'statevector_parallel_threshold',
                  'es_approx_error': 'extended_stabilizer_approximation_error'}

    def options(self) -> dict[str, Any]:
        """The set options under Aer's names."""
        return {aer: getattr(self, mine) for mine, aer in self._AER_NAMES.items()
                if getattr(self, mine) is not None}

    def value(self, name: str) -> float:
        """An option as PEEK reads it."""
        v = getattr(self, name)
        return float(self.DEFAULTS[name] if v is None else v)


@dataclass
class MachineState:
    """The memory map's own storage and the profiler: the zero page, the
    status registers, SYS INSTALL's routines, PROFILE's per-line timings and
    STATS' run history."""

    zero_page: list[float] = field(default_factory=lambda: [0.0] * 64)
    status: dict[int, float] = field(default_factory=lambda: {a: 0.0 for a in STS_NAMES})
    user_sys: dict[int, str] = field(default_factory=dict)                                 # SYS INSTALL: address -> program
    profiling: bool = False
    profile: dict[int, dict[str, float]] = field(default_factory=dict)                     # line -> timings
    profile_start: float = 0.0
    runs: list[dict[str, int]] = field(default_factory=list)                               # STATS: one entry per run


class MemoryMixin(SimulationHost):
    """Memory-mapped PEEK/POKE, SYS, USR, WAIT, DUMP, MAP, MONITOR, CATALOG."""

    def _init_memory(self) -> None:
        self.machine = MachineState()
        self.aer = AerTuning()
        self.machine.status[0xD013] = 1.0
        self._screen_mode: int = 0

    def _update_status(self, **kw: float) -> None:
        for addr, name in STS_NAMES.items():
            if name in kw:
                self.machine.status[addr] = float(kw[name])

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
        if self._def_seg is not None:
            return float(self._segment_read(int(addr)))
        a = int(addr)
        if 0x0000 <= a <= 0x003F:
            return self.machine.zero_page[a]
        if 0x0100 <= a <= 0x01FF:
            return self._peek_qubit(a)
        if 0xD100 <= a <= 0xD1FF:
            qubit = (a - 0xD100) // 2
            field = (a - 0xD100) % 2
            ntype, nparam = self.noise.per_qubit.get(qubit, (0, 0.0))
            return float(ntype) if field == 0 else nparam
        if a in CFG_NAMES:
            return self._peek_config(a)
        if a in (0xD013, 0xD014):
            return self._state_status(a)
        if a in STS_NAMES:
            return self.machine.status.get(a, 0.0)
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
            return 0.0 if self.noise.model is None else 1.0
        if n == 'noise_param':
            return float(self.noise.depol_p) if self.noise.model is not None else 0.0
        if n == 'max_iterations':
            return float(self._max_iterations)
        if n == 'screen_mode':
            return float(self.graphics.canvas.mode if self.graphics.canvas is not None else self._screen_mode)
        if n in AerTuning.DEFAULTS:
            return self.aer.value(n)
        if n == 'precision':
            return 1.0 if self.sim_precision == 'single' else 0.0
        return 0.0

    # ── POKE ───────────────────────────────────────────────────────────

    def _poke(self, addr: float, value: float) -> None:
        if self._def_seg is not None:
            self._segment_write(int(addr), int(value))
            return
        a, v = int(addr), float(value)
        if 0x0000 <= a <= 0x003F:
            if not isinstance(value, (int, float)):
                self.io.writeln(f"?TYPE: zero page expects numeric value, got {type(value).__name__}")
                return
            if abs(v) > 1e15:
                self.io.writeln(f"?WARNING: suspiciously large zero-page value: {v}")
            self.machine.zero_page[a] = v
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
            self.aer.fusion_enable = bool(int(v))
        elif a == 0xD009:
            self.aer.mps_truncation = v
        elif a == 0xD00A:
            self.aer.sv_parallel_threshold = int(v)
        elif a == 0xD00B:
            self.aer.es_approx_error = v
        elif 0xD100 <= a <= 0xD1FF:
            qubit = (a - 0xD100) // 2
            field = (a - 0xD100) % 2
            ntype, nparam = self.noise.per_qubit.get(qubit, (0, 0.0))
            if field == 0:
                self.noise.per_qubit[qubit] = (int(v), nparam)
            else:
                self.noise.per_qubit[qubit] = (ntype, v)
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
            self.machine.user_sys[addr] = name
            self.io.writeln(f"INSTALLED {name} AT ${addr:04X}")
            return
        addr = self._eval_int(rest)
        if addr in SYS_ROUTINES:
            if SYS_ROUTINES[addr] == 'STRESS':
                self.io.writeln("  (note: STRESS runs a 20-qubit circuit; this can take a while)")
            self.cmd_demo(SYS_ROUTINES[addr])
        elif addr in self.machine.user_sys:
            name = self.machine.user_sys[addr]
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
        elif a in self.machine.user_sys:
            name = self.machine.user_sys[a]
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
        if self.machine.user_sys:
            self.io.writeln("\n  User SYS Routines:")
            for addr, name in sorted(self.machine.user_sys.items()):
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
        nz = sum(1 for v in self.machine.zero_page if v != 0)
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
        self.io.writeln(f"  $F000-$FFFF  User SYS         ({len(self.machine.user_sys)} installed)")
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

    def _try_exec_wait(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """WAIT in a program: on an I/O port (the VGA ports &H3C0-&H3DF, the
        keyboard &H60) QBASIC's WAIT port, mask[, xor]; on an address of the
        memory map the WAIT command."""
        m = re.match(r'WAIT\s+(.+)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        parts = self._split_arg_list(m.group(1))
        port = int(self._eval_with_vars(parts[0], run_vars))
        if 0x3C0 <= port <= 0x3DF or port in (0x60, 0x61):
            mask = int(self._eval_with_vars(parts[1], run_vars)) if len(parts) > 1 else 0xFF
            xor = int(self._eval_with_vars(parts[2], run_vars)) if len(parts) > 2 else 0
            self._port_wait(port, mask, xor)
            return True
        self.cmd_wait(m.group(1))
        return True

    def _try_exec_poke(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = RE_POKE.match(stmt)
        if not m:
            return False
        addr = self._eval_with_vars(m.group(1), run_vars)
        val = self._eval_with_vars(m.group(2), run_vars)
        self._poke(addr, val)
        return True

    def _try_exec_sys(self, stmt: str) -> bool:
        m = re.match(r'SYS\s+(.+)', stmt, re.IGNORECASE)
        if not m:
            return False
        self.cmd_sys(m.group(1))
        return True


# ---------------------------------------------------------------------------
# QUBASIC profiler — profile mode, gate count, depth tracking, statistics accumulator.
# ---------------------------------------------------------------------------

MAX_STATS_RUNS = 10000


class ProfilerMixin(SimulationHost):
    """Profiling and statistics for QBasicTerminal."""

    # ── Profile mode ───────────────────────────────────────────────────

    def cmd_profile(self, rest: str = '') -> None:
        """PROFILE [ON|OFF|SHOW] — toggle profiling or show results."""
        arg = rest.strip().upper()
        if arg == 'ON':
            self.machine.profiling = True
            self.machine.profile.clear()
            self.io.writeln("PROFILE ON")
        elif arg == 'OFF':
            self.machine.profiling = False
            self.io.writeln("PROFILE OFF")
        elif arg == 'SHOW' or not arg:
            self._show_profile()
        else:
            self.io.writeln("?USAGE: PROFILE [ON|OFF|SHOW]")

    def _profile_line_start(self, line_num: int) -> None:
        if self.machine.profiling:
            self.machine.profile_start = time.perf_counter()

    def _profile_line_end(self, line_num: int, gates: int = 0) -> None:
        if self.machine.profiling:
            dt = (time.perf_counter() - self.machine.profile_start) * 1000  # ms
            if line_num not in self.machine.profile:
                self.machine.profile[line_num] = {'time_ms': 0.0, 'calls': 0, 'gates': 0}
            entry = self.machine.profile[line_num]
            entry['time_ms'] += dt
            entry['calls'] += 1
            entry['gates'] += gates

    def _show_profile(self) -> None:
        if not self.machine.profile:
            self.io.writeln("  No profile data (PROFILE ON, then RUN)")
            return
        self.io.writeln("\n  Profile Results:")
        self.io.writeln(f"  {'Line':>6}  {'Time(ms)':>10}  {'%':>5}  {'Calls':>6}  {'Gates':>6}  Source")
        total_time = sum(d['time_ms'] for d in self.machine.profile.values())
        for ln in sorted(self.machine.profile.keys()):
            d = self.machine.profile[ln]
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
            self.machine.runs.clear()
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
                    if len(self.machine.runs) >= MAX_STATS_RUNS:
                        self.io.writeln(f"?STATS: run limit ({MAX_STATS_RUNS}) reached, stopping collection")
                        break
                    self.machine.runs.append(dict(self.last_counts))
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
        self.io.writeln(f"Collected {len(self.machine.runs)} runs ({n} trials)")

    def _show_stats(self) -> None:
        if not self.machine.runs:
            self.io.writeln("  No statistics collected (STATS N to run N trials)")
            return
        n = len(self.machine.runs)
        # Aggregate: count how often each state appears across runs
        state_totals: dict[str, list[int]] = {}
        for run in self.machine.runs:
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
        shot_totals = [sum(run.values()) for run in self.machine.runs]
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
        if not self.machine.runs:
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
        n = len(self.machine.runs)
        state_totals: dict[str, list[int]] = {}
        for run in self.machine.runs:
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
        shot_totals = [sum(run.values()) for run in self.machine.runs]
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


# ---------------------------------------------------------------------------
# QUBASIC built-in demo circuits with self-verification.
# ---------------------------------------------------------------------------

class DemoMixin(SimulationHost):
    """Built-in quantum computing demos."""

    def cmd_demo(self, name: str) -> None:
        """DEMO <name> | DEMO LIST — load, list and run a built-in demo circuit, checking its result."""
        demos = {
            'BELL':        self._demo_bell,
            'GHZ':         self._demo_ghz,
            'TELEPORT':    self._demo_teleport,
            'GROVER':      self._demo_grover,
            'QFT':         self._demo_qft,
            'DEUTSCH':     self._demo_deutsch,
            'BERNSTEIN':   self._demo_bernstein,
            'SUPERDENSE':  self._demo_superdense,
            'RANDOM':      self._demo_random,
            'STRESS':      self._demo_stress,
            'LOCC-TELEPORT': self._demo_locc_teleport,
            'LOCC-COORD':    self._demo_locc_coord,
        }
        name = name.upper().strip()
        if not name or name == 'LIST':
            self.io.writeln("  Available demos:")
            for d in demos:
                prefix = "  (LOCC)" if d.startswith('LOCC') else ""
                self.io.writeln(f"    DEMO {d}{prefix}")
            return
        if name not in demos:
            self.io.writeln(f"?UNKNOWN DEMO: {name}")
            self.io.writeln("  Try: DEMO LIST")
            return
        demos[name]()

    def _verify_demo(self, expected_states: list[str], min_frac: float = 0.85,
                     label: str = '') -> bool:
        """Check that expected states dominate the output.

        expected_states: list of bitstrings that should capture >= min_frac of shots.
        Returns True if verification passes.
        """
        if not self.last_counts:
            self.io.writeln("  VERIFY: no results")
            return False
        total = sum(self.last_counts.values())
        hit = sum(self.last_counts.get(s, 0) for s in expected_states)
        frac = hit / total if total > 0 else 0
        tag = f" ({label})" if label else ""
        if frac >= min_frac:
            self.io.writeln(f"  VERIFY PASS{tag}: {frac:.1%} in expected states "
                           f"(threshold {min_frac:.0%})")
            return True
        else:
            self.io.writeln(f"  VERIFY FAIL{tag}: {frac:.1%} in expected states "
                           f"(threshold {min_frac:.0%})")
            return False

    def _demo_bell(self) -> None:
        self.cmd_new()
        self.num_qubits = 2
        self.program = {
            10: "REM === Bell State ===",
            20: "REM The simplest entanglement. Two qubits, perfectly correlated.",
            30: "H 0",
            40: "CX 0,1",
            50: "MEASURE",
        }
        self.io.writeln("LOADED: Bell State (2 qubits)")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['00', '11'], 0.95, 'Bell: only |00> and |11>')

    def _demo_ghz(self) -> None:
        self.cmd_new()
        n = min(self.num_qubits, 8)
        self.num_qubits = n
        self.program = {
            10: "REM === GHZ State ===",
            20: f"REM {n}-qubit Greenberger-Horne-Zeilinger state",
            30: "REM All qubits entangled: 50% all-zeros, 50% all-ones.",
            40: "REM Einstein called this 'spooky action at a distance.'",
            50: "H 0",
        }
        line = 60
        for i in range(1, n):
            self.program[line] = f"CX 0,{i}"
            line += 10
        self.program[line] = "MEASURE"
        self.io.writeln(f"LOADED: GHZ State ({n} qubits)")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['0' * n, '1' * n], 0.95, f'GHZ: only |{"0"*n}> and |{"1"*n}>')

    def _demo_teleport(self) -> None:
        self.cmd_new()
        self.num_qubits = 3
        self.program = {
            10:  "REM === Quantum Teleportation (circuit only) ===",
            20:  "REM Qubit 0: state to teleport (prepared as |+>)",
            30:  "REM Qubit 1-2: entangled Bell pair (the 'channel')",
            40:  "REM Shows the entangling stage of teleportation.",
            50:  "REM Classical correction requires mid-circuit measurement",
            60:  "REM with feedforward, which needs LOCC mode.",
            70:  "REM >>> For full teleportation: DEMO LOCC-TELEPORT <<<",
            80:  "REM",
            90:  "REM Prepare state to teleport (|+> on qubit 0)",
            100: "H 0",
            110: "REM Create Bell pair on qubits 1,2",
            120: "H 1",
            130: "CX 1,2",
            140: "REM Teleportation circuit (without classical correction)",
            150: "CX 0,1",
            160: "H 0",
            170: "MEASURE",
        }
        self.io.writeln("LOADED: Quantum Teleportation — circuit stage (3 qubits)")
        self.io.writeln("  Without correction: uniform over all 8 outcomes.")
        self.io.writeln("  For full protocol with correction: DEMO LOCC-TELEPORT")
        self.cmd_list()
        self.cmd_run()

    def _demo_grover(self) -> None:
        self.cmd_new()
        self.num_qubits = 3
        self.shots = 1024
        self.program = {
            10:  "REM === Grover's Search Algorithm ===",
            20:  "REM Searching for |101> among 8 states",
            30:  "REM Classically: ~4 queries. Quantumly: 2 iterations, ~95%.",
            40:  "REM Quadratic speedup over classical unstructured search.",
            50:  "REM",
            60:  "REM Superposition",
            70:  "H 0 : H 1 : H 2",
            80:  "REM",
            90:  "REM === Grover iteration 1 ===",
            100: "REM Oracle: flip phase of |101> (X on q1, CCZ, undo X)",
            110: "X 1",
            120: "H 2 : CCX 0,1,2 : H 2",
            130: "X 1",
            140: "REM Diffusion: 2|s><s| - I",
            150: "H 0 : H 1 : H 2",
            160: "X 0 : X 1 : X 2",
            170: "H 2 : CCX 0,1,2 : H 2",
            180: "X 0 : X 1 : X 2",
            190: "H 0 : H 1 : H 2",
            200: "REM",
            210: "REM === Grover iteration 2 ===",
            220: "X 1",
            230: "H 2 : CCX 0,1,2 : H 2",
            240: "X 1",
            250: "H 0 : H 1 : H 2",
            260: "X 0 : X 1 : X 2",
            270: "H 2 : CCX 0,1,2 : H 2",
            280: "X 0 : X 1 : X 2",
            290: "H 0 : H 1 : H 2",
            300: "REM",
            310: "MEASURE",
        }
        self.io.writeln("LOADED: Grover's Search (3 qubits, target=|101>)")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['101'], 0.85, 'Grover: target |101>')

    def _demo_qft(self) -> None:
        self.cmd_new()
        self.num_qubits = 4
        self.program = {
            10:  "REM === Quantum Fourier Transform ===",
            20:  "REM The quantum analog of the discrete Fourier transform.",
            30:  "REM Exponentially faster than classical FFT.",
            40:  "REM QFT is typically used as a subroutine (phase info lost on measure).",
            50:  "REM",
            60:  "REM Prepare input state |5> = |0101>",
            70:  "X 0 : X 2",
            80:  "REM",
            90:  "REM QFT on 4 qubits",
            100: "H 3",
            110: "CP PI/2, 2, 3",
            120: "CP PI/4, 1, 3",
            130: "CP PI/8, 0, 3",
            140: "H 2",
            150: "CP PI/2, 1, 2",
            160: "CP PI/4, 0, 2",
            170: "H 1",
            180: "CP PI/2, 0, 1",
            190: "H 0",
            200: "REM Swap for bit reversal",
            210: "SWAP 0,3",
            220: "SWAP 1,2",
            230: "MEASURE",
        }
        self.io.writeln("LOADED: QFT on |0101> (4 qubits)")
        self.cmd_list()
        self.cmd_run()

    def _demo_deutsch(self) -> None:
        self.cmd_new()
        self.num_qubits = 2
        self.program = {
            10:  "REM === Deutsch-Jozsa Algorithm ===",
            20:  "REM Determines if a function is constant or balanced",
            30:  "REM in ONE query. Classically needs 2.",
            40:  "REM Published 1992. The first to demonstrate quantum advantage.",
            50:  "REM",
            60:  "REM Oracle qubit in |->",
            70:  "X 1",
            80:  "H 0 : H 1",
            90:  "REM Oracle (balanced function: f(x) = x)",
            100: "CX 0,1",
            110: "REM Interfere and measure",
            120: "H 0",
            130: "MEASURE",
        }
        self.io.writeln("LOADED: Deutsch-Jozsa (2 qubits, balanced oracle)")
        self.io.writeln("  Expect: qubit 0 = 1 (balanced)")
        self.cmd_list()
        self.cmd_run()
        # q0=1 means bit pattern x1 (states 01 or 11)
        self._verify_demo(['01', '11'], 0.95, 'Deutsch: q0=1 (balanced)')

    def _demo_bernstein(self) -> None:
        self.cmd_new()
        self.num_qubits = 5
        self.program = {
            10:  "REM === Bernstein-Vazirani Algorithm ===",
            20:  "REM Finds a hidden bit string s in ONE query.",
            30:  "REM Secret: s = 1011 (4 bits, display order q3q2q1q0)",
            40:  "REM Classically you'd need 4 queries. Quantum: 1.",
            50:  "REM",
            60:  "REM Ancilla in |->",
            70:  "X 4",
            80:  "H 0 : H 1 : H 2 : H 3 : H 4",
            90:  "REM Oracle: CX from each bit where s=1",
            100: "CX 0,4",
            110: "CX 1,4",
            120: "CX 3,4",
            130: "REM Interfere",
            140: "H 0 : H 1 : H 2 : H 3",
            150: "MEASURE",
        }
        self.io.writeln("LOADED: Bernstein-Vazirani (5 qubits, secret=1011)")
        self.io.writeln("  Expect: measurement shows ...1011 (q3q2q1q0)")
        self.cmd_list()
        self.cmd_run()
        # Ancilla (q4) is random; data qubits should read 1011
        self._verify_demo(['01011', '11011'], 0.95, 'BV: secret=1011')

    def _demo_superdense(self) -> None:
        self.cmd_new()
        self.num_qubits = 2
        self.program = {
            10:  "REM === Superdense Coding ===",
            20:  "REM Send 2 classical bits using 1 qubit.",
            30:  "REM Encoding: 00->I, 01->X, 10->Z, 11->ZX (=iY)",
            40:  "REM Sending message '11'",
            50:  "REM",
            60:  "REM Create Bell pair",
            70:  "H 0",
            80:  "CX 0,1",
            90:  "REM Encode '11': apply Z then X to qubit 0",
            100: "Z 0",
            110: "X 0",
            120: "REM Decode: reverse Bell circuit",
            130: "CX 0,1",
            140: "H 0",
            150: "MEASURE",
        }
        self.io.writeln("LOADED: Superdense Coding (message='11')")
        self.io.writeln("  Expect: |11> with high probability")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['11'], 0.95, 'Superdense: message=11')

    def _demo_random(self) -> None:
        self.cmd_new()
        n = min(self.num_qubits, 8)
        self.num_qubits = n
        self.shots = 1
        self.program = {
            10: "REM === Quantum Random Number Generator ===",
            20: f"REM {n} qubits = {n}-bit truly random number",
            30: "REM (as random as a simulator's PRNG allows)",
        }
        line = 40
        for i in range(n):
            self.program[line] = f"H {i}"
            line += 10
        self.program[line] = "MEASURE"
        self.io.writeln(f"LOADED: Quantum RNG ({n} bits)")
        self.cmd_list()
        self.cmd_run()
        self.shots = 1024

    def _demo_stress(self) -> None:
        self.cmd_new()
        self.io.writeln("  (note: STRESS builds a highly entangled 20-qubit state; this can take a while)")
        n = 20
        self.num_qubits = n
        self.program = {
            10: f"REM === {n}-Qubit Stress Test ===",
            20: f"REM H on all {n} qubits + nearest-neighbor CX chain",
            30: f"REM Creates a highly entangled {n}-qubit state",
        }
        line = 40
        for i in range(n):
            self.program[line] = f"H {i}"
            line += 10
        for i in range(n - 1):
            self.program[line] = f"CX {i},{i+1}"
            line += 10
        for i in range(n):
            self.program[line] = f"RZ PI/4,{i}"
            line += 10
        for i in range(0, n - 1, 2):
            self.program[line] = f"CX {i},{i+1}"
            line += 10
        self.program[line] = "MEASURE"
        self.io.writeln(f"LOADED: {n}-Qubit Stress Test")
        self.cmd_run()

    def _demo_locc_teleport(self) -> None:
        """Quantum teleportation across A/B boundary (JOINT mode)."""
        self.cmd_new()
        self.cmd_locc('JOINT 3 3')
        self.shots = 1024
        self.program = {
            10:  "REM === LOCC Teleportation ===",
            20:  "REM Alice (A) teleports qubit 0 to Bob (B) qubit 0",
            30:  "REM Pre-shared Bell pair: A[2] <-> B[0]",
            40:  "REM",
            50:  "REM Prepare state to teleport: |+> on A[0]",
            60:  "@A H 0",
            70:  "REM Create shared entanglement",
            80:  "SHARE A 2, B 0",
            90:  "REM",
            100: "REM === Teleportation protocol ===",
            110: "REM Alice entangles her data qubit with her half of the pair",
            120: "@A CX 0, 2",
            130: "@A H 0",
            140: "REM Alice measures and sends classical bits to Bob",
            150: "SEND A 0 -> m0",
            160: "SEND A 2 -> m1",
            170: "REM Bob applies corrections based on Alice's results",
            180: "IF m1 THEN @B X 0",
            190: "IF m0 THEN @B Z 0",
            200: "REM Bob's qubit 0 now holds Alice's original state",
            210: "MEASURE",
        }
        self.io.writeln("LOADED: LOCC Teleportation (JOINT 3+3)")
        self.io.writeln("  Alice sends |+> to Bob. Expect Bob's qubit ~ 50/50.")
        self.cmd_list()
        self.cmd_run()
        # Fidelity: |+> teleported to Bob B[0]
        # Z-basis: should be 50/50 (|+> has equal P(0) and P(1))
        # X-basis: should be ~100% |0> (|+> is eigenstate of X with eigenvalue +1)
        if self.last_counts:
            b0_counts = {'0': 0, '1': 0}
            for state, count in self.last_counts.items():
                parts = state.split('|')
                if len(parts) >= 2:
                    bob_bit = parts[-1][-1]
                    b0_counts[bob_bit] = b0_counts.get(bob_bit, 0) + count
            b_total = sum(b0_counts.values())
            if b_total > 0:
                p0 = b0_counts['0'] / b_total
                z_fidelity = 1.0 - abs(p0 - 0.5) * 2
                self.io.writeln(f"  Z-basis fidelity: {z_fidelity:.3f} "
                               f"(Bob B[0]: P(0)={p0:.3f}, P(1)={1-p0:.3f})")
        # X-basis verification via LOCC statevector
        if self.locc and self.locc.joint:
            import numpy as np
            sv = np.ascontiguousarray(self.locc.sv).ravel()
            n = self.locc.n_total
            # Bob qubit 0 is at global index = offset_B + 0
            bob_q = self.locc.offsets[1]
            sv_t = sv.reshape([2] * n)
            ax = n - 1 - bob_q
            t0 = np.moveaxis(sv_t, ax, 0)[0].flatten()
            t1 = np.moveaxis(sv_t, ax, 0)[1].flatten()
            rho_01 = np.sum(np.conj(t0) * t1)
            expect_x = float(2 * rho_01.real)
            self.io.writeln(f"  X-basis verification: <X> on Bob B[0] = {expect_x:.3f} "
                           f"(ideal |+> = 1.000)")

    def _demo_locc_coord(self) -> None:
        """Classical coordination between independent registers (SPLIT mode)."""
        self.cmd_new()
        self.cmd_locc('SPLIT 4 4')
        self.shots = 1024
        self.program = {
            10:  "REM === LOCC Classical Coordination ===",
            20:  "REM Two independent quantum processors sharing classical bits",
            30:  "REM Alice generates a random 2-bit key, Bob copies it",
            40:  "REM",
            50:  "REM Alice: generate random bits",
            60:  "@A H 0 : @A H 1",
            70:  "SEND A 0 -> k0",
            80:  "SEND A 1 -> k1",
            90:  "REM Bob: set qubits to match Alice's key",
            100: "IF k0 THEN @B X 0",
            110: "IF k1 THEN @B X 1",
            120: "REM Both sides now agree on a 2-bit key",
            130: "REM Alice uses her key to prepare an entangled state",
            140: "@A H 2",
            150: "@A CX 2,3",
            160: "REM Bob does the same — they independently create matching states",
            170: "@B H 2",
            180: "@B CX 2,3",
            190: "MEASURE",
        }
        self.io.writeln("LOADED: LOCC Classical Coordination (SPLIT 4+4)")
        self.io.writeln("  Alice and Bob share a random key, then prepare matching states.")
        self.cmd_list()
        self.cmd_run()
