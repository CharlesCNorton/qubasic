"""QUBASIC terminal — REPL, commands, circuit building, LOCC execution."""

import sys
import os
import re
import time
import warnings as _warnings
from collections import OrderedDict
from typing import Any

# Quiet the third-party dependency-version notice (urllib3/chardet) that
# qiskit's transitive imports emit, so it doesn't precede the banner.
_warnings.filterwarnings('ignore', message=r".*match a supported version.*")

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
try:
    from qiskit_aer import AerError
except ImportError:
    AerError = None
import numpy as np

from qubasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    ExecResult,
    _np_gate_matrix, _get_ram_gb, _estimate_gb,
    MAX_QUBITS, DEFAULT_QUBITS, DEFAULT_SHOTS, MAX_UNDO_STACK,
    MAX_LOOP_ITERATIONS,
    OVERHEAD_FACTOR, RAM_BUDGET_FRACTION,
    RE_LINE_NUM, RE_DEF_SINGLE, RE_DEF_BEGIN, RE_REG_INDEX,
    RE_AT_REG, RE_AT_REG_LINE,
    RE_MEAS, RE_RESET, RE_UNITARY, RE_DIM, RE_INPUT,
    RE_CTRL, RE_INV,
    RE_GOTO_GOSUB_TARGET, RE_MEASURE_BASIS, RE_SYNDROME,
    RE_POKE, RE_SYS, RE_OPEN, RE_CLOSE, RE_PRINT_FILE, RE_INPUT_FILE,
    RE_LINE_INPUT, RE_LET_STR, RE_DIM_MULTI,
    RE_DATA, RE_READ, RE_ON_GOTO, RE_ON_GOSUB,
    RE_SELECT_CASE, RE_CASE, RE_DO, RE_LOOP_STMT, RE_EXIT,
    RE_SUB, RE_END_SUB, RE_FUNCTION, RE_END_FUNCTION,
    RE_CALL, RE_LOCAL, RE_STATIC_DECL, RE_SHARED,
    RE_ON_ERROR, RE_RESUME, RE_ERROR_STMT, RE_ASSERT,
    RE_SWAP, RE_DEF_FN, RE_OPTION_BASE,
    RE_PRINT_USING, RE_COLOR, RE_LOCATE, RE_SCREEN, RE_LPRINT,
    RE_ON_MEASURE, RE_ON_TIMER, RE_CHAIN, RE_MERGE,
    RE_REDIM, RE_ERASE, RE_GET,
)
from qubasic_core.executor import ExecutorMixin
from qubasic_core.expression import ExpressionMixin
from qubasic_core.display import DisplayMixin
from qubasic_core.demos import DemoMixin
from qubasic_core.locc import LOCCMixin
from qubasic_core.control_flow import ControlFlowMixin
from qubasic_core.file_io import FileIOMixin
from qubasic_core.analysis import AnalysisMixin
from qubasic_core.sweep import SweepMixin
from qubasic_core.memory import MemoryMixin
from qubasic_core.strings import StringMixin
from qubasic_core.screen import ScreenMixin
from qubasic_core.classic import ClassicMixin
from qubasic_core.subs import SubroutineMixin
from qubasic_core.debug import DebugMixin
from qubasic_core.program_mgmt import ProgramMgmtMixin
from qubasic_core.profiler import ProfilerMixin
from qubasic_core.noise_mixin import NoiseMixin
from qubasic_core.state_display import StateDisplayMixin
from qubasic_core.qol import QoLMixin, did_you_mean, tip_of_the_day, quantum_spin
from qubasic_core.algorithms import AlgorithmsMixin
from qubasic_core.dynamics import DynamicsMixin
from qubasic_core.qec import QECMixin
from qubasic_core.qec2 import QEC2Mixin
from qubasic_core.logical import LogicalMixin
from qubasic_core.benchmarking import BenchmarkingMixin
from qubasic_core.algos2 import Algorithms2Mixin
from qubasic_core.pauliprop import PauliPropMixin
from qubasic_core.qudits import QuditMixin
from qubasic_core.bosonic import BosonicMixin
from qubasic_core.resources import ResourceMixin
from qubasic_core.live import LiveMixin
from qubasic_core.errors import QBasicError, QBasicBuildError, QBasicRangeError
from qubasic_core.io_protocol import StdIOPort
from qubasic_core.parser import parse_stmt
from qubasic_core.engine_state import Engine
from qubasic_core.help_text import HELP_TEXT, BANNER_ART


# ═══════════════════════════════════════════════════════════════════════
# Named quantum states for SET_STATE
# ═══════════════════════════════════════════════════════════════════════

def _resolve_named_state(name: str, n_qubits: int) -> np.ndarray:
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=complex)
    if name == '|0>':
        sv[0] = 1.0
    elif name == '|1>':
        if dim < 2:
            raise ValueError(f"State |1> requires at least 1 qubit (dim >= 2), got dim={dim}")
        sv[1] = 1.0
    elif name == '|+>':
        if dim < 2:
            raise ValueError(f"State |+> requires at least 1 qubit (dim >= 2), got dim={dim}")
        sv[0] = 1.0 / np.sqrt(2)
        sv[1] = 1.0 / np.sqrt(2)
    elif name == '|->':
        if dim < 2:
            raise ValueError(f"State |-> requires at least 1 qubit (dim >= 2), got dim={dim}")
        sv[0] = 1.0 / np.sqrt(2)
        sv[1] = -1.0 / np.sqrt(2)
    elif name == '|BELL>':
        sv[0] = 1.0 / np.sqrt(2)
        sv[dim - 1] = 1.0 / np.sqrt(2)
    elif name == '|GHZ>':
        sv[0] = 1.0 / np.sqrt(2)
        sv[dim - 1] = 1.0 / np.sqrt(2)
    elif name == '|GHZ3>':
        if dim >= 8:
            sv[0] = 1.0 / np.sqrt(2)
            sv[7] = 1.0 / np.sqrt(2)
        else:
            sv[0] = 1.0  # fallback to |0>
    elif name == '|GHZ4>':
        if dim >= 16:
            sv[0] = 1.0 / np.sqrt(2)
            sv[15] = 1.0 / np.sqrt(2)
        else:
            sv[0] = 1.0  # fallback to |0>
    elif name in ('|W>', '|W3>'):
        if dim >= 8:
            sv[1] = 1.0 / np.sqrt(3)
            sv[2] = 1.0 / np.sqrt(3)
            sv[4] = 1.0 / np.sqrt(3)
        else:
            # fallback: equal superposition of available states
            sv[:] = 1.0 / np.sqrt(dim)
    else:
        sv[0] = 1.0
    return sv


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


def _named_state_fits(name: str, n_qubits: int) -> bool:
    """Whether a named SET_STATE target fits the current qubit count."""
    dim = 2 ** n_qubits
    need = {'|1>': 2, '|+>': 2, '|->': 2, '|GHZ3>': 8, '|GHZ4>': 16,
            '|W>': 8, '|W3>': 8}
    return dim >= need.get(name.upper(), 1)


# ═══════════════════════════════════════════════════════════════════════
# The Terminal
# ═══════════════════════════════════════════════════════════════════════

class QBasicTerminal(Engine, ExecutorMixin, ExpressionMixin, DisplayMixin, DemoMixin,
                     LOCCMixin, ControlFlowMixin, FileIOMixin, AnalysisMixin,
                     SweepMixin, MemoryMixin, StringMixin, ScreenMixin, ClassicMixin,
                     SubroutineMixin, DebugMixin, ProgramMgmtMixin, ProfilerMixin,
                     NoiseMixin, StateDisplayMixin, QoLMixin, AlgorithmsMixin,
                     DynamicsMixin, QECMixin, QEC2Mixin, LogicalMixin,
                     BenchmarkingMixin, Algorithms2Mixin,
                     PauliPropMixin, QuditMixin, BosonicMixin, ResourceMixin, LiveMixin):
    # Architecture: QBasicTerminal composes Engine (state) + 32 mixins (behavior).
    #
    # Mixin map (each provides specific methods; see TerminalProtocol for contract):
    #   ExecutorMixin     — build_circuit, _exec_line, _apply_gate_str, run_immediate
    #   ExpressionMixin   — _safe_eval (AST-based, no eval()), eval_expr, _eval_condition
    #   ControlFlowMixin  — FOR/NEXT, WHILE/WEND, DO/LOOP, IF/THEN, SELECT CASE, etc.
    #   DisplayMixin      — print_histogram, _print_statevector, _print_bloch_single
    #   LOCCMixin         — LOCCCommandsMixin + LOCCDisplayMixin + LOCCExecutionMixin
    #   FileIOMixin       — SAVE/LOAD/INCLUDE/IMPORT/EXPORT/CSV/OPEN/CLOSE
    #   AnalysisMixin     — EXPECT/ENTROPY/DENSITY/BENCH/RAM
    #   SweepMixin        — SWEEP parameter scan
    #   MemoryMixin       — PEEK/POKE/SYS/DUMP/MAP/MONITOR
    #   StringMixin       — string functions (LEFT$, RIGHT$, etc.)
    #   ScreenMixin       — SCREEN/COLOR/CLS/LOCATE
    #   ClassicMixin      — DATA/READ, SELECT CASE, DO/LOOP, SWAP, DEF FN
    #   SubroutineMixin   — SUB/FUNCTION with LOCAL/STATIC/SHARED
    #   DebugMixin        — ON ERROR, breakpoints, watch, time-travel, TRON/TROFF
    #   ProgramMgmtMixin  — AUTO/EDIT/COPY/MOVE/FIND/REPLACE/BANK/CHECKSUM
    #   ProfilerMixin     — PROFILE, STATS
    #   NoiseMixin        — noise model configuration
    #   StateDisplayMixin — STATE/HIST/PROBS/BLOCH/CIRCUIT
    #   DemoMixin         — 12 built-in demo circuits
    #
    # Dual execution paths:
    #   Qiskit path: build_circuit -> QuantumCircuit -> AerSimulator (standard mode)
    #   Numpy path:  _locc_execute_program -> LOCCEngine -> numpy tensordot (LOCC mode)
    #   Shared control flow via _exec_control_flow() with divergent gate application.
    #   This dualism exists because Qiskit doesn't support mid-circuit classical
    #   feedforward in the way LOCC protocols require.
    #
    # Variable scope: Scope(persistent=self.variables) wraps runtime vars.
    #   run_vars[name] and self.variables[name] are mirrored for backward compat.
    #   New code should use Scope methods; legacy mirroring is retained in mixins.

    def _gate_info(self, name: str) -> tuple[int, int] | None:
        """Look up (n_params, n_qubits) for a gate name.

        Checks the global GATE_TABLE first, then instance-local custom gates.
        Returns None if the gate is unknown.
        """
        if name in GATE_TABLE:
            return GATE_TABLE[name]
        if name in self._custom_gates:
            matrix = self._custom_gates[name]
            n_qubits = int(np.log2(matrix.shape[0]))
            return (0, n_qubits)
        return None

    @staticmethod
    def _sanitize_path(path: str) -> str:
        """Path sanitization: reject null bytes, control characters, and traversal.

        Blocks directory-traversal sequences (..) and absolute paths that
        escape the working directory.  For a local REPL the user already has
        shell access, but this prevents accidental overwrites of system files
        and hardens INCLUDE chains against path confusion.
        """
        if not path or not path.strip():
            raise ValueError("Empty path")
        if '\x00' in path:
            raise ValueError("Path contains null bytes")
        if any(ord(c) < 32 for c in path):
            raise ValueError("Path contains control characters")
        path = path.strip()
        # Reject traversal sequences — the primary injection vector
        normalized = os.path.normpath(path)
        if '..' in normalized.split(os.sep):
            raise ValueError("Path traversal (..) is not allowed")
        # Reject absolute paths (Unix / or Windows drive letters)
        if os.path.isabs(path):
            raise ValueError("Absolute paths are not allowed")
        # Catch Windows drive letters on non-Windows hosts (e.g. CI on Linux)
        import re as _re
        if _re.match(r'^[A-Za-z]:[/\\]', path):
            raise ValueError("Absolute paths are not allowed")
        # Reject UNC paths (\\server\share) and extended-length paths (\\?\)
        if path.startswith('\\\\'):
            raise ValueError("UNC/extended paths are not allowed")
        return path

    def __init__(self) -> None:
        """Initialize the QUBASIC terminal with default configuration."""
        Engine.__init__(self)
        # Subsystem initialization (mixin state)
        self._init_memory()
        self._init_screen()
        self._init_classic()
        self._init_subs()
        self._init_debug()
        self._init_program_mgmt()
        self._init_profiler()
        self._init_file_handles()
        self._init_qol()
        self._init_dynamics()
        self._init_qudits()
        self._init_bosonic()
        self._init_logical()
        self._init_live()
        # Device model for transpilation (None = unconstrained, all-to-all).
        self._coupling_map: list[list[int]] | None = None
        self._basis_gates: list[str] | None = None
        # Mid-circuit measurement bits for dynamic (feedforward) circuits:
        # variable name -> Qiskit classical register, populated by MEAS in the
        # standard Aer path and consumed by IF <bit> THEN ... via if_test.
        self._classical_bits: dict = {}
        # Pending mixed-state injection (density matrix) for the next RUN.
        self._pending_set_density = None

    # ── Backend factory ─────────────────────────────────────────────

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

    @property
    def _active_sv(self) -> 'np.ndarray | None':
        """Return the current authoritative statevector.

        In LOCC JOINT mode, returns the joint statevector.
        In LOCC SPLIT mode, returns None (use STATE A / STATE B instead,
        since split registers have independent states).
        In standard mode, returns last_sv.
        """
        if self.locc_mode and self.locc:
            if self.locc.joint:
                return np.ascontiguousarray(self.locc.sv).ravel()
            return None  # split: no single SV; use per-register commands
        return self.last_sv

    def _active_sv_for_reg(self, reg: str) -> 'tuple[np.ndarray, int] | tuple[None, int]':
        """Return (statevector, n_qubits) for a specific LOCC register."""
        if not self.locc:
            return None, 0
        if self.locc.joint:
            return np.ascontiguousarray(self.locc.sv).ravel(), self.locc.n_total
        sv = self.locc.svs.get(reg)
        if sv is None:
            return None, 0
        return np.ascontiguousarray(sv).ravel(), self.locc.get_size(reg)

    @property
    def _active_nqubits(self) -> int:
        """Return the qubit count matching _active_sv."""
        if self.locc_mode and self.locc:
            if self.locc.joint:
                return self.locc.n_total
            return 0  # split: no single qubit count
        return self.num_qubits

    # ── REPL ──────────────────────────────────────────────────────────

    def _setup_readline(self) -> None:
        """Set up command history and tab completion."""
        try:
            import readline
            commands = list(GATE_TABLE.keys()) + [
                'RUN', 'LIST', 'NEW', 'SAVE', 'LOAD', 'QUBITS', 'SHOTS',
                'METHOD', 'DEF', 'REG', 'LET', 'STEP', 'STATE', 'HIST',
                'BLOCH', 'PROBS', 'DEMO', 'DELETE', 'RENUM', 'DEFS', 'REGS',
                'VARS', 'HELP', 'CIRCUIT', 'LOCC', 'SEND', 'SHARE',
                'SWEEP', 'INCLUDE', 'EXPORT', 'DECOMPOSE', 'NOISE', 'EXPECT',
                'DENSITY', 'ENTROPY', 'CSV', 'FOR', 'NEXT', 'IF', 'THEN',
                'ELSE', 'WHILE', 'WEND', 'GOTO', 'GOSUB', 'RETURN', 'END',
                'MEASURE', 'BARRIER', 'REM', 'PRINT', 'INPUT', 'DIM', 'UNITARY',
                'LOCCINFO', 'RAM', 'BYE', 'QUIT', 'EXIT',
            ]
            def completer(text, state):
                t = text.upper()
                matches = [c for c in commands if c.startswith(t)]
                matches += [s for s in self.subroutines if s.startswith(t)]
                matches += [v for v in self.variables if v.upper().startswith(t)]
                matches += [r for r in self.registers if r.upper().startswith(t)]
                # File path completion for SAVE/LOAD/INCLUDE
                try:
                    line_buf = readline.get_line_buffer()
                    first = line_buf.split()[0].upper() if line_buf.split() else ''
                    if first in ('SAVE', 'LOAD', 'INCLUDE', 'IMPORT', 'CHAIN', 'MERGE'):
                        import glob as _glob
                        pattern = text + '*.qb' if not text.endswith('.qb') else text + '*'
                        matches = _glob.glob(pattern)
                except Exception:
                    pass
                return matches[state] + ' ' if state < len(matches) else None
            readline.set_completer(completer)
            readline.parse_and_bind('tab: complete')
            readline.set_completer_delims(' \t\n')
            # Bind F1-F3 to load demos (terminal permitting)
            try:
                readline.parse_and_bind('"\\eOP": "DEMO BELL\\n"')
                readline.parse_and_bind('"\\eOQ": "DEMO GHZ\\n"')
                readline.parse_and_bind('"\\eOR": "DEMO GROVER\\n"')
            except Exception:
                pass
        except ImportError:
            # On Windows, readline is not bundled; try pyreadline3 as fallback
            if sys.platform == 'win32':
                try:
                    import pyreadline3  # noqa: F401 — registers as readline
                    import readline
                    readline.parse_and_bind('tab: complete')
                except ImportError:
                    pass  # no readline available — tab completion disabled

    def repl(self) -> None:
        # Enable VT100 escape sequences on Windows console
        if sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass
        self.print_banner()
        self.io.writeln(f"  Tip: {tip_of_the_day()}")
        self._setup_readline()
        while True:
            try:
                prompt = self._status_prompt() if self._prompt == '] ' else self._prompt
                line = self.io.read_line(prompt).strip()
                if line:
                    self.process(line)
            except KeyboardInterrupt:
                self.io.writeln('')
                continue
            except EOFError:
                self.io.writeln("\nBYE")
                break

    def process(self, line: str, *, track_undo: bool = True) -> None:
        """Process a line of input (numbered line or immediate command)."""
        # Accumulate TYPE fields when a pending TYPE definition is active
        if self._pending_type is not None:
            self._accumulate_type_field(line)
            return
        # Line number -> store in program
        m = RE_LINE_NUM.match(line)
        if m:
            num = int(m.group(1))
            content = m.group(2).strip()
            # Save undo state (skip during script/file loading)
            if track_undo:
                self._undo_stack.append(dict(self.program))
                if len(self._undo_stack) > MAX_UNDO_STACK:
                    self._undo_stack.pop(0)
            if content:
                self.program[num] = content
                self._parsed[num] = parse_stmt(content)
            else:
                if num in self.program:
                    del self.program[num]
                    self._parsed.pop(num, None)
                    self.io.writeln(f"DELETED {num}")
            return
        # Immediate command
        self.dispatch(line)

    _CMD_NO_ARG = {
        'RUN': 'cmd_run', 'NEW': 'cmd_new', 'STEP': 'cmd_step',
        'HIST': 'cmd_hist', 'PROBS': 'cmd_probs', 'DEFS': 'cmd_defs',
        'REGS': 'cmd_regs', 'VARS': 'cmd_vars',
        'CIRCUIT': 'cmd_circuit', 'DECOMPOSE': 'cmd_decompose',
        'LOCCINFO': 'cmd_loccinfo',
        'UNDO': 'cmd_undo', 'RAM': 'cmd_ram',
        'BYE': '_quit', 'QUIT': '_quit', 'EXIT': '_quit',
        # Memory
        'MAP': 'cmd_map', 'CATALOG': 'cmd_catalog', 'MONITOR': 'cmd_monitor',
        # Screen
        'CLS': 'cmd_cls', 'PLAY': 'cmd_play',
        # Debug
        'TRON': 'cmd_tron', 'TROFF': 'cmd_troff', 'CONT': 'cmd_cont',
        'HISTORY': 'cmd_history',
        # Program management
        'CHECKSUM': 'cmd_checksum',
        'VERSION': 'cmd_version',
        'PROBE': 'cmd_probe',
        # Classic
        'RESTORE': 'cmd_restore',
        # Characterization
        'PTOMOGRAPHY': 'cmd_ptomography', 'GST': 'cmd_gst',
        'QSTATE': 'cmd_qstate',
    }
    _CMD_WITH_ARG = {
        'LIST': 'cmd_list', 'QUBITS': 'cmd_qubits', 'SHOTS': 'cmd_shots',
        'METHOD': 'cmd_method', 'DEF': 'cmd_def', 'REG': 'cmd_reg',
        'LET': 'cmd_let', 'STATE': 'cmd_state', 'BLOCH': 'cmd_bloch',
        'STATUS': 'cmd_status',
        'DEMO': 'cmd_demo', 'DELETE': 'cmd_delete', 'RENUM': 'cmd_renum',
        'SAVE': 'cmd_save', 'LOAD': 'cmd_load', 'SWEEP': 'cmd_sweep',
        'INCLUDE': 'cmd_include', 'EXPORT': 'cmd_export',
        'NOISE': 'cmd_noise', 'EXPECT': 'cmd_expect',
        'ENTROPY': 'cmd_entropy', 'DENSITY': 'cmd_density',
        'CSV': 'cmd_csv', 'LOCC': 'cmd_locc',
        'SEND': 'cmd_send', 'SHARE': 'cmd_share', 'DIR': 'cmd_dir',
        'CLEAR': 'cmd_clear',
        # Memory
        'PEEK': 'cmd_peek', 'POKE': 'cmd_poke', 'SYS': 'cmd_sys', 'DUMP': 'cmd_dump', 'WAIT': 'cmd_wait',
        # Screen
        'SCREEN': 'cmd_screen', 'COLOR': 'cmd_color', 'LOCATE': 'cmd_locate',
        'PROMPT': 'cmd_prompt',
        # Debug
        'BREAK': 'cmd_breakpoint', 'WATCH': 'cmd_watch', 'PROFILE': 'cmd_profile',
        'REWIND': 'cmd_rewind', 'FORWARD': 'cmd_forward',
        'STATS': 'cmd_stats',
        # Program management
        'AUTO': 'cmd_auto', 'EDIT': 'cmd_edit', 'COPY': 'cmd_copy',
        'MOVE': 'cmd_move', 'FIND': 'cmd_find', 'REPLACE': 'cmd_replace',
        'BANK': 'cmd_bank', 'CHAIN': 'cmd_chain', 'MERGE': 'cmd_merge',
        # File handles
        'OPEN': 'cmd_open', 'CLOSE': 'cmd_close',
        # Module, types, primitives
        'IMPORT': 'cmd_import', 'TYPE': 'cmd_type',
        'SAMPLE': 'cmd_sample', 'ESTIMATE': 'cmd_estimate', 'BENCH': 'cmd_bench',
        'MINIMIZE': 'cmd_minimize', 'GRADIENT': 'cmd_gradient',
        'PRECISION': 'cmd_precision',
        'FIDELITY': 'cmd_fidelity', 'TOMOGRAPHY': 'cmd_tomography', 'RB': 'cmd_rb',
        'COUPLING': 'cmd_coupling', 'BASIS': 'cmd_basis',
        'LOADQASM': 'cmd_loadqasm', 'SAVEPNG': 'cmd_savepng',
        'HAMILTONIAN': 'cmd_hamiltonian', 'LINDBLAD': 'cmd_lindblad',
        'CHANNEL': 'cmd_channel',
        'QEC': 'cmd_qec', 'LOGICAL_ERROR_RATE': 'cmd_logical_error_rate',
        'THRESHOLD': 'cmd_threshold', 'DISTILL': 'cmd_distill', 'LATTICE': 'cmd_lattice',
        'LQUBITS': 'cmd_lqubits',
        'XEB': 'cmd_xeb', 'QVOLUME': 'cmd_qvolume', 'RBINT': 'cmd_rbint',
        'MIRROR': 'cmd_mirror', 'CONCURRENCE': 'cmd_concurrence',
        'NEGATIVITY': 'cmd_negativity',
        'IQPE': 'cmd_iqpe', 'AMPEST': 'cmd_ampest', 'QWALK': 'cmd_qwalk',
        'QKERNEL': 'cmd_qkernel', 'SHOR': 'cmd_shor', 'HHL': 'cmd_hhl',
        'PAULIPROP': 'cmd_pauliprop',
        'QUDIT': 'cmd_qudit', 'QX': 'cmd_qx', 'QZ': 'cmd_qz', 'QF': 'cmd_qf',
        'QSUM': 'cmd_qsum', 'QMEASURE': 'cmd_qmeasure',
        'BOSONIC': 'cmd_bosonic', 'DISPLACE': 'cmd_displace', 'SQUEEZE': 'cmd_squeeze',
        'CAT': 'cmd_cat', 'BS': 'cmd_bs', 'BSTATE': 'cmd_bstate',
        'RESOURCES': 'cmd_resources', 'DEVICE': 'cmd_device', 'OPTIMIZE': 'cmd_optimize',
        'SET_STATE': 'cmd_set_state', 'SET_DENSITY': 'cmd_set_density',
        # Circuit macros
        'CIRCUIT_DEF': 'cmd_circuit_def', 'APPLY_CIRCUIT': 'cmd_apply_circuit',
        'HELP': 'cmd_help', 'CONSISTENCY': 'cmd_consistency',
        'SEED': 'cmd_seed',
        # QoL features
        'COMPARE': 'cmd_compare', 'HEATMAP': 'cmd_heatmap',
        'ANIMATE': 'cmd_animate', 'QUIZ': 'cmd_quiz',
        'DIFF': 'cmd_diff', 'PLOT': 'cmd_plot', 'THEME': 'cmd_theme',
        'CLIP': 'cmd_clip', 'EXPLAIN': 'cmd_explain', 'DRAW': 'cmd_draw',
    }

    def dispatch(self, line: str) -> None:
        """Parse and execute an immediate command or gate.

        Splits the line on whitespace to extract the command keyword,
        then looks up in _CMD_WITH_ARG / _CMD_NO_ARG tables. Unmatched
        lines fall through to run_immediate (gate / subroutine).
        """
        parts = line.split(None, 1)
        if not parts:
            return
        cmd = parts[0].upper()
        rest = parts[1].strip() if len(parts) > 1 else ''

        method_name = self._CMD_WITH_ARG.get(cmd) or self._CMD_NO_ARG.get(cmd)
        if method_name:
            try:
                method = getattr(self, method_name)
                if cmd in self._CMD_WITH_ARG:
                    method(rest)
                else:
                    method()
            except EOFError:
                raise
            except QBasicError as e:
                self.io.writeln(f"?{e.message}")
            except Exception as e:
                self.io.writeln(f"?ERROR: {e}")
        else:
            # Implicit LET: a bare assignment (x = 5, s$ = "hi") routed to the
            # LET handler so it works without the keyword, as in every BASIC.
            from qubasic_core.patterns import RE_IMPLICIT_ASSIGN
            if RE_IMPLICIT_ASSIGN.match(line.strip()):
                try:
                    self.cmd_let(line.strip())
                except QBasicError as e:
                    self.io.writeln(f"?{e.message}")
                except Exception as e:
                    self.io.writeln(f"?ERROR: {e}")
                return
            # Try as immediate gate / subroutine
            try:
                self.run_immediate(line)
            except Exception as e:
                self.io.writeln(f"?SYNTAX ERROR: {e}")
                self._suggest_command(cmd)

    def _quit(self) -> None:
        """Exit the REPL by raising EOFError."""
        raise EOFError

    # ── Commands ──────────────────────────────────────────────────────

    def cmd_qubits(self, rest: str) -> None:
        """Set or display the number of qubits. The ceiling is per METHOD:
        statevector 32, stabilizer 4096, MPS/automatic 1024, and so on."""
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
        """Set or display the number of measurement shots."""
        if not rest:
            self.io.writeln(f"SHOTS = {self.shots}")
            return
        self.shots = max(1, int(rest))
        self.io.writeln(f"SHOTS = {self.shots}")

    def cmd_method(self, rest: str) -> None:
        """Set simulation method (statevector, stabilizer, MPS, ...) or device (CPU/GPU)."""
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
                self.io.writeln(f"  noise: active")
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

    def cmd_list(self, rest: str = '') -> None:
        """LIST — display program lines. LIST SUBS|VARS|ARRAYS for filtered views."""
        arg = rest.strip().upper()
        if arg == 'SUBS':
            return self.cmd_list_subs()
        if arg == 'VARS':
            return self.cmd_list_vars()
        if arg == 'ARRAYS':
            return self.cmd_list_arrays()
        if not self.program:
            self.io.writeln("EMPTY PROGRAM")
            return
        if self._theme_name != 'none' and sys.stdout.isatty():
            return self.cmd_list_colored()
        for num in sorted(self.program.keys()):
            self.io.writeln(f"  {num:5d}  {self.program[num]}")

    def cmd_new(self, *, silent: bool = False) -> None:
        """NEW — clear program, subroutines, registers, and variables."""
        self.clear()
        if not silent:
            self.io.writeln("READY")

    def cmd_delete(self, rest: str) -> None:
        """DELETE <line> or DELETE <start>-<end> — remove program lines."""
        if not rest:
            self.io.writeln("?USAGE: DELETE <line> or DELETE <start>-<end>")
            return
        if '-' in rest:
            lo_s, hi_s = rest.split('-')
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
            for k in list(self.program.keys()):
                if lo <= k <= hi:
                    del self.program[k]
                    self._parsed.pop(k, None)
            self.io.writeln(f"DELETED {lo}-{hi}")
        else:
            n = int(rest)
            if n in self.program:
                del self.program[n]
                self._parsed.pop(n, None)
                self.io.writeln(f"DELETED {n}")
            else:
                self.io.writeln(f"?LINE {n} NOT FOUND")

    def cmd_renum(self, rest: str = '') -> None:
        """RENUM [start] [step] — renumber lines, update GOTO/GOSUB targets."""
        if not self.program:
            return
        parts = rest.split() if rest else []
        start = int(parts[0]) if len(parts) > 0 else 10
        step = int(parts[1]) if len(parts) > 1 else 10
        old_lines = sorted(self.program.keys())
        # Build mapping: old line number -> new line number
        line_map = {}
        for i, old in enumerate(old_lines):
            line_map[old] = start + i * step
        # Renumber and update GOTO/GOSUB references
        new_prog = {}
        for i, old in enumerate(old_lines):
            stmt = self.program[old]
            # Update GOTO/GOSUB targets
            def replace_target(m):
                target = int(m.group(2))
                new_target = line_map.get(target, target)
                return f"{m.group(1)} {new_target}"
            stmt = RE_GOTO_GOSUB_TARGET.sub(replace_target, stmt)

            # Update ON ... GOTO/GOSUB comma-separated target lists.
            # RE_GOTO_GOSUB_TARGET remaps the first number after GOTO/GOSUB,
            # but ON x GOTO 100, 200, 300 leaves the ", 200, 300" tail
            # untouched.  Walk any trailing comma-separated numbers and
            # remap them too.
            def _remap_on_target_list(m):
                keyword = m.group(1)          # GOTO or GOSUB
                nums_str = m.group(2)         # "10, 20, 30"
                parts = nums_str.split(',')
                remapped = []
                for p in parts:
                    stripped = p.strip()
                    if stripped.isdigit():
                        n = int(stripped)
                        remapped.append(str(line_map.get(n, n)))
                    else:
                        remapped.append(p)
                return f"{keyword} {', '.join(remapped)}"
            stmt = re.sub(
                r'(GOTO|GOSUB)\s+(\d+(?:\s*,\s*\d+)+)',
                _remap_on_target_list, stmt, flags=re.IGNORECASE)

            # Update ON ERROR GOTO <line-number> targets.
            def _remap_on_error(m):
                target = int(m.group(1))
                return f"ON ERROR GOTO {line_map.get(target, target)}"
            stmt = re.sub(
                r'\bON\s+ERROR\s+GOTO\s+(\d+)', _remap_on_error, stmt,
                flags=re.IGNORECASE)

            # Update RESUME <line-number> targets.
            def _remap_resume(m):
                target = int(m.group(1))
                return f"RESUME {line_map.get(target, target)}"
            stmt = re.sub(
                r'\bRESUME\s+(\d+)', _remap_resume, stmt,
                flags=re.IGNORECASE)

            new_prog[line_map[old]] = stmt
        # Note: ELSEIF chains are parsed into nested IF/ELSE at parse time,
        # not stored as line references, so they survive RENUM without needing
        # target remapping.
        self.program = new_prog
        self._parsed = {num: parse_stmt(s) for num, s in new_prog.items()}
        # Move breakpoints to their renumbered lines.
        if getattr(self, '_breakpoints', None):
            self._breakpoints = {line_map.get(b, b) for b in self._breakpoints}
        self.io.writeln(f"RENUMBERED {len(new_prog)} LINES (start={start}, step={step})")
        # Computed jumps (ON <var> GOTO via runtime values) can't be remapped statically.
        if any(re.search(r'\bON\b.*\bGO(TO|SUB)\b', s, re.IGNORECASE)
               for s in new_prog.values()):
            self.io.writeln("  (note: verify any computed ON..GOTO/GOSUB targets)")

    # cmd_save, cmd_load provided by FileIOMixin.

    def cmd_def(self, rest: str) -> None:
        """Define a named gate sequence (subroutine), optionally parameterized."""
        # Single-line: DEF BELL = H 0 : CX 0,1
        # Parameterized: DEF ROTATE(angle, q) = RX angle, q : RZ angle, q
        # Multi-line: DEF BEGIN NAME[(params)] ... DEF END
        upper = rest.upper().strip()
        if upper.startswith('BEGIN'):
            return self._def_multiline(rest[5:].strip())

        # DEF FN name(params) = expr — a user expression function, same as the
        # in-program form, so it works at the prompt too.
        if upper.startswith('FN'):
            from qubasic_core.patterns import RE_DEF_FN
            fm = RE_DEF_FN.match(f"DEF {rest}")
            if fm:
                fparams = [p.strip() for p in fm.group(2).split(',') if p.strip()]
                self._user_fns['FN' + fm.group(1).upper()] = {
                    'params': fparams, 'body': fm.group(3).strip()}
                self.io.writeln(f"DEF FN {fm.group(1).upper()}({', '.join(fparams)})")
                return

        m = RE_DEF_SINGLE.match(rest)
        if not m:
            self.io.writeln("?USAGE: DEF NAME[(params)] = GATE : GATE : ...")
            self.io.writeln("        DEF BEGIN NAME[(params)]  (multi-line, end with DEF END)")
            return
        name = m.group(1).upper()
        params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
        body = [s.strip() for s in m.group(3).split(':') if s.strip()]
        if name in GATE_TABLE:
            from qubasic_core.errors import QBasicSyntaxError
            raise QBasicSyntaxError(f"CANNOT REDEFINE BUILT-IN GATE {name}")
        self.subroutines[name] = {'body': body, 'params': params}
        if params:
            self.io.writeln(f"DEF {name}({', '.join(params)}) ({len(body)} gates)")
        else:
            self.io.writeln(f"DEF {name} ({len(body)} gates)")

    def _def_multiline(self, header: str) -> None:
        """Read multi-line DEF block from REPL."""
        m = re.match(r'(\w+)(?:\(([^)]*)\))?', header)
        if not m:
            self.io.writeln("?USAGE: DEF BEGIN NAME[(params)]")
            return
        name = m.group(1).upper()
        params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
        if name in GATE_TABLE:
            from qubasic_core.errors import QBasicSyntaxError
            raise QBasicSyntaxError(f"CANNOT REDEFINE BUILT-IN GATE {name}")

        body = []
        self.io.writeln(f"  DEF {name} (type gates, DEF END to finish)")
        while True:
            try:
                line = self.io.read_line('  . ').strip()
            except (KeyboardInterrupt, EOFError):
                self.io.writeln("\n  CANCELLED")
                return
            if line.upper() == 'DEF END' or line.upper() == 'END':
                break
            if line:
                body.append(line)
        self.subroutines[name] = {'body': body, 'params': params}
        self.io.writeln(f"DEF {name} ({len(body)} gates)")

    def cmd_type(self, rest: str) -> None:
        """TYPE name — define a named record type.

        Usage: TYPE Point
                 x AS FLOAT
                 y AS FLOAT
               END TYPE
        Then: DIM p AS Point
              LET p.x = 3.14

        Works both interactively (REPL) and non-interactively (LOAD/INCLUDE).
        In non-interactive mode, subsequent lines are routed through process()
        which delegates to _accumulate_type_field until END TYPE is seen.
        """
        from qubasic_core.engine import RE_TYPE_BEGIN
        m = RE_TYPE_BEGIN.match(f"TYPE {rest}")
        if not m:
            self.io.writeln("?USAGE: TYPE <name>")
            return
        type_name = m.group(1).upper()
        # Always use the accumulation path — works for both REPL and scripts.
        # In REPL, repl() calls process() per line, which feeds _accumulate_type_field.
        # In scripts, cmd_load/cmd_include call process() per line likewise.
        self._pending_type = {'name': type_name, 'fields': []}
        self.io.writeln(f"  TYPE {type_name} (enter fields, END TYPE to finish)")

    def _accumulate_type_field(self, line: str) -> None:
        """Accumulate a field for a pending TYPE definition, or finalize on END TYPE."""
        from qubasic_core.engine import RE_TYPE_FIELD, RE_END_TYPE
        assert self._pending_type is not None  # only called while a TYPE is open
        stripped = line.strip()
        if RE_END_TYPE.match(stripped):
            # Finalize the type definition
            type_name = self._pending_type['name']
            fields = self._pending_type['fields']
            self._pending_type = None
            self._user_types[type_name] = fields
            self.io.writeln(f"TYPE {type_name} ({len(fields)} fields)")
            return
        fm = RE_TYPE_FIELD.match(stripped)
        if fm:
            self._pending_type['fields'].append((fm.group(1).lower(), fm.group(2).upper()))
        elif stripped:
            self.io.writeln(f"  ?EXPECTED: <name> AS <type>")

    def cmd_circuit_def(self, rest: str) -> None:
        """CIRCUIT_DEF name start-end — define a circuit macro from line range."""
        m = re.match(r'(\w+)\s+(\d+)\s*-\s*(\d+)', rest)
        if not m:
            self.io.writeln("?USAGE: CIRCUIT_DEF <name> <start>-<end>")
            return
        name = m.group(1).upper()
        start, end = int(m.group(2)), int(m.group(3))
        body = []
        for ln in sorted(self.program.keys()):
            if start <= ln <= end:
                body.append(self.program[ln])
        if not body:
            self.io.writeln(f"?NO LINES IN RANGE {start}-{end}")
            return
        self.subroutines[name] = {'body': body, 'params': []}
        self.io.writeln(f"CIRCUIT {name} = lines {start}-{end} ({len(body)} gates)")

    def cmd_apply_circuit(self, rest: str) -> None:
        """APPLY_CIRCUIT name [@offset] — apply a circuit macro."""
        m = re.match(r'(\w+)(?:\s+@(\d+))?', rest)
        if not m:
            self.io.writeln("?USAGE: APPLY_CIRCUIT <name> [@offset]")
            return
        name = m.group(1).upper()
        offset = int(m.group(2)) if m.group(2) else 0
        if name not in self.subroutines:
            self.io.writeln(f"?UNDEFINED CIRCUIT: {name}")
            return
        call_str = f"{name} @{offset}" if offset else name
        self.run_immediate(call_str)

    def cmd_reg(self, rest: str) -> None:
        """REG <name> <size> — allocate a named qubit register."""
        # REG data 3
        parts = rest.split()
        if len(parts) != 2:
            self.io.writeln("?USAGE: REG <name> <size>")
            return
        name = parts[0].lower()
        size = int(parts[1])
        # Allocate starting after existing registers
        start = sum(s for _, s in self.registers.values())
        if start + size > self.num_qubits:
            from qubasic_core.errors import QBasicRangeError
            raise QBasicRangeError(f"NOT ENOUGH QUBITS (need {start+size}, have {self.num_qubits})")

        self.registers[name] = (start, size)
        self.io.writeln(f"REG {name} = qubits {start}-{start+size-1}")

    def cmd_let(self, rest: str) -> None:
        """LET <var> = <expr> — assign a computed value to a variable.
        Supports string variables (LET s$ = "hi"), record fields (LET p.x = 3.14)
        and array elements (LET a(0) = PI, LET m(i, j) = x), matching the
        in-program LET."""
        from qubasic_core.patterns import RE_LET_ARRAY, RE_LET_STR
        am = RE_LET_ARRAY.match(f"LET {rest}")
        if am:
            from qubasic_core.statements import LetArrayStmt
            name, idx_expr, val_expr = am.group(1), am.group(2), am.group(3)
            parsed = LetArrayStmt(raw=f"LET {rest}", name=name,
                                  index_expr=idx_expr, value_expr=val_expr)
            self._cf_let_array(f"LET {rest}", self.variables, parsed)
            shown = (self._eval_string_expr(val_expr) if name.endswith('$')
                     else self.eval_expr(val_expr))
            self.io.writeln(f"{name}({idx_expr.strip()}) = {shown!r}"
                            if isinstance(shown, str)
                            else f"{name}({idx_expr.strip()}) = {shown}")
            return
        sm = RE_LET_STR.match(f"LET {rest}")
        if sm:
            self.cmd_let_str(sm.group(1), sm.group(2))
            return
        m = re.match(r'(\w+(?:\.\w+)?)\s*=\s*(.*)', rest)
        if not m:
            self.io.writeln("?USAGE: LET <var> = <expr>")
            return
        name = m.group(1)
        self._assert_assignable(name)
        raw = self._safe_eval(m.group(2))
        if isinstance(raw, str):
            self.io.writeln(
                f"?TYPE MISMATCH: '{name}' is numeric — use '{name}$' for strings")
            return
        val = float(raw)
        self.variables[name] = val
        if '.' in name:  # mirror into the record dict for LIST VARS
            base, field = name.split('.', 1)
            rec = self.variables.get(base)
            if isinstance(rec, dict):
                rec[field] = val
        self.io.writeln(f"{name} = {val}")

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

    # cmd_defs, cmd_regs, cmd_vars provided by ProgramMgmtMixin.

    # ── Run ───────────────────────────────────────────────────────────

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
        """Execute the stored program."""
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
        self._scan_subs(sorted_lines)
        self._validate_program(sorted_lines)

        # LQUBITS ... CODE STEANE runs the program encoded.
        if self._logical_mode and self._logical_mode.get('encoded'):
            return self._run_steane(t0)

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
            # Trapped build-time errors resume inside build_circuit itself; an
            # exception arriving here with a handler armed means the handler
            # itself failed, which is fatal (classic BASIC semantics).
            if self._error_target is not None and not self._in_error_handler:
                # Build errors are wrapped as "LINE N: <original>"; recover the
                # failing line and any "ERROR <code>" the program raised so that
                # ERR/ERL carry the real values instead of defaulting to 1/0.
                msg = str(e)
                _lm = re.match(r'LINE (\d+):\s*(.*)', msg, re.DOTALL)
                self._err_line = int(_lm.group(1)) if _lm else 0
                _inner = _lm.group(2) if _lm else msg
                _cm = re.search(r'\bERROR (\d+)', _inner)
                self._err_code = int(_cm.group(1)) if _cm else 1
                self._in_error_handler = True
                self.variables['ERR'] = self._err_code
                self.variables['ERL'] = self._err_line
                # Run handler lines through the program executor (not dispatch),
                # so PRINT/LET behave as in a program and no immediate state
                # dump is emitted. Stops at END or RESUME.
                from qubasic_core.exec_context import ExecContext
                from qubasic_core.scope import Scope
                handler_lines = sorted(ln for ln in self.program if ln >= self._error_target)
                qc_h = QuantumCircuit(self.num_qubits)
                h_ctx = ExecContext(sorted_lines=handler_lines, ip=0,
                                    run_vars=Scope(self.variables),
                                    max_iterations=self._max_iterations, qc=qc_h)
                for _hi, ln in enumerate(handler_lines):
                    if _hi >= 200:
                        self.io.writeln("?ERROR HANDLER LIMIT (200 lines)")
                        break
                    up = self.program[ln].strip().upper()
                    if up == 'END' or up.startswith('RESUME'):
                        break
                    try:
                        self._exec_line(self.program[ln].strip(),
                                        parsed=self._get_parsed(ln), ctx=h_ctx)
                    except Exception:
                        pass
                self._in_error_handler = False
            else:
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
        """Step through the program a line at a time, showing the state after
        each. It runs as one live shot: MEAS collapses the state and IF, PRINT
        and LET see the outcome."""
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
        sorted_lines = sorted(self.program.keys())
        self._gosub_stack = []
        self._collect_data()
        self._scan_subs(sorted_lines)
        state = LiveState(self.num_qubits, self.sim_precision, self._live_noise())
        self._prepare_run(state.qc)
        ctx = ExecContext(
            sorted_lines=sorted_lines, ip=0,
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
        sorted_lines = ctx.sorted_lines

        self.io.writeln(f"STEP MODE \u2014 {len(sorted_lines)} lines, {self.num_qubits} qubits")
        self.io.writeln("Press ENTER to advance, A for auto-play, Q to quit\n")

        while ctx.ip < len(sorted_lines):
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            line_num = sorted_lines[ctx.ip]
            stmt = self.program[line_num]
            parsed = self._get_parsed(line_num)

            # Display current line
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

    # run_immediate provided by ExecutorMixin.

    def _validate_program(self, sorted_lines: list[int]) -> None:
        """Pre-execution validation. Catches structural errors before running."""
        from qubasic_core.statements import (
            GotoStmt, GosubStmt, ForStmt, NextStmt, WhileStmt, WendStmt,
            DoStmt, LoopStmt, SubStmt, EndSubStmt, FunctionStmt, EndFunctionStmt,
            IfThenStmt,
        )
        from qubasic_core.statements import CompoundStmt
        line_set = set(sorted_lines)
        for_depth = 0
        while_depth = 0
        do_depth = 0
        for_var_stack: list[str] = []
        flat: list[tuple[int, Any]] = []
        for ln in sorted_lines:
            top = self._get_parsed(ln)
            if isinstance(top, CompoundStmt):
                flat.extend((ln, parse_stmt(p)) for p in top.parts)
            else:
                flat.append((ln, top))
        for ln, parsed in flat:
            if isinstance(parsed, GotoStmt) and parsed.target not in line_set:
                raise RuntimeError(f"LINE {ln}: GOTO {parsed.target} — target line not found")
            if isinstance(parsed, GosubStmt) and parsed.target not in line_set:
                raise RuntimeError(f"LINE {ln}: GOSUB {parsed.target} — target line not found")
            # Validate GOTO/GOSUB targets embedded in IF THEN/ELSE clauses
            if isinstance(parsed, IfThenStmt):
                for clause in (parsed.then_clause, parsed.else_clause):
                    if clause:
                        for m in re.finditer(r'\b(?:GOTO|GOSUB)\s+(\d+)', clause, re.IGNORECASE):
                            target = int(m.group(1))
                            if target not in line_set:
                                raise RuntimeError(
                                    f"LINE {ln}: {m.group(0)} (in IF THEN) — target line not found")
            if isinstance(parsed, ForStmt):
                for_depth += 1
                for_var_stack.append(parsed.var.upper())
            elif isinstance(parsed, NextStmt):
                # A NEXT may close one loop (NEXT / NEXT i) or several (NEXT i, j).
                names = [v.strip() for v in parsed.var.split(',')] if parsed.var.strip() else ['']
                for nm in names:
                    for_depth -= 1
                    if nm and for_var_stack and nm.upper() != for_var_stack[-1]:
                        raise RuntimeError(
                            f"LINE {ln}: NEXT {nm} does not match FOR {for_var_stack[-1]}")
                    if for_var_stack:
                        for_var_stack.pop()
            elif isinstance(parsed, WhileStmt):
                while_depth += 1
            elif isinstance(parsed, WendStmt):
                while_depth -= 1
            elif isinstance(parsed, DoStmt):
                do_depth += 1
            elif isinstance(parsed, LoopStmt):
                do_depth -= 1
        if for_depth > 0:
            raise RuntimeError(f"Unmatched FOR: {for_depth} FOR without NEXT")
        if while_depth > 0:
            raise RuntimeError(f"Unmatched WHILE: {while_depth} WHILE without WEND")
        if do_depth > 0:
            raise RuntimeError(f"Unmatched DO: {do_depth} DO without LOOP")

    # ── Circuit Building (provided by ExecutorMixin) ────────────────

    # Mid-circuit measurement returns this value in Qiskit circuit-build mode.
    # Qiskit defers measurement to simulation time, so the variable cannot hold
    # the actual outcome during circuit construction. For classical feedforward,
    # use LOCC mode with SEND instead.
    MEAS_CIRCUIT_MODE_VALUE = 0

    def _try_exec_meas(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        """Handle MEAS qubit -> var (mid-circuit measurement).

        In standard Aer mode this is a real dynamic-circuit measurement: the
        qubit is measured into a classical register recorded under `var`, which a
        later ``IF var THEN <gate>`` turns into a Qiskit if_test (feedforward) at
        build time. In LOCC mode it routes through the numpy engine as before.
        """
        m = RE_MEAS.match(stmt)
        if not m:
            # Bare MEAS (no "-> bit") is the classic confusion with MEASURE:
            # MEAS is a mid-circuit measurement that must name a classical bit,
            # MEASURE collapses qubits into the result histogram.
            if re.match(r'MEAS\b', stmt, re.IGNORECASE):
                raise ValueError(
                    "MEAS needs a target bit: MEAS <qubit> -> <bit> "
                    "(mid-circuit measurement used by IF). To collapse qubits "
                    "into the result histogram, use MEASURE <qubit>.")
            return False
        qubit = self._resolve_qubit(m.group(1))
        var = m.group(2)
        if self._live is not None:
            bit = self._live.measure(qubit)
            run_vars[var] = bit
            self.variables[var] = bit
            return True
        self._measure_into_bit(qc, qubit, var, 'mc')
        run_vars[var] = self.MEAS_CIRCUIT_MODE_VALUE
        self.variables[var] = self.MEAS_CIRCUIT_MODE_VALUE
        return True

    def _measure_into_bit(self, qc, qubit: int, var: str, prefix: str) -> None:
        """Measure `qubit` into a fresh one-bit register recorded as `var`, so
        IF var THEN ... compiles to feedforward on this outcome. Measuring the
        same name again (in a loop) gets a new register; IF reads the latest."""
        from qiskit.circuit import ClassicalRegister
        taken = {r.name for r in qc.cregs}
        name, k = f'{prefix}_{var}', 1
        while name in taken:
            name, k = f'{prefix}_{var}_{k}', k + 1
        cr = ClassicalRegister(1, name)
        qc.add_register(cr)
        qc.measure(qubit, cr[0])
        self._classical_bits[var] = cr

    def _try_exec_reset(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        """Handle RESET qubit."""
        m = RE_RESET.match(stmt)
        if not m:
            return False
        qubit = self._resolve_qubit(m.group(1))
        if self._live is not None:
            self._live.reset(qubit)
        else:
            qc.reset(qubit)
        return True

    def _try_exec_unitary(self, stmt: str) -> bool:
        """Handle UNITARY name = [[...]] gate definition."""
        m = RE_UNITARY.match(stmt)
        if not m:
            return False
        name = m.group(1).upper()
        try:
            matrix = np.array(self._parse_matrix(m.group(2)), dtype=complex)
            n_qubits = int(np.log2(matrix.shape[0]))
            if matrix.shape != (2**n_qubits, 2**n_qubits):
                raise ValueError("Matrix must be 2^n x 2^n")
            product = matrix @ matrix.conj().T
            if not np.allclose(product, np.eye(matrix.shape[0]), atol=1e-6):
                raise ValueError("Matrix is not unitary (U @ U† ≠ I, atol=1e-6). Use expressions like 1/sqrt(2) instead of truncated decimals.")
            self._custom_gates[name] = matrix
            self.io.writeln(f"UNITARY {name}: {n_qubits}-qubit gate defined")
        except Exception as e:
            self.io.writeln(f"?UNITARY ERROR: {e}")
        return True

    def _try_exec_dim(self, stmt: str) -> bool:
        """Handle DIM name(size) or DIM name(d1, d2, ...) array declaration."""
        m = RE_DIM_MULTI.match(stmt)
        if not m:
            m = RE_DIM.match(stmt)
        if not m:
            return False
        name = m.group(1)
        base = getattr(self, '_option_base', 0)
        dims = [int(d.strip()) for d in m.group(2).split(',')]
        # Inclusive sizing (QBASIC): DIM a(n) spans indices base..n, so the
        # declared top index n is valid (n - base + 1 slots per dimension).
        sizes = [max(0, d - base + 1) for d in dims]
        total = 1
        for s in sizes:
            total *= s
        self.arrays[name] = [('' if name.endswith('$') else 0.0)] * total
        if len(sizes) > 1:
            self._array_dims[name] = sizes
        self._dimmed_arrays.add(name)
        return True

    def _try_exec_dim_type(self, stmt: str) -> bool:
        """Handle DIM name AS TypeName — instantiate a user-defined type."""
        from qubasic_core.engine import RE_DIM_TYPE
        m = RE_DIM_TYPE.match(stmt)
        if not m:
            return False
        var_name = m.group(1)
        type_name = m.group(2).upper()
        if type_name not in self._user_types:
            self.io.writeln(f"?UNDEFINED TYPE: {type_name}")
            return True
        fields = self._user_types[type_name]
        # Create a dict-backed record with typed default values
        defaults = {'INTEGER': 0, 'FLOAT': 0.0, 'STRING': '', 'QUBIT': 0}
        record = {fname: defaults.get(ftype, 0) for fname, ftype in fields}
        self.variables[var_name] = record
        # Also register dotted accessors as variables
        for fname, _ in fields:
            self.variables[f'{var_name}.{fname}'] = record[fname]
        return True

    def _try_exec_redim(self, stmt: str) -> bool:
        """Handle REDIM [PRESERVE] name(size) — resize an array.

        Plain REDIM clears to zeros (QBASIC semantics); REDIM PRESERVE keeps the
        existing elements. Sizing is inclusive (REDIM a(n) spans indices base..n).
        """
        m = RE_REDIM.match(stmt)
        if not m:
            return False
        preserve = bool(m.group(1))
        name = m.group(2)
        base = getattr(self, '_option_base', 0)
        new_len = max(0, int(m.group(3)) - base + 1)
        old = self.arrays.get(name, [])
        if preserve and isinstance(old, list):
            if new_len > len(old):
                self.arrays[name] = old + [0.0] * (new_len - len(old))
            else:
                self.arrays[name] = old[:new_len]
        else:
            self.arrays[name] = [0.0] * new_len
        self._dimmed_arrays.add(name)
        return True

    def _try_exec_erase(self, stmt: str) -> bool:
        """Handle ERASE name — delete a specific array."""
        m = RE_ERASE.match(stmt)
        if not m:
            return False
        name = m.group(1)
        if name in self.arrays:
            del self.arrays[name]
        self._array_dims.pop(name, None)
        self._dimmed_arrays.discard(name)
        return True

    def _try_exec_get(self, stmt: str, run_vars: dict) -> bool:
        """Handle GET var — single keypress input without enter.

        Falls back to reading one character from self.io.read_line
        when not running in a terminal (batch mode, tests, etc.).
        """
        m = RE_GET.match(stmt)
        if not m:
            return False
        var = m.group(1)
        ch = ''
        try:
            import sys as _sys
            if getattr(self.io, 'replay', False):
                ch = self.io.read_line('')[:1]      # a later live shot replays
            elif _sys.stdin.isatty():
                if _sys.platform == 'win32':
                    import msvcrt
                    ch = msvcrt.getwch()
                else:
                    import tty, termios
                    fd = _sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setraw(fd)
                        ch = _sys.stdin.read(1)
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                if hasattr(self.io, 'record'):
                    self.io.record(ch)               # replayed by later shots
            else:
                line = self.io.read_line('')
                ch = line[0] if line else ''
        except (EOFError, KeyboardInterrupt, OSError):
            ch = ''
        if var.endswith('$'):
            run_vars[var] = ch
            self.variables[var] = ch
        else:
            run_vars[var] = float(ord(ch)) if ch else 0.0
            self.variables[var] = run_vars[var]
        return True

    def _try_exec_input(self, stmt: str, run_vars: dict) -> bool:
        """Handle INPUT "prompt", var user input with retry on bad input."""
        m = RE_INPUT.match(stmt)
        if not m:
            return False
        prompt = m.group(1) or m.group(2)
        var = m.group(2)
        for _attempt in range(3):
            try:
                val = self.io.read_line(f"{prompt}? ")
                if var.endswith('$'):
                    run_vars[var] = val
                    self.variables[var] = val
                else:
                    run_vars[var] = float(val) if '.' in val else int(val)
                    self.variables[var] = run_vars[var]
                return True
            except (EOFError, KeyboardInterrupt):
                run_vars[var] = 0
                self.variables[var] = 0
                return True
            except ValueError:
                self.io.writeln("?REDO FROM START")
        run_vars[var] = 0
        self.variables[var] = 0
        return True

    def _try_exec_measure_basis(self, stmt: str, qc, run_vars: dict,
                               *, backend=None) -> bool:
        """Handle MEASURE_X/Y/Z qubit — measurement in a non-computational basis."""
        m = RE_MEASURE_BASIS.match(stmt)
        if not m:
            return False
        basis = m.group(1).upper()
        qubit = self._resolve_qubit(m.group(2))
        var = f"m{basis.lower()}_{qubit}"
        rotation = {'X': ['H'], 'Y': ['SDG', 'H'], 'Z': []}[basis]
        live = self._live
        if live is not None:
            live.flush()
            for g in rotation:
                live.apply_named(g, (), [qubit])
            bit = live.measure(qubit)
            run_vars[var] = bit
            self.variables[var] = bit
            return True
        for g in rotation:
            self._apply_gate(qc, g, [], [qubit])
        self._measure_into_bit(qc, qubit, var, 'meas')
        run_vars[var] = 0
        self.variables[var] = 0
        return True

    # _parse_syndrome provided by ExecutorMixin.

    _PAULI_TO_CONTROLLED = {'X': 'CX', 'Y': 'CY', 'Z': 'CZ'}

    def _try_exec_syndrome(self, stmt: str, qc, run_vars: dict,
                           *, backend=None) -> bool:
        """Handle SYNDROME — non-destructive stabilizer measurement via ancilla."""
        parsed = self._parse_syndrome(stmt, run_vars)
        if parsed is None:
            return False
        pauli_str, qubits, var = parsed
        anc = self.num_qubits - 1
        if anc in qubits:
            raise ValueError(
                f"Qubit {anc} needed as ancilla but appears in stabilizer. "
                f"Increase QUBITS by 1.")
        gates = [('H', [anc])] + [(self._PAULI_TO_CONTROLLED[p], [anc, q])
                                  for p, q in zip(pauli_str, qubits) if p != 'I'] + [('H', [anc])]
        live = self._live
        if live is not None:
            live.flush()
            for g, qs in gates:
                live.apply_named(g, (), qs)
            bit = live.measure(anc)
            live.reset(anc)
            run_vars[var] = bit
            self.variables[var] = bit
            return True
        for g, qs in gates:
            self._apply_gate(qc, g, [], qs)
        self._measure_into_bit(qc, anc, var, 'synd')
        qc.reset(anc)
        run_vars[var] = 0
        self.variables[var] = 0
        return True

    # _exec_line, _RESERVED_KEYWORDS, _RESERVED_NAMES provided by ExecutorMixin.

    def _try_stmt_handlers(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        """Try statement-type handlers in order. Returns True if handled."""
        return (
            self._try_exec_meas(stmt, qc, run_vars, backend=backend)
            or self._try_exec_reset(stmt, qc, run_vars, backend=backend)
            or self._try_exec_measure_basis(stmt, qc, run_vars, backend=backend)
            or self._try_exec_syndrome(stmt, qc, run_vars, backend=backend)
            or self._try_exec_qft(stmt, qc, run_vars, backend=backend)
            or self._try_exec_diffuse(stmt, qc, run_vars, backend=backend)
            or self._try_exec_mcgate(stmt, qc, run_vars, backend=backend)
            or self._try_exec_qaddc(stmt, qc, run_vars, backend=backend)
            or self._try_exec_qadd(stmt, qc, run_vars, backend=backend)
            or self._try_exec_qpe(stmt, qc, run_vars, backend=backend)
            or self._try_exec_evolve(stmt, qc, run_vars, backend=backend)
            or self._try_exec_applychannel(stmt, qc, run_vars, backend=backend)
            or self._try_exec_amplify(stmt, qc, run_vars, backend=backend)
            or self._try_exec_graphstate(stmt, qc, run_vars, backend=backend)
            or self._try_exec_featuremap(stmt, qc, run_vars, backend=backend)
            or self._try_exec_unitary(stmt)
            or self._try_exec_dim(stmt)
            or self._try_exec_dim_type(stmt)
            or self._try_exec_redim(stmt)
            or self._try_exec_erase(stmt)
            or self._try_exec_get(stmt, run_vars)
            or self._try_exec_input(stmt, run_vars)
            or self._try_exec_poke(stmt, run_vars)
            or self._try_exec_sys(stmt)
            or self._exec_print_file(stmt, run_vars)
            or self._exec_input_file(stmt, run_vars)
            or self._exec_lprint(stmt, run_vars)
            or self._try_exec_line_input(stmt, run_vars)
            or self._try_exec_let_str(stmt, run_vars)
            or self._try_exec_print_using(stmt, run_vars)
            or self._try_exec_open_close(stmt)
            or self._try_exec_save_expect(stmt, qc, run_vars)
            or self._try_exec_save_probs(stmt, qc, run_vars)
            or self._try_exec_save_amps(stmt, qc, run_vars)
            or self._try_exec_set_state(stmt, qc)
            or self._try_exec_apply_circuit(stmt, qc, backend=backend)
        )

    def _try_exec_apply_circuit(self, stmt: str, qc, *, backend=None) -> bool:
        """Handle APPLY_CIRCUIT name [@offset] inside program execution."""
        m = re.match(r'APPLY_CIRCUIT\s+(\w+)(?:\s+@(\d+))?', stmt, re.IGNORECASE)
        if not m:
            return False
        name = m.group(1).upper()
        offset = int(m.group(2)) if m.group(2) else 0
        if name not in self.subroutines:
            raise ValueError(f"UNDEFINED CIRCUIT: {name}")
        call_str = f"{name} @{offset}" if offset else name
        self._apply_gate_str(call_str, qc, backend=backend)
        return True

    def _try_exec_poke(self, stmt: str, run_vars: dict) -> bool:
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

    def _try_exec_line_input(self, stmt: str, run_vars: dict) -> bool:
        m = RE_LINE_INPUT.match(stmt)
        if not m:
            return False
        prompt = m.group(1) or m.group(2)
        var = m.group(2)
        try:
            val = self.io.read_line(f"{prompt}? ")
            run_vars[var] = val
            self.variables[var] = val
        except (EOFError, KeyboardInterrupt):
            run_vars[var] = ''
            self.variables[var] = ''
        return True

    def _try_exec_let_str(self, stmt: str, run_vars: dict) -> bool:
        m = RE_LET_STR.match(stmt)
        if not m:
            return False
        name = m.group(1)
        val = self._eval_string_expr(m.group(2), run_vars)
        run_vars[name] = val
        self.variables[name] = val
        return True

    def _try_exec_print_using(self, stmt: str, run_vars: dict) -> bool:
        m = RE_PRINT_USING.match(stmt)
        if not m:
            return False
        fmt = m.group(1)
        vals = [self._eval_with_vars(v.strip(), run_vars)
                for v in m.group(2).split(',') if v.strip()]
        result = fmt
        for val in vals:
            # Find the next format field (run of # and . characters)
            idx = result.find('#')
            if idx < 0:
                break
            end = idx
            while end < len(result) and result[end] in '#.':
                end += 1
            field = result[idx:end]
            field_width = len(field)
            dot_pos = field.find('.')
            if dot_pos >= 0:
                decimals = len(field) - dot_pos - 1
                formatted = f"{val:{field_width}.{decimals}f}"
            else:
                formatted = f"{val:{field_width}.0f}"
            result = result[:idx] + formatted.rjust(field_width) + result[end:]
        self.io.writeln(result)
        return True

    def _try_exec_open_close(self, stmt: str) -> bool:
        m = RE_OPEN.match(stmt)
        if m:
            if getattr(self.io, 'replay', False) and m.group(2).upper() != 'INPUT':
                return True     # later live shots leave the first shot's file alone
            rest = stmt[4:].strip()
            self.cmd_open(rest)
            return True
        m = RE_CLOSE.match(stmt)
        if m:
            self.cmd_close(m.group(1))
            return True
        return False

    def _try_exec_save_expect(self, stmt: str, qc, run_vars: dict) -> bool:
        """SAVE_EXPECT <pauli> <qubits> -> <var> — inline expectation value.

        Also accepts a declared HAMILTONIAN name with no qubit list
        (SAVE_EXPECT HM -> e), recording <HM> for VQE cost functions."""
        from qubasic_core.engine import RE_SAVE_EXPECT
        m = RE_SAVE_EXPECT.match(stmt)
        if not m:
            return False
        pauli_str = m.group(1).upper()
        qubits = [int(q.strip()) for q in m.group(2).replace(',', ' ').split() if q.strip()]
        var = m.group(3)
        try:
            from qiskit.quantum_info import SparsePauliOp
            if not qubits and pauli_str in getattr(self, '_hamiltonians', {}):
                op = self._hamiltonians[pauli_str]
                if op.num_qubits != self.num_qubits:
                    raise ValueError(f"HAMILTONIAN {pauli_str} is {op.num_qubits}-qubit "
                                     f"but QUBITS is {self.num_qubits}")
            else:
                full_pauli = ['I'] * self.num_qubits
                for i, p in enumerate(pauli_str):
                    if i < len(qubits):
                        full_pauli[self.num_qubits - 1 - qubits[i]] = p
                op = SparsePauliOp(''.join(full_pauli))
            if self._live is not None:
                # Live run: the value at this point of this shot (mean over
                # shots once the run ends).
                from qiskit.quantum_info import Statevector
                val = float(np.real(Statevector(self._live.sv).expectation_value(op)))
                self._record_save('exp', var, val)
                run_vars[var] = val
                self.variables[var] = val
                return True
            qc.save_expectation_value(op, list(range(self.num_qubits)), label=f'exp_{var}')
            # Placeholder until the run fills the real value via
            # _extract_save_results. Preserve any prior value instead of
            # zeroing it, so a re-run that re-includes this SAVE line leaves
            # earlier results readable to LET/PRINT during the build pass.
            _prev = self.variables.get(var, 0)
            run_vars[var] = _prev
            self.variables[var] = _prev
        except Exception as e:
            self.io.writeln(f"?SAVE_EXPECT ERROR: {e}")
        return True

    def _try_exec_save_probs(self, stmt: str, qc, run_vars: dict) -> bool:
        """SAVE_PROBS <qubits> -> <var> — inline probability snapshot."""
        from qubasic_core.engine import RE_SAVE_PROBS
        m = RE_SAVE_PROBS.match(stmt)
        if not m:
            return False
        qubits = [int(q.strip()) for q in m.group(1).split(',') if q.strip()]
        var = m.group(2)
        try:
            if self._live is not None:
                from qiskit.quantum_info import Statevector
                probs = Statevector(self._live.sv).probabilities(qubits)
                self._record_save('prob', var, probs)
                run_vars[var] = probs
                self.variables[var] = probs
                self.arrays[var] = probs.tolist()
                return True
            qc.save_probabilities(qubits, label=f'prob_{var}')
            _prev = self.variables.get(var, 0)   # preserve prior value (see SAVE_EXPECT)
            run_vars[var] = _prev
            self.variables[var] = _prev
        except Exception as e:
            self.io.writeln(f"?SAVE_PROBS ERROR: {e}")
        return True

    def _try_exec_save_amps(self, stmt: str, qc, run_vars: dict) -> bool:
        """SAVE_AMPS <indices> -> <var> — save specific amplitudes by index."""
        from qubasic_core.engine import RE_SAVE_AMPS
        m = RE_SAVE_AMPS.match(stmt)
        if not m:
            return False
        indices = [int(q.strip()) for q in m.group(1).split(',') if q.strip()]
        var = m.group(2)
        try:
            if self._live is not None:
                amps = np.array(self._live.sv[indices], dtype=complex)
                self._record_save('amp', var, amps)
                run_vars[var] = amps
                self.variables[var] = amps
                self.arrays[var] = [complex(a) for a in amps]
                return True
            qc.save_amplitudes(indices, label=f'amp_{var}')
            _prev = self.variables.get(var, 0)   # preserve prior value (see SAVE_EXPECT)
            run_vars[var] = _prev
            self.variables[var] = _prev
        except Exception as e:
            self.io.writeln(f"?SAVE_AMPS ERROR: {e}")
        return True

    def _try_exec_set_state(self, stmt: str, qc) -> bool:
        """SET_STATE <statevector> — inject custom statevector mid-circuit.

        Accepts:
          SET_STATE [0.707, 0, 0, 0.707]    — explicit amplitudes
          SET_STATE |+>                       — named state
          SET_STATE |BELL>                    — named entangled state
          SET_STATE |GHZ>                     — GHZ state
        """
        from qubasic_core.engine import RE_SET_STATE
        m = RE_SET_STATE.match(stmt)
        if not m:
            return False
        try:
            sv_expr = m.group(1).strip()
            dim = 2 ** self.num_qubits
            # Try named states first
            if sv_expr.upper() in ('|0>', '|1>', '|+>', '|->', '|BELL>', '|GHZ>',
                                  '|GHZ3>', '|GHZ4>', '|W>', '|W3>'):
                if not _named_state_fits(sv_expr.upper(), self.num_qubits):
                    self.io.writeln(f"  (warning: {sv_expr.upper()} needs more qubits than "
                                    f"{self.num_qubits}; falling back to |0...0>)")
                sv_flat = _resolve_named_state(sv_expr.upper(), self.num_qubits)
            else:
                sv_list = self._parse_matrix(sv_expr)
                sv_flat = np.array(sv_list, dtype=complex).ravel()
            if len(sv_flat) != dim:
                raise ValueError(f"State vector length {len(sv_flat)} != 2^{self.num_qubits} = {dim}")
            norm = float(np.sum(np.abs(sv_flat) ** 2))
            if abs(norm - 1.0) > 1e-6:
                sv_flat = sv_flat / np.sqrt(norm)
                self.io.writeln(f"  (normalized: ||sv||={norm:.6f} -> 1.0)")
            from qiskit.quantum_info import Statevector
            sv_obj = Statevector(sv_flat)
            from qiskit_aer.library import SetStatevector
            qc.append(SetStatevector(sv_obj), list(range(self.num_qubits)))
        except Exception as e:
            self.io.writeln(f"?SET_STATE ERROR: {e}")
        return True

    def cmd_set_state(self, rest: str) -> None:
        """SET_STATE <state> — set the statevector immediately (stores for next RUN)."""
        sv_expr = rest.strip()
        if not sv_expr:
            self.io.writeln("?USAGE: SET_STATE |+> or SET_STATE [0.707, 0, 0, 0.707]")
            return
        try:
            dim = 2 ** self.num_qubits
            if sv_expr.upper() in ('|0>', '|1>', '|+>', '|->', '|BELL>', '|GHZ>',
                                  '|GHZ3>', '|GHZ4>', '|W>', '|W3>'):
                if not _named_state_fits(sv_expr.upper(), self.num_qubits):
                    self.io.writeln(f"  (warning: {sv_expr.upper()} needs more qubits than "
                                    f"{self.num_qubits}; falling back to |0...0>)")
                sv_flat = _resolve_named_state(sv_expr.upper(), self.num_qubits)
            else:
                sv_flat = np.array(self._parse_matrix(sv_expr), dtype=complex).ravel()
            if len(sv_flat) != dim:
                raise ValueError(f"Length {len(sv_flat)} != 2^{self.num_qubits} = {dim}")
            norm = float(np.sum(np.abs(sv_flat) ** 2))
            if abs(norm - 1.0) > 1e-6:
                sv_flat = sv_flat / np.sqrt(norm)
            self.last_sv = sv_flat
            # Persist so the next RUN starts from this state (prepended at build).
            self._pending_set_state = sv_flat
            self.io.writeln(f"STATE SET ({self.num_qubits} qubits) — applied on next RUN")
        except Exception as e:
            self.io.writeln(f"?SET_STATE ERROR: {e}")

    def cmd_set_density(self, rest: str) -> None:
        """SET_DENSITY [[...]] — inject a mixed state (density matrix) for the next RUN.

        Accepts a 2^n x 2^n Hermitian, unit-trace, positive-semidefinite matrix
        (e.g. [[0.5,0],[0,0.5]] for the maximally mixed qubit). Switches the
        method to density_matrix; STATE is unavailable for a mixed state, but
        DENSITY and the measurement histogram reflect it."""
        expr = rest.strip()
        if not expr:
            self.io.writeln("?USAGE: SET_DENSITY [[a,b],[c,d]]  (2^n x 2^n, Hermitian, trace 1)")
            return
        try:
            rho = np.array(self._parse_matrix(expr), dtype=complex)
            dim = 2 ** self.num_qubits
            if rho.shape != (dim, dim):
                raise ValueError(f"matrix is {rho.shape}, need ({dim}, {dim}) for {self.num_qubits} qubit(s)")
            if not np.allclose(rho, rho.conj().T, atol=1e-6):
                raise ValueError("matrix is not Hermitian")
            tr = complex(np.trace(rho))
            if abs(tr - 1.0) > 1e-6:
                if abs(tr) < 1e-12:
                    raise ValueError("matrix has zero trace")
                rho = rho / tr
                self.io.writeln(f"  (normalized: trace {tr.real:.4f} -> 1.0)")
            evals = np.linalg.eigvalsh(rho)
            if evals.min() < -1e-6:
                raise ValueError(f"matrix is not positive semidefinite (min eigenvalue {evals.min():.4f})")
            self._pending_set_density = rho
            self._pending_set_state = None
            self.last_sv = None
            self.sim_method = 'density_matrix'
            self._circuit_cache_key = None
            purity = float(np.real(np.trace(rho @ rho)))
            self.io.writeln(f"DENSITY MATRIX SET ({self.num_qubits} qubits, purity {purity:.4f}) "
                            f"— applied on next RUN (method=density_matrix)")
        except Exception as e:
            self.io.writeln(f"?SET_DENSITY ERROR: {e}")

    # _split_colon_stmts, _substitute_vars, _expand_statement,
    # _offset_qubits, _apply_gate_str, _tokenize_gate, _resolve_qubit,
    # _GATE_DISPATCH, _apply_gate provided by ExecutorMixin.

    # ── Display ───────────────────────────────────────────────────────

    _RE_PRINT_QUBIT = re.compile(r'QUBIT\s*\((.+)\)\s*$', re.IGNORECASE)
    _RE_PRINT_ENT = re.compile(r'ENTANGLEMENT\s*\((.+)\)\s*$', re.IGNORECASE)

    def _current_state(self) -> tuple[Any, int, list[int] | None]:
        """(statevector, qubit count, LOCC register sizes) at the current point
        of execution: the live per-shot state, the LOCC joint state, the state
        of the circuit built so far during RUN, or the last run's state."""
        live = getattr(self, '_live', None)
        if live is not None:
            return live.sv, live.n, None
        if self.locc_mode and self.locc is not None and self.locc.joint:
            return self.locc.sv, self.locc.n_total, list(self.locc.sizes)
        qc = getattr(self, '_build_qc', None)
        if qc is not None:
            return (self._extract_state_now(qc.copy(), self.sim_precision, self._seed),
                    qc.num_qubits, None)
        return self.last_sv, self.num_qubits, None

    def _try_quantum_print(self, text: str, run_vars: dict) -> str | None:
        """Quantum PRINT forms: STATE (Dirac notation), QUBIT(q) (P(1) and the
        Bloch vector), ENTANGLEMENT(a, b) (entanglement of formation of the
        two-qubit reduced state), and @REG (one LOCC register). Each reads the
        state at the point the PRINT executes. Returns None for other text."""
        item = text.strip()
        upper = item.upper()
        if upper.startswith('@') and self.locc_mode and self.locc is not None:
            reg = upper[1:].strip()
            if reg in self.locc.names:
                if self.locc.joint:
                    return self._format_dirac(self.locc.sv, self.locc.n_total,
                                              list(self.locc.sizes))
                return self._format_dirac(self.locc.svs[reg], self.locc.get_size(reg))
            return None
        m_q = self._RE_PRINT_QUBIT.match(item)
        m_e = self._RE_PRINT_ENT.match(item)
        if upper != 'STATE' and not m_q and not m_e:
            return None
        if self.locc_mode and self.locc is not None and not self.locc.joint:
            return "<SPLIT registers have no joint state: PRINT @A, @B, ...>"
        sv, n, segments = self._current_state()
        if sv is None:
            return "<no state>"
        if upper == 'STATE':
            return self._format_dirac(sv, n, segments)
        from qubasic_core import statetools as st
        if m_q:
            q = int(self._eval_with_vars(m_q.group(1), run_vars))
            if q < 0 or q >= n:
                raise ValueError(f"QUBIT {q} OUT OF RANGE (0-{n - 1})")
            x, y, z = st.bloch(sv, q, n)
            return f"q{q}: P(1)={(1 - z) / 2:.4f} Bloch=({x:.3f},{y:.3f},{z:.3f})"
        args = self._split_arg_list(m_e.group(1))
        if len(args) != 2:
            raise ValueError("ENTANGLEMENT takes two qubits: ENTANGLEMENT(a, b)")
        qa, qb = (int(self._eval_with_vars(a, run_vars)) for a in args)
        if qa == qb or not (0 <= qa < n and 0 <= qb < n):
            raise ValueError(f"ENTANGLEMENT needs two different qubits in 0-{n - 1}")
        c = st.concurrence(st.subsystem_rdm(sv, [qa, qb], n))
        return f"E(q{qa},q{qb}) = {st.eof_from_concurrence(c):.6f} ebits (concurrence {c:.6f})"

    def _format_dirac(self, sv, n_qubits: int, segments: list[int] | None = None) -> str:
        """Dirac notation of the first 16 non-zero amplitudes, in the active
        OPTION ENDIAN order; `segments` splits a LOCC joint state into its
        registers (A|B|...)."""
        from qubasic_core.engine import AMPLITUDE_THRESHOLD
        from qubasic_core.statetools import nonzero_indices
        flat = np.asarray(sv).reshape(-1)
        idx, total = nonzero_indices(flat, AMPLITUDE_THRESHOLD, 16)
        parts = []
        for i in idx:
            amp = complex(flat[int(i)])
            bits = format(int(i), f'0{n_qubits}b')
            if segments:
                segs, pos = [], len(bits)
                for size in segments:
                    segs.append(bits[pos - size:pos])
                    pos -= size
                bits = '|'.join(segs)
            state = self._bits(bits)
            if abs(amp.imag) < AMPLITUDE_THRESHOLD:
                if abs(amp.real - 1.0) < 1e-6:
                    parts.append(f"|{state}\u27E9")
                elif abs(amp.real + 1.0) < 1e-6:
                    parts.append(f"-|{state}\u27E9")
                else:
                    parts.append(f"{amp.real:+.4f}|{state}\u27E9")
            else:
                parts.append(f"({amp.real:.3f}{amp.imag:+.3f}j)|{state}\u27E9")
        if total > len(parts):
            parts.append("...")
        return " ".join(parts) if parts else "|0\u27E9"

    # cmd_state, cmd_hist, cmd_probs, cmd_bloch, cmd_circuit provided by StateDisplayMixin.

    # cmd_noise provided by NoiseMixin.

    # cmd_expect, cmd_entropy, cmd_csv, cmd_density provided by AnalysisMixin / FileIOMixin.

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

    # cmd_export provided by FileIOMixin.

    def cmd_decompose(self) -> None:
        """Show gate count breakdown of last circuit."""
        if self.last_circuit is None:
            self.io.writeln("?NO CIRCUIT — RUN first")
            return
        ops: dict[str, int] = {}
        for inst in self.last_circuit.data:
            name = inst.operation.name
            ops[name] = ops.get(name, 0) + 1
        self.io.writeln(f"\n  Circuit: {self.last_circuit.num_qubits} qubits, "
                       f"depth {self.last_circuit.depth()}, {self.last_circuit.size()} gates")
        for name, count in sorted(ops.items(), key=lambda x: -x[1]):
            bar = '█' * min(count, 40)
            self.io.writeln(f"    {name:>10}  {count:>4}  {bar}")
        self.io.writeln('')

    # cmd_include, cmd_sweep provided by FileIOMixin / SweepMixin.

    # LOCC commands (cmd_locc, cmd_send, cmd_share) provided by LOCCMixin.

    # cmd_bench, cmd_ram, cmd_dir provided by AnalysisMixin / FileIOMixin.

    # cmd_clear provided by ProgramMgmtMixin.

    def cmd_undo(self) -> None:
        return self.cmd_undo_preview()

    # LOCC execution (_locc_run, _locc_exec_line, etc.) provided by LOCCMixin.

    # Control flow helpers (_cf_*, _exec_control_flow) provided by ControlFlowMixin.

    # ── Help ──────────────────────────────────────────────────────────

    _EXPERIMENTAL_CMDS = frozenset({
        'PROBE',
    })
    _PARTIAL_CMDS = frozenset({
        'SAMPLE', 'ESTIMATE',  # SamplerV2/EstimatorV2 wrappers
    })

    def cmd_help(self, rest: str = '') -> None:
        arg = rest.strip().upper()
        if arg == 'STATUS':
            self.io.writeln("\n  Command Implementation Status:")
            all_cmds = sorted(set(self._CMD_NO_ARG.keys()) | set(self._CMD_WITH_ARG.keys()))
            for cmd in all_cmds:
                if cmd in self._EXPERIMENTAL_CMDS:
                    tag = 'experimental'
                elif cmd in self._PARTIAL_CMDS:
                    tag = 'partial'
                else:
                    tag = 'native'
                self.io.writeln(f"    {cmd:20s} [{tag}]")
            self.io.writeln(f"\n  {len(all_cmds)} commands registered")
            return
        # HELP <command> / HELP <gate> — show one entry's description.
        if arg:
            method_name = self._CMD_WITH_ARG.get(arg) or self._CMD_NO_ARG.get(arg)
            if method_name:
                method = getattr(type(self), method_name, None)
                doc = (getattr(method, '__doc__', '') or '').strip()
                self.io.writeln(f"\n  {arg}: {doc or '(no description)'}\n")
                return
            gate_key = GATE_ALIASES.get(arg, arg)
            if gate_key in GATE_TABLE:
                n_params, n_qubits = GATE_TABLE[gate_key]
                self.io.writeln(f"\n  {arg}: gate — {n_params} parameter(s), {n_qubits} qubit(s)\n")
                return
            self.io.writeln(f"  No help entry for '{arg}'. Type HELP for the full reference.")
            self._suggest_command(arg)
            return
        self.io.writeln(HELP_TEXT)
        all_cmds = sorted(set(self._CMD_NO_ARG.keys()) | set(self._CMD_WITH_ARG.keys()))
        all_gates = sorted(g for g in GATE_TABLE if g not in GATE_ALIASES)
        self.io.writeln(f"        ALL COMMANDS: {', '.join(all_cmds)}")
        self.io.writeln(f"        ALL GATES: {', '.join(all_gates)}")

    def cmd_probe(self, rest: str = '') -> None:
        """PROBE — exercise CPU, noise, LOCC, and conditional control end to end."""
        results = []
        t0 = time.time()

        # 1. CPU statevector: Bell state
        self.cmd_new(silent=True)
        self.num_qubits = 2
        self.shots = 100
        self.process('10 H 0', track_undo=False)
        self.process('20 CX 0,1', track_undo=False)
        self.process('30 MEASURE', track_undo=False)
        self._noise_model = None
        self._noise_depol_p = 0.0
        self.sim_device = 'CPU'
        self.cmd_run()
        bell_ok = self.last_counts and set(self.last_counts.keys()) <= {'00', '11'}
        results.append(('CPU Bell', bell_ok))

        # 2. Noise: depolarizing on single qubit
        self.cmd_new(silent=True)
        self.num_qubits = 1
        self.shots = 200
        self.cmd_noise('depolarizing 0.3')
        self.process('10 X 0', track_undo=False)
        self.process('20 MEASURE', track_undo=False)
        self.cmd_run()
        noise_ok = self.last_counts and '0' in self.last_counts
        results.append(('Noise depol', noise_ok))
        self.cmd_noise('off')

        # 3. LOCC JOINT: Bell share
        self.cmd_new(silent=True)
        self.cmd_locc('JOINT 1 1')
        self.shots = 100
        self.process('10 SHARE A 0, B 0', track_undo=False)
        self.process('20 MEASURE', track_undo=False)
        self.cmd_run()
        locc_ok = self.last_counts and all(
            s.split('|')[0] == s.split('|')[1]
            for s in self.last_counts if '|' in s)
        results.append(('LOCC JOINT', locc_ok))

        # 4. Conditional feedforward: SEND + IF
        self.cmd_new(silent=True)
        self.cmd_locc('JOINT 2 2')
        self.shots = 100
        self.process('10 @A H 0', track_undo=False)
        self.process('20 SEND A 0 -> m', track_undo=False)
        self.process('30 IF m THEN @B X 0', track_undo=False)
        self.process('40 MEASURE', track_undo=False)
        self.cmd_run()
        cond_ok = self.last_counts is not None and len(self.last_counts) > 0
        results.append(('Conditional ctrl', cond_ok))
        self.cmd_locc('OFF')

        # 5. Combined: noise + LOCC + mid-circuit + correction
        self.cmd_new(silent=True)
        self.cmd_noise('depolarizing 0.1')
        self.cmd_locc('JOINT 2 2')
        self.shots = 200
        self.process('10 @A H 0', track_undo=False)
        self.process('20 SHARE A 1, B 0', track_undo=False)
        self.process('30 SEND A 0 -> m', track_undo=False)
        self.process('40 IF m THEN @B X 0', track_undo=False)
        self.process('50 MEASURE', track_undo=False)
        self.cmd_run()
        combo_ok = self.last_counts is not None and len(self.last_counts) > 0
        results.append(('Noise+LOCC+SEND+IF', combo_ok))
        self.cmd_noise('off')
        self.cmd_locc('OFF')

        # 6. GPU probe (non-fatal)
        gpu_ok = None
        try:
            _b = AerSimulator(method='statevector', device='GPU')
            from qiskit import QuantumCircuit as _QC
            _pqc = _QC(1); _pqc.h(0); _pqc.measure_all()
            _b.run(transpile(_pqc, _b), shots=1).result()
            gpu_ok = True
        except Exception:
            gpu_ok = False
        results.append(('GPU', gpu_ok))

        dt = time.time() - t0
        self.io.writeln(f"\n  PROBE results ({dt:.2f}s):")
        all_pass = True
        for name, ok in results:
            status = 'PASS' if ok else ('SKIP' if ok is None else 'FAIL')
            if not ok and ok is not None:
                all_pass = False
            self.io.writeln(f"    {name:20s} {status}")
        self.io.writeln(f"\n  {'ALL PASSED' if all_pass else 'SOME FAILED'}")

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

    # ── Banner ────────────────────────────────────────────────────────

    def print_banner(self) -> None:
        import platform
        try:
            import qiskit
            qver = qiskit.__version__
        except Exception:
            qver = '?'
        from qubasic_core import capacity as cap
        ram = _get_ram_gb()
        ram_str = f"{ram[0]:.0f} GB RAM" if ram else "RAM unknown"
        max_d = cap.max_qubits('statevector', 'double')
        max_s = cap.max_qubits('statevector', 'single')
        if max_d is None:
            width = "statevector width unknown (install psutil)"
        else:
            width = (f"max ~{max_d} statevector"
                     + (f" ({max_s} single precision)" if max_s and max_s > max_d else ""))
        if hasattr(QBasicTerminal, '_gpu_cache'):
            gpu_str = QBasicTerminal._gpu_cache
        else:
            try:
                gpu_str = ""
                import subprocess
                r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=2)
                if r.returncode == 0 and r.stdout.strip():
                    gpu_str = f" | GPU: {r.stdout.strip().split(chr(10))[0]}"
                QBasicTerminal._gpu_cache = gpu_str
            except Exception:
                gpu_str = ""
                QBasicTerminal._gpu_cache = gpu_str
        info_line = f"Python {platform.python_version()} | Qiskit {qver} | {ram_str}{gpu_str}"
        config_line = (f"{self.num_qubits} qubits | {self.shots} shots | "
                       f"{width}, 4096 stabilizer")
        self.io.writeln(BANNER_ART.format(info_line=info_line, config_line=config_line))

