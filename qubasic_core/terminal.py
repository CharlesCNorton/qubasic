"""QUBASIC terminal: Engine (the program and session state) and
QBasicTerminal, the REPL and the command dispatcher.

QBasicTerminal is Engine plus one mixin per subsystem; host.py declares the
attributes and methods they share, so mypy checks every cross-mixin
reference. A program runs through one of three execution paths, all driven
by the statement list parser.py compiles:

    circuit build   RunPipelineMixin.cmd_run -> ExecutorMixin.build_circuit
                    -> a Qiskit circuit on Aer (dynamic IF on MEAS bits)
    live            LiveMixin: shot by shot on a numpy state, for programs
                    that read measurement results as numbers
    LOCC            LOCCExecutionMixin: numpy registers with SEND and SHARE

Control flow, assignments and the keyword statements are shared by all
three (interpreter.py); only gate application differs. LQUBITS adds the
encoded paths (logical.py).
"""

from __future__ import annotations

import warnings as _warnings

# Quiet the third-party dependency-version notice (urllib3/chardet) that
# qiskit's transitive imports emit, so it doesn't precede the banner.
_warnings.filterwarnings('ignore', message=r".*match a supported version.*")

import os
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qubasic_core.algorithms import (
    EstimationMixin,
    FactoringMixin,
    HHLMixin,
    OptimizeMixin,
    OracleMixin,
    PhaseMixin,
    PrimitivesMixin,
    QMLMixin,
    VariationalMixin,
)
from qubasic_core.analysis import AnalysisMixin, AssistMixin, DemoMixin, ExportMixin, tip_of_the_day
from qubasic_core.benchmarking import BenchmarkingMixin, ResourceMixin
from qubasic_core.devices import FileIOMixin, GraphicsMixin, ScreenMixin, SoundMixin
from qubasic_core.display import BANNER_ART, HELP_TEXT, DisplayMixin, StateDisplayMixin, SweepMixin, VisualizeMixin
from qubasic_core.editing import EditingMixin, PersistenceMixin, ProgramMgmtMixin
from qubasic_core.engine import (
    DEFAULT_QUBITS,
    DEFAULT_SHOTS,
    MAX_LOOP_ITERATIONS,
    MAX_UNDO_STACK,
    IOPort,
    NullIO,
    QBasicError,
    StdIOPort,
)
from qubasic_core.expression import ExpressionMixin, StringMixin, VariableMixin
from qubasic_core.host import HostState
from qubasic_core.interpreter import (
    ClassicMixin,
    ControlFlowMixin,
    DebugMixin,
    StatementMixin,
    SubroutineMixin,
)
from qubasic_core.locc import LOCCMixin
from qubasic_core.logical import EncodedMixin, LatticeMixin, LogicalMixin
from qubasic_core.machine import ConfigMixin, MemoryMixin, ProfilerMixin
from qubasic_core.noise import MitigationMixin, NoiseMixin
from qubasic_core.parser import GATE_ALIASES, GATE_TABLE, RE_LINE_NUM, parse_stmt, strip_comment
from qubasic_core.physics import BosonicMixin, DynamicsMixin, PauliPropMixin, QuditMixin
from qubasic_core.qec import CodesMixin, DecodingMixin, RatesMixin
from qubasic_core.simtools import ram_gb
from qubasic_core.simulator import ExecutorMixin, LiveMixin, RunPipelineMixin

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit_aer.noise import NoiseModel

    from qubasic_core.locc import LOCCEngine


# ---------------------------------------------------------------------------
# QUBASIC engine state: the interpreter's state container.
#
# Engine initializes the program store, variables, arrays, run settings and
# caches. QBasicTerminal inherits it together with the mixins that run
# programs and commands, so programs run on a QBasicTerminal (or through
# qubasic_core.api); Engine alone holds state.
# ---------------------------------------------------------------------------

class Engine(HostState):
    """The interpreter's state: program, variables, arrays, run settings
    and caches (the mixins' _init_* methods add their own)."""

    def __init__(self, *, io: IOPort | None = None) -> None:
        # Program
        self.program: dict[int, str] = {}
        self._parsed: dict[int, Any] = {}
        self._undo_stack: list[dict[int, str]] = []

        # Configuration
        self.num_qubits: int = DEFAULT_QUBITS
        self.shots: int = DEFAULT_SHOTS
        self.sim_method: str = 'automatic'
        self.sim_device: str = 'CPU'
        # Amplitude precision for Aer and the numpy engines: 'double'
        # (complex128) or 'single' (complex64, half the memory).
        self.sim_precision: str = 'double'
        self._noise_model: NoiseModel | None = None
        self._noise_depol_p: float = 0.0
        # Noise channel propagated to the numpy LOCC engine (depolarizing,
        # amplitude_damping, or phase_flip; 'none' if unsupported there).
        self._noise_locc_type: str = 'none'
        self._noise_locc_param: float = 0.0
        # The NOISE model as its command arguments, and the DEVICE that built
        # the model (topology, qubits, T1, T2), for SAVE round-tripping.
        self._noise_spec: str | None = None
        self._device_spec: str | None = None
        self._seed: int | None = None
        self._max_iterations: int = MAX_LOOP_ITERATIONS
        self._include_depth: int = 0

        # Variables and arrays
        self.variables: dict[str, Any] = {}
        self.arrays: dict[str, Any] = {}
        self._array_dims: dict[str, list[int]] = {}
        # Arrays declared with DIM/REDIM enforce their bounds on write;
        # arrays created implicitly by first assignment keep auto-growing.
        self._dimmed_arrays: set[str] = set()
        # Density matrix (and the measure-free circuit to solve it lazily)
        # for a run with SET_DENSITY, computed on demand by DENSITY.
        self._last_density: np.ndarray | None = None
        self._last_density_qc: QuantumCircuit | None = None

        # Subroutines and registers
        self.subroutines: dict[str, Any] = {}
        self.registers: OrderedDict[str, tuple[int, int]] = OrderedDict()

        # Custom gates
        self._custom_gates: dict[str, Any] = {}

        # Execution state
        self._gosub_stack: list[int] = []
        self.step_mode: bool = False
        self.last_counts: dict[str, int] | None = None
        # The last run's statevector. A run that does not keep the state in
        # memory leaves a zero-argument source that computes it on first read
        # (see the last_sv property).
        self._last_sv: Any = None
        self._state_source: Callable[[], Any] | None = None
        self.last_circuit: QuantumCircuit | None = None
        self._circuit_cache_key: Any = None
        self._circuit_cache: tuple[QuantumCircuit, AerSimulator] | None = None
        self._pending_set_state: np.ndarray | None = None  # immediate SET_STATE for next RUN
        self._poke_state_prep: dict[Any, Any] = {}  # qubit prep from POKE $0100

        # LOCC
        self.locc: LOCCEngine | None = None
        self.locc_mode: bool = False

        # I/O
        self.io: IOPort = io or StdIOPort()

        # Security
        self.agent_mode: bool = False
        self._include_stack: list[str] = []

        # User-defined types
        self._user_types: dict[str, list[tuple[str, str]]] = {}
        self._pending_type: dict | None = None

        # Timing
        self._start_time: float = time.time()

    @property
    def last_sv(self) -> Any:
        """The last run's statevector, computed on first read when the run
        deferred it (a deferred source runs at most once)."""
        if self._last_sv is None and self._state_source is not None:
            source, self._state_source = self._state_source, None
            self._last_sv = source()
        return self._last_sv

    @last_sv.setter
    def last_sv(self, value: Any) -> None:
        self._last_sv = value
        self._state_source = None

    def _defer_state(self, source: Any) -> None:
        """Record a zero-argument callable that produces the state on demand."""
        self._last_sv = None
        self._state_source = source

    @contextmanager
    def _silenced(self) -> Iterator[None]:
        """Run a block with its output discarded (the inner runs of COMPARE,
        PLOT, ANIMATE, MINIMIZE, GRADIENT and STATS)."""
        old, self.io = self.io, NullIO()
        try:
            yield
        finally:
            self.io = old

    def _get_parsed(self, line_num: int) -> Any:
        """Get parsed Stmt for a line, lazily parsing if needed.

        Re-parses if the program text has changed since the last parse.
        """
        raw = self.program.get(line_num, '')
        p = self._parsed.get(line_num)
        if p is None or p.raw != raw:
            p = parse_stmt(raw)
            self._parsed[line_num] = p
        return p

    def clear(self) -> None:
        """Reset all state (equivalent to cmd_new)."""
        self.program.clear()
        self._parsed.clear()
        self.subroutines.clear()
        self.registers.clear()
        self.variables.clear()
        self.arrays.clear()
        self._array_dims.clear()
        self._dimmed_arrays.clear()
        self._last_density = None
        self._last_density_qc = None
        self.last_counts = None
        self.last_sv = None
        self.last_circuit = None
        self._circuit_cache_key = None
        self._circuit_cache = None
        self._pending_set_state = None
        self._pending_set_density: Any = None
        self._poke_state_prep = {}

    def _invalidate_run_state(self) -> None:
        """Drop cached results that describe a now-stale configuration.

        Called when the qubit count changes: last_sv, last_counts,
        last_circuit, and the transpile cache all assume the previous
        num_qubits, so reading them afterwards would mislabel (or, for
        the reshape in _peek_qubit/_bloch_vector, crash) the state.
        """
        self.last_counts = None
        self.last_sv = None
        self.last_circuit = None
        self._circuit_cache_key = None
        self._circuit_cache = None


# ---------------------------------------------------------------------------
# QUBASIC terminal: QBasicTerminal, the REPL and the command dispatcher.
#
# QBasicTerminal is Engine (the program and session state, engine_state.py)
# plus one mixin per subsystem; host.py declares the attributes and methods
# they share, so mypy checks every cross-mixin reference. A program runs
# through one of three execution paths, all driven by the statement list
# program.py compiles:
#
#     circuit build   RunPipelineMixin.cmd_run -> ExecutorMixin.build_circuit
#                     -> a Qiskit circuit on Aer (dynamic IF on MEAS bits)
#     live            LiveMixin: shot by shot on a numpy state, for programs
#                     that read measurement results as numbers
#     LOCC            LOCCExecutionMixin: numpy registers with SEND and SHARE
#
# Control flow, assignments and the keyword statements are shared by all
# three (control_flow.py, classic.py, subs.py, stmt_handlers.py); only gate
# application differs. LQUBITS adds the encoded paths (logical.py,
# encoded.py).
# ---------------------------------------------------------------------------


# ═══════════════════════════════════════════════════════════════════════
# The Terminal
# ═══════════════════════════════════════════════════════════════════════

class QBasicTerminal(RunPipelineMixin, StatementMixin, VariableMixin, ConfigMixin, EditingMixin,
                     Engine, ExecutorMixin, ExpressionMixin, DisplayMixin, DemoMixin,
                     LOCCMixin, ControlFlowMixin, FileIOMixin, PersistenceMixin, ExportMixin, AnalysisMixin,
                     SweepMixin, MemoryMixin, StringMixin, ScreenMixin, GraphicsMixin, SoundMixin, ClassicMixin,
                     SubroutineMixin, DebugMixin, ProgramMgmtMixin, ProfilerMixin,
                     NoiseMixin, MitigationMixin, StateDisplayMixin, VisualizeMixin, AssistMixin,
                     PrimitivesMixin, OptimizeMixin, DynamicsMixin, CodesMixin, DecodingMixin, RatesMixin, LogicalMixin,
                     BenchmarkingMixin, EstimationMixin, PhaseMixin, QMLMixin, FactoringMixin, HHLMixin,
                     PauliPropMixin, QuditMixin, BosonicMixin, ResourceMixin, LiveMixin,
                     VariationalMixin, OracleMixin, LatticeMixin, EncodedMixin):
    """The QUBASIC session: program, state, REPL and command dispatch.

    Mixins, by subsystem (each module's docstring describes its own):

        language     EditingMixin, VariableMixin, ExpressionMixin, StringMixin,
                     GraphicsMixin, SoundMixin,
                     ControlFlowMixin, ClassicMixin, SubroutineMixin,
                     StatementMixin, DebugMixin, ProgramMgmtMixin, FileIOMixin, PersistenceMixin
        execution    RunPipelineMixin, ExecutorMixin, LiveMixin, LOCCMixin,
                     NoiseMixin, MitigationMixin, ConfigMixin, MemoryMixin, ProfilerMixin
        display      DisplayMixin, StateDisplayMixin, ScreenMixin, VisualizeMixin,
                     AssistMixin, ExportMixin, DemoMixin, AnalysisMixin, SweepMixin
        quantum      PrimitivesMixin, OptimizeMixin, EstimationMixin, PhaseMixin, QMLMixin,
                     FactoringMixin, HHLMixin, VariationalMixin, OracleMixin,
                     DynamicsMixin, PauliPropMixin, BenchmarkingMixin,
                     ResourceMixin, QuditMixin, BosonicMixin
        QEC          CodesMixin, DecodingMixin, RatesMixin, LatticeMixin,
                     LogicalMixin, EncodedMixin

    This class keeps the REPL, the command tables, HELP, PROBE and the banner.
    """

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
        self._init_graphics()
        self._init_sound()
        self._init_classic()
        self._init_variables()
        self._init_subs()
        self._init_debug()
        self._init_program_mgmt()
        self._init_profiler()
        self._init_file_handles()
        self._init_assist()
        self._init_dynamics()
        self._init_mitigation()
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

    _FILE_COMMANDS = ('SAVE', 'LOAD', 'INCLUDE', 'IMPORT', 'CHAIN', 'MERGE', 'LOADQASM')

    def _completions(self, text: str, line: str = '') -> list[str]:
        """Tab completions for `text` on input `line`: .qb file names after a
        command that reads or writes one, otherwise the commands, gates,
        statement keywords, DEF names, variables and registers it begins."""
        words = line.split()
        naming_a_file = (bool(words) and words[0].upper() in self._FILE_COMMANDS
                         and (len(words) > 1 or line.endswith(' ')))
        if naming_a_file:
            import glob
            pattern = text + '*' if text.lower().endswith(('.qb', '.bas')) else text + '*.qb'
            return sorted(glob.glob(pattern))
        from qubasic_core.parser import STATEMENTS
        up = text.upper()
        keywords = set(self._CMD_WITH_ARG) | set(self._CMD_NO_ARG) | set(GATE_TABLE)
        keywords |= {kw.split()[0] for _, kw, _, _ in STATEMENTS if kw[:1].isalpha()}
        names = [k for k in sorted(keywords) if k.startswith(up)]
        names += sorted(s for s in self.subroutines if s.startswith(up))
        names += sorted(v for v in self.variables if v.upper().startswith(up) and not v.startswith('_'))
        names += sorted(r for r in self.registers if r.upper().startswith(up))
        return names

    def _setup_readline(self) -> None:
        """Set up command history and tab completion (readline, or pyreadline3
        on Windows); without either the REPL reads plain lines."""
        import importlib
        try:
            readline: Any = importlib.import_module('readline')
        except ImportError:
            return

        def completer(text: str, state: int) -> str | None:
            try:
                matches = self._completions(text, readline.get_line_buffer())
            except Exception:
                return None
            return matches[state] + ' ' if state < len(matches) else None

        readline.set_completer(completer)
        readline.parse_and_bind('tab: complete')
        readline.set_completer_delims(' \t\n')
        try:                                  # F1-F3 load demos, terminal permitting
            readline.parse_and_bind('"\\eOP": "DEMO BELL\\n"')
            readline.parse_and_bind('"\\eOQ": "DEMO GHZ\\n"')
            readline.parse_and_bind('"\\eOR": "DEMO GROVER\\n"')
        except Exception:
            pass

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
        # Immediate command (a trailing ' comment is dropped, as in a program)
        self.dispatch(strip_comment(line))

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
        'DISPLAY': 'cmd_display', 'SHOW': 'cmd_show', 'SAVEWAV': 'cmd_savewav',
        'FILES': 'cmd_files',
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
        'QAOA': 'cmd_qaoa', 'VQE': 'cmd_vqe', 'GROVER': 'cmd_grover', 'SIMON': 'cmd_simon',
        'QPE': 'cmd_qpe', 'QCOUNT': 'cmd_qcount', 'DJ': 'cmd_dj', 'BV': 'cmd_bv',
        'MITIGATE': 'cmd_mitigate', 'ZNE': 'cmd_zne',
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
            from qubasic_core.parser import RE_IMPLICIT_ASSIGN
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
        """BYE | QUIT | EXIT — leave the REPL."""
        raise EOFError

    # ── Help ──────────────────────────────────────────────────────────

    # The banner's GPU line, probed once per process (None until then).
    _gpu_cache: str | None = None

    _EXPERIMENTAL_CMDS = frozenset({
        'PROBE',
    })
    _PARTIAL_CMDS = frozenset({
        'SAMPLE', 'ESTIMATE',  # SamplerV2/EstimatorV2 wrappers
    })

    def cmd_help(self, rest: str = '') -> None:
        """HELP [command | statement | function | gate | STATUS] — the command reference, one entry, or the implementation status of every command."""
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
            from qubasic_core.parser import functions_for, statements_for
            entries = statements_for(arg) + functions_for(arg)
            if entries:
                self.io.writeln('')
                for topic, _, syntax, desc in entries:
                    self.io.writeln(f"  {syntax}\n      {desc} ({topic.lower()})")
                self.io.writeln('')
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
            _pqc = _QC(1)
            _pqc.h(0)
            _pqc.measure_all()
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


# ── Banner ────────────────────────────────────────────────────────

    def print_banner(self) -> None:
        import platform
        try:
            import qiskit
            qver = qiskit.__version__
        except Exception:
            qver = '?'
        from qubasic_core import simtools as cap
        ram = ram_gb()
        ram_str = f"{ram[0]:.0f} GB RAM" if ram else "RAM unknown"
        max_d = cap.max_qubits('statevector', 'double')
        max_s = cap.max_qubits('statevector', 'single')
        if max_d is None:
            width = "statevector width unknown (install psutil)"
        else:
            width = (f"max ~{max_d} statevector"
                     + (f" ({max_s} single precision)" if max_s and max_s > max_d else ""))
        if QBasicTerminal._gpu_cache is not None:
            gpu_str = QBasicTerminal._gpu_cache
        else:
            try:
                gpu_str = ""
                import subprocess
                r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=2)
                if r.returncode == 0 and r.stdout.strip():
                    gpu_str = f" | GPU: {r.stdout.strip().split(chr(10))[0]}"
            except Exception:
                gpu_str = ""
            QBasicTerminal._gpu_cache = gpu_str
        info_line = f"Python {platform.python_version()} | Qiskit {qver} | {ram_str}{gpu_str}"
        config_line = (f"{self.num_qubits} qubits | {self.shots} shots | "
                       f"{width}, 4096 stabilizer")
        self.io.writeln(BANNER_ART.format(info_line=info_line, config_line=config_line))
