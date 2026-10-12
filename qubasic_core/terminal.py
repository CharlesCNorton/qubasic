"""QUBASIC terminal: QBasicTerminal, the REPL and the command dispatcher.

QBasicTerminal is Engine (the program and session state, engine_state.py)
plus one mixin per subsystem; host.py declares the attributes and methods
they share, so mypy checks every cross-mixin reference. A program runs
through one of three execution paths, all driven by the statement list
program.py compiles:

    circuit build   RunPipelineMixin.cmd_run -> ExecutorMixin.build_circuit
                    -> a Qiskit circuit on Aer (dynamic IF on MEAS bits)
    live            LiveMixin: shot by shot on a numpy state, for programs
                    that read measurement results as numbers
    LOCC            LOCCExecutionMixin: numpy registers with SEND and SHARE

Control flow, assignments and the keyword statements are shared by all
three (control_flow.py, classic.py, subs.py, stmt_handlers.py); only gate
application differs. LQUBITS adds the encoded paths (logical.py,
encoded.py).
"""

import sys
import os
import time
import warnings as _warnings
from typing import Any

# Quiet the third-party dependency-version notice (urllib3/chardet) that
# qiskit's transitive imports emit, so it doesn't precede the banner.
_warnings.filterwarnings('ignore', message=r".*match a supported version.*")

from qiskit import transpile
from qiskit_aer import AerSimulator
import numpy as np

from qubasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    _get_ram_gb, MAX_UNDO_STACK,
    RE_LINE_NUM,
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
from qubasic_core.qol import QoLMixin, tip_of_the_day
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
from qubasic_core.run_pipeline import RunPipelineMixin, _has_channel, _circuit_fingerprint  # noqa: F401
from qubasic_core.stmt_handlers import (  # noqa: F401
    StatementMixin, _resolve_named_state, _named_state_fits,
)
from qubasic_core.config_commands import ConfigMixin
from qubasic_core.editing import EditingMixin
from qubasic_core.variables import VariableMixin
from qubasic_core.variational import VariationalMixin
from qubasic_core.oracles import OracleMixin
from qubasic_core.lattice import LatticeMixin
from qubasic_core.encoded import EncodedMixin
from qubasic_core.errors import QBasicError
from qubasic_core.parser import parse_stmt
from qubasic_core.engine_state import Engine
from qubasic_core.help_text import HELP_TEXT, BANNER_ART


# ═══════════════════════════════════════════════════════════════════════
# The Terminal
# ═══════════════════════════════════════════════════════════════════════

class QBasicTerminal(RunPipelineMixin, StatementMixin, VariableMixin, ConfigMixin, EditingMixin,
                     Engine, ExecutorMixin, ExpressionMixin, DisplayMixin, DemoMixin,
                     LOCCMixin, ControlFlowMixin, FileIOMixin, AnalysisMixin,
                     SweepMixin, MemoryMixin, StringMixin, ScreenMixin, ClassicMixin,
                     SubroutineMixin, DebugMixin, ProgramMgmtMixin, ProfilerMixin,
                     NoiseMixin, StateDisplayMixin, QoLMixin, AlgorithmsMixin,
                     DynamicsMixin, QECMixin, QEC2Mixin, LogicalMixin,
                     BenchmarkingMixin, Algorithms2Mixin,
                     PauliPropMixin, QuditMixin, BosonicMixin, ResourceMixin, LiveMixin,
                     VariationalMixin, OracleMixin, LatticeMixin, EncodedMixin):
    """The QUBASIC session: program, state, REPL and command dispatch.

    Mixins, by subsystem (each module's docstring describes its own):

        language     EditingMixin, VariableMixin, ExpressionMixin, StringMixin,
                     ControlFlowMixin, ClassicMixin, SubroutineMixin,
                     StatementMixin, DebugMixin, ProgramMgmtMixin, FileIOMixin
        execution    RunPipelineMixin, ExecutorMixin, LiveMixin, LOCCMixin,
                     NoiseMixin, ConfigMixin, MemoryMixin, ProfilerMixin
        display      DisplayMixin, StateDisplayMixin, ScreenMixin, QoLMixin,
                     DemoMixin, AnalysisMixin, SweepMixin
        quantum      AlgorithmsMixin, Algorithms2Mixin, VariationalMixin,
                     OracleMixin, DynamicsMixin, PauliPropMixin, BenchmarkingMixin,
                     ResourceMixin, QuditMixin, BosonicMixin
        QEC          QECMixin, QEC2Mixin, LatticeMixin, LogicalMixin, EncodedMixin

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
        self._init_classic()
        self._init_variables()
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
        import importlib
        try:
            # (readline is POSIX-only; pyreadline3 provides it on Windows)
            readline: Any = importlib.import_module('readline')
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
                    readline = importlib.import_module('readline')
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
        'QAOA': 'cmd_qaoa', 'VQE': 'cmd_vqe', 'GROVER': 'cmd_grover', 'SIMON': 'cmd_simon',
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
        """HELP [command | gate | STATUS] — the command reference, one entry, or the implementation status of every command."""
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

