"""QBASIC terminal — REPL, commands, circuit building, LOCC execution."""

import sys
import os
import re
import time
from collections import OrderedDict

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
try:
    from qiskit_aer import AerError
except ImportError:
    AerError = None
import numpy as np

from qbasic_core.engine import (
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
from qbasic_core.executor import ExecutorMixin
from qbasic_core.expression import ExpressionMixin
from qbasic_core.display import DisplayMixin
from qbasic_core.demos import DemoMixin
from qbasic_core.locc import LOCCMixin
from qbasic_core.control_flow import ControlFlowMixin
from qbasic_core.file_io import FileIOMixin
from qbasic_core.analysis import AnalysisMixin
from qbasic_core.sweep import SweepMixin
from qbasic_core.memory import MemoryMixin
from qbasic_core.strings import StringMixin
from qbasic_core.screen import ScreenMixin
from qbasic_core.classic import ClassicMixin
from qbasic_core.subs import SubroutineMixin
from qbasic_core.debug import DebugMixin
from qbasic_core.program_mgmt import ProgramMgmtMixin
from qbasic_core.profiler import ProfilerMixin
from qbasic_core.noise_mixin import NoiseMixin
from qbasic_core.state_display import StateDisplayMixin
from qbasic_core.errors import QBasicError, QBasicBuildError, QBasicRangeError
from qbasic_core.io_protocol import StdIOPort
from qbasic_core.parser import parse_stmt
from qbasic_core.engine_state import Engine
from qbasic_core.help_text import HELP_TEXT, BANNER_ART


# ═══════════════════════════════════════════════════════════════════════
# Named quantum states for SET_STATE
# ═══════════════════════════════════════════════════════════════════════

def _resolve_named_state(name: str, n_qubits: int) -> np.ndarray:
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=complex)
    if name == '|0>':
        sv[0] = 1.0
    elif name == '|1>':
        sv[min(1, dim - 1)] = 1.0
    elif name == '|+>':
        sv[0] = 1.0 / np.sqrt(2)
        sv[min(1, dim - 1)] = 1.0 / np.sqrt(2)
    elif name == '|->':
        sv[0] = 1.0 / np.sqrt(2)
        sv[min(1, dim - 1)] = -1.0 / np.sqrt(2)
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


# ═══════════════════════════════════════════════════════════════════════
# The Terminal
# ═══════════════════════════════════════════════════════════════════════

class QBasicTerminal(Engine, ExecutorMixin, ExpressionMixin, DisplayMixin, DemoMixin,
                     LOCCMixin, ControlFlowMixin, FileIOMixin, AnalysisMixin,
                     SweepMixin, MemoryMixin, StringMixin, ScreenMixin, ClassicMixin,
                     SubroutineMixin, DebugMixin, ProgramMgmtMixin, ProfilerMixin,
                     NoiseMixin, StateDisplayMixin):
    # Method organization uses the mixin pattern to reduce apparent class
    # size while keeping everything on QBasicTerminal for import compat.
    #
    # Mixins (defined above):
    #   ExpressionMixin  — AST-based safe eval, no eval()
    #   DemoMixin        — built-in demo circuits
    #
    # Concerns grouped by comment headers below:
    #   REPL, Commands, Run, Circuit Building,
    #   Display, File I/O, Analysis,
    #   LOCC Commands, LOCC Execution,
    #   Control Flow, Help.
    #
    # The Qiskit circuit-build path and the numpy LOCC path share
    # control-flow logic via _exec_control_flow() but diverge at
    # gate application: _apply_gate_str (Qiskit) vs _locc_apply_gate
    # (numpy).  This dualism is necessary because Qiskit does not
    # natively support mid-circuit measurement with classical
    # feedforward in the way LOCC protocols require.
    # _get_parsed is inherited from Engine

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
        # Reject UNC paths (\\server\share) and extended-length paths (\\?\)
        if path.startswith('\\\\'):
            raise ValueError("UNC/extended paths are not allowed")
        return path

    def __init__(self) -> None:
        """Initialize the QBASIC terminal with default configuration."""
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
                return matches[state] + ' ' if state < len(matches) else None
            readline.set_completer(completer)
            readline.parse_and_bind('tab: complete')
            readline.set_completer_delims(' \t\n')
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
        self._setup_readline()
        while True:
            try:
                line = self.io.read_line(self._prompt).strip()
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
        'REGS': 'cmd_regs', 'VARS': 'cmd_vars', 'HELP': 'cmd_help',
        'CIRCUIT': 'cmd_circuit', 'DECOMPOSE': 'cmd_decompose',
        'DENSITY': 'cmd_density', 'LOCCINFO': 'cmd_loccinfo',
        'UNDO': 'cmd_undo', 'BENCH': 'cmd_bench', 'RAM': 'cmd_ram',
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
        # Classic
        'RESTORE': 'cmd_restore',
    }
    _CMD_WITH_ARG = {
        'LIST': 'cmd_list', 'QUBITS': 'cmd_qubits', 'SHOTS': 'cmd_shots',
        'METHOD': 'cmd_method', 'DEF': 'cmd_def', 'REG': 'cmd_reg',
        'LET': 'cmd_let', 'STATE': 'cmd_state', 'BLOCH': 'cmd_bloch',
        'DEMO': 'cmd_demo', 'DELETE': 'cmd_delete', 'RENUM': 'cmd_renum',
        'SAVE': 'cmd_save', 'LOAD': 'cmd_load', 'SWEEP': 'cmd_sweep',
        'INCLUDE': 'cmd_include', 'EXPORT': 'cmd_export',
        'NOISE': 'cmd_noise', 'EXPECT': 'cmd_expect',
        'ENTROPY': 'cmd_entropy', 'CSV': 'cmd_csv', 'LOCC': 'cmd_locc',
        'SEND': 'cmd_send', 'SHARE': 'cmd_share', 'DIR': 'cmd_dir',
        'CLEAR': 'cmd_clear',
        # Memory
        'POKE': 'cmd_poke', 'SYS': 'cmd_sys', 'DUMP': 'cmd_dump', 'WAIT': 'cmd_wait',
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
        # Module, types, network, primitives
        'IMPORT': 'cmd_import', 'TYPE': 'cmd_type',
        'CONNECT': 'cmd_connect', 'DISCONNECT': 'cmd_disconnect',
        'SAMPLE': 'cmd_sample', 'ESTIMATE': 'cmd_estimate',
        # Circuit macros
        'CIRCUIT_DEF': 'cmd_circuit_def', 'APPLY_CIRCUIT': 'cmd_apply_circuit',
    }

    def dispatch(self, line: str) -> None:
        """Parse and execute an immediate command or gate."""
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
            # Try as immediate gate / subroutine
            try:
                self.run_immediate(line)
            except Exception as e:
                self.io.writeln(f"?SYNTAX ERROR: {e}")

    def _quit(self) -> None:
        raise EOFError

    # ── Commands ──────────────────────────────────────────────────────

    def cmd_qubits(self, rest: str) -> None:
        """Set or display the number of qubits. Range: 1 to MAX_QUBITS."""
        if not rest:
            self.io.writeln(f"QUBITS = {self.num_qubits}")
            return
        n = int(rest)
        if n < 1 or n > MAX_QUBITS:
            from qbasic_core.errors import QBasicRangeError
            raise QBasicRangeError(f"RANGE: 1-{MAX_QUBITS}")

        self.num_qubits = n
        self.registers.clear()
        est = _estimate_gb(n)
        self.io.writeln(f"{n} QUBITS ALLOCATED  (~{est:.1f} GB per instance)")
        ram = _get_ram_gb()
        if ram:
            total, avail = ram
            budget = total * RAM_BUDGET_FRACTION
            if est > avail:
                self.io.writeln(f"  WARNING: exceeds available RAM ({avail:.1f} GB free of {total:.0f} GB)")
            elif est > budget * 0.5:
                self.io.writeln(f"  NOTE: uses {est/budget*100:.0f}% of 80% RAM budget ({budget:.1f} GB)")
            if n >= 16 and est > 0:
                max_par = int(budget / est)
                if max_par > 0:
                    self.io.writeln(f"  Max parallel instances in 80% budget: ~{max_par}")

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
            self.io.writeln(f"  methods: {', '.join(methods)}")
            self.io.writeln(f"  devices: CPU, GPU")
            return
        val = rest.strip().upper()
        if val in ('GPU', 'CPU'):
            self.sim_device = val
            self.io.writeln(f"DEVICE = {self.sim_device}")
        else:
            self.sim_method = rest.strip().lower()
            self.io.writeln(f"METHOD = {self.sim_method}")

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
        lines = sorted(self.program.keys())
        for num in lines:
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
            a, b = rest.split('-')
            a, b = int(a.strip()), int(b.strip())
            for k in list(self.program.keys()):
                if a <= k <= b:
                    del self.program[k]
                    self._parsed.pop(k, None)
            self.io.writeln(f"DELETED {a}-{b}")
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
        self.program = new_prog
        self._parsed = {num: parse_stmt(s) for num, s in new_prog.items()}
        self.io.writeln(f"RENUMBERED {len(new_prog)} LINES (start={start}, step={step})")

    # cmd_save, cmd_load provided by FileIOMixin.

    def cmd_def(self, rest: str) -> None:
        """Define a named gate sequence (subroutine), optionally parameterized."""
        # Single-line: DEF BELL = H 0 : CX 0,1
        # Parameterized: DEF ROTATE(angle, q) = RX angle, q : RZ angle, q
        # Multi-line: DEF BEGIN NAME[(params)] ... DEF END
        upper = rest.upper().strip()
        if upper.startswith('BEGIN'):
            return self._def_multiline(rest[5:].strip())

        m = RE_DEF_SINGLE.match(rest)
        if not m:
            self.io.writeln("?USAGE: DEF NAME[(params)] = GATE : GATE : ...")
            self.io.writeln("        DEF BEGIN NAME[(params)]  (multi-line, end with DEF END)")
            return
        name = m.group(1).upper()
        params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
        body = [s.strip() for s in m.group(3).split(':') if s.strip()]
        if name in GATE_TABLE:
            from qbasic_core.errors import QBasicSyntaxError
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
            from qbasic_core.errors import QBasicSyntaxError
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
        from qbasic_core.engine import RE_TYPE_BEGIN
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
        from qbasic_core.engine import RE_TYPE_FIELD, RE_END_TYPE
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
            from qbasic_core.errors import QBasicRangeError
            raise QBasicRangeError(f"NOT ENOUGH QUBITS (need {start+size}, have {self.num_qubits})")

        self.registers[name] = (start, size)
        self.io.writeln(f"REG {name} = qubits {start}-{start+size-1}")

    def cmd_let(self, rest: str) -> None:
        """LET <var> = <expr> — assign a computed value to a variable."""
        # LET angle = PI/4
        m = re.match(r'(\w+)\s*=\s*(.*)', rest)
        if not m:
            self.io.writeln("?USAGE: LET <var> = <expr>")
            return
        name = m.group(1)
        val = self.eval_expr(m.group(2))
        self.variables[name] = val
        self.io.writeln(f"{name} = {val}")

    # cmd_defs, cmd_regs, cmd_vars provided by ProgramMgmtMixin.

    # ── Run ───────────────────────────────────────────────────────────

    def cmd_run(self) -> None:
        """Execute the stored program."""
        if self.locc_mode:
            return self._locc_run()
        if not self.program:
            self.io.writeln("NOTHING TO RUN")
            return

        t0 = time.time()
        self._gosub_stack = []
        self._collect_data()
        sorted_lines = sorted(self.program.keys())
        self._scan_subs(sorted_lines)
        self._validate_program(sorted_lines)

        # Build circuit
        try:
            qc, has_measure = self.build_circuit()
        except KeyboardInterrupt:
            self.io.writeln("\n?INTERRUPTED")
            return
        except Exception as e:
            self.io.writeln(f"?BUILD ERROR: {e}")
            return

        # Copy before adding measurements (for statevector extraction)
        qc_sv = qc.copy()

        # Unitary/superop methods need save instructions, not measurements
        if self.sim_method in ('unitary', 'superop'):
            if self.sim_method == 'unitary':
                qc.save_unitary(label='unitary')
            else:
                qc.save_superop(label='superop')
        elif has_measure:
            qc.measure_all()

        self.last_circuit = qc

        # Run with shots (cache transpiled circuit if program unchanged)
        try:
            method = self.sim_method
            if method == 'automatic':
                if self.num_qubits > 28:
                    method = 'matrix_product_state'
                elif not self._noise_model and self._is_clifford(qc):
                    method = 'stabilizer'
            backend_opts = {'method': method}
            if self.sim_device == 'GPU':
                backend_opts['device'] = 'GPU'
            if self._noise_model:
                backend_opts['noise_model'] = self._noise_model
            # Performance tuning from memory-mapped config
            if hasattr(self, '_fusion_enable'):
                backend_opts['fusion_enable'] = self._fusion_enable
            if hasattr(self, '_mps_truncation'):
                backend_opts['matrix_product_state_truncation_threshold'] = self._mps_truncation
            if hasattr(self, '_sv_parallel_threshold'):
                backend_opts['statevector_parallel_threshold'] = self._sv_parallel_threshold
            if hasattr(self, '_es_approx_error'):
                backend_opts['extended_stabilizer_approximation_error'] = self._es_approx_error
            cache_key = (
                tuple(sorted(self.program.items())),
                self.num_qubits, method, self.sim_device,
                id(self._noise_model),
                getattr(self, '_fusion_enable', None),
                getattr(self, '_mps_truncation', None),
                getattr(self, '_sv_parallel_threshold', None),
                getattr(self, '_es_approx_error', None),
            )
            if self._circuit_cache_key == cache_key and self._circuit_cache is not None:
                qc_t, backend = self._circuit_cache
            else:
                backend = AerSimulator(**backend_opts)
                qc_t = transpile(qc, backend)
                self._circuit_cache_key = cache_key
                self._circuit_cache = (qc_t, backend)
            try:
                result = backend.run(qc_t, shots=self.shots).result()
            except KeyboardInterrupt:
                self.io.writeln("\n?INTERRUPTED")
                return
            except Exception as _sim_err:
                _err_msg = str(_sim_err).lower()
                if (AerError is not None and isinstance(_sim_err, AerError) and 'stabilizer' in _err_msg) or \
                   ('stabilizer' in _err_msg and 'invalid parameters' in _err_msg):
                    self._circuit_cache_key = None
                    self._circuit_cache = None
                    sv_opts = {k: v for k, v in backend_opts.items()
                               if k != 'method'}
                    sv_opts['method'] = 'statevector'
                    backend = AerSimulator(**sv_opts)
                    qc_t = transpile(qc, backend)
                    result = backend.run(qc_t, shots=self.shots).result()
                else:
                    raise
            # Extract results based on method
            if method in ('unitary', 'superop'):
                self.last_counts = None
                data = result.data()
                label = 'unitary' if method == 'unitary' else 'superop'
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
                        self.io.writeln(f"    (too large to display — stored in variable '{label}')")
            else:
                self.last_counts = dict(result.get_counts())
            # Extract save instruction results into BASIC variables
            data = result.data()
            for key, val in data.items():
                if key.startswith('exp_'):
                    var = key[4:]
                    self.variables[var] = float(np.real(val))
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
        except Exception as e:
            self.io.writeln(f"?RUNTIME ERROR: {e}")
            return

        # Get statevector from the measurement-free copy
        if method not in ('unitary', 'superop'):
            try:
                qc_sv.save_statevector()
                sv_backend = AerSimulator(method='statevector')
                sv_result = sv_backend.run(transpile(qc_sv, sv_backend)).result()
                self.last_sv = np.array(sv_result.get_statevector())
            except Exception:
                self.last_sv = None
        else:
            self.last_sv = None

        dt = time.time() - t0

        # Update status registers
        depth = qc.depth()
        n_gates = qc.size()
        self._update_status(gate_count=n_gates, circuit_depth=depth,
                           run_time_ms=dt * 1000)

        # Display results
        self.io.writeln(f"\nRAN {len(self.program)} lines, {self.num_qubits} qubits, "
                        f"{self.shots} shots in {dt:.2f}s  [depth={depth}, gates={n_gates}]")
        if method in ('unitary', 'superop'):
            pass  # matrix already displayed above
        elif has_measure and self.last_counts:
            self.print_histogram(self.last_counts)
            self._auto_display()
        else:
            self.io.writeln("(no MEASURE in program — use STATE or PROBS to inspect)")

    def cmd_sample(self, rest: str = '') -> None:
        """SAMPLE [shots] — sample the current circuit using SamplerV2 primitive."""
        if not self.program:
            self.io.writeln("NOTHING TO SAMPLE")
            return
        shots = int(rest.strip()) if rest.strip() else self.shots
        try:
            qc, has_measure = self.build_circuit()
            if not has_measure:
                qc.measure_all()
            from qiskit_aer.primitives import SamplerV2
            sampler = SamplerV2()
            result = sampler.run([qc], shots=shots).result()
            # Extract counts — SamplerV2 result format varies by Qiskit version.
            # Try the V2 get_counts() on named data attributes first, then fall
            # back to iterating data attributes for older layouts.
            pub_result = result[0]
            counts = {}
            # V2 preferred path: named classical registers expose get_counts()
            try:
                for attr_name in dir(pub_result.data):
                    if attr_name.startswith('_'):
                        continue
                    obj = getattr(pub_result.data, attr_name, None)
                    if obj is not None and hasattr(obj, 'get_counts'):
                        counts = dict(obj.get_counts())
                        break
            except Exception:
                pass
            # Fallback: try legacy get_counts directly on data
            if not counts:
                try:
                    counts = dict(pub_result.data.get_counts())
                except Exception:
                    pass
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
        """Step through program, showing state after each line."""
        if not self.program:
            self.io.writeln("NOTHING TO STEP")
            return

        from qbasic_core.exec_context import ExecContext
        from qbasic_core.scope import Scope

        sorted_lines = sorted(self.program.keys())
        qc = QuantumCircuit(self.num_qubits)
        ctx = ExecContext(
            sorted_lines=sorted_lines, ip=0,
            run_vars=Scope(self.variables),
            max_iterations=self._max_iterations, qc=qc,
        )

        self.io.writeln(f"STEP MODE — {len(sorted_lines)} lines, {self.num_qubits} qubits")
        self.io.writeln("Press ENTER to advance, Q to quit\n")

        while ctx.ip < len(sorted_lines):
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            line_num = sorted_lines[ctx.ip]
            stmt = self.program[line_num]
            parsed = self._get_parsed(line_num)

            # Display current line
            self.io.writeln(f">> {line_num:5d}  {stmt}")

            # Execute it
            result = self._exec_line(stmt, parsed=parsed, ctx=ctx)

            # Show state
            try:
                qc_tmp = qc.copy()
                qc_tmp.save_statevector()
                sv_b = AerSimulator(method='statevector')
                sv_r = sv_b.run(transpile(qc_tmp, sv_b)).result()
                sv = np.array(sv_r.get_statevector())
                self._print_sv_compact(sv)
            except Exception:
                self.io.writeln("   (state unavailable)")

            # Wait for input
            try:
                user = self.io.read_line("   [ENTER/Q] ").strip().upper()
                if user == 'Q':
                    self.io.writeln("STOPPED")
                    return
            except (KeyboardInterrupt, EOFError):
                self.io.writeln("\nSTOPPED")
                return

            if isinstance(result, int):
                ctx.ip = result
            else:
                ctx.ip += 1

        self.io.writeln("DONE")

    # run_immediate provided by ExecutorMixin.

    def _validate_program(self, sorted_lines: list[int]) -> None:
        """Pre-execution validation. Catches structural errors before running."""
        from qbasic_core.statements import (
            GotoStmt, GosubStmt, ForStmt, NextStmt, WhileStmt, WendStmt,
            DoStmt, LoopStmt, SubStmt, EndSubStmt, FunctionStmt, EndFunctionStmt,
            IfThenStmt,
        )
        line_set = set(sorted_lines)
        for_depth = 0
        while_depth = 0
        do_depth = 0
        for_var_stack: list[str] = []
        for ln in sorted_lines:
            parsed = self._get_parsed(ln)
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
                for_depth -= 1
                # Check FOR/NEXT variable name matching
                if parsed.var and for_var_stack:
                    expected = for_var_stack[-1]
                    if parsed.var.upper() != expected:
                        raise RuntimeError(
                            f"LINE {ln}: NEXT {parsed.var} does not match FOR {expected}")
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
        """Handle MEAS qubit -> var (mid-circuit measurement)."""
        m = RE_MEAS.match(stmt)
        if not m:
            return False
        qubit = int(self._eval_with_vars(m.group(1), run_vars))
        var = m.group(2)
        if 0 <= qubit < self.num_qubits:
            if not self.locc_mode:
                raise QBasicBuildError(
                    "MEAS requires LOCC mode for classical feedforward. "
                    "Use LOCC SEND instead, or use MEASURE for end-of-circuit measurement.")
            b = backend or qc
            if hasattr(b, 'add_classical_register'):
                cr = b.add_classical_register(f'meas_{var}')
                b.measure(qubit, cr[0])
            else:
                from qiskit.circuit import ClassicalRegister
                cr = ClassicalRegister(1, f'meas_{var}')
                qc.add_register(cr)
                qc.measure(qubit, cr[0])
            run_vars[var] = self.MEAS_CIRCUIT_MODE_VALUE
            self.variables[var] = self.MEAS_CIRCUIT_MODE_VALUE
        return True

    def _try_exec_reset(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        """Handle RESET qubit."""
        m = RE_RESET.match(stmt)
        if not m:
            return False
        qubit = int(self._eval_with_vars(m.group(1), run_vars))
        if 0 <= qubit < self.num_qubits:
            b = backend or qc
            b.reset(qubit)
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
                raise ValueError("Matrix is not unitary (U @ U† ≠ I)")
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
        dims = [int(d.strip()) for d in m.group(2).split(',')]
        total = 1
        for d in dims:
            total *= d
        self.arrays[name] = [0.0] * total
        if len(dims) > 1:
            self._array_dims[name] = dims
        return True

    def _try_exec_redim(self, stmt: str) -> bool:
        """Handle REDIM name(size) — resize an existing array."""
        m = RE_REDIM.match(stmt)
        if not m:
            return False
        name = m.group(1)
        new_size = int(m.group(2))
        old = self.arrays.get(name, [])
        if isinstance(old, list):
            if new_size > len(old):
                self.arrays[name] = old + [0.0] * (new_size - len(old))
            else:
                self.arrays[name] = old[:new_size]
        else:
            self.arrays[name] = [0.0] * new_size
        return True

    def _try_exec_erase(self, stmt: str) -> bool:
        """Handle ERASE name — delete a specific array."""
        m = RE_ERASE.match(stmt)
        if not m:
            return False
        name = m.group(1)
        if name in self.arrays:
            del self.arrays[name]
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
            if _sys.stdin.isatty():
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
        qubit = int(self._eval_with_vars(m.group(2), run_vars))
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"QUBIT {qubit} OUT OF RANGE (0-{self.num_qubits-1})")
        b = backend or qc
        if basis == 'X':
            b.apply_gate('H', (), [qubit]) if hasattr(b, 'apply_gate') else qc.h(qubit)
        elif basis == 'Y':
            if hasattr(b, 'apply_gate'):
                b.apply_gate('SDG', (), [qubit])
                b.apply_gate('H', (), [qubit])
            else:
                qc.sdg(qubit)
                qc.h(qubit)
        var = f"m{basis.lower()}_{qubit}"
        if hasattr(b, 'add_classical_register'):
            cr = b.add_classical_register(f'meas_{var}')
            b.measure(qubit, cr[0])
        else:
            from qiskit.circuit import ClassicalRegister
            cr = ClassicalRegister(1, f'meas_{var}')
            qc.add_register(cr)
            qc.measure(qubit, cr[0])
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
        b = backend or qc
        if hasattr(b, 'apply_gate'):
            b.apply_gate('H', (), [anc])
            for p, q in zip(pauli_str, qubits):
                if p != 'I':
                    b.apply_gate(self._PAULI_TO_CONTROLLED[p], (), [anc, q])
            b.apply_gate('H', (), [anc])
            cr = b.add_classical_register(f'synd_{var}')
            b.measure(anc, cr[0])
            b.reset(anc)
        else:
            qc.h(anc)
            for p, q in zip(pauli_str, qubits):
                if p != 'I':
                    gate_method = getattr(qc, self._PAULI_TO_CONTROLLED[p].lower())
                    gate_method(anc, q)
            qc.h(anc)
            from qiskit.circuit import ClassicalRegister
            cr = ClassicalRegister(1, f'synd_{var}')
            qc.add_register(cr)
            qc.measure(anc, cr[0])
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
            or self._try_exec_unitary(stmt)
            or self._try_exec_dim(stmt)
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
        )

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
        val = self._eval_string_expr(m.group(2))
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
            rest = stmt[4:].strip()
            self.cmd_open(rest)
            return True
        m = RE_CLOSE.match(stmt)
        if m:
            self.cmd_close(m.group(1))
            return True
        return False

    def _try_exec_save_expect(self, stmt: str, qc, run_vars: dict) -> bool:
        """SAVE_EXPECT <pauli> <qubits> -> <var> — inline expectation value."""
        from qbasic_core.engine import RE_SAVE_EXPECT
        m = RE_SAVE_EXPECT.match(stmt)
        if not m:
            return False
        pauli_str = m.group(1).upper()
        qubits = [int(q.strip()) for q in m.group(2).split(',') if q.strip()]
        var = m.group(3)
        try:
            from qiskit.quantum_info import SparsePauliOp
            full_pauli = ['I'] * self.num_qubits
            for i, p in enumerate(pauli_str):
                if i < len(qubits):
                    full_pauli[self.num_qubits - 1 - qubits[i]] = p
            op = SparsePauliOp(''.join(full_pauli))
            qc.save_expectation_value(op, list(range(self.num_qubits)), label=f'exp_{var}')
            run_vars[var] = 0  # actual value available after simulation
            self.variables[var] = 0
        except Exception as e:
            self.io.writeln(f"?SAVE_EXPECT ERROR: {e}")
        return True

    def _try_exec_save_probs(self, stmt: str, qc, run_vars: dict) -> bool:
        """SAVE_PROBS <qubits> -> <var> — inline probability snapshot."""
        from qbasic_core.engine import RE_SAVE_PROBS
        m = RE_SAVE_PROBS.match(stmt)
        if not m:
            return False
        qubits = [int(q.strip()) for q in m.group(1).split(',') if q.strip()]
        var = m.group(2)
        try:
            qc.save_probabilities(qubits, label=f'prob_{var}')
            run_vars[var] = 0
            self.variables[var] = 0
        except Exception as e:
            self.io.writeln(f"?SAVE_PROBS ERROR: {e}")
        return True

    def _try_exec_save_amps(self, stmt: str, qc, run_vars: dict) -> bool:
        """SAVE_AMPS <indices> -> <var> — save specific amplitudes by index."""
        from qbasic_core.engine import RE_SAVE_AMPS
        m = RE_SAVE_AMPS.match(stmt)
        if not m:
            return False
        indices = [int(q.strip()) for q in m.group(1).split(',') if q.strip()]
        var = m.group(2)
        try:
            qc.save_amplitudes(indices, label=f'amp_{var}')
            run_vars[var] = 0
            self.variables[var] = 0
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
        from qbasic_core.engine import RE_SET_STATE
        m = RE_SET_STATE.match(stmt)
        if not m:
            return False
        try:
            sv_expr = m.group(1).strip()
            dim = 2 ** self.num_qubits
            # Try named states first
            if sv_expr.upper() in ('|0>', '|1>', '|+>', '|->', '|BELL>', '|GHZ>',
                                  '|GHZ3>', '|GHZ4>', '|W>', '|W3>'):
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

    # _split_colon_stmts, _substitute_vars, _expand_statement,
    # _offset_qubits, _apply_gate_str, _tokenize_gate, _resolve_qubit,
    # _GATE_DISPATCH, _apply_gate provided by ExecutorMixin.

    # ── Display ───────────────────────────────────────────────────────

    def _try_quantum_print(self, text: str, run_vars: dict) -> str | None:
        """Handle quantum PRINT expressions. Returns formatted string or None."""
        upper = text.strip().upper()
        # PRINT @REG — Dirac notation for LOCC register
        if self.locc_mode and self.locc and upper.startswith('@'):
            reg = upper[1:].strip()
            if reg in self.locc.names:
                sv = self.locc.get_sv(reg)
                n = self.locc.get_n(reg)
                return self._format_dirac(sv, n)
        # PRINT QUBIT(n) — single-qubit Bloch info
        m = re.match(r'QUBIT\s*\((\d+)\)', text, re.IGNORECASE)
        if m and self.last_sv is not None:
            q = int(m.group(1))
            x, y, z = self._bloch_vector(self.last_sv, q)
            p1 = self._peek(0x0100 + q * 8)
            return f"q{q}: P(1)={p1:.4f} Bloch=({x:.3f},{y:.3f},{z:.3f})"
        # PRINT ENTANGLEMENT(a,b) — entropy between qubits
        m = re.match(r'ENTANGLEMENT\s*\((\d+)\s*,\s*(\d+)\)', text, re.IGNORECASE)
        if m and self.last_sv is not None:
            qa, qb = int(m.group(1)), int(m.group(2))
            try:
                from qiskit.quantum_info import Statevector, entropy, partial_trace
                sv_obj = Statevector(np.ascontiguousarray(self.last_sv).ravel())
                keep = [qa]
                trace_out = [q for q in range(self.num_qubits) if q not in keep]
                rho = partial_trace(sv_obj, trace_out)
                ent = entropy(rho, base=2)
                return f"S({qa},{qb}) = {ent:.6f} bits"
            except Exception:
                return f"S({qa},{qb}) = ?"
        # PRINT STATE — full statevector in Dirac notation
        if upper == 'STATE' and self.last_sv is not None:
            return self._format_dirac(self.last_sv, self.num_qubits)
        return None

    def _format_dirac(self, sv, n_qubits: int) -> str:
        """Format a statevector in Dirac notation."""
        from qbasic_core.engine import AMPLITUDE_THRESHOLD
        sv_flat = np.ascontiguousarray(sv).ravel()
        parts = []
        for i, amp in enumerate(sv_flat):
            if abs(amp) > AMPLITUDE_THRESHOLD:
                state = format(i, f'0{n_qubits}b')
                if abs(amp.imag) < AMPLITUDE_THRESHOLD:
                    if abs(amp.real - 1.0) < 1e-6:
                        parts.append(f"|{state}\u27E9")
                    elif abs(amp.real + 1.0) < 1e-6:
                        parts.append(f"-|{state}\u27E9")
                    else:
                        parts.append(f"{amp.real:+.4f}|{state}\u27E9")
                else:
                    parts.append(f"({amp.real:.3f}{amp.imag:+.3f}j)|{state}\u27E9")
                if len(parts) >= 16:
                    parts.append("...")
                    break
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
        ops = {}
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
        if not self._undo_stack:
            self.io.writeln("NOTHING TO UNDO")
            return
        self.program = self._undo_stack.pop()
        self._parsed = {num: parse_stmt(s) for num, s in self.program.items()}
        self.io.writeln(f"UNDO ({len(self.program)} lines)")

    # LOCC execution (_locc_run, _locc_exec_line, etc.) provided by LOCCMixin.

    # Control flow helpers (_cf_*, _exec_control_flow, _find_matching_wend)
    # provided by ControlFlowMixin.

    # ── Help ──────────────────────────────────────────────────────────

    def cmd_help(self) -> None:
        self.io.writeln(HELP_TEXT)
        # Auto-generated command reference from registry
        all_cmds = sorted(set(self._CMD_NO_ARG.keys()) | set(self._CMD_WITH_ARG.keys()))
        all_gates = sorted(g for g in GATE_TABLE if g not in GATE_ALIASES)
        self.io.writeln(f"        ALL COMMANDS: {', '.join(all_cmds)}")
        self.io.writeln(f"        ALL GATES: {', '.join(all_gates)}")

    # ── Banner ────────────────────────────────────────────────────────

    def print_banner(self) -> None:
        import platform
        try:
            import qiskit
            qver = qiskit.__version__
        except Exception:
            qver = '?'
        ram = _get_ram_gb()
        ram_str = f"{ram[0]:.0f} GB RAM" if ram else "RAM unknown"
        max_q = 32
        if ram:
            from qbasic_core.engine import _estimate_gb
            for n in range(32, 0, -1):
                if _estimate_gb(n) < ram[1]:
                    max_q = n
                    break
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
        config_line = f"{self.num_qubits} qubits | {self.shots} shots | max ~{max_q} qubits"
        self.io.writeln(BANNER_ART.format(info_line=info_line, config_line=config_line))

