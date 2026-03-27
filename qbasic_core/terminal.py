"""QBASIC terminal — REPL, commands, circuit building, LOCC execution."""

import sys
import os
import re
import time
import textwrap
from collections import OrderedDict

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

from qbasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    ExecResult,
    _np_gate_matrix, _get_ram_gb, _estimate_gb,
    MAX_QUBITS, DEFAULT_QUBITS, DEFAULT_SHOTS, MAX_UNDO_STACK,
    MAX_LOOP_ITERATIONS, MAX_BLOCH_DISPLAY,
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
from qbasic_core.errors import QBasicError
from qbasic_core.io_protocol import StdIOPort
from qbasic_core.parser import parse_stmt
from qbasic_core.engine_state import Engine


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
    else:
        sv[0] = 1.0
    return sv


# ═══════════════════════════════════════════════════════════════════════
# The Terminal
# ═══════════════════════════════════════════════════════════════════════

class QBasicTerminal(Engine, ExpressionMixin, DisplayMixin, DemoMixin, LOCCMixin,
                     ControlFlowMixin, FileIOMixin, AnalysisMixin, SweepMixin,
                     MemoryMixin, StringMixin, ScreenMixin, ClassicMixin,
                     SubroutineMixin, DebugMixin, ProgramMgmtMixin, ProfilerMixin):
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
            pass

    def repl(self) -> None:
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
        """
        from qbasic_core.engine import RE_TYPE_BEGIN, RE_TYPE_FIELD, RE_END_TYPE
        m = RE_TYPE_BEGIN.match(f"TYPE {rest}")
        if not m:
            self.io.writeln("?USAGE: TYPE <name>")
            return
        type_name = m.group(1).upper()
        fields = []
        self.io.writeln(f"  TYPE {type_name} (enter fields, END TYPE to finish)")
        while True:
            try:
                line = self.io.read_line('  . ').strip()
            except (KeyboardInterrupt, EOFError):
                self.io.writeln("\n  CANCELLED")
                return
            if RE_END_TYPE.match(line):
                break
            fm = RE_TYPE_FIELD.match(line)
            if fm:
                fields.append((fm.group(1).lower(), fm.group(2).upper()))
            elif line:
                self.io.writeln(f"  ?EXPECTED: <name> AS <type>")
        if not hasattr(self, '_user_types'):
            self._user_types = {}
        self._user_types[type_name] = fields
        self.io.writeln(f"TYPE {type_name} ({len(fields)} fields)")

    def cmd_circuit_def(self, rest: str) -> None:
        """CIRCUIT_DEF name start-end — define a circuit macro from line range."""
        import re as _re
        m = _re.match(r'(\w+)\s+(\d+)\s*-\s*(\d+)', rest)
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
        import re as _re
        m = _re.match(r'(\w+)(?:\s+@(\d+))?', rest)
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

    def cmd_defs(self) -> None:
        """List all defined subroutines."""
        if not self.subroutines:
            self.io.writeln("NO SUBROUTINES DEFINED")
            return
        for name, sub in self.subroutines.items():
            if isinstance(sub, list):
                self.io.writeln(f"  {name} = {' : '.join(sub)}")
            else:
                params = f"({', '.join(sub['params'])})" if sub['params'] else ""
                self.io.writeln(f"  {name}{params} = {' : '.join(sub['body'])}")

    def cmd_regs(self) -> None:
        """List all named registers with their qubit ranges."""
        if not self.registers:
            self.io.writeln("NO REGISTERS DEFINED")
            return
        for name, (start, size) in self.registers.items():
            self.io.writeln(f"  {name}[0:{size}] -> qubits {start}-{start+size-1}")

    def cmd_vars(self) -> None:
        """List all variables and their current values."""
        if not self.variables:
            self.io.writeln("NO VARIABLES SET")
            return
        for name, val in self.variables.items():
            self.io.writeln(f"  {name} = {val}")

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
            result = backend.run(qc_t, shots=self.shots).result()
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
            # Extract counts — try multiple access paths for Qiskit version compat
            pub_result = result[0]
            counts = {}
            try:
                # V2 path: data has named classical registers
                for attr_name in dir(pub_result.data):
                    if attr_name.startswith('_'):
                        continue
                    obj = getattr(pub_result.data, attr_name, None)
                    if obj is not None and hasattr(obj, 'get_counts'):
                        counts = dict(obj.get_counts())
                        break
            except Exception:
                pass
            if not counts:
                # Fallback: try legacy get_counts
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
            qc, _ = self.build_circuit()
            from qiskit.quantum_info import SparsePauliOp
            full_pauli = ['I'] * self.num_qubits
            for i, p in enumerate(pauli_str):
                if i < len(qubits):
                    full_pauli[self.num_qubits - 1 - qubits[i]] = p
            op = SparsePauliOp(''.join(full_pauli))
            from qiskit_aer.primitives import EstimatorV2
            estimator = EstimatorV2()
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
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

    def run_immediate(self, line: str) -> None:
        """Execute a single gate command immediately.

        Uses the same _exec_line pipeline as cmd_run for consistency.
        """
        # In LOCC mode, handle @register prefix via the numpy engine
        if self.locc_mode and self.locc:
            m = RE_AT_REG_LINE.match(line)
            if m:
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
        # Build and execute through the same gate pipeline as cmd_run
        from qbasic_core.exec_context import ExecContext
        qc = QuantumCircuit(self.num_qubits)
        imm_ctx = ExecContext(sorted_lines=[0], ip=0,
                              run_vars=dict(self.variables), qc=qc)
        self._exec_line(line, ctx=imm_ctx)
        qc.save_statevector()
        backend = AerSimulator(method='statevector')
        result = backend.run(transpile(qc, backend)).result()
        sv = np.array(result.get_statevector())
        self.last_sv = sv
        self.last_circuit = qc
        self._print_sv_compact(sv)

    def _validate_program(self, sorted_lines: list[int]) -> None:
        """Pre-execution validation. Catches structural errors before running."""
        from qbasic_core.statements import (
            GotoStmt, GosubStmt, ForStmt, NextStmt, WhileStmt, WendStmt,
            DoStmt, LoopStmt, SubStmt, EndSubStmt, FunctionStmt, EndFunctionStmt,
        )
        line_set = set(sorted_lines)
        for_depth = 0
        while_depth = 0
        do_depth = 0
        for ln in sorted_lines:
            parsed = self._get_parsed(ln)
            if isinstance(parsed, GotoStmt) and parsed.target not in line_set:
                raise RuntimeError(f"LINE {ln}: GOTO {parsed.target} — target line not found")
            if isinstance(parsed, GosubStmt) and parsed.target not in line_set:
                raise RuntimeError(f"LINE {ln}: GOSUB {parsed.target} — target line not found")
            if isinstance(parsed, ForStmt):
                for_depth += 1
            elif isinstance(parsed, NextStmt):
                for_depth -= 1
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

    # ── Circuit Building ──────────────────────────────────────────────

    def build_circuit(self) -> tuple['QuantumCircuit', bool]:
        """Compile program lines into a QuantumCircuit. Returns (circuit, has_measure)."""
        from qbasic_core.exec_context import ExecContext
        from qbasic_core.statements import MeasureStmt
        from qbasic_core.scope import Scope
        from qbasic_core.backend import QiskitBackend

        qc = QuantumCircuit(self.num_qubits)
        backend = QiskitBackend(qc, self._apply_gate)
        ctx = ExecContext(
            sorted_lines=sorted(self.program.keys()),
            ip=0,
            run_vars=Scope(self.variables),
            max_iterations=self._max_iterations,
            qc=qc,
            backend=backend,
        )
        has_measure = False

        while ctx.ip < len(ctx.sorted_lines):
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            line_num = ctx.sorted_lines[ctx.ip]
            stmt = self.program[line_num].strip()
            parsed = self._get_parsed(line_num)

            if isinstance(parsed, MeasureStmt):
                has_measure = True
                ctx.ip += 1
                continue

            try:
                result = self._exec_line(stmt, parsed=parsed, ctx=ctx)
            except Exception as e:
                raise RuntimeError(f"LINE {line_num}: {e}") from None

            if result is ExecResult.END:
                break
            elif isinstance(result, int):
                ctx.ip = result
            else:
                ctx.ip += 1

        return qc, has_measure

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
            self.io.writeln(
                f"  ?MEAS {var}: deferred measurement (value={self.MEAS_CIRCUIT_MODE_VALUE} "
                f"during circuit build). Use LOCC SEND for classical feedforward.")
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
            if not hasattr(self, '_array_dims'):
                self._array_dims = {}
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

    def _parse_syndrome(self, stmt: str, run_vars: dict) -> tuple[str, list[int], str] | None:
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

    def _exec_line(self, stmt, qc=None, loop_stack=None, sorted_lines=None,
                   ip=0, run_vars=None, parsed=None, *, ctx=None):
        """Execute one program line.

        Accepts either individual parameters (legacy) or ctx (ExecContext).
        When ctx is provided, qc/loop_stack/sorted_lines/ip/run_vars are
        read from ctx and the individual params are ignored.

        Evaluation order (deterministic, first match wins):
          1. Typed fast-path (parsed Stmt): BARRIER, REM, MEASURE, END, @REG, compound
          2. Control flow: LET, PRINT, GOTO, GOSUB, FOR/NEXT, WHILE/WEND, IF/THEN,
             DATA/READ, ON GOTO/GOSUB, SELECT CASE, DO/LOOP, EXIT, SUB/FUNCTION,
             ON ERROR, ASSERT, STOP, SWAP, DEF FN, OPTION BASE
          3. Statement handlers: MEAS, RESET, MEASURE_X/Y/Z, SYNDROME, UNITARY,
             DIM, REDIM, ERASE, GET, INPUT, POKE, SYS, file I/O, PRINT USING
          4. Colon-separated compound statements
          5. Gate application (subroutine expansion + gate dispatch)

        Returns: int (jump target ip), ExecResult.ADVANCE, or ExecResult.END.
        """
        if ctx is not None:
            qc = ctx.backend.qc if ctx.backend and hasattr(ctx.backend, 'qc') else ctx.qc
            loop_stack = ctx.loop_stack
            sorted_lines = ctx.sorted_lines
            ip = ctx.ip
            run_vars = ctx.run_vars
        from qbasic_core.statements import (
            BarrierStmt, RemStmt, MeasureStmt, EndStmt, ReturnStmt,
            CompoundStmt, AtRegStmt, GotoStmt, GosubStmt,
            ForStmt, NextStmt, WhileStmt, WendStmt, IfThenStmt,
            LetStmt, LetArrayStmt, PrintStmt,
            RawStmt,
        )

        # 1. Typed fast-path (no regex, no string manipulation)
        if parsed is None:
            from qbasic_core.parser import parse_stmt
            parsed = parse_stmt(stmt)
        if isinstance(parsed, BarrierStmt):
            if hasattr(qc, 'barrier'):
                qc.barrier()
            return ExecResult.ADVANCE
        if isinstance(parsed, (RemStmt, MeasureStmt)):
            return ExecResult.ADVANCE
        if isinstance(parsed, EndStmt):
            return ExecResult.END
        if isinstance(parsed, ReturnStmt):
            if not self._gosub_stack:
                raise RuntimeError("RETURN WITHOUT GOSUB")
            return self._gosub_stack.pop()
        if isinstance(parsed, GotoStmt):
            for idx, ln in enumerate(sorted_lines):
                if ln == parsed.target:
                    return idx
            raise RuntimeError(f"GOTO {parsed.target}: LINE NOT FOUND")
        if isinstance(parsed, GosubStmt):
            self._gosub_stack.append(ip + 1)
            for idx, ln in enumerate(sorted_lines):
                if ln == parsed.target:
                    return idx
            raise RuntimeError(f"GOSUB {parsed.target}: LINE NOT FOUND")
        if isinstance(parsed, WendStmt):
            _, r = self._cf_wend(run_vars, loop_stack, sorted_lines, ip)
            return r
        if isinstance(parsed, ForStmt):
            start = self._eval_with_vars(parsed.start_expr, run_vars)
            end = self._eval_with_vars(parsed.end_expr, run_vars)
            step = self._eval_with_vars(parsed.step_expr, run_vars) if parsed.step_expr else 1
            try:
                if start == int(start): start = int(start)
            except (OverflowError, ValueError): pass
            try:
                if end == int(end): end = int(end)
            except (OverflowError, ValueError): pass
            try:
                if isinstance(step, float) and step == int(step): step = int(step)
            except (OverflowError, ValueError): pass
            run_vars[parsed.var] = start
            self.variables[parsed.var] = start
            loop_stack.append({'var': parsed.var, 'current': start, 'end': end,
                               'step': step, 'return_ip': ip})
            return ExecResult.ADVANCE
        if isinstance(parsed, NextStmt):
            if not loop_stack or loop_stack[-1].get('var') != parsed.var:
                if loop_stack:
                    raise RuntimeError(f"NEXT {parsed.var} does not match current FOR {loop_stack[-1].get('var', '?')}")
                raise RuntimeError(f"NEXT {parsed.var} without matching FOR")
            loop = loop_stack[-1]
            loop['current'] += loop['step']
            if (loop['step'] > 0 and loop['current'] <= loop['end']) or \
               (loop['step'] < 0 and loop['current'] >= loop['end']):
                run_vars[parsed.var] = loop['current']
                self.variables[parsed.var] = loop['current']
                return loop['return_ip'] + 1
            else:
                loop_stack.pop()
                return ExecResult.ADVANCE
        if isinstance(parsed, WhileStmt):
            if self._eval_condition(parsed.condition, run_vars):
                loop_stack.append({'type': 'while', 'cond': parsed.condition, 'return_ip': ip})
                return ExecResult.ADVANCE
            else:
                return self._find_matching_wend(sorted_lines, ip)
        if isinstance(parsed, LetStmt):
            val = self._eval_with_vars(parsed.expr, run_vars)
            run_vars[parsed.name] = val
            self.variables[parsed.name] = val
            return ExecResult.ADVANCE
        if isinstance(parsed, LetArrayStmt):
            idx = int(self._eval_with_vars(parsed.index_expr, run_vars))
            val = self._eval_with_vars(parsed.value_expr, run_vars)
            if parsed.name not in self.arrays:
                self.arrays[parsed.name] = [0.0] * (idx + 1)
            while idx >= len(self.arrays[parsed.name]):
                self.arrays[parsed.name].append(0.0)
            self.arrays[parsed.name][idx] = val
            return ExecResult.ADVANCE
        if isinstance(parsed, PrintStmt):
            import re as _re
            text = parsed.expr
            suppress_nl = text.rstrip().endswith(';')
            tab_advance = text.rstrip().endswith(',')
            if suppress_nl:
                text = text.rstrip().removesuffix(';').rstrip()
            elif tab_advance:
                text = text.rstrip().removesuffix(',').rstrip()
            # Quantum PRINT: @REG, QUBIT(n), ENTANGLEMENT(a,b)
            qprint = self._try_quantum_print(text, run_vars)
            if qprint is not None:
                if suppress_nl:
                    self.io.write(qprint)
                elif tab_advance:
                    col = len(qprint) % 14
                    self.io.write(qprint + ' ' * (14 - col if col > 0 else 14))
                else:
                    self.io.writeln(qprint)
                return ExecResult.ADVANCE
            # SPC/TAB inline
            def _spc(m_s):
                return ' ' * max(0, int(self._eval_with_vars(m_s.group(1), run_vars)))
            def _tab(m_t):
                return ' ' * max(0, int(self._eval_with_vars(m_t.group(1), run_vars)))
            text = _re.sub(r'\bSPC\s*\(([^)]+)\)', _spc, text, flags=_re.IGNORECASE)
            text = _re.sub(r'\bTAB\s*\(([^)]+)\)', _tab, text, flags=_re.IGNORECASE)
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                output = text[1:-1]
            else:
                try:
                    output = str(self._eval_with_vars(text, run_vars))
                except Exception:
                    output = text
            if suppress_nl:
                self.io.write(output)
            elif tab_advance:
                col = len(output) % 14
                self.io.write(output + ' ' * (14 - col if col > 0 else 14))
            else:
                self.io.writeln(output)
            return ExecResult.ADVANCE
        if isinstance(parsed, IfThenStmt):
            cond_vars = run_vars
            if self.locc_mode and self.locc:
                cond_vars = {**({} if not hasattr(run_vars, 'as_dict') else run_vars.as_dict()),
                             **self.locc.classical}
                if hasattr(run_vars, 'as_dict'):
                    cond_vars.update(run_vars.as_dict())
                else:
                    cond_vars.update(run_vars)
            result = ExecResult.ADVANCE
            if self._eval_condition(parsed.condition, cond_vars):
                if parsed.then_clause:
                    r = self._exec_line(parsed.then_clause, qc=qc, loop_stack=loop_stack,
                                        sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)
                    if r is not None and r is not ExecResult.ADVANCE:
                        result = r
            elif parsed.else_clause:
                r = self._exec_line(parsed.else_clause, qc=qc, loop_stack=loop_stack,
                                    sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)
                if r is not None and r is not ExecResult.ADVANCE:
                    result = r
            return result
        if isinstance(parsed, AtRegStmt) and not self.locc_mode:
            raise ValueError("@register syntax requires LOCC mode (try: LOCC <n1> <n2>)")
        if isinstance(parsed, CompoundStmt):
            for sub in parsed.parts:
                self._exec_line(sub, qc=qc, loop_stack=loop_stack,
                                sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)
            return ExecResult.ADVANCE

        # 2. Extended typed dispatch — delegates to _cf_* with raw string
        from qbasic_core.statements import (
            DataStmt, ReadStmt, OnGotoStmt, OnGosubStmt,
            SelectCaseStmt, CaseStmt, EndSelectStmt,
            DoStmt, LoopStmt, ExitStmt,
            SwapStmt, DefFnStmt, OptionBaseStmt, RestoreStmt,
            SubStmt, EndSubStmt, FunctionStmt, EndFunctionStmt,
            CallStmt, LocalStmt, StaticStmt, SharedStmt,
            OnErrorStmt, ResumeStmt, ErrorStmt, AssertStmt,
            StopStmt, OnMeasureStmt, OnTimerStmt,
        )
        _cf_map = {
            DataStmt: lambda: self._cf_data(stmt),
            ReadStmt: lambda: self._cf_read(stmt, run_vars),
            OnGotoStmt: lambda: self._cf_on_goto(stmt, run_vars, sorted_lines),
            OnGosubStmt: lambda: self._cf_on_gosub(stmt, run_vars, sorted_lines, ip),
            SelectCaseStmt: lambda: self._cf_select_case(stmt, run_vars, sorted_lines, ip),
            CaseStmt: lambda: self._cf_case(stmt, sorted_lines, ip),
            EndSelectStmt: lambda: self._cf_end_select(stmt),
            DoStmt: lambda: self._cf_do(stmt, run_vars, loop_stack, sorted_lines, ip),
            LoopStmt: lambda: self._cf_loop(stmt, run_vars, loop_stack, sorted_lines, ip),
            ExitStmt: lambda: self._cf_exit(stmt, loop_stack, sorted_lines, ip),
            SwapStmt: lambda: self._cf_swap(stmt, run_vars),
            DefFnStmt: lambda: self._cf_def_fn(stmt, run_vars),
            OptionBaseStmt: lambda: self._cf_option_base(stmt),
            RestoreStmt: lambda: (True, ExecResult.ADVANCE),
            SubStmt: lambda: self._cf_sub(stmt, sorted_lines, ip),
            EndSubStmt: lambda: self._cf_end_sub(stmt),
            FunctionStmt: lambda: self._cf_function(stmt, sorted_lines, ip),
            EndFunctionStmt: lambda: self._cf_end_function(stmt),
            CallStmt: lambda: self._cf_call(stmt, run_vars, sorted_lines, ip),
            LocalStmt: lambda: self._cf_local(stmt, run_vars),
            StaticStmt: lambda: self._cf_static(stmt, run_vars),
            SharedStmt: lambda: self._cf_shared(stmt, run_vars),
            OnErrorStmt: lambda: self._cf_on_error(stmt),
            ResumeStmt: lambda: self._cf_resume(stmt, sorted_lines),
            ErrorStmt: lambda: self._cf_error(stmt),
            AssertStmt: lambda: self._cf_assert(stmt, run_vars),
            StopStmt: lambda: self._cf_stop(stmt, sorted_lines, ip),
            OnMeasureStmt: lambda: self._cf_on_measure(stmt),
            OnTimerStmt: lambda: self._cf_on_timer(stmt),
        }
        handler = _cf_map.get(type(parsed))
        if handler is not None:
            r = handler()
            if r is not None:
                return r[1]

        # 3. Statement handlers
        _backend = ctx.backend if ctx else None
        if self._try_stmt_handlers(stmt, qc, run_vars, backend=_backend):
            return ExecResult.ADVANCE

        # 4. Colon-separated (legacy fallback for unparsed compound)
        if ':' in stmt:
            for sub in self._split_colon_stmts(stmt):
                self._exec_line(sub, qc=qc, loop_stack=loop_stack,
                                sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)
            return ExecResult.ADVANCE

        # 5. Gate application (through backend when available)
        _backend = ctx.backend if ctx else None
        expanded = self._expand_statement(stmt)
        for gate_str in expanded:
            self._apply_gate_str(gate_str, qc, backend=_backend)

        return ExecResult.ADVANCE

    # Names reserved from variable substitution — derived from source tables
    # so they stay in sync automatically.
    _RESERVED_KEYWORDS = frozenset({
        'REM', 'MEASURE', 'BARRIER', 'END', 'RETURN',
        'FOR', 'NEXT', 'WHILE', 'WEND', 'IF', 'THEN', 'ELSE',
        'GOTO', 'GOSUB', 'LET', 'PRINT', 'INPUT', 'DIM',
        'AND', 'OR', 'NOT', 'TO', 'STEP', 'SEND', 'SHARE',
        'MEAS', 'RESET', 'UNITARY', 'CTRL', 'INV',
    })
    _RESERVED_NAMES = (
        set(GATE_TABLE.keys()) | set(GATE_ALIASES.keys()) |
        set(ExpressionMixin._SAFE_CONSTS.keys()) |
        set(ExpressionMixin._SAFE_FUNCS.keys()) |
        _RESERVED_KEYWORDS
    )

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

    # Named quantum states for SET_STATE Dirac notation
    _NAMED_STATES = {
        '|0>': lambda n: _named_sv(n, 0),
        '|1>': lambda n: _named_sv(n, 1),
        '|+>': lambda n: _named_sv_plus(n),
        '|->': lambda n: _named_sv_minus(n),
        '|BELL>': lambda n: _named_sv_bell(n),
        '|GHZ>': lambda n: _named_sv_ghz(n),
    }

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
            if sv_expr.upper() in ('|0>', '|1>', '|+>', '|->', '|BELL>', '|GHZ>'):
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

    @staticmethod
    def _split_colon_stmts(stmt: str) -> list[str]:
        """Split colon-separated statements, inheriting @register prefixes."""
        from qbasic_core.parser import _split_colon_stmts
        return _split_colon_stmts(stmt)

    def _substitute_vars(self, stmt: str, run_vars: dict) -> str:
        """Replace variable names with their values in a statement.

        Tokenizes the statement and replaces eligible identifiers in-place,
        avoiding substitution inside quoted strings, register notation, and
        protected names (gates, keywords, constants, subroutines, custom gates).
        """
        merged = {**self.variables, **run_vars}
        if not merged:
            return stmt
        # Build the set of names that should never be substituted
        protected = (
            self._RESERVED_NAMES |
            {name.lower() for name in self.registers} |
            {name.upper() for name in self.registers} |
            set(self.subroutines.keys()) |
            set(self._custom_gates.keys())
        )
        # Tokenize: split on word boundaries, preserving delimiters
        tokens = re.split(r'(\b\w+\b)', stmt)
        for i, tok in enumerate(tokens):
            if not tok or not tok[0].isalpha():
                continue
            if tok in protected or tok.upper() in protected or tok.lower() in protected:
                continue
            if tok in merged:
                tokens[i] = str(merged[tok])
        return ''.join(tokens)

    def _expand_statement(self, stmt, _call_stack: set[str] | None = None):
        """Expand subroutines. Returns list of gate strings.

        Uses explicit call-stack tracking to detect recursion instead of
        an arbitrary depth counter.
        """
        if _call_stack is None:
            _call_stack = set()
        parts = stmt.split()
        word = parts[0].upper() if parts else ''

        if word not in self.subroutines:
            return [stmt]

        if word in _call_stack:
            raise RuntimeError(f"RECURSIVE SUBROUTINE: {word} calls itself")
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
            def _sub(m):
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
                # Skip register[index] notation — it resolves its own offset
                m = RE_REG_INDEX.match(tok)
                if m:
                    result.append(tok)
                else:
                    try:
                        result.append(str(int(tok) + offset))
                    except ValueError:
                        result.append(tok)
        return ' '.join(result)

    def _apply_gate_str(self, stmt, qc, _call_stack=None, *, backend=None):
        """Parse and apply a single gate string to the circuit.

        When backend is provided, standard gates are dispatched through
        backend.apply_gate() instead of self._apply_gate(qc, ...).
        """
        stmt = stmt.strip()
        if not stmt:
            return

        # Expand subroutines with call-stack tracking for recursion detection
        word = stmt.split()[0].upper() if stmt.split() else ''
        if word in self.subroutines:
            for sub_stmt in self._expand_statement(stmt, _call_stack):
                self._apply_gate_str(sub_stmt, qc, _call_stack, backend=backend)
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

        # CTRL gate ctrl_qubit, target_qubit(s) — controlled version of any gate
        m_ctrl = RE_CTRL.match(stmt)
        if m_ctrl:
            from qiskit.circuit.library import (HGate, XGate, YGate, ZGate,
                SGate, TGate, SdgGate, TdgGate, SXGate, SwapGate)
            gate_name = m_ctrl.group(1).upper()
            args = [a.strip() for a in m_ctrl.group(2).replace(',', ' ').split()]
            ctrl_qubit = self._resolve_qubit(args[0])
            target_qubits = [self._resolve_qubit(a) for a in args[1:]]
            gate_map = {
                'H': HGate(), 'X': XGate(), 'Y': YGate(), 'Z': ZGate(),
                'S': SGate(), 'T': TGate(), 'SDG': SdgGate(), 'TDG': TdgGate(),
                'SX': SXGate(), 'SWAP': SwapGate(),
            }
            all_qubits = [ctrl_qubit] + target_qubits
            if gate_name in gate_map:
                gate = gate_map[gate_name].control(1)
                if backend and hasattr(backend, 'append_controlled'):
                    backend.append_controlled(gate, all_qubits)
                else:
                    qc.append(gate, all_qubits)
            elif gate_name in self._custom_gates:
                from qiskit.circuit.library import UnitaryGate
                gate = UnitaryGate(self._custom_gates[gate_name]).control(1)
                if backend and hasattr(backend, 'append_controlled'):
                    backend.append_controlled(gate, all_qubits)
                else:
                    qc.append(gate, all_qubits)
            else:
                raise ValueError(f"CTRL {gate_name}: gate not found")
            return

        # INV gate qubit(s) — inverse/dagger of a single gate
        m_inv = RE_INV.match(stmt)
        if m_inv:
            gate_name = m_inv.group(1).upper()
            inv_args = m_inv.group(2)
            tokens = self._tokenize_gate(f"{gate_name} {inv_args}")
            gate_key = GATE_ALIASES.get(gate_name, gate_name)
            info = self._gate_info(gate_key)
            if info is not None:
                n_params, n_qubits_needed = info
                t_args = tokens[1:]
                params = [self.eval_expr(a) for a in t_args[:n_params]]
                qubits_inv = [self._resolve_qubit(a) for a in t_args[n_params:n_params+n_qubits_needed]]
                sub_qc = QuantumCircuit(max(qubits_inv) + 1)
                self._apply_gate(sub_qc, gate_key, params, list(range(len(qubits_inv))))
                if backend and hasattr(backend, 'append_inverse'):
                    backend.append_inverse(sub_qc, qubits_inv)
                else:
                    qc.append(sub_qc.inverse(), qubits_inv)
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

        params = [self.eval_expr(a) for a in args[:n_params]]
        qubits = [self._resolve_qubit(a) for a in args[n_params:n_params+n_qubits]]

        # Apply gate through backend when available
        if backend:
            backend.apply_gate(gate_name, tuple(params), qubits)
        else:
            self._apply_gate(qc, gate_name, params, qubits)

    def _tokenize_gate(self, stmt: str) -> list[str]:
        """Split gate statement into tokens, handling commas and register notation.

        Splits on whitespace and commas. Preserves register[index] notation.
        """
        stmt = RE_REG_INDEX.sub(r'\1[\2]', stmt)
        return [t.strip() for t in re.split(r'[,\s]+', stmt) if t.strip()]

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
                raise ValueError(f"UNKNOWN REGISTER: {name}")
            start, size = self.registers[name]
            if idx >= size:
                raise ValueError(f"{name}[{idx}] OUT OF RANGE (size={size})")
            q = start + idx
        else:
            try:
                q = int(self.eval_expr(arg))
            except Exception:
                raise ValueError(f"CANNOT RESOLVE QUBIT: {arg}")
        if q < 0 or q >= limit:
            raise ValueError(f"QUBIT {q} OUT OF RANGE (0-{limit-1})")
        return q

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

    def _apply_gate(self, qc, gate_name, params, qubits):
        """Apply a gate to the quantum circuit."""
        method_name = self._GATE_DISPATCH.get(gate_name)
        if method_name:
            method = getattr(qc, method_name)
            method(*params, *qubits)
        elif gate_name in self._custom_gates:
            from qiskit.circuit.library import UnitaryGate
            qc.append(UnitaryGate(self._custom_gates[gate_name]), qubits)
        else:
            raise ValueError(f"GATE {gate_name} NOT IMPLEMENTED")

    # ── Display ───────────────────────────────────────────────────────

    def _try_quantum_print(self, text: str, run_vars: dict) -> str | None:
        """Handle quantum PRINT expressions. Returns formatted string or None."""
        import re as _re
        upper = text.strip().upper()
        # PRINT @REG — Dirac notation for LOCC register
        if self.locc_mode and self.locc and upper.startswith('@'):
            reg = upper[1:].strip()
            if reg in self.locc.names:
                sv = self.locc.get_sv(reg)
                n = self.locc.get_n(reg)
                return self._format_dirac(sv, n)
        # PRINT QUBIT(n) — single-qubit Bloch info
        m = _re.match(r'QUBIT\s*\((\d+)\)', text, _re.IGNORECASE)
        if m and self.last_sv is not None:
            q = int(m.group(1))
            x, y, z = self._bloch_vector(self.last_sv, q)
            p1 = self._peek(0x0100 + q * 8)
            return f"q{q}: P(1)={p1:.4f} Bloch=({x:.3f},{y:.3f},{z:.3f})"
        # PRINT ENTANGLEMENT(a,b) — entropy between qubits
        m = _re.match(r'ENTANGLEMENT\s*\((\d+)\s*,\s*(\d+)\)', text, _re.IGNORECASE)
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

    def cmd_state(self, rest: str = '') -> None:
        if self.locc_mode:
            return self._locc_state(rest)
        if self.last_sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        self._print_statevector(self.last_sv)

    # _locc_state and _locc_bloch provided by LOCCMixin.

    def cmd_hist(self) -> None:
        if self.last_counts is None:
            self.io.writeln("?NO RESULTS — RUN first")
            return
        self.print_histogram(self.last_counts)

    def cmd_probs(self) -> None:
        if self.last_sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        self._print_probs(self.last_sv)

    def cmd_bloch(self, rest: str) -> None:
        if self.locc_mode:
            return self._locc_bloch(rest)
        if self.last_sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        if rest:
            q = int(rest)
            self._print_bloch_single(self.last_sv, q)
        else:
            for q in range(min(self.num_qubits, MAX_BLOCH_DISPLAY)):
                self._print_bloch_single(self.last_sv, q)
                if q < min(self.num_qubits, MAX_BLOCH_DISPLAY) - 1:
                    self.io.writeln('')

    def cmd_circuit(self) -> None:
        if self.last_circuit is None:
            self.io.writeln("?NO CIRCUIT — RUN first")
            return
        try:
            self.io.writeln(self.last_circuit.draw(output='text'))
        except Exception:
            print(f"Circuit: {self.last_circuit.num_qubits} qubits, "
                  f"depth {self.last_circuit.depth()}, "
                  f"{self.last_circuit.size()} gates")

    def cmd_noise(self, rest: str) -> None:
        """NOISE <type> <params> — set noise model.

        Types:
          off                          Disable noise
          depolarizing <p>             Depolarizing channel (all gates)
          amplitude_damping <p>        T1-like decay
          phase_flip <p>               T2-like dephasing
          thermal <T1> <T2> <t_gate>   Physical decoherence (microseconds)
          readout <p0> <p1>            Measurement bit-flip error
          combined <p_amp> <p_phase>   Amplitude + phase damping
          pauli <px> <py> <pz>         General Pauli channel
          reset <p0> <p1>              Spontaneous reset error
        """
        if not rest or rest.strip().upper() == 'OFF':
            self._noise_model = None
            self.io.writeln("NOISE OFF")
            return
        parts = rest.split()
        ntype = parts[0].lower()
        try:
            from qiskit_aer.noise import (
                NoiseModel, depolarizing_error, amplitude_damping_error,
                phase_damping_error, thermal_relaxation_error,
                phase_amplitude_damping_error, ReadoutError,
                pauli_error, reset_error,
            )
            nm = NoiseModel()
            _1q = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg',
                   'sx', 'rx', 'ry', 'rz', 'p', 'u', 'id']
            _2q = ['cx', 'cy', 'cz', 'ch', 'swap', 'dcx', 'iswap',
                   'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz']
            _3q = ['ccx', 'cswap']
            if ntype == 'depolarizing':
                p = float(parts[1]) if len(parts) > 1 else 0.01
                nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), _1q)
                nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), _2q)
                nm.add_all_qubit_quantum_error(depolarizing_error(p, 3), _3q)
                self.io.writeln(f"NOISE depolarizing p={p}")
            elif ntype == 'amplitude_damping':
                p = float(parts[1]) if len(parts) > 1 else 0.01
                nm.add_all_qubit_quantum_error(amplitude_damping_error(p), _1q)
                self.io.writeln(f"NOISE amplitude_damping p={p}")
            elif ntype == 'phase_flip':
                p = float(parts[1]) if len(parts) > 1 else 0.01
                nm.add_all_qubit_quantum_error(phase_damping_error(p), _1q)
                self.io.writeln(f"NOISE phase_flip p={p}")
            elif ntype == 'thermal':
                t1 = float(parts[1]) if len(parts) > 1 else 50e-6
                t2 = float(parts[2]) if len(parts) > 2 else 70e-6
                tg = float(parts[3]) if len(parts) > 3 else 1e-6
                err = thermal_relaxation_error(t1, t2, tg)
                nm.add_all_qubit_quantum_error(err, _1q)
                self.io.writeln(f"NOISE thermal T1={t1} T2={t2} t_gate={tg}")
            elif ntype == 'readout':
                p0 = float(parts[1]) if len(parts) > 1 else 0.05
                p1 = float(parts[2]) if len(parts) > 2 else 0.1
                re = ReadoutError([[1 - p0, p0], [p1, 1 - p1]])
                nm.add_all_qubit_readout_error(re)
                self.io.writeln(f"NOISE readout p0={p0} p1={p1}")
            elif ntype == 'combined':
                pa = float(parts[1]) if len(parts) > 1 else 0.01
                pp = float(parts[2]) if len(parts) > 2 else 0.01
                nm.add_all_qubit_quantum_error(
                    phase_amplitude_damping_error(pa, pp), _1q)
                self.io.writeln(f"NOISE combined amp={pa} phase={pp}")
            elif ntype == 'pauli':
                px = float(parts[1]) if len(parts) > 1 else 0.01
                py = float(parts[2]) if len(parts) > 2 else 0.01
                pz = float(parts[3]) if len(parts) > 3 else 0.01
                pi = max(0, 1.0 - px - py - pz)
                err = pauli_error([('X', px), ('Y', py), ('Z', pz), ('I', pi)])
                nm.add_all_qubit_quantum_error(err, _1q)
                self.io.writeln(f"NOISE pauli px={px} py={py} pz={pz}")
            elif ntype == 'reset':
                p0 = float(parts[1]) if len(parts) > 1 else 0.01
                p1 = float(parts[2]) if len(parts) > 2 else 0.01
                nm.add_all_qubit_quantum_error(reset_error(p0, p1), _1q)
                self.io.writeln(f"NOISE reset p0={p0} p1={p1}")
            else:
                self.io.writeln(f"?UNKNOWN NOISE TYPE: {ntype}")
                self.io.writeln("  Types: depolarizing, amplitude_damping, phase_flip, thermal,")
                self.io.writeln("         readout, combined, pauli, reset")
                return
            self._noise_model = nm
        except ImportError:
            self.io.writeln("?Noise model requires qiskit-aer noise module")

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
        print(f"\n  Circuit: {self.last_circuit.num_qubits} qubits, "
              f"depth {self.last_circuit.depth()}, {self.last_circuit.size()} gates")
        for name, count in sorted(ops.items(), key=lambda x: -x[1]):
            bar = '█' * min(count, 40)
            self.io.writeln(f"    {name:>10}  {count:>4}  {bar}")
        self.io.writeln('')

    # cmd_include, cmd_sweep provided by FileIOMixin / SweepMixin.

    # LOCC commands (cmd_locc, cmd_send, cmd_share) provided by LOCCMixin.

    # cmd_bench, cmd_ram, cmd_dir provided by AnalysisMixin / FileIOMixin.

    def cmd_clear(self, rest: str) -> None:
        """CLEAR var — remove a variable. CLEAR ARRAYS — clear all arrays."""
        name = rest.strip()
        if not name:
            self.io.writeln("?USAGE: CLEAR <var> or CLEAR ARRAYS")
            return
        if name.upper() == 'ARRAYS':
            self.arrays.clear()
            self.io.writeln("ARRAYS CLEARED")
        elif name in self.variables:
            del self.variables[name]
            self.io.writeln(f"CLEARED {name}")
        elif name in self.arrays:
            del self.arrays[name]
            self.io.writeln(f"CLEARED array {name}")
        else:
            self.io.writeln(f"?{name} NOT FOUND")

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
        print(textwrap.dedent("""
        QBASIC — Quantum BASIC Terminal
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        PROGRAM EDITING
          10 H 0              Enter a numbered line (program statement)
          10                  Delete line 10
          LIST                Show the program
          DELETE 10           Delete line 10
          DELETE 10-50        Delete range
          RENUM               Renumber lines (10, 20, 30, ...)
          NEW                 Clear everything
          SAVE <file>          Save program to .qb file
          LOAD <file>          Load program from .qb file
          RUN                 Execute the program

        GATES (immediate or in a program)
          H 0                 Hadamard on qubit 0
          X 0 / Y 0 / Z 0    Pauli gates
          S 0 / T 0           Phase gates (and SDG, TDG)
          RX PI/4, 0          Rotation gates (RX, RY, RZ, P)
          CX 0, 1             CNOT (control, target)
          CZ 0, 1             Controlled-Z
          CCX 0, 1, 2         Toffoli
          SWAP 0, 1           Swap two qubits
          MEASURE             Add measurements (use in program)

        MULTI-GATE LINES
          10 H 0 : CX 0,1    Multiple gates on one line with ':'

        REGISTERS & SUBROUTINES
          REG data 3          Name a group of qubits
          10 H data[0]        Use register notation
          REGS                List registers
          DEF BELL = H 0 : CX 0,1
                              Define a named gate sequence
          10 BELL             Use it in a program
          10 BELL @2          Use with qubit offset
          DEFS                List subroutines

        VARIABLES & LOOPS
          LET angle = PI/4    Set a variable
          10 RX angle, 0      Use in gate parameters
          10 FOR I = 0 TO 3   Loop (variable substitution in body)
          20   H I
          30 NEXT I
          VARS                List variables

        DISPLAY
          STATE               Show statevector (after RUN)
          HIST                Show measurement histogram
          PROBS               Show probability distribution
          BLOCH [n]           ASCII Bloch sphere for qubit n (or all)
          STEP                Step through program with state display
          CIRCUIT             Show circuit diagram

        CONFIGURATION
          QUBITS n            Set number of qubits (default: 4)
          SHOTS n             Set number of shots (default: 1024)
          METHOD name         Set simulation method (automatic, statevector,
                              matrix_product_state, stabilizer, ...)

        DEMOS
          DEMO LIST           List available demos
          DEMO BELL           Bell state
          DEMO GHZ            GHZ entanglement
          DEMO TELEPORT       Quantum teleportation
          DEMO GROVER         Grover's search
          DEMO QFT            Quantum Fourier transform
          DEMO DEUTSCH        Deutsch-Jozsa algorithm
          DEMO BERNSTEIN      Bernstein-Vazirani
          DEMO SUPERDENSE     Superdense coding
          DEMO RANDOM         Quantum random number generator
          DEMO STRESS         20-qubit stress test
          DEMO LOCC-TELEPORT  Teleportation across A/B boundary (JOINT)
          DEMO LOCC-COORD     Classical coordination (SPLIT)

        LOCC MODE (dual-register distributed quantum simulation)
          LOCC <n_a> <n_b>          SPLIT mode: two independent registers
          LOCC JOINT <n_a> <n_b>    JOINT mode: shared entanglement possible
          LOCC OFF                  Back to normal Aer mode
          LOCC STATUS               Show register info
          @A H 0                    Gate on register A
          @B CX 0,1                 Gate on register B
          SEND A 0 -> x             Mid-circuit measure A[0], store in x
          IF x THEN @B X 0          Conditional gate based on classical bit
          SHARE A 2, B 0            Create Bell pair (JOINT mode only)
          STATE A / STATE B         Inspect register states
          BLOCH A 0 / BLOCH B 0     Bloch spheres per register

          SPLIT: max capacity (31+31), no cross-register entanglement
          JOINT: shared entanglement, limited to ~32 total qubits
          LOCCINFO                  Protocol metrics after run

        BASIS MEASUREMENT
          MEASURE_X qubit         Measure in X basis (H before measure)
          MEASURE_Y qubit         Measure in Y basis (SDG+H before measure)
          MEASURE_Z qubit         Measure in Z basis (standard)
          Results stored in mx_<q>, my_<q>, mz_<q> variables.

        ERROR CORRECTION
          SYNDROME ZZ 0 1 -> s0   Measure Pauli stabilizer non-destructively
          Uses an ancilla (highest qubit index). Pauli string
          length must match qubit count. I/X/Y/Z supported.

        ADVANCED
          UNITARY NAME = [[..]]   Define gate from unitary matrix
          CTRL gate ctrl, tgt     Controlled version of any gate
          INV gate qubit          Inverse/dagger of a gate
          RESET qubit             Reset qubit to |0>
          SWEEP var s e [n]       Run circuit sweeping a variable
          NOISE type [p]          Set noise model (depolarizing, etc.)
          NOISE OFF               Disable noise
          EXPECT Z 0              Expectation value of Pauli operator
          DENSITY                 Show density matrix
          ENTROPY [qubits]        Entanglement entropy
          DECOMPOSE               Gate count breakdown
          EXPORT [file]           Export circuit as OpenQASM
          CSV [file]              Export results as CSV
          RAM                     Memory budget and parallelism estimates
          BENCH                   Benchmark qubit scaling
          INCLUDE file            Merge another .qb file
          DIR [path]              List .qb files
          CLEAR var               Remove a variable or array
          UNDO                    Undo last program edit

        FLOW CONTROL (in programs)
          GOTO line               Jump to line
          GOSUB line / RETURN     Subroutine call with stack
          WHILE expr / WEND       Conditional loop
          IF expr THEN ... ELSE   Conditional (supports expressions)
          END                     Stop execution
          PRINT expr              Output during run
          INPUT "prompt", var     Read user input
          DIM arr(size)           Declare array
          LET arr[i] = val        Array assignment

        EXPRESSIONS
          PI, TAU, E, SQRT2, sin(), cos(), sqrt(), log(), etc.
          Comparisons: ==, !=, <, >, <=, >=, AND, OR, NOT
          Arrays: arr(i) or arr[i]
          Example: LET theta = PI/4 + asin(0.5)
        """))
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
        try:
            gpu_str = ""
            import subprocess
            r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=2)
            if r.returncode == 0 and r.stdout.strip():
                gpu_str = f" | GPU: {r.stdout.strip().split(chr(10))[0]}"
        except Exception:
            gpu_str = ""
        print(textwrap.dedent(f"""\

        ██████╗ ██████╗  █████╗ ███████╗██╗ ██████╗
        ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██║██╔════╝
        ██║   ██║██████╔╝███████║███████╗██║██║
        ██║▄▄ ██║██╔══██╗██╔══██║╚════██║██║██║
        ╚██████╔╝██████╔╝██║  ██║███████║██║╚██████╗
         ╚══▀▀═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝

        Quantum BASIC
        Python {platform.python_version()} | Qiskit {qver} | {ram_str}{gpu_str}
        {self.num_qubits} qubits | {self.shots} shots | max ~{max_q} qubits
        Type HELP for commands, DEMO LIST for demos.
        """))

