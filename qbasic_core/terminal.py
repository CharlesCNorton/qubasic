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


# ═══════════════════════════════════════════════════════════════════════
# The Terminal
# ═══════════════════════════════════════════════════════════════════════

class QBasicTerminal(ExpressionMixin, DisplayMixin, DemoMixin, LOCCMixin, ControlFlowMixin,
                     FileIOMixin, AnalysisMixin, SweepMixin,
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
    def _get_parsed(self, line_num: int):
        """Get parsed statement for a line, lazily parsing if needed."""
        p = self._parsed.get(line_num)
        if p is None:
            raw = self.program.get(line_num, '')
            p = parse_stmt(raw)
            self._parsed[line_num] = p
        return p

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
        self.program = {}           # {line_num: source_string}
        self.num_qubits = DEFAULT_QUBITS
        self.shots = DEFAULT_SHOTS
        self.subroutines = {}       # {NAME: {body, params} or [strings]}
        self.registers = OrderedDict()  # {name: (start_qubit, size)}
        self.variables = {}         # {name: value}
        self.arrays = {}            # {name: [values]}
        self._undo_stack = []       # for UNDO
        self._gosub_stack = []      # for GOSUB/RETURN
        self._custom_gates = {}     # {NAME: np.array matrix}
        self._noise_model = None    # qiskit noise model
        self._max_iterations = MAX_LOOP_ITERATIONS
        self._include_depth = 0     # guard against recursive INCLUDE
        self.last_counts = None
        self.last_sv = None
        self.last_circuit = None
        self.step_mode = False
        self.sim_method = 'automatic'
        self.sim_device = 'CPU'
        # LOCC mode
        self.locc = None
        self.locc_mode = False
        # I/O and parse cache
        self.io = StdIOPort()
        self._parsed = {}  # {line_num: Stmt}
        self._circuit_cache_key = None
        self._circuit_cache = None  # (qc_transpiled, backend)
        # Subsystem initialization
        self._start_time = time.time()
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
                line = input(self._prompt).strip()
                if line:
                    self.process(line)
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print("\nBYE")
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
                    print(f"DELETED {num}")
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
        'STATS': 'cmd_stats',
        # Program management
        'AUTO': 'cmd_auto', 'EDIT': 'cmd_edit', 'COPY': 'cmd_copy',
        'MOVE': 'cmd_move', 'FIND': 'cmd_find', 'REPLACE': 'cmd_replace',
        'BANK': 'cmd_bank', 'CHAIN': 'cmd_chain', 'MERGE': 'cmd_merge',
        # File handles
        'OPEN': 'cmd_open', 'CLOSE': 'cmd_close',
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
                print(f"?{e.message}")
            except Exception as e:
                print(f"?ERROR: {e}")
        else:
            # Try as immediate gate / subroutine
            try:
                self.run_immediate(line)
            except Exception as e:
                print(f"?SYNTAX ERROR: {e}")

    def _quit(self) -> None:
        raise EOFError

    # ── Commands ──────────────────────────────────────────────────────

    def cmd_qubits(self, rest: str) -> None:
        """Set or display the number of qubits. Range: 1 to MAX_QUBITS."""
        if not rest:
            print(f"QUBITS = {self.num_qubits}")
            return
        n = int(rest)
        if n < 1 or n > MAX_QUBITS:
            from qbasic_core.errors import QBasicRangeError
            raise QBasicRangeError(f"RANGE: 1-{MAX_QUBITS}")

        self.num_qubits = n
        self.registers.clear()
        est = _estimate_gb(n)
        print(f"{n} QUBITS ALLOCATED  (~{est:.1f} GB per instance)")
        ram = _get_ram_gb()
        if ram:
            total, avail = ram
            budget = total * RAM_BUDGET_FRACTION
            if est > avail:
                print(f"  WARNING: exceeds available RAM ({avail:.1f} GB free of {total:.0f} GB)")
            elif est > budget * 0.5:
                print(f"  NOTE: uses {est/budget*100:.0f}% of 80% RAM budget ({budget:.1f} GB)")
            if n >= 16 and est > 0:
                max_par = int(budget / est)
                if max_par > 0:
                    print(f"  Max parallel instances in 80% budget: ~{max_par}")

    def cmd_shots(self, rest: str) -> None:
        """Set or display the number of measurement shots."""
        if not rest:
            print(f"SHOTS = {self.shots}")
            return
        self.shots = max(1, int(rest))
        print(f"SHOTS = {self.shots}")

    def cmd_method(self, rest: str) -> None:
        """Set simulation method (statevector, stabilizer, MPS, ...) or device (CPU/GPU)."""
        if not rest:
            print(f"METHOD = {self.sim_method}  DEVICE = {self.sim_device}")
            methods = ['automatic', 'statevector', 'density_matrix',
                       'stabilizer', 'matrix_product_state', 'extended_stabilizer']
            print(f"  methods: {', '.join(methods)}")
            print(f"  devices: CPU, GPU")
            return
        val = rest.strip().upper()
        if val in ('GPU', 'CPU'):
            self.sim_device = val
            print(f"DEVICE = {self.sim_device}")
        else:
            self.sim_method = rest.strip().lower()
            print(f"METHOD = {self.sim_method}")

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
            print("EMPTY PROGRAM")
            return
        lines = sorted(self.program.keys())
        for num in lines:
            print(f"  {num:5d}  {self.program[num]}")

    def cmd_new(self, *, silent: bool = False) -> None:
        """NEW — clear program, subroutines, registers, and variables."""
        self.program.clear()
        self._parsed.clear()
        self.subroutines.clear()
        self.registers.clear()
        self.variables.clear()
        self.last_counts = None
        self.last_sv = None
        self.last_circuit = None
        if not silent:
            print("READY")

    def cmd_delete(self, rest: str) -> None:
        """DELETE <line> or DELETE <start>-<end> — remove program lines."""
        if not rest:
            print("?USAGE: DELETE <line> or DELETE <start>-<end>")
            return
        if '-' in rest:
            a, b = rest.split('-')
            a, b = int(a.strip()), int(b.strip())
            for k in list(self.program.keys()):
                if a <= k <= b:
                    del self.program[k]
                    self._parsed.pop(k, None)
            print(f"DELETED {a}-{b}")
        else:
            n = int(rest)
            if n in self.program:
                del self.program[n]
                self._parsed.pop(n, None)
                print(f"DELETED {n}")
            else:
                print(f"?LINE {n} NOT FOUND")

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
        print(f"RENUMBERED {len(new_prog)} LINES (start={start}, step={step})")

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
            print("?USAGE: DEF NAME[(params)] = GATE : GATE : ...")
            print("        DEF BEGIN NAME[(params)]  (multi-line, end with DEF END)")
            return
        name = m.group(1).upper()
        params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
        body = [s.strip() for s in m.group(3).split(':') if s.strip()]
        if name in GATE_TABLE:
            from qbasic_core.errors import QBasicSyntaxError
            raise QBasicSyntaxError(f"CANNOT REDEFINE BUILT-IN GATE {name}")
        self.subroutines[name] = {'body': body, 'params': params}
        if params:
            print(f"DEF {name}({', '.join(params)}) ({len(body)} gates)")
        else:
            print(f"DEF {name} ({len(body)} gates)")

    def _def_multiline(self, header: str) -> None:
        """Read multi-line DEF block from REPL."""
        m = re.match(r'(\w+)(?:\(([^)]*)\))?', header)
        if not m:
            print("?USAGE: DEF BEGIN NAME[(params)]")
            return
        name = m.group(1).upper()
        params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
        if name in GATE_TABLE:
            from qbasic_core.errors import QBasicSyntaxError
            raise QBasicSyntaxError(f"CANNOT REDEFINE BUILT-IN GATE {name}")

        body = []
        print(f"  DEF {name} (type gates, DEF END to finish)")
        while True:
            try:
                line = input('  . ').strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  CANCELLED")
                return
            if line.upper() == 'DEF END' or line.upper() == 'END':
                break
            if line:
                body.append(line)
        self.subroutines[name] = {'body': body, 'params': params}
        print(f"DEF {name} ({len(body)} gates)")

    def cmd_reg(self, rest: str) -> None:
        """REG <name> <size> — allocate a named qubit register."""
        # REG data 3
        parts = rest.split()
        if len(parts) != 2:
            print("?USAGE: REG <name> <size>")
            return
        name = parts[0].lower()
        size = int(parts[1])
        # Allocate starting after existing registers
        start = sum(s for _, s in self.registers.values())
        if start + size > self.num_qubits:
            from qbasic_core.errors import QBasicRangeError
            raise QBasicRangeError(f"NOT ENOUGH QUBITS (need {start+size}, have {self.num_qubits})")

        self.registers[name] = (start, size)
        print(f"REG {name} = qubits {start}-{start+size-1}")

    def cmd_let(self, rest: str) -> None:
        """LET <var> = <expr> — assign a computed value to a variable."""
        # LET angle = PI/4
        m = re.match(r'(\w+)\s*=\s*(.*)', rest)
        if not m:
            print("?USAGE: LET <var> = <expr>")
            return
        name = m.group(1)
        val = self.eval_expr(m.group(2))
        self.variables[name] = val
        print(f"{name} = {val}")

    def cmd_defs(self) -> None:
        """List all defined subroutines."""
        if not self.subroutines:
            print("NO SUBROUTINES DEFINED")
            return
        for name, sub in self.subroutines.items():
            if isinstance(sub, list):
                print(f"  {name} = {' : '.join(sub)}")
            else:
                params = f"({', '.join(sub['params'])})" if sub['params'] else ""
                print(f"  {name}{params} = {' : '.join(sub['body'])}")

    def cmd_regs(self) -> None:
        """List all named registers with their qubit ranges."""
        if not self.registers:
            print("NO REGISTERS DEFINED")
            return
        for name, (start, size) in self.registers.items():
            print(f"  {name}[0:{size}] -> qubits {start}-{start+size-1}")

    def cmd_vars(self) -> None:
        """List all variables and their current values."""
        if not self.variables:
            print("NO VARIABLES SET")
            return
        for name, val in self.variables.items():
            print(f"  {name} = {val}")

    # ── Run ───────────────────────────────────────────────────────────

    def cmd_run(self) -> None:
        """Execute the stored program."""
        if self.locc_mode:
            return self._locc_run()
        if not self.program:
            print("NOTHING TO RUN")
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
            print("\n?INTERRUPTED")
            return
        except Exception as e:
            print(f"?BUILD ERROR: {e}")
            return

        # Copy before adding measurements (for statevector extraction)
        qc_sv = qc.copy()

        if has_measure:
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
            cache_key = (
                tuple(sorted(self.program.items())),
                self.num_qubits, method, self.sim_device,
                id(self._noise_model),
            )
            if self._circuit_cache_key == cache_key and self._circuit_cache is not None:
                qc_t, backend = self._circuit_cache
            else:
                backend = AerSimulator(**backend_opts)
                qc_t = transpile(qc, backend)
                self._circuit_cache_key = cache_key
                self._circuit_cache = (qc_t, backend)
            result = backend.run(qc_t, shots=self.shots).result()
            self.last_counts = dict(result.get_counts())
        except Exception as e:
            print(f"?RUNTIME ERROR: {e}")
            return

        # Get statevector from the measurement-free copy
        try:
            qc_sv.save_statevector()
            sv_backend = AerSimulator(method='statevector')
            sv_result = sv_backend.run(transpile(qc_sv, sv_backend)).result()
            self.last_sv = np.array(sv_result.get_statevector())
        except Exception:
            self.last_sv = None

        dt = time.time() - t0

        # Update status registers
        depth = qc.depth()
        n_gates = qc.size()
        self._update_status(gate_count=n_gates, circuit_depth=depth,
                           run_time_ms=dt * 1000)

        # Display results
        print(f"\nRAN {len(self.program)} lines, {self.num_qubits} qubits, "
              f"{self.shots} shots in {dt:.2f}s  [depth={depth}, gates={n_gates}]")
        if has_measure:
            self.print_histogram(self.last_counts)
            self._auto_display()
        else:
            print("(no MEASURE in program — use STATE or PROBS to inspect)")

    def cmd_step(self) -> None:
        """Step through program, showing state after each line."""
        if not self.program:
            print("NOTHING TO STEP")
            return

        sorted_lines = sorted(self.program.keys())
        qc = QuantumCircuit(self.num_qubits)
        loop_stack = []
        ip = 0
        step_vars = dict(self.variables)

        print(f"STEP MODE — {len(sorted_lines)} lines, {self.num_qubits} qubits")
        print("Press ENTER to advance, Q to quit\n")

        _iters = 0
        while ip < len(sorted_lines):
            _iters += 1
            if _iters > self._max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({self._max_iterations}) — possible infinite loop")
            line_num = sorted_lines[ip]
            stmt = self.program[line_num]

            # Display current line
            print(f">> {line_num:5d}  {stmt}")

            # Execute it
            result = self._exec_line(stmt, qc, loop_stack, sorted_lines, ip, step_vars)

            # Show state
            try:
                qc_tmp = qc.copy()
                qc_tmp.save_statevector()
                sv_b = AerSimulator(method='statevector')
                sv_r = sv_b.run(transpile(qc_tmp, sv_b)).result()
                sv = np.array(sv_r.get_statevector())
                self._print_sv_compact(sv)
            except Exception:
                print("   (state unavailable)")

            # Wait for input
            try:
                user = input("   [ENTER/Q] ").strip().upper()
                if user == 'Q':
                    print("STOPPED")
                    return
            except (KeyboardInterrupt, EOFError):
                print("\nSTOPPED")
                return

            if isinstance(result, int):
                ip = result
            else:
                ip += 1

        print("DONE")

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
                    print(f"?UNKNOWN REGISTER: {reg} (have {', '.join(self.locc.names)})")
                    return
                self._locc_apply_gate(reg, gate_stmt)
                self._locc_state()
                return
        if line.strip().startswith('@'):
            print("?@register syntax requires LOCC mode (try: LOCC <n1> <n2>)")
            return
        # Build and execute through the same gate pipeline as cmd_run
        qc = QuantumCircuit(self.num_qubits)
        self._exec_line(line, qc, [], [0], 0, dict(self.variables))
        qc.save_statevector()
        backend = AerSimulator(method='statevector')
        result = backend.run(transpile(qc, backend)).result()
        sv = np.array(result.get_statevector())
        self.last_sv = sv
        self.last_circuit = qc
        self._print_sv_compact(sv)

    def _validate_program(self, sorted_lines: list[int]) -> None:
        """Pre-execution validation. Catches structural errors before running."""
        from qbasic_core.statements import GotoStmt, GosubStmt
        line_set = set(sorted_lines)
        for ln in sorted_lines:
            parsed = self._get_parsed(ln)
            if isinstance(parsed, GotoStmt) and parsed.target not in line_set:
                raise RuntimeError(f"LINE {ln}: GOTO {parsed.target} — target line not found")
            if isinstance(parsed, GosubStmt) and parsed.target not in line_set:
                raise RuntimeError(f"LINE {ln}: GOSUB {parsed.target} — target line not found")

    # ── Circuit Building ──────────────────────────────────────────────

    def build_circuit(self) -> tuple['QuantumCircuit', bool]:
        """Compile program lines into a QuantumCircuit. Returns (circuit, has_measure)."""
        from qbasic_core.exec_context import ExecContext
        from qbasic_core.statements import MeasureStmt
        from qbasic_core.scope import Scope

        qc = QuantumCircuit(self.num_qubits)
        ctx = ExecContext(
            sorted_lines=sorted(self.program.keys()),
            ip=0,
            run_vars=Scope(self.variables),
            max_iterations=self._max_iterations,
            qc=qc,
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
                result = self._exec_line(stmt, qc, ctx.loop_stack, ctx.sorted_lines, ctx.ip, ctx.run_vars, parsed)
            except Exception as e:
                raise RuntimeError(f"LINE {line_num}: {e}") from None

            if result is ExecResult.END:
                break
            elif isinstance(result, int):
                ctx.ip = result
            else:
                ctx.ip += 1

        return qc, has_measure

    def _try_exec_meas(self, stmt: str, qc: 'QuantumCircuit', run_vars: dict) -> bool:
        """Handle MEAS qubit -> var (mid-circuit measurement).

        Limitation: in the Qiskit circuit-build path, the variable is set to 0
        unconditionally because Qiskit defers measurement to execution time.
        The actual outcome is not available during circuit construction. For
        true classical feedforward, use LOCC mode with SEND instead.
        """
        m = RE_MEAS.match(stmt)
        if not m:
            return False
        qubit = int(self._eval_with_vars(m.group(1), run_vars))
        var = m.group(2)
        if 0 <= qubit < self.num_qubits:
            from qiskit.circuit import ClassicalRegister
            cr = ClassicalRegister(1, f'meas_{var}')
            qc.add_register(cr)
            qc.measure(qubit, cr[0])
            run_vars[var] = 0
            self.variables[var] = 0
            self.io.writeln(f"  ?WARNING: MEAS {var} is always 0 in circuit mode — "
                          f"IF/THEN branches conditioned on this variable will "
                          f"always take the 0 path. Use LOCC mode with SEND for "
                          f"classical feedforward.")
        return True

    def _try_exec_reset(self, stmt: str, qc: 'QuantumCircuit', run_vars: dict) -> bool:
        """Handle RESET qubit."""
        m = RE_RESET.match(stmt)
        if not m:
            return False
        qubit = int(self._eval_with_vars(m.group(1), run_vars))
        if 0 <= qubit < self.num_qubits:
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
                raise ValueError("Matrix is not unitary (U @ U† ≠ I)")
            self._custom_gates[name] = matrix
            print(f"UNITARY {name}: {n_qubits}-qubit gate defined")
        except Exception as e:
            print(f"?UNITARY ERROR: {e}")
        return True

    def _try_exec_dim(self, stmt: str) -> bool:
        """Handle DIM name(size) array declaration."""
        m = RE_DIM.match(stmt)
        if not m:
            return False
        self.arrays[m.group(1)] = [0.0] * int(m.group(2))
        return True

    def _try_exec_redim(self, stmt: str) -> bool:
        """Handle REDIM name(size) — resize an existing array."""
        from qbasic_core.engine import RE_REDIM
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
        from qbasic_core.engine import RE_ERASE
        m = RE_ERASE.match(stmt)
        if not m:
            return False
        name = m.group(1)
        if name in self.arrays:
            del self.arrays[name]
        return True

    def _try_exec_get(self, stmt: str, run_vars: dict) -> bool:
        """Handle GET var — single keypress input without enter."""
        from qbasic_core.engine import RE_GET
        m = RE_GET.match(stmt)
        if not m:
            return False
        var = m.group(1)
        try:
            import sys as _sys
            if _sys.platform == 'win32':
                import msvcrt
                ch = msvcrt.getwch()
            else:
                import tty, termios
                fd = _sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    ch = _sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            if var.endswith('$'):
                run_vars[var] = ch
                self.variables[var] = ch
            else:
                run_vars[var] = float(ord(ch))
                self.variables[var] = float(ord(ch))
        except (EOFError, KeyboardInterrupt, OSError):
            run_vars[var] = '' if var.endswith('$') else 0
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

    def _try_exec_measure_basis(self, stmt: str, qc: 'QuantumCircuit',
                               run_vars: dict) -> bool:
        """Handle MEASURE_X/Y/Z qubit — measurement in a non-computational basis.

        MEASURE_X q: applies H then measures (X-basis measurement).
        MEASURE_Y q: applies SDG then H then measures (Y-basis measurement).
        MEASURE_Z q: standard computational-basis measurement (same as MEAS).
        The result is stored in a variable named mx_<q>, my_<q>, or mz_<q>.
        """
        m = RE_MEASURE_BASIS.match(stmt)
        if not m:
            return False
        basis = m.group(1).upper()
        qubit = int(self._eval_with_vars(m.group(2), run_vars))
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"QUBIT {qubit} OUT OF RANGE (0-{self.num_qubits-1})")
        # Rotate into the requested basis before measurement
        if basis == 'X':
            qc.h(qubit)
        elif basis == 'Y':
            qc.sdg(qubit)
            qc.h(qubit)
        # Z needs no rotation — already computational basis
        from qiskit.circuit import ClassicalRegister
        var = f"m{basis.lower()}_{qubit}"
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

    def _try_exec_syndrome(self, stmt: str, qc: 'QuantumCircuit',
                           run_vars: dict) -> bool:
        """Handle SYNDROME <stabilizer_spec> — measure a Pauli stabilizer.

        Syntax: SYNDROME <pauli_string> <qubits> -> <var>
        Example: SYNDROME ZZ 0 1 -> s0   (measures Z tensor Z on qubits 0,1)

        Implements non-destructive stabilizer measurement using an ancilla:
        1. Allocate an ancilla qubit (highest index).
        2. For each Pauli in the string, apply controlled-Pauli from ancilla.
        3. Measure the ancilla to get the syndrome bit.
        """
        parsed = self._parse_syndrome(stmt, run_vars)
        if parsed is None:
            return False
        pauli_str, qubits, var = parsed
        anc = self.num_qubits - 1
        if anc in qubits:
            raise ValueError(
                f"Qubit {anc} needed as ancilla but appears in stabilizer. "
                f"Increase QUBITS by 1.")
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

    def _exec_line(self, stmt, qc, loop_stack, sorted_lines, ip, run_vars, parsed=None):
        """Execute one program line.

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
        from qbasic_core.statements import (
            BarrierStmt, RemStmt, MeasureStmt, EndStmt, ReturnStmt,
            CompoundStmt, AtRegStmt, GotoStmt, GosubStmt,
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
        if isinstance(parsed, AtRegStmt) and not self.locc_mode:
            raise ValueError("@register syntax requires LOCC mode (try: LOCC <n1> <n2>)")
        if isinstance(parsed, CompoundStmt):
            for sub in parsed.parts:
                self._exec_line(sub, qc, loop_stack, sorted_lines, ip, run_vars)
            return ExecResult.ADVANCE

        # 2. Control flow
        def _recurse(s, ls, sl, i, rv):
            return self._exec_line(s, qc, ls, sl, i, rv)
        handled, result = self._exec_control_flow(
            stmt, loop_stack, sorted_lines, ip, run_vars, _recurse)
        if handled:
            return result

        # 3. Statement handlers
        if self._try_stmt_handlers(stmt, qc, run_vars):
            return ExecResult.ADVANCE  # all statement handlers advance

        # 4. Colon-separated (legacy fallback for unparsed compound)
        if ':' in stmt:
            for sub in self._split_colon_stmts(stmt):
                self._exec_line(sub, qc, loop_stack, sorted_lines, ip, run_vars)
            return ExecResult.ADVANCE

        # 5. Gate application
        expanded = self._expand_statement(stmt)
        for gate_str in expanded:
            self._apply_gate_str(gate_str, qc)

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

    def _try_stmt_handlers(self, stmt: str, qc: 'QuantumCircuit', run_vars: dict) -> bool:
        """Try statement-type handlers in order. Returns True if handled."""
        return (
            self._try_exec_meas(stmt, qc, run_vars)
            or self._try_exec_reset(stmt, qc, run_vars)
            or self._try_exec_measure_basis(stmt, qc, run_vars)
            or self._try_exec_syndrome(stmt, qc, run_vars)
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
            or self._try_exec_dim_multi(stmt)
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
        # Simple format: # for digit, . for decimal point
        result = fmt
        for val in vals:
            n_hash = result.count('#')
            if n_hash > 0:
                if '.' in result:
                    decimal = result.count('#') - result.index('.') if '.' in result else 0
                    formatted = f"{val:{n_hash}.{max(0, n_hash - result.index('#'))}f}"
                else:
                    formatted = f"{val:{n_hash}.0f}"
                # Replace first run of # with formatted value
                idx = result.find('#')
                end = idx
                while end < len(result) and result[end] in '#.':
                    end += 1
                result = result[:idx] + formatted.rjust(end - idx) + result[end:]
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

    def _try_exec_dim_multi(self, stmt: str) -> bool:
        m = RE_DIM_MULTI.match(stmt)
        if not m:
            return False
        name = m.group(1)
        dims = [int(d.strip()) for d in m.group(2).split(',')]
        total = 1
        for d in dims:
            total *= d
        self.arrays[name] = [0.0] * total
        # Store dimension metadata for multi-dim indexing
        if len(dims) > 1:
            if not hasattr(self, '_array_dims'):
                self._array_dims = {}
            self._array_dims[name] = dims
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

        result = []
        for gate_str in body:
            # Substitute parameters
            gs = gate_str
            for i, pname in enumerate(param_names):
                if i < len(call_args):
                    gs = re.sub(r'\b' + re.escape(pname) + r'\b', call_args[i], gs)
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

    def _apply_gate_str(self, stmt, qc, _call_stack=None):
        """Parse and apply a single gate string to the circuit."""
        stmt = stmt.strip()
        if not stmt:
            return

        # Expand subroutines with call-stack tracking for recursion detection
        word = stmt.split()[0].upper() if stmt.split() else ''
        if word in self.subroutines:
            for sub_stmt in self._expand_statement(stmt, _call_stack):
                self._apply_gate_str(sub_stmt, qc, _call_stack)
            return

        upper = stmt.upper()
        if upper.startswith('REM') or upper.startswith("'") or upper == 'BARRIER':
            if upper == 'BARRIER':
                qc.barrier()
            return
        if upper == 'MEASURE':
            return  # handled at run level

        # CTRL gate ctrl_qubit, target_qubit(s) — controlled version of any gate
        m_ctrl = RE_CTRL.match(stmt)
        if m_ctrl:
            from qiskit.circuit.library import (HGate, XGate, YGate, ZGate,
                SGate, TGate, SdgGate, TdgGate, SXGate, RXGate, RYGate,
                RZGate, PhaseGate, SwapGate, UGate)
            gate_name = m_ctrl.group(1).upper()
            args = [a.strip() for a in m_ctrl.group(2).replace(',', ' ').split()]
            ctrl_qubit = self._resolve_qubit(args[0])
            target_qubits = [self._resolve_qubit(a) for a in args[1:]]
            gate_map = {
                'H': HGate(), 'X': XGate(), 'Y': YGate(), 'Z': ZGate(),
                'S': SGate(), 'T': TGate(), 'SDG': SdgGate(), 'TDG': TdgGate(),
                'SX': SXGate(), 'SWAP': SwapGate(),
            }
            if gate_name in gate_map:
                qc.append(gate_map[gate_name].control(1), [ctrl_qubit] + target_qubits)
            elif gate_name in self._custom_gates:
                from qiskit.circuit.library import UnitaryGate
                qc.append(UnitaryGate(self._custom_gates[gate_name]).control(1),
                          [ctrl_qubit] + target_qubits)
            else:
                raise ValueError(f"CTRL {gate_name}: gate not found")
            return

        # INV gate qubit(s) — inverse/dagger of a single gate
        m_inv = RE_INV.match(stmt)
        if m_inv:
            gate_name = m_inv.group(1).upper()
            inv_args = m_inv.group(2)
            # Build a minimal 1-gate circuit, invert it
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

        # Apply gate
        self._apply_gate(qc, gate_name, params, qubits)

    def _tokenize_gate(self, stmt: str) -> list[str]:
        """Split gate statement into tokens, handling commas and register notation."""
        # Replace commas with spaces, split
        stmt = stmt.replace(',', ' ')
        # Handle register[index] notation
        stmt = RE_REG_INDEX.sub(r'\1[\2]', stmt)
        return stmt.split()

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

    def cmd_state(self, rest: str = '') -> None:
        if self.locc_mode:
            return self._locc_state(rest)
        if self.last_sv is None:
            print("?NO STATE — RUN first")
            return
        self._print_statevector(self.last_sv)

    # _locc_state and _locc_bloch provided by LOCCMixin.

    def cmd_hist(self) -> None:
        if self.last_counts is None:
            print("?NO RESULTS — RUN first")
            return
        self.print_histogram(self.last_counts)

    def cmd_probs(self) -> None:
        if self.last_sv is None:
            print("?NO STATE — RUN first")
            return
        self._print_probs(self.last_sv)

    def cmd_bloch(self, rest: str) -> None:
        if self.locc_mode:
            return self._locc_bloch(rest)
        if self.last_sv is None:
            print("?NO STATE — RUN first")
            return
        if rest:
            q = int(rest)
            self._print_bloch_single(self.last_sv, q)
        else:
            for q in range(min(self.num_qubits, MAX_BLOCH_DISPLAY)):
                self._print_bloch_single(self.last_sv, q)
                if q < min(self.num_qubits, MAX_BLOCH_DISPLAY) - 1:
                    print()

    def cmd_circuit(self) -> None:
        if self.last_circuit is None:
            print("?NO CIRCUIT — RUN first")
            return
        try:
            print(self.last_circuit.draw(output='text'))
        except Exception:
            print(f"Circuit: {self.last_circuit.num_qubits} qubits, "
                  f"depth {self.last_circuit.depth()}, "
                  f"{self.last_circuit.size()} gates")

    def cmd_noise(self, rest: str) -> None:
        """NOISE <type> <param> — set noise model.
        Types: off, depolarizing <p>, amplitude_damping <p>, phase_flip <p>"""
        if not rest or rest.strip().upper() == 'OFF':
            self._noise_model = None
            print("NOISE OFF")
            return
        parts = rest.split()
        ntype = parts[0].lower()
        param = float(parts[1]) if len(parts) > 1 else 0.01
        try:
            from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
            nm = NoiseModel()
            _1q_gates = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg',
                         'sx', 'rx', 'ry', 'rz', 'p', 'u', 'id']
            _2q_gates = ['cx', 'cy', 'cz', 'ch', 'swap', 'dcx', 'iswap',
                         'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz']
            _3q_gates = ['ccx', 'cswap']
            if ntype == 'depolarizing':
                err_1q = depolarizing_error(param, 1)
                err_2q = depolarizing_error(param, 2)
                err_3q = depolarizing_error(param, 3)
                nm.add_all_qubit_quantum_error(err_1q, _1q_gates)
                nm.add_all_qubit_quantum_error(err_2q, _2q_gates)
                nm.add_all_qubit_quantum_error(err_3q, _3q_gates)
            elif ntype == 'amplitude_damping':
                err = amplitude_damping_error(param)
                nm.add_all_qubit_quantum_error(err, _1q_gates)
            elif ntype == 'phase_flip':
                err = phase_damping_error(param)
                nm.add_all_qubit_quantum_error(err, _1q_gates)
            else:
                print(f"?UNKNOWN NOISE TYPE: {ntype}")
                print("  Types: depolarizing, amplitude_damping, phase_flip")
                return
            self._noise_model = nm
            print(f"NOISE {ntype} p={param}")
        except ImportError:
            print("?Noise model requires qiskit-aer noise module")

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
            print("?NO CIRCUIT — RUN first")
            return
        ops = {}
        for inst in self.last_circuit.data:
            name = inst.operation.name
            ops[name] = ops.get(name, 0) + 1
        print(f"\n  Circuit: {self.last_circuit.num_qubits} qubits, "
              f"depth {self.last_circuit.depth()}, {self.last_circuit.size()} gates")
        for name, count in sorted(ops.items(), key=lambda x: -x[1]):
            bar = '█' * min(count, 40)
            print(f"    {name:>10}  {count:>4}  {bar}")
        print()

    # cmd_include, cmd_sweep provided by FileIOMixin / SweepMixin.

    # LOCC commands (cmd_locc, cmd_send, cmd_share) provided by LOCCMixin.

    # cmd_bench, cmd_ram, cmd_dir provided by AnalysisMixin / FileIOMixin.

    def cmd_clear(self, rest: str) -> None:
        """CLEAR var — remove a variable. CLEAR ARRAYS — clear all arrays."""
        name = rest.strip()
        if not name:
            print("?USAGE: CLEAR <var> or CLEAR ARRAYS")
            return
        if name.upper() == 'ARRAYS':
            self.arrays.clear()
            print("ARRAYS CLEARED")
        elif name in self.variables:
            del self.variables[name]
            print(f"CLEARED {name}")
        elif name in self.arrays:
            del self.arrays[name]
            print(f"CLEARED array {name}")
        else:
            print(f"?{name} NOT FOUND")

    def cmd_undo(self) -> None:
        if not self._undo_stack:
            print("NOTHING TO UNDO")
            return
        self.program = self._undo_stack.pop()
        self._parsed = {num: parse_stmt(s) for num, s in self.program.items()}
        print(f"UNDO ({len(self.program)} lines)")

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
        print(f"        ALL COMMANDS: {', '.join(all_cmds)}")
        print(f"        ALL GATES: {', '.join(all_gates)}")

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

        Quantum BASIC v0.3.0
        Python {platform.python_version()} | Qiskit {qver} | {ram_str}{gpu_str}
        {self.num_qubits} qubits | {self.shots} shots | max ~{max_q} qubits
        Type HELP for commands, DEMO LIST for demos.
        """))

