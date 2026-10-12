"""QUBASIC keyword statements: measurement, arrays, input, memory, file I/O, SAVE_*, SET_STATE."""

from __future__ import annotations

import sys
import re
import time
import numpy as np
from typing import Any

from qubasic_core.engine import (
    RE_MEAS, RE_RESET, RE_UNITARY, RE_MEASURE_BASIS, RE_POKE, RE_OPEN,
    RE_LET_STR, RE_GET,
)
from qubasic_core.basic_format import number_text
from qubasic_core.variables import is_string_kind
from qubasic_core.host import HostState

_RE_INPUT_STMT = re.compile(r'INPUT\s*(;)?\s*(?:"([^"]*)"\s*([;,]))?\s*(.*)$', re.IGNORECASE)
_RE_INPUT_NUMBER = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?')


def _input_fields(line: str) -> list[str]:
    """The comma-separated fields of an INPUT line; a field in double quotes
    may hold commas. Surrounding blanks are dropped from unquoted fields."""
    fields: list[str] = []
    buf: list[str] = []
    quoted = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"' and not ''.join(buf).strip():
            j = line.find('"', i + 1)
            if j < 0:
                buf.append(line[i + 1:])
                break
            buf = [line[i + 1:j]]
            quoted = True
            i = j + 1
            continue
        if ch == ',':
            fields.append(''.join(buf) if quoted else ''.join(buf).strip())
            buf, quoted = [], False
        elif not quoted:
            buf.append(ch)
        i += 1
    fields.append(''.join(buf) if quoted else ''.join(buf).strip())
    return fields


def _parse_input_number(text: str):
    """A typed number (int when it has no fraction or exponent), &H/&O/&B
    included, or None."""
    m = re.fullmatch(r'&([HhOoBb])([0-9A-Fa-f]+)', text)
    if m:
        try:
            return int(m.group(2), {'H': 16, 'O': 8, 'B': 2}[m.group(1).upper()])
        except ValueError:
            return None
    if not _RE_INPUT_NUMBER.fullmatch(text):
        return None
    if re.fullmatch(r'[+-]?\d+', text):
        return int(text)
    return float(text.replace('d', 'e').replace('D', 'E'))


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

def _named_state_fits(name: str, n_qubits: int) -> bool:
    """Whether a named SET_STATE target fits the current qubit count."""
    dim = 2 ** n_qubits
    need = {'|1>': 2, '|+>': 2, '|->': 2, '|GHZ3>': 8, '|GHZ4>': 16,
            '|W>': 8, '|W3>': 8}
    return dim >= need.get(name.upper(), 1)


class StatementMixin(HostState):
    """Keyword statements for QBasicTerminal, dispatched by first word through
    _STMT_KEYWORDS: MEAS, RESET, MEASURE_X/Y/Z, SYNDROME, UNITARY, DIM, REDIM,
    ERASE, GET, INPUT, POKE, SYS, file I/O, PRINT USING, SAVE_*, SET_STATE and
    the algorithm primitives."""

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
            if getattr(self.io, 'replay', False):
                ch = self.io.read_line('')[:1]      # a later live shot replays
            elif sys.stdin.isatty():
                if sys.platform == 'win32':
                    import msvcrt
                    ch = msvcrt.getwch()
                else:
                    import tty, termios
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setraw(fd)
                        ch = sys.stdin.read(1)
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
        """INPUT [;]["prompt"{;|,}] var[, var ...] — read one line of
        comma-separated values. A prompt followed by ; shows "? " after it, by
        , shows nothing more; with no prompt the cursor shows "? ". A line with
        the wrong number of values or a non-number for a numeric variable asks
        again ("?Redo from start")."""
        if re.match(r'INPUT\s*#', stmt, re.IGNORECASE):
            return False
        m = _RE_INPUT_STMT.match(stmt.strip())
        if not m or not m.group(4).strip():
            return False
        if m.group(2) is not None:
            prompt = m.group(2) + ('? ' if m.group(3) == ';' else '')
        else:
            prompt = '? '
        targets = [self._lvalue(t, run_vars) for t in self._split_arg_list(m.group(4))]
        for _attempt in range(3):
            try:
                line = self.io.read_line(prompt)
            except (EOFError, KeyboardInterrupt):
                break
            values = self._input_values(line, targets)
            if values is not None:
                for lv, value in zip(targets, values):
                    self._write_lvalue(lv, value, run_vars)
                return True
            self.io.writeln("?Redo from start")
        for lv in targets:
            self._write_lvalue(lv, self._default_of_kind(self._lvalue_kind(lv)), run_vars)
        return True

    def _input_values(self, line: str, targets: list) -> list | None:
        """Values for INPUT's targets from one typed line, or None when the
        line does not fit them."""
        fields = _input_fields(line)
        if len(fields) != len(targets):
            return None
        values: list[Any] = []
        for lv, field in zip(targets, fields):
            if is_string_kind(self._lvalue_kind(lv)):
                values.append(field)
                continue
            text = field.strip()
            if not text:
                values.append(0)
                continue
            num = _parse_input_number(text)
            if num is None:
                return None
            values.append(num)
        return values

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

    # Keyword statements: the statement's first word selects its handlers,
    # each (self, stmt, qc, run_vars, backend) -> handled.
    _STMT_KEYWORDS: dict[str, tuple[Any, ...]] = {
        'MEAS': (lambda s, st, qc, rv, b: s._try_exec_meas(st, qc, rv, backend=b),),
        'RESET': (lambda s, st, qc, rv, b: s._try_exec_reset(st, qc, rv, backend=b),),
        **dict.fromkeys(('MEASURE_X', 'MEASURE_Y', 'MEASURE_Z'), (
            lambda s, st, qc, rv, b: s._try_exec_measure_basis(st, qc, rv, backend=b),)),
        'SYNDROME': (lambda s, st, qc, rv, b: s._try_exec_syndrome(st, qc, rv, backend=b),),
        **dict.fromkeys(('QFT', 'IQFT'), (
            lambda s, st, qc, rv, b: s._try_exec_qft(st, qc, rv, backend=b),)),
        **dict.fromkeys(('DIFFUSE', 'DIFFUSION'), (
            lambda s, st, qc, rv, b: s._try_exec_diffuse(st, qc, rv, backend=b),)),
        **dict.fromkeys(('MCX', 'MCZ', 'MCP'), (
            lambda s, st, qc, rv, b: s._try_exec_mcgate(st, qc, rv, backend=b),)),
        'QADDC': (lambda s, st, qc, rv, b: s._try_exec_qaddc(st, qc, rv, backend=b),),
        'QADD': (lambda s, st, qc, rv, b: s._try_exec_qadd(st, qc, rv, backend=b),),
        'QPE': (lambda s, st, qc, rv, b: s._try_exec_qpe(st, qc, rv, backend=b),),
        'EVOLVE': (lambda s, st, qc, rv, b: s._try_exec_evolve(st, qc, rv, backend=b),),
        'APPLYCHANNEL': (lambda s, st, qc, rv, b: s._try_exec_applychannel(st, qc, rv, backend=b),),
        'AMPLIFY': (lambda s, st, qc, rv, b: s._try_exec_amplify(st, qc, rv, backend=b),),
        'GRAPHSTATE': (lambda s, st, qc, rv, b: s._try_exec_graphstate(st, qc, rv, backend=b),),
        'FEATUREMAP': (lambda s, st, qc, rv, b: s._try_exec_featuremap(st, qc, rv, backend=b),),
        'UNITARY': (lambda s, st, qc, rv, b: s._try_exec_unitary(st),),
        'DIM': (lambda s, st, qc, rv, b: s._try_exec_dim(st, rv),),
        'REDIM': (lambda s, st, qc, rv, b: s._try_exec_redim(st, rv),),
        'ERASE': (lambda s, st, qc, rv, b: s._try_exec_erase(st),),
        'CONST': (lambda s, st, qc, rv, b: s._stmt_const(st, rv),),
        **dict.fromkeys(('DEFINT', 'DEFLNG', 'DEFSNG', 'DEFDBL', 'DEFSTR'), (
            lambda s, st, qc, rv, b: s._stmt_deftype(st),)),
        'DECLARE': (lambda s, st, qc, rv, b: s._stmt_accept(st, 'DECLARE'),),
        'WIDTH': (lambda s, st, qc, rv, b: s._stmt_accept(st, 'WIDTH'),),
        'RANDOMIZE': (lambda s, st, qc, rv, b: s._stmt_randomize(st, rv),),
        'WRITE': (lambda s, st, qc, rv, b: s._stmt_write(st, rv),),
        'CLS': (lambda s, st, qc, rv, b: s._stmt_cls(st),),
        'COLOR': (lambda s, st, qc, rv, b: s._stmt_color(st, rv),),
        'LOCATE': (lambda s, st, qc, rv, b: s._stmt_locate(st, rv),),
        'BEEP': (lambda s, st, qc, rv, b: s._stmt_beep(st),),
        'SLEEP': (lambda s, st, qc, rv, b: s._stmt_sleep(st, rv),),
        'SCREEN': (lambda s, st, qc, rv, b: s._stmt_screen(st, rv),),
        'GET': (lambda s, st, qc, rv, b: s._try_exec_get(st, rv),),
        'INPUT': (lambda s, st, qc, rv, b: s._exec_input_file(st, rv),
                  lambda s, st, qc, rv, b: s._try_exec_input(st, rv)),
        'LINE': (lambda s, st, qc, rv, b: s._try_exec_line_input(st, rv),),
        'POKE': (lambda s, st, qc, rv, b: s._try_exec_poke(st, rv),),
        'SYS': (lambda s, st, qc, rv, b: s._try_exec_sys(st),),
        'PRINT': (lambda s, st, qc, rv, b: s._exec_print_file(st, rv),),
        'LPRINT': (lambda s, st, qc, rv, b: s._exec_lprint(st, rv),),
        'LET': (lambda s, st, qc, rv, b: s._try_exec_let_str(st, rv),),
        **dict.fromkeys(('OPEN', 'CLOSE'), (
            lambda s, st, qc, rv, b: s._try_exec_open_close(st),)),
        'SAVE_EXPECT': (lambda s, st, qc, rv, b: s._try_exec_save_expect(st, qc, rv),),
        'SAVE_PROBS': (lambda s, st, qc, rv, b: s._try_exec_save_probs(st, qc, rv),),
        'SAVE_AMPS': (lambda s, st, qc, rv, b: s._try_exec_save_amps(st, qc, rv),),
        'SET_STATE': (lambda s, st, qc, rv, b: s._try_exec_set_state(st, qc),),
        'APPLY_CIRCUIT': (lambda s, st, qc, rv, b: s._try_exec_apply_circuit(st, qc, backend=b),),
    }

    def _try_stmt_handlers(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        """Run the keyword statement `stmt`; False when no handler takes it."""
        words = stmt.split(None, 1)
        if not words:
            return False
        for handler in self._STMT_KEYWORDS.get(words[0].upper(), ()):
            if handler(self, stmt, qc, run_vars, backend):
                return True
        return False

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
        """LINE INPUT [;]["prompt"{;|,}] var$ — read a whole line, commas and
        quotes included. The prompt prints as written (no "?")."""
        if re.match(r'LINE\s+INPUT\s*#', stmt, re.IGNORECASE):
            return self._exec_line_input_file(stmt, run_vars)
        m = re.match(r'LINE\s+INPUT\s*(;)?\s*(?:"([^"]*)"\s*[;,])?\s*(.+)$',
                     stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        lv = self._lvalue(m.group(3), run_vars)
        try:
            val = self.io.read_line(m.group(2) or '')
        except (EOFError, KeyboardInterrupt):
            val = ''
        self._write_lvalue(lv, val, run_vars)
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
        """PRINT USING fmt$; values (see basic_format)."""
        from qubasic_core.parser import parse_stmt
        from qubasic_core.statements import PrintUsingStmt
        parsed = parse_stmt(stmt)
        if not isinstance(parsed, PrintUsingStmt):
            return False
        self._emit_using(self._using_text(parsed.fmt, parsed.values, run_vars))
        return True

    # ── WRITE and the screen statements ───────────────────────────────

    def _stmt_write(self, stmt: str, run_vars: dict) -> bool:
        """WRITE [#n,] values — comma-separated, strings in quotes, numbers
        without padding (WRITE 1, "a" gives 1,"a")."""
        m = re.match(r'WRITE\b\s*(?:#\s*(\d+)\s*,?)?\s*(.*)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        items = self._split_arg_list(m.group(2)) if m.group(2).strip() else []
        parts = []
        for item in items:
            v = self._safe_eval(item, extra_ns=run_vars)
            parts.append(f'"{v}"' if isinstance(v, str) else number_text(v))
        text = ','.join(parts)
        if m.group(1) is not None:
            return self._write_to_handle(int(m.group(1)), text)
        self.io.writeln(text)
        return True

    @staticmethod
    def _on_terminal() -> bool:
        return bool(sys.stdout and sys.stdout.isatty())

    def _stmt_cls(self, stmt: str) -> bool:
        if not re.match(r'CLS\b', stmt.strip(), re.IGNORECASE):
            return False
        if self._on_terminal() and not getattr(self.io, 'replay', False):
            self.io.write('\033[2J\033[H')
        return True

    # QBASIC colour numbers 0-15 as ANSI codes (8-15 are the bright set).
    _ANSI_FG = (30, 34, 32, 36, 31, 35, 33, 37, 90, 94, 92, 96, 91, 95, 93, 97)
    _COLOR_NAMES = {'black': 0, 'blue': 1, 'green': 2, 'cyan': 3, 'red': 4, 'magenta': 5,
                    'brown': 6, 'yellow': 14, 'white': 15, 'gray': 8, 'grey': 8}

    def _stmt_color(self, stmt: str, run_vars: dict) -> bool:
        """COLOR [fg][, bg] — QBASIC colour numbers (0-15) or names."""
        m = re.match(r'COLOR\b\s*(.*)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        args = [a.strip() for a in m.group(1).split(',')]
        codes = []
        for k, arg in enumerate(args[:2]):
            if not arg:
                continue
            n = self._COLOR_NAMES.get(arg.lower())
            if n is None:
                n = int(round(float(self._safe_eval(arg, extra_ns=run_vars))))
            if not 0 <= n <= 31:
                raise ValueError(f"ILLEGAL FUNCTION CALL: COLOR {n}")
            code = self._ANSI_FG[n % 16]
            codes.append(code if k == 0 else code + 10)
        if codes and self._on_terminal() and not getattr(self.io, 'replay', False):
            self.io.write('\033[' + ';'.join(str(c) for c in codes) + 'm')
        return True

    def _stmt_locate(self, stmt: str, run_vars: dict) -> bool:
        """LOCATE [row][, col] — move the cursor (on a terminal)."""
        m = re.match(r'LOCATE\b\s*(.*)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        args = [a.strip() for a in m.group(1).split(',')]
        nums = [int(round(float(self._safe_eval(a, extra_ns=run_vars)))) if a else None
                for a in args[:2]]
        if self._on_terminal() and not getattr(self.io, 'replay', False) and nums:
            row = nums[0] or 1
            col = (nums[1] if len(nums) > 1 else None) or 1
            self.io.write(f'\033[{row};{col}H')
        return True

    def _stmt_beep(self, stmt: str) -> bool:
        if stmt.strip().upper() != 'BEEP':
            return False
        if not getattr(self.io, 'replay', False):
            self.io.write('\a')
        return True

    def _stmt_sleep(self, stmt: str, run_vars: dict) -> bool:
        """SLEEP [seconds] — pause (later shots of a live run do not wait)."""
        m = re.match(r'SLEEP\b\s*(.*)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        if getattr(self.io, 'replay', False):
            return True
        arg = m.group(1).strip()
        secs = float(self._safe_eval(arg, extra_ns=run_vars)) if arg else 0.0
        if secs > 0:
            time.sleep(secs)
        return True

    def _stmt_screen(self, stmt: str, run_vars: dict) -> bool:
        """SCREEN n inside a program — the display mode used after RUN."""
        m = re.match(r'SCREEN\b\s*(.+)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        mode = int(round(float(self._safe_eval(m.group(1).split(',')[0], extra_ns=run_vars))))
        if not 0 <= mode <= 5:
            raise ValueError(f"ILLEGAL FUNCTION CALL: SCREEN {mode} (0-5)")
        self._screen_mode = mode
        return True

    @staticmethod
    def _stmt_accept(stmt: str, word: str) -> bool:
        """Statements that declare rather than act: DECLARE (procedures need no
        forward declaration here) and WIDTH (a terminal sets its own width)."""
        return bool(re.match(rf'{word}\b', stmt.strip(), re.IGNORECASE))

    def _inkey(self) -> str:
        """INKEY$: a key already pressed, or "" (never waits)."""
        io = self.io
        if getattr(io, 'replay', False):
            try:
                return io.read_line('')[:1]
            except EOFError:
                return ''
        ch = ''
        try:
            if sys.stdin and sys.stdin.isatty():
                if sys.platform == 'win32':
                    import msvcrt
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                else:
                    import select
                    import termios
                    import tty
                    fd = sys.stdin.fileno()
                    old = termios.tcgetattr(fd)
                    try:
                        tty.setcbreak(fd)
                        if select.select([sys.stdin], [], [], 0)[0]:
                            ch = sys.stdin.read(1)
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except (OSError, ValueError):
            ch = ''
        if hasattr(io, 'record'):
            io.record(ch)
        return ch

    def _try_exec_open_close(self, stmt: str) -> bool:
        """OPEN and CLOSE as program statements: silent, with failures raised
        as runtime errors (so ON ERROR can trap them)."""
        m = RE_OPEN.match(stmt)
        if m:
            if getattr(self.io, 'replay', False) and m.group(2).upper() != 'INPUT':
                return True     # later live shots leave the first shot's file alone
            self._open_file(stmt.strip()[4:].strip())
            return True
        if re.match(r'CLOSE\s*$', stmt.strip(), re.IGNORECASE):
            for h in list(self._file_handles):
                self._close_file(h)
            return True
        m = re.match(r'CLOSE\s+(.+)$', stmt.strip(), re.IGNORECASE)
        if m:
            for tok in m.group(1).split(','):
                tok = tok.strip().lstrip('#').strip()
                if tok:
                    self._close_file(int(tok))
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
