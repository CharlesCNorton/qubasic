"""QUBASIC executor mixin — circuit building and line execution."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit, transpile

from qubasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    ExecResult,
    RE_REG_INDEX, RE_AT_REG_LINE,
    RE_CTRL, RE_INV,
    RE_SYNDROME,
)
from qubasic_core.expression import ExpressionMixin
from qubasic_core.errors import QBasicBuildError, QBasicRangeError

if TYPE_CHECKING:
    from qubasic_core.exec_context import ExecContext


def _feedforward_test(condition: str, names) -> tuple[str, int] | None:
    """(bit name, value it must read) when an IF condition tests one
    measurement bit in a form a dynamic circuit can branch on, else None."""
    s = condition.strip()
    m = re.match(r'NOT\s+(\w+)$', s, re.IGNORECASE)
    if m and m.group(1) in names:
        return m.group(1), 0
    if re.match(r'\w+$', s) and s in names:
        return s, 1
    m = re.match(r'(\w+)\s*(==|=|!=|<>)\s*([01])$', s)
    if m and m.group(1) in names:
        val = int(m.group(3))
        return m.group(1), (val if m.group(2) in ('==', '=') else 1 - val)
    return None


class ExecutorMixin:
    """Circuit building and per-line execution for QBasicTerminal.

    Provides the core execution pipeline: circuit compilation from stored
    programs, single-line execution, gate tokenization, qubit resolution,
    subroutine expansion, and immediate-mode gate application.

    All methods access shared terminal state (variables, arrays, subroutines,
    registers, etc.) through ``self``.
    """

    # Names reserved from variable substitution -- derived from source tables
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

    # ── Circuit Building ──────────────────────────────────────────────

    def _program_has_measure(self, sorted_lines) -> bool:
        """True if any reachable statement measures.

        The plain per-line scan misses a MEASURE that only appears inside a DEF
        subroutine body or an IF/THEN-ELSE clause, which left the run on the
        no-measure path with no counts. This recursion-guarded walk follows
        colon compounds, IF clauses, and subroutine bodies so the shots path
        fires whenever a measurement is actually reachable.
        """
        from qubasic_core.statements import MeasureStmt, CompoundStmt, IfThenStmt
        from qubasic_core.parser import parse_stmt

        def scan(text: str, seen: frozenset) -> bool:
            p = parse_stmt(text)
            if isinstance(p, MeasureStmt):
                return True
            if isinstance(p, CompoundStmt):
                return any(scan(part, seen) for part in p.parts)
            if isinstance(p, IfThenStmt):
                if p.then_clause and scan(p.then_clause, seen):
                    return True
                if p.else_clause and scan(p.else_clause, seen):
                    return True
                return False
            words = text.strip().split()
            if words:
                word = words[0].upper().split('(')[0]
                if word in self.subroutines and word not in seen:
                    sub = self.subroutines[word]
                    body = sub['body'] if isinstance(sub, dict) else sub
                    return any(scan(b, seen | {word}) for b in body)
            return False

        return any(scan(self.program[ln], frozenset()) for ln in sorted_lines)

    def _prepare_run(self, qc) -> None:
        """Reset per-run state and emit pending state preparation into qc."""
        # Fresh mid-circuit measurement registry for this build (dynamic IF).
        self._classical_bits: dict[str, Any] = {}
        # Partial-measurement subset (None = measure all at the end).
        self._measure_subset: list[int] | None = None
        # Density matrix (and circuit) captured by a SET_DENSITY run; the
        # matrix is solved lazily by DENSITY from the stored circuit.
        self._last_density = None
        self._last_density_qc = None
        # Apply any qubit state preparation requested via POKE to $0100.
        if getattr(self, '_poke_state_prep', None):
            self._emit_poke_state_prep(qc)
        # Apply a pending immediate SET_STATE so it persists into this RUN.
        _pend = getattr(self, '_pending_set_state', None)
        if _pend is not None and len(_pend) == 2 ** self.num_qubits:
            from qiskit.quantum_info import Statevector
            from qiskit_aer.library import SetStatevector
            qc.append(SetStatevector(Statevector(_pend)), list(range(self.num_qubits)))
        # Inject a pending mixed state (density matrix) — needs density_matrix method.
        _pend_d = getattr(self, '_pending_set_density', None)
        if _pend_d is not None and _pend_d.shape[0] == 2 ** self.num_qubits:
            from qiskit.quantum_info import DensityMatrix
            from qiskit_aer.library import SetDensityMatrix
            qc.append(SetDensityMatrix(DensityMatrix(_pend_d)), list(range(self.num_qubits)))
        self._on_measure_fired = False
        # Error trapping arms when this run executes ON ERROR GOTO, never from
        # a previous run's state.
        self._error_target = None
        self._in_error_handler = False

    def build_circuit(self, backend=None, *, start_ip: int = 0,
                      stop_ip: int | None = None) -> tuple['QuantumCircuit', bool]:
        """Compile program lines into a QuantumCircuit. Returns (circuit, has_measure).

        With a live backend (see qubasic_core.live) the lines execute against
        its per-shot state instead, from line index start_ip up to stop_ip."""
        from qubasic_core.exec_context import ExecContext
        from qubasic_core.scope import Scope
        from qubasic_core.backend import QiskitBackend

        live = backend is not None
        qc = backend.qc if live else QuantumCircuit(self.num_qubits)
        if start_ip == 0:
            self._prepare_run(qc)
        if backend is None:
            backend = QiskitBackend(qc, self._apply_gate)
        ctx = ExecContext(
            sorted_lines=sorted(self.program.keys()),
            ip=start_ip,
            run_vars=Scope(self.variables),
            max_iterations=self._max_iterations,
            qc=qc,
            backend=backend,
        )
        # Reachability scan catches MEASURE inside subroutines and IF clauses,
        # not just literal top-level MEASURE lines.
        has_measure = False if live else self._program_has_measure(ctx.sorted_lines)
        # The circuit under construction, so PRINT STATE / QUBIT(q) inside a
        # program can read the state at that line (a live run reads its own).
        self._build_qc = None if live else qc
        try:
            self._build_loop(ctx, qc, stop_ip)
        finally:
            self._build_qc = None
        return qc, has_measure or bool(self._build_saw_measure)

    def _build_loop(self, ctx, qc, stop_ip: int | None = None) -> None:
        """Execute program lines into qc (the body of build_circuit)."""
        prev, self._strict_names = self._strict_names, False
        try:
            self._build_lines(ctx, qc, stop_ip)
        finally:
            self._strict_names = prev

    def _build_lines(self, ctx, qc, stop_ip: int | None) -> None:
        from qubasic_core.statements import MeasureStmt, CompoundStmt
        handler_lines_run = 0
        self._build_saw_measure = False
        end = len(ctx.sorted_lines) if stop_ip is None else stop_ip
        while ctx.ip < end:
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            line_num = ctx.sorted_lines[ctx.ip]
            stmt = self.program[line_num].strip()
            parsed = self._get_parsed(line_num)

            # Per-activation budget for ON ERROR handlers (RESUME resets it via
            # _in_error_handler), so a handler that never resumes cannot spin.
            if self._in_error_handler:
                handler_lines_run += 1
                if handler_lines_run > 200:
                    self.io.writeln("?ERROR HANDLER LIMIT (200 lines)")
                    break
            else:
                handler_lines_run = 0

            # Debug instrumentation — each is a no-op unless explicitly enabled.
            if self._trace_mode:
                self._trace_line(line_num)
            if self._breakpoints and self._check_breakpoint(line_num, ctx.sorted_lines, ctx.ip):
                break
            if self._on_timer_target is not None:
                _timer_tgt = self._check_timer_callback(ctx.sorted_lines, ctx.ip)
                if _timer_tgt is not None:
                    ctx.ip = _timer_tgt
                    continue

            if isinstance(parsed, MeasureStmt):
                self._build_saw_measure = True
                if parsed.qubits:
                    self._measure_subset = [
                        self._resolve_qubit(q)
                        for q in parsed.qubits.replace(',', ' ').split()]
                if self._on_measure_target is not None and not self._on_measure_fired:
                    self._on_measure_fired = True
                    self._gosub_stack.append(ctx.ip + 1)
                    for _i, _ln in enumerate(ctx.sorted_lines):
                        if _ln == self._on_measure_target:
                            ctx.ip = _i
                            break
                    else:
                        ctx.ip += 1
                    continue
                ctx.ip += 1
                continue
            if isinstance(parsed, CompoundStmt):
                for part in parsed.parts:
                    if part.strip().upper() == 'MEASURE':
                        self._build_saw_measure = True

            self._profile_line_start(line_num)
            gates_before = qc.size()
            try:
                result = self._exec_line(stmt, parsed=parsed, ctx=ctx)
            except QBasicBuildError as e:
                if self._live_needed:
                    raise
                _handler_ip = self._handle_error(e, line_num, ctx.sorted_lines)
                if _handler_ip is None:
                    raise QBasicBuildError(
                        f"LINE {line_num}: {e}"
                    ) from None
                ctx.ip = _handler_ip
                continue
            except Exception as e:
                # A measurement bit read as a number makes the run go live;
                # no ON ERROR handler may take that.
                if self._live_needed:
                    raise
                _handler_ip = self._handle_error(e, line_num, ctx.sorted_lines)
                if _handler_ip is None:
                    raise RuntimeError(f"LINE {line_num}: {e}") from None
                ctx.ip = _handler_ip
                continue
            finally:
                self._profile_line_end(line_num, qc.size() - gates_before)

            if result is ExecResult.END:
                break
            self._advance(ctx, result)

    @staticmethod
    def _advance(ctx, result) -> None:
        """Move ctx past a line: to a jump target (and the statement within
        it, for a ResumePoint) or to the next line."""
        from qubasic_core.control_flow import ResumePoint
        if isinstance(result, int):
            ctx.ip = int(result)
            ctx.resume = result if isinstance(result, ResumePoint) else None
        else:
            ctx.ip += 1
            ctx.resume = None

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
        from qubasic_core.statements import (
            BarrierStmt, RemStmt, MeasureStmt, EndStmt, ReturnStmt,
            CompoundStmt, AtRegStmt, GotoStmt, GosubStmt,
            ForStmt, NextStmt, WhileStmt, WendStmt, IfThenStmt,
            LetStmt, LetArrayStmt, PrintStmt,
            GateStmt, RawStmt,
        )

        # 1. Typed fast-path: terminals that don't need control-flow dispatch
        if parsed is None:
            from qubasic_core.parser import parse_stmt
            parsed = parse_stmt(stmt)

        def _part(sub):
            return self._exec_line(sub, qc=qc, loop_stack=loop_stack,
                                   sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)

        # Continuing inside this line's compound (or one of its IF clauses).
        if ctx is not None and ctx.resume is not None:
            rp, ctx.resume = ctx.resume, None
            if int(rp) == ip:
                resumed = self._resume_compound(rp, parsed, loop_stack, ip, _part)
                if resumed is not None:
                    return resumed
        if isinstance(parsed, CompoundStmt):
            # A line run by an executor loop (ctx) is a whole line; otherwise
            # the compound is an IF clause.
            frame = ('line', ip) if ctx is not None else ('clause', ip, stmt)
            return self._run_compound(list(parsed.parts), _part, loop_stack, ip, frame)
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
        if isinstance(parsed, AtRegStmt) and not self.locc_mode:
            raise ValueError("@register syntax requires LOCC mode (try: LOCC <n1> <n2>)")

        # 1b. Dynamic circuit: IF on a mid-circuit measurement bit (set by MEAS)
        #     compiles to a Qiskit if_test, so the conditional gate is applied at
        #     simulation time based on the real measured value (feedforward).
        if (isinstance(parsed, IfThenStmt) and qc is not None
                and hasattr(qc, 'if_test') and getattr(self, '_classical_bits', None)):
            tgt = self._classical_if_target(parsed.condition)
            if tgt is not None:
                self._exec_dynamic_if(qc, tgt, parsed, loop_stack, sorted_lines, ip, run_vars)
                return ExecResult.ADVANCE

        # 2. Delegate to unified control-flow dispatch (handles GOTO, GOSUB,
        #    FOR/NEXT, WHILE/WEND, IF/THEN, LET, PRINT, DATA/READ, ON GOTO,
        #    SELECT CASE, DO/LOOP, EXIT, SUB/FUNCTION, ON ERROR, etc.)
        handled, result = self._exec_control_flow(
            stmt, loop_stack, sorted_lines, ip, run_vars,
            lambda s, ls, sl, i, rv: self._exec_line(
                s, qc=qc, loop_stack=ls, sorted_lines=sl, ip=i, run_vars=rv),
            parsed=parsed)
        if handled:
            return result

        # 3. Remaining statement handlers (not in control-flow dispatch)
        from qubasic_core.statements import RestoreStmt
        if isinstance(parsed, RestoreStmt):
            self._data_ptr = 0   # reset the DATA read pointer (was a no-op)
            return ExecResult.ADVANCE

        # Multi-line IF block markers — no-ops during execution
        upper = stmt.strip().upper()
        if upper in ('END IF', 'ELSE'):
            return ExecResult.ADVANCE

        # 3. Statement handlers
        _backend = ctx.backend if ctx else None
        if self._try_stmt_handlers(stmt, qc, run_vars, backend=_backend):
            return ExecResult.ADVANCE

        # 5. Gate application
        _backend = ctx.backend if ctx else None

        # Fast path: GateStmt already parsed by the parser
        if isinstance(parsed, GateStmt):
            info = self._gate_info(parsed.name)
            if info is not None:
                n_params, n_qubits = info
                args = list(parsed.args)
                params = [self._eval_operand(a, run_vars) for a in args[:n_params]]
                qubits = [self._resolve_qubit(a) for a in args[n_params:n_params + n_qubits]]
                try:
                    if _backend:
                        _backend.apply_gate(parsed.name, tuple(params), qubits)
                    else:
                        self._apply_gate(qc, parsed.name, params, qubits)
                except Exception as _gate_err:
                    if 'duplicate' in str(_gate_err).lower():
                        raise QBasicBuildError(
                            f"duplicate qubit arguments in {parsed.name}"
                        ) from None
                    raise
                return ExecResult.ADVANCE
            # Fall through to _apply_gate_str for custom gates not in GATE_TABLE

        # A REPL/config command (QUBITS, SHOTS, METHOD, LOCC, ...) used as a
        # numbered program line reaches here only because it is neither a gate
        # nor a statement. Explain it instead of failing with "UNKNOWN GATE".
        # SEND/SHARE are LOCC statements (handled on the LOCC path), so leave
        # them to their own error.
        _first = stmt.split(None, 1)[0].upper() if stmt.split() else ''
        if ((_first in self._CMD_WITH_ARG or _first in self._CMD_NO_ARG)
                and _first not in ('SEND', 'SHARE')):
            raise QBasicBuildError(
                f"{_first} is a configuration/session command, not a program "
                f"statement; run it without a line number (before RUN), e.g. "
                f"QUBITS 3 / SHOTS 1024 / METHOD statevector / LOCC 2 2")

        # Slow path: subroutine expansion + gate dispatch
        expanded = self._expand_statement(stmt)
        for gate_str in expanded:
            self._apply_gate_str(gate_str, qc, backend=_backend)

        return ExecResult.ADVANCE

    def _classical_if_target(self, condition: str):
        """If `condition` tests a mid-circuit measurement bit, return
        (classical_register, target_value); else None.

        Handled forms: ``c``, ``NOT c``, ``c = 0|1``, ``c <> 0|1`` (and the
        ``==`` / ``!=`` spellings).
        """
        cb = getattr(self, '_classical_bits', None)
        if not cb:
            return None
        hit = _feedforward_test(condition, cb)
        if hit is None:
            return None
        name, val = hit
        return cb[name], val

    def _exec_dynamic_if(self, qc, tgt, parsed, loop_stack, sorted_lines, ip, run_vars) -> None:
        """Emit a Qiskit if_test (with optional else) for an IF on a measured bit.

        The then/else clauses are executed inside the if_test context so their
        gates attach to the conditional block and run at simulation time based on
        the real measurement outcome.
        """
        creg, val = tgt

        def _run(clause):
            if clause:
                self._exec_line(clause, qc=qc, loop_stack=loop_stack,
                                sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)

        if parsed.else_clause:
            with qc.if_test((creg, val)) as else_block:
                _run(parsed.then_clause)
            with else_block:
                _run(parsed.else_clause)
        else:
            with qc.if_test((creg, val)):
                _run(parsed.then_clause)

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

    @staticmethod
    def _split_colon_stmts(stmt: str) -> list[str]:
        """Split colon-separated statements, inheriting @register prefixes."""
        from qubasic_core.parser import _split_colon_stmts
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

        def _sub_segment(seg: str) -> str:
            # Tokenize: split on word boundaries, preserving delimiters
            tokens = re.split(r'(\b\w+\b)', seg)
            for i, tok in enumerate(tokens):
                if not tok or not tok[0].isalpha():
                    continue
                if tok in protected or tok.upper() in protected or tok.lower() in protected:
                    continue
                if tok in merged:
                    # Don't expand record bases (dicts) — leave p in p.x intact
                    # so the expression evaluator can resolve the field.
                    if isinstance(merged[tok], dict):
                        continue
                    tokens[i] = str(merged[tok])
            return ''.join(tokens)

        # Substitute only outside quoted string literals: a variable named in a
        # PRINT label (PRINT "S ="; S) must survive verbatim inside the quotes.
        out: list[str] = []
        i, n = 0, len(stmt)
        while i < n:
            ch = stmt[i]
            if ch in ('"', "'"):
                j = i + 1
                while j < n and stmt[j] != ch:
                    j += 1
                out.append(stmt[i:j + 1])           # quoted span, verbatim
                i = j + 1
            else:
                j = i
                while j < n and stmt[j] not in ('"', "'"):
                    j += 1
                out.append(_sub_segment(stmt[i:j]))  # unquoted span
                i = j
        return ''.join(out)

    def _expand_statement(self, stmt, _call_stack: set[str] | None = None):
        """Expand subroutines. Returns list of gate strings.

        Uses explicit call-stack tracking to detect recursion instead of
        an arbitrary depth counter.
        """
        if _call_stack is None:
            _call_stack = set()

        # Handle parenthesized subroutine calls: NAME(arg1, arg2) -> NAME arg1, arg2
        m_call = re.match(r'(\w+)\(([^)]*)\)', stmt)
        if m_call:
            call_name = m_call.group(1).upper()
            if call_name in self.subroutines:
                call_args = m_call.group(2)
                stmt = f"{call_name} {call_args}"

        parts = stmt.split()
        word = parts[0].upper() if parts else ''

        if word not in self.subroutines:
            return [stmt]

        if word in _call_stack:
            raise RuntimeError(f"RECURSION: {word} calls itself")
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
                # Skip register[index] notation -- it resolves its own offset
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
        if _call_stack is None:
            _call_stack = set()
        word = stmt.split()[0].upper() if stmt.split() else ''
        # Also check for parenthesized call syntax: NAME(args)
        m_paren = re.match(r'(\w+)\(', stmt)
        if m_paren:
            paren_name = m_paren.group(1).upper()
            if paren_name in self.subroutines:
                word = paren_name
        if word in self.subroutines:
            if word in _call_stack:
                raise QBasicBuildError(f"RECURSION: {word} calls itself")
            for sub_stmt in self._expand_statement(stmt, _call_stack):
                self._apply_gate_str(sub_stmt, qc, _call_stack | {word}, backend=backend)
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

        # CTRL <gate> [params,] control, target(s) -- one control on any gate
        m_ctrl = RE_CTRL.match(stmt)
        if m_ctrl:
            canonical, params, qubits = self._modifier_operands(
                'CTRL', m_ctrl.group(1), m_ctrl.group(2))
            gate = self._gate_object(canonical, params).control(1)
            if backend and hasattr(backend, 'append_controlled'):
                backend.append_controlled(gate, qubits)
            else:
                qc.append(gate, qubits)
            return

        # INV <gate> [params,] qubit(s) -- inverse (dagger) of one gate
        m_inv = RE_INV.match(stmt)
        if m_inv:
            canonical, params, qubits = self._modifier_operands(
                'INV', m_inv.group(1), m_inv.group(2))
            gate = self._gate_object(canonical, params).inverse()
            if backend and hasattr(backend, 'append_controlled'):
                backend.append_controlled(gate, qubits)
            else:
                qc.append(gate, qubits)
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

        params = [self._eval_operand(a) for a in args[:n_params]]
        qubits = [self._resolve_qubit(a) for a in args[n_params:n_params+n_qubits]]

        # Apply gate through backend when available
        try:
            if backend:
                backend.apply_gate(gate_name, tuple(params), qubits)
            else:
                self._apply_gate(qc, gate_name, params, qubits)
        except Exception as _gate_err:
            if 'duplicate' in str(_gate_err).lower():
                raise QBasicBuildError(
                    f"duplicate qubit arguments in {gate_name}"
                ) from None
            raise

    def _modifier_operands(self, keyword: str, gate_name: str, arg_text: str,
                           limit: int | None = None) -> tuple[str, list, list[int]]:
        """Operands of CTRL/INV: (gate, params, qubits). For CTRL the control
        qubit comes first in `qubits`, then the gate's own qubits."""
        canonical = GATE_ALIASES.get(gate_name.upper(), gate_name.upper())
        info = self._gate_info(canonical)
        if info is None:
            raise ValueError(f"{keyword} {gate_name.upper()}: gate not found")
        n_params, n_q = info
        args = self._tokenize_gate(f"{canonical} {arg_text}")[1:]
        controls = 1 if keyword == 'CTRL' else 0
        if len(args) != n_params + controls + n_q:
            ctl = "1 control and " if controls else ""
            raise ValueError(
                f"{keyword} {canonical} needs {n_params} parameter(s), {ctl}"
                f"{n_q} target qubit(s); got {len(args)} argument(s)")
        params = [self._eval_operand(a) for a in args[:n_params]]
        qubits = [self._resolve_qubit(a, n_qubits=limit) for a in args[n_params:]]
        if len(set(qubits)) != len(qubits):
            raise QBasicBuildError(f"duplicate qubit arguments in {keyword} {canonical}")
        return canonical, params, qubits

    def _gate_object(self, canonical: str, params: list):
        """A Qiskit gate instance for a built-in or UNITARY-defined gate."""
        if canonical in self._custom_gates:
            from qiskit.circuit.library import UnitaryGate
            return UnitaryGate(self._custom_gates[canonical], label=canonical)
        from qiskit.circuit.library import get_standard_gate_name_mapping
        lib = get_standard_gate_name_mapping().get(canonical.lower())
        if lib is None:
            raise ValueError(f"GATE {canonical} NOT IMPLEMENTED")
        return type(lib)(*params)

    def _tokenize_gate(self, stmt: str) -> list[str]:
        """Split gate statement into tokens, handling commas and register notation.

        When arguments are comma-separated, splits on commas only so that
        compound expressions like ``I + 1`` are preserved as single tokens.
        Falls back to whitespace splitting when no commas are present
        (legacy format like ``CX 0 1``).
        """
        stmt = RE_REG_INDEX.sub(r'\1[\2]', stmt)
        parts = stmt.strip().split(None, 1)
        if len(parts) < 2:
            return [parts[0]] if parts else []
        gate = parts[0]
        args_str = parts[1]
        if ',' in args_str:
            # Comma-separated: split on commas, preserving expressions
            args = [a.strip() for a in args_str.split(',') if a.strip()]
        else:
            # Space-separated: split on whitespace (legacy)
            args = [a.strip() for a in args_str.split() if a.strip()]
        return [gate] + args

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
                raise QBasicRangeError(f"UNKNOWN REGISTER: {name}")
            start, size = self.registers[name]
            if idx >= size:
                raise QBasicRangeError(f"{name}[{idx}] OUT OF RANGE (size={size})")
            q = start + idx
        else:
            try:
                q = int(self._eval_operand(arg))
            except Exception:
                raise ValueError(f"CANNOT RESOLVE QUBIT: {arg}")
        if q < 0 or q >= limit:
            raise QBasicRangeError(f"QUBIT {q} OUT OF RANGE (0-{limit-1})")
        return q

    def _apply_gate(self, qc, gate_name, params, qubits):
        """Apply a gate to the quantum circuit."""
        n = qc.num_qubits
        for q in qubits:
            if q < 0 or q >= n:
                raise QBasicRangeError(
                    f"QUBIT {q} OUT OF RANGE (0-{n-1}) in {gate_name}"
                )
        method_name = self._GATE_DISPATCH.get(gate_name)
        if method_name:
            method = getattr(qc, method_name)
            method(*params, *qubits)
        elif gate_name in self._custom_gates:
            from qiskit.circuit.library import UnitaryGate
            qc.append(UnitaryGate(self._custom_gates[gate_name]), qubits)
        else:
            raise ValueError(f"GATE {gate_name} NOT IMPLEMENTED")

    def run_immediate(self, line: str) -> None:
        """Execute a single gate command immediately.

        Uses the same _exec_line pipeline as cmd_run for consistency.
        """
        # In LOCC mode, handle @register prefix via the numpy engine. Split
        # colon-compound @REG lines so each gate is applied (a single @REG line
        # is just the one-part case); a bare @REG line keeps prior behavior.
        if self.locc_mode and self.locc and line.strip().startswith('@'):
            parts = self._split_colon_stmts(line) if ':' in line else [line.strip()]
            for part in parts:
                m = RE_AT_REG_LINE.match(part)
                if not m:
                    self.io.writeln(f"?BAD @register statement: {part}")
                    continue
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
        from qubasic_core.exec_context import ExecContext
        qc = QuantumCircuit(self.num_qubits)
        imm_ctx = ExecContext(sorted_lines=[0], ip=0,
                              run_vars=dict(self.variables), qc=qc)
        self._exec_line(line, ctx=imm_ctx)
        # A classical statement typed at the prompt (PRINT, MEASURE, an IF whose
        # clause printed, ...) adds no circuit operations and has already emitted
        # its own output via _exec_line. Simulating an empty circuit just to print
        # an unchanged |psi> would be spurious, so skip the statevector dump.
        if qc.size() == 0:
            return
        from qubasic_core import capacity as cap
        peak = cap.aer_peak_bytes(self.num_qubits, 'statevector', self.sim_precision, 'end')
        if not cap.fits(peak):
            self.last_circuit = qc
            self.io.writeln(f"  (applied; a {self.num_qubits}-qubit state needs "
                            f"~{cap.fmt_gb(peak)}, so it is not shown)")
            return
        qc.save_statevector(label='qb_state')
        backend = self._make_backend('statevector')
        result = backend.run(transpile(qc, backend), shots=1).result()
        sv = np.asarray(result.data()['qb_state'].data)
        self.last_sv = sv
        self.last_circuit = qc
        self._print_sv_compact(sv)
