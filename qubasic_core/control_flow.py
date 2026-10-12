"""Control flow helpers extracted from QBasicTerminal.

Requires: TerminalProtocol (see qubasic_core.protocol).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from qubasic_core.engine import ExecResult, ExecOutcome
from qubasic_core.parser import parse_stmt
from qubasic_core.statements import (
    RemStmt, MeasureStmt, EndStmt, ReturnStmt, WendStmt,
    LetArrayStmt, LetStmt, LetStrStmt, PrintStmt, GotoStmt, GosubStmt,
    ForStmt, NextStmt, WhileStmt, DataStmt, ReadStmt, OnGotoStmt, OnGosubStmt,
    SelectCaseStmt, CaseStmt, EndSelectStmt, ElseStmt, EndIfStmt,
    DoStmt, LoopStmt, ExitStmt,
    SwapStmt, DefFnStmt, OptionBaseStmt, OptionEndianStmt,
    SubStmt, EndSubStmt, FunctionStmt, EndFunctionStmt, CallStmt,
    LocalStmt, StaticStmt, SharedStmt,
    OnErrorStmt, ResumeStmt, ErrorStmt, AssertStmt, StopStmt,
    OnMeasureStmt, OnTimerStmt, IfJumpStmt, JumpStmt, PrintUsingStmt, MidAssignStmt,
)
from qubasic_core.basic_format import print_text, format_using
from qubasic_core.host import HostState


_RE_TAB_SPC = re.compile(r'(TAB|SPC)\s*\((.+)\)$', re.IGNORECASE)


class ControlFlowMixin(HostState):
    """Mixin providing control flow helpers for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.variables,
    self.arrays, self.locc_mode, self.locc, self._gosub_stack,
    self._eval_with_vars(), self._eval_condition().
    """

    # ── Compiled block structure ────────────────────────────────────
    #
    # Statements run from a CompiledProgram (program.py): `ip` indexes a
    # statement, and the program carries the jump targets of block structure.

    @staticmethod
    def _loop_body(loop: dict[str, Any]) -> int:
        """Jump target for a loop's next pass: the statement after its opener."""
        return loop['return_ip'] + 1

    @staticmethod
    def _drop_loop(loop_stack: list[dict[str, Any]], kind: str, ip: int) -> None:
        """Remove a stale entry for the loop opening at ip (re-entered by a
        jump) along with every loop opened inside it."""
        for idx in range(len(loop_stack) - 1, -1, -1):
            e = loop_stack[idx]
            if e.get('type') == kind and e.get('return_ip') == ip:
                del loop_stack[idx:]
                return

    def _as_program(self, lines: Any) -> Any:
        """The CompiledProgram behind `lines` (a plain line list compiles
        the current program)."""
        return lines if hasattr(lines, 'skip') else self._compiled()

    def _target(self, prog: Any, table: str, ip: int, what: str) -> int:
        """A compiled jump target of statement ip (prog.skip / prog.exits)."""
        targets = getattr(self._as_program(prog), table)
        if ip not in targets:
            raise RuntimeError(f"{what}: no matching block")
        return targets[ip]

    def _jump_to_line(self, prog: Any, target: int | str, what: str) -> int:
        from qubasic_core.program import line_index
        line: int
        if isinstance(target, str):
            labels = getattr(prog, 'labels', {})
            if target.upper() not in labels:
                raise RuntimeError(f"{what} {target}: LABEL NOT FOUND")
            line = int(labels[target.upper()])
        else:
            line = target
        idx = line_index(prog, line)
        if idx is None:
            raise RuntimeError(f"{what} {target}: LINE NOT FOUND")
        return idx

    # ── Control flow helpers (decomposed from _exec_control_flow) ────
    #
    # Each _cf_* method accepts (self, stmt, parsed, ...) where parsed is
    # a required typed Stmt object.  The stmt parameter is retained in the
    # signature for compatibility with _exec_control_flow's argument
    # passing but is not used by the methods themselves.

    @staticmethod
    def _split_arg_list(s: str) -> list[str]:
        """Split a comma list at top level (outside quotes and brackets)."""
        parts: list[str] = []
        buf = ''
        depth = 0
        quote: str | None = None
        for ch in s:
            if quote:
                buf += ch
                if ch == quote:
                    quote = None
            elif ch in ('"', "'"):
                quote = ch
                buf += ch
            elif ch in '([':
                depth += 1
                buf += ch
            elif ch in ')]':
                depth = max(0, depth - 1)
                buf += ch
            elif ch == ',' and depth == 0:
                parts.append(buf)
                buf = ''
            else:
                buf += ch
        if buf.strip():
            parts.append(buf)
        return [p.strip() for p in parts if p.strip()]

    def _cf_let_array(self, stmt: str, run_vars: dict[str, Any],
                      parsed: LetArrayStmt) -> tuple[bool, ExecOutcome]:
        """LET a(i[, j ...])[.field] = expr — a typed element (or record element
        field) write (see variables.py)."""
        name = parsed.name
        self._assert_assignable(name)
        value = self._safe_eval(parsed.value_expr, extra_ns=run_vars)
        indices = [self._safe_eval(p, extra_ns=run_vars)
                   for p in self._split_arg_list(parsed.index_expr)]
        self._array_write(name, indices, value, parsed.field)
        return True, ExecResult.ADVANCE

    def _cf_let_var(self, stmt: str, run_vars: dict[str, Any],
                    parsed: LetStmt) -> tuple[bool, ExecOutcome]:
        """LET v = expr — a typed scalar or record field write."""
        self._store_var(parsed.name, self._safe_eval(parsed.expr, extra_ns=run_vars), run_vars)
        return True, ExecResult.ADVANCE

    def _cf_let_str(self, stmt: str, run_vars: dict[str, Any],
                    parsed: LetStrStmt) -> tuple[bool, ExecOutcome]:
        """LET v$ = <expr> — assign a string to a string variable."""
        self._store_var(parsed.name, self._eval_string_expr(parsed.expr, run_vars), run_vars)
        return True, ExecResult.ADVANCE

    def _cf_mid_assign(self, run_vars: Any, parsed: Any) -> tuple[bool, ExecOutcome]:
        """MID$(v$, start[, length]) = expr — overwrite characters of v$ in
        place; the string keeps its length (QBASIC)."""
        lv = self._lvalue(parsed.var, run_vars)
        text = self._read_lvalue(lv, run_vars)
        if not isinstance(text, str):
            raise ValueError(f"TYPE MISMATCH: MID$ needs a string variable, not {parsed.var}")
        start = int(round(float(self._safe_eval(parsed.start, extra_ns=run_vars))))
        if start < 1 or start > len(text):
            raise ValueError("ILLEGAL FUNCTION CALL: MID$ start outside the string")
        new = self._safe_eval(parsed.expr, extra_ns=run_vars)
        if not isinstance(new, str):
            raise ValueError("TYPE MISMATCH: MID$ = needs a string")
        n = len(new)
        if parsed.length is not None:
            n = min(n, int(round(float(self._safe_eval(parsed.length, extra_ns=run_vars)))))
        n = max(0, min(n, len(text) - start + 1))
        self._write_lvalue(lv, text[:start - 1] + new[:n] + text[start - 1 + n:], run_vars)
        return True, ExecResult.ADVANCE

    @staticmethod
    def _split_print_items(expr: str) -> list[tuple[str, str]]:
        """Split a PRINT argument list into (item, trailing-separator) pairs.

        ';' and ',' are recognized only at top level (outside quotes and
        parentheses/brackets), so PRINT LEFT$(s$, 3) stays one item and a comma
        inside a quoted literal is preserved. The separator recorded with each
        item is the one that follows it; '' marks the final item / no trailing
        separator.
        """
        items: list[tuple[str, str]] = []
        buf = ''
        depth = 0
        quote: str | None = None
        for ch in expr:
            if quote:
                buf += ch
                if ch == quote:
                    quote = None
            elif ch in ('"', "'"):
                quote = ch
                buf += ch
            elif ch in '([':
                depth += 1
                buf += ch
            elif ch in ')]':
                depth = max(0, depth - 1)
                buf += ch
            elif ch in ';,' and depth == 0:
                items.append((buf.strip(), ch))
                buf = ''
            else:
                buf += ch
        if buf.strip():
            items.append((buf.strip(), ''))
        return items

    def _eval_print_item(self, item: str, run_vars: dict[str, Any]) -> str:
        """Evaluate a single PRINT item to its display string."""
        item = item.strip()
        if not item:
            return ''
        # Quoted literal: emit verbatim (no substitution, no SPC/TAB).
        if (item[0] == '"' and item[-1] == '"') or (item[0] == "'" and item[-1] == "'"):
            return item[1:-1]
        # Quantum forms: STATE, QUBIT(q), ENTANGLEMENT(a, b), @REG.
        quantum = self._try_quantum_print(item, run_vars)
        if quantum is not None:
            return quantum
        # (A mid-circuit measurement bit read here makes the run go live; the
        # evaluator raises MidCircuitRead.)
        val = self._safe_eval(item, extra_ns=run_vars)
        # Quantum DATA tokens display in ket notation, not the storage form.
        if isinstance(val, str) and val.startswith('QSTATE:'):
            return f"|{val[7:]}>"
        return self._basic_number(val)

    @staticmethod
    def _basic_number(val: Any) -> str:
        """PRINT form of a value: numbers as QBASIC prints them (see
        basic_format: a leading space in place of a plus sign, a trailing
        space, seven significant digits, .5 for 0.5); anything else as str()."""
        if isinstance(val, bool) or not isinstance(val, (int, float)) and not (
                hasattr(val, 'dtype') and getattr(val, 'ndim', 1) == 0
                and val.dtype.kind in 'iuf'):
            return str(val)
        return print_text(val)

    def _cf_print_using(self, run_vars: Any, parsed: Any) -> tuple[bool, ExecOutcome]:
        """PRINT USING fmt$; values — formatted output (basic_format)."""
        self._emit_using(self._using_text(parsed.fmt, parsed.values, run_vars))
        return True, ExecResult.ADVANCE

    def _using_text(self, fmt_expr: str, values: str, run_vars: Any) -> tuple[str, bool]:
        """(formatted text, newline) for PRINT USING; a trailing ; or , keeps
        the cursor on the line."""
        fmt = self._safe_eval(fmt_expr, extra_ns=run_vars)
        if not isinstance(fmt, str):
            raise ValueError("TYPE MISMATCH: PRINT USING needs a string format")
        items = self._split_print_items(values)
        newline = not items or items[-1][1] == ''
        vals = [self._safe_eval(item, extra_ns=run_vars) for item, _ in items if item]
        return format_using(fmt, vals), newline

    def _emit_using(self, out: tuple[str, bool]) -> None:
        text, newline = out
        if newline:
            self.io.writeln(text)
        else:
            self.io.write(text)

    def _cf_print(self, stmt: str, run_vars: dict[str, Any],
                  parsed: PrintStmt) -> tuple[bool, ExecOutcome]:
        text, newline = self._format_print(parsed.expr, run_vars)
        if newline:
            self.io.writeln(text)
        else:
            self.io.write(text)
        return True, ExecResult.ADVANCE

    def _format_print(self, expr: str, run_vars: Any) -> tuple[str, bool]:
        """(text, newline) for a PRINT list: ; joins, , moves to the next
        14-column zone, TAB(n)/SPC(n) place the cursor, and a trailing ; or ,
        keeps the cursor on the line. A bare PRINT is an empty line."""
        items = self._split_print_items(expr)
        if not items:
            return '', True
        out = ''
        for item, sep in items:
            m = _RE_TAB_SPC.match(item.strip())
            if m:
                n = max(0, int(self._eval_with_vars(m.group(2), run_vars)))
                if m.group(1).upper() == 'SPC':
                    out += ' ' * n
                elif n - 1 > len(out):          # TAB(n): move to column n
                    out += ' ' * (n - 1 - len(out))
            else:
                out += self._eval_print_item(item, run_vars)
            if sep == ',':                  # advance to next 14-column zone
                col = len(out) % 14
                out += ' ' * (14 - col if col else 14)
        return out, items[-1][1] not in (';', ',')

    def _cf_goto(self, stmt: str, sorted_lines: Any,
                 parsed: GotoStmt) -> tuple[bool, int]:
        return True, self._jump_to_line(sorted_lines, parsed.target, 'GOTO')

    def _cf_gosub(self, stmt: str, sorted_lines: Any, ip: int,
                  parsed: GosubStmt) -> tuple[bool, int]:
        target = self._jump_to_line(sorted_lines, parsed.target, 'GOSUB')
        self._gosub_stack.append(ip + 1)        # RETURN resumes at the next statement
        return True, target

    def _cf_for(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]], ip: int,
                parsed: ForStmt, sorted_lines: list[int] | None = None) -> tuple[bool, ExecOutcome]:
        var = parsed.var
        start_expr, end_expr, step_expr = parsed.start_expr, parsed.end_expr, parsed.step_expr
        start = self._eval_with_vars(start_expr, run_vars)
        end = self._eval_with_vars(end_expr, run_vars)
        step = self._eval_with_vars(step_expr, run_vars) if step_expr else 1
        if step == 0:
            raise RuntimeError(f"FOR {var}: STEP 0 would never terminate")
        try:
            if start == int(start): start = int(start)
        except (OverflowError, ValueError):
            pass
        try:
            if end == int(end): end = int(end)
        except (OverflowError, ValueError):
            pass
        try:
            if isinstance(step, float) and step == int(step): step = int(step)
        except (OverflowError, ValueError):
            pass
        stored = self._coerce(var, start)
        run_vars[var] = stored
        self.variables[var] = stored
        # A FOR reached again by a jump replaces its stale entry, along with
        # any loops opened inside it.
        for idx in range(len(loop_stack) - 1, -1, -1):
            if loop_stack[idx].get('var') == var:
                del loop_stack[idx:]
                break
        # Zero-trip: a range the step cannot reach skips the body entirely.
        if (step > 0 and start > end) or (step < 0 and start < end):
            return True, self._target(sorted_lines, 'skip', ip, f"FOR {var}")
        loop_stack.append({'var': var, 'current': start, 'end': end, 'step': step,
                           'return_ip': ip})
        return True, ExecResult.ADVANCE

    def _cf_next(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]],
                 parsed: NextStmt) -> tuple[bool, ExecOutcome]:
        # Supports bare ``NEXT`` (close the innermost FOR) and ``NEXT i, j``
        # (close i, then j). A named NEXT closes any loops still open inside
        # its FOR (left by a jump out of them).
        names = [v.strip() for v in parsed.var.split(',')] if parsed.var.strip() else ['']
        for name in names:
            idx = len(loop_stack) - 1
            while idx >= 0 and (loop_stack[idx].get('var') is None
                                or (name and loop_stack[idx]['var'] != name)):
                idx -= 1
            if idx < 0:
                raise RuntimeError(f"NEXT {name}".rstrip() + " without matching FOR")
            del loop_stack[idx + 1:]
            loop = loop_stack[-1]
            cur_var = loop['var']
            loop['current'] += loop['step']
            # The counter takes the stepped value even when the loop ends, so
            # after FOR i = 1 TO 10 runs out, i is 11 (as in QBASIC).
            stored = self._coerce(cur_var, loop['current'])
            run_vars[cur_var] = stored
            self.variables[cur_var] = stored
            if (loop['step'] > 0 and loop['current'] <= loop['end']) or \
               (loop['step'] < 0 and loop['current'] >= loop['end']):
                return True, self._loop_body(loop)
            loop_stack.pop()
        return True, ExecResult.ADVANCE

    def _cf_while(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]],
                  sorted_lines: list[int], ip: int,
                  parsed: WhileStmt) -> tuple[bool, ExecOutcome]:
        cond = parsed.condition
        self._drop_loop(loop_stack, 'while', ip)
        if self._eval_condition(cond, run_vars):
            loop_stack.append({'type': 'while', 'cond': cond, 'return_ip': ip})
            return True, ExecResult.ADVANCE
        return True, self._target(sorted_lines, 'skip', ip, f"WHILE at line {sorted_lines[ip]}")

    def _cf_if_jump(self, run_vars: Any, parsed: Any) -> tuple[bool, ExecOutcome]:
        """A compiled IF test: on to the THEN statements, or to on_false."""
        cond_vars = run_vars
        if self.locc_mode and self.locc:
            base = run_vars.as_dict() if hasattr(run_vars, 'as_dict') else dict(run_vars)
            cond_vars = {**base, **self.locc.classical}
        if self._eval_condition(parsed.condition, cond_vars):
            return True, ExecResult.ADVANCE
        return True, parsed.on_false

    def _cf_wend(self, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]],
                 sorted_lines: list[int] | None = None, ip: int | None = None) -> tuple[bool, ExecOutcome] | None:
        idx = len(loop_stack) - 1
        while idx >= 0 and loop_stack[idx].get('type') != 'while':
            idx -= 1
        if idx < 0:
            ctx = f" at line {sorted_lines[ip]}" if sorted_lines and ip is not None else ""
            raise RuntimeError(f"WEND{ctx} without matching WHILE")
        del loop_stack[idx + 1:]
        loop = loop_stack[-1]
        if self._eval_condition(loop['cond'], run_vars):
            return True, self._loop_body(loop)
        loop_stack.pop()
        return True, ExecResult.ADVANCE


    def _cf_return(self, sorted_lines: Any = None, parsed: Any = None) -> tuple[bool, ExecOutcome]:
        """RETURN [line] — pop the GOSUB stack and resume after the GOSUB, or
        at the given line."""
        if not self._gosub_stack:
            raise RuntimeError("RETURN WITHOUT GOSUB")
        back = self._gosub_stack.pop()
        target = getattr(parsed, 'target', None)
        if target is not None:
            return True, self._jump_to_line(sorted_lines, target, 'RETURN')
        return True, back

    # ── Type-based dispatch table ────────────────────────────────────
    # Maps parsed Stmt types to handler lambdas.  Each lambda receives
    # (self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars,
    # exec_fn) and returns (handled: bool, result).

    _CF_DISPATCH: dict[type, Callable] = {
        # Trivial handlers (no _cf_* method needed)
        RemStmt:         lambda s, st, p, ls, sl, ip, rv, ef: (True, ExecResult.ADVANCE),
        MeasureStmt:     lambda s, st, p, ls, sl, ip, rv, ef: (True, ExecResult.ADVANCE),
        EndStmt:         lambda s, st, p, ls, sl, ip, rv, ef: (True, ExecResult.END),
        ReturnStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_return(sl, p),
        # Handlers defined in control_flow.py (parsed is positional)
        WendStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_wend(rv, ls, sl, ip),
        LetArrayStmt:    lambda s, st, p, ls, sl, ip, rv, ef: s._cf_let_array(st, rv, p),
        LetStmt:         lambda s, st, p, ls, sl, ip, rv, ef: s._cf_let_var(st, rv, p),
        LetStrStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_let_str(st, rv, p),
        PrintStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_print(st, rv, p),
        PrintUsingStmt:  lambda s, st, p, ls, sl, ip, rv, ef: s._cf_print_using(rv, p),
        MidAssignStmt:   lambda s, st, p, ls, sl, ip, rv, ef: s._cf_mid_assign(rv, p),
        GotoStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_goto(st, sl, p),
        GosubStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_gosub(st, sl, ip, p),
        ForStmt:         lambda s, st, p, ls, sl, ip, rv, ef: s._cf_for(st, rv, ls, ip, p, sl),
        NextStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_next(st, rv, ls, p),
        WhileStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_while(st, rv, ls, sl, ip, p),
        # Handlers defined in classic.py (parsed is keyword-only)
        DataStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_data(st, parsed=p),
        ReadStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_read(st, rv, parsed=p),
        OnGotoStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_on_goto(st, rv, sl, parsed=p),
        OnGosubStmt:     lambda s, st, p, ls, sl, ip, rv, ef: s._cf_on_gosub(st, rv, sl, ip, parsed=p),
        SelectCaseStmt:  lambda s, st, p, ls, sl, ip, rv, ef: s._cf_select_case(st, rv, sl, ip, parsed=p),
        CaseStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_case(st, sl, ip, parsed=p),
        EndSelectStmt:   lambda s, st, p, ls, sl, ip, rv, ef: s._cf_end_select(st, parsed=p),
        # (Compiled block IFs leave only an ELSE outside any IF, and END IF.)
        ElseStmt:        lambda s, st, p, ls, sl, ip, rv, ef: (True, ExecResult.ADVANCE),
        EndIfStmt:       lambda s, st, p, ls, sl, ip, rv, ef: (True, ExecResult.ADVANCE),
        DoStmt:          lambda s, st, p, ls, sl, ip, rv, ef: s._cf_do(st, rv, ls, sl, ip, parsed=p),
        LoopStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_loop(st, rv, ls, sl, ip, parsed=p),
        ExitStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_exit(st, ls, sl, ip, parsed=p),
        SwapStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_swap(st, rv, parsed=p),
        DefFnStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_def_fn(st, rv, parsed=p),
        OptionBaseStmt:  lambda s, st, p, ls, sl, ip, rv, ef: s._cf_option_base(st, parsed=p),
        OptionEndianStmt: lambda s, st, p, ls, sl, ip, rv, ef: s._cf_option_endian(st, parsed=p),
        # Handlers defined in subs.py (parsed is keyword-only)
        SubStmt:         lambda s, st, p, ls, sl, ip, rv, ef: s._cf_sub(st, sl, ip, parsed=p),
        EndSubStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_end_sub(st, parsed=p),
        FunctionStmt:    lambda s, st, p, ls, sl, ip, rv, ef: s._cf_function(st, sl, ip, parsed=p),
        EndFunctionStmt: lambda s, st, p, ls, sl, ip, rv, ef: s._cf_end_function(st, parsed=p),
        CallStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_call(st, rv, sl, ip, parsed=p),
        LocalStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_local(st, rv, parsed=p),
        StaticStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_static(st, rv, parsed=p),
        SharedStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_shared(st, rv, parsed=p),
        # Handlers defined in debug.py (parsed is keyword-only)
        OnErrorStmt:     lambda s, st, p, ls, sl, ip, rv, ef: s._cf_on_error(st, parsed=p),
        ResumeStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_resume(st, sl, parsed=p),
        ErrorStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_error(st, parsed=p),
        AssertStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_assert(st, rv, parsed=p),
        StopStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_stop(st, sl, ip, parsed=p),
        OnMeasureStmt:   lambda s, st, p, ls, sl, ip, rv, ef: s._cf_on_measure(st, parsed=p),
        OnTimerStmt:     lambda s, st, p, ls, sl, ip, rv, ef: s._cf_on_timer(st, parsed=p),
        # Compiled control transfers (program.py)
        IfJumpStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_if_jump(rv, p),
        JumpStmt:        lambda s, st, p, ls, sl, ip, rv, ef: (True, p.target),
    }

    def _exec_control_flow(
        self, stmt: str, loop_stack: list[dict[str, Any]],
        sorted_lines: list[int], ip: int, run_vars: dict[str, Any],
        exec_fn: Callable[..., Any],
        *, parsed=None,
    ) -> tuple[bool, ExecOutcome | None]:
        """Shared control flow for both Qiskit and LOCC execution paths.
        Returns (handled, result) -- if handled is True, result is the
        return value.  exec_fn is the recursive line executor for
        IF/multi-statement dispatch.

        Dispatches via dict lookup on the parsed Stmt type (O(1)).

        If *parsed* is provided, the parse_stmt call is skipped (avoids
        redundant parsing when the caller has already parsed the statement).
        """
        if parsed is None:
            parsed = parse_stmt(stmt)
        handler = self._CF_DISPATCH.get(type(parsed))
        if handler is not None:
            return handler(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn)

        # RawStmt or unmapped type -- not handled by control flow
        return False, None
