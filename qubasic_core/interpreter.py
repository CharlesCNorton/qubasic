"""QUBASIC statement execution shared by every run path: control
flow, the classic statements (DATA/READ, SELECT CASE, DO/LOOP, SWAP, DEF FN),
SUB and FUNCTION, error trapping and the debugger, and the keyword
statements.
"""

from __future__ import annotations

import random
import re
import sys
import time
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np
from qiskit import QuantumCircuit

from qubasic_core.devices import KEY_TRAP_KEYS, defined_keys, read_terminal_keys
from qubasic_core.engine import ExecContext, ExecOutcome, ExecResult
from qubasic_core.expression import (
    ArrayRef,
    format_using,
    is_string_kind,
    next_column,
    number_text,
    print_text,
    type_text,
)
from qubasic_core.host import LanguageHost
from qubasic_core.parser import (
    RE_GET,
    RE_LET_STR,
    RE_UNITARY,
    AssertStmt,
    CallStmt,
    CaseStmt,
    DataStmt,
    DefFnStmt,
    DoStmt,
    ElseStmt,
    EndFunctionStmt,
    EndIfStmt,
    EndSelectStmt,
    EndStmt,
    EndSubStmt,
    ErrorStmt,
    ExitStmt,
    ForStmt,
    FunctionStmt,
    GosubStmt,
    GotoStmt,
    IfJumpStmt,
    JumpStmt,
    LetArrayStmt,
    LetStmt,
    LetStrStmt,
    LocalStmt,
    LoopStmt,
    MeasureStmt,
    MidAssignStmt,
    NextStmt,
    OnErrorStmt,
    OnGosubStmt,
    OnGotoStmt,
    OnMeasureStmt,
    OnTimerStmt,
    OptionBaseStmt,
    OptionEndianStmt,
    PrintStmt,
    PrintUsingStmt,
    ReadStmt,
    RemStmt,
    ResumeStmt,
    ReturnStmt,
    RunStmt,
    SelectCaseStmt,
    SharedStmt,
    StaticStmt,
    Stmt,
    StopStmt,
    SubStmt,
    SwapStmt,
    WendStmt,
    WhileStmt,
    parse_stmt,
)

if TYPE_CHECKING:
    from qubasic_core.engine import CircuitBackend
    from qubasic_core.parser import CompiledProgram


# ---------------------------------------------------------------------------
# QUBASIC control flow: GOTO, GOSUB, FOR, WHILE, DO, IF, SELECT CASE, EXIT,
# procedure calls, PRINT and the assignments, run over a compiled program.
# ---------------------------------------------------------------------------

_RE_TAB_SPC = re.compile(r'(TAB|SPC)\s*\((.+)\)$', re.IGNORECASE)


class ControlFlowMixin(LanguageHost):
    """Mixin providing control flow helpers for QBasicTerminal."""

    # ── Compiled block structure ────────────────────────────────────
    #
    # Statements run from a CompiledProgram (parser.py): `ip` indexes a
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
        from qubasic_core.parser import line_index
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

    def _cf_let_array(self, stmt: str, run_vars: MutableMapping[str, Any],
                      parsed: LetArrayStmt) -> tuple[bool, ExecOutcome]:
        """LET a(i[, j ...])[.field] = expr — a typed element (or record element
        field) write (see VariableMixin)."""
        name = parsed.name
        self._assert_assignable(name)
        value = self._safe_eval(parsed.value_expr, extra_ns=run_vars)
        indices = [self._safe_eval(p, extra_ns=run_vars)
                   for p in self._split_arg_list(parsed.index_expr)]
        self._array_write(name, indices, value, parsed.field)
        return True, ExecResult.ADVANCE

    def _cf_let_var(self, stmt: str, run_vars: MutableMapping[str, Any],
                    parsed: LetStmt) -> tuple[bool, ExecOutcome]:
        """LET v = expr — a typed scalar or record field write."""
        self._store_var(parsed.name, self._safe_eval(parsed.expr, extra_ns=run_vars), run_vars)
        return True, ExecResult.ADVANCE

    def _cf_let_str(self, stmt: str, run_vars: MutableMapping[str, Any],
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

    def _eval_print_item(self, item: str, run_vars: MutableMapping[str, Any]) -> str:
        """Evaluate a single PRINT item to its display string."""
        item = item.strip()
        if not item:
            return ''
        # One quoted literal: emit verbatim ("a" + b$ is an expression).
        if item[0] == '"' and item.find('"', 1) == len(item) - 1:
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
        self._emit_text(*out)

    def _emit_text(self, text: str, newline: bool) -> None:
        """Write program output, keeping the cursor column that POS reads and
        TAB and the print zones count from."""
        start = self.screen.col
        if newline:
            self.io.writeln(text)
            self.screen.col = 0
        else:
            self.io.write(text)
            self.screen.col = next_column(self.screen.col, text)
        if self.graphics.canvas is not None:
            self._gfx_print(text, newline)
        else:
            self._text_put(text, newline, start)

    def _cf_print(self, stmt: str, run_vars: MutableMapping[str, Any],
                  parsed: PrintStmt) -> tuple[bool, ExecOutcome]:
        self._emit_text(*self._format_print(parsed.expr, run_vars, start=self.screen.col))
        return True, ExecResult.ADVANCE

    def _format_print(self, expr: str, run_vars: Any, start: int = 0) -> tuple[str, bool]:
        """(text, newline) for a PRINT list begun at column `start`: ; joins,
        , moves to the next 14-column zone, TAB(n)/SPC(n) place the cursor, and
        a trailing ; or , keeps the cursor on the line. A bare PRINT is an
        empty line."""
        items = self._split_print_items(expr)
        if not items:
            return '', True
        out = ''
        for item, sep in items:
            m = _RE_TAB_SPC.match(item.strip())
            if m:
                n = max(0, int(self._eval_with_vars(m.group(2), run_vars)))
                col = next_column(start, out)
                if m.group(1).upper() == 'SPC':
                    out += ' ' * n
                elif n - 1 > col:               # TAB(n): move to column n
                    out += ' ' * (n - 1 - col)
            else:
                out += self._eval_print_item(item, run_vars)
            if sep == ',':                  # advance to next 14-column zone
                col = next_column(start, out) % 14
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

    def _cf_for(self, stmt: str, run_vars: MutableMapping[str, Any], loop_stack: list[dict[str, Any]], ip: int,
                parsed: ForStmt, sorted_lines: Sequence[int] | None = None) -> tuple[bool, ExecOutcome]:
        var = parsed.var
        start_expr, end_expr, step_expr = parsed.start_expr, parsed.end_expr, parsed.step_expr
        start = self._eval_with_vars(start_expr, run_vars)
        end = self._eval_with_vars(end_expr, run_vars)
        step = self._eval_with_vars(step_expr, run_vars) if step_expr else 1
        if step == 0:
            raise RuntimeError(f"FOR {var}: STEP 0 would never terminate")
        try:
            if start == int(start):
                start = int(start)
        except (OverflowError, ValueError):
            pass
        try:
            if end == int(end):
                end = int(end)
        except (OverflowError, ValueError):
            pass
        try:
            if isinstance(step, float) and step == int(step):
                step = int(step)
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

    def _cf_next(self, stmt: str, run_vars: MutableMapping[str, Any], loop_stack: list[dict[str, Any]],
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

    def _cf_while(self, stmt: str, run_vars: MutableMapping[str, Any], loop_stack: list[dict[str, Any]],
                  sorted_lines: Sequence[int], ip: int,
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

    def _cf_wend(self, run_vars: MutableMapping[str, Any], loop_stack: list[dict[str, Any]],
                 sorted_lines: Sequence[int] | None = None, ip: int | None = None) -> tuple[bool, ExecOutcome] | None:
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
        # ControlFlowMixin's handlers (parsed is positional)
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
        # ClassicMixin's handlers (parsed is keyword-only)
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
        # SubroutineMixin's handlers (parsed is keyword-only)
        SubStmt:         lambda s, st, p, ls, sl, ip, rv, ef: s._cf_sub(st, sl, ip, parsed=p),
        EndSubStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_end_sub(st, parsed=p),
        FunctionStmt:    lambda s, st, p, ls, sl, ip, rv, ef: s._cf_function(st, sl, ip, parsed=p),
        EndFunctionStmt: lambda s, st, p, ls, sl, ip, rv, ef: s._cf_end_function(st, parsed=p),
        CallStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_call(st, rv, sl, ip, parsed=p),
        LocalStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_local(st, rv, parsed=p),
        StaticStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_static(st, rv, parsed=p),
        SharedStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_shared(st, rv, parsed=p),
        # DebugMixin's handlers (parsed is keyword-only)
        OnErrorStmt:     lambda s, st, p, ls, sl, ip, rv, ef: s._cf_on_error(st, parsed=p),
        ResumeStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_resume(st, sl, parsed=p),
        ErrorStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_error(st, parsed=p),
        AssertStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_assert(st, rv, parsed=p),
        StopStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_stop(st, sl, ip, parsed=p),
        RunStmt:         lambda s, st, p, ls, sl, ip, rv, ef: s._cf_run(sl, ls, parsed=p),
        OnMeasureStmt:   lambda s, st, p, ls, sl, ip, rv, ef: s._cf_on_measure(st, parsed=p),
        OnTimerStmt:     lambda s, st, p, ls, sl, ip, rv, ef: s._cf_on_timer(st, parsed=p),
        # Compiled control transfers (parser.py)
        IfJumpStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_if_jump(rv, p),
        JumpStmt:        lambda s, st, p, ls, sl, ip, rv, ef: (True, p.target),
    }

    def _exec_control_flow(
        self, stmt: str, loop_stack: list[dict[str, Any]],
        sorted_lines: Sequence[int], ip: int, run_vars: MutableMapping[str, Any],
        exec_fn: Callable[..., Any],
        *, parsed: Stmt | None = None,
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

    def _compiled(self) -> CompiledProgram:
        """The compiled form of the current program (cached until it changes)."""
        from qubasic_core.parser import compile_program
        key = tuple(sorted(self.program.items()))
        cached = getattr(self, '_compiled_cache', None)
        if cached is None or cached[0] != key:
            cached = (key, compile_program(self.program))
            self._compiled_cache = cached
        return cached[1]


# ---------------------------------------------------------------------------
# QUBASIC classic features — DATA/READ, ON GOTO, SELECT CASE, DO/LOOP, EXIT, SWAP, DEF FN, OPTION BASE.
# ---------------------------------------------------------------------------

def _data_items(text: str) -> list[tuple[str, bool]]:
    """(item, quoted) for each comma-separated DATA item; a double-quoted
    item may hold commas."""
    items: list[tuple[str, bool]] = []
    buf: list[str] = []
    quoted = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"' and not ''.join(buf).strip():
            j = text.find('"', i + 1)
            end = len(text) if j < 0 else j
            buf, quoted = [text[i + 1:end]], True
            i = end + 1
            continue
        if ch == ',':
            items.append((''.join(buf) if quoted else ''.join(buf).strip(), quoted))
            buf, quoted = [], False
        elif not quoted:
            buf.append(ch)
        i += 1
    items.append((''.join(buf) if quoted else ''.join(buf).strip(), quoted))
    return items


@dataclass
class DataState:
    """DATA, READ and RESTORE: the program's DATA values in order, READ's
    position, where each line's values start, and the program text they
    were collected from."""

    pool: list[Any] = field(default_factory=list)
    ptr: int = 0
    line_starts: list[tuple[int, int]] = field(default_factory=list)  # (line, first value's index)
    cache_key: tuple | None = None


class ClassicMixin(LanguageHost):
    """Classic BASIC features for QBasicTerminal."""

    def _init_classic(self) -> None:
        self.data = DataState()
        # OPTION ENDIAN BIG shows bitstrings with qubit 0 leftmost.
        self._endian_big: bool = False
        # RND draws from this generator; RANDOMIZE and SEED reseed it.
        self._rng = random.Random()
        self._rnd_last: float = 0.0

    # ── RND / RANDOMIZE ────────────────────────────────────────────────

    def _rnd(self, x: Any = 1) -> float:
        """RND / RND(x): the next number of the sequence; RND(0) repeats the
        last one, and a negative x reseeds with x first (QBASIC)."""
        v = float(x)
        if v < 0:
            self._rng.seed(v)
        elif v == 0:
            return self._rnd_last
        self._rnd_last = self._rng.random()
        return self._rnd_last

    def _stmt_randomize(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """RANDOMIZE [seed] — reseed RND (from the clock when no seed is given)."""
        m = re.match(r'RANDOMIZE\b\s*(.*)$', stmt, re.IGNORECASE)
        if not m:
            return False
        arg = m.group(1).strip()
        seed = self._safe_eval(arg, extra_ns=run_vars) if arg else time.time()
        self._rng.seed(float(seed))
        return True

    # Quantum state names recognized in DATA statements
    _QUANTUM_STATES = {
        '|0>': '0', '|1>': '1', '|+>': '+', '|->': '-',
        '|0⟩': '0', '|1⟩': '1', '|+⟩': '+', '|-⟩': '-',
        '|BELL>': 'BELL', '|GHZ>': 'GHZ', '|GHZ3>': 'GHZ3', '|GHZ4>': 'GHZ4',
        '|W>': 'W', '|W3>': 'W3',
    }

    def _collect_data(self) -> None:
        """Build the DATA pool from the program's DATA statements, in line
        order. Quoted items are strings (commas inside kept), numbers are
        numbers, quantum state names (|+>, |GHZ3>, ...) become QSTATE tokens,
        and other unquoted items are strings. The pool is cached until the
        DATA statements change; each RUN starts READ from the first item."""
        from qubasic_core.parser import DataStmt
        prog = self._compiled()
        data = tuple((prog.lines[ip], st.values) for ip, st in enumerate(prog.stmts)
                     if isinstance(st, DataStmt))
        self.data.ptr = 0
        if data == self.data.cache_key:
            return
        self.data.pool = []
        self.data.line_starts = []
        for line, values in data:
            self.data.line_starts.append((line, len(self.data.pool)))
            for item, quoted in _data_items(values):
                if quoted:
                    self.data.pool.append(item)
                elif item in self._QUANTUM_STATES:
                    self.data.pool.append(f"QSTATE:{self._QUANTUM_STATES[item]}")
                else:
                    num = _parse_input_number(item) if item else 0
                    self.data.pool.append(num if num is not None else item)
        self.data.cache_key = data

    # ── DATA / READ / RESTORE ──────────────────────────────────────────

    def _cf_data(self, stmt: str, *, parsed: DataStmt) -> tuple[bool, ExecOutcome]:
        """DATA — skip during execution (collected before run)."""
        return True, ExecResult.ADVANCE

    def _cf_read(self, stmt: str, run_vars: MutableMapping[str, Any], *, parsed: ReadStmt) -> tuple[bool, ExecOutcome]:
        var_names = [v.strip() for v in parsed.var_list.split(',')]
        for vname in var_names:
            if self.data.ptr >= len(self.data.pool):
                raise RuntimeError("OUT OF DATA")
            val = self.data.pool[self.data.ptr]
            self.data.ptr += 1
            lv = self._lvalue(vname, run_vars)
            stringy = is_string_kind(self._lvalue_kind(lv))
            if stringy and not isinstance(val, str):
                val = number_text(val)
            elif not stringy and isinstance(val, str) and not val.startswith('QSTATE:'):
                raise RuntimeError(f"TYPE MISMATCH: READ {vname} found \"{val}\"")
            if isinstance(val, str) and not stringy:
                self.variables[lv.name] = val          # a quantum DATA token
                run_vars[lv.name] = val
            else:
                self._write_lvalue(lv, val, run_vars)
        return True, ExecResult.ADVANCE

    def _restore_to(self, line: int | None) -> None:
        """RESTORE [line]: READ continues from the first DATA at or after line."""
        if line is None:
            self.data.ptr = 0
            return
        self.data.ptr = next((start for ln, start in self.data.line_starts if ln >= line),
                              len(self.data.pool))

    def cmd_restore(self, rest: str = '') -> None:
        """RESTORE — reset DATA pointer to beginning."""
        self.data.ptr = 0

    # ── ON expr GOTO / GOSUB ──────────────────────────────────────────

    def _cf_on_goto(self, stmt: str, run_vars: MutableMapping[str, Any],
                    sorted_lines: Sequence[int], *, parsed: OnGotoStmt) -> tuple[bool, ExecOutcome]:
        idx = int(self._eval_with_vars(parsed.expr, run_vars))
        targets = list(parsed.targets)
        if 1 <= idx <= len(targets):
            return True, self._jump_to_line(sorted_lines, targets[idx - 1], 'ON GOTO')
        return True, ExecResult.ADVANCE  # out of range: fall through

    def _cf_on_gosub(self, stmt: str, run_vars: MutableMapping[str, Any],
                     sorted_lines: Sequence[int], ip: int, *, parsed: OnGosubStmt) -> tuple[bool, ExecOutcome]:
        idx = int(self._eval_with_vars(parsed.expr, run_vars))
        targets = list(parsed.targets)
        if 1 <= idx <= len(targets):
            target = self._jump_to_line(sorted_lines, targets[idx - 1], 'ON GOSUB')
            self._gosub_stack.append(ip + 1)
            return True, target
        return True, ExecResult.ADVANCE

    # ── SELECT CASE ───────────────────────────────────────────────────

    def _cf_select_case(self, stmt: str, run_vars: MutableMapping[str, Any],
                        sorted_lines: Sequence[int], ip: int, *, parsed: SelectCaseStmt) -> tuple[bool, ExecOutcome]:
        test_val = self._safe_eval(parsed.expr, extra_ns=run_vars)
        # The first CASE that matches runs its block (the statements after it);
        # none matching continues after END SELECT.
        selects = self._as_program(sorted_lines).selects
        if ip not in selects:
            raise RuntimeError("SELECT CASE without END SELECT")
        cases, end_ip = selects[ip]
        for case_ip, spec in cases:
            if self._case_matches(spec, test_val, run_vars):
                return True, case_ip + 1
        return True, end_ip + 1

    _CASE_OPS = {'=': lambda a, b: a == b, '<>': lambda a, b: a != b,
                 '<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
                 '>': lambda a, b: a > b, '>=': lambda a, b: a >= b}

    def _case_matches(self, spec: str, test_val: Any, run_vars: MutableMapping[str, Any]) -> bool:
        """Whether a CASE clause matches the SELECT value: CASE ELSE, a list of
        values, ranges (lo TO hi) and comparisons (IS > v), as in QBASIC.
        String cases compare exactly."""
        if spec.strip().upper() == 'ELSE':
            return True

        def value(expr: str) -> Any:
            v = self._safe_eval(expr.strip(), extra_ns=run_vars)
            return v if isinstance(v, str) else float(v)

        target = test_val if isinstance(test_val, str) else float(test_val)
        for item in self._split_arg_list(spec):
            try:
                m = re.match(r'IS\s*(<=|>=|<>|=|<|>)\s*(.+)$', item, re.IGNORECASE)
                if m:
                    if self._CASE_OPS[m.group(1)](target, value(m.group(2))):
                        return True
                    continue
                m = re.match(r'(.+?)\s+TO\s+(.+)$', item, re.IGNORECASE)
                if m:
                    if value(m.group(1)) <= target <= value(m.group(2)):
                        return True
                    continue
                if target == value(item):
                    return True
            except (TypeError, ValueError):
                continue    # a string case against a number (or vice versa)
        return False

    def _cf_case(self, stmt: str, sorted_lines: Sequence[int], ip: int, *, parsed: CaseStmt) -> tuple[bool, ExecOutcome]:
        # Reaching another CASE ends the matched block: continue after END SELECT.
        skip = self._as_program(sorted_lines).skip
        return True, skip.get(ip, ExecResult.ADVANCE)

    def _cf_end_select(self, stmt: str, *, parsed: EndSelectStmt) -> tuple[bool, ExecOutcome]:
        return True, ExecResult.ADVANCE

    # ── DO / LOOP ─────────────────────────────────────────────────────

    def _cf_do(self, stmt: str, run_vars: MutableMapping[str, Any],
               loop_stack: list[dict[str, Any]],
               sorted_lines: Sequence[int], ip: int, *, parsed: DoStmt) -> tuple[bool, ExecOutcome]:
        kind = parsed.kind          # WHILE, UNTIL or None
        cond = parsed.condition     # the condition or None
        self._drop_loop(loop_stack, 'do', ip)
        if kind and cond:
            # Pre-test
            result = self._eval_condition(cond, run_vars)
            if kind.upper() == 'UNTIL':
                result = not result
            if not result:
                return True, self._target(sorted_lines, 'skip', ip,
                                          f"DO at line {sorted_lines[ip]}")
        loop_stack.append({'type': 'do', 'return_ip': ip, 'kind': kind, 'cond': cond})
        return True, ExecResult.ADVANCE

    def _cf_loop(self, stmt: str, run_vars: MutableMapping[str, Any],
                 loop_stack: list[dict[str, Any]],
                 sorted_lines: Sequence[int], ip: int, *, parsed: LoopStmt) -> tuple[bool, ExecOutcome | None]:
        loop_kind = parsed.kind
        loop_cond = parsed.condition
        # Only handle as DO/LOOP keyword if we're inside a DO block (loops a
        # jump left open inside it are closed here).
        idx = len(loop_stack) - 1
        while idx >= 0 and loop_stack[idx].get('type') != 'do':
            idx -= 1
        if idx < 0:
            # Outside a DO block a bare LOOP calls a SUB or DEF named LOOP.
            if loop_kind is None and ('LOOP' in self.procs.subs or 'LOOP' in self.subroutines):
                return False, None
            raise ValueError("LOOP without DO")
        del loop_stack[idx + 1:]
        loop = loop_stack[-1]
        # Post-test condition on LOOP
        kind = loop_kind
        cond = loop_cond
        if kind and cond:
            result = self._eval_condition(cond, run_vars)
            if kind.upper() == 'UNTIL':
                result = not result
            if result:
                return True, self._loop_body(loop)
            else:
                loop_stack.pop()
                return True, ExecResult.ADVANCE
        # Pre-test condition was on DO
        if loop.get('kind') and loop.get('cond'):
            result = self._eval_condition(loop['cond'], run_vars)
            if loop['kind'].upper() == 'UNTIL':
                result = not result
            if result:
                return True, self._loop_body(loop)
            else:
                loop_stack.pop()
                return True, ExecResult.ADVANCE
        # Infinite DO/LOOP (no condition) — loop forever
        return True, self._loop_body(loop)

    # ── EXIT ──────────────────────────────────────────────────────────

    def _cf_exit(self, stmt: str, loop_stack: list[dict[str, Any]],
                 sorted_lines: Sequence[int], ip: int, *, parsed: ExitStmt) -> tuple[bool, ExecOutcome]:
        kind = parsed.target
        target = self._as_program(sorted_lines).exits.get(ip)
        if kind in ('FOR', 'WHILE', 'DO'):
            # Leave the innermost loop of that kind (and any loops inside it)
            # for the statement after its NEXT / WEND / LOOP.
            want = kind.lower()
            idx = len(loop_stack) - 1
            while idx >= 0 and not (
                    (want == 'for' and loop_stack[idx].get('var') is not None)
                    or loop_stack[idx].get('type') == want):
                idx -= 1
            if idx >= 0 and target is not None:
                del loop_stack[idx:]
                return True, target
        elif kind in ('SUB', 'FUNCTION'):
            # On to the END SUB / END FUNCTION, which returns.
            return True, target if target is not None else ExecResult.END
        raise RuntimeError(f"EXIT {kind}: no matching block")

    # ── SWAP ──────────────────────────────────────────────────────────

    def _cf_swap(self, stmt: str, run_vars: MutableMapping[str, Any], *, parsed: SwapStmt) -> tuple[bool, ExecOutcome]:
        a, b = parsed.a, parsed.b
        la, lb = self._lvalue(a, run_vars), self._lvalue(b, run_vars)
        va, vb = self._read_lvalue(la, run_vars), self._read_lvalue(lb, run_vars)
        if isinstance(va, str) != isinstance(vb, str):
            raise RuntimeError(f"TYPE MISMATCH: SWAP {a}, {b}")
        self._write_lvalue(la, vb, run_vars)
        self._write_lvalue(lb, va, run_vars)
        return True, ExecResult.ADVANCE

    # ── DEF FN ────────────────────────────────────────────────────────

    def _cf_def_fn(self, stmt: str, run_vars: MutableMapping[str, Any], *, parsed: DefFnStmt) -> tuple[bool, ExecOutcome]:
        params = [p.strip() for p in parsed.params.split(',') if p.strip()]
        self.symbols.fns['FN' + parsed.name.upper()] = {'params': params, 'body': parsed.body.strip()}
        return True, ExecResult.ADVANCE

    # ── OPTION BASE ───────────────────────────────────────────────────

    def _cf_option_base(self, stmt: str, *, parsed: OptionBaseStmt) -> tuple[bool, ExecOutcome]:
        """OPTION BASE 0|1 — the lower bound of arrays declared with an upper
        bound only (VariableMixin)."""
        self.symbols.option_base = parsed.base
        return True, ExecResult.ADVANCE

    def _cf_option_endian(self, stmt: str, *, parsed: OptionEndianStmt) -> tuple[bool, ExecOutcome]:
        """OPTION ENDIAN BIG|LITTLE — bitstring display order (default LITTLE,
        the qiskit convention: qubit 0 rightmost). Display-side only: internal
        counts keys and statevector indexing keep the qiskit order."""
        self._endian_big = parsed.big
        return True, ExecResult.ADVANCE


# ---------------------------------------------------------------------------
# QUBASIC SUB/FUNCTION — procedures with their own variables, as in QBASIC.
#
# A procedure starts with only its parameters, its STATIC variables, the CONST
# names and the variables made module-wide by DIM SHARED; `SHARED x` inside
# the procedure brings in one more of the caller's variables. Everything else
# it assigns is local and is gone when it returns, arrays it creates included.
#
# Parameters take QBASIC's forms: `n`, `n%`, `n AS INTEGER`, `s AS STRING`,
# `p AS Point` (a record) and `a()` or `a() AS INTEGER` (a whole array). An
# argument that is a plain variable, an array element or a record field is
# passed by reference: the procedure's final value of the parameter is written
# back to it. A whole array is always passed by reference (`Sort values()`).
# `SUB name(...) STATIC` keeps every local variable between calls; `STATIC x`
# keeps one. A FUNCTION returns the value last assigned to its own name. LOCAL
# (a QUBASIC extension) zeroes a name inside a procedure.
# ---------------------------------------------------------------------------

_RE_REF_HEAD = re.compile(r'\s*[A-Za-z_][\w.]*[$%!#&]?\s*')
_RE_FIELD_TAIL = re.compile(r'(?:\s*\.\s*[A-Za-z_]\w*[$%!#&]?)*\s*$')
_RE_WHOLE_ARRAY = re.compile(r'\s*([A-Za-z_]\w*[$%!#&]?)\s*\(\s*\)\s*$')
_RE_PARAM = re.compile(r'\s*([A-Za-z_]\w*[$%!#&]?)\s*(\(\s*\))?\s*'
                       r'(?:AS\s+([A-Za-z_]\w*)(?:\s*\*\s*(\d+))?)?\s*$', re.IGNORECASE)


class Param(NamedTuple):
    """One declared parameter (or STATIC/SHARED name): its name, whether it
    is a whole array, and its declared type (None: from suffix or DEFtype)."""
    name: str
    array: bool
    type: str | None


@lru_cache(maxsize=4096)
def parse_param(text: str) -> Param:
    m = _RE_PARAM.match(text)
    if not m:
        raise ValueError(f"SYNTAX ERROR: parameter '{text.strip()}'")
    typ = type_text(m.group(3), m.group(4)) if m.group(3) else None
    return Param(m.group(1), bool(m.group(2)), typ)


def _names_storage(text: str) -> bool:
    """Whether an argument is a variable, a record field or one array element
    (or a field of one), which QBASIC passes by reference; anything else is
    an expression, passed by value."""
    m = _RE_REF_HEAD.match(text)
    if not m:
        return False
    rest = text[m.end():].rstrip()
    if not rest:
        return True
    if not rest.startswith('('):
        return False
    depth = 0
    for i, ch in enumerate(rest):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return bool(_RE_FIELD_TAIL.fullmatch(rest[i + 1:]))
    return False


@dataclass
class ProcedureState:
    """SUB and FUNCTION: their definitions, the call stack, STATIC values, and
    per scope level the names whose changes survive the return, the caller's
    declared types, the arrays that existed before the call and the arrays
    bound to array parameters."""

    subs: dict[str, dict[str, Any]] = field(default_factory=dict)
    functions: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    depth: int = 0
    statics: dict[str, dict[str, Any]] = field(default_factory=lambda: {'_GLOBAL': {}})
    scopes: list[dict[str, Any]] = field(default_factory=list)
    shared_scopes: list[set[str]] = field(default_factory=list)
    type_scopes: list[dict[str, str]] = field(default_factory=list)
    array_scopes: list[set[str]] = field(default_factory=list)
    alias_scopes: list[list[tuple[str, str, Any]]] = field(default_factory=list)


class SubroutineMixin(LanguageHost):
    """SUB/FUNCTION procedures with local scope, typed and array parameters,
    by-reference arguments and LOCAL/STATIC/SHARED declarations (see the
    module docstring)."""

    def _init_subs(self) -> None:
        self.procs = ProcedureState()

    def _scan_subs(self, prog: Any = None) -> None:
        """Index the program's SUB and FUNCTION blocks (statement indices of
        the compiled program)."""
        prog = prog if hasattr(prog, 'subs') else self._compiled()
        self.procs.subs.clear()
        self.procs.functions.clear()
        statics: set[str] = getattr(prog, 'static_routines', set())
        for table, out in ((prog.subs, self.procs.subs), (prog.funcs, self.procs.functions)):
            for name, (start, end, params) in table.items():
                out[name] = {'params': params, 'start_ip': start, 'end_ip': end,
                             'static': name in statics}

    # ── Scope ──────────────────────────────────────────────────────────

    def _module_wide(self, name: str) -> bool:
        """Whether a caller's variable stays visible inside a procedure."""
        base = name.split('.', 1)[0]
        return (name in self.symbols.shared or base in self.symbols.shared
                or name in self.symbols.consts or name.startswith('_')
                or name in ('ERR', 'ERL'))

    def _push_scope(self) -> None:
        """Enter a procedure: keep the caller's variables, declared types and
        arrays for _pop_scope and leave only the module-wide variables visible."""
        caller = dict(self.variables)
        self.procs.scopes.append(caller)
        self.procs.shared_scopes.append(set(n for n in caller if self._module_wide(n)))
        self.procs.type_scopes.append(dict(self.symbols.var_types))
        self.procs.array_scopes.append(set(self.arrays))
        self.procs.alias_scopes.append([])
        self.variables.clear()
        self.variables.update({k: v for k, v in caller.items() if self._module_wide(k)})

    def _pop_scope(self, frame: dict[str, Any] | None = None) -> None:
        """Leave a procedure: save its STATIC variables, unbind its array
        parameters, drop its local arrays, restore the caller's variables and
        declared types, and carry back the SHARED and module-wide variables
        it changed."""
        if frame is None and self.procs.calls:
            frame = self.procs.calls[-1]
        if frame is not None:
            name = frame.get('sub_name') or frame.get('func_name') or ''
            if frame.get('static'):
                params = set(frame.get('params', ()))
                keep = self.procs.shared_scopes[-1] if self.procs.shared_scopes else set()
                self.procs.statics[name] = {k: v for k, v in self.variables.items()
                                           if k not in params and k not in keep
                                           and not self._module_wide(k)}
            elif name in self.procs.statics:
                for vname in list(self.procs.statics[name]):
                    self.procs.statics[name][vname] = self.variables.get(vname, 0)
        if self.procs.alias_scopes:
            self._unbind_arrays(self.procs.alias_scopes.pop())
        if self.procs.array_scopes:
            before = self.procs.array_scopes.pop()
            for arr in list(self.arrays):
                if arr not in before and arr not in self.symbols.shared:
                    self._drop_array(arr)
        if self.procs.type_scopes:
            shared_types = {k: v for k, v in self.symbols.var_types.items() if k in self.symbols.shared}
            self.symbols.var_types.clear()
            self.symbols.var_types.update(self.procs.type_scopes.pop())
            self.symbols.var_types.update(shared_types)
        if self.procs.scopes:
            shared = self.procs.shared_scopes.pop() if self.procs.shared_scopes else set()
            shared_vals = {v: self.variables[v] for v in shared if v in self.variables}
            self.variables.clear()
            self.variables.update(self.procs.scopes.pop())
            self.variables.update(shared_vals)

    # ── Array parameters ───────────────────────────────────────────────

    def _array_entry(self, name: str) -> tuple | None:
        if name not in self.arrays:
            return None
        return (self.arrays[name], self.symbols.array_dims.get(name), self.symbols.array_lb.get(name),
                self.symbols.array_types.get(name), name in self.symbols.dimmed)

    def _set_array_entry(self, name: str, entry: tuple | None) -> None:
        self._drop_array(name)
        if entry is None:
            return
        data, dims, lows, typ, dimmed = entry
        self.arrays[name] = data
        if dims is not None:
            self.symbols.array_dims[name] = dims
        if lows is not None:
            self.symbols.array_lb[name] = lows
        if typ is not None:
            self.symbols.array_types[name] = typ
        if dimmed:
            self.symbols.dimmed.add(name)

    def _bind_array(self, param: str, source: str) -> None:
        """Make array parameter `param` the caller's array `source`."""
        if source not in self.arrays:
            raise ValueError(f"SUBSCRIPT OUT OF RANGE: array {source} is not dimensioned")
        if param == source:
            return
        self.procs.alias_scopes[-1].append((param, source, self._array_entry(param)))
        self._set_array_entry(param, self._array_entry(source))

    def _unbind_arrays(self, aliases: list[tuple[str, str, Any]]) -> None:
        """Write each array parameter back to its caller's array (a REDIM
        inside the procedure resizes the caller's array) and restore whatever
        the parameter's name held before the call."""
        for param, source, previous in reversed(aliases):
            self._set_array_entry(source, self._array_entry(param))
            self._set_array_entry(param, previous)

    # ── Calls ──────────────────────────────────────────────────────────

    def _bind_params(self, name: str, params: list[str], args: list[Any]) -> None:
        if len(args) != len(params):
            raise RuntimeError(f"ARGUMENT-COUNT MISMATCH: {name} takes {len(params)} "
                               f"argument(s), got {len(args)}")
        for text, val in zip(params, args):
            p = parse_param(text)
            if p.array:
                if not isinstance(val, ArrayRef):
                    raise ValueError(f"TYPE MISMATCH: {name} parameter {p.name}() takes an "
                                     f"array, as in {name} arr()")
                self._bind_array(p.name, val.name)
                continue
            if isinstance(val, ArrayRef):
                raise ValueError(f"TYPE MISMATCH: {name} parameter {p.name} takes a value, "
                                 f"not the array {val.name}()")
            if p.type is not None:
                self._check_type(p.type)
                self.symbols.var_types[p.name] = p.type
            self.variables[p.name] = self._coerce(p.name, val)
        for k, v in self.procs.statics.get(name, {}).items():
            self.variables[k] = v

    def _call_args(self, params: list[str], arg_texts: list[str], run_vars: Any) -> list[Any]:
        """Argument values for a call: ArrayRef for a whole array (`a()`, or
        a bare array name for an array parameter), else the evaluated value."""
        decl = [parse_param(t) for t in params]
        args: list[Any] = []
        for i, text in enumerate(arg_texts):
            m = _RE_WHOLE_ARRAY.match(text)
            bare = text.strip()
            if m and m.group(1) in self.arrays:
                args.append(ArrayRef(m.group(1)))
            elif i < len(decl) and decl[i].array and bare in self.arrays:
                args.append(ArrayRef(bare))
            else:
                args.append(self._safe_eval(text, extra_ns=run_vars))
        return args

    # ── Control flow handlers ──────────────────────────────────────────

    def _cf_sub(self, stmt: str, sorted_lines: Sequence[int], ip: int, *, parsed: SubStmt) -> tuple[bool, ExecOutcome]:
        """SUB — skip the block during normal execution."""
        name = parsed.name
        if name in self.procs.subs:
            return True, self.procs.subs[name]['end_ip'] + 1
        return True, ExecResult.ADVANCE

    def _cf_end_sub(self, stmt: str, *, parsed: EndSubStmt) -> tuple[bool, ExecOutcome]:
        if self.procs.calls:
            frame = self.procs.calls.pop()
            back = [(lv, self.variables.get(p)) for p, lv in frame.get('byref', ())]
            self._pop_scope(frame)
            for lv, value in back:
                if value is not None:
                    self._write_lvalue(lv, value)
            return True, frame['return_ip']
        return True, ExecResult.ADVANCE

    def _cf_function(self, stmt: str, sorted_lines: Sequence[int], ip: int, *, parsed: FunctionStmt) -> tuple[bool, ExecOutcome]:
        """FUNCTION — skip the block during normal execution."""
        name = parsed.name
        if name in self.procs.functions:
            return True, self.procs.functions[name]['end_ip'] + 1
        return True, ExecResult.ADVANCE

    def _cf_end_function(self, stmt: str, *, parsed: EndFunctionStmt) -> tuple[bool, ExecOutcome]:
        return True, ExecResult.ADVANCE

    def _cf_call(self, stmt: str, run_vars: MutableMapping[str, Any],
                 sorted_lines: Sequence[int], ip: int, *, parsed: CallStmt) -> tuple[bool, ExecOutcome]:
        name, arg_str = parsed.name, parsed.args
        if name not in self.procs.subs:
            if name in ('ABSOLUTE', 'INTERRUPT', 'INTERRUPTX'):
                raise RuntimeError(f"ILLEGAL FUNCTION CALL: CALL {name} runs machine code, "
                                   f"which QUBASIC does not execute")
            raise RuntimeError(f"SUBPROGRAM NOT DEFINED: {name}")
        sub = self.procs.subs[name]
        arg_texts = self._split_arg_list(arg_str) if arg_str.strip() else []
        args = self._call_args(sub['params'], arg_texts, run_vars)
        names = [parse_param(t).name for t in sub['params']]
        byref = []
        for pname, text, val in zip(names, arg_texts, args):
            if isinstance(val, ArrayRef):
                continue
            if (_names_storage(text) and not self._is_function_name(text)
                    and text.strip() not in self.symbols.consts):
                byref.append((pname, self._lvalue(text, run_vars)))
        self._push_scope()
        self.procs.calls.append({'return_ip': ip + 1, 'sub_name': name, 'byref': byref,
                                 'params': names, 'static': sub.get('static')})
        try:
            self._bind_params(name, sub['params'], args)
        except Exception:
            self._pop_scope(self.procs.calls.pop())
            raise
        return True, sub['start_ip']

    def _is_function_name(self, text: str) -> bool:
        """Whether an argument like f(x) calls a function rather than naming
        an array element (functions are passed by value)."""
        head = text.strip().split('(', 1)[0].strip()
        up = head.upper()
        if head in self.arrays:
            return False
        if '(' not in text:
            return False
        return (up in self.procs.functions or up in self.symbols.fns or ('FN' + up) in self.symbols.fns
                or self._builtin(up) is not None)

    def _invoke_function(self, name: str, args: list[Any], prog: Any) -> Any:
        """Execute a FUNCTION block and return its value (called from the
        expression evaluator). Its statements run through control-flow
        dispatch, one compiled statement at a time."""
        self.procs.depth += 1
        if self.procs.depth > 100:
            self.procs.depth -= 1
            raise RuntimeError("FUNCTION CALL DEPTH LIMIT (100)")
        try:
            prog = self._as_program(prog)
            uname = name.upper()
            if uname not in self.procs.functions:
                raise ValueError(f"FUNCTION NOT DEFINED: {name}")
            func = self.procs.functions[uname]
            self._push_scope()
            frame = {'func_name': uname, 'static': func.get('static'),
                     'params': [parse_param(t).name for t in func['params']]}
            try:
                self._bind_params(uname, func['params'], list(args))
                ctx = ExecContext(
                    sorted_lines=prog,
                    ip=func['start_ip'],
                    run_vars=self.variables,
                    loop_stack=[],
                    max_iterations=self._loop_limit(),
                )

                def _no_clause(s: str, ls: list[dict[str, Any]], sl: Sequence[int], i: int, rv: MutableMapping[str, Any]) -> None:
                    return None

                self.procs.calls.append(frame)
                depth = len(self.procs.calls)
                try:
                    # Run to this FUNCTION's END FUNCTION; a SUB it calls runs
                    # wherever it sits and returns here.
                    while ctx.ip < len(prog) and not (
                            ctx.ip == func['end_ip'] and len(self.procs.calls) == depth):
                        ctx.iteration_count += 1
                        if ctx.iteration_count > ctx.max_iterations:
                            raise RuntimeError("FUNCTION LOOP LIMIT")
                        ip = ctx.ip
                        handled, result = self._exec_control_flow(
                            prog.texts[ip], ctx.loop_stack, prog, ip, ctx.run_vars, _no_clause,
                            parsed=prog.stmts[ip])
                        if not handled:
                            if not self._try_stmt_handlers(prog.texts[ip], None, ctx.run_vars):
                                raise RuntimeError(
                                    f"'{prog.texts[ip]}' cannot run inside FUNCTION {name}: "
                                    f"a FUNCTION runs classical statements (use a SUB for gates)")
                            result = ExecResult.ADVANCE
                        if result is ExecResult.END:
                            break
                        if isinstance(result, int) and not isinstance(result, bool):
                            ctx.ip = result
                        else:
                            ctx.ip += 1
                finally:
                    del self.procs.calls[depth - 1:]
                ret_val = next((v for k, v in self.variables.items() if k.upper() == uname),
                               '' if uname.endswith('$') else 0)
            finally:
                self._pop_scope(frame)
            return ret_val
        finally:
            self.procs.depth -= 1

    # ── LOCAL / STATIC / SHARED ────────────────────────────────────────

    def _declared_names(self, var_list: str) -> list[Param]:
        return [parse_param(v) for v in self._split_arg_list(var_list)]

    def _cf_local(self, stmt: str, run_vars: MutableMapping[str, Any], *, parsed: LocalStmt) -> tuple[bool, ExecOutcome]:
        for p in self._declared_names(parsed.var_list):
            if p.type is not None:
                self.symbols.var_types[p.name] = p.type
            value = self._default_value(p.name)
            run_vars[p.name] = value
            self.variables[p.name] = value
        return True, ExecResult.ADVANCE

    def _cf_static(self, stmt: str, run_vars: MutableMapping[str, Any], *, parsed: StaticStmt) -> tuple[bool, ExecOutcome]:
        var_list = parsed.var_list
        frame = self.procs.calls[-1] if self.procs.calls else {}
        sub_name = frame.get('sub_name') or frame.get('func_name') or '_GLOBAL'
        if sub_name not in self.procs.statics:
            self.procs.statics[sub_name] = {}
        store = self.procs.statics[sub_name]
        for p in self._declared_names(var_list):
            if p.array:
                if self.procs.array_scopes:
                    self.procs.array_scopes[-1].add(p.name)       # keeps its elements between calls
                continue
            if p.type is not None:
                self.symbols.var_types[p.name] = p.type
            val = store.get(p.name, self._default_value(p.name))
            store[p.name] = val
            run_vars[p.name] = val
            self.variables[p.name] = val
        return True, ExecResult.ADVANCE

    def _cf_shared(self, stmt: str, run_vars: MutableMapping[str, Any], *, parsed: SharedStmt) -> tuple[bool, ExecOutcome]:
        for p in self._declared_names(parsed.var_list):
            if p.array:
                if self.procs.array_scopes:
                    self.procs.array_scopes[-1].add(p.name)       # the caller's array, not a local
                continue
            if not self.procs.scopes:
                continue
            if p.type is not None:
                self.symbols.var_types[p.name] = p.type
            caller = self.procs.scopes[-1]
            value = caller.get(p.name, self._default_value(p.name))
            run_vars[p.name] = value
            self.variables[p.name] = value
            if self.procs.shared_scopes:
                self.procs.shared_scopes[-1].add(p.name)
        return True, ExecResult.ADVANCE


# ---------------------------------------------------------------------------
# QUBASIC debug — ON ERROR, TRON/TROFF, STOP/CONT, breakpoints, watch, ASSERT, callbacks.
# ---------------------------------------------------------------------------

MAX_SV_CHECKPOINTS = 200
# Time-travel history: one checkpoint may hold at most CHECKPOINT_MAX_BYTES
# (a 22-qubit double-precision state), and the history keeps the newest
# checkpoints within CHECKPOINT_BUDGET_BYTES.
CHECKPOINT_MAX_BYTES = 64 << 20
CHECKPOINT_BUDGET_BYTES = 256 << 20

# QBASIC's error numbers, matched against the start of an error message.
_ERROR_CODES = (
    ('NEXT WITHOUT FOR', 1), ('SYNTAX ERROR', 2), ('RETURN WITHOUT GOSUB', 3),
    ('OUT OF DATA', 4), ('ILLEGAL FUNCTION CALL', 5), ('OVERFLOW', 6),
    ('OUT OF MEMORY', 7), ('SUBSCRIPT OUT OF RANGE', 9), ('DUPLICATE DEFINITION', 10),
    ('DIVISION BY ZERO', 11), ('TYPE MISMATCH', 13), ("CAN'T CONTINUE", 17),
    ('FUNCTION NOT DEFINED', 18), ('NOT A FUNCTION', 18), ('RESUME WITHOUT ERROR', 20),
    ('FOR WITHOUT NEXT', 26), ('WHILE WITHOUT WEND', 29), ('WEND WITHOUT WHILE', 30),
    ('DUPLICATE LABEL', 33), ('SUBPROGRAM NOT DEFINED', 35),
    ('ARGUMENT-COUNT MISMATCH', 37), ('FIELD OVERFLOW', 50), ('BAD FILE NAME OR NUMBER', 52),
    ('FILE NOT FOUND', 53), ('BAD FILE MODE', 54), ('FILE ALREADY OPEN', 55),
    ('FILE ALREADY EXISTS', 58), ('BAD RECORD LENGTH', 59), ('INPUT PAST END OF FILE', 62),
    ('BAD RECORD NUMBER', 63), ('BAD FILE NAME', 64), ('PERMISSION DENIED', 70),
    ('PATH/FILE ACCESS ERROR', 75), ('PATH NOT FOUND', 76),
    ('LABEL', 8), ('MATH DOMAIN ERROR', 5),
)


def error_code(err: Exception) -> int:
    """The ERR value for an error: the number ERROR n raised, else QBASIC's
    number for the message (1 when none applies)."""
    msg = str(err).upper()
    m = re.match(r'(?:LINE \d+:\s*)?ERROR (\d+)(?!\w)', msg)
    if m:
        return int(m.group(1))
    msg = re.sub(r'^LINE \d+:\s*', '', msg)
    msg = re.sub(r'^[A-Z$]+: ', '', msg) if not any(msg.startswith(k) for k, _ in _ERROR_CODES) else msg
    for key, code in _ERROR_CODES:
        if msg.startswith(key):
            return code
    return 1


@dataclass
class EventState:
    """Event trapping: ON TIMER, ON MEASURE, ON KEY(n) and ON PLAY(n) handlers
    with their ON | OFF | STOP states and the events STOP holds, the keys
    KEY 15-25 define, the handler running for each kind, and the keys read
    from the terminal that INKEY$ and GET have yet to take."""

    timer_target: int | None = None
    timer_interval: float = 0.0
    timer_last: float = 0.0
    timer_state: str = 'OFF'                                      # TIMER ON | OFF | STOP
    timer_pending: bool = False                                   # an event held by TIMER STOP
    measure_target: int | None = None
    measure_fired: bool = False
    key_traps: dict[int, int] = field(default_factory=dict)       # ON KEY(n) handlers
    key_state: dict[int, str] = field(default_factory=dict)       # KEY(n) ON | OFF | STOP
    key_pending: set[int] = field(default_factory=set)            # key events held by STOP
    user_keys: dict[int, set[str]] = field(default_factory=dict)  # keys KEY 15-25 define
    play_trap: tuple[int, int] | None = None                      # ON PLAY(n): (queue size, handler)
    play_state: str = 'OFF'
    play_pending: bool = False
    play_last: int = 0
    depth: dict[str, int] = field(default_factory=dict)           # kind -> GOSUB depth of its running handler
    key_buffer: list[str] = field(default_factory=list)           # keys read, not yet taken
    read_keys: Callable[..., list[str]] = read_terminal_keys


@dataclass
class ErrorState:
    """ON ERROR: the handler's target, and ERR, ERL and the position RESUME
    returns to while a handler runs."""

    target: int | None = None  # ON ERROR GOTO's line
    code: int = 0              # ERR
    line: int = 0              # ERL
    ip: int | None = None      # where RESUME returns
    handling: bool = False


@dataclass
class DebugState:
    """The debugger: TRON, STOP and CONT positions, breakpoints, WATCH
    expressions, and the statevector checkpoints REWIND and FORWARD move
    through."""

    trace: bool = False                                               # TRON
    stopped_ip: int | None = None                                     # where STOP or a breakpoint stopped
    skip_stop_ip: int | None = None                                   # the STOP that CONT steps past
    breakpoints: set[int] = field(default_factory=set)
    watches: list[str] = field(default_factory=list)
    checkpoints: list[tuple[int, Any]] = field(default_factory=list)  # (line, statevector)
    position: int = -1                                                # the checkpoint REWIND and FORWARD are at


class DebugMixin(LanguageHost):
    """Error handling, debugging, and event callbacks for QBasicTerminal."""

    def _init_debug(self) -> None:
        self.errors = ErrorState()
        self.debug = DebugState()
        self.events = EventState()

    # ── Error handling ─────────────────────────────────────────────────

    def _cf_on_error(self, stmt: str, *, parsed: OnErrorStmt) -> tuple[bool, ExecOutcome]:
        target = parsed.target
        if isinstance(target, str):             # labels resolve when the program compiles
            raise ValueError(f"LABEL {target} NOT FOUND")
        if target == 0:
            self.errors.target = None
        else:
            self.errors.target = target
        return True, ExecResult.ADVANCE

    def _cf_resume(self, stmt: str, sorted_lines: Sequence[int], *, parsed: ResumeStmt) -> tuple[bool, ExecOutcome]:
        self.errors.handling = False
        arg = parsed.arg
        from qubasic_core.parser import line_index
        err_ip = self.errors.ip if self.errors.ip is not None else line_index(sorted_lines, self.errors.line)
        if arg is None or arg.strip() in ('', '0'):
            # RESUME (or RESUME 0) — retry the statement that caused the error
            return True, err_ip if err_ip is not None else ExecResult.ADVANCE
        if arg.strip().upper() == 'NEXT':
            # RESUME NEXT — continue with the statement after it
            return True, err_ip + 1 if err_ip is not None else ExecResult.ADVANCE
        # RESUME <line>
        target = int(arg.strip())
        ip = line_index(sorted_lines, target)
        if ip is None:
            raise RuntimeError(f"RESUME: LINE {target} NOT FOUND")
        return True, ip

    def _cf_error(self, stmt: str, *, parsed: ErrorStmt) -> tuple[bool, ExecOutcome]:
        raise RuntimeError(f"ERROR {parsed.code}")

    def _handle_error(self, err: Exception, line_num: int,
                      sorted_lines: Any, ip: int | None = None) -> int | None:
        """Handle a runtime error at statement ip (line line_num). Returns the
        statement to jump to, or None to propagate."""
        if self.errors.target is not None and not self.errors.handling:
            self.errors.code = error_code(err)
            self.errors.line = line_num
            self.errors.ip = ip
            self.errors.handling = True
            self.variables['ERR'] = self.errors.code
            self.variables['ERL'] = self.errors.line
            from qubasic_core.parser import line_index
            return line_index(sorted_lines, self.errors.target)
        return None

    # ── ASSERT ─────────────────────────────────────────────────────────

    def _cf_assert(self, stmt: str, run_vars: MutableMapping[str, Any], *, parsed: AssertStmt) -> tuple[bool, ExecOutcome]:
        cond = parsed.condition
        if not self._eval_condition(cond, run_vars):
            raise RuntimeError(f"ASSERTION FAILED: {cond}")
        return True, ExecResult.ADVANCE

    # ── Time-travel debugging ─────────────────────────────────────────

    def _checkpoint_sv(self, line_num: int) -> None:
        """Save a statevector checkpoint for REWIND/FORWARD.

        A state up to CHECKPOINT_MAX_BYTES (22 qubits at double precision) is
        kept; the oldest checkpoints are dropped beyond MAX_SV_CHECKPOINTS
        entries or CHECKPOINT_BUDGET_BYTES in total.
        """
        sv = self._last_sv
        if sv is None or sv.nbytes > CHECKPOINT_MAX_BYTES:
            return
        import numpy as np
        self.debug.checkpoints.append((line_num, np.array(sv, copy=True)))
        total = sum(c.nbytes for _, c in self.debug.checkpoints)
        while self.debug.checkpoints and (len(self.debug.checkpoints) > MAX_SV_CHECKPOINTS
                                        or total > CHECKPOINT_BUDGET_BYTES):
            total -= self.debug.checkpoints.pop(0)[1].nbytes
        self.debug.position = len(self.debug.checkpoints) - 1


    def cmd_history(self, rest: str = '') -> None:
        """HISTORY — show statevector checkpoint list."""
        if not self.debug.checkpoints:
            self.io.writeln("?NO CHECKPOINTS")
            return
        for i, (ln, _) in enumerate(self.debug.checkpoints):
            marker = " <<" if i == self.debug.position else ""
            self.io.writeln(f"  [{i}] line {ln}{marker}")

    # ── TRON / TROFF ───────────────────────────────────────────────────

    def cmd_tron(self) -> None:
        """TRON — trace on, print each line number during execution."""
        self.debug.trace = True
        self.io.writeln("TRACE ON")

    def cmd_troff(self) -> None:
        """TROFF — stop printing line numbers as the program runs."""
        self.debug.trace = False
        self.io.writeln("TRACE OFF")

    def _trace_line(self, line_num: int) -> None:
        """Print trace output if trace mode is on."""
        if self.debug.trace:
            self.io.write(f"[{line_num}]" + ' ')

    # ── STOP / CONT ────────────────────────────────────────────────────

    def _cf_run(self, sorted_lines: Sequence[int], loop_stack: list[dict[str, Any]], *,
                parsed: RunStmt) -> tuple[bool, ExecOutcome]:
        """RUN [line | label] inside a program: start again from the top or
        that line, with its variables, arrays, files, DATA and traps cleared
        and any gates it applied removed."""
        if parsed.file is not None:
            raise ValueError("RUN file$ inside a program: use CHAIN file$")
        if self._live is not None:
            raise ValueError("RUN inside a program that runs shot by shot")
        self._clear_program_state()
        loop_stack.clear()
        self.data.ptr = 0
        self._reset_events()
        if self._build_qc is not None:
            self._build_qc.clear()
        self._classical_bits.clear()
        if parsed.target is None:
            return True, 0
        if isinstance(parsed.target, str):
            raise ValueError(f"LABEL {parsed.target} NOT FOUND")
        from qubasic_core.parser import line_index
        target = line_index(sorted_lines, parsed.target)
        if target is None:
            raise ValueError(f"LINE {parsed.target} NOT FOUND")
        return True, target

    def _cf_stop(self, stmt: str, sorted_lines: Sequence[int], ip: int, *, parsed: StopStmt) -> tuple[bool, ExecOutcome]:
        # CONT sets _cont_skip_stop_ip to replay past this STOP
        if self.debug.skip_stop_ip is not None and self.debug.skip_stop_ip == ip:
            self.debug.skip_stop_ip = None
            return True, ExecResult.ADVANCE
        line_num = sorted_lines[ip]
        self.io.writeln(f"STOPPED AT LINE {line_num}")
        self.debug.stopped_ip = ip
        self._print_watches()
        return True, ExecResult.END

    def cmd_cont(self) -> None:
        """CONT — continue execution after STOP, resuming from the next line.

        Re-executes the program from the top but skips the STOP that fired,
        so lines after STOP are reached. Variable state is rebuilt by
        replaying all lines before the STOP.
        """
        if self.debug.stopped_ip is None:
            self.io.writeln("?CANNOT CONTINUE")
            return
        self.io.writeln("CONTINUING...")
        self.debug.skip_stop_ip = self.debug.stopped_ip
        self.debug.stopped_ip = None
        self.cmd_run()

    # ── Breakpoints ────────────────────────────────────────────────────

    def cmd_breakpoint(self, rest: str) -> None:
        """BREAK <line> | BREAK CLEAR | BREAK LIST | BREAK <start>-<end> — manage breakpoints."""
        if not rest.strip():
            if self.debug.breakpoints:
                self.io.writeln(f"  Breakpoints: {sorted(self.debug.breakpoints)}")
            else:
                self.io.writeln("  No breakpoints set")
            return
        rest = rest.strip().upper()
        if rest in ('CLEAR', 'ALL'):
            self.debug.breakpoints.clear()
            self.io.writeln("BREAKPOINTS CLEARED")
            return
        if rest == 'LIST':
            if self.debug.breakpoints:
                for bp in sorted(self.debug.breakpoints):
                    src = self.program.get(bp, '(no source)')
                    self.io.writeln(f"  {bp}: {src}")
            else:
                self.io.writeln("  No breakpoints set")
            return
        # Range: BREAK 10-50
        if '-' in rest:
            try:
                a, b = rest.split('-')
                for bp in list(self.debug.breakpoints):
                    if int(a) <= bp <= int(b):
                        self.debug.breakpoints.discard(bp)
                self.io.writeln(f"BREAKPOINTS CLEARED IN {a}-{b}")
                return
            except ValueError:
                pass
        try:
            line = int(rest)
            if line in self.debug.breakpoints:
                self.debug.breakpoints.discard(line)
                self.io.writeln(f"BREAKPOINT REMOVED: {line}")
            else:
                self.debug.breakpoints.add(line)
                self.io.writeln(f"BREAKPOINT SET: {line}")
        except ValueError:
            self.io.writeln("?USAGE: BREAK <line> | BREAK <start>-<end> | BREAK CLEAR | BREAK LIST")

    def _check_breakpoint(self, line_num: int, sorted_lines: Sequence[int], ip: int) -> bool:
        """Check if we should break at this line. Returns True to stop."""
        if line_num in self.debug.breakpoints:
            self.io.writeln(f"BREAKPOINT AT LINE {line_num}")
            self.debug.stopped_ip = ip
            self._print_watches()
            return True
        return False

    # ── Watch expressions ──────────────────────────────────────────────

    def cmd_watch(self, rest: str) -> None:
        """WATCH [expr | CLEAR] — add or remove a watch expression, list them, or clear them."""
        if not rest.strip():
            if self.debug.watches:
                self.io.writeln("  Watch expressions:")
                for w in self.debug.watches:
                    self.io.writeln(f"    {w}")
            else:
                self.io.writeln("  No watches set")
            return
        rest = rest.strip()
        if rest.upper() == 'CLEAR':
            self.debug.watches.clear()
            self.io.writeln("WATCHES CLEARED")
            return
        if rest in self.debug.watches:
            self.debug.watches.remove(rest)
            self.io.writeln(f"WATCH REMOVED: {rest}")
        else:
            self.debug.watches.append(rest)
            self.io.writeln(f"WATCHING: {rest}")

    def _print_watches(self) -> None:
        for w in self.debug.watches:
            try:
                val = self._safe_eval(w)
                self.io.writeln(f"  {w} = {val}")
            except Exception:
                self.io.writeln(f"  {w} = ?")

    # ── Callbacks ──────────────────────────────────────────────────────

    def _cf_on_measure(self, stmt: str, *, parsed: OnMeasureStmt) -> tuple[bool, ExecOutcome]:
        self.events.measure_target = parsed.target
        return True, ExecResult.ADVANCE

    def _cf_on_timer(self, stmt: str, *, parsed: OnTimerStmt) -> tuple[bool, ExecOutcome]:
        if isinstance(parsed.target, str):      # labels resolve when the program compiles
            raise ValueError(f"LABEL {parsed.target} NOT FOUND")
        self.events.timer_interval = float(self._safe_eval(parsed.interval))
        self.events.timer_target = parsed.target
        self.events.timer_last = time.time()
        return True, ExecResult.ADVANCE

    def _stmt_timer(self, stmt: str) -> bool:
        """TIMER ON | OFF | STOP — enable, disable or hold ON TIMER events."""
        m = re.match(r'TIMER\s+(ON|OFF|STOP)\s*$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        self.events.timer_state = m.group(1).upper()
        if self.events.timer_state == 'ON':
            self.events.timer_last = time.time()
        elif self.events.timer_state == 'OFF':
            self.events.timer_pending = False
        return True

    def _stmt_on_event(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """ON KEY(n) GOSUB target and ON PLAY(n) GOSUB target arm a key or a
        background-music trap (GOSUB 0 disarms it); ON COM(n), ON PEN and ON
        STRIG(n) are accepted, and those devices raise no events here."""
        m = re.match(r'ON\s+(KEY|PLAY|COM|PEN|STRIG|UEVENT)\b\s*(?:\((.*?)\))?\s*GOSUB\s+(\S+)\s*$',
                     stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        device = m.group(1).upper()
        if device in ('COM', 'PEN', 'STRIG', 'UEVENT'):
            return True
        if m.group(2) is None:
            raise ValueError(f"SYNTAX ERROR: ON {device}(n) GOSUB line")
        n = int(self._eval_with_vars(m.group(2), run_vars))
        target = self._event_target(m.group(3))
        if device == 'KEY':
            if n not in KEY_TRAP_KEYS and not 15 <= n <= 25:
                raise ValueError(f"ILLEGAL FUNCTION CALL: KEY({n})")
            if target is None:
                self.events.key_traps.pop(n, None)
            else:
                self.events.key_traps[n] = target
        else:
            if not 1 <= n <= 32:
                raise ValueError(f"ILLEGAL FUNCTION CALL: PLAY({n})")
            self.events.play_trap = None if target is None else (n, target)
            self.events.play_last = self._bg_remaining()
        return True

    def _event_target(self, text: str) -> int | None:
        """A trap's handler line, from a line number or a label; 0 disarms."""
        if text.isdigit():
            return int(text) or None
        line = self._compiled().labels.get(text.upper())
        if line is None:
            raise ValueError(f"LABEL {text} NOT FOUND")
        return line

    def _set_play_trap(self, state: str) -> None:
        """PLAY ON | OFF | STOP: arm, disarm or hold the ON PLAY trap."""
        self.events.play_state = state
        if state == 'OFF':
            self.events.play_pending = False

    @staticmethod
    def _stmt_device_events(stmt: str) -> bool:
        """PEN ON|OFF|STOP, STRIG[(n)] ON|OFF|STOP and COM(n) ON|OFF|STOP: accepted."""
        return bool(re.match(r'(PEN|STRIG|COM)\b\s*(?:\(\s*\d+\s*\))?\s*(ON|OFF|STOP)\s*$',
                             stmt.strip(), re.IGNORECASE))

    def _stmt_key(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """KEY(n) ON | OFF | STOP arms, disarms or holds key trap n (KEY(0) every
        trap); KEY n, keys$ defines key 15-25 from shift flags and a scan code,
        or labels function key n; KEY ON | OFF | LIST are accepted."""
        s = stmt.strip()
        m = re.match(r'KEY\s*\((.+?)\)\s*(ON|OFF|STOP)\s*$', s, re.IGNORECASE)
        if m:
            n = int(self._eval_with_vars(m.group(1), run_vars))
            state = m.group(2).upper()
            for k in ([*KEY_TRAP_KEYS, *range(15, 26)] if n == 0 else [n]):
                self.events.key_state[k] = state
                if state == 'OFF':
                    self.events.key_pending.discard(k)
            return True
        m = re.match(r'KEY\s+(\d+)\s*,\s*(.+)$', s, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            text = self._safe_eval(m.group(2), extra_ns=run_vars)
            if not isinstance(text, str):
                raise ValueError("TYPE MISMATCH: KEY n takes a string")
            if 15 <= n <= 25:
                self.events.user_keys[n] = defined_keys(text)
            elif not (1 <= n <= 10 or n in (30, 31)):
                raise ValueError(f"ILLEGAL FUNCTION CALL: KEY {n}")
            return True
        return bool(re.match(r'KEY\s+(ON|OFF|LIST)\s*$', s, re.IGNORECASE))

    def _events_armed(self) -> bool:
        """Whether a timer, key or music trap is armed this run."""
        return self.events.timer_target is not None or bool(self.events.key_traps) or self.events.play_trap is not None

    def _poll_keys(self) -> None:
        """Move the keys waiting at the terminal into the keyboard buffer."""
        if not getattr(self.io, 'replay', False):
            self.events.key_buffer.extend(self.events.read_keys())

    def _trapped_key(self, key: str) -> int | None:
        for n, k in KEY_TRAP_KEYS.items():
            if key == k and n in self.events.key_traps:
                return n
        for n, keys in self.events.user_keys.items():
            if key in keys and n in self.events.key_traps:
                return n
        return None

    def _check_events(self, sorted_lines: Sequence[int], ip: int) -> int | None:
        """The ip of the handler an event calls before the statement at ip
        runs, ip pushed for its RETURN, or None. An event waits while its own
        handler runs, as QBASIC holds it until RETURN."""
        depth = len(self._gosub_stack)
        self.events.depth = {k: d for k, d in self.events.depth.items() if d <= depth}
        if self.events.timer_target is not None and 'timer' not in self.events.depth:
            target = self._check_timer_callback(sorted_lines, ip)
            if target is not None:
                self.events.depth['timer'] = len(self._gosub_stack)
                return target
        if self.events.key_traps:
            self._poll_keys()
            for key in list(self.events.key_buffer):
                n = self._trapped_key(key)
                if n is not None and self.events.key_state.get(n, 'OFF') != 'OFF':
                    self.events.key_buffer.remove(key)
                    self.events.key_pending.add(n)
            for n in sorted(self.events.key_pending):
                if self.events.key_state.get(n) == 'ON' and 'key' not in self.events.depth:
                    self.events.key_pending.discard(n)
                    return self._fire_event('key', self.events.key_traps[n], sorted_lines, ip)
        if self.events.play_trap is not None:
            n, line = self.events.play_trap
            count = self._bg_remaining()
            if count < n <= self.events.play_last and self.events.play_state != 'OFF':
                self.events.play_pending = True
            self.events.play_last = count
            if self.events.play_pending and self.events.play_state == 'ON' and 'play' not in self.events.depth:
                self.events.play_pending = False
                return self._fire_event('play', line, sorted_lines, ip)
        return None

    def _fire_event(self, kind: str, line: int, sorted_lines: Sequence[int], ip: int) -> int:
        from qubasic_core.parser import line_index
        target = line_index(sorted_lines, line)
        if target is None:
            raise ValueError(f"LINE {line} NOT FOUND")
        self._gosub_stack.append(ip)
        self.events.depth[kind] = len(self._gosub_stack)
        return target

    def _reset_events(self) -> None:
        """A run's traps arm only when its own statements arm them."""
        self.events.timer_target = None
        self.events.timer_state = 'OFF'
        self.events.timer_pending = False
        self.events.key_traps.clear()
        self.events.key_state.clear()
        self.events.key_pending.clear()
        self.events.user_keys.clear()
        self.events.play_trap = None
        self.events.play_state = 'OFF'
        self.events.play_pending = False
        self.events.play_last = 0
        self.events.depth.clear()

    def _check_timer_callback(self, sorted_lines: Sequence[int], ip: int) -> int | None:
        """The ip of the ON TIMER handler when its interval has passed and
        TIMER ON is in force (TIMER STOP holds the event until TIMER ON),
        else None."""
        if self.events.timer_target is not None and self.events.timer_interval > 0:
            now = time.time()
            due = now - self.events.timer_last >= self.events.timer_interval
            if self.events.timer_state != 'ON':
                if due and self.events.timer_state == 'STOP':
                    self.events.timer_pending = True
                    self.events.timer_last = now
                return None
            if due or self.events.timer_pending:
                self.events.timer_pending = False
                self.events.timer_last = now
                from qubasic_core.parser import line_index
                target = line_index(sorted_lines, self.events.timer_target)
                if target is not None:
                    self._gosub_stack.append(ip)
                    return target
        return None


# ---------------------------------------------------------------------------
# QUBASIC keyword statements of the language: arrays, input, file I/O, the
# screen, graphics and sound statements and the event traps. The statements
# on qubits are QuantumStatementMixin's (circuits.py).
# ---------------------------------------------------------------------------

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


def _parse_input_number(text: str) -> int | float | None:
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


def merge_statements(*tables: dict[str, tuple[Any, ...]]) -> dict[str, tuple[Any, ...]]:
    """One keyword-statement table from each layer's: a keyword's handlers in
    table order, the first table's first."""
    merged: dict[str, tuple[Any, ...]] = {}
    for table in tables:
        for word, handlers in table.items():
            merged[word] = merged.get(word, ()) + handlers
    return merged


class StatementMixin(LanguageHost):
    """The language's keyword statements, registered in _LANGUAGE_STATEMENTS:
    UNITARY, DIM, REDIM, ERASE, CONST, the DEF types, GET, INPUT and LINE
    INPUT, file I/O, graphics, PLAY and SOUND, PRINT USING, WRITE, the screen
    statements and the event traps. _try_stmt_handlers dispatches every
    layer's statements by first word through the terminal's _STMT_KEYWORDS."""


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

    def _try_exec_get(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """Handle GET var — single keypress input without enter.

        Falls back to reading one character from self.io.read_line
        when not running in a terminal (batch mode, tests, etc.).
        """
        m = RE_GET.match(stmt)
        if not m:
            return False
        var = m.group(1)
        ch = ''
        self._gfx_frame(start=True, force=True)
        try:
            if getattr(self.io, 'replay', False):
                ch = self.io.read_line('')          # a later live shot replays
            elif self.events.key_buffer or sys.stdin.isatty():
                self._poll_keys()
                if not self.events.key_buffer:
                    self.events.key_buffer.extend(self.events.read_keys(True))
                ch = self.events.key_buffer.pop(0) if self.events.key_buffer else ''
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
            run_vars[var] = float(ord(ch[0])) if ch else 0.0
            self.variables[var] = run_vars[var]
        return True

    def _try_exec_input(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
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
        self._gfx_frame(start=True, force=True)
        for _attempt in range(3):
            try:
                line = self.io.read_line(prompt)
            except (EOFError, KeyboardInterrupt):
                break
            values = self._input_values(line, targets)
            if values is not None:
                for lv, value in zip(targets, values):
                    self._write_lvalue(lv, value, run_vars)
                # INPUT ; keeps the cursor after the typed text, else a new line
                self.screen.col = len(prompt) + len(line) if m.group(1) else 0
                if self.graphics.canvas is not None:
                    self._gfx_print(prompt + line, newline=not m.group(1))
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
        for lv, typed in zip(targets, fields):
            if is_string_kind(self._lvalue_kind(lv)):
                values.append(typed)
                continue
            text = typed.strip()
            if not text:
                values.append(0)
                continue
            num = _parse_input_number(text)
            if num is None:
                return None
            values.append(num)
        return values


    # The language's keyword statements: the first word selects the handlers,
    # each (self, stmt, qc, run_vars, backend) -> handled. The terminal merges
    # every layer's table into _STMT_KEYWORDS (merge_statements).
    _LANGUAGE_STATEMENTS: dict[str, tuple[Any, ...]] = {
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
        'GET': (lambda s, st, qc, rv, b: s._gfx_get(st, rv),
                lambda s, st, qc, rv, b: s._stmt_get_put_file(st, rv),
                lambda s, st, qc, rv, b: s._try_exec_get(st, rv)),
        'INPUT': (lambda s, st, qc, rv, b: s._exec_input_file(st, rv),
                  lambda s, st, qc, rv, b: s._try_exec_input(st, rv)),
        'LINE': (lambda s, st, qc, rv, b: s._try_exec_line_input(st, rv),
                 lambda s, st, qc, rv, b: s._gfx_line(st, rv)),
        **dict.fromkeys(('PSET', 'PRESET'), (lambda s, st, qc, rv, b: s._gfx_pset(st, rv),)),
        'CIRCLE': (lambda s, st, qc, rv, b: s._gfx_circle(st, rv),),
        'PAINT': (lambda s, st, qc, rv, b: s._gfx_paint(st, rv),),
        'DRAW': (lambda s, st, qc, rv, b: s._gfx_draw(st, rv),),
        'VIEW': (lambda s, st, qc, rv, b: s._gfx_view(st, rv),),
        'WINDOW': (lambda s, st, qc, rv, b: s._gfx_window(st, rv),),
        'PALETTE': (lambda s, st, qc, rv, b: s._gfx_palette(st, rv),),
        'PCOPY': (lambda s, st, qc, rv, b: s._gfx_pcopy(st, rv),),
        'PLAY': (lambda s, st, qc, rv, b: s._stmt_play(st, rv),),
        'SOUND': (lambda s, st, qc, rv, b: s._stmt_sound(st, rv),),
        'PUT': (lambda s, st, qc, rv, b: s._gfx_put(st, rv),
                lambda s, st, qc, rv, b: s._stmt_get_put_file(st, rv)),
        'FIELD': (lambda s, st, qc, rv, b: s._stmt_field(st, rv),),
        **dict.fromkeys(('LSET', 'RSET'), (lambda s, st, qc, rv, b: s._stmt_lset_rset(st, rv),)),
        'SEEK': (lambda s, st, qc, rv, b: s._stmt_seek(st, rv),),
        **dict.fromkeys(('KILL', 'NAME', 'MKDIR', 'RMDIR', 'CHDIR', 'FILES'), (
            lambda s, st, qc, rv, b: s._stmt_filesystem(st, rv),)),
        'COMMON': (lambda s, st, qc, rv, b: s._stmt_common(st),),
        'OUT': (lambda s, st, qc, rv, b: s._stmt_out(st, rv),),
        'DEF': (lambda s, st, qc, rv, b: s._stmt_def_seg(st, rv),),
        'BSAVE': (lambda s, st, qc, rv, b: s._stmt_bsave(st, rv),),
        'BLOAD': (lambda s, st, qc, rv, b: s._stmt_bload(st, rv),),
        'CLEAR': (lambda s, st, qc, rv, b: s._stmt_clear(st),),
        'RESET': (lambda s, st, qc, rv, b: s._stmt_reset_files(st),),
        **dict.fromkeys(('LOCK', 'UNLOCK'), (lambda s, st, qc, rv, b: s._stmt_lock(st, rv),)),
        'UEVENT': (lambda s, st, qc, rv, b: bool(re.match(r'UEVENT\s+(ON|OFF|STOP)\s*$', st.strip(), re.I)),),
        'SHELL': (lambda s, st, qc, rv, b: s._stmt_shell(st, rv),),
        'ENVIRON': (lambda s, st, qc, rv, b: s._stmt_environ(st, rv),),
        'TIMER': (lambda s, st, qc, rv, b: s._stmt_timer(st),),
        'KEY': (lambda s, st, qc, rv, b: s._stmt_key(st, rv),),
        'ON': (lambda s, st, qc, rv, b: s._stmt_on_event(st, rv),),
        **dict.fromkeys(('PEN', 'STRIG', 'COM'), (lambda s, st, qc, rv, b: s._stmt_device_events(st),)),
        'PRINT': (lambda s, st, qc, rv, b: s._exec_print_file(st, rv),),
        'LPRINT': (lambda s, st, qc, rv, b: s._exec_lprint(st, rv),),
        'LET': (lambda s, st, qc, rv, b: s._try_exec_let_str(st, rv),),
        **dict.fromkeys(('OPEN', 'CLOSE'), (
            lambda s, st, qc, rv, b: s._stmt_open_close(st, rv),)),
    }

    def _try_stmt_handlers(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
        """Run the keyword statement `stmt`; False when no handler takes it."""
        word = re.match(r'\s*([A-Za-z_]\w*)', stmt)
        if not word:
            return False
        for handler in self._STMT_KEYWORDS.get(word.group(1).upper(), ()):
            if handler(self, stmt, qc, run_vars, backend):
                return True
        return False


    def _try_exec_line_input(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """LINE INPUT [;]["prompt"{;|,}] var$ — read a whole line, commas and
        quotes included. The prompt prints as written (no "?")."""
        if re.match(r'LINE\s+INPUT\s*#', stmt, re.IGNORECASE):
            return self._exec_line_input_file(stmt, run_vars)
        m = re.match(r'LINE\s+INPUT\s*(;)?\s*(?:"([^"]*)"\s*[;,])?\s*(.+)$',
                     stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        lv = self._lvalue(m.group(3), run_vars)
        self._gfx_frame(start=True, force=True)
        try:
            val = self.io.read_line(m.group(2) or '')
        except (EOFError, KeyboardInterrupt):
            val = ''
        self._write_lvalue(lv, val, run_vars)
        self.screen.col = len(m.group(2) or '') + len(val) if m.group(1) else 0
        if self.graphics.canvas is not None:
            self._gfx_print((m.group(2) or '') + val, newline=not m.group(1))
        return True

    def _try_exec_let_str(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = RE_LET_STR.match(stmt)
        if not m:
            return False
        name = m.group(1)
        val = self._eval_string_expr(m.group(2), run_vars)
        run_vars[name] = val
        self.variables[name] = val
        return True

    def _try_exec_print_using(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """PRINT USING fmt$; values (see basic_format)."""
        from qubasic_core.parser import PrintUsingStmt, parse_stmt
        parsed = parse_stmt(stmt)
        if not isinstance(parsed, PrintUsingStmt):
            return False
        self._emit_using(self._using_text(parsed.fmt, parsed.values, run_vars))
        return True

    # ── WRITE and the screen statements ───────────────────────────────

    def _stmt_write(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """WRITE [#n,] values — comma-separated, strings in quotes, numbers
        without padding (WRITE 1, "a" gives 1,"a")."""
        m = re.match(r'WRITE\b\s*(?:#\s*([^,]+?)\s*(?:,|$))?\s*(.*)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        items = self._split_arg_list(m.group(2)) if m.group(2).strip() else []
        parts = []
        for item in items:
            v = self._safe_eval(item, extra_ns=run_vars)
            parts.append(f'"{v}"' if isinstance(v, str) else number_text(v))
        text = ','.join(parts)
        if m.group(1) is not None:
            return self._write_to_handle(self._file_number(m.group(1), run_vars), text)
        self._emit_text(text, True)
        return True

    def _stmt_cls(self, stmt: str) -> bool:
        if not re.match(r'CLS\b', stmt.strip(), re.IGNORECASE):
            return False
        self.screen.col = 0
        if self.graphics.canvas is not None:
            self.graphics.canvas.cls()
            self.graphics.row = self.graphics.col = 1
            return True
        self._text_clear()
        if self._on_terminal() and not getattr(self.io, 'replay', False):
            self.io.write('\033[2J\033[H')
        return True

    # QBASIC colour numbers 0-15 as ANSI codes (8-15 are the bright set).
    _ANSI_FG = (30, 34, 32, 36, 31, 35, 33, 37, 90, 94, 92, 96, 91, 95, 93, 97)
    _COLOR_NAMES = {'black': 0, 'blue': 1, 'green': 2, 'cyan': 3, 'red': 4, 'magenta': 5,
                    'brown': 6, 'yellow': 14, 'white': 15, 'gray': 8, 'grey': 8}

    def _stmt_color(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """COLOR [fg][, bg] — QBASIC colour numbers (0-15) or names."""
        m = re.match(r'COLOR\b\s*(.*)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        args = [a.strip() for a in m.group(1).split(',')]
        canvas = self.graphics.canvas
        if canvas is not None:                 # graphics: drawing and background colours
            nums = [None if not a else self._COLOR_NAMES[a.lower()] if a.lower() in self._COLOR_NAMES
                    else int(round(float(self._safe_eval(a, extra_ns=run_vars)))) for a in args[:2]]
            if nums and nums[0] is not None:
                if not 0 <= nums[0] < max(canvas.colors, 16):
                    raise ValueError(f"ILLEGAL FUNCTION CALL: COLOR {nums[0]}")
                canvas.fg = nums[0]
            if len(nums) > 1 and nums[1] is not None:
                canvas.bg = nums[1]
            return True
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

    def _stmt_locate(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """LOCATE [row][, col] — move the cursor: on a graphics screen its text
        cursor, else the terminal's."""
        m = re.match(r'LOCATE\b\s*(.*)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        args = [a.strip() for a in m.group(1).split(',')]
        nums = [int(round(float(self._safe_eval(a, extra_ns=run_vars)))) if a else None
                for a in args[:2]]
        if len(nums) > 1 and nums[1]:
            self.screen.col = nums[1] - 1
        if self.graphics.canvas is not None:
            self._gfx_locate(nums[0] if nums else None, nums[1] if len(nums) > 1 else None)
            return True
        if nums and nums[0] is not None:
            if not 1 <= nums[0] <= 25:
                raise ValueError(f"ILLEGAL FUNCTION CALL: LOCATE row {nums[0]} (1-25)")
            self.screen.row = nums[0]
        if self._on_terminal() and not getattr(self.io, 'replay', False) and nums:
            row = nums[0] or 1
            col = (nums[1] if len(nums) > 1 else None) or 1
            self.io.write(f'\033[{row};{col}H')
        return True

    def _stmt_clear(self, stmt: str) -> bool:
        """CLEAR [, , stack] in a program: every variable and array cleared,
        the files closed and the GOSUB stack emptied, as in QBASIC."""
        if not re.match(r'CLEAR\s*(,.*)?$', stmt.strip(), re.IGNORECASE):
            return False
        self._clear_program_state()
        return True

    def _clear_program_state(self) -> None:
        self.variables.clear()
        self.arrays.clear()
        self.symbols.array_dims.clear()
        self.symbols.dimmed.clear()
        for handle in list(self.files.handles):
            self._close_file(handle)
        self._gosub_stack.clear()

    def _stmt_common(self, stmt: str) -> bool:
        """COMMON [SHARED] [/block/] var, ...: CHAIN keeps every variable here;
        SHARED also opens them to every SUB and FUNCTION, as DIM SHARED does."""
        m = re.match(r'COMMON\b\s*(SHARED\b)?\s*(?:/\w*/)?\s*(.*)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        if m.group(1):
            for item in self._split_arg_list(m.group(2)):
                name = re.match(r'\s*([A-Za-z_]\w*[$%!#&]?)', item)
                if name:
                    self.symbols.shared.add(name.group(1))
        return True

    def _stmt_beep(self, stmt: str) -> bool:
        if stmt.strip().upper() != 'BEEP':
            return False
        if not getattr(self.io, 'replay', False):
            if not self._audio_on():
                self.io.write('\a')
            self._sound_beep()
        return True

    def _stmt_sleep(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """SLEEP [seconds] — pause (later shots of a live run do not wait)."""
        m = re.match(r'SLEEP\b\s*(.*)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        if getattr(self.io, 'replay', False):
            return True
        arg = m.group(1).strip()
        secs = float(self._safe_eval(arg, extra_ns=run_vars)) if arg else 0.0
        self._gfx_frame(start=True, force=True)
        if secs > 0:
            time.sleep(secs)
        return True

    def _stmt_screen(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """SCREEN mode[, ...] inside a program: a QBASIC graphics mode (1, 2,
        7-13) or text (0); 3-5 choose how RUN shows results (see DISPLAY)."""
        m = re.match(r'SCREEN\b\s*(.+)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        first = m.group(1).split(',')[0].strip()
        if not first:
            return True                         # SCREEN , , apage, vpage: one page here
        mode = int(round(float(self._safe_eval(first, extra_ns=run_vars))))
        if mode == 0 or mode in self._GRAPHICS_MODES:
            self._gfx_screen(mode)
        elif 3 <= mode <= 5:
            self._screen_mode = mode
        else:
            raise ValueError(f"ILLEGAL FUNCTION CALL: SCREEN {mode}")
        return True

    @staticmethod
    def _stmt_accept(stmt: str, word: str) -> bool:
        """Statements that declare rather than act: DECLARE (procedures need no
        forward declaration here) and WIDTH (a terminal sets its own width)."""
        return bool(re.match(rf'{word}\b', stmt.strip(), re.IGNORECASE))

    def _inkey(self) -> str:
        """INKEY$: a key already pressed, or "" (never waits); a key without a
        character reads as CHR$(0) and its scan code."""
        self._gfx_frame(start=True)
        io = self.io
        if getattr(io, 'replay', False):
            try:
                return io.read_line('')
            except EOFError:
                return ''
        self._poll_keys()
        ch = self.events.key_buffer.pop(0) if self.events.key_buffer else ''
        if hasattr(io, 'record'):
            io.record(ch)
        return ch


