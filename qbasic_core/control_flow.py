"""Control flow helpers extracted from QBasicTerminal.

Requires: TerminalProtocol (see qbasic_core.protocol).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from qbasic_core.engine import (
    ExecResult, ExecOutcome,
    RE_LET_ARRAY, RE_LET_VAR, RE_PRINT,
    RE_GOTO, RE_GOSUB, RE_FOR, RE_NEXT, RE_WHILE, RE_IF_THEN,
)
from qbasic_core.parser import parse_stmt
from qbasic_core.statements import (
    RawStmt, RemStmt, MeasureStmt, EndStmt, ReturnStmt, WendStmt,
    LetArrayStmt, LetStmt, PrintStmt, GotoStmt, GosubStmt,
    ForStmt, NextStmt, WhileStmt, IfThenStmt,
    DataStmt, ReadStmt, OnGotoStmt, OnGosubStmt,
    SelectCaseStmt, CaseStmt, EndSelectStmt,
    DoStmt, LoopStmt, ExitStmt,
    SwapStmt, DefFnStmt, OptionBaseStmt,
    SubStmt, EndSubStmt, FunctionStmt, EndFunctionStmt, CallStmt,
    LocalStmt, StaticStmt, SharedStmt,
    OnErrorStmt, ResumeStmt, ErrorStmt, AssertStmt, StopStmt,
    OnMeasureStmt, OnTimerStmt,
)


class ControlFlowMixin:
    """Mixin providing control flow helpers for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.variables,
    self.arrays, self.locc_mode, self.locc, self._gosub_stack,
    self._eval_with_vars(), self._eval_condition(), self._substitute_vars().
    """

    # ── Control flow helpers (decomposed from _exec_control_flow) ────
    #
    # Each _cf_* method accepts (self, stmt: str, ..., *, parsed=None).
    # When parsed is provided the method uses the typed fields directly;
    # when it is None the method falls back to regex matching on the raw
    # string.  The raw-string path is retained for backward compatibility
    # but is no longer exercised by the main execution pipeline (callers
    # now always supply a parsed Stmt via _exec_control_flow).
    # Deferred cleanup: the raw-string fallback can be removed once all
    # external call sites are confirmed to pass parsed objects.

    def _cf_let_array(self, stmt: str, run_vars: dict[str, Any],
                      *, parsed: LetArrayStmt | None = None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None:
            m = RE_LET_ARRAY.match(stmt)
            if not m:
                return None
            name, idx_expr, val_expr = m.group(1), m.group(2), m.group(3)
        else:
            name, idx_expr, val_expr = parsed.name, parsed.index_expr, parsed.value_expr
        idx = int(self._eval_with_vars(idx_expr, run_vars))
        val = self._eval_with_vars(val_expr, run_vars)
        if name not in self.arrays:
            self.arrays[name] = [0.0] * (idx + 1)
        while idx >= len(self.arrays[name]):
            self.arrays[name].append(0.0)
        self.arrays[name][idx] = val
        return True, ExecResult.ADVANCE

    def _cf_let_var(self, stmt: str, run_vars: dict[str, Any],
                    *, parsed: LetStmt | None = None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None:
            m = RE_LET_VAR.match(stmt)
            if not m:
                return None
            name, expr = m.group(1), m.group(2)
        else:
            name, expr = parsed.name, parsed.expr
        val = self._eval_with_vars(expr, run_vars)
        run_vars[name] = val
        self.variables[name] = val
        return True, ExecResult.ADVANCE

    def _cf_print(self, stmt: str, run_vars: dict[str, Any],
                  *, parsed: PrintStmt | None = None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None:
            m = RE_PRINT.match(stmt)
            if not m:
                return None
            raw_expr = m.group(1)
        else:
            raw_expr = parsed.expr
        text = self._substitute_vars(raw_expr.strip(), run_vars)
        # Determine trailing separator: ; suppresses newline, , advances to tab
        suppress_newline = raw_expr.rstrip().endswith(';')
        tab_advance = raw_expr.rstrip().endswith(',')
        if suppress_newline:
            text = text.rstrip().removesuffix(';').rstrip()
        elif tab_advance:
            text = text.rstrip().removesuffix(',').rstrip()
        # Evaluate SPC(n) and TAB(n) inline
        def _replace_spc(m_spc):
            n = int(self._eval_with_vars(m_spc.group(1), run_vars))
            return ' ' * max(0, n)
        def _replace_tab(m_tab):
            n = int(self._eval_with_vars(m_tab.group(1), run_vars))
            return ' ' * max(0, n)
        text = re.sub(r'\bSPC\s*\(([^)]+)\)', _replace_spc, text, flags=re.IGNORECASE)
        text = re.sub(r'\bTAB\s*\(([^)]+)\)', _replace_tab, text, flags=re.IGNORECASE)
        # Evaluate the expression
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            output = text[1:-1]
        else:
            try:
                ns = run_vars.as_dict() if hasattr(run_vars, 'as_dict') else dict(run_vars) if not isinstance(run_vars, dict) else run_vars
                result = self._safe_eval(text, extra_ns=ns)
                output = str(result)
            except Exception:
                output = text
        # Output with separator behavior
        if suppress_newline:
            self.io.write(output)
        elif tab_advance:
            col = len(output) % 14
            padding = 14 - col if col > 0 else 14
            self.io.write(output + ' ' * padding)
        else:
            self.io.writeln(output)
        return True, ExecResult.ADVANCE

    def _cf_goto(self, stmt: str, sorted_lines: list[int],
                 *, parsed: GotoStmt | None = None) -> tuple[bool, int] | None:
        if parsed is None:
            m = RE_GOTO.match(stmt)
            if not m:
                return None
            target = int(m.group(1))
        else:
            target = parsed.target
        for idx, ln in enumerate(sorted_lines):
            if ln == target:
                return True, idx
        raise RuntimeError(f"GOTO {target}: LINE NOT FOUND")

    def _cf_gosub(self, stmt: str, sorted_lines: list[int], ip: int,
                  *, parsed: GosubStmt | None = None) -> tuple[bool, int] | None:
        if parsed is None:
            m = RE_GOSUB.match(stmt)
            if not m:
                return None
            target = int(m.group(1))
        else:
            target = parsed.target
        self._gosub_stack.append(ip + 1)
        for idx, ln in enumerate(sorted_lines):
            if ln == target:
                return True, idx
        raise RuntimeError(f"GOSUB {target}: LINE NOT FOUND")

    def _cf_for(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]], ip: int,
                *, parsed: ForStmt | None = None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None:
            m = RE_FOR.match(stmt)
            if not m:
                return None
            var = m.group(1)
            start_expr, end_expr, step_expr = m.group(2), m.group(3), m.group(4)
        else:
            var = parsed.var
            start_expr, end_expr, step_expr = parsed.start_expr, parsed.end_expr, parsed.step_expr
        start = self._eval_with_vars(start_expr, run_vars)
        end = self._eval_with_vars(end_expr, run_vars)
        step = self._eval_with_vars(step_expr, run_vars) if step_expr else 1
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
        run_vars[var] = start
        self.variables[var] = start
        loop_stack.append({'var': var, 'current': start, 'end': end,
                           'step': step, 'return_ip': ip})
        return True, ExecResult.ADVANCE

    def _cf_next(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]],
                 *, parsed: NextStmt | None = None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None:
            m = RE_NEXT.match(stmt)
            if not m:
                return None
            var = m.group(1)
        else:
            var = parsed.var
        if not loop_stack or loop_stack[-1].get('var') != var:
            if loop_stack:
                expected = loop_stack[-1].get('var', '?')
                raise RuntimeError(f"NEXT {var} does not match current FOR {expected}")
            raise RuntimeError(f"NEXT {var} without matching FOR")
        loop = loop_stack[-1]
        loop['current'] += loop['step']
        if (loop['step'] > 0 and loop['current'] <= loop['end']) or \
           (loop['step'] < 0 and loop['current'] >= loop['end']):
            run_vars[var] = loop['current']
            self.variables[var] = loop['current']
            return True, loop['return_ip'] + 1
        else:
            loop_stack.pop()
            return True, ExecResult.ADVANCE

    def _find_matching_wend(self, sorted_lines: list[int], ip: int) -> int:
        """Find the ip after the WEND matching the WHILE at ip.

        Scans forward with proper nesting depth tracking. Returns the ip
        index past the matching WEND. Raises with the WHILE line number
        for clear diagnostics.
        """
        depth = 1
        scan = ip + 1
        while scan < len(sorted_lines):
            s = self.program[sorted_lines[scan]].strip().upper()
            if s.startswith('WHILE '):
                depth += 1
            elif s == 'WEND':
                depth -= 1
                if depth == 0:
                    return scan + 1
            scan += 1
        line_num = sorted_lines[ip]
        raise RuntimeError(f"WHILE at line {line_num} has no matching WEND")

    def _cf_while(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]],
                  sorted_lines: list[int], ip: int,
                  *, parsed: WhileStmt | None = None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None:
            m = RE_WHILE.match(stmt)
            if not m:
                return None
            cond = m.group(1).strip()
        else:
            cond = parsed.condition
        if self._eval_condition(cond, run_vars):
            loop_stack.append({'type': 'while', 'cond': cond, 'return_ip': ip})
            return True, ExecResult.ADVANCE
        else:
            return True, self._find_matching_wend(sorted_lines, ip)

    def _cf_wend(self, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]],
                 sorted_lines: list[int] | None = None, ip: int | None = None) -> tuple[bool, ExecOutcome] | None:
        if not loop_stack or loop_stack[-1].get('type') != 'while':
            ctx = f" at line {sorted_lines[ip]}" if sorted_lines and ip is not None else ""
            raise RuntimeError(f"WEND{ctx} without matching WHILE")
        loop = loop_stack[-1]
        if self._eval_condition(loop['cond'], run_vars):
            return True, loop['return_ip']
        else:
            loop_stack.pop()
            return True, ExecResult.ADVANCE

    def _cf_if_then(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]],
                    sorted_lines: list[int], ip: int,
                    exec_fn: Callable[..., Any],
                    *, parsed: IfThenStmt | None = None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None:
            m = RE_IF_THEN.match(stmt)
            if not m:
                return None
            cond_str = m.group(1).strip()
            then_clause = m.group(2).strip()
            else_clause = m.group(3).strip() if m.group(3) else None
        else:
            cond_str = parsed.condition
            then_clause = parsed.then_clause
            else_clause = parsed.else_clause
        cond_vars = run_vars
        if self.locc_mode and self.locc:
            cond_vars = {**run_vars, **self.locc.classical}
        result = ExecResult.ADVANCE
        if self._eval_condition(cond_str, cond_vars):
            if then_clause:
                r = exec_fn(then_clause, loop_stack, sorted_lines, ip, run_vars)
                if r is not None and r is not ExecResult.ADVANCE:
                    result = r
        elif else_clause:
            r = exec_fn(else_clause, loop_stack, sorted_lines, ip, run_vars)
            if r is not None and r is not ExecResult.ADVANCE:
                result = r
        return True, result

    # ── Type-based dispatch table ────────────────────────────────────
    # Maps parsed Stmt types to handler functions.  Each handler
    # receives (self, stmt, parsed, loop_stack, sorted_lines, ip,
    # run_vars, exec_fn) and returns (handled: bool, result).  Built
    # once as a class variable so the dict lookup cost is paid
    # per-call, not per-class.

    @staticmethod
    def _d_rem(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        return True, ExecResult.ADVANCE

    _d_measure = _d_rem  # same behavior

    @staticmethod
    def _d_end(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        return True, ExecResult.END

    @staticmethod
    def _d_return(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        if not self._gosub_stack:
            raise RuntimeError("RETURN WITHOUT GOSUB")
        return True, self._gosub_stack.pop()

    @staticmethod
    def _d_wend(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        return self._cf_wend(run_vars, loop_stack, sorted_lines, ip)

    @staticmethod
    def _d_let_array(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_let_array(stmt, run_vars, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_let_var(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_let_var(stmt, run_vars, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_print(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_print(stmt, run_vars, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_goto(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_goto(stmt, sorted_lines, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_gosub(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_gosub(stmt, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_for(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_for(stmt, run_vars, loop_stack, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_next(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_next(stmt, run_vars, loop_stack, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_while(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_while(stmt, run_vars, loop_stack, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_if_then(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_if_then(stmt, run_vars, loop_stack, sorted_lines, ip, exec_fn, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_data(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_data(stmt, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_read(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_read(stmt, run_vars, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_on_goto(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_on_goto(stmt, run_vars, sorted_lines, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_on_gosub(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_on_gosub(stmt, run_vars, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_select_case(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_select_case(stmt, run_vars, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_case(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_case(stmt, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_end_select(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_end_select(stmt, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_do(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_do(stmt, run_vars, loop_stack, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_loop(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_loop(stmt, run_vars, loop_stack, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_exit(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_exit(stmt, loop_stack, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_swap(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_swap(stmt, run_vars, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_def_fn(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_def_fn(stmt, run_vars, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_option_base(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_option_base(stmt, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_sub(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_sub(stmt, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_end_sub(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_end_sub(stmt, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_function(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_function(stmt, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_end_function(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_end_function(stmt, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_call(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_call(stmt, run_vars, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_local(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_local(stmt, run_vars, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_static(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_static(stmt, run_vars, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_shared(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_shared(stmt, run_vars, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_on_error(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_on_error(stmt, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_resume(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_resume(stmt, sorted_lines, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_error(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_error(stmt, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_assert(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_assert(stmt, run_vars, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_stop(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_stop(stmt, sorted_lines, ip, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_on_measure(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_on_measure(stmt, parsed=parsed)
        return r if r is not None else (False, None)

    @staticmethod
    def _d_on_timer(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn):
        r = self._cf_on_timer(stmt, parsed=parsed)
        return r if r is not None else (False, None)

    _CF_DISPATCH: dict[type, Callable] = {
        RemStmt:         _d_rem,
        MeasureStmt:     _d_measure,
        EndStmt:         _d_end,
        ReturnStmt:      _d_return,
        WendStmt:        _d_wend,
        LetArrayStmt:    _d_let_array,
        LetStmt:         _d_let_var,
        PrintStmt:       _d_print,
        GotoStmt:        _d_goto,
        GosubStmt:       _d_gosub,
        ForStmt:         _d_for,
        NextStmt:        _d_next,
        WhileStmt:       _d_while,
        IfThenStmt:      _d_if_then,
        DataStmt:        _d_data,
        ReadStmt:        _d_read,
        OnGotoStmt:      _d_on_goto,
        OnGosubStmt:     _d_on_gosub,
        SelectCaseStmt:  _d_select_case,
        CaseStmt:        _d_case,
        EndSelectStmt:   _d_end_select,
        DoStmt:          _d_do,
        LoopStmt:        _d_loop,
        ExitStmt:        _d_exit,
        SwapStmt:        _d_swap,
        DefFnStmt:       _d_def_fn,
        OptionBaseStmt:  _d_option_base,
        SubStmt:         _d_sub,
        EndSubStmt:      _d_end_sub,
        FunctionStmt:    _d_function,
        EndFunctionStmt: _d_end_function,
        CallStmt:        _d_call,
        LocalStmt:       _d_local,
        StaticStmt:      _d_static,
        SharedStmt:      _d_shared,
        OnErrorStmt:     _d_on_error,
        ResumeStmt:      _d_resume,
        ErrorStmt:       _d_error,
        AssertStmt:      _d_assert,
        StopStmt:        _d_stop,
        OnMeasureStmt:   _d_on_measure,
        OnTimerStmt:     _d_on_timer,
    }

    def _exec_control_flow(
        self, stmt: str, loop_stack: list[dict[str, Any]],
        sorted_lines: list[int], ip: int, run_vars: dict[str, Any],
        exec_fn: Callable[..., Any],
        *, parsed=None,
    ) -> tuple[bool, ExecOutcome | None]:
        """Shared control flow for both Qiskit and LOCC execution paths.
        Returns (handled, result) — if handled is True, result is the return value.
        exec_fn is the recursive line executor for IF/multi-statement dispatch.

        Dispatches via dict lookup on the parsed Stmt type (O(1)) instead
        of a linear chain of regex-matching _cf_* calls.

        If *parsed* is provided, the parse_stmt call is skipped (avoids
        redundant parsing when the caller has already parsed the statement).
        """
        if parsed is None:
            parsed = parse_stmt(stmt)
        handler = self._CF_DISPATCH.get(type(parsed))
        if handler is not None:
            return handler(self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars, exec_fn)

        # RawStmt or unmapped type — not handled by control flow
        return False, None
