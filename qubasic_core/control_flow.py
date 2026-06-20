"""Control flow helpers extracted from QBasicTerminal.

Requires: TerminalProtocol (see qubasic_core.protocol).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from qubasic_core.engine import ExecResult, ExecOutcome
from qubasic_core.parser import parse_stmt
from qubasic_core.statements import (
    RawStmt, RemStmt, MeasureStmt, EndStmt, ReturnStmt, WendStmt,
    LetArrayStmt, LetStmt, LetStrStmt, PrintStmt, GotoStmt, GosubStmt,
    ForStmt, NextStmt, WhileStmt, IfThenStmt,
    DataStmt, ReadStmt, OnGotoStmt, OnGosubStmt,
    SelectCaseStmt, CaseStmt, EndSelectStmt, ElseStmt, EndIfStmt,
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
        name, idx_expr, val_expr = parsed.name, parsed.index_expr, parsed.value_expr
        self._assert_assignable(name)
        base = getattr(self, '_option_base', 0)
        # String arrays (name$) hold string values; numeric arrays hold floats.
        if name.endswith('$'):
            val = self._eval_string_expr(val_expr, run_vars)
        else:
            val = self._eval_with_vars(val_expr, run_vars)
        parts = self._split_arg_list(idx_expr)
        if len(parts) > 1:
            # Multi-dimensional write: flatten with the same stride convention
            # the expression-side accessor uses, so reads and writes agree.
            dims = getattr(self, '_array_dims', {}).get(name)
            indices = [int(self._eval_with_vars(p, run_vars)) - base for p in parts]
            if any(i < 0 for i in indices):
                raise RuntimeError(f"ARRAY INDEX OUT OF RANGE: {name}({idx_expr})")
            flat, stride = 0, 1
            for k in range(len(indices) - 1, -1, -1):
                flat += indices[k] * stride
                stride *= dims[k] if dims and k < len(dims) else 1
            if name not in self.arrays:
                raise RuntimeError(f"ARRAY NOT DIMENSIONED: {name} (use DIM first)")
            if flat < 0 or flat >= len(self.arrays[name]):
                raise RuntimeError(f"ARRAY INDEX OUT OF RANGE: {name}({idx_expr})")
            self.arrays[name][flat] = val
            return True, ExecResult.ADVANCE
        idx = int(self._eval_with_vars(idx_expr, run_vars)) - base
        if idx < 0:
            raise RuntimeError(f"ARRAY INDEX OUT OF RANGE: {name}({idx + base})")
        dimmed = getattr(self, '_dimmed_arrays', set())
        if name not in self.arrays:
            # Implicit array: created (and allowed to grow) on first assignment.
            self.arrays[name] = [0.0] * (idx + 1)
        elif idx >= len(self.arrays[name]):
            if name in dimmed:
                # Explicitly DIMmed: writes are bounds-checked like reads,
                # instead of silently auto-extending past the declared size.
                raise RuntimeError(
                    f"ARRAY INDEX OUT OF RANGE: {name}({idx + base}), "
                    f"size {len(self.arrays[name])}")
            while idx >= len(self.arrays[name]):
                self.arrays[name].append(0.0)
        self.arrays[name][idx] = val
        return True, ExecResult.ADVANCE

    def _cf_let_var(self, stmt: str, run_vars: dict[str, Any],
                    parsed: LetStmt) -> tuple[bool, ExecOutcome]:
        name, expr = parsed.name, parsed.expr
        self._assert_assignable(name)
        raw = self._safe_eval(expr, extra_ns=run_vars)
        if isinstance(raw, str):
            raise RuntimeError(
                f"TYPE MISMATCH: '{name}' is numeric; use '{name}$' for strings")
        val = float(raw)
        run_vars[name] = val
        self.variables[name] = val
        return True, ExecResult.ADVANCE

    def _cf_let_str(self, stmt: str, run_vars: dict[str, Any],
                    parsed: LetStrStmt) -> tuple[bool, ExecOutcome]:
        """LET v$ = <expr> — assign a string (or number) to a string variable."""
        name, expr = parsed.name, parsed.expr
        val = self._eval_string_expr(expr, run_vars)
        run_vars[name] = val
        self.variables[name] = val
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
        text = self._substitute_vars(item, run_vars)

        def _spaces(m):
            try:
                return ' ' * max(0, int(self._eval_with_vars(m.group(1), run_vars)))
            except Exception:
                return ''
        text = re.sub(r'\bSPC\s*\(([^)]+)\)', _spaces, text, flags=re.IGNORECASE)
        text = re.sub(r'\bTAB\s*\(([^)]+)\)', _spaces, text, flags=re.IGNORECASE)
        if not text.strip():
            return text                      # standalone SPC/TAB -> whitespace
        ns = run_vars.as_dict() if hasattr(run_vars, 'as_dict') else (
            run_vars if isinstance(run_vars, dict) else dict(run_vars))
        # Surface evaluation errors instead of silently printing the raw source
        # text (which used to turn PRINT SQRT(9) into the literal "SQRT(9)" and
        # an undefined variable into its own name).
        return str(self._safe_eval(text, extra_ns=ns))

    def _cf_print(self, stmt: str, run_vars: dict[str, Any],
                  parsed: PrintStmt) -> tuple[bool, ExecOutcome]:
        items = self._split_print_items(parsed.expr)
        if not items:
            self.io.writeln('')             # bare PRINT -> blank line
            return True, ExecResult.ADVANCE
        out = ''
        for item, sep in items:
            out += self._eval_print_item(item, run_vars)
            if sep == ',':                  # advance to next 14-column zone
                col = len(out) % 14
                out += ' ' * (14 - col if col else 14)
        # A trailing ';' or ',' suppresses the newline (cursor stays on line).
        if items[-1][1] in (';', ','):
            self.io.write(out)
        else:
            self.io.writeln(out)
        return True, ExecResult.ADVANCE

    def _cf_goto(self, stmt: str, sorted_lines: list[int],
                 parsed: GotoStmt) -> tuple[bool, int]:
        target = parsed.target
        for idx, ln in enumerate(sorted_lines):
            if ln == target:
                return True, idx
        raise RuntimeError(f"GOTO {target}: LINE NOT FOUND")

    def _cf_gosub(self, stmt: str, sorted_lines: list[int], ip: int,
                  parsed: GosubStmt) -> tuple[bool, int]:
        target = parsed.target
        self._gosub_stack.append(ip + 1)
        for idx, ln in enumerate(sorted_lines):
            if ln == target:
                return True, idx
        raise RuntimeError(f"GOSUB {target}: LINE NOT FOUND")

    def _cf_for(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]], ip: int,
                parsed: ForStmt) -> tuple[bool, ExecOutcome]:
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
        run_vars[var] = start
        self.variables[var] = start
        loop_stack.append({'var': var, 'current': start, 'end': end,
                           'step': step, 'return_ip': ip})
        return True, ExecResult.ADVANCE

    def _cf_next(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]],
                 parsed: NextStmt) -> tuple[bool, ExecOutcome]:
        # Supports bare ``NEXT`` (close the innermost FOR) and ``NEXT i, j``
        # (close i, then j). Each named variable must match the FOR on top of
        # the loop stack at that point.
        names = [v.strip() for v in parsed.var.split(',')] if parsed.var.strip() else ['']
        for name in names:
            if not loop_stack or loop_stack[-1].get('var') is None:
                raise RuntimeError(f"NEXT {name}".rstrip() + " without matching FOR")
            loop = loop_stack[-1]
            cur_var = loop['var']
            if name and cur_var != name:
                raise RuntimeError(f"NEXT {name} does not match current FOR {cur_var}")
            loop['current'] += loop['step']
            if (loop['step'] > 0 and loop['current'] <= loop['end']) or \
               (loop['step'] < 0 and loop['current'] >= loop['end']):
                run_vars[cur_var] = loop['current']
                self.variables[cur_var] = loop['current']
                return True, loop['return_ip'] + 1
            loop_stack.pop()
        return True, ExecResult.ADVANCE

    def _find_matching_wend(self, sorted_lines: list[int], ip: int) -> int:
        """Find the ip after the WEND matching the WHILE at ip.

        Uses pre-computed jump table when available (O(1)), falls back
        to linear scan with nesting depth tracking.
        """
        jt = getattr(self, '_jump_table', None)
        if jt and ip in jt:
            return jt[ip] + 1
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
                  parsed: WhileStmt) -> tuple[bool, ExecOutcome]:
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
                    parsed: IfThenStmt) -> tuple[bool, ExecOutcome]:
        cond_str = parsed.condition
        then_clause = parsed.then_clause
        else_clause = parsed.else_clause
        cond_vars = run_vars
        if self.locc_mode and self.locc:
            cond_vars = {**run_vars, **self.locc.classical}
        cond_true = self._eval_condition(cond_str, cond_vars)

        # Block form: "IF cond THEN" with no inline THEN/ELSE clauses spans
        # following lines up to a matching ELSE / END IF.
        if not then_clause and else_clause is None:
            else_ip, endif_ip = self._find_if_block(sorted_lines, ip)
            if cond_true:
                return True, ExecResult.ADVANCE  # fall into the THEN block
            if else_ip is not None:
                return True, else_ip + 1          # jump to the ELSE block
            return True, endif_ip + 1             # no ELSE: skip past END IF

        # Single-line form.
        result: ExecOutcome = ExecResult.ADVANCE
        if cond_true:
            if then_clause:
                r = exec_fn(then_clause, loop_stack, sorted_lines, ip, run_vars)
                if r is not None and r is not ExecResult.ADVANCE:
                    result = r
        elif else_clause:
            r = exec_fn(else_clause, loop_stack, sorted_lines, ip, run_vars)
            if r is not None and r is not ExecResult.ADVANCE:
                result = r
        return True, result

    @staticmethod
    def _is_block_if(line: str) -> bool:
        """True for a block-opening ``IF ... THEN`` (nothing after THEN)."""
        return re.match(r'IF\b.*\bTHEN\s*$', line.strip(), re.IGNORECASE) is not None

    def _find_if_block(self, sorted_lines: list[int], ip: int) -> tuple[int | None, int]:
        """From a block IF at ip, find (else_ip, endif_ip) handling nesting."""
        depth = 1
        else_ip: int | None = None
        scan = ip + 1
        while scan < len(sorted_lines):
            s = self.program[sorted_lines[scan]].strip()
            su = s.upper()
            if self._is_block_if(s):
                depth += 1
            elif su == 'END IF':
                depth -= 1
                if depth == 0:
                    return else_ip, scan
            elif su == 'ELSE' and depth == 1 and else_ip is None:
                else_ip = scan
            scan += 1
        raise RuntimeError(f"IF block at line {sorted_lines[ip]} has no matching END IF")

    def _cf_else(self, sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome]:
        """ELSE reached while running a THEN block: skip past the END IF."""
        depth = 1
        scan = ip + 1
        while scan < len(sorted_lines):
            s = self.program[sorted_lines[scan]].strip()
            su = s.upper()
            if self._is_block_if(s):
                depth += 1
            elif su == 'END IF':
                depth -= 1
                if depth == 0:
                    return True, scan + 1
            scan += 1
        return True, ExecResult.ADVANCE

    def _cf_return(self) -> tuple[bool, ExecOutcome]:
        """RETURN — pop the GOSUB stack, or error if there is nothing to return to."""
        if not self._gosub_stack:
            raise RuntimeError("RETURN WITHOUT GOSUB")
        return True, self._gosub_stack.pop()

    # ── Type-based dispatch table ────────────────────────────────────
    # Maps parsed Stmt types to handler lambdas.  Each lambda receives
    # (self, stmt, parsed, loop_stack, sorted_lines, ip, run_vars,
    # exec_fn) and returns (handled: bool, result).

    _CF_DISPATCH: dict[type, Callable] = {
        # Trivial handlers (no _cf_* method needed)
        RemStmt:         lambda s, st, p, ls, sl, ip, rv, ef: (True, ExecResult.ADVANCE),
        MeasureStmt:     lambda s, st, p, ls, sl, ip, rv, ef: (True, ExecResult.ADVANCE),
        EndStmt:         lambda s, st, p, ls, sl, ip, rv, ef: (True, ExecResult.END),
        ReturnStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_return(),
        # Handlers defined in control_flow.py (parsed is positional)
        WendStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_wend(rv, ls, sl, ip),
        LetArrayStmt:    lambda s, st, p, ls, sl, ip, rv, ef: s._cf_let_array(st, rv, p),
        LetStmt:         lambda s, st, p, ls, sl, ip, rv, ef: s._cf_let_var(st, rv, p),
        LetStrStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_let_str(st, rv, p),
        PrintStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_print(st, rv, p),
        GotoStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_goto(st, sl, p),
        GosubStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_gosub(st, sl, ip, p),
        ForStmt:         lambda s, st, p, ls, sl, ip, rv, ef: s._cf_for(st, rv, ls, ip, p),
        NextStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_next(st, rv, ls, p),
        WhileStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_while(st, rv, ls, sl, ip, p),
        IfThenStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_if_then(st, rv, ls, sl, ip, ef, p),
        # Handlers defined in classic.py (parsed is keyword-only)
        DataStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_data(st, parsed=p),
        ReadStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_read(st, rv, parsed=p),
        OnGotoStmt:      lambda s, st, p, ls, sl, ip, rv, ef: s._cf_on_goto(st, rv, sl, parsed=p),
        OnGosubStmt:     lambda s, st, p, ls, sl, ip, rv, ef: s._cf_on_gosub(st, rv, sl, ip, parsed=p),
        SelectCaseStmt:  lambda s, st, p, ls, sl, ip, rv, ef: s._cf_select_case(st, rv, sl, ip, parsed=p),
        CaseStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_case(st, sl, ip, parsed=p),
        EndSelectStmt:   lambda s, st, p, ls, sl, ip, rv, ef: s._cf_end_select(st, parsed=p),
        ElseStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_else(sl, ip),
        EndIfStmt:       lambda s, st, p, ls, sl, ip, rv, ef: (True, ExecResult.ADVANCE),
        DoStmt:          lambda s, st, p, ls, sl, ip, rv, ef: s._cf_do(st, rv, ls, sl, ip, parsed=p),
        LoopStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_loop(st, rv, ls, sl, ip, parsed=p),
        ExitStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_exit(st, ls, sl, ip, parsed=p),
        SwapStmt:        lambda s, st, p, ls, sl, ip, rv, ef: s._cf_swap(st, rv, parsed=p),
        DefFnStmt:       lambda s, st, p, ls, sl, ip, rv, ef: s._cf_def_fn(st, rv, parsed=p),
        OptionBaseStmt:  lambda s, st, p, ls, sl, ip, rv, ef: s._cf_option_base(st, parsed=p),
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
