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


class ControlFlowMixin:
    """Mixin providing control flow helpers for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.variables,
    self.arrays, self.locc_mode, self.locc, self._gosub_stack,
    self._eval_with_vars(), self._eval_condition(), self._substitute_vars().
    """

    # ── Control flow helpers (decomposed from _exec_control_flow) ────

    def _cf_let_array(self, stmt: str, run_vars: dict[str, Any]) -> tuple[bool, ExecOutcome] | None:
        m = RE_LET_ARRAY.match(stmt)
        if not m:
            return None
        name, idx_expr, val_expr = m.group(1), m.group(2), m.group(3)
        idx = int(self._eval_with_vars(idx_expr, run_vars))
        val = self._eval_with_vars(val_expr, run_vars)
        if name not in self.arrays:
            self.arrays[name] = [0.0] * (idx + 1)
        while idx >= len(self.arrays[name]):
            self.arrays[name].append(0.0)
        self.arrays[name][idx] = val
        return True, ExecResult.ADVANCE

    def _cf_let_var(self, stmt: str, run_vars: dict[str, Any]) -> tuple[bool, ExecOutcome] | None:
        m = RE_LET_VAR.match(stmt)
        if not m:
            return None
        name = m.group(1)
        val = self._eval_with_vars(m.group(2), run_vars)
        run_vars[name] = val
        self.variables[name] = val
        return True, ExecResult.ADVANCE

    def _cf_print(self, stmt: str, run_vars: dict[str, Any]) -> tuple[bool, ExecOutcome] | None:
        m = RE_PRINT.match(stmt)
        if not m:
            return None
        raw_expr = m.group(1)
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
                output = str(self._eval_with_vars(text, run_vars))
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

    def _cf_goto(self, stmt: str, sorted_lines: list[int]) -> tuple[bool, int] | None:
        m = RE_GOTO.match(stmt)
        if not m:
            return None
        target = int(m.group(1))
        for idx, ln in enumerate(sorted_lines):
            if ln == target:
                return True, idx
        raise RuntimeError(f"GOTO {target}: LINE NOT FOUND")

    def _cf_gosub(self, stmt: str, sorted_lines: list[int], ip: int) -> tuple[bool, int] | None:
        m = RE_GOSUB.match(stmt)
        if not m:
            return None
        target = int(m.group(1))
        self._gosub_stack.append(ip + 1)
        for idx, ln in enumerate(sorted_lines):
            if ln == target:
                return True, idx
        raise RuntimeError(f"GOSUB {target}: LINE NOT FOUND")

    def _cf_for(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]], ip: int) -> tuple[bool, ExecOutcome] | None:
        m = RE_FOR.match(stmt)
        if not m:
            return None
        var = m.group(1)
        start = self._eval_with_vars(m.group(2), run_vars)
        end = self._eval_with_vars(m.group(3), run_vars)
        step = self._eval_with_vars(m.group(4), run_vars) if m.group(4) else 1
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

    def _cf_next(self, stmt: str, run_vars: dict[str, Any], loop_stack: list[dict[str, Any]]) -> tuple[bool, ExecOutcome] | None:
        m = RE_NEXT.match(stmt)
        if not m:
            return None
        var = m.group(1)
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
                  sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome] | None:
        m = RE_WHILE.match(stmt)
        if not m:
            return None
        cond = m.group(1).strip()
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
                    exec_fn: Callable[..., Any]) -> tuple[bool, ExecOutcome] | None:
        m = RE_IF_THEN.match(stmt)
        if not m:
            return None
        cond_str = m.group(1).strip()
        then_clause = m.group(2).strip()
        else_clause = m.group(3).strip() if m.group(3) else None
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

    def _exec_control_flow(
        self, stmt: str, loop_stack: list[dict[str, Any]],
        sorted_lines: list[int], ip: int, run_vars: dict[str, Any],
        exec_fn: Callable[..., Any],
    ) -> tuple[bool, ExecOutcome | None]:
        """Shared control flow for both Qiskit and LOCC execution paths.
        Returns (handled, result) — if handled is True, result is the return value.
        exec_fn is the recursive line executor for IF/multi-statement dispatch."""
        upper = stmt.upper().strip()

        if upper.startswith('REM') or upper.startswith("'") or upper == 'MEASURE':
            return True, ExecResult.ADVANCE
        if upper == 'END':
            return True, ExecResult.END
        if upper == 'RETURN':
            if not self._gosub_stack:
                raise RuntimeError("RETURN WITHOUT GOSUB")
            return True, self._gosub_stack.pop()
        if upper == 'WEND':
            return self._cf_wend(run_vars, loop_stack, sorted_lines, ip)

        # Core control flow — direct dispatch, short-circuit on match
        r = self._cf_let_array(stmt, run_vars)
        if r is not None: return r
        r = self._cf_let_var(stmt, run_vars)
        if r is not None: return r
        r = self._cf_print(stmt, run_vars)
        if r is not None: return r
        r = self._cf_goto(stmt, sorted_lines)
        if r is not None: return r
        r = self._cf_gosub(stmt, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_for(stmt, run_vars, loop_stack, ip)
        if r is not None: return r
        r = self._cf_next(stmt, run_vars, loop_stack)
        if r is not None: return r
        r = self._cf_while(stmt, run_vars, loop_stack, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_if_then(stmt, run_vars, loop_stack, sorted_lines, ip, exec_fn)
        if r is not None: return r

        # Extended control flow (classic BASIC)
        r = self._cf_data(stmt)
        if r is not None: return r
        r = self._cf_read(stmt, run_vars)
        if r is not None: return r
        r = self._cf_on_goto(stmt, run_vars, sorted_lines)
        if r is not None: return r
        r = self._cf_on_gosub(stmt, run_vars, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_select_case(stmt, run_vars, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_case(stmt, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_end_select(stmt)
        if r is not None: return r
        r = self._cf_do(stmt, run_vars, loop_stack, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_loop(stmt, run_vars, loop_stack, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_exit(stmt, loop_stack, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_swap(stmt, run_vars)
        if r is not None: return r
        r = self._cf_def_fn(stmt, run_vars)
        if r is not None: return r
        r = self._cf_option_base(stmt)
        if r is not None: return r
        # SUB/FUNCTION
        r = self._cf_sub(stmt, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_end_sub(stmt)
        if r is not None: return r
        r = self._cf_function(stmt, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_end_function(stmt)
        if r is not None: return r
        r = self._cf_call(stmt, run_vars, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_local(stmt, run_vars)
        if r is not None: return r
        r = self._cf_static(stmt, run_vars)
        if r is not None: return r
        r = self._cf_shared(stmt, run_vars)
        if r is not None: return r
        # Debug/error handling
        r = self._cf_on_error(stmt)
        if r is not None: return r
        r = self._cf_resume(stmt, sorted_lines)
        if r is not None: return r
        r = self._cf_error(stmt)
        if r is not None: return r
        r = self._cf_assert(stmt, run_vars)
        if r is not None: return r
        r = self._cf_stop(stmt, sorted_lines, ip)
        if r is not None: return r
        r = self._cf_on_measure(stmt)
        if r is not None: return r
        r = self._cf_on_timer(stmt)
        if r is not None: return r

        return False, None
