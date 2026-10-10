"""QUBASIC classic features — DATA/READ, ON GOTO, SELECT CASE, DO/LOOP, EXIT, SWAP, DEF FN, OPTION BASE."""

from __future__ import annotations

import re
from typing import Any

from qubasic_core.engine import (
    ExecResult, ExecOutcome,
    RE_DATA, RE_READ, RE_ON_GOTO, RE_ON_GOSUB,
    RE_SELECT_CASE, RE_CASE,
    RE_DO, RE_LOOP_STMT, RE_EXIT,
    RE_SWAP, RE_DEF_FN, RE_OPTION_BASE,
)
from qubasic_core.statements import SelectCaseStmt, CaseStmt, EndSelectStmt


class ClassicMixin:
    """Classic BASIC features for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.variables,
    self.arrays, self._gosub_stack, self._eval_with_vars(),
    self._eval_condition(), self.eval_expr().
    """

    def _init_classic(self) -> None:
        self._data_pool: list[Any] = []
        self._data_ptr: int = 0
        self._data_cache_key: tuple | None = None
        self._option_base: int = 0
        # OPTION ENDIAN BIG shows bitstrings with qubit 0 leftmost.
        self._endian_big: bool = False
        self._user_fns: dict[str, dict[str, Any]] = {}
        self._select_stack: list[Any] = []

    # Quantum state names recognized in DATA statements
    _QUANTUM_STATES = {
        '|0>': '0', '|1>': '1', '|+>': '+', '|->': '-',
        '|0⟩': '0', '|1⟩': '1', '|+⟩': '+', '|-⟩': '-',
        '|BELL>': 'BELL', '|GHZ>': 'GHZ', '|GHZ3>': 'GHZ3', '|GHZ4>': 'GHZ4',
        '|W>': 'W', '|W3>': 'W3',
    }

    def _collect_data(self) -> None:
        """Scan program for DATA statements and build the data pool.

        Recognizes quantum state names like |+>, |0>, |GHZ3> as special
        string tokens that can trigger state preparation when READ assigns
        them to a variable.

        Caches the result keyed on DATA-containing lines so repeated RUN
        calls skip the rescan when the program hasn't changed.
        """
        # Build cache key from all DATA-containing lines
        data_lines = []
        for ln in sorted(self.program.keys()):
            text = self.program[ln].strip()
            if RE_DATA.match(text):
                data_lines.append((ln, text))
        cache_key = tuple(data_lines)
        if cache_key == self._data_cache_key:
            # Program DATA lines unchanged — just reset the read pointer
            self._data_ptr = 0
            return
        self._data_pool = []
        self._data_ptr = 0
        for _ln, text in data_lines:
            m = RE_DATA.match(text)
            if m:
                for item in m.group(1).split(','):
                    item = item.strip()
                    if (item.startswith('"') and item.endswith('"')) or \
                       (item.startswith("'") and item.endswith("'")):
                        self._data_pool.append(item[1:-1])
                    elif item in self._QUANTUM_STATES:
                        self._data_pool.append(f"QSTATE:{self._QUANTUM_STATES[item]}")
                    else:
                        try:
                            self._data_pool.append(float(item) if '.' in item else int(item))
                        except ValueError:
                            self._data_pool.append(item)
        self._data_cache_key = cache_key

    # ── DATA / READ / RESTORE ──────────────────────────────────────────

    def _cf_data(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        """DATA — skip during execution (collected before run)."""
        if parsed is not None or RE_DATA.match(stmt):
            return True, ExecResult.ADVANCE
        return None

    def _cf_read(self, stmt: str, run_vars: dict[str, Any], *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            var_names = [v.strip() for v in parsed.var_list.split(',')]
        else:
            m = RE_READ.match(stmt)
            if not m:
                return None
            var_names = [v.strip() for v in m.group(1).split(',')]
        for vname in var_names:
            if self._data_ptr >= len(self._data_pool):
                raise RuntimeError("READ: OUT OF DATA")
            val = self._data_pool[self._data_ptr]
            self._data_ptr += 1
            run_vars[vname] = val
            self.variables[vname] = val
        return True, ExecResult.ADVANCE

    def cmd_restore(self, rest: str = '') -> None:
        """RESTORE — reset DATA pointer to beginning."""
        self._data_ptr = 0

    # ── ON expr GOTO / GOSUB ──────────────────────────────────────────

    def _cf_on_goto(self, stmt: str, run_vars: dict[str, Any],
                    sorted_lines: list[int], *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            idx = int(self._eval_with_vars(parsed.expr, run_vars))
            targets = list(parsed.targets)
        else:
            m = RE_ON_GOTO.match(stmt)
            if not m:
                return None
            idx = int(self._eval_with_vars(m.group(1).strip(), run_vars))
            targets = [int(t.strip()) for t in m.group(2).split(',') if t.strip()]
        if 1 <= idx <= len(targets):
            target = targets[idx - 1]
            for i, ln in enumerate(sorted_lines):
                if ln == target:
                    return True, i
            raise RuntimeError(f"ON GOTO: LINE {target} NOT FOUND")
        return True, ExecResult.ADVANCE  # out of range: fall through

    def _cf_on_gosub(self, stmt: str, run_vars: dict[str, Any],
                     sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            idx = int(self._eval_with_vars(parsed.expr, run_vars))
            targets = list(parsed.targets)
        else:
            m = RE_ON_GOSUB.match(stmt)
            if not m:
                return None
            idx = int(self._eval_with_vars(m.group(1).strip(), run_vars))
            targets = [int(t.strip()) for t in m.group(2).split(',') if t.strip()]
        if 1 <= idx <= len(targets):
            target = targets[idx - 1]
            self._gosub_stack.append(self._return_point(ip))
            for i, ln in enumerate(sorted_lines):
                if ln == target:
                    return True, i
            raise RuntimeError(f"ON GOSUB: LINE {target} NOT FOUND")
        return True, ExecResult.ADVANCE

    # ── SELECT CASE ───────────────────────────────────────────────────

    def _cf_select_case(self, stmt: str, run_vars: dict[str, Any],
                        sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            expr = parsed.expr
        else:
            m = RE_SELECT_CASE.match(stmt)
            if not m:
                return None
            expr = m.group(1).strip()
        ns = run_vars.as_dict() if hasattr(run_vars, 'as_dict') else dict(run_vars)
        test_val = self._safe_eval(expr, extra_ns=ns)
        self._select_stack.append(test_val)
        # The first CASE at this nesting level that matches runs its block
        # (the statements after it, on its line and the lines below).
        depth = 1
        for j, kk, p, frame, n_parts in self._stmts_after(sorted_lines, ip):
            if isinstance(p, SelectCaseStmt):
                depth += 1
            elif isinstance(p, EndSelectStmt):
                depth -= 1
                if depth == 0:
                    self._select_stack.pop()
                    return True, self._after_stmt(j, kk, frame, n_parts)
            elif depth == 1 and isinstance(p, CaseStmt) and \
                    self._case_matches(p.value, test_val, run_vars):
                return True, self._after_stmt(j, kk, frame, n_parts)
        raise RuntimeError("SELECT CASE without END SELECT")

    _CASE_OPS = {'=': lambda a, b: a == b, '<>': lambda a, b: a != b,
                 '<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
                 '>': lambda a, b: a > b, '>=': lambda a, b: a >= b}

    def _case_matches(self, spec: str, test_val: Any, run_vars: dict[str, Any]) -> bool:
        """Whether a CASE clause matches the SELECT value: CASE ELSE, a list of
        values, ranges (lo TO hi) and comparisons (IS > v), as in QBASIC.
        String cases compare exactly."""
        if spec.strip().upper() == 'ELSE':
            return True
        ns = run_vars.as_dict() if hasattr(run_vars, 'as_dict') else dict(run_vars)

        def value(expr: str) -> Any:
            v = self._safe_eval(expr.strip(), extra_ns=ns)
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
            except TypeError:
                continue    # a string case against a number (or vice versa)
        return False

    def _cf_case(self, stmt: str, sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None:
            m = RE_CASE.match(stmt)
            if not m:
                return None
        # Reaching another CASE ends the matched block: continue after END SELECT.
        depth = 1
        for j, kk, p, frame, n_parts in self._stmts_after(sorted_lines, ip):
            if isinstance(p, SelectCaseStmt):
                depth += 1
            elif isinstance(p, EndSelectStmt):
                depth -= 1
                if depth == 0:
                    if self._select_stack:
                        self._select_stack.pop()
                    return True, self._after_stmt(j, kk, frame, n_parts)
        return True, ExecResult.ADVANCE

    def _cf_end_select(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None or stmt.strip().upper() == 'END SELECT':
            if not self._select_stack:
                raise RuntimeError("END SELECT without matching SELECT CASE")
            self._select_stack.pop()
            return True, ExecResult.ADVANCE
        return None

    # ── DO / LOOP ─────────────────────────────────────────────────────

    def _cf_do(self, stmt: str, run_vars: dict[str, Any],
               loop_stack: list[dict[str, Any]],
               sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            kind = parsed.kind
            cond = parsed.condition
        else:
            m = RE_DO.match(stmt)
            if not m:
                return None
            kind = m.group(1)  # WHILE or UNTIL or None
            cond = m.group(2)  # condition or None
        self._drop_loop(loop_stack, 'do', ip)
        if kind and cond:
            # Pre-test
            result = self._eval_condition(cond, run_vars)
            if kind.upper() == 'UNTIL':
                result = not result
            if not result:
                target = self._find_block_end(sorted_lines, ip, 'do')
                if target is None:
                    raise RuntimeError(f"DO at line {sorted_lines[ip]} has no matching LOOP")
                return True, target
        loop_stack.append(self._mark_loop(
            {'type': 'do', 'return_ip': ip, 'kind': kind, 'cond': cond}, ip))
        return True, ExecResult.ADVANCE

    def _cf_loop(self, stmt: str, run_vars: dict[str, Any],
                 loop_stack: list[dict[str, Any]],
                 sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            loop_kind = parsed.kind
            loop_cond = parsed.condition
        else:
            m = RE_LOOP_STMT.match(stmt)
            if not m:
                return None
            loop_kind = m.group(1)
            loop_cond = m.group(2)
        # Only handle as DO/LOOP keyword if we're inside a DO block (loops a
        # jump left open inside it are closed here).
        idx = len(loop_stack) - 1
        while idx >= 0 and loop_stack[idx].get('type') != 'do':
            idx -= 1
        if idx < 0:
            # Bare "LOOP" (no WHILE/UNTIL) that matches a SUB name should
            # be handed off to subroutine expansion, not treated as a keyword.
            is_bare = loop_kind is None  # no WHILE/UNTIL suffix
            if is_bare and hasattr(self, '_sub_defs') and 'LOOP' in self._sub_defs:
                return None  # let subroutine expansion handle it
            # Otherwise it's either bare LOOP with no sub, or LOOP WHILE/UNTIL
            # without a DO — both are errors or no-ops.  Return None so the
            # caller can decide (gate dispatch will surface an error if needed).
            return None
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
                 sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            kind = parsed.target
        else:
            m = RE_EXIT.match(stmt)
            if not m:
                return None
            kind = m.group(1).upper()
        if kind in ('FOR', 'WHILE', 'DO'):
            # Leave the innermost loop of that kind (and any loops inside it)
            # for the statement after its NEXT / WEND / LOOP.
            want = kind.lower()
            idx = len(loop_stack) - 1
            while idx >= 0 and not (
                    (want == 'for' and loop_stack[idx].get('var') is not None)
                    or loop_stack[idx].get('type') == want):
                idx -= 1
            target = self._find_block_end(sorted_lines, ip, want)
            if idx >= 0 and target is not None:
                del loop_stack[idx:]
                return True, target
        elif kind in ('SUB', 'FUNCTION'):
            return True, ExecResult.END
        raise RuntimeError(f"EXIT {kind}: no matching block")

    # ── SWAP ──────────────────────────────────────────────────────────

    def _cf_swap(self, stmt: str, run_vars: dict[str, Any], *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            a, b = parsed.a, parsed.b
        else:
            m = RE_SWAP.match(stmt)
            if not m:
                return None
            a, b = m.group(1), m.group(2)
        va = run_vars.get(a, self.variables.get(a, 0))
        vb = run_vars.get(b, self.variables.get(b, 0))
        run_vars[a] = vb
        run_vars[b] = va
        self.variables[a] = vb
        self.variables[b] = va
        return True, ExecResult.ADVANCE

    # ── DEF FN ────────────────────────────────────────────────────────

    def _cf_def_fn(self, stmt: str, run_vars: dict[str, Any], *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            name = 'FN' + parsed.name.upper()
            params = [p.strip() for p in parsed.params.split(',') if p.strip()]
            body = parsed.body.strip()
        else:
            m = RE_DEF_FN.match(stmt)
            if not m:
                return None
            name = 'FN' + m.group(1).upper()
            params = [p.strip() for p in m.group(2).split(',') if p.strip()]
            body = m.group(3).strip()
        self._user_fns[name] = {'params': params, 'body': body}
        return True, ExecResult.ADVANCE

    # ── OPTION BASE ───────────────────────────────────────────────────

    def _cf_option_base(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        # Parsed and stored; array indexing always starts at 0 in this implementation.
        if parsed is not None:
            self._option_base = parsed.base
        else:
            m = RE_OPTION_BASE.match(stmt)
            if not m:
                return None
            self._option_base = int(m.group(1))
        return True, ExecResult.ADVANCE

    def _cf_option_endian(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        """OPTION ENDIAN BIG|LITTLE — bitstring display order (default LITTLE,
        the qiskit convention: qubit 0 rightmost). Display-side only: internal
        counts keys and statevector indexing keep the qiskit order."""
        if parsed is not None:
            self._endian_big = parsed.big
        else:
            from qubasic_core.patterns import RE_OPTION_ENDIAN
            m = RE_OPTION_ENDIAN.match(stmt)
            if not m:
                return None
            self._endian_big = m.group(1).upper() == 'BIG'
        return True, ExecResult.ADVANCE
