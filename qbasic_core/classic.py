"""QBASIC classic features — DATA/READ, ON GOTO, SELECT CASE, DO/LOOP, EXIT, SWAP, DEF FN, OPTION BASE."""

from __future__ import annotations

import re
from typing import Any

from qbasic_core.engine import (
    ExecResult, ExecOutcome,
    RE_DATA, RE_READ, RE_ON_GOTO, RE_ON_GOSUB,
    RE_SELECT_CASE, RE_CASE,
    RE_DO, RE_LOOP_STMT, RE_EXIT,
    RE_SWAP, RE_DEF_FN, RE_OPTION_BASE,
)


class ClassicMixin:
    """Classic BASIC features for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.variables,
    self.arrays, self._gosub_stack, self._eval_with_vars(),
    self._eval_condition(), self.eval_expr().
    """

    def _init_classic(self) -> None:
        self._data_pool: list[Any] = []
        self._data_ptr: int = 0
        self._option_base: int = 0
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
        """
        self._data_pool = []
        self._data_ptr = 0
        for ln in sorted(self.program.keys()):
            m = RE_DATA.match(self.program[ln].strip())
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

    # ── DATA / READ / RESTORE ──────────────────────────────────────────

    def _cf_data(self, stmt: str) -> tuple[bool, ExecOutcome] | None:
        """DATA — skip during execution (collected before run)."""
        if RE_DATA.match(stmt):
            return True, ExecResult.ADVANCE
        return None

    def _cf_read(self, stmt: str, run_vars: dict[str, Any]) -> tuple[bool, ExecOutcome] | None:
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
                    sorted_lines: list[int]) -> tuple[bool, ExecOutcome] | None:
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
                     sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome] | None:
        m = RE_ON_GOSUB.match(stmt)
        if not m:
            return None
        idx = int(self._eval_with_vars(m.group(1).strip(), run_vars))
        targets = [int(t.strip()) for t in m.group(2).split(',') if t.strip()]
        if 1 <= idx <= len(targets):
            target = targets[idx - 1]
            self._gosub_stack.append(ip + 1)
            for i, ln in enumerate(sorted_lines):
                if ln == target:
                    return True, i
            raise RuntimeError(f"ON GOSUB: LINE {target} NOT FOUND")
        return True, ExecResult.ADVANCE

    # ── SELECT CASE ───────────────────────────────────────────────────

    def _cf_select_case(self, stmt: str, run_vars: dict[str, Any],
                        sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome] | None:
        m = RE_SELECT_CASE.match(stmt)
        if not m:
            return None
        test_val = self._eval_with_vars(m.group(1).strip(), run_vars)
        self._select_stack.append(test_val)
        # Scan forward for matching CASE
        scan = ip + 1
        depth = 1
        while scan < len(sorted_lines):
            s = self.program[sorted_lines[scan]].strip().upper()
            if s.startswith('SELECT CASE'):
                depth += 1
            elif s == 'END SELECT':
                depth -= 1
                if depth == 0:
                    self._select_stack.pop()
                    return True, scan + 1  # no matching case
            elif depth == 1 and s.startswith('CASE '):
                case_val = s[5:].strip()
                if case_val == 'ELSE':
                    return True, scan + 1  # execute CASE ELSE block
                try:
                    if float(test_val) == float(self._eval_with_vars(case_val, run_vars)):
                        return True, scan + 1  # execute this CASE block
                except (ValueError, TypeError):
                    if str(test_val) == case_val:
                        return True, scan + 1
            scan += 1
        raise RuntimeError("SELECT CASE without END SELECT")

    def _cf_case(self, stmt: str, sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome] | None:
        m = RE_CASE.match(stmt)
        if not m:
            return None
        # If we reach a CASE during execution, skip to END SELECT (we already matched)
        depth = 1
        scan = ip + 1
        while scan < len(sorted_lines):
            s = self.program[sorted_lines[scan]].strip().upper()
            if s.startswith('SELECT CASE'):
                depth += 1
            elif s == 'END SELECT':
                depth -= 1
                if depth == 0:
                    if self._select_stack:
                        self._select_stack.pop()
                    return True, scan + 1
            scan += 1
        return True, ExecResult.ADVANCE

    def _cf_end_select(self, stmt: str) -> tuple[bool, ExecOutcome] | None:
        if stmt.strip().upper() == 'END SELECT':
            if self._select_stack:
                self._select_stack.pop()
            return True, ExecResult.ADVANCE
        return None

    # ── DO / LOOP ─────────────────────────────────────────────────────

    def _find_matching_loop(self, sorted_lines: list[int], ip: int) -> int:
        depth = 1
        scan = ip + 1
        while scan < len(sorted_lines):
            s = self.program[sorted_lines[scan]].strip().upper()
            if s.startswith('DO'):
                depth += 1
            elif s.startswith('LOOP'):
                depth -= 1
                if depth == 0:
                    return scan
            scan += 1
        raise RuntimeError(f"DO at line {sorted_lines[ip]} has no matching LOOP")

    def _cf_do(self, stmt: str, run_vars: dict[str, Any],
               loop_stack: list[dict[str, Any]],
               sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome] | None:
        m = RE_DO.match(stmt)
        if not m:
            return None
        kind = m.group(1)  # WHILE or UNTIL or None
        cond = m.group(2)  # condition or None
        if kind and cond:
            # Pre-test
            result = self._eval_condition(cond, run_vars)
            if kind.upper() == 'UNTIL':
                result = not result
            if not result:
                # Skip to after LOOP
                loop_ip = self._find_matching_loop(sorted_lines, ip)
                return True, loop_ip + 1
        loop_stack.append({'type': 'do', 'return_ip': ip, 'kind': kind, 'cond': cond})
        return True, ExecResult.ADVANCE

    def _cf_loop(self, stmt: str, run_vars: dict[str, Any],
                 loop_stack: list[dict[str, Any]],
                 sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome] | None:
        m = RE_LOOP_STMT.match(stmt)
        if not m:
            return None
        # Only handle as DO/LOOP keyword if we're inside a DO block
        if not loop_stack or loop_stack[-1].get('type') != 'do':
            # Bare "LOOP" (no WHILE/UNTIL) that matches a SUB name should
            # be handed off to subroutine expansion, not treated as a keyword.
            is_bare = m.group(1) is None  # no WHILE/UNTIL suffix
            if is_bare and hasattr(self, '_sub_defs') and 'LOOP' in self._sub_defs:
                return None  # let subroutine expansion handle it
            # Otherwise it's either bare LOOP with no sub, or LOOP WHILE/UNTIL
            # without a DO — both are errors or no-ops.  Return None so the
            # caller can decide (gate dispatch will surface an error if needed).
            return None
        loop = loop_stack[-1]
        # Post-test condition on LOOP
        kind = m.group(1)
        cond = m.group(2)
        if kind and cond:
            result = self._eval_condition(cond, run_vars)
            if kind.upper() == 'UNTIL':
                result = not result
            if result:
                return True, loop['return_ip']
            else:
                loop_stack.pop()
                return True, ExecResult.ADVANCE
        # Pre-test condition was on DO
        if loop.get('kind') and loop.get('cond'):
            result = self._eval_condition(loop['cond'], run_vars)
            if loop['kind'].upper() == 'UNTIL':
                result = not result
            if result:
                return True, loop['return_ip']
            else:
                loop_stack.pop()
                return True, ExecResult.ADVANCE
        # Infinite DO/LOOP (no condition) — loop forever
        return True, loop['return_ip']

    # ── EXIT ──────────────────────────────────────────────────────────

    def _cf_exit(self, stmt: str, loop_stack: list[dict[str, Any]],
                 sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome] | None:
        m = RE_EXIT.match(stmt)
        if not m:
            return None
        kind = m.group(1).upper()
        if kind == 'FOR':
            # Find matching NEXT
            scan = ip + 1
            depth = 1
            while scan < len(sorted_lines):
                s = self.program[sorted_lines[scan]].strip().upper()
                if s.startswith('FOR '):
                    depth += 1
                elif s.startswith('NEXT '):
                    depth -= 1
                    if depth == 0:
                        if loop_stack and loop_stack[-1].get('var'):
                            loop_stack.pop()
                        return True, scan + 1
                scan += 1
        elif kind == 'WHILE':
            scan = ip + 1
            depth = 1
            while scan < len(sorted_lines):
                s = self.program[sorted_lines[scan]].strip().upper()
                if s.startswith('WHILE '):
                    depth += 1
                elif s == 'WEND':
                    depth -= 1
                    if depth == 0:
                        if loop_stack and loop_stack[-1].get('type') == 'while':
                            loop_stack.pop()
                        return True, scan + 1
                scan += 1
        elif kind == 'DO':
            scan = ip + 1
            depth = 1
            while scan < len(sorted_lines):
                s = self.program[sorted_lines[scan]].strip().upper()
                if s.startswith('DO'):
                    depth += 1
                elif s.startswith('LOOP'):
                    depth -= 1
                    if depth == 0:
                        if loop_stack and loop_stack[-1].get('type') == 'do':
                            loop_stack.pop()
                        return True, scan + 1
                scan += 1
        elif kind in ('SUB', 'FUNCTION'):
            return True, ExecResult.END
        raise RuntimeError(f"EXIT {kind}: no matching block")

    # ── SWAP ──────────────────────────────────────────────────────────

    def _cf_swap(self, stmt: str, run_vars: dict[str, Any]) -> tuple[bool, ExecOutcome] | None:
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

    def _cf_def_fn(self, stmt: str, run_vars: dict[str, Any]) -> tuple[bool, ExecOutcome] | None:
        m = RE_DEF_FN.match(stmt)
        if not m:
            return None
        name = 'FN' + m.group(1).upper()
        params = [p.strip() for p in m.group(2).split(',') if p.strip()]
        body = m.group(3).strip()
        self._user_fns[name] = {'params': params, 'body': body}
        return True, ExecResult.ADVANCE

    def _call_user_fn(self, name: str, args: list[float]) -> float:
        """Call a DEF FN function."""
        fn = self._user_fns.get(name.upper())
        if fn is None:
            raise ValueError(f"UNDEFINED FUNCTION: {name}")
        ns = {}
        for i, pname in enumerate(fn['params']):
            ns[pname] = args[i] if i < len(args) else 0
        return float(self._safe_eval(fn['body'], extra_ns=ns))

    # ── OPTION BASE ───────────────────────────────────────────────────

    def _cf_option_base(self, stmt: str) -> tuple[bool, ExecOutcome] | None:
        m = RE_OPTION_BASE.match(stmt)
        if not m:
            return None
        self._option_base = int(m.group(1))
        return True, ExecResult.ADVANCE
