"""QUBASIC classic features — DATA/READ, ON GOTO, SELECT CASE, DO/LOOP, EXIT, SWAP, DEF FN, OPTION BASE."""

from __future__ import annotations

import random
import re
import time
from typing import Any

from qubasic_core.engine import (
    ExecResult, ExecOutcome,
    RE_DATA, RE_READ, RE_ON_GOTO, RE_ON_GOSUB,
    RE_SELECT_CASE, RE_CASE,
    RE_DO, RE_LOOP_STMT, RE_EXIT,
    RE_SWAP, RE_DEF_FN, RE_OPTION_BASE,
)
from qubasic_core.basic_format import number_text
from qubasic_core.variables import is_string_kind
from qubasic_core.host import HostState


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


class ClassicMixin(HostState):
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
        # RND draws from this generator; RANDOMIZE and SEED reseed it.
        self._rng = random.Random()
        self._rnd_last: float = 0.0
        self._data_line_starts: list[tuple[int, int]] = []

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

    def _stmt_randomize(self, stmt: str, run_vars: dict[str, Any]) -> bool:
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
        from qubasic_core.statements import DataStmt
        from qubasic_core.stmt_handlers import _parse_input_number
        prog = self._compiled()
        data = tuple((prog.lines[ip], st.values) for ip, st in enumerate(prog.stmts)
                     if isinstance(st, DataStmt))
        self._data_ptr = 0
        if data == self._data_cache_key:
            return
        self._data_pool = []
        self._data_line_starts = []
        for line, values in data:
            self._data_line_starts.append((line, len(self._data_pool)))
            for item, quoted in _data_items(values):
                if quoted:
                    self._data_pool.append(item)
                elif item in self._QUANTUM_STATES:
                    self._data_pool.append(f"QSTATE:{self._QUANTUM_STATES[item]}")
                else:
                    num = _parse_input_number(item) if item else 0
                    self._data_pool.append(num if num is not None else item)
        self._data_cache_key = data

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
                raise RuntimeError("OUT OF DATA")
            val = self._data_pool[self._data_ptr]
            self._data_ptr += 1
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
            self._data_ptr = 0
            return
        self._data_ptr = next((start for ln, start in self._data_line_starts if ln >= line),
                              len(self._data_pool))

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
            return True, self._jump_to_line(sorted_lines, targets[idx - 1], 'ON GOTO')
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
            target = self._jump_to_line(sorted_lines, targets[idx - 1], 'ON GOSUB')
            self._gosub_stack.append(ip + 1)
            return True, target
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
        test_val = self._safe_eval(expr, extra_ns=run_vars)
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

    def _case_matches(self, spec: str, test_val: Any, run_vars: dict[str, Any]) -> bool:
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

    def _cf_case(self, stmt: str, sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None:
            m = RE_CASE.match(stmt)
            if not m:
                return None
        # Reaching another CASE ends the matched block: continue after END SELECT.
        skip = self._as_program(sorted_lines).skip
        return True, skip.get(ip, ExecResult.ADVANCE)

    def _cf_end_select(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None or stmt.strip().upper() == 'END SELECT':
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
                return True, self._target(sorted_lines, 'skip', ip,
                                          f"DO at line {sorted_lines[ip]}")
        loop_stack.append({'type': 'do', 'return_ip': ip, 'kind': kind, 'cond': cond})
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

    def _cf_swap(self, stmt: str, run_vars: dict[str, Any], *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            a, b = parsed.a, parsed.b
        else:
            m = RE_SWAP.match(stmt)
            if not m:
                return None
            parts = self._split_arg_list(m.group(1))
            if len(parts) != 2:
                return None
            a, b = parts
        la, lb = self._lvalue(a, run_vars), self._lvalue(b, run_vars)
        va, vb = self._read_lvalue(la, run_vars), self._read_lvalue(lb, run_vars)
        if isinstance(va, str) != isinstance(vb, str):
            raise RuntimeError(f"TYPE MISMATCH: SWAP {a}, {b}")
        self._write_lvalue(la, vb, run_vars)
        self._write_lvalue(lb, va, run_vars)
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
        """OPTION BASE 0|1 — the lower bound of arrays declared with an upper
        bound only (variables.py)."""
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
