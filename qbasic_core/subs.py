"""QBASIC SUB/FUNCTION — proper subroutines with local scope."""

from __future__ import annotations

import re
from typing import Any

from qbasic_core.engine import (
    ExecResult, ExecOutcome,
    RE_SUB, RE_END_SUB, RE_FUNCTION, RE_END_FUNCTION,
    RE_CALL, RE_LOCAL, RE_STATIC_DECL, RE_SHARED,
)


class SubroutineMixin:
    """SUB/FUNCTION subroutines with LOCAL, STATIC, SHARED scoping.

    Requires: TerminalProtocol — uses self.program, self.variables,
    self._eval_with_vars(), self.eval_expr().
    """

    def _init_subs(self) -> None:
        self._sub_defs: dict[str, dict[str, Any]] = {}  # name -> {params, start_ip, end_ip}
        self._func_defs: dict[str, dict[str, Any]] = {}
        self._scope_stack: list[dict[str, Any]] = []
        self._static_vars: dict[str, dict[str, Any]] = {}  # sub_name -> {var: val}
        self._call_stack: list[dict[str, Any]] = []

    def _scan_subs(self, sorted_lines: list[int]) -> None:
        """Scan program for SUB and FUNCTION blocks before execution."""
        self._sub_defs.clear()
        self._func_defs.clear()
        for i, ln in enumerate(sorted_lines):
            stmt = self.program[ln].strip()
            m = RE_SUB.match(stmt)
            if m:
                name = m.group(1).upper()
                params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
                end_ip = self._find_end_block(sorted_lines, i, 'SUB')
                self._sub_defs[name] = {'params': params, 'start_ip': i + 1, 'end_ip': end_ip}
                continue
            m = RE_FUNCTION.match(stmt)
            if m:
                name = m.group(1).upper()
                params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
                end_ip = self._find_end_block(sorted_lines, i, 'FUNCTION')
                self._func_defs[name] = {'params': params, 'start_ip': i + 1, 'end_ip': end_ip}

    def _find_end_block(self, sorted_lines: list[int], start_ip: int, kind: str) -> int:
        scan = start_ip + 1
        end_re = RE_END_SUB if kind == 'SUB' else RE_END_FUNCTION
        while scan < len(sorted_lines):
            s = self.program[sorted_lines[scan]].strip()
            if end_re.match(s):
                return scan
            scan += 1
        ln = sorted_lines[start_ip]
        raise RuntimeError(f"{kind} at line {ln} has no END {kind}")

    # ── Control flow handlers ──────────────────────────────────────────

    def _cf_sub(self, stmt: str, sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome] | None:
        """SUB — skip the block during normal execution."""
        m = RE_SUB.match(stmt)
        if not m:
            return None
        name = m.group(1).upper()
        if name in self._sub_defs:
            return True, self._sub_defs[name]['end_ip'] + 1
        return True, ExecResult.ADVANCE

    def _cf_end_sub(self, stmt: str) -> tuple[bool, ExecOutcome] | None:
        if not RE_END_SUB.match(stmt):
            return None
        if self._call_stack:
            frame = self._call_stack.pop()
            self._pop_scope()
            return True, frame['return_ip']
        return True, ExecResult.ADVANCE

    def _cf_function(self, stmt: str, sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome] | None:
        """FUNCTION — skip the block during normal execution."""
        m = RE_FUNCTION.match(stmt)
        if not m:
            return None
        name = m.group(1).upper()
        if name in self._func_defs:
            return True, self._func_defs[name]['end_ip'] + 1
        return True, ExecResult.ADVANCE

    def _cf_end_function(self, stmt: str) -> tuple[bool, ExecOutcome] | None:
        if not RE_END_FUNCTION.match(stmt):
            return None
        if self._call_stack:
            frame = self._call_stack.pop()
            ret_val = self.variables.get(frame.get('func_name', ''), 0)
            self._pop_scope()
            if frame.get('func_name'):
                self.variables['_FUNC_RETURN'] = ret_val
            return True, frame['return_ip']
        return True, ExecResult.ADVANCE

    def _cf_call(self, stmt: str, run_vars: dict[str, Any],
                 sorted_lines: list[int], ip: int) -> tuple[bool, ExecOutcome] | None:
        m = RE_CALL.match(stmt)
        if not m:
            return None
        name = m.group(1).upper()
        arg_str = m.group(2) or ''
        args = [self._eval_with_vars(a.strip(), run_vars)
                for a in arg_str.split(',') if a.strip()] if arg_str.strip() else []
        if name not in self._sub_defs:
            raise RuntimeError(f"UNDEFINED SUB: {name}")
        sub = self._sub_defs[name]
        self._push_scope()
        # Bind parameters
        for i, pname in enumerate(sub['params']):
            val = args[i] if i < len(args) else 0
            self.variables[pname] = val
        # Restore STATIC vars
        if name in self._static_vars:
            for k, v in self._static_vars[name].items():
                self.variables[k] = v
        self._call_stack.append({'return_ip': ip + 1, 'sub_name': name})
        return True, sub['start_ip']

    def _invoke_function(self, name: str, args: list[float],
                         sorted_lines: list[int]) -> Any:
        """Execute a FUNCTION block and return its value.
        Called from the expression evaluator."""
        uname = name.upper()
        if uname not in self._func_defs:
            raise ValueError(f"UNDEFINED FUNCTION: {name}")
        func = self._func_defs[uname]
        self._push_scope()
        for i, pname in enumerate(func['params']):
            self.variables[pname] = args[i] if i < len(args) else 0
        self.variables[uname] = 0  # return variable
        if uname in self._static_vars:
            for k, v in self._static_vars[uname].items():
                self.variables[k] = v
        # Execute function body line by line
        ip = func['start_ip']
        loop_stack: list[dict[str, Any]] = []
        run_vars = dict(self.variables)
        _iters = 0
        while ip <= func['end_ip']:
            _iters += 1
            if _iters > self._max_iterations:
                raise RuntimeError("FUNCTION LOOP LIMIT")
            stmt = self.program[sorted_lines[ip]].strip()
            if RE_END_FUNCTION.match(stmt):
                break
            # Handle LET for return value
            upper = stmt.upper()
            if upper.startswith('LET '):
                from qbasic_core.engine import RE_LET_VAR
                m = RE_LET_VAR.match(stmt)
                if m:
                    vname = m.group(1)
                    val = self._eval_with_vars(m.group(2), run_vars)
                    run_vars[vname] = val
                    self.variables[vname] = val
            ip += 1
        ret_val = self.variables.get(uname, run_vars.get(uname, 0))
        self._pop_scope()
        return ret_val

    # ── LOCAL / STATIC / SHARED ────────────────────────────────────────

    def _cf_local(self, stmt: str, run_vars: dict[str, Any]) -> tuple[bool, ExecOutcome] | None:
        m = RE_LOCAL.match(stmt)
        if not m:
            return None
        for vname in m.group(1).split(','):
            vname = vname.strip()
            if vname:
                run_vars[vname] = 0
                self.variables[vname] = 0
        return True, ExecResult.ADVANCE

    def _cf_static(self, stmt: str, run_vars: dict[str, Any]) -> tuple[bool, ExecOutcome] | None:
        m = RE_STATIC_DECL.match(stmt)
        if not m:
            return None
        sub_name = self._call_stack[-1]['sub_name'] if self._call_stack else '_GLOBAL'
        if sub_name not in self._static_vars:
            self._static_vars[sub_name] = {}
        for vname in m.group(1).split(','):
            vname = vname.strip()
            if vname:
                val = self._static_vars[sub_name].get(vname, 0)
                run_vars[vname] = val
                self.variables[vname] = val
        return True, ExecResult.ADVANCE

    def _cf_shared(self, stmt: str, run_vars: dict[str, Any]) -> tuple[bool, ExecOutcome] | None:
        m = RE_SHARED.match(stmt)
        if not m:
            return None
        for vname in m.group(1).split(','):
            vname = vname.strip()
            if vname and self._scope_stack and vname in self._scope_stack[-1]:
                run_vars[vname] = self._scope_stack[-1].get(vname, 0)
                self.variables[vname] = run_vars[vname]
        return True, ExecResult.ADVANCE

    # ── Scope management ───────────────────────────────────────────────

    def _push_scope(self) -> None:
        self._scope_stack.append(dict(self.variables))

    def _pop_scope(self) -> None:
        # Save STATIC vars before restoring
        if self._call_stack:
            sub_name = self._call_stack[-1].get('sub_name') or self._call_stack[-1].get('func_name', '')
            if sub_name and sub_name in self._static_vars:
                for vname in self._static_vars[sub_name]:
                    self._static_vars[sub_name][vname] = self.variables.get(vname, 0)
        if self._scope_stack:
            self.variables.clear()
            self.variables.update(self._scope_stack.pop())
