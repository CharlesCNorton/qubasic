"""QUBASIC SUB/FUNCTION — procedures with their own variables, as in QBASIC.

A procedure starts with only its parameters, its STATIC variables, the CONST
names and the variables made module-wide by DIM SHARED; `SHARED x` inside
the procedure brings in one more of the caller's variables. Everything else
it assigns is local and is gone when it returns, arrays it creates included.

Parameters take QBASIC's forms: `n`, `n%`, `n AS INTEGER`, `s AS STRING`,
`p AS Point` (a record) and `a()` or `a() AS INTEGER` (a whole array). An
argument that is a plain variable, an array element or a record field is
passed by reference: the procedure's final value of the parameter is written
back to it. A whole array is always passed by reference (`Sort values()`).
`SUB name(...) STATIC` keeps every local variable between calls; `STATIC x`
keeps one. A FUNCTION returns the value last assigned to its own name. LOCAL
(a QUBASIC extension) zeroes a name inside a procedure.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, NamedTuple

from qubasic_core.engine import (
    ExecResult, ExecOutcome,
    RE_SUB, RE_END_SUB, RE_FUNCTION, RE_END_FUNCTION,
    RE_CALL, RE_LOCAL, RE_STATIC_DECL, RE_SHARED,
)
from qubasic_core.exec_context import ExecContext
from qubasic_core.host import HostState
from qubasic_core.variables import ArrayRef, type_text

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


class SubroutineMixin(HostState):
    """SUB/FUNCTION procedures with local scope, typed and array parameters,
    by-reference arguments and LOCAL/STATIC/SHARED declarations (see the
    module docstring)."""

    def _init_subs(self) -> None:
        self._sub_defs: dict[str, dict[str, Any]] = {}
        self._func_defs: dict[str, dict[str, Any]] = {}
        self._scope_stack: list[dict[str, Any]] = []
        self._static_vars: dict[str, dict[str, Any]] = {'_GLOBAL': {}}
        self._call_stack: list[dict[str, Any]] = []
        self._func_call_depth: int = 0
        # Per scope level: the names whose changes survive _pop_scope, the
        # caller's declared types, the arrays that existed before the call,
        # and the arrays bound to array parameters.
        self._shared_stack: list[set[str]] = []
        self._type_stack: list[dict[str, str]] = []
        self._array_stack: list[set[str]] = []
        self._alias_stack: list[list[tuple[str, str, Any]]] = []

    def _scan_subs(self, prog: Any = None) -> None:
        """Index the program's SUB and FUNCTION blocks (statement indices of
        the compiled program)."""
        prog = prog if hasattr(prog, 'subs') else self._compiled()
        self._sub_defs.clear()
        self._func_defs.clear()
        statics: set[str] = getattr(prog, 'static_routines', set())
        for table, out in ((prog.subs, self._sub_defs), (prog.funcs, self._func_defs)):
            for name, (start, end, params) in table.items():
                out[name] = {'params': params, 'start_ip': start, 'end_ip': end,
                             'static': name in statics}

    # ── Scope ──────────────────────────────────────────────────────────

    def _module_wide(self, name: str) -> bool:
        """Whether a caller's variable stays visible inside a procedure."""
        base = name.split('.', 1)[0]
        return (name in self._global_shared or base in self._global_shared
                or name in self._consts or name.startswith('_')
                or name in ('ERR', 'ERL'))

    def _push_scope(self) -> None:
        """Enter a procedure: keep the caller's variables, declared types and
        arrays for _pop_scope and leave only the module-wide variables visible."""
        caller = dict(self.variables)
        self._scope_stack.append(caller)
        self._shared_stack.append(set(n for n in caller if self._module_wide(n)))
        self._type_stack.append(dict(self._var_types))
        self._array_stack.append(set(self.arrays))
        self._alias_stack.append([])
        self.variables.clear()
        self.variables.update({k: v for k, v in caller.items() if self._module_wide(k)})

    def _pop_scope(self, frame: dict[str, Any] | None = None) -> None:
        """Leave a procedure: save its STATIC variables, unbind its array
        parameters, drop its local arrays, restore the caller's variables and
        declared types, and carry back the SHARED and module-wide variables
        it changed."""
        if frame is None and self._call_stack:
            frame = self._call_stack[-1]
        if frame is not None:
            name = frame.get('sub_name') or frame.get('func_name') or ''
            if frame.get('static'):
                params = set(frame.get('params', ()))
                keep = self._shared_stack[-1] if self._shared_stack else set()
                self._static_vars[name] = {k: v for k, v in self.variables.items()
                                           if k not in params and k not in keep
                                           and not self._module_wide(k)}
            elif name in self._static_vars:
                for vname in list(self._static_vars[name]):
                    self._static_vars[name][vname] = self.variables.get(vname, 0)
        if self._alias_stack:
            self._unbind_arrays(self._alias_stack.pop())
        if self._array_stack:
            before = self._array_stack.pop()
            for arr in list(self.arrays):
                if arr not in before and arr not in self._global_shared:
                    self._drop_array(arr)
        if self._type_stack:
            shared_types = {k: v for k, v in self._var_types.items() if k in self._global_shared}
            self._var_types.clear()
            self._var_types.update(self._type_stack.pop())
            self._var_types.update(shared_types)
        if self._scope_stack:
            shared = self._shared_stack.pop() if self._shared_stack else set()
            shared_vals = {v: self.variables[v] for v in shared if v in self.variables}
            self.variables.clear()
            self.variables.update(self._scope_stack.pop())
            self.variables.update(shared_vals)

    # ── Array parameters ───────────────────────────────────────────────

    def _array_entry(self, name: str) -> tuple | None:
        if name not in self.arrays:
            return None
        return (self.arrays[name], self._array_dims.get(name), self._array_lb.get(name),
                self._array_types.get(name), name in self._dimmed_arrays)

    def _set_array_entry(self, name: str, entry: tuple | None) -> None:
        self._drop_array(name)
        if entry is None:
            return
        data, dims, lows, typ, dimmed = entry
        self.arrays[name] = data
        if dims is not None:
            self._array_dims[name] = dims
        if lows is not None:
            self._array_lb[name] = lows
        if typ is not None:
            self._array_types[name] = typ
        if dimmed:
            self._dimmed_arrays.add(name)

    def _bind_array(self, param: str, source: str) -> None:
        """Make array parameter `param` the caller's array `source`."""
        if source not in self.arrays:
            raise ValueError(f"SUBSCRIPT OUT OF RANGE: array {source} is not dimensioned")
        if param == source:
            return
        self._alias_stack[-1].append((param, source, self._array_entry(param)))
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
                self._var_types[p.name] = p.type
            self.variables[p.name] = self._coerce(p.name, val)
        for k, v in self._static_vars.get(name, {}).items():
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

    def _cf_sub(self, stmt: str, sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        """SUB — skip the block during normal execution."""
        if parsed is not None:
            name = parsed.name
        else:
            m = RE_SUB.match(stmt)
            if not m:
                return None
            name = m.group(1).upper()
        if name in self._sub_defs:
            return True, self._sub_defs[name]['end_ip'] + 1
        return True, ExecResult.ADVANCE

    def _cf_end_sub(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None and not RE_END_SUB.match(stmt):
            return None
        if self._call_stack:
            frame = self._call_stack.pop()
            back = [(lv, self.variables.get(p)) for p, lv in frame.get('byref', ())]
            self._pop_scope(frame)
            for lv, value in back:
                if value is not None:
                    self._write_lvalue(lv, value)
            return True, frame['return_ip']
        return True, ExecResult.ADVANCE

    def _cf_function(self, stmt: str, sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        """FUNCTION — skip the block during normal execution."""
        if parsed is not None:
            name = parsed.name
        else:
            m = RE_FUNCTION.match(stmt)
            if not m:
                return None
            name = m.group(1).upper()
        if name in self._func_defs:
            return True, self._func_defs[name]['end_ip'] + 1
        return True, ExecResult.ADVANCE

    def _cf_end_function(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None and not RE_END_FUNCTION.match(stmt):
            return None
        return True, ExecResult.ADVANCE

    def _cf_call(self, stmt: str, run_vars: dict[str, Any],
                 sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            name = parsed.name
            arg_str = parsed.args
        else:
            m = RE_CALL.match(stmt)
            if not m:
                return None
            name = m.group(1).upper()
            arg_str = m.group(2) or ''
        if name not in self._sub_defs:
            raise RuntimeError(f"SUBPROGRAM NOT DEFINED: {name}")
        sub = self._sub_defs[name]
        arg_texts = self._split_arg_list(arg_str) if arg_str.strip() else []
        args = self._call_args(sub['params'], arg_texts, run_vars)
        names = [parse_param(t).name for t in sub['params']]
        byref = []
        for pname, text, val in zip(names, arg_texts, args):
            if isinstance(val, ArrayRef):
                continue
            if (_names_storage(text) and not self._is_function_name(text)
                    and text.strip() not in self._consts):
                byref.append((pname, self._lvalue(text, run_vars)))
        self._push_scope()
        self._call_stack.append({'return_ip': ip + 1, 'sub_name': name, 'byref': byref,
                                 'params': names, 'static': sub.get('static')})
        try:
            self._bind_params(name, sub['params'], args)
        except Exception:
            self._pop_scope(self._call_stack.pop())
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
        return (up in self._func_defs or up in self._user_fns or ('FN' + up) in self._user_fns
                or self._builtin(up) is not None)

    def _invoke_function(self, name: str, args: list[Any], prog: Any) -> Any:
        """Execute a FUNCTION block and return its value (called from the
        expression evaluator). Its statements run through control-flow
        dispatch, one compiled statement at a time."""
        self._func_call_depth += 1
        if self._func_call_depth > 100:
            self._func_call_depth -= 1
            raise RuntimeError("FUNCTION CALL DEPTH LIMIT (100)")
        try:
            prog = self._as_program(prog)
            uname = name.upper()
            if uname not in self._func_defs:
                raise ValueError(f"FUNCTION NOT DEFINED: {name}")
            func = self._func_defs[uname]
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
                    max_iterations=self._max_iterations,
                )

                def _no_clause(s, ls, sl, i, rv):
                    return None

                self._call_stack.append(frame)
                depth = len(self._call_stack)
                try:
                    # Run to this FUNCTION's END FUNCTION; a SUB it calls runs
                    # wherever it sits and returns here.
                    while ctx.ip < len(prog) and not (
                            ctx.ip == func['end_ip'] and len(self._call_stack) == depth):
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
                    del self._call_stack[depth - 1:]
                ret_val = next((v for k, v in self.variables.items() if k.upper() == uname),
                               '' if uname.endswith('$') else 0)
            finally:
                self._pop_scope(frame)
            return ret_val
        finally:
            self._func_call_depth -= 1

    # ── LOCAL / STATIC / SHARED ────────────────────────────────────────

    def _declared_names(self, var_list: str) -> list[Param]:
        return [parse_param(v) for v in self._split_arg_list(var_list)]

    def _cf_local(self, stmt: str, run_vars: dict[str, Any], *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            var_list = parsed.var_list
        else:
            m = RE_LOCAL.match(stmt)
            if not m:
                return None
            var_list = m.group(1)
        for p in self._declared_names(var_list):
            if p.type is not None:
                self._var_types[p.name] = p.type
            value = self._default_value(p.name)
            run_vars[p.name] = value
            self.variables[p.name] = value
        return True, ExecResult.ADVANCE

    def _cf_static(self, stmt: str, run_vars: dict[str, Any], *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            var_list = parsed.var_list
        else:
            m = RE_STATIC_DECL.match(stmt)
            if not m:
                return None
            var_list = m.group(1)
        frame = self._call_stack[-1] if self._call_stack else {}
        sub_name = frame.get('sub_name') or frame.get('func_name') or '_GLOBAL'
        if sub_name not in self._static_vars:
            self._static_vars[sub_name] = {}
        store = self._static_vars[sub_name]
        for p in self._declared_names(var_list):
            if p.array:
                if self._array_stack:
                    self._array_stack[-1].add(p.name)       # keeps its elements between calls
                continue
            if p.type is not None:
                self._var_types[p.name] = p.type
            val = store.get(p.name, self._default_value(p.name))
            store[p.name] = val
            run_vars[p.name] = val
            self.variables[p.name] = val
        return True, ExecResult.ADVANCE

    def _cf_shared(self, stmt: str, run_vars: dict[str, Any], *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            var_list = parsed.var_list
        else:
            m = RE_SHARED.match(stmt)
            if not m:
                return None
            var_list = m.group(1)
        for p in self._declared_names(var_list):
            if p.array:
                if self._array_stack:
                    self._array_stack[-1].add(p.name)       # the caller's array, not a local
                continue
            if not self._scope_stack:
                continue
            if p.type is not None:
                self._var_types[p.name] = p.type
            caller = self._scope_stack[-1]
            value = caller.get(p.name, self._default_value(p.name))
            run_vars[p.name] = value
            self.variables[p.name] = value
            if self._shared_stack:
                self._shared_stack[-1].add(p.name)
        return True, ExecResult.ADVANCE
