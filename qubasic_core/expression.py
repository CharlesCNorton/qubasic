"""QUBASIC expression evaluation on the QBASIC-precedence engine (basic_expr)."""

from __future__ import annotations

import ast
import math
import operator
import time
from typing import Any

from qubasic_core import basic_expr
from qubasic_core.host import HostState


def _basic_round(x: Any, ndigits: Any = None) -> float:
    """Round half away from zero (BASIC convention), not Python's half-to-even.

    round(2.5) == 3, round(-2.5) == -3, round(2.345, 2) == 2.35.
    """
    nd = int(ndigits) if ndigits is not None else 0
    f = 10 ** nd
    y = float(x) * f
    r = math.floor(y + 0.5) if y >= 0 else math.ceil(y - 0.5)
    return (r / f) if nd else float(r)


def _sgn(x: Any) -> int:
    return (x > 0) - (x < 0)


def _cint(x: Any) -> int:
    """CINT: nearest integer, ties to even (QBASIC), within -32768..32767."""
    v = basic_expr.round_half_even(x)
    if not -32768 <= v <= 32767:
        raise ValueError("OVERFLOW")
    return v


def _clng(x: Any) -> int:
    """CLNG: nearest integer, ties to even, within the 32-bit range."""
    v = basic_expr.round_half_even(x)
    if not -2147483648 <= v <= 2147483647:
        raise ValueError("OVERFLOW")
    return v


class _Names:
    """Identifier resolution for one evaluation: the caller's run variables,
    then program variables, constants, arrays and functions."""

    __slots__ = ('t', 'extra')

    def __init__(self, terminal: Any, extra: Any):
        self.t = terminal
        self.extra = extra

    def _variable(self, name: str) -> tuple[bool, Any]:
        extra = self.extra
        if extra is not None and name in extra:
            return True, extra[name]
        variables = self.t.variables
        if name in variables:
            return True, variables[name]
        return False, None

    def name(self, name: str) -> Any:
        t = self.t
        cb = t._classical_bits
        if cb and name in cb:
            # A mid-circuit measurement bit read as a number: the run goes live.
            from qubasic_core.live import MidCircuitRead
            t._live_needed = True
            raise MidCircuitRead(f"'{name}' is a mid-circuit measurement bit")
        found, val = self._variable(name)
        if found:
            return val
        if '.' in name:
            base, path = name.split('.', 1)
            found, rec = self._variable(base)
            if found and isinstance(rec, dict):
                return t._record_get(rec, path)
        up = name.upper()
        if up in _CONSTANTS:
            return _CONSTANTS[up]
        if up in _ZERO_ARG:
            return getattr(t, _ZERO_ARG[up])()
        if name in t.arrays:
            return t.arrays[name]
        if not t._strict_names:
            # Classic BASIC: an unassigned variable reads as 0 ("" for name$).
            return '' if name.endswith('$') else 0
        raise ValueError(f"UNDEFINED: {name}")

    def call(self, name: str, args: list) -> Any:
        t = self.t
        if name in t.arrays:
            return t._array_read(name, args)
        found, val = self._variable(name)
        if found and callable(val):
            return _apply(name, val, args)
        up = name.upper()
        fns = t._user_fns
        key = up if up.startswith('FN') else 'FN' + up
        if key in fns:
            f = fns[key]
            return t._call_user_fn_expr(f['params'], f['body'], tuple(args))
        if up in t._func_defs:
            return t._invoke_function(up, list(args), t._function_lines())
        fn = t._builtin(up)
        if fn is None:
            if not t._strict_names:
                # An array used before DIM reads as zero, as in QBASIC.
                return '' if name.endswith('$') else 0
            raise ValueError(f"NOT A FUNCTION: {name}")
        return _apply(name, fn, args)

    def index(self, name: str, args: list) -> Any:
        if name in self.t.arrays and len(args) == 1:
            return self.t.arrays[name][int(args[0])]
        raise ValueError(f"UNSUPPORTED SUBSCRIPT: {name}[...]")

    def bound(self, fn: str, name: str, args: list) -> int:
        return self.t._array_bound(fn, name, args[0] if args else 1)

    def array_ref(self, name: str) -> Any:
        """`name()` as an argument: the whole array, when there is one."""
        if name in self.t.arrays:
            from qubasic_core.variables import ArrayRef
            return ArrayRef(name)
        return None


# Names that read as a value with no argument list (and also accept "()").
_ZERO_ARG = {
    'TIMER': '_run_timer', 'RND': '_rnd', 'DATE$': '_date_str', 'TIME$': '_time_str',
    'INKEY$': '_inkey', 'FREEFILE': '_freefile',
}


def _apply(name: str, fn: Any, args: list) -> Any:
    try:
        return fn(*args)
    except ValueError as e:
        raise ValueError(f"{name}: {e}") from None
    except ZeroDivisionError:
        raise ValueError("DIVISION BY ZERO") from None


_CONSTANTS = {
    'PI': math.pi, 'TAU': math.tau, 'E': math.e, 'SQRT2': math.sqrt(2),
    'TRUE': True, 'FALSE': False,
}

_BUILTINS: dict[str, Any] = {}


def _builtin_table(safe_funcs: dict[str, Any]) -> dict[str, Any]:
    """Upper-case name -> function: the math functions plus the string and
    mixed functions (built on first use)."""
    if not _BUILTINS:
        from qubasic_core.strings import STRING_FUNCS, MIXED_FUNCS
        _BUILTINS.update({k.upper(): v for k, v in safe_funcs.items()})
        _BUILTINS.update(STRING_FUNCS)
        _BUILTINS.update(MIXED_FUNCS)
    return _BUILTINS


class ExpressionMixin(HostState):
    """Expression evaluation for QBasicTerminal (no eval()).

    Requires: TerminalProtocol — uses self.variables, self.arrays.
    """

    # Unassigned names are an error while this is set. A running program
    # clears it, so its unassigned variables read as 0 (or ""), as in BASIC;
    # qubit indices, gate parameters and matrix literals stay strict.
    _strict_names: bool = True

    _SAFE_FUNCS = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'atan2': math.atan2,
        'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
        # INT floors toward negative infinity, as in QBASIC (INT(-3.2) = -4).
        # FIX truncates toward zero (FIX(-3.2) = -3) for the other convention.
        'abs': abs, 'int': math.floor, 'fix': math.trunc, 'float': float,
        'min': min, 'max': max, 'round': _basic_round, 'len': len,
        'ceil': math.ceil, 'floor': math.floor,
        # QBASIC spellings: SGN, SQR, ATN, and the conversions (CINT rounds).
        'sgn': _sgn, 'sqr': math.sqrt, 'atn': math.atan,
        'cint': _cint, 'clng': _clng, 'csng': float, 'cdbl': float,
    }
    _SAFE_CONSTS = {
        'PI': math.pi, 'pi': math.pi,
        'TAU': math.tau, 'tau': math.tau,
        'E': math.e, 'e': math.e,
        'SQRT2': math.sqrt(2), 'sqrt2': math.sqrt(2),
        'True': True, 'False': False,
    }

    def _eval_operand(self, expr: str, run_vars: dict[str, Any] | None = None) -> Any:
        """Evaluate a quantum operand (gate parameter, qubit index) with
        unassigned names rejected."""
        prev, self._strict_names = self._strict_names, True
        try:
            if run_vars is None:
                return self.eval_expr(expr)
            return self._eval_with_vars(expr, run_vars)
        finally:
            self._strict_names = prev

    # ── Built-in functions ──────────────────────────────────────────────

    def _builtin(self, up: str) -> Any:
        """The built-in function named `up` (upper case), or None."""
        fn = _builtin_table(self._SAFE_FUNCS).get(up)
        if fn is not None:
            return fn
        if up == 'PEEK':
            return self._peek
        if up == 'USR':
            return self._usr_fn
        if up == 'EOF':
            return self._eof
        if up == 'LOF':
            return self._lof
        if up in _ZERO_ARG:
            method = getattr(self, _ZERO_ARG[up])
            return method if up == 'RND' else (lambda *a: method())
        if up == 'POS':
            return lambda x=0: 0
        if up == 'FRE':
            try:
                import psutil
                return lambda x=0: psutil.virtual_memory().available
            except ImportError:
                return lambda x=0: 0
        return None

    def _function_lines(self) -> Any:
        """The compiled program FUNCTION bodies run from (with the FUNCTION
        table indexed against it)."""
        prog = self._compiled()
        if not self._func_defs and prog.funcs:
            self._scan_subs(prog)
        return prog

    # ── Evaluation ──────────────────────────────────────────────────────

    def _safe_eval(self, expr: Any, extra_ns: Any = None) -> Any:
        """Evaluate a BASIC expression (see basic_expr for the grammar)."""
        tree = basic_expr.parse(str(expr))
        return basic_expr.evaluate(tree, _Names(self, extra_ns))

    def eval_expr(self, expr: str) -> float:
        """Evaluate a mathematical expression with variables."""
        try:
            return float(self._safe_eval(expr))
        except ValueError:
            raise
        except Exception:
            raise ValueError(f"CANNOT EVALUATE: {expr}")

    def _eval_int(self, expr: Any, run_vars: dict[str, Any] | None = None) -> int:
        """Evaluate to an exact integer for qubit indices and memory addresses.

        Avoids the float round-trip of eval_expr so large hex/bit values and
        indices keep full integer precision.
        """
        return int(self._safe_eval(expr, extra_ns=run_vars))

    def _eval_with_vars(self, expr: str, run_vars: Any) -> float | Any:
        """Evaluate expression with runtime variables.

        Returns a float, except that a qiskit ParameterExpression (present
        only during a parametric MINIMIZE/GRADIENT/SWEEP build) passes
        through unfloated so it lands in the circuit as a bindable parameter.
        """
        val = self._safe_eval(expr, extra_ns=run_vars)
        try:
            return float(val)
        except TypeError:
            try:
                from qiskit.circuit import ParameterExpression
            except ImportError:
                raise
            if isinstance(val, ParameterExpression):
                return val
            raise

    def _eval_condition(self, cond: str, run_vars: Any) -> bool:
        """Evaluate a condition: any non-zero value is true."""
        return bool(self._safe_eval(cond, extra_ns=run_vars))

    # ── Matrix literals ─────────────────────────────────────────────────

    _MATRIX_OPS: dict[type, Any] = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _parse_matrix(self, text: str) -> list[Any]:
        """Parse a matrix literal like [[1,0],[0,1j]] (complex entries; the
        constants, math functions and j/im allowed) without eval()."""
        ns = {**self._SAFE_CONSTS, **self._SAFE_FUNCS, 'j': 1j, 'im': 1j}
        for k, v in list(ns.items()):
            ns.setdefault(k.upper(), v)
        try:
            tree = ast.parse(text.strip(), mode='eval')
        except SyntaxError as e:
            raise ValueError(f"MATRIX SYNTAX ERROR: {e}") from None
        return self._ast_eval_matrix(tree.body, ns)

    def _ast_eval_matrix(self, node: ast.AST, ns: dict[str, Any]) -> Any:
        """Evaluate an AST node that should resolve to a nested list of numbers."""
        if isinstance(node, ast.List):
            return [self._ast_eval_matrix(elt, ns) for elt in node.elts]
        return complex(self._ast_eval(node, ns))

    def _ast_eval(self, node: ast.AST, ns: dict[str, Any]) -> Any:
        """One matrix entry: numbers, names from `ns`, + - * / ** and calls."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in ns:
                return ns[node.id]
            raise ValueError(f"UNDEFINED: {node.id}")
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._MATRIX_OPS:
            return self._MATRIX_OPS[type(node.op)](self._ast_eval(node.operand, ns))
        if isinstance(node, ast.BinOp) and type(node.op) in self._MATRIX_OPS:
            try:
                return self._MATRIX_OPS[type(node.op)](self._ast_eval(node.left, ns),
                                                       self._ast_eval(node.right, ns))
            except ZeroDivisionError:
                raise ValueError("DIVISION BY ZERO") from None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = ns.get(node.func.id)
            if callable(func):
                return func(*[self._ast_eval(a, ns) for a in node.args])
            raise ValueError(f"NOT A FUNCTION: {node.func.id}")
        raise ValueError(f"UNSUPPORTED IN A MATRIX: {ast.dump(node)}")

    # ── Names ───────────────────────────────────────────────────────────

    # Built-in constant names that may not be used as variable/array names, so a
    # value like E never silently shadows (or is shadowed by) the constant.
    _RESERVED_CONST_NAMES = frozenset({'PI', 'TAU', 'E', 'SQRT2', 'TRUE', 'FALSE'})

    def _assert_assignable(self, name: str) -> None:
        """Reject assignment to a built-in constant name (case-insensitive)
        or to a CONST."""
        if name.upper() in self._RESERVED_CONST_NAMES:
            raise ValueError(
                f"RESERVED: '{name}' is a built-in constant; choose another name")
        if name in self._consts:
            raise ValueError(f"DUPLICATE DEFINITION: '{name}' is a CONST")

    def _run_timer(self) -> float:
        """TIMER: seconds since midnight, local time (as in QBASIC)."""
        now = time.localtime()
        return now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec + (time.time() % 1.0)

    @staticmethod
    def _date_str() -> str:
        """DATE$: today's date as mm-dd-yyyy."""
        return time.strftime('%m-%d-%Y')

    @staticmethod
    def _time_str() -> str:
        """TIME$: the time of day as hh:mm:ss."""
        return time.strftime('%H:%M:%S')

    def _call_user_fn_expr(self, params: list[str], body: str, args: tuple) -> float:
        """Call a DEF FN function from within expression evaluation."""
        if len(args) < len(params):
            missing = params[len(args):]
            raise ValueError(
                f"DEF FN requires {len(params)} argument(s), got {len(args)}; "
                f"missing: {', '.join(missing)}"
            )
        ns: dict[str, Any] = {}
        for i, pname in enumerate(params):
            ns[pname] = args[i]
        return float(self._safe_eval(body, extra_ns=ns))
