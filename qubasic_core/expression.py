"""QUBASIC expression evaluation — safe AST-based evaluator."""

from __future__ import annotations

import re
import math
import ast
import time
import random
import operator
from typing import Any


def _replace_dollar_outside_strings(expr_str: str) -> str:
    """Rewrite numeric literals and the string sigil, only outside quoted strings.

    Single pass: splits on quote boundaries and, in each unquoted segment,
    applies (in order):
      - ``&HFF``  -> ``0xFF``           (hex literal)
      - ``&B101`` -> ``0b101``          (binary literal)
      - ``$D000`` -> ``0xD000``         (leading-$ hex address, e.g. PEEK $D000)
      - ``name$`` -> ``name_S_``        (trailing-$ string-variable sigil)

    The ``$``-address rewrite uses a negative lookbehind so it never touches a
    trailing string sigil; the two never collide because an address ``$`` is
    always preceded by start/space/operator and a sigil ``$`` always follows a
    word character.
    """
    parts: list[str] = []
    i = 0
    n = len(expr_str)
    while i < n:
        ch = expr_str[i]
        if ch in ('"', "'"):
            quote = ch
            j = i + 1
            while j < n and expr_str[j] != quote:
                j += 1
            parts.append(expr_str[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and expr_str[j] not in ('"', "'"):
                j += 1
            segment = expr_str[i:j]
            segment = re.sub(r'&H([0-9A-Fa-f]+)', r'0x\1', segment)
            segment = re.sub(r'&B([01]+)', r'0b\1', segment)
            segment = re.sub(r'(?<![\w$])\$([0-9A-Fa-f]+)', r'0x\1', segment)
            segment = re.sub(r'(\w+)\$', r'\1_S_', segment)
            parts.append(segment)
            i = j
    return ''.join(parts)


_LOGICAL_SUBS = [
    (re.compile(r'<>|><'), '!='),
    (re.compile(r'\bAND\b', re.IGNORECASE), ' and '),
    (re.compile(r'\bOR\b', re.IGNORECASE), ' or '),
    (re.compile(r'\bNOT\b', re.IGNORECASE), ' not '),
    # XOR maps to bitwise ^, matching BASIC's bitwise logical operators;
    # for 0/1 truth values this is exactly logical exclusive-or.
    (re.compile(r'\bXOR\b', re.IGNORECASE), ' ^ '),
]


def _rewrite_logical_outside_strings(cond: str) -> str:
    """Rewrite BASIC logical/relational operators to Python, only outside strings.

    Word-boundary anchors already protect identifiers like NOTE or ANDY; this
    additionally protects operator-like text inside quoted string literals
    (e.g. ``IF s$ == "AND"``), which the old whole-string regex corrupted.
    """
    parts: list[str] = []
    i = 0
    n = len(cond)
    while i < n:
        ch = cond[i]
        if ch in ('"', "'"):
            quote = ch
            j = i + 1
            while j < n and cond[j] != quote:
                j += 1
            parts.append(cond[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and cond[j] not in ('"', "'"):
                j += 1
            seg = cond[i:j]
            for pat, repl in _LOGICAL_SUBS:
                seg = pat.sub(repl, seg)
            parts.append(seg)
            i = j
    return ''.join(parts)


def _as_int(x: Any) -> int:
    """Coerce a BASIC value to int for bitwise AND/OR/XOR.

    QBasic rounds non-integers; truth values and non-numerics fold to 1/0 so
    ``(a > 0) AND (b > 0)`` works the same as ``6 AND 3``.
    """
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(round(x))
    return 1 if x else 0


def _basic_and(a: Any, b: Any) -> int:
    """BASIC AND — bitwise on integers, and correct for truth values
    (1 & 1 == 1, 1 & 0 == 0). Parsed from ``and`` so precedence stays below
    comparison: ``a > b AND c > d`` groups as ``(a>b) AND (c>d)``."""
    return _as_int(a) & _as_int(b)


def _basic_or(a: Any, b: Any) -> int:
    return _as_int(a) | _as_int(b)


def _basic_xor(a: Any, b: Any) -> int:
    return _as_int(a) ^ _as_int(b)


def _basic_round(x: Any, ndigits: Any = None) -> float:
    """Round half away from zero (BASIC convention), not Python's half-to-even.

    round(2.5) == 3, round(-2.5) == -3, round(2.345, 2) == 2.35.
    """
    nd = int(ndigits) if ndigits is not None else 0
    f = 10 ** nd
    y = float(x) * f
    r = math.floor(y + 0.5) if y >= 0 else math.ceil(y - 0.5)
    return (r / f) if nd else float(r)


class ExpressionMixin:
    """AST-based safe expression evaluation. No eval().

    Requires: TerminalProtocol — uses self.variables, self.arrays.
    """

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
    }
    _SAFE_CONSTS = {
        'PI': math.pi, 'pi': math.pi,
        'TAU': math.tau, 'tau': math.tau,
        'E': math.e, 'e': math.e,
        'SQRT2': math.sqrt(2), 'sqrt2': math.sqrt(2),
        'True': True, 'False': False,
    }
    _AST_OPS: dict[type, Any] = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
        ast.Not: operator.not_,
        ast.Eq: operator.eq, ast.NotEq: operator.ne,
        ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge,
        # AND/OR/XOR are BASIC's bitwise-logical operators (6 AND 3 == 2),
        # robust to float operands; NOT stays logical so IF NOT flag works
        # with 0/1 truth values.
        ast.And: _basic_and,
        ast.Or: _basic_or,
        ast.BitAnd: _basic_and,
        ast.BitOr: _basic_or,
        ast.BitXor: _basic_xor,
        ast.Invert: operator.invert,
    }

    def _ast_eval(self, node: ast.AST, ns: dict[str, Any]) -> Any:
        """Recursively evaluate an AST node against a safe namespace."""
        if isinstance(node, ast.Expression):
            return self._ast_eval(node.body, ns)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex, str)):
                return node.value
            raise ValueError(f"UNSUPPORTED CONSTANT: {node.value!r}")
        if isinstance(node, ast.Name):
            # A mid-circuit measurement bit has no classical value (it is
            # resolved per shot); reading it outside an IF feedforward is an
            # error rather than a silent placeholder 0.
            cb = getattr(self, '_classical_bits', None)
            if cb and node.id in cb:
                raise ValueError(
                    f"'{node.id}' is a mid-circuit measurement bit; its value is "
                    f"per-shot, so it can't be read in a classical expression "
                    f"(use IF {node.id} for feedforward, or LOCC mode for a live value)")
            if node.id in ns:
                return ns[node.id]
            raise ValueError(f"UNDEFINED: {node.id}")
        if isinstance(node, ast.UnaryOp):
            op = self._AST_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"UNSUPPORTED OP: {type(node.op).__name__}")
            return op(self._ast_eval(node.operand, ns))
        if isinstance(node, ast.BinOp):
            op = self._AST_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"UNSUPPORTED OP: {type(node.op).__name__}")
            try:
                return op(self._ast_eval(node.left, ns), self._ast_eval(node.right, ns))
            except ZeroDivisionError:
                raise ValueError("DIVISION BY ZERO") from None
        if isinstance(node, ast.BoolOp):
            op = self._AST_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"UNSUPPORTED OP: {type(node.op).__name__}")
            result = self._ast_eval(node.values[0], ns)
            for val in node.values[1:]:
                result = op(result, self._ast_eval(val, ns))
            return result
        if isinstance(node, ast.Compare):
            # Python-style chaining (a < b < c means (a<b) and (b<c)) so range
            # checks like 0 <= x <= 10 work as written; results use BASIC truth
            # values (-1 for true, 0 for false).
            left = self._ast_eval(node.left, ns)
            for op_node, comparator in zip(node.ops, node.comparators):
                op = self._AST_OPS.get(type(op_node))
                if op is None:
                    raise ValueError(f"UNSUPPORTED OP: {type(op_node).__name__}")
                right = self._ast_eval(comparator, ns)
                if not op(left, right):
                    return 0
                left = right
            return -1
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("ONLY SIMPLE FUNCTION CALLS ALLOWED")
            fname = node.func.id
            func = ns.get(fname)
            if not callable(func):
                raise ValueError(f"NOT A FUNCTION: {fname}")
            args = [self._ast_eval(a, ns) for a in node.args]
            try:
                return func(*args)
            except ValueError as e:
                raise ValueError(f"{fname}: {e}") from None
            except ZeroDivisionError:
                raise ValueError("DIVISION BY ZERO") from None
        if isinstance(node, ast.IfExp):
            if self._ast_eval(node.test, ns):
                return self._ast_eval(node.body, ns)
            return self._ast_eval(node.orelse, ns)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                name = node.value.id
                idx = int(self._ast_eval(node.slice, ns))
                if name in self.arrays:
                    return self.arrays[name][idx]
            raise ValueError("UNSUPPORTED SUBSCRIPT")
        if isinstance(node, ast.Attribute):
            # Record field access for user TYPEs: p.x resolves the flat 'p.x' key.
            if isinstance(node.value, ast.Name):
                key = f"{node.value.id}.{node.attr}"
                if key in ns:
                    return ns[key]
            raise ValueError(f"UNDEFINED FIELD: {getattr(node, 'attr', '?')}")
        raise ValueError(f"UNSUPPORTED EXPRESSION: {ast.dump(node)}")

    def _build_base_ns(self) -> dict[str, Any]:
        """Build the function/constant namespace once per instance.

        This contains everything that does not change between calls:
        safe math functions, constants, and instance-specific BASIC
        builtins (RND, TIMER, POS, PEEK, USR, EOF, FRE, string fns).
        Cached on self._base_ns and returned.
        """
        ns: dict[str, Any] = {**self._SAFE_CONSTS, **self._SAFE_FUNCS}
        # Instance-specific functions
        ns['RND'] = lambda x=0: random.random()
        ns['POS'] = lambda x=0: 0
        if hasattr(self, '_peek'):
            ns['PEEK'] = lambda addr: self._peek(addr)
        if hasattr(self, '_usr_fn'):
            ns['USR'] = lambda addr: self._usr_fn(addr)
        if hasattr(self, '_eof'):
            ns['EOF'] = lambda h: self._eof(h)
        if hasattr(self, '_get_string_ns'):
            ns.update(self._get_string_ns())
        try:
            import psutil
            ns['FRE'] = lambda x=0: psutil.virtual_memory().available
        except ImportError:
            ns['FRE'] = lambda x=0: 0
        # BASIC is case-insensitive for built-in functions and constants, so
        # register upper- and lower-case aliases for every builtin (SQRT and
        # sqrt, RND and rnd, LEFT$ and left$). Variables are merged with their
        # own case in _safe_eval and shadow these.
        for _name in list(ns.keys()):
            ns.setdefault(_name.upper(), ns[_name])
            ns.setdefault(_name.lower(), ns[_name])
        self._base_ns = ns
        return ns

    def _safe_eval(self, expr: Any, extra_ns: dict[str, Any] | None = None) -> Any:
        """Evaluate expression using AST walking — no eval()."""
        # Start from the cached base namespace (functions/constants)
        base = getattr(self, '_base_ns', None) or self._build_base_ns()
        ns = {**base}
        # TIMER must be evaluated fresh each call
        ns['TIMER'] = getattr(self, '_run_timer', time.time)()
        # Merge current variable state
        ns.update(self.variables)
        if extra_ns:
            ns.update(extra_ns)
        _dims = getattr(self, '_array_dims', {})
        _base = getattr(self, '_option_base', 0)
        for aname, adata in self.arrays.items():
            def _array_accessor(*indices, d=adata, n=aname, dims=_dims.get(aname),
                                base=_base):
                if dims and len(indices) > 1:
                    # Multi-dimensional: compute flat index from (i, j, ...)
                    flat = 0
                    stride = 1
                    for k in range(len(indices) - 1, -1, -1):
                        idx = int(indices[k]) - base
                        flat += idx * stride
                        stride *= dims[k] if k < len(dims) else 1
                    idx = flat
                else:
                    idx = (int(indices[0]) - base) if indices else 0
                if idx < 0 or idx >= len(d):
                    raise ValueError(f"ARRAY INDEX OUT OF RANGE: {n}[{idx + base}]")
                return d[idx]
            ns[aname] = _array_accessor
        # User-defined functions (DEF FN) — these can change at runtime
        if hasattr(self, '_user_fns'):
            for fname, fdef in self._user_fns.items():
                fn_params = fdef['params']
                fn_body = fdef['body']
                ns[fname] = lambda *args, p=fn_params, b=fn_body: self._call_user_fn_expr(p, b, args)
            # Also register without FN prefix and in lowercase for flexible invocation
            for fname, fdef in self._user_fns.items():
                fn_params = fdef['params']
                fn_body = fdef['body']
                fn = lambda *args, p=fn_params, b=fn_body: self._call_user_fn_expr(p, b, args)
                # Strip FN prefix if present, add as both upper and lower
                short = fname[2:] if fname.upper().startswith('FN') else fname
                ns[short] = fn
                ns[short.lower()] = fn
                ns[short.upper()] = fn
        # Register FUNCTION blocks as callables in expression context
        if hasattr(self, '_func_defs') and hasattr(self, '_invoke_function'):
            sorted_lines = sorted(self.program.keys()) if hasattr(self, 'program') else []
            for fname in self._func_defs:
                fn = lambda *args, n=fname, sl=sorted_lines: self._invoke_function(n, list(args), sl)
                ns[fname] = fn
                ns[fname.lower()] = fn
        # Add Python-safe aliases for all $-suffixed keys in namespace
        for k, v in list(ns.items()):
            if '$' in k:
                ns[k.replace('$', '_S_')] = v
        expr_str = str(expr).strip()
        # Normalize FN prefix: "FN square(x)" -> "square(x)"
        expr_str = re.sub(r'\bFN\s+(\w+)\s*\(', r'\1(', expr_str, flags=re.IGNORECASE)
        # Rewrite BASIC logical/relational operators (AND, OR, NOT, XOR, <>) to
        # the Python forms the AST understands, only outside quoted strings.
        # Applied to every expression (LET, PRINT, gate params), not just IF
        # conditions, so the documented operators work everywhere.
        expr_str = _rewrite_logical_outside_strings(expr_str)
        # Rewrite numeric literals (&H, &B, $hex addresses) and the string
        # sigil, each only outside quoted string literals.
        expr_str = _replace_dollar_outside_strings(expr_str)
        # The operator rewrite can pad a leading token (NOT x -> " not x"); strip
        # so ast.parse(mode='eval') does not reject it as an unexpected indent.
        expr_str = expr_str.strip()
        if not expr_str:
            raise ValueError("EMPTY EXPRESSION")
        try:
            tree = ast.parse(expr_str, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"SYNTAX ERROR: {e}") from None
        return self._ast_eval(tree, ns)

    def _parse_matrix(self, text: str) -> list[Any]:
        """Parse a matrix literal like [[1,0],[0,1]] using AST — no eval()."""
        ns = {**self._SAFE_CONSTS, **self._SAFE_FUNCS, 'j': 1j, 'im': 1j}
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
        val = self._safe_eval(expr, extra_ns=run_vars) if run_vars is not None \
            else self._safe_eval(expr)
        return int(val)

    def _eval_with_vars(self, expr: str, run_vars: dict[str, Any]) -> float:
        """Evaluate expression with runtime variables."""
        return float(self._safe_eval(expr, extra_ns=run_vars))

    def _eval_condition(self, cond: str, run_vars: dict[str, Any]) -> bool:
        """Evaluate a boolean condition.

        The operator rewrite (AND/OR/NOT/XOR/<>) now happens inside _safe_eval,
        so conditions and ordinary expressions share one consistent path.
        """
        return bool(self._safe_eval(cond, extra_ns=run_vars))

    # Built-in constant names that may not be used as variable/array names, so a
    # value like E never silently shadows (or is shadowed by) the constant.
    _RESERVED_CONST_NAMES = frozenset({'PI', 'TAU', 'E', 'SQRT2', 'TRUE', 'FALSE'})

    def _assert_assignable(self, name: str) -> None:
        """Reject assignment to a built-in constant name (case-insensitive)."""
        if name.upper() in self._RESERVED_CONST_NAMES:
            raise ValueError(
                f"RESERVED: '{name}' is a built-in constant; choose another name")

    def _run_timer(self) -> float:
        """Return elapsed time since terminal start."""
        return time.time() - getattr(self, '_start_time', time.time())

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
