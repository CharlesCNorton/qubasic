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


class ExpressionMixin:
    """AST-based safe expression evaluation. No eval().

    Requires: TerminalProtocol — uses self.variables, self.arrays.
    """

    _SAFE_FUNCS = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'atan2': math.atan2,
        'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
        'abs': abs, 'int': int, 'float': float,
        'min': min, 'max': max, 'round': round, 'len': len,
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
        ast.And: lambda a, b: a and b,
        ast.Or: lambda a, b: a or b,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
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
            left = self._ast_eval(node.left, ns)
            for op_node, comparator in zip(node.ops, node.comparators):
                op = self._AST_OPS.get(type(op_node))
                if op is None:
                    raise ValueError(f"UNSUPPORTED OP: {type(op_node).__name__}")
                right = self._ast_eval(comparator, ns)
                if not op(left, right):
                    return False
                left = right
            return True
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
        # Rewrite numeric literals (&H, &B, $hex addresses) and the string
        # sigil, each only outside quoted string literals.
        expr_str = _replace_dollar_outside_strings(expr_str)
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
        """Evaluate a boolean condition."""
        cond = _rewrite_logical_outside_strings(cond)
        return bool(self._safe_eval(cond, extra_ns=run_vars))

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
