"""QBASIC expression evaluation — safe AST-based evaluator."""

from __future__ import annotations

import re
import math
import ast
import operator
from typing import Any


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
    _AST_OPS = {
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
    }

    def _ast_eval(self, node: ast.AST, ns: dict[str, Any]) -> Any:
        """Recursively evaluate an AST node against a safe namespace."""
        if isinstance(node, ast.Expression):
            return self._ast_eval(node.body, ns)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
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
            return op(self._ast_eval(node.left, ns), self._ast_eval(node.right, ns))
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
            return func(*args)
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
        raise ValueError(f"UNSUPPORTED EXPRESSION: {ast.dump(node)}")

    def _safe_eval(self, expr: Any, extra_ns: dict[str, Any] | None = None) -> Any:
        """Evaluate expression using AST walking — no eval()."""
        ns = {**self._SAFE_CONSTS, **self._SAFE_FUNCS, **self.variables}
        if extra_ns:
            ns.update(extra_ns)
        for aname, adata in self.arrays.items():
            ns[aname] = lambda i, d=adata: d[int(i)]
        expr_str = str(expr).strip()
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

    def _eval_with_vars(self, expr: str, run_vars: dict[str, Any]) -> float:
        """Evaluate expression with runtime variables."""
        return float(self._safe_eval(expr, extra_ns=run_vars))

    def _eval_condition(self, cond: str, run_vars: dict[str, Any]) -> bool:
        """Evaluate a boolean condition."""
        cond = cond.replace('<>', '!=').replace('><', '!=')
        cond = re.sub(r'\bAND\b', ' and ', cond, flags=re.IGNORECASE)
        cond = re.sub(r'\bOR\b', ' or ', cond, flags=re.IGNORECASE)
        cond = re.sub(r'\bNOT\b', ' not ', cond, flags=re.IGNORECASE)
        return bool(self._safe_eval(cond, extra_ns=run_vars))
