"""QBASIC string support — string variables, functions, and operations."""

from __future__ import annotations

import math
from typing import Any


def _left(s: str, n: float) -> str:
    return s[:int(n)]

def _right(s: str, n: float) -> str:
    return s[-int(n):] if int(n) > 0 else ''

def _mid(s: str, start: float, length: float = -1) -> str:
    i = int(start) - 1  # 1-based in BASIC
    if length < 0:
        return s[i:]
    return s[i:i + int(length)]

def _instr(haystack: str, needle: str) -> float:
    pos = haystack.find(needle)
    return float(pos + 1) if pos >= 0 else 0.0  # 1-based, 0 = not found

def _asc(s: str) -> float:
    return float(ord(s[0])) if s else 0.0

def _chr_fn(n: float) -> str:
    return chr(int(n))

def _str_fn(n: float) -> str:
    v = float(n)
    return str(int(v)) if v == int(v) else str(v)

def _val_fn(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

def _hex_fn(n: float) -> str:
    return format(int(n), 'X')

def _bin_fn(n: float) -> str:
    return format(int(n), 'b')

def _len_fn(x: Any) -> float:
    if isinstance(x, str):
        return float(len(x))
    if isinstance(x, (list, tuple)):
        return float(len(x))
    return float(len(str(x)))


# Functions that return strings (names ending in $)
STRING_FUNCS: dict[str, Any] = {
    'LEFT$': _left, 'RIGHT$': _right, 'MID$': _mid,
    'CHR$': _chr_fn, 'STR$': _str_fn, 'HEX$': _hex_fn, 'BIN$': _bin_fn,
}

# Functions that take/return mixed types
MIXED_FUNCS: dict[str, Any] = {
    'ASC': _asc, 'VAL': _val_fn, 'INSTR': _instr, 'LEN': _len_fn,
}


class StringMixin:
    """String variable and function support for QBasicTerminal.

    Requires: TerminalProtocol — uses self.variables, self._safe_eval().
    """

    def _is_string_var(self, name: str) -> bool:
        return name.endswith('$')

    def _get_string_ns(self) -> dict[str, Any]:
        """Return namespace entries for string functions."""
        ns: dict[str, Any] = {}
        ns.update(STRING_FUNCS)
        ns.update(MIXED_FUNCS)
        # String variables
        for k, v in self.variables.items():
            if isinstance(v, str):
                ns[k] = v
        return ns

    def _eval_string_expr(self, expr: str) -> str | float:
        """Evaluate an expression that might return a string."""
        expr = expr.strip()
        # Quoted string literal
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            return expr[1:-1]
        # String variable
        if expr in self.variables and isinstance(self.variables[expr], str):
            return self.variables[expr]
        # Try numeric
        try:
            return self.eval_expr(expr)
        except Exception:
            return expr

    def cmd_let_str(self, name: str, expr: str) -> None:
        """Assign a string value to a string variable."""
        val = self._eval_string_expr(expr)
        self.variables[name] = val
        self.io.writeln(f"{name} = {val!r}" if isinstance(val, str) else f"{name} = {val}")
