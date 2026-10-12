"""QUBASIC string support — string variables, functions, and operations."""

from __future__ import annotations

import re
from typing import Any

from qubasic_core.basic_format import str_text
from qubasic_core.host import HostState


def _count(n: Any, what: str) -> int:
    k = int(round(float(n)))
    if k < 0:
        raise ValueError(f"ILLEGAL FUNCTION CALL: {what} needs a non-negative count")
    return k


def _left(s: str, n: float) -> str:
    return s[:_count(n, 'LEFT$')]


def _right(s: str, n: float) -> str:
    k = _count(n, 'RIGHT$')
    return s[-k:] if k > 0 else ''


def _mid(s: str, start: float, length: float = -1) -> str:
    i = int(round(float(start)))
    if i < 1:
        raise ValueError("ILLEGAL FUNCTION CALL: MID$ start must be 1 or more")
    i -= 1  # 1-based in BASIC
    if length < 0:
        return s[i:]
    return s[i:i + int(round(float(length)))]


def _instr(*args: Any) -> int:
    """INSTR([start,] haystack$, needle$): 1-based position, 0 when absent."""
    if len(args) == 3:
        start, haystack, needle = args
        begin = int(round(float(start)))
        if begin < 1:
            raise ValueError("ILLEGAL FUNCTION CALL: INSTR start must be 1 or more")
    elif len(args) == 2:
        (haystack, needle), begin = args, 1
    else:
        raise ValueError("INSTR takes ([start,] string, substring)")
    if begin > len(haystack):
        return 0
    pos = haystack.find(needle, begin - 1)
    return pos + 1 if pos >= 0 else 0


def _asc(s: str) -> float:
    return float(ord(s[0])) if s else 0.0


def _chr_fn(n: float) -> str:
    return chr(int(n))


def _str_fn(n: float) -> str:
    # BASIC reserves a leading space for the sign of a non-negative number,
    # so STR$(42) is " 42", STR$(-5) is "-5" and STR$(.5) is " .5".
    return str_text(n)


_RE_VAL_NUMBER = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?')


def _val_fn(s: str) -> float:
    """VAL: the number at the start of a string, as QBASIC reads it. Blanks
    are ignored, &H/&O/&B prefixes read hex/octal/binary, and reading stops
    at the first character that cannot continue a number (VAL("12ab") = 12)."""
    if not isinstance(s, str):
        raise ValueError("TYPE MISMATCH: VAL takes a string")
    t = re.sub(r'[ \t\n\r]', '', s)
    m = re.match(r'&([HhOoBb]?)([0-9A-Fa-f]*)', t)
    if m:
        base = {'H': 16, 'O': 8, 'B': 2, '': 8}[m.group(1).upper()]
        digits = ''
        for ch in m.group(2):
            if int(ch, 16) >= base:
                break
            digits += ch
        return float(int(digits, base)) if digits else 0.0
    m = _RE_VAL_NUMBER.match(t)
    if not m:
        return 0.0
    return float(m.group().replace('d', 'e').replace('D', 'E'))


def _hex_fn(n: float) -> str:
    return format(int(round(float(n))) & 0xFFFFFFFF if float(n) < 0 else int(round(float(n))), 'X')


def _oct_fn(n: float) -> str:
    return format(int(round(float(n))) & 0xFFFFFFFF if float(n) < 0 else int(round(float(n))), 'o')


def _bin_fn(n: float) -> str:
    return format(int(n), 'b')


def _len_fn(x: Any) -> float:
    if isinstance(x, str):
        return float(len(x))
    if isinstance(x, (list, tuple)):
        return float(len(x))
    return float(len(str(x)))


def _space(n: float) -> str:
    return ' ' * _count(n, 'SPACE$')


def _string_fn(n: float, c: Any) -> str:
    """STRING$(n, code) or STRING$(n, "x"): n copies of one character."""
    k = _count(n, 'STRING$')
    if isinstance(c, str):
        if not c:
            raise ValueError("ILLEGAL FUNCTION CALL: STRING$ needs a character")
        return c[0] * k
    return chr(int(round(float(c)))) * k


# Functions that return strings (names ending in $)
STRING_FUNCS: dict[str, Any] = {
    'LEFT$': _left, 'RIGHT$': _right, 'MID$': _mid,
    'CHR$': _chr_fn, 'STR$': _str_fn, 'HEX$': _hex_fn, 'OCT$': _oct_fn, 'BIN$': _bin_fn,
    'UCASE$': lambda s: s.upper(), 'LCASE$': lambda s: s.lower(),
    'LTRIM$': lambda s: s.lstrip(' '), 'RTRIM$': lambda s: s.rstrip(' '),
    'SPACE$': _space, 'STRING$': _string_fn,
}

# Functions that take/return mixed types
MIXED_FUNCS: dict[str, Any] = {
    'ASC': _asc, 'VAL': _val_fn, 'INSTR': _instr, 'LEN': _len_fn,
}


class StringMixin(HostState):
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

    def _eval_string_expr(self, expr: str, run_vars: dict[str, Any] | None = None) -> str | float:
        """Evaluate an expression that may return a string or a number.

        Uses the shared expression engine, so concatenation (``"a" + "b"``),
        string functions (``LEFT$(s$, 3)``) and string variables resolve to
        their values, and a string result keeps its type.
        """
        expr = expr.strip()
        if run_vars is not None:
            return self._safe_eval(expr, extra_ns=run_vars)
        return self._safe_eval(expr)

    def cmd_let_str(self, name: str, expr: str) -> None:
        """Assign a string value to a string variable."""
        val = self._store_var(name, self._eval_string_expr(expr))
        self.io.writeln(f"{name} = {val!r}" if isinstance(val, str) else f"{name} = {val}")
