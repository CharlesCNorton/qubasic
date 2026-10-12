"""QUBASIC number formatting: PRINT, STR$ and PRINT USING as QBASIC writes them.

PRINT and STR$ show a floating-point value the way QBASIC shows a SINGLE:
seven significant digits, no leading zero before the decimal point (.5,
-.25), whole values without a point, and E notation (1E+20, 2.5E-05) when the
exponent falls outside the range plain digits can show. Integers print in
full.

PRINT USING follows the QBASIC format language:

    #       digit position              .       decimal point
    ,       thousands separator (left of the point, also a digit position)
    +       sign, at the start or the end of the field
    -       trailing minus for negative values (end of the field)
    **      fill leading space with *        $$     floating dollar sign
    **$     both                             ^^^^   exponent (^^^^^ for 3 digits)
    !       first character of a string      &      the whole string
    \\  \\    the first n characters (n = 2 + the spaces between)
    _       print the next character literally

A number too wide for its field prints as % followed by the number. When
there are more values than fields the format starts over.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


def _is_int(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return hasattr(v, 'dtype') and getattr(v, 'ndim', 1) == 0 and v.dtype.kind in 'iu'


def number_text(v: Any) -> str:
    """The digits QBASIC prints for a number (sign included, no padding)."""
    if _is_int(v):
        return str(int(v))
    f = float(v)
    if f != f or f in (math.inf, -math.inf):
        return str(f)
    s = f"{f:.7g}"
    if 'e' in s:
        mant, exp = s.split('e')
        sign = exp[0]
        digits = exp[1:].lstrip('0').rjust(2, '0')
        s = f"{mant}E{sign}{digits}"
    if s.startswith('0.'):
        s = s[1:]
    elif s.startswith('-0.'):
        s = '-' + s[2:]
    return s


def print_text(v: Any) -> str:
    """A number as PRINT shows it: a leading space in place of a plus sign and
    a trailing space."""
    text = number_text(v)
    return ('' if text.startswith('-') else ' ') + text + ' '


def str_text(v: Any) -> str:
    """STR$: the digits with a leading space for a non-negative value."""
    text = number_text(v)
    return text if text.startswith('-') else ' ' + text


# ── PRINT USING ─────────────────────────────────────────────────────────

class _NumField:
    __slots__ = ('width', 'lead_plus', 'star', 'dollar', 'int_pos', 'comma',
                 'point', 'decimals', 'exp', 'trail')

    def __init__(self) -> None:
        self.width = 0
        self.lead_plus = False
        self.star = False
        self.dollar = False
        self.int_pos = 0
        self.comma = False
        self.point = False
        self.decimals = 0
        self.exp = 0
        self.trail: str | None = None


class _StrField:
    __slots__ = ('kind', 'width')

    def __init__(self, kind: str, width: int) -> None:
        self.kind = kind          # '!', '&' or '\\'
        self.width = width


def _parse_number(fmt: str, i: int) -> tuple[_NumField, int] | None:
    """A numeric field starting at fmt[i], or None if none starts there."""
    n = len(fmt)
    f = _NumField()
    start = i
    if fmt.startswith('+', i):
        f.lead_plus = True
        i += 1
    if fmt.startswith('**$', i):
        f.star = f.dollar = True
        f.int_pos += 2
        i += 3
    elif fmt.startswith('**', i):
        f.star = True
        f.int_pos += 2
        i += 2
    elif fmt.startswith('$$', i):
        f.dollar = True
        f.int_pos += 1
        i += 2
    body = i
    while i < n and fmt[i] in '#,':
        if fmt[i] == ',':
            f.comma = True
        f.int_pos += 1
        i += 1
    if i < n and fmt[i] == '.' and (i + 1 < n and fmt[i + 1] == '#' or i > body
                                    or f.star or f.dollar):
        f.point = True
        i += 1
        while i < n and fmt[i] == '#':
            f.decimals += 1
            i += 1
    if not (f.int_pos or f.decimals or f.star or f.dollar):
        return None
    if i == body and not f.point and not (f.star or f.dollar):
        return None
    if fmt.startswith('^^^^^', i):
        f.exp = 5
        i += 5
    elif fmt.startswith('^^^^', i):
        f.exp = 4
        i += 4
    if not f.lead_plus and i < n and fmt[i] in '+-':
        f.trail = fmt[i]
        i += 1
    f.width = i - start
    return f, i


def _tokens(fmt: str) -> list[Any]:
    """The format split into literal text and fields."""
    out: list[Any] = []
    lit: list[str] = []
    i, n = 0, len(fmt)

    def flush() -> None:
        if lit:
            out.append(''.join(lit))
            lit.clear()

    while i < n:
        ch = fmt[i]
        if ch == '_' and i + 1 < n:
            lit.append(fmt[i + 1])
            i += 2
            continue
        if ch in '!&':
            flush()
            out.append(_StrField(ch, 1 if ch == '!' else 0))
            i += 1
            continue
        if ch == '\\':
            j = i + 1
            while j < n and fmt[j] == ' ':
                j += 1
            if j < n and fmt[j] == '\\':
                flush()
                out.append(_StrField('\\', j - i + 1))
                i = j + 1
                continue
        if ch in '#.+*$':
            parsed = _parse_number(fmt, i)
            if parsed is not None:
                flush()
                out.append(parsed[0])
                i = parsed[1]
                continue
        lit.append(ch)
        i += 1
    flush()
    return out


def _round_text(x: float, decimals: int) -> str:
    q = Decimal(1).scaleb(-decimals)
    return format(Decimal(repr(abs(x))).quantize(q, rounding=ROUND_HALF_UP), 'f')


def _group(digits: str) -> str:
    out = []
    for k, ch in enumerate(reversed(digits)):
        if k and k % 3 == 0:
            out.append(',')
        out.append(ch)
    return ''.join(reversed(out))


def _render_number(f: _NumField, value: Any) -> str:
    if isinstance(value, str):
        raise ValueError("TYPE MISMATCH: PRINT USING number field given a string")
    x = float(value)
    neg = x < 0
    if f.exp:
        k = f.int_pos - (0 if (f.lead_plus or f.trail) else 1)
        k = max(k, 0)
        if x == 0:
            e = 0
        else:
            e = math.floor(math.log10(abs(x))) + (1 if k == 0 else 1 - k)
        mant = abs(x) / (10.0 ** e)
        text = _round_text(mant, f.decimals)
        if float(text) >= 10.0 ** k and x != 0:
            e += 1
            text = _round_text(abs(x) / (10.0 ** e), f.decimals)
        int_part, _, frac = text.partition('.')
        if k == 0:
            int_part = ''
        edigits = f.exp - 2
        etext = f"E{'-' if e < 0 else '+'}{str(abs(e)).rjust(edigits, '0')}"
        digits = int_part + ('.' + frac if f.point else '') + etext
    else:
        text = _round_text(x, f.decimals)
        int_part, _, frac = text.partition('.')
        if f.comma:
            int_part = _group(int_part)
        if int_part == '0' and f.int_pos == 0:
            int_part = ''
        digits = int_part + ('.' + frac if f.point else '')
    lead = ''
    trail = ''
    if f.trail == '+':
        trail = '-' if neg else '+'
    elif f.trail == '-':
        trail = '-' if neg else ' '
    elif f.lead_plus:
        lead = '-' if neg else '+'
    elif neg:
        lead = '-'
    core = lead + ('$' if f.dollar else '') + digits + trail
    if len(core) > f.width:
        return '%' + core
    return ('*' if f.star else ' ') * (f.width - len(core)) + core


def _render_string(f: _StrField, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("TYPE MISMATCH: PRINT USING string field given a number")
    if f.kind == '&':
        return value
    return value[:f.width].ljust(f.width)


def format_using(fmt: str, values: list[Any]) -> str:
    """PRINT USING output for `values` under format `fmt` (see the module
    docstring)."""
    toks = _tokens(fmt)
    fields = [t for t in toks if not isinstance(t, str)]
    if not fields:
        if values:
            raise ValueError("ILLEGAL FUNCTION CALL: PRINT USING format has no field")
        return fmt
    out: list[str] = []
    k = 0
    while True:
        for t in toks:
            if isinstance(t, str):
                out.append(t)
                continue
            if k >= len(values):
                return ''.join(out)
            v = values[k]
            k += 1
            out.append(_render_number(t, v) if isinstance(t, _NumField) else _render_string(t, v))
        if k >= len(values):
            return ''.join(out)
