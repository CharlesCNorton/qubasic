"""QUBASIC expressions and variables: the expression grammar and
evaluator, PRINT number formatting and PRINT USING, the string functions,
the scope stack, typed variables, arrays and records.
"""

from __future__ import annotations

import ast
import cmath
import copy
import math
import operator
import re
import struct
import time
from collections.abc import ItemsView, Iterator, KeysView, MutableMapping, ValuesView
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from typing import Any, Callable, NamedTuple

from qubasic_core.host import LanguageHost

# ---------------------------------------------------------------------------
# QUBASIC expressions: a QBASIC-precedence parser and a tree evaluator.
#
# An expression is tokenized and parsed once (cached by its text) into a small
# tree of tuples, then evaluated against a name resolver, so a loop body or a
# live shot re-evaluates without re-parsing.
#
# Precedence, lowest first::
#
#     IMP   EQV   XOR   OR   AND   NOT
#     comparisons  = == <> != >< < > <= >=   (chained: 0 <= x <= 10)
#     + -   MOD   \ //   * / %   unary - +   ^ (left to right) and ** (right to left)
#     calls name(args), indexing name[args], atoms
#
# Comparisons and NOT give -1 for true and 0 for false. AND, OR, XOR, EQV and
# IMP are bitwise on integers, so they also combine those truth values. MOD and
# \ round their operands to integers (ties to even, as QBASIC converts); MOD
# takes the sign of the dividend and \ truncates toward zero. `//`, `%` and
# `**` keep their Python meanings. LBOUND and UBOUND take an array name, and
# `name()` as a call argument passes the whole array. `a(i).x` reads a field of
# a record element (record variables are names: p.x).
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<pyint>0[xX][0-9A-Fa-f]+|0[bB][01]+|0[oO][0-7]+)
  | (?P<hex>&[Hh][0-9A-Fa-f]+|\$[0-9A-Fa-f]+(?![\w$]))
  | (?P<bin>&[Bb][01]+)
  | (?P<oct>&[Oo]?[0-7]+)
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?[!#]?)
  | (?P<field>\.[A-Za-z_]\w*[$%!#&]?)
  | (?P<str>"[^"]*"|'[^']*')
  | (?P<name>[A-Za-z_][\w.]*[$%!#&]?)
  | (?P<filenum>\#)
  | (?P<op>\*\*|//|<=|>=|<>|><|==|!=|[-+*/\\^%=<>(),\[\]])
""", re.VERBOSE)

_WORD_OPS = frozenset({'AND', 'OR', 'NOT', 'XOR', 'EQV', 'IMP', 'MOD'})
_BOUND_FUNCS = frozenset({'LBOUND', 'UBOUND'})
_CMP_OPS = frozenset({'=', '==', '<>', '!=', '><', '<', '>', '<=', '>='})


def _tokenize(text: str) -> list[tuple[str, Any]]:
    tokens: list[tuple[str, Any]] = []
    pos, n = 0, len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ValueError(f"SYNTAX ERROR: unexpected '{text[pos]}' at column {pos + 1}")
        pos = m.end()
        kind = m.lastgroup
        tok = m.group()
        if kind == 'ws':
            continue
        if kind == 'num':
            body = tok.rstrip('!#').replace('d', 'e').replace('D', 'E')
            value: Any = float(body) if any(c in body for c in '.eE') else int(body)
            tokens.append(('num', value))
        elif kind == 'pyint':
            tokens.append(('num', int(tok, 0)))
        elif kind == 'hex':
            tokens.append(('num', int(tok[2:] if tok[0] == '&' else tok[1:], 16)))
        elif kind == 'bin':
            tokens.append(('num', int(tok[2:], 2)))
        elif kind == 'oct':
            tokens.append(('num', int(tok[2:] if tok[1] in 'Oo' else tok[1:], 8)))
        elif kind == 'str':
            tokens.append(('str', tok[1:-1]))
        elif kind == 'field':
            tokens.append(('field', tok[1:]))
        elif kind == 'filenum':
            # INPUT$(n, #f): the # marks a file number in an argument list.
            if not tokens or tokens[-1] not in (('op', '('), ('op', ',')):
                raise ValueError(f"SYNTAX ERROR: unexpected '#' at column {pos}")
        elif kind == 'name':
            up = tok.upper()
            tokens.append(('op', up) if up in _WORD_OPS else ('name', tok))
        else:
            tokens.append(('op', tok))
    return tokens


class _Parser:
    """Recursive-descent parser over the token list (one level per precedence)."""

    def __init__(self, tokens: list[tuple[str, Any]], text: str) -> None:
        self.toks = tokens
        self.i = 0
        self.text = text

    def peek(self) -> tuple[str, Any] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take_op(self, *ops: str) -> str | None:
        t = self.peek()
        if t is not None and t[0] == 'op' and t[1] in ops:
            self.i += 1
            return t[1]
        return None

    def expect(self, op: str) -> None:
        if self.take_op(op) is None:
            got = self.peek()
            raise ValueError(f"SYNTAX ERROR: expected '{op}'"
                             + (f" before '{got[1]}'" if got else " at end"))

    def parse(self) -> tuple:
        node = self.imp()
        extra = self.peek()
        if extra is not None:
            raise ValueError(f"SYNTAX ERROR: unexpected '{extra[1]}' in {self.text!r}")
        return node

    def _left_assoc(self, sub: Callable[[], tuple], ops: tuple[str, ...]) -> tuple:
        node = sub()
        while True:
            op = self.take_op(*ops)
            if op is None:
                return node
            node = ('bin', op, node, sub())

    def imp(self) -> tuple:
        return self._left_assoc(self.eqv, ('IMP',))

    def eqv(self) -> tuple:
        return self._left_assoc(self.xor, ('EQV',))

    def xor(self) -> tuple:
        return self._left_assoc(self.or_, ('XOR',))

    def or_(self) -> tuple:
        return self._left_assoc(self.and_, ('OR',))

    def and_(self) -> tuple:
        return self._left_assoc(self.not_, ('AND',))

    def not_(self) -> tuple:
        if self.take_op('NOT'):
            return ('not', self.not_())
        return self.cmp()

    def cmp(self) -> tuple:
        first = self.add()
        ops, rest = [], []
        while True:
            t = self.peek()
            if t is None or t[0] != 'op' or t[1] not in _CMP_OPS:
                break
            self.i += 1
            ops.append(t[1])
            rest.append(self.add())
        if not ops:
            return first
        return ('cmp', tuple(ops), (first, *rest))

    def add(self) -> tuple:
        return self._left_assoc(self.mod, ('+', '-'))

    def mod(self) -> tuple:
        return self._left_assoc(self.idiv, ('MOD',))

    def idiv(self) -> tuple:
        return self._left_assoc(self.mul, ('\\', '//'))

    def mul(self) -> tuple:
        return self._left_assoc(self.unary, ('*', '/', '%'))

    def unary(self) -> tuple:
        op = self.take_op('-', '+')
        if op is not None:
            return ('neg', self.unary()) if op == '-' else self.unary()
        return self.power()

    def power(self) -> tuple:
        node = self.postfix()
        while True:
            op = self.take_op('^', '**')
            if op is None:
                return node
            if op == '**':                 # Python: right to left
                return ('bin', '**', node, self.unary())
            if self.take_op('-'):          # 2^-1
                exponent = ('neg', self.postfix())
            else:
                self.take_op('+')
                exponent = self.postfix()
            node = ('bin', '^', node, exponent)

    def postfix(self) -> tuple:
        t = self.peek()
        if t is None:
            raise ValueError("SYNTAX ERROR: expression ends early")
        if t[0] == 'name':
            self.i += 1
            name = t[1]
            nxt = self.peek()
            # FN name(...) calls a DEF FN function
            if name.upper() == 'FN' and nxt is not None and nxt[0] == 'name':
                self.i += 1
                name = 'FN' + nxt[1]
                nxt = self.peek()
            if nxt == ('op', '('):
                self.i += 1
                return self.fields(('call', name, self.args(')')))
            if nxt == ('op', '['):
                self.i += 1
                return self.fields(('index', name, self.args(']')))
            return ('name', name)
        return self.atom()

    def fields(self, node: tuple) -> tuple:
        """Field accesses after an element: a(i).x, a(i).pos.x."""
        while True:
            t = self.peek()
            if t is None or t[0] != 'field':
                return node
            self.i += 1
            node = ('field', node, t[1])

    def args(self, close: str) -> tuple:
        if self.take_op(close):
            return ()
        items = [self.imp()]
        while self.take_op(','):
            items.append(self.imp())
        self.expect(close)
        return tuple(items)

    def atom(self) -> tuple:
        t = self.peek()
        assert t is not None
        self.i += 1
        if t[0] == 'num':
            return ('num', t[1])
        if t[0] == 'str':
            return ('str', t[1])
        if t == ('op', '('):
            node = self.imp()
            self.expect(')')
            return node
        raise ValueError(f"SYNTAX ERROR: unexpected '{t[1]}' in {self.text!r}")


@lru_cache(maxsize=16384)
def parse(text: str) -> tuple:
    """The expression tree for `text` (cached); raises ValueError on bad syntax."""
    text = text.strip()
    if not text:
        raise ValueError("EMPTY EXPRESSION")
    return _Parser(_tokenize(text), text).parse()


# ── Operators ───────────────────────────────────────────────────────────

def round_half_even(x: Any) -> int:
    """A number converted to an integer as QBASIC converts one (CINT, integer
    variables, MOD, \\, the bitwise operators): to the nearest integer, ties
    to the even one, so 2.5 is 2 and 3.5 is 4."""
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    f = float(x)
    if f != f or f in (math.inf, -math.inf):
        raise ValueError("OVERFLOW")
    return int(round(f))


def _as_int(x: Any) -> int:
    """A BASIC value as an integer for the bitwise operators (non-numbers are
    0 or 1 by truth)."""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    try:
        return round_half_even(x)
    except (TypeError, ValueError):
        return 1 if x else 0


def _round_int(x: Any) -> int:
    return round_half_even(x)


def _numeric(op: str, a: Any, b: Any) -> None:
    if isinstance(a, str) or isinstance(b, str):
        raise ValueError(f"TYPE MISMATCH: '{op}' needs numbers")


def _add(a: Any, b: Any) -> Any:
    if isinstance(a, str) != isinstance(b, str):
        raise ValueError("TYPE MISMATCH: cannot add a string and a number")
    return a + b


def _sub(a: Any, b: Any) -> Any:
    _numeric('-', a, b)
    return a - b


def _mul(a: Any, b: Any) -> Any:
    _numeric('*', a, b)
    return a * b


def _div(a: Any, b: Any) -> Any:
    _numeric('/', a, b)
    try:
        return a / b
    except ZeroDivisionError:
        raise ValueError("DIVISION BY ZERO") from None


def _idiv(a: Any, b: Any) -> int:
    _numeric('\\', a, b)
    ia, ib = _round_int(a), _round_int(b)
    if ib == 0:
        raise ValueError("DIVISION BY ZERO")
    q = abs(ia) // abs(ib)
    return q if (ia >= 0) == (ib > 0) else -q


def _floordiv(a: Any, b: Any) -> Any:
    _numeric('//', a, b)
    try:
        return a // b
    except ZeroDivisionError:
        raise ValueError("DIVISION BY ZERO") from None


def _pymod(a: Any, b: Any) -> Any:
    _numeric('%', a, b)
    try:
        return a % b
    except ZeroDivisionError:
        raise ValueError("DIVISION BY ZERO") from None


def _mod(a: Any, b: Any) -> int:
    _numeric('MOD', a, b)
    ia, ib = _round_int(a), _round_int(b)
    if ib == 0:
        raise ValueError("DIVISION BY ZERO")
    return int(math.fmod(ia, ib))


def _pow(a: Any, b: Any) -> Any:
    _numeric('^', a, b)
    try:
        r = a ** b
    except ZeroDivisionError:
        raise ValueError("DIVISION BY ZERO") from None
    except OverflowError:
        raise ValueError("OVERFLOW") from None
    if isinstance(r, complex):
        raise ValueError("ILLEGAL FUNCTION CALL: negative number to a fractional power")
    return r


_BINARY: dict[str, Callable[[Any, Any], Any]] = {
    '+': _add, '-': _sub, '*': _mul, '/': _div, '\\': _idiv, '//': _floordiv,
    '%': _pymod, 'MOD': _mod, '^': _pow, '**': _pow,
    'AND': lambda a, b: _as_int(a) & _as_int(b),
    'OR': lambda a, b: _as_int(a) | _as_int(b),
    'XOR': lambda a, b: _as_int(a) ^ _as_int(b),
    'EQV': lambda a, b: ~(_as_int(a) ^ _as_int(b)),
    'IMP': lambda a, b: (~_as_int(a)) | _as_int(b),
}


def _compare(op: str, a: Any, b: Any) -> bool:
    if isinstance(a, str) != isinstance(b, str):
        raise ValueError("TYPE MISMATCH: cannot compare a string and a number")
    if op in ('=', '=='):
        return a == b
    if op in ('<>', '!=', '><'):
        return a != b
    if op == '<':
        return a < b
    if op == '>':
        return a > b
    if op == '<=':
        return a <= b
    return a >= b


def evaluate(node: tuple, names: Any) -> Any:
    """Evaluate a parsed expression. `names` resolves identifiers:
    name(n), call(n, args), index(n, args) and bound(LBOUND|UBOUND, n, args),
    and may offer length(node), the size LEN gives a variable of fixed size."""
    kind = node[0]
    if kind == 'num' or kind == 'str':
        return node[1]
    if kind == 'name':
        return names.name(node[1])
    if kind == 'bin':
        return _BINARY[node[1]](evaluate(node[2], names), evaluate(node[3], names))
    if kind == 'call':
        args = node[2]
        if node[1].upper() in _BOUND_FUNCS and args and args[0][0] == 'name':
            return names.bound(node[1].upper(), args[0][1],
                               [evaluate(a, names) for a in args[1:]])
        if (node[1].upper() == 'LEN' and len(args) == 1 and args[0][0] in ('name', 'call', 'field')
                and hasattr(names, 'length')):
            size = names.length(args[0])
            if size is not None:
                return size
        return names.call(node[1], [_argument(a, names) for a in args])
    if kind == 'field':
        rec = evaluate(node[1], names)
        if not isinstance(rec, dict):
            raise ValueError(f"TYPE MISMATCH: .{node[2]} needs a record")
        key = node[2].lower()
        if key not in rec:
            raise ValueError(f"ELEMENT NOT DEFINED: .{node[2]}")
        return rec[key]
    if kind == 'neg':
        v = evaluate(node[1], names)
        if isinstance(v, str):
            raise ValueError("TYPE MISMATCH: cannot negate a string")
        return -v
    if kind == 'cmp':
        ops, operands = node[1], node[2]
        left = evaluate(operands[0], names)
        for op, operand in zip(ops, operands[1:]):
            right = evaluate(operand, names)
            if not _compare(op, left, right):
                return 0
            left = right
        return -1
    if kind == 'not':
        return 0 if evaluate(node[1], names) else -1
    if kind == 'index':
        return names.index(node[1], [evaluate(a, names) for a in node[2]])
    raise ValueError(f"UNSUPPORTED EXPRESSION: {kind}")


def _argument(node: tuple, names: Any) -> Any:
    """A call argument: `a()` names a whole array (passed by reference to a
    procedure) when the resolver knows the array; anything else is a value."""
    if node[0] == 'call' and not node[2]:
        ref = getattr(names, 'array_ref', None)
        if ref is not None:
            whole = ref(node[1])
            if whole is not None:
                return whole
    return evaluate(node, names)


def names_in(node: tuple) -> set[str]:
    """Every identifier an expression reads (variables and called names)."""
    found: set[str] = set()

    def walk(n: tuple) -> None:
        kind = n[0]
        if kind == 'name':
            found.add(n[1])
        elif kind in ('call', 'index'):
            found.add(n[1])
            for a in n[2]:
                walk(a)
        elif kind == 'bin':
            walk(n[2])
            walk(n[3])
        elif kind in ('neg', 'not', 'field'):
            walk(n[1])
        elif kind == 'cmp':
            for a in n[2]:
                walk(a)
    walk(node)
    return found


# ---------------------------------------------------------------------------
# QUBASIC number formatting: PRINT, STR$ and PRINT USING as QBASIC writes them.
#
# PRINT and STR$ show a floating-point value the way QBASIC shows a SINGLE:
# seven significant digits, no leading zero before the decimal point (.5,
# -.25), whole values without a point, and E notation (1E+20, 2.5E-05) when the
# exponent falls outside the range plain digits can show. Integers print in
# full.
#
# PRINT USING follows the QBASIC format language:
#
#     #       digit position              .       decimal point
#     ,       thousands separator (left of the point, also a digit position)
#     +       sign, at the start or the end of the field
#     -       trailing minus for negative values (end of the field)
#     **      fill leading space with *        $$     floating dollar sign
#     **$     both                             ^^^^   exponent (^^^^^ for 3 digits)
#     !       first character of a string      &      the whole string
#     \  \    the first n characters (n = 2 + the spaces between)
#     _       print the next character literally
#
# A number too wide for its field prints as % followed by the number. When
# there are more values than fields the format starts over.
# ---------------------------------------------------------------------------

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


def next_column(col: int, text: str) -> int:
    """The cursor column after writing `text` from column `col`."""
    nl = text.rfind('\n')
    return len(text) - nl - 1 if nl >= 0 else col + len(text)


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


# ---------------------------------------------------------------------------
# QUBASIC string support — string variables, functions, and operations.
# ---------------------------------------------------------------------------

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


def _mk(fmt: str, name: str) -> Any:
    """MKI$/MKL$/MKS$/MKD$: a number as the bytes QBASIC stores (one character each)."""
    def mk(x: float) -> str:
        v = int(round(float(x))) if fmt in ('<h', '<i') else float(x)
        try:
            return struct.pack(fmt, v).decode('latin-1')
        except struct.error:
            raise ValueError(f"OVERFLOW: {name}({x})") from None
    return mk


def _cv(fmt: str, name: str) -> Any:
    """CVI/CVL/CVS/CVD: the number a MKx$ string holds."""
    size = struct.calcsize(fmt)

    def cv(s: str) -> float:
        if not isinstance(s, str) or len(s) < size:
            raise ValueError(f"ILLEGAL FUNCTION CALL: {name} needs {size} characters")
        return float(struct.unpack(fmt, s[:size].encode('latin-1', errors='replace'))[0])
    return cv


def _mksmbf(x: float) -> str:
    """MKSMBF$: a number as Microsoft Binary Format single precision."""
    if float(x) == 0:
        return '\x00' * 4
    b = struct.pack('<f', float(x))
    exp = (((b[3] & 0x7F) << 1) | (b[2] >> 7)) + 2
    if exp > 255:
        raise ValueError(f"OVERFLOW: MKSMBF$({x})")
    return bytes([b[0], b[1], (b[2] & 0x7F) | (b[3] & 0x80), exp]).decode('latin-1')


def _cvsmbf(s: str) -> float:
    """CVSMBF: the number a Microsoft Binary Format single holds."""
    if not isinstance(s, str) or len(s) < 4:
        raise ValueError("ILLEGAL FUNCTION CALL: CVSMBF needs 4 characters")
    m = s[:4].encode('latin-1', errors='replace')
    if m[3] == 0:
        return 0.0
    exp = m[3] - 2
    return float(struct.unpack('<f', bytes([m[0], m[1], ((exp << 7) & 0x80) | (m[2] & 0x7F),
                                            (m[2] & 0x80) | (exp >> 1)]))[0])


def _mkdmbf(x: float) -> str:
    """MKDMBF$: a number as Microsoft Binary Format double precision."""
    if float(x) == 0:
        return '\x00' * 8
    bits = struct.unpack('<Q', struct.pack('<d', float(x)))[0]
    exp = ((bits >> 52) & 0x7FF) - 894
    if not 0 < exp < 256:
        raise ValueError(f"OVERFLOW: MKDMBF$({x})")
    body = ((bits & ((1 << 52) - 1)) << 3) | ((bits >> 63) << 55)
    return (body.to_bytes(7, 'little') + bytes([exp])).decode('latin-1')


def _cvdmbf(s: str) -> float:
    """CVDMBF: the number a Microsoft Binary Format double holds."""
    if not isinstance(s, str) or len(s) < 8:
        raise ValueError("ILLEGAL FUNCTION CALL: CVDMBF needs 8 characters")
    raw = s[:8].encode('latin-1', errors='replace')
    if raw[7] == 0:
        return 0.0
    body = int.from_bytes(raw[:7], 'little')
    bits = (((body >> 55) & 1) << 63) | ((raw[7] + 894) << 52) | ((body & ((1 << 55) - 1)) >> 3)
    return float(struct.unpack('<d', struct.pack('<Q', bits))[0])


def _no_address(name: str) -> Any:
    def fn(*_: Any) -> float:
        raise ValueError(f"ILLEGAL FUNCTION CALL: {name} (QUBASIC variables have no memory address)")
    return fn


# Functions that return strings (names ending in $)
STRING_FUNCS: dict[str, Any] = {
    'LEFT$': _left, 'RIGHT$': _right, 'MID$': _mid,
    'CHR$': _chr_fn, 'STR$': _str_fn, 'HEX$': _hex_fn, 'OCT$': _oct_fn, 'BIN$': _bin_fn,
    'UCASE$': lambda s: s.upper(), 'LCASE$': lambda s: s.lower(),
    'LTRIM$': lambda s: s.lstrip(' '), 'RTRIM$': lambda s: s.rstrip(' '),
    'SPACE$': _space, 'STRING$': _string_fn,
    'MKI$': _mk('<h', 'MKI$'), 'MKL$': _mk('<i', 'MKL$'), 'MKS$': _mk('<f', 'MKS$'),
    'MKD$': _mk('<d', 'MKD$'), 'MKSMBF$': _mksmbf, 'MKDMBF$': _mkdmbf,
}

# Functions that take/return mixed types
MIXED_FUNCS: dict[str, Any] = {
    'ASC': _asc, 'VAL': _val_fn, 'INSTR': _instr, 'LEN': _len_fn,
    'CVI': _cv('<h', 'CVI'), 'CVL': _cv('<i', 'CVL'), 'CVS': _cv('<f', 'CVS'),
    'CVD': _cv('<d', 'CVD'), 'CVSMBF': _cvsmbf, 'CVDMBF': _cvdmbf,
    'VARPTR': _no_address('VARPTR'), 'VARSEG': _no_address('VARSEG'), 'SADD': _no_address('SADD'),
    'VARPTR$': _no_address('VARPTR$'),
    # No joystick, light pen or device-control channel is attached.
    'STICK': lambda n=0: 0, 'STRIG': lambda n=0: 0, 'PEN': lambda n=0: 0, 'IOCTL$': lambda n=0: '',
}


class StringMixin(LanguageHost):
    """String variable and function support for QBasicTerminal."""

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

    def _eval_string_expr(self, expr: str, run_vars: MutableMapping[str, Any] | None = None) -> str | float:
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


# ---------------------------------------------------------------------------
# QUBASIC variable scope — the run's view of the variable table.
# ---------------------------------------------------------------------------

class Scope(MutableMapping[str, Any]):
    """The variables a running program reads and writes.

    A view over the terminal's variable table: reads and writes go straight
    to it, so a SUB or FUNCTION that swaps the table for its own locals (see
    SubroutineMixin) is seen by every statement that holds this view. Deleting a name
    that is not there does nothing.
    """

    def __init__(self, persistent: dict[str, Any]) -> None:
        self._persistent = persistent

    def __getitem__(self, name: str) -> Any:
        return self._persistent[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._persistent[name] = value

    def __delitem__(self, name: str) -> None:
        self._persistent.pop(name, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._persistent)

    def __len__(self) -> int:
        return len(self._persistent)

    def __contains__(self, name: object) -> bool:
        return name in self._persistent

    def get(self, name: str, default: Any = None) -> Any:
        return self._persistent.get(name, default)

    def keys(self) -> KeysView[str]:
        return self._persistent.keys()

    def items(self) -> ItemsView[str, Any]:
        return self._persistent.items()

    def values(self) -> ValuesView[Any]:
        return self._persistent.values()

    def as_dict(self) -> dict[str, Any]:
        """A copy of the variables, for expression evaluation."""
        return dict(self._persistent)


# ---------------------------------------------------------------------------
# QUBASIC variables and arrays: types, records, declarations and storage.
#
# Types follow QBASIC. A name's suffix fixes its type (% INTEGER, & LONG,
# ! SINGLE, # DOUBLE, $ STRING); otherwise `DIM name AS <type>` or a
# DEFINT/DEFLNG/DEFSNG/DEFDBL/DEFSTR letter range does; otherwise the name is
# SINGLE. INTEGER and LONG values round on assignment (ties to even) and
# overflow outside 16 and 32 bits; SINGLE and DOUBLE hold the evaluated number;
# a STRING holds text, and a `STRING * n` holds exactly n characters (padded
# with spaces, or cut).
#
# A TYPE block declares a record type; its fields take any of those types or
# another record type:
#
#     TYPE Point
#         x AS SINGLE
#         label AS STRING * 8
#     END TYPE
#     DIM p AS Point, path(1 TO 10) AS Point
#     p.x = 1.5: path(3).label = "start"
#
# A record variable holds a Record (a dict of field values that knows its
# type); `p = q` copies a whole record of the same type.
#
# Arrays are declared by `DIM name(bounds) [AS type]`, each bound an
# expression `hi` (lower bound from OPTION BASE) or `lo TO hi`, several
# declarations to a DIM. Subscripts round to integers and are checked against
# the declared bounds. An array used without DIM is created on first use: one
# dimension grows as it is written, several dimensions get 0 (or OPTION BASE)
# to 10 each, as QBASIC dimensions an undeclared array. LBOUND and UBOUND
# report the bounds. CONST names a value that cannot be assigned again.
# ---------------------------------------------------------------------------

SUFFIX_KIND = {'%': 'INTEGER', '&': 'LONG', '!': 'SINGLE', '#': 'DOUBLE', '$': 'STRING'}
TYPE_NAMES = frozenset({'INTEGER', 'LONG', 'SINGLE', 'DOUBLE', 'STRING'})
# Field types the TYPE command has always taken, as the QBASIC types they mean.
_TYPE_ALIASES = {'FLOAT': 'DOUBLE', 'QUBIT': 'INTEGER'}
_INT_RANGE = {'INTEGER': (-32768, 32767), 'LONG': (-2 ** 31, 2 ** 31 - 1)}
_DEF_KIND = {'DEFINT': 'INTEGER', 'DEFLNG': 'LONG', 'DEFSNG': 'SINGLE',
             'DEFDBL': 'DOUBLE', 'DEFSTR': 'STRING'}
IMPLICIT_UPPER = 10            # an undeclared multi-dimensional array spans base..10
MAX_RECORD_DEPTH = 16

_RE_DEFTYPE = re.compile(r'(DEFINT|DEFLNG|DEFSNG|DEFDBL|DEFSTR)\s+(.+)$', re.IGNORECASE)
_RE_DECL = re.compile(r'([A-Za-z_]\w*[$%!#&]?)\s*(?:\((.*)\))?\s*'
                      r'(?:AS\s+([A-Za-z_]\w*)(?:\s*\*\s*(\d+))?)?\s*$', re.IGNORECASE)
_RE_LVALUE = re.compile(r'\s*([A-Za-z_][\w.]*[$%!#&]?)\s*'
                        r'(?:\((.*)\)((?:\s*\.\s*[A-Za-z_]\w*[$%!#&]?)*))?\s*$')
_RE_BOUND = re.compile(r'(.+?)\s+TO\s+(.+)$', re.IGNORECASE)


class Record(dict):
    """A record value: field name (lower case) -> value, with its TYPE name."""

    def __init__(self, type_name: str, fields: dict | None = None) -> None:
        super().__init__(fields or {})
        self.type_name = type_name

    def __deepcopy__(self, memo: dict) -> Record:
        return Record(self.type_name, {k: copy.deepcopy(v, memo) for k, v in self.items()})


class LValue(NamedTuple):
    """An assignable reference: a variable (name, record paths included), an
    array element (subs), or a field of a record element (field)."""
    name: str
    subs: list | None = None
    field: str | None = None


class ArrayRef(NamedTuple):
    """A whole array passed to a procedure (`Sort values()`), by reference."""
    name: str


def type_text(name: str, length: str | int | None = None) -> str:
    """A declared type in canonical form: 'INTEGER', 'STRING*8', 'POINT'."""
    word = _TYPE_ALIASES.get(name.upper(), name.upper())
    return f"STRING*{int(length)}" if length and word == 'STRING' else word


def fixed_length(kind: str) -> int:
    """n for a 'STRING*n' kind, else 0."""
    return int(kind.split('*', 1)[1]) if kind.startswith('STRING*') else 0


def is_string_kind(kind: str) -> bool:
    return kind == 'STRING' or kind.startswith('STRING*')


def letter_ranges(spec: str) -> list[str]:
    """Letters named by a DEFtype list such as 'A-C, X, I-N'."""
    letters: list[str] = []
    for part in spec.replace(' ', '').split(','):
        if not part:
            continue
        m = re.fullmatch(r'([A-Za-z])(?:-([A-Za-z]))?', part)
        if not m:
            raise ValueError(f"SYNTAX ERROR: bad letter range '{part}'")
        lo = m.group(1).upper()
        hi = (m.group(2) or m.group(1)).upper()
        if hi < lo:
            lo, hi = hi, lo
        letters.extend(chr(c) for c in range(ord(lo), ord(hi) + 1))
    return letters


@dataclass
class SymbolState:
    """How the program's names are declared: variable types (DIM AS) and
    DEFtype letter ranges, CONSTs, SHARED globals, array bounds and element
    types, OPTION BASE, TYPE definitions and DEF FN functions."""

    var_types: dict[str, str] = field(default_factory=dict)                # name -> declared type (DIM x AS ...)
    deftypes: dict[str, str] = field(default_factory=dict)                 # first letter -> type (DEFINT and the rest)
    consts: set[str] = field(default_factory=set)
    shared: set[str] = field(default_factory=set)                          # DIM SHARED and COMMON SHARED names
    array_dims: dict[str, list[int]] = field(default_factory=dict)         # upper bounds per dimension
    array_lb: dict[str, list[int]] = field(default_factory=dict)           # lower bounds per dimension
    array_types: dict[str, str] = field(default_factory=dict)              # element type
    dimmed: set[str] = field(default_factory=set)                          # DIM'd arrays enforce their bounds on write
    option_base: int = 0
    types: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # TYPE name -> (field, type)
    pending_type: dict | None = None                                       # a TYPE block being entered
    fns: dict[str, dict[str, Any]] = field(default_factory=dict)           # DEF FN functions


class VariableMixin(LanguageHost):
    """Typed variables, records, CONST, DIM/REDIM/ERASE and array storage."""

    def _reset_declarations(self, prog: Any = None) -> None:
        """Forget the last run's declarations, then apply the program's
        DEFtype statements (they hold for the whole program, wherever they
        sit, as in QBASIC) and its TYPE blocks."""
        self.symbols.var_types.clear()
        self.symbols.consts.clear()
        self.symbols.shared.clear()
        self.symbols.deftypes.clear()
        if prog is None:
            return
        for text in prog.texts:
            m = _RE_DEFTYPE.match(text.strip())
            if m:
                kind = _DEF_KIND[m.group(1).upper()]
                for letter in letter_ranges(m.group(2)):
                    self.symbols.deftypes[letter] = kind
        for name, fields in getattr(prog, 'types', {}).items():
            self.symbols.types[name] = list(fields)

    # ── Types ────────────────────────────────────────────────────────

    def _kind(self, name: str) -> str:
        """The type of a variable name: INTEGER, LONG, SINGLE, DOUBLE, STRING,
        STRING*n or a TYPE name (a record field's own type for p.x)."""
        if name[-1:] in SUFFIX_KIND:
            return SUFFIX_KIND[name[-1]]
        if '.' in name:
            base, path = name.split('.', 1)
            rec = self.variables.get(base)
            if isinstance(rec, Record):
                return self._path_kind(rec.type_name, path)
        declared = self.symbols.var_types.get(name)
        if declared:
            return declared
        return self.symbols.deftypes.get(name[:1].upper(), 'SINGLE')

    def _type_fields(self, type_name: str) -> list[tuple[str, str]]:
        fields = self.symbols.types.get(type_name.upper())
        if fields is None:
            raise ValueError(f"UNDEFINED TYPE: {type_name}")
        return fields

    def _field_kind(self, type_name: str, field: str) -> str:
        for fname, ftype in self._type_fields(type_name):
            if fname.lower() == field.lower():
                return type_text(*ftype.split('*', 1)) if '*' in ftype else type_text(ftype)
        raise ValueError(f"ELEMENT NOT DEFINED: {type_name}.{field}")

    def _path_kind(self, type_name: str, path: str) -> str:
        """The type a field path (x, or pos.x) reaches from a record type."""
        kind = type_name
        for part in path.split('.'):
            kind = self._field_kind(kind, part)
        return kind

    def _new_record(self, type_name: str, depth: int = 0) -> Record:
        if depth > MAX_RECORD_DEPTH:
            raise ValueError(f"TYPE {type_name} contains itself")
        rec = Record(type_name.upper())
        for fname, _ in self._type_fields(type_name):
            rec[fname.lower()] = self._default_of_kind(self._field_kind(type_name, fname),
                                                       depth + 1)
        return rec

    def _default_of_kind(self, kind: str, depth: int = 0) -> Any:
        if kind in self.symbols.types:
            return self._new_record(kind, depth)
        if is_string_kind(kind):
            return ' ' * fixed_length(kind)
        return 0 if kind in _INT_RANGE else 0.0

    def _default_value(self, name: str, kind: str | None = None) -> Any:
        return self._default_of_kind(kind or self._kind(name))

    def _coerce(self, name: str, value: Any, kind: str | None = None) -> Any:
        """`value` converted for storage in variable `name` (or an element of
        array `name`), raising TYPE MISMATCH or OVERFLOW as QBASIC does."""
        kind = kind or self._kind(name)
        if kind in self.symbols.types:
            if not (isinstance(value, Record) and value.type_name == kind):
                raise ValueError(f"TYPE MISMATCH: '{name}' holds a {kind} record")
            return copy.deepcopy(value)
        if isinstance(value, Record):
            raise ValueError(f"TYPE MISMATCH: '{name}' is not a {value.type_name} record")
        if is_string_kind(kind):
            if not isinstance(value, str):
                raise ValueError(f"TYPE MISMATCH: '{name}' holds a string")
            n = fixed_length(kind)
            return value[:n].ljust(n) if n else value
        if isinstance(value, str):
            raise ValueError(f"TYPE MISMATCH: '{name}' is numeric; use '{name.rstrip('%&!#')}$' "
                             f"for strings")
        if kind in _INT_RANGE:
            v = round_half_even(value)
            lo, hi = _INT_RANGE[kind]
            if not lo <= v <= hi:
                raise ValueError(f"OVERFLOW: {v} does not fit {kind} '{name}'")
            return v
        try:
            return float(value)
        except TypeError:
            try:
                from qiskit.circuit import ParameterExpression
            except ImportError:
                raise
            if isinstance(value, ParameterExpression):
                return value
            raise

    # ── Records ──────────────────────────────────────────────────────

    @staticmethod
    def _record_get(rec: Any, path: str) -> Any:
        """The value at field path `path` (x, or pos.x) of a record."""
        for part in path.split('.'):
            if not isinstance(rec, Record):
                raise ValueError(f"TYPE MISMATCH: .{part} needs a record")
            key = part.lower()
            if key not in rec:
                raise ValueError(f"ELEMENT NOT DEFINED: {rec.type_name}.{part}")
            rec = rec[key]
        return rec

    def _record_set(self, rec: Record, path: str, value: Any) -> Any:
        """Assign field path `path` of a record (typed); returns the stored value."""
        parts = path.split('.')
        for part in parts[:-1]:
            rec = self._record_get(rec, part)
        if not isinstance(rec, Record):
            raise ValueError(f"TYPE MISMATCH: .{parts[-1]} needs a record")
        kind = self._field_kind(rec.type_name, parts[-1])
        v = self._coerce(f"{rec.type_name}.{parts[-1]}", value, kind)
        rec[parts[-1].lower()] = v
        return v

    def _record_of(self, name: str, run_vars: Any = None) -> tuple[Record, str] | None:
        """(record, field path) when `name` is a field of a record variable."""
        if '.' not in name:
            return None
        base, path = name.split('.', 1)
        source = run_vars if run_vars is not None and base in run_vars else self.variables
        rec = source.get(base)
        return (rec, path) if isinstance(rec, Record) else None

    def _store_var(self, name: str, value: Any, run_vars: Any = None) -> Any:
        """Assign a scalar variable or a record field with type conversion;
        returns the stored value."""
        self._assert_assignable(name)
        field = self._record_of(name, run_vars)
        if field is not None:
            return self._record_set(field[0], field[1], value)
        v = self._coerce(name, value)
        if run_vars is not None and run_vars is not self.variables:
            run_vars[name] = v
        self.variables[name] = v
        return v

    # ── CONST and DEFtype ────────────────────────────────────────────

    def _stmt_const(self, stmt: str, run_vars: Any) -> bool:
        """CONST name = value [, ...]. A CONST may take the name of a built-in
        constant (CONST PI = 3.14159, CONST TRUE = -1, as QBASIC programs
        declare them) and then stands for it; assigning to one stays an
        error."""
        m = re.match(r'CONST\s+(.+)$', stmt, re.IGNORECASE)
        if not m:
            return False
        for decl in self._split_arg_list(m.group(1)):
            dm = re.match(r'([A-Za-z_]\w*[$%!#&]?)\s*=\s*(.+)$', decl)
            if not dm:
                raise ValueError(f"SYNTAX ERROR: CONST {decl}")
            name = dm.group(1)
            value = self._safe_eval(dm.group(2), extra_ns=run_vars)
            if name[-1:] in SUFFIX_KIND:
                value = self._coerce(name, value)
            self.symbols.consts.add(name)
            if run_vars is not None and run_vars is not self.variables:
                run_vars[name] = value
            self.variables[name] = value
        return True

    @staticmethod
    def _stmt_deftype(stmt: str) -> bool:
        """DEFINT/DEFLNG/DEFSNG/DEFDBL/DEFSTR take effect for the whole program
        when it starts (see _reset_declarations); the statement itself only
        has to parse."""
        m = _RE_DEFTYPE.match(stmt.strip())
        if not m:
            return False
        letter_ranges(m.group(2))
        return True

    # ── DIM / REDIM / ERASE ──────────────────────────────────────────

    def _bound_pair(self, spec: str, run_vars: Any) -> tuple[int, int]:
        m = _RE_BOUND.match(spec.strip())
        if m:
            lo = round_half_even(self._safe_eval(m.group(1), extra_ns=run_vars))
            hi = round_half_even(self._safe_eval(m.group(2), extra_ns=run_vars))
        else:
            lo = self.symbols.option_base
            hi = round_half_even(self._safe_eval(spec, extra_ns=run_vars))
        if hi < lo:
            raise ValueError(f"SUBSCRIPT OUT OF RANGE: bounds {lo} TO {hi}")
        return lo, hi

    def _check_type(self, typ: str) -> None:
        if typ not in TYPE_NAMES and typ not in self.symbols.types and not typ.startswith('STRING*'):
            raise ValueError(f"UNDEFINED TYPE: {typ}")

    def _make_array(self, name: str, bounds: str, typ: str | None, run_vars: Any,
                    preserve: bool = False) -> None:
        pairs = [self._bound_pair(b, run_vars) for b in self._split_arg_list(bounds)]
        if not pairs:
            raise ValueError(f"SYNTAX ERROR: DIM {name}() needs bounds")
        lows = [lo for lo, _ in pairs]
        sizes = [hi - lo + 1 for lo, hi in pairs]
        total = 1
        for s in sizes:
            total *= s
        if total > 50_000_000:
            raise ValueError(f"OUT OF MEMORY: {name} would hold {total} elements")
        if typ is not None:
            self._check_type(typ)
            self.symbols.array_types[name] = typ
        else:
            self.symbols.array_types.pop(name, None)
        kind = self._array_kind(name)
        old = self.arrays.get(name)
        if kind in self.symbols.types:
            data = [self._new_record(kind) for _ in range(total)]
        else:
            data = [self._default_of_kind(kind)] * total
        if preserve and isinstance(old, list):
            keep = min(len(old), total)
            data[:keep] = old[:keep]
        self.arrays[name] = data
        self.symbols.array_dims[name] = sizes
        self.symbols.array_lb[name] = lows
        self.symbols.dimmed.add(name)

    def _array_kind(self, name: str) -> str:
        return self.symbols.array_types.get(name) or self._kind(name)

    def _try_exec_dim(self, stmt: str, run_vars: Any = None) -> bool:
        """DIM [SHARED] decl[, decl ...] — arrays, typed scalars and records."""
        m = re.match(r'DIM\s+(SHARED\s+)?(.+)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        shared = bool(m.group(1))
        for decl in self._split_arg_list(m.group(2)):
            dm = _RE_DECL.match(decl)
            if not dm:
                raise ValueError(f"SYNTAX ERROR: DIM {decl}")
            name, bounds = dm.group(1), dm.group(2)
            typ = type_text(dm.group(3), dm.group(4)) if dm.group(3) else None
            if shared:
                self.symbols.shared.add(name)
            if bounds is not None and bounds.strip():
                self._make_array(name, bounds, typ, run_vars)
                continue
            if typ is not None:
                self._check_type(typ)
                suffix = SUFFIX_KIND.get(name[-1:])
                if suffix and suffix != typ.split('*')[0]:
                    raise ValueError(f"DUPLICATE DEFINITION: '{name}' AS {typ}")
                self.symbols.var_types[name] = typ
            current = self.variables.get(name)
            if typ is not None and typ in self.symbols.types:
                same = isinstance(current, Record) and current.type_name == typ
                value: Any = current if same else self._new_record(typ)
            elif current is None:
                value = self._default_value(name)
            else:
                try:
                    value = self._coerce(name, current)
                except ValueError:
                    value = self._default_value(name)
            self.variables[name] = value
            if run_vars is not None and run_vars is not self.variables:
                run_vars[name] = value
        return True

    def _try_exec_dim_type(self, stmt: str) -> bool:
        """DIM name AS TypeName (kept for callers of the record form)."""
        return self._try_exec_dim(stmt)

    def _dim_record(self, var_name: str, type_name: str) -> None:
        self.symbols.var_types[var_name] = type_name.upper()
        self.variables[var_name] = self._new_record(type_name)

    def _try_exec_redim(self, stmt: str, run_vars: Any = None) -> bool:
        """REDIM [PRESERVE] name(bounds) [AS type][, ...] — plain REDIM clears,
        PRESERVE keeps the existing elements."""
        m = re.match(r'REDIM\s+(PRESERVE\s+)?(.+)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        preserve = bool(m.group(1))
        for decl in self._split_arg_list(m.group(2)):
            dm = _RE_DECL.match(decl)
            if not dm or not dm.group(2):
                raise ValueError(f"SYNTAX ERROR: REDIM {decl}")
            typ = (type_text(dm.group(3), dm.group(4)) if dm.group(3)
                   else self.symbols.array_types.get(dm.group(1)))
            self._make_array(dm.group(1), dm.group(2), typ, run_vars, preserve=preserve)
        return True

    def _try_exec_erase(self, stmt: str) -> bool:
        """ERASE name[, name ...] — delete arrays."""
        m = re.match(r'ERASE\s+(.+)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        for name in (n.strip() for n in m.group(1).split(',')):
            if name:
                self._drop_array(name)
        return True

    def _drop_array(self, name: str) -> None:
        self.arrays.pop(name, None)
        self.symbols.array_dims.pop(name, None)
        self.symbols.array_lb.pop(name, None)
        self.symbols.array_types.pop(name, None)
        self.symbols.dimmed.discard(name)

    # ── Element access ───────────────────────────────────────────────

    def _array_flat_index(self, name: str, indices: list, *, for_write: bool = False) -> int:
        """Flat (row-major) position of an element; creates an undeclared
        array on first use (see the module docstring)."""
        if name not in self.arrays:
            if len(indices) > 1:
                sizes = [IMPLICIT_UPPER - self.symbols.option_base + 1] * len(indices)
                total = 1
                for s in sizes:
                    total *= s
                self.arrays[name] = [self._default_value(name) for _ in range(total)]
                self.symbols.array_dims[name] = sizes
                self.symbols.array_lb[name] = [self.symbols.option_base] * len(indices)
                self.symbols.dimmed.add(name)
            else:
                self.arrays[name] = []
        data = self.arrays[name]
        dims = self.symbols.array_dims.get(name)
        lows = self.symbols.array_lb.get(name)
        subs = [round_half_even(i) for i in indices]
        if dims is not None and len(subs) != len(dims):
            raise ValueError(f"WRONG NUMBER OF DIMENSIONS: {name} has {len(dims)}")
        if lows is None:
            lows = [self.symbols.option_base] * len(subs)
        if len(subs) == 1 and dims is None:
            idx = subs[0] - lows[0]
            if idx < 0:
                raise ValueError(f"SUBSCRIPT OUT OF RANGE: {name}({subs[0]})")
            if idx >= len(data):
                if not for_write:
                    raise ValueError(f"SUBSCRIPT OUT OF RANGE: {name}({subs[0]})")
                kind = self._array_kind(name)
                data.extend(self._default_of_kind(kind) for _ in range(idx + 1 - len(data)))
            return idx
        flat, stride = 0, 1
        for k in range(len(subs) - 1, -1, -1):
            size = dims[k] if dims is not None else len(data)
            i = subs[k] - lows[k]
            if i < 0 or i >= size:
                shown = ', '.join(str(s) for s in subs)
                raise ValueError(f"SUBSCRIPT OUT OF RANGE: {name}({shown})")
            flat += i * stride
            stride *= size
        return flat

    def _array_read(self, name: str, indices: list) -> Any:
        """Element of a BASIC array (OPTION BASE and declared bounds applied;
        several indices address a multi-dimensional array)."""
        return self.arrays[name][self._array_flat_index(name, indices)]

    def _array_write(self, name: str, indices: list, value: Any, field: str | None = None) -> None:
        """Assign an element of array `name`, or field path `field` of a record element."""
        self._assert_assignable(name)
        idx = self._array_flat_index(name, indices, for_write=True)
        if field:
            elem = self.arrays[name][idx]
            if not isinstance(elem, Record):
                raise ValueError(f"TYPE MISMATCH: {name}() elements are not records")
            self._record_set(elem, field, value)
            return
        self.arrays[name][idx] = self._coerce(name, value, self._array_kind(name))

    def _array_bound(self, fn: str, name: str, dim: Any = 1) -> int:
        """LBOUND(name[, dim]) / UBOUND(name[, dim])."""
        if name not in self.arrays:
            raise ValueError(f"SUBSCRIPT OUT OF RANGE: array {name} is not dimensioned")
        k = round_half_even(dim) - 1
        dims = self.symbols.array_dims.get(name) or [len(self.arrays[name])]
        lows = self.symbols.array_lb.get(name) or [self.symbols.option_base] * len(dims)
        if not 0 <= k < len(dims):
            raise ValueError(f"SUBSCRIPT OUT OF RANGE: {name} has {len(dims)} dimension(s)")
        return lows[k] if fn == 'LBOUND' else lows[k] + dims[k] - 1

    # ── Lvalues (SWAP, INPUT, READ, MID$, by-reference arguments) ──────

    def _lvalue(self, text: str, run_vars: Any) -> LValue:
        """The assignable reference `text` names, its subscripts evaluated."""
        m = _RE_LVALUE.match(text)
        if not m:
            raise ValueError(f"SYNTAX ERROR: '{text.strip()}' is not a variable")
        name, subs, field = m.group(1), m.group(2), m.group(3)
        if subs is None:
            return LValue(name)
        values = [self._safe_eval(s, extra_ns=run_vars) for s in self._split_arg_list(subs)]
        path = re.sub(r'\s+', '', field or '').lstrip('.') or None
        return LValue(name, values, path)

    def _lvalue_kind(self, lv: LValue) -> str:
        """The type a value assigned through `lv` takes."""
        if lv.field:
            elem_type = self.symbols.array_types.get(lv.name, '')
            if elem_type not in self.symbols.types:
                raise ValueError(f"TYPE MISMATCH: {lv.name}() elements are not records")
            return self._path_kind(elem_type, lv.field)
        if lv.subs is not None:
            return self._array_kind(lv.name)
        return self._kind(lv.name)

    def _read_lvalue(self, lv: LValue, run_vars: Any = None) -> Any:
        if lv.subs is None:
            field = self._record_of(lv.name, run_vars)
            if field is not None:
                return self._record_get(*field)
            if run_vars is not None and lv.name in run_vars:
                return run_vars[lv.name]
            return self.variables.get(lv.name, self._default_value(lv.name))
        if lv.name not in self.arrays:
            return self._default_value(lv.name, self._array_kind(lv.name))
        elem = self._array_read(lv.name, lv.subs)
        return self._record_get(elem, lv.field) if lv.field else elem

    def _write_lvalue(self, lv: LValue, value: Any, run_vars: Any = None) -> None:
        if lv.subs is None:
            self._store_var(lv.name, value, run_vars)
        else:
            self._array_write(lv.name, lv.subs, value, lv.field)


# ---------------------------------------------------------------------------
# QUBASIC expression evaluation on the QBASIC-precedence engine (basic_expr).
# ---------------------------------------------------------------------------

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
    v = round_half_even(x)
    if not -32768 <= v <= 32767:
        raise ValueError("OVERFLOW")
    return v


def _clng(x: Any) -> int:
    """CLNG: nearest integer, ties to even, within the 32-bit range."""
    v = round_half_even(x)
    if not -2147483648 <= v <= 2147483647:
        raise ValueError("OVERFLOW")
    return v


class _Names:
    """Identifier resolution for one evaluation: the caller's run variables,
    then program variables, constants, arrays and functions."""

    __slots__ = ('t', 'extra')

    def __init__(self, terminal: Any, extra: Any) -> None:
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
            from qubasic_core.engine import MidCircuitRead
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
        fns = t.symbols.fns
        key = up if up.startswith('FN') else 'FN' + up
        if key in fns:
            f = fns[key]
            return t._call_user_fn_expr(f['params'], f['body'], tuple(args))
        if up in t.procs.functions:
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

    def length(self, node: tuple) -> int | None:
        """LEN of a variable, element or field whose type has a fixed size:
        the bytes QBASIC stores it in (INTEGER 2, LONG 4, SINGLE 4, DOUBLE 8,
        STRING * n n, a record the sum of its fields). None for a
        variable-length string, which LEN measures by its value."""
        t = self.t
        if node[0] == 'name':
            if node[1] in t.arrays:
                return None
            kind = t._kind(node[1])
        elif node[0] == 'call' and node[1] in t.arrays:
            kind = t._array_kind(node[1])
        elif node[0] == 'field' and node[1][0] == 'call' and node[1][1] in t.symbols.array_types:
            kind = t._path_kind(t.symbols.array_types[node[1][1]], node[2])
        else:
            return None
        value = evaluate(node, self)
        if isinstance(value, Record):
            kind = value.type_name
        if kind == 'STRING' or (isinstance(value, str) and not fixed_length(kind)):
            return None
        return t._kind_size(kind, value)

    def array_ref(self, name: str) -> Any:
        """`name()` as an argument: the whole array, when there is one."""
        if name in self.t.arrays:
            return ArrayRef(name)
        return None


# Names that read as a value with no argument list (and also accept "()").
_ZERO_ARG = {
    'TIMER': '_run_timer', 'RND': '_rnd', 'DATE$': '_date_str', 'TIME$': '_time_str',
    'INKEY$': '_inkey', 'FREEFILE': '_freefile', 'CSRLIN': '_csrlin', 'ERDEV': '_erdev',
    'ERDEV$': '_erdev_str',
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

# Matrix literals take complex arguments to these.
_COMPLEX_FUNCS = {'exp': cmath.exp, 'sqrt': cmath.sqrt, 'log': cmath.log,
                  'sin': cmath.sin, 'cos': cmath.cos, 'tan': cmath.tan}

_BUILTINS: dict[str, Any] = {}


def _builtin_table(safe_funcs: dict[str, Any]) -> dict[str, Any]:
    """Upper-case name -> function: the math functions plus the string and
    mixed functions (built on first use)."""
    if not _BUILTINS:
        _BUILTINS.update({k.upper(): v for k, v in safe_funcs.items()})
        _BUILTINS.update(STRING_FUNCS)
        _BUILTINS.update(MIXED_FUNCS)
    return _BUILTINS


class ExpressionMixin(LanguageHost):
    """Expression evaluation for QBasicTerminal (no eval())."""

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

    def _eval_operand(self, expr: str, run_vars: MutableMapping[str, Any] | None = None) -> Any:
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
        if up == 'POINT':
            return self._gfx_point
        if up == 'SEEK':
            return self._seek_fn
        if up == 'LOC':
            return self._loc_fn
        if up == 'INPUT$':
            return self._input_chars
        if up == 'PMAP':
            return self._gfx_pmap
        if up == 'SCREEN':
            return self._screen_fn
        if up == 'PLAY':
            return lambda n=0: self._bg_remaining()
        if up == 'ENVIRON$':
            return self._environ_fn
        if up == 'FILEATTR':
            return self._fileattr
        if up == 'LPOS':
            return lambda n=0: self.files.printer_col + 1
        if up == 'INP':
            return self._inp
        if up in _ZERO_ARG:
            method = getattr(self, _ZERO_ARG[up])
            return method if up == 'RND' else (lambda *a: method())
        if up == 'POS':
            return lambda x=0: self.screen.col + 1
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
        if not self.procs.functions and prog.funcs:
            self._scan_subs(prog)
        return prog

    # ── Evaluation ──────────────────────────────────────────────────────

    def _safe_eval(self, expr: Any, extra_ns: Any = None) -> Any:
        """Evaluate a BASIC expression (see basic_expr for the grammar)."""
        tree = parse(str(expr))
        return evaluate(tree, _Names(self, extra_ns))

    def eval_expr(self, expr: str) -> float:
        """Evaluate a mathematical expression with variables."""
        try:
            return float(self._safe_eval(expr))
        except ValueError:
            raise
        except Exception:
            raise ValueError(f"CANNOT EVALUATE: {expr}")

    def _eval_int(self, expr: Any, run_vars: MutableMapping[str, Any] | None = None) -> int:
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
        constants, math functions and j/im allowed, with exp, sqrt, log and
        the trigonometric functions taking complex values, so exp(j*PI/4)
        works) without eval()."""
        ns = {**self._SAFE_CONSTS, **self._SAFE_FUNCS, **_COMPLEX_FUNCS, 'j': 1j, 'im': 1j}
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
        if name in self.symbols.consts:
            raise ValueError(f"DUPLICATE DEFINITION: '{name}' is a CONST")
        if name.upper() in self._RESERVED_CONST_NAMES:
            raise ValueError(
                f"RESERVED: '{name}' is a built-in constant; choose another name")

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
