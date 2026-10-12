"""QUBASIC expressions: a QBASIC-precedence parser and a tree evaluator.

An expression is tokenized and parsed once (cached by its text) into a small
tree of tuples, then evaluated against a name resolver, so a loop body or a
live shot re-evaluates without re-parsing.

Precedence, lowest first::

    IMP   EQV   XOR   OR   AND   NOT
    comparisons  = == <> != >< < > <= >=   (chained: 0 <= x <= 10)
    + -   MOD   \\ //   * / %   unary - +   ^ (left to right) and ** (right to left)
    calls name(args), indexing name[args], atoms

Comparisons and NOT give -1 for true and 0 for false. AND, OR, XOR, EQV and
IMP are bitwise on integers, so they also combine those truth values. MOD and
\\ round their operands to integers (ties to even, as QBASIC converts); MOD
takes the sign of the dividend and \\ truncates toward zero. `//`, `%` and
`**` keep their Python meanings. LBOUND and UBOUND take an array name, and
`name()` as a call argument passes the whole array. `a(i).x` reads a field of
a record element (record variables are names: p.x).
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Callable

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
        elif kind == 'name':
            up = tok.upper()
            tokens.append(('op', up) if up in _WORD_OPS else ('name', tok))
        else:
            tokens.append(('op', tok))
    return tokens


class _Parser:
    """Recursive-descent parser over the token list (one level per precedence)."""

    def __init__(self, tokens: list[tuple[str, Any]], text: str):
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
    name(n), call(n, args), index(n, args) and bound(LBOUND|UBOUND, n, args)."""
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
