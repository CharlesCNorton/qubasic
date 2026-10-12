"""QUBASIC variables and arrays: types, records, declarations and storage.

Types follow QBASIC. A name's suffix fixes its type (% INTEGER, & LONG,
! SINGLE, # DOUBLE, $ STRING); otherwise `DIM name AS <type>` or a
DEFINT/DEFLNG/DEFSNG/DEFDBL/DEFSTR letter range does; otherwise the name is
SINGLE. INTEGER and LONG values round on assignment (ties to even) and
overflow outside 16 and 32 bits; SINGLE and DOUBLE hold the evaluated number;
a STRING holds text, and a `STRING * n` holds exactly n characters (padded
with spaces, or cut).

A TYPE block declares a record type; its fields take any of those types or
another record type:

    TYPE Point
        x AS SINGLE
        label AS STRING * 8
    END TYPE
    DIM p AS Point, path(1 TO 10) AS Point
    p.x = 1.5: path(3).label = "start"

A record variable holds a Record (a dict of field values that knows its
type); `p = q` copies a whole record of the same type.

Arrays are declared by `DIM name(bounds) [AS type]`, each bound an
expression `hi` (lower bound from OPTION BASE) or `lo TO hi`, several
declarations to a DIM. Subscripts round to integers and are checked against
the declared bounds. An array used without DIM is created on first use: one
dimension grows as it is written, several dimensions get 0 (or OPTION BASE)
to 10 each, as QBASIC dimensions an undeclared array. LBOUND and UBOUND
report the bounds. CONST names a value that cannot be assigned again.
"""

from __future__ import annotations

import copy
import re
from typing import Any, NamedTuple

from qubasic_core.basic_expr import round_half_even
from qubasic_core.host import HostState

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

    def __init__(self, type_name: str, fields: dict | None = None):
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


class VariableMixin(HostState):
    """Typed variables, records, CONST, DIM/REDIM/ERASE and array storage."""

    def _init_variables(self) -> None:
        self._var_types: dict[str, str] = {}
        self._deftypes: dict[str, str] = {}
        self._consts: set[str] = set()
        self._global_shared: set[str] = set()
        self._array_lb: dict[str, list[int]] = {}
        self._array_types: dict[str, str] = {}

    def _reset_declarations(self, prog: Any = None) -> None:
        """Forget the last run's declarations, then apply the program's
        DEFtype statements (they hold for the whole program, wherever they
        sit, as in QBASIC) and its TYPE blocks."""
        self._var_types.clear()
        self._consts.clear()
        self._global_shared.clear()
        self._deftypes.clear()
        if prog is None:
            return
        for text in prog.texts:
            m = _RE_DEFTYPE.match(text.strip())
            if m:
                kind = _DEF_KIND[m.group(1).upper()]
                for letter in letter_ranges(m.group(2)):
                    self._deftypes[letter] = kind
        for name, fields in getattr(prog, 'types', {}).items():
            self._user_types[name] = list(fields)

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
        declared = self._var_types.get(name)
        if declared:
            return declared
        return self._deftypes.get(name[:1].upper(), 'SINGLE')

    def _type_fields(self, type_name: str) -> list[tuple[str, str]]:
        fields = self._user_types.get(type_name.upper())
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
        if kind in self._user_types:
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
        if kind in self._user_types:
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
        m = re.match(r'CONST\s+(.+)$', stmt, re.IGNORECASE)
        if not m:
            return False
        for decl in self._split_arg_list(m.group(1)):
            dm = re.match(r'([A-Za-z_]\w*[$%!#&]?)\s*=\s*(.+)$', decl)
            if not dm:
                raise ValueError(f"SYNTAX ERROR: CONST {decl}")
            name = dm.group(1)
            if name.upper() in self._RESERVED_CONST_NAMES:
                raise ValueError(f"RESERVED: '{name}' is a built-in constant")
            value = self._safe_eval(dm.group(2), extra_ns=run_vars)
            if name[-1:] in SUFFIX_KIND:
                value = self._coerce(name, value)
            self._consts.add(name)
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
            lo = self._option_base
            hi = round_half_even(self._safe_eval(spec, extra_ns=run_vars))
        if hi < lo:
            raise ValueError(f"SUBSCRIPT OUT OF RANGE: bounds {lo} TO {hi}")
        return lo, hi

    def _check_type(self, typ: str) -> None:
        if typ not in TYPE_NAMES and typ not in self._user_types and not typ.startswith('STRING*'):
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
            self._array_types[name] = typ
        else:
            self._array_types.pop(name, None)
        kind = self._array_kind(name)
        old = self.arrays.get(name)
        if kind in self._user_types:
            data = [self._new_record(kind) for _ in range(total)]
        else:
            data = [self._default_of_kind(kind)] * total
        if preserve and isinstance(old, list):
            keep = min(len(old), total)
            data[:keep] = old[:keep]
        self.arrays[name] = data
        self._array_dims[name] = sizes
        self._array_lb[name] = lows
        self._dimmed_arrays.add(name)

    def _array_kind(self, name: str) -> str:
        return self._array_types.get(name) or self._kind(name)

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
                self._global_shared.add(name)
            if bounds is not None and bounds.strip():
                self._make_array(name, bounds, typ, run_vars)
                continue
            if typ is not None:
                self._check_type(typ)
                suffix = SUFFIX_KIND.get(name[-1:])
                if suffix and suffix != typ.split('*')[0]:
                    raise ValueError(f"DUPLICATE DEFINITION: '{name}' AS {typ}")
                self._var_types[name] = typ
            current = self.variables.get(name)
            if typ is not None and typ in self._user_types:
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
        self._var_types[var_name] = type_name.upper()
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
                   else self._array_types.get(dm.group(1)))
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
        self._array_dims.pop(name, None)
        self._array_lb.pop(name, None)
        self._array_types.pop(name, None)
        self._dimmed_arrays.discard(name)

    # ── Element access ───────────────────────────────────────────────

    def _array_flat_index(self, name: str, indices: list, *, for_write: bool = False) -> int:
        """Flat (row-major) position of an element; creates an undeclared
        array on first use (see the module docstring)."""
        if name not in self.arrays:
            if len(indices) > 1:
                sizes = [IMPLICIT_UPPER - self._option_base + 1] * len(indices)
                total = 1
                for s in sizes:
                    total *= s
                self.arrays[name] = [self._default_value(name) for _ in range(total)]
                self._array_dims[name] = sizes
                self._array_lb[name] = [self._option_base] * len(indices)
                self._dimmed_arrays.add(name)
            else:
                self.arrays[name] = []
        data = self.arrays[name]
        dims = self._array_dims.get(name)
        lows = self._array_lb.get(name)
        subs = [round_half_even(i) for i in indices]
        if dims is not None and len(subs) != len(dims):
            raise ValueError(f"WRONG NUMBER OF DIMENSIONS: {name} has {len(dims)}")
        if lows is None:
            lows = [self._option_base] * len(subs)
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
        dims = self._array_dims.get(name) or [len(self.arrays[name])]
        lows = self._array_lb.get(name) or [self._option_base] * len(dims)
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
            elem_type = self._array_types.get(lv.name, '')
            if elem_type not in self._user_types:
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
