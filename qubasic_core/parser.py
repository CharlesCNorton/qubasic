"""QUBASIC syntax: the gate table, the statement patterns, the typed
statements, the statement parser, the compiler that turns numbered lines into
a flat statement list, and the catalog of statements and functions that
HELP, `qubasic --spec` and docs/reference.md describe.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterator

# ---------------------------------------------------------------------------
# The gate vocabulary: each gate name with its parameter and qubit counts,
# and the aliases the parser accepts.
# ---------------------------------------------------------------------------

GATE_TABLE = {
    # 0-param, 1-qubit
    'H':    (0, 1), 'X':    (0, 1), 'Y':    (0, 1), 'Z':    (0, 1),
    'S':    (0, 1), 'T':    (0, 1), 'SDG':  (0, 1), 'TDG':  (0, 1),
    'SX':   (0, 1), 'ID':   (0, 1),
    # 1-param, 1-qubit
    'RX':   (1, 1), 'RY':   (1, 1), 'RZ':   (1, 1), 'P':    (1, 1),
    # 3-param, 1-qubit
    'U':    (3, 1),
    # 0-param, 2-qubit
    'CX':   (0, 2), 'CNOT': (0, 2), 'CZ':   (0, 2), 'CY':   (0, 2),
    'CH':   (0, 2), 'SWAP': (0, 2), 'DCX':  (0, 2), 'ISWAP':(0, 2),
    # 1-param, 2-qubit
    'CRX':  (1, 2), 'CRY':  (1, 2), 'CRZ':  (1, 2), 'CP':   (1, 2),
    'RXX':  (1, 2), 'RYY':  (1, 2), 'RZZ':  (1, 2),
    # 0-param, 3-qubit
    'CCX':  (0, 3), 'TOFFOLI': (0, 3), 'CSWAP': (0, 3), 'FREDKIN': (0, 3),
}

# Aliases
GATE_ALIASES = {
    'CNOT': 'CX', 'TOFFOLI': 'CCX', 'FREDKIN': 'CSWAP',
}


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns for the QUBASIC parser.
# ---------------------------------------------------------------------------

# ═══════════════════════════════════════════════════════════════════════
# Pre-compiled regexes
# ═══════════════════════════════════════════════════════════════════════

RE_LINE_NUM = re.compile(r'^(\d+)\s*(.*)')
RE_DEF_SINGLE = re.compile(r'(\w+)(?:\(([^)]*)\))?\s*=\s*(.*)')
RE_DEF_BEGIN = re.compile(r'DEF\s+BEGIN\s+(\w+)(?:\(([^)]*)\))?', re.IGNORECASE)
RE_REG_INDEX = re.compile(r'(\w+)\[(\d+)\]')
RE_AT_REG = re.compile(r'@([A-Z])\s+', re.IGNORECASE)
RE_AT_REG_LINE = re.compile(r'@([A-Z])\s+(.*)', re.IGNORECASE)
RE_SEND = re.compile(r'SEND\s+([A-Z])\s+(\S+)\s*->\s*(\w+)', re.IGNORECASE)
RE_SHARE = re.compile(r'SHARE\s+([A-Z])\s+(\d+)\s*,?\s*([A-Z])\s+(\d+)', re.IGNORECASE)
RE_MEAS = re.compile(r'MEAS\s+(\S+)\s*->\s*(\w+)', re.IGNORECASE)
RE_RESET = re.compile(r'RESET\s+(\S+)', re.IGNORECASE)
RE_UNITARY = re.compile(r'UNITARY\s+(\w+)\s*=\s*(\[.+\])', re.IGNORECASE)
RE_DIM = re.compile(r'DIM\s+(\w+\$?)\((\d+)\)', re.IGNORECASE)
RE_REDIM = re.compile(r'REDIM\s+(PRESERVE\s+)?(\w+)\((\d+)\)', re.IGNORECASE)
RE_ERASE = re.compile(r'ERASE\s+(\w+)', re.IGNORECASE)
RE_GET = re.compile(r'GET\s+(\w+\$?)', re.IGNORECASE)
RE_INPUT = re.compile(r'INPUT\s+(?:"([^"]*)"\s*,\s*)?(\w+)', re.IGNORECASE)
RE_CTRL = re.compile(r'CTRL\s+(\w+)\s+(.*)', re.IGNORECASE)
RE_INV = re.compile(r'INV\s+(\w+)\s+(.*)', re.IGNORECASE)
RE_LET_ARRAY = re.compile(r'LET\s+(\w+[$%!#&]?)\s*\((.+?)\)\s*=\s*(.*)', re.IGNORECASE)
RE_LET_VAR = re.compile(r'LET\s+([A-Za-z_][\w.]*[%!#&]?)\s*=\s*(.*)', re.IGNORECASE)
# LET a(i).field = expr — a field of a record array element (field path: .x or .pos.x)
RE_LET_FIELD = re.compile(
    r'LET\s+(\w+[$%!#&]?)\s*\((.+)\)\s*((?:\.\s*[A-Za-z_]\w*[$%!#&]?\s*)+)=\s*(.*)', re.IGNORECASE)
RE_PRINT = re.compile(r'PRINT\s+(.*)', re.IGNORECASE)
RE_GOTO = re.compile(r'GOTO\s+(\w+)\s*$', re.IGNORECASE)
RE_GOSUB = re.compile(r'GOSUB\s+(\w+)\s*$', re.IGNORECASE)
RE_FOR = re.compile(
    r'FOR\s+(\w+[%!#&]?)\s*=\s*(.+?)\s+TO\s+(.+?)(?:\s+STEP\s+(.+))?\s*$', re.IGNORECASE)
RE_NEXT = re.compile(r'NEXT(?:\s+(\w+[%!#&]?(?:\s*,\s*\w+[%!#&]?)*))?\s*$', re.IGNORECASE)
RE_WHILE = re.compile(r'WHILE\s+(.+)$', re.IGNORECASE)
RE_IF_THEN = re.compile(
    r'IF\s+(.+?)\s+THEN(?:\s+(.*?))?(?:\s+ELSE\s+(.*))?$', re.IGNORECASE)
RE_ELSEIF = re.compile(
    r'IF\s+(.+?)\s+THEN\s+(.*?)\s+(?:ELSEIF|ELSE\s+IF)\s+(.+)$', re.IGNORECASE)
RE_GOTO_GOSUB_TARGET = re.compile(r'(GOTO|GOSUB)\s+(\d+)', re.IGNORECASE)
RE_MEASURE_BASIS = re.compile(
    r'MEASURE_(X|Y|Z)\s+(\S+)', re.IGNORECASE)
RE_SYNDROME = re.compile(
    r'SYNDROME\s+(.*)', re.IGNORECASE)

# ── Classic BASIC, memory, SUB/FUNCTION, debug ──────────────────────

RE_DATA = re.compile(r'DATA\s+(.*)', re.IGNORECASE)
RE_READ = re.compile(r'READ\s+(.*)', re.IGNORECASE)
RE_ON_GOTO = re.compile(r'ON\s+(.+?)\s+GOTO\s+(\w+(?:\s*,\s*\w+)*)\s*$', re.IGNORECASE)
RE_ON_GOSUB = re.compile(r'ON\s+(.+?)\s+GOSUB\s+(\w+(?:\s*,\s*\w+)*)\s*$', re.IGNORECASE)
RE_SELECT_CASE = re.compile(r'SELECT\s+CASE\s+(.*)', re.IGNORECASE)
RE_CASE = re.compile(r'CASE\s+(.*)', re.IGNORECASE)
RE_DO = re.compile(r'DO(?:\s+(WHILE|UNTIL)\s+(.+))?\s*$', re.IGNORECASE)
RE_LOOP_STMT = re.compile(r'LOOP(?:\s+(WHILE|UNTIL)\s+(.+))?\s*$', re.IGNORECASE)
RE_EXIT = re.compile(r'EXIT\s+(FOR|WHILE|DO|SUB|FUNCTION)\s*$', re.IGNORECASE)
# A parameter list may hold array parameters, a() or a() AS INTEGER.
_PARAMS = r'(?:\(((?:[^()]|\(\s*\))*)\))?'
RE_SUB = re.compile(r'SUB\s+(\w+[$%!#&]?)\s*' + _PARAMS + r'\s*(STATIC)?\s*$', re.IGNORECASE)
RE_END_SUB = re.compile(r'END\s+SUB\s*$', re.IGNORECASE)
RE_FUNCTION = re.compile(r'FUNCTION\s+(\w+[$%!#&]?)\s*' + _PARAMS + r'\s*(STATIC)?\s*$',
                         re.IGNORECASE)
RE_END_FUNCTION = re.compile(r'END\s+FUNCTION\s*$', re.IGNORECASE)
RE_CALL = re.compile(r'CALL\s+(\w+[$%!#&]?)\s*(?:\((.*)\))?\s*$', re.IGNORECASE)
RE_LOCAL = re.compile(r'LOCAL\s+(.*)', re.IGNORECASE)
RE_STATIC_DECL = re.compile(r'STATIC\s+(.*)', re.IGNORECASE)
RE_SHARED = re.compile(r'SHARED\s+(.*)', re.IGNORECASE)
RE_ON_ERROR = re.compile(r'ON\s+ERROR\s+GOTO\s+(\w+)\s*$', re.IGNORECASE)
RE_RESUME = re.compile(r'RESUME(?:\s+(.+))?\s*$', re.IGNORECASE)
RE_ERROR_STMT = re.compile(r'ERROR\s+(\d+)', re.IGNORECASE)
RE_ASSERT = re.compile(r'ASSERT\s+(.*)', re.IGNORECASE)
RE_SWAP = re.compile(r'SWAP\s+(.+)$', re.IGNORECASE)
RE_POKE = re.compile(r'POKE\s+(.+?)\s*,\s*(.+)', re.IGNORECASE)
RE_SYS = re.compile(r'SYS\s+(.+)', re.IGNORECASE)
RE_OPEN = re.compile(
    r'OPEN\s+(.+?)\s+FOR\s+(INPUT|OUTPUT|APPEND|RANDOM|BINARY)\b.*?\s+AS\s+#?\s*(.+?)'
    r'(?:\s+LEN\s*=.*?)?(?:\s+ENCODING\s+"?([^"]*)"?)?\s*$',
    re.IGNORECASE)
RE_CLOSE = re.compile(r'CLOSE\b\s*(.*)$', re.IGNORECASE)
RE_PRINT_FILE = re.compile(r'PRINT\s*#\s*([^,]+?)\s*,\s*(.*)', re.IGNORECASE)
RE_INPUT_FILE = re.compile(r'INPUT\s*#\s*([^,]+?)\s*,\s*(.+)', re.IGNORECASE)
RE_LINE_INPUT = re.compile(
    r'LINE\s+INPUT\s+(?:"([^"]*)"\s*,\s*)?(\w+\$?)', re.IGNORECASE)
RE_OPTION_BASE = re.compile(r'OPTION\s+BASE\s+([01])', re.IGNORECASE)
RE_OPTION_ENDIAN = re.compile(r'OPTION\s+ENDIAN\s+(BIG|LITTLE)', re.IGNORECASE)
RE_IMPORT = re.compile(r'IMPORT\s+"?([^"]+)"?', re.IGNORECASE)
RE_SAVE_EXPECT = re.compile(r'SAVE_EXPECT\s+(\w+)((?:[\s,]+\d+)*)\s*->\s*(\w+)', re.IGNORECASE)
RE_SAVE_PROBS = re.compile(r'SAVE_PROBS\s+([\d\s,]+)\s*->\s*(\w+)', re.IGNORECASE)
RE_SAVE_AMPS = re.compile(r'SAVE_AMPS\s+([\d\s,]+)\s*->\s*(\w+)', re.IGNORECASE)
RE_SET_STATE = re.compile(r'SET_STATE\s+(.*)', re.IGNORECASE)
RE_TYPE_BEGIN = re.compile(r'TYPE\s+(\w+)', re.IGNORECASE)
RE_TYPE_FIELD = re.compile(r'([A-Za-z_]\w*)\s+AS\s+([A-Za-z_]\w*)(?:\s*\*\s*(\d+))?\s*$',
                           re.IGNORECASE)
RE_END_TYPE = re.compile(r'END\s+TYPE', re.IGNORECASE)
RE_DIM_TYPE = re.compile(r'DIM\s+(\w+)\s+AS\s+(\w+)', re.IGNORECASE)
RE_CHAIN = re.compile(r'CHAIN\s+"?([^"]+)"?', re.IGNORECASE)
RE_MERGE = re.compile(r'MERGE\s+"?([^"]+)"?', re.IGNORECASE)
RE_DEF_FN = re.compile(
    r'DEF\s+FN\s*(\w+)\s*\(([^)]*)\)\s*=\s*(.*)', re.IGNORECASE)
RE_PRINT_USING = re.compile(r'PRINT\s+USING\s+(.+)$', re.IGNORECASE)
RE_COLOR = re.compile(r'COLOR\s+(\w+)(?:\s*,\s*(\w+))?', re.IGNORECASE)
RE_LOCATE = re.compile(r'LOCATE\s+(\d+)\s*,\s*(\d+)', re.IGNORECASE)
RE_SCREEN = re.compile(r'SCREEN\s+(\d+)', re.IGNORECASE)
RE_LPRINT = re.compile(r'LPRINT\s+(.*)', re.IGNORECASE)
RE_ON_MEASURE = re.compile(r'ON\s+MEASURE\s+GOSUB\s+(\d+)', re.IGNORECASE)
RE_ON_TIMER = re.compile(r'ON\s+TIMER\s*\((.+?)\)\s+GOSUB\s+(\w+)\s*$', re.IGNORECASE)
RE_DIM_MULTI = re.compile(r'DIM\s+(\w+\$?)\((\d+(?:\s*,\s*\d+)*)\)', re.IGNORECASE)
RE_LET_STR = re.compile(r'LET\s+(\w+\$)\s*=\s*(.*)', re.IGNORECASE)
# MID$(var$, start[, length]) = expr — the statement form of MID$.
RE_MID_ASSIGN = re.compile(r'MID\$\s*\((.+)\)\s*=\s*(.+)$', re.IGNORECASE)
RE_RETURN_TO = re.compile(r'RETURN\s+(\w+)\s*$', re.IGNORECASE)
RE_RESTORE_TO = re.compile(r'RESTORE\s+(\w+)\s*$', re.IGNORECASE)
# Implicit LET: an assignment written without the LET keyword (x = 5, s$ = "hi",
# a(1) = 5, p.x = 3). Anchored lvalue followed by a single '=' that is not part
# of '==' / '<=' / '>=' / '!=' / '<>'. Routed through the LET handlers.
RE_IMPLICIT_ASSIGN = re.compile(
    r'[A-Za-z_][\w.]*[$%!#&]?(?:\s*\((?:[^()]|\([^()]*\))*\)(?:\s*\.\s*[A-Za-z_]\w*[$%!#&]?)*)?'
    r'\s*=(?!=)',
    re.IGNORECASE)


# ---------------------------------------------------------------------------
# QUBASIC typed statement AST — produced by the parser, consumed by the executor.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Stmt:
    """Base for all parsed statements."""
    raw: str


# ── Terminals ──────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RemStmt(Stmt):
    pass

@dataclass(frozen=True, slots=True)
class MeasureStmt(Stmt):
    qubits: str = ''   # '' = measure all; else a comma/space list of qubit indices

@dataclass(frozen=True, slots=True)
class EndStmt(Stmt):
    pass

@dataclass(frozen=True, slots=True)
class ReturnStmt(Stmt):
    target: int | str | None = None    # RETURN <line|label> resumes there

@dataclass(frozen=True, slots=True)
class BarrierStmt(Stmt):
    pass

@dataclass(frozen=True, slots=True)
class WendStmt(Stmt):
    pass

@dataclass(frozen=True, slots=True)
class RestoreStmt(Stmt):
    target: int | str | None = None    # RESTORE <line|label>: DATA from that line

@dataclass(frozen=True, slots=True)
class EndSelectStmt(Stmt):
    pass

@dataclass(frozen=True, slots=True)
class ElseStmt(Stmt):
    pass

@dataclass(frozen=True, slots=True)
class EndIfStmt(Stmt):
    pass

@dataclass(frozen=True, slots=True)
class EndSubStmt(Stmt):
    pass

@dataclass(frozen=True, slots=True)
class EndFunctionStmt(Stmt):
    pass

@dataclass(frozen=True, slots=True)
class StopStmt(Stmt):
    pass

@dataclass(frozen=True, slots=True)
class RunStmt(Stmt):
    target: int | str | None = None    # RUN <line|label> starts there
    file: str | None = None            # RUN "file" (not inside a program)


# ── Control flow ───────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GotoStmt(Stmt):
    target: int | str          # a line number, or a label until compiled

@dataclass(frozen=True, slots=True)
class GosubStmt(Stmt):
    target: int | str

@dataclass(frozen=True, slots=True)
class ForStmt(Stmt):
    var: str
    start_expr: str
    end_expr: str
    step_expr: str | None

@dataclass(frozen=True, slots=True)
class NextStmt(Stmt):
    var: str

@dataclass(frozen=True, slots=True)
class WhileStmt(Stmt):
    condition: str

@dataclass(frozen=True, slots=True)
class IfThenStmt(Stmt):
    condition: str
    then_clause: str
    else_clause: str | None

@dataclass(frozen=True, slots=True)
class DoStmt(Stmt):
    kind: str | None   # WHILE or UNTIL or None
    condition: str | None

@dataclass(frozen=True, slots=True)
class LoopStmt(Stmt):
    kind: str | None
    condition: str | None

@dataclass(frozen=True, slots=True)
class ExitStmt(Stmt):
    target: str   # FOR, WHILE, DO, SUB, FUNCTION

@dataclass(frozen=True, slots=True)
class OnGotoStmt(Stmt):
    expr: str
    targets: tuple[int | str, ...]

@dataclass(frozen=True, slots=True)
class OnGosubStmt(Stmt):
    expr: str
    targets: tuple[int | str, ...]

@dataclass(frozen=True, slots=True)
class SelectCaseStmt(Stmt):
    expr: str

@dataclass(frozen=True, slots=True)
class CaseStmt(Stmt):
    value: str

@dataclass(frozen=True, slots=True)
class CallStmt(Stmt):
    name: str
    args: str

@dataclass(frozen=True, slots=True)
class SubStmt(Stmt):
    name: str
    params: str
    static: bool = False       # SUB name(...) STATIC: every local keeps its value

@dataclass(frozen=True, slots=True)
class FunctionStmt(Stmt):
    name: str
    params: str
    static: bool = False

@dataclass(frozen=True, slots=True)
class OnErrorStmt(Stmt):
    target: int | str

@dataclass(frozen=True, slots=True)
class ResumeStmt(Stmt):
    arg: str | None

@dataclass(frozen=True, slots=True)
class ErrorStmt(Stmt):
    code: int

@dataclass(frozen=True, slots=True)
class AssertStmt(Stmt):
    condition: str

@dataclass(frozen=True, slots=True)
class SwapStmt(Stmt):
    a: str
    b: str

@dataclass(frozen=True, slots=True)
class DefFnStmt(Stmt):
    name: str
    params: str
    body: str

@dataclass(frozen=True, slots=True)
class OptionBaseStmt(Stmt):
    base: int

@dataclass(frozen=True, slots=True)
class OptionEndianStmt(Stmt):
    big: bool

@dataclass(frozen=True, slots=True)
class OnMeasureStmt(Stmt):
    target: int

@dataclass(frozen=True, slots=True)
class OnTimerStmt(Stmt):
    interval: str
    target: int | str     # a line number, or a label until compiled

@dataclass(frozen=True, slots=True)
class DataStmt(Stmt):
    values: str

@dataclass(frozen=True, slots=True)
class ReadStmt(Stmt):
    var_list: str

@dataclass(frozen=True, slots=True)
class LocalStmt(Stmt):
    var_list: str

@dataclass(frozen=True, slots=True)
class StaticStmt(Stmt):
    var_list: str

@dataclass(frozen=True, slots=True)
class SharedStmt(Stmt):
    var_list: str


# ── Assignment ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LetStmt(Stmt):
    name: str
    expr: str

@dataclass(frozen=True, slots=True)
class LetArrayStmt(Stmt):
    name: str
    index_expr: str
    value_expr: str
    field: str | None = None     # a(i).x = ...: the field path of a record element

@dataclass(frozen=True, slots=True)
class LetStrStmt(Stmt):
    name: str
    expr: str

@dataclass(frozen=True, slots=True)
class MidAssignStmt(Stmt):
    """MID$(var$, start[, length]) = expr: overwrite part of a string."""
    var: str
    start: str
    length: str | None
    expr: str


# ── I/O ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PrintStmt(Stmt):
    expr: str

@dataclass(frozen=True, slots=True)
class PrintUsingStmt(Stmt):
    fmt: str
    values: str

@dataclass(frozen=True, slots=True)
class InputStmt(Stmt):
    prompt: str | None
    var: str

@dataclass(frozen=True, slots=True)
class LineInputStmt(Stmt):
    prompt: str | None
    var: str

@dataclass(frozen=True, slots=True)
class GetStmt(Stmt):
    var: str

@dataclass(frozen=True, slots=True)
class DimStmt(Stmt):
    name: str
    size: str

@dataclass(frozen=True, slots=True)
class RedimStmt(Stmt):
    name: str
    size: str

@dataclass(frozen=True, slots=True)
class EraseStmt(Stmt):
    name: str

@dataclass(frozen=True, slots=True)
class PokeStmt(Stmt):
    addr_expr: str
    value_expr: str

@dataclass(frozen=True, slots=True)
class SysStmt(Stmt):
    arg: str

@dataclass(frozen=True, slots=True)
class UnitaryStmt(Stmt):
    name: str
    matrix: str

@dataclass(frozen=True, slots=True)
class OpenStmt(Stmt):
    path: str        # the file-name expression
    mode: str
    handle: str      # the file-number expression
    encoding: str | None = None

@dataclass(frozen=True, slots=True)
class CloseStmt(Stmt):
    handles: str     # the file numbers, '' for every file

@dataclass(frozen=True, slots=True)
class PrintFileStmt(Stmt):
    handle: str
    data: str

@dataclass(frozen=True, slots=True)
class InputFileStmt(Stmt):
    handle: str
    var: str

@dataclass(frozen=True, slots=True)
class LprintStmt(Stmt):
    expr: str

@dataclass(frozen=True, slots=True)
class ScreenStmt(Stmt):
    mode: str

@dataclass(frozen=True, slots=True)
class ColorStmt(Stmt):
    fg: str
    bg: str | None

@dataclass(frozen=True, slots=True)
class LocateStmt(Stmt):
    row: str
    col: str

@dataclass(frozen=True, slots=True)
class ImportStmt(Stmt):
    path: str

@dataclass(frozen=True, slots=True)
class ChainStmt(Stmt):
    path: str

@dataclass(frozen=True, slots=True)
class MergeStmt(Stmt):
    path: str


# ── Quantum ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MeasStmt(Stmt):
    qubit_expr: str
    var: str

@dataclass(frozen=True, slots=True)
class ResetStmt(Stmt):
    qubit_expr: str

@dataclass(frozen=True, slots=True)
class MeasureBasisStmt(Stmt):
    basis: str
    qubit_expr: str

@dataclass(frozen=True, slots=True)
class SyndromeStmt(Stmt):
    rest: str


# ── LOCC ───────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SendStmt(Stmt):
    reg: str
    qubit_expr: str
    var: str

@dataclass(frozen=True, slots=True)
class ShareStmt(Stmt):
    reg1: str
    q1: int
    reg2: str
    q2: int

@dataclass(frozen=True, slots=True)
class AtRegStmt(Stmt):
    reg: str
    inner: str


# ── Compound ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CompoundStmt(Stmt):
    parts: tuple[str, ...]


# ── Compiled control transfers (produced by qubasic_core.parser) ──────

@dataclass(frozen=True, slots=True)
class IfJumpStmt(Stmt):
    """An IF condition: execution continues at the next statement when it
    holds and at `on_false` when it does not. For a single-line IF the THEN
    statements are [then_start, then_end), the ELSE statements
    [else_start, else_end), and `end` follows the whole IF; a block IF has
    then_end = -1."""
    condition: str
    on_false: int
    then_start: int = -1
    then_end: int = -1
    else_start: int = -1
    else_end: int = -1
    end: int = -1


@dataclass(frozen=True, slots=True)
class JumpStmt(Stmt):
    """An unconditional transfer inside compiled block structure."""
    target: int


# ── Gate application ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GateStmt(Stmt):
    """Direct gate application: H 0, CX 0,1, RX PI/4, 0"""
    name: str          # canonical gate name (after alias resolution)
    args: tuple[str, ...]  # raw argument strings (params + qubits, unparsed)


# ── Fallback ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RawStmt(Stmt):
    """Unrecognized statement — falls through to legacy regex path."""
    pass


# ---------------------------------------------------------------------------
# QUBASIC parser — converts raw statement strings into typed Stmt objects.
# ---------------------------------------------------------------------------

# ═══════════════════════════════════════════════════════════════════════
# Keyword dispatch handlers
# ═══════════════════════════════════════════════════════════════════════

def _target(tok: str) -> int | str:
    """A jump target: a line number, or a label name."""
    tok = tok.strip()
    return int(tok) if tok.isdigit() else tok


def _top_split(text: str, seps: str = ',') -> list[str]:
    """Split at top-level separators (outside quotes and brackets)."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch == '"':
            quote = ch
            buf.append(ch)
        elif ch in '([':
            depth += 1
            buf.append(ch)
        elif ch in ')]':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in seps and depth == 0:
            parts.append(''.join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append(''.join(buf))
    return parts


def _handle_goto(text: str, raw: str) -> Stmt | None:
    m = RE_GOTO.match(text)
    if m:
        return GotoStmt(raw=raw, target=_target(m.group(1)))
    return None

def _handle_gosub(text: str, raw: str) -> Stmt | None:
    m = RE_GOSUB.match(text)
    if m:
        return GosubStmt(raw=raw, target=_target(m.group(1)))
    return None

def _handle_for(text: str, raw: str) -> Stmt | None:
    m = RE_FOR.match(text)
    if m:
        return ForStmt(raw=raw, var=m.group(1),
                       start_expr=m.group(2), end_expr=m.group(3),
                       step_expr=m.group(4))
    return None

def _handle_next(text: str, raw: str) -> Stmt | None:
    m = RE_NEXT.match(text)
    if m:
        return NextStmt(raw=raw, var=m.group(1) or '')
    return None

def _handle_while(text: str, raw: str) -> Stmt | None:
    m = RE_WHILE.match(text)
    if m:
        return WhileStmt(raw=raw, condition=m.group(1).strip())
    return None

def _handle_if(text: str, raw: str) -> Stmt | None:
    # Check for ELSEIF first — rewrite as nested IF in the ELSE clause
    m = RE_ELSEIF.match(text)
    if m:
        return IfThenStmt(raw=raw, condition=m.group(1).strip(),
                          then_clause=m.group(2).strip(),
                          else_clause='IF ' + m.group(3).strip())
    m = RE_IF_THEN.match(text)
    if m:
        return IfThenStmt(raw=raw, condition=m.group(1).strip(),
                          then_clause=m.group(2).strip() if m.group(2) else '',
                          else_clause=m.group(3).strip() if m.group(3) else None)
    return None

def _handle_do(text: str, raw: str) -> Stmt | None:
    m = RE_DO.match(text)
    if m:
        return DoStmt(raw=raw, kind=m.group(1), condition=m.group(2))
    return None

def _handle_loop(text: str, raw: str) -> Stmt | None:
    m = RE_LOOP_STMT.match(text)
    if m:
        return LoopStmt(raw=raw, kind=m.group(1), condition=m.group(2))
    return None

def _handle_exit(text: str, raw: str) -> Stmt | None:
    m = RE_EXIT.match(text)
    if m:
        return ExitStmt(raw=raw, target=m.group(1).upper())
    return None

_RE_ON_DEVICE = re.compile(r'ON\s+(KEY|PLAY|COM|PEN|STRIG)\b', re.IGNORECASE)


def _handle_on(text: str, raw: str) -> Stmt | None:
    m = RE_ON_MEASURE.match(text)
    if m:
        return OnMeasureStmt(raw=raw, target=int(m.group(1)))
    m = RE_ON_TIMER.match(text)
    if m:
        return OnTimerStmt(raw=raw, interval=m.group(1), target=_target(m.group(2)))
    m = RE_ON_ERROR.match(text)
    if m:
        return OnErrorStmt(raw=raw, target=_target(m.group(1)))
    if _RE_ON_DEVICE.match(text):
        return None                  # ON KEY(n)/PLAY/COM/PEN/STRIG GOSUB: an event trap
    m = RE_ON_GOTO.match(text)
    if m:
        targets = tuple(_target(t) for t in m.group(2).split(',') if t.strip())
        return OnGotoStmt(raw=raw, expr=m.group(1).strip(), targets=targets)
    m = RE_ON_GOSUB.match(text)
    if m:
        targets = tuple(_target(t) for t in m.group(2).split(',') if t.strip())
        return OnGosubStmt(raw=raw, expr=m.group(1).strip(), targets=targets)
    return None

def _handle_select(text: str, raw: str) -> Stmt | None:
    m = RE_SELECT_CASE.match(text)
    if m:
        return SelectCaseStmt(raw=raw, expr=m.group(1).strip())
    return None

def _handle_case(text: str, raw: str) -> Stmt | None:
    m = RE_CASE.match(text)
    if m:
        return CaseStmt(raw=raw, value=m.group(1).strip())
    return None

def _handle_sub(text: str, raw: str) -> Stmt | None:
    m = RE_SUB.match(text)
    if m:
        return SubStmt(raw=raw, name=m.group(1).upper(),
                       params=m.group(2) or '', static=bool(m.group(3)))
    return None

def _handle_function(text: str, raw: str) -> Stmt | None:
    m = RE_FUNCTION.match(text)
    if m:
        return FunctionStmt(raw=raw, name=m.group(1).upper(),
                            params=m.group(2) or '', static=bool(m.group(3)))
    return None

def _handle_call(text: str, raw: str) -> Stmt | None:
    m = RE_CALL.match(text)
    if m:
        return CallStmt(raw=raw, name=m.group(1).upper(),
                        args=m.group(2) or '')
    return None

def _handle_local(text: str, raw: str) -> Stmt | None:
    m = RE_LOCAL.match(text)
    if m:
        return LocalStmt(raw=raw, var_list=m.group(1))
    return None

def _handle_static(text: str, raw: str) -> Stmt | None:
    m = RE_STATIC_DECL.match(text)
    if m:
        return StaticStmt(raw=raw, var_list=m.group(1))
    return None

def _handle_shared(text: str, raw: str) -> Stmt | None:
    m = RE_SHARED.match(text)
    if m:
        return SharedStmt(raw=raw, var_list=m.group(1))
    return None

def _handle_resume(text: str, raw: str) -> Stmt | None:
    m = RE_RESUME.match(text)
    if m:
        return ResumeStmt(raw=raw, arg=m.group(1))
    return None

def _handle_error(text: str, raw: str) -> Stmt | None:
    m = RE_ERROR_STMT.match(text)
    if m:
        return ErrorStmt(raw=raw, code=int(m.group(1)))
    return None

def _handle_assert(text: str, raw: str) -> Stmt | None:
    m = RE_ASSERT.match(text)
    if m:
        return AssertStmt(raw=raw, condition=m.group(1).strip())
    return None

_RE_SWAP_OPERAND = re.compile(r'[A-Za-z_]\w*[$%!#&]?(?:\s*\(.*\))?$')


def _handle_swap(text: str, raw: str) -> Stmt | None:
    # SWAP a, b exchanges two variables (or array elements); with qubit
    # operands (SWAP 0, 1 / SWAP q, q+1 / SWAP r[0], r[1]) it is the gate.
    m = RE_SWAP.match(text)
    if m:
        parts = [p.strip() for p in _top_split(m.group(1))]
        if len(parts) == 2 and all(_RE_SWAP_OPERAND.match(p) for p in parts):
            return SwapStmt(raw=raw, a=parts[0], b=parts[1])
    return None

def _handle_def(text: str, raw: str) -> Stmt | None:
    m = RE_DEF_FN.match(text)
    if m:
        return DefFnStmt(raw=raw, name=m.group(1),
                         params=m.group(2), body=m.group(3))
    return None

def _handle_option(text: str, raw: str) -> Stmt | None:
    m = RE_OPTION_BASE.match(text)
    if m:
        return OptionBaseStmt(raw=raw, base=int(m.group(1)))
    m = RE_OPTION_ENDIAN.match(text)
    if m:
        return OptionEndianStmt(raw=raw, big=m.group(1).upper() == 'BIG')
    return None

def _handle_data(text: str, raw: str) -> Stmt | None:
    m = RE_DATA.match(text)
    if m:
        return DataStmt(raw=raw, values=m.group(1))
    return None

def _handle_read(text: str, raw: str) -> Stmt | None:
    m = RE_READ.match(text)
    if m:
        return ReadStmt(raw=raw, var_list=m.group(1))
    return None

def _handle_let(text: str, raw: str) -> Stmt | None:
    m = RE_LET_FIELD.match(text)
    if m:
        path = re.sub(r'\s+', '', m.group(3)).lstrip('.')
        return LetArrayStmt(raw=raw, name=m.group(1), index_expr=m.group(2),
                            value_expr=m.group(4), field=path)
    m = RE_LET_ARRAY.match(text)
    if m:
        return LetArrayStmt(raw=raw, name=m.group(1),
                            index_expr=m.group(2), value_expr=m.group(3))
    m = RE_LET_STR.match(text)
    if m:
        return LetStrStmt(raw=raw, name=m.group(1), expr=m.group(2))
    m = RE_LET_VAR.match(text)
    if m:
        return LetStmt(raw=raw, name=m.group(1), expr=m.group(2))
    return None

def _handle_print(text: str, raw: str) -> Stmt | None:
    m = RE_PRINT_USING.match(text)
    if m:
        head, *tail = _top_split(m.group(1), ';')
        return PrintUsingStmt(raw=raw, fmt=head.strip(), values=';'.join(tail).strip())
    m = RE_PRINT_FILE.match(text)
    if m:
        return PrintFileStmt(raw=raw, handle=m.group(1).strip(), data=m.group(2))
    m = RE_PRINT.match(text)
    if m:
        return PrintStmt(raw=raw, expr=m.group(1).strip())
    return None

def _handle_line(text: str, raw: str) -> Stmt | None:
    m = RE_LINE_INPUT.match(text)
    if m:
        return LineInputStmt(raw=raw, prompt=m.group(1), var=m.group(2))
    return None

def _handle_get(text: str, raw: str) -> Stmt | None:
    m = RE_GET.match(text)
    if m:
        return GetStmt(raw=raw, var=m.group(1))
    return None

def _handle_input(text: str, raw: str) -> Stmt | None:
    m = RE_INPUT_FILE.match(text)
    if m:
        return InputFileStmt(raw=raw, handle=m.group(1).strip(), var=m.group(2))
    m = RE_INPUT.match(text)
    if m:
        return InputStmt(raw=raw, prompt=m.group(1), var=m.group(2))
    return None

def _handle_lprint(text: str, raw: str) -> Stmt | None:
    m = RE_LPRINT.match(text)
    if m:
        return LprintStmt(raw=raw, expr=m.group(1))
    return None

def _handle_poke(text: str, raw: str) -> Stmt | None:
    m = RE_POKE.match(text)
    if m:
        return PokeStmt(raw=raw, addr_expr=m.group(1), value_expr=m.group(2))
    return None

def _handle_sys(text: str, raw: str) -> Stmt | None:
    m = RE_SYS.match(text)
    if m:
        return SysStmt(raw=raw, arg=m.group(1))
    return None

def _handle_unitary(text: str, raw: str) -> Stmt | None:
    m = RE_UNITARY.match(text)
    if m:
        return UnitaryStmt(raw=raw, name=m.group(1), matrix=m.group(2))
    return None

def _handle_redim(text: str, raw: str) -> Stmt | None:
    m = RE_REDIM.match(text)
    if m:
        return RedimStmt(raw=raw, name=m.group(1), size=m.group(2))
    return None

def _handle_erase(text: str, raw: str) -> Stmt | None:
    m = RE_ERASE.match(text)
    if m:
        return EraseStmt(raw=raw, name=m.group(1))
    return None

def _handle_dim(text: str, raw: str) -> Stmt | None:
    m = RE_DIM_MULTI.match(text)
    if m:
        return DimStmt(raw=raw, name=m.group(1), size=m.group(2))
    m = RE_DIM.match(text)
    if m:
        return DimStmt(raw=raw, name=m.group(1), size=m.group(2))
    return None

def _handle_open(text: str, raw: str) -> Stmt | None:
    m = RE_OPEN.match(text)
    if m:
        return OpenStmt(raw=raw, path=m.group(1).strip(),
                        mode=m.group(2).upper(), handle=m.group(3).strip(),
                        encoding=m.group(4).strip() if m.group(4) else None)
    return None

def _handle_close(text: str, raw: str) -> Stmt | None:
    m = RE_CLOSE.match(text)
    if m:
        return CloseStmt(raw=raw, handles=m.group(1).strip())
    return None

def _handle_import(text: str, raw: str) -> Stmt | None:
    m = RE_IMPORT.match(text)
    if m:
        return ImportStmt(raw=raw, path=m.group(1).strip())
    return None

def _handle_chain(text: str, raw: str) -> Stmt | None:
    m = RE_CHAIN.match(text)
    if m:
        return ChainStmt(raw=raw, path=m.group(1).strip())
    return None

def _handle_merge(text: str, raw: str) -> Stmt | None:
    m = RE_MERGE.match(text)
    if m:
        return MergeStmt(raw=raw, path=m.group(1).strip())
    return None

def _handle_screen(text: str, raw: str) -> Stmt | None:
    m = RE_SCREEN.match(text)
    if m:
        return ScreenStmt(raw=raw, mode=m.group(1))
    return None

def _handle_color(text: str, raw: str) -> Stmt | None:
    m = RE_COLOR.match(text)
    if m:
        return ColorStmt(raw=raw, fg=m.group(1), bg=m.group(2))
    return None

def _handle_locate(text: str, raw: str) -> Stmt | None:
    m = RE_LOCATE.match(text)
    if m:
        return LocateStmt(raw=raw, row=m.group(1), col=m.group(2))
    return None

def _handle_measure_basis(text: str, raw: str) -> Stmt | None:
    m = RE_MEASURE_BASIS.match(text)
    if m:
        return MeasureBasisStmt(raw=raw, basis=m.group(1).upper(),
                                qubit_expr=m.group(2))
    return None

def _handle_syndrome(text: str, raw: str) -> Stmt | None:
    m = RE_SYNDROME.match(text)
    if m:
        return SyndromeStmt(raw=raw, rest=m.group(1).strip())
    return None

def _handle_meas(text: str, raw: str) -> Stmt | None:
    m = RE_MEAS.match(text)
    if m:
        return MeasStmt(raw=raw, qubit_expr=m.group(1), var=m.group(2))
    return None

def _handle_reset(text: str, raw: str) -> Stmt | None:
    m = RE_RESET.match(text)
    if m:
        return ResetStmt(raw=raw, qubit_expr=m.group(1))
    return None

def _handle_send(text: str, raw: str) -> Stmt | None:
    m = RE_SEND.match(text)
    if m:
        return SendStmt(raw=raw, reg=m.group(1).upper(),
                        qubit_expr=m.group(2), var=m.group(3))
    return None

def _handle_share(text: str, raw: str) -> Stmt | None:
    m = RE_SHARE.match(text)
    if m:
        return ShareStmt(raw=raw, reg1=m.group(1).upper(), q1=int(m.group(2)),
                         reg2=m.group(3).upper(), q2=int(m.group(4)))
    return None


# ═══════════════════════════════════════════════════════════════════════
# First-word dispatch table
# ═══════════════════════════════════════════════════════════════════════

_KEYWORD_PARSERS: dict[str, Callable[[str, str], Stmt | None]] = {
    'GOTO': _handle_goto,
    'GOSUB': _handle_gosub,
    'FOR': _handle_for,
    'NEXT': _handle_next,
    'WHILE': _handle_while,
    'IF': _handle_if,
    'DO': _handle_do,
    'LOOP': _handle_loop,
    'EXIT': _handle_exit,
    'ON': _handle_on,
    'SELECT': _handle_select,
    'CASE': _handle_case,
    'SUB': _handle_sub,
    'FUNCTION': _handle_function,
    'CALL': _handle_call,
    'LOCAL': _handle_local,
    'STATIC': _handle_static,
    'SHARED': _handle_shared,
    'RESUME': _handle_resume,
    'ERROR': _handle_error,
    'ASSERT': _handle_assert,
    'SWAP': _handle_swap,
    'DEF': _handle_def,
    'OPTION': _handle_option,
    'DATA': _handle_data,
    'READ': _handle_read,
    'LET': _handle_let,
    'PRINT': _handle_print,
    'LINE': _handle_line,
    'GET': _handle_get,
    'INPUT': _handle_input,
    'LPRINT': _handle_lprint,
    'POKE': _handle_poke,
    'SYS': _handle_sys,
    'UNITARY': _handle_unitary,
    'REDIM': _handle_redim,
    'ERASE': _handle_erase,
    'DIM': _handle_dim,
    'OPEN': _handle_open,
    'CLOSE': _handle_close,
    'IMPORT': _handle_import,
    'CHAIN': _handle_chain,
    'MERGE': _handle_merge,
    'SCREEN': _handle_screen,
    'COLOR': _handle_color,
    'LOCATE': _handle_locate,
    'MEASURE_X': _handle_measure_basis,
    'MEASURE_Y': _handle_measure_basis,
    'MEASURE_Z': _handle_measure_basis,
    'SYNDROME': _handle_syndrome,
    'MEAS': _handle_meas,
    'RESET': _handle_reset,
    'SEND': _handle_send,
    'SHARE': _handle_share,
}


# A statement starting with one of these owns the rest of its line: an IF's
# clauses run to the end of the line, a comment or DEF body keeps its colons.
_LINE_OWNERS = re.compile(r"\s*(IF|ELSEIF|REM|DEF)\b|\s*'", re.IGNORECASE)
_RE_DATA_START = re.compile(r'\s*DATA\b', re.IGNORECASE)
_RE_REM_START = re.compile(r'\s*REM\b', re.IGNORECASE)


def strip_comment(text: str) -> str:
    """`text` without its trailing ' comment. An apostrophe outside a string
    ends the line's code, except after REM and inside a DATA statement, whose
    unquoted items may hold one (DATA 1, 2 : ' note comments a DATA line).
    A line that is only a comment comes back unchanged."""
    if "'" not in text or _RE_REM_START.match(text):
        return text
    in_string = False
    data = bool(_RE_DATA_START.match(text))
    for i, ch in enumerate(text):
        if ch == '"':
            in_string = not in_string
        elif in_string:
            continue
        elif ch == ':':
            rest = text[i + 1:]
            if _RE_REM_START.match(rest):
                return text
            data = bool(_RE_DATA_START.match(rest))
        elif ch == "'" and not data:
            code = text[:i].rstrip()
            return code if code.strip() else text
    return text


def _colon_pieces(stmt: str) -> list[str]:
    """Split at top-level colons: outside double-quoted strings, and not after
    a statement that owns the rest of its line."""
    pieces: list[str] = []
    buf: list[str] = []
    in_string = False
    for i, ch in enumerate(stmt):
        if ch == '"':
            in_string = not in_string
        elif ch == ':' and not in_string:
            current = ''.join(buf)
            if _LINE_OWNERS.match(current):
                buf.append(stmt[i:])
                break
            pieces.append(current)
            buf = []
            continue
        buf.append(ch)
    pieces.append(''.join(buf))
    return pieces


def _split_colon_stmts(stmt: str) -> list[str]:
    """Split colon-separated statements, inheriting @register prefixes."""
    parts = []
    last_reg = None
    for sub in _colon_pieces(stmt):
        sub = sub.strip()
        if not sub:
            continue
        m_reg = RE_AT_REG.match(sub)
        if m_reg:
            last_reg = m_reg.group(1).upper()
        elif last_reg and not sub.upper().startswith((
                'SEND', 'IF ', 'REM', 'FOR', 'NEXT',
                'SHARE', 'MEASURE', '@')):
            sub = f"@{last_reg} {sub}"
        parts.append(sub)
    return parts


@lru_cache(maxsize=8192)
def parse_stmt(raw: str) -> Stmt:
    """Parse a raw statement string into a typed Stmt.

    Returns RawStmt only for gate applications and truly unrecognized input.
    Memoized: Stmt objects are frozen, so identical text (e.g. an ELSEIF chain's
    nested clause re-evaluated each loop iteration) is parsed only once.
    """
    text = raw.strip()
    if not text:
        return RawStmt(raw=raw)
    if "'" in text:
        code = strip_comment(text)
        if code != text:                      # a trailing ' comment
            return parse_stmt(code)

    upper = text.upper()

    # ? is PRINT, as the QBASIC editor reads it.
    if text.startswith('?'):
        if not text[1:].strip():
            return PrintStmt(raw=raw, expr='')
        return parse_stmt('PRINT ' + text[1:].lstrip())

    # ── Terminals ──────────────────────────────────────────────────
    if upper.startswith('REM') or upper.startswith("'"):
        return RemStmt(raw=raw)
    if upper == 'MEASURE':
        return MeasureStmt(raw=raw)
    # MEASURE <qubit list> — partial measurement of a subset (digits/commas only,
    # so MEASURE_X and similar are not captured here).
    m_sub = re.match(r'MEASURE\s+([\d,\s]+)$', text, re.IGNORECASE)
    if m_sub:
        return MeasureStmt(raw=raw, qubits=m_sub.group(1).strip())
    if upper in ('END', 'SYSTEM'):
        return EndStmt(raw=raw)
    if upper in ('END IF', 'ENDIF'):
        return EndIfStmt(raw=raw)
    if upper == 'ELSE':
        return ElseStmt(raw=raw)
    if upper == 'RETURN':
        return ReturnStmt(raw=raw)
    m_ret = RE_RETURN_TO.match(text)
    if m_ret:
        return ReturnStmt(raw=raw, target=_target(m_ret.group(1)))
    if upper == 'BARRIER':
        return BarrierStmt(raw=raw)
    if upper == 'WEND':
        return WendStmt(raw=raw)
    if upper == 'RESTORE':
        return RestoreStmt(raw=raw)
    m_res = RE_RESTORE_TO.match(text)
    if m_res:
        return RestoreStmt(raw=raw, target=_target(m_res.group(1)))
    if upper == 'END SELECT':
        return EndSelectStmt(raw=raw)
    if upper == 'STOP':
        return StopStmt(raw=raw)
    m_run = re.match(r'RUN(?:\s+(.+))?$', text, re.IGNORECASE)
    if m_run:
        arg = (m_run.group(1) or '').strip()
        if arg.startswith('"'):
            return RunStmt(raw=raw, file=arg.strip('"'))
        return RunStmt(raw=raw, target=_target(arg) if arg else None)
    if upper == 'PRINT':
        return PrintStmt(raw=raw, expr='')       # a blank line

    # ── END SUB / END FUNCTION ────────────────────────────────────
    m = RE_END_SUB.match(text)
    if m:
        return EndSubStmt(raw=raw)
    m = RE_END_FUNCTION.match(text)
    if m:
        return EndFunctionStmt(raw=raw)

    # ── Compound (colon-separated) ────────────────────────────────
    # Split before keyword dispatch, so PRINT 1: PRINT 2 or FOR ... : NEXT
    # is a compound rather than one statement swallowing the rest.
    if ':' in text and not text.startswith('@'):
        parts = _split_colon_stmts(text)
        if len(parts) > 1:
            return CompoundStmt(raw=raw, parts=tuple(parts))
        if len(parts) == 1 and parts[0] != text:      # a trailing colon ends the statement
            return parse_stmt(parts[0])

    # ── First-word dispatch ───────────────────────────────────────
    first_word = text.split(None, 1)[0].upper()
    handler = _KEYWORD_PARSERS.get(first_word)
    if handler is not None:
        result = handler(text, raw)
        if result is not None:
            return result

    # ── @REG lines ────────────────────────────────────────────────
    if text.startswith('@'):
        # A colon-compound @REG line (e.g. "@A H 0 : @A CX 0,1" or
        # "@A H 0 : CX 0,1") must split into per-statement parts with
        # @register inheritance; otherwise the whole tail is captured as one
        # AtRegStmt.inner and the LOCC executor mis-tokenizes it.
        if ':' in text:
            parts = _split_colon_stmts(text)
            if len(parts) > 1:
                return CompoundStmt(raw=raw, parts=tuple(parts))
        m = RE_AT_REG_LINE.match(text)
        if m:
            return AtRegStmt(raw=raw, reg=m.group(1).upper(),
                             inner=m.group(2).strip())

    # ── Compound (colon-separated) ────────────────────────────────
    if ':' in text:
        parts = _split_colon_stmts(text)
        if len(parts) > 1:
            return CompoundStmt(raw=raw, parts=tuple(parts))

    # ── MID$(s$, start[, length]) = expr ───────────────────────────
    m = RE_MID_ASSIGN.match(text)
    if m:
        mid_args = [a.strip() for a in _top_split(m.group(1))]
        if 2 <= len(mid_args) <= 3:
            return MidAssignStmt(raw=raw, var=mid_args[0], start=mid_args[1],
                                 length=mid_args[2] if len(mid_args) == 3 else None,
                                 expr=m.group(2).strip())

    # ── Implicit LET (assignment without the LET keyword) ──────────
    # A line whose lvalue is followed by a single '=' is an assignment, even
    # when it collides with a gate name (X = 5 is "assign X", not the X gate).
    # Tried before gate dispatch so bare assignment works as in every BASIC.
    if RE_IMPLICIT_ASSIGN.match(text):
        result = _handle_let(f"LET {text}", raw)
        if result is not None:
            return result

    # ── Gate application ────────────────────────────────────────────
    canonical = GATE_ALIASES.get(first_word, first_word)
    if canonical in GATE_TABLE:
        rest_args = text.split(None, 1)[1].strip() if ' ' in text.strip() else ''
        if ',' in rest_args:
            args = tuple(a.strip() for a in rest_args.split(',') if a.strip())
        else:
            args = tuple(a.strip() for a in rest_args.split() if a.strip())
        return GateStmt(raw=raw, name=canonical, args=args)

    # ── Fallback (subroutine calls, truly unrecognized) ─────────────
    return RawStmt(raw=raw)


# ---------------------------------------------------------------------------
# QUBASIC program compilation: numbered lines into a flat statement list.
#
# Every statement of every line gets its own index, so control transfers are
# plain indices: GOSUB returns to the next statement, a loop's body starts at
# the statement after its opener, and a single-line IF becomes a conditional
# jump over its THEN statements (and an unconditional jump over its ELSE
# statements). Block structure (FOR/NEXT, WHILE/WEND, DO/LOOP, block
# IF/ELSEIF/ELSE/END IF, SELECT CASE, SUB/FUNCTION, EXIT) is matched here
# once, so a skip or an EXIT is a stored index, and mismatched blocks and
# missing GOTO/GOSUB targets are reported before anything runs.
#
# A line may start with a label (`Start:`), and every jump that takes a line
# number also takes a label: GOTO, GOSUB, ON ... GOTO/GOSUB, ON ERROR GOTO,
# RESUME, RETURN and RESTORE. A statement that starts with the name of a SUB
# calls it, as `CALL name(args)` does. A TYPE ... END TYPE block declares a
# record type (`prog.types`); its lines do not execute.
# ---------------------------------------------------------------------------

_RE_BLOCK_ELSEIF = re.compile(r'ELSE\s*IF\s+(.+?)\s+THEN\s*$', re.IGNORECASE)
_RE_TYPE_HEAD = re.compile(r'TYPE\s+([A-Za-z_]\w*)\s*$', re.IGNORECASE)
_RE_TYPE_FIELD = re.compile(r'([A-Za-z_]\w*)\s+AS\s+([A-Za-z_]\w*)(?:\s*\*\s*(\d+))?\s*$',
                            re.IGNORECASE)
_RE_END_TYPE = re.compile(r'END\s+TYPE\s*$', re.IGNORECASE)
_RE_IF_GOTO = re.compile(r'IF\s+(.+?)\s+GOTO\s+(\w+)\s*$', re.IGNORECASE)
_RE_LABEL = re.compile(r'([A-Za-z_]\w*)\s*:(?![=:])\s*(.*)$', re.DOTALL)
_RE_CALL_WORD = re.compile(r'([A-Za-z_]\w*[$%!#&]?)\s*(.*)$', re.DOTALL)
# Statements that may stand alone before a colon, so `WORD:` is not a label.
_NOT_LABELS = frozenset({
    'END', 'RETURN', 'ELSE', 'WEND', 'LOOP', 'NEXT', 'STOP', 'CLS', 'BEEP',
    'SYSTEM', 'RESTORE', 'BARRIER', 'MEASURE', 'PRINT', 'DO', 'RANDOMIZE',
    'ENDIF', 'REM', 'TRON', 'TROFF', 'SLEEP', 'RESUME', 'WRITE', 'CONT',
})


class CompileError(RuntimeError):
    """A program whose structure cannot run (unmatched block, missing target)."""


class CompiledProgram(Sequence[int]):
    """The statements of a program in execution order.

    Indexing gives the line number of a statement, so code that only knows
    line numbers reads a CompiledProgram like the sorted line list."""

    def __init__(self) -> None:
        self.lines: list[int] = []
        self.texts: list[str] = []
        self.stmts: list[Stmt] = []
        self.first: list[bool] = []          # the first statement of its line
        self.line_ip: dict[int, int] = {}    # line number -> its first statement
        self.skip: dict[int, int] = {}       # FOR/WHILE/DO/CASE/SUB/FUNCTION -> past the block
        self.exits: dict[int, int] = {}      # EXIT -> its target
        self.selects: dict[int, tuple[tuple[tuple[int, str], ...], int]] = {}
        self.subs: dict[str, tuple[int, int, list[str]]] = {}    # name -> (body, END ip, params)
        self.funcs: dict[str, tuple[int, int, list[str]]] = {}
        self.static_routines: set[str] = set()   # SUB/FUNCTION ... STATIC
        self.labels: dict[str, int] = {}         # LABEL (upper case) -> line number
        self.types: dict[str, list[tuple[str, str]]] = {}   # TYPE -> [(field, type)]

    def __getitem__(self, ip: int) -> int:                 # type: ignore[override]
        return self.lines[ip]

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self) -> Iterator[int]:
        return iter(self.lines)

    def append(self, line: int, text: str, stmt: Stmt | None, first: bool) -> int:
        self.lines.append(line)
        self.texts.append(text)
        self.stmts.append(stmt)               # type: ignore[arg-type]
        self.first.append(first)
        return len(self.lines) - 1

    def patch(self, ip: int, **changes: Any) -> None:
        self.stmts[ip] = dataclasses.replace(self.stmts[ip], **changes)


def compile_program(program: dict[int, str]) -> CompiledProgram:
    """Compile a {line: text} program (see the module docstring)."""
    prog = CompiledProgram()
    open_type: tuple[str, int, list[tuple[str, str]]] | None = None
    for line in sorted(program):
        prog.line_ip[line] = len(prog.lines)
        text = strip_comment(program[line].strip())
        if open_type is not None or _RE_TYPE_HEAD.match(text):
            open_type = _type_line(prog, line, text, open_type)
            continue
        m = _RE_LABEL.match(text)
        if m and m.group(1).upper() not in _NOT_LABELS:
            label = m.group(1).upper()
            if label in prog.labels:
                raise CompileError(f"LINE {line}: DUPLICATE LABEL {m.group(1)} "
                                   f"(also line {prog.labels[label]})")
            prog.labels[label] = line
            text = m.group(2).strip()
            if not text:
                prog.append(line, '', RemStmt(raw=program[line].strip()), True)
                continue
        parsed = parse_stmt(text)
        parts = list(parsed.parts) if isinstance(parsed, CompoundStmt) else [text]
        for k, part in enumerate(parts):
            _emit(prog, line, part, k == 0)
        if not parts:
            prog.append(line, '', parse_stmt(''), True)
    if open_type is not None:
        raise CompileError(f"LINE {open_type[1]}: TYPE {open_type[0]} without END TYPE")
    _resolve_labels(prog)
    _match_blocks(prog)
    _bind_sub_calls(prog)
    return prog


def _type_line(prog: CompiledProgram, line: int, text: str,
               open_type: tuple[str, int, list[tuple[str, str]]] | None) -> tuple[str, int, list[tuple[str, str]]] | None:
    """One line of a TYPE block (the header, a field or END TYPE); returns
    the block still open, or None after END TYPE."""
    from qubasic_core.expression import TYPE_NAMES, type_text
    prog.append(line, text, RemStmt(raw=text), True)
    if open_type is None:
        name = _RE_TYPE_HEAD.match(text).group(1).upper()  # type: ignore[union-attr]
        if name in TYPE_NAMES or name in prog.types:
            raise CompileError(f"LINE {line}: DUPLICATE DEFINITION: TYPE {name}")
        return name, line, []
    if _RE_END_TYPE.match(text):
        if not open_type[2]:
            raise CompileError(f"LINE {line}: TYPE {open_type[0]} has no fields")
        prog.types[open_type[0]] = open_type[2]
        return None
    if not text or text.upper().startswith('REM') or text.startswith("'"):
        return open_type
    m = _RE_TYPE_FIELD.match(text)
    if not m:
        raise CompileError(f"LINE {line}: '{text}' in TYPE {open_type[0]} is not a field "
                           f"(name AS type)")
    open_type[2].append((m.group(1).lower(), type_text(m.group(2), m.group(3))))
    return open_type


def _resolve_labels(prog: CompiledProgram) -> None:
    """Replace label targets with line numbers."""
    def line_of(ip: int, target: Any) -> Any:
        if isinstance(target, int) or target is None:
            return target
        if isinstance(target, str) and target.isdigit():
            return int(target)
        line = prog.labels.get(str(target).upper())
        if line is None:
            raise CompileError(f"LINE {prog.lines[ip]}: LABEL {target} NOT FOUND")
        return line

    for ip, st in enumerate(prog.stmts):
        if isinstance(st, (GotoStmt, GosubStmt, OnErrorStmt, OnTimerStmt, RunStmt)) and isinstance(st.target, str):
            prog.patch(ip, target=line_of(ip, st.target))
        elif isinstance(st, (ReturnStmt, RestoreStmt)) and isinstance(st.target, str):
            prog.patch(ip, target=line_of(ip, st.target))
        elif isinstance(st, (OnGotoStmt, OnGosubStmt)) and any(isinstance(t, str) for t in st.targets):
            prog.patch(ip, targets=tuple(line_of(ip, t) for t in st.targets))
        elif isinstance(st, ResumeStmt) and st.arg and st.arg.strip().upper() != 'NEXT' \
                and not st.arg.strip().isdigit():
            prog.patch(ip, arg=str(line_of(ip, st.arg.strip())))


def _bind_sub_calls(prog: CompiledProgram) -> None:
    """A statement starting with a SUB's name calls it (QBASIC needs no CALL)."""
    if not prog.subs:
        return
    for ip, st in enumerate(prog.stmts):
        if not isinstance(st, RawStmt):
            continue
        m = _RE_CALL_WORD.match(prog.texts[ip])
        if not m or m.group(1).upper() not in prog.subs:
            continue
        args = m.group(2).strip()
        if args.startswith('(') and args.endswith(')') and _balanced(args[1:-1]):
            args = args[1:-1]
        prog.stmts[ip] = CallStmt(raw=prog.texts[ip], name=m.group(1).upper(), args=args)


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _emit(prog: CompiledProgram, line: int, text: str, first: bool) -> None:
    text = text.strip()
    parsed = parse_stmt(text)
    if isinstance(parsed, IfThenStmt) and (parsed.then_clause or parsed.else_clause is not None):
        _emit_single_if(prog, line, text, parsed, first)
        return
    m = _RE_IF_GOTO.match(text)
    if m and not isinstance(parsed, IfThenStmt):
        _emit_single_if(prog, line, text, IfThenStmt(raw=text, condition=m.group(1),
                                                     then_clause=f"GOTO {m.group(2)}",
                                                     else_clause=None), first)
        return
    m = _RE_BLOCK_ELSEIF.match(text)
    if m and not isinstance(parsed, IfThenStmt):
        # Reached by falling out of the previous branch: leave the IF; reached
        # by that branch's false condition: test this one.
        prog.append(line, text, JumpStmt(raw=text, target=-1), first)
        prog.append(line, text, IfJumpStmt(raw=text, condition=m.group(1), on_false=-1), False)
        return
    if isinstance(parsed, NextStmt) and ',' in parsed.var:
        for k, var in enumerate(v.strip() for v in parsed.var.split(',')):
            prog.append(line, f"NEXT {var}", NextStmt(raw=f"NEXT {var}", var=var), first and k == 0)
        return
    prog.append(line, text, parsed, first)


def _emit_clause(prog: CompiledProgram, line: int, clause: str) -> None:
    clause = clause.strip()
    if not clause:
        return
    if clause.isdigit():                      # IF cond THEN 100
        clause = f"GOTO {clause}"
    for part in _split_colon_stmts(clause):
        _emit(prog, line, part, False)


def _emit_single_if(prog: CompiledProgram, line: int, text: str, p: IfThenStmt,
                    first: bool) -> None:
    ip = prog.append(line, text, IfJumpStmt(raw=text, condition=p.condition, on_false=-1), first)
    then_start = len(prog)
    _emit_clause(prog, line, p.then_clause or '')
    if p.else_clause is not None:
        jump = prog.append(line, text, JumpStmt(raw=text, target=-1), False)
        then_end, else_start = jump, jump + 1
        _emit_clause(prog, line, p.else_clause)
        end = len(prog)
        prog.patch(jump, target=end)
    else:
        then_end = else_start = end = len(prog)
    prog.patch(ip, on_false=else_start, then_start=then_start, then_end=then_end,
               else_start=else_start, else_end=end, end=end)


def _match_blocks(prog: CompiledProgram) -> None:
    """Pair block openers with closers and store every jump target."""
    def fail(ip: int, msg: str) -> CompileError:
        return CompileError(f"LINE {prog.lines[ip]}: {msg}")

    loops: list[tuple[str, int, str]] = []        # (kind, opener ip, FOR variable)
    pending_exits: dict[int, list[int]] = {}       # loop/SUB opener ip -> EXIT ips
    ifs: list[dict[str, Any]] = []
    selects: list[tuple[int, list[tuple[int, str]]]] = []
    routines: list[tuple[str, int]] = []           # ('SUB'|'FUNCTION', opener ip)

    for ip, st in enumerate(prog.stmts):
        if isinstance(st, ForStmt):
            loops.append(('FOR', ip, st.var))
        elif isinstance(st, NextStmt):
            if not loops or loops[-1][0] != 'FOR':
                raise fail(ip, f"NEXT {st.var}".rstrip() + " without FOR")
            _, opener, var = loops.pop()
            if st.var and st.var.upper() != var.upper():
                raise fail(ip, f"NEXT {st.var} does not match FOR {var}")
            _close_loop(prog, opener, ip, pending_exits)
        elif isinstance(st, WhileStmt):
            loops.append(('WHILE', ip, ''))
        elif isinstance(st, WendStmt):
            if not loops or loops[-1][0] != 'WHILE':
                raise fail(ip, "WEND without WHILE")
            _close_loop(prog, loops.pop()[1], ip, pending_exits)
        elif isinstance(st, DoStmt):
            loops.append(('DO', ip, ''))
        elif isinstance(st, LoopStmt):
            if loops and loops[-1][0] == 'DO':
                _close_loop(prog, loops.pop()[1], ip, pending_exits)
            # (a LOOP outside DO may name a SUB; it runs as a call)
        elif isinstance(st, ExitStmt):
            kind = st.target.upper()
            if kind in ('SUB', 'FUNCTION'):
                owner = next((o for k, o in reversed(routines) if k == kind), None)
            else:
                owner = next((o for k, o, _ in reversed(loops) if k == kind), None)
            if owner is None:
                if kind in ('FOR', 'WHILE', 'DO'):
                    raise fail(ip, f"EXIT {kind} outside a {kind} loop")
            else:
                pending_exits.setdefault(owner, []).append(ip)
        elif isinstance(st, IfJumpStmt) and st.on_false == -1 and st.then_end == -1:
            before = prog.stmts[ip - 1] if ip > 0 else None
            if isinstance(before, JumpStmt) and before.target == -1 \
                    and prog.texts[ip - 1] == prog.texts[ip]:
                # block ELSEIF: the jump before it leaves the previous branch
                if not ifs:
                    raise fail(ip, "ELSEIF without IF")
                top = ifs[-1]
                if top['branch'] is None:
                    raise fail(ip, "ELSEIF after ELSE")
                prog.patch(top['branch'], on_false=ip)
                top['jumps'].append(ip - 1)
                top['branch'] = ip
            else:
                ifs.append({'branch': ip, 'jumps': []})
        elif isinstance(st, IfThenStmt):
            # block IF: test, then fall into the THEN lines
            prog.stmts[ip] = IfJumpStmt(raw=st.raw, condition=st.condition, on_false=-1)
            ifs.append({'branch': ip, 'jumps': []})
        elif isinstance(st, ElseStmt):
            if ifs and ifs[-1]['branch'] is not None:
                top = ifs[-1]
                prog.patch(top['branch'], on_false=ip + 1)
                prog.stmts[ip] = JumpStmt(raw=st.raw, target=-1)
                top['jumps'].append(ip)
                top['branch'] = None
        elif isinstance(st, EndIfStmt):
            if ifs:
                top = ifs.pop()
                if top['branch'] is not None:
                    prog.patch(top['branch'], on_false=ip + 1)
                for j in top['jumps']:
                    prog.patch(j, target=ip + 1)
        elif isinstance(st, SelectCaseStmt):
            selects.append((ip, []))
        elif isinstance(st, CaseStmt):
            if selects:
                selects[-1][1].append((ip, st.value))
        elif isinstance(st, EndSelectStmt):
            if not selects:
                raise fail(ip, "END SELECT without SELECT CASE")
            opener, cases = selects.pop()
            prog.selects[opener] = (tuple(cases), ip)
            for case_ip, _ in cases:
                prog.skip[case_ip] = ip + 1
        elif isinstance(st, (SubStmt, FunctionStmt)):
            routines.append(('SUB' if isinstance(st, SubStmt) else 'FUNCTION', ip))
        elif isinstance(st, (EndSubStmt, EndFunctionStmt)):
            kind = 'SUB' if isinstance(st, EndSubStmt) else 'FUNCTION'
            if routines and routines[-1][0] == kind:
                _, opener = routines.pop()
                head = prog.stmts[opener]
                params = [p.strip() for p in head.params.split(',') if p.strip()]  # type: ignore[attr-defined]
                table = prog.subs if kind == 'SUB' else prog.funcs
                table[head.name.upper()] = (opener + 1, ip, params)  # type: ignore[attr-defined]
                if getattr(head, 'static', False):
                    prog.static_routines.add(head.name.upper())  # type: ignore[attr-defined]
                prog.skip[opener] = ip + 1
                for e in pending_exits.pop(opener, []):
                    prog.exits[e] = ip
        elif isinstance(st, (GotoStmt, GosubStmt)):
            if st.target not in prog.line_ip:
                word = 'GOTO' if isinstance(st, GotoStmt) else 'GOSUB'
                raise fail(ip, f"{word} {st.target} — target line not found")
        elif isinstance(st, (OnGotoStmt, OnGosubStmt)):
            word = 'ON GOTO' if isinstance(st, OnGotoStmt) else 'ON GOSUB'
            for t in st.targets:
                if t not in prog.line_ip:
                    raise fail(ip, f"{word} {t} — target line not found")
        elif isinstance(st, OnErrorStmt):
            if st.target != 0 and st.target not in prog.line_ip:
                raise fail(ip, f"ON ERROR GOTO {st.target} — target line not found")
        elif isinstance(st, (ReturnStmt, RestoreStmt)):
            if st.target is not None and st.target not in prog.line_ip:
                word = 'RETURN' if isinstance(st, ReturnStmt) else 'RESTORE'
                raise fail(ip, f"{word} {st.target} — target line not found")
        elif isinstance(st, ResumeStmt):
            arg = (st.arg or '').strip()
            if arg.isdigit() and int(arg) != 0 and int(arg) not in prog.line_ip:
                raise fail(ip, f"RESUME {arg} — target line not found")

    if loops:
        kind, opener, _ = loops[-1]
        closer = {'FOR': 'NEXT', 'WHILE': 'WEND', 'DO': 'LOOP'}[kind]
        raise fail(opener, f"{kind} without {closer}")
    if ifs:
        raise fail(ifs[-1]['branch'], "IF block has no matching END IF")
    if selects:
        raise fail(selects[-1][0], "SELECT CASE without END SELECT")


def _close_loop(prog: CompiledProgram, opener: int, closer: int,
                pending_exits: dict[int, list[int]]) -> None:
    prog.skip[opener] = closer + 1
    for e in pending_exits.pop(opener, []):
        prog.exits[e] = closer + 1


def line_index(lines: Any, target: int) -> int | None:
    """Index of the first statement of line `target` (None when absent), for
    a CompiledProgram or a plain sorted line list."""
    table = getattr(lines, 'line_ip', None)
    if table is not None:
        return table.get(target)
    for i, ln in enumerate(lines):
        if ln == target:
            return i
    return None


# ---------------------------------------------------------------------------
# The statement and function catalog.
#
# Every program statement and built-in function, with its syntax and a
# one-line description, grouped by topic. HELP <keyword>, `qubasic --spec`
# and docs/reference.md (written by tools/generate.py) read it; the
# commands typed at the prompt are described by their own docstrings.
# ---------------------------------------------------------------------------

# The commands typed at the prompt, by topic (every command in exactly one).
COMMAND_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('Program editing', ('LIST', 'EDIT', 'DELETE', 'RENUM', 'AUTO', 'COPY', 'MOVE', 'FIND', 'REPLACE',
                         'UNDO', 'NEW', 'BANK', 'DIFF', 'CHECKSUM', 'EXPLAIN')),
    ('Programs and files', ('LOAD', 'SAVE', 'MERGE', 'CHAIN', 'INCLUDE', 'IMPORT', 'DIR', 'FILES', 'OPEN',
                            'CLOSE', 'LOADQASM', 'EXPORT', 'CSV', 'CLIP', 'SAVEPNG', 'SAVEWAV')),
    ('Running and debugging', ('RUN', 'STEP', 'CONT', 'TRON', 'TROFF', 'BREAK', 'WATCH', 'REWIND',
                               'FORWARD', 'HISTORY', 'PROFILE', 'STATS', 'SEED')),
    ('Definitions and variables', ('LET', 'VARS', 'CLEAR', 'RESTORE', 'DEF', 'DEFS', 'TYPE', 'REG', 'REGS',
                                   'CIRCUIT_DEF', 'APPLY_CIRCUIT', 'SET_STATE', 'SET_DENSITY')),
    ('Configuration', ('QUBITS', 'SHOTS', 'METHOD', 'PRECISION', 'STATUS', 'VERSION', 'COUPLING', 'BASIS',
                       'DEVICE', 'OPTIMIZE', 'RAM', 'BENCH')),
    ('Noise and mitigation', ('NOISE', 'MITIGATE', 'ZNE')),
    ('Screen and sound', ('SCREEN', 'DISPLAY', 'SHOW', 'CLS', 'COLOR', 'LOCATE', 'PLAY', 'AUDIO', 'PROMPT', 'THEME')),
    ('Results and state', ('HIST', 'PROBS', 'STATE', 'BLOCH', 'DRAW', 'CIRCUIT', 'DECOMPOSE', 'DENSITY',
                           'HEATMAP', 'PLOT', 'ANIMATE', 'SWEEP', 'COMPARE')),
    ('Analysis', ('EXPECT', 'ENTROPY', 'FIDELITY', 'CONCURRENCE', 'NEGATIVITY', 'CONSISTENCY', 'TOMOGRAPHY',
                  'PTOMOGRAPHY', 'SAMPLE', 'ESTIMATE', 'PAULIPROP')),
    ('Algorithms', ('GROVER', 'SIMON', 'DJ', 'BV', 'QPE', 'QCOUNT', 'IQPE', 'AMPEST', 'SHOR', 'HHL', 'QAOA',
                    'VQE', 'MINIMIZE', 'GRADIENT', 'QWALK', 'QKERNEL')),
    ('Benchmarking', ('RB', 'RBINT', 'XEB', 'QVOLUME', 'MIRROR', 'GST')),
    ('Error correction', ('QEC', 'LOGICAL_ERROR_RATE', 'THRESHOLD', 'LQUBITS', 'LATTICE', 'DISTILL',
                          'RESOURCES')),
    ('Hamiltonians and open systems', ('HAMILTONIAN', 'CHANNEL', 'LINDBLAD')),
    ('LOCC', ('LOCC', 'LOCCINFO', 'SEND', 'SHARE')),
    ('Qudits and bosonic modes', ('QUDIT', 'QX', 'QZ', 'QF', 'QSUM', 'QSTATE', 'QMEASURE', 'BOSONIC',
                                  'DISPLACE', 'SQUEEZE', 'CAT', 'BS', 'BSTATE')),
    ('Memory map', ('PEEK', 'POKE', 'SYS', 'WAIT', 'DUMP', 'MAP', 'MONITOR', 'CATALOG')),
    ('Help and session', ('HELP', 'DEMO', 'QUIZ', 'PROBE', 'BYE', 'QUIT', 'EXIT')),
)

# (topic, keyword, syntax, description)
STATEMENTS: tuple[tuple[str, str, str, str], ...] = (
    # ── Program flow ─────────────────────────────────────────────────────
    ('Program flow', 'GOTO', 'GOTO line | label', 'jump to a line number or label'),
    ('Program flow', 'GOSUB', 'GOSUB line | label', 'call a subroutine; RETURN comes back to the next statement'),
    ('Program flow', 'RETURN', 'RETURN [line | label]', 'return from GOSUB, optionally to another line'),
    ('Program flow', 'ON', 'ON n GOTO | GOSUB target1, target2, ...', 'jump or call by the value of n (1 picks the first target)'),
    ('Program flow', 'IF', 'IF cond THEN stmts [ELSE stmts]', 'single-line conditional; THEN or ELSE may name a line'),
    ('Program flow', 'IF', 'IF cond THEN / ELSEIF cond THEN / ELSE / END IF', 'block conditional'),
    ('Program flow', 'FOR', 'FOR v = a TO b [STEP s] ... NEXT [v]', 'counted loop; zero-trip when the step cannot reach b'),
    ('Program flow', 'WHILE', 'WHILE cond ... WEND', 'pre-test loop'),
    ('Program flow', 'DO', 'DO [WHILE | UNTIL cond] ... LOOP [WHILE | UNTIL cond]', 'loop with the test at either end, or none'),
    ('Program flow', 'EXIT', 'EXIT FOR | WHILE | DO | SUB | FUNCTION', 'leave the innermost loop or procedure of that kind'),
    ('Program flow', 'SELECT', 'SELECT CASE expr / CASE v, a TO b, IS > v / CASE ELSE / END SELECT', 'multi-way branch on values, ranges and comparisons'),
    ('Program flow', 'END', 'END', 'end the program'),
    ('Program flow', 'STOP', 'STOP', 'break into the debugger; CONT resumes'),
    ('Program flow', 'SYSTEM', 'SYSTEM', 'end the program'),
    ('Program flow', 'SLEEP', 'SLEEP [seconds]', 'pause'),
    ('Program flow', 'REM', "REM text | ' text", 'comment'),
    ('Program flow', 'label', 'name:', 'a line label, the target of GOTO, GOSUB, RESTORE, RESUME and ON ERROR'),
    # ── Procedures ───────────────────────────────────────────────────────
    ('Procedures', 'SUB', 'SUB name [(params)] [STATIC] ... END SUB', 'procedure with its own variables; params by reference'),
    ('Procedures', 'FUNCTION', 'FUNCTION name [(params)] [STATIC] ... END FUNCTION', 'function returning the value assigned to its name'),
    ('Procedures', 'CALL', 'CALL name [(args)] | name args', 'call a SUB'),
    ('Procedures', 'DECLARE', 'DECLARE SUB | FUNCTION name (params)', 'accepted for QBASIC compatibility'),
    ('Procedures', 'LOCAL', 'LOCAL v, ...', 'zero names inside a procedure (QUBASIC extension)'),
    ('Procedures', 'STATIC', 'STATIC v, ...', 'keep a procedure variable between calls'),
    ('Procedures', 'SHARED', 'SHARED v, ...', "use one of the module's variables inside a procedure"),
    ('Procedures', 'DEF FN', 'DEF FNname[(params)] = expr', 'single-expression function'),
    ('Procedures', 'DEF', 'DEF NAME[(params)] = gates', 'gate macro (also DEF BEGIN NAME(params) ... DEF END)'),
    ('Procedures', 'APPLY_CIRCUIT', 'APPLY_CIRCUIT name [@offset]', 'apply a CIRCUIT_DEF macro, shifted by offset qubits'),
    # ── Variables and data ───────────────────────────────────────────────
    ('Variables and data', 'LET', '[LET] var = expr', 'assignment; var may be an element a(i) or a field r.x'),
    ('Variables and data', 'CONST', 'CONST name = expr [, ...]', 'named constant'),
    ('Variables and data', 'DIM', 'DIM [SHARED] a([lo TO] hi, ...) [AS type] | DIM v AS type', 'declare arrays and typed variables'),
    ('Variables and data', 'REDIM', 'REDIM [PRESERVE] a([lo TO] hi, ...)', 'resize an array; PRESERVE keeps its contents'),
    ('Variables and data', 'ERASE', 'ERASE a [, ...]', 'delete arrays'),
    ('Variables and data', 'SWAP', 'SWAP a, b', 'exchange two variables, elements or fields'),
    ('Variables and data', 'DEFINT', 'DEFINT | DEFLNG | DEFSNG | DEFDBL | DEFSTR A-Z', 'default type of names by first letter'),
    ('Variables and data', 'TYPE', 'TYPE Name / field AS type / END TYPE', 'record type: INTEGER, LONG, SINGLE, DOUBLE, STRING, STRING * n or a record'),
    ('Variables and data', 'OPTION', 'OPTION BASE 0 | 1 | OPTION ENDIAN BIG | LITTLE', 'array lower bound; bitstring display order'),
    ('Variables and data', 'DATA', 'DATA v1, v2, ...', 'values for READ (numbers, strings, |state> names)'),
    ('Variables and data', 'READ', 'READ var, ...', 'take the next DATA values'),
    ('Variables and data', 'RESTORE', 'RESTORE [line | label]', 'rewind READ to the first (or a given) DATA'),
    ('Variables and data', 'RANDOMIZE', 'RANDOMIZE [seed | TIMER]', 'reseed RND'),
    ('Variables and data', 'MID$', 'MID$(s$, start [, length]) = text$', 'overwrite part of a string'),
    ('Variables and data', 'COMMON', 'COMMON [SHARED] var, ...', 'variables kept across CHAIN (SHARED also opens them to procedures)'),
    # ── Input and output ─────────────────────────────────────────────────
    ('Input and output', 'PRINT', 'PRINT [items] [; | ,]', 'print; ; joins, , moves to the next 14-column zone (also ?)'),
    ('Input and output', 'PRINT USING', 'PRINT USING fmt$; items', 'formatted print (#, ., ,, +, -, **, $$, ^^^^, !, \\ \\, &)'),
    ('Input and output', 'PRINT STATE', 'PRINT STATE | QUBIT(q) | ENTANGLEMENT(a, b) | @REG', 'the quantum state at this line'),
    ('Input and output', 'INPUT', 'INPUT ["prompt" ; | ,] var, ...', 'read values; a bad entry asks again'),
    ('Input and output', 'LINE INPUT', 'LINE INPUT ["prompt";] s$', 'read a whole line'),
    ('Input and output', 'WRITE', 'WRITE items', 'print comma-separated values, strings quoted'),
    ('Input and output', 'GET', 'GET k$', 'read one key'),
    ('Input and output', 'LPRINT', 'LPRINT items', 'print to the log (stderr)'),
    ('Input and output', 'CLS', 'CLS', 'clear the screen (and the graphics page)'),
    ('Input and output', 'COLOR', 'COLOR [fg] [, bg]', 'text colours, or the drawing colour on a graphics SCREEN'),
    ('Input and output', 'LOCATE', 'LOCATE row, col', 'move the text cursor'),
    ('Input and output', 'WIDTH', 'WIDTH columns', 'accepted for QBASIC compatibility'),
    ('Input and output', 'BEEP', 'BEEP', 'bell, and an 800 Hz tone in the sound track'),
    # ── Files ────────────────────────────────────────────────────────────
    ('Files', 'OPEN', 'OPEN f$ FOR INPUT | OUTPUT | APPEND | RANDOM | BINARY [ACCESS a] [LOCK l] AS #n [LEN = r]', 'open a file (also OPEN mode$, #n, f$ [, r])'),
    ('Files', 'CLOSE', 'CLOSE [#n, ...]', 'close files (all when none named)'),
    ('Files', 'PRINT #', 'PRINT #n, items | PRINT #n, USING fmt$; items', 'print to a file'),
    ('Files', 'WRITE #', 'WRITE #n, items', 'write comma-separated values'),
    ('Files', 'INPUT #', 'INPUT #n, var, ...', 'read values from a file'),
    ('Files', 'LINE INPUT #', 'LINE INPUT #n, s$', 'read a line from a file'),
    ('Files', 'FIELD', 'FIELD #n, width AS s$, ...', 'lay string variables over a RANDOM record'),
    ('Files', 'LSET', 'LSET s$ = text$ | RSET s$ = text$', 'fill a FIELD variable, left or right justified'),
    ('Files', 'GET #', 'GET #n [, record] [, var]', 'read a RANDOM record or BINARY bytes'),
    ('Files', 'PUT #', 'PUT #n [, record] [, var]', 'write a RANDOM record or BINARY bytes'),
    ('Files', 'SEEK', 'SEEK #n, position', 'move to a record (RANDOM) or byte (BINARY)'),
    ('Files', 'KILL', 'KILL f$ | NAME old$ AS new$', 'delete or rename a file'),
    ('Files', 'MKDIR', 'MKDIR d$ | RMDIR d$ | CHDIR d$', 'make, remove or enter a directory'),
    ('Files', 'FILES', 'FILES [pattern$]', 'list files'),
    ('Files', 'RESET', 'RESET', 'close every file (RESET q resets a qubit)'),
    ('Files', 'LOCK', 'LOCK | UNLOCK [#]n[, record | [start] TO end]', 'accepted for an open file'),
    ('Files', 'BSAVE', 'BSAVE file$, offset, length | BLOAD file$[, offset]', 'save or load bytes of the DEF SEG segment (SCREEN 13 pixels at &HA000, the text screen at &HB800)'),
    # ── Graphics ─────────────────────────────────────────────────────────
    ('Graphics', 'SCREEN', 'SCREEN mode', '0 text; 1, 2, 7-13 graphics; 3-5 the quantum display modes'),
    ('Graphics', 'PSET', 'PSET [STEP] (x, y) [, colour] | PRESET ...', 'plot a point (PRESET in the background colour)'),
    ('Graphics', 'LINE', 'LINE [[STEP] (x1, y1)] - [STEP] (x2, y2) [, [colour] [, B | BF] [, style]]', 'line, box or filled box'),
    ('Graphics', 'CIRCLE', 'CIRCLE [STEP] (x, y), r [, [colour] [, [start] [, [end] [, aspect]]]]', 'circle, ellipse, arc or sector'),
    ('Graphics', 'PAINT', 'PAINT [STEP] (x, y) [, [colour] [, border]]', 'flood fill up to a border colour'),
    ('Graphics', 'DRAW', 'DRAW commands$', 'turtle drawing: U D L R E F G H M B N A TA C S P'),
    ('Graphics', 'VIEW', 'VIEW [[SCREEN] (x1, y1) - (x2, y2) [, [fill] [, border]]]', 'viewport'),
    ('Graphics', 'WINDOW', 'WINDOW [[SCREEN] (x1, y1) - (x2, y2)]', 'logical coordinates over the viewport'),
    ('Graphics', 'PALETTE', 'PALETTE [attribute, colour] | PALETTE USING a(start)', 'remap colour attributes'),
    ('Graphics', 'GET', 'GET [STEP] (x1, y1) - [STEP] (x2, y2), a', 'copy a block of the screen into an array'),
    ('Graphics', 'PUT', 'PUT [STEP] (x, y), a [, PSET | PRESET | AND | OR | XOR]', 'draw a block saved by GET'),
    ('Graphics', 'PCOPY', 'PCOPY source, destination', 'copy a video page'),
    # ── Sound ────────────────────────────────────────────────────────────
    ('Sound', 'PLAY', 'PLAY commands$', 'music macro language: notes, O, L, T, P, N, MN/ML/MS and MF/MB (foreground or background music)'),
    ('Sound', 'SOUND', 'SOUND frequency, ticks', 'a tone for ticks / 18.2 seconds'),
    # ── Events ───────────────────────────────────────────────────────────
    ('Events', 'ON TIMER', 'ON TIMER(seconds) GOSUB line | label', 'call a handler every interval while TIMER ON is in force'),
    ('Events', 'TIMER', 'TIMER ON | OFF | STOP', 'enable, disable or hold timer events'),
    ('Program', 'RUN', 'RUN [line | label]', 'start the program again with its variables, arrays, files and traps cleared'),
    ('Program', 'CLEAR', 'CLEAR [, , stack]', 'clear every variable and array, close the files and empty the GOSUB stack'),
    ('Program', 'SHELL', 'SHELL command$', 'run a command in the system shell and print its output (off in agent mode)'),
    ('Program', 'ENVIRON', 'ENVIRON "NAME=value"', 'set an environment variable (off in agent mode)'),
    ('Memory', 'DEF SEG', 'DEF SEG [= segment]', 'the segment PEEK, POKE, BSAVE and BLOAD address (&HA000 SCREEN 13 pixels, &HB800 the text screen, 0 the BIOS clock)'),
    ('Memory', 'OUT', 'OUT port, value', 'the VGA palette ports: &H3C8 picks an entry, three &H3C9 writes set its red, green, blue (0-63)'),
    ('Memory', 'WAIT', 'WAIT port, mask [, xor]', 'hold until (INP(port) XOR xor) AND mask is not 0; &H3DA bit 8 is the vertical retrace, 70 times a second'),
    ('Screen', 'VIEW PRINT', 'VIEW PRINT [top TO bottom]', 'the text rows that scroll'),
    ('Events', 'ON MEASURE', 'ON MEASURE GOSUB line', "call a handler when the run's first MEASURE executes"),
    ('Events', 'ON KEY', 'ON KEY(n) GOSUB line | label', 'call a handler when key n is pressed under KEY(n) ON: F1-F10 (1-10), the arrows (11-14), F11 and F12 (30, 31), or a key KEY 15-25 defines'),
    ('Events', 'ON PLAY', 'ON PLAY(n) GOSUB line | label', 'call a handler when the background music queue drops below n notes, under PLAY ON'),
    ('Events', 'PLAY ON', 'PLAY ON | OFF | STOP', 'arm, disarm or hold the ON PLAY trap'),
    ('Events', 'ON COM', 'ON COM(n) | PEN | STRIG(n) | UEVENT GOSUB target', 'accepted; those devices raise no events here'),
    ('Events', 'UEVENT', 'UEVENT ON | OFF | STOP', 'accepted; no user event source is attached'),
    ('Events', 'KEY', 'KEY(n) ON | OFF | STOP | KEY n, keys$ | KEY ON | OFF | LIST', 'arm, disarm or hold key trap n (0 for all); define key 15-25 from CHR$(shift flags) + CHR$(scan code); KEY ON, OFF and LIST are accepted'),
    ('Events', 'PEN', 'PEN | STRIG(n) | COM(n) ON | OFF | STOP', 'accepted for QBASIC compatibility'),
    # ── Errors ───────────────────────────────────────────────────────────
    ('Errors', 'ON ERROR', 'ON ERROR GOTO line | label | 0', 'trap runtime errors (0 turns trapping off)'),
    ('Errors', 'RESUME', 'RESUME [0 | NEXT | line | label]', 'leave an error handler: retry, skip, or jump'),
    ('Errors', 'ERROR', 'ERROR n', 'raise error n (ERR and ERL report it)'),
    ('Errors', 'ASSERT', 'ASSERT cond', 'stop with an error when cond is false'),
    # ── Quantum circuit ──────────────────────────────────────────────────
    ('Quantum circuit', 'gate', 'H q | RX theta, q | CX c, t | ...', 'a gate from the gate table (HELP <gate>)'),
    ('Quantum circuit', 'CTRL', 'CTRL gate [params,] controls..., target', 'controlled version of any gate'),
    ('Quantum circuit', 'INV', 'INV gate args', 'inverse of a gate'),
    ('Quantum circuit', 'UNITARY', 'UNITARY NAME = [[...], ...]', 'define a gate from a unitary matrix'),
    ('Quantum circuit', 'BARRIER', 'BARRIER', 'optimization barrier'),
    ('Quantum circuit', 'RESET', 'RESET q', 'reset a qubit to |0>'),
    ('Quantum circuit', 'MEASURE', 'MEASURE [q, ...]', 'measure every qubit (or a subset) into the histogram'),
    ('Quantum circuit', 'MEAS', 'MEAS q -> bit', 'mid-circuit measurement; IF bit THEN gates feeds forward, any other read runs the program live'),
    ('Quantum circuit', 'MEASURE_X', 'MEASURE_X | MEASURE_Y | MEASURE_Z q', 'measure in a Pauli basis (result in mx_q, my_q, mz_q)'),
    ('Quantum circuit', 'SYNDROME', 'SYNDROME paulis q... -> var', 'non-destructive stabilizer measurement through an ancilla'),
    ('Quantum circuit', 'SET_STATE', 'SET_STATE |name> | [amplitudes]', 'inject a state at this point'),
    ('Quantum circuit', 'QFT', 'QFT lo-hi | IQFT lo-hi', 'quantum Fourier transform and its inverse'),
    ('Quantum circuit', 'DIFFUSE', 'DIFFUSE lo-hi | DIFFUSION lo-hi', 'Grover diffusion operator'),
    ('Quantum circuit', 'MCX', 'MCX c1, c2, ..., t | MCZ q... | MCP theta, c..., t', 'multi-controlled X, Z and phase'),
    ('Quantum circuit', 'QADD', 'QADD a-range, b-range | QADDC k, range', 'in-place register addition (mod 2^n)'),
    ('Quantum circuit', 'QPE', 'QPE counting-range target UGATE', 'phase estimation of a UNITARY'),
    ('Quantum circuit', 'AMPLIFY', 'AMPLIFY bits', 'one amplitude-amplification step toward a basis state'),
    ('Quantum circuit', 'GRAPHSTATE', 'GRAPHSTATE a-b, b-c, ...', 'prepare a graph state'),
    ('Quantum circuit', 'FEATUREMAP', 'FEATUREMAP x0 x1 ...', 'ZZ feature-map encoding'),
    ('Quantum circuit', 'EVOLVE', 'EVOLVE H, time, steps', 'Trotterized exp(-iHt) of a HAMILTONIAN'),
    ('Quantum circuit', 'APPLYCHANNEL', 'APPLYCHANNEL name q...', 'apply a CHANNEL (Kraus operators)'),
    ('Quantum circuit', 'SAVE_EXPECT', 'SAVE_EXPECT obs q... -> var | SAVE_EXPECT H -> var', 'record an expectation value after RUN'),
    ('Quantum circuit', 'SAVE_PROBS', 'SAVE_PROBS q... -> array', 'record probabilities after RUN'),
    ('Quantum circuit', 'SAVE_AMPS', 'SAVE_AMPS i, j, ... -> array', 'record amplitudes after RUN'),
    # ── LOCC ─────────────────────────────────────────────────────────────
    ('LOCC', '@', '@REG gate args', 'a gate on one register of a LOCC session'),
    ('LOCC', 'SEND', 'SEND REG q -> bit', 'measure a qubit and send the bit to the other parties'),
    ('LOCC', 'SHARE', 'SHARE REG q, REG q', 'create a Bell pair across registers (JOINT mode)'),
    # ── Memory ───────────────────────────────────────────────────────────
    ('Memory', 'POKE', 'POKE address, value', 'write the memory map (qubit state, QPU settings, noise)'),
    ('Memory', 'SYS', 'SYS address', 'call a SYS routine'),
)

# (topic, name, syntax, description)
FUNCTIONS: tuple[tuple[str, str, str, str], ...] = (
    ('Math', 'ABS', 'ABS(x)', 'absolute value'),
    ('Math', 'SGN', 'SGN(x)', '-1, 0 or 1'),
    ('Math', 'INT', 'INT(x)', 'largest integer not above x'),
    ('Math', 'FIX', 'FIX(x)', 'x truncated toward zero'),
    ('Math', 'CINT', 'CINT(x) | CLNG(x)', 'round half to even into INTEGER or LONG range'),
    ('Math', 'CSNG', 'CSNG(x) | CDBL(x)', 'convert to single or double'),
    ('Math', 'ROUND', 'ROUND(x [, digits])', 'round half to even'),
    ('Math', 'CEIL', 'CEIL(x) | FLOOR(x)', 'round up or down'),
    ('Math', 'SQR', 'SQR(x) | SQRT(x)', 'square root'),
    ('Math', 'EXP', 'EXP(x) | LOG(x)', 'exponential and natural logarithm'),
    ('Math', 'SIN', 'SIN(x) | COS(x) | TAN(x)', 'trigonometry in radians'),
    ('Math', 'ATN', 'ATN(x) | ATAN(x) | ASIN(x) | ACOS(x) | ATAN2(y, x)', 'inverse trigonometry'),
    ('Math', 'MIN', 'MIN(a, b, ...) | MAX(a, b, ...)', 'smallest or largest argument'),
    ('Math', 'FLOAT', 'FLOAT(x)', 'x as a floating-point number'),
    ('Math', 'RND', 'RND [(n)]', 'random number in [0, 1); RND(0) repeats the last'),
    ('Strings', 'LEN', 'LEN(s$)', 'length'),
    ('Strings', 'LEFT$', 'LEFT$(s$, n) | RIGHT$(s$, n) | MID$(s$, start [, n])', 'substrings'),
    ('Strings', 'INSTR', 'INSTR([start,] s$, find$)', 'position of find$ in s$, or 0'),
    ('Strings', 'UCASE$', 'UCASE$(s$) | LCASE$(s$)', 'change case'),
    ('Strings', 'LTRIM$', 'LTRIM$(s$) | RTRIM$(s$)', 'strip leading or trailing spaces'),
    ('Strings', 'SPACE$', 'SPACE$(n) | STRING$(n, c)', 'repeated characters'),
    ('Strings', 'CHR$', 'CHR$(n) | ASC(s$)', 'character from code and back'),
    ('Strings', 'STR$', 'STR$(x) | VAL(s$)', 'number to string and back (VAL reads &H, &O)'),
    ('Strings', 'HEX$', 'HEX$(n) | OCT$(n) | BIN$(n)', 'number in base 16, 8 or 2'),
    ('Strings', 'MKI$', 'MKI$(n) | MKL$(n) | MKS$(x) | MKD$(x)', 'number as its 2, 4, 4 or 8 file bytes'),
    ('Strings', 'CVI', 'CVI(s$) | CVL(s$) | CVS(s$) | CVD(s$)', 'file bytes back to a number'),
    ('Arrays', 'LBOUND', 'LBOUND(a [, dim]) | UBOUND(a [, dim])', 'bounds of an array dimension'),
    ('Files', 'EOF', 'EOF(n)', 'true (-1) at the end of file n'),
    ('Files', 'LOF', 'LOF(n)', 'length of file n in bytes'),
    ('Files', 'LOC', 'LOC(n)', 'current record or byte position'),
    ('Files', 'SEEK', 'SEEK(n)', 'next record or byte position'),
    ('Files', 'INPUT$', 'INPUT$(count [, #n])', 'read count characters'),
    ('Files', 'FREEFILE', 'FREEFILE', 'lowest free file number'),
    ('Graphics', 'POINT', 'POINT(x, y) | POINT(n)', 'colour of a pixel, or a cursor coordinate'),
    ('Graphics', 'PMAP', 'PMAP(c, n)', 'convert between logical and physical coordinates'),
    ('System', 'TIMER', 'TIMER', 'seconds since midnight'),
    ('System', 'DATE$', 'DATE$ | TIME$', 'today as mm-dd-yyyy, now as hh:mm:ss'),
    ('System', 'INKEY$', 'INKEY$', 'the key waiting, or ""'),
    ('System', 'PEEK', 'PEEK(address)', 'read the memory map'),
    ('System', 'USR', 'USR(address)', 'call a SYS routine for its value'),
    ('System', 'FRE', 'FRE(n)', 'free memory in bytes'),
    ('System', 'POS', 'POS(n)', 'text cursor column'),
    ('System', 'ERR', 'ERR | ERL', 'code and line of the trapped error'),
    ('System', 'FN', 'FNname(args)', 'call a DEF FN function'),
    ('System', 'CSRLIN', 'CSRLIN', 'text cursor row'),
    ('System', 'SCREEN', 'SCREEN(row, column [, 1])', 'the character code at a text position, or with 1 its colour'),
    ('System', 'PLAY', 'PLAY(n)', 'notes left in the background music queue'),
    ('System', 'ENVIRON$', 'ENVIRON$(name$ | n)', 'an environment variable, or the nth NAME=value entry ("" in agent mode)'),
    ('System', 'FILEATTR', 'FILEATTR(n, 1 | 2)', "file n's mode (1 INPUT, 2 OUTPUT, 4 RANDOM, 8 APPEND, 32 BINARY) or handle"),
    ('System', 'LPOS', 'LPOS(n)', 'printer column'),
    ('System', 'INP', 'INP(port)', 'the VGA status (&H3DA) and palette data (&H3C9) ports, the keyboard port (&H60); other ports read 0'),
    ('System', 'ERDEV', 'ERDEV | ERDEV$', 'the last device error and its device (none occur)'),
    ('System', 'STICK', 'STICK(n) | STRIG(n) | PEN(n) | IOCTL$(n)', '0 or "": no joystick, light pen or device channel is attached'),
    ('System', 'VARPTR', 'VARPTR(v) | VARSEG(v) | SADD(s$) | VARPTR$(v)', 'an error: variables have no memory address'),
    ('Conversion', 'MKSMBF$', 'MKSMBF$(x) | MKDMBF$(x)', 'a number in Microsoft Binary Format, 4 or 8 characters'),
    ('Conversion', 'CVSMBF', 'CVSMBF(s$) | CVDMBF(s$)', 'the number a Microsoft Binary Format string holds'),
)


def statements_for(keyword: str) -> list[tuple[str, str, str, str]]:
    """Catalog statements whose keyword is `keyword` or begins with it as a
    word (case-insensitive): PRINT finds PRINT, PRINT USING and PRINT #."""
    up = ' '.join(keyword.split()).upper()
    return [e for e in STATEMENTS if e[1].upper() == up or e[1].upper().split()[0] == up]


def functions_for(name: str) -> list[tuple[str, str, str, str]]:
    """Catalog functions named `name`, directly or among a syntax's |-joined forms."""
    up = name.strip().upper()
    found = []
    for entry in FUNCTIONS:
        forms = [f.strip().split('(')[0].split()[0].upper() for f in entry[2].split('|') if f.strip()]
        if entry[1].upper() == up or up in forms:
            found.append(entry)
    return found
