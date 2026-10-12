"""QUBASIC typed statement AST — produced by the parser, consumed by the executor."""

from __future__ import annotations

from dataclasses import dataclass


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
    target: int

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
    path: str
    mode: str
    handle: int
    encoding: str | None = None

@dataclass(frozen=True, slots=True)
class CloseStmt(Stmt):
    handle: int

@dataclass(frozen=True, slots=True)
class PrintFileStmt(Stmt):
    handle: int
    data: str

@dataclass(frozen=True, slots=True)
class InputFileStmt(Stmt):
    handle: int
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


# ── Compiled control transfers (produced by qubasic_core.program) ──────

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
