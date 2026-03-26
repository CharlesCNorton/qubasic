"""QBASIC typed statement AST — produced by the parser, consumed by the executor."""

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
    pass


@dataclass(frozen=True, slots=True)
class EndStmt(Stmt):
    pass


@dataclass(frozen=True, slots=True)
class ReturnStmt(Stmt):
    pass


@dataclass(frozen=True, slots=True)
class BarrierStmt(Stmt):
    pass


@dataclass(frozen=True, slots=True)
class WendStmt(Stmt):
    pass


# ── Control flow ───────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GotoStmt(Stmt):
    target: int


@dataclass(frozen=True, slots=True)
class GosubStmt(Stmt):
    target: int


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


# ── I/O ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PrintStmt(Stmt):
    expr: str


# ── Quantum ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GateStmt(Stmt):
    gate: str
    params: tuple[str, ...]
    qubits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MeasStmt(Stmt):
    qubit_expr: str
    var: str


@dataclass(frozen=True, slots=True)
class ResetStmt(Stmt):
    qubit_expr: str


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


# ── Fallback ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RawStmt(Stmt):
    """Unrecognized statement — falls through to legacy regex path."""
    pass
