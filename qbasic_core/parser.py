"""QBASIC parser — converts raw statement strings into typed Stmt objects."""

from __future__ import annotations

import re

from qbasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    RE_LET_ARRAY, RE_LET_VAR, RE_PRINT,
    RE_GOTO, RE_GOSUB, RE_FOR, RE_NEXT, RE_WHILE, RE_IF_THEN,
    RE_MEAS, RE_RESET, RE_SEND, RE_SHARE, RE_AT_REG_LINE, RE_AT_REG,
)
from qbasic_core.statements import (
    Stmt, RawStmt, RemStmt, MeasureStmt, EndStmt, ReturnStmt,
    BarrierStmt, WendStmt,
    GotoStmt, GosubStmt, ForStmt, NextStmt, WhileStmt, IfThenStmt,
    LetStmt, LetArrayStmt, PrintStmt,
    GateStmt, MeasStmt, ResetStmt,
    SendStmt, ShareStmt, AtRegStmt,
    CompoundStmt,
)


def _split_colon_stmts(stmt: str) -> list[str]:
    """Split colon-separated statements, inheriting @register prefixes."""
    parts = []
    last_reg = None
    for sub in stmt.split(':'):
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


def parse_stmt(raw: str) -> Stmt:
    """Parse a raw statement string into a typed Stmt.

    Returns RawStmt for anything not yet covered, so the legacy
    regex-on-the-fly path can handle it.
    """
    text = raw.strip()
    if not text:
        return RawStmt(raw=raw)

    upper = text.upper()

    # ── Terminals ──────────────────────────────────────────────────
    if upper.startswith('REM') or upper.startswith("'"):
        return RemStmt(raw=raw)
    if upper == 'MEASURE':
        return MeasureStmt(raw=raw)
    if upper == 'END':
        return EndStmt(raw=raw)
    if upper == 'RETURN':
        return ReturnStmt(raw=raw)
    if upper == 'BARRIER':
        return BarrierStmt(raw=raw)
    if upper == 'WEND':
        return WendStmt(raw=raw)

    # ── Control flow ──────────────────────────────────────────────
    m = RE_GOTO.match(text)
    if m:
        return GotoStmt(raw=raw, target=int(m.group(1)))

    m = RE_GOSUB.match(text)
    if m:
        return GosubStmt(raw=raw, target=int(m.group(1)))

    m = RE_FOR.match(text)
    if m:
        return ForStmt(raw=raw, var=m.group(1),
                       start_expr=m.group(2), end_expr=m.group(3),
                       step_expr=m.group(4))

    m = RE_NEXT.match(text)
    if m:
        return NextStmt(raw=raw, var=m.group(1))

    m = RE_WHILE.match(text)
    if m:
        return WhileStmt(raw=raw, condition=m.group(1).strip())

    m = RE_IF_THEN.match(text)
    if m:
        return IfThenStmt(raw=raw, condition=m.group(1).strip(),
                          then_clause=m.group(2).strip(),
                          else_clause=m.group(3).strip() if m.group(3) else None)

    # ── Assignment ────────────────────────────────────────────────
    m = RE_LET_ARRAY.match(text)
    if m:
        return LetArrayStmt(raw=raw, name=m.group(1),
                            index_expr=m.group(2), value_expr=m.group(3))

    m = RE_LET_VAR.match(text)
    if m:
        return LetStmt(raw=raw, name=m.group(1), expr=m.group(2))

    # ── I/O ───────────────────────────────────────────────────────
    m = RE_PRINT.match(text)
    if m:
        return PrintStmt(raw=raw, expr=m.group(1).strip())

    # ── Quantum ───────────────────────────────────────────────────
    m = RE_MEAS.match(text)
    if m:
        return MeasStmt(raw=raw, qubit_expr=m.group(1), var=m.group(2))

    m = RE_RESET.match(text)
    if m:
        return ResetStmt(raw=raw, qubit_expr=m.group(1))

    # ── LOCC ──────────────────────────────────────────────────────
    m = RE_SEND.match(text)
    if m:
        return SendStmt(raw=raw, reg=m.group(1).upper(),
                        qubit_expr=m.group(2), var=m.group(3))

    m = RE_SHARE.match(text)
    if m:
        return ShareStmt(raw=raw, reg1=m.group(1).upper(), q1=int(m.group(2)),
                         reg2=m.group(3).upper(), q2=int(m.group(4)))

    m = RE_AT_REG_LINE.match(text)
    if m:
        return AtRegStmt(raw=raw, reg=m.group(1).upper(),
                         inner=m.group(2).strip())

    # ── Compound (colon-separated) ────────────────────────────────
    if ':' in text:
        parts = _split_colon_stmts(text)
        if len(parts) > 1:
            return CompoundStmt(raw=raw, parts=tuple(parts))

    # ── Fallback ──────────────────────────────────────────────────
    return RawStmt(raw=raw)
