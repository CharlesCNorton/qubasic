"""QBASIC parser — converts raw statement strings into typed Stmt objects."""

from __future__ import annotations

import re

from qbasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    RE_LET_ARRAY, RE_LET_VAR, RE_PRINT,
    RE_GOTO, RE_GOSUB, RE_FOR, RE_NEXT, RE_WHILE, RE_IF_THEN,
    RE_MEAS, RE_RESET, RE_SEND, RE_SHARE, RE_AT_REG_LINE, RE_AT_REG,
    RE_DATA, RE_READ, RE_ON_GOTO, RE_ON_GOSUB,
    RE_SELECT_CASE, RE_CASE, RE_DO, RE_LOOP_STMT, RE_EXIT,
    RE_SUB, RE_END_SUB, RE_FUNCTION, RE_END_FUNCTION, RE_CALL,
    RE_LOCAL, RE_STATIC_DECL, RE_SHARED,
    RE_ON_ERROR, RE_RESUME, RE_ERROR_STMT, RE_ASSERT,
    RE_SWAP, RE_DEF_FN, RE_OPTION_BASE,
    RE_POKE, RE_SYS, RE_UNITARY, RE_DIM, RE_REDIM, RE_ERASE,
    RE_GET, RE_INPUT, RE_LINE_INPUT,
    RE_PRINT_USING, RE_OPEN, RE_CLOSE, RE_PRINT_FILE, RE_INPUT_FILE,
    RE_LPRINT, RE_SCREEN, RE_COLOR, RE_LOCATE,
    RE_ON_MEASURE, RE_ON_TIMER, RE_IMPORT, RE_CHAIN, RE_MERGE,
    RE_LET_STR, RE_DIM_MULTI, RE_MEASURE_BASIS, RE_SYNDROME,
)
from qbasic_core.statements import (
    Stmt, RawStmt, GateStmt, RemStmt, MeasureStmt, EndStmt, ReturnStmt,
    BarrierStmt, WendStmt, RestoreStmt, EndSelectStmt, EndSubStmt,
    EndFunctionStmt, StopStmt,
    GotoStmt, GosubStmt, ForStmt, NextStmt, WhileStmt, IfThenStmt,
    DoStmt, LoopStmt, ExitStmt,
    OnGotoStmt, OnGosubStmt, SelectCaseStmt, CaseStmt,
    CallStmt, SubStmt, FunctionStmt,
    OnErrorStmt, ResumeStmt, ErrorStmt, AssertStmt,
    SwapStmt, DefFnStmt, OptionBaseStmt,
    OnMeasureStmt, OnTimerStmt, DataStmt, ReadStmt,
    LocalStmt, StaticStmt, SharedStmt,
    LetStmt, LetArrayStmt, LetStrStmt, PrintStmt, PrintUsingStmt,
    InputStmt, LineInputStmt, GetStmt,
    DimStmt, RedimStmt, EraseStmt,
    PokeStmt, SysStmt, UnitaryStmt,
    OpenStmt, CloseStmt, PrintFileStmt, InputFileStmt, LprintStmt,
    ScreenStmt, ColorStmt, LocateStmt, ImportStmt, ChainStmt, MergeStmt,
    MeasStmt, ResetStmt, MeasureBasisStmt, SyndromeStmt,
    SendStmt, ShareStmt, AtRegStmt,
    CompoundStmt,
)


# ═══════════════════════════════════════════════════════════════════════
# Keyword dispatch handlers
# ═══════════════════════════════════════════════════════════════════════

def _handle_goto(text, raw):
    m = RE_GOTO.match(text)
    if m:
        return GotoStmt(raw=raw, target=int(m.group(1)))
    return None

def _handle_gosub(text, raw):
    m = RE_GOSUB.match(text)
    if m:
        return GosubStmt(raw=raw, target=int(m.group(1)))
    return None

def _handle_for(text, raw):
    m = RE_FOR.match(text)
    if m:
        return ForStmt(raw=raw, var=m.group(1),
                       start_expr=m.group(2), end_expr=m.group(3),
                       step_expr=m.group(4))
    return None

def _handle_next(text, raw):
    m = RE_NEXT.match(text)
    if m:
        return NextStmt(raw=raw, var=m.group(1))
    return None

def _handle_while(text, raw):
    m = RE_WHILE.match(text)
    if m:
        return WhileStmt(raw=raw, condition=m.group(1).strip())
    return None

def _handle_if(text, raw):
    m = RE_IF_THEN.match(text)
    if m:
        return IfThenStmt(raw=raw, condition=m.group(1).strip(),
                          then_clause=m.group(2).strip(),
                          else_clause=m.group(3).strip() if m.group(3) else None)
    return None

def _handle_do(text, raw):
    m = RE_DO.match(text)
    if m:
        return DoStmt(raw=raw, kind=m.group(1), condition=m.group(2))
    return None

def _handle_loop(text, raw):
    m = RE_LOOP_STMT.match(text)
    if m:
        return LoopStmt(raw=raw, kind=m.group(1), condition=m.group(2))
    return None

def _handle_exit(text, raw):
    m = RE_EXIT.match(text)
    if m:
        return ExitStmt(raw=raw, target=m.group(1).upper())
    return None

def _handle_on(text, raw):
    m = RE_ON_MEASURE.match(text)
    if m:
        return OnMeasureStmt(raw=raw, target=int(m.group(1)))
    m = RE_ON_TIMER.match(text)
    if m:
        return OnTimerStmt(raw=raw, interval=m.group(1), target=int(m.group(2)))
    m = RE_ON_ERROR.match(text)
    if m:
        return OnErrorStmt(raw=raw, target=int(m.group(1)))
    m = RE_ON_GOTO.match(text)
    if m:
        targets = tuple(int(t.strip()) for t in m.group(2).split(',') if t.strip())
        return OnGotoStmt(raw=raw, expr=m.group(1).strip(), targets=targets)
    m = RE_ON_GOSUB.match(text)
    if m:
        targets = tuple(int(t.strip()) for t in m.group(2).split(',') if t.strip())
        return OnGosubStmt(raw=raw, expr=m.group(1).strip(), targets=targets)
    return None

def _handle_select(text, raw):
    m = RE_SELECT_CASE.match(text)
    if m:
        return SelectCaseStmt(raw=raw, expr=m.group(1).strip())
    return None

def _handle_case(text, raw):
    m = RE_CASE.match(text)
    if m:
        return CaseStmt(raw=raw, value=m.group(1).strip())
    return None

def _handle_sub(text, raw):
    m = RE_SUB.match(text)
    if m:
        return SubStmt(raw=raw, name=m.group(1).upper(),
                       params=m.group(2) or '')
    return None

def _handle_function(text, raw):
    m = RE_FUNCTION.match(text)
    if m:
        return FunctionStmt(raw=raw, name=m.group(1).upper(),
                            params=m.group(2) or '')
    return None

def _handle_call(text, raw):
    m = RE_CALL.match(text)
    if m:
        return CallStmt(raw=raw, name=m.group(1).upper(),
                        args=m.group(2) or '')
    return None

def _handle_local(text, raw):
    m = RE_LOCAL.match(text)
    if m:
        return LocalStmt(raw=raw, var_list=m.group(1))
    return None

def _handle_static(text, raw):
    m = RE_STATIC_DECL.match(text)
    if m:
        return StaticStmt(raw=raw, var_list=m.group(1))
    return None

def _handle_shared(text, raw):
    m = RE_SHARED.match(text)
    if m:
        return SharedStmt(raw=raw, var_list=m.group(1))
    return None

def _handle_resume(text, raw):
    m = RE_RESUME.match(text)
    if m:
        return ResumeStmt(raw=raw, arg=m.group(1))
    return None

def _handle_error(text, raw):
    m = RE_ERROR_STMT.match(text)
    if m:
        return ErrorStmt(raw=raw, code=int(m.group(1)))
    return None

def _handle_assert(text, raw):
    m = RE_ASSERT.match(text)
    if m:
        return AssertStmt(raw=raw, condition=m.group(1).strip())
    return None

def _handle_swap(text, raw):
    m = RE_SWAP.match(text)
    if m:
        return SwapStmt(raw=raw, a=m.group(1), b=m.group(2))
    return None

def _handle_def(text, raw):
    m = RE_DEF_FN.match(text)
    if m:
        return DefFnStmt(raw=raw, name=m.group(1),
                         params=m.group(2), body=m.group(3))
    return None

def _handle_option(text, raw):
    m = RE_OPTION_BASE.match(text)
    if m:
        return OptionBaseStmt(raw=raw, base=int(m.group(1)))
    return None

def _handle_data(text, raw):
    m = RE_DATA.match(text)
    if m:
        return DataStmt(raw=raw, values=m.group(1))
    return None

def _handle_read(text, raw):
    m = RE_READ.match(text)
    if m:
        return ReadStmt(raw=raw, var_list=m.group(1))
    return None

def _handle_let(text, raw):
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

def _handle_print(text, raw):
    m = RE_PRINT_USING.match(text)
    if m:
        return PrintUsingStmt(raw=raw, fmt=m.group(1), values=m.group(2))
    m = RE_PRINT_FILE.match(text)
    if m:
        return PrintFileStmt(raw=raw, handle=int(m.group(1)), data=m.group(2))
    m = RE_PRINT.match(text)
    if m:
        return PrintStmt(raw=raw, expr=m.group(1).strip())
    return None

def _handle_line(text, raw):
    m = RE_LINE_INPUT.match(text)
    if m:
        return LineInputStmt(raw=raw, prompt=m.group(1), var=m.group(2))
    return None

def _handle_get(text, raw):
    m = RE_GET.match(text)
    if m:
        return GetStmt(raw=raw, var=m.group(1))
    return None

def _handle_input(text, raw):
    m = RE_INPUT_FILE.match(text)
    if m:
        return InputFileStmt(raw=raw, handle=int(m.group(1)), var=m.group(2))
    m = RE_INPUT.match(text)
    if m:
        return InputStmt(raw=raw, prompt=m.group(1), var=m.group(2))
    return None

def _handle_lprint(text, raw):
    m = RE_LPRINT.match(text)
    if m:
        return LprintStmt(raw=raw, expr=m.group(1))
    return None

def _handle_poke(text, raw):
    m = RE_POKE.match(text)
    if m:
        return PokeStmt(raw=raw, addr_expr=m.group(1), value_expr=m.group(2))
    return None

def _handle_sys(text, raw):
    m = RE_SYS.match(text)
    if m:
        return SysStmt(raw=raw, arg=m.group(1))
    return None

def _handle_unitary(text, raw):
    m = RE_UNITARY.match(text)
    if m:
        return UnitaryStmt(raw=raw, name=m.group(1), matrix=m.group(2))
    return None

def _handle_redim(text, raw):
    m = RE_REDIM.match(text)
    if m:
        return RedimStmt(raw=raw, name=m.group(1), size=m.group(2))
    return None

def _handle_erase(text, raw):
    m = RE_ERASE.match(text)
    if m:
        return EraseStmt(raw=raw, name=m.group(1))
    return None

def _handle_dim(text, raw):
    m = RE_DIM_MULTI.match(text)
    if m:
        return DimStmt(raw=raw, name=m.group(1), size=m.group(2))
    m = RE_DIM.match(text)
    if m:
        return DimStmt(raw=raw, name=m.group(1), size=m.group(2))
    return None

def _handle_open(text, raw):
    m = RE_OPEN.match(text)
    if m:
        return OpenStmt(raw=raw, path=m.group(1).strip(),
                        mode=m.group(2).upper(), handle=int(m.group(3)),
                        encoding=m.group(4).strip() if m.group(4) else None)
    return None

def _handle_close(text, raw):
    m = RE_CLOSE.match(text)
    if m:
        return CloseStmt(raw=raw, handle=int(m.group(1)))
    return None

def _handle_import(text, raw):
    m = RE_IMPORT.match(text)
    if m:
        return ImportStmt(raw=raw, path=m.group(1).strip())
    return None

def _handle_chain(text, raw):
    m = RE_CHAIN.match(text)
    if m:
        return ChainStmt(raw=raw, path=m.group(1).strip())
    return None

def _handle_merge(text, raw):
    m = RE_MERGE.match(text)
    if m:
        return MergeStmt(raw=raw, path=m.group(1).strip())
    return None

def _handle_screen(text, raw):
    m = RE_SCREEN.match(text)
    if m:
        return ScreenStmt(raw=raw, mode=m.group(1))
    return None

def _handle_color(text, raw):
    m = RE_COLOR.match(text)
    if m:
        return ColorStmt(raw=raw, fg=m.group(1), bg=m.group(2))
    return None

def _handle_locate(text, raw):
    m = RE_LOCATE.match(text)
    if m:
        return LocateStmt(raw=raw, row=m.group(1), col=m.group(2))
    return None

def _handle_measure_basis(text, raw):
    m = RE_MEASURE_BASIS.match(text)
    if m:
        return MeasureBasisStmt(raw=raw, basis=m.group(1).upper(),
                                qubit_expr=m.group(2))
    return None

def _handle_syndrome(text, raw):
    m = RE_SYNDROME.match(text)
    if m:
        return SyndromeStmt(raw=raw, rest=m.group(1).strip())
    return None

def _handle_meas(text, raw):
    m = RE_MEAS.match(text)
    if m:
        return MeasStmt(raw=raw, qubit_expr=m.group(1), var=m.group(2))
    return None

def _handle_reset(text, raw):
    m = RE_RESET.match(text)
    if m:
        return ResetStmt(raw=raw, qubit_expr=m.group(1))
    return None

def _handle_send(text, raw):
    m = RE_SEND.match(text)
    if m:
        return SendStmt(raw=raw, reg=m.group(1).upper(),
                        qubit_expr=m.group(2), var=m.group(3))
    return None

def _handle_share(text, raw):
    m = RE_SHARE.match(text)
    if m:
        return ShareStmt(raw=raw, reg1=m.group(1).upper(), q1=int(m.group(2)),
                         reg2=m.group(3).upper(), q2=int(m.group(4)))
    return None


# ═══════════════════════════════════════════════════════════════════════
# First-word dispatch table
# ═══════════════════════════════════════════════════════════════════════

_KEYWORD_PARSERS = {
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

    Returns RawStmt only for gate applications and truly unrecognized input.
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
    if upper == 'RESTORE':
        return RestoreStmt(raw=raw)
    if upper == 'END SELECT':
        return EndSelectStmt(raw=raw)
    if upper == 'STOP':
        return StopStmt(raw=raw)

    # ── END SUB / END FUNCTION ────────────────────────────────────
    m = RE_END_SUB.match(text)
    if m:
        return EndSubStmt(raw=raw)
    m = RE_END_FUNCTION.match(text)
    if m:
        return EndFunctionStmt(raw=raw)

    # ── First-word dispatch ───────────────────────────────────────
    first_word = text.split(None, 1)[0].upper()
    handler = _KEYWORD_PARSERS.get(first_word)
    if handler is not None:
        result = handler(text, raw)
        if result is not None:
            return result

    # ── @REG lines ────────────────────────────────────────────────
    if text.startswith('@'):
        m = RE_AT_REG_LINE.match(text)
        if m:
            return AtRegStmt(raw=raw, reg=m.group(1).upper(),
                             inner=m.group(2).strip())

    # ── Compound (colon-separated) ────────────────────────────────
    if ':' in text:
        parts = _split_colon_stmts(text)
        if len(parts) > 1:
            return CompoundStmt(raw=raw, parts=tuple(parts))

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
