"""QUBASIC program compilation: numbered lines into a flat statement list.

Every statement of every line gets its own index, so control transfers are
plain indices: GOSUB returns to the next statement, a loop's body starts at
the statement after its opener, and a single-line IF becomes a conditional
jump over its THEN statements (and an unconditional jump over its ELSE
statements). Block structure (FOR/NEXT, WHILE/WEND, DO/LOOP, block
IF/ELSEIF/ELSE/END IF, SELECT CASE, SUB/FUNCTION, EXIT) is matched here
once, so a skip or an EXIT is a stored index, and mismatched blocks and
missing GOTO/GOSUB targets are reported before anything runs.

A line may start with a label (`Start:`), and every jump that takes a line
number also takes a label: GOTO, GOSUB, ON ... GOTO/GOSUB, ON ERROR GOTO,
RESUME, RETURN and RESTORE. A statement that starts with the name of a SUB
calls it, as `CALL name(args)` does. A TYPE ... END TYPE block declares a
record type (`prog.types`); its lines do not execute.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from typing import Any, Iterator

from qubasic_core.parser import parse_stmt, _split_colon_stmts
from qubasic_core.statements import (
    Stmt, CompoundStmt, IfThenStmt, IfJumpStmt, JumpStmt, NextStmt, ForStmt,
    WhileStmt, WendStmt, DoStmt, LoopStmt, ExitStmt, ElseStmt, EndIfStmt,
    SelectCaseStmt, CaseStmt, EndSelectStmt, SubStmt, EndSubStmt,
    FunctionStmt, EndFunctionStmt, GotoStmt, GosubStmt, OnGotoStmt, OnGosubStmt,
    OnErrorStmt, ResumeStmt, ReturnStmt, RestoreStmt, RemStmt, RawStmt, CallStmt,
)

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


class CompiledProgram(Sequence):
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

    def __getitem__(self, ip):                 # type: ignore[override]
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
        text = program[line].strip()
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
               open_type: tuple[str, int, list[tuple[str, str]]] | None):
    """One line of a TYPE block (the header, a field or END TYPE); returns
    the block still open, or None after END TYPE."""
    from qubasic_core.variables import TYPE_NAMES, type_text
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
        if isinstance(st, (GotoStmt, GosubStmt, OnErrorStmt)) and isinstance(st.target, str):
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
