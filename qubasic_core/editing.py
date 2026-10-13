"""QUBASIC program editing: LIST, NEW, DELETE, RENUM, DEF, TYPE and the
other definition commands, AUTO, EDIT, banks, FIND/REPLACE, CHAIN and MERGE,
and SAVE, LOAD, INCLUDE and IMPORT.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from qubasic_core.engine import MAX_INCLUDE_DEPTH
from qubasic_core.expression import Record
from qubasic_core.host import LanguageHost
from qubasic_core.parser import GATE_TABLE, RE_CHAIN, RE_DEF_BEGIN, RE_DEF_SINGLE, RE_MERGE, parse_stmt

# ---------------------------------------------------------------------------
# QUBASIC program editing commands: LIST, NEW, DELETE, RENUM, DEF, TYPE, CIRCUIT_DEF, REG, LET, UNDO.
# ---------------------------------------------------------------------------

# Line-number references RENUM rewrites: the keyword (group 1) and the
# number or comma list of numbers after it (group 2).
_RENUM_REFS = (
    re.compile(r'(\b(?:GOTO|GOSUB)\s+)(\d+(?:\s*,\s*\d+)*)', re.IGNORECASE),
    re.compile(r'(\b(?:RETURN|RESTORE|RESUME)\s+)(\d+)\b', re.IGNORECASE),
    re.compile(r'(\bTHEN\s+)(\d+)(?=\s*(?:$|ELSE\b|:))', re.IGNORECASE),
    re.compile(r'(\bELSE\s+)(\d+)(?=\s*(?:$|:))', re.IGNORECASE),
)


def renumber_refs(text: str, line_map: dict[int, int]) -> str:
    """`text` with every line-number reference mapped through line_map,
    outside string literals; a comment line is left as it is."""
    stripped = text.lstrip()
    if stripped.upper().startswith('REM') or stripped.startswith("'"):
        return text

    def remap(m: re.Match) -> str:
        nums = re.sub(r'\d+', lambda d: str(line_map.get(int(d.group()), int(d.group()))),
                      m.group(2))
        return m.group(1) + nums

    pieces = text.split('"')
    for i in range(0, len(pieces), 2):
        for pattern in _RENUM_REFS:
            pieces[i] = pattern.sub(remap, pieces[i])
    return '"'.join(pieces)


class EditingMixin(LanguageHost):
    """Program editing and definition commands for QBasicTerminal."""

    def cmd_list(self, rest: str = '') -> None:
        """LIST [SUBS | VARS | ARRAYS] — show the program, or its procedures, variables or arrays."""
        arg = rest.strip().upper()
        if arg == 'SUBS':
            return self.cmd_list_subs()
        if arg == 'VARS':
            return self.cmd_list_vars()
        if arg == 'ARRAYS':
            return self.cmd_list_arrays()
        if not self.program:
            self.io.writeln("EMPTY PROGRAM")
            return
        if self._theme_name != 'none' and sys.stdout.isatty():
            return self.cmd_list_colored()
        for num in sorted(self.program.keys()):
            self.io.writeln(f"  {num:5d}  {self.program[num]}")

    def cmd_new(self, *, silent: bool = False) -> None:
        """NEW — clear program, subroutines, registers, and variables."""
        self.clear()
        if not silent:
            self.io.writeln("READY")

    def cmd_delete(self, rest: str) -> None:
        """DELETE line | start-end — remove program lines."""
        if not rest:
            self.io.writeln("?USAGE: DELETE <line> or DELETE <start>-<end>")
            return
        if '-' in rest:
            lo_s, hi_s = rest.split('-')
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
            for k in list(self.program.keys()):
                if lo <= k <= hi:
                    del self.program[k]
                    self._parsed.pop(k, None)
            self.io.writeln(f"DELETED {lo}-{hi}")
        else:
            n = int(rest)
            if n in self.program:
                del self.program[n]
                self._parsed.pop(n, None)
                self.io.writeln(f"DELETED {n}")
            else:
                self.io.writeln(f"?LINE {n} NOT FOUND")

    def cmd_renum(self, rest: str = '') -> None:
        """RENUM [start] [step] — renumber lines and every line-number reference
        (GOTO, GOSUB, ON ... GOTO/GOSUB lists, ON ERROR GOTO, RESUME, RETURN,
        RESTORE, and IF ... THEN/ELSE <line>); string literals and comments keep
        their text."""
        if not self.program:
            return
        parts = rest.split() if rest else []
        start = int(parts[0]) if len(parts) > 0 else 10
        step = int(parts[1]) if len(parts) > 1 else 10
        old_lines = sorted(self.program.keys())
        line_map = {old: start + i * step for i, old in enumerate(old_lines)}
        new_prog = {line_map[old]: renumber_refs(self.program[old], line_map)
                    for old in old_lines}
        self.program = new_prog
        self._parsed = {num: parse_stmt(s) for num, s in new_prog.items()}
        # Move breakpoints to their renumbered lines.
        if self.debug.breakpoints:
            self.debug.breakpoints = {line_map.get(b, b) for b in self.debug.breakpoints}
        self.io.writeln(f"RENUMBERED {len(new_prog)} LINES (start={start}, step={step})")
        # Computed jumps (ON <var> GOTO via runtime values) can't be remapped statically.
        if any(re.search(r'\bON\b.*\bGO(TO|SUB)\b', s, re.IGNORECASE)
               for s in new_prog.values()):
            self.io.writeln("  (note: verify any computed ON..GOTO/GOSUB targets)")

    def cmd_def(self, rest: str) -> None:
        """DEF NAME[(params)] = gate : gate ... — define a named gate sequence (also DEF BEGIN ... DEF END blocks and DEF FN functions)."""
        # Single-line: DEF BELL = H 0 : CX 0,1
        # Parameterized: DEF ROTATE(angle, q) = RX angle, q : RZ angle, q
        # Multi-line: DEF BEGIN NAME[(params)] ... DEF END
        upper = rest.upper().strip()
        if upper.startswith('BEGIN'):
            return self._def_multiline(rest[5:].strip())

        # DEF FN name(params) = expr — a user expression function, same as the
        # in-program form, so it works at the prompt too.
        if upper.startswith('FN'):
            from qubasic_core.parser import RE_DEF_FN
            fm = RE_DEF_FN.match(f"DEF {rest}")
            if fm:
                fparams = [p.strip() for p in fm.group(2).split(',') if p.strip()]
                self.symbols.fns['FN' + fm.group(1).upper()] = {
                    'params': fparams, 'body': fm.group(3).strip()}
                self.io.writeln(f"DEF FN {fm.group(1).upper()}({', '.join(fparams)})")
                return

        m = RE_DEF_SINGLE.match(rest)
        if not m:
            self.io.writeln("?USAGE: DEF NAME[(params)] = GATE : GATE : ...")
            self.io.writeln("        DEF BEGIN NAME[(params)]  (multi-line, end with DEF END)")
            return
        name = m.group(1).upper()
        params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
        body = [s.strip() for s in m.group(3).split(':') if s.strip()]
        if name in GATE_TABLE:
            from qubasic_core.engine import QBasicSyntaxError
            raise QBasicSyntaxError(f"CANNOT REDEFINE BUILT-IN GATE {name}")
        self.subroutines[name] = {'body': body, 'params': params}
        if params:
            self.io.writeln(f"DEF {name}({', '.join(params)}) ({len(body)} gates)")
        else:
            self.io.writeln(f"DEF {name} ({len(body)} gates)")

    def _def_multiline(self, header: str) -> None:
        """Read multi-line DEF block from REPL."""
        m = re.match(r'(\w+)(?:\(([^)]*)\))?', header)
        if not m:
            self.io.writeln("?USAGE: DEF BEGIN NAME[(params)]")
            return
        name = m.group(1).upper()
        params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
        if name in GATE_TABLE:
            from qubasic_core.engine import QBasicSyntaxError
            raise QBasicSyntaxError(f"CANNOT REDEFINE BUILT-IN GATE {name}")

        body = []
        self.io.writeln(f"  DEF {name} (type gates, DEF END to finish)")
        while True:
            try:
                line = self.io.read_line('  . ').strip()
            except (KeyboardInterrupt, EOFError):
                self.io.writeln("\n  CANCELLED")
                return
            if line.upper() == 'DEF END' or line.upper() == 'END':
                break
            if line:
                body.append(line)
        self.subroutines[name] = {'body': body, 'params': params}
        self.io.writeln(f"DEF {name} ({len(body)} gates)")

    def cmd_type(self, rest: str) -> None:
        """TYPE name — define a named record type.

        Usage: TYPE Point
                 x AS SINGLE
                 y AS SINGLE
               END TYPE
        Then: DIM p AS Point
              LET p.x = 3.14

        Works both interactively (REPL) and non-interactively (LOAD/INCLUDE).
        In non-interactive mode, subsequent lines are routed through process()
        which delegates to _accumulate_type_field until END TYPE is seen.
        """
        from qubasic_core.parser import RE_TYPE_BEGIN
        m = RE_TYPE_BEGIN.match(f"TYPE {rest}")
        if not m:
            self.io.writeln("?USAGE: TYPE <name>")
            return
        type_name = m.group(1).upper()
        # Always use the accumulation path — works for both REPL and scripts.
        # In REPL, repl() calls process() per line, which feeds _accumulate_type_field.
        # In scripts, cmd_load/cmd_include call process() per line likewise.
        self.symbols.pending_type = {'name': type_name, 'fields': []}
        self.io.writeln(f"  TYPE {type_name} (enter fields, END TYPE to finish)")

    def _accumulate_type_field(self, line: str) -> None:
        """Accumulate a field for a pending TYPE definition, or finalize on END TYPE."""
        from qubasic_core.parser import RE_END_TYPE, RE_TYPE_FIELD
        assert self.symbols.pending_type is not None  # only called while a TYPE is open
        stripped = line.strip()
        if RE_END_TYPE.match(stripped):
            # Finalize the type definition
            type_name = self.symbols.pending_type['name']
            fields = self.symbols.pending_type['fields']
            self.symbols.pending_type = None
            self.symbols.types[type_name] = fields
            self.io.writeln(f"TYPE {type_name} ({len(fields)} fields)")
            return
        fm = RE_TYPE_FIELD.match(stripped)
        if fm:
            ftype = fm.group(2).upper() + (f"*{fm.group(3)}" if fm.group(3) else '')
            self.symbols.pending_type['fields'].append((fm.group(1).lower(), ftype))
        elif stripped:
            self.io.writeln("  ?EXPECTED: <name> AS <type>")

    def cmd_circuit_def(self, rest: str) -> None:
        """CIRCUIT_DEF name start-end — define a circuit macro from line range."""
        m = re.match(r'(\w+)\s+(\d+)\s*-\s*(\d+)', rest)
        if not m:
            self.io.writeln("?USAGE: CIRCUIT_DEF <name> <start>-<end>")
            return
        name = m.group(1).upper()
        start, end = int(m.group(2)), int(m.group(3))
        body = []
        for ln in sorted(self.program.keys()):
            if start <= ln <= end:
                body.append(self.program[ln])
        if not body:
            self.io.writeln(f"?NO LINES IN RANGE {start}-{end}")
            return
        self.subroutines[name] = {'body': body, 'params': []}
        self.io.writeln(f"CIRCUIT {name} = lines {start}-{end} ({len(body)} gates)")


    def cmd_reg(self, rest: str) -> None:
        """REG <name> <size> — allocate a named qubit register."""
        # REG data 3
        parts = rest.split()
        if len(parts) != 2:
            self.io.writeln("?USAGE: REG <name> <size>")
            return
        name = parts[0].lower()
        size = int(parts[1])
        # Allocate starting after existing registers
        start = sum(s for _, s in self.registers.values())
        if start + size > self.num_qubits:
            from qubasic_core.engine import QBasicRangeError
            raise QBasicRangeError(f"NOT ENOUGH QUBITS (need {start+size}, have {self.num_qubits})")

        self.registers[name] = (start, size)
        self.io.writeln(f"REG {name} = qubits {start}-{start+size-1}")

    def cmd_let(self, rest: str) -> None:
        """LET <var> = <expr> — assign a variable at the prompt, typed as in a
        program: strings (s$), typed names (n% rounds), record fields (p.x)
        and array elements (a(0), m(i, j))."""
        from qubasic_core.parser import LetArrayStmt, LetStmt, LetStrStmt
        parsed = parse_stmt(f"LET {rest.strip()}")
        try:
            if isinstance(parsed, LetArrayStmt):
                self._cf_let_array(parsed.raw, self.variables, parsed)
                subs = [self._safe_eval(s) for s in self._split_arg_list(parsed.index_expr)]
                shown = self._array_read(parsed.name, subs)
                label = f"{parsed.name}({parsed.index_expr.strip()})"
            elif isinstance(parsed, LetStrStmt):
                shown = self._store_var(parsed.name, self._eval_string_expr(parsed.expr))
                label = parsed.name
            elif isinstance(parsed, LetStmt):
                self._cf_let_var(parsed.raw, self.variables, parsed)
                shown, label = self.variables[parsed.name], parsed.name
            else:
                self.io.writeln("?USAGE: LET <var> = <expr>")
                return
        except ValueError as e:
            self.io.writeln(f"?{e}")
            return
        self.io.writeln(f"{label} = {shown!r}" if isinstance(shown, str) else f"{label} = {shown}")

    def cmd_undo(self) -> None:
        """UNDO — undo the last program edit, showing what changes."""
        return self.cmd_undo_preview()

    def cmd_list_colored(self) -> None:
        """LIST with syntax highlighting."""
        if not self.program:
            self.io.writeln("EMPTY PROGRAM")
            return
        t = self._theme
        from qubasic_core.parser import GATE_ALIASES, GATE_TABLE
        gate_names = set(GATE_TABLE.keys()) | set(GATE_ALIASES.keys())
        flow_kw = {'FOR', 'NEXT', 'WHILE', 'WEND', 'IF', 'THEN', 'ELSE',
                    'GOTO', 'GOSUB', 'RETURN', 'END', 'DO', 'LOOP', 'EXIT',
                    'SUB', 'FUNCTION', 'CALL', 'SELECT', 'CASE'}
        for num in sorted(self.program.keys()):
            raw = self.program[num]
            upper = raw.strip().upper()
            if upper.startswith('REM') or upper.startswith("'"):
                colored = f"{t['comment']}{raw}{t['reset']}"
            else:
                words = raw.split()
                parts = []
                for w in words:
                    wu = w.upper().rstrip(',').rstrip(':')
                    if wu in gate_names:
                        parts.append(f"{t['gate']}{w}{t['reset']}")
                    elif wu in flow_kw:
                        parts.append(f"{t['flow']}{w}{t['reset']}")
                    elif w.startswith('"') or w.startswith("'"):
                        parts.append(f"{t['string']}{w}{t['reset']}")
                    else:
                        parts.append(w)
                colored = ' '.join(parts)
            self.io.writeln(f"  {num:5d}  {colored}")

    def cmd_undo_preview(self) -> None:
        """UNDO — show what will change and restore."""
        if not self.editor.undo:
            self.io.writeln("?NOTHING TO UNDO")
            return
        prev = self.editor.undo[-1]
        # Show diff
        added = set(self.program.keys()) - set(prev.keys())
        removed = set(prev.keys()) - set(self.program.keys())
        changed = {k for k in set(self.program.keys()) & set(prev.keys())
                    if self.program[k] != prev[k]}
        if added:
            self.io.writeln(f"  Undo will remove: {sorted(added)}")
        if removed:
            self.io.writeln(f"  Undo will restore: {sorted(removed)}")
        if changed:
            for k in sorted(changed):
                self.io.writeln(f"  Undo will revert {k}: {self.program[k]} -> {prev[k]}")
        # Apply
        self.program = self.editor.undo.pop()
        self._parsed.clear()
        self.io.writeln(f"UNDO ({len(self.program)} lines)")


# ---------------------------------------------------------------------------
# QUBASIC program management — AUTO, EDIT, COPY, MOVE, FIND, REPLACE, slots, CHECKSUM, CHAIN, MERGE, LIST+.
# ---------------------------------------------------------------------------

@dataclass
class EditorState:
    """Editing the program: the undo stack, the program banks, AUTO numbering,
    the INCLUDE nesting, and the lines a .bas import numbered."""

    undo: list[dict[int, str]] = field(default_factory=list)        # program snapshots for UNDO
    slots: dict[int, dict[int, str]] = field(default_factory=dict)  # BANK: programs by slot
    slot: int = 0
    auto: bool = False                                              # AUTO numbering
    auto_line: int = 10
    auto_step: int = 10
    include_depth: int = 0
    include_stack: list[str] = field(default_factory=list)          # the files INCLUDE is inside
    imported: set[int] = field(default_factory=set)                 # lines a .bas import numbered


class ProgramMgmtMixin(LanguageHost):
    """Program management commands for QBasicTerminal."""

    # ── AUTO ───────────────────────────────────────────────────────────

    def cmd_auto(self, rest: str = '') -> None:
        """AUTO [start][, step] — auto-generate line numbers."""
        parts = rest.replace(',', ' ').split()
        if parts:
            self.editor.auto_line = int(parts[0])
        if len(parts) > 1:
            self.editor.auto_step = int(parts[1])
        self.editor.auto = True
        self.io.writeln(f"AUTO {self.editor.auto_line}, {self.editor.auto_step} — type . to stop")
        while self.editor.auto:
            try:
                line = self.io.read_line(f'{self.editor.auto_line} ').rstrip()
                if line == '.':
                    self.editor.auto = False
                    break
                if line:
                    self.process(f'{self.editor.auto_line} {line}')
                self.editor.auto_line += self.editor.auto_step
            except (KeyboardInterrupt, EOFError):
                self.editor.auto = False
                self.io.writeln('')
                break

    # ── EDIT ───────────────────────────────────────────────────────────

    def cmd_edit(self, rest: str) -> None:
        """EDIT <line> — edit a specific line."""
        if not rest.strip():
            self.io.writeln("?USAGE: EDIT <line>")
            return
        num = int(rest.strip())
        if num not in self.program:
            self.io.writeln(f"?LINE {num} NOT FOUND")
            return
        current = self.program[num]
        self.io.writeln(f"  {num} {current}")
        try:
            new_content = self.io.read_line(f'{num} ').rstrip()
            if new_content:
                self.process(f"{num} {new_content}")     # as typed: parsed, and UNDO can revert it
                self.io.writeln(f"  UPDATED {num}")
            else:
                self.io.writeln("  (unchanged)")
        except (KeyboardInterrupt, EOFError):
            self.io.writeln("\n  (cancelled)")

    # ── LIST enhancements ──────────────────────────────────────────────

    def cmd_list_subs(self) -> None:
        """LIST SUBS — list all SUB/FUNCTION definitions."""
        found = False
        for ln in sorted(self.program.keys()):
            stmt = self.program[ln].strip().upper()
            if stmt.startswith('SUB ') or stmt.startswith('FUNCTION '):
                self.io.writeln(f"  {ln:5d}  {self.program[ln]}")
                found = True
        if self.subroutines:
            for name, sub in self.subroutines.items():
                if isinstance(sub, dict):
                    params = f"({', '.join(sub['params'])})" if sub['params'] else ""
                    self.io.writeln(f"  DEF  {name}{params} ({len(sub['body'])} gates)")
                found = True
        if not found:
            self.io.writeln("  No subroutines defined")

    def cmd_list_vars(self) -> None:
        """LIST VARS — alias for VARS with types."""
        if not self.variables:
            self.io.writeln("  No variables")
            return
        for name, val in sorted(self.variables.items()):
            typ = type(val).__name__
            self.io.writeln(f"  {name} = {val}  ({typ})")

    def cmd_list_arrays(self) -> None:
        """LIST ARRAYS — list all arrays with sizes."""
        if not self.arrays:
            self.io.writeln("  No arrays")
            return
        for name, data in sorted(self.arrays.items()):
            if isinstance(data, list):
                self.io.writeln(f"  {name}({len(data)}) = {data[:5]}{'...' if len(data) > 5 else ''}")
            elif isinstance(data, dict):
                self.io.writeln(f"  {name}({len(data)} dims)")

    # ── Program slots (BANK) ──────────────────────────────────────────

    def cmd_bank(self, rest: str) -> None:
        """BANK <n> — switch to program slot n."""
        if not rest.strip():
            self.io.writeln(f"  Current slot: {self.editor.slot}")
            self.io.writeln(f"  Slots in use: {sorted(self.editor.slots.keys())}")
            return
        slot = int(rest.strip())
        # Save current
        self.editor.slots[self.editor.slot] = dict(self.program)
        # Load target
        self.editor.slot = slot
        if slot in self.editor.slots:
            self.program = dict(self.editor.slots[slot])
            self.io.writeln(f"BANK {slot} ({len(self.program)} lines)")
        else:
            self.program = {}
            self.io.writeln(f"BANK {slot} (empty)")

    # ── COPY / MOVE ────────────────────────────────────────────────────

    def cmd_copy(self, rest: str) -> None:
        """COPY <start>-<end> TO <dest> — copy line range."""
        m = self._parse_range_to(rest)
        if not m:
            self.io.writeln("?USAGE: COPY <start>-<end> TO <dest>")
            return
        start, end, dest = m
        offset = dest - start
        for ln in sorted(self.program.keys()):
            if start <= ln <= end:
                self.program[ln + offset] = self.program[ln]
        self.io.writeln(f"COPIED {start}-{end} TO {dest}-{end + offset}")

    def cmd_move(self, rest: str) -> None:
        """MOVE <start>-<end> TO <dest> — move line range."""
        m = self._parse_range_to(rest)
        if not m:
            self.io.writeln("?USAGE: MOVE <start>-<end> TO <dest>")
            return
        start, end, dest = m
        offset = dest - start
        lines = {ln: self.program[ln] for ln in sorted(self.program.keys())
                 if start <= ln <= end}
        for ln in lines:
            del self.program[ln]
        for ln, stmt in lines.items():
            self.program[ln + offset] = stmt
        self.io.writeln(f"MOVED {start}-{end} TO {dest}-{end + offset}")

    @staticmethod
    def _parse_range_to(rest: str) -> tuple[int, int, int] | None:
        m = re.match(r'(\d+)\s*-\s*(\d+)\s+TO\s+(\d+)', rest, re.IGNORECASE)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    # ── FIND / REPLACE ─────────────────────────────────────────────────

    def cmd_find(self, rest: str) -> None:
        """FIND "text" — search program lines."""
        text = rest.strip().strip('"').strip("'")
        if not text:
            self.io.writeln("?USAGE: FIND \"text\"")
            return
        found = 0
        for ln in sorted(self.program.keys()):
            if text.upper() in self.program[ln].upper():
                self.io.writeln(f"  {ln:5d}  {self.program[ln]}")
                found += 1
        self.io.writeln(f"  {found} match(es)")

    def cmd_replace(self, rest: str) -> None:
        """REPLACE "old" WITH "new" — find and replace in program."""
        m = re.match(r'"([^"]+)"\s+WITH\s+"([^"]*)"', rest, re.IGNORECASE)
        if not m:
            self.io.writeln('?USAGE: REPLACE "old" WITH "new"')
            return
        old, new = m.group(1), m.group(2)
        count = 0
        for ln in sorted(self.program.keys()):
            if old in self.program[ln]:
                self.program[ln] = self.program[ln].replace(old, new)
                self.io.writeln(f"  {ln:5d}  {self.program[ln]}")
                count += 1
        self.io.writeln(f"  {count} replacement(s)")

    # ── CHECKSUM ───────────────────────────────────────────────────────

    def cmd_checksum(self) -> None:
        """CHECKSUM — hash of program listing for verification."""
        if not self.program:
            self.io.writeln("?EMPTY PROGRAM")
            return
        content = '\n'.join(f'{ln} {self.program[ln]}'
                           for ln in sorted(self.program.keys()))
        h = hashlib.md5(content.encode()).hexdigest()[:8].upper()
        self.io.writeln(f"CHECKSUM: {h} ({len(self.program)} lines)")

    # ── CHAIN / MERGE ──────────────────────────────────────────────────

    @staticmethod
    def _load_lines_with_defs(lines: list[str], process_fn: Callable[[str], Any]) -> int:
        """Iterate stripped lines, handling DEF BEGIN...DEF END blocks.

        Skips blank lines and # comments. For each DEF BEGIN block, collects
        the body lines and calls process_fn with a synthesized single-line DEF
        command. For all other lines, calls process_fn directly.

        Returns the count of logical lines/blocks processed (excluding skips).
        """
        count = 0
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
            if re.match(r'DEF\s+BEGIN\s+', line, re.IGNORECASE):
                dm = RE_DEF_BEGIN.match(line)
                if dm:
                    name = dm.group(1).upper()
                    params = [p.strip() for p in dm.group(2).split(',')] if dm.group(2) else []
                    body = []
                    i += 1
                    while i < len(lines):
                        bl = lines[i].strip()
                        if bl.upper() in ('DEF END', 'END'):
                            break
                        if bl and not bl.startswith('#'):
                            body.append(bl)
                        i += 1
                    body_str = ' : '.join(body)
                    param_str = f"({', '.join(params)})" if params else ""
                    process_fn(f"DEF {name}{param_str} = {body_str}")
                    count += 1
            else:
                process_fn(line)
                count += 1
            i += 1
        return count

    def cmd_chain(self, rest: str) -> None:
        """CHAIN "file" — load and run a program, preserving variables."""
        m = RE_CHAIN.match(f"CHAIN {rest}")
        if not m:
            self.io.writeln('?USAGE: CHAIN "filename"')
            return
        path = m.group(1).strip()
        try:
            path = self._sanitize_path(path)
        except ValueError as e:
            self.io.writeln(f"?CHAIN ERROR: {e}")
            return
        path = self._find_qb_file(path)
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        saved_vars = dict(self.variables)
        saved_arrays = {k: list(v) if isinstance(v, list) else v
                        for k, v in self.arrays.items()}
        self.program.clear()
        with open(path, 'r', encoding='utf-8-sig') as f:
            lines = [l.rstrip('\n\r') for l in f.readlines()]
        self._load_lines_with_defs(
            lines, lambda line: self.process(line, track_undo=False))
        self.variables.update(saved_vars)
        self.arrays.update(saved_arrays)
        self.io.writeln(f"CHAINED {path}")
        if self.program:
            self.cmd_run()

    def cmd_merge(self, rest: str) -> None:
        """MERGE "file" — merge lines from another file without clearing."""
        m = RE_MERGE.match(f"MERGE {rest}")
        if not m:
            self.io.writeln('?USAGE: MERGE "filename"')
            return
        path = m.group(1).strip()
        try:
            path = self._sanitize_path(path)
        except ValueError as e:
            self.io.writeln(f"?MERGE ERROR: {e}")
            return
        path = self._find_qb_file(path)
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        with open(path, 'r', encoding='utf-8-sig') as f:
            lines = [l.rstrip('\n\r') for l in f.readlines()]
        existing_before = set(self.program.keys())
        count = self._load_lines_with_defs(
            lines, lambda line: self.process(line, track_undo=False))
        # Detect lines that existed before the merge and were overwritten by
        # the loaded file.  Re-parse the raw file for numbered lines and
        # intersect with the set of line numbers that existed beforehand.
        merged_nums: set[int] = set()
        for raw in lines:
            m = re.match(r'^\s*(\d+)\s', raw)
            if m:
                merged_nums.add(int(m.group(1)))
        n_overwritten = len(existing_before & merged_nums)
        if n_overwritten:
            self.io.writeln(f"  WARNING: {n_overwritten} line(s) overwritten by merge")
        self.io.writeln(f"MERGED {path} ({count} lines)")

    # ── Introspection ─────────────────────────────────────────────────

    def cmd_defs(self) -> None:
        """DEFS — list the DEF subroutines and the program's SUB and FUNCTION blocks."""
        found = False
        if self.subroutines:
            for name, sub in self.subroutines.items():
                if isinstance(sub, list):
                    self.io.writeln(f"  {name} = {' : '.join(sub)}")
                else:
                    params = f"({', '.join(sub['params'])})" if sub['params'] else ""
                    self.io.writeln(f"  {name}{params} = {' : '.join(sub['body'])}")
                found = True
        # Also show SUB/FUNCTION blocks in the program
        from qubasic_core.parser import RE_FUNCTION, RE_SUB
        for ln in sorted(self.program.keys()):
            stmt = self.program[ln].strip()
            m = RE_SUB.match(stmt)
            if m:
                self.io.writeln(f"  SUB {m.group(1).upper()}({m.group(2) or ''}) at line {ln}")
                found = True
            m = RE_FUNCTION.match(stmt)
            if m:
                self.io.writeln(f"  FUNCTION {m.group(1).upper()}({m.group(2) or ''}) at line {ln}")
                found = True
        if not found:
            self.io.writeln("NO SUBROUTINES DEFINED")

    def cmd_regs(self) -> None:
        """REGS — list the named registers and their qubit ranges."""
        if not self.registers:
            self.io.writeln("NO REGISTERS DEFINED")
            return
        for name, (start, size) in self.registers.items():
            self.io.writeln(f"  {name}[0:{size}] -> qubits {start}-{start+size-1}")

    def cmd_vars(self) -> None:
        """VARS — list the variables and their values."""
        if not self.variables:
            self.io.writeln("NO VARIABLES SET")
            return
        for name, val in self.variables.items():
            self.io.writeln(f"  {name} = {val}")

    def cmd_clear(self, rest: str) -> None:
        """CLEAR var | ARRAYS — remove a variable or array, or every array."""
        name = rest.strip()
        if not name:
            self.io.writeln("?USAGE: CLEAR <var> or CLEAR ARRAYS")
            return
        if name.upper() == 'ARRAYS':
            self.arrays.clear()
            self.io.writeln("ARRAYS CLEARED")
        elif name in self.variables:
            del self.variables[name]
            self.io.writeln(f"CLEARED {name}")
        elif name in self.arrays:
            del self.arrays[name]
            self.io.writeln(f"CLEARED array {name}")
        else:
            self.io.writeln(f"?{name} NOT FOUND")


# ---------------------------------------------------------------------------
# QUBASIC program files: SAVE and LOAD (.qb sessions and .bas source),
# INCLUDE, IMPORT, DIR and the library search path.
# ---------------------------------------------------------------------------

def _record_leaves(rec: Record, prefix: str = '') -> list[tuple[str, Any]]:
    """(field path, value) for every non-record field of a record."""
    out: list[tuple[str, Any]] = []
    for key, val in rec.items():
        path = f"{prefix}{key}"
        if isinstance(val, Record):
            out.extend(_record_leaves(val, path + '.'))
        else:
            out.append((path, val))
    return out


def _literal(val: Any) -> str:
    """A value as a BASIC literal for SAVE."""
    if isinstance(val, str):
        return '"' + val.replace('"', '" + CHR$(34) + "') + '"'
    return str(val)


def number_source(lines: list[str]) -> list[tuple[int, str]]:
    """(line number, text) for QBASIC source lines. Blank lines are skipped;
    a line that starts with a number keeps it, and runs of unnumbered lines
    are numbered in steps of up to 10 that fit before the next numbered line."""
    entries: list[tuple[int | None, str]] = []
    for raw in lines:
        text = raw.rstrip()
        if not text.strip():
            continue
        m = re.match(r'\s*(\d+)(?:\s+(.*))?$', text)
        if m:
            if m.group(2):
                entries.append((int(m.group(1)), m.group(2).strip()))
            continue
        entries.append((None, text.strip()))
    out: list[tuple[int, str]] = []
    prev = 0
    i = 0
    while i < len(entries):
        num, text = entries[i]
        if num is not None:
            if num <= prev:
                raise ValueError(f"line {num} is out of order (after {prev})")
            out.append((num, text))
            prev = num
            i += 1
            continue
        j = i
        while j < len(entries) and entries[j][0] is None:
            j += 1
        count = j - i
        nxt = entries[j][0] if j < len(entries) else None
        step = 10 if nxt is None else min(10, (nxt - prev) // (count + 1))
        if step < 1:
            raise ValueError(f"no room to number {count} line(s) between {prev} and {nxt}")
        for k in range(count):
            out.append((prev + (k + 1) * step, entries[i + k][1]))
        prev = out[-1][0]
        i = j
    return out


class PersistenceMixin(LanguageHost):
    """Programs and session settings on disk."""

    def cmd_save(self, rest: str) -> None:
        """SAVE <filename> — write program, config, and definitions to a .qb file."""
        if not rest:
            self.io.writeln("?USAGE: SAVE <filename>")
            return
        try:
            path = self._sanitize_path(rest)
        except ValueError as e:
            self.io.writeln(f"?SAVE ERROR: {e}")
            return
        checked = self._check_agent_path(path, "SAVE")
        if checked is None:
            return
        path = checked
        if path.lower().endswith('.bas'):
            imported_bas: set[int] = self.editor.imported
            lines = [n for n in sorted(self.program) if n not in imported_bas]
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    for num in lines:
                        f.write(f"{num} {self.program[num]}\n")
                self.io.writeln(f"SAVED {len(lines)} lines to {path}")
            except OSError as e:
                self.io.writeln(f"?SAVE ERROR: {e}")
            return
        if not path.endswith('.qb'):
            path += '.qb'
        try:
            with open(path, 'w', encoding='utf-8') as f:
                # Save config
                f.write(f"QUBITS {self.num_qubits}\n")
                f.write(f"SHOTS {self.shots}\n")
                if self.sim_method != 'automatic':
                    f.write(f"METHOD {self.sim_method}\n")
                if self.sim_device != 'CPU':
                    f.write(f"METHOD {self.sim_device}\n")
                if self.sim_precision != 'double':
                    f.write(f"PRECISION {self.sim_precision.upper()}\n")
                if getattr(self, '_screen_mode', 0):
                    f.write(f"DISPLAY {self._screen_mode}\n")
                if self.symbols.option_base:
                    f.write(f"OPTION BASE {self.symbols.option_base}\n")
                if getattr(self, '_endian_big', False):
                    f.write("OPTION ENDIAN BIG\n")
                if getattr(self, '_seed', None) is not None:
                    f.write(f"SEED {self._seed}\n")
                for line in self._device_lines():
                    f.write(line + "\n")
                # Save LOCC config if active
                if self.locc_mode and self.locc:
                    mode = "JOINT" if self.locc.joint else "SPLIT"
                    sizes = ' '.join(str(s) for s in self.locc.sizes)
                    f.write(f"LOCC {mode} {sizes}\n")
                # Save user-defined TYPEs (multi-line block, re-read on LOAD)
                for tname, fields in self.symbols.types.items():
                    f.write(f"TYPE {tname}\n")
                    for fname, ftype in fields:
                        f.write(f"  {fname} AS {ftype}\n")
                    f.write("END TYPE\n")
                # Save DEF FN user functions
                for fname, fdef in self.symbols.fns.items():
                    short = fname[2:] if fname.upper().startswith('FN') else fname
                    params = ', '.join(fdef['params'])
                    f.write(f"DEF FN {short}({params}) = {fdef['body']}\n")
                # Save custom gates as executable UNITARY commands
                for name, matrix in self._custom_gates.items():
                    rows = matrix.tolist()
                    mat_str = '[' + ','.join(
                        '[' + ','.join(str(v) for v in row) + ']'
                        for row in rows
                    ) + ']'
                    f.write(f"UNITARY {name} = {mat_str}\n")
                # Save subroutines
                for name, sub in self.subroutines.items():
                    if isinstance(sub, list):
                        f.write(f"DEF {name} = {' : '.join(sub)}\n")
                    else:
                        params = f"({', '.join(sub['params'])})" if sub['params'] else ""
                        f.write(f"DEF {name}{params} = {' : '.join(sub['body'])}\n")
                # Save registers
                for name, (start, size) in self.registers.items():
                    f.write(f"REG {name} {size}\n")
                # Save variables (internal _vars skipped; a record as its DIM
                # and one LET per field)
                for name, val in self.variables.items():
                    if name.startswith('_'):
                        continue
                    if isinstance(val, Record):
                        f.write(f"DIM {name} AS {val.type_name}\n")
                        for path, leaf in _record_leaves(val):
                            f.write(f"LET {name}.{path} = {_literal(leaf)}\n")
                        continue
                    if not isinstance(val, (int, float, str)):
                        continue        # arrays of results, matrices, records of other kinds
                    f.write(f"LET {name} = {_literal(val)}\n")
                # Save program (exclude lines injected by IMPORT)
                imported: set[int] = self.editor.imported
                saved = [n for n in sorted(self.program.keys()) if n not in imported]
                for num in saved:
                    f.write(f"{num} {self.program[num]}\n")
            self.io.writeln(f"SAVED {len(saved)} lines to {path}")
        except Exception as e:
            self.io.writeln(f"?SAVE ERROR: {e}")

    def _device_lines(self) -> list[str]:
        """The session lines that rebuild the noise model, device model and
        logical-qubit mode on LOAD (SAVE writes them after QUBITS and METHOD)."""
        lines: list[str] = []
        lm = self._logical_mode
        edges = ', '.join(f"{a}-{b}" for a, b in
                          sorted({tuple(sorted(e)) for e in self._coupling_map or ()}))
        if self._device_spec:
            lines.append(f"DEVICE {self._device_spec}")
            if edges:
                lines.append(f"COUPLING {edges}")
            elif self._device_spec.split()[0] not in ('all', 'full'):
                lines.append("COUPLING OFF")
        else:
            spec = (lm.get('saved_noise') or (None, None))[1] if lm else self.noise.spec
            if spec:
                lines.append(f"NOISE {spec}")
            if edges:
                lines.append(f"COUPLING {edges}")
        if self._basis_gates:
            lines.append("BASIS " + ', '.join(self._basis_gates))
        if lm:
            if lm['code'] == 'STEANE':
                lines.append(f"LQUBITS {lm['n']} CODE STEANE PHYS {lm['p']:g}")
            else:
                mode = (' ENCODED' if lm.get('force_encoded')
                        else '' if lm.get('encoded') else ' MODEL')
                lines.append(f"LQUBITS {lm['n']} CODE {lm['code']} {lm['d']} "
                             f"PHYS {lm['p']:g}{mode}")
        return lines

    def _load_bas(self, path: str) -> int:
        """Load QBASIC source (.bas): every non-blank line is a program line;
        numbered lines keep their numbers and the rest are numbered to fit
        around them. Returns the number of lines loaded."""
        with open(path, 'r', encoding='utf-8-sig') as f:
            numbered = number_source(f.read().splitlines())
        self.cmd_new(silent=True)
        for num, text in numbered:
            self.process(f"{num} {text}", track_undo=False)
        return len(numbered)

    def cmd_load(self, rest: str) -> None:
        """LOAD <filename> — clear state and load a .qb program file (or QBASIC
        source, .bas, whose unnumbered lines are numbered)."""
        if not rest:
            self.io.writeln("?USAGE: LOAD <filename>")
            return
        try:
            path = self._sanitize_path(rest)
        except ValueError as e:
            self.io.writeln(f"?LOAD ERROR: {e}")
            return
        if os.path.isdir(path):
            self.io.writeln(f"?{path} is a directory, not a file")
            return
        if not path.endswith('.qb') and not os.path.isfile(path):
            path += '.qb'
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        if path.lower().endswith('.bas'):
            try:
                count = self._load_bas(path)
            except (ValueError, OSError) as e:
                self.io.writeln(f"?LOAD ERROR: {e}")
                return
            self.io.writeln(f"LOADED {path} ({count} lines)")
            return
        try:
            prev_qubits = self.num_qubits
            prev_shots = self.shots
            self.cmd_new(silent=True)
            with open(path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.rstrip('\n\r')
                    if line and not line.startswith('#'):
                        self.process(line, track_undo=False)
            self.io.writeln(f"LOADED {path} ({len(self.program)} lines)")
            if self.num_qubits != prev_qubits:
                self.io.writeln(f"  (QUBITS changed: {prev_qubits} -> {self.num_qubits})")
            if self.shots != prev_shots:
                self.io.writeln(f"  (SHOTS changed: {prev_shots} -> {self.shots})")
        except Exception as e:
            self.io.writeln(f"?LOAD ERROR: {e}")

    def cmd_include(self, rest: str) -> None:
        """INCLUDE file.qb — merge another program's lines into current.

        Depth-limited to MAX_INCLUDE_DEPTH to prevent infinite recursion.
        Cycle detection via _include_stack prevents A->B->A loops.
        SAVE, LOAD, and INCLUDE are blocked inside included files so that
        an included script cannot write arbitrary files or recurse further
        without the user's direct interaction.
        """
        if self.editor.include_depth >= MAX_INCLUDE_DEPTH:
            self.io.writeln(f"?INCLUDE DEPTH LIMIT ({MAX_INCLUDE_DEPTH}) — possible recursion")
            return
        try:
            path = self._sanitize_path(rest)
        except ValueError as e:
            self.io.writeln(f"?INCLUDE ERROR: {e}")
            return
        path = self._find_qb_file(path)
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        resolved = os.path.realpath(path)
        if resolved in self.editor.include_stack:
            self.io.writeln(f"?INCLUDE CYCLE: {path} already in include chain")
            return
        count = 0
        self.editor.include_depth += 1
        self.editor.include_stack.append(resolved)
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.rstrip('\n\r')
                    if not line or line.startswith('#'):
                        continue
                    # Block file-writing commands inside includes
                    first_word = line.split(None, 1)[0].upper() if line.strip() else ''
                    if first_word in ('SAVE', 'LOAD', 'EXPORT', 'CSV'):
                        self.io.writeln(f"  ?BLOCKED IN INCLUDE: {first_word}")
                        continue
                    self.process(line, track_undo=False)
                    count += 1
        finally:
            self.editor.include_stack.pop()
            self.editor.include_depth -= 1
        self.io.writeln(f"INCLUDED {path} ({count} lines)")

    def cmd_import(self, rest: str) -> None:
        """IMPORT "file.qb" — load subroutine definitions with namespace prefix.

        DEFs in the imported file are prefixed with the module name.
        E.g., IMPORT "math.qb" makes DEF ROT available as MATH.ROT.
        """
        path = rest.strip().strip('"').strip("'")
        if not path:
            self.io.writeln('?USAGE: IMPORT "filename"')
            return
        try:
            path = self._sanitize_path(path)
        except ValueError as e:
            self.io.writeln(f"?IMPORT ERROR: {e}")
            return
        path = self._find_qb_file(path)
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        # Module name from filename (without extension)
        mod_name = os.path.splitext(os.path.basename(path))[0].upper()
        # Parse the file for DEF statements
        import_count = 0
        # Reserved high range for injected SUB/FUNCTION lines, tracked so SAVE
        # and listings can exclude them.
        _import_line_counter = 900000
        in_procedure = False
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.rstrip('\n\r').strip()
                if not line or line.startswith('#'):
                    continue
                # Look for DEF name = body
                upper = line.upper()
                if in_procedure and not upper.startswith(('END SUB', 'END FUNCTION')):
                    ln = _import_line_counter          # a SUB or FUNCTION body line
                    while ln in self.program:
                        ln += 10
                    _import_line_counter = ln + 10
                    self.process(f"{ln} {line}", track_undo=False)
                    self.editor.imported.add(ln)
                    continue
                if upper.startswith('DEF ') and '=' in line and not upper.startswith('DEF BEGIN'):
                    from qubasic_core.parser import RE_DEF_SINGLE
                    dm = RE_DEF_SINGLE.match(line[4:].strip())
                    if dm:
                        name = dm.group(1).upper()
                        params = [p.strip() for p in dm.group(2).split(',')] if dm.group(2) else []
                        body = [s.strip() for s in dm.group(3).split(':') if s.strip()]
                        qualified = f"{mod_name}.{name}"
                        self.subroutines[qualified] = {'body': body, 'params': params}
                        import_count += 1
                # SUB/FUNCTION must live in self.program for _scan_subs and CALL
                # to find them, so inject into a reserved high range and record
                # the line numbers as imported (excluded from SAVE).
                elif upper.startswith(('SUB ', 'FUNCTION ', 'END SUB', 'END FUNCTION')):
                    ln = _import_line_counter
                    while ln in self.program:
                        ln += 10
                    _import_line_counter = ln + 10
                    self.process(f"{ln} {line}", track_undo=False)
                    self.editor.imported.add(ln)
                    in_procedure = not upper.startswith('END')
                    import_count += in_procedure
        self.io.writeln(f"IMPORTED {mod_name} ({import_count} definitions from {path})")

    def cmd_dir(self, rest: str = '') -> None:
        """DIR [path] — list the .qb and .bas programs in a directory."""
        if rest.strip():
            path = self._sanitize_path(rest)
        else:
            path = '.'
        try:
            files = [f for f in os.listdir(path) if f.lower().endswith(('.qb', '.bas'))]
            if files:
                for f in sorted(files):
                    size = os.path.getsize(os.path.join(path, f))
                    self.io.writeln(f"  {f:<30} {size:>6} bytes")
            else:
                self.io.writeln("  No .qb or .bas files found")
        except Exception as e:
            self.io.writeln(f"?DIR ERROR: {e}")

    def _qb_search_dirs(self) -> list[str]:
        """Directories searched for .qb files: cwd then QUBASIC_PATH entries."""
        dirs = ['.']
        env = os.environ.get('QUBASIC_PATH', '')
        if env:
            dirs += [d for d in env.split(os.pathsep) if d]
        return dirs

    def _find_qb_file(self, path: str) -> str:
        """Resolve a sanitized relative .qb path against cwd then the library
        search path. Returns the first existing match, else the cwd candidate."""
        candidates = [path] + ([path + '.qb'] if not path.endswith('.qb') else [])
        for d in self._qb_search_dirs():
            for c in candidates:
                full = c if d == '.' else os.path.join(d, c)
                if os.path.isfile(full):
                    return full
        return candidates[-1]

    def _check_agent_path(self, path: str, cmd: str) -> str | None:
        """In agent mode, restrict paths to cwd. Returns resolved path or None on error.

        Uses pathlib is_relative_to for safe containment check instead of
        string prefix matching, which can be bypassed via symlinks or
        crafted path components that share a common prefix string.
        """
        if not self.agent_mode:
            return path
        resolved = Path(os.path.realpath(os.path.join(os.getcwd(), path)))
        cwd_real = Path(os.path.realpath(os.getcwd()))
        if not resolved.is_relative_to(cwd_real):
            self.io.writeln(f"?{cmd} BLOCKED: path escapes working directory in agent mode")
            return None
        return str(resolved)
