"""QBASIC program management — AUTO, EDIT, COPY, MOVE, FIND, REPLACE, slots, CHECKSUM, CHAIN, MERGE, LIST+."""

from __future__ import annotations

import os
import re
import hashlib
from typing import Any

from qbasic_core.engine import RE_CHAIN, RE_MERGE, RE_DEF_BEGIN


class ProgramMgmtMixin:
    """Program management commands for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.variables,
    self.num_qubits, self.shots, self.process(), self.cmd_new(),
    self._sanitize_path().
    """

    def _init_program_mgmt(self) -> None:
        self._program_slots: dict[int, dict[int, str]] = {}
        self._current_slot: int = 0
        self._auto_mode: bool = False
        self._auto_line: int = 10
        self._auto_step: int = 10

    # ── AUTO ───────────────────────────────────────────────────────────

    def cmd_auto(self, rest: str = '') -> None:
        """AUTO [start][, step] — auto-generate line numbers."""
        parts = rest.replace(',', ' ').split()
        if parts:
            self._auto_line = int(parts[0])
        if len(parts) > 1:
            self._auto_step = int(parts[1])
        self._auto_mode = True
        self.io.writeln(f"AUTO {self._auto_line}, {self._auto_step} — type . to stop")
        while self._auto_mode:
            try:
                line = self.io.read_line(f'{self._auto_line} ').rstrip()
                if line == '.':
                    self._auto_mode = False
                    break
                if line:
                    self.process(f'{self._auto_line} {line}')
                self._auto_line += self._auto_step
            except (KeyboardInterrupt, EOFError):
                self._auto_mode = False
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
                self.program[num] = new_content
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
            self.io.writeln(f"  Current slot: {self._current_slot}")
            self.io.writeln(f"  Slots in use: {sorted(self._program_slots.keys())}")
            return
        slot = int(rest.strip())
        # Save current
        self._program_slots[self._current_slot] = dict(self.program)
        # Load target
        self._current_slot = slot
        if slot in self._program_slots:
            self.program = dict(self._program_slots[slot])
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
    def _load_lines_with_defs(lines: list[str], process_fn) -> int:
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
        if not path.endswith('.qb'):
            path += '.qb'
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        saved_vars = dict(self.variables)
        self.program.clear()
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.rstrip('\n\r') for l in f.readlines()]
        self._load_lines_with_defs(
            lines, lambda line: self.process(line, track_undo=False))
        self.variables.update(saved_vars)
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
        if not path.endswith('.qb'):
            path += '.qb'
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.rstrip('\n\r') for l in f.readlines()]
        count = self._load_lines_with_defs(
            lines, lambda line: self.process(line, track_undo=False))
        self.io.writeln(f"MERGED {path} ({count} lines)")

    # ── Introspection ─────────────────────────────────────────────────

    def cmd_defs(self) -> None:
        """List all defined subroutines."""
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
        from qbasic_core.engine import RE_SUB, RE_FUNCTION
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
        """List all named registers with their qubit ranges."""
        if not self.registers:
            self.io.writeln("NO REGISTERS DEFINED")
            return
        for name, (start, size) in self.registers.items():
            self.io.writeln(f"  {name}[0:{size}] -> qubits {start}-{start+size-1}")

    def cmd_vars(self) -> None:
        """List all variables and their current values."""
        if not self.variables:
            self.io.writeln("NO VARIABLES SET")
            return
        for name, val in self.variables.items():
            self.io.writeln(f"  {name} = {val}")

    def cmd_clear(self, rest: str) -> None:
        """CLEAR var — remove a variable. CLEAR ARRAYS — clear all arrays."""
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
