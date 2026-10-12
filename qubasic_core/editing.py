"""QUBASIC program editing commands: LIST, NEW, DELETE, RENUM, DEF, TYPE, CIRCUIT_DEF, REG, LET, UNDO."""

from __future__ import annotations

import sys
import re

from qubasic_core.engine import GATE_TABLE, RE_DEF_SINGLE
from qubasic_core.parser import parse_stmt
from qubasic_core.host import HostState

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


class EditingMixin(HostState):
    """Program editing and definition commands for QBasicTerminal."""

    def cmd_list(self, rest: str = '') -> None:
        """LIST — display program lines. LIST SUBS|VARS|ARRAYS for filtered views."""
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
        """DELETE <line> or DELETE <start>-<end> — remove program lines."""
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
        if getattr(self, '_breakpoints', None):
            self._breakpoints = {line_map.get(b, b) for b in self._breakpoints}
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
            from qubasic_core.patterns import RE_DEF_FN
            fm = RE_DEF_FN.match(f"DEF {rest}")
            if fm:
                fparams = [p.strip() for p in fm.group(2).split(',') if p.strip()]
                self._user_fns['FN' + fm.group(1).upper()] = {
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
            from qubasic_core.errors import QBasicSyntaxError
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
            from qubasic_core.errors import QBasicSyntaxError
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
                 x AS FLOAT
                 y AS FLOAT
               END TYPE
        Then: DIM p AS Point
              LET p.x = 3.14

        Works both interactively (REPL) and non-interactively (LOAD/INCLUDE).
        In non-interactive mode, subsequent lines are routed through process()
        which delegates to _accumulate_type_field until END TYPE is seen.
        """
        from qubasic_core.engine import RE_TYPE_BEGIN
        m = RE_TYPE_BEGIN.match(f"TYPE {rest}")
        if not m:
            self.io.writeln("?USAGE: TYPE <name>")
            return
        type_name = m.group(1).upper()
        # Always use the accumulation path — works for both REPL and scripts.
        # In REPL, repl() calls process() per line, which feeds _accumulate_type_field.
        # In scripts, cmd_load/cmd_include call process() per line likewise.
        self._pending_type = {'name': type_name, 'fields': []}
        self.io.writeln(f"  TYPE {type_name} (enter fields, END TYPE to finish)")

    def _accumulate_type_field(self, line: str) -> None:
        """Accumulate a field for a pending TYPE definition, or finalize on END TYPE."""
        from qubasic_core.engine import RE_TYPE_FIELD, RE_END_TYPE
        assert self._pending_type is not None  # only called while a TYPE is open
        stripped = line.strip()
        if RE_END_TYPE.match(stripped):
            # Finalize the type definition
            type_name = self._pending_type['name']
            fields = self._pending_type['fields']
            self._pending_type = None
            self._user_types[type_name] = fields
            self.io.writeln(f"TYPE {type_name} ({len(fields)} fields)")
            return
        fm = RE_TYPE_FIELD.match(stripped)
        if fm:
            ftype = fm.group(2).upper() + (f"*{fm.group(3)}" if fm.group(3) else '')
            self._pending_type['fields'].append((fm.group(1).lower(), ftype))
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

    def cmd_apply_circuit(self, rest: str) -> None:
        """APPLY_CIRCUIT name [@offset] — apply a circuit macro."""
        m = re.match(r'(\w+)(?:\s+@(\d+))?', rest)
        if not m:
            self.io.writeln("?USAGE: APPLY_CIRCUIT <name> [@offset]")
            return
        name = m.group(1).upper()
        offset = int(m.group(2)) if m.group(2) else 0
        if name not in self.subroutines:
            self.io.writeln(f"?UNDEFINED CIRCUIT: {name}")
            return
        call_str = f"{name} @{offset}" if offset else name
        self.run_immediate(call_str)

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
            from qubasic_core.errors import QBasicRangeError
            raise QBasicRangeError(f"NOT ENOUGH QUBITS (need {start+size}, have {self.num_qubits})")

        self.registers[name] = (start, size)
        self.io.writeln(f"REG {name} = qubits {start}-{start+size-1}")

    def cmd_let(self, rest: str) -> None:
        """LET <var> = <expr> — assign a variable at the prompt, typed as in a
        program: strings (s$), typed names (n% rounds), record fields (p.x)
        and array elements (a(0), m(i, j))."""
        from qubasic_core.statements import LetArrayStmt, LetStmt, LetStrStmt
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
