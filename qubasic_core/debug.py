"""QUBASIC debug — ON ERROR, TRON/TROFF, STOP/CONT, breakpoints, watch, ASSERT, callbacks."""

from __future__ import annotations

import re
import time
from typing import Any

MAX_SV_CHECKPOINTS = 200
# Time-travel history: one checkpoint may hold at most CHECKPOINT_MAX_BYTES
# (a 22-qubit double-precision state), and the history keeps the newest
# checkpoints within CHECKPOINT_BUDGET_BYTES.
CHECKPOINT_MAX_BYTES = 64 << 20
CHECKPOINT_BUDGET_BYTES = 256 << 20

from qubasic_core.engine import (
    ExecResult, ExecOutcome,
    RE_ON_ERROR, RE_RESUME, RE_ERROR_STMT, RE_ASSERT,
    RE_ON_MEASURE, RE_ON_TIMER,
)
from qubasic_core.host import HostState

# QBASIC's error numbers, matched against the start of an error message.
_ERROR_CODES = (
    ('NEXT WITHOUT FOR', 1), ('SYNTAX ERROR', 2), ('RETURN WITHOUT GOSUB', 3),
    ('OUT OF DATA', 4), ('ILLEGAL FUNCTION CALL', 5), ('OVERFLOW', 6),
    ('OUT OF MEMORY', 7), ('SUBSCRIPT OUT OF RANGE', 9), ('DUPLICATE DEFINITION', 10),
    ('DIVISION BY ZERO', 11), ('TYPE MISMATCH', 13), ("CAN'T CONTINUE", 17),
    ('FUNCTION NOT DEFINED', 18), ('NOT A FUNCTION', 18), ('RESUME WITHOUT ERROR', 20),
    ('FOR WITHOUT NEXT', 26), ('WHILE WITHOUT WEND', 29), ('WEND WITHOUT WHILE', 30),
    ('DUPLICATE LABEL', 33), ('SUBPROGRAM NOT DEFINED', 35),
    ('ARGUMENT-COUNT MISMATCH', 37), ('BAD FILE NAME OR NUMBER', 52),
    ('FILE NOT FOUND', 53), ('BAD FILE MODE', 54), ('INPUT PAST END OF FILE', 62),
    ('LABEL', 8), ('MATH DOMAIN ERROR', 5),
)


def error_code(err: Exception) -> int:
    """The ERR value for an error: the number ERROR n raised, else QBASIC's
    number for the message (1 when none applies)."""
    msg = str(err).upper()
    m = re.match(r'(?:LINE \d+:\s*)?ERROR (\d+)(?!\w)', msg)
    if m:
        return int(m.group(1))
    msg = re.sub(r'^LINE \d+:\s*', '', msg)
    msg = re.sub(r'^[A-Z$]+: ', '', msg) if not any(msg.startswith(k) for k, _ in _ERROR_CODES) else msg
    for key, code in _ERROR_CODES:
        if msg.startswith(key):
            return code
    return 1


class DebugMixin(HostState):
    """Error handling, debugging, and event callbacks for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.variables,
    self._eval_condition(), self._gosub_stack.
    """

    def _init_debug(self) -> None:
        self._error_target: int | None = None
        self._err_code: int = 0
        self._err_line: int = 0
        self._err_ip: int | None = None
        self._in_error_handler: bool = False
        self._trace_mode: bool = False
        self._stopped_ip: int | None = None
        self._cont_skip_stop_ip: int | None = None
        self._breakpoints: set[int] = set()
        self._watches: list[str] = []
        self._on_measure_target: int | None = None
        self._on_measure_fired: bool = False
        self._on_timer_target: int | None = None
        self._on_timer_interval: float = 0.0
        self._on_timer_last: float = 0.0
        # Time-travel: statevector checkpoints indexed by gate count
        self._sv_checkpoints: list[tuple[int, Any]] = []  # [(line_num, sv_copy)]
        self._tt_position: int = -1  # current position in checkpoint list

    # ── Error handling ─────────────────────────────────────────────────

    def _cf_on_error(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            target = parsed.target
        else:
            m = RE_ON_ERROR.match(stmt)
            if not m:
                return None
            target = int(m.group(1))
        if target == 0:
            self._error_target = None
        else:
            self._error_target = target
        return True, ExecResult.ADVANCE

    def _cf_resume(self, stmt: str, sorted_lines: list[int], *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            self._in_error_handler = False
            arg = parsed.arg
        else:
            m = RE_RESUME.match(stmt)
            if not m:
                return None
            self._in_error_handler = False
            arg = m.group(1)
        from qubasic_core.program import line_index
        err_ip = self._err_ip if self._err_ip is not None else line_index(sorted_lines, self._err_line)
        if arg is None or arg.strip() in ('', '0'):
            # RESUME (or RESUME 0) — retry the statement that caused the error
            return True, err_ip if err_ip is not None else ExecResult.ADVANCE
        if arg.strip().upper() == 'NEXT':
            # RESUME NEXT — continue with the statement after it
            return True, err_ip + 1 if err_ip is not None else ExecResult.ADVANCE
        # RESUME <line>
        target = int(arg.strip())
        ip = line_index(sorted_lines, target)
        if ip is None:
            raise RuntimeError(f"RESUME: LINE {target} NOT FOUND")
        return True, ip

    def _cf_error(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            code = parsed.code
        else:
            m = RE_ERROR_STMT.match(stmt)
            if not m:
                return None
            code = int(m.group(1))
        raise RuntimeError(f"ERROR {code}")

    def _handle_error(self, err: Exception, line_num: int,
                      sorted_lines: Any, ip: int | None = None) -> int | None:
        """Handle a runtime error at statement ip (line line_num). Returns the
        statement to jump to, or None to propagate."""
        if self._error_target is not None and not self._in_error_handler:
            self._err_code = error_code(err)
            self._err_line = line_num
            self._err_ip = ip
            self._in_error_handler = True
            self.variables['ERR'] = self._err_code
            self.variables['ERL'] = self._err_line
            from qubasic_core.program import line_index
            return line_index(sorted_lines, self._error_target)
        return None

    # ── ASSERT ─────────────────────────────────────────────────────────

    def _cf_assert(self, stmt: str, run_vars: dict[str, Any], *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            cond = parsed.condition
        else:
            m = RE_ASSERT.match(stmt)
            if not m:
                return None
            cond = m.group(1).strip()
        if not self._eval_condition(cond, run_vars):
            raise RuntimeError(f"ASSERTION FAILED: {cond}")
        return True, ExecResult.ADVANCE

    # ── Time-travel debugging ─────────────────────────────────────────

    def _checkpoint_sv(self, line_num: int) -> None:
        """Save a statevector checkpoint for REWIND/FORWARD.

        A state up to CHECKPOINT_MAX_BYTES (22 qubits at double precision) is
        kept; the oldest checkpoints are dropped beyond MAX_SV_CHECKPOINTS
        entries or CHECKPOINT_BUDGET_BYTES in total.
        """
        sv = self._last_sv
        if sv is None or sv.nbytes > CHECKPOINT_MAX_BYTES:
            return
        import numpy as np
        self._sv_checkpoints.append((line_num, np.array(sv, copy=True)))
        total = sum(c.nbytes for _, c in self._sv_checkpoints)
        while self._sv_checkpoints and (len(self._sv_checkpoints) > MAX_SV_CHECKPOINTS
                                        or total > CHECKPOINT_BUDGET_BYTES):
            total -= self._sv_checkpoints.pop(0)[1].nbytes
        self._tt_position = len(self._sv_checkpoints) - 1

    def cmd_rewind(self, rest: str = '') -> None:
        """REWIND [N] — go back N steps in the statevector history."""
        if not self._sv_checkpoints:
            self.io.writeln("?NO CHECKPOINTS — use STEP mode to build history")
            return
        n = int(rest.strip()) if rest.strip() else 1
        new_pos = max(0, self._tt_position - n)
        self._tt_position = new_pos
        line_num, sv = self._sv_checkpoints[new_pos]
        self.last_sv = sv.copy()
        self.io.writeln(f"REWIND to step {new_pos} (line {line_num})")
        self._print_sv_compact(sv)

    def cmd_forward(self, rest: str = '') -> None:
        """FORWARD [N] — go forward N steps in the statevector history."""
        if not self._sv_checkpoints:
            self.io.writeln("?NO CHECKPOINTS")
            return
        n = int(rest.strip()) if rest.strip() else 1
        new_pos = min(len(self._sv_checkpoints) - 1, self._tt_position + n)
        self._tt_position = new_pos
        line_num, sv = self._sv_checkpoints[new_pos]
        self.last_sv = sv.copy()
        self.io.writeln(f"FORWARD to step {new_pos} (line {line_num})")
        self._print_sv_compact(sv)

    def cmd_history(self, rest: str = '') -> None:
        """HISTORY — show statevector checkpoint list."""
        if not self._sv_checkpoints:
            self.io.writeln("?NO CHECKPOINTS")
            return
        for i, (ln, _) in enumerate(self._sv_checkpoints):
            marker = " <<" if i == self._tt_position else ""
            self.io.writeln(f"  [{i}] line {ln}{marker}")

    # ── TRON / TROFF ───────────────────────────────────────────────────

    def cmd_tron(self) -> None:
        """TRON — trace on, print each line number during execution."""
        self._trace_mode = True
        self.io.writeln("TRACE ON")

    def cmd_troff(self) -> None:
        """TROFF — trace off."""
        self._trace_mode = False
        self.io.writeln("TRACE OFF")

    def _trace_line(self, line_num: int) -> None:
        """Print trace output if trace mode is on."""
        if self._trace_mode:
            self.io.write(f"[{line_num}]" + ' ')

    # ── STOP / CONT ────────────────────────────────────────────────────

    def _cf_stop(self, stmt: str, sorted_lines: list[int], ip: int, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is None and stmt.strip().upper() != 'STOP':
            return None
        # CONT sets _cont_skip_stop_ip to replay past this STOP
        if self._cont_skip_stop_ip is not None and self._cont_skip_stop_ip == ip:
            self._cont_skip_stop_ip = None
            return True, ExecResult.ADVANCE
        line_num = sorted_lines[ip]
        self.io.writeln(f"STOPPED AT LINE {line_num}")
        self._stopped_ip = ip
        self._print_watches()
        return True, ExecResult.END

    def cmd_cont(self) -> None:
        """CONT — continue execution after STOP, resuming from the next line.

        Re-executes the program from the top but skips the STOP that fired,
        so lines after STOP are reached. Variable state is rebuilt by
        replaying all lines before the STOP.
        """
        if self._stopped_ip is None:
            self.io.writeln("?CANNOT CONTINUE")
            return
        self.io.writeln("CONTINUING...")
        self._cont_skip_stop_ip = self._stopped_ip
        self._stopped_ip = None
        self.cmd_run()

    # ── Breakpoints ────────────────────────────────────────────────────

    def cmd_breakpoint(self, rest: str) -> None:
        """BREAK <line> | BREAK CLEAR | BREAK LIST | BREAK <start>-<end> — manage breakpoints."""
        if not rest.strip():
            if self._breakpoints:
                self.io.writeln(f"  Breakpoints: {sorted(self._breakpoints)}")
            else:
                self.io.writeln("  No breakpoints set")
            return
        rest = rest.strip().upper()
        if rest in ('CLEAR', 'ALL'):
            self._breakpoints.clear()
            self.io.writeln("BREAKPOINTS CLEARED")
            return
        if rest == 'LIST':
            if self._breakpoints:
                for bp in sorted(self._breakpoints):
                    src = self.program.get(bp, '(no source)')
                    self.io.writeln(f"  {bp}: {src}")
            else:
                self.io.writeln("  No breakpoints set")
            return
        # Range: BREAK 10-50
        if '-' in rest:
            try:
                a, b = rest.split('-')
                for bp in list(self._breakpoints):
                    if int(a) <= bp <= int(b):
                        self._breakpoints.discard(bp)
                self.io.writeln(f"BREAKPOINTS CLEARED IN {a}-{b}")
                return
            except ValueError:
                pass
        try:
            line = int(rest)
            if line in self._breakpoints:
                self._breakpoints.discard(line)
                self.io.writeln(f"BREAKPOINT REMOVED: {line}")
            else:
                self._breakpoints.add(line)
                self.io.writeln(f"BREAKPOINT SET: {line}")
        except ValueError:
            self.io.writeln("?USAGE: BREAK <line> | BREAK <start>-<end> | BREAK CLEAR | BREAK LIST")

    def _check_breakpoint(self, line_num: int, sorted_lines: list[int], ip: int) -> bool:
        """Check if we should break at this line. Returns True to stop."""
        if line_num in self._breakpoints:
            self.io.writeln(f"BREAKPOINT AT LINE {line_num}")
            self._stopped_ip = ip
            self._print_watches()
            return True
        return False

    # ── Watch expressions ──────────────────────────────────────────────

    def cmd_watch(self, rest: str) -> None:
        """WATCH <expr> — add/remove/list watch expressions."""
        if not rest.strip():
            if self._watches:
                self.io.writeln("  Watch expressions:")
                for w in self._watches:
                    self.io.writeln(f"    {w}")
            else:
                self.io.writeln("  No watches set")
            return
        rest = rest.strip()
        if rest.upper() == 'CLEAR':
            self._watches.clear()
            self.io.writeln("WATCHES CLEARED")
            return
        if rest in self._watches:
            self._watches.remove(rest)
            self.io.writeln(f"WATCH REMOVED: {rest}")
        else:
            self._watches.append(rest)
            self.io.writeln(f"WATCHING: {rest}")

    def _print_watches(self) -> None:
        for w in self._watches:
            try:
                val = self._safe_eval(w)
                self.io.writeln(f"  {w} = {val}")
            except Exception:
                self.io.writeln(f"  {w} = ?")

    # ── Callbacks ──────────────────────────────────────────────────────

    def _cf_on_measure(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            self._on_measure_target = parsed.target
        else:
            m = RE_ON_MEASURE.match(stmt)
            if not m:
                return None
            self._on_measure_target = int(m.group(1))
        return True, ExecResult.ADVANCE

    def _cf_on_timer(self, stmt: str, *, parsed=None) -> tuple[bool, ExecOutcome] | None:
        if parsed is not None:
            self._on_timer_interval = float(parsed.interval)
            self._on_timer_target = parsed.target
        else:
            m = RE_ON_TIMER.match(stmt)
            if not m:
                return None
            self._on_timer_interval = float(m.group(1))
            self._on_timer_target = int(m.group(2))
        self._on_timer_last = time.time()
        return True, ExecResult.ADVANCE

    def _check_timer_callback(self, sorted_lines: list[int], ip: int) -> int | None:
        """Check if ON TIMER callback should fire. Returns ip to jump to, or None."""
        if self._on_timer_target is not None and self._on_timer_interval > 0:
            now = time.time()
            if now - self._on_timer_last >= self._on_timer_interval:
                self._on_timer_last = now
                from qubasic_core.program import line_index
                target = line_index(sorted_lines, self._on_timer_target)
                if target is not None:
                    self._gosub_stack.append(ip)
                    return target
        return None
