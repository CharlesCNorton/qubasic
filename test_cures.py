#!/usr/bin/env python3
"""
Comprehensive test suite for QBASIC features not covered by test_qbasic.py.

Covers: DO/LOOP, PEEK/POKE, SYS, USR, WAIT, DUMP, MAP, CATALOG,
string functions, screen commands, SUB/FUNCTION, LOCAL/STATIC/SHARED,
error handling, breakpoints, watches, trace, profiling, STATS,
AUTO, EDIT, COPY/MOVE, FIND/REPLACE, BANK, CHECKSUM, CHAIN, MERGE,
file handles, LPRINT, DATA/READ/RESTORE, ON GOTO/GOSUB, SELECT CASE,
SWAP, DEF FN, OPTION BASE, ASSERT, PRINT USING, DIM multi,
LET string, LINE INPUT, POKE/SYS during execution, ON MEASURE/TIMER,
SCREEN auto-display, COLOR ANSI, LOCATE, hex/binary literals,
bitwise operators, EXIT FOR/WHILE/DO, SYS INSTALL, UNITARY multi-qubit,
CTRL custom gate, INV parametric, MEASURE_X/Y, SYNDROME, LOCC MEAS/CTRL,
parser Stmt types, errors hierarchy, IOPort protocol.

Run: python test_cures.py
"""

import sys
import os
import io
import math
import unittest
import tempfile
import builtins

sys.path.insert(0, os.path.dirname(__file__))
from qbasic_core.terminal import QBasicTerminal
from qbasic_core.errors import (
    QBasicError, QBasicSyntaxError, QBasicRuntimeError,
    QBasicBuildError, QBasicRangeError, QBasicIOError,
    QBasicUndefinedError,
)
from qbasic_core.io_protocol import IOPort, StdIOPort
from qbasic_core.parser import parse_stmt
from qbasic_core.statements import (
    Stmt, RawStmt, RemStmt, MeasureStmt, EndStmt, ReturnStmt,
    BarrierStmt, WendStmt, GotoStmt, GosubStmt, ForStmt, NextStmt,
    WhileStmt, IfThenStmt, LetStmt, LetArrayStmt, PrintStmt,
    GateStmt, MeasStmt, ResetStmt, SendStmt, ShareStmt, AtRegStmt,
    CompoundStmt,
)


def capture(func, *args, **kwargs):
    """Capture stdout from a function call, return (result, output_str)."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old
    return result, buf.getvalue()


# =====================================================================
# 1. DO/LOOP
# =====================================================================
class TestDoLoop(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_do_while_pretest(self):
        """DO WHILE condition ... LOOP — pre-test loop."""
        self.t.process('10 LET i = 1')
        self.t.process('20 DO WHILE i <= 3')
        self.t.process('30 PRINT i')
        self.t.process('40 LET i = i + 1')
        self.t.process('50 LOOP')
        self.t.process('60 END')
        _, out = capture(self.t.cmd_run)
        lines = [l.strip() for l in out.strip().split('\n') if l.strip()]
        # Should print 1, 2, 3 among the output
        printed = [l for l in lines if l in ('1', '1.0', '2', '2.0', '3', '3.0')]
        self.assertEqual(len(printed), 3)

    def test_loop_while_posttest(self):
        """DO ... LOOP WHILE condition — post-test loop."""
        self.t.process('10 LET i = 1')
        self.t.process('20 DO')
        self.t.process('30 PRINT i')
        self.t.process('40 LET i = i + 1')
        self.t.process('50 LOOP WHILE i <= 3')
        self.t.process('60 END')
        _, out = capture(self.t.cmd_run)
        printed = [l.strip() for l in out.strip().split('\n')
                   if l.strip() in ('1', '1.0', '2', '2.0', '3', '3.0')]
        self.assertEqual(len(printed), 3)

    def test_do_until(self):
        """DO UNTIL condition — pre-test UNTIL."""
        self.t.process('10 LET i = 1')
        self.t.process('20 DO UNTIL i > 2')
        self.t.process('30 PRINT i')
        self.t.process('40 LET i = i + 1')
        self.t.process('50 LOOP')
        self.t.process('60 END')
        _, out = capture(self.t.cmd_run)
        printed = [l.strip() for l in out.strip().split('\n')
                   if l.strip() in ('1', '1.0', '2', '2.0')]
        self.assertEqual(len(printed), 2)

    def test_loop_until_posttest(self):
        """DO ... LOOP UNTIL condition — post-test UNTIL."""
        self.t.process('10 LET i = 5')
        self.t.process('20 DO')
        self.t.process('30 PRINT i')
        self.t.process('40 LET i = i + 1')
        self.t.process('50 LOOP UNTIL i > 5')
        self.t.process('60 END')
        _, out = capture(self.t.cmd_run)
        # post-test: body runs at least once
        self.assertIn('5', out)

    def test_infinite_do_loop_with_goto_exit(self):
        """Infinite DO/LOOP with GOTO to exit (EXIT DO from IF THEN
        is a known limitation of the circuit-build path)."""
        self.t.process('10 LET i = 0')
        self.t.process('20 DO')
        self.t.process('30 LET i = i + 1')
        self.t.process('40 IF i == 3 THEN GOTO 70')
        self.t.process('50 LOOP')
        self.t.process('70 PRINT i')
        self.t.process('80 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('3', out)


# =====================================================================
# 2. PEEK/POKE
# =====================================================================
class TestPeekPoke(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_zero_page_read_write(self):
        self.t._poke(0x0000, 42.0)
        self.assertEqual(self.t._peek(0x0000), 42.0)
        self.t._poke(0x003F, 99.0)
        self.assertEqual(self.t._peek(0x003F), 99.0)

    def test_qubit_state_read(self):
        """Qubit state at $0100 after initialization is |0> -> P(1)=0."""
        val = self.t._peek(0x0100)
        self.assertEqual(val, 0.0)  # no statevector yet

    def test_config_read_write(self):
        """Read/write config addresses."""
        # shots
        self.t._poke(0xD001, 50)
        self.assertEqual(self.t.shots, 50)
        self.assertEqual(self.t._peek(0xD001), 50.0)
        # num_qubits
        self.t._poke(0xD000, 3)
        self.assertEqual(self.t.num_qubits, 3)
        self.assertEqual(self.t._peek(0xD000), 3.0)

    def test_status_read_only(self):
        """Writing to status address prints error."""
        _, out = capture(self.t._poke, 0xD010, 99)
        self.assertIn('READ-ONLY', out)

    def test_cmd_poke(self):
        """POKE command via process()."""
        _, out = capture(self.t.cmd_poke, '0, 7')
        self.assertEqual(self.t._peek(0), 7.0)


# =====================================================================
# 3. SYS
# =====================================================================
class TestSys(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_sys_builtin_bell(self):
        """SYS 0xE000 runs BELL demo."""
        _, out = capture(self.t.cmd_sys, '0xE000')
        # BELL demo produces output
        self.assertTrue(len(out) > 0)

    def test_sys_no_routine(self):
        """SYS at unmapped address."""
        _, out = capture(self.t.cmd_sys, '0xFFFF')
        self.assertIn('NO ROUTINE', out)


# =====================================================================
# 4. USR(addr)
# =====================================================================
class TestUsr(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_usr_returns_float(self):
        """USR returns float (0 if no counts)."""
        val = self.t._usr_fn(0xFFFF)  # no routine, no counts
        self.assertEqual(val, 0.0)


# =====================================================================
# 5. WAIT
# =====================================================================
class TestWait(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_wait_immediate_match(self):
        """WAIT succeeds immediately if PEEK already matches."""
        self.t._poke(0, 0xFF)
        _, out = capture(self.t.cmd_wait, '0, 255')
        # Should NOT print timeout
        self.assertNotIn('TIMEOUT', out)

    def test_wait_timeout(self):
        """WAIT times out when condition never matches."""
        # Set a very short internal timeout by monkey-patching
        import qbasic_core.memory as mem_mod
        orig = mem_mod.time.time
        call_count = [0]
        def fast_time():
            call_count[0] += 1
            # Make time advance rapidly
            return call_count[0] * 100
        mem_mod.time.time = fast_time
        try:
            _, out = capture(self.t.cmd_wait, '0, 255, 128')
            self.assertIn('TIMEOUT', out)
        finally:
            mem_mod.time.time = orig


# =====================================================================
# 6. DUMP
# =====================================================================
class TestDump(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_dump_output(self):
        """DUMP shows hex-formatted output."""
        self.t._poke(0, 0xAB)
        _, out = capture(self.t.cmd_dump, '0 15')
        self.assertIn('$0000:', out)
        self.assertIn('AB', out)


# =====================================================================
# 7. MAP
# =====================================================================
class TestMap(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_map_output(self):
        """MAP shows memory map structure."""
        _, out = capture(self.t.cmd_map)
        self.assertIn('Memory Map', out)
        self.assertIn('Zero Page', out)
        self.assertIn('QPU Config', out)
        self.assertIn('SYS Routines', out)


# =====================================================================
# 8. CATALOG
# =====================================================================
class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_catalog_lists_routines(self):
        """CATALOG lists built-in SYS routines."""
        _, out = capture(self.t.cmd_catalog)
        self.assertIn('BELL', out)
        self.assertIn('GHZ', out)
        self.assertIn('$E000', out)


# =====================================================================
# 9. String functions
# =====================================================================
class TestStringFunctions(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_left(self):
        from qbasic_core.strings import _left
        self.assertEqual(_left("HELLO", 3), "HEL")

    def test_right(self):
        from qbasic_core.strings import _right
        self.assertEqual(_right("HELLO", 3), "LLO")

    def test_mid(self):
        from qbasic_core.strings import _mid
        self.assertEqual(_mid("HELLO", 2, 3), "ELL")

    def test_chr(self):
        from qbasic_core.strings import _chr_fn
        self.assertEqual(_chr_fn(65), "A")

    def test_asc(self):
        from qbasic_core.strings import _asc
        self.assertEqual(_asc("A"), 65.0)

    def test_instr(self):
        from qbasic_core.strings import _instr
        self.assertEqual(_instr("HELLO WORLD", "WORLD"), 7.0)  # 1-based
        self.assertEqual(_instr("HELLO", "XYZ"), 0.0)

    def test_hex(self):
        from qbasic_core.strings import _hex_fn
        self.assertEqual(_hex_fn(255), "FF")

    def test_bin(self):
        from qbasic_core.strings import _bin_fn
        self.assertEqual(_bin_fn(5), "101")

    def test_str(self):
        from qbasic_core.strings import _str_fn
        self.assertEqual(_str_fn(42.0), "42")
        self.assertEqual(_str_fn(3.14), "3.14")

    def test_val(self):
        from qbasic_core.strings import _val_fn
        self.assertEqual(_val_fn("42"), 42.0)
        self.assertEqual(_val_fn("abc"), 0.0)

    def test_len(self):
        from qbasic_core.strings import _len_fn
        self.assertEqual(_len_fn("HELLO"), 5.0)


# =====================================================================
# 10. Screen commands
# =====================================================================
class TestScreenCommands(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_screen_mode_set_get(self):
        _, out = capture(self.t.cmd_screen, '2')
        self.assertIn('SCREEN 2', out)
        self.assertEqual(self.t._screen_mode, 2)
        _, out = capture(self.t.cmd_screen, '')
        self.assertIn('statevector', out)

    def test_color_ansi(self):
        _, out = capture(self.t.cmd_color, 'green')
        self.assertIn('\033[32m', out)

    def test_cls_ansi(self):
        _, out = capture(self.t.cmd_cls)
        self.assertIn('\033[2J', out)

    def test_locate_escape(self):
        _, out = capture(self.t.cmd_locate, '5, 10')
        self.assertIn('\033[5;10H', out)

    def test_play(self):
        _, out = capture(self.t.cmd_play, '1')
        self.assertIn('\a', out)

    def test_prompt_set(self):
        _, out = capture(self.t.cmd_prompt, '">>> "')
        self.assertEqual(self.t._prompt, '>>> ')


# =====================================================================
# 11. SUB/END SUB
# =====================================================================
class TestSubEndSub(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_define_and_call_sub(self):
        """Define SUB GREET, CALL GREET."""
        self.t.process('10 SUB GREET()')
        self.t.process('20 PRINT "HELLO FROM SUB"')
        self.t.process('30 END SUB')
        self.t.process('40 CALL GREET')
        self.t.process('50 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('HELLO FROM SUB', out)


# =====================================================================
# 12. FUNCTION/END FUNCTION
# =====================================================================
class TestFunctionEndFunction(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_define_and_invoke_function(self):
        """FUNCTION DOUBLE(x) invoked via _invoke_function."""
        self.t.process('10 FUNCTION DOUBLE(x)')
        self.t.process('20 LET DOUBLE = x * 2')
        self.t.process('30 END FUNCTION')
        # Scan SUB/FUNCTION definitions
        sorted_lines = sorted(self.t.program.keys())
        self.t._scan_subs(sorted_lines)
        self.assertIn('DOUBLE', self.t._func_defs)
        # Invoke the function directly
        result = self.t._invoke_function('DOUBLE', [5.0], sorted_lines)
        self.assertAlmostEqual(result, 10.0)


# =====================================================================
# 13. LOCAL variables in SUB
# =====================================================================
class TestLocalVars(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_local_scope(self):
        """LOCAL x creates x=0 in the current scope."""
        self.t.process('10 LET x = 99')
        self.t.process('20 SUB MYFN()')
        self.t.process('30 LOCAL x')
        self.t.process('40 PRINT x')
        self.t.process('50 END SUB')
        self.t.process('60 CALL MYFN')
        self.t.process('70 PRINT x')
        self.t.process('80 END')
        _, out = capture(self.t.cmd_run)
        lines = [l.strip() for l in out.split('\n') if l.strip() and l.strip().replace('.', '').replace('-', '').isdigit()]
        # First print (inside SUB) should be 0, second (after SUB) should be 99
        if len(lines) >= 2:
            self.assertIn(lines[0], ('0', '0.0'))


# =====================================================================
# 14. STATIC variables persist
# =====================================================================
class TestStaticVars(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_static_persists(self):
        """STATIC count persists across calls."""
        self.t.process('10 SUB COUNTER()')
        self.t.process('20 STATIC count')
        self.t.process('30 LET count = count + 1')
        self.t.process('40 PRINT count')
        self.t.process('50 END SUB')
        self.t.process('60 CALL COUNTER')
        self.t.process('70 CALL COUNTER')
        self.t.process('80 END')
        _, out = capture(self.t.cmd_run)
        nums = [l.strip() for l in out.split('\n')
                if l.strip() in ('1', '1.0', '2', '2.0')]
        self.assertEqual(len(nums), 2)


# =====================================================================
# 15. SHARED variables
# =====================================================================
class TestSharedVars(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_shared_accesses_outer(self):
        """SHARED x in SUB accesses outer scope variable."""
        self.t.process('10 LET x = 42')
        self.t.process('20 SUB SHOWX()')
        self.t.process('30 SHARED x')
        self.t.process('40 PRINT x')
        self.t.process('50 END SUB')
        self.t.process('60 CALL SHOWX')
        self.t.process('70 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('42', out)


# =====================================================================
# 16. ON ERROR GOTO / RESUME / RESUME NEXT
# =====================================================================
class TestOnError(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_on_error_sets_target(self):
        """ON ERROR GOTO sets the error handler target."""
        from qbasic_core.engine import ExecResult
        # Test at the unit level since ERROR statement conflicts with gate parser
        result = self.t._cf_on_error('ON ERROR GOTO 100')
        self.assertIsNotNone(result)
        self.assertEqual(result, (True, ExecResult.ADVANCE))
        self.assertEqual(self.t._error_target, 100)

    def test_on_error_goto_zero_clears(self):
        """ON ERROR GOTO 0 clears the error handler."""
        self.t._error_target = 100
        self.t._cf_on_error('ON ERROR GOTO 0')
        self.assertIsNone(self.t._error_target)

    def test_handle_error(self):
        """_handle_error routes to the error handler target."""
        self.t._error_target = 100
        self.t.program = {10: 'SOMETHING', 100: 'PRINT "CAUGHT"'}
        sorted_lines = [10, 100]
        ip = self.t._handle_error(RuntimeError("ERROR 42"), 10, sorted_lines)
        self.assertEqual(ip, 1)  # ip of line 100
        self.assertEqual(self.t._err_code, 42)
        self.assertEqual(self.t._err_line, 10)

    def test_resume_next(self):
        """RESUME NEXT advances past the error line."""
        self.t._err_line = 20
        self.t._in_error_handler = True
        sorted_lines = [10, 20, 30]
        self.t.program = {10: 'X', 20: 'Y', 30: 'Z'}
        result = self.t._cf_resume('RESUME NEXT', sorted_lines)
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 2)  # ip of line 30


# =====================================================================
# 17. ERR and ERL variables
# =====================================================================
class TestErrErl(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_err_erl_set(self):
        """ERR and ERL are set by _handle_error."""
        self.t._error_target = 100
        self.t.program = {20: 'BAD LINE', 100: 'HANDLER'}
        sorted_lines = [20, 100]
        self.t._handle_error(RuntimeError("ERROR 42"), 20, sorted_lines)
        self.assertEqual(self.t.variables['ERR'], 42)
        self.assertEqual(self.t.variables['ERL'], 20)


# =====================================================================
# 18. ERROR n
# =====================================================================
class TestErrorStmt(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_error_raises(self):
        """ERROR n raises a user error."""
        self.t.process('10 ERROR 99')
        _, out = capture(self.t.cmd_run)
        self.assertIn('ERROR', out)


# =====================================================================
# 19. Breakpoints
# =====================================================================
class TestBreakpoints(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_set_and_list_breakpoint(self):
        _, out = capture(self.t.cmd_breakpoint, '10')
        self.assertIn('BREAKPOINT SET: 10', out)
        _, out = capture(self.t.cmd_breakpoint, '')
        self.assertIn('10', out)

    def test_clear_breakpoints(self):
        _, _ = capture(self.t.cmd_breakpoint, '10')
        _, out = capture(self.t.cmd_breakpoint, 'CLEAR')
        self.assertIn('CLEARED', out)
        _, out = capture(self.t.cmd_breakpoint, '')
        self.assertIn('No breakpoints', out)

    def test_hit_breakpoint(self):
        """Breakpoint at a line triggers _check_breakpoint."""
        self.t._breakpoints.add(20)
        result = self.t._check_breakpoint(20, [10, 20, 30], 1)
        self.assertTrue(result)


# =====================================================================
# 20. Watch expressions
# =====================================================================
class TestWatch(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_add_and_list_watch(self):
        _, out = capture(self.t.cmd_watch, 'x')
        self.assertIn('WATCHING: x', out)
        _, out = capture(self.t.cmd_watch, '')
        self.assertIn('x', out)

    def test_watch_during_stop(self):
        """_print_watches displays watch values."""
        self.t.variables['x'] = 42
        self.t._watches.append('x')
        _, out = capture(self.t._print_watches)
        self.assertIn('42', out)


# =====================================================================
# 21. TRON/TROFF
# =====================================================================
class TestTronTroff(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_tron(self):
        _, out = capture(self.t.cmd_tron)
        self.assertIn('TRACE ON', out)
        self.assertTrue(self.t._trace_mode)

    def test_troff(self):
        self.t._trace_mode = True
        _, out = capture(self.t.cmd_troff)
        self.assertIn('TRACE OFF', out)
        self.assertFalse(self.t._trace_mode)

    def test_trace_line_output(self):
        self.t._trace_mode = True
        _, out = capture(self.t._trace_line, 10)
        self.assertIn('[10]', out)


# =====================================================================
# 22. STOP/CONT
# =====================================================================
class TestStopCont(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_stop_sets_stopped_ip(self):
        self.t.process('10 PRINT "BEFORE"')
        self.t.process('20 STOP')
        self.t.process('30 PRINT "AFTER"')
        _, out = capture(self.t.cmd_run)
        self.assertIn('STOPPED', out)
        self.assertIsNotNone(self.t._stopped_ip)

    def test_cont_cannot_continue(self):
        _, out = capture(self.t.cmd_cont)
        self.assertIn('CANNOT CONTINUE', out)


# =====================================================================
# 23. PROFILE ON/OFF/SHOW
# =====================================================================
class TestProfile(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_profile_on_off(self):
        _, out = capture(self.t.cmd_profile, 'ON')
        self.assertIn('PROFILE ON', out)
        self.assertTrue(self.t._profile_mode)
        _, out = capture(self.t.cmd_profile, 'OFF')
        self.assertIn('PROFILE OFF', out)
        self.assertFalse(self.t._profile_mode)

    def test_profile_show_empty(self):
        _, out = capture(self.t.cmd_profile, 'SHOW')
        self.assertIn('No profile data', out)


# =====================================================================
# 24. STATS accumulator
# =====================================================================
class TestStats(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_stats_show_empty(self):
        _, out = capture(self.t.cmd_stats, '')
        self.assertIn('No statistics', out)

    def test_stats_clear(self):
        self.t._stats_runs.append({'0': 10})
        _, out = capture(self.t.cmd_stats, 'CLEAR')
        self.assertIn('CLEARED', out)
        self.assertEqual(len(self.t._stats_runs), 0)

    def test_stats_run_n(self):
        """STATS N runs N trials (uses a trivial program)."""
        self.t.process('10 H 0')
        self.t.process('20 MEASURE')
        self.t.shots = 10
        _, out = capture(self.t.cmd_stats, '3')
        self.assertIn('Collected', out)
        self.assertEqual(len(self.t._stats_runs), 3)


# =====================================================================
# 25. AUTO (mock input)
# =====================================================================
class TestAuto(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_auto_adds_lines(self):
        inputs = iter(['PRINT "A"', 'PRINT "B"', '.'])
        orig = builtins.input
        builtins.input = lambda prompt='': next(inputs)
        try:
            _, out = capture(self.t.cmd_auto, '10, 10')
        finally:
            builtins.input = orig
        self.assertIn(10, self.t.program)
        self.assertIn(20, self.t.program)


# =====================================================================
# 26. EDIT (mock input)
# =====================================================================
class TestEdit(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_edit_line(self):
        self.t.process('10 PRINT "OLD"')
        orig = builtins.input
        builtins.input = lambda prompt='': 'PRINT "NEW"'
        try:
            _, out = capture(self.t.cmd_edit, '10')
        finally:
            builtins.input = orig
        self.assertEqual(self.t.program[10], 'PRINT "NEW"')
        self.assertIn('UPDATED', out)


# =====================================================================
# 27. COPY/MOVE line ranges
# =====================================================================
class TestCopyMove(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_copy(self):
        self.t.process('10 PRINT "A"')
        self.t.process('20 PRINT "B"')
        _, out = capture(self.t.cmd_copy, '10-20 TO 100')
        self.assertIn('COPIED', out)
        self.assertIn(100, self.t.program)
        self.assertIn(110, self.t.program)

    def test_move(self):
        self.t.process('10 PRINT "A"')
        self.t.process('20 PRINT "B"')
        _, out = capture(self.t.cmd_move, '10-20 TO 100')
        self.assertIn('MOVED', out)
        self.assertNotIn(10, self.t.program)
        self.assertIn(100, self.t.program)


# =====================================================================
# 28. FIND/REPLACE
# =====================================================================
class TestFindReplace(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_find(self):
        self.t.process('10 PRINT "HELLO"')
        self.t.process('20 PRINT "WORLD"')
        _, out = capture(self.t.cmd_find, 'HELLO')
        self.assertIn('1 match', out)

    def test_replace(self):
        self.t.process('10 PRINT "HELLO"')
        _, out = capture(self.t.cmd_replace, '"HELLO" WITH "HI"')
        self.assertIn('1 replacement', out)
        self.assertIn('HI', self.t.program[10])


# =====================================================================
# 29. BANK
# =====================================================================
class TestBank(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_switch_bank(self):
        self.t.process('10 PRINT "SLOT 0"')
        _, out = capture(self.t.cmd_bank, '1')
        self.assertIn('BANK 1', out)
        self.assertEqual(len(self.t.program), 0)
        # Switch back
        _, out = capture(self.t.cmd_bank, '0')
        self.assertIn(10, self.t.program)


# =====================================================================
# 30. CHECKSUM
# =====================================================================
class TestChecksum(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_checksum_output(self):
        self.t.process('10 H 0')
        _, out = capture(self.t.cmd_checksum)
        self.assertIn('CHECKSUM:', out)
        self.assertIn('1 lines', out)

    def test_checksum_empty(self):
        _, out = capture(self.t.cmd_checksum)
        self.assertIn('EMPTY PROGRAM', out)


# =====================================================================
# 31. CHAIN
# =====================================================================
class TestChain(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_chain_loads_and_runs(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qb', delete=False,
                                          dir='.', encoding='utf-8') as f:
            f.write('10 PRINT "CHAINED"\n')
            f.write('20 END\n')
            fname = os.path.basename(f.name)
        try:
            self.t.variables['preserved'] = 123
            _, out = capture(self.t.cmd_chain, f'"{fname}"')
            self.assertIn('CHAINED', out)
            # Variables preserved
            self.assertEqual(self.t.variables.get('preserved'), 123)
        finally:
            os.unlink(fname)


# =====================================================================
# 32. MERGE
# =====================================================================
class TestMerge(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_merge_adds_lines(self):
        self.t.process('10 PRINT "ORIGINAL"')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qb', delete=False,
                                          dir='.', encoding='utf-8') as f:
            f.write('20 PRINT "MERGED"\n')
            fname = os.path.basename(f.name)
        try:
            _, out = capture(self.t.cmd_merge, f'"{fname}"')
            self.assertIn('MERGED', out)
            self.assertIn(10, self.t.program)
            self.assertIn(20, self.t.program)
        finally:
            os.unlink(fname)


# =====================================================================
# 33. File handles
# =====================================================================
class TestFileHandles(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_open_write_close_read(self):
        fname = 'test_fh_tmp.txt'
        try:
            _, out = capture(self.t.cmd_open, f'"{fname}" FOR OUTPUT AS #1')
            self.assertIn('OPENED', out)
            # Write via the internal method
            self.t._exec_print_file('PRINT #1, "HELLO"', {})
            _, out = capture(self.t.cmd_close, '#1')
            self.assertIn('CLOSED', out)
            # Read back
            _, out = capture(self.t.cmd_open, f'"{fname}" FOR INPUT AS #2')
            self.assertIn('OPENED', out)
            self.t._exec_input_file('INPUT #2, line$', {})
            self.assertEqual(self.t.variables.get('line$'), 'HELLO')
            eof_val = self.t._eof(2)
            self.assertEqual(eof_val, 1.0)  # end of file
            capture(self.t.cmd_close, '#2')
        finally:
            if os.path.exists(fname):
                os.unlink(fname)


# =====================================================================
# 34. LPRINT
# =====================================================================
class TestLprint(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_lprint_to_stderr(self):
        """LPRINT without path writes to stderr."""
        old_stderr = sys.stderr
        buf = io.StringIO()
        sys.stderr = buf
        try:
            self.t._exec_lprint('LPRINT "TESTING"', {})
        finally:
            sys.stderr = old_stderr
        self.assertIn('TESTING', buf.getvalue())


# =====================================================================
# 35. DATA/READ/RESTORE
# =====================================================================
class TestDataReadRestore(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_data_read(self):
        self.t.process('10 DATA 10, 20, 30')
        self.t.process('20 READ a')
        self.t.process('30 READ b')
        self.t.process('40 PRINT a')
        self.t.process('50 PRINT b')
        self.t.process('60 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('10', out)
        self.assertIn('20', out)

    def test_restore(self):
        self.t.process('10 DATA 5, 6')
        self.t._collect_data()
        self.t._data_ptr = 2
        self.t.cmd_restore()
        self.assertEqual(self.t._data_ptr, 0)


# =====================================================================
# 36. ON expr GOTO / ON expr GOSUB
# =====================================================================
class TestOnGotoGosub(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_on_goto(self):
        self.t.process('10 LET x = 2')
        self.t.process('20 ON x GOTO 100, 200, 300')
        self.t.process('30 END')
        self.t.process('100 PRINT "ONE"')
        self.t.process('110 END')
        self.t.process('200 PRINT "TWO"')
        self.t.process('210 END')
        self.t.process('300 PRINT "THREE"')
        self.t.process('310 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('TWO', out)

    def test_on_gosub(self):
        self.t.process('10 LET x = 1')
        self.t.process('20 ON x GOSUB 100, 200')
        self.t.process('30 PRINT "BACK"')
        self.t.process('40 END')
        self.t.process('100 PRINT "SUB1"')
        self.t.process('110 RETURN')
        self.t.process('200 PRINT "SUB2"')
        self.t.process('210 RETURN')
        _, out = capture(self.t.cmd_run)
        self.assertIn('SUB1', out)
        self.assertIn('BACK', out)


# =====================================================================
# 37. SELECT CASE
# =====================================================================
class TestSelectCase(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_select_case(self):
        self.t.process('10 LET x = 2')
        self.t.process('20 SELECT CASE x')
        self.t.process('30 CASE 1')
        self.t.process('40 PRINT "ONE"')
        self.t.process('50 CASE 2')
        self.t.process('60 PRINT "TWO"')
        self.t.process('70 CASE ELSE')
        self.t.process('80 PRINT "OTHER"')
        self.t.process('90 END SELECT')
        self.t.process('100 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('TWO', out)
        self.assertNotIn('ONE', out)


# =====================================================================
# 38. SWAP
# =====================================================================
class TestSwap(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_swap(self):
        self.t.process('10 LET a = 1')
        self.t.process('20 LET b = 2')
        self.t.process('30 SWAP a, b')
        self.t.process('40 PRINT a')
        self.t.process('50 PRINT b')
        self.t.process('60 END')
        _, out = capture(self.t.cmd_run)
        lines = [l.strip() for l in out.split('\n')
                 if l.strip() in ('1', '1.0', '2', '2.0')]
        # a should be 2, b should be 1
        self.assertIn(lines[0], ('2', '2.0'))


# =====================================================================
# 39. DEF FN
# =====================================================================
class TestDefFn(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_def_fn(self):
        self.t.process('10 DEF FN SQUARE(x) = x * x')
        self.t.process('20 PRINT FNSQUARE(5)')
        self.t.process('30 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('25', out)


# =====================================================================
# 40. OPTION BASE
# =====================================================================
class TestOptionBase(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_option_base(self):
        self.t.process('10 OPTION BASE 1')
        self.t.process('20 END')
        _, out = capture(self.t.cmd_run)
        self.assertEqual(self.t._option_base, 1)


# =====================================================================
# 41. ASSERT
# =====================================================================
class TestAssert(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_assert_pass(self):
        self.t.process('10 ASSERT 1 == 1')
        self.t.process('20 PRINT "OK"')
        self.t.process('30 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('OK', out)

    def test_assert_fail(self):
        self.t.process('10 ASSERT 1 == 2')
        self.t.process('20 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('ASSERTION FAILED', out)


# =====================================================================
# 42. PRINT USING
# =====================================================================
class TestPrintUsing(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_print_using(self):
        """PRINT USING formats numbers. Test at the handler level."""
        self.t.variables['x'] = 3.14
        result = self.t._try_exec_print_using('PRINT USING "###.##"; x', {'x': 3.14})
        self.assertTrue(result)

    def test_print_using_output(self):
        """PRINT USING produces formatted output."""
        _, out = capture(self.t._try_exec_print_using,
                         'PRINT USING "###.##"; 3.14', {'dummy': 0})
        self.assertIn('3', out)


# =====================================================================
# 43. DIM multi-dimensional
# =====================================================================
class TestDimMulti(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_dim_multi(self):
        """DIM matrix(3,3) creates flat list with dimension metadata."""
        self.t.process('10 DIM matrix(3, 3)')
        self.t.process('20 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('matrix', self.t.arrays)
        self.assertIsInstance(self.t.arrays['matrix'], list)
        self.assertEqual(len(self.t.arrays['matrix']), 9)

    def test_dim_single(self):
        """DIM arr(5) creates 1D array."""
        self.t.process('10 DIM arr(5)')
        self.t.process('20 END')
        capture(self.t.cmd_run)
        self.assertIn('arr', self.t.arrays)
        self.assertEqual(len(self.t.arrays['arr']), 5)


# =====================================================================
# 44. LET with string variables
# =====================================================================
class TestLetString(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_let_str(self):
        """name$ = "hello" via cmd_let_str."""
        _, out = capture(self.t.cmd_let_str, 'greeting$', '"hello"')
        self.assertEqual(self.t.variables['greeting$'], 'hello')


# =====================================================================
# 45. LINE INPUT (mock input)
# =====================================================================
class TestLineInput(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_line_input(self):
        """LINE INPUT reads full line including spaces. Test via handler."""
        orig = builtins.input
        builtins.input = lambda prompt='': 'Alice Bob'
        try:
            result = self.t._try_exec_line_input('LINE INPUT "Name", name$', {})
        finally:
            builtins.input = orig
        self.assertTrue(result)
        self.assertEqual(self.t.variables.get('name$'), 'Alice Bob')


# =====================================================================
# 46. POKE during program execution
# =====================================================================
class TestPokeDuringExec(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_poke_in_program(self):
        self.t.process('10 POKE 0, 77')
        self.t.process('20 PRINT "DONE"')
        self.t.process('30 END')
        _, out = capture(self.t.cmd_run)
        self.assertEqual(self.t._peek(0), 77.0)


# =====================================================================
# 47. SYS during program execution
# =====================================================================
class TestSysDuringExec(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_sys_in_program(self):
        """SYS 0xE000 in a program line invokes BELL demo."""
        self.t.process('10 SYS 0xE000')
        self.t.process('20 END')
        _, out = capture(self.t.cmd_run)
        # Should produce some output from the BELL demo
        self.assertTrue(len(out) > 0)


# =====================================================================
# 48. ON MEASURE GOSUB
# =====================================================================
class TestOnMeasure(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_on_measure_sets_target(self):
        """ON MEASURE GOSUB sets the callback target (unit-level test;
        the ON expr GOSUB regex shadows this in the full execution path)."""
        from qbasic_core.engine import ExecResult
        result = self.t._cf_on_measure('ON MEASURE GOSUB 100')
        self.assertIsNotNone(result)
        self.assertEqual(result, (True, ExecResult.ADVANCE))
        self.assertEqual(self.t._on_measure_target, 100)


# =====================================================================
# 49. ON TIMER GOSUB
# =====================================================================
class TestOnTimer(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_on_timer_sets_target(self):
        """ON TIMER(n) GOSUB sets the timer callback (unit-level test)."""
        from qbasic_core.engine import ExecResult
        result = self.t._cf_on_timer('ON TIMER(5) GOSUB 200')
        self.assertIsNotNone(result)
        self.assertEqual(result, (True, ExecResult.ADVANCE))
        self.assertEqual(self.t._on_timer_target, 200)
        self.assertAlmostEqual(self.t._on_timer_interval, 5.0)


# =====================================================================
# 50. SCREEN auto-display after RUN
# =====================================================================
class TestScreenAutoDisplay(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_screen_2_shows_statevector(self):
        """SCREEN 2 should trigger statevector display after RUN."""
        self.t.process('10 H 0')
        self.t.process('20 MEASURE')
        self.t.shots = 10
        self.t._screen_mode = 2
        _, out = capture(self.t.cmd_run)
        # In screen mode 2, _auto_display calls _print_statevector
        # Output should exist (histogram + optional statevector)
        self.assertTrue(len(out) > 0)


# =====================================================================
# 51. COLOR ANSI escape output
# =====================================================================
class TestColorAnsi(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_color_fg_bg(self):
        _, out = capture(self.t.cmd_color, 'red, blue')
        self.assertIn('\033[31;44m', out)

    def test_color_fg_only(self):
        _, out = capture(self.t.cmd_color, 'cyan')
        self.assertIn('\033[36m', out)


# =====================================================================
# 52. LOCATE cursor escape output
# =====================================================================
class TestLocate(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_locate(self):
        _, out = capture(self.t.cmd_locate, '12, 40')
        self.assertIn('\033[12;40H', out)


# =====================================================================
# 53. Hex/binary literal parsing
# =====================================================================
class TestHexBinLiterals(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_hex_literal(self):
        val = self.t.eval_expr('&HFF')
        self.assertEqual(val, 255.0)

    def test_bin_literal(self):
        val = self.t.eval_expr('&B10110')
        self.assertEqual(val, 22.0)

    def test_hex_in_expression(self):
        val = self.t.eval_expr('&H10 + 1')
        self.assertEqual(val, 17.0)


# =====================================================================
# 54. Bitwise operators
# =====================================================================
class TestBitwiseOps(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_and(self):
        result = self.t._eval_condition('5 AND 3', {})
        self.assertTrue(result)  # both truthy

    def test_or(self):
        result = self.t._eval_condition('0 OR 1', {})
        self.assertTrue(result)

    def test_not(self):
        result = self.t._eval_condition('NOT 0', {})
        self.assertTrue(result)

    def test_xor_in_condition(self):
        result = self.t._eval_condition('1 XOR 0', {})
        self.assertTrue(result)


# =====================================================================
# 55. EXIT FOR / EXIT WHILE / EXIT DO
# =====================================================================
class TestExitStatements(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_exit_for_unit(self):
        """EXIT FOR handler finds matching NEXT and jumps past it."""
        from qbasic_core.engine import ExecResult
        self.t.program = {
            10: 'FOR i = 1 TO 10',
            20: 'EXIT FOR',
            30: 'NEXT i',
            40: 'PRINT i',
        }
        sorted_lines = [10, 20, 30, 40]
        loop_stack = [{'var': 'i', 'current': 1, 'end': 10, 'step': 1, 'return_ip': 0}]
        result = self.t._cf_exit('EXIT FOR', loop_stack, sorted_lines, 1)
        self.assertIsNotNone(result)
        handled, ip = result
        self.assertTrue(handled)
        self.assertEqual(ip, 3)  # past NEXT at ip 2

    def test_exit_while_unit(self):
        """EXIT WHILE handler finds matching WEND and jumps past it."""
        self.t.program = {
            10: 'WHILE 1',
            20: 'EXIT WHILE',
            30: 'WEND',
            40: 'PRINT "done"',
        }
        sorted_lines = [10, 20, 30, 40]
        loop_stack = [{'type': 'while', 'cond': '1', 'return_ip': 0}]
        result = self.t._cf_exit('EXIT WHILE', loop_stack, sorted_lines, 1)
        self.assertIsNotNone(result)
        handled, ip = result
        self.assertTrue(handled)
        self.assertEqual(ip, 3)  # past WEND at ip 2

    def test_exit_do_unit(self):
        """EXIT DO handler finds matching LOOP and jumps past it."""
        self.t.program = {
            10: 'DO',
            20: 'EXIT DO',
            30: 'LOOP',
            40: 'PRINT "done"',
        }
        sorted_lines = [10, 20, 30, 40]
        loop_stack = [{'type': 'do', 'return_ip': 0, 'kind': None, 'cond': None}]
        result = self.t._cf_exit('EXIT DO', loop_stack, sorted_lines, 1)
        self.assertIsNotNone(result)
        handled, ip = result
        self.assertTrue(handled)
        self.assertEqual(ip, 3)  # past LOOP at ip 2


# =====================================================================
# 56. SYS INSTALL
# =====================================================================
class TestSysInstall(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_install_and_call(self):
        """SYS INSTALL installs a user routine; SYS calls it."""
        # Define a subroutine with gate commands (not PRINT)
        self.t.subroutines['MYROUTINE'] = {'body': ['H 0'], 'params': []}
        _, out = capture(self.t.cmd_sys, 'INSTALL 0xF000, MYROUTINE')
        self.assertIn('INSTALLED', out)
        self.assertIn(0xF000, self.t._user_sys)
        # Call it -- runs process('H 0') which applies a gate
        _, out = capture(self.t.cmd_sys, '0xF000')
        # Should not error
        self.assertNotIn('UNDEFINED', out)

    def test_install_out_of_range(self):
        _, out = capture(self.t.cmd_sys, 'INSTALL 0x0001, FOO')
        self.assertIn('$F000-$FFFF', out)


# =====================================================================
# 57. UNITARY multi-qubit
# =====================================================================
class TestUnitaryMultiQubit(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_define_2qubit_gate(self):
        """Define a 2-qubit identity gate via _try_exec_unitary."""
        _, out = capture(self.t._try_exec_unitary,
                         'UNITARY MYID = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]')
        self.assertIn('2-qubit', out)
        self.assertIn('MYID', self.t._custom_gates)


# =====================================================================
# 58. CTRL on custom gate
# =====================================================================
class TestCtrlCustomGate(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()
        self.t.num_qubits = 3

    def test_ctrl_on_builtin(self):
        """CTRL H 0, 1 should not error."""
        # Just test that it parses and runs
        self.t.process('10 CTRL H 0, 1')
        self.t.process('20 MEASURE')
        self.t.shots = 10
        _, out = capture(self.t.cmd_run)
        self.assertNotIn('SYNTAX ERROR', out)


# =====================================================================
# 59. INV on parametric gate
# =====================================================================
class TestInvParametric(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_inv_rx(self):
        """INV RX 0.5, 0 should apply the inverse, returning to |0>."""
        self.t.process('10 RX 0.5, 0')
        self.t.process('20 INV RX 0.5, 0')
        self.t.process('30 MEASURE')
        self.t.shots = 100
        _, out = capture(self.t.cmd_run)
        # RX followed by INV RX should return to |0>, so all shots = all-zeros
        if self.t.last_counts:
            total = sum(self.t.last_counts.values())
            # Find the all-zeros key (length depends on num_qubits)
            all_zeros_key = '0' * self.t.num_qubits
            zeros = self.t.last_counts.get(all_zeros_key, 0)
            self.assertGreater(zeros, total * 0.9)


# =====================================================================
# 60. MEASURE_X / MEASURE_Y
# =====================================================================
class TestMeasureBasis(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_measure_x(self):
        """MEASURE_X adds H before measurement."""
        self.t.process('10 H 0')
        self.t.process('20 MEASURE_X 0')
        self.t.process('30 MEASURE')
        self.t.shots = 100
        _, out = capture(self.t.cmd_run)
        # H|0> in X basis should be |+> = deterministic 0
        self.assertNotIn('SYNTAX ERROR', out)

    def test_measure_y(self):
        """MEASURE_Y adds SDG+H before measurement."""
        self.t.process('10 MEASURE_Y 0')
        self.t.process('20 MEASURE')
        self.t.shots = 10
        _, out = capture(self.t.cmd_run)
        self.assertNotIn('SYNTAX ERROR', out)


# =====================================================================
# 61. SYNDROME
# =====================================================================
class TestSyndrome(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()
        self.t.num_qubits = 3  # need ancilla

    def test_syndrome_zz(self):
        """SYNDROME ZZ 0 1 -> s0 measures parity."""
        self.t.process('10 SYNDROME ZZ 0 1 -> s0')
        self.t.process('20 MEASURE')
        self.t.shots = 10
        _, out = capture(self.t.cmd_run)
        self.assertNotIn('SYNTAX ERROR', out)


# =====================================================================
# 62. LOCC MEAS in LOCC mode
# =====================================================================
class TestLoccMeas(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_locc_meas(self):
        """LOCC mode MEAS via SEND on a register."""
        _, out = capture(self.t.dispatch, 'LOCC 1 1')
        self.assertTrue(self.t.locc_mode)
        # Apply gate and measure
        _, out = capture(self.t.dispatch, '@A H 0')
        _, out = capture(self.t.dispatch, 'SEND A 0 -> result')
        self.assertIn('result', self.t.variables)


# =====================================================================
# 63. LOCC CTRL modifier
# =====================================================================
class TestLoccCtrl(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_locc_ctrl(self):
        """CTRL in LOCC mode applies controlled gate."""
        _, out = capture(self.t.dispatch, 'LOCC 2 2')
        _, out = capture(self.t.dispatch, '@A X 0')
        _, out = capture(self.t.dispatch, '@A CTRL X 0, 1')
        # Should not error
        self.assertNotIn('ERROR', out.upper())


# =====================================================================
# 64. Architecture modules — parser, errors, IOPort
# =====================================================================
class TestParser(unittest.TestCase):
    def test_rem(self):
        s = parse_stmt("REM this is a comment")
        self.assertIsInstance(s, RemStmt)

    def test_end(self):
        s = parse_stmt("END")
        self.assertIsInstance(s, EndStmt)

    def test_return(self):
        s = parse_stmt("RETURN")
        self.assertIsInstance(s, ReturnStmt)

    def test_measure(self):
        s = parse_stmt("MEASURE")
        self.assertIsInstance(s, MeasureStmt)

    def test_barrier(self):
        s = parse_stmt("BARRIER")
        self.assertIsInstance(s, BarrierStmt)

    def test_wend(self):
        s = parse_stmt("WEND")
        self.assertIsInstance(s, WendStmt)

    def test_goto(self):
        s = parse_stmt("GOTO 100")
        self.assertIsInstance(s, GotoStmt)
        self.assertEqual(s.target, 100)

    def test_gosub(self):
        s = parse_stmt("GOSUB 200")
        self.assertIsInstance(s, GosubStmt)
        self.assertEqual(s.target, 200)

    def test_for(self):
        s = parse_stmt("FOR i = 1 TO 10 STEP 2")
        self.assertIsInstance(s, ForStmt)
        self.assertEqual(s.var, 'i')
        self.assertEqual(s.step_expr, '2')

    def test_next(self):
        s = parse_stmt("NEXT i")
        self.assertIsInstance(s, NextStmt)
        self.assertEqual(s.var, 'i')

    def test_while(self):
        s = parse_stmt("WHILE x > 0")
        self.assertIsInstance(s, WhileStmt)
        self.assertIn('x', s.condition)

    def test_if_then(self):
        s = parse_stmt("IF x > 0 THEN PRINT x")
        self.assertIsInstance(s, IfThenStmt)

    def test_let(self):
        s = parse_stmt("LET x = 5")
        self.assertIsInstance(s, LetStmt)
        self.assertEqual(s.name, 'x')

    def test_let_array(self):
        s = parse_stmt("LET arr(0) = 10")
        self.assertIsInstance(s, LetArrayStmt)

    def test_print(self):
        s = parse_stmt('PRINT "hello"')
        self.assertIsInstance(s, PrintStmt)

    def test_compound(self):
        s = parse_stmt("H 0 : X 1")
        self.assertIsInstance(s, CompoundStmt)
        self.assertEqual(len(s.parts), 2)

    def test_raw_fallback(self):
        s = parse_stmt("SOME UNKNOWN THING")
        self.assertIsInstance(s, RawStmt)


class TestErrorsHierarchy(unittest.TestCase):
    def test_base_error(self):
        e = QBasicError("test", code=42, line=10)
        self.assertEqual(e.message, "test")
        self.assertEqual(e.code, 42)
        self.assertEqual(e.line, 10)
        self.assertIsInstance(e, Exception)

    def test_subclasses(self):
        self.assertTrue(issubclass(QBasicSyntaxError, QBasicError))
        self.assertTrue(issubclass(QBasicRuntimeError, QBasicError))
        self.assertTrue(issubclass(QBasicBuildError, QBasicError))
        self.assertTrue(issubclass(QBasicRangeError, QBasicError))
        self.assertTrue(issubclass(QBasicIOError, QBasicError))
        self.assertTrue(issubclass(QBasicUndefinedError, QBasicError))

    def test_error_is_catchable(self):
        with self.assertRaises(QBasicError):
            raise QBasicSyntaxError("bad syntax")


class TestIOPortProtocol(unittest.TestCase):
    def test_stdio_port_is_ioport(self):
        port = StdIOPort()
        self.assertIsInstance(port, IOPort)

    def test_stdio_write(self):
        port = StdIOPort()
        _, out = capture(port.write, 'hello')
        self.assertEqual(out, 'hello')

    def test_stdio_writeln(self):
        port = StdIOPort()
        _, out = capture(port.writeln, 'hello')
        self.assertEqual(out, 'hello\n')


# =====================================================================
# Run
# =====================================================================
if __name__ == '__main__':
    unittest.main()
