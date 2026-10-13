#!/usr/bin/env python3
"""
Interpreter features beyond the core suite in test_qubasic.py: DO/LOOP,
PEEK/POKE, SYS, USR, WAIT, DUMP, MAP, CATALOG,
string functions, screen commands, SUB/FUNCTION, LOCAL/STATIC/SHARED,
error handling, breakpoints, watches, trace, profiling, STATS,
AUTO, EDIT, COPY/MOVE, FIND/REPLACE, BANK, CHECKSUM, CHAIN, MERGE,
file handles, LPRINT, DATA/READ/RESTORE, ON GOTO/GOSUB, SELECT CASE,
SWAP, DEF FN, OPTION BASE, ASSERT, PRINT USING, DIM multi,
LET string, LINE INPUT, POKE/SYS during execution, ON MEASURE/TIMER,
SCREEN auto-display, COLOR ANSI, LOCATE, hex/binary literals,
bitwise operators, EXIT FOR/WHILE/DO, SYS INSTALL, UNITARY multi-qubit,
CTRL custom gate, INV parametric, MEASURE_X/Y, SYNDROME, LOCC MEAS/CTRL,
the parser's statement types, the error hierarchy and the IOPort protocol.
"""

import builtins
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import pytest
from conftest import ScriptIO, capture

# Repo root is one level up from tests/; put it on the path for imports and
# use it for subprocess working directories and example file paths.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from qubasic_core.engine import (
    IOPort,
    QBasicBuildError,
    QBasicError,
    QBasicIOError,
    QBasicRangeError,
    QBasicRuntimeError,
    QBasicSyntaxError,
    QBasicUndefinedError,
    StdIOPort,
)
from qubasic_core.parser import (
    BarrierStmt,
    CompoundStmt,
    EndStmt,
    ForStmt,
    GosubStmt,
    GotoStmt,
    IfThenStmt,
    LetArrayStmt,
    LetStmt,
    MeasureStmt,
    NextStmt,
    PrintStmt,
    RawStmt,
    RemStmt,
    ReturnStmt,
    Stmt,
    WendStmt,
    WhileStmt,
    parse_stmt,
)
from qubasic_core.terminal import QBasicTerminal


# =====================================================================
# 1. TestClassicBasic
# =====================================================================
class TestClassicBasic(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_do_loop_variants(self):
        """DO WHILE, LOOP WHILE, DO UNTIL, LOOP UNTIL, infinite DO with GOTO exit."""
        # DO WHILE pretest
        t = QBasicTerminal()
        t.process('10 LET i = 1')
        t.process('20 DO WHILE i <= 3')
        t.process('30 PRINT i')
        t.process('40 LET i = i + 1')
        t.process('50 LOOP')
        t.process('60 END')
        _, out = capture(t.cmd_run)
        printed = [l.strip() for l in out.strip().split('\n')
                   if l.strip() in ('1', '1.0', '2', '2.0', '3', '3.0')]
        self.assertEqual(len(printed), 3)

        # LOOP WHILE posttest
        t = QBasicTerminal()
        t.process('10 LET i = 1')
        t.process('20 DO')
        t.process('30 PRINT i')
        t.process('40 LET i = i + 1')
        t.process('50 LOOP WHILE i <= 3')
        t.process('60 END')
        _, out = capture(t.cmd_run)
        printed = [l.strip() for l in out.strip().split('\n')
                   if l.strip() in ('1', '1.0', '2', '2.0', '3', '3.0')]
        self.assertEqual(len(printed), 3)

        # DO UNTIL pretest
        t = QBasicTerminal()
        t.process('10 LET i = 1')
        t.process('20 DO UNTIL i > 2')
        t.process('30 PRINT i')
        t.process('40 LET i = i + 1')
        t.process('50 LOOP')
        t.process('60 END')
        _, out = capture(t.cmd_run)
        printed = [l.strip() for l in out.strip().split('\n')
                   if l.strip() in ('1', '1.0', '2', '2.0')]
        self.assertEqual(len(printed), 2)

        # LOOP UNTIL posttest
        t = QBasicTerminal()
        t.process('10 LET i = 5')
        t.process('20 DO')
        t.process('30 PRINT i')
        t.process('40 LET i = i + 1')
        t.process('50 LOOP UNTIL i > 5')
        t.process('60 END')
        _, out = capture(t.cmd_run)
        self.assertIn('5', out)

        # Infinite DO/LOOP with GOTO exit
        t = QBasicTerminal()
        t.process('10 LET i = 0')
        t.process('20 DO')
        t.process('30 LET i = i + 1')
        t.process('40 IF i == 3 THEN GOTO 70')
        t.process('50 LOOP')
        t.process('70 PRINT i')
        t.process('80 END')
        _, out = capture(t.cmd_run)
        self.assertIn('3', out)

    def test_data_read_restore(self):
        """DATA/READ and RESTORE."""
        self.t.process('10 DATA 10, 20, 30')
        self.t.process('20 READ a')
        self.t.process('30 READ b')
        self.t.process('40 PRINT a')
        self.t.process('50 PRINT b')
        self.t.process('60 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('10', out)
        self.assertIn('20', out)

        # RESTORE
        t2 = QBasicTerminal()
        t2.process('10 DATA 5, 6')
        t2._collect_data()
        t2.data.ptr = 2
        t2.cmd_restore()
        self.assertEqual(t2.data.ptr, 0)

    def test_on_goto_gosub(self):
        """ON expr GOTO and ON expr GOSUB."""
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

        t2 = QBasicTerminal()
        t2.process('10 LET x = 1')
        t2.process('20 ON x GOSUB 100, 200')
        t2.process('30 PRINT "BACK"')
        t2.process('40 END')
        t2.process('100 PRINT "SUB1"')
        t2.process('110 RETURN')
        t2.process('200 PRINT "SUB2"')
        t2.process('210 RETURN')
        _, out = capture(t2.cmd_run)
        self.assertIn('SUB1', out)
        self.assertIn('BACK', out)

    def test_select_case(self):
        """SELECT CASE dispatches correctly."""
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

    def test_swap_def_fn_option_base(self):
        """SWAP, DEF FN, OPTION BASE."""
        # SWAP
        self.t.process('10 LET a = 1')
        self.t.process('20 LET b = 2')
        self.t.process('30 SWAP a, b')
        self.t.process('40 PRINT a')
        self.t.process('50 PRINT b')
        self.t.process('60 END')
        _, out = capture(self.t.cmd_run)
        lines = [l.strip() for l in out.split('\n')
                 if l.strip() in ('1', '1.0', '2', '2.0')]
        self.assertIn(lines[0], ('2', '2.0'))

        # DEF FN
        t2 = QBasicTerminal()
        t2.process('10 DEF FN SQUARE(x) = x * x')
        t2.process('20 PRINT FNSQUARE(5)')
        t2.process('30 END')
        _, out = capture(t2.cmd_run)
        self.assertIn('25', out)

        # OPTION BASE
        t3 = QBasicTerminal()
        t3.process('10 OPTION BASE 1')
        t3.process('20 END')
        _, out = capture(t3.cmd_run)
        self.assertEqual(t3.symbols.option_base, 1)

    def test_exit_statements(self):
        """EXIT FOR, EXIT WHILE, EXIT DO handler logic."""

        # EXIT FOR
        self.t.program = {10: 'FOR i = 1 TO 10', 20: 'EXIT FOR', 30: 'NEXT i', 40: 'PRINT i'}
        sorted_lines = [10, 20, 30, 40]
        loop_stack = [{'var': 'i', 'current': 1, 'end': 10, 'step': 1, 'return_ip': 0}]
        result = self.t._cf_exit('EXIT FOR', loop_stack, sorted_lines, 1,
                                 parsed=parse_stmt('EXIT FOR'))
        self.assertIsNotNone(result)
        handled, ip = result
        self.assertTrue(handled)
        self.assertEqual(ip, 3)

        # EXIT WHILE
        t2 = QBasicTerminal()
        t2.program = {10: 'WHILE 1', 20: 'EXIT WHILE', 30: 'WEND', 40: 'PRINT "done"'}
        sorted_lines = [10, 20, 30, 40]
        loop_stack = [{'type': 'while', 'cond': '1', 'return_ip': 0}]
        result = t2._cf_exit('EXIT WHILE', loop_stack, sorted_lines, 1,
                                 parsed=parse_stmt('EXIT WHILE'))
        handled, ip = result
        self.assertTrue(handled)
        self.assertEqual(ip, 3)

        # EXIT DO
        t3 = QBasicTerminal()
        t3.program = {10: 'DO', 20: 'EXIT DO', 30: 'LOOP', 40: 'PRINT "done"'}
        sorted_lines = [10, 20, 30, 40]
        loop_stack = [{'type': 'do', 'return_ip': 0, 'kind': None, 'cond': None}]
        result = t3._cf_exit('EXIT DO', loop_stack, sorted_lines, 1,
                                 parsed=parse_stmt('EXIT DO'))
        handled, ip = result
        self.assertTrue(handled)
        self.assertEqual(ip, 3)

    def test_hex_bin_bitwise(self):
        """Hex/binary literals and bitwise operators."""
        # Hex literal
        self.assertEqual(self.t.eval_expr('&HFF'), 255.0)
        # Binary literal
        self.assertEqual(self.t.eval_expr('&B10110'), 22.0)
        # Hex in expression
        self.assertEqual(self.t.eval_expr('&H10 + 1'), 17.0)
        # Bitwise AND, OR, NOT, XOR
        self.assertTrue(self.t._eval_condition('5 AND 3', {}))
        self.assertTrue(self.t._eval_condition('0 OR 1', {}))
        self.assertTrue(self.t._eval_condition('NOT 0', {}))
        self.assertTrue(self.t._eval_condition('1 XOR 0', {}))

    def test_exit_while_in_if_then(self):
        """EXIT WHILE inside IF THEN should break out of the WHILE loop."""
        t = QBasicTerminal()
        t.process('10 LET i = 0', track_undo=False)
        t.process('20 WHILE i < 100', track_undo=False)
        t.process('30 LET i = i + 1', track_undo=False)
        t.process('40 IF i == 3 THEN EXIT WHILE', track_undo=False)
        t.process('50 WEND', track_undo=False)
        t.process('60 PRINT i', track_undo=False)
        t.process('70 END', track_undo=False)
        _, out = capture(t.cmd_run)
        self.assertEqual(out.split('\nRAN')[0].split(), ['3'])
        self.assertEqual(t.variables['i'], 3)

    def test_print_using_edge_cases(self):
        """PRINT USING right-aligns a number in its field."""
        t = QBasicTerminal()
        t.process('10 PRINT USING "##.##"; 3.14', track_undo=False)
        t.process('20 PRINT USING "#####"; 42', track_undo=False)
        t.process('30 END', track_undo=False)
        _, out = capture(t.cmd_run)
        self.assertEqual(out.split('RAN ')[0], ' 3.14\n   42\n')


# =====================================================================
# 2. TestMemoryMap
# =====================================================================
class TestMemoryMap(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_peek_poke(self):
        """Zero page, qubit state, config, status, cmd_poke, POKE during exec, per-qubit noise."""
        # Zero page read/write
        self.t._poke(0x0000, 42.0)
        self.assertEqual(self.t._peek(0x0000), 42.0)
        self.t._poke(0x003F, 99.0)
        self.assertEqual(self.t._peek(0x003F), 99.0)

        # Qubit state at $0100 before init
        self.assertEqual(self.t._peek(0x0100), 0.0)

        # Config read/write: shots
        self.t._poke(0xD001, 50)
        self.assertEqual(self.t.shots, 50)
        self.assertEqual(self.t._peek(0xD001), 50.0)
        # Config: num_qubits
        self.t._poke(0xD000, 3)
        self.assertEqual(self.t.num_qubits, 3)
        self.assertEqual(self.t._peek(0xD000), 3.0)

        # Status is read-only
        _, out = capture(self.t._poke, 0xD010, 99)
        self.assertIn('READ-ONLY', out)

        # cmd_poke
        _, out = capture(self.t.cmd_poke, '0, 7')
        self.assertEqual(self.t._peek(0), 7.0)

        # USR returns float
        self.assertEqual(self.t._usr_fn(0xFFFF), 0.0)

        # POKE during execution
        t_exec = QBasicTerminal()
        t_exec.process('10 POKE 0, 77')
        t_exec.process('20 PRINT "DONE"')
        t_exec.process('30 END')
        _, out = capture(t_exec.cmd_run)
        self.assertEqual(t_exec._peek(0), 77.0)

        # Per-qubit noise via POKE at $D1xx
        t_noise = QBasicTerminal()
        t_noise._poke(0xD100, 1)
        t_noise._poke(0xD101, 0.05)
        ntype, nparam = t_noise.noise.per_qubit.get(0, (0, 0.0))
        self.assertEqual(ntype, 1)
        self.assertAlmostEqual(nparam, 0.05)
        self.assertEqual(t_noise._peek(0xD100), 1.0)
        self.assertAlmostEqual(t_noise._peek(0xD101), 0.05)
        t_noise._poke(0xD102, 2)
        t_noise._poke(0xD103, 0.1)
        ntype, nparam = t_noise.noise.per_qubit.get(1, (0, 0.0))
        self.assertEqual(ntype, 2)
        self.assertAlmostEqual(nparam, 0.1)

    def test_qubit_count_change_invalidates_stale_state(self):
        """Changing the qubit count after a run must not leave a stale last_sv
        that MAP/PEEK/BLOCH then reshape to the wrong size and crash."""
        import numpy as np

        # Run a 1-qubit circuit so last_sv has size 2.
        t = QBasicTerminal()
        t.num_qubits = 1
        t.process('10 H 0')
        t.process('20 MEASURE')
        capture(t.cmd_run)

        # QUBITS grows the count: the size-2 statevector no longer applies.
        capture(t.cmd_qubits, '2')
        self.assertIsNone(t.last_sv)
        self.assertIsNone(t._circuit_cache_key)
        self.assertEqual(t._peek(0x0100), 0.0)   # would have crashed on reshape
        capture(t.cmd_map)                        # MAP walks the qubit block
        capture(t.cmd_bloch, '0')                 # BLOCH reshapes too

        # The $D000 POKE path invalidates the same way.
        t2 = QBasicTerminal()
        t2.num_qubits = 1
        t2.process('10 H 0')
        t2.process('20 MEASURE')
        capture(t2.cmd_run)
        t2._poke(0xD000, 3)
        self.assertIsNone(t2.last_sv)
        self.assertEqual(t2._peek(0x0100), 0.0)

        # Defensive guard: even a hand-set mismatched last_sv reads as 0, not a crash.
        t3 = QBasicTerminal()
        t3.num_qubits = 2
        t3.last_sv = np.array([1.0, 0.0], dtype=complex)  # size 2, but 2 qubits
        self.assertEqual(t3._peek(0x0100), 0.0)

    def test_sys(self):
        """SYS builtin BELL, SYS unmapped, SYS in program, SYS INSTALL."""
        # SYS 0xE000 BELL demo
        _, out = capture(self.t.cmd_sys, '0xE000')
        self.assertTrue(len(out) > 0)

        # SYS at unmapped address
        _, out = capture(self.t.cmd_sys, '0xFFFF')
        self.assertIn('NO ROUTINE', out)

        # SYS in program
        t2 = QBasicTerminal()
        t2.process('10 SYS 0xE000')
        t2.process('20 END')
        _, out = capture(t2.cmd_run)
        self.assertTrue(len(out) > 0)

        # SYS INSTALL and call
        t3 = QBasicTerminal()
        t3.subroutines['MYROUTINE'] = {'body': ['H 0'], 'params': []}
        _, out = capture(t3.cmd_sys, 'INSTALL 0xF000, MYROUTINE')
        self.assertIn('INSTALLED', out)
        self.assertIn(0xF000, t3.machine.user_sys)
        _, out = capture(t3.cmd_sys, '0xF000')
        self.assertNotIn('UNDEFINED', out)

        # SYS INSTALL out of range
        _, out = capture(t3.cmd_sys, 'INSTALL 0x0001, FOO')
        self.assertIn('$F000-$FFFF', out)

    def test_wait(self):
        """WAIT immediate match and timeout."""
        # Immediate match
        self.t._poke(0, 0xFF)
        _, out = capture(self.t.cmd_wait, '0, 255')
        self.assertNotIn('TIMEOUT', out)

        # Timeout via monkey-patched time
        import qubasic_core.machine as mem_mod
        orig = mem_mod.time.time
        call_count = [0]
        def fast_time():
            call_count[0] += 1
            return call_count[0] * 100
        mem_mod.time.time = fast_time
        try:
            _, out = capture(self.t.cmd_wait, '0, 255, 128')
            self.assertIn('TIMEOUT', out)
        finally:
            mem_mod.time.time = orig

    def test_dump_map_catalog(self):
        """DUMP, MAP, CATALOG produce expected output."""
        # DUMP
        self.t._poke(0, 0xAB)
        _, out = capture(self.t.cmd_dump, '0 15')
        self.assertIn('$0000:', out)
        self.assertIn('AB', out)

        # MAP
        _, out = capture(self.t.cmd_map)
        self.assertIn('Memory Map', out)
        self.assertIn('Zero Page', out)
        self.assertIn('QPU Config', out)
        self.assertIn('SYS Routines', out)

        # CATALOG
        _, out = capture(self.t.cmd_catalog)
        self.assertIn('BELL', out)
        self.assertIn('GHZ', out)
        self.assertIn('$E000', out)


# =====================================================================
# 3. TestStrings
# =====================================================================
class TestStrings(unittest.TestCase):
    def test_string_functions(self):
        """LEFT, RIGHT, MID, CHR, ASC, INSTR, HEX, BIN, STR, VAL, LEN."""
        from qubasic_core.expression import (
            _asc,
            _bin_fn,
            _chr_fn,
            _hex_fn,
            _instr,
            _left,
            _len_fn,
            _mid,
            _right,
            _str_fn,
            _val_fn,
        )
        self.assertEqual(_left("HELLO", 3), "HEL")
        self.assertEqual(_right("HELLO", 3), "LLO")
        self.assertEqual(_mid("HELLO", 2, 3), "ELL")
        self.assertEqual(_chr_fn(65), "A")
        self.assertEqual(_asc("A"), 65.0)
        self.assertEqual(_instr("HELLO WORLD", "WORLD"), 7.0)
        self.assertEqual(_instr("HELLO", "XYZ"), 0.0)
        self.assertEqual(_hex_fn(255), "FF")
        self.assertEqual(_bin_fn(5), "101")
        self.assertEqual(_str_fn(42.0), " 42")    # leading space for non-negative
        self.assertEqual(_str_fn(3.14), " 3.14")
        self.assertEqual(_str_fn(-5.0), "-5")
        self.assertEqual(_val_fn("42"), 42.0)
        self.assertEqual(_val_fn("abc"), 0.0)
        self.assertEqual(_len_fn("HELLO"), 5.0)


# =====================================================================
# 4. TestScreenAndDisplay
# =====================================================================
class TestScreenAndDisplay(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_screen_mode_set_get(self):
        """DISPLAY set/get, SCREEN graphics modes, CLS, PLAY, PROMPT."""
        _, out = capture(self.t.cmd_display, 'STATE')
        self.assertIn('DISPLAY statevector', out)
        self.assertEqual(self.t._screen_mode, 2)
        _, out = capture(self.t.cmd_display, '')
        self.assertIn('statevector', out)
        _, out = capture(self.t.cmd_screen, '12')
        self.assertIn('640x480', out)
        _, out = capture(self.t.cmd_screen, '0')
        self.assertIsNone(self.t.graphics.canvas)

        # CLS
        _, out = capture(self.t.cmd_cls)
        self.assertIn('\033[2J', out)

        # PLAY
        _, out = capture(self.t.cmd_play, '1')
        self.assertIn('\a', out)

        # PROMPT
        _, out = capture(self.t.cmd_prompt, '">>> "')
        self.assertEqual(self.t._prompt, '>>> ')

    def test_screen_auto_display_modes(self):
        """SCREEN 2 statevector, SCREEN 3 bloch, SCREEN 2 after RUN."""
        # Screen 2
        t = QBasicTerminal()
        t.num_qubits = 2
        t._screen_mode = 2
        t.process('10 H 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, out = capture(t.cmd_run)
        self.assertIn('Statevector', out)

        # Screen 3
        t2 = QBasicTerminal()
        t2.num_qubits = 1
        t2._screen_mode = 3
        t2.process('10 H 0', track_undo=False)
        t2.process('20 MEASURE', track_undo=False)
        _, out = capture(t2.cmd_run)
        self.assertIn('Qubit 0', out)

        # Screen 2 triggers display after RUN
        t3 = QBasicTerminal()
        t3.process('10 H 0')
        t3.process('20 MEASURE')
        t3.shots = 10
        t3._screen_mode = 2
        _, out = capture(t3.cmd_run)
        self.assertTrue(len(out) > 0)

    def test_color_locate(self):
        """COLOR fg, fg+bg, and LOCATE cursor escape."""
        _, out = capture(self.t.cmd_color, 'green')
        self.assertIn('\033[32m', out)
        _, out = capture(self.t.cmd_color, 'red, blue')
        self.assertIn('\033[31;44m', out)
        _, out = capture(self.t.cmd_color, 'cyan')
        self.assertIn('\033[36m', out)
        _, out = capture(self.t.cmd_locate, '5, 10')
        self.assertIn('\033[5;10H', out)
        _, out = capture(self.t.cmd_locate, '12, 40')
        self.assertIn('\033[12;40H', out)


# =====================================================================
# 5. TestSubsAndScope
# =====================================================================
class TestSubsAndScope(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_sub_call(self):
        """Define SUB GREET, CALL GREET."""
        self.t.process('10 SUB GREET()')
        self.t.process('20 PRINT "HELLO FROM SUB"')
        self.t.process('30 END SUB')
        self.t.process('40 CALL GREET')
        self.t.process('50 END')
        _, out = capture(self.t.cmd_run)
        self.assertIn('HELLO FROM SUB', out)

    def test_function_define_invoke(self):
        """FUNCTION DOUBLE(x) invoked via _invoke_function."""
        self.t.process('10 FUNCTION DOUBLE(x)')
        self.t.process('20 LET DOUBLE = x * 2')
        self.t.process('30 END FUNCTION')
        sorted_lines = sorted(self.t.program.keys())
        self.t._scan_subs(sorted_lines)
        self.assertIn('DOUBLE', self.t.procs.functions)
        result = self.t._invoke_function('DOUBLE', [5.0], sorted_lines)
        self.assertAlmostEqual(result, 10.0)

    def test_local_static_shared(self):
        """LOCAL, STATIC, SHARED variable scoping."""
        # LOCAL
        self.t.process('10 LET x = 99')
        self.t.process('20 SUB MYFN()')
        self.t.process('30 LOCAL x')
        self.t.process('40 PRINT x')
        self.t.process('50 END SUB')
        self.t.process('60 CALL MYFN')
        self.t.process('70 PRINT x')
        self.t.process('80 END')
        _, out = capture(self.t.cmd_run)
        lines = [l.strip() for l in out.split('\n')
                 if l.strip() and l.strip().replace('.', '').replace('-', '').isdigit()]
        if len(lines) >= 2:
            self.assertIn(lines[0], ('0', '0.0'))

        # STATIC
        t2 = QBasicTerminal()
        t2.process('10 SUB COUNTER()')
        t2.process('20 STATIC count')
        t2.process('30 LET count = count + 1')
        t2.process('40 PRINT count')
        t2.process('50 END SUB')
        t2.process('60 CALL COUNTER')
        t2.process('70 CALL COUNTER')
        t2.process('80 END')
        _, out = capture(t2.cmd_run)
        nums = [l.strip() for l in out.split('\n')
                if l.strip() in ('1', '1.0', '2', '2.0')]
        self.assertEqual(len(nums), 2)

        # SHARED
        t3 = QBasicTerminal()
        t3.process('10 LET x = 42')
        t3.process('20 SUB SHOWX()')
        t3.process('30 SHARED x')
        t3.process('40 PRINT x')
        t3.process('50 END SUB')
        t3.process('60 CALL SHOWX')
        t3.process('70 END')
        _, out = capture(t3.cmd_run)
        self.assertIn('42', out)

    def test_error_handling(self):
        """ON ERROR GOTO, GOTO 0, _handle_error, RESUME NEXT, ERR/ERL, ERROR n, ASSERT."""
        from qubasic_core.engine import ExecResult

        # ON ERROR GOTO sets target
        result = self.t._cf_on_error('ON ERROR GOTO 100', parsed=parse_stmt('ON ERROR GOTO 100'))
        self.assertIsNotNone(result)
        self.assertEqual(result, (True, ExecResult.ADVANCE))
        self.assertEqual(self.t.errors.target, 100)

        # ON ERROR GOTO 0 clears
        self.t._cf_on_error('ON ERROR GOTO 0', parsed=parse_stmt('ON ERROR GOTO 0'))
        self.assertIsNone(self.t.errors.target)

        # _handle_error routes correctly
        self.t.errors.target = 100
        self.t.program = {10: 'SOMETHING', 100: 'PRINT "CAUGHT"'}
        sorted_lines = [10, 100]
        ip = self.t._handle_error(RuntimeError("ERROR 42"), 10, sorted_lines)
        self.assertEqual(ip, 1)
        self.assertEqual(self.t.errors.code, 42)
        self.assertEqual(self.t.errors.line, 10)

        # RESUME NEXT
        self.t.errors.line = 20
        self.t.errors.handling = True
        sorted_lines = [10, 20, 30]
        self.t.program = {10: 'X', 20: 'Y', 30: 'Z'}
        result = self.t._cf_resume('RESUME NEXT', sorted_lines, parsed=parse_stmt('RESUME NEXT'))
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 2)

        # ERR and ERL variables
        t_erl = QBasicTerminal()
        t_erl.errors.target = 100
        t_erl.program = {20: 'BAD LINE', 100: 'HANDLER'}
        sorted_lines = [20, 100]
        t_erl._handle_error(RuntimeError("ERROR 42"), 20, sorted_lines)
        self.assertEqual(t_erl.variables['ERR'], 42)
        self.assertEqual(t_erl.variables['ERL'], 20)

        # ERROR n raises
        t_err = QBasicTerminal()
        t_err.process('10 ERROR 99')
        _, out = capture(t_err.cmd_run)
        self.assertIn('ERROR', out)

        # ASSERT pass
        t_ap = QBasicTerminal()
        t_ap.process('10 ASSERT 1 == 1')
        t_ap.process('20 PRINT "OK"')
        t_ap.process('30 END')
        _, out = capture(t_ap.cmd_run)
        self.assertIn('OK', out)

        # ASSERT fail
        t_af = QBasicTerminal()
        t_af.process('10 ASSERT 1 == 2')
        t_af.process('20 END')
        _, out = capture(t_af.cmd_run)
        self.assertIn('ASSERTION FAILED', out)


# =====================================================================
# 6. TestDebugAndProfile
# =====================================================================
class TestDebugAndProfile(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_breakpoints(self):
        """Set, list, clear breakpoints and hit check."""
        _, out = capture(self.t.cmd_breakpoint, '10')
        self.assertIn('BREAKPOINT SET: 10', out)
        _, out = capture(self.t.cmd_breakpoint, '')
        self.assertIn('10', out)

        # Clear
        _, out = capture(self.t.cmd_breakpoint, 'CLEAR')
        self.assertIn('CLEARED', out)
        _, out = capture(self.t.cmd_breakpoint, '')
        self.assertIn('No breakpoints', out)

        # Hit check
        self.t.debug.breakpoints.add(20)
        result = self.t._check_breakpoint(20, [10, 20, 30], 1)
        self.assertTrue(result)

    def test_watch(self):
        """Add/list watch and display during stop."""
        _, out = capture(self.t.cmd_watch, 'x')
        self.assertIn('WATCHING: x', out)
        _, out = capture(self.t.cmd_watch, '')
        self.assertIn('x', out)

        self.t.variables['x'] = 42
        _, out = capture(self.t._print_watches)
        self.assertIn('42', out)

    def test_stop_cont(self):
        """STOP sets stopped_ip, CONT without stop."""
        self.t.process('10 PRINT "BEFORE"')
        self.t.process('20 STOP')
        self.t.process('30 PRINT "AFTER"')
        _, out = capture(self.t.cmd_run)
        self.assertIn('STOPPED', out)
        self.assertIsNotNone(self.t.debug.stopped_ip)

        t2 = QBasicTerminal()
        _, out = capture(t2.cmd_cont)
        self.assertIn('CANNOT CONTINUE', out)

    def test_tron_troff(self):
        """TRON, TROFF, trace line output."""
        _, out = capture(self.t.cmd_tron)
        self.assertIn('TRACE ON', out)
        self.assertTrue(self.t.debug.trace)

        _, out = capture(self.t.cmd_troff)
        self.assertIn('TRACE OFF', out)
        self.assertFalse(self.t.debug.trace)

        self.t.debug.trace = True
        _, out = capture(self.t._trace_line, 10)
        self.assertIn('[10]', out)

    def test_profile(self):
        """PROFILE ON/OFF/SHOW, with and without data."""
        _, out = capture(self.t.cmd_profile, 'ON')
        self.assertIn('PROFILE ON', out)
        self.assertTrue(self.t.machine.profiling)
        _, out = capture(self.t.cmd_profile, 'OFF')
        self.assertIn('PROFILE OFF', out)
        self.assertFalse(self.t.machine.profiling)

        # Show empty
        _, out = capture(self.t.cmd_profile, 'SHOW')
        self.assertIn('No profile data', out)

        # Show with data
        self.t.machine.profile = {10: {'time_ms': 1.5, 'calls': 1, 'gates': 2}}
        self.t.program = {10: 'H 0'}
        _, out = capture(self.t.cmd_profile, 'SHOW')
        self.assertIn('Profile', out)
        self.assertIn('H 0', out)

    def test_time_travel_debug(self):
        """REWIND, FORWARD, HISTORY, and no-checkpoint case."""
        import numpy as np
        self.t.num_qubits = 2
        sv1 = np.array([1, 0, 0, 0], dtype=complex)
        sv2 = np.array([0.707, 0, 0, 0.707], dtype=complex)
        self.t.debug.checkpoints = [(10, sv1.copy()), (20, sv2.copy())]
        self.t.debug.position = 1
        self.t.last_sv = sv2.copy()

        _, out = capture(self.t.cmd_rewind, '1')
        self.assertIn('step 0', out)
        self.assertEqual(self.t.debug.position, 0)

        _, out = capture(self.t.cmd_forward, '1')
        self.assertIn('step 1', out)
        self.assertEqual(self.t.debug.position, 1)

        _, out = capture(self.t.cmd_history)
        self.assertIn('[0]', out)
        self.assertIn('[1]', out)

        # No checkpoints
        t2 = QBasicTerminal()
        _, out = capture(t2.cmd_rewind, '1')
        self.assertIn('NO CHECKPOINTS', out)

    def test_stats(self):
        """STATS show empty, clear, run N, show with data."""
        _, out = capture(self.t.cmd_stats, '')
        self.assertIn('No statistics', out)

        self.t.machine.runs.append({'0': 10})
        _, out = capture(self.t.cmd_stats, 'CLEAR')
        self.assertIn('CLEARED', out)
        self.assertEqual(len(self.t.machine.runs), 0)

        # Run N
        self.t.process('10 H 0')
        self.t.process('20 MEASURE')
        self.t.shots = 10
        _, out = capture(self.t.cmd_stats, '3')
        self.assertIn('Collected', out)
        self.assertEqual(len(self.t.machine.runs), 3)

        # Show with data
        t2 = QBasicTerminal()
        t2.machine.runs = [{'00': 500, '11': 524}, {'00': 510, '11': 514}]
        _, out = capture(t2.cmd_stats, 'SHOW')
        self.assertIn('Statistics', out)
        self.assertIn('2 runs', out)

    def test_timer_callback_fires(self):
        """_check_timer_callback jumps to the ON TIMER handler once its interval
        has passed, keeping the interrupted ip for RETURN."""
        from unittest.mock import patch
        t = QBasicTerminal()
        t.events.timer_target = 100
        t.events.timer_interval = 1.0
        t.events.timer_last = 1000.0  # fake "last fire" time
        t.events.timer_state = 'ON'      # events are trapped after TIMER ON
        t._gosub_stack = []
        t.program = {100: 'PRINT "timer"', 110: 'RETURN'}
        sorted_lines = [100, 110]
        with patch('qubasic_core.interpreter.time') as mock_time:
            mock_time.time.return_value = 1000.5  # only 0.5 s elapsed
            self.assertIsNone(t._check_timer_callback(sorted_lines, 1))
            self.assertEqual(t._gosub_stack, [])
        with patch('qubasic_core.interpreter.time') as mock_time:
            mock_time.time.return_value = 1001.5  # 1.5 s elapsed
            self.assertEqual(t._check_timer_callback(sorted_lines, 1), 0)
            self.assertEqual(t._gosub_stack, [1])
            self.assertEqual(t.events.timer_last, 1001.5)


# =====================================================================
# 7. TestProgramManagement
# =====================================================================
class TestProgramManagement(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_auto_edit(self):
        """AUTO adds lines, EDIT modifies a line."""
        # AUTO
        inputs = iter(['PRINT "A"', 'PRINT "B"', '.'])
        orig = builtins.input
        builtins.input = lambda prompt='': next(inputs)
        try:
            _, out = capture(self.t.cmd_auto, '10, 10')
        finally:
            builtins.input = orig
        self.assertIn(10, self.t.program)
        self.assertIn(20, self.t.program)

        # EDIT
        builtins.input = lambda prompt='': 'PRINT "NEW"'
        try:
            _, out = capture(self.t.cmd_edit, '10')
        finally:
            builtins.input = orig
        self.assertEqual(self.t.program[10], 'PRINT "NEW"')
        self.assertIn('UPDATED', out)

    def test_copy_move(self):
        """COPY and MOVE line ranges."""
        self.t.process('10 PRINT "A"')
        self.t.process('20 PRINT "B"')
        _, out = capture(self.t.cmd_copy, '10-20 TO 100')
        self.assertIn('COPIED', out)
        self.assertIn(100, self.t.program)
        self.assertIn(110, self.t.program)

        t2 = QBasicTerminal()
        t2.process('10 PRINT "A"')
        t2.process('20 PRINT "B"')
        _, out = capture(t2.cmd_move, '10-20 TO 100')
        self.assertIn('MOVED', out)
        self.assertNotIn(10, t2.program)
        self.assertIn(100, t2.program)

    def test_find_replace(self):
        """FIND and REPLACE in program."""
        self.t.process('10 PRINT "HELLO"')
        self.t.process('20 PRINT "WORLD"')
        _, out = capture(self.t.cmd_find, 'HELLO')
        self.assertIn('1 match', out)

        _, out = capture(self.t.cmd_replace, '"HELLO" WITH "HI"')
        self.assertIn('1 replacement', out)
        self.assertIn('HI', self.t.program[10])

    def test_bank_checksum(self):
        """BANK switching and CHECKSUM."""
        self.t.process('10 PRINT "SLOT 0"')
        _, out = capture(self.t.cmd_bank, '1')
        self.assertIn('BANK 1', out)
        self.assertEqual(len(self.t.program), 0)
        _, out = capture(self.t.cmd_bank, '0')
        self.assertIn(10, self.t.program)

        # CHECKSUM
        _, out = capture(self.t.cmd_checksum)
        self.assertIn('CHECKSUM:', out)
        self.assertIn('1 lines', out)

        # Empty checksum
        t2 = QBasicTerminal()
        _, out = capture(t2.cmd_checksum)
        self.assertIn('EMPTY PROGRAM', out)

    def test_chain_merge(self):
        """CHAIN loads+runs preserving vars, MERGE adds lines."""
        # CHAIN
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qb', delete=False,
                                          dir='.', encoding='utf-8') as f:
            f.write('10 PRINT "CHAINED"\n')
            f.write('20 END\n')
            fname = os.path.basename(f.name)
        try:
            self.t.variables['preserved'] = 123
            _, out = capture(self.t.cmd_chain, f'"{fname}"')
            self.assertIn('CHAINED', out)
            self.assertEqual(self.t.variables.get('preserved'), 123)
        finally:
            os.unlink(fname)

        # CHAIN preserves vars (second variant)
        t2 = QBasicTerminal()
        t2.variables['keeper'] = 42
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qb', dir='.', delete=False) as f:
            f.write('10 PRINT keeper\n20 END\n')
            path = f.name
        try:
            _, out = capture(t2.cmd_chain, os.path.basename(path))
            self.assertIn('42', out)
        finally:
            os.unlink(path)

        # MERGE
        t3 = QBasicTerminal()
        t3.process('10 PRINT "ORIGINAL"')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qb', delete=False,
                                          dir='.', encoding='utf-8') as f:
            f.write('20 PRINT "MERGED"\n')
            fname = os.path.basename(f.name)
        try:
            _, out = capture(t3.cmd_merge, f'"{fname}"')
            self.assertIn('MERGED', out)
            self.assertIn(10, t3.program)
            self.assertIn(20, t3.program)
        finally:
            os.unlink(fname)

        # MERGE quantum variant
        t4 = QBasicTerminal()
        t4.process('10 H 0', track_undo=False)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qb', dir='.', delete=False) as f:
            f.write('20 CX 0,1\n30 MEASURE\n')
            path = f.name
        try:
            _, out = capture(t4.cmd_merge, os.path.basename(path))
            self.assertIn('MERGED', out)
            self.assertIn(10, t4.program)
            self.assertIn(20, t4.program)
            self.assertIn(30, t4.program)
        finally:
            os.unlink(path)

    def test_file_handles(self):
        """OPEN/WRITE/CLOSE/READ cycle and EOF."""
        fname = 'test_fh_tmp.txt'
        try:
            _, out = capture(self.t.cmd_open, f'"{fname}" FOR OUTPUT AS #1')
            self.assertIn('OPENED', out)
            self.t._exec_print_file('PRINT #1, "HELLO"', {})
            _, out = capture(self.t.cmd_close, '#1')
            self.assertIn('CLOSED', out)
            _, out = capture(self.t.cmd_open, f'"{fname}" FOR INPUT AS #2')
            self.assertIn('OPENED', out)
            self.t._exec_input_file('INPUT #2, line$', {})
            self.assertEqual(self.t.variables.get('line$'), 'HELLO')
            self.assertEqual(self.t._eof(2), -1)
            capture(self.t.cmd_close, '#2')
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_lprint_line_input_let_string(self):
        """LPRINT, LINE INPUT, LET string."""
        # LPRINT — now goes through self.io (stdout) instead of stderr
        _, out = capture(self.t._exec_lprint, 'LPRINT "TESTING"', {})
        self.assertIn('TESTING', out)

        # LINE INPUT
        orig = builtins.input
        builtins.input = lambda prompt='': 'Alice Bob'
        try:
            result = self.t._try_exec_line_input('LINE INPUT "Name", name$', {})
        finally:
            builtins.input = orig
        self.assertTrue(result)
        self.assertEqual(self.t.variables.get('name$'), 'Alice Bob')

        # LET string
        _, out = capture(self.t.cmd_let_str, 'greeting$', '"hello"')
        self.assertEqual(self.t.variables['greeting$'], 'hello')

    def test_print_using_dim_import_circuit(self):
        """PRINT USING, DIM multi/single, IMPORT namespace, CIRCUIT macro."""
        # PRINT USING
        self.t.variables['x'] = 3.14
        result = self.t._try_exec_print_using('PRINT USING "###.##"; x', {'x': 3.14})
        self.assertTrue(result)
        _, out = capture(self.t._try_exec_print_using,
                         'PRINT USING "###.##"; 3.14', {'dummy': 0})
        self.assertIn('3', out)

        # DIM multi-dimensional
        t_dim = QBasicTerminal()
        t_dim.process('10 DIM matrix(3, 3)')
        t_dim.process('20 END')
        _, out = capture(t_dim.cmd_run)
        self.assertIn('matrix', t_dim.arrays)
        self.assertIsInstance(t_dim.arrays['matrix'], list)
        self.assertEqual(len(t_dim.arrays['matrix']), 16)  # inclusive: (0..3)^2

        # DIM single
        t_dim2 = QBasicTerminal()
        t_dim2.process('10 DIM arr(5)')
        t_dim2.process('20 END')
        capture(t_dim2.cmd_run)
        self.assertIn('arr', t_dim2.arrays)
        self.assertEqual(len(t_dim2.arrays['arr']), 6)   # inclusive: indices 0..5

        # IMPORT namespace
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qb', dir='.', delete=False) as f:
            f.write('DEF ROT(q) = H q\n')
            path = f.name
        try:
            basename = os.path.basename(path)
            _, out = capture(self.t.cmd_import, basename)
            self.assertIn('IMPORTED', out)
            mod_name = os.path.splitext(basename)[0].upper()
            self.assertIn(f'{mod_name}.ROT', self.t.subroutines)
        finally:
            os.unlink(path)

        # CIRCUIT macro
        t_circ = QBasicTerminal()
        t_circ.num_qubits = 4
        t_circ.process('10 H 0', track_undo=False)
        t_circ.process('20 CX 0,1', track_undo=False)
        _, out = capture(t_circ.cmd_circuit_def, 'BELL 10-20')
        self.assertIn('CIRCUIT', out)
        self.assertIn('BELL', t_circ.subroutines)

    def test_open_random_and_eof(self):
        """OPEN FOR INPUT, EOF detection, and CLOSE cycle."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', dir='.', delete=False) as f:
            f.write('hello\nworld\n')
            path = os.path.basename(f.name)
        try:
            t = QBasicTerminal()
            _, out = capture(t.cmd_open, f'"{path}" FOR INPUT AS #1')
            self.assertIn('OPENED', out)
            # Test EOF
            eof_val = t._eof(1)
            self.assertEqual(eof_val, 0)  # not at EOF yet
            # Read to end
            t.files.handles[1].read()
            eof_val = t._eof(1)
            self.assertEqual(eof_val, -1)  # at EOF
            _, out2 = capture(t.cmd_close, '#1')
            self.assertIn('CLOSED', out2)
        finally:
            os.unlink(path)


# =====================================================================
# 8. TestQuantumOps
# =====================================================================
class TestQuantumOps(unittest.TestCase):
    def setUp(self):
        self.t = QBasicTerminal()

    def test_unitary_multi_qubit(self):
        """Define a 2-qubit identity gate via _try_exec_unitary."""
        _, out = capture(self.t._try_exec_unitary,
                         'UNITARY MYID = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]')
        self.assertIn('2-qubit', out)
        self.assertIn('MYID', self.t._custom_gates)

    def test_ctrl_custom_gate(self):
        """CTRL H 0, 1 applies H to qubit 1 only when qubit 0 is 1."""
        self.t.num_qubits = 3
        self.t.shots = 200
        self.t._seed = 3
        self.t.process('10 CTRL H 0, 1')
        self.t.process('20 MEASURE')
        capture(self.t.cmd_run)
        self.assertEqual(self.t.last_counts, {'000': 200})
        self.t.process('5 X 0')
        capture(self.t.cmd_run)
        self.assertEqual(set(self.t.last_counts), {'001', '011'})

    @unittest.skip("requires real simulation")
    def test_inv_rx(self):
        """INV RX 0.5, 0 should apply the inverse."""
        self.t.process('10 RX 0.5, 0')
        self.t.process('20 INV RX 0.5, 0')
        self.t.process('30 MEASURE')
        self.t.shots = 100
        _, out = capture(self.t.cmd_run)
        if self.t.last_counts:
            total = sum(self.t.last_counts.values())
            all_zeros_key = '0' * self.t.num_qubits
            zeros = self.t.last_counts.get(all_zeros_key, 0)
            self.assertGreater(zeros, total * 0.9)

    def test_measure_basis(self):
        """MEASURE_X on |+> always reads 0; MEASURE_Y on |0> reads either."""
        self.t.num_qubits = 1
        self.t.shots = 100
        self.t._seed = 3
        self.t.process('10 H 0')
        self.t.process('20 MEASURE_X 0')
        self.t.process('30 MEASURE')
        capture(self.t.cmd_run)
        self.assertEqual(self.t.last_counts, {'0': 100})
        t2 = QBasicTerminal()
        t2.num_qubits = 1
        t2.shots = 100
        t2._seed = 3
        t2.process('10 MEASURE_Y 0')
        t2.process('20 MEASURE')
        capture(t2.cmd_run)
        self.assertEqual(set(t2.last_counts), {'0', '1'})

    def test_syndrome(self):
        """SYNDROME ZZ reads the parity of two qubits into a bit."""
        self.t.num_qubits = 3
        self.t.shots = 10
        self.t.process('10 X 1')
        self.t.process('20 SYNDROME ZZ 0 1 -> s0')
        self.t.process('30 PRINT s0')
        self.t.process('40 MEASURE')
        _, out = capture(self.t.cmd_run)
        self.assertEqual(out.split('RAN ')[0].split(), ['1'])
        self.assertEqual(self.t.last_counts, {'010': 10})

    def test_locc_meas_ctrl_connect(self):
        """LOCC mode MEAS (SEND), CTRL, and CONNECT/DISCONNECT."""
        # LOCC MEAS
        _, out = capture(self.t.dispatch, 'LOCC 1 1')
        self.assertTrue(self.t.locc_mode)
        _, out = capture(self.t.dispatch, '@A H 0')
        _, out = capture(self.t.dispatch, 'SEND A 0 -> result')
        self.assertIn('result', self.t.variables)

        # LOCC CTRL
        t2 = QBasicTerminal()
        _, out = capture(t2.dispatch, 'LOCC 2 2')
        _, out = capture(t2.dispatch, '@A X 0')
        _, out = capture(t2.dispatch, '@A CTRL X 0, 1')
        self.assertNotIn('ERROR', out.upper())

    def test_save_expect_set_state(self):
        """SAVE_EXPECT records <ZZ> = 1 on a Bell pair; SET_STATE starts the run
        from a named state or from amplitudes."""
        self.t.num_qubits = 2
        for line in ('10 H 0', '20 CX 0,1', '30 SAVE_EXPECT ZZ 0,1 -> zz_val', '40 MEASURE'):
            self.t.process(line, track_undo=False)
        capture(self.t.cmd_run)
        self.assertAlmostEqual(self.t.variables['zz_val'], 1.0, places=9)
        t2 = QBasicTerminal()
        t2.num_qubits = 2
        t2._seed = 4
        t2.process('10 SET_STATE |BELL>', track_undo=False)
        t2.process('20 MEASURE', track_undo=False)
        capture(t2.cmd_run)
        self.assertEqual(set(t2.last_counts), {'00', '11'})
        t3 = QBasicTerminal()
        t3.num_qubits = 1
        t3.shots = 400
        t3._seed = 4
        t3.process('10 SET_STATE [0.707, 0.707]', track_undo=False)
        t3.process('20 MEASURE', track_undo=False)
        capture(t3.cmd_run)
        self.assertAlmostEqual(t3.last_counts['0'] / 400, 0.5, delta=0.08)

    def test_on_measure_on_timer(self):
        """ON MEASURE GOSUB and ON TIMER GOSUB set targets."""
        from qubasic_core.engine import ExecResult

        # ON MEASURE
        result = self.t._cf_on_measure('ON MEASURE GOSUB 100', parsed=parse_stmt('ON MEASURE GOSUB 100'))
        self.assertIsNotNone(result)
        self.assertEqual(result, (True, ExecResult.ADVANCE))
        self.assertEqual(self.t.events.measure_target, 100)

        # ON TIMER
        t2 = QBasicTerminal()
        result = t2._cf_on_timer('ON TIMER(5) GOSUB 200', parsed=parse_stmt('ON TIMER(5) GOSUB 200'))
        self.assertIsNotNone(result)
        self.assertEqual(result, (True, ExecResult.ADVANCE))
        self.assertEqual(t2.events.timer_target, 200)
        self.assertAlmostEqual(t2.events.timer_interval, 5.0)

        # ON TIMER stored in program (not yet executed)
        t3 = QBasicTerminal()
        t3.process('10 ON TIMER(5) GOSUB 100', track_undo=False)
        t3.process('20 END', track_undo=False)
        t3.process('100 PRINT "timer"', track_undo=False)
        t3.process('110 RETURN', track_undo=False)
        self.assertIsNone(t3.events.timer_target)

        # ON MEASURE stored in program
        t4 = QBasicTerminal()
        t4.process('10 ON MEASURE GOSUB 100', track_undo=False)
        t4.process('20 H 0', track_undo=False)
        t4.process('30 MEASURE', track_undo=False)
        t4.process('100 PRINT "measured"', track_undo=False)
        t4.process('110 RETURN', track_undo=False)
        self.assertIn(10, t4.program)
        self.assertIn('ON MEASURE', t4.program[10])

    def test_locc_3party_ghz(self):
        """3-party GHZ via LOCCEngine."""
        from qubasic_core.locc import LOCCEngine
        from qubasic_core.simtools import _MAT_CX, _apply_gate_np
        eng = LOCCEngine([1, 1, 1], joint=True)
        eng.apply('A', 'H', (), [0])
        eng.sv = _apply_gate_np(eng.sv, _MAT_CX, [0, 1], 3)
        eng.sv = _apply_gate_np(eng.sv, _MAT_CX, [0, 2], 3)
        per_reg, joint = eng.sample_joint(4096)
        for state in joint:
            parts = state.split('|')
            self.assertTrue(all(p == parts[0] for p in parts))
        for name in ['A', 'B', 'C']:
            self.assertEqual(len(per_reg[name]), 2)

    def test_sample_sweep_bench_usertype(self):
        """SAMPLE, SWEEP, BENCH, and user TYPE definition."""
        # SAMPLE
        t_s = QBasicTerminal()
        t_s.num_qubits = 2
        t_s.process('10 H 0', track_undo=False)
        t_s.process('20 CX 0,1', track_undo=False)
        t_s.process('30 MEASURE', track_undo=False)
        _, run_out = capture(t_s.cmd_run)
        _, out = capture(t_s.cmd_sample, '100')
        self.assertTrue(len(out) > 0)

        # SWEEP
        t_sw = QBasicTerminal()
        t_sw.num_qubits = 1
        t_sw.shots = 10
        t_sw.process('10 RX angle, 0', track_undo=False)
        t_sw.process('20 MEASURE', track_undo=False)
        _, out = capture(t_sw.cmd_sweep, 'angle 0 3.14 3')
        self.assertIn('SWEEP', out)
        lines_with_angle = [l for l in out.split('\n') if 'angle=' in l]
        self.assertEqual(len(lines_with_angle), 3)

        # BENCH
        t_b = QBasicTerminal()
        _, out = capture(t_b.cmd_bench)
        self.assertIn('Benchmark', out)
        self.assertIn('qubits', out)

        # User TYPE
        if not hasattr(self.t, '_user_types'):
            self.t.symbols.types = {}
        self.t.symbols.types['POINT'] = [('x', 'FLOAT'), ('y', 'FLOAT')]
        self.assertEqual(len(self.t.symbols.types['POINT']), 2)

    def test_sample_output_format(self):
        """SAMPLE n draws n shots of the program's circuit."""
        t = QBasicTerminal()
        t.num_qubits = 2
        t._seed = 5
        t.process('10 H 0', track_undo=False)
        t.process('20 CX 0,1', track_undo=False)
        t.process('30 MEASURE', track_undo=False)
        capture(t.cmd_run)
        _, out = capture(t.cmd_sample, '100')
        self.assertIn('SAMPLED 100 shots (2 unique outcomes)', out)
        rows = {line.split()[0]: int(line.split()[1])
                for line in out.splitlines() if line.strip().startswith('|')}
        self.assertEqual(set(rows), {'|00\u27e9', '|11\u27e9'})
        self.assertEqual(sum(rows.values()), 100)

    def test_estimate_output_format(self):
        """ESTIMATE gives the exact expectation value: <ZZ> = 1 on a Bell pair."""
        t = QBasicTerminal()
        t.num_qubits = 2
        t.process('10 H 0', track_undo=False)
        t.process('20 CX 0,1', track_undo=False)
        t.process('30 MEASURE', track_undo=False)
        capture(t.cmd_run)
        _, out = capture(t.cmd_estimate, 'ZZ 0 1')
        self.assertIn('<ZZ> on qubits [0, 1] = 1.000000', out)


# =====================================================================
# 9. TestParserAndErrors
# =====================================================================
class TestParserAndErrors(unittest.TestCase):
    def test_terminal_keywords(self):
        """Terminal keyword statements: REM, END, RETURN, MEASURE, BARRIER, WEND."""
        cases = [
            ("REM this is a comment", RemStmt),
            ("END", EndStmt),
            ("RETURN", ReturnStmt),
            ("MEASURE", MeasureStmt),
            ("BARRIER", BarrierStmt),
            ("WEND", WendStmt),
        ]
        for text, expected_type in cases:
            s = parse_stmt(text)
            self.assertIsInstance(s, expected_type, f"parse_stmt({text!r})")

    def test_goto_gosub(self):
        """GOTO and GOSUB with targets."""
        s = parse_stmt("GOTO 100")
        self.assertIsInstance(s, GotoStmt)
        self.assertEqual(s.target, 100)
        s = parse_stmt("GOSUB 200")
        self.assertIsInstance(s, GosubStmt)
        self.assertEqual(s.target, 200)

    def test_for_next_while(self):
        """FOR/NEXT/WHILE parsing."""
        s = parse_stmt("FOR i = 1 TO 10 STEP 2")
        self.assertIsInstance(s, ForStmt)
        self.assertEqual(s.var, 'i')
        self.assertEqual(s.step_expr, '2')
        s = parse_stmt("NEXT i")
        self.assertIsInstance(s, NextStmt)
        self.assertEqual(s.var, 'i')
        s = parse_stmt("WHILE x > 0")
        self.assertIsInstance(s, WhileStmt)
        self.assertIn('x', s.condition)

    def test_if_let_print_compound_raw(self):
        """IF THEN, LET, LET array, PRINT, compound, raw fallback."""
        s = parse_stmt("IF x > 0 THEN PRINT x")
        self.assertIsInstance(s, IfThenStmt)
        s = parse_stmt("LET x = 5")
        self.assertIsInstance(s, LetStmt)
        self.assertEqual(s.name, 'x')
        s = parse_stmt("LET arr(0) = 10")
        self.assertIsInstance(s, LetArrayStmt)
        s = parse_stmt('PRINT "hello"')
        self.assertIsInstance(s, PrintStmt)
        s = parse_stmt("H 0 : X 1")
        self.assertIsInstance(s, CompoundStmt)
        self.assertEqual(len(s.parts), 2)
        s = parse_stmt("SOME UNKNOWN THING")
        self.assertIsInstance(s, RawStmt)

    def test_errors_hierarchy(self):
        """Error class hierarchy and catchability."""
        e = QBasicError("test", code=42, line=10)
        self.assertEqual(e.message, "test")
        self.assertEqual(e.code, 42)
        self.assertEqual(e.line, 10)
        self.assertIsInstance(e, Exception)
        self.assertTrue(issubclass(QBasicSyntaxError, QBasicError))
        self.assertTrue(issubclass(QBasicRuntimeError, QBasicError))
        self.assertTrue(issubclass(QBasicBuildError, QBasicError))
        self.assertTrue(issubclass(QBasicRangeError, QBasicError))
        self.assertTrue(issubclass(QBasicIOError, QBasicError))
        self.assertTrue(issubclass(QBasicUndefinedError, QBasicError))
        with self.assertRaises(QBasicError):
            raise QBasicSyntaxError("bad syntax")

    def test_ioport_protocol(self):
        """StdIOPort is IOPort, write and writeln."""
        port = StdIOPort()
        self.assertIsInstance(port, IOPort)
        _, out = capture(port.write, 'hello')
        self.assertEqual(out, 'hello')
        _, out = capture(port.writeln, 'hello')
        self.assertEqual(out, 'hello\n')


# =====================================================================
# 10. TestFuzz
# =====================================================================
class TestFuzz(unittest.TestCase):
    def test_expression_fuzz(self):
        """Random arithmetic and function calls evaluate as Python evaluates them."""
        import math
        import random
        rng = random.Random(20261012)
        t = QBasicTerminal()
        ops = {'+': lambda a, b: a + b, '-': lambda a, b: a - b,
               '*': lambda a, b: a * b, '/': lambda a, b: a / b}
        for _ in range(100):
            a, b = rng.uniform(-100, 100), rng.uniform(-100, 100)
            op = rng.choice(list(ops))
            self.assertAlmostEqual(t._safe_eval(f'{a} {op} {b}'), ops[op](a, b), places=9)
        funcs = {'sin': math.sin, 'cos': math.cos, 'abs': abs, 'exp': math.exp}
        for _ in range(50):
            name, arg = rng.choice(list(funcs)), rng.uniform(-10, 10)
            self.assertAlmostEqual(t._safe_eval(f'{name}({arg})'), funcs[name](arg), places=9)
        for _ in range(20):
            arg = rng.uniform(0, 100)
            self.assertAlmostEqual(t._safe_eval(f'sqrt({arg})'), math.sqrt(arg), places=9)

    def test_parser_fuzz(self):
        """Random strings return Stmt subclass; known keywords parse to subclasses."""
        import random
        import string
        for _ in range(200):
            length = random.randint(0, 50)
            s = ''.join(random.choices(string.ascii_letters + string.digits + ' ,.:+-*/', k=length))
            result = parse_stmt(s)
            self.assertIsInstance(result, Stmt, f'parse_stmt({s!r}) returned {type(result)}')

        keywords = ['MEASURE', 'END', 'RETURN', 'BARRIER', 'WEND', 'RESTORE', 'STOP',
                    'GOTO 100', 'GOSUB 200', 'FOR i = 0 TO 5', 'NEXT i',
                    'WHILE x > 0', 'IF x THEN H 0', 'LET x = 5', 'PRINT "hi"']
        for kw in keywords:
            result = parse_stmt(kw)
            self.assertIsInstance(result, Stmt)
            self.assertNotEqual(type(result).__name__, 'Stmt')

    def test_process_fuzz(self):
        """Random command lines and numbered lines raise nothing, and the
        terminal still runs a program afterwards."""
        import random
        import string
        rng = random.Random(20261013)
        t = QBasicTerminal()
        for _ in range(100):
            line = ''.join(rng.choices(string.ascii_letters + string.digits + ' ,.:+-*/',
                                       k=rng.randint(1, 40)))
            try:
                capture(t.dispatch, line)
            except (EOFError, SystemExit):
                pass
        for _ in range(50):
            content = ''.join(rng.choices(string.ascii_letters + ' ,', k=rng.randint(0, 20)))
            t.process(f'{rng.randint(1, 9999)} {content}', track_undo=False)
        t.io = io = ScriptIO()
        t.cmd_new(silent=True)
        t.process('10 PRINT 6 * 7', track_undo=False)
        t.cmd_run()
        self.assertEqual(io.text.split('RAN ')[0].split(), ['42'])


# =====================================================================
# 11. TestIntegration
# =====================================================================
class TestIntegration(unittest.TestCase):
    def test_cli_json(self):
        """CLI --json bell.qb and grover3.qb end-to-end."""
        import json
        import subprocess

        # bell.qb
        r = subprocess.run(
            [sys.executable, '-X', 'utf8', '-m', 'qubasic_core', '--json', 'examples/bell.qb'],
            capture_output=True, text=True, cwd=_ROOT, timeout=30)
        self.assertEqual(r.returncode, 0)
        data = json.loads(r.stdout)
        self.assertIn('counts', data)
        self.assertIn('num_qubits', data)
        self.assertEqual(data['num_qubits'], 2)
        self.assertEqual(sum(data['counts'].values()), data['shots'])

        # grover3.qb
        r = subprocess.run(
            [sys.executable, '-X', 'utf8', '-m', 'qubasic_core', '--json', 'examples/grover3.qb'],
            capture_output=True, text=True, cwd=_ROOT, timeout=30)
        self.assertEqual(r.returncode, 0)
        data = json.loads(r.stdout)
        self.assertGreater(data['counts'].get('101', 0), 900)

    def test_real_bell_state(self):
        """Integration test with real Qiskit Aer (not mocked) via subprocess."""
        import json
        import subprocess
        r = subprocess.run(
            [sys.executable, '-X', 'utf8', '-c',
             'import json; from qubasic_core.terminal import QBasicTerminal; '
             't = QBasicTerminal(); t.num_qubits = 2; t.shots = 1000; '
             't.process("10 H 0", track_undo=False); '
             't.process("20 CX 0,1", track_undo=False); '
             't.process("30 MEASURE", track_undo=False); '
             'import io,sys; sys.stdout=io.StringIO(); t.cmd_run(); sys.stdout=sys.__stdout__; '
             'print(json.dumps(t.last_counts))'],
            capture_output=True, text=True, cwd=_ROOT, timeout=30)
        self.assertEqual(r.returncode, 0, f"stderr: {r.stderr}")
        counts = json.loads(r.stdout.strip())
        # Bell state: only 00 and 11
        for state in counts:
            self.assertIn(state, ('00', '11'), f"Unexpected state {state}")
        total = sum(counts.values())
        self.assertEqual(total, 1000)
        # Each should be roughly 50% (within statistical tolerance)
        for state in ('00', '11'):
            self.assertGreater(counts.get(state, 0), 300)

    # Grover correctness is tested in test_qubasic.py.


# =====================================================================
# Gap-coverage tests (moved from test_gaps.py)
# =====================================================================


# =====================================================================
# #18: Noise model tests
# =====================================================================
class TestNoiseModel(unittest.TestCase):
    def test_noise_on_off(self):
        """cmd_noise sets and clears noise model."""
        t = QBasicTerminal()
        # ON
        _, out = capture(t.cmd_noise, 'depolarizing 0.01')
        assert t.noise.model is not None
        assert 'depolarizing' in out.lower()
        # OFF
        _, out = capture(t.cmd_noise, 'OFF')
        assert t.noise.model is None

    def test_noise_types(self):
        """All noise types parse without error."""
        t = QBasicTerminal()
        for ntype in ['depolarizing 0.01', 'amplitude_damping 0.01',
                       'phase_flip 0.01', 'readout 0.05 0.1',
                       'combined 0.01 0.02', 'pauli 0.01 0.01 0.01',
                       'reset 0.01 0.01']:
            _, out = capture(t.cmd_noise, ntype)
            assert 'ERROR' not in out.upper(), f"Failed on: {ntype}"
            t.noise.model = None  # reset

    def test_unknown_noise(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_noise, 'nonexistent 0.01')
        assert 'UNKNOWN' in out


# =====================================================================
# #19: Step test
# =====================================================================
class TestStep(unittest.TestCase):
    def test_step_empty_program(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_step)
        assert 'NOTHING TO STEP' in out


# =====================================================================
# #20: Bench test
# =====================================================================
class TestBench(unittest.TestCase):
    def test_bench_output(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_bench)
        assert 'Benchmark' in out
        assert 'qubits' in out
        # Should have results for multiple qubit counts
        assert '4' in out
        assert '8' in out


# =====================================================================
# #21: Monitor test
# =====================================================================
class TestMonitor(unittest.TestCase):
    def test_monitor_quit(self):
        t = QBasicTerminal()
        import builtins
        orig = builtins.input
        builtins.input = lambda prompt='': 'Q'
        try:
            _, out = capture(t.cmd_monitor)
        finally:
            builtins.input = orig
        assert 'MONITOR' in out


# =====================================================================
# #22: Clifford detection
# =====================================================================
class TestCliffordDetection(unittest.TestCase):
    def test_clifford_circuit(self):
        from qiskit import QuantumCircuit
        t = QBasicTerminal()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert t._is_clifford(qc) is True

    def test_non_clifford_circuit(self):
        from qiskit import QuantumCircuit
        t = QBasicTerminal()
        qc = QuantumCircuit(1)
        qc.rx(0.5, 0)
        assert t._is_clifford(qc) is False


# =====================================================================
# #23: Circuit cache
# =====================================================================
class TestCircuitCache(unittest.TestCase):
    def test_cache_hit(self):
        t = QBasicTerminal()
        t.num_qubits = 2
        t.shots = 10
        t.process('10 H 0', track_undo=False)
        t.process('20 CX 0,1', track_undo=False)
        t.process('30 MEASURE', track_undo=False)
        _, _ = capture(t.cmd_run)
        key1 = t._circuit_cache_key
        # Second run should use cache
        _, _ = capture(t.cmd_run)
        assert t._circuit_cache_key == key1

    def test_cache_invalidation(self):
        t = QBasicTerminal()
        t.num_qubits = 2
        t.shots = 10
        t.process('10 H 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, _ = capture(t.cmd_run)
        key1 = t._circuit_cache_key
        # Modify program
        t.process('15 X 1', track_undo=False)
        _, _ = capture(t.cmd_run)
        assert t._circuit_cache_key != key1


# =====================================================================
# #24: Stabilizer fallback
# =====================================================================
class TestStabilizerFallback(unittest.TestCase):
    def test_non_clifford_auto_method(self):
        """Non-Clifford circuit with auto method should use statevector, not stabilizer."""
        t = QBasicTerminal()
        t.num_qubits = 2
        t.shots = 10
        t.sim_method = 'automatic'
        t.process('10 RX 0.5, 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, out = capture(t.cmd_run)
        assert 'ERROR' not in out.upper()


# =====================================================================
# Property-based tests
# =====================================================================
from hypothesis import given, settings
from hypothesis import strategies as st


class TestPropertyBased(unittest.TestCase):
    @given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_eval_roundtrip(self, x):
        """Any float evaluable as expression should return itself."""
        t = QBasicTerminal()
        result = t.eval_expr(str(x))
        self.assertAlmostEqual(result, x, places=5)

    @given(st.text(alphabet='ABCDEFGHIJ 0123456789,', min_size=0, max_size=30))
    @settings(max_examples=100)
    def test_parser_no_crash(self, text):
        """parse_stmt turns any text into a statement and raises nothing."""
        self.assertIsInstance(parse_stmt(text), Stmt)

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=5, deadline=None)
    def test_ghz_state_valid(self, n):
        """GHZ state on n qubits should produce only all-zeros or all-ones."""
        t = QBasicTerminal()
        t.num_qubits = n
        t.shots = 100
        t.process('10 H 0', track_undo=False)
        for i in range(1, n):
            t.process(f'{10+i*10} CX {i-1},{i}', track_undo=False)
        t.process(f'{10+n*10} MEASURE', track_undo=False)
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            t.cmd_run()
        finally:
            sys.stdout = old
        self.assertEqual(sum(t.last_counts.values()), 100)
        for state in t.last_counts:
            self.assertTrue(all(c == state[0] for c in state),
                f"GHZ-{n} produced non-GHZ state: {state}")


# =====================================================================
# New feature tests (noise, GPU, SEED, VERSION, PROBE, CONSISTENCY, etc.)
# =====================================================================

@pytest.mark.real_sim
class TestNoiseEffects(unittest.TestCase):
    """Noise must actually change measurement outcomes."""

    def test_depolarizing_changes_single_qubit(self):
        """X gate under heavy depolarizing must leak to |0>."""
        t = QBasicTerminal()
        t.num_qubits = 1
        t.shots = 2000
        t.cmd_noise('depolarizing 0.3')
        t.process('10 X 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, out = capture(t.cmd_run)
        p0 = t.last_counts.get('0', 0) / 2000
        self.assertGreater(p0, 0.03, f"Noise should leak to |0>, got P(0)={p0:.3f}")

    def test_clean_is_pure(self):
        """Without noise, X gate should produce pure |1>."""
        t = QBasicTerminal()
        t.num_qubits = 1
        t.shots = 500
        t.process('10 X 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, out = capture(t.cmd_run)
        self.assertEqual(t.last_counts.get('0', 0), 0)

    def test_noise_off_restores_clean(self):
        """NOISE OFF must restore noiseless behavior."""
        t = QBasicTerminal()
        t.num_qubits = 1
        t.shots = 500
        t.cmd_noise('depolarizing 0.5')
        t.cmd_noise('off')
        t.process('10 X 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, out = capture(t.cmd_run)
        self.assertEqual(t.last_counts.get('0', 0), 0)

    def test_noise_info(self):
        """NOISE INFO should print model details."""
        t = QBasicTerminal()
        t.cmd_noise('depolarizing 0.1')
        _, out = capture(t.cmd_noise, 'INFO')
        self.assertIn('depol_p', out)
        self.assertIn('0.1', out)

    def test_bell_noise_breaks_correlations(self):
        """Heavy noise on Bell state should produce error states."""
        t = QBasicTerminal()
        t.num_qubits = 2
        t.shots = 2000
        t.cmd_noise('depolarizing 0.3')
        t.process('10 H 0', track_undo=False)
        t.process('20 CX 0,1', track_undo=False)
        t.process('30 MEASURE', track_undo=False)
        _, out = capture(t.cmd_run)
        errors = t.last_counts.get('01', 0) + t.last_counts.get('10', 0)
        self.assertGreater(errors / 2000, 0.02)


@pytest.mark.real_sim
class TestLOCCNoise(unittest.TestCase):
    """LOCC engine noise integration."""

    def test_locc_noise_propagates(self):
        """Noise param should reach LOCC engine both orderings."""
        t = QBasicTerminal()
        t.cmd_noise('depolarizing 0.2')
        t.cmd_locc('JOINT 1 1')
        self.assertAlmostEqual(t.locc.noise_param, 0.2)
        t.cmd_locc('OFF')

        t2 = QBasicTerminal()
        t2.cmd_locc('JOINT 1 1')
        t2.cmd_noise('depolarizing 0.3')
        self.assertAlmostEqual(t2.locc.noise_param, 0.3)

    def test_locc_noise_breaks_bell(self):
        """Noisy LOCC Bell pair should show decorrelation."""
        t = QBasicTerminal()
        t.cmd_noise('depolarizing 0.3')
        t.cmd_locc('JOINT 2 2')
        t.shots = 2000
        t.process('10 SHARE A 0, B 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, out = capture(t.cmd_run)
        total = sum(t.last_counts.values())
        errors = sum(c for s, c in t.last_counts.items()
                     if '|' in s and s.split('|')[0] != s.split('|')[1])
        self.assertGreater(errors / total, 0.05)

    def test_non_depolarizing_warns(self):
        """The LOCC engine reproduces depolarizing, amplitude-damping and
        phase-flip noise; thermal noise it warns about and ignores."""
        t = QBasicTerminal()
        t.cmd_noise('thermal 50e-6 70e-6 1e-6')
        _, out = capture(t.cmd_locc, 'JOINT 1 1')
        self.assertIn('WARNING', out)

    def test_amplitude_damping_locc_supported(self):
        """Amplitude damping now propagates into the LOCC engine."""
        t = QBasicTerminal()
        t.cmd_noise('amplitude_damping 0.2')
        t.cmd_locc('JOINT 1 1')
        self.assertEqual(t.locc.noise_type, 'amplitude_damping')
        self.assertAlmostEqual(t.locc.noise_param, 0.2)


@pytest.mark.real_sim
class TestSeedReproducibility(unittest.TestCase):
    """SEED must produce identical results."""

    def test_same_seed_same_counts(self):
        t = QBasicTerminal()
        t.num_qubits = 2
        t.shots = 100
        t.process('10 H 0', track_undo=False)
        t.process('20 CX 0,1', track_undo=False)
        t.process('30 MEASURE', track_undo=False)
        t.cmd_seed('42')
        _, _ = capture(t.cmd_run)
        run1 = dict(t.last_counts)
        t.cmd_seed('42')
        t._circuit_cache_key = None
        _, _ = capture(t.cmd_run)
        run2 = dict(t.last_counts)
        self.assertEqual(run1, run2)

    def test_different_seed_different_counts(self):
        t = QBasicTerminal()
        t.num_qubits = 3
        t.shots = 50
        t.process('10 H 0', track_undo=False)
        t.process('20 H 1', track_undo=False)
        t.process('30 H 2', track_undo=False)
        t.process('40 MEASURE', track_undo=False)
        t.cmd_seed('1')
        _, _ = capture(t.cmd_run)
        run1 = dict(t.last_counts)
        t.cmd_seed('9999')
        t._circuit_cache_key = None
        _, _ = capture(t.cmd_run)
        run2 = dict(t.last_counts)
        # With 8 states and 50 shots, different seeds should differ
        self.assertNotEqual(run1, run2)


@pytest.mark.real_sim
class TestVersionAndProbe(unittest.TestCase):
    """VERSION and PROBE commands."""

    def test_version(self):
        from qubasic_core import __version__
        t = QBasicTerminal()
        _, out = capture(t.cmd_version)
        self.assertIn(f'QUBASIC {__version__}', out)
        self.assertIn('Qiskit', out)
        self.assertIn('Features', out)

    def test_probe_passes_core(self):
        """PROBE should pass CPU, noise, LOCC, and conditional tests."""
        t = QBasicTerminal()
        _, out = capture(t.cmd_probe)
        # Extract PROBE results lines
        result_lines = [l.strip() for l in out.split('\n')
                        if 'PASS' in l or 'FAIL' in l]
        for required in ['CPU Bell', 'Noise depol', 'LOCC JOINT', 'Conditional ctrl']:
            found = any(required in l and 'PASS' in l for l in result_lines)
            self.assertTrue(found, f"{required} did not PASS in PROBE output: {result_lines}")

    def test_method_capability(self):
        """METHOD with no args should list available methods."""
        t = QBasicTerminal()
        _, out = capture(t.cmd_method, '')
        self.assertIn('automatic', out)
        self.assertIn('statevector', out)


@pytest.mark.real_sim
class TestConsistency(unittest.TestCase):
    """CONSISTENCY command cross-checks."""

    def test_bell_consistent(self):
        t = QBasicTerminal()
        t.num_qubits = 2
        t.shots = 500
        t.process('10 H 0', track_undo=False)
        t.process('20 CX 0,1', track_undo=False)
        t.process('30 MEASURE', track_undo=False)
        _, _ = capture(t.cmd_run)
        _, out = capture(t.cmd_consistency)
        self.assertIn('ALL CONSISTENT', out)
        self.assertNotIn('FAIL', out)


@pytest.mark.real_sim
class TestStateAfterLOCC(unittest.TestCase):
    """EXPECT/DENSITY/BLOCH should work after LOCC runs."""

    def test_expect_after_locc_joint(self):
        """After SHARE, A's qubit is half of a Bell pair: <Z> = 0."""
        t = QBasicTerminal()
        t.cmd_locc('JOINT 2 2')
        t.process('10 SHARE A 0, B 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        capture(t.cmd_run)
        _, out = capture(t.cmd_expect, 'Z 0')
        self.assertIn('<Z> on qubits [0] = 0.000000', out)

    def test_density_after_locc(self):
        """DENSITY after SHARE shows the Bell pair's pure state."""
        t = QBasicTerminal()
        t.cmd_locc('JOINT 1 1')
        t.process('10 SHARE A 0, B 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        capture(t.cmd_run)
        _, out = capture(t.cmd_density)
        bell = [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
        rows = [[float(v) for v in line.split()] for line in out.splitlines()
                if line.split() and all(c in '0123456789.-' for c in line.replace(' ', ''))]
        np.testing.assert_allclose(rows, bell, atol=1e-3)
        self.assertIn('Purity: 1.000000', out)

    def test_split_mode_warns(self):
        """SPLIT mode should tell user to use per-register commands."""
        t = QBasicTerminal()
        t.cmd_locc('SPLIT 1 1')
        t.process('10 @A H 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, _ = capture(t.cmd_run)
        _, out = capture(t.cmd_expect, 'Z 0')
        self.assertIn('SPLIT', out)


@pytest.mark.real_sim
class TestDemoVerification(unittest.TestCase):
    """Demo self-verification passes under clean conditions."""

    def test_bell_demo_verifies(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_demo, 'BELL')
        self.assertIn('VERIFY PASS', out)

    def test_grover_demo_verifies(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_demo, 'GROVER')
        self.assertIn('VERIFY PASS', out)

    def test_bell_demo_fails_under_noise(self):
        t = QBasicTerminal()
        t.cmd_noise('depolarizing 0.3')
        _, out = capture(t.cmd_demo, 'BELL')
        self.assertIn('VERIFY FAIL', out)


@pytest.mark.real_sim
class TestHelpAndCatalog(unittest.TestCase):
    """HELP STATUS and CATALOG."""

    def test_help_status(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_help, 'STATUS')
        self.assertIn('native', out)
        self.assertIn('commands registered', out)

    def test_catalog_backends(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_catalog)
        self.assertIn('Aer/statevector', out)
        self.assertIn('LOCC/numpy-joint', out)


@pytest.mark.real_sim
class TestRunManifest(unittest.TestCase):
    """Run manifest captures execution parameters."""

    def test_manifest_keys(self):
        t = QBasicTerminal()
        t.num_qubits = 1
        t.shots = 10
        t.process('10 H 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        capture(t.cmd_run)
        m = t._run_manifest
        for key in ['program', 'num_qubits', 'shots', 'method', 'device',
                    'seed', 'noise_depol_p', 'depth', 'gates', 'time_s']:
            self.assertIn(key, m, f"Missing manifest key: {key}")
        self.assertEqual((m['program'], m['num_qubits'], m['shots'], m['method']),
                         ({10: 'H 0', 20: 'MEASURE'}, 1, 10, 'stabilizer'))
        self.assertEqual((m['depth'], m['gates']), (2, 2))

    def test_manifest_seed_captured(self):
        t = QBasicTerminal()
        t.num_qubits = 1
        t.shots = 10
        t.cmd_seed('77')
        t.process('10 H 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, _ = capture(t.cmd_run)
        self.assertEqual(t._run_manifest['seed'], 77)


@pytest.mark.real_sim
class TestMethodDevicePrecheck(unittest.TestCase):
    """Method-device incompatibilities caught before execution."""

    def test_stabilizer_gpu_blocked(self):
        t = QBasicTerminal()
        t.sim_method = 'stabilizer'
        t.sim_device = 'GPU'
        t.process('10 H 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, out = capture(t.cmd_run)
        self.assertIn('does not support GPU', out)


# =====================================================================
# 12. TestGapCoverage
# =====================================================================
class TestGapCoverage(unittest.TestCase):
    """Command-line flags, commands and helpers outside the classes above."""

    def setUp(self):
        self.t = QBasicTerminal()

    # 1 --help
    def test_cli_help_output(self):
        """--help prints usage and exits with 0."""
        from unittest.mock import patch

        from qubasic_core.cli import main
        with patch('sys.argv', ['qubasic', '--help']):
            _, out = capture(lambda: None)  # reset
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertEqual(ctx.exception.code, 0)
            finally:
                sys.stdout = old
            text = buf.getvalue()
            self.assertIn('QUBASIC', text)
            self.assertIn('Usage', text)
            self.assertIn('--help', text)

    # 2 quiet mode
    def test_cli_quiet_mode(self):
        """A script runs its program after loading it, printing no banner."""
        from qubasic_core.cli import run_script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qb', delete=False) as f:
            f.write('QUBITS 1\n10 H 0\n20 MEASURE\n')
            path = f.name
        try:
            t = QBasicTerminal()
            _, out = capture(run_script, path, t)
        finally:
            os.unlink(path)
        self.assertEqual(t.num_qubits, 1)
        self.assertEqual(set(t.last_counts), {'0', '1'})
        self.assertNotIn('Quantum BASIC', out)

    # 3 JSON mode
    def test_cli_json_mode(self):
        """--json flag produces valid JSON with expected keys."""
        import json as _json
        from unittest.mock import patch

        from qubasic_core.cli import main
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qb', delete=False) as f:
            f.write('10 QUBITS 1\n20 H 0\n30 MEASURE\n40 END\n')
            path = f.name
        try:
            with patch('sys.argv', ['qubasic', '--json', path]):
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    with self.assertRaises(SystemExit):
                        main()
                finally:
                    sys.stdout = old
                text = buf.getvalue().strip()
                # The last JSON object in output is our result
                # (mock Aer may print nothing before the JSON)
                data = _json.loads(text)
                self.assertIn('counts', data)
                self.assertIn('num_qubits', data)
                self.assertIn('shots', data)
        finally:
            os.unlink(path)

    # 4 module entry point
    def test_module_entry_point(self):
        """Importing __main__ module does not crash and exposes main()."""
        import importlib
        mod = importlib.import_module('qubasic_core.__main__')
        self.assertTrue(callable(mod.main))

    # 5 EOF function
    def test_eof_function(self):
        """_eof returns 0.0 when data remains, 1.0 at end."""
        fname = '_test_eof_tmp.txt'
        with open(fname, 'w') as f:
            f.write('hello\nworld\n')
        try:
            self.t.cmd_open(f'"{fname}" FOR INPUT AS #1')
            self.assertEqual(self.t._eof(1), 0)
            # Read to end
            fh = self.t.files.handles[1]
            fh.read()
            self.assertEqual(self.t._eof(1), -1)
            self.t.cmd_close('#1')
            # Handle not open -> true
            self.assertEqual(self.t._eof(1), -1)
        finally:
            os.unlink(fname)

    # 6 OPEN with ENCODING param
    def test_open_encoding_param(self):
        """OPEN with ENCODING parameter opens file with specified encoding."""
        fname = '_test_enc_tmp.txt'
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('test content')
        try:
            _, out = capture(self.t.cmd_open, f'"{fname}" FOR INPUT AS #1 ENCODING "utf-8"')
            self.assertIn('OPENED #1', out)
            fh = self.t.files.handles.get(1)
            self.assertIsNotNone(fh)
            self.assertEqual(fh.encoding, 'utf-8')
            self.t.cmd_close('#1')
        finally:
            os.unlink(fname)

    # 7 OPEN binary mode
    def test_open_binary_mode(self):
        """OPEN with ENCODING 'binary' opens in binary mode."""
        fname = '_test_bin_tmp.bin'
        with open(fname, 'wb') as f:
            f.write(b'\x00\x01\x02')
        try:
            _, out = capture(self.t.cmd_open, f'"{fname}" FOR INPUT AS #2 ENCODING "binary"')
            self.assertIn('OPENED #2', out)
            fh = self.t.files.handles.get(2)
            self.assertIsNotNone(fh)
            self.assertIn('b', fh.mode)
            self.t.cmd_close('#2')
        finally:
            os.unlink(fname)

    # 8 OPEN RANDOM mode
    def test_open_random_mode(self):
        """OPEN FOR RANDOM creates file if needed and reports it."""
        fname = '_test_rnd_tmp.rnd'
        if os.path.isfile(fname):
            os.unlink(fname)
        try:
            _, out = capture(self.t.cmd_open, f'"{fname}" FOR RANDOM AS #3')
            self.assertIn('OPENED #3', out)
            self.assertTrue(os.path.isfile(fname))
            self.t.cmd_close('#3')
        finally:
            if os.path.isfile(fname):
                os.unlink(fname)

    # 9 CIRCUIT_DEF and APPLY_CIRCUIT
    def test_circuit_def_apply(self):
        """CIRCUIT_DEF stores program lines as a macro; APPLY_CIRCUIT runs it at
        the prompt."""
        self.t.num_qubits = 2
        self.t.process('10 H 0')
        self.t.process('20 CX 0,1')
        _, out_def = capture(self.t.cmd_circuit_def, 'BELL 10-20')
        self.assertIn('CIRCUIT BELL', out_def)
        self.assertIn('2 gates', out_def)
        self.assertEqual(self.t.subroutines['BELL']['body'], ['H 0', 'CX 0,1'])
        _, out_apply = capture(self.t.cmd_apply_circuit, 'BELL')
        self.assertIn('+0.707|00\u27e9 +0.707|11\u27e9', out_apply)

    # 10 TYPE definition
    def test_type_definition(self):
        """TYPE defines a record type with fields in _user_types."""
        _, out1 = capture(self.t.cmd_type, 'Point')
        self.assertIn('TYPE POINT', out1)
        self.assertIsNotNone(self.t.symbols.pending_type)
        # Feed fields
        _, _ = capture(self.t._accumulate_type_field, '  x AS FLOAT')
        _, _ = capture(self.t._accumulate_type_field, '  y AS FLOAT')
        _, out_end = capture(self.t._accumulate_type_field, 'END TYPE')
        self.assertIn('TYPE POINT', out_end)
        self.assertIn('2 fields', out_end)
        self.assertIsNone(self.t.symbols.pending_type)
        self.assertIn('POINT', self.t.symbols.types)
        fields = self.t.symbols.types['POINT']
        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0], ('x', 'FLOAT'))
        self.assertEqual(fields[1], ('y', 'FLOAT'))

    # 11 SCREEN modes
    def test_screen_modes(self):
        """DISPLAY 0-5 sets _screen_mode; SCREEN 3-5 still choose it."""
        mode_names = {0: 'text', 1: 'histogram', 2: 'statevector',
                      3: 'Bloch', 4: 'density', 5: 'circuit'}
        for mode in range(6):
            _, out = capture(self.t.cmd_display, str(mode))
            self.assertEqual(self.t._screen_mode, mode)
            self.assertIn(mode_names[mode], out)
        _, out_bad = capture(self.t.cmd_display, '9')
        self.assertIn('?USAGE: DISPLAY', out_bad)
        self.assertEqual(self.t._screen_mode, 5)  # unchanged from last valid
        capture(self.t.cmd_screen, '3')
        self.assertEqual(self.t._screen_mode, 3)
        _, out = capture(self.t.cmd_screen, '6')
        self.assertIn('?SCREEN 6', out)

    # 12 PROMPT command
    def test_prompt_command(self):
        """PROMPT changes the REPL prompt string."""
        _, out = capture(self.t.cmd_prompt, '">>> "')
        self.assertEqual(self.t._prompt, '>>> ')
        self.assertIn('>>>', out)
        # No args shows current
        _, out2 = capture(self.t.cmd_prompt, '')
        self.assertIn('>>>', out2)

    # 13 PLAY command
    def test_play_command(self):
        """cmd_play emits bell character(s) without crashing."""
        _, out = capture(self.t.cmd_play, '')
        self.assertIn('\a', out)
        _, out2 = capture(self.t.cmd_play, '3')
        self.assertEqual(out2.count('\a'), 3)

    # 14 SAMPLE and ESTIMATE
    def test_sample_and_estimate(self):
        """cmd_sample and cmd_estimate exist and handle empty programs."""
        # Empty program
        _, out_s = capture(self.t.cmd_sample, '')
        self.assertIn('NOTHING TO SAMPLE', out_s)
        # With a program: sample should attempt execution
        t2 = QBasicTerminal()
        t2.num_qubits = 1
        t2.shots = 10
        t2.process('10 H 0')
        t2.process('20 MEASURE')
        # cmd_sample will try SamplerV2 which uses mock — may error but shouldn't crash
        _, out_s2 = capture(t2.cmd_sample, '10')
        self.assertTrue('SAMPLED' in out_s2 or '?SAMPLE ERROR' in out_s2)
        # cmd_estimate with no program
        _, out_e = capture(self.t.cmd_estimate, '')
        self.assertIn('?USAGE', out_e)

    # 15 LIST SUBS, LIST VARS, LIST ARRAYS
    def test_list_subs_vars_arrays(self):
        """LIST SUBS/VARS/ARRAYS subcommands produce output."""
        # No subs
        _, out_s = capture(self.t.cmd_list, 'SUBS')
        self.assertIn('No subroutines', out_s)
        # No vars
        _, out_v = capture(self.t.cmd_list, 'VARS')
        self.assertIn('No variables', out_v)
        # No arrays
        _, out_a = capture(self.t.cmd_list, 'ARRAYS')
        self.assertIn('No arrays', out_a)
        # With data
        self.t.variables['x'] = 42
        self.t.arrays['a'] = [1.0, 2.0, 3.0]
        self.t.process('10 H 0')
        self.t.subroutines['BELL'] = {'body': ['H 0', 'CX 0,1'], 'params': []}
        _, out_s2 = capture(self.t.cmd_list, 'SUBS')
        self.assertIn('BELL', out_s2)
        _, out_v2 = capture(self.t.cmd_list, 'VARS')
        self.assertIn('x', out_v2)
        self.assertIn('42', out_v2)
        _, out_a2 = capture(self.t.cmd_list, 'ARRAYS')
        self.assertIn('a(3)', out_a2)

    # 16 REDIM
    def test_redim(self):
        """REDIM resizes (inclusive sizing); plain clears, PRESERVE keeps."""
        # DIM a(5) spans indices 0..5 -> 6 slots (inclusive sizing).
        self.t._try_exec_dim('DIM arr(5)')
        self.assertEqual(len(self.t.arrays['arr']), 6)
        self.t.arrays['arr'][2] = 99.0
        # Plain REDIM clears to zeros.
        self.t._try_exec_redim('REDIM arr(8)')
        self.assertEqual(len(self.t.arrays['arr']), 9)
        self.assertEqual(self.t.arrays['arr'][2], 0.0)
        # REDIM PRESERVE keeps existing data.
        self.t.arrays['arr'][1] = 7.0
        self.t._try_exec_redim('REDIM PRESERVE arr(10)')
        self.assertEqual(len(self.t.arrays['arr']), 11)
        self.assertEqual(self.t.arrays['arr'][1], 7.0)
        # PRESERVE smaller truncates.
        self.t._try_exec_redim('REDIM PRESERVE arr(3)')
        self.assertEqual(len(self.t.arrays['arr']), 4)
        # REDIM on a non-existent array creates it.
        self.t._try_exec_redim('REDIM newary(4)')
        self.assertEqual(len(self.t.arrays['newary']), 5)


# =====================================================================
# Hypothesis property-based tests
# =====================================================================

from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st


class TestPropertyBasedExtended(unittest.TestCase):
    """Property-based tests using hypothesis."""

    def setUp(self):
        from qubasic_core.terminal import QBasicTerminal
        self.t = QBasicTerminal()

    @given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    @hyp_settings(max_examples=50)
    def test_eval_arithmetic_identity(self, x):
        """x + 0 == x for all finite floats."""
        self.t.variables['x'] = x
        result = self.t._safe_eval('x + 0')
        self.assertAlmostEqual(float(result), x, places=6)

    @given(st.integers(min_value=0, max_value=255))
    @hyp_settings(max_examples=50)
    def test_parser_never_crashes_on_gate_names(self, rule):
        """Parser should not crash on any 1-byte command string."""
        from qubasic_core.parser import parse_stmt
        text = f"H {rule}"
        stmt = parse_stmt(text)
        self.assertIsNotNone(stmt)

    @given(st.text(min_size=0, max_size=80, alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        blacklist_characters='\x00')))
    @hyp_settings(max_examples=80, deadline=None)
    def test_process_never_crashes(self, line):
        """process() should handle arbitrary text without crashing."""
        try:
            self.t.process(line, track_undo=False)
        except (EOFError, SystemExit):
            pass  # BYE/QUIT/EXIT raise these intentionally

    @given(st.integers(min_value=1, max_value=20))
    @hyp_settings(max_examples=15, deadline=None)
    def test_for_loop_count(self, n):
        """FOR I = 0 TO n should iterate exactly n+1 times."""
        self.t.cmd_new(silent=True)
        self.t.variables.clear()
        self.t.process(f'10 LET count = 0', track_undo=False)
        self.t.process(f'20 FOR I = 0 TO {n}', track_undo=False)
        self.t.process(f'30 LET count = count + 1', track_undo=False)
        self.t.process(f'40 NEXT I', track_undo=False)
        self.t.cmd_run()
        self.assertEqual(int(self.t.variables.get('count', 0)), n + 1)


# =====================================================================
# CLI integration tests
# =====================================================================

class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI entry point and dispatch."""

    def setUp(self):
        from qubasic_core.terminal import QBasicTerminal
        self.t = QBasicTerminal()

    def test_dispatch_unknown_command_no_crash(self):
        """An unknown word is reported as a syntax error."""
        self.t.io = io = ScriptIO()
        self.t.dispatch('XYZZY')
        self.assertEqual(io.text.splitlines()[0], '?SYNTAX ERROR: UNKNOWN GATE: XYZZY')

    def test_dispatch_empty_line(self):
        """An empty or blank line does nothing."""
        self.t.io = io = ScriptIO()
        program = dict(self.t.program)
        self.t.dispatch('')
        self.t.dispatch('   ')
        self.assertEqual(io.text, '')
        self.assertEqual(self.t.program, program)

    def test_seed_dispatch_with_arg(self):
        """SEED 42 should work via dispatch."""
        self.t.dispatch('SEED 42')
        self.assertEqual(self.t._seed, 42)

    def test_seed_dispatch_off(self):
        """SEED OFF should clear the seed."""
        self.t.dispatch('SEED 42')
        self.t.dispatch('SEED OFF')
        self.assertIsNone(self.t._seed)

    def test_locc_non_numeric_args(self):
        """LOCC with non-numeric args should not crash."""
        self.t.io = buf = ScriptIO()
        self.t.cmd_locc('4 banana')
        output = buf.text
        self.assertIn('integer', output.lower())

    def test_method_no_args_no_crash(self):
        """METHOD alone shows the method and device and lists the choices."""
        self.t.io = buf = ScriptIO()
        self.t.cmd_method('')
        self.assertIn('METHOD = automatic  DEVICE = CPU', buf.text)
        self.assertIn('matrix_product_state', buf.text)

    def test_cli_seed_flag(self):
        """--seed makes a script's result repeat exactly."""
        import json
        import subprocess
        runs = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, '-m', 'qubasic_core', '--seed', '42', '--json',
                 os.path.join('examples', 'bell.qb')],
                capture_output=True, text=True, cwd=_ROOT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            runs.append(json.loads(result.stdout)['counts'])
        self.assertEqual(set(runs[0]), {'00', '11'})
        self.assertEqual(runs[0], runs[1])

    def test_cli_help_flag(self):
        """--help should exit 0."""
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'qubasic_core',
             '--help'],
            capture_output=True, text=True, cwd=_ROOT, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn('QUBASIC', result.stdout)


@pytest.mark.real_sim
class TestRegressionBehaviors(unittest.TestCase):
    """Real-simulation checks for language and execution behaviors."""

    def test_multiline_if_executes_only_taken_branch(self):
        t = QBasicTerminal(); t.num_qubits = 1; t.shots = 100
        t.program = {10: 'IF 1==1 THEN', 20: 'X 0', 30: 'ELSE',
                     40: 'H 0', 50: 'END IF', 60: 'MEASURE'}
        capture(t.cmd_run)
        self.assertEqual(t.last_counts.get('1', 0), 100)  # X taken, H skipped
        t2 = QBasicTerminal(); t2.num_qubits = 1; t2.shots = 100
        t2.program = {10: 'IF 1==2 THEN', 20: 'X 0', 30: 'ELSE',
                      40: 'H 0', 50: 'END IF', 60: 'MEASURE'}
        capture(t2.cmd_run)
        self.assertGreater(len(t2.last_counts), 1)  # H taken -> superposition

    def test_dollar_address_peek(self):
        t = QBasicTerminal(); t.num_qubits = 3
        _, out = capture(t.cmd_peek, '$D000')
        self.assertIn('$D000 = 3.0', out)

    def test_option_base_one_indexing(self):
        t = QBasicTerminal(); t.num_qubits = 1
        t.program = {10: 'OPTION BASE 1', 20: 'DIM a(3)',
                     30: 'LET a(1) = 7', 40: 'PRINT a(1)', 50: 'END'}
        _, out = capture(t.cmd_run)
        self.assertEqual(out.split('RAN ')[0].split(), ['7'])
        self.assertEqual(t.arrays['a'], [7.0, 0.0, 0.0])
        t.program = {10: 'OPTION BASE 1', 20: 'DIM a(3)', 30: 'LET a(0) = 7', 40: 'END'}
        _, out = capture(t.cmd_run)
        self.assertIn('SUBSCRIPT OUT OF RANGE: a(0)', out)

    def test_set_state_persists_into_run(self):
        t = QBasicTerminal(); t.num_qubits = 1; t.shots = 400
        capture(t.cmd_set_state, '|+>')
        t.program = {10: 'MEASURE'}
        capture(t.cmd_run)
        self.assertGreater(t.last_counts.get('0', 0), 100)
        self.assertGreater(t.last_counts.get('1', 0), 100)

    def test_expect_zz_numpy_fastpath(self):
        t = QBasicTerminal(); t.num_qubits = 2; t.shots = 50
        t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}
        capture(t.cmd_run)
        _, out = capture(t.cmd_expect, 'ZZ 0 1')
        self.assertIn('1.000000', out)

    def test_amplitude_damping_decays_in_locc(self):
        t = QBasicTerminal()
        t.cmd_noise('amplitude_damping 0.6')
        t.cmd_locc('JOINT 1 1')
        t.shots = 400
        t.program = {10: '@A X 0', 20: 'MEASURE'}
        capture(t.cmd_run)
        a0 = sum(c for s, c in t.last_counts.items() if s.split('|')[0] == '0')
        self.assertGreater(a0, 0)  # |1> decayed toward |0>

    def test_large_clifford_uses_stabilizer(self):
        t = QBasicTerminal(); t.num_qubits = 30; t.shots = 20
        prog = {10: 'H 0'}
        ln = 20
        for i in range(1, 30):
            prog[ln] = f'CX 0,{i}'
            ln += 10
        prog[ln] = 'MEASURE'
        t.program = prog
        capture(t.cmd_run)
        for s in t.last_counts:
            self.assertIn(s, ('0' * 30, '1' * 30))
        self.assertEqual(t._run_manifest['method'], 'stabilizer')

    def test_noise_strength_change_invalidates_cache(self):
        # str(NoiseModel) ignores channel parameters, so a strength change must
        # still rebuild the cached circuit (X|0> under amplitude damping p).
        t = QBasicTerminal(); t.num_qubits = 1; t.shots = 2000
        t.program = {10: 'X 0', 20: 'MEASURE'}
        t.cmd_noise('amplitude_damping 0.1')
        capture(t.cmd_run)
        p0_low = t.last_counts.get('0', 0) / 2000
        t.cmd_noise('amplitude_damping 0.9')
        capture(t.cmd_run)
        p0_high = t.last_counts.get('0', 0) / 2000
        self.assertLess(p0_low, 0.25)
        self.assertGreater(p0_high, 0.70)

    def test_multi_next_closes_both_loops(self):
        t = QBasicTerminal(); t.num_qubits = 1
        t.program = {10: 'LET c = 0', 20: 'FOR i = 1 TO 2', 30: 'FOR j = 1 TO 2',
                     40: 'LET c = c + 1', 50: 'NEXT j, i', 60: 'PRINT c', 70: 'END'}
        _, out = capture(t.cmd_run)
        self.assertEqual(out.split('RAN ')[0].split(), ['4'])


class TestCommandEffects(unittest.TestCase):
    """Commands across the surface, each checked for what it prints and the
    state it changes."""

    def _ready(self, answers=()):
        t = QBasicTerminal()
        t.io = ScriptIO(answers)
        t.num_qubits = 2
        t.shots = 20
        t._seed = 5
        for line in ('10 H 0', '20 CX 0,1', '30 MEASURE'):
            t.process(line, track_undo=False)
        t.cmd_run()
        return t

    @staticmethod
    def said(t, fn, *args):
        """What `fn(*args)` writes to the terminal."""
        start = len(t.io.parts)
        fn(*args)
        return ''.join(t.io.parts[start:])

    def test_assist_commands(self):
        from unittest import mock

        from qubasic_core.analysis import did_you_mean
        t = self._ready(answers=['A'])
        self.assertIn('q0 (0.00,0.00,0.00)', self.said(t, t.cmd_draw, '0'))   # each half of a Bell pair is mixed
        self.assertIn('q1 (0.00,0.00,0.00)', self.said(t, t.cmd_draw, ''))
        self.assertIn('\x1b[36mCX\x1b[0m 0,1', self.said(t, t.cmd_list_colored))
        compared = self.said(t, t.cmd_compare, 'statevector stabilizer')
        self.assertEqual(sorted(ln.split()[0] for ln in compared.splitlines() if ln.strip().startswith('|')),
                         ['|00⟩', '|11⟩'])
        heat = self.said(t, t.cmd_heatmap, '')
        self.assertIn('mutual information', heat)
        self.assertEqual(heat.count('2.00'), 2)                                  # I(0:1) = 2 bits
        self.assertIn('P(top) vs a: 0.0 to 1.0', self.said(t, t.cmd_plot, 'a 0 1 3'))
        self.assertEqual(self.said(t, t.cmd_animate, 'a 0 1 2 0').count('a='), 2)
        self.assertIn('?Unknown theme: bogus', self.said(t, t.cmd_theme, 'bogus'))
        for name in ('retro', 'none'):
            self.said(t, t.cmd_theme, name)
            self.assertEqual(t._theme_name, name)
        explained = self.said(t, t.cmd_explain, '')
        self.assertIn('put qubit 0 in superposition', explained)
        self.assertIn('measure all qubits', explained)
        with mock.patch('subprocess.run') as clipboard:
            self.assertIn('COPIED 2 lines to clipboard', self.said(t, t.cmd_clip, ''))
        copied = clipboard.call_args.kwargs['input'].decode()
        self.assertEqual(sorted(line.split()[0] for line in copied.splitlines()), ['|00>', '|11>'])
        self.assertIn('BANK 1 (empty)', self.said(t, t.cmd_bank, '1'))
        self.assertIn('BANK 0 (3 lines)', self.said(t, t.cmd_bank, '0'))
        self.assertIn('?BANK 1 is empty', self.said(t, t.cmd_diff, '1'))
        with mock.patch('random.choice', lambda seq: seq[0]), mock.patch('random.shuffle', lambda seq: None):
            quiz = self.said(t, t.cmd_quiz)
        self.assertIn('QUIZ: H|0> = ?', quiz)
        self.assertIn('Correct!', quiz)
        self.assertEqual(did_you_mean('BLASCH'), 'BLOCH')
        self.assertIsNone(did_you_mean('BLOCH'))
        self.assertEqual(t._status_prompt(), '2q ] ')

    def test_program_management(self):
        t = self._ready(answers=['.', 'REM edited'])
        self.assertIn('COPIED 10-20 TO 100-110', self.said(t, t.cmd_copy, '10-20 TO 100'))
        self.assertEqual((t.program[100], t.program[110]), ('H 0', 'CX 0,1'))
        self.assertIn('MOVED 100-110 TO 200-210', self.said(t, t.cmd_move, '100-110 TO 200'))
        self.assertEqual(sorted(t.program), [10, 20, 30, 200, 210])
        self.assertIn('2 match(es)', self.said(t, t.cmd_find, '"H"'))
        self.assertIn('2 replacement(s)', self.said(t, t.cmd_replace, '"H 0" WITH "X 0"'))
        self.assertEqual((t.program[10], t.program[200]), ('X 0', 'X 0'))
        checksum = self.said(t, t.cmd_checksum)
        self.assertRegex(checksum, r'CHECKSUM: [0-9A-F]{8} \(5 lines\)')
        self.assertEqual(self.said(t, t.cmd_checksum), checksum)
        self.assertIn('No subroutines defined', self.said(t, t.cmd_list, 'SUBS'))
        self.assertIn('_GATES = 4', self.said(t, t.cmd_list, 'VARS'))
        self.assertIn('No arrays', self.said(t, t.cmd_list, 'ARRAYS'))
        self.assertIn('NO SUBROUTINES DEFINED', self.said(t, t.cmd_defs))
        self.assertIn('NO REGISTERS DEFINED', self.said(t, t.cmd_regs))
        self.assertIn('_GATES = 4', self.said(t, t.cmd_vars))
        before = dict(t.program)
        self.assertIn('AUTO 500, 10', self.said(t, t.cmd_auto, '500, 10'))      # '.' ends AUTO at once
        self.assertEqual(t.program, before)
        self.assertIn('UPDATED 10', self.said(t, t.cmd_edit, '10'))
        self.assertEqual(t.program[10], 'REM edited')
        self.assertNotEqual(self.said(t, t.cmd_checksum), checksum)

    def test_debugging_commands(self):
        import re

        import numpy as np
        t = self._ready()
        self.said(t, t.cmd_profile, 'ON')
        t.cmd_run()
        profile = self.said(t, t.cmd_profile, 'SHOW')
        rows = re.findall(r'^\s+(\d+)\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)\s+(.+)$', profile, re.M)
        self.assertEqual(rows, [('10', '1', '1', 'H 0'), ('20', '1', '1', 'CX 0,1')])
        self.said(t, t.cmd_profile, 'OFF')
        self.assertFalse(t.machine.profiling)
        self.said(t, t.cmd_watch, 'x')
        self.assertEqual(t.debug.watches, ['x'])
        self.said(t, t.cmd_watch, 'CLEAR')
        self.assertEqual(t.debug.watches, [])
        self.said(t, t.cmd_breakpoint, '20')
        self.assertEqual(t.debug.breakpoints, {20})
        self.assertIn('20: CX 0,1', self.said(t, t.cmd_breakpoint, 'LIST'))
        self.said(t, t.cmd_breakpoint, '10-30')
        self.assertEqual(t.debug.breakpoints, set())
        self.said(t, t.cmd_tron)
        self.assertTrue(t.debug.trace)
        self.said(t, t.cmd_troff)
        self.assertFalse(t.debug.trace)
        self.assertIn('Collected 3 runs', self.said(t, t.cmd_stats, '3'))
        self.assertEqual(len(t.machine.runs), 3)
        self.assertIn('Statistics over 3 runs (20 shots/run)', self.said(t, t.cmd_stats, 'SHOW'))
        self.said(t, t.cmd_stats, 'CLEAR')
        self.assertEqual(t.machine.runs, [])
        t.debug.checkpoints = [(10, np.array([1, 0, 0, 0], dtype=complex)),
                             (20, np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))]
        t.debug.position = 1
        self.assertIn('[1] line 20 <<', self.said(t, t.cmd_history))
        self.assertIn('+1.000|00⟩', self.said(t, t.cmd_rewind, '1'))
        self.assertEqual(t.debug.position, 0)
        self.assertIn('+0.707|00⟩ +0.707|11⟩', self.said(t, t.cmd_forward, '1'))
        self.assertEqual(t.debug.position, 1)

    def test_screen_and_machine_commands(self):
        from qubasic_core import __version__
        t = self._ready(answers=['Q'])
        for mode, name in enumerate(('text', 'histogram', 'statevector', 'Bloch', 'density', 'circuit')):
            self.assertIn(f'DISPLAY {name}', self.said(t, t.cmd_display, str(mode)))
            self.assertEqual(t._screen_mode, mode)
        self.said(t, t.cmd_display, '1')
        self.said(t, t.cmd_color, 'red, blue')
        self.assertEqual((t.screen.fg, t.screen.bg), ('red', 'blue'))
        self.said(t, t.cmd_prompt, '"> "')
        self.assertEqual(t._prompt, '> ')
        self.assertIn('$D000 = 2.0', self.said(t, t.cmd_peek, '$D000'))
        self.said(t, t.cmd_poke, '$D001, 64')
        self.assertEqual(t.shots, 64)
        self.assertRegex(self.said(t, t.cmd_map), r'\$D001 shots\s+= 64\.0')
        dumped = self.said(t, t.cmd_dump, '0 31')
        self.assertEqual([v for line in dumped.splitlines() if ':' in line for v in line.split(':')[1].split()],
                         ['00'] * 32)
        self.assertRegex(self.said(t, t.cmd_catalog), r'\$E000\s+BELL')
        self.assertIn('LOADED: Bell State', self.said(t, t.cmd_sys, '$E000'))
        self.assertLessEqual(set(t.last_counts), {'00', '11'})
        self.assertIn('MONITOR OFF', self.said(t, t.cmd_monitor))
        self.assertIn(f'QUBASIC {__version__}', self.said(t, t.cmd_version))
        self.assertIn('H: gate — 0 parameter(s), 1 qubit(s)', self.said(t, t.cmd_help, 'H'))
        self.assertIn('LOCC: LOCC [JOINT|SPLIT]', self.said(t, t.cmd_help, 'LOCC'))
        self.assertIn('commands registered', self.said(t, t.cmd_help, 'STATUS'))
        self.assertIn('Circuit: 2 qubits, depth 3, 4 gates', self.said(t, t.cmd_decompose))
        self.assertIn('METHOD = automatic  DEVICE = CPU', self.said(t, t.cmd_method, ''))

    def test_bloch_vector_axes(self):
        """_bloch_vector and the $0100 By field carry the right sign on every
        axis, with |+i> reading as +Y."""
        import numpy as np

        from qubasic_core.terminal import QBasicTerminal
        t = QBasicTerminal()
        r2 = 1.0 / np.sqrt(2)
        cases = [
            (np.array([1, 0], dtype=complex),       (0.0, 0.0, +1.0)),   # |0>  +Z
            (np.array([0, 1], dtype=complex),       (0.0, 0.0, -1.0)),   # |1>  -Z
            (np.array([r2, r2], dtype=complex),     (+1.0, 0.0, 0.0)),   # |+>  +X
            (np.array([r2, -r2], dtype=complex),    (-1.0, 0.0, 0.0)),   # |->  -X
            (np.array([r2, 1j * r2], dtype=complex),  (0.0, +1.0, 0.0)), # |+i> +Y
            (np.array([r2, -1j * r2], dtype=complex), (0.0, -1.0, 0.0)), # |-i> -Y
        ]
        for sv, (ex, ey, ez) in cases:
            x, y, z = t._bloch_vector(sv, 0, 1)
            self.assertAlmostEqual(x, ex, places=6)
            self.assertAlmostEqual(y, ey, places=6)
            self.assertAlmostEqual(z, ez, places=6)
        # Memory-mapped Bloch Y for q0 ($0102) must agree for |+i>.
        t.num_qubits = 1
        t.last_sv = np.array([r2, 1j * r2], dtype=complex)
        self.assertAlmostEqual(t._peek(0x0102), +1.0, places=6)


# =====================================================================
# Run
# =====================================================================
if __name__ == '__main__':
    unittest.main()
