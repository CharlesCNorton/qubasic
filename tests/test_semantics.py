"""BASIC statement semantics and live (shot-by-shot) execution."""

import math
import time
import unittest

import numpy as np

from qubasic_core.terminal import QBasicTerminal


class _IO:
    """Captures output; answers INPUT from a list and counts the reads."""

    def __init__(self, answers=()):
        self.parts = []
        self.answers = list(answers)
        self.reads = 0

    def write(self, text):
        self.parts.append(text)

    def writeln(self, text):
        self.parts.append(text + '\n')

    def read_line(self, prompt):
        self.reads += 1
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    @property
    def text(self):
        return ''.join(self.parts)


def run(lines, nq=1, shots=64, seed=11, answers=(), setup=None):
    t = QBasicTerminal()
    t.num_qubits = nq
    t.shots = shots
    t._seed = seed
    io = _IO(answers)
    t.io = io
    if setup:
        setup(t)
    for line in lines:
        t.process(line, track_undo=False)
    t.cmd_run()
    return t, io


def printed(io):
    """Program output before the RAN summary line."""
    return io.text.split('\nRAN ')[0]


class TestCompoundStatements(unittest.TestCase):

    def test_keyword_led_compounds(self):
        _, io = run(['10 PRINT 1: PRINT 2', '20 LET a = 1: LET b = 2', '30 PRINT a + b'])
        self.assertEqual(printed(io).split(), ['1', '2', '3'])

    def test_reset_compound(self):
        t, _ = run(['10 H 0', '20 RESET 0: X 0', '30 MEASURE'])
        self.assertEqual(set(t.last_counts), {'1'})

    def test_colon_inside_string_is_not_a_separator(self):
        _, io = run(['10 PRINT "time: 5": PRINT "ok"'])
        self.assertEqual(printed(io).splitlines(), ['time: 5', 'ok'])

    def test_if_owns_rest_of_line(self):
        _, io = run(['10 x = 1', '20 IF x = 2 THEN PRINT "a": PRINT "b"', '30 PRINT "c"'])
        self.assertEqual(printed(io).split(), ['c'])

    def test_jumps_leave_a_compound(self):
        _, io = run(['10 PRINT "x": GOTO 30', '20 PRINT "skipped"', '30 PRINT "y": END',
                     '40 PRINT "after end"'])
        self.assertEqual(printed(io).split(), ['x', 'y'])

    def test_gosub_returns_to_next_statement(self):
        _, io = run(['10 GOSUB 100: PRINT "back"', '20 END', '100 PRINT "sub": RETURN'])
        self.assertEqual(printed(io).split(), ['sub', 'back'])

    def test_gosub_inside_if_clause_returns_into_clause(self):
        _, io = run(['10 IF 1 THEN GOSUB 100: PRINT "after"', '20 END',
                     '100 PRINT "in": RETURN'])
        self.assertEqual(printed(io).split(), ['in', 'after'])


class TestLoops(unittest.TestCase):

    def test_single_line_for(self):
        t, _ = run(['10 FOR i = 1 TO 3: FOR j = 1 TO 2: k = k + 1: NEXT j: NEXT i'])
        self.assertEqual(t.variables['k'], 6)

    def test_for_opened_in_a_compound_closed_later(self):
        _, io = run(['10 FOR i = 1 TO 2: PRINT "a"; i', '20 NEXT i'])
        self.assertEqual(printed(io).split(), ['a', '1', 'a', '2'])

    def test_zero_trip_for(self):
        _, io = run(['10 FOR i = 5 TO 1', '20 PRINT "body"', '30 NEXT i', '40 PRINT "after"'])
        self.assertEqual(printed(io).split(), ['after'])

    def test_while_inside_for_keeps_the_loop_stack(self):
        t, io = run(['10 FOR i = 1 TO 2', '20 x = 0', '30 WHILE x < 3', '40 x = x + 1',
                     '50 WEND', '60 NEXT i', '70 PRINT i; x'])
        self.assertEqual(printed(io).split(), ['3', '3'])

    def test_do_loop_until_on_one_line(self):
        t, _ = run(['10 DO: n = n + 1: LOOP UNTIL n >= 4'])
        self.assertEqual(t.variables['n'], 4)

    def test_exit_for(self):
        t, _ = run(['10 FOR i = 1 TO 9', '20 IF i = 4 THEN EXIT FOR', '30 NEXT', '40 r = i'])
        self.assertEqual(t.variables['r'], 4)


class TestExpressions(unittest.TestCase):

    def test_single_equals_compares_in_if(self):
        _, io = run(['10 x = 2', '20 IF x = 2 THEN PRINT "yes" ELSE PRINT "no"'])
        self.assertEqual(printed(io).split(), ['yes'])

    def test_comparison_value(self):
        t, _ = run(['10 flag = 3 = 3', '20 other = 2 = 3'])
        self.assertEqual((t.variables['flag'], t.variables['other']), (-1, 0))

    def test_unassigned_reads_zero_in_a_program(self):
        t, _ = run(['10 count = count + 1', '20 s$ = t$ + "x"'])
        self.assertEqual((t.variables['count'], t.variables['s$']), (1, 'x'))

    def test_prompt_and_quantum_operands_stay_strict(self):
        t = QBasicTerminal()
        with self.assertRaises(ValueError):
            t._safe_eval('never_set')
        _, io = run(['10 H q'])
        self.assertIn('CANNOT RESOLVE QUBIT', io.text)

    def test_basic_operators_and_functions(self):
        t = QBasicTerminal()
        values = [t._safe_eval(e) for e in
                  ('10 MOD 3', '-7 MOD 3', '2 ^ 10', '5 XOR 3', 'SGN(-4)', 'SQR(16)',
                   'CINT(2.5)', 'ATN(1) * 4')]
        self.assertEqual(values[:7], [1, -1, 1024, 6, -1, 4.0, 2])
        self.assertAlmostEqual(values[7], math.pi)

    def test_bare_print_is_a_blank_line(self):
        _, io = run(['10 PRINT "a"', '20 PRINT', '30 PRINT "b"'])
        self.assertEqual(printed(io), 'a\n\nb\n')

    def test_print_number_format(self):
        _, io = run(['10 PRINT 1; 2.5; -3', '20 PRINT 7 / 2; 1E20'])
        lines = printed(io).splitlines()
        self.assertEqual(lines[0], ' 1  2.5 -3 ')
        self.assertEqual(lines[1], ' 3.5  1E+20 ')

    def test_select_case_forms(self):
        def pick(value):
            _, io = run([f'10 x = {value}', '20 SELECT CASE x', '30 CASE 1, 2: PRINT "low"',
                         '40 CASE 3 TO 8', '50 PRINT "mid"', '60 CASE IS > 8', '70 PRINT "high"',
                         '80 CASE ELSE', '90 PRINT "other"', '100 END SELECT'])
            return printed(io).split()
        self.assertEqual([pick(v) for v in (2, 5, 9, -1)],
                         [['low'], ['mid'], ['high'], ['other']])

    def test_select_case_strings_are_exact(self):
        _, io = run(['10 s$ = "abc"', '20 SELECT CASE s$', '30 CASE "ABC": PRINT "upper"',
                     '40 CASE "abc": PRINT "lower"', '50 END SELECT'])
        self.assertEqual(printed(io).split(), ['lower'])


class TestLiveExecution(unittest.TestCase):

    def test_print_reads_the_measured_bit(self):
        t, io = run(['10 X 0', '20 MEAS 0 -> m', '30 PRINT "m ="; m', '40 MEASURE'])
        self.assertIn('m = 1', io.text)
        self.assertIn('method=live', io.text)
        self.assertEqual(t.last_counts, {'1': 64})

    def test_only_the_first_shot_prints(self):
        _, io = run(['10 H 0', '20 MEAS 0 -> m', '30 PRINT "outcome"; m', '40 MEASURE'],
                    shots=50)
        self.assertEqual(io.text.count('outcome'), 1)

    def test_input_is_asked_once_and_replayed(self):
        t, io = run(['10 INPUT "angle", a', '20 RY a, 0', '30 MEAS 0 -> m',
                     '40 PRINT m', '50 MEASURE'], shots=40, answers=['3.14159'])
        self.assertEqual(io.reads, 1)
        self.assertEqual(sum(t.last_counts.values()), 40)
        self.assertGreater(t.last_counts.get('1', 0), 35)

    def test_repeat_until_success(self):
        t, _ = run(['10 H 0', '20 MEAS 0 -> m', '30 IF m = 1 THEN GOTO 50',
                    '40 RESET 0: GOTO 10', '50 MEASURE'], shots=100)
        self.assertEqual(t.last_counts, {'1': 100})

    def test_bell_bits_agree(self):
        _, io = run(['10 H 0: CX 0, 1', '20 MEAS 0 -> a', '30 MEAS 1 -> b',
                     '40 IF a <> b THEN PRINT "MISMATCH"', '50 MEASURE'], nq=2, shots=100)
        self.assertNotIn('MISMATCH', io.text)

    def test_syndrome_value_is_real(self):
        t, _ = run(['10 X 0', '20 SYNDROME ZZ 0 1 -> s', '30 v = s'], nq=3, shots=8)
        self.assertEqual(t.variables['v'], 1)

    def test_save_expect_is_the_shot_mean(self):
        t, _ = run(['10 H 0', '20 MEAS 0 -> m', '30 SAVE_EXPECT Z 0 -> e', '40 PRINT m',
                    '50 MEASURE'], shots=400)
        self.assertLess(abs(t.variables['e']), 0.2)

    def test_readout_noise_flips_recorded_bits(self):
        def noisy(t):
            t.cmd_noise('readout 0.2 0.2')
        t, _ = run(['10 MEAS 0 -> m', '20 c = m', '30 MEASURE'], shots=2000, setup=noisy)
        frac = t.last_counts.get('1', 0) / 2000
        self.assertGreater(frac, 0.15)
        self.assertLess(frac, 0.25)

    def test_feedforward_still_compiles(self):
        t, io = run(['10 H 0', '20 MEAS 0 -> m', '30 IF m = 1 THEN X 1', '40 MEASURE'],
                    nq=2, shots=200)
        self.assertNotIn('method=live', io.text)
        self.assertEqual(set(t.last_counts), {'00', '11'})

    def test_measurement_in_a_loop_compiles(self):
        t, io = run(['10 FOR i = 1 TO 3', '20 H 0', '30 MEAS 0 -> m', '40 IF m THEN X 0',
                     '50 NEXT i', '60 MEASURE'], shots=100)
        self.assertNotIn('method=live', io.text)
        self.assertEqual(t.last_counts, {'0': 100})

    def test_build_fallback_prints_nothing_twice(self):
        def blind(t):
            t._needs_live = lambda lines: False
        t, io = run(['10 PRINT "start"', '20 H 0', '30 MEAS 0 -> m', '40 PRINT m',
                     '50 MEASURE'], shots=20, setup=blind)
        self.assertEqual(io.text.count('start'), 1)
        self.assertIn('method=live', io.text)

    def test_step_collapses_on_meas(self):
        t = QBasicTerminal()
        t.num_qubits = 2
        t.io = _IO()
        t._step_auto, t._step_delay = True, 0
        for line in ['10 H 0', '20 MEAS 0 -> m', '30 IF m = 1 THEN X 1']:
            t.process(line, track_undo=False)
        t.cmd_step()
        sv = np.asarray(t.last_sv)
        probs = np.abs(sv) ** 2
        self.assertAlmostEqual(float(probs[0] + probs[3]), 1.0, places=9)
        self.assertTrue(probs[0] in (0.0, 1.0) or abs(probs[0] - 1) < 1e-9 or probs[0] < 1e-9)


class TestRunPaths(unittest.TestCase):

    def test_classical_program_skips_the_simulator(self):
        t, _ = run(['10 FOR i = 1 TO 50', '20 s = s + i', '30 NEXT i'], nq=4)
        t0 = time.time()
        t.cmd_run()
        self.assertLess(time.time() - t0, 0.15)
        self.assertEqual(complex(np.asarray(t.last_sv)[0]), 1)

    def test_applychannel_leaves_method_and_keeps_mixed_state(self):
        def channel(t):
            t.cmd_channel(f'AD = [[1,0],[0,{math.sqrt(0.9)}]] ; [[0,{math.sqrt(0.1)}],[0,0]]')
        t, io = run(['10 X 0', '20 APPLYCHANNEL AD 0'], setup=channel)
        self.assertEqual(t.sim_method, 'automatic')
        self.assertIn('DENSITY', io.text)
        io.parts.clear()
        t.cmd_density('')
        self.assertIn('0.900', io.text)

    def test_locc_prints_once(self):
        t = QBasicTerminal()
        t.io = _IO()
        t.shots = 30
        t.cmd_locc('JOINT 1 1')
        for line in ['10 SHARE A 0, B 0', '20 SEND A 0 -> c', '30 PRINT "c is"; c',
                     '40 MEASURE']:
            t.process(line, track_undo=False)
        t.cmd_run()
        self.assertEqual(t.io.text.count('c is'), 1)


class TestWebRepl(unittest.TestCase):

    def test_run_line_captures_output_and_input_does_not_block(self):
        from qubasic_core.web_repl import BufferIO, run_line
        t = QBasicTerminal()
        t.num_qubits = 1
        out = run_line(t, '10 INPUT "x", x', BufferIO())
        out = run_line(t, '20 PRINT x + 1', BufferIO())
        out = run_line(t, 'RUN', BufferIO())
        self.assertIn(' 1 ', out)


if __name__ == '__main__':
    unittest.main()
