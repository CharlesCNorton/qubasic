"""The BASIC language: syntax and semantics, QBASIC compatibility, the
keyword statements, files, graphics and sound, SAVE and LOAD, program editing
and the debugger."""

import math
import os
import tempfile
import time
import unittest
import wave
from unittest import mock

import numpy as np
from conftest import InTempDir, ScriptIO, feed, run_basic, term

from qubasic_core import api
from qubasic_core.api import CaptureIO
from qubasic_core.devices import (
    MODES,
    Canvas,
    PlayState,
    note_frequency,
    parse_play,
    parse_png,
    png_bytes,
    synthesize,
    wav_bytes,
)
from qubasic_core.editing import number_source
from qubasic_core.expression import format_using, number_text, print_text, str_text
from qubasic_core.interpreter import _input_fields, _parse_input_number, _resolve_named_state, error_code
from qubasic_core.parser import CompileError, compile_program, parse_stmt, strip_comment
from qubasic_core.terminal import QBasicTerminal

# ---------------------------------------------------------------------------
# QBASIC language fidelity: formatting, types, arrays, procedures, labels, I/O.
# ---------------------------------------------------------------------------


def run(source, answers=(), nq=1, seed=None):
    """Run QBASIC source lines (numbered 10, 20, ... when unnumbered)."""
    t = QBasicTerminal()
    t.num_qubits = nq
    t.shots = 16
    t._seed = seed
    io = ScriptIO(answers)
    t.io = io
    for num, text in number_source(source.strip().splitlines()):
        t.process(f"{num} {text}", track_undo=False)
    t.cmd_run()
    return t, io.text.split('\nRAN ')[0], io


class TestNumberFormatting(unittest.TestCase):

    def test_print_and_str(self):
        cases = [(0.5, '.5'), (-0.25, '-.25'), (1 / 3, '.3333333'), (12345678.0, '1.234568E+07'),
                 (1e-5, '1E-05'), (1e20, '1E+20'), (3.0, '3'), (7, '7'), (0.0001, '.0001'),
                 (1234567.0, '1234567'), (2.5e-05, '2.5E-05')]
        for value, want in cases:
            with self.subTest(value=value):
                self.assertEqual(number_text(value), want)
        self.assertEqual(print_text(0.5), ' .5 ')
        self.assertEqual(print_text(-2), '-2 ')
        self.assertEqual(str_text(3.5), ' 3.5')
        self.assertEqual(str_text(-1), '-1')

    def test_print_using_reference_cases(self):
        cases = [
            ("##.##", [.78], " 0.78"), ("###.##", [987.654], "987.65"),
            ("##.## ", [10.2, 5.3, 66.789, .234], "10.20  5.30 66.79  0.23 "),
            ("+##.## ", [-68.95, 2.4, 55.6, -.9], "-68.95  +2.40 +55.60  -0.90 "),
            ("##.##- ", [-68.95, 22.449, -7.01], "68.95- 22.45   7.01- "),
            ("**#.# ", [12.39, -0.9, 765.1], "*12.4 *-0.9 765.1 "),
            ("$$###.##", [456.78], " $456.78"), ("**$##.##", [2.34], "***$2.34"),
            ("####,.##", [1234.5], "1,234.50"), ("##.##^^^^", [234.56], " 2.35E+02"),
            (".####^^^^-", [-888888], ".8889E+06-"), ("+.##^^^^", [123], "+.12E+03"),
            ("##.##", [111.22], "%111.22"), (".##", [.999], "%1.00"),
            ("!", ["abc"], "a"), ("\\  \\", ["abcdef"], "abcd"), ("&", ["hello"], "hello"),
            ("_!##", [5], "! 5"), ("Total: ###", [42], "Total:  42"),
        ]
        for fmt, values, want in cases:
            with self.subTest(fmt=fmt):
                self.assertEqual(format_using(fmt, values), want)

    def test_print_using_type_checks(self):
        with self.assertRaises(ValueError):
            format_using("##", ["text"])
        with self.assertRaises(ValueError):
            format_using("&", [5])
        with self.assertRaises(ValueError):
            format_using("no fields", [1])
        self.assertEqual(format_using("plain", []), "plain")

    def test_program_output(self):
        _, out, _ = run('PRINT 1/3; 0.5; -0.25\nPRINT USING "##.##"; 3.14159\n'
                        'f$ = "$$##.##"\nPRINT USING f$; 2.5;\nPRINT "|"')
        self.assertEqual(out, ' .3333333  .5 -.25 \n 3.14\n  $2.50|\n')


class TestConversions(unittest.TestCase):

    def test_round_half_even(self):
        t = QBasicTerminal()
        self.assertEqual([t._safe_eval(e) for e in
                          ('CINT(2.5)', 'CINT(3.5)', 'CINT(-2.5)', 'CLNG(1.5)',
                           '5.5 \\ 1', '2.5 MOD 2', '7 AND 3.5', '5 EQV 3', '5 IMP 3')],
                         [2, 4, -2, 2, 6, 0, 4, -7, -5])

    def test_cint_overflow(self):
        t = QBasicTerminal()
        with self.assertRaises(ValueError):
            t._safe_eval('CINT(40000)')
        self.assertEqual(t._safe_eval('CLNG(40000)'), 40000)

    def test_typed_variables(self):
        t, out, _ = run('DEFINT A-C\na = 3.7\nDIM y AS DOUBLE\ny = 3.7\nz! = 2.5\nn& = 2.5\n'
                        'PRINT a; y; z!; n&')
        self.assertEqual(out.split(), ['4', '3.7', '2.5', '2'])
        self.assertIsInstance(t.variables['a'], int)

    def test_overflow_and_type_mismatch_codes(self):
        _, out, _ = run('ON ERROR GOTO 90\nx% = 40000\nEND\n90 PRINT ERR\nRESUME NEXT')
        self.assertEqual(out.split(), ['6'])
        _, out, _ = run('ON ERROR GOTO 90\ns$ = 5\nEND\n90 PRINT ERR\nRESUME NEXT')
        self.assertEqual(out.split(), ['13'])

    def test_defstr(self):
        t, _, _ = run('DEFSTR S\nsname = "ada"')
        self.assertEqual(t.variables['sname'], 'ada')


class TestArrays(unittest.TestCase):

    def test_bounds_and_lbound_ubound(self):
        _, out, _ = run('DIM a(1 TO 3, 0 TO 1) AS INTEGER\na(2, 1) = 7.6\n'
                        'PRINT a(2, 1); LBOUND(a); UBOUND(a); LBOUND(a, 2); UBOUND(a, 2)\n'
                        'n = 4\nDIM b$(n), c(2 TO n)\nPRINT UBOUND(b$); LBOUND(c); "["; b$(4); "]"')
        self.assertEqual(out.split('\n')[0].split(), ['8', '1', '3', '0', '1'])
        self.assertEqual(out.split('\n')[1].split(), ['4', '2', '[]'])

    def test_subscript_out_of_range(self):
        _, out, _ = run('ON ERROR GOTO 90\nDIM a(2 TO 4)\na(1) = 5\nEND\n90 PRINT ERR\nRESUME NEXT')
        self.assertEqual(out.split(), ['9'])

    def test_implicit_arrays(self):
        t, out, _ = run('PRINT q(3)\nm(1, 2) = 5\nPRINT m(1, 2); UBOUND(m, 2)\ng(20) = 1\n'
                        'PRINT UBOUND(g)')
        self.assertEqual(out.split(), ['0', '5', '10', '20'])

    def test_redim_preserve_and_erase(self):
        t, _, _ = run('DIM a(3)\na(1) = 9\nREDIM PRESERVE a(1 TO 5)\nv = a(1)\nERASE a\n'
                      'DIM b(2), c(2)\nERASE b, c')
        self.assertEqual(t.variables['v'], 0.0)
        for name in ('a', 'b', 'c'):
            self.assertNotIn(name, t.arrays)

    def test_swap_elements_and_gate(self):
        t, out, _ = run('DIM a(3)\na(1) = 1: a(2) = 2\nSWAP a(1), a(2)\nPRINT a(1); a(2)')
        self.assertEqual(out.split(), ['2', '1'])
        t, _, _ = run('X 0\nSWAP 0, 1\nMEASURE', nq=2)
        self.assertEqual(set(t.last_counts), {'10'})


class TestConstAndDeclarations(unittest.TestCase):

    def test_const(self):
        t, out, _ = run('CONST pi2 = 6.28, msg = "hi"\nPRINT pi2; msg\n'
                        'ON ERROR GOTO 90\npi2 = 1\nEND\n90 PRINT ERR')
        self.assertEqual(out.split(), ['6.28', 'hi', '10'])

    def test_accepted_statements(self):
        _, out, _ = run('DECLARE SUB Foo ()\nWIDTH 80\nCLS\nCOLOR 2, 0\nLOCATE 1, 1\nSCREEN 0\n'
                        'SLEEP 0\nFoo\nSYSTEM\nPRINT "never"\nSUB Foo\nPRINT "foo"\nEND SUB')
        self.assertEqual(out.strip(), 'foo')

    def test_beep_and_question_mark(self):
        _, out, _ = run('BEEP\n? "q"; 1')
        self.assertEqual(out, '\aq 1 \n')

    def test_classical_program_has_no_run_summary(self):
        t = QBasicTerminal()
        io = ScriptIO()
        t.io = io
        t.process('10 PRINT "only this"', track_undo=False)
        t.cmd_run()
        self.assertEqual(io.text, 'only this\n')


class TestLabelsAndJumps(unittest.TestCase):

    def test_goto_gosub_labels(self):
        _, out, _ = run('GOSUB Show\nGOTO Done\nShow: PRINT "in sub"\nRETURN\nDone:\nPRINT "done"')
        self.assertEqual(out.split(), ['in', 'sub', 'done'])

    def test_on_goto_and_restore_labels(self):
        _, out, _ = run('k = 2\nON k GOTO A1, B1\nA1: PRINT "a": END\nB1: READ x\nRESTORE Second\n'
                        'READ y\nPRINT x; y\nFirst: DATA 1\nSecond: DATA 2')
        self.assertEqual(out.split(), ['1', '2'])

    def test_on_error_and_resume_labels(self):
        _, out, _ = run('ON ERROR GOTO Handler\nx = 1 / 0\nPRINT "after"\nEND\n'
                        'Handler: PRINT "trapped"; ERR\nRESUME NEXT')
        self.assertEqual(out.split(), ['trapped', '11', 'after'])

    def test_return_to_line(self):
        _, out, _ = run('10 GOSUB 100\n20 PRINT "skipped"\n30 PRINT "landed"\n40 END\n'
                        '100 RETURN 30')
        self.assertEqual(out.split(), ['landed'])

    def test_restore_line(self):
        _, out, _ = run('READ a, b\nRESTORE 60\nREAD c\nPRINT a; b; c\n50 DATA 9, 8\n60 DATA 7')
        self.assertEqual(out.split(), ['9', '8', '7'])

    def test_label_errors(self):
        with self.assertRaises(CompileError):
            compile_program({10: 'GOTO Nowhere'})
        with self.assertRaises(CompileError):
            compile_program({10: 'A1: PRINT 1', 20: 'A1: PRINT 2'})
        prog = compile_program({10: 'END: PRINT 1'})
        self.assertEqual(prog.labels, {})


class TestProcedures(unittest.TestCase):

    def test_local_scope_and_byref(self):
        _, out, _ = run('x = 5\nCALL Bump(x)\nPRINT x\nCALL Show\nCALL Bump((x))\nPRINT x\nEND\n'
                        'SUB Bump(v)\n v = v + 1\nEND SUB\nSUB Show\n PRINT "x inside:"; x\nEND SUB')
        self.assertEqual(out.split(), ['6', 'x', 'inside:', '0', '6'])

    def test_byref_array_element(self):
        _, out, _ = run('DIM a(2)\na(1) = 10\nInc a(1)\nPRINT a(1)\nSUB Inc(v)\nv = v + 1\nEND SUB')
        self.assertEqual(out.split(), ['11'])

    def test_dim_shared_and_shared(self):
        _, out, _ = run('DIM SHARED total\ntotal = 1\nAdd 5\nAdd 2\nPRINT total\ny = 3\nUseY\n'
                        'PRINT y\nEND\nSUB Add (n)\n total = total + n\nEND SUB\n'
                        'SUB UseY\n SHARED y\n y = y * 2\nEND SUB')
        self.assertEqual(out.split(), ['8', '6'])

    def test_static_routine_and_static_var(self):
        _, out, _ = run('Counter\nCounter\nTick\nTick\nPRINT\nSUB Counter STATIC\n k = k + 1\n'
                        ' PRINT k;\nEND SUB\nSUB Tick\n STATIC n\n n = n + 10\n PRINT n;\nEND SUB')
        self.assertEqual(out.split(), ['1', '2', '10', '20'])

    def test_recursive_function_with_suffix(self):
        _, out, _ = run('PRINT Fact&(10); Greet$("ada")\nFUNCTION Fact& (n)\n'
                        ' IF n <= 1 THEN Fact& = 1 ELSE Fact& = n * Fact&(n - 1)\nEND FUNCTION\n'
                        'FUNCTION Greet$ (n$)\n Greet$ = "hi " + n$\nEND FUNCTION')
        self.assertEqual(out.split(), ['3628800', 'hi', 'ada'])

    def test_function_calls_a_sub(self):
        _, out, _ = run('PRINT Twice(4)\nFUNCTION Twice (v)\n DIM SHARED tmp\n Helper v\n'
                        ' Twice = v * 2\nEND FUNCTION\nSUB Helper (w)\n PRINT "helper"; w;\nEND SUB')
        self.assertEqual(out.split(), ['helper', '4', '8'])

    def test_argument_count_mismatch(self):
        _, out, _ = run('ON ERROR GOTO 90\nCALL Two(1)\nEND\n90 PRINT ERR\nSUB Two(a, b)\nEND SUB')
        self.assertEqual(out.split(), ['37'])


class TestInputOutput(unittest.TestCase):

    def test_input_multiple_values_and_prompts(self):
        _, out, io = run('INPUT "Two numbers"; a, b\nINPUT n$\nINPUT "Name", m$\nPRINT a + b; n$; m$',
                         answers=['3, 4', 'zed', 'kim'])
        self.assertEqual(out.split(), ['7', 'zedkim'])
        self.assertEqual(io.prompts, ['Two numbers? ', '? ', 'Name'])

    def test_input_redo_from_start(self):
        _, out, io = run('INPUT a, b\nPRINT a; b', answers=['1', 'x, 2', '5, 6'])
        self.assertEqual(out.count('?Redo from start'), 2)
        self.assertTrue(out.endswith(' 5  6 \n'))

    def test_line_input(self):
        t, _, io = run('LINE INPUT "Text: "; s$', answers=['a, "b", c'])
        self.assertEqual(t.variables['s$'], 'a, "b", c')
        self.assertEqual(io.prompts, ['Text: '])

    def test_write_and_mid_assignment(self):
        _, out, _ = run('WRITE 1, "a", 2.5\ns$ = "hello world"\nMID$(s$, 1, 5) = "HOWDY"\nPRINT s$')
        self.assertEqual(out, '1,"a",2.5\nHOWDY world\n')

    def test_file_statements(self):
        with tempfile.TemporaryDirectory(dir='.') as d:
            rel = os.path.relpath(os.path.join(d, 'data.txt'))
            _, out, _ = run(f'f = FREEFILE\nOPEN "{rel}" FOR OUTPUT AS #1\nWRITE #1, 1, "two"\n'
                            f'PRINT #1, 3.5; "x"\nPRINT #1, USING "##.#"; 2.25\nCLOSE #1\n'
                            f'OPEN "{rel}" FOR INPUT AS #1\nn = LOF(1)\nINPUT #1, a, b$\n'
                            f'LINE INPUT #1, c$\nINPUT #1, d\nPRINT f; a; b$; "|"; c$; "|"; d; EOF(1); n > 0\n'
                            f'CLOSE #1')
            self.assertEqual(out, ' 1  1 two| 3.5 x| 2.3 -1 -1 \n')
            with open(rel, encoding='utf-8') as f:
                self.assertEqual(f.read(), '1,"two"\n 3.5 x\n 2.3\n')

    def test_input_past_end(self):
        with tempfile.TemporaryDirectory(dir='.') as d:
            rel = os.path.relpath(os.path.join(d, 'e.txt'))
            open(rel, 'w').close()
            _, out, _ = run(f'ON ERROR GOTO 90\nOPEN "{rel}" FOR INPUT AS #1\nINPUT #1, a\nEND\n'
                            f'90 PRINT ERR\nCLOSE #1')
            self.assertEqual(out.split()[-1], '62')


class TestFunctions(unittest.TestCase):

    def test_string_functions(self):
        t = QBasicTerminal()
        got = [t._safe_eval(e) for e in (
            'UCASE$("ab")', 'LCASE$("CD")', 'LTRIM$("  x ")', 'RTRIM$(" y  ")', 'SPACE$(2)',
            'STRING$(3, "*")', 'STRING$(2, 65)', 'OCT$(8)', 'HEX$(255)', 'INSTR(3, "abcabc", "b")',
            'INSTR("abc", "z")', 'VAL("12ab")', 'VAL(" 1 2 ")', 'VAL("&HFF")', 'VAL("&O17")',
            'VAL("-1.5E2x")', 'STR$(.5)', 'MID$("hello", 2, 3)')]
        self.assertEqual(got, ['AB', 'cd', 'x ', ' y', '  ', '***', 'AA', '10', 'FF', 5, 0, 12.0,
                               12.0, 255.0, 15.0, -150.0, ' .5', 'ell'])

    def test_rnd_randomize_and_seed(self):
        _, out, _ = run('RANDOMIZE 7\na = RND\nRANDOMIZE 7\nb = RND\nPRINT a = b; RND(0) = b')
        self.assertEqual(out.split(), ['-1', '-1'])
        t1 = QBasicTerminal()
        t1.cmd_seed('5')
        t2 = QBasicTerminal()
        t2.cmd_seed('5')
        self.assertEqual(t1._safe_eval('RND'), t2._safe_eval('RND'))

    def test_clock_functions(self):
        t = QBasicTerminal()
        self.assertTrue(0 <= t._safe_eval('TIMER') < 86401)
        self.assertRegex(t._safe_eval('DATE$'), r'^\d\d-\d\d-\d{4}$')
        self.assertRegex(t._safe_eval('TIME$'), r'^\d\d:\d\d:\d\d$')
        self.assertEqual(t._safe_eval('INKEY$'), '')

    def test_error_codes(self):
        for msg, code in [('ERROR 42', 42), ('OVERFLOW: x', 6), ('LINE 20: SUBSCRIPT OUT OF RANGE', 9),
                          ('DIVISION BY ZERO', 11), ('SQR: math domain error', 5),
                          ('RETURN WITHOUT GOSUB', 3), ('OUT OF DATA', 4), ('unknown', 1)]:
            with self.subTest(msg=msg):
                self.assertEqual(error_code(Exception(msg)), code)


class TestBasSource(unittest.TestCase):

    def test_number_source(self):
        self.assertEqual(number_source(['PRINT 1', '', '100 PRINT 2', 'GOTO 100', 'x = 1']),
                         [(10, 'PRINT 1'), (100, 'PRINT 2'), (110, 'GOTO 100'), (120, 'x = 1')])
        self.assertEqual(number_source(['a', 'b', '5 c']), [(1, 'a'), (2, 'b'), (5, 'c')])
        with self.assertRaises(ValueError):
            number_source(['a'] * 11 + ['11 z'])
        with self.assertRaises(ValueError):
            number_source(['20 a', '10 b'])

    def test_classic_example_runs_unchanged(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.relpath(os.path.join(here, '..', 'examples', 'classic.bas'))
        t = QBasicTerminal()
        io = ScriptIO()
        t.io = io
        t._load_bas(path)
        t.cmd_run()
        self.assertEqual(io.text, 'Sorted: 3  4  5  7  9  15  26  31 \n'
                                  'Primes: 3  5  7  31 \nMean: 12.50\nmedium\n')

    def test_save_and_load_bas(self):
        with tempfile.TemporaryDirectory(dir='.') as d:
            rel = os.path.relpath(os.path.join(d, 'p.bas'))
            t = QBasicTerminal()
            t.io = ScriptIO()
            t.process('10 PRINT "x"', track_undo=False)
            t.process('20 GOTO 40', track_undo=False)
            t.process('40 END', track_undo=False)
            t.cmd_save(rel)
            t2 = QBasicTerminal()
            t2.io = ScriptIO()
            t2.cmd_load(rel)
            self.assertEqual(t2.program, {10: 'PRINT "x"', 20: 'GOTO 40', 40: 'END'})


RECORDS = '''
TYPE Vec
    x AS SINGLE
    y AS SINGLE
END TYPE
TYPE Body
    pos AS Vec
    name AS STRING * 6
    hits AS INTEGER
END TYPE
'''


class TestRecords(unittest.TestCase):

    def test_fields_types_and_copies(self):
        t, out, _ = run(RECORDS + '''
DIM b AS Body, crowd(1 TO 3) AS Body
b.pos.x = 1.5: b.name = "planetoid": b.hits = 2.5
crowd(2).pos.y = -4: crowd(2).name = "ab"
crowd(3) = b
b.hits = 9
PRINT b.pos.x; "["; b.name; "]"; b.hits
PRINT crowd(2).pos.y; "["; crowd(2).name; "]"; crowd(3).hits
SWAP crowd(1), crowd(2)
PRINT crowd(1).pos.y; crowd(2).pos.y''')
        self.assertEqual(out, ' 1.5 [planet] 9 \n-4 [ab    ] 2 \n-4  0 \n')
        self.assertEqual(t.variables['b'].type_name, 'BODY')

    def test_input_read_and_byref_fields(self):
        _, out, _ = run(RECORDS + '''
DIM v AS Vec, path(2) AS Vec
INPUT v.x, path(1).y
READ path(2).x
Nudge v.x
Nudge path(1).y
PRINT v.x; path(1).y; path(2).x
DATA 7
SUB Nudge (n AS SINGLE)
    n = n + 0.5
END SUB''', answers=['1, 2'])
        self.assertEqual(out.split(), ['1.5', '2.5', '7'])

    def test_record_parameter_by_reference(self):
        _, out, _ = run(RECORDS + '''
DIM a AS Vec
a.x = 3
Scale a, 2
PRINT a.x
SUB Scale (p AS Vec, k AS INTEGER)
    p.x = p.x * k
END SUB''')
        self.assertEqual(out.split(), ['6'])

    def test_type_errors(self):
        with self.assertRaises(CompileError):
            compile_program({10: 'TYPE T', 20: 'x AS INTEGER'})
        with self.assertRaises(CompileError):
            compile_program({10: 'TYPE T', 20: 'x INTEGER', 30: 'END TYPE'})
        _, out, _ = run('ON ERROR GOTO 90\nDIM p AS Nope\nEND\n90 PRINT "trapped"')
        self.assertIn('trapped', out)
        _, out, _ = run(RECORDS + 'ON ERROR GOTO 90\nDIM v AS Vec\nv = 5\nEND\n90 PRINT ERR')
        self.assertEqual(out.split(), ['13'])

    def test_fixed_length_strings(self):
        _, out, _ = run('DIM s AS STRING * 4, names(2) AS STRING * 3\ns = "abcdef"\n'
                        'names(1) = "z"\nPRINT "["; s; "]["; names(1); "]["; names(2); "]"')
        self.assertEqual(out, '[abcd][z  ][   ]\n')

    def test_repl_type_and_record(self):
        t = QBasicTerminal()
        t.io = ScriptIO()
        for line in ('TYPE Pt', '  x AS INTEGER', '  tag AS STRING * 2', 'END TYPE'):
            t.process(line, track_undo=False)
        t.dispatch('DIM p AS Pt')
        t.dispatch('p.x = 2.5')
        t.dispatch('p.tag = "abc"')
        self.assertEqual(t.variables['p'], {'x': 2, 'tag': 'ab'})


class TestProcedureParameters(unittest.TestCase):

    def test_typed_and_array_parameters(self):
        t, out, _ = run('''
DECLARE SUB Total (a() AS INTEGER, n AS INTEGER, s AS LONG)
DIM v(1 TO 4) AS INTEGER
FOR i = 1 TO 4: v(i) = i * 10: NEXT
s& = 0
Total v(), 4, s&
PRINT s&
CALL Grow(v())
PRINT UBOUND(v); v(6)
PRINT Avg!(v())
END
SUB Total (a() AS INTEGER, n AS INTEGER, s AS LONG)
    FOR k = 1 TO n: s = s + a(k): NEXT
    DIM scratch(5)
END SUB
SUB Grow (a())
    REDIM PRESERVE a(1 TO 6) AS INTEGER
    a(6) = 99
END SUB
FUNCTION Avg! (a() AS INTEGER)
    t = 0
    FOR k = LBOUND(a) TO UBOUND(a): t = t + a(k): NEXT
    Avg! = t / (UBOUND(a) - LBOUND(a) + 1)
END FUNCTION''')
        self.assertEqual(out.split(), ['100', '6', '99', '33.16667'])
        self.assertEqual(sorted(t.arrays), ['v'])          # the SUB's own array is gone

    def test_parameter_type_is_enforced(self):
        _, out, _ = run('ON ERROR GOTO 90\nShow 2.5\nEND\n90 PRINT ERR\nRESUME NEXT\n'
                        'SUB Show (s AS STRING)\nPRINT s\nEND SUB')
        self.assertEqual(out.split(), ['13'])
        _, out, _ = run('Half 7\nSUB Half (n AS INTEGER)\nPRINT n / 2\nEND SUB')
        self.assertEqual(out.split(), ['3.5'])

    def test_array_argument_errors(self):
        _, out, _ = run('ON ERROR GOTO 90\nx = 1\nSortIt x\nEND\n90 PRINT ERR\n'
                        'SUB SortIt (a())\nEND SUB')
        self.assertEqual(out.split(), ['13'])


class TestEditingAndChecks(unittest.TestCase):

    def test_renum_rewrites_every_reference(self):
        t = QBasicTerminal()
        t.io = ScriptIO()
        t.program = {5: 'GOSUB 30', 7: 'IF x THEN 30 ELSE 40', 9: 'ON k GOTO 30, 40',
                     11: 'RESTORE 40', 13: 'PRINT "GOTO 30"', 15: "REM GOTO 30",
                     30: 'RETURN 40', 40: 'DATA 1: ON ERROR GOTO 30: RESUME 40'}
        t.cmd_renum('100 10')
        self.assertEqual(t.program, {
            100: 'GOSUB 160', 110: 'IF x THEN 160 ELSE 170', 120: 'ON k GOTO 160, 170',
            130: 'RESTORE 170', 140: 'PRINT "GOTO 30"', 150: 'REM GOTO 30',
            160: 'RETURN 170', 170: 'DATA 1: ON ERROR GOTO 160: RESUME 170'})

    def test_missing_targets_are_compile_errors(self):
        for text in ('ON k GOTO 10, 99', 'ON ERROR GOTO 99', 'RETURN 99', 'RESTORE 99',
                     'RESUME 99'):
            with self.subTest(text=text):
                with self.assertRaises(CompileError):
                    compile_program({10: text})
        compile_program({10: 'ON ERROR GOTO 0', 20: 'RESUME 0'})

    def test_resume_zero_retries(self):
        _, out, _ = run('ON ERROR GOTO 90\nn = 0\nPRINT 10 / n\nEND\n90 n = 2\nRESUME 0')
        self.assertEqual(out.split(), ['5'])

    def test_prompt_let_is_typed(self):
        t = QBasicTerminal()
        t.io = ScriptIO()
        t.dispatch('n% = 2.5')
        t.dispatch('LET big& = 1E12')
        t.dispatch('s$ = 3')
        self.assertEqual(t.variables['n%'], 2)
        self.assertNotIn('big&', t.variables)
        self.assertIn('OVERFLOW', t.io.text)
        self.assertIn('TYPE MISMATCH', t.io.text)

    def test_save_round_trips_session_state(self):
        with tempfile.TemporaryDirectory(dir='.') as d:
            rel = os.path.relpath(os.path.join(d, 's.qb'))
            t = QBasicTerminal()
            t.io = ScriptIO()
            t.num_qubits = 3
            t.cmd_precision('SINGLE')
            t.cmd_noise('depolarizing 0.001 0.01; readout 0.02')
            t.cmd_coupling('linear')
            t.cmd_basis('rz, sx, cx')
            for line in ('TYPE Pt', 'x AS INTEGER', 'END TYPE'):
                t.process(line, track_undo=False)
            t.dispatch('DIM p AS Pt')
            t.dispatch('p.x = 4')
            t.dispatch('msg$ = "say " + CHR$(34) + "hi" + CHR$(34)')
            t.cmd_save(rel)
            t2 = QBasicTerminal()
            t2.io = ScriptIO()
            t2.cmd_load(rel)
            self.assertEqual(t2.sim_precision, 'single')
            self.assertEqual(t2._noise_spec, 'depolarizing 0.001 0.01; readout 0.02')
            self.assertEqual(sorted(map(tuple, t2._coupling_map)), sorted(map(tuple, t._coupling_map)))
            self.assertEqual(t2._basis_gates, ['rz', 'sx', 'cx'])
            self.assertEqual(t2.variables['p'], {'x': 4})
            self.assertEqual(t2.variables['msg$'], 'say "hi"')

    def test_save_round_trips_device(self):
        with tempfile.TemporaryDirectory(dir='.') as d:
            rel = os.path.relpath(os.path.join(d, 'd.qb'))
            t = QBasicTerminal()
            t.io = ScriptIO()
            t.cmd_device('ring 4 80 60')
            t.cmd_save(rel)
            t2 = QBasicTerminal()
            t2.io = ScriptIO()
            t2.cmd_load(rel)
            self.assertEqual(t2._device_spec, 'ring 4 80 60')
            self.assertIsNotNone(t2._noise_model)
            self.assertNotIn('?', t2.io.text)


# ---------------------------------------------------------------------------
# BASIC statement semantics and live (shot-by-shot) execution.
# ---------------------------------------------------------------------------


def run_lines(lines, nq=1, shots=64, seed=11, answers=(), setup=None):
    t = QBasicTerminal()
    t.num_qubits = nq
    t.shots = shots
    t._seed = seed
    io = ScriptIO(answers)
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
        _, io = run_lines(['10 PRINT 1: PRINT 2', '20 LET a = 1: LET b = 2', '30 PRINT a + b'])
        self.assertEqual(printed(io).split(), ['1', '2', '3'])

    def test_reset_compound(self):
        t, _ = run_lines(['10 H 0', '20 RESET 0: X 0', '30 MEASURE'])
        self.assertEqual(set(t.last_counts), {'1'})

    def test_colon_inside_string_is_not_a_separator(self):
        _, io = run_lines(['10 PRINT "time: 5": PRINT "ok"'])
        self.assertEqual(printed(io).splitlines(), ['time: 5', 'ok'])

    def test_if_owns_rest_of_line(self):
        _, io = run_lines(['10 x = 1', '20 IF x = 2 THEN PRINT "a": PRINT "b"', '30 PRINT "c"'])
        self.assertEqual(printed(io).split(), ['c'])

    def test_jumps_leave_a_compound(self):
        _, io = run_lines(['10 PRINT "x": GOTO 30', '20 PRINT "skipped"', '30 PRINT "y": END',
                     '40 PRINT "after end"'])
        self.assertEqual(printed(io).split(), ['x', 'y'])

    def test_gosub_returns_to_next_statement(self):
        _, io = run_lines(['10 GOSUB 100: PRINT "back"', '20 END', '100 PRINT "sub": RETURN'])
        self.assertEqual(printed(io).split(), ['sub', 'back'])

    def test_gosub_inside_if_clause_returns_into_clause(self):
        _, io = run_lines(['10 IF 1 THEN GOSUB 100: PRINT "after"', '20 END',
                     '100 PRINT "in": RETURN'])
        self.assertEqual(printed(io).split(), ['in', 'after'])


class TestLoops(unittest.TestCase):

    def test_single_line_for(self):
        t, _ = run_lines(['10 FOR i = 1 TO 3: FOR j = 1 TO 2: k = k + 1: NEXT j: NEXT i'])
        self.assertEqual(t.variables['k'], 6)

    def test_for_opened_in_a_compound_closed_later(self):
        _, io = run_lines(['10 FOR i = 1 TO 2: PRINT "a"; i', '20 NEXT i'])
        self.assertEqual(printed(io).split(), ['a', '1', 'a', '2'])

    def test_zero_trip_for(self):
        _, io = run_lines(['10 FOR i = 5 TO 1', '20 PRINT "body"', '30 NEXT i', '40 PRINT "after"'])
        self.assertEqual(printed(io).split(), ['after'])

    def test_while_inside_for_keeps_the_loop_stack(self):
        t, io = run_lines(['10 FOR i = 1 TO 2', '20 x = 0', '30 WHILE x < 3', '40 x = x + 1',
                     '50 WEND', '60 NEXT i', '70 PRINT i; x'])
        self.assertEqual(printed(io).split(), ['3', '3'])

    def test_do_loop_until_on_one_line(self):
        t, _ = run_lines(['10 DO: n = n + 1: LOOP UNTIL n >= 4'])
        self.assertEqual(t.variables['n'], 4)

    def test_exit_for(self):
        t, _ = run_lines(['10 FOR i = 1 TO 9', '20 IF i = 4 THEN EXIT FOR', '30 NEXT', '40 r = i'])
        self.assertEqual(t.variables['r'], 4)


class TestExpressions(unittest.TestCase):

    def test_single_equals_compares_in_if(self):
        _, io = run_lines(['10 x = 2', '20 IF x = 2 THEN PRINT "yes" ELSE PRINT "no"'])
        self.assertEqual(printed(io).split(), ['yes'])

    def test_comparison_value(self):
        t, _ = run_lines(['10 flag = 3 = 3', '20 other = 2 = 3'])
        self.assertEqual((t.variables['flag'], t.variables['other']), (-1, 0))

    def test_unassigned_reads_zero_in_a_program(self):
        t, _ = run_lines(['10 count = count + 1', '20 s$ = t$ + "x"'])
        self.assertEqual((t.variables['count'], t.variables['s$']), (1, 'x'))

    def test_prompt_and_quantum_operands_stay_strict(self):
        t = QBasicTerminal()
        with self.assertRaises(ValueError):
            t._safe_eval('never_set')
        _, io = run_lines(['10 H q'])
        self.assertIn('CANNOT RESOLVE QUBIT', io.text)

    def test_basic_operators_and_functions(self):
        t = QBasicTerminal()
        values = [t._safe_eval(e) for e in
                  ('10 MOD 3', '-7 MOD 3', '2 ^ 10', '5 XOR 3', 'SGN(-4)', 'SQR(16)',
                   'CINT(2.5)', 'ATN(1) * 4')]
        self.assertEqual(values[:7], [1, -1, 1024, 6, -1, 4.0, 2])
        self.assertAlmostEqual(values[7], math.pi)

    def test_bare_print_is_a_blank_line(self):
        _, io = run_lines(['10 PRINT "a"', '20 PRINT', '30 PRINT "b"'])
        self.assertEqual(printed(io), 'a\n\nb\n')

    def test_print_number_format(self):
        _, io = run_lines(['10 PRINT 1; 2.5; -3', '20 PRINT 7 / 2; 1E20'])
        lines = printed(io).splitlines()
        self.assertEqual(lines[0], ' 1  2.5 -3 ')
        self.assertEqual(lines[1], ' 3.5  1E+20 ')

    def test_select_case_forms(self):
        def pick(value):
            _, io = run_lines([f'10 x = {value}', '20 SELECT CASE x', '30 CASE 1, 2: PRINT "low"',
                         '40 CASE 3 TO 8', '50 PRINT "mid"', '60 CASE IS > 8', '70 PRINT "high"',
                         '80 CASE ELSE', '90 PRINT "other"', '100 END SELECT'])
            return printed(io).split()
        self.assertEqual([pick(v) for v in (2, 5, 9, -1)],
                         [['low'], ['mid'], ['high'], ['other']])

    def test_select_case_strings_are_exact(self):
        _, io = run_lines(['10 s$ = "abc"', '20 SELECT CASE s$', '30 CASE "ABC": PRINT "upper"',
                     '40 CASE "abc": PRINT "lower"', '50 END SELECT'])
        self.assertEqual(printed(io).split(), ['lower'])


class TestLiveExecution(unittest.TestCase):

    def test_print_reads_the_measured_bit(self):
        t, io = run_lines(['10 X 0', '20 MEAS 0 -> m', '30 PRINT "m ="; m', '40 MEASURE'])
        self.assertIn('m = 1', io.text)
        self.assertIn('method=live', io.text)
        self.assertEqual(t.last_counts, {'1': 64})

    def test_only_the_first_shot_prints(self):
        _, io = run_lines(['10 H 0', '20 MEAS 0 -> m', '30 PRINT "outcome"; m', '40 MEASURE'],
                    shots=50)
        self.assertEqual(io.text.count('outcome'), 1)

    def test_input_is_asked_once_and_replayed(self):
        t, io = run_lines(['10 INPUT "angle", a', '20 RY a, 0', '30 MEAS 0 -> m',
                     '40 PRINT m', '50 MEASURE'], shots=40, answers=['3.14159'])
        self.assertEqual(io.reads, 1)
        self.assertEqual(sum(t.last_counts.values()), 40)
        self.assertGreater(t.last_counts.get('1', 0), 35)

    def test_repeat_until_success(self):
        t, _ = run_lines(['10 H 0', '20 MEAS 0 -> m', '30 IF m = 1 THEN GOTO 50',
                    '40 RESET 0: GOTO 10', '50 MEASURE'], shots=100)
        self.assertEqual(t.last_counts, {'1': 100})

    def test_bell_bits_agree(self):
        _, io = run_lines(['10 H 0: CX 0, 1', '20 MEAS 0 -> a', '30 MEAS 1 -> b',
                     '40 IF a <> b THEN PRINT "MISMATCH"', '50 MEASURE'], nq=2, shots=100)
        self.assertNotIn('MISMATCH', io.text)

    def test_syndrome_value_is_real(self):
        t, _ = run_lines(['10 X 0', '20 SYNDROME ZZ 0 1 -> s', '30 v = s'], nq=3, shots=8)
        self.assertEqual(t.variables['v'], 1)

    def test_save_expect_is_the_shot_mean(self):
        t, _ = run_lines(['10 H 0', '20 MEAS 0 -> m', '30 SAVE_EXPECT Z 0 -> e', '40 PRINT m',
                    '50 MEASURE'], shots=400)
        self.assertLess(abs(t.variables['e']), 0.2)

    def test_readout_noise_flips_recorded_bits(self):
        def noisy(t):
            t.cmd_noise('readout 0.2 0.2')
        t, _ = run_lines(['10 MEAS 0 -> m', '20 c = m', '30 MEASURE'], shots=2000, setup=noisy)
        frac = t.last_counts.get('1', 0) / 2000
        self.assertGreater(frac, 0.15)
        self.assertLess(frac, 0.25)

    def test_feedforward_still_compiles(self):
        t, io = run_lines(['10 H 0', '20 MEAS 0 -> m', '30 IF m = 1 THEN X 1', '40 MEASURE'],
                    nq=2, shots=200)
        self.assertNotIn('method=live', io.text)
        self.assertEqual(set(t.last_counts), {'00', '11'})

    def test_measurement_in_a_loop_compiles(self):
        t, io = run_lines(['10 FOR i = 1 TO 3', '20 H 0', '30 MEAS 0 -> m', '40 IF m THEN X 0',
                     '50 NEXT i', '60 MEASURE'], shots=100)
        self.assertNotIn('method=live', io.text)
        self.assertEqual(t.last_counts, {'0': 100})

    def test_build_fallback_prints_nothing_twice(self):
        def blind(t):
            t._needs_live = lambda lines: False
        t, io = run_lines(['10 PRINT "start"', '20 H 0', '30 MEAS 0 -> m', '40 PRINT m',
                     '50 MEASURE'], shots=20, setup=blind)
        self.assertEqual(io.text.count('start'), 1)
        self.assertIn('method=live', io.text)

    def test_step_collapses_on_meas(self):
        t = QBasicTerminal()
        t.num_qubits = 2
        t.io = ScriptIO()
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
        t, _ = run_lines(['10 FOR i = 1 TO 50', '20 s = s + i', '30 NEXT i'], nq=4)
        t0 = time.time()
        t.cmd_run()
        self.assertLess(time.time() - t0, 0.15)
        self.assertEqual(complex(np.asarray(t.last_sv)[0]), 1)

    def test_applychannel_leaves_method_and_keeps_mixed_state(self):
        def channel(t):
            t.cmd_channel(f'AD = [[1,0],[0,{math.sqrt(0.9)}]] ; [[0,{math.sqrt(0.1)}],[0,0]]')
        t, io = run_lines(['10 X 0', '20 APPLYCHANNEL AD 0'], setup=channel)
        self.assertEqual(t.sim_method, 'automatic')
        self.assertIn('DENSITY', io.text)
        io.parts.clear()
        t.cmd_density('')
        self.assertIn('0.900', io.text)

    def test_locc_prints_once(self):
        t = QBasicTerminal()
        t.io = ScriptIO()
        t.shots = 30
        t.cmd_locc('JOINT 1 1')
        for line in ['10 SHARE A 0, B 0', '20 SEND A 0 -> c', '30 PRINT "c is"; c',
                     '40 MEASURE']:
            t.process(line, track_undo=False)
        t.cmd_run()
        self.assertEqual(t.io.text.count('c is'), 1)


class TestWebRepl(unittest.TestCase):

    def test_run_line_captures_output_and_input_does_not_block(self):
        from qubasic_core.cli import BufferIO, run_line
        t = QBasicTerminal()
        t.num_qubits = 1
        out = run_line(t, '10 INPUT "x", x', BufferIO())
        out = run_line(t, '20 PRINT x + 1', BufferIO())
        out = run_line(t, 'RUN', BufferIO())
        self.assertIn(' 1 ', out)


# ---------------------------------------------------------------------------
# QBASIC source details: trailing comments, CONST over built-in names, LEN
# of typed variables and records, and the cursor column behind POS, TAB and
# the print zones.
# ---------------------------------------------------------------------------


class TestTrailingComments(unittest.TestCase):

    def test_strip_comment(self):
        self.assertEqual(strip_comment("PRINT 1 ' note"), 'PRINT 1')
        self.assertEqual(strip_comment('PRINT "it\'s" \' note'), 'PRINT "it\'s"')
        self.assertEqual(strip_comment("' only a comment"), "' only a comment")
        self.assertEqual(strip_comment("x = 1: REM it's: here"), "x = 1: REM it's: here")
        self.assertEqual(strip_comment("DATA don't, stop"), "DATA don't, stop")
        self.assertEqual(strip_comment("DATA a, b : ' note"), 'DATA a, b :')
        self.assertEqual(strip_comment("DATA a : PRINT 1 ' note"), 'DATA a : PRINT 1')

    def test_comments_after_any_statement(self):
        out, errors = run_basic('''
k = 0 ' start
FOR i = 1 TO 3 ' loop
  k = k + i ' add
NEXT i ' done
IF k = 6 THEN PRINT "six" ELSE PRINT "other" ' check
SELECT CASE k ' branch
  CASE 6: PRINT "case six" ' hit
END SELECT
DIM a(2) ' array
a(1) = 4: PRINT a(1) ' show
GOSUB Done ' call
END ' stop
Done:
PRINT "sub" ' inside
RETURN ' back
''')
        self.assertEqual(errors, [])
        self.assertEqual(out.split(), ['six', 'case', 'six', '4', 'sub'])

    def test_comments_in_quantum_programs(self):
        r = api.run("10 X 0 ' flip\n20 MEASURE ' all", qubits=1, shots=50, seed=1)
        self.assertEqual(r['errors'], [])
        self.assertEqual(r['counts'], {'1': 50})
        r = api.run("10 H 0\n20 MEAS 0 -> m\n30 IF m = 0 THEN RESET 0: GOTO 10 ' again\n40 MEASURE",
                    qubits=1, shots=40, seed=2)
        self.assertEqual(r['counts'], {'1': 40})

    def test_data_ends_at_a_colon(self):
        out, errors = run_basic("READ a$, b$\nPRINT a$; \"|\"; b$\nDATA don't, stop : ' note")
        self.assertEqual((out.strip(), errors), ("don't|stop", []))
        self.assertEqual(parse_stmt('x = 1:').__class__.__name__, 'LetStmt')


class TestConstOverBuiltins(unittest.TestCase):

    def test_const_may_name_a_builtin_constant(self):
        out, errors = run_basic('CONST TRUE = -1, FALSE = NOT TRUE\nCONST PI = 3.14\nPRINT TRUE; FALSE; PI')
        self.assertEqual((out.split(), errors), (['-1', '0', '3.14'], []))

    def test_assigning_still_fails(self):
        _, errors = run_basic('PI = 3')
        self.assertTrue(errors and 'RESERVED' in errors[0])
        _, errors = run_basic('CONST PI = 3.14\nPI = 4')
        self.assertTrue(errors and 'DUPLICATE DEFINITION' in errors[0])


class TestLen(unittest.TestCase):

    def test_storage_sizes(self):
        out, errors = run_basic('''
TYPE Entry
  label AS STRING * 6
  value AS DOUBLE
END TYPE
DIM rec AS Entry, tab(1 TO 2) AS Entry, f AS STRING * 10
x% = 5: y& = 7: z! = 1.5: w# = 2.5: s$ = "abc": u = 3
PRINT LEN(rec); LEN(rec.value); LEN(rec.label); LEN(tab(1)); LEN(tab(2).value)
PRINT LEN(x%); LEN(y&); LEN(z!); LEN(w#); LEN(s$); LEN(u); LEN(f); LEN("hello"); LEN(s$ + "de")
''')
        self.assertEqual(errors, [])
        self.assertEqual(out.split(), ['14', '8', '6', '14', '8', '2', '4', '4', '8', '3', '4', '10', '5', '5'])


class TestCursorColumn(unittest.TestCase):

    def test_pos_tab_and_zones_follow_earlier_output(self):
        out, errors = run_basic('''
PRINT "ab";
PRINT POS(0)
PRINT "abc";
PRINT TAB(10); "x"
PRINT "abcdef";
PRINT , "z"
WRITE 1, "a"
PRINT POS(0)
''')
        self.assertEqual(errors, [])
        lines = out.splitlines()
        self.assertEqual(lines[0], 'ab 3 ')
        self.assertEqual(lines[1].index('x'), 9)
        self.assertEqual(lines[2].index('z'), 14)
        self.assertEqual(lines[4], ' 1 ')

    def test_input_returns_to_column_one(self):
        r = api.run('PRINT "a";\nINPUT "n"; n\nPRINT POS(0)', basic=True, inputs=['5'])
        self.assertIn(' 1 ', r['output'])

    def test_print_to_a_file_keeps_its_own_column(self):
        import os
        import tempfile
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                _, errors = run_basic('OPEN "cols.txt" FOR OUTPUT AS #1\nPRINT #1, "abc";\n'
                                'PRINT #1, TAB(6); "x"\nCLOSE #1')
                self.assertEqual(errors, [])
                with open('cols.txt', encoding='latin-1') as f:
                    self.assertEqual(f.read(), 'abc  x\n')
            finally:
                os.chdir(cwd)


# ---------------------------------------------------------------------------
# Keyword statements: state injection, INPUT parsing, the screen statements
# on a non-terminal, SAVE_* inside live runs, and the accepted-for-compatibility
# statements.
# ---------------------------------------------------------------------------


class TestStateInjection(unittest.TestCase):

    def test_named_states(self):
        s = 1 / math.sqrt(2)
        cases = {'|0>': [1, 0, 0, 0, 0, 0, 0, 0], '|1>': [0, 1, 0, 0, 0, 0, 0, 0],
                 '|+>': [s, s, 0, 0, 0, 0, 0, 0], '|->': [s, -s, 0, 0, 0, 0, 0, 0],
                 '|BELL>': [s, 0, 0, 0, 0, 0, 0, s], '|GHZ3>': [s, 0, 0, 0, 0, 0, 0, s]}
        for name, want in cases.items():
            np.testing.assert_allclose(_resolve_named_state(name, 3), want, atol=1e-12)
        w = _resolve_named_state('|W>', 3)
        np.testing.assert_allclose(np.abs(w[[1, 2, 4]]) ** 2, [1 / 3] * 3)
        np.testing.assert_allclose(_resolve_named_state('|W3>', 1), [s, s])
        np.testing.assert_allclose(_resolve_named_state('|GHZ4>', 2), [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            _resolve_named_state('|+>', 0)

    def test_set_state_in_a_program_and_at_the_prompt(self):
        t = term()
        feed(t, '10 SET_STATE |BELL>', '20 SAVE_EXPECT ZZ 0 1 -> zz', '30 SAVE_EXPECT Z 0 -> z', 'RUN')
        self.assertAlmostEqual(t.variables['zz'], 1.0)
        self.assertAlmostEqual(t.variables['z'], 0.0)
        out = feed(t, '10 SET_STATE [1, 0, 0, 1]', 'RUN')
        self.assertIn('normalized', out)
        out = feed(t, '10 SET_STATE [1, 0]', 'RUN', '10 SET_STATE [0, 0, 0, 0]', 'RUN')
        self.assertIn('length 2 != 2^2 = 4', out)
        self.assertIn('state vector is zero', out)
        out = feed(t, '10 SET_STATE |GHZ4>', 'RUN')
        self.assertIn('needs more qubits', out)
        t2 = term(1)
        out = feed(t2, 'SET_STATE', 'SET_STATE |->', '10 SAVE_EXPECT X 0 -> x', 'RUN')
        self.assertIn('?USAGE: SET_STATE', out)
        self.assertAlmostEqual(t2.variables['x'], -1.0)
        self.assertIn('?SET_STATE ERROR', feed(t2, 'SET_STATE [1, 2, 3]'))

    def test_set_density(self):
        t = term(1)
        out = feed(t, 'SET_DENSITY [[0.5, 0], [0, 0.5]]', '10 MEASURE', 'RUN')
        self.assertIn('purity 0.5000', out)
        self.assertEqual(t.sim_method, 'density_matrix')
        p1 = t.last_counts.get('1', 0) / sum(t.last_counts.values())
        self.assertAlmostEqual(p1, 0.5, delta=0.08)
        for bad, msg in (('', '?USAGE: SET_DENSITY'), ('[[1, 0], [0, 0], [0, 0]]', 'need (2, 2)'),
                         ('[[1, 1], [0, 0]]', 'not Hermitian'), ('[[0, 0], [0, 0]]', 'zero trace'),
                         ('[[2, 0], [0, -1]]', 'positive semidefinite')):
            self.assertIn(msg, feed(t, f'SET_DENSITY {bad}'))
        self.assertIn('normalized: trace 2.0000', feed(t, 'SET_DENSITY [[1, 0], [0, 1]]'))


class TestInputParsing(unittest.TestCase):

    def test_fields(self):
        self.assertEqual(_input_fields(' 1 , "a, b" ,c'), ['1', 'a, b', 'c'])
        self.assertEqual(_input_fields('"open'), ['open'])
        self.assertEqual(_parse_input_number('&HFF'), 255)
        self.assertEqual(_parse_input_number('&O17'), 15)
        self.assertEqual(_parse_input_number('&B101'), 5)
        self.assertIsNone(_parse_input_number('&HZZ'))
        self.assertEqual(_parse_input_number('1.5D2'), 150.0)
        self.assertIsNone(_parse_input_number('abc'))

    def test_input_retries_and_defaults(self):
        r = api.run('INPUT "n"; n\nPRINT n * 2', basic=True, inputs=['x', '21'])
        self.assertIn('?Redo from start', r['output'])
        self.assertIn('42', r['output'])
        r = api.run('INPUT a, b$\nPRINT a; b$', basic=True, inputs=['5, "hi, there"'])
        self.assertIn('5 hi, there', r['output'])
        r = api.run('INPUT n\nPRINT n + 1', basic=True, inputs=[])
        self.assertIn(' 1 ', r['output'])
        r = api.run('LINE INPUT "line: "; s$\nPRINT "[" + s$ + "]"', basic=True, inputs=['a, "b"'])
        self.assertIn('[a, "b"]', r['output'])
        r = api.run('GET k$\nPRINT ASC(k$)\nGET k\nPRINT k', basic=True, inputs=['A', 'B'])
        words = r['output'].split()                  # the answers are echoed: A 65 B 66
        self.assertEqual(words[-4:], ['A', '65', 'B', '66'])


class TestScreenStatements(unittest.TestCase):

    def test_off_a_terminal(self):
        r = api.run('CLS\nCOLOR 14, 1\nCOLOR yellow\nLOCATE 2, 5\nBEEP\nSLEEP 0\nWIDTH 80\nPRINT "ok"',
                    basic=True)
        self.assertEqual(r['errors'], [])
        self.assertIn('ok', r['output'])
        r = api.run('COLOR 40', basic=True)
        self.assertIn('ILLEGAL FUNCTION CALL: COLOR 40', r['errors'][0])

    def test_screen_statement(self):
        t = term()
        feed(t, '10 SCREEN 4', 'RUN')
        self.assertEqual(t._screen_mode, 4)
        feed(t, '10 SCREEN 12', '20 COLOR 4, 1', '30 PSET (5, 5)', 'RUN')
        self.assertEqual((t._canvas.mode, t._canvas.fg, t._canvas.bg), (12, 4, 1))
        self.assertIn('ILLEGAL FUNCTION CALL: COLOR 99', feed(t, '10 SCREEN 12', '20 COLOR 99', 'RUN'))
        self.assertIn('ILLEGAL FUNCTION CALL: SCREEN 6', feed(t, '10 SCREEN 6', 'RUN'))
        feed(t, '10 SCREEN , , 0, 0', 'RUN')

    def test_on_device_traps_are_not_computed_jumps(self):
        r = api.run('players = 2\n5 ON players GOSUB 10, 20\n6 END\n10 PRINT "one": RETURN\n'
                    '20 PRINT "two": RETURN', basic=False)
        self.assertTrue(r['output'].strip().endswith('two'))
        self.assertNotIn('one', r['output'])

    def test_accepted_statements(self):
        r = api.run('DECLARE SUB X ()\nKEY OFF\nKEY 1, "LIST"\nKEY(1) ON\nON KEY(1) GOSUB 9\n'
                    'PEN ON\nSTRIG(0) OFF\nCOM(1) STOP\nCOMMON SHARED total\nPRINT "fine"', basic=True)
        self.assertEqual((r['errors'], r['output'].strip()), ([], 'fine'))


class TestSavesInLiveRuns(unittest.TestCase):

    def test_saves_average_over_shots(self):
        t = term(1)
        t.shots = 400
        feed(t, '10 H 0', '20 SAVE_EXPECT Z 0 -> z', '30 SAVE_PROBS 0 -> p', '40 SAVE_AMPS 0, 1 -> a',
             '50 MEAS 0 -> m', '60 PRINT m;', 'RUN')
        self.assertAlmostEqual(t.variables['z'], 0.0, places=6)
        np.testing.assert_allclose(t.arrays['p'], [0.5, 0.5], atol=1e-6)
        self.assertAlmostEqual(abs(complex(t.arrays['a'][1])), 1 / math.sqrt(2), places=6)

    def test_save_errors_and_hamiltonians(self):
        t = term()
        feed(t, 'HAMILTONIAN H = 1.0 ZZ 0 1', '10 X 0', '20 SAVE_EXPECT H -> e', 'RUN')
        self.assertAlmostEqual(t.variables['e'], -1.0)
        feed(t, 'QUBITS 3')
        out = feed(t, '10 SAVE_EXPECT H -> e', 'RUN')
        self.assertIn('is 2-qubit but QUBITS is 3', out)

    def test_apply_circuit_and_poke_inside_programs(self):
        t = term(3)
        feed(t, '10 X 0', '20 CX 0, 1', 'CIRCUIT_DEF PAIR 10-20', 'NEW', 'QUBITS 3',
             '10 APPLY_CIRCUIT PAIR @1', '20 POKE 9, 4', '30 MEASURE')
        self.assertIn('UNDEFINED CIRCUIT', feed(t, '5 APPLY_CIRCUIT NOPE', 'RUN'))
        feed(t, '5')
        feed(t, 'CIRCUIT_DEF PAIR 10-10')


# ---------------------------------------------------------------------------
# QBASIC I/O beyond PRINT: RANDOM and BINARY files, FIELD/LSET/RSET, the
# MKx$/CVx conversions, expression file numbers, the file system statements,
# PLAY/SOUND/BEEP, TIMER events, COMMON and the device-event statements.
# ---------------------------------------------------------------------------


class TestRandomFiles(InTempDir):

    def test_field_lset_rset_put_get(self):
        _, out, _ = run('''
OPEN "people.dat" FOR RANDOM AS #1 LEN = 20
FIELD #1, 12 AS nm$, 8 AS age$
LSET nm$ = "Ada"
RSET age$ = MKI$(36)
PUT #1, 1
LSET nm$ = "Grace"
RSET age$ = MKI$(85)
PUT #1
PRINT LOF(1); LOC(1); SEEK(1)
GET #1, 1
PRINT "["; nm$; "]"; CVI(RIGHT$(age$, 2))
GET #1, 2
PRINT RTRIM$(nm$); CVI(RIGHT$(age$, 2)); EOF(1)
GET #1, 3
PRINT EOF(1)
CLOSE #1
''')
        self.assertEqual(out.split('\n')[:4],
                         [' 40  2  3 ', '[Ada         ] 36 ', 'Grace 85  0 ', '-1 '])

    def test_records_through_put_and_get(self):
        _, out, _ = run('''
TYPE Pt
  x AS INTEGER
  y AS DOUBLE
  tag AS STRING * 4
END TYPE
DIM p AS Pt, q AS Pt
p.x = 7: p.y = 2.5: p.tag = "ab"
OPEN "pts.dat" FOR RANDOM AS #2 LEN = LEN(p)
PUT #2, 3, p
GET #2, 3, q
PRINT q.x; q.y; "["; q.tag; "]"; LOF(2)
CLOSE
''')
        self.assertEqual(out.split('\n')[0], ' 7  2.5 [ab  ] 42 ')     # three 14-byte records

    def test_errors(self):
        _, out, _ = run('ON ERROR GOTO 90\nOPEN "r.dat" FOR RANDOM AS #1 LEN = 4\n'
                     'FIELD #1, 3 AS a$, 3 AS b$\nEND\n90 PRINT ERR')
        self.assertEqual(out.split(), ['50'])               # FIELD overflow
        _, out, _ = run('ON ERROR GOTO 90\nOPEN "r.dat" FOR RANDOM AS #1\nGET #1, 0, x%\nEND\n90 PRINT ERR')
        self.assertEqual(out.split(), ['63'])               # bad record number
        _, out, _ = run('ON ERROR GOTO 90\nOPEN "t.txt" FOR OUTPUT AS #1\nGET #1, 1, x%\nEND\n90 PRINT ERR')
        self.assertEqual(out.split(), ['54'])               # bad file mode


class TestBinaryFiles(InTempDir):

    def test_values_round_trip(self):
        _, out, _ = run('''
OPEN "raw.bin" FOR BINARY AS #3
n% = 513: l& = 70000: s! = 1.5: d# = 3.25: t$ = "hey"
PUT #3, , n%
PUT #3, , l&
PUT #3, , s!
PUT #3, , d#
PUT #3, , t$
PRINT LOF(3); LOC(3); SEEK(3)
SEEK #3, 1
a% = 0: b& = 0: c! = 0: e# = 0: u$ = SPACE$(3)
GET #3, , a%
GET #3, , b&
GET #3, , c!
GET #3, , e#
GET #3, , u$
PRINT a%; b&; c!; e#; u$; EOF(3)
GET #3, 2, w%
PRINT w%
CLOSE #3
''')
        self.assertEqual(out.split('\n')[:3], [' 21  21  22 ', ' 513  70000  1.5  3.25 hey-1 ', ' 28674 '])
        with open('raw.bin', 'rb') as fh:
            self.assertEqual(fh.read(2), b'\x01\x02')

    def test_conversion_functions(self):
        t = QBasicTerminal()
        for expr, want in (('CVI(MKI$(-2))', -2), ('CVL(MKL$(100000))', 100000),
                           ('CVS(MKS$(0.5))', 0.5), ('CVD(MKD$(1/8))', 0.125),
                           ('LEN(MKD$(1))', 8)):
            with self.subTest(expr=expr):
                self.assertEqual(t._safe_eval(expr), want)
        with self.assertRaises(ValueError):
            t._safe_eval('MKI$(70000)')
        with self.assertRaises(ValueError):
            t._safe_eval('CVI("a")')


class TestSequentialFiles(InTempDir):

    def test_expression_file_numbers_and_input_chars(self):
        _, out, _ = run('''
f$ = "notes.txt"
fn = FREEFILE
OPEN f$ FOR OUTPUT AS #fn
PRINT #fn, "alpha"; 1
WRITE #fn, "beta", 2
CLOSE #fn
OPEN "A", #fn, f$
PRINT #fn, "gamma"
CLOSE fn
OPEN f$ FOR INPUT AS fn
LINE INPUT #fn, a$
INPUT #fn, b$, k
c$ = INPUT$(5, #fn)
PRINT a$; "|"; b$; k; "|"; c$; EOF(fn); LOC(fn)
CLOSE
''')
        self.assertEqual(out.split('\n')[0], 'alpha 1 |beta 2 |gamma 0  0 ')

    def test_keyboard_input_chars(self):
        _, out, _ = run('k$ = INPUT$(3)\nPRINT "["; k$; "]"', answers=['abcdef'])
        self.assertEqual(out.split('\n')[0], '[abc]')


class TestFileSystem(InTempDir):

    def test_mkdir_name_chdir_kill_rmdir_files(self):
        t, out, _ = run('''
MKDIR "sub"
OPEN "sub/a.txt" FOR OUTPUT AS #1
PRINT #1, "x"
CLOSE
NAME "sub/a.txt" AS "sub/b.txt"
CHDIR "sub"
FILES
OPEN "../top.txt" FOR OUTPUT AS #1
CLOSE
KILL "*.txt"
CHDIR ".."
RMDIR "sub"
FILES "*.txt"
''')
        self.assertEqual(out.split('\n')[:2], ['b.txt', 'top.txt'])
        self.assertEqual(sorted(os.listdir('.')), ['top.txt'])

    def test_errors(self):
        for stmt, code in (('KILL "nothing.here"', 53), ('CHDIR ".."', 70), ('RMDIR "nodir"', 76),
                           ('NAME "x" AS "y"', 53), ('OPEN "missing.txt" FOR INPUT AS #1', 53)):
            with self.subTest(stmt=stmt):
                _, out, _ = run(f'ON ERROR GOTO 90\n{stmt}\nEND\n90 PRINT ERR')
                self.assertEqual(out.split(), [str(code)])
        open('dup.txt', 'w').close()
        _, out, _ = run('ON ERROR GOTO 90\nMKDIR "d"\nNAME "dup.txt" AS "dup.txt"\nEND\n90 PRINT ERR')
        self.assertEqual(out.split(), ['58'])


class TestSound(unittest.TestCase):

    def test_play_language(self):
        state = PlayState()
        events = parse_play('T120 L4 O3 C E G > C', state)
        notes = [round(f, 1) for f, _ in events if f]
        self.assertEqual(notes, [261.6, 329.6, 392.0, 523.3])
        self.assertAlmostEqual(sum(d for _, d in events), 2.0)
        self.assertEqual(state.octave, 4)
        self.assertAlmostEqual(sum(d for _, d in parse_play('ML L8 C.', PlayState())), 0.375 * 0.5 * 2)
        self.assertAlmostEqual(note_frequency(46), 440.0)        # O3 A
        self.assertEqual(parse_play('N0', PlayState())[0][0], 0.0)
        self.assertAlmostEqual(parse_play('P4', PlayState())[0][1], 0.5)
        for bad in ('O9', 'T10', 'L0', 'N99', 'X'):
            with self.assertRaises(ValueError):
                parse_play(bad, PlayState())

    def test_program_sound_and_savewav(self):
        t, out, _ = run('PLAY "T180 O4 L8 CDEFGAB>C"\nSOUND 440, 18.2\nBEEP\nPRINT "done"')
        self.assertEqual(out, '\adone\n')
        self.assertAlmostEqual(t._sound_seconds(), 8 * (60 / 180 / 2) + 1.0 + 0.25, places=2)
        frames = len(synthesize(t._sound_events))
        with tempfile.TemporaryDirectory(dir='.') as d:
            path = os.path.relpath(os.path.join(d, 'tune.wav'))
            t.dispatch(f'SAVEWAV {path} CLEAR')
            self.assertIn('SAVED', t.io.text)
            with wave.open(path) as w:
                self.assertEqual((w.getnchannels(), w.getframerate()), (1, 22050))
                self.assertEqual(w.getnframes(), frames)
        self.assertEqual(t._sound_events, [])

    def test_synthesis(self):
        samples = synthesize([(440.0, 0.1), (0.0, 0.05)])
        self.assertEqual(len(samples), int(0.15 * 22050))
        self.assertTrue(wav_bytes(samples).startswith(b'RIFF'))

    def test_errors(self):
        _, out, _ = run('ON ERROR GOTO 90\nSOUND 10, 5\nEND\n90 PRINT ERR')
        self.assertEqual(out.split(), ['5'])

    def test_music_macro_language(self):
        state = PlayState()
        events = parse_play('T120 O3 L4 C D# E- F+ N0 N37 P8 R4 MN C ML D MS E MF MB > C < C C8. ;', state)
        tones = [f for f, _ in events if f > 0]
        self.assertAlmostEqual(tones[0], note_frequency(37))      # C of octave 3
        self.assertEqual(state.octave, 3)
        self.assertGreater(len(events), 15)
        for bad in ('C99', 'N99', 'O9', 'L0', 'T10', 'P0', 'MX', 'Q'):
            with self.assertRaises(ValueError):
                parse_play(bad, PlayState())
        samples = synthesize([(440.0, 0.01), (0.0, 0.01), (220.0, 0.0)])
        self.assertEqual(samples.dtype, np.int16)
        self.assertEqual(len(synthesize([])), 0)
        self.assertTrue(wav_bytes(samples).startswith(b'RIFF'))
        self.assertAlmostEqual(note_frequency(46), 440.0)              # A above middle C (note 37)

    def test_statements_and_savewav(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                t = term(1)
                t.agent_mode = True
                out = feed(t, 'SAVEWAV', 'SAVEWAV quiet', 'SAVEWAV ../x')
                self.assertIn('?USAGE: SAVEWAV', out)
                self.assertIn('?NO SOUND', out)
                self.assertIn('?SAVEWAV ERROR', out)
                feed(t, '10 PLAY "L8 CDE"', '20 SOUND 440, 18.2', '30 SOUND 32767, 9.1', '40 BEEP',
                     '50 PLAY ON', 'RUN')
                out = feed(t, 'SAVEWAV tune CLEAR')
                self.assertRegex(out, r'SAVED 2\.\d\d s of sound')
                self.assertTrue(os.path.getsize('tune.wav') > 1000)
                self.assertIn('?NO SOUND', feed(t, 'SAVEWAV again'))
                for stmt, msg in (('SOUND 10, 1', 'ILLEGAL FUNCTION CALL'), ('SOUND 440', 'SOUND frequency'),
                                  ('PLAY 5', 'TYPE MISMATCH')):
                    self.assertIn(msg, feed(t, f'10 {stmt}', '20', '30', '40', '50', 'RUN'))
            finally:
                os.chdir(cwd)


class TestEvents(unittest.TestCase):

    def test_timer_on_off_stop(self):
        from unittest.mock import patch
        t = QBasicTerminal()
        t._on_timer_target, t._on_timer_interval, t._on_timer_last = 100, 1.0, 0.0
        t._gosub_stack = []
        lines = [100, 110]
        with patch('qubasic_core.interpreter.time') as clock:
            clock.time.return_value = 5.0
            self.assertIsNone(t._check_timer_callback(lines, 0))      # TIMER OFF
            t._stmt_timer('TIMER STOP')
            self.assertIsNone(t._check_timer_callback(lines, 0))      # held
            t._stmt_timer('TIMER ON')
            clock.time.return_value = 5.1
            self.assertEqual(t._check_timer_callback(lines, 0), 0)     # the held event fires
        self.assertFalse(t._stmt_timer('TIMER NOW'))

    def test_on_timer_with_a_label_and_statements_accepted(self):
        _, out, _ = run('''
ON TIMER(60) GOSUB Tick
TIMER ON
KEY OFF
KEY(1) ON
ON KEY(1) GOSUB Tick
PEN OFF
STRIG(0) ON
TIMER OFF
PRINT "ran"
END
Tick: RETURN
''')
        self.assertEqual(out.strip(), 'ran')

    def test_common_shared(self):
        _, out, _ = run('COMMON SHARED total\ntotal = 2\nAddOne\nPRINT total\nSUB AddOne\n total = total + 1\nEND SUB')
        self.assertEqual(out.split(), ['3'])


# ---------------------------------------------------------------------------
# File statements: path rules, both OPEN syntaxes, the QBASIC error codes
# ON ERROR sees, sequential and binary reads, record files through FIELD and
# typed variables, position functions, LPRINT and the file-system statements.
# ---------------------------------------------------------------------------


def err_of(statements):
    """The ERR code QBASIC reports for the failing statement(s)."""
    out, errors = run_basic('ON ERROR GOTO Trap\n' + statements + '\nEND\nTrap:\nPRINT "ERR"; ERR\nEND')
    assert not errors, errors
    return int(out.split('ERR')[-1].split()[0])


class TestPathsAndOpen(InTempDir):

    def test_error_codes(self):
        cases = {
            'OPEN "" FOR OUTPUT AS #1': 64,
            'OPEN "/etc/x" FOR OUTPUT AS #1': 70,
            'OPEN "C:/x" FOR OUTPUT AS #1': 70,
            'OPEN "../x" FOR OUTPUT AS #1': 70,
            'OPEN "nope.txt" FOR INPUT AS #1': 53,
            'OPEN "a.txt" FOR OUTPUT AS #0': 52,
            'OPEN "a.txt" FOR OUTPUT AS #1: OPEN "b.txt" FOR OUTPUT AS #1': 55,
            'OPEN "r.dat" FOR RANDOM AS #1 LEN = 0': 59,
            'PRINT #3, "x"': 52,
            'OPEN "a.txt" FOR OUTPUT AS #1: CLOSE #1: OPEN "a.txt" FOR INPUT AS #1: PRINT #1, "x"': 54,
            'OPEN "r.dat" FOR RANDOM AS #1: PRINT #1, "x"': 54,
            'OPEN "a.txt" FOR OUTPUT AS #1: INPUT #1, v': 54,
            'OPEN "a.txt" FOR OUTPUT AS #1: CLOSE: OPEN "a.txt" FOR INPUT AS #1: INPUT #1, v': 62,
            'OPEN "a.txt" FOR OUTPUT AS #1: GET #1, 1, v%': 54,
            'KILL "missing.txt"': 53,
            'OPEN "k.txt" FOR OUTPUT AS #1: KILL "k.txt"': 55,
            'MKDIR "d": MKDIR "d"': 75,
            'CHDIR "nowhere"': 76,
            'RMDIR "nowhere"': 76,
            'NAME "x1.txt" AS "x2.txt"': 53,
            'FILES "*.none"': 53,
        }
        for statements, code in cases.items():
            with self.subTest(statements=statements):
                self.assertEqual(err_of(statements), code)

    def test_old_open_syntax_and_encodings(self):
        out, errors = run_basic('OPEN "O", #1, "old.txt"\nPRINT #1, "hello"\nCLOSE #1\n'
                          'OPEN "I", #1, "old.txt"\nLINE INPUT #1, s$\nCLOSE\nPRINT s$\n'
                          'OPEN "R", #2, "rec.dat", 16\nPRINT LOF(2)\nCLOSE #2\n'
                          'OPEN "bytes.bin" FOR OUTPUT AS #3 ENCODING binary\nPRINT #3, "ab"\nCLOSE #3')
        self.assertEqual(errors, [])
        self.assertEqual(out.split(), ['hello', '0'])
        self.assertEqual(open('bytes.bin', 'rb').read(), b'ab\n')
        _, errors = run_basic('OPEN "Z", #1, "x"')
        self.assertIn('SYNTAX ERROR: OPEN', errors[0])

    def test_prompt_open_and_close(self):
        t = QBasicTerminal()
        t.io = CaptureIO()
        t.agent_mode = True
        for line in ('OPEN "p.txt" FOR OUTPUT AS #4', 'CLOSE #4', 'OPEN "p.txt" FOR OUTPUT AS #5',
                     'OPEN "q.txt" FOR OUTPUT AS #6', 'CLOSE', 'OPEN "missing" FOR INPUT AS #1', 'CLOSE #999'):
            t.process(line)
        for text in ('OPENED #4', 'CLOSED #4', 'CLOSED ALL', '?OPEN ERROR: FILE NOT FOUND', '?CLOSE ERROR'):
            self.assertIn(text, t.io.text)


class TestReading(InTempDir):

    def test_sequential_reads_and_positions(self):
        with open('data.txt', 'w', newline='') as f:
            f.write('3, "a, b", 4.5\nwhole line here\nxyz')
        out, errors = run_basic('OPEN "data.txt" FOR INPUT AS #1\nINPUT #1, n, s$, x\nLINE INPUT #1, w$\n'
                          'PRINT n; s$; x; "|"; w$; "|"; INPUT$(2, #1); EOF(1)\nPRINT INPUT$(1, #1); EOF(1)\n'
                          'PRINT LOC(1); SEEK(1); LOF(1)\nSEEK #1, 1\nINPUT #1, again\nPRINT again; FREEFILE\n'
                          'CLOSE\nPRINT EOF(1)')
        self.assertEqual(errors, [])
        lines = out.splitlines()
        self.assertEqual(lines[0], ' 3 a, b 4.5 |whole line here|xy 0 ')
        self.assertEqual(lines[1], 'z-1 ')
        self.assertEqual(lines[2].split(), ['0', '35', '34'])          # 34 bytes, all read
        self.assertEqual(lines[3].split(), ['3', '2'])
        self.assertEqual(lines[4].split(), ['-1'])
        self.assertEqual(err_of('OPEN "data.txt" FOR INPUT AS #1: x$ = INPUT$(99, #1)'), 62)

    def test_binary_files(self):
        out, errors = run_basic('OPEN "b.bin" FOR BINARY AS #1\nn% = 513: d# = 2.5: s$ = "hey"\n'
                          'PUT #1, 1, n%\nPUT #1, , d#\nPUT #1, , s$\nPRINT LOC(1); SEEK(1); LOF(1)\n'
                          'GET #1, 1, m%\nGET #1, 3, e#\nt$ = SPACE$(3): GET #1, 11, t$\n'
                          'PRINT m%; e#; t$; EOF(1)\nSEEK #1, 2\nPRINT SEEK(1)\nCLOSE')
        self.assertEqual(errors, [])
        lines = out.splitlines()
        self.assertEqual(lines[0].split(), ['13', '14', '13'])
        self.assertEqual(lines[1].split(), ['513', '2.5', 'hey-1'])
        self.assertEqual(lines[2].split(), ['2'])
        self.assertEqual(err_of('OPEN "b.bin" FOR BINARY AS #1: n% = 1E9: PUT #1, 1, n%'), 6)

    def test_field_buffers(self):
        out, errors = run_basic('OPEN "f.dat" FOR RANDOM AS #1 LEN = 10\nFIELD #1, 4 AS a$, 6 AS b$\n'
                          'LSET a$ = "ab": RSET b$ = "xyz"\nPUT #1, 2\nLSET a$ = "": RSET b$ = ""\n'
                          'GET #1, 2\nPRINT "["; a$; "]["; b$; "]"; LOC(1); SEEK(1)\nGET #1, 5\nPRINT EOF(1)\nCLOSE')
        self.assertEqual(errors, [])
        lines = out.splitlines()
        self.assertEqual(lines[0], '[ab  ][   xyz] 2  3 ')
        self.assertEqual(lines[1].strip(), '-1')
        self.assertEqual(err_of('OPEN "f.dat" FOR RANDOM AS #1 LEN = 4: FIELD #1, 3 AS a$, 3 AS b$'), 50)
        self.assertEqual(err_of('OPEN "f.dat" FOR RANDOM AS #1: GET #1, 0'), 63)
        _, errors = run_basic('OPEN "f.dat" FOR RANDOM AS #1\nPUT #1, 1')
        self.assertIn('needs a variable', errors[0])
        _, errors = run_basic('x = 5\nLSET x = 3')
        self.assertIn('TYPE MISMATCH', errors[0])

    def test_input_chars_from_the_keyboard(self):
        out, _ = run_basic('k$ = INPUT$(3)\nPRINT "[" + k$ + "]"', inputs=['ab', 'cd'])
        self.assertIn('[abc]', out)


class TestPrintersAndFileSystem(InTempDir):

    def test_lprint(self):
        t = QBasicTerminal()
        t.io = CaptureIO()
        t.process('10 LPRINT "to the printer"')
        t.process('20 LPRINT 6 * 7')
        t.process('RUN')
        self.assertIn('to the printer', t.io.text)
        self.assertIn('42', t.io.text)
        t._lprint_path = 'printer.log'
        t.process('RUN')
        self.assertEqual(open('printer.log').read().split('\n')[:2], ['to the printer', '42.0'])

    def test_file_system(self):
        out, errors = run_basic('MKDIR "sub"\nCHDIR "sub"\nOPEN "in.txt" FOR OUTPUT AS #1\nCLOSE\n'
                          'NAME "in.txt" AS "out.txt"\nFILES\nCHDIR ".."\nFILES "sub/*.txt"\n'
                          'OPEN "sub/a1.tmp" FOR OUTPUT AS #1: CLOSE\nOPEN "sub/a2.tmp" FOR OUTPUT AS #1: CLOSE\n'
                          'KILL "sub/*.tmp"\nKILL "sub/out.txt"\nRMDIR "sub"\nPRINT "done"')
        self.assertEqual(errors, [])
        self.assertEqual(out.split(), ['out.txt', 'out.txt', 'done'])
        self.assertFalse(os.path.exists('sub'))
        self.assertEqual(err_of('OPEN "x.txt" FOR OUTPUT AS #1: CLOSE: OPEN "y.txt" FOR OUTPUT AS #1: CLOSE: '
                                'NAME "x.txt" AS "y.txt"'), 58)
        _, errors = run_basic('NAME "x.txt"')
        self.assertIn('NAME old AS new', errors[0])
        _, errors = run_basic('MKDIR')
        self.assertIn('SYNTAX ERROR: MKDIR path', errors[0])
        t = QBasicTerminal()
        t.io = CaptureIO()
        t.process('FILES "*.none"')
        self.assertIn('?FILE NOT FOUND', t.io.text)


# ---------------------------------------------------------------------------
# QBASIC graphics: SCREEN modes, the drawing statements, POINT and PMAP,
# DRAW, VIEW and WINDOW, GET and PUT, rendering and PNG output.
# ---------------------------------------------------------------------------


class TestCanvas(unittest.TestCase):

    def test_modes_and_palettes(self):
        for mode, (w, h, colors) in MODES.items():
            c = Canvas(mode)
            self.assertEqual(c.pixels.shape, (h, w))
            self.assertGreaterEqual(len(c.palette), colors)
        self.assertEqual(Canvas(13).palette[15], (255, 255, 255))
        with self.assertRaises(ValueError):
            Canvas(5)

    def test_pset_point_and_line(self):
        c = Canvas(13)
        c.pset(3, 4, 9)
        self.assertEqual(c.point(3, 4), 9)
        self.assertEqual(c.point(400, 4), -1)
        c.line(0, 0, 9, 9, 2)
        self.assertTrue(all(c.point(k, k) == 2 for k in range(10)))
        c.line(0, 20, 19, 20, 3, style=0xF0F0)
        row = [c.point(x, 20) for x in range(16)]
        self.assertEqual(row, [3] * 4 + [0] * 4 + [3] * 4 + [0] * 4)

    def test_boxes(self):
        c = Canvas(12)
        c.line(10, 10, 19, 14, 4, box='BF')
        self.assertEqual(int((c.pixels == 4).sum()), 10 * 5)
        c.line(30, 30, 39, 34, 5, box='B')
        self.assertEqual(int((c.pixels == 5).sum()), 2 * 10 + 2 * 3)

    def test_circle_aspect_and_arcs(self):
        c = Canvas(12)                       # default aspect 1 at 640x480
        c.circle(100, 100, 20, 14)
        self.assertEqual(c.point(120, 100), 14)
        self.assertEqual(c.point(100, 80), 14)
        self.assertEqual(c.point(100, 100), 0)
        d = Canvas(13)                       # 320x200: y radius 5/6 of x radius
        d.circle(160, 100, 60, 10)
        self.assertEqual(d.point(220, 100), 10)
        self.assertEqual(d.point(160, 50), 10)
        self.assertEqual(d.point(160, 40), 0)
        e = Canvas(12)
        e.circle(200, 200, 30, 11, -0.0001, -math.pi / 2)    # a sector: radii drawn
        self.assertEqual(e.point(200, 200), 11)
        self.assertEqual(e.point(215, 200), 11)
        with self.assertRaises(ValueError):
            e.circle(0, 0, 5, 1, 7.0)

    def test_paint_stops_at_the_border(self):
        c = Canvas(12)
        c.line(10, 10, 50, 40, 15, box='B')
        c.paint(20, 20, 2, 15)
        inside = (50 - 10 - 1) * (40 - 10 - 1)
        self.assertEqual(int((c.pixels == 2).sum()), inside)
        self.assertEqual(c.point(5, 5), 0)
        c.paint(10, 10, 3, 15)               # starting on the border fills nothing
        self.assertEqual(int((c.pixels == 3).sum()), 0)

    def test_draw_language(self):
        c = Canvas(12)
        c.draw('BM100,100 C4 R10 D10 L10 U10')
        self.assertEqual(int((c.pixels == 4).sum()), 40)
        self.assertEqual(c.last, (100.0, 100.0))
        c.draw('BM200,200 NR5 ND5')            # N returns to the start
        self.assertEqual(c.last, (200.0, 200.0))
        c.draw('BM300,300 S8 C6 R5')           # scale 8: two pixels per unit
        self.assertEqual(c.point(310, 300), 6)
        c.draw('BM400,300 S4 TA90 C7 R10')     # rotated a quarter turn: up
        self.assertEqual(c.point(400, 290), 7)
        c.draw('BM50,400 C9 M+20,+0 M70,420')
        self.assertEqual(c.point(70, 400), 9)
        self.assertEqual(c.point(70, 420), 9)
        c.draw('BM500,400 C1 R20 D20 L20 U20 BM+5,+5 P12,1')
        self.assertEqual(c.point(510, 410), 12)
        for bad in ('Q5', 'A7', 'S0', 'M'):
            with self.assertRaises(ValueError):
                c.draw(bad)

    def test_view_and_window(self):
        c = Canvas(12)
        c.set_view((100, 100, 199, 149), border=15)
        c.pset(0, 0, 3)                       # relative to the viewport
        self.assertEqual(int(c.pixels[100, 100]), 3)
        self.assertEqual(int(c.pixels[99, 99]), 15)
        c.line(-50, 10, 500, 10, 4)           # clipped to the viewport
        self.assertEqual(int((c.pixels == 4).sum()), 100)
        c.set_view(None)
        c.set_window((-1, -1, 1, 1))
        self.assertEqual(c.pmap(0, 0), 320)
        self.assertEqual(c.pmap(1, 1), 0)      # y grows upward
        self.assertAlmostEqual(c.pmap(c.pmap(0.5, 0), 2), 0.5, places=2)
        c.set_window((0, 0, 10, 10), screen=True)
        self.assertEqual(c.pmap(10, 1), 479)  # WINDOW SCREEN: y grows downward
        with self.assertRaises(ValueError):
            c.set_window((1, 1, 1, 2))

    def test_get_put_actions(self):
        c = Canvas(13)
        c.line(0, 0, 3, 3, 5, box='BF')
        block = c.get_block(0, 0, 3, 3)
        c.put_block(10, 10, block, 'PSET')
        self.assertEqual(c.point(12, 12), 5)
        c.put_block(10, 10, block, 'XOR')     # XOR twice restores the background
        self.assertEqual(c.point(12, 12), 0)
        c.put_block(20, 20, block, 'PRESET')
        self.assertEqual(c.point(20, 20), 250)
        with self.assertRaises(ValueError):
            c.put_block(318, 0, block)
        with self.assertRaises(ValueError):
            c.put_block(0, 0, block, 'NAND')

    def test_palette(self):
        c = Canvas(13)
        c.set_palette(1, 63)                  # 65536*b + 256*g + r, 0-63 each: red
        self.assertEqual(c.palette[1], (255, 0, 0))
        e = Canvas(9)
        e.set_palette(1, 4)                   # EGA colour 4: red
        self.assertEqual(e.palette[1], (170, 0, 0))
        e.set_palette()
        self.assertEqual(e.palette[1], (0, 0, 170))

    def test_text_rendering(self):
        c = Canvas(13)
        c.line(0, 0, 319, 0, 15)
        rows = c.text_rows(80, color=False)
        self.assertEqual(len(rows), 25)       # 200 rows, 8 to a cell of two 4x4 samples
        self.assertEqual(rows[0][0], '=')     # a white top half over a black bottom half
        self.assertEqual(rows[1], '')
        colored = c.text_rows(40, color=True)
        self.assertIn('▀', colored[0])
        self.assertIn('\x1b[38;2;255;255;255m', colored[0])

    def test_png_round_trip(self):
        c = Canvas(13)
        c.circle(100, 100, 40, 12)
        rgb = c.rgb()
        data = png_bytes(rgb)
        self.assertTrue(data.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertTrue(np.array_equal(parse_png(data), rgb))


class TestGraphicsStatements(unittest.TestCase):

    def test_program_draws_and_reads_back(self):
        t, out, _ = run('''
SCREEN 13
LINE (10, 10)-(40, 30), 4, BF
PSET STEP(0, 0), 15
CIRCLE (160, 100), 30, 10
PAINT (160, 100), 1, 10
d$ = "BM200,150 C12 R20"
DRAW d$
PRINT POINT(20, 20); POINT(40, 30); POINT(210, 150); POINT(0)
''')
        self.assertTrue(out.startswith(' 4  15  12  220 \n'), out[:40])   # DRAW moved the cursor
        self.assertIn('SCREEN 13 (320x200, 256 colours):', out)
        self.assertEqual(t._canvas.point(160, 100), 1)

    def test_window_pmap_and_step(self):
        t, out, _ = run('''
SCREEN 12
WINDOW (-10, -10)-(10, 10)
PSET (0, 0), 14
LINE -STEP(5, 5), 13
PRINT PMAP(0, 0); PMAP(0, 1); POINT(0, 0)
''')
        self.assertTrue(out.startswith(' 320  240  13 \n'), out[:30])     # LINE repainted (0, 0)
        c = t._canvas
        self.assertEqual(c.point(5, 5), 13)

    def test_get_put_through_arrays(self):
        t, _, _ = run('''
SCREEN 13
DIM spr(100)
LINE (0, 0)-(7, 7), 6, BF
GET (0, 0)-(7, 7), spr
PUT (100, 100), spr, PSET
PRINT spr(0); spr(1)
''')
        c = t._canvas
        self.assertEqual(c.point(104, 104), 6)
        self.assertEqual(t.arrays['spr'][:2], [8.0, 8.0])

    def test_view_window_palette_pcopy_and_cls(self):
        t, _, _ = run('''
SCREEN 7
VIEW (10, 10)-(100, 100), 1, 15
WINDOW SCREEN (0, 0)-(10, 10)
PSET (5, 5), 4
PALETTE 4, 2
PCOPY 0, 1
CLS
VIEW
WINDOW
PSET (0, 0), 3
''')
        c = t._canvas
        self.assertEqual(c.point(55, 55), 0)          # CLS cleared the viewport
        self.assertEqual(c.point(0, 0), 3)
        self.assertEqual(c.palette[4], (0, 170, 0))

    def test_graphics_need_a_graphics_screen(self):
        _, out, _ = run('ON ERROR GOTO 90\nPSET (1, 1)\nEND\n90 PRINT "trapped"; ERR')
        self.assertIn('trapped', out)
        _, out, _ = run('SCREEN 13\nSCREEN 0\nON ERROR GOTO 90\nCIRCLE (5, 5), 3\nEND\n90 PRINT "closed"')
        self.assertIn('closed', out)

    def test_line_input_and_keyboard_get_still_work(self):
        t = QBasicTerminal()
        io = ScriptIO()
        io.read_line = lambda prompt: 'hello, world'
        t.io = io
        t.process('10 LINE INPUT "> "; s$', track_undo=False)
        t.process('20 GET k$', track_undo=False)
        t.cmd_run()
        self.assertEqual(t.variables['s$'], 'hello, world')
        self.assertEqual(t.variables['k$'], 'h')

    def test_screen_display_and_show_commands(self):
        t = QBasicTerminal()
        t.io = ScriptIO()
        t.dispatch('SCREEN 13')
        t.dispatch('LINE (0, 0)-(319, 199), 14')
        t.dispatch('SHOW 40')
        self.assertIn('SCREEN 13 (320x200, 256 colours):', t.io.text)
        t.dispatch('DISPLAY BLOCH')
        self.assertEqual(t._screen_mode, 3)
        t.dispatch('COLOR 9, 1')
        self.assertEqual((t._canvas.fg, t._canvas.bg), (9, 1))
        t.dispatch('CLS')
        self.assertEqual(t._canvas.point(0, 0), 1)
        t.dispatch('DRAW "BM10,10 C4 R5"')
        self.assertEqual(t._canvas.point(15, 10), 4)
        t.dispatch('SCREEN 0')
        self.assertIsNone(t._canvas)

    def test_savepng_screen(self):
        t = QBasicTerminal()
        t.io = ScriptIO()
        t.dispatch('SCREEN 12')
        t.dispatch('CIRCLE (320, 240), 100, 12')
        with tempfile.TemporaryDirectory(dir='.') as d:
            path = os.path.relpath(os.path.join(d, 'pic.png'))
            t.dispatch(f'SAVEPNG {path}')
            self.assertIn('SAVED screen image', t.io.text)
            with open(path, 'rb') as fh:
                rgb = parse_png(fh.read())
        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertEqual(tuple(rgb[240, 420]), (255, 85, 85))


# ---------------------------------------------------------------------------
# SAVE and LOAD round trips of a whole session, QBASIC source files,
# INCLUDE, IMPORT, DIR, the library search path and agent-mode path checks.
# ---------------------------------------------------------------------------


class TestSaveLoad(InTempDir):

    def test_a_session_round_trips(self):
        t = term(3, seed=None, agent=True)
        feed(t, 'SHOTS 300', 'METHOD statevector', 'PRECISION SINGLE', 'DISPLAY STATE', 'OPTION BASE 1',
             'OPTION ENDIAN BIG', 'SEED 4', 'NOISE depolarizing 0.01; readout 0.02', 'COUPLING linear',
             'BASIS rz, sx, x, cx', 'TYPE Pt', 'x AS INTEGER', 'name AS STRING * 4', 'END TYPE',
             'DIM p AS Pt', 'LET p.x = 7', 'LET p.name = "ab"', 'DEF FN SQ(v) = v * v',
             'UNITARY SW = [[0, 1], [1, 0]]', 'DEF BELL = H 0 : CX 0, 1', 'REG data 2',
             'LET s$ = "say \\"hi\\""', 'LET n = 2.5', '10 BELL', '20 SW 2', '30 MEASURE')
        out = feed(t, 'SAVE session')
        self.assertIn('SAVED 3 lines to', out)
        text = open('session.qb').read()
        for line in ('SHOTS 300', 'METHOD statevector', 'PRECISION SINGLE', 'DISPLAY 2', 'OPTION BASE 1',
                     'OPTION ENDIAN BIG', 'SEED 4', 'NOISE depolarizing 0.01; readout 0.02', 'BASIS rz, sx, x, cx',
                     'TYPE PT', 'DIM p AS PT', 'LET p.x = 7', 'DEF FN SQ(v) = v * v', 'UNITARY SW',
                     'DEF BELL = H 0 : CX 0, 1', 'REG data 2', 'LET n = 2.5', '10 BELL'):
            self.assertIn(line, text)
        t2 = term(1, seed=None, agent=True)
        out = feed(t2, 'LOAD session')
        self.assertIn('LOADED', out)
        self.assertIn('QUBITS changed: 1 -> 3', out)
        self.assertEqual((t2.shots, t2.sim_method, t2.sim_precision, t2._screen_mode, t2._option_base),
                         (300, 'statevector', 'single', 2, 1))
        self.assertTrue(t2._endian_big)
        self.assertEqual((t2._seed, t2._noise_spec, t2._basis_gates), (4, 'depolarizing 0.01; readout 0.02',
                                                                       ['rz', 'sx', 'x', 'cx']))
        self.assertEqual((t2.variables['p']['x'], t2.variables['p']['name'], t2.variables['n']), (7, 'ab  ', 2.5))
        self.assertEqual(t2._safe_eval('FNSQ(3)'), 9)
        self.assertIn('SW', t2._custom_gates)
        self.assertEqual(t2.registers['data'], (0, 2))

    def test_logical_device_and_locc_lines(self):
        t = term(seed=None, agent=True)
        feed(t, 'LQUBITS 2 CODE SURFACE 3 PHYS 0.001', '10 H 0')
        feed(t, 'SAVE lq')
        self.assertIn('LQUBITS 2 CODE SURFACE 3 PHYS 0.001', open('lq.qb').read())
        t2 = term(5, seed=None, agent=True)
        feed(t2, 'DEVICE linear 5', '10 H 0', 'SAVE dev')
        text = open('dev.qb').read()
        self.assertIn('DEVICE linear 5', text)
        self.assertIn('COUPLING 0-1, 1-2, 2-3, 3-4', text)
        t3 = term(seed=None, agent=True)
        feed(t3, 'LOCC JOINT 2 1', '10 @A H 0', 'SAVE locc')
        self.assertIn('LOCC JOINT 2 1', open('locc.qb').read())

    def test_bas_files(self):
        t = term(seed=None, agent=True)
        feed(t, '10 PRINT "hi"', '20 END', 'SAVE prog.bas')
        self.assertEqual(open('prog.bas').read(), '10 PRINT "hi"\n20 END\n')
        with open('src.bas', 'w') as f:
            f.write('PRINT 1\n\n100 PRINT 2\nPRINT 3\n')
        t2 = term(seed=None, agent=True)
        out = feed(t2, 'LOAD src.bas')
        self.assertIn('LOADED src.bas (3 lines)', out)
        self.assertEqual(sorted(t2.program), [10, 100, 110])
        with open('bad.bas', 'w') as f:
            f.write('20 PRINT 1\n10 PRINT 2\n')
        self.assertIn('out of order', feed(t2, 'LOAD bad.bas'))

    def test_number_source(self):
        self.assertEqual(number_source(['a', 'b', '25 c', 'd']), [(8, 'a'), (16, 'b'), (25, 'c'), (35, 'd')])
        with self.assertRaises(ValueError):
            number_source(['1 a', 'b', 'c', '2 d'])

    def test_errors(self):
        t = term(seed=None, agent=True)
        os.mkdir('folder')
        out = feed(t, 'SAVE', 'LOAD', 'LOAD missing', 'LOAD folder', 'SAVE ../escape', 'LOAD ../escape')
        for msg in ('?USAGE: SAVE', '?USAGE: LOAD', '?FILE NOT FOUND', 'is a directory',
                    '?SAVE ERROR: Path traversal', '?LOAD ERROR: Path traversal'):
            self.assertIn(msg, out)


class TestIncludeImportDir(InTempDir):

    def test_include(self):
        with open('lib.qb', 'w') as f:
            f.write('# a library\nDEF FLIP = X 0\nSAVE nope\n10 FLIP\n')
        t = term(seed=None, agent=True)
        out = feed(t, 'INCLUDE lib')
        self.assertIn('BLOCKED IN INCLUDE: SAVE', out)
        self.assertIn('INCLUDED lib.qb (2 lines)', out)
        self.assertIn('FLIP', t.subroutines)
        with open('a.qb', 'w') as f:
            f.write('INCLUDE b.qb\n')
        with open('b.qb', 'w') as f:
            f.write('INCLUDE a.qb\n')
        self.assertIn('?INCLUDE CYCLE', feed(t, 'INCLUDE a.qb'))
        self.assertIn('?FILE NOT FOUND', feed(t, 'INCLUDE none'))

    def test_import_namespaces_definitions(self):
        with open('gates.qb', 'w') as f:
            f.write('DEF ROT(a) = RX a, 0 : RZ a, 0\nSUB Prep (n)\n  X n\nEND SUB\n')
        t = term(seed=None, agent=True)
        out = feed(t, 'IMPORT "gates.qb"')
        self.assertIn('IMPORTED GATES (2 definitions', out)
        self.assertIn('GATES.ROT', t.subroutines)
        feed(t, '10 CALL Prep(1)', '20 MEASURE', 'RUN', 'SAVE withimport')
        self.assertEqual(set(t.last_counts), {'10'})          # the body ran: X on qubit 1
        text = open('withimport.qb').read()
        self.assertNotIn('900000', text)
        self.assertIn('?USAGE: IMPORT', feed(t, 'IMPORT'))
        self.assertIn('?FILE NOT FOUND', feed(t, 'IMPORT "none"'))

    def test_dir_and_search_path(self):
        t = term(seed=None, agent=True)
        self.assertIn('No .qb files found', feed(t, 'DIR'))
        os.mkdir('libs')
        with open(os.path.join('libs', 'extra.qb'), 'w') as f:
            f.write('DEF ONE = X 0\n')
        self.assertIn('extra.qb', feed(t, 'DIR libs'))
        with mock.patch.dict(os.environ, {'QUBASIC_PATH': os.path.abspath('libs')}):
            out = feed(t, 'INCLUDE extra')
        self.assertIn('INCLUDED', out)
        self.assertIn('ONE', t.subroutines)


# ---------------------------------------------------------------------------
# Program editing and banks, and the debugger (breakpoints, watches, trace,
# STOP/CONT, time travel, error trapping).
# ---------------------------------------------------------------------------


class TestEditing(unittest.TestCase):

    def test_auto_edit_delete_renum(self):
        t = term(1, inputs=['H 0', '', 'X 0', '.'])
        out = feed(t, 'AUTO 100, 5')
        self.assertEqual(t.program, {100: 'H 0', 110: 'X 0'})
        self.assertIn('AUTO 100, 5', out)
        t2 = term(1, inputs=['Y 0'])
        feed(t2, '10 H 0')
        out = feed(t2, 'EDIT 10', 'EDIT 99', 'EDIT')
        self.assertEqual(t2.program[10], 'Y 0')
        self.assertIn('?LINE 99 NOT FOUND', out)
        self.assertIn('?USAGE: EDIT', out)
        t2.io = CaptureIO([''])
        feed(t2, 'EDIT 10')
        self.assertIn('(unchanged)', t2.io.text)
        t2.io = CaptureIO([])
        feed(t2, 'EDIT 10')
        self.assertIn('(cancelled)', t2.io.text)
        t3 = term(1)
        feed(t3, '10 H 0', '20 X 0', '30 Z 0', '40 GOTO 20')
        out = feed(t3, 'DELETE 30', 'DELETE 99', 'DELETE', 'DELETE 10-20', 'RENUM 100 20')
        self.assertIn('DELETED 30', out)
        self.assertIn('?LINE 99 NOT FOUND', out)
        self.assertIn('?USAGE: DELETE', out)
        self.assertEqual(t3.program, {100: 'GOTO 20'})
        self.assertIn('RENUMBERED 1 LINES', out)

    def test_copy_move_find_replace_checksum_bank(self):
        t = term(1)
        feed(t, '10 H 0', '20 X 0')
        out = feed(t, 'COPY 10-20 TO 100', 'MOVE 100-110 TO 200', 'COPY 1 TO', 'MOVE nope')
        self.assertEqual(sorted(t.program), [10, 20, 200, 210])
        self.assertIn('?USAGE: COPY', out)
        self.assertIn('?USAGE: MOVE', out)
        out = feed(t, 'FIND "x 0"', 'FIND', 'REPLACE "X 0" WITH "Y 0"', 'REPLACE x')
        self.assertIn('2 match(es)', out)
        self.assertIn('?USAGE: FIND', out)
        self.assertIn('2 replacement(s)', out)
        self.assertIn('?USAGE: REPLACE', out)
        out = feed(t, 'CHECKSUM', 'BANK', 'BANK 2', 'BANK 0', 'LIST SUBS', 'LIST VARS', 'LIST ARRAYS')
        self.assertRegex(out, r'CHECKSUM: [0-9A-F]{8} \(4 lines\)')
        self.assertIn('BANK 2 (empty)', out)
        self.assertIn('BANK 0 (4 lines)', out)
        self.assertIn('No subroutines defined', out)
        self.assertIn('No arrays', out)
        self.assertIn('?EMPTY PROGRAM', feed(term(1), 'CHECKSUM'))

    def test_definitions(self):
        t = term(inputs=['H 0', 'CX 0, 1', 'DEF END'])
        out = feed(t, 'DEF BEGIN PAIR', 'DEF FN TWICE(x) = 2 * x', 'DEF BADONE', 'DEFS')
        self.assertIn('DEF PAIR (2 gates)', out)
        self.assertIn('DEF FN TWICE(x)', out)
        self.assertIn('?USAGE: DEF', out)
        self.assertIn('PAIR = H 0 : CX 0, 1', out)
        self.assertEqual(t._safe_eval('FNTWICE(4)'), 8)
        self.assertIn('CANNOT REDEFINE BUILT-IN GATE', feed(t, 'DEF H = X 0'))
        self.assertIn('CANCELLED', feed(term(1, inputs=[]), 'DEF BEGIN NEVER'))
        out = feed(t, 'TYPE Pt', 'x AS SINGLE', 'oops', 'END TYPE', 'TYPE', 'REG data 1', 'REG big 9', 'REG x')
        self.assertIn('TYPE PT (1 fields)', out)
        self.assertIn('?EXPECTED: <name> AS <type>', out)
        self.assertIn('?USAGE: TYPE', out)
        self.assertIn('NOT ENOUGH QUBITS', out)
        self.assertIn('?USAGE: REG', out)
        out = feed(t, 'REGS', 'CIRCUIT_DEF X 1-2', 'CIRCUIT_DEF', 'APPLY_CIRCUIT NOPE', 'APPLY_CIRCUIT')
        self.assertIn('data[0:1] -> qubits 0-0', out)
        self.assertIn('?NO LINES IN RANGE 1-2', out)
        self.assertIn('?USAGE: CIRCUIT_DEF', out)
        self.assertIn('?UNDEFINED CIRCUIT: NOPE', out)
        out = feed(t, 'LET a(2) = 7', 'LET s$ = "q"', 'LET n% = 2.6', 'LET 5', 'LET n% = 1E9')
        for text in ('a(2) = 7', "s$ = 'q'", 'n% = 3', '?USAGE: LET', '?OVERFLOW'):
            self.assertIn(text, out)
        out = feed(t, 'VARS', 'CLEAR s$', 'CLEAR a', 'CLEAR nope', 'CLEAR', 'CLEAR ARRAYS')
        for text in ("s$ = q", 'CLEARED s$', '?nope NOT FOUND', '?USAGE: CLEAR', 'ARRAYS CLEARED'):
            self.assertIn(text, out)
        self.assertIn('NO VARIABLES SET', feed(term(1), 'VARS'))
        self.assertIn('NO REGISTERS DEFINED', feed(term(1), 'REGS'))
        self.assertIn('NO SUBROUTINES DEFINED', feed(term(1), 'DEFS'))

    def test_chain_and_merge(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                with open('next.qb', 'w') as f:
                    f.write('10 PRINT "chained"; carried\n')
                with open('more.qb', 'w') as f:
                    f.write('# extra lines\n20 PRINT "merged"\nDEF BEGIN FLIP\nX 0\nDEF END\n')
                t = term(1)
                out = feed(t, 'LET carried = 5', 'CHAIN "next.qb"')
                self.assertIn('chained 5', out)
                out = feed(t, '20 PRINT "old"', 'MERGE "more.qb"', 'MERGE', 'MERGE "none.qb"', 'CHAIN',
                           'CHAIN "none.qb"', 'CHAIN "../x.qb"')
                for text in ('1 line(s) overwritten', 'MERGED', '?USAGE: MERGE', '?FILE NOT FOUND',
                             '?USAGE: CHAIN', '?CHAIN ERROR'):
                    self.assertIn(text, out)
                self.assertIn('FLIP', t.subroutines)
            finally:
                os.chdir(cwd)


class TestDebugger(unittest.TestCase):

    def test_breakpoints_watch_trace(self):
        t = term(1)
        feed(t, '10 x = 1', '20 x = x + 1', '30 PRINT x')
        out = feed(t, 'BREAK 20', 'BREAK', 'BREAK 10-30', 'BREAK CLEAR', 'WATCH x', 'WATCH', 'WATCH x',
                   'WATCH CLEAR', 'TRON', 'RUN', 'TROFF')
        self.assertIn('[10]', out)
        for text in ('WATCHING: x', 'WATCH REMOVED: x', 'WATCHES CLEARED', 'TRACE ON', 'TRACE OFF'):
            self.assertIn(text, out)
        self.assertIn('No watches set', feed(t, 'WATCH'))

    def test_stop_and_cont(self):
        t = term(1)
        feed(t, '10 x = 1', '20 STOP', '30 PRINT "after"; x', 'WATCH x')
        out = feed(t, 'RUN')
        self.assertIn('STOPPED AT LINE 20', out)
        self.assertIn('x = 1', out)
        self.assertIn('after 1', feed(t, 'CONT'))

    def test_time_travel(self):
        t = term(1, inputs=['A0'])
        out = feed(t, 'REWIND', 'FORWARD', 'HISTORY')
        self.assertIn('?NO CHECKPOINTS', out)
        feed(t, '10 H 0', '20 Z 0', '30 H 0', 'STEP')
        out = feed(t, 'HISTORY', 'REWIND 2', 'FORWARD', 'FORWARD 9')
        self.assertIn('[2] line 30 <<', out)
        self.assertIn('REWIND to step 0 (line 10)', out)
        self.assertIn('FORWARD to step 1', out)
        self.assertIn('FORWARD to step 2', out)

    def test_error_statements(self):
        t = term(1)
        feed(t, '10 ON ERROR GOTO 100', '20 ERROR 42', '30 PRINT "resumed"', '40 END',
             '100 PRINT "caught"; ERR', '110 RESUME 30')
        out = feed(t, 'RUN')
        self.assertIn('caught 42', out)
        self.assertIn('resumed', out)
        t2 = term(1)
        feed(t2, '10 ASSERT 1 = 2')
        self.assertIn('ASSERTION FAILED: 1 = 2', feed(t2, 'RUN'))
        t3 = term(1)
        feed(t3, '10 ON ERROR GOTO 100', '20 x = 1 / 0', '30 END', '100 RESUME 999')
        self.assertIn('RESUME 999', feed(t3, 'RUN'))                  # caught when the program compiles


if __name__ == '__main__':
    unittest.main()
