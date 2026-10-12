"""QBASIC language fidelity: formatting, types, arrays, procedures, labels, I/O."""

import os
import tempfile
import time
import unittest

from qubasic_core.basic_format import format_using, number_text, print_text, str_text
from qubasic_core.debug import error_code
from qubasic_core.file_io import number_source
from qubasic_core.program import CompileError, compile_program
from qubasic_core.terminal import QBasicTerminal


class _IO:
    def __init__(self, answers=()):
        self.parts = []
        self.answers = list(answers)
        self.prompts = []

    def write(self, text):
        self.parts.append(text)

    def writeln(self, text):
        self.parts.append(text + '\n')

    def read_line(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    @property
    def text(self):
        return ''.join(self.parts)


def run(source, answers=(), nq=1, seed=None):
    """Run QBASIC source lines (numbered 10, 20, ... when unnumbered)."""
    t = QBasicTerminal()
    t.num_qubits = nq
    t.shots = 16
    t._seed = seed
    io = _IO(answers)
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
        io = _IO()
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
        io = _IO()
        t.io = io
        t._load_bas(path)
        t.cmd_run()
        self.assertEqual(io.text, 'Sorted: 3  4  5  7  9  15  26  31 \n'
                                  'Primes: 3  5  7  31 \nMean: 12.50\nmedium\n')

    def test_save_and_load_bas(self):
        with tempfile.TemporaryDirectory(dir='.') as d:
            rel = os.path.relpath(os.path.join(d, 'p.bas'))
            t = QBasicTerminal()
            t.io = _IO()
            t.process('10 PRINT "x"', track_undo=False)
            t.process('20 GOTO 40', track_undo=False)
            t.process('40 END', track_undo=False)
            t.cmd_save(rel)
            t2 = QBasicTerminal()
            t2.io = _IO()
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
        t.io = _IO()
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
        t.io = _IO()
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
        t.io = _IO()
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
            t.io = _IO()
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
            t2.io = _IO()
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
            t.io = _IO()
            t.cmd_device('ring 4 80 60')
            t.cmd_save(rel)
            t2 = QBasicTerminal()
            t2.io = _IO()
            t2.cmd_load(rel)
            self.assertEqual(t2._device_spec, 'ring 4 80 60')
            self.assertIsNotNone(t2._noise_model)
            self.assertNotIn('?', t2.io.text)


if __name__ == '__main__':
    unittest.main()
