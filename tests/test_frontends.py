"""Front ends and presentation: the CLI, web REPL and Jupyter kernel, the
REPL loop, displays, export, analysis and benchmarking commands."""

import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from conftest import InTempDir, feed, term
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qubasic_core import cli, display, simulator
from qubasic_core.api import CaptureIO

# ---------------------------------------------------------------------------
# Benchmarking commands, the web REPL server, and the CLI entry point.
# ---------------------------------------------------------------------------


class TestBenchmarks(unittest.TestCase):

    def test_xeb_is_near_one_without_noise(self):
        t = term(seed=3, shots=200)
        t.cmd_xeb('2 3 3')
        self.assertIn('XEB', t.io.text.upper())
        self.assertNotIn('?', t.io.text.split('\n')[0])

    def test_qvolume_passes_noiseless(self):
        t = term(seed=3, shots=200)
        t.cmd_qvolume('2 4')
        self.assertIn('PASS', t.io.text.upper())

    def test_rbint_reports_fidelity(self):
        t = term(1, seed=3, shots=100)
        t.cmd_rbint('H 4')
        self.assertIn('fidelity', t.io.text.lower())

    def test_mirror_returns_to_zero(self):
        t = term(seed=3, shots=100)
        t.cmd_mirror('2 2')
        self.assertIn('1.000', t.io.text)

    def test_gst_identity_channel(self):
        t = term(1, seed=3, shots=400)
        t.process('10 H 0', track_undo=False)
        t.process('20 H 0', track_undo=False)
        t.cmd_gst('')
        self.assertIn('TRANSFER MATRIX', t.io.text.upper())

    def test_concurrence_and_negativity_of_a_bell_pair(self):
        t = term(seed=3, shots=200)
        for line in ('10 H 0', '20 CX 0, 1'):
            t.process(line, track_undo=False)
        t.cmd_run()
        t.cmd_concurrence('0 1')
        t.cmd_negativity('0')
        self.assertIn('1.0000', t.io.text)


class TestCommandSurface(unittest.TestCase):
    """Smoke coverage for codes, devices, qudits, bosonic modes and the memory map."""

    def test_bb_code_and_bp_decoder(self):
        t = term(seed=3, shots=200)
        t.cmd_qec('BB 6 6')
        t.cmd_logical_error_rate('BB 0.02 200')
        self.assertIn('[[72,12,6]]', t.io.text.replace(' ', ''))
        self.assertIn('BP', t.io.text.upper())

    def test_resources_and_device_models(self):
        t = term(seed=3, shots=200)
        t.cmd_resources('1e-12 0.001')
        self.assertIn('_FT_DISTANCE', t.variables)
        t.cmd_device('heavyhex 7')
        self.assertIsNotNone(t._coupling_map)
        t.cmd_device('')
        t.cmd_device('OFF')
        self.assertIsNone(t._coupling_map)
        t.cmd_device('torus 4')
        self.assertIn('unknown topology', t.io.text)

    def test_qudit_fourier_and_shift(self):
        t = term(seed=3, shots=200)
        t.cmd_qudit('3 2')
        t.cmd_qx('0 1')
        t.cmd_qsum('0 1')
        t.cmd_qstate('')
        self.assertIn('|11>', t.io.text)
        t.cmd_qf('0')
        t.cmd_qmeasure('300')
        self.assertNotIn('ERROR', t.io.text)

    def test_bosonic_modes(self):
        """Displacement, squeezing and a beamsplitter give each mode the mean
        photon number the formulas give; CAT replaces a free mode and refuses
        one entangled with the others."""
        t = term(seed=3, shots=200)

        def nbar(term_, mode):
            return float(np.sum(np.arange(term_.bosons.cutoff) * term_._mode_rdm_diag(mode)))

        t.cmd_bosonic('2 12')
        t.cmd_displace('0 1.0')                      # <n> = |alpha|^2
        t.cmd_squeeze('1 0.3')                       # <n> = sinh(r)^2
        n0, n1 = 1.0, np.sinh(0.3) ** 2
        self.assertAlmostEqual(nbar(t, 0), n0, places=4)
        self.assertAlmostEqual(nbar(t, 1), n1, places=4)
        t.cmd_bs('0 1 0.5')                          # mixes by cos^2 and sin^2 of theta
        c, s = np.cos(0.5) ** 2, np.sin(0.5) ** 2
        self.assertAlmostEqual(nbar(t, 0), c * n0 + s * n1, places=3)
        self.assertAlmostEqual(nbar(t, 1), s * n0 + c * n1, places=3)
        t.cmd_cat('0 1.0')
        self.assertIn('?CAT ERROR: mode 0 is entangled with the other modes', t.io.text)
        t.cmd_bstate('')
        self.assertIn('Mode 0: <n> = 0.7915', t.io.text)
        t2 = term(seed=3)
        t2.cmd_bosonic('2 12')
        t2.cmd_displace('1 0.5')
        t2.cmd_cat('0 1.0')                          # even cat: <n> = |alpha|^2 tanh |alpha|^2
        self.assertAlmostEqual(nbar(t2, 0), np.tanh(1.0), places=3)
        self.assertAlmostEqual(nbar(t2, 1), 0.25, places=3)

    def test_loccinfo_after_send(self):
        t = term(seed=3, shots=200)
        t.cmd_locc('JOINT 1 1')
        for line in ('10 SHARE A 0, B 0', '20 SEND A 0 -> c', '30 IF c THEN @B X 0',
                     '40 MEASURE'):
            t.process(line, track_undo=False)
        t.cmd_run()
        t.cmd_loccinfo()
        self.assertIn('SEND', t.io.text.upper())

    def test_memory_map_views(self):
        t = term(seed=3, shots=200)
        t.process('10 H 0', track_undo=False)
        t.cmd_run()
        t.cmd_map()
        t.cmd_dump('')
        self.assertIn('$D00C', t.io.text.upper())


class _Conn:
    """An in-memory connection: the request bytes in, the response bytes out."""

    def __init__(self, data):
        self._in = io.BytesIO(data)
        self.out = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        return self._in if 'r' in mode else self.out

    def sendall(self, data):
        self.out.write(data)


class TestWebServer(unittest.TestCase):
    """The request handler, driven through in-memory connections."""

    def setUp(self):
        from qubasic_core.cli import make_server
        self.httpd, self.token = make_server(0, '127.0.0.1')

    def tearDown(self):
        self.httpd.server_close()

    def _request(self, raw):
        conn = _Conn(raw)
        self.httpd.RequestHandlerClass(conn, ('127.0.0.1', 0), self.httpd)
        head, _, body = conn.out.getvalue().partition(b'\r\n\r\n')
        return int(head.split()[1]), body

    def _get(self, token):
        return self._request(f'GET /?t={token} HTTP/1.0\r\n\r\n'.encode())

    def _post(self, line, token=None):
        body = json.dumps({'line': line}).encode()
        raw = (f'POST /run?t={token or self.token} HTTP/1.0\r\n'
               f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n'
               ).encode() + body
        return self._request(raw)

    def test_page_needs_the_token(self):
        code, body = self._get(self.token)
        self.assertEqual(code, 200)
        self.assertIn(b'QUBASIC', body)
        self.assertEqual(self._get('wrong')[0], 403)

    def test_lines_build_and_run_a_program(self):
        for line in ('QUBITS 1', '10 X 0', '20 MEASURE'):
            self._post(line)
        code, body = self._post('RUN')
        self.assertEqual(code, 200)
        self.assertIn('RAN 2 lines', json.loads(body)['output'])

    def test_bad_token_and_oversized_body_are_refused(self):
        self.assertEqual(self._post('PRINT 1', token='nope')[0], 403)
        raw = (f'POST /run?t={self.token} HTTP/1.0\r\nContent-Length: 999999\r\n\r\n').encode()
        self.assertEqual(self._request(raw)[0], 413)


class TestCli(unittest.TestCase):

    def _main(self, *args):
        from qubasic_core import cli
        old_argv, old_out = sys.argv, sys.stdout
        sys.argv = ['qubasic', *args]
        sys.stdout = buf = io.StringIO()
        try:
            try:
                cli.main()
            except SystemExit:
                pass
        finally:
            sys.argv, sys.stdout = old_argv, old_out
        return buf.getvalue()

    def test_spec_lists_live_semantics(self):
        spec = json.loads(self._main('--spec'))
        names = {s['name'] for s in spec['statements']}
        self.assertIn('MEAS', names)
        self.assertIn('SELECT', names)

    def test_json_script(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'bell.qb')
            with open(path, 'w', encoding='utf-8') as f:
                f.write('QUBITS 2\n10 H 0\n20 CX 0, 1\n30 MEASURE\n')
            out = self._main('--json', '--seed', '7', path)
        data = json.loads(out[out.index('{'):])
        self.assertEqual(set(data['counts']), {'00', '11'})

    def test_version_and_help(self):
        from qubasic_core import __version__
        self.assertIn(__version__, self._main('--version'))
        self.assertIn('--json', self._main('--help'))


# ---------------------------------------------------------------------------
# The Jupyter kernel and kernelspec installer, run against stand-ins for
# ipykernel and jupyter_client so they are exercised without the extra.
# ---------------------------------------------------------------------------

class _FakeKernel:
    """The parts of ipykernel's Kernel that QubasicKernel uses."""

    def __init__(self, **kwargs):
        self.sent = []
        self.iopub_socket = object()
        self.execution_count = 1
        self.asked = []

    def send_response(self, socket, kind, content):
        self.sent.append((kind, content))

    def raw_input(self, prompt):
        self.asked.append(prompt)
        return '42'


def _fake_modules(installed):
    kernelbase = types.ModuleType('ipykernel.kernelbase')
    kernelbase.Kernel = _FakeKernel
    kernelapp = types.ModuleType('ipykernel.kernelapp')
    launched = []

    class IPKernelApp:
        @staticmethod
        def launch_instance(kernel_class):
            launched.append(kernel_class)

    kernelapp.IPKernelApp = IPKernelApp
    kernelspec = types.ModuleType('jupyter_client.kernelspec')

    class KernelSpecManager:
        def install_kernel_spec(self, source, name, user):
            with open(os.path.join(source, 'kernel.json'), encoding='utf-8') as f:
                installed.append((name, user, json.load(f)))
            return f'/kernels/{name}'

    kernelspec.KernelSpecManager = KernelSpecManager
    modules = {
        'ipykernel': types.ModuleType('ipykernel'),
        'ipykernel.kernelbase': kernelbase,
        'ipykernel.kernelapp': kernelapp,
        'jupyter_client': types.ModuleType('jupyter_client'),
        'jupyter_client.kernelspec': kernelspec,
    }
    return modules, launched


class TestJupyterKernel(unittest.TestCase):

    def setUp(self):
        self.installed = []
        modules, self.launched = _fake_modules(self.installed)
        for name, module in modules.items():
            self.addCleanup(self._restore, name, sys.modules.get(name))
            sys.modules[name] = module
        from qubasic_core import jupyter_kernel
        self.jk = jupyter_kernel

    @staticmethod
    def _restore(name, previous):
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    def test_cells_share_one_session(self):
        kernel = self.jk._make_kernel_class()()
        reply = kernel.do_execute('QUBITS 2\nSEED 4\n10 H 0\n20 CX 0, 1\n30 MEASURE', silent=False)
        self.assertEqual(reply['status'], 'ok')
        reply = kernel.do_execute('RUN', silent=False)
        self.assertEqual(reply['status'], 'ok')
        text = kernel.sent[-1][1]['text']
        self.assertIn('|00', text)
        self.assertEqual(set(kernel.term.last_counts), {'00', '11'})
        sent = len(kernel.sent)
        kernel.do_execute('PRINT 6 * 7', silent=True)
        self.assertEqual(len(kernel.sent), sent)

    def test_input_goes_through_the_notebook(self):
        kernel = self.jk._make_kernel_class()()
        kernel.do_execute('10 INPUT "n"; n\n20 PRINT n * 2\nRUN', silent=False, allow_stdin=True)
        self.assertEqual(kernel.asked, ['n? '])
        self.assertIn('84', kernel.sent[-1][1]['text'])

    def test_a_failing_cell_reports_an_error(self):
        kernel = self.jk._make_kernel_class()()
        with mock.patch.object(kernel.term, 'process', side_effect=RuntimeError('boom')):
            reply = kernel.do_execute('PRINT 1', silent=False)
        self.assertEqual(reply['status'], 'error')
        self.assertIn('?ERROR: boom', kernel.sent[-1][1]['text'])

    def test_install_and_launch(self):
        self.assertEqual(self.jk.install_kernelspec(), '/kernels/qubasic')
        name, user, spec = self.installed[0]
        self.assertEqual((name, user, spec['display_name'], spec['language']),
                         ('qubasic', True, 'QUBASIC', 'qubasic'))
        self.assertIn('qubasic_core.jupyter_kernel', spec['argv'])
        self.jk.main()
        self.assertEqual(self.launched[0].implementation, 'qubasic')


# ---------------------------------------------------------------------------
# The REPL loop and tab completion, command-line flags, the plain and rich
# histograms and statevector tables, the Bloch sphere labels, and the assist
# commands (THEME, EXPLAIN, UNDO, DIFF, COMPARE, QUIZ, CLIP).
# ---------------------------------------------------------------------------


def main(*args):
    old_argv, old_out = sys.argv, sys.stdout
    sys.argv = ['qubasic', *args]
    sys.stdout = buf = io.StringIO()
    code = 0
    try:
        cli.main()
    except SystemExit as e:
        code = e.code or 0
    finally:
        sys.argv, sys.stdout = old_argv, old_out
    return code, buf.getvalue()


class TestRepl(unittest.TestCase):

    def test_loop_runs_lines_until_end_of_input(self):
        t = term(inputs=['QUBITS 1', '10 X 0', '20 MEASURE', '', 'RUN'])
        with mock.patch.object(t, '_setup_readline'):
            t.repl()
        self.assertIn('Tip:', t.io.text)
        self.assertIn("|1", t.io.text)
        self.assertTrue(t.io.text.rstrip().endswith('BYE'))

    def test_interrupt_returns_to_the_prompt(self):
        class Io(CaptureIO):
            calls = 0

            def read_line(self, prompt):
                Io.calls += 1
                if Io.calls == 1:
                    raise KeyboardInterrupt
                return super().read_line(prompt)

        t = term()
        t.io = Io(['PRINT 7'])
        with mock.patch.object(t, '_setup_readline'):
            t.repl()
        self.assertIn(' 7 ', t.io.text)
        self.assertEqual(t._status_prompt(), '2q ] ')
        feed(t, 'LOCC 1 1', 'NOISE depolarizing 0.1')
        self.assertEqual(t._status_prompt(), '2q LOCC noisy ] ')

    def test_completions(self):
        t = term()
        feed(t, 'DEF BELLPAIR = H 0 : CX 0, 1', 'LET bvalue = 3', 'REG BREG 1')
        self.assertIn('PRINT', t._completions('PR'))
        self.assertIn('PRECISION', t._completions('pre'))
        self.assertIn('BELLPAIR', t._completions('BELL'))
        self.assertIn('bvalue', t._completions('BV'))
        self.assertIn('breg', t._completions('BR'))
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                open('alpha.qb', 'w').close()
                open('alpha.bas', 'w').close()
                self.assertEqual(t._completions('al', 'LOAD al'), ['alpha.qb'])
                self.assertEqual(t._completions('', 'SAVE '), ['alpha.qb'])
                self.assertEqual(t._completions('alpha.bas', 'LOAD alpha.bas'), ['alpha.bas'])
            finally:
                os.chdir(cwd)

    def test_readline_setup(self):
        fake = types.ModuleType('readline')
        fake.bound = []
        fake.set_completer = lambda f: setattr(fake, 'completer', f)
        fake.parse_and_bind = fake.bound.append
        fake.set_completer_delims = lambda d: setattr(fake, 'delims', d)
        fake.get_line_buffer = lambda: 'QU'
        t = term()
        with mock.patch.dict(sys.modules, {'readline': fake}):
            t._setup_readline()
        self.assertEqual(fake.completer('QU', 0), 'QUBITS ')
        self.assertIsNone(fake.completer('QU', 99))
        self.assertEqual(fake.delims, ' \t\n')
        self.assertIn('tab: complete', fake.bound)
        self.assertIn('"\\eOP": "DEMO BELL\\n"', fake.bound)


class TestCommandLine(unittest.TestCase):

    def test_version_and_help(self):
        code, out = main('--version')
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith('QUBASIC '))
        code, out = main('--help')
        self.assertIn('--install-kernel', out)

    def test_scripts(self):
        with tempfile.TemporaryDirectory() as d:
            ok = os.path.join(d, 'ok.qb')
            with open(ok, 'w') as f:
                f.write('QUBITS 1\n10 X 0\n20 MEASURE\n')
            code, out = main('--quiet', '--seed', '1', ok)
            self.assertEqual(code, 0)
            self.assertNotIn('QUBASIC', out.split('\n')[0])
            self.assertIn('|1', out)
            code, out = main('--json', os.path.join(d, 'missing.qb'))
            self.assertEqual((code, json.loads(out)['error'][:14]), (1, 'FILE NOT FOUND'))
            code, out = main(os.path.join(d, 'missing.qb'))
            self.assertIn('?FILE NOT FOUND', out)
            with mock.patch.object(cli, 'run_script', side_effect=RuntimeError('broken')):
                code, out = main('--json', ok)
                self.assertEqual((code, json.loads(out)), (1, {'error': 'broken'}))
                code, out = main('--quiet', ok)
                self.assertIn('?ERROR: broken', out)
            bas = os.path.join(d, 'p.bas')
            with open(bas, 'w') as f:
                f.write('PRINT "from bas"\n')
            code, out = main('--quiet', bas)
            self.assertIn('from bas', out)

    def test_install_kernel(self):
        with mock.patch('qubasic_core.jupyter_kernel.install_kernelspec', return_value='/k/qubasic'):
            code, out = main('--install-kernel')
        self.assertEqual(code, 0)
        self.assertIn('/k/qubasic', out)
        with mock.patch('qubasic_core.jupyter_kernel.install_kernelspec', side_effect=ImportError):
            code, out = main('--install-kernel')
        self.assertEqual(code, 1)
        self.assertIn('pip install qubasic[jupyter]', out)

    def test_spec(self):
        code, out = main('--spec')
        spec = json.loads(out)
        self.assertEqual(code, 0)
        self.assertTrue(any(f['name'] == 'LEN' for f in spec['functions']))
        self.assertIn('PI', spec['constants'])


class TestHistogramsAndTables(unittest.TestCase):

    def _wide_run(self, n=6):
        t = term(n)
        t.shots = 4000
        feed(t, f'10 FOR q = 0 TO {n - 1}: H q: NEXT q', '20 MEASURE', 'RUN')
        return t

    def test_rich_tables_truncate(self):
        t = self._wide_run()
        self.assertIn('Showing top', t.io.text)
        self.assertIn('(remaining)', t.io.text)
        t7 = term(7)
        out = feed(t7, '10 FOR q = 0 TO 6: H q: NEXT q', 'RUN', 'STATE')
        self.assertIn('+64 more', out)                      # 128 amplitudes, 64 shown

    def test_plain_output(self):
        t = term()
        with mock.patch.object(display, '_RICH', False), mock.patch.object(simulator, '_RICH', False):
            out = feed(t, '10 H 0', '20 MEASURE', 'RUN')
            self.assertIn('%)', out)
            out = feed(t, '20', 'RUN', 'STATE')
            self.assertIn('Statevector (2 qubits)', out)
            t6 = self._wide_run()
            self.assertIn('Showing top', t6.io.text)
            t._theme = {'reset': '<r>', 'bar_hi': '<h>', 'bar_mid': '<m>', 'bar_lo': '<l>'}
            with mock.patch.object(sys.stdout, 'isatty', return_value=True):
                t.print_histogram({'00': 60, '01': 25, '11': 15})
            self.assertIn('<h>', t.io.text)
            self.assertIn('<m>', t.io.text)

    def test_bloch_labels(self):
        cases = (('H 0: S 0', '|+i⟩'), ('H 0: SDG 0', '|-i⟩'), ('H 0', '|+⟩'), ('X 0: H 0', '|-⟩'),
                 ('X 0', '|1⟩ (south pole)'), ('RY 1, 0', 'θ=1.00'))
        for prog, label in cases:
            t = term(1)
            out = feed(t, f'10 {prog}', 'RUN', 'BLOCH 0')
            self.assertIn(label, out, prog)
        t = term()
        out = feed(t, '10 H 0', '20 CX 0, 1', 'RUN', 'BLOCH 0')
        self.assertIn('maximally mixed', out)
        self.assertIn('(0.000, 0.000, 0.000)', out)

    def test_step_shows_compact_states(self):
        t = term(4, inputs=['A0'])
        feed(t, '10 FOR q = 0 TO 3: H q: NEXT q', '20 S 0')
        out = feed(t, 'STEP')
        self.assertIn('...', out)
        self.assertIn('j)|', out)


class TestAssist(unittest.TestCase):

    def test_theme_explain_list(self):
        t = term()
        out = feed(t, 'THEME', 'THEME neon', 'THEME retro')
        self.assertIn('Current theme: default', out)
        self.assertIn('?Unknown theme: neon', out)
        self.assertIn('THEME = retro', out)
        self.assertIn('EMPTY PROGRAM', feed(t, 'EXPLAIN'))
        feed(t, '10 REM setup', '20 H 0 : CX 0, 1', '30 RX PI/4, 1', '40 FOR i = 1 TO 2', '50 NEXT i',
             '60 PRINT "x"', '70 FOO')
        out = feed(t, 'EXPLAIN')
        for text in ('(comment)', 'put qubit 0 in superposition; CNOT: flip 1 if 0=|1>',
                     'rotate 1 around X by PI/4', 'begin loop', 'output text or value', 'FOO'):
            self.assertIn(text, out)
        t2 = term()
        feed(t2, 'LOCC 1 1', '10 @A H 0', '20 SEND A 0 -> m', 'LOCC JOINT 1 1', '30 SHARE A 0, B 0')
        out = feed(t2, 'EXPLAIN')
        self.assertIn('@A: H 0', out)
        self.assertIn('measure and send classical bit', out)
        self.assertIn('create shared entanglement', out)
        self.assertIn('EMPTY PROGRAM', feed(term(), 'LIST'))
        t._theme = {'comment': '<c>', 'gate': '<g>', 'flow': '<f>', 'string': '<s>', 'reset': '</>'}
        t.cmd_list_colored()
        self.assertIn('<g>H</>', t.io.text)
        self.assertIn('<f>FOR</>', t.io.text)
        self.assertIn('<s>"x"</>', t.io.text)
        self.assertIn('<c>REM setup</>', t.io.text)

    def test_undo_diff_compare(self):
        t = term()
        t.process('10 H 0')
        t.process('20 X 1')
        t.process('20 Y 1')
        out = feed(t, 'UNDO')
        self.assertIn('Undo will revert 20: Y 1 -> X 1', out)
        t.process('30 Z 0')
        self.assertIn('Undo will remove: [30]', feed(t, 'UNDO'))
        t.process('20')
        self.assertIn('Undo will restore: [20]', feed(t, 'UNDO'))
        t.editor.undo.clear()
        self.assertIn('?NOTHING TO UNDO', feed(t, 'UNDO'))
        out = feed(t, 'DIFF', 'DIFF 3')
        self.assertIn('?USAGE: DIFF', out)
        self.assertIn('?BANK 3 is empty', out)
        t.editor.slots[3] = {10: 'H 0'}
        self.assertIn('+20 X 1', feed(t, 'DIFF 3'))
        t.editor.slots[4] = dict(t.program)
        self.assertIn('No differences with bank 4', feed(t, 'DIFF 4'))
        feed(t, '30 MEASURE')
        out = feed(t, 'COMPARE statevector stabilizer', 'COMPARE statevector')
        self.assertIn('stabilizer', out)
        self.assertIn('?USAGE: COMPARE', out)

    def test_quiz_and_clip(self):
        t = term(inputs=['Z'])
        out = feed(t, 'QUIZ')
        self.assertIn('Wrong -- the answer is', out)
        with mock.patch('random.choice', side_effect=lambda seq: seq[0]), \
                mock.patch('random.shuffle'):                   # options stay in order: A is right
            self.assertIn('Correct!', feed(term(inputs=['A']), 'QUIZ'))
            self.assertNotIn('Correct', feed(term(inputs=[]), 'QUIZ'))
        t2 = term()
        self.assertIn('?NO RESULTS TO COPY', feed(t2, 'CLIP'))
        feed(t2, '10 H 0', '20 MEASURE', 'RUN')
        with mock.patch('subprocess.run', side_effect=OSError('no clipboard')):
            out = feed(t2, 'CLIP')
        self.assertIn('?Clipboard not available', out)
        with mock.patch('subprocess.run') as run:
            out = feed(t2, 'CLIP')
        self.assertIn('COPIED 2 lines', out)
        run.assert_called_once()
        t3 = term(1)
        feed(t3, '10 H 0', 'RUN')
        with mock.patch('subprocess.run', side_effect=OSError):
            out = feed(t3, 'CLIP')
        self.assertIn('+0.7071+0.0000j', out)


# ---------------------------------------------------------------------------
# State display, export and quantum-ML commands: the outputs they produce,
# the files they write and the errors they report.
# ---------------------------------------------------------------------------


class TestStateDisplay(unittest.TestCase):

    def test_commands_before_a_run(self):
        t = term(agent=True)
        out = feed(t, 'STATE', 'HIST', 'PROBS', 'BLOCH', 'CIRCUIT', 'DECOMPOSE')
        for msg in ('?NO STATE', '?NO RESULTS', '?NO CIRCUIT'):
            self.assertIn(msg, out)

    def test_state_probs_bloch_circuit_decompose(self):
        t = term(agent=True)
        feed(t, '10 H 0', '20 CX 0, 1', 'RUN')
        out = feed(t, 'STATE', 'PROBS', 'BLOCH', 'BLOCH 1', 'CIRCUIT', 'DECOMPOSE')
        self.assertIn('0.7071', out)
        self.assertIn('50.0', out)
        self.assertIn('cx', out)
        self.assertIn('Circuit: 2 qubits', out)

    def test_print_forms(self):
        t = term(3, agent=True)
        feed(t, '10 H 0', '20 CX 0, 1', '30 S 1', '40 PRINT STATE', '50 PRINT QUBIT(2)',
             '60 PRINT ENTANGLEMENT(0, 1)', '70 X 2', '80 Z 2', '90 PRINT STATE')
        out = feed(t, 'RUN')
        self.assertIn('+0.7071|000⟩', out)
        self.assertIn('0.707j)|011⟩', out)
        self.assertIn('q2: P(1)=0.0000 Bloch=(0.000,0.000,1.000)', out)
        self.assertIn('E(q0,q1) = 1.000000 ebits (concurrence 1.000000)', out)
        self.assertIn('-0.7071|100⟩', out)

    def test_print_errors(self):
        for stmt, msg in (('PRINT QUBIT(5)', 'OUT OF RANGE'), ('PRINT ENTANGLEMENT(0, 0)', 'two different'),
                          ('PRINT ENTANGLEMENT(0)', 'takes two qubits')):
            t = term(agent=True)
            out = feed(t, '10 H 0', f'20 {stmt}', 'RUN')
            self.assertIn(msg, out)

    def test_dirac_notation_truncates_after_sixteen_terms(self):
        t = term(5, agent=True)
        feed(t, '10 FOR q = 0 TO 4: H q: NEXT q', '20 PRINT STATE')
        out = feed(t, 'RUN')
        self.assertIn('...', out)
        self.assertEqual(out.count('⟩'), 16)

    def test_basis_and_phase_states(self):
        t = term(1, agent=True)
        self.assertEqual(t._format_dirac(np.array([0, 1]), 1), '|1⟩')
        self.assertEqual(t._format_dirac(np.array([0, -1]), 1), '-|1⟩')
        self.assertEqual(t._format_dirac(np.array([0, 0]), 1), '|0⟩')
        self.assertEqual(t._format_dirac(np.array([1, 0, 0, 0]), 2, segments=[1, 1]), '|0|0⟩')

    def test_live_run_prints_the_shot_state(self):
        t = term(1, agent=True)
        t.shots = 4
        feed(t, '10 H 0', '20 MEAS 0 -> m', '30 PRINT m; STATE')
        out = feed(t, 'RUN')
        self.assertTrue('|0⟩' in out or '|1⟩' in out)

    def test_locc_registers(self):
        t = term(agent=True)
        feed(t, 'LOCC 1 1', '10 @A X 0', '20 PRINT @A', '30 PRINT @B', '40 PRINT STATE')
        out = feed(t, 'RUN')
        self.assertIn('|1⟩', out)
        self.assertIn('SPLIT registers have no joint state', out)
        out = feed(t, 'STATE A', 'BLOCH A 0')
        self.assertIn('Register A', out)
        t2 = term(agent=True)
        feed(t2, 'LOCC JOINT 1 1', '10 @A X 0', '20 PRINT @A', '30 PRINT STATE')
        out = feed(t2, 'RUN')
        self.assertIn('|1|0⟩', out)              # register A first


class TestExport(InTempDir):

    def test_export_and_loadqasm_round_trip(self):
        t = term(agent=True)
        feed(t, '10 H 0', '20 CX 0, 1', '30 RZ 0.25, 1', '40 BARRIER', '50 RESET 0', '60 MEASURE', 'RUN')
        out = feed(t, 'EXPORT')
        self.assertIn('OPENQASM 2.0', out)
        out = feed(t, 'EXPORT bell.qasm', 'EXPORT bell.qasm')
        self.assertIn('EXPORTED to', out)
        self.assertIn('overwriting', out)
        t2 = term(agent=True)
        out = feed(t2, 'LOADQASM bell')
        self.assertIn('2 qubits', out)
        self.assertEqual([t2.program[k] for k in sorted(t2.program)],
                         ['H 0', 'CX 0, 1', 'RZ 0.25, 1', 'BARRIER', 'RESET 0', 'MEASURE'])

    def test_loadqasm_imports_unknown_gates_as_matrices(self):
        from qiskit.circuit.library import XXPlusYYGate
        qc = QuantumCircuit(2)
        qc.append(XXPlusYYGate(0.3, 0.2), [0, 1])
        lines, n_unknown, custom = term(agent=True)._circuit_to_qb(qc)
        self.assertEqual((lines, n_unknown), (['G0 0, 1'], 1))
        self.assertEqual(custom['G0'].shape, (4, 4))
        known = QuantumCircuit(2)
        known.rzx(0.3, 0, 1)
        self.assertEqual(term(agent=True)._circuit_to_qb(known)[:2], (['RZX 0.3, 0, 1'], 0))

    def test_loadqasm_errors(self):
        t = term(agent=True)
        with open('bad.qasm', 'w') as f:
            f.write('not qasm at all')
        out = feed(t, 'LOADQASM', 'LOADQASM missing.qasm', 'LOADQASM bad.qasm', 'LOADQASM /abs/path.qasm')
        self.assertIn('?USAGE: LOADQASM', out)
        self.assertIn('?FILE NOT FOUND', out)
        self.assertIn('could not parse', out)

    def test_export_needs_a_run(self):
        self.assertIn('?NO CIRCUIT', feed(term(agent=True), 'EXPORT'))
        self.assertIn('?NO RESULTS', feed(term(agent=True), 'CSV'))

    def test_csv_to_screen_and_file(self):
        t = term(agent=True)
        feed(t, '10 H 0', '20 CX 0, 1', '30 MEASURE', 'RUN')
        out = feed(t, 'CSV')
        self.assertIn('state,count,probability', out)
        out = feed(t, 'CSV results.csv', 'CSV results.csv')
        self.assertIn('EXPORTED 2 states', out)
        self.assertIn('overwriting', out)
        text = open('results.csv').read()
        self.assertIn('state,amplitude_re,amplitude_im,probability', text)
        t2 = term(5, agent=True)
        feed(t2, '10 FOR q = 0 TO 4: H q: NEXT q', 'RUN')
        out = feed(t2, 'CSV')
        self.assertIn('more)', out)

    def test_amplitude_rows_keep_the_most_probable(self):
        t = term(3, agent=True)
        t.MAX_EXPORT_AMPLITUDES = 2
        sv = np.array([0.1, 0.9, 0.3, 0, 0, 0.2, 0, 0.2], dtype=complex)
        rows = t._amplitude_rows(sv / np.linalg.norm(sv))
        self.assertTrue(rows[0].startswith('# 5 non-zero amplitudes'))
        self.assertEqual([r.split(',')[0] for r in rows[1:]], ['001', '010'])

    def test_savepng(self):
        t = term(agent=True)
        out = feed(t, 'SAVEPNG')
        self.assertIn('?USAGE: SAVEPNG', out)
        out = feed(t, 'SAVEPNG pic screen', 'SAVEPNG pic hist', 'SAVEPNG pic bloch', 'SAVEPNG pic circuit',
                   'SAVEPNG pic sideways')
        self.assertIn('?NO GRAPHICS SCREEN', out)
        feed(t, '10 H 0', '20 MEASURE', 'RUN')
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.assertIn('requires matplotlib', feed(t, 'SAVEPNG hist.png hist'))
            return
        out = feed(t, 'SAVEPNG hist.png hist', 'SAVEPNG bloch.png bloch', 'SAVEPNG pic sideways')
        self.assertIn('SAVED hist image', out)
        self.assertIn('SAVED bloch image', out)
        self.assertIn('must be screen, hist, bloch or circuit', out)
        self.assertTrue(os.path.getsize('hist.png') > 0)
        feed(t, 'SCREEN 13', '5 PSET (1, 1), 4', 'RUN')
        self.assertIn('SAVED screen image', feed(t, 'SAVEPNG frame'))
        with open('frame.png', 'rb') as f:
            self.assertEqual(f.read(8), b'\x89PNG\r\n\x1a\n')


class TestQML(unittest.TestCase):

    def test_feature_map_matches_qiskit(self):
        t = term(3, agent=True)
        x = [0.3, 1.1, 2.0]
        qc = QuantumCircuit(3)
        t._emit_feature_map(qc, x)
        try:
            from qiskit.circuit.library import zz_feature_map
            ref = zz_feature_map(3, reps=2, entanglement='linear')
        except ImportError:
            from qiskit.circuit.library import ZZFeatureMap
            ref = ZZFeatureMap(3, reps=2, entanglement='linear')
        ref = ref.assign_parameters(x)
        self.assertAlmostEqual(abs(np.vdot(Statevector(qc).data, Statevector(ref).data)), 1.0, places=9)

    def test_featuremap_statement(self):
        t = term(agent=True)
        feed(t, '10 FEATUREMAP 0.5, 1.5', '20 SAVE_PROBS 0, 1 -> p', 'RUN')
        self.assertAlmostEqual(sum(t.arrays['p']), 1.0)
        out = feed(t, '10 FEATUREMAP 0.5', 'RUN')
        self.assertIn('needs 2 values', out)

    def test_qkernel(self):
        t = term(agent=True)
        feed(t, 'QKERNEL 0.5 0.3 ; 0.5 0.3')
        self.assertAlmostEqual(t.variables['_QKERNEL'], 1.0)
        feed(t, 'QKERNEL 0.5 0.3 ; 0.4 0.9')
        self.assertLess(t.variables['_QKERNEL'], 1.0)
        out = feed(t, 'QKERNEL 0.5 0.3', 'QKERNEL 0.5 ; 0.1 0.2')
        self.assertIn('?USAGE: QKERNEL', out)
        self.assertIn('?QKERNEL ERROR', out)


# ---------------------------------------------------------------------------
# Analysis commands against exact values: expectation values, entropies,
# density matrices, consistency checks, fidelities and state and process
# tomography.
# ---------------------------------------------------------------------------


def bell():
    t = term()
    feed(t, '10 H 0', '20 CX 0, 1', 'RUN')
    return t


class TestExpectAndEntropy(unittest.TestCase):

    def test_expect(self):
        t = bell()
        out = feed(t, 'EXPECT ZZ 0 1', 'EXPECT XX 0 1', 'EXPECT YY', 'EXPECT Z 1')
        self.assertIn('<ZZ> on qubits [0, 1] = 1.000000', out)
        self.assertIn('<XX> on qubits [0, 1] = 1.000000', out)
        self.assertIn('<YY> on qubits [0, 1] = -1.000000', out)
        self.assertIn('<Z> on qubits [1] = 0.000000', out)
        out = feed(t, 'EXPECT', 'EXPECT Q 0', 'EXPECT ZZ 0', 'EXPECT Z 5')
        for msg in ('?USAGE: EXPECT', 'must be over I, X, Y, Z', 'needs 2 qubit(s), got 1', 'out of range'):
            self.assertIn(msg, out)
        self.assertIn('?NO STATE', feed(term(), 'EXPECT Z 0'))

    def test_entropy(self):
        t = bell()
        out = feed(t, 'ENTROPY', 'ENTROPY 0 1', 'ENTROPY 4')
        self.assertIn('S(A) = 1.000000 bits', out)
        self.assertIn('Maximally entangled', out)
        self.assertIn('S(A) = 0.000000 bits', out)
        self.assertIn('separable', out)
        self.assertIn('?ENTROPY ERROR', out)
        self.assertIn('?NO STATE', feed(term(), 'ENTROPY'))

    def test_split_registers(self):
        t = term()
        feed(t, 'LOCC 1 1', '10 @A H 0', 'RUN')
        out = feed(t, 'EXPECT A X 0', 'ENTROPY A 0', 'DENSITY A', 'EXPECT Z 0', 'ENTROPY', 'DENSITY')
        self.assertIn('<X> on qubits [0] = 1.000000', out)
        self.assertIn('S(A) = 0.000000', out)
        self.assertIn('0.500    0.500', out)
        self.assertIn('?SPLIT mode: prefix a register, e.g. EXPECT A Z 0', out)
        self.assertIn('?SPLIT mode: prefix a register, e.g. ENTROPY A 0', out)
        self.assertIn('?SPLIT mode: prefix a register, e.g. DENSITY A', out)


class TestDensity(unittest.TestCase):

    def test_pure_mixed_and_large(self):
        t = term(1)
        feed(t, '10 H 0', '20 S 0', 'RUN')
        out = feed(t, 'DENSITY')
        self.assertIn('+0.00-0.50j', out)
        self.assertIn('Purity: 1.000000', out)
        t2 = term(1)
        feed(t2, 'SET_DENSITY [[0.75, 0], [0, 0.25]]', '10 BARRIER', 'RUN')
        out = feed(t2, 'DENSITY')
        self.assertIn('0.750', out)
        self.assertIn('Purity: 0.625000', out)
        t3 = term(5)
        feed(t3, '10 H 0', 'RUN')
        self.assertIn('32x32 (too large to display)', feed(t3, 'DENSITY'))
        self.assertIn('?NO STATE', feed(term(), 'DENSITY'))


class TestConsistencyAndFidelity(unittest.TestCase):

    def test_consistency(self):
        t = term()
        t.shots = 2000
        feed(t, '10 H 0', '20 CX 0, 1', '30 MEASURE', 'RUN')
        out = feed(t, 'CONSISTENCY')
        self.assertIn('ALL CONSISTENT', out)
        self.assertIn('Histogram ~ state', out)
        feed(t, '30 MEASURE 0', 'RUN')
        self.assertIn('partial MEASURE subset', feed(t, 'CONSISTENCY'))
        feed(t, '30 MEASURE', 'NOISE depolarizing 0.05', 'RUN')
        self.assertIn('one noisy trajectory', feed(t, 'CONSISTENCY'))
        ok, detail = t._histogram_agrees(np.array([1, 0, 0, 0]), {'00': 10, '11': 1}, 2)
        self.assertFalse(ok)
        self.assertIn('forbids', detail)
        self.assertEqual(t._histogram_agrees(np.array([1, 0]), {'0': 50}, 1), (True, 'one outcome'))
        self.assertIn('?NO STATE', feed(term(), 'CONSISTENCY'))

    def test_fidelity(self):
        t = bell()
        out = feed(t, 'FIDELITY |BELL>', 'FIDELITY [1, 0, 0, -1]', 'FIDELITY [0, 1, 1, 0]',
                   'FIDELITY [1, 0]', 'FIDELITY')
        self.assertIn('F = |<target|psi>|^2 = 1.000000', out)
        self.assertIn('states match', out)
        self.assertIn('F = |<target|psi>|^2 = 0.000000', out)
        self.assertIn('orthogonal', out)
        self.assertIn('target length 2 != 2^2 = 4', out)
        self.assertIn('?USAGE: FIDELITY', out)
        self.assertIn('?NO STATE', feed(term(), 'FIDELITY |0>'))


class TestTomography(unittest.TestCase):

    def test_state_tomography(self):
        t = bell()
        out = feed(t, 'TOMOGRAPHY')
        self.assertIn('Fidelity to the simulated pure state = 1.000000', out)
        self.assertIn('Purity Tr(rho^2) = 1.000000', out)
        out = feed(t, 'TOMOGRAPHY 4000')
        fid = float(out.split('pure state = ')[1].split()[0])
        self.assertGreater(fid, 0.95)
        t4 = term(4)
        feed(t4, '10 H 0', 'RUN')
        self.assertIn('limited to 3 qubits', feed(t4, 'TOMOGRAPHY'))
        self.assertIn('?NO STATE', feed(term(), 'TOMOGRAPHY'))

    def test_process_tomography(self):
        t = term(1)
        feed(t, '10 H 0')
        out = feed(t, 'PTOMOGRAPHY')
        rows = {line.split()[0]: [float(v) for v in line.split()[1:]] for line in out.splitlines()
                if line.strip()[:1] in 'IXYZ' and len(line.split()) == 5}
        np.testing.assert_allclose(rows['X'], [0, 0, 0, 1], atol=1e-9)     # H maps Z to X
        self.assertIn('Trace-preserving: True   Unital: True', out)
        feed(t, 'NOISE amplitude_damping 0.2', '10 ID 0')
        self.assertIn('Unital: False', feed(t, 'PTOMOGRAPHY'))
        self.assertIn('limited to 2 qubits', feed(term(3), '10 H 0', 'PTOMOGRAPHY'))
        self.assertIn('?NOTHING TO CHARACTERIZE', feed(term(), 'PTOMOGRAPHY'))


class TestBench(unittest.TestCase):

    def test_bench_rows(self):
        out = feed(term(), 'BENCH 2 3')
        rows = [line.split() for line in out.splitlines() if line.strip()[:1].isdigit()]
        self.assertEqual([r[0] for r in rows], ['2', '3'])
        self.assertTrue(all(r[1].endswith('s') for r in rows))



class _FakeBackend:
    """A five-qubit stand-in for an IBM device: GenericBackendV2's target and
    calibration, with a status like the real ones."""

    def __init__(self):
        from qiskit.providers.fake_provider import GenericBackendV2
        self._real = GenericBackendV2(5, seed=3)
        self.name = 'fake_five'
        self.num_qubits = 5
        self.target = self._real.target

    def __getattr__(self, attr):
        return getattr(self._real, attr)

    def status(self):
        return types.SimpleNamespace(operational=True, pending_jobs=4)


class _FakeJob:
    def __init__(self, result, backend):
        self._result, self._backend = result, backend

    def job_id(self):
        return 'job-123'

    def result(self):
        return self._result

    def backend(self):
        return self._backend

    def usage(self):
        return 2.0


class _FakeRuntime:
    """qiskit_ibm_runtime as QPU and DEVICE IBM use it, with Aer behind it."""

    def __init__(self):
        backend = _FakeBackend()
        jobs = {}

        class Service:
            def backends(self):
                return [backend]

            def backend(self, name):
                return backend

            def job(self, job_id):
                return jobs[job_id]

        class Sampler:
            def __init__(self, mode):
                self.mode = mode

            def run(self, pubs, shots):
                from qiskit_aer.primitives import SamplerV2
                job = _FakeJob(SamplerV2(seed=7).run(pubs, shots=shots).result(), self.mode)
                jobs[job.job_id()] = job
                return job

        self.QiskitRuntimeService = Service
        self.SamplerV2 = Sampler


class TestIBMHardware(unittest.TestCase):
    """QPU and DEVICE IBM against a stand-in for Qiskit Runtime."""

    def setUp(self):
        from qubasic_core import benchmarking
        patcher = mock.patch.object(benchmarking, '_runtime', return_value=_FakeRuntime())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_qpu_lists_runs_and_collects(self):
        t = term(2, seed=1)
        self.assertIn('fake_five', feed(t, 'QPU LIST'))
        out = feed(t, '10 H 0', '20 CX 0, 1', '30 MEASURE', 'QPU fake_five SHOTS 400')
        self.assertIn('job job-123 queued, 400 shots', out)
        self.assertIn('2.0 s of QPU time', out)
        self.assertEqual(set(t.last_counts), {'00', '11'})
        self.assertEqual(sum(t.last_counts.values()), 400)
        t.last_counts = None
        feed(t, 'QPU JOB job-123')
        self.assertEqual(sum(t.last_counts.values()), 400)

    def test_qpu_refuses_what_a_device_cannot_run(self):
        t = term(1, seed=1)
        self.assertIn('no MEASURE', feed(t, '10 H 0', 'QPU fake_five'))
        out = feed(t, '20 MEAS 0 -> m', '30 PRINT m', '40 MEASURE', 'QPU fake_five')
        self.assertIn('runs shot by shot', out)
        self.assertIn('?USAGE: QPU', feed(t, 'QPU'))

    def test_device_ibm_simulates_the_calibrated_device(self):
        t = term(2, seed=1, shots=2000)
        out = feed(t, 'DEVICE IBM fake_five', '10 H 0', '20 CX 0, 1', '30 MEASURE', 'RUN')
        self.assertIn('DEVICE IBM fake_five: 5 qubits', out)
        self.assertIn('routed onto device', out)
        wrong = (t.last_counts.get('01', 0) + t.last_counts.get('10', 0)) / 2000
        self.assertGreater(wrong, 0.0)                               # the calibration's errors show
        self.assertLess(wrong, 0.2)
        feed(t, 'DEVICE OFF')
        self.assertIsNone(t._device_target)

    def test_delay_decoheres_only_under_a_device_model(self):
        t = term(1, seed=1, shots=2000)
        feed(t, '10 H 0', '20 DELAY 0, 300', '30 H 0', '40 MEASURE', 'RUN')
        self.assertEqual(t.last_counts, {'0': 2000})                 # ideal: idling changes nothing
        feed(t, 'DEVICE IBM fake_five', 'RUN')
        self.assertGreater(t.last_counts.get('1', 0), 200)           # T1 and T2 over 300 us
        self.assertIn('negative', feed(term(1), '10 DELAY 0, -1', 'RUN'))


if __name__ == '__main__':
    unittest.main()
