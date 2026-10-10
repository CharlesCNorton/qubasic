"""Benchmarking commands, the web REPL server, and the CLI entry point."""

import io
import json
import os
import sys
import tempfile
import unittest

from qubasic_core.terminal import QBasicTerminal


class _IO:
    def __init__(self):
        self.parts = []

    def write(self, text):
        self.parts.append(text)

    def writeln(self, text):
        self.parts.append(text + '\n')

    def read_line(self, prompt):
        raise EOFError

    @property
    def text(self):
        return ''.join(self.parts)


def terminal(nq=2, shots=200):
    t = QBasicTerminal()
    t.io = _IO()
    t.num_qubits = nq
    t.shots = shots
    t._seed = 3
    return t


class TestBenchmarks(unittest.TestCase):

    def test_xeb_is_near_one_without_noise(self):
        t = terminal()
        t.cmd_xeb('2 3 3')
        self.assertIn('XEB', t.io.text.upper())
        self.assertNotIn('?', t.io.text.split('\n')[0])

    def test_qvolume_passes_noiseless(self):
        t = terminal()
        t.cmd_qvolume('2 4')
        self.assertIn('PASS', t.io.text.upper())

    def test_rbint_reports_fidelity(self):
        t = terminal(nq=1, shots=100)
        t.cmd_rbint('H 4')
        self.assertIn('fidelity', t.io.text.lower())

    def test_mirror_returns_to_zero(self):
        t = terminal(shots=100)
        t.cmd_mirror('2 2')
        self.assertIn('1.000', t.io.text)

    def test_gst_identity_channel(self):
        t = terminal(nq=1, shots=400)
        t.process('10 H 0', track_undo=False)
        t.process('20 H 0', track_undo=False)
        t.cmd_gst('')
        self.assertIn('TRANSFER MATRIX', t.io.text.upper())

    def test_concurrence_and_negativity_of_a_bell_pair(self):
        t = terminal()
        for line in ('10 H 0', '20 CX 0, 1'):
            t.process(line, track_undo=False)
        t.cmd_run()
        t.cmd_concurrence('0 1')
        t.cmd_negativity('0')
        self.assertIn('1.0000', t.io.text)


class TestCommandSurface(unittest.TestCase):
    """Smoke coverage for codes, devices, qudits, bosonic modes and the memory map."""

    def test_bb_code_and_bp_decoder(self):
        t = terminal()
        t.cmd_qec('BB 6 6')
        t.cmd_logical_error_rate('BB 0.02 200')
        self.assertIn('[[72,12,6]]', t.io.text.replace(' ', ''))
        self.assertIn('BP', t.io.text.upper())

    def test_resources_and_device_models(self):
        t = terminal()
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
        t = terminal()
        t.cmd_qudit('3 2')
        t.cmd_qx('0 1')
        t.cmd_qsum('0 1')
        t.cmd_qstate('')
        self.assertIn('|11>', t.io.text)
        t.cmd_qf('0')
        t.cmd_qmeasure('300')
        self.assertNotIn('ERROR', t.io.text)

    def test_bosonic_modes(self):
        t = terminal()
        t.cmd_bosonic('2 12')
        t.cmd_displace('0 1.0')
        t.cmd_squeeze('1 0.3')
        t.cmd_bs('0 1 0.5')
        t.cmd_bstate('')
        t.cmd_cat('0 1.0')
        self.assertNotIn('ERROR', t.io.text)

    def test_loccinfo_after_send(self):
        t = terminal()
        t.cmd_locc('JOINT 1 1')
        for line in ('10 SHARE A 0, B 0', '20 SEND A 0 -> c', '30 IF c THEN @B X 0',
                     '40 MEASURE'):
            t.process(line, track_undo=False)
        t.cmd_run()
        t.cmd_loccinfo()
        self.assertIn('SEND', t.io.text.upper())

    def test_memory_map_views(self):
        t = terminal()
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
        from qubasic_core.web_repl import make_server
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


if __name__ == '__main__':
    unittest.main()
