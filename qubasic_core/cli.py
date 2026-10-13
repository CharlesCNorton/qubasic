"""QUBASIC command line (`qubasic`, `python -m qubasic_core`) and the
browser REPL (`qubasic --web`).
"""

from __future__ import annotations

import io
import json
import os
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from qubasic_core.editing import ProgramMgmtMixin
from qubasic_core.terminal import QBasicTerminal

# ---------------------------------------------------------------------------
# QUBASIC — command-line entry point.
#
# Usage:
#     qubasic                       Interactive REPL (installed console script)
#     python -m qubasic_core        Same, run as a module
#     qubasic script.qb             Run a script file
# ---------------------------------------------------------------------------

#!/usr/bin/env python3


# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
# Piped input (heredocs, PowerShell pipes) often arrives with a UTF-8 BOM
# that would otherwise mangle the first command; utf-8-sig strips it.
if sys.stdin and hasattr(sys.stdin, 'reconfigure'):
    try:
        if not sys.stdin.isatty():
            sys.stdin.reconfigure(encoding='utf-8-sig')
    except Exception:
        pass


# The command line, as --help prints it and docs/reference.md lists it.
USAGE = (
    ('qubasic', 'Interactive REPL'),
    ('qubasic script.qb', 'Run a script file (a .bas file is numbered and RUN)'),
    ('qubasic --quiet script', 'Suppress banner and progress (also -q)'),
    ('qubasic --json script', 'Output results as JSON'),
    ('qubasic --agent script', 'Confine file writes to the working dir'),
    ('qubasic --seed N script', 'Set random seed for reproducibility'),
    ('qubasic --spec', 'Print the commands, gates, functions and conventions as JSON'),
    ('qubasic --web [port]', 'Serve a browser REPL (localhost, token-gated)'),
    ('qubasic --install-kernel', 'Install the Jupyter kernelspec'),
    ('qubasic --version', 'Show version (also -v)'),
    ('qubasic --help', 'Show this help (also -h)'),
    ('python -m qubasic_core', 'Run without the installed console script'),
)


def run_script(path: str, terminal: 'QBasicTerminal') -> None:
    """Run a .qb script file (multi-line DEF blocks included), or a QBASIC
    source file (.bas), which is numbered and RUN.

    After loading all lines, auto-runs the program if it contains
    numbered lines with a MEASURE statement.
    """
    if path.lower().endswith('.bas'):
        # QBASIC source runs from the top, as RUN does in QBASIC.
        terminal._load_bas(path)
        terminal.cmd_run()
        return
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = [l.rstrip('\n\r') for l in f.readlines()]
    ProgramMgmtMixin._load_lines_with_defs(
        lines, lambda line: terminal.process(line, track_undo=False))

    # Auto-run if the program has a reachable MEASURE (incl. in subs / IF
    # clauses), unless the script already issued an explicit RUN (which would
    # otherwise execute and print the program twice).
    explicit_run = any(l.strip().upper() == 'RUN' for l in lines)
    if (terminal.program and not explicit_run
            and terminal._program_has_measure(sorted(terminal.program))):
        terminal.cmd_run()


def main() -> None:
    import json as _json
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

    from qubasic_core import __version__

    args = sys.argv[1:]
    quiet = '--quiet' in args or '-q' in args
    json_mode = '--json' in args
    agent_mode = '--agent' in args
    spec_mode = '--spec' in args
    web_mode = '--web' in args
    kernel_install = '--install-kernel' in args
    seed_val = None
    for flag in ('--quiet', '-q', '--json', '--agent', '--spec',
                 '--web', '--install-kernel'):
        args = [a for a in args if a != flag]
    # Parse --seed N
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == '--seed' and i + 1 < len(args):
            seed_val = int(args[i + 1])
            i += 2
        else:
            filtered.append(args[i])
            i += 1
    args = filtered

    if any(a in ('-v', '--version') for a in args):
        print(f"QUBASIC {__version__}")
        sys.exit(0)

    if any(a in ('-h', '--help') for a in args):
        print(f"QUBASIC {__version__} — Quantum BASIC Interactive Terminal")
        print()
        print("Usage:")
        for invocation, effect in USAGE:
            print(f"  {invocation:<26}{effect}")
        print()
        print("Type HELP inside the REPL for full command reference.")
        sys.exit(0)

    if kernel_install:
        try:
            from qubasic_core.jupyter_kernel import install_kernelspec
            dest = install_kernelspec()
            print(f"QUBASIC kernelspec installed: {dest}")
            print("Open Jupyter and pick the 'QUBASIC' kernel.")
            sys.exit(0)
        except ImportError:
            print("?--install-kernel needs jupyter_client + ipykernel "
                  "(pip install qubasic[jupyter])")
            sys.exit(1)

    if web_mode:
        port = 8811
        if args and args[0].isdigit():
            port = int(args[0])
        serve(port)
        sys.exit(0)

    if spec_mode:
        # Machine-readable contract for agents: commands, gates, functions,
        # constants, and conventions for the installed version.
        from qubasic_core.expression import ExpressionMixin
        from qubasic_core.parser import GATE_TABLE

        def _doc_parts(mname: str) -> tuple[str, str]:
            doc = (getattr(getattr(QBasicTerminal, mname, None), '__doc__', '') or '').strip()
            line = doc.split('\n')[0].strip()
            # Docstrings format the first line as "SIGNATURE — description".
            for sep in ('—', ' - '):
                if sep in line:
                    sig, _, hlp = line.partition(sep)
                    return sig.strip(), hlp.strip()
            return '', line

        cmds = []
        for tbl, takes in ((QBasicTerminal._CMD_WITH_ARG, True),
                           (QBasicTerminal._CMD_NO_ARG, False)):
            for cname, mname in tbl.items():
                sig, hlp = _doc_parts(mname)
                cmds.append({'name': cname, 'takes_arg': takes,
                             'signature': sig or cname, 'help': hlp})
        from qubasic_core.parser import FUNCTIONS, STATEMENTS
        statements = [{'name': n, 'topic': t, 'signature': s, 'help': h}
                      for (t, n, s, h) in STATEMENTS]
        functions = [{'name': n, 'topic': t, 'signature': s, 'help': h}
                     for (t, n, s, h) in FUNCTIONS]
        spec = {
            'name': 'qubasic',
            'version': __version__,
            'bit_order': 'little-endian (qubit 0 = rightmost bit)',
            'true_value': -1,
            'commands': sorted(cmds, key=lambda c: c['name']),
            'statements': statements,
            'gates': sorted(GATE_TABLE.keys()),
            'functions': functions,
            'constants': sorted({k.upper() for k in ExpressionMixin._SAFE_CONSTS}),
        }
        print(_json.dumps(spec, indent=2))
        sys.exit(0)

    term = QBasicTerminal()
    # JSON mode implies agent use, so confine file writes to the working dir.
    term.agent_mode = agent_mode or json_mode
    if seed_val is not None:
        import numpy as _np
        term._seed = seed_val
        _np.random.seed(seed_val)

    if args:
        path = args[0]
        if not os.path.isfile(path):
            if json_mode:
                print(_json.dumps({'error': f'FILE NOT FOUND: {path}'}, indent=2))
            else:
                print(f"?FILE NOT FOUND: {path}")
            sys.exit(1)
        if quiet or json_mode:
            import io
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            err = None
            try:
                run_script(path, term)
            except Exception as e:  # surface as structured error in JSON mode
                err = str(e)
            finally:
                sys.stdout = old
            if json_mode:
                if err is not None:
                    print(_json.dumps({'error': err}, indent=2))
                    sys.exit(1)
                print(_json.dumps(term.result(), indent=2))
            else:
                # --quiet (no --json): the banner was never emitted into the
                # buffer (only the non-captured path calls print_banner), so the
                # captured text is exactly the results. Print them, matching the
                # documented "suppress banner, output results only" behavior.
                print(buf.getvalue(), end='')
                if err is not None:
                    print(f"?ERROR: {err}")
                    sys.exit(1)
        else:
            term.print_banner()
            run_script(path, term)
        # Exit 0 on success; 1 if a measured program produced no counts.
        expects_measure = bool(term.program) and term._program_has_measure(sorted(term.program))
        sys.exit(0 if term.last_counts is not None or not expects_measure else 1)
    else:
        term.repl()


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# QUBASIC web REPL — a single-page terminal served by the standard library.
#
#     qubasic --web            Serve on http://127.0.0.1:8811/?t=<token>
#     qubasic --web 9000       Custom port
#     QUBASIC_WEB_HOST=0.0.0.0 qubasic --web    Expose on the LAN
#
# One QBasicTerminal per server (a session, like the REPL). The page POSTs each
# line to /run and appends the output. A random URL token gates every request;
# the server binds to localhost unless QUBASIC_WEB_HOST says otherwise. No
# external assets, no dependencies beyond the standard library.
# ---------------------------------------------------------------------------

# Largest request body accepted (one REPL line plus JSON framing).
MAX_BODY = 64 * 1024


class BufferIO:
    """IOPort collecting one request's output. There is no keyboard behind a
    web request, so INPUT reads end-of-input (INPUT then assigns 0)."""

    def __init__(self) -> None:
        self.buf = io.StringIO()

    def write(self, text: str) -> None:
        self.buf.write(text)

    def writeln(self, text: str) -> None:
        self.buf.write(text + '\n')

    def read_line(self, prompt: str) -> str:
        self.buf.write(prompt)
        raise EOFError


def run_line(term: QBasicTerminal, line: str, port: BufferIO) -> str:
    """Process one line on `term` with its output captured through `port`
    (and stdout, for the few paths that print directly)."""
    old_io, old_stdout = term.io, sys.stdout
    term.io = port
    sys.stdout = port.buf
    try:
        term.process(line, track_undo=False)
    except Exception as e:
        port.writeln(f"?ERROR: {e}")
    finally:
        term.io, sys.stdout = old_io, old_stdout
    return port.buf.getvalue()


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QUBASIC</title>
<style>
  body { background:#0a0a12; color:#c8f0c8; font:14px/1.45 Consolas, monospace;
         margin:0; display:flex; flex-direction:column; height:100vh; }
  #out { flex:1; overflow-y:auto; padding:12px; white-space:pre-wrap;
         word-break:break-all; }
  #bar { display:flex; border-top:1px solid #2a2a3a; }
  #in  { flex:1; background:#11111c; color:#e8ffe8; border:0; padding:10px 12px;
         font:inherit; outline:none; }
  .cmd { color:#7ec8ff; }
  .err { color:#ff8080; }
</style></head><body>
<div id="out"></div>
<div id="bar"><input id="in" autofocus autocomplete="off"
     placeholder="] 10 H 0   (Enter to run; HELP for commands)"></div>
<script>
const out = document.getElementById('out'), inp = document.getElementById('in');
const token = new URLSearchParams(location.search).get('t') || '';
const hist = []; let hi = 0;
function append(text, cls) {
  const d = document.createElement('div');
  if (cls) d.className = cls;
  d.textContent = text;
  out.appendChild(d); out.scrollTop = out.scrollHeight;
}
append('QUBASIC web REPL — type HELP for commands, DEMO LIST for demos.');
inp.addEventListener('keydown', async (e) => {
  if (e.key === 'ArrowUp')   { if (hi > 0) inp.value = hist[--hi]; return; }
  if (e.key === 'ArrowDown') { inp.value = (hi < hist.length - 1) ? hist[++hi] : ''; return; }
  if (e.key !== 'Enter') return;
  const line = inp.value; inp.value = '';
  if (!line.trim()) return;
  hist.push(line); hi = hist.length;
  append('] ' + line, 'cmd');
  try {
    const r = await fetch('/run?t=' + encodeURIComponent(token), {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({line})});
    const j = await r.json();
    if (j.output) append(j.output.replace(/\\n$/, ''));
  } catch (err) { append('?CONNECTION ERROR: ' + err, 'err'); }
});
</script></body></html>"""


def make_server(port: int = 8811, host: str | None = None) -> tuple[ThreadingHTTPServer, str]:
    """(server, token): a ready HTTP server with its own terminal. Port 0
    picks a free port (server.server_address has it)."""
    from qubasic_core.terminal import QBasicTerminal

    term = QBasicTerminal()
    term.agent_mode = True                     # confine file writes to the cwd
    token = secrets.token_urlsafe(12)
    host = host or os.environ.get('QUBASIC_WEB_HOST', '127.0.0.1')
    # One terminal serves every request; requests run one at a time.
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _authed(self) -> bool:
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            return secrets.compare_digest(q.get('t', [''])[0], token)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if not self._authed():
                self._send(403, b'forbidden (token required)', 'text/plain')
                return
            self._send(200, _PAGE.encode('utf-8'), 'text/html; charset=utf-8')

        def do_POST(self) -> None:
            if not self._authed():
                self._send(403, b'{}', 'application/json')
                return
            try:
                length = int(self.headers.get('Content-Length', 0) or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                self._send(413, b'{}', 'application/json')
                return
            try:
                payload = json.loads(self.rfile.read(length) or b'{}')
                line = str(payload.get('line', ''))[:4096]
            except Exception:
                self._send(400, b'{}', 'application/json')
                return
            with lock:
                output = run_line(term, line, BufferIO())
            body = json.dumps({'output': output}).encode('utf-8')
            self._send(200, body, 'application/json')

        def log_message(self, *args: Any) -> None:                       # quiet access log
            pass

    return ThreadingHTTPServer((host, port), Handler), token


def serve(port: int = 8811) -> None:
    from qubasic_core import __version__
    httpd, token = make_server(port)
    host = httpd.server_address[0]
    if isinstance(host, bytes):
        host = host.decode()
    print(f"QUBASIC {__version__} web REPL: http://{host}:{port}/?t={token}")
    print("(Ctrl+C to stop; set QUBASIC_WEB_HOST=0.0.0.0 to expose on the LAN)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBYE")
