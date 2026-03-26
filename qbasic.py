#!/usr/bin/env python3
"""
QBASIC — Quantum BASIC Interactive Terminal

Usage:
    python qbasic.py              Interactive REPL
    python qbasic.py script.qb    Run a script file
"""

import sys
import os
import re

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

from qbasic_core.engine import RE_DEF_BEGIN
from qbasic_core.terminal import QBasicTerminal


def run_script(path: str, terminal: 'QBasicTerminal') -> None:
    """Run a .qb script file. Supports multi-line DEF blocks.

    After loading all lines, auto-runs the program if it contains
    numbered lines with a MEASURE statement.
    """
    with open(path, 'r') as f:
        lines = [l.rstrip('\n\r') for l in f.readlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        # Handle multi-line DEF BEGIN in scripts by collecting
        # the body lines and routing through cmd_def's data path.
        if re.match(r'DEF\s+BEGIN\s+', line, re.IGNORECASE):
            m = RE_DEF_BEGIN.match(line)
            if m:
                name = m.group(1).upper()
                params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
                body = []
                i += 1
                while i < len(lines):
                    bl = lines[i].strip()
                    if bl.upper() in ('DEF END', 'END'):
                        break
                    if bl and not bl.startswith('#'):
                        body.append(bl)
                    i += 1
                # Build a single-line DEF command and route through process()
                # so all validation (e.g. built-in name rejection) applies.
                body_str = ' : '.join(body)
                param_str = f"({', '.join(params)})" if params else ""
                terminal.process(f"DEF {name}{param_str} = {body_str}", track_undo=False)
        else:
            terminal.process(line, track_undo=False)
        i += 1

    # Auto-run if the program has a MEASURE statement
    has_measure = any(
        terminal.program.get(ln, '').strip().upper() == 'MEASURE'
        for ln in terminal.program
    )
    if terminal.program and has_measure:
        terminal.cmd_run()


def main():
    import json as _json

    args = sys.argv[1:]
    quiet = '--quiet' in args or '-q' in args
    json_mode = '--json' in args
    if quiet:
        args = [a for a in args if a not in ('--quiet', '-q')]
    if json_mode:
        args = [a for a in args if a != '--json']

    if any(a in ('-h', '--help') for a in args):
        from qbasic_core import __version__
        print(f"QBASIC {__version__} — Quantum BASIC Interactive Terminal")
        print()
        print("Usage:")
        print("  qbasic                  Interactive REPL")
        print("  qbasic script.qb        Run a script file")
        print("  qbasic --quiet script    Suppress banner and progress")
        print("  qbasic --json script     Output results as JSON")
        print("  qbasic --help            Show this help")
        print()
        print("Type HELP inside the REPL for full command reference.")
        sys.exit(0)

    term = QBasicTerminal()

    if args:
        path = args[0]
        if os.path.isfile(path):
            if quiet or json_mode:
                import io
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    run_script(path, term)
                finally:
                    sys.stdout = old
                if json_mode:
                    result = {
                        'counts': term.last_counts or {},
                        'num_qubits': term.num_qubits,
                        'shots': term.shots,
                    }
                    print(_json.dumps(result, indent=2))
                elif not quiet:
                    print(buf.getvalue())
            else:
                term.print_banner()
                run_script(path, term)
            # Exit code: 0 if results exist, 1 if error
            sys.exit(0 if term.last_counts is not None or not any(
                term.program.get(ln, '').strip().upper() == 'MEASURE'
                for ln in term.program) else 1)
        else:
            print(f"?FILE NOT FOUND: {path}")
            sys.exit(1)
    else:
        term.repl()


if __name__ == '__main__':
    main()
