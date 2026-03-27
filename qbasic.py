#!/usr/bin/env python3
"""
QBASIC — Quantum BASIC Interactive Terminal

Usage:
    python qbasic.py              Interactive REPL
    python qbasic.py script.qb    Run a script file
"""

import sys
import os

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

from qbasic_core.terminal import QBasicTerminal
from qbasic_core.program_mgmt import ProgramMgmtMixin


def run_script(path: str, terminal: 'QBasicTerminal') -> None:
    """Run a .qb script file. Supports multi-line DEF blocks.

    After loading all lines, auto-runs the program if it contains
    numbered lines with a MEASURE statement.
    """
    with open(path, 'r') as f:
        lines = [l.rstrip('\n\r') for l in f.readlines()]
    ProgramMgmtMixin._load_lines_with_defs(
        lines, lambda line: terminal.process(line, track_undo=False))

    # Auto-run if the program has a MEASURE statement
    has_measure = any(
        terminal.program.get(ln, '').strip().upper() == 'MEASURE'
        for ln in terminal.program
    )
    if terminal.program and has_measure:
        terminal.cmd_run()


def main():
    import json as _json
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

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
