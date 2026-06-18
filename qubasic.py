#!/usr/bin/env python3
"""
QUBASIC — Quantum BASIC Interactive Terminal

Usage:
    python qubasic.py             Interactive REPL
    python qubasic.py script.qb   Run a script file
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

from qubasic_core.terminal import QBasicTerminal
from qubasic_core.program_mgmt import ProgramMgmtMixin


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

    from qubasic_core import __version__

    args = sys.argv[1:]
    quiet = '--quiet' in args or '-q' in args
    json_mode = '--json' in args
    agent_mode = '--agent' in args
    seed_val = None
    for flag in ('--quiet', '-q', '--json', '--agent'):
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
        print("  qubasic                  Interactive REPL")
        print("  qubasic script.qb        Run a script file")
        print("  qubasic --quiet script    Suppress banner and progress")
        print("  qubasic --json script     Output results as JSON")
        print("  qubasic --agent script    Confine file writes to the working dir")
        print("  qubasic --seed N script   Set random seed for reproducibility")
        print("  qubasic --version         Show version")
        print("  qubasic --help            Show this help")
        print()
        print("Type HELP inside the REPL for full command reference.")
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
            elif not quiet:
                print(buf.getvalue())
                if err is not None:
                    print(f"?ERROR: {err}")
                    sys.exit(1)
        else:
            term.print_banner()
            run_script(path, term)
        # Exit 0 on success; 1 if a measured program produced no counts.
        sys.exit(0 if term.last_counts is not None or not any(
            term.program.get(ln, '').strip().upper() == 'MEASURE'
            for ln in term.program) else 1)
    else:
        term.repl()


if __name__ == '__main__':
    main()
