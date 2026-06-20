#!/usr/bin/env python3
"""
QUBASIC — command-line entry point.

Usage:
    qubasic                       Interactive REPL (installed console script)
    python -m qubasic_core        Same, run as a module
    qubasic script.qb             Run a script file
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

# Statement-level operations that are parsed inline rather than dispatched as
# REPL commands or gates, so they are absent from the command tables and the
# gate table. Listed in --spec (name, signature, description) for agents.
_SPEC_STATEMENTS = [
    ('MEAS', 'MEAS <qubit> -> <bit>', 'mid-circuit measurement into a classical bit (drives IF feedforward)'),
    ('MEASURE', 'MEASURE [qubit list]', 'measure all qubits, or a subset, into the result histogram'),
    ('MEASURE_X', 'MEASURE_X <qubit>', 'measure in the X basis (result in mx_<q>)'),
    ('MEASURE_Y', 'MEASURE_Y <qubit>', 'measure in the Y basis (result in my_<q>)'),
    ('MEASURE_Z', 'MEASURE_Z <qubit>', 'measure in the Z basis (result in mz_<q>)'),
    ('SYNDROME', 'SYNDROME <paulis> <qubits> -> <var>', 'non-destructive stabilizer measurement via an ancilla'),
    ('RESET', 'RESET <qubit>', 'reset a qubit to |0>'),
    ('BARRIER', 'BARRIER', 'optimization barrier'),
    ('QFT', 'QFT <lo>-<hi>', 'quantum Fourier transform over a qubit range'),
    ('IQFT', 'IQFT <lo>-<hi>', 'inverse quantum Fourier transform'),
    ('DIFFUSE', 'DIFFUSE <lo>-<hi>', 'Grover diffusion operator'),
    ('MCX', 'MCX <ctrl,...>, <target>', 'multi-controlled X'),
    ('MCZ', 'MCZ <ctrl,...>', 'multi-controlled Z'),
    ('MCP', 'MCP <theta>, <ctrl,...>, <target>', 'multi-controlled phase'),
    ('QADD', 'QADD <a-range>, <b-range>', 'in-place register add A += B (mod 2^n)'),
    ('QADDC', 'QADDC <k>, <range>', 'in-place constant add A += k (mod 2^n)'),
    ('QPE', 'QPE <range> <target> <UGATE>', 'quantum phase estimation of a unitary'),
    ('AMPLIFY', 'AMPLIFY <marked>', 'one amplitude-amplification (Grover) step'),
    ('GRAPHSTATE', 'GRAPHSTATE <a-b, b-c, ...>', 'prepare a graph/cluster state'),
    ('FEATUREMAP', 'FEATUREMAP <x0> <x1> ...', 'ZZ feature-map data encoding'),
    ('EVOLVE', 'EVOLVE <H>, <time>, <steps>', 'Trotterized Hamiltonian time evolution'),
    ('APPLYCHANNEL', 'APPLYCHANNEL <name> <qubit>', 'apply a defined Kraus channel'),
    ('SAVE_EXPECT', 'SAVE_EXPECT <obs> <qubits> -> <var>', 'record an expectation value into a variable after RUN'),
    ('SAVE_PROBS', 'SAVE_PROBS <qubits> -> <array>', 'record a probability snapshot into an array after RUN'),
    ('SAVE_AMPS', 'SAVE_AMPS <lo>,<hi> -> <array>', 'record amplitudes into an array after RUN'),
    ('CTRL', 'CTRL <gate> <ctrl>, <target>', 'controlled version of any gate'),
    ('INV', 'INV <gate> <args>', 'inverse/dagger of a gate'),
    ('UNITARY', 'UNITARY <NAME> = [[...]]', 'define a custom gate from a unitary matrix'),
    ('GOTO', 'GOTO <line>', 'jump to a line number'),
    ('GOSUB', 'GOSUB <line>', 'call a line block; RETURN resumes'),
    ('FOR', 'FOR <v> = <a> TO <b> [STEP <s>]', 'counted loop, closed by NEXT'),
    ('WHILE', 'WHILE <cond>', 'pre-test loop, closed by WEND'),
    ('DO', 'DO [WHILE|UNTIL <cond>]', 'loop, closed by LOOP [WHILE|UNTIL]'),
    ('IF', 'IF <cond> THEN <stmt> [ELSE <stmt>]', 'conditional (single-line, or block ending in END IF)'),
    ('SELECT', 'SELECT CASE <expr>', 'multi-way branch (CASE / CASE ELSE / END SELECT)'),
    ('DATA', 'DATA <v1>, <v2>, ...', 'inline data consumed by READ'),
    ('READ', 'READ <var>, ...', 'read the next DATA values'),
    ('DEF', 'DEF <NAME>[(p)] = <gates>', 'define a gate-sequence subroutine (also DEF FN, DEF BEGIN)'),
    ('SUB', 'SUB <name>(<args>)', 'structured subroutine (END SUB; LOCAL/STATIC/SHARED)'),
    ('FUNCTION', 'FUNCTION <name>(<args>)', 'function returning a value (END FUNCTION)'),
    ('DIM', 'DIM <name>(<size>[,...])', 'declare an array (name$ for strings; inclusive sizing)'),
    ('REDIM', 'REDIM [PRESERVE] <name>(<size>)', 'resize an array; PRESERVE keeps existing data'),
]


def run_script(path: str, terminal: 'QBasicTerminal') -> None:
    """Run a .qb script file. Supports multi-line DEF blocks.

    After loading all lines, auto-runs the program if it contains
    numbered lines with a MEASURE statement.
    """
    with open(path, 'r') as f:
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


def main():
    import json as _json
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

    from qubasic_core import __version__

    args = sys.argv[1:]
    quiet = '--quiet' in args or '-q' in args
    json_mode = '--json' in args
    agent_mode = '--agent' in args
    spec_mode = '--spec' in args
    seed_val = None
    for flag in ('--quiet', '-q', '--json', '--agent', '--spec'):
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
        print("  qubasic --quiet script    Suppress banner and progress (also -q)")
        print("  qubasic --json script     Output results as JSON")
        print("  qubasic --agent script    Confine file writes to the working dir")
        print("  qubasic --seed N script   Set random seed for reproducibility")
        print("  qubasic --version         Show version (also -v)")
        print("  qubasic --help            Show this help (also -h)")
        print("  python -m qubasic_core    Run without the installed console script")
        print()
        print("Type HELP inside the REPL for full command reference.")
        sys.exit(0)

    if spec_mode:
        # Machine-readable contract for agents: commands, gates, functions,
        # constants, and conventions for the installed version.
        from qubasic_core.engine import GATE_TABLE
        from qubasic_core.expression import ExpressionMixin

        def _doc_parts(mname):
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
        statements = [{'name': n, 'signature': s, 'help': h}
                      for (n, s, h) in _SPEC_STATEMENTS]
        spec = {
            'name': 'qubasic',
            'version': __version__,
            'bit_order': 'little-endian (qubit 0 = rightmost bit)',
            'true_value': -1,
            'commands': sorted(cmds, key=lambda c: c['name']),
            'statements': sorted(statements, key=lambda s: s['name']),
            'gates': sorted(GATE_TABLE.keys()),
            'functions': sorted(
                set(ExpressionMixin._SAFE_FUNCS)
                | {'RND', 'TIMER', 'POS', 'PEEK', 'USR', 'EOF', 'FRE',
                   'LEFT$', 'RIGHT$', 'MID$', 'CHR$', 'STR$', 'HEX$', 'BIN$',
                   'ASC', 'VAL', 'INSTR', 'LEN'}),
            'constants': sorted(ExpressionMixin._SAFE_CONSTS.keys()),
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
