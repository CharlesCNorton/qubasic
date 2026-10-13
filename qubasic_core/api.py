"""Run QUBASIC programs from Python.

    from qubasic_core import api

    result = api.run('''
    10 H 0
    20 CX 0, 1
    30 MEASURE
    ''', qubits=2, shots=1000, seed=7)
    result['counts']        # {'00': 503, '11': 497}
    result['output']        # what the program printed
    result['errors']        # the ?... lines, if any

`source` is a .qb-style text (numbered program lines plus commands such as
QUBITS, NOISE or DEF) or, when no line is numbered, QBASIC source whose lines
are numbered as LOAD numbers a .bas file. The result is the terminal's
`result()` (counts, variables, the statevector of a small register, method,
depth, gate count) plus the program's output.

A Session keeps one terminal across calls, as the REPL does:

    s = api.Session(qubits=3, seed=1)
    s.execute('10 H 0')
    s.execute('20 MEASURE')
    s.run()['counts']
"""

from __future__ import annotations

import io
import re
import sys
from typing import Any, Iterable

_RE_NUMBERED = re.compile(r'\s*\d+\s')


class CaptureIO:
    """Collects output; INPUT reads from a fixed list of answers, then ends."""

    def __init__(self, inputs: Iterable[str] = ()) -> None:
        self.parts: list[str] = []
        self.inputs = list(inputs)

    def write(self, text: str) -> None:
        self.parts.append(text)

    def writeln(self, text: str) -> None:
        self.parts.append(text + '\n')

    def read_line(self, prompt: str) -> str:
        self.parts.append(prompt)
        if not self.inputs:
            raise EOFError
        answer = str(self.inputs.pop(0))
        self.parts.append(answer + '\n')
        return answer

    @property
    def text(self) -> str:
        return ''.join(self.parts)


class Session:
    """One QUBASIC terminal driven from Python (see the module docstring).

    Keyword settings apply before any input: qubits, shots, seed, method,
    precision and noise (a NOISE specification such as 'depolarizing 0.01').
    `inputs` answers INPUT statements in order. File access stays inside the
    working directory unless agent=False."""

    def __init__(self, *, qubits: int | None = None, shots: int | None = None,
                 seed: int | None = None, method: str | None = None,
                 precision: str | None = None, noise: str | None = None,
                 inputs: Iterable[str] = (), agent: bool = True) -> None:
        from qubasic_core.terminal import QBasicTerminal
        self.terminal = QBasicTerminal()
        self.terminal.agent_mode = agent
        self.io = CaptureIO(inputs)
        self.terminal.io = self.io
        settings = [('METHOD', method), ('PRECISION', precision), ('QUBITS', qubits),
                    ('SHOTS', shots), ('SEED', seed), ('NOISE', noise)]
        for word, value in settings:
            if value is not None:
                self.execute(f"{word} {value}")
        self.io.parts.clear()

    def execute(self, source: str, *, basic: bool = False) -> str:
        """Process the lines of `source` as typed at the prompt (numbered lines
        enter the program, the rest run as commands) and return what they
        printed. basic=True numbers QBASIC source first, as LOAD numbers a
        .bas file."""
        from qubasic_core.editing import ProgramMgmtMixin, number_source
        lines = source.splitlines()
        if basic:
            lines = [f"{n} {text}" for n, text in number_source(lines)]
        start = len(self.io.parts)
        old_stdout, sys.stdout = sys.stdout, io.StringIO()
        try:
            ProgramMgmtMixin._load_lines_with_defs(
                lines, lambda line: self.terminal.process(line, track_undo=False))
            spill = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        if spill:
            self.io.write(spill)
        return ''.join(self.io.parts[start:])

    def run(self) -> dict[str, Any]:
        """RUN the program; the result dict (see run())."""
        return self.result(self.execute('RUN'))

    def result(self, output: str = '') -> dict[str, Any]:
        """The terminal's result() for the last run, with `output` (the text
        given) and `errors` (its ?... lines) added."""
        out = self.terminal.result()
        out['output'] = output
        out['errors'] = [line.strip() for line in output.splitlines()
                         if line.lstrip().startswith('?')]
        return out

    @property
    def counts(self) -> dict[str, int] | None:
        """The last run's histogram, or None before a run that measures."""
        return self.terminal.last_counts

    @property
    def variables(self) -> dict[str, Any]:
        """The program's variables as the last run left them."""
        return self.terminal.variables

    @property
    def state(self) -> Any:
        """The last run's statevector (computed on first read when deferred)."""
        return self.terminal.last_sv


def run(source: str, *, basic: bool | None = None, **settings: Any) -> dict[str, Any]:
    """Load `source`, RUN it and return the result: counts, num_qubits, shots,
    bit_order, variables, statevector (8 qubits or fewer), method (and engine
    for a live run), depth, gates, output and errors. A source with no numbered line is QBASIC source
    (basic=None decides so; True or False forces it). Settings are Session's
    (qubits, shots, seed, method, precision, noise, inputs, agent)."""
    if basic is None:
        lines = [line for line in source.splitlines() if line.strip()]
        basic = bool(lines) and not any(_RE_NUMBERED.match(line) for line in lines)
    session = Session(**settings)
    loaded = session.execute(source, basic=basic)
    if not session.terminal.program:
        return session.result(loaded)
    result = session.run()
    return session.result(loaded + result['output'])
