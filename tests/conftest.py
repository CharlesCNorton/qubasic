"""Pytest configuration and the helpers the test modules share."""

import io
import os
import sys
import tempfile
import unittest

# Runs are sized against 2 GB however much memory the machine has, so the
# suite behaves the same everywhere and leaves the rest of the machine free.
os.environ.setdefault('QUBASIC_MEMORY_LIMIT', '2GB')

from qubasic_core import api  # noqa: E402
from qubasic_core.api import CaptureIO  # noqa: E402
from qubasic_core.terminal import QBasicTerminal  # noqa: E402


class ScriptIO:
    """Collects output and answers INPUT from a list, recording each prompt."""

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
    def reads(self):
        return len(self.prompts)

    @property
    def text(self):
        return ''.join(self.parts)


def term(qubits=2, seed=5, inputs=(), shots=None, agent=False):
    """A terminal writing to a CaptureIO, which echoes INPUT answers from `inputs`."""
    t = QBasicTerminal()
    t.io = CaptureIO(inputs)
    t.num_qubits = qubits
    t._seed = seed
    if shots is not None:
        t.shots = shots
    t.agent_mode = agent
    return t


def feed(t, *lines):
    """Enter each line as typed; return what the terminal wrote."""
    start = len(t.io.parts)
    for line in lines:
        t.process(line, track_undo=False)
    return ''.join(t.io.parts[start:])


def run_basic(source, **settings):
    """Run QBASIC source through the Python API; return (output, error lines)."""
    r = api.run(source, basic=True, **settings)
    return r['output'], r['errors']


def capture(func, *args, **kwargs):
    """Call func with stdout captured; return (result, what it printed)."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old
    return result, buf.getvalue()


class InTempDir(unittest.TestCase):
    """Each test runs in its own temporary working directory."""

    def setUp(self):
        self._cwd = os.getcwd()
        self._dir = tempfile.TemporaryDirectory()
        os.chdir(self._dir.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._dir.cleanup()
