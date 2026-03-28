"""Pytest configuration — provides fast mock for non-simulation tests."""

import sys

import pytest


class BufferIOPort:
    """IOPort that captures output for testing."""
    def __init__(self):
        self.output = []

    def write(self, text: str) -> None:
        self.output.append(text)

    def writeln(self, text: str) -> None:
        self.output.append(text + '\n')

    def read_line(self, prompt: str) -> str:
        raise EOFError("BufferIOPort does not support input")

    def get_output(self) -> str:
        return ''.join(self.output)

    def clear(self) -> None:
        self.output.clear()


@pytest.fixture
def fast_terminal():
    """QBasicTerminal with minimal config for fast non-simulation tests."""
    from qbasic_core.terminal import QBasicTerminal
    t = QBasicTerminal()
    t.num_qubits = 2
    t.shots = 10
    return t


@pytest.fixture
def buffer_terminal():
    """QBasicTerminal with captured output."""
    from qbasic_core.terminal import QBasicTerminal
    from qbasic_core.mock_backend import MockAerSimulator
    buf = BufferIOPort()
    t = QBasicTerminal()
    t.io = buf
    t.num_qubits = 2
    t.shots = 10
    return t, buf


@pytest.fixture
def mock_aer(monkeypatch):
    """Patch AerSimulator with a fast mock — use for non-quantum tests."""
    from qbasic_core.mock_backend import patch_aer
    patch_aer(monkeypatch)


@pytest.fixture(autouse=True)
def _auto_mock_for_cures(request, monkeypatch):
    """Auto-apply mock AerSimulator for test_cures.py tests."""
    if 'test_cures' in request.node.nodeid:
        from qbasic_core.mock_backend import patch_aer
        patch_aer(monkeypatch)


def capture(func, *args, **kwargs):
    """Capture stdout from a function call, return (result, output_str)."""
    import io as _io
    buf = _io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old
    return result, buf.getvalue()
