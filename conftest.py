"""Pytest configuration — provides fast mock for non-simulation tests."""

import pytest


@pytest.fixture
def fast_terminal():
    """QBasicTerminal with minimal config for fast non-simulation tests."""
    from qbasic_core.terminal import QBasicTerminal
    t = QBasicTerminal()
    t.num_qubits = 2
    t.shots = 10
    return t


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
