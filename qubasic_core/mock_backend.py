"""Mock Qiskit backend for fast testing.

Produces uniform random counts without running a real simulation.
Use for tests that verify REPL behavior, not quantum correctness.
"""

from __future__ import annotations

import random
from typing import Any
from unittest.mock import MagicMock


class MockResult:
    """Fake Qiskit result that returns uniform counts."""

    def __init__(self, n_qubits: int, shots: int):
        self._n = n_qubits
        self._shots = shots

    def get_counts(self) -> dict[str, int]:
        states = [format(i, f'0{self._n}b') for i in range(2**self._n)]
        counts: dict[str, int] = {}
        remaining = self._shots
        for s in states[:-1]:
            c = random.randint(0, remaining)
            if c > 0:
                counts[s] = c
                remaining -= c
        if remaining > 0:
            counts[states[-1]] = remaining
        return counts

    def get_statevector(self):
        import numpy as np
        dim = 2 ** self._n
        sv = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
        return sv

    def data(self, *args) -> dict:
        """A uniform saved state under the label RUN uses; no other saves."""
        from qiskit.quantum_info import Statevector
        return {'qb_state': Statevector(self.get_statevector())}


class MockAerSimulator:
    """Drop-in replacement for AerSimulator that skips simulation."""

    def __init__(self, **kwargs):
        self._method = kwargs.get('method', 'automatic')
        self.num_qubits = None  # unlimited — satisfies transpiler check

    def run(self, qc, shots=1024, **kwargs):
        n = qc.num_qubits if hasattr(qc, 'num_qubits') else 2
        result = MockResult(n, shots)
        mock_job = MagicMock()
        mock_job.result.return_value = result
        return mock_job


def patch_aer(monkeypatch):
    """Patch AerSimulator with MockAerSimulator and transpile with identity."""
    monkeypatch.setattr('qubasic_core.terminal.AerSimulator', MockAerSimulator)
    monkeypatch.setattr('qubasic_core.terminal.transpile', lambda qc, backend, **kw: qc)
    monkeypatch.setattr('qubasic_core.analysis.AerSimulator', MockAerSimulator)
    monkeypatch.setattr('qubasic_core.analysis.transpile', lambda qc, backend, **kw: qc)
    monkeypatch.setattr('qubasic_core.sweep.AerSimulator', MockAerSimulator)
    monkeypatch.setattr('qubasic_core.sweep.transpile', lambda qc, backend, **kw: qc)
    monkeypatch.setattr('qubasic_core.executor.transpile', lambda qc, backend, **kw: qc)
