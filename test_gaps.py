#!/usr/bin/env python3
"""
Gap-coverage tests for QBASIC features.

Covers: noise model, step, bench, monitor, Clifford detection,
circuit cache, and stabilizer fallback.

Run: python -m pytest test_gaps.py -v
"""

import sys
import os
import io
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from qbasic_core.terminal import QBasicTerminal


def capture(func, *args, **kwargs):
    """Capture stdout from a function call, return (result, output_str)."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old
    return result, buf.getvalue()


# =====================================================================
# #18: Noise model tests
# =====================================================================
class TestNoiseModel(unittest.TestCase):
    def test_noise_on_off(self):
        """cmd_noise sets and clears noise model."""
        t = QBasicTerminal()
        # ON
        _, out = capture(t.cmd_noise, 'depolarizing 0.01')
        assert t._noise_model is not None
        assert 'depolarizing' in out.lower()
        # OFF
        _, out = capture(t.cmd_noise, 'OFF')
        assert t._noise_model is None

    def test_noise_types(self):
        """All noise types parse without error."""
        t = QBasicTerminal()
        for ntype in ['depolarizing 0.01', 'amplitude_damping 0.01',
                       'phase_flip 0.01', 'readout 0.05 0.1',
                       'combined 0.01 0.02', 'pauli 0.01 0.01 0.01',
                       'reset 0.01 0.01']:
            _, out = capture(t.cmd_noise, ntype)
            assert 'ERROR' not in out.upper(), f"Failed on: {ntype}"
            t._noise_model = None  # reset

    def test_unknown_noise(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_noise, 'nonexistent 0.01')
        assert 'UNKNOWN' in out


# =====================================================================
# #19: Step test
# =====================================================================
class TestStep(unittest.TestCase):
    def test_step_empty_program(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_step)
        assert 'NOTHING TO STEP' in out


# =====================================================================
# #20: Bench test
# =====================================================================
class TestBench(unittest.TestCase):
    def test_bench_output(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_bench)
        assert 'Benchmark' in out
        assert 'qubits' in out
        # Should have results for multiple qubit counts
        assert '4' in out
        assert '8' in out


# =====================================================================
# #21: Monitor test
# =====================================================================
class TestMonitor(unittest.TestCase):
    def test_monitor_quit(self):
        t = QBasicTerminal()
        import builtins
        orig = builtins.input
        builtins.input = lambda prompt='': 'Q'
        try:
            _, out = capture(t.cmd_monitor)
        finally:
            builtins.input = orig
        assert 'MONITOR' in out


# =====================================================================
# #22: Clifford detection
# =====================================================================
class TestCliffordDetection(unittest.TestCase):
    def test_clifford_circuit(self):
        from qiskit import QuantumCircuit
        t = QBasicTerminal()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert t._is_clifford(qc) is True

    def test_non_clifford_circuit(self):
        from qiskit import QuantumCircuit
        t = QBasicTerminal()
        qc = QuantumCircuit(1)
        qc.rx(0.5, 0)
        assert t._is_clifford(qc) is False


# =====================================================================
# #23: Circuit cache
# =====================================================================
class TestCircuitCache(unittest.TestCase):
    def test_cache_hit(self):
        t = QBasicTerminal()
        t.num_qubits = 2
        t.shots = 10
        t.process('10 H 0', track_undo=False)
        t.process('20 CX 0,1', track_undo=False)
        t.process('30 MEASURE', track_undo=False)
        _, _ = capture(t.cmd_run)
        key1 = t._circuit_cache_key
        # Second run should use cache
        _, _ = capture(t.cmd_run)
        assert t._circuit_cache_key == key1

    def test_cache_invalidation(self):
        t = QBasicTerminal()
        t.num_qubits = 2
        t.shots = 10
        t.process('10 H 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, _ = capture(t.cmd_run)
        key1 = t._circuit_cache_key
        # Modify program
        t.process('15 X 1', track_undo=False)
        _, _ = capture(t.cmd_run)
        assert t._circuit_cache_key != key1


# =====================================================================
# #24: Stabilizer fallback
# =====================================================================
class TestStabilizerFallback(unittest.TestCase):
    def test_non_clifford_auto_method(self):
        """Non-Clifford circuit with auto method should use statevector, not stabilizer."""
        t = QBasicTerminal()
        t.num_qubits = 2
        t.shots = 10
        t.sim_method = 'automatic'
        t.process('10 RX 0.5, 0', track_undo=False)
        t.process('20 MEASURE', track_undo=False)
        _, out = capture(t.cmd_run)
        assert 'ERROR' not in out.upper()


# =====================================================================
# Run
# =====================================================================
if __name__ == '__main__':
    unittest.main()
