"""In-place chunked numpy engine: equivalence with dense linear algebra.

The chunk size is shrunk so small states exercise the split paths that only
large states take in normal use.
"""

import unittest

import numpy as np

from qubasic_core import gates
from qubasic_core.gates import (
    _apply_gate_inplace, _apply_gate_np, _measure_inplace, _sample_counts,
    _marginal_p1, _np_gate_matrix,
)
from qubasic_core.locc_engine import LOCCEngine


def _dense_apply(sv, matrix, qubits, n):
    """Reference: build the full 2^n x 2^n operator (MSB-first over qubits)."""
    k = len(qubits)
    full = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for col in range(2 ** n):
        sub = 0
        for i, q in enumerate(qubits):
            sub |= ((col >> q) & 1) << (k - 1 - i)
        for row_sub in range(2 ** k):
            amp = matrix[row_sub, sub]
            if amp == 0:
                continue
            row = col
            for i, q in enumerate(qubits):
                bit = (row_sub >> (k - 1 - i)) & 1
                row = (row & ~(1 << q)) | (bit << q)
            full[row, col] += amp
    return full @ sv


def _random_state(n, rng):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def _random_unitary(k, rng):
    m = rng.normal(size=(2 ** k, 2 ** k)) + 1j * rng.normal(size=(2 ** k, 2 ** k))
    q, r = np.linalg.qr(m)
    return q * (np.diag(r) / np.abs(np.diag(r)))


class TestChunkedEngine(unittest.TestCase):
    def setUp(self):
        self._saved = gates._CHUNK_AMPS
        gates._CHUNK_AMPS = 4          # force splitting on 6-qubit states

    def tearDown(self):
        gates._CHUNK_AMPS = self._saved

    def test_apply_matches_dense_for_every_target_layout(self):
        rng = np.random.default_rng(7)
        n = 6
        for qubits in ([0], [5], [2], [0, 5], [5, 0], [3, 1], [1, 4, 2], [5, 3, 0]):
            sv = _random_state(n, rng)
            u = _random_unitary(len(qubits), rng)
            ref = _dense_apply(sv, u, qubits, n)
            out = sv.copy()
            _apply_gate_inplace(out, u, qubits, n)
            np.testing.assert_allclose(out, ref, atol=1e-12, err_msg=str(qubits))
            np.testing.assert_allclose(_apply_gate_np(sv, u, qubits, n), ref, atol=1e-12)

    def test_apply_keeps_single_precision(self):
        rng = np.random.default_rng(3)
        sv = _random_state(5, rng).astype(np.complex64)
        _apply_gate_inplace(sv, _np_gate_matrix('H'), [2], 5)
        self.assertEqual(sv.dtype, np.complex64)
        self.assertAlmostEqual(float(np.vdot(sv, sv).real), 1.0, places=5)

    def test_measure_collapses_in_place(self):
        rng = np.random.default_rng(11)
        n = 6
        for q in range(n):
            sv = _random_state(n, rng)
            p1 = _marginal_p1(sv, q, n)
            ref_p1 = sum(abs(sv[i]) ** 2 for i in range(2 ** n) if (i >> q) & 1)
            self.assertAlmostEqual(p1, ref_p1, places=12)
            np.random.seed(q)
            out = _measure_inplace(sv, q, n)
            mask = np.array([((i >> q) & 1) == out for i in range(2 ** n)])
            self.assertTrue(np.allclose(sv[~mask], 0))
            self.assertAlmostEqual(float(np.vdot(sv, sv).real), 1.0, places=12)

    def test_sampling_matches_born_rule_across_chunks(self):
        n = 6
        sv = np.zeros(2 ** n, dtype=complex)
        sv[3], sv[40], sv[63] = np.sqrt(0.5), np.sqrt(0.3), np.sqrt(0.2)
        np.random.seed(1)
        counts = _sample_counts(sv, n, 20000)
        self.assertEqual(set(counts), {format(3, '06b'), format(40, '06b'), format(63, '06b')})
        self.assertAlmostEqual(counts[format(3, '06b')] / 20000, 0.5, delta=0.02)
        self.assertAlmostEqual(counts[format(40, '06b')] / 20000, 0.3, delta=0.02)

    def test_locc_engine_runs_in_place(self):
        eng = LOCCEngine([3, 3], joint=True)
        buf = eng.sv
        eng.apply('A', 'H', (), [0])
        eng.share('A', 2, 'B', 0)
        eng.apply('B', 'CX', (), [0, 1])
        self.assertIs(eng.sv, buf)              # no reallocation
        outcome = eng.send('A', 0)
        self.assertIn(outcome, (0, 1))
        self.assertIs(eng.sv, buf)
        self.assertAlmostEqual(float(np.vdot(eng.sv, eng.sv).real), 1.0, places=12)

    def test_locc_single_precision_dtype(self):
        eng = LOCCEngine([2, 2], precision='single')
        eng.apply('A', 'H', (), [0])
        self.assertEqual(eng.svs['A'].dtype, np.complex64)


if __name__ == '__main__':
    unittest.main()
