"""Encoded Steane simulation (LQUBITS ... CODE STEANE) and the exact LINDBLAD path."""

import itertools
import unittest

import numpy as np

from qubasic_core.logical import STEANE_CHECKS, steane_decode
from qubasic_core.terminal import QBasicTerminal


class _IO:
    def __init__(self):
        self.parts = []

    def write(self, text):
        self.parts.append(text)

    def writeln(self, text):
        self.parts.append(text + '\n')

    def read_line(self, prompt):
        raise EOFError

    @property
    def text(self):
        return ''.join(self.parts)


def steane_run(lines, p, shots=1000, n=2):
    t = QBasicTerminal()
    t.io = _IO()
    t.shots = shots
    t._seed = 5
    t.cmd_lqubits(f'{n} CODE STEANE PHYS {p}')
    for line in lines:
        t.process(line, track_undo=False)
    t.cmd_run()
    return t


class TestSteaneEncoded(unittest.TestCase):

    def test_decoder_corrects_every_single_flip(self):
        zero = set()
        for bits in itertools.product((0, 1), repeat=3):
            w = [0] * 7
            for bit, check in zip(bits, STEANE_CHECKS):
                if bit:
                    for i in check:
                        w[i] ^= 1
            zero.add(tuple(w))
        for w in zero:
            for logical in (0, 1):
                word = [x ^ logical for x in w]
                self.assertEqual(steane_decode(''.join(map(str, word))), logical)
                for i in range(7):
                    flipped = word.copy()
                    flipped[i] ^= 1
                    self.assertEqual(steane_decode(''.join(map(str, flipped))), logical)

    def test_noiseless_bell_pair(self):
        t = steane_run(['10 H 0', '20 CX 0, 1', '30 MEASURE'], 0.0)
        self.assertEqual(set(t.last_counts), {'00', '11'})
        self.assertIn('15 physical qubits', t.io.text)

    def test_logical_s_is_transversal_sdg(self):
        # H S S H = X on logical qubit 1.
        t = steane_run(['10 H 1', '20 S 1', '30 S 1', '40 H 1', '50 MEASURE'], 0.0)
        self.assertEqual(t.last_counts, {'10': 1000})

    def test_cy_and_sx(self):
        t = steane_run(['10 X 0', '20 SX 1', '30 SX 1', '40 CY 0, 1', '50 MEASURE'], 0.0)
        self.assertEqual(t.last_counts, {'01': 1000})

    def test_noise_is_corrected_below_threshold(self):
        t = steane_run(['10 H 0', '20 CX 0, 1', '30 MEASURE'], 0.003, shots=1500)
        self.assertLess(t.variables['_LOGICAL_TVD'], 0.05)

    def test_non_clifford_is_refused(self):
        t = steane_run(['10 T 0', '20 MEASURE'], 0.0)
        self.assertIn('no transversal Steane form', t.io.text)
        self.assertIsNone(t.last_counts)

    def test_off_restores_physical_mode(self):
        t = steane_run(['10 X 0', '20 MEASURE'], 0.0)
        t.cmd_lqubits('OFF')
        t.cmd_run()
        self.assertNotIn('Steane', t.io.text.split('LQUBITS OFF')[-1])


class TestLindbladExact(unittest.TestCase):

    def test_matches_superoperator_exponential(self):
        from scipy.linalg import expm
        t = QBasicTerminal()
        t.io = _IO()
        t.num_qubits = 2
        t.process('10 H 0', track_undo=False)
        t.process('20 CX 0, 1', track_undo=False)
        t.cmd_run()
        t.cmd_lindblad('NONE, 0.7, 200, 0.4 Z 0; 0.3 X 1; 0.2 SP 0; 0.5 N 1')
        got = [t.variables[f'_RHO{i}'] for i in range(4)]
        eye = np.eye(2)
        ops = {'Z': np.diag([1, -1]), 'X': np.array([[0, 1], [1, 0]]),
               'SP': np.array([[0, 0], [1, 0]]), 'N': np.diag([0, 1])}

        def emb(op, q):
            return np.kron(eye, op) if q == 0 else np.kron(op, eye)
        jumps = [np.sqrt(0.4) * emb(ops['Z'], 0), np.sqrt(0.3) * emb(ops['X'], 1),
                 np.sqrt(0.2) * emb(ops['SP'], 0), np.sqrt(0.5) * emb(ops['N'], 1)]
        gen = sum(np.kron(L, L.conj()) - 0.5 * np.kron(L.conj().T @ L, np.eye(4))
                  - 0.5 * np.kron(np.eye(4), (L.conj().T @ L).T) for L in jumps)
        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = (expm(gen * 0.7) @ np.outer(psi, psi).reshape(-1)).reshape(4, 4)
        np.testing.assert_allclose(got, np.real(np.diag(rho)), atol=1e-9)

    def test_eight_qubits_run_exactly(self):
        t = QBasicTerminal()
        t.io = _IO()
        t.num_qubits = 8
        t.cmd_hamiltonian('H = ISING 1 0.5')
        t.cmd_lindblad('H, 0.2, 10, 0.3 SM 0')
        self.assertNotIn('?LINDBLAD', t.io.text)
        self.assertAlmostEqual(sum(t.variables[f'_RHO{i}'] for i in range(256)), 1.0, places=9)


if __name__ == '__main__':
    unittest.main()
