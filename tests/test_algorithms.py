"""Algorithms: general HHL, Shor with modular arithmetic, QAOA, VQE, Grover, Simon."""

import unittest
from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qubasic_core import shor
from qubasic_core.terminal import QBasicTerminal
from qubasic_core.variational import graph_edges


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


def term(seed=7):
    t = QBasicTerminal()
    t.io = _IO()
    t._seed = seed
    return t


def basis_value(qc, regs):
    sv = Statevector(qc).data
    idx = int(np.argmax(np.abs(sv)))
    assert abs(abs(sv[idx]) - 1) < 1e-6
    return {name: sum(((idx >> q) & 1) << i for i, q in enumerate(qs)) for name, qs in regs.items()}


def prep(qc, qs, v):
    for i, q in enumerate(qs):
        if (v >> i) & 1:
            qc.x(q)


class TestHHL(unittest.TestCase):

    def check(self, args, min_fid=0.995):
        t = term()
        t.cmd_hhl(args)
        self.assertNotIn('?HHL', t.io.text, t.io.text)
        self.assertGreater(t.variables['_HHL_FIDELITY'], min_fid)
        return t

    def test_two_by_two_forms(self):
        self.check('1 0 0 2 1 1')
        self.check('[[2,1],[1,3]] [1,0]')

    def test_padded_and_larger_systems(self):
        t = self.check('[[3,1,0],[1,2,0.5],[0,0.5,4]] [1,2,3]')
        self.assertIn('3x3 system', t.io.text)
        self.check('[[4,1,0,0],[1,3,1,0],[0,1,2,0.5],[0,0,0.5,5]] [1,0,1,0] CLOCK 9')

    def test_non_hermitian_by_dilation(self):
        t = self.check('[[1,2],[3,4]] [1,1]')
        self.assertIn('Hermitian dilation', t.io.text)
        x = np.array(t.arrays['_HHL_X'])
        cl = np.linalg.solve(np.array([[1, 2], [3, 4]]), [1, 1])
        self.assertAlmostEqual(abs(np.vdot(cl / np.linalg.norm(cl), x)) ** 2, 1, places=3)

    def test_shots_sample_the_solution(self):
        t = self.check('[[2,0],[0,1]] [1,1] SHOTS 4000')
        self.assertIn('passed post-selection', t.io.text)

    def test_errors(self):
        for args in ('[[1,1],[1,1]] [1,0]', '[[1,0],[0,1]] [1,0,0]', '[[1,0],[0,2]] [0,0]',
                     '1 2 3', '[[1,0],[0,2]] [1,1] CLOCK 2'):
            t = term()
            t.cmd_hhl(args)
            self.assertIn('?HHL ERROR', t.io.text, args)


class TestShorArithmetic(unittest.TestCase):

    def test_fourier_adder(self):
        for a in range(8):
            for b in range(8):
                qc = QuantumCircuit(3)
                prep(qc, [0, 1, 2], b)
                shor.qft(qc, [0, 1, 2])
                shor.phi_add(qc, a, [0, 1, 2])
                shor.qft(qc, [0, 1, 2], inverse=True)
                self.assertEqual(basis_value(qc, {'b': [0, 1, 2]})['b'], (a + b) % 8)

    def test_modular_adder_and_inverse(self):
        N, n = 11, 4
        bq, anc, ctl = list(range(n + 1)), n + 1, [n + 2, n + 3]
        for a in range(N):
            for b in range(N):
                for c1, c2 in ((1, 1), (1, 0)):
                    qc = QuantumCircuit(n + 4)
                    prep(qc, bq, b)
                    if c1:
                        qc.x(ctl[0])
                    if c2:
                        qc.x(ctl[1])
                    shor.qft(qc, bq)
                    shor.phi_add_mod(qc, a, N, bq, anc, ctl)
                    shor.qft(qc, bq, inverse=True)
                    r = basis_value(qc, {'b': bq, 'anc': [anc]})
                    self.assertEqual(r['b'], (a + b) % N if c1 and c2 else b)
                    self.assertEqual(r['anc'], 0)
        qc = QuantumCircuit(n + 4)
        prep(qc, bq, 7)
        qc.x(ctl[0])
        qc.x(ctl[1])
        shor.qft(qc, bq)
        shor.phi_add_mod(qc, 9, N, bq, anc, ctl)
        shor.phi_add_mod(qc, 9, N, bq, anc, ctl, inverse=True)
        shor.qft(qc, bq, inverse=True)
        self.assertEqual(basis_value(qc, {'b': bq})['b'], 7)

    def test_controlled_modular_multiplier(self):
        N, n = 15, 4
        L = shor.layout(n)
        for a in (2, 7, 11):
            for x in range(1, N):
                for c in (0, 1):
                    qc = QuantumCircuit(L.total)
                    if c:
                        qc.x(L.c)
                    prep(qc, L.x, x)
                    shor.controlled_modmul(qc, a, N, L.c, L.x, L.b, L.anc)
                    r = basis_value(qc, {'x': L.x, 'b': L.b, 'anc': [L.anc]})
                    self.assertEqual((r['x'], r['b'], r['anc']), ((a * x) % N if c else x, 0, 0))

    def test_classical_helpers(self):
        self.assertEqual(shor.phase_value('0010', 4), Fraction(2, 16))
        self.assertEqual(shor.order_from_phase(Fraction(1, 4), 7, 15), 4)
        self.assertIsNone(shor.order_from_phase(Fraction(0), 7, 15))
        self.assertEqual(shor.factors_from_order(7, 4, 15), (3, 5))
        self.assertIsNone(shor.factors_from_order(4, 3, 21))
        self.assertEqual(shor.classical_shortcut(22), ('even', (2, 11)))
        self.assertEqual(shor.classical_shortcut(17), ('prime', None))
        self.assertEqual(shor.classical_shortcut(49), ('power', (7, 7)))
        self.assertEqual(shor.classical_shortcut(15), ('', None))


class TestShorCommand(unittest.TestCase):

    def test_factor_15_with_the_circuit(self):
        t = term(seed=11)
        t.cmd_shor('15 7')
        self.assertIn('2n+3', t.io.text)
        self.assertEqual(t.variables['_SHOR_ORDER'], 4)
        self.assertEqual(t.variables['_SHOR_FACTOR'], 3)

    def test_oracle_mode_and_shortcuts(self):
        t = term(seed=11)
        t.cmd_shor('15 7 ORACLE')
        self.assertIn('permutation-oracle', t.io.text)
        self.assertEqual(t.variables['_SHOR_FACTOR'], 3)
        for n, want in (('17', 'prime'), ('22', 'even'), ('49', 'prime power')):
            t = term()
            t.cmd_shor(n)
            self.assertIn(want, t.io.text)

    def test_limits(self):
        t = term()
        t.cmd_shor('299 3')
        self.assertIn('?SHOR', t.io.text)
        t = term()
        t.cmd_shor('')
        self.assertIn('USAGE', t.io.text)


class TestQAOA(unittest.TestCase):

    def test_graph_parsing(self):
        rng = np.random.default_rng(1)
        self.assertEqual(graph_edges('0-1, 1-2:0.5', rng), ([(0, 1, 1.0), (1, 2, 0.5)], 3))
        self.assertEqual(len(graph_edges('RING 5', rng)[0]), 5)
        self.assertEqual(len(graph_edges('COMPLETE 4', rng)[0]), 6)
        for bad in ('0-0', 'x', ''):
            with self.assertRaises(ValueError):
                graph_edges(bad, rng)

    def test_maxcut(self):
        t = term()
        t.dispatch('QAOA RING 6 P 2')
        self.assertGreater(t.variables['_QAOA_RATIO'], 0.8)
        self.assertGreater(t.variables['_QAOA_POPT'], 0.3)
        t = term()
        t.dispatch('QAOA MAXCUT 0-1, 1-2, 2-0 P 1 METHOD NM')
        self.assertGreater(t.variables['_QAOA_RATIO'], 0.75)

    def test_hamiltonian_and_program(self):
        t = term()
        t.num_qubits = 3
        t.dispatch('HAMILTONIAN HZ = 1.0 ZZ 0 1 + 1.0 ZZ 1 2 + 0.3 Z 0')
        t.dispatch('QAOA HZ P 2 PROGRAM')
        self.assertIn('PROGRAM:', t.io.text)
        self.assertLess(t.variables['_QAOA_EXPECT'], -1.0)
        t.shots = 2000
        t.cmd_run()
        self.assertEqual(max(t.last_counts, key=t.last_counts.get), '101')

    def test_non_diagonal_rejected(self):
        t = term()
        t.num_qubits = 2
        t.dispatch('HAMILTONIAN HX = 1.0 X 0')
        t.dispatch('QAOA HX')
        self.assertIn('diagonal', t.io.text)


class TestVQE(unittest.TestCase):

    def test_h2_reaches_fci(self):
        t = term()
        t.dispatch('HAMILTONIAN H2 = MOLECULE H2 0.7414')
        t.dispatch('VQE H2 LAYERS 2 PROGRAM')
        self.assertAlmostEqual(t.variables['_VQE_ENERGY'], -1.137270, places=4)
        self.assertAlmostEqual(t.variables['_VQE_EXACT'], -1.137270, places=4)
        t.cmd_run()
        self.assertAlmostEqual(t.variables['energy'], t.variables['_VQE_ENERGY'], places=4)

    def test_errors(self):
        t = term()
        t.dispatch('VQE NOPE')
        self.assertIn('unknown HAMILTONIAN', t.io.text)
        t.num_qubits = 2
        t.dispatch('HAMILTONIAN H = 1.0 ZZ 0 1')
        t.dispatch('VQE H ANSATZ XYZ')
        self.assertIn('ANSATZ', t.io.text)


class TestGroverSimon(unittest.TestCase):

    def test_grover_success(self):
        t = term()
        t.dispatch('GROVER 1011')
        self.assertGreater(t.variables['_GROVER_P'], 0.9)
        self.assertEqual(t.variables['_GROVER_ITERS'], 3)
        t = term()
        t.dispatch('GROVER 101 010')
        self.assertGreater(t.variables['_GROVER_P'], 0.95)
        t = term()
        t.dispatch('GROVER 11 ITERS 0 SHOTS 4000')
        self.assertLess(t.variables['_GROVER_P'], 0.35)

    def test_grover_program_and_endian(self):
        t = term()
        t._endian_big = True
        t.dispatch('GROVER 110 PROGRAM')
        t.shots = 1000
        t.cmd_run()
        top = max(t.last_counts, key=t.last_counts.get)
        self.assertEqual(top, '011')          # internal order of the typed 110

    def test_grover_usage(self):
        for bad in ('', '12', '10 101'):
            t = term()
            t.dispatch(f'GROVER {bad}')
            self.assertTrue(t.io.text.startswith('?'), t.io.text)

    def test_simon(self):
        t = term()
        t.dispatch('SIMON 1101')
        self.assertEqual(t.variables['_SIMON_OK'], -1)
        t = term()
        t.dispatch('SIMON 000')
        self.assertIn('USAGE', t.io.text)


if __name__ == '__main__':
    unittest.main()
