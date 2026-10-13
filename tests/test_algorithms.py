"""Algorithms against exact answers: HHL, Shor, QAOA, VQE, Grover, Simon,
phase and amplitude estimation, the optimizers and the circuit primitives."""

import math
import unittest
from fractions import Fraction

import numpy as np
from conftest import feed, term
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

from qubasic_core import algorithms
from qubasic_core.algorithms import graph_edges
from qubasic_core.qec import _bp_decode, _osd0

# ---------------------------------------------------------------------------
# Algorithms: general HHL, Shor with modular arithmetic, QAOA, VQE, Grover, Simon.
# ---------------------------------------------------------------------------


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
        t = term(4, seed=7)
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
            t = term(4, seed=7)
            t.cmd_hhl(args)
            self.assertIn('?HHL ERROR', t.io.text, args)


class TestShorArithmetic(unittest.TestCase):

    def test_fourier_adder(self):
        for a in range(8):
            for b in range(8):
                qc = QuantumCircuit(3)
                prep(qc, [0, 1, 2], b)
                algorithms.qft(qc, [0, 1, 2])
                algorithms.phi_add(qc, a, [0, 1, 2])
                algorithms.qft(qc, [0, 1, 2], inverse=True)
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
                    algorithms.qft(qc, bq)
                    algorithms.phi_add_mod(qc, a, N, bq, anc, ctl)
                    algorithms.qft(qc, bq, inverse=True)
                    r = basis_value(qc, {'b': bq, 'anc': [anc]})
                    self.assertEqual(r['b'], (a + b) % N if c1 and c2 else b)
                    self.assertEqual(r['anc'], 0)
        qc = QuantumCircuit(n + 4)
        prep(qc, bq, 7)
        qc.x(ctl[0])
        qc.x(ctl[1])
        algorithms.qft(qc, bq)
        algorithms.phi_add_mod(qc, 9, N, bq, anc, ctl)
        algorithms.phi_add_mod(qc, 9, N, bq, anc, ctl, inverse=True)
        algorithms.qft(qc, bq, inverse=True)
        self.assertEqual(basis_value(qc, {'b': bq})['b'], 7)

    def test_controlled_modular_multiplier(self):
        N, n = 15, 4
        L = algorithms.layout(n)
        for a in (2, 7, 11):
            for x in range(1, N):
                for c in (0, 1):
                    qc = QuantumCircuit(L.total)
                    if c:
                        qc.x(L.c)
                    prep(qc, L.x, x)
                    algorithms.controlled_modmul(qc, a, N, L.c, L.x, L.b, L.anc)
                    r = basis_value(qc, {'x': L.x, 'b': L.b, 'anc': [L.anc]})
                    self.assertEqual((r['x'], r['b'], r['anc']), ((a * x) % N if c else x, 0, 0))

    def test_classical_helpers(self):
        self.assertEqual(algorithms.phase_value('0010', 4), Fraction(2, 16))
        self.assertEqual(algorithms.order_from_phase(Fraction(1, 4), 7, 15), 4)
        self.assertIsNone(algorithms.order_from_phase(Fraction(0), 7, 15))
        self.assertEqual(algorithms.factors_from_order(7, 4, 15), (3, 5))
        self.assertIsNone(algorithms.factors_from_order(4, 3, 21))
        self.assertEqual(algorithms.classical_shortcut(22), ('even', (2, 11)))
        self.assertEqual(algorithms.classical_shortcut(17), ('prime', None))
        self.assertEqual(algorithms.classical_shortcut(49), ('power', (7, 7)))
        self.assertEqual(algorithms.classical_shortcut(15), ('', None))


class TestShorCommand(unittest.TestCase):

    def test_factor_15_with_the_circuit(self):
        t = term(4, seed=11)
        t.cmd_shor('15 7')
        self.assertIn('2n+3', t.io.text)
        self.assertEqual(t.variables['_SHOR_ORDER'], 4)
        self.assertEqual(t.variables['_SHOR_FACTOR'], 3)

    def test_oracle_mode_and_shortcuts(self):
        t = term(4, seed=11)
        t.cmd_shor('15 7 ORACLE')
        self.assertIn('permutation-oracle', t.io.text)
        self.assertEqual(t.variables['_SHOR_FACTOR'], 3)
        for n, want in (('17', 'prime'), ('22', 'even'), ('49', 'prime power')):
            t = term(4, seed=7)
            t.cmd_shor(n)
            self.assertIn(want, t.io.text)

    def test_limits(self):
        t = term(4, seed=7)
        t.cmd_shor('299 3')
        self.assertIn('?SHOR', t.io.text)
        t = term(4, seed=7)
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
        t = term(4, seed=7)
        t.dispatch('QAOA RING 6 P 2')
        self.assertGreater(t.variables['_QAOA_RATIO'], 0.8)
        self.assertGreater(t.variables['_QAOA_POPT'], 0.3)
        t = term(4, seed=7)
        t.dispatch('QAOA MAXCUT 0-1, 1-2, 2-0 P 1 METHOD NM')
        self.assertGreater(t.variables['_QAOA_RATIO'], 0.75)

    def test_hamiltonian_and_program(self):
        t = term(4, seed=7)
        t.num_qubits = 3
        t.dispatch('HAMILTONIAN HZ = 1.0 ZZ 0 1 + 1.0 ZZ 1 2 + 0.3 Z 0')
        t.dispatch('QAOA HZ P 2 PROGRAM')
        self.assertIn('PROGRAM:', t.io.text)
        self.assertLess(t.variables['_QAOA_EXPECT'], -1.0)
        t.shots = 2000
        t.cmd_run()
        self.assertEqual(max(t.last_counts, key=t.last_counts.get), '101')

    def test_non_diagonal_rejected(self):
        t = term(4, seed=7)
        t.num_qubits = 2
        t.dispatch('HAMILTONIAN HX = 1.0 X 0')
        t.dispatch('QAOA HX')
        self.assertIn('diagonal', t.io.text)


class TestVQE(unittest.TestCase):

    def test_h2_reaches_fci(self):
        t = term(4, seed=7)
        t.dispatch('HAMILTONIAN H2 = MOLECULE H2 0.7414')
        t.dispatch('VQE H2 LAYERS 2 PROGRAM')
        self.assertAlmostEqual(t.variables['_VQE_ENERGY'], -1.137270, places=4)
        self.assertAlmostEqual(t.variables['_VQE_EXACT'], -1.137270, places=4)
        t.cmd_run()
        self.assertAlmostEqual(t.variables['energy'], t.variables['_VQE_ENERGY'], places=4)

    def test_errors(self):
        t = term(4, seed=7)
        t.dispatch('VQE NOPE')
        self.assertIn('unknown HAMILTONIAN', t.io.text)
        t.num_qubits = 2
        t.dispatch('HAMILTONIAN H = 1.0 ZZ 0 1')
        t.dispatch('VQE H ANSATZ XYZ')
        self.assertIn('ANSATZ', t.io.text)


class TestGroverSimon(unittest.TestCase):

    def test_grover_success(self):
        t = term(4, seed=7)
        t.dispatch('GROVER 1011')
        self.assertGreater(t.variables['_GROVER_P'], 0.9)
        self.assertEqual(t.variables['_GROVER_ITERS'], 3)
        t = term(4, seed=7)
        t.dispatch('GROVER 101 010')
        self.assertGreater(t.variables['_GROVER_P'], 0.95)
        t = term(4, seed=7)
        t.dispatch('GROVER 11 ITERS 0 SHOTS 4000')
        self.assertLess(t.variables['_GROVER_P'], 0.35)

    def test_grover_program_and_endian(self):
        t = term(4, seed=7)
        t._endian_big = True
        t.dispatch('GROVER 110 PROGRAM')
        t.shots = 1000
        t.cmd_run()
        top = max(t.last_counts, key=t.last_counts.get)
        self.assertEqual(top, '011')          # internal order of the typed 110

    def test_grover_usage(self):
        for bad in ('', '12', '10 101'):
            t = term(4, seed=7)
            t.dispatch(f'GROVER {bad}')
            self.assertTrue(t.io.text.startswith('?'), t.io.text)

    def test_simon(self):
        t = term(4, seed=7)
        t.dispatch('SIMON 1101')
        self.assertEqual(t.variables['_SIMON_OK'], -1)
        t = term(4, seed=7)
        t.dispatch('SIMON 000')
        self.assertIn('USAGE', t.io.text)


class TestPhaseEstimation(unittest.TestCase):

    def test_qpe_distribution_reads_a_known_phase(self):
        from qiskit import QuantumCircuit

        from qubasic_core.algorithms import qpe_distribution
        U = np.diag([1, np.exp(2j * np.pi * 0.375)])
        prep = QuantumCircuit(1)
        prep.x(0)
        probs = qpe_distribution(U, prep, 4)
        self.assertAlmostEqual(probs[6], 1.0)              # 0.375 * 16

    def test_hamiltonian_energy(self):
        t = term(4, seed=3)
        t.dispatch('HAMILTONIAN H2 = MOLECULE H2 0.7414')
        t.dispatch('QPE H2 BITS 8')
        self.assertAlmostEqual(t.variables['_QPE_ENERGY'], t.variables['_QPE_EXACT'], delta=0.01)
        self.assertGreater(t.variables['_QPE_OVERLAP'], 0.95)
        t.dispatch('QPE H2 BITS 5 SHOTS 100 STATE 0011')
        self.assertIn('start |0011>', t.io.text)
        t.dispatch('QPE H2 STATE 01')
        self.assertIn('?QPE ERROR', t.io.text)

    def test_qpe_statement_form_runs_at_the_prompt(self):
        """QPE at the prompt writes the eigenphase into the counting qubits, the
        first one listed holding the most significant bit."""
        for matrix, state in (('[[1, 0], [0, -1]]', '|1001\u27e9'),      # phase 1/2
                              ('[[1, 0], [0, 1j]]', '|1010\u27e9')):     # phase 1/4
            t = term(4, seed=7)
            out = feed(t, f'UNITARY PH = {matrix}', 'X 3 : QPE 0-2 3 PH')
            self.assertIn(f'+1.000{state}', out)

    def test_counting(self):
        t = term(4, seed=7)
        t.dispatch('QCOUNT 0011 0101 1111 BITS 6')
        self.assertEqual(round(t.variables['_QCOUNT']), 3)
        self.assertEqual(t.variables['_QCOUNT_EXACT'], 3)
        t.dispatch('QCOUNT 12')
        self.assertIn('?USAGE: QCOUNT', t.io.text)


class TestDeutschJozsaBernsteinVazirani(unittest.TestCase):

    def test_dj(self):
        for kind, want in (('BALANCED', 0), ('CONSTANT', -1)):
            for seed in (1, 2):
                t = term(4, seed=seed)
                t.dispatch(f'DJ 4 {kind}')
                self.assertEqual(t.variables['_DJ_CONSTANT'], want)
        t = term(4, seed=7)
        t.dispatch('DJ 4 MAYBE')
        self.assertIn('?USAGE: DJ', t.io.text)

    def test_bv(self):
        t = term(4, seed=7)
        t.dispatch('BV 1011001')
        self.assertEqual(t.variables['_BV_OK'], -1)
        t._endian_big = True
        t.dispatch('BV 100')
        self.assertEqual(t.variables['_BV_OK'], -1)
        t.dispatch('BV 102')
        self.assertIn('?USAGE: BV', t.io.text)


class TestVQEExcitedStates(unittest.TestCase):

    def test_deflation_finds_every_level(self):
        t = term(4, seed=3)
        t.num_qubits = 2
        t.dispatch('HAMILTONIAN H = 1.0 ZZ 0 1 + 0.5 X 0 + 0.3 X 1')
        t.dispatch('VQE H LAYERS 2 EXCITED 3')
        np.testing.assert_allclose(t.arrays['_VQE_ENERGIES'], t.arrays['_VQE_EXACT_LEVELS'], atol=1e-6)


# ---------------------------------------------------------------------------
# Optimizers and circuit primitives checked against exact results: MINIMIZE
# with each optimizer, GRADIENT against the analytic derivative, QFT, diffusion
# and multi-controlled gates against their matrices, the quantum walk and graph
# states.
# ---------------------------------------------------------------------------


def circuit_of(t, *stmts):
    """The circuit the program lines `stmts` build."""
    t.program.clear()
    for k, s in enumerate(stmts):
        t.process(f'{10 * (k + 1)} {s}', track_undo=False)
    qc, _ = t.build_circuit()
    return qc.remove_final_measurements(inplace=False)


class TestMinimize(unittest.TestCase):

    def _ansatz(self):
        t = term(1, seed=3)
        feed(t, '10 RY theta, 0', '20 SAVE_EXPECT Z 0 -> z', 'theta = 0.4')
        return t

    def test_each_optimizer_finds_the_minimum(self):
        for method in ('NM', 'SPSA', 'GRAD'):
            with self.subTest(method=method):
                t = self._ansatz()
                out = feed(t, f'MINIMIZE theta -> z ITERS 120 STEP 0.6 METHOD {method}')
                self.assertIn('parametric compile', out)
                self.assertLess(t.variables['_COST'], -0.99)
                self.assertAlmostEqual(abs(t.variables['theta']) % (2 * math.pi), math.pi, delta=0.15)

    def test_rebuild_path_for_programs_that_cannot_bind_parameters(self):
        t = term(1, seed=3)
        feed(t, '10 IF theta > 100 THEN X 0', '20 RY theta, 0', '30 SAVE_EXPECT Z 0 -> z', 'theta = 0.4')
        out = feed(t, 'MINIMIZE theta -> 2 * z ITERS 60')
        self.assertNotIn('parametric compile', out)
        self.assertLess(t.variables['_COST'], -1.98)

    def test_errors(self):
        out = feed(term(1, seed=3), 'MINIMIZE theta -> z')
        self.assertIn('?NOTHING TO MINIMIZE', out)
        t = self._ansatz()
        out = feed(t, 'MINIMIZE theta', 'MINIMIZE theta -> z METHOD ADAM', 'MINIMIZE theta -> undefined_cost')
        self.assertIn('?USAGE: MINIMIZE', out)
        self.assertIn("?UNKNOWN OPTIMIZER 'ADAM'", out)
        self.assertIn('?MINIMIZE ERROR', out)

    def test_nelder_mead_in_two_dimensions(self):
        t = term(1, seed=3)
        best, fval = t._nelder_mead(lambda v: (v[0] - 1) ** 2 + (v[1] + 2) ** 2, [0.0, 0.0], 300, 0.5)
        np.testing.assert_allclose(best, [1, -2], atol=1e-3)
        self.assertEqual(t._nelder_mead(lambda v: 7.0, [], 10, 0.5), ([], 7.0))


class TestGradient(unittest.TestCase):

    def test_parameter_shift_matches_the_derivative(self):
        t = term(seed=3)
        feed(t, '10 RY a, 0', '20 RX b, 1', '30 SAVE_EXPECT Z 0 -> za', '40 SAVE_EXPECT Z 1 -> zb',
             'a = 0.3', 'b = 1.1')
        feed(t, 'GRADIENT a, b -> za + 2 * zb')
        self.assertAlmostEqual(t.variables['_GRAD_a'], -math.sin(0.3), places=6)
        self.assertAlmostEqual(t.variables['_GRAD_b'], -2 * math.sin(1.1), places=6)
        self.assertEqual((t.variables['a'], t.variables['b']), (0.3, 1.1))

    def test_errors(self):
        self.assertIn('?NOTHING TO DIFFERENTIATE', feed(term(1, seed=3), 'GRADIENT a -> z'))
        t = term(1, seed=3)
        feed(t, '10 RY a, 0')
        self.assertIn('?USAGE: GRADIENT', feed(t, 'GRADIENT a'))


class TestPrimitives(unittest.TestCase):

    def test_qft_and_inverse_match_qiskit(self):
        t = term(3, seed=3)
        qc = circuit_of(t, 'QFT 0-2')
        ref = QuantumCircuit(3)
        ref.append(QFTGate(3), [0, 1, 2])
        self.assertTrue(Operator(qc).equiv(Operator(ref)))
        qc = circuit_of(t, 'QFT 0-2', 'IQFT 0, 1, 2')
        self.assertTrue(Operator(qc).equiv(Operator(np.eye(8))))
        qc = circuit_of(t, 'QFT 2-0')
        ref = QuantumCircuit(3)
        ref.append(QFTGate(3), [2, 1, 0])
        self.assertTrue(Operator(qc).equiv(Operator(ref)))

    def test_diffusion_is_the_reflection_about_the_uniform_state(self):
        t = term(3, seed=3)
        s = np.full(8, 1 / math.sqrt(8))
        want = 2 * np.outer(s, s) - np.eye(8)
        for spec in ('DIFFUSE 0-2', 'DIFFUSION'):
            self.assertTrue(Operator(circuit_of(t, spec)).equiv(Operator(want)))
        t1 = term(1, seed=3)
        self.assertTrue(Operator(circuit_of(t1, 'DIFFUSE 0')).equiv(Operator(-np.array([[0, 1], [1, 0]]))))

    def test_multi_controlled_gates(self):
        t = term(3, seed=3)
        u = Operator(circuit_of(t, 'MCX 0, 1, 2')).data
        self.assertAlmostEqual(abs(u[7, 3]), 1.0)          # |011> -> |111>
        u = Operator(circuit_of(t, 'MCZ 0, 1, 2')).data
        np.testing.assert_allclose(np.diag(u), [1] * 7 + [-1], atol=1e-9)
        u = Operator(circuit_of(t, 'MCP PI/3, 0, 1, 2')).data
        self.assertAlmostEqual(u[7, 7], np.exp(1j * math.pi / 3))
        for stmt, msg in (('MCP', 'needs an angle'), ('MCX 0', 'at least one control'), ('QFT 0-5', 'OUT OF RANGE')):
            out = feed(t, '10 ' + stmt, 'RUN')
            self.assertIn(msg, out)

    def test_register_arithmetic(self):
        t = term(4, seed=3)
        sv = Statevector(circuit_of(t, 'X 0', 'X 2', 'X 3', 'QADD 0-1, 2-3'))   # a = 1, b = 3
        self.assertAlmostEqual(abs(sv.data[0b1100]), 1.0)                    # a = 0 (1 + 3 mod 4)
        sv = Statevector(circuit_of(t, 'X 1', 'QADDC 3, 0-1'))                # 2 + 3 mod 4 = 1
        self.assertAlmostEqual(abs(sv.data[0b0001]), 1.0)
        out = feed(t, '10 QADD 0-1, 2', 'RUN')
        self.assertIn('equal size', out)

    def test_phase_estimation_statement(self):
        t = term(4, seed=3)
        feed(t, 'UNITARY PH = [[1, 0], [0, exp(2 * PI * j * 0.375)]]')
        sv = Statevector(circuit_of(t, 'X 3', 'QPE 0-2 3 PH'))
        self.assertAlmostEqual(abs(sv.data[0b1011]), 1.0)                    # 0.375 * 8 = 3
        for stmt, msg in (('QPE 0-2 3 NOPE', 'unknown UNITARY'), ('QPE 0-1 2-3 PH', '1-qubit')):
            self.assertIn(msg, feed(t, '10 ' + stmt, 'RUN'))

    def test_locc_mode_refuses_standard_primitives(self):
        t = term(1, seed=3)
        feed(t, 'LOCC 2 2')
        for stmt in ('QFT 0-1', 'GRAPHSTATE', 'FEATUREMAP 1, 2'):
            self.assertIn('LOCC', feed(t, '10 ' + stmt, 'RUN'))


class TestWalkAndGraphStates(unittest.TestCase):

    def test_quantum_walk_spreads_symmetrically(self):
        t = term(4, seed=3)
        out = feed(t, 'QWALK 3')
        probs = {int(line.split()[1]): float(line.split()[2]) for line in out.splitlines()
                 if line.strip().startswith('pos')}
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
        self.assertEqual(sorted(probs), [1, 3, 5, 7])            # odd distance from 4 after 3 steps
        self.assertIn('?USAGE: QWALK', feed(t, 'QWALK many'))
        self.assertIn('at least 2 qubits', feed(term(1, seed=3), 'QWALK 2'))

    def test_graph_states(self):
        def graph_state(n, edges):
            qc = QuantumCircuit(n)
            qc.h(range(n))
            for a, b in edges:
                qc.cz(a, b)
            return Statevector(qc)

        t = term(3, seed=3)
        for spec, edges in (('LINE', [(0, 1), (1, 2)]), ('RING', [(0, 1), (1, 2), (0, 2)]),
                            ('COMPLETE', [(0, 1), (0, 2), (1, 2)]), ('0-2, 2-1', [(0, 2), (1, 2)])):
            sv = Statevector(circuit_of(t, f'GRAPHSTATE {spec}'))
            self.assertTrue(sv.equiv(graph_state(3, edges)), spec)
        t2 = term(seed=3)
        self.assertTrue(Statevector(circuit_of(t2, 'GRAPHSTATE RING')).equiv(graph_state(2, [(0, 1)])))
        for spec, msg in (('0-5', 'two different qubits'), ('0+1', 'a-b pairs')):
            self.assertIn(msg, feed(t, f'10 GRAPHSTATE {spec}', 'RUN'))


# ---------------------------------------------------------------------------
# Iterative phase estimation, amplitude estimation and amplification, the
# BP+OSD decoder's pieces, Hamiltonian and channel definitions, and LINDBLAD
# evolution, against exact answers and their error messages.
# ---------------------------------------------------------------------------


class TestEstimation(unittest.TestCase):

    def test_iqpe(self):
        t = term(1, seed=3)
        feed(t, 'UNITARY PH = [[1, 0], [0, exp(2 * PI * j * 0.375)]]', '10 X 0')
        out = feed(t, 'IQPE 3 0 PH')
        self.assertIn('IQPE phase = 0.375000  (bits 011)', out)
        self.assertEqual(t.variables['_IQPE'], 0.375)
        out = feed(t, 'IQPE 3', 'IQPE 3 0 NOPE')
        self.assertIn('?USAGE: IQPE', out)
        self.assertIn("unknown UNITARY 'NOPE'", out)
        t2 = term(1, seed=3)
        feed(t2, 'UNITARY PH = [[1, 0], [0, -1]]')
        self.assertIn('no program', feed(t2, 'IQPE 2 0 PH'))

    def test_amplitude_estimation(self):
        t = term(1, seed=3)
        feed(t, '10 RY 1.2, 0')
        out = feed(t, 'AMPEST 4 0 SHOTS 400')
        self.assertIn('exact', out)
        self.assertAlmostEqual(t.variables['_AMPEST'], math.sin(0.6) ** 2, delta=0.05)
        t2 = term(seed=3)
        feed(t2, '10 H 0', '20 CX 0, 1')
        feed(t2, 'AMPEST |11>')
        self.assertAlmostEqual(t2.variables['_AMPEST'], 0.5, delta=0.05)
        t3 = term(seed=3)
        feed(t3, 'AMPEST 3 1')                                  # no program: H on every qubit
        self.assertAlmostEqual(t3.variables['_AMPEST'], 0.5, delta=0.06)
        for spec, msg in (('', '?USAGE: AMPEST'), ('|1>', 'good state must be 2 bits'), ('7', 'out of range'),
                          ('-1 0', 'm must be >= 0')):
            self.assertIn(msg, feed(t3, f'AMPEST {spec}'))
        t4 = term(1, seed=3)
        feed(t4, '10 H 0', '20 RESET 0')
        self.assertIn('must be unitary', feed(t4, 'AMPEST 0'))

    def test_amplify(self):
        t = term(seed=3)
        feed(t, '10 H 0', '20 H 1', '30 AMPLIFY 10', '40 SAVE_PROBS 0, 1 -> p', 'RUN')
        self.assertAlmostEqual(t.arrays['p'][2], 1.0, places=6)    # one step finds 1 of 4
        feed(t, 'OPTION ENDIAN BIG', 'RUN')
        self.assertAlmostEqual(t.arrays['p'][1], 1.0, places=6)    # the same text names |01> internally
        self.assertIn('must be 2 bits', feed(t, '30 AMPLIFY 1', 'RUN'))
        feed(t, 'LOCC 1 1')
        self.assertIn('requires @A/@B prefix', feed(t, '10 AMPLIFY 1', '20', '40', 'RUN'))


class TestDecoderPieces(unittest.TestCase):

    def test_bp_and_osd(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        e, llr, ok = _bp_decode(H, np.array([1, 1], dtype=np.uint8), 0.1)
        self.assertTrue(ok)
        np.testing.assert_array_equal(e, [0, 1, 0])
        e = _osd0(H, np.array([1, 0], dtype=np.uint8), np.array([-1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(H.dot(e) % 2, [1, 0])
        np.testing.assert_array_equal(e, [1, 0, 0])
        _, _, ok = _bp_decode(H, np.array([1, 1], dtype=np.uint8), 0.1, max_iter=0)
        self.assertFalse(ok)


class TestHamiltoniansAndChannels(unittest.TestCase):

    def test_definition_errors(self):
        t = term(seed=3)
        cases = {
            'HAMILTONIAN': '?USAGE: HAMILTONIAN',
            'HAMILTONIAN A = 1.0 QZ 0 1': "bad Pauli 'QZ'",
            'HAMILTONIAN A = 1.0 ZZ 0': "needs 2 qubit(s), got 1",
            'HAMILTONIAN A = 1.0': 'has no Pauli operator',
            'HAMILTONIAN A = 1.0 Z 5': 'qubit 5 out of range',
            'HAMILTONIAN A = MOLECULE LIH': 'MOLECULE supports H2',
            'CHANNEL': '?USAGE: CHANNEL',
            'CHANNEL K = [[1, 0], [0, 0.5]]': 'CPTP',
        }
        for line, msg in cases.items():
            with self.subTest(line=line):
                self.assertIn(msg, feed(t, line))
        feed(t, 'HAMILTONIAN F = HUBBARD 1.0 0.5', 'HAMILTONIAN R = RYDBERG 1.0 0.3 2.0')
        self.assertTrue(np.allclose(t._hamiltonians['F'].to_matrix(), t._hamiltonians['F'].to_matrix().conj().T))
        self.assertIn('unknown CHANNEL', feed(t, '10 APPLYCHANNEL NOPE 0', 'RUN'))

    def test_lindblad(self):
        t = term(1, seed=3)
        feed(t, '10 X 0', 'RUN', 'LINDBLAD NONE, 1.0, 200, 1.0 SM 0')      # decay from |1>
        self.assertAlmostEqual(t.variables['_RHO1'], math.exp(-1), places=4)
        out = feed(t, 'LINDBLAD NONE, 1.0, 50, 1.0 SM 0 TRAJ 400')
        self.assertNotIn('?', out)
        for bad, msg in (('NONE, 1, 10, 1.0 SM', "must be '<rate> <op> <qubit>'"),
                         ('NONE, 1, 10, 1.0 QQ 0', "unknown jump op 'QQ'"),
                         ('NONE, 1, 10, 1.0 SM 4', 'qubit 4 out of range'),
                         ('NOPE, 1, 10, 1.0 SM 0', "unknown HAMILTONIAN 'NOPE'")):
            self.assertIn(msg, feed(t, f'LINDBLAD {bad}'))
        self.assertIn('?USAGE: LINDBLAD', feed(t, 'LINDBLAD 1.0'))
        t11 = term(11, seed=3)
        self.assertIn('limited to', feed(t11, 'LINDBLAD NONE, 1.0, 10, 1.0 SM 0'))
        t16 = term(16, seed=3)
        self.assertIn('TRAJ is limited to 15 qubits', feed(t16, 'LINDBLAD NONE, 1.0, 10, 1.0 SM 0 TRAJ'))


if __name__ == '__main__':
    unittest.main()
