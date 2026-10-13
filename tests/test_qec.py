"""Error correction: lattice surgery, circuit-level noise and detector error
models, and encoded simulation with open-system channels."""

import itertools
import unittest

import numpy as np
from conftest import ScriptIO, term

from qubasic_core import qec
from qubasic_core.logical import STEANE_CHECKS, steane_decode
from qubasic_core.qec import (
    DetectorModel,
    _gf2_rank,
    detector_model,
    fault_effects,
    memory_circuit,
    repetition_memory,
    surface_memory,
    to_stim,
)
from qubasic_core.terminal import QBasicTerminal

# ---------------------------------------------------------------------------
# Surface-code geometry, detector models, lattice surgery and encoded logical qubits.
# ---------------------------------------------------------------------------


def symplectic(p):
    n = len(p.qubits)
    order = {q: k for k, q in enumerate(p.qubits)}
    rows = []
    for f in p.x_checks:
        v = np.zeros(2 * n, np.uint8)
        v[[order[q] for q in f]] = 1
        rows.append(v)
    for f in p.z_checks:
        v = np.zeros(2 * n, np.uint8)
        v[[n + order[q] for q in f]] = 1
        rows.append(v)
    return np.array(rows), order, n


class TestGeometry(unittest.TestCase):

    def test_patches_are_valid_codes(self):
        for rows, cols in ((3, 3), (5, 5), (7, 3), (3, 7)):
            p = qec.make_patch(rows, cols)
            M, order, n = symplectic(p)
            comm = (M[:, :n].astype(int) @ M[:, n:].T.astype(int)
                    + M[:, n:].astype(int) @ M[:, :n].T.astype(int)) % 2
            self.assertFalse(comm.any())
            self.assertEqual(n - _gf2_rank(M), 1)
            code = qec.code_dict(p)
            self.assertEqual(code['n'], rows * cols)
            self.assertEqual(len(code['stab']), rows * cols - 1)

    def test_h_relabel_preserves_the_code(self):
        for d in (3, 5):
            p = qec.make_patch(d, d)
            perm = qec.h_relabel(d)
            self.assertEqual({tuple(sorted(perm[q] for q in f)) for f in p.z_checks},
                             {tuple(sorted(f)) for f in p.x_checks})
            self.assertEqual({tuple(sorted(perm[q] for q in f)) for f in p.x_checks},
                             {tuple(sorted(f)) for f in p.z_checks})

    def test_surgery_layouts(self):
        for kind in ('ZZ', 'XX'):
            sg = qec.surgery(3, kind)
            self.assertEqual(sg.n_data, 21)
            self.assertEqual(len(sg.new_checks), 6)
            checks = sg.merged.z_checks if kind == 'ZZ' else sg.merged.x_checks
            target = (sg.a.logical_z + sg.b.logical_z if kind == 'ZZ'
                      else sg.a.logical_x + sg.b.logical_x)
            sub = qec.product_subset(checks, target, sg.n_data)
            prod = np.zeros(sg.n_data, dtype=np.uint8)
            for k in sub:
                prod[checks[k]] ^= 1
            want = np.zeros(sg.n_data, dtype=np.uint8)
            want[target] = 1
            self.assertTrue(np.array_equal(prod, want))
        with self.assertRaises(ValueError):
            qec.surgery(4)
        with self.assertRaises(ValueError):
            qec.surgery(3, 'YY')
        with self.assertRaises(ValueError):
            qec.product_subset([[0, 1]], [0], 2)


class TestDetectorModel(unittest.TestCase):

    def test_repetition_memory(self):
        # A distance-5 repetition code in one round: decoding fixes up to two flips.
        n = 5
        model = DetectorModel(n - 1, 1)
        for q in range(n):
            dets = [k for k in (q - 1, q) if 0 <= k < n - 1]
            model.add(0.05, dets, 1 if q == 0 else 0)
        rate, per = model.logical_failure_rate(20000, np.random.default_rng(1))
        self.assertLess(rate, 0.01)
        self.assertEqual(model.decode(np.zeros(n - 1, dtype=np.uint8)), 0)
        with self.assertRaises(ValueError):
            model.add(0.1, [0, 1, 2])

    def test_agrees_with_pymatching(self):
        try:
            import pymatching
        except ImportError:
            self.skipTest('pymatching not installed')
        import math
        t = term(4, seed=3)
        model = t._phenomenological_model(t._qec_code('SURFACE', 5), 0.03, 5)
        det, obs = model.sample(3000, np.random.default_rng(11))
        m = pymatching.Matching()
        for pr, ds, o in zip(model.probs, model.dets, model.obs):
            faults = {0} if o & 1 else set()
            w = math.log((1 - pr) / pr)
            if len(ds) == 2:
                m.add_edge(ds[0], ds[1], fault_ids=faults, weight=w, merge_strategy='smallest-weight')
            else:
                m.add_boundary_edge(ds[0], fault_ids=faults, weight=w,
                                    merge_strategy='smallest-weight')
        theirs = np.mean(m.decode_batch(det)[:, 0] != obs[:, 0])
        ours = np.mean(np.array([model.decode(row) & 1 for row in det]) != obs[:, 0])
        self.assertLess(abs(ours - theirs), 0.03)

    def test_sampling_shapes(self):
        model = DetectorModel(2, 2)
        model.add(0.5, [0], 1)
        model.add(0.5, [0, 1], 2)
        det, obs = model.sample(100, np.random.default_rng(0))
        self.assertEqual(det.shape, (100, 2))
        self.assertEqual(obs.shape, (100, 2))


class TestPhenomenologicalMemory(unittest.TestCase):

    def test_threshold_behaviour(self):
        # Equal round counts, below (p=0.02) and well above (p=0.08) threshold.
        t = term(4, seed=2)
        rates = {}
        for p in (0.02, 0.08):
            for d in (3, 5):
                t.cmd_logical_error_rate(f'SURFACE {d} {p} PHENOM 3000 5')
                rates[(p, d)] = t.variables['_LER']
        self.assertLess(rates[(0.02, 5)], rates[(0.02, 3)])
        self.assertGreater(rates[(0.08, 5)], rates[(0.08, 3)])

    def test_unmatchable_code_is_refused(self):
        t = term(4, seed=3)
        t.cmd_logical_error_rate('STEANE 0.01 PHENOM')
        self.assertIn('not matchable', t.io.text)


class TestLatticeSurgery(unittest.TestCase):

    def test_zz_and_xx_merges(self):
        cases = [('0 1', 0.0), ('1 1', 1.0), ('+ - XX', 0.0), ('0 0 XX', None), ('+ +', None)]
        for args, plus in cases:
            with self.subTest(args=args):
                t = term(4, seed=3)
                t.cmd_lattice(args + ' SHOTS 120')
                self.assertEqual(t.variables['_LATTICE_AGREE'], 1.0, t.io.text)
                if plus is None:
                    self.assertTrue(0.25 < t.variables['_LATTICE_PLUS'] < 0.75)
                else:
                    self.assertEqual(t.variables['_LATTICE_PLUS'], plus)

    def test_distance_five(self):
        t = term(4, seed=3)
        t.cmd_lattice('0 0 D 5 SHOTS 40')
        self.assertIn('55 data qubits', t.io.text)
        self.assertEqual(t.variables['_LATTICE_PLUS'], 1.0)

    def test_cnot_truth_table(self):
        for c in '01':
            for target in '01':
                t = term(4, seed=3)
                t.cmd_lattice(f'CNOT {c} {target} SHOTS 60')
                self.assertEqual(t.variables['_LATTICE_AGREE'], 1.0, t.io.text)

    def test_noisy_merge_improves_with_distance(self):
        t = term(4)
        t.cmd_lattice('RATE 3 0.01 TRIALS 3000')
        r3 = t.variables['_LATTICE_RATE']
        t.cmd_lattice('RATE 5 0.01 TRIALS 3000')
        r5 = t.variables['_LATTICE_RATE']
        self.assertGreater(r3, 0)
        self.assertLess(r5, r3)
        t.cmd_lattice('RATE 3 0.01 ROUNDS 1 TRIALS 3000')
        self.assertGreater(t.variables['_LATTICE_RATE'], r3)

    def test_usage_and_errors(self):
        for args in ('', '0', 'Q 1', 'CNOT +', 'RATE 3', 'RATE 3 0.7', 'CNOT + 0'):
            t = term(4, seed=3)
            t.cmd_lattice(args)
            self.assertTrue(t.io.text.lstrip().startswith('?'), args)


class TestEncodedLogicalQubits(unittest.TestCase):

    def run_encoded(self, code, lines, p, shots=1500, n=2, mode=''):
        t = term(4, seed=3)
        t.cmd_lqubits(f'{n} CODE {code} PHYS {p} {mode}'.strip())
        t.shots = shots
        for k, line in enumerate(lines):
            t.process(f'{(k + 1) * 10} {line}', track_undo=False)
        t.cmd_run()
        return t

    def test_noiseless_programs_match_the_ideal(self):
        for code in ('SURFACE 3', 'SURFACE 5'):
            t = self.run_encoded(code, ['H 0', 'CX 0, 1', 'H 1', 'CZ 0, 1', 'H 0', 'MEASURE'], 0.0)
            self.assertEqual(t.variables['_LOGICAL_ERR'], 0.0)
            self.assertLess(t.variables['_LOGICAL_TVD'], 0.05)
        t = self.run_encoded('REP 5', ['X 0', 'CX 0, 1', 'X 0', 'SWAP 0, 1', 'MEASURE'], 0.0)
        self.assertEqual(t.last_counts, {'01': 1500})

    def test_distance_suppresses_logical_errors(self):
        prog = ['X 0', 'H 1', 'H 1', 'CX 0, 1', 'MEASURE']
        e3 = self.run_encoded('SURFACE 3', prog, 0.02, shots=3000).variables['_LOGICAL_ERR']
        e5 = self.run_encoded('SURFACE 5', prog, 0.02, shots=3000).variables['_LOGICAL_ERR']
        self.assertGreater(e3, 0.0)
        self.assertLess(e5, e3)

    def test_measure_subset_and_shor(self):
        t = self.run_encoded('SHOR', ['X 1', 'CX 1, 0', 'MEASURE 0'], 0.0)
        self.assertEqual(t.last_counts, {'1': 1500})

    def test_fallback_and_modes(self):
        t = self.run_encoded('SURFACE 3', ['T 0', 'MEASURE'], 0.001, shots=200)
        self.assertIn('modeled logical channel', t.io.text)
        self.assertIn('surgery', t.io.text)
        t = self.run_encoded('SURFACE 3', ['T 0', 'MEASURE'], 0.001, shots=200, mode='ENCODED')
        self.assertIn('no transversal form', t.io.text)
        t = self.run_encoded('SURFACE 3', ['H 0', 'MEASURE'], 0.001, shots=200, mode='MODEL')
        self.assertIn('surgery', t.io.text)
        self.assertNotIn('_LOGICAL_ERR', t.variables)
        t = self.run_encoded('SURFACE 3', ['MEAS 0 -> m', 'MEASURE'], 0.001, shots=50)
        self.assertIn('MEASURE rather than MEAS', t.io.text)

    def test_status_lines(self):
        t = term(4, seed=3)
        t.cmd_lqubits('2 CODE SURFACE 5 PHYS 1e-3')
        t.cmd_lqubits('STATUS')
        self.assertIn('Pauli-frame', t.io.text)


# ---------------------------------------------------------------------------
# Circuit-level memory experiments on the built-in noise model.
# ---------------------------------------------------------------------------

def _qiskit_circuit(circ):
    """The noiseless circuit as a Qiskit circuit (one classical bit per measurement)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(circ.n_qubits, circ.n_meas)
    for op in circ.ops:
        if op[0] == 'R':
            qc.reset(op[1])
        elif op[0] == 'H':
            qc.h(op[1])
        elif op[0] == 'CX':
            qc.cx(op[1], op[2])
        elif op[0] == 'M':
            qc.measure(op[1], op[2])
    return qc


class TestCircuits(unittest.TestCase):

    def test_sizes(self):
        surf = surface_memory(3, 3)
        self.assertEqual(surf.n_qubits, 17)
        self.assertEqual(len(surf.detectors), 4 * 4)          # 4 Z checks x (3 rounds + readout)
        self.assertEqual(len(surf.observable), 3)
        rep = repetition_memory(5, 4)
        self.assertEqual((rep.n_qubits, len(rep.detectors)), (9, 4 * 5))
        with self.assertRaises(ValueError):
            memory_circuit('STEANE', 3, 3)

    def test_noiseless_detectors_and_observable_are_zero(self):
        from qiskit_aer import AerSimulator
        for circ in (surface_memory(3, 2), surface_memory(5, 1), repetition_memory(4, 3)):
            with self.subTest(circuit=circ.label):
                counts = AerSimulator(method='stabilizer').run(
                    _qiskit_circuit(circ), shots=40, seed_simulator=3).result().get_counts()
                for key in counts:
                    bits = [int(b) for b in key.replace(' ', '')[::-1]]
                    for ms in circ.detectors:
                        self.assertEqual(sum(bits[m] for m in ms) % 2, 0)
                    self.assertEqual(sum(bits[m] for m in circ.observable) % 2, 0)

    def test_every_fault_splits_into_matchable_parts(self):
        for circ in (surface_memory(3, 3), surface_memory(5, 2), repetition_memory(5, 3)):
            with self.subTest(circuit=circ.label):
                model, unsplit = detector_model(circ, 0.001)
                self.assertEqual(unsplit, 0)
                self.assertTrue(all(len(d) <= 2 for d in model.dets))
                probs, D, L = fault_effects(circ, 0.001)
                self.assertEqual(len(probs), D.shape[0])
        empty, _ = detector_model(surface_memory(3, 1), 0.0)
        self.assertEqual(empty.probs, [])

    def test_distance_suppresses_errors_below_threshold(self):
        rates = {}
        for d in (3, 5):
            model, _ = detector_model(surface_memory(d, 3), 0.001)
            rates[d], _ = model.logical_failure_rate(4000, np.random.default_rng(7))
        self.assertGreater(rates[3], rates[5])

    def test_matches_stim_when_installed(self):
        try:
            import stim
        except ImportError:
            self.skipTest('stim not installed')
        for circ in (surface_memory(3, 3), repetition_memory(5, 3)):
            model, _ = detector_model(circ, 0.002)
            dem = stim.Circuit(to_stim(circ, 0.002)).detector_error_model(decompose_errors=True)
            theirs = set()
            for inst in dem.flattened():
                if inst.type != 'error':
                    continue
                part = []
                for tgt in inst.targets_copy() + [None]:
                    if tgt is None or tgt.is_separator():
                        dets = tuple(sorted(t.val for t in part if t.is_relative_detector_id()))
                        theirs.add((dets, int(any(t.is_logical_observable_id() for t in part))))
                        part = []
                    else:
                        part.append(tgt)
            self.assertEqual(set(zip(model.dets, model.obs)), theirs)


class TestCommand(unittest.TestCase):

    def test_circuit_mode_needs_no_stim(self):
        t = QBasicTerminal()
        t.io = ScriptIO()
        t._seed = 2
        t.dispatch('LOGICAL_ERROR_RATE SURFACE 3 0.003 CIRCUIT 1500')
        self.assertIn('CIRCUIT-level noise', t.io.text)
        self.assertIn('detectors = 16', t.io.text)
        self.assertGreater(t.variables['_LER'], 0)
        t.dispatch('LOGICAL_ERROR_RATE REP 3 0.01 CIRCUIT 500 2')
        self.assertIn('2 rounds', t.io.text)
        t.dispatch('LOGICAL_ERROR_RATE STEANE 0.01 CIRCUIT')
        self.assertIn('?LOGICAL_ERROR_RATE ERROR', t.io.text)

    def test_threshold_crossing_estimate(self):
        cross = QBasicTerminal._threshold_crossing
        ps = [0.01, 0.02, 0.04]
        self.assertAlmostEqual(cross(ps, [[0.02, 0.01], [0.04, 0.04], [0.08, 0.16]]), 0.02)
        between = cross([0.01, 0.02], [[0.02, 0.01], [0.04, 0.08]])
        self.assertTrue(0.01 < between < 0.02)
        self.assertIsNone(cross(ps, [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]))
        self.assertIsNone(cross([0.01], [[0.1, 0.2]]))

    def test_threshold_modes(self):
        t = QBasicTerminal()
        t.io = ScriptIO()
        t._seed = 4
        t.dispatch('THRESHOLD REP 0.005 0.02 2 3 5 TRIALS 300 CIRCUIT')
        self.assertIn('circuit level, d rounds', t.io.text)
        t.dispatch('THRESHOLD SURFACE 0.01 0.05 2 3 5 TRIALS 300 PHENOM')
        self.assertIn('phenomenological, d rounds', t.io.text)
        t.dispatch('THRESHOLD SURFACE 0.05 0.15 3 3 5 TRIALS 3000')
        self.assertIn('Threshold estimate', t.io.text)
        self.assertTrue(0.05 < t.variables['_THRESHOLD'] < 0.15)


# ---------------------------------------------------------------------------
# Encoded Steane simulation (LQUBITS ... CODE STEANE) and the exact LINDBLAD path.
# ---------------------------------------------------------------------------

def steane_run(lines, p, shots=1000, n=2):
    t = QBasicTerminal()
    t.io = ScriptIO()
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
        t.io = ScriptIO()
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
        t.io = ScriptIO()
        t.num_qubits = 8
        t.cmd_hamiltonian('H = ISING 1 0.5')
        t.cmd_lindblad('H, 0.2, 10, 0.3 SM 0')
        self.assertNotIn('?LINDBLAD', t.io.text)
        self.assertAlmostEqual(sum(t.variables[f'_RHO{i}'] for i in range(256)), 1.0, places=9)


if __name__ == '__main__':
    unittest.main()
