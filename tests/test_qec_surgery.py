"""Surface-code geometry, detector models, lattice surgery and encoded logical qubits."""

import unittest

import numpy as np

from qubasic_core import surface as S
from qubasic_core.dem import DetectorModel
from qubasic_core.qec2 import _gf2_rank
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


def term(seed=3):
    t = QBasicTerminal()
    t.io = _IO()
    t._seed = seed
    return t


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
            p = S.make_patch(rows, cols)
            M, order, n = symplectic(p)
            comm = (M[:, :n].astype(int) @ M[:, n:].T.astype(int)
                    + M[:, n:].astype(int) @ M[:, :n].T.astype(int)) % 2
            self.assertFalse(comm.any())
            self.assertEqual(n - _gf2_rank(M), 1)
            code = S.code_dict(p)
            self.assertEqual(code['n'], rows * cols)
            self.assertEqual(len(code['stab']), rows * cols - 1)

    def test_h_relabel_preserves_the_code(self):
        for d in (3, 5):
            p = S.make_patch(d, d)
            perm = S.h_relabel(d)
            self.assertEqual({tuple(sorted(perm[q] for q in f)) for f in p.z_checks},
                             {tuple(sorted(f)) for f in p.x_checks})
            self.assertEqual({tuple(sorted(perm[q] for q in f)) for f in p.x_checks},
                             {tuple(sorted(f)) for f in p.z_checks})

    def test_surgery_layouts(self):
        for kind in ('ZZ', 'XX'):
            sg = S.surgery(3, kind)
            self.assertEqual(sg.n_data, 21)
            self.assertEqual(len(sg.new_checks), 6)
            checks = sg.merged.z_checks if kind == 'ZZ' else sg.merged.x_checks
            target = (sg.a.logical_z + sg.b.logical_z if kind == 'ZZ'
                      else sg.a.logical_x + sg.b.logical_x)
            sub = S.product_subset(checks, target, sg.n_data)
            prod = np.zeros(sg.n_data, dtype=np.uint8)
            for k in sub:
                prod[checks[k]] ^= 1
            want = np.zeros(sg.n_data, dtype=np.uint8)
            want[target] = 1
            self.assertTrue(np.array_equal(prod, want))
        with self.assertRaises(ValueError):
            S.surgery(4)
        with self.assertRaises(ValueError):
            S.surgery(3, 'YY')
        with self.assertRaises(ValueError):
            S.product_subset([[0, 1]], [0], 2)


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
        t = term()
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
        t = term(seed=2)
        rates = {}
        for p in (0.02, 0.08):
            for d in (3, 5):
                t.cmd_logical_error_rate(f'SURFACE {d} {p} PHENOM 3000 5')
                rates[(p, d)] = t.variables['_LER']
        self.assertLess(rates[(0.02, 5)], rates[(0.02, 3)])
        self.assertGreater(rates[(0.08, 5)], rates[(0.08, 3)])

    def test_unmatchable_code_is_refused(self):
        t = term()
        t.cmd_logical_error_rate('STEANE 0.01 PHENOM')
        self.assertIn('not matchable', t.io.text)


class TestLatticeSurgery(unittest.TestCase):

    def test_zz_and_xx_merges(self):
        cases = [('0 1', 0.0), ('1 1', 1.0), ('+ - XX', 0.0), ('0 0 XX', None), ('+ +', None)]
        for args, plus in cases:
            with self.subTest(args=args):
                t = term()
                t.cmd_lattice(args + ' SHOTS 120')
                self.assertEqual(t.variables['_LATTICE_AGREE'], 1.0, t.io.text)
                if plus is None:
                    self.assertTrue(0.25 < t.variables['_LATTICE_PLUS'] < 0.75)
                else:
                    self.assertEqual(t.variables['_LATTICE_PLUS'], plus)

    def test_distance_five(self):
        t = term()
        t.cmd_lattice('0 0 D 5 SHOTS 40')
        self.assertIn('55 data qubits', t.io.text)
        self.assertEqual(t.variables['_LATTICE_PLUS'], 1.0)

    def test_cnot_truth_table(self):
        for c in '01':
            for target in '01':
                t = term()
                t.cmd_lattice(f'CNOT {c} {target} SHOTS 60')
                self.assertEqual(t.variables['_LATTICE_AGREE'], 1.0, t.io.text)

    def test_noisy_merge_improves_with_distance(self):
        t = term(seed=5)
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
            t = term()
            t.cmd_lattice(args)
            self.assertTrue(t.io.text.lstrip().startswith('?'), args)


class TestEncodedLogicalQubits(unittest.TestCase):

    def run_encoded(self, code, lines, p, shots=1500, n=2, mode=''):
        t = term()
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
        t = term()
        t.cmd_lqubits('2 CODE SURFACE 5 PHYS 1e-3')
        t.cmd_lqubits('STATUS')
        self.assertIn('Pauli-frame', t.io.text)


if __name__ == '__main__':
    unittest.main()
