"""Golden-script and metamorphic tests — whole programs, end to end.

These tests run complete .qb programs through the same path the CLI uses and
assert on physics identities (an operation followed by its inverse is the
identity, teleportation preserves states, EXPORT round-trips through LOADQASM)
and on whole-program behaviour (RESUME, $D014, SWEEP's restore, XEB's
normalization, OPTION ENDIAN), all on real Aer simulations.
"""

import io
import sys
import unittest

import numpy as np

from qubasic_core.editing import ProgramMgmtMixin
from qubasic_core.terminal import QBasicTerminal


def _has(*mods):
    try:
        for m in mods:
            __import__(m)
        return True
    except ImportError:
        return False


def run_script(lines, seed=42):
    """Feed lines through the CLI's loader; return (terminal, output)."""
    t = QBasicTerminal()
    t._seed = seed
    np.random.seed(seed)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        ProgramMgmtMixin._load_lines_with_defs(
            list(lines), lambda line: t.process(line, track_undo=False))
    finally:
        sys.stdout = old
    return t, buf.getvalue()


class TestQuantumIdentities(unittest.TestCase):
    def test_inverse_roundtrip_is_identity(self):
        t, _ = run_script([
            'QUBITS 1',
            '10 RX 0.7, 0',
            '20 INV RX 0.7, 0',
            'RUN',
        ])
        sv = np.ascontiguousarray(t.last_sv).ravel()
        self.assertAlmostEqual(abs(sv[0]) ** 2, 1.0, places=9)

    def test_qft_iqft_roundtrip(self):
        t, _ = run_script([
            'QUBITS 4',
            '10 X 0 : X 2',
            '20 QFT 0-3',
            '30 IQFT 0-3',
            'RUN',
        ])
        sv = np.ascontiguousarray(t.last_sv).ravel()
        self.assertAlmostEqual(abs(sv[0b0101]) ** 2, 1.0, places=9)

    def test_teleport_preserves_state(self):
        theta = 1.234
        t, _ = run_script([
            'QUBITS 3',
            'SHOTS 512',
            f'10 RY {theta}, 0',
            '20 H 1',
            '30 CX 1,2',
            '40 CX 0,1',
            '50 H 0',
            '60 MEAS 0 -> m0',
            '70 MEAS 1 -> m1',
            '80 IF m1 THEN X 2',
            '90 IF m0 THEN Z 2',
            '100 SAVE_EXPECT Z 2 -> ez',
            '110 MEASURE 2',
            'RUN',
        ])
        self.assertAlmostEqual(t.variables['ez'], np.cos(theta), places=9)

    def test_export_loadqasm_fixed_point(self):
        import os
        import tempfile
        # The path sanitizer forbids absolute paths, so work inside a temp cwd.
        with tempfile.TemporaryDirectory() as td:
            old_cwd = os.getcwd()
            os.chdir(td)
            try:
                t, out1 = run_script([
                    'QUBITS 2',
                    'SHOTS 512',
                    '10 RY 0.5, 0',
                    '20 CX 0,1',
                    '30 MEASURE',
                    'RUN',
                    'EXPORT roundtrip.qasm',
                ])
                self.assertIn('OpenQASM 2.0', out1)
                counts_a = dict(t.last_counts)
                t2, out2 = run_script(['LOADQASM roundtrip.qasm', 'SHOTS 512', 'RUN'])
                self.assertIn('2 qubits', out2)
                self.assertEqual(dict(t2.last_counts), counts_a)
            finally:
                os.chdir(old_cwd)

    def test_molecule_h2_fci_energy(self):
        from qubasic_core.physics import h2_hamiltonian
        op = h2_hamiltonian(0.7414)
        w = np.linalg.eigvalsh(np.asarray(op.to_matrix()))
        self.assertAlmostEqual(float(w[0]), -1.13727, places=4)

    def test_molecule_vqe_reaches_fci(self):
        t, out = run_script([
            'QUBITS 4',
            'HAMILTONIAN HM = MOLECULE H2 0.7414',
            'a = 0.1',
            '10 X 0',
            '20 X 1',
            '30 RY a, 2',
            '40 CX 2,3',
            '50 CX 2,0',
            '60 CX 2,1',
            '70 SAVE_EXPECT HM -> e',
            'MINIMIZE a -> e ITERS 100',
        ])
        self.assertIn('parametric compile', out)
        self.assertAlmostEqual(t.variables['_COST'], -1.13727, places=3)

    def test_parametric_minimize_tfim(self):
        t, out = run_script([
            'QUBITS 2',
            'a = 0.5', 'b = 0.5', 'c = 0.5',
            '10 RY a, 0',
            '20 RY b, 1',
            '30 CX 0,1',
            '40 RY c, 0',
            '50 SAVE_EXPECT ZZ 0,1 -> zz',
            '60 SAVE_EXPECT X 0 -> x0',
            '70 SAVE_EXPECT X 1 -> x1',
            'MINIMIZE a, b, c -> -zz - x0 - x1 ITERS 300',
        ])
        self.assertIn('parametric compile', out)
        self.assertAlmostEqual(t.variables['_COST'], -np.sqrt(5), places=6)

    def test_lindblad_traj_amplitude_damping(self):
        t, out = run_script([
            'QUBITS 1',
            '10 X 0',
            'RUN',
            'LINDBLAD NONE, 1.0, 100, 1.0 SM 0 TRAJ 500',
        ])
        self.assertIn('MCWF', out)
        self.assertAlmostEqual(t.variables['_RHO1'], np.exp(-1.0), delta=0.06)


class TestProgramBehaviour(unittest.TestCase):
    def test_resume_next_resumes(self):
        _, out = run_script([
            '10 ON ERROR GOTO 100',
            '20 ERROR 42',
            '30 PRINT "resumed"',
            '40 END',
            '100 PRINT "handler"',
            '110 RESUME NEXT',
            'RUN',
        ])
        self.assertIn('handler', out)
        self.assertIn('resumed', out)

    def test_d014_entropy_register(self):
        t, _ = run_script([
            'QUBITS 2',
            '10 H 0',
            '20 CX 0,1',
            '30 MEASURE',
            'RUN',
        ])
        self.assertAlmostEqual(t._peek(0xD014), 1.0, places=6)

    def test_sweep_restores_variable(self):
        t, _ = run_script([
            'QUBITS 1',
            't = 0.5',
            '10 RX t, 0',
            '20 MEASURE',
            'SWEEP t 0 PI 3',
        ])
        self.assertEqual(t.variables['t'], 0.5)

    def test_xeb_self_normalized_ideal(self):
        t, _ = run_script(['QUBITS 2', 'SHOTS 2000', 'XEB 2 8 30'])
        self.assertAlmostEqual(t.variables['_XEB'], 1.0, delta=0.15)

    def test_per_method_caps(self):
        t, _ = run_script(['QUBITS 100'])          # automatic: allowed
        self.assertEqual(t.num_qubits, 100)
        _, out = run_script(['METHOD statevector', 'QUBITS 100'])
        self.assertIn('RANGE', out)

    def test_option_endian_big_display(self):
        t, out = run_script([
            'QUBITS 3',
            'SHOTS 128',
            'OPTION ENDIAN BIG',
            '10 X 0',
            '20 MEASURE',
            'RUN',
            'STATE',
        ])
        self.assertIn('qubit 0 = leftmost', out)
        self.assertIn('|100', out)                       # displayed big-endian
        self.assertEqual(dict(t.last_counts), {'001': 128})   # internal keys unchanged
        self.assertEqual(t.result()['counts'], {'100': 128})  # JSON follows the toggle
        self.assertIn('big-endian', t.result()['bit_order'])

    def test_option_endian_amplify_matches_display(self):
        # Under BIG, the AMPLIFY target reads as displayed: '100' marks the
        # state whose histogram line says |100> (internally |001>).
        t, _ = run_script([
            'QUBITS 3',
            'SHOTS 512',
            'OPTION ENDIAN BIG',
            '10 H 0 : H 1 : H 2',
            '20 AMPLIFY 100',
            '30 AMPLIFY 100',
            '40 MEASURE',
            'RUN',
        ])
        top = max(t.last_counts, key=t.last_counts.get)
        self.assertEqual(top, '001')

    def test_option_endian_save_roundtrip(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            old_cwd = os.getcwd()
            os.chdir(td)
            try:
                run_script([
                    'OPTION ENDIAN BIG',
                    '10 H 0',
                    '20 MEASURE',
                    'SAVE endian.qb',
                ])
                t2, _ = run_script(['LOAD endian.qb'])
                self.assertTrue(getattr(t2, '_endian_big', False))
            finally:
                os.chdir(old_cwd)


class TestResearchQEC(unittest.TestCase):
    def test_bb_code_parameters(self):
        from qubasic_core.qec import _bb_code
        code = _bb_code(6, 6)
        self.assertEqual(code['n'], 72)
        self.assertEqual(code['k'], 12)
        self.assertFalse(np.any(code['Hx'].dot(code['Hz'].T) % 2))
        self.assertEqual(code['Lx'].shape[0], 12)
        self.assertEqual(code['Lz'].shape[0], 12)

    def test_bb_bp_osd_low_p(self):
        t, out = run_script(['LOGICAL_ERROR_RATE BB 0.003 100'])
        self.assertIn('BP+OSD', out)
        self.assertLessEqual(t.variables['_LER'], 0.02)

    @unittest.skipUnless(_has('pymatching'), 'needs pymatching')
    def test_mwpm_surface(self):
        t, _ = run_script(['LOGICAL_ERROR_RATE SURFACE 5 0.02 MWPM 5000'])
        self.assertLess(t.variables['_LER'], 0.02)

    @unittest.skipUnless(_has('stim', 'pymatching'), 'needs stim + pymatching')
    def test_circuit_level_distance_suppression(self):
        t3, _ = run_script(['LOGICAL_ERROR_RATE REP 3 0.02 CIRCUIT STIM 20000'])
        ler3 = t3.variables['_LER']
        t7, _ = run_script(['LOGICAL_ERROR_RATE REP 7 0.02 CIRCUIT STIM 20000'])
        ler7 = t7.variables['_LER']
        self.assertGreater(ler3, 0.0)
        self.assertLess(ler7, ler3)

    def test_lqubits_logical_run(self):
        t, out = run_script([
            'LQUBITS 2 CODE SURFACE 5 PHYS 1e-3 MODEL',
            'SHOTS 512',
            '10 H 0',
            '20 CX 0,1',
            '30 MEASURE',
            'RUN',
            'LQUBITS OFF',
        ])
        self.assertIn('p_L', out)
        self.assertIn('surgery', out)
        counts = t.last_counts
        good = counts.get('00', 0) + counts.get('11', 0)
        self.assertGreater(good / sum(counts.values()), 0.98)

    def test_stabilizer_scale(self):
        t, _ = run_script([
            'METHOD stabilizer',
            'QUBITS 100',
            'SHOTS 64',
            '10 H 0',
            '20 FOR I = 0 TO 98',
            '30 CX I, I+1',
            '40 NEXT I',
            '50 MEASURE',
            'RUN',
        ])
        keys = set(t.last_counts)
        self.assertTrue(keys <= {'0' * 100, '1' * 100})


if __name__ == '__main__':
    unittest.main()
