"""Simulation: the numpy engine, noise models and mitigation, the RUN
pipeline, live runs, the circuit builder, LOCC mode, the memory map, the
configuration commands and the qudit engine."""

import json
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from conftest import feed, term
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qubasic_core import simtools
from qubasic_core.api import CaptureIO
from qubasic_core.locc import LOCCEngine
from qubasic_core.noise import correct_readout, extrapolate, fold_circuit
from qubasic_core.simtools import (
    _apply_gate_inplace,
    _apply_gate_np,
    _marginal_p1,
    _measure_inplace,
    _np_gate_matrix,
    _sample_counts,
)
from qubasic_core.simulator import BufferedIO, LiveNoise, LiveState, ShotIO

# ---------------------------------------------------------------------------
# In-place chunked numpy engine: equivalence with dense linear algebra.
#
# The chunk size is shrunk so small states exercise the split paths that only
# large states take in normal use.
# ---------------------------------------------------------------------------

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
        self._saved = simtools._CHUNK_AMPS
        simtools._CHUNK_AMPS = 4          # force splitting on 6-qubit states

    def tearDown(self):
        simtools._CHUNK_AMPS = self._saved

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


# ---------------------------------------------------------------------------
# Noise models: channels on every gate width, composition, NOISE ADD, LOCC propagation.
# ---------------------------------------------------------------------------


def run(t, lines):
    for k, line in enumerate(lines):
        t.process(f'{(k + 1) * 10} {line}', track_undo=False)
    t.cmd_run()
    return t.last_counts


class TestChannels(unittest.TestCase):

    def test_separate_two_qubit_depolarizing_rate(self):
        t = term(seed=4, shots=2000)
        t.cmd_noise('depolarizing 0 0.3')
        counts = run(t, ['X 0', 'X 1', 'MEASURE'])
        self.assertEqual(counts, {'11': 2000})          # one-qubit gates noiseless
        t2 = term(seed=4, shots=2000)
        t2.cmd_noise('depolarizing 0 0.3')
        counts = run(t2, ['X 0', 'CX 0, 1', 'MEASURE'])
        self.assertGreater(sum(c for k, c in counts.items() if k != '11'), 100)

    def test_single_qubit_channels_reach_multi_qubit_gates(self):
        for spec in ('amplitude_damping 0.1', 'phase_flip 0.1', 'thermal 50 70 1 5',
                     'combined 0.1 0.1', 'pauli 0.01 0.02 0.03', 'reset 0.01 0.01'):
            with self.subTest(spec=spec):
                t = term(seed=4, shots=2000)
                t.cmd_noise(spec)
                errors = t.noise.model._default_quantum_errors
                self.assertEqual(errors['cx'].num_qubits, 2)
                self.assertEqual(errors['ccx'].num_qubits, 3)
                self.assertEqual(errors['h'].num_qubits, 1)

    def test_amplitude_damping_on_cx_decays(self):
        t = term(seed=4, shots=2000)
        t.cmd_noise('amplitude_damping 0.4')
        counts = run(t, ['X 0', 'X 1', 'CX 0, 1', 'CX 0, 1', 'MEASURE'])
        self.assertGreater(counts.get('00', 0) + counts.get('01', 0) + counts.get('10', 0), 600)

    def test_parameter_validation(self):
        t = term(seed=4, shots=2000)
        t.cmd_noise('thermal 10 50 1')
        self.assertIn('T2 <= 2 T1', t.io.text)
        self.assertIsNone(t.noise.model)
        t.cmd_noise('pauli 0.5 0.5 0.5')
        self.assertIn('sum past 1', t.io.text)
        t.cmd_noise('sparkle 0.1')
        self.assertIn('UNKNOWN NOISE TYPE', t.io.text)
        self.assertIsNone(t.noise.model)


class TestComposition(unittest.TestCase):

    def test_semicolon_composes_and_add_extends(self):
        t = term(seed=4, shots=2000)
        t.cmd_noise('depolarizing 0.01; readout 0.05')
        self.assertEqual(t.noise.spec, 'depolarizing 0.01; readout 0.05')
        self.assertIsNotNone(t.noise.model._default_readout_error)
        t.cmd_noise('ADD crosstalk 0.02')
        self.assertEqual(t.noise.spec, 'depolarizing 0.01; readout 0.05; crosstalk 0.02')
        t.cmd_noise('INFO')
        self.assertIn('crosstalk 0.02', t.io.text)

    def test_readout_errors_compose(self):
        t = term(seed=4, shots=2000)
        t.cmd_noise('readout 0.1 0.2; readout 0.3 0.1')
        p = np.asarray(t.noise.model._default_readout_error.probabilities)
        want = np.array([[0.9, 0.1], [0.2, 0.8]]) @ np.array([[0.7, 0.3], [0.1, 0.9]])
        np.testing.assert_allclose(p, want)

    def test_add_without_a_model_starts_one(self):
        t = term(seed=4, shots=2000)
        t.cmd_noise('ADD readout 0.1')
        self.assertEqual(t.noise.spec, 'readout 0.1')

    def test_composed_noise_runs_live(self):
        t = term(1, seed=4, shots=400)
        t.cmd_noise('depolarizing 0.05; readout 0.1')
        run(t, ['H 0', 'MEAS 0 -> m', 'PRINT m', 'MEASURE'])
        self.assertIn('noise=trajectories', t.io.text)
        self.assertEqual(sum(t.last_counts.values()), 400)


class TestLoccPropagation(unittest.TestCase):

    def test_single_channel_reaches_locc_composite_does_not(self):
        t = term(seed=4, shots=2000)
        t.cmd_noise('phase_flip 0.2')
        t.cmd_locc('JOINT 1 1')
        self.assertEqual((t.locc.noise_type, t.locc.noise_param), ('phase_flip', 0.2))
        t.cmd_noise('depolarizing 0.1; readout 0.1')
        self.assertEqual(t.locc.noise_type, 'none')
        t.cmd_noise('OFF')
        self.assertEqual(t.locc.noise_param, 0.0)


# ---------------------------------------------------------------------------
# Error mitigation: readout correction and zero-noise extrapolation.
# ---------------------------------------------------------------------------


class TestReadoutCorrection(unittest.TestCase):

    def test_inverts_a_known_confusion_exactly(self):
        errors = [(0.05, 0.10), (0.02, 0.07)]
        true = {'00': 0.1, '01': 0.2, '10': 0.3, '11': 0.4}
        p = np.zeros(4)
        for key, v in true.items():
            p[int(key, 2)] = v
        A0 = np.array([[1 - 0.05, 0.10], [0.05, 1 - 0.10]])
        A1 = np.array([[1 - 0.02, 0.07], [0.02, 1 - 0.07]])
        noisy = np.kron(A1, A0) @ p                      # qubit 0 is the low bit
        counts = {format(i, '02b'): int(round(v * 1_000_000)) for i, v in enumerate(noisy)}
        fixed = correct_readout(counts, errors)
        for key, v in true.items():
            self.assertAlmostEqual(fixed[key], v, places=4)

    def test_result_is_a_distribution(self):
        fixed = correct_readout({'0': 3, '1': 997}, [(0.2, 0.0)])
        self.assertAlmostEqual(sum(fixed.values()), 1.0)
        self.assertTrue(all(v >= 0 for v in fixed.values()))


class TestFolding(unittest.TestCase):

    def test_folded_circuit_is_the_same_operation(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.3, 1)
        qc.t(0)
        for scale in (1, 3, 5):
            folded = fold_circuit(qc, scale)
            self.assertEqual(len(folded.data), scale * len(qc.data))
            self.assertTrue(Operator(folded).equiv(Operator(qc)))

    def test_rejects_even_scales_and_measurements(self):
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        with self.assertRaises(ValueError):
            fold_circuit(qc, 2)
        qc.measure(0, 0)
        with self.assertRaises(ValueError):
            fold_circuit(qc, 3)


class TestExtrapolation(unittest.TestCase):

    def test_each_fit_is_exact_on_its_own_family(self):
        xs = [1, 3, 5]
        self.assertAlmostEqual(extrapolate(xs, [2 - 0.1 * x for x in xs], 'LINEAR'), 2.0)
        self.assertAlmostEqual(extrapolate(xs, [1 + 0.2 * x - 0.03 * x * x for x in xs], 'QUADRATIC'), 1.0)
        self.assertAlmostEqual(extrapolate(xs, [0.5 + 0.1 * x + 0.01 * x * x for x in xs], 'RICHARDSON'), 0.5)
        self.assertAlmostEqual(extrapolate(xs, [0.9 * np.exp(-0.2 * x) for x in xs], 'EXP'), 0.9)
        self.assertAlmostEqual(extrapolate(xs, [-0.9 * np.exp(-0.2 * x) for x in xs], 'EXP'), -0.9)

    def test_bad_fits(self):
        with self.assertRaises(ValueError):
            extrapolate([1, 3], [0.9, 0.8], 'QUADRATIC')
        with self.assertRaises(ValueError):
            extrapolate([1, 3], [0.9, 0.8], 'CUBIC')


class TestMitigateCommand(unittest.TestCase):

    def _program(self, t):
        t.dispatch('QUBITS 3')
        for line in ('10 X 0', '20 X 2', '30 MEASURE'):
            t.process(line)

    def test_readout_mitigation_recovers_the_prepared_state(self):
        t = term(4, seed=11)
        self._program(t)
        t.shots = 4000
        t.dispatch('NOISE readout 0.05 0.08')
        t.dispatch('MITIGATE READOUT ON')
        self.assertIn('MITIGATE READOUT ON', t.io.text)
        t.dispatch('RUN')
        raw = t.last_counts.get('101', 0) / sum(t.last_counts.values())
        self.assertLess(raw, 0.9)
        self.assertGreater(t.last_mitigated['101'], 0.97)
        self.assertIn('readout-mitigated', t.io.text)
        t.dispatch('MITIGATE STATUS')
        self.assertIn('3 qubits', t.io.text)
        t.dispatch('MITIGATE READOUT OFF')
        self.assertIsNone(t.last_mitigated)
        t.dispatch('MITIGATE')
        self.assertTrue(t.io.text.rstrip().endswith('MITIGATE READOUT OFF'))

    def test_width_change_needs_recalibration(self):
        t = term(4, seed=11)
        self._program(t)
        t.dispatch('MITIGATE READOUT ON')
        t.dispatch('QUBITS 4')
        t.dispatch('RUN')
        self.assertIn('recalibrate', t.io.text)

    def test_usage(self):
        t = term(4, seed=11)
        t.dispatch('MITIGATE SOMETHING')
        self.assertIn('?USAGE: MITIGATE', t.io.text)


class TestZNECommand(unittest.TestCase):

    def _ghz(self, t):
        t.dispatch('QUBITS 3')
        for line in ('10 H 0', '20 CX 0, 1', '30 CX 1, 2', '40 MEASURE'):
            t.process(line)

    def test_extrapolation_beats_the_raw_value(self):
        t = term(4, seed=11)
        self._ghz(t)
        t.dispatch('NOISE depolarizing 0.01 0.04')
        t.dispatch('ZNE 1.0 ZZ 0 1 + 1.0 ZZ 1 2')
        self.assertNotIn('?', t.io.text)
        ideal, noisy = t.variables['_ZNE_IDEAL'], t.variables['_ZNE_NOISY']
        self.assertAlmostEqual(ideal, 2.0, places=6)
        linear = t.variables['_ZNE']
        self.assertLess(abs(linear - ideal), abs(noisy - ideal))
        t.dispatch('ZNE 1.0 ZZ 0 1 + 1.0 ZZ 1 2 SCALES 1 3 5 7 FIT EXP')
        self.assertLess(abs(t.variables['_ZNE'] - ideal), abs(linear - ideal))

    def test_named_hamiltonian_and_noiseless(self):
        t = term(4, seed=11)
        self._ghz(t)
        t.dispatch('HAMILTONIAN OBS = 1.0 XXX 0 1 2')
        t.dispatch('ZNE OBS')
        self.assertAlmostEqual(t.variables['_ZNE'], 1.0, places=6)

    def test_errors(self):
        t = term(4, seed=11)
        t.dispatch('ZNE')
        self.assertIn('?USAGE: ZNE', t.io.text)
        t.dispatch('ZNE 1.0 Z 0')
        self.assertIn('no program', t.io.text)
        self._ghz(t)
        t.dispatch('ZNE 1.0 Z 0 SCALES 1 2')
        self.assertIn('odd', t.io.text)
        t.process('15 MEASURE 0')
        t.dispatch('ZNE 1.0 Z 0')
        self.assertIn('?ZNE ERROR', t.io.text)


# ---------------------------------------------------------------------------
# The RUN pipeline's paths: unitary and superoperator methods, single
# precision, memory refusals, per-qubit noise from the memory map, dynamic
# circuits, device routing, SAMPLE, ESTIMATE, STEP and the JSON result.
# ---------------------------------------------------------------------------


class TestMethods(unittest.TestCase):

    def test_unitary_and_superop(self):
        t = term(1)
        out = feed(t, 'METHOD unitary', '10 H 0', 'RUN')
        self.assertIn('UNITARY (2x2)', out)
        np.testing.assert_allclose(np.abs(t.variables['unitary']), np.full((2, 2), 1 / np.sqrt(2)), atol=1e-9)
        out = feed(t, 'METHOD superop', 'RUN')
        self.assertIn('SUPEROP (4x4)', out)
        self.assertEqual(t.variables['superop'].shape, (4, 4))

    def test_single_precision_and_density_matrix(self):
        t = term()
        out = feed(t, 'PRECISION SINGLE', '10 H 0', '20 CX 0, 1', '30 MEASURE', 'RUN')
        self.assertIn('precision=single', out)
        self.assertEqual(set(t.last_counts), {'00', '11'})
        t2 = term()
        out = feed(t2, 'METHOD density_matrix', 'NOISE amplitude_damping 0.3', '10 X 0', '20 MEASURE', 'RUN')
        self.assertIn('method=density_matrix', out)
        self.assertIn('noise=on', out)
        self.assertIn('00', t2.last_counts)

    def test_memory_refusals(self):
        with mock.patch.dict(os.environ, {'QUBASIC_MEMORY_LIMIT': '4MB'}):
            t = term(1)
            t.sim_method = 'statevector'
            feed(t, 'QUBITS 22', '10 H 0', '20 MEASURE')
            out = feed(t, 'RUN')
            self.assertIn('try PRECISION SINGLE or METHOD matrix_product_state', out)
            self.assertIn('?STEP: a 22-qubit state needs', feed(t, 'STEP'))
            t.sim_method = 'automatic'
            out = feed(t, '10 T 0', 'RUN')
            self.assertIn('method=matrix_product_state', out)

    def test_memory_mapped_noise(self):
        t = term()
        t.shots = 400
        feed(t, 'POKE &HD100, 2', 'POKE &HD101, 1', '10 X 0', '20 X 1', '30 CX 1, 0', '40 MEASURE', 'RUN')
        self.assertTrue(all(k[-1] == '0' for k in t.last_counts))   # qubit 0 always decays to |0>
        noise = t._build_qubit_noise()
        self.assertIn('cx', noise.noise_instructions)


class TestDynamicAndRouting(unittest.TestCase):

    def test_feedforward_counts_keep_the_final_register(self):
        t = term()
        feed(t, '10 H 0', '20 MEAS 0 -> b', '30 IF b THEN X 1', '40 MEASURE', 'RUN')
        self.assertTrue(all(' ' not in k for k in t.last_counts))
        self.assertEqual(set(t.last_counts), {'00', '11'})

    def test_routed_summary(self):
        t = term(3)
        out = feed(t, 'COUPLING linear', 'BASIS rz, sx, x, cx', '10 H 0', '20 CX 0, 2', '30 MEASURE', 'RUN')
        self.assertIn('routed onto device', out)
        self.assertIn('swaps=', out)
        self.assertIn('basis=rz+sx+x+cx', out)


class TestPrimitivesAndStep(unittest.TestCase):

    def test_sample_and_estimate(self):
        t = term()
        self.assertIn('NOTHING TO SAMPLE', feed(t, 'SAMPLE'))
        feed(t, '10 H 0', '20 CX 0, 1', '30 MEASURE 1')
        out = feed(t, 'SAMPLE 300')
        self.assertIn('SAMPLED 300 shots', out)
        self.assertLessEqual(set(t.last_counts), {'0', '1'})
        out = feed(t, 'ESTIMATE ZZ 0 1', 'NOISE depolarizing 0.1', 'ESTIMATE ZZ 0 1')
        values = [float(line.split('=')[1]) for line in out.splitlines() if '<ZZ>' in line]
        self.assertAlmostEqual(values[0], 1.0)
        self.assertLess(values[1], 0.99)

    def test_step(self):
        t = term(1, inputs=['', 'Q'])
        feed(t, '10 H 0', '20 X 0', '30 H 0')
        out = feed(t, 'STEP')
        self.assertIn('STEP MODE', out)
        self.assertIn('STOPPED', out)
        t2 = term(1, inputs=['A1'])
        feed(t2, '10 H 0', '20 MEAS 0 -> m', '30 PRINT m', '40 END', '50 X 0')
        out = feed(t2, 'STEP')
        self.assertIn('DONE', out)
        self.assertNotIn('>>    50', out)
        t3 = term(1, inputs=[])
        feed(t3, '10 H 0')
        self.assertIn('STOPPED', feed(t3, 'STEP'))
        self.assertIn('NOTHING TO STEP', feed(term(), 'STEP'))
        t4 = term(1, inputs=['A'])
        feed(t4, '10 H 0', '20 BOGUS 0')
        self.assertIn('?LINE 20', feed(t4, 'STEP'))

    def test_result_dictionary(self):
        t = term()
        feed(t, 'OPTION ENDIAN BIG', '10 X 0', '20 MEASURE', 'label$ = "x"', 'k = 3', 'RUN')
        r = t.result()
        self.assertEqual(r['counts'], {'10': t.shots})
        self.assertIn('big-endian', r['bit_order'])
        self.assertEqual(r['variables'], {'label$': 'x', 'k': 3})
        self.assertEqual(len(r['statevector']), 4)
        self.assertEqual((r['method'], r['gates']), ('stabilizer', t.variables['_GATES']))


# ---------------------------------------------------------------------------
# Live (shot-by-shot) runs: custom and controlled gates, injected states,
# Kraus channels, noise as trajectories with readout error, per-qubit noise,
# INPUT replay across shots, and the output buffering around a build that
# turns live.
# ---------------------------------------------------------------------------


def p_one(t, qubit_char=-1):
    total = sum(t.last_counts.values())
    return sum(c for k, c in t.last_counts.items() if k[qubit_char] == '1') / total


class TestLiveGates(unittest.TestCase):

    def test_custom_controlled_and_injected_states(self):
        t = term(seed=3)
        t.shots = 50
        out = feed(t, 'UNITARY SWAPPY = [[0, 1], [1, 0]]', '10 SWAPPY 0', '20 CTRL H 0, 1', '30 MEAS 0 -> m',
                   '40 PRINT "m="; m', '50 MEASURE', 'RUN')
        self.assertIn('method=live', out)
        self.assertIn('m= 1', out)
        self.assertEqual({k[-1] for k in t.last_counts}, {'1'})
        t2 = term(seed=3)
        t2.shots = 40
        feed(t2, '10 SET_STATE [0, 0, 0, 1]', '20 MEAS 1 -> b', '30 IF b = 1 THEN X 0', '40 MEASURE', 'RUN')
        self.assertEqual(t2.last_counts, {'10': 40})

    def test_kraus_channel(self):
        t = term(1, seed=3)
        t.shots = 300
        feed(t, 'CHANNEL AD = [[1, 0], [0, 0.6]] ; [[0, 0.8], [0, 0]]', '10 X 0', '20 APPLYCHANNEL AD 0',
             '30 MEAS 0 -> m', '40 PRINT m;', '50 MEASURE', 'RUN')
        self.assertAlmostEqual(p_one(t), 0.36, delta=0.1)              # survival 0.6^2

    def test_live_rejects_what_it_cannot_run(self):
        state = LiveState(1)
        state.qc.save_statevector()
        with self.assertRaises(ValueError):
            state.flush()
        two = LiveState(2)
        from qiskit.quantum_info import Statevector
        from qiskit_aer.library import SetStatevector
        two.qc.append(SetStatevector(Statevector([0, 1])), [0])
        with self.assertRaises(ValueError):
            two.flush()


class TestTrajectories(unittest.TestCase):

    def test_noise_models(self):
        t = term(1, seed=3)
        t.shots = 400
        feed(t, 'NOISE depolarizing 0.3', '10 X 0', '20 MEAS 0 -> m', '30 PRINT m;', '40 MEASURE', 'RUN')
        self.assertAlmostEqual(p_one(t), 0.85, delta=0.06)              # X then I/2 with probability 0.3
        t2 = term(1, seed=3)
        t2.shots = 400
        feed(t2, 'NOISE amplitude_damping 0.5', '10 X 0', '20 MEAS 0 -> m', '30 PRINT m;', '40 MEASURE',
             'RUN')
        self.assertLess(p_one(t2), 0.75)
        t3 = term(1, seed=3)
        t3.shots = 400
        feed(t3, 'NOISE readout 0.2 0.0', '10 MEAS 0 -> m', '20 PRINT m;', '30 MEASURE', 'RUN')
        self.assertAlmostEqual(p_one(t3), 0.2, delta=0.06)              # readout flips the record only

    def test_per_qubit_noise(self):
        t = term(seed=3)
        t.shots = 200
        feed(t, 'POKE &HD100, 2', 'POKE &HD101, 1', '10 X 0', '20 X 1', '30 MEAS 1 -> m', '40 PRINT m;',
             '50 MEASURE', 'RUN')
        self.assertEqual({k[-1] for k in t.last_counts}, {'0'})          # qubit 0 always decays
        self.assertEqual({k[0] for k in t.last_counts}, {'1'})

    def test_channel_lookup(self):
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(0.1, 1), ['x'])
        nm.add_quantum_error(depolarizing_error(0.2, 1), ['x'], [1])
        nm.add_readout_error(ReadoutError([[0.9, 0.1], [0.2, 0.8]]), [1])
        noise = LiveNoise(nm)
        self.assertIsNot(noise.channel('x', [1]), noise.channel('x', [0]))
        self.assertIsNone(noise.channel('cx', [0, 1]))
        self.assertIsNone(noise.channel('x', [0, 1]))
        self.assertIsNone(noise.readout(0))
        np.testing.assert_allclose(noise.readout(1), [[0.9, 0.1], [0.2, 0.8]])


class TestStabilizerTableau(unittest.TestCase):
    """The stabilizer tableau against qiskit's statevector, and live runs on
    it past the statevector's memory."""

    def test_random_clifford_circuits_match_the_statevector(self):
        import itertools

        from qiskit.quantum_info import Pauli, Statevector, random_clifford
        rng = np.random.default_rng(11)
        on_tableau = {'h': 'h', 's': 's', 'sdg': 'sdg', 'x': 'pauli_x', 'y': 'pauli_y', 'z': 'pauli_z'}
        for n in (1, 2, 3, 4):
            for _ in range(10):
                tab = simtools.Tableau(n)
                sv = Statevector.from_label('0' * n)
                for _ in range(24):
                    kind = int(rng.integers(0, 10))
                    a, b = (int(v) for v in rng.permutation(n)[:2]) if n > 1 else (0, 0)
                    qc = QuantumCircuit(n)
                    if kind < 6:
                        gate = list(on_tableau)[kind]
                        getattr(tab, on_tableau[gate])(a)
                        getattr(qc, gate)(a)
                    elif kind == 6 and n > 1:
                        tab.cx(a, b)
                        qc.cx(a, b)
                    elif kind == 7 and n > 1:
                        gate = 'cz' if rng.random() < 0.5 else 'swap'
                        getattr(tab, gate)(a, b)
                        getattr(qc, gate)(a, b)
                    elif kind == 8 and n > 1:
                        m = random_clifford(2, seed=int(rng.integers(1 << 30))).to_matrix()
                        tab.apply_table(simtools.clifford_table(m), [a, b])
                        qc.unitary(m, [a, b])
                    else:
                        qs = [int(q) for q in rng.permutation(n)[:min(n, 2)]]
                        np.testing.assert_allclose(tab.probabilities(qs), sv.probabilities(qs), atol=1e-9)
                        q = qs[0]
                        p1 = sv.probabilities([q])[1]
                        bit = 1 if p1 > 0.75 else 0 if p1 < 0.25 else int(rng.integers(0, 2))
                        got = tab.measure(q, force=bit) if tab.is_random(q) else tab.measure(q)
                        self.assertEqual(got, bit)
                        proj = np.zeros((2, 2))
                        proj[bit, bit] = 1
                        sv = sv.evolve(proj, [q])
                        sv = Statevector(sv.data / np.linalg.norm(sv.data))
                        continue
                    sv = sv.evolve(qc)
                for label in map(''.join, itertools.product('IXYZ', repeat=n)):
                    px = np.array([c in 'XY' for c in label[::-1]])
                    pz = np.array([c in 'ZY' for c in label[::-1]])
                    self.assertAlmostEqual(tab.expectation(px, pz),
                                           sv.expectation_value(Pauli(label)).real, places=9)

    def test_clifford_tables_and_pauli_channels(self):
        from qiskit.quantum_info import Kraus
        from qiskit_aer.noise import amplitude_damping_error, depolarizing_error, phase_damping_error
        self.assertIsNone(simtools.clifford_table(_np_gate_matrix('T')))
        self.assertIsNotNone(simtools.clifford_table(_np_gate_matrix('RZ', (math.pi / 2,))))
        self.assertIsNone(simtools.clifford_table(_np_gate_matrix('RZ', (0.3,))))
        cum, codes = simtools.pauli_mixture(Kraus(depolarizing_error(0.3, 1)).data)
        self.assertEqual(list(codes), [0, 1, 2, 3])
        np.testing.assert_allclose(np.diff(cum, prepend=0), [0.775, 0.075, 0.075, 0.075])
        _, codes = simtools.pauli_mixture(Kraus(phase_damping_error(0.4)).data)
        self.assertEqual(list(codes), [0, 2])                                  # dephasing: I and Z
        self.assertIsNone(simtools.pauli_mixture(Kraus(amplitude_damping_error(0.4)).data))

    def test_a_wide_clifford_program_runs_live(self):
        t = term(60, seed=5, shots=40)
        out = feed(t, '10 H 0', '20 FOR i = 0 TO 58: CX i, i + 1: NEXT i', '30 MEAS 0 -> m',
                   '40 PRINT "m ="; m', '50 MEAS 59 -> k', '60 IF k <> m THEN PRINT "broken"', '70 MEASURE',
                   'RUN')
        self.assertIn('stabilizer tableau', out)
        self.assertNotIn('broken', out)
        self.assertEqual(set(t.last_counts), {'0' * 60, '1' * 60})
        self.assertEqual(t.result()['counts'], t.last_counts)
        out = feed(t, '20 T 0', 'RUN')
        self.assertIn('T is not a Clifford gate', out)
        self.assertIn('60-qubit statevector needs', out)

    def test_method_stabilizer_agrees_with_the_statevector(self):
        prog = ['10 X 0', '15 H 0', '20 H 1', '30 CX 1, 2', '40 CX 0, 1', '50 H 0', '60 MEAS 0 -> a', '70 MEAS 1 -> b',
                '80 IF b = 1 THEN X 2', '90 IF a = 1 THEN Z 2', '100 PRINT a + b', '110 H 2', '120 MEASURE 2']
        for method in ('automatic', 'stabilizer'):
            t = term(3, seed=2, shots=30)
            out = feed(t, f'METHOD {method}', *prog, 'RUN')
            self.assertEqual(t.last_counts, {'1': 30})                         # |-> teleported, then H
        self.assertIn('stabilizer tableau', out)

    def test_saved_values_and_state_on_the_tableau(self):
        t = term(40, seed=1, shots=20)
        out = feed(t, '10 H 0', '20 CX 0, 1', '30 SAVE_EXPECT ZZ 0, 1 -> zz', '40 SAVE_EXPECT XX 0, 1 -> xx',
                   '50 SAVE_EXPECT Z 0 -> z', '60 SAVE_PROBS 0, 1 -> p', '70 SAVE_AMPS 0 -> a', '80 MEAS 0 -> m',
                   '90 PRINT m', '100 MEASURE 0, 1', 'RUN')
        self.assertIn('SAVE_AMPS reads amplitudes', out)
        self.assertEqual((t.variables['zz'], t.variables['xx'], t.variables['z']), (1.0, 1.0, 0.0))
        np.testing.assert_allclose(t.arrays['p'], [0.5, 0, 0, 0.5])
        self.assertEqual(set(t.last_counts), {'00', '11'})
        t2 = term(40, seed=1, shots=10)
        out = feed(t2, '10 H 0', '20 CX 0, 1', '30 MEAS 2 -> m', '40 PRINT m', '50 MEASURE 0, 1', 'RUN',
                   'STATE', 'BLOCH 2', 'BLOCH 0', 'PROBS')
        self.assertIn('+' + '_' * 38 + 'XX', out)
        self.assertIn('+' + '_' * 38 + 'ZZ', out)
        self.assertIn('Qubit 2  (0.000, 0.000, 1.000)', out)
        self.assertIn('Qubit 0  (0.000, 0.000, 0.000)  maximally mixed', out)
        self.assertIn('?PROBS reads a statevector', out)
        feed(t2, 'OPTION ENDIAN BIG')
        self.assertIn('+XX' + '_' * 38, feed(t2, 'STATE'))

    def test_pauli_noise_runs_on_the_tableau(self):
        t = term(40, seed=3, shots=400)
        feed(t, 'NOISE depolarizing 0.3', '10 X 0', '20 MEAS 0 -> m', '30 PRINT m;', '40 MEASURE 0', 'RUN')
        self.assertAlmostEqual(p_one(t), 0.85, delta=0.06)                   # X flips back with p/2
        t2 = term(40, seed=3, shots=10)
        out = feed(t2, 'NOISE amplitude_damping 0.5', '10 X 0', '20 MEAS 0 -> m', '30 PRINT m', 'RUN')
        self.assertIn('the noise on X is not a Pauli channel', out)


class TestAerGatesAndNoise(unittest.TestCase):
    """The gates and noise Aer added: ECR, CS, CSDG, CSX, SXDG, CCZ and RZX,
    the stabilizer method on ECR, SXDG and RZ at multiples of pi/2, and the
    Pauli-Lindblad channel."""

    def test_gate_matrices_match_qiskit(self):
        from qiskit.circuit.library import (
            CCZGate,
            CSdgGate,
            CSGate,
            CSXGate,
            ECRGate,
            RZXGate,
            SXdgGate,
        )
        for name, params, gate in (('SXDG', (), SXdgGate()), ('ECR', (), ECRGate()), ('CS', (), CSGate()),
                                   ('CSDG', (), CSdgGate()), ('CSX', (), CSXGate()), ('CCZ', (), CCZGate()),
                                   ('RZX', (0.7,), RZXGate(0.7))):
            np.testing.assert_allclose(_np_gate_matrix(name, params),
                                       Operator(gate).reverse_qargs().data, atol=1e-12, err_msg=name)

    def test_new_gates_run_and_clifford_ones_take_the_stabilizer_method(self):
        t = term(2, seed=1, shots=200)
        out = feed(t, '10 H 0', '20 ECR 0, 1', '30 RZ PI/2, 0', '40 SXDG 1', '50 MEASURE', 'RUN')
        self.assertIn('method=stabilizer', out)
        out = feed(t, '30 RZ 0.3, 0', 'RUN')
        self.assertNotIn('method=stabilizer', out)
        t3 = term(3, seed=1)
        feed(t3, '10 X 0', '20 X 1', '30 H 2', '40 CCZ 0, 1, 2', '50 H 2', '60 CS 0, 1', '70 CSX 0, 2',
             '80 RZX 0.4, 0, 1', 'RUN')
        qc = QuantumCircuit(3)
        qc.x([0, 1])
        qc.h(2)
        qc.ccz(0, 1, 2)
        qc.h(2)
        qc.cs(0, 1)
        qc.csx(0, 2)
        qc.rzx(0.4, 0, 1)
        from qiskit.quantum_info import Statevector
        self.assertAlmostEqual(abs(np.vdot(Statevector(qc).data, t3.last_sv)), 1.0, places=9)

    def test_pauli_lindblad_noise(self):
        flip = (1 - math.exp(-2 * 0.1)) / 2                  # an X generator at rate r flips (1 - e^-2r)/2
        t = term(1, seed=4, shots=6000)
        out = feed(t, 'NOISE lindblad X 0.1', '10 X 0', '20 MEASURE', 'RUN')
        self.assertIn('lindblad X:0.1', feed(t, 'NOISE INFO') + out)
        self.assertAlmostEqual(t.last_counts.get('0', 0) / 6000, flip, delta=0.015)
        t2 = term(2, seed=4, shots=6000)
        feed(t2, 'NOISE lindblad XX 0.1', '10 CX 0, 1', '20 MEASURE', 'RUN')
        self.assertAlmostEqual(t2.last_counts.get('11', 0) / 6000, flip, delta=0.015)
        t3 = term(40, seed=4, shots=1500)                     # a live run on the stabilizer tableau
        out = feed(t3, 'NOISE lindblad X 0.1', '10 X 0', '20 MEAS 0 -> m', '30 PRINT m;', '40 MEASURE 0', 'RUN')
        self.assertIn('stabilizer tableau', out)
        self.assertAlmostEqual(t3.last_counts.get('0', 0) / 1500, flip, delta=0.03)
        for bad in ('NOISE lindblad X', 'NOISE lindblad Q 0.1', 'NOISE lindblad XYZ 0.1', 'NOISE lindblad X -1'):
            self.assertIn('?', feed(term(1), bad), bad)


class TestInputAcrossShots(unittest.TestCase):

    def test_first_shot_reads_and_later_shots_replay(self):
        t = term(1, seed=3, inputs=['2'])
        t.shots = 20
        out = feed(t, '10 INPUT "reps"; r', '20 H 0', '30 MEAS 0 -> m', '40 PRINT "r ="; r; m', '50 MEASURE',
                   'RUN')
        self.assertEqual(out.count('reps?'), 1)
        self.assertEqual(out.count('r = 2'), 1)
        self.assertEqual(sum(t.last_counts.values()), 20)

    def test_shot_io(self):
        sink = CaptureIO(['typed'])
        rio = ShotIO(sink, skip=4)
        rio.write('abc')
        rio.writeln('def')
        self.assertEqual(sink.text, 'ef\n')
        self.assertEqual(rio.read_line('? '), 'typed')
        rio.record('k')
        rio.start_replay()
        rio.writeln('hidden')
        self.assertEqual((rio.read_line(''), rio.read_line('')), ('typed', 'k'))
        with self.assertRaises(EOFError):
            rio.read_line('')
        self.assertNotIn('hidden', sink.text)

    def test_buffered_io(self):
        sink = CaptureIO(['yes'])
        buf = BufferedIO(sink)
        buf.write('a')
        buf.writeln('b')
        self.assertEqual(sink.text, '')
        self.assertEqual(buf.read_line('? '), 'yes')
        self.assertTrue(sink.text.startswith('ab\n'))
        buf.writeln('dropped')
        buf.discard()
        buf.record('x')
        self.assertEqual(buf.answers, ['yes', 'x'])
        self.assertNotIn('dropped', sink.text)


# ---------------------------------------------------------------------------
# The circuit builder: DEF macros (offsets, parenthesized calls, nested
# modifiers, recursion), named registers, gate argument errors, and gates typed
# at the prompt, including LOCC register lines.
# ---------------------------------------------------------------------------


def built(t):
    qc, _ = t.build_circuit()
    return qc


class TestMacros(unittest.TestCase):

    def test_offsets_parameters_and_modifiers(self):
        t = term(4)
        feed(t, 'DEF BELL = H 0 : CX 0, 1', 'DEF ROT(a, q) = RX a, q : REM turn : BARRIER : RZ a, q',
             'DEF MIXED = CTRL X 0, 1 : INV S 0', '10 BELL @2', '20 ROT(0.5, 1)', '30 MIXED')
        qc = built(t)
        names = [i.operation.name for i in qc.data]
        self.assertEqual(names[:2], ['h', 'cx'])
        self.assertEqual([qc.find_bit(q).index for q in qc.data[1].qubits], [2, 3])
        self.assertIn('barrier', names)
        self.assertIn('rx', names)
        self.assertIn('sdg', [i.operation.name for i in qc.decompose().data] + names)
        self.assertEqual(t._offset_qubits('RX 0.5, 1', 2), 'RX 0.5 3')
        self.assertEqual(t._offset_qubits('H', 2), 'H')
        self.assertEqual(t._offset_qubits('CX q, 1', 1), 'CX q 2')

    def test_registers(self):
        t = term(4)
        feed(t, 'REG data 2', 'REG anc 2', '10 X data[1]', '20 CX data[1], anc[0]')
        qc = built(t)
        self.assertEqual([qc.find_bit(q).index for q in qc.data[1].qubits], [1, 2])
        self.assertIn('OUT OF RANGE (size=2)', feed(t, '10 X data[5]', 'RUN'))
        self.assertIn('UNKNOWN REGISTER: nope', feed(t, '10 X nope[0]', 'RUN'))

    def test_build_errors(self):
        cases = {
            'DEF LOOPY = LOOPY': 'RECURSION',
            'DEF BAD = FROB 0': 'UNKNOWN GATE: FROB',
            'DEF SHORT = CX 0': 'needs 0 param(s) and 2 qubit(s)',
            'DEF DUP = CX 1, 1': 'duplicate qubit',
            'DEF CTL = CTRL X 0': 'needs 0 parameter(s), 1 control and 1 target',
            'DEF CTLDUP = CTRL X 1, 1': 'duplicate qubit arguments in CTRL X',
            'DEF WHAT = CTRL FROB 0, 1': 'CTRL FROB: gate not found',
            'DEF FAR = H 7': 'OUT OF RANGE',
            'DEF ODD = H q?': 'CANNOT RESOLVE QUBIT',
        }
        for definition, msg in cases.items():
            with self.subTest(definition=definition):
                t = term()
                name = definition.split()[1]
                out = feed(t, definition, f'10 {name}', 'RUN')
                self.assertIn(msg, out)

    def test_gate_objects(self):
        t = term()
        feed(t, 'UNITARY W = [[0, 1], [1, 0]]')
        self.assertTrue(Operator(t._gate_object('W', [])).equiv(Operator(np.array([[0, 1], [1, 0]]))))
        with self.assertRaises(ValueError):
            t._gate_object('NOSUCH', [])
        with self.assertRaises(ValueError):
            t._apply_gate(built(t), 'NOSUCH', [], [0])


class TestImmediateMode(unittest.TestCase):

    def test_gates_at_the_prompt(self):
        t = term()
        out = feed(t, 'H 0', 'H 0 : CX 0, 1')                  # each typed line starts from |00>
        self.assertIn('|ψ⟩ = +0.707|00⟩ +0.707|01⟩', out)
        self.assertIn('+0.707|11⟩', out)
        self.assertEqual(feed(t, 'PRINT 1').strip(), '1')
        with mock.patch.dict(os.environ, {'QUBASIC_MEMORY_LIMIT': '1MB'}):
            t2 = term(30)
            self.assertIn('so it is not shown', feed(t2, 'H 0'))

    def test_locc_register_lines(self):
        t = term()
        self.assertIn('requires LOCC mode', feed(t, '@A H 0'))
        feed(t, 'LOCC 1 1')
        out = feed(t, '@A X 0 : @B H 0', '@C H 0', '@A')
        self.assertIn('UNKNOWN REGISTER: C', out)
        self.assertIn('BAD @register statement', out)
        sv = t.locc.svs['A']
        self.assertAlmostEqual(abs(complex(np.asarray(sv).reshape(-1)[1])), 1.0)


# ---------------------------------------------------------------------------
# LOCC register programs: measurements, resets, basis measurements and
# syndromes inside a register, CTRL, INV and UNITARY gates on the numpy engine,
# and the errors a register line can raise.
# ---------------------------------------------------------------------------


class TestRegisterStatements(unittest.TestCase):

    def test_measure_reset_and_bases(self):
        t = term(4)
        t.shots = 30
        feed(t, 'LOCC 3 1', '10 @A X 0', '20 @A MEAS 0 -> m', '30 @A RESET 0', '40 @A H 1',
             '50 @A MEASURE_X 1', '60 @A SDG 2', '70 @A H 2', '80 @A MEASURE_Y 2', '90 PRINT m; mx_1; my_2',
             '100 MEASURE')
        out = feed(t, 'RUN')
        self.assertNotIn('?', out)
        self.assertEqual(t.variables['m'], 1)
        self.assertEqual(t.variables['mx_1'], 0)                  # |+> reads + in the X basis
        self.assertIn(t.variables['my_2'], (0, 1))
        a_counts = {}
        for key, c in t.last_counts.items():
            a_counts[key.split('|')[0][-1]] = a_counts.get(key.split('|')[0][-1], 0) + c
        self.assertEqual(set(a_counts), {'0'})                    # qubit 0 was reset

    def test_syndrome(self):
        t = term(4)
        t.shots = 20
        feed(t, 'LOCC 3 1', '10 @A X 0', '20 @A SYNDROME ZZ 0 1 -> s', '30 PRINT "s="; s', '40 MEASURE')
        self.assertIn('s= 1', feed(t, 'RUN'))
        t2 = term(4)
        feed(t2, 'LOCC 2 1', '10 @A SYNDROME ZZ 0 1 -> s', '20 MEASURE')
        self.assertIn('needed as ancilla', feed(t2, 'RUN'))

    def test_modifiers_and_custom_gates(self):
        t = term(4)
        feed(t, 'LOCC 2 2', 'UNITARY FLIP = [[0, 1], [1, 0]]')
        feed(t, '@A FLIP 0', '@A CTRL X 0, 1', '@B H 0', '@B INV H 0', '@B CTRL FLIP 0, 1')
        a = np.asarray(t.locc.svs['A']).reshape(-1)
        self.assertAlmostEqual(abs(a[3]), 1.0)                      # |11>
        b = np.asarray(t.locc.svs['B']).reshape(-1)
        self.assertAlmostEqual(abs(b[0]), 1.0)                      # H then INV H
        out = feed(t, '@A FROB 0', '@A CX 0', '@A H 5')
        self.assertIn('UNKNOWN GATE: FROB', out)
        self.assertIn('needs 0 params + 2 qubits', out)
        self.assertIn('OUT OF RANGE', out)

    def test_lines_need_a_register(self):
        t = term(4)
        feed(t, 'LOCC 1 1', '10 H 0', '20 MEASURE')
        self.assertIn('requires @A/@B prefix', feed(t, 'RUN'))


# ---------------------------------------------------------------------------
# The memory map, run statistics and profiling, the configuration commands,
# LOCC sessions and the qudit engine: values read back, state prepared
# through POKE, files written and the errors reported.
# ---------------------------------------------------------------------------


class TestMemoryMap(unittest.TestCase):

    def test_zero_page_and_unmapped_addresses(self):
        t = term(agent=True)
        out = feed(t, 'POKE 5, 42', 'PEEK 5', 'POKE &H9000, 1', 'PEEK &H9000', 'POKE 5', 'PEEK')
        self.assertIn('$0005 = 42.0', out)
        self.assertIn('unmapped', out)
        self.assertIn('?USAGE: POKE', out)
        self.assertIn('?USAGE: PEEK', out)
        self.assertEqual(t._peek(0x9000), 0.0)
        self.assertIn('suspiciously large', feed(t, 'POKE 1, 1E16'))

    def test_qpu_configuration_round_trips(self):
        t = term(agent=True)
        for addr, value in ((0xD001, 300), (0xD002, 2), (0xD006, 77), (0xD008, 0), (0xD009, 1e-10),
                            (0xD00A, 12), (0xD00B, 0.02), (0xD00C, 1), (0xD000, 3)):
            t._poke(addr, value)
            self.assertAlmostEqual(t._peek(addr), value, msg=hex(addr))
        self.assertEqual((t.shots, t.sim_method, t.num_qubits, t.sim_precision), (300, 'stabilizer', 3, 'single'))
        t._poke(0xD003, 0)
        self.assertEqual(t._peek(0xD003), 0.0)
        t._poke(0xD000, 10 ** 6)                                 # clamped to the method's ceiling
        self.assertEqual(t.num_qubits, 4096)
        self.assertIn('?READ-ONLY', feed(t, 'POKE &HD010, 1'))

    def test_screen_mode_address_acts_as_screen(self):
        t = term(agent=True)
        t._poke(0xD007, 13)
        self.assertEqual((t.graphics.canvas.mode, t._peek(0xD007)), (13, 13.0))
        t._poke(0xD007, 0)
        self.assertIsNone(t.graphics.canvas)
        t._poke(0xD007, 3)
        self.assertEqual((t._screen_mode, t._peek(0xD007)), (3, 3.0))

    def test_noise_addresses(self):
        t = term(agent=True)
        self.assertEqual((t._peek(0xD004), t._peek(0xD005)), (0.0, 0.0))
        feed(t, 'NOISE depolarizing 0.02')
        self.assertEqual((t._peek(0xD004), t._peek(0xD005)), (1.0, 0.02))
        feed(t, 'POKE &HD100, 1', 'POKE &HD101, 0.05')
        self.assertEqual((t._peek(0xD100), t._peek(0xD101)), (1.0, 0.05))

    def test_qubit_state_block(self):
        t = term(agent=True)
        self.assertEqual(t._peek(0x0100), 0.0)                    # no state yet
        feed(t, '10 H 0', '20 X 1', 'RUN')
        self.assertAlmostEqual(t._peek(0x0100), 0.5)               # P(1) of qubit 0
        self.assertAlmostEqual(t._peek(0x0101), 1.0)               # Bloch x
        self.assertAlmostEqual(t._peek(0x0103), 0.0)               # Bloch z
        self.assertAlmostEqual(t._peek(0x0108), 1.0)               # P(1) of qubit 1
        self.assertAlmostEqual(t._peek(0x010E), 1.0)               # Re(beta) of qubit 1
        self.assertEqual(t._peek(0x0100 + 8 * 5), 0.0)             # no qubit 5
        self.assertAlmostEqual(t._peek(0xD013), 1.0)
        self.assertAlmostEqual(t._peek(0xD014), 0.0)

    def test_poking_qubit_state_prepares_the_next_run(self):
        t = term(agent=True)
        out = feed(t, 'POKE &H0100, 0.25', 'POKE &H0109, 0', 'POKE &H010A, 1', 'POKE &H010B, 0',
                   'POKE &H0104, 1', 'POKE &H0100 + 8 * 7, 1')
        self.assertIn('POKE q0: P(1)=0.2500', out)
        self.assertIn('NOT WRITABLE', out)
        self.assertIn('OUT OF RANGE', out)
        feed(t, '10 SAVE_EXPECT Z 0 -> z0', '20 SAVE_EXPECT Y 1 -> y1', 'RUN')
        self.assertAlmostEqual(t.variables['z0'], 0.5)             # P(1) = 1/4
        self.assertAlmostEqual(t.variables['y1'], 1.0)             # Bloch vector +y

    def test_sys_usr_and_catalog(self):
        t = term(agent=True)
        out = feed(t, 'DEF FLIP = X 0', 'SYS INSTALL &HF000, FLIP', 'SYS INSTALL &H1000, FLIP',
                   'SYS INSTALL &HF000', 'CATALOG')
        self.assertIn('INSTALLED FLIP AT $F000', out)
        self.assertIn('USER SYS RANGE', out)
        self.assertIn('?USAGE: SYS INSTALL', out)
        self.assertIn('$F000  FLIP', out)
        out = feed(t, 'SYS &HE000', 'SYS &HF000', 'SYS &HE123')
        self.assertIn('?NO ROUTINE AT $E123', out)
        feed(t, 'SYS INSTALL &HF010, MISSING')
        self.assertIn('?UNDEFINED ROUTINE', feed(t, 'SYS &HF010'))
        t2 = term(agent=True)
        feed(t2, 'DEF SAVER = SAVE x.qb', 'SYS INSTALL &HF020, SAVER')
        self.assertEqual(t2._usr_fn(0xF020), 0.0)
        self.assertIn('BLOCKED IN USR: SAVE', t2.io.text)
        self.assertIn(t2._usr_fn(0xE000), (0.0, 3.0))               # BELL reads 00 or 11

    def test_wait_dump_map_monitor(self):
        t = term(agent=True)
        feed(t, 'POKE 3, 6')
        self.assertEqual(feed(t, 'WAIT 3, 2'), '')
        self.assertIn('?WAIT TIMEOUT', feed(t, 'WAIT 3, 1, 1, 0.1'))
        self.assertIn('?USAGE: WAIT', feed(t, 'WAIT 3'))
        out = feed(t, 'DUMP 0 15')
        self.assertIn('$0000: 00 00 00 06', out)
        feed(t, '10 H 0', '20 H 1', 'RUN')
        out = feed(t, 'MAP')
        self.assertIn('q0: P(1)=0.500', out)
        self.assertIn('$D00C precision', out)
        t2 = term(inputs=['3', '7=2', 'oops=', '?', 'Q'], agent=True)
        out = feed(t2, 'MONITOR')
        self.assertIn('$0007 <- 2.0', out)
        self.assertEqual(t2.machine.zero_page[7], 2.0)
        self.assertIn('MONITOR OFF', out)


class TestStatsAndProfile(unittest.TestCase):

    def test_stats_accumulate_and_export(self):
        t = term(agent=True)
        feed(t, '10 H 0', '20 MEASURE')
        out = feed(t, 'STATS', 'STATS 0', 'STATS many', 'STATS CSV')
        self.assertIn('No statistics collected', out)
        self.assertIn('at least 1 run', out)
        self.assertIn('?USAGE: STATS', out)
        out = feed(t, 'STATS 4', 'STATS SHOW')
        self.assertIn('Collected 4 runs', out)
        self.assertIn('Statistics over 4 runs', out)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                out = feed(t, 'STATS CSV', 'STATS CSV stats.csv')
                self.assertIn('?USAGE: STATS CSV', out)
                self.assertIn('STATS EXPORTED', out)
                rows = open('stats.csv').read().splitlines()
                self.assertEqual(rows[0], 'state,mean,stddev,min,max,probability')
                self.assertEqual(len(rows), 3)
            finally:
                os.chdir(cwd)
        self.assertIn('STATS CLEARED', feed(t, 'STATS CLEAR'))

    def test_profile(self):
        t = term(agent=True)
        feed(t, '10 H 0', '20 CX 0, 1', '30 MEASURE')
        out = feed(t, 'PROFILE ON', 'RUN', 'PROFILE SHOW', 'PROFILE OFF', 'PROFILE')
        self.assertIn('PROFILE ON', out)
        self.assertIn('PROFILE OFF', out)


class TestConfiguration(unittest.TestCase):

    def test_method_names_aliases_and_probe(self):
        t = term(40, agent=True)
        t.sim_method = 'stabilizer'
        out = feed(t, 'METHOD sv')
        self.assertIn('QUBITS clamped to 32', out)
        for alias, name in (('mps', 'matrix_product_state'), ('dm', 'density_matrix'),
                            ('clifford', 'stabilizer'), ('auto', 'automatic')):
            feed(t, f'METHOD {alias}')
            self.assertEqual(t.sim_method, name)
        out = feed(t, 'METHOD stabilzer')
        self.assertIn('did you mean stabilizer', out)
        out = feed(t, 'METHOD')
        self.assertIn('methods: automatic', out)
        self.assertIn('devices: CPU', out)
        out = feed(t, 'METHOD CPU')
        self.assertIn('DEVICE = CPU', out)

    def test_precision_coupling_basis(self):
        t = term(4, agent=True)
        out = feed(t, 'PRECISION', 'PRECISION SINGLE', 'PRECISION HALF')
        self.assertIn('PRECISION = DOUBLE', out)
        self.assertIn('widest statevector run', out)
        self.assertIn('SINGLE or DOUBLE', out)
        for spec, edges in (('linear', 3), ('ring', 4), ('0-1, 1-2', 2)):
            out = feed(t, f'COUPLING {spec}')
            self.assertIn(f'({edges} edge(s))', out)
        self.assertIn('heavyhex', feed(t, 'COUPLING heavyhex'))
        self.assertIn("?bad edge 'x'", feed(t, 'COUPLING x'))
        self.assertIn('all-to-all', feed(t, 'COUPLING OFF', 'COUPLING'))
        out = feed(t, 'BASIS rz, sx, x, cx', 'BASIS', 'BASIS OFF')
        self.assertIn("['rz', 'sx', 'x', 'cx']", out)
        self.assertIn('default (all gates)', out)
        t3 = term(agent=True)
        self.assertIn('(1 edge(s))', feed(t3, 'COUPLING ring'))

    def test_routed_runs_respect_the_coupling_map(self):
        t = term(3, agent=True)
        feed(t, 'COUPLING linear', 'BASIS rz, sx, x, cx', '10 H 0', '20 CX 0, 2', '30 MEASURE', 'RUN')
        ops = set(t._last_transpiled.count_ops()) - {'measure', 'barrier'}
        self.assertLessEqual(ops, {'rz', 'sx', 'x', 'cx'})
        self.assertEqual(set(t.last_counts), {'000', '101'})

    def test_status(self):
        t = term(agent=True)
        feed(t, 'NOISE depolarizing 0.01', 'COUPLING linear', 'BASIS cx, rz, sx', 'SET_STATE |BELL>',
             'DISPLAY STATE', 'TRON', 'SCREEN 13', 'LOCC 1 1', 'UNITARY G = [[0, 1], [1, 0]]')
        out = feed(t, 'STATUS')
        for text in ('noise        on', 'coupling', 'basis gates', 'pending      SET_STATE',
                     'display      statevector', 'graphics     SCREEN 13', 'trace        on', 'LOCC mode',
                     "custom gates ['G']"):
            self.assertIn(text, out)
        d = json.loads(feed(t, 'STATUS JSON'))
        self.assertEqual((d['graphics_screen'], d['screen_mode'], d['locc_registers']), (13, 2, ['A', 'B']))

    def test_seed_and_version(self):
        t = term(agent=True)
        out = feed(t, 'SEED 9', 'SEED OFF', 'VERSION')
        self.assertIn('SEED = 9', out)
        self.assertIn('SEED OFF', out)
        self.assertIn('Qiskit Aer', out)
        self.assertIsNone(t._seed)

    def test_qubits_reports(self):
        t = term(agent=True)
        out = feed(t, 'QUBITS', 'QUBITS 20', 'QUBITS 40')
        self.assertIn('QUBITS = 2', out)
        self.assertIn('Max parallel instances', out)
        self.assertIn('exceeds', out)
        t.sim_method = 'statevector'
        out = feed(t, 'QUBITS 31', 'QUBITS 33')
        self.assertIn('RUN will refuse', out)
        self.assertIn('RANGE: 1-32', out)
        for method in ('stabilizer', 'matrix_product_state'):
            t.sim_method = method
            self.assertIn('ALLOCATED', feed(t, 'QUBITS 50'))


class TestLOCCCommands(unittest.TestCase):

    def test_session_lifecycle(self):
        t = term(agent=True)
        out = feed(t, 'LOCC', 'LOCC STATUS', 'LOCC 2 2', 'LOCC', 'LOCC STATUS', 'LOCC SHOTCAP',
                   'LOCC SHOTCAP 50', 'LOCC SHOTCAP x', 'LOCC SHOTCAP OFF', 'LOCC OFF')
        for text in ('LOCC OFF. Usage', 'NOT IN LOCC MODE', 'LOCC SPLIT: A=2q  B=2q', 'RAM per instance',
                     'LOCC SHOTCAP = auto', 'LOCC SHOTCAP = 50', '?USAGE: LOCC SHOTCAP', 'SHOTCAP OFF',
                     'back to normal Aer mode'):
            self.assertIn(text, out)
        self.assertFalse(t.locc_mode)

    def test_teleportation_with_send(self):
        t = term(agent=True)
        t.shots = 200
        feed(t, 'LOCC JOINT 2 1', '10 @A RY 1.2, 0', '20 SHARE A 1, B 0', '30 @A CX 0, 1', '40 @A H 0',
             '50 SEND A 0 -> m0', '60 SEND A 1 -> m1', '70 IF m1 THEN @B X 0', '80 IF m0 THEN @B Z 0',
             '90 MEASURE')
        out = feed(t, 'RUN', 'LOCCINFO')
        self.assertNotIn('?', out)
        self.assertIn('Joint', out)
        ones = sum(c for k, c in t.last_counts.items() if k.split('|')[-1] == '1') / t.shots
        self.assertAlmostEqual(ones, math.sin(0.6) ** 2, delta=0.12)

    def test_errors(self):
        t = term(agent=True)
        out = feed(t, 'LOCC 2', 'LOCC 2 banana', 'SEND A 0 -> m', 'SHARE A 0, B 0')
        for text in ('?USAGE: LOCC', 'sizes must be integers', '?SEND requires LOCC mode',
                     '?SHARE requires LOCC mode'):
            self.assertIn(text, out)
        feed(t, 'LOCC 1 1')
        self.assertIn('?SHARE requires LOCC JOINT mode', feed(t, 'SHARE A 0, B 0'))


class TestQudits(unittest.TestCase):

    def test_gates_and_measurement(self):
        t = term(1)
        out = feed(t, 'QUDIT', 'QSTATE', 'QMEASURE', 'QUDIT 1', 'QUDIT 40', 'QUDIT 3 20')
        for text in ('?USAGE: QUDIT', '?NO QUDIT STATE', 'dimension must be 2-36', 'too large'):
            self.assertIn(text, out)
        feed(t, 'QUDIT 3 2', 'QF 0', 'QSUM 0 1')
        np.testing.assert_allclose(np.abs(t.qudit.sv[[0, 4, 8]]) ** 2, [1 / 3] * 3, atol=1e-12)
        out = feed(t, 'QSTATE', 'QMEASURE 300', 'QUDIT')
        self.assertIn('|11>', out)
        self.assertIn('QMEASURE (300 shots, base-3)', out)
        self.assertIn('QUDIT: 2 qudits of dimension 3', out)
        feed(t, 'QUDIT 5 1', 'QX 0 2', 'QZ 0')
        self.assertAlmostEqual(complex(t.qudit.sv[2]), np.exp(2j * math.pi * 2 / 5))
        out = feed(t, 'QX 3', 'QZ', 'QF 9', 'QSUM 0')
        for text in ('?QX ERROR', '?QZ ERROR', '?QF ERROR', '?QSUM ERROR'):
            self.assertIn(text, out)
        t2 = term(1)
        self.assertIn('no qudit register', feed(t2, 'QX 0'))


if __name__ == '__main__':
    unittest.main()
