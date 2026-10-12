"""Noise models: channels on every gate width, composition, NOISE ADD, LOCC propagation."""

import unittest

import numpy as np

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


def term(nq=2, shots=2000, seed=4):
    t = QBasicTerminal()
    t.io = _IO()
    t.num_qubits = nq
    t.shots = shots
    t._seed = seed
    return t


def run(t, lines):
    for k, line in enumerate(lines):
        t.process(f'{(k + 1) * 10} {line}', track_undo=False)
    t.cmd_run()
    return t.last_counts


class TestChannels(unittest.TestCase):

    def test_separate_two_qubit_depolarizing_rate(self):
        t = term()
        t.cmd_noise('depolarizing 0 0.3')
        counts = run(t, ['X 0', 'X 1', 'MEASURE'])
        self.assertEqual(counts, {'11': 2000})          # one-qubit gates noiseless
        t2 = term()
        t2.cmd_noise('depolarizing 0 0.3')
        counts = run(t2, ['X 0', 'CX 0, 1', 'MEASURE'])
        self.assertGreater(sum(c for k, c in counts.items() if k != '11'), 100)

    def test_single_qubit_channels_reach_multi_qubit_gates(self):
        for spec in ('amplitude_damping 0.1', 'phase_flip 0.1', 'thermal 50 70 1 5',
                     'combined 0.1 0.1', 'pauli 0.01 0.02 0.03', 'reset 0.01 0.01'):
            with self.subTest(spec=spec):
                t = term()
                t.cmd_noise(spec)
                errors = t._noise_model._default_quantum_errors
                self.assertEqual(errors['cx'].num_qubits, 2)
                self.assertEqual(errors['ccx'].num_qubits, 3)
                self.assertEqual(errors['h'].num_qubits, 1)

    def test_amplitude_damping_on_cx_decays(self):
        t = term()
        t.cmd_noise('amplitude_damping 0.4')
        counts = run(t, ['X 0', 'X 1', 'CX 0, 1', 'CX 0, 1', 'MEASURE'])
        self.assertGreater(counts.get('00', 0) + counts.get('01', 0) + counts.get('10', 0), 600)

    def test_parameter_validation(self):
        t = term()
        t.cmd_noise('thermal 10 50 1')
        self.assertIn('T2 <= 2 T1', t.io.text)
        self.assertIsNone(t._noise_model)
        t.cmd_noise('pauli 0.5 0.5 0.5')
        self.assertIn('sum past 1', t.io.text)
        t.cmd_noise('sparkle 0.1')
        self.assertIn('UNKNOWN NOISE TYPE', t.io.text)
        self.assertIsNone(t._noise_model)


class TestComposition(unittest.TestCase):

    def test_semicolon_composes_and_add_extends(self):
        t = term()
        t.cmd_noise('depolarizing 0.01; readout 0.05')
        self.assertEqual(t._noise_spec, 'depolarizing 0.01; readout 0.05')
        self.assertIsNotNone(t._noise_model._default_readout_error)
        t.cmd_noise('ADD crosstalk 0.02')
        self.assertEqual(t._noise_spec, 'depolarizing 0.01; readout 0.05; crosstalk 0.02')
        t.cmd_noise('INFO')
        self.assertIn('crosstalk 0.02', t.io.text)

    def test_readout_errors_compose(self):
        t = term()
        t.cmd_noise('readout 0.1 0.2; readout 0.3 0.1')
        p = np.asarray(t._noise_model._default_readout_error.probabilities)
        want = np.array([[0.9, 0.1], [0.2, 0.8]]) @ np.array([[0.7, 0.3], [0.1, 0.9]])
        np.testing.assert_allclose(p, want)

    def test_add_without_a_model_starts_one(self):
        t = term()
        t.cmd_noise('ADD readout 0.1')
        self.assertEqual(t._noise_spec, 'readout 0.1')

    def test_composed_noise_runs_live(self):
        t = term(nq=1, shots=400)
        t.cmd_noise('depolarizing 0.05; readout 0.1')
        run(t, ['H 0', 'MEAS 0 -> m', 'PRINT m', 'MEASURE'])
        self.assertIn('noise=trajectories', t.io.text)
        self.assertEqual(sum(t.last_counts.values()), 400)


class TestLoccPropagation(unittest.TestCase):

    def test_single_channel_reaches_locc_composite_does_not(self):
        t = term()
        t.cmd_noise('phase_flip 0.2')
        t.cmd_locc('JOINT 1 1')
        self.assertEqual((t.locc.noise_type, t.locc.noise_param), ('phase_flip', 0.2))
        t.cmd_noise('depolarizing 0.1; readout 0.1')
        self.assertEqual(t.locc.noise_type, 'none')
        t.cmd_noise('OFF')
        self.assertEqual(t.locc.noise_param, 0.0)


if __name__ == '__main__':
    unittest.main()
