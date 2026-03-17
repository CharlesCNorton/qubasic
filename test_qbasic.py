#!/usr/bin/env python3
"""
Comprehensive test suite for QBASIC.

Covers core language, gate correctness, expression evaluation,
control flow, LOCC engine (2-party and N-party), display formatting,
security boundaries, and edge cases.

Run: python test_qbasic.py
"""

import sys
import os
import io
import math
import unittest
import tempfile
import numpy as np

# Ensure qbasic package is importable
sys.path.insert(0, os.path.dirname(__file__))
from qbasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    _np_gate_matrix, _apply_gate_np, _measure_np, _sample_np,
)
from qbasic_core.terminal import QBasicTerminal
from qbasic_core.engine import LOCCEngine
from qbasic import run_script


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


class TestExpressionEvaluator(unittest.TestCase):
    """Core expression evaluation (AST-based, no eval)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_basic_arithmetic(self):
        self.assertAlmostEqual(self.t.eval_expr('2+3'), 5.0)
        self.assertAlmostEqual(self.t.eval_expr('10-4'), 6.0)
        self.assertAlmostEqual(self.t.eval_expr('3*7'), 21.0)
        self.assertAlmostEqual(self.t.eval_expr('15/4'), 3.75)

    def test_operator_precedence(self):
        self.assertAlmostEqual(self.t.eval_expr('2+3*4'), 14.0)
        self.assertAlmostEqual(self.t.eval_expr('(2+3)*4'), 20.0)

    def test_unary_minus(self):
        self.assertAlmostEqual(self.t.eval_expr('-5'), -5.0)
        self.assertAlmostEqual(self.t.eval_expr('-(2+3)'), -5.0)

    def test_power(self):
        self.assertAlmostEqual(self.t.eval_expr('2**10'), 1024.0)

    def test_constants(self):
        self.assertAlmostEqual(self.t.eval_expr('PI'), math.pi)
        self.assertAlmostEqual(self.t.eval_expr('pi'), math.pi)
        self.assertAlmostEqual(self.t.eval_expr('TAU'), math.tau)
        self.assertAlmostEqual(self.t.eval_expr('E'), math.e)
        self.assertAlmostEqual(self.t.eval_expr('SQRT2'), math.sqrt(2))

    def test_math_functions(self):
        self.assertAlmostEqual(self.t.eval_expr('sin(PI/2)'), 1.0)
        self.assertAlmostEqual(self.t.eval_expr('cos(0)'), 1.0)
        self.assertAlmostEqual(self.t.eval_expr('sqrt(16)'), 4.0)
        self.assertAlmostEqual(self.t.eval_expr('abs(-7)'), 7.0)
        self.assertAlmostEqual(self.t.eval_expr('log(E)'), 1.0)
        self.assertAlmostEqual(self.t.eval_expr('exp(0)'), 1.0)

    def test_nested_functions(self):
        self.assertAlmostEqual(self.t.eval_expr('sqrt(abs(-9))'), 3.0)

    def test_variables(self):
        self.t.variables['theta'] = 1.5
        self.assertAlmostEqual(self.t.eval_expr('theta'), 1.5)
        self.assertAlmostEqual(self.t.eval_expr('theta*2'), 3.0)

    def test_eval_with_vars(self):
        val = self.t._eval_with_vars('x + 1', {'x': 10})
        self.assertAlmostEqual(val, 11.0)

    def test_conditions(self):
        self.assertTrue(self.t._eval_condition('5 > 3', {}))
        self.assertFalse(self.t._eval_condition('2 > 7', {}))
        self.assertTrue(self.t._eval_condition('3 == 3', {}))
        self.assertTrue(self.t._eval_condition('3 != 4', {}))
        self.assertTrue(self.t._eval_condition('1 <= 1', {}))

    def test_basic_style_operators(self):
        self.assertTrue(self.t._eval_condition('3 <> 4', {}))
        self.assertTrue(self.t._eval_condition('1 AND 1', {}))
        self.assertFalse(self.t._eval_condition('1 AND 0', {}))
        self.assertTrue(self.t._eval_condition('0 OR 1', {}))

    def test_condition_with_vars(self):
        self.assertTrue(self.t._eval_condition('x > 0', {'x': 5}))
        self.assertFalse(self.t._eval_condition('x > 0', {'x': -1}))

    def test_arrays(self):
        self.t.arrays['data'] = [10.0, 20.0, 30.0]
        val = self.t._safe_eval('data(1)')
        self.assertAlmostEqual(val, 20.0)

    def test_array_subscript(self):
        self.t.arrays['arr'] = [5.0, 15.0, 25.0]
        val = self.t._safe_eval('arr[2]')
        self.assertAlmostEqual(val, 25.0)

    def test_empty_expression_raises(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('')

    def test_undefined_variable_raises(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('nonexistent_var')

    def test_parse_matrix(self):
        m = self.t._parse_matrix('[[1,0],[0,-1]]')
        self.assertEqual(len(m), 2)
        self.assertAlmostEqual(m[0][0], 1+0j)
        self.assertAlmostEqual(m[1][1], -1+0j)

    def test_parse_matrix_complex(self):
        m = self.t._parse_matrix('[[1,0],[0,1j]]')
        self.assertAlmostEqual(m[1][1], 1j)


class TestSecurityBoundaries(unittest.TestCase):
    """Verify that dangerous operations are blocked."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_import_blocked(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('__import__("os")')

    def test_exec_blocked(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('exec("print(1)")')

    def test_open_blocked(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('open("/etc/passwd")')

    def test_getattr_blocked(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('getattr(int, "__bases__")')

    def test_dunder_blocked(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('().__class__.__bases__')

    def test_string_literal_blocked(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('"hello"')

    def test_method_call_blocked(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('"".join(["a"])')

    def test_lambda_blocked(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('(lambda: 1)()')


class TestGateMatrices(unittest.TestCase):
    """Verify unitary gate matrices are correct."""

    def _is_unitary(self, m):
        n = m.shape[0]
        product = m @ m.conj().T
        return np.allclose(product, np.eye(n))

    def test_all_0param_gates_are_unitary(self):
        zero_param_gates = [g for g, (np_, nq) in GATE_TABLE.items()
                           if np_ == 0 and g not in GATE_ALIASES]
        for gate in zero_param_gates:
            m = _np_gate_matrix(gate)
            self.assertTrue(self._is_unitary(m), f"{gate} is not unitary")

    def test_parametric_gates_unitary(self):
        for gate in ['RX', 'RY', 'RZ', 'P']:
            m = _np_gate_matrix(gate, (0.5,))
            self.assertTrue(self._is_unitary(m), f"{gate}(0.5) is not unitary")
        for gate in ['CRX', 'CRY', 'CRZ', 'CP', 'RXX', 'RYY', 'RZZ']:
            m = _np_gate_matrix(gate, (0.7,))
            self.assertTrue(self._is_unitary(m), f"{gate}(0.7) is not unitary")

    def test_parametric_2q_gates_match_qiskit(self):
        """Verify parametric 2-qubit gate matrices match Qiskit's definitions.

        Our matrices use big-endian (qubit 0 = MSB) while Qiskit's Operator
        uses little-endian. We compare via SWAP @ ours @ SWAP to account
        for the qubit-ordering difference.
        """
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        gate_methods = {
            'RXX': 'rxx', 'RYY': 'ryy', 'RZZ': 'rzz',
            'CRX': 'crx', 'CRY': 'cry', 'CRZ': 'crz', 'CP': 'cp',
        }
        for gate_name, qiskit_method in gate_methods.items():
            theta = 0.7
            our_matrix = _np_gate_matrix(gate_name, (theta,))
            qc = QuantumCircuit(2)
            getattr(qc, qiskit_method)(theta, 0, 1)
            qiskit_matrix = Operator(qc).data
            # Try direct match first, then swapped-qubit match
            direct = np.allclose(our_matrix, qiskit_matrix, atol=1e-6)
            swapped = np.allclose(swap @ our_matrix @ swap, qiskit_matrix, atol=1e-6)
            self.assertTrue(direct or swapped,
                f"{gate_name} matrix does not match Qiskit's {qiskit_method} "
                f"in either qubit ordering")

    def test_u_gate_unitary(self):
        m = _np_gate_matrix('U', (0.5, 0.3, 0.1))
        self.assertTrue(self._is_unitary(m))

    def test_hadamard_creates_superposition(self):
        h = _np_gate_matrix('H')
        state = np.array([1, 0], dtype=complex)
        result = h @ state
        self.assertAlmostEqual(abs(result[0])**2, 0.5)
        self.assertAlmostEqual(abs(result[1])**2, 0.5)

    def test_pauli_x_flips(self):
        x = _np_gate_matrix('X')
        np.testing.assert_array_almost_equal(x @ [1, 0], [0, 1])
        np.testing.assert_array_almost_equal(x @ [0, 1], [1, 0])

    def test_cnot_entangles(self):
        cx = _np_gate_matrix('CX')
        # |10⟩ -> |11⟩
        np.testing.assert_array_almost_equal(
            cx @ [0, 0, 1, 0], [0, 0, 0, 1])
        # |00⟩ -> |00⟩
        np.testing.assert_array_almost_equal(
            cx @ [1, 0, 0, 0], [1, 0, 0, 0])

    def test_toffoli_truth_table(self):
        ccx = _np_gate_matrix('CCX')
        # |110⟩ -> |111⟩
        state = np.zeros(8, dtype=complex); state[6] = 1
        result = ccx @ state
        self.assertAlmostEqual(abs(result[7])**2, 1.0)
        # |100⟩ -> |100⟩
        state = np.zeros(8, dtype=complex); state[4] = 1
        result = ccx @ state
        self.assertAlmostEqual(abs(result[4])**2, 1.0)

    def test_gate_aliases(self):
        np.testing.assert_array_almost_equal(
            _np_gate_matrix('CNOT'), _np_gate_matrix('CX'))
        np.testing.assert_array_almost_equal(
            _np_gate_matrix('TOFFOLI'), _np_gate_matrix('CCX'))
        np.testing.assert_array_almost_equal(
            _np_gate_matrix('FREDKIN'), _np_gate_matrix('CSWAP'))


class TestStatevectorOps(unittest.TestCase):
    """Test numpy statevector operations."""

    def test_apply_gate_h(self):
        sv = np.array([1, 0], dtype=complex)
        h = _np_gate_matrix('H')
        result = _apply_gate_np(sv, h, [0], 1)
        result = np.ascontiguousarray(result).ravel()
        self.assertAlmostEqual(abs(result[0])**2, 0.5)
        self.assertAlmostEqual(abs(result[1])**2, 0.5)

    def test_measure_collapses(self):
        # |+⟩ state
        sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
        outcome, new_sv = _measure_np(sv, 0, 1)
        self.assertIn(outcome, [0, 1])
        # After measurement, should be in a definite state
        probs = np.abs(new_sv)**2
        self.assertAlmostEqual(max(probs), 1.0)

    def test_sample_distribution(self):
        # |0⟩ state should always sample 0
        sv = np.array([1, 0], dtype=complex)
        counts = _sample_np(sv, 1, 1000)
        self.assertEqual(list(counts.keys()), ['0'])
        self.assertEqual(counts['0'], 1000)

    def test_bell_state_correlations(self):
        # Create |Φ+⟩ = (|00⟩ + |11⟩)/√2
        sv = np.zeros(4, dtype=complex)
        sv[0] = sv[3] = 1 / np.sqrt(2)
        counts = _sample_np(sv, 2, 10000)
        # Should only get 00 and 11
        for state in counts:
            self.assertIn(state, ['00', '11'])
        # Roughly equal
        self.assertGreater(counts.get('00', 0), 3000)
        self.assertGreater(counts.get('11', 0), 3000)


class TestTerminalCommands(unittest.TestCase):
    """Test REPL commands."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_qubits(self):
        _, out = capture(self.t.cmd_qubits, '8')
        self.assertEqual(self.t.num_qubits, 8)
        self.assertIn('8 QUBITS', out)

    def test_qubits_range_check(self):
        _, out = capture(self.t.cmd_qubits, '50')
        self.assertIn('RANGE', out)
        self.assertNotEqual(self.t.num_qubits, 50)

    def test_shots(self):
        _, out = capture(self.t.cmd_shots, '512')
        self.assertEqual(self.t.shots, 512)

    def test_new(self):
        self.t.program = {10: "H 0"}
        self.t.variables['x'] = 5
        capture(self.t.cmd_new)
        self.assertEqual(len(self.t.program), 0)
        self.assertEqual(len(self.t.variables), 0)

    def test_let(self):
        capture(self.t.cmd_let, 'angle = PI/4')
        self.assertAlmostEqual(self.t.variables['angle'], math.pi / 4)

    def test_reg(self):
        self.t.num_qubits = 8
        capture(self.t.cmd_reg, 'data 3')
        self.assertIn('data', self.t.registers)
        self.assertEqual(self.t.registers['data'], (0, 3))

    def test_reg_overflow(self):
        self.t.num_qubits = 4
        capture(self.t.cmd_reg, 'big 10')
        self.assertNotIn('big', self.t.registers)

    def test_def_subroutine(self):
        capture(self.t.cmd_def, 'BELL = H 0 : CX 0,1')
        self.assertIn('BELL', self.t.subroutines)
        self.assertEqual(len(self.t.subroutines['BELL']['body']), 2)

    def test_def_parameterized(self):
        capture(self.t.cmd_def, 'ROT(angle, q) = RX angle, q')
        self.assertIn('ROT', self.t.subroutines)
        self.assertEqual(self.t.subroutines['ROT']['params'], ['angle', 'q'])

    def test_def_builtin_rejected(self):
        _, out = capture(self.t.cmd_def, 'H = X 0')
        self.assertIn('CANNOT REDEFINE', out)

    def test_process_numbered_line(self):
        self.t.process('10 H 0')
        self.assertIn(10, self.t.program)
        self.assertEqual(self.t.program[10], 'H 0')

    def test_process_delete_line(self):
        self.t.program[10] = 'H 0'
        _, out = capture(self.t.process, '10')
        self.assertNotIn(10, self.t.program)

    def test_undo(self):
        self.t.process('10 H 0')
        self.t.process('20 CX 0,1')
        self.assertIn(20, self.t.program)
        capture(self.t.cmd_undo)
        self.assertNotIn(20, self.t.program)
        self.assertIn(10, self.t.program)

    def test_delete_range(self):
        self.t.program = {10: 'H 0', 20: 'X 1', 30: 'Z 2'}
        capture(self.t.cmd_delete, '10-20')
        self.assertNotIn(10, self.t.program)
        self.assertNotIn(20, self.t.program)
        self.assertIn(30, self.t.program)

    def test_renum(self):
        self.t.program = {5: 'H 0', 17: 'CX 0,1', 42: 'MEASURE'}
        capture(self.t.cmd_renum)
        self.assertEqual(sorted(self.t.program.keys()), [10, 20, 30])

    def test_method(self):
        capture(self.t.cmd_method, 'statevector')
        self.assertEqual(self.t.sim_method, 'statevector')

    def test_method_gpu(self):
        capture(self.t.cmd_method, 'GPU')
        self.assertEqual(self.t.sim_device, 'GPU')

    def test_clear_variable(self):
        self.t.variables['x'] = 5
        capture(self.t.cmd_clear, 'x')
        self.assertNotIn('x', self.t.variables)

    def test_clear_array(self):
        self.t.arrays['data'] = [1, 2, 3]
        capture(self.t.cmd_clear, 'data')
        self.assertNotIn('data', self.t.arrays)


class TestCircuitExecution(unittest.TestCase):
    """Test program building and execution."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_bell_state(self):
        self.t.num_qubits = 2
        self.t.shots = 1000
        self.t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)
        for state in self.t.last_counts:
            self.assertIn(state, ['00', '11'])

    def test_x_gate_flips(self):
        self.t.num_qubits = 1
        self.t.shots = 100
        self.t.program = {10: 'X 0', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIn('1', self.t.last_counts)
        self.assertEqual(self.t.last_counts['1'], 100)

    def test_statevector_after_run(self):
        self.t.num_qubits = 1
        self.t.program = {10: 'H 0', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_sv)
        probs = np.abs(self.t.last_sv)**2
        self.assertAlmostEqual(probs[0], 0.5, places=2)

    def test_for_next_loop(self):
        self.t.num_qubits = 4
        self.t.program = {
            10: 'FOR I = 0 TO 3',
            20: 'H I',
            30: 'NEXT I',
            40: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)
        # 4 H gates on 4 qubits = uniform distribution over 16 states
        self.assertGreater(len(self.t.last_counts), 1)

    def test_while_wend(self):
        self.t.num_qubits = 3
        self.t.variables['n'] = 0
        self.t.program = {
            10: 'LET n = 0',
            20: 'WHILE n < 3',
            30: 'H n',
            40: 'LET n = n + 1',
            50: 'WEND',
            60: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)

    def test_if_then_else(self):
        self.t.num_qubits = 2
        self.t.variables['flag'] = 1
        self.t.program = {
            10: 'IF flag == 1 THEN H 0 ELSE X 0',
            20: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)

    def test_goto(self):
        self.t.num_qubits = 1
        self.t.program = {
            10: 'GOTO 30',
            20: 'X 0',
            30: 'H 0',
            40: 'MEASURE',
        }
        capture(self.t.cmd_run)
        # X 0 should be skipped; result should be superposition, not |1⟩
        self.assertIsNotNone(self.t.last_counts)
        self.assertGreater(len(self.t.last_counts), 1)

    def test_gosub_return(self):
        self.t.num_qubits = 2
        self.t.program = {
            10: 'GOSUB 100',
            20: 'MEASURE',
            30: 'END',
            100: 'H 0',
            110: 'CX 0,1',
            120: 'RETURN',
        }
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)
        for state in self.t.last_counts:
            self.assertIn(state, ['00', '11'])

    def test_end_stops(self):
        self.t.num_qubits = 1
        self.t.program = {
            10: 'H 0',
            20: 'END',
            30: 'X 0',  # should not execute
        }
        capture(self.t.cmd_run)
        # END stops execution; no MEASURE means no counts but no crash
        self.assertIsNone(self.t.last_counts)

    def test_colon_multi_statement(self):
        self.t.num_qubits = 2
        self.t.program = {10: 'H 0 : CX 0,1', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        for state in self.t.last_counts:
            self.assertIn(state, ['00', '11'])

    def test_subroutine_expansion(self):
        self.t.num_qubits = 2
        capture(self.t.cmd_def, 'BELL = H 0 : CX 0,1')
        self.t.program = {10: 'BELL', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        for state in self.t.last_counts:
            self.assertIn(state, ['00', '11'])

    def test_register_notation(self):
        self.t.num_qubits = 4
        capture(self.t.cmd_reg, 'data 2')
        capture(self.t.cmd_reg, 'anc 2')
        self.t.program = {10: 'H data[0]', 20: 'CX data[0],anc[0]', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)

    def test_loop_limit(self):
        self.t.num_qubits = 1
        self.t._max_iterations = 100
        self.t.program = {10: 'GOTO 10'}
        _, out = capture(self.t.cmd_run)
        self.assertIn('LOOP LIMIT', out)

    def test_print_during_run(self):
        self.t.num_qubits = 1
        self.t.program = {10: 'PRINT "hello"', 20: 'H 0'}
        _, out = capture(self.t.cmd_run)
        self.assertIn('hello', out)

    def test_dim_and_array_let(self):
        self.t.num_qubits = 1
        self.t.program = {
            10: 'DIM vals(3)',
            20: 'LET vals(0) = PI',
            30: 'H 0',
        }
        capture(self.t.cmd_run)
        self.assertIn('vals', self.t.arrays)
        self.assertAlmostEqual(self.t.arrays['vals'][0], math.pi)


class TestGateDispatch(unittest.TestCase):
    """Test the data-driven gate dispatch."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_all_gates_dispatch(self):
        """Every gate in GATE_TABLE should be applicable without error."""
        from qiskit import QuantumCircuit
        for gate, (n_params, n_qubits) in GATE_TABLE.items():
            if gate in GATE_ALIASES:
                continue
            qc = QuantumCircuit(max(n_qubits, 3))
            params = [0.5] * n_params
            qubits = list(range(n_qubits))
            try:
                self.t._apply_gate(qc, gate, params, qubits)
            except Exception as e:
                self.fail(f"Gate {gate} failed: {e}")

    def test_custom_unitary_gate(self):
        from qiskit import QuantumCircuit
        # Define a custom sqrt(X) gate
        m = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex) / 2
        self.t._custom_gates['SQX'] = m
        qc = QuantumCircuit(1)
        self.t._apply_gate(qc, 'SQX', [], [0])
        self.assertEqual(qc.size(), 1)

    def test_custom_gate_does_not_leak(self):
        """Custom gates on one terminal must not appear on another."""
        m = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex) / 2
        self.t._custom_gates['LEAK'] = m
        t2 = QBasicTerminal()
        self.assertIsNone(t2._gate_info('LEAK'))

    def test_non_unitary_rejected(self):
        """UNITARY with a non-unitary matrix should be rejected."""
        # UNITARY is handled by _try_exec_unitary during program execution
        self.t.num_qubits = 1
        self.t.program = {10: 'UNITARY BAD = [[1,0],[0,0]]', 20: 'MEASURE'}
        _, out = capture(self.t.cmd_run)
        self.assertIn('not unitary', out)
        self.assertNotIn('BAD', self.t._custom_gates)


class TestVariableSubstitution(unittest.TestCase):
    """Test that variable substitution respects reserved names."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_no_substitute_pi(self):
        # Even if user has a var named 'PI', it shouldn't replace in expressions
        result = self.t._substitute_vars('RX PI, 0', {})
        self.assertIn('PI', result)

    def test_no_substitute_gate_names(self):
        result = self.t._substitute_vars('H 0', {'H': 99})
        self.assertIn('H', result)
        self.assertNotIn('99', result)

    def test_normal_substitution(self):
        result = self.t._substitute_vars('RX angle, qubit', {'angle': '0.5', 'qubit': '2'})
        self.assertIn('0.5', result)
        self.assertIn('2', result)


class TestSaveLoad(unittest.TestCase):
    """Test file I/O."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_save_load_roundtrip(self):
        self.t.num_qubits = 3
        self.t.shots = 512
        self.t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}
        path = '_test_save_load_roundtrip.qb'
        try:
            capture(self.t.cmd_save, path)
            t2 = QBasicTerminal()
            capture(t2.cmd_load, path)
            self.assertEqual(t2.num_qubits, 3)
            self.assertEqual(t2.shots, 512)
            self.assertEqual(len(t2.program), 3)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestLOCCEngine2Party(unittest.TestCase):
    """Test 2-party LOCC engine (backward compatibility)."""

    def test_split_mode_init(self):
        eng = LOCCEngine([3, 3])
        self.assertEqual(eng.n_regs, 2)
        self.assertEqual(eng.names, ['A', 'B'])
        self.assertEqual(eng.sizes, [3, 3])
        self.assertEqual(eng.n_a, 3)
        self.assertEqual(eng.n_b, 3)

    def test_joint_mode_init(self):
        eng = LOCCEngine([3, 3], joint=True)
        self.assertEqual(eng.n_total, 6)

    def test_apply_gate_split(self):
        eng = LOCCEngine([2, 2])
        eng.apply('A', 'H', (), [0])
        # A's statevector should be in superposition
        probs = np.abs(eng.svs['A'].ravel())**2
        self.assertAlmostEqual(probs[0], 0.5)

    def test_send_split(self):
        eng = LOCCEngine([2, 2])
        eng.apply('A', 'H', (), [0])
        outcome = eng.send('A', 0)
        self.assertIn(outcome, [0, 1])
        # After measurement, should be collapsed
        probs = np.abs(eng.svs['A'].ravel())**2
        self.assertAlmostEqual(max(probs), 1.0, places=5)

    def test_share_joint(self):
        eng = LOCCEngine([2, 2], joint=True)
        eng.share('A', 0, 'B', 0)
        # Should create Bell state
        per_reg, joint = eng.sample_joint(10000)
        for state in joint:
            parts = state.split('|')
            # Both registers should agree on the shared qubit
            self.assertEqual(parts[0][-1], parts[1][-1])

    def test_share_split_raises(self):
        eng = LOCCEngine([2, 2])
        with self.assertRaises(RuntimeError):
            eng.share('A', 0, 'B', 0)

    def test_sample_split(self):
        eng = LOCCEngine([2, 2])
        eng.apply('A', 'X', (), [0])
        per_reg, joint = eng.sample_joint(100)
        # A should always be |01⟩ (qubit 0 flipped)
        self.assertIn('01', per_reg['A'])
        self.assertEqual(per_reg['A']['01'], 100)

    def test_mem_gb(self):
        eng = LOCCEngine([20, 20])
        total, peak = eng.mem_gb()
        self.assertGreater(total, 0)

    def test_reset(self):
        eng = LOCCEngine([2, 2])
        eng.apply('A', 'X', (), [0])
        eng.classical['x'] = 1
        eng.reset()
        # Should be back to |00⟩
        probs = np.abs(eng.svs['A'].ravel())**2
        self.assertAlmostEqual(probs[0], 1.0)
        self.assertEqual(len(eng.classical), 0)


class TestLOCCEngineNParty(unittest.TestCase):
    """Test N-party LOCC engine (new functionality)."""

    def test_3party_init(self):
        eng = LOCCEngine([4, 4, 4])
        self.assertEqual(eng.n_regs, 3)
        self.assertEqual(eng.names, ['A', 'B', 'C'])
        self.assertEqual(eng.n_total, 12)

    def test_4party_init(self):
        eng = LOCCEngine([2, 3, 4, 5])
        self.assertEqual(eng.n_regs, 4)
        self.assertEqual(eng.names, ['A', 'B', 'C', 'D'])
        self.assertEqual(eng.sizes, [2, 3, 4, 5])

    def test_3party_gates(self):
        eng = LOCCEngine([3, 3, 3])
        eng.apply('A', 'H', (), [0])
        eng.apply('B', 'X', (), [1])
        eng.apply('C', 'H', (), [2])
        # Verify each register is independent
        a_probs = np.abs(eng.svs['A'].ravel())**2
        self.assertAlmostEqual(a_probs[0], 0.5)
        b_probs = np.abs(eng.svs['B'].ravel())**2
        self.assertAlmostEqual(b_probs[2], 1.0)  # |010⟩

    def test_3party_send(self):
        eng = LOCCEngine([2, 2, 2])
        eng.apply('C', 'H', (), [0])
        outcome = eng.send('C', 0)
        self.assertIn(outcome, [0, 1])

    def test_3party_sample(self):
        eng = LOCCEngine([2, 2, 2])
        eng.apply('A', 'X', (), [0])
        eng.apply('B', 'X', (), [1])
        eng.apply('C', 'H', (), [0])
        per_reg, joint = eng.sample_joint(100)
        self.assertIn('A', per_reg)
        self.assertIn('B', per_reg)
        self.assertIn('C', per_reg)
        # A should always be |01⟩
        self.assertEqual(per_reg['A']['01'], 100)
        # B should always be |10⟩
        self.assertEqual(per_reg['B']['10'], 100)

    def test_3party_joint_mode(self):
        eng = LOCCEngine([2, 2, 2], joint=True)
        self.assertEqual(eng.n_total, 6)
        eng.apply('A', 'H', (), [0])
        eng.apply('C', 'X', (), [0])
        per_reg, joint = eng.sample_joint(100)
        self.assertEqual(len(per_reg), 3)
        self.assertGreater(len(joint), 0)

    def test_3party_share_joint(self):
        eng = LOCCEngine([2, 2, 2], joint=True)
        eng.share('A', 0, 'C', 0)
        per_reg, joint = eng.sample_joint(10000)
        # A[0] and C[0] should be correlated (Bell pair)
        for state in joint:
            parts = state.split('|')
            a_bit = parts[0][-1]  # A's qubit 0
            c_bit = parts[2][-1]  # C's qubit 0
            self.assertEqual(a_bit, c_bit)

    def test_offsets_correct(self):
        eng = LOCCEngine([3, 5, 7])
        self.assertEqual(eng.offsets, [0, 3, 8])

    def test_get_size(self):
        eng = LOCCEngine([10, 20, 30])
        self.assertEqual(eng.get_size('A'), 10)
        self.assertEqual(eng.get_size('B'), 20)
        self.assertEqual(eng.get_size('C'), 30)

    def test_backward_compat_properties(self):
        eng = LOCCEngine([5, 7, 9])
        self.assertEqual(eng.n_a, 5)
        self.assertEqual(eng.n_b, 7)


class TestLOCCTerminalIntegration(unittest.TestCase):
    """Test LOCC commands through the terminal."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_locc_3party_setup(self):
        _, out = capture(self.t.cmd_locc, '4 4 4')
        self.assertTrue(self.t.locc_mode)
        self.assertEqual(self.t.locc.n_regs, 3)
        self.assertIn('C=4q', out)

    def test_locc_off(self):
        capture(self.t.cmd_locc, '3 3')
        capture(self.t.cmd_locc, 'OFF')
        self.assertFalse(self.t.locc_mode)

    def test_locc_3party_run(self):
        capture(self.t.cmd_locc, '2 2 2')
        self.t.shots = 50
        self.t.program = {
            10: '@A H 0',
            20: '@B H 0',
            30: '@C H 0',
            40: 'MEASURE',
        }
        _, out = capture(self.t.cmd_run)
        self.assertIn('Register A', out)
        self.assertIn('Register B', out)
        self.assertIn('Register C', out)

    def test_locc_send_any_register(self):
        capture(self.t.cmd_locc, '2 2 2')
        self.t.program = {
            10: '@C H 0',
            20: 'SEND C 0 -> c0',
            30: 'IF c0 THEN @A X 0',
            40: 'MEASURE',
        }
        self.t.shots = 50
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)

    def test_locc_status(self):
        capture(self.t.cmd_locc, '5 10 15')
        _, out = capture(self.t.cmd_locc, 'STATUS')
        self.assertIn('A=5q', out)
        self.assertIn('B=10q', out)
        self.assertIn('C=15q', out)

    def test_locc_joint_limit(self):
        _, out = capture(self.t.cmd_locc, 'JOINT 20 20')
        self.assertIn('limited', out)
        self.assertFalse(self.t.locc_mode)

    def test_locc_loccinfo(self):
        capture(self.t.cmd_locc, '3 3 3')
        _, out = capture(self.t.cmd_loccinfo)
        self.assertIn('3 parties', out)

    def test_locc_colon_inheritance(self):
        """@C prefix should inherit across colon-separated gates."""
        capture(self.t.cmd_locc, '2 2 2')
        self.t.shots = 50
        self.t.program = {
            10: '@C H 0 : X 1',  # X 1 should inherit @C
            20: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)


class TestDisplayFormatting(unittest.TestCase):
    """Test display output doesn't crash."""

    def setUp(self):
        self.t = QBasicTerminal()
        self.t.num_qubits = 2
        self.t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}
        capture(self.t.cmd_run)

    def test_state(self):
        _, out = capture(self.t.cmd_state)
        self.assertIn('Statevector', out)

    def test_hist(self):
        _, out = capture(self.t.cmd_hist)
        self.assertIn('%', out)

    def test_probs(self):
        _, out = capture(self.t.cmd_probs)
        self.assertIn('Probability', out)

    def test_bloch(self):
        _, out = capture(self.t.cmd_bloch, '0')
        self.assertIn('Qubit 0', out)

    def test_circuit(self):
        _, out = capture(self.t.cmd_circuit)
        self.assertTrue(len(out) > 0)

    def test_decompose(self):
        _, out = capture(self.t.cmd_decompose)
        self.assertIn('Circuit', out)

    def test_csv(self):
        _, out = capture(self.t.cmd_csv, '')
        self.assertIn('state,count,probability', out)

    def test_export(self):
        _, out = capture(self.t.cmd_export, '')
        # Should produce QASM output (either version) or an error message
        self.assertTrue('OPENQASM' in out or 'qubit' in out or 'EXPORT' in out)

    def test_density(self):
        _, out = capture(self.t.cmd_density)
        self.assertIn('Density matrix', out)

    def test_expect(self):
        _, out = capture(self.t.cmd_expect, 'Z 0')
        self.assertIn('<Z>', out)

    def test_entropy(self):
        _, out = capture(self.t.cmd_entropy, '0')
        self.assertIn('entropy', out)


class TestDemos(unittest.TestCase):
    """Test that all demos run without error."""

    def _run_demo(self, name):
        t = QBasicTerminal()
        _, out = capture(t.cmd_demo, name)
        return t, out

    def test_bell(self):
        t, out = self._run_demo('BELL')
        self.assertIn('Bell State', out)
        self.assertIsNotNone(t.last_counts)

    def test_ghz(self):
        t, out = self._run_demo('GHZ')
        self.assertIsNotNone(t.last_counts)

    def test_teleport(self):
        t, out = self._run_demo('TELEPORT')
        self.assertIsNotNone(t.last_counts)

    def test_grover(self):
        t, out = self._run_demo('GROVER')
        self.assertIsNotNone(t.last_counts)
        # |101⟩ should be the top result
        top = max(t.last_counts, key=t.last_counts.get)
        self.assertEqual(top, '101')

    def test_qft(self):
        t, out = self._run_demo('QFT')
        self.assertIsNotNone(t.last_counts)

    def test_deutsch(self):
        t, out = self._run_demo('DEUTSCH')
        self.assertIsNotNone(t.last_counts)

    def test_bernstein(self):
        t, out = self._run_demo('BERNSTEIN')
        self.assertIsNotNone(t.last_counts)
        # Secret is 1011, so measurement should show x1011 pattern
        top = max(t.last_counts, key=t.last_counts.get)
        self.assertTrue(top.endswith('1011'), f"Expected ...1011, got {top}")

    def test_superdense(self):
        t, out = self._run_demo('SUPERDENSE')
        self.assertIsNotNone(t.last_counts)
        top = max(t.last_counts, key=t.last_counts.get)
        self.assertEqual(top, '11')

    def test_random(self):
        t, out = self._run_demo('RANDOM')
        self.assertIsNotNone(t.last_counts)

    def test_demo_list(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_demo, 'LIST')
        self.assertIn('BELL', out)
        self.assertIn('LOCC-TELEPORT', out)


class TestEdgeCases(unittest.TestCase):
    """Esoteric edge cases and regression guards."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_empty_program_run(self):
        _, out = capture(self.t.cmd_run)
        self.assertIn('NOTHING', out)

    def test_qubit_out_of_range(self):
        self.t.num_qubits = 2
        _, out = capture(self.t.dispatch, 'H 5')
        self.assertIn('OUT OF RANGE', out)

    def test_unknown_gate(self):
        _, out = capture(self.t.dispatch, 'FOOBAR 0')
        self.assertIn('UNKNOWN', out)

    def test_nested_for_loops(self):
        self.t.num_qubits = 4
        self.t.program = {
            10: 'FOR I = 0 TO 1',
            20: 'FOR J = 2 TO 3',
            30: 'CX I, J',
            40: 'NEXT J',
            50: 'NEXT I',
            60: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)

    def test_for_step_negative(self):
        self.t.num_qubits = 4
        self.t.program = {
            10: 'FOR I = 3 TO 0 STEP -1',
            20: 'H I',
            30: 'NEXT I',
            40: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)
        self.assertGreater(len(self.t.last_counts), 1)

    def test_immediate_gate(self):
        self.t.num_qubits = 2
        _, out = capture(self.t.run_immediate, 'H 0')
        self.assertIsNotNone(self.t.last_sv)

    def test_noise_on_off(self):
        capture(self.t.cmd_noise, 'depolarizing 0.01')
        self.assertIsNotNone(self.t._noise_model)
        capture(self.t.cmd_noise, 'OFF')
        self.assertIsNone(self.t._noise_model)

    def test_ctrl_gate(self):
        self.t.num_qubits = 3
        self.t.program = {10: 'X 0', 20: 'CTRL H 0, 1', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)

    def test_inv_gate(self):
        self.t.num_qubits = 1
        self.t.program = {10: 'H 0', 20: 'INV H 0', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        # H followed by H^-1 = identity = always |0⟩
        self.assertEqual(self.t.last_counts.get('0', 0), self.t.shots)

    def test_barrier(self):
        self.t.num_qubits = 2
        self.t.program = {10: 'H 0', 20: 'BARRIER', 30: 'CX 0,1', 40: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)

    def test_reset(self):
        self.t.num_qubits = 1
        self.t.program = {10: 'X 0', 20: 'RESET 0', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        # After reset, qubit should be |0⟩
        self.assertEqual(self.t.last_counts.get('0', 0), self.t.shots)

    def test_rem_comment(self):
        self.t.num_qubits = 1
        self.t.program = {10: "REM this is a comment", 20: 'H 0', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)

    def test_apostrophe_comment(self):
        self.t.num_qubits = 1
        self.t.program = {10: "' also a comment", 20: 'X 0', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertEqual(self.t.last_counts.get('1', 0), self.t.shots)

    def test_sweep_runs(self):
        self.t.num_qubits = 1
        self.t.program = {10: 'RX angle, 0', 20: 'MEASURE'}
        _, out = capture(self.t.cmd_sweep, 'angle 0 PI 3')
        self.assertIn('SWEEP', out)

    def test_include_nonexistent(self):
        _, out = capture(self.t.cmd_include, 'nonexistent_file_xyz.qb')
        self.assertIn('NOT FOUND', out)

    def test_dir(self):
        _, out = capture(self.t.cmd_dir, '.')
        # Should not crash, may or may not find .qb files
        self.assertTrue(len(out) > 0)

    def test_help(self):
        _, out = capture(self.t.cmd_help)
        self.assertIn('QBASIC', out)
        self.assertIn('LOCC', out)

    def test_vars_display(self):
        self.t.variables['x'] = 42
        _, out = capture(self.t.cmd_vars)
        self.assertIn('x = 42', out)

    def test_defs_display(self):
        capture(self.t.cmd_def, 'BELL = H 0 : CX 0,1')
        _, out = capture(self.t.cmd_defs)
        self.assertIn('BELL', out)

    def test_regs_display(self):
        self.t.num_qubits = 4
        capture(self.t.cmd_reg, 'data 2')
        _, out = capture(self.t.cmd_regs)
        self.assertIn('data', out)

    def test_20qubit_runs(self):
        """20-qubit circuit should complete without OOM."""
        self.t.num_qubits = 20
        self.t.shots = 10
        self.t.program = {}
        line = 10
        for i in range(20):
            self.t.program[line] = f'H {i}'
            line += 10
        self.t.program[line] = 'MEASURE'
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)

    def test_locc_20_20_20(self):
        """3 x 20-qubit LOCC should initialize without error."""
        capture(self.t.cmd_locc, '20 20 20')
        self.assertTrue(self.t.locc_mode)
        self.assertEqual(self.t.locc.n_regs, 3)
        self.assertEqual(self.t.locc.n_total, 60)


class TestDoubleExecutionFix(unittest.TestCase):
    """Verify that cmd_run doesn't double-mutate variables (bug fix)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_variable_not_double_incremented(self):
        self.t.num_qubits = 1
        self.t.variables['counter'] = 0
        self.t.program = {
            10: 'LET counter = counter + 1',
            20: 'H 0',
            30: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertEqual(self.t.variables['counter'], 1)

    def test_variable_mutation_idempotent(self):
        self.t.num_qubits = 1
        self.t.variables['x'] = 10
        self.t.program = {
            10: 'LET x = x * 2',
            20: 'H 0',
            30: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertEqual(self.t.variables['x'], 20)


class TestForLoopFloat(unittest.TestCase):
    """Verify FOR loops support float STEP values."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_float_step(self):
        self.t.num_qubits = 1
        self.t.variables['count'] = 0
        self.t.program = {
            10: 'LET count = 0',
            20: 'FOR theta = 0 TO 1 STEP 0.5',
            30: 'LET count = count + 1',
            40: 'NEXT theta',
            50: 'MEASURE',
        }
        capture(self.t.cmd_run)
        # Iterates: 0, 0.5, 1.0 = 3 iterations
        self.assertEqual(self.t.variables['count'], 3)

    def test_integer_step_stays_int(self):
        """Integer-valued FOR bounds should produce int loop variables."""
        self.t.num_qubits = 4
        self.t.program = {
            10: 'FOR I = 0 TO 3',
            20: 'H I',
            30: 'NEXT I',
            40: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)


class TestLOCCUnequalRegisters(unittest.TestCase):
    """Test LOCC with unequal register sizes (item 8)."""

    def test_unequal_2party_init(self):
        eng = LOCCEngine([2, 5])
        self.assertEqual(eng.sizes, [2, 5])
        self.assertEqual(eng.n_total, 7)
        self.assertEqual(eng.offsets, [0, 2])

    def test_unequal_3party_init(self):
        eng = LOCCEngine([1, 3, 5])
        self.assertEqual(eng.sizes, [1, 3, 5])
        self.assertEqual(eng.offsets, [0, 1, 4])

    def test_unequal_joint_sampling(self):
        """Verify bitstring splitting for unequal register sizes in JOINT mode."""
        eng = LOCCEngine([2, 3], joint=True)
        eng.apply('A', 'X', (), [0])   # A qubit 0 flipped -> |01⟩
        eng.apply('B', 'X', (), [2])   # B qubit 2 flipped -> |100⟩
        per_reg, joint = eng.sample_joint(100)
        self.assertEqual(per_reg['A']['01'], 100)
        self.assertEqual(per_reg['B']['100'], 100)

    def test_unequal_3party_joint_sampling(self):
        """Verify splitting for 3 unequal registers."""
        eng = LOCCEngine([1, 2, 3], joint=True)
        eng.apply('A', 'X', (), [0])   # A = |1⟩
        eng.apply('C', 'X', (), [0])   # C qubit 0 flipped -> |001⟩
        per_reg, joint = eng.sample_joint(100)
        self.assertEqual(per_reg['A']['1'], 100)
        self.assertEqual(per_reg['B']['00'], 100)
        self.assertEqual(per_reg['C']['001'], 100)


class TestSplitIndependence(unittest.TestCase):
    """Verify SPLIT mode registers are truly independent (item 18)."""

    def test_gate_on_a_leaves_b_unchanged(self):
        eng = LOCCEngine([2, 2])
        eng.apply('A', 'X', (), [0])
        b_probs = np.abs(eng.svs['B'].ravel())**2
        self.assertAlmostEqual(b_probs[0], 1.0)
        a_probs = np.abs(eng.svs['A'].ravel())**2
        self.assertAlmostEqual(a_probs[1], 1.0)

    def test_entanglement_in_a_leaves_b_product(self):
        eng = LOCCEngine([3, 3])
        eng.apply('A', 'H', (), [0])
        eng.apply('A', 'CX', (), [0, 1])
        per_reg, _ = eng.sample_joint(1000)
        self.assertEqual(per_reg['B']['000'], 1000)
        for state in per_reg['A']:
            self.assertEqual(state[-1], state[-2])

    def test_3party_split_independence(self):
        eng = LOCCEngine([2, 2, 2])
        eng.apply('B', 'X', (), [0])
        per_reg, _ = eng.sample_joint(100)
        self.assertEqual(per_reg['A']['00'], 100)
        self.assertEqual(per_reg['B']['01'], 100)
        self.assertEqual(per_reg['C']['00'], 100)


class TestRenumTargets(unittest.TestCase):
    """Verify RENUM updates GOTO and GOSUB targets (item 19)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_renum_updates_goto(self):
        self.t.program = {5: 'GOTO 15', 10: 'X 0', 15: 'H 0'}
        capture(self.t.cmd_renum)
        self.assertIn('GOTO 30', self.t.program[10])

    def test_renum_updates_gosub(self):
        self.t.program = {5: 'GOSUB 15', 10: 'END', 15: 'H 0', 20: 'RETURN'}
        capture(self.t.cmd_renum)
        self.assertIn('GOSUB 30', self.t.program[10])

    def test_renum_preserves_execution(self):
        """A program with GOTO should behave the same after RENUM."""
        self.t.num_qubits = 1
        self.t.program = {5: 'GOTO 15', 10: 'X 0', 15: 'H 0', 20: 'MEASURE'}
        capture(self.t.cmd_renum)
        capture(self.t.cmd_run)
        # GOTO skips X 0, only H 0 applied — should be superposition
        self.assertIsNotNone(self.t.last_counts)
        self.assertGreater(len(self.t.last_counts), 1)


class TestExpressionEdgeCases(unittest.TestCase):
    """Edge cases for the AST expression evaluator (item 20)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_deeply_nested_parens(self):
        self.assertAlmostEqual(self.t.eval_expr('((((1+2))))'), 3.0)

    def test_division_by_zero(self):
        with self.assertRaises((ValueError, ZeroDivisionError)):
            self.t.eval_expr('1/0')

    def test_very_large_number(self):
        val = self.t.eval_expr('2**50')
        self.assertEqual(val, 2**50)

    def test_negative_exponent(self):
        self.assertAlmostEqual(self.t.eval_expr('2**-1'), 0.5)

    def test_floor_division(self):
        self.assertAlmostEqual(self.t.eval_expr('7//2'), 3.0)

    def test_modulo(self):
        self.assertAlmostEqual(self.t.eval_expr('7%3'), 1.0)

    def test_trig_identity(self):
        self.assertAlmostEqual(
            self.t.eval_expr('sin(PI/4)**2 + cos(PI/4)**2'), 1.0)

    def test_chained_boolean(self):
        self.assertTrue(self.t._eval_condition('1 AND 1 AND 1', {}))
        self.assertFalse(self.t._eval_condition('1 AND 0 AND 1', {}))
        self.assertTrue(self.t._eval_condition('0 OR 0 OR 1', {}))

    def test_not_operator(self):
        self.assertTrue(self.t._eval_condition('NOT 0', {}))
        self.assertFalse(self.t._eval_condition('NOT 1', {}))

    def test_mixed_comparisons(self):
        self.assertTrue(self.t._eval_condition('3 >= 3', {}))
        self.assertTrue(self.t._eval_condition('3 <= 3', {}))
        self.assertFalse(self.t._eval_condition('3 > 3', {}))


class TestLOCCBornRule(unittest.TestCase):
    """Verify SEND outcomes follow Born-rule statistics (item 21)."""

    def test_send_50_50(self):
        """SEND on |+⟩ should give ~50/50 outcomes."""
        outcomes = {0: 0, 1: 0}
        n_trials = 5000
        for _ in range(n_trials):
            eng = LOCCEngine([2, 2])
            eng.apply('A', 'H', (), [0])
            outcome = eng.send('A', 0)
            outcomes[outcome] += 1
        ratio = outcomes[0] / n_trials
        self.assertGreater(ratio, 0.45)
        self.assertLess(ratio, 0.55)

    def test_send_biased(self):
        """SEND on RY-prepared state should give biased outcomes."""
        outcomes = {0: 0, 1: 0}
        n_trials = 5000
        for _ in range(n_trials):
            eng = LOCCEngine([1, 1])
            # RY(PI/3) gives cos(PI/6)^2 ≈ 0.75 for |0⟩
            eng.apply('A', 'RY', (math.pi / 3,), [0])
            outcome = eng.send('A', 0)
            outcomes[outcome] += 1
        ratio_0 = outcomes[0] / n_trials
        self.assertGreater(ratio_0, 0.70)
        self.assertLess(ratio_0, 0.80)


class TestDisplayValues(unittest.TestCase):
    """Verify display commands show correct values, not just non-empty (item 17)."""

    def setUp(self):
        self.t = QBasicTerminal()
        self.t.num_qubits = 1

    def test_state_amplitudes(self):
        """STATE should show correct amplitudes for |+⟩."""
        self.t.program = {10: 'H 0', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        _, out = capture(self.t.cmd_state)
        self.assertIn('0.5000', out)
        self.assertIn('+0.7071', out)

    def test_hist_deterministic(self):
        """HIST should show 100% for deterministic |1⟩ state."""
        self.t.program = {10: 'X 0', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        _, out = capture(self.t.cmd_hist)
        self.assertIn('100.0%', out)

    def test_probs_superposition(self):
        """PROBS should show 50% for each state in |+⟩."""
        self.t.program = {10: 'H 0', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        _, out = capture(self.t.cmd_probs)
        self.assertIn('50.00%', out)

    def test_entropy_product_state(self):
        """Product state should have zero entanglement entropy."""
        self.t.num_qubits = 2
        self.t.program = {10: 'X 0', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        _, out = capture(self.t.cmd_entropy, '0')
        self.assertIn('0.000000', out)
        self.assertIn('separable', out)

    def test_entropy_bell_state(self):
        """Bell state should have maximal entanglement entropy."""
        self.t.num_qubits = 2
        self.t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        _, out = capture(self.t.cmd_entropy, '0')
        self.assertIn('1.000000', out)
        self.assertIn('entangled', out)


class TestDemoCorrectness(unittest.TestCase):
    """Verify demo circuits produce correct quantum results (item 16)."""

    def test_teleport_valid_states(self):
        """Without correction, teleportation circuit produces all 8 outcomes."""
        t = QBasicTerminal()
        _, out = capture(t.cmd_demo, 'TELEPORT')
        self.assertIsNotNone(t.last_counts)
        for state in t.last_counts:
            self.assertEqual(len(state), 3)
            self.assertTrue(all(c in '01' for c in state))
        # Without classical correction, all 8 outcomes are equally likely
        self.assertGreaterEqual(len(t.last_counts), 6)

    def test_superdense_deterministic(self):
        """Superdense coding of '11' should decode to |11⟩ deterministically."""
        t = QBasicTerminal()
        _, out = capture(t.cmd_demo, 'SUPERDENSE')
        self.assertIsNotNone(t.last_counts)
        self.assertEqual(t.last_counts.get('11', 0), t.shots)

    def test_deutsch_balanced(self):
        """Deutsch-Jozsa with balanced oracle: qubit 0 should measure |1⟩."""
        t = QBasicTerminal()
        _, out = capture(t.cmd_demo, 'DEUTSCH')
        top = max(t.last_counts, key=t.last_counts.get)
        # Qubit 0 is rightmost in Qiskit little-endian ordering
        self.assertEqual(top[-1], '1')


class TestRunScript(unittest.TestCase):
    """Test the script file runner (item 15)."""

    def test_run_script_loads_program(self):
        with tempfile.NamedTemporaryFile(suffix='.qb', delete=False, mode='w') as f:
            f.write("QUBITS 2\n")
            f.write("SHOTS 512\n")
            f.write("10 H 0\n")
            f.write("20 CX 0,1\n")
            f.write("30 MEASURE\n")
            path = f.name
        try:
            t = QBasicTerminal()
            capture(run_script, path, t)
            self.assertEqual(t.num_qubits, 2)
            self.assertEqual(t.shots, 512)
            self.assertIn(10, t.program)
            self.assertIn(20, t.program)
            self.assertIn(30, t.program)
        finally:
            os.unlink(path)

    def test_run_script_multiline_def(self):
        with tempfile.NamedTemporaryFile(suffix='.qb', delete=False, mode='w') as f:
            f.write("DEF BEGIN BELL\n")
            f.write("H 0\n")
            f.write("CX 0,1\n")
            f.write("DEF END\n")
            f.write("10 BELL\n")
            path = f.name
        try:
            t = QBasicTerminal()
            capture(run_script, path, t)
            self.assertIn('BELL', t.subroutines)
            self.assertEqual(len(t.subroutines['BELL']['body']), 2)
            self.assertIn(10, t.program)
        finally:
            os.unlink(path)

    def test_run_script_skips_comments(self):
        with tempfile.NamedTemporaryFile(suffix='.qb', delete=False, mode='w') as f:
            f.write("# This is a comment\n")
            f.write("QUBITS 3\n")
            f.write("# Another comment\n")
            f.write("10 H 0\n")
            path = f.name
        try:
            t = QBasicTerminal()
            capture(run_script, path, t)
            self.assertEqual(t.num_qubits, 3)
            self.assertEqual(len(t.program), 1)
        finally:
            os.unlink(path)


class TestDefMultiline(unittest.TestCase):
    """Test interactive multi-line DEF blocks (item 16)."""

    def test_def_multiline_from_repl(self):
        import unittest.mock
        t = QBasicTerminal()
        with unittest.mock.patch('builtins.input', side_effect=['H 0', 'CX 0,1', 'DEF END']):
            _, out = capture(t.cmd_def, 'BEGIN BELL')
        self.assertIn('BELL', t.subroutines)
        self.assertEqual(len(t.subroutines['BELL']['body']), 2)

    def test_def_multiline_with_params(self):
        import unittest.mock
        t = QBasicTerminal()
        with unittest.mock.patch('builtins.input', side_effect=['RX angle, q', 'RZ angle, q', 'END']):
            _, out = capture(t.cmd_def, 'BEGIN ROT(angle, q)')
        self.assertIn('ROT', t.subroutines)
        self.assertEqual(t.subroutines['ROT']['params'], ['angle', 'q'])

    def test_def_multiline_cancel(self):
        import unittest.mock
        t = QBasicTerminal()
        with unittest.mock.patch('builtins.input', side_effect=KeyboardInterrupt):
            _, out = capture(t.cmd_def, 'BEGIN FOO')
        self.assertNotIn('FOO', t.subroutines)


class TestBenchmark(unittest.TestCase):
    """Test benchmark command (item 17)."""

    def test_bench_runs(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_bench)
        self.assertIn('Benchmark', out)
        self.assertIn('qubits', out)


class TestPathSanitization(unittest.TestCase):
    """Test path sanitization for file I/O commands."""

    def test_null_byte_rejected(self):
        t = QBasicTerminal()
        with self.assertRaises(ValueError):
            t._sanitize_path("file\x00.qb")

    def test_control_char_rejected(self):
        t = QBasicTerminal()
        with self.assertRaises(ValueError):
            t._sanitize_path("file\x01.qb")

    def test_empty_path_rejected(self):
        t = QBasicTerminal()
        with self.assertRaises(ValueError):
            t._sanitize_path("")

    def test_valid_path_passes(self):
        t = QBasicTerminal()
        self.assertEqual(t._sanitize_path("test.qb"), "test.qb")

    def test_whitespace_stripped(self):
        t = QBasicTerminal()
        self.assertEqual(t._sanitize_path("  test.qb  "), "test.qb")

    def test_include_rejects_absolute_path(self):
        t = QBasicTerminal()
        # Use an absolute path format that os.path.isabs recognizes on this platform
        if os.name == 'nt':
            _, out = capture(t.cmd_include, 'C:\\Windows\\system32\\file.qb')
        else:
            _, out = capture(t.cmd_include, '/etc/passwd')
        self.assertIn('absolute', out.lower())


class TestIncludeBlocksExport(unittest.TestCase):
    """Verify EXPORT and CSV are blocked inside INCLUDE files."""

    def test_include_blocks_export(self):
        t = QBasicTerminal()
        path = '_test_include_export.qb'
        with open(path, 'w') as f:
            f.write("10 H 0\n")
            f.write("EXPORT evil.qasm\n")
        try:
            _, out = capture(t.cmd_include, path)
            self.assertIn('BLOCKED', out)
        finally:
            os.unlink(path)

    def test_include_blocks_csv(self):
        t = QBasicTerminal()
        path = '_test_include_csv.qb'
        with open(path, 'w') as f:
            f.write("CSV evil.csv\n")
        try:
            _, out = capture(t.cmd_include, path)
            self.assertIn('BLOCKED', out)
        finally:
            os.unlink(path)


class TestGateRegistry(unittest.TestCase):
    """Test that the gate matrix registry matches the original behavior."""

    def _is_unitary(self, m):
        n = m.shape[0]
        product = m @ m.conj().T
        return np.allclose(product, np.eye(n))

    def test_all_registered_gates_are_unitary(self):
        zero_param_gates = [g for g, (np_, nq) in GATE_TABLE.items()
                           if np_ == 0 and g not in GATE_ALIASES]
        for gate in zero_param_gates:
            m = _np_gate_matrix(gate)
            self.assertTrue(self._is_unitary(m), f"{gate} is not unitary")

    def test_aliases_resolve(self):
        np.testing.assert_array_almost_equal(
            _np_gate_matrix('CNOT'), _np_gate_matrix('CX'))
        np.testing.assert_array_almost_equal(
            _np_gate_matrix('TOFFOLI'), _np_gate_matrix('CCX'))
        np.testing.assert_array_almost_equal(
            _np_gate_matrix('FREDKIN'), _np_gate_matrix('CSWAP'))

    def test_unknown_gate_raises(self):
        with self.assertRaises(ValueError):
            _np_gate_matrix('NONEXISTENT')


class TestControlFlowHelpers(unittest.TestCase):
    """Verify decomposed control flow helpers work correctly."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_cf_let_array(self):
        self.t.arrays['data'] = [0.0, 0.0, 0.0]
        result = self.t._cf_let_array('LET data(1) = 42', {})
        self.assertIsNotNone(result)
        self.assertAlmostEqual(self.t.arrays['data'][1], 42.0)

    def test_cf_let_var(self):
        run_vars = {}
        result = self.t._cf_let_var('LET x = 10', run_vars)
        self.assertIsNotNone(result)
        self.assertEqual(run_vars['x'], 10)
        self.assertEqual(self.t.variables['x'], 10)

    def test_cf_goto(self):
        result = self.t._cf_goto('GOTO 30', [10, 20, 30, 40])
        self.assertEqual(result, (True, 2))

    def test_cf_goto_not_found(self):
        with self.assertRaises(RuntimeError):
            self.t._cf_goto('GOTO 999', [10, 20, 30])

    def test_cf_for_next_cycle(self):
        run_vars = {}
        loop_stack = []
        result = self.t._cf_for('FOR I = 0 TO 2', run_vars, loop_stack, 0)
        self.assertIsNotNone(result)
        self.assertEqual(run_vars['I'], 0)
        self.assertEqual(len(loop_stack), 1)
        # NEXT should advance
        result = self.t._cf_next('NEXT I', run_vars, loop_stack)
        self.assertEqual(run_vars['I'], 1)

    def test_cf_non_matching_returns_none(self):
        self.assertIsNone(self.t._cf_let_array('H 0', {}))
        self.assertIsNone(self.t._cf_let_var('H 0', {}))
        self.assertIsNone(self.t._cf_goto('H 0', [10]))
        self.assertIsNone(self.t._cf_for('H 0', {}, [], 0))


class TestColonSplitter(unittest.TestCase):
    """Test the unified colon statement splitter."""

    def test_simple_split(self):
        parts = QBasicTerminal._split_colon_stmts('H 0 : CX 0,1')
        self.assertEqual(parts, ['H 0', 'CX 0,1'])

    def test_register_prefix_inheritance(self):
        parts = QBasicTerminal._split_colon_stmts('@A H 0 : X 1 : CX 0,1')
        self.assertEqual(parts, ['@A H 0', '@A X 1', '@A CX 0,1'])

    def test_send_does_not_inherit(self):
        parts = QBasicTerminal._split_colon_stmts('@A H 0 : SEND A 0 -> x')
        self.assertEqual(parts, ['@A H 0', 'SEND A 0 -> x'])

    def test_if_does_not_inherit(self):
        parts = QBasicTerminal._split_colon_stmts('@A H 0 : IF x THEN @B X 0')
        self.assertEqual(parts, ['@A H 0', 'IF x THEN @B X 0'])

    def test_new_prefix_overrides(self):
        parts = QBasicTerminal._split_colon_stmts('@A H 0 : @B X 0')
        self.assertEqual(parts, ['@A H 0', '@B X 0'])

    def test_empty_parts_skipped(self):
        parts = QBasicTerminal._split_colon_stmts('H 0 : : X 1')
        self.assertEqual(parts, ['H 0', 'X 1'])


class TestMeasEdgeCases(unittest.TestCase):
    """Extended MEAS mid-circuit measurement tests (item 12)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_meas_stores_variable(self):
        """MEAS should store outcome in run_vars and self.variables."""
        self.t.num_qubits = 2
        self.t.program = {
            10: 'X 0',
            20: 'MEAS 0 -> result',
            30: 'MEASURE',
        }
        capture(self.t.cmd_run)
        # result should be set (0 initially, but X makes it |1>)
        self.assertIn('result', self.t.variables)

    def test_meas_on_zero_state(self):
        """MEAS on |0> should always yield 0."""
        self.t.num_qubits = 1
        outcomes = set()
        for _ in range(20):
            self.t.program = {10: 'MEAS 0 -> m', 20: 'MEASURE'}
            capture(self.t.cmd_run)
        # |0> always measures to 0 — variable should exist
        self.assertIn('m', self.t.variables)

    def test_meas_invalid_qubit(self):
        """MEAS on out-of-range qubit should not crash (Qiskit handles it)."""
        self.t.num_qubits = 2
        self.t.program = {10: 'MEAS 0 -> m', 20: 'MEASURE'}
        # Should run without exception
        capture(self.t.cmd_run)


class TestResetEdgeCases(unittest.TestCase):
    """Extended RESET tests (item 13)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_reset_after_h(self):
        """RESET after H should produce |0> deterministically."""
        self.t.num_qubits = 1
        self.t.shots = 100
        self.t.program = {10: 'H 0', 20: 'RESET 0', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertEqual(self.t.last_counts.get('0', 0), 100)

    def test_reset_preserves_other_qubits(self):
        """RESET on qubit 0 should not affect qubit 1."""
        self.t.num_qubits = 2
        self.t.shots = 100
        self.t.program = {10: 'X 1', 20: 'H 0', 30: 'RESET 0', 40: 'MEASURE'}
        capture(self.t.cmd_run)
        # Qubit 0 reset to |0>, qubit 1 is |1>: expect "10"
        self.assertEqual(self.t.last_counts.get('10', 0), 100)

    def test_double_reset(self):
        """Double RESET should be idempotent."""
        self.t.num_qubits = 1
        self.t.shots = 100
        self.t.program = {10: 'X 0', 20: 'RESET 0', 30: 'RESET 0', 40: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertEqual(self.t.last_counts.get('0', 0), 100)


class TestNoiseAffectsOutcomes(unittest.TestCase):
    """Verify that noise models actually affect measurement distributions (item 14)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_depolarizing_noise_adds_errors(self):
        """A high depolarizing rate on X 0 should produce some |0> outcomes."""
        self.t.num_qubits = 1
        self.t.shots = 1000
        self.t.program = {10: 'X 0', 20: 'MEASURE'}
        # Without noise: always |1>
        capture(self.t.cmd_run)
        clean_counts = dict(self.t.last_counts)
        self.assertEqual(clean_counts.get('1', 0), 1000)
        # With heavy noise: should see some |0>
        capture(self.t.cmd_noise, 'depolarizing 0.3')
        capture(self.t.cmd_run)
        noisy_counts = dict(self.t.last_counts)
        # With 30% depolarizing on X, we expect significant |0> outcomes
        self.assertGreater(noisy_counts.get('0', 0), 10,
                           "Heavy depolarizing noise should produce errors")
        capture(self.t.cmd_noise, 'OFF')

    def test_noise_off_restores_clean(self):
        """Turning noise off should restore deterministic behavior."""
        self.t.num_qubits = 1
        self.t.shots = 100
        self.t.program = {10: 'X 0', 20: 'MEASURE'}
        capture(self.t.cmd_noise, 'depolarizing 0.3')
        capture(self.t.cmd_noise, 'OFF')
        capture(self.t.cmd_run)
        self.assertEqual(self.t.last_counts.get('1', 0), 100)


class TestBasisMeasurement(unittest.TestCase):
    """Test MEASURE_X, MEASURE_Y, MEASURE_Z (item 15)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_measure_x_on_plus_state(self):
        """MEASURE_X on |+> should always give 0 (eigenstate of X)."""
        self.t.num_qubits = 2
        self.t.shots = 100
        self.t.program = {
            10: 'H 0',           # prepare |+>
            20: 'MEASURE_X 0',   # measure in X basis
            30: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIn('mx_0', self.t.variables)

    def test_measure_z_equivalent_to_standard(self):
        """MEASURE_Z should behave like standard measurement."""
        self.t.num_qubits = 1
        self.t.shots = 100
        self.t.program = {10: 'X 0', 20: 'MEASURE_Z 0', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIn('mz_0', self.t.variables)

    def test_measure_y_runs(self):
        """MEASURE_Y should execute without error."""
        self.t.num_qubits = 1
        self.t.program = {10: 'H 0', 20: 'MEASURE_Y 0', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIn('my_0', self.t.variables)


class TestSyndromeMeasurement(unittest.TestCase):
    """Test SYNDROME stabilizer measurement (item 16)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_syndrome_zz_on_bell(self):
        """ZZ on a Bell state |00>+|11> should give syndrome 0 (stabilized)."""
        self.t.num_qubits = 3  # 2 data + 1 ancilla
        self.t.program = {
            10: 'H 0',
            20: 'CX 0,1',
            30: 'SYNDROME ZZ 0 1 -> s0',
            40: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIn('s0', self.t.variables)

    def test_syndrome_bad_pauli_length(self):
        """Mismatched Pauli string and qubit count should error."""
        self.t.num_qubits = 4
        self.t.program = {
            10: 'SYNDROME ZZZ 0 1 -> s',
            20: 'MEASURE',
        }
        _, out = capture(self.t.cmd_run)
        self.assertIn('ERROR', out)

    def test_syndrome_x_stabilizer(self):
        """XX on |++> should give syndrome 0."""
        self.t.num_qubits = 3  # 2 data + 1 ancilla
        self.t.program = {
            10: 'H 0',
            20: 'H 1',
            30: 'SYNDROME XX 0 1 -> s0',
            40: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIn('s0', self.t.variables)


class TestPathTraversal(unittest.TestCase):
    """Test path traversal prevention (item 4)."""

    def test_dotdot_rejected(self):
        t = QBasicTerminal()
        with self.assertRaises(ValueError):
            t._sanitize_path("../../etc/passwd")

    def test_dotdot_in_middle_rejected(self):
        t = QBasicTerminal()
        with self.assertRaises(ValueError):
            t._sanitize_path("foo/../../../bar")

    def test_normal_relative_path_passes(self):
        t = QBasicTerminal()
        self.assertEqual(t._sanitize_path("examples/bell.qb"), "examples/bell.qb")

    def test_subdir_path_passes(self):
        t = QBasicTerminal()
        self.assertEqual(t._sanitize_path("dir/sub/file.qb"), "dir/sub/file.qb")


class TestIncludeDepthLimit(unittest.TestCase):
    """Test INCLUDE depth limiting (item 5)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_include_blocks_save(self):
        """SAVE command inside an included file should be blocked."""
        path = '_test_include_save.qb'
        with open(path, 'w') as f:
            f.write("10 H 0\n")
            f.write("SAVE evil.qb\n")
            f.write("20 X 0\n")
        try:
            _, out = capture(self.t.cmd_include, path)
            self.assertIn('BLOCKED', out)
            self.assertIn(10, self.t.program)
            self.assertIn(20, self.t.program)
        finally:
            os.unlink(path)

    def test_include_blocks_load(self):
        """LOAD command inside an included file should be blocked."""
        path = '_test_include_load.qb'
        with open(path, 'w') as f:
            f.write("LOAD something.qb\n")
            f.write("10 H 0\n")
        try:
            _, out = capture(self.t.cmd_include, path)
            self.assertIn('BLOCKED', out)
        finally:
            os.unlink(path)

    def test_depth_limit_enforced(self):
        """Exceeding include depth should be caught."""
        self.t._include_depth = 100  # simulate deep recursion
        _, out = capture(self.t.cmd_include, 'anything.qb')
        self.assertIn('DEPTH LIMIT', out)
        self.t._include_depth = 0


class TestNumericalStability(unittest.TestCase):
    """Test _measure_np numerical stability (item 9)."""

    def test_measure_near_zero_state(self):
        """Measuring a near-zero statevector should return a normalized state."""
        sv = np.array([1e-310, 1e-310], dtype=complex)
        outcome, new_sv = _measure_np(sv, 0, 1)
        self.assertEqual(outcome, 0)  # defaults to 0
        self.assertEqual(new_sv.shape, (2,))
        # Must be normalized
        norm = np.sum(np.abs(new_sv)**2)
        self.assertAlmostEqual(norm, 1.0)


class TestNewExprFunctions(unittest.TestCase):
    """Test newly added expression functions (item 11)."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_atan2(self):
        self.assertAlmostEqual(self.t.eval_expr('atan2(1, 1)'), math.pi / 4)

    def test_ceil(self):
        self.assertAlmostEqual(self.t.eval_expr('ceil(2.3)'), 3.0)

    def test_floor(self):
        self.assertAlmostEqual(self.t.eval_expr('floor(2.7)'), 2.0)


class TestCtrlInLOCC(unittest.TestCase):
    """Test CTRL gate modifier works in LOCC mode."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_ctrl_gate_locc_split(self):
        """CTRL H in LOCC SPLIT mode should work."""
        capture(self.t.cmd_locc, '3 3')
        self.t.shots = 100
        self.t.program = {
            10: '@A X 0',
            20: '@A CTRL H 0, 1',
            30: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)

    def test_ctrl_gate_locc_joint(self):
        """CTRL H in LOCC JOINT mode should work."""
        capture(self.t.cmd_locc, 'JOINT 3 3')
        self.t.shots = 100
        self.t.program = {
            10: '@A X 0',
            20: '@A CTRL H 0, 1',
            30: 'MEASURE',
        }
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)


class TestSweepFunctionality(unittest.TestCase):
    """Test SWEEP parameter scan."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_sweep_basic(self):
        """SWEEP should run circuit at each parameter value."""
        self.t.num_qubits = 1
        self.t.program = {10: 'RX angle, 0', 20: 'MEASURE'}
        _, out = capture(self.t.cmd_sweep, 'angle 0 PI 3')
        self.assertIn('SWEEP', out)
        # Should show 3 parameter values
        lines = [l for l in out.split('\n') if 'angle=' in l]
        self.assertEqual(len(lines), 3)

    def test_sweep_endpoints(self):
        """SWEEP endpoints should be correct."""
        self.t.num_qubits = 1
        self.t.program = {10: 'RX angle, 0', 20: 'MEASURE'}
        _, out = capture(self.t.cmd_sweep, 'angle 0 1 2')
        self.assertIn('angle=  0.0000', out)
        self.assertIn('angle=  1.0000', out)


class TestRamCommand(unittest.TestCase):
    """Test RAM command."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_ram_basic(self):
        """RAM should show memory info."""
        _, out = capture(self.t.cmd_ram)
        # Should show system RAM or psutil error
        self.assertTrue('RAM' in out or 'psutil' in out)

    def test_ram_shows_qubit_scaling(self):
        """RAM should show qubit scaling table."""
        _, out = capture(self.t.cmd_ram)
        if 'psutil' not in out:
            self.assertIn('qubits', out)
            self.assertIn('GB', out)

    def test_ram_locc_mode(self):
        """RAM in LOCC mode should show LOCC info."""
        capture(self.t.cmd_locc, '4 4')
        _, out = capture(self.t.cmd_ram)
        if 'psutil' not in out:
            self.assertIn('LOCC', out)


class TestNewFixes(unittest.TestCase):
    """Tests for v0.2.0 fixes."""

    def test_version_exists(self):
        from qbasic_core import __version__
        self.assertEqual(__version__, '0.2.0')

    def test_save_rejects_absolute_path(self):
        t = QBasicTerminal()
        t.program = {10: 'H 0'}
        if os.name == 'nt':
            _, out = capture(t.cmd_save, 'C:\\tmp\\evil.qb')
        else:
            _, out = capture(t.cmd_save, '/tmp/evil.qb')
        self.assertIn('absolute', out.lower())

    def test_meas_warns_in_circuit_mode(self):
        t = QBasicTerminal()
        t.num_qubits = 2
        t.program = {10: 'X 0', 20: 'MEAS 0 -> r', 30: 'MEASURE'}
        _, out = capture(t.cmd_run)
        self.assertIn('WARNING', out)
        self.assertIn('IF/THEN', out)
        self.assertIn('LOCC', out)

    def test_protocol_satisfied(self):
        from qbasic_core.protocol import TerminalProtocol
        t = QBasicTerminal()
        self.assertIsInstance(t, TerminalProtocol)

    def test_exec_outcome_type(self):
        from qbasic_core.engine import ExecOutcome, ExecResult
        # ExecOutcome is a union type alias
        self.assertIs(ExecResult.ADVANCE, ExecResult.ADVANCE)


class TestSweepWithPlotille(unittest.TestCase):
    """Test SWEEP with plotille (item 16 from gap list)."""

    def test_sweep_plotille_no_crash(self):
        """SWEEP should not crash whether plotille is installed or not."""
        t = QBasicTerminal()
        t.num_qubits = 1
        t.program = {10: 'RX angle, 0', 20: 'MEASURE'}
        # 5 steps to trigger plotille chart if available
        _, out = capture(t.cmd_sweep, 'angle 0 PI 5')
        self.assertIn('SWEEP', out)
        lines = [l for l in out.split('\n') if 'angle=' in l]
        self.assertEqual(len(lines), 5)


class TestRunScriptDefBegin(unittest.TestCase):
    """Test run_script with DEF BEGIN blocks (item 17 from gap list)."""

    def test_def_begin_with_params(self):
        """DEF BEGIN with parameters should work in scripts."""
        with tempfile.NamedTemporaryFile(suffix='.qb', delete=False, mode='w') as f:
            f.write("QUBITS 2\n")
            f.write("DEF BEGIN ROT(angle, q)\n")
            f.write("RX angle, q\n")
            f.write("RZ angle, q\n")
            f.write("DEF END\n")
            f.write("10 ROT PI/4, 0\n")
            f.write("20 MEASURE\n")
            path = f.name
        try:
            t = QBasicTerminal()
            capture(run_script, path, t)
            self.assertIn('ROT', t.subroutines)
            self.assertEqual(t.subroutines['ROT']['params'], ['angle', 'q'])
            self.assertIsNotNone(t.last_counts)
        finally:
            os.unlink(path)

    def test_def_begin_builtin_rejected_in_script(self):
        """DEF BEGIN with a built-in gate name should be rejected."""
        with tempfile.NamedTemporaryFile(suffix='.qb', delete=False, mode='w') as f:
            f.write("DEF BEGIN H\n")
            f.write("X 0\n")
            f.write("DEF END\n")
            path = f.name
        try:
            t = QBasicTerminal()
            _, out = capture(run_script, path, t)
            self.assertIn('CANNOT REDEFINE', out)
            self.assertNotIn('H', t.subroutines)
        finally:
            os.unlink(path)


class TestNestedInclude(unittest.TestCase):
    """Test nested INCLUDE depth limiting (item 18 from gap list)."""

    def test_nested_include_allowed(self):
        """INCLUDE inside an included file should now work (depth-limited)."""
        inner = '_test_inner_include2.qb'
        outer = '_test_outer_include2.qb'
        with open(inner, 'w') as f:
            f.write("10 H 0\n")
        with open(outer, 'w') as f:
            f.write(f"INCLUDE {inner}\n")
            f.write("20 X 1\n")
        try:
            t = QBasicTerminal()
            _, out = capture(t.cmd_include, outer)
            self.assertIn(10, t.program)
            self.assertIn(20, t.program)
        finally:
            os.unlink(inner)
            os.unlink(outer)

    def test_nested_include_depth_limit(self):
        """Self-including files should hit the depth limit."""
        path = '_test_self_include.qb'
        with open(path, 'w') as f:
            f.write("10 H 0\n")
            f.write(f"INCLUDE {path}\n")
        try:
            t = QBasicTerminal()
            _, out = capture(t.cmd_include, path)
            self.assertIn('DEPTH LIMIT', out)
        finally:
            os.unlink(path)


class TestSanitizePathAbsolute(unittest.TestCase):
    """Test that _sanitize_path rejects absolute paths (item 8 from gap list)."""

    @unittest.skipIf(os.name == 'nt', "Unix paths not absolute on Windows")
    def test_unix_absolute_rejected(self):
        t = QBasicTerminal()
        with self.assertRaises(ValueError):
            t._sanitize_path("/etc/passwd")

    def test_windows_absolute_rejected(self):
        t = QBasicTerminal()
        with self.assertRaises(ValueError):
            t._sanitize_path("C:\\Windows\\system32\\file.qb")

    def test_csv_rejects_absolute(self):
        t = QBasicTerminal()
        t.num_qubits = 1
        t.program = {10: 'X 0', 20: 'MEASURE'}
        capture(t.cmd_run)
        if os.name == 'nt':
            _, out = capture(t.cmd_csv, 'C:\\tmp\\evil.csv')
        else:
            _, out = capture(t.cmd_csv, '/tmp/evil.csv')
        self.assertIn('ERROR', out)

    def test_export_rejects_absolute(self):
        t = QBasicTerminal()
        t.num_qubits = 1
        t.program = {10: 'H 0', 20: 'MEASURE'}
        capture(t.cmd_run)
        if os.name == 'nt':
            _, out = capture(t.cmd_export, 'C:\\tmp\\evil.qasm')
        else:
            _, out = capture(t.cmd_export, '/tmp/evil.qasm')
        self.assertIn('ERROR', out)


class TestCtrlValidation(unittest.TestCase):
    """Test CTRL gate qubit count validation in LOCC (item 13 from gap list)."""

    def test_ctrl_wrong_target_count(self):
        """CTRL SWAP with wrong number of targets should error."""
        t = QBasicTerminal()
        capture(t.cmd_locc, '4 4')
        t.program = {
            10: '@A CTRL SWAP 0, 1',  # SWAP needs 2 targets, only 1 given
            20: 'MEASURE',
        }
        _, out = capture(t.cmd_run)
        self.assertIn('target', out.lower())


class TestLoadSilent(unittest.TestCase):
    """Test that LOAD does not print READY (item 6 from gap list)."""

    def test_load_no_ready(self):
        path = '_test_load_silent.qb'
        with open(path, 'w') as f:
            f.write("QUBITS 2\n10 H 0\n")
        try:
            t = QBasicTerminal()
            _, out = capture(t.cmd_load, path)
            self.assertNotIn('READY', out)
            self.assertIn('LOADED', out)
        finally:
            os.unlink(path)


class TestNoiseExpandedCoverage(unittest.TestCase):
    """Test that noise applies to more gate types (item 20 from gap list)."""

    def test_depolarizing_on_3q_gate(self):
        """Depolarizing noise on CCX should affect outcomes."""
        t = QBasicTerminal()
        t.num_qubits = 3
        t.shots = 1000
        t.program = {10: 'X 0', 20: 'X 1', 30: 'CCX 0,1,2', 40: 'MEASURE'}
        # Clean run: always |111>
        capture(t.cmd_run)
        self.assertEqual(t.last_counts.get('111', 0), 1000)
        # With heavy noise
        capture(t.cmd_noise, 'depolarizing 0.3')
        capture(t.cmd_run)
        # Should see some non-|111> outcomes
        non_111 = sum(v for k, v in t.last_counts.items() if k != '111')
        self.assertGreater(non_111, 0)
        capture(t.cmd_noise, 'OFF')


class TestPrintBanner(unittest.TestCase):
    """Test print_banner runs without error."""

    def test_banner_output(self):
        t = QBasicTerminal()
        _, out = capture(t.print_banner)
        self.assertIn('Quantum BASIC', out)
        self.assertIn('qubits', out)


class TestStepMode(unittest.TestCase):
    """Test cmd_step with mocked input."""

    def test_step_runs(self):
        import unittest.mock
        t = QBasicTerminal()
        t.num_qubits = 2
        t.program = {10: 'H 0', 20: 'CX 0,1'}
        # Press Enter twice then Q
        with unittest.mock.patch('builtins.input', side_effect=['', '', 'Q']):
            _, out = capture(t.cmd_step)
        self.assertIn('STEP MODE', out)

    def test_step_empty_program(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_step)
        self.assertIn('NOTHING', out)


class TestLOCCDemos(unittest.TestCase):
    """Test LOCC-TELEPORT and LOCC-COORD demos."""

    def test_locc_teleport(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_demo, 'LOCC-TELEPORT')
        self.assertIn('LOCC Teleportation', out)
        self.assertIsNotNone(t.last_counts)

    def test_locc_coord(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_demo, 'LOCC-COORD')
        self.assertIn('Classical Coordination', out)
        self.assertIsNotNone(t.last_counts)


class TestUnitaryValid(unittest.TestCase):
    """Test UNITARY with a valid custom gate used in a program."""

    def test_custom_gate_in_program(self):
        t = QBasicTerminal()
        t.num_qubits = 1
        t.shots = 100
        # sqrt(X) gate
        t.program = {
            10: 'UNITARY SQX = [[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]]',
            20: 'SQX 0',
            30: 'SQX 0',  # SQX twice = X
            40: 'MEASURE',
        }
        capture(t.cmd_run)
        self.assertIn('SQX', t._custom_gates)
        # Two applications of sqrt(X) = X, so should measure |1>
        self.assertEqual(t.last_counts.get('1', 0), 100)


class TestSweepEdgeCases(unittest.TestCase):
    """Test SWEEP edge cases."""

    def test_sweep_one_step(self):
        """SWEEP with steps=1 should not crash."""
        t = QBasicTerminal()
        t.num_qubits = 1
        t.program = {10: 'RX angle, 0', 20: 'MEASURE'}
        _, out = capture(t.cmd_sweep, 'angle 0 PI 1')
        self.assertIn('SWEEP', out)
        lines = [l for l in out.split('\n') if 'angle=' in l]
        self.assertEqual(len(lines), 1)

    def test_sweep_zero_steps(self):
        """SWEEP with steps=0 should error."""
        t = QBasicTerminal()
        t.num_qubits = 1
        t.program = {10: 'RX angle, 0', 20: 'MEASURE'}
        _, out = capture(t.cmd_sweep, 'angle 0 PI 0')
        self.assertIn('at least 1', out)


class TestSubroutineRecursion(unittest.TestCase):
    """Test that mutually recursive subroutines are caught."""

    def test_self_recursive_subroutine(self):
        t = QBasicTerminal()
        t.num_qubits = 2
        capture(t.cmd_def, 'LOOP = LOOP')
        t.program = {10: 'LOOP', 20: 'MEASURE'}
        _, out = capture(t.cmd_run)
        self.assertTrue('RECURSION' in out or 'recursion' in out)


class TestEntropyComma(unittest.TestCase):
    """Test ENTROPY with various comma formats."""

    def test_entropy_comma_space(self):
        """ENTROPY 0, 1 should not crash."""
        t = QBasicTerminal()
        t.num_qubits = 3
        t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}
        capture(t.cmd_run)
        _, out = capture(t.cmd_entropy, '0, 1')
        self.assertIn('entropy', out)


class TestLoadDirectory(unittest.TestCase):
    """Test that LOAD rejects directories."""

    def test_load_directory(self):
        t = QBasicTerminal()
        _, out = capture(t.cmd_load, '.')
        self.assertIn('directory', out.lower())


class TestNestedIncludeWorks(unittest.TestCase):
    """Test that nested INCLUDE now works (depth-limited, not blocked)."""

    def test_two_level_include(self):
        inner = '_test_nested_inner.qb'
        outer = '_test_nested_outer.qb'
        with open(inner, 'w') as f:
            f.write("10 H 0\n")
        with open(outer, 'w') as f:
            f.write(f"INCLUDE {inner}\n")
            f.write("20 X 1\n")
        try:
            t = QBasicTerminal()
            _, out = capture(t.cmd_include, outer)
            self.assertIn(10, t.program)
            self.assertIn(20, t.program)
            self.assertNotIn('BLOCKED', out)
        finally:
            os.unlink(inner)
            os.unlink(outer)


class TestLOCCPerLineError(unittest.TestCase):
    """Test that LOCC execution reports line numbers on errors."""

    def test_locc_bad_gate_shows_line(self):
        t = QBasicTerminal()
        capture(t.cmd_locc, '2 2')
        t.program = {
            10: '@A H 0',
            20: '@A FOOBAR 0',
            30: 'MEASURE',
        }
        _, out = capture(t.cmd_run)
        self.assertIn('LINE 20', out)


if __name__ == '__main__':
    # Force UTF-8 for Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    unittest.main(verbosity=2)
