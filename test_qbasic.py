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
from conftest import capture


# ---------------------------------------------------------------------------
# 1. TestExpressions
# ---------------------------------------------------------------------------
class TestExpressions(unittest.TestCase):
    """Expression evaluator: arithmetic, functions, variables, conditions,
    arrays, errors, matrix parsing."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_arithmetic(self):
        for expr, expected in [
            ('2+3', 5.0), ('10-4', 6.0), ('3*7', 21.0), ('15/4', 3.75),
            ('2+3*4', 14.0), ('(2+3)*4', 20.0),
            ('-5', -5.0), ('-(2+3)', -5.0),
            ('2**10', 1024.0), ('2**50', 2**50),
            ('2**-1', 0.5), ('7//2', 3.0), ('7%3', 1.0),
            ('((((1+2))))', 3.0),
        ]:
            with self.subTest(expr=expr):
                self.assertAlmostEqual(self.t.eval_expr(expr), expected)

    def test_constants(self):
        for name, val in [('PI', math.pi), ('pi', math.pi),
                          ('TAU', math.tau), ('E', math.e),
                          ('SQRT2', math.sqrt(2))]:
            with self.subTest(name=name):
                self.assertAlmostEqual(self.t.eval_expr(name), val)

    def test_functions(self):
        for expr, expected in [
            ('sin(PI/2)', 1.0), ('cos(0)', 1.0), ('sqrt(16)', 4.0),
            ('abs(-7)', 7.0), ('log(E)', 1.0), ('exp(0)', 1.0),
            ('sqrt(abs(-9))', 3.0),
            ('atan2(1, 1)', math.pi / 4),
            ('ceil(2.3)', 3.0), ('floor(2.7)', 2.0),
            ('sin(PI/4)**2 + cos(PI/4)**2', 1.0),
        ]:
            with self.subTest(expr=expr):
                self.assertAlmostEqual(self.t.eval_expr(expr), expected)

    def test_variables(self):
        self.t.variables['theta'] = 1.5
        self.assertAlmostEqual(self.t.eval_expr('theta'), 1.5)
        self.assertAlmostEqual(self.t.eval_expr('theta*2'), 3.0)
        self.assertAlmostEqual(self.t._eval_with_vars('x + 1', {'x': 10}), 11.0)

    def test_conditions(self):
        ec = self.t._eval_condition
        for expr, vars_, expected in [
            ('5 > 3', {}, True), ('2 > 7', {}, False), ('3 == 3', {}, True),
            ('3 != 4', {}, True), ('1 <= 1', {}, True),
            ('3 <> 4', {}, True), ('1 AND 1', {}, True),
            ('1 AND 0', {}, False), ('0 OR 1', {}, True),
            ('x > 0', {'x': 5}, True), ('x > 0', {'x': -1}, False),
            ('1 AND 1 AND 1', {}, True), ('1 AND 0 AND 1', {}, False),
            ('0 OR 0 OR 1', {}, True),
            ('NOT 0', {}, True), ('NOT 1', {}, False),
            ('3 >= 3', {}, True), ('3 <= 3', {}, True), ('3 > 3', {}, False),
        ]:
            with self.subTest(expr=expr, vars_=vars_):
                self.assertEqual(bool(ec(expr, vars_)), expected)

    def test_arrays(self):
        self.t.arrays['data'] = [10.0, 20.0, 30.0]
        self.assertAlmostEqual(self.t._safe_eval('data(1)'), 20.0)
        self.t.arrays['arr'] = [5.0, 15.0, 25.0]
        self.assertAlmostEqual(self.t._safe_eval('arr[2]'), 25.0)

    def test_eval_errors(self):
        with self.assertRaises(ValueError):
            self.t._safe_eval('')
        with self.assertRaises(ValueError):
            self.t._safe_eval('nonexistent_var')
        with self.assertRaises((ValueError, ZeroDivisionError)):
            self.t.eval_expr('1/0')

    def test_parse_matrix(self):
        m = self.t._parse_matrix('[[1,0],[0,-1]]')
        self.assertEqual(len(m), 2)
        self.assertAlmostEqual(m[0][0], 1+0j)
        self.assertAlmostEqual(m[1][1], -1+0j)
        self.assertAlmostEqual(self.t._parse_matrix('[[1,0],[0,1j]]')[1][1], 1j)


# ---------------------------------------------------------------------------
# 2. TestSecurity
# ---------------------------------------------------------------------------
class TestSecurity(unittest.TestCase):
    """Security boundaries: blocked expressions, path sanitization,
    traversal prevention, absolute-path rejection on I/O commands."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_blocked_expressions(self):
        for expr in [
            '__import__("os")', 'exec("print(1)")', 'open("/etc/passwd")',
            'getattr(int, "__bases__")', '().__class__.__bases__',
            '"".join(["a"])', '(lambda: 1)()',
        ]:
            with self.subTest(expr=expr):
                with self.assertRaises(ValueError):
                    self.t._safe_eval(expr)
        # string literals should pass
        self.assertEqual(self.t._safe_eval('"hello"'), "hello")

    def test_path_sanitization(self):
        for bad in ["file\x00.qb", "file\x01.qb", "", "../../etc/passwd",
                     "foo/../../../bar", "C:\\Windows\\system32\\file.qb"]:
            with self.subTest(path=repr(bad)):
                with self.assertRaises(ValueError):
                    self.t._sanitize_path(bad)
        for good, expected in [("test.qb", "test.qb"), ("  test.qb  ", "test.qb"),
                               ("examples/bell.qb", "examples/bell.qb"),
                               ("dir/sub/file.qb", "dir/sub/file.qb")]:
            with self.subTest(path=good):
                self.assertEqual(self.t._sanitize_path(good), expected)

    @unittest.skipIf(os.name == 'nt', "Unix paths not absolute on Windows")
    def test_unix_absolute_rejected(self):
        with self.assertRaises(ValueError):
            self.t._sanitize_path("/etc/passwd")

    def test_io_commands_reject_absolute(self):
        # INCLUDE
        abs_path = 'C:\\Windows\\system32\\file.qb' if os.name == 'nt' else '/etc/passwd'
        _, out = capture(self.t.cmd_include, abs_path)
        self.assertIn('absolute', out.lower())
        # SAVE
        self.t.program = {10: 'H 0'}
        abs_save = 'C:\\tmp\\evil.qb' if os.name == 'nt' else '/tmp/evil.qb'
        _, out = capture(self.t.cmd_save, abs_save)
        self.assertIn('absolute', out.lower())
        # CSV and EXPORT need results first
        self.t.num_qubits = 1
        self.t.program = {10: 'X 0', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        abs_csv = 'C:\\tmp\\evil.csv' if os.name == 'nt' else '/tmp/evil.csv'
        _, out = capture(self.t.cmd_csv, abs_csv)
        self.assertIn('ERROR', out)
        abs_qasm = 'C:\\tmp\\evil.qasm' if os.name == 'nt' else '/tmp/evil.qasm'
        _, out = capture(self.t.cmd_export, abs_qasm)
        self.assertIn('ERROR', out)


# ---------------------------------------------------------------------------
# 3. TestGates
# ---------------------------------------------------------------------------
class TestGates(unittest.TestCase):
    """Gate matrices, dispatch, registry, custom gates, CTRL validation."""

    def setUp(self):
        self.t = QBasicTerminal()

    def _is_unitary(self, m):
        return np.allclose(m @ m.conj().T, np.eye(m.shape[0]))

    def test_all_gates_unitary(self):
        # 0-param gates
        for g in [g for g, (np_, nq) in GATE_TABLE.items()
                   if np_ == 0 and g not in GATE_ALIASES]:
            with self.subTest(gate=g):
                self.assertTrue(self._is_unitary(_np_gate_matrix(g)), f"{g} not unitary")
        # 1-param gates
        for g in ['RX', 'RY', 'RZ', 'P']:
            self.assertTrue(self._is_unitary(_np_gate_matrix(g, (0.5,))))
        for g in ['CRX', 'CRY', 'CRZ', 'CP', 'RXX', 'RYY', 'RZZ']:
            self.assertTrue(self._is_unitary(_np_gate_matrix(g, (0.7,))))
        # U gate
        self.assertTrue(self._is_unitary(_np_gate_matrix('U', (0.5, 0.3, 0.1))))

    def test_parametric_2q_gates_match_qiskit(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        for gate_name, qiskit_method in {'RXX': 'rxx', 'RYY': 'ryy', 'RZZ': 'rzz',
                                          'CRX': 'crx', 'CRY': 'cry', 'CRZ': 'crz',
                                          'CP': 'cp'}.items():
            our = _np_gate_matrix(gate_name, (0.7,))
            qc = QuantumCircuit(2)
            getattr(qc, qiskit_method)(0.7, 0, 1)
            qk = Operator(qc).data
            self.assertTrue(np.allclose(our, qk, atol=1e-6) or
                            np.allclose(swap @ our @ swap, qk, atol=1e-6),
                            f"{gate_name} mismatch")

    def test_specific_gate_behaviors(self):
        # Hadamard
        r = _np_gate_matrix('H') @ np.array([1, 0], dtype=complex)
        self.assertAlmostEqual(abs(r[0])**2, 0.5)
        self.assertAlmostEqual(abs(r[1])**2, 0.5)
        # Pauli-X
        x = _np_gate_matrix('X')
        np.testing.assert_array_almost_equal(x @ [1, 0], [0, 1])
        np.testing.assert_array_almost_equal(x @ [0, 1], [1, 0])
        # CNOT
        cx = _np_gate_matrix('CX')
        np.testing.assert_array_almost_equal(cx @ [0, 0, 1, 0], [0, 0, 0, 1])
        np.testing.assert_array_almost_equal(cx @ [1, 0, 0, 0], [1, 0, 0, 0])
        # Toffoli
        ccx = _np_gate_matrix('CCX')
        s = np.zeros(8, dtype=complex); s[6] = 1
        self.assertAlmostEqual(abs((ccx @ s)[7])**2, 1.0)
        s2 = np.zeros(8, dtype=complex); s2[4] = 1
        self.assertAlmostEqual(abs((ccx @ s2)[4])**2, 1.0)
        # Aliases
        np.testing.assert_array_almost_equal(_np_gate_matrix('CNOT'), _np_gate_matrix('CX'))
        np.testing.assert_array_almost_equal(_np_gate_matrix('TOFFOLI'), _np_gate_matrix('CCX'))
        np.testing.assert_array_almost_equal(_np_gate_matrix('FREDKIN'), _np_gate_matrix('CSWAP'))
        # Unknown gate
        with self.assertRaises(ValueError):
            _np_gate_matrix('NONEXISTENT')

    def test_dispatch_all_gates(self):
        from qiskit import QuantumCircuit
        for gate, (n_params, n_qubits) in GATE_TABLE.items():
            if gate in GATE_ALIASES:
                continue
            qc = QuantumCircuit(max(n_qubits, 3))
            try:
                self.t._apply_gate(qc, gate, [0.5]*n_params, list(range(n_qubits)))
            except Exception as e:
                self.fail(f"Gate {gate} failed: {e}")

    def test_custom_gates(self):
        from qiskit import QuantumCircuit
        m = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex) / 2
        self.t._custom_gates['SQX'] = m
        qc = QuantumCircuit(1)
        self.t._apply_gate(qc, 'SQX', [], [0])
        self.assertEqual(qc.size(), 1)
        # Leak check
        self.t._custom_gates['LEAK'] = m
        self.assertIsNone(QBasicTerminal()._gate_info('LEAK'))
        # Non-unitary rejected
        self.t.num_qubits = 1
        self.t.program = {10: 'UNITARY BAD = [[1,0],[0,0]]', 20: 'MEASURE'}
        _, out = capture(self.t.cmd_run)
        self.assertIn('not unitary', out)
        self.assertNotIn('BAD', self.t._custom_gates)
        # Valid custom gate in program (sqrt(X) applied twice = X)
        t2 = QBasicTerminal()
        t2.num_qubits = 1; t2.shots = 100
        t2.program = {
            10: 'UNITARY SQX = [[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]]',
            20: 'SQX 0', 30: 'SQX 0', 40: 'MEASURE',
        }
        capture(t2.cmd_run)
        self.assertIn('SQX', t2._custom_gates)
        self.assertEqual(t2.last_counts.get('1', 0), 100)

    def test_ctrl_wrong_target_count(self):
        capture(self.t.cmd_locc, '4 4')
        self.t.program = {10: '@A CTRL SWAP 0, 1', 20: 'MEASURE'}
        _, out = capture(self.t.cmd_run)
        self.assertIn('target', out.lower())


# ---------------------------------------------------------------------------
# 4. TestStatevector
# ---------------------------------------------------------------------------
class TestStatevector(unittest.TestCase):
    """Statevector operations and numerical stability."""

    def test_apply_and_measure(self):
        # apply H
        sv = np.array([1, 0], dtype=complex)
        result = np.ascontiguousarray(
            _apply_gate_np(sv, _np_gate_matrix('H'), [0], 1)).ravel()
        self.assertAlmostEqual(abs(result[0])**2, 0.5)
        self.assertAlmostEqual(abs(result[1])**2, 0.5)
        # measure collapses
        sv2 = np.array([1, 1], dtype=complex) / np.sqrt(2)
        outcome, new_sv = _measure_np(sv2, 0, 1)
        self.assertIn(outcome, [0, 1])
        self.assertAlmostEqual(max(np.abs(new_sv)**2), 1.0)
        # sample |0> always 0
        counts = _sample_np(np.array([1, 0], dtype=complex), 1, 1000)
        self.assertEqual(list(counts.keys()), ['0'])
        self.assertEqual(counts['0'], 1000)

    def test_bell_state_correlations(self):
        sv = np.zeros(4, dtype=complex)
        sv[0] = sv[3] = 1 / np.sqrt(2)
        counts = _sample_np(sv, 2, 10000)
        for state in counts:
            self.assertIn(state, ['00', '11'])
        self.assertGreater(counts.get('00', 0), 3000)
        self.assertGreater(counts.get('11', 0), 3000)

    def test_measure_near_zero_state(self):
        sv = np.array([1e-310, 1e-310], dtype=complex)
        outcome, new_sv = _measure_np(sv, 0, 1)
        self.assertEqual(outcome, 0)
        self.assertEqual(new_sv.shape, (2,))
        self.assertAlmostEqual(np.sum(np.abs(new_sv)**2), 1.0)


# ---------------------------------------------------------------------------
# 5. TestCommands
# ---------------------------------------------------------------------------
class TestCommands(unittest.TestCase):
    """REPL commands and variable substitution."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_qubit_and_shot_setup(self):
        _, out = capture(self.t.cmd_qubits, '8')
        self.assertEqual(self.t.num_qubits, 8)
        self.assertIn('8 QUBITS', out)
        _, out = capture(self.t.dispatch, 'QUBITS 50')
        self.assertIn('RANGE', out)
        _, out = capture(self.t.cmd_shots, '512')
        self.assertEqual(self.t.shots, 512)

    def test_new_let_clear(self):
        self.t.program = {10: "H 0"}; self.t.variables['x'] = 5
        capture(self.t.cmd_new)
        self.assertEqual(len(self.t.program), 0)
        self.assertEqual(len(self.t.variables), 0)
        capture(self.t.cmd_let, 'angle = PI/4')
        self.assertAlmostEqual(self.t.variables['angle'], math.pi / 4)
        self.t.variables['x'] = 5
        capture(self.t.cmd_clear, 'x')
        self.assertNotIn('x', self.t.variables)
        self.t.arrays['data'] = [1, 2, 3]
        capture(self.t.cmd_clear, 'data')
        self.assertNotIn('data', self.t.arrays)

    def test_reg(self):
        self.t.num_qubits = 8
        capture(self.t.cmd_reg, 'data 3')
        self.assertIn('data', self.t.registers)
        self.assertEqual(self.t.registers['data'], (0, 3))
        self.t.num_qubits = 4
        capture(self.t.dispatch, 'REG big 10')
        self.assertNotIn('big', self.t.registers)

    def test_def_subroutine(self):
        capture(self.t.cmd_def, 'BELL = H 0 : CX 0,1')
        self.assertIn('BELL', self.t.subroutines)
        self.assertEqual(len(self.t.subroutines['BELL']['body']), 2)
        capture(self.t.cmd_def, 'ROT(angle, q) = RX angle, q')
        self.assertIn('ROT', self.t.subroutines)
        self.assertEqual(self.t.subroutines['ROT']['params'], ['angle', 'q'])
        _, out = capture(self.t.dispatch, 'DEF H = X 0')
        self.assertIn('CANNOT REDEFINE', out)

    def test_program_editing(self):
        self.t.process('10 H 0')
        self.assertIn(10, self.t.program)
        self.assertEqual(self.t.program[10], 'H 0')
        self.t.process('20 CX 0,1')
        capture(self.t.cmd_undo)
        self.assertNotIn(20, self.t.program)
        self.assertIn(10, self.t.program)
        capture(self.t.process, '10')  # delete line
        self.assertNotIn(10, self.t.program)
        self.t.program = {10: 'H 0', 20: 'X 1', 30: 'Z 2'}
        capture(self.t.cmd_delete, '10-20')
        self.assertNotIn(10, self.t.program)
        self.assertIn(30, self.t.program)

    def test_renum(self):
        self.t.program = {5: 'H 0', 17: 'CX 0,1', 42: 'MEASURE'}
        capture(self.t.cmd_renum)
        self.assertEqual(sorted(self.t.program.keys()), [10, 20, 30])

    def test_method(self):
        capture(self.t.cmd_method, 'statevector')
        self.assertEqual(self.t.sim_method, 'statevector')
        capture(self.t.cmd_method, 'GPU')
        self.assertEqual(self.t.sim_device, 'GPU')

    def test_variable_substitution(self):
        self.assertIn('PI', self.t._substitute_vars('RX PI, 0', {}))
        r = self.t._substitute_vars('H 0', {'H': 99})
        self.assertIn('H', r); self.assertNotIn('99', r)
        r2 = self.t._substitute_vars('RX angle, qubit', {'angle': '0.5', 'qubit': '2'})
        self.assertIn('0.5', r2); self.assertIn('2', r2)


# ---------------------------------------------------------------------------
# 6. TestExecution
# ---------------------------------------------------------------------------
class TestExecution(unittest.TestCase):
    """Circuit execution, control flow, colon splitter, double-exec fix,
    FOR-loop floats, RENUM targets."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_bell_state(self):
        self.t.num_qubits = 2; self.t.shots = 1000
        self.t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        for s in self.t.last_counts:
            self.assertIn(s, ['00', '11'])

    def test_x_gate_and_statevector(self):
        self.t.num_qubits = 1; self.t.shots = 100
        self.t.program = {10: 'X 0', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertEqual(self.t.last_counts['1'], 100)
        self.t.program = {10: 'H 0', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_sv)
        probs = np.abs(self.t.last_sv)**2
        self.assertAlmostEqual(probs[0], 0.5, places=2)

    def test_loops(self):
        # FOR/NEXT
        t = QBasicTerminal(); t.num_qubits = 4
        t.program = {10: 'FOR I = 0 TO 3', 20: 'H I', 30: 'NEXT I', 40: 'MEASURE'}
        capture(t.cmd_run)
        self.assertGreater(len(t.last_counts), 1)
        # WHILE/WEND
        t2 = QBasicTerminal(); t2.num_qubits = 3
        t2.program = {10: 'LET n = 0', 20: 'WHILE n < 3', 30: 'H n',
                      40: 'LET n = n + 1', 50: 'WEND', 60: 'MEASURE'}
        capture(t2.cmd_run)
        self.assertIsNotNone(t2.last_counts)
        # Float STEP
        t3 = QBasicTerminal(); t3.num_qubits = 1; t3.variables['count'] = 0
        t3.program = {10: 'LET count = 0', 20: 'FOR theta = 0 TO 1 STEP 0.5',
                      30: 'LET count = count + 1', 40: 'NEXT theta', 50: 'MEASURE'}
        capture(t3.cmd_run)
        self.assertEqual(t3.variables['count'], 3)
        # Integer step
        t4 = QBasicTerminal(); t4.num_qubits = 4
        t4.program = {10: 'FOR I = 0 TO 3', 20: 'H I', 30: 'NEXT I', 40: 'MEASURE'}
        capture(t4.cmd_run)
        self.assertIsNotNone(t4.last_counts)
        # Loop limit
        t5 = QBasicTerminal(); t5.num_qubits = 1; t5._max_iterations = 100
        t5.program = {10: 'GOTO 10'}
        _, out = capture(t5.cmd_run)
        self.assertIn('LOOP LIMIT', out)

    def test_branching(self):
        # IF/THEN/ELSE
        t = QBasicTerminal(); t.num_qubits = 2; t.variables['flag'] = 1
        t.program = {10: 'IF flag == 1 THEN H 0 ELSE X 0', 20: 'MEASURE'}
        capture(t.cmd_run)
        self.assertIsNotNone(t.last_counts)
        # GOTO
        t2 = QBasicTerminal(); t2.num_qubits = 1
        t2.program = {10: 'GOTO 30', 20: 'X 0', 30: 'H 0', 40: 'MEASURE'}
        capture(t2.cmd_run)
        self.assertGreater(len(t2.last_counts), 1)
        # GOSUB/RETURN
        t3 = QBasicTerminal(); t3.num_qubits = 2
        t3.program = {10: 'GOSUB 100', 20: 'MEASURE', 30: 'END',
                      100: 'H 0', 110: 'CX 0,1', 120: 'RETURN'}
        capture(t3.cmd_run)
        for s in t3.last_counts:
            self.assertIn(s, ['00', '11'])
        # END stops
        t4 = QBasicTerminal(); t4.num_qubits = 1
        t4.program = {10: 'H 0', 20: 'END', 30: 'X 0'}
        capture(t4.cmd_run)
        self.assertIsNone(t4.last_counts)

    def test_colon_subroutine_register(self):
        t = QBasicTerminal(); t.num_qubits = 2
        t.program = {10: 'H 0 : CX 0,1', 20: 'MEASURE'}
        capture(t.cmd_run)
        for s in t.last_counts:
            self.assertIn(s, ['00', '11'])
        # Subroutine expansion
        capture(t.cmd_def, 'BELL = H 0 : CX 0,1')
        t.program = {10: 'BELL', 20: 'MEASURE'}
        capture(t.cmd_run)
        for s in t.last_counts:
            self.assertIn(s, ['00', '11'])
        # Register notation
        t2 = QBasicTerminal(); t2.num_qubits = 4
        capture(t2.cmd_reg, 'data 2'); capture(t2.cmd_reg, 'anc 2')
        t2.program = {10: 'H data[0]', 20: 'CX data[0],anc[0]', 30: 'MEASURE'}
        capture(t2.cmd_run)
        self.assertIsNotNone(t2.last_counts)

    def test_print_and_dim(self):
        t = QBasicTerminal(); t.num_qubits = 1
        t.program = {10: 'PRINT "hello"', 20: 'H 0'}
        _, out = capture(t.cmd_run)
        self.assertIn('hello', out)
        t2 = QBasicTerminal(); t2.num_qubits = 1
        t2.program = {10: 'DIM vals(3)', 20: 'LET vals(0) = PI', 30: 'H 0'}
        capture(t2.cmd_run)
        self.assertAlmostEqual(t2.arrays['vals'][0], math.pi)

    def test_double_execution_fix(self):
        t = QBasicTerminal(); t.num_qubits = 1; t.variables['counter'] = 0
        t.program = {10: 'LET counter = counter + 1', 20: 'H 0', 30: 'MEASURE'}
        capture(t.cmd_run)
        self.assertEqual(t.variables['counter'], 1)
        t2 = QBasicTerminal(); t2.num_qubits = 1; t2.variables['x'] = 10
        t2.program = {10: 'LET x = x * 2', 20: 'H 0', 30: 'MEASURE'}
        capture(t2.cmd_run)
        self.assertEqual(t2.variables['x'], 20)

    def test_cf_helpers(self):
        self.t.arrays['data'] = [0.0, 0.0, 0.0]
        self.assertIsNotNone(self.t._cf_let_array('LET data(1) = 42', {}))
        self.assertAlmostEqual(self.t.arrays['data'][1], 42.0)
        rv = {}
        self.assertIsNotNone(self.t._cf_let_var('LET x = 10', rv))
        self.assertEqual(rv['x'], 10)
        self.assertEqual(self.t._cf_goto('GOTO 30', [10, 20, 30, 40]), (True, 2))
        with self.assertRaises(RuntimeError):
            self.t._cf_goto('GOTO 999', [10, 20, 30])
        rv2 = {}; ls = []
        self.t._cf_for('FOR I = 0 TO 2', rv2, ls, 0)
        self.assertEqual(rv2['I'], 0)
        self.t._cf_next('NEXT I', rv2, ls)
        self.assertEqual(rv2['I'], 1)
        for fn, args in [(self.t._cf_let_array, ('H 0', {})),
                         (self.t._cf_let_var, ('H 0', {})),
                         (self.t._cf_goto, ('H 0', [10])),
                         (self.t._cf_for, ('H 0', {}, [], 0))]:
            self.assertIsNone(fn(*args))

    def test_colon_splitter(self):
        sp = QBasicTerminal._split_colon_stmts
        for input_, expected in [
            ('H 0 : CX 0,1', ['H 0', 'CX 0,1']),
            ('@A H 0 : X 1 : CX 0,1', ['@A H 0', '@A X 1', '@A CX 0,1']),
            ('@A H 0 : SEND A 0 -> x', ['@A H 0', 'SEND A 0 -> x']),
            ('@A H 0 : IF x THEN @B X 0', ['@A H 0', 'IF x THEN @B X 0']),
            ('@A H 0 : @B X 0', ['@A H 0', '@B X 0']),
            ('H 0 : : X 1', ['H 0', 'X 1']),
        ]:
            with self.subTest(input_=input_):
                self.assertEqual(sp(input_), expected)

    def test_renum_targets(self):
        self.t.program = {5: 'GOTO 15', 10: 'X 0', 15: 'H 0'}
        capture(self.t.cmd_renum)
        self.assertIn('GOTO 30', self.t.program[10])
        self.t.program = {5: 'GOSUB 15', 10: 'END', 15: 'H 0', 20: 'RETURN'}
        capture(self.t.cmd_renum)
        self.assertIn('GOSUB 30', self.t.program[10])
        # preserves execution
        self.t.num_qubits = 1
        self.t.program = {5: 'GOTO 15', 10: 'X 0', 15: 'H 0', 20: 'MEASURE'}
        capture(self.t.cmd_renum)
        capture(self.t.cmd_run)
        self.assertGreater(len(self.t.last_counts), 1)


# ---------------------------------------------------------------------------
# 7. TestDisplay
# ---------------------------------------------------------------------------
class TestDisplay(unittest.TestCase):
    """Display formatting, display values, banner."""

    def setUp(self):
        self.t = QBasicTerminal()
        self.t.num_qubits = 2
        self.t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}
        capture(self.t.cmd_run)

    def test_display_commands(self):
        for cmd, substr in [
            (lambda: self.t.cmd_state(), 'Statevector'),
            (lambda: self.t.cmd_hist(), '%'),
            (lambda: self.t.cmd_probs(), 'Probability'),
            (lambda: self.t.cmd_bloch('0'), 'Qubit 0'),
            (lambda: self.t.cmd_decompose(), 'Circuit'),
            (lambda: self.t.cmd_csv(''), 'state,count,probability'),
            (lambda: self.t.cmd_density(), 'Density matrix'),
            (lambda: self.t.cmd_expect('Z 0'), '<Z>'),
            (lambda: self.t.cmd_entropy('0'), 'entropy'),
        ]:
            _, out = capture(cmd)
            self.assertIn(substr, out)
        _, out = capture(self.t.cmd_circuit)
        self.assertTrue(len(out) > 0)
        _, out = capture(self.t.cmd_export, '')
        self.assertTrue('OPENQASM' in out or 'qubit' in out or 'EXPORT' in out)

    def test_display_values(self):
        # |+> amplitudes
        t2 = QBasicTerminal(); t2.num_qubits = 1
        t2.program = {10: 'H 0', 20: 'MEASURE'}; capture(t2.cmd_run)
        _, out = capture(t2.cmd_state)
        self.assertIn('0.5000', out); self.assertIn('+0.7071', out)
        # deterministic hist
        t3 = QBasicTerminal(); t3.num_qubits = 1
        t3.program = {10: 'X 0', 20: 'MEASURE'}; capture(t3.cmd_run)
        _, out = capture(t3.cmd_hist)
        self.assertIn('100.0%', out)
        # superposition probs
        _, out = capture(t2.cmd_probs)
        self.assertIn('50.00%', out)
        # entropy: product vs bell
        t4 = QBasicTerminal(); t4.num_qubits = 2
        t4.program = {10: 'X 0', 20: 'MEASURE'}; capture(t4.cmd_run)
        _, out = capture(t4.cmd_entropy, '0')
        self.assertIn('0.000000', out); self.assertIn('separable', out)
        t5 = QBasicTerminal(); t5.num_qubits = 2
        t5.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}; capture(t5.cmd_run)
        _, out = capture(t5.cmd_entropy, '0')
        self.assertIn('1.000000', out); self.assertIn('entangled', out)

    def test_banner(self):
        _, out = capture(QBasicTerminal().print_banner)
        self.assertIn('Quantum BASIC', out); self.assertIn('qubits', out)


# ---------------------------------------------------------------------------
# 8. TestLOCC
# ---------------------------------------------------------------------------
class TestLOCC(unittest.TestCase):
    """LOCC engine: 2-party, N-party, terminal integration, unequal registers,
    split independence, Born rule, CTRL in LOCC, per-line error, demos."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_2party_engine(self):
        eng = LOCCEngine([3, 3])
        self.assertEqual(eng.n_regs, 2)
        self.assertEqual(eng.names, ['A', 'B'])
        self.assertEqual(eng.n_a, 3); self.assertEqual(eng.n_b, 3)
        eng_j = LOCCEngine([3, 3], joint=True)
        self.assertEqual(eng_j.n_total, 6)
        # apply, send, sample, mem, reset
        eng2 = LOCCEngine([2, 2])
        eng2.apply('A', 'H', (), [0])
        probs = np.abs(eng2.svs['A'].ravel())**2
        self.assertAlmostEqual(probs[0], 0.5)
        outcome = eng2.send('A', 0)
        self.assertIn(outcome, [0, 1])
        eng3 = LOCCEngine([2, 2])
        eng3.apply('A', 'X', (), [0])
        per_reg, _ = eng3.sample_joint(100)
        self.assertEqual(per_reg['A']['01'], 100)
        total, _ = LOCCEngine([20, 20]).mem_gb()
        self.assertGreater(total, 0)
        eng3.classical['x'] = 1; eng3.reset()
        a_probs = np.abs(eng3.svs['A'].ravel())**2
        self.assertAlmostEqual(a_probs[0], 1.0)
        self.assertEqual(len(eng3.classical), 0)

    def test_share_and_split_raises(self):
        eng_j = LOCCEngine([2, 2], joint=True)
        eng_j.share('A', 0, 'B', 0)
        _, joint = eng_j.sample_joint(10000)
        for state in joint:
            parts = state.split('|')
            self.assertEqual(parts[0][-1], parts[1][-1])
        with self.assertRaises(RuntimeError):
            LOCCEngine([2, 2]).share('A', 0, 'B', 0)

    def test_nparty_engine(self):
        eng3 = LOCCEngine([4, 4, 4])
        self.assertEqual(eng3.n_regs, 3)
        self.assertEqual(eng3.n_total, 12)
        eng4 = LOCCEngine([2, 3, 4, 5])
        self.assertEqual(eng4.n_regs, 4)
        self.assertEqual(eng4.sizes, [2, 3, 4, 5])
        # gates
        eng = LOCCEngine([3, 3, 3])
        eng.apply('A', 'H', (), [0]); eng.apply('B', 'X', (), [1])
        a_probs = np.abs(eng.svs['A'].ravel())**2
        b_probs = np.abs(eng.svs['B'].ravel())**2
        self.assertAlmostEqual(a_probs[0], 0.5)
        self.assertAlmostEqual(b_probs[2], 1.0)
        # send
        eng2 = LOCCEngine([2, 2, 2])
        eng2.apply('C', 'H', (), [0])
        self.assertIn(eng2.send('C', 0), [0, 1])
        # sample
        eng3s = LOCCEngine([2, 2, 2])
        eng3s.apply('A', 'X', (), [0]); eng3s.apply('B', 'X', (), [1])
        per_reg, _ = eng3s.sample_joint(100)
        self.assertEqual(per_reg['A']['01'], 100)
        self.assertEqual(per_reg['B']['10'], 100)
        # joint mode
        eng_j = LOCCEngine([2, 2, 2], joint=True)
        eng_j.apply('A', 'H', (), [0]); eng_j.apply('C', 'X', (), [0])
        per_reg, joint = eng_j.sample_joint(100)
        self.assertEqual(len(per_reg), 3)
        # share across A-C
        eng_j2 = LOCCEngine([2, 2, 2], joint=True)
        eng_j2.share('A', 0, 'C', 0)
        _, joint = eng_j2.sample_joint(10000)
        for state in joint:
            parts = state.split('|')
            self.assertEqual(parts[0][-1], parts[2][-1])
        # offsets, sizes, compat
        eng_o = LOCCEngine([3, 5, 7])
        self.assertEqual(eng_o.offsets, [0, 3, 8])
        self.assertEqual(eng_o.get_size('C'), 7)
        eng_c = LOCCEngine([5, 7, 9])
        self.assertEqual(eng_c.n_a, 5); self.assertEqual(eng_c.n_b, 7)

    def test_unequal_registers(self):
        eng2 = LOCCEngine([2, 5])
        self.assertEqual(eng2.n_total, 7); self.assertEqual(eng2.offsets, [0, 2])
        eng3 = LOCCEngine([1, 3, 5])
        self.assertEqual(eng3.offsets, [0, 1, 4])
        # joint sampling unequal 2-party
        ej = LOCCEngine([2, 3], joint=True)
        ej.apply('A', 'X', (), [0]); ej.apply('B', 'X', (), [2])
        per_reg, _ = ej.sample_joint(100)
        self.assertEqual(per_reg['A']['01'], 100)
        self.assertEqual(per_reg['B']['100'], 100)
        # joint sampling unequal 3-party
        ej3 = LOCCEngine([1, 2, 3], joint=True)
        ej3.apply('A', 'X', (), [0]); ej3.apply('C', 'X', (), [0])
        per_reg, _ = ej3.sample_joint(100)
        self.assertEqual(per_reg['A']['1'], 100)
        self.assertEqual(per_reg['B']['00'], 100)
        self.assertEqual(per_reg['C']['001'], 100)

    def test_split_independence(self):
        eng = LOCCEngine([2, 2])
        eng.apply('A', 'X', (), [0])
        b_probs = np.abs(eng.svs['B'].ravel())**2
        a_probs = np.abs(eng.svs['A'].ravel())**2
        self.assertAlmostEqual(b_probs[0], 1.0)
        self.assertAlmostEqual(a_probs[1], 1.0)
        eng2 = LOCCEngine([3, 3])
        eng2.apply('A', 'H', (), [0]); eng2.apply('A', 'CX', (), [0, 1])
        per_reg, _ = eng2.sample_joint(1000)
        self.assertEqual(per_reg['B']['000'], 1000)
        eng3 = LOCCEngine([2, 2, 2])
        eng3.apply('B', 'X', (), [0])
        per_reg, _ = eng3.sample_joint(100)
        self.assertEqual(per_reg['A']['00'], 100)
        self.assertEqual(per_reg['C']['00'], 100)

    def test_born_rule(self):
        outcomes = {0: 0, 1: 0}
        for _ in range(5000):
            eng = LOCCEngine([2, 2])
            eng.apply('A', 'H', (), [0])
            outcomes[eng.send('A', 0)] += 1
        ratio = outcomes[0] / 5000
        self.assertGreater(ratio, 0.45); self.assertLess(ratio, 0.55)
        outcomes2 = {0: 0, 1: 0}
        for _ in range(5000):
            eng = LOCCEngine([1, 1])
            eng.apply('A', 'RY', (math.pi / 3,), [0])
            outcomes2[eng.send('A', 0)] += 1
        r0 = outcomes2[0] / 5000
        self.assertGreater(r0, 0.70); self.assertLess(r0, 0.80)

    def test_terminal_integration(self):
        _, out = capture(self.t.cmd_locc, '4 4 4')
        self.assertTrue(self.t.locc_mode)
        self.assertEqual(self.t.locc.n_regs, 3)
        self.assertIn('C=4q', out)
        capture(self.t.cmd_locc, 'OFF')
        self.assertFalse(self.t.locc_mode)
        # 3-party run
        capture(self.t.cmd_locc, '2 2 2')
        self.t.shots = 50
        self.t.program = {10: '@A H 0', 20: '@B H 0', 30: '@C H 0', 40: 'MEASURE'}
        _, out = capture(self.t.cmd_run)
        for reg in ['Register A', 'Register B', 'Register C']:
            self.assertIn(reg, out)
        # SEND from any register
        capture(self.t.cmd_locc, '2 2 2')
        self.t.shots = 50
        self.t.program = {10: '@C H 0', 20: 'SEND C 0 -> c0',
                          30: 'IF c0 THEN @A X 0', 40: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)
        # STATUS
        capture(self.t.cmd_locc, '5 10 15')
        _, out = capture(self.t.cmd_locc, 'STATUS')
        self.assertIn('A=5q', out); self.assertIn('C=15q', out)
        # JOINT limit
        _, out = capture(self.t.cmd_locc, 'JOINT 20 20')
        self.assertIn('limited', out)
        # LOCCINFO
        capture(self.t.cmd_locc, '3 3 3')
        _, out = capture(self.t.cmd_loccinfo)
        self.assertIn('3 parties', out)
        # colon inheritance
        capture(self.t.cmd_locc, '2 2 2')
        self.t.shots = 50
        self.t.program = {10: '@C H 0 : X 1', 20: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)
        # per-line error
        capture(self.t.cmd_locc, '2 2')
        self.t.program = {10: '@A H 0', 20: '@A FOOBAR 0', 30: 'MEASURE'}
        _, out = capture(self.t.cmd_run)
        self.assertIn('LINE 20', out)
        # CTRL in LOCC
        capture(self.t.cmd_locc, '3 3')
        self.t.shots = 100
        self.t.program = {10: '@A X 0', 20: '@A CTRL H 0, 1', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)
        capture(self.t.cmd_locc, 'JOINT 3 3')
        self.t.shots = 100
        self.t.program = {10: '@A X 0', 20: '@A CTRL H 0, 1', 30: 'MEASURE'}
        capture(self.t.cmd_run)
        self.assertIsNotNone(self.t.last_counts)
        # 20-qubit LOCC
        capture(self.t.cmd_locc, '20 20 20')
        self.assertEqual(self.t.locc.n_total, 60)

    def test_locc_demos(self):
        _, out = capture(self.t.cmd_demo, 'LOCC-TELEPORT')
        self.assertIn('LOCC Teleportation', out)
        self.assertIsNotNone(self.t.last_counts)
        t2 = QBasicTerminal()
        _, out = capture(t2.cmd_demo, 'LOCC-COORD')
        self.assertIn('Classical Coordination', out)
        self.assertIsNotNone(t2.last_counts)


# ---------------------------------------------------------------------------
# 9. TestFileIO
# ---------------------------------------------------------------------------
class TestFileIO(unittest.TestCase):
    """Save/load, include depth, nested include, blocks export,
    run_script, DEF BEGIN."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_save_load(self):
        self.t.num_qubits = 3; self.t.shots = 512
        self.t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}
        path = '_test_save_load_roundtrip.qb'
        try:
            capture(self.t.cmd_save, path)
            t2 = QBasicTerminal(); capture(t2.cmd_load, path)
            self.assertEqual(t2.num_qubits, 3)
            self.assertEqual(t2.shots, 512)
            self.assertEqual(len(t2.program), 3)
        finally:
            if os.path.exists(path):
                os.unlink(path)
        # LOAD silent
        path2 = '_test_load_silent.qb'
        with open(path2, 'w') as f:
            f.write("QUBITS 2\n10 H 0\n")
        try:
            t3 = QBasicTerminal()
            _, out = capture(t3.cmd_load, path2)
            self.assertNotIn('READY', out); self.assertIn('LOADED', out)
        finally:
            os.unlink(path2)
        # directory rejected
        _, out = capture(self.t.cmd_load, '.')
        self.assertIn('directory', out.lower())

    def test_include_security(self):
        for cmd in ["EXPORT evil.qasm", "CSV evil.csv", "SAVE evil.qb", "LOAD something.qb"]:
            path = '_test_include_block.qb'
            with open(path, 'w') as f:
                f.write(f"10 H 0\n{cmd}\n20 X 0\n")
            try:
                _, out = capture(QBasicTerminal().cmd_include, path)
                self.assertIn('BLOCKED', out, f"{cmd} should be blocked")
            finally:
                os.unlink(path)
        # depth limit
        self.t._include_depth = 100
        _, out = capture(self.t.cmd_include, 'anything.qb')
        self.assertIn('DEPTH LIMIT', out)
        self.t._include_depth = 0
        # nonexistent
        _, out = capture(self.t.cmd_include, 'nonexistent_file_xyz.qb')
        self.assertIn('NOT FOUND', out)

    def test_nested_include(self):
        inner = '_test_inner.qb'; outer = '_test_outer.qb'
        with open(inner, 'w') as f:
            f.write("10 H 0\n")
        with open(outer, 'w') as f:
            f.write(f"INCLUDE {inner}\n20 X 1\n")
        try:
            _, out = capture(self.t.cmd_include, outer)
            self.assertIn(10, self.t.program)
            self.assertIn(20, self.t.program)
            self.assertNotIn('BLOCKED', out)
        finally:
            os.unlink(inner); os.unlink(outer)
        # self-including depth limit
        path = '_test_self.qb'
        with open(path, 'w') as f:
            f.write(f"10 H 0\nINCLUDE {path}\n")
        try:
            _, out = capture(QBasicTerminal().cmd_include, path)
            self.assertTrue('DEPTH LIMIT' in out or 'INCLUDE CYCLE' in out)
        finally:
            os.unlink(path)

    def test_run_script(self):
        # basic
        with tempfile.NamedTemporaryFile(suffix='.qb', delete=False, mode='w') as f:
            f.write("QUBITS 2\nSHOTS 512\n10 H 0\n20 CX 0,1\n30 MEASURE\n"); path = f.name
        try:
            t2 = QBasicTerminal(); capture(run_script, path, t2)
            self.assertEqual(t2.num_qubits, 2); self.assertEqual(t2.shots, 512)
            self.assertIn(10, t2.program)
        finally:
            os.unlink(path)
        # multiline DEF
        with tempfile.NamedTemporaryFile(suffix='.qb', delete=False, mode='w') as f:
            f.write("DEF BEGIN BELL\nH 0\nCX 0,1\nDEF END\n10 BELL\n"); path = f.name
        try:
            t3 = QBasicTerminal(); capture(run_script, path, t3)
            self.assertIn('BELL', t3.subroutines)
            self.assertEqual(len(t3.subroutines['BELL']['body']), 2)
        finally:
            os.unlink(path)
        # comments skipped
        with tempfile.NamedTemporaryFile(suffix='.qb', delete=False, mode='w') as f:
            f.write("# Comment\nQUBITS 3\n10 H 0\n"); path = f.name
        try:
            t4 = QBasicTerminal(); capture(run_script, path, t4)
            self.assertEqual(t4.num_qubits, 3)
        finally:
            os.unlink(path)

    def test_def_begin_in_script(self):
        # with params
        with tempfile.NamedTemporaryFile(suffix='.qb', delete=False, mode='w') as f:
            f.write("QUBITS 2\nDEF BEGIN ROT(angle, q)\nRX angle, q\nRZ angle, q\n"
                    "DEF END\n10 ROT PI/4, 0\n20 MEASURE\n"); path = f.name
        try:
            t2 = QBasicTerminal(); capture(run_script, path, t2)
            self.assertEqual(t2.subroutines['ROT']['params'], ['angle', 'q'])
            self.assertIsNotNone(t2.last_counts)
        finally:
            os.unlink(path)
        # builtin rejected
        with tempfile.NamedTemporaryFile(suffix='.qb', delete=False, mode='w') as f:
            f.write("DEF BEGIN H\nX 0\nDEF END\n"); path = f.name
        try:
            _, out = capture(run_script, path, QBasicTerminal())
            self.assertIn('CANNOT REDEFINE', out)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 10. TestDemos
# ---------------------------------------------------------------------------
class TestDemos(unittest.TestCase):
    """Demo circuits run and produce correct results."""

    def _run(self, name):
        t = QBasicTerminal()
        _, out = capture(t.cmd_demo, name)
        return t, out

    def test_demos_run(self):
        for name in ['BELL', 'GHZ', 'QFT', 'RANDOM']:
            with self.subTest(demo=name):
                t, _ = self._run(name)
                self.assertIsNotNone(t.last_counts)
        t, out = self._run('BELL')
        self.assertIn('Bell State', out)

    def test_demo_correctness(self):
        # Grover
        t, _ = self._run('GROVER')
        self.assertEqual(max(t.last_counts, key=t.last_counts.get), '101')
        # Bernstein-Vazirani
        t, _ = self._run('BERNSTEIN')
        self.assertTrue(max(t.last_counts, key=t.last_counts.get).endswith('1011'))
        # Superdense
        t, _ = self._run('SUPERDENSE')
        self.assertEqual(t.last_counts.get('11', 0), t.shots)
        # Deutsch
        t, _ = self._run('DEUTSCH')
        self.assertEqual(max(t.last_counts, key=t.last_counts.get)[-1], '1')
        # Teleport: all 8 outcomes possible
        t, _ = self._run('TELEPORT')
        for s in t.last_counts:
            self.assertEqual(len(s), 3)
        self.assertGreaterEqual(len(t.last_counts), 6)

    def test_demo_list(self):
        _, out = capture(QBasicTerminal().cmd_demo, 'LIST')
        self.assertIn('BELL', out); self.assertIn('LOCC-TELEPORT', out)


# ---------------------------------------------------------------------------
# 11. TestMeasurement
# ---------------------------------------------------------------------------
class TestMeasurement(unittest.TestCase):
    """Basis measurement, MEAS, SYNDROME, RESET."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_basis_measurements(self):
        for basis, var_prefix, prep in [
            ('MEASURE_X', 'mx_0', 'H 0'), ('MEASURE_Z', 'mz_0', 'X 0'),
            ('MEASURE_Y', 'my_0', 'H 0'),
        ]:
            with self.subTest(basis=basis):
                t = QBasicTerminal(); t.num_qubits = 2 if basis == 'MEASURE_X' else 1
                t.program = {10: prep, 20: f'{basis} 0', 30: 'MEASURE'}
                capture(t.cmd_run)
                self.assertIn(var_prefix, t.variables)

    def test_meas(self):
        # MEAS without LOCC mode should raise a build error
        t = QBasicTerminal(); t.num_qubits = 2
        t.program = {10: 'X 0', 20: 'MEAS 0 -> result', 30: 'MEASURE'}
        _, out = capture(t.cmd_run)
        self.assertIn('LOCC', out)
        self.assertIn('BUILD ERROR', out)
        # MEAS works in LOCC mode
        t2 = QBasicTerminal()
        capture(t2.cmd_locc, '2 2')
        t2.program = {10: '@A H 0', 20: 'SEND A 0 -> result', 30: 'MEASURE'}
        capture(t2.cmd_run)
        self.assertIn('result', t2.variables)

    def test_syndrome(self):
        t = QBasicTerminal(); t.num_qubits = 3
        t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'SYNDROME ZZ 0 1 -> s0', 40: 'MEASURE'}
        capture(t.cmd_run)
        self.assertIn('s0', t.variables)
        # bad pauli length
        t2 = QBasicTerminal(); t2.num_qubits = 4
        t2.program = {10: 'SYNDROME ZZZ 0 1 -> s', 20: 'MEASURE'}
        _, out = capture(t2.cmd_run)
        self.assertIn('ERROR', out)
        # XX stabilizer
        t3 = QBasicTerminal(); t3.num_qubits = 3
        t3.program = {10: 'H 0', 20: 'H 1', 30: 'SYNDROME XX 0 1 -> s0', 40: 'MEASURE'}
        capture(t3.cmd_run)
        self.assertIn('s0', t3.variables)

    def test_reset(self):
        for prog, expected_state in [
            ({10: 'X 0', 20: 'RESET 0', 30: 'MEASURE'}, '0'),
            ({10: 'H 0', 20: 'RESET 0', 30: 'MEASURE'}, '0'),
            ({10: 'X 0', 20: 'RESET 0', 30: 'RESET 0', 40: 'MEASURE'}, '0'),
        ]:
            with self.subTest(prog=prog):
                t = QBasicTerminal(); t.num_qubits = 1; t.shots = 100
                t.program = prog; capture(t.cmd_run)
                self.assertEqual(t.last_counts.get(expected_state, 0), 100)
        # preserves other qubits
        t2 = QBasicTerminal(); t2.num_qubits = 2; t2.shots = 100
        t2.program = {10: 'X 1', 20: 'H 0', 30: 'RESET 0', 40: 'MEASURE'}
        capture(t2.cmd_run)
        self.assertEqual(t2.last_counts.get('10', 0), 100)


# ---------------------------------------------------------------------------
# 12. TestMisc
# ---------------------------------------------------------------------------
class TestMisc(unittest.TestCase):
    """Edge cases, fixes, DEF multiline, subroutine recursion, RAM, noise,
    sweep, benchmark, step mode, entropy comma, debug."""

    def setUp(self):
        self.t = QBasicTerminal()

    def test_edge_cases(self):
        _, out = capture(self.t.cmd_run)
        self.assertIn('NOTHING', out)
        self.t.num_qubits = 2
        _, out = capture(self.t.dispatch, 'H 5')
        self.assertIn('OUT OF RANGE', out)
        _, out = capture(self.t.dispatch, 'FOOBAR 0')
        self.assertIn('UNKNOWN', out)
        # nested for, negative step (keep 4 qubits throughout)
        t4 = QBasicTerminal(); t4.num_qubits = 4
        t4.program = {10: 'FOR I = 0 TO 1', 20: 'FOR J = 2 TO 3',
                      30: 'CX I, J', 40: 'NEXT J', 50: 'NEXT I', 60: 'MEASURE'}
        capture(t4.cmd_run)
        self.assertIsNotNone(t4.last_counts)
        t4b = QBasicTerminal(); t4b.num_qubits = 4
        t4b.program = {10: 'FOR I = 3 TO 0 STEP -1', 20: 'H I',
                       30: 'NEXT I', 40: 'MEASURE'}
        capture(t4b.cmd_run)
        self.assertGreater(len(t4b.last_counts), 1)
        # immediate gate
        capture(t4b.run_immediate, 'H 0')
        self.assertIsNotNone(t4b.last_sv)
        # CTRL
        t3 = QBasicTerminal(); t3.num_qubits = 3
        t3.program = {10: 'X 0', 20: 'CTRL H 0, 1', 30: 'MEASURE'}
        capture(t3.cmd_run)
        self.assertIsNotNone(t3.last_counts)
        # INV
        t1 = QBasicTerminal(); t1.num_qubits = 1
        t1.program = {10: 'H 0', 20: 'INV H 0', 30: 'MEASURE'}
        capture(t1.cmd_run)
        self.assertEqual(t1.last_counts.get('0', 0), t1.shots)
        # BARRIER
        t2 = QBasicTerminal(); t2.num_qubits = 2
        t2.program = {10: 'H 0', 20: 'BARRIER', 30: 'CX 0,1', 40: 'MEASURE'}
        capture(t2.cmd_run)
        self.assertIsNotNone(t2.last_counts)
        # comments
        tc = QBasicTerminal(); tc.num_qubits = 1
        tc.program = {10: "REM comment", 20: 'H 0', 30: 'MEASURE'}
        capture(tc.cmd_run)
        self.assertIsNotNone(tc.last_counts)
        tc2 = QBasicTerminal(); tc2.num_qubits = 1
        tc2.program = {10: "' comment", 20: 'X 0', 30: 'MEASURE'}
        capture(tc2.cmd_run)
        self.assertEqual(tc2.last_counts.get('1', 0), tc2.shots)
        # 20-qubit
        t20 = QBasicTerminal(); t20.num_qubits = 20; t20.shots = 10; t20.program = {}
        ln = 10
        for i in range(20):
            t20.program[ln] = f'H {i}'; ln += 10
        t20.program[ln] = 'MEASURE'
        capture(t20.cmd_run)
        self.assertIsNotNone(t20.last_counts)

    def test_info_commands(self):
        _, out = capture(self.t.cmd_dir, '.'); self.assertTrue(len(out) > 0)
        _, out = capture(self.t.cmd_help)
        self.assertIn('QBASIC', out); self.assertIn('LOCC', out)
        self.t.variables['x'] = 42
        _, out = capture(self.t.cmd_vars); self.assertIn('x = 42', out)
        capture(self.t.cmd_def, 'BELL = H 0 : CX 0,1')
        _, out = capture(self.t.cmd_defs); self.assertIn('BELL', out)
        self.t.num_qubits = 4; capture(self.t.cmd_reg, 'data 2')
        _, out = capture(self.t.cmd_regs); self.assertIn('data', out)

    def test_noise(self):
        capture(self.t.cmd_noise, 'depolarizing 0.01')
        self.assertIsNotNone(self.t._noise_model)
        capture(self.t.cmd_noise, 'OFF')
        self.assertIsNone(self.t._noise_model)
        # heavy depolarizing adds errors (1 qubit)
        t1 = QBasicTerminal(); t1.num_qubits = 1; t1.shots = 1000
        t1.program = {10: 'X 0', 20: 'MEASURE'}
        capture(t1.cmd_run)
        self.assertEqual(t1.last_counts.get('1', 0), 1000)
        capture(t1.cmd_noise, 'depolarizing 0.3')
        capture(t1.cmd_run)
        self.assertGreater(t1.last_counts.get('0', 0), 10)
        # off restores clean
        capture(t1.cmd_noise, 'OFF')
        t1.shots = 100; capture(t1.cmd_run)
        self.assertEqual(t1.last_counts.get('1', 0), 100)
        # 3-qubit gate
        t3 = QBasicTerminal(); t3.num_qubits = 3; t3.shots = 1000
        t3.program = {10: 'X 0', 20: 'X 1', 30: 'CCX 0,1,2', 40: 'MEASURE'}
        capture(t3.cmd_run)
        self.assertEqual(t3.last_counts.get('111', 0), 1000)
        capture(t3.cmd_noise, 'depolarizing 0.3'); capture(t3.cmd_run)
        self.assertGreater(sum(v for k, v in t3.last_counts.items() if k != '111'), 0)
        capture(t3.cmd_noise, 'OFF')

    def test_sweep(self):
        self.t.num_qubits = 1
        self.t.program = {10: 'RX angle, 0', 20: 'MEASURE'}
        _, out = capture(self.t.cmd_sweep, 'angle 0 PI 3')
        self.assertIn('SWEEP', out)
        self.assertEqual(len([l for l in out.split('\n') if 'angle=' in l]), 3)
        _, out = capture(self.t.cmd_sweep, 'angle 0 1 2')
        self.assertIn('angle=  0.0000', out); self.assertIn('angle=  1.0000', out)
        _, out = capture(self.t.cmd_sweep, 'angle 0 PI 5')
        self.assertEqual(len([l for l in out.split('\n') if 'angle=' in l]), 5)
        _, out = capture(self.t.cmd_sweep, 'angle 0 PI 1')
        self.assertEqual(len([l for l in out.split('\n') if 'angle=' in l]), 1)
        _, out = capture(self.t.cmd_sweep, 'angle 0 PI 0')
        self.assertIn('at least 1', out)

    def test_ram(self):
        _, out = capture(self.t.cmd_ram)
        self.assertTrue('RAM' in out or 'psutil' in out)
        if 'psutil' not in out:
            self.assertIn('qubits', out)
        capture(self.t.cmd_locc, '4 4')
        _, out = capture(self.t.cmd_ram)
        if 'psutil' not in out:
            self.assertIn('LOCC', out)

    def test_benchmark(self):
        _, out = capture(self.t.cmd_bench)
        self.assertIn('Benchmark', out)

    def test_def_multiline(self):
        import unittest.mock
        with unittest.mock.patch('builtins.input', side_effect=['H 0', 'CX 0,1', 'DEF END']):
            capture(self.t.cmd_def, 'BEGIN BELL')
        self.assertIn('BELL', self.t.subroutines)
        self.assertEqual(len(self.t.subroutines['BELL']['body']), 2)
        t2 = QBasicTerminal()
        with unittest.mock.patch('builtins.input', side_effect=['RX angle, q', 'RZ angle, q', 'END']):
            capture(t2.cmd_def, 'BEGIN ROT(angle, q)')
        self.assertEqual(t2.subroutines['ROT']['params'], ['angle', 'q'])
        t3 = QBasicTerminal()
        with unittest.mock.patch('builtins.input', side_effect=KeyboardInterrupt):
            capture(t3.cmd_def, 'BEGIN FOO')
        self.assertNotIn('FOO', t3.subroutines)

    def test_subroutine_recursion(self):
        t = QBasicTerminal(); t.num_qubits = 2
        capture(t.cmd_def, 'LOOP = LOOP')
        t.program = {10: 'LOOP', 20: 'MEASURE'}
        _, out = capture(t.cmd_run)
        self.assertTrue(any(w in out.lower() for w in ['recursion', 'loop limit', 'depth']))

    def test_step_mode(self):
        import unittest.mock
        t = QBasicTerminal(); t.num_qubits = 2; t.program = {10: 'H 0', 20: 'CX 0,1'}
        with unittest.mock.patch('builtins.input', side_effect=['', '', 'Q']):
            _, out = capture(t.cmd_step)
        self.assertIn('STEP MODE', out)
        _, out = capture(QBasicTerminal().cmd_step)
        self.assertIn('NOTHING', out)

    def test_entropy_comma(self):
        t = QBasicTerminal(); t.num_qubits = 3
        t.program = {10: 'H 0', 20: 'CX 0,1', 30: 'MEASURE'}
        capture(t.cmd_run)
        _, out = capture(t.cmd_entropy, '0, 1')
        self.assertIn('entropy', out)

    def test_new_fixes(self):
        import qbasic_core
        self.assertTrue(hasattr(qbasic_core, 'QBasicTerminal'))
        t = QBasicTerminal(); t.num_qubits = 2
        t.program = {10: 'X 0', 20: 'MEAS 0 -> r', 30: 'MEASURE'}
        _, out = capture(t.cmd_run)
        self.assertIn('LOCC', out); self.assertIn('BUILD ERROR', out)
        from qbasic_core.protocol import TerminalProtocol
        self.assertIsInstance(t, TerminalProtocol)
        from qbasic_core.engine import ExecOutcome, ExecResult
        self.assertIs(ExecResult.ADVANCE, ExecResult.ADVANCE)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    unittest.main(verbosity=2)
