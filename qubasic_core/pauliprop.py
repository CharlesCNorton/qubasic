"""QUBASIC Pauli propagation (sparse Pauli dynamics).

Estimate <O> for a Pauli observable by backpropagating it through the circuit in
the Heisenberg picture, O -> G^dag O G gate by gate, keeping it as a sparse sum
of Paulis and truncating terms below a coefficient threshold. This reaches far
larger circuits than a statevector when the propagated observable stays sparse,
and is exact (matches EXPECT) when nothing is truncated. Pure numpy, no engine.

  PAULIPROP <pauli> [qubits] [threshold]
"""

from __future__ import annotations

import re

import numpy as np

_P1 = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


class PauliPropMixin:
    """Sparse Pauli-dynamics observable estimator for QBasicTerminal.

    Requires: TerminalProtocol — uses self.num_qubits, self.build_circuit(),
    self._eval_with_vars(), self.io.
    """

    @staticmethod
    def _local_pauli_matrix(chars):
        """Kron of single-qubit Paulis with the LAST char as most significant,
        matching Qiskit's little-endian gate-matrix convention (first qubit = LSB)."""
        mat = np.array([[1]], dtype=complex)
        for c in reversed(chars):
            mat = np.kron(mat, _P1[c])
        return mat

    def _conjugate_term(self, pauli: list, coeff: complex, qubits: list, Gdag_x_G):
        """Conjugate one Pauli term on the given gate qubits, returning new terms.

        Gdag_x_G(localPauliMatrix) -> the matrix G^dag P G; we decompose it back
        into local Paulis via traces and distribute over the gate's qubits."""
        k = len(qubits)
        local = [pauli[q] for q in qubits]
        M = Gdag_x_G(self._local_pauli_matrix(local))
        out = []
        import itertools
        for combo in itertools.product('IXYZ', repeat=k):
            basis = self._local_pauli_matrix(combo)
            c = np.trace(basis.conj().T @ M) / (2 ** k)
            if abs(c) < 1e-12:
                continue
            new = list(pauli)
            for q, ch in zip(qubits, combo):
                new[q] = ch
            out.append((new, coeff * c))
        return out

    def cmd_pauliprop(self, rest: str) -> None:
        """PAULIPROP <pauli> [qubits] [threshold] — Heisenberg-picture <O> estimate.

        Backpropagates the Pauli observable through the program's circuit, pruning
        terms with |coefficient| below the threshold (default 1e-10, i.e. exact)."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: PAULIPROP <pauli> [qubits] [threshold]")
            return
        if not self.program:
            self.io.writeln("?NOTHING TO PROPAGATE — enter a program first")
            return
        n = self.num_qubits
        pauli_str = parts[0].upper()
        if set(pauli_str) - set('IXYZ'):
            self.io.writeln("?PAULIPROP: observable must be over I, X, Y, Z")
            return
        rest_toks = parts[1:]
        threshold = 1e-10
        if rest_toks and re.match(r'^[0-9.eE+-]+$', rest_toks[-1]) and (
                '.' in rest_toks[-1] or 'e' in rest_toks[-1].lower()):
            threshold = float(rest_toks[-1]); rest_toks = rest_toks[:-1]
        qubits = [int(self._eval_with_vars(t, {})) for t in rest_toks] if rest_toks \
            else list(range(len(pauli_str)))
        try:
            # Build the qubit-indexed observable string (position i = qubit i).
            obs = ['I'] * n
            for p, q in zip(pauli_str, qubits):
                obs[q] = p
            terms = {''.join(obs): 1.0 + 0j}
            qc, _ = self.build_circuit()
            # Backpropagate through gates in reverse: O <- G^dag O G.
            max_terms = 0
            for inst in reversed(qc.data):
                op = inst.operation
                nm = op.name.lower()
                if nm in ('measure', 'barrier', 'save_statevector', 'reset', 'snapshot'):
                    continue
                try:
                    G = np.asarray(op.to_matrix(), dtype=complex)
                except Exception:
                    self.io.writeln(f"?PAULIPROP: gate '{nm}' has no matrix; unsupported")
                    return
                gq = [qc.find_bit(q).index for q in inst.qubits]
                Gd = G.conj().T

                def conj(P, _G=G, _Gd=Gd):
                    return _Gd @ P @ _G

                new_terms: dict = {}
                for ps, coeff in terms.items():
                    plist = list(ps)
                    for newp, c in self._conjugate_term(plist, coeff, gq, conj):
                        key = ''.join(newp)
                        new_terms[key] = new_terms.get(key, 0j) + c
                # Prune.
                terms = {k: v for k, v in new_terms.items() if abs(v) >= threshold}
                max_terms = max(max_terms, len(terms))
            # Evaluate on |0...0>: a term contributes its coeff iff every qubit is I or Z.
            val = 0.0
            for ps, coeff in terms.items():
                if set(ps) <= {'I', 'Z'}:
                    val += coeff.real
            self.io.writeln(f"  <{pauli_str}> on {qubits} = {val:.6f}  "
                            f"(Pauli propagation, peak {max_terms} terms, threshold {threshold:g})")
            self.variables['_PAULIPROP'] = float(val)
        except Exception as e:
            self.io.writeln(f"?PAULIPROP ERROR: {e}")
