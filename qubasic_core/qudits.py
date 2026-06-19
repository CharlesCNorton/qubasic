"""QUBASIC qudit engine — d-level systems beyond qubits.

A standalone numpy statevector simulator for n qudits of dimension d, with the
generalized Pauli (shift/clock) gates, the qudit Fourier transform, and the
controlled-sum entangler. Independent of the qubit (Aer) engine.

  QUDIT <d> [n]     Initialize n qudits of dimension d in |0...0>
  QX <q> [k]        Shift gate X^k: |j> -> |j+k mod d>
  QZ <q> [k]        Clock gate Z^k: |j> -> w^(kj)|j>, w = exp(2 pi i / d)
  QF <q>            Qudit Fourier transform (generalized Hadamard)
  QSUM <c> <t>      Controlled sum: |a,b> -> |a, (b+a) mod d>
  QSTATE            Show the qudit statevector
  QMEASURE [shots]  Sample measurement outcomes (base-d strings)
"""

from __future__ import annotations

import numpy as np


class QuditMixin:
    """A d-level (qudit) statevector engine for QBasicTerminal.

    Requires: TerminalProtocol — uses self.shots, self._seed, self.io.
    """

    def _init_qudits(self) -> None:
        self._qd = 0
        self._qn = 0
        self._qsv = None

    def _qudit_apply(self, mat, targets):
        """Apply a single- or two-qudit matrix to the qudit statevector."""
        d, n = self._qd, self._qn
        k = len(targets)
        t = self._qsv.reshape([d] * n)
        axes = [q for q in targets]
        t = np.tensordot(mat.reshape([d] * (2 * k)), t,
                         axes=(list(range(k, 2 * k)), axes))
        t = np.moveaxis(t, list(range(k)), axes)
        self._qsv = np.ascontiguousarray(t).reshape(-1)

    def cmd_qudit(self, rest: str) -> None:
        """QUDIT <d> [n] — initialize n qudits of dimension d in |0...0>."""
        parts = rest.split()
        if not parts:
            if self._qd:
                self.io.writeln(f"QUDIT: {self._qn} qudits of dimension {self._qd}")
            else:
                self.io.writeln("?USAGE: QUDIT <d> [n]")
            return
        d = int(parts[0])
        n = int(parts[1]) if len(parts) > 1 else 1
        if d < 2:
            self.io.writeln("?QUDIT dimension must be >= 2")
            return
        if d ** n > 1 << 20:
            self.io.writeln(f"?QUDIT state d^n = {d ** n} too large")
            return
        self._qd, self._qn = d, n
        self._qsv = np.zeros(d ** n, dtype=complex)
        self._qsv[0] = 1.0
        self.io.writeln(f"QUDIT: {n} qudit(s) of dimension {d} in |{'0' * n}>")

    def _qcheck(self, q):
        if not self._qd:
            raise ValueError("no qudit register (use QUDIT <d> [n] first)")
        if q < 0 or q >= self._qn:
            raise ValueError(f"qudit {q} out of range (0-{self._qn - 1})")

    def cmd_qx(self, rest: str) -> None:
        """QX <qudit> [k] — generalized shift X^k."""
        parts = rest.split()
        try:
            q = int(parts[0]); k = int(parts[1]) if len(parts) > 1 else 1
            self._qcheck(q)
            d = self._qd
            X = np.zeros((d, d), dtype=complex)
            for j in range(d):
                X[(j + k) % d, j] = 1.0
            self._qudit_apply(X, [q])
        except Exception as e:
            self.io.writeln(f"?QX ERROR: {e}")

    def cmd_qz(self, rest: str) -> None:
        """QZ <qudit> [k] — generalized clock Z^k."""
        parts = rest.split()
        try:
            q = int(parts[0]); k = int(parts[1]) if len(parts) > 1 else 1
            self._qcheck(q)
            d = self._qd
            w = np.exp(2j * np.pi / d)
            Z = np.diag([w ** ((k * j) % d) for j in range(d)]).astype(complex)
            self._qudit_apply(Z, [q])
        except Exception as e:
            self.io.writeln(f"?QZ ERROR: {e}")

    def cmd_qf(self, rest: str) -> None:
        """QF <qudit> — qudit Fourier transform (generalized Hadamard)."""
        try:
            q = int(rest.split()[0])
            self._qcheck(q)
            d = self._qd
            w = np.exp(2j * np.pi / d)
            F = np.array([[w ** ((i * j) % d) for j in range(d)] for i in range(d)],
                         dtype=complex) / np.sqrt(d)
            self._qudit_apply(F, [q])
        except Exception as e:
            self.io.writeln(f"?QF ERROR: {e}")

    def cmd_qsum(self, rest: str) -> None:
        """QSUM <control> <target> — |a,b> -> |a, (b+a) mod d> (qudit CNOT)."""
        parts = rest.split()
        try:
            c, t = int(parts[0]), int(parts[1])
            self._qcheck(c); self._qcheck(t)
            d = self._qd
            SUM = np.zeros((d * d, d * d), dtype=complex)
            for a in range(d):
                for b in range(d):
                    SUM[a * d + ((b + a) % d), a * d + b] = 1.0
            self._qudit_apply(SUM, [c, t])
        except Exception as e:
            self.io.writeln(f"?QSUM ERROR: {e}")

    def cmd_qstate(self, rest: str = '') -> None:
        """QSTATE — show non-zero amplitudes of the qudit statevector."""
        if not self._qd:
            self.io.writeln("?NO QUDIT STATE — use QUDIT <d> [n]")
            return
        d, n = self._qd, self._qn
        self.io.writeln(f"\n  Qudit state (d={d}, {n} qudit(s)):")
        for i, amp in enumerate(self._qsv):
            if abs(amp) > 1e-9:
                digits = np.base_repr(i, d).rjust(n, '0')
                self.io.writeln(f"    |{digits}>  {amp.real:+.4f}{amp.imag:+.4f}j  "
                                f"(P={abs(amp) ** 2:.4f})")

    def cmd_qmeasure(self, rest: str = '') -> None:
        """QMEASURE [shots] — sample qudit measurement outcomes."""
        if not self._qd:
            self.io.writeln("?NO QUDIT STATE — use QUDIT <d> [n]")
            return
        d, n = self._qd, self._qn
        shots = int(rest.strip()) if rest.strip() else self.shots
        probs = np.abs(self._qsv) ** 2
        probs = probs / probs.sum()
        rng = np.random.default_rng(self._seed)
        draws = rng.choice(len(probs), size=shots, p=probs)
        unique, counts = np.unique(draws, return_counts=True)
        self.io.writeln(f"\n  QMEASURE ({shots} shots, base-{d}):")
        for idx, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
            digits = np.base_repr(int(idx), d).rjust(n, '0')
            self.io.writeln(f"    |{digits}>  {c}  ({100 * c / shots:.1f}%)")
