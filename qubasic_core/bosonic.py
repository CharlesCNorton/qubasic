"""QUBASIC bosonic / continuous-variable engine.

A truncated Fock-space numpy simulator for n modes, with displacement,
squeezing, beamsplitters, and Schrodinger-cat preparation. Unitaries are built
from their (anti-Hermitian) generators via eigendecomposition, so no external
CV library is needed. Truncation at the Fock cutoff makes large displacements
approximate.

  BOSONIC <modes> <cutoff>   Initialize modes in vacuum, truncated at `cutoff` photons
  DISPLACE <mode> <re> [im]  Displacement D(alpha) of a mode
  SQUEEZE <mode> <r>         Single-mode squeezing S(r)
  CAT <mode> <re> [im]       Prepare an even Schrodinger-cat state
  BS <m1> <m2> <theta>       Beamsplitter between two modes
  BSTATE [mode]              Show Fock amplitudes, <n>, and photon distribution
"""

from __future__ import annotations

import numpy as np


def _expm_anti(G: np.ndarray) -> np.ndarray:
    """exp(G) for an anti-Hermitian G, via eigendecomposition of the Hermitian iG."""
    H = 1j * G
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w)) @ V.conj().T


class BosonicMixin:
    """Continuous-variable Fock-space engine for QBasicTerminal.

    Requires: TerminalProtocol — uses self._eval_with_vars(), self.io.
    """

    def _init_bosonic(self) -> None:
        self._bmodes = 0
        self._bcut = 0
        self._bsv = None

    def _annihilation(self):
        N = self._bcut
        a = np.zeros((N, N), dtype=complex)
        for n in range(1, N):
            a[n - 1, n] = np.sqrt(n)
        return a

    def _bos_apply(self, mat, modes):
        N, m = self._bcut, self._bmodes
        k = len(modes)
        t = self._bsv.reshape([N] * m)
        t = np.tensordot(mat.reshape([N] * (2 * k)), t,
                         axes=(list(range(k, 2 * k)), list(modes)))
        t = np.moveaxis(t, list(range(k)), list(modes))
        self._bsv = np.ascontiguousarray(t).reshape(-1)

    def cmd_bosonic(self, rest: str) -> None:
        """BOSONIC <modes> <cutoff> — initialize modes in vacuum."""
        parts = rest.split()
        if len(parts) < 2:
            if self._bmodes:
                self.io.writeln(f"BOSONIC: {self._bmodes} mode(s), cutoff {self._bcut}")
            else:
                self.io.writeln("?USAGE: BOSONIC <modes> <cutoff>")
            return
        m = int(parts[0]); cut = int(parts[1])
        if cut ** m > 1 << 20:
            self.io.writeln(f"?BOSONIC cutoff^modes = {cut ** m} too large")
            return
        self._bmodes, self._bcut = m, cut
        self._bsv = np.zeros(cut ** m, dtype=complex)
        self._bsv[0] = 1.0
        self.io.writeln(f"BOSONIC: {m} mode(s), Fock cutoff {cut}, in vacuum")

    def _bcheck(self, mode):
        if not self._bmodes:
            raise ValueError("no bosonic register (use BOSONIC <modes> <cutoff>)")
        if mode < 0 or mode >= self._bmodes:
            raise ValueError(f"mode {mode} out of range (0-{self._bmodes - 1})")

    def cmd_displace(self, rest: str) -> None:
        """DISPLACE <mode> <re> [im] — displacement D(alpha) = exp(alpha a^dag - alpha* a)."""
        parts = rest.split()
        try:
            mode = int(parts[0]); self._bcheck(mode)
            alpha = complex(float(self._eval_with_vars(parts[1], {})),
                            float(self._eval_with_vars(parts[2], {})) if len(parts) > 2 else 0.0)
            a = self._annihilation()
            self._bos_apply(_expm_anti(alpha * a.conj().T - np.conj(alpha) * a), [mode])
            self._report_n(mode)
        except Exception as e:
            self.io.writeln(f"?DISPLACE ERROR: {e}")

    def cmd_squeeze(self, rest: str) -> None:
        """SQUEEZE <mode> <r> — single-mode squeezing S(r) = exp(r/2 (a^2 - a^dag^2))."""
        parts = rest.split()
        try:
            mode = int(parts[0]); self._bcheck(mode)
            r = float(self._eval_with_vars(parts[1], {}))
            a = self._annihilation()
            self._bos_apply(_expm_anti(0.5 * r * (a @ a - a.conj().T @ a.conj().T)), [mode])
            self._report_n(mode)
        except Exception as e:
            self.io.writeln(f"?SQUEEZE ERROR: {e}")

    def cmd_bs(self, rest: str) -> None:
        """BS <m1> <m2> <theta> — beamsplitter exp(theta (a^dag b - a b^dag))."""
        parts = rest.split()
        try:
            m1, m2 = int(parts[0]), int(parts[1])
            self._bcheck(m1); self._bcheck(m2)
            theta = float(self._eval_with_vars(parts[2], {}))
            N = self._bcut
            a = self._annihilation()
            I = np.eye(N, dtype=complex)
            aa = np.kron(a, I); bb = np.kron(I, a)
            G = theta * (aa.conj().T @ bb - aa @ bb.conj().T)
            self._bos_apply(_expm_anti(G), [m1, m2])
            self.io.writeln(f"  beamsplitter theta={theta} on modes {m1},{m2}")
        except Exception as e:
            self.io.writeln(f"?BS ERROR: {e}")

    def cmd_cat(self, rest: str) -> None:
        """CAT <mode> <re> [im] — prepare an even cat state (|alpha> + |-alpha>)/N."""
        parts = rest.split()
        try:
            mode = int(parts[0]); self._bcheck(mode)
            alpha = complex(float(self._eval_with_vars(parts[1], {})),
                            float(self._eval_with_vars(parts[2], {})) if len(parts) > 2 else 0.0)
            N = self._bcut
            a = self._annihilation()
            vac = np.zeros(N, dtype=complex); vac[0] = 1.0
            Dp = _expm_anti(alpha * a.conj().T - np.conj(alpha) * a)
            Dm = _expm_anti(-alpha * a.conj().T + np.conj(alpha) * a)
            cat = Dp @ vac + Dm @ vac
            cat = cat / np.linalg.norm(cat)
            # Replace the mode's state (assumes a fresh/vacuum register for clarity).
            if self._bmodes == 1:
                self._bsv = cat
            else:
                # Project the mode onto the cat by applying it as a state-prep matrix
                # from vacuum on that mode (only valid if the mode is in vacuum).
                prep = np.outer(cat, vac.conj())
                prep += np.eye(N) - np.outer(vac, vac.conj())  # identity off the vacuum
                self._bos_apply(prep, [mode])
            self.io.writeln(f"  cat state |alpha>+|-alpha>, alpha={alpha} on mode {mode}")
            self._report_n(mode)
        except Exception as e:
            self.io.writeln(f"?CAT ERROR: {e}")

    def _mode_rdm_diag(self, mode):
        """Photon-number distribution (diagonal of the reduced density matrix) of a mode."""
        N, m = self._bcut, self._bmodes
        t = self._bsv.reshape([N] * m)
        t = np.moveaxis(t, mode, 0).reshape(N, -1)
        return np.real(np.sum(np.abs(t) ** 2, axis=1))

    def _report_n(self, mode):
        diag = self._mode_rdm_diag(mode)
        nbar = float(np.sum(np.arange(self._bcut) * diag))
        self.io.writeln(f"  mode {mode}: <n> = {nbar:.4f}")

    def cmd_bstate(self, rest: str = '') -> None:
        """BSTATE [mode] — show <n> and the photon-number distribution."""
        if not self._bmodes:
            self.io.writeln("?NO BOSONIC STATE — use BOSONIC <modes> <cutoff>")
            return
        modes = [int(rest)] if rest.strip() else list(range(self._bmodes))
        for mode in modes:
            diag = self._mode_rdm_diag(mode)
            nbar = float(np.sum(np.arange(self._bcut) * diag))
            self.io.writeln(f"\n  Mode {mode}: <n> = {nbar:.4f}")
            for k in range(self._bcut):
                if diag[k] > 1e-4:
                    bar = '#' * int(40 * diag[k])
                    self.io.writeln(f"    |{k}>  {diag[k]:6.4f}  {bar}")
