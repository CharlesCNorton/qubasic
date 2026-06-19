"""QUBASIC quantum error correction.

Stabilizer codes with an optimal (minimum-weight lookup) decoder, logical error
rates, and threshold sweeps, computed by classical Pauli-frame simulation (no
statevector needed, so it is exact and fast). Built-in codes: the repetition
(bit-flip) code at any odd distance, the [[7,1,3]] Steane code, and the [[9,1,3]]
Shor code.

  QEC LIST                       List the built-in codes
  QEC STEANE                     Show a code's stabilizers and logical operators
  LOGICAL_ERROR_RATE STEANE 0.05 Monte-Carlo logical error rate at physical rate p
  THRESHOLD REP 0.0 0.5 11       Sweep p and compare code distances (find the crossing)
"""

from __future__ import annotations

import re
import itertools

import numpy as np

# Phase-free single-qubit Pauli multiplication (we only need the group structure).
_PMUL = {
    ('I', 'I'): 'I', ('I', 'X'): 'X', ('I', 'Y'): 'Y', ('I', 'Z'): 'Z',
    ('X', 'I'): 'X', ('X', 'X'): 'I', ('X', 'Y'): 'Z', ('X', 'Z'): 'Y',
    ('Y', 'I'): 'Y', ('Y', 'X'): 'Z', ('Y', 'Y'): 'I', ('Y', 'Z'): 'X',
    ('Z', 'I'): 'Z', ('Z', 'X'): 'Y', ('Z', 'Y'): 'X', ('Z', 'Z'): 'I',
}


def _anticommute(a: str, b: str) -> int:
    """1 if Pauli strings a and b anticommute, else 0."""
    cnt = 0
    for pa, pb in zip(a, b):
        if pa != 'I' and pb != 'I' and pa != pb:
            cnt ^= 1
    return cnt


def _pmul(a: str, b: str) -> str:
    return ''.join(_PMUL[(pa, pb)] for pa, pb in zip(a, b))


def _weight(p: str) -> int:
    return sum(1 for c in p if c != 'I')


class QECMixin:
    """Stabilizer codes, decoding, and logical error rates for QBasicTerminal.

    Requires: TerminalProtocol — uses self._eval_with_vars(), self._seed, self.io.
    """

    def _qec_code(self, name: str, distance: int = 3) -> dict:
        """Return a code descriptor: data qubit count, stabilizers, logical X/Z,
        distance, and the error alphabet the decoder corrects."""
        name = name.upper()
        if name in ('REP', 'REPETITION', 'BITFLIP'):
            d = distance if distance % 2 == 1 else distance + 1
            stab = []
            for i in range(d - 1):
                s = ['I'] * d
                s[i] = 'Z'; s[i + 1] = 'Z'
                stab.append(''.join(s))
            return {'n': d, 'stab': stab, 'lx': 'X' * d, 'lz': 'Z' + 'I' * (d - 1),
                    'd': d, 'alphabet': 'IX', 'name': f'repetition d={d}'}
        if name == 'STEANE':
            Hm = [[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]]
            stab = [''.join('X' if b else 'I' for b in row) for row in Hm]
            stab += [''.join('Z' if b else 'I' for b in row) for row in Hm]
            return {'n': 7, 'stab': stab, 'lx': 'X' * 7, 'lz': 'Z' * 7,
                    'd': 3, 'alphabet': 'IXYZ', 'name': 'Steane [[7,1,3]]'}
        if name == 'SHOR':
            stab = []
            for b in range(3):
                base = b * 3
                for a, c in ((base, base + 1), (base + 1, base + 2)):
                    s = ['I'] * 9; s[a] = 'Z'; s[c] = 'Z'; stab.append(''.join(s))
            stab.append('XXXXXX' + 'III')
            stab.append('III' + 'XXXXXX')
            return {'n': 9, 'stab': stab, 'lx': 'X' * 9, 'lz': 'ZIIZIIZII',
                    'd': 3, 'alphabet': 'IXYZ', 'name': 'Shor [[9,1,3]]'}
        raise ValueError(f"unknown code '{name}' (try REP, STEANE, SHOR; QEC LIST)")

    def _qec_decoder(self, code: dict) -> dict:
        """Build the minimum-weight lookup decoder: syndrome -> recovery Pauli.

        Enumerates every error over the code's alphabet, keeping the lowest-weight
        representative per syndrome. Optimal for these small codes."""
        n = code['n']
        stab = code['stab']
        alphabet = code['alphabet']
        table: dict = {}
        for combo in itertools.product(alphabet, repeat=n):
            err = ''.join(combo)
            synd = tuple(_anticommute(err, s) for s in stab)
            w = _weight(err)
            if synd not in table or w < table[synd][1]:
                table[synd] = (err, w)
        return {synd: err for synd, (err, _w) in table.items()}

    def _random_pauli_error(self, code: dict, p: float, rng) -> str:
        """Sample an i.i.d. error per qubit: X for a bit-flip code, depolarizing else."""
        n = code['n']
        out = []
        if code['alphabet'] == 'IX':
            for _ in range(n):
                out.append('X' if rng.random() < p else 'I')
        else:
            for _ in range(n):
                r = rng.random()
                if r < p:
                    out.append('XYZ'[rng.integers(3)])
                else:
                    out.append('I')
        return ''.join(out)

    def _logical_error_rate(self, code: dict, p: float, trials: int, rng) -> float:
        decoder = self._qec_decoder(code)
        stab, lx, lz = code['stab'], code['lx'], code['lz']
        fails = 0
        for _ in range(trials):
            err = self._random_pauli_error(code, p, rng)
            synd = tuple(_anticommute(err, s) for s in stab)
            recovery = decoder.get(synd, 'I' * code['n'])
            residual = _pmul(err, recovery)
            if _anticommute(residual, lx) or _anticommute(residual, lz):
                fails += 1
        return fails / trials

    # ── Commands ────────────────────────────────────────────────────────

    def cmd_qec(self, rest: str) -> None:
        """QEC LIST | QEC <code> [distance] — show a stabilizer code.

        Lists the built-in codes, or prints a code's stabilizer generators and
        logical operators (and verifies they form a valid code)."""
        arg = rest.strip().upper()
        if not arg or arg == 'LIST':
            self.io.writeln("\n  Built-in QEC codes:")
            self.io.writeln("    REP [d]    repetition / bit-flip code, odd distance d (default 3)")
            self.io.writeln("    STEANE     [[7,1,3]] CSS code")
            self.io.writeln("    SHOR       [[9,1,3]] code")
            return
        parts = arg.split()
        try:
            d = int(parts[1]) if len(parts) > 1 else 3
            code = self._qec_code(parts[0], d)
        except Exception as e:
            self.io.writeln(f"?QEC ERROR: {e}")
            return
        self.io.writeln(f"\n  {code['name']}: {code['n']} data qubits, distance {code['d']}")
        self.io.writeln(f"  Stabilizers ({len(code['stab'])}):")
        for s in code['stab']:
            self.io.writeln(f"    {s}")
        self.io.writeln(f"  Logical X: {code['lx']}")
        self.io.writeln(f"  Logical Z: {code['lz']}")
        # Validity self-check: logicals commute with every stabilizer and
        # anticommute with each other.
        ok = bool(all(not _anticommute(code['lx'], s) for s in code['stab'])
                  and all(not _anticommute(code['lz'], s) for s in code['stab'])
                  and _anticommute(code['lx'], code['lz']))
        self.io.writeln(f"  Valid code (logicals normalize the stabilizer group): {ok}")

    def cmd_logical_error_rate(self, rest: str) -> None:
        """LOGICAL_ERROR_RATE <code> [distance] <p> [trials] — Monte-Carlo logical error rate.

        Injects i.i.d. physical errors at rate p, extracts the syndrome, decodes
        with the optimal lookup decoder, applies the recovery, and reports the
        fraction of trials left with a logical error."""
        parts = rest.split()
        if len(parts) < 2:
            self.io.writeln("?USAGE: LOGICAL_ERROR_RATE <code> [distance] <p> [trials]")
            return
        try:
            name = parts[0]
            idx = 1
            distance = 3
            if len(parts) > 2 and parts[1].isdigit() and float(parts[1]) >= 1:
                distance = int(parts[1]); idx = 2
            p = float(self._eval_with_vars(parts[idx], {}))
            trials = int(parts[idx + 1]) if len(parts) > idx + 1 else 20000
            code = self._qec_code(name, distance)
            rng = np.random.default_rng(self._seed)
            ler = self._logical_error_rate(code, p, trials, rng)
            self.io.writeln(f"\n  {code['name']}: physical p={p}, {trials} trials")
            self.io.writeln(f"  Logical error rate = {ler:.6f}")
            self.variables['_LER'] = ler
        except Exception as e:
            self.io.writeln(f"?LOGICAL_ERROR_RATE ERROR: {e}")

    def cmd_threshold(self, rest: str) -> None:
        """THRESHOLD <code> <p1> <p2> <steps> [d1 d2 ...] — sweep physical error rate.

        Reports the logical error rate across [p1, p2] for several code distances
        (default 3, 5, 7 for the repetition code), so the threshold crossing where
        larger codes start helping is visible."""
        parts = rest.split()
        if len(parts) < 4:
            self.io.writeln("?USAGE: THRESHOLD <code> <p1> <p2> <steps> [distances]")
            return
        try:
            name = parts[0]
            p1 = float(self._eval_with_vars(parts[1], {}))
            p2 = float(self._eval_with_vars(parts[2], {}))
            steps = int(parts[3])
            distances = [int(x) for x in parts[4:]] if len(parts) > 4 else [3, 5, 7]
            ps = [p1 + (p2 - p1) * i / max(1, steps - 1) for i in range(steps)]
            codes = [self._qec_code(name, d) for d in distances]
            rng = np.random.default_rng(self._seed)
            trials = 8000
            self.io.writeln(f"\n  Threshold sweep ({name.upper()}), {trials} trials/point:")
            self.io.writeln("    p      " + ''.join(f"  d={c['d']:<8}" for c in codes))
            for p in ps:
                row = f"  {p:5.3f}  "
                for c in codes:
                    row += f"  {self._logical_error_rate(c, p, trials, rng):<8.4f}"
                self.io.writeln(row)
        except Exception as e:
            self.io.writeln(f"?THRESHOLD ERROR: {e}")
