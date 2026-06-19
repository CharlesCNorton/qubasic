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
        if name in ('SURFACE', 'SURF'):
            return self._surface_code(distance)
        raise ValueError(f"unknown code '{name}' (try REP, STEANE, SHOR, SURFACE; QEC LIST)")

    def _surface_code(self, d: int) -> dict:
        """Rotated surface code of odd distance d (d^2 data qubits on a d x d grid).

        Bulk faces are weight-4 stabilizers alternating X/Z by checkerboard parity;
        boundary faces are weight-2, X on the top/bottom edges and Z on the
        left/right edges. Logical X is an X string along the top row, logical Z a
        Z string down the left column. The construction is validated by the
        caller's checks (it must correct every weight-1 error)."""
        if d % 2 == 0:
            d += 1
        n = d * d

        def idx(r, c):
            return r * d + c

        stab = []
        for r in range(-1, d):
            for c in range(-1, d):
                corners = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
                qs = [idx(i, j) for (i, j) in corners if 0 <= i < d and 0 <= j < d]
                if len(qs) < 2:
                    continue
                typ = 'Z' if (r + c) % 2 == 0 else 'X'
                if len(qs) == 4:
                    stab.append((typ, qs))
                else:
                    on_tb = (r == -1 or r == d - 1)
                    on_lr = (c == -1 or c == d - 1)
                    if on_tb and not on_lr and typ == 'X':
                        stab.append((typ, qs))
                    elif on_lr and not on_tb and typ == 'Z':
                        stab.append((typ, qs))
        stab_strs = []
        for typ, qs in stab:
            s = ['I'] * n
            for q in qs:
                s[q] = typ
            stab_strs.append(''.join(s))
        # Logical X runs down a column (between the top/bottom X boundaries);
        # logical Z runs along a row (between the left/right Z boundaries). They
        # cross at qubit 0 and so anticommute.
        lx = ['I'] * n
        for r in range(d):
            lx[idx(r, 0)] = 'X'
        lz = ['I'] * n
        for c in range(d):
            lz[idx(0, c)] = 'Z'
        return {'n': n, 'stab': stab_strs, 'lx': ''.join(lx), 'lz': ''.join(lz),
                'd': d, 'alphabet': 'IXYZ', 'name': f'rotated surface d={d}'}

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

    def _logical_error_rate(self, code: dict, p: float, trials: int, rng, uf: bool = False) -> float:
        decoder = None if uf else self._qec_decoder(code)
        stab, lx, lz = code['stab'], code['lx'], code['lz']
        fails = 0
        for _ in range(trials):
            err = self._random_pauli_error(code, p, rng)
            synd = tuple(_anticommute(err, s) for s in stab)
            recovery = self._qec_matching_decode(code, synd) if uf \
                else decoder.get(synd, 'I' * code['n'])
            residual = _pmul(err, recovery)
            if _anticommute(residual, lx) or _anticommute(residual, lz):
                fails += 1
        return fails / trials

    def _qec_matching_decode(self, code: dict, syndrome) -> str:
        """Union-find / matching decoder: minimum-weight matching of syndrome
        defects on the matching graph, correcting the connecting qubits.

        Scalable alternative to the exponential lookup table (it never enumerates
        all errors). Defects are matched to each other or to the boundary by exact
        minimum-weight matching over shortest paths, and the path qubits flipped.
        """
        import functools
        from collections import deque
        n = code['n']
        stabs = code['stab']
        xr = ['I'] * n
        zr = ['I'] * n
        for etype, dtype, rec in (('X', 'Z', xr), ('Z', 'X', zr)):
            det = [i for i, s in enumerate(stabs) if set(s) - {'I'} == {dtype}]
            defects = tuple(i for i in det if syndrome[i])
            if not defects:
                continue
            adj = {i: [] for i in det}
            adj['B'] = []
            for q in range(n):
                touch = [i for i in det if stabs[i][q] == dtype]
                if len(touch) == 2:
                    adj[touch[0]].append((touch[1], q)); adj[touch[1]].append((touch[0], q))
                elif len(touch) == 1:
                    adj[touch[0]].append(('B', q)); adj['B'].append((touch[0], q))

            def bfs(src):
                prev = {src: (None, None)}
                dq = deque([src])
                while dq:
                    u = dq.popleft()
                    for v, q in adj.get(u, []):
                        if v not in prev:
                            prev[v] = (u, q); dq.append(v)
                return prev

            prevs = {s: bfs(s) for s in defects}

            def path_qubits(src, tgt):
                prev = prevs[src]
                qs, node = [], tgt
                while node != src and prev.get(node, (None, None))[0] is not None:
                    par, q = prev[node]
                    qs.append(q); node = par
                return qs

            def dist(src, tgt):
                return len(path_qubits(src, tgt)) if tgt in prevs[src] else 10 ** 9

            @functools.lru_cache(maxsize=None)
            def solve(rem):
                if not rem:
                    return (0, ())
                i, rest = rem[0], rem[1:]
                best_c, best_m = dist(i, 'B') + solve(rest)[0], ((i, 'B'),) + solve(rest)[1]
                for k, j in enumerate(rest):
                    sub = solve(rest[:k] + rest[k + 1:])
                    c = dist(i, j) + sub[0]
                    if c < best_c:
                        best_c, best_m = c, ((i, j),) + sub[1]
                return (best_c, best_m)

            _, matches = solve(defects)
            for a, b in matches:
                for q in path_qubits(a, b):
                    rec[q] = etype
        out = []
        for q in range(n):
            x, z = xr[q] == 'X', zr[q] == 'Z'
            out.append('Y' if (x and z) else 'X' if x else 'Z' if z else 'I')
        return ''.join(out)

    def cmd_distill(self, rest: str) -> None:
        """DISTILL <p> [trials] — 15-to-1 magic-state (T-state) distillation.

        Models the protocol's error detection by the [15,11,3] Hamming code:
        15 noisy T-states (Z error rate p), accept on a trivial syndrome, and a
        logical output error occurs only for an undetected nonzero codeword. The
        output error rate scales as ~35 p^3, well below the input p."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: DISTILL <p> [trials]")
            return
        try:
            p = float(self._eval_with_vars(parts[0], {}))
            trials = int(parts[1]) if len(parts) > 1 else 200000
            # [15,11,3] Hamming parity-check matrix: columns are 1..15 in binary.
            H = np.array([[ (c >> b) & 1 for c in range(1, 16)] for b in range(4)])
            rng = np.random.default_rng(self._seed)
            accepted = 0
            errors = 0
            for _ in range(trials):
                e = (rng.random(15) < p).astype(int)
                synd = H.dot(e) % 2
                if not synd.any():            # accept: trivial syndrome
                    accepted += 1
                    if e.any():               # nonzero codeword -> logical error
                        errors += 1
            acc = accepted / trials
            p_out = errors / accepted if accepted else 0.0
            self.io.writeln(f"\n  15-to-1 magic-state distillation:")
            self.io.writeln(f"    input error p      = {p:g}")
            self.io.writeln(f"    acceptance rate    = {acc:.4f}")
            self.io.writeln(f"    output error p_out = {p_out:.3e}   (~35 p^3 = {35 * p ** 3:.3e})")
            self.io.writeln(f"    suppression p/p_out= {p / p_out:.1f}x" if p_out > 0 else
                            "    output error: none observed")
            self.variables['_DISTILL_POUT'] = p_out
        except Exception as e:
            self.io.writeln(f"?DISTILL ERROR: {e}")

    def cmd_lattice(self, rest: str) -> None:
        """LATTICE <stateA> <stateB> — lattice-surgery logical ZZ measurement.

        Encodes two distance-3 repetition logical qubits in the given logical
        states (0, 1, +, -), then performs a merge that measures the joint
        operator Zbar_A Zbar_B (the basis of a lattice-surgery CNOT) via an
        ancilla, and reports the logical parity outcome."""
        parts = rest.split()
        if len(parts) < 2:
            self.io.writeln("?USAGE: LATTICE <stateA> <stateB>   (states: 0, 1, +, -)")
            return
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        try:
            sa, sb = parts[0], parts[1]
            # 3 data qubits per patch (A: 0-2, B: 3-5) + 1 merge ancilla (6).
            qc = QuantumCircuit(7, 1)

            def encode(base, state):
                if state in ('1',):
                    qc.x(base)
                elif state in ('+', '-'):
                    qc.h(base)
                    if state == '-':
                        qc.z(base)
                qc.cx(base, base + 1); qc.cx(base, base + 2)   # repetition encode
            encode(0, sa); encode(3, sb)
            # Merge: measure Zbar_A Zbar_B = Z_{A0} Z_{B0} via an ancilla.
            anc = 6
            qc.h(anc)
            qc.cz(anc, 0)
            qc.cz(anc, 3)
            qc.h(anc)
            qc.measure(anc, 0)
            backend = AerSimulator(noise_model=self._noise_model) if self._noise_model else AerSimulator()
            counts = backend.run(transpile(qc, backend), shots=max(500, self.shots)).result().get_counts()
            par = '+1 (even)' if counts.get('0', 0) >= counts.get('1', 0) else '-1 (odd)'
            self.io.writeln(f"\n  Lattice surgery: merge of |{sa}>_L (A) and |{sb}>_L (B)")
            self.io.writeln(f"    joint Zbar_A Zbar_B measurement -> {par}   {dict(counts)}")
            self.io.writeln(f"    (a deterministic joint parity for computational inputs; the "
                            f"building block of a lattice-surgery CNOT)")
        except Exception as e:
            self.io.writeln(f"?LATTICE ERROR: {e}")

    # ── Commands ────────────────────────────────────────────────────────

    def cmd_qec(self, rest: str) -> None:
        """QEC LIST | QEC <code> [distance] — show a stabilizer code.

        Lists the built-in codes, or prints a code's stabilizer generators and
        logical operators (and verifies they form a valid code)."""
        arg = rest.strip().upper()
        if not arg or arg == 'LIST':
            self.io.writeln("\n  Built-in QEC codes:")
            self.io.writeln("    REP [d]     repetition / bit-flip code, odd distance d (default 3)")
            self.io.writeln("    STEANE      [[7,1,3]] CSS code")
            self.io.writeln("    SHOR        [[9,1,3]] code")
            self.io.writeln("    SURFACE [d] rotated surface code, odd distance d (default 3)")
            self.io.writeln("  Decoders: optimal lookup (default), union-find (LOGICAL_ERROR_RATE ... UF)")
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
            uf = any(t.upper() in ('UF', 'UNION-FIND', 'MATCHING') for t in parts)
            tail = [t for t in parts[idx:] if t.upper() not in ('UF', 'UNION-FIND', 'MATCHING')]
            p = float(self._eval_with_vars(tail[0], {}))
            trials = int(tail[1]) if len(tail) > 1 else 20000
            code = self._qec_code(name, distance)
            rng = np.random.default_rng(self._seed)
            ler = self._logical_error_rate(code, p, trials, rng, uf=uf)
            self.io.writeln(f"\n  {code['name']}: physical p={p}, {trials} trials"
                            f"{' (union-find decoder)' if uf else ' (lookup decoder)'}")
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
