"""QUBASIC error correction: GF(2) algebra, rotated surface-code patches,
the stabilizer codes, decoders, detector error models, circuit-level noise
experiments, and LOGICAL_ERROR_RATE and THRESHOLD.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qubasic_core.host import ApplicationsHost

# ---------------------------------------------------------------------------
# Linear algebra over GF(2) and CSS codes from their check matrices.
# ---------------------------------------------------------------------------

def _gf2_rref(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row-echelon form over GF(2). Returns (R, pivot_columns)."""
    A = (M.copy() % 2).astype(np.uint8)
    r = 0
    pivots: list[int] = []
    for c in range(A.shape[1]):
        if r >= A.shape[0]:
            break
        hits = np.nonzero(A[r:, c])[0]
        if hits.size == 0:
            continue
        piv = r + hits[0]
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        mask = A[:, c].astype(bool)
        mask[r] = False
        A[mask] ^= A[r]
        pivots.append(c)
        r += 1
    return A[:r], pivots


def _gf2_rank(M: np.ndarray) -> int:
    if M.size == 0:
        return 0
    return _gf2_rref(M)[0].shape[0]


def _gf2_nullspace(M: np.ndarray) -> np.ndarray:
    """Basis of the right kernel of M over GF(2), one vector per row."""
    R, pivots = _gf2_rref(M)
    n = M.shape[1]
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(n, np.uint8)
        v[f] = 1
        for i, c in enumerate(pivots):
            if i < R.shape[0] and R[i, f]:
                v[c] = 1
        basis.append(v)
    return np.array(basis, np.uint8) if basis else np.zeros((0, n), np.uint8)


def _css_logicals(Hx: np.ndarray, Hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logical operator bases of a CSS code.

    X logicals span ker(Hz) modulo rowspace(Hx); Z logicals span ker(Hx)
    modulo rowspace(Hz). Greedy coset extraction: keep a kernel vector iff it
    grows the rank of the accumulated (stabilizer + kept logicals) matrix.
    """
    def coset_basis(kernel_of: np.ndarray, mod_rows: np.ndarray) -> np.ndarray:
        ker = _gf2_nullspace(kernel_of)
        acc = (mod_rows % 2).astype(np.uint8)
        base_rank = _gf2_rank(acc)
        out = []
        for v in ker:
            trial = np.vstack([acc, v[None, :]])
            rk = _gf2_rank(trial)
            if rk > base_rank:
                out.append(v)
                acc, base_rank = trial, rk
        return np.array(out, np.uint8) if out else np.zeros((0, kernel_of.shape[1]), np.uint8)

    return coset_basis(Hz, Hx), coset_basis(Hx, Hz)


def _css_from_strings(code: dict) -> dict:
    """Matrix form (Hx, Hz, Lx, Lz) of a CSS string-stabilizer code, the shape
    the BP+OSD decoder takes."""
    n = code['n']
    hx, hz = [], []
    for s in code['stab']:
        kinds = set(s) - {'I'}
        row = np.array([ch != 'I' for ch in s], dtype=np.uint8)
        if kinds == {'X'}:
            hx.append(row)
        elif kinds == {'Z'}:
            hz.append(row)
        else:
            raise ValueError("BP+OSD needs a CSS code (pure X or pure Z stabilizers)")
    Hx = np.array(hx, dtype=np.uint8) if hx else np.zeros((0, n), np.uint8)
    Hz = np.array(hz, dtype=np.uint8) if hz else np.zeros((0, n), np.uint8)
    Lx = np.array([[ch in 'XY' for ch in code['lx']]], dtype=np.uint8)
    Lz = np.array([[ch in 'ZY' for ch in code['lz']]], dtype=np.uint8)
    return {'n': n, 'Hx': Hx, 'Hz': Hz, 'Lx': Lx, 'Lz': Lz, 'name': code['name'],
            'alphabet': code.get('alphabet', 'IXYZ')}


def _bb_code(l: int = 6, m: int = 6) -> dict:
    """Bivariate-bicycle CSS code on a 2*l*m qubit torus.

    A = x^3 + y + y^2 and B = y^3 + x + x^2 with x, y the cyclic shifts of
    Z_l and Z_m; Hx = [A|B], Hz = [B^T|A^T]. l=m=6 gives [[72,12,6]], l=12
    m=6 the [[144,12,12]] "gross" code. k is computed from GF(2) ranks, not
    assumed.
    """
    Sl = np.roll(np.eye(l, dtype=np.uint8), 1, axis=1)
    Sm = np.roll(np.eye(m, dtype=np.uint8), 1, axis=1)
    x = np.kron(Sl, np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), Sm)
    mp = lambda M, k: np.linalg.matrix_power(M.astype(np.uint8), k) % 2
    A = (mp(x, 3) + y + mp(y, 2)) % 2
    B = (mp(y, 3) + x + mp(x, 2)) % 2
    Hx = np.concatenate([A, B], axis=1).astype(np.uint8)
    Hz = np.concatenate([B.T, A.T], axis=1).astype(np.uint8)
    n = 2 * l * m
    k = n - _gf2_rank(Hx) - _gf2_rank(Hz)
    Lx, Lz = _css_logicals(Hx, Hz)
    known_d = {(6, 6): 6, (12, 6): 12, (9, 6): 8, (12, 12): 12, (15, 3): 8}
    d = known_d.get((l, m))
    basis_bound = int(min(int(v.sum()) for v in np.vstack([Lx, Lz]))) if k else 0
    return {'n': n, 'k': k, 'd': d, 'd_bound': basis_bound, 'l': l, 'm': m,
            'Hx': Hx, 'Hz': Hz, 'Lx': Lx, 'Lz': Lz, 'type': 'css_matrix',
            'name': f'bivariate bicycle l={l} m={m}'}


# ---------------------------------------------------------------------------
# Rotated surface-code geometry: patches, merged patches, and their checks.
#
# A patch of R rows and C columns of data qubits (qubit (r, c) has index
# r * C + c within the patch) carries the rotated surface code of
# qec.py's construction: a face whose top-left corner is (r, c), for
# -1 <= r < R and -1 <= c < C, is a Z check when r + c is even and an X check
# when it is odd; every face with four qubits is a check, and of the two-qubit
# faces on the edge the X faces sit on the top and bottom boundaries and the Z
# faces on the left and right ones. Logical X runs down column 0 (between the
# X boundaries) and logical Z along row 0 (between the Z boundaries), so the
# patch has X distance R and Z distance C.
#
# Lattice surgery places two d x d patches next to each other with a line of
# fresh qubits between them and measures the checks of the single larger patch
# they form. Stacked vertically (A above, gap row, B below) the merged patch
# measures Z_A Z_B; side by side (A left, gap column, B right) it measures
# X_A X_B. The product of the merged checks that equals the joint logical
# operator is found by GF(2) elimination (`product_subset`).
# ---------------------------------------------------------------------------

@dataclass
class Patch:
    """A rotated surface-code patch on qubits indexed by `index`."""
    rows: int
    cols: int
    index: dict[tuple[int, int], int]                 # (r, c) -> qubit index
    x_checks: list[list[int]] = field(default_factory=list)
    z_checks: list[list[int]] = field(default_factory=list)
    logical_x: list[int] = field(default_factory=list)   # column 0
    logical_z: list[int] = field(default_factory=list)   # row 0

    @property
    def qubits(self) -> list[int]:
        return sorted(self.index.values())


def faces(rows: int, cols: int) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """(X checks, Z checks) of an R x C rotated patch as lists of (r, c)."""
    xs: list[list[tuple[int, int]]] = []
    zs: list[list[tuple[int, int]]] = []
    for r in range(-1, rows):
        for c in range(-1, cols):
            cells = [(i, j) for i, j in ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))
                     if 0 <= i < rows and 0 <= j < cols]
            if len(cells) < 2:
                continue
            kind = 'Z' if (r + c) % 2 == 0 else 'X'
            if len(cells) == 2:
                top_bottom = r in (-1, rows - 1)
                left_right = c in (-1, cols - 1)
                if not ((top_bottom and not left_right and kind == 'X')
                        or (left_right and not top_bottom and kind == 'Z')):
                    continue
            (zs if kind == 'Z' else xs).append(cells)
    return xs, zs


def make_patch(rows: int, cols: int, offset: int = 0,
               index: dict[tuple[int, int], int] | None = None) -> Patch:
    """An R x C patch on qubits offset .. offset + R*C - 1 (row-major), or on
    an explicit (r, c) -> qubit map."""
    if index is None:
        index = {(r, c): offset + r * cols + c for r in range(rows) for c in range(cols)}
    xs, zs = faces(rows, cols)
    return Patch(rows, cols, index,
                 x_checks=[[index[p] for p in f] for f in xs],
                 z_checks=[[index[p] for p in f] for f in zs],
                 logical_x=[index[(r, 0)] for r in range(rows)],
                 logical_z=[index[(0, c)] for c in range(cols)])


def code_dict(p: Patch, name: str = '') -> dict:
    """The patch as a string-stabilizer code for the decoders below (qubits
    renumbered 0..n-1 in patch order)."""
    order = {q: k for k, q in enumerate(p.qubits)}
    n = len(order)

    def string(qs: list[int], ch: str) -> str:
        s = ['I'] * n
        for q in qs:
            s[order[q]] = ch
        return ''.join(s)

    stab = [string(f, 'X') for f in p.x_checks] + [string(f, 'Z') for f in p.z_checks]
    return {'n': n, 'stab': stab, 'lx': string(p.logical_x, 'X'), 'lz': string(p.logical_z, 'Z'),
            'd': min(p.rows, p.cols), 'alphabet': 'IXYZ',
            'name': name or f'rotated surface {p.rows}x{p.cols}'}


def h_relabel(d: int) -> list[int]:
    """perm[k] = the position a qubit at position k takes after transversal
    H, so the patch is again in the standard orientation: a quarter turn
    (transpose, then mirror the columns). Positions are r * d + c."""
    perm = [0] * (d * d)
    for r in range(d):
        for c in range(d):
            perm[r * d + c] = c * d + (d - 1 - r)
    return perm


@dataclass
class Surgery:
    """Two d x d patches A and B, the gap line between them, and the merged
    patch whose checks a lattice-surgery merge measures."""
    d: int
    kind: str                   # 'ZZ' (A above B) or 'XX' (A left of B)
    a: Patch
    b: Patch
    gap: list[int]
    merged: Patch
    new_checks: list[int]       # indices into merged checks (X then Z) that are new

    @property
    def n_data(self) -> int:
        return 2 * self.d * self.d + self.d


def surgery(d: int, kind: str = 'ZZ') -> Surgery:
    """The lattice-surgery layout for measuring Z_A Z_B or X_A X_B."""
    if d % 2 == 0 or d < 3:
        raise ValueError("distance must be odd and at least 3")
    kind = kind.upper()
    if kind == 'ZZ':
        rows, cols = 2 * d + 1, d
        idx = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
        a = make_patch(d, d, index={(r, c): idx[(r, c)] for r in range(d) for c in range(d)})
        b = make_patch(d, d, index={(r, c): idx[(r + d + 1, c)] for r in range(d) for c in range(d)})
        gap = [idx[(d, c)] for c in range(d)]
    elif kind == 'XX':
        rows, cols = d, 2 * d + 1
        idx = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
        a = make_patch(d, d, index={(r, c): idx[(r, c)] for r in range(d) for c in range(d)})
        b = make_patch(d, d, index={(r, c): idx[(r, c + d + 1)] for r in range(d) for c in range(d)})
        gap = [idx[(r, d)] for r in range(d)]
    else:
        raise ValueError("lattice surgery measures ZZ or XX")
    merged = make_patch(rows, cols, index=idx)
    old = {tuple(sorted(f)) for f in a.x_checks + b.x_checks + a.z_checks + b.z_checks}
    checks = merged.x_checks + merged.z_checks
    new = [k for k, f in enumerate(checks) if tuple(sorted(f)) not in old]
    return Surgery(d, kind, a, b, gap, merged, new)


def product_subset(checks: list[list[int]], target: list[int], n: int) -> list[int]:
    """Indices of checks (supports, all of one Pauli type) whose product is
    the operator on `target`, by GF(2) elimination; raises if none exists."""
    m = len(checks)
    A = np.zeros((n, m), dtype=np.uint8)
    for k, f in enumerate(checks):
        A[f, k] = 1
    b = np.zeros(n, dtype=np.uint8)
    b[target] = 1
    aug = np.concatenate([A, b[:, None]], axis=1)
    row, pivots = 0, []
    for col in range(m):
        hits = np.nonzero(aug[row:, col])[0]
        if hits.size == 0:
            continue
        piv = row + hits[0]
        if piv != row:
            aug[[row, piv]] = aug[[piv, row]]
        mask = aug[:, col].astype(bool)
        mask[row] = False
        aug[mask] ^= aug[row]
        pivots.append(col)
        row += 1
        if row == n:
            break
    if aug[row:, -1].any():
        raise ValueError("the target is not a product of these checks")
    x = np.zeros(m, dtype=np.uint8)
    for r_, col in enumerate(pivots):
        x[col] = aug[r_, -1]
    return [k for k in range(m) if x[k]]


# ---------------------------------------------------------------------------
# QUBASIC code library: stabilizer codes by name, and the QEC and DISTILL commands.
# ---------------------------------------------------------------------------

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
    # Precondition: both strings over the IXYZ alphabet. The try keeps the
    # per-trial hot path free of per-character validation.
    try:
        return ''.join(_PMUL[(pa, pb)] for pa, pb in zip(a, b))
    except KeyError as e:
        raise ValueError(f"Pauli strings must be over IXYZ (got {e.args[0]})") from None


def _weight(p: str) -> int:
    return sum(1 for c in p if c != 'I')


class CodesMixin(ApplicationsHost):
    """Named stabilizer codes, their parameters and encoding demonstrations."""

    def _qec_code(self, name: str, distance: int = 3) -> dict:
        """Return a code descriptor: data qubit count, stabilizers, logical X/Z,
        distance, and the error alphabet the decoder corrects. Descriptors are
        cached per (name, distance), so decoder tables built on them persist."""
        cache = self.__dict__.setdefault('_qec_cache', {})
        key = (name.upper(), distance)
        if key not in cache:
            cache[key] = self._qec_code_build(name, distance)
        return cache[key]

    def _qec_code_build(self, name: str, distance: int = 3) -> dict:
        name = name.upper()
        if name in ('REP', 'REPETITION', 'BITFLIP'):
            d = distance if distance % 2 == 1 else distance + 1
            stab = []
            for i in range(d - 1):
                s = ['I'] * d
                s[i] = 'Z'
                s[i + 1] = 'Z'
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
                    s = ['I'] * 9
                    s[a] = 'Z'
                    s[c] = 'Z'
                    stab.append(''.join(s))
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

        def idx(r: int, c: int) -> int:
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

    def _qec_binary(self, code: dict) -> dict:
        """Cached symplectic matrices and logical vectors of a string code."""
        if '_bin' in code:
            return code['_bin']

        def vec(s: str) -> tuple[np.ndarray, np.ndarray]:
            x = np.array([ch in 'XY' for ch in s], dtype=np.uint8)
            z = np.array([ch in 'ZY' for ch in s], dtype=np.uint8)
            return x, z

        xs, zs = zip(*(vec(s) for s in code['stab']))
        lx = vec(code['lx'])
        lz = vec(code['lz'])
        sx, sz = np.array(xs), np.array(zs)
        b = {'Sx': sx, 'Sz': sz, 'LX': lx, 'LZ': lz,
             'rank': _gf2_rank(np.concatenate([sx, sz], axis=1))}
        code['_bin'] = b
        return b

    def _code_rate(self, code: dict, p: float, trials: int, rng: np.random.Generator,
                   decoder: str | None) -> tuple[float, str]:
        """(logical error rate, decoder label) for a string code."""
        if decoder == 'bp':
            return self._qec_bb_rate(_css_from_strings(code), p, trials, rng), 'BP+OSD decoder'
        chosen = self._choose_decoder(code, decoder)
        rate = self._logical_error_rate(code, p, trials, rng, decoder=chosen)
        return rate, self._DECODER_LABEL[chosen]

    def cmd_qec(self, rest: str) -> None:
        """QEC LIST | QEC <code> [distance] — a code's stabilizers and logical operators, or the list of codes.

        Lists the built-in codes, or prints a code's stabilizer generators and
        logical operators (and verifies they form a valid code)."""
        arg = rest.strip().upper()
        if not arg or arg == 'LIST':
            self.io.writeln("\n  Built-in QEC codes:")
            self.io.writeln("    REP [d]     repetition / bit-flip code, odd distance d (default 3)")
            self.io.writeln("    STEANE      [[7,1,3]] CSS code")
            self.io.writeln("    SHOR        [[9,1,3]] code")
            self.io.writeln("    SURFACE [d] rotated surface code, odd distance d (default 3)")
            self.io.writeln("    BB [l m]    bivariate-bicycle qLDPC (6 6 -> [[72,12,6]], 12 6 -> [[144,12,12]])")
            self.io.writeln("  Decoders: minimum-weight lookup (LOOKUP), exact matching (MATCH),")
            self.io.writeln("            union-find (UF), pymatching (MWPM), BP+OSD (BP, BB codes)")
            self.io.writeln("  Noise:    code capacity (default), PHENOM (space-time memory),")
            self.io.writeln("            CIRCUIT (stim circuit-level); LATTICE RATE for surgery")
            return
        parts = arg.split()
        if parts[0] == 'BB':
            try:
                l = int(parts[1]) if len(parts) > 1 else 6
                m = int(parts[2]) if len(parts) > 2 else 6
                self._qec_show_bb(l, m)
            except Exception as e:
                self.io.writeln(f"?QEC ERROR: {e}")
            return
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
            self.io.writeln("\n  15-to-1 magic-state distillation:")
            self.io.writeln(f"    input error p      = {p:g}")
            self.io.writeln(f"    acceptance rate    = {acc:.4f}")
            self.io.writeln(f"    output error p_out = {p_out:.3e}   (~35 p^3 = {35 * p ** 3:.3e})")
            self.io.writeln(f"    suppression p/p_out= {p / p_out:.1f}x" if p_out > 0 else
                            "    output error: none observed")
            self.variables['_DISTILL_POUT'] = p_out
        except Exception as e:
            self.io.writeln(f"?DISTILL ERROR: {e}")


# ---------------------------------------------------------------------------
# QUBASIC decoders: minimum-weight lookup tables, blossom matching and BP+OSD.
# ---------------------------------------------------------------------------

def _bp_decode(H: np.ndarray, syndrome: np.ndarray, p: float,
               max_iter: int = 30, alpha: float = 0.8) -> tuple[np.ndarray, np.ndarray, bool]:
    """Normalized min-sum belief propagation for syndrome decoding.

    Returns (error_estimate, posterior_llrs, converged). The check-node sign
    convention absorbs the syndrome, so a satisfied decoding has H e = s.
    """
    r, n = H.shape
    rows, cols = np.nonzero(H)
    L0 = float(np.log((1.0 - p) / max(p, 1e-12)))
    edge_by_check = [np.where(rows == c)[0] for c in range(r)]
    edge_by_bit = [np.where(cols == b)[0] for b in range(n)]
    M_bc = np.full(len(rows), L0)
    M_cb = np.zeros(len(rows))
    Lq = np.full(n, L0)
    sgn_syn = 1.0 - 2.0 * syndrome.astype(float)
    e_hat = np.zeros(n, np.uint8)
    for _ in range(max_iter):
        for c in range(r):
            e = edge_by_check[c]
            vals = M_bc[e]
            signs = np.where(vals >= 0, 1.0, -1.0)
            absv = np.abs(vals)
            order = np.argsort(absv)
            m1 = absv[order[0]]
            m2 = absv[order[1]] if e.size > 1 else m1
            prod_sign = float(np.prod(signs)) * sgn_syn[c]
            mins = np.full(e.size, m1)
            mins[order[0]] = m2
            M_cb[e] = alpha * prod_sign * signs * mins
        for b in range(n):
            eb = edge_by_bit[b]
            tot = L0 + float(np.sum(M_cb[eb]))
            Lq[b] = tot
            M_bc[eb] = tot - M_cb[eb]
        e_hat = (Lq < 0).astype(np.uint8)
        if np.array_equal(H.dot(e_hat) % 2, syndrome):
            return e_hat, Lq, True
    return e_hat, Lq, False


def _osd0(H: np.ndarray, syndrome: np.ndarray, Lq: np.ndarray) -> np.ndarray:
    """Ordered-statistics decoding, order 0.

    Columns are ranked most-likely-in-error first (ascending posterior LLR),
    Gaussian elimination picks a pivot set concentrated on those columns, and
    the syndrome is solved with all non-pivot bits at 0.
    """
    n = H.shape[1]
    order = np.argsort(Lq)
    Hw = np.concatenate([H[:, order], syndrome[:, None]], axis=1).astype(np.uint8)
    row = 0
    pivots: list[int] = []
    for col in range(n):
        if row >= Hw.shape[0]:
            break
        hits = np.nonzero(Hw[row:, col])[0]
        if hits.size == 0:
            continue
        piv = row + hits[0]
        if piv != row:
            Hw[[row, piv]] = Hw[[piv, row]]
        mask = Hw[:, col].astype(bool)
        mask[row] = False
        Hw[mask] ^= Hw[row]
        pivots.append(col)
        row += 1
    e_perm = np.zeros(n, np.uint8)
    for i, col in enumerate(pivots):
        e_perm[col] = Hw[i, -1]
    e = np.zeros(n, np.uint8)
    e[order] = e_perm
    return e


# ---------------------------------------------------------------------------
# Union-find decoding (Delfosse and Nickerson, Quantum 5, 595 (2021)).
#
# On a decoding graph whose vertices are detectors plus one boundary vertex
# and whose edges are error mechanisms, each defect starts a cluster. Every
# odd cluster (an odd number of defects and no boundary vertex) grows one
# unit along each edge leaving it per round; an edge whose growth reaches its
# length joins the clusters at its ends. When no odd cluster is left, the
# peeling decoder picks a correction inside the grown edges: a spanning
# forest, rooted at the boundary wherever the boundary was reached, is taken
# apart from the leaves, each defect taking the edge to its parent and
# handing its defect up. Unweighted graphs give every edge length 2 (growth
# in half edges); a weighted graph scales its log-likelihood weights so the
# lightest edge has length 16, fine enough that growth follows the weights.
# ---------------------------------------------------------------------------

def union_find(n_vertices: int, edges: Sequence[tuple[int, int]], lengths: Sequence[int],
               defects: Iterable[int], boundary: int | None = None) -> list[int]:
    """The indices of edges forming a correction for `defects` (the
    boundary vertex, when given, absorbs any parity), by union-find cluster
    growth and peeling."""
    from collections import deque
    incident: list[list[int]] = [[] for _ in range(n_vertices)]
    for e, (u, v) in enumerate(edges):
        incident[u].append(e)
        incident[v].append(e)
    parent = list(range(n_vertices))
    members = [[v] for v in range(n_vertices)]
    parity = [0] * n_vertices
    touches = [False] * n_vertices
    if boundary is not None:
        touches[boundary] = True
    marked = sorted({int(d) for d in defects})
    for d in marked:
        parity[d] = 1

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def odd_clusters() -> set[int]:
        return {r for r in map(find, marked) if parity[r] and not touches[r]}

    growth = [0] * len(edges)
    grown = [False] * len(edges)
    active = odd_clusters()
    while active:
        joined: list[int] = []
        grew = False
        for root in active:
            for v in members[root]:
                for e in incident[v]:
                    if not grown[e]:
                        grew = True
                        growth[e] += 1
                        if growth[e] >= lengths[e]:
                            grown[e] = True
                            joined.append(e)
        if not grew:
            raise ValueError("no correction: an odd cluster has nowhere to grow")
        for e in joined:
            a, b = find(edges[e][0]), find(edges[e][1])
            if a != b:
                if len(members[a]) < len(members[b]):
                    a, b = b, a
                parent[b] = a
                members[a].extend(members[b])
                members[b] = []
                parity[a] ^= parity[b]
                touches[a] = touches[a] or touches[b]
        active = odd_clusters()
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_vertices)]
    for e, (u, v) in enumerate(edges):
        if grown[e]:
            adj[u].append((v, e))
            adj[v].append((u, e))
    seen = [False] * n_vertices
    via = [-1] * n_vertices
    order: list[int] = []
    for root in ([boundary] if boundary is not None else []) + marked:
        if seen[root]:
            continue
        seen[root] = True
        queue = deque([root])
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, e in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    via[v] = e
                    queue.append(v)
    state = [False] * n_vertices
    for d in marked:
        state[d] = True
    correction = []
    for v in reversed(order):
        if via[v] >= 0 and state[v]:
            e = via[v]
            correction.append(e)
            u = edges[e][0] if edges[e][1] == v else edges[e][1]
            state[u] = not state[u]
            state[v] = False
    return correction


class DecodingMixin(ApplicationsHost):
    """Decoder choice and syndrome decoding for code-capacity experiments."""

    # Lookup tables are built while the reachable syndrome space is at most
    # this large; past it the decoder falls back to matching.
    LOOKUP_MAX_SYNDROMES = 1 << 16

    _DECODER_LABEL = {'lookup': 'lookup decoder', 'match': 'blossom matching decoder',
                      'uf': 'union-find decoder', 'pymatching': 'pymatching MWPM decoder'}

    # Decoder flags accepted by LOGICAL_ERROR_RATE and THRESHOLD.
    _DECODER_FLAGS = {'LOOKUP': 'lookup', 'MATCH': 'match', 'MATCHING': 'match',
                      'UF': 'uf', 'UNION-FIND': 'uf', 'MWPM': 'pymatching',
                      'BP': 'bp'}

    def _lookup_feasible(self, code: dict) -> bool:
        b = self._qec_binary(code)
        return code['n'] <= 64 and (1 << b['rank']) <= self.LOOKUP_MAX_SYNDROMES

    def _qec_lookup_table(self, code: dict) -> dict:
        """Minimum-weight lookup decoder: syndrome (as an int, stabilizer i =
        bit i) -> (x_mask, z_mask) recovery, found by enumerating errors in
        order of increasing weight until every reachable syndrome is covered.
        Cached on the code."""
        if '_lookup' in code:
            return code['_lookup']
        if not self._lookup_feasible(code):
            raise ValueError(f"{code['name']}: lookup table would need "
                             f"2^{self._qec_binary(code)['rank']} syndromes")
        b = self._qec_binary(code)
        n, m = code['n'], len(code['stab'])
        # Syndrome contribution of X and Z on each qubit, as bitmasks.
        cx = [sum(int(b['Sz'][i, q]) << i for i in range(m)) for q in range(n)]
        cz = [sum(int(b['Sx'][i, q]) << i for i in range(m)) for q in range(n)]
        paulis = [('X', 1, 0), ('Y', 1, 1), ('Z', 0, 1)]
        paulis = [p for p in paulis if p[0] in code['alphabet']]
        target = 1 << b['rank']
        table: dict = {0: (0, 0)}
        for w in range(1, n + 1):
            if len(table) >= target:
                break
            for support in itertools.combinations(range(n), w):
                for assign in itertools.product(paulis, repeat=w):
                    synd = xm = zm = 0
                    for q, (_, px, pz) in zip(support, assign):
                        if px:
                            synd ^= cx[q]
                            xm |= 1 << q
                        if pz:
                            synd ^= cz[q]
                            zm |= 1 << q
                    if synd not in table:
                        table[synd] = (xm, zm)
                if len(table) >= target:
                    break
        code['_lookup'] = table
        return table

    def _qec_decoder(self, code: dict) -> dict:
        """The minimum-weight lookup decoder as syndrome tuple -> recovery
        Pauli string (one entry per reachable syndrome)."""
        n, m = code['n'], len(code['stab'])
        out = {}
        for synd, (xm, zm) in self._qec_lookup_table(code).items():
            key = tuple((synd >> i) & 1 for i in range(m))
            out[key] = ''.join('Y' if (xm >> q) & 1 and (zm >> q) & 1 else
                               'X' if (xm >> q) & 1 else
                               'Z' if (zm >> q) & 1 else 'I' for q in range(n))
        return out

    def _match_graph(self, code: dict) -> list:
        """Per error type, the detector graph of a matchable CSS code: the
        detectors, its edges (detector, detector or boundary, qubit), and
        shortest-path lengths and qubit paths between every pair of detectors
        and from each detector to the boundary. Cached."""
        if '_match' in code:
            return code['_match']
        from collections import deque
        n, stabs = code['n'], code['stab']
        graphs = []
        for etype, dtype in (('X', 'Z'), ('Z', 'X')):
            det = [i for i, s in enumerate(stabs) if set(s) - {'I'} == {dtype}]
            if not det:
                continue
            pos = {d: k for k, d in enumerate(det)}
            B = len(det)
            adj: list[list] = [[] for _ in range(B + 1)]
            edges: list[tuple[int, int, int]] = []
            for q in range(n):
                touch = [pos[i] for i in det if stabs[i][q] == dtype]
                if len(touch) > 2:
                    raise ValueError("matching needs each qubit in at most two checks of a type")
                if len(touch) == 2:
                    adj[touch[0]].append((touch[1], q))
                    adj[touch[1]].append((touch[0], q))
                    edges.append((touch[0], touch[1], q))
                elif len(touch) == 1:
                    adj[touch[0]].append((B, q))
                    adj[B].append((touch[0], q))
                    edges.append((touch[0], B, q))
            dist = np.full((B + 1, B + 1), -1, dtype=np.int64)
            paths: dict = {}
            for src in range(B):
                # BFS parents: node -> (parent node, qubit on the edge to it)
                prev: dict[int, tuple[int, int]] = {src: (src, -1)}
                dq = deque([src])
                while dq:
                    u = dq.popleft()
                    for v, q in adj[u]:
                        if v not in prev:
                            prev[v] = (u, q)
                            if v != B:
                                dq.append(v)      # the boundary is a sink, not a relay
                for tgt in prev:
                    qs: list[int] = []
                    node = tgt
                    while node != src:
                        node, q = prev[node]
                        qs.append(q)
                    dist[src, tgt] = len(qs)
                    paths[(src, tgt)] = qs
            graphs.append({'etype': etype, 'det': det, 'dist': dist, 'paths': paths, 'B': B,
                           'edges': edges})
        code['_match'] = graphs
        return graphs

    def _match_recovery(self, code: dict, synd_bits: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(rx, rz) recovery from exact minimum-weight perfect matching of the
        syndrome defects (blossom algorithm via rustworkx); each defect pairs
        with another or with the boundary through its own boundary copy."""
        import rustworkx as rx_
        n = code['n']
        rx_v = np.zeros(n, dtype=np.uint8)
        rz_v = np.zeros(n, dtype=np.uint8)
        for g in self._match_graph(code):
            defects = [k for k, i in enumerate(g['det']) if synd_bits[i]]
            if not defects:
                continue
            k = len(defects)
            dist, B = g['dist'], g['B']
            wmax = int(dist.max()) + 1
            graph = rx_.PyGraph()
            graph.add_nodes_from(range(2 * k))
            for a in range(k):
                for c in range(a + 1, k):
                    d = dist[defects[a], defects[c]]
                    if d >= 0:
                        graph.add_edge(a, c, wmax - int(d))
                    graph.add_edge(k + a, k + c, wmax)        # boundary copies pair freely
                db = dist[defects[a], B]
                if db >= 0:
                    graph.add_edge(a, k + a, wmax - int(db))
            matching = rx_.max_weight_matching(graph, max_cardinality=True,
                                               weight_fn=lambda w: w)
            target = rx_v if g['etype'] == 'X' else rz_v
            for u, v in matching:
                u, v = min(u, v), max(u, v)
                if u >= k:
                    continue                                   # boundary-boundary
                src = defects[u]
                dst = B if v >= k else defects[v]
                for q in g['paths'].get((src, dst), ()):
                    target[q] ^= 1
        return rx_v, rz_v

    def _uf_recovery(self, code: dict, synd_bits: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(rx, rz) recovery from the union-find decoder on each error type's
        detector graph, every qubit an edge of length 2."""
        n = code['n']
        rx_v = np.zeros(n, dtype=np.uint8)
        rz_v = np.zeros(n, dtype=np.uint8)
        for g in self._match_graph(code):
            defects = [k for k, i in enumerate(g['det']) if synd_bits[i]]
            if not defects:
                continue
            edges = g['edges']
            target = rx_v if g['etype'] == 'X' else rz_v
            for e in union_find(g['B'] + 1, [(u, v) for u, v, _ in edges], [2] * len(edges),
                                defects, boundary=g['B']):
                target[edges[e][2]] ^= 1
        return rx_v, rz_v

    def _qec_matching_decode(self, code: dict, syndrome: Sequence[int]) -> str:
        """Matching decoder on one syndrome; returns the recovery Pauli string."""
        rxv, rzv = self._match_recovery(code, list(syndrome))
        return ''.join('Y' if a and b else 'X' if a else 'Z' if b else 'I'
                       for a, b in zip(rxv, rzv))

    def _matchable(self, code: dict) -> bool:
        try:
            self._match_graph(code)
            return all(set(s) - {'I'} in ({'X'}, {'Z'}) for s in code['stab'])
        except ValueError:
            return False

    def _choose_decoder(self, code: dict, requested: str | None = None) -> str:
        """'lookup', 'match', 'uf' or 'pymatching' for a string code: the
        requested one when it applies, else lookup while feasible, else
        matching."""
        if requested in ('lookup',) and self._lookup_feasible(code):
            return 'lookup'
        if requested == 'pymatching':
            try:
                import pymatching  # noqa: F401
                return 'pymatching'
            except ImportError:
                requested = 'match'
        if requested in ('match', 'uf'):
            if not self._matchable(code):
                raise ValueError(f"{code['name']} is not matchable (each qubit must be in "
                                 f"at most two checks of a type); use LOOKUP or BP")
            return requested
        if requested == 'lookup':
            raise ValueError(f"{code['name']}: lookup table would need "
                             f"2^{self._qec_binary(code)['rank']} syndromes; use MATCH")
        if self._lookup_feasible(code):
            return 'lookup'
        if self._matchable(code):
            return 'match'
        raise ValueError(f"{code['name']}: no decoder fits (lookup too large, not matchable)")

    def _syndromes(self, code: dict, ex: np.ndarray, ez: np.ndarray) -> np.ndarray:
        """Syndrome bits (rows: error samples, columns: the code's stabilizers)
        of X parts ex and Z parts ez."""
        b = self._qec_binary(code)
        return ((ex.astype(np.int64) @ b['Sz'].T.astype(np.int64)
                 + ez.astype(np.int64) @ b['Sx'].T.astype(np.int64)) & 1).astype(np.uint8)

    def _decode_batch(self, code: dict, synd: np.ndarray, decoder: str) -> tuple[np.ndarray, np.ndarray]:
        """(X recovery, Z recovery) for each syndrome row, by the lookup table,
        matching or union-find; each distinct syndrome is decoded once."""
        n = code['n']
        keys, inverse = np.unique(np.packbits(synd, axis=1, bitorder='little'),
                                  axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        m = synd.shape[1]
        rec_x = np.zeros((len(keys), n), dtype=np.uint8)
        rec_z = np.zeros((len(keys), n), dtype=np.uint8)
        table = self._qec_lookup_table(code) if decoder == 'lookup' else None
        for k, packed in enumerate(keys):
            bits = np.unpackbits(packed, bitorder='little')[:m]
            if not bits.any():
                continue
            if table is not None:
                s_val = int(sum(int(v) << i for i, v in enumerate(bits)))
                xm, zm = table.get(s_val, (0, 0))
                rec_x[k] = [(xm >> q) & 1 for q in range(n)]
                rec_z[k] = [(zm >> q) & 1 for q in range(n)]
            elif decoder == 'uf':
                rec_x[k], rec_z[k] = self._uf_recovery(code, bits)
            else:
                rec_x[k], rec_z[k] = self._match_recovery(code, bits)
        return rec_x[inverse], rec_z[inverse]

    def _split_decoder_flags(self, parts: list[str]) -> tuple[list[str], str | None, set]:
        """(remaining tokens, requested decoder, other flags) from a command."""
        decoder = None
        other = set()
        rest = []
        for t in parts:
            u = t.upper()
            if u in self._DECODER_FLAGS:
                decoder = self._DECODER_FLAGS[u]
            elif u in ('CIRCUIT', 'PHENOM', 'STIM'):
                other.add(u)
            else:
                rest.append(t)
        return rest, decoder, other


# ---------------------------------------------------------------------------
# Detector error models: independent error mechanisms, sampled and decoded.
#
# A model lists error mechanisms, each with a probability, the detectors it
# flips and the observables it flips. Sampling draws every mechanism
# independently for many shots at once; decoding pairs the fired detectors
# (and the boundary) by exact minimum-weight perfect matching on the graph
# whose edges are the mechanisms that flip one or two detectors, weighted
# log((1 - p) / p), and predicts the observables the matched paths flip
# (blossom algorithm via rustworkx). A shot fails when a predicted observable
# differs from the actual one. Each distinct detector pattern is decoded once.
# ---------------------------------------------------------------------------

class DetectorModel:
    """Error mechanisms over `n_det` detectors and `n_obs` observables."""

    def __init__(self, n_det: int, n_obs: int = 1) -> None:
        self.n_det = n_det
        self.n_obs = n_obs
        self.probs: list[float] = []
        self.dets: list[tuple[int, ...]] = []
        self.obs: list[int] = []            # bitmask over observables
        # What sampling draws, when it differs from the decoding graph: the
        # mechanisms as they occur, including those that flip more than two
        # detectors (the graph holds their one- and two-detector parts).
        self.events: list[tuple[float, tuple[int, ...], int]] = []
        self._graph: Any = None
        self._uf: Any = None

    def add(self, p: float, dets: Sequence[int], obs_mask: int = 0) -> None:
        """A mechanism of probability p flipping `dets` and the observables in obs_mask."""
        dets = tuple(sorted(set(int(d) for d in dets)))
        if len(dets) > 2:
            raise ValueError("a matchable mechanism flips at most two detectors")
        if p <= 0 or (not dets and not obs_mask):
            return
        self.probs.append(float(p))
        self.dets.append(dets)
        self.obs.append(int(obs_mask))
        self._graph = None
        self._uf = None

    def add_event(self, p: float, dets: Sequence[int], obs_mask: int = 0) -> None:
        """A mechanism sampling draws (it may flip any number of detectors)."""
        if p > 0:
            self.events.append((float(p), tuple(sorted(set(int(d) for d in dets))), int(obs_mask)))

    # ── sampling ──────────────────────────────────────────────────────

    def sample(self, shots: int, rng: Any) -> tuple[np.ndarray, np.ndarray]:
        """(detector flips [shots x n_det], observable flips [shots x n_obs]),
        drawing the events when there are any, else the graph's mechanisms."""
        if self.events:
            probs = [e[0] for e in self.events]
            pairs = [(e[1], e[2]) for e in self.events]
        else:
            probs = list(self.probs)
            pairs = list(zip(self.dets, self.obs))
        m = len(probs)
        fired = rng.random((shots, m)) < np.array(probs)[None, :]
        D = np.zeros((m, self.n_det), dtype=np.uint8)
        L = np.zeros((m, self.n_obs), dtype=np.uint8)
        for k, (ds, o) in enumerate(pairs):
            D[k, list(ds)] = 1
            for j in range(self.n_obs):
                L[k, j] = (o >> j) & 1
        f = fired.astype(np.int32)
        return ((f @ D) & 1).astype(np.uint8), ((f @ L) & 1).astype(np.uint8)

    # ── matching ──────────────────────────────────────────────────────

    def _build_graph(self) -> dict:
        """Shortest paths between detectors (node n_det is the boundary),
        with the observable parity of each path."""
        if self._graph is not None:
            return self._graph
        B = self.n_det
        adj: list[dict[int, tuple[float, int]]] = [dict() for _ in range(B + 1)]
        for p, ds, o in zip(self.probs, self.dets, self.obs):
            if not ds:
                continue
            u, v = (ds[0], B) if len(ds) == 1 else ds
            w = math.log((1 - p) / p) if p < 0.5 else 1e-9
            if v not in adj[u] or w < adj[u][v][0]:
                adj[u][v] = (w, o)
                adj[v][u] = (w, o)
        dist = np.full((B + 1, B + 1), np.inf)
        par = np.zeros((B + 1, B + 1), dtype=np.int64)
        for src in range(B + 1):
            d = {src: 0.0}
            o_par = {src: 0}
            heap = [(0.0, src)]
            while heap:
                du, u = heapq.heappop(heap)
                if du > d.get(u, np.inf):
                    continue
                if u == B and src != B:
                    continue            # the boundary absorbs paths, it relays nothing
                for v, (w, o) in adj[u].items():
                    nd = du + w
                    if nd < d.get(v, np.inf) - 1e-12:
                        d[v] = nd
                        o_par[v] = o_par[u] ^ o
                        heapq.heappush(heap, (nd, v))
            for v, dv in d.items():
                dist[src, v] = dv
                par[src, v] = o_par[v]
        self._graph = {'dist': dist, 'par': par, 'B': B}
        return self._graph

    def decode(self, fired: np.ndarray) -> int:
        """Observable mask predicted for one detector pattern."""
        defects = [int(i) for i in np.flatnonzero(fired)]
        if not defects:
            return 0
        import rustworkx as rx
        g = self._build_graph()
        dist, par, B = g['dist'], g['par'], g['B']
        k = len(defects)
        finite = dist[np.isfinite(dist)]
        big = float(finite.max() if finite.size else 1.0) * 2 + 10.0

        def w(x: float) -> int:          # the blossom matcher takes integer weights
            return int(round(x * 1000))
        graph = rx.PyGraph()
        graph.add_nodes_from(range(2 * k))
        for a in range(k):
            for c in range(a + 1, k):
                dac = dist[defects[a], defects[c]]
                if np.isfinite(dac):
                    graph.add_edge(a, c, w(big - dac))
                graph.add_edge(k + a, k + c, w(big))
            dab = dist[defects[a], B]
            if np.isfinite(dab):
                graph.add_edge(a, k + a, w(big - dab))
        matching = rx.max_weight_matching(graph, max_cardinality=True, weight_fn=lambda w: w)
        mask = 0
        for u, v in matching:
            u, v = min(u, v), max(u, v)
            if u >= k:
                continue
            mask ^= int(par[defects[u], B if v >= k else defects[v]])
        return mask

    def _uf_graph(self) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """The decoding graph for union-find: one edge per detector pair (the
        likeliest mechanism), its integer length and its observable mask."""
        if self._uf is not None:
            return self._uf
        B = self.n_det
        best: dict[tuple[int, int], tuple[float, int]] = {}
        for p, ds, o in zip(self.probs, self.dets, self.obs):
            if not ds:
                continue
            key = (ds[0], B) if len(ds) == 1 else (ds[0], ds[1])
            w = math.log((1 - p) / p) if p < 0.5 else 0.0
            if key not in best or w < best[key][0]:
                best[key] = (w, o)
        keys = list(best)
        positive = [w for w, _ in best.values() if w > 0]
        unit = min(positive) / 16 if positive else 1.0
        self._uf = (keys, [max(1, round(best[k][0] / unit)) for k in keys], [best[k][1] for k in keys])
        return self._uf

    def decode_uf(self, fired: np.ndarray) -> int:
        """Observable mask predicted for one detector pattern by union-find."""
        defects = [int(i) for i in np.flatnonzero(fired)]
        if not defects:
            return 0
        edges, lengths, masks = self._uf_graph()
        mask = 0
        for e in union_find(self.n_det + 1, edges, lengths, defects, boundary=self.n_det):
            mask ^= masks[e]
        return mask

    def logical_failure_rate(self, shots: int, rng: Any, decoder: str = 'match') -> tuple[float, np.ndarray]:
        """(rate of shots where any observable is mispredicted, per-observable
        failure rates), decoding by matching or, for 'uf', union-find."""
        decode = self.decode_uf if decoder == 'uf' else self.decode
        det, obs = self.sample(shots, rng)
        keys, inverse = np.unique(np.packbits(det, axis=1, bitorder='little'),
                                  axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        pred = np.zeros((len(keys), self.n_obs), dtype=np.uint8)
        for i, packed in enumerate(keys):
            bits = np.unpackbits(packed, bitorder='little')[:self.n_det]
            mask = decode(bits)
            for j in range(self.n_obs):
                pred[i, j] = (mask >> j) & 1
        wrong = pred[inverse] ^ obs
        return float(np.mean(wrong.any(axis=1))), wrong.mean(axis=0)


# ---------------------------------------------------------------------------
# Circuit-level noise for memory experiments, with no external dependency.
#
# A memory experiment prepares the code's logical |0>, measures every check
# for `rounds` rounds with the syndrome-extraction circuit, and reads out the
# data qubits; the logical observable is Z_L. This module writes that circuit
# as a list of operations (reset R, Hadamard H, CNOT CX, measurement M, and
# noise sites), finds the detectors and the observable each single fault flips
# by propagating it through the circuit as a Pauli frame (all faults at once,
# as columns of bit matrices), and collects the faults into a detector error
# model (dem.DetectorModel) that space-time matching decodes.
#
# The noise is the standard circuit-level depolarizing model of strength p:
# each CNOT is followed by a two-qubit depolarizing channel (the 15
# non-identity Paulis with probability p/15 each), each Hadamard and each data
# qubit idle during a CNOT layer by a one-qubit channel (p/3 each), each
# reset prepares |1> with probability p, and each measurement reports the
# wrong result with probability p. Data qubits also take a one-qubit channel
# at the start of every round.
#
# Surface-code checks are measured in the hook-safe order (see
# _surface_schedule). A fault that flips more than two of the detectors a Z
# memory experiment watches is split into single-fault parts for the decoding
# graph, as decomposition into graph-like errors does, while sampling draws
# every fault as it occurs.
# to_stim() gives the same circuit in stim's language, for comparison.
# ---------------------------------------------------------------------------

_PAULIS_1 = ((1, 0), (1, 1), (0, 1))                      # X, Y, Z as (x, z)
_PAULIS_2 = tuple((a, b) for a in ((0, 0),) + _PAULIS_1 for b in ((0, 0),) + _PAULIS_1
                  if a != (0, 0) or b != (0, 0))


@dataclass
class MemoryCircuit:
    """A memory experiment's circuit, its measurements, detectors and observable."""
    n_qubits: int
    ops: list[tuple] = field(default_factory=list)     # ('R'|'H'|'M', q) or ('CX', c, t) or noise
    n_meas: int = 0
    detectors: list[list[int]] = field(default_factory=list)   # measurement indices XORed
    observable: list[int] = field(default_factory=list)
    data: list[int] = field(default_factory=list)
    label: str = ''

    def measure(self, q: int) -> int:
        self.ops.append(('M', q, self.n_meas))
        self.n_meas += 1
        return self.n_meas - 1


def _surface_schedule(patch: Patch) -> tuple[list[tuple[str, list[int]]], dict[int, tuple[int, int]]]:
    """Each check as (kind, its data qubits in measurement order), absent
    corners as -1. Logical X runs down a column and logical Z along a row, so
    an X check visits its corners in Z order (top-left, top-right, bottom-left,
    bottom-right), leaving any hook error on a horizontal pair, and a Z check
    in N order (top-left, bottom-left, top-right, bottom-right), leaving it on
    a vertical pair: across the logical operator of the same type."""
    pos = {q: rc for rc, q in patch.index.items()}
    xs, zs = faces(patch.rows, patch.cols)
    checks = []
    for kind, kind_faces in (('X', xs), ('Z', zs)):
        for face in kind_faces:
            r0 = min(r for r, _ in face) if face else 0
            c0 = min(c for _, c in face) if face else 0
            corners = {(r, c) for r, c in face}
            # a boundary face of two qubits: its top-left corner is outside the patch
            if len(face) == 2:
                rows = {r for r, _ in face}
                cols = {c for _, c in face}
                if len(rows) == 1:          # top or bottom edge: a horizontal pair
                    r0 = r0 - 1 if r0 == 0 else r0
                if len(cols) == 1:          # left or right edge: a vertical pair
                    c0 = c0 - 1 if c0 == 0 else c0
            order = ([(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)] if kind == 'X'
                     else [(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)])
            checks.append((kind, [patch.index[rc] if rc in corners else -1 for rc in order]))
    return checks, pos


def surface_memory(d: int, rounds: int) -> MemoryCircuit:
    """Z memory of the distance-d rotated surface code for `rounds` rounds."""
    patch = make_patch(d, d)
    checks, _ = _surface_schedule(patch)
    n_data = d * d
    anc = list(range(n_data, n_data + len(checks)))
    circ = MemoryCircuit(n_qubits=n_data + len(checks), data=list(range(n_data)),
                         label=f"rotated surface code, distance {d}")
    for q in range(n_data):
        circ.ops += [('R', q), ('NOISE_RESET', q)]
    prev: dict[int, int] = {}
    for rnd in range(rounds):
        for q in range(n_data):
            circ.ops.append(('NOISE_1', q))
        for a in anc:
            circ.ops += [('R', a), ('NOISE_RESET', a)]
        for (kind, _), a in zip(checks, anc):
            if kind == 'X':
                circ.ops += [('H', a), ('NOISE_1', a)]
        for layer in range(4):
            busy = set()
            for (kind, order), a in zip(checks, anc):
                q = order[layer]
                if q < 0:
                    continue
                c, t = (a, q) if kind == 'X' else (q, a)
                circ.ops += [('CX', c, t), ('NOISE_2', c, t)]
                busy.add(q)
            for q in range(n_data):
                if q not in busy:
                    circ.ops.append(('NOISE_1', q))
        for (kind, _), a in zip(checks, anc):
            if kind == 'X':
                circ.ops += [('H', a), ('NOISE_1', a)]
        for (kind, _), a in zip(checks, anc):
            circ.ops.append(('NOISE_MEAS', a))
            m = circ.measure(a)
            if kind == 'Z':
                circ.detectors.append([m] if a not in prev else [m, prev[a]])
            prev[a] = m
    final = {}
    for q in range(n_data):
        circ.ops.append(('NOISE_MEAS', q))
        final[q] = circ.measure(q)
    for (kind, order), a in zip(checks, anc):
        if kind == 'Z':
            circ.detectors.append([final[q] for q in order if q >= 0] + [prev[a]])
    circ.observable = [final[q] for q in patch.logical_z]
    return circ


def repetition_memory(d: int, rounds: int) -> MemoryCircuit:
    """Memory of the distance-d bit-flip repetition code (ZZ checks)."""
    anc = list(range(d, 2 * d - 1))
    circ = MemoryCircuit(n_qubits=2 * d - 1, data=list(range(d)),
                         label=f"repetition code, distance {d}")
    for q in range(d):
        circ.ops += [('R', q), ('NOISE_RESET', q)]
    prev: dict[int, int] = {}
    for rnd in range(rounds):
        for q in range(d):
            circ.ops.append(('NOISE_1', q))
        for a in anc:
            circ.ops += [('R', a), ('NOISE_RESET', a)]
        for step in (0, 1):
            busy = set()
            for k, a in enumerate(anc):
                q = k + step
                circ.ops += [('CX', q, a), ('NOISE_2', q, a)]
                busy.add(q)
            for q in range(d):
                if q not in busy:
                    circ.ops.append(('NOISE_1', q))
        for a in anc:
            circ.ops.append(('NOISE_MEAS', a))
            m = circ.measure(a)
            circ.detectors.append([m] if a not in prev else [m, prev[a]])
            prev[a] = m
    final = {}
    for q in range(d):
        circ.ops.append(('NOISE_MEAS', q))
        final[q] = circ.measure(q)
    for k, a in enumerate(anc):
        circ.detectors.append([final[k], final[k + 1], prev[a]])
    circ.observable = [final[0]]
    return circ


def fault_effects(circ: MemoryCircuit, p: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(probability of each fault, detectors it flips [faults x detectors],
    observable it flips [faults]) for every single fault of the noise model."""
    probs: list[float] = []
    starts: list[tuple[int, tuple[tuple[int, int, int], ...]]] = []   # (op index, ((q, x, z), ...))
    for i, op in enumerate(circ.ops):
        kind = op[0]
        if kind == 'NOISE_1':
            for x, z in _PAULIS_1:
                probs.append(p / 3)
                starts.append((i, ((op[1], x, z),)))
        elif kind == 'NOISE_2':
            for (xa, za), (xb, zb) in _PAULIS_2:
                probs.append(p / 15)
                starts.append((i, ((op[1], xa, za), (op[2], xb, zb))))
        elif kind in ('NOISE_RESET', 'NOISE_MEAS'):
            probs.append(p)
            starts.append((i, ((op[1], 1, 0),)))
    nf, nq = len(probs), circ.n_qubits
    X = np.zeros((nf, nq), dtype=bool)
    Z = np.zeros((nf, nq), dtype=bool)
    meas = np.zeros((nf, circ.n_meas), dtype=bool)
    by_op: dict[int, list[int]] = {}
    for f, (i, _) in enumerate(starts):
        by_op.setdefault(i, []).append(f)
    for i, op in enumerate(circ.ops):
        kind = op[0]
        if kind == 'CX':
            c, t = op[1], op[2]
            X[:, t] ^= X[:, c]
            Z[:, c] ^= Z[:, t]
        elif kind == 'H':
            q = op[1]
            X[:, q], Z[:, q] = Z[:, q].copy(), X[:, q].copy()
        elif kind == 'R':
            X[:, op[1]] = False
            Z[:, op[1]] = False
        elif kind == 'M':
            meas[:, op[2]] = X[:, op[1]]
        for f in by_op.get(i, ()):
            for q, x, z in starts[f][1]:
                X[f, q] ^= bool(x)
                Z[f, q] ^= bool(z)
    D = np.zeros((nf, len(circ.detectors)), dtype=bool)
    for k, ms in enumerate(circ.detectors):
        D[:, k] = np.count_nonzero(meas[:, ms], axis=1) % 2 == 1
    L = np.count_nonzero(meas[:, circ.observable], axis=1) % 2 == 1
    return np.array(probs), D, L


def _combine(p: float, q: float) -> float:
    """Probability that exactly one of two independent mechanisms fires."""
    return p * (1 - q) + q * (1 - p)


def detector_model(circ: MemoryCircuit, p: float) -> tuple[DetectorModel, int]:
    """The decoding model of the circuit at noise strength p, and the number of
    faults that could not be split into matchable parts (left out)."""
    probs, D, L = fault_effects(circ, p)
    merged: dict[tuple[tuple[int, ...], int], float] = {}
    for pr, row, obs in zip(probs, D, L):
        dets = tuple(int(k) for k in np.flatnonzero(row))
        key = (dets, int(obs))
        if not dets and not obs:
            continue
        merged[key] = _combine(merged.get(key, 0.0), float(pr))
    simple = {k: v for k, v in merged.items() if len(k[0]) <= 2}
    edges = {k[0]: k[1] for k in simple}           # detector set -> its observable parity
    unsplit = 0
    for (dets, obs), pr in merged.items():
        if len(dets) <= 2:
            continue
        parts = _split(dets, obs, edges)
        if parts is None:
            unsplit += 1
            continue
        for part in parts:
            simple[part] = _combine(simple.get(part, 0.0), pr)
    model = DetectorModel(len(circ.detectors), 1)
    for (dets, obs), pr in simple.items():
        model.add(pr, dets, obs)
    for (dets, obs), pr in merged.items():
        model.add_event(pr, dets, obs)
    return model, unsplit


def _split(dets: tuple[int, ...], obs: int, edges: dict[tuple[int, ...], int]) -> list[tuple[tuple[int, ...], int]] | None:
    """Partition dets into one- and two-detector parts that are themselves
    single-fault mechanisms, with observable parities that add up to obs."""
    items = list(dets)
    if len(items) > 8:
        return None

    def search(rest: list[int], need: int) -> list[tuple[tuple[int, ...], int]] | None:
        if not rest:
            return [] if need == 0 else None
        first = rest[0]
        for size in (2, 1):
            for others in itertools.combinations(rest[1:], size - 1):
                part = tuple(sorted((first,) + others))
                if part not in edges:
                    continue
                left = [d for d in rest if d not in part]
                tail = search(left, need ^ edges[part])
                if tail is not None:
                    return [(part, edges[part])] + tail
        return None

    return search(items, obs)


def to_stim(circ: MemoryCircuit, p: float) -> str:
    """The circuit with its noise, detectors and observable in stim's language."""
    lines = []
    for op in circ.ops:
        kind = op[0]
        if kind == 'R':
            lines.append(f"R {op[1]}")
        elif kind == 'H':
            lines.append(f"H {op[1]}")
        elif kind == 'CX':
            lines.append(f"CX {op[1]} {op[2]}")
        elif kind == 'M':
            lines.append(f"M {op[1]}")
        elif kind == 'NOISE_1':
            lines.append(f"DEPOLARIZE1({p}) {op[1]}")
        elif kind == 'NOISE_2':
            lines.append(f"DEPOLARIZE2({p}) {op[1]} {op[2]}")
        elif kind in ('NOISE_RESET', 'NOISE_MEAS'):
            lines.append(f"X_ERROR({p}) {op[1]}")
    for ms in circ.detectors:
        lines.append("DETECTOR " + ' '.join(f"rec[{m - circ.n_meas}]" for m in ms))
    lines.append("OBSERVABLE_INCLUDE(0) " + ' '.join(f"rec[{m - circ.n_meas}]" for m in circ.observable))
    return '\n'.join(lines)


def memory_circuit(code: str, d: int, rounds: int) -> MemoryCircuit:
    name = code.upper()
    if name in ('SURFACE', 'SURF'):
        if d < 2:
            raise ValueError("the surface code needs distance 2 or more")
        return surface_memory(d, rounds)
    if name in ('REP', 'REPETITION', 'BITFLIP'):
        if d < 2:
            raise ValueError("the repetition code needs distance 2 or more")
        return repetition_memory(d, rounds)
    raise ValueError(f"circuit-level memory supports SURFACE and REP (got {code})")


# ---------------------------------------------------------------------------
# QUBASIC logical error rates: code-capacity, phenomenological and
# circuit-level memory experiments, and THRESHOLD.
# ---------------------------------------------------------------------------

class RatesMixin(ApplicationsHost):
    """LOGICAL_ERROR_RATE and THRESHOLD."""

    def _logical_error_rate(self, code: dict, p: float, trials: int, rng: np.random.Generator,
                            uf: bool = False, decoder: str | None = None) -> float:
        """Code-capacity Monte Carlo: i.i.d. errors (X for a bit-flip code,
        depolarizing otherwise), decode, count residual logical errors.
        Vectorized over trials; each distinct syndrome is decoded once."""
        decoder = self._choose_decoder(code, 'uf' if uf else decoder)
        if decoder == 'pymatching':
            return self._qec_mwpm_rate(code, p, trials, rng)
        b = self._qec_binary(code)
        n = code['n']
        if code['alphabet'] == 'IX':
            ex = (rng.random((trials, n)) < p).astype(np.uint8)
            ez = np.zeros_like(ex)
        else:
            hit = rng.random((trials, n)) < p
            which = rng.integers(0, 3, (trials, n))          # 0 X, 1 Y, 2 Z
            ex = (hit & (which != 2)).astype(np.uint8)
            ez = (hit & (which != 0)).astype(np.uint8)
        synd = self._syndromes(code, ex, ez)
        rec_x, rec_z = self._decode_batch(code, synd, decoder)
        res_x = ex ^ rec_x
        res_z = ez ^ rec_z
        lxx, lxz = b['LX']
        lzx, lzz = b['LZ']
        fail = (((res_x.astype(np.int64) @ lxz + res_z.astype(np.int64) @ lxx) & 1)
                | ((res_x.astype(np.int64) @ lzz + res_z.astype(np.int64) @ lzx) & 1))
        return float(np.mean(fail))

    def cmd_logical_error_rate(self, rest: str) -> None:
        """LOGICAL_ERROR_RATE <code> [distance] <p> [trials] [LOOKUP|MATCH|UF|MWPM|BP|CIRCUIT [STIM]|PHENOM] — logical error rate under code-capacity, phenomenological or circuit-level noise.

        Code-capacity Monte Carlo (i.i.d. physical errors, decode, count
        residual logical errors), vectorized over trials. The decoder defaults
        to the minimum-weight lookup table while the code's syndrome space
        allows it and to exact minimum-weight matching (blossom) beyond;
        MATCH forces matching, UF the union-find decoder (also under PHENOM
        and CIRCUIT), MWPM pymatching when installed and BP the internal
        BP+OSD. CIRCUIT switches to circuit-level noise: a
        memory experiment of d rounds of syndrome extraction (MemoryCircuit),
        its detector error model found by propagating every fault through the
        circuit, decoded by space-time matching (the trailing counts are shots,
        default 5000, and rounds); CIRCUIT STIM runs stim's generated circuits
        with pymatching instead (default 100000 shots). PHENOM
        runs a memory experiment under phenomenological noise (data errors
        and check-measurement errors at rate p in each of d rounds, then a
        noisy readout) decoded by space-time matching, with no stim needed.
        BB codes decode with BP+OSD."""
        parts = rest.split()
        if len(parts) < 2:
            self.io.writeln("?USAGE: LOGICAL_ERROR_RATE <code> [distance] <p> [trials] "
                            "[LOOKUP|MATCH|UF|MWPM|BP|CIRCUIT|PHENOM]")
            return
        try:
            name = parts[0]
            idx = 1
            distance = 3
            _FLAGS = tuple(self._DECODER_FLAGS) + ('CIRCUIT', 'PHENOM', 'STIM')
            if name.upper() == 'BB':
                # BB takes l and m instead of a distance.
                l = m = 6
                if len(parts) > 3 and parts[1].isdigit() and parts[2].isdigit():
                    l, m = int(parts[1]), int(parts[2])
                    idx = 3
                tail = [t for t in parts[idx:] if t.upper() not in _FLAGS]
                p = float(self._eval_with_vars(tail[0], {}))
                if not 0.0 <= p <= 1.0:
                    raise ValueError(f"p={p} out of range (expected 0..1; "
                                     f"usage: LOGICAL_ERROR_RATE BB [l m] <p> [trials])")
                trials = int(tail[1]) if len(tail) > 1 else 1000
                code = _bb_code(l, m)
                rng = np.random.default_rng(self._seed)
                t0 = __import__('time').time()
                ler = self._qec_bb_rate(code, p, trials, rng)
                dt = __import__('time').time() - t0
                self.io.writeln(f"\n  {code['name']} [[{code['n']},{code['k']},"
                                f"{code['d'] or '?'}]]: physical p={p}, {trials} trials "
                                f"(BP+OSD decoder, {dt:.1f}s)")
                self.io.writeln(f"  Logical error rate = {ler:.6f}")
                self.variables['_LER'] = ler
                return
            parts, decoder, other = self._split_decoder_flags(parts)
            if len(parts) > 2 and parts[1].isdigit() and float(parts[1]) >= 1:
                distance = int(parts[1])
                idx = 2
            tail = parts[idx:]
            p = float(self._eval_with_vars(tail[0], {}))
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"p={p} out of range (expected 0..1)")
            if 'CIRCUIT' in other:
                stim = 'STIM' in other
                shots = int(tail[1]) if len(tail) > 1 else (100000 if stim else 5000)
                rounds = int(tail[2]) if len(tail) > 2 else None
                if stim:
                    self._qec_circuit_level(name, distance, p, shots, rounds)
                else:
                    self._qec_circuit_model(name, distance, p, shots, rounds, decoder)
                return
            if 'PHENOM' in other:
                shots = int(tail[1]) if len(tail) > 1 else 10000
                rounds = int(tail[2]) if len(tail) > 2 else None
                self._qec_phenomenological(name, distance, p, shots, rounds, decoder)
                return
            trials = int(tail[1]) if len(tail) > 1 else 20000
            code = self._qec_code(name, distance)
            rng = np.random.default_rng(self._seed)
            t0 = __import__('time').time()
            ler, dec = self._code_rate(code, p, trials, rng, decoder)
            dt = __import__('time').time() - t0
            self.io.writeln(f"\n  {code['name']}: physical p={p}, {trials} trials ({dec}, {dt:.1f}s)")
            self.io.writeln(f"  Logical error rate = {ler:.6f}")
            self.variables['_LER'] = ler
        except Exception as e:
            self.io.writeln(f"?LOGICAL_ERROR_RATE ERROR: {e}")

    def cmd_threshold(self, rest: str) -> None:
        """THRESHOLD <code> <p1> <p2> <steps> [d1 d2 ...] [TRIALS n] [LOOKUP|MATCH|UF|MWPM|BP|PHENOM|CIRCUIT] — logical error rate against p for several distances, and the threshold where they cross.

        Reports the logical error rate (with a one-sigma interval) across
        [p1, p2] for several code distances (default 3, 5, 7) and estimates the
        threshold: where the curves of the smallest and the largest distance
        cross, interpolated in log-log ($_THRESHOLD). Code-capacity noise by
        default, with the decoder LOGICAL_ERROR_RATE would pick unless a flag
        is given; PHENOM and CIRCUIT sweep memory experiments of d rounds under
        phenomenological or circuit-level noise."""
        parts = rest.split()
        if len(parts) < 4:
            self.io.writeln("?USAGE: THRESHOLD <code> <p1> <p2> <steps> [distances] "
                            "[TRIALS n] [LOOKUP|MATCH|UF|MWPM|BP|PHENOM|CIRCUIT]")
            return
        try:
            parts, decoder, other = self._split_decoder_flags(parts)
            mode = 'circuit' if 'CIRCUIT' in other else 'phenom' if 'PHENOM' in other else 'capacity'
            trials = 8000 if mode == 'capacity' else 3000
            if any(t.upper() == 'TRIALS' for t in parts):
                k = [t.upper() for t in parts].index('TRIALS')
                trials = int(parts[k + 1])
                parts = parts[:k] + parts[k + 2:]
            name = parts[0]
            p1 = float(self._eval_with_vars(parts[1], {}))
            p2 = float(self._eval_with_vars(parts[2], {}))
            steps = int(parts[3])
            distances = [int(x) for x in parts[4:]] if len(parts) > 4 else [3, 5, 7]
            ps = [p1 + (p2 - p1) * i / max(1, steps - 1) for i in range(steps)]
            rng = np.random.default_rng(self._seed)
            labels: list[str] = []
            rate_fns = []
            shown: list[int] = []
            spacetime = 'uf' if decoder == 'uf' else 'match'
            spacetime_label = 'union-find' if decoder == 'uf' else 'matching'
            for d in distances:
                if mode == 'capacity':
                    c = self._qec_code(name, d)
                    if any(c['d'] == s for s in shown):
                        continue
                    shown.append(c['d'])
                    labels.append('BP+OSD' if decoder == 'bp' else self._choose_decoder(c, decoder))
                    rate_fns.append(lambda p, c=c: self._code_rate(c, p, trials, rng, decoder)[0])
                elif mode == 'phenom':
                    c = self._qec_code(name, d)
                    shown.append(c['d'])
                    labels.append(spacetime_label)
                    rate_fns.append(lambda p, c=c: self._phenomenological_model(c, p, c['d'])
                                    .logical_failure_rate(trials, rng, spacetime)[0])
                else:
                    circ = memory_circuit(name, d, d)
                    shown.append(d)
                    labels.append(spacetime_label)
                    rate_fns.append(lambda p, circ=circ: detector_model(circ, p)[0]
                                    .logical_failure_rate(trials, rng, spacetime)[0])
            what = {'capacity': 'code capacity', 'phenom': 'phenomenological, d rounds',
                    'circuit': 'circuit level, d rounds'}[mode]
            self.io.writeln(f"\n  Threshold sweep ({name.upper()}, {what}), {trials} trials/point:")
            self.io.writeln("    p      " + ''.join(f"  d={d:<14}" for d in shown))
            self.io.writeln("  decoder  " + ''.join(f"  {lab[:14]:<16}" for lab in labels))
            table: list[list[float]] = []
            for p in ps:
                row = f"  {p:5.3f}  "
                rates = []
                for fn in rate_fns:
                    rate = float(fn(p))
                    sigma = (rate * (1 - rate) / trials) ** 0.5
                    rates.append(rate)
                    row += f"  {rate:<7.4f}+-{sigma:<7.4f}"
                table.append(rates)
                self.io.writeln(row)
            crossing = self._threshold_crossing(ps, table)
            if crossing is None:
                self.io.writeln(f"  No crossing of d={shown[0]} and d={shown[-1]} in [{p1:g}, {p2:g}]")
            else:
                self.io.writeln(f"  Threshold estimate: p = {crossing:.4f} (d={shown[0]} and d={shown[-1]} cross)")
                self.variables['_THRESHOLD'] = crossing
        except Exception as e:
            self.io.writeln(f"?THRESHOLD ERROR: {e}")

    @staticmethod
    def _threshold_crossing(ps: list[float], table: list[list[float]]) -> float | None:
        """Where the first and last columns of table cross, interpolated in
        log rate against log p (None without a sign change)."""
        if len(ps) < 2 or not table or len(table[0]) < 2:
            return None
        eps = 1e-6
        diffs = [np.log(r[-1] + eps) - np.log(r[0] + eps) for r in table]
        for k in range(len(ps) - 1):
            a, b = diffs[k], diffs[k + 1]
            if a == 0:
                return ps[k]
            if a < 0 < b or a > 0 > b:
                la, lb = np.log(max(ps[k], eps)), np.log(max(ps[k + 1], eps))
                return float(np.exp(la + (lb - la) * a / (a - b)))
        return None

    def _qec_css_matrices(self, code: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(Hz, Hx, lz_support, lx_support) as uint8 arrays from a string code.

        Hz rows are the Z-type stabilizers (they detect X errors), Hx rows the
        X-type ones (they detect Z errors).
        """
        n = code['n']
        Hz_rows, Hx_rows = [], []
        for s in code['stab']:
            kinds = set(s) - {'I'}
            row = np.array([0 if ch == 'I' else 1 for ch in s], np.uint8)
            if kinds == {'Z'}:
                Hz_rows.append(row)
            elif kinds == {'X'}:
                Hx_rows.append(row)
            else:
                raise ValueError("MWPM path needs a CSS code (pure X/Z stabilizers)")
        Hz = np.array(Hz_rows, np.uint8) if Hz_rows else np.zeros((0, n), np.uint8)
        Hx = np.array(Hx_rows, np.uint8) if Hx_rows else np.zeros((0, n), np.uint8)
        lz = np.array([1 if ch == 'Z' else 0 for ch in code['lz']], np.uint8)
        lx = np.array([1 if ch == 'X' else 0 for ch in code['lx']], np.uint8)
        return Hz, Hx, lz, lx

    def _qec_mwpm_rate(self, code: dict, p: float, trials: int, rng: np.random.Generator) -> float:
        import pymatching
        Hz, Hx, lz, lx = self._qec_css_matrices(code)
        n = code['n']
        if code['alphabet'] == 'IX':
            ex = (rng.random((trials, n)) < p).astype(np.uint8)
            ez = np.zeros_like(ex)
        else:
            hit = rng.random((trials, n)) < p
            which = rng.integers(0, 3, (trials, n))
            ex = (hit & ((which == 0) | (which == 1))).astype(np.uint8)
            ez = (hit & ((which == 2) | (which == 1))).astype(np.uint8)
        fail = np.zeros(trials, bool)
        for H, l_opp, e in ((Hz, lz, ex), (Hx, lx, ez)):
            if H.shape[0] == 0:
                continue
            if int(H.sum(axis=0).max()) > 2:
                raise ValueError("MWPM needs a matchable code (each qubit in <=2 "
                                 "same-type checks); use the lookup/UF/BP decoders")
            matcher = pymatching.Matching(H)
            synd = e.dot(H.T) % 2
            rec = matcher.decode_batch(synd).astype(np.uint8)
            resid = e ^ rec
            fail |= (resid.dot(l_opp) % 2).astype(bool)
        return float(np.mean(fail))

    _STIM_TASKS = {
        'REP': 'repetition_code:memory',
        'REPETITION': 'repetition_code:memory',
        'BITFLIP': 'repetition_code:memory',
        'SURFACE': 'surface_code:rotated_memory_z',
        'SURF': 'surface_code:rotated_memory_z',
    }

    def _qec_circuit_model(self, name: str, d: int, p: float, shots: int, rounds: int | None,
                           decoder: str | None = None) -> None:
        """A circuit-level memory experiment on the built-in model (no stim),
        decoded by space-time matching or, with UF, union-find."""
        rounds = rounds or d
        circ = memory_circuit(name, d, rounds)
        t0 = time.time()
        model, unsplit = detector_model(circ, p)
        spacetime = 'uf' if decoder == 'uf' else 'match'
        per_shot, _ = model.logical_failure_rate(shots, np.random.default_rng(self._seed), spacetime)
        dt = time.time() - t0
        per_round = 1.0 - (1.0 - per_shot) ** (1.0 / rounds) if per_shot < 1.0 else 1.0
        self.io.writeln(f"\n  {circ.label}: CIRCUIT-level noise p={p}, {rounds} rounds, {shots:,} shots")
        note = f", {unsplit} unsplit faults left out" if unsplit else ''
        self.io.writeln(f"  physical qubits = {circ.n_qubits}  detectors = {model.n_det}  "
                        f"fault mechanisms = {len(model.events)}{note}  "
                        f"(space-time {'union-find' if spacetime == 'uf' else 'matching'})")
        self.io.writeln(f"  Logical error rate = {per_shot:.3e} per shot   {per_round:.3e} per round")
        self.io.writeln(f"  ({dt:.2f}s)")
        self.variables['_LER'] = per_shot
        self.variables['_LER_ROUND'] = per_round

    def _qec_circuit_level(self, name: str, d: int, p: float,
                           shots: int, rounds: int | None) -> None:
        """The circuit-level memory experiment on stim's generated circuits."""
        try:
            import pymatching
            import stim
        except ImportError:
            self.io.writeln("?CIRCUIT STIM needs stim + pymatching (pip install stim pymatching); "
                            "CIRCUIT alone runs the built-in circuit model")
            return
        task = self._STIM_TASKS.get(name.upper())
        if task is None:
            self.io.writeln(f"?CIRCUIT mode supports REP and SURFACE (got {name})")
            return
        if d % 2 == 0:
            d += 1
        rounds = rounds or d
        circuit = stim.Circuit.generated(
            task, distance=d, rounds=rounds,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p)
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler(
            seed=self._seed if self._seed is not None else None)
        t0 = time.time()
        dets, obs = sampler.sample(shots, separate_observables=True)
        preds = matcher.decode_batch(dets)
        errs = int(np.sum(np.any(preds != obs, axis=1)))
        dt = time.time() - t0
        per_shot = errs / shots
        per_round = 1.0 - (1.0 - per_shot) ** (1.0 / rounds) if per_shot < 1.0 else 1.0
        label = 'rotated surface' if 'surface' in task else 'repetition'
        self.io.writeln(f"\n  {label} d={d}: CIRCUIT-level noise p={p}, "
                        f"{rounds} rounds, {shots:,} shots")
        self.io.writeln(f"  physical qubits = {circuit.num_qubits}  "
                        f"detectors = {dem.num_detectors}  (stim + pymatching MWPM)")
        self.io.writeln(f"  Logical error rate = {per_shot:.3e} per shot   "
                        f"{per_round:.3e} per round")
        self.io.writeln(f"  ({dt:.2f}s: {shots / max(dt, 1e-9):,.0f} shots/s)")
        self.variables['_LER'] = per_shot
        self.variables['_LER_ROUND'] = per_round

    def _phenomenological_model(self, code: dict, p: float, rounds: int) -> DetectorModel:
        """Detector model of a Z-basis memory experiment: before each of the
        rounds every data qubit takes an X error with probability p and every
        Z check outcome flips with probability p; a final readout (with X
        flips at rate p) closes the record. The observable is logical Z."""
        z_checks = [[q for q, ch in enumerate(st) if ch == 'Z'] for st in code['stab']
                    if set(st) - {'I'} == {'Z'}]
        if not z_checks:
            raise ValueError(f"{code['name']} has no Z checks to detect X errors")
        n = code['n']
        holds = [[k for k, f in enumerate(z_checks) if q in f] for q in range(n)]
        if any(len(h) > 2 for h in holds):
            raise ValueError(f"{code['name']} is not matchable (a qubit in more than two Z checks)")
        lz = {q for q, ch in enumerate(code['lz']) if ch in 'ZY'}
        nc = len(z_checks)

        def det(t: int, k: int) -> int:
            return (t - 1) * nc + k

        model = DetectorModel(nc * (rounds + 1), 1)
        for t in range(1, rounds + 2):
            for q in range(n):
                model.add(p, [det(t, k) for k in holds[q]], 1 if q in lz else 0)
            if t <= rounds:
                for k in range(nc):
                    model.add(p, [det(t, k), det(t + 1, k)], 0)
        return model

    def _qec_phenomenological(self, name: str, d: int, p: float, shots: int,
                              rounds: int | None, decoder: str | None = None) -> None:
        code = self._qec_code(name, d)
        rounds = rounds or code['d']
        model = self._phenomenological_model(code, p, rounds)
        rng = np.random.default_rng(self._seed)
        t0 = time.time()
        spacetime = 'uf' if decoder == 'uf' else 'match'
        rate, _ = model.logical_failure_rate(shots, rng, spacetime)
        dt = time.time() - t0
        per_round = 1.0 - (1.0 - rate) ** (1.0 / rounds) if rate < 1.0 else 1.0
        method = 'union-find' if spacetime == 'uf' else 'matching'
        self.io.writeln(f"\n  {code['name']}: phenomenological noise p={p}, {rounds} rounds, "
                        f"{shots:,} shots ({model.n_det} detectors, space-time {method}, {dt:.1f}s)")
        self.io.writeln(f"  Logical error rate = {rate:.3e} per shot   {per_round:.3e} per round")
        self.variables['_LER'] = rate
        self.variables['_LER_ROUND'] = per_round

    def _qec_bb_rate(self, code: dict, p: float, trials: int, rng: np.random.Generator) -> float:
        Hx, Hz, Lx, Lz = code['Hx'], code['Hz'], code['Lx'], code['Lz']
        n = code['n']
        bitflip = code.get('alphabet') == 'IX'
        # Depolarizing: marginal X (or Z) flip probability is 2p/3; a bit-flip
        # code sees X errors at rate p only.
        p_marg = p if bitflip else 2.0 * p / 3.0
        fails = 0
        for _ in range(trials):
            if bitflip:
                ex = (rng.random(n) < p).astype(np.uint8)
                ez = np.zeros(n, dtype=np.uint8)
            else:
                hit = rng.random(n) < p
                which = rng.integers(0, 3, n)
                ex = (hit & ((which == 0) | (which == 1))).astype(np.uint8)
                ez = (hit & ((which == 2) | (which == 1))).astype(np.uint8)
            failed = False
            for H, L, e in ((Hz, Lz, ex), (Hx, Lx, ez)):
                s = H.dot(e) % 2
                e_hat, Lq, ok = _bp_decode(H, s, p_marg)
                if not ok:
                    e_hat = _osd0(H, s, Lq)
                resid = e ^ e_hat
                if np.any(L.dot(resid) % 2):
                    failed = True
                    break
            fails += failed
        return fails / trials

    def _qec_show_bb(self, l: int, m: int) -> None:
        code = _bb_code(l, m)
        n, k = code['n'], code['k']
        d_txt = str(code['d']) if code['d'] else f"unknown (<= {code['d_bound']} from the logical basis)"
        self.io.writeln(f"\n  {code['name']}: [[{n},{k},{d_txt}]]")
        self.io.writeln(f"  Hx: {code['Hx'].shape[0]} X-checks, weight "
                        f"{int(code['Hx'].sum(axis=1).max())}; Hz: {code['Hz'].shape[0]} "
                        f"Z-checks, weight {int(code['Hz'].sum(axis=1).max())}")
        css_ok = not np.any(code['Hx'].dot(code['Hz'].T) % 2)
        self.io.writeln(f"  CSS commutation Hx Hz^T = 0: {css_ok}")
        self.io.writeln(f"  k from GF(2) ranks = {k}; logical basis {code['Lx'].shape[0]} X"
                        f" + {code['Lz'].shape[0]} Z operators")
        self.io.writeln(f"  Decode with: LOGICAL_ERROR_RATE BB {l} {m} <p> [trials]  (BP+OSD)")
