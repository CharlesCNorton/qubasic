"""Rotated surface-code geometry: patches, merged patches, and their checks.

A patch of R rows and C columns of data qubits (qubit (r, c) has index
r * C + c within the patch) carries the rotated surface code of
qec.py's construction: a face whose top-left corner is (r, c), for
-1 <= r < R and -1 <= c < C, is a Z check when r + c is even and an X check
when it is odd; every face with four qubits is a check, and of the two-qubit
faces on the edge the X faces sit on the top and bottom boundaries and the Z
faces on the left and right ones. Logical X runs down column 0 (between the
X boundaries) and logical Z along row 0 (between the Z boundaries), so the
patch has X distance R and Z distance C.

Lattice surgery places two d x d patches next to each other with a line of
fresh qubits between them and measures the checks of the single larger patch
they form. Stacked vertically (A above, gap row, B below) the merged patch
measures Z_A Z_B; side by side (A left, gap column, B right) it measures
X_A X_B. The product of the merged checks that equals the joint logical
operator is found by GF(2) elimination (`product_subset`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    """The patch as a string-stabilizer code for QECMixin's decoders (qubits
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
