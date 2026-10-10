"""Gate matrices, builders, and numpy simulation primitives."""

from __future__ import annotations

import itertools

import numpy as np

# Amplitudes per block for chunked in-place updates (64 MiB at complex128).
# States at or below this size are updated in one block.
_CHUNK_AMPS = 1 << 22

# ═══════════════════════════════════════════════════════════════════════
# Gate tables: name -> (num_params, num_qubits)
# ═══════════════════════════════════════════════════════════════════════

GATE_TABLE = {
    # 0-param, 1-qubit
    'H':    (0, 1), 'X':    (0, 1), 'Y':    (0, 1), 'Z':    (0, 1),
    'S':    (0, 1), 'T':    (0, 1), 'SDG':  (0, 1), 'TDG':  (0, 1),
    'SX':   (0, 1), 'ID':   (0, 1),
    # 1-param, 1-qubit
    'RX':   (1, 1), 'RY':   (1, 1), 'RZ':   (1, 1), 'P':    (1, 1),
    # 3-param, 1-qubit
    'U':    (3, 1),
    # 0-param, 2-qubit
    'CX':   (0, 2), 'CNOT': (0, 2), 'CZ':   (0, 2), 'CY':   (0, 2),
    'CH':   (0, 2), 'SWAP': (0, 2), 'DCX':  (0, 2), 'ISWAP':(0, 2),
    # 1-param, 2-qubit
    'CRX':  (1, 2), 'CRY':  (1, 2), 'CRZ':  (1, 2), 'CP':   (1, 2),
    'RXX':  (1, 2), 'RYY':  (1, 2), 'RZZ':  (1, 2),
    # 0-param, 3-qubit
    'CCX':  (0, 3), 'TOFFOLI': (0, 3), 'CSWAP': (0, 3), 'FREDKIN': (0, 3),
}

# Aliases
GATE_ALIASES = {
    'CNOT': 'CX', 'TOFFOLI': 'CCX', 'FREDKIN': 'CSWAP',
}


# ═══════════════════════════════════════════════════════════════════════
# Pre-computed constant gate matrices (allocated once, not per call)
# ═══════════════════════════════════════════════════════════════════════

_MAT_CY = np.eye(4, dtype=complex)
_MAT_CY[2,2], _MAT_CY[2,3], _MAT_CY[3,2], _MAT_CY[3,3] = 0, -1j, 1j, 0

_MAT_CH = np.eye(4, dtype=complex)
_h = 1/np.sqrt(2)
_MAT_CH[2,2], _MAT_CH[2,3], _MAT_CH[3,2], _MAT_CH[3,3] = _h, _h, _h, -_h

_MAT_CCX = np.eye(8, dtype=complex)
_MAT_CCX[6,6], _MAT_CCX[6,7], _MAT_CCX[7,6], _MAT_CCX[7,7] = 0, 1, 1, 0

_MAT_CSWAP = np.eye(8, dtype=complex)
_MAT_CSWAP[5,5], _MAT_CSWAP[5,6], _MAT_CSWAP[6,5], _MAT_CSWAP[6,6] = 0, 1, 1, 0

_MAT_H    = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
_MAT_X    = np.array([[0,1],[1,0]], dtype=complex)
_MAT_Y    = np.array([[0,-1j],[1j,0]], dtype=complex)
_MAT_Z    = np.array([[1,0],[0,-1]], dtype=complex)
_MAT_S    = np.array([[1,0],[0,1j]], dtype=complex)
_MAT_T    = np.array([[1,0],[0,np.exp(1j*np.pi/4)]], dtype=complex)
_MAT_SDG  = np.array([[1,0],[0,-1j]], dtype=complex)
_MAT_TDG  = np.array([[1,0],[0,np.exp(-1j*np.pi/4)]], dtype=complex)
_MAT_SX   = np.array([[1+1j,1-1j],[1-1j,1+1j]], dtype=complex) / 2
_MAT_ID   = np.eye(2, dtype=complex)
_MAT_CX   = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
_MAT_CZ   = np.diag([1, 1, 1, -1]).astype(complex)
_MAT_SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
_MAT_DCX  = np.array([[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]], dtype=complex)
_MAT_ISWAP = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)


# ═══════════════════════════════════════════════════════════════════════
# Parameterized gate matrix builders
# ═══════════════════════════════════════════════════════════════════════

def _mat_crx(p):
    t = p[0]
    c, s = np.cos(t/2), np.sin(t/2)
    m = np.eye(4, dtype=complex)
    m[2,2], m[2,3] = c, -1j*s
    m[3,2], m[3,3] = -1j*s, c
    return m

def _mat_cry(p):
    t = p[0]
    c, s = np.cos(t/2), np.sin(t/2)
    m = np.eye(4, dtype=complex)
    m[2,2], m[2,3] = c, -s
    m[3,2], m[3,3] = s, c
    return m

def _mat_crz(p):
    t = p[0]
    m = np.eye(4, dtype=complex)
    m[2,2], m[3,3] = np.exp(-1j*t/2), np.exp(1j*t/2)
    return m

def _mat_rxx(p):
    t = p[0]
    ct, st = np.cos(t/2), np.sin(t/2)
    m = np.zeros((4,4), dtype=complex)
    m[0,0] = m[1,1] = m[2,2] = m[3,3] = ct
    m[0,3] = m[3,0] = m[1,2] = m[2,1] = -1j*st
    return m

def _mat_ryy(p):
    # RYY(t) = exp(-i * t/2 * YY).  The off-diagonal signs are asymmetric
    # because Y = [[0,-i],[i,0]] yields YY with sign pattern:
    #   |00>-|11> coupling: +i sin(t/2)   (anti-diagonal corners)
    #   |01>-|10> coupling: -i sin(t/2)   (inner off-diagonal)
    t = p[0]
    ct, st = np.cos(t/2), np.sin(t/2)
    m = np.zeros((4,4), dtype=complex)
    m[0,0] = m[1,1] = m[2,2] = m[3,3] = ct
    m[0,3] = m[3,0] = 1j*st
    m[1,2] = m[2,1] = -1j*st
    return m


# ═══════════════════════════════════════════════════════════════════════
# Gate matrix registry: name -> callable(params) -> np.ndarray
# ═══════════════════════════════════════════════════════════════════════

# Adding a new gate only requires an entry here and in GATE_TABLE.
# 0-param gates return pre-computed constants (no allocation per call).
_GATE_BUILDERS = {
    # 0-param, 1-qubit
    'H':     lambda p: _MAT_H,
    'X':     lambda p: _MAT_X,
    'Y':     lambda p: _MAT_Y,
    'Z':     lambda p: _MAT_Z,
    'S':     lambda p: _MAT_S,
    'T':     lambda p: _MAT_T,
    'SDG':   lambda p: _MAT_SDG,
    'TDG':   lambda p: _MAT_TDG,
    'SX':    lambda p: _MAT_SX,
    'ID':    lambda p: _MAT_ID,
    # 1-param, 1-qubit
    'RX':    lambda p: np.array([[np.cos(p[0]/2), -1j*np.sin(p[0]/2)],
                                 [-1j*np.sin(p[0]/2), np.cos(p[0]/2)]], dtype=complex),
    'RY':    lambda p: np.array([[np.cos(p[0]/2), -np.sin(p[0]/2)],
                                 [np.sin(p[0]/2), np.cos(p[0]/2)]], dtype=complex),
    'RZ':    lambda p: np.array([[np.exp(-1j*p[0]/2), 0],
                                 [0, np.exp(1j*p[0]/2)]], dtype=complex),
    'P':     lambda p: np.array([[1, 0], [0, np.exp(1j*p[0])]], dtype=complex),
    # 3-param, 1-qubit
    'U':     lambda p: np.array([
                 [np.cos(p[0]/2), -np.exp(1j*p[2])*np.sin(p[0]/2)],
                 [np.exp(1j*p[1])*np.sin(p[0]/2),
                  np.exp(1j*(p[1]+p[2]))*np.cos(p[0]/2)]], dtype=complex),
    # 0-param, 2-qubit
    'CX':    lambda p: _MAT_CX,
    'CZ':    lambda p: _MAT_CZ,
    'CY':    lambda p: _MAT_CY,
    'CH':    lambda p: _MAT_CH,
    'SWAP':  lambda p: _MAT_SWAP,
    'DCX':   lambda p: _MAT_DCX,
    'ISWAP': lambda p: _MAT_ISWAP,
    # 1-param, 2-qubit
    'CRX':   _mat_crx,
    'CRY':   _mat_cry,
    'CRZ':   _mat_crz,
    'CP':    lambda p: np.diag([1, 1, 1, np.exp(1j*p[0])]).astype(complex),
    'RXX':   _mat_rxx,
    'RYY':   _mat_ryy,
    'RZZ':   lambda p: np.diag([np.exp(-1j*p[0]/2), np.exp(1j*p[0]/2),
                                 np.exp(1j*p[0]/2), np.exp(-1j*p[0]/2)]),
    # 0-param, 3-qubit
    'CCX':   lambda p: _MAT_CCX,
    'CSWAP': lambda p: _MAT_CSWAP,
}


# ═══════════════════════════════════════════════════════════════════════
# Numpy simulation primitives
# ═══════════════════════════════════════════════════════════════════════

# Fast-path lookup for 0-param gates: return pre-computed constants directly,
# skipping the builder callable and its lambda overhead.
_GATE_CONSTANTS: dict[str, np.ndarray] = {
    'H':     _MAT_H,
    'X':     _MAT_X,
    'Y':     _MAT_Y,
    'Z':     _MAT_Z,
    'S':     _MAT_S,
    'T':     _MAT_T,
    'SDG':   _MAT_SDG,
    'TDG':   _MAT_TDG,
    'SX':    _MAT_SX,
    'ID':    _MAT_ID,
    'CX':    _MAT_CX,
    'CZ':    _MAT_CZ,
    'CY':    _MAT_CY,
    'CH':    _MAT_CH,
    'SWAP':  _MAT_SWAP,
    'DCX':   _MAT_DCX,
    'ISWAP': _MAT_ISWAP,
    'CCX':   _MAT_CCX,
    'CSWAP': _MAT_CSWAP,
}


def _np_gate_matrix(name: str, params: tuple[float, ...] = ()) -> np.ndarray:
    """Return the unitary matrix for a named gate via registry lookup."""
    canonical = GATE_ALIASES.get(name, name)
    # Fast path: 0-param gates return pre-computed constants directly.
    mat = _GATE_CONSTANTS.get(canonical)
    if mat is not None:
        return mat
    builder = _GATE_BUILDERS.get(canonical)
    if builder is None:
        raise ValueError(f"No matrix for gate: {name}")
    return builder(params)


def _blocks(psi: np.ndarray, n_qubits: int, pinned: dict[int, int] | None = None,
            keep: tuple[int, ...] = (), max_amps: int | None = None):
    """Yield views that tile psi (shape [2]*n) without copying.

    `pinned` fixes axes to one index ({axis: 0|1}); the remaining free axes
    not in `keep` are split, leading axes first, until a block holds at most
    max_amps amplitudes. Pinned and split axes become length-1 slices, so
    every block keeps all n axes and stays a writable view.
    """
    if max_amps is None:
        max_amps = _CHUNK_AMPS
    pinned = pinned or {}
    free = [a for a in range(n_qubits) if a not in pinned and a not in keep]
    per_block = 1 << (n_qubits - len(pinned))
    split: list[int] = []
    for a in free:
        if per_block <= max_amps:
            break
        split.append(a)
        per_block >>= 1
    for bits in itertools.product((0, 1), repeat=len(split)):
        sl: list = [slice(None)] * n_qubits
        for a, v in pinned.items():
            sl[a] = slice(v, v + 1)
        for a, v in zip(split, bits):
            sl[a] = slice(v, v + 1)
        yield psi[tuple(sl)]


def _as_tensor(sv: np.ndarray, n_qubits: int) -> np.ndarray:
    """A [2]*n view of a C-contiguous statevector (raises if it would copy)."""
    if not sv.flags.c_contiguous:
        raise ValueError("statevector must be C-contiguous for in-place updates")
    return sv.reshape([2] * n_qubits)


def _apply_gate_inplace(
    sv: np.ndarray, matrix: np.ndarray, qubits: list[int], n_qubits: int,
) -> np.ndarray:
    """Apply a k-qubit gate to a contiguous statevector in place.

    The matrix is MSB-first over `qubits` (qubits[0] is the most significant
    bit of its row index). Work proceeds in blocks of at most _CHUNK_AMPS
    amplitudes, so the scratch memory is one block rather than a second copy
    of the state. Returns sv.
    """
    k = len(qubits)
    if k == 0:
        return sv
    psi = _as_tensor(sv, n_qubits)
    axes = [n_qubits - 1 - q for q in qubits]
    gate_t = np.asarray(matrix, dtype=sv.dtype).reshape([2] * (2 * k))
    src, dst = list(range(k, 2 * k)), list(range(k))
    for block in _blocks(psi, n_qubits, keep=tuple(axes)):
        res = np.tensordot(gate_t, block, axes=(src, axes))
        block[...] = np.moveaxis(res, dst, axes)
    return sv


def _apply_gate_np(
    sv: np.ndarray, matrix: np.ndarray, qubits: list[int], n_qubits: int,
) -> np.ndarray:
    """Apply a k-qubit gate and return a new flat statevector (input unchanged)."""
    out = np.array(sv, dtype=np.result_type(sv, np.complex64), copy=True).reshape(-1)
    return _apply_gate_inplace(out, matrix, qubits, n_qubits)


def _subspace_norm2(psi: np.ndarray, n_qubits: int, axis: int, value: int) -> float:
    """Sum of |amplitude|^2 over the half of psi where `axis` == value."""
    total = 0.0
    for block in _blocks(psi, n_qubits, pinned={axis: value}):
        total += float(np.vdot(block, block).real)
    return total


def _marginal_p1(sv: np.ndarray, qubit: int, n_qubits: int) -> float:
    """Probability that `qubit` reads 1, without copying the state."""
    return _subspace_norm2(_as_tensor(sv, n_qubits), n_qubits, n_qubits - 1 - qubit, 1)


def _measure_inplace(sv: np.ndarray, qubit: int, n_qubits: int) -> int:
    """Born-rule measurement of one qubit; collapses sv in place.

    Draws from numpy's global generator (so SEED reproduces runs). When both
    marginals underflow, the state resets to |0...0> and the outcome is 0.
    """
    psi = _as_tensor(sv, n_qubits)
    ax = n_qubits - 1 - qubit
    p0 = _subspace_norm2(psi, n_qubits, ax, 0)
    p1 = _subspace_norm2(psi, n_qubits, ax, 1)
    total = p0 + p1
    if total < 1e-300:
        sv[...] = 0
        sv[0] = 1.0
        return 0
    outcome = 0 if np.random.random() < p0 / total else 1
    if outcome == 0 and p0 < 1e-300:
        outcome = 1
    elif outcome == 1 and p1 < 1e-300:
        outcome = 0
    scale = 1.0 / np.sqrt(p0 if outcome == 0 else p1)
    for block in _blocks(psi, n_qubits, pinned={ax: 1 - outcome}):
        block[...] = 0
    for block in _blocks(psi, n_qubits, pinned={ax: outcome}):
        block *= scale
    return outcome


def _measure_np(
    sv: np.ndarray, qubit: int, n_qubits: int,
) -> tuple[int, np.ndarray]:
    """Measure one qubit; returns (outcome, collapsed copy). Input unchanged."""
    out = np.array(sv, dtype=np.result_type(sv, np.complex64), copy=True).reshape(-1)
    outcome = _measure_inplace(out, qubit, n_qubits)
    return outcome, out


def _sample_counts(sv: np.ndarray, n_qubits: int, shots: int) -> dict[str, int]:
    """Sample measurement outcomes from a statevector.

    States of at most _CHUNK_AMPS amplitudes are sampled in one draw. Larger
    states draw a multinomial split of the shots over contiguous chunks, then
    sample within each chunk, so scratch memory stays one chunk.
    """
    flat = np.asarray(sv).reshape(-1)
    size = flat.size
    if size <= _CHUNK_AMPS:
        probs = np.abs(flat) ** 2
        probs /= probs.sum()
        indices = np.random.choice(size, size=shots, p=probs)
    else:
        starts = list(range(0, size, _CHUNK_AMPS))
        masses = np.array([float(np.vdot(flat[s:s + _CHUNK_AMPS],
                                          flat[s:s + _CHUNK_AMPS]).real) for s in starts])
        per = np.random.multinomial(shots, masses / masses.sum())
        parts = []
        for s, k in zip(starts, per):
            if k == 0:
                continue
            pr = np.abs(flat[s:s + _CHUNK_AMPS]) ** 2
            pr /= pr.sum()
            parts.append(s + np.random.choice(pr.size, size=int(k), p=pr))
        indices = np.concatenate(parts)
    unique, ucounts = np.unique(indices, return_counts=True)
    return {format(int(idx), f'0{n_qubits}b'): int(cnt) for idx, cnt in zip(unique, ucounts)}


def _sample_np(sv: np.ndarray, n_qubits: int, shots: int) -> dict[str, int]:
    """Sample measurement outcomes from a statevector."""
    return _sample_counts(sv, n_qubits, shots)


def _sample_one_np(sv: np.ndarray, n_qubits: int) -> str:
    """Sample a single measurement outcome."""
    counts = _sample_counts(sv, n_qubits, 1)
    return next(iter(counts))
