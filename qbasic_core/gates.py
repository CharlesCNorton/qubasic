"""Gate matrices, builders, and numpy simulation primitives."""

from __future__ import annotations

import numpy as np

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

def _np_gate_matrix(name: str, params: tuple[float, ...] = ()) -> np.ndarray:
    """Return the unitary matrix for a named gate via registry lookup."""
    canonical = GATE_ALIASES.get(name, name)
    builder = _GATE_BUILDERS.get(canonical)
    if builder is None:
        raise ValueError(f"No matrix for gate: {name}")
    return builder(params)


def _apply_gate_np(
    sv: np.ndarray, matrix: np.ndarray, qubits: list[int], n_qubits: int,
) -> np.ndarray:
    """Apply a k-qubit gate to a statevector using numpy tensordot.

    Returns a tensor of shape [2]*n (possibly non-contiguous view).
    This avoids an extra 32 GB copy for large qubit counts.
    The caller stores the result directly; previous sv is freed.
    """
    k = len(qubits)
    sv_t = sv.reshape([2] * n_qubits)  # view if already contiguous
    axes = [n_qubits - 1 - q for q in qubits]
    gate_t = matrix.reshape([2] * (2 * k))
    # Contract gate's input axes (k..2k-1) with sv's target axes
    result = np.tensordot(gate_t, sv_t, axes=(list(range(k, 2*k)), axes))
    # Move gate output axes (0..k-1) to the target positions (view, no copy)
    return np.moveaxis(result, list(range(k)), axes)


def _measure_np(
    sv: np.ndarray, qubit: int, n_qubits: int,
) -> tuple[int, np.ndarray]:
    """Mid-circuit measurement with Born-rule sampling and state collapse.

    Computes marginal probabilities for |0> and |1> on the target qubit,
    samples an outcome, and collapses + renormalizes the statevector.
    Includes a numerical floor to avoid division-by-zero when both
    marginals underflow to zero.
    """
    sv = np.ascontiguousarray(sv).reshape([2] * n_qubits)
    ax = n_qubits - 1 - qubit
    idx_0 = [slice(None)] * n_qubits
    idx_1 = [slice(None)] * n_qubits
    idx_0[ax] = 0
    idx_1[ax] = 1
    p0 = float(np.sum(np.abs(sv[tuple(idx_0)])**2))
    p1 = float(np.sum(np.abs(sv[tuple(idx_1)])**2))
    total = p0 + p1
    if total < 1e-300:
        # Numerical underflow: both marginals are essentially zero.
        # Default to outcome 0 and return a normalized |0> on this qubit.
        new_sv = np.zeros(2**n_qubits, dtype=complex)
        new_sv[0] = 1.0
        return 0, new_sv
    outcome = 0 if np.random.random() < p0 / total else 1
    new_sv = np.zeros_like(sv)
    if outcome == 0:
        new_sv[tuple(idx_0)] = sv[tuple(idx_0)] / np.sqrt(p0)
    else:
        new_sv[tuple(idx_1)] = sv[tuple(idx_1)] / np.sqrt(p1)
    return outcome, new_sv.reshape(-1)


def _sample_np(sv: np.ndarray, n_qubits: int, shots: int) -> dict[str, int]:
    """Sample measurement outcomes from a statevector."""
    probs = np.abs(np.ascontiguousarray(sv).ravel())**2
    probs /= probs.sum()
    indices = np.random.choice(probs.size, size=shots, p=probs)
    unique, ucounts = np.unique(indices, return_counts=True)
    return {format(idx, f'0{n_qubits}b'): int(cnt) for idx, cnt in zip(unique, ucounts)}


def _sample_one_np(sv: np.ndarray, n_qubits: int) -> str:
    """Sample a single measurement outcome."""
    counts = _sample_np(sv, n_qubits, 1)
    return next(iter(counts))
