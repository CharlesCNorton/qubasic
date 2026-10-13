"""QUBASIC simulation tools: the memory model, the in-place numpy gate
engine, statevector analysis (reduced states, entropies, entanglement),
coupling-map topologies and the gate backends.
"""

from __future__ import annotations

import itertools
import os
import re
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from qubasic_core.parser import GATE_ALIASES

# ---------------------------------------------------------------------------
# QUBASIC capacity model — what a simulation costs and what fits in memory.
#
# Peak resident memory of Aer statevector runs, measured with Qiskit 2.3 and
# Aer 0.17, relative to the raw state S (2^n amplitudes at the run precision):
#
#     measured run, counts only                      1.04 S
#     measure-free run that saves the final state    1.04 S  (Aer hands the
#                                                             buffer over)
#     run that saves the state and then measures     2.04 S  (Aer copies it)
#
# A saved state always comes back as complex128, so at single precision a
# saved state adds 2 S. The numpy engines (LOCC, live measurement, STEP)
# update the state in place in chunks of CHUNK_BYTES, so they need S plus one
# chunk, and a second S while a prefix snapshot is held.
#
# Runs are sized against the memory the OS reports as available, capped by
# the environment variable QUBASIC_MEMORY_LIMIT when it is set (a size such as
# 4GB, 512MB or 2.5GiB), so a shared machine can reserve memory for other work.
# ---------------------------------------------------------------------------

CHUNK_BYTES = 1 << 26        # scratch for chunked in-place numpy updates (64 MiB)
AER_FACTOR = 1.04
HEADROOM = 0.85              # fraction of available memory one run may claim

# Methods whose memory is set by register width; the others (stabilizer,
# matrix_product_state, extended_stabilizer) scale with structure instead.
WIDTH_BOUND = ('statevector', 'automatic', 'density_matrix', 'unitary', 'superop')

GB = 1e9
LIMIT_ENV = 'QUBASIC_MEMORY_LIMIT'

_UNITS = {'': 1, 'B': 1, 'K': 10**3, 'KB': 10**3, 'M': 10**6, 'MB': 10**6, 'G': 10**9, 'GB': 10**9,
          'T': 10**12, 'TB': 10**12, 'KIB': 1 << 10, 'MIB': 1 << 20, 'GIB': 1 << 30, 'TIB': 1 << 40}
_RE_SIZE = re.compile(r'^\s*(\d+(?:\.\d*)?|\.\d+)\s*([A-Za-z]*)\s*$')


def parse_size(text: str) -> int:
    """Bytes in a size: a number with an optional unit, decimal (KB, MB, GB, TB,
    or K, M, G, T) or binary (KiB, MiB, GiB, TiB); a bare number is bytes."""
    m = _RE_SIZE.match(text)
    unit = m.group(2).upper() if m else None
    if m is None or unit not in _UNITS:
        raise ValueError(f"not a size: {text!r} (use e.g. 4GB, 512MB, 2GiB)")
    return int(float(m.group(1)) * _UNITS[unit])


def memory_limit() -> int | None:
    """The QUBASIC_MEMORY_LIMIT ceiling in bytes; None when unset or unreadable."""
    raw = os.environ.get(LIMIT_ENV, '').strip()
    if not raw:
        return None
    try:
        return parse_size(raw)
    except ValueError:
        return None


def amp_bytes(precision: str = 'double') -> int:
    """Bytes per complex amplitude at an Aer/numpy precision."""
    return 8 if precision == 'single' else 16


def state_bytes(n: int, method: str = 'statevector',
                precision: str = 'double') -> int | None:
    """Raw size of the simulator state, or None when width does not set it."""
    b = amp_bytes(precision)
    if method in ('statevector', 'automatic'):
        return b << n
    if method in ('density_matrix', 'unitary'):
        return b << (2 * n)
    if method == 'superop':
        return b << (4 * n)
    return None


def aer_peak_bytes(n: int, method: str = 'statevector', precision: str = 'double',
                   capture: str = 'none') -> int | None:
    """Peak memory of one Aer run.

    capture: 'none' (counts only), 'end' (measure-free run that saves the
    final state) or 'mid' (save the state, then measure).
    """
    s = state_bytes(n, method, precision)
    if s is None:
        return None
    peak = int(s * AER_FACTOR)
    if method in ('statevector', 'automatic') and capture != 'none':
        if precision == 'single':
            peak += 16 << n                      # complex128 copy of a complex64 state
        elif capture == 'mid':
            peak += s
    return peak


def numpy_peak_bytes(n: int, precision: str = 'double', snapshot: bool = False) -> int:
    """Peak memory of the in-place numpy engines for an n-qubit state."""
    s = amp_bytes(precision) << n
    return s * (2 if snapshot else 1) + min(s, CHUNK_BYTES)


def _capped(measured: int | None) -> int | None:
    limit = memory_limit()
    if limit is None:
        return measured
    return limit if measured is None else min(measured, limit)


def available_bytes() -> int | None:
    """Memory available now (the OS figure, capped by QUBASIC_MEMORY_LIMIT),
    or None without psutil and without a limit."""
    try:
        import psutil
        measured: int | None = int(psutil.virtual_memory().available)
    except Exception:
        measured = None
    return _capped(measured)


def total_bytes() -> int | None:
    """Installed memory, capped by QUBASIC_MEMORY_LIMIT."""
    try:
        import psutil
        measured: int | None = int(psutil.virtual_memory().total)
    except Exception:
        measured = None
    return _capped(measured)


def ram_gb() -> tuple[float, float] | None:
    """(total, available) memory in GB as runs see it (capped by
    QUBASIC_MEMORY_LIMIT), or None when unknown."""
    total, avail = total_bytes(), available_bytes()
    if total is None or avail is None:
        return None
    return total / GB, avail / GB


def fits(peak: int | None, available: int | None = None,
         headroom: float = HEADROOM) -> bool:
    """Whether a run with this peak fits in available memory.

    A peak of None (structure-bound methods) and an unknown amount of
    available memory both count as fitting; the simulator then decides.
    """
    if peak is None:
        return True
    avail = available_bytes() if available is None else available
    if avail is None:
        return True
    return peak <= avail * headroom


def max_qubits(method: str = 'statevector', precision: str = 'double',
               capture: str = 'none', available: int | None = None,
               limit: int = 64) -> int | None:
    """Widest register whose run fits in available memory (None if unknown)."""
    avail = available_bytes() if available is None else available
    if avail is None:
        return None
    best = 0
    for n in range(1, limit + 1):
        if fits(aer_peak_bytes(n, method, precision, capture), avail):
            best = n
        else:
            break
    return best


def max_numpy_qubits(precision: str = 'double', snapshot: bool = False,
                     available: int | None = None, limit: int = 64) -> int | None:
    avail = available_bytes() if available is None else available
    if avail is None:
        return None
    best = 0
    for n in range(1, limit + 1):
        if fits(numpy_peak_bytes(n, precision, snapshot), avail):
            best = n
        else:
            break
    return best


def fmt_gb(b: float | int | None) -> str:
    """Decimal gigabytes, matching the rest of the terminal's output."""
    if b is None:
        return '?'
    g = b / GB
    if g >= 100:
        return f"{g:,.0f} GB"
    if g >= 0.01:
        return f"{g:.2f} GB"
    return f"{b / 1e6:.1f} MB"


# ---------------------------------------------------------------------------
# Gate matrices, builders, and numpy simulation primitives.
# ---------------------------------------------------------------------------

# Amplitudes per block for chunked in-place updates (64 MiB at complex128).
# States at or below this size are updated in one block.
_CHUNK_AMPS = 1 << 22

# ═══════════════════════════════════════════════════════════════════════
# Gate tables: name -> (num_params, num_qubits)
# ═══════════════════════════════════════════════════════════════════════


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

def _mat_crx(p: Sequence[float]) -> np.ndarray:
    t = p[0]
    c, s = np.cos(t/2), np.sin(t/2)
    m = np.eye(4, dtype=complex)
    m[2,2], m[2,3] = c, -1j*s
    m[3,2], m[3,3] = -1j*s, c
    return m

def _mat_cry(p: Sequence[float]) -> np.ndarray:
    t = p[0]
    c, s = np.cos(t/2), np.sin(t/2)
    m = np.eye(4, dtype=complex)
    m[2,2], m[2,3] = c, -s
    m[3,2], m[3,3] = s, c
    return m

def _mat_crz(p: Sequence[float]) -> np.ndarray:
    t = p[0]
    m = np.eye(4, dtype=complex)
    m[2,2], m[3,3] = np.exp(-1j*t/2), np.exp(1j*t/2)
    return m

def _mat_rxx(p: Sequence[float]) -> np.ndarray:
    t = p[0]
    ct, st = np.cos(t/2), np.sin(t/2)
    m = np.zeros((4,4), dtype=complex)
    m[0,0] = m[1,1] = m[2,2] = m[3,3] = ct
    m[0,3] = m[3,0] = m[1,2] = m[2,1] = -1j*st
    return m

def _mat_ryy(p: Sequence[float]) -> np.ndarray:
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
            keep: tuple[int, ...] = (), max_amps: int | None = None) -> Iterator[np.ndarray]:
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


# ---------------------------------------------------------------------------
# QUBASIC state analysis on statevectors without dense 2^n x 2^n matrices.
#
# Reduced density matrices are accumulated block by block from views of the
# state, so memory stays at the reduced matrix plus one block. Qubit q is bit
# q of the amplitude index (qubit 0 least significant).
# ---------------------------------------------------------------------------

def _flat(sv: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(sv)).reshape(-1)


def n_qubits_of(sv: np.ndarray) -> int:
    size = int(np.asarray(sv).size)
    n = size.bit_length() - 1
    if 1 << n != size:
        raise ValueError(f"state length {size} is not a power of two")
    return n


def subsystem_rdm(sv: np.ndarray, qubits: list[int], n: int | None = None) -> np.ndarray:
    """Reduced density matrix of `qubits` (traces out the rest).

    The row index is MSB-first over `qubits` (qubits[0] is its most
    significant bit). Size 2^k x 2^k for k = len(qubits).
    """
    flat = _flat(sv)
    n = n_qubits_of(flat) if n is None else n
    k = len(qubits)
    if len(set(qubits)) != k:
        raise ValueError("repeated qubit in subsystem")
    for q in qubits:
        if q < 0 or q >= n:
            raise ValueError(f"qubit {q} out of range (0-{n - 1})")
    psi = flat.reshape([2] * n)
    axes = [n - 1 - q for q in qubits]
    rho = np.zeros((1 << k, 1 << k), dtype=complex)
    for block in _blocks(psi, n, keep=tuple(axes)):
        m = np.moveaxis(block, axes, list(range(k))).reshape(1 << k, -1)
        rho += m @ m.conj().T
    return rho


def qubit_rdm(sv: np.ndarray, q: int, n: int | None = None) -> np.ndarray:
    """2x2 reduced density matrix of one qubit."""
    return subsystem_rdm(sv, [q], n)


def bloch(sv: np.ndarray, q: int, n: int | None = None) -> tuple[float, float, float]:
    """Bloch vector (x, y, z) of qubit q: <X> = 2 Re rho10, <Y> = 2 Im rho10."""
    r = qubit_rdm(sv, q, n)
    return float(2 * r[1, 0].real), float(2 * r[1, 0].imag), float((r[0, 0] - r[1, 1]).real)


def entropy_bits(eigs: np.ndarray) -> float:
    """Von Neumann entropy (bits) from density-matrix eigenvalues."""
    lam = np.clip(np.real(np.asarray(eigs, dtype=complex)), 0.0, None)
    lam = lam[lam > 1e-14]
    if lam.size == 0:
        return 0.0
    return max(0.0, float(-np.sum(lam * np.log2(lam)))) + 0.0      # never -0.0


# Above this many qubits on the smaller side of a cut, the reduced density
# matrix (4^k complex numbers) is refused rather than allocated.
MAX_CUT_QUBITS = 13


def schmidt_probs(sv: np.ndarray, part_a: list[int], n: int | None = None) -> np.ndarray:
    """Squared Schmidt coefficients of the pure state across A | rest."""
    flat = _flat(sv)
    n = n_qubits_of(flat) if n is None else n
    a = sorted(set(part_a))
    b = [q for q in range(n) if q not in a]
    side = a if len(a) <= len(b) else b
    if not side:
        return np.array([1.0])
    if len(side) > MAX_CUT_QUBITS:
        raise ValueError(
            f"cut needs a {len(side)}-qubit reduced state (4^{len(side)} entries); "
            f"choose a part with at most {MAX_CUT_QUBITS} qubits on one side")
    lam = np.linalg.eigvalsh(subsystem_rdm(flat, side, n))
    return np.clip(lam.real, 0.0, None)


def entanglement_entropy(sv: np.ndarray, part_a: list[int], n: int | None = None) -> float:
    return entropy_bits(schmidt_probs(sv, part_a, n))


def negativity_pure(probs: np.ndarray) -> tuple[float, float]:
    """(negativity, log2 negativity) of a pure state from Schmidt probabilities:
    N = ((sum sqrt(p))^2 - 1) / 2, E_N = 2 log2(sum sqrt(p))."""
    s = float(np.sum(np.sqrt(np.clip(probs, 0.0, None))))
    return (s * s - 1.0) / 2.0, 2.0 * np.log2(s) if s > 0 else 0.0


def negativity_mixed(rho: np.ndarray, part_a: list[int], n: int) -> tuple[float, float]:
    """(negativity, log2 negativity) of a density matrix across A | rest by
    partial transpose of A (row index little-endian: qubit q = bit q)."""
    t = rho.reshape([2] * n + [2] * n)
    perm = list(range(2 * n))
    for q in part_a:
        ax = n - 1 - q
        perm[ax], perm[ax + n] = perm[ax + n], perm[ax]
    rt = np.transpose(t, perm).reshape(1 << n, 1 << n)
    tr = float(np.sum(np.abs(np.linalg.eigvalsh(0.5 * (rt + rt.conj().T)))))
    return (tr - 1.0) / 2.0, float(np.log2(tr)) if tr > 0 else 0.0


_YY = np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]]))


def concurrence(rho4: np.ndarray) -> float:
    """Wootters concurrence of a two-qubit density matrix."""
    rt = _YY @ rho4.conj() @ _YY
    ev = np.sqrt(np.clip(np.sort(np.real(np.linalg.eigvals(rho4 @ rt)))[::-1], 0.0, None))
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


def eof_from_concurrence(c: float) -> float:
    """Entanglement of formation (ebits) from concurrence."""
    if c <= 0:
        return 0.0
    x = (1.0 + np.sqrt(max(0.0, 1.0 - c * c))) / 2.0
    if x >= 1.0:
        return 0.0
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


def mutual_information(sv: np.ndarray, a: int, b: int, n: int | None = None) -> float:
    """I(a:b) = S(a) + S(b) - S(ab) in bits (0 to 2)."""
    rho_ab = subsystem_rdm(sv, [a, b], n)
    ra = np.array([[rho_ab[0, 0] + rho_ab[1, 1], rho_ab[0, 2] + rho_ab[1, 3]],
                   [rho_ab[2, 0] + rho_ab[3, 1], rho_ab[2, 2] + rho_ab[3, 3]]])
    rb = np.array([[rho_ab[0, 0] + rho_ab[2, 2], rho_ab[0, 1] + rho_ab[2, 3]],
                   [rho_ab[1, 0] + rho_ab[3, 2], rho_ab[1, 1] + rho_ab[3, 3]]])
    return (entropy_bits(np.linalg.eigvalsh(ra)) + entropy_bits(np.linalg.eigvalsh(rb))
            - entropy_bits(np.linalg.eigvalsh(rho_ab)))


def _parity(idx: np.ndarray, mask: int) -> np.ndarray:
    v = idx & mask
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(v) & 1
    out = np.zeros_like(v)
    m = mask
    b = 0
    while m:
        if m & 1:
            out ^= (v >> b) & 1
        m >>= 1
        b += 1
    return out


def z_expectation(sv: np.ndarray, qubits: list[int]) -> float:
    """<Z_q1 Z_q2 ...> computed chunk by chunk."""
    flat = _flat(sv)
    mask = 0
    for q in qubits:
        mask ^= 1 << q
    total = 0.0
    step = _CHUNK_AMPS
    for s in range(0, flat.size, step):
        chunk = flat[s:s + step]
        p = chunk.real ** 2 + chunk.imag ** 2
        sign = 1.0 - 2.0 * _parity(np.arange(s, s + chunk.size, dtype=np.int64), mask)
        total += float(np.dot(p, sign))
    return total


def norm2(sv: np.ndarray) -> float:
    flat = _flat(sv)
    return float(np.vdot(flat, flat).real)


def nonzero_indices(sv: np.ndarray, threshold: float, limit: int,
                    count: bool = True) -> tuple[np.ndarray, int]:
    """First `limit` indices (ascending) with |amplitude| > threshold, and
    the total number of such indices (-1 when count is False)."""
    flat = _flat(sv)
    thr2 = threshold * threshold
    found: list[np.ndarray] = []
    have = 0
    total = 0
    step = _CHUNK_AMPS
    for s in range(0, flat.size, step):
        chunk = flat[s:s + step]
        mask = (chunk.real ** 2 + chunk.imag ** 2) > thr2
        if count:
            total += int(np.count_nonzero(mask))
        if have < limit:
            idx = np.flatnonzero(mask)[:limit - have] + s
            found.append(idx)
            have += idx.size
        elif not count:
            break
    out = np.concatenate(found) if found else np.zeros(0, dtype=np.int64)
    return out, (total if count else -1)


def _select_top(idx: np.ndarray, p: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k largest p (ties broken toward the lower index)."""
    if p.size <= k:
        return idx, p
    kth = np.partition(p, -k)[-k]
    above = np.flatnonzero(p > kth)
    tied = np.flatnonzero(p == kth)
    tied = tied[np.argsort(idx[tied], kind='stable')][:k - above.size]
    sel = np.concatenate([above, tied])
    return idx[sel], p[sel]


def top_k(sv: np.ndarray, k: int, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray, int]:
    """(indices, probabilities) of the k most probable outcomes, most probable
    first (ties by index), and the number of outcomes above threshold^2."""
    flat = _flat(sv)
    thr2 = threshold * threshold
    best_i = np.zeros(0, dtype=np.int64)
    best_p = np.zeros(0)
    total = 0
    step = _CHUNK_AMPS
    for s in range(0, flat.size, step):
        chunk = flat[s:s + step]
        p = chunk.real ** 2 + chunk.imag ** 2
        total += int(np.count_nonzero(p > thr2))
        ci, cp = _select_top(np.arange(s, s + p.size, dtype=np.int64), p, k)
        best_i, best_p = _select_top(np.concatenate([best_i, ci]),
                                     np.concatenate([best_p, cp]), k)
    order = np.lexsort((best_i, -best_p))
    best_i, best_p = best_i[order], best_p[order]
    live = best_p > thr2
    return best_i[live], best_p[live], total


# ---------------------------------------------------------------------------
# Device topologies as coupling-map edge lists.
# ---------------------------------------------------------------------------

def heavy_hex_edges(n: int) -> list[list[int]]:
    """Undirected edges of an n-qubit connected piece of IBM's heavy-hex
    lattice: the smallest odd-distance heavy-hex map with at least n qubits,
    cut to the first n nodes in breadth-first order and relabelled 0..n-1."""
    from collections import deque

    from qiskit.transpiler import CouplingMap
    d = 3
    cm = CouplingMap.from_heavy_hex(d)
    while cm.size() < n:
        d += 2
        cm = CouplingMap.from_heavy_hex(d)
    adj: dict[int, set[int]] = {}
    for a, b in cm.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    order: list[int] = []
    seen = {0}
    dq = deque([0])
    while dq and len(order) < n:
        u = dq.popleft()
        order.append(u)
        for v in sorted(adj.get(u, ())):
            if v not in seen:
                seen.add(v)
                dq.append(v)
    label = {old: i for i, old in enumerate(order)}
    edges = set()
    for a, b in cm.get_edges():
        if a in label and b in label:
            edges.add(tuple(sorted((label[a], label[b]))))
    return [list(e) for e in sorted(edges)]


# ---------------------------------------------------------------------------
# QUBASIC backend abstraction — Qiskit and numpy behind a common interface.
# ---------------------------------------------------------------------------

class QiskitBackend:
    """Wraps a QuantumCircuit for the circuit-build path."""

    def __init__(self, qc: Any, apply_gate_fn: Any) -> None:
        self._qc = qc
        self._apply_gate = apply_gate_fn

    def apply_gate(self, gate_name: str, params: tuple[float, ...],
                   qubits: list[int]) -> None:
        self._apply_gate(self._qc, gate_name, list(params), qubits)

    def barrier(self) -> None:
        self._qc.barrier()

    def reset(self, qubit: int) -> None:
        self._qc.reset(qubit)

    def measure(self, qubit: int, classical_bit: Any) -> None:
        self._qc.measure(qubit, classical_bit)

    def add_classical_register(self, name: str, size: int = 1) -> Any:
        from qiskit.circuit import ClassicalRegister
        cr = ClassicalRegister(size, name)
        self._qc.add_register(cr)
        return cr

    def h(self, qubit: int) -> None:
        self._qc.h(qubit)

    def append_inverse(self, sub_qc: Any, qubits: list[int]) -> None:
        self._qc.append(sub_qc.inverse(), qubits)

    def append_controlled(self, gate: Any, qubits: list[int]) -> None:
        self._qc.append(gate, qubits)

    @property
    def qc(self) -> Any:
        return self._qc


class LOCCRegBackend:
    """Wraps a LOCCEngine register for the numpy path."""

    def __init__(self, engine: Any, reg: str) -> None:
        self._engine = engine
        self._reg = reg

    def apply_gate(self, gate_name: str, params: tuple[float, ...],
                   qubits: list[int]) -> None:
        self._engine.apply(self._reg, gate_name, params, qubits)

    def barrier(self) -> None:
        pass

    def reset(self, qubit: int) -> None:
        outcome = self._engine.send(self._reg, qubit)
        if outcome == 1:
            self._engine.apply(self._reg, 'X', (), [qubit])
