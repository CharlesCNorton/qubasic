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
# QUBASIC stabilizer tableau: Clifford circuits at widths no statevector
# reaches.
#
# A stabilizer state on n qubits is held as Aaronson and Gottesman's tableau
# (Phys. Rev. A 70, 052328): n destabilizer and n stabilizer generators, each
# a Hermitian Pauli with a sign. The generators' x and z bits are packed 64 to
# a word along the generator axis, word-major, so a gate rewrites a few
# contiguous rows of words and a measurement multiplies generators a word at
# a time. A gate outside the named fast paths goes through its table of how
# it conjugates each Pauli on its qubits, computed once per matrix, which is
# also what decides whether a matrix is a Clifford at all.
# ---------------------------------------------------------------------------

_ONE = np.uint64(1)
_POPCOUNT8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)
_PAULI_MATS = (_MAT_ID, _MAT_X, _MAT_Z, _MAT_Y)          # by code x + 2z
_CLIFFORD_TABLES: dict[bytes, tuple[np.ndarray, np.ndarray] | None] = {}


def _popcount(words: np.ndarray) -> np.ndarray:
    """Set bits in each element of a uint64 array."""
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(words).astype(np.int64)
    w = np.ascontiguousarray(words)
    return _POPCOUNT8[w.view(np.uint8)].reshape(w.shape + (8,)).sum(axis=-1)


def tableau_bytes(n: int) -> int:
    """Peak memory of a live run on an n-qubit tableau: the state, the start
    snapshot, the first shot's copy and a measurement's scratch generators."""
    return 6 * (2 * ((n + 63) >> 6) * 2 * n * 8) + 64 * n


def pauli_code_matrix(code: int, k: int) -> np.ndarray:
    """The k-qubit Pauli whose bits 2j and 2j+1 are qubit j's x and z, in
    Qiskit order (qubit 0 least significant)."""
    m = np.ones((1, 1), dtype=complex)
    for j in range(k):
        m = np.kron(_PAULI_MATS[(code >> (2 * j)) & 3], m)
    return m


def clifford_table(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """How a unitary on k <= 4 qubits (Qiskit order) conjugates each Pauli on
    its qubits: image codes and sign bits indexed by Pauli code, or None when
    the unitary is not a Clifford."""
    u = np.ascontiguousarray(matrix, dtype=complex)
    key = u.tobytes()
    if key in _CLIFFORD_TABLES:
        return _CLIFFORD_TABLES[key]
    d = u.shape[0]
    k = d.bit_length() - 1
    table = None
    if 1 <= k <= 4 and d == 1 << k and u.shape == (d, d):
        paulis = np.array([pauli_code_matrix(c, k) for c in range(4 ** k)])
        images = u @ paulis @ u.conj().T
        coeff = np.einsum('qij,pji->pq', paulis, images) / d
        image = np.argmax(np.abs(coeff), axis=1)
        value = coeff[np.arange(image.size), image]
        if np.allclose(value, np.sign(value.real), atol=1e-6):
            table = (image.astype(np.int64), (value.real < 0).astype(np.uint64))
    if len(_CLIFFORD_TABLES) < 4096:
        _CLIFFORD_TABLES[key] = table
    return table


def pauli_mixture(ops: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """A channel given by Kraus operators on k <= 3 qubits as a mixture of
    Paulis, (cumulative probabilities, Pauli codes), when its chi matrix in
    the Pauli basis is diagonal; None for any other channel."""
    k = ops[0].shape[0].bit_length() - 1
    if k > 3:
        return None
    paulis = np.array([pauli_code_matrix(c, k) for c in range(4 ** k)])
    coeff = np.einsum('aij,kji->ka', paulis, np.asarray(ops, dtype=complex)) / (1 << k)
    chi = coeff.T @ coeff.conj()
    probs = np.real(np.diag(chi)).copy()
    if not np.allclose(chi - np.diag(np.diag(chi)), 0.0, atol=1e-9):
        return None
    codes = np.flatnonzero(probs > 1e-15)
    weights = probs[codes] / probs[codes].sum()
    return np.cumsum(weights), codes


class Tableau:
    """A stabilizer state on n qubits. Generator rows 0..n-1 are the
    destabilizers and n..2n-1 the stabilizers; x and z are (words, 2n) uint64
    arrays holding qubit q at bit q & 63 of word q >> 6, and r the signs. A
    generator is (-1)^r times the tensor product of I, X, Z or Y (x = z = 1)
    on each qubit."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.words = (n + 63) >> 6
        self.x = np.zeros((self.words, 2 * n), dtype=np.uint64)
        self.z = np.zeros((self.words, 2 * n), dtype=np.uint64)
        self.r = np.zeros(2 * n, dtype=np.uint64)
        q = np.arange(n)
        bits = np.left_shift(_ONE, (q & 63).astype(np.uint64))
        self.x[q >> 6, q] = bits
        self.z[q >> 6, q + n] = bits

    def copy(self) -> 'Tableau':
        t = Tableau.__new__(Tableau)
        t.n, t.words = self.n, self.words
        t.x, t.z, t.r = self.x.copy(), self.z.copy(), self.r.copy()
        return t

    def assign(self, other: 'Tableau') -> None:
        np.copyto(self.x, other.x)
        np.copyto(self.z, other.z)
        np.copyto(self.r, other.r)

    def _bits(self, a: int) -> tuple[np.ndarray, np.ndarray, np.uint64, np.ndarray, np.ndarray]:
        """Qubit a's word rows (views) and shift, and its x and z bits over
        every generator (copies)."""
        xw, zw = self.x[a >> 6], self.z[a >> 6]
        s = np.uint64(a & 63)
        return xw, zw, s, (xw >> s) & _ONE, (zw >> s) & _ONE

    # Gates: each conjugates every generator.

    def h(self, a: int) -> None:
        xw, zw, s, xa, za = self._bits(a)
        self.r ^= xa & za
        xa ^= za
        xa <<= s
        xw ^= xa
        zw ^= xa

    def s(self, a: int) -> None:
        xw, zw, s, xa, za = self._bits(a)
        self.r ^= xa & za
        xa <<= s
        zw ^= xa

    def sdg(self, a: int) -> None:
        xw, zw, s, xa, za = self._bits(a)
        za ^= _ONE
        self.r ^= xa & za
        xa <<= s
        zw ^= xa

    def pauli_x(self, a: int) -> None:
        self.r ^= (self.z[a >> 6] >> np.uint64(a & 63)) & _ONE

    def pauli_z(self, a: int) -> None:
        self.r ^= (self.x[a >> 6] >> np.uint64(a & 63)) & _ONE

    def pauli_y(self, a: int) -> None:
        w, s = a >> 6, np.uint64(a & 63)
        self.r ^= ((self.x[w] ^ self.z[w]) >> s) & _ONE

    def cx(self, a: int, b: int) -> None:
        _, zwa, sa, xa, za = self._bits(a)
        xwb, _, sb, xb, zb = self._bits(b)
        t = xb ^ za
        t ^= _ONE
        t &= xa
        t &= zb
        self.r ^= t
        xa <<= sb
        xwb ^= xa
        zb <<= sa
        zwa ^= zb

    def cz(self, a: int, b: int) -> None:
        _, zwa, sa, xa, za = self._bits(a)
        _, zwb, sb, xb, zb = self._bits(b)
        t = za ^ zb
        t &= xa
        t &= xb
        self.r ^= t
        zwa ^= xb << sa
        zwb ^= xa << sb

    def swap(self, a: int, b: int) -> None:
        xwa, zwa, sa, xa, za = self._bits(a)
        xwb, zwb, sb, xb, zb = self._bits(b)
        xa ^= xb
        za ^= zb
        xwa ^= xa << sa
        xwb ^= xa << sb
        zwa ^= za << sa
        zwb ^= za << sb

    def apply_table(self, table: tuple[np.ndarray, np.ndarray], qubits: Sequence[int]) -> None:
        """A Clifford given by clifford_table, its qubit j acting on qubits[j]."""
        image, sign = table
        cols = [self._bits(q) for q in qubits]
        code = np.zeros(2 * self.n, dtype=np.int64)
        for j, (_, _, _, xq, zq) in enumerate(cols):
            code |= (xq.astype(np.int64) << (2 * j)) | (zq.astype(np.int64) << (2 * j + 1))
        new = image[code].astype(np.uint64)
        self.r ^= sign[code]
        for j, (xw, zw, s, xq, zq) in enumerate(cols):
            xw ^= (xq ^ ((new >> np.uint64(2 * j)) & _ONE)) << s
            zw ^= (zq ^ ((new >> np.uint64(2 * j + 1)) & _ONE)) << s

    def apply_pauli(self, code: int, qubits: Sequence[int]) -> None:
        """The Pauli with this code (as in clifford_table) on `qubits`."""
        for j, q in enumerate(qubits):
            c = (code >> (2 * j)) & 3
            if c == 1:
                self.pauli_x(q)
            elif c == 2:
                self.pauli_z(q)
            elif c == 3:
                self.pauli_y(q)

    # Generator products. A product of Hermitian Paulis P1 P2 picks up i to
    # the power |x1&z1| + |x2&z2| + 2|z1&x2| - |x3&z3|, where x3 and z3 are
    # the product's bits; commuting factors keep it real.

    def _left_multiply(self, rows: np.ndarray, p: int) -> None:
        """Generator p times each of `rows`, into those rows."""
        xp, zp = self.x[:, p:p + 1], self.z[:, p:p + 1]
        xh, zh = self.x[:, rows], self.z[:, rows]
        nx, nz = xh ^ xp, zh ^ zp
        e = (int(_popcount(xp & zp).sum()) + _popcount(xh & zh).sum(axis=0)
             + 2 * _popcount(xh & zp).sum(axis=0) - _popcount(nx & nz).sum(axis=0)
             + 2 * int(self.r[p]) + 2 * self.r[rows].astype(np.int64))
        self.x[:, rows] = nx
        self.z[:, rows] = nz
        self.r[rows] = ((e % 4) == 2).astype(np.uint64)

    def _product_sign(self, rows: np.ndarray) -> int:
        """The sign bit of the product of commuting generators: the phase
        exponents of the running products telescope, leaving each factor's
        own count, twice its overlap with the product of the factors before
        it, and the final product's count."""
        if rows.size == 0:
            return 0
        x, z = self.x[:, rows], self.z[:, rows]
        acc = np.bitwise_xor.accumulate(x, axis=1)
        before = np.zeros_like(x)
        before[:, 1:] = acc[:, :-1]
        xq, zq = acc[:, -1], np.bitwise_xor.accumulate(z, axis=1)[:, -1]
        e = (int(_popcount(x & z).sum()) + 2 * int(_popcount(z & before).sum())
             - int(_popcount(xq & zq).sum()) + 2 * int(np.count_nonzero(self.r[rows])))
        return int((e % 4) == 2)

    # Measurement

    def is_random(self, a: int) -> bool:
        """Whether measuring qubit a has a random outcome."""
        return bool(((self.x[a >> 6, self.n:] >> np.uint64(a & 63)) & _ONE).any())

    def measure(self, a: int, force: int | None = None) -> int:
        """Measure Z on qubit a and collapse. A random outcome is drawn from
        numpy's global generator (so SEED reproduces runs), or is `force`."""
        n = self.n
        w, s = a >> 6, np.uint64(a & 63)
        xa = (self.x[w] >> s) & _ONE
        hits = np.flatnonzero(xa[n:])
        if hits.size:
            p = n + int(hits[0])
            rows = np.flatnonzero(xa)
            rows = rows[(rows != p) & (rows != p - n)]
            if rows.size:
                self._left_multiply(rows, p)
            self.x[:, p - n] = self.x[:, p]
            self.z[:, p - n] = self.z[:, p]
            self.r[p - n] = self.r[p]
            self.x[:, p] = 0
            self.z[:, p] = 0
            self.z[w, p] = _ONE << s
            bit = int(force) if force is not None else int(np.random.random() < 0.5)
            self.r[p] = bit
            return bit
        return self._product_sign(np.flatnonzero(xa[:n]) + n)

    def reset(self, a: int) -> None:
        if self.measure(a):
            self.pauli_x(a)

    # Reading the state

    def _pack(self, bits: np.ndarray) -> np.ndarray:
        words = np.zeros(self.words, dtype=np.uint64)
        idx = np.flatnonzero(bits)
        np.bitwise_or.at(words, idx >> 6, np.left_shift(_ONE, (idx & 63).astype(np.uint64)))
        return words

    def expectation(self, px: np.ndarray, pz: np.ndarray) -> int:
        """<P> for the Hermitian Pauli with per-qubit bits px and pz: 0 when
        it anticommutes with a stabilizer, otherwise the sign with which the
        stabilizer group contains it."""
        n = self.n
        wx, wz = self._pack(px)[:, None], self._pack(pz)[:, None]
        anti = (_popcount(self.z & wx).sum(axis=0) + _popcount(self.x & wz).sum(axis=0)) & 1
        if anti[n:].any():
            return 0
        return -1 if self._product_sign(np.flatnonzero(anti[:n]) + n) else 1

    def bloch(self, q: int) -> tuple[float, float, float]:
        """Qubit q's Bloch vector (<X>, <Y>, <Z>), each 0 or +-1."""
        if not 0 <= q < self.n:
            raise ValueError(f"QUBIT {q} OUT OF RANGE (0-{self.n - 1})")
        one = np.zeros(self.n, dtype=bool)
        one[q] = True
        none = np.zeros(self.n, dtype=bool)
        return (float(self.expectation(one, none)), float(self.expectation(one, one)),
                float(self.expectation(none, one)))

    def probabilities(self, qubits: Sequence[int]) -> np.ndarray:
        """Outcome probabilities over `qubits` (index bit j for qubits[j]),
        branching on copies only where an outcome is random."""
        out = np.zeros(1 << len(qubits))

        def walk(t: Tableau, j: int, index: int, p: float) -> None:
            if j == len(qubits):
                out[index] += p
                return
            q = qubits[j]
            if t.is_random(q):
                other = t.copy()
                walk(other, j + 1, index | (other.measure(q, force=1) << j), p / 2)
                walk(t, j + 1, index | (t.measure(q, force=0) << j), p / 2)
            else:
                walk(t, j + 1, index | (t.measure(q) << j), p)

        walk(self.copy(), 0, 0, 1.0)
        return out

    def generators(self) -> list[str]:
        """The stabilizer generators as signed Pauli strings, qubit 0 first
        and _ for the identity."""
        n = self.n
        q = np.arange(n)
        shift = (q & 63).astype(np.uint64)[:, None]
        xb = ((self.x[q >> 6, n:] >> shift) & _ONE).astype(np.int64)
        zb = ((self.z[q >> 6, n:] >> shift) & _ONE).astype(np.int64)
        chars = np.array(list('_XZY'))[(xb + 2 * zb).T]
        return [('-' if self.r[n + g] else '+') + ''.join(chars[g]) for g in range(n)]


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
