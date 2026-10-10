"""QUBASIC capacity model — what a simulation costs and what fits in memory.

Peak resident memory of Aer statevector runs, measured with Qiskit 2.3 and
Aer 0.17, relative to the raw state S (2^n amplitudes at the run precision):

    measured run, counts only                      1.04 S
    measure-free run that saves the final state    1.04 S  (Aer hands the
                                                            buffer over)
    run that saves the state and then measures     2.04 S  (Aer copies it)

A saved state always comes back as complex128, so at single precision a
saved state adds 2 S. The numpy engines (LOCC, live measurement, STEP)
update the state in place in chunks of CHUNK_BYTES, so they need S plus one
chunk, and a second S while a prefix snapshot is held.
"""

from __future__ import annotations

CHUNK_BYTES = 1 << 26        # scratch for chunked in-place numpy updates (64 MiB)
AER_FACTOR = 1.04
HEADROOM = 0.85              # fraction of available memory one run may claim

# Methods whose memory is set by register width; the others (stabilizer,
# matrix_product_state, extended_stabilizer) scale with structure instead.
WIDTH_BOUND = ('statevector', 'automatic', 'density_matrix', 'unitary', 'superop')

GB = 1e9


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


def available_bytes() -> int | None:
    """Memory the OS reports as available now, or None without psutil."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def total_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


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
