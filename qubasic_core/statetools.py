"""QUBASIC state analysis on statevectors without dense 2^n x 2^n matrices.

Reduced density matrices are accumulated block by block from views of the
state, so memory stays at the reduced matrix plus one block. Qubit q is bit
q of the amplitude index (qubit 0 least significant).
"""

from __future__ import annotations

import numpy as np

from qubasic_core import gates as _gates


def _flat(sv) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(sv)).reshape(-1)


def n_qubits_of(sv) -> int:
    size = int(np.asarray(sv).size)
    n = size.bit_length() - 1
    if 1 << n != size:
        raise ValueError(f"state length {size} is not a power of two")
    return n


def subsystem_rdm(sv, qubits: list[int], n: int | None = None) -> np.ndarray:
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
    for block in _gates._blocks(psi, n, keep=tuple(axes)):
        m = np.moveaxis(block, axes, list(range(k))).reshape(1 << k, -1)
        rho += m @ m.conj().T
    return rho


def qubit_rdm(sv, q: int, n: int | None = None) -> np.ndarray:
    """2x2 reduced density matrix of one qubit."""
    return subsystem_rdm(sv, [q], n)


def bloch(sv, q: int, n: int | None = None) -> tuple[float, float, float]:
    """Bloch vector (x, y, z) of qubit q: <X> = 2 Re rho10, <Y> = 2 Im rho10."""
    r = qubit_rdm(sv, q, n)
    return float(2 * r[1, 0].real), float(2 * r[1, 0].imag), float((r[0, 0] - r[1, 1]).real)


def entropy_bits(eigs) -> float:
    """Von Neumann entropy (bits) from density-matrix eigenvalues."""
    lam = np.clip(np.real(np.asarray(eigs, dtype=complex)), 0.0, None)
    lam = lam[lam > 1e-14]
    if lam.size == 0:
        return 0.0
    return float(-np.sum(lam * np.log2(lam)))


# Above this many qubits on the smaller side of a cut, the reduced density
# matrix (4^k complex numbers) is refused rather than allocated.
MAX_CUT_QUBITS = 13


def schmidt_probs(sv, part_a: list[int], n: int | None = None) -> np.ndarray:
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


def entanglement_entropy(sv, part_a: list[int], n: int | None = None) -> float:
    return entropy_bits(schmidt_probs(sv, part_a, n))


def negativity_pure(probs) -> tuple[float, float]:
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


def mutual_information(sv, a: int, b: int, n: int | None = None) -> float:
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


def z_expectation(sv, qubits: list[int]) -> float:
    """<Z_q1 Z_q2 ...> computed chunk by chunk."""
    flat = _flat(sv)
    mask = 0
    for q in qubits:
        mask ^= 1 << q
    total = 0.0
    step = _gates._CHUNK_AMPS
    for s in range(0, flat.size, step):
        chunk = flat[s:s + step]
        p = chunk.real ** 2 + chunk.imag ** 2
        sign = 1.0 - 2.0 * _parity(np.arange(s, s + chunk.size, dtype=np.int64), mask)
        total += float(np.dot(p, sign))
    return total


def norm2(sv) -> float:
    flat = _flat(sv)
    return float(np.vdot(flat, flat).real)


def nonzero_indices(sv, threshold: float, limit: int,
                    count: bool = True) -> tuple[np.ndarray, int]:
    """First `limit` indices (ascending) with |amplitude| > threshold, and
    the total number of such indices (-1 when count is False)."""
    flat = _flat(sv)
    thr2 = threshold * threshold
    found: list[np.ndarray] = []
    have = 0
    total = 0
    step = _gates._CHUNK_AMPS
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


def top_k(sv, k: int, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray, int]:
    """(indices, probabilities) of the k most probable outcomes, most probable
    first (ties by index), and the number of outcomes above threshold^2."""
    flat = _flat(sv)
    thr2 = threshold * threshold
    best_i = np.zeros(0, dtype=np.int64)
    best_p = np.zeros(0)
    total = 0
    step = _gates._CHUNK_AMPS
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
