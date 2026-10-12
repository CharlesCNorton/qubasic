"""Shor's algorithm with modular-arithmetic circuits (Beauregard, 2003).

Order finding for a mod N on 2n + 3 qubits, n = bit length of N:

    c        one control qubit, reused for every bit of the phase
    x        n qubits holding the exponentiation register, starting at |1>
    b        n + 1 qubits, a work register kept in the Fourier basis
    anc      one ancilla for the modular reduction

Addition of a classical constant happens in the Fourier basis (Draper's
adder: one phase rotation per qubit), modular addition a + b mod N is built
from it with one overflow ancilla, controlled modular multiplication
x -> a x mod N is n modular additions plus a controlled swap and the inverse
multiplication by a^-1, and phase estimation reads the order with a single
control qubit measured 2n times, each later rotation conditioned on the
bits already read (the semiclassical inverse Fourier transform).

The classical part checks the easy cases (even N, prime powers, a shared
factor) and turns a measured phase into a period by continued fractions.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Iterable, NamedTuple

import numpy as np


class Layout(NamedTuple):
    c: int
    x: list[int]
    b: list[int]
    anc: int
    total: int


def layout(n: int) -> Layout:
    """Qubit indices of the 2n + 3 qubit order-finding register."""
    return Layout(c=0, x=list(range(1, n + 1)), b=list(range(n + 1, 2 * n + 2)),
                  anc=2 * n + 2, total=2 * n + 3)


def qft(qc, qs: list[int], inverse: bool = False) -> None:
    """Fourier transform over qs (qs[0] the least significant bit), without
    the final bit reversal: afterwards qubit qs[l] carries the phase
    2 pi y 2^l / 2^m of basis value y."""
    m = len(qs)
    order = list(reversed(qs))           # most significant first
    if not inverse:
        for i in range(m):
            qc.h(order[i])
            for j in range(i + 1, m):
                qc.cp(np.pi / 2 ** (j - i), order[j], order[i])
    else:
        for i in reversed(range(m)):
            for j in reversed(range(i + 1, m)):
                qc.cp(-np.pi / 2 ** (j - i), order[j], order[i])
            qc.h(order[i])


def phi_add(qc, a: int, qs: list[int], controls: Iterable[int] = (), inverse: bool = False) -> None:
    """Add the constant a (mod 2^m) to a register held in the Fourier basis
    by `qft`: qubit qs[l] carries the phase 2 pi y / 2^(l+1), so adding a
    rotates it by 2 pi a / 2^(l+1); each rotation is controlled by every
    qubit in controls."""
    controls = list(controls)
    sign = -1.0 if inverse else 1.0
    for l, target in enumerate(qs):
        period = 1 << (l + 1)
        frac = (a % period) / period
        if frac == 0:
            continue
        angle = sign * 2 * np.pi * frac
        if controls:
            qc.mcp(angle, controls, target)
        else:
            qc.p(angle, target)


def phi_add_mod(qc, a: int, N: int, b: list[int], anc: int, controls: list[int],
                inverse: bool = False) -> None:
    """Doubly controlled (a + b) mod N on a Fourier-basis register b of n + 1
    qubits (b < N and a < N), the overflow ancilla anc returned to |0>."""
    msb = b[-1]
    if not inverse:
        phi_add(qc, a, b, controls)
        phi_add(qc, N, b, inverse=True)
        qft(qc, b, inverse=True)
        qc.cx(msb, anc)
        qft(qc, b)
        phi_add(qc, N, b, [anc])
        phi_add(qc, a, b, controls, inverse=True)
        qft(qc, b, inverse=True)
        qc.x(msb)
        qc.cx(msb, anc)
        qc.x(msb)
        qft(qc, b)
        phi_add(qc, a, b, controls)
    else:
        phi_add(qc, a, b, controls, inverse=True)
        qft(qc, b, inverse=True)
        qc.x(msb)
        qc.cx(msb, anc)
        qc.x(msb)
        qft(qc, b)
        phi_add(qc, a, b, controls)
        phi_add(qc, N, b, [anc], inverse=True)
        qft(qc, b, inverse=True)
        qc.cx(msb, anc)
        qft(qc, b)
        phi_add(qc, N, b)
        phi_add(qc, a, b, controls, inverse=True)


def cmult(qc, a: int, N: int, c: int, x: list[int], b: list[int], anc: int,
          inverse: bool = False) -> None:
    """Controlled |x>|b> -> |x>|(b + a x) mod N> (b in the computational basis)."""
    n = len(x)
    qft(qc, b)
    steps = [(i, (a << i) % N) for i in range(n)]
    for i, ai in (reversed(steps) if inverse else steps):
        phi_add_mod(qc, ai, N, b, anc, [c, x[i]], inverse=inverse)
    qft(qc, b, inverse=True)


def controlled_modmul(qc, a: int, N: int, c: int, x: list[int], b: list[int], anc: int) -> None:
    """Controlled x -> a x mod N in place (b returns to |0>); a must be a unit mod N."""
    cmult(qc, a, N, c, x, b, anc)
    for xi, bi in zip(x, b):
        qc.cswap(c, xi, bi)
    cmult(qc, pow(a, -1, N), N, c, x, b, anc, inverse=True)


def order_finding_circuit(a: int, N: int):
    """The semiclassical order-finding circuit for a mod N; its classical
    register `phase` holds the 2n phase bits, bit k the k-th read (the least
    significant bit of the estimate first)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ClassicalRegister, QuantumRegister
    n = N.bit_length()
    L = layout(n)
    t = 2 * n
    q = QuantumRegister(L.total, 'q')
    creg = ClassicalRegister(t, 'phase')
    qc = QuantumCircuit(q, creg)
    c, x, b, anc = L.c, L.x, L.b, L.anc
    qc.x(x[0])                                   # |x> = |1>
    for step in range(t):
        k = t - 1 - step                         # U^(2^k): least significant phase bit first
        qc.h(c)
        controlled_modmul(qc, pow(a, 1 << k, N), N, c, x, b, anc)
        for j in range(step):                    # bits read so far correct this one
            angle = -np.pi / (1 << (step - j))
            with qc.if_test((creg[j], 1)):
                qc.p(angle, c)
        qc.h(c)
        qc.measure(c, creg[step])
        qc.reset(c)
    return qc


def phase_value(bits: str, t: int) -> Fraction:
    """The phase estimate y / 2^t from a counts key of the `phase` register
    (rightmost character = bit 0, the first bit read = the least significant)."""
    y = 0
    for step in range(t):
        if bits[len(bits) - 1 - step] == '1':
            y |= 1 << step
    return Fraction(y, 1 << t)


def order_from_phase(phase: Fraction, a: int, N: int) -> int | None:
    """The order r of a mod N from a phase estimate s/r, by continued
    fractions (small multiples of the denominator are tried too)."""
    if phase == 0:
        return None
    r = phase.limit_denominator(N).denominator
    for mult in range(1, 8):
        cand = r * mult
        if cand > N:
            break
        if pow(a, cand, N) == 1:
            return cand
    return None


def factors_from_order(a: int, r: int, N: int) -> tuple[int, int] | None:
    if r % 2:
        return None
    y = pow(a, r // 2, N)
    if y == N - 1:
        return None
    for g in (math.gcd(y - 1, N), math.gcd(y + 1, N)):
        if 1 < g < N:
            return g, N // g
    return None


def classical_shortcut(N: int) -> tuple[str, tuple[int, int] | None]:
    """('even' | 'prime' | 'power' | '', factors) — the cases order finding
    does not need."""
    if N % 2 == 0:
        return 'even', (2, N // 2)
    if _is_prime(N):
        return 'prime', None
    for k in range(2, N.bit_length() + 1):
        root = round(N ** (1.0 / k))
        for cand in (root - 1, root, root + 1):
            if cand > 1 and cand ** k == N:
                return 'power', (cand, N // cand)
    return '', None


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True
