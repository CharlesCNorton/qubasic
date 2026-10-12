"""QUBASIC oracle algorithms: Grover search and Simon's algorithm.

    GROVER 101 [011 ...] [ITERS k] [SHOTS n] [PROGRAM]
    SIMON 110 [SHOTS n]

GROVER marks the given basis states (typed in the displayed bit order) with
a phase-flip oracle and runs the optimal number of Grover iterations,
floor(pi / (4 theta)) for sin^2(theta) = marked / 2^n, unless ITERS is
given; the report compares the measured success probability with
sin^2((2k + 1) theta).

SIMON hides a nonzero secret s in a two-to-one function f(x) = f(x XOR s)
(the textbook oracle: copy x, then XOR s in controlled by one bit where s is
1), runs Simon's circuit (Hadamards, oracle, Hadamards, measure x) and
solves the measured equations y . s = 0 over GF(2) for s.
"""

from __future__ import annotations

import math
import re

import numpy as np
from qubasic_core.host import HostState


class OracleMixin(HostState):
    """GROVER and SIMON commands for QBasicTerminal."""

    def _internal_bits(self, typed: str) -> str:
        """A typed bitstring in the internal order (qubit n-1 first)."""
        return typed[::-1] if getattr(self, '_endian_big', False) else typed

    def _emit_phase_flip(self, qc, bits: str) -> None:
        """Phase-flip the basis state `bits` (internal order, qubit n-1 first)."""
        n = len(bits)
        zeros = [q for q in range(n) if bits[n - 1 - q] == '0']
        for q in zeros:
            qc.x(q)
        self._emit_mcz(qc, list(range(n)))
        for q in zeros:
            qc.x(q)

    def cmd_grover(self, rest: str) -> None:
        """GROVER <bits> [<bits> ...] [ITERS k] [SHOTS n] [PROGRAM] — Grover search for marked states."""
        text = rest
        opts: dict = {}
        for key in ('ITERS', 'SHOTS'):
            m = re.search(rf'\b{key}\s+(\d+)', text, re.IGNORECASE)
            if m:
                opts[key] = int(m.group(1))
                text = text[:m.start()] + text[m.end():]
        program = bool(re.search(r'\bPROGRAM\b', text, re.IGNORECASE))
        text = re.sub(r'\bPROGRAM\b', '', text, flags=re.IGNORECASE)
        marked = [t for t in text.replace(',', ' ').split() if t]
        if not marked or any(set(t) - {'0', '1'} for t in marked):
            self.io.writeln("?USAGE: GROVER <bits> [<bits> ...] [ITERS k] [SHOTS n] [PROGRAM]")
            return
        n = len(marked[0])
        if any(len(t) != n for t in marked):
            self.io.writeln("?GROVER: every marked state needs the same number of bits")
            return
        if n > 20:
            self.io.writeln("?GROVER searches at most 20 qubits")
            return
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_aer import AerSimulator
            internal = sorted(set(self._internal_bits(t) for t in marked))
            M, N = len(internal), 1 << n
            if M >= N:
                raise ValueError("every state is marked")
            theta = math.asin(math.sqrt(M / N))
            k = opts.get('ITERS', max(0, int(math.floor(math.pi / (4 * theta)))))
            qc = QuantumCircuit(n)
            for q in range(n):
                qc.h(q)
            for _ in range(k):
                for bits in internal:
                    self._emit_phase_flip(qc, bits)
                for q in range(n):
                    qc.h(q)
                self._emit_phase_flip(qc, '0' * n)
                for q in range(n):
                    qc.h(q)
            shots = opts.get('SHOTS', self.shots)
            meas = qc.copy()
            meas.measure_all()
            backend = AerSimulator(noise_model=self._noise_model) if self._noise_model else AerSimulator()
            kw: dict = {'shots': shots}
            if self._seed is not None:
                kw['seed_simulator'] = self._seed
            counts = backend.run(transpile(meas, backend), **kw).result().get_counts()
            hits = sum(c for b, c in counts.items() if b in internal)
            p_meas = hits / shots
            p_theory = math.sin((2 * k + 1) * theta) ** 2
            self.io.writeln(f"\n  Grover: {n} qubits, {M} marked of {N}, {k} iteration(s)")
            self.io.writeln(f"    success probability {p_meas:.4f} measured, "
                            f"{p_theory:.4f} predicted (sin^2((2k+1)theta))")
            self.last_counts = dict(counts)
            self.print_histogram(counts)
            self.variables['_GROVER_P'] = p_meas
            self.variables['_GROVER_ITERS'] = k
            if program:
                typed = ' '.join(self._bits(b) for b in internal)
                flip = 'Z 0' if n == 1 else 'MCZ ' + ', '.join(str(q) for q in range(n))
                lines = [f"REM Grover search for {typed}"] + [f"H {q}" for q in range(n)]
                for _ in range(k):
                    for bits in internal:
                        zeros = [q for q in range(n) if bits[n - 1 - q] == '0']
                        lines += [f"X {q}" for q in zeros] + [flip] + [f"X {q}" for q in zeros]
                    lines.append(f"DIFFUSE 0-{n - 1}")
                lines.append("MEASURE")
                self._write_program(n, lines)
                self.io.writeln(f"    PROGRAM: {len(lines)} lines written (QUBITS {n})")
        except Exception as e:
            self.io.writeln(f"?GROVER ERROR: {e}")

    def cmd_simon(self, rest: str) -> None:
        """SIMON <secret bits> [SHOTS n] — Simon's algorithm: recover a hidden XOR period."""
        text = rest
        shots = None
        m = re.search(r'\bSHOTS\s+(\d+)', text, re.IGNORECASE)
        if m:
            shots = int(m.group(1))
            text = text[:m.start()] + text[m.end():]
        secret = text.strip()
        if not secret or set(secret) - {'0', '1'} or '1' not in secret:
            self.io.writeln("?USAGE: SIMON <nonzero secret bits> [SHOTS n]")
            return
        n = len(secret)
        if 2 * n > 24:
            self.io.writeln("?SIMON takes secrets of at most 12 bits")
            return
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_aer import AerSimulator
            from qubasic_core.qec2 import _gf2_nullspace
            s_int = self._internal_bits(secret)            # qubit n-1 first
            s_bits = [int(s_int[n - 1 - j]) for j in range(n)]   # s_bits[j] = bit of qubit j
            pivot = next(j for j in range(n) if s_bits[j])
            qc = QuantumCircuit(2 * n, n)
            for q in range(n):
                qc.h(q)
            for q in range(n):                           # f(x) = x ...
                qc.cx(q, n + q)
            for j in range(n):                           # ... XOR s when x_pivot = 1
                if s_bits[j]:
                    qc.cx(pivot, n + j)
            for q in range(n):
                qc.h(q)
            qc.measure(range(n), range(n))
            backend = AerSimulator()
            kw: dict = {'shots': shots or max(4 * n, 32)}
            if self._seed is not None:
                kw['seed_simulator'] = self._seed
            counts = backend.run(transpile(qc, backend), **kw).result().get_counts()
            rows = [[int(key[n - 1 - j]) for j in range(n)] for key in counts]
            null = _gf2_nullspace(np.array(rows, dtype=np.uint8))
            self.io.writeln(f"\n  Simon: {n}-bit secret, {kw['shots']} shots, "
                            f"{len(counts)} distinct equations y.s = 0")
            if null.shape[0] != 1:
                self.io.writeln(f"    the equations leave {null.shape[0]} candidate directions; "
                                f"add SHOTS")
                return
            found_bits = null[0]
            found = ''.join(str(int(found_bits[q])) for q in range(n - 1, -1, -1))
            shown = self._bits(found)
            ok = found == s_int
            self.io.writeln(f"    recovered s = {shown}  ({'matches' if ok else 'differs from'} "
                            f"the secret)")
            self.variables['_SIMON_OK'] = -1 if ok else 0
        except Exception as e:
            self.io.writeln(f"?SIMON ERROR: {e}")
