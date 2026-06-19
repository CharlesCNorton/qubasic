"""QUBASIC advanced algorithm primitives.

Amplitude estimation and amplification, iterative phase estimation, quantum
walks, graph/cluster states for measurement-based computing, quantum-machine-
learning feature maps and kernels, Shor order finding, and HHL. All offline.

  IQPE <bits> <target> NAME     Iterative phase estimation of a UNITARY
  AMPEST <bits> <good>          Amplitude estimation of the |good> amplitude
  AMPLIFY <bitstring>           One amplitude-amplification (Grover) step
  QWALK <steps>                 Discrete-time quantum walk on a cycle
  GRAPHSTATE <edges>            Prepare a graph/cluster state (e.g. 0-1, 1-2)
  FEATUREMAP <x0> <x1> ...      ZZ feature-map data encoding (one value per qubit)
  QKERNEL <x..> ; <y..>         Quantum kernel |<phi(x)|phi(y)>|^2
  SHOR <N> [a]                  Order finding / factoring of small N
  HHL <a> <b> <c> <d> <v0> <v1> Solve a 2x2 Hermitian system A x = v
"""

from __future__ import annotations

import re
import math
from fractions import Fraction

import numpy as np


class Algorithms2Mixin:
    """Advanced algorithm primitives for QBasicTerminal.

    Requires: TerminalProtocol — uses self.num_qubits, self._custom_gates,
    self._eval_with_vars(), self._emit_qft(), self._emit_mcz(), self.io.
    """

    # ── Iterative phase estimation ──────────────────────────────────────

    def cmd_iqpe(self, rest: str) -> None:
        """IQPE <bits> <target_range> <UNITARY> — iterative phase estimation.

        Estimates the eigenphase of a UNITARY (target prepared in an eigenstate)
        one bit at a time with a single ancilla, to `bits` of precision."""
        parts = rest.split()
        if len(parts) < 3:
            self.io.writeln("?USAGE: IQPE <bits> <target> <UNITARY>")
            return
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import UnitaryGate
        from qiskit_aer import AerSimulator
        try:
            bits = int(parts[0])
            targ = self._alg_qubit_list(parts[1])
            uname = parts[2].upper()
            if uname not in self._custom_gates:
                raise ValueError(f"unknown UNITARY '{uname}'")
            U = self._custom_gates[uname]
            prep, _ = self.build_circuit()    # the program prepares the target eigenstate
            nq = self.num_qubits
            anc = nq                          # dedicated ancilla beyond the system
            backend = AerSimulator(noise_model=self._noise_model) if self._noise_model else AerSimulator()
            phase_bits = [0] * bits
            # Estimate from least significant bit upward, feeding back the phase.
            for k in range(bits - 1, -1, -1):
                qc = QuantumCircuit(nq + 1, 1)
                qc.compose(prep, qubits=range(nq), inplace=True)
                qc.h(anc)
                Uk = np.linalg.matrix_power(U, 2 ** k)
                qc.append(UnitaryGate(Uk).control(1), [anc] + list(targ))
                fb = sum(phase_bits[j] / (2 ** (j - k + 1)) for j in range(k + 1, bits))
                qc.p(-2 * np.pi * fb, anc)
                qc.h(anc)
                qc.measure(anc, 0)
                counts = backend.run(transpile(qc, backend), shots=400).result().get_counts()
                phase_bits[k] = 1 if counts.get('1', 0) > counts.get('0', 0) else 0
            phase = sum(b / 2 ** (i + 1) for i, b in enumerate(phase_bits))
            self.io.writeln(f"  IQPE phase = {phase:.6f}  (bits {''.join(map(str, phase_bits))})")
            self.variables['_IQPE'] = phase
        except Exception as e:
            self.io.writeln(f"?IQPE ERROR: {e}")

    # ── Amplitude estimation ────────────────────────────────────────────

    def cmd_ampest(self, rest: str) -> None:
        """AMPEST <bits> <good_qubit> — estimate the amplitude of the marked state.

        Prepares uniform superposition on all qubits, treats |good_qubit = 1> as
        the good subspace, and runs canonical (QPE-based) amplitude estimation,
        reporting the estimated probability a = sin^2(pi y / 2^bits)."""
        parts = rest.split()
        if len(parts) < 2:
            self.io.writeln("?USAGE: AMPEST <bits> <good_qubit>")
            return
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        try:
            bits = int(parts[0])
            good = int(parts[1])
            nq = self.num_qubits
            # State qubits 0..nq-1 prepared in uniform superposition; the Grover
            # operator Q = A S_0 A^dag S_chi with A = H^{nq}, chi = (good == 1).
            count = list(range(nq, nq + bits))
            total = nq + bits
            qc = QuantumCircuit(total, bits)
            for q in range(nq):
                qc.h(q)
            for cq in count:
                qc.h(cq)

            def oracle(circ):       # phase flip when good qubit is 1
                circ.z(good)

            def diffusion(circ):    # A S_0 A^dag = H^n (2|0><0|-I) H^n on state qubits
                for q in range(nq):
                    circ.h(q)
                for q in range(nq):
                    circ.x(q)
                self._emit_mcz(circ, list(range(nq)))
                for q in range(nq):
                    circ.x(q)
                for q in range(nq):
                    circ.h(q)

            for j, cq in enumerate(count):
                for _ in range(2 ** j):
                    # Controlled Grover operator Q (oracle then diffusion),
                    # controlled on the counting qubit.
                    qc.cz(cq, good)          # controlled oracle (Z on good)
                    # controlled diffusion: conjugate the controlled-MCZ by H,X
                    for q in range(nq):
                        qc.h(q); qc.x(q)
                    qc.h(nq - 1)
                    qc.mcx([cq] + list(range(nq - 1)), nq - 1)
                    qc.h(nq - 1)
                    for q in range(nq):
                        qc.x(q); qc.h(q)
            self._emit_qft(qc, count, inverse=True, swaps=True)
            qc.measure(count, range(bits))
            backend = AerSimulator(noise_model=self._noise_model) if self._noise_model else AerSimulator()
            counts = backend.run(transpile(qc, backend), shots=max(2000, self.shots)).result().get_counts()
            y = int(max(counts, key=counts.get), 2)
            a = math.sin(math.pi * y / 2 ** bits) ** 2
            # Amplitude estimation is symmetric in y and 2^bits - y.
            a = min(a, math.sin(math.pi * (2 ** bits - y) / 2 ** bits) ** 2) if a > 0.5 else a
            self.io.writeln(f"  Amplitude estimate a = {a:.4f}  (y={y}/{2 ** bits})")
            self.variables['_AMPEST'] = a
        except Exception as e:
            self.io.writeln(f"?AMPEST ERROR: {e}")

    # ── Amplitude amplification (one Grover step) ───────────────────────

    def _try_exec_amplify(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        m = re.match(r'AMPLIFY\s+([01]+)\s*$', stmt, re.IGNORECASE)
        if not m:
            return False
        if getattr(self, 'locc_mode', False):
            raise ValueError("AMPLIFY is not available in LOCC mode")
        target = m.group(1)
        n = self.num_qubits
        if len(target) != n:
            raise ValueError(f"AMPLIFY target must be {n} bits, got {len(target)}")
        # Oracle: phase-flip |target> by X-conjugating the all-ones MCZ.
        flip = [i for i, b in enumerate(target) if b == '0']  # bitstring is q_{n-1}..q0
        qubits = list(range(n))
        idx = {q: n - 1 - q for q in qubits}   # bit position of qubit q in target
        zero_qubits = [q for q in qubits if target[idx[q]] == '0']
        for q in zero_qubits:
            qc.x(q)
        self._emit_mcz(qc, qubits)
        for q in zero_qubits:
            qc.x(q)
        # Diffusion about the uniform state.
        for q in qubits:
            qc.h(q)
        for q in qubits:
            qc.x(q)
        self._emit_mcz(qc, qubits)
        for q in qubits:
            qc.x(q)
        for q in qubits:
            qc.h(q)
        return True

    # ── Quantum walk ────────────────────────────────────────────────────

    def cmd_qwalk(self, rest: str) -> None:
        """QWALK <steps> — discrete-time quantum walk on a cycle of 2^(n-1) nodes.

        Uses qubit 0 as the coin and qubits 1..n-1 as the position register
        (binary), applying a Hadamard coin and a coin-controlled increment/
        decrement shift each step, then shows the position distribution."""
        try:
            steps = int(rest.strip()) if rest.strip() else 5
        except ValueError:
            self.io.writeln("?USAGE: QWALK <steps>")
            return
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        n = self.num_qubits
        if n < 2:
            self.io.writeln("?QWALK needs at least 2 qubits (1 coin + position)")
            return
        pos = list(range(1, n))
        m = len(pos)
        qc = QuantumCircuit(n, m)
        # Start in the middle of the line.
        mid = (2 ** m) // 2
        for j in range(m):
            if (mid >> j) & 1:
                qc.x(pos[j])
        for _ in range(steps):
            qc.h(0)                       # Hadamard coin
            # coin=1 -> increment position (mod 2^m): controlled increment
            self._controlled_increment(qc, 0, pos, +1)
            qc.x(0)
            self._controlled_increment(qc, 0, pos, -1)
            qc.x(0)
        qc.measure(pos, range(m))
        backend = AerSimulator()
        counts = backend.run(transpile(qc, backend), shots=max(2000, self.shots)).result().get_counts()
        self.io.writeln(f"\n  Quantum walk, {steps} steps, {2 ** m} positions:")
        total = sum(counts.values())
        for state in sorted(counts, key=lambda s: int(s, 2)):
            p = counts[state] / total
            bar = '#' * int(40 * p)
            self.io.writeln(f"    pos {int(state, 2):>3}  {p:6.3f}  {bar}")

    def _controlled_increment(self, qc, ctrl, pos, sign):
        """Coin-controlled +-1 modular increment of the position register.

        pos[0] is the LSB. Increment is a ripple of multi-controlled X gates;
        decrement is the increment conjugated by X on every position qubit.
        """
        m = len(pos)
        if sign < 0:
            for q in pos:
                qc.x(q)
            self._controlled_increment(qc, ctrl, pos, +1)
            for q in pos:
                qc.x(q)
            return
        for k in range(m - 1, -1, -1):
            controls = [ctrl] + pos[:k]
            if len(controls) == 1:
                qc.cx(ctrl, pos[k])
            else:
                qc.mcx(controls, pos[k])

    # ── Graph / cluster states (MBQC resource) ──────────────────────────

    def _try_exec_graphstate(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        m = re.match(r'GRAPHSTATE\s+(.*)$', stmt, re.IGNORECASE)
        if not m:
            return False
        if getattr(self, 'locc_mode', False):
            raise ValueError("GRAPHSTATE is not available in LOCC mode")
        spec = m.group(1).strip()
        for q in range(self.num_qubits):
            qc.h(q)
        if spec.upper() in ('LINE', 'CHAIN', ''):
            edges = [(i, i + 1) for i in range(self.num_qubits - 1)]
        elif spec.upper() == 'RING':
            edges = [(i, (i + 1) % self.num_qubits) for i in range(self.num_qubits)]
        elif spec.upper() == 'COMPLETE':
            edges = [(i, j) for i in range(self.num_qubits) for j in range(i + 1, self.num_qubits)]
        else:
            edges = []
            for tok in spec.replace(',', ' ').split():
                a, b = tok.split('-')
                edges.append((int(a), int(b)))
        for a, b in edges:
            qc.cz(a, b)
        return True

    # ── QML feature map and kernel ──────────────────────────────────────

    def _emit_feature_map(self, qc, x, reps=2):
        """ZZ feature map: H + RZ(2 x_i) per qubit, RZZ(2(pi-x_i)(pi-x_j)) on pairs."""
        n = len(x)
        for _ in range(reps):
            for i in range(n):
                qc.h(i)
                qc.p(2.0 * x[i], i)
            for i in range(n - 1):
                ang = 2.0 * (np.pi - x[i]) * (np.pi - x[i + 1])
                qc.cx(i, i + 1); qc.p(ang, i + 1); qc.cx(i, i + 1)

    def _try_exec_featuremap(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        m = re.match(r'FEATUREMAP\s+(.*)$', stmt, re.IGNORECASE)
        if not m:
            return False
        if getattr(self, 'locc_mode', False):
            raise ValueError("FEATUREMAP is not available in LOCC mode")
        x = [float(self._eval_with_vars(t, run_vars)) for t in m.group(1).replace(',', ' ').split()]
        if len(x) != self.num_qubits:
            raise ValueError(f"FEATUREMAP needs {self.num_qubits} values, got {len(x)}")
        self._emit_feature_map(qc, x)
        return True

    def cmd_qkernel(self, rest: str) -> None:
        """QKERNEL <x..> ; <y..> — quantum kernel |<phi(x)|phi(y)>|^2.

        Builds the ZZ feature-map states for data points x and y and returns
        their fidelity, the kernel entry used by quantum support vector machines.
        K(x, x) = 1."""
        if ';' not in rest:
            self.io.writeln("?USAGE: QKERNEL <x values> ; <y values>")
            return
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        try:
            xs, ys = rest.split(';', 1)
            x = [float(self._eval_with_vars(t, {})) for t in xs.replace(',', ' ').split()]
            y = [float(self._eval_with_vars(t, {})) for t in ys.replace(',', ' ').split()]
            n = max(len(x), len(y))
            qcx = QuantumCircuit(n); self._emit_feature_map(qcx, x)
            qcy = QuantumCircuit(n); self._emit_feature_map(qcy, y)
            k = abs(np.vdot(Statevector(qcx).data, Statevector(qcy).data)) ** 2
            self.io.writeln(f"  Quantum kernel K(x,y) = {k:.6f}")
            self.variables['_QKERNEL'] = float(k)
        except Exception as e:
            self.io.writeln(f"?QKERNEL ERROR: {e}")

    # ── Shor order finding / factoring ──────────────────────────────────

    def cmd_shor(self, rest: str) -> None:
        """SHOR <N> [a] — factor a small odd composite N by quantum order finding.

        Builds the modular-multiplication unitary x -> a*x mod N as a permutation,
        runs phase estimation to find the order r of a mod N, and recovers a factor
        via gcd(a^(r/2) +- 1, N)."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: SHOR <N> [a]")
            return
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import UnitaryGate
        from qiskit_aer import AerSimulator
        try:
            N = int(parts[0])
            if N % 2 == 0:
                self.io.writeln(f"  N={N} is even -> factor 2"); return
            w = N.bit_length()
            rng = np.random.default_rng(self._seed)
            a = int(parts[1]) if len(parts) > 1 else int(rng.integers(2, N - 1))
            g = math.gcd(a, N)
            if g > 1:
                self.io.writeln(f"  gcd({a},{N})={g} (lucky) -> factors {g} and {N // g}")
                return
            # Permutation unitary U_a |x> = |a x mod N> (identity on x >= N).
            dim = 2 ** w
            P = np.zeros((dim, dim))
            for x in range(dim):
                P[(a * x % N) if x < N else x, x] = 1.0
            count = 2 * w
            total = count + w
            qc = QuantumCircuit(total, count)
            for c in range(count):
                qc.h(c)
            qc.x(count)        # work register starts in |1>
            for j in range(count):
                Uk = np.linalg.matrix_power(P, 2 ** j).astype(complex)
                qc.append(UnitaryGate(Uk).control(1), [j] + list(range(count, total)))
            self._emit_qft(qc, list(range(count)), inverse=True, swaps=True)
            qc.measure(range(count), range(count))
            backend = AerSimulator()
            counts = backend.run(transpile(qc, backend), shots=max(2000, self.shots)).result().get_counts()
            self.io.writeln(f"\n  Shor: N={N}, a={a}, {count} counting qubits")
            found = None
            for bits in sorted(counts, key=counts.get, reverse=True)[:8]:
                y = int(bits, 2)
                if y == 0:
                    continue
                r = Fraction(y, 2 ** count).limit_denominator(N).denominator
                if r > 1 and pow(a, r, N) == 1:
                    found = r
                    break
            if not found:
                self.io.writeln("  order finding inconclusive (retry with another a)")
                return
            self.io.writeln(f"  Order r = {found} (a^r mod N = {pow(a, found, N)})")
            if found % 2 == 0:
                x = pow(a, found // 2, N)
                f1, f2 = math.gcd(x - 1, N), math.gcd(x + 1, N)
                facs = [f for f in (f1, f2) if 1 < f < N]
                if facs:
                    self.io.writeln(f"  Factors of {N}: {facs[0]} x {N // facs[0]}")
                    self.variables['_SHOR_FACTOR'] = facs[0]
                    return
            self.io.writeln("  odd order or trivial factor (retry with another a)")
        except Exception as e:
            self.io.writeln(f"?SHOR ERROR: {e}")

    # ── HHL linear solver (2x2) ─────────────────────────────────────────

    def cmd_hhl(self, rest: str) -> None:
        """HHL a b c d v0 v1 — solve the 2x2 Hermitian system [[a,b],[c,d]] x = v.

        Runs the HHL algorithm (phase estimation, eigenvalue inversion, uncompute)
        and reports the normalized solution direction, compared to the classical
        A^{-1} v."""
        parts = rest.split()
        if len(parts) < 6:
            self.io.writeln("?USAGE: HHL a b c d v0 v1   (A=[[a,b],[c,d]], v=[v0,v1])")
            return
        try:
            vals = [float(self._eval_with_vars(p, {})) for p in parts[:6]]
            A = np.array([[vals[0], vals[1]], [vals[2], vals[3]]], dtype=complex)
            b = np.array([vals[4], vals[5]], dtype=complex)
            if not np.allclose(A, A.conj().T):
                self.io.writeln("?HHL: A must be Hermitian")
                return
            from qiskit import QuantumCircuit, transpile
            from qiskit.circuit.library import UnitaryGate
            from qiskit.quantum_info import Statevector
            nclock = 4
            sys, clock, anc = 0, list(range(1, 1 + nclock)), 1 + nclock
            total = 2 + nclock
            bnorm = b / np.linalg.norm(b)
            qc = QuantumCircuit(total)
            qc.initialize(bnorm, [sys])
            # QPE with e^{i A t}; choose t so eigenvalues map into [0, 2^nclock).
            evals = np.linalg.eigvalsh(A)
            t = 2 * np.pi / (2 ** nclock) * (2 ** nclock - 1) / max(abs(evals).max(), 1e-9)
            from scipy.linalg import expm  # noqa
        except Exception:
            # scipy may be absent; build expm from eigh.
            pass
        try:
            w, V = np.linalg.eigh(A)
            U = V @ np.diag(np.exp(1j * w * t)) @ V.conj().T
            for c in clock:
                qc.h(c)
            for j, c in enumerate(clock):
                Uk = np.linalg.matrix_power(U, 2 ** j)
                qc.append(UnitaryGate(Uk).control(1), [c, sys])
            self._emit_qft(qc, clock, inverse=True, swaps=True)
            # Eigenvalue inversion: controlled RY on ancilla, angle ~ 1/lambda.
            C = 1.0
            for k in range(1, 2 ** nclock):
                lam = 2 * np.pi * k / (2 ** nclock) / t
                theta = 2 * np.arcsin(max(-1, min(1, C / lam)))
                bits = format(k, f'0{nclock}b')
                ctrl_state = bits[::-1]
                for j, bit in enumerate(ctrl_state):
                    if bit == '0':
                        qc.x(clock[j])
                qc.append(UnitaryGate(np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                                                 [np.sin(theta / 2), np.cos(theta / 2)]])).control(nclock),
                          clock + [anc])
                for j, bit in enumerate(ctrl_state):
                    if bit == '0':
                        qc.x(clock[j])
            self._emit_qft(qc, clock, inverse=False, swaps=True)
            for j, c in enumerate(clock):
                Uk = np.linalg.matrix_power(U.conj().T, 2 ** j)
                qc.append(UnitaryGate(Uk).control(1), [c, sys])
            for c in clock:
                qc.h(c)
            sv = Statevector(qc).data.reshape([2] * total)
            # Post-select ancilla = 1, clock = 0; read the system qubit.
            # Index ordering: qubit 0 (sys) is the least significant.
            amp0 = sv.reshape(-1)[(1 << (total - 1)) | 0]   # anc=1, others 0, sys=0
            amp1 = sv.reshape(-1)[(1 << (total - 1)) | 1]   # sys=1
            x_q = np.array([amp0, amp1])
            if np.linalg.norm(x_q) < 1e-9:
                self.io.writeln("  HHL: post-selection amplitude vanished (increase clock bits)")
                return
            x_q = x_q / np.linalg.norm(x_q)
            x_cl = np.linalg.solve(A, b)
            x_cl = x_cl / np.linalg.norm(x_cl)
            overlap = abs(np.vdot(x_cl, x_q)) ** 2
            self.io.writeln(f"  HHL solution (normalized): [{x_q[0].real:+.3f}, {x_q[1].real:+.3f}]")
            self.io.writeln(f"  Classical A^-1 v:          [{x_cl[0].real:+.3f}, {x_cl[1].real:+.3f}]")
            self.io.writeln(f"  Direction fidelity = {overlap:.4f}")
            self.variables['_HHL_FIDELITY'] = float(overlap)
        except Exception as e:
            self.io.writeln(f"?HHL ERROR: {e}")
