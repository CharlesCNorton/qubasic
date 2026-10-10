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
            if not self.program:
                self.io.writeln("  (no program: the target starts in |0...0>, which reads "
                                "phase 0 for a diagonal unitary — add prep gates, "
                                "e.g. 10 X 0, to select another eigenstate)")
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
        """AMPEST [m] <good> [SHOTS n] — amplitude estimation of the program's state.

        The program (measure-free and unitary) is the state preparation A; with
        no program, A is a Hadamard on every qubit. <good> marks the good
        outcomes: a qubit index (good when it reads 1) or a basis state |bits>
        in the display's bit order. Maximum-likelihood amplitude estimation
        runs Q^k A|0> for k = 0, 1, 2, 4, ..., 2^(m-1) Grover iterations
        (Q = A S0 A^dag S_good, default m = 5), counts the good outcomes, and
        maximizes the joint likelihood over theta with a = sin^2(theta). The
        reported spread is the Fisher-information standard deviation."""
        parts = rest.split()
        shots = self.shots
        if any(p.upper() == 'SHOTS' for p in parts):
            k = [p.upper() for p in parts].index('SHOTS')
            shots = int(parts[k + 1])
            parts = parts[:k] + parts[k + 2:]
        if not parts:
            self.io.writeln("?USAGE: AMPEST [m] <qubit | |bits>> [SHOTS n]")
            return
        if len(parts) == 1:
            m, good_spec = 5, parts[0]
        else:
            m, good_spec = int(parts[0]), parts[1]
        try:
            a_hat, sigma, exact, ks = self._mlae(good_spec, m, shots)
            ks_txt = ','.join(str(k) for k in ks)
            exact_txt = f"; exact {exact:.4f}" if exact is not None else ""
            self.io.writeln(f"  Amplitude estimate a = {a_hat:.4f} ± {sigma:.4f}  "
                            f"(MLAE, Grover powers {ks_txt}, {shots} shots each{exact_txt})")
            self.variables['_AMPEST'] = a_hat
            self.variables['_AMPEST_SIGMA'] = sigma
        except Exception as e:
            self.io.writeln(f"?AMPEST ERROR: {e}")

    def _mlae(self, good_spec: str, m: int, shots: int):
        """Maximum-likelihood amplitude estimation (Suzuki et al., 2020).

        Returns (a_hat, sigma_a, exact_a or None, grover_powers)."""
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import Gate
        if m < 0:
            raise ValueError("m must be >= 0")
        n = self.num_qubits
        if self.program:
            prep, _ = self.build_circuit()
            for inst in prep.data:
                op = inst.operation
                if op.name == 'barrier':
                    continue
                if not isinstance(op, Gate):
                    raise ValueError(f"the state preparation must be unitary; '{op.name}' "
                                     f"is not (AMPEST takes a measure-free, reset-free program)")
        else:
            prep = QuantumCircuit(n)
            for q in range(n):
                prep.h(q)
        prep = prep.copy()
        prep.remove_final_measurements(inplace=True)
        spec = good_spec.strip()
        target_bits = None
        good_qubit = None
        if spec.startswith('|'):
            bits = spec.strip('|>⟩')
            if len(bits) != n or set(bits) - {'0', '1'}:
                raise ValueError(f"good state must be {n} bits, e.g. |{'1' * n}>")
            target_bits = bits[::-1] if self._endian_big else bits     # internal q_{n-1}..q0
        else:
            good_qubit = int(self._eval_with_vars(spec, {}))
            if not 0 <= good_qubit < n:
                raise ValueError(f"qubit {good_qubit} out of range (0-{n - 1})")
        qubits = list(range(n))

        def phase_flip(circ, bits_hi_to_lo: str) -> None:
            zeros = [q for q in qubits if bits_hi_to_lo[n - 1 - q] == '0']
            for q in zeros:
                circ.x(q)
            self._emit_mcz(circ, qubits)
            for q in zeros:
                circ.x(q)

        def s_good(circ) -> None:
            if good_qubit is not None:
                circ.z(good_qubit)
            elif target_bits is not None:
                phase_flip(circ, target_bits)

        a_inv = prep.inverse()
        grover = QuantumCircuit(n)
        s_good(grover)
        grover.compose(a_inv, inplace=True)
        phase_flip(grover, '0' * n)
        grover.compose(prep, inplace=True)

        ks = [0] + [2 ** j for j in range(m)]
        circuits = []
        for k in ks:
            qc = QuantumCircuit(n, 1 if good_qubit is not None else n)
            qc.compose(prep, inplace=True)
            for _ in range(k):
                qc.compose(grover, inplace=True)
            if good_qubit is not None:
                qc.measure(good_qubit, 0)
            else:
                qc.measure(qubits, list(range(n)))
            circuits.append(qc)
        method, precision = self._select_method(circuits[-1])
        backend = self._make_backend(method if method != 'stabilizer' else 'automatic',
                                     include_noise=True, precision=precision)
        kw: dict = {'shots': shots}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        result = backend.run(transpile(circuits, backend, optimization_level=self._transpile_opt_level),
                             **kw).result()
        hits = []
        for i in range(len(ks)):
            counts = result.get_counts(i)
            if good_qubit is not None:
                hits.append(counts.get('1', 0))
            else:
                hits.append(counts.get(target_bits, 0))
        hits_a = np.array(hits, dtype=float)
        odd = np.array([2 * k + 1 for k in ks], dtype=float)

        def loglik(theta):
            s = np.clip(np.sin(odd * theta) ** 2, 1e-12, 1 - 1e-12)
            return float(np.sum(hits_a * np.log(s) + (shots - hits_a) * np.log(1 - s)))

        grid = np.linspace(0.0, np.pi / 2, 20001)
        vals = np.array([loglik(t) for t in grid])
        best = int(np.argmax(vals))
        lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
        gr = (np.sqrt(5) - 1) / 2                      # golden-section refinement
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        for _ in range(60):
            if loglik(c) > loglik(d):
                hi = d
            else:
                lo = c
            c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        theta = (lo + hi) / 2
        a_hat = float(np.sin(theta) ** 2)
        fisher = 4.0 * shots * float(np.sum(odd ** 2))
        sigma = float(abs(np.sin(2 * theta)) / np.sqrt(fisher))
        exact = None
        from qubasic_core import capacity as cap
        if cap.fits(cap.aer_peak_bytes(n, 'statevector', 'double', 'end')) and n <= 26:
            from qiskit.quantum_info import Statevector
            probs = np.abs(Statevector(prep).data) ** 2
            if good_qubit is not None:
                idx = np.arange(probs.size)
                exact = float(probs[((idx >> good_qubit) & 1) == 1].sum())
            elif target_bits is not None:
                exact = float(probs[int(target_bits, 2)])
        return a_hat, sigma, exact, ks

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
        # The target reads in the active display convention, so what you see in
        # the histogram is what you type here; internally it stays q_{n-1}..q0.
        if getattr(self, '_endian_big', False):
            target = target[::-1]
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
        """HHL a b c d v0 v1 [clock] — solve the 2x2 Hermitian system [[a,b],[c,d]] x = v.

        Runs the HHL algorithm: phase estimation of e^{iAt} on a signed clock
        register of `clock` qubits (default 7; negative eigenvalues read as
        two's complement), a uniformly controlled RY that rotates the ancilla
        by 2 arcsin(C / lambda), and uncomputation. As HHL assumes, the
        eigenvalue range is given classically: t maps the largest |lambda| to
        the clock's top value and C is 0.9 of the smallest |lambda|. Reports
        the post-selected solution direction, its fidelity to the classical
        A^-1 v, and the probability that post-selection succeeds."""
        parts = rest.split()
        if len(parts) < 6:
            self.io.writeln("?USAGE: HHL a b c d v0 v1 [clock]   (A=[[a,b],[c,d]], v=[v0,v1])")
            return
        try:
            vals = [float(self._eval_with_vars(p, {})) for p in parts[:6]]
            nclock = int(parts[6]) if len(parts) > 6 else 7
            if not 3 <= nclock <= 10:
                raise ValueError("clock must be 3-10 qubits")
            A = np.array([[vals[0], vals[1]], [vals[2], vals[3]]], dtype=complex)
            b = np.array([vals[4], vals[5]], dtype=complex)
            if not np.allclose(A, A.conj().T):
                raise ValueError("A must be Hermitian")
            if np.linalg.norm(b) < 1e-12:
                raise ValueError("v must be nonzero")
            w, V = np.linalg.eigh(A)
            if np.min(np.abs(w)) < 1e-9:
                raise ValueError("A is singular")
            x_q, success = self._hhl_solve(A, b, w, V, nclock)
            x_cl = np.linalg.solve(A, b)
            x_cl = x_cl / np.linalg.norm(x_cl)
            overlap = abs(np.vdot(x_cl, x_q)) ** 2
            self.io.writeln(f"  HHL solution (normalized): [{x_q[0].real:+.3f}, {x_q[1].real:+.3f}]")
            self.io.writeln(f"  Classical A^-1 v:          [{x_cl[0].real:+.3f}, {x_cl[1].real:+.3f}]")
            self.io.writeln(f"  Direction fidelity = {overlap:.4f}   "
                            f"(post-selection succeeds with p = {success:.4f}, {nclock} clock qubits)")
            self.variables['_HHL_FIDELITY'] = float(overlap)
            self.variables['_HHL_SUCCESS'] = float(success)
        except Exception as e:
            self.io.writeln(f"?HHL ERROR: {e}")

    def _hhl_solve(self, A, b, w, V, nclock: int) -> tuple[np.ndarray, float]:
        """Build and simulate the HHL circuit; returns (normalized x, success p)."""
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate, UCRYGate
        from qiskit.quantum_info import Statevector
        sys_q, clock, anc = 0, list(range(1, 1 + nclock)), 1 + nclock
        total = 2 + nclock
        half = 1 << (nclock - 1)
        # e^{iAt} with the largest |lambda| at clock value half - 1, so every
        # eigenvalue lands in the signed range (-half, half).
        t = 2 * np.pi * (half - 1) / ((1 << nclock) * float(np.max(np.abs(w))))
        U = V @ np.diag(np.exp(1j * w * t)) @ V.conj().T
        qc = QuantumCircuit(total)
        qc.initialize(b / np.linalg.norm(b), [sys_q])
        for c in clock:
            qc.h(c)
        for j, c in enumerate(clock):
            qc.append(UnitaryGate(np.linalg.matrix_power(U, 2 ** j)).control(1), [c, sys_q])
        self._emit_qft(qc, clock, inverse=True, swaps=True)
        # C/lambda must stay <= 1 for every clock value phase estimation lands
        # on; 0.9 of the smallest |eigenvalue| keeps the spread clear of the
        # arcsin clip while post-selection still succeeds about half the time.
        C = 0.9 * float(np.min(np.abs(w)))
        angles = [0.0]
        for k in range(1, 1 << nclock):
            ks = k if k < half else k - (1 << nclock)
            lam = 2 * np.pi * ks / ((1 << nclock) * t)
            angles.append(float(2 * np.arcsin(np.clip(C / lam, -1.0, 1.0))))
        qc.append(UCRYGate(angles), [anc] + clock)
        self._emit_qft(qc, clock, inverse=False, swaps=True)
        for j, c in enumerate(clock):
            qc.append(UnitaryGate(np.linalg.matrix_power(U.conj().T, 2 ** j)).control(1),
                      [c, sys_q])
        for c in clock:
            qc.h(c)
        sv = Statevector(qc).data
        # Post-select ancilla = 1 and clock = 0; qubit 0 (sys) is the low bit.
        x = np.array([sv[1 << (total - 1)], sv[(1 << (total - 1)) | 1]])
        success = float(np.vdot(x, x).real)
        if success < 1e-12:
            raise ValueError("post-selection amplitude vanished (add clock qubits)")
        return x / np.sqrt(success), success
