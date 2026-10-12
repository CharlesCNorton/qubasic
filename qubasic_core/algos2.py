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
from qubasic_core.host import HostState


class Algorithms2Mixin(HostState):
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

    # Widest N the modular-arithmetic circuit runs (2n + 3 qubits, n = 8).
    SHOR_MAX_BITS = 8

    def cmd_shor(self, rest: str) -> None:
        """SHOR <N> [a] [SHOTS n] [ORACLE] — factor N by quantum order finding.

        Runs Beauregard's circuit: modular exponentiation from Fourier-basis
        adders and controlled modular multipliers on 2n + 3 qubits, with the
        phase read one bit at a time through a single control qubit (see
        shor.py). The order r of a mod N comes from the measured phases by
        continued fractions, and a factor from gcd(a^(r/2) +- 1, N). Even N,
        primes and prime powers are settled classically. ORACLE runs phase
        estimation of the permutation matrix x -> a x mod N instead, on 3n
        qubits."""
        parts = rest.split()
        oracle = any(p_.upper() == 'ORACLE' for p_ in parts)
        parts = [p_ for p_ in parts if p_.upper() != 'ORACLE']
        shots = None
        if any(p_.upper() == 'SHOTS' for p_ in parts):
            k = [p_.upper() for p_ in parts].index('SHOTS')
            shots = int(parts[k + 1])
            parts = parts[:k] + parts[k + 2:]
        if not parts:
            self.io.writeln("?USAGE: SHOR <N> [a] [SHOTS n] [ORACLE]")
            return
        from qubasic_core import shor
        try:
            N = int(parts[0])
            if N < 4:
                raise ValueError("N must be at least 4")
            kind, facs = shor.classical_shortcut(N)
            if kind == 'prime':
                self.io.writeln(f"  N={N} is prime: nothing to factor")
                return
            if kind:
                assert facs is not None
                self.io.writeln(f"  N={N} is {'even' if kind == 'even' else 'a prime power'}: "
                                f"{facs[0]} x {facs[1]} (no order finding needed)")
                self.variables['_SHOR_FACTOR'] = facs[0]
                return
            n = N.bit_length()
            if not oracle and n > self.SHOR_MAX_BITS:
                raise ValueError(f"N={N} needs {2 * n + 3} qubits; the circuit runs N < "
                                 f"{1 << self.SHOR_MAX_BITS} (ORACLE reaches a little further)")
            rng = np.random.default_rng(self._seed)
            fixed_a = int(parts[1]) if len(parts) > 1 else None
            attempts = [fixed_a] if fixed_a is not None else [None] * 4
            tried: set[int] = set()
            for attempt in attempts:
                a = attempt if attempt is not None else int(rng.integers(2, N - 1))
                while attempt is None and a in tried and len(tried) < N - 3:
                    a = int(rng.integers(2, N - 1))
                tried.add(a)
                g = math.gcd(a, N)
                if g > 1:
                    self.io.writeln(f"  gcd({a},{N})={g} (lucky) -> factors {g} and {N // g}")
                    self.variables['_SHOR_FACTOR'] = g
                    return
                if oracle:
                    phases = self._shor_oracle_phases(a, N, shots or max(2000, self.shots))
                else:
                    phases = self._shor_circuit_phases(a, N, shots or 24)
                r = None
                for ph, _count in phases:
                    r = shor.order_from_phase(ph, a, N)
                    if r:
                        break
                if not r:
                    self.io.writeln("  order finding inconclusive for this a")
                    continue
                self.io.writeln(f"  Order r = {r} ({a}^{r} mod {N} = {pow(a, r, N)})")
                self.variables['_SHOR_ORDER'] = r
                found = shor.factors_from_order(a, r, N)
                if found:
                    self.io.writeln(f"  Factors of {N}: {min(found)} x {max(found)}")
                    self.variables['_SHOR_FACTOR'] = min(found)
                    return
                self.io.writeln("  odd order or a^(r/2) = -1 mod N: no factor from this a")
                if fixed_a is not None:
                    return
            self.io.writeln("  no factor found (retry, or give another a)")
        except Exception as e:
            self.io.writeln(f"?SHOR ERROR: {e}")

    def _shor_circuit_phases(self, a: int, N: int, shots: int) -> list:
        """Run the order-finding circuit; (phase, count) pairs, most frequent first."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qubasic_core import shor
        qc = shor.order_finding_circuit(a, N)
        t = 2 * N.bit_length()
        sim = AerSimulator(method='statevector', precision=self.sim_precision)
        qt = transpile(qc, sim)
        kw: dict = {'shots': shots}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        counts = sim.run(qt, **kw).result().get_counts()
        self.io.writeln(f"\n  Shor: N={N}, a={a}: modular-arithmetic circuit, {qc.num_qubits} qubits "
                        f"(2n+3), {qt.size()} operations, {shots} shots")
        tally: dict = {}
        for bits, c in counts.items():
            ph = shor.phase_value(bits, t)
            tally[ph] = tally.get(ph, 0) + c
        ranked = sorted(tally.items(), key=lambda kv: -kv[1])
        shown = ', '.join(f"{ph.numerator}/{ph.denominator} ({c})" for ph, c in ranked[:6])
        self.io.writeln(f"  measured phases: {shown}")
        return ranked

    def _shor_oracle_phases(self, a: int, N: int, shots: int) -> list:
        """Phase estimation of the permutation unitary x -> a x mod N."""
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import UnitaryGate
        from qiskit_aer import AerSimulator
        w = N.bit_length()
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
        kw: dict = {'shots': shots}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        counts = backend.run(transpile(qc, backend), **kw).result().get_counts()
        self.io.writeln(f"\n  Shor: N={N}, a={a}: permutation-oracle phase estimation, "
                        f"{count} counting qubits")
        ranked = sorted(((Fraction(int(bits, 2), 2 ** count), c) for bits, c in counts.items()),
                        key=lambda kv: -kv[1])
        return ranked

    # ── HHL linear solver ───────────────────────────────────────────────

    def _parse_hhl_args(self, rest: str) -> tuple[np.ndarray, np.ndarray, int | None, int]:
        """(A, b, clock qubits or None, shots) from either form:
        HHL a b c d v0 v1 [clock] (2x2) or HHL [[...]] [v...] [CLOCK n] [SHOTS n]."""
        text = rest.strip()
        shots = 0
        m = re.search(r'\bSHOTS\s+(\d+)\s*$', text, re.IGNORECASE)
        if m:
            shots = int(m.group(1))
            text = text[:m.start()].rstrip()
        clock = None
        m = re.search(r'\bCLOCK\s+(\d+)\s*$', text, re.IGNORECASE)
        if m:
            clock = int(m.group(1))
            text = text[:m.start()].rstrip()
        if text.startswith('['):
            depth, end = 0, -1
            for i, ch in enumerate(text):
                depth += ch == '['
                depth -= ch == ']'
                if depth == 0:
                    end = i
                    break
            if end < 0:
                raise ValueError("unbalanced brackets in the matrix")
            A = np.array(self._parse_matrix(text[:end + 1]), dtype=complex)
            vec = text[end + 1:].strip()
            if not vec.startswith('['):
                raise ValueError("give the right-hand side as a list, e.g. [1, 0]")
            b: np.ndarray = np.array(self._parse_matrix(vec), dtype=complex).ravel()
        else:
            parts = text.split()
            if len(parts) < 6:
                raise ValueError("HHL a b c d v0 v1 [clock]  or  HHL [[...]] [v...] [CLOCK n]")
            vals = [float(self._eval_with_vars(p_, {})) for p_ in parts[:6]]
            if len(parts) > 6 and clock is None:
                clock = int(parts[6])
            A = np.array([[vals[0], vals[1]], [vals[2], vals[3]]], dtype=complex)
            b = np.array([vals[4], vals[5]], dtype=complex)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")
        if b.size != A.shape[0]:
            raise ValueError(f"A is {A.shape[0]}x{A.shape[0]} but v has {b.size} entries")
        return A, b, clock, shots

    def cmd_hhl(self, rest: str) -> None:
        """HHL [[A]] [v] [CLOCK n] [SHOTS n] — solve A x = v with the HHL algorithm (any size).

        Also HHL a b c d v0 v1 [clock] for a 2x2 system. A Hermitian A runs as
        given; any other invertible A runs as its Hermitian dilation
        [[0, A], [A^dag, 0]] on [v, 0], whose solution holds x. The system is
        padded to a power of two. HHL runs phase estimation of e^{iAt} on a
        signed clock register (negative eigenvalues read as two's complement),
        a uniformly controlled RY that rotates the ancilla by 2 arcsin(C/lambda),
        and uncomputation; as HHL assumes, the eigenvalue range is given
        classically: t maps the largest |lambda| to the clock's top value and
        C is 0.9 of the smallest |lambda|. Reports the post-selected solution,
        its fidelity to the classical A^-1 v and the post-selection success
        probability; SHOTS n also samples the solution register."""
        try:
            A, b, nclock, shots = self._parse_hhl_args(rest)
            n = A.shape[0]
            if np.linalg.norm(b) < 1e-12:
                raise ValueError("v must be nonzero")
            if abs(np.linalg.det(A)) < 1e-12:
                raise ValueError("A is singular")
            hermitian = bool(np.allclose(A, A.conj().T, atol=1e-10))
            if hermitian:
                H, rhs = A, b
            else:
                H = np.block([[np.zeros((n, n)), A], [A.conj().T, np.zeros((n, n))]])
                rhs = np.concatenate([b, np.zeros(n, dtype=complex)])
            m = H.shape[0]
            k = max(1, int(np.ceil(np.log2(m))))
            dim = 1 << k
            w0 = np.linalg.eigvalsh(H)
            if dim > m:                     # pad with the largest |eigenvalue|
                pad = float(np.max(np.abs(w0)))
                Hp = np.eye(dim, dtype=complex) * pad
                Hp[:m, :m] = H
                bp = np.zeros(dim, dtype=complex)
                bp[:m] = rhs
                H, rhs = Hp, bp
            if nclock is None:
                nclock = 7 if dim <= 2 else 8
            if not 3 <= nclock <= 12:
                raise ValueError("clock must be 3-12 qubits")
            if k + nclock + 1 > 24:
                raise ValueError(f"{k + nclock + 1} qubits is beyond the statevector HHL "
                                 f"(at most 24): use a smaller system or clock")
            w, V = np.linalg.eigh(H)
            x_full, success = self._hhl_solve(H, rhs, w, V, nclock)
            x_q = x_full[:m] if hermitian else x_full[n:2 * n]
            nq = np.linalg.norm(x_q)
            x_q = x_q / nq if nq > 0 else x_q
            x_cl = np.linalg.solve(A, b)
            x_cl = x_cl / np.linalg.norm(x_cl)
            overlap = float(abs(np.vdot(x_cl, x_q)) ** 2)

            def fmt(vec):
                if np.allclose(vec.imag, 0, atol=1e-6):
                    return '[' + ', '.join(f"{v.real:+.3f}" for v in vec) + ']'
                return '[' + ', '.join(f"{v.real:+.3f}{v.imag:+.3f}j" for v in vec) + ']'

            shown = n <= 8
            label = '' if hermitian else ' (Hermitian dilation)'
            self.io.writeln(f"  HHL: {n}x{n} system{label}, {k} system + {nclock} clock + 1 "
                            f"ancilla qubits")
            if shown:
                self.io.writeln(f"  HHL solution (normalized): {fmt(x_q)}")
                self.io.writeln(f"  Classical A^-1 v:          {fmt(x_cl)}")
            self.io.writeln(f"  Direction fidelity = {overlap:.4f}   "
                            f"(post-selection succeeds with p = {success:.4f}, {nclock} clock qubits)")
            if shots:
                rng = np.random.default_rng(self._seed)
                kept = int(rng.binomial(shots, min(max(success, 0.0), 1.0)))
                probs = np.abs(x_q) ** 2
                probs = probs / probs.sum()
                draws = rng.multinomial(kept, probs) if kept else np.zeros(n, dtype=int)
                width = max(1, int(np.ceil(np.log2(max(n, 2)))))
                self.io.writeln(f"  Sampled {shots} shots: {kept} passed post-selection")
                for i in np.argsort(-draws)[:min(n, 16)]:
                    if draws[i]:
                        self.io.writeln(f"    |{format(int(i), f'0{width}b')}>  {int(draws[i]):>6}  "
                                        f"(ideal {probs[i]:.4f})")
            self.variables['_HHL_FIDELITY'] = overlap
            self.variables['_HHL_SUCCESS'] = float(success)
            self.arrays['_HHL_X'] = [complex(v) for v in x_q]
        except Exception as e:
            self.io.writeln(f"?HHL ERROR: {e}")

    def _hhl_solve(self, A, b, w, V, nclock: int) -> tuple[np.ndarray, float]:
        """Build and simulate the HHL circuit for a Hermitian A of size 2^k;
        returns (normalized x, success probability)."""
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate, UCRYGate
        from qiskit.quantum_info import Statevector
        dim = A.shape[0]
        k = int(round(np.log2(dim)))
        sys_q = list(range(k))
        clock = list(range(k, k + nclock))
        anc = k + nclock
        total = k + nclock + 1
        half = 1 << (nclock - 1)
        # e^{iAt} with the largest |lambda| at clock value half - 1, so every
        # eigenvalue lands in the signed range (-half, half).
        t = 2 * np.pi * (half - 1) / ((1 << nclock) * float(np.max(np.abs(w))))
        U = V @ np.diag(np.exp(1j * w * t)) @ V.conj().T
        qc = QuantumCircuit(total)
        qc.initialize(b / np.linalg.norm(b), sys_q)
        for c in clock:
            qc.h(c)
        for j, c in enumerate(clock):
            qc.append(UnitaryGate(np.linalg.matrix_power(U, 2 ** j)).control(1), [c] + sys_q)
        self._emit_qft(qc, clock, inverse=True, swaps=True)
        # C/lambda must stay <= 1 for every clock value phase estimation lands
        # on; 0.9 of the smallest |eigenvalue| keeps the spread clear of the
        # arcsin clip while post-selection still succeeds often.
        C = 0.9 * float(np.min(np.abs(w)))
        angles = [0.0]
        for kk in range(1, 1 << nclock):
            ks = kk if kk < half else kk - (1 << nclock)
            lam = 2 * np.pi * ks / ((1 << nclock) * t)
            angles.append(float(2 * np.arcsin(np.clip(C / lam, -1.0, 1.0))))
        qc.append(UCRYGate(angles), [anc] + clock)
        self._emit_qft(qc, clock, inverse=False, swaps=True)
        for j, c in enumerate(clock):
            qc.append(UnitaryGate(np.linalg.matrix_power(U.conj().T, 2 ** j)).control(1),
                      [c] + sys_q)
        for c in clock:
            qc.h(c)
        sv = Statevector(qc).data
        # Post-select ancilla = 1 and clock = 0; the system register is the low bits.
        base = 1 << (total - 1)
        x = np.array([sv[base | i] for i in range(dim)])
        success = float(np.vdot(x, x).real)
        if success < 1e-12:
            raise ValueError("post-selection amplitude vanished (add clock qubits)")
        return x / np.sqrt(success), success
