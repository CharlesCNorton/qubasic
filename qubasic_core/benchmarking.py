"""QUBASIC frontier benchmarking and verification.

Cross-entropy benchmarking, quantum volume, interleaved and mirror randomized
benchmarking, a linear-inversion 1-qubit gate-set tomography, and entanglement
measures. All offline on Qiskit/Aer/numpy.

  XEB [n] [depth] [trials]   Linear cross-entropy benchmarking fidelity
  QVOLUME [n]                Quantum volume heavy-output test at width n
  RBINT <gate> [maxlen]      Interleaved RB: fidelity of a specific Clifford gate
  MIRROR [maxlen] [samples]  Mirror-circuit benchmarking (scalable)
  GST                        1-qubit linear-inversion gate-set-style process estimate
  CONCURRENCE [a b]          Two-qubit concurrence of a qubit pair
  NEGATIVITY [qubits]        Entanglement negativity across a bipartition
"""

from __future__ import annotations

import re

import numpy as np


class BenchmarkingMixin:
    """XEB, quantum volume, RB variants, GST, and entanglement measures.

    Requires: TerminalProtocol — uses self.num_qubits, self.shots, self._seed,
    self._noise_model, self._active_sv, self._single_qubit_cliffords(), self.io.
    """

    # ── helpers ──────────────────────────────────────────────────────────

    def _bench_backend(self):
        from qiskit_aer import AerSimulator
        return AerSimulator(noise_model=self._noise_model) if self._noise_model else AerSimulator()

    def _random_circuit(self, n: int, depth: int, rng):
        """Brickwork random circuit: random U3 layer + CZ entangling layer."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        for d in range(depth):
            for q in range(n):
                qc.u(rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi),
                     rng.uniform(0, 2 * np.pi), q)
            for q in range(d % 2, n - 1, 2):
                qc.cz(q, q + 1)
        return qc

    # ── XEB ───────────────────────────────────────────────────────────────

    def cmd_xeb(self, rest: str) -> None:
        """XEB [n_qubits] [depth] [trials] — linear cross-entropy benchmarking.

        For each random circuit, computes the ideal output distribution, samples
        it (through the active noise model), and forms the self-normalized
        linear XEB fidelity F = (2^n <p_ideal(sampled)> - 1)/(2^n sum(p^2) - 1),
        pooled over trials. The denominator is the numerator's value under
        perfect sampling, so F = 1 ideal with no finite-dimension bias (the raw
        estimator tops out at (D-1)/(D+1), only 0.6 at 2 qubits) and F decays
        toward 0 as noise scrambles the output."""
        from qiskit import transpile
        parts = rest.split()
        n = int(parts[0]) if len(parts) > 0 else min(self.num_qubits, 4)
        depth = int(parts[1]) if len(parts) > 1 else 10
        trials = int(parts[2]) if len(parts) > 2 else 20
        rng = np.random.default_rng(self._seed)
        backend = self._bench_backend()
        shots = max(500, self.shots)
        num_sum = 0.0
        den_sum = 0.0
        try:
            from qiskit.quantum_info import Statevector
            for _ in range(trials):
                qc = self._random_circuit(n, depth, rng)
                probs = np.abs(Statevector(qc).data) ** 2
                qcm = qc.copy(); qcm.measure_all()
                kw = {'shots': shots}
                if self._seed is not None:
                    kw['seed_simulator'] = int(rng.integers(1 << 30))
                counts = backend.run(transpile(qcm, backend), **kw).result().get_counts()
                acc = 0.0
                for bits, c in counts.items():
                    acc += probs[int(bits, 2)] * c
                num_sum += (2 ** n) * (acc / shots) - 1.0
                den_sum += (2 ** n) * float(np.sum(probs ** 2)) - 1.0
            f = float(num_sum / den_sum) if abs(den_sum) > 1e-12 else 0.0
            self.io.writeln(f"\n  XEB: {n} qubits, depth {depth}, {trials} circuits")
            self.io.writeln(f"  Linear XEB fidelity = {f:.4f}  (1.0 ideal, 0.0 fully scrambled)")
            self.variables['_XEB'] = f
        except Exception as e:
            self.io.writeln(f"?XEB ERROR: {e}")

    # ── Quantum volume ─────────────────────────────────────────────────────

    def cmd_qvolume(self, rest: str) -> None:
        """QVOLUME [n] [trials] — quantum volume heavy-output test at width n.

        Builds QV model circuits (n layers of random SU(4) on random qubit pairs),
        computes each circuit's heavy outputs (probability above the median), and
        measures the heavy-output probability through the active noise model. The
        width passes if HOP > 2/3; quantum volume is then 2^n."""
        from qiskit import transpile
        from qiskit.quantum_info import Statevector, random_unitary
        parts = rest.split()
        n = int(parts[0]) if len(parts) > 0 else min(self.num_qubits, 4)
        trials = int(parts[1]) if len(parts) > 1 else 20
        if n < 2:
            self.io.writeln("?QVOLUME needs n >= 2")
            return
        rng = np.random.default_rng(self._seed)
        np.random.seed(self._seed if self._seed is not None else None)
        backend = self._bench_backend()
        shots = max(500, self.shots)
        hops = []
        try:
            from qiskit import QuantumCircuit
            for _ in range(trials):
                qc = QuantumCircuit(n)
                for _layer in range(n):
                    perm = list(rng.permutation(n))
                    for i in range(0, n - 1, 2):
                        a, b = perm[i], perm[i + 1]
                        qc.append(random_unitary(4, seed=int(rng.integers(1 << 30))), [a, b])
                probs = np.abs(Statevector(qc).data) ** 2
                median = float(np.median(probs))
                qcm = qc.copy(); qcm.measure_all()
                counts = backend.run(transpile(qcm, backend), shots=shots).result().get_counts()
                hop = sum(c for b, c in counts.items() if probs[int(b, 2)] > median) / shots
                hops.append(hop)
            mean_hop = float(np.mean(hops))
            passed = mean_hop > 2.0 / 3.0
            self.io.writeln(f"\n  Quantum volume test: width {n}, {trials} circuits")
            self.io.writeln(f"  Heavy-output probability = {mean_hop:.4f}  (threshold 0.667)")
            self.io.writeln(f"  Width {n} {'PASSES' if passed else 'FAILS'}"
                            + (f" -> quantum volume >= {2 ** n}" if passed else ""))
            self.variables['_HOP'] = mean_hop
        except Exception as e:
            self.io.writeln(f"?QVOLUME ERROR: {e}")

    # ── Interleaved RB ─────────────────────────────────────────────────────

    def _rb_survival(self, lengths, samples, interleave, rng):
        """Mean survival per length for single-qubit RB, optionally interleaving
        a fixed gate-word after each random Clifford."""
        from qiskit import QuantumCircuit, transpile
        cliffords = self._single_qubit_cliffords()
        gate_method = {'H': 'h', 'S': 's'}
        backend = self._bench_backend()
        shots = max(300, self.shots)
        out = []
        for m in lengths:
            ps = []
            for _ in range(samples):
                qc = QuantumCircuit(1, 1)
                net = np.eye(2, dtype=complex)
                for idx in rng.integers(len(cliffords), size=m):
                    word, mat = cliffords[idx]
                    for g in word:
                        getattr(qc, gate_method[g])(0)
                    net = mat @ net
                    if interleave is not None:
                        gword, gmat = interleave
                        for g in gword:
                            getattr(qc, gate_method[g])(0)
                        net = gmat @ net
                inv = net.conj().T
                ridx = max(range(len(cliffords)),
                           key=lambda i: abs(np.trace(cliffords[i][1].conj().T @ inv)))
                for g in cliffords[ridx][0]:
                    getattr(qc, gate_method[g])(0)
                qc.measure(0, 0)
                counts = backend.run(transpile(qc, backend, optimization_level=0),
                                     shots=shots).result().get_counts()
                ps.append(counts.get('0', 0) / sum(counts.values()))
            out.append(float(np.mean(ps)))
        return out

    @staticmethod
    def _fit_decay(lengths, survival, fit_fn):
        if max(survival) - min(survival) < 0.02:
            return 1.0
        best, _ = fit_fn(
            lambda v: sum((p - (v[0] * min(max(v[1], 0), 1.2) ** mm + v[2])) ** 2
                          for mm, p in zip(lengths, survival)),
            [0.5, 0.99, 0.5], 400, 0.2)
        return min(max(best[1], 0.0), 1.0)

    def cmd_rbint(self, rest: str) -> None:
        """RBINT <gate> [max_length] — interleaved RB for a single-qubit Clifford.

        Runs reference RB and RB with <gate> (H or S) interleaved after each
        random Clifford, then reports the interleaved gate's average fidelity."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: RBINT <H|S> [max_length]")
            return
        g = parts[0].upper()
        if g not in ('H', 'S'):
            self.io.writeln("?RBINT gate must be H or S")
            return
        cliffords = self._single_qubit_cliffords()
        gmat = {'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
                'S': np.array([[1, 0], [0, 1j]], dtype=complex)}[g]
        max_len = int(parts[1]) if len(parts) > 1 else 32
        lengths = []
        m = 1
        while m <= max_len:
            lengths.append(m); m *= 2
        rng = np.random.default_rng(self._seed)
        try:
            ref = self._rb_survival(lengths, 10, None, rng)
            itl = self._rb_survival(lengths, 10, ([g], gmat), rng)
            f_ref = self._fit_decay(lengths, ref, self._nelder_mead)
            f_int = self._fit_decay(lengths, itl, self._nelder_mead)
            gate_fid = 1.0 - (1.0 - f_int / f_ref) / 2.0 if f_ref > 0 else 0.0
            self.io.writeln(f"\n  Interleaved RB for {g}: f_ref={f_ref:.5f}, f_int={f_int:.5f}")
            self.io.writeln(f"  Estimated {g} gate fidelity = {gate_fid:.5f}")
            self.variables['_GATE_FID'] = gate_fid
        except Exception as e:
            self.io.writeln(f"?RBINT ERROR: {e}")

    def cmd_mirror(self, rest: str) -> None:
        """MIRROR [max_length] [samples] — mirror-circuit benchmarking.

        Applies a random Clifford layer sequence then its exact inverse and
        measures the probability of returning to |0...0>, which decays with depth
        under noise. Scales beyond RB because it needs no full Clifford group."""
        from qiskit import QuantumCircuit, transpile
        parts = rest.split()
        max_len = int(parts[0]) if len(parts) > 0 else 16
        samples = int(parts[1]) if len(parts) > 1 else 10
        n = self.num_qubits
        rng = np.random.default_rng(self._seed)
        backend = self._bench_backend()
        shots = max(300, self.shots)
        lengths = []
        m = 1
        while m <= max_len:
            lengths.append(m); m *= 2
        try:
            self.io.writeln(f"\n  Mirror benchmarking ({n} qubits):")
            self.io.writeln(f"  {'length':>8}  {'return P':>9}")
            for m in lengths:
                ps = []
                for _ in range(samples):
                    qc = QuantumCircuit(n, n)
                    layers = []
                    for _d in range(m):
                        layer = []
                        for q in range(n):
                            gate = ('h', 's', 'x', 'id')[int(rng.integers(4))]
                            getattr(qc, gate)(q); layer.append((gate, q))
                        for q in range(0, n - 1, 2):
                            qc.cx(q, q + 1); layer.append(('cx', q))
                        layers.append(layer)
                    qc.barrier()
                    qc2 = qc.inverse()
                    full = qc.compose(qc2)
                    full.measure(range(n), range(n))
                    counts = backend.run(transpile(full, backend, optimization_level=0),
                                         shots=shots).result().get_counts()
                    ps.append(counts.get('0' * n, 0) / sum(counts.values()))
                self.io.writeln(f"  {m:>8}  {float(np.mean(ps)):>9.4f}")
        except Exception as e:
            self.io.writeln(f"?MIRROR ERROR: {e}")

    # ── Gate-set-style 1-qubit linear inversion ────────────────────────────

    def cmd_gst(self, rest: str = '') -> None:
        """GST — linear-inversion process estimate of the program's channel (1-2 qubits).

        Prepares an informationally complete product-fiducial set, evolves each
        input through the circuit, measures every Pauli-basis setting, and
        linear-inverts the expectations into the process Pauli Transfer Matrix
        (4x4 for one qubit, 16x16 for two). With a noise model active the
        estimate reflects the noisy channel."""
        n = self.num_qubits
        if n not in (1, 2):
            self.io.writeln("?GST is implemented for 1-2 qubits (set QUBITS 1 or 2)")
            return
        if not self.program:
            self.io.writeln(f"?NOTHING TO CHARACTERIZE — enter a {n}-qubit program")
            return
        import itertools as _it
        from qiskit import QuantumCircuit, transpile
        preps1 = {'Z+': [], 'Z-': [('x',)], 'X+': [('h',)], 'Y+': [('h',), ('s',)]}
        meas1 = {'Z': [], 'X': [('h',)], 'Y': [('sdg',), ('h',)]}
        # Pauli vectors (I, X, Y, Z components) of the fiducial inputs.
        sb = {'Z+': (1, 0, 0, 1), 'Z-': (1, 0, 0, -1),
              'X+': (1, 1, 0, 0), 'Y+': (1, 0, 1, 0)}
        try:
            base, _ = self.build_circuit()
            backend = self._bench_backend()
            shots = max(2000, self.shots)
            prep_names = list(_it.product(preps1, repeat=n))   # index 0 = qubit 0
            set_names = list(_it.product(meas1, repeat=n))
            dim = 4 ** n
            # Pauli index: base-4 digits per qubit (I,X,Y,Z), qubit 0 least
            # significant. Every sub-Pauli of a setting (I on any subset)
            # marginalizes out of the same counts.
            E = np.zeros((len(prep_names), dim))
            E[:, 0] = 1.0
            for pi, pn in enumerate(prep_names):
                for sn in set_names:
                    qc = QuantumCircuit(n, n)
                    for q in range(n):
                        for g in preps1[pn[q]]:
                            getattr(qc, g[0])(q)
                    qc = qc.compose(base)
                    for q in range(n):
                        for g in meas1[sn[q]]:
                            getattr(qc, g[0])(q)
                    qc.measure(range(n), range(n))
                    counts = backend.run(transpile(qc, backend), shots=shots).result().get_counts()
                    tot = sum(counts.values())
                    for mask in range(1, 2 ** n):
                        idx = sum((('X', 'Y', 'Z').index(sn[q]) + 1) * 4 ** q
                                  for q in range(n) if mask >> q & 1)
                        acc = 0.0
                        for bits, c in counts.items():
                            sgn = 1
                            for q in range(n):
                                if mask >> q & 1 and bits[len(bits) - 1 - q] == '1':
                                    sgn = -sgn
                            acc += sgn * c
                        E[pi, idx] = acc / tot
            # Input matrix S (products of fiducial Pauli vectors, qubit 0 as the
            # least-significant kron factor); the PTM solves E = S R^T.
            S = np.zeros((len(prep_names), dim))
            for pi, pn in enumerate(prep_names):
                v = np.array([1.0])
                for q in range(n - 1, -1, -1):
                    v = np.kron(v, np.array(sb[pn[q]], dtype=float))
                S[pi] = v
            R = np.linalg.lstsq(S, E, rcond=None)[0].T

            def _label(idx: int) -> str:
                return ''.join('IXYZ'[(idx >> (2 * q)) & 3]
                               for q in range(n - 1, -1, -1))
            labels = [_label(i) for i in range(dim)]
            self.io.writeln(f"\n  GST process estimate (Pauli Transfer Matrix, "
                            f"rows {'/'.join(labels) if n == 1 else 'II..ZZ'}):")
            if n == 2:
                self.io.writeln("        " + " ".join(f"{l:>7}" for l in labels))
            for i in range(dim):
                self.io.writeln(f"    {labels[i]:>2}  "
                                + ' '.join(f"{R[i, j]:+.3f}" for j in range(dim)))
            d = 2 ** n
            self.io.writeln(f"  Avg gate fidelity to identity: "
                            f"{(np.trace(R) + d) / (d * d + d):.4f}")
        except Exception as e:
            self.io.writeln(f"?GST ERROR: {e}")

    # ── Entanglement measures ──────────────────────────────────────────────

    def cmd_concurrence(self, rest: str = '') -> None:
        """CONCURRENCE [a b] — Wootters concurrence of the two-qubit pair (a, b).

        0 for a product state, 1 for a maximally entangled (Bell) pair."""
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        n = self._active_nqubits
        parts = rest.replace(',', ' ').split()
        a, b = (int(parts[0]), int(parts[1])) if len(parts) >= 2 else (0, 1)
        try:
            from qubasic_core import statetools as st
            if a == b or not (0 <= a < n and 0 <= b < n):
                raise ValueError(f"need two different qubits in 0-{n - 1}")
            c = st.concurrence(st.subsystem_rdm(sv, [a, b], n))
            self.io.writeln(f"  Concurrence C(q{a}, q{b}) = {c:.6f}")
            self.variables['_CONCURRENCE'] = float(c)
        except Exception as e:
            self.io.writeln(f"?CONCURRENCE ERROR: {e}")

    def cmd_negativity(self, rest: str = '') -> None:
        """NEGATIVITY [qubits] — entanglement negativity across a bipartition.

        Partial-transposes subsystem A (the listed qubits, default qubit 0) and
        returns N = (||rho^{T_A}||_1 - 1) / 2. 0.5 for a Bell pair."""
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        n = self._active_nqubits
        A = [int(x) for x in rest.replace(',', ' ').split()] if rest.strip() else [0]
        try:
            from qubasic_core import statetools as st
            for q in A:
                if q < 0 or q >= n:
                    raise ValueError(f"qubit {q} out of range (0-{n - 1})")
            # A pure state's negativity follows from its Schmidt coefficients
            # across A | rest, so the 4^n partial transpose is never built.
            neg, logneg = st.negativity_pure(st.schmidt_probs(sv, A, n))
            self.io.writeln(f"  Negativity (A={A}) = {neg:.6f}")
            self.io.writeln(f"  Logarithmic negativity = {logneg:.6f}")
            self.variables['_NEGATIVITY'] = float(neg)
        except Exception as e:
            self.io.writeln(f"?NEGATIVITY ERROR: {e}")
