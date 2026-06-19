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
        it (through the active noise model), and forms the linear XEB fidelity
        F = 2^n * <p_ideal(sampled)> - 1, averaged over trials. F ~ 1 for an ideal
        run and decays toward 0 as noise scrambles the output."""
        from qiskit import transpile
        parts = rest.split()
        n = int(parts[0]) if len(parts) > 0 else min(self.num_qubits, 4)
        depth = int(parts[1]) if len(parts) > 1 else 10
        trials = int(parts[2]) if len(parts) > 2 else 20
        rng = np.random.default_rng(self._seed)
        backend = self._bench_backend()
        shots = max(500, self.shots)
        fids = []
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
                fids.append((2 ** n) * (acc / shots) - 1.0)
            f = float(np.mean(fids))
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
                heavy = {format(i, f'0{n}b') for i, p in enumerate(probs) if p > median}
                qcm = qc.copy(); qcm.measure_all()
                counts = backend.run(transpile(qcm, backend), shots=shots).result().get_counts()
                hop = sum(c for b, c in counts.items() if b in heavy) / shots
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
                            gate = rng.choice(['h', 's', 'x', 'id'])
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
        """GST — 1-qubit linear-inversion process estimate of the program's channel.

        Prepares an informationally complete set of input states, evolves them
        through the circuit, and linear-inverts the measured Pauli expectations
        into the process Pauli Transfer Matrix. With a noise model active the
        estimate reflects the noisy gate."""
        if self.num_qubits != 1:
            self.io.writeln("?GST is implemented for 1 qubit (set QUBITS 1)")
            return
        if not self.program:
            self.io.writeln("?NOTHING TO CHARACTERIZE — enter a 1-qubit program")
            return
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Statevector
        # Input fiducials covering the Bloch sphere and the Pauli measurements.
        preps = {'Z+': [], 'Z-': [('x',)], 'X+': [('h',)], 'Y+': [('h',), ('s',)]}
        paulis = {'X': [('h',)], 'Y': [('sdg',), ('h',)], 'Z': []}
        try:
            base, _ = self.build_circuit()
            backend = self._bench_backend()
            shots = max(2000, self.shots)
            # Estimate <P> for each (prep, pauli) after the gate.
            exp = {}
            for pname, pre in preps.items():
                for mname, post in paulis.items():
                    qc = QuantumCircuit(1, 1)
                    for g in pre:
                        getattr(qc, g[0])(0)
                    qc = qc.compose(base)
                    for g in post:
                        getattr(qc, g[0])(0)
                    qc.measure(0, 0)
                    counts = backend.run(transpile(qc, backend), shots=shots).result().get_counts()
                    p0 = counts.get('0', 0) / sum(counts.values())
                    exp[(pname, mname)] = 2 * p0 - 1
            # Bloch vectors of the prepared inputs (rows: I,X,Y,Z components).
            sb = {'Z+': (0, 0, 1), 'Z-': (0, 0, -1), 'X+': (1, 0, 0), 'Y+': (0, 1, 0)}
            A = np.array([[1, *sb[p]] for p in preps])  # 4x4 prep matrix in Pauli basis
            R = np.zeros((4, 4))
            R[0, 0] = 1.0
            for mi, mname in enumerate(['X', 'Y', 'Z'], start=1):
                b = np.array([exp[(p, mname)] for p in preps])
                R[mi] = np.linalg.lstsq(A, b, rcond=None)[0]
            self.io.writeln("\n  GST process estimate (Pauli Transfer Matrix, rows I/X/Y/Z):")
            for i, lab in enumerate('IXYZ'):
                self.io.writeln(f"    {lab}  " + '  '.join(f"{R[i, j]:+.3f}" for j in range(4)))
            self.io.writeln(f"  Avg gate fidelity to identity: {(np.trace(R) + 2) / 6:.4f}")
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
        parts = rest.split()
        a, b = (int(parts[0]), int(parts[1])) if len(parts) >= 2 else (0, 1)
        try:
            from qiskit.quantum_info import Statevector, partial_trace
            sv_obj = Statevector(np.ascontiguousarray(sv).ravel())
            rho = np.asarray(partial_trace(sv_obj, [q for q in range(n) if q not in (a, b)]).data)
            Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            YY = np.kron(Y, Y)
            rho_tilde = YY @ rho.conj() @ YY
            ev = np.linalg.eigvals(rho @ rho_tilde)
            ev = np.sqrt(np.maximum(0.0, np.sort(np.real(ev))[::-1]))
            c = max(0.0, ev[0] - ev[1] - ev[2] - ev[3])
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
            psi = np.ascontiguousarray(sv).ravel()
            rho = np.outer(psi, psi.conj()).reshape([2] * n + [2] * n)
            # Transpose the row/column indices of subsystem A.
            perm = list(range(2 * n))
            for q in A:
                ax = n - 1 - q
                perm[ax], perm[ax + n] = perm[ax + n], perm[ax]
            rho_ta = np.transpose(rho, perm).reshape(2 ** n, 2 ** n)
            tr_norm = float(np.sum(np.abs(np.linalg.eigvalsh(0.5 * (rho_ta + rho_ta.conj().T)))))
            neg = (tr_norm - 1.0) / 2.0
            self.io.writeln(f"  Negativity (A={A}) = {neg:.6f}")
            self.io.writeln(f"  Logarithmic negativity = {np.log2(tr_norm):.6f}")
            self.variables['_NEGATIVITY'] = float(neg)
        except Exception as e:
            self.io.writeln(f"?NEGATIVITY ERROR: {e}")
