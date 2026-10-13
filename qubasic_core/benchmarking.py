"""QUBASIC benchmarking and characterization (RB, XEB, QVOLUME,
MIRROR, GST, CONCURRENCE, NEGATIVITY) and fault-tolerant resource estimates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from qubasic_core.host import ApplicationsHost
from qubasic_core.simtools import heavy_hex_edges

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# QUBASIC frontier benchmarking and verification.
#
# Cross-entropy benchmarking, quantum volume, interleaved and mirror randomized
# benchmarking, a linear-inversion 1-qubit gate-set tomography, and entanglement
# measures. All offline on Qiskit/Aer/numpy.
#
#   XEB [n] [depth] [trials]   Linear cross-entropy benchmarking fidelity
#   QVOLUME [n]                Quantum volume heavy-output test at width n
#   RBINT <gate> [maxlen]      Interleaved RB: fidelity of a specific Clifford gate
#   MIRROR [maxlen] [samples]  Mirror-circuit benchmarking (scalable)
#   GST                        1-qubit linear-inversion gate-set-style process estimate
#   CONCURRENCE [a b]          Two-qubit concurrence of a qubit pair
#   NEGATIVITY [qubits]        Entanglement negativity across a bipartition
# ---------------------------------------------------------------------------

class BenchmarkingMixin(ApplicationsHost):
    """XEB, quantum volume, RB variants, GST, and entanglement measures."""

    # ── helpers ──────────────────────────────────────────────────────────

    def _bench_backend(self) -> AerSimulator:
        from qiskit_aer import AerSimulator
        return AerSimulator(noise_model=self.noise.model) if self.noise.model else AerSimulator()

    def _random_circuit(self, n: int, depth: int, rng: np.random.Generator) -> QuantumCircuit:
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
                qcm = qc.copy()
                qcm.measure_all()
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
                qcm = qc.copy()
                qcm.measure_all()
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

    def _rb_survival(self, lengths: list[int], samples: int, interleave: tuple[list[str], np.ndarray] | None, rng: np.random.Generator) -> list[float]:
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
    def _fit_decay(lengths: Sequence[int], survival: Sequence[float], fit_fn: Callable[..., tuple[list[float], float]]) -> float:
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
        gmat = {'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
                'S': np.array([[1, 0], [0, 1j]], dtype=complex)}[g]
        max_len = int(parts[1]) if len(parts) > 1 else 32
        lengths = []
        m = 1
        while m <= max_len:
            lengths.append(m)
            m *= 2
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
            lengths.append(m)
            m *= 2
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
                            getattr(qc, gate)(q)
                            layer.append((gate, q))
                        for q in range(0, n - 1, 2):
                            qc.cx(q, q + 1)
                            layer.append(('cx', q))
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
            from qubasic_core import simtools as st
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
            from qubasic_core import simtools as st
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

    def _single_qubit_cliffords(self) -> list[tuple[list[str], np.ndarray]]:
        """Return the 24 single-qubit Cliffords as (gate_word, matrix) pairs.

        Generated by BFS over words in {H, S} (which generate the group up to
        global phase), cached on the instance. The word, applied in order, gives
        the stored matrix, so RB sequences run as native h/s gates that the
        active noise model decorates."""
        cached = getattr(self, '_rb_cliffords', None)
        if cached is not None:
            return cached
        from collections import deque
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        gens = (('H', H), ('S', S))

        def canon(M: np.ndarray) -> tuple[Any, ...]:
            for v in M.ravel():
                if abs(v) > 1e-9:
                    M = M / (v / abs(v))   # fix global phase: first nonzero entry real+
                    break
            return tuple(np.round(M, 6).ravel())

        ident = np.eye(2, dtype=complex)
        table: dict[tuple[Any, ...], tuple[list[str], np.ndarray]] = {canon(ident): ([], ident)}
        queue = deque(table)
        while queue:
            word, M = table[queue.popleft()]
            for gname, G in gens:
                M2 = G @ M
                k2 = canon(M2)
                if k2 not in table:
                    table[k2] = (word + [gname], M2)
                    queue.append(k2)
        self._rb_cliffords = list(table.values())
        return self._rb_cliffords


# ---------------------------------------------------------------------------
# QUBASIC compilation and resource estimation.
#
# Fault-tolerant resource estimation under the surface code, calibrated offline
# device models (per-qubit T1/T2 plus a topology), and a transpiler pass that
# reports circuit optimization. All offline.
#
#   RESOURCES <target_pL> <phys_p>   Estimate surface-code distance and qubit/time cost
#   DEVICE linear|ring|heavyhex [n] [T1us] [T2us]   Load a calibrated device noise model
#   OPTIMIZE [level]                 Transpile the program and report the reduction
# ---------------------------------------------------------------------------

# Surface-code logical error rate per code cycle, p_L = A (p / p_th)^((d+1)/2)
# with A = 0.1 and p_th = 0.01 (Litinski, "A Game of Surface Codes", 2019),
# shared by RESOURCES and the LQUBITS model.
SC_A = 0.1
SC_THRESHOLD = 0.01


def surface_logical_rate(p: float, d: int) -> float:
    """Logical error rate of a distance-d surface code at physical rate p."""
    return SC_A * (p / SC_THRESHOLD) ** ((d + 1) / 2)


class ResourceMixin(ApplicationsHost):
    """Resource estimation, calibrated device models, and circuit optimization."""

    _SC_THRESHOLD = SC_THRESHOLD
    _SC_A = SC_A

    def cmd_resources(self, rest: str) -> None:
        """RESOURCES <target_logical_error> <physical_error> — surface-code distance, physical qubits and runtime for a target logical error rate.

        Finds the smallest surface-code distance d reaching the target logical
        error rate at the given physical error rate, then estimates physical
        qubits (2 d^2 - 1 per logical patch), the program's T-count, T-factory
        overhead, and an approximate code-cycle / runtime figure."""
        parts = rest.split()
        if len(parts) < 2:
            self.io.writeln("?USAGE: RESOURCES <target_logical_error> <physical_error>")
            return
        try:
            target = float(self._eval_with_vars(parts[0], {}))
            p = float(self._eval_with_vars(parts[1], {}))
            if p >= self._SC_THRESHOLD:
                self.io.writeln(f"?physical error {p} is above the surface-code threshold "
                                f"~{self._SC_THRESHOLD}; error correction cannot help")
                return
            # Smallest odd d with A (p/p_th)^((d+1)/2) <= target.
            d = 1
            while surface_logical_rate(p, d) > target and d < 99:
                d += 2
            pL = surface_logical_rate(p, d)
            n_logical = self.num_qubits
            phys_per_logical = 2 * d * d - 1
            data_qubits = n_logical * phys_per_logical
            # T-count and logical depth from the program circuit, if any.
            t_count = depth = 0
            if self.program:
                try:
                    qc, _ = self.build_circuit()
                    ops = qc.count_ops()
                    t_count = ops.get('t', 0) + ops.get('tdg', 0)
                    depth = qc.depth()
                except Exception:
                    t_count = depth = 0
            # 15-to-1 T factory footprint and a parallelism assumption.
            factory_qubits = 15 * phys_per_logical
            n_factories = max(1, min(t_count, 8)) if t_count else 0
            total_qubits = data_qubits + n_factories * factory_qubits
            cycle_us = 1.0   # ~1 microsecond per surface-code cycle (typical SC qubit)
            runtime_us = depth * d * cycle_us + t_count * d * cycle_us
            self.io.writeln("\n  Fault-tolerant resource estimate (surface code):")
            self.io.writeln(f"    target logical error : {target:g}")
            self.io.writeln(f"    physical error       : {p:g}  (threshold ~{self._SC_THRESHOLD})")
            self.io.writeln(f"    code distance        : d = {d}  (achieves p_L ~ {pL:.2e})")
            self.io.writeln(f"    logical qubits       : {n_logical}")
            self.io.writeln(f"    physical / logical   : {phys_per_logical}")
            self.io.writeln(f"    data qubits          : {data_qubits:,}")
            if t_count:
                self.io.writeln(f"    T-count              : {t_count}")
                self.io.writeln(f"    T-factories (15-to-1): {n_factories} x {factory_qubits:,} qubits")
            self.io.writeln(f"    total physical qubits: ~{total_qubits:,}")
            self.io.writeln(f"    est. runtime         : ~{runtime_us:.0f} us "
                            f"({depth} logical depth, ~{cycle_us} us/cycle)")
            self.variables['_FT_DISTANCE'] = d
            self.variables['_FT_QUBITS'] = total_qubits
        except Exception as e:
            self.io.writeln(f"?RESOURCES ERROR: {e}")

    def cmd_device(self, rest: str) -> None:
        """DEVICE <topology> [n] [T1us] [T2us] | DEVICE IBM <device> | DEVICE OFF — load a device model.

        Builds a NoiseModel with per-qubit thermal relaxation (T1/T2 and gate
        times) and a coupling map for the chosen topology (linear, ring, heavyhex,
        all), so RUN simulates a realistic hardware device with no network.
        DEVICE IBM <device> builds Aer's noise model from a real IBM device's
        current calibration (its T1, T2, gate and readout errors) and routes on
        its coupling map, through the saved Qiskit Runtime account. DEVICE OFF
        clears it."""
        parts = rest.split()
        if not parts:
            coupling = 'set' if self._coupling_map else 'none'
            self.io.writeln(f"DEVICE: noise={'on' if self.noise.model else 'off'}, "
                            f"coupling={coupling}")
            return
        if parts[0].upper() in ('OFF', 'NONE'):
            self.noise.model = None
            self.noise.spec = None
            self._device_spec = None
            self._coupling_map = None
            self._device_target = None
            self._circuit_cache_key = None
            self.io.writeln("DEVICE OFF")
            return
        if parts[0].upper() == 'IBM':
            try:
                if len(parts) < 2:
                    raise ValueError("DEVICE IBM needs a device name (QPU LIST shows them)")
                from qiskit_aer.noise import NoiseModel
                backend = self._ibm_service().backend(parts[1])
                self.noise.model = NoiseModel.from_backend(backend)
                self.noise.spec = None
                self._device_target = backend.target
                self._coupling_map = None
                self._basis_gates = None
                self._device_spec = f"IBM {parts[1]}"
                self._circuit_cache_key = None
                self.io.writeln(f"DEVICE IBM {parts[1]}: {backend.num_qubits} qubits; noise from its "
                                f"current calibration, routed on its coupling map")
            except Exception as e:
                self.io.writeln(f"?DEVICE ERROR: {e}")
            return
        topo = parts[0].lower()
        n = int(parts[1]) if len(parts) > 1 else self.num_qubits
        t1 = float(parts[2]) * 1e-6 if len(parts) > 2 else 100e-6
        t2 = float(parts[3]) * 1e-6 if len(parts) > 3 else 80e-6
        try:
            from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
            nm = NoiseModel()
            t_1q, t_2q = 50e-9, 300e-9   # typical gate durations
            from qubasic_core.noise import _1Q as _1q
            from qubasic_core.noise import _2Q as _2q
            for q in range(n):
                nm.add_quantum_error(thermal_relaxation_error(t1, t2, t_1q), _1q, [q])
            err2 = thermal_relaxation_error(t1, t2, t_2q).expand(
                thermal_relaxation_error(t1, t2, t_2q))
            for a in range(n):
                for b in range(n):
                    if a != b:
                        nm.add_quantum_error(err2, _2q, [a, b])
            if topo == 'linear':
                cm = [[i, i + 1] for i in range(n - 1)]
            elif topo == 'ring':
                cm = [[i, (i + 1) % n] for i in range(n)]
            elif topo in ('heavyhex', 'heavy-hex'):
                cm = heavy_hex_edges(n)
            elif topo in ('all', 'full'):
                cm = [[i, j] for i in range(n) for j in range(n) if i < j]
            else:
                raise ValueError(f"unknown topology '{topo}' (linear, ring, heavyhex, all)")
            if n != self.num_qubits:
                self.io.writeln(f"  (device has {n} qubits; QUBITS is {self.num_qubits})")
            # 2-qubit noise only on coupled pairs: routing keeps every 2-qubit
            # gate on an edge of the coupling map.
            if topo not in ('all', 'full'):
                nm = NoiseModel()
                for q in range(n):
                    nm.add_quantum_error(thermal_relaxation_error(t1, t2, t_1q), _1q, [q])
                for a, b in cm:
                    nm.add_quantum_error(err2, _2q, [a, b])
                    nm.add_quantum_error(err2, _2q, [b, a])
            edges = []
            for a, b in cm:
                edges += [[a, b], [b, a]]
            self.noise.model = nm
            self._coupling_map = edges if topo not in ('all', 'full') else None
            self.noise.spec = None
            self._device_spec = f"{topo} {n} {t1 * 1e6:g} {t2 * 1e6:g}"
            self._circuit_cache_key = None
            self.io.writeln(f"DEVICE {topo}: {n} qubits, T1={t1 * 1e6:.0f}us T2={t2 * 1e6:.0f}us, "
                            f"thermal noise + coupling map loaded")
        except Exception as e:
            self.io.writeln(f"?DEVICE ERROR: {e}")

    def cmd_optimize(self, rest: str = '') -> None:
        """OPTIMIZE [level] — transpile the program and report the reduction.

        Runs the Qiskit transpiler at the given optimization level (default 3)
        against the active basis/coupling model and reports the depth and gate
        count before and after."""
        if not self.program:
            self.io.writeln("?NOTHING TO OPTIMIZE — enter a program first")
            return
        level = int(rest.strip()) if rest.strip() else 3
        try:
            from qiskit_aer import AerSimulator
            qc, _ = self.build_circuit()
            backend = AerSimulator()
            opt = self._transpile_routed(qc, backend, level=level)
            self._last_transpiled = opt
            self.io.writeln(f"\n  OPTIMIZE (level {level}):")
            self.io.writeln(f"    before: depth {qc.depth()}, gates {qc.size()}")
            self.io.writeln(f"    after : depth {opt.depth()}, gates {opt.size()}")
            dd = qc.depth() - opt.depth()
            if dd >= 0:
                self.io.writeln(f"    reduced depth by {dd} ({100 * dd / max(1, qc.depth()):.0f}%)")
            else:
                self.io.writeln(f"    increased depth by {-dd} (basis/routing constraints outweigh the gains)")
            self.variables['_OPT_DEPTH'] = opt.depth()
        except Exception as e:
            self.io.writeln(f"?OPTIMIZE ERROR: {e}")


# ---------------------------------------------------------------------------
# IBM quantum hardware through Qiskit Runtime.
#
#   QPU LIST                    the devices the saved account reaches, with queues
#   QPU <device> [SHOTS n]      run the program there and show the histogram
#   QPU JOB <id>                collect a job whose wait was interrupted
#   DEVICE IBM <device>         simulate that device: Aer's noise model from its
#                               current calibration, routed on its coupling map
#
# The account is the one QiskitRuntimeService.save_account stored. A device
# run transpiles the program for the device's native gates and qubits and runs
# it through SamplerV2; it spends the account's QPU time, which the report
# gives. Programs that read measured bits as numbers run live, shot by shot,
# and stay on the simulator.
# ---------------------------------------------------------------------------

def _runtime() -> Any:
    """qiskit_ibm_runtime, or a ValueError saying how to get it."""
    try:
        import qiskit_ibm_runtime
    except ImportError as e:
        raise ValueError("needs qiskit-ibm-runtime (pip install qubasic[ibm]) and an IBM Quantum "
                         "account saved with QiskitRuntimeService.save_account") from e
    return qiskit_ibm_runtime


class HardwareMixin(ApplicationsHost):
    """QPU: programs on IBM quantum hardware (see the module docstring)."""

    def _init_hardware(self) -> None:
        self._ibm: Any = None                       # the QiskitRuntimeService, made on first use
        self._ibm_jobs: list[tuple[str, str]] = []  # (job id, device) of each QPU run

    def _ibm_service(self) -> Any:
        if self._ibm is None:
            self._ibm = _runtime().QiskitRuntimeService()
        return self._ibm

    def cmd_qpu(self, rest: str) -> None:
        """QPU LIST | QPU <device> [SHOTS n] | QPU JOB <id> — run the program on IBM quantum hardware.

        QPU LIST shows the devices the saved Qiskit Runtime account reaches
        and their queues. QPU <device> transpiles the program for the device,
        runs it through SamplerV2 and shows the histogram; Ctrl-C stops the
        wait, the job keeps its place, and QPU JOB <id> collects it later.
        Each run spends the account's QPU time, which the report gives."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: QPU LIST | QPU <device> [SHOTS n] | QPU JOB <id>")
            return
        try:
            word = parts[0].upper()
            if word == 'LIST':
                self._qpu_list()
            elif word == 'JOB':
                if len(parts) < 2:
                    raise ValueError("QPU JOB needs a job id (QPU runs list theirs)")
                self._qpu_collect(self._ibm_service().job(parts[1]))
            else:
                shots = self.shots
                if len(parts) >= 3 and parts[1].upper() == 'SHOTS':
                    shots = int(parts[2])
                self._qpu_run(parts[0], shots)
        except KeyboardInterrupt:
            self.io.writeln("\n?STOPPED WAITING: the job keeps its place; QPU JOB <id> collects it")
        except Exception as e:
            self.io.writeln(f"?QPU ERROR: {e}")

    def _qpu_list(self) -> None:
        self.io.writeln("\n  device            qubits  queue  status")
        for b in self._ibm_service().backends():
            st = b.status()
            state = 'online' if st.operational else 'offline'
            self.io.writeln(f"  {b.name:<16}  {b.num_qubits:6d}  {st.pending_jobs:5d}  {state}")

    def _qpu_run(self, name: str, shots: int) -> None:
        if not self.program:
            raise ValueError("NOTHING TO RUN")
        if self._needs_live(sorted(self.program)):
            raise ValueError("this program reads measured bits as numbers and runs shot by shot, "
                             "which a device cannot")
        qc, has_measure = self.build_circuit()
        if not has_measure:
            raise ValueError("the program has no MEASURE, so the device would return nothing")
        self._add_final_measurement(qc)
        backend = self._ibm_service().backend(name)
        import warnings

        from qiskit.transpiler import generate_preset_pass_manager
        with warnings.catch_warnings():                 # a DELAY rounded to the device's clock cycle
            warnings.filterwarnings('ignore', message=r'Duration is rounded', category=UserWarning)
            isa = generate_preset_pass_manager(backend=backend, optimization_level=1).run(qc)
        two = sum(1 for inst in isa.data if inst.operation.num_qubits == 2)
        job = _runtime().SamplerV2(mode=backend).run([isa], shots=shots)
        self._ibm_jobs.append((job.job_id(), name))
        self.io.writeln(f"QPU {name}: job {job.job_id()} queued, {shots} shots, depth {isa.depth()}, "
                        f"{two} two-qubit gates on the device")
        self._qpu_collect(job)

    def _qpu_collect(self, job: Any) -> None:
        """Wait for a job, then keep and show its histogram and QPU time."""
        data = job.result()[0].data
        registers = list(data.keys())
        name = next((r for r in ('sub', 'meas') if r in registers), registers[0])
        counts = dict(getattr(data, name).get_counts())
        self.last_counts = counts
        seconds = None
        try:
            seconds = float(job.usage())
        except Exception:
            try:
                seconds = float(job.metrics()['usage']['quantum_seconds'])
            except Exception:
                pass
        spent = f", {seconds:.1f} s of QPU time" if seconds is not None else ""
        device = job.backend().name if callable(getattr(job, 'backend', None)) else '?'
        self.io.writeln(f"\nQPU job {job.job_id()} on {device}: {sum(counts.values())} shots{spent}")
        self.print_histogram(counts)
