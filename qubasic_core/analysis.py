"""QUBASIC analysis mixin — EXPECT, ENTROPY, DENSITY, BENCH, RAM."""

from __future__ import annotations

import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

from qubasic_core.engine import (
    _estimate_gb, _get_ram_gb,
    OVERHEAD_FACTOR, RAM_BUDGET_FRACTION,
)


class AnalysisMixin:
    """Analysis and introspection commands for QBasicTerminal.

    Requires: TerminalProtocol — uses self.last_sv, self.last_counts,
    self.num_qubits, self.shots, self.sim_method, self.sim_device,
    self.locc, self.locc_mode.
    """

    def _resolve_analysis_target(self, rest: str):
        """Return (sv, n_qubits, remaining_rest) for a state-analysis command.

        In LOCC SPLIT mode a leading register letter selects that register
        (e.g. ``EXPECT A Z 0``); otherwise the active statevector is used.
        """
        parts = rest.split()
        if self.locc_mode and self.locc and not self.locc.joint and parts:
            cand = parts[0].upper()
            if cand in self.locc.names:
                sv, n = self._active_sv_for_reg(cand)
                return sv, n, ' '.join(parts[1:])
        return self._active_sv, self._active_nqubits, rest

    def cmd_expect(self, rest: str) -> None:
        """EXPECT [reg] <pauli> [qubits] — compute expectation value.
        Examples: EXPECT Z 0, EXPECT ZZ 0 1, EXPECT A Z 0 (SPLIT register A)"""
        sv, n, rest = self._resolve_analysis_target(rest)
        if sv is None:
            if self.locc_mode and self.locc and not self.locc.joint:
                self.io.writeln("?SPLIT mode: prefix a register, e.g. EXPECT A Z 0")
            else:
                self.io.writeln("?NO STATE — RUN first")
            return
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: EXPECT [reg] <Z|X|Y|ZZ|...> [qubits]")
            return
        pauli_str = parts[0].upper()
        qubits = [int(q) for q in parts[1:]] if len(parts) > 1 else list(range(len(pauli_str)))
        # Fast numpy path for diagonal (Z/I-only) observables — no qiskit copy.
        if set(pauli_str) <= {'Z', 'I'}:
            svf = np.ascontiguousarray(sv).ravel()
            probs = np.abs(svf) ** 2
            idx = np.arange(svf.size)
            sign = np.ones(svf.size)
            for i, p in enumerate(pauli_str):
                if p == 'Z' and i < len(qubits):
                    sign = sign * np.where(((idx >> qubits[i]) & 1) == 1, -1.0, 1.0)
            val = float(np.sum(probs * sign))
            self.io.writeln(f"  <{pauli_str}> on qubits {qubits} = {val:.6f}")
            return
        try:
            from qiskit.quantum_info import Statevector, SparsePauliOp
            sv_q = Statevector(np.ascontiguousarray(sv).ravel())
            full_pauli = ['I'] * n
            for i, p in enumerate(pauli_str):
                if i < len(qubits):
                    full_pauli[n - 1 - qubits[i]] = p
            op = SparsePauliOp(''.join(full_pauli))
            val = sv_q.expectation_value(op)
            self.io.writeln(f"  <{pauli_str}> on qubits {qubits} = {val.real:.6f}")
        except Exception as e:
            self.io.writeln(f"?EXPECT ERROR: {e}")

    def cmd_entropy(self, rest: str = '') -> None:
        """ENTROPY [reg] [qubits] — entanglement entropy of qubits vs rest.
        Examples: ENTROPY 0 | ENTROPY 0 1 | ENTROPY A 0 (SPLIT register A)"""
        sv, n, rest = self._resolve_analysis_target(rest)
        if sv is None:
            if self.locc_mode and self.locc and not self.locc.joint:
                self.io.writeln("?SPLIT mode: prefix a register, e.g. ENTROPY A 0")
            else:
                self.io.writeln("?NO STATE — RUN first")
            return
        if rest.strip():
            partition_a = [int(q) for q in rest.replace(',', ' ').split() if q.strip()]
        else:
            partition_a = [0]
        try:
            from qiskit.quantum_info import Statevector, entropy, partial_trace
            sv_obj = Statevector(np.ascontiguousarray(sv).ravel())
            keep = partition_a
            rho_a = partial_trace(sv_obj, [q for q in range(n) if q not in keep])
            ent = entropy(rho_a, base=2)
            self.io.writeln(f"  Entanglement entropy S(A) = {ent:.6f} bits")
            self.io.writeln(f"  Partition A: qubits {partition_a}")
            self.io.writeln(f"  Partition B: qubits {[q for q in range(n) if q not in partition_a]}")
            if ent < 0.01:
                self.io.writeln(f"  -> Qubits are separable (product state)")
            elif abs(ent - len(partition_a)) < 0.01:
                self.io.writeln(f"  -> Maximally entangled")
        except Exception as e:
            self.io.writeln(f"?ENTROPY ERROR: {e}")

    def cmd_density(self, rest: str = '') -> None:
        """Show density matrix (DENSITY [reg]); summarizes for large systems."""
        # A mixed state set via SET_DENSITY has no statevector; solve its
        # density matrix on demand (cached) from the captured circuit.
        dm = getattr(self, '_last_density', None)
        if dm is None and getattr(self, '_last_density_qc', None) is not None:
            dm = self._density_from_qc(self._last_density_qc)
            self._last_density = dm
        if dm is not None and not rest.strip():
            rho = np.ascontiguousarray(dm)
            dim = rho.shape[0]
            self.io.writeln(f"\n  Density matrix ({dim}x{dim}):\n")
            if dim <= 16:
                for i in range(dim):
                    row = []
                    for j in range(dim):
                        v = complex(rho[i, j])
                        if abs(v.imag) < 1e-6:
                            row.append(f"{v.real:7.3f}")
                        else:
                            row.append(f"{v.real:+.2f}{v.imag:+.2f}j")
                    self.io.writeln(f"    {'  '.join(row)}")
            else:
                self.io.writeln(f"    ({dim}x{dim}, too large to display)")
            self.io.writeln(f"\n  Purity: {float(np.real(np.trace(rho @ rho))):.6f}")
            self.io.writeln('')
            return
        sv, n, rest = self._resolve_analysis_target(rest)
        if sv is None:
            if self.locc_mode and self.locc and not self.locc.joint:
                self.io.writeln("?SPLIT mode: prefix a register, e.g. DENSITY A")
            else:
                self.io.writeln("?NO STATE — RUN first")
            return
        sv = np.ascontiguousarray(sv).ravel()
        dim = 2**n
        if dim > 16:
            # |psi><psi| is rank-1, so purity is ||psi||^4 and the von Neumann
            # entropy is 0; report directly instead of building a 2^n x 2^n matrix.
            norm2 = float(np.vdot(sv, sv).real)
            self.io.writeln(f"  Density matrix: {dim}x{dim} (too large to display)")
            self.io.writeln(f"  Pure state |psi><psi|: purity {norm2 ** 2:.6f}")
            self.io.writeln(f"  Von Neumann entropy: 0.000000 bits")
            self.io.writeln(f"  (use ENTROPY <qubits> for reduced-state entanglement)")
            return
        rho = np.outer(sv, sv.conj())
        self.io.writeln(f"\n  Density matrix ({dim}x{dim}):\n")
        for i in range(dim):
            row = []
            for j in range(dim):
                v = rho[i, j]
                if abs(v.imag) < 1e-6:
                    row.append(f"{v.real:7.3f}")
                else:
                    row.append(f"{v.real:+.2f}{v.imag:+.2f}j")
            self.io.writeln(f"    {'  '.join(row)}")
        self.io.writeln(f"\n  Purity: {np.real(np.trace(rho @ rho)):.6f}")
        self.io.writeln('')

    def cmd_bench(self, rest: str = '') -> None:
        """BENCH [n1 n2 ...] — benchmark simulation at various qubit counts."""
        if rest.strip():
            qubit_counts = [int(x) for x in rest.replace(',', ' ').split() if x.strip()]
        else:
            qubit_counts = [4, 8, 12, 16, 20, 24, 28]
        self.io.writeln("\n  Benchmark (H + CX chain + measure):")
        try:
            import psutil
            has_psutil = True
        except ImportError:
            has_psutil = False
        header_mem = '  mem_gb' if has_psutil else ''
        self.io.writeln(f"  {'qubits':>8}  {'method':>20}  {'time':>8}{header_mem}")
        for n in qubit_counts:
            qc = QuantumCircuit(n)
            for i in range(n):
                qc.h(i)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            qc.measure_all()
            method = 'stabilizer' if n > 28 else 'automatic'
            backend = AerSimulator(method=method)
            t0 = time.time()
            try:
                result = backend.run(transpile(qc, backend), shots=100).result()
                dt = time.time() - t0
                mem_str = f"  {psutil.virtual_memory().used / 1e9:>7.1f}" if has_psutil else ""
                self.io.writeln(f"  {n:>8}  {method:>20}  {dt:>7.2f}s{mem_str}")
            except Exception as e:
                self.io.writeln(f"  {n:>8}  {method:>20}  FAILED: {e}")
        self.io.writeln('')

    def cmd_consistency(self, rest: str = '') -> None:
        """CONSISTENCY — cross-check histogram, SV, density, Bloch, and EXPECT."""
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        n = self._active_nqubits
        sv = np.ascontiguousarray(sv).ravel()
        checks: list = []

        # 1. SV normalization
        norm = float(np.sum(np.abs(sv)**2))
        ok = abs(norm - 1.0) < 1e-6
        checks.append(('SV norm == 1', ok, f"{norm:.8f}"))

        # 2. Density matrix purity
        rho = np.outer(sv, sv.conj())
        purity = float(np.real(np.trace(rho @ rho)))
        ok2 = purity <= 1.0 + 1e-6
        checks.append(('Purity <= 1', ok2, f"{purity:.8f}"))

        # 3. Bloch vector length <= 1 for each qubit
        bloch_ok = True
        for q in range(min(n, 8)):
            x, y, z = self._bloch_vector(sv, q, n)
            r = (x**2 + y**2 + z**2) ** 0.5
            if r > 1.0 + 1e-4:
                bloch_ok = False
                break
        checks.append(('Bloch |r| <= 1', bloch_ok, ''))

        # 4. EXPECT Z on qubit 0 matches P(0) - P(1)
        try:
            from qiskit.quantum_info import Statevector, SparsePauliOp
            sv_q = Statevector(sv)
            pauli_z = ['I'] * n
            pauli_z[n - 1] = 'Z'
            op = SparsePauliOp(''.join(pauli_z))
            ez = float(sv_q.expectation_value(op).real)
            # Compare with direct calculation
            sv_t = sv.reshape([2] * n)
            ax = n - 1
            t0 = np.moveaxis(sv_t, ax, 0)[0].flatten()
            t1 = np.moveaxis(sv_t, ax, 0)[1].flatten()
            p0 = float(np.sum(np.abs(t0)**2))
            p1 = float(np.sum(np.abs(t1)**2))
            ez_direct = p0 - p1
            ok4 = abs(ez - ez_direct) < 1e-6
            checks.append(('<Z> consistent', ok4, f"qiskit={ez:.6f} direct={ez_direct:.6f}"))
        except Exception as _e:
            checks.append(('<Z> consistent', None, f"skip: {_e}"))

        # 5. Histogram vs SV (if counts available)
        if self.last_counts:
            total = sum(self.last_counts.values())
            hist_p0 = self.last_counts.get('0' * n, 0) / total
            sv_p0 = float(np.abs(sv[0])**2)
            # Loose check: histogram is statistical, allow 10% deviation
            ok5 = abs(hist_p0 - sv_p0) < 0.15 or total < 50
            checks.append(('Hist~SV P(|0>)', ok5,
                          f"hist={hist_p0:.3f} sv={sv_p0:.3f}"))

        self.io.writeln(f"\n  Consistency checks ({n} qubits):")
        all_pass = True
        for name, ok, detail in checks:
            if ok is None:
                status = 'SKIP'
            elif ok:
                status = 'PASS'
            else:
                status = 'FAIL'
                all_pass = False
            extra = f"  {detail}" if detail else ""
            self.io.writeln(f"    {name:25s} {status}{extra}")
        self.io.writeln(f"\n  {'ALL CONSISTENT' if all_pass else 'INCONSISTENCY DETECTED'}")

    def cmd_fidelity(self, rest: str) -> None:
        """FIDELITY <target> — state fidelity |<target|psi>|^2 of the current state.

        <target> is a named state (|0>, |1>, |+>, |->, |BELL>, |GHZ>, |GHZ3>,
        |GHZ4>, |W>, |W3>) or an explicit amplitude list like [0.707, 0, 0, 0.707].
        """
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        n = self._active_nqubits
        target = rest.strip()
        if not target:
            self.io.writeln("?USAGE: FIDELITY <named state | [amplitudes]>")
            return
        dim = 2 ** n
        try:
            named = {'|0>', '|1>', '|+>', '|->', '|BELL>', '|GHZ>',
                     '|GHZ3>', '|GHZ4>', '|W>', '|W3>'}
            if target.upper() in named:
                from qubasic_core.terminal import _resolve_named_state
                tvec = _resolve_named_state(target.upper(), n)
            else:
                tvec = np.array(self._parse_matrix(target), dtype=complex).ravel()
            if len(tvec) != dim:
                raise ValueError(f"target length {len(tvec)} != 2^{n} = {dim}")
            tnorm = np.linalg.norm(tvec)
            if tnorm > 1e-12:
                tvec = tvec / tnorm
            psi = np.ascontiguousarray(sv).ravel()
            pnorm = np.linalg.norm(psi)
            if pnorm > 1e-12:
                psi = psi / pnorm
            fid = float(np.abs(np.vdot(tvec, psi)) ** 2)
            self.io.writeln(f"  Fidelity F = |<target|psi>|^2 = {fid:.6f}")
            if fid > 0.999:
                self.io.writeln(f"  -> states match")
            elif fid < 0.001:
                self.io.writeln(f"  -> orthogonal")
        except Exception as e:
            self.io.writeln(f"?FIDELITY ERROR: {e}")

    def cmd_tomography(self, rest: str = '') -> None:
        """TOMOGRAPHY [shots] — reconstruct the density matrix from Pauli expectations.

        Builds rho = (1/2^n) sum_P <P> P over all n-qubit Pauli strings P. With no
        argument the expectations are exact; with a shot count each <P> is estimated
        from that many simulated measurements (statistical tomography). Limited to
        <= 3 qubits (3^n Pauli settings)."""
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        n = self._active_nqubits
        if n > 3:
            self.io.writeln(f"?TOMOGRAPHY limited to 3 qubits (have {n}); use ENTROPY/EXPECT instead")
            return
        import itertools
        shots = int(rest.strip()) if rest.strip() else 0
        psi = np.ascontiguousarray(sv).ravel()
        psi = psi / (np.linalg.norm(psi) or 1.0)
        paulis = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        }
        dim = 2 ** n
        rho = np.zeros((dim, dim), dtype=complex)
        for combo in itertools.product('IXYZ', repeat=n):
            op = np.array([[1]], dtype=complex)
            for p in combo:
                op = np.kron(op, paulis[p])
            exp = float(np.real(np.vdot(psi, op @ psi)))
            if shots and any(p != 'I' for p in combo):
                # Simulate finite-shot estimation: outcome +-1 with p=(1+exp)/2.
                p_plus = min(1.0, max(0.0, (1 + exp) / 2))
                hits = np.random.binomial(shots, p_plus)
                exp = (2 * hits - shots) / shots
            rho += exp * op
        rho /= dim
        self.io.writeln(f"\n  Reconstructed density matrix ({dim}x{dim}, "
                        f"{'exact' if not shots else f'{shots} shots/basis'}):")
        for i in range(dim):
            row = '  '.join(
                (f"{rho[i, j].real:+.3f}" if abs(rho[i, j].imag) < 1e-6
                 else f"{rho[i, j].real:+.2f}{rho[i, j].imag:+.2f}j")
                for j in range(dim))
            self.io.writeln(f"    {row}")
        purity = float(np.real(np.trace(rho @ rho)))
        fid = float(np.real(np.vdot(psi, rho @ psi)))
        self.io.writeln(f"  Purity Tr(rho^2) = {purity:.6f}")
        self.io.writeln(f"  Fidelity to the simulated pure state = {fid:.6f}")

    def cmd_ptomography(self, rest: str = '') -> None:
        """PTOMOGRAPHY — reconstruct the circuit's process as a Pauli Transfer Matrix.

        Builds the channel of the current program (the unitary, or the full
        noisy channel when a noise model is active) and reports its PTM, where
        R[i,j] = (1/d) Tr(P_i E(P_j)) over Pauli operators. Trace-preserving and
        unital flags and the average gate fidelity to the identity are shown.
        Limited to <= 2 qubits (the process matrix is 4^n x 4^n)."""
        if not self.program:
            self.io.writeln("?NOTHING TO CHARACTERIZE — enter a program first")
            return
        n = self.num_qubits
        if n > 2:
            self.io.writeln(f"?PTOMOGRAPHY limited to 2 qubits (4^n process matrix); have {n}")
            return
        import itertools
        try:
            qc, _ = self.build_circuit()
            from qiskit.quantum_info import Operator, SuperOp, PTM
            if self._noise_model is not None:
                from qiskit_aer import AerSimulator
                from qiskit import transpile
                qc2 = qc.copy()
                qc2.save_superop()
                b = AerSimulator(method='superop', noise_model=self._noise_model)
                res = b.run(transpile(qc2, b, optimization_level=self._transpile_opt_level)).result()
                ptm = PTM(SuperOp(np.asarray(res.data()['superop'])))
                src = 'noisy channel'
            else:
                ptm = PTM(Operator(qc))
                src = 'unitary'
            R = np.real(np.asarray(ptm.data))
            labels = [''.join(p) for p in itertools.product('IXYZ', repeat=n)]
            self.io.writeln(f"\n  Pauli Transfer Matrix ({len(R)}x{len(R)}, {src}):")
            self.io.writeln('         ' + ''.join(f'{lb:>7}' for lb in labels))
            for i, row in enumerate(R):
                cells = ''.join(f'{v:+7.3f}' for v in row)
                self.io.writeln(f"    {labels[i]:>4} {cells}")
            tp = bool(np.allclose(R[0], [1.0] + [0.0] * (len(R) - 1), atol=1e-6))
            unital = bool(np.allclose(R[:, 0], [1.0] + [0.0] * (len(R) - 1), atol=1e-6))
            d = 2 ** n
            f_avg = (float(np.trace(R)) + d) / (d * d + d)   # avg gate fidelity to identity
            self.io.writeln(f"  Trace-preserving: {tp}   Unital: {unital}")
            self.io.writeln(f"  Avg gate fidelity to identity: {f_avg:.6f}")
        except Exception as e:
            self.io.writeln(f"?PTOMOGRAPHY ERROR: {e}")

    def _single_qubit_cliffords(self):
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

        def canon(M):
            for v in M.ravel():
                if abs(v) > 1e-9:
                    M = M / (v / abs(v))   # fix global phase: first nonzero entry real+
                    break
            return tuple(np.round(M, 6).ravel())

        ident = np.eye(2, dtype=complex)
        table = {canon(ident): ([], ident)}
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

    def cmd_rb(self, rest: str = '') -> None:
        """RB [max_length] [samples] — single-qubit randomized benchmarking.

        Runs random Clifford sequences of growing length, each ending with the
        recovery Clifford that inverts the sequence (ideal output |0>), measures
        the survival probability, and fits p(m) = A f^m + B to extract the decay
        f and the error per Clifford (1 - f)/2. Reflects the active noise model
        (with no noise, f ~ 1). Defaults: max_length 16, 10 sequences per length."""
        parts = rest.split()
        max_len = int(parts[0]) if len(parts) > 0 else 16
        samples = int(parts[1]) if len(parts) > 1 else 10
        cliffords = self._single_qubit_cliffords()
        gate_method = {'H': 'h', 'S': 's'}
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        backend = AerSimulator(noise_model=self._noise_model) if self._noise_model else AerSimulator()
        shots = max(200, self.shots)
        if self._seed is not None:
            np.random.seed(self._seed)
        lengths = []
        m = 1
        while m <= max_len:
            lengths.append(m)
            m *= 2
        self.io.writeln(f"\n  Randomized benchmarking (1 qubit, {samples} sequences/length):")
        self.io.writeln(f"  {'length':>8}  {'survival':>9}")
        survival = []
        for m in lengths:
            ps = []
            for _ in range(samples):
                qc = QuantumCircuit(1, 1)
                net = np.eye(2, dtype=complex)
                for idx in np.random.randint(len(cliffords), size=m):
                    word, mat = cliffords[idx]
                    for g in word:
                        getattr(qc, gate_method[g])(0)
                    net = mat @ net
                inv = net.conj().T
                ridx = max(range(len(cliffords)),
                           key=lambda i: abs(np.trace(cliffords[i][1].conj().T @ inv)))
                for g in cliffords[ridx][0]:
                    getattr(qc, gate_method[g])(0)
                qc.measure(0, 0)
                kw = {'shots': shots}
                if self._seed is not None:
                    kw['seed_simulator'] = self._seed
                # optimization_level=0: the sequence composes to ~identity, so any
                # gate cancellation would strip the gates (and their noise) and
                # flatten the decay. RB needs every gate physically executed.
                counts = backend.run(
                    transpile(qc, backend, optimization_level=0), **kw).result().get_counts()
                ps.append(counts.get('0', 0) / sum(counts.values()))
            mean = float(np.mean(ps))
            survival.append(mean)
            self.io.writeln(f"  {m:>8}  {mean:>9.4f}")

        if max(survival) - min(survival) < 0.02:
            # Flat survival: no measurable decay, so f is underdetermined. The
            # physical reading is no error (f = 1), not whatever the fit drifts to.
            A, f, B = 0.0, 1.0, float(np.mean(survival))
        else:
            def obj(v):
                A, f, B = v
                f = min(max(f, 0.0), 1.2)
                return sum((p - (A * f ** mm + B)) ** 2 for mm, p in zip(lengths, survival))

            best, _ = self._nelder_mead(obj, [0.5, 0.99, 0.5], 400, 0.2)
            A, f, B = best
            f = min(max(f, 0.0), 1.0)
        epc = (1.0 - f) / 2.0
        self.io.writeln(f"  fit p(m) = {A:.3f} * {f:.5f}^m + {B:.3f}")
        self.io.writeln(f"  Decay f = {f:.5f}   error per Clifford = {epc:.5f}")
        self.variables['_RB_F'] = f
        self.variables['_RB_EPC'] = epc

    def cmd_ram(self) -> None:
        """RAM — show memory budget, per-instance cost, and parallelism estimates."""
        ram = _get_ram_gb()
        if not ram:
            self.io.writeln("?psutil not installed — RAM detection unavailable")
            self.io.writeln("  Install: pip install psutil")
            return
        total, avail = ram
        budget = total * RAM_BUDGET_FRACTION

        self.io.writeln(f"\n  System RAM: {total:.1f} GB total, {avail:.1f} GB available")
        self.io.writeln(f"  80% budget: {budget:.1f} GB\n")

        if self.locc_mode and self.locc:
            mode = "JOINT" if self.locc.joint else "SPLIT"
            sizes_str = '+'.join(str(s) for s in self.locc.sizes)
            est, _ = self.locc.mem_gb()
            max_par = int(budget / est) if est > 0 else 0
            self.io.writeln(f"  Current: LOCC {mode} {sizes_str}q")
            self.io.writeln(f"    Per instance: {est:.2f} GB")
            if max_par > 0:
                self.io.writeln(f"    Parallel:     ~{max_par} instances in budget")
        else:
            n = self.num_qubits
            est = _estimate_gb(n)
            max_par = int(budget / est) if est > 0 else 0
            self.io.writeln(f"  Current: {n} qubits")
            self.io.writeln(f"    Per instance: {est:.2f} GB")
            if max_par > 0:
                self.io.writeln(f"    Parallel:     ~{max_par} instances in budget")

        self.io.writeln(f"\n  Qubit scaling (per instance, with {OVERHEAD_FACTOR:.0f}x overhead):")
        for nq in [16, 20, 24, 28, 30, 32]:
            e = _estimate_gb(nq)
            mp = int(budget / e) if e > 0 and budget >= e else 0
            marker = " <--" if nq == self.num_qubits else ""
            if mp > 0:
                self.io.writeln(f"    {nq:>2} qubits: {e:>8.2f} GB  ->  ~{mp:>6,} parallel{marker}")
            else:
                self.io.writeln(f"    {nq:>2} qubits: {e:>8.2f} GB  ->  exceeds budget{marker}")
        self.io.writeln('')
