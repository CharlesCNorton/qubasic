"""QUBASIC analysis and interchange: expectation values,
entanglement, tomography, OpenQASM and CSV export, PNG pictures, the assist
commands and the demos.
"""

from __future__ import annotations

import difflib
import os
import random
import sys
import time
from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qubasic_core.engine import RAM_BUDGET_FRACTION
from qubasic_core.host import PresentationHost
from qubasic_core.parser import GATE_ALIASES, GATE_TABLE, parse_stmt

# ---------------------------------------------------------------------------
# QUBASIC analysis mixin — EXPECT, ENTROPY, DENSITY, BENCH, RAM.
# ---------------------------------------------------------------------------

class AnalysisMixin(PresentationHost):
    """Analysis and introspection commands for QBasicTerminal."""

    def _resolve_analysis_target(self, rest: str) -> tuple[np.ndarray | None, int, str]:
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
        """EXPECT [reg] pauli [qubits] — expectation value of a Pauli string such as ZZ 0 1 (reg: a LOCC SPLIT register)."""
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
        if set(pauli_str) - set('IXYZ'):
            self.io.writeln("?EXPECT: observable must be over I, X, Y, Z")
            return
        if len(pauli_str) != len(qubits):
            self.io.writeln(f"?EXPECT: {pauli_str} needs {len(pauli_str)} qubit(s), got {len(qubits)}")
            return
        for q in qubits:
            if q < 0 or q >= n:
                self.io.writeln(f"?EXPECT: qubit {q} out of range (0-{n - 1})")
                return
        # Diagonal (Z/I-only) observables: a chunked parity sum, no copies.
        if set(pauli_str) <= {'Z', 'I'}:
            from qubasic_core.simtools import z_expectation
            val = z_expectation(sv, [q for p, q in zip(pauli_str, qubits) if p == 'Z'])
            self.io.writeln(f"  <{pauli_str}> on qubits {qubits} = {val:.6f}")
            return
        try:
            from qiskit.quantum_info import SparsePauliOp, Statevector
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
        """ENTROPY [reg] [qubits] — entanglement entropy in bits of the qubits against the rest (reg: a LOCC SPLIT register)."""
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
            from qubasic_core.simtools import entanglement_entropy
            for q in partition_a:
                if q < 0 or q >= n:
                    raise ValueError(f"qubit {q} out of range (0-{n - 1})")
            ent = entanglement_entropy(sv, partition_a, n)
            self.io.writeln(f"  Entanglement entropy S(A) = {ent:.6f} bits")
            self.io.writeln(f"  Partition A: qubits {partition_a}")
            self.io.writeln(f"  Partition B: qubits {[q for q in range(n) if q not in partition_a]}")
            if ent < 0.01:
                self.io.writeln("  -> Qubits are separable (product state)")
            elif abs(ent - len(partition_a)) < 0.01:
                self.io.writeln("  -> Maximally entangled")
        except Exception as e:
            self.io.writeln(f"?ENTROPY ERROR: {e}")

    def cmd_density(self, rest: str = '') -> None:
        """DENSITY [reg] — the density matrix of the current state (a summary for large registers)."""
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
            self.io.writeln("  Von Neumann entropy: 0.000000 bits")
            self.io.writeln("  (use ENTROPY <qubits> for reduced-state entanglement)")
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
        """BENCH [n1 n2 ...] — time a statevector run (H, a CX chain and T on every qubit, 100 shots) at each qubit count with its memory; a count that does not fit in available memory is skipped."""
        from qubasic_core import simtools as cap
        if rest.strip():
            qubit_counts = [int(x) for x in rest.replace(',', ' ').split() if x.strip()]
        else:
            qubit_counts = [4, 8, 12, 16, 20, 24, 28]
        prec = self.sim_precision
        self.io.writeln(f"\n  Benchmark (H, CX chain and T on every qubit; statevector, {prec} precision, "
                        f"100 shots):")
        self.io.writeln(f"  {'qubits':>8}  {'time':>8}  {'memory':>10}")
        for n in qubit_counts:
            peak = cap.aer_peak_bytes(n, 'statevector', prec)
            if not cap.fits(peak):
                self.io.writeln(f"  {n:>8}  skipped: needs ~{cap.fmt_gb(peak)}, "
                                f"{cap.fmt_gb(cap.available_bytes())} available")
                continue
            qc = QuantumCircuit(n)
            for i in range(n):
                qc.h(i)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            for i in range(n):
                qc.t(i)
            qc.measure_all()
            backend = AerSimulator(method='statevector', precision=prec)
            t0 = time.time()
            try:
                backend.run(transpile(qc, backend), shots=100).result()
            except Exception as e:
                self.io.writeln(f"  {n:>8}  FAILED: {e}")
                continue
            self.io.writeln(f"  {n:>8}  {time.time() - t0:>7.2f}s  {cap.fmt_gb(peak):>10}")
        self.io.writeln('')

    def cmd_consistency(self, rest: str = '') -> None:
        """CONSISTENCY — cross-check the state, its reduced qubit states, two
        independent <Z> computations, and the histogram against the state's
        distribution. Memory stays at the state plus one scratch block."""
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        from qubasic_core import simtools as st
        n = self._active_nqubits
        sv = np.asarray(sv).reshape(-1)
        checks: list = []

        # 1. Normalization (a pure state's purity is ||psi||^4, so this also
        #    covers Tr(rho^2) = 1 without building rho).
        norm = st.norm2(sv)
        checks.append(('SV norm == 1', abs(norm - 1.0) < 1e-6, f"{norm:.8f}"))

        # 2. Every reduced qubit state is a valid density matrix (Bloch |r| <= 1,
        #    purity in [1/2, 1]); all qubits for up to 2^30 work, else the first 8.
        span = n if (n << n) <= (1 << 30) else min(n, 8)
        worst_r, worst_q = 0.0, 0
        for q in range(span):
            x, y, z = st.bloch(sv, q, n)
            r = (x * x + y * y + z * z) ** 0.5
            if r > worst_r:
                worst_r, worst_q = r, q
        checks.append((f'Bloch |r| <= 1 ({span} qubits)', worst_r <= 1.0 + 1e-6,
                       f"max {worst_r:.6f} on q{worst_q}"))

        # 3. <Z_0> two ways: the Bloch z of qubit 0, and the parity sum.
        try:
            ez_rdm = st.bloch(sv, 0, n)[2]
            ez_sum = st.z_expectation(sv, [0])
            checks.append(('<Z> consistent', abs(ez_rdm - ez_sum) < 1e-6,
                           f"reduced-state={ez_rdm:.6f} parity-sum={ez_sum:.6f}"))
        except Exception as _e:
            checks.append(('<Z> consistent', None, f"skip: {_e}"))

        # 4. Histogram vs the state's distribution (chi-square), when the counts
        #    cover the whole register of a noiseless run.
        counts = self.last_counts
        noisy = self._noise_model is not None or bool(self._qubit_noise)
        if counts and not self.locc_mode:
            nbits = len(next(iter(counts)))
            if noisy:
                checks.append(('Histogram ~ state', None,
                               "skip: the state is one noisy trajectory, the histogram averages"))
            elif nbits != n:
                checks.append(('Histogram ~ state', None, "skip: partial MEASURE subset"))
            else:
                ok5, detail = self._histogram_agrees(sv, counts, n)
                checks.append(('Histogram ~ state', ok5, detail))

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

    @staticmethod
    def _histogram_agrees(sv: np.ndarray, counts: dict, n: int) -> tuple[bool, str]:
        """Chi-square goodness of fit of counts to |psi|^2 (outcomes with fewer
        than 5 expected counts pooled); passes when p > 1e-4."""
        total = sum(counts.values())
        keys = list(counts)
        obs = np.array([counts[k] for k in keys], dtype=float)
        probs = np.array([abs(complex(sv[int(k, 2)])) ** 2 for k in keys])
        observed_mass = float(probs.sum())
        if any(p < 1e-12 for p in probs):
            return False, "an outcome the state forbids was measured"
        exp = probs * total
        big = exp >= 5
        o_cells = list(obs[big])
        e_cells = list(exp[big])
        rest_obs = float(obs[~big].sum())
        rest_exp = float(exp[~big].sum()) + (1.0 - observed_mass) * total
        if rest_exp > 0:
            o_cells.append(rest_obs)
            e_cells.append(rest_exp)
        if len(o_cells) < 2:
            return True, "one outcome"
        o_arr, e_arr = np.array(o_cells), np.array(e_cells)
        chi2 = float(np.sum((o_arr - e_arr) ** 2 / e_arr))
        dof = len(o_cells) - 1
        try:
            from scipy.stats import chi2 as _chi2
            p = float(_chi2.sf(chi2, dof))
        except Exception:
            # Wilson-Hilferty normal approximation of the chi-square tail.
            z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / np.sqrt(2 / (9 * dof))
            from math import erfc, sqrt
            p = 0.5 * erfc(z / sqrt(2))
        return p > 1e-4, f"chi2={chi2:.1f} dof={dof} p={p:.3g}"

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
                from qubasic_core.simulator import _resolve_named_state
                tvec = _resolve_named_state(target.upper(), n)
            else:
                tvec = np.array(self._parse_matrix(target), dtype=np.complex128).ravel()
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
                self.io.writeln("  -> states match")
            elif fid < 0.001:
                self.io.writeln("  -> orthogonal")
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
        psi: np.ndarray = np.ascontiguousarray(sv).ravel()
        psi = psi / (np.linalg.norm(psi) or 1.0)
        paulis = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        }
        dim = 2 ** n
        rho = np.zeros((dim, dim), dtype=complex)
        rng = np.random.default_rng(self._seed)
        for combo in itertools.product('IXYZ', repeat=n):
            op = np.array([[1]], dtype=complex)
            for p in combo:
                op = np.kron(op, paulis[p])
            exp = float(np.real(np.vdot(psi, op @ psi)))
            if shots and any(p != 'I' for p in combo):
                # Simulate finite-shot estimation: outcome +-1 with p=(1+exp)/2.
                p_plus = min(1.0, max(0.0, (1 + exp) / 2))
                hits = rng.binomial(shots, p_plus)
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
            from qiskit.quantum_info import PTM, Operator, SuperOp
            if self._noise_model is not None:
                from qiskit import transpile
                from qiskit_aer import AerSimulator
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
            def obj(v: Sequence[float]) -> float:
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
        """RAM — memory now available, what the current configuration costs,
        and the widest statevector that fits at each precision."""
        from qubasic_core import simtools as cap
        total, avail = cap.total_bytes(), cap.available_bytes()
        if total is None or avail is None:
            self.io.writeln("?psutil not installed — RAM detection unavailable")
            self.io.writeln("  Install: pip install psutil")
            return
        usable = avail * cap.HEADROOM
        budget = total * RAM_BUDGET_FRACTION
        self.io.writeln(f"\n  System RAM: {cap.fmt_gb(total)} total, {cap.fmt_gb(avail)} available "
                        f"(one run may use {cap.HEADROOM:.0%}: {cap.fmt_gb(usable)})")

        prec = self.sim_precision
        if self.locc_mode and self.locc:
            mode = "JOINT" if self.locc.joint else "SPLIT"
            sizes_str = '+'.join(str(s) for s in self.locc.sizes)
            est, _ = self.locc.mem_gb()
            snap, _ = self.locc.mem_gb(snapshot=True)
            self.io.writeln(f"\n  Current: LOCC {mode} {sizes_str}q ({self.locc.precision} precision, "
                            f"in-place numpy engine)")
            self.io.writeln(f"    state + scratch       {est:.2f} GB")
            self.io.writeln(f"    with SEND snapshot    {snap:.2f} GB")
            par = int(budget / (est * 1e9)) if est > 0 else 0
            if par > 0:
                self.io.writeln(f"    Max parallel instances in 80% budget: ~{par}")
        else:
            n = self.num_qubits
            counts = cap.aer_peak_bytes(n, 'statevector', prec)
            mid = cap.aer_peak_bytes(n, 'statevector', prec, 'mid')
            self.io.writeln(f"\n  Current: {n} qubits, METHOD {self.sim_method}, PRECISION {prec}")
            if self.sim_method in ('stabilizer', 'matrix_product_state', 'extended_stabilizer'):
                self.io.writeln("    memory is set by the circuit's structure, not 2^n")
            else:
                self.io.writeln(f"    statevector run            {cap.fmt_gb(counts)}"
                                f"{'' if cap.fits(counts, avail) else '  (does not fit now)'}")
                self.io.writeln(f"    with the state kept        {cap.fmt_gb(mid)}"
                                f"{'' if cap.fits(mid, avail) else '  (state recomputed on use)'}")
                if counts and n >= 16:
                    par = int(budget // counts)
                    if par > 0:
                        self.io.writeln(f"    Max parallel instances in 80% budget: ~{par}")

        self.io.writeln("\n  Statevector width (peak per run: counts only / state kept in the run):")
        self.io.writeln(f"    {'qubits':>6}  {'double':>22}  {'single':>22}")
        widest = {p: cap.max_qubits('statevector', p, available=avail) for p in ('double', 'single')}
        known = [w for w in widest.values() if w is not None] or [16]
        lo = max(16, min(known) - 3)
        for nq in range(lo, max(known) + 2):
            cells = []
            for p in ('double', 'single'):
                a = cap.aer_peak_bytes(nq, 'statevector', p)
                b = cap.aer_peak_bytes(nq, 'statevector', p, 'mid')
                mark = ' ' if cap.fits(a, avail) else '*'
                cells.append(f"{cap.fmt_gb(a):>9} / {cap.fmt_gb(b):>9}{mark}")
            here = "  <--" if nq == self.num_qubits else ""
            self.io.writeln(f"    {nq:>6}  {cells[0]:>22}  {cells[1]:>22}{here}")
        self.io.writeln("    (* does not fit in available memory now)")
        self.io.writeln(f"  Widest statevector run now: {widest['double']} qubits "
                        f"({widest['single']} at PRECISION SINGLE)")
        sv_numpy = cap.max_numpy_qubits(prec, available=avail)
        self.io.writeln(f"  LOCC / live-measurement / STEP engine (in place): {sv_numpy} qubits")
        self.io.writeln("  Beyond the wall: METHOD stabilizer to 4096 qubits (polynomial memory),")
        self.io.writeln("  MPS/automatic to 1024 (memory set by entanglement, not width).")
        self.io.writeln('')


# ---------------------------------------------------------------------------
# QUBASIC circuit and result interchange: EXPORT and LOADQASM (OpenQASM),
# CSV, SAVEPNG, and the amplitude rows CSV and CLIP write.
# ---------------------------------------------------------------------------

class ExportMixin(PresentationHost):
    """Circuits and results in other formats."""

    # Amplitude rows written by CSV/CLIP: every non-zero amplitude up to this
    # many, else this many of the most probable ones.
    MAX_EXPORT_AMPLITUDES = 1 << 20

    def cmd_loadqasm(self, rest: str) -> None:
        """LOADQASM <file> — import an OpenQASM file as an editable program.

        Parses OpenQASM 3 (falling back to 2.0) into a circuit and translates it
        into numbered QUBASIC gate lines, so the result can be LISTed, edited, and
        re-RUN. Gates outside the built-in set are imported as UNITARY blocks when
        a matrix is available, otherwise reported and skipped. Reverses EXPORT."""
        if not rest.strip():
            self.io.writeln("?USAGE: LOADQASM <file.qasm>")
            return
        try:
            path = self._sanitize_path(rest.strip())
        except ValueError as e:
            self.io.writeln(f"?LOADQASM ERROR: {e}")
            return
        if not os.path.isfile(path) and os.path.isfile(path + '.qasm'):
            path += '.qasm'
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        text = open(path, 'r', encoding='utf-8-sig').read()
        qc = None
        errors = []
        for loader, mod in (('loads', 'qiskit.qasm3'), ('loads', 'qiskit.qasm2')):
            try:
                m = __import__(mod, fromlist=[loader])
                qc = getattr(m, loader)(text)
                break
            except Exception as e:
                errors.append(f"{mod}: {e}")
        if qc is None:
            self.io.writeln("?LOADQASM: could not parse as OpenQASM 3 or 2")
            for err in errors:
                self.io.writeln(f"  {err}")
            if any('qiskit_qasm3_import' in err for err in errors):
                self.io.writeln("  (3.0 import needs the optional package: "
                                "pip install qubasic[qasm3])")
            return
        lines, n_unknown, custom = self._circuit_to_qb(qc)
        self.cmd_new(silent=True)
        self.num_qubits = qc.num_qubits
        for name, mat in custom.items():
            self._custom_gates[name] = mat
        ln = 10
        for stmt in lines:
            self.program[ln] = stmt
            self._parsed[ln] = parse_stmt(stmt)
            ln += 10
        self.io.writeln(f"LOADQASM {path}: {qc.num_qubits} qubits, {len(lines)} lines")
        if n_unknown:
            self.io.writeln(f"  ({n_unknown} non-standard gate(s) imported as UNITARY)")

    def _circuit_to_qb(self, qc: QuantumCircuit) -> tuple[list[str], int, dict]:
        """Translate a QuantumCircuit into QUBASIC gate lines.

        Returns (lines, n_unknown, custom_gates). Standard gates map to their
        QUBASIC form; unknown gates with a matrix become UNITARY definitions.
        """
        lines: list[str] = []
        custom: dict = {}
        n_unknown = 0
        has_measure = False
        names = [inst.operation.name.lower() for inst in qc.data]
        for k, inst in enumerate(qc.data):
            op = inst.operation
            nm = names[k]
            qidx = [qc.find_bit(q).index for q in inst.qubits]
            if nm == 'measure':
                has_measure = True
                continue
            if nm == 'barrier':
                rest = names[k + 1:]
                if not (rest and all(r == 'measure' for r in rest)):   # MEASURE's own barrier
                    lines.append('BARRIER')
                continue
            if nm == 'reset':
                lines.append(f'RESET {qidx[0]}')
                continue
            canonical = GATE_ALIASES.get(nm.upper(), nm.upper())
            if canonical in GATE_TABLE:
                n_params = GATE_TABLE[canonical][0]
                params = [repr(float(p)) for p in op.params[:n_params]]
                args = params + [str(q) for q in qidx]
                lines.append(f"{canonical} {', '.join(args)}")
            else:
                try:
                    import numpy as _np
                    mat = _np.asarray(op.to_matrix(), dtype=complex)
                    gname = f"G{len(custom)}"
                    custom[gname] = mat
                    lines.append(f"{gname} {', '.join(str(q) for q in qidx)}")
                    n_unknown += 1
                except Exception:
                    lines.append(f"REM (skipped unsupported gate: {nm})")
                    n_unknown += 1
        if has_measure:
            lines.append('MEASURE')
        return lines, n_unknown, custom

    def cmd_savepng(self, rest: str) -> None:
        """SAVEPNG <file> [screen|hist|bloch|circuit] — save a picture as a PNG image.

        'screen' writes the graphics SCREEN at its own resolution, 'hist' the
        last measurement histogram, 'bloch' the Bloch spheres of the current
        statevector and 'circuit' the circuit diagram. The default is the
        graphics screen when one is open, else the histogram when results
        exist, else the circuit. The plots need matplotlib (circuit diagrams
        also pylatexenc)."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: SAVEPNG <file> [screen|hist|bloch|circuit]")
            return
        try:
            path = self._sanitize_path(parts[0])
        except ValueError as e:
            self.io.writeln(f"?SAVEPNG ERROR: {e}")
            return
        checked = self._check_agent_path(path, "SAVEPNG")
        if checked is None:
            return
        path = checked
        if not path.lower().endswith('.png'):
            path += '.png'
        default = 'screen' if self._canvas is not None else ('hist' if self.last_counts else 'circuit')
        what = parts[1].lower() if len(parts) > 1 else default
        if what == 'screen':
            if self._canvas is None:
                self.io.writeln("?NO GRAPHICS SCREEN — SCREEN 12 (or 1, 2, 7-13) first")
                return
            try:
                with open(path, 'wb') as fh:
                    fh.write(self._gfx_png())
            except OSError as e:
                self.io.writeln(f"?SAVEPNG ERROR: {e}")
                return
            self.io.writeln(f"SAVED screen image to {path}")
            return
        try:
            import matplotlib
            matplotlib.use('Agg')  # headless, no display needed
        except ImportError:
            self.io.writeln("?SAVEPNG requires matplotlib (pip install matplotlib)")
            return
        try:
            if what in ('hist', 'histogram'):
                if not self.last_counts:
                    self.io.writeln("?NO RESULTS — RUN with MEASURE first")
                    return
                from qiskit.visualization import plot_histogram
                fig = plot_histogram(self.last_counts)
            elif what == 'bloch':
                sv = self._active_sv
                if sv is None:
                    self.io.writeln("?NO STATE — RUN first")
                    return
                from qiskit.visualization import plot_bloch_multivector
                fig = plot_bloch_multivector(np.ascontiguousarray(sv).ravel())
            elif what == 'circuit':
                if self.last_circuit is None:
                    self.io.writeln("?NO CIRCUIT — RUN first")
                    return
                try:
                    fig = self.last_circuit.draw('mpl')
                except Exception as e:
                    self.io.writeln(f"?circuit image needs pylatexenc (pip install pylatexenc): {e}")
                    return
            else:
                self.io.writeln("?SAVEPNG type must be screen, hist, bloch or circuit")
                return
            fig.savefig(path, bbox_inches='tight', dpi=120)
            import matplotlib.pyplot as plt
            plt.close(fig)
            self.io.writeln(f"SAVED {what} image to {path}")
        except Exception as e:
            self.io.writeln(f"?SAVEPNG ERROR: {e}")

    def cmd_export(self, rest: str) -> None:
        """EXPORT [filename] — export circuit as OpenQASM (2.0, else 3.0)."""
        if self.last_circuit is None:
            self.io.writeln("?NO CIRCUIT — RUN first")
            return
        qasm = None
        version = None
        errors = []
        # QASM 2.0 first so the output round-trips through LOADQASM without
        # the optional qiskit_qasm3_import package; 3.0 covers dynamic
        # circuits (if_test) and anything else 2.0 cannot express.
        try:
            from qiskit.qasm2 import dumps as _dumps2
            qasm = _dumps2(self.last_circuit)
            version = '2.0'
        except Exception as e:
            errors.append(f"qasm2: {e}")
        if qasm is None:
            try:
                from qiskit.qasm3 import dumps as _dumps3
                qasm = _dumps3(self.last_circuit)
                version = '3.0'
            except Exception as e:
                errors.append(f"qasm3: {e}")
        if qasm is None:
            self.io.writeln("?EXPORT: OpenQASM export not available.")
            for err in errors:
                self.io.writeln(f"  {err}")
            return
        if rest.strip():
            try:
                path = self._sanitize_path(rest)
            except ValueError as e:
                self.io.writeln(f"?EXPORT ERROR: {e}")
                return
            checked = self._check_agent_path(path, "EXPORT")
            if checked is None:
                return
            path = checked
            if os.path.exists(path):
                self.io.writeln(f"  (overwriting {path})")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(qasm)
            self.io.writeln(f"EXPORTED to {path} (OpenQASM {version})")
        else:
            self.io.writeln(qasm)

    def cmd_csv(self, rest: str) -> None:
        """CSV [filename] — export last results to CSV.

        Includes measurement counts, and statevector amplitudes if available.
        """
        sv = self._export_state()
        if self.last_counts is None and sv is None:
            self.io.writeln("?NO RESULTS — RUN first")
            return
        lines = []
        if self.last_counts:
            total = sum(self.last_counts.values())
            lines.append("state,count,probability")
            for state, count in sorted(self.last_counts.items(), key=lambda x: -x[1]):
                lines.append(f"{self._bits(state)},{count},{count/total:.6f}")
        if sv is not None:
            lines.append("")
            lines.append("state,amplitude_re,amplitude_im,probability")
            lines.extend(self._amplitude_rows(sv))
        if rest.strip():
            try:
                path = self._sanitize_path(rest)
            except ValueError as e:
                self.io.writeln(f"?CSV ERROR: {e}")
                return
            checked = self._check_agent_path(path, "CSV")
            if checked is None:
                return
            path = checked
            if os.path.exists(path):
                self.io.writeln(f"  (overwriting {path})")
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            what = f"{len(self.last_counts)} states" if self.last_counts else "statevector"
            self.io.writeln(f"EXPORTED {what} to {path}")
        else:
            for line in lines[:20]:
                self.io.writeln(f"  {line}")
            if len(lines) > 20:
                self.io.writeln(f"  ... ({len(lines)-20} more)")

    def _export_state(self) -> np.ndarray | None:
        """The state CSV/CLIP export: one already in memory, or (for up to 16
        qubits) one computed on demand; a large deferred state is skipped."""
        if self._last_sv is not None:
            return self._last_sv
        if self.num_qubits <= 16:
            return self.last_sv
        return None

    def _amplitude_rows(self, sv: np.ndarray, sep: str = ',') -> list[str]:
        """'state,re,im,p' rows for non-zero amplitudes in OPTION ENDIAN order."""
        from qubasic_core.simtools import nonzero_indices, top_k
        flat = np.asarray(sv).reshape(-1)
        n = int(flat.size).bit_length() - 1
        idx, total = nonzero_indices(flat, 1e-4, self.MAX_EXPORT_AMPLITUDES)
        note = []
        if total > idx.size:
            idx, _, _ = top_k(flat, self.MAX_EXPORT_AMPLITUDES, 1e-4)
            idx = np.sort(idx)
            note = [f"# {total} non-zero amplitudes; the {idx.size} most probable are listed"]
        amps = flat[idx]
        rows = [f"{self._bits(format(int(i), f'0{n}b'))}{sep}{a.real:.8f}{sep}{a.imag:.8f}"
                f"{sep}{abs(a) ** 2:.8f}" for i, a in zip(idx, amps)]
        return note + rows


# ---------------------------------------------------------------------------
# QUBASIC conveniences: command suggestions, tips, themes, the status
# prompt, COMPARE, DIFF, QUIZ, EXPLAIN and CLIP.
# ---------------------------------------------------------------------------

_TIPS = [
    "Use CTRL H 0, 1 to make any gate controlled.",
    "STEP mode lets you watch the statevector evolve line by line.",
    "REWIND goes back in time during STEP mode.",
    "LOCC JOINT 3 3 enables quantum teleportation with classical correction.",
    "SWEEP var 0 PI 10 shows how probability changes with a parameter.",
    "NOISE depolarizing 0.05 adds realistic noise to your circuit.",
    "SEED 42 makes your results reproducible.",
    "DEMO LIST shows all 12 built-in quantum algorithms.",
    "BLOCH draws an ASCII Bloch sphere for each qubit.",
    "INV RX 0.5, 0 applies the inverse (dagger) of any gate.",
    "UNITARY MYGATE = [[0,1],[1,0]] defines a custom gate from a matrix.",
    "DEF BELL = H 0 : CX 0,1 creates a reusable gate sequence.",
    "POKE $D000, 8 sets the qubit count via the memory map.",
    "SYS $E000 runs a built-in algorithm by address.",
    "STATS 100 runs 100 trials and shows mean/stddev per state.",
    "PROFILE ON then RUN shows per-line execution time.",
    "ENTROPY 0 measures entanglement of qubit 0 with the rest.",
    "EXPECT ZZ 0 1 computes the two-qubit ZZ correlation.",
    "BANK 1 switches to a second program slot (auto-saves current).",
    "SCREEN 3 auto-displays Bloch spheres after every RUN.",
    "CONSISTENCY cross-checks statevector, density, Bloch, and histogram.",
    "FOR I = 0 TO 7 / H I / NEXT I applies H to 8 qubits in a loop.",
    "FIND \"CX\" searches your program for a string.",
    "CHECKSUM gives an MD5 hash for program verification.",
    "10 H 0 : CX 0,1 : RZ PI/4, 0 puts three gates on one line.",
]


def tip_of_the_day() -> str:
    """Return a random tip."""
    return random.choice(_TIPS)


_ALL_COMMANDS = [
    'RUN', 'LIST', 'NEW', 'SAVE', 'LOAD', 'QUBITS', 'SHOTS', 'METHOD',
    'DEF', 'REG', 'LET', 'STEP', 'STATE', 'HIST', 'BLOCH', 'PROBS',
    'DEMO', 'DELETE', 'RENUM', 'DEFS', 'REGS', 'VARS', 'HELP',
    'CIRCUIT', 'LOCC', 'SEND', 'SHARE', 'SWEEP', 'INCLUDE', 'EXPORT',
    'DECOMPOSE', 'NOISE', 'EXPECT', 'DENSITY', 'ENTROPY', 'CSV',
    'MEASURE', 'BARRIER', 'PRINT', 'INPUT', 'DIM', 'UNITARY',
    'LOCCINFO', 'RAM', 'UNDO', 'CLEAR', 'PEEK', 'POKE', 'SYS',
    'DUMP', 'MAP', 'CATALOG', 'MONITOR', 'SCREEN', 'COLOR', 'CLS',
    'LOCATE', 'PLAY', 'PROMPT', 'BREAK', 'WATCH', 'PROFILE', 'STATS',
    'REWIND', 'FORWARD', 'HISTORY', 'TRON', 'TROFF', 'CONT',
    'AUTO', 'EDIT', 'COPY', 'MOVE', 'FIND', 'REPLACE', 'BANK',
    'CHAIN', 'MERGE', 'CHECKSUM', 'VERSION', 'SEED', 'PROBE',
    'RESTORE', 'OPEN', 'CLOSE', 'IMPORT', 'SAMPLE', 'ESTIMATE',
    'BENCH', 'SET_STATE', 'CIRCUIT_DEF', 'APPLY_CIRCUIT',
    'CONSISTENCY', 'COMPARE', 'HEATMAP', 'ANIMATE', 'QUIZ',
    'DIFF', 'PLOT', 'THEME', 'CLIP', 'EXPLAIN',
    'QFT', 'IQFT', 'DIFFUSE', 'MCX', 'MCZ', 'MCP', 'QADD', 'QADDC', 'QPE',
    'MINIMIZE', 'GRADIENT', 'FIDELITY', 'TOMOGRAPHY', 'COUPLING', 'BASIS',
    'LOADQASM', 'SET_DENSITY', 'SAVEPNG', 'PTOMOGRAPHY', 'RB', 'MEASURE',
    'HAMILTONIAN', 'EVOLVE', 'LINDBLAD', 'CHANNEL', 'APPLYCHANNEL',
    'QEC', 'LOGICAL_ERROR_RATE', 'THRESHOLD', 'DISTILL', 'LATTICE',
    'XEB', 'QVOLUME', 'RBINT', 'MIRROR', 'GST', 'CONCURRENCE', 'NEGATIVITY',
    'IQPE', 'AMPEST', 'AMPLIFY', 'QWALK', 'GRAPHSTATE', 'FEATUREMAP', 'QKERNEL',
    'SHOR', 'HHL', 'PAULIPROP', 'QAOA', 'VQE', 'GROVER', 'SIMON',
    'QUDIT', 'QX', 'QZ', 'QF', 'QSUM', 'QSTATE', 'QMEASURE',
    'BOSONIC', 'DISPLACE', 'SQUEEZE', 'CAT', 'BS', 'BSTATE',
    'RESOURCES', 'DEVICE', 'OPTIMIZE',
    # Gates
    'H', 'X', 'Y', 'Z', 'S', 'T', 'SDG', 'TDG', 'SX', 'ID',
    'RX', 'RY', 'RZ', 'P', 'U', 'CX', 'CZ', 'CY', 'CH',
    'SWAP', 'DCX', 'ISWAP', 'CRX', 'CRY', 'CRZ', 'CP',
    'RXX', 'RYY', 'RZZ', 'CCX', 'CSWAP',
]


def did_you_mean(word: str) -> str | None:
    """Return a suggestion for a misspelled command, or None."""
    w = word.upper()
    if w in _ALL_COMMANDS:
        return None
    matches = difflib.get_close_matches(w, _ALL_COMMANDS, n=1, cutoff=0.6)
    return matches[0] if matches else None


THEMES = {
    'default': {'reset': '\033[0m', 'gate': '\033[36m', 'flow': '\033[33m',
                'comment': '\033[2m', 'string': '\033[32m', 'number': '\033[35m',
                'error': '\033[31m', 'bar_hi': '\033[32m', 'bar_mid': '\033[33m',
                'bar_lo': '\033[2m', 'prompt': ''},
    'retro':   {'reset': '\033[0m', 'gate': '\033[32m', 'flow': '\033[32m',
                'comment': '\033[2;32m', 'string': '\033[32m', 'number': '\033[32m',
                'error': '\033[1;32m', 'bar_hi': '\033[1;32m', 'bar_mid': '\033[32m',
                'bar_lo': '\033[2;32m', 'prompt': '\033[32m'},
    'none':    {'reset': '', 'gate': '', 'flow': '', 'comment': '', 'string': '',
                'number': '', 'error': '', 'bar_hi': '', 'bar_mid': '', 'bar_lo': '',
                'prompt': ''},
}


class AssistMixin(PresentationHost):
    """Interactive helpers around the REPL."""

    def _init_assist(self) -> None:
        self._theme: dict[str, str] = THEMES['default']
        self._theme_name: str = 'default'

    def _suggest_command(self, word: str) -> None:
        """Print a suggestion if the word is close to a known command."""
        suggestion = did_you_mean(word)
        if suggestion:
            self.io.writeln(f"  Did you mean: {suggestion}?")

    def _status_prompt(self) -> str:
        """Build a prompt showing current config."""
        parts = [f"{self.num_qubits}q"]
        if self.locc_mode:
            parts.append("LOCC")
        if self._noise_model:
            parts.append("noisy")
        return ' '.join(parts) + ' ] '


    def cmd_compare(self, rest: str) -> None:
        """COMPARE method1 method2 — run circuit with two methods and diff results."""
        parts = rest.split()
        if len(parts) < 2:
            self.io.writeln("?USAGE: COMPARE <method1> <method2>")
            return
        m1, m2 = parts[0].lower(), parts[1].lower()
        old_method = self.sim_method
        results = {}
        for m in [m1, m2]:
            self.sim_method = m
            with self._silenced():
                self.cmd_run()
            results[m] = dict(self.last_counts) if self.last_counts else {}
        self.sim_method = old_method
        # Display comparison
        all_states = sorted(set(list(results[m1].keys()) + list(results[m2].keys())))
        total1 = sum(results[m1].values()) or 1
        total2 = sum(results[m2].values()) or 1
        self.io.writeln(f"\n  {'State':>12} {m1:>12} {m2:>12} {'diff':>8}")
        for s in all_states[:32]:
            c1 = results[m1].get(s, 0)
            c2 = results[m2].get(s, 0)
            p1, p2 = c1 / total1, c2 / total2
            diff = p2 - p1
            self.io.writeln(f"  |{self._bits(s)}\u27E9 {p1:>11.3%} {p2:>11.3%} {diff:>+7.3%}")
        self.io.writeln('')

    def cmd_quiz(self, rest: str = '') -> None:
        """QUIZ — quantum computing quiz."""
        quizzes = [
            ("H|0> = ?", ["|+>", "|1>", "|->", "|0>"], 0),
            ("CX|10> = ?", ["|11>", "|10>", "|00>", "|01>"], 0),
            ("X|0> = ?", ["|1>", "|0>", "|+>", "|->"], 0),
            ("Z|+> = ?", ["|->", "|+>", "|0>", "|1>"], 0),
            ("H|1> = ?", ["|->", "|+>", "|0>", "|1>"], 0),
            ("What gate creates superposition?", ["H", "X", "Z", "CX"], 0),
            ("Bell state is:", ["|00>+|11>", "|00>+|01>", "|01>+|10>", "|00>-|11>"], 0),
            ("CNOT flips target when control is:", ["|1>", "|0>", "|+>", "|->"], 0),
        ]
        q = random.choice(quizzes)
        prompt, options, correct = q
        self.io.writeln(f"\n  QUIZ: {prompt}")
        indices = list(range(len(options)))
        random.shuffle(indices)
        correct_label = None
        for label_idx, orig_idx in enumerate(indices):
            letter = chr(65 + label_idx)
            if orig_idx == correct:
                correct_label = letter
            self.io.writeln(f"    {letter}) {options[orig_idx]}")
        try:
            answer = self.io.read_line("  Your answer: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            self.io.writeln("")
            return
        if answer == correct_label:
            self.io.writeln("  Correct!")
        else:
            self.io.writeln(f"  Wrong -- the answer is {correct_label}) {options[correct]}")
        self.io.writeln('')

    def cmd_diff(self, rest: str) -> None:
        """DIFF <slot> — diff current program against another bank slot."""
        if not rest.strip():
            self.io.writeln("?USAGE: DIFF <slot>")
            return
        slot = int(rest.strip())
        other = self._program_slots.get(slot, {})
        if not other:
            self.io.writeln(f"?BANK {slot} is empty")
            return
        current_lines = [f"{n} {self.program[n]}" for n in sorted(self.program.keys())]
        other_lines = [f"{n} {other[n]}" for n in sorted(other.keys())]
        diff = list(difflib.unified_diff(other_lines, current_lines,
                                          fromfile=f'bank {slot}', tofile='current',
                                          lineterm=''))
        if diff:
            for line in diff:
                self.io.writeln(f"  {line}")
        else:
            self.io.writeln(f"  No differences with bank {slot}")


    def cmd_theme(self, rest: str) -> None:
        """THEME [name] — switch color theme (default, retro, none)."""
        name = rest.strip().lower() if rest.strip() else ''
        if not name:
            self.io.writeln(f"  Current theme: {self._theme_name}")
            self.io.writeln(f"  Available: {', '.join(THEMES.keys())}")
            return
        if name not in THEMES:
            self.io.writeln(f"?Unknown theme: {name}. Available: {', '.join(THEMES.keys())}")
            return
        self._theme = THEMES[name]
        self._theme_name = name
        self.io.writeln(f"THEME = {name}")

    def cmd_explain(self, rest: str = '') -> None:
        """EXPLAIN — describe each program line in plain English."""
        if not self.program:
            self.io.writeln("EMPTY PROGRAM")
            return
        from qubasic_core.parser import GATE_ALIASES, GATE_TABLE
        gate_desc = {
            'H': 'put qubit {q} in superposition',
            'X': 'flip qubit {q} (NOT)',
            'Y': 'Pauli-Y on qubit {q}',
            'Z': 'flip phase of qubit {q}',
            'S': 'S-phase on qubit {q}',
            'T': 'T-phase on qubit {q}',
            'SDG': 'S-dagger on qubit {q}',
            'TDG': 'T-dagger on qubit {q}',
            'SX': 'sqrt(X) on qubit {q}',
            'ID': 'identity on qubit {q}',
            'CX': 'CNOT: flip {q1} if {q0}=|1>',
            'CZ': 'CZ: phase flip if both |1>',
            'CY': 'controlled-Y',
            'CH': 'controlled-H',
            'CCX': 'Toffoli: flip {q2} if {q0},{q1}=|1>',
            'CSWAP': 'Fredkin: swap if control=|1>',
            'SWAP': 'swap qubits {q0} and {q1}',
            'RX': 'rotate {q} around X by {p}',
            'RY': 'rotate {q} around Y by {p}',
            'RZ': 'rotate {q} around Z by {p}',
            'P': 'phase({p}) on qubit {q}',
            'CP': 'controlled-phase({p})',
            'CRX': 'controlled-RX({p})',
            'CRY': 'controlled-RY({p})',
            'CRZ': 'controlled-RZ({p})',
            'RXX': 'XX interaction({p})',
            'RYY': 'YY interaction({p})',
            'RZZ': 'ZZ interaction({p})',
            'U': 'general unitary U({p})',
            'DCX': 'double-CNOT',
            'ISWAP': 'iSWAP',
        }
        flow_desc = {
            'FOR': 'begin loop', 'NEXT': 'end of loop',
            'WHILE': 'loop while condition holds', 'WEND': 'end of WHILE loop',
            'IF': 'conditional branch', 'GOTO': 'jump to line',
            'GOSUB': 'call subroutine', 'RETURN': 'return from subroutine',
            'END': 'stop execution', 'MEASURE': 'measure all qubits',
            'BARRIER': 'optimization barrier', 'PRINT': 'output text or value',
            'LET': 'set variable', 'DIM': 'create array',
            'DO': 'begin DO loop', 'LOOP': 'end of DO loop',
            'EXIT': 'exit loop early', 'SELECT': 'begin SELECT CASE',
            'CASE': 'match case', 'CALL': 'call subroutine',
            'SUB': 'define subroutine', 'FUNCTION': 'define function',
            'DATA': 'inline data', 'READ': 'read from DATA',
            'RESTORE': 'reset DATA pointer',
        }
        for num in sorted(self.program.keys()):
            raw = self.program[num].strip()
            upper = raw.upper()
            if upper.startswith('REM') or upper.startswith("'"):
                self.io.writeln(f"  {num:5d}  (comment)")
                continue
            # Split on colons and explain each part
            parts = [s.strip() for s in raw.split(':') if s.strip()]
            descs = []
            for part in parts:
                word = part.split()[0].upper() if part.split() else ''
                canonical = GATE_ALIASES.get(word, word)
                if canonical in gate_desc:
                    info = GATE_TABLE.get(canonical)
                    args = part.split(None, 1)[1].strip() if ' ' in part else ''
                    arg_list = [a.strip() for a in args.split(',')]
                    n_params = info[0] if info else 0
                    params = arg_list[:n_params]
                    qubits = arg_list[n_params:]
                    fmt = gate_desc[canonical]
                    fmt = fmt.replace('{q}', ', '.join(qubits) if qubits else '?')
                    fmt = fmt.replace('{p}', ', '.join(params) if params else '?')
                    for i, q in enumerate(qubits):
                        fmt = fmt.replace(f'{{q{i}}}', q)
                    descs.append(fmt)
                elif word in flow_desc:
                    descs.append(flow_desc[word])
                elif word.startswith('@'):
                    reg = word[1:]
                    inner = part.split(None, 1)[1].strip() if ' ' in part else ''
                    descs.append(f"@{reg}: {inner}")
                elif word == 'SEND':
                    descs.append("measure and send classical bit")
                elif word == 'SHARE':
                    descs.append("create shared entanglement")
                else:
                    descs.append(part)
            self.io.writeln(f"  {num:5d}  {'; '.join(descs)}")

    def cmd_clip(self, rest: str = '') -> None:
        """CLIP — copy last results to clipboard (if available)."""
        lines = []
        sv = self._export_state() if not self.last_counts else None
        if self.last_counts:
            total = sum(self.last_counts.values())
            for s, c in sorted(self.last_counts.items(), key=lambda x: -x[1]):
                lines.append(f"|{self._bits(s)}> {c}/{total} ({100*c/total:.1f}%)")
        elif sv is not None:
            for row in self._amplitude_rows(sv, sep=' '):
                if row.startswith('#'):
                    lines.append(row)
                    continue
                state, re_, im_, _p = row.split(' ')
                lines.append(f"|{state}> {float(re_):+.4f}{float(im_):+.4f}j")
        else:
            self.io.writeln("?NO RESULTS TO COPY")
            return
        text = '\n'.join(lines)
        try:
            if sys.platform == 'win32':
                import subprocess
                subprocess.run(['clip'], input=text.encode(), check=True)
            elif sys.platform == 'darwin':
                import subprocess
                subprocess.run(['pbcopy'], input=text.encode(), check=True)
            else:
                import subprocess
                subprocess.run(['xclip', '-selection', 'clipboard'],
                               input=text.encode(), check=True, timeout=2)
            self.io.writeln(f"COPIED {len(lines)} lines to clipboard")
        except Exception:
            self.io.writeln("?Clipboard not available. Output:")
            for line in lines[:20]:
                self.io.writeln(f"  {line}")
            if len(lines) > 20:
                self.io.writeln(f"  ... ({len(lines) - 20} more)")
