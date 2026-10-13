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
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qubasic_core.engine import RAM_BUDGET_FRACTION
from qubasic_core.host import HostState
from qubasic_core.parser import GATE_ALIASES, GATE_TABLE, parse_stmt

# ---------------------------------------------------------------------------
# QUBASIC analysis mixin — EXPECT, ENTROPY, DENSITY, BENCH, RAM.
# ---------------------------------------------------------------------------

class AnalysisMixin(HostState):
    """Analysis and introspection commands for QBasicTerminal.

    Requires: TerminalProtocol — uses self.last_sv, self.last_counts,
    self.num_qubits, self.shots, self.sim_method, self.sim_device,
    self.locc, self.locc_mode.
    """

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
                backend.run(transpile(qc, backend), shots=100).result()
                dt = time.time() - t0
                mem_str = f"  {psutil.virtual_memory().used / 1e9:>7.1f}" if has_psutil else ""
                self.io.writeln(f"  {n:>8}  {method:>20}  {dt:>7.2f}s{mem_str}")
            except Exception as e:
                self.io.writeln(f"  {n:>8}  {method:>20}  FAILED: {e}")
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
                from qubasic_core.interpreter import _resolve_named_state
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

class ExportMixin(HostState):
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
# prompt, colored LIST, COMPARE, DIFF, QUIZ, EXPLAIN, CLIP and UNDO previews.
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


class AssistMixin(HostState):
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

    def cmd_list_colored(self) -> None:
        """LIST with syntax highlighting."""
        if not self.program:
            self.io.writeln("EMPTY PROGRAM")
            return
        t = self._theme
        from qubasic_core.parser import GATE_ALIASES, GATE_TABLE
        gate_names = set(GATE_TABLE.keys()) | set(GATE_ALIASES.keys())
        flow_kw = {'FOR', 'NEXT', 'WHILE', 'WEND', 'IF', 'THEN', 'ELSE',
                    'GOTO', 'GOSUB', 'RETURN', 'END', 'DO', 'LOOP', 'EXIT',
                    'SUB', 'FUNCTION', 'CALL', 'SELECT', 'CASE'}
        for num in sorted(self.program.keys()):
            raw = self.program[num]
            upper = raw.strip().upper()
            if upper.startswith('REM') or upper.startswith("'"):
                colored = f"{t['comment']}{raw}{t['reset']}"
            else:
                words = raw.split()
                parts = []
                for w in words:
                    wu = w.upper().rstrip(',').rstrip(':')
                    if wu in gate_names:
                        parts.append(f"{t['gate']}{w}{t['reset']}")
                    elif wu in flow_kw:
                        parts.append(f"{t['flow']}{w}{t['reset']}")
                    elif w.startswith('"') or w.startswith("'"):
                        parts.append(f"{t['string']}{w}{t['reset']}")
                    else:
                        parts.append(w)
                colored = ' '.join(parts)
            self.io.writeln(f"  {num:5d}  {colored}")

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

    def _circuit_complexity(self) -> str:
        """Return a complexity summary string for the last circuit."""
        if not self.last_circuit:
            return ''
        qc = self.last_circuit
        ops = qc.count_ops()
        t_count = ops.get('t', 0) + ops.get('tdg', 0)
        cx_count = ops.get('cx', 0) + ops.get('cnot', 0)
        parts = []
        if t_count:
            parts.append(f"T-count={t_count}")
        if cx_count:
            parts.append(f"CNOT={cx_count}")
        return ', '.join(parts)

    def cmd_undo_preview(self) -> None:
        """UNDO — show what will change and restore."""
        if not self._undo_stack:
            self.io.writeln("?NOTHING TO UNDO")
            return
        prev = self._undo_stack[-1]
        # Show diff
        added = set(self.program.keys()) - set(prev.keys())
        removed = set(prev.keys()) - set(self.program.keys())
        changed = {k for k in set(self.program.keys()) & set(prev.keys())
                    if self.program[k] != prev[k]}
        if added:
            self.io.writeln(f"  Undo will remove: {sorted(added)}")
        if removed:
            self.io.writeln(f"  Undo will restore: {sorted(removed)}")
        if changed:
            for k in sorted(changed):
                self.io.writeln(f"  Undo will revert {k}: {self.program[k]} -> {prev[k]}")
        # Apply
        self.program = self._undo_stack.pop()
        self._parsed.clear()
        self.io.writeln(f"UNDO ({len(self.program)} lines)")

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


# ---------------------------------------------------------------------------
# QUBASIC built-in demo circuits with self-verification.
# ---------------------------------------------------------------------------

class DemoMixin(HostState):
    """Built-in quantum computing demos.

    Requires: TerminalProtocol — uses self.program, self.num_qubits,
    self.shots, self.cmd_new(), self.cmd_run(), self.cmd_list(), self.cmd_locc().
    """

    def cmd_demo(self, name: str) -> None:
        """DEMO <name> | DEMO LIST — load, list and run a built-in demo circuit, checking its result."""
        demos = {
            'BELL':        self._demo_bell,
            'GHZ':         self._demo_ghz,
            'TELEPORT':    self._demo_teleport,
            'GROVER':      self._demo_grover,
            'QFT':         self._demo_qft,
            'DEUTSCH':     self._demo_deutsch,
            'BERNSTEIN':   self._demo_bernstein,
            'SUPERDENSE':  self._demo_superdense,
            'RANDOM':      self._demo_random,
            'STRESS':      self._demo_stress,
            'LOCC-TELEPORT': self._demo_locc_teleport,
            'LOCC-COORD':    self._demo_locc_coord,
        }
        name = name.upper().strip()
        if not name or name == 'LIST':
            self.io.writeln("  Available demos:")
            for d in demos:
                prefix = "  (LOCC)" if d.startswith('LOCC') else ""
                self.io.writeln(f"    DEMO {d}{prefix}")
            return
        if name not in demos:
            self.io.writeln(f"?UNKNOWN DEMO: {name}")
            self.io.writeln("  Try: DEMO LIST")
            return
        demos[name]()

    def _verify_demo(self, expected_states: list[str], min_frac: float = 0.85,
                     label: str = '') -> bool:
        """Check that expected states dominate the output.

        expected_states: list of bitstrings that should capture >= min_frac of shots.
        Returns True if verification passes.
        """
        if not self.last_counts:
            self.io.writeln("  VERIFY: no results")
            return False
        total = sum(self.last_counts.values())
        hit = sum(self.last_counts.get(s, 0) for s in expected_states)
        frac = hit / total if total > 0 else 0
        tag = f" ({label})" if label else ""
        if frac >= min_frac:
            self.io.writeln(f"  VERIFY PASS{tag}: {frac:.1%} in expected states "
                           f"(threshold {min_frac:.0%})")
            return True
        else:
            self.io.writeln(f"  VERIFY FAIL{tag}: {frac:.1%} in expected states "
                           f"(threshold {min_frac:.0%})")
            return False

    def _demo_bell(self) -> None:
        self.cmd_new()
        self.num_qubits = 2
        self.program = {
            10: "REM === Bell State ===",
            20: "REM The simplest entanglement. Two qubits, perfectly correlated.",
            30: "H 0",
            40: "CX 0,1",
            50: "MEASURE",
        }
        self.io.writeln("LOADED: Bell State (2 qubits)")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['00', '11'], 0.95, 'Bell: only |00> and |11>')

    def _demo_ghz(self) -> None:
        self.cmd_new()
        n = min(self.num_qubits, 8)
        self.num_qubits = n
        self.program = {
            10: "REM === GHZ State ===",
            20: f"REM {n}-qubit Greenberger-Horne-Zeilinger state",
            30: "REM All qubits entangled: 50% all-zeros, 50% all-ones.",
            40: "REM Einstein called this 'spooky action at a distance.'",
            50: "H 0",
        }
        line = 60
        for i in range(1, n):
            self.program[line] = f"CX 0,{i}"
            line += 10
        self.program[line] = "MEASURE"
        self.io.writeln(f"LOADED: GHZ State ({n} qubits)")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['0' * n, '1' * n], 0.95, f'GHZ: only |{"0"*n}> and |{"1"*n}>')

    def _demo_teleport(self) -> None:
        self.cmd_new()
        self.num_qubits = 3
        self.program = {
            10:  "REM === Quantum Teleportation (circuit only) ===",
            20:  "REM Qubit 0: state to teleport (prepared as |+>)",
            30:  "REM Qubit 1-2: entangled Bell pair (the 'channel')",
            40:  "REM Shows the entangling stage of teleportation.",
            50:  "REM Classical correction requires mid-circuit measurement",
            60:  "REM with feedforward, which needs LOCC mode.",
            70:  "REM >>> For full teleportation: DEMO LOCC-TELEPORT <<<",
            80:  "REM",
            90:  "REM Prepare state to teleport (|+> on qubit 0)",
            100: "H 0",
            110: "REM Create Bell pair on qubits 1,2",
            120: "H 1",
            130: "CX 1,2",
            140: "REM Teleportation circuit (without classical correction)",
            150: "CX 0,1",
            160: "H 0",
            170: "MEASURE",
        }
        self.io.writeln("LOADED: Quantum Teleportation — circuit stage (3 qubits)")
        self.io.writeln("  Without correction: uniform over all 8 outcomes.")
        self.io.writeln("  For full protocol with correction: DEMO LOCC-TELEPORT")
        self.cmd_list()
        self.cmd_run()

    def _demo_grover(self) -> None:
        self.cmd_new()
        self.num_qubits = 3
        self.shots = 1024
        self.program = {
            10:  "REM === Grover's Search Algorithm ===",
            20:  "REM Searching for |101> among 8 states",
            30:  "REM Classically: ~4 queries. Quantumly: 2 iterations, ~95%.",
            40:  "REM Quadratic speedup over classical unstructured search.",
            50:  "REM",
            60:  "REM Superposition",
            70:  "H 0 : H 1 : H 2",
            80:  "REM",
            90:  "REM === Grover iteration 1 ===",
            100: "REM Oracle: flip phase of |101> (X on q1, CCZ, undo X)",
            110: "X 1",
            120: "H 2 : CCX 0,1,2 : H 2",
            130: "X 1",
            140: "REM Diffusion: 2|s><s| - I",
            150: "H 0 : H 1 : H 2",
            160: "X 0 : X 1 : X 2",
            170: "H 2 : CCX 0,1,2 : H 2",
            180: "X 0 : X 1 : X 2",
            190: "H 0 : H 1 : H 2",
            200: "REM",
            210: "REM === Grover iteration 2 ===",
            220: "X 1",
            230: "H 2 : CCX 0,1,2 : H 2",
            240: "X 1",
            250: "H 0 : H 1 : H 2",
            260: "X 0 : X 1 : X 2",
            270: "H 2 : CCX 0,1,2 : H 2",
            280: "X 0 : X 1 : X 2",
            290: "H 0 : H 1 : H 2",
            300: "REM",
            310: "MEASURE",
        }
        self.io.writeln("LOADED: Grover's Search (3 qubits, target=|101>)")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['101'], 0.85, 'Grover: target |101>')

    def _demo_qft(self) -> None:
        self.cmd_new()
        self.num_qubits = 4
        self.program = {
            10:  "REM === Quantum Fourier Transform ===",
            20:  "REM The quantum analog of the discrete Fourier transform.",
            30:  "REM Exponentially faster than classical FFT.",
            40:  "REM QFT is typically used as a subroutine (phase info lost on measure).",
            50:  "REM",
            60:  "REM Prepare input state |5> = |0101>",
            70:  "X 0 : X 2",
            80:  "REM",
            90:  "REM QFT on 4 qubits",
            100: "H 3",
            110: "CP PI/2, 2, 3",
            120: "CP PI/4, 1, 3",
            130: "CP PI/8, 0, 3",
            140: "H 2",
            150: "CP PI/2, 1, 2",
            160: "CP PI/4, 0, 2",
            170: "H 1",
            180: "CP PI/2, 0, 1",
            190: "H 0",
            200: "REM Swap for bit reversal",
            210: "SWAP 0,3",
            220: "SWAP 1,2",
            230: "MEASURE",
        }
        self.io.writeln("LOADED: QFT on |0101> (4 qubits)")
        self.cmd_list()
        self.cmd_run()

    def _demo_deutsch(self) -> None:
        self.cmd_new()
        self.num_qubits = 2
        self.program = {
            10:  "REM === Deutsch-Jozsa Algorithm ===",
            20:  "REM Determines if a function is constant or balanced",
            30:  "REM in ONE query. Classically needs 2.",
            40:  "REM Published 1992. The first to demonstrate quantum advantage.",
            50:  "REM",
            60:  "REM Oracle qubit in |->",
            70:  "X 1",
            80:  "H 0 : H 1",
            90:  "REM Oracle (balanced function: f(x) = x)",
            100: "CX 0,1",
            110: "REM Interfere and measure",
            120: "H 0",
            130: "MEASURE",
        }
        self.io.writeln("LOADED: Deutsch-Jozsa (2 qubits, balanced oracle)")
        self.io.writeln("  Expect: qubit 0 = 1 (balanced)")
        self.cmd_list()
        self.cmd_run()
        # q0=1 means bit pattern x1 (states 01 or 11)
        self._verify_demo(['01', '11'], 0.95, 'Deutsch: q0=1 (balanced)')

    def _demo_bernstein(self) -> None:
        self.cmd_new()
        self.num_qubits = 5
        self.program = {
            10:  "REM === Bernstein-Vazirani Algorithm ===",
            20:  "REM Finds a hidden bit string s in ONE query.",
            30:  "REM Secret: s = 1011 (4 bits, display order q3q2q1q0)",
            40:  "REM Classically you'd need 4 queries. Quantum: 1.",
            50:  "REM",
            60:  "REM Ancilla in |->",
            70:  "X 4",
            80:  "H 0 : H 1 : H 2 : H 3 : H 4",
            90:  "REM Oracle: CX from each bit where s=1",
            100: "CX 0,4",
            110: "CX 1,4",
            120: "CX 3,4",
            130: "REM Interfere",
            140: "H 0 : H 1 : H 2 : H 3",
            150: "MEASURE",
        }
        self.io.writeln("LOADED: Bernstein-Vazirani (5 qubits, secret=1011)")
        self.io.writeln("  Expect: measurement shows ...1011 (q3q2q1q0)")
        self.cmd_list()
        self.cmd_run()
        # Ancilla (q4) is random; data qubits should read 1011
        self._verify_demo(['01011', '11011'], 0.95, 'BV: secret=1011')

    def _demo_superdense(self) -> None:
        self.cmd_new()
        self.num_qubits = 2
        self.program = {
            10:  "REM === Superdense Coding ===",
            20:  "REM Send 2 classical bits using 1 qubit.",
            30:  "REM Encoding: 00->I, 01->X, 10->Z, 11->ZX (=iY)",
            40:  "REM Sending message '11'",
            50:  "REM",
            60:  "REM Create Bell pair",
            70:  "H 0",
            80:  "CX 0,1",
            90:  "REM Encode '11': apply Z then X to qubit 0",
            100: "Z 0",
            110: "X 0",
            120: "REM Decode: reverse Bell circuit",
            130: "CX 0,1",
            140: "H 0",
            150: "MEASURE",
        }
        self.io.writeln("LOADED: Superdense Coding (message='11')")
        self.io.writeln("  Expect: |11> with high probability")
        self.cmd_list()
        self.cmd_run()
        self._verify_demo(['11'], 0.95, 'Superdense: message=11')

    def _demo_random(self) -> None:
        self.cmd_new()
        n = min(self.num_qubits, 8)
        self.num_qubits = n
        self.shots = 1
        self.program = {
            10: "REM === Quantum Random Number Generator ===",
            20: f"REM {n} qubits = {n}-bit truly random number",
            30: "REM (as random as a simulator's PRNG allows)",
        }
        line = 40
        for i in range(n):
            self.program[line] = f"H {i}"
            line += 10
        self.program[line] = "MEASURE"
        self.io.writeln(f"LOADED: Quantum RNG ({n} bits)")
        self.cmd_list()
        self.cmd_run()
        self.shots = 1024

    def _demo_stress(self) -> None:
        self.cmd_new()
        self.io.writeln("  (note: STRESS builds a highly entangled 20-qubit state; this can take a while)")
        n = 20
        self.num_qubits = n
        self.program = {
            10: f"REM === {n}-Qubit Stress Test ===",
            20: f"REM H on all {n} qubits + nearest-neighbor CX chain",
            30: f"REM Creates a highly entangled {n}-qubit state",
        }
        line = 40
        for i in range(n):
            self.program[line] = f"H {i}"
            line += 10
        for i in range(n - 1):
            self.program[line] = f"CX {i},{i+1}"
            line += 10
        for i in range(n):
            self.program[line] = f"RZ PI/4,{i}"
            line += 10
        for i in range(0, n - 1, 2):
            self.program[line] = f"CX {i},{i+1}"
            line += 10
        self.program[line] = "MEASURE"
        self.io.writeln(f"LOADED: {n}-Qubit Stress Test")
        self.cmd_run()

    def _demo_locc_teleport(self) -> None:
        """Quantum teleportation across A/B boundary (JOINT mode)."""
        self.cmd_new()
        self.cmd_locc('JOINT 3 3')
        self.shots = 1024
        self.program = {
            10:  "REM === LOCC Teleportation ===",
            20:  "REM Alice (A) teleports qubit 0 to Bob (B) qubit 0",
            30:  "REM Pre-shared Bell pair: A[2] <-> B[0]",
            40:  "REM",
            50:  "REM Prepare state to teleport: |+> on A[0]",
            60:  "@A H 0",
            70:  "REM Create shared entanglement",
            80:  "SHARE A 2, B 0",
            90:  "REM",
            100: "REM === Teleportation protocol ===",
            110: "REM Alice entangles her data qubit with her half of the pair",
            120: "@A CX 0, 2",
            130: "@A H 0",
            140: "REM Alice measures and sends classical bits to Bob",
            150: "SEND A 0 -> m0",
            160: "SEND A 2 -> m1",
            170: "REM Bob applies corrections based on Alice's results",
            180: "IF m1 THEN @B X 0",
            190: "IF m0 THEN @B Z 0",
            200: "REM Bob's qubit 0 now holds Alice's original state",
            210: "MEASURE",
        }
        self.io.writeln("LOADED: LOCC Teleportation (JOINT 3+3)")
        self.io.writeln("  Alice sends |+> to Bob. Expect Bob's qubit ~ 50/50.")
        self.cmd_list()
        self.cmd_run()
        # Fidelity: |+> teleported to Bob B[0]
        # Z-basis: should be 50/50 (|+> has equal P(0) and P(1))
        # X-basis: should be ~100% |0> (|+> is eigenstate of X with eigenvalue +1)
        if self.last_counts:
            b0_counts = {'0': 0, '1': 0}
            for state, count in self.last_counts.items():
                parts = state.split('|')
                if len(parts) >= 2:
                    bob_bit = parts[-1][-1]
                    b0_counts[bob_bit] = b0_counts.get(bob_bit, 0) + count
            b_total = sum(b0_counts.values())
            if b_total > 0:
                p0 = b0_counts['0'] / b_total
                z_fidelity = 1.0 - abs(p0 - 0.5) * 2
                self.io.writeln(f"  Z-basis fidelity: {z_fidelity:.3f} "
                               f"(Bob B[0]: P(0)={p0:.3f}, P(1)={1-p0:.3f})")
        # X-basis verification via LOCC statevector
        if self.locc and self.locc.joint:
            import numpy as np
            sv = np.ascontiguousarray(self.locc.sv).ravel()
            n = self.locc.n_total
            # Bob qubit 0 is at global index = offset_B + 0
            bob_q = self.locc.offsets[1]
            sv_t = sv.reshape([2] * n)
            ax = n - 1 - bob_q
            t0 = np.moveaxis(sv_t, ax, 0)[0].flatten()
            t1 = np.moveaxis(sv_t, ax, 0)[1].flatten()
            rho_01 = np.sum(np.conj(t0) * t1)
            expect_x = float(2 * rho_01.real)
            self.io.writeln(f"  X-basis verification: <X> on Bob B[0] = {expect_x:.3f} "
                           f"(ideal |+> = 1.000)")

    def _demo_locc_coord(self) -> None:
        """Classical coordination between independent registers (SPLIT mode)."""
        self.cmd_new()
        self.cmd_locc('SPLIT 4 4')
        self.shots = 1024
        self.program = {
            10:  "REM === LOCC Classical Coordination ===",
            20:  "REM Two independent quantum processors sharing classical bits",
            30:  "REM Alice generates a random 2-bit key, Bob copies it",
            40:  "REM",
            50:  "REM Alice: generate random bits",
            60:  "@A H 0 : @A H 1",
            70:  "SEND A 0 -> k0",
            80:  "SEND A 1 -> k1",
            90:  "REM Bob: set qubits to match Alice's key",
            100: "IF k0 THEN @B X 0",
            110: "IF k1 THEN @B X 1",
            120: "REM Both sides now agree on a 2-bit key",
            130: "REM Alice uses her key to prepare an entangled state",
            140: "@A H 2",
            150: "@A CX 2,3",
            160: "REM Bob does the same — they independently create matching states",
            170: "@B H 2",
            180: "@B CX 2,3",
            190: "MEASURE",
        }
        self.io.writeln("LOADED: LOCC Classical Coordination (SPLIT 4+4)")
        self.io.writeln("  Alice and Bob share a random key, then prepare matching states.")
        self.cmd_list()
        self.cmd_run()
