"""QBASIC analysis mixin — EXPECT, ENTROPY, DENSITY, BENCH, RAM."""

from __future__ import annotations

import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

from qbasic_core.engine import (
    _estimate_gb, _get_ram_gb,
    OVERHEAD_FACTOR, RAM_BUDGET_FRACTION,
)


class AnalysisMixin:
    """Analysis and introspection commands for QBasicTerminal.

    Requires: TerminalProtocol — uses self.last_sv, self.last_counts,
    self.num_qubits, self.shots, self.sim_method, self.sim_device,
    self.locc, self.locc_mode.
    """

    def cmd_expect(self, rest: str) -> None:
        """EXPECT <pauli> [qubits] — compute expectation value.
        Examples: EXPECT Z 0, EXPECT ZZ 0 1, EXPECT X 0"""
        if self.last_sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: EXPECT <Z|X|Y|ZZ|...> [qubits]")
            return
        pauli_str = parts[0].upper()
        qubits = [int(q) for q in parts[1:]] if len(parts) > 1 else list(range(len(pauli_str)))

        try:
            from qiskit.quantum_info import Statevector, SparsePauliOp
            sv = Statevector(np.ascontiguousarray(self.last_sv).ravel())
            # Build Pauli string for full system
            full_pauli = ['I'] * self.num_qubits
            for i, p in enumerate(pauli_str):
                if i < len(qubits):
                    full_pauli[self.num_qubits - 1 - qubits[i]] = p
            op = SparsePauliOp(''.join(full_pauli))
            val = sv.expectation_value(op)
            self.io.writeln(f"  <{pauli_str}> on qubits {qubits} = {val.real:.6f}")
        except Exception as e:
            self.io.writeln(f"?EXPECT ERROR: {e}")

    def cmd_entropy(self, rest: str = '') -> None:
        """ENTROPY [qubits] — entanglement entropy of specified qubits vs rest.
        Examples: ENTROPY 0  |  ENTROPY 0 1  |  ENTROPY (defaults to qubit 0)"""
        if self.last_sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        if rest.strip():
            partition_a = [int(q) for q in rest.replace(',', ' ').split() if q.strip()]
        else:
            partition_a = [0]
        n = self.num_qubits
        try:
            from qiskit.quantum_info import Statevector, entropy, partial_trace
            sv_obj = Statevector(np.ascontiguousarray(self.last_sv).ravel())
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

    def cmd_density(self) -> None:
        """Show density matrix (or partial trace for small systems)."""
        if self.last_sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        sv = np.ascontiguousarray(self.last_sv).ravel()
        rho = np.outer(sv, sv.conj())
        n = self.num_qubits
        dim = 2**n
        if dim > 16:
            self.io.writeln(f"  Density matrix: {dim}x{dim} (too large to display)")
            self.io.writeln(f"  Purity: {np.real(np.trace(rho @ rho)):.6f}")
            self.io.write(f"  Von Neumann entropy: ")
            eigvals = np.linalg.eigvalsh(rho)
            eigvals = eigvals[eigvals > 1e-15]
            ent = -np.sum(eigvals * np.log2(eigvals))
            self.io.writeln(f"{ent:.6f} bits")
            return
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
