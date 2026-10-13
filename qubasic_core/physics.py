"""QUBASIC physical models: Hamiltonians, channels and Lindblad
evolution, molecular hydrogen in STO-3G, Pauli propagation, qudits and
bosonic modes.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping, Sequence
from math import erf, exp, pi, sqrt
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from qiskit import QuantumCircuit

from qubasic_core.host import HostState

if TYPE_CHECKING:
    import scipy.sparse as sp
    from qiskit.quantum_info import SparsePauliOp

    from qubasic_core.engine import CircuitBackend


# ---------------------------------------------------------------------------
# QUBASIC Hamiltonian dynamics and open systems.
#
# Declare a Hamiltonian as a Pauli sum (or a built-in model), evolve a register
# under it with a Trotter product formula, and evolve open systems with a Lindblad
# master equation. All offline on Qiskit/numpy.
#
#   HAMILTONIAN H = 1.0 ZZ 0 1 + 0.5 X 0     Declare a Pauli-sum Hamiltonian
#   HAMILTONIAN H = ISING 1.0 0.5            Transverse-field Ising (J, h)
#   HAMILTONIAN H = HEISENBERG 1.0           Heisenberg XXX chain (coupling J)
#   EVOLVE H, t [, steps]                    Trotterized e^{-iHt} appended to the circuit
#   LINDBLAD H, t, steps, <rate> <op> <q>    Open-system master-equation evolution
# ---------------------------------------------------------------------------

# Single-qubit operators usable as Lindblad jump operators.
_JUMP_OPS = {
    'SM': np.array([[0, 1], [0, 0]], dtype=complex),   # sigma-minus (decay)
    'SP': np.array([[0, 0], [1, 0]], dtype=complex),   # sigma-plus (excite)
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    'N': np.array([[0, 0], [0, 1]], dtype=complex),    # number / dephasing-ish
}

# Widest register the exact (density-matrix) LINDBLAD path integrates.
LINDBLAD_DENSE_MAX = 10


def _add_sandwich(out: np.ndarray, rho: np.ndarray, m: np.ndarray, qubit: int, n: int) -> None:
    """out += M rho M^dag for a single-qubit M on `qubit`, from views of the
    quarter blocks of rho (no full-size temporaries)."""
    hi, lo = 2 ** (n - 1 - qubit), 2 ** qubit
    t = rho.reshape(hi, 2, lo, hi, 2, lo)
    o = out.reshape(hi, 2, lo, hi, 2, lo)
    nz = [(a, b) for a in range(2) for b in range(2) if m[a, b] != 0]
    for a, b in nz:
        for d, c in nz:
            o[:, a, :, :, d, :] += (m[a, b] * np.conj(m[d, c])) * t[:, b, :, :, c, :]


def _embed_sparse(op1: np.ndarray, qubit: int, n: int) -> sp.csr_matrix:
    """A single-qubit operator on `qubit` of n qubits as a sparse matrix
    (qubit 0 least significant)."""
    import scipy.sparse as sp
    return sp.kron(sp.kron(sp.identity(2 ** (n - 1 - qubit), format='csr'),
                           sp.csr_matrix(op1)),
                   sp.identity(2 ** qubit, format='csr'), format='csr')


class DynamicsMixin(HostState):
    """Hamiltonian declaration, Trotter evolution, and Lindblad open dynamics.

    Requires: TerminalProtocol — uses self.num_qubits, self.variables,
    self.last_sv, self._safe_eval(), self._eval_with_vars(), self.io.
    """

    def _init_dynamics(self) -> None:
        self._hamiltonians: dict = {}   # name -> SparsePauliOp

    # ── Hamiltonian construction ────────────────────────────────────────

    def _pauli_label(self, pauli: str, qubits: list[int]) -> str:
        """Build an n-qubit Qiskit Pauli label (qubit n-1 leftmost) for a term."""
        n = self.num_qubits
        lab = ['I'] * n
        for p, q in zip(pauli, qubits):
            if q < 0 or q >= n:
                raise ValueError(f"qubit {q} out of range (0-{n - 1})")
            lab[n - 1 - q] = p
        return ''.join(lab)

    def _parse_pauli_sum(self, spec: str) -> SparsePauliOp:
        from qiskit.quantum_info import SparsePauliOp
        labels: list[str] = []
        coeffs: list[complex] = []
        for term in re.split(r'\s*\+\s*', spec.strip()):
            toks = term.split()
            if not toks:
                continue
            try:
                coeff = float(self._safe_eval(toks[0]))
                rest = toks[1:]
            except Exception:
                coeff = 1.0
                rest = toks
            if not rest:
                raise ValueError(f"term '{term}' has no Pauli operator")
            pauli = rest[0].upper()
            if set(pauli) - set('IXYZ'):
                raise ValueError(f"bad Pauli '{pauli}' (use I, X, Y, Z)")
            qubits = [int(self._eval_with_vars(t, {})) for t in rest[1:]]
            if len(pauli) != len(qubits):
                raise ValueError(f"Pauli '{pauli}' needs {len(pauli)} qubit(s), got {len(qubits)}")
            labels.append(self._pauli_label(pauli, qubits))
            coeffs.append(coeff)
        if not labels:
            raise ValueError("empty Hamiltonian")
        return SparsePauliOp(labels, coeffs)

    def _build_hamiltonian(self, spec: str) -> SparsePauliOp:
        from qiskit.quantum_info import SparsePauliOp
        n = self.num_qubits
        up = spec.upper().split()
        kind = up[0] if up else ''
        if kind == 'MOLECULE':
            # Built-in STO-3G engine (no pyscf): MOLECULE H2 [R_angstrom].
            parts = spec.split()
            mol = parts[1].upper() if len(parts) > 1 else 'H2'
            if mol != 'H2':
                raise ValueError("MOLECULE supports H2 (built-in STO-3G engine); "
                                 "other species need external integrals")
            r = float(self._safe_eval(parts[2])) if len(parts) > 2 else 0.7414
            return h2_hamiltonian(r)
        if kind == 'ISING':
            # Transverse-field Ising: -J sum Z_i Z_{i+1} - h sum X_i.
            parts = spec.split()
            J = float(self._safe_eval(parts[1])) if len(parts) > 1 else 1.0
            h = float(self._safe_eval(parts[2])) if len(parts) > 2 else 1.0
            labels, coeffs = [], []
            for i in range(n - 1):
                labels.append(self._pauli_label('ZZ', [i, i + 1]))
                coeffs.append(-J)
            for i in range(n):
                labels.append(self._pauli_label('X', [i]))
                coeffs.append(-h)
            return SparsePauliOp(labels, coeffs)
        if kind == 'HEISENBERG':
            # XXX chain: J sum (X X + Y Y + Z Z) on neighbours.
            parts = spec.split()
            J = float(self._safe_eval(parts[1])) if len(parts) > 1 else 1.0
            labels, coeffs = [], []
            for i in range(n - 1):
                for p in 'XYZ':
                    labels.append(self._pauli_label(p + p, [i, i + 1]))
                    coeffs.append(J)
            return SparsePauliOp(labels, coeffs)
        if kind in ('HUBBARD', 'FERMI'):
            # 1D spinless fermion chain under Jordan-Wigner: hopping -t and
            # nearest-neighbour interaction V. For adjacent sites the JW string
            # cancels, so c_i^dag c_{i+1} + h.c. = (X_i X_{i+1} + Y_i Y_{i+1})/2,
            # and n_i = (I - Z_i)/2.
            parts = spec.split()
            tt = float(self._safe_eval(parts[1])) if len(parts) > 1 else 1.0
            V = float(self._safe_eval(parts[2])) if len(parts) > 2 else 0.0
            labels, coeffs = [], []
            for i in range(n - 1):
                labels.append(self._pauli_label('XX', [i, i + 1]))
                coeffs.append(-tt / 2)
                labels.append(self._pauli_label('YY', [i, i + 1]))
                coeffs.append(-tt / 2)
                if V:
                    # V n_i n_{i+1} = V/4 (I - Z_i - Z_{i+1} + Z_i Z_{i+1})
                    labels.append(self._pauli_label('II', [i, i + 1]))
                    coeffs.append(V / 4)
                    labels.append(self._pauli_label('Z', [i]))
                    coeffs.append(-V / 4)
                    labels.append(self._pauli_label('Z', [i + 1]))
                    coeffs.append(-V / 4)
                    labels.append(self._pauli_label('ZZ', [i, i + 1]))
                    coeffs.append(V / 4)
            return SparsePauliOp(labels, coeffs).simplify()
        if kind == 'RYDBERG':
            # Neutral-atom analog model: (Omega/2) sum X_i - Delta sum n_i
            # + V sum n_i n_{i+1} (nearest-neighbour blockade), n_i = (I - Z_i)/2.
            parts = spec.split()
            omega = float(self._safe_eval(parts[1])) if len(parts) > 1 else 1.0
            delta = float(self._safe_eval(parts[2])) if len(parts) > 2 else 0.0
            V = float(self._safe_eval(parts[3])) if len(parts) > 3 else 1.0
            labels, coeffs = [], []
            for i in range(n):
                labels.append(self._pauli_label('X', [i]))
                coeffs.append(omega / 2)
                labels.append(self._pauli_label('Z', [i]))
                coeffs.append(delta / 2)
            for i in range(n - 1):
                labels.append(self._pauli_label('II', [i, i + 1]))
                coeffs.append(V / 4)
                labels.append(self._pauli_label('Z', [i]))
                coeffs.append(-V / 4)
                labels.append(self._pauli_label('Z', [i + 1]))
                coeffs.append(-V / 4)
                labels.append(self._pauli_label('ZZ', [i, i + 1]))
                coeffs.append(V / 4)
            return SparsePauliOp(labels, coeffs).simplify()
        return self._parse_pauli_sum(spec)

    # ── User-defined Kraus channels ─────────────────────────────────────

    def cmd_channel(self, rest: str) -> None:
        """CHANNEL <name> = K0 ; K1 ; ... — define a quantum channel by its Kraus operators.

        Each Ki is a matrix literal (single- or multi-qubit). The CPTP condition
        sum Ki^dag Ki = I is checked. Apply it with APPLYCHANNEL <name> <qubit(s)>:
        shots sample it as quantum trajectories, and a run without MEASURE keeps
        the mixed state for DENSITY."""
        m = re.match(r'(\w+)\s*=\s*(.+)$', rest.strip())
        if not m:
            self.io.writeln("?USAGE: CHANNEL <name> = <K0> ; <K1> ; ...")
            return
        name = m.group(1).upper()
        try:
            ops = [np.array(self._parse_matrix(k.strip()), dtype=np.complex128)
                   for k in m.group(2).split(';') if k.strip()]
            dim = ops[0].shape[0]
            check = sum(K.conj().T @ K for K in ops)
            if not np.allclose(check, np.eye(dim), atol=1e-6):
                raise ValueError("Kraus operators must satisfy sum Ki^dag Ki = I (CPTP)")
            self._channels = getattr(self, '_channels', {})
            self._channels[name] = ops
            self.io.writeln(f"CHANNEL {name}: {len(ops)} Kraus operator(s), "
                            f"{int(np.log2(dim))}-qubit")
        except Exception as e:
            self.io.writeln(f"?CHANNEL ERROR: {e}")

    def _try_exec_applychannel(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
        m = re.match(r'APPLYCHANNEL\s+(\w+)\s+(.+)$', stmt, re.IGNORECASE)
        if not m:
            return False
        if getattr(self, 'locc_mode', False):
            raise ValueError("APPLYCHANNEL is not available in LOCC mode")
        name = m.group(1).upper()
        channels = getattr(self, '_channels', {})
        if name not in channels:
            raise ValueError(f"unknown CHANNEL '{name}' (define one with CHANNEL {name} = ...)")
        qubits = [self._resolve_qubit(q) for q in m.group(2).replace(',', ' ').split()]
        from qiskit.quantum_info import Kraus
        # Shots sample the channel as quantum trajectories; a run with no
        # MEASURE keeps the mixed state for DENSITY.
        qc.append(Kraus(channels[name]).to_instruction(), qubits)
        return True

    def cmd_hamiltonian(self, rest: str) -> None:
        """HAMILTONIAN name = pauli sum | ISING J h | HEISENBERG J | HUBBARD t V | RYDBERG Omega Delta V | MOLECULE H2 [R] — declare a Hamiltonian.

        A Pauli sum is terms joined by '+', each '<coeff> <pauli> <qubits>', e.g.
        HAMILTONIAN H = 1.0 ZZ 0 1 + 0.5 X 0. Use it with EVOLVE or LINDBLAD."""
        m = re.match(r'(\w+)\s*=\s*(.+)$', rest.strip())
        if not m:
            self.io.writeln("?USAGE: HAMILTONIAN <name> = <terms>")
            return
        name = m.group(1).upper()
        try:
            op = self._build_hamiltonian(m.group(2).strip())
            self._hamiltonians[name] = op
            self.io.writeln(f"HAMILTONIAN {name}: {len(op)} term(s), {op.num_qubits} qubits")
        except Exception as e:
            self.io.writeln(f"?HAMILTONIAN ERROR: {e}")

    # ── Trotterized unitary evolution ────────────────────────────────────

    def _try_exec_evolve(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
        m = re.match(r'EVOLVE\s+(\w+)\s*,\s*([^,]+?)(?:\s*,\s*(\d+))?\s*$', stmt, re.IGNORECASE)
        if not m:
            return False
        if getattr(self, 'locc_mode', False):
            raise ValueError("EVOLVE is not available in LOCC mode")
        name = m.group(1).upper()
        if name not in self._hamiltonians:
            raise ValueError(f"unknown HAMILTONIAN '{name}' (declare one with HAMILTONIAN {name} = ...)")
        t = float(self._eval_with_vars(m.group(2), run_vars))
        steps = int(m.group(3)) if m.group(3) else 1
        H = self._hamiltonians[name]
        if H.num_qubits != self.num_qubits:
            raise ValueError(f"HAMILTONIAN {name} is {H.num_qubits}-qubit; circuit has {self.num_qubits}")
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.synthesis import SuzukiTrotter
        synth = SuzukiTrotter(order=2, reps=max(1, steps))
        qc.append(PauliEvolutionGate(H, time=t, synthesis=synth), list(range(self.num_qubits)))
        return True

    # ── Lindblad open-system evolution ──────────────────────────────────

    def cmd_lindblad(self, rest: str) -> None:
        """LINDBLAD <H|NONE>, <time>, <steps>, <rate> <op> <q> [; ...] [TRAJ [n]] — Lindblad master-equation evolution, exact or by quantum trajectories.

        Integrates drho/dt = -i[H, rho] + sum_k g_k (L_k rho L_k^dag - {L_k^dag L_k, rho}/2)
        with RK4 from the current state (or |0...0>). Jump operators are single-qubit:
        SM (decay), SP (excite), X, Y, Z, N. Reports the final density matrix.
        The density matrix is dense and every operator sparse, so the exact path
        reaches 10 qubits. TRAJ switches to Monte Carlo wavefunction unraveling
        (statevector-sized, default 200 trajectories) for up to 15 qubits.
        Example: LINDBLAD NONE, 1.0, 200, 1.0 SM 0"""
        ntraj = None
        m_traj = re.search(r'\bTRAJ(?:\s+(\d+))?\s*$', rest, re.IGNORECASE)
        if m_traj:
            ntraj = int(m_traj.group(1)) if m_traj.group(1) else 200
            rest = rest[:m_traj.start()].rstrip().rstrip(',')
        parts = [p.strip() for p in rest.split(',')]
        if len(parts) < 4:
            self.io.writeln("?USAGE: LINDBLAD <H|NONE>, <time>, <steps>, <rate> <op> <q> [; ...] [TRAJ [n]]")
            return
        n = self.num_qubits
        dim = 2 ** n
        if ntraj is None and n > LINDBLAD_DENSE_MAX:
            self.io.writeln(f"?LINDBLAD exact path is limited to {LINDBLAD_DENSE_MAX} qubits "
                            f"({dim}x{dim} rho); append TRAJ for Monte Carlo wavefunction "
                            f"(up to 15)")
            return
        if ntraj is not None and n > 15:
            self.io.writeln(f"?LINDBLAD TRAJ is limited to 15 qubits; have {n}")
            return
        if ntraj is not None:
            self._lindblad_traj(parts, ntraj)
            return
        try:
            import scipy.sparse as sp
            hname = parts[0].upper()
            if hname in ('NONE', '0', ''):
                H = sp.csr_matrix((dim, dim), dtype=complex)
            else:
                if hname not in self._hamiltonians:
                    raise ValueError(f"unknown HAMILTONIAN '{hname}'")
                H = sp.csr_matrix(self._hamiltonians[hname].to_matrix(sparse=True), dtype=complex)
            t = float(self._safe_eval(parts[1]))
            steps = max(1, int(self._safe_eval(parts[2])))
            jumps = []
            for spec in ' '.join(parts[3:]).split(';'):
                toks = spec.split()
                if len(toks) != 3:
                    raise ValueError(f"jump '{spec.strip()}' must be '<rate> <op> <qubit>'")
                rate = float(self._safe_eval(toks[0]))
                opname = toks[1].upper()
                if opname not in _JUMP_OPS:
                    raise ValueError(f"unknown jump op '{opname}' (use {', '.join(_JUMP_OPS)})")
                q = int(self._eval_with_vars(toks[2], {}))
                if not 0 <= q < n:
                    raise ValueError(f"qubit {q} out of range (0-{n - 1})")
                jumps.append((np.sqrt(rate) * _JUMP_OPS[opname], q))
            # Initial density matrix from the current statevector, else |0...0>.
            if self.last_sv is not None and np.asarray(self.last_sv).size == dim:
                psi: np.ndarray = np.ascontiguousarray(self.last_sv).ravel().astype(complex)
                psi = psi / (np.linalg.norm(psi) or 1.0)
                rho = np.outer(psi, psi.conj())
            else:
                rho = np.zeros((dim, dim), dtype=complex)
                rho[0, 0] = 1.0
            # drho/dt = -i (H_eff rho - rho H_eff^dag) + sum_k L_k rho L_k^dag with
            # H_eff = H - (i/2) sum_k L_k^dag L_k. rho stays Hermitian, so
            # rho H_eff^dag = (H_eff rho)^dag; each L_k rho L_k^dag is a sum of
            # quarter blocks of rho, one per pair of non-zero entries of L_k.
            h_eff = sp.csr_matrix(H - 0.5j * sum(
                (_embed_sparse(M.conj().T @ M, q, n) for M, q in jumps),
                sp.csr_matrix((dim, dim), dtype=complex)))

            def drho(r: np.ndarray) -> np.ndarray:
                out = h_eff @ r
                out *= -1j
                out += out.conj().T
                for M, q in jumps:
                    _add_sandwich(out, r, M, q, n)
                return out

            dt = t / steps
            for _ in range(steps):
                k1 = drho(rho)
                k2 = drho(rho + 0.5 * dt * k1)
                k3 = drho(rho + 0.5 * dt * k2)
                k4 = drho(rho + dt * k3)
                rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            rho = 0.5 * (rho + rho.conj().T)        # re-Hermitize against drift
            tr = float(np.real(np.trace(rho)))
            if abs(tr) > 1e-12:
                rho = rho / tr
            self._pending_set_density = None
            self.last_sv = None
            pops = np.real(np.diag(rho))
            self.io.writeln(f"\n  Lindblad evolution to t={t} ({steps} RK4 steps):")
            if dim <= 8:
                for i in range(dim):
                    row = '  '.join(
                        (f"{rho[i, j].real:+.3f}" if abs(rho[i, j].imag) < 1e-6
                         else f"{rho[i, j].real:+.2f}{rho[i, j].imag:+.2f}j")
                        for j in range(dim))
                    self.io.writeln(f"    {row}")
            self.io.writeln("  Populations: " + ', '.join(
                f"|{self._bits(format(i, f'0{n}b'))}>={pops[i]:.4f}"
                for i in range(dim) if pops[i] > 1e-4))
            purity = float(np.vdot(rho, rho).real)      # Tr(rho^2) for Hermitian rho
            self.io.writeln(f"  Purity Tr(rho^2) = {purity:.6f}")
            for i in range(dim):
                self.variables[f'_RHO{i}'] = float(pops[i])
        except Exception as e:
            self.io.writeln(f"?LINDBLAD ERROR: {e}")

    def _lindblad_traj(self, parts: list[str], ntraj: int) -> None:
        """Monte Carlo wavefunction (quantum-trajectory) Lindblad evolution.

        Waiting-time unraveling: RK4 integration of the non-Hermitian
        d|psi>/dt = -i H_eff |psi| with H_eff = H - (i/2) sum L^dag L; a jump
        fires when the decaying norm^2 crosses a pre-drawn uniform, the channel
        chosen with probability proportional to <L^dag L>. Memory is one
        statevector, so 15 qubits fit where the dense 4^n rho cannot."""
        import scipy.sparse as sp
        n = self.num_qubits
        dim = 2 ** n
        try:
            hname = parts[0].upper()
            if hname in ('NONE', '0', ''):
                Heff = sp.csr_matrix((dim, dim), dtype=complex)
            else:
                if hname not in self._hamiltonians:
                    raise ValueError(f"unknown HAMILTONIAN '{hname}'")
                op = self._hamiltonians[hname]
                if op.num_qubits != n:
                    raise ValueError(f"HAMILTONIAN {hname} is {op.num_qubits}-qubit "
                                     f"but QUBITS is {n}")
                Heff = op.to_matrix(sparse=True).tocsr().astype(complex)
            t = float(self._safe_eval(parts[1]))
            steps = max(1, int(self._safe_eval(parts[2])))

            def embed_sparse(op2: np.ndarray, q: int) -> sp.csr_matrix:
                left = sp.identity(2 ** (n - 1 - q), format='csr', dtype=complex)
                right = sp.identity(2 ** q, format='csr', dtype=complex)
                return sp.kron(left, sp.kron(sp.csr_matrix(op2), right)).tocsr()

            jumps = []
            for spec in ' '.join(parts[3:]).split(';'):
                toks = spec.split()
                if len(toks) != 3:
                    raise ValueError(f"jump '{spec.strip()}' must be '<rate> <op> <qubit>'")
                rate = float(self._safe_eval(toks[0]))
                opname = toks[1].upper()
                if opname not in _JUMP_OPS:
                    raise ValueError(f"unknown jump op '{opname}' (use {', '.join(_JUMP_OPS)})")
                q = int(self._eval_with_vars(toks[2], {}))
                jumps.append(np.sqrt(rate) * embed_sparse(_JUMP_OPS[opname], q))
            if not jumps:
                raise ValueError("TRAJ needs at least one jump operator")
            LdLs = [(L.getH() @ L).tocsr() for L in jumps]
            for LdL in LdLs:
                Heff = Heff - 0.5j * LdL
            if self.last_sv is not None and np.asarray(self.last_sv).size == dim:
                psi0: np.ndarray = np.ascontiguousarray(self.last_sv).ravel().astype(complex)
                psi0 = psi0 / (np.linalg.norm(psi0) or 1.0)
            else:
                psi0 = np.zeros(dim, dtype=complex)
                psi0[0] = 1.0
            dt = t / steps
            rng = np.random.default_rng(self._seed)
            pops = np.zeros(dim)
            total_jumps = 0
            for _ in range(ntraj):
                psi = psi0.copy()
                r = rng.random()
                for _ in range(steps):
                    k1 = -1j * (Heff @ psi)
                    k2 = -1j * (Heff @ (psi + 0.5 * dt * k1))
                    k3 = -1j * (Heff @ (psi + 0.5 * dt * k2))
                    k4 = -1j * (Heff @ (psi + dt * k3))
                    psi = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                    if float(np.real(np.vdot(psi, psi))) <= r:
                        w = np.array([max(float(np.real(np.vdot(psi, LdL @ psi))), 0.0)
                                      for LdL in LdLs])
                        ws = float(w.sum())
                        if ws <= 0:
                            break
                        k = int(rng.choice(len(jumps), p=w / ws))
                        psi = jumps[k] @ psi
                        psi = psi / (np.linalg.norm(psi) or 1.0)
                        r = rng.random()
                        total_jumps += 1
                nrm = np.linalg.norm(psi)
                if nrm > 0:
                    psi = psi / nrm
                pops += np.abs(psi) ** 2
            pops /= ntraj
            self._pending_set_density = None
            self.last_sv = None
            self.io.writeln(f"\n  Lindblad evolution to t={t} "
                            f"(MCWF: {ntraj} trajectories x {steps} RK4 steps, {n} qubits):")
            self.io.writeln(f"  Mean jumps/trajectory = {total_jumps / ntraj:.2f}")
            top = np.argsort(pops)[::-1]
            shown = [i for i in top[:8] if pops[i] > 1e-4]
            self.io.writeln("  Populations: " + ', '.join(
                f"|{self._bits(format(int(i), f'0{n}b'))}>={pops[i]:.4f}" for i in shown))
            if dim <= 256:
                for i in range(dim):
                    self.variables[f'_RHO{i}'] = float(pops[i])
        except Exception as e:
            self.io.writeln(f"?LINDBLAD ERROR: {e}")


# ---------------------------------------------------------------------------
# QUBASIC molecular Hamiltonians — self-contained STO-3G integrals engine.
#
# HAMILTONIAN H = MOLECULE H2 [R_angstrom] builds the exact 4-qubit
# Jordan-Wigner Hamiltonian of molecular hydrogen in the STO-3G minimal basis:
# Gaussian s-orbital integrals (Szabo & Ostlund closed forms), restricted
# Hartree-Fock molecular orbitals (analytic by symmetry for H2), the
# spin-orbital tensors, and a Jordan-Wigner transform done by building the
# second-quantized 16x16 matrix from ladder operators and decomposing it with
# SparsePauliOp.from_operator. No pyscf, no openfermion.
#
# Ground truth: exact diagonalization of the result at R = 0.7414 A gives the
# FCI/STO-3G energy -1.1373 Ha (nuclear repulsion included as the identity
# coefficient).
# ---------------------------------------------------------------------------

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

# STO-3G hydrogen 1s: (exponent, contraction) with zeta = 1.24 folded in.
_STO3G_H = ((3.42525091, 0.15432897),
            (0.62391373, 0.53532814),
            (0.16885540, 0.44463454))


def _f0(x: float) -> float:
    """Boys function F0(x) = (1/2) sqrt(pi/x) erf(sqrt(x))."""
    if x < 1e-12:
        return 1.0 - x / 3.0
    return 0.5 * sqrt(pi / x) * erf(sqrt(x))


def _h2_ao_integrals(r_bohr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AO overlap, core Hamiltonian, and two-electron integrals for H2.

    Two 1s contractions on the z-axis at 0 and r_bohr. Primitive formulas are
    the standard s-orbital closed forms; contraction coefficients absorb the
    primitive normalization (2a/pi)^(3/4).
    """
    centers = (0.0, r_bohr)
    prims = []
    for c in centers:
        prims.append([(a, d * (2.0 * a / pi) ** 0.75) for a, d in _STO3G_H])

    def sep2(i: int, j: int) -> float:
        return (centers[i] - centers[j]) ** 2

    S = np.zeros((2, 2))
    T = np.zeros((2, 2))
    V = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            r2 = sep2(i, j)
            for a, ca in prims[i]:
                for b, cb in prims[j]:
                    g = a + b
                    pre = ca * cb * exp(-a * b / g * r2)
                    S[i, j] += pre * (pi / g) ** 1.5
                    T[i, j] += pre * (a * b / g) * (3.0 - 2.0 * a * b / g * r2) \
                        * (pi / g) ** 1.5
                    # Gaussian product center (z coordinate).
                    zp = (a * centers[i] + b * centers[j]) / g
                    for zc in centers:                      # both protons, Z=1
                        V[i, j] += -pre * (2.0 * pi / g) * _f0(g * (zp - zc) ** 2)

    eri = np.zeros((2, 2, 2, 2))                            # chemists (ij|kl)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for a, ca in prims[i]:
                        for b, cb in prims[j]:
                            for c, cc in prims[k]:
                                for d, cd in prims[l]:
                                    g1, g2 = a + b, c + d
                                    zp = (a * centers[i] + b * centers[j]) / g1
                                    zq = (c * centers[k] + d * centers[l]) / g2
                                    pre = (ca * cb * cc * cd
                                           * exp(-a * b / g1 * sep2(i, j))
                                           * exp(-c * d / g2 * sep2(k, l)))
                                    eri[i, j, k, l] += pre * 2.0 * pi ** 2.5 / (
                                        g1 * g2 * sqrt(g1 + g2)) * _f0(
                                        g1 * g2 / (g1 + g2) * (zp - zq) ** 2)
    return S, T + V, eri


def h2_hamiltonian(r_angstrom: float = 0.7414) -> SparsePauliOp:
    """4-qubit Jordan-Wigner SparsePauliOp for H2/STO-3G at bond length R.

    Spin-orbital order (qubit index): 0 = g up, 1 = g down, 2 = u up,
    3 = u down, with g/u the symmetric/antisymmetric molecular orbitals.
    Nuclear repulsion rides on the identity term.
    """
    from qiskit.quantum_info import Operator, SparsePauliOp

    r_bohr = r_angstrom * BOHR_PER_ANGSTROM
    S, hcore, eri = _h2_ao_integrals(r_bohr)
    s12 = S[0, 1]
    # RHF MOs by inversion symmetry: gerade and ungerade combinations.
    cg = 1.0 / sqrt(2.0 * (1.0 + s12))
    cu = 1.0 / sqrt(2.0 * (1.0 - s12))
    C = np.array([[cg, cu], [cg, -cu]])
    h_mo = C.T @ hcore @ C
    eri_mo = np.einsum('ip,jq,kr,ls,ijkl->pqrs', C, C, C, C, eri)

    # Spatial -> spin orbitals: index 2*mo + spin, spin in (0 up, 1 down).
    n_so = 4
    h_so = np.zeros((n_so, n_so))
    g_so = np.zeros((n_so, n_so, n_so, n_so))    # physicists <pq|rs>
    for p in range(n_so):
        for q in range(n_so):
            if p % 2 == q % 2:
                h_so[p, q] = h_mo[p // 2, q // 2]
            for r in range(n_so):
                for s in range(n_so):
                    if p % 2 == r % 2 and q % 2 == s % 2:
                        g_so[p, q, r, s] = eri_mo[p // 2, r // 2, q // 2, s // 2]

    # Jordan-Wigner ladder operators as dense 16x16 matrices. Qubit j is the
    # j-th kron factor from the right (little-endian, matching the display).
    I2 = np.eye(2)
    Z2 = np.diag([1.0, -1.0])
    low = np.array([[0.0, 1.0], [0.0, 0.0]])     # sigma-minus: |0><1|
    def ann(j: int) -> np.ndarray:
        out = np.array([[1.0]])
        for k in range(n_so - 1, -1, -1):
            out = np.kron(out, low if k == j else (Z2 if k < j else I2))
        return out
    a = [ann(j) for j in range(n_so)]
    ad = [m.conj().T for m in a]

    H = np.zeros((16, 16), dtype=complex)
    for p in range(n_so):
        for q in range(n_so):
            if h_so[p, q]:
                H += h_so[p, q] * (ad[p] @ a[q])
    for p in range(n_so):
        for q in range(n_so):
            for r in range(n_so):
                for s in range(n_so):
                    if g_so[p, q, r, s]:
                        H += 0.5 * g_so[p, q, r, s] * (ad[p] @ ad[q] @ a[s] @ a[r])
    e_nuc = 1.0 / r_bohr
    H += e_nuc * np.eye(16)
    op = SparsePauliOp.from_operator(Operator(H)).simplify(atol=1e-10)
    return op


# ---------------------------------------------------------------------------
# QUBASIC Pauli propagation (sparse Pauli dynamics).
#
# Estimate <O> for a Pauli observable by backpropagating it through the circuit in
# the Heisenberg picture, O -> G^dag O G gate by gate, keeping it as a sparse sum
# of Paulis and truncating terms below a coefficient threshold. This reaches far
# larger circuits than a statevector when the propagated observable stays sparse,
# and is exact (matches EXPECT) when nothing is truncated. Pure numpy, no engine.
#
#   PAULIPROP <pauli> [qubits] [threshold]
# ---------------------------------------------------------------------------

_P1 = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


class PauliPropMixin(HostState):
    """Sparse Pauli-dynamics observable estimator for QBasicTerminal.

    Requires: TerminalProtocol — uses self.num_qubits, self.build_circuit(),
    self._eval_with_vars(), self.io.
    """

    @staticmethod
    def _local_pauli_matrix(chars: Sequence[str]) -> np.ndarray:
        """Kron of single-qubit Paulis with the LAST char as most significant,
        matching Qiskit's little-endian gate-matrix convention (first qubit = LSB)."""
        mat = np.array([[1]], dtype=complex)
        for c in reversed(chars):
            mat = np.kron(mat, _P1[c])
        return mat

    def _conjugate_term(self, pauli: list, coeff: complex, qubits: list, Gdag_x_G: Callable[[np.ndarray], np.ndarray]) -> list[tuple[list[str], complex]]:
        """Conjugate one Pauli term on the given gate qubits, returning new terms.

        Gdag_x_G(localPauliMatrix) -> the matrix G^dag P G; we decompose it back
        into local Paulis via traces and distribute over the gate's qubits."""
        k = len(qubits)
        local = [pauli[q] for q in qubits]
        M = Gdag_x_G(self._local_pauli_matrix(local))
        out = []
        import itertools
        for combo in itertools.product('IXYZ', repeat=k):
            basis = self._local_pauli_matrix(combo)
            c = np.trace(basis.conj().T @ M) / (2 ** k)
            if abs(c) < 1e-12:
                continue
            new = list(pauli)
            for q, ch in zip(qubits, combo):
                new[q] = ch
            out.append((new, coeff * c))
        return out

    def cmd_pauliprop(self, rest: str) -> None:
        """PAULIPROP <pauli> [qubits] [threshold] — expectation value by Heisenberg-picture Pauli propagation, truncated below threshold.

        Backpropagates the Pauli observable through the program's circuit, pruning
        terms with |coefficient| below the threshold (default 1e-10, i.e. exact)."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: PAULIPROP <pauli> [qubits] [threshold]")
            return
        if not self.program:
            self.io.writeln("?NOTHING TO PROPAGATE — enter a program first")
            return
        n = self.num_qubits
        pauli_str = parts[0].upper()
        if set(pauli_str) - set('IXYZ'):
            self.io.writeln("?PAULIPROP: observable must be over I, X, Y, Z")
            return
        rest_toks = parts[1:]
        threshold = 1e-10
        if rest_toks and re.match(r'^[0-9.eE+-]+$', rest_toks[-1]) and (
                '.' in rest_toks[-1] or 'e' in rest_toks[-1].lower()):
            threshold = float(rest_toks[-1])
            rest_toks = rest_toks[:-1]
        qubits = [int(self._eval_with_vars(t, {})) for t in rest_toks] if rest_toks \
            else list(range(len(pauli_str)))
        try:
            # Build the qubit-indexed observable string (position i = qubit i).
            obs = ['I'] * n
            for p, q in zip(pauli_str, qubits):
                obs[q] = p
            terms = {''.join(obs): 1.0 + 0j}
            qc, _ = self.build_circuit()
            # Backpropagate through gates in reverse: O <- G^dag O G.
            max_terms = 0
            from qiskit.circuit import Gate
            for inst in qc.data:
                op = inst.operation
                if op.name != 'barrier' and not isinstance(op, Gate):
                    self.io.writeln(f"?PAULIPROP: '{op.name}' is not unitary; Heisenberg "
                                    f"propagation needs a measure-free, reset-free program")
                    return
            for inst in reversed(qc.data):
                op = inst.operation
                nm = op.name.lower()
                if nm == 'barrier':
                    continue
                try:
                    G = np.asarray(op.to_matrix(), dtype=complex)
                except Exception:
                    self.io.writeln(f"?PAULIPROP: gate '{nm}' has no matrix; unsupported")
                    return
                gq = [qc.find_bit(q).index for q in inst.qubits]
                Gd = G.conj().T

                def conj(P: np.ndarray, _G: np.ndarray = G, _Gd: np.ndarray = Gd) -> np.ndarray:
                    return _Gd @ P @ _G

                new_terms: dict = {}
                for ps, coeff in terms.items():
                    plist = list(ps)
                    for newp, c in self._conjugate_term(plist, coeff, gq, conj):
                        key = ''.join(newp)
                        new_terms[key] = new_terms.get(key, 0j) + c
                # Prune.
                terms = {k: v for k, v in new_terms.items() if abs(v) >= threshold}
                max_terms = max(max_terms, len(terms))
            # Evaluate on |0...0>: a term contributes its coeff iff every qubit is I or Z.
            val = 0.0
            for ps, coeff in terms.items():
                if set(ps) <= {'I', 'Z'}:
                    val += coeff.real
            self.io.writeln(f"  <{pauli_str}> on {qubits} = {val:.6f}  "
                            f"(Pauli propagation, peak {max_terms} terms, threshold {threshold:g})")
            self.variables['_PAULIPROP'] = float(val)
        except Exception as e:
            self.io.writeln(f"?PAULIPROP ERROR: {e}")


# ---------------------------------------------------------------------------
# QUBASIC qudit engine — d-level systems beyond qubits.
#
# A standalone numpy statevector simulator for n qudits of dimension d, with the
# generalized Pauli (shift/clock) gates, the qudit Fourier transform, and the
# controlled-sum entangler. Independent of the qubit (Aer) engine.
#
#   QUDIT <d> [n]     Initialize n qudits of dimension d in |0...0>
#   QX <q> [k]        Shift gate X^k: |j> -> |j+k mod d>
#   QZ <q> [k]        Clock gate Z^k: |j> -> w^(kj)|j>, w = exp(2 pi i / d)
#   QF <q>            Qudit Fourier transform (generalized Hadamard)
#   QSUM <c> <t>      Controlled sum: |a,b> -> |a, (b+a) mod d>
#   QSTATE            Show the qudit statevector
#   QMEASURE [shots]  Sample measurement outcomes (base-d strings)
# ---------------------------------------------------------------------------

class QuditMixin(HostState):
    """A d-level (qudit) statevector engine for QBasicTerminal.

    Requires: TerminalProtocol — uses self.shots, self._seed, self.io.
    """

    def _init_qudits(self) -> None:
        self._qd = 0
        self._qn = 0
        self._qsv: np.ndarray = np.zeros(0, dtype=complex)   # set by QUDIT

    def _qudit_apply(self, mat: np.ndarray, targets: list[int]) -> None:
        """Apply a single- or two-qudit matrix to the qudit statevector.

        Qudit q is digit q of the base-d index (qudit 0 least significant),
        the qubit convention, so a state prints with qudit 0 rightmost.
        """
        d, n = self._qd, self._qn
        k = len(targets)
        t = self._qsv.reshape([d] * n)
        axes = [n - 1 - q for q in targets]
        t = np.tensordot(mat.reshape([d] * (2 * k)), t,
                         axes=(list(range(k, 2 * k)), axes))
        t = np.moveaxis(t, list(range(k)), axes)
        self._qsv = np.ascontiguousarray(t).reshape(-1)

    def cmd_qudit(self, rest: str) -> None:
        """QUDIT <d> [n] — initialize n qudits of dimension d in |0...0>."""
        parts = rest.split()
        if not parts:
            if self._qd:
                self.io.writeln(f"QUDIT: {self._qn} qudits of dimension {self._qd}")
            else:
                self.io.writeln("?USAGE: QUDIT <d> [n]")
            return
        d = int(parts[0])
        n = int(parts[1]) if len(parts) > 1 else 1
        if not 2 <= d <= 36:
            self.io.writeln("?QUDIT dimension must be 2-36 (one digit per qudit)")
            return
        if d ** n > 1 << 20:
            self.io.writeln(f"?QUDIT state d^n = {d ** n} too large")
            return
        self._qd, self._qn = d, n
        self._qsv = np.zeros(d ** n, dtype=complex)
        self._qsv[0] = 1.0
        self.io.writeln(f"QUDIT: {n} qudit(s) of dimension {d} in |{'0' * n}>")

    def _qcheck(self, q: int) -> None:
        if not self._qd:
            raise ValueError("no qudit register (use QUDIT <d> [n] first)")
        if q < 0 or q >= self._qn:
            raise ValueError(f"qudit {q} out of range (0-{self._qn - 1})")

    def cmd_qx(self, rest: str) -> None:
        """QX <qudit> [k] — generalized shift X^k."""
        parts = rest.split()
        try:
            q = int(parts[0])
            k = int(parts[1]) if len(parts) > 1 else 1
            self._qcheck(q)
            d = self._qd
            X = np.zeros((d, d), dtype=complex)
            for j in range(d):
                X[(j + k) % d, j] = 1.0
            self._qudit_apply(X, [q])
        except Exception as e:
            self.io.writeln(f"?QX ERROR: {e}")

    def cmd_qz(self, rest: str) -> None:
        """QZ <qudit> [k] — generalized clock Z^k."""
        parts = rest.split()
        try:
            q = int(parts[0])
            k = int(parts[1]) if len(parts) > 1 else 1
            self._qcheck(q)
            d = self._qd
            w = np.exp(2j * np.pi / d)
            Z = np.diag([w ** ((k * j) % d) for j in range(d)]).astype(complex)
            self._qudit_apply(Z, [q])
        except Exception as e:
            self.io.writeln(f"?QZ ERROR: {e}")

    def cmd_qf(self, rest: str) -> None:
        """QF <qudit> — qudit Fourier transform (generalized Hadamard)."""
        try:
            q = int(rest.split()[0])
            self._qcheck(q)
            d = self._qd
            w = np.exp(2j * np.pi / d)
            F = np.array([[w ** ((i * j) % d) for j in range(d)] for i in range(d)],
                         dtype=complex) / np.sqrt(d)
            self._qudit_apply(F, [q])
        except Exception as e:
            self.io.writeln(f"?QF ERROR: {e}")

    def cmd_qsum(self, rest: str) -> None:
        """QSUM <control> <target> — |a,b> -> |a, (b+a) mod d> (qudit CNOT)."""
        parts = rest.split()
        try:
            c, t = int(parts[0]), int(parts[1])
            self._qcheck(c)
            self._qcheck(t)
            d = self._qd
            SUM = np.zeros((d * d, d * d), dtype=complex)
            for a in range(d):
                for b in range(d):
                    SUM[a * d + ((b + a) % d), a * d + b] = 1.0
            self._qudit_apply(SUM, [c, t])
        except Exception as e:
            self.io.writeln(f"?QSUM ERROR: {e}")

    def cmd_qstate(self, rest: str = '') -> None:
        """QSTATE — show non-zero amplitudes of the qudit statevector."""
        if not self._qd:
            self.io.writeln("?NO QUDIT STATE — use QUDIT <d> [n]")
            return
        d, n = self._qd, self._qn
        self.io.writeln(f"\n  Qudit state (d={d}, {n} qudit(s)):")
        for i in np.flatnonzero(np.abs(self._qsv) > 1e-9):
            amp = complex(self._qsv[i])
            digits = self._bits(np.base_repr(int(i), d).rjust(n, '0'))
            self.io.writeln(f"    |{digits}>  {amp.real:+.4f}{amp.imag:+.4f}j  "
                            f"(P={abs(amp) ** 2:.4f})")

    def cmd_qmeasure(self, rest: str = '') -> None:
        """QMEASURE [shots] — sample qudit measurement outcomes."""
        if not self._qd:
            self.io.writeln("?NO QUDIT STATE — use QUDIT <d> [n]")
            return
        d, n = self._qd, self._qn
        shots = int(rest.strip()) if rest.strip() else self.shots
        probs = np.abs(self._qsv) ** 2
        probs = probs / probs.sum()
        rng = np.random.default_rng(self._seed)
        draws = rng.choice(len(probs), size=shots, p=probs)
        unique, counts = np.unique(draws, return_counts=True)
        self.io.writeln(f"\n  QMEASURE ({shots} shots, base-{d}):")
        for idx, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
            digits = self._bits(np.base_repr(int(idx), d).rjust(n, '0'))
            self.io.writeln(f"    |{digits}>  {c}  ({100 * c / shots:.1f}%)")


# ---------------------------------------------------------------------------
# QUBASIC bosonic / continuous-variable engine.
#
# A truncated Fock-space numpy simulator for n modes, with displacement,
# squeezing, beamsplitters, and Schrodinger-cat preparation. Unitaries are built
# from their (anti-Hermitian) generators via eigendecomposition, so no external
# CV library is needed. Truncation at the Fock cutoff makes large displacements
# approximate.
#
#   BOSONIC <modes> <cutoff>   Initialize modes in vacuum, truncated at `cutoff` photons
#   DISPLACE <mode> <re> [im]  Displacement D(alpha) of a mode
#   SQUEEZE <mode> <r>         Single-mode squeezing S(r)
#   CAT <mode> <re> [im]       Prepare an even Schrodinger-cat state
#   BS <m1> <m2> <theta>       Beamsplitter between two modes
#   BSTATE [mode]              Show Fock amplitudes, <n>, and photon distribution
# ---------------------------------------------------------------------------

def _expm_anti(G: np.ndarray) -> np.ndarray:
    """exp(G) for an anti-Hermitian G, via eigendecomposition of the Hermitian iG."""
    H = 1j * G
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w)) @ V.conj().T


class BosonicMixin(HostState):
    """Continuous-variable Fock-space engine for QBasicTerminal.

    Requires: TerminalProtocol — uses self._eval_with_vars(), self.io.
    """

    def _init_bosonic(self) -> None:
        self._bmodes = 0
        self._bcut = 0
        self._bsv: np.ndarray = np.zeros(0, dtype=complex)   # set by BOSONIC

    def _annihilation(self) -> np.ndarray:
        N = self._bcut
        a = np.zeros((N, N), dtype=complex)
        for n in range(1, N):
            a[n - 1, n] = np.sqrt(n)
        return a

    def _bos_apply(self, mat: np.ndarray, modes: list[int]) -> None:
        N, m = self._bcut, self._bmodes
        k = len(modes)
        t = self._bsv.reshape([N] * m)
        t = np.tensordot(mat.reshape([N] * (2 * k)), t,
                         axes=(list(range(k, 2 * k)), list(modes)))
        t = np.moveaxis(t, list(range(k)), list(modes))
        self._bsv = np.ascontiguousarray(t).reshape(-1)

    def cmd_bosonic(self, rest: str) -> None:
        """BOSONIC <modes> <cutoff> — initialize modes in vacuum."""
        parts = rest.split()
        if len(parts) < 2:
            if self._bmodes:
                self.io.writeln(f"BOSONIC: {self._bmodes} mode(s), cutoff {self._bcut}")
            else:
                self.io.writeln("?USAGE: BOSONIC <modes> <cutoff>")
            return
        m = int(parts[0])
        cut = int(parts[1])
        if cut ** m > 1 << 20:
            self.io.writeln(f"?BOSONIC cutoff^modes = {cut ** m} too large")
            return
        self._bmodes, self._bcut = m, cut
        self._bsv = np.zeros(cut ** m, dtype=complex)
        self._bsv[0] = 1.0
        self.io.writeln(f"BOSONIC: {m} mode(s), Fock cutoff {cut}, in vacuum")

    def _bcheck(self, mode: int) -> None:
        if not self._bmodes:
            raise ValueError("no bosonic register (use BOSONIC <modes> <cutoff>)")
        if mode < 0 or mode >= self._bmodes:
            raise ValueError(f"mode {mode} out of range (0-{self._bmodes - 1})")

    def cmd_displace(self, rest: str) -> None:
        """DISPLACE <mode> <re> [im] — displacement D(alpha) = exp(alpha a^dag - alpha* a)."""
        parts = rest.split()
        try:
            mode = int(parts[0])
            self._bcheck(mode)
            alpha = complex(float(self._eval_with_vars(parts[1], {})),
                            float(self._eval_with_vars(parts[2], {})) if len(parts) > 2 else 0.0)
            a = self._annihilation()
            self._bos_apply(_expm_anti(alpha * a.conj().T - np.conj(alpha) * a), [mode])
            self._report_n(mode)
        except Exception as e:
            self.io.writeln(f"?DISPLACE ERROR: {e}")

    def cmd_squeeze(self, rest: str) -> None:
        """SQUEEZE <mode> <r> — single-mode squeezing S(r) = exp(r/2 (a^2 - a^dag^2))."""
        parts = rest.split()
        try:
            mode = int(parts[0])
            self._bcheck(mode)
            r = float(self._eval_with_vars(parts[1], {}))
            a = self._annihilation()
            self._bos_apply(_expm_anti(0.5 * r * (a @ a - a.conj().T @ a.conj().T)), [mode])
            self._report_n(mode)
        except Exception as e:
            self.io.writeln(f"?SQUEEZE ERROR: {e}")

    def cmd_bs(self, rest: str) -> None:
        """BS <m1> <m2> <theta> — beamsplitter exp(theta (a^dag b - a b^dag))."""
        parts = rest.split()
        try:
            m1, m2 = int(parts[0]), int(parts[1])
            self._bcheck(m1)
            self._bcheck(m2)
            theta = float(self._eval_with_vars(parts[2], {}))
            N = self._bcut
            a = self._annihilation()
            I = np.eye(N, dtype=complex)
            aa = np.kron(a, I)
            bb = np.kron(I, a)
            G = theta * (aa.conj().T @ bb - aa @ bb.conj().T)
            self._bos_apply(_expm_anti(G), [m1, m2])
            self.io.writeln(f"  beamsplitter theta={theta} on modes {m1},{m2}")
        except Exception as e:
            self.io.writeln(f"?BS ERROR: {e}")

    def cmd_cat(self, rest: str) -> None:
        """CAT <mode> <re> [im] — prepare an even cat state (|alpha> + |-alpha>)/N."""
        parts = rest.split()
        try:
            mode = int(parts[0])
            self._bcheck(mode)
            alpha = complex(float(self._eval_with_vars(parts[1], {})),
                            float(self._eval_with_vars(parts[2], {})) if len(parts) > 2 else 0.0)
            N = self._bcut
            a = self._annihilation()
            vac = np.zeros(N, dtype=complex)
            vac[0] = 1.0
            Dp = _expm_anti(alpha * a.conj().T - np.conj(alpha) * a)
            Dm = _expm_anti(-alpha * a.conj().T + np.conj(alpha) * a)
            cat = Dp @ vac + Dm @ vac
            cat = cat / np.linalg.norm(cat)
            # Replace the mode's state (assumes a fresh/vacuum register for clarity).
            if self._bmodes == 1:
                self._bsv = cat
            else:
                # Project the mode onto the cat by applying it as a state-prep matrix
                # from vacuum on that mode (only valid if the mode is in vacuum).
                prep = np.outer(cat, vac.conj())
                prep += np.eye(N) - np.outer(vac, vac.conj())  # identity off the vacuum
                self._bos_apply(prep, [mode])
            self.io.writeln(f"  cat state |alpha>+|-alpha>, alpha={alpha} on mode {mode}")
            self._report_n(mode)
        except Exception as e:
            self.io.writeln(f"?CAT ERROR: {e}")

    def _mode_rdm_diag(self, mode: int) -> np.ndarray:
        """Photon-number distribution (diagonal of the reduced density matrix) of a mode."""
        N, m = self._bcut, self._bmodes
        t: np.ndarray = self._bsv.reshape([N] * m)
        t = np.moveaxis(t, mode, 0).reshape(N, -1)
        return np.real(np.sum(np.abs(t) ** 2, axis=1))

    def _report_n(self, mode: int) -> None:
        diag = self._mode_rdm_diag(mode)
        nbar = float(np.sum(np.arange(self._bcut) * diag))
        self.io.writeln(f"  mode {mode}: <n> = {nbar:.4f}")

    def cmd_bstate(self, rest: str = '') -> None:
        """BSTATE [mode] — show <n> and the photon-number distribution."""
        if not self._bmodes:
            self.io.writeln("?NO BOSONIC STATE — use BOSONIC <modes> <cutoff>")
            return
        modes = [int(rest)] if rest.strip() else list(range(self._bmodes))
        for mode in modes:
            diag = self._mode_rdm_diag(mode)
            nbar = float(np.sum(np.arange(self._bcut) * diag))
            self.io.writeln(f"\n  Mode {mode}: <n> = {nbar:.4f}")
            for k in range(self._bcut):
                if diag[k] > 1e-4:
                    bar = '#' * int(40 * diag[k])
                    self.io.writeln(f"    |{k}>  {diag[k]:6.4f}  {bar}")
