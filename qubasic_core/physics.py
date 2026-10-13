"""QUBASIC physical models: Hamiltonians, channels and Lindblad
evolution, molecules in STO-3G, Pauli propagation, qudits and bosonic
modes.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from qiskit import QuantumCircuit

from qubasic_core.host import SimulationHost

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


class DynamicsMixin(SimulationHost):
    """Hamiltonian declaration, Trotter evolution, and Lindblad open dynamics."""

    def _init_dynamics(self) -> None:
        self._hamiltonians: dict = {}   # name -> SparsePauliOp
        self._ham_molecules: dict[str, Molecule] = {}     # the MOLECULE ones, by name
        self._molecule_built: Molecule | None = None

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
            mol = parse_molecule(spec.strip()[len('MOLECULE'):].strip() or 'H2', self._safe_eval)
            self._molecule_built = mol
            return mol.hamiltonian
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
        """HAMILTONIAN name = pauli sum | ISING J h | HEISENBERG J | HUBBARD t V | RYDBERG Omega Delta V | MOLECULE <molecule> [R] [CHARGE q] [FREEZE [n]] [ACTIVE e o] — declare a Hamiltonian.

        A Pauli sum is terms joined by '+', each '<coeff> <pauli> <qubits>', e.g.
        HAMILTONIAN H = 1.0 ZZ 0 1 + 0.5 X 0. Use it with EVOLVE or LINDBLAD.
        MOLECULE builds a molecule's STO-3G qubit Hamiltonian: H2, HEH+, LIH,
        BEH2, H2O, NH3, N2 or H4 at bond length R in angstroms, or a geometry
        in quotes ("O 0 0 0; H 0.76 0.59 0; H -0.76 0.59 0"). FREEZE drops
        the core orbitals and ACTIVE keeps e electrons in o orbitals; VQE
        ANSATZ UCCSD reads the electron count it records."""
        m = re.match(r'(\w+)\s*=\s*(.+)$', rest.strip())
        if not m:
            self.io.writeln("?USAGE: HAMILTONIAN <name> = <terms>")
            return
        name = m.group(1).upper()
        try:
            self._molecule_built = None
            op = self._build_hamiltonian(m.group(2).strip())
            self._hamiltonians[name] = op
            self.io.writeln(f"HAMILTONIAN {name}: {len(op)} term(s), {op.num_qubits} qubits")
            mol = self._molecule_built
            if mol is None:
                self._ham_molecules.pop(name, None)
                return
            self._ham_molecules[name] = mol
            core = f", {mol.frozen} core orbital(s) frozen" if mol.frozen else ""
            self.io.writeln(f"  {mol.label}, STO-3G: {mol.electrons} electrons in {mol.orbitals} "
                            f"orbitals{core}")
            exact = ""
            if mol.exact_energy is not None:
                exact = f", {'FCI' if mol.complete else 'CASCI'} {mol.exact_energy:.6f} Ha"
                self.variables['_EXACT_ENERGY'] = mol.exact_energy
            self.io.writeln(f"  Hartree-Fock {mol.hf_energy:.6f} Ha{exact}")
            self.variables['_HF_ENERGY'] = mol.hf_energy
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
# QUBASIC molecular Hamiltonians — a self-contained STO-3G engine.
#
# HAMILTONIAN H = MOLECULE LIH builds a molecule's qubit Hamiltonian with no
# external chemistry package: STO-3G basis functions for H to Ne (Hehre,
# Stewart and Pople's universal 1s and 2sp expansions scaled by each
# element's Slater exponent), McMurchie-Davidson integrals over Cartesian s
# and p Gaussians (Hermite expansion coefficients and the Boys function),
# restricted Hartree-Fock with DIIS, an optional frozen core and active
# space, and the Jordan-Wigner transform of the second-quantized Hamiltonian
# over interleaved spin orbitals (qubit 2p spin up, 2p+1 spin down, so the
# Hartree-Fock state fills the lowest qubits). Nuclear repulsion and the
# frozen core ride on the identity coefficient.
#
# Ground truth: water at Crawford's reference geometry gives the textbook
# STO-3G SCF energy, -74.94208 Ha, and H2 at 0.7414 A the FCI energy
# -1.13727 Ha.
# ---------------------------------------------------------------------------

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

# The universal STO-3G expansions of a Slater 1s and 2sp function with
# exponent 1: (Gaussian exponent, coefficient) and (exponent, 2s, 2p).
_STO3G_1S = ((2.227660584, 0.1543289673), (0.4057711562, 0.5353281423),
             (0.1098175104, 0.4446345422))
_STO3G_2SP = ((0.9942027300, -0.09996722919, 0.1559162750),
              (0.2310313333, 0.3995128261, 0.6076837186),
              (0.0751385600, 0.7001154689, 0.3919573931))
# Element: (nuclear charge, 1s Slater exponent, 2sp Slater exponent).
_ELEMENTS = {
    'H': (1, 1.24, None), 'HE': (2, 1.69, None), 'LI': (3, 2.69, 0.80),
    'BE': (4, 3.68, 1.15), 'B': (5, 4.68, 1.50), 'C': (6, 5.67, 1.72),
    'N': (7, 6.67, 1.95), 'O': (8, 7.66, 2.25), 'F': (9, 8.65, 2.55),
    'NE': (10, 9.64, 2.88),
}
# Named molecules: (geometry in angstroms from the bond length R, default R,
# charge). Water and ammonia take their experimental angles.
_MOLECULES: dict[str, tuple[Callable[[float], list[tuple[str, tuple[float, float, float]]]], float, int]] = {
    'H2': (lambda r: [('H', (0, 0, 0)), ('H', (0, 0, r))], 0.7414, 0),
    'HEH+': (lambda r: [('He', (0, 0, 0)), ('H', (0, 0, r))], 0.772, 1),
    'LIH': (lambda r: [('Li', (0, 0, 0)), ('H', (0, 0, r))], 1.5949, 0),
    'BEH2': (lambda r: [('Be', (0, 0, 0)), ('H', (0, 0, r)), ('H', (0, 0, -r))], 1.3264, 0),
    'H2O': (lambda r: [('O', (0, 0, 0)), ('H', (r * sin(0.9114), r * cos(0.9114), 0)),
                       ('H', (-r * sin(0.9114), r * cos(0.9114), 0))], 0.9584, 0),
    'NH3': (lambda r: [('N', (0, 0, 0))] + [
        ('H', (r * 0.9391 * cos(2 * pi * k / 3), r * 0.9391 * sin(2 * pi * k / 3), -r * 0.3437))
        for k in range(3)], 1.0124, 0),
    'N2': (lambda r: [('N', (0, 0, 0)), ('N', (0, 0, r))], 1.0977, 0),
    'H4': (lambda r: [('H', (0, 0, k * r)) for k in range(4)], 0.9, 0),
}


def _hermite_e(i: int, j: int, t: int, qx: float, a: np.ndarray, b: np.ndarray) -> Any:
    """McMurchie-Davidson coefficient E_t^{ij} of the product of two 1D
    Gaussians with exponents a and b whose centres differ by qx."""
    if t < 0 or t > i + j:
        return 0.0
    p = a + b
    q = a * b / p
    if i == j == t == 0:
        return np.exp(-q * qx * qx)
    if j == 0:
        return (_hermite_e(i - 1, j, t - 1, qx, a, b) / (2 * p)
                - q * qx / a * _hermite_e(i - 1, j, t, qx, a, b)
                + (t + 1) * _hermite_e(i - 1, j, t + 1, qx, a, b))
    return (_hermite_e(i, j - 1, t - 1, qx, a, b) / (2 * p)
            + q * qx / b * _hermite_e(i, j - 1, t, qx, a, b)
            + (t + 1) * _hermite_e(i, j - 1, t + 1, qx, a, b))


def _hermite_r(tmax: int, umax: int, vmax: int, p: np.ndarray, pc: np.ndarray) -> dict:
    """Hermite Coulomb integrals R_tuv for t <= tmax, u <= umax, v <= vmax,
    elementwise over arrays of exponents p and separations pc (last axis xyz)."""
    from scipy.special import hyp1f1
    arg = p * np.sum(pc * pc, axis=-1)
    boys = [(-2 * p) ** n * hyp1f1(n + 0.5, n + 1.5, -arg) / (2 * n + 1)
            for n in range(tmax + umax + vmax + 1)]
    memo: dict = {}

    def r(t: int, u: int, v: int, n: int) -> Any:
        if t < 0 or u < 0 or v < 0:
            return 0.0
        key = (t, u, v, n)
        if key not in memo:
            if t == u == v == 0:
                memo[key] = boys[n]
            elif t == u == 0:
                memo[key] = (v - 1) * r(t, u, v - 2, n + 1) + pc[..., 2] * r(t, u, v - 1, n + 1)
            elif t == 0:
                memo[key] = (u - 1) * r(t, u - 2, v, n + 1) + pc[..., 1] * r(t, u - 1, v, n + 1)
            else:
                memo[key] = (t - 1) * r(t - 2, u, v, n + 1) + pc[..., 0] * r(t - 1, u, v, n + 1)
        return memo[key]

    return {(t, u, v): r(t, u, v, 0) for t in range(tmax + 1)
            for u in range(umax + 1) for v in range(vmax + 1)}


class _Pair:
    """Two contracted Gaussians as arrays over their primitive pairs: the
    exponents, product centres, coefficient products and the Hermite
    coefficients of each axis."""

    def __init__(self, fa: tuple, fb: tuple) -> None:
        ca, self.la, ea, da = fa
        cb, self.lb, eb, db = fb
        self.a = np.repeat(ea, len(eb))
        self.b = np.tile(eb, len(ea))
        self.coef = np.repeat(da, len(db)) * np.tile(db, len(da))
        self.p = self.a + self.b
        self.centre = (self.a[:, None] * ca + self.b[:, None] * cb) / self.p[:, None]
        self.q = ca - cb
        self.e = [[_hermite_e(self.la[k], self.lb[k], t, self.q[k], self.a, self.b)
                   for t in range(self.la[k] + self.lb[k] + 1)] for k in range(3)]


def _sto3g_basis(atoms: list[tuple[str, np.ndarray]]) -> list[tuple]:
    """Contracted STO-3G functions (centre, (l, m, n), exponents, normalized
    coefficients) for atoms given in bohr."""
    raw = []
    for sym, xyz in atoms:
        _, z1s, z2sp = _ELEMENTS[sym]
        raw.append((xyz, (0, 0, 0), np.array([a for a, _ in _STO3G_1S]) * z1s ** 2,
                    np.array([d for _, d in _STO3G_1S])))
        if z2sp is not None:
            exps = np.array([a for a, _, _ in _STO3G_2SP]) * z2sp ** 2
            raw.append((xyz, (0, 0, 0), exps, np.array([d for _, d, _ in _STO3G_2SP])))
            for lmn in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                raw.append((xyz, lmn, exps, np.array([d for _, _, d in _STO3G_2SP])))
    basis = []
    for centre, lmn, exps, coefs in raw:
        norm = (2 * exps / pi) ** 0.75 * (4 * exps) ** (sum(lmn) / 2)
        f = (centre, lmn, exps, coefs * norm)
        pair = _Pair(f, f)
        s = float(np.sum(pair.coef * pair.e[0][0] * pair.e[1][0] * pair.e[2][0]
                         * (pi / pair.p) ** 1.5))
        basis.append((centre, lmn, exps, coefs * norm / sqrt(s)))
    return basis


def _one_electron(basis: list[tuple], nuclei: list[tuple[int, np.ndarray]]
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Overlap and core Hamiltonian (kinetic plus nuclear attraction)."""
    n = len(basis)
    S, H = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pr = _Pair(basis[i], basis[j])
            la, lb = pr.la, pr.lb

            def ov(k: int, l2: int, pr: _Pair = pr) -> Any:
                return _hermite_e(pr.la[k], l2, 0, pr.q[k], pr.a, pr.b) if l2 >= 0 else 0.0

            base = (pi / pr.p) ** 1.5
            s = [ov(k, lb[k]) for k in range(3)]
            S[i, j] = np.sum(pr.coef * s[0] * s[1] * s[2] * base)
            kin = 0.0
            for k in range(3):
                l2 = lb[k]
                d2 = (pr.b * (2 * l2 + 1) * s[k] - 2 * pr.b ** 2 * ov(k, l2 + 2)
                      - 0.5 * l2 * (l2 - 1) * ov(k, l2 - 2))
                kin = kin + d2 * s[(k + 1) % 3] * s[(k + 2) % 3]
            H[i, j] = np.sum(pr.coef * kin * base)
            for charge, c in nuclei:
                rr = _hermite_r(la[0] + lb[0], la[1] + lb[1], la[2] + lb[2], pr.p, pr.centre - c)
                acc = sum(pr.e[0][t] * pr.e[1][u] * pr.e[2][v] * rr[(t, u, v)]
                          for t in range(la[0] + lb[0] + 1) for u in range(la[1] + lb[1] + 1)
                          for v in range(la[2] + lb[2] + 1))
                H[i, j] -= charge * np.sum(pr.coef * 2 * pi / pr.p * acc)
    return S, H


def _two_electron(basis: list[tuple]) -> np.ndarray:
    """Electron repulsion integrals (ij|kl) in chemists' notation, each
    unique one computed once over all its primitive quartets."""
    n = len(basis)
    keys = [(i, j) for i in range(n) for j in range(i + 1)]
    pairs = {key: _Pair(basis[key[0]], basis[key[1]]) for key in keys}
    eri = np.zeros((n, n, n, n))
    for x, (i, j) in enumerate(keys):
        p1 = pairs[(i, j)]
        l1 = [len(p1.e[k]) - 1 for k in range(3)]
        for k, l in keys[:x + 1]:
            p2 = pairs[(k, l)]
            l2 = [len(p2.e[m]) - 1 for m in range(3)]
            pp, qq = p1.p[:, None], p2.p[None, :]
            alpha = pp * qq / (pp + qq)
            rr = _hermite_r(l1[0] + l2[0], l1[1] + l2[1], l1[2] + l2[2], alpha,
                            p1.centre[:, None, :] - p2.centre[None, :, :])
            acc = 0.0
            for t in range(l1[0] + 1):
                for u in range(l1[1] + 1):
                    for v in range(l1[2] + 1):
                        left = (p1.e[0][t] * p1.e[1][u] * p1.e[2][v])[:, None]
                        for tau in range(l2[0] + 1):
                            for nu in range(l2[1] + 1):
                                for phi in range(l2[2] + 1):
                                    right = (p2.e[0][tau] * p2.e[1][nu] * p2.e[2][phi])[None, :]
                                    term = left * right * rr[(t + tau, u + nu, v + phi)]
                                    acc = acc - term if (tau + nu + phi) % 2 else acc + term
            val = float(np.sum(p1.coef[:, None] * p2.coef[None, :] * 2 * pi ** 2.5
                               / (pp * qq * np.sqrt(pp + qq)) * acc))
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    eri[a, b, c, d] = eri[c, d, a, b] = val
    return eri


def _scf(S: np.ndarray, H: np.ndarray, eri: np.ndarray, n_occ: int, X: np.ndarray,
         guess: np.ndarray, max_iter: int) -> tuple[float, np.ndarray] | None:
    """Roothaan-Hall iterations with DIIS from the orbitals of a guessed Fock
    matrix; None when they do not converge."""
    C = X @ np.linalg.eigh(X.T @ guess @ X)[1]
    P = 2 * C[:, :n_occ] @ C[:, :n_occ].T
    fock_hist: list[np.ndarray] = []
    err_hist: list[np.ndarray] = []
    e_old = None
    for _ in range(max_iter):
        F = H + np.einsum('pqrs,rs->pq', eri, P) - 0.5 * np.einsum('prqs,rs->pq', eri, P)
        e_elec = 0.5 * float(np.sum(P * (H + F)))
        err = F @ P @ S - S @ P @ F
        if e_old is not None and abs(e_elec - e_old) < 1e-12 and np.max(np.abs(err)) < 1e-9:
            return e_elec, C
        e_old = e_elec
        fock_hist = (fock_hist + [F])[-8:]
        err_hist = (err_hist + [err])[-8:]
        m = len(fock_hist)
        if m > 1:
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0.0
            B[:m, :m] = [[float(np.sum(a * b)) for b in err_hist] for a in err_hist]
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                weights = np.linalg.solve(B, rhs)[:m]
                F = sum(w * f for w, f in zip(weights, fock_hist))
            except np.linalg.LinAlgError:
                pass
        C = X @ np.linalg.eigh(X.T @ F @ X)[1]
        P = 2 * C[:, :n_occ] @ C[:, :n_occ].T
    return None


def rhf(S: np.ndarray, H: np.ndarray, eri: np.ndarray, n_occ: int,
        max_iter: int = 200) -> tuple[float, np.ndarray]:
    """Closed-shell Hartree-Fock, (electronic energy, molecular orbital
    coefficients by orbital energy): DIIS iterations from the core
    Hamiltonian and from the generalized Wolfsberg-Helmholz guess
    (F_ij = 1.75 S_ij (H_ii + H_jj) / 2), keeping the lower solution."""
    evals, evecs = np.linalg.eigh(S)
    X = evecs @ np.diag(evals ** -0.5) @ evecs.T
    diag = np.diag(H)
    gwh = 0.875 * S * (diag[:, None] + diag[None, :])
    np.fill_diagonal(gwh, diag)
    found = [r for r in (_scf(S, H, eri, n_occ, X, g, max_iter) for g in (H, gwh)) if r is not None]
    if not found:
        raise ValueError("Hartree-Fock did not converge")
    return min(found, key=lambda r: r[0])


def jw_ladder(n: int) -> list[SparsePauliOp]:
    """Jordan-Wigner annihilation operators on n qubits: a_j = (X_j + iY_j)/2
    times Z on every lower qubit."""
    from qiskit.quantum_info import SparsePauliOp
    return [SparsePauliOp(['I' * (n - 1 - j) + 'X' + 'Z' * j, 'I' * (n - 1 - j) + 'Y' + 'Z' * j],
                          [0.5, 0.5j]) for j in range(n)]


def _jordan_wigner(h: np.ndarray, eri: np.ndarray, e_const: float) -> SparsePauliOp:
    """E + sum h_pq a+p aq + 1/2 sum (pr|qs) a+p a+q as ar over the
    interleaved spin orbitals of spatial integrals h and eri."""
    from qiskit.quantum_info import SparsePauliOp
    n = 2 * h.shape[0]
    a = jw_ladder(n)
    ad = [op.adjoint() for op in a]
    terms = [SparsePauliOp('I' * n, [e_const])]
    terms += [h[p // 2, q // 2] * (ad[p] @ a[q]) for p in range(n) for q in range(n)
              if p % 2 == q % 2 and abs(h[p // 2, q // 2]) > 1e-12]
    create = {(p, q): ad[p] @ ad[q] for p in range(n) for q in range(n) if p != q}
    destroy = {(s, r): a[s] @ a[r] for s in range(n) for r in range(n) if s != r}
    for (p, q), cpq in create.items():
        for (s, r), asr in destroy.items():
            if r % 2 == p % 2 and s % 2 == q % 2:
                g = eri[p // 2, r // 2, q // 2, s // 2]
                if abs(g) > 1e-12:
                    terms.append(0.5 * g * (cpq @ asr))
    return SparsePauliOp.sum(terms).simplify(atol=1e-10)


def fermion_sector(n_so: int, n_elec: int) -> np.ndarray:
    """The determinants of n_elec electrons, half spin up (even qubits) and
    half down (odd), over n_so interleaved spin orbitals, as sorted integers."""
    from itertools import combinations
    up, down = range(0, n_so, 2), range(1, n_so, 2)
    k = n_elec // 2
    dets = [sum(1 << q for q in cu + cd) for cu in combinations(up, n_elec - k)
            for cd in combinations(down, k)]
    return np.array(sorted(dets), dtype=np.int64)


def sector_matrix(op: SparsePauliOp, basis: np.ndarray) -> Any:
    """The real part of a Pauli operator restricted to the basis states in
    `basis` (sorted integers), as a sparse matrix; the operator must map the
    span of `basis` into itself."""
    import scipy.sparse as sps
    n = op.num_qubits
    weights = 1 << np.arange(n, dtype=np.int64)
    xs = op.paulis.x.astype(np.int64) @ weights
    zs = op.paulis.z.astype(np.int64) @ weights
    ny = np.count_nonzero(op.paulis.x & op.paulis.z, axis=1)
    phases = (-1j) ** ((op.paulis.phase + ny) % 4) * op.coeffs
    rows, cols, vals = [], [], []
    idx = np.arange(len(basis))
    for xm, zm, ph in zip(xs, zs, phases):
        target = basis ^ xm
        pos = np.searchsorted(basis, target)
        pos[pos == len(basis)] = 0
        keep = basis[pos] == target
        parity = np.zeros(len(basis), dtype=np.int64)
        bits = target & zm
        while bits.any():
            parity ^= bits & 1
            bits = bits >> 1
        rows.append(pos[keep])
        cols.append(idx[keep])
        vals.append(np.real(ph * (1 - 2 * parity[keep])))
    m = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(basis), len(basis)))
    m.eliminate_zeros()
    return m


def sector_ground(m: Any) -> float:
    """The lowest eigenvalue of a real symmetric sparse matrix."""
    if m.shape[0] <= 400:
        return float(np.linalg.eigvalsh(m.toarray())[0])
    from scipy.sparse.linalg import eigsh
    return float(eigsh(m, k=1, which='SA')[0][0])


@dataclass
class Molecule:
    """A molecular qubit Hamiltonian and what it describes: the active
    electrons and spatial orbitals (two qubits each), the frozen core, whether
    the active space is every orbital, and the Hartree-Fock energy and the
    exact energy within the active space, in hartree."""
    label: str
    hamiltonian: SparsePauliOp
    electrons: int
    orbitals: int
    frozen: int
    complete: bool
    hf_energy: float
    exact_energy: float | None


def build_molecule(atoms: list[tuple[str, tuple[float, float, float]]], *, charge: int = 0,
                   frozen: int = 0, active: tuple[int, int] | None = None,
                   label: str = 'molecule') -> Molecule:
    """The STO-3G qubit Hamiltonian of atoms placed in angstroms. `frozen`
    core orbitals are filled and dropped, and active=(electrons, orbitals)
    keeps that many electrons in the orbitals around the Fermi level."""
    atoms_bohr = []
    for sym, xyz in atoms:
        key = sym.upper()
        if key not in _ELEMENTS:
            raise ValueError(f"STO-3G here covers H to Ne, not {sym}")
        atoms_bohr.append((key, np.array(xyz, dtype=float) * BOHR_PER_ANGSTROM))
    nuclei = [(_ELEMENTS[s][0], xyz) for s, xyz in atoms_bohr]
    n_elec = sum(z for z, _ in nuclei) - charge
    if n_elec <= 0 or n_elec % 2:
        raise ValueError(f"{n_elec} electrons: the Hartree-Fock reference here is closed-shell "
                         f"(an even number of electrons)")
    basis = _sto3g_basis(atoms_bohr)
    S, H = _one_electron(basis, nuclei)
    eri = _two_electron(basis)
    e_nuc = sum(nuclei[i][0] * nuclei[j][0] / float(np.linalg.norm(nuclei[i][1] - nuclei[j][1]))
                for i in range(len(nuclei)) for j in range(i))
    e_elec, C = rhf(S, H, eri, n_elec // 2)
    n_mo = len(basis)
    if active is not None:
        act_e, act_o = active
        if act_e % 2 or act_e > n_elec or act_e < 0:
            raise ValueError(f"ACTIVE needs an even number of electrons up to {n_elec}")
        frozen = (n_elec - act_e) // 2
        if act_o < act_e // 2 or frozen + act_o > n_mo:
            raise ValueError(f"ACTIVE {act_e} {act_o}: {act_e} electrons need at least "
                             f"{act_e // 2} orbitals and at most {n_mo - frozen} are free")
    else:
        if not 0 <= frozen < n_elec // 2 + (n_elec == 0):
            raise ValueError(f"cannot freeze {frozen} orbitals of {n_elec // 2} occupied")
        act_e, act_o = n_elec - 2 * frozen, n_mo - frozen
    h_mo = C.T @ H @ C
    eri_mo = np.einsum('ip,jq,kr,ls,ijkl->pqrs', C, C, C, C, eri, optimize=True)
    core, act = slice(0, frozen), slice(frozen, frozen + act_o)
    e_core = (2 * np.trace(h_mo[core, core]) + 2 * np.einsum('iijj->', eri_mo[core, core, core, core])
              - np.einsum('ijji->', eri_mo[core, core, core, core]))
    h_act = (h_mo[act, act] + 2 * np.einsum('pqcc->pq', eri_mo[act, act, core, core])
             - np.einsum('pccq->pq', eri_mo[act, core, core, act]))
    op = _jordan_wigner(h_act, eri_mo[act, act, act, act], e_nuc + float(e_core))
    fci = None
    basis_dets = fermion_sector(2 * act_o, act_e)
    if len(basis_dets) <= 50_000:
        fci = sector_ground(sector_matrix(op, basis_dets))
    return Molecule(label, op, act_e, act_o, frozen, frozen == 0 and act_o == n_mo,
                    e_elec + e_nuc, fci)


def parse_molecule(spec: str, evaluate: Callable[[str], Any]) -> Molecule:
    """MOLECULE <name> [R] | MOLECULE "<atom x y z; ...>" (angstroms), then
    [CHARGE q] [FREEZE [n]] [ACTIVE electrons orbitals]."""
    m = re.match(r'\s*("[^"]*"|\S+)\s*(.*)$', spec)
    if not m:
        raise ValueError("MOLECULE needs a molecule: " + ', '.join(_MOLECULES) + ' or "atom x y z; ..."')
    what, tail = m.group(1), m.group(2)
    opts: dict[str, Any] = {}
    for key, count in (('CHARGE', 1), ('ACTIVE', 2), ('FREEZE', None)):
        pat = rf'\b{key}\b((?:\s+[-+]?[\w.()*/+-]+){{{count}}})' if count else rf'\b{key}\b(\s+\d+)?'
        mm = re.search(pat, tail, re.IGNORECASE)
        if mm:
            opts[key] = mm.group(1).split() if mm.group(1) else []
            tail = tail[:mm.start()] + tail[mm.end():]
    rest = tail.split()
    charge = int(evaluate(opts['CHARGE'][0])) if 'CHARGE' in opts else 0
    if what.startswith('"'):
        atoms = []
        for part in what.strip('"').split(';'):
            toks = part.split()
            if not toks:
                continue
            if len(toks) != 4:
                raise ValueError(f"atom '{part.strip()}' is not 'symbol x y z'")
            atoms.append((toks[0], (float(evaluate(toks[1])), float(evaluate(toks[2])),
                                    float(evaluate(toks[3])))))
        label = ' '.join(a for a, _ in atoms)
        if rest:
            raise ValueError(f"unexpected '{' '.join(rest)}' after the geometry")
    else:
        key = what.upper()
        if key not in _MOLECULES:
            raise ValueError(f"unknown molecule {what}: use one of {', '.join(_MOLECULES)} "
                             f"or a geometry in quotes")
        geometry, r0, charge0 = _MOLECULES[key]
        if len(rest) > 1:
            raise ValueError(f"unexpected '{' '.join(rest[1:])}' after {what} {rest[0]}")
        r = float(evaluate(rest[0])) if rest else r0
        atoms = geometry(r)
        charge = charge if 'CHARGE' in opts else charge0
        label = f"{key} at R = {r:g} A"
    frozen = 0
    if 'FREEZE' in opts:
        frozen = int(opts['FREEZE'][0]) if opts['FREEZE'] else sum(
            1 for sym, _ in atoms if _ELEMENTS.get(sym.upper(), (0, 0, None))[2] is not None)
    active = None
    if 'ACTIVE' in opts:
        active = (int(evaluate(opts['ACTIVE'][0])), int(evaluate(opts['ACTIVE'][1])))
    return build_molecule(atoms, charge=charge, frozen=frozen, active=active, label=label)


def h2_hamiltonian(r_angstrom: float = 0.7414) -> SparsePauliOp:
    """The 4-qubit Jordan-Wigner Hamiltonian of H2 in STO-3G at bond length R."""
    geometry, _, _ = _MOLECULES['H2']
    return build_molecule(geometry(r_angstrom), label='H2').hamiltonian


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


class PauliPropMixin(SimulationHost):
    """Sparse Pauli-dynamics observable estimator for QBasicTerminal."""

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

class QuditMixin(SimulationHost):
    """A d-level (qudit) statevector engine for QBasicTerminal."""

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


class BosonicMixin(SimulationHost):
    """Continuous-variable Fock-space engine for QBasicTerminal."""

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
        """CAT <mode> <re> [im] — put a mode in an even cat state (|alpha> + |-alpha>)/N, keeping the other modes; a mode entangled with them is refused."""
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
            self._replace_mode(mode, cat)
            self.io.writeln(f"  cat state |alpha>+|-alpha>, alpha={alpha} on mode {mode}")
            self._report_n(mode)
        except Exception as e:
            self.io.writeln(f"?CAT ERROR: {e}")

    def _replace_mode(self, mode: int, state: np.ndarray) -> None:
        """Put one mode in `state`, keeping the other modes' state; a mode
        entangled with the others has no state of its own to replace."""
        N, m = self._bcut, self._bmodes
        t = np.moveaxis(self._bsv.reshape([N] * m), mode, 0).reshape(N, -1)
        _, s, vh = np.linalg.svd(t, full_matrices=False)
        if len(s) > 1 and s[1] > 1e-6:
            raise ValueError(f"mode {mode} is entangled with the other modes")
        rest = s[0] * vh[0]
        joint = np.outer(state, rest).reshape([N] * m)
        self._bsv = np.moveaxis(joint, 0, mode).reshape(-1)

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
