"""QUBASIC Hamiltonian dynamics and open systems.

Declare a Hamiltonian as a Pauli sum (or a built-in model), evolve a register
under it with a Trotter product formula, and evolve open systems with a Lindblad
master equation. All offline on Qiskit/numpy.

  HAMILTONIAN H = 1.0 ZZ 0 1 + 0.5 X 0     Declare a Pauli-sum Hamiltonian
  HAMILTONIAN H = ISING 1.0 0.5            Transverse-field Ising (J, h)
  HAMILTONIAN H = HEISENBERG 1.0           Heisenberg XXX chain (coupling J)
  EVOLVE H, t [, steps]                    Trotterized e^{-iHt} appended to the circuit
  LINDBLAD H, t, steps, <rate> <op> <q>    Open-system master-equation evolution
"""

from __future__ import annotations

import re

import numpy as np

# Single-qubit operators usable as Lindblad jump operators.
_JUMP_OPS = {
    'SM': np.array([[0, 1], [0, 0]], dtype=complex),   # sigma-minus (decay)
    'SP': np.array([[0, 0], [1, 0]], dtype=complex),   # sigma-plus (excite)
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    'N': np.array([[0, 0], [0, 1]], dtype=complex),    # number / dephasing-ish
}


class DynamicsMixin:
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

    def _parse_pauli_sum(self, spec: str):
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

    def _build_hamiltonian(self, spec: str):
        from qiskit.quantum_info import SparsePauliOp
        n = self.num_qubits
        up = spec.upper().split()
        kind = up[0] if up else ''
        if kind == 'ISING':
            # Transverse-field Ising: -J sum Z_i Z_{i+1} - h sum X_i.
            parts = spec.split()
            J = float(self._safe_eval(parts[1])) if len(parts) > 1 else 1.0
            h = float(self._safe_eval(parts[2])) if len(parts) > 2 else 1.0
            labels, coeffs = [], []
            for i in range(n - 1):
                labels.append(self._pauli_label('ZZ', [i, i + 1])); coeffs.append(-J)
            for i in range(n):
                labels.append(self._pauli_label('X', [i])); coeffs.append(-h)
            return SparsePauliOp(labels, coeffs)
        if kind == 'HEISENBERG':
            # XXX chain: J sum (X X + Y Y + Z Z) on neighbours.
            parts = spec.split()
            J = float(self._safe_eval(parts[1])) if len(parts) > 1 else 1.0
            labels, coeffs = [], []
            for i in range(n - 1):
                for p in 'XYZ':
                    labels.append(self._pauli_label(p + p, [i, i + 1])); coeffs.append(J)
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
                labels.append(self._pauli_label('XX', [i, i + 1])); coeffs.append(-tt / 2)
                labels.append(self._pauli_label('YY', [i, i + 1])); coeffs.append(-tt / 2)
                if V:
                    # V n_i n_{i+1} = V/4 (I - Z_i - Z_{i+1} + Z_i Z_{i+1})
                    labels.append(self._pauli_label('II', [i, i + 1])); coeffs.append(V / 4)
                    labels.append(self._pauli_label('Z', [i])); coeffs.append(-V / 4)
                    labels.append(self._pauli_label('Z', [i + 1])); coeffs.append(-V / 4)
                    labels.append(self._pauli_label('ZZ', [i, i + 1])); coeffs.append(V / 4)
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
                labels.append(self._pauli_label('X', [i])); coeffs.append(omega / 2)
                labels.append(self._pauli_label('Z', [i])); coeffs.append(delta / 2)
            for i in range(n - 1):
                labels.append(self._pauli_label('II', [i, i + 1])); coeffs.append(V / 4)
                labels.append(self._pauli_label('Z', [i])); coeffs.append(-V / 4)
                labels.append(self._pauli_label('Z', [i + 1])); coeffs.append(-V / 4)
                labels.append(self._pauli_label('ZZ', [i, i + 1])); coeffs.append(V / 4)
            return SparsePauliOp(labels, coeffs).simplify()
        return self._parse_pauli_sum(spec)

    # ── User-defined Kraus channels ─────────────────────────────────────

    def cmd_channel(self, rest: str) -> None:
        """CHANNEL <name> = K0 ; K1 ; ... — define a quantum channel by its Kraus operators.

        Each Ki is a matrix literal (single- or multi-qubit). The CPTP condition
        sum Ki^dag Ki = I is checked. Apply it with APPLYCHANNEL <name> <qubit(s)>;
        a non-unitary channel routes the run through the density_matrix method."""
        m = re.match(r'(\w+)\s*=\s*(.+)$', rest.strip())
        if not m:
            self.io.writeln("?USAGE: CHANNEL <name> = <K0> ; <K1> ; ...")
            return
        name = m.group(1).upper()
        try:
            ops = [np.array(self._parse_matrix(k.strip()), dtype=complex)
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

    def _try_exec_applychannel(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
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
        qc.append(Kraus(channels[name]).to_instruction(), qubits)
        # A non-unitary channel needs a method that propagates mixed states.
        if self.sim_method in ('automatic', 'statevector', 'stabilizer'):
            self.sim_method = 'density_matrix'
        return True

    def cmd_hamiltonian(self, rest: str) -> None:
        """HAMILTONIAN <name> = <pauli sum | ISING J h | HEISENBERG J> — declare a Hamiltonian.

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

    def _try_exec_evolve(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
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

    def _embed_op(self, op1: np.ndarray, qubit: int) -> np.ndarray:
        """Embed a single-qubit operator on `qubit` into the full n-qubit space."""
        n = self.num_qubits
        mats = [op1 if i == qubit else np.eye(2, dtype=complex) for i in range(n)]
        # Qubit 0 is the least significant; build kron with qubit n-1 outermost.
        full = np.array([[1]], dtype=complex)
        for i in range(n - 1, -1, -1):
            full = np.kron(full, mats[i])
        return full

    def cmd_lindblad(self, rest: str) -> None:
        """LINDBLAD <H|NONE>, <time>, <steps>, <rate> <op> <q> [; ...] — open-system evolution.

        Integrates drho/dt = -i[H, rho] + sum_k g_k (L_k rho L_k^dag - {L_k^dag L_k, rho}/2)
        with RK4 from the current state (or |0...0>). Jump operators are single-qubit:
        SM (decay), SP (excite), X, Y, Z, N. Reports the final density matrix.
        Example: LINDBLAD NONE, 1.0, 200, 1.0 SM 0"""
        parts = [p.strip() for p in rest.split(',')]
        if len(parts) < 4:
            self.io.writeln("?USAGE: LINDBLAD <H|NONE>, <time>, <steps>, <rate> <op> <q> [; ...]")
            return
        n = self.num_qubits
        dim = 2 ** n
        if n > 5:
            self.io.writeln(f"?LINDBLAD limited to 5 qubits (dense {dim}x{dim} rho); have {n}")
            return
        try:
            hname = parts[0].upper()
            if hname in ('NONE', '0', ''):
                H = np.zeros((dim, dim), dtype=complex)
            else:
                if hname not in self._hamiltonians:
                    raise ValueError(f"unknown HAMILTONIAN '{hname}'")
                H = np.asarray(self._hamiltonians[hname].to_matrix(), dtype=complex)
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
                jumps.append((np.sqrt(rate) * self._embed_op(_JUMP_OPS[opname], q)))
            # Initial density matrix from the current statevector, else |0...0>.
            if self.last_sv is not None and np.asarray(self.last_sv).size == dim:
                psi = np.ascontiguousarray(self.last_sv).ravel()
                psi = psi / (np.linalg.norm(psi) or 1.0)
                rho = np.outer(psi, psi.conj())
            else:
                rho = np.zeros((dim, dim), dtype=complex)
                rho[0, 0] = 1.0
            precomp = [(L, L.conj().T @ L) for L in jumps]

            def drho(r):
                out = -1j * (H @ r - r @ H)
                for L, LdL in precomp:
                    out += L @ r @ L.conj().T - 0.5 * (LdL @ r + r @ LdL)
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
            self.io.writeln(f"  Populations: " + ', '.join(
                f"|{format(i, f'0{n}b')}>={pops[i]:.4f}" for i in range(dim) if pops[i] > 1e-4))
            purity = float(np.real(np.trace(rho @ rho)))
            self.io.writeln(f"  Purity Tr(rho^2) = {purity:.6f}")
            for i in range(dim):
                self.variables[f'_RHO{i}'] = float(pops[i])
        except Exception as e:
            self.io.writeln(f"?LINDBLAD ERROR: {e}")
