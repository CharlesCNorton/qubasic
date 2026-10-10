"""LOCC quantum engine — multi-register simulation with classical channels."""

from __future__ import annotations

import numpy as np
from qubasic_core.gates import (
    _np_gate_matrix, _apply_gate_inplace, _measure_inplace, _marginal_p1,
    _sample_counts,
)

# ═══════════════════════════════════════════════════════════════════════
# LOCC-specific constants
# ═══════════════════════════════════════════════════════════════════════

LOCC_MAX_JOINT_QUBITS = 33
LOCC_MAX_SPLIT_QUBITS = 33
LOCC_MAX_REGISTERS = 26
LOCC_SEND_SHOT_CAP = 100
LOCC_SEND_QUBIT_THRESHOLD = 20


class LOCCEngine:
    """N-register quantum simulation with classical channels.

    Supports 2-26 independent quantum registers (A through Z).
    SPLIT mode: independent statevectors per register. Max capacity.
    JOINT mode: one statevector, LOCC constraints enforced.
    """

    def __init__(self, sizes: list[int], joint: bool = False,
                 noise_param: float = 0.0, noise_type: str = 'depolarizing',
                 precision: str = 'double'):
        """Initialize LOCC engine with given register sizes.

        noise_param: per-gate noise strength (0 = noiseless).
        noise_type: 'depolarizing', 'amplitude_damping', or 'phase_flip'.
        Noise is applied as a Monte-Carlo trajectory after each gate, matching
        the Qiskit Aer channel definitions (depolarizing uses the standard
        1 - 3p/4 identity weighting).
        precision: 'double' (complex128) or 'single' (complex64) amplitudes.
        States are updated in place, so memory is one copy of each register's
        state plus a bounded scratch block.
        """
        self.sizes = list(sizes)
        self.n_regs = len(self.sizes)
        self.names = [chr(ord('A') + i) for i in range(self.n_regs)]
        self.joint = joint
        self.noise_param = noise_param
        self.noise_type = noise_type
        self.precision = precision
        self.dtype = np.complex64 if precision == 'single' else np.complex128
        self.classical: dict[str, int] = {}
        # Precompute offsets for JOINT mode
        self.offsets = []
        off = 0
        for s in self.sizes:
            self.offsets.append(off)
            off += s
        self.n_total = sum(self.sizes)
        self.reset()

    # Backward-compatible properties
    @property
    def n_a(self):
        return self.sizes[0] if self.sizes else 0

    @property
    def n_b(self):
        return self.sizes[1] if len(self.sizes) > 1 else 0

    def _idx(self, reg):
        """Register name -> index."""
        return ord(reg) - ord('A')

    def reset(self) -> None:
        """Reset all registers to |0> and clear classical state."""
        self.classical.clear()
        self.correction_log: list[str] = []
        if self.joint:
            self.sv = np.zeros(2**self.n_total, dtype=self.dtype)
            self.sv[0] = 1.0
        else:
            self.svs = {}
            for name, size in zip(self.names, self.sizes):
                sv = np.zeros(2**size, dtype=self.dtype)
                sv[0] = 1.0
                self.svs[name] = sv

    def snapshot(self) -> dict:
        """Capture current quantum state for later restore."""
        if self.joint:
            return {'joint': True, 'sv': self.sv.copy(), 'classical': dict(self.classical)}
        return {'joint': False,
                'svs': {n: sv.copy() for n, sv in self.svs.items()},
                'classical': dict(self.classical)}

    def restore(self, snap: dict) -> None:
        """Restore quantum state from a snapshot, reusing the live buffers."""
        self.classical = dict(snap['classical'])
        if snap['joint']:
            if getattr(self, 'sv', None) is not None and self.sv.shape == snap['sv'].shape \
                    and self.sv.flags.c_contiguous:
                np.copyto(self.sv, snap['sv'])
            else:
                self.sv = snap['sv'].copy()
        else:
            for n, sv in snap['svs'].items():
                cur = self.svs.get(n)
                if cur is not None and cur.shape == sv.shape and cur.flags.c_contiguous:
                    np.copyto(cur, sv)
                else:
                    self.svs[n] = sv.copy()

    def _check_qubits(self, reg: str, qubits: list[int]) -> None:
        """Validate that all qubit indices are in range for the given register."""
        size = self.sizes[self._idx(reg)]
        for q in qubits:
            if q < 0 or q >= size:
                raise ValueError(
                    f"Qubit {q} out of range for register {reg} (size {size})"
                )

    def _target(self, reg: str, qubits: list[int]) -> tuple[np.ndarray, list[int], int]:
        """(state array, global qubit indices, total qubits) for reg's qubits."""
        if self.joint:
            off = self.offsets[self._idx(reg)]
            return self._own(self.sv, joint=True), [q + off for q in qubits], self.n_total
        return self._own(self.svs[reg], reg=reg), list(qubits), self.sizes[self._idx(reg)]

    def _own(self, sv: np.ndarray, joint: bool = False, reg: str | None = None) -> np.ndarray:
        """Return a C-contiguous flat state of the engine dtype, storing it back
        when a caller replaced the array with a view or another dtype."""
        if sv.flags.c_contiguous and sv.dtype == self.dtype and sv.ndim == 1:
            return sv
        fixed = np.ascontiguousarray(sv, dtype=self.dtype).reshape(-1)
        if joint:
            self.sv = fixed
        elif reg is not None:
            self.svs[reg] = fixed
        return fixed

    def _apply_single(self, reg: str, matrix: np.ndarray, q: int) -> None:
        """Apply a single-qubit matrix (unitary or Kraus operator) to reg[q]."""
        sv, qs, n = self._target(reg, [q])
        _apply_gate_inplace(sv, matrix, qs, n)

    def _marginal_p1(self, reg: str, q: int) -> float:
        """Probability that reg[q] is measured |1> (non-destructive)."""
        sv, qs, n = self._target(reg, [q])
        return _marginal_p1(sv, qs[0], n)

    def _renormalize(self, reg: str) -> None:
        sv, _, _ = self._target(reg, [])
        nrm = float(np.sqrt(np.vdot(sv, sv).real))
        if nrm > 1e-15:
            sv *= 1.0 / nrm

    def _apply_noise(self, reg: str, qubits: list[int]) -> None:
        """Apply the configured noise channel to each target qubit (Monte-Carlo).

        Supported channels: depolarizing, amplitude_damping, phase_flip (phase
        damping). Conventions match Qiskit Aer so a given parameter behaves the
        same in the LOCC numpy path and the standard Aer path.
        """
        p = self.noise_param
        if p <= 0 or self.noise_type in ('none', None):
            return
        nt = self.noise_type
        for q in qubits:
            if nt == 'depolarizing':
                # Identity w.p. 1 - 3p/4; X/Y/Z each w.p. p/4 (Aer convention).
                if np.random.random() < 0.75 * p:
                    pauli = _np_gate_matrix('XYZ'[np.random.randint(3)], ())
                    self._apply_single(reg, pauli, q)
            elif nt == 'amplitude_damping':
                g = p
                if np.random.random() < g * self._marginal_p1(reg, q):
                    self._apply_single(reg, np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex), q)
                else:
                    self._apply_single(reg, np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex), q)
                self._renormalize(reg)
            elif nt in ('phase_flip', 'phase_damping'):
                g = p
                if np.random.random() < g * self._marginal_p1(reg, q):
                    self._apply_single(reg, np.array([[0, 0], [0, np.sqrt(g)]], dtype=complex), q)
                else:
                    self._apply_single(reg, np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex), q)
                self._renormalize(reg)

    def apply(self, reg: str, gate_name: str, params: tuple[float, ...], qubits: list[int]) -> None:
        """Apply a gate to a register, then apply noise if configured.

        Qubit indices are register-LOCAL (0..size-1); the register offset into
        the joint statevector is added internally. Callers that index the raw
        statevector (e.g. for fidelity/inspection) must convert with
        ``offsets[reg] + local`` themselves.
        """
        self._check_qubits(reg, qubits)
        matrix = _np_gate_matrix(gate_name, tuple(params))
        sv, qs, n = self._target(reg, qubits)
        _apply_gate_inplace(sv, matrix, qs, n)
        self._apply_noise(reg, qubits)

    def share(self, reg1: str, q1: int, reg2: str, q2: int) -> None:
        """Create Bell pair |Phi+> between reg1[q1] and reg2[q2]. JOINT only.

        ``q1``/``q2`` are register-LOCAL indices; offsets are added internally.
        """
        if not self.joint:
            raise RuntimeError("SHARE requires LOCC JOINT mode")
        self._check_qubits(reg1, [q1])
        self._check_qubits(reg2, [q2])
        h = _np_gate_matrix('H', ())
        cx = _np_gate_matrix('CX', ())
        actual1 = q1 + self.offsets[self._idx(reg1)]
        actual2 = q2 + self.offsets[self._idx(reg2)]
        sv = self._own(self.sv, joint=True)
        _apply_gate_inplace(sv, h, [actual1], self.n_total)
        self._apply_noise(reg1, [q1])
        _apply_gate_inplace(sv, cx, [actual1, actual2], self.n_total)
        self._apply_noise(reg1, [q1])
        self._apply_noise(reg2, [q2])

    def send(self, reg: str, qubit: int) -> int:
        """Measure a qubit (Born rule) and return the classical outcome.

        ``qubit`` is register-LOCAL (0..size-1); the offset is added internally.
        """
        self._check_qubits(reg, [qubit])
        sv, qs, n = self._target(reg, [qubit])
        return _measure_inplace(sv, qs[0], n)

    def get_sv(self, reg: str) -> np.ndarray:
        if self.joint:
            return self.sv
        return self.svs[reg]

    def get_n(self, reg: str) -> int:
        if self.joint:
            return self.n_total
        return self.sizes[self._idx(reg)]

    def get_size(self, reg: str) -> int:
        return self.sizes[self._idx(reg)]

    def sample_joint(self, shots: int) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
        """Sample and return (per_reg_counts, joint_counts)."""
        if self.joint:
            raw = _sample_counts(self._own(self.sv, joint=True), self.n_total, shots)
            per_reg: dict[str, dict[str, int]] = {name: {} for name in self.names}
            joint: dict[str, int] = {}
            for state, count in raw.items():
                # Split the joint bitstring into per-register segments.
                # The joint statevector is ordered A[0..nA-1] B[0..nB-1] ...
                # with register A occupying the most-significant qubits.
                # The bitstring is MSB-first, so register A's bits are at
                # the left end. We walk backward from the right to peel
                # off each register's segment from least-significant first.
                parts = []
                pos = len(state)
                for i in range(self.n_regs):
                    size = self.sizes[i]
                    segment = state[pos - size:pos] if pos >= size else '0' * size
                    parts.append(segment)
                    pos -= size
                for name, part in zip(self.names, parts):
                    per_reg[name][part] = per_reg[name].get(part, 0) + count
                jkey = '|'.join(parts)
                joint[jkey] = joint.get(jkey, 0) + count
            return per_reg, joint
        else:
            per_reg = {}
            for name in self.names:
                size = self.sizes[self._idx(name)]
                per_reg[name] = _sample_counts(self._own(self.svs[name], reg=name), size, shots)
            return per_reg, {}

    def apply_matrix(self, reg: str, matrix: np.ndarray, qubits: list[int]) -> None:
        """Apply a raw unitary matrix to qubits in a register.

        ``qubits`` are register-LOCAL indices; offsets are added internally.
        The matrix is MSB-first over ``qubits`` (qubits[0] is the most
        significant bit of its index), the convention of the built-in gates.
        """
        self._check_qubits(reg, qubits)
        sv, qs, n = self._target(reg, qubits)
        _apply_gate_inplace(sv, matrix, qs, n)
        self._apply_noise(reg, qubits)

    def mem_gb(self, snapshot: bool = False) -> tuple[float, float]:  # (total_gb, peak_gb)
        """Return (total_gb, peak_gb): memory for every register's state and for
        the largest one, including the in-place scratch block. `snapshot`
        adds the second copy a SEND-based run holds for its prefix restore."""
        from qubasic_core.capacity import numpy_peak_bytes
        if self.joint:
            total = numpy_peak_bytes(self.n_total, self.precision, snapshot) / 1e9
            return total, total
        per = [numpy_peak_bytes(s, self.precision, snapshot) / 1e9 for s in self.sizes]
        return sum(per), max(per)
