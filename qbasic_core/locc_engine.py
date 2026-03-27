"""LOCC quantum engine — multi-register simulation with classical channels."""

from __future__ import annotations

import numpy as np
from qbasic_core.gates import (
    _np_gate_matrix, _apply_gate_np, _measure_np, _sample_np,
)

# ═══════════════════════════════════════════════════════════════════════
# LOCC-specific constants
# ═══════════════════════════════════════════════════════════════════════

LOCC_MAX_JOINT_QUBITS = 33
LOCC_MAX_SPLIT_QUBITS = 33
LOCC_MAX_REGISTERS = 26
LOCC_SEND_SHOT_CAP = 100
LOCC_SEND_QUBIT_THRESHOLD = 20

# Realistic multiplier for statevector simulation overhead.
OVERHEAD_FACTOR = 3.0


class LOCCEngine:
    """N-register quantum simulation with classical channels.

    Supports 2-26 independent quantum registers (A through Z).
    SPLIT mode: independent statevectors per register. Max capacity.
    JOINT mode: one statevector, LOCC constraints enforced.
    """

    def __init__(self, sizes: list[int], joint: bool = False):
        """Initialize LOCC engine with given register sizes."""
        self.sizes = list(sizes)
        self.n_regs = len(self.sizes)
        self.names = [chr(ord('A') + i) for i in range(self.n_regs)]
        self.joint = joint
        self.classical = {}
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
        if self.joint:
            self.sv = np.zeros(2**self.n_total, dtype=complex)
            self.sv[0] = 1.0
        else:
            self.svs = {}
            for name, size in zip(self.names, self.sizes):
                sv = np.zeros(2**size, dtype=complex)
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
        """Restore quantum state from a snapshot."""
        self.classical = dict(snap['classical'])
        if snap['joint']:
            self.sv = snap['sv'].copy()
        else:
            self.svs = {n: sv.copy() for n, sv in snap['svs'].items()}

    def _check_qubits(self, reg: str, qubits: list[int]) -> None:
        """Validate that all qubit indices are in range for the given register."""
        size = self.sizes[self._idx(reg)]
        for q in qubits:
            if q < 0 or q >= size:
                raise ValueError(
                    f"Qubit {q} out of range for register {reg} (size {size})"
                )

    def apply(self, reg: str, gate_name: str, params: tuple[float, ...], qubits: list[int]) -> None:
        """Apply a gate to a specific register."""
        self._check_qubits(reg, qubits)
        matrix = _np_gate_matrix(gate_name, tuple(params))
        if self.joint:
            idx = self._idx(reg)
            actual = [q + self.offsets[idx] for q in qubits]
            self.sv = _apply_gate_np(self.sv, matrix, actual, self.n_total)
        else:
            size = self.sizes[self._idx(reg)]
            self.svs[reg] = _apply_gate_np(self.svs[reg], matrix, qubits, size)

    def share(self, reg1: str, q1: int, reg2: str, q2: int) -> None:
        """Create Bell pair |Phi+> between reg1[q1] and reg2[q2]. JOINT only."""
        if not self.joint:
            raise RuntimeError("SHARE requires LOCC JOINT mode")
        self._check_qubits(reg1, [q1])
        self._check_qubits(reg2, [q2])
        h = _np_gate_matrix('H', ())
        cx = _np_gate_matrix('CX', ())
        actual1 = q1 + self.offsets[self._idx(reg1)]
        actual2 = q2 + self.offsets[self._idx(reg2)]
        self.sv = _apply_gate_np(self.sv, h, [actual1], self.n_total)
        self.sv = _apply_gate_np(self.sv, cx, [actual1, actual2], self.n_total)

    def send(self, reg: str, qubit: int) -> int:
        """Measure a qubit (Born rule) and return the classical outcome."""
        self._check_qubits(reg, [qubit])
        if self.joint:
            actual = qubit + self.offsets[self._idx(reg)]
            outcome, self.sv = _measure_np(self.sv, actual, self.n_total)
        else:
            size = self.sizes[self._idx(reg)]
            outcome, self.svs[reg] = _measure_np(self.svs[reg], qubit, size)
        return outcome

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
            raw = _sample_np(self.sv, self.n_total, shots)
            per_reg = {name: {} for name in self.names}
            joint = {}
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
                    parts.append(state[pos - size:pos])
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
                per_reg[name] = _sample_np(self.svs[name], size, shots)
            return per_reg, {}

    def apply_matrix(self, reg: str, matrix: np.ndarray, qubits: list[int]) -> None:
        """Apply a raw unitary matrix to qubits in a register."""
        self._check_qubits(reg, qubits)
        if self.joint:
            idx = self._idx(reg)
            actual = [q + self.offsets[idx] for q in qubits]
            self.sv = _apply_gate_np(self.sv, matrix, actual, self.n_total)
        else:
            size = self.sizes[self._idx(reg)]
            self.svs[reg] = _apply_gate_np(self.svs[reg], matrix, qubits, size)

    def mem_gb(self) -> tuple[float, float]:  # (total_gb, peak_gb)
        """Return (total_gb, peak_gb) realistic memory estimates including overhead."""
        if self.joint:
            total = (2**self.n_total) * 16 * OVERHEAD_FACTOR / 1e9
            return total, total
        total = sum((2**s) * 16 * OVERHEAD_FACTOR / 1e9 for s in self.sizes)
        peak = max((2**s) * 16 * OVERHEAD_FACTOR / 1e9 for s in self.sizes)
        return total, peak
