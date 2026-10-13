"""QUBASIC LOCC mode: the multi-register engine with classical
channels, its commands, displays and execution.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import MutableMapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from qubasic_core.engine import RAM_BUDGET_FRACTION, ExecResult
from qubasic_core.host import SimulationHost
from qubasic_core.parser import GATE_ALIASES, RE_CTRL, RE_INV
from qubasic_core.simtools import (
    _apply_gate_inplace,
    _marginal_p1,
    _measure_inplace,
    _np_gate_matrix,
    _sample_counts,
    _sample_one_np,
    ram_gb,
)

if TYPE_CHECKING:
    from qubasic_core.engine import ExecOutcome
    from qubasic_core.parser import CompiledProgram, Stmt


# ---------------------------------------------------------------------------
# LOCC quantum engine — multi-register simulation with classical channels.
# ---------------------------------------------------------------------------

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
                 precision: str = 'double') -> None:
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
    def n_a(self) -> int:
        return self.sizes[0] if self.sizes else 0

    @property
    def n_b(self) -> int:
        return self.sizes[1] if len(self.sizes) > 1 else 0

    def _idx(self, reg: str) -> int:
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
        from qubasic_core.simtools import numpy_peak_bytes
        if self.joint:
            total = numpy_peak_bytes(self.n_total, self.precision, snapshot) / 1e9
            return total, total
        per = [numpy_peak_bytes(s, self.precision, snapshot) / 1e9 for s in self.sizes]
        return sum(per), max(per)


# ---------------------------------------------------------------------------
# QUBASIC LOCC commands mixin — cmd_locc, cmd_send, cmd_share, cmd_loccinfo.
# ---------------------------------------------------------------------------

class LOCCCommandsMixin(SimulationHost):
    """LOCC user-facing commands for QBasicTerminal."""

    @property
    def _locc_engine(self) -> LOCCEngine:
        """The LOCC engine; an error when LOCC mode is off."""
        if self.locc is None:
            raise RuntimeError("LOCC mode is off (try: LOCC <n1> <n2>)")
        return self.locc

    def _locc_noise_param(self) -> float:
        """Return the scalar depolarizing probability for LOCC noise.

        Stored by cmd_noise when a depolarizing model is set.
        """
        return self.noise.depol_p

    def cmd_locc(self, rest: str) -> None:
        """LOCC [JOINT|SPLIT] <n1> <n2> [...] | OFF | STATUS | SHOTCAP <n> — registers that share classical bits (SPLIT) or entanglement (JOINT).

        SPLIT gives independent registers (coordinate through SEND and IF),
        JOINT one joint state in which SHARE creates entanglement. SHOTCAP
        bounds shots for SEND-based protocols on large registers."""
        args = rest.upper().split()
        if not args:
            if self.locc_mode:
                mode = "JOINT" if self._locc_engine.joint else "SPLIT"
                reg_desc = '  '.join(f"{n}={s}q" for n, s in
                                     zip(self._locc_engine.names, self._locc_engine.sizes))
                self.io.writeln(f"LOCC {mode}: {reg_desc}")
                tot, peak = self._locc_engine.mem_gb()
                self.io.writeln(f"  RAM per instance: {tot:.1f} GB (with overhead)")
                if self._locc_engine.classical:
                    self.io.writeln(f"  Classical: {self._locc_engine.classical}")
            else:
                self.io.writeln("LOCC OFF. Usage: LOCC [JOINT] <n1> <n2> [n3 ...]")
            return

        if args[0] == 'OFF':
            self.locc = None
            self.locc_mode = False
            self.io.writeln("LOCC OFF — back to normal Aer mode")
            return

        if args[0] == 'SHOTCAP':
            # LOCC SHOTCAP <n> | OFF — cap shots for SEND-based protocols on
            # large registers.
            if len(args) < 2:
                cur = getattr(self, '_locc_shot_cap', None)
                self.io.writeln(f"LOCC SHOTCAP = {cur if cur else 'auto'}")
                return
            if args[1] == 'OFF':
                self._locc_shot_cap = None
                self.io.writeln("LOCC SHOTCAP OFF (auto)")
                return
            try:
                self._locc_shot_cap = max(1, int(args[1]))
                self.io.writeln(f"LOCC SHOTCAP = {self._locc_shot_cap}")
            except ValueError:
                self.io.writeln("?USAGE: LOCC SHOTCAP <n> | OFF")
            return

        if args[0] == 'STATUS':
            if not self.locc_mode:
                self.io.writeln("NOT IN LOCC MODE")
                return
            mode = "JOINT" if self._locc_engine.joint else "SPLIT"
            reg_desc = '  '.join(f"{n}={s}q" for n, s in
                                 zip(self._locc_engine.names, self._locc_engine.sizes))
            self.io.writeln(f"LOCC {mode}: {reg_desc}")
            tot, peak = self._locc_engine.mem_gb()
            self.io.writeln(f"  RAM per instance: {tot:.1f} GB (with overhead)")
            ram = ram_gb()
            if ram:
                budget = ram[0] * RAM_BUDGET_FRACTION
                max_par = int(budget / tot) if tot > 0 else 0
                if max_par > 0:
                    self.io.writeln(f"  Max parallel in 80% budget: ~{max_par}")
            self.io.writeln(f"  Classical vars: {self._locc_engine.classical if self._locc_engine.classical else '(none)'}")
            return

        joint = False
        nums = args
        if args[0] == 'JOINT':
            joint = True
            nums = args[1:]
        elif args[0] == 'SPLIT':
            nums = args[1:]

        # Support "2+2+2" notation: split on + if present
        expanded = []
        for n in nums:
            expanded.extend(n.split('+'))
        nums = [n for n in expanded if n]

        if len(nums) < 2:
            self.io.writeln("?USAGE: LOCC [JOINT|SPLIT] <n1> <n2> [n3 ...]")
            return
        if len(nums) > LOCC_MAX_REGISTERS:
            self.io.writeln(f"?MAX {LOCC_MAX_REGISTERS} registers (A-Z)")
            return

        try:
            sizes = [int(n) for n in nums]
        except ValueError:
            self.io.writeln("?LOCC register sizes must be integers")
            return
        total = sum(sizes)
        if joint and total > LOCC_MAX_JOINT_QUBITS:
            self.io.writeln(f"?JOINT mode limited to {LOCC_MAX_JOINT_QUBITS} total qubits (requested {total})")
            return
        if not joint and max(sizes) > LOCC_MAX_SPLIT_QUBITS:
            self.io.writeln(f"?Each register limited to {LOCC_MAX_SPLIT_QUBITS} qubits")
            return

        # Pre-check RAM before allocating
        mode = "JOINT" if joint else "SPLIT"
        noise_p = self.noise.locc_param
        noise_t = self.noise.locc_type
        temp_eng = LOCCEngine(sizes, joint=joint, noise_param=noise_p, noise_type=noise_t)
        tot, peak = temp_eng.mem_gb()
        ram = ram_gb()
        if ram and tot > ram[1]:
            self.io.writeln(f"?BLOCKED: LOCC {mode} needs ~{tot:.1f} GB but only "
                           f"{ram[1]:.1f} GB available. Reduce register sizes.")
            return

        self.locc = temp_eng
        self.locc_mode = True
        reg_desc = '  '.join(f"{n}={s}q" for n, s in
                             zip(self._locc_engine.names, sizes))
        self.io.writeln(f"LOCC {mode}: {reg_desc}  ({total} total)")
        self.io.writeln(f"  RAM per instance: {tot:.1f} GB (with overhead)")
        if ram:
            sys_total, avail = ram
            budget = sys_total * RAM_BUDGET_FRACTION
            if tot > 0:
                max_par = int(budget / tot)
                if max_par > 0:
                    self.io.writeln(f"  Max parallel instances in 80% budget: ~{max_par}")
        if not joint:
            self.io.writeln("  Registers are INDEPENDENT — no cross-register entanglement")
            self.io.writeln("  Use SEND/IF for classical coordination")
        else:
            self.io.writeln("  Joint statevector — use SHARE for pre-shared entanglement")
        if peak > 30:
            self.io.writeln("  WARNING: large registers. Keep SHOTS low for SEND-based protocols.")
        if self.noise.model and noise_p == 0:
            self.io.writeln("  WARNING: the active noise model has no LOCC numpy form and is "
                            "ignored here; LOCC runs one depolarizing, amplitude_damping or "
                            "phase_flip channel.")

    def cmd_send(self, rest: str) -> None:
        """SEND <reg> <qubit> -> <var> — measure a LOCC register qubit into a classical variable."""
        if not self.locc_mode:
            self.io.writeln("?SEND requires LOCC mode")
            return
        m = re.match(r'([A-Z])\s+(\S+)\s*->\s*(\w+)', rest, re.IGNORECASE)
        if not m:
            self.io.writeln("?USAGE: SEND <reg> <qubit> -> <var>")
            return
        reg = m.group(1).upper()
        if reg not in self._locc_engine.names:
            self.io.writeln(f"?UNKNOWN REGISTER: {reg} (have {', '.join(self._locc_engine.names)})")
            return
        qubit = int(self.eval_expr(m.group(2)))
        var = m.group(3)
        n_reg = self._locc_engine.get_size(reg)
        if qubit < 0 or qubit >= n_reg:
            self.io.writeln(f"?QUBIT {qubit} OUT OF RANGE for register {reg} (0-{n_reg-1})")
            return
        outcome = self._locc_engine.send(reg, qubit)
        self.variables[var] = outcome
        self._locc_engine.classical[var] = outcome
        self.io.writeln(f"  {reg}[{qubit}] -> {var} = {outcome}")

    def cmd_share(self, rest: str) -> None:
        """SHARE <reg1> <q1>, <reg2> <q2> — create a Bell pair across two LOCC registers (JOINT mode)."""
        if not self.locc_mode:
            self.io.writeln("?SHARE requires LOCC mode")
            return
        m = re.match(r'([A-Z])\s+(\d+)\s*,?\s*([A-Z])\s+(\d+)', rest, re.IGNORECASE)
        if not m:
            self.io.writeln("?USAGE: SHARE <reg1> <qubit> <reg2> <qubit>")
            return
        reg1, q1 = m.group(1).upper(), int(m.group(2))
        reg2, q2 = m.group(3).upper(), int(m.group(4))
        for r in (reg1, reg2):
            if r not in self._locc_engine.names:
                self.io.writeln(f"?UNKNOWN REGISTER: {r}")
                return
        try:
            self._locc_engine.share(reg1, q1, reg2, q2)
            self.io.writeln(f"  Bell pair |Phi+> created: {reg1}[{q1}] <-> {reg2}[{q2}]")
        except RuntimeError as e:
            self.io.writeln(f"?{e}")

    def cmd_loccinfo(self) -> None:
        """LOCCINFO — protocol metrics of the last LOCC run."""
        if not self.locc_mode:
            self.io.writeln("?NOT IN LOCC MODE")
            return
        mode = "JOINT" if self._locc_engine.joint else "SPLIT"
        self.io.writeln(f"\n  LOCC Protocol Metrics ({mode})")
        reg_desc = '  '.join(f"{n}={s}q" for n, s in
                             zip(self._locc_engine.names, self._locc_engine.sizes))
        self.io.writeln(f"  Registers: {reg_desc}  ({self._locc_engine.n_regs} parties)")

        # Entanglement creation
        n_shares = sum(1 for l in self.program.values()
                       if re.search(r'\bSHARE\b', l, re.IGNORECASE))
        if n_shares:
            self.io.writeln(f"  Entanglement: {n_shares} SHARE operation(s)")

        # Classical communication
        n_classical = len(self._locc_engine.classical)
        self.io.writeln(f"  Classical bits exchanged: {n_classical}")
        if self._locc_engine.classical:
            for k, v in self._locc_engine.classical.items():
                self.io.writeln(f"    {k} = {v}")

        # Measurement outcomes and correction paths
        n_sends = sum(1 for l in self.program.values()
                      if re.search(r'\bSEND\b', l, re.IGNORECASE))
        n_ifs = sum(1 for l in self.program.values()
                    if re.search(r'\bIF\b.*\bTHEN\b.*@', l, re.IGNORECASE))
        self.io.writeln(f"  SEND operations: {n_sends}")
        self.io.writeln(f"  Conditional corrections: {n_ifs}")
        self.io.writeln(f"  Communication rounds: ~{n_sends}")

        # Branch statistics: totals over every SEND across all shots of the
        # last run (the classical dict only holds each bit's final value).
        if n_sends > 0:
            _log = getattr(self.locc, 'correction_log', [])
            n0 = sum(1 for e in _log if str(e).rstrip().endswith('=0'))
            n1 = sum(1 for e in _log if str(e).rstrip().endswith('=1'))
            if n0 or n1:
                self.io.writeln(f"  Branch stats: {n0} zeros, {n1} ones "
                               f"(all shots, last run)")

        # Correction log (from last run)
        log = getattr(self.locc, 'correction_log', [])
        if log:
            self.io.writeln(f"  Correction log ({len(log)} entries, last run):")
            for entry in log[-20:]:  # cap display
                self.io.writeln(f"    {entry}")
            if len(log) > 20:
                self.io.writeln(f"    ... ({len(log) - 20} more)")

        # Noise
        noise_p = getattr(self.locc, 'noise_param', 0.0)
        if noise_p > 0:
            self.io.writeln(f"  Noise: depolarizing p={noise_p}")

        tot, peak = self._locc_engine.mem_gb()
        self.io.writeln(f"  Memory: {tot:.1f} GB")
        self.io.writeln('')


# ---------------------------------------------------------------------------
# QUBASIC LOCC execution mixin — program execution in LOCC mode.
# ---------------------------------------------------------------------------

class LOCCExecutionMixin(SimulationHost):
    """LOCC execution methods for QBasicTerminal."""

    def _locc_run(self) -> None:
        """Execute program in LOCC mode (N registers)."""
        sorted_lines = sorted(self.program.keys())
        if not sorted_lines:
            self.io.writeln("NOTHING TO RUN")
            return
        prog = self._compiled()
        self._scan_subs(prog)
        self._reset_declarations(prog)
        self._gosub_stack.clear()
        has_measure = self._program_has_measure(sorted_lines)
        has_send = any(re.search(r'\bSEND\b', text, re.IGNORECASE) for text in prog.texts)
        if has_send:
            self._locc_run_with_send(prog, has_measure)
        else:
            self._locc_run_vectorized(prog, has_measure)
        # Sync last_sv from LOCC engine so EXPECT/DENSITY/BLOCH work
        self.last_sv = self._active_sv

    def _locc_run_with_send(self, sorted_lines: CompiledProgram, has_measure: bool) -> None:
        """LOCC execution with SEND — prefix/suffix split optimization.

        Executes the deterministic prefix (before first SEND) once,
        snapshots the quantum state, then re-executes only the suffix
        (from first SEND onward) per shot. Falls back to full re-execution
        if the prefix contains jumps that could skip past the SEND.
        """
        from qubasic_core.expression import Scope
        from qubasic_core.parser import (
            CallStmt,
            DoStmt,
            ForStmt,
            GosubStmt,
            GotoStmt,
            IfJumpStmt,
            JumpStmt,
            OnGosubStmt,
            OnGotoStmt,
            SendStmt,
            WhileStmt,
        )
        prog = sorted_lines

        # A loop, call or jump before the first SEND could skip it or come
        # back to it, so the prefix runs once only when every transfer in it
        # stays before the SEND (an IF that jumps past it is unsafe too).
        _jump_types = (GotoStmt, GosubStmt, OnGotoStmt, OnGosubStmt,
                       ForStmt, WhileStmt, DoStmt, CallStmt)
        first_send_ip = next((i for i, p in enumerate(prog.stmts)
                              if isinstance(p, SendStmt)), None)
        has_prefix_jumps = False
        limit = first_send_ip or 0
        for i in range(limit):
            p = prog.stmts[i]
            if isinstance(p, _jump_types) or (
                    isinstance(p, IfJumpStmt) and max(p.on_false, p.end) > limit) or (
                    isinstance(p, JumpStmt) and p.target > limit):
                has_prefix_jumps = True
                break

        sizes_str = '+'.join(str(s) for s in self._locc_engine.sizes)
        mode = "JOINT" if self._locc_engine.joint else "SPLIT"
        shots = self.shots
        max_q = max(self._locc_engine.sizes)
        # Cap overridable via the LOCC SHOTCAP command.
        user_cap = getattr(self, '_locc_shot_cap', None)
        effective_cap = user_cap if user_cap else LOCC_SEND_SHOT_CAP
        if max_q > LOCC_SEND_QUBIT_THRESHOLD and shots > effective_cap and not user_cap:
            self.io.writeln(f"  WARNING: capping at {effective_cap} shots for "
                            f"{max_q}-qubit LOCC w/ SEND (override: LOCC SHOTCAP <n>)")
            shots = effective_cap

        # Execute deterministic prefix once (skip if jumps make it unsafe, or
        # when noise is active: each shot needs its own noise draws)
        self._locc_engine.reset()
        noisy = getattr(self.locc, 'noise_param', 0.0) > 0
        if has_prefix_jumps or noisy or first_send_ip == 0 or first_send_ip is None:
            send_ip = 0  # no safe prefix — re-exec from start
        else:
            prefix_vars = dict(self.variables)
            try:
                send_ip = self._locc_execute_program(sorted_lines, stop_before_send=True,
                                                      run_vars=Scope(prefix_vars))
            except (RuntimeError, ValueError) as e:
                self.io.writeln(f"?RUNTIME ERROR: {e}")
                return

        # Snapshot state after prefix
        snap = self._locc_engine.snapshot()
        snap_vars = dict(self.variables)

        per_reg: dict[str, dict[str, int]] = {name: {} for name in self._locc_engine.names}
        counts_joint: dict[str, int] = {}
        t0 = time.time()
        progress_interval = max(1, shots // 10)
        # Only the first shot prints; later shots replay its INPUT answers.
        from qubasic_core.simulator import ShotIO
        real_io = self.io
        shot_io = ShotIO(real_io)
        first_vars: dict = {}

        for shot in range(shots):
            # Restore to post-prefix state
            self._locc_engine.restore(snap)
            self.variables.clear()
            self.variables.update(snap_vars)
            if shot:
                shot_io.start_replay()
            self.io = shot_io
            try:
                self._locc_execute_program(sorted_lines, start_ip=send_ip,
                                           run_vars=Scope(self.variables))
            except (RuntimeError, ValueError) as e:
                self.io = real_io
                self.io.writeln(f"?RUNTIME ERROR: {e}")
                return
            finally:
                self.io = real_io
            if shot == 0:
                first_vars = dict(self.variables)
            if has_measure:
                if self._locc_engine.joint:
                    out = _sample_one_np(self._locc_engine.sv, self._locc_engine.n_total)
                    parts = []
                    pos = len(out)
                    for i in range(self._locc_engine.n_regs):
                        size = self._locc_engine.sizes[i]
                        parts.append(out[pos - size:pos])
                        pos -= size
                else:
                    parts = [_sample_one_np(self._locc_engine.svs[name],
                             self._locc_engine.get_size(name))
                             for name in self._locc_engine.names]
                for name, bits in zip(self._locc_engine.names, parts):
                    per_reg[name][bits] = per_reg[name].get(bits, 0) + 1
                jkey = '|'.join(parts)
                counts_joint[jkey] = counts_joint.get(jkey, 0) + 1
            if shots > 50 and (shot + 1) % progress_interval == 0:
                pct = 100 * (shot + 1) // shots
                from qubasic_core.engine import quantum_spin
                spin = quantum_spin(shot)
                _prog = f"  {spin} {pct}% ({shot+1}/{shots} shots)..."
                if hasattr(self.io, 'write') and sys.stdout.isatty():
                    self.io.write(_prog + '\r')
                # Non-terminal: skip progress to avoid \r noise

        if shots > 50 and sys.stdout.isatty():
            self.io.write(" " * 40 + '\r')
        # Variables read as the first (printed) shot left them.
        self.variables.clear()
        self.variables.update(first_vars)
        dt = time.time() - t0
        self.io.writeln(f"\nRAN {len(self.program)} lines, LOCC {mode} "
                        f"{sizes_str}q, {shots} shots in {dt:.2f}s")
        if has_measure:
            self._locc_display_results(per_reg, counts_joint)
            self.last_counts = counts_joint

    def _locc_run_vectorized(self, sorted_lines: CompiledProgram, has_measure: bool) -> None:
        """LOCC execution without SEND — single execution, vectorized sampling.

        When noise is active, re-executes per shot so that Monte Carlo noise
        fires independently each time (matching the SEND path behavior).
        """
        sizes_str = '+'.join(str(s) for s in self._locc_engine.sizes)
        mode = "JOINT" if self._locc_engine.joint else "SPLIT"
        t0 = time.time()

        noisy = getattr(self.locc, 'noise_param', 0.0) > 0

        if noisy and has_measure:
            # Per-shot execution so noise fires independently each time; only
            # the first shot prints.
            from qubasic_core.simulator import ShotIO
            per_reg: dict[str, dict[str, int]] = {name: {} for name in self._locc_engine.names}
            counts_joint: dict[str, int] = {}
            real_io = self.io
            shot_io = ShotIO(real_io)
            base_vars = dict(self.variables)
            first_vars: dict = {}
            for _shot in range(self.shots):
                self._locc_engine.reset()
                self.variables.clear()
                self.variables.update(base_vars)
                if _shot:
                    shot_io.start_replay()
                self.io = shot_io
                try:
                    self._locc_execute_program(sorted_lines)
                except (RuntimeError, ValueError) as e:
                    self.io = real_io
                    self.io.writeln(f"?RUNTIME ERROR: {e}")
                    return
                finally:
                    self.io = real_io
                if _shot == 0:
                    first_vars = dict(self.variables)
                if self._locc_engine.joint:
                    out = _sample_one_np(self._locc_engine.sv, self._locc_engine.n_total)
                    parts = []
                    pos = len(out)
                    for i in range(self._locc_engine.n_regs):
                        size = self._locc_engine.sizes[i]
                        parts.append(out[pos - size:pos])
                        pos -= size
                else:
                    parts = [_sample_one_np(self._locc_engine.svs[name],
                             self._locc_engine.get_size(name))
                             for name in self._locc_engine.names]
                for name, bits in zip(self._locc_engine.names, parts):
                    per_reg[name][bits] = per_reg[name].get(bits, 0) + 1
                jkey = '|'.join(parts)
                counts_joint[jkey] = counts_joint.get(jkey, 0) + 1
            self.variables.clear()
            self.variables.update(first_vars)
            dt = time.time() - t0
            self.io.writeln(f"\nRAN {len(self.program)} lines, LOCC {mode} "
                            f"{sizes_str}q, {self.shots} shots in {dt:.2f}s")
            self._locc_display_results(per_reg, counts_joint)
            self.last_counts = counts_joint if counts_joint else per_reg.get('A', {})
            return

        self._locc_engine.reset()
        try:
            self._locc_execute_program(sorted_lines)
        except (RuntimeError, ValueError) as e:
            self.io.writeln(f"?RUNTIME ERROR: {e}")
            return

        if has_measure:
            per_reg, cj = self._locc_engine.sample_joint(self.shots)
            dt = time.time() - t0
            self.io.writeln(f"\nRAN {len(self.program)} lines, LOCC {mode} "
                            f"{sizes_str}q, {self.shots} shots in {dt:.2f}s")
            self._locc_display_results(per_reg, cj)
            if not cj:
                self.io.writeln("\n  (SPLIT mode, no SEND — registers are independent)")
            self.last_counts = cj if cj else per_reg.get('A', {})
        else:
            dt = time.time() - t0
            self.io.writeln(f"\nRAN {len(self.program)} lines, LOCC in {dt:.2f}s")
            regs = ', '.join(self._locc_engine.names)
            self.io.writeln(f"(no MEASURE — use STATE {regs} to inspect)")

    def _locc_execute_program(self, prog: CompiledProgram, *, start_ip: int = 0,
                              run_vars: MutableMapping[str, Any] | None = None, stop_before_send: bool = False) -> int:
        """Execute the compiled program in LOCC mode from statement start_ip.

        Returns the statement where execution stopped (end of program, or the
        first SEND when stop_before_send is set).
        """
        from qubasic_core.engine import ExecContext
        from qubasic_core.expression import Scope
        from qubasic_core.parser import MeasureStmt, SendStmt
        if run_vars is None:
            run_vars = Scope(self.variables)
        ctx = ExecContext(
            sorted_lines=prog, ip=start_ip, run_vars=run_vars,
            max_iterations=self._loop_limit(),
        )
        n = len(prog)
        prev_strict, self._strict_names = self._strict_names, False
        try:
            while ctx.ip < n:
                ctx.iteration_count += 1
                if ctx.iteration_count > ctx.max_iterations:
                    raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
                parsed = prog.stmts[ctx.ip]
                if isinstance(parsed, MeasureStmt):
                    ctx.ip += 1
                    continue
                if stop_before_send and isinstance(parsed, SendStmt):
                    return ctx.ip
                try:
                    result = self._locc_exec_line(prog.texts[ctx.ip], ctx.loop_stack, prog,
                                                  ctx.ip, run_vars, parsed=parsed)
                except Exception as e:
                    raise RuntimeError(f"LINE {prog.lines[ctx.ip]}: {e}") from None
                if result is ExecResult.END:
                    break
                self._advance(ctx, result)
        finally:
            self._strict_names = prev_strict
        return ctx.ip

    def _locc_exec_line(self, stmt: str, loop_stack: list[dict[str, Any]], sorted_lines: Sequence[int], ip: int, run_vars: MutableMapping[str, Any], *, parsed: Stmt | None = None) -> ExecOutcome:
        """Execute one compiled statement in LOCC mode."""
        from qubasic_core.parser import (
            AtRegStmt,
            BarrierStmt,
            EndStmt,
            MeasureStmt,
            RemStmt,
            ReturnStmt,
            SendStmt,
            ShareStmt,
            parse_stmt,
        )
        if parsed is None:
            parsed = parse_stmt(stmt)

        # Fast-path for terminals
        if isinstance(parsed, (RemStmt, MeasureStmt, BarrierStmt)):
            return ExecResult.ADVANCE
        if isinstance(parsed, EndStmt):
            return ExecResult.END
        if isinstance(parsed, ReturnStmt):
            return self._cf_return(sorted_lines, parsed)[1]

        # Fast-path for LOCC-specific statements
        if isinstance(parsed, SendStmt):
            qubit = int(self.eval_expr(parsed.qubit_expr))
            outcome = self._locc_engine.send(parsed.reg, qubit)
            run_vars[parsed.var] = outcome
            self.variables[parsed.var] = outcome
            self._locc_engine.classical[parsed.var] = outcome
            self._locc_engine.correction_log.append(
                f"SEND {parsed.reg}[{qubit}] -> {parsed.var}={outcome}")
            return ExecResult.ADVANCE
        if isinstance(parsed, ShareStmt):
            self._locc_engine.share(parsed.reg1, parsed.q1, parsed.reg2, parsed.q2)
            return ExecResult.ADVANCE
        if isinstance(parsed, AtRegStmt):
            if self._locc_try_special(parsed.reg, parsed.inner, run_vars):
                return ExecResult.ADVANCE
            self._locc_apply_gate(parsed.reg, parsed.inner)
            return ExecResult.ADVANCE

        # Control flow
        handled, result = self._exec_control_flow(
            stmt, loop_stack, sorted_lines, ip, run_vars,
            lambda s, ls, sl, i, rv: self._locc_exec_line(s, ls, sl, i, rv),
            parsed=parsed)
        if handled:
            assert result is not None          # a handled statement has an outcome
            return result

        raise ValueError(f"LOCC mode requires @A/@B prefix, SEND, SHARE, or IF: {stmt}")

    def _locc_try_special(self, reg: str, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """Handle MEAS/RESET/MEASURE_X/Y/Z/SYNDROME in LOCC mode."""
        from qubasic_core.parser import MeasStmt, MeasureBasisStmt, ResetStmt, parse_stmt
        p = parse_stmt(stmt)

        if isinstance(p, MeasStmt):
            qubit = int(self._eval_with_vars(p.qubit_expr, run_vars))
            outcome = self._locc_engine.send(reg, qubit)
            run_vars[p.var] = outcome
            self.variables[p.var] = outcome
            self._locc_engine.classical[p.var] = outcome
            return True

        if isinstance(p, ResetStmt):
            qubit = int(self._eval_with_vars(p.qubit_expr, run_vars))
            outcome = self._locc_engine.send(reg, qubit)
            if outcome == 1:
                self._locc_engine.apply(reg, 'X', (), [qubit])
            return True

        if isinstance(p, MeasureBasisStmt):
            qubit = int(self._eval_with_vars(p.qubit_expr, run_vars))
            if p.basis == 'X':
                self._locc_engine.apply(reg, 'H', (), [qubit])
            elif p.basis == 'Y':
                self._locc_engine.apply(reg, 'SDG', (), [qubit])
                self._locc_engine.apply(reg, 'H', (), [qubit])
            outcome = self._locc_engine.send(reg, qubit)
            var = f"m{p.basis.lower()}_{qubit}"
            run_vars[var] = outcome
            self.variables[var] = outcome
            self._locc_engine.classical[var] = outcome
            return True

        parsed = self._parse_syndrome(stmt, run_vars)
        if parsed is not None:
            pauli_str, qubits, var = parsed
            n_reg = self._locc_engine.get_size(reg)
            anc = n_reg - 1
            if anc in qubits:
                raise ValueError(
                    f"Qubit {anc} needed as ancilla but appears in stabilizer. "
                    f"Increase register size by 1.")
            self._locc_engine.apply(reg, 'H', (), [anc])
            for pc, q in zip(pauli_str, qubits):
                if pc != 'I':
                    self._locc_engine.apply(reg, self._PAULI_TO_CONTROLLED[pc], (), [anc, q])
            self._locc_engine.apply(reg, 'H', (), [anc])
            outcome = self._locc_engine.send(reg, anc)
            if outcome == 1:
                self._locc_engine.apply(reg, 'X', (), [anc])
            run_vars[var] = outcome
            self.variables[var] = outcome
            self._locc_engine.classical[var] = outcome
            return True

        return False

    def _locc_apply_gate(self, reg: str, gate_stmt: str) -> None:
        """Parse and apply a gate to a LOCC register via numpy.

        Uses LOCCRegBackend for standard gates. CTRL and INV modifiers
        use apply_matrix directly since they need matrix construction.
        """
        from qubasic_core.simtools import LOCCRegBackend
        backend = LOCCRegBackend(self.locc, reg)
        n_reg = self._locc_engine.get_size(reg)
        expanded = self._expand_statement(gate_stmt)
        for stmt in expanded:
            tokens = self._tokenize_gate(stmt)
            if not tokens:
                continue
            gate_name = tokens[0].upper()
            gate_name = GATE_ALIASES.get(gate_name, gate_name)
            if gate_name == 'BARRIER':
                continue

            # CTRL modifier: one control on any built-in or UNITARY gate
            m_ctrl = RE_CTRL.match(stmt)
            if m_ctrl:
                canonical, params, qubits = self._modifier_operands(
                    'CTRL', m_ctrl.group(1), m_ctrl.group(2), limit=n_reg)
                inner, targets = self._np_operand(canonical, params, qubits[1:])
                dim = inner.shape[0]
                controlled = np.eye(2 * dim, dtype=complex)
                controlled[dim:, dim:] = inner
                self._locc_engine.apply_matrix(reg, controlled, [qubits[0]] + targets)
                continue

            # INV modifier
            m_inv = RE_INV.match(stmt)
            if m_inv:
                canonical, params, qubits = self._modifier_operands(
                    'INV', m_inv.group(1), m_inv.group(2), limit=n_reg)
                inner, targets = self._np_operand(canonical, params, qubits)
                self._locc_engine.apply_matrix(reg, inner.conj().T, targets)
                continue

            info = self._gate_info(gate_name)
            if info is None:
                raise ValueError(f"UNKNOWN GATE: {gate_name}")
            n_params, n_qubits_needed = info
            args = tokens[1:]
            if len(args) < n_params + n_qubits_needed:
                raise ValueError(f"{gate_name} needs {n_params} params + "
                                 f"{n_qubits_needed} qubits")
            gparams = tuple(self.eval_expr(a) for a in args[:n_params])
            qubits = [self._resolve_qubit(a, n_qubits=n_reg) for a in args[n_params:n_params+n_qubits_needed]]
            if gate_name in self._custom_gates:
                matrix, targets = self._np_operand(gate_name, gparams, qubits)
                self._locc_engine.apply_matrix(reg, matrix, targets)
            else:
                backend.apply_gate(gate_name, gparams, qubits)

    def _np_operand(self, canonical: str, params: Sequence[float], qubits: list[int]) -> tuple[np.ndarray, list[int]]:
        """(matrix, qubits) in the numpy engine's MSB-first convention.

        Built-in gate matrices are MSB-first over the listed qubits already. A
        UNITARY matrix follows Qiskit's order, where the first listed qubit is
        the least significant bit of its index, so its qubit list is reversed.
        """
        if canonical in self._custom_gates:
            return np.asarray(self._custom_gates[canonical]), list(reversed(qubits))
        return _np_gate_matrix(canonical, tuple(params)), list(qubits)

    def _locc_display_results(self, per_reg: dict, counts_joint: dict) -> None:
        """Display per-register and joint histograms."""
        for name in self._locc_engine.names:
            size = self._locc_engine.get_size(name)
            self.io.writeln(f"\n  Register {name} ({size}q):")
            self.print_histogram(per_reg[name])
        if counts_joint and self._locc_engine.n_regs <= 4:
            jlabel = '|'.join(self._locc_engine.names)
            self.io.writeln(f"\n  Joint ({jlabel}):")
            self.print_histogram(counts_joint)


# ---------------------------------------------------------------------------
# QUBASIC LOCC mixin — re-exports from split modules.
# ---------------------------------------------------------------------------

class LOCCMixin(LOCCCommandsMixin, LOCCExecutionMixin):
    """Combined LOCC mixin — commands, display, and execution."""
    pass
