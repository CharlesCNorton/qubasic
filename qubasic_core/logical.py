"""QUBASIC logical qubits: lattice surgery, encoded simulation and the
LQUBITS mode.
"""

from __future__ import annotations

import re
from math import comb
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from qubasic_core import qec as S
from qubasic_core.host import HostState

if TYPE_CHECKING:
    from qiskit import ClassicalRegister, QuantumCircuit

    from qubasic_core.qec import DetectorModel


# ---------------------------------------------------------------------------
# QUBASIC lattice surgery on rotated surface-code patches.
#
#     LATTICE <a> <b> [ZZ|XX] [D d] [SHOTS n]   merge and split two patches
#     LATTICE CNOT <c> <t> [D d] [SHOTS n]      a logical CNOT by lattice surgery
#     LATTICE RATE <d[,d...]> <p> [ROUNDS r] [TRIALS n] [ZZ|XX]   noisy merge
#
# The first two simulate the physical circuit exactly on Aer's stabilizer
# simulator. Each patch is prepared in its logical state (a Clifford circuit
# synthesized from the code's stabilizers and the logical operator), a merge
# initializes the line of qubits between two patches (|+> for ZZ, |0> for XX)
# and measures every check of the merged patch through an ancilla, and a split
# measures that line again (X for ZZ, Z for XX). The merge outcome is the
# product of the merged checks that multiply to the joint logical operator,
# read from the same round. A merge reports the outcome distribution and how
# often the logical readout of the split patches agrees with it; CNOT runs
# M_ZZ(C, A), M_XX(A, T) and M_Z(A) with the ancilla patch A in |+>, applies the
# Pauli frame (an X on T when the ZZ outcome, the Z outcome of A and the split
# outcome on the shared row disagree in parity) and reports the truth table
# read from the control and target patches.
#
# RATE estimates how often a noisy ZZ (or XX) merge decodes to the wrong
# outcome. Under phenomenological noise every data qubit of the merged patch
# takes an error with probability p before each of the r rounds (default d)
# and every check outcome flips with probability p; the final readout of the
# two patches is noisy too. Detectors compare consecutive rounds (the new
# checks have no value before the merge), and the space-time syndrome is
# decoded by exact minimum-weight matching (dem.py). A failure is a wrong merge
# outcome or a logical error on either patch.
# ---------------------------------------------------------------------------

_STATES = ('0', '1', '+', '-')


def _logical_prep(patch: S.Patch, state: str) -> tuple[QuantumCircuit, list[int]]:
    """A circuit (on the patch's qubits in sorted order) preparing |state>_L
    with every check +1."""
    from qiskit.quantum_info import StabilizerState
    qubits = patch.qubits
    order = {q: k for k, q in enumerate(qubits)}
    n = len(qubits)

    def label(support: list[int], ch: str, sign: str = '') -> str:
        s = ['I'] * n
        for q in support:
            s[n - 1 - order[q]] = ch
        return sign + ''.join(s)

    stabs = [label(f, 'X') for f in patch.x_checks] + [label(f, 'Z') for f in patch.z_checks]
    logical = {'0': label(patch.logical_z, 'Z'), '1': label(patch.logical_z, 'Z', '-'),
               '+': label(patch.logical_x, 'X'), '-': label(patch.logical_x, 'X', '-')}[state]
    return StabilizerState.from_stabilizer_list(stabs + [logical]).clifford.to_circuit(), qubits


def _measure_checks(qc: QuantumCircuit, checks: list[list[int]], kinds: list[str], anc: int, creg: ClassicalRegister) -> None:
    for k, (f, kind) in enumerate(zip(checks, kinds)):
        if kind == 'Z':
            for q in f:
                qc.cx(q, anc)
        else:
            qc.h(anc)
            for q in f:
                qc.cx(anc, q)
            qc.h(anc)
        qc.measure(anc, creg[k])
        qc.reset(anc)


def _bits(word: str, idx: list[int]) -> int:
    """XOR of the register bits at idx (word: register string, bit 0 rightmost)."""
    v = 0
    for i in idx:
        v ^= int(word[len(word) - 1 - i])
    return v


class LatticeMixin(HostState):
    """LATTICE: exact and noisy lattice surgery (see the module docstring)."""

    def _lattice_run(self, qc: QuantumCircuit, shots: int) -> list[list[str]]:
        """Per-shot register words, in register creation order."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method='stabilizer')
        kw: dict = {'shots': shots, 'memory': True}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        mem = sim.run(transpile(qc, sim), **kw).result().get_memory()
        return [list(reversed(word.split())) for word in mem]

    def _merge_circuit(self, sg: S.Surgery, a: str, b: str) -> tuple[QuantumCircuit, list[list[int]], list[int]]:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ClassicalRegister, QuantumRegister
        n = sg.n_data + 1
        anc = sg.n_data
        q = QuantumRegister(n, 'q')
        checks = sg.merged.x_checks + sg.merged.z_checks
        kinds = ['X'] * len(sg.merged.x_checks) + ['Z'] * len(sg.merged.z_checks)
        data = sg.a.qubits + sg.b.qubits
        cm = ClassicalRegister(len(checks), 'merge')
        cg = ClassicalRegister(len(sg.gap), 'split')
        cd = ClassicalRegister(len(data), 'data')
        qc = QuantumCircuit(q, cm, cg, cd)
        for patch, st in ((sg.a, a), (sg.b, b)):
            prep, qs = _logical_prep(patch, st)
            qc.compose(prep, qubits=qs, inplace=True)
        if sg.kind == 'ZZ':
            for g in sg.gap:
                qc.h(g)
        _measure_checks(qc, checks, kinds, anc, cm)
        for k, g in enumerate(sg.gap):
            if sg.kind == 'ZZ':
                qc.h(g)
            qc.measure(g, cg[k])
        for k, dq in enumerate(data):
            if sg.kind == 'XX':
                qc.h(dq)
            qc.measure(dq, cd[k])
        return qc, checks, data

    def _lattice_merge(self, a: str, b: str, kind: str, d: int, shots: int) -> None:
        sg = S.surgery(d, kind)
        qc, checks, data = self._merge_circuit(sg, a, b)
        nx = len(sg.merged.x_checks)
        if kind == 'ZZ':
            sub = [nx + k for k in S.product_subset(sg.merged.z_checks,
                                                    sg.a.logical_z + sg.b.logical_z, sg.n_data)]
            ra, rb = sg.a.logical_z, sg.b.logical_z
        else:
            sub = S.product_subset(sg.merged.x_checks, sg.a.logical_x + sg.b.logical_x, sg.n_data)
            ra, rb = sg.a.logical_x, sg.b.logical_x
        pos = {q: k for k, q in enumerate(data)}
        outcomes = {0: 0, 1: 0}
        agree = 0
        for merge_w, _split_w, data_w in self._lattice_run(qc, shots):
            m = _bits(merge_w, sub)
            outcomes[m] += 1
            readout = _bits(data_w, [pos[q] for q in ra]) ^ _bits(data_w, [pos[q] for q in rb])
            agree += readout == m
        op = 'Zbar_A Zbar_B' if kind == 'ZZ' else 'Xbar_A Xbar_B'
        self.io.writeln(f"\n  Lattice surgery {kind}: |{a}>_L (A) and |{b}>_L (B), distance {d} "
                        f"rotated surface patches")
        self.io.writeln(f"    {sg.n_data} data qubits + 1 ancilla; merge measures "
                        f"{len(checks)} checks, {len(sg.new_checks)} of them new; "
                        f"the outcome is the product of {len(sub)}")
        self.io.writeln(f"    {op} = +1: {outcomes[0]}   -1: {outcomes[1]}   ({shots} shots)")
        self.io.writeln(f"    split readout agrees with the merge outcome in "
                        f"{agree}/{shots} shots")
        self.variables['_LATTICE_PLUS'] = outcomes[0] / shots
        self.variables['_LATTICE_AGREE'] = agree / shots

    def _lattice_cnot(self, c: str, t: str, d: int, shots: int) -> None:
        """CNOT from M_ZZ(C, A), M_XX(A, T), M_Z(A) on three patches: C above A,
        T to the right of A (exact stabilizer simulation)."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import ClassicalRegister, QuantumRegister
        if c not in '01' or t not in '01':
            raise ValueError("CNOT takes computational-basis inputs 0 or 1")
        # Global grid: rows 0..2d, cols 0..2d. C: rows 0..d-1, cols 0..d-1;
        # gap row d (cols 0..d-1); A: rows d+1..2d, cols 0..d-1; gap column d
        # (rows d+1..2d); T: rows d+1..2d, cols d+1..2d.
        cells: dict[tuple[int, int], int] = {}

        def cell(r: int, col: int) -> int:
            if (r, col) not in cells:
                cells[(r, col)] = len(cells)
            return cells[(r, col)]
        C = S.make_patch(d, d, index={(r, k): cell(r, k) for r in range(d) for k in range(d)})
        gap1 = [cell(d, k) for k in range(d)]
        A = S.make_patch(d, d, index={(r, k): cell(r + d + 1, k) for r in range(d) for k in range(d)})
        gap2 = [cell(r + d + 1, d) for r in range(d)]
        T = S.make_patch(d, d, index={(r, k): cell(r + d + 1, k + d + 1)
                                      for r in range(d) for k in range(d)})
        m1 = S.make_patch(2 * d + 1, d, index={(r, k): cell(r, k) for r in range(2 * d + 1)
                                               for k in range(d)})
        m2 = S.make_patch(d, 2 * d + 1, index={(r, k): cell(r + d + 1, k) for r in range(d)
                                               for k in range(2 * d + 1)})
        n_data = len(cells)
        anc = n_data
        q = QuantumRegister(n_data + 1, 'q')
        checks1 = m1.x_checks + m1.z_checks
        kinds1 = ['X'] * len(m1.x_checks) + ['Z'] * len(m1.z_checks)
        checks2 = m2.x_checks + m2.z_checks
        kinds2 = ['X'] * len(m2.x_checks) + ['Z'] * len(m2.z_checks)
        r1 = ClassicalRegister(len(checks1), 'zz')
        s1 = ClassicalRegister(d, 'split1')
        r2 = ClassicalRegister(len(checks2), 'xx')
        s2 = ClassicalRegister(d, 'split2')
        data = C.qubits + A.qubits + T.qubits
        rd = ClassicalRegister(len(data), 'data')
        qc = QuantumCircuit(q, r1, s1, r2, s2, rd)
        for patch, st in ((C, c), (A, '+'), (T, t)):
            prep, qs = _logical_prep(patch, st)
            qc.compose(prep, qubits=qs, inplace=True)
        for g in gap1:
            qc.h(g)
        _measure_checks(qc, checks1, kinds1, anc, r1)
        for k, g in enumerate(gap1):
            qc.h(g)
            qc.measure(g, s1[k])
        _measure_checks(qc, checks2, kinds2, anc, r2)
        for k, g in enumerate(gap2):
            qc.measure(g, s2[k])
        for k, dq in enumerate(data):
            qc.measure(dq, rd[k])
        n1 = len(m1.x_checks)
        sub1 = [n1 + k for k in S.product_subset(m1.z_checks, C.logical_z + A.logical_z, n_data)]
        pos = {qq: k for k, qq in enumerate(data)}
        row0_gap = 0                      # the gap qubit on A's and T's row 0
        table: dict[tuple[int, int], int] = {}
        ok = 0
        for zz_w, _s1_w, _xx_w, s2_w, data_w in self._lattice_run(qc, shots):
            mzz = _bits(zz_w, sub1)
            mz_a = _bits(data_w, [pos[x] for x in A.logical_z])
            g0 = int(s2_w[len(s2_w) - 1 - row0_gap])
            c_out = _bits(data_w, [pos[x] for x in C.logical_z])
            t_out = _bits(data_w, [pos[x] for x in T.logical_z]) ^ mzz ^ mz_a ^ g0
            table[(c_out, t_out)] = table.get((c_out, t_out), 0) + 1
            ok += (c_out, t_out) == (int(c), int(t) ^ int(c))
        self.io.writeln(f"\n  Lattice-surgery CNOT, distance {d}: control |{c}>_L, target |{t}>_L, "
                        f"ancilla patch in |+>_L")
        self.io.writeln(f"    {n_data} data qubits + 1 ancilla; M_ZZ(C,A) over {len(checks1)} checks, "
                        f"M_XX(A,T) over {len(checks2)}, then M_Z(A)")
        outs = ', '.join(f"|{co}{to}> {n}" for (co, to), n in sorted(table.items()))
        self.io.writeln(f"    readout (control target): {outs}   expected |{c}{int(t) ^ int(c)}>")
        self.io.writeln(f"    correct in {ok}/{shots} shots")
        self.variables['_LATTICE_AGREE'] = ok / shots

    def _lattice_model(self, d: int, p: float, rounds: int, kind: str) -> tuple[DetectorModel, int]:
        """The phenomenological detector model of a noisy merge: observable 0
        is the merge outcome, 1 and 2 the logical readouts of A and B."""
        from qubasic_core.qec import DetectorModel
        sg = S.surgery(d, kind)
        if kind == 'ZZ':
            checks = sg.merged.z_checks
            nx = len(sg.merged.x_checks)
            new = {k - nx for k in sg.new_checks if k >= nx}
            sub = set(S.product_subset(checks, sg.a.logical_z + sg.b.logical_z, sg.n_data))
            la, lb = set(sg.a.logical_z), set(sg.b.logical_z)
        else:
            checks = sg.merged.x_checks
            new = {k for k in sg.new_checks if k < len(sg.merged.x_checks)}
            sub = set(S.product_subset(checks, sg.a.logical_x + sg.b.logical_x, sg.n_data))
            la, lb = set(sg.a.logical_x), set(sg.b.logical_x)
        nc = len(checks)
        gap = set(sg.gap)
        holds = [[k for k, f in enumerate(checks) if q in f] for q in range(sg.n_data)]
        # Detector ids: round t (1..rounds) for every check, the new ones from
        # round 2; round rounds+1 (the final readout) for the old checks only.
        det: dict[tuple[int, int], int] = {}
        for t in range(1, rounds + 1):
            for k in range(nc):
                if not (t == 1 and k in new):
                    det[(t, k)] = len(det)
        for k in range(nc):
            if k not in new:
                det[(rounds + 1, k)] = len(det)
        model = DetectorModel(len(det), 3)
        for t in range(1, rounds + 2):
            for q in range(sg.n_data):
                if t == rounds + 1 and q in gap:
                    continue           # the line is measured in the other basis at the split
                flips = [det[(t, k)] for k in holds[q] if (t, k) in det]
                obs = 0
                if t == 1 and sum(k in sub for k in holds[q]) % 2:
                    obs |= 1
                if q in la:
                    obs |= 2
                if q in lb:
                    obs |= 4
                model.add(p, flips, obs)
            if t <= rounds:
                for k in range(nc):
                    flips = [det[(tt, k)] for tt in (t, t + 1) if (tt, k) in det]
                    model.add(p, flips, 1 if (t == 1 and k in sub) else 0)
        return model, len(det)

    def _lattice_rate(self, ds: list[int], p: float, rounds: int | None, trials: int,
                      kind: str) -> None:
        rng = np.random.default_rng(self._seed)
        self.io.writeln(f"\n  Noisy lattice-surgery {kind} merge: phenomenological p={p:g}, "
                        f"{trials} trials per distance (space-time matching)")
        self.io.writeln(f"    {'d':>3}  {'rounds':>6}  {'detectors':>9}  {'failure':>9}  "
                        f"{'merge':>9}  {'patch A':>9}  {'patch B':>9}")
        for d in ds:
            r = rounds or d
            model, ndet = self._lattice_model(d, p, r, kind)
            rate, per = model.logical_failure_rate(trials, rng)
            self.io.writeln(f"    {d:>3}  {r:>6}  {ndet:>9}  {rate:>9.4f}  {per[0]:>9.4f}  "
                            f"{per[1]:>9.4f}  {per[2]:>9.4f}")
            self.variables['_LATTICE_RATE'] = rate

    def cmd_lattice(self, rest: str) -> None:
        """LATTICE <a> <b> [ZZ|XX] [D d] | CNOT <c> <t> [D d] | RATE <d,...> <p> [ROUNDS r] — lattice surgery on surface-code patches: a merge, a CNOT, or a noisy merge's failure rate.

        Merges and splits rotated surface-code patches in an exact stabilizer
        simulation, runs a lattice-surgery CNOT, or estimates the logical
        failure rate of a noisy merge under phenomenological noise with
        space-time matching. See qubasic_core/logical.py."""
        text = rest.strip()
        opts: dict[str, Any] = {}
        for key in ('D', 'SHOTS', 'ROUNDS', 'TRIALS'):
            m = re.search(rf'\b{key}\s+(\d+)', text, re.IGNORECASE)
            if m:
                opts[key] = int(m.group(1))
                text = (text[:m.start()] + text[m.end():]).strip()
        kind = 'XX' if re.search(r'\bXX\b', text, re.IGNORECASE) else 'ZZ'
        text = re.sub(r'\b(ZZ|XX)\b', '', text, flags=re.IGNORECASE).strip()
        parts = text.split()
        usage = ("?USAGE: LATTICE <a> <b> [ZZ|XX] [D d] [SHOTS n]  |  LATTICE CNOT <c> <t> [D d]  |  "
                 "LATTICE RATE <d[,d..]> <p> [ROUNDS r] [TRIALS n]   (states: 0, 1, +, -)")
        try:
            d = opts.get('D', 3)
            shots = opts.get('SHOTS', 200)
            if parts and parts[0].upper() == 'RATE':
                if len(parts) < 3:
                    self.io.writeln(usage)
                    return
                ds = [int(x) for x in parts[1].split(',') if x]
                p = float(self._eval_with_vars(parts[2], {}))
                if not 0 < p < 0.5:
                    raise ValueError("p must be in (0, 0.5)")
                self._lattice_rate(ds, p, opts.get('ROUNDS'), opts.get('TRIALS', 2000), kind)
                return
            if parts and parts[0].upper() == 'CNOT':
                if len(parts) < 3:
                    self.io.writeln(usage)
                    return
                self._lattice_cnot(parts[1], parts[2], d, shots)
                return
            if len(parts) < 2 or parts[0] not in _STATES or parts[1] not in _STATES:
                self.io.writeln(usage)
                return
            self._lattice_merge(parts[0], parts[1], kind, d, shots)
        except Exception as e:
            self.io.writeln(f"?LATTICE ERROR: {e}")


# ---------------------------------------------------------------------------
# QUBASIC encoded logical qubits simulated by Pauli frames.
#
# `LQUBITS n CODE SURFACE d PHYS p` (also REP and SHOR) runs a Clifford
# program encoded: each logical qubit is a patch of the code, and every
# logical gate is a transversal physical gate on the patches it touches:
#
#     X, Y, Z    on every qubit of the patch
#     H          (surface code) on every qubit, then the patch is relabelled
#                by a quarter turn, which takes the code back to its own layout
#     CX         qubit k of the control patch to qubit k of the target patch
#     CZ         H, CX, H on the target (surface code)
#     SWAP       exchanges which patch is which (no physical gate)
#
# Depolarizing noise at the physical rate p follows every physical gate
# (probability 3p/4 of an X, Y or Z on each qubit of a one-qubit layer, 15p/16
# of a non-identity pair on each pair of a CX layer), and an X flip at rate p
# precedes each readout. A round of ideal syndrome extraction follows every
# logical gate on the patches it touched, and the decoder (minimum-weight
# matching for the surface and repetition codes, the lookup table for Shor's
# code) updates the Pauli frame rather than the qubits. The readout's
# syndrome is decoded once more, and a logical qubit's outcome flips when the
# residual X error crosses its logical Z.
#
# Because every operation is a Clifford gate and every error a Pauli, the
# errors and the ideal computation separate exactly: the ideal outcomes are
# sampled from the logical circuit (Aer's stabilizer simulator) and each shot's
# outcome is flipped by the logical error its frame accumulated.
# ---------------------------------------------------------------------------

# Logical gates each code implements transversally (qiskit gate names).
_TRANSVERSAL = {
    'SURFACE': {'id', 'x', 'y', 'z', 'h', 'cx', 'cz', 'swap'},
    'REP': {'id', 'x', 'y', 'z', 'cx', 'swap'},
    'SHOR': {'id', 'x', 'y', 'z', 'cx', 'swap'},
}


class FrameSimulation:
    """Vectorized Pauli-frame simulation of an encoded Clifford circuit."""

    def __init__(self, code: dict, n_logical: int, p: float, shots: int, rng: Any,
                 syndromes: Callable, decode: Callable, h_perm: list[int] | None) -> None:
        n = code['n']
        self.n = n
        self.p = p
        self.rng = rng
        self.shots = shots
        self.syndromes = syndromes
        self.decode = decode
        self.h_perm = None if h_perm is None else np.array(h_perm)
        shape = (shots, n_logical, n)
        self.ex = np.zeros(shape, dtype=np.uint8)
        self.ez = np.zeros(shape, dtype=np.uint8)
        self.fx = np.zeros(shape, dtype=np.uint8)
        self.fz = np.zeros(shape, dtype=np.uint8)
        self.lz = np.array([ch in 'ZY' for ch in code['lz']], dtype=np.uint8)
        self.rounds = 0

    # noise
    def _noise1(self, l: int) -> None:
        if self.p <= 0:
            return
        hit = self.rng.random((self.shots, self.n)) < 0.75 * self.p
        which = self.rng.integers(0, 3, (self.shots, self.n))      # 0 X, 1 Y, 2 Z
        self.ex[:, l] ^= (hit & (which != 2)).astype(np.uint8)
        self.ez[:, l] ^= (hit & (which != 0)).astype(np.uint8)

    def _noise2(self, a: int, b: int) -> None:
        if self.p <= 0:
            return
        hit = self.rng.random((self.shots, self.n)) < (15 / 16) * self.p
        which = self.rng.integers(1, 16, (self.shots, self.n))     # (Pa, Pb), not (I, I)
        pa, pb = which % 4, which // 4                             # 0 I, 1 X, 2 Y, 3 Z
        for l, pauli in ((a, pa), (b, pb)):
            self.ex[:, l] ^= (hit & ((pauli == 1) | (pauli == 2))).astype(np.uint8)
            self.ez[:, l] ^= (hit & ((pauli == 2) | (pauli == 3))).astype(np.uint8)

    # gates
    def gate(self, name: str, qs: list[int]) -> None:
        if name in ('id', 'x', 'y', 'z'):
            self._noise1(qs[0])
        elif name == 'h':
            l = qs[0]
            perm = self.h_perm
            for e_x, e_z in ((self.ex, self.ez), (self.fx, self.fz)):
                old_x, old_z = e_x[:, l].copy(), e_z[:, l].copy()
                if perm is None:
                    e_x[:, l], e_z[:, l] = old_z, old_x
                else:
                    e_x[:, l][:, perm] = old_z
                    e_z[:, l][:, perm] = old_x
            self._noise1(l)
        elif name == 'cx':
            a, b = qs
            for e_x, e_z in ((self.ex, self.ez), (self.fx, self.fz)):
                e_x[:, b] ^= e_x[:, a]
                e_z[:, a] ^= e_z[:, b]
            self._noise2(a, b)
        elif name == 'swap':
            a, b = qs
            for arr in (self.ex, self.ez, self.fx, self.fz):
                arr[:, [a, b]] = arr[:, [b, a]]
            return
        else:
            raise ValueError(f"no transversal form for {name}")
        for l in qs:
            self.correct(l)

    def correct(self, l: int) -> None:
        """One round of ideal syndrome extraction on patch l, decoded into the frame."""
        synd = self.syndromes(self.ex[:, l] ^ self.fx[:, l], self.ez[:, l] ^ self.fz[:, l])
        rx, rz = self.decode(synd)
        self.fx[:, l] ^= rx
        self.fz[:, l] ^= rz
        self.rounds += 1

    def readout_flips(self, targets: list[int]) -> np.ndarray:
        """(shots x len(targets)) logical flips of a Z readout of `targets`."""
        out = np.zeros((self.shots, len(targets)), dtype=np.uint8)
        for j, l in enumerate(targets):
            if self.p > 0:
                self.ex[:, l] ^= (self.rng.random((self.shots, self.n)) < self.p).astype(np.uint8)
            self.correct(l)
            resid = self.ex[:, l] ^ self.fx[:, l]
            out[:, j] = (resid.astype(np.int64) @ self.lz.astype(np.int64)) & 1
        return out


class EncodedMixin(HostState):
    """RUN under LQUBITS with an encoded code (see the module docstring)."""

    def _encodable(self, logical: QuantumCircuit, code_name: str) -> str | None:
        """None when every gate of the logical circuit is transversal for the
        code, else the name of the first gate that is not."""
        allowed = _TRANSVERSAL.get(code_name, set())
        for inst in logical.data:
            name = inst.operation.name
            if name in ('barrier', 'measure'):
                continue
            if name not in allowed:
                return name
        return None

    def _run_encoded(self, t0: float) -> bool:
        """RUN the program encoded. Returns False (after a note) when the
        program uses a gate the code cannot apply transversally, so the
        caller runs the modeled logical channel instead."""
        import time
        lm = self._logical_mode
        if lm is None:
            return False
        if lm['code'] == 'STEANE':
            self._run_steane(t0)
            return True
        try:
            logical, _ = self.build_circuit()
        except Exception as e:
            self.io.writeln(f"?BUILD ERROR: {e}")
            return True
        if self._classical_bits:
            self.io.writeln(f"?CODE {lm['code']} measures logical qubits once, at the end; "
                            f"use MEASURE rather than MEAS")
            return True
        cname = 'SURFACE' if lm['code'] in ('SURFACE', 'SURF') else \
            'REP' if lm['code'] in ('REP', 'REPETITION', 'BITFLIP') else lm['code']
        bad = self._encodable(logical, cname)
        if bad is not None:
            if lm.get('force_encoded'):
                self.io.writeln(f"?'{bad.upper()}' has no transversal form on {cname}; the "
                                f"encoded simulation runs {', '.join(sorted(g.upper() for g in _TRANSVERSAL[cname]))}")
                return True
            self.io.writeln(f"  ('{bad.upper()}' has no transversal form on {cname}: this run "
                            f"uses the modeled logical channel)")
            return False
        d, p, n = lm['d'], lm['p'], logical.num_qubits
        code = self._qec_code(cname, d)
        decoder = self._choose_decoder(code)
        from qubasic_core.qec import h_relabel
        h_perm = h_relabel(code['d']) if cname == 'SURFACE' else None
        rng = np.random.default_rng(self._seed)
        sim = FrameSimulation(code, n, p, self.shots, rng,
                              lambda ex, ez: self._syndromes(code, ex, ez),
                              lambda synd: self._decode_batch(code, synd, decoder), h_perm)
        for inst in logical.data:
            name = inst.operation.name
            qs = [logical.find_bit(q).index for q in inst.qubits]
            if name in ('barrier', 'measure'):
                continue
            if name == 'cz':
                sim.gate('h', [qs[1]])
                sim.gate('cx', qs)
                sim.gate('h', [qs[1]])
            else:
                sim.gate(name, qs)
        targets = list(self._measure_subset) if self._measure_subset else list(range(n))
        flips = sim.readout_flips(targets)
        ideal = self._ideal_samples(logical, targets, self.shots)
        outcomes = ideal ^ flips
        counts: dict[str, int] = {}
        for row in outcomes:
            key = ''.join(str(int(v)) for v in row[::-1])
            counts[key] = counts.get(key, 0) + 1
        err = float(np.mean(flips.any(axis=1)))
        self.last_counts = counts
        self.last_sv = None
        self.last_circuit = logical
        dt = time.time() - t0
        per_patch = code['n'] + len(code['stab'])
        self.io.writeln(f"\nRAN {len(self.program)} lines, {n} logical qubits ({code['name']}), "
                        f"{self.shots} shots in {dt:.2f}s")
        self.io.writeln(f"  encoded: {n * code['n']} data + {n * len(code['stab'])} measurement "
                        f"qubits ({per_patch} per patch), {sim.rounds // max(n, 1)} syndrome "
                        f"round(s) per patch, {self._DECODER_LABEL[decoder]}")
        self.io.writeln(f"  depolarizing p={p:g} after every physical gate and on readout; "
                        f"logical error in {err:.4f} of shots")
        self.variables['_LOGICAL_ERR'] = err
        ideal_dist = self._ideal_distribution(logical, targets)
        if ideal_dist is not None:
            keys = set(ideal_dist) | set(counts)
            tvd = 0.5 * sum(abs(counts.get(k, 0) / self.shots - ideal_dist.get(k, 0.0)) for k in keys)
            self.variables['_LOGICAL_TVD'] = tvd
            self.io.writeln(f"  decoded readout vs ideal: total variation distance {tvd:.4f}")
        self.print_histogram(counts)
        return True

    def _ideal_samples(self, logical: QuantumCircuit, targets: list[int], shots: int) -> np.ndarray:
        """(shots x len(targets)) ideal outcomes of the logical Clifford circuit."""
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        qc = QuantumCircuit(logical.num_qubits, len(targets))
        clean = logical.remove_final_measurements(inplace=False)
        qc.compose(clean, inplace=True)
        for j, q in enumerate(targets):
            qc.measure(q, j)
        sim = AerSimulator(method='stabilizer')
        kw: dict = {'shots': shots, 'memory': True}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        mem = sim.run(transpile(qc, sim), **kw).result().get_memory()
        k = len(targets)
        return np.array([[int(word[k - 1 - j]) for j in range(k)] for word in mem], dtype=np.uint8)


# ---------------------------------------------------------------------------
# QUBASIC logical-qubit mode — program at the logical level of a QEC code.
#
#   LQUBITS 2 CODE STEANE PHYS 1e-3     2 logical qubits, encoded Steane simulation
#   LQUBITS 2 CODE SURFACE 5            2 logical qubits, rotated surface d=5
#   LQUBITS 2 CODE SURFACE 5 PHYS 1e-3  ...at physical error rate p
#   LQUBITS 2 CODE SURFACE 5 MODEL      the modeled logical channel only
#   LQUBITS STATUS                      Show the active logical configuration
#   LQUBITS OFF                         Back to physical qubits
#
# While active, program gates address LOGICAL qubits and the histogram is over
# logical outcomes.
#
# CODE STEANE runs the program encoded as a circuit: each logical qubit is a
# [[7,1,3]] block of 7 physical qubits, logical Clifford gates are transversal,
# depolarizing noise at the physical rate follows every physical gate and
# readout, a round of syndrome extraction (ideal, with conditional
# corrections) follows every logical gate, and readout is decoded block by
# block. The report compares the decoded histogram with the ideal one.
#
# CODE SURFACE, REP and SHOR run a program of transversal gates encoded too,
# by Pauli frames with a decoder after every logical gate (encoded.py). A
# program with other gates (T, rotations), and every program under MODEL, runs
# on the modeled code instead: every operation carries the code's per-op
# logical error channel (X/Z flips at the code-capacity logical rate for the
# chosen code, distance, and physical p), measurement carries a logical
# readout flip, and RUN appends a lattice-surgery compilation report (surgery
# ops, syndrome rounds, physical qubits, wall time, error budget). Other
# commands (SAMPLE, SWEEP, ...) see the modeled channel.
# ---------------------------------------------------------------------------

# Steane [[7,1,3]] code: the Hamming parity checks, shared by its X and Z
# stabilizers. Check k covers the qubits q whose q+1 has bit 2-k set, so a
# single flipped qubit q gives the syndrome value q+1.
STEANE_CHECKS = ((3, 4, 5, 6), (1, 2, 5, 6), (0, 2, 4, 6))

# Logical single-qubit Cliffords as transversal physical gates. Transversal S
# implements logical S^dag on this code, so S and SDG swap.
_STEANE_1Q = {
    'id': [], 'x': ['x'], 'y': ['y'], 'z': ['z'], 'h': ['h'],
    's': ['sdg'], 'sdg': ['s'], 'sx': ['h', 'sdg', 'h'], 'sxdg': ['h', 's', 'h'],
}
_STEANE_2Q = {'cx', 'cz', 'swap'}


def steane_decode(bits: str) -> int:
    """Logical Z value of one block's 7 measured bits (bits[i] = qubit i):
    correct the single flip its Hamming syndrome names, then take parity."""
    b = [int(c) for c in bits]
    syndrome = 0
    for i, v in enumerate(b):
        if v:
            syndrome ^= i + 1
    if syndrome:
        b[syndrome - 1] ^= 1
    return sum(b) & 1


class LogicalMixin(HostState):
    """LQUBITS logical-qubit mode for QBasicTerminal.

    Requires: TerminalProtocol — uses self.io, self.variables, self.cmd_qubits,
    self._noise_model, self.last_circuit, self._qec_code / _logical_error_rate
    from the code library (codes.py).
    """

    def _init_logical(self) -> None:
        self._logical_mode: dict | None = None
        self._logical_pl_cache: dict = {}

    # ── Per-operation logical error probability ─────────────────────────

    def _logical_pl(self, name: str, d: int, p: float) -> float:
        """Code-capacity logical error probability for one logical operation.

        Surface uses the scaling estimate p_L = 0.1 (p/0.01)^((d+1)/2) that
        RESOURCES uses; the repetition code uses the exact majority-vote
        failure sum; Steane/Shor (d=3) use a cached code-capacity Monte Carlo
        with the minimum-weight lookup decoder."""
        key = (name, d, round(p, 12))
        if key in self._logical_pl_cache:
            return self._logical_pl_cache[key]
        name = name.upper()
        if name in ('SURFACE', 'SURF'):
            from qubasic_core.benchmarking import surface_logical_rate
            pl = surface_logical_rate(p, d)
        elif name in ('REP', 'REPETITION', 'BITFLIP'):
            t = (d - 1) // 2
            pl = sum(comb(d, j) * p ** j * (1 - p) ** (d - j)
                     for j in range(t + 1, d + 1))
        else:
            code = self._qec_code(name, d)
            rng = np.random.default_rng(self._seed)
            pl = self._logical_error_rate(code, p, 20000, rng)
        pl = float(min(max(pl, 0.0), 0.75))
        self._logical_pl_cache[key] = pl
        return pl

    # ── Command ─────────────────────────────────────────────────────────

    def cmd_lqubits(self, rest: str) -> None:
        """LQUBITS <n> CODE <name> [d] [PHYS p] [MODEL|ENCODED] | STATUS | OFF — run programs on logical qubits, simulated encoded or as a modeled channel.

        Program gates then act on error-corrected logical qubits. RUN simulates
        a program of transversal gates encoded (STEANE as a circuit, SURFACE,
        REP and SHOR by Pauli frames); otherwise each op carries the code's
        logical error channel and RUN reports the lattice-surgery compilation
        (ops, rounds, physical qubits, budget). MODEL always uses the channel,
        ENCODED refuses programs the code cannot run transversally."""
        arg = rest.strip()
        if not arg or arg.upper() == 'STATUS':
            lm = self._logical_mode
            if not lm:
                self.io.writeln("LQUBITS OFF (physical mode)")
            elif lm.get('encoded') and lm['code'] == 'STEANE':
                self.io.writeln(f"LQUBITS {lm['n']}: STEANE [[7,1,3]] phys p={lm['p']:g}  "
                                f"(encoded simulation, {7 * lm['n'] + 1} physical qubits)")
            elif lm.get('encoded'):
                self.io.writeln(f"LQUBITS {lm['n']}: {lm['code']} d={lm['d']} phys p={lm['p']:g}  "
                                f"(encoded Pauli-frame simulation of transversal programs; "
                                f"modeled p_L={lm['pl']:.3e}/op otherwise)")
            else:
                self.io.writeln(f"LQUBITS {lm['n']}: {lm['code']} d={lm['d']} "
                                f"phys p={lm['p']:g}  p_L={lm['pl']:.3e}/op (model)")
            return
        if arg.upper() == 'OFF':
            saved = (self._logical_mode or {}).get('saved_noise')
            self._logical_mode = None
            if saved:
                self._noise_model, self._noise_spec = saved
            else:
                self._noise_model = self._noise_spec = None
            self._circuit_cache_key = None
            restored = " (previous NOISE restored)" if saved and saved[0] is not None else ""
            self.io.writeln(f"LQUBITS OFF — back to physical qubits{restored}")
            return
        import re as _re
        mode = _re.search(r'\s+(MODEL|ENCODED)\s*$', arg, _re.IGNORECASE)
        mode_word = mode.group(1).upper() if mode else ''
        if mode:
            arg = arg[:mode.start()]
        m = _re.match(r'(\d+)\s+CODE\s+(\w+)(?:\s+(\d+))?(?:\s+PHYS\s+(\S+))?\s*$',
                      arg, _re.IGNORECASE)
        if not m:
            self.io.writeln("?USAGE: LQUBITS <n> CODE <SURFACE|REP|STEANE|SHOR> [d] [PHYS p] "
                            "[MODEL|ENCODED]  |  LQUBITS OFF")
            return
        try:
            n = int(m.group(1))
            cname = m.group(2).upper()
            d = int(m.group(3)) if m.group(3) else 5
            if d % 2 == 0:
                d += 1
            p = float(self._eval_with_vars(m.group(4), {})) if m.group(4) else 1e-3
            if cname == 'STEANE':
                saved = (self._logical_mode['saved_noise'] if self._logical_mode
                         else (self._noise_model, self._noise_spec))
                self._noise_model, self._noise_spec = saved    # no modeled channel
                self._logical_mode = {'n': n, 'code': cname, 'd': 3, 'p': p,
                                      'pl': self._logical_pl(cname, 3, p),
                                      'encoded': True, 'patch': 7, 'saved_noise': saved}
                self.cmd_qubits(str(n))
                self.io.writeln(f"LQUBITS {n}: STEANE [[7,1,3]], physical p={p:g} "
                                f"(encoded simulation: {7 * n + 1} physical qubits)")
                return
            pl = self._logical_pl(cname, d, p)

            from qiskit_aer.noise import NoiseModel, ReadoutError, pauli_error
            # Independent logical X and Z flips at rate pl per op.
            px = pz = pl * (1 - pl)
            py = pl * pl
            err1 = pauli_error([('X', px), ('Z', pz), ('Y', py),
                                ('I', 1.0 - px - pz - py)])
            _1q = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg',
                   'sx', 'rx', 'ry', 'rz', 'p', 'u', 'id']
            _2q = ['cx', 'cy', 'cz', 'ch', 'swap', 'dcx', 'iswap',
                   'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz']
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(err1, _1q)
            nm.add_all_qubit_quantum_error(err1.tensor(err1), _2q)
            nm.add_all_qubit_readout_error(
                ReadoutError([[1 - pl, pl], [pl, 1 - pl]]))
            saved = (self._logical_mode['saved_noise'] if self._logical_mode
                     else (self._noise_model, self._noise_spec))
            self._noise_model = nm
            self._noise_spec = None
            self._logical_mode = {'n': n, 'code': cname, 'd': d, 'p': p, 'pl': pl,
                                  'saved_noise': saved, 'encoded': mode_word != 'MODEL',
                                  'force_encoded': mode_word == 'ENCODED'}
            self.cmd_qubits(str(n))
            patch = 2 * d * d - 1 if cname in ('SURFACE', 'SURF') else \
                self._qec_code(cname, d)['n']
            self._logical_mode['patch'] = patch
            self.io.writeln(f"LQUBITS {n}: {cname} d={d}, physical p={p:g} "
                            f"({patch} physical qubits per patch)")
            if mode_word != 'MODEL':
                self.io.writeln("  RUN simulates a program of transversal gates encoded (Pauli "
                                "frames, a decoder after every logical gate); other programs use "
                                f"the modeled channel, p_L = {pl:.3e} per op")
                return
            self.io.writeln(f"  p_L = {pl:.3e} per logical op (the modeled logical error "
                            f"channel per operation)")
            if pl > 0.05:
                self.io.writeln("  WARNING: p is at or above threshold for this "
                                "distance — logical qubits are worse than physical")
        except Exception as e:
            self.io.writeln(f"?LQUBITS ERROR: {e}")

    # ── Post-run lattice-surgery report ─────────────────────────────────

    def _logical_trailer(self) -> None:
        lm = self._logical_mode
        if not lm or self.last_circuit is None:
            return
        d, pl, patch, n = lm['d'], lm['pl'], lm['patch'], lm['n']
        ops = dict(self.last_circuit.count_ops())
        ops.pop('measure', None)
        ops.pop('barrier', None)
        _2q_names = {'cx', 'cy', 'cz', 'ch', 'swap', 'dcx', 'iswap',
                     'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz'}
        n2 = sum(c for g, c in ops.items() if g in _2q_names)
        n_t = ops.get('t', 0) + ops.get('tdg', 0)
        n1 = sum(ops.values()) - n2
        # Lattice-surgery schedule: a 2-qubit gate is a merge + split (2 surgery
        # ops, 2d rounds via an ancilla patch); 1-qubit Cliffords are
        # transversal (d rounds of syndrome extraction each); final readout d.
        surgery = 2 * n2
        rounds = 2 * d * n2 + d * n1 + d
        phys = (n + (1 if n2 else 0)) * patch
        total_ops = n1 + 2 * n2
        budget = 1.0 - (1.0 - pl) ** max(total_ops, 1)
        self.io.writeln(f"  logical: {lm['code']} d={d}  p_L={pl:.2e}/op  "
                        f"{surgery} surgery ops ({n2} CX-class)  "
                        f"{n_t} T (distillation required)")
        self.io.writeln(f"  logical: ~{rounds} syndrome rounds "
                        f"(~{rounds:.0f} us at 1 us/round), "
                        f"{phys:,} physical qubits "
                        f"({n} patches{' + 1 routing ancilla' if n2 else ''})")
        self.io.writeln(f"  logical: cumulative error budget = "
                        f"1-(1-p_L)^{total_ops} = {budget:.3e}")

    # ── Encoded Steane simulation ───────────────────────────────────────

    def _steane_encode(self, logical: QuantumCircuit, p: float) -> tuple:
        """The physical circuit for a logical Clifford circuit on Steane blocks.

        Returns (circuit, rounds): block l is data qubits 7l..7l+6, the last
        qubit is the syndrome ancilla. Depolarizing noise at rate p follows
        every physical gate; each logical gate is followed by one round of
        ideal syndrome extraction with conditional corrections on the blocks
        it touched."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import ClassicalRegister
        from qiskit_aer.noise import depolarizing_error
        n = logical.num_qubits
        anc = 7 * n
        enc = QuantumCircuit(7 * n + 1)
        err1 = depolarizing_error(p, 1).to_instruction() if p > 0 else None
        err2 = depolarizing_error(p, 2).to_instruction() if p > 0 else None

        def block(l: int) -> list[int]:
            return list(range(7 * l, 7 * l + 7))

        def gate1(name: str, qubits: list[int]) -> None:
            for q in qubits:
                getattr(enc, name)(q)
                if err1 is not None:
                    enc.append(err1, [q])

        def gate2(name: str, a: int, b: int) -> None:
            for qa, qb in zip(block(a), block(b)):
                getattr(enc, name)(qa, qb)
                if err2 is not None:
                    enc.append(err2, [qa, qb])

        # |0_L> on every block: one pivot per check, fanned out over the check.
        for l in range(n):
            b = block(l)
            for pivot, check in zip((3, 1, 0), STEANE_CHECKS):
                enc.h(b[pivot])
                for i in check:
                    if i != pivot:
                        enc.cx(b[pivot], b[i])

        rounds = 0

        def syndrome_round(l: int) -> None:
            nonlocal rounds
            b = block(l)
            for kind in ('Z', 'X'):     # Z checks find X flips, X checks find Z flips
                cr = ClassicalRegister(3, f's{kind}{l}r{rounds}')
                enc.add_register(cr)
                for k, check in enumerate(STEANE_CHECKS):
                    if kind == 'Z':
                        for i in check:
                            enc.cx(b[i], anc)
                    else:
                        enc.h(anc)
                        for i in check:
                            enc.cx(anc, b[i])
                        enc.h(anc)
                    enc.measure(anc, cr[2 - k])
                    enc.reset(anc)
                for v in range(1, 8):
                    with enc.if_test((cr, v)):
                        (enc.x if kind == 'Z' else enc.z)(b[v - 1])
            rounds += 1

        for inst in logical.data:
            name = inst.operation.name
            qs = [logical.find_bit(q).index for q in inst.qubits]
            if name in ('barrier', 'measure'):
                continue
            if name in _STEANE_1Q:
                for g in _STEANE_1Q[name]:
                    gate1(g, block(qs[0]))
                touched = qs
            elif name in _STEANE_2Q:
                gate2(name, qs[0], qs[1])
                touched = qs
            elif name == 'cy':      # CY = S_t . CX . S_t^dag
                gate1('s', block(qs[1]))
                gate2('cx', qs[0], qs[1])
                gate1('sdg', block(qs[1]))
                touched = qs
            else:
                raise ValueError(
                    f"'{name.upper()}' has no transversal Steane form; CODE STEANE runs "
                    f"Clifford programs (H, S, SDG, SX, X, Y, Z, CX, CY, CZ, SWAP). "
                    f"T and rotations need magic states; CODE SURFACE models them")
            for l in touched:
                syndrome_round(l)
        return enc, rounds

    def _run_steane(self, t0: float) -> None:
        """RUN under LQUBITS ... CODE STEANE: the encoded simulation."""
        import time

        from qiskit.circuit import ClassicalRegister
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import pauli_error
        lm = self._logical_mode
        if lm is None:
            return
        p = lm['p']
        try:
            logical, _ = self.build_circuit()
        except Exception as e:
            self.io.writeln(f"?BUILD ERROR: {e}")
            return
        if self._classical_bits:
            self.io.writeln("?CODE STEANE measures logical qubits once, at the end; "
                            "use MEASURE rather than MEAS")
            return
        n = logical.num_qubits
        targets = list(self._measure_subset) if self._measure_subset else list(range(n))
        try:
            enc, rounds = self._steane_encode(logical, p)
        except ValueError as e:
            self.io.writeln(f"?{e}")
            return
        out = ClassicalRegister(7 * len(targets), 'out')
        enc.add_register(out)
        flip = pauli_error([('X', p), ('I', 1 - p)]).to_instruction() if p > 0 else None
        for j, l in enumerate(targets):
            for i in range(7):
                if flip is not None:
                    enc.append(flip, [7 * l + i])
                enc.measure(7 * l + i, out[7 * j + i])
        kw: dict = {'shots': self.shots}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        raw = AerSimulator(method='stabilizer').run(enc, **kw).result().get_counts()
        counts: dict[str, int] = {}
        width = 7 * len(targets)
        for key, c in raw.items():
            bits = key.split()[0][::-1]         # 'out' was added last; bits[i] = clbit i
            assert len(bits) == width
            vals = [steane_decode(bits[7 * j:7 * j + 7]) for j in range(len(targets))]
            lkey = ''.join(str(v) for v in reversed(vals))
            counts[lkey] = counts.get(lkey, 0) + c
        self.last_counts = counts
        self.last_sv = None
        self.last_circuit = logical
        dt = time.time() - t0
        self.io.writeln(f"\nRAN {len(self.program)} lines, {n} logical qubits "
                        f"(Steane [[7,1,3]]), {self.shots} shots in {dt:.2f}s")
        self.io.writeln(f"  encoded: {7 * n + 1} physical qubits, {rounds} syndrome "
                        f"round(s), depolarizing p={p:g} after every physical gate "
                        f"and on readout")
        ideal = self._ideal_distribution(logical, targets)
        if ideal is not None:
            tot = sum(counts.values())
            keys = set(ideal) | set(counts)
            tvd = 0.5 * sum(abs(counts.get(k, 0) / tot - ideal.get(k, 0.0)) for k in keys)
            self.variables['_LOGICAL_TVD'] = tvd
            self.io.writeln(f"  decoded readout vs ideal: total variation distance {tvd:.4f}")
        self.print_histogram(counts)

    @staticmethod
    def _ideal_distribution(logical: QuantumCircuit, targets: list[int]) -> dict[str, float] | None:
        """Noiseless outcome distribution over `targets` (qubit 0 rightmost)."""
        if logical.num_qubits > 16:
            return None
        from qiskit.quantum_info import Statevector
        clean = logical.remove_final_measurements(inplace=False)
        probs = Statevector(clean).probabilities_dict(qargs=targets)
        return {k: float(v) for k, v in probs.items() if v > 1e-12}
