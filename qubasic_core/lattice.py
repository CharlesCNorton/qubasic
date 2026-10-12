"""QUBASIC lattice surgery on rotated surface-code patches.

    LATTICE <a> <b> [ZZ|XX] [D d] [SHOTS n]   merge and split two patches
    LATTICE CNOT <c> <t> [D d] [SHOTS n]      a logical CNOT by lattice surgery
    LATTICE RATE <d[,d...]> <p> [ROUNDS r] [TRIALS n] [ZZ|XX]   noisy merge

The first two simulate the physical circuit exactly on Aer's stabilizer
simulator. Each patch is prepared in its logical state (a Clifford circuit
synthesized from the code's stabilizers and the logical operator), a merge
initializes the line of qubits between two patches (|+> for ZZ, |0> for XX)
and measures every check of the merged patch through an ancilla, and a split
measures that line again (X for ZZ, Z for XX). The merge outcome is the
product of the merged checks that multiply to the joint logical operator,
read from the same round. A merge reports the outcome distribution and how
often the logical readout of the split patches agrees with it; CNOT runs
M_ZZ(C, A), M_XX(A, T) and M_Z(A) with the ancilla patch A in |+>, applies the
Pauli frame (an X on T when the ZZ outcome, the Z outcome of A and the split
outcome on the shared row disagree in parity) and reports the truth table
read from the control and target patches.

RATE estimates how often a noisy ZZ (or XX) merge decodes to the wrong
outcome. Under phenomenological noise every data qubit of the merged patch
takes an error with probability p before each of the r rounds (default d)
and every check outcome flips with probability p; the final readout of the
two patches is noisy too. Detectors compare consecutive rounds (the new
checks have no value before the merge), and the space-time syndrome is
decoded by exact minimum-weight matching (dem.py). A failure is a wrong merge
outcome or a logical error on either patch.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from qubasic_core import surface as S
from qubasic_core.host import HostState

_STATES = ('0', '1', '+', '-')


def _logical_prep(patch: S.Patch, state: str):
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


def _measure_checks(qc, checks: list[list[int]], kinds: list[str], anc: int, creg) -> None:
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

    def _lattice_run(self, qc, shots: int) -> list[list[str]]:
        """Per-shot register words, in register creation order."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method='stabilizer')
        kw: dict = {'shots': shots, 'memory': True}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        mem = sim.run(transpile(qc, sim), **kw).result().get_memory()
        return [list(reversed(word.split())) for word in mem]

    def _merge_circuit(self, sg: S.Surgery, a: str, b: str):
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

    def _lattice_model(self, d: int, p: float, rounds: int, kind: str):
        """The phenomenological detector model of a noisy merge: observable 0
        is the merge outcome, 1 and 2 the logical readouts of A and B."""
        from qubasic_core.dem import DetectorModel
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
        """LATTICE <a> <b> [ZZ|XX] [D d] | CNOT <c> <t> [D d] | RATE <d,...> <p> [ROUNDS r] — lattice surgery.

        Merges and splits rotated surface-code patches in an exact stabilizer
        simulation, runs a lattice-surgery CNOT, or estimates the logical
        failure rate of a noisy merge under phenomenological noise with
        space-time matching. See qubasic_core/lattice.py."""
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
