"""QUBASIC logical-qubit mode — program at the logical level of a QEC code.

  LQUBITS 2 CODE STEANE PHYS 1e-3     2 logical qubits, encoded Steane simulation
  LQUBITS 2 CODE SURFACE 5            2 logical qubits, rotated surface d=5
  LQUBITS 2 CODE SURFACE 5 PHYS 1e-3  ...at physical error rate p
  LQUBITS 2 CODE SURFACE 5 MODEL      the modeled logical channel only
  LQUBITS STATUS                      Show the active logical configuration
  LQUBITS OFF                         Back to physical qubits

While active, program gates address LOGICAL qubits and the histogram is over
logical outcomes.

CODE STEANE runs the program encoded as a circuit: each logical qubit is a
[[7,1,3]] block of 7 physical qubits, logical Clifford gates are transversal,
depolarizing noise at the physical rate follows every physical gate and
readout, a round of syndrome extraction (ideal, with conditional
corrections) follows every logical gate, and readout is decoded block by
block. The report compares the decoded histogram with the ideal one.

CODE SURFACE, REP and SHOR run a program of transversal gates encoded too,
by Pauli frames with a decoder after every logical gate (encoded.py). A
program with other gates (T, rotations), and every program under MODEL, runs
on the modeled code instead: every operation carries the code's per-op
logical error channel (X/Z flips at the code-capacity logical rate for the
chosen code, distance, and physical p), measurement carries a logical
readout flip, and RUN appends a lattice-surgery compilation report (surgery
ops, syndrome rounds, physical qubits, wall time, error budget). Other
commands (SAMPLE, SWEEP, ...) see the modeled channel.
"""

from __future__ import annotations

from math import comb

import numpy as np
from qubasic_core.host import HostState

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
    from QECMixin.
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
            from qubasic_core.resources import surface_logical_rate
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
        """LQUBITS <n> CODE <name> [d] [PHYS p] [MODEL|ENCODED] | STATUS | OFF — logical-qubit mode.

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

            from qiskit_aer.noise import NoiseModel, pauli_error, ReadoutError
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

    def _steane_encode(self, logical, p: float) -> tuple:
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

        def block(l):
            return list(range(7 * l, 7 * l + 7))

        def gate1(name, qubits):
            for q in qubits:
                getattr(enc, name)(q)
                if err1 is not None:
                    enc.append(err1, [q])

        def gate2(name, a, b):
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

        def syndrome_round(l):
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
    def _ideal_distribution(logical, targets: list[int]) -> dict[str, float] | None:
        """Noiseless outcome distribution over `targets` (qubit 0 rightmost)."""
        if logical.num_qubits > 16:
            return None
        from qiskit.quantum_info import Statevector
        clean = logical.remove_final_measurements(inplace=False)
        probs = Statevector(clean).probabilities_dict(qargs=targets)
        return {k: float(v) for k, v in probs.items() if v > 1e-12}
