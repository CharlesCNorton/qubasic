"""QUBASIC encoded logical qubits simulated by Pauli frames.

`LQUBITS n CODE SURFACE d PHYS p` (also REP and SHOR) runs a Clifford
program encoded: each logical qubit is a patch of the code, and every
logical gate is a transversal physical gate on the patches it touches:

    X, Y, Z    on every qubit of the patch
    H          (surface code) on every qubit, then the patch is relabelled
               by a quarter turn, which takes the code back to its own layout
    CX         qubit k of the control patch to qubit k of the target patch
    CZ         H, CX, H on the target (surface code)
    SWAP       exchanges which patch is which (no physical gate)

Depolarizing noise at the physical rate p follows every physical gate
(probability 3p/4 of an X, Y or Z on each qubit of a one-qubit layer, 15p/16
of a non-identity pair on each pair of a CX layer), and an X flip at rate p
precedes each readout. A round of ideal syndrome extraction follows every
logical gate on the patches it touched, and the decoder (minimum-weight
matching for the surface and repetition codes, the lookup table for Shor's
code) updates the Pauli frame rather than the qubits. The readout's
syndrome is decoded once more, and a logical qubit's outcome flips when the
residual X error crosses its logical Z.

Because every operation is a Clifford gate and every error a Pauli, the
errors and the ideal computation separate exactly: the ideal outcomes are
sampled from the logical circuit (Aer's stabilizer simulator) and each shot's
outcome is flipped by the logical error its frame accumulated.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from qubasic_core.host import HostState

# Logical gates each code implements transversally (qiskit gate names).
_TRANSVERSAL = {
    'SURFACE': {'id', 'x', 'y', 'z', 'h', 'cx', 'cz', 'swap'},
    'REP': {'id', 'x', 'y', 'z', 'cx', 'swap'},
    'SHOR': {'id', 'x', 'y', 'z', 'cx', 'swap'},
}


class FrameSimulation:
    """Vectorized Pauli-frame simulation of an encoded Clifford circuit."""

    def __init__(self, code: dict, n_logical: int, p: float, shots: int, rng: Any,
                 syndromes: Callable, decode: Callable, h_perm: list[int] | None):
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

    def _encodable(self, logical, code_name: str) -> str | None:
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
        from qubasic_core.surface import h_relabel
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

    def _ideal_samples(self, logical, targets: list[int], shots: int) -> np.ndarray:
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
