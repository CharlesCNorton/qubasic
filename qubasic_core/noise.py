"""QUBASIC noise models (NOISE) and error mitigation (MITIGATE, ZNE)."""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from qubasic_core.host import SimulationHost

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer.noise import NoiseModel, QuantumError


# ---------------------------------------------------------------------------
# QUBASIC noise model configuration.
#
# A noise model is one or more channels joined by ';':
#
#     NOISE depolarizing 0.001 0.01; readout 0.02 0.03
#
# Gate channels follow every gate they name: depolarizing has its own
# two-qubit rate (an n-qubit gate takes the n-qubit depolarizing channel), and
# the single-qubit channels (amplitude damping, dephasing, thermal relaxation,
# Pauli, reset) act on each qubit a multi-qubit gate touches. Channels on the
# same gate compose in the order given, and readout errors compose as
# successive flips. NOISE ADD appends a channel to the active model; SAVE
# writes the whole model back as one NOISE line.
# ---------------------------------------------------------------------------

_1Q = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'sx', 'sxdg', 'rx', 'ry', 'rz', 'p', 'u', 'id']
_2Q = ['cx', 'cy', 'cz', 'ch', 'swap', 'dcx', 'iswap', 'ecr', 'cs', 'csdg', 'csx', 'crx',
       'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz', 'rzx']
_3Q = ['ccx', 'cswap', 'ccz']

NOISE_TYPES = ('depolarizing', 'amplitude_damping', 'phase_flip', 'thermal', 'readout',
               'combined', 'pauli', 'lindblad', 'reset', 'crosstalk')

# Channels the numpy LOCC engine reproduces (one channel, its first parameter).
_LOCC_TYPES = frozenset({'depolarizing', 'amplitude_damping', 'phase_flip'})


def _floats(params: list[str], defaults: tuple[float, ...]) -> list[float]:
    vals = [float(p) for p in params[:len(defaults)]]
    return vals + list(defaults[len(vals):])


def _per_qubit(nm: NoiseModel, err: QuantumError) -> None:
    """A single-qubit channel after every gate, on each qubit the gate touches."""
    nm.add_all_qubit_quantum_error(err, _1Q, warnings=False)
    two = err.tensor(err)
    nm.add_all_qubit_quantum_error(two, _2Q, warnings=False)
    nm.add_all_qubit_quantum_error(two.tensor(err), _3Q, warnings=False)


def add_channel(nm: NoiseModel, ntype: str, params: list[str], readout: list) -> str:
    """Add one channel to NoiseModel `nm` and return its description. A
    readout channel is collected in `readout` (applied once, composed)."""
    from qiskit_aer.noise import (
        amplitude_damping_error,
        depolarizing_error,
        pauli_error,
        phase_amplitude_damping_error,
        phase_damping_error,
        reset_error,
        thermal_relaxation_error,
    )
    if ntype == 'depolarizing':
        p1, p2 = _floats(params, (0.01, float('nan')))
        if p2 != p2:                                   # one rate: every gate width
            p2 = p1
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), _1Q, warnings=False)
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), _2Q, warnings=False)
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 3), _3Q, warnings=False)
        return f"depolarizing p1={p1:g} p2={p2:g}"
    if ntype == 'amplitude_damping':
        (p,) = _floats(params, (0.01,))
        _per_qubit(nm, amplitude_damping_error(p))
        return f"amplitude_damping p={p:g}"
    if ntype == 'phase_flip':
        (p,) = _floats(params, (0.01,))
        _per_qubit(nm, phase_damping_error(p))
        return f"phase_flip p={p:g}"
    if ntype == 'thermal':
        t1, t2, tg, tg2 = _floats(params, (50e-6, 70e-6, 1e-6, float('nan')))
        if tg2 != tg2:
            tg2 = tg
        if t2 > 2 * t1:
            raise ValueError(f"thermal needs T2 <= 2 T1 (T1={t1:g}, T2={t2:g})")
        one = thermal_relaxation_error(t1, t2, tg)
        two = thermal_relaxation_error(t1, t2, tg2)
        nm.add_all_qubit_quantum_error(one, _1Q, warnings=False)
        nm.add_all_qubit_quantum_error(two.tensor(two), _2Q, warnings=False)
        nm.add_all_qubit_quantum_error(two.tensor(two).tensor(two), _3Q, warnings=False)
        return f"thermal T1={t1:g} T2={t2:g} t_gate={tg:g} t_gate2={tg2:g}"
    if ntype == 'readout':
        p0, p1 = _floats(params, (0.05, float('nan')))
        if p1 != p1:
            p1 = p0
        readout.append(np.array([[1 - p0, p0], [p1, 1 - p1]]))
        return f"readout p(0->1)={p0:g} p(1->0)={p1:g}"
    if ntype == 'combined':
        pa, pp = _floats(params, (0.01, 0.01))
        _per_qubit(nm, phase_amplitude_damping_error(pa, pp))
        return f"combined amp={pa:g} phase={pp:g}"
    if ntype == 'pauli':
        px, py, pz = _floats(params, (0.01, 0.01, 0.01))
        if px + py + pz > 1:
            raise ValueError("pauli probabilities sum past 1")
        _per_qubit(nm, pauli_error([('X', px), ('Y', py), ('Z', pz), ('I', 1 - px - py - pz)]))
        return f"pauli px={px:g} py={py:g} pz={pz:g}"
    if ntype == 'lindblad':
        return _lindblad(nm, params)
    if ntype == 'reset':
        p0, p1 = _floats(params, (0.01, 0.01))
        _per_qubit(nm, reset_error(p0, p1))
        return f"reset p0={p0:g} p1={p1:g}"
    if ntype == 'crosstalk':
        (p,) = _floats(params, (0.01,))
        nm.add_all_qubit_quantum_error(pauli_error([('ZZ', p), ('II', 1 - p)]), _2Q,
                                       warnings=False)
        return f"crosstalk (correlated ZZ after 2-qubit gates) p={p:g}"
    raise ValueError(f"UNKNOWN NOISE TYPE: {ntype}")


def _lindblad(nm: NoiseModel, params: list[str]) -> str:
    """The Pauli-Lindblad channel exp(sum r_k (P_k rho P_k - rho)) of
    generator and rate pairs: single-qubit generators act after every gate
    on each qubit it touches, two-qubit ones (Qiskit order, the rightmost
    letter on the gate's first qubit) after every two-qubit gate."""
    from qiskit.quantum_info import PauliList
    from qiskit_aer.noise import PauliLindbladError
    if not params or len(params) % 2:
        raise ValueError("lindblad needs generator and rate pairs: lindblad X 0.01 ZZ 0.002")
    by_width: dict[int, tuple[list[str], list[float]]] = {}
    for label, rate in zip(params[0::2], params[1::2]):
        label = label.upper()
        if not label or set(label) - set('IXYZ') or len(label) > 2 or set(label) == {'I'}:
            raise ValueError(f"lindblad generator {label!r}: one or two of X, Y, Z (and I)")
        r = float(rate)
        if r < 0:
            raise ValueError(f"lindblad rate {r:g} is negative")
        gens, rates = by_width.setdefault(len(label), ([], []))
        gens.append(label)
        rates.append(r)
    if 1 in by_width:
        _per_qubit(nm, PauliLindbladError(PauliList(by_width[1][0]), by_width[1][1]))
    if 2 in by_width:
        nm.add_all_qubit_quantum_error(PauliLindbladError(PauliList(by_width[2][0]), by_width[2][1]),
                                       _2Q, warnings=False)
    terms = [f"{g}:{r:g}" for gens, rates in by_width.values() for g, r in zip(gens, rates)]
    return "lindblad " + ' '.join(terms)


# ---------------------------------------------------------------------------
# Defects (NOISE DEFECT): a two-level fluctuator on one qubit. While the
# defect is excited the qubit's frequency is higher by `khz`; the defect
# switches with time constant `tau_us` and is excited a fraction `p` of the
# time, so a Ramsey experiment beats and an echo undoes what the defect did
# unless it switched in between, which no Markov channel reproduces.
#
# A run carries each defect as one extra qubit after the circuit's own. It
# starts excited with probability p, and every DELAY of its qubit applies the
# exact joint channel of the qubit and the defect over that time: the
# defect's state is classical (its coherences are dropped), its populations
# follow the switching rates, and the qubit's coherence takes the telegraph
# characteristic function exp((Q + i Omega) t) between defect states, so the
# defect keeps its state from one DELAY to the next. Before transpiling, a
# labelled one-qubit barrier marks each DELAY of a defect's qubit; it moves
# with the qubit through layout and routing, and is replaced by the channel
# afterwards. Relaxation during the DELAY commutes with the defect's phase,
# so applying the channel after the delay is exact.
# ---------------------------------------------------------------------------

DEFECT_MARK = 'qb_defect'
_US = {'s': 1e6, 'ms': 1e3, 'us': 1.0, 'ns': 1e-3, 'ps': 1e-6}


@dataclass(frozen=True)
class Defect:
    """A two-level defect on one qubit (see the comment above)."""

    qubit: int
    khz: float
    tau_us: float
    p: float = 0.5


def defect_kraus(d: Defect, us: float) -> list[np.ndarray]:
    """Kraus operators of the exact channel on [qubit, defect] over `us`
    microseconds of idling, the qubit the low bit."""
    from scipy.linalg import expm
    rate = 1.0 / d.tau_us
    up, down = d.p * rate, (1 - d.p) * rate
    gen = np.array([[-up, down], [up, -down]])
    pop = expm(gen * us)
    coh = expm((gen + 1j * np.diag([0.0, 2 * math.pi * d.khz * 1e-3])) * us)
    ops = []
    for k in range(2):
        for j in range(2):
            e = pop[j, k]                              # P(defect ends in j | starts in k)
            if e < 1e-15:
                continue
            hop = np.zeros((2, 2))
            hop[j, k] = 1.0
            m = coh[j, k]
            ops.append(np.kron(hop, np.diag([math.sqrt(e), m / math.sqrt(e)])))
            rest = e - abs(m) ** 2 / e
            if rest > 1e-15:
                ops.append(np.kron(hop, np.diag([0.0, math.sqrt(rest)])))
    return ops


def mark_defects(qc: QuantumCircuit, qubits: set[int]) -> QuantumCircuit:
    """qc with a labelled one-qubit barrier after every DELAY of a qubit in
    `qubits`, naming the qubit and the delay in microseconds; qc itself when
    no such DELAY exists."""
    from qiskit.circuit import Barrier

    def idles(inst: Any) -> bool:
        return inst.operation.name == 'delay' and qc.find_bit(inst.qubits[0]).index in qubits
    if not any(idles(inst) for inst in qc.data):
        return qc
    out = qc.copy_empty_like()
    for inst in qc.data:
        out.append(inst)
        if idles(inst):
            op = inst.operation
            if op.unit not in _US:
                raise ValueError(f"a defect needs a DELAY in time units, not {op.unit}")
            label = f"{DEFECT_MARK} {qc.find_bit(inst.qubits[0]).index} {float(op.duration) * _US[op.unit]!r}"
            out.append(Barrier(1, label=label), [inst.qubits[0]])
    return out


def attach_defects(qc: QuantumCircuit, defects: Sequence[Defect]) -> QuantumCircuit:
    """Replace the defect marks of a transpiled circuit with the defects: one
    extra qubit each, excited with probability p, and at each mark the exact
    channel of the marked qubit and every defect on it."""
    from qiskit import QuantumRegister
    from qiskit.quantum_info import Kraus
    from qiskit_aer.noise import pauli_error

    def mark(inst: Any) -> str:
        op = inst.operation
        label = (op.label or '') if op.name == 'barrier' else ''
        return label if label.startswith(DEFECT_MARK) else ''
    if not any(mark(inst) for inst in qc.data):
        return qc
    env = QuantumRegister(len(defects), 'defect')
    out = qc.copy_empty_like()
    with warnings.catch_warnings():                    # Aer ignores the transpiled layout
        warnings.filterwarnings('ignore', message=r'Trying to add QuantumRegister', category=UserWarning)
        out.add_register(env)
    for i, d in enumerate(defects):
        out.append(pauli_error([('X', d.p), ('I', 1 - d.p)]).to_instruction(), [env[i]])
    channels: dict[tuple[int, float], Any] = {}
    for inst in qc.data:
        label = mark(inst)
        if not label:
            out.append(inst)
            continue
        _, q, us = label.split()
        for i, d in enumerate(defects):
            if d.qubit == int(q):
                key = (i, float(us))
                if key not in channels:
                    channels[key] = Kraus(defect_kraus(d, float(us))).to_instruction()
                out.append(channels[key], [inst.qubits[0], env[i]])
    return out


class NoiseMixin(SimulationHost):
    """NOISE: the active noise model (see the module docstring)."""

    def cmd_noise(self, rest: str) -> None:
        """NOISE <type> <params> [; <type> <params> ...] | ADD <type> <params> | DEFECT <q> <kHz> <tau_us> [p] | INFO | OFF — set, extend, list or clear the noise model.

        NOISE DEFECT q kHz tau_us [p] attaches a two-level defect to qubit q:
        while the qubit idles in a DELAY its frequency is kHz higher whenever
        the defect is excited, the defect switching with time constant tau_us
        and excited a fraction p (default 0.5) of the time. The defect keeps
        its state between DELAYs, so an echo undoes it unless it switched.
        NOISE DEFECT lists the defects and NOISE DEFECT OFF removes them; they
        stay through NOISE and DEVICE changes, and NOISE OFF clears them too.

        Types:
          depolarizing <p1> [p2]        p1 after 1-qubit gates, p2 (default p1) after 2- and 3-qubit gates
          amplitude_damping <p>         T1-like decay
          phase_flip <p>                T2-like dephasing
          thermal <T1> <T2> <tg> [tg2]  thermal relaxation over gate times tg (1-qubit), tg2 (multi-qubit)
          readout <p01> [p10]           measurement flips p(0->1), p(1->0) (default p01)
          combined <p_amp> <p_phase>    amplitude + phase damping
          pauli <px> <py> <pz>          Pauli channel
          reset <p0> <p1>               spontaneous reset to |0> or |1>
          lindblad <P> <r> [<P> <r> ...]  Pauli-Lindblad generators and rates
          crosstalk <p>                 correlated ZZ after 2-qubit gates
        The single-qubit channels act on every qubit a multi-qubit gate touches;
        times take any one unit. Channels joined by ';' compose."""
        arg = rest.strip()
        if arg.upper() == 'INFO':
            self._noise_info()
            return
        if not arg or arg.upper() == 'OFF':
            self.noise.model = None
            self.noise.depol_p = 0.0
            self.noise.locc_type = 'none'
            self.noise.locc_param = 0.0
            self.noise.spec = None
            self.noise.defects = ()
            self._device_spec = None
            if self.locc is not None:
                self.locc.noise_param = 0.0
                self.locc.noise_type = 'none'
            self.io.writeln("NOISE OFF")
            return
        m = re.match(r'DEFECT\b\s*(.*)$', arg, re.IGNORECASE)
        if m:
            try:
                self._noise_defect(m.group(1).strip())
            except ValueError as e:
                self.io.writeln(f"?NOISE ERROR: {e}")
            return
        m = re.match(r'ADD\s+(.+)$', arg, re.IGNORECASE)
        if m:
            spec = f"{self.noise.spec}; {m.group(1).strip()}" if self.noise.spec else m.group(1).strip()
        else:
            spec = arg
        components = [c.strip() for c in spec.split(';') if c.strip()]
        try:
            from qiskit_aer.noise import NoiseModel, ReadoutError
        except ImportError:
            self.io.writeln("?Noise model requires qiskit-aer noise module")
            return
        nm = NoiseModel()
        readout: list = []
        lines = []
        kinds = []
        try:
            for comp in components:
                parts = comp.split()
                kind = parts[0].lower()
                lines.append(add_channel(nm, kind, parts[1:], readout))
                kinds.append((kind, parts[1:]))
        except ValueError as e:
            msg = str(e)
            self.io.writeln(f"?{msg}" if msg.startswith('UNKNOWN') else f"?NOISE ERROR: {msg}")
            if msg.startswith('UNKNOWN'):
                self.io.writeln(f"  Types: {', '.join(NOISE_TYPES)}")
            return
        if readout:
            total = readout[0]
            for extra in readout[1:]:
                total = total @ extra
            nm.add_all_qubit_readout_error(ReadoutError(total), warnings=False)
        self.noise.model = nm
        self.noise.spec = '; '.join(components)
        self._device_spec = None
        self._circuit_cache_key = None
        self.noise.depol_p = next((float(p[0]) if p else 0.01
                                    for k, p in kinds if k == 'depolarizing'), 0.0)
        if len(kinds) == 1 and kinds[0][0] in _LOCC_TYPES:
            self.noise.locc_type = kinds[0][0]
            self.noise.locc_param = float(kinds[0][1][0]) if kinds[0][1] else 0.01
        else:
            self.noise.locc_type = 'none'
            self.noise.locc_param = 0.0
        if self.locc is not None:
            self.locc.noise_type = self.noise.locc_type
            self.locc.noise_param = self.noise.locc_param
        for line in lines:
            self.io.writeln(f"NOISE {line}")

    def _noise_defect(self, args: str) -> None:
        """NOISE DEFECT q kHz tau_us [p] | NOISE DEFECT OFF | NOISE DEFECT."""
        if not args:
            if not self.noise.defects:
                self.io.writeln("  no defects")
            for d in self.noise.defects:
                self.io.writeln(f"  DEFECT on qubit {d.qubit}: {d.khz:g} kHz while excited, "
                                f"switching time {d.tau_us:g} us, excited {d.p:g} of the time")
            return
        if args.upper() == 'OFF':
            self.noise.defects = ()
            self._circuit_cache_key = None
            self.io.writeln("NOISE DEFECT OFF")
            return
        parts = args.replace(',', ' ').split()
        if len(parts) not in (3, 4):
            raise ValueError("NOISE DEFECT <qubit> <kHz> <tau_us> [p]")
        q = self._resolve_qubit(parts[0])
        khz, tau = self.eval_expr(parts[1]), self.eval_expr(parts[2])
        p = self.eval_expr(parts[3]) if len(parts) == 4 else 0.5
        if tau <= 0:
            raise ValueError(f"a defect's switching time must be positive, not {tau:g}")
        if not 0 < p < 1:
            raise ValueError(f"a defect is excited a fraction of the time between 0 and 1, not {p:g}")
        d = Defect(q, float(khz), float(tau), float(p))
        self.noise.defects = self.noise.defects + (d,)
        self._circuit_cache_key = None
        self.io.writeln(f"NOISE DEFECT on qubit {q}: {d.khz:g} kHz while excited, "
                        f"switching time {d.tau_us:g} us, excited {d.p:g} of the time")

    def _noise_info(self) -> None:
        if self.noise.model is None and not self.noise.defects:
            self.io.writeln("  NOISE OFF (no noise model active)")
            return
        if self.noise.model is None:
            self._noise_defect('')
            return
        self.io.writeln("  Noise model active:")
        if self.noise.spec:
            for comp in self.noise.spec.split(';'):
                self.io.writeln(f"    {comp.strip()}")
        elif getattr(self, '_device_spec', None):
            self.io.writeln(f"    DEVICE {self._device_spec}")
        self.io.writeln(f"    depol_p = {self.noise.depol_p}")
        for line in str(self.noise.model).split('\n'):
            self.io.writeln(f"    {line}")
        if self.locc is not None and self.locc.noise_param > 0:
            self.io.writeln(f"    LOCC engine: {self.locc.noise_type} p={self.locc.noise_param}")
        if self.noise.defects:
            self._noise_defect('')


# ---------------------------------------------------------------------------
# QUBASIC error mitigation: readout correction and zero-noise extrapolation.
#
#     MITIGATE READOUT [ON|OFF] | STATUS
#     ZNE <HAMILTONIAN | pauli sum> [SCALES 1 3 5] [FIT LINEAR|QUADRATIC|RICHARDSON|EXP]
#
# MITIGATE READOUT calibrates each qubit's readout under the active noise
# model (prepare all |0>, then all |1>, and measure: P(1|0) and P(0|1) per
# qubit). While it is on, every RUN with MEASURE also shows the histogram
# with that readout error inverted (the tensor product of the per-qubit
# inverse matrices applied to the measured distribution, then projected back
# onto probabilities), and keeps it as `last_mitigated`.
#
# ZNE measures an observable on the program's circuit (which must end in its
# final MEASURE, or have none, with no mid-circuit measurement) with the noise
# amplified by gate folding: at odd scale lambda every gate g becomes
# g (g^dagger g)^((lambda - 1) / 2), so the circuit is the same operation with
# lambda times the gate noise. The noisy values are extrapolated to lambda = 0
# (a straight line by default; QUADRATIC, RICHARDSON through every point, or
# an exponential decay) and printed beside the noiseless value.
# ---------------------------------------------------------------------------

MAX_MITIGATED_QUBITS = 20


def correct_readout(counts: dict[str, int], errors: list[tuple[float, float]]) -> dict[str, float]:
    """The distribution behind `counts` (keys with qubit 0 rightmost) once the
    per-qubit readout errors (P(1|0), P(0|1)) are inverted."""
    n = len(errors)
    total = sum(counts.values())
    p = np.zeros(1 << n)
    for key, c in counts.items():
        p[int(key.replace(' ', ''), 2)] += c / total
    t = p.reshape([2] * n)
    for q, (e0, e1) in enumerate(errors):
        A = np.array([[1 - e0, e1], [e0, 1 - e1]])
        t = np.moveaxis(np.tensordot(np.linalg.inv(A), t, axes=([1], [n - 1 - q])), 0, n - 1 - q)
    q_vec = np.clip(t.reshape(-1), 0, None)
    q_vec /= q_vec.sum() or 1.0
    return {format(i, f'0{n}b'): float(v) for i, v in enumerate(q_vec) if v > 1e-12}


def fold_circuit(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """qc with every gate g replaced by g (g^dagger g)^((scale - 1) / 2)."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"fold scales are odd integers, not {scale}")
    out = qc.copy_empty_like()
    k = (scale - 1) // 2
    for inst in qc.data:
        op = inst.operation
        if op.name == 'barrier':
            out.append(op, inst.qubits, inst.clbits)
            continue
        if op.name in ('measure', 'reset') or getattr(op, 'condition', None) is not None:
            raise ValueError(f"ZNE folds gates only; the circuit has a {op.name}")
        out.append(op, inst.qubits, inst.clbits)
        for _ in range(k):
            out.append(op.inverse(), inst.qubits, inst.clbits)
            out.append(op, inst.qubits, inst.clbits)
    return out


def extrapolate(scales: Sequence[float], values: Sequence[float], fit: str) -> float:
    """The value at scale 0 of the chosen fit through (scales, values)."""
    x, y = np.asarray(scales, float), np.asarray(values, float)
    if fit == 'LINEAR':
        return float(np.polyval(np.polyfit(x, y, 1), 0.0))
    if fit == 'QUADRATIC':
        if len(x) < 3:
            raise ValueError("QUADRATIC needs three scales")
        return float(np.polyval(np.polyfit(x, y, 2), 0.0))
    if fit == 'RICHARDSON':
        total = 0.0
        for i in range(len(x)):
            w = np.prod([x[j] / (x[j] - x[i]) for j in range(len(x)) if j != i])
            total += w * y[i]
        return float(total)
    if fit == 'EXP':
        if np.all(y > 0) or np.all(y < 0):
            b, log_a = np.polyfit(x, np.log(np.abs(y)), 1)
            return float(np.sign(y[0]) * math.exp(log_a))
        return float(np.polyval(np.polyfit(x, y, 1), 0.0))
    raise ValueError(f"FIT is LINEAR, QUADRATIC, RICHARDSON or EXP, not {fit}")


@dataclass
class NoiseState:
    """The active noise: the Aer model and the NOISE arguments that built it,
    the depolarizing rate, the channel the LOCC engine applies, the per-qubit
    noise poked into the memory map, the NOISE DEFECT fluctuators, and
    readout calibration for MITIGATE."""

    model: NoiseModel | None = None
    spec: str | None = None                                                # the NOISE arguments, for SAVE
    depol_p: float = 0.0
    locc_type: str = 'none'                    # the LOCC engine's channel, or 'none'
    locc_param: float = 0.0
    # Poked at $D100 + 2*qubit: qubit -> (type, parameter), type 1 depolarizing,
    # 2 amplitude damping, 3 phase flip.
    per_qubit: dict[int, tuple[int, float]] = field(default_factory=dict)
    defects: tuple[Defect, ...] = ()
    readout: list[tuple[float, float]] | None = None                       # MITIGATE READOUT's calibration


class MitigationMixin(SimulationHost):
    """MITIGATE READOUT and ZNE."""

    def _init_mitigation(self) -> None:
        self.last_mitigated: dict[str, float] | None = None

    def cmd_mitigate(self, rest: str) -> None:
        """MITIGATE READOUT [ON|OFF] | STATUS — calibrate and invert readout error on every RUN."""
        words = rest.upper().split()
        if not words or words[0] == 'STATUS':
            if self.noise.readout is None:
                self.io.writeln("MITIGATE READOUT OFF")
            else:
                worst = max(max(e) for e in self.noise.readout)
                self.io.writeln(f"MITIGATE READOUT ON ({len(self.noise.readout)} qubits, "
                                f"worst flip probability {worst:.4f})")
            return
        if words[0] != 'READOUT' or (len(words) > 1 and words[1] not in ('ON', 'OFF')):
            self.io.writeln("?USAGE: MITIGATE READOUT [ON|OFF] | STATUS")
            return
        if len(words) > 1 and words[1] == 'OFF':
            self.noise.readout = None
            self.last_mitigated = None
            self.io.writeln("MITIGATE READOUT OFF")
            return
        n = self.num_qubits
        if n > MAX_MITIGATED_QUBITS:
            self.io.writeln(f"?MITIGATE READOUT corrects at most {MAX_MITIGATED_QUBITS} qubits")
            return
        try:
            self.noise.readout = self._calibrate_readout(n)
        except Exception as e:
            self.io.writeln(f"?MITIGATE ERROR: {e}")
            return
        flips = ', '.join(f"q{q} {e0:.3f}/{e1:.3f}" for q, (e0, e1) in enumerate(self.noise.readout[:6]))
        more = ' ...' if n > 6 else ''
        self.io.writeln(f"MITIGATE READOUT ON: P(1|0)/P(0|1) {flips}{more}")

    def _calibrate_readout(self, n: int, shots: int = 4000) -> list[tuple[float, float]]:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        backend = AerSimulator(noise_model=self.noise.model) if self.noise.model else AerSimulator()
        errors = []
        marginals = []
        for prep in (0, 1):
            qc = QuantumCircuit(n)
            if prep:
                qc.x(range(n))
            qc.measure_all()
            counts = backend.run(transpile(qc, backend), shots=shots,
                                 seed_simulator=self._seed).result().get_counts()
            ones = np.zeros(n)
            for key, c in counts.items():
                bits = key.replace(' ', '')
                for q in range(n):
                    if bits[n - 1 - q] == '1':
                        ones[q] += c
            marginals.append(ones / shots)
        for q in range(n):
            e0 = float(marginals[0][q])            # read 1 having prepared 0
            e1 = float(1 - marginals[1][q])        # read 0 having prepared 1
            if e0 + e1 >= 1:
                raise ValueError(f"qubit {q}'s readout carries no information")
            errors.append((e0, e1))
        return errors

    def _mitigate_after_run(self) -> None:
        """Show the readout-corrected histogram of the run just finished."""
        errors = self.noise.readout
        if errors is None or not self.last_counts:
            return
        key_len = len(next(iter(self.last_counts)).replace(' ', ''))
        if key_len != len(errors):
            self.io.writeln("  (readout mitigation skipped: recalibrate with MITIGATE READOUT ON "
                            "after changing QUBITS)")
            return
        self.last_mitigated = correct_readout(self.last_counts, errors)
        top = sorted(self.last_mitigated.items(), key=lambda kv: -kv[1])[:8]
        self.io.writeln("  readout-mitigated: " + '  '.join(
            f"|{self._bits(k)}> {v:.4f}" for k, v in top))

    # ── zero-noise extrapolation ─────────────────────────────────────────

    def _zne_observable(self, text: str) -> SparsePauliOp:
        name = text.strip().upper()
        if name in self._hamiltonians:
            return self._hamiltonians[name]
        return self._parse_pauli_sum(text)

    def cmd_zne(self, rest: str) -> None:
        """ZNE <HAMILTONIAN | pauli sum> [SCALES 1 3 5] [FIT LINEAR|QUADRATIC|RICHARDSON|EXP] — zero-noise extrapolation."""
        text = rest
        fit = 'LINEAR'
        m = re.search(r'\bFIT\s+(\w+)', text, re.IGNORECASE)
        if m:
            fit = m.group(1).upper()
            text = text[:m.start()] + text[m.end():]
        scales = [1, 3, 5]
        m = re.search(r'\bSCALES\s+([\d,\s]+)', text, re.IGNORECASE)
        if m:
            scales = [int(s) for s in re.split(r'[,\s]+', m.group(1).strip()) if s]
            text = text[:m.start()] + text[m.end():]
        if not text.strip():
            self.io.writeln("?USAGE: ZNE <HAMILTONIAN | pauli sum> [SCALES 1 3 5] [FIT LINEAR|QUADRATIC|RICHARDSON|EXP]")
            return
        try:
            if len(scales) < 2:
                raise ValueError("ZNE needs at least two scales")
            observable = self._zne_observable(text)
            if not self.program:
                raise ValueError("no program to run")
            qc, _ = self.build_circuit()
            base = qc.remove_final_measurements(inplace=False)
            if base.num_qubits != observable.num_qubits:
                if observable.num_qubits > base.num_qubits:
                    raise ValueError(f"the observable acts on {observable.num_qubits} qubits, "
                                     f"the program on {base.num_qubits}")
                observable = _identity(base.num_qubits - observable.num_qubits).tensor(observable)
            noisy = [self._noisy_expectation(fold_circuit(base, s), observable) for s in scales]
            ideal = self._noisy_expectation(base, observable, noise=False)
            estimate = extrapolate(scales, noisy, fit)
            self.io.writeln(f"\n  ZNE ({fit.lower()} fit) on {base.num_qubits} qubits, "
                            f"{'noise model' if self.noise.model else 'no noise model'}:")
            for s, v in zip(scales, noisy):
                self.io.writeln(f"    scale {s:<3} <O> = {v:+.6f}")
            self.io.writeln(f"    extrapolated  <O> = {estimate:+.6f}")
            self.io.writeln(f"    noiseless     <O> = {ideal:+.6f}   (error {estimate - ideal:+.2e}, "
                            f"unmitigated {noisy[0] - ideal:+.2e})")
            self.variables['_ZNE'] = estimate
            self.variables['_ZNE_NOISY'] = noisy[0]
            self.variables['_ZNE_IDEAL'] = ideal
        except Exception as e:
            self.io.writeln(f"?ZNE ERROR: {e}")

    def _noisy_expectation(self, qc: QuantumCircuit, observable: SparsePauliOp, noise: bool = True) -> float:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        model = self.noise.model if noise else None
        circ = qc.copy()
        circ.save_expectation_value(observable, list(range(circ.num_qubits)))
        if model is not None and circ.num_qubits <= 12:
            backend = AerSimulator(method='density_matrix', noise_model=model)
            shots: Any = 1
        else:
            backend = AerSimulator(method='statevector', noise_model=model)
            shots = 2000 if model is not None else 1
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=shots,
                             seed_simulator=self._seed).result()
        return float(np.real(result.data(0)['expectation_value']))


def _identity(n: int) -> SparsePauliOp:
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp('I' * n)
