"""QUBASIC noise model configuration.

A noise model is one or more channels joined by ';':

    NOISE depolarizing 0.001 0.01; readout 0.02 0.03

Gate channels follow every gate they name: depolarizing has its own
two-qubit rate (an n-qubit gate takes the n-qubit depolarizing channel), and
the single-qubit channels (amplitude damping, dephasing, thermal relaxation,
Pauli, reset) act on each qubit a multi-qubit gate touches. Channels on the
same gate compose in the order given, and readout errors compose as
successive flips. NOISE ADD appends a channel to the active model; SAVE
writes the whole model back as one NOISE line.
"""

from __future__ import annotations

import re

import numpy as np

from qubasic_core.host import HostState

_1Q = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'sx', 'rx', 'ry', 'rz', 'p', 'u', 'id']
_2Q = ['cx', 'cy', 'cz', 'ch', 'swap', 'dcx', 'iswap', 'crx', 'cry', 'crz', 'cp',
       'rxx', 'ryy', 'rzz']
_3Q = ['ccx', 'cswap']

NOISE_TYPES = ('depolarizing', 'amplitude_damping', 'phase_flip', 'thermal', 'readout',
               'combined', 'pauli', 'reset', 'crosstalk')

# Channels the numpy LOCC engine reproduces (one channel, its first parameter).
_LOCC_TYPES = frozenset({'depolarizing', 'amplitude_damping', 'phase_flip'})


def _floats(params: list[str], defaults: tuple[float, ...]) -> list[float]:
    vals = [float(p) for p in params[:len(defaults)]]
    return vals + list(defaults[len(vals):])


def _per_qubit(nm, err) -> None:
    """A single-qubit channel after every gate, on each qubit the gate touches."""
    nm.add_all_qubit_quantum_error(err, _1Q, warnings=False)
    two = err.tensor(err)
    nm.add_all_qubit_quantum_error(two, _2Q, warnings=False)
    nm.add_all_qubit_quantum_error(two.tensor(err), _3Q, warnings=False)


def add_channel(nm, ntype: str, params: list[str], readout: list) -> str:
    """Add one channel to NoiseModel `nm` and return its description. A
    readout channel is collected in `readout` (applied once, composed)."""
    from qiskit_aer.noise import (
        depolarizing_error, amplitude_damping_error, phase_damping_error,
        thermal_relaxation_error, phase_amplitude_damping_error, pauli_error, reset_error,
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


class NoiseMixin(HostState):
    """NOISE: the active noise model (see the module docstring)."""

    def cmd_noise(self, rest: str) -> None:
        """NOISE <type> <params> [; <type> <params> ...] | ADD <type> <params> | INFO | OFF — set the noise model.

        Types:
          depolarizing <p1> [p2]        p1 after 1-qubit gates, p2 (default p1) after 2- and 3-qubit gates
          amplitude_damping <p>         T1-like decay
          phase_flip <p>                T2-like dephasing
          thermal <T1> <T2> <tg> [tg2]  thermal relaxation over gate times tg (1-qubit), tg2 (multi-qubit)
          readout <p01> [p10]           measurement flips p(0->1), p(1->0) (default p01)
          combined <p_amp> <p_phase>    amplitude + phase damping
          pauli <px> <py> <pz>          Pauli channel
          reset <p0> <p1>               spontaneous reset to |0> or |1>
          crosstalk <p>                 correlated ZZ after 2-qubit gates
        The single-qubit channels act on every qubit a multi-qubit gate touches;
        times take any one unit. Channels joined by ';' compose."""
        arg = rest.strip()
        if arg.upper() == 'INFO':
            self._noise_info()
            return
        if not arg or arg.upper() == 'OFF':
            self._noise_model = None
            self._noise_depol_p = 0.0
            self._noise_locc_type = 'none'
            self._noise_locc_param = 0.0
            self._noise_spec = None
            self._device_spec = None
            if getattr(self, 'locc', None) is not None:
                self.locc.noise_param = 0.0
                self.locc.noise_type = 'none'
            self.io.writeln("NOISE OFF")
            return
        m = re.match(r'ADD\s+(.+)$', arg, re.IGNORECASE)
        if m:
            spec = f"{self._noise_spec}; {m.group(1).strip()}" if self._noise_spec else m.group(1).strip()
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
        self._noise_model = nm
        self._noise_spec = '; '.join(components)
        self._device_spec = None
        self._circuit_cache_key = None
        self._noise_depol_p = next((float(p[0]) if p else 0.01
                                    for k, p in kinds if k == 'depolarizing'), 0.0)
        if len(kinds) == 1 and kinds[0][0] in _LOCC_TYPES:
            self._noise_locc_type = kinds[0][0]
            self._noise_locc_param = float(kinds[0][1][0]) if kinds[0][1] else 0.01
        else:
            self._noise_locc_type = 'none'
            self._noise_locc_param = 0.0
        if getattr(self, 'locc', None) is not None:
            self.locc.noise_type = self._noise_locc_type
            self.locc.noise_param = self._noise_locc_param
        for line in lines:
            self.io.writeln(f"NOISE {line}")

    def _noise_info(self) -> None:
        if self._noise_model is None:
            self.io.writeln("  NOISE OFF (no noise model active)")
            return
        self.io.writeln("  Noise model active:")
        if self._noise_spec:
            for comp in self._noise_spec.split(';'):
                self.io.writeln(f"    {comp.strip()}")
        elif getattr(self, '_device_spec', None):
            self.io.writeln(f"    DEVICE {self._device_spec}")
        self.io.writeln(f"    depol_p = {self._noise_depol_p}")
        for line in str(self._noise_model).split('\n'):
            self.io.writeln(f"    {line}")
        if getattr(self, 'locc', None) is not None and self.locc.noise_param > 0:
            self.io.writeln(f"    LOCC engine: {self.locc.noise_type} p={self.locc.noise_param}")
