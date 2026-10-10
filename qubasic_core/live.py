"""QUBASIC live execution: programs that read measurement results as numbers.

A program whose classical code reads a mid-circuit measurement bit (PRINT m,
LET y = m + 1, a GOTO or loop chosen by it) cannot be one Qiskit circuit,
because the value differs from shot to shot. Such a program runs live: each
shot executes the program from the top on a numpy statevector, MEAS
collapses the state and returns its outcome, and IF, PRINT and LET see real
bits. Gates the executor appends to the shot's circuit are applied to the
state when a measurement or a state read needs them. An active noise model
runs as quantum trajectories (gate errors as Kraus jumps, readout error as
classical flips).

Only the first shot prints. Later shots replay the first shot's INPUT
answers and skip file output, so a run reads as one execution of the
program followed by the histogram over every shot.
"""

from __future__ import annotations

import re
import sys
import time
from typing import Any

import numpy as np

from qubasic_core.errors import QBasicBuildError
from qubasic_core.gates import (
    GATE_ALIASES, _GATE_BUILDERS, _np_gate_matrix, _apply_gate_inplace,
    _measure_inplace, _sample_counts,
)


class MidCircuitRead(ValueError):
    """A mid-circuit measurement bit was read as a number during a circuit
    build; the program runs live instead."""


# ── IO wrappers ─────────────────────────────────────────────────────────

class ShotIO:
    """IO for shot-by-shot execution. The first shot writes through and
    records every line read; after start_replay() output is dropped and the
    recorded lines are returned in order. `answers` and `skip` carry what an
    abandoned circuit build already read and showed."""

    def __init__(self, io: Any, answers: list[str] | None = None, skip: int = 0):
        self._io = io
        self.answers: list[str] = list(answers or [])
        self._pos = 0
        self._skip = skip
        self.replay = False

    def start_replay(self) -> None:
        self.replay = True
        self._pos = 0

    def _emit(self, text: str, newline: bool) -> None:
        if self.replay:
            return
        if self._skip:
            full = text + ('\n' if newline else '')
            if len(full) <= self._skip:
                self._skip -= len(full)
                return
            text, newline = full[self._skip:], False
            self._skip = 0
        if newline:
            self._io.writeln(text)
        else:
            self._io.write(text)

    def write(self, text: str) -> None:
        self._emit(text, False)

    def writeln(self, text: str) -> None:
        self._emit(text, True)

    def read_line(self, prompt: str) -> str:
        if self._pos < len(self.answers):
            self._pos += 1
            return self.answers[self._pos - 1]
        if self.replay:
            raise EOFError
        answer = self._io.read_line(prompt)
        self.record(answer)
        return answer

    def record(self, answer: str) -> None:
        """Record input read past this IO (a keypress from GET)."""
        self.answers.append(answer)
        self._pos += 1


class BufferedIO:
    """Holds a circuit build's output until the build finishes, so a build
    abandoned for a live run prints nothing twice. Reading input shows what
    is held first; `shown` and `answers` record what reached the user."""

    def __init__(self, io: Any):
        self._io = io
        self._held: list[tuple[str, bool]] = []
        self.shown = 0
        self.answers: list[str] = []

    def write(self, text: str) -> None:
        self._held.append((text, False))

    def writeln(self, text: str) -> None:
        self._held.append((text, True))

    def flush(self) -> None:
        for text, newline in self._held:
            if newline:
                self._io.writeln(text)
            else:
                self._io.write(text)
            self.shown += len(text) + newline
        self._held.clear()

    def discard(self) -> None:
        self._held.clear()

    def read_line(self, prompt: str) -> str:
        self.flush()
        answer = self._io.read_line(prompt)
        self.answers.append(answer)
        return answer

    def record(self, answer: str) -> None:
        self.flush()
        self.answers.append(answer)


# ── Noise as trajectories ───────────────────────────────────────────────

class _Channel:
    """A quantum channel ready for trajectory sampling. A mixed-unitary
    channel (Pauli, depolarizing) draws its branch with fixed probabilities;
    a general one weighs each Kraus operator by its probability on the
    current state. Operators are Qiskit-ordered over the channel's qubits."""

    def __init__(self, error: Any):
        from qiskit.quantum_info import Kraus
        ops = [np.asarray(k, dtype=complex) for k in Kraus(error).data]
        self.n = int(round(np.log2(ops[0].shape[0])))
        d = ops[0].shape[0]
        eye = np.eye(d)
        weights, unitaries = [], []
        mixed = True
        for k in ops:
            g = k.conj().T @ k
            c = float(g[0, 0].real)
            if c < 1e-15:
                continue
            if not np.allclose(g, c * eye, atol=1e-10):
                mixed = False
                break
            weights.append(c)
            unitaries.append(k / np.sqrt(c))
        self.mixed = mixed
        if mixed:
            w = np.array(weights) / sum(weights)
            self.cum = np.cumsum(w)
            self.ops = unitaries
            self.trivial = [bool(np.allclose(u, u[0, 0] * eye)) for u in unitaries]
        else:
            self.ops = ops


class LiveNoise:
    """An Aer noise model as trajectory channels: per instruction name and
    qubits (qubit-specific errors override all-qubit ones, as in Aer), plus
    single-qubit readout errors."""

    def __init__(self, model: Any):
        self._default = {name: _Channel(err) for name, err in
                         getattr(model, '_default_quantum_errors', {}).items()}
        self._local = {name: {tuple(q): _Channel(err) for q, err in per.items()}
                       for name, per in getattr(model, '_local_quantum_errors', {}).items()}
        self._ro_default = self._readout(getattr(model, '_default_readout_error', None))
        self._ro_local = {tuple(q): self._readout(e) for q, e in
                          getattr(model, '_local_readout_errors', {}).items()}

    @staticmethod
    def _readout(err: Any) -> np.ndarray | None:
        if err is None:
            return None
        p = np.asarray(err.probabilities, dtype=float)
        return p if p.shape == (2, 2) else None

    def channel(self, name: str, qubits: list[int]) -> _Channel | None:
        local = self._local.get(name)
        if local:
            ch = local.get(tuple(qubits))
            if ch is not None:
                return ch
        ch = self._default.get(name)
        if ch is not None and ch.n == len(qubits):
            return ch
        return None

    def readout(self, qubit: int) -> np.ndarray | None:
        return self._ro_local.get((qubit,), self._ro_default)


# ── The per-shot state ──────────────────────────────────────────────────

_CONTROL_FLOW_OPS = frozenset({'if_else', 'while_loop', 'for_loop', 'switch_case', 'box'})


def _numeric(params) -> bool:
    try:
        for p in params:
            float(p)
        return True
    except (TypeError, ValueError):
        return False


class LiveState:
    """The statevector of one live shot plus the circuit fragment the
    executor is appending to. Built-in gates apply at once; everything else
    queues in `qc` until flush(). Qubit q is bit q of the amplitude index."""

    def __init__(self, n: int, precision: str = 'double', noise: LiveNoise | None = None):
        from qiskit import QuantumCircuit
        self.n = n
        self.dtype = np.complex64 if precision == 'single' else np.complex128
        self._sv: np.ndarray = np.zeros(1 << n, dtype=self.dtype)
        self._sv[0] = 1.0
        self.qc = QuantumCircuit(n)
        self.noise = noise
        self.ops = 0

    def reset_state(self, init: np.ndarray | None = None) -> None:
        if init is None:
            self._sv[...] = 0
            self._sv[0] = 1.0
        else:
            np.copyto(self._sv, init)
        self.qc.clear()

    @property
    def sv(self) -> np.ndarray:
        self.flush()
        return self._sv

    def flush(self) -> None:
        """Apply the instructions queued in qc, in order, and clear it."""
        qc = self.qc
        if len(qc.data) == 0:
            return
        index = {bit: i for i, bit in enumerate(qc.qubits)}
        for inst in list(qc.data):
            self.apply_operation(inst.operation, [index[q] for q in inst.qubits])
        qc.clear()

    # Gates

    def apply_named(self, name: str, params: tuple, qubits: list[int]) -> None:
        """A built-in gate (GATE_TABLE name), MSB-first over `qubits`."""
        canonical = GATE_ALIASES.get(name, name)
        _apply_gate_inplace(self._sv, _np_gate_matrix(canonical, tuple(params)),
                            list(qubits), self.n)
        self.ops += 1
        self._noise_after(canonical.lower(), qubits)

    def apply_matrix(self, matrix: np.ndarray, qubits: list[int], name: str = 'unitary') -> None:
        """A unitary in Qiskit order over `qubits` (qubits[0] least significant)."""
        _apply_gate_inplace(self._sv, np.asarray(matrix, dtype=complex), list(qubits)[::-1], self.n)
        self.ops += 1
        self._noise_after(name, qubits)

    def apply_operation(self, op: Any, qubits: list[int]) -> None:
        name = op.name
        if name in ('barrier', 'delay'):
            return
        if name == 'measure':
            self.measure(qubits[0])
            return
        if name == 'reset':
            self.reset(qubits[0])
            return
        if name == 'set_statevector':
            if list(qubits) != list(range(self.n)):
                raise ValueError("SET_STATE in a live run must cover every qubit")
            np.copyto(self._sv, np.asarray(op.params[0], dtype=complex).reshape(-1))
            return
        if name == 'kraus':
            self._kraus([np.asarray(k, dtype=complex) for k in op.params], qubits)
            return
        if name in _CONTROL_FLOW_OPS or name.startswith('save_') or name == 'set_density_matrix':
            raise ValueError(f"'{name}' is not available in a live run")
        upper = name.upper()
        if upper in _GATE_BUILDERS and _numeric(op.params):
            self.apply_named(upper, tuple(float(p) for p in op.params), qubits)
            return
        if name == 'unitary' or (op.num_qubits <= 6 and hasattr(op, '__array__')):
            try:
                self.apply_matrix(op.to_matrix(), qubits, name)
                return
            except Exception:
                pass
        definition = getattr(op, 'definition', None)
        if definition is not None:
            sub = {bit: i for i, bit in enumerate(definition.qubits)}
            for inst in definition.data:
                self.apply_operation(inst.operation, [qubits[sub[q]] for q in inst.qubits])
            return
        from qiskit.quantum_info import Operator
        self.apply_matrix(Operator(op).data, qubits, name)

    # Non-unitary steps

    def _noise_after(self, name: str, qubits: list[int]) -> None:
        if self.noise is not None:
            ch = self.noise.channel(name, list(qubits))
            if ch is not None:
                self._apply_channel(ch, list(qubits))

    def _apply_channel(self, ch: _Channel, qubits: list[int]) -> None:
        if ch.mixed:
            i = int(np.searchsorted(ch.cum, np.random.random(), side='right'))
            i = min(i, len(ch.ops) - 1)
            if not ch.trivial[i]:
                _apply_gate_inplace(self._sv, ch.ops[i], qubits[::-1], self.n)
        else:
            self._kraus(ch.ops, qubits)

    def _kraus(self, ops: list[np.ndarray], qubits: list[int]) -> None:
        """One trajectory step: pick K_i with probability ||K_i psi||^2."""
        from qubasic_core.statetools import subsystem_rdm
        msb = list(qubits)[::-1]
        if len(ops) == 1:
            pick = 0
        else:
            rho = subsystem_rdm(self._sv, msb, self.n)
            probs = np.array([max(float(np.real(np.trace(k @ rho @ k.conj().T))), 0.0)
                              for k in ops])
            probs /= probs.sum()
            pick = int(np.searchsorted(np.cumsum(probs), np.random.random(), side='right'))
            pick = min(pick, len(ops) - 1)
        _apply_gate_inplace(self._sv, ops[pick], msb, self.n)
        norm = float(np.sqrt(np.vdot(self._sv, self._sv).real))
        if norm > 0:
            self._sv *= 1.0 / norm

    def measure(self, qubit: int) -> int:
        """Projective Z measurement; returns the recorded bit (readout error
        included)."""
        self.flush()
        self._noise_after('measure', [qubit])
        bit = _measure_inplace(self._sv, qubit, self.n)
        if self.noise is not None:
            p = self.noise.readout(qubit)
            if p is not None and np.random.random() < p[bit][1 - bit]:
                bit = 1 - bit
        return bit

    def reset(self, qubit: int) -> None:
        self.flush()
        if _measure_inplace(self._sv, qubit, self.n):
            _apply_gate_inplace(self._sv, _np_gate_matrix('X'), [qubit], self.n)
        self._noise_after('reset', [qubit])

    def sample_final(self, subset: list[int] | None) -> str:
        """The final MEASURE of a shot: one outcome over `subset` (or every
        qubit), as a bitstring with the last measured qubit leftmost."""
        self.flush()
        qubits = list(subset) if subset else list(range(self.n))
        if self.noise is not None:
            for q in qubits:
                self._noise_after('measure', [q])
        key = next(iter(_sample_counts(self._sv, self.n, 1)))
        bits = {q: int(key[self.n - 1 - q]) for q in qubits}
        if self.noise is not None:
            for q in qubits:
                p = self.noise.readout(q)
                if p is not None and np.random.random() < p[bits[q]][1 - bits[q]]:
                    bits[q] = 1 - bits[q]
        return ''.join(str(bits[q]) for q in reversed(qubits))


class LiveBackend:
    """Executor backend for a live shot: built-in gates go straight to the
    state, custom gates as their matrices, and other circuit operations
    queue in the shot's circuit."""

    def __init__(self, state: LiveState, custom_gates: dict[str, np.ndarray]):
        self.state = state
        self._custom = custom_gates

    @property
    def qc(self) -> Any:
        return self.state.qc

    def apply_gate(self, gate_name: str, params: tuple, qubits: list[int]) -> None:
        if len(set(qubits)) != len(qubits):
            raise QBasicBuildError(f"duplicate qubit arguments in {gate_name}")
        self.state.flush()
        canonical = GATE_ALIASES.get(gate_name, gate_name)
        if canonical in self._custom:
            self.state.apply_matrix(self._custom[canonical], list(qubits))
        else:
            self.state.apply_named(canonical, tuple(params), list(qubits))

    def barrier(self) -> None:
        pass

    def reset(self, qubit: int) -> None:
        self.state.reset(qubit)

    def measure(self, qubit: int, classical_bit: Any = None) -> int:
        return self.state.measure(qubit)

    def h(self, qubit: int) -> None:
        self.apply_gate('H', (), [qubit])

    def append_inverse(self, sub_qc: Any, qubits: list[int]) -> None:
        self.state.qc.append(sub_qc.inverse(), qubits)

    def append_controlled(self, gate: Any, qubits: list[int]) -> None:
        self.state.qc.append(gate, qubits)


# ── Deciding when a program needs to run live ───────────────────────────

_RE_BIT_TARGET = re.compile(r'\b(?:MEAS\s+\S+|SYNDROME\b[^:]*?)\s*->\s*(\w+)', re.IGNORECASE)
_RE_IDENT = re.compile(r'[A-Za-z_]\w*')
_RE_BASIS_BIT = re.compile(r'm[xyz]_\d+$')
_QUANTUM_CLAUSE_WORDS = frozenset({'CTRL', 'INV', 'RESET', 'BARRIER'})


def _identifiers(text: str) -> set[str]:
    """Identifiers outside string literals, without MEAS/SYNDROME targets."""
    text = _RE_BIT_TARGET.sub('', text)
    out: set[str] = set()
    for i, seg in enumerate(text.split('"')):
        if i % 2 == 0:
            out.update(_RE_IDENT.findall(seg))
    return out


class LiveMixin:
    """Live (shot-by-shot) execution for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.variables,
    self.arrays, self.shots, self.num_qubits, self.io, self.build_circuit(),
    self._program_has_measure(), self._noise_model, self._custom_gates.
    """

    _gosub_stack: list

    def _init_live(self) -> None:
        self._live: LiveState | None = None
        self._live_needed = False
        self._live_saves: dict[str, tuple[str, list]] = {}

    # Static check

    def _bit_names(self, texts: list[str]) -> tuple[set[str], bool]:
        names: set[str] = set()
        basis = False
        for t in texts:
            names.update(m.group(1) for m in _RE_BIT_TARGET.finditer(t))
            if re.search(r'\bMEASURE_[XYZ]\b', t, re.IGNORECASE):
                basis = True
        return names, basis

    def _needs_live(self, sorted_lines: list[int]) -> bool:
        """Whether the program reads a measurement bit anywhere a compiled
        dynamic circuit cannot: anything but IF <bit> THEN <gates> [ELSE
        <gates>] on one bit."""
        from qubasic_core.parser import parse_stmt
        from qubasic_core.statements import CompoundStmt, IfThenStmt
        from qubasic_core.executor import _feedforward_test
        texts = [self.program[ln] for ln in sorted_lines]
        names, basis = self._bit_names(texts)
        if not names and not basis:
            return False

        def is_bit(ident: str) -> bool:
            return ident in names or (basis and bool(_RE_BASIS_BIT.match(ident)))

        def reads(text: str) -> bool:
            return any(is_bit(i) for i in _identifiers(text))

        def quantum_only(clause: str | None) -> bool:
            if not clause:
                return True
            for part in self._split_colon_stmts(clause):
                word = part.split(None, 1)[0].upper().split('(')[0] if part.strip() else ''
                if not (word in _QUANTUM_CLAUSE_WORDS or self._gate_info(word) is not None
                        or word in self.subroutines):
                    return False
            return not reads(clause)

        def check(text: str) -> bool:
            p = parse_stmt(text)
            if isinstance(p, CompoundStmt):
                return any(check(part) for part in p.parts)
            if isinstance(p, IfThenStmt):
                if not reads(p.condition):
                    return (bool(p.then_clause) and check(p.then_clause)) or (
                        p.else_clause is not None and check(p.else_clause))
                bit_names = {i for i in _identifiers(p.condition) if is_bit(i)}
                return not (p.then_clause and _feedforward_test(p.condition, bit_names)
                            and quantum_only(p.then_clause)
                            and quantum_only(p.else_clause))
            return reads(text)

        return any(check(t) for t in texts)

    # Running

    def _live_noise(self) -> LiveNoise | None:
        model = self._noise_model
        if model is None and self._qubit_noise:
            model = self._build_qubit_noise()
        return LiveNoise(model) if model is not None else None

    def _live_prefix(self, sorted_lines: list[int]) -> int:
        """Number of leading lines that run the same in every shot: plain
        gates, assignments and PRINTs with no jumps, input, randomness or
        measurement. They run once and each shot starts after them."""
        from qubasic_core.statements import GateStmt, LetStmt, PrintStmt, RemStmt, BarrierStmt
        k = 0
        for ln in sorted_lines:
            p = self._get_parsed(ln)
            text = self.program[ln]
            if not isinstance(p, (GateStmt, LetStmt, PrintStmt, RemStmt, BarrierStmt)):
                break
            if re.search(r'\b(RND|TIMER|INKEY|PEEK|INPUT|FN)\b', text, re.IGNORECASE):
                break
            k += 1
        return k

    def _record_save(self, kind: str, var: str, value: Any) -> None:
        self._live_saves.setdefault(var, (kind, []))[1].append(value)

    def _run_live(self, t0: float, sorted_lines: list[int], carry: BufferedIO | None = None) -> None:
        """Run the program shot by shot (see the module docstring)."""
        from qubasic_core import capacity as cap
        n = self.num_qubits
        prec = self.sim_precision
        if self._pending_set_density is not None:
            self.io.writeln("?A program that reads measurement results runs shot by shot "
                            "on a pure state; use SET_STATE rather than SET_DENSITY")
            return
        peak = cap.numpy_peak_bytes(n, prec)
        if not cap.fits(peak):
            self.io.writeln(f"?MEMORY: a live run at {n} qubits needs ~{cap.fmt_gb(peak)}, "
                            f"{cap.fmt_gb(cap.available_bytes())} available")
            return
        has_measure = self._program_has_measure(sorted_lines)
        shots = self.shots if has_measure else 1
        noise = self._live_noise()
        state = LiveState(n, prec, noise)
        backend = LiveBackend(state, self._custom_gates)
        shot_io = ShotIO(self.io, carry.answers if carry else None, carry.shown if carry else 0)
        real_io = self.io
        base_vars = dict(self.variables)
        base_arrays = {k: list(v) for k, v in self.arrays.items()}
        keep_copy = shots > 1 and cap.fits(cap.numpy_peak_bytes(n, prec, True))
        if self._seed is not None:
            np.random.seed(self._seed)
        self._live_saves = {}
        counts: dict[str, int] = {}
        first_vars: dict[str, Any] = {}
        first_arrays: dict[str, Any] = {}
        first_sv = None
        first_ops = 0
        start, snapshot, start_vars, start_arrays = 0, None, base_vars, base_arrays
        prefix_ops = 0
        progress = shots > 50 and sys.stdout.isatty()
        self._live = state
        self.io = shot_io
        try:
            prefix = self._live_prefix(sorted_lines) if noise is None and keep_copy else 0
            if prefix:
                self._gosub_stack = []
                self.build_circuit(backend=backend, stop_ip=prefix)
                snapshot = state.sv.copy()
                prefix_ops = state.ops
                start, start_vars = prefix, dict(self.variables)
                start_arrays = {k: list(v) for k, v in self.arrays.items()}
            for shot in range(shots):
                if shot:
                    shot_io.start_replay()
                if shot or snapshot is None:
                    state.reset_state(snapshot)
                    self.variables.clear()
                    self.variables.update(start_vars)
                    self.arrays.clear()
                    self.arrays.update({k: list(v) for k, v in start_arrays.items()})
                state.ops = 0
                self._gosub_stack = []
                self._data_ptr = 0
                self.build_circuit(backend=backend, start_ip=start)
                if has_measure:
                    key = state.sample_final(self._measure_subset)
                    counts[key] = counts.get(key, 0) + 1
                if shot == 0:
                    first_vars = dict(self.variables)
                    first_arrays = {k: list(v) for k, v in self.arrays.items()}
                    first_ops = prefix_ops + state.ops
                    if shots == 1:
                        first_sv = state.sv
                    elif keep_copy:
                        first_sv = state.sv.copy()
                if progress and (shot + 1) % max(1, shots // 10) == 0:
                    real_io.write(f"  {100 * (shot + 1) // shots}% ({shot + 1}/{shots} shots)\r")
        except KeyboardInterrupt:
            self.io = real_io
            self.io.writeln("\n?INTERRUPTED")
            return
        except Exception as e:
            self.io = real_io
            self.io.writeln(f"?RUNTIME ERROR: {e}")
            return
        finally:
            self.io = real_io
            self._live = None
        if progress:
            real_io.write(" " * 40 + "\r")

        self.variables.clear()
        self.variables.update(first_vars)
        self.arrays.clear()
        self.arrays.update(first_arrays)
        self._finish_live_saves()
        self.last_sv = first_sv
        self.last_counts = counts if has_measure else None
        self.last_circuit = None
        dt = time.time() - t0
        self._update_status(gate_count=first_ops, circuit_depth=0, run_time_ms=dt * 1000)
        self.variables['_TIME'] = dt
        self._run_manifest = {
            'program': dict(self.program), 'num_qubits': n, 'shots': shots,
            'method': 'live', 'precision': prec, 'device': 'CPU', 'seed': self._seed,
            'noise_depol_p': self._noise_depol_p, 'depth': 0, 'gates': first_ops,
            'time_s': dt,
        }
        tags = ["method=live"]
        if prec == 'single':
            tags.append("precision=single")
        if noise is not None:
            tags.append("noise=trajectories")
        shot_word = f"{shots} shot{'s' if shots != 1 else ''}"
        self.io.writeln(f"\nRAN {len(self.program)} lines, {n} qubits, {shot_word} in {dt:.2f}s  "
                        f"[{first_ops} gates/shot, {', '.join(tags)}]")
        if has_measure and counts:
            self.print_histogram(counts)
        elif not has_measure:
            self.io.writeln("(no MEASURE — one shot; use STATE, PROBS, or BLOCH to inspect)")

    def _finish_live_saves(self) -> None:
        """SAVE_EXPECT / SAVE_PROBS hold their mean over the shots after a
        live run; SAVE_AMPS holds the first shot's amplitudes."""
        for var, (kind, values) in self._live_saves.items():
            if kind == 'amp':
                val = values[0]
                self.arrays[var] = [complex(v) for v in val]
            else:
                val = np.mean(np.array(values), axis=0)
                if kind == 'prob':
                    self.arrays[var] = list(np.asarray(val, dtype=float))
                else:
                    val = float(val)
            self.variables[var] = val
