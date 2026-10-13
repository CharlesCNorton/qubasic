"""QUBASIC runs: the last run's results and how they print, Aer options and
method selection, the RUN pipeline, and live (shot-by-shot) execution for
programs that read measurement results.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qubasic_core.circuits import _feedforward_test
from qubasic_core.engine import (
    _RICH,
    AMPLITUDE_THRESHOLD,
    HISTOGRAM_BAR_WIDTH,
    MAX_DISPLAY_AMPLITUDES,
    MAX_HISTOGRAM_STATES,
    ExecResult,
    QBasicBuildError,
    QBasicRangeError,
    RunReport,
    _RichTable,
)
from qubasic_core.host import SimulationHost
from qubasic_core.parser import (
    GATE_ALIASES,
)
from qubasic_core.simtools import (
    _GATE_BUILDERS,
    Tableau,
    _apply_gate_inplace,
    _measure_inplace,
    _np_gate_matrix,
    _sample_counts,
    clifford_table,
    pauli_mixture,
)

if TYPE_CHECKING:

    from qiskit.result import Result
    from qiskit_aer.noise import NoiseModel

    from qubasic_core.engine import ExecContext


# ---------------------------------------------------------------------------
# The last run's results and how to read them: the statevector (computed on
# first read when a run deferred it), the active state under LOCC, the
# bitstring order, the run summary's complexity note, and the printed forms
# every layer shows results in (the histogram, the statevector table, the
# one-line state and the LOCC registers).
# ---------------------------------------------------------------------------

class RunStateMixin(SimulationHost):
    """The last run's state, the conventions for reading it and the printed
    forms of results."""

    @property
    def last_sv(self) -> Any:
        """The last run's statevector, computed on first read when the run
        deferred it (a deferred source runs at most once)."""
        if self._last_sv is None and self._state_source is not None:
            source, self._state_source = self._state_source, None
            self._last_sv = source()
        return self._last_sv

    @last_sv.setter
    def last_sv(self, value: Any) -> None:
        self._last_sv = value
        self._state_source = None
        self._last_tableau = None

    def _defer_state(self, source: Any) -> None:
        """Record a zero-argument callable that produces the state on demand."""
        self._last_sv = None
        self._state_source = source
        self._last_tableau = None

    def _invalidate_run_state(self) -> None:
        """Drop cached results that describe a now-stale configuration.

        Called when the qubit count changes: last_sv, last_counts,
        last_circuit, and the transpile cache all assume the previous
        num_qubits, so reading them afterwards would mislabel (or, for
        the reshape in _peek_qubit/_bloch_vector, crash) the state.
        """
        self.last_counts = None
        self.last_sv = None
        self.last_circuit = None
        self._circuit_cache_key = None
        self._circuit_cache = None

    @property
    def _active_sv(self) -> 'np.ndarray | None':
        """Return the current authoritative statevector.

        In LOCC JOINT mode, returns the joint statevector.
        In LOCC SPLIT mode, returns None (use STATE A / STATE B instead,
        since split registers have independent states).
        In standard mode, returns last_sv.
        """
        if self.locc_mode and self.locc:
            if self.locc.joint:
                return np.ascontiguousarray(self.locc.sv).ravel()
            return None  # split: no single SV; use per-register commands
        return self.last_sv

    def _active_sv_for_reg(self, reg: str) -> 'tuple[np.ndarray, int] | tuple[None, int]':
        """Return (statevector, n_qubits) for a specific LOCC register."""
        if not self.locc:
            return None, 0
        if self.locc.joint:
            return np.ascontiguousarray(self.locc.sv).ravel(), self.locc.n_total
        sv = self.locc.svs.get(reg)
        if sv is None:
            return None, 0
        return np.ascontiguousarray(sv).ravel(), self.locc.get_size(reg)

    @property
    def _active_nqubits(self) -> int:
        """Return the qubit count matching _active_sv."""
        if self.locc_mode and self.locc:
            if self.locc.joint:
                return self.locc.n_total
            return 0  # split: no single qubit count
        return self.num_qubits

    def _bits(self, s: str) -> str:
        """A bitstring in the active display convention.

        Little-endian (qubit 0 rightmost, the qiskit order) unless
        OPTION ENDIAN BIG reversed it. LOCC joint keys ('0|01|1') reverse
        per register segment, keeping the A|B|C register order.
        """
        if not getattr(self, '_endian_big', False):
            return s
        if '|' in s:
            return '|'.join(seg[::-1] for seg in s.split('|'))
        return s[::-1]

    def _circuit_complexity(self) -> str:
        """Return a complexity summary string for the last circuit."""
        if not self.last_circuit:
            return ''
        qc = self.last_circuit
        ops = qc.count_ops()
        t_count = ops.get('t', 0) + ops.get('tdg', 0)
        cx_count = ops.get('cx', 0) + ops.get('cnot', 0)
        parts = []
        if t_count:
            parts.append(f"T-count={t_count}")
        if cx_count:
            parts.append(f"CNOT={cx_count}")
        return ', '.join(parts)

    def _rich_print(self, *renderables: Any) -> None:
        """Render rich content to a string and emit it through the IOPort.

        Routing rich output through self.io (instead of straight to stdout)
        means headless/agent callers with a custom IOPort capture tables and
        statevectors too, and the IOPort's encoding fallback applies uniformly.
        ANSI colour is kept only when stdout is a real terminal.
        """
        import io as _io

        from rich.console import Console
        buf = _io.StringIO()
        force = bool(sys.stdout and sys.stdout.isatty())
        Console(file=buf, highlight=False, force_terminal=force).print(*renderables)
        self.io.write(buf.getvalue())

    def _bit_order_note(self, display: list) -> str | None:
        """Human-facing reminder of which bit is which qubit.

        Bitstrings are little-endian (qubit 0 is the rightmost character), the
        most common source of off-by-reverse mistakes when reading a histogram,
        unless OPTION ENDIAN BIG flipped the display order. When the result
        covers the whole register the positions are labelled per qubit; for a
        measured subset only the convention is shown.
        """
        if not display:
            return None
        big = getattr(self, '_endian_big', False)
        nbits = len(display[0][0])
        nq = getattr(self, 'num_qubits', nbits)
        if nbits == nq and nbits <= 16:
            order = range(nbits) if big else range(nbits - 1, -1, -1)
            labels = ' '.join(f'q{i}' for i in order)
            return f"  bit order  {labels}   (qubit 0 = {'leftmost' if big else 'rightmost'})"
        return f"  bit order  qubit 0 = {'leftmost' if big else 'rightmost'} bit"

    def print_histogram(self, counts: dict[str, int]) -> None:
        """Measurement histogram with optional rich-table formatting."""
        total = sum(counts.values())
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        display = sorted_counts[:MAX_HISTOGRAM_STATES]

        note = self._bit_order_note(display)
        if note:
            self.io.writeln(note)

        if _RICH:
            self._print_histogram_rich(display, sorted_counts, total)
        else:
            self._print_histogram_plain(display, sorted_counts, total)

    def _print_histogram_rich(self, display: list, sorted_counts: list,
                              total: int) -> None:
        """Rich-formatted histogram with colored bars."""
        table = _RichTable(show_header=True, header_style="bold cyan",
                           box=None, padding=(0, 1))
        table.add_column("State", justify="right", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("%", justify="right")
        table.add_column("Distribution", min_width=HISTOGRAM_BAR_WIDTH)

        max_count = max(c for _, c in display) if display else 1
        if len(sorted_counts) > MAX_HISTOGRAM_STATES:
            self._rich_print(
                f"\n  [dim]Showing top {MAX_HISTOGRAM_STATES} of "
                f"{len(sorted_counts)} outcomes:[/dim]\n")
        else:
            self._rich_print()

        for state, count in display:
            pct = 100 * count / total
            bar_len = int(HISTOGRAM_BAR_WIDTH * count / max_count)
            bar = '\u2588' * bar_len
            color = "green" if pct > 40 else "yellow" if pct > 10 else "dim"
            table.add_row(
                f"|{self._bits(state)}\u27E9",
                str(count),
                f"{pct:5.1f}%",
                f"[{color}]{bar}[/{color}]")

        if len(sorted_counts) > MAX_HISTOGRAM_STATES:
            rest_count = sum(c for _, c in sorted_counts[MAX_HISTOGRAM_STATES:])
            table.add_row(
                "...", str(rest_count),
                f"{100*rest_count/total:5.1f}%", "[dim](remaining)[/dim]")

        try:
            self._rich_print(table)
            self._rich_print()
        except UnicodeEncodeError:
            self._print_histogram_plain(display, sorted_counts, total)

    def _print_histogram_plain(self, display: list, sorted_counts: list,
                               total: int) -> None:
        """Plain-text histogram (fallback when rich is not available)."""
        if len(sorted_counts) > MAX_HISTOGRAM_STATES:
            self.io.writeln(f"\n  Showing top {MAX_HISTOGRAM_STATES} of {len(sorted_counts)} outcomes:\n")
        else:
            self.io.writeln('')

        max_count = max(c for _, c in display) if display else 1
        max_label = max(len(k) for k, _ in display) if display else 1

        _theme = getattr(self, '_theme', {})
        for state, count in display:
            pct = 100 * count / total
            bar_len = int(HISTOGRAM_BAR_WIDTH * count / max_count)
            bar = '\u2588' * bar_len
            ket = f"|{self._bits(state)}\u27E9"
            # Colorize bar by probability
            if _theme and sys.stdout.isatty():
                rst = _theme.get('reset', '')
                if pct > 40:
                    bar = f"{_theme.get('bar_hi', '')}{bar}{rst}"
                elif pct > 10:
                    bar = f"{_theme.get('bar_mid', '')}{bar}{rst}"
                else:
                    bar = f"{_theme.get('bar_lo', '')}{bar}{rst}"
            self.io.writeln(f"  {ket:>{max_label+3}}  {count:>6}  ({pct:5.1f}%)  {bar}")

        if len(sorted_counts) > MAX_HISTOGRAM_STATES:
            rest_count = sum(c for _, c in sorted_counts[MAX_HISTOGRAM_STATES:])
            self.io.writeln(f"  {'...':>{max_label+3}}  {rest_count:>6}  "
                            f"({100*rest_count/total:5.1f}%)  (remaining)")
        self.io.writeln('')

    def _print_statevector(self, sv: np.ndarray, n_qubits: int | None = None) -> None:
        """Print the first non-zero amplitudes of the statevector (index order).

        Only the shown amplitudes are formatted; the rest are counted in one
        vectorized pass, so a large state prints in seconds.
        """
        from qubasic_core.simtools import nonzero_indices
        sv = np.asarray(sv).reshape(-1)
        n = n_qubits if n_qubits is not None else self.num_qubits
        idx, total = nonzero_indices(sv, AMPLITUDE_THRESHOLD, MAX_DISPLAY_AMPLITUDES)
        rows = []
        for i in idx:
            amp = complex(sv[int(i)])
            rows.append((f"|{self._bits(format(int(i), f'0{n}b'))}\u27E9",
                         f"{amp.real:+.4f}{amp.imag:+.4f}j", abs(amp) ** 2))
        remaining = total - len(rows)

        if _RICH:
            table = _RichTable(show_header=True, header_style="bold cyan",
                               box=None, padding=(0, 1))
            table.add_column("State", justify="right", style="bold")
            table.add_column("Amplitude", justify="right")
            table.add_column("P", justify="right")
            self._rich_print(f"\n  [bold]Statevector ({n} qubits):[/bold]")
            for ket, amp_s, prob in rows:
                table.add_row(ket, amp_s, f"{prob:.4f}")
            if remaining > 0:
                table.add_row("...", "", f"+{remaining} more")
            try:
                self._rich_print(table)
                self._rich_print()
                return
            except UnicodeEncodeError:
                pass  # fall through to plain-text path below

        self.io.writeln(f"\n  Statevector ({n} qubits):")
        for ket, amp_s, prob in rows:
            self.io.writeln(f"  {ket}  {amp_s}  (P={prob:.4f})")
        if remaining > 0:
            self.io.writeln(f"  ... and {remaining} more non-zero amplitudes")
        self.io.writeln('')

    def _print_sv_compact(self, sv: np.ndarray) -> None:
        """Compact statevector display for step mode (first 8 non-zero terms)."""
        from qubasic_core.simtools import nonzero_indices
        sv = np.asarray(sv).reshape(-1)
        n = self.num_qubits
        idx, _ = nonzero_indices(sv, AMPLITUDE_THRESHOLD, 9, count=False)
        parts = []
        for i in idx[:8]:
            amp = complex(sv[int(i)])
            state = self._bits(format(int(i), f'0{n}b'))
            if abs(amp.imag) < AMPLITUDE_THRESHOLD:
                parts.append(f"{amp.real:+.3f}|{state}\u27E9")
            else:
                parts.append(f"({amp.real:.2f}{amp.imag:+.2f}j)|{state}\u27E9")
        if len(idx) > 8:
            parts.append("...")
        self.io.writeln(f"   |\u03C8\u27E9 = {' '.join(parts)}")

    def _locc_state(self, rest: str = '') -> None:
        reg = rest.strip().upper() if rest else ''
        if self._locc_engine.joint:
            if not reg or reg in self._locc_engine.names:
                sizes = '+'.join(str(s) for s in self._locc_engine.sizes)
                self.io.writeln(f"\n  Joint statevector ({sizes} qubits):")
                self._print_statevector(self._locc_engine.sv, self._locc_engine.n_total)
        else:
            show = [reg] if reg and reg in self._locc_engine.names else self._locc_engine.names
            for name in show:
                size = self._locc_engine.get_size(name)
                self.io.writeln(f"\n  Register {name} ({size} qubits):")
                self._print_statevector(self._locc_engine.svs[name], size)


# ---------------------------------------------------------------------------
# QUBASIC run pipeline: backend options, method selection, RUN, SAMPLE,
# ESTIMATE, STEP and its REWIND and FORWARD.
# ---------------------------------------------------------------------------

def _has_channel(qc: QuantumCircuit) -> bool:
    """Whether a circuit applies a non-unitary channel (APPLYCHANNEL)."""
    return any(inst.operation.name in ('kraus', 'superop', 'quantum_channel')
               for inst in qc.data)


def _expectation(val: Any) -> float:
    """An expectation value as a float, with rounding residue under 1e-12 read as 0."""
    v = float(np.real(val))
    return 0.0 if abs(v) < 1e-12 else v


def _circuit_fingerprint(qc: QuantumCircuit) -> str:
    """Stable digest of a circuit's registers and instructions.

    Covers operation names, parameters (arrays by content), qubit and clbit
    indices, nested control-flow blocks, if_test conditions, and the operator
    and step count of Hamiltonian evolution gates.
    """
    import hashlib

    from qiskit import QuantumCircuit as _QC
    h = hashlib.blake2b(digest_size=16)

    def feed(obj: Any) -> None:
        if isinstance(obj, np.ndarray):
            h.update(obj.tobytes())
        elif isinstance(obj, _QC):
            h.update(_circuit_fingerprint(obj).encode())
        elif hasattr(obj, 'data') and isinstance(getattr(obj, 'data', None), np.ndarray):
            h.update(obj.data.tobytes())
        elif isinstance(obj, (list, tuple)):
            h.update(b'[')
            for x in obj:
                feed(x)
            h.update(b']')
        else:
            h.update(repr(obj).encode())

    h.update(repr([(r.name, r.size) for r in qc.qregs]).encode())
    h.update(repr([(r.name, r.size) for r in qc.cregs]).encode())
    for inst in qc.data:
        op = inst.operation
        h.update(op.name.encode())
        feed(list(op.params))
        h.update(repr([qc.find_bit(q).index for q in inst.qubits]).encode())
        h.update(repr([qc.find_bit(c).index for c in inst.clbits]).encode())
        cond = getattr(op, 'condition', None)
        if cond is not None:
            target, value = cond
            h.update(repr((getattr(target, 'name', target), value)).encode())
        oper = getattr(op, 'operator', None)
        if oper is not None:
            h.update(repr(oper).encode())
            synth = getattr(op, 'synthesis', None)
            h.update(repr((type(synth).__name__, getattr(synth, 'reps', None),
                           getattr(synth, 'order', None))).encode())
    return h.hexdigest()


class RunPipelineMixin(SimulationHost):
    """Running programs for QBasicTerminal: Aer backend options, memory-aware
    method selection, the RUN pipeline (circuit build, live fallback, shots,
    state capture, result display), SAMPLE, ESTIMATE and STEP."""

    _PRECISION_METHODS = ('automatic', 'statevector', 'density_matrix', 'unitary', 'superop')

    def _aer_options(self, method: str = 'statevector', *, include_noise: bool = True,
                     precision: str | None = None) -> dict:
        """AerSimulator options for one run: method, device, precision, noise,
        and the tuning knobs set through the memory map ($D008-$D00B)."""
        opts: dict = {'method': method}
        if self.sim_device == 'GPU':
            opts['device'] = 'GPU'
        prec = precision or self.sim_precision
        if prec == 'single' and method in self._PRECISION_METHODS:
            opts['precision'] = 'single'
        if include_noise:
            noise = self.noise.model
            if not noise and self.noise.per_qubit:
                noise = self._build_qubit_noise()
            if noise:
                opts['noise_model'] = noise
        opts.update(self.aer.options())
        return opts

    def _make_backend(self, method: str = 'statevector', *,
                      include_noise: bool = False,
                      precision: str | None = None) -> 'AerSimulator':
        """Build an AerSimulator respecting device, precision, tuning and
        (optionally) noise, so every execution path is configured alike."""
        return AerSimulator(**self._aer_options(method, include_noise=include_noise,
                                                precision=precision))

    def _build_qubit_noise(self) -> NoiseModel | None:
        """Build a NoiseModel from per-qubit memory-mapped noise settings."""
        try:
            from qiskit_aer.noise import (
                NoiseModel,
                amplitude_damping_error,
                depolarizing_error,
                phase_damping_error,
            )
            nm = NoiseModel()
            _type_map = {1: depolarizing_error, 2: amplitude_damping_error,
                         3: phase_damping_error}
            _1q_gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'p', 'u',
                         'id', 's', 't', 'sdg', 'tdg', 'sx']
            _2q_gates = ['cx', 'cy', 'cz', 'ch', 'swap', 'dcx', 'iswap',
                         'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz']
            id1 = depolarizing_error(0.0, 1)  # identity channel for tensoring
            for q, (ntype, nparam) in self.noise.per_qubit.items():
                if ntype in _type_map and nparam > 0 and q < self.num_qubits:
                    if ntype == 1:
                        err = _type_map[ntype](nparam, 1)
                    else:
                        err = _type_map[ntype](nparam)
                    nm.add_quantum_error(err, _1q_gates, [q])
                    # Also fire the per-qubit error after any 2-qubit gate that
                    # touches q (identity on the partner qubit).
                    e_first = id1.tensor(err)   # err on qubit 0 of the pair
                    e_second = err.tensor(id1)  # err on qubit 1 of the pair
                    for j in range(self.num_qubits):
                        if j == q:
                            continue
                        nm.add_quantum_error(e_first, _2q_gates, [q, j])
                        nm.add_quantum_error(e_second, _2q_gates, [j, q])
            return nm
        except ImportError:
            return None

    @property
    def _transpile_opt_level(self) -> int:
        """Optimization level for transpilation.

        When a noise model is active, use level 0 so the transpiler does not
        collapse gate sequences (e.g. H-H -> identity) before noise channels
        can attach to them.  Without noise, use default (1) for performance.
        """
        if self.noise.model is not None:
            return 0
        if self.noise.per_qubit:
            return 0
        return 1

    def _transpile_kwargs(self) -> dict:
        """coupling_map / basis_gates kwargs for transpile() (device model)."""
        kw: dict = {}
        if self._device_target is not None:
            kw['target'] = self._device_target            # DEVICE IBM: the real device's target
            return kw
        if self._coupling_map:
            kw['coupling_map'] = self._coupling_map
        if self._basis_gates:
            kw['basis_gates'] = self._basis_gates
        return kw

    def _transpile_routed(self, qc: QuantumCircuit, backend: AerSimulator, level: int | None = None) -> QuantumCircuit:
        """transpile() honoring the device model. Passing coupling_map or
        basis_gates alongside a backend is exactly how the offline device
        model works, so qiskit's advisory warning about it is silenced, as is
        its note that a DELAY was rounded to the device's clock cycle."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message=r'Providing `coupling_map`', category=UserWarning)
            warnings.filterwarnings('ignore', message=r'Duration is rounded', category=UserWarning)
            return transpile(
                qc, backend,
                optimization_level=self._transpile_opt_level if level is None else level,
                **self._transpile_kwargs())

    def _run_kwargs(self) -> dict:
        """Build keyword args for backend.run() with optional seed."""
        kw: dict = {'shots': self.shots}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        return kw

    def _select_method(self, qc: QuantumCircuit) -> tuple[str, str]:
        """Choose (method, precision) for a run.

        Under METHOD automatic: noiseless Clifford circuits use the stabilizer
        simulator at any width; otherwise the statevector family runs while it
        fits in available memory, at double precision when that fits and at
        single precision when only that fits; past that, matrix_product_state.
        Other methods run as set.
        """
        from qubasic_core import simtools as cap
        method, prec = self.sim_method, self.sim_precision
        if method != 'automatic':
            return method, prec
        noiseless = not self.noise.model and not self.noise.per_qubit
        if noiseless and self._is_clifford(qc):
            return 'stabilizer', prec
        n = self.num_qubits
        if cap.fits(cap.aer_peak_bytes(n, 'statevector', prec)):
            return 'automatic', prec
        if prec == 'double' and cap.fits(cap.aer_peak_bytes(n, 'statevector', 'single')):
            return 'statevector', 'single'
        return 'matrix_product_state', prec

    def _build_backend_opts(self, method: str, precision: str | None = None) -> dict:
        """AerSimulator options for a measured RUN (noise included)."""
        return self._aer_options(method, include_noise=True, precision=precision)

    def _check_fits(self, method: str, precision: str) -> None:
        """Refuse a width-bound run that cannot fit in available memory."""
        from qubasic_core import simtools as cap
        n = self.num_qubits
        peak = cap.aer_peak_bytes(n, 'statevector' if method == 'automatic' else method, precision)
        if cap.fits(peak):
            return
        hint = []
        if precision == 'double' and method in self._PRECISION_METHODS:
            hint.append("PRECISION SINGLE")
        if method in ('statevector', 'automatic'):
            hint.append("METHOD matrix_product_state")
        tail = f"; try {' or '.join(hint)}" if hint else ""
        raise QBasicRangeError(
            f"MEMORY: METHOD {method} at {n} qubits needs ~{cap.fmt_gb(peak)}, "
            f"{cap.fmt_gb(cap.available_bytes())} available{tail}")

    def _capture_mode(self, method: str, precision: str) -> str:
        """How a measured run keeps its final state for STATE, BLOCH and the
        other inspection commands: 'mid' saves it inside the run (no second
        simulation; needs a copy's worth of memory), 'lazy' recomputes it with
        a measure-free run on first use (1.04x the state), 'none' when even
        that cannot fit."""
        from qubasic_core import simtools as cap
        n = self.num_qubits
        if method in ('automatic', 'statevector') and not self._coupling_map and self._device_target is None:
            if cap.fits(cap.aer_peak_bytes(n, 'statevector', precision, 'mid')):
                return 'mid'
        if cap.fits(cap.aer_peak_bytes(n, 'statevector', precision, 'end')):
            return 'lazy'
        return 'none'

    def _defer_state_extraction(self, qc_sv: QuantumCircuit, precision: str) -> None:
        """Leave a source that computes the measure-free circuit's state on
        first read of last_sv (one noisy trajectory when noise is active)."""
        seed = self._seed
        self._defer_state(lambda: self._extract_state_now(qc_sv, precision, seed))

    def _extract_state_now(self, qc_sv: QuantumCircuit, precision: str, seed: int | None = None) -> np.ndarray | None:
        """Simulate the measure-free circuit and return its final state, or
        None (with a note) when it cannot fit in available memory."""
        from qubasic_core import simtools as cap
        n = qc_sv.num_qubits
        peak = cap.aer_peak_bytes(n, 'statevector', precision, 'end')
        if not cap.fits(peak):
            hint = " (PRECISION SINGLE halves it)" if precision == 'double' else ""
            self.io.writeln(f"  (no statevector: {n} qubits needs ~{cap.fmt_gb(peak)}, "
                            f"{cap.fmt_gb(cap.available_bytes())} available{hint})")
            return None
        if peak is not None and peak > 1e9:
            self.io.writeln(f"  (computing the {n}-qubit state, ~{cap.fmt_gb(peak)})")
        try:
            q2 = qc_sv.copy()
            q2.save_statevector(label='qb_state')
            backend = self._make_backend('statevector', include_noise=True, precision=precision)
            kw: dict = {'shots': 1}
            if seed is not None:
                kw['seed_simulator'] = seed
            res = backend.run(transpile(q2, backend,
                                        optimization_level=self._transpile_opt_level),
                              **kw).result()
            return np.asarray(res.data()['qb_state'].data)
        except Exception as e:
            self.io.writeln(f"  (statevector unavailable: {e})")
            return None

    def _extract_save_results(self, result: Result) -> None:
        """Extract SAVE_EXPECT/SAVE_PROBS/SAVE_AMPS into BASIC variables."""
        data = result.data()
        for key, val in data.items():
            if key.startswith('exp_'):
                self.variables[key[4:]] = _expectation(val)
            elif key.startswith('prob_'):
                var = key[5:]
                self.variables[var] = val
                if isinstance(val, np.ndarray):
                    self.arrays[var] = val.tolist()
            elif key.startswith('amp_'):
                var = key[4:]
                if isinstance(val, np.ndarray):
                    self.arrays[var] = [complex(v) for v in val]
                self.variables[var] = val

    def _density_from_qc(self, qc_sv: QuantumCircuit) -> np.ndarray | None:
        """Solve the density matrix of a measure-free circuit, for DENSITY.

        Computed on demand (not on every run) from the circuit captured when a
        SET_DENSITY state was prepared.
        """
        try:
            q2 = qc_sv.copy()
            q2.save_density_matrix()
            dm_backend = self._make_backend('density_matrix', include_noise=True)
            _kw = {}
            if self._seed is not None:
                _kw['seed_simulator'] = self._seed
            res = dm_backend.run(
                transpile(q2, dm_backend, optimization_level=self._transpile_opt_level),
                **_kw).result()
            data = res.data(0)
            dm = data.get('density_matrix') if hasattr(data, 'get') else None
            return np.array(dm) if dm is not None else None
        except Exception:
            return None

    def _finalize_run(self, qc: QuantumCircuit, method: str, t0: float, precision: str | None = None) -> None:
        """Update status, manifest, metrics after a successful run."""
        dt = time.time() - t0
        depth = qc.depth()
        n_gates = qc.size()
        self._update_status(gate_count=n_gates, circuit_depth=depth,
                           run_time_ms=dt * 1000)
        self.variables['_DEPTH'] = depth
        self.variables['_GATES'] = n_gates
        self.variables['_TIME'] = dt
        self._run_manifest = {
            'program': dict(self.program),
            'num_qubits': self.num_qubits,
            'shots': self.shots,
            'method': method,
            'precision': precision or self.sim_precision,
            'device': self.sim_device,
            'seed': self._seed,
            'noise_depol_p': self.noise.depol_p,
            'depth': depth,
            'gates': n_gates,
            'time_s': dt,
        }

    def _run_no_measure(self, qc: QuantumCircuit, qc_sv: QuantumCircuit, t0: float, precision: str | None = None) -> None:
        """Execute the no-MEASURE path: one measure-free statevector run that
        keeps the final state (1.04x the state; Aer hands the buffer over), or
        the lazily solved density matrix for a SET_DENSITY state."""
        from qubasic_core import simtools as cap
        prec = precision or self.sim_precision
        n = self.num_qubits
        peak = cap.aer_peak_bytes(n, 'statevector', prec, 'end')
        fits = cap.fits(peak)
        if self._pending_set_density is not None or _has_channel(qc_sv):
            # A mixed state has no statevector: DENSITY solves it on demand
            # from the measure-free circuit.
            self.last_sv = None
            self._last_density = None
            self._last_density_qc = qc_sv
        elif not qc_sv.data:
            # Nothing to simulate (size() would also skip SET_STATE and SAVE_*
            # directives): the state is |0...0>, built when first read.
            dtype = np.complex64 if prec == 'single' else np.complex128

            def _zero_state(_n: int = n, _dtype: Any = dtype) -> np.ndarray:
                sv = np.zeros(1 << _n, dtype=_dtype)
                sv[0] = 1.0
                return sv
            self._defer_state(_zero_state)
        elif not fits:
            self.last_sv = None
        else:
            try:
                q2 = qc_sv.copy()
                q2.save_statevector(label='qb_state')
                sv_backend = self._make_backend('statevector', include_noise=True, precision=prec)
                kw: dict = {'shots': 1}
                if self._seed is not None:
                    kw['seed_simulator'] = self._seed
                sv_result = sv_backend.run(
                    transpile(q2, sv_backend, optimization_level=self._transpile_opt_level),
                    **kw).result()
                self.last_sv = np.asarray(sv_result.data()['qb_state'].data)
                ensemble = self._ensemble_saves(qc_sv, prec)
                self._extract_save_results(ensemble if ensemble is not None else sv_result)
            except (RuntimeError, ValueError, TypeError, KeyError):
                self.last_sv = None
        self.last_counts = None
        self._finalize_run(qc, 'statevector', t0, prec)
        depth = qc.depth()
        n_gates = qc.size()
        if not qc.data:
            return      # a classical program: its own output is the result
        dt = time.time() - t0
        tag = ", precision=single" if prec == 'single' else ""
        self.io.writeln(f"\nRAN {len(self.program)} lines, {n} qubits "
                        f"in {dt:.2f}s  [depth={depth}, gates={n_gates}{tag}]")
        if self._last_density_qc is not None:
            self.io.writeln("(no MEASURE; the state is mixed — use DENSITY to inspect)")
        elif not fits:
            hint = "; PRECISION SINGLE halves it" if prec == 'double' else ""
            self.io.writeln(f"(no MEASURE; a {n}-qubit statevector needs ~{cap.fmt_gb(peak)}, "
                            f"{cap.fmt_gb(cap.available_bytes())} available{hint})")
        else:
            self.io.writeln("(no MEASURE — use STATE, PROBS, or BLOCH to inspect)")

    def _ensemble_saves(self, qc: QuantumCircuit, precision: str) -> Result | None:
        """Under noise, the SAVE_* values of a measure-free circuit over the
        ensemble rather than one trajectory: exactly on the density matrix
        when it fits, otherwise averaged over SHOTS trajectories. None when
        the run is noiseless, saves nothing, or the ensemble run fails."""
        from qubasic_core import simtools as cap
        if not (self.noise.model or self.noise.per_qubit):
            return None
        if not any(inst.operation.name.startswith('save_') for inst in qc.data):
            return None
        n = qc.num_qubits
        if cap.fits(cap.aer_peak_bytes(n, 'density_matrix', precision)):
            method, shots = 'density_matrix', 1
        else:
            method, shots = 'statevector', self.shots
        backend = self._make_backend(method, include_noise=True, precision=precision)
        kw: dict = {'shots': shots}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        try:
            return backend.run(transpile(qc, backend, optimization_level=self._transpile_opt_level),
                               **kw).result()
        except Exception:
            return None

    def _add_final_measurement(self, qc: QuantumCircuit) -> None:
        """Append the program's final measurement: its MEASURE subset into a
        'sub' register, or every qubit."""
        subset = self._measure_subset
        if subset:
            from qiskit.circuit import ClassicalRegister
            cr = ClassicalRegister(len(subset), 'sub')
            qc.add_register(cr)
            for i, q in enumerate(subset):
                qc.measure(q, cr[i])
        else:
            qc.measure_all()

    def _keep_state(self, result: Result, qc_sv: QuantumCircuit, method: str, precision: str, capture: str) -> None:
        """Keep the final state of a measured run: from the run itself when it
        saved one, otherwise as a deferred extraction from the measure-free
        circuit (computed on first use)."""
        if method in ('unitary', 'superop'):
            self.last_sv = None
            return
        if self._pending_set_density is not None or _has_channel(qc_sv):
            # Mixed state: DENSITY solves the density matrix on demand.
            self.last_sv = None
            self._last_density = None
            self._last_density_qc = qc_sv
            return
        if capture == 'mid':
            try:
                data = result.data()
                if 'qb_state' in data:
                    self.last_sv = np.asarray(data['qb_state'].data)
                    return
            except Exception:
                pass
        self._defer_state_extraction(qc_sv, precision)

    def _run_with_fallback(self, qc: QuantumCircuit, backend_opts: dict, method: str,
                           capture: str = 'none') -> 'Any':
        """Run the circuit with shots, handling GPU and stabilizer fallbacks."""
        # str(NoiseModel) does not vary with channel parameters (amplitude_damping
        # 0.1 and 0.9 stringify identically), so fingerprint noise by the exact
        # NOISE command and the per-qubit memory-mapped noise instead — otherwise
        # changing noise strength between runs reuses the first circuit.
        _noise_key = (
            self.noise.spec,
            id(self.noise.model) if self.noise.model is not None else None,
            tuple(sorted(self.noise.per_qubit.items())),
        )
        # The built circuit itself is the key: every input that shapes it
        # (program text, DEF/UNITARY/HAMILTONIAN/CHANNEL definitions, variable
        # and array values, SET_STATE, POKE state prep) lands in its
        # instructions, so a fingerprint of the instructions cannot go stale.
        _dev_key = (
            tuple(map(tuple, self._coupling_map)) if self._coupling_map else None,
            tuple(self._basis_gates) if self._basis_gates else None,
        )
        cache_key = (
            _circuit_fingerprint(qc),
            method, self.sim_device,
            backend_opts.get('precision', 'double'), capture,
            _noise_key, _dev_key,
            tuple(sorted(self.aer.options().items())),
        )
        if self._circuit_cache_key == cache_key and self._circuit_cache is not None:
            qc_t, backend = self._circuit_cache
        else:
            backend = AerSimulator(**backend_opts)
            qc_t = self._transpile_routed(qc, backend)
            self._circuit_cache_key = cache_key
            self._circuit_cache = (qc_t, backend)
            self._last_transpiled = qc_t
        try:
            return backend.run(qc_t, **self._run_kwargs()).result()
        except KeyboardInterrupt:
            self.io.writeln("\n?INTERRUPTED")
            raise
        except Exception as _sim_err:
            _err_msg = str(_sim_err).lower()
            if 'gpu' in _err_msg and 'not supported' in _err_msg:
                self.io.writeln("?GPU EXECUTION FAILED — falling back to CPU")
                self.sim_device = 'CPU'
                self._circuit_cache_key = None
                self._circuit_cache = None
                backend_opts.pop('device', None)
                backend = AerSimulator(**backend_opts)
                qc_t = self._transpile_routed(qc, backend)
                return backend.run(qc_t, **self._run_kwargs()).result()
            elif 'stabilizer' in _err_msg or 'invalid parameters' in _err_msg:
                self._circuit_cache_key = None
                self._circuit_cache = None
                sv_opts = {k: v for k, v in backend_opts.items() if k != 'method'}
                sv_opts['method'] = 'statevector'
                self.io.writeln("  (stabilizer failed — falling back to statevector)")
                backend = AerSimulator(**sv_opts)
                qc_t = self._transpile_routed(qc, backend)
                return backend.run(qc_t, **self._run_kwargs()).result()
            raise

    def _start_run(self) -> tuple[float, list[int]] | None:
        """Open a RUN of the stored program: (start time, sorted line
        numbers) once the method can run it and the program compiles (block
        structure and GOTO/GOSUB targets checked up front), or None after
        saying why it cannot run."""
        if not self.program:
            self.io.writeln("NOTHING TO RUN")
            return None
        _unsupported_gpu = {'stabilizer', 'extended_stabilizer', 'unitary', 'superop'}
        if self.sim_device == 'GPU' and self.sim_method in _unsupported_gpu:
            self.io.writeln(f"?METHOD '{self.sim_method}' does not support GPU — "
                           f"use METHOD statevector or density_matrix")
            return None
        t0 = time.time()
        self._gosub_stack = []
        self._collect_data()
        sorted_lines = sorted(self.program.keys())
        self._scan_subs(self._compiled())
        return t0, sorted_lines

    def _simulate(self, t0: float, sorted_lines: list[int]) -> RunReport | None:
        """Run the opened program, as one circuit on Aer or live shot by shot,
        printing its output and summary line, and report what is left to show;
        None when the run failed or printed everything itself."""
        # A program that reads measurement results as numbers runs live.
        self._live_needed = False
        if self._needs_live(sorted_lines):
            return self._run_live(t0, sorted_lines)

        # Build circuit; its output is held so a build that turns out to need
        # a live run prints nothing twice.
        held = BufferedIO(self.io)
        real_io, self.io = self.io, held
        try:
            qc, has_measure = self.build_circuit()
        except KeyboardInterrupt:
            self.io = real_io
            held.flush()
            self.io.writeln("\n?INTERRUPTED")
            return None
        except Exception as e:
            self.io = real_io
            if self._live_needed:
                held.discard()
                return self._run_live(t0, sorted_lines, carry=held)
            held.flush()
            self.last_circuit = None
            # ON ERROR traps errors inside the build loop (ExecutorMixin
            # ._build_lines); an error that reaches here is untrapped.
            self.io.writeln(f"?BUILD ERROR: {e}")
            return None
        finally:
            self.io = real_io
        if self._live_needed:
            held.discard()
            return self._run_live(t0, sorted_lines, carry=held)
        held.flush()

        qc_sv = qc.copy()
        method, precision = self._select_method(qc_sv)
        if method == 'stabilizer' and not self._is_clifford(qc_sv):
            self.io.writeln("  (METHOD stabilizer runs Clifford circuits only: this one runs on the statevector)")
            method = 'statevector'
        try:
            self._check_fits(method, precision)
        except QBasicRangeError as e:
            self.io.writeln(f"?{e.message}")
            return None

        capture = 'none'
        run_qc = qc
        if self.sim_method in ('unitary', 'superop'):
            qc.save_unitary(label='unitary') if self.sim_method == 'unitary' else qc.save_superop(label='superop')
        elif has_measure:
            capture = self._capture_mode(method, precision)
            if capture == 'mid':
                run_qc = qc.copy()
                run_qc.save_statevector(label='qb_state')
                self._add_final_measurement(run_qc)
            self._add_final_measurement(qc)

        self.last_circuit = qc
        self._last_transpiled = None

        # No-MEASURE path
        if not has_measure and self.sim_method not in ('unitary', 'superop'):
            self._run_no_measure(qc, qc_sv, t0, precision)
            return None

        # Run with shots
        backend_opts = self._build_backend_opts(method, precision)
        try:
            result = self._run_with_fallback(run_qc, backend_opts, method, capture)
        except KeyboardInterrupt:
            return None
        except Exception as e:
            _err = str(e).lower()
            if 'gpu' in _err:
                subsys = 'device/GPU'
            elif 'noise' in _err or 'kraus' in _err:
                subsys = 'noise'
            elif 'stabilizer' in _err:
                subsys = 'backend/stabilizer'
            elif 'mps' in _err or 'matrix_product' in _err:
                subsys = 'backend/MPS'
            else:
                subsys = f'backend/{method}'
            self.io.writeln(f"?RUNTIME ERROR [{subsys}]: {e}")
            return None

        # Extract results
        if method in ('unitary', 'superop'):
            self.last_counts = None
            data = result.data()
            label = method
            mat = data.get(label)
            if mat is not None:
                mat_np = np.asarray(mat)
                self.variables[label] = mat_np
                dim = mat_np.shape[0]
                self.io.writeln(f"\n  {label.upper()} ({dim}x{dim}):")
                if dim <= 16:
                    for i in range(dim):
                        row = '  '.join(f"{v.real:+.3f}{v.imag:+.3f}j" for v in mat_np[i])
                        self.io.writeln(f"    {row}")
                else:
                    self.io.writeln(f"    (too large — stored in variable '{label}')")
        else:
            try:
                self.last_counts = dict(result.get_counts())
            except Exception:
                self.io.writeln(f"  (method '{method}' produced no counts: running on the statevector)")
                method = 'statevector'
                self._circuit_cache_key = None
                sv_opts = {k: v for k, v in backend_opts.items() if k != 'method'}
                sv_opts['method'] = 'statevector'
                sv_backend = AerSimulator(**sv_opts)
                sv_qc = transpile(qc, sv_backend, optimization_level=self._transpile_opt_level)
                result = sv_backend.run(sv_qc, **self._run_kwargs()).result()
                self.last_counts = dict(result.get_counts())
            # Dynamic-circuit runs carry extra mid-circuit MEAS registers, so
            # get_counts returns space-separated per-register keys. The final
            # MEASURE register is added last (leftmost token); keep only it so
            # the reported histogram is the actual end-of-circuit outcome.
            if self._classical_bits and self.last_counts and \
                    any(' ' in k for k in self.last_counts):
                collapsed: dict[str, int] = {}
                for key, cnt in self.last_counts.items():
                    outcome = key.split()[0]
                    collapsed[outcome] = collapsed.get(outcome, 0) + cnt
                self.last_counts = collapsed

        self._extract_save_results(result)
        self._keep_state(result, qc_sv, method, precision, capture)
        self._finalize_run(qc, method, t0, precision)

        # Display results with execution metadata
        m = self._run_manifest
        assert m is not None     # _finalize_run records it
        _meta_parts = [f"method={m['method']}"]
        if m.get('precision') == 'single':
            _meta_parts.append("precision=single")
        if m['device'] != 'CPU':
            _meta_parts.append(f"device={m['device']}")
        if self.noise.model is not None:
            _noise_tag = f"noise=depol({m['noise_depol_p']})" if m['noise_depol_p'] > 0 else "noise=on"
            _meta_parts.append(_noise_tag)
        _meta = ', '.join(_meta_parts)
        _throughput = f", {m['gates']/m['time_s']:.0f} gates/s" if m['time_s'] > 0.001 and m['gates'] >= 1000 else ""
        _complexity = self._circuit_complexity()
        _cx_str = f", {_complexity}" if _complexity else ""
        self.io.writeln(f"\nRAN {len(self.program)} lines, {self.num_qubits} qubits, "
                        f"{self.shots} shots in {m['time_s']:.2f}s  "
                        f"[depth={m['depth']}, gates={m['gates']}{_throughput}, {_meta}{_cx_str}]")
        qt = getattr(self, '_last_transpiled', None)
        if (self._coupling_map or self._basis_gates or self._device_target is not None) and qt is not None:
            swaps = qt.count_ops().get('swap', 0)
            _basis = f", basis={'+'.join(self._basis_gates)}" if self._basis_gates else ""
            _topo = f", swaps={swaps}" if self._coupling_map else ""
            self.io.writeln(f"  routed onto device: depth={qt.depth()}, gates={qt.size()}{_topo}{_basis}")
        report = RunReport(logical=bool(self._logical_mode),
                           bell=m['time_s'] > 2.0 and sys.stdout.isatty())
        if method in ('unitary', 'superop'):
            return report                           # the matrix is shown above
        if has_measure and self.last_counts:
            report.counts, report.display = self.last_counts, True
        else:
            report.note = "(no MEASURE in program \u2014 use STATE or PROBS to inspect)"
        return report

    def result(self) -> dict:
        """Structured result of the last run, for headless/agent/JSON callers.

        JSON-serializable: counts, qubit/shot config, user variables, the
        statevector (when small enough), and key run-manifest fields.
        """
        _big = getattr(self, '_endian_big', False)
        _counts = self.last_counts or {}
        if _big and _counts:
            _counts = {k[::-1]: v for k, v in _counts.items()}
        out: dict = {
            'counts': _counts,
            'num_qubits': self.num_qubits,
            # Bitstring keys follow the active OPTION ENDIAN convention; the
            # bit_order field is the self-describing record of which one.
            'shots': self.shots,
            'bit_order': ('big-endian (qubit 0 = leftmost bit)' if _big
                          else 'little-endian (qubit 0 = rightmost bit)'),
        }
        uvars = {k: v for k, v in self.variables.items()
                 if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
        if uvars:
            out['variables'] = uvars
        # Only small states go into JSON; reading a deferred state that would
        # be dropped anyway would cost a simulation for nothing.
        sv = self.last_sv if self.num_qubits <= 8 else None
        if sv is not None:
            svf = np.ascontiguousarray(sv).ravel()
            if svf.size <= 256:  # keep JSON output bounded
                out['statevector'] = [[float(a.real), float(a.imag)] for a in svf]
        manifest = getattr(self, '_run_manifest', None)
        if manifest:
            out['method'] = manifest.get('method')
            if manifest.get('engine'):
                out['engine'] = manifest['engine']
            out['depth'] = manifest.get('depth')
            out['gates'] = manifest.get('gates')
        return out

    def cmd_sample(self, rest: str = '') -> None:
        """SAMPLE [shots] — sample the program through Aer's SamplerV2 primitive:
        the program's MEASURE subset (or every qubit), with the active noise
        model, precision and SEED."""
        if not self.program:
            self.io.writeln("NOTHING TO SAMPLE")
            return
        shots = int(rest.strip()) if rest.strip() else self.shots
        try:
            qc, _ = self.build_circuit()
            method, precision = self._select_method(qc)
            self._add_final_measurement(qc)
            opts = self._aer_options(method, include_noise=True, precision=precision)
            qc_t = self._transpile_routed(qc, AerSimulator(**opts))
            from qiskit_aer.primitives import SamplerV2
            sampler = SamplerV2(default_shots=shots, seed=self._seed,
                                options={'backend_options': opts})
            pub_result = sampler.run([qc_t]).result()[0]
            register = 'sub' if self._measure_subset else 'meas'
            counts = dict(getattr(pub_result.data, register).get_counts())
            self.last_counts = counts
            self.io.writeln(f"SAMPLED {shots} shots ({len(counts)} unique outcomes)")
            if counts:
                self.print_histogram(counts)
        except Exception as e:
            self.io.writeln(f"?SAMPLE ERROR: {e}")

    def cmd_estimate(self, rest: str) -> None:
        """ESTIMATE <pauli> [qubits] — expectation value through Aer's EstimatorV2 primitive, with the active noise model, precision and SEED."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: ESTIMATE <Z|ZZ|XY|...> [qubits]")
            return
        pauli_str = parts[0].upper()
        qubits = [int(q) for q in parts[1:]] if len(parts) > 1 else list(range(len(pauli_str)))
        try:
            from qiskit.quantum_info import SparsePauliOp
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit_aer.primitives import EstimatorV2
            qc, _ = self.build_circuit()
            full_pauli = ['I'] * self.num_qubits
            for i, p in enumerate(pauli_str):
                if i < len(qubits):
                    full_pauli[self.num_qubits - 1 - qubits[i]] = p
            op = SparsePauliOp(''.join(full_pauli))
            method, precision = self._select_method(qc)
            options: dict[str, Any] = {
                'backend_options': self._aer_options(method, include_noise=True, precision=precision)}
            if self._seed is not None:
                options['run_options'] = {'seed': self._seed}
            estimator = EstimatorV2(options=options)
            pm = generate_preset_pass_manager(optimization_level=0)
            qc_t = pm.run(qc)
            result = estimator.run([(qc_t, op)]).result()
            val = result[0].data.evs
            self.io.writeln(f"  <{pauli_str}> on qubits {qubits} = {float(val):.6f}")
        except Exception as e:
            self.io.writeln(f"?ESTIMATE ERROR: {e}")

    def cmd_step(self) -> None:
        """STEP — step through the program a line at a time, showing the state after each.

        It runs as one live shot: MEAS collapses the state and IF, PRINT and
        LET see the outcome."""
        if not self.program:
            self.io.writeln("NOTHING TO STEP")
            return

        from qubasic_core import simtools as cap
        from qubasic_core.engine import ExecContext
        from qubasic_core.expression import Scope
        peak = cap.numpy_peak_bytes(self.num_qubits, self.sim_precision)
        if not cap.fits(peak):
            self.io.writeln(f"?STEP: a {self.num_qubits}-qubit state needs ~{cap.fmt_gb(peak)}, "
                            f"{cap.fmt_gb(cap.available_bytes())} available")
            return
        prog = self._compiled()
        self._gosub_stack = []
        self._collect_data()
        self._scan_subs(prog)
        state = LiveState(self.num_qubits, self.sim_precision, self._live_noise())
        self._prepare_run(state.qc)
        ctx = ExecContext(
            sorted_lines=prog, ip=0,
            run_vars=Scope(self.variables),
            max_iterations=self._loop_limit(), qc=state.qc,
            backend=LiveBackend(state, self._custom_gates),
        )
        self._live = state
        prev_strict, self._strict_names = self._strict_names, False
        try:
            self._step_lines(ctx, state)
        finally:
            self._live = None
            self._strict_names = prev_strict
            self._step_auto = False

    def _step_lines(self, ctx: ExecContext, state: LiveState) -> None:
        prog = sorted_lines = ctx.sorted_lines

        self.io.writeln(f"STEP MODE \u2014 {len(self.program)} lines, {self.num_qubits} qubits")
        self.io.writeln("Press ENTER to advance, A for auto-play, Q to quit\n")

        while ctx.ip < len(sorted_lines):
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            line_num = prog.lines[ctx.ip]
            stmt = prog.texts[ctx.ip]
            parsed = prog.stmts[ctx.ip]

            # Display current statement
            self.io.writeln(f">> {line_num:5d}  {stmt}")

            # Execute it, then show the state it leaves.
            try:
                result = self._exec_line(stmt, parsed=parsed, ctx=ctx)
                sv = state.sv
            except Exception as e:
                self.io.writeln(f"?LINE {line_num}: {e}")
                return
            self.last_sv = sv
            self._checkpoint_sv(line_num)
            self._print_sv_compact(sv)

            # Wait for input (or auto-advance)
            if not getattr(self, '_step_auto', False):
                try:
                    user = self.io.read_line("   [ENTER/A/Q] ").strip().upper()
                    if user == 'Q':
                        self.io.writeln("STOPPED")
                        return
                    if user == 'A' or user.startswith('A'):
                        self._step_auto = True
                        delay = 0.5
                        if len(user) > 1:
                            try:
                                delay = float(user[1:]) / 1000
                            except ValueError:
                                pass
                        self._step_delay = delay
                except (KeyboardInterrupt, EOFError):
                    self.io.writeln("\nSTOPPED")
                    return
            if getattr(self, '_step_auto', False):
                import time as _time
                _time.sleep(getattr(self, '_step_delay', 0.5))

            if result is ExecResult.END:
                break
            self._advance(ctx, result)

        self.io.writeln("DONE")

    def _is_clifford(self, qc: 'QuantumCircuit') -> bool:
        """Check if circuit contains only Clifford gates.

        Custom/unitary gates are conservatively treated as non-Clifford.
        Verifying that an arbitrary unitary normalizes the Pauli group is
        expensive, and the consequence of a false positive (stabilizer sim
        producing wrong results) is worse than a false negative (falling
        back to statevector sim, which always works).
        """
        clifford_gates = {'h', 'x', 'y', 'z', 's', 'sdg', 'cx', 'cz', 'cy', 'ecr',
                          'swap', 'id', 'sx', 'sxdg', 'barrier', 'delay', 'measure', 'reset'}
        for inst in qc.data:
            name = inst.operation.name.lower()
            if name == 'rz':
                try:
                    turns = float(inst.operation.params[0]) / (np.pi / 2)
                except (TypeError, ValueError):
                    return False          # an unbound parameter
                if abs(turns - round(turns)) > 1e-9:
                    return False
                continue
            if name == 'unitary':
                return False  # custom gates: conservatively non-Clifford
            if name not in clifford_gates:
                return False
        return True

    def cmd_rewind(self, rest: str = '') -> None:
        """REWIND [N] — go back N steps in the statevector history."""
        if not self.debug.checkpoints:
            self.io.writeln("?NO CHECKPOINTS — use STEP mode to build history")
            return
        n = int(rest.strip()) if rest.strip() else 1
        new_pos = max(0, self.debug.position - n)
        self.debug.position = new_pos
        line_num, sv = self.debug.checkpoints[new_pos]
        self.last_sv = sv.copy()
        self.io.writeln(f"REWIND to step {new_pos} (line {line_num})")
        self._print_sv_compact(sv)

    def cmd_forward(self, rest: str = '') -> None:
        """FORWARD [N] — go forward N steps in the statevector history."""
        if not self.debug.checkpoints:
            self.io.writeln("?NO CHECKPOINTS")
            return
        n = int(rest.strip()) if rest.strip() else 1
        new_pos = min(len(self.debug.checkpoints) - 1, self.debug.position + n)
        self.debug.position = new_pos
        line_num, sv = self.debug.checkpoints[new_pos]
        self.last_sv = sv.copy()
        self.io.writeln(f"FORWARD to step {new_pos} (line {line_num})")
        self._print_sv_compact(sv)


# ---------------------------------------------------------------------------
# QUBASIC live execution: programs that read measurement results as numbers.
#
# A program whose classical code reads a mid-circuit measurement bit (PRINT m,
# LET y = m + 1, a GOTO or loop chosen by it) cannot be one Qiskit circuit,
# because the value differs from shot to shot. Such a program runs live: each
# shot executes the program from the top on a numpy statevector, MEAS
# collapses the state and returns its outcome, and IF, PRINT and LET see real
# bits. Gates the executor appends to the shot's circuit are applied to the
# state when a measurement or a state read needs them. An active noise model
# runs as quantum trajectories (gate errors as Kraus jumps, readout error as
# classical flips).
#
# Under METHOD stabilizer, or under METHOD automatic when the statevector does
# not fit in memory, each shot holds a stabilizer tableau instead, which costs
# O(n^2) bits at any width: Clifford gates and Pauli noise channels run on it,
# SAVE_EXPECT and SAVE_PROBS read it exactly, and any other operation stops
# the run with the reason the run holds a tableau.
#
# Only the first shot prints. Later shots replay the first shot's INPUT
# answers and skip file output, so a run reads as one execution of the
# program followed by the histogram over every shot.
# ---------------------------------------------------------------------------

# ── IO wrappers ─────────────────────────────────────────────────────────

class ShotIO:
    """IO for shot-by-shot execution. The first shot writes through and
    records every line read; after start_replay() output is dropped and the
    recorded lines are returned in order. `answers` and `skip` carry what an
    abandoned circuit build already read and showed."""

    def __init__(self, io: Any, answers: list[str] | None = None, skip: int = 0) -> None:
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

    def __init__(self, io: Any) -> None:
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

    def __init__(self, error: Any) -> None:
        from qiskit.quantum_info import Kraus
        ops = [np.asarray(k, dtype=complex) for k in Kraus(error).data]
        self.paulis = pauli_mixture(ops)          # what a stabilizer tableau samples
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

    def __init__(self, model: Any) -> None:
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


def _numeric(params: Sequence[Any]) -> bool:
    try:
        for p in params:
            float(p)
        return True
    except (TypeError, ValueError):
        return False


class _ShotState:
    """What the two live states share: the circuit fragment the executor
    appends to, applied in order by flush()."""

    qc: Any

    def apply_operation(self, op: Any, qubits: list[int]) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        """Apply the instructions queued in qc, in order, and clear it."""
        qc = self.qc
        if len(qc.data) == 0:
            return
        index = {bit: i for i, bit in enumerate(qc.qubits)}
        for inst in list(qc.data):
            self.apply_operation(inst.operation, [index[q] for q in inst.qubits])
        qc.clear()


class LiveState(_ShotState):
    """The statevector of one live shot plus the circuit fragment the
    executor is appending to. Built-in gates apply at once; everything else
    queues in `qc` until flush(). Qubit q is bit q of the amplitude index."""

    engine = 'statevector'

    def __init__(self, n: int, precision: str = 'double', noise: LiveNoise | None = None) -> None:
        from qiskit import QuantumCircuit
        self.n = n
        self.dtype = np.complex64 if precision == 'single' else np.complex128
        self._sv: np.ndarray = np.zeros(1 << n, dtype=self.dtype)
        self._sv[0] = 1.0
        self.qc = QuantumCircuit(n)
        self.noise = noise
        self.ops = 0

    def reset_state(self, init: Any = None) -> None:
        if init is None:
            self._sv[...] = 0
            self._sv[0] = 1.0
        else:
            np.copyto(self._sv, init)
        self.qc.clear()

    def snapshot(self) -> np.ndarray:
        return self.sv.copy()

    @property
    def sv(self) -> np.ndarray:
        self.flush()
        return self._sv

    # Reading the state (SAVE_EXPECT, SAVE_PROBS, SAVE_AMPS)

    def expectation(self, op: Any) -> float:
        from qiskit.quantum_info import Statevector
        return float(np.real(Statevector(self.sv).expectation_value(op)))

    def probabilities(self, qubits: list[int]) -> np.ndarray:
        from qiskit.quantum_info import Statevector
        return Statevector(self.sv).probabilities(qubits)

    def amplitudes(self, indices: list[int]) -> np.ndarray:
        return np.array(self.sv[indices], dtype=complex)

    # Gates

    def apply_named(self, name: str, params: tuple, qubits: list[int]) -> None:
        """A built-in gate (GATE_TABLE name), MSB-first over `qubits`."""
        canonical = GATE_ALIASES.get(name, name)
        _apply_gate_inplace(self._sv, _np_gate_matrix(canonical, tuple(params)),
                            list(qubits), self.n)
        self.ops += 1
        self._noise_after(canonical.lower(), qubits)

    def apply_matrix(self, matrix: np.ndarray, qubits: list[int], name: str = 'unitary',
                     label: str | None = None) -> None:
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
        from qubasic_core.simtools import subsystem_rdm
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


_TABLEAU_GATES: dict[str, Callable[..., None]] = {
    'H': Tableau.h, 'S': Tableau.s, 'SDG': Tableau.sdg, 'X': Tableau.pauli_x,
    'Y': Tableau.pauli_y, 'Z': Tableau.pauli_z, 'CX': Tableau.cx, 'CZ': Tableau.cz,
    'SWAP': Tableau.swap,
}


class TableauState(_ShotState):
    """One live shot on a stabilizer tableau, for a Clifford program wider
    than a statevector fits or run under METHOD stabilizer. Clifford gates
    and Pauli noise update the tableau; anything else stops the run with
    `why`, the reason it holds one. Operations queue in `qc` as in LiveState."""

    engine = 'stabilizer tableau'

    def __init__(self, n: int, noise: LiveNoise | None, why: str) -> None:
        from qiskit import QuantumCircuit
        self.n = n
        self.tab = Tableau(n)
        self.qc = QuantumCircuit(n)
        self.noise = noise
        self.ops = 0
        self.why = why

    def _refuse(self, what: str) -> ValueError:
        return ValueError(f"{what}, and this live run holds a stabilizer tableau ({self.why})")

    def reset_state(self, init: Any = None) -> None:
        if init is None:
            self.tab = Tableau(self.n)
        else:
            self.tab.assign(init)
        self.qc.clear()

    def snapshot(self) -> Tableau:
        self.flush()
        return self.tab.copy()

    @property
    def sv(self) -> np.ndarray:
        raise self._refuse("the statevector is not held")

    # Gates

    def _clifford(self, matrix: np.ndarray, qubits: list[int], label: str) -> None:
        """A unitary in Qiskit order over `qubits` (qubits[0] least significant)."""
        table = clifford_table(matrix) if len(qubits) <= 4 else None
        if table is None:
            raise self._refuse(f"{label} is not a Clifford gate")
        self.tab.apply_table(table, qubits)

    def apply_named(self, name: str, params: tuple, qubits: list[int]) -> None:
        """A built-in gate (GATE_TABLE name), MSB-first over `qubits`."""
        canonical = GATE_ALIASES.get(name, name)
        fast = None if params else _TABLEAU_GATES.get(canonical)
        if fast is not None:
            fast(self.tab, *qubits)
        elif canonical != 'ID':
            label = f"{canonical}({', '.join(f'{float(p):g}' for p in params)})" if params else canonical
            self._clifford(_np_gate_matrix(canonical, tuple(params)), list(qubits)[::-1], label)
        self.ops += 1
        self._noise_after(canonical.lower(), qubits)

    def apply_matrix(self, matrix: np.ndarray, qubits: list[int], name: str = 'unitary',
                     label: str | None = None) -> None:
        """A unitary in Qiskit order over `qubits` (qubits[0] least
        significant); `label` names it in a refusal."""
        self._clifford(np.asarray(matrix, dtype=complex), list(qubits), label or name.upper())
        self.ops += 1
        self._noise_after(name, qubits)

    def apply_operation(self, op: Any, qubits: list[int]) -> None:
        name = op.name
        if name in ('barrier', 'delay', 'id'):
            return
        if name == 'measure':
            self.measure(qubits[0])
            return
        if name == 'reset':
            self.reset(qubits[0])
            return
        if name == 'kraus':
            mix = pauli_mixture([np.asarray(k, dtype=complex) for k in op.params])
            if mix is None:
                raise self._refuse("the channel is not a Pauli channel")
            self._sample_pauli(mix, qubits)
            return
        if name in ('set_statevector', 'set_density_matrix'):
            raise self._refuse("SET_STATE needs the statevector")
        if name in _CONTROL_FLOW_OPS or name.startswith('save_'):
            raise ValueError(f"'{name}' is not available in a live run")
        upper = name.upper()
        if upper in _GATE_BUILDERS and _numeric(op.params):
            self.apply_named(upper, tuple(float(p) for p in op.params), qubits)
            return
        matrix = None
        if op.num_qubits <= 4:
            try:
                matrix = op.to_matrix()
            except Exception:
                matrix = None
        if matrix is not None:
            self.apply_matrix(matrix, qubits, name)
            return
        definition = getattr(op, 'definition', None)
        if definition is None:
            raise self._refuse(f"{upper} has no Clifford form")
        sub = {bit: i for i, bit in enumerate(definition.qubits)}
        for inst in definition.data:
            self.apply_operation(inst.operation, [qubits[sub[q]] for q in inst.qubits])

    # Noise: Pauli channels sampled per gate, readout error on recorded bits

    def _sample_pauli(self, mix: tuple[np.ndarray, np.ndarray], qubits: list[int]) -> None:
        cum, codes = mix
        i = min(int(np.searchsorted(cum, np.random.random(), side='right')), len(codes) - 1)
        self.tab.apply_pauli(int(codes[i]), qubits)

    def _noise_after(self, name: str, qubits: list[int]) -> None:
        if self.noise is None:
            return
        ch = self.noise.channel(name, list(qubits))
        if ch is None:
            return
        if ch.paulis is None:
            raise self._refuse(f"the noise on {name.upper()} is not a Pauli channel")
        self._sample_pauli(ch.paulis, list(qubits))

    def _recorded(self, qubit: int, bit: int) -> int:
        if self.noise is not None:
            p = self.noise.readout(qubit)
            if p is not None and np.random.random() < p[bit][1 - bit]:
                return 1 - bit
        return bit

    def measure(self, qubit: int) -> int:
        self.flush()
        self._noise_after('measure', [qubit])
        return self._recorded(qubit, self.tab.measure(qubit))

    def reset(self, qubit: int) -> None:
        self.flush()
        self.tab.reset(qubit)
        self._noise_after('reset', [qubit])

    def sample_final(self, subset: list[int] | None) -> str:
        """The final MEASURE of a shot, as in LiveState.sample_final: drawn
        from a copy, so the shot's state stays as the program left it."""
        self.flush()
        qubits = list(subset) if subset else list(range(self.n))
        if self.noise is not None:
            for q in qubits:
                self._noise_after('measure', [q])
        final = self.tab.copy()
        bits = {q: final.measure(q) for q in qubits}
        bits = {q: self._recorded(q, b) for q, b in bits.items()}
        return ''.join(str(bits[q]) for q in reversed(qubits))

    # Reading the state

    def expectation(self, op: Any) -> float:
        """<op> for a SparsePauliOp: each Pauli term is 0 or +-1 on a
        stabilizer state."""
        self.flush()
        total = 0.0
        for pauli, coeff in zip(op.paulis, op.coeffs):
            value = self.tab.expectation(np.asarray(pauli.x), np.asarray(pauli.z))
            total += float(np.real(coeff * (-1j) ** int(pauli.phase))) * value
        return total

    def probabilities(self, qubits: list[int]) -> np.ndarray:
        if len(qubits) > 20:
            raise self._refuse(f"SAVE_PROBS over {len(qubits)} qubits lists 2^{len(qubits)} outcomes")
        self.flush()
        return self.tab.probabilities(qubits)

    def amplitudes(self, indices: list[int]) -> np.ndarray:
        raise self._refuse("SAVE_AMPS reads amplitudes")


class LiveBackend:
    """Executor backend for a live shot: built-in gates go straight to the
    state, custom gates as their matrices, and other circuit operations
    queue in the shot's circuit."""

    def __init__(self, state: LiveState | TableauState, custom_gates: dict[str, np.ndarray]) -> None:
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
            self.state.apply_matrix(self._custom[canonical], list(qubits), label=canonical)
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


class LiveMixin(SimulationHost):
    """Live (shot-by-shot) execution for QBasicTerminal."""

    def _init_live(self) -> None:
        self._live: LiveState | TableauState | None = None
        self._live_needed = False
        self._live_saves: dict[str, tuple[str, list]] = {}
        self._last_tableau: Tableau | None = None

    # Static check

    def _bit_names(self, texts: list[str]) -> tuple[set[str], bool]:
        names: set[str] = set()
        basis = False
        for t in texts:
            names.update(m.group(1) for m in _RE_BIT_TARGET.finditer(t))
            if re.search(r'\bMEASURE_[XYZ]\b', t, re.IGNORECASE):
                basis = True
        return names, basis

    def _needs_live(self, sorted_lines: Sequence[int]) -> bool:
        """Whether the program reads a measurement bit anywhere a compiled
        dynamic circuit cannot: anything but IF <bit> THEN <gates> [ELSE
        <gates>] on one bit."""
        from qubasic_core.parser import CompoundStmt, IfThenStmt, parse_stmt, strip_comment
        texts = [strip_comment(self.program[ln]) for ln in sorted_lines]
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
        model = self.noise.model
        if model is None and self.noise.per_qubit:
            model = self._build_qubit_noise()
        return LiveNoise(model) if model is not None else None

    def _live_prefix(self, sorted_lines: Any = None) -> int:
        """Number of leading statements that run the same in every shot:
        plain gates, assignments and PRINTs with no jumps, input, randomness
        or measurement. They run once and each shot starts after them."""
        from qubasic_core.parser import BarrierStmt, GateStmt, LetStmt, PrintStmt, RemStmt
        prog = self._compiled()
        k = 0
        for p, text in zip(prog.stmts, prog.texts):
            if not isinstance(p, (GateStmt, LetStmt, PrintStmt, RemStmt, BarrierStmt)):
                break
            if re.search(r'\b(RND|TIMER|INKEY|PEEK|INPUT|FN)\b', text, re.IGNORECASE):
                break
            k += 1
        return k

    def _record_save(self, kind: str, var: str, value: Any) -> None:
        self._live_saves.setdefault(var, (kind, []))[1].append(value)

    def _run_live(self, t0: float, sorted_lines: Sequence[int],
                  carry: BufferedIO | None = None) -> RunReport | None:
        """Run the program shot by shot (see the module docstring) and report
        its histogram, or the note for a run without MEASURE."""
        from qubasic_core import simtools as cap
        n = self.num_qubits
        prec = self.sim_precision
        if self._pending_set_density is not None:
            self.io.writeln("?A program that reads measurement results runs shot by shot "
                            "on a pure state; use SET_STATE rather than SET_DENSITY")
            return None
        peak = cap.numpy_peak_bytes(n, prec)
        why = None                        # why the run holds a stabilizer tableau
        if self.sim_method == 'stabilizer':
            why = "METHOD stabilizer"
        elif not cap.fits(peak):
            if self.sim_method != 'automatic':
                self.io.writeln(f"?MEMORY: a live run at {n} qubits needs ~{cap.fmt_gb(peak)}, "
                                f"{cap.fmt_gb(cap.available_bytes())} available")
                return None
            why = f"a {n}-qubit statevector needs ~{cap.fmt_gb(peak)}"
        if why is not None and not cap.fits(cap.tableau_bytes(n)):
            self.io.writeln(f"?MEMORY: a live run at {n} qubits needs ~{cap.fmt_gb(cap.tableau_bytes(n))} "
                            f"on the stabilizer tableau, {cap.fmt_gb(cap.available_bytes())} available")
            return None
        has_measure = self._program_has_measure(sorted_lines)
        shots = self.shots if has_measure else 1
        noise = self._live_noise()
        state: LiveState | TableauState = (LiveState(n, prec, noise) if why is None
                                           else TableauState(n, noise, why))
        backend = LiveBackend(state, self._custom_gates)
        shot_io = ShotIO(self.io, carry.answers if carry else None, carry.shown if carry else 0)
        real_io = self.io
        base_vars = dict(self.variables)
        base_arrays = {k: list(v) for k, v in self.arrays.items()}
        keep_copy = shots > 1 and (why is not None or cap.fits(cap.numpy_peak_bytes(n, prec, True)))
        if self._seed is not None:
            np.random.seed(self._seed)
        self._live_saves = {}
        counts: dict[str, int] = {}
        first_vars: dict[str, Any] = {}
        first_arrays: dict[str, Any] = {}
        first_sv = None
        first_tab = None
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
                snapshot = state.snapshot()
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
                self.data.ptr = 0
                self.build_circuit(backend=backend, start_ip=start)
                if has_measure:
                    key = state.sample_final(self._measure_subset)
                    counts[key] = counts.get(key, 0) + 1
                if shot == 0:
                    first_vars = dict(self.variables)
                    first_arrays = {k: list(v) for k, v in self.arrays.items()}
                    first_ops = prefix_ops + state.ops
                    if isinstance(state, TableauState):
                        first_tab = state.snapshot()
                    elif shots == 1:
                        first_sv = state.sv
                    elif keep_copy:
                        first_sv = state.sv.copy()
                if progress and (shot + 1) % max(1, shots // 10) == 0:
                    real_io.write(f"  {100 * (shot + 1) // shots}% ({shot + 1}/{shots} shots)\r")
        except KeyboardInterrupt:
            self.io = real_io
            self.io.writeln("\n?INTERRUPTED")
            return None
        except Exception as e:
            self.io = real_io
            self.io.writeln(f"?RUNTIME ERROR: {e}")
            return None
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
        self._last_tableau = first_tab
        self.last_counts = counts if has_measure else None
        self.last_circuit = None
        dt = time.time() - t0
        self._update_status(gate_count=first_ops, circuit_depth=0, run_time_ms=dt * 1000)
        self.variables['_TIME'] = dt
        self._run_manifest = {
            'program': dict(self.program), 'num_qubits': n, 'shots': shots,
            'method': 'live', 'engine': state.engine, 'precision': prec, 'device': 'CPU',
            'seed': self._seed, 'noise_depol_p': self.noise.depol_p, 'depth': 0,
            'gates': first_ops, 'time_s': dt,
        }
        tags = ["method=live"]
        if first_tab is not None:
            tags.append("stabilizer tableau")
        elif prec == 'single':
            tags.append("precision=single")
        if noise is not None:
            tags.append("noise=trajectories")
        shot_word = f"{shots} shot{'s' if shots != 1 else ''}"
        self.io.writeln(f"\nRAN {len(self.program)} lines, {n} qubits, {shot_word} in {dt:.2f}s  "
                        f"[{first_ops} gates/shot, {', '.join(tags)}]")
        if has_measure and counts:
            return RunReport(counts=counts)
        if first_tab is not None:
            return RunReport(note="(no MEASURE — one shot; STATE lists the stabilizers, BLOCH reads a qubit)")
        return RunReport(note=None if has_measure else
                         "(no MEASURE — one shot; use STATE, PROBS, or BLOCH to inspect)")

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
                    val = _expectation(val)
            self.variables[var] = val
