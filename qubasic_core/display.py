"""QUBASIC display: help text, histograms and statevector views,
state inspection commands, visualizations and parameter sweeps.
"""

from __future__ import annotations

import math
import re
import sys
import time
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qubasic_core.engine import (
    _RICH,
    AMPLITUDE_THRESHOLD,
    HISTOGRAM_BAR_WIDTH,
    MAX_BLOCH_DISPLAY,
    MAX_DISPLAY_AMPLITUDES,
    MAX_HISTOGRAM_STATES,
    _RichTable,
)
from qubasic_core.host import PresentationHost

if TYPE_CHECKING:
    from rich.console import Console

try:
    import plotille as _plotille
except ImportError:
    _plotille = None


# ---------------------------------------------------------------------------
# QUBASIC help text and banner art.
# ---------------------------------------------------------------------------

HELP_TEXT = """\

QUBASIC — Quantum BASIC Terminal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HELP <name> describes one command, statement, function or gate;
  docs/reference.md lists them all and docs/tutorial.md walks through them.

PROGRAMS
  10 H 0              Enter a line (a line number alone deletes it)
  LIST, EDIT n, DELETE a-b, RENUM, UNDO, NEW, FIND, REPLACE
  RUN, STEP           Run the program, or step through it line by line
  SAVE f, LOAD f      .qb sessions; LOAD and SAVE also take QBASIC .bas source
  H 0 : CX 0, 1       Typed without a number: runs at once on |0...0>

THE BASIC LANGUAGE (QBASIC's)
  x% = 2: s$ = "hi"   Typed names (% & ! # $), DEFINT A-Z, DIM v AS DOUBLE
  TYPE ... END TYPE   Records: DIM p AS Point, crowd(1 TO 3) AS Point
  DIM a(1 TO 3, 0 TO 1), REDIM PRESERVE, LBOUND, UBOUND
  FOR/NEXT, WHILE/WEND, DO/LOOP, IF/ELSEIF/END IF, SELECT CASE, EXIT
  GOTO/GOSUB line or label, SUB, FUNCTION, DEF FN, CALL, STATIC, SHARED
  PRINT [USING], INPUT, LINE INPUT, WRITE, DATA/READ/RESTORE
  OPEN f$ FOR INPUT|OUTPUT|APPEND|RANDOM|BINARY AS #n, FIELD, GET #, PUT #
  ON ERROR GOTO, RESUME, ERROR n, ON TIMER(n) GOSUB, TIMER ON
  SCREEN 13, PSET, LINE, CIRCLE, PAINT, DRAW, SHOW, SAVEPNG f SCREEN
  PLAY "CDE", SOUND f, t, BEEP, SAVEWAV f

QUANTUM STATEMENTS
  H X Y Z S T SDG TDG SX, RX RY RZ P U, CX CZ CY CH SWAP, CRX CP RZZ, CCX CSWAP
  CTRL, INV, UNITARY NAME = [[...]], DEF NAME = gates, REG name n
  MEASURE [qubits]    Measure into the histogram
  MEAS q -> bit       Mid-circuit: IF bit THEN gates feeds forward; any
                      other use of the bit runs the program live, per shot
  QFT, IQFT, DIFFUSE, MCX, QADD, QPE, GRAPHSTATE, EVOLVE H, t, n
  SAVE_EXPECT ZZ 0 1 -> v, SAVE_PROBS, SAVE_AMPS, SET_STATE |BELL>

RESULTS AND ANALYSIS
  HIST, STATE, PROBS, BLOCH [q], DENSITY, CIRCUIT, DISPLAY STATE
  EXPECT ZZ 0 1, ENTROPY, FIDELITY |BELL>, TOMOGRAPHY, SWEEP, MINIMIZE

SIMULATION AND NOISE
  QUBITS n, SHOTS n, SEED n, METHOD name, PRECISION SINGLE, STATUS, RAM
  NOISE depolarizing 0.001 0.01; readout 0.02    (NOISE INFO, NOISE OFF)
  MITIGATE READOUT ON, ZNE 1.0 ZZ 0 1             Error mitigation

ALGORITHMS AND ERROR CORRECTION
  GROVER, SIMON, DJ, BV, QPE H, QCOUNT, SHOR, HHL, QAOA, VQE, AMPEST
  QEC SURFACE 3, LOGICAL_ERROR_RATE ... [CIRCUIT], THRESHOLD, LQUBITS, LATTICE
  RB, XEB, QVOLUME, MIRROR, GST, RESOURCES, DISTILL

MORE
  LOCC 2 2 (SEND, SHARE), QUDIT, BOSONIC, HAMILTONIAN, LINDBLAD
  PEEK/POKE/SYS/MAP (memory map), DEMO LIST, EXPORT, LOADQASM, CSV
"""

BANNER_ART = """\

 ██████╗ ██╗   ██╗██████╗  █████╗ ███████╗██╗ ██████╗
██╔═══██╗██║   ██║██╔══██╗██╔══██╗██╔════╝██║██╔════╝
██║   ██║██║   ██║██████╔╝███████║███████╗██║██║
██║▄▄ ██║██║   ██║██╔══██╗██╔══██║╚════██║██║██║
╚██████╔╝╚██████╔╝██████╔╝██║  ██║███████║██║╚██████╗
 ╚══▀▀═╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝

Quantum BASIC
{info_line}
{config_line}
Type HELP for commands, DEMO LIST for demos.
"""


# ---------------------------------------------------------------------------
# QUBASIC display — histograms, statevector, Bloch sphere rendering.
# ---------------------------------------------------------------------------

def _get_console() -> Console | None:
    """Get a Rich console bound to current sys.stdout (test-safe)."""
    if _RICH:
        from rich.console import Console
        return Console(file=sys.stdout, highlight=False)
    return None


class DisplayMixin(PresentationHost):
    """Display and visualization methods for the QUBASIC terminal."""

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

    def _print_probs(self, sv: np.ndarray, n_qubits: int | None = None) -> None:
        """Print the most probable outcomes with a histogram bar each."""
        from qubasic_core.simtools import top_k
        n = n_qubits if n_qubits is not None else self.num_qubits
        idx, probs, total = top_k(sv, MAX_HISTOGRAM_STATES, AMPLITUDE_THRESHOLD)

        self.io.writeln(f"\n  Probability distribution ({total} non-zero):\n")
        max_p = float(probs[0]) if probs.size else 1.0

        for i, p in zip(idx, probs):
            state = self._bits(format(int(i), f'0{n}b'))
            bar = '\u2588' * int(HISTOGRAM_BAR_WIDTH * p / max_p)
            self.io.writeln(f"  |{state}\u27E9  {p*100:6.2f}%  {bar}")

        if total > MAX_HISTOGRAM_STATES:
            self.io.writeln(f"  ... and {total - MAX_HISTOGRAM_STATES} more states")
        self.io.writeln('')

    def _print_bloch_single(self, sv: np.ndarray, qubit: int, n_qubits: int | None = None) -> None:
        """ASCII Bloch sphere for a single qubit."""
        x, y, z = (round(c, 9) + 0.0 for c in self._bloch_vector(sv, qubit, n_qubits))

        # Determine state label
        if math.sqrt(x**2 + y**2 + z**2) < 0.01:
            label = "maximally mixed"
        elif z > 0.99:
            label = "|0\u27E9 (north pole)"
        elif z < -0.99:
            label = "|1\u27E9 (south pole)"
        elif abs(x - 1) < 0.01:
            label = "|+\u27E9"
        elif abs(x + 1) < 0.01:
            label = "|-\u27E9"
        elif abs(y - 1) < 0.01:
            label = "|+i\u27E9"
        elif abs(y + 1) < 0.01:
            label = "|-i\u27E9"
        else:
            theta = math.acos(max(-1, min(1, z)))
            phi = math.atan2(y, x)
            label = f"\u03B8={theta:.2f} \u03C6={phi:.2f}"

        # Draw ASCII Bloch sphere (XZ plane projection, 15x15)
        R = 6
        W = 2 * R + 1
        grid = [[' '] * W for _ in range(W)]
        cx, cy = R, R

        # Circle
        for angle in range(360):
            rad = math.radians(angle)
            gx = round(cx + R * math.cos(rad))
            gy = round(cy + R * math.sin(rad))
            if 0 <= gx < W and 0 <= gy < W:
                if grid[gy][gx] == ' ':
                    grid[gy][gx] = '\u00B7'

        # Axes
        for i in range(W):
            if grid[cy][i] == ' ':
                grid[cy][i] = '-'
            if grid[i][cx] == ' ':
                grid[i][cx] = '|'
        grid[cy][cx] = '+'

        # State point (XZ projection)
        px = round(cx + x * (R - 1))
        pz = round(cy - z * (R - 1))
        if 0 <= px < W and 0 <= pz < W:
            grid[pz][px] = '\u25CF'

        # Labels on the sphere
        self.io.writeln(f"  Qubit {qubit}  ({x:.3f}, {y:.3f}, {z:.3f})  {label}")
        _ket0 = '|0\u27E9'
        _ket1 = '|1\u27E9'
        self.io.writeln(f"{_ket0:^{W+4}}")
        for row in grid:
            self.io.writeln(f"  {''.join(row)}")
        self.io.writeln(f"{_ket1:^{W+4}}")

    def _bloch_vector(self, sv: np.ndarray, qubit: int, n_qubits: int | None = None) -> tuple[float, float, float]:
        """Bloch vector (x, y, z) of one qubit: <X> = 2 Re(rho_10),
        <Y> = 2 Im(rho_10), <Z> = rho_00 - rho_11, from the reduced state."""
        from qubasic_core.simtools import bloch
        n = n_qubits if n_qubits is not None else self.num_qubits
        if qubit < 0 or qubit >= n:
            raise ValueError(f"QUBIT {qubit} OUT OF RANGE (0-{n - 1})")
        return bloch(sv, qubit, n)

    def _auto_display(self) -> None:
        """Display results using current SCREEN mode after RUN."""
        mode = self._screen_mode
        if mode <= 1:
            return  # text/histogram mode: histogram already shown by cmd_run
        if mode == 2:
            if self.last_sv is not None:
                self._print_statevector(self.last_sv)
        elif mode == 3:
            if self.last_sv is not None:
                from qubasic_core.engine import MAX_BLOCH_DISPLAY
                for q in range(min(self.num_qubits, MAX_BLOCH_DISPLAY)):
                    self._print_bloch_single(self.last_sv, q)
                    self.io.writeln('')
        elif mode == 4:
            self.cmd_density()
        elif mode == 5:
            self.cmd_circuit()


# ---------------------------------------------------------------------------
# QUBASIC state display mixin — STATE, BLOCH, HIST, PROBS, PRINT STATE, DECOMPOSE.
# ---------------------------------------------------------------------------

class StateDisplayMixin(PresentationHost):
    """State inspection commands for QBasicTerminal.

    All state commands use _active_sv / _active_nqubits to read from
    whichever engine (Qiskit or LOCC) last executed, ensuring displays
    always reflect the actual executed state.
    """

    def cmd_state(self, rest: str = '') -> None:
        """STATE [reg] — the statevector of the current state (one register in LOCC SPLIT mode)."""
        if self.locc_mode:
            return self._locc_state(rest)
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        self._print_statevector(sv, self._active_nqubits)

    def cmd_hist(self) -> None:
        """HIST — the last run's measurement histogram."""
        if self.last_counts is None:
            self.io.writeln("?NO RESULTS — RUN first")
            return
        self.print_histogram(self.last_counts)

    def cmd_probs(self) -> None:
        """PROBS — the probability distribution of the current state."""
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        self._print_probs(sv, self._active_nqubits)

    def cmd_bloch(self, rest: str) -> None:
        """BLOCH [qubit] — ASCII Bloch sphere of one qubit, or of each qubit (up to 8); BLOCH <reg> [qubit] in LOCC mode."""
        if self.locc_mode:
            return self._locc_bloch(rest)
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        n = self._active_nqubits
        if rest:
            q = int(rest)
            self._print_bloch_single(sv, q, n)
        else:
            for q in range(min(n, MAX_BLOCH_DISPLAY)):
                self._print_bloch_single(sv, q, n)
                if q < min(n, MAX_BLOCH_DISPLAY) - 1:
                    self.io.writeln('')

    def cmd_circuit(self) -> None:
        """CIRCUIT — draw the last run's circuit as text."""
        if self.last_circuit is None:
            self.io.writeln("?NO CIRCUIT — RUN first")
            return
        try:
            self.io.writeln(self.last_circuit.draw(output='text'))
        except Exception:
            self.io.writeln(f"Circuit: {self.last_circuit.num_qubits} qubits, "
                           f"depth {self.last_circuit.depth()}, "
                           f"{self.last_circuit.size()} gates")

    _RE_PRINT_QUBIT = re.compile(r'QUBIT\s*\((.+)\)\s*$', re.IGNORECASE)

    _RE_PRINT_ENT = re.compile(r'ENTANGLEMENT\s*\((.+)\)\s*$', re.IGNORECASE)

    def _current_state(self) -> tuple[Any, int, list[int] | None]:
        """(statevector, qubit count, LOCC register sizes) at the current point
        of execution: the live per-shot state, the LOCC joint state, the state
        of the circuit built so far during RUN, or the last run's state."""
        live = getattr(self, '_live', None)
        if live is not None:
            return live.sv, live.n, None
        if self.locc_mode and self.locc is not None and self.locc.joint:
            return self.locc.sv, self.locc.n_total, list(self.locc.sizes)
        qc = getattr(self, '_build_qc', None)
        if qc is not None:
            return (self._extract_state_now(qc.copy(), self.sim_precision, self._seed),
                    qc.num_qubits, None)
        return self.last_sv, self.num_qubits, None

    def _try_quantum_print(self, text: str, run_vars: MutableMapping[str, Any]) -> str | None:
        """Quantum PRINT forms: STATE (Dirac notation), QUBIT(q) (P(1) and the
        Bloch vector), ENTANGLEMENT(a, b) (entanglement of formation of the
        two-qubit reduced state), and @REG (one LOCC register). Each reads the
        state at the point the PRINT executes. Returns None for other text."""
        item = text.strip()
        upper = item.upper()
        if upper.startswith('@') and self.locc_mode and self.locc is not None:
            reg = upper[1:].strip()
            if reg in self.locc.names:
                if self.locc.joint:
                    return self._format_dirac(self.locc.sv, self.locc.n_total,
                                              list(self.locc.sizes))
                return self._format_dirac(self.locc.svs[reg], self.locc.get_size(reg))
            return None
        m_q = self._RE_PRINT_QUBIT.match(item)
        m_e = self._RE_PRINT_ENT.match(item)
        if upper != 'STATE' and not m_q and not m_e:
            return None
        if self.locc_mode and self.locc is not None and not self.locc.joint:
            return "<SPLIT registers have no joint state: PRINT @A, @B, ...>"
        sv, n, segments = self._current_state()
        if sv is None:
            return "<no state>"
        if upper == 'STATE':
            return self._format_dirac(sv, n, segments)
        from qubasic_core import simtools as st
        if m_q:
            q = int(self._eval_with_vars(m_q.group(1), run_vars))
            if q < 0 or q >= n:
                raise ValueError(f"QUBIT {q} OUT OF RANGE (0-{n - 1})")
            x, y, z = (round(c, 3) + 0.0 for c in st.bloch(sv, q, n))
            p1 = min(1.0, max(0.0, (1 - z) / 2)) + 0.0
            return f"q{q}: P(1)={p1:.4f} Bloch=({x:.3f},{y:.3f},{z:.3f})"
        assert m_e is not None                 # STATE and QUBIT( returned above
        args = self._split_arg_list(m_e.group(1))
        if len(args) != 2:
            raise ValueError("ENTANGLEMENT takes two qubits: ENTANGLEMENT(a, b)")
        qa, qb = (int(self._eval_with_vars(a, run_vars)) for a in args)
        if qa == qb or not (0 <= qa < n and 0 <= qb < n):
            raise ValueError(f"ENTANGLEMENT needs two different qubits in 0-{n - 1}")
        c = st.concurrence(st.subsystem_rdm(sv, [qa, qb], n))
        return f"E(q{qa},q{qb}) = {st.eof_from_concurrence(c):.6f} ebits (concurrence {c:.6f})"

    def _format_dirac(self, sv: np.ndarray, n_qubits: int, segments: list[int] | None = None) -> str:
        """Dirac notation of the first 16 non-zero amplitudes, in the active
        OPTION ENDIAN order; `segments` splits a LOCC joint state into its
        registers (A|B|...)."""
        from qubasic_core.engine import AMPLITUDE_THRESHOLD
        from qubasic_core.simtools import nonzero_indices
        flat = np.asarray(sv).reshape(-1)
        idx, total = nonzero_indices(flat, AMPLITUDE_THRESHOLD, 16)
        parts = []
        for i in idx:
            amp = complex(flat[int(i)])
            bits = format(int(i), f'0{n_qubits}b')
            if segments:
                segs, pos = [], len(bits)
                for size in segments:
                    segs.append(bits[pos - size:pos])
                    pos -= size
                bits = '|'.join(segs)
            state = self._bits(bits)
            if abs(amp.imag) < AMPLITUDE_THRESHOLD:
                if abs(amp.real - 1.0) < 1e-6:
                    parts.append(f"|{state}\u27E9")
                elif abs(amp.real + 1.0) < 1e-6:
                    parts.append(f"-|{state}\u27E9")
                else:
                    parts.append(f"{amp.real:+.4f}|{state}\u27E9")
            else:
                parts.append(f"({amp.real:.3f}{amp.imag:+.3f}j)|{state}\u27E9")
        if total > len(parts):
            parts.append("...")
        return " ".join(parts) if parts else "|0\u27E9"

    def cmd_decompose(self) -> None:
        """DECOMPOSE — gate counts of the last run's circuit."""
        if self.last_circuit is None:
            self.io.writeln("?NO CIRCUIT — RUN first")
            return
        ops: dict[str, int] = {}
        for inst in self.last_circuit.data:
            name = inst.operation.name
            ops[name] = ops.get(name, 0) + 1
        self.io.writeln(f"\n  Circuit: {self.last_circuit.num_qubits} qubits, "
                       f"depth {self.last_circuit.depth()}, {self.last_circuit.size()} gates")
        for name, count in sorted(ops.items(), key=lambda x: -x[1]):
            bar = '█' * min(count, 40)
            self.io.writeln(f"    {name:>10}  {count:>4}  {bar}")
        self.io.writeln('')

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

    def _locc_bloch(self, rest: str) -> None:
        m = re.match(r'([A-Z])\s*(\d*)', rest.strip(), re.IGNORECASE) if rest.strip() else None
        if m and m.group(1):
            reg = m.group(1).upper()
            if reg not in self._locc_engine.names:
                self.io.writeln(f"?UNKNOWN REGISTER: {reg}")
                return
            sv = self._locc_engine.get_sv(reg)
            n = self._locc_engine.get_n(reg)
            idx = self._locc_engine._idx(reg)
            if m.group(2):
                q = int(m.group(2))
                actual_q = q if not self._locc_engine.joint else q + self._locc_engine.offsets[idx]
                self.io.writeln(f"  [Register {reg}, qubit {q}]")
                self._print_bloch_single(sv, actual_q, n)
            else:
                n_show = self._locc_engine.get_size(reg)
                for q in range(min(n_show, 4)):
                    actual_q = q if not self._locc_engine.joint else q + self._locc_engine.offsets[idx]
                    self.io.writeln(f"  [Register {reg}, qubit {q}]")
                    self._print_bloch_single(sv, actual_q, n)
                    self.io.writeln('')
        else:
            self.io.writeln(f"?USAGE: BLOCH <reg> [qubit]  (registers: {', '.join(self._locc_engine.names)})")


# ---------------------------------------------------------------------------
# QUBASIC visualization: braille Bloch spheres (DRAW), HEATMAP, ANIMATE and PLOT.
# ---------------------------------------------------------------------------

_BRAILLE_BASE = 0x2800


# Braille dot positions: col0=(0,1,2,6), col1=(3,4,5,7)
_BRAILLE_DOTS = [
    (0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04), (0, 3, 0x40),
    (1, 0, 0x08), (1, 1, 0x10), (1, 2, 0x20), (1, 3, 0x80),
]


def braille_bloch(x: float, y: float, z: float, radius: int = 10) -> list[str]:
    """Render a Bloch sphere in braille characters (XZ plane projection).

    Returns a list of strings (lines) forming the braille image.
    """
    W = radius * 2 + 1  # pixels
    # Braille cell = 2 wide x 4 tall pixels
    cols = (W + 1) // 2
    rows = (W + 3) // 4

    grid = [[0] * cols for _ in range(rows)]

    def _set_pixel(px: int, py: int) -> None:
        if 0 <= px < W and 0 <= py < W:
            bc = px // 2
            br = py // 4
            dx = px % 2
            dy = py % 4
            for dc, dr, bit in _BRAILLE_DOTS:
                if dc == dx and dr == dy:
                    grid[br][bc] |= bit

    cx, cy = radius, radius

    # Draw circle
    for angle in range(0, 360, 3):
        rad = math.radians(angle)
        px = round(cx + radius * math.cos(rad))
        py = round(cy + radius * math.sin(rad))
        _set_pixel(px, py)

    # Axes (horizontal and vertical through center)
    for i in range(W):
        _set_pixel(i, cy)
        _set_pixel(cx, i)

    # State point
    px = round(cx + x * (radius - 1))
    pz = round(cy - z * (radius - 1))
    # Draw a 2x2 block for visibility
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            _set_pixel(px + dx, pz + dy)

    lines = []
    for row in grid:
        line = ''.join(chr(_BRAILLE_BASE + cell) for cell in row)
        lines.append(line)
    return lines


class VisualizeMixin(PresentationHost):
    """Text-mode pictures of states and sweeps."""

    def cmd_draw(self, rest: str) -> None:
        """DRAW [qubit] — braille Bloch sphere; on a graphics SCREEN, DRAW "commands"
        runs QBASIC's DRAW macro language instead."""
        if self._canvas is not None and rest.strip() and not rest.strip().isdigit():
            self._gfx_draw('DRAW ' + rest, self.variables)
            return
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE -- RUN first")
            return
        n = self._active_nqubits
        if rest.strip():
            q = int(rest.strip())
            x, y, z = self._bloch_vector(sv, q, n)
            self._draw_braille_bloch(x, y, z, q)
        else:
            for q in range(min(n, 4)):
                x, y, z = self._bloch_vector(sv, q, n)
                self._draw_braille_bloch(x, y, z, q)

    def _draw_braille_bloch(self, x: float, y: float, z: float, qubit: int) -> None:
        label = f"q{qubit} ({x:.2f},{y:.2f},{z:.2f})"
        self.io.writeln(f"  {label}")
        for line in braille_bloch(x, y, z):
            self.io.writeln(f"    {line}")

    def cmd_heatmap(self, rest: str = '') -> None:
        """HEATMAP [CONCURRENCE] — pairwise correlation grid of the state.

        Default cells show the quantum mutual information I(i:j) = S(i) + S(j)
        - S(ij) in bits (0 to 2: classical and quantum correlation). HEATMAP
        CONCURRENCE shows the Wootters concurrence instead (0 to 1: pairwise
        entanglement only; a GHZ pair reads 0, a Bell pair 1)."""
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE -- RUN first")
            return
        from qubasic_core import simtools as st
        mode = 'concurrence' if rest.strip().upper().startswith('CONC') else 'mi'
        n = self._active_nqubits
        n_show = min(n, 12)
        scale = 1.0 if mode == 'concurrence' else 2.0
        title = ("pairwise concurrence, 0-1" if mode == 'concurrence'
                 else "quantum mutual information I(i:j), 0-2 bits")
        self.io.writeln(f"\n  Heatmap ({title}, {n_show} qubits):")
        _shades = ' \u2591\u2592\u2593\u2588'
        self.io.writeln('     ' + ''.join(f'{q:>6}' for q in range(n_show)))
        cells: dict = {}
        for qi in range(n_show):
            for qj in range(qi + 1, n_show):
                if mode == 'concurrence':
                    v = st.concurrence(st.subsystem_rdm(sv, [qi, qj], n))
                else:
                    v = st.mutual_information(sv, qi, qj, n)
                cells[(qi, qj)] = cells[(qj, qi)] = v
        for qi in range(n_show):
            row = f'  {qi:>2} '
            for qj in range(n_show):
                if qi == qj:
                    row += '     -'
                    continue
                v = cells[(qi, qj)]
                if v < 0.005:
                    row += '     .'
                    continue
                shade = _shades[min(len(_shades) - 1, 1 + int(3 * v / scale))]
                row += f' {shade}{v:4.2f}'
            self.io.writeln(row)
        self.io.writeln('')

    def cmd_animate(self, rest: str) -> None:
        """ANIMATE var start end [steps] [delay] — animated parameter sweep."""
        parts = rest.split()
        if len(parts) < 3:
            self.io.writeln("?USAGE: ANIMATE <var> <start> <end> [steps] [delay_ms]")
            return
        var = parts[0]
        start = self.eval_expr(parts[1])
        end = self.eval_expr(parts[2])
        steps = int(parts[3]) if len(parts) > 3 else 10
        delay = float(parts[4]) / 1000 if len(parts) > 4 else 0.3
        values = [start + (end - start) * i / max(1, steps - 1) for i in range(steps)]

        had_prior, prior = var in self.variables, self.variables.get(var)
        try:
            for i, val in enumerate(values):
                self.variables[var] = val
                with self._silenced():
                    self.cmd_run()
                if self.last_counts:
                    ranked = sorted(self.last_counts.items(), key=lambda x: -x[1])
                    total = sum(self.last_counts.values())
                    top = ranked[0]
                    bar = '\u2588' * int(30 * top[1] / total)
                    frame = f"\r  {var}={val:>8.3f}  |{self._bits(top[0])}\u27E9 {top[1]/total:>5.1%} {bar:<30}"
                    self.io.write(frame)
                time.sleep(delay)
        finally:
            self._restore_var(var, had_prior, prior)
        self.io.writeln('')

    def _restore_var(self, var: str, had_prior: bool, prior: Any) -> None:
        """Put a swept variable back the way the program had it."""
        if had_prior:
            self.variables[var] = prior
        else:
            self.variables.pop(var, None)

    def cmd_plot(self, rest: str) -> None:
        """PLOT var start end [steps] — braille line plot of P(|0...0>) vs variable."""
        parts = rest.split()
        if len(parts) < 3:
            self.io.writeln("?USAGE: PLOT <var> <start> <end> [steps]")
            return
        var = parts[0]
        start = self.eval_expr(parts[1])
        end = self.eval_expr(parts[2])
        steps = int(parts[3]) if len(parts) > 3 else 20
        values = [start + (end - start) * i / max(1, steps - 1) for i in range(steps)]

        xs, ys = [], []
        had_prior, prior = var in self.variables, self.variables.get(var)
        try:
            for val in values:
                self.variables[var] = val
                with self._silenced():
                    self.cmd_run()
                if self.last_counts:
                    total = sum(self.last_counts.values())
                    counts = self.last_counts
                    top_state = max(counts, key=lambda k: counts[k])
                    xs.append(val)
                    ys.append(self.last_counts[top_state] / total)
        finally:
            self._restore_var(var, had_prior, prior)

        if not xs:
            self.io.writeln("?No data to plot")
            return
        # ASCII plot
        W, H = 60, 12
        y_min, y_max = min(ys), max(ys)
        if y_max - y_min < 0.001:
            y_max = y_min + 0.1
        self.io.writeln(f"\n  P(top) vs {var}: {start} to {end}")
        canvas = [[' '] * W for _ in range(H)]
        for i, (xv, yv) in enumerate(zip(xs, ys)):
            col = int((i / max(1, len(xs) - 1)) * (W - 1))
            row = H - 1 - int(((yv - y_min) / (y_max - y_min)) * (H - 1))
            row = max(0, min(H - 1, row))
            col = max(0, min(W - 1, col))
            canvas[row][col] = '\u2022'
        for r, line in enumerate(canvas):
            label = f"{y_max - (y_max - y_min) * r / (H - 1):.2f}" if r in (0, H - 1) else '    '
            self.io.writeln(f"  {label:>5} |{''.join(line)}|")
        self.io.writeln(f"        {start:<10.3f}{' ' * (W - 20)}{end:>10.3f}")
        self.io.writeln('')


# ---------------------------------------------------------------------------
# QUBASIC sweep mixin — parameter sweep command.
# ---------------------------------------------------------------------------

class SweepMixin(PresentationHost):
    """Parameter sweep command for QBasicTerminal."""

    def cmd_sweep(self, rest: str) -> None:
        """SWEEP var start end [steps] — run circuit for each parameter value.

        When plotille is available, appends a braille line chart of the
        top-state probability vs. the swept variable.
        """
        parts = rest.split()
        if len(parts) < 3:
            self.io.writeln("?USAGE: SWEEP <var> <start> <end> [steps]")
            return
        var = parts[0]
        start = self.eval_expr(parts[1])
        end = self.eval_expr(parts[2])
        steps = int(parts[3]) if len(parts) > 3 else 10
        if steps < 1:
            self.io.writeln("?SWEEP needs at least 1 step")
            return

        self.io.writeln(f"\nSWEEP {var} from {start} to {end} in {steps} steps:")
        if steps == 1:
            values = [start]
        else:
            values = [start + (end - start) * i / (steps - 1) for i in range(steps)]
        backend_opts = {'method': self.sim_method}
        if self.sim_device == 'GPU':
            backend_opts['device'] = 'GPU'
        if self._noise_model:
            backend_opts['noise_model'] = self._noise_model
        backend = AerSimulator(**backend_opts)
        sweep_xs: list[float] = []
        sweep_ys: list[float] = []
        # Parametric fast path: compile once with the swept variable as a
        # bound Parameter, then re-bind per point (no rebuild, no retranspile).
        pctx = None
        try:
            pctx = self._alg_parametric_setup([var], sampling=True)
        except Exception:
            pctx = None
        if pctx is not None:
            self.io.writeln("  (parametric compile: transpiled once, bound per point)")
        # Transient progress bar in an interactive terminal only; piped and
        # captured output is unchanged.
        _prog = _ptask = None
        try:
            import sys as _sys
            if _sys.stdout.isatty() and len(values) > 2:
                from rich.progress import Progress
                _prog = Progress(transient=True)
                _prog.start()
                _ptask = _prog.add_task(f"SWEEP {var}", total=len(values))
        except Exception:
            _prog = None
        # The sweep temporarily overwrites the variable; restore it afterward
        # so a later RUN/STATS still sees the program's own value.
        had_prior = var in self.variables
        prior = self.variables.get(var)
        try:
            for val in values:
                self.variables[var] = val
                try:
                    if pctx is not None:
                        bound = pctx['qc'].assign_parameters(
                            {next(iter(pctx['pmap'].values())): float(val)})
                        _kw = {'shots': self.shots}
                        if self._seed is not None:
                            _kw['seed_simulator'] = self._seed
                        result = pctx['backend'].run(bound, **_kw).result()
                    else:
                        qc, has_measure = self.build_circuit()
                        if has_measure:
                            qc.measure_all()
                        result = backend.run(transpile(qc, backend, optimization_level=self._transpile_opt_level), shots=self.shots).result()
                    counts = dict(result.get_counts())
                    ranked = sorted(counts.items(), key=lambda x: -x[1])
                    top = ranked[0]
                    bar_len = int(30 * top[1] / self.shots)
                    top2 = f" |{self._bits(ranked[1][0])}\u27E9={ranked[1][1]}" if len(ranked) > 1 else ""
                    n_unique = len(ranked)
                    bar = '\u2588' * bar_len
                    self.io.writeln(f"  {var}={val:8.4f}  |{self._bits(top[0])}\u27E9 {top[1]:>5}/{self.shots} "
                                   f"{bar}{top2}  ({n_unique} states)")
                    sweep_xs.append(val)
                    sweep_ys.append(top[1] / self.shots)
                except Exception as e:
                    self.io.writeln(f"  {var}={val:8.4f}  ERROR: {e}")
                if _prog is not None and _ptask is not None:
                    _prog.advance(_ptask)
        finally:
            if _prog is not None:
                _prog.stop()
            if had_prior:
                self.variables[var] = prior
            else:
                self.variables.pop(var, None)

        # Plotille chart of P(top state) vs variable
        if _plotille is not None and len(sweep_xs) >= 2:
            self.io.writeln('')
            self.io.writeln(_plotille.plot(
                sweep_xs, sweep_ys,
                width=60, height=15,
                X_label=var,
                Y_label='P(top)',
                y_min=0.0, y_max=1.0,
                lc='cyan'))
        self.io.writeln('')
