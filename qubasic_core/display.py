"""QUBASIC display — histograms, statevector, Bloch sphere rendering."""

import sys
import math
import numpy as np

from qubasic_core.engine import (
    MAX_HISTOGRAM_STATES, MAX_DISPLAY_AMPLITUDES,
    HISTOGRAM_BAR_WIDTH, AMPLITUDE_THRESHOLD,
    _RICH, _RichTable,
)


def _get_console():
    """Get a Rich console bound to current sys.stdout (test-safe)."""
    if _RICH:
        from rich.console import Console
        return Console(file=sys.stdout, highlight=False)
    return None


class DisplayMixin:
    """Display and visualization methods for the QUBASIC terminal.

    Requires: TerminalProtocol — uses self.num_qubits, self.arrays.
    """

    def _rich_print(self, *renderables) -> None:
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

    def _print_statevector(self, sv, n_qubits=None):
        """Print the first non-zero amplitudes of the statevector (index order).

        Only the shown amplitudes are formatted; the rest are counted in one
        vectorized pass, so a large state prints in seconds.
        """
        from qubasic_core.statetools import nonzero_indices
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

    def _print_sv_compact(self, sv):
        """Compact statevector display for step mode (first 8 non-zero terms)."""
        from qubasic_core.statetools import nonzero_indices
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

    def _print_probs(self, sv, n_qubits=None):
        """Print the most probable outcomes with a histogram bar each."""
        from qubasic_core.statetools import top_k
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

    def _print_bloch_single(self, sv, qubit, n_qubits=None):
        """ASCII Bloch sphere for a single qubit."""
        x, y, z = self._bloch_vector(sv, qubit, n_qubits)

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

    def _bloch_vector(self, sv, qubit, n_qubits=None):
        """Bloch vector (x, y, z) of one qubit: <X> = 2 Re(rho_10),
        <Y> = 2 Im(rho_10), <Z> = rho_00 - rho_11, from the reduced state."""
        from qubasic_core.statetools import bloch
        n = n_qubits if n_qubits is not None else self.num_qubits
        if qubit < 0 or qubit >= n:
            raise ValueError(f"QUBIT {qubit} OUT OF RANGE (0-{n - 1})")
        return bloch(sv, qubit, n)
