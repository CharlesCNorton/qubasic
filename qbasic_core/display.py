"""QBASIC display — histograms, statevector, Bloch sphere rendering."""

import math
import numpy as np

from qbasic_core.engine import (
    MAX_HISTOGRAM_STATES, MAX_DISPLAY_AMPLITUDES,
    HISTOGRAM_BAR_WIDTH, AMPLITUDE_THRESHOLD,
    _RICH, _console, _RichTable,
)


class DisplayMixin:
    """Display and visualization methods for the QBASIC terminal.

    Requires: TerminalProtocol — uses self.num_qubits, self.arrays.
    """

    def print_histogram(self, counts: dict[str, int]) -> None:
        """Measurement histogram with optional rich-table formatting."""
        total = sum(counts.values())
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        display = sorted_counts[:MAX_HISTOGRAM_STATES]

        if _RICH and _console:
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
            _console.print(
                f"\n  [dim]Showing top {MAX_HISTOGRAM_STATES} of "
                f"{len(sorted_counts)} outcomes:[/dim]\n")
        else:
            _console.print()

        for state, count in display:
            pct = 100 * count / total
            bar_len = int(HISTOGRAM_BAR_WIDTH * count / max_count)
            bar = '\u2588' * bar_len
            color = "green" if pct > 40 else "yellow" if pct > 10 else "dim"
            table.add_row(
                f"|{state}\u27E9",
                str(count),
                f"{pct:5.1f}%",
                f"[{color}]{bar}[/{color}]")

        if len(sorted_counts) > MAX_HISTOGRAM_STATES:
            rest_count = sum(c for _, c in sorted_counts[MAX_HISTOGRAM_STATES:])
            table.add_row(
                "...", str(rest_count),
                f"{100*rest_count/total:5.1f}%", "[dim](remaining)[/dim]")

        _console.print(table)
        _console.print()

    def _print_histogram_plain(self, display: list, sorted_counts: list,
                               total: int) -> None:
        """Plain-text histogram (fallback when rich is not available)."""
        if len(sorted_counts) > MAX_HISTOGRAM_STATES:
            self.io.writeln(f"\n  Showing top {MAX_HISTOGRAM_STATES} of {len(sorted_counts)} outcomes:\n")
        else:
            self.io.writeln('')

        max_count = max(c for _, c in display) if display else 1
        max_label = max(len(k) for k, _ in display) if display else 1

        for state, count in display:
            pct = 100 * count / total
            bar_len = int(HISTOGRAM_BAR_WIDTH * count / max_count)
            bar = '\u2588' * bar_len
            ket = f"|{state}\u27E9"
            self.io.writeln(f"  {ket:>{max_label+3}}  {count:>6}  ({pct:5.1f}%)  {bar}")

        if len(sorted_counts) > MAX_HISTOGRAM_STATES:
            rest_count = sum(c for _, c in sorted_counts[MAX_HISTOGRAM_STATES:])
            print(f"  {'...':>{max_label+3}}  {rest_count:>6}  "
                  f"({100*rest_count/total:5.1f}%)  (remaining)")
        self.io.writeln('')

    def _print_statevector(self, sv, n_qubits=None):
        """Print non-zero amplitudes of the statevector."""
        sv = np.ascontiguousarray(sv).ravel()
        n = n_qubits if n_qubits is not None else self.num_qubits

        if _RICH and _console:
            table = _RichTable(show_header=True, header_style="bold cyan",
                               box=None, padding=(0, 1))
            table.add_column("State", justify="right", style="bold")
            table.add_column("Amplitude", justify="right")
            table.add_column("P", justify="right")
            _console.print(f"\n  [bold]Statevector ({n} qubits):[/bold]")
            count = 0
            for i, amp in enumerate(sv):
                if abs(amp) > AMPLITUDE_THRESHOLD:
                    state = format(i, f'0{n}b')
                    prob = abs(amp)**2
                    table.add_row(
                        f"|{state}\u27E9",
                        f"{amp.real:+.4f}{amp.imag:+.4f}j",
                        f"{prob:.4f}")
                    count += 1
                    if count >= MAX_DISPLAY_AMPLITUDES:
                        remaining = sum(1 for a in sv[i+1:]
                                        if abs(a) > AMPLITUDE_THRESHOLD)
                        if remaining:
                            table.add_row("...", "", f"+{remaining} more")
                        break
            _console.print(table)
            _console.print()
            return

        self.io.writeln(f"\n  Statevector ({n} qubits):")
        count = 0
        for i, amp in enumerate(sv):
            if abs(amp) > AMPLITUDE_THRESHOLD:
                state = format(i, f'0{n}b')
                prob = abs(amp)**2
                print(f"  |{state}\u27E9  {amp.real:+.4f}{amp.imag:+.4f}j  "
                      f"(P={prob:.4f})")
                count += 1
                if count >= MAX_DISPLAY_AMPLITUDES:
                    remaining = sum(1 for a in sv[i+1:] if abs(a) > AMPLITUDE_THRESHOLD)
                    if remaining:
                        self.io.writeln(f"  ... and {remaining} more non-zero amplitudes")
                    break
        self.io.writeln('')

    def _print_sv_compact(self, sv):
        """Compact statevector display for step mode."""
        n = self.num_qubits
        parts = []
        for i, amp in enumerate(sv):
            if abs(amp) > AMPLITUDE_THRESHOLD:
                state = format(i, f'0{n}b')
                if abs(amp.imag) < AMPLITUDE_THRESHOLD:
                    parts.append(f"{amp.real:+.3f}|{state}\u27E9")
                else:
                    parts.append(f"({amp.real:.2f}{amp.imag:+.2f}j)|{state}\u27E9")
                if len(parts) >= 8:
                    parts.append("...")
                    break
        self.io.writeln(f"   |\u03C8\u27E9 = {' '.join(parts)}")

    def _print_probs(self, sv):
        """Print probability distribution with histogram."""
        n = self.num_qubits
        probs = []
        for i, amp in enumerate(sv):
            p = abs(amp)**2
            if p > AMPLITUDE_THRESHOLD:
                state = format(i, f'0{n}b')
                probs.append((state, p))

        probs.sort(key=lambda x: -x[1])
        display = probs[:MAX_HISTOGRAM_STATES]

        self.io.writeln(f"\n  Probability distribution ({len(probs)} non-zero):\n")
        max_p = max(p for _, p in display) if display else 1

        for state, p in display:
            bar_len = int(HISTOGRAM_BAR_WIDTH * p / max_p)
            bar = '\u2588' * bar_len
            self.io.writeln(f"  |{state}\u27E9  {p*100:6.2f}%  {bar}")

        if len(probs) > MAX_HISTOGRAM_STATES:
            self.io.writeln(f"  ... and {len(probs)-MAX_HISTOGRAM_STATES} more states")
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
        self.io.writeln(f"{'|0\u27E9':^{W+4}}")
        for row in grid:
            self.io.writeln(f"  {''.join(row)}")
        self.io.writeln(f"{'|1\u27E9':^{W+4}}")

    def _bloch_vector(self, sv, qubit, n_qubits=None):
        """Compute the Bloch vector for a single qubit from the statevector."""
        n = n_qubits if n_qubits is not None else self.num_qubits
        sv_arr = np.array(sv).reshape([2] * n)

        target_axis = n - 1 - qubit
        tensor = np.moveaxis(sv_arr, target_axis, 0)
        t0 = tensor[0].flatten()
        t1 = tensor[1].flatten()

        rho_00 = np.sum(np.abs(t0)**2)
        rho_11 = np.sum(np.abs(t1)**2)
        rho_01 = np.sum(np.conj(t0) * t1)

        x = float(2 * rho_01.real)
        y = float(-2 * rho_01.imag)
        z = float(rho_00 - rho_11)

        return x, y, z
