"""QUBASIC profiler — profile mode, gate count, depth tracking, statistics accumulator."""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from qubasic_core.host import HostState

MAX_STATS_RUNS = 10000


class ProfilerMixin(HostState):
    """Profiling and statistics for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.last_counts,
    self.shots, self.cmd_run().
    """

    def _init_profiler(self) -> None:
        self._profile_mode: bool = False
        self._profile_data: dict[int, dict[str, float]] = {}
        self._profile_start: float = 0.0
        self._stats_runs: list[dict[str, int]] = []

    # ── Profile mode ───────────────────────────────────────────────────

    def cmd_profile(self, rest: str = '') -> None:
        """PROFILE [ON|OFF|SHOW] — toggle profiling or show results."""
        arg = rest.strip().upper()
        if arg == 'ON':
            self._profile_mode = True
            self._profile_data.clear()
            self.io.writeln("PROFILE ON")
        elif arg == 'OFF':
            self._profile_mode = False
            self.io.writeln("PROFILE OFF")
        elif arg == 'SHOW' or not arg:
            self._show_profile()
        else:
            self.io.writeln("?USAGE: PROFILE [ON|OFF|SHOW]")

    def _profile_line_start(self, line_num: int) -> None:
        if self._profile_mode:
            self._profile_start = time.perf_counter()

    def _profile_line_end(self, line_num: int, gates: int = 0) -> None:
        if self._profile_mode:
            dt = (time.perf_counter() - self._profile_start) * 1000  # ms
            if line_num not in self._profile_data:
                self._profile_data[line_num] = {'time_ms': 0.0, 'calls': 0, 'gates': 0}
            entry = self._profile_data[line_num]
            entry['time_ms'] += dt
            entry['calls'] += 1
            entry['gates'] += gates

    def _show_profile(self) -> None:
        if not self._profile_data:
            self.io.writeln("  No profile data (PROFILE ON, then RUN)")
            return
        self.io.writeln("\n  Profile Results:")
        self.io.writeln(f"  {'Line':>6}  {'Time(ms)':>10}  {'%':>5}  {'Calls':>6}  {'Gates':>6}  Source")
        total_time = sum(d['time_ms'] for d in self._profile_data.values())
        for ln in sorted(self._profile_data.keys()):
            d = self._profile_data[ln]
            src = self.program.get(ln, '')[:40]
            pct = 100 * d['time_ms'] / total_time if total_time > 0 else 0
            self.io.writeln(f"  {ln:>6}  {d['time_ms']:>10.2f}  {pct:>5.1f}  {d['calls']:>6}  "
                            f"{d['gates']:>6}  {src}")
        self.io.writeln(f"\n  Total: {total_time:.2f} ms")
        self.io.writeln('')

    # ── Statistics accumulator ─────────────────────────────────────────

    def cmd_stats(self, rest: str = '') -> None:
        """STATS [N|SHOW|CLEAR|CSV <file>] — multi-run statistics accumulator."""
        arg = rest.strip().upper()
        if arg == 'CLEAR':
            self._stats_runs.clear()
            self.io.writeln("STATS CLEARED")
            return
        if arg == 'SHOW' or not arg:
            self._show_stats()
            return
        if arg.startswith('CSV'):
            self._stats_export_csv(rest[3:].strip())
            return
        # STATS N — run N trials
        try:
            n = int(arg)
        except ValueError:
            self.io.writeln("?USAGE: STATS [N|SHOW|CLEAR|CSV <file>]")
            return
        if n < 1:
            self.io.writeln("?STATS needs at least 1 run")
            return
        self.io.writeln(f"\nRunning {n} trials...")
        old_io = self.io
        old_stdout = sys.stdout
        devnull = open(os.devnull, 'w')  # opened once for all trials
        base_seed = self._seed
        try:
            for trial in range(n):
                # A fixed SEED would make every trial identical, so vary it per
                # trial (and invalidate the cache) while keeping reproducibility.
                if base_seed is not None:
                    self._seed = base_seed + trial
                    np.random.seed(base_seed + trial)
                    self._circuit_cache_key = None
                sys.stdout = devnull
                try:
                    with self._silenced():
                        self.cmd_run()
                finally:
                    sys.stdout = old_stdout
                if self.last_counts:
                    if len(self._stats_runs) >= MAX_STATS_RUNS:
                        self.io.writeln(f"?STATS: run limit ({MAX_STATS_RUNS}) reached, stopping collection")
                        break
                    self._stats_runs.append(dict(self.last_counts))
                # Spinner only in an interactive terminal: carriage-return
                # animation turns into junk in piped or captured output.
                if n > 10 and (trial + 1) % (n // 10) == 0 and sys.stdout.isatty():
                    from qubasic_core.qol import quantum_spin
                    spin = quantum_spin(trial)
                    self.io.write(f"  {spin} {100 * (trial + 1) // n}%..." + '\r')
        finally:
            sys.stdout = old_stdout
            self.io = old_io
            devnull.close()
            if base_seed is not None:
                self._seed = base_seed
        if n > 10 and sys.stdout.isatty():
            self.io.write(" " * 30 + '\r')
        self.io.writeln(f"Collected {len(self._stats_runs)} runs ({n} trials)")

    def _show_stats(self) -> None:
        if not self._stats_runs:
            self.io.writeln("  No statistics collected (STATS N to run N trials)")
            return
        n = len(self._stats_runs)
        # Aggregate: count how often each state appears across runs
        state_totals: dict[str, list[int]] = {}
        for run in self._stats_runs:
            for state, count in run.items():
                if state not in state_totals:
                    state_totals[state] = []
                state_totals[state].append(count)
            # States not seen in this run get 0
            for state in state_totals:
                if state not in run:
                    state_totals[state].append(0)
        # Pad lists to same length.  This is O(states * n) but n is bounded
        # by MAX_STATS_RUNS (enforced during collection), so it stays tractable.
        for state in state_totals:
            while len(state_totals[state]) < n:
                state_totals[state].append(0)
        # Compute per-run shot totals for probability conversion
        shot_totals = [sum(run.values()) for run in self._stats_runs]
        avg_shots = sum(shot_totals) / len(shot_totals) if shot_totals else 1
        self.io.writeln(f"\n  Statistics over {n} runs ({avg_shots:.0f} shots/run):")
        self.io.writeln(f"  {'State':>10}  {'Mean':>8}  {'StdDev':>8}  {'Min':>6}  {'Max':>6}  {'P(mean)':>8}")
        for state in sorted(state_totals.keys()):
            vals = state_totals[state]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(variance)
            prob = mean / avg_shots if avg_shots > 0 else 0
            self.io.writeln(f"  |{self._bits(state)}\u27E9  {mean:>8.1f}  {std:>8.2f}  {min(vals):>6}  {max(vals):>6}  {prob:>7.4f}")
        self.io.writeln('')

    def _stats_export_csv(self, path: str) -> None:
        """Export accumulated stats to CSV."""
        if not self._stats_runs:
            self.io.writeln("  No statistics collected")
            return
        if not path:
            self.io.writeln("?USAGE: STATS CSV <filename>")
            return
        try:
            path = self._sanitize_path(path)
        except (ValueError, AttributeError) as e:
            self.io.writeln(f"?STATS CSV ERROR: {e}")
            return
        n = len(self._stats_runs)
        state_totals: dict[str, list[int]] = {}
        for run in self._stats_runs:
            for state, count in run.items():
                if state not in state_totals:
                    state_totals[state] = []
                state_totals[state].append(count)
            for state in state_totals:
                if state not in run:
                    state_totals[state].append(0)
        for state in state_totals:
            while len(state_totals[state]) < n:
                state_totals[state].append(0)
        shot_totals = [sum(run.values()) for run in self._stats_runs]
        avg_shots = sum(shot_totals) / len(shot_totals) if shot_totals else 1
        lines = ['state,mean,stddev,min,max,probability']
        for state in sorted(state_totals.keys()):
            vals = state_totals[state]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(variance)
            prob = mean / avg_shots if avg_shots > 0 else 0
            lines.append(f"{state},{mean:.2f},{std:.4f},{min(vals)},{max(vals)},{prob:.6f}")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        self.io.writeln(f"STATS EXPORTED to {path} ({n} runs, {len(state_totals)} states)")
