"""QUBASIC quality-of-life features — visual, fun, and convenience commands."""

from __future__ import annotations

import difflib
import math
import os
import random
import sys
import time
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Quantum spinner (#21)
# ═══════════════════════════════════════════════════════════════════════

_QUANTUM_SPINNER = [
    '|0\u27E9', '|+\u27E9', '|1\u27E9', '|-\u27E9',
    '|i\u27E9', '|\u03C8\u27E9', '|0\u27E9', '|\u03D5\u27E9',
]


def quantum_spin(step: int) -> str:
    """Return the spinner frame for the given step."""
    return _QUANTUM_SPINNER[step % len(_QUANTUM_SPINNER)]


# ═══════════════════════════════════════════════════════════════════════
# Tips of the day (#11)
# ═══════════════════════════════════════════════════════════════════════

_TIPS = [
    "Use CTRL H 0, 1 to make any gate controlled.",
    "STEP mode lets you watch the statevector evolve line by line.",
    "REWIND goes back in time during STEP mode.",
    "LOCC JOINT 3 3 enables quantum teleportation with classical correction.",
    "SWEEP var 0 PI 10 shows how probability changes with a parameter.",
    "NOISE depolarizing 0.05 adds realistic noise to your circuit.",
    "SEED 42 makes your results reproducible.",
    "DEMO LIST shows all 12 built-in quantum algorithms.",
    "BLOCH draws an ASCII Bloch sphere for each qubit.",
    "INV RX 0.5, 0 applies the inverse (dagger) of any gate.",
    "UNITARY MYGATE = [[0,1],[1,0]] defines a custom gate from a matrix.",
    "DEF BELL = H 0 : CX 0,1 creates a reusable gate sequence.",
    "POKE $D000, 8 sets the qubit count via the memory map.",
    "SYS $E000 runs a built-in algorithm by address.",
    "STATS 100 runs 100 trials and shows mean/stddev per state.",
    "PROFILE ON then RUN shows per-line execution time.",
    "ENTROPY 0 measures entanglement of qubit 0 with the rest.",
    "EXPECT ZZ 0 1 computes the two-qubit ZZ correlation.",
    "BANK 1 switches to a second program slot (auto-saves current).",
    "SCREEN 3 auto-displays Bloch spheres after every RUN.",
    "CONSISTENCY cross-checks statevector, density, Bloch, and histogram.",
    "FOR I = 0 TO 7 / H I / NEXT I applies H to 8 qubits in a loop.",
    "FIND \"CX\" searches your program for a string.",
    "CHECKSUM gives an MD5 hash for program verification.",
    "10 H 0 : CX 0,1 : RZ PI/4, 0 puts three gates on one line.",
]


def tip_of_the_day() -> str:
    """Return a random tip."""
    return random.choice(_TIPS)


# ═══════════════════════════════════════════════════════════════════════
# "Did you mean?" (#4)
# ═══════════════════════════════════════════════════════════════════════

_ALL_COMMANDS = [
    'RUN', 'LIST', 'NEW', 'SAVE', 'LOAD', 'QUBITS', 'SHOTS', 'METHOD',
    'DEF', 'REG', 'LET', 'STEP', 'STATE', 'HIST', 'BLOCH', 'PROBS',
    'DEMO', 'DELETE', 'RENUM', 'DEFS', 'REGS', 'VARS', 'HELP',
    'CIRCUIT', 'LOCC', 'SEND', 'SHARE', 'SWEEP', 'INCLUDE', 'EXPORT',
    'DECOMPOSE', 'NOISE', 'EXPECT', 'DENSITY', 'ENTROPY', 'CSV',
    'MEASURE', 'BARRIER', 'PRINT', 'INPUT', 'DIM', 'UNITARY',
    'LOCCINFO', 'RAM', 'UNDO', 'CLEAR', 'PEEK', 'POKE', 'SYS',
    'DUMP', 'MAP', 'CATALOG', 'MONITOR', 'SCREEN', 'COLOR', 'CLS',
    'LOCATE', 'PLAY', 'PROMPT', 'BREAK', 'WATCH', 'PROFILE', 'STATS',
    'REWIND', 'FORWARD', 'HISTORY', 'TRON', 'TROFF', 'CONT',
    'AUTO', 'EDIT', 'COPY', 'MOVE', 'FIND', 'REPLACE', 'BANK',
    'CHAIN', 'MERGE', 'CHECKSUM', 'VERSION', 'SEED', 'PROBE',
    'RESTORE', 'OPEN', 'CLOSE', 'IMPORT', 'SAMPLE', 'ESTIMATE',
    'BENCH', 'SET_STATE', 'CIRCUIT_DEF', 'APPLY_CIRCUIT',
    'CONSISTENCY', 'COMPARE', 'HEATMAP', 'ANIMATE', 'QUIZ',
    'DIFF', 'PLOT', 'THEME', 'CLIP', 'EXPLAIN',
    # Gates
    'H', 'X', 'Y', 'Z', 'S', 'T', 'SDG', 'TDG', 'SX', 'ID',
    'RX', 'RY', 'RZ', 'P', 'U', 'CX', 'CZ', 'CY', 'CH',
    'SWAP', 'DCX', 'ISWAP', 'CRX', 'CRY', 'CRZ', 'CP',
    'RXX', 'RYY', 'RZZ', 'CCX', 'CSWAP',
]


def did_you_mean(word: str) -> str | None:
    """Return a suggestion for a misspelled command, or None."""
    w = word.upper()
    if w in _ALL_COMMANDS:
        return None
    matches = difflib.get_close_matches(w, _ALL_COMMANDS, n=1, cutoff=0.6)
    return matches[0] if matches else None


# ═══════════════════════════════════════════════════════════════════════
# Color themes (#19)
# ═══════════════════════════════════════════════════════════════════════

THEMES = {
    'default': {'reset': '\033[0m', 'gate': '\033[36m', 'flow': '\033[33m',
                'comment': '\033[2m', 'string': '\033[32m', 'number': '\033[35m',
                'error': '\033[31m', 'bar_hi': '\033[32m', 'bar_mid': '\033[33m',
                'bar_lo': '\033[2m', 'prompt': ''},
    'retro':   {'reset': '\033[0m', 'gate': '\033[32m', 'flow': '\033[32m',
                'comment': '\033[2;32m', 'string': '\033[32m', 'number': '\033[32m',
                'error': '\033[1;32m', 'bar_hi': '\033[1;32m', 'bar_mid': '\033[32m',
                'bar_lo': '\033[2;32m', 'prompt': '\033[32m'},
    'none':    {'reset': '', 'gate': '', 'flow': '', 'comment': '', 'string': '',
                'number': '', 'error': '', 'bar_hi': '', 'bar_mid': '', 'bar_lo': '',
                'prompt': ''},
}


# ═══════════════════════════════════════════════════════════════════════
# Braille Bloch sphere (#6)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# QoL Mixin
# ═══════════════════════════════════════════════════════════════════════

class QoLMixin:
    """Quality-of-life commands for QBasicTerminal.

    Provides: COMPARE, HEATMAP, ANIMATE, QUIZ, DIFF, PLOT, THEME,
    CLIP, EXPLAIN, and enhanced display features.
    """

    def _init_qol(self) -> None:
        self._theme: dict[str, str] = THEMES['default']
        self._theme_name: str = 'default'

    # ── #4: "Did you mean?" in dispatch ──────────────────────────────

    def _suggest_command(self, word: str) -> None:
        """Print a suggestion if the word is close to a known command."""
        suggestion = did_you_mean(word)
        if suggestion:
            self.io.writeln(f"  Did you mean: {suggestion}?")

    # ── #5: Status bar prompt ─────────────────────────────────────────

    def _status_prompt(self) -> str:
        """Build a prompt showing current config."""
        parts = [f"{self.num_qubits}q"]
        if self.locc_mode:
            parts.append("LOCC")
        if self._noise_model:
            parts.append("noisy")
        return ' '.join(parts) + ' ] '

    # ── #6: Braille Bloch sphere ──────────────────────────────────────

    def cmd_draw(self, rest: str) -> None:
        """DRAW [qubit] — braille Bloch sphere."""
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

    # ── #7: Color-coded LIST ──────────────────────────────────────────

    def cmd_list_colored(self) -> None:
        """LIST with syntax highlighting."""
        if not self.program:
            self.io.writeln("EMPTY PROGRAM")
            return
        t = self._theme
        from qubasic_core.gates import GATE_TABLE, GATE_ALIASES
        gate_names = set(GATE_TABLE.keys()) | set(GATE_ALIASES.keys())
        flow_kw = {'FOR', 'NEXT', 'WHILE', 'WEND', 'IF', 'THEN', 'ELSE',
                    'GOTO', 'GOSUB', 'RETURN', 'END', 'DO', 'LOOP', 'EXIT',
                    'SUB', 'FUNCTION', 'CALL', 'SELECT', 'CASE'}
        for num in sorted(self.program.keys()):
            raw = self.program[num]
            upper = raw.strip().upper()
            if upper.startswith('REM') or upper.startswith("'"):
                colored = f"{t['comment']}{raw}{t['reset']}"
            else:
                words = raw.split()
                parts = []
                for w in words:
                    wu = w.upper().rstrip(',').rstrip(':')
                    if wu in gate_names:
                        parts.append(f"{t['gate']}{w}{t['reset']}")
                    elif wu in flow_kw:
                        parts.append(f"{t['flow']}{w}{t['reset']}")
                    elif w.startswith('"') or w.startswith("'"):
                        parts.append(f"{t['string']}{w}{t['reset']}")
                    else:
                        parts.append(w)
                colored = ' '.join(parts)
            self.io.writeln(f"  {num:5d}  {colored}")

    # ── #8: Gate throughput in RUN summary ────────────────────────────
    # (Injected into cmd_run output — see terminal.py modification)

    # ── #9: COMPARE ──────────────────────────────────────────────────

    def cmd_compare(self, rest: str) -> None:
        """COMPARE method1 method2 — run circuit with two methods and diff results."""
        parts = rest.split()
        if len(parts) < 2:
            self.io.writeln("?USAGE: COMPARE <method1> <method2>")
            return
        m1, m2 = parts[0].lower(), parts[1].lower()
        old_method = self.sim_method
        results = {}
        for m in [m1, m2]:
            self.sim_method = m
            old_io = self.io

            class _NullIO:
                def write(self, t: str) -> None: pass
                def writeln(self, t: str) -> None: pass
                def read_line(self, p: str) -> str: return ''
            self.io = _NullIO()
            try:
                self.cmd_run()
            finally:
                self.io = old_io
            results[m] = dict(self.last_counts) if self.last_counts else {}
        self.sim_method = old_method
        # Display comparison
        all_states = sorted(set(list(results[m1].keys()) + list(results[m2].keys())))
        total1 = sum(results[m1].values()) or 1
        total2 = sum(results[m2].values()) or 1
        self.io.writeln(f"\n  {'State':>12} {m1:>12} {m2:>12} {'diff':>8}")
        for s in all_states[:32]:
            c1 = results[m1].get(s, 0)
            c2 = results[m2].get(s, 0)
            p1, p2 = c1 / total1, c2 / total2
            diff = p2 - p1
            self.io.writeln(f"  |{s}\u27E9 {p1:>11.3%} {p2:>11.3%} {diff:>+7.3%}")
        self.io.writeln('')

    # ── #10: HEATMAP ─────────────────────────────────────────────────

    def cmd_heatmap(self, rest: str = '') -> None:
        """HEATMAP — qubit-qubit entanglement entropy grid."""
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE -- RUN first")
            return
        n = self._active_nqubits
        n_show = min(n, 12)
        self.io.writeln(f"\n  Entanglement heatmap ({n_show} qubits):")
        _shades = ' \u2591\u2592\u2593\u2588'
        header = '     ' + ''.join(f'{q:>4}' for q in range(n_show))
        self.io.writeln(header)
        sv_flat = np.ascontiguousarray(sv).ravel()
        for qi in range(n_show):
            row = f'  {qi:>2} '
            for qj in range(n_show):
                if qi == qj:
                    row += '  - '
                    continue
                keep = sorted({qi, qj})
                trace_out = [q for q in range(n) if q not in keep]
                psi_t = sv_flat.reshape([2] * n)
                if trace_out:
                    rho = np.tensordot(psi_t, psi_t.conj(), axes=(trace_out, trace_out))
                else:
                    rho = np.outer(sv_flat, sv_flat.conj())
                rho = rho.reshape(4, 4)
                ev = np.linalg.eigvalsh(rho)
                ev = ev[ev > 1e-14]
                S = -np.sum(ev * np.log2(ev)) if len(ev) > 0 else 0.0
                idx = min(len(_shades) - 1, int(S * 2))
                row += f' {_shades[idx]}{S:.0f} ' if S > 0.05 else '  . '
            self.io.writeln(row)
        self.io.writeln('')

    # ── #12: ANIMATE ──────────────────────────────────────────────────

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

        class _NullIO:
            def write(self, t: str) -> None: pass
            def writeln(self, t: str) -> None: pass
            def read_line(self, p: str) -> str: return ''

        for i, val in enumerate(values):
            self.variables[var] = val
            old_io = self.io
            self.io = _NullIO()
            try:
                self.cmd_run()
            finally:
                self.io = old_io
            if self.last_counts:
                ranked = sorted(self.last_counts.items(), key=lambda x: -x[1])
                total = sum(self.last_counts.values())
                top = ranked[0]
                bar = '\u2588' * int(30 * top[1] / total)
                frame = f"\r  {var}={val:>8.3f}  |{top[0]}\u27E9 {top[1]/total:>5.1%} {bar:<30}"
                old_io.write(frame)
            time.sleep(delay)
        self.io.writeln('')

    # ── #13: QUIZ ─────────────────────────────────────────────────────

    def cmd_quiz(self, rest: str = '') -> None:
        """QUIZ — quantum computing quiz."""
        quizzes = [
            ("H|0> = ?", ["|+>", "|1>", "|->", "|0>"], 0),
            ("CX|10> = ?", ["|11>", "|10>", "|00>", "|01>"], 0),
            ("X|0> = ?", ["|1>", "|0>", "|+>", "|->"], 0),
            ("Z|+> = ?", ["|->", "|+>", "|0>", "|1>"], 0),
            ("H|1> = ?", ["|->", "|+>", "|0>", "|1>"], 0),
            ("What gate creates superposition?", ["H", "X", "Z", "CX"], 0),
            ("Bell state is:", ["|00>+|11>", "|00>+|01>", "|01>+|10>", "|00>-|11>"], 0),
            ("CNOT flips target when control is:", ["|1>", "|0>", "|+>", "|->"], 0),
        ]
        q = random.choice(quizzes)
        prompt, options, correct = q
        self.io.writeln(f"\n  QUIZ: {prompt}")
        indices = list(range(len(options)))
        random.shuffle(indices)
        correct_label = None
        for label_idx, orig_idx in enumerate(indices):
            letter = chr(65 + label_idx)
            marker = ""
            if orig_idx == correct:
                correct_label = letter
            self.io.writeln(f"    {letter}) {options[orig_idx]}")
        try:
            answer = self.io.read_line("  Your answer: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            self.io.writeln("")
            return
        if answer == correct_label:
            self.io.writeln("  Correct!")
        else:
            self.io.writeln(f"  Wrong -- the answer is {correct_label}) {options[correct]}")
        self.io.writeln('')

    # ── #15: DIFF ─────────────────────────────────────────────────────

    def cmd_diff(self, rest: str) -> None:
        """DIFF <slot> — diff current program against another bank slot."""
        if not rest.strip():
            self.io.writeln("?USAGE: DIFF <slot>")
            return
        slot = int(rest.strip())
        other = self._program_slots.get(slot, {})
        if not other:
            self.io.writeln(f"?BANK {slot} is empty")
            return
        current_lines = [f"{n} {self.program[n]}" for n in sorted(self.program.keys())]
        other_lines = [f"{n} {other[n]}" for n in sorted(other.keys())]
        diff = list(difflib.unified_diff(other_lines, current_lines,
                                          fromfile=f'bank {slot}', tofile='current',
                                          lineterm=''))
        if diff:
            for line in diff:
                self.io.writeln(f"  {line}")
        else:
            self.io.writeln(f"  No differences with bank {slot}")

    # ── #16: Circuit complexity readout ────────────────────────────────

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

    # ── #17: PLOT ─────────────────────────────────────────────────────

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

        class _NullIO:
            def write(self, t: str) -> None: pass
            def writeln(self, t: str) -> None: pass
            def read_line(self, p: str) -> str: return ''

        xs, ys = [], []
        for val in values:
            self.variables[var] = val
            old_io = self.io
            self.io = _NullIO()
            try:
                self.cmd_run()
            finally:
                self.io = old_io
            if self.last_counts:
                total = sum(self.last_counts.values())
                top_state = max(self.last_counts, key=self.last_counts.get)
                xs.append(val)
                ys.append(self.last_counts[top_state] / total)

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

    # ── #18: UNDO with preview ────────────────────────────────────────

    def cmd_undo_preview(self) -> None:
        """UNDO — show what will change and restore."""
        if not self._undo_stack:
            self.io.writeln("?NOTHING TO UNDO")
            return
        prev = self._undo_stack[-1]
        # Show diff
        added = set(self.program.keys()) - set(prev.keys())
        removed = set(prev.keys()) - set(self.program.keys())
        changed = {k for k in set(self.program.keys()) & set(prev.keys())
                    if self.program[k] != prev[k]}
        if added:
            self.io.writeln(f"  Undo will remove: {sorted(added)}")
        if removed:
            self.io.writeln(f"  Undo will restore: {sorted(removed)}")
        if changed:
            for k in sorted(changed):
                self.io.writeln(f"  Undo will revert {k}: {self.program[k]} -> {prev[k]}")
        # Apply
        self.program = self._undo_stack.pop()
        self._parsed.clear()
        self.io.writeln(f"UNDO ({len(self.program)} lines)")

    # ── #19: THEME ────────────────────────────────────────────────────

    def cmd_theme(self, rest: str) -> None:
        """THEME [name] — switch color theme (default, retro, none)."""
        name = rest.strip().lower() if rest.strip() else ''
        if not name:
            self.io.writeln(f"  Current theme: {self._theme_name}")
            self.io.writeln(f"  Available: {', '.join(THEMES.keys())}")
            return
        if name not in THEMES:
            self.io.writeln(f"?Unknown theme: {name}. Available: {', '.join(THEMES.keys())}")
            return
        self._theme = THEMES[name]
        self._theme_name = name
        self.io.writeln(f"THEME = {name}")

    # ── #22: EXPLAIN ──────────────────────────────────────────────────

    def cmd_explain(self, rest: str = '') -> None:
        """EXPLAIN — describe each program line in plain English."""
        if not self.program:
            self.io.writeln("EMPTY PROGRAM")
            return
        from qubasic_core.gates import GATE_TABLE, GATE_ALIASES
        gate_desc = {
            'H': 'put qubit {q} in superposition (Hadamard)',
            'X': 'flip qubit {q} (NOT gate)',
            'Y': 'apply Pauli-Y to qubit {q}',
            'Z': 'flip the phase of qubit {q}',
            'S': 'apply pi/2 phase to qubit {q}',
            'T': 'apply pi/4 phase to qubit {q}',
            'CX': 'entangle qubits {q} (CNOT)',
            'CZ': 'apply controlled-Z on qubits {q}',
            'CCX': 'apply Toffoli (AND gate) on qubits {q}',
            'SWAP': 'swap qubits {q}',
            'RX': 'rotate qubit around X axis',
            'RY': 'rotate qubit around Y axis',
            'RZ': 'rotate qubit around Z axis',
        }
        flow_desc = {
            'FOR': 'begin loop',
            'NEXT': 'end of loop',
            'WHILE': 'loop while condition holds',
            'WEND': 'end of WHILE loop',
            'IF': 'conditional branch',
            'GOTO': 'jump to line',
            'GOSUB': 'call subroutine',
            'RETURN': 'return from subroutine',
            'END': 'stop execution',
            'MEASURE': 'measure all qubits',
            'BARRIER': 'optimization barrier',
            'PRINT': 'output text or value',
            'LET': 'set variable',
            'DIM': 'create array',
        }
        for num in sorted(self.program.keys()):
            stmt = self.program[num].strip()
            upper = stmt.upper()
            word = upper.split()[0] if stmt.split() else ''
            word = GATE_ALIASES.get(word, word)
            if upper.startswith('REM') or upper.startswith("'"):
                desc = "(comment)"
            elif word in gate_desc:
                args = stmt.split(None, 1)[1] if ' ' in stmt else ''
                desc = gate_desc[word].format(q=args)
            elif word in flow_desc:
                desc = flow_desc[word]
            elif ':' in stmt:
                desc = f"multiple operations ({stmt.count(':') + 1} parts)"
            elif word.startswith('@'):
                desc = f"gate on LOCC register {word[1:]}"
            elif word == 'SEND':
                desc = "measure and send classical bit"
            elif word == 'SHARE':
                desc = "create shared entanglement"
            else:
                desc = stmt
            self.io.writeln(f"  {num:5d}  {desc}")

    # ── #24: CLIP ─────────────────────────────────────────────────────

    def cmd_clip(self, rest: str = '') -> None:
        """CLIP — copy last results to clipboard (if available)."""
        lines = []
        if self.last_counts:
            total = sum(self.last_counts.values())
            for s, c in sorted(self.last_counts.items(), key=lambda x: -x[1]):
                lines.append(f"|{s}> {c}/{total} ({100*c/total:.1f}%)")
        elif self.last_sv is not None:
            n = self._active_nqubits
            sv = np.ascontiguousarray(self.last_sv).ravel()
            for i, a in enumerate(sv):
                if abs(a) > 1e-8:
                    lines.append(f"|{format(i, f'0{n}b')}> {a.real:+.4f}{a.imag:+.4f}j")
        else:
            self.io.writeln("?NO RESULTS TO COPY")
            return
        text = '\n'.join(lines)
        try:
            if sys.platform == 'win32':
                import subprocess
                subprocess.run(['clip'], input=text.encode(), check=True)
            elif sys.platform == 'darwin':
                import subprocess
                subprocess.run(['pbcopy'], input=text.encode(), check=True)
            else:
                import subprocess
                subprocess.run(['xclip', '-selection', 'clipboard'],
                               input=text.encode(), check=True, timeout=2)
            self.io.writeln(f"COPIED {len(lines)} lines to clipboard")
        except Exception:
            self.io.writeln("?Clipboard not available. Output:")
            for line in lines[:20]:
                self.io.writeln(f"  {line}")
            if len(lines) > 20:
                self.io.writeln(f"  ... ({len(lines) - 20} more)")
