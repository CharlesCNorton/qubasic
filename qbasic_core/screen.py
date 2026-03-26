"""QBASIC screen control — SCREEN, COLOR, CLS, LOCATE, PLAY, prompt."""

from __future__ import annotations

import sys


class ScreenMixin:
    """Screen control commands for QBasicTerminal.

    Requires: TerminalProtocol — uses self._screen_mode, self.last_counts,
    self.last_sv, self.last_circuit, self.print_histogram(),
    self._print_statevector(), self._print_bloch_single(),
    self.cmd_density(), self.cmd_circuit().
    """

    def _init_screen(self) -> None:
        self._color_fg: str = ''
        self._color_bg: str = ''
        self._prompt: str = '] '

    def cmd_screen(self, rest: str) -> None:
        """SCREEN mode — set display mode.
        0=text  1=histogram  2=statevector  3=Bloch  4=density  5=circuit"""
        if not rest.strip():
            names = {0: 'text', 1: 'histogram', 2: 'statevector',
                     3: 'Bloch', 4: 'density', 5: 'circuit'}
            print(f"SCREEN = {self._screen_mode} ({names.get(self._screen_mode, '?')})")
            return
        mode = int(rest.strip())
        if mode < 0 or mode > 5:
            print("?SCREEN 0-5")
            return
        self._screen_mode = mode
        names = {0: 'text', 1: 'histogram', 2: 'statevector',
                 3: 'Bloch', 4: 'density', 5: 'circuit'}
        print(f"SCREEN {mode} ({names[mode]})")

    def _auto_display(self) -> None:
        """Display results using current SCREEN mode after RUN."""
        mode = getattr(self, '_screen_mode', 0)
        if mode == 0:
            return  # text mode: default behavior (histogram already shown by cmd_run)
        if mode == 1:
            if self.last_counts:
                self.print_histogram(self.last_counts)
        elif mode == 2:
            if self.last_sv is not None:
                self._print_statevector(self.last_sv)
        elif mode == 3:
            if self.last_sv is not None:
                from qbasic_core.engine import MAX_BLOCH_DISPLAY
                for q in range(min(self.num_qubits, MAX_BLOCH_DISPLAY)):
                    self._print_bloch_single(self.last_sv, q)
                    print()
        elif mode == 4:
            if hasattr(self, 'cmd_density'):
                self.cmd_density()
        elif mode == 5:
            if hasattr(self, 'cmd_circuit'):
                self.cmd_circuit()

    def cmd_color(self, rest: str) -> None:
        """COLOR foreground[, background] — set terminal colors."""
        from qbasic_core.engine import RE_COLOR
        m = RE_COLOR.match(f"COLOR {rest}")
        if not m:
            print("?USAGE: COLOR <fg>[, <bg>]")
            return
        self._color_fg = m.group(1).lower()
        self._color_bg = m.group(2).lower() if m.group(2) else ''
        # Apply via ANSI if Rich is not available
        colors = {
            'black': 30, 'red': 31, 'green': 32, 'yellow': 33,
            'blue': 34, 'magenta': 35, 'cyan': 36, 'white': 37,
        }
        fg_code = colors.get(self._color_fg, 37)
        if self._color_bg:
            bg_code = colors.get(self._color_bg, 40) + 10
            print(f"\033[{fg_code};{bg_code}m", end='')
        else:
            print(f"\033[{fg_code}m", end='')
        print(f"COLOR {self._color_fg}" + (f", {self._color_bg}" if self._color_bg else ""))

    def cmd_cls(self) -> None:
        """CLS — clear screen."""
        print('\033[2J\033[H', end='', flush=True)

    def cmd_locate(self, rest: str) -> None:
        """LOCATE row, col — position cursor."""
        from qbasic_core.engine import RE_LOCATE
        m = RE_LOCATE.match(f"LOCATE {rest}")
        if not m:
            print("?USAGE: LOCATE <row>, <col>")
            return
        row, col = int(m.group(1)), int(m.group(2))
        print(f"\033[{row};{col}H", end='', flush=True)

    def cmd_play(self, rest: str = '') -> None:
        """PLAY — terminal bell/beep."""
        count = 1
        if rest.strip():
            try:
                count = int(rest.strip())
            except ValueError:
                pass
        for _ in range(count):
            print('\a', end='', flush=True)

    def cmd_prompt(self, rest: str) -> None:
        """PROMPT <string> — set the REPL prompt."""
        if not rest.strip():
            print(f"PROMPT = {self._prompt!r}")
            return
        text = rest.strip()
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        self._prompt = text
        print(f"PROMPT = {self._prompt!r}")
