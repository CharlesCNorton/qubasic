"""QUBASIC devices: the graphics framebuffer and statements, sound,
screen control and files.
"""

from __future__ import annotations

import colorsys
import fnmatch
import io
import math
import os
import re
import shutil
import struct
import time
import wave
import zlib
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import IO, Any

import numpy as np

from qubasic_core.engine import NullIO
from qubasic_core.expression import Record, fixed_length, is_string_kind, next_column
from qubasic_core.host import LanguageHost
from qubasic_core.parser import RE_LPRINT, _top_split

# ---------------------------------------------------------------------------
# QBASIC graphics as a framebuffer: the SCREEN modes, their palettes, and the
# drawing operations PSET, LINE, CIRCLE, PAINT, DRAW, POINT, VIEW, WINDOW,
# PMAP, GET and PUT, with the picture shown as terminal text or saved as PNG.
#
# Coordinates follow QBASIC. Physical pixels count from the top left of the
# screen; VIEW restricts drawing to a viewport, and WINDOW gives logical
# coordinates that map onto it (y grows upward unless WINDOW SCREEN). The
# graphics cursor is the last point referenced, the origin of STEP and of DRAW.
# ---------------------------------------------------------------------------

# mode -> (width, height, colours)
MODES: dict[int, tuple[int, int, int]] = {
    1: (320, 200, 4), 2: (640, 200, 2), 7: (320, 200, 16), 8: (640, 200, 16),
    9: (640, 350, 16), 10: (640, 350, 4), 11: (640, 480, 2), 12: (640, 480, 16),
    13: (320, 200, 256),
}

EGA16: list[tuple[int, int, int]] = [
    (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170), (170, 0, 0), (170, 0, 170),
    (170, 85, 0), (170, 170, 170), (85, 85, 85), (85, 85, 255), (85, 255, 85),
    (85, 255, 255), (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255),
]


def _vga256() -> list[tuple[int, int, int]]:
    """The VGA default palette: the 16 EGA colours, 16 greys, 216 hues at
    three brightnesses and saturations, then black."""
    pal = list(EGA16)
    for k in range(16):
        grey = round(255 * k / 15)
        pal.append((grey, grey, grey))
    for value in (1.0, 0.45, 0.25):
        for sat in (1.0, 0.55, 0.3):
            for h in range(24):
                r, g, b = colorsys.hsv_to_rgb((2 / 3 + h / 24) % 1.0, sat, value)
                pal.append((round(r * 255), round(g * 255), round(b * 255)))
    pal += [(0, 0, 0)] * (256 - len(pal))
    return pal


def default_palette(mode: int) -> list[tuple[int, int, int]]:
    if mode == 13:
        return _vga256()
    if mode == 1:
        return [EGA16[0], EGA16[3], EGA16[5], EGA16[7]]
    if mode in (2, 11):
        return [EGA16[0], EGA16[15]]
    if mode == 10:
        return [EGA16[0], EGA16[8], EGA16[7], EGA16[15]]
    return list(EGA16)


def ega64(n: int) -> tuple[int, int, int]:
    """An EGA colour number 0-63 (bits rgbRGB) as RGB."""
    def level(hi: int, lo: int) -> int:
        return 170 * hi + 85 * lo
    return (level((n >> 2) & 1, (n >> 5) & 1), level((n >> 1) & 1, (n >> 4) & 1),
            level(n & 1, (n >> 3) & 1))


def vga_long(c: int) -> tuple[int, int, int]:
    """A PALETTE value 65536*blue + 256*green + red, components 0-63, as RGB."""
    r, g, b = c & 0x3F, (c >> 8) & 0x3F, (c >> 16) & 0x3F
    return (round(r * 255 / 63), round(g * 255 / 63), round(b * 255 / 63))


class Canvas:
    """One SCREEN's framebuffer and drawing state."""

    def __init__(self, mode: int) -> None:
        if mode not in MODES:
            raise ValueError(f"SCREEN {mode} is not a graphics mode ({', '.join(map(str, sorted(MODES)))})")
        self.mode = mode
        self.width, self.height, self.colors = MODES[mode]
        self.pixels = np.zeros((self.height, self.width), dtype=np.uint16)
        self.palette = default_palette(mode)
        self.fg = self.colors - 1 if mode != 13 else 15
        self.bg = 0
        self.view = (0, 0, self.width - 1, self.height - 1)
        self.view_screen = False         # VIEW SCREEN: coordinates stay absolute
        self.window: tuple[float, float, float, float] | None = None
        self.window_screen = False       # WINDOW SCREEN: y grows downward
        self.last = (float(self.width // 2), float(self.height // 2))   # logical
        self.draw_angle = 0.0            # DRAW A/TA, degrees counterclockwise
        self.draw_scale = 4              # DRAW S: units of 1/4 pixel
        self.drawn = False

    # ── coordinates ─────────────────────────────────────────────────────

    def to_physical(self, x: float, y: float) -> tuple[float, float]:
        vx1, vy1, vx2, vy2 = self.view
        if self.window is not None:
            wx1, wy1, wx2, wy2 = self.window
            px = vx1 + (x - wx1) * (vx2 - vx1) / (wx2 - wx1)
            if self.window_screen:
                py = vy1 + (y - wy1) * (vy2 - vy1) / (wy2 - wy1)
            else:
                py = vy2 - (y - wy1) * (vy2 - vy1) / (wy2 - wy1)
            return px, py
        if self.view_screen:
            return x, y
        return vx1 + x, vy1 + y

    def to_logical(self, px: float, py: float) -> tuple[float, float]:
        vx1, vy1, vx2, vy2 = self.view
        if self.window is not None:
            wx1, wy1, wx2, wy2 = self.window
            x = wx1 + (px - vx1) * (wx2 - wx1) / (vx2 - vx1)
            if self.window_screen:
                y = wy1 + (py - vy1) * (wy2 - wy1) / (vy2 - vy1)
            else:
                y = wy1 + (vy2 - py) * (wy2 - wy1) / (vy2 - vy1)
            return x, y
        if self.view_screen:
            return px, py
        return px - vx1, py - vy1

    def scale_x(self) -> float:
        """Physical pixels per logical unit along x."""
        if self.window is None:
            return 1.0
        wx1, _, wx2, _ = self.window
        vx1, _, vx2, _ = self.view
        return abs((vx2 - vx1) / (wx2 - wx1))

    def pmap(self, value: float, fn: int) -> float:
        if fn == 0:
            return round(self.to_physical(value, 0)[0])
        if fn == 1:
            return round(self.to_physical(0, value)[1])
        if fn == 2:
            return self.to_logical(value, 0)[0]
        if fn == 3:
            return self.to_logical(0, value)[1]
        raise ValueError("PMAP function must be 0-3")

    def set_view(self, box: tuple[int, int, int, int] | None, screen: bool = False,
                 fill: int | None = None, border: int | None = None) -> None:
        if box is None:
            self.view, self.view_screen = (0, 0, self.width - 1, self.height - 1), False
            return
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0, x1), min(self.width - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(self.height - 1, y2)))
        if border is not None:
            self._rect_physical(x1 - 1, y1 - 1, x2 + 1, y2 + 1, border, fill=False, clip=False)
        if fill is not None:
            self.pixels[y1:y2 + 1, x1:x2 + 1] = fill
        self.view, self.view_screen = (x1, y1, x2, y2), screen

    def set_window(self, box: tuple[float, float, float, float] | None, screen: bool = False) -> None:
        if box is None:
            self.window = None
            return
        x1, y1, x2, y2 = box
        if x1 == x2 or y1 == y2:
            raise ValueError("WINDOW needs a non-empty rectangle")
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        self.window, self.window_screen = (x1, y1, x2, y2), screen

    # ── pixels ──────────────────────────────────────────────────────────

    def _plot(self, px: int, py: int, c: int, clip: bool = True) -> None:
        if clip:
            vx1, vy1, vx2, vy2 = self.view
            if not (vx1 <= px <= vx2 and vy1 <= py <= vy2):
                return
        if 0 <= px < self.width and 0 <= py < self.height:
            self.pixels[py, px] = c % self.colors if self.colors < 256 else c & 0xFF
            self.drawn = True

    def color_or_fg(self, c: int | None) -> int:
        return self.fg if c is None else int(c)

    def pset(self, x: float, y: float, c: int | None = None) -> None:
        px, py = self.to_physical(x, y)
        self._plot(round(px), round(py), self.color_or_fg(c))
        self.last = (x, y)

    def preset(self, x: float, y: float, c: int | None = None) -> None:
        self.pset(x, y, self.bg if c is None else c)

    def point(self, x: float, y: float) -> int:
        px, py = (round(v) for v in self.to_physical(x, y))
        vx1, vy1, vx2, vy2 = self.view
        if not (vx1 <= px <= vx2 and vy1 <= py <= vy2):
            return -1
        return int(self.pixels[py, px])

    def cursor(self, n: int) -> float:
        """POINT(n): 0/1 the physical, 2/3 the logical graphics cursor."""
        if n in (2, 3):
            return self.last[n - 2]
        if n in (0, 1):
            return round(self.to_physical(*self.last)[n])
        raise ValueError("POINT(n) takes 0-3")

    # ── lines and boxes ─────────────────────────────────────────────────

    def _line_physical(self, x1: int, y1: int, x2: int, y2: int, c: int,
                       style: int | None = None, clip: bool = True) -> None:
        dx, dy = abs(x2 - x1), -abs(y2 - y1)
        sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
        err = dx + dy
        bit = 0
        while True:
            if style is None or (style >> (15 - bit % 16)) & 1:
                self._plot(x1, y1, c, clip)
            bit += 1
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def _rect_physical(self, x1: int, y1: int, x2: int, y2: int, c: int, fill: bool,
                       style: int | None = None, clip: bool = True) -> None:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if fill:
            if clip:
                vx1, vy1, vx2, vy2 = self.view
                x1, x2, y1, y2 = max(x1, vx1), min(x2, vx2), max(y1, vy1), min(y2, vy2)
            x1, x2 = max(x1, 0), min(x2, self.width - 1)
            y1, y2 = max(y1, 0), min(y2, self.height - 1)
            if x1 <= x2 and y1 <= y2:
                self.pixels[y1:y2 + 1, x1:x2 + 1] = c % self.colors if self.colors < 256 else c & 0xFF
                self.drawn = True
            return
        for a, b, cc, d in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
            self._line_physical(a, b, cc, d, c, style, clip)

    def line(self, x1: float, y1: float, x2: float, y2: float, c: int | None = None,
             box: str | None = None, style: int | None = None) -> None:
        p1 = [round(v) for v in self.to_physical(x1, y1)]
        p2 = [round(v) for v in self.to_physical(x2, y2)]
        col = self.color_or_fg(c)
        if box in ('B', 'BF'):
            self._rect_physical(p1[0], p1[1], p2[0], p2[1], col, box == 'BF', style)
        else:
            self._line_physical(p1[0], p1[1], p2[0], p2[1], col, style)
        self.last = (x2, y2)

    # ── circles ─────────────────────────────────────────────────────────

    def default_aspect(self) -> float:
        return 4 / 3 * self.height / self.width

    def circle(self, x: float, y: float, r: float, c: int | None = None,
               start: float | None = None, end: float | None = None,
               aspect: float | None = None) -> None:
        col = self.color_or_fg(c)
        cx, cy = self.to_physical(x, y)
        rp = abs(r) * self.scale_x()
        asp = self.default_aspect() if aspect is None else float(aspect)
        if asp <= 1:
            rx, ry = rp, rp * asp
        else:
            rx, ry = rp / asp, rp
        a0 = 0.0 if start is None else start
        a1 = 2 * math.pi if end is None else end
        for a in (a0, a1):
            if abs(a) > 2 * math.pi + 1e-9:
                raise ValueError("CIRCLE angles must lie in -2*PI to 2*PI")
        radius0, radius1 = a0 < 0, a1 < 0
        a0, a1 = abs(a0), abs(a1)
        if a1 <= a0:
            a1 += 2 * math.pi
        steps = max(16, int(2 * math.pi * max(rx, ry) * (a1 - a0) / (2 * math.pi)) * 2 + 8)
        prev = None
        for k in range(steps + 1):
            t = a0 + (a1 - a0) * k / steps
            p = (round(cx + rx * math.cos(t)), round(cy - ry * math.sin(t)))
            if prev is None:
                self._plot(p[0], p[1], col)
            else:
                self._line_physical(prev[0], prev[1], p[0], p[1], col)
            prev = p
        centre = (round(cx), round(cy))
        if radius0:
            self._line_physical(centre[0], centre[1], round(cx + rx * math.cos(a0)),
                                round(cy - ry * math.sin(a0)), col)
        if radius1:
            self._line_physical(centre[0], centre[1], round(cx + rx * math.cos(a1)),
                                round(cy - ry * math.sin(a1)), col)
        self.last = (x, y)

    # ── PAINT ───────────────────────────────────────────────────────────

    def paint(self, x: float, y: float, c: int | None = None, border: int | None = None) -> None:
        """Flood-fill from (x, y) up to pixels of the border colour."""
        from scipy import ndimage
        col = self.color_or_fg(c)
        edge = col if border is None else int(border)
        px, py = (round(v) for v in self.to_physical(x, y))
        vx1, vy1, vx2, vy2 = self.view
        self.last = (x, y)
        if not (vx1 <= px <= vx2 and vy1 <= py <= vy2):
            return
        region = self.pixels[vy1:vy2 + 1, vx1:vx2 + 1]
        if region[py - vy1, px - vx1] == edge:
            return
        labels, _ = ndimage.label(region != edge)
        mask = labels == labels[py - vy1, px - vx1]
        region[mask] = col
        self.drawn = True

    # ── DRAW ────────────────────────────────────────────────────────────

    _RE_DRAW_N = re.compile(r'\s*([+-]?\d+(?:\.\d+)?)?')
    _RE_DRAW_M = re.compile(r'\s*([+-]?)\s*(\d+(?:\.\d+)?)\s*,\s*([+-]?)\s*(\d+(?:\.\d+)?)')

    def draw(self, commands: str) -> None:
        """The DRAW macro language: U D L R E F G H [n], M [+|-]x,[+|-]y,
        B (move without drawing) and N (return afterwards) prefixes, A n,
        TA n, C n, S n and P paint,border."""
        pos = 0
        blind = back = False
        text = commands.replace(';', ' ')
        while pos < len(text):
            if text[pos].isspace():
                pos += 1
                continue
            letter = text[pos].upper()
            pos += 1
            if letter in 'BN':
                if letter == 'B':
                    blind = True
                else:
                    back = True
                continue
            if letter == 'T' and pos < len(text) and text[pos].upper() == 'A':
                letter, pos = 'TA', pos + 1
            if letter == 'M':
                m2 = self._RE_DRAW_M.match(text, pos)
                if not m2:
                    raise ValueError(f"DRAW: M needs x,y at {text[pos:pos + 12]!r}")
                pos = m2.end()
                sx, xv, sy, yv = m2.groups()
                if sx or sy:
                    dx = float(xv) * (-1 if sx == '-' else 1)
                    dy = float(yv) * (-1 if sy == '-' else 1)
                    self._draw_move(dx, dy, blind, back, rotate=False)
                else:
                    self._draw_to(float(xv), float(yv), blind, back)
                blind = back = False
                continue
            m = self._RE_DRAW_N.match(text, pos)
            num_text = m.group(1) if m else None
            pos = m.end() if m else pos
            n = float(num_text) if num_text else None
            if letter in 'UDLREFGH':
                units = 1.0 if n is None else n
                step = units * self.draw_scale / 4
                dx, dy = {'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0), 'E': (1, -1),
                          'F': (1, 1), 'G': (-1, 1), 'H': (-1, -1)}[letter]
                self._draw_move(dx * step, dy * step, blind, back, rotate=True)
                blind = back = False
            elif letter == 'A':
                if n is None or int(n) not in (0, 1, 2, 3):
                    raise ValueError("DRAW: A takes 0-3")
                self.draw_angle = 90.0 * int(n)
            elif letter == 'TA':
                if n is None or not -360 <= n <= 360:
                    raise ValueError("DRAW: TA takes -360 to 360")
                self.draw_angle = float(n)
            elif letter == 'C':
                self.fg = int(n or 0)
            elif letter == 'S':
                if n is None or not 1 <= n <= 255:
                    raise ValueError("DRAW: S takes 1-255")
                self.draw_scale = int(n)
            elif letter == 'P':
                m3 = re.match(r'\s*,\s*(\d+)', text[pos:])
                if n is None or not m3:
                    raise ValueError("DRAW: P needs paint,border")
                pos += m3.end()
                self.paint(*self.last, int(n), int(m3.group(1)))
            else:
                raise ValueError(f"DRAW: unknown command {letter!r}")

    def _draw_move(self, dx: float, dy: float, blind: bool, back: bool, rotate: bool) -> None:
        if rotate and self.draw_angle:
            a = math.radians(self.draw_angle)
            dx, dy = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
        px, py = self.to_physical(*self.last)
        self._draw_to_physical(px + dx, py + dy, blind, back)

    def _draw_to(self, x: float, y: float, blind: bool, back: bool) -> None:
        px, py = self.to_physical(x, y)
        self._draw_to_physical(px, py, blind, back)

    def _draw_to_physical(self, px: float, py: float, blind: bool, back: bool) -> None:
        sx, sy = self.to_physical(*self.last)
        if not blind:
            self._line_physical(round(sx), round(sy), round(px), round(py), self.fg)
        if not back:
            self.last = self.to_logical(px, py)

    # ── GET / PUT ───────────────────────────────────────────────────────

    def get_block(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        p1 = [round(v) for v in self.to_physical(x1, y1)]
        p2 = [round(v) for v in self.to_physical(x2, y2)]
        xa, xb = sorted((p1[0], p2[0]))
        ya, yb = sorted((p1[1], p2[1]))
        if xa < 0 or ya < 0 or xb >= self.width or yb >= self.height:
            raise ValueError("GET: the rectangle leaves the screen")
        return self.pixels[ya:yb + 1, xa:xb + 1].copy()

    def put_block(self, x: float, y: float, block: np.ndarray, action: str = 'XOR') -> None:
        px, py = (round(v) for v in self.to_physical(x, y))
        h, w = block.shape
        if px < 0 or py < 0 or px + w > self.width or py + h > self.height:
            raise ValueError("PUT: the image leaves the screen")
        target = self.pixels[py:py + h, px:px + w]
        top = self.colors - 1
        if action == 'PSET':
            target[:] = block
        elif action == 'PRESET':
            target[:] = top - block
        elif action == 'AND':
            target &= block
        elif action == 'OR':
            target |= block
        elif action == 'XOR':
            target ^= block
        else:
            raise ValueError(f"PUT action {action!r} is not PSET, PRESET, AND, OR or XOR")
        self.drawn = True

    # ── screen ──────────────────────────────────────────────────────────

    def cls(self) -> None:
        vx1, vy1, vx2, vy2 = self.view
        self.pixels[vy1:vy2 + 1, vx1:vx2 + 1] = self.bg
        self.last = self.to_logical((vx1 + vx2) / 2, (vy1 + vy2) / 2)

    def set_palette(self, attribute: int | None = None, color: int | None = None) -> None:
        if attribute is None:
            self.palette = default_palette(self.mode)
            return
        if not 0 <= attribute < len(self.palette):
            raise ValueError(f"PALETTE attribute {attribute} out of range")
        assert color is not None
        if self.mode in (12, 13):
            self.palette[attribute] = vga_long(int(color))
        else:
            self.palette[attribute] = ega64(int(color) & 63)

    def rgb(self) -> np.ndarray:
        """The picture as an RGB array (height, width, 3)."""
        lut = np.array(self.palette, dtype=np.uint8)
        return lut[np.clip(self.pixels, 0, len(lut) - 1)]

    # ── output ──────────────────────────────────────────────────────────

    def text_rows(self, max_cols: int = 100, color: bool = True) -> list[str]:
        """The picture as terminal rows of half-block cells (each cell two
        pixels high), shrunk by a whole factor to fit max_cols. Each sample
        shows the pixel of its block that differs most from the background."""
        f = max(1, math.ceil(self.width / max_cols))
        rgb = self.rgb().astype(np.int32)
        bg = np.array(self.palette[self.bg], dtype=np.int32)
        h = (self.height // f) * f
        w = (self.width // f) * f
        blocks = rgb[:h, :w].reshape(h // f, f, w // f, f, 3).transpose(0, 2, 1, 3, 4).reshape(h // f, w // f, f * f, 3)
        dist = np.abs(blocks - bg).sum(axis=3)
        pick = dist.argmax(axis=2)
        small = np.take_along_axis(blocks, pick[:, :, None, None], axis=2)[:, :, 0, :]
        if small.shape[0] % 2:
            small = np.concatenate([small, np.broadcast_to(bg, (1, small.shape[1], 3))], axis=0)
        rows = []
        for top, bottom in zip(small[0::2], small[1::2]):
            if color:
                parts, prev = [], None
                for a, b in zip(top, bottom):
                    key = (tuple(a), tuple(b))
                    if key != prev:
                        parts.append(f"\x1b[38;2;{a[0]};{a[1]};{a[2]}m\x1b[48;2;{b[0]};{b[1]};{b[2]}m")
                        prev = key
                    parts.append('▀')
                rows.append(''.join(parts) + '\x1b[0m')
            else:
                ramp = ' .:-=+*#%@'
                lum = (0.299 * (top + bottom)[:, 0] + 0.587 * (top + bottom)[:, 1]
                       + 0.114 * (top + bottom)[:, 2]) / 2
                bgl = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                level = np.abs(lum - bgl) / 255 * (len(ramp) - 1)
                rows.append(''.join(ramp[int(round(v))] for v in level).rstrip())
        return rows


def png_bytes(rgb: np.ndarray) -> bytes:
    """An RGB array (height, width, 3) of uint8 as a PNG file."""
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00' + row.tobytes() for row in np.ascontiguousarray(rgb, dtype=np.uint8))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')


def parse_png(data: bytes) -> np.ndarray:
    """Decode a PNG written by png_bytes (8-bit RGB, filter 0) back to an array."""
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos, w, h, idat = 8, 0, 0, b''
    while pos < len(data):
        n = struct.unpack('>I', data[pos:pos + 4])[0]
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if tag == b'IHDR':
            w, h = struct.unpack('>II', body[:8])
        elif tag == b'IDAT':
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    stride = 1 + 3 * w
    rows = [np.frombuffer(raw[k * stride + 1:(k + 1) * stride], dtype=np.uint8).reshape(w, 3)
            for k in range(h)]
    return np.stack(rows)


# ---------------------------------------------------------------------------
# QUBASIC graphics statements: SCREEN modes 1, 2 and 7-13 with PSET, PRESET,
# LINE, CIRCLE, PAINT, DRAW, VIEW, WINDOW, PALETTE, GET and PUT on images,
# PCOPY, and the POINT and PMAP functions (the framebuffer is Canvas, above).
#
# A program's picture is drawn as colour half-block cells on a terminal and
# as text shading elsewhere. On a terminal it is redrawn in place while the
# program runs, once the program waits for time or a key or has run for half a
# second; otherwise it is shown when RUN ends. SHOW redraws it and SAVEPNG
# <file> SCREEN writes it as a PNG.
# ---------------------------------------------------------------------------

_RE_STEP_PAREN = re.compile(r'\s*(STEP\b)?\s*\(', re.IGNORECASE)
_PUT_ACTIONS = ('PSET', 'PRESET', 'AND', 'OR', 'XOR')


def _match_paren(text: str, start: int) -> int:
    """Index of the ')' closing the '(' at text[start]."""
    depth, quote = 0, False
    for k in range(start, len(text)):
        ch = text[k]
        if ch == '"':
            quote = not quote
        elif not quote:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return k
    raise ValueError("unbalanced parentheses")


class GraphicsMixin(LanguageHost):
    """QBASIC's graphics statements on a Canvas."""

    def _init_graphics(self) -> None:
        self._canvas: Canvas | None = None
        self._sprites: dict[tuple[str, int], np.ndarray] = {}
        self._gfx_live = False                      # this run draws its frames in place
        self._gfx_frame_at = 0.0                    # when the last frame was drawn
        self._gfx_shown: tuple[bytes, bytes] | None = None   # that frame's pixels and palette
        self._gfx_run_start = 0.0

    # ── helpers ─────────────────────────────────────────────────────────

    def _gfx(self) -> Canvas:
        if self._canvas is None:
            raise ValueError("ILLEGAL FUNCTION CALL: graphics need SCREEN 1, 2 or 7-13")
        return self._canvas

    def _gfx_num(self, expr: str, run_vars: MutableMapping[str, Any]) -> float:
        return float(self._eval_with_vars(expr.strip(), run_vars))

    def _gfx_opt(self, fields: list[str], k: int, run_vars: MutableMapping[str, Any]) -> float | None:
        if k < len(fields) and fields[k].strip():
            return self._gfx_num(fields[k], run_vars)
        return None

    def _gfx_coord(self, text: str, pos: int, run_vars: MutableMapping[str, Any]) -> tuple[float, float, int]:
        """[STEP](x, y) at text[pos:]: the logical point and the index after it."""
        m = _RE_STEP_PAREN.match(text, pos)
        if not m:
            raise ValueError(f"SYNTAX ERROR: expected (x, y) at {text[pos:pos + 16]!r}")
        open_at = m.end() - 1
        close = _match_paren(text, open_at)
        parts = _top_split(text[open_at + 1:close])
        if len(parts) != 2:
            raise ValueError("SYNTAX ERROR: a point is (x, y)")
        x, y = self._gfx_num(parts[0], run_vars), self._gfx_num(parts[1], run_vars)
        if m.group(1):
            lx, ly = self._gfx().last
            x, y = lx + x, ly + y
        return x, y, close + 1

    @staticmethod
    def _gfx_rest(text: str, pos: int) -> list[str]:
        """The comma-separated fields after a point ([] when none)."""
        rest = text[pos:].strip()
        if not rest:
            return []
        if not rest.startswith(','):
            raise ValueError(f"SYNTAX ERROR near {rest[:16]!r}")
        return _top_split(rest[1:])

    # ── SCREEN ──────────────────────────────────────────────────────────

    def _gfx_screen(self, mode: int) -> None:
        """Open a graphics mode with a cleared screen (SCREEN 0 closes it)."""
        if mode == 0:
            self._canvas = None
            return
        self._canvas = Canvas(mode)
        self._sprites.clear()

    def _gfx_begin_run(self) -> None:
        if self._canvas is not None:
            self._canvas.drawn = False
        self._gfx_live = False
        self._gfx_shown = None
        self._gfx_run_start = time.monotonic()

    def _gfx_present(self) -> None:
        """Show the picture a run drew (on a terminal in colour): in place when
        the run drew frames, else after its output."""
        c = self._canvas
        if c is None or not c.drawn or isinstance(self.io, NullIO):
            return
        if self._gfx_live:
            self._gfx_draw_in_place(c)
            return
        self._gfx_show()

    def _gfx_frame(self, *, start: bool = False, force: bool = False) -> None:
        """During a run on a terminal, redraw the graphics screen in place when
        it has changed, so an animation shows as it runs. Frames begin when the
        program waits for time or a key (start), or after half a second of
        running, and come at most 15 times a second unless the program is
        about to wait (force)."""
        c = self._canvas
        if c is None or not c.drawn or getattr(self.io, 'replay', False) or not self._on_terminal():
            return
        now = time.monotonic()
        if not (self._gfx_live or start or now - self._gfx_run_start >= 0.5):
            return
        if self._gfx_live and not force and now - self._gfx_frame_at < 1 / 15:
            return
        frame = (c.pixels.tobytes(), np.asarray(c.palette).tobytes())
        if frame == self._gfx_shown:
            return
        self._gfx_draw_in_place(c)
        self._gfx_shown = frame
        self._gfx_frame_at = now

    def _gfx_draw_in_place(self, c: Canvas) -> None:
        """Draw the screen at the top of the terminal and put the cursor back
        where the program's text output left it; the first frame scrolls the
        text down to make room."""
        cols = min(160, shutil.get_terminal_size((100, 40)).columns - 1)
        rows = [f"SCREEN {c.mode} ({c.width}x{c.height}, {c.colors} colours):"]
        rows += c.text_rows(max(8, cols), color=True)
        if not self._gfx_live:
            self.io.write('\n' * len(rows))
            self._gfx_live = True
        self.io.write('\x1b7\x1b[H' + ('\x1b[K\n').join(rows) + '\x1b[K\x1b8')

    def _gfx_show(self, cols: int | None = None) -> None:
        c = self._gfx()
        color = self._on_terminal()
        if cols is None:
            cols = min(160, shutil.get_terminal_size((100, 40)).columns - 1) if color else 80
        self.io.writeln(f"SCREEN {c.mode} ({c.width}x{c.height}, {c.colors} colours):")
        for row in c.text_rows(max(8, cols), color=color):
            self.io.writeln(row)

    def cmd_show(self, rest: str = '') -> None:
        """SHOW [columns] — draw the graphics screen in the terminal."""
        try:
            self._gfx_show(int(rest) if rest.strip() else None)
        except ValueError as e:
            self.io.writeln(f"?{e}")

    def _gfx_png(self) -> bytes:
        return png_bytes(self._gfx().rgb())

    # ── statements ──────────────────────────────────────────────────────

    def _gfx_pset(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = re.match(r'\s*(PSET|PRESET)\b', stmt, re.IGNORECASE)
        if not m:
            return False
        c = self._gfx()
        x, y, pos = self._gfx_coord(stmt, m.end(), run_vars)
        fields = self._gfx_rest(stmt, pos)
        col = self._gfx_opt(fields, 0, run_vars)
        colour = None if col is None else int(col)
        if m.group(1).upper() == 'PSET':
            c.pset(x, y, colour)
        else:
            c.preset(x, y, colour)
        return True

    def _gfx_line(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = re.match(r'\s*LINE\b\s*', stmt, re.IGNORECASE)
        if not m or re.match(r'\s*LINE\s+INPUT\b', stmt, re.IGNORECASE):
            return False
        c = self._gfx()
        pos = m.end()
        if stmt[pos:pos + 1] == '-':
            x1, y1 = c.last
            pos += 1
        else:
            x1, y1, pos = self._gfx_coord(stmt, pos, run_vars)
            dash = re.match(r'\s*-', stmt[pos:])
            if not dash:
                raise ValueError("SYNTAX ERROR: LINE (x1, y1)-(x2, y2)")
            pos += dash.end()
        c.last = (x1, y1)
        x2, y2, pos = self._gfx_coord(stmt, pos, run_vars)
        fields = self._gfx_rest(stmt, pos)
        col = self._gfx_opt(fields, 0, run_vars)
        box = fields[1].strip().upper() if len(fields) > 1 and fields[1].strip() else None
        if box not in (None, 'B', 'BF'):
            raise ValueError(f"SYNTAX ERROR: LINE takes B or BF, not {box}")
        style = self._gfx_opt(fields, 2, run_vars)
        c.line(x1, y1, x2, y2, None if col is None else int(col), box,
               None if style is None else int(style) & 0xFFFF)
        return True

    def _gfx_circle(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = re.match(r'\s*CIRCLE\b', stmt, re.IGNORECASE)
        if not m:
            return False
        c = self._gfx()
        x, y, pos = self._gfx_coord(stmt, m.end(), run_vars)
        fields = self._gfx_rest(stmt, pos)
        if not fields or not fields[0].strip():
            raise ValueError("SYNTAX ERROR: CIRCLE (x, y), radius")
        r = self._gfx_num(fields[0], run_vars)
        col = self._gfx_opt(fields, 1, run_vars)
        c.circle(x, y, r, None if col is None else int(col), self._gfx_opt(fields, 2, run_vars),
                 self._gfx_opt(fields, 3, run_vars), self._gfx_opt(fields, 4, run_vars))
        return True

    def _gfx_paint(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = re.match(r'\s*PAINT\b', stmt, re.IGNORECASE)
        if not m:
            return False
        c = self._gfx()
        x, y, pos = self._gfx_coord(stmt, m.end(), run_vars)
        fields = self._gfx_rest(stmt, pos)
        paint = self._gfx_opt(fields, 0, run_vars)
        border = self._gfx_opt(fields, 1, run_vars)
        c.paint(x, y, None if paint is None else int(paint), None if border is None else int(border))
        return True

    def _gfx_draw(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = re.match(r'\s*DRAW\b\s*(.*)$', stmt, re.IGNORECASE)
        if not m or not m.group(1).strip():
            return False
        commands = self._safe_eval(m.group(1), extra_ns=run_vars)
        if not isinstance(commands, str):
            raise ValueError("TYPE MISMATCH: DRAW takes a string")
        self._gfx().draw(commands)
        return True

    def _gfx_view(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = re.match(r'\s*VIEW\b\s*(SCREEN\b)?\s*', stmt, re.IGNORECASE)
        if not m:
            return False
        if re.match(r'\s*VIEW\s+PRINT\b', stmt, re.IGNORECASE):
            return True                       # a text viewport: the terminal scrolls
        c = self._gfx()
        pos = m.end()
        if not stmt[pos:].strip():
            c.set_view(None)
            return True
        x1, y1, pos = self._gfx_coord(stmt, pos, run_vars)
        dash = re.match(r'\s*-', stmt[pos:])
        if not dash:
            raise ValueError("SYNTAX ERROR: VIEW (x1, y1)-(x2, y2)")
        x2, y2, pos = self._gfx_coord(stmt, pos + dash.end(), run_vars)
        fields = self._gfx_rest(stmt, pos)
        fill = self._gfx_opt(fields, 0, run_vars)
        border = self._gfx_opt(fields, 1, run_vars)
        c.set_view((round(x1), round(y1), round(x2), round(y2)), screen=bool(m.group(1)),
                   fill=None if fill is None else int(fill),
                   border=None if border is None else int(border))
        return True

    def _gfx_window(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = re.match(r'\s*WINDOW\b\s*(SCREEN\b)?\s*', stmt, re.IGNORECASE)
        if not m:
            return False
        c = self._gfx()
        pos = m.end()
        if not stmt[pos:].strip():
            c.set_window(None)
            return True
        saved, c.window = c.window, None      # the corners are plain numbers
        try:
            x1, y1, pos = self._gfx_coord(stmt, pos, run_vars)
            dash = re.match(r'\s*-', stmt[pos:])
            if not dash:
                raise ValueError("SYNTAX ERROR: WINDOW (x1, y1)-(x2, y2)")
            x2, y2, pos = self._gfx_coord(stmt, pos + dash.end(), run_vars)
        except ValueError:
            c.window = saved
            raise
        c.set_window((x1, y1, x2, y2), screen=bool(m.group(1)))
        return True

    def _gfx_palette(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = re.match(r'\s*PALETTE\b\s*(USING\b)?\s*(.*)$', stmt, re.IGNORECASE)
        if not m:
            return False
        c = self._gfx()
        if m.group(1):
            name = re.match(r'([A-Za-z_]\w*[%&!#]?)', m.group(2).strip())
            if not name or name.group(1) not in self.arrays:
                raise ValueError("PALETTE USING needs an array")
            values = list(self.arrays[name.group(1)])
            for attr, colour in enumerate(values[:len(c.palette)]):
                if int(colour) >= 0:
                    c.set_palette(attr, int(colour))
            return True
        fields = _top_split(m.group(2)) if m.group(2).strip() else []
        if not fields:
            c.set_palette()
        elif len(fields) == 2:
            c.set_palette(int(self._gfx_num(fields[0], run_vars)), int(self._gfx_num(fields[1], run_vars)))
        else:
            raise ValueError("SYNTAX ERROR: PALETTE [attribute, colour]")
        return True

    def _gfx_pcopy(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """PCOPY source, destination: one display page here, so nothing moves."""
        return bool(re.match(r'\s*PCOPY\b', stmt, re.IGNORECASE))

    def _gfx_array_target(self, text: str, run_vars: MutableMapping[str, Any]) -> tuple[str, int]:
        m = re.match(r'\s*([A-Za-z_]\w*[%&!#]?)\s*(?:\((.*)\))?\s*$', text)
        if not m:
            raise ValueError(f"SYNTAX ERROR: expected an array, got {text.strip()!r}")
        name = m.group(1)
        index = int(self._gfx_num(m.group(2), run_vars)) if m.group(2) and m.group(2).strip() else 0
        return name, index

    def _gfx_get(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = re.match(r'\s*GET\b\s*(?=STEP\b|\()', stmt, re.IGNORECASE)
        if not m:
            return False
        c = self._gfx()
        x1, y1, pos = self._gfx_coord(stmt, m.end(), run_vars)
        dash = re.match(r'\s*-', stmt[pos:])
        if not dash:
            raise ValueError("SYNTAX ERROR: GET (x1, y1)-(x2, y2), array")
        c.last = (x1, y1)
        x2, y2, pos = self._gfx_coord(stmt, pos + dash.end(), run_vars)
        fields = self._gfx_rest(stmt, pos)
        if len(fields) != 1:
            raise ValueError("SYNTAX ERROR: GET (x1, y1)-(x2, y2), array")
        name, index = self._gfx_array_target(fields[0], run_vars)
        block = c.get_block(x1, y1, x2, y2)
        self._sprites[(name.upper(), index)] = block
        if name in self.arrays and len(self.arrays[name]) >= 2:
            h, w = block.shape
            self.arrays[name][0], self.arrays[name][1] = float(w), float(h)
        return True

    def _gfx_put(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        m = re.match(r'\s*PUT\b\s*(?=STEP\b|\()', stmt, re.IGNORECASE)
        if not m:
            return False
        c = self._gfx()
        x, y, pos = self._gfx_coord(stmt, m.end(), run_vars)
        fields = self._gfx_rest(stmt, pos)
        if not fields:
            raise ValueError("SYNTAX ERROR: PUT (x, y), array[, action]")
        name, index = self._gfx_array_target(fields[0], run_vars)
        action = fields[1].strip().upper() if len(fields) > 1 and fields[1].strip() else 'XOR'
        if action not in _PUT_ACTIONS:
            raise ValueError(f"SYNTAX ERROR: PUT action {action}")
        block = self._sprites.get((name.upper(), index))
        if block is None:
            raise ValueError(f"ILLEGAL FUNCTION CALL: {name} holds no image (use GET first)")
        c.put_block(x, y, block, action)
        c.last = (x, y)
        return True

    # ── functions ───────────────────────────────────────────────────────

    def _gfx_point(self, *args: float) -> float:
        """POINT(x, y): the colour there (-1 outside the viewport); POINT(n):
        the graphics cursor, 0/1 physical and 2/3 logical coordinates."""
        c = self._gfx()
        if len(args) == 2:
            return float(c.point(args[0], args[1]))
        if len(args) == 1:
            return float(c.cursor(int(args[0])))
        raise ValueError("POINT takes (x, y) or (n)")

    def _gfx_pmap(self, value: float, fn: float) -> float:
        """PMAP(coordinate, n): 0/1 logical to physical x/y, 2/3 back."""
        return float(self._gfx().pmap(value, int(fn)))

    @staticmethod
    def _gfx_modes() -> str:
        return ', '.join(str(m) for m in sorted(MODES))


# ---------------------------------------------------------------------------
# QUBASIC sound: PLAY's music macro language, SOUND and BEEP, gathered into a
# waveform that SAVEWAV writes as a WAV file (nothing plays through a speaker).
#
# PLAY follows QBASIC: notes A-G with # or + (sharp) and - (flat), N n (note
# number 0-84, 0 a rest), O n (octave 0-6, middle C starts octave 3), < and >,
# L n (default length 1-64), T n (tempo, quarter notes a minute, 32-255),
# P n or R n (pause), dots (each adds half the previous value), MN, ML and
# MS (normal, legato, staccato) and MF, MB (foreground or background, the same
# here). SOUND frequency, duration takes the duration in clock ticks (18.2 a
# second); frequency 32767 is silence.
# ---------------------------------------------------------------------------

TICKS_PER_SECOND = 18.2
SAMPLE_RATE = 22050
_NOTE_SEMITONE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def note_frequency(n: int) -> float:
    """Hz of note number n (1-84); N 37 is middle C, the C of octave 3."""
    return 261.6255653 * 2 ** ((n - 37) / 12)


@dataclass
class PlayState:
    """What PLAY remembers between strings, as QBASIC does."""
    tempo: int = 120
    length: int = 4
    octave: int = 4
    articulation: float = 7 / 8


def parse_play(text: str, state: PlayState) -> list[tuple[float, float]]:
    """(frequency, seconds) events for a PLAY string; frequency 0 is silence."""
    events: list[tuple[float, float]] = []
    s = text.upper().replace(' ', '')
    pos = 0

    def number() -> int | None:
        nonlocal pos
        m = re.match(r'\d+', s[pos:])
        if not m:
            return None
        pos += m.end()
        return int(m.group())

    def dots(base: float) -> float:
        nonlocal pos
        total, add = base, base
        while pos < len(s) and s[pos] == '.':
            add /= 2
            total += add
            pos += 1
        return total

    def whole() -> float:
        return 4 * 60.0 / state.tempo

    while pos < len(s):
        ch = s[pos]
        pos += 1
        if ch in _NOTE_SEMITONE:
            semitone = _NOTE_SEMITONE[ch]
            if pos < len(s) and s[pos] in '#+':
                semitone += 1
                pos += 1
            elif pos < len(s) and s[pos] == '-':
                semitone -= 1
                pos += 1
            n = number()
            if n is not None and not 1 <= n <= 64:
                raise ValueError(f"PLAY: note length {n} is not 1-64")
            seconds = dots(whole() / (n or state.length))
            note = 12 * state.octave + semitone + 1
            freq = note_frequency(note)
            events.append((freq, seconds * state.articulation))
            if state.articulation < 1:
                events.append((0.0, seconds * (1 - state.articulation)))
        elif ch == 'N':
            n = number()
            if n is None or not 0 <= n <= 84:
                raise ValueError("PLAY: N takes 0-84")
            seconds = dots(whole() / state.length)
            if n == 0:
                events.append((0.0, seconds))
            else:
                events.append((note_frequency(n), seconds * state.articulation))
                if state.articulation < 1:
                    events.append((0.0, seconds * (1 - state.articulation)))
        elif ch == 'O':
            n = number()
            if n is None or not 0 <= n <= 6:
                raise ValueError("PLAY: O takes 0-6")
            state.octave = n
        elif ch == '<':
            state.octave = max(0, state.octave - 1)
        elif ch == '>':
            state.octave = min(6, state.octave + 1)
        elif ch == 'L':
            n = number()
            if n is None or not 1 <= n <= 64:
                raise ValueError("PLAY: L takes 1-64")
            state.length = n
        elif ch == 'T':
            n = number()
            if n is None or not 32 <= n <= 255:
                raise ValueError("PLAY: T takes 32-255")
            state.tempo = n
        elif ch in 'PR':
            n = number()
            if n is None or not 1 <= n <= 64:
                raise ValueError(f"PLAY: {ch} takes 1-64")
            events.append((0.0, dots(whole() / n)))
        elif ch == 'M':
            kind = s[pos:pos + 1]
            pos += 1
            if kind == 'N':
                state.articulation = 7 / 8
            elif kind == 'L':
                state.articulation = 1.0
            elif kind == 'S':
                state.articulation = 3 / 4
            elif kind not in ('F', 'B'):
                raise ValueError(f"PLAY: unknown M{kind}")
        elif ch == ';':
            continue
        else:
            raise ValueError(f"PLAY: unknown command {ch!r}")
    return events


def synthesize(events: list[tuple[float, float]], rate: int = SAMPLE_RATE) -> np.ndarray:
    """A mono 16-bit waveform: each tone a softened square wave."""
    chunks = []
    for freq, seconds in events:
        n = max(0, int(round(seconds * rate)))
        if n == 0:
            continue
        if freq <= 0:
            chunks.append(np.zeros(n))
            continue
        t = np.arange(n) / rate
        wave_ = np.sign(np.sin(2 * np.pi * freq * t)) * 0.3
        ramp = min(n // 2, int(0.004 * rate))
        if ramp:
            env = np.ones(n)
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
            wave_ *= env
        chunks.append(wave_)
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return (np.concatenate(chunks) * 32767).astype(np.int16)


def wav_bytes(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype('<i2').tobytes())
    return buf.getvalue()


class SoundMixin(LanguageHost):
    """PLAY, SOUND and BEEP gathered for SAVEWAV."""

    def _init_sound(self) -> None:
        self._play_state = PlayState()
        self._sound_events: list[tuple[float, float]] = []

    def _sound_seconds(self) -> float:
        return sum(seconds for _, seconds in self._sound_events)

    def _stmt_play(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """PLAY string: queue its notes."""
        m = re.match(r'\s*PLAY\b\s*(.*)$', stmt, re.IGNORECASE)
        if not m or not m.group(1).strip():
            return False
        if re.match(r'(ON|OFF|STOP)\s*$', m.group(1).strip(), re.IGNORECASE):
            return True                       # PLAY ON/OFF/STOP: no music events here
        text = self._safe_eval(m.group(1), extra_ns=run_vars)
        if not isinstance(text, str):
            raise ValueError("TYPE MISMATCH: PLAY takes a string")
        self._sound_events.extend(parse_play(text, self._play_state))
        return True

    def _stmt_sound(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """SOUND frequency, duration: a tone of duration clock ticks."""
        m = re.match(r'\s*SOUND\b\s*(.*)$', stmt, re.IGNORECASE)
        if not m:
            return False
        parts = [p.strip() for p in m.group(1).split(',')]
        if len(parts) != 2 or not all(parts):
            raise ValueError("SYNTAX ERROR: SOUND frequency, duration")
        freq = float(self._eval_with_vars(parts[0], run_vars))
        ticks = float(self._eval_with_vars(parts[1], run_vars))
        if not (37 <= freq <= 32767) or not 0 <= ticks <= 65535:
            raise ValueError(f"ILLEGAL FUNCTION CALL: SOUND {freq:g}, {ticks:g}")
        self._sound_events.append((0.0 if freq >= 32767 else freq, ticks / TICKS_PER_SECOND))
        return True

    def _sound_beep(self) -> None:
        self._sound_events.append((800.0, 0.25))

    def cmd_savewav(self, rest: str) -> None:
        """SAVEWAV <file> [CLEAR] — write the sound PLAY, SOUND and BEEP made as a
        WAV file (22050 Hz, mono); CLEAR starts the next recording afresh."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: SAVEWAV <file> [CLEAR]")
            return
        try:
            path = self._sanitize_path(parts[0])
        except ValueError as e:
            self.io.writeln(f"?SAVEWAV ERROR: {e}")
            return
        checked = self._check_agent_path(path, "SAVEWAV")
        if checked is None:
            return
        path = checked if checked.lower().endswith('.wav') else checked + '.wav'
        if not self._sound_events:
            self.io.writeln("?NO SOUND — PLAY, SOUND or BEEP first")
            return
        try:
            with open(path, 'wb') as fh:
                fh.write(wav_bytes(synthesize(self._sound_events)))
        except OSError as e:
            self.io.writeln(f"?SAVEWAV ERROR: {e}")
            return
        self.io.writeln(f"SAVED {self._sound_seconds():.2f} s of sound to {path}")
        if len(parts) > 1 and parts[1].upper() == 'CLEAR':
            self._sound_events.clear()


# ---------------------------------------------------------------------------
# QUBASIC screen control — SCREEN, COLOR, CLS, LOCATE, PLAY, prompt.
# ---------------------------------------------------------------------------

class ScreenMixin(LanguageHost):
    """Screen control commands for QBasicTerminal."""

    def _init_screen(self) -> None:
        self._color_fg: str = ''
        self._color_bg: str = ''
        self._prompt: str = '] '

    _DISPLAY_NAMES = {0: 'text', 1: 'histogram', 2: 'statevector', 3: 'Bloch',
                      4: 'density', 5: 'circuit'}
    _DISPLAY_WORDS = {'TEXT': 0, 'HIST': 1, 'HISTOGRAM': 1, 'STATE': 2, 'STATEVECTOR': 2,
                      'BLOCH': 3, 'DENSITY': 4, 'CIRCUIT': 5}

    def cmd_screen(self, rest: str) -> None:
        """SCREEN [mode] — open a QBASIC graphics mode (1, 2, 7-13) or return to
        text (0); 3-5 also choose how RUN shows results, as DISPLAY does."""
        arg = rest.split(',')[0].strip()
        if not arg:
            canvas = self._canvas
            if canvas is not None:
                self.io.writeln(f"SCREEN {canvas.mode} ({canvas.width}x{canvas.height}, "
                                f"{canvas.colors} colours)")
            else:
                self.io.writeln(f"SCREEN 0 (text); DISPLAY {self._DISPLAY_NAMES[self._screen_mode]}")
            return
        try:
            mode = int(arg)
        except ValueError:
            self.io.writeln("?USAGE: SCREEN <mode>")
            return
        if mode == 0 or mode in self._GRAPHICS_MODES:
            self._gfx_screen(mode)
            if mode:
                c = self._canvas
                assert c is not None
                self.io.writeln(f"SCREEN {mode} ({c.width}x{c.height}, {c.colors} colours)")
            else:
                self.io.writeln("SCREEN 0 (text)")
        elif 3 <= mode <= 5:
            self.cmd_display(str(mode))
        else:
            self.io.writeln(f"?SCREEN {mode}: use 0, {self._gfx_modes()} (3-5 set DISPLAY)")

    _GRAPHICS_MODES = (1, 2, 7, 8, 9, 10, 11, 12, 13)

    def cmd_display(self, rest: str) -> None:
        """DISPLAY [TEXT|HIST|STATE|BLOCH|DENSITY|CIRCUIT | 0-5] — what RUN shows
        after the histogram (the statevector, Bloch spheres, density matrix or
        circuit), or nothing extra."""
        arg = rest.strip().upper()
        if not arg:
            self.io.writeln(f"DISPLAY {self._DISPLAY_NAMES[self._screen_mode]}")
            return
        mode = self._DISPLAY_WORDS.get(arg)
        if mode is None and arg.isdigit() and 0 <= int(arg) <= 5:
            mode = int(arg)
        if mode is None:
            self.io.writeln("?USAGE: DISPLAY TEXT|HIST|STATE|BLOCH|DENSITY|CIRCUIT")
            return
        self._screen_mode = mode
        self.io.writeln(f"DISPLAY {self._DISPLAY_NAMES[mode]}")


    def cmd_color(self, rest: str) -> None:
        """COLOR foreground[, background] — set terminal colors (on a graphics
        SCREEN, the drawing and background colour numbers)."""
        if self._canvas is not None:
            self._stmt_color('COLOR ' + rest, self.variables)
            return
        from qubasic_core.parser import RE_COLOR
        m = RE_COLOR.match(f"COLOR {rest}")
        if not m:
            self.io.writeln("?USAGE: COLOR <fg>[, <bg>]")
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
            self.io.write(f"\033[{fg_code};{bg_code}m")
        else:
            self.io.write(f"\033[{fg_code}m")
        self.io.writeln(f"COLOR {self._color_fg}" + (f", {self._color_bg}" if self._color_bg else ""))

    def cmd_cls(self) -> None:
        """CLS — clear the screen (on a graphics SCREEN, the picture)."""
        if self._canvas is not None:
            self._canvas.cls()
            return
        self.io.write('\033[2J\033[H')

    def cmd_locate(self, rest: str) -> None:
        """LOCATE row, col — position cursor."""
        from qubasic_core.parser import RE_LOCATE
        m = RE_LOCATE.match(f"LOCATE {rest}")
        if not m:
            self.io.writeln("?USAGE: LOCATE <row>, <col>")
            return
        row, col = int(m.group(1)), int(m.group(2))
        self.io.write(f"\033[{row};{col}H")

    def cmd_play(self, rest: str = '') -> None:
        """PLAY ["music"] — queue a PLAY string for SAVEWAV, or ring the terminal
        bell (PLAY n rings it n times)."""
        if rest.strip() and not rest.strip().isdigit():
            self._stmt_play('PLAY ' + rest, self.variables)
            self.io.writeln(f"PLAY: {self._sound_seconds():.2f} s queued (SAVEWAV <file> saves it)")
            return
        count = 1
        if rest.strip():
            try:
                count = int(rest.strip())
            except ValueError:
                pass
        for _ in range(count):
            self.io.write('\a')

    def cmd_prompt(self, rest: str) -> None:
        """PROMPT <string> — set the REPL prompt."""
        if not rest.strip():
            self.io.writeln(f"PROMPT = {self._prompt!r}")
            return
        text = rest.strip()
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        self._prompt = text
        self.io.writeln(f"PROMPT = {self._prompt!r}")


# ---------------------------------------------------------------------------
# QUBASIC file statements: OPEN (INPUT, OUTPUT, APPEND, RANDOM, BINARY),
# CLOSE, PRINT #, WRITE # (StatementMixin), INPUT #, LINE INPUT #, FIELD, LSET,
# RSET, GET #, PUT #, SEEK, LPRINT; the FREEFILE, LOF, EOF, LOC, SEEK and
# INPUT$ functions; and KILL, NAME, MKDIR, RMDIR, CHDIR and FILES.
#
# File numbers are expressions (OPEN f$ FOR INPUT AS #fn). A RANDOM file holds
# records of LEN bytes (128 by default), read and written whole: through the
# variables a FIELD statement lays over the record buffer (set with LSET and
# RSET), or as one variable (a number, a string or a TYPE record). A BINARY
# file reads and writes variables at byte positions. Numbers are stored as
# QBASIC stores them: INTEGER 2 bytes, LONG 4, SINGLE 4, DOUBLE 8, little
# endian; strings one byte per character (Latin-1).
#
# Paths are relative to the directory CHDIR sets (the working directory at
# first); in agent mode every path stays inside the working directory.
# ---------------------------------------------------------------------------

_RE_OPEN = re.compile(
    r'OPEN\s+(?P<file>.+?)\s+FOR\s+(?P<mode>INPUT|OUTPUT|APPEND|RANDOM|BINARY)\b'
    r'(?:\s+ACCESS\s+(?:READ\s+WRITE|READ|WRITE))?'
    r'(?:\s+(?:SHARED|LOCK\s+READ\s+WRITE|LOCK\s+READ|LOCK\s+WRITE))?'
    r'\s+AS\s+#?\s*(?P<num>.+?)'
    r'(?:\s+LEN\s*=\s*(?P<len>.+?))?'
    r'(?:\s+ENCODING\s+"?(?P<enc>[^"]*)"?)?\s*$', re.IGNORECASE)
# The GW-BASIC form: OPEN mode$, [#]n, file$[, reclen]
_RE_OPEN_OLD = re.compile(r'OPEN\s+(?P<mode>"[IOARB]"|\w+\$?)\s*,\s*#?\s*(?P<rest>.+)$', re.IGNORECASE)
_OLD_MODES = {'I': 'INPUT', 'O': 'OUTPUT', 'A': 'APPEND', 'R': 'RANDOM', 'B': 'BINARY'}
_NUM_FORMATS = {'INTEGER': '<h', 'LONG': '<i', 'SINGLE': '<f', 'DOUBLE': '<d'}


@dataclass
class RecordFile:
    """A RANDOM or BINARY file's state beyond the open handle."""
    mode: str                         # 'RANDOM' or 'BINARY'
    reclen: int = 128
    fields: list[tuple[str, int]] = field(default_factory=list)   # (variable, width)
    last: int = 0                     # LOC: last record (RANDOM) or byte (BINARY)
    next: int = 1                     # SEEK: next record or byte position
    past_end: bool = False            # the last GET read beyond the end


def _latin(data: bytes) -> str:
    return data.decode('latin-1')


def _bytes(text: str) -> bytes:
    return text.encode('latin-1', errors='replace')


class FileIOMixin(LanguageHost):
    """Numbered file handles, as QBASIC opens them."""

    def _init_file_handles(self) -> None:
        self._file_handles: dict[int, Any] = {}
        self._record_files: dict[int, RecordFile] = {}
        # Values read from a line by INPUT # and not yet assigned, per handle.
        self._file_fields: dict[int, list[str]] = {}
        self._lprint_path: str | None = None
        self._imported_lines: set[int] = set()
        self._file_dir = ''                       # CHDIR: relative to the working directory
        self._file_cols: dict[int, int] = {}      # PRINT # cursor column, per handle
        self._print_col = 0                       # screen cursor column (POS)

    # ── paths and numbers ───────────────────────────────────────────────

    def _basic_path(self, name: str, cmd: str) -> str:
        """A file name from a BASIC statement, resolved against CHDIR's
        directory; it may not leave the working directory."""
        if not name or not name.strip():
            raise RuntimeError("BAD FILE NAME: empty")
        if any(ord(c) < 32 for c in name):
            raise RuntimeError("BAD FILE NAME: control characters")
        name = name.strip()
        if os.path.isabs(name) or re.match(r'^[A-Za-z]:', name) or name.startswith(('//', '\\')):
            raise RuntimeError(f"PERMISSION DENIED: {name} is an absolute path")
        path = os.path.normpath(os.path.join(self._file_dir or '.', name))
        if path == '..' or path.startswith('..' + os.sep):
            raise RuntimeError(f"PERMISSION DENIED: {name} is outside the working directory")
        if self.agent_mode:
            checked = self._check_agent_path(path, cmd)
            if checked is None:
                raise RuntimeError(f"PERMISSION DENIED: {name} is outside the working directory")
            path = checked
        return path

    def _file_name_value(self, text: str, run_vars: MutableMapping[str, Any]) -> str:
        """A file-name expression; a bare unquoted name stands for itself."""
        text = text.strip()
        try:
            value = self._safe_eval(text, extra_ns=run_vars)
        except Exception:
            return text
        return value if isinstance(value, str) else text

    def _file_number(self, text: str, run_vars: MutableMapping[str, Any]) -> int:
        n = int(round(float(self._safe_eval(text.strip().lstrip('#').strip(), extra_ns=run_vars))))
        if not 1 <= n <= 255:
            raise RuntimeError(f"BAD FILE NAME OR NUMBER: #{n}")
        return n

    # ── OPEN and CLOSE ──────────────────────────────────────────────────

    def _parse_open(self, stmt: str, run_vars: MutableMapping[str, Any]) -> tuple[str, str, int, int, str]:
        """(file name, mode, number, record length, encoding) of an OPEN."""
        text = stmt.strip()
        m = _RE_OPEN.match(text)
        if m:
            name = self._file_name_value(m.group('file'), run_vars)
            mode = m.group('mode').upper()
            num = self._file_number(m.group('num'), run_vars)
            reclen = int(round(float(self._safe_eval(m.group('len'), extra_ns=run_vars)))) if m.group('len') else 128
            enc = (m.group('enc') or 'utf-8').strip() or 'utf-8'
            return name, mode, num, reclen, enc
        m = _RE_OPEN_OLD.match(text)
        if m:
            letter = str(self._safe_eval(m.group('mode'), extra_ns=run_vars)).strip().upper()[:1]
            parts = _top_split(m.group('rest'))
            if letter not in _OLD_MODES or len(parts) not in (2, 3):
                raise ValueError('SYNTAX ERROR: OPEN mode$, #n, file$[, reclen]')
            num = self._file_number(parts[0], run_vars)
            name = self._file_name_value(parts[1], run_vars)
            reclen = int(round(float(self._safe_eval(parts[2], extra_ns=run_vars)))) if len(parts) == 3 else 128
            return name, _OLD_MODES[letter], num, reclen, 'utf-8'
        raise ValueError('SYNTAX ERROR: OPEN "file" FOR INPUT|OUTPUT|APPEND|RANDOM|BINARY AS #n [LEN = n]')

    def _open_file(self, rest: str, run_vars: MutableMapping[str, Any] | None = None) -> tuple[int, str, str]:
        """Open `file FOR mode AS #n ...` (the text after OPEN); returns
        (number, path, mode) or raises with the QBASIC error."""
        rv = self.variables if run_vars is None else run_vars
        name, mode, num, reclen, enc = self._parse_open('OPEN ' + rest, rv)
        if num in self._file_handles:
            raise RuntimeError(f"FILE ALREADY OPEN: #{num}")
        path = self._basic_path(name, 'OPEN')
        if mode == 'INPUT' and not os.path.isfile(path):
            raise RuntimeError(f"FILE NOT FOUND: {name}")
        if mode in ('RANDOM', 'BINARY'):
            if not 1 <= reclen <= 32767:
                raise RuntimeError(f"BAD RECORD LENGTH: {reclen}")
            if not os.path.isfile(path):
                open(path, 'wb').close()
            self._file_handles[num] = open(path, 'r+b')
            self._record_files[num] = RecordFile(mode, reclen)
        elif enc.lower() == 'binary':
            self._file_handles[num] = open(path, {'INPUT': 'rb', 'OUTPUT': 'wb', 'APPEND': 'ab'}[mode])
        else:
            self._file_handles[num] = open(path, {'INPUT': 'r', 'OUTPUT': 'w', 'APPEND': 'a'}[mode],
                                           encoding=enc)
        return num, path, mode

    def cmd_open(self, rest: str) -> None:
        """OPEN file FOR INPUT|OUTPUT|APPEND|RANDOM|BINARY AS #n [LEN = n] — open a file."""
        try:
            handle, path, mode_str = self._open_file(rest)
        except Exception as e:
            self.io.writeln(f"?OPEN ERROR: {e}")
            return
        self.io.writeln(f"OPENED #{handle} ({path}, {mode_str})")

    def _close_file(self, handle: int) -> bool:
        f = self._file_handles.pop(handle, None)
        self._file_fields.pop(handle, None)
        self._file_cols.pop(handle, None)
        self._record_files.pop(handle, None)
        if f is None:
            return False
        f.close()
        return True

    def _close_list(self, text: str, run_vars: MutableMapping[str, Any]) -> list[int]:
        if not text.strip():
            closed = list(self._file_handles)
            for h in closed:
                self._close_file(h)
            return closed
        closed = []
        for tok in _top_split(text):
            if tok.strip():
                h = self._file_number(tok, run_vars)
                self._close_file(h)
                closed.append(h)
        return closed

    def cmd_close(self, rest: str) -> None:
        """CLOSE [#n[, #m ...]] — close files (every file with no number)."""
        try:
            closed = self._close_list(rest, self.variables)
        except Exception as e:
            self.io.writeln(f"?CLOSE ERROR: {e}")
            return
        self.io.writeln("CLOSED ALL" if not rest.strip() else
                        "CLOSED " + ', '.join(f"#{h}" for h in closed))

    def _stmt_open_close(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """OPEN and CLOSE in a program: silent, failures raised for ON ERROR."""
        word = stmt.strip().split(None, 1)
        if not word:
            return False
        if word[0].upper() == 'OPEN':
            if getattr(self.io, 'replay', False):
                _, mode, _, _, _ = self._parse_open(stmt, run_vars)
                if mode != 'INPUT':
                    return True     # later live shots leave the first shot's file alone
            self._open_file(stmt.strip()[4:].strip(), run_vars)
            return True
        if word[0].upper() == 'CLOSE':
            self._close_list(word[1] if len(word) > 1 else '', run_vars)
            return True
        return False

    # ── sequential files ────────────────────────────────────────────────

    def _writable(self, handle: int) -> IO[Any]:
        """The open file for handle; BAD FILE NAME OR NUMBER (52) when it is
        not open and BAD FILE MODE (54) when it cannot be written."""
        if handle not in self._file_handles:
            raise RuntimeError(f"BAD FILE NAME OR NUMBER: #{handle} is not open")
        if handle in self._record_files:
            raise RuntimeError(f"BAD FILE MODE: #{handle} is a {self._record_files[handle].mode} file")
        f = self._file_handles[handle]
        fmode = getattr(f, 'mode', '')
        if fmode.startswith('r') and '+' not in fmode:
            raise RuntimeError(f"BAD FILE MODE: #{handle} is open for input")
        return f

    def _write_to_handle(self, handle: int, text: str, newline: bool = True) -> bool:
        if getattr(self.io, 'replay', False):
            return True     # later live shots write nothing; the first shot did
        f = self._writable(handle)
        data = text + ('\n' if newline else '')
        f.write(_bytes(data) if 'b' in getattr(f, 'mode', '') else data)
        f.flush()
        return True

    def _split_handle(self, text: str) -> tuple[str, str] | None:
        """'#n, rest' -> ('n', 'rest'): the file number before the first
        top-level comma."""
        parts = _top_split(text, ',')
        if len(parts) < 2:
            return None
        return parts[0].strip(), ','.join(parts[1:])

    def _exec_print_file(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """PRINT #n, list — what PRINT would show, written to the file
        (PRINT #n, USING fmt$; values formats as PRINT USING)."""
        m = re.match(r'\s*PRINT\s*#\s*(.+)$', stmt, re.IGNORECASE)
        if not m:
            return False
        split = self._split_handle(m.group(1))
        if split is None:
            raise ValueError("SYNTAX ERROR: PRINT #n, list")
        handle = self._file_number(split[0], run_vars)
        data = split[1].strip()
        um = re.match(r'USING\s+(.+)$', data, re.IGNORECASE)
        col = self._file_cols.get(handle, 0)
        if um:
            head, *tail = _top_split(um.group(1), ';')
            text, newline = self._using_text(head.strip(), ';'.join(tail), run_vars)
        else:
            text, newline = self._format_print(data, run_vars, start=col)
        self._file_cols[handle] = 0 if newline else next_column(col, text)
        return self._write_to_handle(handle, text, newline)

    def _readable(self, handle: int) -> IO[str]:
        if handle not in self._file_handles:
            raise RuntimeError(f"BAD FILE NAME OR NUMBER: #{handle} is not open")
        f = self._file_handles[handle]
        fmode = getattr(f, 'mode', '')
        if fmode.startswith(('w', 'a')) and '+' not in fmode:
            raise RuntimeError(f"BAD FILE MODE: #{handle} is not open for input")
        return f

    def _read_text_line(self, handle: int) -> str:
        f = self._readable(handle)
        line = f.readline()
        if isinstance(line, bytes):
            line = _latin(line)
        return line

    def _exec_input_file(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """INPUT #n, var[, var ...] — read comma- or line-separated values;
        reading past the end of the file is an error, as in QBASIC (test
        EOF(n) first)."""
        m = re.match(r'\s*INPUT\s*#\s*(.+)$', stmt, re.IGNORECASE)
        if not m:
            return False
        split = self._split_handle(m.group(1))
        if split is None:
            raise ValueError("SYNTAX ERROR: INPUT #n, variable")
        handle = self._file_number(split[0], run_vars)
        pending = self._file_fields.setdefault(handle, [])
        from qubasic_core.interpreter import _input_fields, _parse_input_number
        for target in self._split_arg_list(split[1]):
            lv = self._lvalue(target, run_vars)
            while not pending:
                line = self._read_text_line(handle)
                if not line:
                    raise RuntimeError("INPUT PAST END OF FILE")
                pending.extend(_input_fields(line.rstrip('\r\n')))
            fld = pending.pop(0)
            if is_string_kind(self._lvalue_kind(lv)):
                value: Any = fld
            else:
                value = _parse_input_number(fld.strip()) if fld.strip() else 0
                if value is None:
                    from qubasic_core.expression import _val_fn
                    value = _val_fn(fld)
            self._write_lvalue(lv, value, run_vars)
        return True

    def _exec_line_input_file(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """LINE INPUT #n, var$ — the next whole line of the file."""
        m = re.match(r'\s*LINE\s+INPUT\s*#\s*(.+)$', stmt, re.IGNORECASE)
        if not m:
            return False
        split = self._split_handle(m.group(1))
        if split is None:
            raise ValueError("SYNTAX ERROR: LINE INPUT #n, variable$")
        handle = self._file_number(split[0], run_vars)
        self._file_fields.pop(handle, None)
        line = self._read_text_line(handle)
        if not line:
            raise RuntimeError("INPUT PAST END OF FILE")
        self._write_lvalue(self._lvalue(split[1], run_vars), line.rstrip('\r\n'), run_vars)
        return True

    def _input_chars(self, n: Any, handle: Any = None) -> str:
        """INPUT$(n[, #f]): n characters from the file, or typed at the keyboard."""
        count = int(n)
        if count < 0:
            raise ValueError("ILLEGAL FUNCTION CALL: INPUT$ count")
        if handle is None:
            text = ''
            while len(text) < count:
                try:
                    text += self.io.read_line('')
                except EOFError:
                    break
            return text[:count]
        h = int(handle)
        f = self._readable(h)
        data = f.read(count)
        text = _latin(data) if isinstance(data, bytes) else data
        if len(text) < count:
            raise RuntimeError("INPUT PAST END OF FILE")
        return text

    # ── RANDOM and BINARY files ─────────────────────────────────────────

    def _record_file(self, handle: int) -> tuple[Any, RecordFile]:
        if handle not in self._file_handles:
            raise RuntimeError(f"BAD FILE NAME OR NUMBER: #{handle} is not open")
        info = self._record_files.get(handle)
        if info is None:
            raise RuntimeError(f"BAD FILE MODE: #{handle} is not a RANDOM or BINARY file")
        return self._file_handles[handle], info

    def _stmt_field(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """FIELD #n, width AS var$[, width AS var$ ...] — lay string variables
        over the record buffer."""
        m = re.match(r'\s*FIELD\s*#?\s*(.+)$', stmt, re.IGNORECASE)
        if not m:
            return False
        split = self._split_handle(m.group(1))
        if split is None:
            raise ValueError("SYNTAX ERROR: FIELD #n, width AS var$")
        handle = self._file_number(split[0], run_vars)
        _, info = self._record_file(handle)
        fields: list[tuple[str, int]] = []
        total = 0
        for part in _top_split(split[1]):
            fm = re.match(r'\s*(.+?)\s+AS\s+([A-Za-z_]\w*\$)\s*$', part, re.IGNORECASE)
            if not fm:
                raise ValueError(f"SYNTAX ERROR: FIELD needs width AS name$, got {part.strip()!r}")
            width = int(round(float(self._safe_eval(fm.group(1), extra_ns=run_vars))))
            total += width
            if width < 0 or total > info.reclen:
                raise RuntimeError(f"FIELD OVERFLOW: {total} bytes in a {info.reclen}-byte record")
            fields.append((fm.group(2), width))
            run_vars[fm.group(2)] = ' ' * width
            self.variables[fm.group(2)] = ' ' * width
        info.fields = fields
        return True

    def _field_width(self, name: str) -> int | None:
        for info in self._record_files.values():
            for var, width in info.fields:
                if var.upper() == name.upper():
                    return width
        return None

    def _stmt_lset_rset(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """LSET var$ = value (left-justified) and RSET var$ = value (right):
        the value fills the variable's width (its FIELD width, or its length)."""
        m = re.match(r'\s*(LSET|RSET)\s+(.+?)\s*=\s*(.+)$', stmt, re.IGNORECASE)
        if not m:
            return False
        lv = self._lvalue(m.group(2), run_vars)
        value = self._safe_eval(m.group(3), extra_ns=run_vars)
        if isinstance(value, Record):
            self._write_lvalue(lv, value, run_vars)
            return True
        if not isinstance(value, str):
            raise ValueError("TYPE MISMATCH: LSET and RSET take strings")
        width = self._field_width(lv.name) if not lv.subs and not lv.field else None
        if width is None:
            current = self._read_lvalue(lv, run_vars)
            width = len(current) if isinstance(current, str) else len(value)
        text = value[:width]
        text = text.ljust(width) if m.group(1).upper() == 'LSET' else text.rjust(width)
        self._write_lvalue(lv, text, run_vars)
        return True

    def _kind_size(self, kind: str, value: Any) -> int:
        if kind in _NUM_FORMATS:
            return struct.calcsize(_NUM_FORMATS[kind])
        n = fixed_length(kind)
        if n:
            return n
        if kind == 'STRING':
            return len(value) if isinstance(value, str) else 0
        fields = self._type_fields(kind)
        return sum(self._kind_size(k, None if not isinstance(value, Record) else value.get(f))
                   for f, k in fields)

    def _pack(self, kind: str, value: Any) -> bytes:
        if kind in _NUM_FORMATS:
            fmt = _NUM_FORMATS[kind]
            v = int(round(value)) if fmt in ('<h', '<i') else float(value)
            try:
                return struct.pack(fmt, v)
            except struct.error:
                raise RuntimeError(f"OVERFLOW: {value} does not fit a {kind}") from None
        n = fixed_length(kind)
        if n:
            return _bytes(str(value)[:n].ljust(n))
        if kind == 'STRING':
            return _bytes(str(value))
        fields = self._type_fields(kind)
        if not isinstance(value, Record):
            raise ValueError(f"TYPE MISMATCH: {kind}")
        return b''.join(self._pack(k, value[f]) for f, k in fields)

    def _unpack(self, kind: str, data: bytes, template: Any) -> Any:
        if kind in _NUM_FORMATS:
            v = struct.unpack(_NUM_FORMATS[kind], data)[0]
            return float(v) if kind in ('SINGLE', 'DOUBLE') else int(v)
        if is_string_kind(kind):
            return _latin(data)
        fields = self._type_fields(kind)
        rec = self._new_record(kind)
        pos = 0
        for f, k in fields:
            size = self._kind_size(k, rec[f])
            rec[f] = self._unpack(k, data[pos:pos + size], rec[f])
            pos += size
        return rec

    def _record_args(self, text: str, run_vars: MutableMapping[str, Any]) -> tuple[int, int | None, str | None]:
        """'#n[, [record][, variable]]' -> (number, record or None, variable or None)."""
        parts = _top_split(text)
        handle = self._file_number(parts[0], run_vars)
        rec = None
        if len(parts) > 1 and parts[1].strip():
            rec = int(round(float(self._safe_eval(parts[1], extra_ns=run_vars))))
        var = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        if len(parts) > 3:
            raise ValueError("SYNTAX ERROR: GET/PUT #n[, record][, variable]")
        return handle, rec, var

    def _stmt_get_put_file(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """GET #n[, record][, variable] and PUT #n[, record][, variable]."""
        m = re.match(r'\s*(GET|PUT)\s*(?=#|\d)(.*)$', stmt, re.IGNORECASE)
        if not m:
            return False
        is_get = m.group(1).upper() == 'GET'
        handle, rec, var = self._record_args(m.group(2), run_vars)
        f, info = self._record_file(handle)
        position = info.next if rec is None else rec
        if position < 1:
            raise RuntimeError(f"BAD RECORD NUMBER: {position}")
        if info.mode == 'RANDOM':
            offset = (position - 1) * info.reclen
        else:
            offset = position - 1
        if var is None:
            if info.mode != 'RANDOM' or not info.fields:
                raise ValueError("SYNTAX ERROR: GET/PUT needs a variable (no FIELD buffer)")
            if is_get:
                f.seek(offset)
                data = f.read(info.reclen)
                info.past_end = len(data) < info.reclen
                data = data.ljust(info.reclen, b'\x00')
                pos = 0
                for name, width in info.fields:
                    text = _latin(data[pos:pos + width])
                    run_vars[name] = text
                    self.variables[name] = text
                    pos += width
            else:
                buf = b''.join(_bytes(str(run_vars.get(name, '')))[:width].ljust(width)
                               for name, width in info.fields)
                f.seek(offset)
                f.write(buf.ljust(info.reclen, b' '))
                f.flush()
            info.last = position
            info.next = position + 1
            return True
        lv = self._lvalue(var, run_vars)
        kind = self._lvalue_kind(lv)
        current = self._read_lvalue(lv, run_vars)
        size = info.reclen if info.mode == 'RANDOM' and kind == 'STRING' else self._kind_size(kind, current)
        if info.mode == 'RANDOM' and size > info.reclen:
            raise RuntimeError(f"FIELD OVERFLOW: a {size}-byte variable in a {info.reclen}-byte record")
        f.seek(offset)
        if is_get:
            data = f.read(size)
            info.past_end = len(data) < size
            data = data.ljust(size, b'\x00')
            self._write_lvalue(lv, self._unpack(kind, data, current), run_vars)
        else:
            data = self._pack(kind, current)
            if info.mode == 'RANDOM':
                data = data[:info.reclen].ljust(info.reclen, b'\x00')
            f.write(data)
            f.flush()
        if info.mode == 'RANDOM':
            info.last, info.next = position, position + 1
        else:
            info.last, info.next = position + size - 1, position + size
        return True

    def _stmt_seek(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """SEEK #n, position — the next record (RANDOM) or byte (others) read or written."""
        m = re.match(r'\s*SEEK\s*#?\s*(.+)$', stmt, re.IGNORECASE)
        if not m:
            return False
        parts = _top_split(m.group(1))
        if len(parts) != 2:
            raise ValueError("SYNTAX ERROR: SEEK #n, position")
        handle = self._file_number(parts[0], run_vars)
        pos = int(round(float(self._safe_eval(parts[1], extra_ns=run_vars))))
        if pos < 1:
            raise RuntimeError(f"BAD RECORD NUMBER: {pos}")
        if handle in self._record_files:
            self._record_files[handle].next = pos
        else:
            self._readable(handle).seek(pos - 1)
            self._file_fields.pop(handle, None)
        return True

    def _seek_fn(self, handle: Any) -> int:
        """SEEK(n): the next record (RANDOM) or byte position."""
        h = int(handle)
        if h in self._record_files:
            return self._record_files[h].next
        if h not in self._file_handles:
            raise RuntimeError(f"BAD FILE NAME OR NUMBER: #{h} is not open")
        return int(self._file_handles[h].tell()) + 1

    def _loc_fn(self, handle: Any) -> int:
        """LOC(n): the last record (RANDOM) or byte (BINARY) read or written;
        for sequential files the position in 128-byte blocks."""
        h = int(handle)
        if h in self._record_files:
            return self._record_files[h].last
        if h not in self._file_handles:
            raise RuntimeError(f"BAD FILE NAME OR NUMBER: #{h} is not open")
        return int(self._file_handles[h].tell()) // 128

    def _freefile(self) -> int:
        """FREEFILE: the lowest file number not in use."""
        n = 1
        while n in self._file_handles:
            n += 1
        return n

    def _lof(self, handle: Any) -> int:
        """LOF(n): length of the open file in bytes."""
        h = int(handle)
        if h not in self._file_handles:
            raise RuntimeError(f"BAD FILE NAME OR NUMBER: #{h} is not open")
        f = self._file_handles[h]
        try:
            f.flush()
        except Exception:
            pass
        return os.fstat(f.fileno()).st_size

    def _eof(self, handle: float) -> int:
        """EOF(n) — -1 (true) at the end of the file (for RANDOM, after a GET
        past the last record), else 0."""
        h = int(handle)
        if h not in self._file_handles:
            return -1
        info = self._record_files.get(h)
        f = self._file_handles[h]
        if info is not None:
            if info.mode == 'RANDOM':
                return -1 if info.past_end else 0
            return -1 if info.next - 1 >= self._lof(h) else 0
        if self._file_fields.get(h):
            return 0
        pos = f.tell()
        try:
            ch = f.read(1)
            if not ch:
                return -1
            f.seek(pos)
        except Exception:
            return -1
        return 0

    # ── LPRINT ──────────────────────────────────────────────────────────

    def _exec_lprint(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """Handle LPRINT data — output to log file or stderr."""
        m = RE_LPRINT.match(stmt)
        if not m:
            return False
        if getattr(self.io, 'replay', False):
            return True
        data = m.group(1).strip()
        if (data.startswith('"') and data.endswith('"')):
            text = data[1:-1]
        else:
            try:
                text = str(self._eval_with_vars(data, run_vars) if run_vars else self.eval_expr(data))
            except Exception:
                text = data
        if self._lprint_path:
            with open(self._lprint_path, 'a', encoding='utf-8') as f:
                f.write(text + '\n')
        else:
            self.io.writeln(text)
        return True

    # ── the file system ─────────────────────────────────────────────────

    def _stmt_filesystem(self, stmt: str, run_vars: MutableMapping[str, Any]) -> bool:
        """KILL spec, NAME old AS new, MKDIR, RMDIR, CHDIR path and FILES [spec]."""
        m = re.match(r'\s*(KILL|MKDIR|RMDIR|CHDIR|FILES|NAME)\b\s*(.*)$', stmt, re.IGNORECASE)
        if not m:
            return False
        word, arg = m.group(1).upper(), m.group(2).strip()
        if word == 'NAME':
            nm = re.match(r'(.+?)\s+AS\s+(.+)$', arg, re.IGNORECASE)
            if not nm:
                raise ValueError("SYNTAX ERROR: NAME old AS new")
            old = self._basic_path(self._file_name_value(nm.group(1), run_vars), 'NAME')
            new = self._basic_path(self._file_name_value(nm.group(2), run_vars), 'NAME')
            if not os.path.exists(old):
                raise RuntimeError(f"FILE NOT FOUND: {nm.group(1).strip()}")
            if os.path.exists(new):
                raise RuntimeError(f"FILE ALREADY EXISTS: {nm.group(2).strip()}")
            os.rename(old, new)
            return True
        if word == 'FILES':
            spec = self._file_name_value(arg, run_vars) if arg else '*'
            self._files_listing(spec)
            return True
        if not arg:
            raise ValueError(f"SYNTAX ERROR: {word} path")
        name = self._file_name_value(arg, run_vars)
        if word == 'KILL':
            folder, pattern = os.path.split(name)
            base = self._basic_path(folder or '.', 'KILL')
            if any(ch in pattern for ch in '*?'):
                matches = [os.path.join(base, f) for f in sorted(os.listdir(base))
                           if fnmatch.fnmatch(f.upper(), pattern.upper())
                           and os.path.isfile(os.path.join(base, f))]
            else:
                target = os.path.join(base, pattern)
                matches = [target] if os.path.isfile(target) else []
            if not matches:
                raise RuntimeError(f"FILE NOT FOUND: {name}")
            open_paths = {os.path.realpath(getattr(f, 'name', '')) for f in self._file_handles.values()}
            for path in matches:
                if os.path.realpath(path) in open_paths:
                    raise RuntimeError(f"FILE ALREADY OPEN: {os.path.basename(path)}")
            for path in matches:
                os.remove(path)
            return True
        path = self._basic_path(name, word)
        try:
            if word == 'MKDIR':
                os.mkdir(path)
            elif word == 'RMDIR':
                os.rmdir(path)
            else:
                if not os.path.isdir(path):
                    raise RuntimeError(f"PATH NOT FOUND: {name}")
                self._file_dir = '' if path == '.' else path
        except FileNotFoundError:
            raise RuntimeError(f"PATH NOT FOUND: {name}") from None
        except FileExistsError:
            raise RuntimeError(f"PATH/FILE ACCESS ERROR: {name} exists") from None
        except OSError as e:
            raise RuntimeError(f"PATH/FILE ACCESS ERROR: {e.strerror or e}") from None
        return True

    def _files_listing(self, spec: str) -> None:
        folder, pattern = os.path.split(spec)
        base = self._basic_path(folder or '.', 'FILES')
        if not os.path.isdir(base):
            raise RuntimeError(f"PATH NOT FOUND: {folder}")
        names = [f + ('\\' if os.path.isdir(os.path.join(base, f)) else '')
                 for f in sorted(os.listdir(base)) if fnmatch.fnmatch(f.upper(), (pattern or '*').upper())]
        if not names:
            raise RuntimeError(f"FILE NOT FOUND: {spec}")
        width = max(len(n) for n in names) + 2
        per_row = max(1, 78 // width)
        for k in range(0, len(names), per_row):
            self.io.writeln(''.join(n.ljust(width) for n in names[k:k + per_row]).rstrip())

    def cmd_files(self, rest: str = '') -> None:
        """FILES [spec] — list the files matching spec (* and ? wildcards)."""
        try:
            self._files_listing(self._file_name_value(rest, self.variables) if rest.strip() else '*')
        except Exception as e:
            self.io.writeln(f"?{e}")
