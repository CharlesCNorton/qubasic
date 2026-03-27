"""QBASIC memory map — PEEK, POKE, SYS, USR, WAIT, DUMP, MAP, MONITOR, CATALOG."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Memory Map
# ═══════════════════════════════════════════════════════════════════════
#   $0000-$003F  Zero Page (64 slots, SYS parameter passing)
#   $0100-$01FF  Qubit State (8 addresses per qubit, 32 qubits max)
#                Per qubit N at $0100 + N*8:
#                  +0 P(|1⟩)   +1 Bloch X  +2 Bloch Y  +3 Bloch Z
#                  +4 Re(α)    +5 Im(α)    +6 Re(β)    +7 Im(β)
#   $D000-$D007  QPU Config (read/write)
#   $D010-$D014  QPU Status (read-only)
#   $E000-$E0B0  SYS Routine Table (built-in algorithms)
#   $F000-$FFFF  User SYS Routines
# ═══════════════════════════════════════════════════════════════════════

QUBIT_BLOCK = 8

CFG_NAMES = {
    0xD000: 'num_qubits', 0xD001: 'shots', 0xD002: 'sim_method',
    0xD003: 'sim_device', 0xD004: 'noise_type', 0xD005: 'noise_param',
    0xD006: 'max_iterations', 0xD007: 'screen_mode',
    # Performance tuning (Aer backend options)
    0xD008: 'fusion_enable',        # 0/1
    0xD009: 'mps_truncation',       # float threshold
    0xD00A: 'sv_parallel_threshold', # int
    0xD00B: 'es_approx_error',      # float (extended stabilizer)
}

STS_NAMES = {
    0xD010: 'gate_count', 0xD011: 'circuit_depth', 0xD012: 'run_time_ms',
    0xD013: 'total_probability', 0xD014: 'entanglement_entropy',
}

SIM_METHODS_FWD = {0: 'automatic', 1: 'statevector', 2: 'stabilizer',
                   3: 'matrix_product_state', 4: 'density_matrix'}
SIM_METHODS_REV = {v: k for k, v in SIM_METHODS_FWD.items()}
SIM_DEVICES_FWD = {0: 'CPU', 1: 'GPU'}
SIM_DEVICES_REV = {v: k for k, v in SIM_DEVICES_FWD.items()}

SYS_ROUTINES = {
    0xE000: 'BELL',      0xE010: 'GHZ',       0xE020: 'QFT',
    0xE030: 'GROVER',    0xE040: 'TELEPORT',   0xE050: 'DEUTSCH',
    0xE060: 'BERNSTEIN', 0xE070: 'SUPERDENSE', 0xE080: 'RANDOM',
    0xE090: 'STRESS',    0xE0A0: 'LOCC-TELEPORT', 0xE0B0: 'LOCC-COORD',
}


class MemoryMixin:
    """Memory-mapped PEEK/POKE, SYS, USR, WAIT, DUMP, MAP, MONITOR, CATALOG.

    Requires: TerminalProtocol — uses self.num_qubits, self.shots,
    self.sim_method, self.sim_device, self._noise_model, self._max_iterations,
    self.last_sv, self.variables, self.cmd_demo(), self.eval_expr().
    """

    def _init_memory(self) -> None:
        self._zero_page: list[float] = [0.0] * 64
        self._status: dict[int, float] = {a: 0.0 for a in STS_NAMES}
        self._status[0xD013] = 1.0
        self._user_sys: dict[int, str] = {}
        self._screen_mode: int = 0
        # Per-qubit noise: $D100 + qubit*2 = noise_type, +1 = noise_param
        # noise_type: 0=none, 1=depolarizing, 2=amplitude_damping, 3=phase_flip
        self._qubit_noise: dict[int, tuple[int, float]] = {}  # qubit -> (type, param)

    def _update_status(self, **kw: float) -> None:
        for addr, name in STS_NAMES.items():
            if name in kw:
                self._status[addr] = float(kw[name])
        if self.last_sv is not None:
            sv = np.ascontiguousarray(self.last_sv).ravel()
            self._status[0xD013] = float(np.sum(np.abs(sv) ** 2))

    # ── PEEK ───────────────────────────────────────────────────────────

    def _peek(self, addr: float) -> float:
        a = int(addr)
        if 0x0000 <= a <= 0x003F:
            return self._zero_page[a]
        if 0x0100 <= a <= 0x01FF:
            return self._peek_qubit(a)
        if 0xD100 <= a <= 0xD1FF:
            qubit = (a - 0xD100) // 2
            field = (a - 0xD100) % 2
            ntype, nparam = self._qubit_noise.get(qubit, (0, 0.0))
            return float(ntype) if field == 0 else nparam
        if a in CFG_NAMES:
            return self._peek_config(a)
        if a in STS_NAMES:
            return self._status.get(a, 0.0)
        return 0.0

    def _peek_qubit(self, addr: int) -> float:
        off = addr - 0x0100
        qubit, field = off // QUBIT_BLOCK, off % QUBIT_BLOCK
        if self.last_sv is None or qubit >= self.num_qubits:
            return 0.0
        sv = np.ascontiguousarray(self.last_sv).ravel().reshape([2] * self.num_qubits)
        ax = self.num_qubits - 1 - qubit
        t = np.moveaxis(sv, ax, 0)
        t0, t1 = t[0].flatten(), t[1].flatten()
        rho_00 = float(np.sum(np.abs(t0) ** 2))
        rho_11 = float(np.sum(np.abs(t1) ** 2))
        rho_01 = complex(np.sum(np.conj(t0) * t1))
        return [rho_11,
                float(2 * rho_01.real),
                float(-2 * rho_01.imag),
                float(rho_00 - rho_11),
                float(np.sqrt(max(0, rho_00))),
                0.0,
                float(np.sqrt(max(0, rho_11))),
                0.0][field] if field < 8 else 0.0

    def _peek_config(self, addr: int) -> float:
        n = CFG_NAMES[addr]
        if n == 'num_qubits':     return float(self.num_qubits)
        if n == 'shots':          return float(self.shots)
        if n == 'sim_method':     return float(SIM_METHODS_REV.get(self.sim_method, 0))
        if n == 'sim_device':     return float(SIM_DEVICES_REV.get(self.sim_device, 0))
        if n == 'noise_type':     return 0.0 if self._noise_model is None else 1.0
        if n == 'noise_param':    return 0.0
        if n == 'max_iterations': return float(self._max_iterations)
        if n == 'screen_mode':    return float(self._screen_mode)
        if n == 'fusion_enable':  return float(getattr(self, '_fusion_enable', 1))
        if n == 'mps_truncation': return float(getattr(self, '_mps_truncation', 1e-16))
        if n == 'sv_parallel_threshold': return float(getattr(self, '_sv_parallel_threshold', 14))
        if n == 'es_approx_error': return float(getattr(self, '_es_approx_error', 0.05))
        return 0.0

    # ── POKE ───────────────────────────────────────────────────────────

    def _poke(self, addr: float, value: float) -> None:
        a, v = int(addr), float(value)
        if 0x0000 <= a <= 0x003F:
            self._zero_page[a] = v
            return
        writers = {
            0xD000: lambda: setattr(self, 'num_qubits', max(1, min(32, int(v)))),
            0xD001: lambda: setattr(self, 'shots', max(1, int(v))),
            0xD002: lambda: setattr(self, 'sim_method', SIM_METHODS_FWD.get(int(v), 'automatic')),
            0xD003: lambda: setattr(self, 'sim_device', SIM_DEVICES_FWD.get(int(v), 'CPU')),
            0xD006: lambda: setattr(self, '_max_iterations', max(1, int(v))),
            0xD007: lambda: setattr(self, '_screen_mode', int(v)),
            0xD008: lambda: setattr(self, '_fusion_enable', bool(int(v))),
            0xD009: lambda: setattr(self, '_mps_truncation', v),
            0xD00A: lambda: setattr(self, '_sv_parallel_threshold', int(v)),
            0xD00B: lambda: setattr(self, '_es_approx_error', v),
        }
        if a in writers:
            writers[a]()
        elif 0xD100 <= a <= 0xD1FF:
            qubit = (a - 0xD100) // 2
            field = (a - 0xD100) % 2
            ntype, nparam = self._qubit_noise.get(qubit, (0, 0.0))
            if field == 0:
                self._qubit_noise[qubit] = (int(v), nparam)
            else:
                self._qubit_noise[qubit] = (ntype, v)
        elif 0xD010 <= a <= 0xD01F:
            self.io.writeln(f"?READ-ONLY: ${a:04X}")
        elif 0x0100 <= a <= 0x01FF:
            # Bloch sphere POKE — prepare qubit state via memory map
            import math
            off = a - 0x0100
            qubit, field = off // QUBIT_BLOCK, off % QUBIT_BLOCK
            if qubit >= self.num_qubits:
                self.io.writeln(f"?QUBIT {qubit} OUT OF RANGE")
                return
            if field == 0:
                # POKE P(|1⟩) — set probability via RY rotation
                p1 = max(0.0, min(1.0, v))
                theta = 2 * math.asin(math.sqrt(p1))
                # This requires a circuit context — store for next RUN
                if not hasattr(self, '_poke_state_prep'):
                    self._poke_state_prep = {}
                self._poke_state_prep[qubit] = ('RY', theta)
                self.io.writeln(f"POKE q{qubit}: P(1)={p1:.4f} -> RY({theta:.4f})")
            elif field in (1, 2, 3):
                # Bloch x, y, z — store target vector
                if not hasattr(self, '_poke_state_prep'):
                    self._poke_state_prep = {}
                labels = {1: 'Bx', 2: 'By', 3: 'Bz'}
                self._poke_state_prep[(qubit, labels[field])] = v
                self.io.writeln(f"POKE q{qubit}.{labels[field]} = {v:.4f}")
            else:
                self.io.writeln(f"?FIELD {field} NOT WRITABLE (use 0=P(1), 1-3=Bloch)")

    # ── Commands ───────────────────────────────────────────────────────

    def cmd_poke(self, rest: str) -> None:
        """POKE addr, value — write to memory-mapped address."""
        from qbasic_core.engine import RE_POKE
        m = RE_POKE.match(f"POKE {rest}")
        if not m:
            self.io.writeln("?USAGE: POKE <addr>, <value>")
            return
        addr = self.eval_expr(m.group(1))
        val = self.eval_expr(m.group(2))
        self._poke(addr, val)

    def cmd_sys(self, rest: str) -> None:
        """SYS addr — execute a built-in or user routine."""
        rest = rest.strip().upper()
        if rest.startswith('INSTALL'):
            parts = rest[7:].strip().split(',', 1)
            if len(parts) != 2:
                self.io.writeln("?USAGE: SYS INSTALL <addr>, <subroutine_name>")
                return
            addr = int(self.eval_expr(parts[0].strip()))
            name = parts[1].strip()
            if not (0xF000 <= addr <= 0xFFFF):
                self.io.writeln(f"?USER SYS RANGE: $F000-$FFFF (got ${addr:04X})")
                return
            self._user_sys[addr] = name
            self.io.writeln(f"INSTALLED {name} AT ${addr:04X}")
            return
        addr = int(self.eval_expr(rest))
        if addr in SYS_ROUTINES:
            self.cmd_demo(SYS_ROUTINES[addr])
        elif addr in self._user_sys:
            name = self._user_sys[addr]
            if name in self.subroutines:
                sub = self.subroutines[name]
                body = sub['body'] if isinstance(sub, dict) else sub
                for stmt in body:
                    self.process(stmt)
            else:
                self.io.writeln(f"?UNDEFINED ROUTINE: {name}")
        else:
            self.io.writeln(f"?NO ROUTINE AT ${addr:04X}")

    # File-writing commands blocked from USR execution to prevent
    # user-defined routines from performing uncontrolled I/O.
    _USR_BLOCKED_CMDS = frozenset({'SAVE', 'EXPORT', 'CSV', 'OPEN'})

    def _usr_fn(self, addr: float) -> float:
        """USR(addr) — execute routine, return last measurement result."""
        a = int(addr)
        if a in SYS_ROUTINES:
            self.cmd_demo(SYS_ROUTINES[a])
        elif a in self._user_sys:
            name = self._user_sys[a]
            if name in self.subroutines:
                sub = self.subroutines[name]
                body = sub['body'] if isinstance(sub, dict) else sub
                for stmt in body:
                    first_word = stmt.split(None, 1)[0].upper() if stmt.strip() else ''
                    if first_word in self._USR_BLOCKED_CMDS:
                        self.io.writeln(f"  ?BLOCKED IN USR: {first_word}")
                        continue
                    self.process(stmt)
        if self.last_counts:
            top = max(self.last_counts, key=self.last_counts.get)
            return float(int(top, 2))
        return 0.0

    def cmd_wait(self, rest: str) -> None:
        """WAIT addr, mask[, value[, timeout]] — block until (PEEK(addr) AND mask) == value."""
        parts = [p.strip() for p in rest.split(',')]
        if len(parts) < 2:
            self.io.writeln("?USAGE: WAIT <addr>, <mask>[, <value>[, <timeout>]]")
            return
        addr = int(self.eval_expr(parts[0]))
        mask = int(self.eval_expr(parts[1]))
        target = int(self.eval_expr(parts[2])) if len(parts) > 2 else mask
        timeout = float(self.eval_expr(parts[3])) if len(parts) > 3 else 30.0
        t0 = time.time()
        while time.time() - t0 < timeout:
            val = int(self._peek(addr))
            if (val & mask) == target:
                return
            time.sleep(0.05)
        self.io.writeln("?WAIT TIMEOUT")

    def cmd_catalog(self) -> None:
        """CATALOG — list all SYS routines with addresses."""
        self.io.writeln("\n  Built-in SYS Routines:")
        for addr, name in sorted(SYS_ROUTINES.items()):
            self.io.writeln(f"    ${addr:04X}  {name}")
        if self._user_sys:
            self.io.writeln("\n  User SYS Routines:")
            for addr, name in sorted(self._user_sys.items()):
                self.io.writeln(f"    ${addr:04X}  {name}")
        self.io.writeln('')

    def cmd_dump(self, rest: str = '') -> None:
        """DUMP [start] [end] — hex dump of memory map."""
        parts = rest.split()
        start = int(self.eval_expr(parts[0])) if parts else 0x0000
        end = int(self.eval_expr(parts[1])) if len(parts) > 1 else start + 0x3F
        self.io.writeln('')
        for row_start in range(start, end + 1, 16):
            vals = [self._peek(row_start + i) for i in range(16) if row_start + i <= end]
            hex_part = ' '.join(f'{int(v) & 0xFF:02X}' if abs(v) < 256 else f'{v:4.1f}'[:4]
                               for v in vals)
            self.io.writeln(f"  ${row_start:04X}: {hex_part}")
        self.io.writeln('')

    def cmd_map(self) -> None:
        """MAP — print the full memory map with current values."""
        self.io.writeln("\n  Memory Map:")
        self.io.writeln(f"  $0000-$003F  Zero Page        (64 slots)")
        nz = sum(1 for v in self._zero_page if v != 0)
        self.io.writeln(f"               {nz} non-zero values")
        self.io.writeln(f"  $0100-$01FF  Qubit State      ({self.num_qubits} qubits, 8 addr/qubit)")
        if self.last_sv is not None:
            for q in range(min(self.num_qubits, 4)):
                p1 = self._peek(0x0100 + q * 8)
                bz = self._peek(0x0100 + q * 8 + 3)
                self.io.writeln(f"               q{q}: P(1)={p1:.3f} Bz={bz:.3f}")
            if self.num_qubits > 4:
                self.io.writeln(f"               ... ({self.num_qubits - 4} more)")
        self.io.writeln(f"  $D000-$D007  QPU Config")
        for addr, name in sorted(CFG_NAMES.items()):
            val = self._peek_config(addr)
            self.io.writeln(f"               ${addr:04X} {name:16s} = {val}")
        self.io.writeln(f"  $D010-$D014  QPU Status (read-only)")
        for addr, name in sorted(STS_NAMES.items()):
            val = self._status.get(addr, 0.0)
            self.io.writeln(f"               ${addr:04X} {name:20s} = {val}")
        self.io.writeln(f"  $E000-$E0B0  SYS Routines     ({len(SYS_ROUTINES)} built-in)")
        self.io.writeln(f"  $F000-$FFFF  User SYS         ({len(self._user_sys)} installed)")
        self.io.writeln('')

    def cmd_monitor(self) -> None:
        """MONITOR — interactive hex monitor for PEEK/POKE."""
        self.io.writeln("MONITOR — type address to PEEK, addr=val to POKE, Q to quit")
        while True:
            try:
                line = self.io.read_line('* ').strip()
            except (KeyboardInterrupt, EOFError):
                self.io.writeln('')
                break
            if not line or line.upper() == 'Q':
                break
            if '=' in line:
                parts = line.split('=', 1)
                try:
                    addr = int(self.eval_expr(parts[0].strip()))
                    val = self.eval_expr(parts[1].strip())
                    self._poke(addr, val)
                    self.io.writeln(f"  ${addr:04X} <- {val}")
                except Exception as e:
                    self.io.writeln(f"?{e}")
            else:
                try:
                    addr = int(self.eval_expr(line))
                    val = self._peek(addr)
                    self.io.writeln(f"  ${addr:04X} = {val}")
                except Exception as e:
                    self.io.writeln(f"?{e}")
        self.io.writeln("MONITOR OFF")
