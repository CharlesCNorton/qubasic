"""QUBASIC LOCC display mixin — state inspection and Bloch sphere for LOCC registers."""

import re
from qubasic_core.host import HostState


class LOCCDisplayMixin(HostState):
    """LOCC display methods for QBasicTerminal.

    Requires: TerminalProtocol — uses self.locc, self._print_statevector(),
    self._print_bloch_single(), self.print_histogram(), self.io.
    """

    def _locc_state(self, rest: str = '') -> None:
        reg = rest.strip().upper() if rest else ''
        if self.locc.joint:
            if not reg or reg in self.locc.names:
                sizes = '+'.join(str(s) for s in self.locc.sizes)
                self.io.writeln(f"\n  Joint statevector ({sizes} qubits):")
                self._print_statevector(self.locc.sv, self.locc.n_total)
        else:
            show = [reg] if reg and reg in self.locc.names else self.locc.names
            for name in show:
                size = self.locc.get_size(name)
                self.io.writeln(f"\n  Register {name} ({size} qubits):")
                self._print_statevector(self.locc.svs[name], size)

    def _locc_bloch(self, rest: str) -> None:
        m = re.match(r'([A-Z])\s*(\d*)', rest.strip(), re.IGNORECASE) if rest.strip() else None
        if m and m.group(1):
            reg = m.group(1).upper()
            if reg not in self.locc.names:
                self.io.writeln(f"?UNKNOWN REGISTER: {reg}")
                return
            sv = self.locc.get_sv(reg)
            n = self.locc.get_n(reg)
            idx = self.locc._idx(reg)
            if m.group(2):
                q = int(m.group(2))
                actual_q = q if not self.locc.joint else q + self.locc.offsets[idx]
                self.io.writeln(f"  [Register {reg}, qubit {q}]")
                self._print_bloch_single(sv, actual_q, n)
            else:
                n_show = self.locc.get_size(reg)
                for q in range(min(n_show, 4)):
                    actual_q = q if not self.locc.joint else q + self.locc.offsets[idx]
                    self.io.writeln(f"  [Register {reg}, qubit {q}]")
                    self._print_bloch_single(sv, actual_q, n)
                    self.io.writeln('')
        else:
            self.io.writeln(f"?USAGE: BLOCH <reg> [qubit]  (registers: {', '.join(self.locc.names)})")

    def _locc_display_results(self, per_reg: dict, counts_joint: dict) -> None:
        """Display per-register and joint histograms."""
        for name in self.locc.names:
            size = self.locc.get_size(name)
            self.io.writeln(f"\n  Register {name} ({size}q):")
            self.print_histogram(per_reg[name])
        if counts_joint and self.locc.n_regs <= 4:
            jlabel = '|'.join(self.locc.names)
            self.io.writeln(f"\n  Joint ({jlabel}):")
            self.print_histogram(counts_joint)
