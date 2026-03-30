"""QUBASIC LOCC commands mixin — cmd_locc, cmd_send, cmd_share, cmd_connect, cmd_disconnect, cmd_loccinfo."""

import re

from qubasic_core.engine import (
    LOCCEngine,
    _get_ram_gb,
    LOCC_MAX_JOINT_QUBITS, LOCC_MAX_SPLIT_QUBITS, LOCC_MAX_REGISTERS,
    RAM_BUDGET_FRACTION,
)


class LOCCCommandsMixin:
    """LOCC user-facing commands for QBasicTerminal.

    Requires: TerminalProtocol — uses self.locc, self.locc_mode, self.program,
    self.variables, self.eval_expr(), self.io.
    """

    def _locc_noise_param(self) -> float:
        """Return the scalar depolarizing probability for LOCC noise.

        Stored by cmd_noise when a depolarizing model is set.
        """
        return getattr(self, '_noise_depol_p', 0.0)

    def cmd_locc(self, rest):
        args = rest.upper().split()
        if not args:
            if self.locc_mode:
                mode = "JOINT" if self.locc.joint else "SPLIT"
                reg_desc = '  '.join(f"{n}={s}q" for n, s in
                                     zip(self.locc.names, self.locc.sizes))
                self.io.writeln(f"LOCC {mode}: {reg_desc}")
                tot, peak = self.locc.mem_gb()
                self.io.writeln(f"  RAM per instance: {tot:.1f} GB (with overhead)")
                if self.locc.classical:
                    self.io.writeln(f"  Classical: {self.locc.classical}")
            else:
                self.io.writeln("LOCC OFF. Usage: LOCC [JOINT] <n1> <n2> [n3 ...]")
            return

        if args[0] == 'OFF':
            self.locc = None
            self.locc_mode = False
            self.io.writeln("LOCC OFF — back to normal Aer mode")
            return

        if args[0] == 'STATUS':
            if not self.locc_mode:
                self.io.writeln("NOT IN LOCC MODE")
                return
            mode = "JOINT" if self.locc.joint else "SPLIT"
            reg_desc = '  '.join(f"{n}={s}q" for n, s in
                                 zip(self.locc.names, self.locc.sizes))
            self.io.writeln(f"LOCC {mode}: {reg_desc}")
            tot, peak = self.locc.mem_gb()
            self.io.writeln(f"  RAM per instance: {tot:.1f} GB (with overhead)")
            ram = _get_ram_gb()
            if ram:
                budget = ram[0] * RAM_BUDGET_FRACTION
                max_par = int(budget / tot) if tot > 0 else 0
                if max_par > 0:
                    self.io.writeln(f"  Max parallel in 80% budget: ~{max_par}")
            self.io.writeln(f"  Classical vars: {self.locc.classical if self.locc.classical else '(none)'}")
            return

        joint = False
        nums = args
        if args[0] == 'JOINT':
            joint = True
            nums = args[1:]
        elif args[0] == 'SPLIT':
            nums = args[1:]

        # Support "2+2+2" notation: split on + if present
        expanded = []
        for n in nums:
            expanded.extend(n.split('+'))
        nums = [n for n in expanded if n]

        if len(nums) < 2:
            self.io.writeln("?USAGE: LOCC [JOINT|SPLIT] <n1> <n2> [n3 ...]")
            return
        if len(nums) > LOCC_MAX_REGISTERS:
            self.io.writeln(f"?MAX {LOCC_MAX_REGISTERS} registers (A-Z)")
            return

        sizes = [int(n) for n in nums]
        total = sum(sizes)
        if joint and total > LOCC_MAX_JOINT_QUBITS:
            self.io.writeln(f"?JOINT mode limited to {LOCC_MAX_JOINT_QUBITS} total qubits (requested {total})")
            return
        if not joint and max(sizes) > LOCC_MAX_SPLIT_QUBITS:
            self.io.writeln(f"?Each register limited to {LOCC_MAX_SPLIT_QUBITS} qubits")
            return

        # Pre-check RAM before allocating
        mode = "JOINT" if joint else "SPLIT"
        noise_p = self._locc_noise_param()
        temp_eng = LOCCEngine(sizes, joint=joint, noise_param=noise_p)
        tot, peak = temp_eng.mem_gb()
        ram = _get_ram_gb()
        if ram and tot > ram[1]:
            self.io.writeln(f"?BLOCKED: LOCC {mode} needs ~{tot:.1f} GB but only "
                           f"{ram[1]:.1f} GB available. Reduce register sizes.")
            return

        self.locc = temp_eng
        self.locc_mode = True
        reg_desc = '  '.join(f"{n}={s}q" for n, s in
                             zip(self.locc.names, sizes))
        self.io.writeln(f"LOCC {mode}: {reg_desc}  ({total} total)")
        self.io.writeln(f"  RAM per instance: {tot:.1f} GB (with overhead)")
        if ram:
            sys_total, avail = ram
            budget = sys_total * RAM_BUDGET_FRACTION
            if tot > 0:
                max_par = int(budget / tot)
                if max_par > 0:
                    self.io.writeln(f"  Max parallel instances in 80% budget: ~{max_par}")
        if not joint:
            self.io.writeln(f"  Registers are INDEPENDENT — no cross-register entanglement")
            self.io.writeln(f"  Use SEND/IF for classical coordination")
        else:
            self.io.writeln(f"  Joint statevector — use SHARE for pre-shared entanglement")
        if peak > 30:
            self.io.writeln(f"  WARNING: large registers. Keep SHOTS low for SEND-based protocols.")

    def cmd_send(self, rest):
        if not self.locc_mode:
            self.io.writeln("?SEND requires LOCC mode")
            return
        m = re.match(r'([A-Z])\s+(\S+)\s*->\s*(\w+)', rest, re.IGNORECASE)
        if not m:
            self.io.writeln("?USAGE: SEND <reg> <qubit> -> <var>")
            return
        reg = m.group(1).upper()
        if reg not in self.locc.names:
            self.io.writeln(f"?UNKNOWN REGISTER: {reg} (have {', '.join(self.locc.names)})")
            return
        qubit = int(self.eval_expr(m.group(2)))
        var = m.group(3)
        n_reg = self.locc.get_size(reg)
        if qubit < 0 or qubit >= n_reg:
            self.io.writeln(f"?QUBIT {qubit} OUT OF RANGE for register {reg} (0-{n_reg-1})")
            return
        outcome = self.locc.send(reg, qubit)
        self.variables[var] = outcome
        self.locc.classical[var] = outcome
        self.io.writeln(f"  {reg}[{qubit}] -> {var} = {outcome}")

    def cmd_share(self, rest):
        if not self.locc_mode:
            self.io.writeln("?SHARE requires LOCC mode")
            return
        m = re.match(r'([A-Z])\s+(\d+)\s*,?\s*([A-Z])\s+(\d+)', rest, re.IGNORECASE)
        if not m:
            self.io.writeln("?USAGE: SHARE <reg1> <qubit> <reg2> <qubit>")
            return
        reg1, q1 = m.group(1).upper(), int(m.group(2))
        reg2, q2 = m.group(3).upper(), int(m.group(4))
        for r in (reg1, reg2):
            if r not in self.locc.names:
                self.io.writeln(f"?UNKNOWN REGISTER: {r}")
                return
        try:
            self.locc.share(reg1, q1, reg2, q2)
            self.io.writeln(f"  Bell pair |Phi+> created: {reg1}[{q1}] <-> {reg2}[{q2}]")
        except RuntimeError as e:
            self.io.writeln(f"?{e}")

    def cmd_loccinfo(self):
        """Show LOCC protocol metrics after a run."""
        if not self.locc_mode:
            self.io.writeln("?NOT IN LOCC MODE")
            return
        mode = "JOINT" if self.locc.joint else "SPLIT"
        self.io.writeln(f"\n  LOCC Protocol Metrics ({mode})")
        reg_desc = '  '.join(f"{n}={s}q" for n, s in
                             zip(self.locc.names, self.locc.sizes))
        self.io.writeln(f"  Registers: {reg_desc}  ({self.locc.n_regs} parties)")

        # Entanglement creation
        n_shares = sum(1 for l in self.program.values()
                       if re.search(r'\bSHARE\b', l, re.IGNORECASE))
        if n_shares:
            self.io.writeln(f"  Entanglement: {n_shares} SHARE operation(s)")

        # Classical communication
        n_classical = len(self.locc.classical)
        self.io.writeln(f"  Classical bits exchanged: {n_classical}")
        if self.locc.classical:
            for k, v in self.locc.classical.items():
                self.io.writeln(f"    {k} = {v}")

        # Measurement outcomes and correction paths
        n_sends = sum(1 for l in self.program.values()
                      if re.search(r'\bSEND\b', l, re.IGNORECASE))
        n_ifs = sum(1 for l in self.program.values()
                    if re.search(r'\bIF\b.*\bTHEN\b.*@', l, re.IGNORECASE))
        self.io.writeln(f"  SEND operations: {n_sends}")
        self.io.writeln(f"  Conditional corrections: {n_ifs}")
        self.io.writeln(f"  Communication rounds: ~{n_sends}")

        # Branch statistics from classical bits
        if n_sends > 0 and self.locc.classical:
            bits = [v for v in self.locc.classical.values() if isinstance(v, (int, float))]
            if bits:
                n0 = sum(1 for b in bits if b == 0)
                n1 = sum(1 for b in bits if b == 1)
                self.io.writeln(f"  Branch stats: {n0} zeros, {n1} ones "
                               f"(last run)")

        # Noise
        noise_p = getattr(self.locc, 'noise_param', 0.0)
        if noise_p > 0:
            self.io.writeln(f"  Noise: depolarizing p={noise_p}")

        tot, peak = self.locc.mem_gb()
        self.io.writeln(f"  Memory: {tot:.1f} GB")
        self.io.writeln('')
