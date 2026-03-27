"""QBASIC LOCC mixin — commands, execution, and display for multi-register mode."""

import re
import time

import numpy as np

from qbasic_core.engine import (
    LOCCEngine, ExecResult,
    GATE_TABLE, GATE_ALIASES,
    _np_gate_matrix, _sample_one_np,
    _get_ram_gb,
    LOCC_MAX_JOINT_QUBITS, LOCC_MAX_SPLIT_QUBITS, LOCC_MAX_REGISTERS,
    LOCC_SEND_SHOT_CAP, LOCC_SEND_QUBIT_THRESHOLD,
    RAM_BUDGET_FRACTION,
    RE_SEND, RE_SHARE, RE_AT_REG_LINE,
    RE_MEAS, RE_RESET, RE_MEASURE_BASIS,
    RE_CTRL, RE_INV,
)


class LOCCMixin:
    """LOCC commands, execution, and display for QBasicTerminal.

    Requires: TerminalProtocol — uses self.locc, self.locc_mode, self.shots,
    self.program, self.variables, self.subroutines, self._max_iterations,
    self._custom_gates, self.eval_expr(), self._eval_with_vars(),
    self._substitute_vars(), self._expand_statement(), self._tokenize_gate(),
    self._gate_info(), self._resolve_qubit(), self._parse_syndrome(),
    self._split_colon_stmts(), self._exec_control_flow(),
    self.print_histogram(), self._print_statevector(), self._print_bloch_single().
    """

    # ── LOCC Commands ─────────────────────────────────────────────────

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
        temp_eng = LOCCEngine(sizes, joint=joint)
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

    def cmd_connect(self, rest: str) -> None:
        """CONNECT "host:port" AS <reg> — attach a remote quantum register.

        Creates a local register that can be used with LOCC commands.
        The connection info is stored for future network-backed execution.
        Currently uses local simulation as a stand-in for the network layer.
        """
        import re as _re
        m = _re.match(r'"?([^"]+)"?\s+AS\s+([A-Z])', rest, _re.IGNORECASE)
        if not m:
            self.io.writeln('?USAGE: CONNECT "host:port" AS <reg>')
            return
        endpoint = m.group(1).strip()
        reg = m.group(2).upper()
        if not hasattr(self, '_connections'):
            self._connections = {}
        self._connections[reg] = endpoint
        # If not in LOCC mode, auto-enter with a default local register + the remote
        if not self.locc_mode:
            self.cmd_locc(f'3 3')
        self.io.writeln(f"CONNECTED {reg} -> {endpoint} (local simulation)")

    def cmd_disconnect(self, rest: str) -> None:
        """DISCONNECT <reg> — detach a remote register."""
        reg = rest.strip().upper()
        if hasattr(self, '_connections') and reg in self._connections:
            del self._connections[reg]
            self.io.writeln(f"DISCONNECTED {reg}")
        else:
            self.io.writeln(f"?{reg} NOT CONNECTED")

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
        n_classical = len(self.locc.classical)
        self.io.writeln(f"  Classical bits exchanged: {n_classical}")
        if self.locc.classical:
            for k, v in self.locc.classical.items():
                self.io.writeln(f"    {k} = {v}")
        n_sends = sum(1 for l in self.program.values()
                      if re.search(r'\bSEND\b', l, re.IGNORECASE))
        self.io.writeln(f"  SEND operations: {n_sends}")
        self.io.writeln(f"  Communication rounds: ~{n_sends}")
        tot, peak = self.locc.mem_gb()
        self.io.writeln(f"  Memory: {tot:.1f} GB")
        self.io.writeln('')

    # ── LOCC Display ──────────────────────────────────────────────────

    def _locc_state(self, rest=''):
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

    def _locc_bloch(self, rest):
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

    # ── LOCC Execution ────────────────────────────────────────────────

    def _locc_run(self):
        """Execute program in LOCC mode (N registers)."""
        sorted_lines = sorted(self.program.keys())
        if not sorted_lines:
            self.io.writeln("NOTHING TO RUN")
            return
        has_measure = any(self.program[l].strip().upper() == 'MEASURE'
                         for l in sorted_lines)
        has_send = any(re.search(r'\bSEND\b', self.program[l], re.IGNORECASE)
                       for l in sorted_lines)
        if has_send:
            self._locc_run_with_send(sorted_lines, has_measure)
        else:
            self._locc_run_vectorized(sorted_lines, has_measure)

    def _locc_run_with_send(self, sorted_lines, has_measure):
        """LOCC execution with SEND — prefix/suffix split optimization.

        Executes the deterministic prefix (before first SEND) once,
        snapshots the quantum state, then re-executes only the suffix
        (from first SEND onward) per shot. Falls back to full re-execution
        if the prefix contains jumps that could skip past the SEND.
        """
        from qbasic_core.scope import Scope
        from qbasic_core.statements import SendStmt, GotoStmt, GosubStmt

        # Check if prefix contains jumps — if so, fall back to full re-exec
        first_send_ip = None
        for i, ln in enumerate(sorted_lines):
            p = self._get_parsed(ln)
            if isinstance(p, SendStmt):
                first_send_ip = i
                break
        has_prefix_jumps = False
        if first_send_ip is not None and first_send_ip > 0:
            for i in range(first_send_ip):
                p = self._get_parsed(sorted_lines[i])
                if isinstance(p, (GotoStmt, GosubStmt)):
                    has_prefix_jumps = True
                    break

        sizes_str = '+'.join(str(s) for s in self.locc.sizes)
        mode = "JOINT" if self.locc.joint else "SPLIT"
        shots = self.shots
        max_q = max(self.locc.sizes)
        if max_q > LOCC_SEND_QUBIT_THRESHOLD and shots > LOCC_SEND_SHOT_CAP:
            self.io.writeln(f"  WARNING: capping at {LOCC_SEND_SHOT_CAP} shots for "
                            f"{max_q}-qubit LOCC w/ SEND (per-shot re-execution)")
            shots = LOCC_SEND_SHOT_CAP

        # Execute deterministic prefix once (skip if jumps make it unsafe)
        self.locc.reset()
        if has_prefix_jumps or first_send_ip == 0 or first_send_ip is None:
            send_ip = 0  # no safe prefix — re-exec from start
        else:
            prefix_vars = dict(self.variables)
            try:
                send_ip = self._locc_execute_program(sorted_lines, stop_before_send=True,
                                                      run_vars=Scope(prefix_vars))
            except (RuntimeError, ValueError) as e:
                self.io.writeln(f"?RUNTIME ERROR: {e}")
                return

        # Snapshot state after prefix
        snap = self.locc.snapshot()
        snap_vars = dict(self.variables)

        per_reg = {name: {} for name in self.locc.names}
        counts_joint = {}
        t0 = time.time()
        progress_interval = max(1, shots // 10)

        for shot in range(shots):
            # Restore to post-prefix state
            self.locc.restore(snap)
            self.variables.clear()
            self.variables.update(snap_vars)
            try:
                self._locc_execute_program(sorted_lines, start_ip=send_ip,
                                           run_vars=Scope(self.variables))
            except (RuntimeError, ValueError) as e:
                self.io.writeln(f"?RUNTIME ERROR: {e}")
                return
            if has_measure:
                if self.locc.joint:
                    out = _sample_one_np(self.locc.sv, self.locc.n_total)
                    parts = []
                    pos = len(out)
                    for i in range(self.locc.n_regs):
                        size = self.locc.sizes[i]
                        parts.append(out[pos - size:pos])
                        pos -= size
                else:
                    parts = [_sample_one_np(self.locc.svs[name],
                             self.locc.get_size(name))
                             for name in self.locc.names]
                for name, bits in zip(self.locc.names, parts):
                    per_reg[name][bits] = per_reg[name].get(bits, 0) + 1
                jkey = '|'.join(parts)
                counts_joint[jkey] = counts_joint.get(jkey, 0) + 1
            if shots > 50 and (shot + 1) % progress_interval == 0:
                pct = 100 * (shot + 1) // shots
                self.io.write(f"  {pct}% ({shot+1}/{shots} shots)...\r")

        if shots > 50:
            self.io.write(" " * 40 + '\r')
        dt = time.time() - t0
        self.io.writeln(f"\nRAN {len(self.program)} lines, LOCC {mode} "
                        f"{sizes_str}q, {shots} shots in {dt:.2f}s")
        if has_measure:
            self._locc_display_results(per_reg, counts_joint)
            self.last_counts = counts_joint

    def _locc_run_vectorized(self, sorted_lines, has_measure):
        """LOCC execution without SEND — single execution, vectorized sampling."""
        sizes_str = '+'.join(str(s) for s in self.locc.sizes)
        mode = "JOINT" if self.locc.joint else "SPLIT"
        t0 = time.time()

        self.locc.reset()
        try:
            self._locc_execute_program(sorted_lines)
        except (RuntimeError, ValueError) as e:
            self.io.writeln(f"?RUNTIME ERROR: {e}")
            return

        if has_measure:
            per_reg, cj = self.locc.sample_joint(self.shots)
            dt = time.time() - t0
            self.io.writeln(f"\nRAN {len(self.program)} lines, LOCC {mode} "
                            f"{sizes_str}q, {self.shots} shots in {dt:.2f}s")
            self._locc_display_results(per_reg, cj)
            if not cj:
                self.io.writeln(f"\n  (SPLIT mode, no SEND — registers are independent)")
            self.last_counts = cj if cj else per_reg.get('A', {})
        else:
            dt = time.time() - t0
            self.io.writeln(f"\nRAN {len(self.program)} lines, LOCC in {dt:.2f}s")
            regs = ', '.join(self.locc.names)
            self.io.writeln(f"(no MEASURE — use STATE {regs} to inspect)")

    def _locc_display_results(self, per_reg, counts_joint):
        """Display per-register and joint histograms."""
        for name in self.locc.names:
            size = self.locc.get_size(name)
            self.io.writeln(f"\n  Register {name} ({size}q):")
            self.print_histogram(per_reg[name])
        if counts_joint and self.locc.n_regs <= 4:
            jlabel = '|'.join(self.locc.names)
            self.io.writeln(f"\n  Joint ({jlabel}):")
            self.print_histogram(counts_joint)

    def _locc_execute_program(self, sorted_lines, *, start_ip: int = 0,
                              run_vars=None, stop_before_send: bool = False) -> int:
        """Execute LOCC program lines.

        Returns the ip where execution stopped (end of program or at first SEND
        if stop_before_send is True).
        """
        from qbasic_core.scope import Scope
        from qbasic_core.exec_context import ExecContext
        from qbasic_core.statements import MeasureStmt, SendStmt
        if run_vars is None:
            run_vars = Scope(self.variables)
        ctx = ExecContext(
            sorted_lines=sorted_lines, ip=start_ip, run_vars=run_vars,
            max_iterations=self._max_iterations,
        )
        _parsed = [self._get_parsed(ln) for ln in sorted_lines]
        _stmts = [self.program[ln].strip() for ln in sorted_lines]
        n_lines = len(sorted_lines)
        while ctx.ip < n_lines:
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            if isinstance(_parsed[ctx.ip], MeasureStmt):
                ctx.ip += 1
                continue
            if stop_before_send and isinstance(_parsed[ctx.ip], SendStmt):
                return ctx.ip
            try:
                result = self._locc_exec_line(_stmts[ctx.ip], ctx.loop_stack, sorted_lines, ctx.ip, run_vars)
            except Exception as e:
                raise RuntimeError(f"LINE {sorted_lines[ctx.ip]}: {e}") from None
            if result is ExecResult.END:
                break
            elif isinstance(result, int):
                ctx.ip = result
            else:
                ctx.ip += 1
        return ctx.ip

    def _locc_exec_line(self, stmt, loop_stack, sorted_lines, ip, run_vars):
        """Execute one line in LOCC mode."""
        from qbasic_core.parser import parse_stmt
        from qbasic_core.statements import (
            SendStmt, ShareStmt, AtRegStmt, CompoundStmt,
            RemStmt, MeasureStmt, EndStmt, BarrierStmt, ReturnStmt,
        )
        parsed = parse_stmt(stmt)

        # Fast-path for terminals
        if isinstance(parsed, (RemStmt, MeasureStmt, BarrierStmt)):
            return ExecResult.ADVANCE
        if isinstance(parsed, EndStmt):
            return ExecResult.END
        if isinstance(parsed, ReturnStmt):
            if not self._gosub_stack:
                raise RuntimeError("RETURN WITHOUT GOSUB")
            return self._gosub_stack.pop()
        if isinstance(parsed, CompoundStmt):
            for sub in parsed.parts:
                self._locc_exec_line(sub, loop_stack, sorted_lines, ip, run_vars)
            return ExecResult.ADVANCE

        # Fast-path for LOCC-specific statements
        if isinstance(parsed, SendStmt):
            qubit = int(self.eval_expr(parsed.qubit_expr))
            outcome = self.locc.send(parsed.reg, qubit)
            run_vars[parsed.var] = outcome
            self.variables[parsed.var] = outcome
            self.locc.classical[parsed.var] = outcome
            return ExecResult.ADVANCE
        if isinstance(parsed, ShareStmt):
            self.locc.share(parsed.reg1, parsed.q1, parsed.reg2, parsed.q2)
            return ExecResult.ADVANCE
        if isinstance(parsed, AtRegStmt):
            if self._locc_try_special(parsed.reg, parsed.inner, run_vars):
                return ExecResult.ADVANCE
            self._locc_apply_gate(parsed.reg, parsed.inner)
            return ExecResult.ADVANCE

        # Control flow
        handled, result = self._exec_control_flow(
            stmt, loop_stack, sorted_lines, ip, run_vars,
            lambda s, ls, sl, i, rv: self._locc_exec_line(s, ls, sl, i, rv))
        if handled:
            return result

        if ':' in stmt:
            for sub in self._split_colon_stmts(stmt):
                self._locc_exec_line(sub, loop_stack, sorted_lines, ip, run_vars)
            return ExecResult.ADVANCE

        raise ValueError(f"LOCC mode requires @A/@B prefix, SEND, SHARE, or IF: {stmt}")

    def _locc_try_special(self, reg, stmt, run_vars):
        """Handle MEAS/RESET/MEASURE_X/Y/Z/SYNDROME in LOCC mode."""
        from qbasic_core.parser import parse_stmt
        from qbasic_core.statements import MeasStmt, ResetStmt, MeasureBasisStmt, SyndromeStmt
        p = parse_stmt(stmt)

        if isinstance(p, MeasStmt):
            qubit = int(self._eval_with_vars(p.qubit_expr, run_vars))
            outcome = self.locc.send(reg, qubit)
            run_vars[p.var] = outcome
            self.variables[p.var] = outcome
            self.locc.classical[p.var] = outcome
            return True

        if isinstance(p, ResetStmt):
            qubit = int(self._eval_with_vars(p.qubit_expr, run_vars))
            outcome = self.locc.send(reg, qubit)
            if outcome == 1:
                self.locc.apply(reg, 'X', (), [qubit])
            return True

        if isinstance(p, MeasureBasisStmt):
            qubit = int(self._eval_with_vars(p.qubit_expr, run_vars))
            if p.basis == 'X':
                self.locc.apply(reg, 'H', (), [qubit])
            elif p.basis == 'Y':
                self.locc.apply(reg, 'SDG', (), [qubit])
                self.locc.apply(reg, 'H', (), [qubit])
            outcome = self.locc.send(reg, qubit)
            var = f"m{p.basis.lower()}_{qubit}"
            run_vars[var] = outcome
            self.variables[var] = outcome
            self.locc.classical[var] = outcome
            return True

        parsed = self._parse_syndrome(stmt, run_vars)
        if parsed is not None:
            pauli_str, qubits, var = parsed
            n_reg = self.locc.get_size(reg)
            anc = n_reg - 1
            if anc in qubits:
                raise ValueError(
                    f"Qubit {anc} needed as ancilla but appears in stabilizer. "
                    f"Increase register size by 1.")
            self.locc.apply(reg, 'H', (), [anc])
            for p, q in zip(pauli_str, qubits):
                if p != 'I':
                    self.locc.apply(reg, self._PAULI_TO_CONTROLLED[p], (), [anc, q])
            self.locc.apply(reg, 'H', (), [anc])
            outcome = self.locc.send(reg, anc)
            if outcome == 1:
                self.locc.apply(reg, 'X', (), [anc])
            run_vars[var] = outcome
            self.variables[var] = outcome
            self.locc.classical[var] = outcome
            return True

        return False

    def _locc_apply_gate(self, reg, gate_stmt):
        """Parse and apply a gate to a LOCC register via numpy.

        Uses LOCCRegBackend for standard gates. CTRL and INV modifiers
        use apply_matrix directly since they need matrix construction.
        """
        from qbasic_core.backend import LOCCRegBackend
        backend = LOCCRegBackend(self.locc, reg)
        n_reg = self.locc.get_size(reg)
        expanded = self._expand_statement(gate_stmt)
        for stmt in expanded:
            tokens = self._tokenize_gate(stmt)
            if not tokens:
                continue
            gate_name = tokens[0].upper()
            gate_name = GATE_ALIASES.get(gate_name, gate_name)
            if gate_name == 'BARRIER':
                continue

            # CTRL modifier
            m_ctrl = RE_CTRL.match(stmt)
            if m_ctrl:
                inner_name = GATE_ALIASES.get(m_ctrl.group(1).upper(), m_ctrl.group(1).upper())
                args = [a.strip() for a in m_ctrl.group(2).replace(',', ' ').split()]
                ctrl_qubit = self._resolve_qubit(args[0], n_qubits=n_reg)
                target_qubits = [self._resolve_qubit(a, n_qubits=n_reg) for a in args[1:]]
                inner_matrix = _np_gate_matrix(inner_name, ())
                dim = inner_matrix.shape[0]
                n_inner_qubits = int(np.log2(dim))
                if len(target_qubits) != n_inner_qubits:
                    raise ValueError(
                        f"CTRL {inner_name} needs 1 control + {n_inner_qubits} "
                        f"target qubit(s), got {len(target_qubits)} target(s)")
                controlled = np.eye(2 * dim, dtype=complex)
                controlled[dim:, dim:] = inner_matrix
                self.locc.apply_matrix(reg, controlled, [ctrl_qubit] + target_qubits)
                continue

            # INV modifier
            m_inv = RE_INV.match(stmt)
            if m_inv:
                inner_name = GATE_ALIASES.get(m_inv.group(1).upper(), m_inv.group(1).upper())
                inv_args = [a.strip() for a in m_inv.group(2).replace(',', ' ').split()]
                info = self._gate_info(inner_name)
                if info is None:
                    raise ValueError(f"UNKNOWN GATE: {inner_name}")
                n_params, n_qubits_needed = info
                params = [self.eval_expr(a) for a in inv_args[:n_params]]
                qubits = [self._resolve_qubit(a, n_qubits=n_reg) for a in inv_args[n_params:n_params+n_qubits_needed]]
                matrix = _np_gate_matrix(inner_name, tuple(params)).conj().T
                self.locc.apply_matrix(reg, matrix, qubits)
                continue

            # Standard gate — through backend
            info = self._gate_info(gate_name)
            if info is None:
                raise ValueError(f"UNKNOWN GATE: {gate_name}")
            n_params, n_qubits_needed = info
            args = tokens[1:]
            if len(args) < n_params + n_qubits_needed:
                raise ValueError(f"{gate_name} needs {n_params} params + "
                                 f"{n_qubits_needed} qubits")
            params = tuple(self.eval_expr(a) for a in args[:n_params])
            qubits = [self._resolve_qubit(a, n_qubits=n_reg) for a in args[n_params:n_params+n_qubits_needed]]
            backend.apply_gate(gate_name, params, qubits)
