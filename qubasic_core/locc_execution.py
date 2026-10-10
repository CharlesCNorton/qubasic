"""QUBASIC LOCC execution mixin — program execution in LOCC mode."""

import re
import sys
import time

import numpy as np

from qubasic_core.engine import (
    ExecResult,
    GATE_TABLE, GATE_ALIASES,
    _np_gate_matrix, _sample_one_np,
    LOCC_SEND_SHOT_CAP, LOCC_SEND_QUBIT_THRESHOLD,
    RE_CTRL, RE_INV,
)


class LOCCExecutionMixin:
    """LOCC execution methods for QBasicTerminal.

    Requires: TerminalProtocol — uses self.locc, self.program, self.shots,
    self.variables, self._max_iterations, self._custom_gates,
    self.eval_expr(), self._eval_with_vars(), self._expand_statement(),
    self._tokenize_gate(), self._gate_info(), self._resolve_qubit(),
    self._parse_syndrome(), self._split_colon_stmts(), self._exec_control_flow(),
    self._get_parsed(), self._gosub_stack, self._locc_display_results(),
    self.last_counts, self.io.
    """

    def _locc_run(self) -> None:
        """Execute program in LOCC mode (N registers)."""
        sorted_lines = sorted(self.program.keys())
        if not sorted_lines:
            self.io.writeln("NOTHING TO RUN")
            return
        has_measure = self._program_has_measure(sorted_lines)
        has_send = any(re.search(r'\bSEND\b', self.program[l], re.IGNORECASE)
                       for l in sorted_lines)
        if has_send:
            self._locc_run_with_send(sorted_lines, has_measure)
        else:
            self._locc_run_vectorized(sorted_lines, has_measure)
        # Sync last_sv from LOCC engine so EXPECT/DENSITY/BLOCH work
        self.last_sv = self._active_sv

    def _locc_run_with_send(self, sorted_lines: list[int], has_measure: bool) -> None:
        """LOCC execution with SEND — prefix/suffix split optimization.

        Executes the deterministic prefix (before first SEND) once,
        snapshots the quantum state, then re-executes only the suffix
        (from first SEND onward) per shot. Falls back to full re-execution
        if the prefix contains jumps that could skip past the SEND.
        """
        from qubasic_core.scope import Scope
        from qubasic_core.statements import (
            SendStmt, GotoStmt, GosubStmt, OnGotoStmt, OnGosubStmt,
            ForStmt, WhileStmt, DoStmt, IfThenStmt,
        )

        # Any control transfer in the prefix could skip the SEND or re-enter,
        # so the prefix/suffix split is unsafe and we fall back to full re-exec.
        _jump_types = (GotoStmt, GosubStmt, OnGotoStmt, OnGosubStmt,
                       ForStmt, WhileStmt, DoStmt)
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
                if isinstance(p, _jump_types):
                    has_prefix_jumps = True
                    break
                if isinstance(p, IfThenStmt):
                    _clauses = f"{p.then_clause or ''} {p.else_clause or ''}"
                    if re.search(r'\b(GOTO|GOSUB)\b', _clauses, re.IGNORECASE):
                        has_prefix_jumps = True
                        break

        sizes_str = '+'.join(str(s) for s in self.locc.sizes)
        mode = "JOINT" if self.locc.joint else "SPLIT"
        shots = self.shots
        max_q = max(self.locc.sizes)
        # Cap overridable via the LOCC SHOTCAP command.
        user_cap = getattr(self, '_locc_shot_cap', None)
        effective_cap = user_cap if user_cap else LOCC_SEND_SHOT_CAP
        if max_q > LOCC_SEND_QUBIT_THRESHOLD and shots > effective_cap and not user_cap:
            self.io.writeln(f"  WARNING: capping at {effective_cap} shots for "
                            f"{max_q}-qubit LOCC w/ SEND (override: LOCC SHOTCAP <n>)")
            shots = effective_cap

        # Execute deterministic prefix once (skip if jumps make it unsafe, or
        # when noise is active: each shot needs its own noise draws)
        self.locc.reset()
        noisy = getattr(self.locc, 'noise_param', 0.0) > 0
        if has_prefix_jumps or noisy or first_send_ip == 0 or first_send_ip is None:
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

        per_reg: dict[str, dict[str, int]] = {name: {} for name in self.locc.names}
        counts_joint: dict[str, int] = {}
        t0 = time.time()
        progress_interval = max(1, shots // 10)
        # Only the first shot prints; later shots replay its INPUT answers.
        from qubasic_core.live import ShotIO
        real_io = self.io
        shot_io = ShotIO(real_io)
        first_vars: dict = {}

        for shot in range(shots):
            # Restore to post-prefix state
            self.locc.restore(snap)
            self.variables.clear()
            self.variables.update(snap_vars)
            if shot:
                shot_io.start_replay()
            self.io = shot_io
            try:
                self._locc_execute_program(sorted_lines, start_ip=send_ip,
                                           run_vars=Scope(self.variables))
            except (RuntimeError, ValueError) as e:
                self.io = real_io
                self.io.writeln(f"?RUNTIME ERROR: {e}")
                return
            finally:
                self.io = real_io
            if shot == 0:
                first_vars = dict(self.variables)
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
                from qubasic_core.qol import quantum_spin
                spin = quantum_spin(shot)
                _prog = f"  {spin} {pct}% ({shot+1}/{shots} shots)..."
                if hasattr(self.io, 'write') and sys.stdout.isatty():
                    self.io.write(_prog + '\r')
                # Non-terminal: skip progress to avoid \r noise

        if shots > 50 and sys.stdout.isatty():
            self.io.write(" " * 40 + '\r')
        # Variables read as the first (printed) shot left them.
        self.variables.clear()
        self.variables.update(first_vars)
        dt = time.time() - t0
        self.io.writeln(f"\nRAN {len(self.program)} lines, LOCC {mode} "
                        f"{sizes_str}q, {shots} shots in {dt:.2f}s")
        if has_measure:
            self._locc_display_results(per_reg, counts_joint)
            self.last_counts = counts_joint

    def _locc_run_vectorized(self, sorted_lines: list[int], has_measure: bool) -> None:
        """LOCC execution without SEND — single execution, vectorized sampling.

        When noise is active, re-executes per shot so that Monte Carlo noise
        fires independently each time (matching the SEND path behavior).
        """
        sizes_str = '+'.join(str(s) for s in self.locc.sizes)
        mode = "JOINT" if self.locc.joint else "SPLIT"
        t0 = time.time()

        noisy = getattr(self.locc, 'noise_param', 0.0) > 0

        if noisy and has_measure:
            # Per-shot execution so noise fires independently each time; only
            # the first shot prints.
            from qubasic_core.live import ShotIO
            per_reg: dict[str, dict[str, int]] = {name: {} for name in self.locc.names}
            counts_joint: dict[str, int] = {}
            real_io = self.io
            shot_io = ShotIO(real_io)
            base_vars = dict(self.variables)
            first_vars: dict = {}
            for _shot in range(self.shots):
                self.locc.reset()
                self.variables.clear()
                self.variables.update(base_vars)
                if _shot:
                    shot_io.start_replay()
                self.io = shot_io
                try:
                    self._locc_execute_program(sorted_lines)
                except (RuntimeError, ValueError) as e:
                    self.io = real_io
                    self.io.writeln(f"?RUNTIME ERROR: {e}")
                    return
                finally:
                    self.io = real_io
                if _shot == 0:
                    first_vars = dict(self.variables)
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
            self.variables.clear()
            self.variables.update(first_vars)
            dt = time.time() - t0
            self.io.writeln(f"\nRAN {len(self.program)} lines, LOCC {mode} "
                            f"{sizes_str}q, {self.shots} shots in {dt:.2f}s")
            self._locc_display_results(per_reg, counts_joint)
            self.last_counts = counts_joint if counts_joint else per_reg.get('A', {})
            return

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

    def _locc_execute_program(self, sorted_lines, *, start_ip: int = 0,
                              run_vars=None, stop_before_send: bool = False) -> int:
        """Execute LOCC program lines.

        Returns the ip where execution stopped (end of program or at first SEND
        if stop_before_send is True).
        """
        from qubasic_core.scope import Scope
        from qubasic_core.exec_context import ExecContext
        from qubasic_core.statements import MeasureStmt, SendStmt
        if run_vars is None:
            run_vars = Scope(self.variables)
        ctx = ExecContext(
            sorted_lines=sorted_lines, ip=start_ip, run_vars=run_vars,
            max_iterations=self._max_iterations,
        )
        _parsed = [self._get_parsed(ln) for ln in sorted_lines]
        _stmts = [self.program[ln].strip() for ln in sorted_lines]
        n_lines = len(sorted_lines)
        prev_strict, self._strict_names = self._strict_names, False
        try:
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
                    result = self._locc_exec_line(_stmts[ctx.ip], ctx.loop_stack, sorted_lines,
                                                  ctx.ip, run_vars, parsed=_parsed[ctx.ip], ctx=ctx)
                except Exception as e:
                    raise RuntimeError(f"LINE {sorted_lines[ctx.ip]}: {e}") from None
                if result is ExecResult.END:
                    break
                self._advance(ctx, result)
        finally:
            self._strict_names = prev_strict
        return ctx.ip

    def _locc_exec_line(self, stmt, loop_stack, sorted_lines, ip, run_vars, *, parsed=None,
                        ctx=None):
        """Execute one line in LOCC mode (ctx: the executor loop's context,
        for a whole line; absent for a compound's parts and IF clauses)."""
        from qubasic_core.parser import parse_stmt
        from qubasic_core.statements import (
            SendStmt, ShareStmt, AtRegStmt, CompoundStmt,
            RemStmt, MeasureStmt, EndStmt, BarrierStmt, ReturnStmt,
        )
        if parsed is None:
            parsed = parse_stmt(stmt)

        def _part(sub):
            return self._locc_exec_line(sub, loop_stack, sorted_lines, ip, run_vars)

        if ctx is not None and ctx.resume is not None:
            rp, ctx.resume = ctx.resume, None
            if int(rp) == ip:
                resumed = self._resume_compound(rp, parsed, loop_stack, ip, _part)
                if resumed is not None:
                    return resumed

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
            frame = ('line', ip) if ctx is not None else ('clause', ip, stmt)
            return self._run_compound(list(parsed.parts), _part, loop_stack, ip, frame)

        # Fast-path for LOCC-specific statements
        if isinstance(parsed, SendStmt):
            qubit = int(self.eval_expr(parsed.qubit_expr))
            outcome = self.locc.send(parsed.reg, qubit)
            run_vars[parsed.var] = outcome
            self.variables[parsed.var] = outcome
            self.locc.classical[parsed.var] = outcome
            self.locc.correction_log.append(
                f"SEND {parsed.reg}[{qubit}] -> {parsed.var}={outcome}")
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
            lambda s, ls, sl, i, rv: self._locc_exec_line(s, ls, sl, i, rv),
            parsed=parsed)
        if handled:
            return result

        if ':' in stmt:
            for sub in self._split_colon_stmts(stmt):
                self._locc_exec_line(sub, loop_stack, sorted_lines, ip, run_vars)
            return ExecResult.ADVANCE

        raise ValueError(f"LOCC mode requires @A/@B prefix, SEND, SHARE, or IF: {stmt}")

    def _locc_try_special(self, reg: str, stmt: str, run_vars: dict) -> bool:
        """Handle MEAS/RESET/MEASURE_X/Y/Z/SYNDROME in LOCC mode."""
        from qubasic_core.parser import parse_stmt
        from qubasic_core.statements import MeasStmt, ResetStmt, MeasureBasisStmt, SyndromeStmt
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

    def _locc_apply_gate(self, reg: str, gate_stmt: str) -> None:
        """Parse and apply a gate to a LOCC register via numpy.

        Uses LOCCRegBackend for standard gates. CTRL and INV modifiers
        use apply_matrix directly since they need matrix construction.
        """
        from qubasic_core.backend import LOCCRegBackend
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

            # CTRL modifier: one control on any built-in or UNITARY gate
            m_ctrl = RE_CTRL.match(stmt)
            if m_ctrl:
                canonical, params, qubits = self._modifier_operands(
                    'CTRL', m_ctrl.group(1), m_ctrl.group(2), limit=n_reg)
                inner, targets = self._np_operand(canonical, params, qubits[1:])
                dim = inner.shape[0]
                controlled = np.eye(2 * dim, dtype=complex)
                controlled[dim:, dim:] = inner
                self.locc.apply_matrix(reg, controlled, [qubits[0]] + targets)
                continue

            # INV modifier
            m_inv = RE_INV.match(stmt)
            if m_inv:
                canonical, params, qubits = self._modifier_operands(
                    'INV', m_inv.group(1), m_inv.group(2), limit=n_reg)
                inner, targets = self._np_operand(canonical, params, qubits)
                self.locc.apply_matrix(reg, inner.conj().T, targets)
                continue

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
            if gate_name in self._custom_gates:
                matrix, targets = self._np_operand(gate_name, params, qubits)
                self.locc.apply_matrix(reg, matrix, targets)
            else:
                backend.apply_gate(gate_name, params, qubits)

    def _np_operand(self, canonical: str, params, qubits: list[int]) -> tuple[np.ndarray, list[int]]:
        """(matrix, qubits) in the numpy engine's MSB-first convention.

        Built-in gate matrices are MSB-first over the listed qubits already. A
        UNITARY matrix follows Qiskit's order, where the first listed qubit is
        the least significant bit of its index, so its qubit list is reversed.
        """
        if canonical in self._custom_gates:
            return np.asarray(self._custom_gates[canonical]), list(reversed(qubits))
        return _np_gate_matrix(canonical, tuple(params)), list(qubits)
