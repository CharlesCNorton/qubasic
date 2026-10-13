"""QUBASIC circuit building: the executor that turns a program into a Qiskit
circuit a line at a time (or applies it to a live shot's state through the
backend it is given), and the quantum statements it runs: mid-circuit and
basis measurements, RESET, SYNDROME, the SAVE_* recorders, SET_STATE and
SET_DENSITY, APPLY_CIRCUIT, POKE and SYS, EVOLVE and APPLYCHANNEL.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit, transpile

from qubasic_core.engine import (
    MAX_CIRCUIT_INSTRUCTIONS,
    ExecResult,
    QBasicBuildError,
    QBasicRangeError,
)
from qubasic_core.host import SimulationHost
from qubasic_core.parser import (
    GATE_ALIASES,
    GATE_TABLE,
    RE_AT_REG_LINE,
    RE_CTRL,
    RE_INV,
    RE_MEAS,
    RE_MEASURE_BASIS,
    RE_REG_INDEX,
    RE_RESET,
    RE_SYNDROME,
    CompiledProgram,
)

if TYPE_CHECKING:
    from collections.abc import Container, Iterable

    from qiskit import ClassicalRegister
    from qiskit.circuit import Gate

    from qubasic_core.engine import CircuitBackend, ExecContext, ExecOutcome
    from qubasic_core.parser import IfJumpStmt, Stmt


# ---------------------------------------------------------------------------
# QUBASIC executor mixin — circuit building and line execution.
# ---------------------------------------------------------------------------

def _feedforward_test(condition: str, names: Container[str]) -> tuple[str, int] | None:
    """(bit name, value it must read) when an IF condition tests one
    measurement bit in a form a dynamic circuit can branch on, else None."""
    s = condition.strip()
    m = re.match(r'NOT\s+(\w+)$', s, re.IGNORECASE)
    if m and m.group(1) in names:
        return m.group(1), 0
    if re.match(r'\w+$', s) and s in names:
        return s, 1
    m = re.match(r'(\w+)\s*(==|=|!=|<>)\s*([01])$', s)
    if m and m.group(1) in names:
        val = int(m.group(3))
        return m.group(1), (val if m.group(2) in ('==', '=') else 1 - val)
    return None


class ExecutorMixin(SimulationHost):
    """Circuit building and per-line execution for QBasicTerminal.

    Provides the core execution pipeline: circuit compilation from stored
    programs, single-line execution, gate tokenization, qubit resolution,
    subroutine expansion, and immediate-mode gate application.

    All methods access shared terminal state (variables, arrays, subroutines,
    registers, etc.) through ``self``.
    """

    # Gate dispatch: name -> (method_name, arg_pattern)
    # arg_pattern: 'q' = qubits only, 'pq' = params then qubits, 'ppq' = 3-param then qubit
    _GATE_DISPATCH = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'T': 't', 'SDG': 'sdg', 'TDG': 'tdg',
        'SX': 'sx', 'SXDG': 'sxdg', 'ID': 'id',
        'CX': 'cx', 'CZ': 'cz', 'CY': 'cy', 'CH': 'ch',
        'SWAP': 'swap', 'DCX': 'dcx', 'ISWAP': 'iswap',
        'ECR': 'ecr', 'CS': 'cs', 'CSDG': 'csdg', 'CSX': 'csx',
        'CCX': 'ccx', 'CSWAP': 'cswap', 'CCZ': 'ccz',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz', 'P': 'p',
        'CRX': 'crx', 'CRY': 'cry', 'CRZ': 'crz', 'CP': 'cp',
        'RXX': 'rxx', 'RYY': 'ryy', 'RZZ': 'rzz', 'RZX': 'rzx',
        'U': 'u',
    }

    # ── Circuit Building ──────────────────────────────────────────────

    def _program_has_measure(self, sorted_lines: Iterable[int]) -> bool:
        """True if any reachable statement measures.

        The plain per-line scan misses a MEASURE that only appears inside a DEF
        subroutine body or an IF/THEN-ELSE clause, which left the run on the
        no-measure path with no counts. This recursion-guarded walk follows
        colon compounds, IF clauses, and subroutine bodies so the shots path
        fires whenever a measurement is actually reachable.
        """
        from qubasic_core.parser import CompoundStmt, IfThenStmt, MeasureStmt, parse_stmt

        def scan(text: str, seen: frozenset) -> bool:
            p = parse_stmt(text)
            if isinstance(p, MeasureStmt):
                return True
            if isinstance(p, CompoundStmt):
                return any(scan(part, seen) for part in p.parts)
            if isinstance(p, IfThenStmt):
                if p.then_clause and scan(p.then_clause, seen):
                    return True
                if p.else_clause and scan(p.else_clause, seen):
                    return True
                return False
            words = text.strip().split()
            if words:
                word = words[0].upper().split('(')[0]
                if word in self.subroutines and word not in seen:
                    sub = self.subroutines[word]
                    body = sub['body'] if isinstance(sub, dict) else sub
                    return any(scan(b, seen | {word}) for b in body)
            return False

        return any(scan(self.program[ln], frozenset()) for ln in sorted_lines)

    def _prepare_run(self, qc: QuantumCircuit) -> None:
        """Reset per-run state and emit pending state preparation into qc."""
        # Fresh mid-circuit measurement registry for this build (dynamic IF).
        self._classical_bits: dict[str, Any] = {}
        # Partial-measurement subset (None = measure all at the end).
        self._measure_subset: list[int] | None = None
        # Density matrix (and circuit) captured by a SET_DENSITY run; the
        # matrix is solved lazily by DENSITY from the stored circuit.
        self._last_density = None
        self._last_density_qc = None
        # Apply any qubit state preparation requested via POKE to $0100.
        if getattr(self, '_poke_state_prep', None):
            self._emit_poke_state_prep(qc)
        # Apply a pending immediate SET_STATE so it persists into this RUN.
        _pend = getattr(self, '_pending_set_state', None)
        if _pend is not None and len(_pend) == 2 ** self.num_qubits:
            from qiskit.quantum_info import Statevector
            from qiskit_aer.library import SetStatevector
            qc.append(SetStatevector(Statevector(_pend)), list(range(self.num_qubits)))
        # Inject a pending mixed state (density matrix) — needs density_matrix method.
        _pend_d = getattr(self, '_pending_set_density', None)
        if _pend_d is not None and _pend_d.shape[0] == 2 ** self.num_qubits:
            from qiskit.quantum_info import DensityMatrix
            from qiskit_aer.library import SetDensityMatrix
            qc.append(SetDensityMatrix(DensityMatrix(_pend_d)), list(range(self.num_qubits)))
        self.events.measure_fired = False
        # Error trapping arms when this run executes ON ERROR GOTO, never from
        # a previous run's state.
        self.errors.target = None
        self.errors.handling = False
        self._reset_events()
        # Declarations (DIM AS, CONST, DIM SHARED) start over; DEFtype ranges
        # apply to the whole program.
        self._reset_declarations(self._compiled())

    def build_circuit(self, backend: CircuitBackend | None = None, *, start_ip: int = 0,
                      stop_ip: int | None = None) -> tuple['QuantumCircuit', bool]:
        """Compile program lines into a QuantumCircuit. Returns (circuit, has_measure).

        With a live backend (see qubasic_core.simulator) the lines execute against
        its per-shot state instead, from line index start_ip up to stop_ip."""
        from qubasic_core.engine import ExecContext
        from qubasic_core.expression import Scope
        from qubasic_core.simtools import QiskitBackend

        live = backend is not None
        qc = backend.qc if backend is not None else QuantumCircuit(self.num_qubits)
        if start_ip == 0:
            self._prepare_run(qc)
        if backend is None:
            backend = QiskitBackend(qc, self._apply_gate)
        prog = self._compiled()
        ctx = ExecContext(
            sorted_lines=prog,
            ip=start_ip,
            run_vars=Scope(self.variables),
            max_iterations=self._loop_limit(),
            qc=qc,
            backend=backend,
        )
        # Reachability scan catches MEASURE inside subroutines and IF clauses,
        # not just literal top-level MEASURE lines.
        has_measure = False if live else self._program_has_measure(sorted(self.program))
        # The circuit under construction, so PRINT STATE / QUBIT(q) inside a
        # program can read the state at that line (a live run reads its own).
        self._build_qc = None if live else qc
        try:
            self._build_loop(ctx, qc, stop_ip)
        finally:
            self._build_qc = None
        return qc, has_measure or bool(self._build_saw_measure)

    def _build_loop(self, ctx: ExecContext, qc: QuantumCircuit, stop_ip: int | None = None) -> None:
        """Execute program lines into qc (the body of build_circuit)."""
        prev, self._strict_names = self._strict_names, False
        try:
            self._build_lines(ctx, qc, stop_ip)
        finally:
            self._strict_names = prev

    def _build_lines(self, ctx: ExecContext, qc: QuantumCircuit, stop_ip: int | None) -> None:
        from qubasic_core.parser import MeasureStmt, line_index
        handler_lines_run = 0
        self._build_saw_measure = False
        prog = ctx.sorted_lines
        end = len(prog) if stop_ip is None else stop_ip
        while ctx.ip < end:
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            line_num = prog.lines[ctx.ip]
            stmt = prog.texts[ctx.ip]
            parsed = prog.stmts[ctx.ip]
            first = prog.first[ctx.ip]

            # Per-activation budget for ON ERROR handlers (RESUME resets it via
            # _in_error_handler), so a handler that never resumes cannot spin.
            if self.errors.handling:
                handler_lines_run += 1
                if handler_lines_run > 200:
                    self.io.writeln("?ERROR HANDLER LIMIT (200 lines)")
                    break
            else:
                handler_lines_run = 0

            # Debug instrumentation (line-level), each a no-op unless enabled.
            if first:
                if self.debug.trace:
                    self._trace_line(line_num)
                if self.debug.breakpoints and self._check_breakpoint(line_num, prog, ctx.ip):
                    break
            if self._events_armed():
                _event_tgt = self._check_events(prog, ctx.ip)
                if _event_tgt is not None:
                    ctx.ip = _event_tgt
                    continue

            if isinstance(parsed, MeasureStmt):
                self._build_saw_measure = True
                if parsed.qubits:
                    self._measure_subset = [
                        self._resolve_qubit(q)
                        for q in parsed.qubits.replace(',', ' ').split()]
                if self.events.measure_target is not None and not self.events.measure_fired:
                    self.events.measure_fired = True
                    target = line_index(prog, self.events.measure_target)
                    if target is not None:
                        self._gosub_stack.append(ctx.ip + 1)
                        ctx.ip = target
                        continue
                ctx.ip += 1
                continue

            if self.graphics.canvas is not None:
                self._gfx_frame()
            self._profile_line_start(line_num)
            gates_before = len(qc.data)
            if gates_before > MAX_CIRCUIT_INSTRUCTIONS:
                raise RuntimeError(f"CIRCUIT LIMIT ({MAX_CIRCUIT_INSTRUCTIONS:,} instructions)")
            try:
                result = self._exec_line(stmt, parsed=parsed, ctx=ctx)
            except QBasicBuildError as e:
                if self._live_needed:
                    raise
                _handler_ip = self._handle_error(e, line_num, ctx.sorted_lines, ctx.ip)
                if _handler_ip is None:
                    raise QBasicBuildError(
                        f"LINE {line_num}: {e}"
                    ) from None
                ctx.ip = _handler_ip
                continue
            except Exception as e:
                # A measurement bit read as a number makes the run go live;
                # no ON ERROR handler may take that.
                if self._live_needed:
                    raise
                _handler_ip = self._handle_error(e, line_num, ctx.sorted_lines, ctx.ip)
                if _handler_ip is None:
                    raise RuntimeError(f"LINE {line_num}: {e}") from None
                ctx.ip = _handler_ip
                continue
            finally:
                self._profile_line_end(line_num, len(qc.data) - gates_before)

            if result is ExecResult.END:
                break
            self._advance(ctx, result)

    @staticmethod
    def _advance(ctx: ExecContext, result: ExecOutcome) -> None:
        """Move ctx to a jump target, or to the next statement."""
        if isinstance(result, int) and not isinstance(result, bool):
            ctx.ip = result
        else:
            ctx.ip += 1


    def _exec_line(self, stmt: str, qc: QuantumCircuit | None = None, loop_stack: list[dict[str, Any]] | None = None, sorted_lines: Sequence[int] | None = None,
                   ip: int = 0, run_vars: MutableMapping[str, Any] | None = None, parsed: Stmt | None = None, *, ctx: ExecContext | None = None) -> ExecOutcome:
        """Execute one statement of a compiled program (see parser.compile_program).

        Accepts either individual parameters or ctx (ExecContext). When ctx is
        provided, qc/loop_stack/sorted_lines/ip/run_vars are read from ctx and
        the individual params are ignored.

        Evaluation order (deterministic, first match wins):
          1. Typed fast-path (parsed Stmt): BARRIER, REM, MEASURE, END, RETURN, @REG
          2. Feedforward: an IF on a mid-circuit measurement bit
          3. Control flow: LET, PRINT, GOTO, GOSUB, FOR/NEXT, WHILE/WEND, IF,
             DATA/READ, ON GOTO/GOSUB, SELECT CASE, DO/LOOP, EXIT, SUB/FUNCTION,
             ON ERROR, ASSERT, STOP, SWAP, DEF FN, OPTION BASE
          4. Keyword statements: MEAS, RESET, MEASURE_X/Y/Z, SYNDROME, UNITARY,
             DIM, REDIM, ERASE, GET, INPUT, POKE, SYS, file I/O, PRINT USING, ...
          5. Gate application (subroutine expansion + gate dispatch)

        Returns: int (jump target ip), ExecResult.ADVANCE, or ExecResult.END.
        """
        if ctx is not None:
            qc = ctx.backend.qc if ctx.backend and hasattr(ctx.backend, 'qc') else ctx.qc
            loop_stack = ctx.loop_stack
            sorted_lines = ctx.sorted_lines
            ip = ctx.ip
            run_vars = ctx.run_vars
        if loop_stack is None or sorted_lines is None or run_vars is None:
            raise TypeError("_exec_line needs ctx, or loop_stack, sorted_lines and run_vars")
        from qubasic_core.parser import (
            AtRegStmt,
            BarrierStmt,
            EndStmt,
            GateStmt,
            IfJumpStmt,
            MeasureStmt,
            RemStmt,
            ReturnStmt,
        )

        # 1. Typed fast-path: terminals that don't need control-flow dispatch
        if parsed is None:
            from qubasic_core.parser import parse_stmt
            parsed = parse_stmt(stmt)
        if isinstance(parsed, BarrierStmt):
            if qc is not None and hasattr(qc, 'barrier'):
                qc.barrier()
            return ExecResult.ADVANCE
        if isinstance(parsed, (RemStmt, MeasureStmt)):
            return ExecResult.ADVANCE
        if isinstance(parsed, EndStmt):
            return ExecResult.END
        if isinstance(parsed, ReturnStmt):
            return self._cf_return(sorted_lines, parsed)[1]
        if isinstance(parsed, AtRegStmt) and not self.locc_mode:
            raise ValueError("@register syntax requires LOCC mode (try: LOCC <n1> <n2>)")

        # 2. Dynamic circuit: an IF on a mid-circuit measurement bit (set by
        #    MEAS) compiles to a Qiskit if_test, so the conditional gates run at
        #    simulation time on the real measured value (feedforward).
        if (isinstance(parsed, IfJumpStmt) and parsed.then_end >= 0 and qc is not None
                and isinstance(sorted_lines, CompiledProgram)
                and hasattr(qc, 'if_test') and getattr(self, '_classical_bits', None)):
            tgt = self._classical_if_target(parsed.condition)
            if tgt is not None:
                self._feedforward_if(qc, tgt, parsed, loop_stack, sorted_lines, run_vars)
                return parsed.end

        # 3. Control-flow dispatch (GOTO, GOSUB, FOR/NEXT, WHILE/WEND, IF, LET,
        #    PRINT, DATA/READ, ON GOTO, SELECT CASE, DO/LOOP, EXIT, SUB/FUNCTION,
        #    ON ERROR, ...)
        handled, result = self._exec_control_flow(
            stmt, loop_stack, sorted_lines, ip, run_vars,
            lambda s, ls, sl, i, rv: self._exec_line(
                s, qc=qc, loop_stack=ls, sorted_lines=sl, ip=i, run_vars=rv),
            parsed=parsed)
        if handled:
            assert result is not None          # a handled statement has an outcome
            return result

        from qubasic_core.parser import RestoreStmt
        if isinstance(parsed, RestoreStmt):
            if isinstance(parsed.target, str):      # labels resolve when the program compiles
                raise ValueError(f"LABEL {parsed.target} NOT FOUND")
            self._restore_to(parsed.target)
            return ExecResult.ADVANCE

        # 4. Keyword statements
        _backend = ctx.backend if ctx else None
        if self._try_stmt_handlers(stmt, qc, run_vars, backend=_backend):
            return ExecResult.ADVANCE

        # 5. Gate application

        # Fast path: GateStmt already parsed by the parser
        if isinstance(parsed, GateStmt):
            info = self._gate_info(parsed.name)
            if info is not None:
                n_params, n_qubits = info
                args = list(parsed.args)
                params = [self._eval_operand(a, run_vars) for a in args[:n_params]]
                qubits = [self._resolve_qubit(a) for a in args[n_params:n_params + n_qubits]]
                try:
                    if _backend:
                        _backend.apply_gate(parsed.name, tuple(params), qubits)
                    else:
                        self._apply_gate(qc, parsed.name, params, qubits)
                except Exception as _gate_err:
                    if 'duplicate' in str(_gate_err).lower():
                        raise QBasicBuildError(
                            f"duplicate qubit arguments in {parsed.name}"
                        ) from None
                    raise
                return ExecResult.ADVANCE
            # Fall through to _apply_gate_str for custom gates not in GATE_TABLE

        # A REPL/config command (QUBITS, SHOTS, METHOD, LOCC, ...) used as a
        # numbered program line reaches here only because it is neither a gate
        # nor a statement. Explain it instead of failing with "UNKNOWN GATE".
        # SEND/SHARE are LOCC statements (handled on the LOCC path), so leave
        # them to their own error.
        _first = stmt.split(None, 1)[0].upper() if stmt.split() else ''
        if self._is_command(_first) and _first not in ('SEND', 'SHARE'):
            raise QBasicBuildError(
                f"{_first} is a configuration/session command, not a program "
                f"statement; run it without a line number (before RUN), e.g. "
                f"QUBITS 3 / SHOTS 1024 / METHOD statevector / LOCC 2 2")

        # Slow path: subroutine expansion + gate dispatch
        expanded = self._expand_statement(stmt)
        for gate_str in expanded:
            self._apply_gate_str(gate_str, qc, backend=_backend)

        return ExecResult.ADVANCE

    def _classical_if_target(self, condition: str) -> tuple[ClassicalRegister, int] | None:
        """If `condition` tests a mid-circuit measurement bit, return
        (classical_register, target_value); else None.

        Handled forms: ``c``, ``NOT c``, ``c = 0|1``, ``c <> 0|1`` (and the
        ``==`` / ``!=`` spellings).
        """
        cb = getattr(self, '_classical_bits', None)
        if not cb:
            return None
        hit = _feedforward_test(condition, cb)
        if hit is None:
            return None
        name, val = hit
        return cb[name], val

    def _feedforward_if(self, qc: QuantumCircuit, tgt: tuple[ClassicalRegister, int], parsed: IfJumpStmt, loop_stack: list[dict[str, Any]], prog: CompiledProgram, run_vars: MutableMapping[str, Any]) -> None:
        """Emit a Qiskit if_test (with else) for an IF on a measured bit: the
        compiled THEN and ELSE statements run inside the conditional blocks,
        so their gates apply at simulation time on the real outcome."""
        creg, val = tgt

        def _run(start: int, end: int) -> None:
            for i in range(start, end):
                self._exec_line(prog.texts[i], qc=qc, loop_stack=loop_stack,
                                sorted_lines=prog, ip=i, run_vars=run_vars,
                                parsed=prog.stmts[i])

        if parsed.else_end > parsed.else_start:
            with qc.if_test((creg, val)) as else_block:
                _run(parsed.then_start, parsed.then_end)
            with else_block:
                _run(parsed.else_start, parsed.else_end)
        else:
            with qc.if_test((creg, val)):
                _run(parsed.then_start, parsed.then_end)

    def _parse_syndrome(self, stmt: str, run_vars: MutableMapping[str, Any]) -> tuple[str, list[int], str] | None:
        """Parse SYNDROME statement. Returns (pauli_str, qubits, var) or None."""
        m = RE_SYNDROME.match(stmt)
        if not m:
            return None
        rest = m.group(1).strip()
        parts = rest.split('->')
        if len(parts) != 2:
            raise ValueError("SYNDROME syntax: SYNDROME <paulis> <qubits> -> <var>")
        var = parts[1].strip()
        tokens = parts[0].split()
        if len(tokens) < 2:
            raise ValueError("SYNDROME needs a Pauli string and qubit list")
        pauli_str = tokens[0].upper()
        qubit_args = tokens[1:]
        if len(pauli_str) != len(qubit_args):
            raise ValueError(
                f"Pauli string length ({len(pauli_str)}) must match "
                f"qubit count ({len(qubit_args)})")
        for p in pauli_str:
            if p not in 'IXYZ':
                raise ValueError(f"Unknown Pauli: {p} (use I, X, Y, Z)")
        qubits = [int(self._eval_with_vars(q, run_vars)) for q in qubit_args]
        return pauli_str, qubits, var

    @staticmethod
    def _split_colon_stmts(stmt: str) -> list[str]:
        """Split colon-separated statements, inheriting @register prefixes."""
        from qubasic_core.parser import _split_colon_stmts
        return _split_colon_stmts(stmt)

    def _expand_statement(self, stmt: str, _call_stack: set[str] | None = None) -> list[str]:
        """Expand subroutines. Returns list of gate strings.

        Uses explicit call-stack tracking to detect recursion instead of
        an arbitrary depth counter.
        """
        if _call_stack is None:
            _call_stack = set()

        # Handle parenthesized subroutine calls: NAME(arg1, arg2) -> NAME arg1, arg2
        m_call = re.match(r'(\w+)\(([^)]*)\)', stmt)
        if m_call:
            call_name = m_call.group(1).upper()
            if call_name in self.subroutines:
                call_args = m_call.group(2)
                stmt = f"{call_name} {call_args}"

        parts = stmt.split()
        word = parts[0].upper() if parts else ''

        if word not in self.subroutines:
            return [stmt]

        if word in _call_stack:
            raise RuntimeError(f"RECURSION: {word} calls itself")
        _call_stack = _call_stack | {word}

        sub = self.subroutines[word]
        # Handle both legacy (list) and new (dict with params) format
        if isinstance(sub, list):
            body = sub
            param_names = []
        else:
            body = sub['body']
            param_names = sub['params']

        # Parse arguments: NAME arg1, arg2 @offset
        rest = stmt[len(word):].strip()
        offset = 0
        m_off = re.search(r'@(\d+)', rest)
        if m_off:
            offset = int(m_off.group(1))
            rest = rest[:m_off.start()].strip()

        # Parse call arguments
        call_args = [a.strip() for a in rest.split(',') if a.strip()] if rest else []

        # Build param map for single-pass substitution
        param_map = {}
        for i, pname in enumerate(param_names):
            if i < len(call_args):
                param_map[pname] = call_args[i]
        if param_map:
            pattern = re.compile(r'\b(' + '|'.join(re.escape(p) for p in param_map) + r')\b')
            def _sub(m: re.Match[str]) -> str:
                return param_map[m.group(1)]

        result = []
        for gate_str in body:
            gs = pattern.sub(_sub, gate_str) if param_map else gate_str
            if offset:
                gs = self._offset_qubits(gs, offset)
            result.append(gs)
        return result

    def _offset_qubits(self, gate_str: str, offset: int) -> str:
        """Add offset to qubit indices in a gate string, preserving parameters.

        Handles both plain integers and register[index] notation.
        """
        tokens = self._tokenize_gate(gate_str)
        if len(tokens) < 2:
            return gate_str
        gate_name = tokens[0].upper()
        gate_key = GATE_ALIASES.get(gate_name, gate_name)
        info = self._gate_info(gate_key)
        n_params = info[0] if info else 0
        result = [tokens[0]]
        for i, tok in enumerate(tokens[1:]):
            if i < n_params:
                result.append(tok)
            else:
                # Skip register[index] notation -- it resolves its own offset
                m = RE_REG_INDEX.match(tok)
                if m:
                    result.append(tok)
                else:
                    try:
                        result.append(str(int(tok) + offset))
                    except ValueError:
                        result.append(tok)
        return ' '.join(result)

    def _apply_gate_str(self, stmt: str, qc: QuantumCircuit, _call_stack: set[str] | None = None, *, backend: CircuitBackend | None = None) -> None:
        """Parse and apply a single gate string to the circuit.

        When backend is provided, standard gates are dispatched through
        backend.apply_gate() instead of self._apply_gate(qc, ...).
        """
        stmt = stmt.strip()
        if not stmt:
            return

        # Expand subroutines with call-stack tracking for recursion detection
        if _call_stack is None:
            _call_stack = set()
        word = stmt.split()[0].upper() if stmt.split() else ''
        # Also check for parenthesized call syntax: NAME(args)
        m_paren = re.match(r'(\w+)\(', stmt)
        if m_paren:
            paren_name = m_paren.group(1).upper()
            if paren_name in self.subroutines:
                word = paren_name
        if word in self.subroutines:
            if word in _call_stack:
                raise QBasicBuildError(f"RECURSION: {word} calls itself")
            for sub_stmt in self._expand_statement(stmt, _call_stack):
                self._apply_gate_str(sub_stmt, qc, _call_stack | {word}, backend=backend)
            return

        upper = stmt.upper()
        if upper.startswith('REM') or upper.startswith("'") or upper == 'BARRIER':
            if upper == 'BARRIER':
                if backend:
                    backend.barrier()
                else:
                    qc.barrier()
            return
        if upper == 'MEASURE':
            return  # handled at run level

        # CTRL <gate> [params,] control, target(s) -- one control on any gate
        m_ctrl = RE_CTRL.match(stmt)
        if m_ctrl:
            canonical, params, qubits = self._modifier_operands(
                'CTRL', m_ctrl.group(1), m_ctrl.group(2))
            gate = self._gate_object(canonical, params).control(1)
            if backend and hasattr(backend, 'append_controlled'):
                backend.append_controlled(gate, qubits)
            else:
                qc.append(gate, qubits)
            return

        # INV <gate> [params,] qubit(s) -- inverse (dagger) of one gate
        m_inv = RE_INV.match(stmt)
        if m_inv:
            canonical, params, qubits = self._modifier_operands(
                'INV', m_inv.group(1), m_inv.group(2))
            gate = self._gate_object(canonical, params).inverse()
            if backend and hasattr(backend, 'append_controlled'):
                backend.append_controlled(gate, qubits)
            else:
                qc.append(gate, qubits)
            return

        # Parse: GATE [params] qubits
        # Tokenize
        tokens = self._tokenize_gate(stmt)
        if not tokens:
            return

        gate_name = tokens[0].upper()
        gate_name = GATE_ALIASES.get(gate_name, gate_name)

        info = self._gate_info(gate_name)
        if info is None:
            raise ValueError(f"UNKNOWN GATE: {gate_name}")

        n_params, n_qubits = info
        args = tokens[1:]

        # Parse arguments: first n_params are parameters, rest are qubits
        if len(args) < n_params + n_qubits:
            raise ValueError(
                f"{gate_name} needs {n_params} param(s) and {n_qubits} qubit(s), "
                f"got {len(args)} arg(s)")

        params = [self._eval_operand(a) for a in args[:n_params]]
        qubits = [self._resolve_qubit(a) for a in args[n_params:n_params+n_qubits]]

        # Apply gate through backend when available
        try:
            if backend:
                backend.apply_gate(gate_name, tuple(params), qubits)
            else:
                self._apply_gate(qc, gate_name, params, qubits)
        except Exception as _gate_err:
            if 'duplicate' in str(_gate_err).lower():
                raise QBasicBuildError(
                    f"duplicate qubit arguments in {gate_name}"
                ) from None
            raise

    def _modifier_operands(self, keyword: str, gate_name: str, arg_text: str,
                           limit: int | None = None) -> tuple[str, list, list[int]]:
        """Operands of CTRL/INV: (gate, params, qubits). For CTRL the control
        qubit comes first in `qubits`, then the gate's own qubits."""
        canonical = GATE_ALIASES.get(gate_name.upper(), gate_name.upper())
        info = self._gate_info(canonical)
        if info is None:
            raise ValueError(f"{keyword} {gate_name.upper()}: gate not found")
        n_params, n_q = info
        args = self._tokenize_gate(f"{canonical} {arg_text}")[1:]
        controls = 1 if keyword == 'CTRL' else 0
        if len(args) != n_params + controls + n_q:
            ctl = "1 control and " if controls else ""
            raise ValueError(
                f"{keyword} {canonical} needs {n_params} parameter(s), {ctl}"
                f"{n_q} target qubit(s); got {len(args)} argument(s)")
        params = [self._eval_operand(a) for a in args[:n_params]]
        qubits = [self._resolve_qubit(a, n_qubits=limit) for a in args[n_params:]]
        if len(set(qubits)) != len(qubits):
            raise QBasicBuildError(f"duplicate qubit arguments in {keyword} {canonical}")
        return canonical, params, qubits

    def _gate_object(self, canonical: str, params: list) -> Gate:
        """A Qiskit gate instance for a built-in or UNITARY-defined gate."""
        if canonical in self._custom_gates:
            from qiskit.circuit.library import UnitaryGate
            return UnitaryGate(self._custom_gates[canonical], label=canonical)
        from qiskit.circuit.library import get_standard_gate_name_mapping
        lib = get_standard_gate_name_mapping().get(canonical.lower())
        if lib is None:
            raise ValueError(f"GATE {canonical} NOT IMPLEMENTED")
        return type(lib)(*params)

    def _tokenize_gate(self, stmt: str) -> list[str]:
        """Split gate statement into tokens, handling commas and register notation.

        When arguments are comma-separated, splits on commas only so that
        compound expressions like ``I + 1`` are preserved as single tokens.
        Falls back to whitespace splitting when no commas are present
        (legacy format like ``CX 0 1``).
        """
        stmt = RE_REG_INDEX.sub(r'\1[\2]', stmt)
        parts = stmt.strip().split(None, 1)
        if len(parts) < 2:
            return [parts[0]] if parts else []
        gate = parts[0]
        args_str = parts[1]
        if ',' in args_str:
            # Comma-separated: split on commas, preserving expressions
            args = [a.strip() for a in args_str.split(',') if a.strip()]
        else:
            # Space-separated: split on whitespace (legacy)
            args = [a.strip() for a in args_str.split() if a.strip()]
        return [gate] + args

    def _resolve_qubit(self, arg: str, *, n_qubits: int | None = None) -> int:
        """Resolve a qubit argument and validate range.

        Accepts: integer literal, register[index], or expression.
        Validates against n_qubits (defaults to self.num_qubits).
        """
        limit = n_qubits if n_qubits is not None else self.num_qubits
        m = RE_REG_INDEX.match(arg)
        if m:
            name = m.group(1).lower()
            idx = int(m.group(2))
            if name not in self.registers:
                raise QBasicRangeError(f"UNKNOWN REGISTER: {name}")
            start, size = self.registers[name]
            if idx >= size:
                raise QBasicRangeError(f"{name}[{idx}] OUT OF RANGE (size={size})")
            q = start + idx
        else:
            try:
                q = int(self._eval_operand(arg))
            except Exception:
                raise ValueError(f"CANNOT RESOLVE QUBIT: {arg}")
        if q < 0 or q >= limit:
            raise QBasicRangeError(f"QUBIT {q} OUT OF RANGE (0-{limit-1})")
        return q

    def _apply_gate(self, qc: QuantumCircuit, gate_name: str, params: list[float], qubits: list[int]) -> None:
        """Apply a gate to the quantum circuit."""
        n = qc.num_qubits
        for q in qubits:
            if q < 0 or q >= n:
                raise QBasicRangeError(
                    f"QUBIT {q} OUT OF RANGE (0-{n-1}) in {gate_name}"
                )
        method_name = self._GATE_DISPATCH.get(gate_name)
        if method_name:
            method = getattr(qc, method_name)
            method(*params, *qubits)
        elif gate_name in self._custom_gates:
            from qiskit.circuit.library import UnitaryGate
            qc.append(UnitaryGate(self._custom_gates[gate_name]), qubits)
        else:
            raise ValueError(f"GATE {gate_name} NOT IMPLEMENTED")

    def run_immediate(self, line: str) -> None:
        """Execute a single gate command immediately.

        Uses the same _exec_line pipeline as cmd_run for consistency.
        """
        # In LOCC mode, handle @register prefix via the numpy engine. Split
        # colon-compound @REG lines so each gate is applied (a single @REG line
        # is just the one-part case); a bare @REG line keeps prior behavior.
        if self.locc_mode and self.locc and line.strip().startswith('@'):
            parts = self._split_colon_stmts(line) if ':' in line else [line.strip()]
            for part in parts:
                m = RE_AT_REG_LINE.match(part)
                if not m:
                    self.io.writeln(f"?BAD @register statement: {part}")
                    continue
                reg = m.group(1).upper()
                gate_stmt = m.group(2).strip()
                if reg not in self.locc.names:
                    self.io.writeln(f"?UNKNOWN REGISTER: {reg} (have {', '.join(self.locc.names)})")
                    return
                self._locc_apply_gate(reg, gate_stmt)
            self._locc_state()
            return
        if line.strip().startswith('@'):
            self.io.writeln("?@register syntax requires LOCC mode (try: LOCC <n1> <n2>)")
            return
        # Compile the line as a one-line program and run it through the same
        # statement pipeline as RUN.
        from qubasic_core.engine import ExecContext
        from qubasic_core.parser import compile_program
        qc = QuantumCircuit(self.num_qubits)
        prog = compile_program({0: line})
        imm_ctx = ExecContext(sorted_lines=prog, ip=0,
                              run_vars=dict(self.variables), qc=qc)
        while imm_ctx.ip < len(prog):
            imm_ctx.iteration_count += 1
            if imm_ctx.iteration_count > self._loop_limit():
                raise RuntimeError(f"LOOP LIMIT ({self._max_iterations}) — possible infinite loop")
            result = self._exec_line(prog.texts[imm_ctx.ip], parsed=prog.stmts[imm_ctx.ip],
                                     ctx=imm_ctx)
            if result is ExecResult.END:
                break
            self._advance(imm_ctx, result)
        # A classical statement typed at the prompt (PRINT, MEASURE, an IF whose
        # clause printed, ...) adds no circuit operations and has already emitted
        # its own output via _exec_line. Simulating an empty circuit just to print
        # an unchanged |psi> would be spurious, so skip the statevector dump.
        if qc.size() == 0:
            return
        from qubasic_core import simtools as cap
        peak = cap.aer_peak_bytes(self.num_qubits, 'statevector', self.sim_precision, 'end')
        if not cap.fits(peak):
            self.last_circuit = qc
            self.io.writeln(f"  (applied; a {self.num_qubits}-qubit state needs "
                            f"~{cap.fmt_gb(peak)}, so it is not shown)")
            return
        qc.save_statevector(label='qb_state')
        backend = self._make_backend('statevector')
        result = backend.run(transpile(qc, backend), shots=1).result()
        sv = np.asarray(result.data()['qb_state'].data)
        self.last_sv = sv
        self.last_circuit = qc
        self._print_sv_compact(sv)

    def _gate_info(self, name: str) -> tuple[int, int] | None:
        """Look up (n_params, n_qubits) for a gate name.

        Checks the global GATE_TABLE first, then instance-local custom gates.
        Returns None if the gate is unknown.
        """
        if name in GATE_TABLE:
            return GATE_TABLE[name]
        if name in self._custom_gates:
            matrix = self._custom_gates[name]
            n_qubits = int(np.log2(matrix.shape[0]))
            return (0, n_qubits)
        return None


# ---------------------------------------------------------------------------
# Quantum statements: mid-circuit and basis measurements, RESET, SYNDROME, the
# SAVE_* recorders, SET_STATE and SET_DENSITY, APPLY_CIRCUIT, and POKE and SYS
# with EVOLVE and APPLYCHANNEL, registered in _QUANTUM_STATEMENTS.
# ---------------------------------------------------------------------------

def _resolve_named_state(name: str, n_qubits: int) -> np.ndarray:
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=complex)
    if name == '|0>':
        sv[0] = 1.0
    elif name == '|1>':
        if dim < 2:
            raise ValueError(f"State |1> requires at least 1 qubit (dim >= 2), got dim={dim}")
        sv[1] = 1.0
    elif name == '|+>':
        if dim < 2:
            raise ValueError(f"State |+> requires at least 1 qubit (dim >= 2), got dim={dim}")
        sv[0] = 1.0 / np.sqrt(2)
        sv[1] = 1.0 / np.sqrt(2)
    elif name == '|->':
        if dim < 2:
            raise ValueError(f"State |-> requires at least 1 qubit (dim >= 2), got dim={dim}")
        sv[0] = 1.0 / np.sqrt(2)
        sv[1] = -1.0 / np.sqrt(2)
    elif name == '|BELL>':
        sv[0] = 1.0 / np.sqrt(2)
        sv[dim - 1] = 1.0 / np.sqrt(2)
    elif name == '|GHZ>':
        sv[0] = 1.0 / np.sqrt(2)
        sv[dim - 1] = 1.0 / np.sqrt(2)
    elif name == '|GHZ3>':
        if dim >= 8:
            sv[0] = 1.0 / np.sqrt(2)
            sv[7] = 1.0 / np.sqrt(2)
        else:
            sv[0] = 1.0  # fallback to |0>
    elif name == '|GHZ4>':
        if dim >= 16:
            sv[0] = 1.0 / np.sqrt(2)
            sv[15] = 1.0 / np.sqrt(2)
        else:
            sv[0] = 1.0  # fallback to |0>
    elif name in ('|W>', '|W3>'):
        if dim >= 8:
            sv[1] = 1.0 / np.sqrt(3)
            sv[2] = 1.0 / np.sqrt(3)
            sv[4] = 1.0 / np.sqrt(3)
        else:
            # fallback: equal superposition of available states
            sv[:] = 1.0 / np.sqrt(dim)
    else:
        sv[0] = 1.0
    return sv


def _named_state_fits(name: str, n_qubits: int) -> bool:
    """Whether a named SET_STATE target fits the current qubit count."""
    dim = 2 ** n_qubits
    need = {'|1>': 2, '|+>': 2, '|->': 2, '|GHZ3>': 8, '|GHZ4>': 16,
            '|W>': 8, '|W3>': 8}
    return dim >= need.get(name.upper(), 1)


class QuantumStatementMixin(SimulationHost):
    """The keyword statements that act on qubits."""

    # Mid-circuit measurement returns this value in Qiskit circuit-build mode.
    # Qiskit defers measurement to simulation time, so the variable cannot hold
    # the actual outcome during circuit construction. For classical feedforward,
    # use LOCC mode with SEND instead.
    MEAS_CIRCUIT_MODE_VALUE = 0

    def _try_exec_meas(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
        """Handle MEAS qubit -> var (mid-circuit measurement).

        In standard Aer mode this is a real dynamic-circuit measurement: the
        qubit is measured into a classical register recorded under `var`, which a
        later ``IF var THEN <gate>`` turns into a Qiskit if_test (feedforward) at
        build time. In LOCC mode it routes through the numpy engine as before.
        """
        m = RE_MEAS.match(stmt)
        if not m:
            # Bare MEAS (no "-> bit") is the classic confusion with MEASURE:
            # MEAS is a mid-circuit measurement that must name a classical bit,
            # MEASURE collapses qubits into the result histogram.
            if re.match(r'MEAS\b', stmt, re.IGNORECASE):
                raise ValueError(
                    "MEAS needs a target bit: MEAS <qubit> -> <bit> "
                    "(mid-circuit measurement used by IF). To collapse qubits "
                    "into the result histogram, use MEASURE <qubit>.")
            return False
        qubit = self._resolve_qubit(m.group(1))
        var = m.group(2)
        if self._live is not None:
            bit = self._live.measure(qubit)
            run_vars[var] = bit
            self.variables[var] = bit
            return True
        self._measure_into_bit(qc, qubit, var, 'mc')
        run_vars[var] = self.MEAS_CIRCUIT_MODE_VALUE
        self.variables[var] = self.MEAS_CIRCUIT_MODE_VALUE
        return True

    def _measure_into_bit(self, qc: QuantumCircuit, qubit: int, var: str, prefix: str) -> None:
        """Measure `qubit` into a fresh one-bit register recorded as `var`, so
        IF var THEN ... compiles to feedforward on this outcome. Measuring the
        same name again (in a loop) gets a new register; IF reads the latest."""
        from qiskit.circuit import ClassicalRegister
        taken = {r.name for r in qc.cregs}
        name, k = f'{prefix}_{var}', 1
        while name in taken:
            name, k = f'{prefix}_{var}_{k}', k + 1
        cr = ClassicalRegister(1, name)
        qc.add_register(cr)
        qc.measure(qubit, cr[0])
        self._classical_bits[var] = cr

    def _try_exec_reset(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
        """Handle RESET qubit."""
        m = RE_RESET.match(stmt)
        if not m:
            return False
        qubit = self._resolve_qubit(m.group(1))
        if self._live is not None:
            self._live.reset(qubit)
        else:
            qc.reset(qubit)
        return True

    def _try_exec_measure_basis(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any],
                               *, backend: CircuitBackend | None = None) -> bool:
        """Handle MEASURE_X/Y/Z qubit — measurement in a non-computational basis."""
        m = RE_MEASURE_BASIS.match(stmt)
        if not m:
            return False
        basis = m.group(1).upper()
        qubit = self._resolve_qubit(m.group(2))
        var = f"m{basis.lower()}_{qubit}"
        rotation = {'X': ['H'], 'Y': ['SDG', 'H'], 'Z': []}[basis]
        live = self._live
        if live is not None:
            live.flush()
            for g in rotation:
                live.apply_named(g, (), [qubit])
            bit = live.measure(qubit)
            run_vars[var] = bit
            self.variables[var] = bit
            return True
        for g in rotation:
            self._apply_gate(qc, g, [], [qubit])
        self._measure_into_bit(qc, qubit, var, 'meas')
        run_vars[var] = 0
        self.variables[var] = 0
        return True

    _PAULI_TO_CONTROLLED = {'X': 'CX', 'Y': 'CY', 'Z': 'CZ'}

    def _try_exec_syndrome(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any],
                           *, backend: CircuitBackend | None = None) -> bool:
        """Handle SYNDROME — non-destructive stabilizer measurement via ancilla."""
        parsed = self._parse_syndrome(stmt, run_vars)
        if parsed is None:
            return False
        pauli_str, qubits, var = parsed
        anc = self.num_qubits - 1
        if anc in qubits:
            raise ValueError(
                f"Qubit {anc} needed as ancilla but appears in stabilizer. "
                f"Increase QUBITS by 1.")
        gates = [('H', [anc])] + [(self._PAULI_TO_CONTROLLED[p], [anc, q])
                                  for p, q in zip(pauli_str, qubits) if p != 'I'] + [('H', [anc])]
        live = self._live
        if live is not None:
            live.flush()
            for g, qs in gates:
                live.apply_named(g, (), qs)
            bit = live.measure(anc)
            live.reset(anc)
            run_vars[var] = bit
            self.variables[var] = bit
            return True
        for g, qs in gates:
            self._apply_gate(qc, g, [], qs)
        self._measure_into_bit(qc, anc, var, 'synd')
        qc.reset(anc)
        run_vars[var] = 0
        self.variables[var] = 0
        return True

    def _try_exec_delay(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any]) -> bool:
        """DELAY q[, q ...], microseconds: leave qubits idle. An ideal run is
        unchanged; DEVICE IBM's noise model and a QPU run decohere them."""
        m = re.match(r'DELAY\s+(.+)$', stmt, re.IGNORECASE)
        if not m:
            return False
        parts = [p.strip() for p in m.group(1).split(',')]
        if len(parts) < 2:
            raise ValueError("DELAY q[, q ...], microseconds")
        us = float(self._eval_with_vars(parts[-1], run_vars))
        if us < 0:
            raise ValueError(f"DELAY of {us:g} us is negative")
        qubits = [self._resolve_qubit(p) for p in parts[:-1]]
        if self._live is None:
            for q in qubits:
                qc.delay(us, q, unit='us')
        return True

    def _try_exec_apply_circuit(self, stmt: str, qc: QuantumCircuit, *, backend: CircuitBackend | None = None) -> bool:
        """Handle APPLY_CIRCUIT name [@offset] inside program execution."""
        m = re.match(r'APPLY_CIRCUIT\s+(\w+)(?:\s+@(\d+))?', stmt, re.IGNORECASE)
        if not m:
            return False
        name = m.group(1).upper()
        offset = int(m.group(2)) if m.group(2) else 0
        if name not in self.subroutines:
            raise ValueError(f"UNDEFINED CIRCUIT: {name}")
        call_str = f"{name} @{offset}" if offset else name
        self._apply_gate_str(call_str, qc, backend=backend)
        return True

    def _try_exec_save_expect(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any]) -> bool:
        """SAVE_EXPECT <pauli> <qubits> -> <var> — inline expectation value.

        Also accepts a declared HAMILTONIAN name with no qubit list
        (SAVE_EXPECT HM -> e), recording <HM> for VQE cost functions."""
        from qubasic_core.parser import RE_SAVE_EXPECT
        m = RE_SAVE_EXPECT.match(stmt)
        if not m:
            return False
        pauli_str = m.group(1).upper()
        qubits = [int(q.strip()) for q in m.group(2).replace(',', ' ').split() if q.strip()]
        var = m.group(3)
        try:
            from qiskit.quantum_info import SparsePauliOp
            if not qubits and pauli_str in getattr(self, '_hamiltonians', {}):
                op = self._hamiltonians[pauli_str]
                if op.num_qubits != self.num_qubits:
                    raise ValueError(f"HAMILTONIAN {pauli_str} is {op.num_qubits}-qubit "
                                     f"but QUBITS is {self.num_qubits}")
            else:
                full_pauli = ['I'] * self.num_qubits
                for i, p in enumerate(pauli_str):
                    if i < len(qubits):
                        full_pauli[self.num_qubits - 1 - qubits[i]] = p
                op = SparsePauliOp(''.join(full_pauli))
            if self._live is not None:
                # Live run: the value at this point of this shot (mean over
                # shots once the run ends).
                val = self._live.expectation(op)
                self._record_save('exp', var, val)
                run_vars[var] = val
                self.variables[var] = val
                return True
            qc.save_expectation_value(op, list(range(self.num_qubits)), label=f'exp_{var}')
            # Placeholder until the run fills the real value via
            # _extract_save_results. Preserve any prior value instead of
            # zeroing it, so a re-run that re-includes this SAVE line leaves
            # earlier results readable to LET/PRINT during the build pass.
            _prev = self.variables.get(var, 0)
            run_vars[var] = _prev
            self.variables[var] = _prev
        except Exception as e:
            self.io.writeln(f"?SAVE_EXPECT ERROR: {e}")
        return True

    def _try_exec_save_probs(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any]) -> bool:
        """SAVE_PROBS <qubits> -> <var> — inline probability snapshot."""
        from qubasic_core.parser import RE_SAVE_PROBS
        m = RE_SAVE_PROBS.match(stmt)
        if not m:
            return False
        qubits = [int(q.strip()) for q in m.group(1).split(',') if q.strip()]
        var = m.group(2)
        try:
            if self._live is not None:
                probs = self._live.probabilities(qubits)
                self._record_save('prob', var, probs)
                run_vars[var] = probs
                self.variables[var] = probs
                self.arrays[var] = probs.tolist()
                return True
            qc.save_probabilities(qubits, label=f'prob_{var}')
            _prev = self.variables.get(var, 0)   # preserve prior value (see SAVE_EXPECT)
            run_vars[var] = _prev
            self.variables[var] = _prev
        except Exception as e:
            self.io.writeln(f"?SAVE_PROBS ERROR: {e}")
        return True

    def _try_exec_save_amps(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any]) -> bool:
        """SAVE_AMPS <indices> -> <var> — save specific amplitudes by index."""
        from qubasic_core.parser import RE_SAVE_AMPS
        m = RE_SAVE_AMPS.match(stmt)
        if not m:
            return False
        indices = [int(q.strip()) for q in m.group(1).split(',') if q.strip()]
        var = m.group(2)
        try:
            if self._live is not None:
                amps = self._live.amplitudes(indices)
                self._record_save('amp', var, amps)
                run_vars[var] = amps
                self.variables[var] = amps
                self.arrays[var] = [complex(a) for a in amps]
                return True
            qc.save_amplitudes(indices, label=f'amp_{var}')
            _prev = self.variables.get(var, 0)   # preserve prior value (see SAVE_EXPECT)
            run_vars[var] = _prev
            self.variables[var] = _prev
        except Exception as e:
            self.io.writeln(f"?SAVE_AMPS ERROR: {e}")
        return True

    def _state_from_spec(self, spec: str) -> np.ndarray:
        """The normalized statevector SET_STATE describes: a named state (|0>,
        |1>, |+>, |->, |BELL>, |GHZ>, |GHZ3>, |GHZ4>, |W>, |W3>) or a list of
        amplitudes, rescaled to unit norm with a note when it is not."""
        dim = 2 ** self.num_qubits
        name = spec.strip().upper()
        if name in self._NAMED_STATES:
            if not _named_state_fits(name, self.num_qubits):
                self.io.writeln(f"  (warning: {name} needs more qubits than "
                                f"{self.num_qubits}; falling back to |0...0>)")
            sv = _resolve_named_state(name, self.num_qubits)
        else:
            sv = np.array(self._parse_matrix(spec), dtype=complex).ravel()
        if len(sv) != dim:
            raise ValueError(f"State vector length {len(sv)} != 2^{self.num_qubits} = {dim}")
        norm = float(np.sum(np.abs(sv) ** 2))
        if norm < 1e-12:
            raise ValueError("the state vector is zero")
        if abs(norm - 1.0) > 1e-6:
            sv = sv / np.sqrt(norm)
            self.io.writeln(f"  (normalized: ||sv||={norm:.6f} -> 1.0)")
        return sv

    def _try_exec_set_state(self, stmt: str, qc: QuantumCircuit) -> bool:
        """SET_STATE |name> | [amplitudes] — inject a state at this point of
        the circuit (see _state_from_spec)."""
        from qubasic_core.parser import RE_SET_STATE
        m = RE_SET_STATE.match(stmt)
        if not m:
            return False
        try:
            from qiskit.quantum_info import Statevector
            from qiskit_aer.library import SetStatevector
            sv = self._state_from_spec(m.group(1))
            qc.append(SetStatevector(Statevector(sv)), list(range(self.num_qubits)))
        except Exception as e:
            self.io.writeln(f"?SET_STATE ERROR: {e}")
        return True

    def cmd_set_state(self, rest: str) -> None:
        """SET_STATE <state> — set the statevector immediately (stores for next RUN)."""
        if not rest.strip():
            self.io.writeln("?USAGE: SET_STATE |+> or SET_STATE [0.707, 0, 0, 0.707]")
            return
        try:
            sv = self._state_from_spec(rest)
            self.last_sv = sv
            # Persist so the next RUN starts from this state (prepended at build).
            self._pending_set_state = sv
            self.io.writeln(f"STATE SET ({self.num_qubits} qubits) — applied on next RUN")
        except Exception as e:
            self.io.writeln(f"?SET_STATE ERROR: {e}")

    def cmd_set_density(self, rest: str) -> None:
        """SET_DENSITY [[...]] — inject a mixed state (density matrix) for the next RUN.

        Accepts a 2^n x 2^n Hermitian, unit-trace, positive-semidefinite matrix
        (e.g. [[0.5,0],[0,0.5]] for the maximally mixed qubit). Switches the
        method to density_matrix; STATE is unavailable for a mixed state, but
        DENSITY and the measurement histogram reflect it."""
        expr = rest.strip()
        if not expr:
            self.io.writeln("?USAGE: SET_DENSITY [[a,b],[c,d]]  (2^n x 2^n, Hermitian, trace 1)")
            return
        try:
            rho = np.array(self._parse_matrix(expr), dtype=complex)
            dim = 2 ** self.num_qubits
            if rho.shape != (dim, dim):
                raise ValueError(f"matrix is {rho.shape}, need ({dim}, {dim}) for {self.num_qubits} qubit(s)")
            if not np.allclose(rho, rho.conj().T, atol=1e-6):
                raise ValueError("matrix is not Hermitian")
            tr = complex(np.trace(rho))
            if abs(tr - 1.0) > 1e-6:
                if abs(tr) < 1e-12:
                    raise ValueError("matrix has zero trace")
                rho = rho / tr
                self.io.writeln(f"  (normalized: trace {tr.real:.4f} -> 1.0)")
            evals = np.linalg.eigvalsh(rho)
            if evals.min() < -1e-6:
                raise ValueError(f"matrix is not positive semidefinite (min eigenvalue {evals.min():.4f})")
            self._pending_set_density = rho
            self._pending_set_state = None
            self.last_sv = None
            self.sim_method = 'density_matrix'
            self._circuit_cache_key = None
            purity = float(np.real(np.trace(rho @ rho)))
            self.io.writeln(f"DENSITY MATRIX SET ({self.num_qubits} qubits, purity {purity:.4f}) "
                            f"— applied on next RUN (method=density_matrix)")
        except Exception as e:
            self.io.writeln(f"?SET_DENSITY ERROR: {e}")

    def cmd_apply_circuit(self, rest: str) -> None:
        """APPLY_CIRCUIT name [@offset] — apply a circuit macro."""
        m = re.match(r'(\w+)(?:\s+@(\d+))?', rest)
        if not m:
            self.io.writeln("?USAGE: APPLY_CIRCUIT <name> [@offset]")
            return
        name = m.group(1).upper()
        offset = int(m.group(2)) if m.group(2) else 0
        if name not in self.subroutines:
            self.io.writeln(f"?UNDEFINED CIRCUIT: {name}")
            return
        call_str = f"{name} @{offset}" if offset else name
        self.run_immediate(call_str)

    # Keyword statements on qubits, merged into _STMT_KEYWORDS by the terminal;
    # each handler is (self, stmt, qc, run_vars, backend) -> handled.
    _QUANTUM_STATEMENTS: dict[str, tuple[Any, ...]] = {
        'MEAS': (lambda s, st, qc, rv, b: s._try_exec_meas(st, qc, rv, backend=b),),
        'RESET': (lambda s, st, qc, rv, b: s._try_exec_reset(st, qc, rv, backend=b),),
        **dict.fromkeys(('MEASURE_X', 'MEASURE_Y', 'MEASURE_Z'), (
            lambda s, st, qc, rv, b: s._try_exec_measure_basis(st, qc, rv, backend=b),)),
        'SYNDROME': (lambda s, st, qc, rv, b: s._try_exec_syndrome(st, qc, rv, backend=b),),
        'EVOLVE': (lambda s, st, qc, rv, b: s._try_exec_evolve(st, qc, rv, backend=b),),
        'APPLYCHANNEL': (lambda s, st, qc, rv, b: s._try_exec_applychannel(st, qc, rv, backend=b),),
        'POKE': (lambda s, st, qc, rv, b: s._try_exec_poke(st, rv),),
        'WAIT': (lambda s, st, qc, rv, b: s._try_exec_wait(st, rv),),
        'SYS': (lambda s, st, qc, rv, b: s._try_exec_sys(st),),
        'SAVE_EXPECT': (lambda s, st, qc, rv, b: s._try_exec_save_expect(st, qc, rv),),
        'SAVE_PROBS': (lambda s, st, qc, rv, b: s._try_exec_save_probs(st, qc, rv),),
        'SAVE_AMPS': (lambda s, st, qc, rv, b: s._try_exec_save_amps(st, qc, rv),),
        'SET_STATE': (lambda s, st, qc, rv, b: s._try_exec_set_state(st, qc),),
        'APPLY_CIRCUIT': (lambda s, st, qc, rv, b: s._try_exec_apply_circuit(st, qc, backend=b),),
        'DELAY': (lambda s, st, qc, rv, b: s._try_exec_delay(st, qc, rv),),
    }

    _NAMED_STATES = ('|0>', '|1>', '|+>', '|->', '|BELL>', '|GHZ>', '|GHZ3>', '|GHZ4>', '|W>', '|W3>')
