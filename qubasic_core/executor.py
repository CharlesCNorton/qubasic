"""QUBASIC executor mixin — circuit building and line execution."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit, transpile

from qubasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    ExecResult,
    RE_REG_INDEX, RE_AT_REG_LINE,
    RE_CTRL, RE_INV,
    RE_SYNDROME,
)
from qubasic_core.expression import ExpressionMixin
from qubasic_core.errors import QBasicBuildError, QBasicRangeError

if TYPE_CHECKING:
    from qubasic_core.exec_context import ExecContext


class ExecutorMixin:
    """Circuit building and per-line execution for QBasicTerminal.

    Provides the core execution pipeline: circuit compilation from stored
    programs, single-line execution, gate tokenization, qubit resolution,
    subroutine expansion, and immediate-mode gate application.

    All methods access shared terminal state (variables, arrays, subroutines,
    registers, etc.) through ``self``.
    """

    # Names reserved from variable substitution -- derived from source tables
    # so they stay in sync automatically.
    _RESERVED_KEYWORDS = frozenset({
        'REM', 'MEASURE', 'BARRIER', 'END', 'RETURN',
        'FOR', 'NEXT', 'WHILE', 'WEND', 'IF', 'THEN', 'ELSE',
        'GOTO', 'GOSUB', 'LET', 'PRINT', 'INPUT', 'DIM',
        'AND', 'OR', 'NOT', 'TO', 'STEP', 'SEND', 'SHARE',
        'MEAS', 'RESET', 'UNITARY', 'CTRL', 'INV',
    })
    _RESERVED_NAMES = (
        set(GATE_TABLE.keys()) | set(GATE_ALIASES.keys()) |
        set(ExpressionMixin._SAFE_CONSTS.keys()) |
        set(ExpressionMixin._SAFE_FUNCS.keys()) |
        _RESERVED_KEYWORDS
    )

    # Gate dispatch: name -> (method_name, arg_pattern)
    # arg_pattern: 'q' = qubits only, 'pq' = params then qubits, 'ppq' = 3-param then qubit
    _GATE_DISPATCH = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'T': 't', 'SDG': 'sdg', 'TDG': 'tdg',
        'SX': 'sx', 'ID': 'id',
        'CX': 'cx', 'CZ': 'cz', 'CY': 'cy', 'CH': 'ch',
        'SWAP': 'swap', 'DCX': 'dcx', 'ISWAP': 'iswap',
        'CCX': 'ccx', 'CSWAP': 'cswap',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz', 'P': 'p',
        'CRX': 'crx', 'CRY': 'cry', 'CRZ': 'crz', 'CP': 'cp',
        'RXX': 'rxx', 'RYY': 'ryy', 'RZZ': 'rzz',
        'U': 'u',
    }

    # ── Circuit Building ──────────────────────────────────────────────

    def build_circuit(self) -> tuple['QuantumCircuit', bool]:
        """Compile program lines into a QuantumCircuit. Returns (circuit, has_measure)."""
        from qubasic_core.exec_context import ExecContext
        from qubasic_core.statements import MeasureStmt, CompoundStmt
        from qubasic_core.scope import Scope
        from qubasic_core.backend import QiskitBackend

        qc = QuantumCircuit(self.num_qubits)
        # Apply any qubit state preparation requested via POKE to $0100.
        if getattr(self, '_poke_state_prep', None):
            self._emit_poke_state_prep(qc)
        # Apply a pending immediate SET_STATE so it persists into this RUN.
        _pend = getattr(self, '_pending_set_state', None)
        if _pend is not None and len(_pend) == 2 ** self.num_qubits:
            from qiskit.quantum_info import Statevector
            from qiskit_aer.library import SetStatevector
            qc.append(SetStatevector(Statevector(_pend)), list(range(self.num_qubits)))
        backend = QiskitBackend(qc, self._apply_gate)
        ctx = ExecContext(
            sorted_lines=sorted(self.program.keys()),
            ip=0,
            run_vars=Scope(self.variables),
            max_iterations=self._max_iterations,
            qc=qc,
            backend=backend,
        )
        has_measure = False
        self._on_measure_fired = False

        while ctx.ip < len(ctx.sorted_lines):
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            line_num = ctx.sorted_lines[ctx.ip]
            stmt = self.program[line_num].strip()
            parsed = self._get_parsed(line_num)

            # Debug instrumentation — each is a no-op unless explicitly enabled.
            if self._trace_mode:
                self._trace_line(line_num)
            if self._breakpoints and self._check_breakpoint(line_num, ctx.sorted_lines, ctx.ip):
                break
            if self._on_timer_target is not None:
                _timer_tgt = self._check_timer_callback(ctx.sorted_lines, ctx.ip)
                if _timer_tgt is not None:
                    ctx.ip = _timer_tgt
                    continue

            if isinstance(parsed, MeasureStmt):
                has_measure = True
                if self._on_measure_target is not None and not self._on_measure_fired:
                    self._on_measure_fired = True
                    self._gosub_stack.append(ctx.ip + 1)
                    for _i, _ln in enumerate(ctx.sorted_lines):
                        if _ln == self._on_measure_target:
                            ctx.ip = _i
                            break
                    else:
                        ctx.ip += 1
                    continue
                ctx.ip += 1
                continue
            if isinstance(parsed, CompoundStmt):
                for part in parsed.parts:
                    if part.strip().upper() == 'MEASURE':
                        has_measure = True

            if hasattr(self, '_profile_line_start'):
                self._profile_line_start(line_num)
            try:
                result = self._exec_line(stmt, parsed=parsed, ctx=ctx)
            except QBasicBuildError as e:
                raise QBasicBuildError(
                    f"LINE {line_num}: {e}"
                ) from None
            except Exception as e:
                raise RuntimeError(f"LINE {line_num}: {e}") from None
            finally:
                if hasattr(self, '_profile_line_end'):
                    self._profile_line_end(line_num)

            if result is ExecResult.END:
                break
            elif isinstance(result, int):
                ctx.ip = result
            else:
                ctx.ip += 1

        return qc, has_measure

    def _exec_line(self, stmt, qc=None, loop_stack=None, sorted_lines=None,
                   ip=0, run_vars=None, parsed=None, *, ctx=None):
        """Execute one program line.

        Accepts either individual parameters (legacy) or ctx (ExecContext).
        When ctx is provided, qc/loop_stack/sorted_lines/ip/run_vars are
        read from ctx and the individual params are ignored.

        Evaluation order (deterministic, first match wins):
          1. Typed fast-path (parsed Stmt): BARRIER, REM, MEASURE, END, @REG, compound
          2. Control flow: LET, PRINT, GOTO, GOSUB, FOR/NEXT, WHILE/WEND, IF/THEN,
             DATA/READ, ON GOTO/GOSUB, SELECT CASE, DO/LOOP, EXIT, SUB/FUNCTION,
             ON ERROR, ASSERT, STOP, SWAP, DEF FN, OPTION BASE
          3. Statement handlers: MEAS, RESET, MEASURE_X/Y/Z, SYNDROME, UNITARY,
             DIM, REDIM, ERASE, GET, INPUT, POKE, SYS, file I/O, PRINT USING
          4. Colon-separated compound statements
          5. Gate application (subroutine expansion + gate dispatch)

        Returns: int (jump target ip), ExecResult.ADVANCE, or ExecResult.END.
        """
        if ctx is not None:
            qc = ctx.backend.qc if ctx.backend and hasattr(ctx.backend, 'qc') else ctx.qc
            loop_stack = ctx.loop_stack
            sorted_lines = ctx.sorted_lines
            ip = ctx.ip
            run_vars = ctx.run_vars
        from qubasic_core.statements import (
            BarrierStmt, RemStmt, MeasureStmt, EndStmt, ReturnStmt,
            CompoundStmt, AtRegStmt, GotoStmt, GosubStmt,
            ForStmt, NextStmt, WhileStmt, WendStmt, IfThenStmt,
            LetStmt, LetArrayStmt, PrintStmt,
            GateStmt, RawStmt,
        )

        # 1. Typed fast-path: terminals that don't need control-flow dispatch
        if parsed is None:
            from qubasic_core.parser import parse_stmt
            parsed = parse_stmt(stmt)
        if isinstance(parsed, BarrierStmt):
            if hasattr(qc, 'barrier'):
                qc.barrier()
            return ExecResult.ADVANCE
        if isinstance(parsed, (RemStmt, MeasureStmt)):
            return ExecResult.ADVANCE
        if isinstance(parsed, EndStmt):
            return ExecResult.END
        if isinstance(parsed, ReturnStmt):
            if not self._gosub_stack:
                raise RuntimeError("RETURN WITHOUT GOSUB")
            return self._gosub_stack.pop()
        if isinstance(parsed, AtRegStmt) and not self.locc_mode:
            raise ValueError("@register syntax requires LOCC mode (try: LOCC <n1> <n2>)")
        if isinstance(parsed, CompoundStmt):
            for sub in parsed.parts:
                self._exec_line(sub, qc=qc, loop_stack=loop_stack,
                                sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)
            return ExecResult.ADVANCE

        # 2. Delegate to unified control-flow dispatch (handles GOTO, GOSUB,
        #    FOR/NEXT, WHILE/WEND, IF/THEN, LET, PRINT, DATA/READ, ON GOTO,
        #    SELECT CASE, DO/LOOP, EXIT, SUB/FUNCTION, ON ERROR, etc.)
        handled, result = self._exec_control_flow(
            stmt, loop_stack, sorted_lines, ip, run_vars,
            lambda s, ls, sl, i, rv: self._exec_line(
                s, qc=qc, loop_stack=ls, sorted_lines=sl, ip=i, run_vars=rv),
            parsed=parsed)
        if handled:
            return result

        # 3. Remaining statement handlers (not in control-flow dispatch)
        from qubasic_core.statements import RestoreStmt
        if isinstance(parsed, RestoreStmt):
            return ExecResult.ADVANCE

        # Multi-line IF block markers — no-ops during execution
        upper = stmt.strip().upper()
        if upper in ('END IF', 'ELSE'):
            return ExecResult.ADVANCE

        # 3. Statement handlers
        _backend = ctx.backend if ctx else None
        if self._try_stmt_handlers(stmt, qc, run_vars, backend=_backend):
            return ExecResult.ADVANCE

        # 4. Colon-separated (legacy fallback for unparsed compound)
        if ':' in stmt:
            for sub in self._split_colon_stmts(stmt):
                self._exec_line(sub, qc=qc, loop_stack=loop_stack,
                                sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)
            return ExecResult.ADVANCE

        # 5. Gate application
        _backend = ctx.backend if ctx else None

        # Fast path: GateStmt already parsed by the parser
        if isinstance(parsed, GateStmt):
            info = self._gate_info(parsed.name)
            if info is not None:
                n_params, n_qubits = info
                args = list(parsed.args)
                params = [self._eval_with_vars(a, run_vars) for a in args[:n_params]]
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

        # Slow path: subroutine expansion + gate dispatch
        expanded = self._expand_statement(stmt)
        for gate_str in expanded:
            self._apply_gate_str(gate_str, qc, backend=_backend)

        return ExecResult.ADVANCE

    def _parse_syndrome(self, stmt: str, run_vars: dict) -> tuple[str, list[int], str] | None:
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

    def _substitute_vars(self, stmt: str, run_vars: dict) -> str:
        """Replace variable names with their values in a statement.

        Tokenizes the statement and replaces eligible identifiers in-place,
        avoiding substitution inside quoted strings, register notation, and
        protected names (gates, keywords, constants, subroutines, custom gates).
        """
        merged = {**self.variables, **run_vars}
        if not merged:
            return stmt
        # Build the set of names that should never be substituted
        protected = (
            self._RESERVED_NAMES |
            {name.lower() for name in self.registers} |
            {name.upper() for name in self.registers} |
            set(self.subroutines.keys()) |
            set(self._custom_gates.keys())
        )
        # Tokenize: split on word boundaries, preserving delimiters
        tokens = re.split(r'(\b\w+\b)', stmt)
        for i, tok in enumerate(tokens):
            if not tok or not tok[0].isalpha():
                continue
            if tok in protected or tok.upper() in protected or tok.lower() in protected:
                continue
            if tok in merged:
                # Don't expand record bases (dicts) — leave p in p.x intact so
                # the expression evaluator can resolve the field.
                if isinstance(merged[tok], dict):
                    continue
                tokens[i] = str(merged[tok])
        return ''.join(tokens)

    def _expand_statement(self, stmt, _call_stack: set[str] | None = None):
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
            def _sub(m):
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

    def _apply_gate_str(self, stmt, qc, _call_stack=None, *, backend=None):
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

        # CTRL gate ctrl_qubit, target_qubit(s) -- controlled version of any gate
        m_ctrl = RE_CTRL.match(stmt)
        if m_ctrl:
            from qiskit.circuit.library import (HGate, XGate, YGate, ZGate,
                SGate, TGate, SdgGate, TdgGate, SXGate, SwapGate)
            gate_name = m_ctrl.group(1).upper()
            args = [a.strip() for a in m_ctrl.group(2).replace(',', ' ').split()]
            ctrl_qubit = self._resolve_qubit(args[0])
            target_qubits = [self._resolve_qubit(a) for a in args[1:]]
            gate_map = {
                'H': HGate(), 'X': XGate(), 'Y': YGate(), 'Z': ZGate(),
                'S': SGate(), 'T': TGate(), 'SDG': SdgGate(), 'TDG': TdgGate(),
                'SX': SXGate(), 'SWAP': SwapGate(),
            }
            all_qubits = [ctrl_qubit] + target_qubits
            if gate_name in gate_map:
                gate = gate_map[gate_name].control(1)
                if backend and hasattr(backend, 'append_controlled'):
                    backend.append_controlled(gate, all_qubits)
                else:
                    qc.append(gate, all_qubits)
            elif gate_name in self._custom_gates:
                from qiskit.circuit.library import UnitaryGate
                gate = UnitaryGate(self._custom_gates[gate_name]).control(1)
                if backend and hasattr(backend, 'append_controlled'):
                    backend.append_controlled(gate, all_qubits)
                else:
                    qc.append(gate, all_qubits)
            else:
                raise ValueError(f"CTRL {gate_name}: gate not found")
            return

        # INV gate qubit(s) -- inverse/dagger of a single gate
        m_inv = RE_INV.match(stmt)
        if m_inv:
            gate_name = m_inv.group(1).upper()
            inv_args = m_inv.group(2)
            tokens = self._tokenize_gate(f"{gate_name} {inv_args}")
            gate_key = GATE_ALIASES.get(gate_name, gate_name)
            info = self._gate_info(gate_key)
            if info is not None:
                n_params, n_qubits_needed = info
                t_args = tokens[1:]
                params = [self.eval_expr(a) for a in t_args[:n_params]]
                qubits_inv = [self._resolve_qubit(a) for a in t_args[n_params:n_params+n_qubits_needed]]
                sub_qc = QuantumCircuit(n_qubits_needed)
                self._apply_gate(sub_qc, gate_key, params, list(range(n_qubits_needed)))
                if backend and hasattr(backend, 'append_inverse'):
                    backend.append_inverse(sub_qc, qubits_inv)
                else:
                    qc.append(sub_qc.inverse(), qubits_inv)
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

        params = [self.eval_expr(a) for a in args[:n_params]]
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
                q = int(self.eval_expr(arg))
            except Exception:
                raise ValueError(f"CANNOT RESOLVE QUBIT: {arg}")
        if q < 0 or q >= limit:
            raise QBasicRangeError(f"QUBIT {q} OUT OF RANGE (0-{limit-1})")
        return q

    def _apply_gate(self, qc, gate_name, params, qubits):
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
        # In LOCC mode, handle @register prefix via the numpy engine
        if self.locc_mode and self.locc:
            m = RE_AT_REG_LINE.match(line)
            if m:
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
        # Build and execute through the same gate pipeline as cmd_run
        from qubasic_core.exec_context import ExecContext
        qc = QuantumCircuit(self.num_qubits)
        imm_ctx = ExecContext(sorted_lines=[0], ip=0,
                              run_vars=dict(self.variables), qc=qc)
        self._exec_line(line, ctx=imm_ctx)
        qc.save_statevector()
        backend = self._make_backend('statevector')
        result = backend.run(transpile(qc, backend)).result()
        sv = np.array(result.get_statevector())
        self.last_sv = sv
        self.last_circuit = qc
        self._print_sv_compact(sv)
