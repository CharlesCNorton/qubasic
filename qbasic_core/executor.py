"""QBASIC executor mixin — circuit building and line execution."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qbasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    ExecResult,
    RE_REG_INDEX, RE_AT_REG_LINE,
    RE_CTRL, RE_INV,
    RE_SYNDROME,
)
from qbasic_core.expression import ExpressionMixin
from qbasic_core.errors import QBasicBuildError, QBasicRangeError

if TYPE_CHECKING:
    from qbasic_core.exec_context import ExecContext


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
        from qbasic_core.exec_context import ExecContext
        from qbasic_core.statements import MeasureStmt, CompoundStmt
        from qbasic_core.scope import Scope
        from qbasic_core.backend import QiskitBackend

        qc = QuantumCircuit(self.num_qubits)
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

        while ctx.ip < len(ctx.sorted_lines):
            ctx.iteration_count += 1
            if ctx.iteration_count > ctx.max_iterations:
                raise RuntimeError(f"LOOP LIMIT ({ctx.max_iterations}) — possible infinite loop")
            line_num = ctx.sorted_lines[ctx.ip]
            stmt = self.program[line_num].strip()
            parsed = self._get_parsed(line_num)

            if isinstance(parsed, MeasureStmt):
                has_measure = True
                ctx.ip += 1
                continue
            if isinstance(parsed, CompoundStmt):
                for part in parsed.parts:
                    if part.strip().upper() == 'MEASURE':
                        has_measure = True

            try:
                result = self._exec_line(stmt, parsed=parsed, ctx=ctx)
            except QBasicBuildError as e:
                raise QBasicBuildError(
                    f"LINE {line_num}: {e}"
                ) from None
            except Exception as e:
                raise RuntimeError(f"LINE {line_num}: {e}") from None

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
        from qbasic_core.statements import (
            BarrierStmt, RemStmt, MeasureStmt, EndStmt, ReturnStmt,
            CompoundStmt, AtRegStmt, GotoStmt, GosubStmt,
            ForStmt, NextStmt, WhileStmt, WendStmt, IfThenStmt,
            LetStmt, LetArrayStmt, PrintStmt,
            GateStmt, RawStmt,
        )

        # 1. Typed fast-path (no regex, no string manipulation)
        if parsed is None:
            from qbasic_core.parser import parse_stmt
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
        if isinstance(parsed, GotoStmt):
            for idx, ln in enumerate(sorted_lines):
                if ln == parsed.target:
                    return idx
            raise RuntimeError(f"GOTO {parsed.target}: LINE NOT FOUND")
        if isinstance(parsed, GosubStmt):
            self._gosub_stack.append(ip + 1)
            for idx, ln in enumerate(sorted_lines):
                if ln == parsed.target:
                    return idx
            raise RuntimeError(f"GOSUB {parsed.target}: LINE NOT FOUND")
        if isinstance(parsed, WendStmt):
            _, r = self._cf_wend(run_vars, loop_stack, sorted_lines, ip)
            return r
        if isinstance(parsed, ForStmt):
            start = self._eval_with_vars(parsed.start_expr, run_vars)
            end = self._eval_with_vars(parsed.end_expr, run_vars)
            step = self._eval_with_vars(parsed.step_expr, run_vars) if parsed.step_expr else 1
            try:
                if start == int(start): start = int(start)
            except (OverflowError, ValueError): pass
            try:
                if end == int(end): end = int(end)
            except (OverflowError, ValueError): pass
            try:
                if isinstance(step, float) and step == int(step): step = int(step)
            except (OverflowError, ValueError): pass
            run_vars[parsed.var] = start
            self.variables[parsed.var] = start
            loop_stack.append({'var': parsed.var, 'current': start, 'end': end,
                               'step': step, 'return_ip': ip})
            return ExecResult.ADVANCE
        if isinstance(parsed, NextStmt):
            if not loop_stack or loop_stack[-1].get('var') != parsed.var:
                if loop_stack:
                    raise RuntimeError(f"NEXT {parsed.var} does not match current FOR {loop_stack[-1].get('var', '?')}")
                raise RuntimeError(f"NEXT {parsed.var} without matching FOR")
            loop = loop_stack[-1]
            loop['current'] += loop['step']
            if (loop['step'] > 0 and loop['current'] <= loop['end']) or \
               (loop['step'] < 0 and loop['current'] >= loop['end']):
                run_vars[parsed.var] = loop['current']
                self.variables[parsed.var] = loop['current']
                return loop['return_ip'] + 1
            else:
                loop_stack.pop()
                return ExecResult.ADVANCE
        if isinstance(parsed, WhileStmt):
            if self._eval_condition(parsed.condition, run_vars):
                loop_stack.append({'type': 'while', 'cond': parsed.condition, 'return_ip': ip})
                return ExecResult.ADVANCE
            else:
                return self._find_matching_wend(sorted_lines, ip)
        if isinstance(parsed, LetStmt):
            val = self._eval_with_vars(parsed.expr, run_vars)
            run_vars[parsed.name] = val
            self.variables[parsed.name] = val
            return ExecResult.ADVANCE
        if isinstance(parsed, LetArrayStmt):
            idx = int(self._eval_with_vars(parsed.index_expr, run_vars))
            val = self._eval_with_vars(parsed.value_expr, run_vars)
            if parsed.name not in self.arrays:
                self.arrays[parsed.name] = [0.0] * (idx + 1)
            while idx >= len(self.arrays[parsed.name]):
                self.arrays[parsed.name].append(0.0)
            self.arrays[parsed.name][idx] = val
            return ExecResult.ADVANCE
        if isinstance(parsed, PrintStmt):
            text = parsed.expr
            suppress_nl = text.rstrip().endswith(';')
            tab_advance = text.rstrip().endswith(',')
            if suppress_nl:
                text = text.rstrip().removesuffix(';').rstrip()
            elif tab_advance:
                text = text.rstrip().removesuffix(',').rstrip()
            # Quantum PRINT: @REG, QUBIT(n), ENTANGLEMENT(a,b)
            qprint = self._try_quantum_print(text, run_vars)
            if qprint is not None:
                if suppress_nl:
                    self.io.write(qprint)
                elif tab_advance:
                    col = len(qprint) % 14
                    self.io.write(qprint + ' ' * (14 - col if col > 0 else 14))
                else:
                    self.io.writeln(qprint)
                return ExecResult.ADVANCE
            # SPC/TAB inline
            def _spc(m_s):
                return ' ' * max(0, int(self._eval_with_vars(m_s.group(1), run_vars)))
            def _tab(m_t):
                return ' ' * max(0, int(self._eval_with_vars(m_t.group(1), run_vars)))
            text = re.sub(r'\bSPC\s*\(([^)]+)\)', _spc, text, flags=re.IGNORECASE)
            text = re.sub(r'\bTAB\s*\(([^)]+)\)', _tab, text, flags=re.IGNORECASE)
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                output = text[1:-1]
            else:
                try:
                    ns = run_vars.as_dict() if hasattr(run_vars, 'as_dict') else dict(run_vars) if not isinstance(run_vars, dict) else run_vars
                    result = self._safe_eval(text, extra_ns=ns)
                    output = str(result)
                except Exception:
                    output = text
            if suppress_nl:
                self.io.write(output)
            elif tab_advance:
                col = len(output) % 14
                self.io.write(output + ' ' * (14 - col if col > 0 else 14))
            else:
                self.io.writeln(output)
            return ExecResult.ADVANCE
        if isinstance(parsed, IfThenStmt):
            cond_vars = run_vars
            if self.locc_mode and self.locc:
                cond_vars = {**({} if not hasattr(run_vars, 'as_dict') else run_vars.as_dict()),
                             **self.locc.classical}
                if hasattr(run_vars, 'as_dict'):
                    cond_vars.update(run_vars.as_dict())
                else:
                    cond_vars.update(run_vars)
            result = ExecResult.ADVANCE
            if self._eval_condition(parsed.condition, cond_vars):
                if parsed.then_clause:
                    r = self._exec_line(parsed.then_clause, qc=qc, loop_stack=loop_stack,
                                        sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)
                    if r is not None and r is not ExecResult.ADVANCE:
                        result = r
            elif parsed.else_clause:
                r = self._exec_line(parsed.else_clause, qc=qc, loop_stack=loop_stack,
                                    sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)
                if r is not None and r is not ExecResult.ADVANCE:
                    result = r
            return result
        if isinstance(parsed, AtRegStmt) and not self.locc_mode:
            raise ValueError("@register syntax requires LOCC mode (try: LOCC <n1> <n2>)")
        if isinstance(parsed, CompoundStmt):
            for sub in parsed.parts:
                self._exec_line(sub, qc=qc, loop_stack=loop_stack,
                                sorted_lines=sorted_lines, ip=ip, run_vars=run_vars)
            return ExecResult.ADVANCE

        # 2. Extended typed dispatch -- direct isinstance checks (no dict/lambda overhead)
        from qbasic_core.statements import (
            DataStmt, ReadStmt, OnGotoStmt, OnGosubStmt,
            SelectCaseStmt, CaseStmt, EndSelectStmt,
            DoStmt, LoopStmt, ExitStmt,
            SwapStmt, DefFnStmt, OptionBaseStmt, RestoreStmt,
            SubStmt, EndSubStmt, FunctionStmt, EndFunctionStmt,
            CallStmt, LocalStmt, StaticStmt, SharedStmt,
            OnErrorStmt, ResumeStmt, ErrorStmt, AssertStmt,
            StopStmt, OnMeasureStmt, OnTimerStmt,
        )
        if isinstance(parsed, DataStmt):
            r = self._cf_data(stmt, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, ReadStmt):
            r = self._cf_read(stmt, run_vars, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, OnGotoStmt):
            r = self._cf_on_goto(stmt, run_vars, sorted_lines, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, OnGosubStmt):
            r = self._cf_on_gosub(stmt, run_vars, sorted_lines, ip, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, SelectCaseStmt):
            r = self._cf_select_case(stmt, run_vars, sorted_lines, ip, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, CaseStmt):
            r = self._cf_case(stmt, sorted_lines, ip, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, EndSelectStmt):
            r = self._cf_end_select(stmt, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, DoStmt):
            r = self._cf_do(stmt, run_vars, loop_stack, sorted_lines, ip, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, LoopStmt):
            r = self._cf_loop(stmt, run_vars, loop_stack, sorted_lines, ip, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, ExitStmt):
            r = self._cf_exit(stmt, loop_stack, sorted_lines, ip, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, SwapStmt):
            r = self._cf_swap(stmt, run_vars, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, DefFnStmt):
            r = self._cf_def_fn(stmt, run_vars, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, OptionBaseStmt):
            r = self._cf_option_base(stmt, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, RestoreStmt):
            return ExecResult.ADVANCE
        if isinstance(parsed, SubStmt):
            r = self._cf_sub(stmt, sorted_lines, ip, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, EndSubStmt):
            r = self._cf_end_sub(stmt, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, FunctionStmt):
            r = self._cf_function(stmt, sorted_lines, ip, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, EndFunctionStmt):
            r = self._cf_end_function(stmt, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, CallStmt):
            r = self._cf_call(stmt, run_vars, sorted_lines, ip, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, LocalStmt):
            r = self._cf_local(stmt, run_vars, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, StaticStmt):
            r = self._cf_static(stmt, run_vars, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, SharedStmt):
            r = self._cf_shared(stmt, run_vars, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, OnErrorStmt):
            r = self._cf_on_error(stmt, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, ResumeStmt):
            r = self._cf_resume(stmt, sorted_lines, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, ErrorStmt):
            r = self._cf_error(stmt, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, AssertStmt):
            r = self._cf_assert(stmt, run_vars, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, StopStmt):
            r = self._cf_stop(stmt, sorted_lines, ip, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, OnMeasureStmt):
            r = self._cf_on_measure(stmt, parsed=parsed)
            if r is not None:
                return r[1]
        if isinstance(parsed, OnTimerStmt):
            r = self._cf_on_timer(stmt, parsed=parsed)
            if r is not None:
                return r[1]

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
        from qbasic_core.parser import _split_colon_stmts
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
            raise RuntimeError(f"RECURSIVE SUBROUTINE: {word} calls itself")
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
        word = stmt.split()[0].upper() if stmt.split() else ''
        # Also check for parenthesized call syntax: NAME(args)
        m_paren = re.match(r'(\w+)\(', stmt)
        if m_paren:
            paren_name = m_paren.group(1).upper()
            if paren_name in self.subroutines:
                word = paren_name
        if word in self.subroutines:
            for sub_stmt in self._expand_statement(stmt, _call_stack):
                self._apply_gate_str(sub_stmt, qc, _call_stack, backend=backend)
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
        from qbasic_core.exec_context import ExecContext
        qc = QuantumCircuit(self.num_qubits)
        imm_ctx = ExecContext(sorted_lines=[0], ip=0,
                              run_vars=dict(self.variables), qc=qc)
        self._exec_line(line, ctx=imm_ctx)
        qc.save_statevector()
        backend = AerSimulator(method='statevector')
        result = backend.run(transpile(qc, backend)).result()
        sv = np.array(result.get_statevector())
        self.last_sv = sv
        self.last_circuit = qc
        self._print_sv_compact(sv)
