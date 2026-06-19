"""QUBASIC algorithm primitives and optimization.

Composable circuit building blocks applied to a qubit range, plus a classical
optimization driver. All run fully locally on the Qiskit/Aer path:

  QFT <range>            Quantum Fourier transform over a qubit range
  IQFT <range>           Inverse QFT
  DIFFUSE <range>        Grover diffusion operator (2|s><s| - I) over a range
  MCX c1,c2,...,t        Multi-controlled X (arbitrary control count)
  MCZ c1,c2,...,t        Multi-controlled Z
  MCP theta, c1,...,t    Multi-controlled phase
  QADD a0-a1, b0-b1      In-place adder: register A += register B (mod 2^n)
  QADDC k, a0-a1         In-place adder: register A += constant k (mod 2^n)
  QPE count, targ, NAME  Phase estimation of a custom UNITARY on a target register

A range is "lo-hi" (inclusive), a space/comma list of indices, or empty (all
qubits). These are Qiskit-path primitives; they raise in LOCC mode.

Optimization (commands, not circuit instructions):

  MINIMIZE v1[, v2 ...] -> cost   Minimize a program-computed cost over variables
  GRADIENT v1[, v2 ...] -> cost   Parameter-shift gradient of cost w.r.t. each var
"""

from __future__ import annotations

import re

import numpy as np


_RE_RANGE = re.compile(r'^(\d+)\s*-\s*(\d+)$')


class _NullIO:
    """Swallow output during inner optimization runs."""
    def write(self, text: str) -> None: pass
    def writeln(self, text: str) -> None: pass
    def read_line(self, prompt: str) -> str: return ''


class AlgorithmsMixin:
    """Algorithm primitives (QFT, QPE, diffusion, adders, multi-controlled
    gates) and a MINIMIZE/GRADIENT optimization driver for QBasicTerminal.

    Requires: TerminalProtocol — uses self.num_qubits, self.variables,
    self._custom_gates, self._eval_with_vars(), self._resolve_qubit(),
    self.cmd_run(), self.io.
    """

    # ── qubit-range parsing ────────────────────────────────────────────

    def _alg_qubit_list(self, spec: str, run_vars=None) -> list[int]:
        """Resolve a range spec to a validated list of qubit indices.

        Accepts 'lo-hi' (inclusive, ascending or descending), a space/comma
        separated list of expressions, or '' for all qubits.
        """
        spec = (spec or '').strip()
        if not spec:
            qs = list(range(self.num_qubits))
        else:
            m = _RE_RANGE.match(spec)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                qs = list(range(lo, hi + 1)) if lo <= hi else list(range(lo, hi - 1, -1))
            else:
                rv = run_vars if run_vars is not None else {}
                qs = [int(self._eval_with_vars(x, rv))
                      for x in spec.replace(',', ' ').split()]
        for q in qs:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(f"QUBIT {q} OUT OF RANGE (0-{self.num_qubits - 1})")
        return qs

    def _alg_guard_standard(self, name: str) -> None:
        if getattr(self, 'locc_mode', False):
            raise ValueError(f"{name} is not available in LOCC mode (standard Aer path only)")

    # ── QFT ─────────────────────────────────────────────────────────────

    def _emit_qft(self, qc, qs: list[int], inverse: bool = False, swaps: bool = True) -> None:
        """Apply a QFT (or its inverse) over the qubit list qs.

        qs is interpreted least-significant-first (qs[0] is the LSB), matching
        the statevector convention where qubit 0 is the low bit. With swaps=True
        the standard bit-reversal is included so QFT|x> matches the textbook
        transform and qubit qs[l] carries bit l of the transformed register.
        """
        qs = list(qs)[::-1]   # internal algorithm is MSB-first; input is LSB-first
        n = len(qs)
        if not inverse:
            for i in range(n):
                qc.h(qs[i])
                for j in range(i + 1, n):
                    qc.cp(np.pi / (2 ** (j - i)), qs[j], qs[i])
            if swaps:
                for i in range(n // 2):
                    qc.swap(qs[i], qs[n - 1 - i])
        else:
            if swaps:
                for i in range(n // 2):
                    qc.swap(qs[i], qs[n - 1 - i])
            for i in reversed(range(n)):
                for j in reversed(range(i + 1, n)):
                    qc.cp(-np.pi / (2 ** (j - i)), qs[j], qs[i])
                qc.h(qs[i])

    def _try_exec_qft(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        m = re.match(r'(IQFT|QFT)\b\s*(.*)', stmt, re.IGNORECASE)
        if not m:
            return False
        self._alg_guard_standard(m.group(1).upper())
        qs = self._alg_qubit_list(m.group(2), run_vars)
        self._emit_qft(qc, qs, inverse=m.group(1).upper() == 'IQFT')
        return True

    # ── Grover diffusion ─────────────────────────────────────────────────

    def _emit_mcz(self, qc, qs: list[int]) -> None:
        """Multi-controlled Z over qs (phase flip of the all-ones state)."""
        if len(qs) == 1:
            qc.z(qs[0])
        elif len(qs) == 2:
            qc.cz(qs[0], qs[1])
        else:
            t = qs[-1]
            qc.h(t)
            qc.mcx(qs[:-1], t)
            qc.h(t)

    def _try_exec_diffuse(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        m = re.match(r'(?:DIFFUSE|DIFFUSION)\b\s*(.*)', stmt, re.IGNORECASE)
        if not m:
            return False
        self._alg_guard_standard('DIFFUSE')
        qs = self._alg_qubit_list(m.group(1), run_vars)
        for q in qs:
            qc.h(q)
        for q in qs:
            qc.x(q)
        self._emit_mcz(qc, qs)
        for q in qs:
            qc.x(q)
        for q in qs:
            qc.h(q)
        return True

    # ── Multi-controlled gates ───────────────────────────────────────────

    def _try_exec_mcgate(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        m = re.match(r'(MCX|MCZ|MCP)\b\s*(.*)', stmt, re.IGNORECASE)
        if not m:
            return False
        self._alg_guard_standard(m.group(1).upper())
        name = m.group(1).upper()
        args = [a.strip() for a in m.group(2).replace(',', ' ').split() if a.strip()]
        theta = None
        if name == 'MCP':
            if not args:
                raise ValueError("MCP needs an angle, controls, and a target")
            theta = self._eval_with_vars(args[0], run_vars)
            args = args[1:]
        qs = [self._resolve_qubit(a) for a in args]
        if len(qs) < 2:
            raise ValueError(f"{name} needs at least one control and one target")
        controls, target = qs[:-1], qs[-1]
        if name == 'MCX':
            qc.mcx(controls, target)
        elif name == 'MCZ':
            qc.h(target)
            qc.mcx(controls, target)
            qc.h(target)
        else:
            qc.mcp(theta, controls, target)
        return True

    # ── QFT adders (Draper) ──────────────────────────────────────────────

    def _emit_qadd_phases(self, qc, a_qs: list[int], addend) -> None:
        """Apply the Fourier-basis phase rotations that add `addend` into a_qs.

        a_qs is the QFT'd register with qubit 0 of the range as the LSB (matching
        the statevector convention). With QFT|x> = (1/sqrt(N)) sum_k e^{2pi i x k/N}|k>,
        adding y multiplies amplitude |k> by e^{2pi i y k/N}; since k = sum_l k_l 2^l,
        that factors into a phase 2*pi*2^(l+m)/2^n on a_qs[l] per addend bit m.
        `addend` is a list of control qubits (same length, LSB-first) for register
        addition, or an int for constant addition.
        """
        n = len(a_qs)
        for l in range(n):              # a_qs[l] has weight 2^l (q0 = LSB)
            for m in range(n):          # addend bit m has weight 2^m
                d = n - l - m           # phases that are multiples of 2*pi are identity
                if d < 1:
                    continue
                angle = 2 * np.pi / (2 ** d)
                if isinstance(addend, int):
                    if (addend >> m) & 1:
                        qc.p(angle, a_qs[l])
                else:
                    qc.cp(angle, addend[m], a_qs[l])

    def _try_exec_qadd(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        m = re.match(r'QADD\b\s+(\S+(?:\s*-\s*\S+)?)\s*,\s*(\S+(?:\s*-\s*\S+)?)\s*$',
                     stmt, re.IGNORECASE)
        if not m:
            return False
        self._alg_guard_standard('QADD')
        a_qs = self._alg_qubit_list(m.group(1), run_vars)
        b_qs = self._alg_qubit_list(m.group(2), run_vars)
        if len(a_qs) != len(b_qs):
            raise ValueError("QADD needs two registers of equal size")
        self._emit_qft(qc, a_qs, inverse=False, swaps=True)
        self._emit_qadd_phases(qc, a_qs, b_qs)
        self._emit_qft(qc, a_qs, inverse=True, swaps=True)
        return True

    def _try_exec_qaddc(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        m = re.match(r'QADDC\b\s+(\S+)\s*,\s*(\S+(?:\s*-\s*\S+)?)\s*$', stmt, re.IGNORECASE)
        if not m:
            return False
        self._alg_guard_standard('QADDC')
        k = int(self._eval_with_vars(m.group(1), run_vars))
        a_qs = self._alg_qubit_list(m.group(2), run_vars)
        n = len(a_qs)
        k %= (1 << n)
        self._emit_qft(qc, a_qs, inverse=False, swaps=True)
        self._emit_qadd_phases(qc, a_qs, k)
        self._emit_qft(qc, a_qs, inverse=True, swaps=True)
        return True

    # ── Quantum phase estimation ─────────────────────────────────────────

    def _try_exec_qpe(self, stmt: str, qc, run_vars: dict, *, backend=None) -> bool:
        m = re.match(r'QPE\b\s+(\S+(?:\s*-\s*\S+)?)\s+(\S+(?:\s*-\s*\S+)?)\s+(\w+)\s*$',
                     stmt, re.IGNORECASE)
        if not m:
            return False
        self._alg_guard_standard('QPE')
        count_qs = self._alg_qubit_list(m.group(1), run_vars)
        targ_qs = self._alg_qubit_list(m.group(2), run_vars)
        uname = m.group(3).upper()
        if uname not in self._custom_gates:
            raise ValueError(f"QPE: unknown UNITARY '{uname}' (define one with UNITARY {uname} = ...)")
        U = self._custom_gates[uname]
        if 2 ** len(targ_qs) != U.shape[0]:
            raise ValueError(
                f"QPE: UNITARY {uname} is {int(np.log2(U.shape[0]))}-qubit but the "
                f"target register has {len(targ_qs)} qubit(s)")
        from qiskit.circuit.library import UnitaryGate
        for cq in count_qs:
            qc.h(cq)
        # Counting qubit count_qs[l] controls U^(2^l), so after the inverse QFT
        # it holds bit l of the phase estimate (count_qs[0] = LSB, matching the
        # statevector convention). The estimate is then value/2^m.
        for k, cq in enumerate(count_qs):
            Uk = np.linalg.matrix_power(U, 2 ** k)
            qc.append(UnitaryGate(Uk).control(1), [cq] + list(targ_qs))
        self._emit_qft(qc, count_qs, inverse=True, swaps=True)
        return True

    # ── Optimization: MINIMIZE / GRADIENT ────────────────────────────────

    def _alg_eval_cost(self, params: list[str], vec, cost_expr: str) -> float:
        """Set params from vec, RUN silently, then evaluate the cost expression.

        The cost is evaluated AFTER the run, so it sees variables the program
        produced via SAVE_EXPECT/SAVE_PROBS during this run (a plain variable
        name works, as does a combination like 'z0 + 0.5*z1').
        """
        for name, val in zip(params, vec):
            self.variables[name] = float(val)
        old_io = self.io
        self.io = _NullIO()
        try:
            self.cmd_run()
        finally:
            self.io = old_io
        try:
            val = self._safe_eval(cost_expr)
        except Exception as e:
            raise ValueError(
                f"cannot evaluate cost '{cost_expr}' after RUN ({e}); have the "
                f"program set its terms (e.g. SAVE_EXPECT ... -> z0)")
        return float(np.real(complex(val)))

    @staticmethod
    def _parse_opt_args(rest: str):
        """Parse 'v1, v2 -> <cost expr> [ITERS n] [STEP s]'.

        Returns (params, cost_expr, opts). The cost may be any expression in the
        variables the program sets, not just a single name.
        """
        m = re.match(r'(.+?)\s*->\s*(.+)$', rest.strip(), re.IGNORECASE)
        if not m:
            return None
        params = [p.strip() for p in m.group(1).split(',') if p.strip()]
        tail = m.group(2)
        opts: dict = {}
        im = re.search(r'\s+ITERS\s+(\d+)\s*', tail, re.IGNORECASE)
        if im:
            opts['iters'] = int(im.group(1))
            tail = tail[:im.start()] + ' ' + tail[im.end():]
        sm = re.search(r'\s+STEP\s+([0-9.eE+-]+)\s*', tail, re.IGNORECASE)
        if sm:
            opts['step'] = float(sm.group(1))
            tail = tail[:sm.start()] + ' ' + tail[sm.end():]
        cost_expr = tail.strip()
        if not cost_expr:
            return None
        return params, cost_expr, opts

    def _nelder_mead(self, f, x0: list[float], iters: int, step: float):
        """Compact Nelder-Mead simplex minimizer (dependency-free, offline)."""
        n = len(x0)
        x0 = [float(v) for v in x0]
        if n == 0:
            return x0, f(x0)
        # Build initial simplex
        simplex = [list(x0)]
        for i in range(n):
            pt = list(x0)
            pt[i] += step if pt[i] == 0 else step * (1.0 + abs(pt[i]))
            simplex.append(pt)
        fvals = [f(p) for p in simplex]
        a, g, r, s = 1.0, 2.0, 0.5, 0.5  # reflect, expand, contract, shrink
        for _ in range(iters):
            order = sorted(range(n + 1), key=lambda k: fvals[k])
            simplex = [simplex[k] for k in order]
            fvals = [fvals[k] for k in order]
            if abs(fvals[-1] - fvals[0]) < 1e-9:
                break
            centroid = [sum(simplex[k][d] for k in range(n)) / n for d in range(n)]
            worst = simplex[-1]
            xr = [centroid[d] + a * (centroid[d] - worst[d]) for d in range(n)]
            fr = f(xr)
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[0]:
                xe = [centroid[d] + g * (xr[d] - centroid[d]) for d in range(n)]
                fe = f(xe)
                simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
            else:
                xc = [centroid[d] + r * (worst[d] - centroid[d]) for d in range(n)]
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    best = simplex[0]
                    for k in range(1, n + 1):
                        simplex[k] = [best[d] + s * (simplex[k][d] - best[d]) for d in range(n)]
                        fvals[k] = f(simplex[k])
        order = sorted(range(n + 1), key=lambda k: fvals[k])
        return simplex[order[0]], fvals[order[0]]

    def cmd_minimize(self, rest: str) -> None:
        """MINIMIZE v1[, v2 ...] -> cost [ITERS n] [STEP s].

        Runs a classical (Nelder-Mead) optimizer over the listed variables,
        re-RUNning the program each step and reading the program-computed
        `cost` variable. This is the VQE/QAOA driver: build a parametrized
        ansatz, SAVE_EXPECT the Hamiltonian into `cost`, then MINIMIZE.
        """
        if not self.program:
            self.io.writeln("?NOTHING TO MINIMIZE — enter a program first")
            return
        parsed = self._parse_opt_args(rest)
        if not parsed:
            self.io.writeln("?USAGE: MINIMIZE <var>[, <var> ...] -> <cost> [ITERS n] [STEP s]")
            return
        params, cost_expr, opts = parsed
        if not params:
            self.io.writeln("?MINIMIZE needs at least one variable")
            return
        iters = opts.get('iters', 100)
        step = opts.get('step', 0.5)
        x0 = [float(self.variables.get(p, 0.0)) for p in params]
        evals = [0]

        def f(vec):
            evals[0] += 1
            return self._alg_eval_cost(params, vec, cost_expr)

        try:
            best, fval = self._nelder_mead(f, x0, iters, step)
        except Exception as e:
            self.io.writeln(f"?MINIMIZE ERROR: {e}")
            return
        for name, val in zip(params, best):
            self.variables[name] = float(val)
        self.variables['_COST'] = float(fval)
        self.io.writeln(f"\n  MINIMIZE converged ({evals[0]} evaluations):")
        for name, val in zip(params, best):
            self.io.writeln(f"    {name} = {val:.6f}")
        self.io.writeln(f"    cost ({cost_expr}) = {fval:.6f}")

    def cmd_gradient(self, rest: str) -> None:
        """GRADIENT v1[, v2 ...] -> cost — parameter-shift gradient of cost.

        For each variable, evaluates cost at +pi/2 and -pi/2 shifts and reports
        d(cost)/d(var) = (cost(+) - cost(-)) / 2, the exact parameter-shift rule
        for cost functions that are expectation values of standard rotations.
        """
        if not self.program:
            self.io.writeln("?NOTHING TO DIFFERENTIATE — enter a program first")
            return
        parsed = self._parse_opt_args(rest)
        if not parsed:
            self.io.writeln("?USAGE: GRADIENT <var>[, <var> ...] -> <cost>")
            return
        params, cost_expr, _ = parsed
        base = [float(self.variables.get(p, 0.0)) for p in params]
        shift = np.pi / 2
        self.io.writeln(f"\n  Parameter-shift gradient of {cost_expr}:")
        grad = {}
        for i, name in enumerate(params):
            plus = list(base); plus[i] += shift
            minus = list(base); minus[i] -= shift
            fp = self._alg_eval_cost(params, plus, cost_expr)
            fm = self._alg_eval_cost(params, minus, cost_expr)
            g = (fp - fm) / 2.0
            grad[name] = g
            self.io.writeln(f"    d/d({name}) = {g:+.6f}")
        # Restore the original parameter values.
        for name, val in zip(params, base):
            self.variables[name] = val
        for name, g in grad.items():
            self.variables[f'_GRAD_{name}'] = g
