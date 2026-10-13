"""QUBASIC algorithms: circuit primitives, optimizers (MINIMIZE,
GRADIENT), phase and amplitude estimation, quantum machine learning, oracle
algorithms, QAOA and VQE, Shor's factoring and HHL.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, MutableMapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np

from qubasic_core.host import HostState

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    from qubasic_core.engine import CircuitBackend


# ---------------------------------------------------------------------------
# QUBASIC circuit primitives: QFT, multi-controlled gates, Grover diffusion,
# Fourier-basis adders, phase estimation, quantum walks and graph states.
# ---------------------------------------------------------------------------

_RE_RANGE = re.compile(r'^(\d+)\s*-\s*(\d+)$')


class PrimitivesMixin(HostState):
    """Program statements that expand to standard subcircuits."""

    def _alg_qubit_list(self, spec: str, run_vars: MutableMapping[str, Any] | None = None) -> list[int]:
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

    def _emit_qft(self, qc: QuantumCircuit, qs: list[int], inverse: bool = False, swaps: bool = True) -> None:
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

    def _try_exec_qft(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
        m = re.match(r'(IQFT|QFT)\b\s*(.*)', stmt, re.IGNORECASE)
        if not m:
            return False
        self._alg_guard_standard(m.group(1).upper())
        qs = self._alg_qubit_list(m.group(2), run_vars)
        self._emit_qft(qc, qs, inverse=m.group(1).upper() == 'IQFT')
        return True

    def _emit_mcz(self, qc: QuantumCircuit, qs: list[int]) -> None:
        """Multi-controlled Z over qs (phase flip of the all-ones state)."""
        if not qs:
            raise ValueError("MCZ needs at least one qubit")
        if len(qs) == 1:
            qc.z(qs[0])
        elif len(qs) == 2:
            qc.cz(qs[0], qs[1])
        else:
            t = qs[-1]
            qc.h(t)
            qc.mcx(qs[:-1], t)
            qc.h(t)

    def _try_exec_diffuse(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
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

    def _try_exec_mcgate(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
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

    def _emit_qadd_phases(self, qc: QuantumCircuit, a_qs: list[int], addend: list[int] | int) -> None:
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

    def _try_exec_qadd(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
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

    def _try_exec_qaddc(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
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

    def _try_exec_qpe(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
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

    def cmd_qwalk(self, rest: str) -> None:
        """QWALK <steps> — discrete-time quantum walk on a cycle of 2^(n-1) nodes.

        Uses qubit 0 as the coin and qubits 1..n-1 as the position register
        (binary), applying a Hadamard coin and a coin-controlled increment/
        decrement shift each step, then shows the position distribution."""
        try:
            steps = int(rest.strip()) if rest.strip() else 5
        except ValueError:
            self.io.writeln("?USAGE: QWALK <steps>")
            return
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        n = self.num_qubits
        if n < 2:
            self.io.writeln("?QWALK needs at least 2 qubits (1 coin + position)")
            return
        pos = list(range(1, n))
        m = len(pos)
        qc = QuantumCircuit(n, m)
        # Start in the middle of the line.
        mid = (2 ** m) // 2
        for j in range(m):
            if (mid >> j) & 1:
                qc.x(pos[j])
        for _ in range(steps):
            qc.h(0)                       # Hadamard coin
            # coin=1 -> increment position (mod 2^m): controlled increment
            self._controlled_increment(qc, 0, pos, +1)
            qc.x(0)
            self._controlled_increment(qc, 0, pos, -1)
            qc.x(0)
        qc.measure(pos, range(m))
        backend = AerSimulator()
        kw = {'seed_simulator': self._seed} if self._seed is not None else {}
        counts = backend.run(transpile(qc, backend), shots=max(2000, self.shots), **kw).result().get_counts()
        self.io.writeln(f"\n  Quantum walk, {steps} steps, {2 ** m} positions:")
        total = sum(counts.values())
        for state in sorted(counts, key=lambda s: int(s, 2)):
            p = counts[state] / total
            bar = '#' * int(40 * p)
            self.io.writeln(f"    pos {int(state, 2):>3}  {p:6.3f}  {bar}")

    def _controlled_increment(self, qc: QuantumCircuit, ctrl: int, pos: list[int], sign: int) -> None:
        """Coin-controlled +-1 modular increment of the position register.

        pos[0] is the LSB. Increment is a ripple of multi-controlled X gates;
        decrement is the increment conjugated by X on every position qubit.
        """
        m = len(pos)
        if sign < 0:
            for q in pos:
                qc.x(q)
            self._controlled_increment(qc, ctrl, pos, +1)
            for q in pos:
                qc.x(q)
            return
        for k in range(m - 1, -1, -1):
            controls = [ctrl] + pos[:k]
            if len(controls) == 1:
                qc.cx(ctrl, pos[k])
            else:
                qc.mcx(controls, pos[k])

    def _try_exec_graphstate(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
        m = re.match(r'GRAPHSTATE\s+(.*)$', stmt, re.IGNORECASE)
        if not m:
            return False
        if getattr(self, 'locc_mode', False):
            raise ValueError("GRAPHSTATE is not available in LOCC mode")
        spec = m.group(1).strip()
        n = self.num_qubits
        if spec.upper() in ('LINE', 'CHAIN', ''):
            pairs = [(i, i + 1) for i in range(n - 1)]
        elif spec.upper() == 'RING':
            pairs = [(i, (i + 1) % n) for i in range(n)]
        elif spec.upper() == 'COMPLETE':
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            pairs = []
            for tok in spec.replace(',', ' ').split():
                a, _, b = tok.partition('-')
                if not (a.strip().isdigit() and b.strip().isdigit()):
                    raise ValueError(f"GRAPHSTATE edges are a-b pairs, not '{tok}'")
                pairs.append((int(a), int(b)))
        edges: list[tuple[int, int]] = []
        for a, b in pairs:                 # each edge once (a 2-qubit RING has one)
            if not (0 <= a < n and 0 <= b < n) or a == b:
                raise ValueError(f"GRAPHSTATE edge {a}-{b} needs two different qubits in 0-{n - 1}")
            if (min(a, b), max(a, b)) not in edges:
                edges.append((min(a, b), max(a, b)))
        for q in range(n):
            qc.h(q)
        for a, b in edges:
            qc.cz(a, b)
        return True


# ---------------------------------------------------------------------------
# QUBASIC classical optimization over programs: MINIMIZE and GRADIENT.
# ---------------------------------------------------------------------------

class OptimizeMixin(HostState):
    """Nelder-Mead, SPSA and gradient descent over program variables."""

    def _alg_eval_cost(self, params: list[str], vec: Iterable[float], cost_expr: str) -> float:
        """Set params from vec, RUN silently, then evaluate the cost expression.

        The cost is evaluated AFTER the run, so it sees variables the program
        produced via SAVE_EXPECT/SAVE_PROBS during this run (a plain variable
        name works, as does a combination like 'z0 + 0.5*z1').
        """
        for name, val in zip(params, vec):
            self.variables[name] = float(val)
        with self._silenced():
            self.cmd_run()
        try:
            val = self._safe_eval(cost_expr)
        except Exception as e:
            raise ValueError(
                f"cannot evaluate cost '{cost_expr}' after RUN ({e}); have the "
                f"program set its terms (e.g. SAVE_EXPECT ... -> z0)")
        return float(np.real(complex(val)))

    def _alg_parametric_setup(self, params: list[str], sampling: bool = False) -> dict[str, Any] | None:
        """Compile the program ONCE with qiskit Parameters bound to the given
        variables. Per-evaluation cost then binds values into the cached,
        already-transpiled circuit instead of rebuilding and retranspiling —
        an order of magnitude on variational loops.

        Returns an evaluation context, or None when the program is not
        parameter-compatible (nonlinear expressions the symbolic path cannot
        carry, feedforward bits, LOCC mode, or the wrong measure/no-measure
        mode for the caller), in which case callers fall back to rebuild.
        """
        if getattr(self, 'locc_mode', False):
            return None
        from qiskit.circuit import Parameter
        saved = {p: self.variables.get(p) for p in params}
        pobjs = {p: Parameter(p) for p in params}
        try:
            for p in params:
                self.variables[p] = pobjs[p]
            with self._silenced():
                qc, has_measure = self.build_circuit()
            if getattr(self, '_classical_bits', None):
                return None
            if sampling != bool(has_measure):
                return None
            if not qc.parameters:
                return None
            if sampling:
                qc = qc.copy()
                qc.measure_all()
            from qiskit_aer import AerSimulator
            method = 'statevector' if self.sim_method == 'automatic' else self.sim_method
            backend = AerSimulator(**self._build_backend_opts(method))
            qc_t = self._transpile_routed(qc, backend)
            pmap = {p: pobjs[p] for p in params if pobjs[p] in qc_t.parameters}
            if not pmap:
                return None
            noisy = bool(self._noise_model or getattr(self, '_qubit_noise', None))
            shots = self.shots if (sampling or noisy) else 1
            return {'qc': qc_t, 'backend': backend, 'pmap': pmap, 'shots': shots}
        except Exception:
            return None
        finally:
            for p, v in saved.items():
                if v is None:
                    self.variables.pop(p, None)
                else:
                    self.variables[p] = v

    def _alg_eval_cost_parametric(self, ctx: dict, params: list[str], vec: Iterable[float],
                                  cost_expr: str) -> float:
        """Bind values into the precompiled circuit, run, evaluate the cost."""
        binding = {ctx['pmap'][p]: float(v) for p, v in zip(params, vec)
                   if p in ctx['pmap']}
        bound = ctx['qc'].assign_parameters(binding)
        kw: dict = {'shots': ctx['shots']}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        result = ctx['backend'].run(bound, **kw).result()
        for name, val in zip(params, vec):
            self.variables[name] = float(val)
        self._extract_save_results(result)
        val = self._safe_eval(cost_expr)
        return float(np.real(complex(val)))

    @staticmethod
    def _parse_opt_args(rest: str) -> tuple[list[str], str, dict[str, Any]] | None:
        """Parse 'v1, v2 -> <cost expr> [ITERS n] [STEP s] [METHOD NM|SPSA|GRAD]'.

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
        mm = re.search(r'\s+METHOD\s+([A-Za-z-]+)\s*', tail, re.IGNORECASE)
        if mm:
            opts['method'] = mm.group(1).upper()
            tail = tail[:mm.start()] + ' ' + tail[mm.end():]
        cost_expr = tail.strip()
        if not cost_expr:
            return None
        return params, cost_expr, opts

    def _nelder_mead(self, f: Callable[[list[float]], float], x0: list[float], iters: int, step: float) -> tuple[list[float], float]:
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

    def _spsa(self, f: Callable[[list[float]], float], x0: list[float], iters: int, step: float) -> tuple[list[float], float]:
        """Simultaneous-perturbation stochastic approximation (Spall gains).

        Two cost evaluations per iteration regardless of dimension, robust to
        shot noise — the standard optimizer for noisy VQE landscapes."""
        rng = np.random.default_rng(self._seed)
        x = np.array([float(v) for v in x0])
        A = max(1.0, iters / 10.0)
        alpha, gamma, c0 = 0.602, 0.101, 0.1
        best_x, best_f = x.copy(), float('inf')
        for k in range(iters):
            ak = step / (k + 1 + A) ** alpha
            ck = c0 / (k + 1) ** gamma
            delta = rng.choice([-1.0, 1.0], size=x.size)
            fp = f(list(x + ck * delta))
            fm = f(list(x - ck * delta))
            for cand_x, cand_f in ((x + ck * delta, fp), (x - ck * delta, fm)):
                if cand_f < best_f:
                    best_x, best_f = cand_x.copy(), cand_f
            ghat = (fp - fm) / (2.0 * ck) * delta
            x = x - ak * ghat
        f_final = f(list(x))
        if f_final <= best_f:
            return list(x), f_final
        return list(best_x), best_f

    def _grad_descent(self, f: Callable[[list[float]], float], x0: list[float], iters: int, step: float) -> tuple[list[float], float]:
        """Gradient descent via the parameter-shift rule (pi/2 shifts).

        Exact gradients for costs built from standard rotation gates; the
        learning rate adapts (grow on improvement, halve on overshoot)."""
        x = np.array([float(v) for v in x0])
        shift = np.pi / 2.0
        fx = f(list(x))
        lr = step
        for _ in range(iters):
            g = np.zeros_like(x)
            for i in range(x.size):
                xp = x.copy()
                xp[i] += shift
                xm = x.copy()
                xm[i] -= shift
                g[i] = (f(list(xp)) - f(list(xm))) / 2.0
            if float(np.linalg.norm(g)) < 1e-7:
                break
            xn = x - lr * g
            fn = f(list(xn))
            if fn <= fx:
                x, fx = xn, fn
                lr *= 1.1
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        return list(x), fx

    def cmd_minimize(self, rest: str) -> None:
        """MINIMIZE v1[, v2 ...] -> cost [ITERS n] [STEP s] [METHOD NM|SPSA|GRAD] — minimize a program-computed cost.

        Runs a classical optimizer (Nelder-Mead by default) over the listed
        variables, re-running the program each step and evaluating `cost`
        afterwards. This is the VQE/QAOA driver: build a parametrized
        ansatz, SAVE_EXPECT the Hamiltonian into `cost`, then MINIMIZE."""
        if not self.program:
            self.io.writeln("?NOTHING TO MINIMIZE — enter a program first")
            return
        parsed = self._parse_opt_args(rest)
        if not parsed:
            self.io.writeln("?USAGE: MINIMIZE <var>[, <var> ...] -> <cost> [ITERS n] [STEP s] [METHOD NM|SPSA|GRAD]")
            return
        params, cost_expr, opts = parsed
        if not params:
            self.io.writeln("?MINIMIZE needs at least one variable")
            return
        iters = opts.get('iters', 100)
        step = opts.get('step', 0.5)
        opt_name = opts.get('method', 'NM')
        optimizer = {'NM': self._nelder_mead, 'NELDER-MEAD': self._nelder_mead,
                     'SPSA': self._spsa,
                     'GRAD': self._grad_descent, 'GD': self._grad_descent}.get(opt_name)
        if optimizer is None:
            self.io.writeln(f"?UNKNOWN OPTIMIZER '{opt_name}' (use NM, SPSA, or GRAD)")
            return
        x0 = [float(self.variables.get(p, 0.0)) for p in params]
        evals = [0]
        # Compile once with bound Parameters when the ansatz permits; every
        # evaluation then re-binds instead of rebuilding and retranspiling.
        ctx = self._alg_parametric_setup(params)

        if ctx is not None:
            def f(vec: list[float]) -> float:
                evals[0] += 1
                return self._alg_eval_cost_parametric(ctx, params, vec, cost_expr)
        else:
            def f(vec: list[float]) -> float:
                evals[0] += 1
                return self._alg_eval_cost(params, vec, cost_expr)

        try:
            best, fval = optimizer(f, x0, iters, step)
        except Exception as e:
            self.io.writeln(f"?MINIMIZE ERROR: {e}")
            return
        for name, val in zip(params, best):
            self.variables[name] = float(val)
        self.variables['_COST'] = float(fval)
        _how = f"{evals[0]} evaluations, {opt_name.lower()}" \
            + (", parametric compile" if ctx is not None else "")
        self.io.writeln(f"\n  MINIMIZE converged ({_how}):")
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
        ctx = self._alg_parametric_setup(params)
        if ctx is not None:
            eval_cost = lambda vec: self._alg_eval_cost_parametric(ctx, params, vec, cost_expr)
        else:
            eval_cost = lambda vec: self._alg_eval_cost(params, vec, cost_expr)
        self.io.writeln(f"\n  Parameter-shift gradient of {cost_expr}:")
        grad = {}
        for i, name in enumerate(params):
            plus = list(base)
            plus[i] += shift
            minus = list(base)
            minus[i] -= shift
            fp = eval_cost(plus)
            fm = eval_cost(minus)
            g = (fp - fm) / 2.0
            grad[name] = g
            self.io.writeln(f"    d/d({name}) = {g:+.6f}")
        # Restore the original parameter values.
        for name, val in zip(params, base):
            self.variables[name] = val
        for name, g in grad.items():
            self.variables[f'_GRAD_{name}'] = g


# ---------------------------------------------------------------------------
# QUBASIC phase and amplitude estimation: IQPE, AMPEST and AMPLIFY.
# ---------------------------------------------------------------------------

class EstimationMixin(HostState):
    """Iterative phase estimation, maximum-likelihood amplitude estimation, amplitude amplification."""

    def cmd_iqpe(self, rest: str) -> None:
        """IQPE <bits> <target_range> <UNITARY> — iterative phase estimation of a UNITARY eigenphase, one bit per round.

        Estimates the eigenphase of a UNITARY (target prepared in an eigenstate)
        one bit at a time with a single ancilla, to `bits` of precision."""
        parts = rest.split()
        if len(parts) < 3:
            self.io.writeln("?USAGE: IQPE <bits> <target> <UNITARY>")
            return
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import UnitaryGate
        from qiskit_aer import AerSimulator
        try:
            bits = int(parts[0])
            targ = self._alg_qubit_list(parts[1])
            uname = parts[2].upper()
            if uname not in self._custom_gates:
                raise ValueError(f"unknown UNITARY '{uname}'")
            U = self._custom_gates[uname]
            if not self.program:
                self.io.writeln("  (no program: the target starts in |0...0>, which reads "
                                "phase 0 for a diagonal unitary — add prep gates, "
                                "e.g. 10 X 0, to select another eigenstate)")
            prep, _ = self.build_circuit()    # the program prepares the target eigenstate
            nq = self.num_qubits
            anc = nq                          # dedicated ancilla beyond the system
            backend = AerSimulator(noise_model=self._noise_model) if self._noise_model else AerSimulator()
            phase_bits = [0] * bits
            # Estimate from least significant bit upward, feeding back the phase.
            for k in range(bits - 1, -1, -1):
                qc = QuantumCircuit(nq + 1, 1)
                qc.compose(prep, qubits=range(nq), inplace=True)
                qc.h(anc)
                Uk = np.linalg.matrix_power(U, 2 ** k)
                qc.append(UnitaryGate(Uk).control(1), [anc] + list(targ))
                fb = sum(phase_bits[j] / (2 ** (j - k + 1)) for j in range(k + 1, bits))
                qc.p(-2 * np.pi * fb, anc)
                qc.h(anc)
                qc.measure(anc, 0)
                kw = {'seed_simulator': self._seed + k} if self._seed is not None else {}
                counts = backend.run(transpile(qc, backend), shots=400, **kw).result().get_counts()
                phase_bits[k] = 1 if counts.get('1', 0) > counts.get('0', 0) else 0
            phase = sum(b / 2 ** (i + 1) for i, b in enumerate(phase_bits))
            self.io.writeln(f"  IQPE phase = {phase:.6f}  (bits {''.join(map(str, phase_bits))})")
            self.variables['_IQPE'] = phase
        except Exception as e:
            self.io.writeln(f"?IQPE ERROR: {e}")

    def cmd_ampest(self, rest: str) -> None:
        """AMPEST [m] <good> [SHOTS n] — amplitude estimation of the program's state.

        The program (measure-free and unitary) is the state preparation A; with
        no program, A is a Hadamard on every qubit. <good> marks the good
        outcomes: a qubit index (good when it reads 1) or a basis state |bits>
        in the display's bit order. Maximum-likelihood amplitude estimation
        runs Q^k A|0> for k = 0, 1, 2, 4, ..., 2^(m-1) Grover iterations
        (Q = A S0 A^dag S_good, default m = 5), counts the good outcomes, and
        maximizes the joint likelihood over theta with a = sin^2(theta). The
        reported spread is the Fisher-information standard deviation."""
        parts = rest.split()
        shots = self.shots
        if any(p.upper() == 'SHOTS' for p in parts):
            k = [p.upper() for p in parts].index('SHOTS')
            shots = int(parts[k + 1])
            parts = parts[:k] + parts[k + 2:]
        if not parts:
            self.io.writeln("?USAGE: AMPEST [m] <qubit | |bits>> [SHOTS n]")
            return
        if len(parts) == 1:
            m, good_spec = 5, parts[0]
        else:
            m, good_spec = int(parts[0]), parts[1]
        try:
            a_hat, sigma, exact, ks = self._mlae(good_spec, m, shots)
            ks_txt = ','.join(str(k) for k in ks)
            exact_txt = f"; exact {exact:.4f}" if exact is not None else ""
            self.io.writeln(f"  Amplitude estimate a = {a_hat:.4f} ± {sigma:.4f}  "
                            f"(MLAE, Grover powers {ks_txt}, {shots} shots each{exact_txt})")
            self.variables['_AMPEST'] = a_hat
            self.variables['_AMPEST_SIGMA'] = sigma
        except Exception as e:
            self.io.writeln(f"?AMPEST ERROR: {e}")

    def _mlae(self, good_spec: str, m: int, shots: int) -> tuple[float, float, float | None, list[int]]:
        """Maximum-likelihood amplitude estimation (Suzuki et al., 2020).

        Returns (a_hat, sigma_a, exact_a or None, grover_powers)."""
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import Gate
        if m < 0:
            raise ValueError("m must be >= 0")
        n = self.num_qubits
        if self.program:
            prep, _ = self.build_circuit()
            for inst in prep.data:
                op = inst.operation
                if op.name == 'barrier':
                    continue
                if not isinstance(op, Gate):
                    raise ValueError(f"the state preparation must be unitary; '{op.name}' "
                                     f"is not (AMPEST takes a measure-free, reset-free program)")
        else:
            prep = QuantumCircuit(n)
            for q in range(n):
                prep.h(q)
        prep = prep.copy()
        prep.remove_final_measurements(inplace=True)
        spec = good_spec.strip()
        target_bits = None
        good_qubit = None
        if spec.startswith('|'):
            bits = spec.strip('|>⟩')
            if len(bits) != n or set(bits) - {'0', '1'}:
                raise ValueError(f"good state must be {n} bits, e.g. |{'1' * n}>")
            target_bits = bits[::-1] if self._endian_big else bits     # internal q_{n-1}..q0
        else:
            good_qubit = int(self._eval_with_vars(spec, {}))
            if not 0 <= good_qubit < n:
                raise ValueError(f"qubit {good_qubit} out of range (0-{n - 1})")
        qubits = list(range(n))

        def phase_flip(circ: QuantumCircuit, bits_hi_to_lo: str) -> None:
            zeros = [q for q in qubits if bits_hi_to_lo[n - 1 - q] == '0']
            for q in zeros:
                circ.x(q)
            self._emit_mcz(circ, qubits)
            for q in zeros:
                circ.x(q)

        def s_good(circ: QuantumCircuit) -> None:
            if good_qubit is not None:
                circ.z(good_qubit)
            elif target_bits is not None:
                phase_flip(circ, target_bits)

        a_inv = prep.inverse()
        grover = QuantumCircuit(n)
        s_good(grover)
        grover.compose(a_inv, inplace=True)
        phase_flip(grover, '0' * n)
        grover.compose(prep, inplace=True)

        ks = [0] + [2 ** j for j in range(m)]
        circuits = []
        for k in ks:
            qc = QuantumCircuit(n, 1 if good_qubit is not None else n)
            qc.compose(prep, inplace=True)
            for _ in range(k):
                qc.compose(grover, inplace=True)
            if good_qubit is not None:
                qc.measure(good_qubit, 0)
            else:
                qc.measure(qubits, list(range(n)))
            circuits.append(qc)
        method, precision = self._select_method(circuits[-1])
        backend = self._make_backend(method if method != 'stabilizer' else 'automatic',
                                     include_noise=True, precision=precision)
        kw: dict = {'shots': shots}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        result = backend.run(transpile(circuits, backend, optimization_level=self._transpile_opt_level),
                             **kw).result()
        hits = []
        for i in range(len(ks)):
            counts = result.get_counts(i)
            if good_qubit is not None:
                hits.append(counts.get('1', 0))
            else:
                hits.append(counts.get(target_bits, 0))
        hits_a = np.array(hits, dtype=float)
        odd = np.array([2 * k + 1 for k in ks], dtype=float)

        def loglik(theta: float) -> float:
            s = np.clip(np.sin(odd * theta) ** 2, 1e-12, 1 - 1e-12)
            return float(np.sum(hits_a * np.log(s) + (shots - hits_a) * np.log(1 - s)))

        grid = np.linspace(0.0, np.pi / 2, 20001)
        vals = np.array([loglik(t) for t in grid])
        best = int(np.argmax(vals))
        lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
        gr = (np.sqrt(5) - 1) / 2                      # golden-section refinement
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        for _ in range(60):
            if loglik(c) > loglik(d):
                hi = d
            else:
                lo = c
            c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        theta = (lo + hi) / 2
        a_hat = float(np.sin(theta) ** 2)
        fisher = 4.0 * shots * float(np.sum(odd ** 2))
        sigma = float(abs(np.sin(2 * theta)) / np.sqrt(fisher))
        exact = None
        from qubasic_core import simtools as cap
        if cap.fits(cap.aer_peak_bytes(n, 'statevector', 'double', 'end')) and n <= 26:
            from qiskit.quantum_info import Statevector
            probs = np.abs(Statevector(prep).data) ** 2
            if good_qubit is not None:
                idx = np.arange(probs.size)
                exact = float(probs[((idx >> good_qubit) & 1) == 1].sum())
            elif target_bits is not None:
                exact = float(probs[int(target_bits, 2)])
        return a_hat, sigma, exact, ks

    def _try_exec_amplify(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
        m = re.match(r'AMPLIFY\s+([01]+)\s*$', stmt, re.IGNORECASE)
        if not m:
            return False
        if getattr(self, 'locc_mode', False):
            raise ValueError("AMPLIFY is not available in LOCC mode")
        target = m.group(1)
        n = self.num_qubits
        if len(target) != n:
            raise ValueError(f"AMPLIFY target must be {n} bits, got {len(target)}")
        # The target reads in the active display convention, so what you see in
        # the histogram is what you type here; internally it stays q_{n-1}..q0.
        if getattr(self, '_endian_big', False):
            target = target[::-1]
        # Oracle: phase-flip |target> by X-conjugating the all-ones MCZ.
        qubits = list(range(n))
        idx = {q: n - 1 - q for q in qubits}   # bit position of qubit q in target
        zero_qubits = [q for q in qubits if target[idx[q]] == '0']
        for q in zero_qubits:
            qc.x(q)
        self._emit_mcz(qc, qubits)
        for q in zero_qubits:
            qc.x(q)
        # Diffusion about the uniform state.
        for q in qubits:
            qc.h(q)
        for q in qubits:
            qc.x(q)
        self._emit_mcz(qc, qubits)
        for q in qubits:
            qc.x(q)
        for q in qubits:
            qc.h(q)
        return True


# ---------------------------------------------------------------------------
# QUBASIC quantum machine learning: FEATUREMAP and QKERNEL.
# ---------------------------------------------------------------------------

class QMLMixin(HostState):
    """Feature-map circuits and quantum kernel matrices."""

    def _emit_feature_map(self, qc: QuantumCircuit, x: Sequence[float], reps: int = 2) -> None:
        """ZZ feature map: H + RZ(2 x_i) per qubit, RZZ(2(pi-x_i)(pi-x_j)) on pairs."""
        n = len(x)
        for _ in range(reps):
            for i in range(n):
                qc.h(i)
                qc.p(2.0 * x[i], i)
            for i in range(n - 1):
                ang = 2.0 * (np.pi - x[i]) * (np.pi - x[i + 1])
                qc.cx(i, i + 1)
                qc.p(ang, i + 1)
                qc.cx(i, i + 1)

    def _try_exec_featuremap(self, stmt: str, qc: QuantumCircuit, run_vars: MutableMapping[str, Any], *, backend: CircuitBackend | None = None) -> bool:
        m = re.match(r'FEATUREMAP\s+(.*)$', stmt, re.IGNORECASE)
        if not m:
            return False
        if getattr(self, 'locc_mode', False):
            raise ValueError("FEATUREMAP is not available in LOCC mode")
        x = [float(self._eval_with_vars(t, run_vars)) for t in m.group(1).replace(',', ' ').split()]
        if len(x) != self.num_qubits:
            raise ValueError(f"FEATUREMAP needs {self.num_qubits} values, got {len(x)}")
        self._emit_feature_map(qc, x)
        return True

    def cmd_qkernel(self, rest: str) -> None:
        """QKERNEL <x..> ; <y..> — quantum kernel |<phi(x)|phi(y)>|^2.

        Builds the ZZ feature-map states for data points x and y and returns
        their fidelity, the kernel entry used by quantum support vector machines.
        K(x, x) = 1."""
        if ';' not in rest:
            self.io.writeln("?USAGE: QKERNEL <x values> ; <y values>")
            return
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        try:
            xs, ys = rest.split(';', 1)
            x = [float(self._eval_with_vars(t, {})) for t in xs.replace(',', ' ').split()]
            y = [float(self._eval_with_vars(t, {})) for t in ys.replace(',', ' ').split()]
            if len(x) != len(y) or not x:
                raise ValueError(f"x and y need the same number of values ({len(x)} and {len(y)} given)")
            n = len(x)
            qcx = QuantumCircuit(n)
            self._emit_feature_map(qcx, x)
            qcy = QuantumCircuit(n)
            self._emit_feature_map(qcy, y)
            k = abs(np.vdot(Statevector(qcx).data, Statevector(qcy).data)) ** 2
            self.io.writeln(f"  Quantum kernel K(x,y) = {k:.6f}")
            self.variables['_QKERNEL'] = float(k)
        except Exception as e:
            self.io.writeln(f"?QKERNEL ERROR: {e}")


# ---------------------------------------------------------------------------
# QUBASIC phase estimation commands: QPE of a HAMILTONIAN's energy and quantum
# counting.
#
#     QPE H [BITS n] [STATE bits] [SHOTS s]
#     QCOUNT 101 [011 ...] [BITS n] [SHOTS s]
#
# Both run textbook phase estimation as a circuit: Hadamards on n counting
# qubits, the controlled powers U, U^2, U^4, ... on the system, an inverse
# quantum Fourier transform, and a reading of the counting register k, so
# U's eigenphase is k / 2^n to within 1 / 2^n. The circuit is simulated
# exactly (and SHOTS samples it).
#
# QPE takes U = exp(-i H t) with t = pi / sum|c| for H = sum c P, which keeps
# every energy inside the phase window, so E = -2 pi phi / t with phi taken in
# (-1/2, 1/2]. The system starts in a basis state: STATE, or the one of
# lowest diagonal energy; the chance of reading the ground energy is that
# state's overlap with the ground state.
#
# QCOUNT takes the Grover operator G = (2|s><s| - 1)(1 - 2 sum |m><m|) over
# n search qubits started in |s>, whose eigenphases phi satisfy
# marked = 2^n sin^2(pi phi), so one reading of phi counts the marked states.
# ---------------------------------------------------------------------------

MAX_QPE_QUBITS = 16


def _inverse_qft(qc: QuantumCircuit, qubits: list[int]) -> None:
    """The inverse QFT on qubits (qubits[0] the least significant bit)."""
    n = len(qubits)
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])
    for j in range(n):
        for k in range(j):
            qc.cp(-math.pi / (1 << (j - k)), qubits[k], qubits[j])
        qc.h(qubits[j])


def qpe_distribution(U: np.ndarray, prep: QuantumCircuit, bits: int) -> np.ndarray:
    """P(k) for the counting register after phase estimation of U from prep."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Statevector
    n = int(round(math.log2(U.shape[0])))
    qc = QuantumCircuit(bits + n)
    qc.compose(prep, qubits=list(range(bits, bits + n)), inplace=True)
    for j in range(bits):
        qc.h(j)
    power = U
    for j in range(bits):
        qc.append(UnitaryGate(power, check_input=False).control(1), [j] + list(range(bits, bits + n)))
        power = power @ power
    _inverse_qft(qc, list(range(bits)))
    return np.asarray(Statevector(qc).probabilities(list(range(bits))))


def _options(rest: str, keys: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    opts: dict[str, str] = {}
    text = rest
    for key in keys:
        m = re.search(rf'\b{key}\s+(\S+)', text, re.IGNORECASE)
        if m:
            opts[key] = m.group(1)
            text = text[:m.start()] + text[m.end():]
    return text.strip(), opts


class PhaseMixin(HostState):
    """QPE (energy estimation) and QCOUNT (quantum counting)."""

    def _phase_readout(self, probs: np.ndarray, shots: int | None) -> tuple[int, np.ndarray]:
        """The reported counting value: the most likely, or the most frequent of
        `shots` samples; and the distribution reported (exact or sampled)."""
        if shots:
            rng = np.random.default_rng(self._seed)
            counts = np.bincount(rng.choice(len(probs), size=shots, p=probs / probs.sum()),
                                 minlength=len(probs))
            return int(np.argmax(counts)), counts / shots
        return int(np.argmax(probs)), probs

    def cmd_qpe(self, rest: str) -> None:
        """QPE <HAMILTONIAN> [BITS n] [STATE bits] [SHOTS s] — energy by quantum phase estimation.

        (QPE count target UNITARY, the program statement, also runs here.)"""
        text, opts = _options(rest, ('BITS', 'STATE', 'SHOTS'))
        name = text.upper()
        if name not in getattr(self, '_hamiltonians', {}):
            if text:
                self.run_immediate('QPE ' + rest)
                return
            self.io.writeln("?USAGE: QPE <HAMILTONIAN> [BITS n] [STATE bits] [SHOTS s]")
            return
        try:
            op = self._hamiltonians[name]
            n = op.num_qubits
            bits = int(opts.get('BITS', 7))
            if bits < 1 or n + bits > MAX_QPE_QUBITS:
                raise ValueError(f"BITS must keep qubits + BITS within {MAX_QPE_QUBITS}")
            H = np.asarray(op.to_matrix())
            evals, evecs = np.linalg.eigh(H)
            norm = float(np.sum(np.abs(op.coeffs))) or 1.0
            t = math.pi / norm
            U = (evecs * np.exp(-1j * evals * t)) @ evecs.conj().T
            if 'STATE' in opts:
                display = opts['STATE']
                if len(display) != n or set(display) - {'0', '1'}:
                    raise ValueError(f"STATE takes {n} bits")
                index = int(self._internal_bits(display), 2)
            else:
                index = int(np.argmin(np.real(np.diag(H))))
            from qiskit import QuantumCircuit
            prep = QuantumCircuit(n)
            for q in range(n):
                if (index >> q) & 1:
                    prep.x(q)
            probs = qpe_distribution(U, prep, bits)
            shots = int(opts['SHOTS']) if 'SHOTS' in opts else None
            k, dist = self._phase_readout(probs, shots)

            def energy(kk: int) -> float:
                phi = kk / (1 << bits)
                if phi > 0.5:
                    phi -= 1.0
                return -2 * math.pi * phi / t

            e_est = energy(k)
            resolution = 2 * math.pi / t / (1 << bits)
            overlap = float(abs(evecs[index, 0]) ** 2)
            state = format(index, f'0{n}b')
            shown = self._bits(state) if hasattr(self, '_bits') else state
            self.io.writeln(f"\n  QPE {name}: {n} system qubits, {bits} counting bits, t = {t:.4f}, "
                            f"start |{shown}> (ground-state overlap {overlap:.3f})")
            self.io.writeln(f"    estimate      E = {e_est:+.6f} +- {resolution / 2:.4f}   "
                            f"(k = {k}, p = {dist[k]:.3f})")
            self.io.writeln(f"    exact ground  E0 = {evals[0]:+.6f}")
            top = np.argsort(dist)[::-1][:4]
            self.io.writeln("    readings      " + '  '.join(f"{energy(int(j)):+.4f} ({dist[j]:.3f})"
                                                           for j in top if dist[j] > 0.005))
            self.variables['_QPE_ENERGY'] = e_est
            self.variables['_QPE_EXACT'] = float(evals[0])
            self.variables['_QPE_OVERLAP'] = overlap
        except Exception as e:
            self.io.writeln(f"?QPE ERROR: {e}")

    def cmd_qcount(self, rest: str) -> None:
        """QCOUNT <bits> [<bits> ...] [BITS n] [SHOTS s] — quantum counting of marked states."""
        text, opts = _options(rest, ('BITS', 'SHOTS'))
        marked = [m for m in text.replace(',', ' ').split() if m]
        if not marked or any(set(m) - {'0', '1'} for m in marked) or len({len(m) for m in marked}) != 1:
            self.io.writeln("?USAGE: QCOUNT <bits> [<bits> ...] [BITS n] [SHOTS s]")
            return
        try:
            n = len(marked[0])
            bits = int(opts.get('BITS', n + 3))
            if bits < 1 or n + bits > MAX_QPE_QUBITS:
                raise ValueError(f"search qubits + BITS must stay within {MAX_QPE_QUBITS}")
            N = 1 << n
            targets = sorted({int(self._internal_bits(m), 2) for m in marked})
            oracle = np.ones(N)
            oracle[targets] = -1
            s = np.full(N, 1 / math.sqrt(N))
            G = (2 * np.outer(s, s) - np.eye(N)) @ np.diag(oracle)
            from qiskit import QuantumCircuit
            prep = QuantumCircuit(n)
            prep.h(range(n))
            probs = qpe_distribution(G.astype(complex), prep, bits)
            shots = int(opts['SHOTS']) if 'SHOTS' in opts else None
            k, dist = self._phase_readout(probs, shots)
            phi = k / (1 << bits)
            estimate = N * math.sin(math.pi * phi) ** 2
            self.io.writeln(f"\n  QCOUNT: {n} search qubits ({N} states), {bits} counting bits")
            self.io.writeln(f"    reading k = {k} (p = {dist[k]:.3f})  ->  marked ~ {estimate:.2f}  "
                            f"(rounded {round(estimate)}; actually {len(targets)})")
            self.variables['_QCOUNT'] = estimate
            self.variables['_QCOUNT_EXACT'] = len(targets)
        except Exception as e:
            self.io.writeln(f"?QCOUNT ERROR: {e}")


# ---------------------------------------------------------------------------
# QUBASIC oracle algorithms: Grover search, Simon's algorithm, Deutsch-Jozsa
# and Bernstein-Vazirani.
#
#     GROVER 101 [011 ...] [ITERS k] [SHOTS n] [PROGRAM]
#     SIMON 110 [SHOTS n]
#     DJ 5 BALANCED [SEED s] [SHOTS k]
#     BV 10110 [SHOTS k]
#
# GROVER marks the given basis states (typed in the displayed bit order) with
# a phase-flip oracle and runs the optimal number of Grover iterations,
# floor(pi / (4 theta)) for sin^2(theta) = marked / 2^n, unless ITERS is
# given; the report compares the measured success probability with
# sin^2((2k + 1) theta).
#
# SIMON hides a nonzero secret s in a two-to-one function f(x) = f(x XOR s)
# (the textbook oracle: copy x, then XOR s in controlled by one bit where s is
# 1), runs Simon's circuit (Hadamards, oracle, Hadamards, measure x) and
# solves the measured equations y . s = 0 over GF(2) for s.
# ---------------------------------------------------------------------------

class OracleMixin(HostState):
    """GROVER and SIMON commands for QBasicTerminal."""

    def _internal_bits(self, typed: str) -> str:
        """A typed bitstring in the internal order (qubit n-1 first)."""
        return typed[::-1] if getattr(self, '_endian_big', False) else typed

    def _emit_phase_flip(self, qc: QuantumCircuit, bits: str) -> None:
        """Phase-flip the basis state `bits` (internal order, qubit n-1 first)."""
        n = len(bits)
        zeros = [q for q in range(n) if bits[n - 1 - q] == '0']
        for q in zeros:
            qc.x(q)
        self._emit_mcz(qc, list(range(n)))
        for q in zeros:
            qc.x(q)

    def cmd_grover(self, rest: str) -> None:
        """GROVER <bits> [<bits> ...] [ITERS k] [SHOTS n] [PROGRAM] — Grover search for marked states."""
        text = rest
        opts: dict = {}
        for key in ('ITERS', 'SHOTS'):
            m = re.search(rf'\b{key}\s+(\d+)', text, re.IGNORECASE)
            if m:
                opts[key] = int(m.group(1))
                text = text[:m.start()] + text[m.end():]
        program = bool(re.search(r'\bPROGRAM\b', text, re.IGNORECASE))
        text = re.sub(r'\bPROGRAM\b', '', text, flags=re.IGNORECASE)
        marked = [t for t in text.replace(',', ' ').split() if t]
        if not marked or any(set(t) - {'0', '1'} for t in marked):
            self.io.writeln("?USAGE: GROVER <bits> [<bits> ...] [ITERS k] [SHOTS n] [PROGRAM]")
            return
        n = len(marked[0])
        if any(len(t) != n for t in marked):
            self.io.writeln("?GROVER: every marked state needs the same number of bits")
            return
        if n > 20:
            self.io.writeln("?GROVER searches at most 20 qubits")
            return
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_aer import AerSimulator
            internal = sorted(set(self._internal_bits(t) for t in marked))
            M, N = len(internal), 1 << n
            if M >= N:
                raise ValueError("every state is marked")
            theta = math.asin(math.sqrt(M / N))
            k = opts.get('ITERS', max(0, int(math.floor(math.pi / (4 * theta)))))
            qc = QuantumCircuit(n)
            for q in range(n):
                qc.h(q)
            for _ in range(k):
                for bits in internal:
                    self._emit_phase_flip(qc, bits)
                for q in range(n):
                    qc.h(q)
                self._emit_phase_flip(qc, '0' * n)
                for q in range(n):
                    qc.h(q)
            shots = opts.get('SHOTS', self.shots)
            meas = qc.copy()
            meas.measure_all()
            backend = AerSimulator(noise_model=self._noise_model) if self._noise_model else AerSimulator()
            kw: dict = {'shots': shots}
            if self._seed is not None:
                kw['seed_simulator'] = self._seed
            counts = backend.run(transpile(meas, backend), **kw).result().get_counts()
            hits = sum(c for b, c in counts.items() if b in internal)
            p_meas = hits / shots
            p_theory = math.sin((2 * k + 1) * theta) ** 2
            self.io.writeln(f"\n  Grover: {n} qubits, {M} marked of {N}, {k} iteration(s)")
            self.io.writeln(f"    success probability {p_meas:.4f} measured, "
                            f"{p_theory:.4f} predicted (sin^2((2k+1)theta))")
            self.last_counts = dict(counts)
            self.print_histogram(counts)
            self.variables['_GROVER_P'] = p_meas
            self.variables['_GROVER_ITERS'] = k
            if program:
                typed = ' '.join(self._bits(b) for b in internal)
                flip = 'Z 0' if n == 1 else 'MCZ ' + ', '.join(str(q) for q in range(n))
                lines = [f"REM Grover search for {typed}"] + [f"H {q}" for q in range(n)]
                for _ in range(k):
                    for bits in internal:
                        zeros = [q for q in range(n) if bits[n - 1 - q] == '0']
                        lines += [f"X {q}" for q in zeros] + [flip] + [f"X {q}" for q in zeros]
                    lines.append(f"DIFFUSE 0-{n - 1}")
                lines.append("MEASURE")
                self._write_program(n, lines)
                self.io.writeln(f"    PROGRAM: {len(lines)} lines written (QUBITS {n})")
        except Exception as e:
            self.io.writeln(f"?GROVER ERROR: {e}")

    def cmd_simon(self, rest: str) -> None:
        """SIMON <secret bits> [SHOTS n] — Simon's algorithm: recover a hidden XOR period."""
        text = rest
        shots = None
        m = re.search(r'\bSHOTS\s+(\d+)', text, re.IGNORECASE)
        if m:
            shots = int(m.group(1))
            text = text[:m.start()] + text[m.end():]
        secret = text.strip()
        if not secret or set(secret) - {'0', '1'} or '1' not in secret:
            self.io.writeln("?USAGE: SIMON <nonzero secret bits> [SHOTS n]")
            return
        n = len(secret)
        if 2 * n > 24:
            self.io.writeln("?SIMON takes secrets of at most 12 bits")
            return
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_aer import AerSimulator

            from qubasic_core.qec import _gf2_nullspace
            s_int = self._internal_bits(secret)            # qubit n-1 first
            s_bits = [int(s_int[n - 1 - j]) for j in range(n)]   # s_bits[j] = bit of qubit j
            pivot = next(j for j in range(n) if s_bits[j])
            qc = QuantumCircuit(2 * n, n)
            for q in range(n):
                qc.h(q)
            for q in range(n):                           # f(x) = x ...
                qc.cx(q, n + q)
            for j in range(n):                           # ... XOR s when x_pivot = 1
                if s_bits[j]:
                    qc.cx(pivot, n + j)
            for q in range(n):
                qc.h(q)
            qc.measure(range(n), range(n))
            backend = AerSimulator()
            kw: dict = {'shots': shots or max(4 * n, 32)}
            if self._seed is not None:
                kw['seed_simulator'] = self._seed
            counts = backend.run(transpile(qc, backend), **kw).result().get_counts()
            rows = [[int(key[n - 1 - j]) for j in range(n)] for key in counts]
            null = _gf2_nullspace(np.array(rows, dtype=np.uint8))
            self.io.writeln(f"\n  Simon: {n}-bit secret, {kw['shots']} shots, "
                            f"{len(counts)} distinct equations y.s = 0")
            if null.shape[0] != 1:
                self.io.writeln(f"    the equations leave {null.shape[0]} candidate directions; "
                                f"add SHOTS")
                return
            found_bits = null[0]
            found = ''.join(str(int(found_bits[q])) for q in range(n - 1, -1, -1))
            shown = self._bits(found)
            ok = found == s_int
            self.io.writeln(f"    recovered s = {shown}  ({'matches' if ok else 'differs from'} "
                            f"the secret)")
            self.variables['_SIMON_OK'] = -1 if ok else 0
        except Exception as e:
            self.io.writeln(f"?SIMON ERROR: {e}")

    def cmd_dj(self, rest: str) -> None:
        """DJ <n> CONSTANT|BALANCED [SEED s] [SHOTS k] — Deutsch-Jozsa: one query tells constant from balanced.

        A random f of the chosen kind on n bits (a constant 0 or 1, or a truth
        table with exactly half its values 1) acts as a phase oracle between
        two layers of Hadamards; every shot reads all zeros exactly when f is
        constant."""
        parts = rest.upper().split()
        try:
            n = int(parts[0])
            kind = parts[1]
        except (IndexError, ValueError):
            self.io.writeln("?USAGE: DJ <n> CONSTANT|BALANCED [SEED s] [SHOTS k]")
            return
        if kind not in ('CONSTANT', 'BALANCED') or not 1 <= n <= 12:
            self.io.writeln("?USAGE: DJ <n> CONSTANT|BALANCED (1 <= n <= 12)")
            return
        opts = {k: int(v) for k, v in zip(parts[2::2], parts[3::2]) if k in ('SEED', 'SHOTS') and v.isdigit()}
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import DiagonalGate
        from qiskit_aer import AerSimulator
        rng = np.random.default_rng(opts.get('SEED', self._seed))
        N = 1 << n
        if kind == 'CONSTANT':
            f = np.full(N, int(rng.integers(2)))
        else:
            f = np.zeros(N, dtype=int)
            f[rng.permutation(N)[:N // 2]] = 1
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        qc.append(DiagonalGate([(-1.0) ** int(v) for v in f]), list(range(n)))
        qc.h(range(n))
        qc.measure(range(n), range(n))
        shots = opts.get('SHOTS', 64)
        backend = AerSimulator()
        counts = backend.run(transpile(qc, backend), shots=shots,
                             seed_simulator=self._seed).result().get_counts()
        zeros = counts.get('0' * n, 0)
        verdict = 'CONSTANT' if zeros == shots else 'BALANCED'
        self.io.writeln(f"\n  DJ: n={n}, f {kind.lower()}, one oracle query, {shots} shots")
        self.io.writeln(f"    all-zero readings {zeros}/{shots}  ->  {verdict}")
        self.variables['_DJ_CONSTANT'] = -1 if verdict == 'CONSTANT' else 0

    def cmd_bv(self, rest: str) -> None:
        """BV <secret bits> [SHOTS k] — Bernstein-Vazirani: one query reads a secret s from f(x) = s . x."""
        parts = rest.split()
        secret = parts[0] if parts else ''
        if not secret or set(secret) - {'0', '1'} or len(secret) > 20:
            self.io.writeln("?USAGE: BV <secret bits> [SHOTS k]")
            return
        shots = int(parts[2]) if len(parts) > 2 and parts[1].upper() == 'SHOTS' else 32
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        n = len(secret)
        internal = self._internal_bits(secret)
        qc = QuantumCircuit(n + 1, n)
        qc.x(n)
        qc.h(range(n + 1))
        for q in range(n):
            if internal[n - 1 - q] == '1':
                qc.cx(q, n)
        qc.h(range(n))
        qc.measure(range(n), range(n))
        backend = AerSimulator()
        counts = backend.run(transpile(qc, backend), shots=shots,
                             seed_simulator=self._seed).result().get_counts()
        top = max(counts, key=counts.get)
        found = self._bits(top)
        self.io.writeln(f"\n  BV: {n}-bit secret, one oracle query, {shots} shots")
        self.io.writeln(f"    read {found} in {counts[top]}/{shots} shots"
                        f"{'' if found == secret else f'  (secret {secret})'}")
        self.variables['_BV_OK'] = -1 if found == secret else 0


# ---------------------------------------------------------------------------
# QUBASIC variational algorithms: QAOA and VQE as commands.
#
#     QAOA MAXCUT 0-1, 1-2, 2-0:2 [P p] [METHOD m] [SHOTS n] [PROGRAM]
#     QAOA RING 6 | QAOA COMPLETE 5 | QAOA <diagonal HAMILTONIAN> ...
#     VQE <HAMILTONIAN> [LAYERS L] [ANSATZ RY|RYRZ] [METHOD m] [RESTARTS k] [PROGRAM]
#
# Both simulate their ansatz on a numpy statevector and optimize with the
# optimizers MINIMIZE uses (Nelder-Mead, SPSA, parameter-shift gradient
# descent) or L-BFGS with parameter-shift gradients (the default).
#
# QAOA maximizes a diagonal objective C (the cut value of a weighted graph, or
# minus the energy of a Hamiltonian of Z terms): p layers of e^{-i gamma C}
# and the X mixer e^{-i beta sum X}, starting from |+...+>. Depth p is reached
# layer by layer, each optimum interpolated into the next depth (Zhou et al.,
# 2020). The report gives <C>, the exact optimum by enumeration, the
# approximation ratio, the probability of measuring an optimal solution and a
# sample of the output.
#
# VQE minimizes <H> over a hardware-efficient ansatz (RY, or RY and RZ, on
# every qubit between layers of a CX chain), with random restarts, and reports
# the energy beside the exact ground energy.
#
# PROGRAM writes the optimized circuit into the program (QUBITS set to match),
# so it can be RUN, sampled under NOISE, or exported.
# ---------------------------------------------------------------------------

MAX_VARIATIONAL_QUBITS = 22


def _apply_rx_all(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    """e^{-i beta X} on every qubit."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    t = psi.reshape([2] * n)
    for ax in range(n):
        a0 = np.take(t, 0, axis=ax)
        a1 = np.take(t, 1, axis=ax)
        t = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return t.reshape(-1)


def _apply_1q(psi: np.ndarray, n: int, q: int, m: np.ndarray) -> np.ndarray:
    t = psi.reshape([2] * n)
    ax = n - 1 - q
    t = np.moveaxis(np.tensordot(m, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def _apply_cx(psi: np.ndarray, n: int, c: int, tq: int) -> np.ndarray:
    idx = np.arange(psi.size)
    sel = ((idx >> c) & 1) == 1
    out = psi.copy()
    flip = idx ^ (1 << tq)
    out[sel] = psi[flip[sel]]
    return out


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


def graph_edges(spec: str, rng: Any) -> tuple[list[tuple[int, int, float]], int]:
    """(edges, nodes) from 'a-b[:w], ...', 'RING n', 'COMPLETE n' or 'RANDOM n [p]'."""
    toks = spec.replace(',', ' ').split()
    if toks and toks[0].upper() in ('RING', 'COMPLETE', 'RANDOM'):
        kind = toks[0].upper()
        n = int(toks[1]) if len(toks) > 1 else 4
        if kind == 'RING':
            edges = [(i, (i + 1) % n, 1.0) for i in range(n)]
        elif kind == 'COMPLETE':
            edges = [(i, j, 1.0) for i in range(n) for j in range(i + 1, n)]
        else:
            p = float(toks[2]) if len(toks) > 2 else 0.5
            edges = [(i, j, 1.0) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
        return edges, n
    edges = []
    for tok in toks:
        m = re.fullmatch(r'(\d+)-(\d+)(?::([-+0-9.eE]+))?', tok)
        if not m:
            raise ValueError(f"bad edge '{tok}' (use a-b or a-b:weight)")
        a, b = int(m.group(1)), int(m.group(2))
        if a == b:
            raise ValueError(f"edge {tok} joins a node to itself")
        edges.append((a, b, float(m.group(3)) if m.group(3) else 1.0))
    if not edges:
        raise ValueError("no edges given")
    return edges, max(max(a, b) for a, b, _ in edges) + 1


class VariationalMixin(HostState):
    """QAOA and VQE commands for QBasicTerminal (see the module docstring)."""

    # ── shared helpers ────────────────────────────────────────────────

    def _var_options(self, rest: str) -> tuple[str, dict[str, Any]]:
        """Strip trailing options (P, LAYERS, METHOD, ITERS, SHOTS, RESTARTS,
        EXCITED, ANSATZ, PROGRAM) from a command line."""
        opts: dict[str, Any] = {}
        text = rest
        for key, conv in (('P', int), ('LAYERS', int), ('ITERS', int), ('SHOTS', int),
                          ('RESTARTS', int), ('EXCITED', int), ('METHOD', str), ('ANSATZ', str)):
            m = re.search(rf'\b{key}\s+(\S+)', text, re.IGNORECASE)
            if m:
                opts[key.lower()] = conv(m.group(1)) if conv is int else m.group(1).upper()
                text = text[:m.start()] + text[m.end():]
        if re.search(r'\bPROGRAM\b', text, re.IGNORECASE):
            opts['program'] = True
            text = re.sub(r'\bPROGRAM\b', '', text, flags=re.IGNORECASE)
        return text.strip(), opts

    def _optimize(self, f: Callable, x0: np.ndarray, method: str, iters: int,
                  grad: Callable | None = None) -> tuple[np.ndarray, float, int]:
        """Minimize f from x0; returns (x, f(x), evaluations)."""
        evals = [0]

        def counted(v: np.ndarray | Sequence[float]) -> float:
            evals[0] += 1
            return f(np.asarray(v, dtype=float))

        if method in ('LBFGS', 'L-BFGS', 'BFGS'):
            from scipy.optimize import minimize

            def gradient(v: np.ndarray) -> np.ndarray:
                assert grad is not None
                evals[0] += 2 * len(v)
                return grad(np.asarray(v, dtype=float))
            res = minimize(counted, x0, jac=gradient if grad is not None else None,
                           method='L-BFGS-B', options={'maxiter': iters})
            return np.asarray(res.x), float(res.fun), evals[0]
        table = {'NM': self._nelder_mead, 'NELDER-MEAD': self._nelder_mead,
                 'SPSA': self._spsa, 'GRAD': self._grad_descent, 'GD': self._grad_descent}
        if method not in table:
            raise ValueError(f"unknown METHOD {method} (LBFGS, NM, SPSA, GRAD)")
        best, fval = table[method](lambda v: counted(v), list(x0), iters, 0.3)
        return np.asarray(best, dtype=float), float(fval), evals[0]

    def _write_program(self, n: int, lines: list[str]) -> None:
        """Replace the program with `lines` (numbered 10, 20, ...) on n qubits;
        variables, definitions and Hamiltonians stay."""
        self.program.clear()
        self._parsed.clear()
        if n != self.num_qubits:
            self._invalidate_run_state()
            self.registers.clear()
        self.num_qubits = n
        for k, text in enumerate(lines):
            self.process(f"{(k + 1) * 10} {text}", track_undo=False)

    # ── QAOA ──────────────────────────────────────────────────────────

    def _qaoa_problem(self, spec: str, rng: Any) -> tuple[np.ndarray, int, str, list]:
        """(objective vector to maximize, qubits, description, cost terms) where
        terms are (coefficient, qubits) of the Z products in the phase layer."""
        name = spec.strip().upper()
        if name in getattr(self, '_hamiltonians', {}):
            op = self._hamiltonians[name]
            n = op.num_qubits
            if n > MAX_VARIATIONAL_QUBITS:
                raise ValueError(f"QAOA simulates at most {MAX_VARIATIONAL_QUBITS} qubits")
            idx = np.arange(1 << n)
            energy = np.zeros(1 << n)
            terms = []
            for label, coeff in zip(op.paulis.to_labels(), op.coeffs):
                if set(label) - {'I', 'Z'}:
                    raise ValueError(f"QAOA needs a diagonal Hamiltonian (Z terms only); "
                                     f"{name} has {label}")
                qs = [n - 1 - k for k, ch in enumerate(label) if ch == 'Z']
                parity = np.zeros(1 << n, dtype=np.int64)
                for q in qs:
                    parity ^= (idx >> q) & 1
                energy += float(np.real(coeff)) * (1 - 2 * parity)
                if qs:
                    terms.append((-float(np.real(coeff)), qs))
            return -energy, n, f"HAMILTONIAN {name} ({len(op)} terms)", terms
        body = re.sub(r'^MAXCUT\b', '', spec.strip(), flags=re.IGNORECASE).strip()
        edges, n = graph_edges(body, rng)
        if n > MAX_VARIATIONAL_QUBITS:
            raise ValueError(f"QAOA simulates at most {MAX_VARIATIONAL_QUBITS} nodes")
        idx = np.arange(1 << n)
        cut = np.zeros(1 << n)
        for a, b, w in edges:
            cut += w * (((idx >> a) & 1) != ((idx >> b) & 1))
        # w (1 - Z_a Z_b) / 2: the phase layer sees -w/2 on Z_a Z_b.
        terms = [(-w / 2, [a, b]) for a, b, w in edges]
        return cut, n, f"MaxCut, {n} nodes, {len(edges)} edges", terms

    @staticmethod
    def _qaoa_state(obj: np.ndarray, n: int, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
        psi = np.full(1 << n, 1 / np.sqrt(1 << n), dtype=complex)
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * obj)
            psi = _apply_rx_all(psi, n, b)
        return psi

    def _qaoa_optimize(self, obj: np.ndarray, n: int, p: int, method: str, iters: int) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Optimal (gammas, betas, <C>, evaluations) at depth p, grown layer by layer."""
        def neg(v: np.ndarray) -> float:
            k = len(v) // 2
            psi = self._qaoa_state(obj, n, v[:k], v[k:])
            return -float(np.dot(np.abs(psi) ** 2, obj))

        def grad(v: np.ndarray) -> np.ndarray:
            g = np.zeros_like(v)
            h = 1e-5
            for i in range(len(v)):
                e = np.zeros_like(v)
                e[i] = h
                g[i] = (neg(v + e) - neg(v - e)) / (2 * h)
            return g

        evals = 0
        scale = float(np.max(np.abs(obj))) or 1.0
        grid_g = np.linspace(0.05, np.pi, 24) / scale * 2
        grid_b = np.linspace(0.05, np.pi / 2, 16)
        best = min(((g, b) for g in grid_g for b in grid_b), key=lambda gb: neg(np.array(gb)))
        evals += len(grid_g) * len(grid_b)
        x, fval, e = self._optimize(neg, np.array(best, dtype=float), method, iters,
                                    grad if method in ('LBFGS', 'L-BFGS', 'BFGS') else None)
        evals += e
        for depth in range(2, p + 1):
            k = depth - 1
            gam, bet = x[:k], x[k:]

            def interp(v: np.ndarray) -> np.ndarray:
                ext = np.concatenate([[0.0], v, [0.0]])
                return np.array([(i / k) * ext[i] + ((k - i) / k) * ext[i + 1]
                                 for i in range(depth)])
            x0 = np.concatenate([interp(gam), interp(bet)])
            x, fval, e = self._optimize(neg, x0, method, iters,
                                        grad if method in ('LBFGS', 'L-BFGS', 'BFGS') else None)
            evals += e
        return x[:p], x[p:], -fval, evals

    def cmd_qaoa(self, rest: str) -> None:
        """QAOA <graph | HAMILTONIAN> [P p] [METHOD m] [ITERS n] [SHOTS n] [PROGRAM] — QAOA on a MaxCut graph or a diagonal Hamiltonian.

        The graph is 'a-b[:w], ...', RING n, COMPLETE n or RANDOM n [p] (MAXCUT
        may precede it); a declared HAMILTONIAN of Z terms is minimized
        instead. See qubasic_core/algorithms.py."""
        text, opts = self._var_options(rest)
        if not text:
            self.io.writeln("?USAGE: QAOA MAXCUT 0-1, 1-2, 2-0 [P p] [METHOD m] [SHOTS n] [PROGRAM]")
            return
        try:
            rng = np.random.default_rng(self._seed)
            obj, n, what, terms = self._qaoa_problem(text, rng)
            p = max(1, opts.get('p', 2))
            method = opts.get('method', 'LBFGS')
            gammas, betas, expect, evals = self._qaoa_optimize(obj, n, p, method,
                                                               opts.get('iters', 200))
            psi = self._qaoa_state(obj, n, gammas, betas)
            probs = np.abs(psi) ** 2
            best_val = float(np.max(obj))
            worst_val = float(np.min(obj))
            optimal = np.isclose(obj, best_val)
            p_opt = float(probs[optimal].sum())
            ham = what.startswith('HAMILTONIAN')
            self.io.writeln(f"\n  QAOA {what}: p={p}, {method.lower()} ({evals} evaluations)")
            self.io.writeln("    gamma = [" + ', '.join(f"{g:.4f}" for g in gammas) + "]")
            self.io.writeln("    beta  = [" + ', '.join(f"{b:.4f}" for b in betas) + "]")
            if ham:
                self.io.writeln(f"    <H> = {-expect:.6f}   ground energy {-best_val:.6f}")
                ratio = (expect - worst_val) / (best_val - worst_val) if best_val > worst_val else 1.0
            else:
                ratio = expect / best_val if best_val else 1.0
                self.io.writeln(f"    <C> = {expect:.4f} of max cut {best_val:g} "
                                f"(approximation ratio {ratio:.4f})")
            self.io.writeln(f"    P(optimal) = {p_opt:.4f}")
            top = np.argsort(-probs)[:4]
            label = 'energy' if ham else 'cut'
            shown = ', '.join(f"{self._bits(format(int(z), f'0{n}b'))} "
                              f"({label} {(-obj[z] if ham else obj[z]):g}, p={probs[z]:.3f})"
                              for z in top)
            self.io.writeln(f"    most likely: {shown}")
            shots = opts.get('shots', self.shots)
            draws = rng.choice(probs.size, size=shots, p=probs / probs.sum())
            best_draw = int(draws[np.argmax(obj[draws])])
            self.io.writeln(f"    {shots} shots: best {label} {(-obj[best_draw] if ham else obj[best_draw]):g} "
                            f"({self._bits(format(best_draw, f'0{n}b'))})")
            self.variables['_QAOA_EXPECT'] = float(-expect if ham else expect)
            self.variables['_QAOA_RATIO'] = float(ratio)
            self.variables['_QAOA_POPT'] = p_opt
            self.arrays['_QAOA_GAMMA'] = [float(g) for g in gammas]
            self.arrays['_QAOA_BETA'] = [float(b) for b in betas]
            if opts.get('program'):
                self._qaoa_program(n, terms, gammas, betas)
        except Exception as e:
            self.io.writeln(f"?QAOA ERROR: {e}")

    def _qaoa_program(self, n: int, terms: list, gammas: np.ndarray, betas: np.ndarray) -> None:
        """Write the optimized QAOA circuit as a program (one-, two- and
        many-qubit Z phases as RZ, RZZ and CX ladders)."""
        lines = ['REM QAOA circuit written by QAOA ... PROGRAM']
        lines += [f"H {q}" for q in range(n)]
        for g, b in zip(gammas, betas):
            for coeff, qs in terms:
                theta = 2 * g * coeff          # e^{-i g c Z..} = R_Z..(2 g c)
                if len(qs) == 1:
                    lines.append(f"RZ {float(theta)!r}, {qs[0]}")
                elif len(qs) == 2:
                    lines.append(f"RZZ {float(theta)!r}, {qs[0]}, {qs[1]}")
                else:
                    for a, c in zip(qs, qs[1:]):
                        lines.append(f"CX {a}, {c}")
                    lines.append(f"RZ {float(theta)!r}, {qs[-1]}")
                    for a, c in reversed(list(zip(qs, qs[1:]))):
                        lines.append(f"CX {a}, {c}")
            lines += [f"RX {float(2 * b)!r}, {q}" for q in range(n)]
        lines.append('MEASURE')
        self._write_program(n, lines)
        self.io.writeln(f"    PROGRAM: {len(lines)} lines written (QUBITS {n}); RUN samples it")

    # ── VQE ───────────────────────────────────────────────────────────

    @staticmethod
    def _ansatz_state(n: int, layers: int, params: np.ndarray, kind: str) -> np.ndarray:
        psi = np.zeros(1 << n, dtype=complex)
        psi[0] = 1.0
        k = 0
        per = 2 if kind == 'RYRZ' else 1
        for layer in range(layers + 1):
            for q in range(n):
                psi = _apply_1q(psi, n, q, _ry(params[k]))
                k += 1
                if per == 2:
                    psi = _apply_1q(psi, n, q, _rz(params[k]))
                    k += 1
            if layer < layers:
                for q in range(n - 1):
                    psi = _apply_cx(psi, n, q, q + 1)
        return psi

    def cmd_vqe(self, rest: str) -> None:
        """VQE <HAMILTONIAN> [LAYERS L] [ANSATZ RY|RYRZ] [METHOD m] [RESTARTS k] [EXCITED k] [PROGRAM] — variational ground and excited-state energies of a Hamiltonian.

        Minimizes <H> over a hardware-efficient ansatz (L layers of a CX chain
        between rotation layers, default L=2) and compares with the exact
        ground energy. EXCITED k then finds k more levels by variational
        quantum deflation: each minimizes <H> + beta * sum |<psi_j|psi>|^2 over
        the states already found ($_VQE_ENERGIES holds every level). See
        qubasic_core/algorithms.py."""
        text, opts = self._var_options(rest)
        name = text.strip().upper()
        if not name:
            self.io.writeln("?USAGE: VQE <HAMILTONIAN> [LAYERS L] [ANSATZ RY|RYRZ] [METHOD m] "
                            "[RESTARTS k] [PROGRAM]")
            return
        try:
            if name not in getattr(self, '_hamiltonians', {}):
                raise ValueError(f"unknown HAMILTONIAN '{name}' (declare one with HAMILTONIAN)")
            op = self._hamiltonians[name]
            n = op.num_qubits
            if n > 16:
                raise ValueError("VQE simulates at most 16 qubits")
            layers = opts.get('layers', 2)
            kind = opts.get('ansatz', 'RY')
            if kind not in ('RY', 'RYRZ'):
                raise ValueError("ANSATZ is RY or RYRZ")
            method = opts.get('method', 'LBFGS')
            restarts = max(1, opts.get('restarts', 3))
            Hs = op.to_matrix(sparse=True).tocsr()
            n_params = n * (layers + 1) * (2 if kind == 'RYRZ' else 1)

            def energy(v: np.ndarray) -> float:
                psi = self._ansatz_state(n, layers, v, kind)
                return float(np.real(np.vdot(psi, Hs @ psi)))

            def grad(v: np.ndarray) -> np.ndarray:
                g = np.zeros_like(v)
                for i in range(len(v)):
                    e = np.zeros_like(v)
                    e[i] = np.pi / 2
                    g[i] = (energy(v + e) - energy(v - e)) / 2
                return g

            rng = np.random.default_rng(self._seed)
            best: tuple[float, np.ndarray | None] = (np.inf, None)
            total_evals = 0
            for _ in range(restarts):
                x0 = rng.uniform(-np.pi, np.pi, n_params)
                x, fval, ev = self._optimize(energy, x0, method, opts.get('iters', 300),
                                             grad if method in ('LBFGS', 'L-BFGS', 'BFGS') else None)
                total_evals += ev
                if fval < best[0]:
                    best = (fval, x)
            e_vqe, params = best
            assert params is not None
            if Hs.shape[0] <= 4096:
                e0 = float(np.linalg.eigvalsh(Hs.toarray())[0])
            else:
                from scipy.sparse.linalg import eigsh
                e0 = float(eigsh(Hs, k=1, which='SA')[0][0])
            err = e_vqe - e0
            self.io.writeln(f"\n  VQE {name}: {n} qubits, {kind} ansatz, {layers} layer(s), "
                            f"{n_params} parameters, {method.lower()} x{restarts} "
                            f"({total_evals} evaluations)")
            self.io.writeln(f"    VQE energy   = {e_vqe:.6f}")
            self.io.writeln(f"    exact ground = {e0:.6f}   (error {err:.2e})")
            self.variables['_VQE_ENERGY'] = e_vqe
            self.variables['_VQE_EXACT'] = e0
            self.variables['_VQE_ERROR'] = err
            self.arrays['_VQE_PARAMS'] = [float(v) for v in params]
            excited = max(0, opts.get('excited', 0))
            if excited:
                self._vqe_excited(n, layers, kind, method, restarts, opts.get('iters', 300),
                                  Hs, params, e_vqe, excited, float(np.sum(np.abs(op.coeffs))), rng)
            if opts.get('program'):
                self._vqe_program(n, layers, params, kind, name)
        except Exception as e:
            self.io.writeln(f"?VQE ERROR: {e}")

    def _vqe_excited(self, n: int, layers: int, kind: str, method: str, restarts: int,
                     iters: int, Hs: Any, params: np.ndarray, e0_vqe: float, excited: int,
                     norm: float, rng: np.random.Generator) -> None:
        """Variational quantum deflation: level k minimizes <H> plus beta times
        the overlap with each lower level found, beta larger than the spectrum's
        width so that no lower state can win."""
        beta = 2.0 * norm + 1.0
        states = [self._ansatz_state(n, layers, params, kind)]
        energies = [e0_vqe]
        n_params = len(params)
        for _ in range(excited):
            lower = tuple(states)

            def penalized(v: np.ndarray, lower: tuple = lower) -> float:
                psi = self._ansatz_state(n, layers, v, kind)
                return float(np.real(np.vdot(psi, Hs @ psi))
                             + beta * sum(abs(np.vdot(s, psi)) ** 2 for s in lower))

            def pgrad(v: np.ndarray) -> np.ndarray:
                g = np.zeros_like(v)
                for i in range(len(v)):
                    e = np.zeros_like(v)
                    e[i] = np.pi / 2
                    g[i] = (penalized(v + e) - penalized(v - e)) / 2
                return g

            best: tuple[float, np.ndarray | None] = (np.inf, None)
            for _ in range(restarts):
                x, fval, _ev = self._optimize(penalized, rng.uniform(-np.pi, np.pi, n_params), method,
                                              iters, pgrad if method in ('LBFGS', 'L-BFGS', 'BFGS') else None)
                if fval < best[0]:
                    best = (fval, x)
            assert best[1] is not None
            psi = self._ansatz_state(n, layers, best[1], kind)
            states.append(psi)
            energies.append(float(np.real(np.vdot(psi, Hs @ psi))))
        dense = Hs.toarray() if Hs.shape[0] <= 4096 else None
        exact = list(np.linalg.eigvalsh(dense)[:excited + 1]) if dense is not None else []
        self.io.writeln(f"    deflation (beta {beta:.3g}):")
        for k, e in enumerate(energies):
            ref = f"   exact {exact[k]:.6f}   (error {e - exact[k]:.2e})" if k < len(exact) else ''
            self.io.writeln(f"      level {k}: {e:.6f}{ref}")
        self.arrays['_VQE_ENERGIES'] = [float(e) for e in energies]
        if exact:
            self.arrays['_VQE_EXACT_LEVELS'] = [float(e) for e in exact]

    def _vqe_program(self, n: int, layers: int, params: np.ndarray, kind: str, name: str) -> None:
        lines = [f'REM VQE ansatz for {name} written by VQE ... PROGRAM']
        k = 0
        for layer in range(layers + 1):
            for q in range(n):
                lines.append(f"RY {float(params[k])!r}, {q}")
                k += 1
                if kind == 'RYRZ':
                    lines.append(f"RZ {float(params[k])!r}, {q}")
                    k += 1
            if layer < layers:
                lines += [f"CX {q}, {q + 1}" for q in range(n - 1)]
        lines.append(f"SAVE_EXPECT {name} -> energy")
        self._write_program(n, lines)
        self.io.writeln(f"    PROGRAM: {len(lines)} lines written (QUBITS {n}); RUN sets 'energy'")


# ---------------------------------------------------------------------------
# QUBASIC factoring: SHOR, order finding with Beauregard's circuit or a
# permutation oracle (the arithmetic circuits are in shor.py).
# ---------------------------------------------------------------------------

class FactoringMixin(HostState):
    """The SHOR command."""

    # Widest N the modular-arithmetic circuit runs (2n + 3 qubits, n = 8).
    SHOR_MAX_BITS = 8

    def cmd_shor(self, rest: str) -> None:
        """SHOR <N> [a] [SHOTS n] [ORACLE] — factor N by quantum order finding.

        Runs Beauregard's circuit: modular exponentiation from Fourier-basis
        adders and controlled modular multipliers on 2n + 3 qubits, with the
        phase read one bit at a time through a single control qubit (see
        shor.py). The order r of a mod N comes from the measured phases by
        continued fractions, and a factor from gcd(a^(r/2) +- 1, N). Even N,
        primes and prime powers are settled classically. ORACLE runs phase
        estimation of the permutation matrix x -> a x mod N instead, on 3n
        qubits."""
        parts = rest.split()
        oracle = any(p_.upper() == 'ORACLE' for p_ in parts)
        parts = [p_ for p_ in parts if p_.upper() != 'ORACLE']
        shots = None
        if any(p_.upper() == 'SHOTS' for p_ in parts):
            k = [p_.upper() for p_ in parts].index('SHOTS')
            shots = int(parts[k + 1])
            parts = parts[:k] + parts[k + 2:]
        if not parts:
            self.io.writeln("?USAGE: SHOR <N> [a] [SHOTS n] [ORACLE]")
            return
        try:
            N = int(parts[0])
            if N < 4:
                raise ValueError("N must be at least 4")
            kind, facs = classical_shortcut(N)
            if kind == 'prime':
                self.io.writeln(f"  N={N} is prime: nothing to factor")
                return
            if kind:
                assert facs is not None
                self.io.writeln(f"  N={N} is {'even' if kind == 'even' else 'a prime power'}: "
                                f"{facs[0]} x {facs[1]} (no order finding needed)")
                self.variables['_SHOR_FACTOR'] = facs[0]
                return
            n = N.bit_length()
            if not oracle and n > self.SHOR_MAX_BITS:
                raise ValueError(f"N={N} needs {2 * n + 3} qubits; the circuit runs N < "
                                 f"{1 << self.SHOR_MAX_BITS} (ORACLE reaches a little further)")
            rng = np.random.default_rng(self._seed)
            fixed_a = int(parts[1]) if len(parts) > 1 else None
            attempts = [fixed_a] if fixed_a is not None else [None] * 4
            tried: set[int] = set()
            for attempt in attempts:
                a = attempt if attempt is not None else int(rng.integers(2, N - 1))
                while attempt is None and a in tried and len(tried) < N - 3:
                    a = int(rng.integers(2, N - 1))
                tried.add(a)
                g = math.gcd(a, N)
                if g > 1:
                    self.io.writeln(f"  gcd({a},{N})={g} (lucky) -> factors {g} and {N // g}")
                    self.variables['_SHOR_FACTOR'] = g
                    return
                if oracle:
                    phases = self._shor_oracle_phases(a, N, shots or max(2000, self.shots))
                else:
                    phases = self._shor_circuit_phases(a, N, shots or 24)
                r = None
                for ph, _count in phases:
                    r = order_from_phase(ph, a, N)
                    if r:
                        break
                if not r:
                    self.io.writeln("  order finding inconclusive for this a")
                    continue
                self.io.writeln(f"  Order r = {r} ({a}^{r} mod {N} = {pow(a, r, N)})")
                self.variables['_SHOR_ORDER'] = r
                found = factors_from_order(a, r, N)
                if found:
                    self.io.writeln(f"  Factors of {N}: {min(found)} x {max(found)}")
                    self.variables['_SHOR_FACTOR'] = min(found)
                    return
                self.io.writeln("  odd order or a^(r/2) = -1 mod N: no factor from this a")
                if fixed_a is not None:
                    return
            self.io.writeln("  no factor found (retry, or give another a)")
        except Exception as e:
            self.io.writeln(f"?SHOR ERROR: {e}")

    def _shor_circuit_phases(self, a: int, N: int, shots: int) -> list:
        """Run the order-finding circuit; (phase, count) pairs, most frequent first."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        qc = order_finding_circuit(a, N)
        t = 2 * N.bit_length()
        sim = AerSimulator(method='statevector', precision=self.sim_precision)
        qt = transpile(qc, sim)
        kw: dict = {'shots': shots}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        counts = sim.run(qt, **kw).result().get_counts()
        self.io.writeln(f"\n  Shor: N={N}, a={a}: modular-arithmetic circuit, {qc.num_qubits} qubits "
                        f"(2n+3), {qt.size()} operations, {shots} shots")
        tally: dict = {}
        for bits, c in counts.items():
            ph = phase_value(bits, t)
            tally[ph] = tally.get(ph, 0) + c
        ranked = sorted(tally.items(), key=lambda kv: -kv[1])
        shown = ', '.join(f"{ph.numerator}/{ph.denominator} ({c})" for ph, c in ranked[:6])
        self.io.writeln(f"  measured phases: {shown}")
        return ranked

    def _shor_oracle_phases(self, a: int, N: int, shots: int) -> list:
        """Phase estimation of the permutation unitary x -> a x mod N."""
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import UnitaryGate
        from qiskit_aer import AerSimulator
        w = N.bit_length()
        dim = 2 ** w
        P = np.zeros((dim, dim))
        for x in range(dim):
            P[(a * x % N) if x < N else x, x] = 1.0
        count = 2 * w
        total = count + w
        qc = QuantumCircuit(total, count)
        for c in range(count):
            qc.h(c)
        qc.x(count)        # work register starts in |1>
        for j in range(count):
            Uk = np.linalg.matrix_power(P, 2 ** j).astype(complex)
            qc.append(UnitaryGate(Uk).control(1), [j] + list(range(count, total)))
        self._emit_qft(qc, list(range(count)), inverse=True, swaps=True)
        qc.measure(range(count), range(count))
        backend = AerSimulator()
        kw: dict = {'shots': shots}
        if self._seed is not None:
            kw['seed_simulator'] = self._seed
        counts = backend.run(transpile(qc, backend), **kw).result().get_counts()
        self.io.writeln(f"\n  Shor: N={N}, a={a}: permutation-oracle phase estimation, "
                        f"{count} counting qubits")
        ranked = sorted(((Fraction(int(bits, 2), 2 ** count), c) for bits, c in counts.items()),
                        key=lambda kv: -kv[1])
        return ranked


# ---------------------------------------------------------------------------
# Shor's algorithm with modular-arithmetic circuits (Beauregard, 2003).
#
# Order finding for a mod N on 2n + 3 qubits, n = bit length of N:
#
#     c        one control qubit, reused for every bit of the phase
#     x        n qubits holding the exponentiation register, starting at |1>
#     b        n + 1 qubits, a work register kept in the Fourier basis
#     anc      one ancilla for the modular reduction
#
# Addition of a classical constant happens in the Fourier basis (Draper's
# adder: one phase rotation per qubit), modular addition a + b mod N is built
# from it with one overflow ancilla, controlled modular multiplication
# x -> a x mod N is n modular additions plus a controlled swap and the inverse
# multiplication by a^-1, and phase estimation reads the order with a single
# control qubit measured 2n times, each later rotation conditioned on the
# bits already read (the semiclassical inverse Fourier transform).
#
# The classical part checks the easy cases (even N, prime powers, a shared
# factor) and turns a measured phase into a period by continued fractions.
# ---------------------------------------------------------------------------

class Layout(NamedTuple):
    c: int
    x: list[int]
    b: list[int]
    anc: int
    total: int


def layout(n: int) -> Layout:
    """Qubit indices of the 2n + 3 qubit order-finding register."""
    return Layout(c=0, x=list(range(1, n + 1)), b=list(range(n + 1, 2 * n + 2)),
                  anc=2 * n + 2, total=2 * n + 3)


def qft(qc: QuantumCircuit, qs: list[int], inverse: bool = False) -> None:
    """Fourier transform over qs (qs[0] the least significant bit), without
    the final bit reversal: afterwards qubit qs[l] carries the phase
    2 pi y 2^l / 2^m of basis value y."""
    m = len(qs)
    order = list(reversed(qs))           # most significant first
    if not inverse:
        for i in range(m):
            qc.h(order[i])
            for j in range(i + 1, m):
                qc.cp(np.pi / 2 ** (j - i), order[j], order[i])
    else:
        for i in reversed(range(m)):
            for j in reversed(range(i + 1, m)):
                qc.cp(-np.pi / 2 ** (j - i), order[j], order[i])
            qc.h(order[i])


def phi_add(qc: QuantumCircuit, a: int, qs: list[int], controls: Iterable[int] = (), inverse: bool = False) -> None:
    """Add the constant a (mod 2^m) to a register held in the Fourier basis
    by `qft`: qubit qs[l] carries the phase 2 pi y / 2^(l+1), so adding a
    rotates it by 2 pi a / 2^(l+1); each rotation is controlled by every
    qubit in controls."""
    controls = list(controls)
    sign = -1.0 if inverse else 1.0
    for l, target in enumerate(qs):
        period = 1 << (l + 1)
        frac = (a % period) / period
        if frac == 0:
            continue
        angle = sign * 2 * np.pi * frac
        if controls:
            qc.mcp(angle, controls, target)
        else:
            qc.p(angle, target)


def phi_add_mod(qc: QuantumCircuit, a: int, N: int, b: list[int], anc: int, controls: list[int],
                inverse: bool = False) -> None:
    """Doubly controlled (a + b) mod N on a Fourier-basis register b of n + 1
    qubits (b < N and a < N), the overflow ancilla anc returned to |0>."""
    msb = b[-1]
    if not inverse:
        phi_add(qc, a, b, controls)
        phi_add(qc, N, b, inverse=True)
        qft(qc, b, inverse=True)
        qc.cx(msb, anc)
        qft(qc, b)
        phi_add(qc, N, b, [anc])
        phi_add(qc, a, b, controls, inverse=True)
        qft(qc, b, inverse=True)
        qc.x(msb)
        qc.cx(msb, anc)
        qc.x(msb)
        qft(qc, b)
        phi_add(qc, a, b, controls)
    else:
        phi_add(qc, a, b, controls, inverse=True)
        qft(qc, b, inverse=True)
        qc.x(msb)
        qc.cx(msb, anc)
        qc.x(msb)
        qft(qc, b)
        phi_add(qc, a, b, controls)
        phi_add(qc, N, b, [anc], inverse=True)
        qft(qc, b, inverse=True)
        qc.cx(msb, anc)
        qft(qc, b)
        phi_add(qc, N, b)
        phi_add(qc, a, b, controls, inverse=True)


def cmult(qc: QuantumCircuit, a: int, N: int, c: int, x: list[int], b: list[int], anc: int,
          inverse: bool = False) -> None:
    """Controlled |x>|b> -> |x>|(b + a x) mod N> (b in the computational basis)."""
    n = len(x)
    qft(qc, b)
    steps = [(i, (a << i) % N) for i in range(n)]
    for i, ai in (reversed(steps) if inverse else steps):
        phi_add_mod(qc, ai, N, b, anc, [c, x[i]], inverse=inverse)
    qft(qc, b, inverse=True)


def controlled_modmul(qc: QuantumCircuit, a: int, N: int, c: int, x: list[int], b: list[int], anc: int) -> None:
    """Controlled x -> a x mod N in place (b returns to |0>); a must be a unit mod N."""
    cmult(qc, a, N, c, x, b, anc)
    for xi, bi in zip(x, b):
        qc.cswap(c, xi, bi)
    cmult(qc, pow(a, -1, N), N, c, x, b, anc, inverse=True)


def order_finding_circuit(a: int, N: int) -> QuantumCircuit:
    """The semiclassical order-finding circuit for a mod N; its classical
    register `phase` holds the 2n phase bits, bit k the k-th read (the least
    significant bit of the estimate first)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ClassicalRegister, QuantumRegister
    n = N.bit_length()
    L = layout(n)
    t = 2 * n
    q = QuantumRegister(L.total, 'q')
    creg = ClassicalRegister(t, 'phase')
    qc = QuantumCircuit(q, creg)
    c, x, b, anc = L.c, L.x, L.b, L.anc
    qc.x(x[0])                                   # |x> = |1>
    for step in range(t):
        k = t - 1 - step                         # U^(2^k): least significant phase bit first
        qc.h(c)
        controlled_modmul(qc, pow(a, 1 << k, N), N, c, x, b, anc)
        for j in range(step):                    # bits read so far correct this one
            angle = -np.pi / (1 << (step - j))
            with qc.if_test((creg[j], 1)):
                qc.p(angle, c)
        qc.h(c)
        qc.measure(c, creg[step])
        qc.reset(c)
    return qc


def phase_value(bits: str, t: int) -> Fraction:
    """The phase estimate y / 2^t from a counts key of the `phase` register
    (rightmost character = bit 0, the first bit read = the least significant)."""
    y = 0
    for step in range(t):
        if bits[len(bits) - 1 - step] == '1':
            y |= 1 << step
    return Fraction(y, 1 << t)


def order_from_phase(phase: Fraction, a: int, N: int) -> int | None:
    """The order r of a mod N from a phase estimate s/r, by continued
    fractions (small multiples of the denominator are tried too)."""
    if phase == 0:
        return None
    r = phase.limit_denominator(N).denominator
    for mult in range(1, 8):
        cand = r * mult
        if cand > N:
            break
        if pow(a, cand, N) == 1:
            return cand
    return None


def factors_from_order(a: int, r: int, N: int) -> tuple[int, int] | None:
    if r % 2:
        return None
    y = pow(a, r // 2, N)
    if y == N - 1:
        return None
    for g in (math.gcd(y - 1, N), math.gcd(y + 1, N)):
        if 1 < g < N:
            return g, N // g
    return None


def classical_shortcut(N: int) -> tuple[str, tuple[int, int] | None]:
    """('even' | 'prime' | 'power' | '', factors) — the cases order finding
    does not need."""
    if N % 2 == 0:
        return 'even', (2, N // 2)
    if _is_prime(N):
        return 'prime', None
    for k in range(2, N.bit_length() + 1):
        root = round(N ** (1.0 / k))
        for cand in (root - 1, root, root + 1):
            if cand > 1 and cand ** k == N:
                return 'power', (cand, N // cand)
    return '', None


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


# ---------------------------------------------------------------------------
# QUBASIC linear systems: HHL for Hermitian and dilated non-Hermitian matrices.
# ---------------------------------------------------------------------------

class HHLMixin(HostState):
    """The HHL command."""

    def _parse_hhl_args(self, rest: str) -> tuple[np.ndarray, np.ndarray, int | None, int]:
        """(A, b, clock qubits or None, shots) from either form:
        HHL a b c d v0 v1 [clock] (2x2) or HHL [[...]] [v...] [CLOCK n] [SHOTS n]."""
        text = rest.strip()
        shots = 0
        m = re.search(r'\bSHOTS\s+(\d+)\s*$', text, re.IGNORECASE)
        if m:
            shots = int(m.group(1))
            text = text[:m.start()].rstrip()
        clock = None
        m = re.search(r'\bCLOCK\s+(\d+)\s*$', text, re.IGNORECASE)
        if m:
            clock = int(m.group(1))
            text = text[:m.start()].rstrip()
        if text.startswith('['):
            depth, end = 0, -1
            for i, ch in enumerate(text):
                depth += ch == '['
                depth -= ch == ']'
                if depth == 0:
                    end = i
                    break
            if end < 0:
                raise ValueError("unbalanced brackets in the matrix")
            A = np.array(self._parse_matrix(text[:end + 1]), dtype=complex)
            vec = text[end + 1:].strip()
            if not vec.startswith('['):
                raise ValueError("give the right-hand side as a list, e.g. [1, 0]")
            b: np.ndarray = np.array(self._parse_matrix(vec), dtype=complex).ravel()
        else:
            parts = text.split()
            if len(parts) < 6:
                raise ValueError("HHL a b c d v0 v1 [clock]  or  HHL [[...]] [v...] [CLOCK n]")
            vals = [float(self._eval_with_vars(p_, {})) for p_ in parts[:6]]
            if len(parts) > 6 and clock is None:
                clock = int(parts[6])
            A = np.array([[vals[0], vals[1]], [vals[2], vals[3]]], dtype=complex)
            b = np.array([vals[4], vals[5]], dtype=complex)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")
        if b.size != A.shape[0]:
            raise ValueError(f"A is {A.shape[0]}x{A.shape[0]} but v has {b.size} entries")
        return A, b, clock, shots

    def cmd_hhl(self, rest: str) -> None:
        """HHL [[A]] [v] [CLOCK n] [SHOTS n] — solve A x = v with the HHL algorithm (any size).

        Also HHL a b c d v0 v1 [clock] for a 2x2 system. A Hermitian A runs as
        given; any other invertible A runs as its Hermitian dilation
        [[0, A], [A^dag, 0]] on [v, 0], whose solution holds x. The system is
        padded to a power of two. HHL runs phase estimation of e^{iAt} on a
        signed clock register (negative eigenvalues read as two's complement),
        a uniformly controlled RY that rotates the ancilla by 2 arcsin(C/lambda),
        and uncomputation; as HHL assumes, the eigenvalue range is given
        classically: t maps the largest |lambda| to the clock's top value and
        C is 0.9 of the smallest |lambda|. Reports the post-selected solution,
        its fidelity to the classical A^-1 v and the post-selection success
        probability; SHOTS n also samples the solution register."""
        try:
            A, b, nclock, shots = self._parse_hhl_args(rest)
            n = A.shape[0]
            if np.linalg.norm(b) < 1e-12:
                raise ValueError("v must be nonzero")
            if abs(np.linalg.det(A)) < 1e-12:
                raise ValueError("A is singular")
            hermitian = bool(np.allclose(A, A.conj().T, atol=1e-10))
            if hermitian:
                H, rhs = A, b
            else:
                H = np.block([[np.zeros((n, n)), A], [A.conj().T, np.zeros((n, n))]])
                rhs = np.concatenate([b, np.zeros(n, dtype=complex)])
            m = H.shape[0]
            k = max(1, int(np.ceil(np.log2(m))))
            dim = 1 << k
            w0 = np.linalg.eigvalsh(H)
            if dim > m:                     # pad with the largest |eigenvalue|
                pad = float(np.max(np.abs(w0)))
                Hp = np.eye(dim, dtype=complex) * pad
                Hp[:m, :m] = H
                bp = np.zeros(dim, dtype=complex)
                bp[:m] = rhs
                H, rhs = Hp, bp
            if nclock is None:
                nclock = 7 if dim <= 2 else 8
            if not 3 <= nclock <= 12:
                raise ValueError("clock must be 3-12 qubits")
            if k + nclock + 1 > 24:
                raise ValueError(f"{k + nclock + 1} qubits is beyond the statevector HHL "
                                 f"(at most 24): use a smaller system or clock")
            w, V = np.linalg.eigh(H)
            x_full, success = self._hhl_solve(H, rhs, w, V, nclock)
            x_q = x_full[:m] if hermitian else x_full[n:2 * n]
            nq = np.linalg.norm(x_q)
            x_q = x_q / nq if nq > 0 else x_q
            x_cl = np.linalg.solve(A, b)
            x_cl = x_cl / np.linalg.norm(x_cl)
            overlap = float(abs(np.vdot(x_cl, x_q)) ** 2)

            def fmt(vec: np.ndarray) -> str:
                if np.allclose(vec.imag, 0, atol=1e-6):
                    return '[' + ', '.join(f"{v.real:+.3f}" for v in vec) + ']'
                return '[' + ', '.join(f"{v.real:+.3f}{v.imag:+.3f}j" for v in vec) + ']'

            shown = n <= 8
            label = '' if hermitian else ' (Hermitian dilation)'
            self.io.writeln(f"  HHL: {n}x{n} system{label}, {k} system + {nclock} clock + 1 "
                            f"ancilla qubits")
            if shown:
                self.io.writeln(f"  HHL solution (normalized): {fmt(x_q)}")
                self.io.writeln(f"  Classical A^-1 v:          {fmt(x_cl)}")
            self.io.writeln(f"  Direction fidelity = {overlap:.4f}   "
                            f"(post-selection succeeds with p = {success:.4f}, {nclock} clock qubits)")
            if shots:
                rng = np.random.default_rng(self._seed)
                kept = int(rng.binomial(shots, min(max(success, 0.0), 1.0)))
                probs = np.abs(x_q) ** 2
                probs = probs / probs.sum()
                draws = rng.multinomial(kept, probs) if kept else np.zeros(n, dtype=int)
                width = max(1, int(np.ceil(np.log2(max(n, 2)))))
                self.io.writeln(f"  Sampled {shots} shots: {kept} passed post-selection")
                for i in np.argsort(-draws)[:min(n, 16)]:
                    if draws[i]:
                        self.io.writeln(f"    |{format(int(i), f'0{width}b')}>  {int(draws[i]):>6}  "
                                        f"(ideal {probs[i]:.4f})")
            self.variables['_HHL_FIDELITY'] = overlap
            self.variables['_HHL_SUCCESS'] = float(success)
            self.arrays['_HHL_X'] = [complex(v) for v in x_q]
        except Exception as e:
            self.io.writeln(f"?HHL ERROR: {e}")

    def _hhl_solve(self, A: np.ndarray, b: np.ndarray, w: np.ndarray, V: np.ndarray, nclock: int) -> tuple[np.ndarray, float]:
        """Build and simulate the HHL circuit for a Hermitian A of size 2^k;
        returns (normalized x, success probability)."""
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UCRYGate, UnitaryGate
        from qiskit.quantum_info import Statevector
        dim = A.shape[0]
        k = int(round(np.log2(dim)))
        sys_q = list(range(k))
        clock = list(range(k, k + nclock))
        anc = k + nclock
        total = k + nclock + 1
        half = 1 << (nclock - 1)
        # e^{iAt} with the largest |lambda| at clock value half - 1, so every
        # eigenvalue lands in the signed range (-half, half).
        t = 2 * np.pi * (half - 1) / ((1 << nclock) * float(np.max(np.abs(w))))
        U = V @ np.diag(np.exp(1j * w * t)) @ V.conj().T
        qc = QuantumCircuit(total)
        qc.initialize(b / np.linalg.norm(b), sys_q)
        for c in clock:
            qc.h(c)
        for j, c in enumerate(clock):
            qc.append(UnitaryGate(np.linalg.matrix_power(U, 2 ** j)).control(1), [c] + sys_q)
        self._emit_qft(qc, clock, inverse=True, swaps=True)
        # C/lambda must stay <= 1 for every clock value phase estimation lands
        # on; 0.9 of the smallest |eigenvalue| keeps the spread clear of the
        # arcsin clip while post-selection still succeeds often.
        C = 0.9 * float(np.min(np.abs(w)))
        angles = [0.0]
        for kk in range(1, 1 << nclock):
            ks = kk if kk < half else kk - (1 << nclock)
            lam = 2 * np.pi * ks / ((1 << nclock) * t)
            angles.append(float(2 * np.arcsin(np.clip(C / lam, -1.0, 1.0))))
        qc.append(UCRYGate(angles), [anc] + clock)
        self._emit_qft(qc, clock, inverse=False, swaps=True)
        for j, c in enumerate(clock):
            qc.append(UnitaryGate(np.linalg.matrix_power(U.conj().T, 2 ** j)).control(1),
                      [c] + sys_q)
        for c in clock:
            qc.h(c)
        sv = Statevector(qc).data
        # Post-select ancilla = 1 and clock = 0; the system register is the low bits.
        base = 1 << (total - 1)
        x = np.array([sv[base | i] for i in range(dim)])
        success = float(np.vdot(x, x).real)
        if success < 1e-12:
            raise ValueError("post-selection amplitude vanished (add clock qubits)")
        return x / np.sqrt(success), success
