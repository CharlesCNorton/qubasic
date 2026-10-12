"""QUBASIC variational algorithms: QAOA and VQE as commands.

    QAOA MAXCUT 0-1, 1-2, 2-0:2 [P p] [METHOD m] [SHOTS n] [PROGRAM]
    QAOA RING 6 | QAOA COMPLETE 5 | QAOA <diagonal HAMILTONIAN> ...
    VQE <HAMILTONIAN> [LAYERS L] [ANSATZ RY|RYRZ] [METHOD m] [RESTARTS k] [PROGRAM]

Both simulate their ansatz on a numpy statevector and optimize with the
optimizers MINIMIZE uses (Nelder-Mead, SPSA, parameter-shift gradient
descent) or L-BFGS with parameter-shift gradients (the default).

QAOA maximizes a diagonal objective C (the cut value of a weighted graph, or
minus the energy of a Hamiltonian of Z terms): p layers of e^{-i gamma C}
and the X mixer e^{-i beta sum X}, starting from |+...+>. Depth p is reached
layer by layer, each optimum interpolated into the next depth (Zhou et al.,
2020). The report gives <C>, the exact optimum by enumeration, the
approximation ratio, the probability of measuring an optimal solution and a
sample of the output.

VQE minimizes <H> over a hardware-efficient ansatz (RY, or RY and RZ, on
every qubit between layers of a CX chain), with random restarts, and reports
the energy beside the exact ground energy.

PROGRAM writes the optimized circuit into the program (QUBITS set to match),
so it can be RUN, sampled under NOISE, or exported.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
from qubasic_core.host import HostState

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
        ANSATZ, PROGRAM) from a command line."""
        opts: dict[str, Any] = {}
        text = rest
        for key, conv in (('P', int), ('LAYERS', int), ('ITERS', int), ('SHOTS', int),
                          ('RESTARTS', int), ('METHOD', str), ('ANSATZ', str)):
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

        def counted(v):
            evals[0] += 1
            return f(np.asarray(v, dtype=float))

        if method in ('LBFGS', 'L-BFGS', 'BFGS'):
            from scipy.optimize import minimize

            def gradient(v):
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
    def _qaoa_state(obj: np.ndarray, n: int, gammas, betas) -> np.ndarray:
        psi = np.full(1 << n, 1 / np.sqrt(1 << n), dtype=complex)
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * obj)
            psi = _apply_rx_all(psi, n, b)
        return psi

    def _qaoa_optimize(self, obj: np.ndarray, n: int, p: int, method: str, iters: int):
        """Optimal (gammas, betas, <C>, evaluations) at depth p, grown layer by layer."""
        def neg(v):
            k = len(v) // 2
            psi = self._qaoa_state(obj, n, v[:k], v[k:])
            return -float(np.dot(np.abs(psi) ** 2, obj))

        def grad(v):
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

            def interp(v):
                ext = np.concatenate([[0.0], v, [0.0]])
                return np.array([(i / k) * ext[i] + ((k - i) / k) * ext[i + 1]
                                 for i in range(depth)])
            x0 = np.concatenate([interp(gam), interp(bet)])
            x, fval, e = self._optimize(neg, x0, method, iters,
                                        grad if method in ('LBFGS', 'L-BFGS', 'BFGS') else None)
            evals += e
        return x[:p], x[p:], -fval, evals

    def cmd_qaoa(self, rest: str) -> None:
        """QAOA <graph | HAMILTONIAN> [P p] [METHOD m] [ITERS n] [SHOTS n] [PROGRAM] — QAOA optimization.

        The graph is 'a-b[:w], ...', RING n, COMPLETE n or RANDOM n [p] (MAXCUT
        may precede it); a declared HAMILTONIAN of Z terms is minimized
        instead. See qubasic_core/variational.py."""
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

    def _qaoa_program(self, n: int, terms: list, gammas, betas) -> None:
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
        """VQE <HAMILTONIAN> [LAYERS L] [ANSATZ RY|RYRZ] [METHOD m] [RESTARTS k] [PROGRAM] — variational ground energy.

        Minimizes <H> over a hardware-efficient ansatz (L layers of a CX chain
        between rotation layers, default L=2) and compares with the exact
        ground energy. See qubasic_core/variational.py."""
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

            def energy(v):
                psi = self._ansatz_state(n, layers, v, kind)
                return float(np.real(np.vdot(psi, Hs @ psi)))

            def grad(v):
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
            if opts.get('program'):
                self._vqe_program(n, layers, params, kind, name)
        except Exception as e:
            self.io.writeln(f"?VQE ERROR: {e}")

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
