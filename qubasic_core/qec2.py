"""QUBASIC research-grade QEC extensions.

Three layers beyond the code-capacity lookup/union-find machinery in qec.py:

  Circuit-level noise  LOGICAL_ERROR_RATE SURFACE 21 0.001 CIRCUIT [shots]
                       stim-generated syndrome-extraction circuits (noisy
                       gates, measurements, resets over d rounds), decoded
                       from the detector error model by pymatching MWPM.
  Batched MWPM         LOGICAL_ERROR_RATE SURFACE 11 0.05 MWPM [trials]
                       code-capacity Monte Carlo with pymatching, vectorized
                       over trials (large distances the lookup table cannot).
  qLDPC / BB codes     QEC BB [l m]  and  LOGICAL_ERROR_RATE BB 0.01 [trials]
                       bivariate-bicycle codes (l=m=6 gives [[72,12,6]],
                       l=12 m=6 the [[144,12,12]] gross code) decoded by an
                       internal belief-propagation + ordered-statistics
                       (BP+OSD-0) decoder — no external dependency.

stim and pymatching are optional imports; commands that need them say so.
"""

from __future__ import annotations

import time

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# GF(2) linear algebra (dense, uint8)
# ═══════════════════════════════════════════════════════════════════════

def _gf2_rref(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row-echelon form over GF(2). Returns (R, pivot_columns)."""
    A = (M.copy() % 2).astype(np.uint8)
    r = 0
    pivots: list[int] = []
    for c in range(A.shape[1]):
        if r >= A.shape[0]:
            break
        hits = np.nonzero(A[r:, c])[0]
        if hits.size == 0:
            continue
        piv = r + hits[0]
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        mask = A[:, c].astype(bool)
        mask[r] = False
        A[mask] ^= A[r]
        pivots.append(c)
        r += 1
    return A[:r], pivots


def _gf2_rank(M: np.ndarray) -> int:
    if M.size == 0:
        return 0
    return _gf2_rref(M)[0].shape[0]


def _gf2_nullspace(M: np.ndarray) -> np.ndarray:
    """Basis of the right kernel of M over GF(2), one vector per row."""
    R, pivots = _gf2_rref(M)
    n = M.shape[1]
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(n, np.uint8)
        v[f] = 1
        for i, c in enumerate(pivots):
            if i < R.shape[0] and R[i, f]:
                v[c] = 1
        basis.append(v)
    return np.array(basis, np.uint8) if basis else np.zeros((0, n), np.uint8)


def _css_logicals(Hx: np.ndarray, Hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logical operator bases of a CSS code.

    X logicals span ker(Hz) modulo rowspace(Hx); Z logicals span ker(Hx)
    modulo rowspace(Hz). Greedy coset extraction: keep a kernel vector iff it
    grows the rank of the accumulated (stabilizer + kept logicals) matrix.
    """
    def coset_basis(kernel_of: np.ndarray, mod_rows: np.ndarray) -> np.ndarray:
        ker = _gf2_nullspace(kernel_of)
        acc = (mod_rows % 2).astype(np.uint8)
        base_rank = _gf2_rank(acc)
        out = []
        for v in ker:
            trial = np.vstack([acc, v[None, :]])
            rk = _gf2_rank(trial)
            if rk > base_rank:
                out.append(v)
                acc, base_rank = trial, rk
        return np.array(out, np.uint8) if out else np.zeros((0, kernel_of.shape[1]), np.uint8)

    return coset_basis(Hz, Hx), coset_basis(Hx, Hz)


def _css_from_strings(code: dict) -> dict:
    """Matrix form (Hx, Hz, Lx, Lz) of a CSS string-stabilizer code, the shape
    the BP+OSD decoder takes."""
    n = code['n']
    hx, hz = [], []
    for s in code['stab']:
        kinds = set(s) - {'I'}
        row = np.array([ch != 'I' for ch in s], dtype=np.uint8)
        if kinds == {'X'}:
            hx.append(row)
        elif kinds == {'Z'}:
            hz.append(row)
        else:
            raise ValueError("BP+OSD needs a CSS code (pure X or pure Z stabilizers)")
    Hx = np.array(hx, dtype=np.uint8) if hx else np.zeros((0, n), np.uint8)
    Hz = np.array(hz, dtype=np.uint8) if hz else np.zeros((0, n), np.uint8)
    Lx = np.array([[ch in 'XY' for ch in code['lx']]], dtype=np.uint8)
    Lz = np.array([[ch in 'ZY' for ch in code['lz']]], dtype=np.uint8)
    return {'n': n, 'Hx': Hx, 'Hz': Hz, 'Lx': Lx, 'Lz': Lz, 'name': code['name'],
            'alphabet': code.get('alphabet', 'IXYZ')}


# ═══════════════════════════════════════════════════════════════════════
# Bivariate-bicycle codes (Bravyi et al. family)
# ═══════════════════════════════════════════════════════════════════════

def _bb_code(l: int = 6, m: int = 6) -> dict:
    """Bivariate-bicycle CSS code on a 2*l*m qubit torus.

    A = x^3 + y + y^2 and B = y^3 + x + x^2 with x, y the cyclic shifts of
    Z_l and Z_m; Hx = [A|B], Hz = [B^T|A^T]. l=m=6 gives [[72,12,6]], l=12
    m=6 the [[144,12,12]] "gross" code. k is computed from GF(2) ranks, not
    assumed.
    """
    Sl = np.roll(np.eye(l, dtype=np.uint8), 1, axis=1)
    Sm = np.roll(np.eye(m, dtype=np.uint8), 1, axis=1)
    x = np.kron(Sl, np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), Sm)
    mp = lambda M, k: np.linalg.matrix_power(M.astype(np.uint8), k) % 2
    A = (mp(x, 3) + y + mp(y, 2)) % 2
    B = (mp(y, 3) + x + mp(x, 2)) % 2
    Hx = np.concatenate([A, B], axis=1).astype(np.uint8)
    Hz = np.concatenate([B.T, A.T], axis=1).astype(np.uint8)
    n = 2 * l * m
    k = n - _gf2_rank(Hx) - _gf2_rank(Hz)
    Lx, Lz = _css_logicals(Hx, Hz)
    known_d = {(6, 6): 6, (12, 6): 12, (9, 6): 8, (12, 12): 12, (15, 3): 8}
    d = known_d.get((l, m))
    basis_bound = int(min(int(v.sum()) for v in np.vstack([Lx, Lz]))) if k else 0
    return {'n': n, 'k': k, 'd': d, 'd_bound': basis_bound, 'l': l, 'm': m,
            'Hx': Hx, 'Hz': Hz, 'Lx': Lx, 'Lz': Lz, 'type': 'css_matrix',
            'name': f'bivariate bicycle l={l} m={m}'}


# ═══════════════════════════════════════════════════════════════════════
# Belief propagation + ordered-statistics decoding (BP+OSD-0)
# ═══════════════════════════════════════════════════════════════════════

def _bp_decode(H: np.ndarray, syndrome: np.ndarray, p: float,
               max_iter: int = 30, alpha: float = 0.8):
    """Normalized min-sum belief propagation for syndrome decoding.

    Returns (error_estimate, posterior_llrs, converged). The check-node sign
    convention absorbs the syndrome, so a satisfied decoding has H e = s.
    """
    r, n = H.shape
    rows, cols = np.nonzero(H)
    L0 = float(np.log((1.0 - p) / max(p, 1e-12)))
    edge_by_check = [np.where(rows == c)[0] for c in range(r)]
    edge_by_bit = [np.where(cols == b)[0] for b in range(n)]
    M_bc = np.full(len(rows), L0)
    M_cb = np.zeros(len(rows))
    Lq = np.full(n, L0)
    sgn_syn = 1.0 - 2.0 * syndrome.astype(float)
    e_hat = np.zeros(n, np.uint8)
    for _ in range(max_iter):
        for c in range(r):
            e = edge_by_check[c]
            vals = M_bc[e]
            signs = np.where(vals >= 0, 1.0, -1.0)
            absv = np.abs(vals)
            order = np.argsort(absv)
            m1 = absv[order[0]]
            m2 = absv[order[1]] if e.size > 1 else m1
            prod_sign = float(np.prod(signs)) * sgn_syn[c]
            mins = np.full(e.size, m1)
            mins[order[0]] = m2
            M_cb[e] = alpha * prod_sign * signs * mins
        for b in range(n):
            eb = edge_by_bit[b]
            tot = L0 + float(np.sum(M_cb[eb]))
            Lq[b] = tot
            M_bc[eb] = tot - M_cb[eb]
        e_hat = (Lq < 0).astype(np.uint8)
        if np.array_equal(H.dot(e_hat) % 2, syndrome):
            return e_hat, Lq, True
    return e_hat, Lq, False


def _osd0(H: np.ndarray, syndrome: np.ndarray, Lq: np.ndarray) -> np.ndarray:
    """Ordered-statistics decoding, order 0.

    Columns are ranked most-likely-in-error first (ascending posterior LLR),
    Gaussian elimination picks a pivot set concentrated on those columns, and
    the syndrome is solved with all non-pivot bits at 0.
    """
    n = H.shape[1]
    order = np.argsort(Lq)
    Hw = np.concatenate([H[:, order], syndrome[:, None]], axis=1).astype(np.uint8)
    row = 0
    pivots: list[int] = []
    for col in range(n):
        if row >= Hw.shape[0]:
            break
        hits = np.nonzero(Hw[row:, col])[0]
        if hits.size == 0:
            continue
        piv = row + hits[0]
        if piv != row:
            Hw[[row, piv]] = Hw[[piv, row]]
        mask = Hw[:, col].astype(bool)
        mask[row] = False
        Hw[mask] ^= Hw[row]
        pivots.append(col)
        row += 1
    e_perm = np.zeros(n, np.uint8)
    for i, col in enumerate(pivots):
        e_perm[col] = Hw[i, -1]
    e = np.zeros(n, np.uint8)
    e[order] = e_perm
    return e


# ═══════════════════════════════════════════════════════════════════════
# Mixin
# ═══════════════════════════════════════════════════════════════════════

class QEC2Mixin:
    """Circuit-level QEC, MWPM decoding, and qLDPC codes for QBasicTerminal.

    Requires: TerminalProtocol — uses self.io, self._seed, self.variables,
    and the string-stabilizer codes from QECMixin (self._qec_code).
    """

    # ── CSS parity matrices from string-stabilizer codes ────────────────

    def _qec_css_matrices(self, code: dict):
        """(Hz, Hx, lz_support, lx_support) as uint8 arrays from a string code.

        Hz rows are the Z-type stabilizers (they detect X errors), Hx rows the
        X-type ones (they detect Z errors).
        """
        n = code['n']
        Hz_rows, Hx_rows = [], []
        for s in code['stab']:
            kinds = set(s) - {'I'}
            row = np.array([0 if ch == 'I' else 1 for ch in s], np.uint8)
            if kinds == {'Z'}:
                Hz_rows.append(row)
            elif kinds == {'X'}:
                Hx_rows.append(row)
            else:
                raise ValueError("MWPM path needs a CSS code (pure X/Z stabilizers)")
        Hz = np.array(Hz_rows, np.uint8) if Hz_rows else np.zeros((0, n), np.uint8)
        Hx = np.array(Hx_rows, np.uint8) if Hx_rows else np.zeros((0, n), np.uint8)
        lz = np.array([1 if ch == 'Z' else 0 for ch in code['lz']], np.uint8)
        lx = np.array([1 if ch == 'X' else 0 for ch in code['lx']], np.uint8)
        return Hz, Hx, lz, lx

    # ── Batched code-capacity MWPM (pymatching) ─────────────────────────

    def _qec_mwpm_rate(self, code: dict, p: float, trials: int, rng) -> float:
        import pymatching
        Hz, Hx, lz, lx = self._qec_css_matrices(code)
        n = code['n']
        if code['alphabet'] == 'IX':
            ex = (rng.random((trials, n)) < p).astype(np.uint8)
            ez = np.zeros_like(ex)
        else:
            hit = rng.random((trials, n)) < p
            which = rng.integers(0, 3, (trials, n))
            ex = (hit & ((which == 0) | (which == 1))).astype(np.uint8)
            ez = (hit & ((which == 2) | (which == 1))).astype(np.uint8)
        fail = np.zeros(trials, bool)
        for H, l_opp, e in ((Hz, lz, ex), (Hx, lx, ez)):
            if H.shape[0] == 0:
                continue
            if int(H.sum(axis=0).max()) > 2:
                raise ValueError("MWPM needs a matchable code (each qubit in <=2 "
                                 "same-type checks); use the lookup/UF/BP decoders")
            matcher = pymatching.Matching(H)
            synd = e.dot(H.T) % 2
            rec = matcher.decode_batch(synd).astype(np.uint8)
            resid = e ^ rec
            fail |= (resid.dot(l_opp) % 2).astype(bool)
        return float(np.mean(fail))

    # ── Circuit-level noise (stim + pymatching) ─────────────────────────

    _STIM_TASKS = {
        'REP': 'repetition_code:memory',
        'REPETITION': 'repetition_code:memory',
        'BITFLIP': 'repetition_code:memory',
        'SURFACE': 'surface_code:rotated_memory_z',
        'SURF': 'surface_code:rotated_memory_z',
    }

    def _qec_circuit_level(self, name: str, d: int, p: float,
                           shots: int, rounds: int | None) -> None:
        try:
            import stim
            import pymatching
        except ImportError:
            self.io.writeln("?CIRCUIT-level QEC needs stim + pymatching "
                            "(pip install stim pymatching)")
            return
        task = self._STIM_TASKS.get(name.upper())
        if task is None:
            self.io.writeln(f"?CIRCUIT mode supports REP and SURFACE (got {name})")
            return
        if d % 2 == 0:
            d += 1
        rounds = rounds or d
        circuit = stim.Circuit.generated(
            task, distance=d, rounds=rounds,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p)
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler(
            seed=self._seed if self._seed is not None else None)
        t0 = time.time()
        dets, obs = sampler.sample(shots, separate_observables=True)
        preds = matcher.decode_batch(dets)
        errs = int(np.sum(np.any(preds != obs, axis=1)))
        dt = time.time() - t0
        per_shot = errs / shots
        per_round = 1.0 - (1.0 - per_shot) ** (1.0 / rounds) if per_shot < 1.0 else 1.0
        label = 'rotated surface' if 'surface' in task else 'repetition'
        self.io.writeln(f"\n  {label} d={d}: CIRCUIT-level noise p={p}, "
                        f"{rounds} rounds, {shots:,} shots")
        self.io.writeln(f"  physical qubits = {circuit.num_qubits}  "
                        f"detectors = {dem.num_detectors}  (stim + pymatching MWPM)")
        self.io.writeln(f"  Logical error rate = {per_shot:.3e} per shot   "
                        f"{per_round:.3e} per round")
        self.io.writeln(f"  ({dt:.2f}s: {shots / max(dt, 1e-9):,.0f} shots/s)")
        self.variables['_LER'] = per_shot
        self.variables['_LER_ROUND'] = per_round

    # ── qLDPC (BB) code-capacity with BP+OSD ────────────────────────────

    def _qec_bb_rate(self, code: dict, p: float, trials: int, rng) -> float:
        Hx, Hz, Lx, Lz = code['Hx'], code['Hz'], code['Lx'], code['Lz']
        n = code['n']
        bitflip = code.get('alphabet') == 'IX'
        # Depolarizing: marginal X (or Z) flip probability is 2p/3; a bit-flip
        # code sees X errors at rate p only.
        p_marg = p if bitflip else 2.0 * p / 3.0
        fails = 0
        for _ in range(trials):
            if bitflip:
                ex = (rng.random(n) < p).astype(np.uint8)
                ez = np.zeros(n, dtype=np.uint8)
            else:
                hit = rng.random(n) < p
                which = rng.integers(0, 3, n)
                ex = (hit & ((which == 0) | (which == 1))).astype(np.uint8)
                ez = (hit & ((which == 2) | (which == 1))).astype(np.uint8)
            failed = False
            for H, L, e in ((Hz, Lz, ex), (Hx, Lx, ez)):
                s = H.dot(e) % 2
                e_hat, Lq, ok = _bp_decode(H, s, p_marg)
                if not ok:
                    e_hat = _osd0(H, s, Lq)
                resid = e ^ e_hat
                if np.any(L.dot(resid) % 2):
                    failed = True
                    break
            fails += failed
        return fails / trials

    def _qec_show_bb(self, l: int, m: int) -> None:
        code = _bb_code(l, m)
        n, k = code['n'], code['k']
        d_txt = str(code['d']) if code['d'] else f"unknown (<= {code['d_bound']} from the logical basis)"
        self.io.writeln(f"\n  {code['name']}: [[{n},{k},{d_txt}]]")
        self.io.writeln(f"  Hx: {code['Hx'].shape[0]} X-checks, weight "
                        f"{int(code['Hx'].sum(axis=1).max())}; Hz: {code['Hz'].shape[0]} "
                        f"Z-checks, weight {int(code['Hz'].sum(axis=1).max())}")
        css_ok = not np.any(code['Hx'].dot(code['Hz'].T) % 2)
        self.io.writeln(f"  CSS commutation Hx Hz^T = 0: {css_ok}")
        self.io.writeln(f"  k from GF(2) ranks = {k}; logical basis {code['Lx'].shape[0]} X"
                        f" + {code['Lz'].shape[0]} Z operators")
        self.io.writeln(f"  Decode with: LOGICAL_ERROR_RATE BB {l} {m} <p> [trials]  (BP+OSD)")
