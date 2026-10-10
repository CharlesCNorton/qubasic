"""QUBASIC quantum error correction.

Stabilizer codes with an optimal (minimum-weight lookup) decoder, logical error
rates, and threshold sweeps, computed by classical Pauli-frame simulation (no
statevector needed, so it is exact and fast). Built-in codes: the repetition
(bit-flip) code at any odd distance, the [[7,1,3]] Steane code, and the [[9,1,3]]
Shor code.

  QEC LIST                       List the built-in codes
  QEC STEANE                     Show a code's stabilizers and logical operators
  LOGICAL_ERROR_RATE STEANE 0.05 Monte-Carlo logical error rate at physical rate p
  THRESHOLD REP 0.0 0.5 11       Sweep p and compare code distances (find the crossing)
"""

from __future__ import annotations

import re
import itertools

import numpy as np

# Phase-free single-qubit Pauli multiplication (we only need the group structure).
_PMUL = {
    ('I', 'I'): 'I', ('I', 'X'): 'X', ('I', 'Y'): 'Y', ('I', 'Z'): 'Z',
    ('X', 'I'): 'X', ('X', 'X'): 'I', ('X', 'Y'): 'Z', ('X', 'Z'): 'Y',
    ('Y', 'I'): 'Y', ('Y', 'X'): 'Z', ('Y', 'Y'): 'I', ('Y', 'Z'): 'X',
    ('Z', 'I'): 'Z', ('Z', 'X'): 'Y', ('Z', 'Y'): 'X', ('Z', 'Z'): 'I',
}


def _anticommute(a: str, b: str) -> int:
    """1 if Pauli strings a and b anticommute, else 0."""
    cnt = 0
    for pa, pb in zip(a, b):
        if pa != 'I' and pb != 'I' and pa != pb:
            cnt ^= 1
    return cnt


def _pmul(a: str, b: str) -> str:
    # Precondition: both strings over the IXYZ alphabet. The try keeps the
    # per-trial hot path free of per-character validation.
    try:
        return ''.join(_PMUL[(pa, pb)] for pa, pb in zip(a, b))
    except KeyError as e:
        raise ValueError(f"Pauli strings must be over IXYZ (got {e.args[0]})") from None


def _weight(p: str) -> int:
    return sum(1 for c in p if c != 'I')


class QECMixin:
    """Stabilizer codes, decoding, and logical error rates for QBasicTerminal.

    Requires: TerminalProtocol — uses self._eval_with_vars(), self._seed, self.io.
    """

    def _qec_code(self, name: str, distance: int = 3) -> dict:
        """Return a code descriptor: data qubit count, stabilizers, logical X/Z,
        distance, and the error alphabet the decoder corrects. Descriptors are
        cached per (name, distance), so decoder tables built on them persist."""
        cache = self.__dict__.setdefault('_qec_cache', {})
        key = (name.upper(), distance)
        if key not in cache:
            cache[key] = self._qec_code_build(name, distance)
        return cache[key]

    def _qec_code_build(self, name: str, distance: int = 3) -> dict:
        name = name.upper()
        if name in ('REP', 'REPETITION', 'BITFLIP'):
            d = distance if distance % 2 == 1 else distance + 1
            stab = []
            for i in range(d - 1):
                s = ['I'] * d
                s[i] = 'Z'; s[i + 1] = 'Z'
                stab.append(''.join(s))
            return {'n': d, 'stab': stab, 'lx': 'X' * d, 'lz': 'Z' + 'I' * (d - 1),
                    'd': d, 'alphabet': 'IX', 'name': f'repetition d={d}'}
        if name == 'STEANE':
            Hm = [[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]]
            stab = [''.join('X' if b else 'I' for b in row) for row in Hm]
            stab += [''.join('Z' if b else 'I' for b in row) for row in Hm]
            return {'n': 7, 'stab': stab, 'lx': 'X' * 7, 'lz': 'Z' * 7,
                    'd': 3, 'alphabet': 'IXYZ', 'name': 'Steane [[7,1,3]]'}
        if name == 'SHOR':
            stab = []
            for b in range(3):
                base = b * 3
                for a, c in ((base, base + 1), (base + 1, base + 2)):
                    s = ['I'] * 9; s[a] = 'Z'; s[c] = 'Z'; stab.append(''.join(s))
            stab.append('XXXXXX' + 'III')
            stab.append('III' + 'XXXXXX')
            return {'n': 9, 'stab': stab, 'lx': 'X' * 9, 'lz': 'ZIIZIIZII',
                    'd': 3, 'alphabet': 'IXYZ', 'name': 'Shor [[9,1,3]]'}
        if name in ('SURFACE', 'SURF'):
            return self._surface_code(distance)
        raise ValueError(f"unknown code '{name}' (try REP, STEANE, SHOR, SURFACE; QEC LIST)")

    def _surface_code(self, d: int) -> dict:
        """Rotated surface code of odd distance d (d^2 data qubits on a d x d grid).

        Bulk faces are weight-4 stabilizers alternating X/Z by checkerboard parity;
        boundary faces are weight-2, X on the top/bottom edges and Z on the
        left/right edges. Logical X is an X string along the top row, logical Z a
        Z string down the left column. The construction is validated by the
        caller's checks (it must correct every weight-1 error)."""
        if d % 2 == 0:
            d += 1
        n = d * d

        def idx(r, c):
            return r * d + c

        stab = []
        for r in range(-1, d):
            for c in range(-1, d):
                corners = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
                qs = [idx(i, j) for (i, j) in corners if 0 <= i < d and 0 <= j < d]
                if len(qs) < 2:
                    continue
                typ = 'Z' if (r + c) % 2 == 0 else 'X'
                if len(qs) == 4:
                    stab.append((typ, qs))
                else:
                    on_tb = (r == -1 or r == d - 1)
                    on_lr = (c == -1 or c == d - 1)
                    if on_tb and not on_lr and typ == 'X':
                        stab.append((typ, qs))
                    elif on_lr and not on_tb and typ == 'Z':
                        stab.append((typ, qs))
        stab_strs = []
        for typ, qs in stab:
            s = ['I'] * n
            for q in qs:
                s[q] = typ
            stab_strs.append(''.join(s))
        # Logical X runs down a column (between the top/bottom X boundaries);
        # logical Z runs along a row (between the left/right Z boundaries). They
        # cross at qubit 0 and so anticommute.
        lx = ['I'] * n
        for r in range(d):
            lx[idx(r, 0)] = 'X'
        lz = ['I'] * n
        for c in range(d):
            lz[idx(0, c)] = 'Z'
        return {'n': n, 'stab': stab_strs, 'lx': ''.join(lx), 'lz': ''.join(lz),
                'd': d, 'alphabet': 'IXYZ', 'name': f'rotated surface d={d}'}

    # ── Binary (symplectic) form of a string code ─────────────────────
    #
    # A Pauli on n qubits is a pair of bit vectors (x, z): X=(1,0), Z=(0,1),
    # Y=(1,1). Two Paulis anticommute when x1.z2 + z1.x2 is odd, so the
    # syndrome of an error e against stabilizer rows (Sx, Sz) is
    # e_x Sz^T + e_z Sx^T (mod 2).

    def _qec_binary(self, code: dict) -> dict:
        """Cached symplectic matrices and logical vectors of a string code."""
        if '_bin' in code:
            return code['_bin']

        def vec(s: str):
            x = np.array([ch in 'XY' for ch in s], dtype=np.uint8)
            z = np.array([ch in 'ZY' for ch in s], dtype=np.uint8)
            return x, z

        xs, zs = zip(*(vec(s) for s in code['stab']))
        lx = vec(code['lx'])
        lz = vec(code['lz'])
        from qubasic_core.qec2 import _gf2_rank
        sx, sz = np.array(xs), np.array(zs)
        b = {'Sx': sx, 'Sz': sz, 'LX': lx, 'LZ': lz,
             'rank': _gf2_rank(np.concatenate([sx, sz], axis=1))}
        code['_bin'] = b
        return b

    # Lookup tables are built while the reachable syndrome space is at most
    # this large; past it the decoder falls back to matching.
    LOOKUP_MAX_SYNDROMES = 1 << 16

    def _lookup_feasible(self, code: dict) -> bool:
        b = self._qec_binary(code)
        return code['n'] <= 64 and (1 << b['rank']) <= self.LOOKUP_MAX_SYNDROMES

    def _qec_lookup_table(self, code: dict) -> dict:
        """Minimum-weight lookup decoder: syndrome (as an int, stabilizer i =
        bit i) -> (x_mask, z_mask) recovery, found by enumerating errors in
        order of increasing weight until every reachable syndrome is covered.
        Cached on the code."""
        if '_lookup' in code:
            return code['_lookup']
        if not self._lookup_feasible(code):
            raise ValueError(f"{code['name']}: lookup table would need "
                             f"2^{self._qec_binary(code)['rank']} syndromes")
        b = self._qec_binary(code)
        n, m = code['n'], len(code['stab'])
        # Syndrome contribution of X and Z on each qubit, as bitmasks.
        cx = [sum(int(b['Sz'][i, q]) << i for i in range(m)) for q in range(n)]
        cz = [sum(int(b['Sx'][i, q]) << i for i in range(m)) for q in range(n)]
        paulis = [('X', 1, 0), ('Y', 1, 1), ('Z', 0, 1)]
        paulis = [p for p in paulis if p[0] in code['alphabet']]
        target = 1 << b['rank']
        table: dict = {0: (0, 0)}
        for w in range(1, n + 1):
            if len(table) >= target:
                break
            for support in itertools.combinations(range(n), w):
                for assign in itertools.product(paulis, repeat=w):
                    synd = xm = zm = 0
                    for q, (_, px, pz) in zip(support, assign):
                        if px:
                            synd ^= cx[q]
                            xm |= 1 << q
                        if pz:
                            synd ^= cz[q]
                            zm |= 1 << q
                    if synd not in table:
                        table[synd] = (xm, zm)
                if len(table) >= target:
                    break
        code['_lookup'] = table
        return table

    def _qec_decoder(self, code: dict) -> dict:
        """The minimum-weight lookup decoder as syndrome tuple -> recovery
        Pauli string (one entry per reachable syndrome)."""
        n, m = code['n'], len(code['stab'])
        out = {}
        for synd, (xm, zm) in self._qec_lookup_table(code).items():
            key = tuple((synd >> i) & 1 for i in range(m))
            out[key] = ''.join('Y' if (xm >> q) & 1 and (zm >> q) & 1 else
                               'X' if (xm >> q) & 1 else
                               'Z' if (zm >> q) & 1 else 'I' for q in range(n))
        return out

    # ── Matching decoder (exact minimum-weight perfect matching) ──────

    def _match_graph(self, code: dict) -> list:
        """Per error type, the detector graph of a matchable CSS code: the
        detectors, and shortest-path lengths and qubit paths between every
        pair of detectors and from each detector to the boundary. Cached."""
        if '_match' in code:
            return code['_match']
        from collections import deque
        n, stabs = code['n'], code['stab']
        graphs = []
        for etype, dtype in (('X', 'Z'), ('Z', 'X')):
            det = [i for i, s in enumerate(stabs) if set(s) - {'I'} == {dtype}]
            if not det:
                continue
            pos = {d: k for k, d in enumerate(det)}
            B = len(det)
            adj: list[list] = [[] for _ in range(B + 1)]
            for q in range(n):
                touch = [pos[i] for i in det if stabs[i][q] == dtype]
                if len(touch) > 2:
                    raise ValueError("matching needs each qubit in at most two checks of a type")
                if len(touch) == 2:
                    adj[touch[0]].append((touch[1], q))
                    adj[touch[1]].append((touch[0], q))
                elif len(touch) == 1:
                    adj[touch[0]].append((B, q))
                    adj[B].append((touch[0], q))
            dist = np.full((B + 1, B + 1), -1, dtype=np.int64)
            paths: dict = {}
            for src in range(B):
                # BFS parents: node -> (parent node, qubit on the edge to it)
                prev: dict[int, tuple[int, int]] = {src: (src, -1)}
                dq = deque([src])
                while dq:
                    u = dq.popleft()
                    for v, q in adj[u]:
                        if v not in prev:
                            prev[v] = (u, q)
                            if v != B:
                                dq.append(v)      # the boundary is a sink, not a relay
                for tgt in prev:
                    qs: list[int] = []
                    node = tgt
                    while node != src:
                        node, q = prev[node]
                        qs.append(q)
                    dist[src, tgt] = len(qs)
                    paths[(src, tgt)] = qs
            graphs.append({'etype': etype, 'det': det, 'dist': dist, 'paths': paths, 'B': B})
        code['_match'] = graphs
        return graphs

    def _match_recovery(self, code: dict, synd_bits) -> tuple[np.ndarray, np.ndarray]:
        """(rx, rz) recovery from exact minimum-weight perfect matching of the
        syndrome defects (blossom algorithm via rustworkx); each defect pairs
        with another or with the boundary through its own boundary copy."""
        import rustworkx as rx_
        n = code['n']
        rx_v = np.zeros(n, dtype=np.uint8)
        rz_v = np.zeros(n, dtype=np.uint8)
        for g in self._match_graph(code):
            defects = [k for k, i in enumerate(g['det']) if synd_bits[i]]
            if not defects:
                continue
            k = len(defects)
            dist, B = g['dist'], g['B']
            wmax = int(dist.max()) + 1
            graph = rx_.PyGraph()
            graph.add_nodes_from(range(2 * k))
            for a in range(k):
                for c in range(a + 1, k):
                    d = dist[defects[a], defects[c]]
                    if d >= 0:
                        graph.add_edge(a, c, wmax - int(d))
                    graph.add_edge(k + a, k + c, wmax)        # boundary copies pair freely
                db = dist[defects[a], B]
                if db >= 0:
                    graph.add_edge(a, k + a, wmax - int(db))
            matching = rx_.max_weight_matching(graph, max_cardinality=True,
                                               weight_fn=lambda w: w)
            target = rx_v if g['etype'] == 'X' else rz_v
            for u, v in matching:
                u, v = min(u, v), max(u, v)
                if u >= k:
                    continue                                   # boundary-boundary
                src = defects[u]
                dst = B if v >= k else defects[v]
                for q in g['paths'].get((src, dst), ()):
                    target[q] ^= 1
        return rx_v, rz_v

    def _qec_matching_decode(self, code: dict, syndrome) -> str:
        """Matching decoder on one syndrome; returns the recovery Pauli string."""
        rxv, rzv = self._match_recovery(code, list(syndrome))
        return ''.join('Y' if a and b else 'X' if a else 'Z' if b else 'I'
                       for a, b in zip(rxv, rzv))

    def _matchable(self, code: dict) -> bool:
        try:
            self._match_graph(code)
            return all(set(s) - {'I'} in ({'X'}, {'Z'}) for s in code['stab'])
        except ValueError:
            return False

    # ── Monte Carlo logical error rate ─────────────────────────────────

    def _choose_decoder(self, code: dict, requested: str | None = None) -> str:
        """'lookup', 'match' or 'pymatching' for a string code: the requested
        one when it applies, else lookup while feasible, else matching."""
        if requested in ('lookup',) and self._lookup_feasible(code):
            return 'lookup'
        if requested == 'pymatching':
            try:
                import pymatching  # noqa: F401
                return 'pymatching'
            except ImportError:
                requested = 'match'
        if requested == 'match':
            if not self._matchable(code):
                raise ValueError(f"{code['name']} is not matchable (each qubit must be in "
                                 f"at most two checks of a type); use LOOKUP or BP")
            return 'match'
        if requested == 'lookup':
            raise ValueError(f"{code['name']}: lookup table would need "
                             f"2^{self._qec_binary(code)['rank']} syndromes; use MATCH")
        if self._lookup_feasible(code):
            return 'lookup'
        if self._matchable(code):
            return 'match'
        raise ValueError(f"{code['name']}: no decoder fits (lookup too large, not matchable)")

    _DECODER_LABEL = {'lookup': 'lookup decoder', 'match': 'blossom matching decoder',
                      'pymatching': 'pymatching MWPM decoder'}

    def _logical_error_rate(self, code: dict, p: float, trials: int, rng,
                            uf: bool = False, decoder: str | None = None) -> float:
        """Code-capacity Monte Carlo: i.i.d. errors (X for a bit-flip code,
        depolarizing otherwise), decode, count residual logical errors.
        Vectorized over trials; each distinct syndrome is decoded once."""
        decoder = self._choose_decoder(code, 'match' if uf else decoder)
        if decoder == 'pymatching':
            return self._qec_mwpm_rate(code, p, trials, rng)
        b = self._qec_binary(code)
        n = code['n']
        if code['alphabet'] == 'IX':
            ex = (rng.random((trials, n)) < p).astype(np.uint8)
            ez = np.zeros_like(ex)
        else:
            hit = rng.random((trials, n)) < p
            which = rng.integers(0, 3, (trials, n))          # 0 X, 1 Y, 2 Z
            ex = (hit & (which != 2)).astype(np.uint8)
            ez = (hit & (which != 0)).astype(np.uint8)
        synd = ((ex.astype(np.int64) @ b['Sz'].T.astype(np.int64)
                 + ez.astype(np.int64) @ b['Sx'].T.astype(np.int64)) & 1).astype(np.uint8)
        keys, inverse = np.unique(np.packbits(synd, axis=1, bitorder='little'),
                                  axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        m = synd.shape[1]
        rec_x = np.zeros((len(keys), n), dtype=np.uint8)
        rec_z = np.zeros((len(keys), n), dtype=np.uint8)
        table = self._qec_lookup_table(code) if decoder == 'lookup' else None
        for k, packed in enumerate(keys):
            bits = np.unpackbits(packed, bitorder='little')[:m]
            if table is not None:
                s = int(sum(int(v) << i for i, v in enumerate(bits)))
                xm, zm = table.get(s, (0, 0))
                rec_x[k] = [(xm >> q) & 1 for q in range(n)]
                rec_z[k] = [(zm >> q) & 1 for q in range(n)]
            else:
                rec_x[k], rec_z[k] = self._match_recovery(code, bits)
        res_x = ex ^ rec_x[inverse]
        res_z = ez ^ rec_z[inverse]
        lxx, lxz = b['LX']
        lzx, lzz = b['LZ']
        fail = (((res_x.astype(np.int64) @ lxz + res_z.astype(np.int64) @ lxx) & 1)
                | ((res_x.astype(np.int64) @ lzz + res_z.astype(np.int64) @ lzx) & 1))
        return float(np.mean(fail))

    def cmd_distill(self, rest: str) -> None:
        """DISTILL <p> [trials] — 15-to-1 magic-state (T-state) distillation.

        Models the protocol's error detection by the [15,11,3] Hamming code:
        15 noisy T-states (Z error rate p), accept on a trivial syndrome, and a
        logical output error occurs only for an undetected nonzero codeword. The
        output error rate scales as ~35 p^3, well below the input p."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: DISTILL <p> [trials]")
            return
        try:
            p = float(self._eval_with_vars(parts[0], {}))
            trials = int(parts[1]) if len(parts) > 1 else 200000
            # [15,11,3] Hamming parity-check matrix: columns are 1..15 in binary.
            H = np.array([[ (c >> b) & 1 for c in range(1, 16)] for b in range(4)])
            rng = np.random.default_rng(self._seed)
            accepted = 0
            errors = 0
            for _ in range(trials):
                e = (rng.random(15) < p).astype(int)
                synd = H.dot(e) % 2
                if not synd.any():            # accept: trivial syndrome
                    accepted += 1
                    if e.any():               # nonzero codeword -> logical error
                        errors += 1
            acc = accepted / trials
            p_out = errors / accepted if accepted else 0.0
            self.io.writeln(f"\n  15-to-1 magic-state distillation:")
            self.io.writeln(f"    input error p      = {p:g}")
            self.io.writeln(f"    acceptance rate    = {acc:.4f}")
            self.io.writeln(f"    output error p_out = {p_out:.3e}   (~35 p^3 = {35 * p ** 3:.3e})")
            self.io.writeln(f"    suppression p/p_out= {p / p_out:.1f}x" if p_out > 0 else
                            "    output error: none observed")
            self.variables['_DISTILL_POUT'] = p_out
        except Exception as e:
            self.io.writeln(f"?DISTILL ERROR: {e}")

    def cmd_lattice(self, rest: str) -> None:
        """LATTICE <stateA> <stateB> — lattice-surgery logical ZZ measurement.

        Encodes two distance-3 repetition logical qubits in the given logical
        states (0, 1, +, -), then performs a merge that measures the joint
        operator Zbar_A Zbar_B (the basis of a lattice-surgery CNOT) via an
        ancilla, and reports the logical parity outcome."""
        parts = rest.split()
        if len(parts) < 2:
            self.io.writeln("?USAGE: LATTICE <stateA> <stateB>   (states: 0, 1, +, -)")
            return
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        try:
            sa, sb = parts[0], parts[1]
            # 3 data qubits per patch (A: 0-2, B: 3-5) + 1 merge ancilla (6).
            qc = QuantumCircuit(7, 1)

            def encode(base, state):
                if state in ('1',):
                    qc.x(base)
                elif state in ('+', '-'):
                    qc.h(base)
                    if state == '-':
                        qc.z(base)
                qc.cx(base, base + 1); qc.cx(base, base + 2)   # repetition encode
            encode(0, sa); encode(3, sb)
            # Merge: measure Zbar_A Zbar_B = Z_{A0} Z_{B0} via an ancilla.
            anc = 6
            qc.h(anc)
            qc.cz(anc, 0)
            qc.cz(anc, 3)
            qc.h(anc)
            qc.measure(anc, 0)
            backend = AerSimulator(noise_model=self._noise_model) if self._noise_model else AerSimulator()
            counts = backend.run(transpile(qc, backend), shots=max(500, self.shots)).result().get_counts()
            par = '+1 (even)' if counts.get('0', 0) >= counts.get('1', 0) else '-1 (odd)'
            self.io.writeln(f"\n  Lattice surgery: merge of |{sa}>_L (A) and |{sb}>_L (B)")
            self.io.writeln(f"    joint Zbar_A Zbar_B measurement -> {par}   {dict(counts)}")
            self.io.writeln(f"    (a deterministic joint parity for computational inputs; the "
                            f"building block of a lattice-surgery CNOT)")
        except Exception as e:
            self.io.writeln(f"?LATTICE ERROR: {e}")

    # ── Commands ────────────────────────────────────────────────────────

    def cmd_qec(self, rest: str) -> None:
        """QEC LIST | QEC <code> [distance] — show a stabilizer code.

        Lists the built-in codes, or prints a code's stabilizer generators and
        logical operators (and verifies they form a valid code)."""
        arg = rest.strip().upper()
        if not arg or arg == 'LIST':
            self.io.writeln("\n  Built-in QEC codes:")
            self.io.writeln("    REP [d]     repetition / bit-flip code, odd distance d (default 3)")
            self.io.writeln("    STEANE      [[7,1,3]] CSS code")
            self.io.writeln("    SHOR        [[9,1,3]] code")
            self.io.writeln("    SURFACE [d] rotated surface code, odd distance d (default 3)")
            self.io.writeln("    BB [l m]    bivariate-bicycle qLDPC (6 6 -> [[72,12,6]], 12 6 -> [[144,12,12]])")
            self.io.writeln("  Decoders: optimal lookup (default), union-find (UF), pymatching (MWPM),")
            self.io.writeln("            BP+OSD (BB codes), stim circuit-level (LOGICAL_ERROR_RATE ... CIRCUIT)")
            return
        parts = arg.split()
        if parts[0] == 'BB':
            try:
                l = int(parts[1]) if len(parts) > 1 else 6
                m = int(parts[2]) if len(parts) > 2 else 6
                self._qec_show_bb(l, m)
            except Exception as e:
                self.io.writeln(f"?QEC ERROR: {e}")
            return
        try:
            d = int(parts[1]) if len(parts) > 1 else 3
            code = self._qec_code(parts[0], d)
        except Exception as e:
            self.io.writeln(f"?QEC ERROR: {e}")
            return
        self.io.writeln(f"\n  {code['name']}: {code['n']} data qubits, distance {code['d']}")
        self.io.writeln(f"  Stabilizers ({len(code['stab'])}):")
        for s in code['stab']:
            self.io.writeln(f"    {s}")
        self.io.writeln(f"  Logical X: {code['lx']}")
        self.io.writeln(f"  Logical Z: {code['lz']}")
        # Validity self-check: logicals commute with every stabilizer and
        # anticommute with each other.
        ok = bool(all(not _anticommute(code['lx'], s) for s in code['stab'])
                  and all(not _anticommute(code['lz'], s) for s in code['stab'])
                  and _anticommute(code['lx'], code['lz']))
        self.io.writeln(f"  Valid code (logicals normalize the stabilizer group): {ok}")

    # Decoder flags accepted by LOGICAL_ERROR_RATE and THRESHOLD.
    _DECODER_FLAGS = {'LOOKUP': 'lookup', 'MATCH': 'match', 'MATCHING': 'match',
                      'UF': 'match', 'UNION-FIND': 'match', 'MWPM': 'pymatching',
                      'BP': 'bp'}

    def _split_decoder_flags(self, parts: list[str]) -> tuple[list[str], str | None, set]:
        """(remaining tokens, requested decoder, other flags) from a command."""
        decoder = None
        other = set()
        rest = []
        for t in parts:
            u = t.upper()
            if u in self._DECODER_FLAGS:
                decoder = self._DECODER_FLAGS[u]
            elif u in ('CIRCUIT',):
                other.add(u)
            else:
                rest.append(t)
        return rest, decoder, other

    def _code_rate(self, code: dict, p: float, trials: int, rng,
                   decoder: str | None) -> tuple[float, str]:
        """(logical error rate, decoder label) for a string code."""
        if decoder == 'bp':
            from qubasic_core.qec2 import _css_from_strings
            return self._qec_bb_rate(_css_from_strings(code), p, trials, rng), 'BP+OSD decoder'
        chosen = self._choose_decoder(code, decoder)
        rate = self._logical_error_rate(code, p, trials, rng, decoder=chosen)
        return rate, self._DECODER_LABEL[chosen]

    def cmd_logical_error_rate(self, rest: str) -> None:
        """LOGICAL_ERROR_RATE <code> [distance] <p> [trials] [LOOKUP|MATCH|MWPM|BP|CIRCUIT] — logical error rate.

        Code-capacity Monte Carlo (i.i.d. physical errors, decode, count
        residual logical errors), vectorized over trials. The decoder defaults
        to the minimum-weight lookup table while the code's syndrome space
        allows it and to exact minimum-weight matching (blossom) beyond;
        MATCH (alias UF) forces matching, MWPM uses pymatching when installed,
        BP the internal BP+OSD. CIRCUIT switches to circuit-level noise: stim
        syndrome-extraction circuits over d rounds decoded from the detector
        error model (the trailing count is then shots, default 100000). BB
        codes decode with BP+OSD."""
        parts = rest.split()
        if len(parts) < 2:
            self.io.writeln("?USAGE: LOGICAL_ERROR_RATE <code> [distance] <p> [trials] "
                            "[LOOKUP|MATCH|MWPM|BP|CIRCUIT]")
            return
        try:
            name = parts[0]
            idx = 1
            distance = 3
            _FLAGS = tuple(self._DECODER_FLAGS) + ('CIRCUIT',)
            if name.upper() == 'BB':
                # BB takes l and m instead of a distance.
                l = m = 6
                if len(parts) > 3 and parts[1].isdigit() and parts[2].isdigit():
                    l, m = int(parts[1]), int(parts[2]); idx = 3
                tail = [t for t in parts[idx:] if t.upper() not in _FLAGS]
                p = float(self._eval_with_vars(tail[0], {}))
                if not 0.0 <= p <= 1.0:
                    raise ValueError(f"p={p} out of range (expected 0..1; "
                                     f"usage: LOGICAL_ERROR_RATE BB [l m] <p> [trials])")
                trials = int(tail[1]) if len(tail) > 1 else 1000
                from qubasic_core.qec2 import _bb_code
                code = _bb_code(l, m)
                rng = np.random.default_rng(self._seed)
                t0 = __import__('time').time()
                ler = self._qec_bb_rate(code, p, trials, rng)
                dt = __import__('time').time() - t0
                self.io.writeln(f"\n  {code['name']} [[{code['n']},{code['k']},"
                                f"{code['d'] or '?'}]]: physical p={p}, {trials} trials "
                                f"(BP+OSD decoder, {dt:.1f}s)")
                self.io.writeln(f"  Logical error rate = {ler:.6f}")
                self.variables['_LER'] = ler
                return
            parts, decoder, other = self._split_decoder_flags(parts)
            if len(parts) > 2 and parts[1].isdigit() and float(parts[1]) >= 1:
                distance = int(parts[1]); idx = 2
            tail = parts[idx:]
            p = float(self._eval_with_vars(tail[0], {}))
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"p={p} out of range (expected 0..1)")
            if 'CIRCUIT' in other:
                shots = int(tail[1]) if len(tail) > 1 else 100000
                rounds = int(tail[2]) if len(tail) > 2 else None
                self._qec_circuit_level(name, distance, p, shots, rounds)
                return
            trials = int(tail[1]) if len(tail) > 1 else 20000
            code = self._qec_code(name, distance)
            rng = np.random.default_rng(self._seed)
            t0 = __import__('time').time()
            ler, dec = self._code_rate(code, p, trials, rng, decoder)
            dt = __import__('time').time() - t0
            self.io.writeln(f"\n  {code['name']}: physical p={p}, {trials} trials ({dec}, {dt:.1f}s)")
            self.io.writeln(f"  Logical error rate = {ler:.6f}")
            self.variables['_LER'] = ler
        except Exception as e:
            self.io.writeln(f"?LOGICAL_ERROR_RATE ERROR: {e}")

    def cmd_threshold(self, rest: str) -> None:
        """THRESHOLD <code> <p1> <p2> <steps> [d1 d2 ...] [TRIALS n] [LOOKUP|MATCH|MWPM|BP] — sweep p.

        Reports the logical error rate across [p1, p2] for several code distances
        (default 3, 5, 7), so the threshold crossing where larger codes start
        helping is visible. Each distance uses the decoder LOGICAL_ERROR_RATE
        would pick for it unless a decoder flag is given."""
        parts = rest.split()
        if len(parts) < 4:
            self.io.writeln("?USAGE: THRESHOLD <code> <p1> <p2> <steps> [distances] "
                            "[TRIALS n] [LOOKUP|MATCH|MWPM|BP]")
            return
        try:
            parts, decoder, _ = self._split_decoder_flags(parts)
            trials = 8000
            if any(t.upper() == 'TRIALS' for t in parts):
                k = [t.upper() for t in parts].index('TRIALS')
                trials = int(parts[k + 1])
                parts = parts[:k] + parts[k + 2:]
            name = parts[0]
            p1 = float(self._eval_with_vars(parts[1], {}))
            p2 = float(self._eval_with_vars(parts[2], {}))
            steps = int(parts[3])
            distances = [int(x) for x in parts[4:]] if len(parts) > 4 else [3, 5, 7]
            ps = [p1 + (p2 - p1) * i / max(1, steps - 1) for i in range(steps)]
            codes: list[dict] = []
            for d in distances:
                c = self._qec_code(name, d)
                if all(c is not other for other in codes):
                    codes.append(c)
            rng = np.random.default_rng(self._seed)
            labels = []
            for c in codes:
                labels.append('BP+OSD' if decoder == 'bp'
                              else self._choose_decoder(c, decoder))
            self.io.writeln(f"\n  Threshold sweep ({name.upper()}), {trials} trials/point:")
            self.io.writeln("    p      " + ''.join(f"  d={c['d']:<8}" for c in codes))
            self.io.writeln("  decoder  " + ''.join(f"  {lab[:8]:<10}" for lab in labels))
            for p in ps:
                row = f"  {p:5.3f}  "
                for c in codes:
                    rate, _ = self._code_rate(c, p, trials, rng, decoder)
                    row += f"  {rate:<8.4f}"
                self.io.writeln(row)
        except Exception as e:
            self.io.writeln(f"?THRESHOLD ERROR: {e}")
