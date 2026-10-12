"""Detector error models: independent error mechanisms, sampled and decoded.

A model lists error mechanisms, each with a probability, the detectors it
flips and the observables it flips. Sampling draws every mechanism
independently for many shots at once; decoding pairs the fired detectors
(and the boundary) by exact minimum-weight perfect matching on the graph
whose edges are the mechanisms that flip one or two detectors, weighted
log((1 - p) / p), and predicts the observables the matched paths flip
(blossom algorithm via rustworkx). A shot fails when a predicted observable
differs from the actual one. Each distinct detector pattern is decoded once.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np


class DetectorModel:
    """Error mechanisms over `n_det` detectors and `n_obs` observables."""

    def __init__(self, n_det: int, n_obs: int = 1) -> None:
        self.n_det = n_det
        self.n_obs = n_obs
        self.probs: list[float] = []
        self.dets: list[tuple[int, ...]] = []
        self.obs: list[int] = []            # bitmask over observables
        self._graph: Any = None

    def add(self, p: float, dets, obs_mask: int = 0) -> None:
        """A mechanism of probability p flipping `dets` and the observables in obs_mask."""
        dets = tuple(sorted(set(int(d) for d in dets)))
        if len(dets) > 2:
            raise ValueError("a matchable mechanism flips at most two detectors")
        if p <= 0 or (not dets and not obs_mask):
            return
        self.probs.append(float(p))
        self.dets.append(dets)
        self.obs.append(int(obs_mask))
        self._graph = None

    # ── sampling ──────────────────────────────────────────────────────

    def sample(self, shots: int, rng: Any) -> tuple[np.ndarray, np.ndarray]:
        """(detector flips [shots x n_det], observable flips [shots x n_obs])."""
        m = len(self.probs)
        fired = rng.random((shots, m)) < np.array(self.probs)[None, :]
        D = np.zeros((m, self.n_det), dtype=np.uint8)
        L = np.zeros((m, self.n_obs), dtype=np.uint8)
        for k, (ds, o) in enumerate(zip(self.dets, self.obs)):
            D[k, list(ds)] = 1
            for j in range(self.n_obs):
                L[k, j] = (o >> j) & 1
        f = fired.astype(np.int32)
        return ((f @ D) & 1).astype(np.uint8), ((f @ L) & 1).astype(np.uint8)

    # ── matching ──────────────────────────────────────────────────────

    def _build_graph(self) -> dict:
        """Shortest paths between detectors (node n_det is the boundary),
        with the observable parity of each path."""
        if self._graph is not None:
            return self._graph
        B = self.n_det
        adj: list[dict[int, tuple[float, int]]] = [dict() for _ in range(B + 1)]
        for p, ds, o in zip(self.probs, self.dets, self.obs):
            if not ds:
                continue
            u, v = (ds[0], B) if len(ds) == 1 else ds
            w = math.log((1 - p) / p) if p < 0.5 else 1e-9
            if v not in adj[u] or w < adj[u][v][0]:
                adj[u][v] = (w, o)
                adj[v][u] = (w, o)
        dist = np.full((B + 1, B + 1), np.inf)
        par = np.zeros((B + 1, B + 1), dtype=np.int64)
        for src in range(B + 1):
            d = {src: 0.0}
            o_par = {src: 0}
            heap = [(0.0, src)]
            while heap:
                du, u = heapq.heappop(heap)
                if du > d.get(u, np.inf):
                    continue
                if u == B and src != B:
                    continue            # the boundary absorbs paths, it relays nothing
                for v, (w, o) in adj[u].items():
                    nd = du + w
                    if nd < d.get(v, np.inf) - 1e-12:
                        d[v] = nd
                        o_par[v] = o_par[u] ^ o
                        heapq.heappush(heap, (nd, v))
            for v, dv in d.items():
                dist[src, v] = dv
                par[src, v] = o_par[v]
        self._graph = {'dist': dist, 'par': par, 'B': B}
        return self._graph

    def decode(self, fired: np.ndarray) -> int:
        """Observable mask predicted for one detector pattern."""
        defects = [int(i) for i in np.flatnonzero(fired)]
        if not defects:
            return 0
        import rustworkx as rx
        g = self._build_graph()
        dist, par, B = g['dist'], g['par'], g['B']
        k = len(defects)
        finite = dist[np.isfinite(dist)]
        big = float(finite.max() if finite.size else 1.0) * 2 + 10.0

        def w(x: float) -> int:          # the blossom matcher takes integer weights
            return int(round(x * 1000))
        graph = rx.PyGraph()
        graph.add_nodes_from(range(2 * k))
        for a in range(k):
            for c in range(a + 1, k):
                dac = dist[defects[a], defects[c]]
                if np.isfinite(dac):
                    graph.add_edge(a, c, w(big - dac))
                graph.add_edge(k + a, k + c, w(big))
            dab = dist[defects[a], B]
            if np.isfinite(dab):
                graph.add_edge(a, k + a, w(big - dab))
        matching = rx.max_weight_matching(graph, max_cardinality=True, weight_fn=lambda w: w)
        mask = 0
        for u, v in matching:
            u, v = min(u, v), max(u, v)
            if u >= k:
                continue
            mask ^= int(par[defects[u], B if v >= k else defects[v]])
        return mask

    def logical_failure_rate(self, shots: int, rng: Any) -> tuple[float, np.ndarray]:
        """(rate of shots where any observable is mispredicted, per-observable
        failure rates)."""
        det, obs = self.sample(shots, rng)
        keys, inverse = np.unique(np.packbits(det, axis=1, bitorder='little'),
                                  axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        pred = np.zeros((len(keys), self.n_obs), dtype=np.uint8)
        for i, packed in enumerate(keys):
            bits = np.unpackbits(packed, bitorder='little')[:self.n_det]
            mask = self.decode(bits)
            for j in range(self.n_obs):
                pred[i, j] = (mask >> j) & 1
        wrong = pred[inverse] ^ obs
        return float(np.mean(wrong.any(axis=1))), wrong.mean(axis=0)
