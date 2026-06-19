"""QUBASIC compilation and resource estimation.

Fault-tolerant resource estimation under the surface code, calibrated offline
device models (per-qubit T1/T2 plus a topology), and a transpiler pass that
reports circuit optimization. All offline.

  RESOURCES <target_pL> <phys_p>   Estimate surface-code distance and qubit/time cost
  DEVICE linear|ring|heavyhex [n] [T1us] [T2us]   Load a calibrated device noise model
  OPTIMIZE [level]                 Transpile the program and report the reduction
"""

from __future__ import annotations

import math
import re

import numpy as np


class ResourceMixin:
    """Resource estimation, calibrated device models, and circuit optimization.

    Requires: TerminalProtocol — uses self.num_qubits, self.program,
    self.build_circuit(), self._noise_model, self._coupling_map, self.io.
    """

    # Standard surface-code parameters (circuit-level): threshold and the
    # sub-threshold scaling p_L ~ A (p/p_th)^((d+1)/2).
    _SC_THRESHOLD = 0.01
    _SC_A = 0.03

    def cmd_resources(self, rest: str) -> None:
        """RESOURCES <target_logical_error> <physical_error> — fault-tolerant estimate.

        Finds the smallest surface-code distance d reaching the target logical
        error rate at the given physical error rate, then estimates physical
        qubits (2 d^2 - 1 per logical patch), the program's T-count, T-factory
        overhead, and an approximate code-cycle / runtime figure."""
        parts = rest.split()
        if len(parts) < 2:
            self.io.writeln("?USAGE: RESOURCES <target_logical_error> <physical_error>")
            return
        try:
            target = float(self._eval_with_vars(parts[0], {}))
            p = float(self._eval_with_vars(parts[1], {}))
            if p >= self._SC_THRESHOLD:
                self.io.writeln(f"?physical error {p} is above the surface-code threshold "
                                f"~{self._SC_THRESHOLD}; error correction cannot help")
                return
            # Smallest odd d with A (p/p_th)^((d+1)/2) <= target.
            d = 1
            while self._SC_A * (p / self._SC_THRESHOLD) ** ((d + 1) / 2) > target and d < 99:
                d += 2
            pL = self._SC_A * (p / self._SC_THRESHOLD) ** ((d + 1) / 2)
            n_logical = self.num_qubits
            phys_per_logical = 2 * d * d - 1
            data_qubits = n_logical * phys_per_logical
            # T-count from the program circuit, if any.
            t_count = 0
            if self.program:
                try:
                    qc, _ = self.build_circuit()
                    ops = qc.count_ops()
                    t_count = ops.get('t', 0) + ops.get('tdg', 0)
                except Exception:
                    t_count = 0
            # 15-to-1 T factory footprint and a parallelism assumption.
            factory_qubits = 15 * phys_per_logical
            n_factories = max(1, min(t_count, 8)) if t_count else 0
            total_qubits = data_qubits + n_factories * factory_qubits
            cycle_us = 1.0   # ~1 microsecond per surface-code cycle (typical SC qubit)
            depth = qc.depth() if self.program else 0
            runtime_us = depth * d * cycle_us + t_count * d * cycle_us
            self.io.writeln(f"\n  Fault-tolerant resource estimate (surface code):")
            self.io.writeln(f"    target logical error : {target:g}")
            self.io.writeln(f"    physical error       : {p:g}  (threshold ~{self._SC_THRESHOLD})")
            self.io.writeln(f"    code distance        : d = {d}  (achieves p_L ~ {pL:.2e})")
            self.io.writeln(f"    logical qubits       : {n_logical}")
            self.io.writeln(f"    physical / logical   : {phys_per_logical}")
            self.io.writeln(f"    data qubits          : {data_qubits:,}")
            if t_count:
                self.io.writeln(f"    T-count              : {t_count}")
                self.io.writeln(f"    T-factories (15-to-1): {n_factories} x {factory_qubits:,} qubits")
            self.io.writeln(f"    total physical qubits: ~{total_qubits:,}")
            self.io.writeln(f"    est. runtime         : ~{runtime_us:.0f} us "
                            f"({depth} logical depth, ~{cycle_us} us/cycle)")
            self.variables['_FT_DISTANCE'] = d
            self.variables['_FT_QUBITS'] = total_qubits
        except Exception as e:
            self.io.writeln(f"?RESOURCES ERROR: {e}")

    def cmd_device(self, rest: str) -> None:
        """DEVICE <topology> [n] [T1us] [T2us] — load a calibrated offline device model.

        Builds a NoiseModel with per-qubit thermal relaxation (T1/T2 and gate
        times) and a coupling map for the chosen topology (linear, ring, heavyhex,
        all), so RUN simulates a realistic hardware device with no network. DEVICE
        OFF clears it."""
        parts = rest.split()
        if not parts:
            cm = 'set' if self._coupling_map else 'none'
            self.io.writeln(f"DEVICE: noise={'on' if self._noise_model else 'off'}, coupling={cm}")
            return
        if parts[0].upper() in ('OFF', 'NONE'):
            self._noise_model = None
            self._coupling_map = None
            self._circuit_cache_key = None
            self.io.writeln("DEVICE OFF")
            return
        topo = parts[0].lower()
        n = int(parts[1]) if len(parts) > 1 else self.num_qubits
        t1 = float(parts[2]) * 1e-6 if len(parts) > 2 else 100e-6
        t2 = float(parts[3]) * 1e-6 if len(parts) > 3 else 80e-6
        try:
            from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
            nm = NoiseModel()
            t_1q, t_2q = 50e-9, 300e-9   # typical gate durations
            _1q = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'sx', 'rx', 'ry', 'rz', 'p', 'u', 'id']
            _2q = ['cx', 'cy', 'cz', 'ch', 'swap', 'dcx', 'iswap', 'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz']
            for q in range(n):
                nm.add_quantum_error(thermal_relaxation_error(t1, t2, t_1q), _1q, [q])
            err2 = thermal_relaxation_error(t1, t2, t_2q).expand(
                thermal_relaxation_error(t1, t2, t_2q))
            for a in range(n):
                for b in range(n):
                    if a != b:
                        nm.add_quantum_error(err2, _2q, [a, b])
            if topo == 'linear':
                cm = [[i, i + 1] for i in range(n - 1)]
            elif topo == 'ring':
                cm = [[i, (i + 1) % n] for i in range(n)]
            elif topo in ('heavyhex', 'heavy-hex'):
                cm = [[i, i + 1] for i in range(n - 1)] + [[i, i + 2] for i in range(0, n - 2, 4)]
            else:
                cm = [[i, j] for i in range(n) for j in range(n) if i != j]
            edges = []
            for a, b in cm:
                edges += [[a, b], [b, a]]
            self._noise_model = nm
            self._coupling_map = edges if topo != 'all' else None
            self._noise_spec = f"device {topo} T1={t1 * 1e6}us T2={t2 * 1e6}us"
            self._circuit_cache_key = None
            self.io.writeln(f"DEVICE {topo}: {n} qubits, T1={t1 * 1e6:.0f}us T2={t2 * 1e6:.0f}us, "
                            f"thermal noise + coupling map loaded")
        except Exception as e:
            self.io.writeln(f"?DEVICE ERROR: {e}")

    def cmd_optimize(self, rest: str = '') -> None:
        """OPTIMIZE [level] — transpile the program and report the reduction.

        Runs the Qiskit transpiler at the given optimization level (default 3)
        against the active basis/coupling model and reports the depth and gate
        count before and after."""
        if not self.program:
            self.io.writeln("?NOTHING TO OPTIMIZE — enter a program first")
            return
        level = int(rest.strip()) if rest.strip() else 3
        try:
            from qiskit import transpile
            from qiskit_aer import AerSimulator
            qc, _ = self.build_circuit()
            backend = AerSimulator()
            kw = dict(self._transpile_kwargs())
            opt = transpile(qc, backend, optimization_level=level, **kw)
            self._last_transpiled = opt
            self.io.writeln(f"\n  OPTIMIZE (level {level}):")
            self.io.writeln(f"    before: depth {qc.depth()}, gates {qc.size()}")
            self.io.writeln(f"    after : depth {opt.depth()}, gates {opt.size()}")
            dd = qc.depth() - opt.depth()
            self.io.writeln(f"    reduced depth by {dd} ({100 * dd / max(1, qc.depth()):.0f}%)")
            self.variables['_OPT_DEPTH'] = opt.depth()
        except Exception as e:
            self.io.writeln(f"?OPTIMIZE ERROR: {e}")
