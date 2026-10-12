"""QUBASIC state display mixin — STATE, BLOCH, HIST, PROBS, PRINT STATE, DECOMPOSE."""

import re
from typing import Any

import numpy as np

from qubasic_core.engine import MAX_BLOCH_DISPLAY
from qubasic_core.host import HostState


class StateDisplayMixin(HostState):
    """State inspection commands for QBasicTerminal.

    All state commands use _active_sv / _active_nqubits to read from
    whichever engine (Qiskit or LOCC) last executed, ensuring displays
    always reflect the actual executed state.
    """

    def cmd_state(self, rest: str = '') -> None:
        """STATE [reg] — the statevector of the current state (one register in LOCC SPLIT mode)."""
        if self.locc_mode:
            return self._locc_state(rest)
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        self._print_statevector(sv, self._active_nqubits)

    def cmd_hist(self) -> None:
        """HIST — the last run's measurement histogram."""
        if self.last_counts is None:
            self.io.writeln("?NO RESULTS — RUN first")
            return
        self.print_histogram(self.last_counts)

    def cmd_probs(self) -> None:
        """PROBS — the probability distribution of the current state."""
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        self._print_probs(sv, self._active_nqubits)

    def cmd_bloch(self, rest: str) -> None:
        """BLOCH [qubit] — ASCII Bloch sphere of one qubit, or of each qubit (up to 8); BLOCH <reg> [qubit] in LOCC mode."""
        if self.locc_mode:
            return self._locc_bloch(rest)
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        n = self._active_nqubits
        if rest:
            q = int(rest)
            self._print_bloch_single(sv, q, n)
        else:
            for q in range(min(n, MAX_BLOCH_DISPLAY)):
                self._print_bloch_single(sv, q, n)
                if q < min(n, MAX_BLOCH_DISPLAY) - 1:
                    self.io.writeln('')

    def cmd_circuit(self) -> None:
        """CIRCUIT — draw the last run's circuit as text."""
        if self.last_circuit is None:
            self.io.writeln("?NO CIRCUIT — RUN first")
            return
        try:
            self.io.writeln(self.last_circuit.draw(output='text'))
        except Exception:
            self.io.writeln(f"Circuit: {self.last_circuit.num_qubits} qubits, "
                           f"depth {self.last_circuit.depth()}, "
                           f"{self.last_circuit.size()} gates")

    _RE_PRINT_QUBIT = re.compile(r'QUBIT\s*\((.+)\)\s*$', re.IGNORECASE)

    _RE_PRINT_ENT = re.compile(r'ENTANGLEMENT\s*\((.+)\)\s*$', re.IGNORECASE)

    def _current_state(self) -> tuple[Any, int, list[int] | None]:
        """(statevector, qubit count, LOCC register sizes) at the current point
        of execution: the live per-shot state, the LOCC joint state, the state
        of the circuit built so far during RUN, or the last run's state."""
        live = getattr(self, '_live', None)
        if live is not None:
            return live.sv, live.n, None
        if self.locc_mode and self.locc is not None and self.locc.joint:
            return self.locc.sv, self.locc.n_total, list(self.locc.sizes)
        qc = getattr(self, '_build_qc', None)
        if qc is not None:
            return (self._extract_state_now(qc.copy(), self.sim_precision, self._seed),
                    qc.num_qubits, None)
        return self.last_sv, self.num_qubits, None

    def _try_quantum_print(self, text: str, run_vars: dict) -> str | None:
        """Quantum PRINT forms: STATE (Dirac notation), QUBIT(q) (P(1) and the
        Bloch vector), ENTANGLEMENT(a, b) (entanglement of formation of the
        two-qubit reduced state), and @REG (one LOCC register). Each reads the
        state at the point the PRINT executes. Returns None for other text."""
        item = text.strip()
        upper = item.upper()
        if upper.startswith('@') and self.locc_mode and self.locc is not None:
            reg = upper[1:].strip()
            if reg in self.locc.names:
                if self.locc.joint:
                    return self._format_dirac(self.locc.sv, self.locc.n_total,
                                              list(self.locc.sizes))
                return self._format_dirac(self.locc.svs[reg], self.locc.get_size(reg))
            return None
        m_q = self._RE_PRINT_QUBIT.match(item)
        m_e = self._RE_PRINT_ENT.match(item)
        if upper != 'STATE' and not m_q and not m_e:
            return None
        if self.locc_mode and self.locc is not None and not self.locc.joint:
            return "<SPLIT registers have no joint state: PRINT @A, @B, ...>"
        sv, n, segments = self._current_state()
        if sv is None:
            return "<no state>"
        if upper == 'STATE':
            return self._format_dirac(sv, n, segments)
        from qubasic_core import statetools as st
        if m_q:
            q = int(self._eval_with_vars(m_q.group(1), run_vars))
            if q < 0 or q >= n:
                raise ValueError(f"QUBIT {q} OUT OF RANGE (0-{n - 1})")
            x, y, z = st.bloch(sv, q, n)
            return f"q{q}: P(1)={(1 - z) / 2:.4f} Bloch=({x:.3f},{y:.3f},{z:.3f})"
        args = self._split_arg_list(m_e.group(1))
        if len(args) != 2:
            raise ValueError("ENTANGLEMENT takes two qubits: ENTANGLEMENT(a, b)")
        qa, qb = (int(self._eval_with_vars(a, run_vars)) for a in args)
        if qa == qb or not (0 <= qa < n and 0 <= qb < n):
            raise ValueError(f"ENTANGLEMENT needs two different qubits in 0-{n - 1}")
        c = st.concurrence(st.subsystem_rdm(sv, [qa, qb], n))
        return f"E(q{qa},q{qb}) = {st.eof_from_concurrence(c):.6f} ebits (concurrence {c:.6f})"

    def _format_dirac(self, sv, n_qubits: int, segments: list[int] | None = None) -> str:
        """Dirac notation of the first 16 non-zero amplitudes, in the active
        OPTION ENDIAN order; `segments` splits a LOCC joint state into its
        registers (A|B|...)."""
        from qubasic_core.engine import AMPLITUDE_THRESHOLD
        from qubasic_core.statetools import nonzero_indices
        flat = np.asarray(sv).reshape(-1)
        idx, total = nonzero_indices(flat, AMPLITUDE_THRESHOLD, 16)
        parts = []
        for i in idx:
            amp = complex(flat[int(i)])
            bits = format(int(i), f'0{n_qubits}b')
            if segments:
                segs, pos = [], len(bits)
                for size in segments:
                    segs.append(bits[pos - size:pos])
                    pos -= size
                bits = '|'.join(segs)
            state = self._bits(bits)
            if abs(amp.imag) < AMPLITUDE_THRESHOLD:
                if abs(amp.real - 1.0) < 1e-6:
                    parts.append(f"|{state}\u27E9")
                elif abs(amp.real + 1.0) < 1e-6:
                    parts.append(f"-|{state}\u27E9")
                else:
                    parts.append(f"{amp.real:+.4f}|{state}\u27E9")
            else:
                parts.append(f"({amp.real:.3f}{amp.imag:+.3f}j)|{state}\u27E9")
        if total > len(parts):
            parts.append("...")
        return " ".join(parts) if parts else "|0\u27E9"

    def cmd_decompose(self) -> None:
        """DECOMPOSE — gate counts of the last run's circuit."""
        if self.last_circuit is None:
            self.io.writeln("?NO CIRCUIT — RUN first")
            return
        ops: dict[str, int] = {}
        for inst in self.last_circuit.data:
            name = inst.operation.name
            ops[name] = ops.get(name, 0) + 1
        self.io.writeln(f"\n  Circuit: {self.last_circuit.num_qubits} qubits, "
                       f"depth {self.last_circuit.depth()}, {self.last_circuit.size()} gates")
        for name, count in sorted(ops.items(), key=lambda x: -x[1]):
            bar = '█' * min(count, 40)
            self.io.writeln(f"    {name:>10}  {count:>4}  {bar}")
        self.io.writeln('')
