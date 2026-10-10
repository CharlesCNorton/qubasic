"""QUBASIC state display mixin — STATE, BLOCH, HIST, PROBS commands."""

from qubasic_core.engine import MAX_BLOCH_DISPLAY


class StateDisplayMixin:
    """State inspection commands for QBasicTerminal.

    All state commands use _active_sv / _active_nqubits to read from
    whichever engine (Qiskit or LOCC) last executed, ensuring displays
    always reflect the actual executed state.
    """

    def cmd_state(self, rest: str = '') -> None:
        if self.locc_mode:
            return self._locc_state(rest)
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        self._print_statevector(sv, self._active_nqubits)

    def cmd_hist(self) -> None:
        if self.last_counts is None:
            self.io.writeln("?NO RESULTS — RUN first")
            return
        self.print_histogram(self.last_counts)

    def cmd_probs(self) -> None:
        sv = self._active_sv
        if sv is None:
            self.io.writeln("?NO STATE — RUN first")
            return
        self._print_probs(sv, self._active_nqubits)

    def cmd_bloch(self, rest: str) -> None:
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
        if self.last_circuit is None:
            self.io.writeln("?NO CIRCUIT — RUN first")
            return
        try:
            self.io.writeln(self.last_circuit.draw(output='text'))
        except Exception:
            self.io.writeln(f"Circuit: {self.last_circuit.num_qubits} qubits, "
                           f"depth {self.last_circuit.depth()}, "
                           f"{self.last_circuit.size()} gates")
