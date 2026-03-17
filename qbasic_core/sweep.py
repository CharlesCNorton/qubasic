"""QBASIC sweep mixin — parameter sweep command."""

from __future__ import annotations

from qiskit import transpile
from qiskit_aer import AerSimulator

try:
    import plotille as _plotille
except ImportError:
    _plotille = None


class SweepMixin:
    """Parameter sweep command for QBasicTerminal.

    Requires: TerminalProtocol — uses self.variables, self.shots,
    self.sim_method, self.sim_device, self._noise_model,
    self.eval_expr(), self.build_circuit().
    """

    def cmd_sweep(self, rest: str) -> None:
        """SWEEP var start end [steps] — run circuit for each parameter value.

        When plotille is available, appends a braille line chart of the
        top-state probability vs. the swept variable.
        """
        parts = rest.split()
        if len(parts) < 3:
            print("?USAGE: SWEEP <var> <start> <end> [steps]")
            return
        var = parts[0]
        start = self.eval_expr(parts[1])
        end = self.eval_expr(parts[2])
        steps = int(parts[3]) if len(parts) > 3 else 10
        if steps < 1:
            print("?SWEEP needs at least 1 step")
            return

        print(f"\nSWEEP {var} from {start} to {end} in {steps} steps:")
        if steps == 1:
            values = [start]
        else:
            values = [start + (end - start) * i / (steps - 1) for i in range(steps)]
        backend_opts = {'method': self.sim_method}
        if self.sim_device == 'GPU':
            backend_opts['device'] = 'GPU'
        if self._noise_model:
            backend_opts['noise_model'] = self._noise_model
        backend = AerSimulator(**backend_opts)
        sweep_xs: list[float] = []
        sweep_ys: list[float] = []
        for val in values:
            self.variables[var] = val
            try:
                qc, has_measure = self.build_circuit()
                if has_measure:
                    qc.measure_all()
                result = backend.run(transpile(qc, backend), shots=self.shots).result()
                counts = dict(result.get_counts())
                ranked = sorted(counts.items(), key=lambda x: -x[1])
                top = ranked[0]
                bar_len = int(30 * top[1] / self.shots)
                top2 = f" |{ranked[1][0]}\u27E9={ranked[1][1]}" if len(ranked) > 1 else ""
                n_unique = len(ranked)
                print(f"  {var}={val:8.4f}  |{top[0]}\u27E9 {top[1]:>5}/{self.shots} "
                      f"{'\u2588' * bar_len}{top2}  ({n_unique} states)")
                sweep_xs.append(val)
                sweep_ys.append(top[1] / self.shots)
            except Exception as e:
                print(f"  {var}={val:8.4f}  ERROR: {e}")

        # Plotille chart of P(top state) vs variable
        if _plotille is not None and len(sweep_xs) >= 2:
            print()
            print(_plotille.plot(
                sweep_xs, sweep_ys,
                width=60, height=15,
                X_label=var,
                Y_label='P(top)',
                lc='cyan'))
        print()
