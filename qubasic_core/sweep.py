"""QUBASIC sweep mixin — parameter sweep command."""

from __future__ import annotations

from qiskit import transpile
from qiskit_aer import AerSimulator
from qubasic_core.host import HostState

try:
    import plotille as _plotille
except ImportError:
    _plotille = None


class SweepMixin(HostState):
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
            self.io.writeln("?USAGE: SWEEP <var> <start> <end> [steps]")
            return
        var = parts[0]
        start = self.eval_expr(parts[1])
        end = self.eval_expr(parts[2])
        steps = int(parts[3]) if len(parts) > 3 else 10
        if steps < 1:
            self.io.writeln("?SWEEP needs at least 1 step")
            return

        self.io.writeln(f"\nSWEEP {var} from {start} to {end} in {steps} steps:")
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
        # Parametric fast path: compile once with the swept variable as a
        # bound Parameter, then re-bind per point (no rebuild, no retranspile).
        pctx = None
        try:
            pctx = self._alg_parametric_setup([var], sampling=True)
        except Exception:
            pctx = None
        if pctx is not None:
            self.io.writeln("  (parametric compile: transpiled once, bound per point)")
        # Transient progress bar in an interactive terminal only; piped and
        # captured output is unchanged.
        _prog = _ptask = None
        try:
            import sys as _sys
            if _sys.stdout.isatty() and len(values) > 2:
                from rich.progress import Progress
                _prog = Progress(transient=True)
                _prog.start()
                _ptask = _prog.add_task(f"SWEEP {var}", total=len(values))
        except Exception:
            _prog = None
        # The sweep temporarily overwrites the variable; restore it afterward
        # so a later RUN/STATS still sees the program's own value.
        had_prior = var in self.variables
        prior = self.variables.get(var)
        try:
            for val in values:
                self.variables[var] = val
                try:
                    if pctx is not None:
                        bound = pctx['qc'].assign_parameters(
                            {next(iter(pctx['pmap'].values())): float(val)})
                        _kw = {'shots': self.shots}
                        if self._seed is not None:
                            _kw['seed_simulator'] = self._seed
                        result = pctx['backend'].run(bound, **_kw).result()
                    else:
                        qc, has_measure = self.build_circuit()
                        if has_measure:
                            qc.measure_all()
                        result = backend.run(transpile(qc, backend, optimization_level=self._transpile_opt_level), shots=self.shots).result()
                    counts = dict(result.get_counts())
                    ranked = sorted(counts.items(), key=lambda x: -x[1])
                    top = ranked[0]
                    bar_len = int(30 * top[1] / self.shots)
                    top2 = f" |{self._bits(ranked[1][0])}\u27E9={ranked[1][1]}" if len(ranked) > 1 else ""
                    n_unique = len(ranked)
                    bar = '\u2588' * bar_len
                    self.io.writeln(f"  {var}={val:8.4f}  |{self._bits(top[0])}\u27E9 {top[1]:>5}/{self.shots} "
                                   f"{bar}{top2}  ({n_unique} states)")
                    sweep_xs.append(val)
                    sweep_ys.append(top[1] / self.shots)
                except Exception as e:
                    self.io.writeln(f"  {var}={val:8.4f}  ERROR: {e}")
                if _prog is not None and _ptask is not None:
                    _prog.advance(_ptask)
        finally:
            if _prog is not None:
                _prog.stop()
            if had_prior:
                self.variables[var] = prior
            else:
                self.variables.pop(var, None)

        # Plotille chart of P(top state) vs variable
        if _plotille is not None and len(sweep_xs) >= 2:
            self.io.writeln('')
            self.io.writeln(_plotille.plot(
                sweep_xs, sweep_ys,
                width=60, height=15,
                X_label=var,
                Y_label='P(top)',
                y_min=0.0, y_max=1.0,
                lc='cyan'))
        self.io.writeln('')
