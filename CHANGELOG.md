# Changelog

## 0.2.0 — 2026-03-17

- Add `.gitignore`, CI workflow, `py.typed` marker, `__version__` attribute.
- Add `--help` CLI flag.
- Add type annotations to core engine, expression evaluator, and control flow.
- Add `TerminalProtocol` to formalize mixin contracts.
- Warn at runtime when `MEAS` is used in Qiskit circuit mode (variable always 0).
- Reject absolute paths in `SAVE` and `LOAD` commands.

## 0.1.0 — 2025

- Initial release: Quantum BASIC interactive terminal.
- LOCC dual-register distributed quantum simulation.
- N-party LOCC (up to 26 registers).
- AST-based safe expression evaluator.
- 30+ quantum gates, noise models, parameter sweeps.
- Built-in demos: Bell, GHZ, teleportation, Grover, QFT, Deutsch-Jozsa, and more.
