# Changelog

## 0.3.0 — 2026-03-26

### Bug fixes
- Fix `_cf_loop` NameError: post-test `LOOP WHILE`/`LOOP UNTIL` was crashing.
- Fix `_cf_shared` operator precedence: variables were not properly shared across scopes.
- Fix `_cf_case` unreachable dead code in SELECT CASE scanning.
- Fix `_cf_wend` return type annotation to match handler contract.
- Fix IF/THEN not propagating jump results from EXIT/GOTO in THEN clause.
- Fix `_recurse` in `_exec_line` discarding return values from nested execution.

### Architecture
- Add structured error hierarchy (`qbasic_core/errors.py`).
- Add I/O protocol for engine decoupling (`qbasic_core/io_protocol.py`).
- Add typed statement AST (`qbasic_core/statements.py`).
- Add statement parser (`qbasic_core/parser.py`) with lazy parse cache.
- Add execution context object (`qbasic_core/exec_context.py`).
- Add unified variable scope model (`qbasic_core/scope.py`).
- Add quantum backend abstraction (`qbasic_core/backend.py`).
- Extract `_split_colon_stmts` to parser module.
- Wire `QBasicError` into `dispatch()` for structured error handling.
- Add `_get_parsed()` for lazy parse cache on direct `.program` assignment.
- Parse cache maintained through `process()`, `cmd_new()`, `cmd_delete()`, `cmd_renum()`, `cmd_undo()`.

### New features
- `SPC(n)` and `TAB(n)` in PRINT statements.
- PRINT separator behavior: semicolon suppresses newline, comma advances to next 14-column tab zone.
- INPUT with retry on bad input (3 attempts, `?REDO FROM START` on failure).
- `GET var` — single keypress input without enter.
- `REDIM name(size)` — resize arrays.
- `ERASE name` — delete a specific array.
- Random access file mode (`OPEN file FOR RANDOM AS #n`).

### Packaging
- Remove `requirements.txt` (duplicated `pyproject.toml`).
- Replace `os.system('cls')` with ANSI escape sequences.
- CI triggers on both `master` and `main` branches; runs both test suites.
- Skip undo stack during script/file loading for efficiency.

### Tests
- Add 141 new tests covering: DO/LOOP, PEEK/POKE, SYS/USR/WAIT/DUMP/MAP/CATALOG, string functions, screen commands, SUB/FUNCTION, LOCAL/STATIC/SHARED, ON ERROR/RESUME, breakpoints, watches, TRON/TROFF, STOP/CONT, profiler, STATS, program management, CHAIN/MERGE, file handles, DATA/READ/RESTORE, ON GOTO/GOSUB, SELECT CASE, SWAP, DEF FN, OPTION BASE, ASSERT, PRINT USING, DIM multi-dim, string variables, hex/binary literals, bitwise operators, EXIT statements, SYS INSTALL, UNITARY multi-qubit, CTRL/INV, MEASURE_X/Y, SYNDROME, LOCC handlers, parser/error modules.
- Total: 417 tests (up from 276).

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
