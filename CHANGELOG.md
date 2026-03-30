# Changelog

## 0.4.0 (2026-03-29)

- **Noise correctness**: transpile with optimization_level=0 when noisy so gates survive for noise attachment
- **Noisy statevector**: STATE/BLOCH/DENSITY now reflect the noisy executed state, not the ideal state
- **LOCC noise**: Monte Carlo depolarizing noise in the numpy LOCC engine with per-shot execution
- **GPU**: _make_backend centralizes device flag to all execution paths; graceful probe and fallback
- **cmd_run decomposition**: extracted _run_no_measure, _run_with_fallback, _extract_statevector, _finalize_run, _select_method, _build_backend_opts, _run_kwargs
- **State consistency**: _active_sv/_active_nqubits unify LOCC and standard paths for all state commands
- **SPLIT mode**: EXPECT/DENSITY correctly report that per-register commands are needed
- **Non-depolarizing noise warning**: entering LOCC mode with unsupported noise types warns explicitly
- SEED command for deterministic reproducible results
- VERSION command with build ID, simulator versions, and feature flags
- PROBE command: one-shot exercise of CPU, noise, LOCC, conditional, and combined paths
- CONSISTENCY command: cross-check SV norm, purity, Bloch vectors, EXPECT, and histogram
- METHOD capability map: real probing of each method and GPU availability
- HELP STATUS: tags all 93 commands as native/experimental/partial
- CATALOG shows backend behind each SYS routine
- RUN prints method, device, noise params in summary line
- Demo self-verification: Bell, GHZ, Grover, Deutsch, BV, Superdense auto-check with pass/fail thresholds
- Teleportation fidelity output with X-basis verification
- LOCCINFO: entanglement creation, correction log, branch statistics, noise status
- Method-device pre-check blocks incompatible combinations before execution
- Runtime errors identify failing subsystem (GPU/noise/stabilizer/MPS)
- Run manifest captures all execution parameters for replay
- Correction log in LOCC engine tracks SEND outcomes
- NOISE INFO prints exact channels, operations, qubits
- 25 new tests covering noise, LOCC noise, SEED, VERSION, PROBE, CONSISTENCY, demos, state-after-LOCC, manifest, and method-device pre-check (196 total, up from 171)
- real_sim pytest marker for tests requiring actual Qiskit Aer simulation

## 0.3.1 (2026-03-29)

- Fix f-string backslash escapes that broke import on Python 3.10/3.11

## 0.3.0 (2026-03-28)

- FUNCTION return value fix, APPLY_CIRCUIT in programs, stabilizer fallback
- Bump to 0.3.0

## 0.2.0 (2026-03-28)

- Rename qbasic -> qubasic everywhere (PyPI name conflict)

## 0.1.0 (2026-03-28)

Initial PyPI release.

- BASIC REPL with line-numbered program editing
- 30+ quantum gates (H, X, Y, Z, CX, CCX, RX, RY, RZ, CP, SWAP, etc.)
- LOCC mode: 2-26 party distributed quantum simulation (SPLIT and JOINT)
- Full BASIC language: FOR/NEXT, WHILE/WEND, DO/LOOP, SELECT CASE, SUB/FUNCTION, IF/THEN/ELSE
- AST-based expression evaluator (no eval)
- Noise models: depolarizing, amplitude damping, phase flip, thermal, readout, combined, Pauli, reset
- Debugging: STEP, TRON/TROFF, breakpoints, watch, time-travel (REWIND/FORWARD), PROFILE
- Memory map: PEEK/POKE/SYS/DUMP/MAP/MONITOR
- Analysis: EXPECT, ENTROPY, DENSITY, SWEEP, BENCH, RAM
- File I/O: SAVE/LOAD/INCLUDE/IMPORT, OPEN/CLOSE/PRINT#/INPUT#, CSV, OpenQASM 3.0 export
- 12 built-in demos: Bell, GHZ, Grover, QFT, Deutsch-Jozsa, Bernstein-Vazirani, Superdense, Teleport, LOCC
- JSON output mode for agent/pipeline integration
- String variable resolution in PRINT (LEFT$, RIGHT$, CHR$, concatenation)
- DEF FN and parameterized DEF subroutine invocation
- Trusted publisher CI/CD to PyPI
