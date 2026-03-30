# Changelog

## 0.5.0 (2026-03-30)

- **Colorized histograms**: bars colored green/yellow/dim by probability in terminal
- **Animated STEP mode**: press A during STEP for auto-advance with configurable delay
- **"Did you mean?"**: typo suggestions for misspelled commands (BLASCH -> BLOCH)
- **Status bar prompt**: REPL prompt shows qubit count, LOCC, noise status
- **DRAW command**: braille-character Bloch sphere rendering
- **Color-coded LIST**: gates in cyan, flow in yellow, comments dim (THEME controls)
- **Gate throughput**: RUN summary shows gates/s and circuit complexity (T-count, CNOT count)
- **COMPARE command**: run circuit with two methods and diff output distributions
- **HEATMAP command**: qubit-qubit entanglement entropy grid
- **Startup tip of the day**: random tip shown at REPL launch
- **ANIMATE command**: animated parameter sweep with in-place terminal updates
- **QUIZ mode**: interactive quantum computing quiz with multiple choice
- **Tab completion for file paths**: SAVE/LOAD/INCLUDE complete .qb filenames
- **DIFF command**: diff current program against another BANK slot
- **PLOT command**: ASCII scatter plot of P(top state) vs swept variable
- **UNDO preview**: shows what lines will change before applying
- **THEME command**: switch color schemes (default, retro, none)
- **F1/F2/F3 demo shortcuts**: load Bell, GHZ, Grover demos via function keys
- **Quantum spinner**: |0>, |+>, |1>, |-> cycling during STATS and LOCC progress
- **EXPLAIN command**: describe each program line in plain English
- **LOCC progress spinner**: quantum-themed progress during SEND-based LOCC runs
- **CLIP command**: copy last results to system clipboard
- **Sound on completion**: terminal bell for runs exceeding 2 seconds
- New QoLMixin with 24 quality-of-life features

## 0.4.1 (2026-03-30)

- **Fix SEED dispatch**: moved from no-arg to with-arg dispatch table so `SEED 42` works from the REPL
- **Fix GPU probe**: `cmd_method` GPU probe used undefined `_pqc` variable; now uses `_pqc_m`
- **Fix LOCC non-numeric args**: `LOCC 4 banana` no longer crashes with unguarded ValueError
- **Coverage threshold**: raised CI coverage floor from 60% to 75%
- **Property-based tests**: 4 new hypothesis tests (arithmetic identity, parser fuzzing, process fuzzing, FOR loop count)
- **CLI integration tests**: 8 new tests covering dispatch, SEED, LOCC error handling, METHOD probe, --seed flag, --help
- **Parser imports**: import regexes from patterns.py directly instead of double-indirection through engine.py
- **Expression simplification**: `_replace_dollar_outside_strings` reduced from two passes to one
- **Jump table**: pre-compute WHILE/WEND and DO/LOOP ip mappings in `_scan_subs`; `_find_matching_wend` and `_find_matching_loop` use O(1) lookup when available
- **Scope cleanup**: removed dual-write hack in `Scope.__setitem__`; writes go to `_runtime` only, `_persistent` is read-through fallback
- **STATS output**: redirect stdout during stats runs to suppress rich console output in non-TTY mode
- **Copyright year**: LICENSE updated from 2025 to 2026
- **CLI --seed flag**: `qubasic --seed N script.qb` sets deterministic seed before execution
- **Type annotations**: added return type annotations to 14 unannotated mixin methods across locc_commands, locc_display, locc_execution, demos

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
