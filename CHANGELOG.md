# Changelog

## 0.8.0 (2026-06-19)

### Added
- Partial measurement: `MEASURE 0, 2` measures a subset of qubits and reports the histogram over just those qubits (the bare `MEASURE` still measures all).
- Process tomography: `PTOMOGRAPHY` reconstructs the circuit's Pauli Transfer Matrix (the unitary, or the full noisy channel via the superoperator when a noise model is active), and reports the trace-preserving and unital flags plus the average gate fidelity to the identity. Limited to <= 2 qubits.
- Randomized benchmarking: `RB [max_length] [samples]` runs single-qubit RB over the 24-element Clifford group (each sequence ends with its recovery Clifford), fits the survival decay `p(m) = A f^m + B`, and reports the decay `f` and the error per Clifford `(1 - f)/2`. Reflects the active noise model.

## 0.7.0 (2026-06-19)

### Added
- Dynamic circuits in the standard Aer path: `MEAS q -> c` is a real mid-circuit measurement and `IF c THEN <gate>` (also `IF c == 0`, `NOT c`, and `ELSE`) compiles to a Qiskit `if_test`, so feedforward corrections run on the actual outcome without LOCC mode. The mid-circuit measurement registers are kept out of the reported histogram.
- Algorithm primitives applied to a qubit range: `QFT`/`IQFT`, `DIFFUSE` (Grover diffusion), `MCX`/`MCZ`/`MCP` (arbitrary control count), `QADD`/`QADDC` (in-place register and constant addition mod 2^n), and `QPE` (phase estimation of a UNITARY). The QFT is least-significant-first (qubit 0 = low bit) so register reads are consistent.
- Optimization drivers: `MINIMIZE v1, v2 -> <cost expr>` runs a dependency-free Nelder-Mead optimizer over circuit parameters (VQE/QAOA), and `GRADIENT` computes the parameter-shift gradient. The cost is any expression evaluated after each run from variables the program sets (for example via `SAVE_EXPECT`).
- Device model for transpilation: `COUPLING linear|ring|full|<edges>` constrains two-qubit connectivity and `BASIS <gates>` restricts to a native gate set, both offline; RUN reports the routed depth and SWAP count.
- `FIDELITY <target>` (state fidelity against a named state or amplitude list) and `TOMOGRAPHY [shots]` (density-matrix reconstruction from Pauli expectations, exact or shot-sampled).
- `LOADQASM <file>` imports OpenQASM as an editable program (2.0 built-in; 3.0 requires the optional `qiskit_qasm3_import` package).
- `SET_DENSITY [[...]]` injects a mixed state (density matrix) for the next run, using the density_matrix method.
- `SAVEPNG <file> hist|bloch|circuit` saves a plot as a PNG (requires matplotlib; circuit diagrams also need pylatexenc).

## 0.6.5 (2026-06-19)

### Fixed
- The Bloch sphere Y component had an inverted sign, so a qubit in |+i> was reported at -Y (and |-i> at +Y). `BLOCH`, `DRAW`, `PRINT QUBIT(n)`, and the memory-mapped Bloch-Y field (`$0100 + q*8 + 2`) were all affected, and `BLOCH` mislabeled |+i> as "|-i>". X and Z were already correct. Y now matches the standard convention (`<Y> = +2 Im(<1|rho|0>)`), verified against Qiskit expectation values.

## 0.6.4 (2026-06-19)

### Fixed
- Immediate-mode `PRINT` at the REPL no longer prints a spurious statevector after its output. `PRINT` is not a dispatch-table command, so it fell through to the immediate gate path, which always dumped `|psi>`. That path now skips the dump when the typed line added no circuit operations, so gate and subroutine entries still show state while `PRINT`, `MEASURE`, and other classical statements do not.
- `--quiet` now prints results with the banner suppressed, as documented. The results branch was unreachable (guarded by `elif not quiet` inside an `if quiet or json_mode` block), so `--quiet` previously produced no output at all.
- A `MEASURE` reachable only inside a `DEF`/`SUB` body, an `IF ... THEN/ELSE` clause, or a colon compound is now detected, so the program takes the shots path (and `qubasic script.qb` auto-runs) instead of the no-measure statevector path.
- `LET m(i, j) = x` writes a multi-dimensional array element using the same flat-stride convention the expression-side accessor reads, so multi-dimensional reads and writes agree.
- Immediate-mode `LET a(i) = <expr>` at the REPL assigns the array element, matching the in-program `LET` (previously only the in-program form handled array targets).

## 0.6.3 (2026-06-18)

### Fixed
- Colon-compound `@register` lines in LOCC mode now apply every gate. `@A H 0 : @A CX 0,1` (and the inherited form `@A H 0 : CX 0,1`) were captured as a single statement and mis-tokenized, so all but the first gate were silently dropped. The parser now splits them into per-statement parts with register inheritance, and the immediate-mode REPL path does the same.
- `SAVE_EXPECT`, `SAVE_PROBS`, and `SAVE_AMPS` no longer overwrite their target variable with 0 at circuit-build time. The placeholder keeps any prior value, so a program that re-runs with the SAVE line still present can read the previous result during the build pass (the post-run extraction then fills in the fresh value).
- `PRINT` honors `;` and `,` between multiple items: `PRINT "S ="; x` concatenates and `PRINT a, b` advances to the next 14-column zone. Separators inside quoted strings and inside call parentheses (`PRINT LEFT$(s$, 3)`) are no longer treated as item breaks.
- `PRINT` variable substitution no longer rewrites identifiers inside quoted string literals, so a label such as `PRINT "value of S"` is emitted verbatim even when `S` is a defined variable.

## 0.6.2 (2026-06-18)

### Fixed
- Changing the qubit count after a run (via `QUBITS n` or `POKE $D000`) now invalidates the cached statevector, so `MAP`, a `PEEK` of the qubit-state block, and `BLOCH` no longer reshape a stale statevector to the new qubit count and crash.

## 0.6.1 (2026-06-18)

### Fixed
- Changing noise strength between runs (for example `NOISE amplitude_damping 0.1` then `0.9`) now rebuilds the circuit instead of reusing the first noise level, so sequential noise comparisons are correct.

### Changed
- The CLI now lives in `qubasic_core/cli.py`; the `qubasic` console script and `python -m qubasic_core` are unchanged, but the former top-level `qubasic` module is removed. Test suites moved under `tests/`.

## 0.6.0 (2026-06-18)

Correctness, scalability, and usability improvements.

### Fixed
- **Time-travel debugging now works**: STEP records statevector checkpoints, so REWIND/FORWARD/HISTORY operate (previously `_checkpoint_sv` was never called).
- **TRON, breakpoints, ON TIMER, ON MEASURE now fire**: the debug hooks are wired into the execution loop instead of being dead code.
- **PLOT and ANIMATE sweep correctly**: the transpiled-circuit cache key now includes variable bindings, so parameter sweeps no longer reuse the first circuit.
- **`$XXXX` memory addresses parse**: PEEK/POKE/DUMP/SYS/WAIT accept the canonical hex-address notation, not only `0x`/decimal/`&H`.
- **Multi-line IF/ELSE/END IF**: the untaken branch is properly skipped (previously both branches ran).
- **ON ERROR** reports the real ERR/ERL for build-path errors and runs handler lines through the program executor (no spurious state dumps).
- **OPTION BASE 1** is honored in array indexing; FOR with STEP 0 is rejected; bare `NEXT` and `NEXT i, j` are supported.
- **CSV/EXPORT** report correctly when only a statevector (no counts) is present.

### Performance / scalability
- Stabilizer is chosen before MPS for large Clifford circuits, and the full 2^n statevector is no longer rebuilt after every measured run (a 30-qubit GHZ went from ~16s to <1s).
- STEP evolves the statevector incrementally; DENSITY and EXPECT avoid building large matrices; HEATMAP and STATS reuse work.

### Quantum / LOCC
- LOCC depolarizing noise matches the Qiskit Aer convention; amplitude- and phase-damping channels are now simulated in the LOCC engine.
- EXPECT/ENTROPY/DENSITY/BLOCH accept a register argument in SPLIT mode; `LOCC SHOTCAP` replaces the magic shot-cap variable; SEND prefix/suffix optimization detects all control transfers.
- Per-qubit memory-mapped noise applies to multi-qubit gates; complex single-qubit amplitudes are exposed in the qubit-state block; POKE-driven state preparation is applied at build.

### Language / interface
- User-defined TYPE record fields can be read and assigned (`p.x`); ELSEIF chains and re-parsed clauses are memoized.
- SAVE/LOAD round-trip noise, seed, screen mode, DEF FN, and TYPE definitions; CHAIN/MERGE/INCLUDE/IMPORT honor a `QUBASIC_PATH` search path; IMPORT no longer pollutes SAVE output.
- CLI gains `--version` and `--agent`; `--json` output is enriched (statevector, variables, manifest) and emits structured errors; the agent write-sandbox engages under `--json`/`--agent`.
- `SET_STATE` persists into the next RUN and warns when a named state cannot fit; per-command `HELP <name>`; dependency-version warnings are suppressed before the banner.

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
