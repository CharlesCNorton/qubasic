# Changelog

## 0.20.0 (2026-10-12)

### Added
- Records: `TYPE Name … END TYPE` declares fields of INTEGER, LONG, SINGLE, DOUBLE, STRING, `STRING * n` or another record type. `DIM b AS Body` and `DIM crowd(1 TO 3) AS Body` create records and arrays of records, whose fields (`b.pos.x`, `crowd(2).name`) work in expressions, LET, INPUT, READ, SWAP and by-reference arguments. Assigning a whole record copies it, fixed-length strings pad and truncate, and a malformed TYPE block is a compile error. Record variables work at the prompt, and SAVE writes them.
- Typed and array procedure parameters: in `SUB Total (a() AS INTEGER, n AS INTEGER, s AS LONG)` each argument converts to its parameter's type, and a string passed for a number or the reverse is TYPE MISMATCH (13). `v()` passes an array by reference, so a REDIM PRESERVE inside the routine reaches the caller, and records pass by reference. FUNCTIONs take array arguments, and arrays DIMmed inside a procedure are local to the call.
- Line labels (`Handler:`) as targets of GOTO, GOSUB, ON … GOTO/GOSUB, RESTORE, ON ERROR GOTO and RESUME, and `RETURN n`, `RESTORE n` and `RESUME 0`, which retries the failing statement. An undefined or duplicate label, or a jump to a missing line, is a compile error.
- Typed variables: the `%`, `&`, `!`, `#` and `$` suffixes, `DEFINT`/`DEFLNG`/`DEFSNG`/`DEFDBL`/`DEFSTR` letter ranges and `DIM x AS type`, with OVERFLOW (6) and TYPE MISMATCH (13) on assignment and DUPLICATE DEFINITION (10) on assigning a CONST. CINT and CLNG round half to even, `\`, `EQV` and `IMP` are operators, and AND, OR, XOR and NOT act on rounded integers.
- Arrays take lower bounds (`DIM a(1 TO 3, 0 TO 1) AS INTEGER`) and bounds computed from expressions. LBOUND and UBOUND take a dimension argument, an index outside the bounds is SUBSCRIPT OUT OF RANGE (9), and SWAP exchanges elements. An array used without DIM is created on first use: one dimension grows as it is written, and several dimensions span OPTION BASE to 10.
- Input and output: INPUT reads several values, with the `;` and `,` prompt forms and `?Redo from start`. LINE INPUT, WRITE and the `MID$` statement are available. `OPEN … FOR INPUT|OUTPUT|APPEND AS #n` works with PRINT #, PRINT # USING, WRITE #, INPUT #, LINE INPUT #, CLOSE, FREEFILE and LOF, and reading past the end of a file is error 62. `?` is PRINT, BEEP writes a bell, SYSTEM ends the program, and DECLARE, WIDTH, CLS, COLOR, LOCATE, SCREEN and SLEEP are accepted.
- PRINT USING takes QBASIC's field set: `#`, `.`, `,`, `+`, a trailing `-`, `**`, `$$`, `**$`, `^^^^`, `!`, `\  \`, `&` and `_` literals, with `%` marking a number too wide for its field.
- Functions: UCASE$, LCASE$, LTRIM$, RTRIM$, SPACE$, STRING$, OCT$, HEX$, INSTR with a start position, TIMER, DATE$, TIME$, INKEY$, `RANDOMIZE n` and `RND(0)`. VAL parses as QBASIC does (`&H`, `&O`, embedded spaces), and ERR reports QBASIC error codes.
- QBASIC source files: LOAD and the command line take `.bas` files and number their unnumbered lines, and `SAVE name.bas` writes the program lines alone. `examples/classic.bas` is a classical QBASIC program.
- Composable noise: `NOISE depolarizing 0.001 0.01; readout 0.02` joins channels with `;`, `NOISE ADD` extends the active model and `NOISE INFO` lists it. `depolarizing p1 [p2]` takes a separate rate for two- and three-qubit gates, and single-qubit channels act on every qubit a multi-qubit gate touches. `thermal T1 T2 tg [tg2]` takes a multi-qubit gate time and requires T2 ≤ 2·T1, `readout p01 [p10]` is asymmetric, and readout errors compose as a matrix product.
- `LATTICE` runs lattice surgery on rotated surface-code patches of any odd distance in a stabilizer simulation (`qubasic_core/lattice.py`, `surface.py`). It merges and splits patches prepared in 0, 1, + or − along ZZ or XX and compares the merge outcome with the split readout (`_LATTICE_PLUS`, `_LATTICE_AGREE`). `LATTICE CNOT c t` runs the merge-and-split CNOT, and `LATTICE RATE d,… p [ROUNDS r]` estimates a noisy merge's logical failure rate under phenomenological noise, decoded by space-time matching (`_LATTICE_RATE`).
- `LOGICAL_ERROR_RATE … PHENOM` runs a memory experiment under phenomenological noise (data and check-measurement errors in each of d rounds, then a noisy readout). It decodes by space-time matching on a detector error model (`qubasic_core/dem.py`) and needs no stim.
- Encoded simulation for more codes: under LQUBITS, RUN simulates a program of transversal gates encoded, STEANE as a circuit and SURFACE, REP and SHOR by Pauli frames, and reports the logical error rate (`_LOGICAL_ERR`) and the TVD. Other programs run on the modeled logical channel with the lattice-surgery compilation report, and `MODEL` or `ENCODED` forces one path.
- `GROVER <bits> [<bits> ...]` searches for one or more marked states with the optimal iteration count (or `ITERS k`), reading the states in OPTION ENDIAN order, and `PROGRAM` writes its circuit. `SIMON <secret>` recovers a hidden XOR period.
- `QAOA` runs on MaxCut graphs (`0-1, 1-2:0.5`, `RING n`, `COMPLETE n`, `RANDOM n [p]`) or a diagonal HAMILTONIAN, with `P`, `METHOD`, `ITERS`, `SHOTS` and `PROGRAM`. It reports the approximation ratio and the probability of an optimal answer (`_QAOA_RATIO`, `_QAOA_POPT`).
- `VQE <HAMILTONIAN>` minimizes over a hardware-efficient RY or RY-RZ ansatz with `LAYERS`, `RESTARTS` and `PROGRAM`, and prints the exact ground energy beside its result (`_VQE_ENERGY`, `_VQE_EXACT`).
- Python API (`qubasic_core.run`, `qubasic_core.Session`): runs `.qb` or QBASIC source under settings (qubits, shots, seed, method, precision, noise, INPUT answers) and returns counts, variables, the statevector of a small register, the printed output and the error lines.

### Changed
- HHL solves systems of any size, given as matrix and vector literals and padded to a power of two, with `CLOCK n` and `SHOTS n` sampling the post-selected solution (`_HHL_FIDELITY`, `_HHL_X`). A non-Hermitian matrix runs through its Hermitian dilation.
- SHOR runs Beauregard's circuit for N < 256: Fourier-basis adders and controlled modular multipliers on 2n + 3 qubits, with the phase read through one recycled control qubit. `ORACLE` runs phase estimation of the permutation x → a·x mod N. Even N, primes and prime powers are settled classically (`_SHOR_ORDER`, `_SHOR_FACTOR`).
- RENUM rewrites every line-number reference: GOTO, GOSUB, ON … GOTO/GOSUB lists, ON ERROR GOTO, RESUME, RETURN, RESTORE and IF … THEN/ELSE line numbers. Strings and comments keep their text.
- SAVE writes PRECISION, DEVICE, COUPLING, BASIS, the LQUBITS mode, the composed NOISE specification and strings containing quotes, so LOAD restores the session.
- PRINT and STR$ show numbers to seven significant digits without a leading zero (`.5`, `.3333333`, `1.234568E+07`).
- An assignment at the prompt follows the variable's type: `n% = 2.5` stores 2, and OVERFLOW and TYPE MISMATCH are reported.
- EOF returns -1 at end of file and 0 otherwise, QBASIC's true and false.
- A program without gates prints only its own output, with no RUN summary.
- NOISE OFF also clears a DEVICE model and LOCC noise. LOCC mode takes the noise model when it is a single depolarizing, amplitude-damping or phase-flip channel.
- The inner runs of COMPARE, PLOT, ANIMATE, MINIMIZE, GRADIENT and STATS discard the program's printed output.
- Expressions are parsed with QBASIC precedence into a tree and evaluated (`qubasic_core/basic_expr.py`).
- Each command's docstring opens with its syntax, which `HELP <command>` prints.
- The terminal is split into modules by concern, and mypy checks attribute access across the mixins against `HostState` (`qubasic_core/host.py`).
- scipy and rustworkx are declared dependencies, and the wheel includes `examples/classic.bas`.

## 0.19.0 (2026-10-10)

### Added
- Live execution: a program that reads a mid-circuit measurement bit as a number (PRINT, LET, comparisons, GOTO, loops) runs shot by shot on a numpy statevector, with MEAS returning the collapsed outcome. Only the first shot prints, later shots replay its INPUT answers and write no files, and the summary reads `method=live`. Afterwards variables hold the first shot's values and SAVE_EXPECT/SAVE_PROBS the mean over shots. Noise models run as quantum trajectories with readout error. MEASURE_X/Y/Z and SYNDROME bits behave the same. IF <bit> THEN <gates> still compiles to a dynamic circuit.
- `LQUBITS <n> CODE STEANE [PHYS p]` simulates the program encoded: [[7,1,3]] blocks, transversal Clifford gates, depolarizing noise after every physical gate and on readout, a syndrome-extraction round with conditional corrections after every logical gate, and block-wise decoded readout on the stabilizer simulator. RUN reports the total variation distance from the ideal distribution (`_LOGICAL_TVD`). The other codes are labeled as modeled.
- `PRECISION SINGLE|DOUBLE` (also `$D00C`): complex64 amplitudes for Aer statevector-family methods, LOCC, STEP and live runs.
- Memory model (`qubasic_core/capacity.py`): runs are sized against the RAM available at the moment, and one that cannot fit is refused with its requirement and the alternatives. The banner, QUBITS, STATUS and RAM report the widest statevector run at each precision.
- Entanglement and state analysis without dense 2^n x 2^n matrices (`qubasic_core/statetools.py`): reduced density matrices accumulated block by block, used by ENTROPY, NEGATIVITY, CONCURRENCE, EXPECT, CONSISTENCY, BLOCH and HEATMAP.
- `PRINT STATE`, `PRINT QUBIT(q)` and `PRINT ENTANGLEMENT(a, b)` inside a program show the state at that line.
- HEATMAP shows pairwise quantum mutual information; `HEATMAP CONCURRENCE` shows concurrence.
- Exact minimum-weight perfect matching decoder (rustworkx blossom on the code's matching graph with boundary nodes; `MATCH`, alias `UF`). Lookup tables are built in order of error weight, and the default decoder is lookup while the syndrome space allows and matching beyond. THRESHOLD takes decoder flags and `TRIALS n`.
- DEVICE heavyhex builds Qiskit's heavy-hex coupling map. Device noise sits on coupled pairs only.

### Changed
- Colon compounds run every statement whatever it is (`PRINT 1: PRINT 2`, `LET a = 1: LET b = 2`, `RESET 0: X 0`), and a colon inside a string is text. GOTO, GOSUB, RETURN and END inside a compound take effect. An IF owns the rest of its line, as do REM, `'` and DEF.
- Loops follow QBASIC. Single-line loops (`FOR I = 1 TO 3: H I: NEXT I`) and loops opened mid-line work. A FOR whose range the step cannot reach is zero-trip, and the counter ends one step past the limit. WHILE and DO keep one loop-stack entry per loop. NEXT, WEND and LOOP close loops a jump left open. EXIT FOR/WHILE/DO search statement by statement. GOSUB, ON GOSUB and CALL return to the next statement on the line, inside IF clauses as well.
- A single `=` inside an expression is equality (`IF x = 2 THEN`, `flag = a = b`). `^` is exponentiation and `MOD` the BASIC remainder (sign of the dividend). SGN, SQR, ATN, CINT, CLNG, CSNG and CDBL are available. A bare `PRINT` prints a blank line.
- In a running program an unassigned variable reads 0 (`""` for `name$`); qubit indices, gate parameters, matrix literals and prompt commands still reject unknown names.
- PRINT formats numbers as BASIC does: a leading space for non-negative values, a trailing space, whole numbers without a decimal point, exponents as `1E+20`.
- SELECT CASE takes value lists, `a TO b` ranges, `IS <op> v` comparisons, exact string cases and `CASE v: statement` on one line.
- STEP runs as one live shot, so MEAS collapses the displayed state and IF/PRINT see the outcome. STEP honors END.
- LOCC runs print on the first shot only, keep the first shot's variables, and re-execute the prefix per shot when noise is active.
- APPLYCHANNEL leaves METHOD unchanged. Shots sample the channel as trajectories, and a run without MEASURE keeps the mixed state for DENSITY.
- The exact LINDBLAD path keeps the Hamiltonian and jump operators sparse and reaches 10 qubits.
- A RUN with no gates skips the simulator.
- Measured runs keep their final state from inside the same simulation when memory allows. Otherwise it is computed on first use.
- The transpile cache is keyed on a digest of the built circuit, so DEF, UNITARY, SET_STATE and variable changes rebuild it.
- The numpy engine (LOCC, STEP, live runs) updates states in place, block by block.
- AMPEST is maximum-likelihood amplitude estimation over the program's state (`AMPEST [m] <qubit | |bits>> [SHOTS n]`), with a Fisher-information spread.
- HHL uses a signed clock register and a uniformly controlled RY, and reports fidelity and post-selection probability.
- CTRL takes gate parameters and UNITARY gates in both execution paths; UNITARY gates run in LOCC mode.
- OPTION ENDIAN applies to CSV, CLIP, COMPARE, ANIMATE, Lindblad populations and qudit output (qudit 0 rightmost).
- MEAS and RESET reject out-of-range qubits. A MEAS repeated in a loop compiles, with feedforward reading the latest outcome. SYNDROME and MEASURE_X/Y/Z bits drive IF feedforward.
- The web REPL serializes requests on its one terminal, compares the token in constant time, caps request size, and treats INPUT as end of input. The Jupyter kernel captures output per cell and asks INPUT through the notebook when allowed.
- Time-travel checkpoints hold up to 64 MiB each within a 256 MiB budget.
- LQUBITS OFF restores the noise model that was active before.
- SAMPLE samples the program's MEASURE subset (or every qubit) with the active noise model, precision and SEED.
- METHOD rejects unknown method names. POKE $D000 clamps to the method's qubit ceiling.
- PAULIPROP refuses programs that measure or reset.
- PROFILE reports the gates each line adds.
- CHAIN and MERGE read files that begin with a byte-order mark.

## 0.18.0 (2026-07-03)

### Added
- `OPTION ENDIAN BIG|LITTLE` — bitstring display order, the BASIC-flavored sibling of `OPTION BASE`. BIG shows qubit 0 leftmost (the textbook order) across every displayed bitstring: histograms and their bit-order header, STATE, PROBS, STEP's live statevector, SWEEP lines, STATS, LOCC register and joint histograms (per-register, keeping the A|B|C order), CSV, and the JSON `counts` (whose `bit_order` field records the active convention, as does STATUS). Bitstring-shaped input follows the display, so the `AMPLIFY` target you type is the histogram line you read. Internal counts keys and statevector indexing keep the qiskit little-endian order, so programs that string-match keys are unaffected.

### Changed
- `SAVE` persists `OPTION BASE` and `OPTION ENDIAN`, so a LOADed program restores its conventions.

## 0.17.0 (2026-07-02)

Hardening from an SMT-verifier triage (touchstone-prover over all 651
functions: 153 proved trap-free, 3 refutations traced, none reachable from
the language surface).

### Changed
- `_pmul` raises a clear ValueError on a non-Pauli character and `_emit_mcz` on an empty qubit list, instead of a raw KeyError/IndexError — both are internal preconditions (every caller feeds alphabet-constrained values), now self-documenting.
- `_eval_with_vars` declares the ParameterExpression passthrough in its return annotation instead of claiming `-> float`.

## 0.16.0 (2026-07-02)

Research-grade error correction, per-method scale, parametric variational
loops, molecular Hamiltonians, and two new front ends (Jupyter, web). The
existing command surface is unchanged; everything below is additive.

### Added
- Circuit-level QEC: `LOGICAL_ERROR_RATE <code> <d> <p> CIRCUIT [shots]` runs stim-generated syndrome-extraction circuits (noisy gates, measurements, and resets over d rounds) decoded from the detector error model by pymatching MWPM — a d=21 surface patch samples ~13k shots/s. `MWPM` runs batched pymatching at code capacity for distances the lookup table cannot reach. Both need the `[qec]` extra (stim + pymatching).
- Bivariate-bicycle qLDPC codes: `QEC BB [l m]` builds the family from cyclic-shift polynomials (6 6 gives [[72,12,6]], 12 6 the [[144,12,12]] gross code) with k verified by GF(2) ranks, and `LOGICAL_ERROR_RATE BB <p>` decodes with an internal belief-propagation + ordered-statistics (BP+OSD-0) decoder — no external dependency.
- Logical-qubit mode: `LQUBITS <n> CODE <name> <d> [PHYS p]` runs the program on logical qubits — every op carries the code's per-op logical error channel, measurement a logical readout flip, and RUN appends the lattice-surgery report (surgery ops, syndrome rounds, physical qubits, error budget). `LQUBITS OFF` returns to physical mode.
- Molecular Hamiltonians: `HAMILTONIAN H = MOLECULE H2 [R]` builds the exact 4-qubit Jordan-Wigner Hamiltonian from a self-contained STO-3G integrals engine (no pyscf); exact diagonalization reproduces the FCI energy -1.1373 Ha at R=0.7414, and `SAVE_EXPECT <name> -> var` now accepts a declared Hamiltonian for VQE cost functions.
- Optimizers: `MINIMIZE ... METHOD SPSA` (two evaluations per iteration, robust to shot noise) and `METHOD GRAD` (parameter-shift gradient descent with adaptive step) join Nelder-Mead.
- Lindblad trajectories: `LINDBLAD ... TRAJ [n]` switches to Monte Carlo wavefunction unraveling (sparse H_eff + quantum jumps), raising the open-system ceiling from 5 to 15 qubits.
- Jupyter kernel (`qubasic --install-kernel`, then pick "QUBASIC") and a token-gated browser REPL (`qubasic --web [port]`, localhost by default).
- A golden-script test layer (`tests/test_golden.py`): whole programs asserted end to end on physics identities (inverse round trips, teleportation fidelity, EXPORT-LOADQASM fixed point, FCI energies) and on the 0.15.0 audit regressions.

### Changed
- Qubit ceilings are per METHOD: statevector keeps the 32-qubit memory wall, stabilizer reaches 4096 (an 881-qubit GHZ runs in ~12 s), MPS/automatic 1024, extended_stabilizer 63. Switching METHOD clamps QUBITS when needed, and QUBITS/RAM report memory in method-appropriate terms instead of a misleading 2^n estimate.
- MINIMIZE, GRADIENT, and SWEEP compile the circuit once with bound qiskit Parameters and re-bind per evaluation when the ansatz permits (reported as "parametric compile"), an order of magnitude on variational loops; non-symbolic programs fall back to the rebuild path.
- GST characterizes 2-qubit programs (product fiducials, 16x16 Pauli Transfer Matrix) as well as 1-qubit.
- The STATS progress spinner writes only to an interactive terminal, and SWEEP shows a transient progress bar there; piped and captured output is unchanged.
- `PRINT`ing a quantum DATA token shows ket notation (|GHZ3>) instead of the storage form; IQPE hints when the target register has no eigenstate preparation; LOCCINFO branch statistics count every SEND across all shots instead of the bits' final values; LOADQASM points at `pip install qubasic[qasm3]` when the 3.0 importer is missing.

## 0.15.0 (2026-07-02)

Error-handling, calibration, and I/O fixes from a full-surface audit. The
quantum engine is unchanged.

### Fixed
- `RESUME`, `RESUME NEXT`, and `RESUME <line>` return to the mainline after a trapped error: the executor routes per-line errors through the ON ERROR handler in-loop, so a handler runs as ordinary program code and resumes where the README says. Error trapping arms per run (a stale target from a previous RUN no longer traps early), the 200-line handler budget applies per activation, and an error raised inside a handler is fatal.
- `$D014` (entanglement_entropy) is populated after each run with the qubit-0-vs-rest entropy in bits, matching `ENTROPY 0`; it always read 0.0.
- `EXPORT` emits OpenQASM 2.0 when the circuit allows, so its output round-trips through `LOADQASM` without the optional `qiskit_qasm3_import` package. Dynamic circuits still export as 3.0, and the message reports the version written.
- `SWEEP` restores the swept variable afterward instead of leaving it at the final sweep point, so a following RUN or STATS sees the program's own value.
- A UTF-8 BOM no longer mangles the first statement: piped stdin and all program readers (script files, `LOAD`, `INCLUDE`, `IMPORT`, `LOADQASM`) decode utf-8-sig.

### Changed
- `XEB` reports the self-normalized linear fidelity (sampled numerator over the perfect-sampling denominator `2^n sum(p^2) - 1`), so 1.0 is the true ideal at any width; the raw estimator tops out at (D-1)/(D+1), 0.6 at 2 qubits.
- `OPTIMIZE` reports a depth increase as such instead of a negative reduction.
- The `SWEEP` chart's y-axis is fixed to [0, 1].
- `DUMP` prints registers wider than a byte compactly (`698.6`) instead of truncating the decimal.
- Qiskit's advisory warning about passing coupling_map/basis_gates with a backend is silenced on the device-model path, where that combination is by design.

## 0.14.0 (2026-06-19)

Second audit-gap round: a complete agent contract, BASIC truth values, range
comparisons, louder mid-circuit-bit handling, string-array defaults, and lazy
density inspection. The quantum engine is unchanged.

### Added
- `qubasic --spec` now reports the full language surface: a `statements` section (QFT, MEAS, SYNDROME, EVOLVE, QADD, CTRL/INV, the control-flow keywords, ...) alongside commands, each with a `signature` field, plus `true_value: -1`.
- `DIM s$(n)` declares a string array; unread elements default to `""` instead of `0.0`.

### Changed
- Comparisons return BASIC truth values: `-1` for true, `0` for false (so `LET t = (a > b)` is `-1`).
- Comparison chaining is Python-style, so `0 <= x <= 10` reads as `(0 <= x) AND (x <= 10)`.
- Reading a mid-circuit measurement bit (`MEAS`/`SYNDROME -> var`) in a classical expression now raises a clear error instead of silently using the placeholder 0; `IF <bit>` feedforward is unaffected.
- `DENSITY` solves the density matrix lazily, on demand, instead of on every measured `SET_DENSITY` run.

## 0.13.0 (2026-06-19)

Closes the remaining audit gaps in the classic-BASIC layer (conventions,
inspection, and an agent contract). The quantum engine is unchanged.

### Added
- `qubasic --spec` emits a machine-readable JSON contract (version, commands with arg flag and one-line help, gates, functions, constants, bit order) so an agent can load the exact surface of the installed version.
- `REDIM PRESERVE name(n)` keeps existing elements; plain `REDIM` clears to zeros (QBASIC semantics).
- `DENSITY` shows the density matrix after a measured `SET_DENSITY` run, not only a no-MEASURE one.

### Changed
- `DIM` uses inclusive sizing: `DIM a(n)` spans indices base..n (the declared top index is valid), matching QBASIC instead of the previous C-style element count.
- `round()` rounds half away from zero (`round(2.5)` = 3), not Python's banker's rounding.
- `STR$(n)` reserves a leading space for non-negative numbers (`STR$(42)` = " 42").
- A chained comparison such as `a < b < c` now raises a clear "ambiguous" error instead of silently using Python chaining; write `(a < b) AND (b < c)`.
- `PRINT`ing a mid-circuit measurement bit (`MEAS`/`SYNDROME -> var`) shows "mid-circuit bit, resolved per shot" rather than the placeholder 0.

## 0.12.0 (2026-06-19)

Correctness and robustness pass on the classic-BASIC layer, from an extended
adversarial audit. The quantum engine is unchanged.

### Fixed
- `SHARED` variables in a `SUB` now propagate their modifications back to the caller instead of being discarded when the call's scope is restored (an accumulator across two calls now sums correctly: `8`, not `3`).
- `RESTORE` resets the `DATA` read pointer inside a program; it was a no-op, so a `READ` after `RESTORE` raised `OUT OF DATA`.
- Writes to an explicitly `DIM`med array are bounds-checked like reads instead of silently auto-extending past the declared size. Implicit (undimensioned) arrays still grow on first assignment.
- String arrays work: `LET s$(i) = ...` stores and reads string elements.
- `SET_DENSITY` followed by inspection without a `MEASURE` no longer raises a raw "unable to translate" error from Aer; the density matrix is captured and shown by `DENSITY`.
- A script that issues an explicit `RUN` and also contains a `MEASURE` no longer auto-runs (and prints) the program a second time.

### Changed
- `STATUS` reports the user-variable count only, excluding the internal `_DEPTH`/`_GATES`/`_TIME` set after a run.
- `DEF FN name(x) = expr` now works at the prompt (immediate mode), matching the in-program form.
- `HELP` refreshed (`STATUS`, `INT` floors / `FIX` truncates, implicit `LET`, `<>`/`XOR`, corrected `MEAS`); `--help` notes the `-q`/`-v`/`-h` short flags and `python -m qubasic_core`.

## 0.11.1 (2026-06-19)

### Fixed
- A configuration/session command (`QUBITS`, `SHOTS`, `METHOD`, `LOCC`, ...) placed on a numbered program line now reports a clear error explaining it must run without a line number (before `RUN`), instead of the misleading `UNKNOWN GATE`. Unnumbered config in a script was, and remains, the correct usage.

## 0.11.0 (2026-06-19)

Convention-conformance and observability pass: make the language behave the way
a Qiskit/Python/BASIC user already expects, and make hidden state legible. The
quantum engine is unchanged.

### Added
- `STATUS` (and `STATUS JSON`): dumps every active mode (qubits, shots, method, device, seed, `OPTION BASE`, LOCC mode, noise, coupling/basis, pending `SET_STATE`/`SET_DENSITY`, bank) so behavior can be read instead of inferred.
- Implicit `LET`: a bare assignment works without the keyword (`x = 5`, `s$ = "hi"`, `a(1) = 5`), as in every BASIC.
- `FIX(x)` truncates toward zero, complementing `INT`.
- Histograms print a `q(n-1) ... q1 q0` bit-order header and the JSON result carries a `bit_order` field, making the little-endian convention (qubit 0 = rightmost) explicit.

### Changed
- `INT(x)` floors toward negative infinity (`INT(-3.2)` = -4), matching QBASIC; use `FIX` for truncation toward zero.
- Built-in constants (`PI`, `E`, `TAU`, `SQRT2`) are reserved: a same-named variable can no longer silently shadow them, and assigning one is an error.
- `MEAS` without a target bit reports a clear error distinguishing it from `MEASURE` (mid-circuit measurement into a classical bit vs collapse into the histogram) instead of `UNKNOWN GATE`.

## 0.10.1 (2026-06-19)

Expression, string, and PRINT fixes in the classic-BASIC layer. The quantum
engine is unchanged.

### Fixed
- Math functions resolve in any case: `SQRT`, `SIN`, `ABS`, `INT`, ... now work as well as their lowercase spellings, matching the rest of the language. Runtime and string functions (`RND`, `LEFT$`, ...) are likewise case-insensitive.
- `AND`, `OR`, `NOT`, `XOR`, and `<>` work in every expression context (`LET`, `PRINT`, gate parameters), not only in `IF` conditions. `AND`/`OR`/`XOR` are bitwise on integers (`6 AND 3` = 2) while preserving operator precedence below comparison, so `IF a > b AND c > d` still groups correctly; `NOT` is logical.
- String variables are fully usable: `LET s$ = "foo" + "bar"`, `LET t$ = LEFT$(s$, 3)`, and other computed strings store their real value in both immediate and program mode, instead of being mistaken for a quoted literal or forced through `float()`.
- `PRINT` surfaces evaluation errors instead of silently printing the unevaluated source text. Assigning a string to a numeric variable reports a `TYPE MISMATCH` rather than crashing on a float conversion.

## 0.10.0 (2026-06-19)

Completes the frontier set: fault-tolerant QEC extras and a compilation/resources
group. All offline on Qiskit/Aer/numpy.

### Added
- Surface code: `QEC SURFACE [d]`, a rotated surface code at odd distance d (validated to correct every weight-1 error).
- Union-find / matching decoder: exact minimum-weight matching of syndrome defects, scalable and dependency-free, selected with the `UF` flag on `LOGICAL_ERROR_RATE`. It tracks the optimal lookup decoder on the repetition and surface codes; the lookup decoder remains the default and the right choice for non-topological codes like Steane and Shor.
- Magic-state distillation: `DISTILL <p>` runs the 15-to-1 protocol (via the [15,11,3] Hamming detection), with cubic output-error suppression ~35 p^3.
- Lattice surgery: `LATTICE <a> <b>` performs the joint Zbar_A Zbar_B measurement that merges two repetition patches (the building block of a lattice-surgery CNOT).
- Fault-tolerant resource estimation: `RESOURCES <target_pL> <physical_p>` reports the surface-code distance, physical qubits per logical, T-count, T-factory overhead, and an approximate runtime.
- Calibrated device models: `DEVICE linear|ring|heavyhex|all [n] [T1us] [T2us]` builds a per-qubit thermal-relaxation noise model and coupling map for offline hardware-realistic simulation.
- `NOISE crosstalk <p>`: a correlated two-qubit ZZ crosstalk channel on entangling gates.
- `OPTIMIZE [level]`: transpiles the program and reports the depth and gate-count reduction.

## 0.9.0 (2026-06-19)

Frontier and experimental methods. All offline on Qiskit/Aer/numpy; nothing
needs a cloud backend or an external simulation engine.

### Added
- Hamiltonian dynamics and open systems: `HAMILTONIAN` (Pauli sum, plus `ISING`, `HEISENBERG`, `HUBBARD` Jordan-Wigner, and `RYDBERG` builders), `EVOLVE` (Suzuki-Trotter time evolution, verified to fidelity 1.0 against the exact propagator), `LINDBLAD` (RK4 master-equation open-system evolution), and `CHANNEL`/`APPLYCHANNEL` (user-defined CPTP Kraus channels).
- Quantum error correction: `QEC` (repetition, Steane, Shor codes with self-verified logical operators), an optimal minimum-weight lookup decoder, `LOGICAL_ERROR_RATE` (Pauli-frame Monte Carlo), and `THRESHOLD` sweeps (the repetition crossing lands at p = 0.5).
- Frontier benchmarking: `XEB` (cross-entropy benchmarking), `QVOLUME` (quantum volume heavy-output test), `RBINT` (interleaved RB), `MIRROR` (mirror benchmarking), `GST` (1-qubit linear-inversion gate-set-style estimate), and entanglement measures `CONCURRENCE` and `NEGATIVITY`.
- Advanced algorithms: `IQPE` (iterative phase estimation), `AMPEST` (amplitude estimation), `AMPLIFY` (amplitude amplification), `QWALK` (discrete-time quantum walk), `GRAPHSTATE` (graph/cluster states for measurement-based computing), `FEATUREMAP`/`QKERNEL` (quantum-machine-learning encoding and kernels), `SHOR` (order finding and factoring, factors 15 into 3 x 5), and `HHL` (2x2 linear solver).
- Pauli propagation: `PAULIPROP` estimates Pauli expectation values by Heisenberg-picture backpropagation with weight truncation (exact when nothing is pruned, matching `EXPECT`).
- Qudit engine: `QUDIT`, `QX`, `QZ`, `QF`, `QSUM`, `QSTATE`, `QMEASURE` for d-level systems (independent of the qubit engine).
- Bosonic / continuous-variable engine: `BOSONIC`, `DISPLACE`, `SQUEEZE`, `CAT`, `BS`, `BSTATE` for truncated Fock-space modes.

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
