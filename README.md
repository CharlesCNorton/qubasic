# QUBASIC

Quantum computing in a BASIC REPL.

```
] 10 H 0
] 20 CX 0,1
] 30 MEASURE
] RUN

RAN 3 lines, 4 qubits, 1024 shots in 0.12s  [depth=2, gates=2]

  State  Count       %  Distribution
 |00>     518   50.6%  ██████████████████
 |11>     506   49.4%  █████████████████
```

QUBASIC is a quantum computing environment built on Qiskit Aer that uses BASIC syntax for circuit construction, execution, and analysis. It runs as an interactive REPL, as a script interpreter, or as a headless Python engine.

**For agents**: minimal syntax, maximum context-window efficiency. `10 H 0 / 20 CX 0,1 / 30 MEASURE / RUN` replaces 8 lines of Qiskit Python. JSON output mode (`--json`) for pipeline integration. `Engine` class for direct API use without subprocess overhead.

**For humans**: type `DEMO GROVER` and get a working 3-qubit search algorithm you can LIST, edit, and re-RUN. Type `STEP` to watch the statevector evolve line by line. Type `REWIND 3` to go back. No imports, no boilerplate, no compile step.

---

## Install

```
pip install qubasic
```

Development install:

```
pip install -e ".[charts]"
```

Requires Python >= 3.10, Qiskit >= 1.0, qiskit-aer >= 0.13.

## Usage

```
qubasic                       Interactive REPL (installed console script)
python -m qubasic_core         Same, without installing
qubasic script.qb             Run a script file
qubasic --quiet script         Suppress banner, output results only
qubasic --json script          Machine-readable JSON output
qubasic --spec                Print a JSON contract (commands, gates, functions)
qubasic --help                Show CLI help
```

Headless (no REPL):

```python
from qubasic_core.engine_state import Engine
from qubasic_core.terminal import QBasicTerminal

t = QBasicTerminal()
t.num_qubits = 2
t.process('10 H 0', track_undo=False)
t.process('20 CX 0,1', track_undo=False)
t.process('30 MEASURE', track_undo=False)
t.cmd_run()
print(t.last_counts)  # {'00': 512, '11': 512}
```

---

## Program editing

```
10 H 0              Enter a numbered line
10                   Delete line 10
LIST                 Show the program
LIST SUBS            Show subroutine definitions
LIST VARS            Show variables with types
LIST ARRAYS          Show arrays with sizes
DELETE 10            Delete line 10
DELETE 10-50         Delete range
RENUM                Renumber lines (10, 20, 30, ...)
RENUM 100 5          Renumber starting at 100, step 5
NEW                  Clear everything
UNDO                 Undo last edit
SAVE file.qb         Save program to file
LOAD file.qb         Load program from file
INCLUDE file.qb      Merge lines from file
IMPORT "lib.qb"      Load DEFs with namespace (LIB.NAME)
CHAIN "file.qb"      Load and run, preserving variables
MERGE "file.qb"      Merge without clearing
DIR                  List .qb files
AUTO                 Auto-generate line numbers
AUTO 100, 5          Start at 100, step 5
EDIT 20              Edit a specific line
COPY 10-50 TO 100   Copy line range
MOVE 10-50 TO 100   Move line range
FIND "text"          Search program lines
REPLACE "old" WITH "new"
BANK 1               Switch to program slot 1
CHECKSUM             MD5 hash of program listing
```

## Gates

### 0-parameter, 1-qubit
| Gate | Description |
|------|-------------|
| `H 0` | Hadamard |
| `X 0` | Pauli-X (NOT) |
| `Y 0` | Pauli-Y |
| `Z 0` | Pauli-Z |
| `S 0` | Phase (pi/2) |
| `T 0` | T gate (pi/4) |
| `SDG 0` | S-dagger |
| `TDG 0` | T-dagger |
| `SX 0` | Sqrt(X) |
| `ID 0` | Identity |

### 1-parameter, 1-qubit
| Gate | Description |
|------|-------------|
| `RX theta, 0` | Rotation around X |
| `RY theta, 0` | Rotation around Y |
| `RZ theta, 0` | Rotation around Z |
| `P theta, 0` | Phase gate |

### 3-parameter, 1-qubit
| Gate | Description |
|------|-------------|
| `U theta, phi, lambda, 0` | General single-qubit unitary |

### 0-parameter, 2-qubit
| Gate | Description |
|------|-------------|
| `CX 0,1` | CNOT (aliases: CNOT) |
| `CZ 0,1` | Controlled-Z |
| `CY 0,1` | Controlled-Y |
| `CH 0,1` | Controlled-H |
| `SWAP 0,1` | Swap |
| `DCX 0,1` | Double-CNOT |
| `ISWAP 0,1` | iSWAP |

### 1-parameter, 2-qubit
| Gate | Description |
|------|-------------|
| `CRX theta, 0, 1` | Controlled-RX |
| `CRY theta, 0, 1` | Controlled-RY |
| `CRZ theta, 0, 1` | Controlled-RZ |
| `CP theta, 0, 1` | Controlled-Phase |
| `RXX theta, 0, 1` | XX interaction |
| `RYY theta, 0, 1` | YY interaction |
| `RZZ theta, 0, 1` | ZZ interaction |

### 0-parameter, 3-qubit
| Gate | Description |
|------|-------------|
| `CCX 0,1,2` | Toffoli (aliases: TOFFOLI) |
| `CSWAP 0,1,2` | Fredkin (aliases: FREDKIN) |

### Modifiers
```
CTRL H 0, 1           Controlled version of any gate
INV RX 0.5, 0         Inverse/dagger of any gate
UNITARY NAME = [[..]]  Define gate from unitary matrix
BARRIER                Optimization barrier
RESET 0                Reset qubit to |0>
```

### Multi-gate lines
```
10 H 0 : CX 0,1 : RZ PI/4, 0    Colon-separated
```

## Configuration

```
QUBITS 8             Set qubit count (1-32)
SHOTS 2048           Set measurement shots
METHOD statevector   Set simulation method
METHOD GPU           Set simulation device
STATUS               Show every active mode (qubits, method, LOCC, noise, ...)
STATUS JSON          Same, as machine-readable JSON
```

### Simulation methods
`automatic`, `statevector`, `density_matrix`, `stabilizer`, `matrix_product_state`, `extended_stabilizer`, `unitary`, `superop`

Automatic selection: stabilizer for Clifford-only circuits, MPS for >28 qubits, statevector otherwise.

## Variables and expressions

```
LET angle = PI/4
x = sin(angle) * 2       LET is optional (x = 5 also works)
10 RX angle, 0           Use in gate parameters
VARS                     List all variables
CLEAR x                  Remove a variable
```

Functions and keywords are case-insensitive (`SQRT` and `sqrt` both work).

### Constants
`PI`, `TAU`, `E`, `SQRT2`, `True`, `False` (reserved; not usable as variable names)

### Math functions
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sqrt`, `log`, `exp`, `abs`, `int` (floors), `fix` (truncates), `float`, `min`, `max`, `round`, `ceil`, `floor`, `len`

### Runtime functions
`RND(x)` random, `TIMER` elapsed seconds, `FRE(0)` free RAM bytes, `POS(0)` cursor column, `PEEK(addr)` memory read, `USR(addr)` call routine

### String functions
`LEFT$(s,n)`, `RIGHT$(s,n)`, `MID$(s,n,len)`, `CHR$(n)`, `STR$(n)`, `HEX$(n)`, `BIN$(n)`, `ASC(c)`, `VAL(s)`, `INSTR(haystack,needle)`, `LEN(s)`

### Operators
Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
Comparison: `==`, `!=`, `<>`, `<`, `>`, `<=`, `>=`
Logical: `AND`, `OR`, `NOT`, `XOR`
Bitwise: `AND`, `OR`, `XOR` on integers (`6 AND 3` = 2); `NOT` is logical
Hex/binary literals: `&HFF`, `&B10110`

## Arrays

```
DIM data(10)            1D array
DIM matrix(3, 3)        Multi-dimensional (flat storage)
LET data(0) = PI
LET names$(0) = "alice" String array (name$ elements hold strings)
REDIM data(20)          Resize, clearing to zeros
REDIM PRESERVE data(20) Resize, keeping existing data
ERASE data              Delete array
OPTION BASE 1           Set array index base
```

`DIM a(n)` is inclusive: it spans indices base..n, so the declared top index is valid. A `DIM`med array enforces its declared bounds on write; an undimensioned array grows on first assignment.

## Control flow

```
10 GOTO 50
10 GOSUB 100 / RETURN
10 FOR I = 0 TO 3
20   H I
30 NEXT I
10 FOR theta = 0 TO PI STEP 0.1
10 WHILE n < 10 / WEND
10 DO WHILE x > 0 / LOOP
10 DO / LOOP UNTIL x == 0
10 IF flag == 1 THEN H 0 ELSE X 0
10 SELECT CASE x / CASE 1 / CASE 2 / CASE ELSE / END SELECT
10 ON n GOTO 100, 200, 300
10 ON n GOSUB 100, 200
10 DATA 1, 2, 3, "hello"
10 READ x, y, z
RESTORE
EXIT FOR / EXIT WHILE / EXIT DO
END
```

## Subroutines

```
DEF BELL = H 0 : CX 0,1
DEF ROT(angle, q) = RX angle, q : RZ angle, q
10 BELL                  Call subroutine
10 BELL @2               Call with qubit offset

DEF BEGIN TELEPORT(q)    Multi-line definition
  H q
  CX q, q+1
DEF END

SUB PREPARE(n)           Structured subroutine
  LOCAL i
  FOR i = 0 TO n
    H i
  NEXT i
END SUB
CALL PREPARE(3)

FUNCTION DOUBLE(x)       Function with return value
  LET DOUBLE = x * 2
END FUNCTION

CIRCUIT_DEF BELL 10-20   Capture line range as macro
APPLY_CIRCUIT BELL @4    Apply with qubit offset
```

Scope: `LOCAL`, `STATIC`, `SHARED` inside SUB/FUNCTION.

## I/O

```
PRINT "hello"            Output text
PRINT x                  Output expression value
PRINT x;                 Suppress newline
PRINT x,                 Advance to next tab zone (14 columns)
PRINT SPC(5)             Insert spaces
PRINT TAB(20)            Tab to column
PRINT USING "##.##"; x   Formatted output
PRINT STATE              Dirac notation of statevector
PRINT QUBIT(0)           Single-qubit Bloch info
PRINT ENTANGLEMENT(0,1)  Entanglement entropy between qubits
PRINT @A                 LOCC register state (Dirac notation)
INPUT "name", x          Prompt and read (retries on bad input)
LINE INPUT "text", s$    Read full line including spaces
GET k$                   Single keypress without enter
LPRINT expr              Output to stderr/log
```

### File handles
```
OPEN "data.txt" FOR OUTPUT AS #1
PRINT #1, "hello"
CLOSE #1
OPEN "data.txt" FOR INPUT AS #1
INPUT #1, line$
EOF(1)                   Returns 1 at end of file
CLOSE #1
OPEN "db.txt" FOR RANDOM AS #2
OPEN "log.txt" FOR APPEND AS #3
```

### Export
```
EXPORT                   Print OpenQASM to screen
EXPORT file.qasm         Save OpenQASM to file
CSV                      Print results as CSV
CSV file.csv             Save results to CSV
```

## Display

```
STATE                    Full statevector (amplitudes + probabilities)
HIST                     Measurement histogram
PROBS                    Probability distribution
BLOCH 0                  ASCII Bloch sphere for qubit 0
BLOCH                    All qubits (up to 8)
CIRCUIT                  Circuit diagram (text)
DECOMPOSE                Gate count breakdown
DENSITY                  Density matrix
```

Bitstrings are little-endian: qubit 0 is the rightmost character. Histograms print a `q(n-1) ... q1 q0` header so the mapping is explicit.

### Screen modes
```
SCREEN 0                 Text (default)
SCREEN 1                 Histogram
SCREEN 2                 Statevector
SCREEN 3                 Bloch spheres
SCREEN 4                 Density matrix
SCREEN 5                 Circuit diagram
```

After RUN, the selected screen mode auto-displays.

```
COLOR green              Set foreground color
COLOR cyan, black        Set foreground and background
CLS                      Clear screen
LOCATE 5, 10             Position cursor
PLAY                     Terminal bell
PROMPT "> "              Change REPL prompt
```

## Analysis

```
EXPECT Z 0               Expectation value <Z> on qubit 0
EXPECT ZZ 0 1            Two-qubit observable
ENTROPY 0                Entanglement entropy (qubit 0 vs rest)
ENTROPY 0 1              Entropy of partition {0,1} vs rest
RAM                      Memory budget and parallelism estimates
BENCH                    Benchmark simulation at various qubit counts
SWEEP var 0 PI 10        Sweep a parameter, show P(top state) vs variable
SAMPLE 500               Sample via SamplerV2 primitive
ESTIMATE ZZ 0 1          Observable estimation via EstimatorV2
```

### Inline circuit instructions
```
10 SAVE_EXPECT ZZ 0,1 -> zz_val   Expectation value (result in variable after RUN)
10 SAVE_PROBS 0,1 -> probs        Probability snapshot (result in array after RUN)
10 SAVE_AMPS 0,3 -> amps          Specific amplitudes by index (array after RUN)
10 SET_STATE |+>                   Inject named state mid-circuit
10 SET_STATE |BELL>                Bell state
10 SET_STATE |GHZ>                 GHZ state
10 SET_STATE [0.707, 0, 0, 0.707]  Explicit amplitudes (auto-normalized)
```

### Statistics
```
STATS 100                Run 100 trials, collect statistics
STATS SHOW               Show mean/stddev/min/max per state
STATS CLEAR              Reset accumulator
```

### Characterization
```
FIDELITY |BELL>          State fidelity |<target|psi>|^2 vs a named state
FIDELITY [0.707,0,0,0.707]   ...or an explicit amplitude list
TOMOGRAPHY               State tomography: reconstruct rho from Pauli expectations
TOMOGRAPHY 2000          Statistical tomography (2000 shots per Pauli basis)
PTOMOGRAPHY              Process tomography: reconstruct the Pauli Transfer Matrix (<=2 qubits)
RB                       Single-qubit randomized benchmarking (fits decay -> error/Clifford)
RB 64 12                 Sequence lengths up to 64, 12 random sequences each
```

### Partial measurement
```
MEASURE                  Measure all qubits (the default)
MEASURE 0, 2             Measure only qubits 0 and 2; histogram is over that subset
```

## Algorithm primitives

Composable building blocks applied to a qubit range (`lo-hi`, a list, or all
qubits). Qiskit-path only.

```
QFT 0-3                  Quantum Fourier transform over qubits 0..3
IQFT 0-3                 Inverse QFT
DIFFUSE 0-2              Grover diffusion operator (2|s><s| - I)
MCX 0,1,2, 3             Multi-controlled X (any number of controls, last = target)
MCZ 0,1,2                Multi-controlled Z
MCP theta, 0,1, 2        Multi-controlled phase
QADD 0-2, 3-5            In-place register add: A += B (mod 2^n), qubit 0 = LSB
QADDC 5, 0-2             In-place constant add: A += 5 (mod 2^n)
QPE 0-3 4 UGATE          Phase estimation of a UNITARY on a target register
```

## Optimization

Classical drivers over circuit parameters (VQE/QAOA). Build a parametrized
ansatz, compute a cost with `SAVE_EXPECT`, then minimize it.

```
10 RY theta, 0
20 SAVE_EXPECT Z 0 -> cost
MINIMIZE theta -> cost              Nelder-Mead minimization (dependency-free)
MINIMIZE a, b -> z0 + 0.5*z1 ITERS 200   Multi-parameter; cost is any post-run expression
GRADIENT theta -> cost             Parameter-shift gradient d(cost)/d(theta)
```

## Dynamic circuits (feedforward)

In standard mode, `MEAS` is a real mid-circuit measurement and `IF` on the
measured bit compiles to a Qiskit `if_test`, so the conditional gate runs at
simulation time based on the actual outcome (no LOCC mode needed).

```
10 H 0
20 MEAS 0 -> c           Mid-circuit measurement into classical bit c
30 IF c THEN X 0         Feedforward correction (also: IF c == 0, NOT c, with ELSE)
40 MEASURE
```

## Mixed states

```
SET_DENSITY [[0.5,0],[0,0.5]]    Inject a density matrix (uses density_matrix method)
```

## Device model

Constrain transpilation to a hardware topology and native gate set, fully
offline. After RUN, the routed depth and SWAP count are reported.

```
COUPLING linear          1D chain connectivity (also: ring, full, OFF, or 0-1, 1-2, ...)
BASIS rz, sx, x, cx      Restrict to a native gate set (BASIS OFF to clear)
```

## Import and images

```
LOADQASM file.qasm       Import OpenQASM as an editable program (2.0 built-in; 3.0 needs qiskit_qasm3_import)
SAVEPNG out.png hist     Save the histogram as a PNG (also: bloch, circuit; needs matplotlib)
```

## Hamiltonian dynamics and open systems

```
HAMILTONIAN H = 1.0 ZZ 0 1 + 0.5 X 0    Declare a Pauli-sum Hamiltonian
HAMILTONIAN H = ISING 1.0 0.5           Builders: ISING, HEISENBERG, HUBBARD, RYDBERG
10 EVOLVE H, 1.5, 20                     Trotterized e^{-iHt} (time, steps) in a circuit
LINDBLAD NONE, 1.0, 200, 1.0 SM 0        Open-system master-equation evolution
CHANNEL AD = [[1,0],[0,0.95]] ; [[0,0.31],[0,0]]   Define a Kraus channel
10 APPLYCHANNEL AD 0                      Apply a custom channel
```

## Error correction

```
QEC STEANE               Show a code (REP [d], STEANE, SHOR, SURFACE [d]) and its stabilizers
LOGICAL_ERROR_RATE STEANE 0.02   Monte-Carlo logical error rate (optimal lookup decoder)
LOGICAL_ERROR_RATE SURFACE 0.02 UF   Same, with the union-find / matching decoder
THRESHOLD REP 0.0 0.5 11         Sweep p across distances 3/5/7 (crossing at 0.5)
DISTILL 0.02             15-to-1 magic-state distillation (output error ~35 p^3)
LATTICE 0 1              Lattice-surgery joint Zbar-Zbar measurement of two patches
```

Codes: repetition (any odd distance), Steane [[7,1,3]], Shor [[9,1,3]], rotated
surface. Decoders: an optimal minimum-weight lookup table (all codes) and a
scalable union-find / matching decoder (topological codes, via the `UF` flag).

## Benchmarking and verification

```
XEB 4 10 20              Cross-entropy benchmarking fidelity
QVOLUME 4                Quantum volume heavy-output test
RBINT H                  Interleaved RB fidelity of a Clifford gate
MIRROR 16 10             Mirror-circuit benchmarking
GST                      1-qubit gate-set-style process estimate
CONCURRENCE 0 1          Two-qubit concurrence
NEGATIVITY 0             Entanglement negativity across a bipartition
PAULIPROP ZZ 0 1         Pauli-propagation expectation (Heisenberg, truncated)
```

## Advanced algorithms

```
IQPE 4 0 UGATE           Iterative phase estimation of a UNITARY eigenphase
AMPEST 5 0               Amplitude estimation of the marked amplitude
10 AMPLIFY 101            One amplitude-amplification (Grover) step
QWALK 5                  Discrete-time quantum walk on a cycle
10 GRAPHSTATE 0-1, 1-2    Prepare a graph/cluster state (MBQC resource)
10 FEATUREMAP x0 x1       ZZ feature-map data encoding (QML)
QKERNEL 0.5 0.3 ; 0.4 0.2    Quantum kernel |<phi(x)|phi(y)>|^2
SHOR 15                  Order finding / factoring of small N
HHL 1 0 0 2 1 1          Solve a 2x2 Hermitian system A x = v
```

## Beyond qubits

```
QUDIT 3 2                d-level systems: QX, QZ, QF, QSUM, QSTATE, QMEASURE
BOSONIC 1 25             Continuous-variable Fock modes
DISPLACE 0 1.0           ...DISPLACE, SQUEEZE, CAT, BS, BSTATE
```

## Compilation and resources

```
RESOURCES 1e-12 0.001    Fault-tolerant estimate (surface-code distance, qubits, runtime)
DEVICE linear 5          Calibrated offline device model (per-qubit T1/T2 + coupling map)
DEVICE ring 5 80 60      ...with custom T1/T2 in microseconds (also: heavyhex, all, OFF)
OPTIMIZE 3               Transpile the program and report the depth/gate reduction
NOISE crosstalk 0.01     Correlated two-qubit ZZ crosstalk on entangling gates
```

## Noise models

```
NOISE depolarizing 0.01          Depolarizing channel (all gates)
NOISE amplitude_damping 0.01     T1-like decay
NOISE phase_flip 0.01            T2-like dephasing
NOISE thermal 50e-6 70e-6 1e-6   T1/T2 decoherence (microseconds)
NOISE readout 0.05 0.1           Measurement bit-flip (p(0->1), p(1->0))
NOISE combined 0.01 0.02         Amplitude + phase damping
NOISE pauli 0.01 0.01 0.01       Pauli channel (px, py, pz)
NOISE reset 0.01 0.01            Spontaneous reset error
NOISE OFF                        Disable noise
```

### Per-qubit noise (memory-mapped)
```
POKE $D100, 1            Qubit 0 noise type (0=none, 1=depolarizing, 2=amp_damp, 3=phase)
POKE $D101, 0.05         Qubit 0 noise parameter
POKE $D102, 1            Qubit 1 noise type
POKE $D103, 0.02         Qubit 1 noise parameter
```

## LOCC mode

Dual-register distributed quantum simulation with classical communication.

```
LOCC 4 4                 SPLIT mode: two independent 4-qubit registers (A, B)
LOCC JOINT 3 3           JOINT mode: shared entanglement possible
LOCC 4 4 4               3-party (A, B, C)
LOCC OFF                 Return to normal mode
LOCC STATUS              Show register info
```

### LOCC commands
```
@A H 0                   Gate on register A
@B CX 0,1                Gate on register B
@A H 0 : CX 0,1         Colon inheritance within register
SEND A 0 -> m0           Mid-circuit measure A[0], store classical bit
IF m0 THEN @B X 0        Conditional gate based on classical bit
SHARE A 2, B 0           Create Bell pair (JOINT mode only)
MEAS 0 -> result         Mid-circuit measurement within register
RESET 0                  Reset qubit within register
MEASURE_X 0              X-basis measurement
MEASURE_Y 0              Y-basis measurement
SYNDROME ZZ 0 1 -> s0    Stabilizer measurement
STATE A                  Inspect register A
BLOCH A 0                Bloch sphere for register A, qubit 0
LOCCINFO                 Protocol metrics
```

SPLIT: max capacity (up to 33 qubits per register), no cross-register entanglement. Use SEND/IF for classical coordination.

JOINT: shared entanglement via SHARE. Limited to 33 total qubits.

### LOCC optimization
Programs with SEND execute the deterministic prefix (before first SEND) once, snapshot the statevector, then re-execute only the suffix per shot. Complexity: O(prefix + shots * suffix) instead of O(shots * total).

## Basis measurement and error correction

```
MEASURE_X 0              Measure in X basis (stores in mx_0)
MEASURE_Y 0              Measure in Y basis (stores in my_0)
MEASURE_Z 0              Measure in Z basis (stores in mz_0)
SYNDROME ZZ 0 1 -> s0    Non-destructive stabilizer measurement
```

SYNDROME uses an ancilla (highest qubit index). Pauli string length must match qubit count. I, X, Y, Z supported.

## Memory map

```
$0000-$003F   Zero Page (64 slots, SYS parameter passing)
$0100-$01FF   Qubit State (8 addresses per qubit)
$D000-$D00B   QPU Config (read/write)
$D010-$D014   QPU Status (read-only)
$D100-$D1FF   Per-Qubit Noise (2 addresses per qubit)
$E000-$E0B0   SYS Routine Table
$F000-$FFFF   User SYS Routines
```

### Qubit state ($0100 + qubit * 8)
| Offset | Field |
|--------|-------|
| +0 | P(\|1>) probability |
| +1 | Bloch X |
| +2 | Bloch Y |
| +3 | Bloch Z |
| +4 | Re(alpha) |
| +5 | Im(alpha) |
| +6 | Re(beta) |
| +7 | Im(beta) |

### QPU config ($D000-$D00B)
| Address | Name | Values |
|---------|------|--------|
| $D000 | num_qubits | 1-32 |
| $D001 | shots | 1+ |
| $D002 | sim_method | 0=auto, 1=statevector, 2=stabilizer, 3=MPS, 4=density |
| $D003 | sim_device | 0=CPU, 1=GPU |
| $D004 | noise_type | read-only |
| $D005 | noise_param | read-only |
| $D006 | max_iterations | loop limit |
| $D007 | screen_mode | 0-5 |
| $D008 | fusion_enable | 0/1 |
| $D009 | mps_truncation | float threshold |
| $D00A | sv_parallel_threshold | int |
| $D00B | es_approx_error | float |

### QPU status ($D010-$D014, read-only)
| Address | Name |
|---------|------|
| $D010 | gate_count |
| $D011 | circuit_depth |
| $D012 | run_time_ms |
| $D013 | total_probability |
| $D014 | entanglement_entropy |

### Per-qubit noise ($D100 + qubit * 2)
| Offset | Field |
|--------|-------|
| +0 | noise_type (0=none, 1=depolarizing, 2=amp_damp, 3=phase) |
| +1 | noise_param (float) |

### Bloch sphere POKE
POKE to qubit state addresses ($0100+q*8) prepares the qubit:
- Field 0 (P(|1>)): sets probability via RY rotation
- Fields 1-3 (Bloch x,y,z): stores target coordinates

### SYS routines ($E000-$E0B0)
| Address | Routine |
|---------|---------|
| $E000 | BELL |
| $E010 | GHZ |
| $E020 | QFT |
| $E030 | GROVER |
| $E040 | TELEPORT |
| $E050 | DEUTSCH |
| $E060 | BERNSTEIN |
| $E070 | SUPERDENSE |
| $E080 | RANDOM |
| $E090 | STRESS |
| $E0A0 | LOCC-TELEPORT |
| $E0B0 | LOCC-COORD |

### SYS commands
```
SYS $E000               Execute built-in routine (BELL)
SYS INSTALL $F000, SUB  Install user routine at address
CATALOG                  List all SYS routines
DUMP $0000 $003F         Hex dump memory range
MAP                      Full memory map with values
MONITOR                  Interactive hex monitor (type address to PEEK, addr=val to POKE)
WAIT $D013, 1            Block until (PEEK(addr) AND mask) matches
```

## Demos

```
DEMO BELL                Bell state (2 qubits)
DEMO GHZ                 GHZ entanglement (up to 8 qubits)
DEMO TELEPORT            Quantum teleportation circuit
DEMO GROVER              Grover's search (3 qubits, target |101>)
DEMO QFT                 Quantum Fourier transform (4 qubits)
DEMO DEUTSCH             Deutsch-Jozsa algorithm
DEMO BERNSTEIN           Bernstein-Vazirani (secret=1011)
DEMO SUPERDENSE          Superdense coding (message=11)
DEMO RANDOM              Quantum random number generator
DEMO STRESS              20-qubit stress test
DEMO LOCC-TELEPORT       Full teleportation with classical correction (JOINT)
DEMO LOCC-COORD          Classical coordination (SPLIT)
DEMO LIST                List all demos
```

## Debugging

```
STEP                     Step through program with state display
TRON                     Trace on (print each line number during execution)
TROFF                    Trace off
STOP                     Break at this line during execution
CONT                     Continue after STOP
BREAK 20                 Set breakpoint at line 20
BREAK                    List breakpoints
BREAK CLEAR              Clear all breakpoints
WATCH x                  Monitor variable during STOP/STEP
WATCH CLEAR              Clear all watches
PROFILE ON               Enable per-line profiling
PROFILE OFF              Disable
PROFILE SHOW             Show time/calls/gates per line
ASSERT x > 0             Halt if condition fails
```

### Time-travel debugging
```
REWIND 3                 Go back 3 statevector checkpoints
FORWARD 1                Go forward 1 checkpoint
HISTORY                  Show all checkpoints with current position
```

Checkpoints are saved during STEP mode for circuits up to 16 qubits. Each checkpoint stores the full statevector after that gate application.

## Error handling

```
ON ERROR GOTO 1000       Trap runtime errors
RESUME                   Retry the failed line
RESUME NEXT              Skip to next line
ERROR 42                 Raise user-defined error
```

Variables `ERR` (error code) and `ERL` (error line) are set when an error is trapped.

## Types

```
TYPE Point
  x AS FLOAT
  y AS FLOAT
END TYPE
```

## Named types

Defines a record structure. Fields: `INTEGER`, `FLOAT`, `STRING`, `QUBIT`.

## Quantum DATA

```
10 DATA |+>, |0>, |GHZ3>, 3.14, "hello"
20 READ state$, state2$, state3$, angle, name$
```

Quantum state names (`|+>`, `|0>`, `|1>`, `|->`, `|BELL>`, `|GHZ>`, `|GHZ3>`, `|GHZ4>`, `|W>`, `|W3>`) are recognized in DATA statements and stored as `QSTATE:` tokens.

## Performance

### Auto-scaling
QBASIC detects available RAM, estimates per-instance memory, and reports maximum qubit count and parallelism budget at startup and via the `RAM` command.

### Simulation method selection
- **automatic**: stabilizer for Clifford-only circuits, MPS for >28 qubits, statevector otherwise
- **stabilizer**: polynomial-time for Clifford circuits (H, S, CX, SWAP, etc.)
- **matrix_product_state**: memory-efficient for low-entanglement circuits; handles the full 32-qubit range that would exhaust statevector
- **extended_stabilizer**: approximate simulation for near-Clifford circuits
- **statevector**: exact, limited by RAM (~28 qubits on 16GB)
- **density_matrix**: includes mixed states, ~14 qubits on 16GB
- **unitary**: returns the full unitary matrix of the circuit
- **superop**: returns the superoperator (quantum channel)

### Tuning via memory map
```
POKE $D008, 1            Enable gate fusion (default on)
POKE $D008, 0            Disable gate fusion
POKE $D009, 1e-12        MPS truncation threshold (tighter = more accurate)
POKE $D00A, 16           Statevector parallel threshold
POKE $D00B, 0.01         Extended stabilizer approximation error
```

### Circuit caching
Transpiled circuits are cached between RUN calls when the program and configuration have not changed. Cache invalidates on any program edit, qubit count change, method change, or tuning parameter change.

### LOCC optimization
Programs with SEND use prefix/suffix splitting: the deterministic prefix (before first SEND) executes once, the statevector is snapshotted, and only the suffix (from SEND onward) re-executes per shot. Falls back to full re-execution if the prefix contains GOTO/GOSUB.

## JSON output

```
qubasic --json examples/bell.qb
```

```json
{
  "counts": {"00": 518, "11": 506},
  "num_qubits": 2,
  "shots": 1024
}
```

## Architecture

```
qubasic_core/
  cli.py                 CLI entry point (qubasic / python -m qubasic_core)
  __main__.py            Module entry point
  engine_state.py        Engine: standalone state container
  terminal.py            QBasicTerminal: REPL + command dispatch
  engine.py              Constants, gate tables, numpy simulation, LOCCEngine
  parser.py              60+ typed Stmt objects from raw strings
  statements.py          Stmt type definitions
  exec_context.py        ExecContext: unified execution state
  scope.py               Scope: layered variable system
  backend.py             QiskitBackend / LOCCRegBackend abstraction
  errors.py              Structured error hierarchy
  io_protocol.py         IOPort protocol for I/O decoupling
  expression.py          AST-based safe expression evaluator (no eval())
  control_flow.py        FOR/NEXT, WHILE/WEND, IF/THEN dispatch
  display.py             Histograms, statevector, Bloch sphere rendering
  locc.py                LOCC commands, execution, display
  analysis.py            EXPECT, ENTROPY, DENSITY, BENCH, RAM
  sweep.py               Parameter sweep with plotille charts
  memory.py              PEEK/POKE/SYS/DUMP/MAP/MONITOR
  strings.py             String functions
  screen.py              SCREEN/COLOR/CLS/LOCATE
  classic.py             DATA/READ, SELECT CASE, DO/LOOP, SWAP, DEF FN
  subs.py                SUB/FUNCTION with LOCAL/STATIC/SHARED
  debug.py               ON ERROR, TRON/TROFF, breakpoints, time-travel
  program_mgmt.py        AUTO/EDIT/COPY/MOVE/FIND/REPLACE/BANK
  profiler.py            PROFILE, gate/depth tracking, STATS
  file_io.py             SAVE/LOAD/INCLUDE/IMPORT/EXPORT/CSV/OPEN/CLOSE
  demos.py               Built-in demo circuits
  protocol.py            TerminalProtocol (mixin contract)
  mock_backend.py        MockAerSimulator for fast testing
tests/                   Test suites (test_qubasic.py, test_features.py)
examples/                Sample .qb programs
```

`Engine` holds all program state. `QBasicTerminal` inherits `Engine` + 20 mixins. Execution methods live on `QBasicTerminal`, so headless/agent use should instantiate `QBasicTerminal` (the `Engine` base is a state container only).

## License

MIT
