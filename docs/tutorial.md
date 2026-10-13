# QUBASIC tutorial

This tutorial walks through QUBASIC from a first circuit to error correction.
Type the lines of each example at the `]` prompt (`qubasic` starts it), or save
them in a `.qb` file and run `qubasic file.qb`. Numbered lines are stored as the
program; anything else runs at once. The examples here are executed by the test
suite (`tests/test_project.py`), and where an `output` block follows an example the
lines it shows are checked against what QUBASIC prints. Results that come from
sampling are shown as plain text: with `SEED` set they repeat from run to run on
one installation, but the exact counts depend on the simulator version.

## A first circuit

A Bell pair: a Hadamard puts qubit 0 in superposition, a CNOT copies it onto
qubit 1, and the measurement shows only `00` and `11`.

```qubasic
QUBITS 2
SEED 7
10 H 0
20 CX 0, 1
30 MEASURE
RUN
```

```text
RAN 3 lines, 2 qubits, 1024 shots in 0.02s  [depth=3, gates=4, method=stabilizer, CNOT=1]
  bit order  q1 q0   (qubit 0 = rightmost)

 State  Count       %  Distribution
  |11⟩    521   50.9%  ███████████████████████████████████
  |00⟩    503   49.1%  █████████████████████████████████
```

The summary names the simulation method QUBASIC chose: a circuit of Clifford
gates runs on the stabilizer simulator at any width, and other circuits on a
statevector while one fits in memory.

Bitstrings put qubit 0 on the right, as Qiskit does; `OPTION ENDIAN BIG`
reverses every displayed bitstring. `LIST` shows the program, a line number on
its own deletes that line, and typing a line number that exists replaces it.

```qubasic
20 CZ 0, 1
LIST
20 CX 0, 1
```

```output
10  H 0
20  CZ 0, 1
30  MEASURE
```

A gate typed without a line number runs at once, on a fresh |0…0⟩, and shows
the state it makes; a colon joins several gates into one such line.

```qubasic
H 0 : CX 0, 1
```

```output
|ψ⟩ = +0.707|00⟩ +0.707|11⟩
```

`STATE` prints the amplitudes of the last run, and `PROBS` the probabilities.
A program can also record values while it runs: `SAVE_EXPECT` stores an
expectation value in a variable when RUN finishes.

```qubasic
NEW
QUBITS 2
10 H 0
20 CX 0, 1
30 SAVE_EXPECT ZZ 0 1 -> zz
40 SAVE_EXPECT Z 0 -> z
RUN
PRINT "ZZ ="; zz; " Z ="; z
```

```output
ZZ = 1  Z = 0
```

## BASIC

QUBASIC runs QBASIC programs. Line numbers are optional in a `.bas` file and
when the program is given through the Python API; at the prompt each program
line carries a number.

```qubasic
NEW
10 TYPE Planet
20   name AS STRING * 8
30   moons AS INTEGER
40 END TYPE
50 DIM p(1 TO 3) AS Planet
60 FOR i = 1 TO 3
70   READ p(i).name, p(i).moons
80 NEXT i
90 total% = 0
100 FOR i = 1 TO 3
110   PRINT USING "\      \ ##"; p(i).name; p(i).moons
120   total% = total% + p(i).moons
130 NEXT i
140 PRINT "Moons:"; total%
150 DATA Earth, 1, Mars, 2, Jupiter, 95
RUN
```

```output
Earth     1
Mars      2
Jupiter  95
Moons: 98
```

Procedures have their own variables and take arguments by reference.
`FUNCTION` returns the value assigned to its name, labels can stand in for
line numbers, and `SELECT CASE` branches on values, ranges and comparisons.

```qubasic
NEW
10 DECLARE FUNCTION Fact& (n%)
20 FOR k = 1 TO 4
30   GOSUB Classify
40 NEXT k
50 END
60 Classify:
70 SELECT CASE Fact&(k)
80   CASE 1: PRINT k; "! is one"
90   CASE 2 TO 9: PRINT k; "! is small:"; Fact&(k)
100  CASE IS >= 10: PRINT k; "! is"; Fact&(k)
110 END SELECT
120 RETURN
130 FUNCTION Fact& (n%)
140   IF n% <= 1 THEN Fact& = 1 ELSE Fact& = n% * Fact&(n% - 1)
150 END FUNCTION
RUN
```

```output
 1 ! is one
 2 ! is small: 2
 3 ! is small: 6
 4 ! is 24
```

Variables follow QBASIC's types: `%` integer, `&` long, `!` single, `#`
double and `$` string, or `DIM x AS type`. Assigning out of range is an
OVERFLOW error and a string to a number a TYPE MISMATCH, both trappable with
`ON ERROR GOTO`.

```qubasic
NEW
10 ON ERROR GOTO Handler
20 n% = 40000
30 PRINT "not reached"
40 END
50 Handler:
60 PRINT "error"; ERR; "at line"; ERL
70 RESUME NEXT
RUN
```

```output
error 6 at line 20
not reached
```

## Measurement inside a program

`MEAS q -> bit` measures one qubit in the middle of the circuit. An `IF` on the
bit that applies gates becomes a dynamic circuit, as in teleportation: the
state of qubit 0 arrives on qubit 2 once the two corrections are applied.

```qubasic
NEW
QUBITS 3
10 RY 1.1, 0
20 H 1
30 CX 1, 2
40 CX 0, 1
50 H 0
60 MEAS 0 -> a
70 MEAS 1 -> b
80 IF b THEN X 2
90 IF a THEN Z 2
100 SAVE_EXPECT Z 2 -> z2
RUN
PRINT USING "<Z2> = #.###"; z2
PRINT USING "cos(1.1) = #.###"; COS(1.1)
```

```output
<Z2> = 0.454
cos(1.1) = 0.454
```

When a program uses a measured bit any other way (prints it, does arithmetic
with it, or loops on it) it runs live, one shot at a time, and each statement
sees the real outcome. Only the first shot prints; the histogram covers all of
them. This loop repeats until the qubit reads 1.

```qubasic
NEW
QUBITS 1
SHOTS 200
10 tries = 0
20 H 0
30 MEAS 0 -> m
40 tries = tries + 1
50 IF m = 0 THEN RESET 0: GOTO 20
60 MEASURE
RUN
```

The summary line of that run reads `method=live`, and every shot ends in `1`.

A live run holds a statevector per shot while one fits in memory. Past that,
or under `METHOD stabilizer`, each shot holds a stabilizer tableau, so a
Clifford program that branches on its measurements runs at widths no
statevector reaches. This one entangles 100 qubits, reads both ends in every
shot and checks one against the other.

```qubasic
NEW
QUBITS 100
SHOTS 50
10 H 0
20 FOR i = 0 TO 98: CX i, i + 1: NEXT i
30 MEAS 0 -> first
40 MEAS 99 -> last
50 IF first <> last THEN PRINT "the ends disagree"
60 MEASURE 0, 99
RUN
BLOCH 50
```

The summary reads `method=live, stabilizer tableau`, the histogram holds only
`00` and `11`, and `BLOCH 50` shows qubit 50 on a pole, collapsed with the
rest by the first measurement. `STATE` lists the stabilizer generators.

## Noise and error mitigation

`NOISE` attaches a noise model to every run. Channels combine with `;`, and
`NOISE INFO` lists what is active.

```qubasic
NEW
QUBITS 2
10 H 0
20 CX 0, 1
30 MEASURE
NOISE depolarizing 0.01 0.03; readout 0.04 0.06
NOISE INFO
```

With readout error present, `MITIGATE READOUT ON` calibrates each qubit's
flip probabilities and every following RUN also shows the histogram with that
error inverted (kept in `last_mitigated` for the Python API).

```qubasic
MITIGATE READOUT ON
RUN
```

Zero-noise extrapolation measures an observable with the gate noise amplified
by folding (each gate g becomes g g† g at scale 3) and extrapolates to zero
noise. `_ZNE` holds the estimate and `_ZNE_IDEAL` the noiseless value.

```qubasic
NOISE depolarizing 0.01 0.03
ZNE 1.0 ZZ 0 1 SCALES 1 3 5 FIT EXP
PRINT USING "raw #.### mitigated #.### exact #.###"; _ZNE_NOISY; _ZNE; _ZNE_IDEAL
NOISE OFF
```

```output
raw 0.970 mitigated 1.000 exact 1.000
```

Depolarizing noise shrinks ⟨ZZ⟩ by the same factor at every fold, so the
exponential fit recovers the noiseless value exactly here; other noise and
other fits leave a residual error, which ZNE prints beside the estimate.

A qubit also loses coherence to qubits the program itself couples it to, with
no noise model at all. Entangled with eight others, its ⟨X⟩ falls to a few
percent: the phase has moved into the environment. Because the environment is
part of the program, the loss can be undone, as a Hahn echo undoes it on real
hardware: flipping the qubit halfway through reverses the coupling and the
coherence returns in full. A `NOISE` channel describes the same loss as a
fixed rate, and no pulse brings it back.

```qubasic
NEW
QUBITS 9
10 H 0
20 FOR k = 1 TO 8: H k: RZZ 0.4 + 0.1 * k, 0, k: NEXT k
30 SAVE_EXPECT X 0 -> plain
RUN
PRINT USING "without echo #.###"; plain
```

```output
without echo 0.021
```

```qubasic
NEW
QUBITS 9
10 H 0
20 FOR k = 1 TO 8: H k: RZZ (0.4 + 0.1 * k) / 2, 0, k: NEXT k
30 X 0
40 FOR k = 1 TO 8: RZZ (0.4 + 0.1 * k) / 2, 0, k: NEXT k
50 SAVE_EXPECT X 0 -> echoed
RUN
PRINT USING "with echo #.###"; echoed
```

```output
with echo 1.000
```

## Algorithms

Each algorithm command builds and runs its circuits and prints the result next
to the exact answer. Grover search finds marked states with the optimal number
of iterations:

```qubasic
NEW
SEED 7
GROVER 1011
```

Quantum phase estimation reads the ground energy of molecular hydrogen in the
STO-3G basis, whose Hamiltonian QUBASIC computes itself:

```qubasic
HAMILTONIAN H2 = MOLECULE H2 0.7414
QPE H2 BITS 10
PRINT USING "QPE ##.## Ha, exact ##.## Ha"; _QPE_ENERGY; _QPE_EXACT
```

```output
QPE -1.14 Ha, exact -1.14 Ha
```

The reading lands on a grid of spacing 2π/(t·2^BITS), so more counting bits
give more digits.

The variational eigensolver reaches the same energy, and `EXCITED k` finds the
next levels by deflation. QAOA solves MaxCut on a graph:

```qubasic
VQE H2 LAYERS 1
QAOA RING 4 P 2
```

`MOLECULE` builds other small molecules the same way, from STO-3G integrals
and a Hartree-Fock calculation done in QUBASIC: H2, HEH+, LIH, BEH2, H2O, NH3,
N2 and H4 by name, or any geometry of atoms from H to Ne in quotes. `FREEZE`
drops the core orbitals, and `ANSATZ UCCSD` runs unitary coupled cluster from
the Hartree-Fock state, which is exact for two active electrons.

```qubasic
HAMILTONIAN LIH = MOLECULE LIH FREEZE
VQE LIH ANSATZ UCCSD
PRINT USING "UCCSD ##.###### Ha, exact ##.###### Ha"; _VQE_ENERGY; _VQE_EXACT
```

```output
UCCSD -7.882176 Ha, exact -7.882176 Ha
```

Other commands in the same family: `SIMON`, `DJ`, `BV`, `QCOUNT`, `IQPE`,
`AMPEST`, `SHOR`, `HHL`, `MINIMIZE` over a program of your own, and the
benchmarks `RB`, `XEB`, `QVOLUME` and `MIRROR`.

## Error correction

`QEC` shows a code and its stabilizers. `LOGICAL_ERROR_RATE` estimates how
often the decoded logical qubit fails, at code capacity by default, under
phenomenological noise with `PHENOM`, or with every gate, measurement and reset
of the syndrome circuit noisy with `CIRCUIT`.

```qubasic
QEC STEANE
LOGICAL_ERROR_RATE REP 5 0.05 2000
LOGICAL_ERROR_RATE SURFACE 3 0.002 CIRCUIT 2000
```

`LQUBITS` turns the program's qubits into logical qubits. Under the Steane
code the program runs encoded: seven physical qubits per logical qubit,
transversal gates, a round of syndrome extraction after each gate and a
decoded readout.

```qubasic
NEW
LQUBITS 2 CODE STEANE PHYS 0.001
10 H 0
20 CX 0, 1
30 MEASURE
RUN
LQUBITS OFF
```

## Graphics and sound

`SCREEN 13` opens QBASIC's 320 x 200, 256-colour mode. The picture appears when
the program ends: as colour cells on a terminal, as shading elsewhere.
`SAVEPNG <file> SCREEN` writes it as an image.

```qubasic
NEW
10 SCREEN 13
20 LINE (20, 20)-(300, 180), 14, B
30 CIRCLE (160, 100), 60, 12
40 PAINT (160, 100), 4, 12
50 FOR x = 0 TO 319 STEP 4
60   PSET (x, 100 + 40 * SIN(x / 25)), 10
70 NEXT x
80 DRAW "BM160,30 C15 R20 D20 L20 U20"
RUN
SAVEPNG picture.png SCREEN
```

`PLAY` takes QBASIC's music strings and `SOUND` a frequency and a duration;
`SAVEWAV` writes what they made as a WAV file.

```qubasic
NEW
10 PLAY "T120 O3 L8 C D E F G4 G4"
20 SOUND 440, 9
RUN
SAVEWAV tune.wav
```

## Files

Sequential files work as in QBASIC, and RANDOM files hold fixed-length
records, read and written whole.

```qubasic
NEW
10 TYPE Entry
20   label AS STRING * 6
30   value AS DOUBLE
40 END TYPE
50 DIM rec AS Entry
60 OPEN "table.dat" FOR RANDOM AS #1 LEN = LEN(rec)
70 FOR i = 1 TO 3
80   rec.label = "row" + LTRIM$(STR$(i))
90   rec.value = i * i / 2
100  PUT #1, i, rec
110 NEXT i
120 GET #1, 2, rec
130 PRINT rec.label; rec.value; LOF(1)
140 CLOSE #1
RUN
```

```output
row2   2  42
```

`LEN` gives a record's size in bytes (6 for the label, 8 for the DOUBLE), so the
file holds three 14-byte records.

## From Python

The `qubasic_core` package runs programs without a terminal. `run` takes
program text and returns counts, variables, the printed output and any error
lines; a `Session` keeps one interpreter across calls.

```python
from qubasic_core import Session, run

result = run('''
10 H 0
20 CX 0, 1
30 MEASURE
''', qubits=2, shots=1000, seed=7)
assert set(result['counts']) == {'00', '11'}

s = Session(qubits=1, seed=3)
s.execute('10 RY theta, 0')
s.execute('20 SAVE_EXPECT Z 0 -> z')
s.execute('theta = 0')
assert abs(s.run()['variables']['z'] - 1.0) < 1e-9
```

For everything else, `HELP` lists the commands, `HELP <name>` describes one,
and [reference.md](reference.md) lists every command, statement, function and
gate.
