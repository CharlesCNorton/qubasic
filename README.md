# QUBASIC

Quantum computing in a BASIC interpreter.

```
] 10 H 0
] 20 CX 0, 1
] 30 MEASURE
] RUN

RAN 3 lines, 2 qubits, 1024 shots in 0.02s  [depth=3, gates=4, method=stabilizer, CNOT=1]
  bit order  q1 q0   (qubit 0 = rightmost)

 State  Count       %  Distribution
  |11⟩    521   50.9%  ███████████████████████████████████
  |00⟩    503   49.1%  █████████████████████████████████
```

QUBASIC is a BASIC interpreter whose programs build and run quantum circuits.
The language is QBASIC's: typed variables, records, arrays with bounds,
SUB and FUNCTION, labels, SELECT CASE, PRINT USING, sequential, random and
binary files, graphics on QBASIC's SCREEN modes, PLAY and SOUND, timers and
error trapping, so QBASIC programs run unchanged, with the graphics screen
drawn in the terminal and sound recorded for `SAVEWAV`. Gates are statements
(`H 0`, `CX 0, 1`), measurements feed back into the program, and RUN simulates
the result on Qiskit Aer or on QUBASIC's own engines. Around that sit noise
models and error mitigation, quantum algorithms, error-correcting codes with
circuit-level noise, and analysis, benchmarking and visualization commands.
It runs as an interactive REPL, as a script interpreter, as a Python library,
as a Jupyter kernel and as a browser REPL.

**Documentation**: [tutorial](docs/tutorial.md) (from a first circuit to error
correction, every example tested), [reference](docs/reference.md) (every
command, statement, function and gate, generated from the code), and
[Development](#development) below for how the interpreter is built. At the
prompt, `HELP` lists the commands and `HELP <name>` describes one.

## Install

```
pip install qubasic
```

Requires Python 3.10 or newer; Qiskit, Qiskit Aer, NumPy, SciPy and rustworkx
come with it. Optional extras: `qubasic[qec]` (stim and pymatching, for the
`STIM` and `MWPM` paths of the error-correction commands), `qubasic[jupyter]`
(the notebook kernel), `qubasic[qasm3]` (OpenQASM 3 import) and
`qubasic[charts]` (sweep charts). For development, `pip install -e ".[charts]"`.

## Running it

```
qubasic                       interactive REPL
qubasic program.qb            run a QUBASIC script (numbered lines plus commands)
qubasic program.bas           run QBASIC source; unnumbered lines are numbered as loaded
qubasic --quiet program.qb    results only, no banner
qubasic --json program.qb     the result as JSON (counts, variables, state, method)
qubasic --seed 7 program.qb   reproducible sampling
qubasic --spec                a JSON description of every command, statement and function
qubasic --web [port]          browser REPL on localhost, token-gated
qubasic --install-kernel      install the Jupyter kernelspec (then pick "QUBASIC")
python -m qubasic_core        the same without the console script
```

From Python, `run` executes a program and returns its result, and a `Session`
keeps one interpreter across calls:

```python
from qubasic_core import Session, run

result = run('''
10 H 0
20 CX 0, 1
30 MEASURE
''', qubits=2, shots=1000, seed=7)
result['counts']      # {'00': 503, '11': 497}
result['output']      # what the program printed
result['errors']      # the ?... lines, if any

s = Session(qubits=1, seed=3)
s.execute('10 RY theta, 0')
s.execute('20 SAVE_EXPECT Z 0 -> z')
s.execute('theta = 0.5')
s.run()['variables']['z']     # cos(0.5)
```

`run` and `Session` take `qubits`, `shots`, `seed`, `method`, `precision`,
`noise` (a NOISE specification) and `inputs` (answers for INPUT). Source with
no numbered line is read as QBASIC source.

## The language

Numbered lines form the program; anything else runs at once (gates typed at
the prompt act on a fresh |0…0⟩ and show the state they make). `LIST`, `EDIT`,
`DELETE`, `RENUM` (which rewrites every line reference), `UNDO`, `SAVE` and
`LOAD` manage it, and `LOAD` reads `.bas` files as QBASIC source.

```
10 TYPE Planet
20   name AS STRING * 8
30   moons AS INTEGER
40 END TYPE
50 DIM p(1 TO 3) AS Planet
60 FOR i = 1 TO 3: READ p(i).name, p(i).moons: NEXT i
70 FOR i = 1 TO 3
80   PRINT USING "\      \ ##"; p(i).name; p(i).moons
90 NEXT i
100 DATA Earth, 1, Mars, 2, Jupiter, 95
```

The dialect follows QBASIC: `%` `&` `!` `#` `$` suffixes, `DEFINT`-style letter
ranges and `DIM ... AS`, with OVERFLOW and TYPE MISMATCH errors on assignment;
records (nested up to 16 deep), fixed-length strings, arrays with lower bounds and
`LBOUND`/`UBOUND`; `SUB` and `FUNCTION` with local variables, `STATIC`,
`SHARED`, by-reference arguments and typed or whole-array parameters; labels as
jump targets; block `IF`/`ELSEIF`, `SELECT CASE`, `DO`/`LOOP`,
`WHILE`/`WEND`, `FOR` with QBASIC's zero-trip rule; `ON ERROR GOTO` and
`RESUME` with QBASIC's error codes; `INPUT` and `LINE INPUT`, `PRINT USING`
with the full field set, and the string, math and conversion functions. An
unassigned variable reads as 0 inside a program. Expressions are parsed with
QBASIC precedence and evaluated without Python's `eval`.

Files: `OPEN` for INPUT, OUTPUT, APPEND, RANDOM and BINARY, with `PRINT #`,
`WRITE #`, `INPUT #`, `LINE INPUT #`, `FIELD`, `LSET`/`RSET`, `GET #`/`PUT #`
of records, `SEEK`, `LOF`, `LOC`, `EOF`, `INPUT$`, `MKI$`/`CVI` and their
relatives, and `KILL`, `NAME`, `MKDIR`, `RMDIR`, `CHDIR` and `FILES`.

Graphics: `SCREEN` 1, 2 and 7 to 13 open QBASIC's graphics modes, with `PSET`,
`LINE` (boxes, styles), `CIRCLE` (arcs, ellipses), `PAINT`, `DRAW`, `VIEW`,
`WINDOW`, `PALETTE`, `GET`/`PUT` of images, `POINT` and `PMAP`. The picture is
drawn in the terminal in colour cells, or as shading off a terminal. At a
terminal it is redrawn in place while the program runs, from the first
`SLEEP`, `INKEY$`, `GET` or `INPUT` or after half a second, so animations and
games show as they run; otherwise it appears when the program ends. `SHOW`
redraws it and `SAVEPNG file.png SCREEN` saves it. At an interactive terminal a
program runs until it ends or Ctrl+C stops it; off a terminal and in agent mode
a run stops after 100,000 statements (`POKE $D006` changes the limit), and a
circuit build stops at a million instructions everywhere.

Sound: `PLAY` takes QBASIC's music macro language and `SOUND` a frequency and
duration; `SAVEWAV file.wav` writes the result. Events: `ON TIMER(n) GOSUB`
with `TIMER ON|OFF|STOP`, and `ON MEASURE GOSUB`.

## Quantum programs

| Gates | |
|---|---|
| one qubit | `H X Y Z S T SDG TDG SX ID` |
| one qubit, angles | `RX RY RZ P` (one angle), `U` (three) |
| two qubits | `CX CZ CY CH SWAP DCX ISWAP`, `CRX CRY CRZ CP RXX RYY RZZ` (one angle) |
| three qubits | `CCX CSWAP` |

Angles come first: `RX PI/4, 0`, `CRZ theta, 0, 1`. `CTRL` and `INV` modify any
gate, `UNITARY NAME = [[...]]` defines one from a matrix, `DEF NAME = gates`
defines a macro, and `QFT`, `IQFT`, `DIFFUSE`, `MCX`, `MCZ`, `MCP`, `QADD`,
`QPE`, `GRAPHSTATE`, `FEATUREMAP` and `EVOLVE` (Trotterized time evolution)
are statements that expand into gates. Bitstrings put qubit 0 on the right, as
Qiskit does; `OPTION ENDIAN BIG` reverses every displayed bitstring.

`MEASURE` measures every qubit (or a list) into the histogram. `MEAS q -> bit`
measures in the middle of the program: an `IF` on the bit that applies gates
compiles to a dynamic circuit, and a program that uses a measured bit any
other way (prints it, computes with it, loops on it) runs live, one shot at a
time, with every statement seeing the real outcome:

```
10 H 0
20 MEAS 0 -> m
30 IF m = 0 THEN RESET 0: GOTO 10     ' repeat until the qubit reads 1
40 MEASURE
```

`SAVE_EXPECT`, `SAVE_PROBS` and `SAVE_AMPS` record values into variables when
RUN finishes; `STATE`, `PROBS`, `BLOCH`, `DENSITY`, `CIRCUIT`, `EXPECT`,
`ENTROPY`, `FIDELITY` and `HEATMAP` inspect the last state; `STEP` runs a
program a line at a time and `REWIND`/`FORWARD` move through the states it
passed.

## Simulation

`METHOD automatic` (the default) runs noiseless Clifford circuits on the
stabilizer simulator at any width, other circuits on a statevector while one
fits in the memory available (at double precision, then single), and wider
ones as matrix product states; the summary line names the method used.
`METHOD` also selects `statevector`, `density_matrix`, `stabilizer`,
`matrix_product_state`, `extended_stabilizer`, `unitary` or `superop`, and
`PRECISION SINGLE` halves statevector memory. Each run is sized against the
memory free at that moment, and one that cannot fit is refused with what it
needs and what to change; `RAM` reports what fits. `QUBASIC_MEMORY_LIMIT=4GB`
caps what QUBASIC treats as available. `COUPLING`, `BASIS` and `DEVICE`
constrain transpilation to a hardware topology and native gates.

## Noise and error mitigation

```
NOISE depolarizing 0.001 0.01; readout 0.02 0.03     channels joined with ;
NOISE thermal 50e-6 70e-6 50e-9                       T1, T2 and gate time
NOISE ADD amplitude_damping 0.01                      extend the active model
MITIGATE READOUT ON                                   invert readout error on every RUN
ZNE 1.0 ZZ 0 1 SCALES 1 3 5 FIT EXP                   zero-noise extrapolation
```

`MITIGATE READOUT` calibrates each qubit's flip probabilities and shows every
following histogram with them inverted. `ZNE` measures an observable with the
gate noise amplified by folding and extrapolates to zero noise with a linear,
quadratic, Richardson or exponential fit, printing the noiseless value beside
it.

## Algorithms

`GROVER`, `SIMON`, `DJ` (Deutsch-Jozsa), `BV` (Bernstein-Vazirani), `QPE`
(energy of a Hamiltonian by phase estimation), `QCOUNT` (quantum counting),
`IQPE`, `AMPEST` (maximum-likelihood amplitude estimation), `SHOR`
(Beauregard's circuit for N < 256), `HHL` (any size), `QAOA` (MaxCut graphs or
a diagonal Hamiltonian), `VQE` (with `EXCITED k` for excited states),
`QWALK` and `QKERNEL`. `MINIMIZE` and `GRADIENT` optimize a program's own
parameters (Nelder-Mead, SPSA or parameter-shift gradient descent), compiling
the circuit once with bound parameters when the program allows.
`HAMILTONIAN H = MOLECULE H2 0.7414` builds molecular hydrogen's STO-3G
Hamiltonian with a built-in integrals engine, and `ISING`, `HEISENBERG`,
`HUBBARD` and `RYDBERG` build model Hamiltonians; `LINDBLAD` evolves open
systems. Benchmarks: `RB`, `RBINT`, `XEB`, `QVOLUME`, `MIRROR`, `GST`,
`TOMOGRAPHY` and `PTOMOGRAPHY`.

## Error correction

```
QEC SURFACE 5                                        a code and its stabilizers
LOGICAL_ERROR_RATE SURFACE 7 0.05 MATCH              code capacity, matching decoder
LOGICAL_ERROR_RATE SURFACE 5 0.002 CIRCUIT           circuit-level noise
LOGICAL_ERROR_RATE BB 0.01                           bivariate-bicycle code, BP+OSD
THRESHOLD SURFACE 0.002 0.012 5 CIRCUIT              threshold from crossing curves
LQUBITS 2 CODE STEANE PHYS 0.001                     run the program encoded
LATTICE CNOT 0 1                                     lattice-surgery CNOT
```

Codes: repetition, Steane, Shor, rotated surface and bivariate-bicycle qLDPC.
Decoders: minimum-weight lookup, exact minimum-weight matching, pymatching and
BP+OSD. `PHENOM` adds measurement errors over d rounds, and `CIRCUIT` makes
every gate, measurement and reset of the syndrome circuit noisy, building the
detector error model itself (`CIRCUIT STIM` uses stim instead). `LQUBITS`
runs a program on logical qubits, simulated encoded or on a modeled logical
channel, and `DISTILL` and `RESOURCES` estimate magic-state distillation and
fault-tolerant costs.

## More models

`LOCC 4 4` splits the machine into registers that share only classical bits
(`SEND`) or Bell pairs (`SHARE`, in `LOCC JOINT`). `QUDIT` simulates d-level
systems and `BOSONIC` Fock-space modes with displacement, squeezing, cat
states and beamsplitters. A memory map (`PEEK`, `POKE`, `SYS`, `MAP`,
`MONITOR`) exposes qubit state, simulator settings and per-qubit noise as
addresses.

## Development

```
python -m pytest tests/                     the test suite
ruff check qubasic_core tests tools         lint
mypy qubasic_core                           types (strict)
python tools/generate.py                    regenerate host.py's stubs and docs/reference.md
```

The interpreter is one object, `QBasicTerminal` (`terminal.py`), assembled
from a state container and 58 mixins, one per concern, in six layers. A module
imports only from its own layer or the layers below it, and the same holds for
calls through `self`: each layer has a host class in `host.py` declaring what
its mixins may use, the attributes and methods of its own layer and the ones
below, and each mixin derives from its layer's host, so mypy rejects a call
into a higher layer. The few places where a lower layer must reach up, such as
RUN printing its histogram or LOAD entering lines as if typed, are hooks: a
method declared with a docstring at the lowest layer that calls it and
implemented only above it. `tools/generate.py` writes the method stubs and
places each attribute at the lowest layer that uses it.

| Layer | Modules |
|---|---|
| core | `host` (the layer table and hosts), `engine` (limits, result codes, errors, the I/O protocol, output and path services) |
| language | `parser` (statement parser, compiler, catalog), `expression` (expressions, variables, formatting), `interpreter` (control flow, procedures, statements), `devices` (graphics, sound, screen, files), `editing` |
| simulation | `simtools` (capacity, the numpy statevector engine, backends), `simulator` (RUN, circuit build, quantum statements, live runs), `noise`, `locc`, `physics` (dynamics, chemistry, qudits, bosonic modes), `machine` (settings, memory map, profiler, demos) |
| applications | `algorithms`, `benchmarking`, `qec` (codes, decoders, circuit-level noise), `logical` (logical qubits, lattice surgery) |
| presentation | `display`, `analysis` |
| composition | `terminal`, `api`, `cli` (with the browser REPL), `jupyter_kernel` |

`process(line)` stores a numbered line, parsing it once into a typed
statement, and passes anything else to `dispatch`, which calls the matching
`cmd_*` method or runs the line as an immediate statement. RUN compiles the
program, resolving labels and matching blocks, and reports a missing target or
an unmatched block before anything runs. It then executes the program as a
Qiskit circuit on the Aer method the memory allows, live and shot by shot on
the numpy engine when the program reads a measured bit, as LOCC registers, or
encoded on logical qubits.

A command is a method `cmd_<name>(self, rest)` on the mixin of its concern
whose docstring opens with its syntax, registered in `_CMD_WITH_ARG` or
`_CMD_NO_ARG` (`terminal.py`) and placed under a topic in `COMMAND_TOPICS`
(`parser.py`), after which `tools/generate.py` declares it in its layer's host
and lists it in the reference. A statement is a handler in its layer's table
(`_LANGUAGE_STATEMENTS` in `interpreter.py`, `_QUANTUM_STATEMENTS` in
`simulator.py`, `_CIRCUIT_STATEMENTS` in `algorithms.py`), which the terminal
merges into `_STMT_KEYWORDS`, with its line in the catalog's `STATEMENTS`.
`tests/test_project.py` checks the layering of imports and of calls, the
hooks, the stubs, the docstrings, the catalog's coverage of every dispatched
keyword and the generated reference.
Continuous integration runs the suite on Python 3.10 to 3.13 across Linux,
Windows and macOS with coverage, the lint and type checks,
`tools/generate.py --check`, and a build of the wheel that installs it and
runs a program.

## License

MIT
