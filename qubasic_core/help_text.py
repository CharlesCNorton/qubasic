"""QUBASIC help text and banner art."""

HELP_TEXT = """\

QUBASIC — Quantum BASIC Terminal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROGRAM EDITING
  10 H 0              Enter a numbered line (program statement)
  10                  Delete line 10
  LIST                Show the program
  DELETE 10           Delete line 10
  DELETE 10-50        Delete range
  RENUM               Renumber lines (10, 20, 30, ...)
  NEW                 Clear everything
  SAVE <file>          Save program to .qb file
  LOAD <file>          Load program from .qb file
  RUN                 Execute the program

GATES (immediate or in a program)
  H 0                 Hadamard on qubit 0
  X 0 / Y 0 / Z 0    Pauli gates
  S 0 / T 0           Phase gates (and SDG, TDG)
  RX PI/4, 0          Rotation gates (RX, RY, RZ, P)
  CX 0, 1             CNOT (control, target)
  CZ 0, 1             Controlled-Z
  CCX 0, 1, 2         Toffoli
  SWAP 0, 1           Swap two qubits
  MEASURE             Add measurements (use in program)

MULTI-GATE LINES
  10 H 0 : CX 0,1    Multiple gates on one line with ':'

REGISTERS & SUBROUTINES
  REG data 3          Name a group of qubits
  10 H data[0]        Use register notation
  REGS                List registers
  DEF BELL = H 0 : CX 0,1
                      Define a named gate sequence
  10 BELL             Use it in a program
  10 BELL @2          Use with qubit offset
  DEFS                List subroutines

VARIABLES & LOOPS
  LET angle = PI/4    Set a variable
  10 RX angle, 0      Use in gate parameters
  10 FOR I = 0 TO 3   Loop (variable substitution in body)
  20   H I
  30 NEXT I
  VARS                List variables

DISPLAY
  STATE               Show statevector (after RUN)
  HIST                Show measurement histogram
  PROBS               Show probability distribution
  BLOCH [n]           ASCII Bloch sphere for qubit n (or all)
  STEP                Step through program with state display
  CIRCUIT             Show circuit diagram

CONFIGURATION
  QUBITS n            Set number of qubits (default: 4)
  SHOTS n             Set number of shots (default: 1024)
  METHOD name         Set simulation method (automatic, statevector,
                      matrix_product_state, stabilizer, ...)

DEMOS
  DEMO LIST           List available demos
  DEMO BELL           Bell state
  DEMO GHZ            GHZ entanglement
  DEMO TELEPORT       Quantum teleportation
  DEMO GROVER         Grover's search
  DEMO QFT            Quantum Fourier transform
  DEMO DEUTSCH        Deutsch-Jozsa algorithm
  DEMO BERNSTEIN      Bernstein-Vazirani
  DEMO SUPERDENSE     Superdense coding
  DEMO RANDOM         Quantum random number generator
  DEMO STRESS         20-qubit stress test
  DEMO LOCC-TELEPORT  Teleportation across A/B boundary (JOINT)
  DEMO LOCC-COORD     Classical coordination (SPLIT)

LOCC MODE (dual-register distributed quantum simulation)
  LOCC <n_a> <n_b>          SPLIT mode: two independent registers
  LOCC JOINT <n_a> <n_b>    JOINT mode: shared entanglement possible
  LOCC OFF                  Back to normal Aer mode
  LOCC STATUS               Show register info
  @A H 0                    Gate on register A
  @B CX 0,1                 Gate on register B
  SEND A 0 -> x             Mid-circuit measure A[0], store in x
  IF x THEN @B X 0          Conditional gate based on classical bit
  SHARE A 2, B 0            Create Bell pair (JOINT mode only)
  STATE A / STATE B         Inspect register states
  BLOCH A 0 / BLOCH B 0     Bloch spheres per register

  SPLIT: max capacity (31+31), no cross-register entanglement
  JOINT: shared entanglement, limited to ~32 total qubits
  LOCCINFO                  Protocol metrics after run
  CONNECT "host:port" AS C   Attach remote register (stub — local sim only)
  DISCONNECT C               Detach remote register

BASIS MEASUREMENT
  MEASURE_X qubit         Measure in X basis (H before measure)
  MEASURE_Y qubit         Measure in Y basis (SDG+H before measure)
  MEASURE_Z qubit         Measure in Z basis (standard)
  Results stored in mx_<q>, my_<q>, mz_<q> variables.

ERROR CORRECTION
  SYNDROME ZZ 0 1 -> s0   Measure Pauli stabilizer non-destructively
  Uses an ancilla (highest qubit index). Pauli string
  length must match qubit count. I/X/Y/Z supported.

ADVANCED
  UNITARY NAME = [[..]]   Define gate from unitary matrix (standard basis order)
  CTRL gate ctrl, tgt     Controlled version of any gate
  INV gate qubit          Inverse/dagger of a gate
  RESET qubit             Reset qubit to |0>
  SET_STATE |name>        Set qubit state (|0>, |1>, |+>, |->, |BELL>, |GHZ>)
  SET_STATE [a, b, ...]   Set explicit amplitudes (auto-normalized)
  SWEEP var s e [n]       Run circuit sweeping a variable
  NOISE type [p]          Set noise model (depolarizing, etc.)
  NOISE OFF               Disable noise
  EXPECT Z 0              Expectation value of Pauli operator
  DENSITY                 Show density matrix
  ENTROPY [qubits]        Entanglement entropy
  DECOMPOSE               Gate count breakdown
  EXPORT [file]           Export circuit as OpenQASM (not available in LOCC mode)
  CSV [file]              Export results as CSV (includes statevector if available)
  RAM                     Memory budget and parallelism estimates (3x overhead)
  BENCH [n1 n2 ...]       Benchmark qubit scaling (default: 4 8 12 16 20 24 28)
  PEEK addr               Read memory-mapped address
  USR(addr)               Execute SYS routine, return top measurement as integer
  WAIT addr,mask[,val,t]  Block until (PEEK(addr) AND mask) == val (default t=30s)
  CIRCUIT_DEF name s-e    Define gate macro from line range (see also DEF)
  INCLUDE file            Merge another .qb file
  DIR [path]              List .qb files
  CLEAR var               Remove a variable or array
  UNDO                    Undo last program edit
  BANK n                  Switch to program slot n (auto-saves current)
  BANK                    Show current slot and list used slots

INLINE CIRCUIT INSTRUCTIONS (in programs, results available after RUN)
  SAVE_EXPECT ZZ 0,1 -> v   Expectation value -> variable
  SAVE_PROBS 0,1 -> p       Probability snapshot -> array
  SAVE_AMPS 0,3 -> a        Specific amplitudes -> array
  MEAS qubit -> var          Mid-circuit measurement (LOCC mode)
  MEASURE_X/Y/Z qubit       Basis measurement

FLOW CONTROL (in programs)
  GOTO line               Jump to line
  GOSUB line / RETURN     Subroutine call with stack
  WHILE expr / WEND       Conditional loop
  IF expr THEN ... ELSE   Conditional (single-line)
  IF expr THEN / ... / END IF    Multi-line block
  IF ... ELSEIF ... ELSE  Chained conditions
  DO [WHILE|UNTIL] / LOOP  Pre/post-test loops
  SELECT CASE / CASE / END SELECT
  SUB name(args) / END SUB      Subroutine with scope
  FUNCTION name(args) / END FUNCTION  Function with return value
  END                     Stop execution
  PRINT expr              Output during run
  INPUT "prompt", var     Read user input
  DIM arr(size)           Declare array
  LET arr[i] = val        Array assignment

EXPRESSIONS
  PI, TAU, E, SQRT2, sin(), cos(), sqrt(), log(), etc.
  Comparisons: ==, !=, <, >, <=, >=, AND, OR, NOT
  Arrays: arr(i) or arr[i]
  Example: LET theta = PI/4 + asin(0.5)
"""

BANNER_ART = """\

 ██████╗ ██╗   ██╗██████╗  █████╗ ███████╗██╗ ██████╗
██╔═══██╗██║   ██║██╔══██╗██╔══██╗██╔════╝██║██╔════╝
██║   ██║██║   ██║██████╔╝███████║███████╗██║██║
██║▄▄ ██║██║   ██║██╔══██╗██╔══██║╚════██║██║██║
╚██████╔╝╚██████╔╝██████╔╝██║  ██║███████║██║╚██████╗
 ╚══▀▀═╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝

Quantum BASIC
{info_line}
{config_line}
Type HELP for commands, DEMO LIST for demos.
"""
