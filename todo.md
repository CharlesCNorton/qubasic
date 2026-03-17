1. PEEK(addr) — read qubit amplitude, probability, or Bloch coordinate depending on address range
2. POKE addr, value — write qubit state, set configuration register, or define noise parameter by address
3. Memory map: define address ranges for qubit state, QPU config, display control, noise parameters, and classical memory
4. Qubit addresses: each qubit gets a block of addresses (real amplitude, imag amplitude, probability, Bloch x, Bloch y, Bloch z)
5. Config addresses: shots, num_qubits, sim_method, sim_device, noise_type, noise_param all readable/writable by address
6. Status addresses: read-only registers for gate_count, circuit_depth, last_run_time, total_probability, entanglement_entropy
7. SYS addr — execute a built-in algorithm routine mapped to an address (Bell, GHZ, QFT, Grover, teleport, etc.)
8. SYS with parameter passing via designated memory locations (like C64 passing args through zero page)
9. USR(addr) — call a user-defined gate sequence as a function, return a value (measurement result)
10. WAIT addr, mask, value — block until a memory-mapped status register matches a condition
11. DATA / READ / RESTORE — classic BASIC data statements for embedding gate parameters, angle lists, circuit descriptions
12. ON expr GOTO line, line, line — computed branch on measurement outcome or variable value
13. ON expr GOSUB line, line, line — computed subroutine call
14. DEF FN name(x) = expr — single-line function definitions (classic BASIC style, complements existing multi-statement DEF)
15. Boot sequence: print system info (Python version, Qiskit version, available RAM, max qubits, backend, GPU availability)
16. Boot memory test: probe actual qubit capacity at startup (how many qubits fit in RAM)
17. SCREEN mode — switch display mode (0=text, 1=histogram, 2=statevector, 3=Bloch, 4=density matrix, 5=circuit diagram)
18. Persistent SCREEN mode: after RUN, automatically display in the selected mode instead of default histogram
19. COLOR foreground, background — set Rich terminal colors for output
20. CLS — clear screen
21. LOCATE row, col — cursor positioning for structured output
22. SPC(n) and TAB(n) in PRINT statements
23. PRINT USING format_string — formatted numeric output (angles, probabilities, amplitudes)
24. PRINT separator behavior: semicolon suppresses newline, comma advances to next tab stop
25. Multiple PRINT zones for tabular output of measurement results
26. INPUT with type validation and retry (current INPUT silently defaults to 0 on error)
27. LINE INPUT — read a full line including spaces
28. GET — single keypress input without enter (for interactive stepping)
29. TIMER — return elapsed time (for benchmarking inside programs)
30. RND(x) — random number generator accessible in expressions (quantum-seeded when available)
31. FRE(0) — return free memory in bytes (actual available RAM for simulation)
32. POS(0) — return current cursor column
33. STR$(n) and VAL(s) — string/number conversion functions
34. String variables (A$, name$) — currently only numeric variables exist
35. String concatenation with + operator
36. LEFT$(s,n), RIGHT$(s,n), MID$(s,n,len) — string functions
37. LEN(s) for strings, CHR$(n), ASC(c) — character functions
38. String comparison in IF/THEN conditions
39. INSTR(haystack, needle) — substring search
40. Bitwise operators: AND, OR, XOR, NOT operating on integers (for classical post-processing of measurement bitstrings)
41. Hex and binary literals: &HFF, &B10110 — natural for qubit state specification
42. HEX$(n), BIN$(n) — format numbers as hex/binary strings
43. Multi-dimensional arrays: DIM A(rows, cols) — for storing measurement result tables
44. REDIM — resize arrays
45. ERASE — delete a specific array
46. OPTION BASE 0/1 — array index base
47. SWAP var1, var2 — swap two variables
48. SELECT CASE / CASE / END SELECT — structured alternative to chained IF/THEN
49. DO / LOOP WHILE|UNTIL — post-test loops (complement existing WHILE/WEND)
50. EXIT FOR / EXIT WHILE / EXIT DO — early loop termination
51. SUB name(params) / END SUB — proper subroutines with local scope (beyond current DEF)
52. FUNCTION name(params) / END FUNCTION — functions that return values
53. LOCAL variables inside SUB/FUNCTION — variable scoping
54. STATIC variables inside SUB/FUNCTION — persist across calls
55. SHARED — declare a variable accessible from within SUB/FUNCTION
56. CALL name(args) — explicit subroutine invocation
57. Error handling: ON ERROR GOTO line — trap runtime errors
58. RESUME / RESUME NEXT — continue after error
59. ERR and ERL variables — error code and error line
60. ERROR n — raise a user-defined error
61. TRON / TROFF — trace on/off, print each line number as it executes (debug mode, extends existing STEP)
62. STOP — break into debugger at a specific line
63. CONT — continue execution after STOP
64. Breakpoints: settable on specific line numbers
65. Watch expressions: monitor a variable or PEEK address during stepping
66. OPEN file FOR INPUT/OUTPUT/APPEND — file handle I/O (beyond current SAVE/LOAD)
67. PRINT #n, data / INPUT #n, var — read/write to file handles
68. CLOSE #n — close file handle
69. EOF(n) — test end of file
70. Sequential and random access file modes
71. LPRINT — output to a log file or secondary stream
72. Chained program execution: CHAIN "program.qb" — load and run, optionally preserving variables
73. MERGE "program.qb" — merge lines from another file without clearing (stronger INCLUDE)
74. AUTO start, step — auto-generate line numbers while typing
75. EDIT line — jump to a specific line for editing
76. LIST start-end — already exists, but add LIST SUBS, LIST VARS, LIST ARRAYS as filtered views
77. DUMP — hex dump of the memory map (qubit states, config, status as a memory dump table)
78. MAP — print the full memory map with current values
79. MONITOR — enter a hex monitor mode for direct PEEK/POKE interaction (like C64 machine language monitor)
80. Immediate mode PEEK/POKE — usable outside programs at the REPL prompt
81. Batch mode: pipe a .qb file to stdin, collect results on stdout, exit automatically
82. Return exit code from batch mode (0 success, nonzero on error)
83. Quiet mode flag (--quiet) — suppress banner and progress, output only results
84. JSON output mode (--json) — machine-readable results for pipeline integration
85. Multiple programs in memory: numbered program slots (like BASIC 7.0 BANK)
86. Copy/move line ranges: COPY 100-200 TO 500, MOVE 100-200 TO 500
87. FIND "text" — search program lines for a string
88. REPLACE "old" WITH "new" — find and replace in program lines
89. Profile mode: after RUN, report time spent per line
90. Gate count per line in profile output
91. Cumulative gate depth tracking visible during STEP mode
92. Measurement statistics accumulator: run N trials, collect mean/stddev/histogram across runs
93. ASSERT condition — halt with error if condition fails (for self-testing programs)
94. Checksum/hash of program listing (for verifying type-in accuracy)
95. CATALOG — list all SYS routines with their addresses and descriptions
96. User-installable SYS routines: define a gate sequence and assign it a SYS address
97. Interrupt-like callbacks: ON MEASURE GOSUB — trigger a subroutine when measurement occurs
98. ON TIMER(n) GOSUB — periodic callback during long runs
99. PLAY — queue a tone/beep on measurement events (terminal bell or frequency via audio)
100. Configurable READY prompt (customize the ] prompt character/string)
