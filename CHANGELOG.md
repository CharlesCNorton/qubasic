# Changelog

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
