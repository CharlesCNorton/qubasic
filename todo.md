# QBASIC — Cure List

1. Complete parse-layer migration: handle all Stmt types in `_exec_line` without legacy regex fallback.
2. Make control-flow return types uniform across all handlers.
3. Remove remaining side effects from parsing helpers.
4. Define a minimal internal instruction set (typed opcodes).
5. Unify subroutine expansion with the execution model.
6. Wire `QiskitBackend`/`LOCCRegBackend` into execution (replace direct `qc.*` calls).
7. Unify LOCC and non-LOCC execution paths at the instruction level.
8. Eliminate the dual gate-application paths (`_apply_gate` vs `_locc_apply_gate`).
9. Make measurement semantics explicit and consistent.
10. Reduce mixin state coupling.
11. Add a validation phase before execution.
12. Ensure deterministic program execution ordering.
13. Optimize LOCC SEND mode.
14. Mock the Qiskit simulation backend in the test suite.
