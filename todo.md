# QBASIC — Cure List

1. Complete parse-layer migration: handle remaining Stmt types (FOR/NEXT, WHILE/WEND, IF/THEN, LET, PRINT) in the fast path.
2. Remove remaining side effects from parsing helpers.
3. Define a minimal internal instruction set (typed opcodes).
4. Unify subroutine expansion with the execution model.
5. Wire `QiskitBackend`/`LOCCRegBackend` into execution (replace direct `qc.*` calls).
6. Unify LOCC and non-LOCC execution paths at the instruction level.
7. Eliminate the dual gate-application paths (`_apply_gate` vs `_locc_apply_gate`).
8. Make measurement semantics explicit and consistent.
9. Reduce mixin state coupling.
10. Optimize LOCC SEND mode.
