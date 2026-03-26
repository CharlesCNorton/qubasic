# QBASIC — Cure List

1. Decouple REPL I/O from the execution engine. Remove all `input()`/`print()` from execution paths; route through `self.io`.
2. Define a structured error model. Migrate all `print("?ERROR: ...")` patterns to raise `QBasicError` subclasses.
3. Replace regex-driven parsing with a structured parse step. Wire `_exec_line` to dispatch on typed `Stmt` objects instead of raw strings.
4. Remove repeated string splitting and re-tokenization during execution.
5. Separate parsing from execution completely. `_exec_line` must operate only on parsed objects.
6. Centralize execution state into `ExecContext`. Wire it through `build_circuit`, `_exec_line`, and all control-flow helpers.
7. Eliminate the dual variable system. Wire `Scope` into execution, replacing `self.variables`/`run_vars` pairs.
8. Make control-flow and statement-execution return types uniform.
9. Normalize the statement handler interface. Replace lambda-based `_stmt_handlers` with uniform callables.
10. Define and enforce a clear order of statement evaluation.
11. Remove side effects from parsing helpers.
12. Define a minimal internal instruction set.
13. Unify subroutine expansion with the execution model.
14. Replace implicit recursion limits with structured call tracking.
15. Formalize qubit addressing and register resolution.
16. Separate backend-specific logic from execution logic. Wire `QiskitBackend`/`LOCCRegBackend` into execution.
17. Unify LOCC and non-LOCC execution paths at the instruction level.
18. Make measurement semantics explicit and consistent.
19. Consolidate immediate execution and program execution.
20. Eliminate the dual gate-application paths.
21. Reduce mixin state coupling.
22. Remove the lambda-list rebuild in `_exec_control_flow`.
23. Fix `_substitute_vars` to use parsed representations.
24. Generate `cmd_help` output from the command registry.
25. Unify the array model.
26. Eliminate silent fallbacks and implicit behavior.
27. Add a validation phase before execution.
28. Ensure deterministic program execution ordering.
29. Cache transpiled Qiskit circuits.
30. Optimize LOCC SEND mode.
31. Mock the Qiskit simulation backend in the test suite.
