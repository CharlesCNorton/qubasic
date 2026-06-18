"""QUBASIC execution context — unified mutable state for program execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecContext:
    """Mutable execution state passed through the entire call tree."""

    sorted_lines: list[int]
    ip: int
    run_vars: Any  # dict[str, Any] or a Scope wrapper
    loop_stack: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 100_000

    # Qiskit circuit-build path (None in LOCC path)
    qc: Any = None

    # Backend abstraction (QiskitBackend or LOCCRegBackend)
    backend: Any = None
