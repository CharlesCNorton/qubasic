"""The terminal's public surface as a runtime-checkable Protocol.

TerminalProtocol names the state and methods an embedding or a front end
(the web REPL, the Jupyter kernel, qubasic_core.api) relies on, so
``isinstance(obj, TerminalProtocol)`` tells whether an object can stand in
for a QBasicTerminal. The full contract between the mixins themselves, every
shared attribute and method, is HostState in qubasic_core/host.py, which
mypy checks.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TerminalProtocol(Protocol):
    """Attributes and methods that mixins expect on the host terminal."""

    # ── State ──────────────────────────────────────────────────────────

    program: dict[int, str]
    num_qubits: int
    shots: int
    subroutines: dict[str, Any]
    registers: OrderedDict[str, tuple[int, int]]
    variables: dict[str, Any]
    arrays: dict[str, list[float]]
    last_counts: dict[str, int] | None
    last_sv: np.ndarray | None
    last_circuit: Any | None
    step_mode: bool
    sim_method: str
    sim_device: str
    locc: Any | None
    locc_mode: bool

    _undo_stack: list[dict[int, str]]
    _gosub_stack: list[int]
    _custom_gates: dict[str, np.ndarray]
    _noise_model: Any | None
    _max_iterations: int
    _include_depth: int
    _parsed: dict[int, Any]
    _circuit_cache_key: Any | None
    _circuit_cache: Any | None
    io: Any  # IOPort

    # ── Methods mixins call on the host ────────────────────────────────

    def _get_parsed(self, line_num: int) -> Any: ...
    def eval_expr(self, expr: str) -> float: ...
    def _safe_eval(self, expr: str, extra_ns: dict[str, Any] | None = None) -> Any: ...
    def _eval_with_vars(self, expr: str, run_vars: dict[str, Any]) -> float: ...
    def _eval_condition(self, cond: str, run_vars: dict[str, Any]) -> bool: ...
    def _gate_info(self, name: str) -> tuple[int, int] | None: ...
    def _resolve_qubit(self, arg: str) -> int: ...
    def _expand_statement(self, stmt: str) -> list[str]: ...
    def _tokenize_gate(self, stmt: str) -> list[str]: ...
    def _parse_syndrome(self, stmt: str, run_vars: dict[str, Any]) -> tuple[str, list[int], str] | None: ...
    def _split_colon_stmts(self, stmt: str) -> list[str]: ...
    def _print_statevector(self, sv: np.ndarray, n_qubits: int | None = None) -> None: ...
    def _print_bloch_single(self, sv: np.ndarray, qubit: int, n_qubits: int | None = None) -> None: ...
    def print_histogram(self, counts: dict[str, int]) -> None: ...
    def cmd_new(self, *, silent: bool = False) -> None: ...
    def cmd_run(self) -> None: ...
    def cmd_list(self, rest: str = '') -> None: ...
    def cmd_locc(self, rest: str) -> None: ...
    def process(self, line: str) -> None: ...
