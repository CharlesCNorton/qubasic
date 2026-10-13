"""QUBASIC core: limits and defaults, execution outcomes, the error
hierarchy, the I/O port protocol, the protocols the mixins and backends
satisfy, the execution context and the progress spinner.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

import numpy as np

try:
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable
    _RICH = True
    _console: Any = _RichConsole(highlight=False)
except ImportError:
    _RICH = False
    _RichTable = None  # type: ignore[assignment,misc]
    _console = None


# ---------------------------------------------------------------------------
# QUBASIC engine constants: limits and defaults, the exec-result sentinels,
# and the optional rich console.
# ---------------------------------------------------------------------------

# ═══════════════════════════════════════════════════════════════════════
# Limits and defaults
# ═══════════════════════════════════════════════════════════════════════

MAX_QUBITS = 32
# Per-method qubit ceilings. The 32-qubit wall is a statevector memory limit
# only: the stabilizer tableau is polynomial (thousands of qubits), MPS is
# bounded by entanglement rather than width, and `automatic` resolves to one
# of those at scale.
METHOD_MAX_QUBITS = {
    'automatic': 1024,
    'statevector': 32,
    'density_matrix': 16,
    'stabilizer': 4096,
    'matrix_product_state': 1024,
    'extended_stabilizer': 63,
    'unitary': 16,
    'superop': 8,
}
DEFAULT_QUBITS = 4
DEFAULT_SHOTS = 1024
MAX_UNDO_STACK = 50
MAX_LOOP_ITERATIONS = 100_000
MAX_HISTOGRAM_STATES = 32
MAX_DISPLAY_AMPLITUDES = 64
MAX_BLOCH_DISPLAY = 8
HISTOGRAM_BAR_WIDTH = 35
AMPLITUDE_THRESHOLD = 1e-8
MAX_INCLUDE_DEPTH = 8
RAM_BUDGET_FRACTION = 0.8

# ═══════════════════════════════════════════════════════════════════════
# Exec-result sentinels
# ═══════════════════════════════════════════════════════════════════════
# _exec_line and helpers return one of:
#   ExecResult.ADVANCE — advance instruction pointer by one
#   ExecResult.END     — stop execution
#   int                — jump to that instruction-pointer index

class ExecResult(Enum):
    """Instruction execution result sentinels."""
    ADVANCE = auto()
    END = auto()


# The return type of _exec_line and helpers: ADVANCE, END, or a jump target ip.
ExecOutcome = ExecResult | int

# ═══════════════════════════════════════════════════════════════════════
# Optional rich-terminal packages
# ═══════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# QUBASIC structured error hierarchy.
# ---------------------------------------------------------------------------

class QBasicError(Exception):
    """Base for all QUBASIC errors."""

    def __init__(self, message: str, *, code: int = 0, line: int | None = None) -> None:
        self.message = message
        self.code = code
        self.line = line
        super().__init__(message)


class QBasicSyntaxError(QBasicError):
    """Parse-time or syntax error."""


class QBasicRuntimeError(QBasicError):
    """Execution-time error."""


class QBasicBuildError(QBasicError):
    """Circuit/program build error."""


class QBasicRangeError(QBasicError):
    """Value out of range."""


class QBasicIOError(QBasicError):
    """File or I/O error."""


class QBasicUndefinedError(QBasicError):
    """Reference to undefined name."""


class MidCircuitRead(ValueError):
    """A mid-circuit measurement bit was read as a number during a circuit
    build; the program runs live instead."""


# ---------------------------------------------------------------------------
# QUBASIC I/O protocol — decouples execution engine from print/input.
# ---------------------------------------------------------------------------

@runtime_checkable
class IOPort(Protocol):
    """Abstract I/O port for execution output and user input."""

    def write(self, text: str) -> None: ...
    def writeln(self, text: str) -> None: ...
    def read_line(self, prompt: str) -> str: ...


class NullIO:
    """Discards output and reads end-of-input: the IO of an inner run whose
    output is not shown (COMPARE, PLOT, ANIMATE, MINIMIZE, STATS)."""

    def write(self, text: str) -> None:
        pass

    def writeln(self, text: str) -> None:
        pass

    def read_line(self, prompt: str) -> str:
        return ''


class StdIOPort:
    """Default I/O: stdout/stdin."""

    def write(self, text: str) -> None:
        try:
            print(text, end='', flush=True)
        except UnicodeEncodeError:
            print(text.encode('ascii', 'replace').decode('ascii'), end='', flush=True)

    def writeln(self, text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', 'replace').decode('ascii'))

    def read_line(self, prompt: str) -> str:
        return input(prompt)


# ---------------------------------------------------------------------------
# The terminal's public surface as a runtime-checkable Protocol.
#
# TerminalProtocol names the state and methods an embedding or a front end
# (the web REPL, the Jupyter kernel, qubasic_core.api) relies on, so
# ``isinstance(obj, TerminalProtocol)`` tells whether an object can stand in
# for a QBasicTerminal. The full contract between the mixins themselves, every
# shared attribute and method, is HostState in qubasic_core/host.py, which
# mypy checks.
# ---------------------------------------------------------------------------

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
    def _eval_with_vars(self, expr: str, run_vars: MutableMapping[str, Any]) -> float: ...
    def _eval_condition(self, cond: str, run_vars: MutableMapping[str, Any]) -> bool: ...
    def _gate_info(self, name: str) -> tuple[int, int] | None: ...
    def _resolve_qubit(self, arg: str) -> int: ...
    def _expand_statement(self, stmt: str) -> list[str]: ...
    def _tokenize_gate(self, stmt: str) -> list[str]: ...
    def _parse_syndrome(self, stmt: str, run_vars: MutableMapping[str, Any]) -> tuple[str, list[int], str] | None: ...
    def _split_colon_stmts(self, stmt: str) -> list[str]: ...
    def _print_statevector(self, sv: np.ndarray, n_qubits: int | None = None) -> None: ...
    def _print_bloch_single(self, sv: np.ndarray, qubit: int, n_qubits: int | None = None) -> None: ...
    def print_histogram(self, counts: dict[str, int]) -> None: ...
    def cmd_new(self, *, silent: bool = False) -> None: ...
    def cmd_run(self) -> None: ...
    def cmd_list(self, rest: str = '') -> None: ...
    def cmd_locc(self, rest: str) -> None: ...
    def process(self, line: str) -> None: ...


class QuantumBackend(Protocol):
    """Abstract quantum backend for gate application."""

    def apply_gate(self, gate_name: str, params: tuple[float, ...],
                   qubits: list[int]) -> None: ...
    def barrier(self) -> None: ...
    def reset(self, qubit: int) -> None: ...


class CircuitBackend(Protocol):
    """What the circuit builder needs of its backend: the circuit it adds to,
    gate application, barriers and controlled operations (QiskitBackend, and
    LiveBackend for a live shot)."""

    @property
    def qc(self) -> Any: ...

    def apply_gate(self, gate_name: str, params: tuple[float, ...],
                   qubits: list[int]) -> None: ...

    def barrier(self) -> None: ...

    def append_controlled(self, gate: Any, qubits: list[int]) -> None: ...


# ---------------------------------------------------------------------------
# QUBASIC execution context — unified mutable state for program execution.
# ---------------------------------------------------------------------------

@dataclass
class ExecContext:
    """Mutable execution state passed through the entire call tree."""

    sorted_lines: Any  # the CompiledProgram (indexing gives line numbers)
    ip: int
    run_vars: MutableMapping[str, Any]  # a dict, or a Scope over the variables
    loop_stack: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 100_000

    # Qiskit circuit-build path (None in LOCC path)
    qc: Any = None

    # Backend abstraction (QiskitBackend or LOCCRegBackend)
    backend: Any = None


# ---------------------------------------------------------------------------
# Progress indication for long runs: spinner frames drawn from quantum states.
# ---------------------------------------------------------------------------

_QUANTUM_SPINNER = [
    '|0\u27E9', '|+\u27E9', '|1\u27E9', '|-\u27E9',
    '|i\u27E9', '|\u03C8\u27E9', '|0\u27E9', '|\u03D5\u27E9',
]


def quantum_spin(step: int) -> str:
    """Return the spinner frame for the given step."""
    return _QUANTUM_SPINNER[step % len(_QUANTUM_SPINNER)]
