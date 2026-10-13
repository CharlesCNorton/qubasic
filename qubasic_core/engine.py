"""QUBASIC core: limits and defaults, execution outcomes, the error
hierarchy, the I/O port protocol, the protocols the mixins and backends
satisfy, the execution context and the progress spinner.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

import numpy as np

from qubasic_core.host import CoreHost

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
MAX_STABILIZERS_SHOWN = 32
HISTOGRAM_BAR_WIDTH = 35
AMPLITUDE_THRESHOLD = 1e-8
MAX_INCLUDE_DEPTH = 8
# Instructions one circuit build may produce, whatever the statement cap.
MAX_CIRCUIT_INSTRUCTIONS = 1_000_000
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

@dataclass
class RunReport:
    """What a RUN leaves for the RUN command to show after the summary line:
    the counts to draw as a histogram (readout mitigation follows it), whether
    DISPLAY and the LQUBITS trailer apply, the note to print when there is
    nothing to draw, and whether the run was long enough to ring the bell."""
    counts: dict[str, int] | None = None
    display: bool = False
    logical: bool = False
    note: str | None = None
    bell: bool = False


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
# shared attribute and method layer by layer, is the hosts in
# qubasic_core/host.py, which mypy checks.
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

    editor: Any          # EditorState: undo, banks, AUTO, INCLUDE
    noise: Any           # NoiseState: the active noise model
    _gosub_stack: list[int]
    _custom_gates: dict[str, np.ndarray]
    _max_iterations: int
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


# ---------------------------------------------------------------------------
# Services every layer's mixins use: discarding a block's output, and the
# rules for the file paths programs and commands name.
# ---------------------------------------------------------------------------

class ServicesMixin(CoreHost):
    """Output suppression, path checking, and what running on a terminal
    changes, for every layer."""

    @staticmethod
    def _on_terminal() -> bool:
        """Whether output goes to an interactive terminal."""
        return bool(sys.stdout and sys.stdout.isatty())

    def _loop_limit(self) -> int:
        """How many statements a run may execute: the configured limit
        (_max_iterations, POKE $D006) off a terminal and in agent mode, where
        nobody can interrupt a run, and no limit at an interactive terminal,
        where Ctrl+C stops a program as it does in QBASIC."""
        if self.agent_mode or not self._on_terminal():
            return self._max_iterations
        return sys.maxsize

    @contextmanager
    def _silenced(self) -> Iterator[None]:
        """Run a block with its output discarded (the inner runs of COMPARE,
        PLOT, ANIMATE, MINIMIZE, GRADIENT and STATS)."""
        old, self.io = self.io, NullIO()
        try:
            yield
        finally:
            self.io = old

    @staticmethod
    def _sanitize_path(path: str) -> str:
        """Path sanitization: reject null bytes, control characters, and traversal.

        Blocks directory-traversal sequences (..) and absolute paths that
        escape the working directory.  For a local REPL the user already has
        shell access, but this prevents accidental overwrites of system files
        and hardens INCLUDE chains against path confusion.
        """
        if not path or not path.strip():
            raise ValueError("Empty path")
        if '\x00' in path:
            raise ValueError("Path contains null bytes")
        if any(ord(c) < 32 for c in path):
            raise ValueError("Path contains control characters")
        path = path.strip()
        # Reject traversal sequences — the primary injection vector
        normalized = os.path.normpath(path)
        if '..' in normalized.split(os.sep):
            raise ValueError("Path traversal (..) is not allowed")
        # Reject absolute paths (Unix / or Windows drive letters)
        if os.path.isabs(path):
            raise ValueError("Absolute paths are not allowed")
        # Catch Windows drive letters on non-Windows hosts (e.g. CI on Linux)
        import re as _re
        if _re.match(r'^[A-Za-z]:[/\\]', path):
            raise ValueError("Absolute paths are not allowed")
        # Reject UNC paths (\\server\share) and extended-length paths (\\?\)
        if path.startswith('\\\\'):
            raise ValueError("UNC/extended paths are not allowed")
        return path
