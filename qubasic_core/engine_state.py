"""QUBASIC engine state — standalone execution state container.

Extracted from QBasicTerminal to break the god-object. Engine holds all
program state, variables, arrays, execution configuration, and caches.
QBasicTerminal composes Engine and adds the REPL shell and command dispatch.

Tests and headless callers can use Engine directly without the REPL.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

from qubasic_core.engine import (
    DEFAULT_QUBITS, DEFAULT_SHOTS, MAX_LOOP_ITERATIONS, MAX_UNDO_STACK,
)
from qubasic_core.io_protocol import StdIOPort, IOPort
from qubasic_core.parser import parse_stmt


class Engine:
    """Standalone execution state container.

    Holds all program state that was previously scattered across
    QBasicTerminal's __init__ and 16 mixin _init_* methods.
    """

    def __init__(self, *, io: IOPort | None = None) -> None:
        # Program
        self.program: dict[int, str] = {}
        self._parsed: dict[int, Any] = {}
        self._undo_stack: list[dict[int, str]] = []

        # Configuration
        self.num_qubits: int = DEFAULT_QUBITS
        self.shots: int = DEFAULT_SHOTS
        self.sim_method: str = 'automatic'
        self.sim_device: str = 'CPU'
        self._noise_model: Any = None
        self._noise_depol_p: float = 0.0
        self._seed: int | None = None
        self._max_iterations: int = MAX_LOOP_ITERATIONS
        self._include_depth: int = 0

        # Variables and arrays
        self.variables: dict[str, Any] = {}
        self.arrays: dict[str, Any] = {}
        self._array_dims: dict[str, list[int]] = {}

        # Subroutines and registers
        self.subroutines: dict[str, Any] = {}
        self.registers: OrderedDict[str, tuple[int, int]] = OrderedDict()

        # Custom gates
        self._custom_gates: dict[str, Any] = {}

        # Execution state
        self._gosub_stack: list[int] = []
        self.step_mode: bool = False
        self.last_counts: dict[str, int] | None = None
        self.last_sv: Any = None
        self.last_circuit: Any = None
        self._circuit_cache_key: Any = None
        self._circuit_cache: Any = None

        # LOCC
        self.locc: Any = None
        self.locc_mode: bool = False

        # I/O
        self.io: IOPort = io or StdIOPort()

        # Security
        self.agent_mode: bool = False
        self._include_stack: list[str] = []

        # User-defined types
        self._user_types: dict[str, list[tuple[str, str]]] = {}
        self._pending_type: dict | None = None

        # Timing
        self._start_time: float = time.time()

    def _get_parsed(self, line_num: int) -> Any:
        """Get parsed Stmt for a line, lazily parsing if needed.

        Re-parses if the program text has changed since the last parse.
        """
        raw = self.program.get(line_num, '')
        p = self._parsed.get(line_num)
        if p is None or p.raw != raw:
            p = parse_stmt(raw)
            self._parsed[line_num] = p
        return p

    def clear(self) -> None:
        """Reset all state (equivalent to cmd_new)."""
        self.program.clear()
        self._parsed.clear()
        self.subroutines.clear()
        self.registers.clear()
        self.variables.clear()
        self.arrays.clear()
        self._array_dims.clear()
        self.last_counts = None
        self.last_sv = None
        self.last_circuit = None
        self._circuit_cache_key = None
        self._circuit_cache = None
