"""QBASIC engine — constants, gate tables, numpy simulation."""

from __future__ import annotations

from enum import Enum, auto
import numpy as np
from qbasic_core.patterns import (
    RE_LINE_NUM, RE_DEF_SINGLE, RE_DEF_BEGIN, RE_REG_INDEX,
    RE_AT_REG, RE_AT_REG_LINE, RE_SEND, RE_SHARE, RE_MEAS, RE_RESET,
    RE_UNITARY, RE_DIM, RE_REDIM, RE_ERASE, RE_GET, RE_INPUT,
    RE_CTRL, RE_INV, RE_LET_ARRAY, RE_LET_VAR, RE_PRINT,
    RE_GOTO, RE_GOSUB, RE_FOR, RE_NEXT, RE_WHILE, RE_IF_THEN,
    RE_GOTO_GOSUB_TARGET, RE_MEASURE_BASIS, RE_SYNDROME,
    RE_DATA, RE_READ, RE_ON_GOTO, RE_ON_GOSUB,
    RE_SELECT_CASE, RE_CASE, RE_DO, RE_LOOP_STMT, RE_EXIT,
    RE_SUB, RE_END_SUB, RE_FUNCTION, RE_END_FUNCTION, RE_CALL,
    RE_LOCAL, RE_STATIC_DECL, RE_SHARED,
    RE_ON_ERROR, RE_RESUME, RE_ERROR_STMT, RE_ASSERT,
    RE_SWAP, RE_POKE, RE_SYS, RE_OPEN, RE_CLOSE,
    RE_PRINT_FILE, RE_INPUT_FILE, RE_LINE_INPUT, RE_OPTION_BASE,
    RE_IMPORT, RE_SAVE_EXPECT, RE_SAVE_PROBS, RE_SAVE_AMPS,
    RE_SET_STATE, RE_TYPE_BEGIN, RE_TYPE_FIELD, RE_END_TYPE, RE_DIM_TYPE,
    RE_CHAIN, RE_MERGE, RE_DEF_FN, RE_PRINT_USING,
    RE_COLOR, RE_LOCATE, RE_SCREEN, RE_LPRINT,
    RE_ON_MEASURE, RE_ON_TIMER, RE_DIM_MULTI, RE_LET_STR,
)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Limits and defaults
# ═══════════════════════════════════════════════════════════════════════

MAX_QUBITS = 32
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
# Realistic multiplier for Qiskit Aer statevector simulation:
#   1x  statevector itself (2^n complex128 = 2^n * 16 bytes)
#  ~1x  transpilation intermediates and circuit representation
#  ~1x  Aer internal copy during simulation and measurement collapse
# Conservative estimate; actual overhead varies by method and circuit depth.
OVERHEAD_FACTOR = 3.0
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

try:
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable
    from rich.panel import Panel as _RichPanel
    from rich.text import Text as _RichText
    _RICH = True
    _console = _RichConsole(highlight=False)
except ImportError:
    _RICH = False
    _console = None



def _estimate_gb(n_qubits: int) -> float:
    """Estimate realistic memory for one n-qubit statevector including overhead."""
    return (2 ** n_qubits) * 16 * OVERHEAD_FACTOR / 1e9


def _get_ram_gb() -> tuple[float, float] | None:
    """Return (total_gb, available_gb) or None if psutil is unavailable."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total / 1e9, mem.available / 1e9
    except ImportError:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Gate matrices, builders, and simulation primitives (re-exported from gates.py)
# ═══════════════════════════════════════════════════════════════════════

from qbasic_core.gates import (
    GATE_TABLE, GATE_ALIASES, _np_gate_matrix, _apply_gate_np,
    _measure_np, _sample_np, _sample_one_np, _GATE_BUILDERS,
    _MAT_CX, _MAT_CY, _MAT_CH, _MAT_CCX, _MAT_CSWAP,
    _MAT_H, _MAT_X, _MAT_Y, _MAT_Z, _MAT_S, _MAT_T,
    _MAT_SDG, _MAT_TDG, _MAT_SX, _MAT_ID, _MAT_CZ,
    _MAT_SWAP, _MAT_DCX, _MAT_ISWAP,
)

# ═══════════════════════════════════════════════════════════════════════
# LOCC engine and constants (re-exported from locc_engine.py)
# ═══════════════════════════════════════════════════════════════════════

from qbasic_core.locc_engine import (
    LOCCEngine,
    LOCC_MAX_JOINT_QUBITS, LOCC_MAX_SPLIT_QUBITS, LOCC_MAX_REGISTERS,
    LOCC_SEND_SHOT_CAP, LOCC_SEND_QUBIT_THRESHOLD,
)

