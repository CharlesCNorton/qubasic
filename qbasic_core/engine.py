"""QBASIC engine — constants, gate tables, numpy simulation, LOCCEngine."""

from __future__ import annotations

import re
from enum import Enum, auto
from typing import Callable

import numpy as np

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
LOCC_MAX_JOINT_QUBITS = 33
LOCC_MAX_SPLIT_QUBITS = 33
LOCC_MAX_REGISTERS = 26
LOCC_SEND_SHOT_CAP = 100
LOCC_SEND_QUBIT_THRESHOLD = 20
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
# Pre-compiled regexes
# ═══════════════════════════════════════════════════════════════════════

RE_LINE_NUM = re.compile(r'^(\d+)\s*(.*)')
RE_DEF_SINGLE = re.compile(r'(\w+)(?:\(([^)]*)\))?\s*=\s*(.*)')
RE_DEF_BEGIN = re.compile(r'DEF\s+BEGIN\s+(\w+)(?:\(([^)]*)\))?', re.IGNORECASE)
RE_REG_INDEX = re.compile(r'(\w+)\[(\d+)\]')
RE_AT_REG = re.compile(r'@([A-Z])\s+', re.IGNORECASE)
RE_AT_REG_LINE = re.compile(r'@([A-Z])\s+(.*)', re.IGNORECASE)
RE_SEND = re.compile(r'SEND\s+([A-Z])\s+(\S+)\s*->\s*(\w+)', re.IGNORECASE)
RE_SHARE = re.compile(r'SHARE\s+([A-Z])\s+(\d+)\s*,?\s*([A-Z])\s+(\d+)', re.IGNORECASE)
RE_MEAS = re.compile(r'MEAS\s+(\S+)\s*->\s*(\w+)', re.IGNORECASE)
RE_RESET = re.compile(r'RESET\s+(\S+)', re.IGNORECASE)
RE_UNITARY = re.compile(r'UNITARY\s+(\w+)\s*=\s*(\[.+\])', re.IGNORECASE)
RE_DIM = re.compile(r'DIM\s+(\w+)\((\d+)\)', re.IGNORECASE)
RE_REDIM = re.compile(r'REDIM\s+(\w+)\((\d+)\)', re.IGNORECASE)
RE_ERASE = re.compile(r'ERASE\s+(\w+)', re.IGNORECASE)
RE_GET = re.compile(r'GET\s+(\w+\$?)', re.IGNORECASE)
RE_INPUT = re.compile(r'INPUT\s+(?:"([^"]*)"\s*,\s*)?(\w+)', re.IGNORECASE)
RE_CTRL = re.compile(r'CTRL\s+(\w+)\s+(.*)', re.IGNORECASE)
RE_INV = re.compile(r'INV\s+(\w+)\s+(.*)', re.IGNORECASE)
RE_LET_ARRAY = re.compile(r'LET\s+(\w+)\((.+?)\)\s*=\s*(.*)', re.IGNORECASE)
RE_LET_VAR = re.compile(r'LET\s+(\w+)\s*=\s*(.*)', re.IGNORECASE)
RE_PRINT = re.compile(r'PRINT\s+(.*)', re.IGNORECASE)
RE_GOTO = re.compile(r'GOTO\s+(\d+)\s*$', re.IGNORECASE)
RE_GOSUB = re.compile(r'GOSUB\s+(\d+)\s*$', re.IGNORECASE)
RE_FOR = re.compile(
    r'FOR\s+(\w+)\s*=\s*(.+?)\s+TO\s+(.+?)(?:\s+STEP\s+(.+))?\s*$', re.IGNORECASE)
RE_NEXT = re.compile(r'NEXT\s+(\w+)\s*$', re.IGNORECASE)
RE_WHILE = re.compile(r'WHILE\s+(.+)$', re.IGNORECASE)
RE_IF_THEN = re.compile(
    r'IF\s+(.+?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*))?$', re.IGNORECASE)
RE_GOTO_GOSUB_TARGET = re.compile(r'(GOTO|GOSUB)\s+(\d+)', re.IGNORECASE)
RE_MEASURE_BASIS = re.compile(
    r'MEASURE_(X|Y|Z)\s+(\S+)', re.IGNORECASE)
RE_SYNDROME = re.compile(
    r'SYNDROME\s+(.*)', re.IGNORECASE)

# ── Classic BASIC, memory, SUB/FUNCTION, debug ──────────────────────

RE_DATA = re.compile(r'DATA\s+(.*)', re.IGNORECASE)
RE_READ = re.compile(r'READ\s+(.*)', re.IGNORECASE)
RE_ON_GOTO = re.compile(r'ON\s+(.+?)\s+GOTO\s+([\d\s,]+)', re.IGNORECASE)
RE_ON_GOSUB = re.compile(r'ON\s+(.+?)\s+GOSUB\s+([\d\s,]+)', re.IGNORECASE)
RE_SELECT_CASE = re.compile(r'SELECT\s+CASE\s+(.*)', re.IGNORECASE)
RE_CASE = re.compile(r'CASE\s+(.*)', re.IGNORECASE)
RE_DO = re.compile(r'DO(?:\s+(WHILE|UNTIL)\s+(.+))?\s*$', re.IGNORECASE)
RE_LOOP_STMT = re.compile(r'LOOP(?:\s+(WHILE|UNTIL)\s+(.+))?\s*$', re.IGNORECASE)
RE_EXIT = re.compile(r'EXIT\s+(FOR|WHILE|DO|SUB|FUNCTION)\s*$', re.IGNORECASE)
RE_SUB = re.compile(r'SUB\s+(\w+)(?:\(([^)]*)\))?\s*$', re.IGNORECASE)
RE_END_SUB = re.compile(r'END\s+SUB\s*$', re.IGNORECASE)
RE_FUNCTION = re.compile(r'FUNCTION\s+(\w+)(?:\(([^)]*)\))?\s*$', re.IGNORECASE)
RE_END_FUNCTION = re.compile(r'END\s+FUNCTION\s*$', re.IGNORECASE)
RE_CALL = re.compile(r'CALL\s+(\w+)(?:\(([^)]*)\))?\s*$', re.IGNORECASE)
RE_LOCAL = re.compile(r'LOCAL\s+(.*)', re.IGNORECASE)
RE_STATIC_DECL = re.compile(r'STATIC\s+(.*)', re.IGNORECASE)
RE_SHARED = re.compile(r'SHARED\s+(.*)', re.IGNORECASE)
RE_ON_ERROR = re.compile(r'ON\s+ERROR\s+GOTO\s+(\d+)', re.IGNORECASE)
RE_RESUME = re.compile(r'RESUME(?:\s+(.+))?\s*$', re.IGNORECASE)
RE_ERROR_STMT = re.compile(r'ERROR\s+(\d+)', re.IGNORECASE)
RE_ASSERT = re.compile(r'ASSERT\s+(.*)', re.IGNORECASE)
RE_SWAP = re.compile(r'SWAP\s+(\w+\$?)\s*,\s*(\w+\$?)', re.IGNORECASE)
RE_POKE = re.compile(r'POKE\s+(.+?)\s*,\s*(.+)', re.IGNORECASE)
RE_SYS = re.compile(r'SYS\s+(.+)', re.IGNORECASE)
RE_OPEN = re.compile(
    r'OPEN\s+"?([^"]+)"?\s+FOR\s+(INPUT|OUTPUT|APPEND|RANDOM)\s+AS\s+#?(\d+)',
    re.IGNORECASE)
RE_CLOSE = re.compile(r'CLOSE\s+#?(\d+)', re.IGNORECASE)
RE_PRINT_FILE = re.compile(r'PRINT\s+#(\d+)\s*,\s*(.*)', re.IGNORECASE)
RE_INPUT_FILE = re.compile(r'INPUT\s+#(\d+)\s*,\s*(\w+\$?)', re.IGNORECASE)
RE_LINE_INPUT = re.compile(
    r'LINE\s+INPUT\s+(?:"([^"]*)"\s*,\s*)?(\w+\$?)', re.IGNORECASE)
RE_OPTION_BASE = re.compile(r'OPTION\s+BASE\s+([01])', re.IGNORECASE)
RE_CHAIN = re.compile(r'CHAIN\s+"?([^"]+)"?', re.IGNORECASE)
RE_MERGE = re.compile(r'MERGE\s+"?([^"]+)"?', re.IGNORECASE)
RE_DEF_FN = re.compile(
    r'DEF\s+FN\s*(\w+)\s*\(([^)]*)\)\s*=\s*(.*)', re.IGNORECASE)
RE_PRINT_USING = re.compile(
    r'PRINT\s+USING\s+"([^"]+)"\s*;\s*(.*)', re.IGNORECASE)
RE_COLOR = re.compile(r'COLOR\s+(\w+)(?:\s*,\s*(\w+))?', re.IGNORECASE)
RE_LOCATE = re.compile(r'LOCATE\s+(\d+)\s*,\s*(\d+)', re.IGNORECASE)
RE_SCREEN = re.compile(r'SCREEN\s+(\d+)', re.IGNORECASE)
RE_LPRINT = re.compile(r'LPRINT\s+(.*)', re.IGNORECASE)
RE_ON_MEASURE = re.compile(r'ON\s+MEASURE\s+GOSUB\s+(\d+)', re.IGNORECASE)
RE_ON_TIMER = re.compile(r'ON\s+TIMER\s*\((\d+)\)\s+GOSUB\s+(\d+)', re.IGNORECASE)
RE_DIM_MULTI = re.compile(r'DIM\s+(\w+)\((\d+(?:\s*,\s*\d+)*)\)', re.IGNORECASE)
RE_LET_STR = re.compile(r'LET\s+(\w+\$)\s*=\s*(.*)', re.IGNORECASE)

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


# Gate tables: name -> (num_params, num_qubits)
GATE_TABLE = {
    # 0-param, 1-qubit
    'H':    (0, 1), 'X':    (0, 1), 'Y':    (0, 1), 'Z':    (0, 1),
    'S':    (0, 1), 'T':    (0, 1), 'SDG':  (0, 1), 'TDG':  (0, 1),
    'SX':   (0, 1), 'ID':   (0, 1),
    # 1-param, 1-qubit
    'RX':   (1, 1), 'RY':   (1, 1), 'RZ':   (1, 1), 'P':    (1, 1),
    # 3-param, 1-qubit
    'U':    (3, 1),
    # 0-param, 2-qubit
    'CX':   (0, 2), 'CNOT': (0, 2), 'CZ':   (0, 2), 'CY':   (0, 2),
    'CH':   (0, 2), 'SWAP': (0, 2), 'DCX':  (0, 2), 'ISWAP':(0, 2),
    # 1-param, 2-qubit
    'CRX':  (1, 2), 'CRY':  (1, 2), 'CRZ':  (1, 2), 'CP':   (1, 2),
    'RXX':  (1, 2), 'RYY':  (1, 2), 'RZZ':  (1, 2),
    # 0-param, 3-qubit
    'CCX':  (0, 3), 'TOFFOLI': (0, 3), 'CSWAP': (0, 3), 'FREDKIN': (0, 3),
}

# Aliases
GATE_ALIASES = {
    'CNOT': 'CX', 'TOFFOLI': 'CCX', 'FREDKIN': 'CSWAP',
}


# ═══════════════════════════════════════════════════════════════════════
# LOCC Engine — Numpy-based dual-register quantum simulation
# ═══════════════════════════════════════════════════════════════════════

_MAT_CY = np.eye(4, dtype=complex)
_MAT_CY[2,2], _MAT_CY[2,3], _MAT_CY[3,2], _MAT_CY[3,3] = 0, -1j, 1j, 0

_MAT_CH = np.eye(4, dtype=complex)
_h = 1/np.sqrt(2)
_MAT_CH[2,2], _MAT_CH[2,3], _MAT_CH[3,2], _MAT_CH[3,3] = _h, _h, _h, -_h

def _mat_crx(p):
    t = p[0]
    c, s = np.cos(t/2), np.sin(t/2)
    m = np.eye(4, dtype=complex)
    m[2,2], m[2,3] = c, -1j*s
    m[3,2], m[3,3] = -1j*s, c
    return m

def _mat_cry(p):
    t = p[0]
    c, s = np.cos(t/2), np.sin(t/2)
    m = np.eye(4, dtype=complex)
    m[2,2], m[2,3] = c, -s
    m[3,2], m[3,3] = s, c
    return m

def _mat_crz(p):
    t = p[0]
    m = np.eye(4, dtype=complex)
    m[2,2], m[3,3] = np.exp(-1j*t/2), np.exp(1j*t/2)
    return m

def _mat_rxx(p):
    t = p[0]
    ct, st = np.cos(t/2), np.sin(t/2)
    m = np.zeros((4,4), dtype=complex)
    m[0,0] = m[1,1] = m[2,2] = m[3,3] = ct
    m[0,3] = m[3,0] = m[1,2] = m[2,1] = -1j*st
    return m

def _mat_ryy(p):
    # RYY(t) = exp(-i * t/2 * YY).  The off-diagonal signs are asymmetric
    # because Y = [[0,-i],[i,0]] yields YY with sign pattern:
    #   |00>-|11> coupling: +i sin(t/2)   (anti-diagonal corners)
    #   |01>-|10> coupling: -i sin(t/2)   (inner off-diagonal)
    t = p[0]
    ct, st = np.cos(t/2), np.sin(t/2)
    m = np.zeros((4,4), dtype=complex)
    m[0,0] = m[1,1] = m[2,2] = m[3,3] = ct
    m[0,3] = m[3,0] = 1j*st
    m[1,2] = m[2,1] = -1j*st
    return m

_MAT_CCX = np.eye(8, dtype=complex)
_MAT_CCX[6,6], _MAT_CCX[6,7], _MAT_CCX[7,6], _MAT_CCX[7,7] = 0, 1, 1, 0

_MAT_CSWAP = np.eye(8, dtype=complex)
_MAT_CSWAP[5,5], _MAT_CSWAP[5,6], _MAT_CSWAP[6,5], _MAT_CSWAP[6,6] = 0, 1, 1, 0

# Pre-computed constant gate matrices (allocated once, not per call).
_MAT_H    = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
_MAT_X    = np.array([[0,1],[1,0]], dtype=complex)
_MAT_Y    = np.array([[0,-1j],[1j,0]], dtype=complex)
_MAT_Z    = np.array([[1,0],[0,-1]], dtype=complex)
_MAT_S    = np.array([[1,0],[0,1j]], dtype=complex)
_MAT_T    = np.array([[1,0],[0,np.exp(1j*np.pi/4)]], dtype=complex)
_MAT_SDG  = np.array([[1,0],[0,-1j]], dtype=complex)
_MAT_TDG  = np.array([[1,0],[0,np.exp(-1j*np.pi/4)]], dtype=complex)
_MAT_SX   = np.array([[1+1j,1-1j],[1-1j,1+1j]], dtype=complex) / 2
_MAT_ID   = np.eye(2, dtype=complex)
_MAT_CX   = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
_MAT_CZ   = np.diag([1, 1, 1, -1]).astype(complex)
_MAT_SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
_MAT_DCX  = np.array([[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]], dtype=complex)
_MAT_ISWAP = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)

# Gate matrix registry: name -> callable(params) -> np.ndarray
# Adding a new gate only requires an entry here and in GATE_TABLE.
# 0-param gates return pre-computed constants (no allocation per call).
_GATE_BUILDERS = {
    # 0-param, 1-qubit
    'H':     lambda p: _MAT_H,
    'X':     lambda p: _MAT_X,
    'Y':     lambda p: _MAT_Y,
    'Z':     lambda p: _MAT_Z,
    'S':     lambda p: _MAT_S,
    'T':     lambda p: _MAT_T,
    'SDG':   lambda p: _MAT_SDG,
    'TDG':   lambda p: _MAT_TDG,
    'SX':    lambda p: _MAT_SX,
    'ID':    lambda p: _MAT_ID,
    # 1-param, 1-qubit
    'RX':    lambda p: np.array([[np.cos(p[0]/2), -1j*np.sin(p[0]/2)],
                                 [-1j*np.sin(p[0]/2), np.cos(p[0]/2)]], dtype=complex),
    'RY':    lambda p: np.array([[np.cos(p[0]/2), -np.sin(p[0]/2)],
                                 [np.sin(p[0]/2), np.cos(p[0]/2)]], dtype=complex),
    'RZ':    lambda p: np.array([[np.exp(-1j*p[0]/2), 0],
                                 [0, np.exp(1j*p[0]/2)]], dtype=complex),
    'P':     lambda p: np.array([[1, 0], [0, np.exp(1j*p[0])]], dtype=complex),
    # 3-param, 1-qubit
    'U':     lambda p: np.array([
                 [np.cos(p[0]/2), -np.exp(1j*p[2])*np.sin(p[0]/2)],
                 [np.exp(1j*p[1])*np.sin(p[0]/2),
                  np.exp(1j*(p[1]+p[2]))*np.cos(p[0]/2)]], dtype=complex),
    # 0-param, 2-qubit
    'CX':    lambda p: _MAT_CX,
    'CZ':    lambda p: _MAT_CZ,
    'CY':    lambda p: _MAT_CY,
    'CH':    lambda p: _MAT_CH,
    'SWAP':  lambda p: _MAT_SWAP,
    'DCX':   lambda p: _MAT_DCX,
    'ISWAP': lambda p: _MAT_ISWAP,
    # 1-param, 2-qubit
    'CRX':   _mat_crx,
    'CRY':   _mat_cry,
    'CRZ':   _mat_crz,
    'CP':    lambda p: np.diag([1, 1, 1, np.exp(1j*p[0])]).astype(complex),
    'RXX':   _mat_rxx,
    'RYY':   _mat_ryy,
    'RZZ':   lambda p: np.diag([np.exp(-1j*p[0]/2), np.exp(1j*p[0]/2),
                                 np.exp(1j*p[0]/2), np.exp(-1j*p[0]/2)]),
    # 0-param, 3-qubit
    'CCX':   lambda p: _MAT_CCX,
    'CSWAP': lambda p: _MAT_CSWAP,
}


def _np_gate_matrix(name: str, params: tuple[float, ...] = ()) -> np.ndarray:
    """Return the unitary matrix for a named gate via registry lookup."""
    canonical = GATE_ALIASES.get(name, name)
    builder = _GATE_BUILDERS.get(canonical)
    if builder is None:
        raise ValueError(f"No matrix for gate: {name}")
    return builder(params)


def _apply_gate_np(
    sv: np.ndarray, matrix: np.ndarray, qubits: list[int], n_qubits: int,
) -> np.ndarray:
    """Apply a k-qubit gate to a statevector using numpy tensordot.

    Returns a tensor of shape [2]*n (possibly non-contiguous view).
    This avoids an extra 32 GB copy for large qubit counts.
    The caller stores the result directly; previous sv is freed.
    """
    k = len(qubits)
    sv_t = sv.reshape([2] * n_qubits)  # view if already contiguous
    axes = [n_qubits - 1 - q for q in qubits]
    gate_t = matrix.reshape([2] * (2 * k))
    # Contract gate's input axes (k..2k-1) with sv's target axes
    result = np.tensordot(gate_t, sv_t, axes=(list(range(k, 2*k)), axes))
    # Move gate output axes (0..k-1) to the target positions (view, no copy)
    return np.moveaxis(result, list(range(k)), axes)


def _measure_np(
    sv: np.ndarray, qubit: int, n_qubits: int,
) -> tuple[int, np.ndarray]:
    """Mid-circuit measurement with Born-rule sampling and state collapse.

    Computes marginal probabilities for |0> and |1> on the target qubit,
    samples an outcome, and collapses + renormalizes the statevector.
    Includes a numerical floor to avoid division-by-zero when both
    marginals underflow to zero.
    """
    sv = np.ascontiguousarray(sv).reshape([2] * n_qubits)
    ax = n_qubits - 1 - qubit
    idx_0 = [slice(None)] * n_qubits
    idx_1 = [slice(None)] * n_qubits
    idx_0[ax] = 0
    idx_1[ax] = 1
    p0 = float(np.sum(np.abs(sv[tuple(idx_0)])**2))
    p1 = float(np.sum(np.abs(sv[tuple(idx_1)])**2))
    total = p0 + p1
    if total < 1e-300:
        # Numerical underflow: both marginals are essentially zero.
        # Default to outcome 0 and return a normalized |0⟩ on this qubit.
        new_sv = np.zeros(2**n_qubits, dtype=complex)
        new_sv[0] = 1.0
        return 0, new_sv
    outcome = 0 if np.random.random() < p0 / total else 1
    new_sv = np.zeros_like(sv)
    if outcome == 0:
        new_sv[tuple(idx_0)] = sv[tuple(idx_0)] / np.sqrt(p0)
    else:
        new_sv[tuple(idx_1)] = sv[tuple(idx_1)] / np.sqrt(p1)
    return outcome, new_sv.reshape(-1)


def _sample_np(sv: np.ndarray, n_qubits: int, shots: int) -> dict[str, int]:
    """Sample measurement outcomes from a statevector."""
    probs = np.abs(np.ascontiguousarray(sv).ravel())**2
    probs /= probs.sum()
    indices = np.random.choice(probs.size, size=shots, p=probs)
    unique, ucounts = np.unique(indices, return_counts=True)
    return {format(idx, f'0{n_qubits}b'): int(cnt) for idx, cnt in zip(unique, ucounts)}


def _sample_one_np(sv: np.ndarray, n_qubits: int) -> str:
    """Sample a single measurement outcome."""
    counts = _sample_np(sv, n_qubits, 1)
    return next(iter(counts))


class LOCCEngine:
    """N-register quantum simulation with classical channels.

    Supports 2-26 independent quantum registers (A through Z).
    SPLIT mode: independent statevectors per register. Max capacity.
    JOINT mode: one statevector, LOCC constraints enforced.
    """

    def __init__(self, sizes: list[int], joint: bool = False):
        """Initialize LOCC engine with given register sizes."""
        self.sizes = list(sizes)
        self.n_regs = len(self.sizes)
        self.names = [chr(ord('A') + i) for i in range(self.n_regs)]
        self.joint = joint
        self.classical = {}
        # Precompute offsets for JOINT mode
        self.offsets = []
        off = 0
        for s in self.sizes:
            self.offsets.append(off)
            off += s
        self.n_total = sum(self.sizes)
        self.reset()

    # Backward-compatible properties
    @property
    def n_a(self):
        return self.sizes[0] if self.sizes else 0

    @property
    def n_b(self):
        return self.sizes[1] if len(self.sizes) > 1 else 0

    def _idx(self, reg):
        """Register name -> index."""
        return ord(reg) - ord('A')

    def reset(self) -> None:
        """Reset all registers to |0> and clear classical state."""
        self.classical.clear()
        if self.joint:
            self.sv = np.zeros(2**self.n_total, dtype=complex)
            self.sv[0] = 1.0
        else:
            self.svs = {}
            for name, size in zip(self.names, self.sizes):
                sv = np.zeros(2**size, dtype=complex)
                sv[0] = 1.0
                self.svs[name] = sv

    def apply(self, reg: str, gate_name: str, params: tuple[float, ...], qubits: list[int]) -> None:
        """Apply a gate to a specific register."""
        matrix = _np_gate_matrix(gate_name, tuple(params))
        if self.joint:
            idx = self._idx(reg)
            actual = [q + self.offsets[idx] for q in qubits]
            self.sv = _apply_gate_np(self.sv, matrix, actual, self.n_total)
        else:
            size = self.sizes[self._idx(reg)]
            self.svs[reg] = _apply_gate_np(self.svs[reg], matrix, qubits, size)

    def share(self, reg1: str, q1: int, reg2: str, q2: int) -> None:
        """Create Bell pair |Phi+> between reg1[q1] and reg2[q2]. JOINT only."""
        if not self.joint:
            raise RuntimeError("SHARE requires LOCC JOINT mode")
        h = _np_gate_matrix('H', ())
        cx = _np_gate_matrix('CX', ())
        actual1 = q1 + self.offsets[self._idx(reg1)]
        actual2 = q2 + self.offsets[self._idx(reg2)]
        self.sv = _apply_gate_np(self.sv, h, [actual1], self.n_total)
        self.sv = _apply_gate_np(self.sv, cx, [actual1, actual2], self.n_total)

    def send(self, reg: str, qubit: int) -> int:
        """Measure a qubit (Born rule) and return the classical outcome."""
        if self.joint:
            actual = qubit + self.offsets[self._idx(reg)]
            outcome, self.sv = _measure_np(self.sv, actual, self.n_total)
        else:
            size = self.sizes[self._idx(reg)]
            outcome, self.svs[reg] = _measure_np(self.svs[reg], qubit, size)
        return outcome

    def get_sv(self, reg: str) -> np.ndarray:
        if self.joint:
            return self.sv
        return self.svs[reg]

    def get_n(self, reg: str) -> int:
        if self.joint:
            return self.n_total
        return self.sizes[self._idx(reg)]

    def get_size(self, reg: str) -> int:
        return self.sizes[self._idx(reg)]

    def sample_joint(self, shots: int) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
        """Sample and return (per_reg_counts, joint_counts)."""
        if self.joint:
            raw = _sample_np(self.sv, self.n_total, shots)
            per_reg = {name: {} for name in self.names}
            joint = {}
            for state, count in raw.items():
                # Split the joint bitstring into per-register segments.
                # The joint statevector is ordered A[0..nA-1] B[0..nB-1] ...
                # with register A occupying the most-significant qubits.
                # The bitstring is MSB-first, so register A's bits are at
                # the left end. We walk backward from the right to peel
                # off each register's segment from least-significant first.
                parts = []
                pos = len(state)
                for i in range(self.n_regs):
                    size = self.sizes[i]
                    parts.append(state[pos - size:pos])
                    pos -= size
                for name, part in zip(self.names, parts):
                    per_reg[name][part] = per_reg[name].get(part, 0) + count
                jkey = '|'.join(parts)
                joint[jkey] = joint.get(jkey, 0) + count
            return per_reg, joint
        else:
            per_reg = {}
            for name in self.names:
                size = self.sizes[self._idx(name)]
                per_reg[name] = _sample_np(self.svs[name], size, shots)
            return per_reg, {}

    def apply_matrix(self, reg: str, matrix: np.ndarray, qubits: list[int]) -> None:
        """Apply a raw unitary matrix to qubits in a register."""
        if self.joint:
            idx = self._idx(reg)
            actual = [q + self.offsets[idx] for q in qubits]
            self.sv = _apply_gate_np(self.sv, matrix, actual, self.n_total)
        else:
            size = self.sizes[self._idx(reg)]
            self.svs[reg] = _apply_gate_np(self.svs[reg], matrix, qubits, size)

    def mem_gb(self) -> tuple[float, float]:  # (total_gb, peak_gb)
        """Return (total_gb, peak_gb) realistic memory estimates including overhead."""
        if self.joint:
            total = (2**self.n_total) * 16 * OVERHEAD_FACTOR / 1e9
            return total, total
        total = sum((2**s) * 16 * OVERHEAD_FACTOR / 1e9 for s in self.sizes)
        peak = max((2**s) * 16 * OVERHEAD_FACTOR / 1e9 for s in self.sizes)
        return total, peak

