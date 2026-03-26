"""QBASIC — Quantum BASIC Interactive Terminal (package)."""

__version__ = '0.3.0'

from qbasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    LOCCEngine, ExecResult, ExecOutcome,
)
from qbasic_core.expression import ExpressionMixin
from qbasic_core.display import DisplayMixin
from qbasic_core.locc import LOCCMixin
from qbasic_core.control_flow import ControlFlowMixin
from qbasic_core.terminal import QBasicTerminal
from qbasic_core.demos import DemoMixin
from qbasic_core.file_io import FileIOMixin
from qbasic_core.analysis import AnalysisMixin
from qbasic_core.sweep import SweepMixin
from qbasic_core.memory import MemoryMixin
from qbasic_core.strings import StringMixin
from qbasic_core.screen import ScreenMixin
from qbasic_core.classic import ClassicMixin
from qbasic_core.subs import SubroutineMixin
from qbasic_core.debug import DebugMixin
from qbasic_core.program_mgmt import ProgramMgmtMixin
from qbasic_core.profiler import ProfilerMixin
from qbasic_core.protocol import TerminalProtocol
from qbasic_core.errors import (
    QBasicError, QBasicSyntaxError, QBasicRuntimeError,
    QBasicBuildError, QBasicRangeError, QBasicIOError, QBasicUndefinedError,
)
from qbasic_core.io_protocol import IOPort, StdIOPort
from qbasic_core.statements import Stmt, RawStmt
from qbasic_core.parser import parse_stmt
from qbasic_core.exec_context import ExecContext
from qbasic_core.scope import Scope
from qbasic_core.backend import QuantumBackend, QiskitBackend, LOCCRegBackend

__all__ = [
    'QBasicTerminal', 'LOCCEngine', 'ExecResult', 'ExecOutcome',
    'ExpressionMixin', 'DisplayMixin', 'DemoMixin', 'ControlFlowMixin',
    'FileIOMixin', 'AnalysisMixin', 'SweepMixin',
    'MemoryMixin', 'StringMixin', 'ScreenMixin', 'ClassicMixin',
    'SubroutineMixin', 'DebugMixin', 'ProgramMgmtMixin', 'ProfilerMixin',
    'TerminalProtocol',
    'QBasicError', 'QBasicSyntaxError', 'QBasicRuntimeError',
    'QBasicBuildError', 'QBasicRangeError', 'QBasicIOError', 'QBasicUndefinedError',
    'IOPort', 'StdIOPort',
    'Stmt', 'RawStmt', 'parse_stmt',
    'ExecContext', 'Scope',
    'QuantumBackend', 'QiskitBackend', 'LOCCRegBackend',
    'GATE_TABLE', 'GATE_ALIASES',
    '__version__',
]
