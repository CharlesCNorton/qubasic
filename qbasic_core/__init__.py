"""QBASIC — Quantum BASIC Interactive Terminal (package)."""

# Lightweight imports that don't pull in Qiskit/numpy
from qbasic_core.statements import Stmt, GateStmt, RawStmt
from qbasic_core.errors import (
    QBasicError, QBasicSyntaxError, QBasicRuntimeError,
    QBasicBuildError, QBasicRangeError, QBasicIOError, QBasicUndefinedError,
)
from qbasic_core.io_protocol import IOPort, StdIOPort
from qbasic_core.exec_context import ExecContext

__all__ = [
    'QBasicTerminal', 'LOCCEngine', 'ExecResult', 'ExecOutcome',
    'ExpressionMixin', 'DisplayMixin', 'DemoMixin', 'ControlFlowMixin',
    'LOCCMixin', 'LOCCCommandsMixin', 'LOCCDisplayMixin', 'LOCCExecutionMixin',
    'NoiseMixin', 'StateDisplayMixin', 'HELP_TEXT', 'BANNER_ART',
    'FileIOMixin', 'AnalysisMixin', 'SweepMixin',
    'MemoryMixin', 'StringMixin', 'ScreenMixin', 'ClassicMixin',
    'SubroutineMixin', 'DebugMixin', 'ProgramMgmtMixin', 'ProfilerMixin',
    'TerminalProtocol',
    'Engine', 'ExecutorMixin',
    'QBasicError', 'QBasicSyntaxError', 'QBasicRuntimeError',
    'QBasicBuildError', 'QBasicRangeError', 'QBasicIOError', 'QBasicUndefinedError',
    'IOPort', 'StdIOPort',
    'Stmt', 'GateStmt', 'RawStmt', 'parse_stmt',
    'ExecContext', 'Scope',
    'QuantumBackend', 'QiskitBackend', 'LOCCRegBackend',
    'GATE_TABLE', 'GATE_ALIASES',
]

__version__ = '0.0.0'

def __getattr__(name):
    """Lazy import heavy modules on first access."""
    _lazy_imports = {
        'GATE_TABLE': 'qbasic_core.engine',
        'GATE_ALIASES': 'qbasic_core.engine',
        'LOCCEngine': 'qbasic_core.engine',
        'ExecResult': 'qbasic_core.engine',
        'ExecOutcome': 'qbasic_core.engine',
        'QBasicTerminal': 'qbasic_core.terminal',
        'Engine': 'qbasic_core.engine_state',
        'ExecutorMixin': 'qbasic_core.executor',
        'ExpressionMixin': 'qbasic_core.expression',
        'DisplayMixin': 'qbasic_core.display',
        'LOCCMixin': 'qbasic_core.locc',
        'LOCCCommandsMixin': 'qbasic_core.locc_commands',
        'LOCCDisplayMixin': 'qbasic_core.locc_display',
        'LOCCExecutionMixin': 'qbasic_core.locc_execution',
        'ControlFlowMixin': 'qbasic_core.control_flow',
        'DemoMixin': 'qbasic_core.demos',
        'FileIOMixin': 'qbasic_core.file_io',
        'AnalysisMixin': 'qbasic_core.analysis',
        'SweepMixin': 'qbasic_core.sweep',
        'MemoryMixin': 'qbasic_core.memory',
        'StringMixin': 'qbasic_core.strings',
        'ScreenMixin': 'qbasic_core.screen',
        'ClassicMixin': 'qbasic_core.classic',
        'SubroutineMixin': 'qbasic_core.subs',
        'DebugMixin': 'qbasic_core.debug',
        'ProgramMgmtMixin': 'qbasic_core.program_mgmt',
        'ProfilerMixin': 'qbasic_core.profiler',
        'NoiseMixin': 'qbasic_core.noise_mixin',
        'StateDisplayMixin': 'qbasic_core.state_display',
        'TerminalProtocol': 'qbasic_core.protocol',
        'HELP_TEXT': 'qbasic_core.help_text',
        'BANNER_ART': 'qbasic_core.help_text',
        'QuantumBackend': 'qbasic_core.backend',
        'QiskitBackend': 'qbasic_core.backend',
        'LOCCRegBackend': 'qbasic_core.backend',
        'Scope': 'qbasic_core.scope',
        'parse_stmt': 'qbasic_core.parser',
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        val = getattr(module, name)
        globals()[name] = val  # cache for next access
        return val
    raise AttributeError(f"module 'qbasic_core' has no attribute {name}")
