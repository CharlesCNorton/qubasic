"""QUBASIC — Quantum BASIC Interactive Terminal (package)."""

# Lightweight imports that don't pull in Qiskit/numpy
from qubasic_core.statements import Stmt, GateStmt, RawStmt
from qubasic_core.errors import (
    QBasicError, QBasicSyntaxError, QBasicRuntimeError,
    QBasicBuildError, QBasicRangeError, QBasicIOError, QBasicUndefinedError,
)
from qubasic_core.io_protocol import IOPort, StdIOPort
from qubasic_core.exec_context import ExecContext

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

__version__ = '0.5.1'

def __getattr__(name):
    """Lazy import heavy modules on first access."""
    _lazy_imports = {
        'GATE_TABLE': 'qubasic_core.engine',
        'GATE_ALIASES': 'qubasic_core.engine',
        'LOCCEngine': 'qubasic_core.engine',
        'ExecResult': 'qubasic_core.engine',
        'ExecOutcome': 'qubasic_core.engine',
        'QBasicTerminal': 'qubasic_core.terminal',
        'Engine': 'qubasic_core.engine_state',
        'ExecutorMixin': 'qubasic_core.executor',
        'ExpressionMixin': 'qubasic_core.expression',
        'DisplayMixin': 'qubasic_core.display',
        'LOCCMixin': 'qubasic_core.locc',
        'LOCCCommandsMixin': 'qubasic_core.locc_commands',
        'LOCCDisplayMixin': 'qubasic_core.locc_display',
        'LOCCExecutionMixin': 'qubasic_core.locc_execution',
        'ControlFlowMixin': 'qubasic_core.control_flow',
        'DemoMixin': 'qubasic_core.demos',
        'FileIOMixin': 'qubasic_core.file_io',
        'AnalysisMixin': 'qubasic_core.analysis',
        'SweepMixin': 'qubasic_core.sweep',
        'MemoryMixin': 'qubasic_core.memory',
        'StringMixin': 'qubasic_core.strings',
        'ScreenMixin': 'qubasic_core.screen',
        'ClassicMixin': 'qubasic_core.classic',
        'SubroutineMixin': 'qubasic_core.subs',
        'DebugMixin': 'qubasic_core.debug',
        'ProgramMgmtMixin': 'qubasic_core.program_mgmt',
        'ProfilerMixin': 'qubasic_core.profiler',
        'NoiseMixin': 'qubasic_core.noise_mixin',
        'StateDisplayMixin': 'qubasic_core.state_display',
        'TerminalProtocol': 'qubasic_core.protocol',
        'HELP_TEXT': 'qubasic_core.help_text',
        'BANNER_ART': 'qubasic_core.help_text',
        'QuantumBackend': 'qubasic_core.backend',
        'QiskitBackend': 'qubasic_core.backend',
        'LOCCRegBackend': 'qubasic_core.backend',
        'Scope': 'qubasic_core.scope',
        'parse_stmt': 'qubasic_core.parser',
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        val = getattr(module, name)
        globals()[name] = val  # cache for next access
        return val
    raise AttributeError(f"module 'qubasic_core' has no attribute {name}")
