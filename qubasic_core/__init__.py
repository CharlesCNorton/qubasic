"""QUBASIC — Quantum BASIC Interactive Terminal (package).

    from qubasic_core import run, Session    # qubasic_core.api
    run('10 H 0 : CX 0, 1 : MEASURE', qubits=2)['counts']
"""

from __future__ import annotations

from typing import Any

from qubasic_core.engine import (
    ExecContext,
    IOPort,
    QBasicBuildError,
    QBasicError,
    QBasicIOError,
    QBasicRangeError,
    QBasicRuntimeError,
    QBasicSyntaxError,
    QBasicUndefinedError,
    StdIOPort,
)

# Lightweight imports that don't pull in Qiskit/numpy
from qubasic_core.parser import GateStmt, RawStmt, Stmt

__all__ = [
    'QBasicTerminal', 'LOCCEngine', 'ExecResult', 'ExecOutcome',
    'ExpressionMixin', 'DisplayMixin', 'DemoMixin', 'ControlFlowMixin',
    'LOCCMixin', 'LOCCCommandsMixin', 'LOCCExecutionMixin',
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
    'run', 'Session',
]

__version__ = '0.31.1'

def __getattr__(name: str) -> Any:
    """Lazy import heavy modules on first access."""
    _lazy_imports = {
        'GATE_TABLE': 'qubasic_core.parser',
        'GATE_ALIASES': 'qubasic_core.parser',
        'LOCCEngine': 'qubasic_core.locc',
        'ExecResult': 'qubasic_core.engine',
        'ExecOutcome': 'qubasic_core.engine',
        'QBasicTerminal': 'qubasic_core.terminal',
        'Engine': 'qubasic_core.terminal',
        'ExecutorMixin': 'qubasic_core.circuits',
        'ExpressionMixin': 'qubasic_core.expression',
        'DisplayMixin': 'qubasic_core.display',
        'LOCCMixin': 'qubasic_core.locc',
        'LOCCCommandsMixin': 'qubasic_core.locc',
        'LOCCExecutionMixin': 'qubasic_core.locc',
        'ControlFlowMixin': 'qubasic_core.interpreter',
        'DemoMixin': 'qubasic_core.machine',
        'FileIOMixin': 'qubasic_core.devices',
        'AnalysisMixin': 'qubasic_core.analysis',
        'SweepMixin': 'qubasic_core.display',
        'MemoryMixin': 'qubasic_core.machine',
        'StringMixin': 'qubasic_core.expression',
        'ScreenMixin': 'qubasic_core.devices',
        'ClassicMixin': 'qubasic_core.interpreter',
        'SubroutineMixin': 'qubasic_core.interpreter',
        'DebugMixin': 'qubasic_core.interpreter',
        'ProgramMgmtMixin': 'qubasic_core.editing',
        'ProfilerMixin': 'qubasic_core.machine',
        'NoiseMixin': 'qubasic_core.noise',
        'StateDisplayMixin': 'qubasic_core.display',
        'TerminalProtocol': 'qubasic_core.engine',
        'HELP_TEXT': 'qubasic_core.display',
        'BANNER_ART': 'qubasic_core.display',
        'QuantumBackend': 'qubasic_core.engine',
        'QiskitBackend': 'qubasic_core.simtools',
        'LOCCRegBackend': 'qubasic_core.simtools',
        'Scope': 'qubasic_core.expression',
        'parse_stmt': 'qubasic_core.parser',
        'run': 'qubasic_core.api',
        'Session': 'qubasic_core.api',
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        val = getattr(module, name)
        globals()[name] = val  # cache for next access
        return val
    raise AttributeError(f"module 'qubasic_core' has no attribute {name}")
