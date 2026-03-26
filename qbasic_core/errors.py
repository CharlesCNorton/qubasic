"""QBASIC structured error hierarchy."""

from __future__ import annotations


class QBasicError(Exception):
    """Base for all QBASIC errors."""

    def __init__(self, message: str, *, code: int = 0, line: int | None = None):
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
