"""QBASIC I/O protocol — decouples execution engine from print/input."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IOPort(Protocol):
    """Abstract I/O port for execution output and user input."""

    def write(self, text: str) -> None: ...
    def writeln(self, text: str) -> None: ...
    def read_line(self, prompt: str) -> str: ...


class StdIOPort:
    """Default I/O: stdout/stdin."""

    def write(self, text: str) -> None:
        print(text, end='', flush=True)

    def writeln(self, text: str) -> None:
        print(text)

    def read_line(self, prompt: str) -> str:
        return input(prompt)
