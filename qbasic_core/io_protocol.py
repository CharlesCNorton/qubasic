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
