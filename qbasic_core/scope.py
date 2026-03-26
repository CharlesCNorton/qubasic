"""QBASIC variable scope — unified layered scope model."""

from __future__ import annotations

from typing import Any


class Scope:
    """Layered variable scope: persistent + runtime.

    Reads check runtime first, then persistent.
    Writes mirror to both for backward compatibility with code
    that reads self.variables directly.
    """

    def __init__(self, persistent: dict[str, Any]):
        self._persistent = persistent
        self._runtime: dict[str, Any] = {}

    def get(self, name: str, default: Any = None) -> Any:
        if name in self._runtime:
            return self._runtime[name]
        return self._persistent.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self._runtime or name in self._persistent

    def __getitem__(self, name: str) -> Any:
        if name in self._runtime:
            return self._runtime[name]
        return self._persistent[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._runtime[name] = value
        self._persistent[name] = value

    def as_dict(self) -> dict[str, Any]:
        """Merged view for expression evaluation."""
        return {**self._persistent, **self._runtime}

    def reset_runtime(self) -> None:
        self._runtime.clear()
