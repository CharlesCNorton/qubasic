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

    def keys(self):
        return set(self._persistent.keys()) | set(self._runtime.keys())

    def items(self):
        merged = {**self._persistent, **self._runtime}
        return merged.items()

    def values(self):
        merged = {**self._persistent, **self._runtime}
        return merged.values()

    def update(self, other):
        if isinstance(other, dict):
            for k, v in other.items():
                self[k] = v
        elif isinstance(other, Scope):
            for k, v in other.items():
                self[k] = v

    def as_dict(self) -> dict[str, Any]:
        """Merged view for expression evaluation."""
        return {**self._persistent, **self._runtime}

    def __delitem__(self, name: str) -> None:
        self._runtime.pop(name, None)
        self._persistent.pop(name, None)

    def pop(self, name: str, *default):
        # Check runtime first, then persistent, before falling back to default.
        if name in self._runtime:
            val = self._runtime.pop(name)
            self._persistent.pop(name, None)
            return val
        if name in self._persistent:
            return self._persistent.pop(name)
        if default:
            return default[0]
        raise KeyError(name)
