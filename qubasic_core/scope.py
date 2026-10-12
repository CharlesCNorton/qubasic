"""QUBASIC variable scope — the run's view of the variable table."""

from __future__ import annotations

from typing import Any


class Scope:
    """The variables a running program reads and writes.

    A view over the terminal's variable table: reads and writes go straight
    to it, so a SUB or FUNCTION that swaps the table for its own locals (see
    subs.py) is seen by every statement that holds this view.
    """

    def __init__(self, persistent: dict[str, Any]):
        self._persistent = persistent

    def get(self, name: str, default: Any = None) -> Any:
        return self._persistent.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self._persistent

    def __getitem__(self, name: str) -> Any:
        return self._persistent[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._persistent[name] = value

    def keys(self):
        return self._persistent.keys()

    def items(self):
        return self._persistent.items()

    def values(self):
        return self._persistent.values()

    def update(self, other):
        for k, v in dict(other.items()).items():
            self._persistent[k] = v

    def as_dict(self) -> dict[str, Any]:
        """A copy of the variables, for expression evaluation."""
        return dict(self._persistent)

    def __delitem__(self, name: str) -> None:
        self._persistent.pop(name, None)

    def pop(self, name: str, *default):
        if name in self._persistent:
            return self._persistent.pop(name)
        if default:
            return default[0]
        raise KeyError(name)
