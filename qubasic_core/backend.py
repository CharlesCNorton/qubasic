"""QUBASIC backend abstraction — Qiskit and numpy behind a common interface."""

from __future__ import annotations

from typing import Any, Protocol



class QuantumBackend(Protocol):
    """Abstract quantum backend for gate application."""

    def apply_gate(self, gate_name: str, params: tuple[float, ...],
                   qubits: list[int]) -> None: ...
    def barrier(self) -> None: ...
    def reset(self, qubit: int) -> None: ...


class QiskitBackend:
    """Wraps a QuantumCircuit for the circuit-build path."""

    def __init__(self, qc: Any, apply_gate_fn: Any):
        self._qc = qc
        self._apply_gate = apply_gate_fn

    def apply_gate(self, gate_name: str, params: tuple[float, ...],
                   qubits: list[int]) -> None:
        self._apply_gate(self._qc, gate_name, list(params), qubits)

    def barrier(self) -> None:
        self._qc.barrier()

    def reset(self, qubit: int) -> None:
        self._qc.reset(qubit)

    def measure(self, qubit: int, classical_bit: Any) -> None:
        self._qc.measure(qubit, classical_bit)

    def add_classical_register(self, name: str, size: int = 1) -> Any:
        from qiskit.circuit import ClassicalRegister
        cr = ClassicalRegister(size, name)
        self._qc.add_register(cr)
        return cr

    def h(self, qubit: int) -> None:
        self._qc.h(qubit)

    def append_inverse(self, sub_qc: Any, qubits: list[int]) -> None:
        self._qc.append(sub_qc.inverse(), qubits)

    def append_controlled(self, gate: Any, qubits: list[int]) -> None:
        self._qc.append(gate, qubits)

    @property
    def qc(self) -> Any:
        return self._qc


class LOCCRegBackend:
    """Wraps a LOCCEngine register for the numpy path."""

    def __init__(self, engine: Any, reg: str):
        self._engine = engine
        self._reg = reg

    def apply_gate(self, gate_name: str, params: tuple[float, ...],
                   qubits: list[int]) -> None:
        self._engine.apply(self._reg, gate_name, params, qubits)

    def barrier(self) -> None:
        pass

    def reset(self, qubit: int) -> None:
        outcome = self._engine.send(self._reg, qubit)
        if outcome == 1:
            self._engine.apply(self._reg, 'X', (), [qubit])
