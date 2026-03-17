"""QBASIC file I/O mixin — SAVE, LOAD, INCLUDE, DIR, EXPORT, CSV."""

from __future__ import annotations

import os
import re

import numpy as np

from qbasic_core.engine import (
    GATE_TABLE,
    MAX_INCLUDE_DEPTH,
)


class FileIOMixin:
    """File I/O commands for QBasicTerminal.

    Requires: TerminalProtocol — uses self.program, self.num_qubits, self.shots,
    self.sim_method, self.sim_device, self._custom_gates, self.subroutines,
    self.registers, self.variables, self.last_counts, self.last_circuit,
    self._include_depth, self._sanitize_path(), self.cmd_new(), self.process().
    """

    def cmd_save(self, rest: str) -> None:
        """SAVE <filename> — write program, config, and definitions to a .qb file."""
        if not rest:
            print("?USAGE: SAVE <filename>")
            return
        try:
            path = self._sanitize_path(rest)
        except ValueError as e:
            print(f"?SAVE ERROR: {e}")
            return
        if not path.endswith('.qb'):
            path += '.qb'
        try:
            with open(path, 'w', encoding='utf-8') as f:
                # Save config
                f.write(f"QUBITS {self.num_qubits}\n")
                f.write(f"SHOTS {self.shots}\n")
                if self.sim_method != 'automatic':
                    f.write(f"METHOD {self.sim_method}\n")
                if self.sim_device != 'CPU':
                    f.write(f"METHOD {self.sim_device}\n")
                # Save custom gates as executable UNITARY commands
                for name, matrix in self._custom_gates.items():
                    rows = matrix.tolist()
                    mat_str = '[' + ','.join(
                        '[' + ','.join(str(v) for v in row) + ']'
                        for row in rows
                    ) + ']'
                    f.write(f"UNITARY {name} = {mat_str}\n")
                # Save subroutines
                for name, sub in self.subroutines.items():
                    if isinstance(sub, list):
                        f.write(f"DEF {name} = {' : '.join(sub)}\n")
                    else:
                        params = f"({', '.join(sub['params'])})" if sub['params'] else ""
                        f.write(f"DEF {name}{params} = {' : '.join(sub['body'])}\n")
                # Save registers
                for name, (start, size) in self.registers.items():
                    f.write(f"REG {name} {size}\n")
                # Save variables
                for name, val in self.variables.items():
                    f.write(f"LET {name} = {val}\n")
                # Save program
                for num in sorted(self.program.keys()):
                    f.write(f"{num} {self.program[num]}\n")
            print(f"SAVED {len(self.program)} lines to {path}")
        except Exception as e:
            print(f"?SAVE ERROR: {e}")

    def cmd_load(self, rest: str) -> None:
        """LOAD <filename> — clear state and load a .qb program file."""
        if not rest:
            print("?USAGE: LOAD <filename>")
            return
        try:
            path = self._sanitize_path(rest)
        except ValueError as e:
            print(f"?LOAD ERROR: {e}")
            return
        if os.path.isdir(path):
            print(f"?{path} is a directory, not a file")
            return
        if not path.endswith('.qb') and not os.path.isfile(path):
            path += '.qb'
        if not os.path.isfile(path):
            print(f"?FILE NOT FOUND: {path}")
            return
        try:
            self.cmd_new(silent=True)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n\r')
                    if line and not line.startswith('#'):
                        self.process(line)
            print(f"LOADED {path} ({len(self.program)} lines)")
        except Exception as e:
            print(f"?LOAD ERROR: {e}")

    def cmd_include(self, rest: str) -> None:
        """INCLUDE file.qb — merge another program's lines into current.

        Depth-limited to MAX_INCLUDE_DEPTH to prevent infinite recursion.
        SAVE, LOAD, and INCLUDE are blocked inside included files so that
        an included script cannot write arbitrary files or recurse further
        without the user's direct interaction.
        """
        if self._include_depth >= MAX_INCLUDE_DEPTH:
            print(f"?INCLUDE DEPTH LIMIT ({MAX_INCLUDE_DEPTH}) — possible recursion")
            return
        try:
            path = self._sanitize_path(rest)
        except ValueError as e:
            print(f"?INCLUDE ERROR: {e}")
            return
        if not path.endswith('.qb') and not os.path.isfile(path):
            path += '.qb'
        if not os.path.isfile(path):
            print(f"?FILE NOT FOUND: {path}")
            return
        count = 0
        self._include_depth += 1
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n\r')
                    if not line or line.startswith('#'):
                        continue
                    # Block file-writing commands inside includes
                    first_word = line.split(None, 1)[0].upper() if line.strip() else ''
                    if first_word in ('SAVE', 'LOAD', 'EXPORT', 'CSV'):
                        print(f"  ?BLOCKED IN INCLUDE: {first_word}")
                        continue
                    self.process(line)
                    count += 1
        finally:
            self._include_depth -= 1
        print(f"INCLUDED {path} ({count} lines)")

    def cmd_dir(self, rest: str = '') -> None:
        """List .qb files in current or specified directory."""
        if rest.strip():
            path = self._sanitize_path(rest)
        else:
            path = '.'
        try:
            files = [f for f in os.listdir(path) if f.endswith('.qb')]
            if files:
                for f in sorted(files):
                    size = os.path.getsize(os.path.join(path, f))
                    print(f"  {f:<30} {size:>6} bytes")
            else:
                print("  No .qb files found")
        except Exception as e:
            print(f"?DIR ERROR: {e}")

    def cmd_export(self, rest: str) -> None:
        """EXPORT [filename] — export circuit as OpenQASM 3.0."""
        if self.last_circuit is None:
            print("?NO CIRCUIT — RUN first")
            return
        qasm = None
        errors = []
        try:
            from qiskit.qasm3 import dumps
            qasm = dumps(self.last_circuit)
        except Exception as e:
            errors.append(str(e))
        if qasm is None:
            print("?EXPORT: OpenQASM export not available.")
            for err in errors:
                print(f"  {err}")
            return
        if rest.strip():
            try:
                path = self._sanitize_path(rest)
            except ValueError as e:
                print(f"?EXPORT ERROR: {e}")
                return
            if os.path.exists(path):
                print(f"  (overwriting {path})")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(qasm)
            print(f"EXPORTED to {path} (OpenQASM 3.0)")
        else:
            print(qasm)

    def cmd_csv(self, rest: str) -> None:
        """CSV [filename] — export last results to CSV."""
        if self.last_counts is None:
            print("?NO RESULTS — RUN first")
            return
        total = sum(self.last_counts.values())
        lines = ["state,count,probability"]
        for state, count in sorted(self.last_counts.items(), key=lambda x: -x[1]):
            lines.append(f"{state},{count},{count/total:.6f}")
        if rest.strip():
            try:
                path = self._sanitize_path(rest)
            except ValueError as e:
                print(f"?CSV ERROR: {e}")
                return
            if os.path.exists(path):
                print(f"  (overwriting {path})")
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            print(f"EXPORTED {len(self.last_counts)} states to {path}")
        else:
            for line in lines[:20]:
                print(f"  {line}")
            if len(lines) > 20:
                print(f"  ... ({len(lines)-20} more)")
