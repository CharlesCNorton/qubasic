"""QBASIC file I/O mixin — SAVE, LOAD, INCLUDE, DIR, EXPORT, CSV, OPEN/CLOSE/PRINT#/INPUT#/EOF/LPRINT."""

from __future__ import annotations

import os
import re
import sys

import numpy as np

from qbasic_core.engine import (
    GATE_TABLE,
    MAX_INCLUDE_DEPTH,
    RE_OPEN, RE_CLOSE, RE_PRINT_FILE, RE_INPUT_FILE, RE_LPRINT,
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
            self.io.writeln("?USAGE: SAVE <filename>")
            return
        try:
            path = self._sanitize_path(rest)
        except ValueError as e:
            self.io.writeln(f"?SAVE ERROR: {e}")
            return
        checked = self._check_agent_path(path, "SAVE")
        if checked is None:
            return
        path = checked
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
                # Save LOCC config if active
                if self.locc_mode and self.locc:
                    mode = "JOINT" if self.locc.joint else "SPLIT"
                    sizes = ' '.join(str(s) for s in self.locc.sizes)
                    f.write(f"LOCC {mode} {sizes}\n")
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
            self.io.writeln(f"SAVED {len(self.program)} lines to {path}")
        except Exception as e:
            self.io.writeln(f"?SAVE ERROR: {e}")

    def cmd_load(self, rest: str) -> None:
        """LOAD <filename> — clear state and load a .qb program file."""
        if not rest:
            self.io.writeln("?USAGE: LOAD <filename>")
            return
        try:
            path = self._sanitize_path(rest)
        except ValueError as e:
            self.io.writeln(f"?LOAD ERROR: {e}")
            return
        if os.path.isdir(path):
            self.io.writeln(f"?{path} is a directory, not a file")
            return
        if not path.endswith('.qb') and not os.path.isfile(path):
            path += '.qb'
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        try:
            prev_qubits = self.num_qubits
            prev_shots = self.shots
            self.cmd_new(silent=True)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n\r')
                    if line and not line.startswith('#'):
                        self.process(line, track_undo=False)
            self.io.writeln(f"LOADED {path} ({len(self.program)} lines)")
            if self.num_qubits != prev_qubits:
                self.io.writeln(f"  (QUBITS changed: {prev_qubits} -> {self.num_qubits})")
            if self.shots != prev_shots:
                self.io.writeln(f"  (SHOTS changed: {prev_shots} -> {self.shots})")
        except Exception as e:
            self.io.writeln(f"?LOAD ERROR: {e}")

    def cmd_include(self, rest: str) -> None:
        """INCLUDE file.qb — merge another program's lines into current.

        Depth-limited to MAX_INCLUDE_DEPTH to prevent infinite recursion.
        Cycle detection via _include_stack prevents A->B->A loops.
        SAVE, LOAD, and INCLUDE are blocked inside included files so that
        an included script cannot write arbitrary files or recurse further
        without the user's direct interaction.
        """
        if self._include_depth >= MAX_INCLUDE_DEPTH:
            self.io.writeln(f"?INCLUDE DEPTH LIMIT ({MAX_INCLUDE_DEPTH}) — possible recursion")
            return
        try:
            path = self._sanitize_path(rest)
        except ValueError as e:
            self.io.writeln(f"?INCLUDE ERROR: {e}")
            return
        if not path.endswith('.qb') and not os.path.isfile(path):
            path += '.qb'
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        resolved = os.path.realpath(path)
        if resolved in self._include_stack:
            self.io.writeln(f"?INCLUDE CYCLE: {path} already in include chain")
            return
        count = 0
        self._include_depth += 1
        self._include_stack.append(resolved)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n\r')
                    if not line or line.startswith('#'):
                        continue
                    # Block file-writing commands inside includes
                    first_word = line.split(None, 1)[0].upper() if line.strip() else ''
                    if first_word in ('SAVE', 'LOAD', 'EXPORT', 'CSV'):
                        self.io.writeln(f"  ?BLOCKED IN INCLUDE: {first_word}")
                        continue
                    self.process(line, track_undo=False)
                    count += 1
        finally:
            self._include_stack.pop()
            self._include_depth -= 1
        self.io.writeln(f"INCLUDED {path} ({count} lines)")

    def cmd_import(self, rest: str) -> None:
        """IMPORT "file.qb" — load subroutine definitions with namespace prefix.

        DEFs in the imported file are prefixed with the module name.
        E.g., IMPORT "math.qb" makes DEF ROT available as MATH.ROT.
        """
        from qbasic_core.engine import RE_IMPORT
        m = RE_IMPORT.match(f"IMPORT {rest}") if not rest.startswith('IMPORT') else RE_IMPORT.match(rest)
        path = rest.strip().strip('"').strip("'")
        if not path:
            self.io.writeln('?USAGE: IMPORT "filename"')
            return
        try:
            path = self._sanitize_path(path)
        except ValueError as e:
            self.io.writeln(f"?IMPORT ERROR: {e}")
            return
        if not path.endswith('.qb') and not os.path.isfile(path):
            path += '.qb'
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        # Module name from filename (without extension)
        mod_name = os.path.splitext(os.path.basename(path))[0].upper()
        # Parse the file for DEF statements
        import_count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n\r').strip()
                if not line or line.startswith('#'):
                    continue
                # Look for DEF name = body
                upper = line.upper()
                if upper.startswith('DEF ') and '=' in line and not upper.startswith('DEF BEGIN'):
                    from qbasic_core.engine import RE_DEF_SINGLE
                    dm = RE_DEF_SINGLE.match(line[4:].strip())
                    if dm:
                        name = dm.group(1).upper()
                        params = [p.strip() for p in dm.group(2).split(',')] if dm.group(2) else []
                        body = [s.strip() for s in dm.group(3).split(':') if s.strip()]
                        qualified = f"{mod_name}.{name}"
                        self.subroutines[qualified] = {'body': body, 'params': params}
                        import_count += 1
                # Also import SUB/FUNCTION definitions as numbered lines
                elif upper.startswith('SUB ') or upper.startswith('FUNCTION '):
                    # Store the line for later _scan_subs to pick up
                    # Use high line numbers to avoid conflicts
                    max_ln = max(self.program.keys()) if self.program else 0
                    base = max(max_ln + 1000, 90000)
                    self.process(f"{base + import_count * 10} {line}", track_undo=False)
                    import_count += 1
                elif upper in ('END SUB', 'END FUNCTION') and import_count > 0:
                    max_ln = max(self.program.keys()) if self.program else 0
                    self.process(f"{max_ln + 10} {line}", track_undo=False)
                    import_count += 1
        self.io.writeln(f"IMPORTED {mod_name} ({import_count} definitions from {path})")

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
                    self.io.writeln(f"  {f:<30} {size:>6} bytes")
            else:
                self.io.writeln("  No .qb files found")
        except Exception as e:
            self.io.writeln(f"?DIR ERROR: {e}")

    def cmd_export(self, rest: str) -> None:
        """EXPORT [filename] — export circuit as OpenQASM 3.0."""
        if self.last_circuit is None:
            self.io.writeln("?NO CIRCUIT — RUN first")
            return
        qasm = None
        errors = []
        try:
            from qiskit.qasm3 import dumps
            qasm = dumps(self.last_circuit)
        except Exception as e:
            errors.append(str(e))
        if qasm is None:
            self.io.writeln("?EXPORT: OpenQASM export not available.")
            for err in errors:
                self.io.writeln(f"  {err}")
            return
        if rest.strip():
            try:
                path = self._sanitize_path(rest)
            except ValueError as e:
                self.io.writeln(f"?EXPORT ERROR: {e}")
                return
            checked = self._check_agent_path(path, "EXPORT")
            if checked is None:
                return
            path = checked
            if os.path.exists(path):
                self.io.writeln(f"  (overwriting {path})")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(qasm)
            self.io.writeln(f"EXPORTED to {path} (OpenQASM 3.0)")
        else:
            self.io.writeln(qasm)

    def cmd_csv(self, rest: str) -> None:
        """CSV [filename] — export last results to CSV.

        Includes measurement counts, and statevector amplitudes if available.
        """
        if self.last_counts is None and self.last_sv is None:
            self.io.writeln("?NO RESULTS — RUN first")
            return
        lines = []
        if self.last_counts:
            total = sum(self.last_counts.values())
            lines.append("state,count,probability")
            for state, count in sorted(self.last_counts.items(), key=lambda x: -x[1]):
                lines.append(f"{state},{count},{count/total:.6f}")
        if self.last_sv is not None:
            lines.append("")
            lines.append("state,amplitude_re,amplitude_im,probability")
            sv = np.ascontiguousarray(self.last_sv).ravel()
            for i, amp in enumerate(sv):
                p = abs(amp)**2
                if p > 1e-8:
                    state = format(i, f'0{self.num_qubits}b')
                    lines.append(f"{state},{amp.real:.8f},{amp.imag:.8f},{p:.8f}")
        if rest.strip():
            try:
                path = self._sanitize_path(rest)
            except ValueError as e:
                self.io.writeln(f"?CSV ERROR: {e}")
                return
            checked = self._check_agent_path(path, "CSV")
            if checked is None:
                return
            path = checked
            if os.path.exists(path):
                self.io.writeln(f"  (overwriting {path})")
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            self.io.writeln(f"EXPORTED {len(self.last_counts)} states to {path}")
        else:
            for line in lines[:20]:
                self.io.writeln(f"  {line}")
            if len(lines) > 20:
                self.io.writeln(f"  ... ({len(lines)-20} more)")

    # ── File handles: OPEN, CLOSE, PRINT#, INPUT#, EOF, LPRINT ────────

    def _init_file_handles(self) -> None:
        self._file_handles: dict[int, Any] = {}
        self._lprint_path: str | None = None

    def _check_agent_path(self, path: str, cmd: str) -> str | None:
        """In agent mode, restrict paths to cwd. Returns resolved path or None on error.

        Uses pathlib is_relative_to for safe containment check instead of
        string prefix matching, which can be bypassed via symlinks or
        crafted path components that share a common prefix string.
        """
        if not self.agent_mode:
            return path
        from pathlib import Path
        resolved = Path(os.path.realpath(os.path.join(os.getcwd(), path)))
        cwd_real = Path(os.path.realpath(os.getcwd()))
        if not resolved.is_relative_to(cwd_real):
            self.io.writeln(f"?{cmd} BLOCKED: path escapes working directory in agent mode")
            return None
        return str(resolved)

    def cmd_open(self, rest: str) -> None:
        """OPEN file FOR INPUT|OUTPUT|APPEND AS #n [ENCODING "enc"]"""
        m = RE_OPEN.match(f"OPEN {rest}")
        if not m:
            self.io.writeln('?USAGE: OPEN "file" FOR INPUT|OUTPUT|APPEND AS #n [ENCODING "enc"]')
            return
        path, mode_str, handle = m.group(1).strip(), m.group(2).upper(), int(m.group(3))
        encoding = m.group(4).strip() if m.group(4) else 'utf-8'
        try:
            path = self._sanitize_path(path)
        except ValueError as e:
            self.io.writeln(f"?OPEN ERROR: {e}")
            return
        checked = self._check_agent_path(path, "OPEN")
        if checked is None:
            return
        path = checked
        mode_map = {'INPUT': 'r', 'OUTPUT': 'w', 'APPEND': 'a', 'RANDOM': 'r+'}
        mode = mode_map.get(mode_str, 'r')
        try:
            if encoding.lower() == 'binary':
                bin_mode = mode + 'b'
                if mode_str == 'RANDOM' and not os.path.isfile(path):
                    open(path, 'wb').close()
                if mode_str == 'RANDOM' and os.path.isfile(path):
                    self.io.writeln(f"  (opening existing file {path} for random access)")
                self._file_handles[handle] = open(path, bin_mode)
            else:
                if mode_str == 'RANDOM' and not os.path.isfile(path):
                    open(path, 'w', encoding=encoding).close()
                if mode_str == 'RANDOM' and os.path.isfile(path):
                    self.io.writeln(f"  (opening existing file {path} for random access)")
                self._file_handles[handle] = open(path, mode, encoding=encoding)
            self.io.writeln(f"OPENED #{handle} ({path}, {mode_str})")
        except Exception as e:
            self.io.writeln(f"?OPEN ERROR: {e}")

    def cmd_close(self, rest: str) -> None:
        """CLOSE #n — close a file handle."""
        m = RE_CLOSE.match(f"CLOSE {rest}")
        if not m:
            self.io.writeln("?USAGE: CLOSE #n")
            return
        handle = int(m.group(1))
        if handle in self._file_handles:
            self._file_handles[handle].close()
            del self._file_handles[handle]
            self.io.writeln(f"CLOSED #{handle}")
        else:
            self.io.writeln(f"?HANDLE #{handle} NOT OPEN")

    def _exec_print_file(self, stmt: str, run_vars: dict[str, Any]) -> bool:
        """Handle PRINT #n, data during execution."""
        m = RE_PRINT_FILE.match(stmt)
        if not m:
            return False
        handle = int(m.group(1))
        data = m.group(2).strip()
        if handle not in self._file_handles:
            self.io.writeln(f"?HANDLE #{handle} NOT OPEN")
            return True
        f = self._file_handles[handle]
        fmode = getattr(f, 'mode', '')
        if fmode.startswith('r') and '+' not in fmode:
            self.io.writeln(f"?HANDLE #{handle} NOT OPEN FOR WRITING")
            return True
        # Evaluate data
        if (data.startswith('"') and data.endswith('"')):
            f.write(data[1:-1] + '\n')
        else:
            try:
                val = self._eval_with_vars(data, run_vars) if run_vars else self.eval_expr(data)
                f.write(str(val) + '\n')
            except Exception:
                f.write(data + '\n')
        f.flush()
        return True

    def _exec_input_file(self, stmt: str, run_vars: dict[str, Any]) -> bool:
        """Handle INPUT #n, var during execution."""
        m = RE_INPUT_FILE.match(stmt)
        if not m:
            return False
        handle = int(m.group(1))
        var = m.group(2)
        if handle not in self._file_handles:
            self.io.writeln(f"?HANDLE #{handle} NOT OPEN")
            return True
        f = self._file_handles[handle]
        fmode = getattr(f, 'mode', '')
        if fmode.startswith(('w', 'a')) and '+' not in fmode:
            self.io.writeln(f"?HANDLE #{handle} NOT OPEN FOR READING")
            return True
        line = f.readline()
        if not line:
            run_vars[var] = 0
            self.variables[var] = 0
        else:
            line = line.strip()
            if var.endswith('$'):
                run_vars[var] = line
                self.variables[var] = line
            else:
                try:
                    val = float(line) if '.' in line else int(line)
                except ValueError:
                    val = line
                run_vars[var] = val
                self.variables[var] = val
        return True

    def _exec_lprint(self, stmt: str, run_vars: dict[str, Any]) -> bool:
        """Handle LPRINT data — output to log file or stderr."""
        m = RE_LPRINT.match(stmt)
        if not m:
            return False
        data = m.group(1).strip()
        if (data.startswith('"') and data.endswith('"')):
            text = data[1:-1]
        else:
            try:
                text = str(self._eval_with_vars(data, run_vars) if run_vars else self.eval_expr(data))
            except Exception:
                text = data
        if self._lprint_path:
            with open(self._lprint_path, 'a', encoding='utf-8') as f:
                f.write(text + '\n')
        else:
            print(text, file=sys.stderr)
        return True

    def _eof(self, handle: float) -> float:
        """EOF(n) — return 1 if at end of file, 0 otherwise."""
        h = int(handle)
        if h not in self._file_handles:
            return 1.0
        f = self._file_handles[h]
        pos = f.tell()
        try:
            ch = f.read(1)
            if not ch:
                return 1.0
            f.seek(pos)
        except (UnicodeDecodeError, Exception):
            return 1.0
        return 0.0
