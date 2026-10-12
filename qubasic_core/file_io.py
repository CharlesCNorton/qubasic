"""QUBASIC file I/O mixin — SAVE, LOAD, INCLUDE, DIR, EXPORT, CSV, OPEN/CLOSE/PRINT#/INPUT#/EOF/LPRINT."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from qubasic_core.engine import (
    GATE_TABLE, GATE_ALIASES,
    MAX_INCLUDE_DEPTH,
    RE_OPEN, RE_CLOSE, RE_PRINT_FILE, RE_LPRINT,
)
from qubasic_core.parser import parse_stmt
from qubasic_core.host import HostState
from qubasic_core.variables import Record, is_string_kind


def _record_leaves(rec: Record, prefix: str = '') -> list[tuple[str, Any]]:
    """(field path, value) for every non-record field of a record."""
    out: list[tuple[str, Any]] = []
    for key, val in rec.items():
        path = f"{prefix}{key}"
        if isinstance(val, Record):
            out.extend(_record_leaves(val, path + '.'))
        else:
            out.append((path, val))
    return out


def _literal(val: Any) -> str:
    """A value as a BASIC literal for SAVE."""
    if isinstance(val, str):
        return '"' + val.replace('"', '" + CHR$(34) + "') + '"'
    return str(val)


def number_source(lines: list[str]) -> list[tuple[int, str]]:
    """(line number, text) for QBASIC source lines. Blank lines are skipped;
    a line that starts with a number keeps it, and runs of unnumbered lines
    are numbered in steps of up to 10 that fit before the next numbered line."""
    entries: list[tuple[int | None, str]] = []
    for raw in lines:
        text = raw.rstrip()
        if not text.strip():
            continue
        m = re.match(r'\s*(\d+)(?:\s+(.*))?$', text)
        if m:
            if m.group(2):
                entries.append((int(m.group(1)), m.group(2).strip()))
            continue
        entries.append((None, text.strip()))
    out: list[tuple[int, str]] = []
    prev = 0
    i = 0
    while i < len(entries):
        num, text = entries[i]
        if num is not None:
            if num <= prev:
                raise ValueError(f"line {num} is out of order (after {prev})")
            out.append((num, text))
            prev = num
            i += 1
            continue
        j = i
        while j < len(entries) and entries[j][0] is None:
            j += 1
        count = j - i
        nxt = entries[j][0] if j < len(entries) else None
        step = 10 if nxt is None else min(10, (nxt - prev) // (count + 1))
        if step < 1:
            raise ValueError(f"no room to number {count} line(s) between {prev} and {nxt}")
        for k in range(count):
            out.append((prev + (k + 1) * step, entries[i + k][1]))
        prev = out[-1][0]
        i = j
    return out


class FileIOMixin(HostState):
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
        if path.lower().endswith('.bas'):
            imported_bas: set[int] = getattr(self, '_imported_lines', set())
            lines = [n for n in sorted(self.program) if n not in imported_bas]
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    for num in lines:
                        f.write(f"{num} {self.program[num]}\n")
                self.io.writeln(f"SAVED {len(lines)} lines to {path}")
            except OSError as e:
                self.io.writeln(f"?SAVE ERROR: {e}")
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
                if self.sim_precision != 'double':
                    f.write(f"PRECISION {self.sim_precision.upper()}\n")
                if getattr(self, '_screen_mode', 0):
                    f.write(f"SCREEN {self._screen_mode}\n")
                if getattr(self, '_option_base', 0):
                    f.write(f"OPTION BASE {self._option_base}\n")
                if getattr(self, '_endian_big', False):
                    f.write("OPTION ENDIAN BIG\n")
                if getattr(self, '_seed', None) is not None:
                    f.write(f"SEED {self._seed}\n")
                for line in self._device_lines():
                    f.write(line + "\n")
                # Save LOCC config if active
                if self.locc_mode and self.locc:
                    mode = "JOINT" if self.locc.joint else "SPLIT"
                    sizes = ' '.join(str(s) for s in self.locc.sizes)
                    f.write(f"LOCC {mode} {sizes}\n")
                # Save user-defined TYPEs (multi-line block, re-read on LOAD)
                for tname, fields in getattr(self, '_user_types', {}).items():
                    f.write(f"TYPE {tname}\n")
                    for fname, ftype in fields:
                        f.write(f"  {fname} AS {ftype}\n")
                    f.write("END TYPE\n")
                # Save DEF FN user functions
                for fname, fdef in getattr(self, '_user_fns', {}).items():
                    short = fname[2:] if fname.upper().startswith('FN') else fname
                    params = ', '.join(fdef['params'])
                    f.write(f"DEF FN {short}({params}) = {fdef['body']}\n")
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
                # Save variables (internal _vars skipped; a record as its DIM
                # and one LET per field)
                for name, val in self.variables.items():
                    if name.startswith('_'):
                        continue
                    if isinstance(val, Record):
                        f.write(f"DIM {name} AS {val.type_name}\n")
                        for path, leaf in _record_leaves(val):
                            f.write(f"LET {name}.{path} = {_literal(leaf)}\n")
                        continue
                    if not isinstance(val, (int, float, str)):
                        continue        # arrays of results, matrices, records of other kinds
                    f.write(f"LET {name} = {_literal(val)}\n")
                # Save program (exclude lines injected by IMPORT)
                imported: set[int] = getattr(self, '_imported_lines', set())
                saved = [n for n in sorted(self.program.keys()) if n not in imported]
                for num in saved:
                    f.write(f"{num} {self.program[num]}\n")
            self.io.writeln(f"SAVED {len(saved)} lines to {path}")
        except Exception as e:
            self.io.writeln(f"?SAVE ERROR: {e}")

    def _device_lines(self) -> list[str]:
        """The session lines that rebuild the noise model, device model and
        logical-qubit mode on LOAD (SAVE writes them after QUBITS and METHOD)."""
        lines: list[str] = []
        lm = self._logical_mode
        edges = ', '.join(f"{a}-{b}" for a, b in
                          sorted({tuple(sorted(e)) for e in self._coupling_map or ()}))
        if self._device_spec:
            lines.append(f"DEVICE {self._device_spec}")
            if edges:
                lines.append(f"COUPLING {edges}")
            elif self._device_spec.split()[0] not in ('all', 'full'):
                lines.append("COUPLING OFF")
        else:
            spec = (lm.get('saved_noise') or (None, None))[1] if lm else self._noise_spec
            if spec:
                lines.append(f"NOISE {spec}")
            if edges:
                lines.append(f"COUPLING {edges}")
        if self._basis_gates:
            lines.append("BASIS " + ', '.join(self._basis_gates))
        if lm:
            if lm['code'] == 'STEANE':
                lines.append(f"LQUBITS {lm['n']} CODE STEANE PHYS {lm['p']:g}")
            else:
                mode = (' ENCODED' if lm.get('force_encoded')
                        else '' if lm.get('encoded') else ' MODEL')
                lines.append(f"LQUBITS {lm['n']} CODE {lm['code']} {lm['d']} "
                             f"PHYS {lm['p']:g}{mode}")
        return lines

    def _load_bas(self, path: str) -> int:
        """Load QBASIC source (.bas): every non-blank line is a program line;
        numbered lines keep their numbers and the rest are numbered to fit
        around them. Returns the number of lines loaded."""
        with open(path, 'r', encoding='utf-8-sig') as f:
            numbered = number_source(f.read().splitlines())
        self.cmd_new(silent=True)
        for num, text in numbered:
            self.process(f"{num} {text}", track_undo=False)
        return len(numbered)

    def cmd_load(self, rest: str) -> None:
        """LOAD <filename> — clear state and load a .qb program file (or QBASIC
        source, .bas, whose unnumbered lines are numbered)."""
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
        if path.lower().endswith('.bas'):
            try:
                count = self._load_bas(path)
            except (ValueError, OSError) as e:
                self.io.writeln(f"?LOAD ERROR: {e}")
                return
            self.io.writeln(f"LOADED {path} ({count} lines)")
            return
        try:
            prev_qubits = self.num_qubits
            prev_shots = self.shots
            self.cmd_new(silent=True)
            with open(path, 'r', encoding='utf-8-sig') as f:
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
        path = self._find_qb_file(path)
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
            with open(path, 'r', encoding='utf-8-sig') as f:
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
        path = rest.strip().strip('"').strip("'")
        if not path:
            self.io.writeln('?USAGE: IMPORT "filename"')
            return
        try:
            path = self._sanitize_path(path)
        except ValueError as e:
            self.io.writeln(f"?IMPORT ERROR: {e}")
            return
        path = self._find_qb_file(path)
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        # Module name from filename (without extension)
        mod_name = os.path.splitext(os.path.basename(path))[0].upper()
        # Parse the file for DEF statements
        import_count = 0
        # Reserved high range for injected SUB/FUNCTION lines, tracked so SAVE
        # and listings can exclude them.
        _import_line_counter = 900000
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.rstrip('\n\r').strip()
                if not line or line.startswith('#'):
                    continue
                # Look for DEF name = body
                upper = line.upper()
                if upper.startswith('DEF ') and '=' in line and not upper.startswith('DEF BEGIN'):
                    from qubasic_core.engine import RE_DEF_SINGLE
                    dm = RE_DEF_SINGLE.match(line[4:].strip())
                    if dm:
                        name = dm.group(1).upper()
                        params = [p.strip() for p in dm.group(2).split(',')] if dm.group(2) else []
                        body = [s.strip() for s in dm.group(3).split(':') if s.strip()]
                        qualified = f"{mod_name}.{name}"
                        self.subroutines[qualified] = {'body': body, 'params': params}
                        import_count += 1
                # SUB/FUNCTION must live in self.program for _scan_subs and CALL
                # to find them, so inject into a reserved high range and record
                # the line numbers as imported (excluded from SAVE).
                elif upper.startswith('SUB ') or upper.startswith('FUNCTION '):
                    ln = _import_line_counter
                    while ln in self.program:
                        ln += 10
                    _import_line_counter = ln + 10
                    self.process(f"{ln} {line}", track_undo=False)
                    self._imported_lines.add(ln)
                    import_count += 1
                elif upper in ('END SUB', 'END FUNCTION') and import_count > 0:
                    ln = _import_line_counter
                    while ln in self.program:
                        ln += 10
                    _import_line_counter = ln + 10
                    self.process(f"{ln} {line}", track_undo=False)
                    self._imported_lines.add(ln)
                    import_count += 1
        self.io.writeln(f"IMPORTED {mod_name} ({import_count} definitions from {path})")

    def cmd_loadqasm(self, rest: str) -> None:
        """LOADQASM <file> — import an OpenQASM file as an editable program.

        Parses OpenQASM 3 (falling back to 2.0) into a circuit and translates it
        into numbered QUBASIC gate lines, so the result can be LISTed, edited, and
        re-RUN. Gates outside the built-in set are imported as UNITARY blocks when
        a matrix is available, otherwise reported and skipped. Reverses EXPORT."""
        if not rest.strip():
            self.io.writeln("?USAGE: LOADQASM <file.qasm>")
            return
        try:
            path = self._sanitize_path(rest.strip())
        except ValueError as e:
            self.io.writeln(f"?LOADQASM ERROR: {e}")
            return
        if not os.path.isfile(path) and os.path.isfile(path + '.qasm'):
            path += '.qasm'
        if not os.path.isfile(path):
            self.io.writeln(f"?FILE NOT FOUND: {path}")
            return
        text = open(path, 'r', encoding='utf-8-sig').read()
        qc = None
        errors = []
        for loader, mod in (('loads', 'qiskit.qasm3'), ('loads', 'qiskit.qasm2')):
            try:
                m = __import__(mod, fromlist=[loader])
                qc = getattr(m, loader)(text)
                break
            except Exception as e:
                errors.append(f"{mod}: {e}")
        if qc is None:
            self.io.writeln("?LOADQASM: could not parse as OpenQASM 3 or 2")
            for e in errors:
                self.io.writeln(f"  {e}")
            if any('qiskit_qasm3_import' in e for e in errors):
                self.io.writeln("  (3.0 import needs the optional package: "
                                "pip install qubasic[qasm3])")
            return
        lines, n_unknown, custom = self._circuit_to_qb(qc)
        self.cmd_new(silent=True)
        self.num_qubits = qc.num_qubits
        for name, mat in custom.items():
            self._custom_gates[name] = mat
        ln = 10
        for stmt in lines:
            self.program[ln] = stmt
            self._parsed[ln] = parse_stmt(stmt)
            ln += 10
        self.io.writeln(f"LOADQASM {path}: {qc.num_qubits} qubits, {len(lines)} lines")
        if n_unknown:
            self.io.writeln(f"  ({n_unknown} non-standard gate(s) imported as UNITARY)")

    def _circuit_to_qb(self, qc) -> tuple[list[str], int, dict]:
        """Translate a QuantumCircuit into QUBASIC gate lines.

        Returns (lines, n_unknown, custom_gates). Standard gates map to their
        QUBASIC form; unknown gates with a matrix become UNITARY definitions.
        """
        lines: list[str] = []
        custom: dict = {}
        n_unknown = 0
        has_measure = False
        for inst in qc.data:
            op = inst.operation
            nm = op.name.lower()
            qidx = [qc.find_bit(q).index for q in inst.qubits]
            if nm == 'measure':
                has_measure = True
                continue
            if nm == 'barrier':
                lines.append('BARRIER')
                continue
            if nm == 'reset':
                lines.append(f'RESET {qidx[0]}')
                continue
            canonical = GATE_ALIASES.get(nm.upper(), nm.upper())
            if canonical in GATE_TABLE:
                n_params = GATE_TABLE[canonical][0]
                params = [repr(float(p)) for p in op.params[:n_params]]
                args = params + [str(q) for q in qidx]
                lines.append(f"{canonical} {', '.join(args)}")
            else:
                try:
                    import numpy as _np
                    mat = _np.asarray(op.to_matrix(), dtype=complex)
                    gname = f"G{len(custom)}"
                    custom[gname] = mat
                    lines.append(f"{gname} {', '.join(str(q) for q in qidx)}")
                    n_unknown += 1
                except Exception:
                    lines.append(f"REM (skipped unsupported gate: {nm})")
                    n_unknown += 1
        if has_measure:
            lines.append('MEASURE')
        return lines, n_unknown, custom

    def cmd_savepng(self, rest: str) -> None:
        """SAVEPNG <file> [hist|bloch|circuit] — save a plot as a PNG image.

        'hist' renders the last measurement histogram, 'bloch' the multi-qubit
        Bloch spheres of the current statevector, and 'circuit' the circuit
        diagram. Defaults to the histogram when results exist, else the circuit.
        Requires matplotlib (circuit diagrams also need pylatexenc)."""
        parts = rest.split()
        if not parts:
            self.io.writeln("?USAGE: SAVEPNG <file> [hist|bloch|circuit]")
            return
        try:
            path = self._sanitize_path(parts[0])
        except ValueError as e:
            self.io.writeln(f"?SAVEPNG ERROR: {e}")
            return
        checked = self._check_agent_path(path, "SAVEPNG")
        if checked is None:
            return
        path = checked
        if not path.lower().endswith('.png'):
            path += '.png'
        what = parts[1].lower() if len(parts) > 1 else ('hist' if self.last_counts else 'circuit')
        try:
            import matplotlib
            matplotlib.use('Agg')  # headless, no display needed
        except ImportError:
            self.io.writeln("?SAVEPNG requires matplotlib (pip install matplotlib)")
            return
        try:
            if what in ('hist', 'histogram'):
                if not self.last_counts:
                    self.io.writeln("?NO RESULTS — RUN with MEASURE first")
                    return
                from qiskit.visualization import plot_histogram
                fig = plot_histogram(self.last_counts)
            elif what == 'bloch':
                sv = self._active_sv
                if sv is None:
                    self.io.writeln("?NO STATE — RUN first")
                    return
                from qiskit.visualization import plot_bloch_multivector
                fig = plot_bloch_multivector(np.ascontiguousarray(sv).ravel())
            elif what == 'circuit':
                if self.last_circuit is None:
                    self.io.writeln("?NO CIRCUIT — RUN first")
                    return
                try:
                    fig = self.last_circuit.draw('mpl')
                except Exception as e:
                    self.io.writeln(f"?circuit image needs pylatexenc (pip install pylatexenc): {e}")
                    return
            else:
                self.io.writeln("?SAVEPNG type must be hist, bloch, or circuit")
                return
            fig.savefig(path, bbox_inches='tight', dpi=120)
            import matplotlib.pyplot as plt
            plt.close(fig)
            self.io.writeln(f"SAVED {what} image to {path}")
        except Exception as e:
            self.io.writeln(f"?SAVEPNG ERROR: {e}")

    def cmd_dir(self, rest: str = '') -> None:
        """DIR [path] — list the .qb files in a directory."""
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
        """EXPORT [filename] — export circuit as OpenQASM (2.0, else 3.0)."""
        if self.last_circuit is None:
            self.io.writeln("?NO CIRCUIT — RUN first")
            return
        qasm = None
        version = None
        errors = []
        # QASM 2.0 first so the output round-trips through LOADQASM without
        # the optional qiskit_qasm3_import package; 3.0 covers dynamic
        # circuits (if_test) and anything else 2.0 cannot express.
        try:
            from qiskit.qasm2 import dumps as _dumps2
            qasm = _dumps2(self.last_circuit)
            version = '2.0'
        except Exception as e:
            errors.append(f"qasm2: {e}")
        if qasm is None:
            try:
                from qiskit.qasm3 import dumps as _dumps3
                qasm = _dumps3(self.last_circuit)
                version = '3.0'
            except Exception as e:
                errors.append(f"qasm3: {e}")
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
            self.io.writeln(f"EXPORTED to {path} (OpenQASM {version})")
        else:
            self.io.writeln(qasm)

    def cmd_csv(self, rest: str) -> None:
        """CSV [filename] — export last results to CSV.

        Includes measurement counts, and statevector amplitudes if available.
        """
        sv = self._export_state()
        if self.last_counts is None and sv is None:
            self.io.writeln("?NO RESULTS — RUN first")
            return
        lines = []
        if self.last_counts:
            total = sum(self.last_counts.values())
            lines.append("state,count,probability")
            for state, count in sorted(self.last_counts.items(), key=lambda x: -x[1]):
                lines.append(f"{self._bits(state)},{count},{count/total:.6f}")
        if sv is not None:
            lines.append("")
            lines.append("state,amplitude_re,amplitude_im,probability")
            lines.extend(self._amplitude_rows(sv))
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
            what = f"{len(self.last_counts)} states" if self.last_counts else "statevector"
            self.io.writeln(f"EXPORTED {what} to {path}")
        else:
            for line in lines[:20]:
                self.io.writeln(f"  {line}")
            if len(lines) > 20:
                self.io.writeln(f"  ... ({len(lines)-20} more)")

    # Amplitude rows written by CSV/CLIP: every non-zero amplitude up to this
    # many, else this many of the most probable ones.
    MAX_EXPORT_AMPLITUDES = 1 << 20

    def _export_state(self):
        """The state CSV/CLIP export: one already in memory, or (for up to 16
        qubits) one computed on demand; a large deferred state is skipped."""
        if self._last_sv is not None:
            return self._last_sv
        if self.num_qubits <= 16:
            return self.last_sv
        return None

    def _amplitude_rows(self, sv, sep: str = ',') -> list[str]:
        """'state,re,im,p' rows for non-zero amplitudes in OPTION ENDIAN order."""
        from qubasic_core.statetools import nonzero_indices, top_k
        flat = np.asarray(sv).reshape(-1)
        n = int(flat.size).bit_length() - 1
        idx, total = nonzero_indices(flat, 1e-4, self.MAX_EXPORT_AMPLITUDES)
        note = []
        if total > idx.size:
            idx, _, _ = top_k(flat, self.MAX_EXPORT_AMPLITUDES, 1e-4)
            idx = np.sort(idx)
            note = [f"# {total} non-zero amplitudes; the {idx.size} most probable are listed"]
        amps = flat[idx]
        rows = [f"{self._bits(format(int(i), f'0{n}b'))}{sep}{a.real:.8f}{sep}{a.imag:.8f}"
                f"{sep}{abs(a) ** 2:.8f}" for i, a in zip(idx, amps)]
        return note + rows

    # ── File handles: OPEN, CLOSE, PRINT#, INPUT#, EOF, LPRINT ────────

    def _init_file_handles(self) -> None:
        self._file_handles: dict[int, Any] = {}
        # Values read from a line by INPUT # and not yet assigned, per handle.
        self._file_fields: dict[int, list[str]] = {}
        self._lprint_path: str | None = None
        self._imported_lines: set[int] = set()

    def _qb_search_dirs(self) -> list[str]:
        """Directories searched for .qb files: cwd then QUBASIC_PATH entries."""
        dirs = ['.']
        env = os.environ.get('QUBASIC_PATH', '')
        if env:
            dirs += [d for d in env.split(os.pathsep) if d]
        return dirs

    def _find_qb_file(self, path: str) -> str:
        """Resolve a sanitized relative .qb path against cwd then the library
        search path. Returns the first existing match, else the cwd candidate."""
        candidates = [path] + ([path + '.qb'] if not path.endswith('.qb') else [])
        for d in self._qb_search_dirs():
            for c in candidates:
                full = c if d == '.' else os.path.join(d, c)
                if os.path.isfile(full):
                    return full
        return candidates[-1]

    def _check_agent_path(self, path: str, cmd: str) -> str | None:
        """In agent mode, restrict paths to cwd. Returns resolved path or None on error.

        Uses pathlib is_relative_to for safe containment check instead of
        string prefix matching, which can be bypassed via symlinks or
        crafted path components that share a common prefix string.
        """
        if not self.agent_mode:
            return path
        resolved = Path(os.path.realpath(os.path.join(os.getcwd(), path)))
        cwd_real = Path(os.path.realpath(os.getcwd()))
        if not resolved.is_relative_to(cwd_real):
            self.io.writeln(f"?{cmd} BLOCKED: path escapes working directory in agent mode")
            return None
        return str(resolved)

    def _open_file(self, rest: str) -> tuple[int, str, str]:
        """Open a file handle for `OPEN file FOR mode AS #n [ENCODING "enc"]`;
        returns (handle, path, mode) or raises with the QBASIC error."""
        m = RE_OPEN.match(f"OPEN {rest}")
        if not m:
            raise ValueError('SYNTAX ERROR: OPEN "file" FOR INPUT|OUTPUT|APPEND|RANDOM AS #n')
        path, mode_str, handle = m.group(1).strip(), m.group(2).upper(), int(m.group(3))
        encoding = m.group(4).strip() if m.group(4) else 'utf-8'
        if handle in self._file_handles:
            raise RuntimeError(f"FILE ALREADY OPEN: #{handle}")
        path = self._sanitize_path(path)
        checked = self._check_agent_path(path, "OPEN")
        if checked is None:
            raise RuntimeError(f"PERMISSION DENIED: {path} is outside the working directory")
        path = checked
        mode = {'INPUT': 'r', 'OUTPUT': 'w', 'APPEND': 'a', 'RANDOM': 'r+'}.get(mode_str, 'r')
        if mode_str == 'INPUT' and not os.path.isfile(path):
            raise RuntimeError(f"FILE NOT FOUND: {path}")
        binary = encoding.lower() == 'binary'
        if mode_str == 'RANDOM' and not os.path.isfile(path):
            open(path, 'wb').close()
        if binary:
            self._file_handles[handle] = open(path, mode + 'b')
        else:
            self._file_handles[handle] = open(path, mode, encoding=encoding)
        return handle, path, mode_str

    def cmd_open(self, rest: str) -> None:
        """OPEN "file" FOR INPUT|OUTPUT|APPEND|RANDOM AS #n [ENCODING "enc"] — open a file handle."""
        try:
            handle, path, mode_str = self._open_file(rest)
        except Exception as e:
            self.io.writeln(f"?OPEN ERROR: {e}")
            return
        self.io.writeln(f"OPENED #{handle} ({path}, {mode_str})")

    def _close_file(self, handle: int) -> bool:
        f = self._file_handles.pop(handle, None)
        self._file_fields.pop(handle, None)
        if f is None:
            return False
        f.close()
        return True

    def cmd_close(self, rest: str) -> None:
        """CLOSE [#n] — close a file handle (every handle with no number)."""
        if not rest.strip():
            for h in list(self._file_handles):
                self._close_file(h)
            self.io.writeln("CLOSED ALL")
            return
        m = RE_CLOSE.match(f"CLOSE {rest}")
        if not m:
            self.io.writeln("?USAGE: CLOSE [#n]")
            return
        handle = int(m.group(1))
        if self._close_file(handle):
            self.io.writeln(f"CLOSED #{handle}")
        else:
            self.io.writeln(f"?HANDLE #{handle} NOT OPEN")

    def _writable(self, handle: int):
        """The open file for handle, or None after reporting why it cannot
        be written."""
        if handle not in self._file_handles:
            self.io.writeln(f"?HANDLE #{handle} NOT OPEN")
            return None
        f = self._file_handles[handle]
        fmode = getattr(f, 'mode', '')
        if fmode.startswith('r') and '+' not in fmode:
            self.io.writeln(f"?HANDLE #{handle} NOT OPEN FOR WRITING")
            return None
        return f

    def _write_to_handle(self, handle: int, text: str, newline: bool = True) -> bool:
        if getattr(self.io, 'replay', False):
            return True     # later live shots write nothing; the first shot did
        f = self._writable(handle)
        if f is not None:
            f.write(text + ('\n' if newline else ''))
            f.flush()
        return True

    def _exec_print_file(self, stmt: str, run_vars: dict[str, Any]) -> bool:
        """PRINT #n, list — what PRINT would show, written to the file
        (PRINT #n, USING fmt$; values formats as PRINT USING)."""
        m = RE_PRINT_FILE.match(stmt)
        if not m:
            return False
        handle = int(m.group(1))
        data = m.group(2).strip()
        um = re.match(r'USING\s+(.+)$', data, re.IGNORECASE)
        if um:
            from qubasic_core.parser import _top_split
            head, *tail = _top_split(um.group(1), ';')
            text, newline = self._using_text(head.strip(), ';'.join(tail), run_vars)
        else:
            text, newline = self._format_print(data, run_vars)
        return self._write_to_handle(handle, text, newline)

    def _readable(self, handle: int):
        if handle not in self._file_handles:
            raise RuntimeError(f"BAD FILE NAME OR NUMBER: #{handle} is not open")
        f = self._file_handles[handle]
        fmode = getattr(f, 'mode', '')
        if fmode.startswith(('w', 'a')) and '+' not in fmode:
            raise RuntimeError(f"BAD FILE MODE: #{handle} is not open for input")
        return f

    def _exec_input_file(self, stmt: str, run_vars: dict[str, Any]) -> bool:
        """INPUT #n, var[, var ...] — read comma- or line-separated values;
        reading past the end of the file is an error, as in QBASIC (test
        EOF(n) first)."""
        m = re.match(r'INPUT\s*#\s*(\d+)\s*,\s*(.+)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        handle = int(m.group(1))
        f = self._readable(handle)
        pending = self._file_fields.setdefault(handle, [])
        from qubasic_core.stmt_handlers import _input_fields, _parse_input_number
        for target in self._split_arg_list(m.group(2)):
            lv = self._lvalue(target, run_vars)
            while not pending:
                line = f.readline()
                if not line:
                    raise RuntimeError("INPUT PAST END OF FILE")
                pending.extend(_input_fields(line.rstrip('\r\n')))
            field = pending.pop(0)
            if is_string_kind(self._lvalue_kind(lv)):
                value: Any = field
            else:
                value = _parse_input_number(field.strip()) if field.strip() else 0
                if value is None:
                    from qubasic_core.strings import _val_fn
                    value = _val_fn(field)
            self._write_lvalue(lv, value, run_vars)
        return True

    def _exec_line_input_file(self, stmt: str, run_vars: dict[str, Any]) -> bool:
        """LINE INPUT #n, var$ — the next whole line of the file."""
        m = re.match(r'LINE\s+INPUT\s*#\s*(\d+)\s*,\s*(.+)$', stmt.strip(), re.IGNORECASE)
        if not m:
            return False
        handle = int(m.group(1))
        f = self._readable(handle)
        self._file_fields.pop(handle, None)
        line = f.readline()
        if not line:
            raise RuntimeError("INPUT PAST END OF FILE")
        self._write_lvalue(self._lvalue(m.group(2), run_vars), line.rstrip('\r\n'), run_vars)
        return True

    def _freefile(self) -> int:
        """FREEFILE: the lowest file number not in use."""
        n = 1
        while n in self._file_handles:
            n += 1
        return n

    def _lof(self, handle: Any) -> int:
        """LOF(n): length of the open file in bytes."""
        h = int(handle)
        if h not in self._file_handles:
            raise RuntimeError(f"BAD FILE NAME OR NUMBER: #{h} is not open")
        f = self._file_handles[h]
        try:
            f.flush()
        except Exception:
            pass
        return os.fstat(f.fileno()).st_size

    def _exec_lprint(self, stmt: str, run_vars: dict[str, Any]) -> bool:
        """Handle LPRINT data — output to log file or stderr."""
        m = RE_LPRINT.match(stmt)
        if not m:
            return False
        if getattr(self.io, 'replay', False):
            return True
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
            self.io.writeln(text)
        return True

    def _eof(self, handle: float) -> int:
        """EOF(n) — -1 (true) at the end of the file, else 0."""
        h = int(handle)
        if h not in self._file_handles:
            return -1
        if self._file_fields.get(h):
            return 0
        f = self._file_handles[h]
        pos = f.tell()
        try:
            ch = f.read(1)
            if not ch:
                return -1
            f.seek(pos)
        except (UnicodeDecodeError, Exception):
            return -1
        return 0
