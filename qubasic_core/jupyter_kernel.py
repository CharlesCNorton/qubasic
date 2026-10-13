"""QUBASIC Jupyter kernel — run .qb cells in a notebook.

Install the kernelspec once, then pick "QUBASIC" in Jupyter:

    qubasic --install-kernel
    jupyter lab

Each cell's lines feed the same QBasicTerminal the REPL uses (numbered lines
build the program, bare commands execute immediately), so a notebook is an
interactive session with persistent state. Requires ipykernel (installed with
`pip install qubasic[jupyter]`).
"""

from __future__ import annotations

import sys
from typing import Any


def _make_kernel_class() -> type:
    from ipykernel.kernelbase import Kernel

    from qubasic_core import __version__
    from qubasic_core.editing import ProgramMgmtMixin
    from qubasic_core.terminal import QBasicTerminal

    class QubasicKernel(Kernel):
        implementation = 'qubasic'
        implementation_version = __version__
        language = 'qubasic'
        language_version = __version__
        language_info = {
            'name': 'qubasic',
            'mimetype': 'text/plain',
            'file_extension': '.qb',
        }
        banner = f"QUBASIC {__version__} — Quantum BASIC"

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.term = QBasicTerminal()
            self.term.agent_mode = True   # confine file writes to the cwd

        def do_execute(self, code: str, silent: bool, store_history: bool = True,
                       user_expressions: dict[str, Any] | None = None, allow_stdin: bool = False, **kwargs: Any) -> dict[str, Any]:
            from qubasic_core.cli import BufferIO
            kernel = self

            class CellIO(BufferIO):
                """Cell output; INPUT asks through the notebook when the
                frontend allows stdin."""

                def read_line(self, prompt: str) -> str:
                    if not allow_stdin:
                        return super().read_line(prompt)
                    return kernel.raw_input(prompt)

            port = CellIO()
            old_io, old_stdout = self.term.io, sys.stdout
            self.term.io = port
            sys.stdout = port.buf
            status = 'ok'
            try:
                lines = [l.rstrip('\r\n') for l in code.split('\n')]
                ProgramMgmtMixin._load_lines_with_defs(
                    lines, lambda line: self.term.process(line, track_undo=False))
            except Exception as e:
                status = 'error'
                port.writeln(f"?ERROR: {e}")
            finally:
                self.term.io, sys.stdout = old_io, old_stdout
            out = port.buf.getvalue()
            if out and not silent:
                self.send_response(self.iopub_socket, 'stream',
                                   {'name': 'stdout', 'text': out})
            return {'status': status, 'execution_count': self.execution_count,
                    'payload': [], 'user_expressions': {}}

    return QubasicKernel


KERNEL_SPEC = {
    'argv': [sys.executable, '-m', 'qubasic_core.jupyter_kernel', '-f',
             '{connection_file}'],
    'display_name': 'QUBASIC',
    'language': 'qubasic',
}


def install_kernelspec() -> str:
    """Write the QUBASIC kernelspec into the user's Jupyter kernels dir."""
    import json
    import os
    import tempfile

    from jupyter_client.kernelspec import KernelSpecManager
    with tempfile.TemporaryDirectory() as td:
        spec_dir = os.path.join(td, 'qubasic')
        os.mkdir(spec_dir)
        with open(os.path.join(spec_dir, 'kernel.json'), 'w', encoding='utf-8') as f:
            json.dump(KERNEL_SPEC, f, indent=2)
        dest = KernelSpecManager().install_kernel_spec(
            spec_dir, 'qubasic', user=True)
    return dest


def main() -> None:
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=_make_kernel_class())


if __name__ == '__main__':
    main()
