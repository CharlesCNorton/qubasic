"""Allow running qbasic_core as a module: python -m qbasic_core."""

from qbasic_core.terminal import QBasicTerminal

if __name__ == '__main__':
    QBasicTerminal().repl()
