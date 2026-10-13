"""Regenerate the files written from the code: HostState's method and
property stubs (qubasic_core/host.py) and docs/reference.md.

HostState declares everything QBasicTerminal's classes share. Its attribute
section is written by hand; the method and property sections are rewritten
here, one stub per method copied from the class that defines it first in
QBasicTerminal's MRO (decorators that change the signature kept, default
values written as ...), and names the stubs need that host.py does not
import are added to its TYPE_CHECKING block.

The reference lists every command (from the command tables and the syntax
line that opens each command's docstring), every statement and function
(the catalog in qubasic_core/parser.py), the gate table, the constants and
the memory map, so it changes when they do.

usage: python tools/generate.py [--check]

--check writes nothing and exits 1 when either file is out of date.
"""

from __future__ import annotations

import ast
import builtins
import copy
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HOST = os.path.join(ROOT, 'qubasic_core', 'host.py')
REFERENCE = os.path.join(ROOT, 'docs', 'reference.md')
NL = '\n'
IND = ' ' * 8
METHODS = IND + '# Methods, defined across the mixins and resolved through the MRO'
PROPERTIES = IND + '# Properties'
KEEP_DECORATORS = {'staticmethod', 'classmethod', 'contextmanager'}
TRANSPARENT = {'functools.lru_cache', 'lru_cache', 'functools.cache', 'cache'}


# ── HostState stubs ──────────────────────────────────────────────────────

def classes() -> list[type]:
    from qubasic_core.terminal import QBasicTerminal
    return [c for c in QBasicTerminal.__mro__ if c is not object and c.__name__ != 'HostState']


_trees: dict[str, ast.Module] = {}


def module_tree(name: str) -> ast.Module:
    if name not in _trees:
        path = importlib.import_module(name).__file__
        assert path is not None
        _trees[name] = ast.parse(open(path, encoding='utf-8').read())
    return _trees[name]


def class_node(cls: type) -> ast.ClassDef:
    for node in module_tree(cls.__module__).body:
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            return node
    raise LookupError(cls)


def decorator_names(fn: ast.FunctionDef) -> list[str]:
    return [ast.unparse(d.func if isinstance(d, ast.Call) else d) for d in fn.decorator_list]


def stub(fn: ast.FunctionDef, decorators: list[str]) -> list[str]:
    args = copy.deepcopy(fn.args)
    args.defaults = [ast.Constant(value=...) for _ in args.defaults]
    args.kw_defaults = [None if d is None else ast.Constant(value=...) for d in args.kw_defaults]
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns is not None else ''
    return [IND + '@' + d for d in decorators] + [f"{IND}def {fn.name}({ast.unparse(args)}){ret}: ..."]


def annotation_names(fn: ast.FunctionDef) -> set[str]:
    nodes: list[ast.AST] = [a.annotation for a in fn.args.posonlyargs + fn.args.args
                            + fn.args.kwonlyargs if a.annotation is not None]
    for extra in (fn.args.vararg, fn.args.kwarg):
        if extra is not None and extra.annotation is not None:
            nodes.append(extra.annotation)
    if fn.returns is not None:
        nodes.append(fn.returns)
    names: set[str] = set()
    while nodes:
        node = nodes.pop()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                try:
                    nodes.append(ast.parse(sub.value, mode='eval'))
                except SyntaxError:
                    pass
    return names


def bindings(module: str) -> dict[str, str]:
    """Each top-level name of a module -> the import line that binds it in host.py."""
    out: dict[str, str] = {}

    def scan(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    alias = a.asname or a.name.split('.')[0]
                    out[alias] = f"import {a.name}" + (f" as {a.asname}" if a.asname else '')
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = node.module
                if node.level:
                    base = module.rsplit('.', node.level)[0] + '.' + node.module
                for a in node.names:
                    out[a.asname or a.name] = (f"from {base} import {a.name}"
                                               + (f" as {a.asname}" if a.asname else ''))
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                out.setdefault(node.name, f"from {module} import {node.name}")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        out.setdefault(t.id, f"from {module} import {t.id}")
            elif isinstance(node, ast.If):
                scan(node.body)
                scan(node.orelse)
            elif isinstance(node, ast.Try):
                scan(node.body)
    scan(module_tree(module).body)
    return out


def host_imports(text: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            names |= {a.asname or a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names}
    return names


def stubs() -> tuple[list[str], list[str], dict[str, str], list[str]]:
    method_lines: list[str] = []
    prop_lines: list[str] = []
    needed: dict[str, str] = {}
    problems: list[str] = []
    seen: set[str] = set()
    entries: list[tuple[str, type, list[ast.FunctionDef]]] = []
    for cls in classes():
        cnode = class_node(cls)
        for name, value in cls.__dict__.items():
            if name.startswith('__') or name in seen:
                continue
            if not (callable(value) or isinstance(value, (property, staticmethod, classmethod))):
                continue
            defs = [n for n in cnode.body if isinstance(n, ast.FunctionDef) and n.name == name]
            if not defs:
                continue                 # assigned, not defined (a table of callables)
            seen.add(name)
            entries.append((name, cls, defs))
    for name, cls, defs in sorted(entries, key=lambda e: e[0]):
        is_prop = any('property' in d or d.endswith('.setter') for fn in defs
                      for d in decorator_names(fn))
        target = prop_lines if is_prop else method_lines
        for fn in defs:
            decos = decorator_names(fn)
            if is_prop:
                kept = decos[:1]
            else:
                kept = [d for d in decos if d in KEEP_DECORATORS]
                other = [d for d in decos if d not in KEEP_DECORATORS and d not in TRANSPARENT]
                if other:
                    problems.append(f"{cls.__name__}.{name}: decorator {other} not modelled")
            target.extend(stub(fn, kept))
            table = bindings(cls.__module__)
            for n in annotation_names(fn) | {d for d in kept if d == 'contextmanager'}:
                if n in vars(builtins):
                    continue
                if n not in table:
                    problems.append(f"{cls.__name__}.{name}: cannot resolve {n!r}")
                    continue
                if needed.setdefault(n, table[n]) != table[n]:
                    problems.append(f"{n!r} bound two ways: {needed[n]} / {table[n]}")
    return method_lines, prop_lines, needed, problems


def host_text() -> tuple[str, str]:
    """(current host.py, regenerated host.py)."""
    text = open(HOST, encoding='utf-8').read()
    lines = text.split(NL)
    m0 = lines.index(METHODS) + 1
    end = lines.index('else:')
    methods, props, needed, problems = stubs()
    new_lines = lines[:m0] + methods + [''] + [PROPERTIES] + props + lines[end:]
    have = host_imports(text)
    missing = sorted({line for name, line in needed.items() if name not in have})
    if missing:
        anchor = new_lines.index('if TYPE_CHECKING:') + 1
        new_lines[anchor:anchor] = ['    ' + ln for ln in missing]
    for p in problems:
        print('note:', p)
    return text, NL.join(new_lines)


# ── docs/reference.md ────────────────────────────────────────────────────

def _summary(doc: str) -> tuple[str, str]:
    """(syntax, description) from a docstring's first paragraph."""
    para = []
    for line in (doc or '').strip().splitlines():
        if not line.strip():
            break
        para.append(line.strip())
    text = ' '.join(para)
    for sep in (' — ', ' - '):
        if sep in text:
            syntax, _, desc = text.partition(sep)
            return syntax.strip(), _sentence(desc)
    return '', _sentence(text)


def _sentence(text: str) -> str:
    """A description with its first letter capitalized and one closing period."""
    text = text.strip().rstrip('.')
    return (text[:1].upper() + text[1:] + '.') if text else ''


def _escape(text: str) -> str:
    return text.replace('|', '\\|')


def build_reference() -> str:
    from qubasic_core import machine
    from qubasic_core import parser as catalog
    from qubasic_core.expression import ExpressionMixin
    from qubasic_core.parser import GATE_ALIASES, GATE_TABLE
    from qubasic_core.terminal import QBasicTerminal

    out = ['# QUBASIC reference', '',
           'This file is generated by `tools/generate.py` from the command docstrings, the',
           'statement and function catalog (`qubasic_core/parser.py`), the gate table and the',
           'memory map; `HELP <name>` prints the same entries at the prompt.', '',
           'Syntax uses `[optional]` and `a | b` for alternatives; lower-case words and `<names>`',
           'stand for values you supply.',
           'Commands are typed at the prompt (most also work as program lines); statements are',
           'program lines, and most can be typed at the prompt as well.', '']

    # Commands, by topic (catalog.COMMAND_TOPICS names every command once)
    tables = {**QBasicTerminal._CMD_NO_ARG, **QBasicTerminal._CMD_WITH_ARG}
    placed = [name for _, names in catalog.COMMAND_TOPICS for name in names]
    if sorted(placed) != sorted(tables):
        missing = sorted(set(tables) - set(placed))
        extra = sorted(set(placed) - set(tables))
        doubled = sorted({n for n in placed if placed.count(n) > 1})
        raise SystemExit(f'COMMAND_TOPICS is out of step: missing {missing}, unknown {extra}, '
                         f'listed twice {doubled}')
    out += ['## Commands', '']
    for title, names in catalog.COMMAND_TOPICS:
        out += [f'### {title}', '', '| Command | Syntax | Description |', '|---|---|---|']
        for name in names:
            syntax, desc = _summary(getattr(QBasicTerminal, tables[name]).__doc__ or '')
            out.append(f'| {name} | `{_escape(syntax or name)}` | {_escape(desc)} |')
        out.append('')

    # Statements and functions
    for heading, rows in (('Statements', catalog.STATEMENTS), ('Functions', catalog.FUNCTIONS)):
        out += [f'## {heading}', '']
        topics: dict[str, list[tuple[str, str]]] = {}
        for topic, _, syntax, desc in rows:
            topics.setdefault(topic, []).append((syntax, desc))
        for topic, items in topics.items():
            out += [f'### {topic}', '', '| Syntax | Description |', '|---|---|']
            out += [f'| `{_escape(s)}` | {_escape(_sentence(d))} |' for s, d in items]
            out.append('')

    # Gates
    aliases: dict[str, list[str]] = {}
    for alias, canonical in GATE_ALIASES.items():
        aliases.setdefault(canonical, []).append(alias)
    out += ['## Gates', '', 'Parameters come first, then qubits: `RX theta, q`, `CRZ theta, c, t`, '
            '`U theta, phi, lambda, q`.', '',
            '| Gate | Parameters | Qubits | Aliases |', '|---|---|---|---|']
    for gate in sorted(g for g in GATE_TABLE if g not in GATE_ALIASES):
        n_params, n_qubits = GATE_TABLE[gate]
        out.append(f"| {gate} | {n_params} | {n_qubits} | {', '.join(sorted(aliases.get(gate, [])))} |")
    out.append('')

    consts = sorted({k.upper() for k in ExpressionMixin._SAFE_CONSTS})
    out += ['## Constants', '', ', '.join(f'`{c}`' for c in consts) +
            ' (TRUE is -1 in comparisons, as in BASIC).', '']

    # Memory map
    out += ['## Memory map', '', '| Address | Contents |', '|---|---|',
            '| `$0000-$003F` | zero page: 64 slots for SYS parameters |',
            '| `$0100-$01FF` | qubit q (0-31) at `$0100 + 8*q`: +0 P(1), +1..+3 Bloch x, y, z, '
            '+4..+7 Re/Im of alpha and beta |',
            '| `$D100 + 2*q` | qubit q noise: +0 type (0 none, 1 depolarizing, 2 amplitude damping, '
            '3 phase flip), +1 parameter |',
            '| `$F000-$FFFF` | user SYS routines (SYS INSTALL) |', '']
    out += ['| Config (read/write) | Name |', '|---|---|']
    out += [f'| `${addr:04X}` | {name} |' for addr, name in sorted(machine.CFG_NAMES.items())]
    out += ['', 'Simulation method codes at `$D002`: ' +
            ', '.join(f'{k} {v}' for k, v in sorted(machine.SIM_METHODS_FWD.items())) + '.', '']
    out += ['| Status (read-only) | Name |', '|---|---|']
    out += [f'| `${addr:04X}` | {name} |' for addr, name in sorted(machine.STS_NAMES.items())]
    out += ['', '| SYS routine | Program |', '|---|---|']
    out += [f'| `${addr:04X}` | {name} |' for addr, name in sorted(machine.SYS_ROUTINES.items())]
    out.append('')
    return NL.join(out)


def main() -> int:
    check = '--check' in sys.argv[1:]
    current_host, new_host = host_text()
    current_ref = open(REFERENCE, encoding='utf-8').read() if os.path.exists(REFERENCE) else ''
    new_ref = build_reference()
    stale = [label for label, cur, new in (('qubasic_core/host.py', current_host, new_host),
                                           ('docs/reference.md', current_ref, new_ref)) if cur != new]
    if check:
        for label in stale:
            print(f'{label} is out of date: run python tools/generate.py')
        return 1 if stale else 0
    for path, cur, new in ((HOST, current_host, new_host), (REFERENCE, current_ref, new_ref)):
        if cur != new:
            with open(path, 'w', encoding='utf-8', newline=NL) as f:
                f.write(new)
    print('up to date' if not stale else 'wrote ' + ', '.join(stale))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
