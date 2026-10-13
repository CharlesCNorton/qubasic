"""The project against itself: package layering and the layer hosts, the
documentation (tutorial, reference, catalog) and the example programs."""

import ast
import glob
import inspect
import os
import re
import sys
import tempfile
import unittest

from qubasic_core import parser
from qubasic_core.api import CaptureIO
from qubasic_core.cli import run_script
from qubasic_core.editing import ProgramMgmtMixin
from qubasic_core.host import LAYERS as HOST_LAYERS
from qubasic_core.terminal import QBasicTerminal

# ---------------------------------------------------------------------------
# Architecture: package layering, the layer hosts, command docstrings.
#
# The package is layered (qubasic_core.host.LAYERS):
#
#     core          host, engine
#     language      parser, expression, interpreter, devices, editing
#     simulation    simtools, simulator, noise, locc, physics, machine
#     applications  algorithms, benchmarking, qec, logical
#     presentation  display, analysis
#     composition   terminal, api, cli, jupyter_kernel
#
# Each module imports only from its own layer or the layers below it, at
# module level or inside a function; host.py may name any layer's types in
# its TYPE_CHECKING block. Each mixin derives from its layer's host, which
# declares what that layer and the ones below share, so a call through self
# reaches no higher layer; a hook, declared at the lowest layer that calls
# it, is implemented only above it. No module outside the composition layer
# imports a mixin class.
# ---------------------------------------------------------------------------

PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qubasic_core')

LAYERS = tuple((name, set(members)) for name, members in HOST_LAYERS)
COMPOSITION = len(LAYERS) - 1
HOSTS = ('CoreHost', 'LanguageHost', 'SimulationHost', 'ApplicationsHost', 'PresentationHost', 'HostState')


def host_classes():
    """host.py's layer hosts as parsed classes, lowest layer first."""
    tree = ast.parse(open(os.path.join(PKG_DIR, 'host.py'), encoding='utf-8').read())
    block = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'TYPE_CHECKING')
    found = {n.name: n for n in block.body if isinstance(n, ast.ClassDef)}
    return [found[name] for name in HOSTS]


def modules():
    """(dotted name, path) for every module in the package."""
    out = []
    for root, _, files in os.walk(PKG_DIR):
        for fn in files:
            if fn.endswith('.py'):
                path = os.path.join(root, fn)
                rel = os.path.relpath(path, PKG_DIR)[:-3].replace(os.sep, '.')
                out.append(('qubasic_core.' + rel, path))
    return sorted(out)


def layer_of(dotted):
    name = dotted.split('.')[1] if '.' in dotted else '__init__'
    for rank, (_, members) in enumerate(LAYERS):
        if name in members:
            return rank
    raise AssertionError(f"{dotted} has no layer; add it to LAYERS in qubasic_core/host.py")


def internal_imports(tree, dotted):
    """(line, imported module, inside TYPE_CHECKING) for qubasic_core imports."""
    type_only = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) in ('TYPE_CHECKING',
                                                                    'typing.TYPE_CHECKING'):
            for sub in node.body:
                for inner in ast.walk(sub):
                    type_only.add(id(inner))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            base = node.module
            if node.level:
                pkg = dotted.rsplit('.', node.level)[0]
                base = pkg + ('.' + node.module if node.module else '')
            if not base.startswith('qubasic_core'):
                continue
            if base == 'qubasic_core':
                for a in node.names:
                    target = 'qubasic_core.' + a.name
                    found.append((node.lineno, target, id(node) in type_only))
            else:
                found.append((node.lineno, base, id(node) in type_only))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith('qubasic_core.'):
                    found.append((node.lineno, a.name, id(node) in type_only))
    return found


def resolve(target, known):
    """The module an import names (a package attribute may be a module or a name)."""
    while target not in known and '.' in target:
        target = target.rsplit('.', 1)[0]
    return target


class TestLayering(unittest.TestCase):

    def test_imports_respect_layers(self):
        mods = dict(modules())
        known = set(mods) | {'qubasic_core'}
        violations = []
        for dotted, path in mods.items():
            tree = ast.parse(open(path, encoding='utf-8').read())
            mine = layer_of(dotted)
            for line, target, type_only in internal_imports(tree, dotted):
                target = resolve(target, known)
                if target == 'qubasic_core':
                    continue                       # package metadata (__version__)
                if type_only and dotted == 'qubasic_core.host':
                    continue
                theirs = layer_of(target)
                if theirs > mine:
                    violations.append(f"{dotted}:{line} imports {target}")
        self.assertEqual(violations, [])

    def test_no_import_cycles_at_module_level(self):
        mods = dict(modules())
        known = set(mods)
        graph = {}
        for dotted, path in mods.items():
            tree = ast.parse(open(path, encoding='utf-8').read())
            deps = set()
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    sub = ast.Module(body=[node], type_ignores=[])
                    for _, target, _ in internal_imports(sub, dotted):
                        t = resolve(target, known)
                        if t in known and t != dotted:
                            deps.add(t)
            graph[dotted] = deps
        state, stack, cycles = {}, [], []

        def visit(m):
            state[m] = 1
            stack.append(m)
            for d in graph.get(m, ()):
                if state.get(d) == 1:
                    cycles.append(stack[stack.index(d):] + [d])
                elif d not in state:
                    visit(d)
            stack.pop()
            state[m] = 2

        for m in graph:
            if m not in state:
                visit(m)
        self.assertEqual(cycles, [])

    def test_mixins_meet_only_through_the_host(self):
        offenders = []
        for dotted, path in modules():
            if layer_of(dotted) == COMPOSITION:
                continue
            tree = ast.parse(open(path, encoding='utf-8').read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and \
                        node.module.startswith('qubasic_core'):
                    for a in node.names:
                        if a.name.endswith('Mixin'):
                            offenders.append(f"{dotted} imports {a.name}")
        self.assertEqual(offenders, [])


class TestHostContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from qubasic_core.terminal import QBasicTerminal
        cls.mro = [c for c in QBasicTerminal.__mro__ if c is not object and c.__name__ not in HOSTS]
        cls.stubs, cls.hooks, cls.declared = {}, {}, {}
        for rank, host in enumerate(host_classes()):
            for n in host.body:
                if isinstance(n, ast.FunctionDef):
                    if ast.get_docstring(n) is None:
                        cls.stubs.setdefault(n.name, []).append(n)
                    else:
                        cls.hooks[n.name] = (rank, n)
                    cls.declared.setdefault(n.name, rank)
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    cls.declared.setdefault(n.target.id, rank)

    def owner(self, name):
        return next((c for c in self.mro if name in c.__dict__), None)

    @staticmethod
    def signature_text(fn):
        args = fn.args
        for i in range(len(args.defaults)):
            args.defaults[i] = ast.Constant(value=...)
        args.kw_defaults = [None if d is None else ast.Constant(value=...) for d in args.kw_defaults]
        ret = ast.unparse(fn.returns) if fn.returns is not None else ''
        return ast.unparse(args) + ' -> ' + ret

    def definitions(self, cls, name):
        src = inspect.getsource(cls)
        cnode = ast.parse(src.replace('\n    ', '\n') if src.startswith('    ') else src).body[0]
        return [n for n in cnode.body if isinstance(n, ast.FunctionDef) and n.name == name]

    def test_stubs_and_hooks_match_their_definitions(self):
        mismatches = []
        declared = {name: stubs for name, stubs in self.stubs.items()}
        declared.update({name: [node] for name, (_, node) in self.hooks.items()})
        for name, stubs in declared.items():
            cls = self.owner(name)
            self.assertIsNotNone(cls, f"host.py declares {name}, which no class defines")
            got = [self.signature_text(s) for s in stubs]
            want = [self.signature_text(d) for d in self.definitions(cls, name)]
            if got != want:
                mismatches.append(f"{name}: host {got} vs {cls.__name__} {want}")
        self.assertEqual(mismatches, [], 'run tools/generate.py')

    def test_every_shared_method_is_declared(self):
        from qubasic_core.terminal import QBasicTerminal
        missing = []
        for cls in self.mro:
            if cls is QBasicTerminal:
                continue
            for name, val in cls.__dict__.items():
                if inspect.isfunction(val) or isinstance(val, (staticmethod, classmethod, property)):
                    if not name.startswith('__') and name not in self.declared:
                        missing.append(f"{cls.__name__}.{name}")
        self.assertEqual(missing, [], 'run tools/generate.py')

    def test_each_mixin_derives_from_its_layers_host(self):
        wrong = []
        for dotted, path in modules():
            rank = layer_of(dotted)
            for node in ast.parse(open(path, encoding='utf-8').read()).body:
                if not isinstance(node, ast.ClassDef) or dotted == 'qubasic_core.host':
                    continue
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in HOSTS and base.id != HOSTS[rank]:
                        wrong.append(f"{dotted}.{node.name} derives from {base.id}, not {HOSTS[rank]}")
        self.assertEqual(wrong, [])

    def test_no_layer_reaches_into_a_layer_above(self):
        """Each class reaches through self (and through the first argument of
        the lambdas in its statement tables) only names its own class defines
        or its layer's host declares."""
        reaches = []
        for cls in self.mro:
            rank = layer_of(cls.__module__)
            if rank == COMPOSITION:
                continue
            node = ast.parse(inspect.getsource(cls).replace('\n    ', '\n')
                             if inspect.getsource(cls).startswith('    ') else inspect.getsource(cls)).body[0]
            own = set(cls.__dict__)
            receivers = {'self'} | {n.args.args[0].arg for n in ast.walk(node)
                                    if isinstance(n, ast.Lambda) and n.args.args}
            for n in ast.walk(node):
                if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                        and n.value.id in receivers and n.attr not in own):
                    declared = self.declared.get(n.attr)
                    if declared is not None and declared > rank:
                        reaches.append(f"{cls.__name__}: {n.attr} is declared in {HOSTS[declared]}")
        self.assertEqual(sorted(set(reaches)), [])

    def test_hooks_are_implemented_only_above_their_layer(self):
        wrong = []
        for name, (rank, _) in self.hooks.items():
            definers = [c for c in self.mro if name in c.__dict__]
            if not definers:
                wrong.append(f"{name}: no class implements it")
            for c in definers:
                if layer_of(c.__module__) <= rank:
                    wrong.append(f"{name}: {c.__name__} implements it in {HOSTS[rank]}'s layer or below")
        self.assertEqual(wrong, [])

    def test_the_statement_table_merges_every_layer(self):
        from qubasic_core.algorithms import PrimitivesMixin
        from qubasic_core.circuits import QuantumStatementMixin
        from qubasic_core.interpreter import StatementMixin
        tables = (StatementMixin._LANGUAGE_STATEMENTS, QuantumStatementMixin._QUANTUM_STATEMENTS,
                  PrimitivesMixin._CIRCUIT_STATEMENTS)
        merged = QBasicTerminal._STMT_KEYWORDS
        self.assertEqual(set(merged), set().union(*tables))
        for word, handlers in merged.items():
            self.assertEqual(handlers, tuple(h for t in tables for h in t.get(word, ())))


class TestCommandDocstrings(unittest.TestCase):

    def test_each_command_documents_its_syntax(self):
        from qubasic_core.terminal import QBasicTerminal
        t = QBasicTerminal()
        bad = []
        tables = dict(t._CMD_WITH_ARG) | dict(t._CMD_NO_ARG)
        for word, attr in tables.items():
            method = getattr(t, attr, None) if isinstance(attr, str) else attr
            doc = (getattr(method, '__doc__', None) or '').strip()
            head = doc.split('\n')[0]
            if ' — ' not in head:
                bad.append(f"{word}: no 'SYNTAX — description' line")
            elif word.split()[0] not in head.split(' — ')[0].upper():
                bad.append(f"{word}: syntax line does not name the command")
        self.assertEqual(bad, [])


# ---------------------------------------------------------------------------
# The documentation against the code: the tutorial's examples run and print
# what it shows, the generated reference is current, and the catalog covers
# every statement keyword, built-in function and command.
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, 'docs')
sys.path.insert(0, os.path.join(ROOT, 'tools'))


def _covered(word: str) -> bool:
    """A keyword is covered when a catalog statement starts with it or names
    it in its syntax."""
    for _, keyword, syntax, _ in parser.STATEMENTS:
        if keyword.upper().split()[0] == word or re.search(rf'(?<![\w$]){re.escape(word)}(?![\w$])',
                                                           syntax.upper()):
            return True
    return False


class TestTutorial(unittest.TestCase):

    def test_examples_run_and_print_what_the_tutorial_shows(self):
        text = open(os.path.join(DOCS, 'tutorial.md'), encoding='utf-8').read()
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as work:
            os.chdir(work)
            try:
                for section in re.split(r'^## ', text, flags=re.M)[1:]:
                    with self.subTest(section=section.splitlines()[0]):
                        self._run_section(section)
            finally:
                os.chdir(cwd)

    def _run_section(self, section: str) -> None:
        t = QBasicTerminal()
        t.io = CaptureIO()
        t._seed = 7
        t.agent_mode = True
        blocks = list(re.finditer(r'```(qubasic|python|output)\n(.*?)```', section, flags=re.S))
        last = ''
        for m in blocks:
            kind, body = m.group(1), m.group(2)
            if kind == 'python':
                exec(compile(body, 'tutorial', 'exec'), {})
            elif kind == 'qubasic':
                start = len(t.io.parts)
                ProgramMgmtMixin._load_lines_with_defs(
                    body.splitlines(), lambda line: t.process(line, track_undo=False))
                last = ''.join(t.io.parts[start:])
                errors = [ln for ln in last.splitlines() if ln.startswith('?')]
                self.assertEqual(errors, [], f"{body}\n---\n{last}")
            else:
                shown = [ln.strip() for ln in last.splitlines()]
                pos = 0
                for want in (ln.strip() for ln in body.splitlines() if ln.strip()):
                    while pos < len(shown) and shown[pos] != want:
                        pos += 1
                    self.assertLess(pos, len(shown), f"tutorial line {want!r} not printed:\n{last}")
                    pos += 1


class TestReference(unittest.TestCase):

    def test_reference_is_current(self):
        import generate
        current = open(os.path.join(DOCS, 'reference.md'), encoding='utf-8').read()
        self.assertEqual(current, generate.build_reference(),
                         'docs/reference.md is out of date: run python tools/generate.py')


class TestCatalog(unittest.TestCase):

    def test_every_dispatched_statement_keyword_is_described(self):
        missing = [k for k in QBasicTerminal._STMT_KEYWORDS if not _covered(k)]
        self.assertEqual(missing, [])

    def test_control_flow_and_parsed_statements_are_described(self):
        words = ('GOTO GOSUB RETURN ON IF FOR NEXT WHILE WEND DO LOOP EXIT SELECT CASE END STOP REM '
                 'SUB FUNCTION CALL DEF ERROR RESUME ASSERT SWAP OPTION DATA READ RESTORE LOCAL '
                 'STATIC SHARED LET PRINT INPUT GET DIM REDIM ERASE TYPE MEAS RESET MEASURE MEASURE_X '
                 'SYNDROME SEND SHARE MID$ SYSTEM SLEEP CONST TIMER').split()
        self.assertEqual([w for w in words if not _covered(w)], [])

    def test_every_builtin_function_is_described(self):
        from qubasic_core.expression import _ZERO_ARG, ExpressionMixin, _builtin_table
        names = set(_builtin_table(ExpressionMixin._SAFE_FUNCS)) | set(_ZERO_ARG)
        names |= {'PEEK', 'USR', 'EOF', 'LOF', 'POINT', 'SEEK', 'LOC', 'INPUT$', 'PMAP', 'POS', 'FRE',
                  'LBOUND', 'UBOUND', 'SCREEN', 'PLAY', 'ENVIRON$', 'FILEATTR', 'LPOS', 'INP'}
        self.assertEqual(sorted(n for n in names if not parser.functions_for(n)), [])

    def test_every_command_has_one_topic(self):
        placed = [n for _, names in parser.COMMAND_TOPICS for n in names]
        commands = set(QBasicTerminal._CMD_WITH_ARG) | set(QBasicTerminal._CMD_NO_ARG)
        self.assertEqual(len(placed), len(set(placed)))
        self.assertEqual(set(placed), commands)

    def test_help_reads_the_catalog(self):
        t = QBasicTerminal()
        t.io = CaptureIO()
        t.dispatch('HELP PRINT')
        t.dispatch('HELP LEN')
        t.dispatch('HELP LINE INPUT')
        out = t.io.text
        self.assertIn('PRINT USING fmt$; items', out)
        self.assertIn('PRINT #n, items', out)
        self.assertIn('LEN(s$)', out)
        self.assertIn('LINE INPUT ["prompt";] s$', out)


# ---------------------------------------------------------------------------
# Every program in examples/ runs as `qubasic <file>` runs it, with no error.
# ---------------------------------------------------------------------------

EXAMPLES = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', '*')))


class TestExamples(unittest.TestCase):

    def test_examples_run_cleanly(self):
        self.assertGreaterEqual(len(EXAMPLES), 6)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as work:
            os.chdir(work)
            try:
                for path in EXAMPLES:
                    with self.subTest(example=os.path.basename(path)):
                        t = QBasicTerminal()
                        t.io = CaptureIO()
                        t.agent_mode = True
                        t._seed = 1
                        run_script(os.path.abspath(path), t)
                        errors = [ln for ln in t.io.text.splitlines() if ln.startswith('?')]
                        self.assertEqual(errors, [], t.io.text[-2000:])
            finally:
                os.chdir(cwd)

    def test_what_the_examples_compute(self):
        def run(name):
            t = QBasicTerminal()
            t.io = CaptureIO()
            t.agent_mode = True
            run_script(os.path.join(os.path.dirname(EXAMPLES[0]), name), t)
            return t

        t = run('classic.bas')
        self.assertIn('Primes: 3  5  7  31', t.io.text)
        t = run('bell.qb')
        self.assertEqual(set(t.last_counts), {'00', '11'})
        t = run('grover3.qb')
        self.assertGreater(t.last_counts.get('101', 0) / sum(t.last_counts.values()), 0.85)
        t = run('graphics.bas')
        self.assertIsNotNone(t.graphics.canvas)
        self.assertIn('colour at the centre:', t.io.text)


if __name__ == '__main__':
    unittest.main()
