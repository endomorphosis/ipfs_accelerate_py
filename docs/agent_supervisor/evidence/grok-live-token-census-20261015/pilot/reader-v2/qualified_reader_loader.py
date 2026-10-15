"""Load every pinned pure dependency, including imports inside validators."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys

STANDARD = frozenset({'__future__', 'argparse', 'hashlib', 'json', 'math', 'pathlib', 're'})


def load_reader(directory):
    root = Path(directory).resolve()
    frozen = json.loads((root / 'reporting-freeze.json').read_text())
    if set(frozen) != {'schema', 'files'} or frozen['schema'] != 'pure-reader-module-closure@1':
        raise ValueError('invalid_reader_closure')
    modules = {}
    for row in frozen['files']:
        if type(row) is not dict or set(row) != {'path', 'sha256', 'bytes'}:
            raise ValueError('invalid_reader_binding')
        path = Path(row['path'])
        if path.parent != root or path.is_symlink() or path.suffix != '.py':
            raise ValueError('invalid_reader_module_path')
        name = path.stem
        if not (name.startswith('diagnostic_') or name in {'safe_diagnostics', 'safe_public_validation', 'read_pilot'}) or name in modules:
            raise ValueError('invalid_reader_module_name')
        raw = path.read_bytes()
        if type(row['bytes']) is not int or len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('reader_binding_changed')
        modules[name] = path
    if 'read_pilot' not in modules:
        raise ValueError('missing_reader_entrypoint')
    for name, path in modules.items():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('relative_reader_import')
                imports = [node.module]
            else:
                continue
            if any(item not in STANDARD and item not in modules for item in imports):
                raise ValueError('unbound_reader_dependency')
        loaded = sys.modules.get(name)
        if loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != path:
            raise ValueError('foreign_reader_module')
    previous_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        # Do not merely import the entrypoint: some closed validators defer
        # their dependency import until a non-null observation is encountered.
        for name in modules:
            loaded = importlib.import_module(name)
            if Path(loaded.__file__).resolve() != modules[name]:
                raise ValueError('foreign_reader_module')
    finally:
        sys.path[:] = previous_path
    return sys.modules['read_pilot']
