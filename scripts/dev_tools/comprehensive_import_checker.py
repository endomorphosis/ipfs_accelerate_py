#!/usr/bin/env python3
"""
Python Import Checker for ipfs_accelerate_py

Validates import statements in Python files.
"""

import sys
import argparse
import ast
import contextlib
import io
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple
import importlib.util


OPTIONAL_MODULE_PREFIXES = {
    "TTS",
    "cloudkit_worker",
    "datasets",
    "faster_whisper",
    "gguf",
    "joblib",
    "librosa",
    "nncf",
    "openvino",
    "openvino_genai",
    "optimum",
    "playwright",
    "prometheus_client",
    "pydub",
    "pysbd",
    "sentence_transformers",
    "selenium",
    "sentencepiece",
    "sklearn",
    "soundfile",
    "tiktoken",
    "toml",
}


@contextlib.contextmanager
def _suppress_import_noise():
    """Mute incidental logs/warnings while probing optional module availability."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(stdout_buffer),
            contextlib.redirect_stderr(stderr_buffer),
        ):
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(previous_disable)


@lru_cache(maxsize=None)
def _module_exists(module_name: str) -> bool:
    """Return whether an importable module exists."""
    try:
        with _suppress_import_noise():
            return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _is_optional_module(module_name: str) -> bool:
    """Return whether a module belongs to a known optional dependency family."""
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in OPTIONAL_MODULE_PREFIXES
    )


def _annotate_parents(tree: ast.AST) -> None:
    """Attach parent pointers so nested imports can be identified cheaply."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "parent", parent)


def _is_nested_import(node: ast.AST) -> bool:
    """Return whether an import is nested inside a function/class scope."""
    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return True
        parent = getattr(parent, "parent", None)
    return False


def _node_in_statements(node: ast.AST, statements: List[ast.stmt]) -> bool:
    """Return whether a node is contained within one of the provided statements."""
    for statement in statements:
        if statement is node:
            return True
        for child in ast.walk(statement):
            if child is node:
                return True
    return False


def _handler_catches_import_error(handler: ast.ExceptHandler) -> bool:
    """Return whether an except handler catches import-resolution failures."""
    if handler.type is None:
        return True

    names: List[str] = []
    if isinstance(handler.type, ast.Name):
        names.append(handler.type.id)
    elif isinstance(handler.type, ast.Tuple):
        names.extend(element.id for element in handler.type.elts if isinstance(element, ast.Name))

    return any(name in {"ImportError", "ModuleNotFoundError"} for name in names)


def _is_guarded_optional_import(node: ast.AST) -> bool:
    """Return whether an import sits in a try block guarded by import-related handlers."""
    child = node
    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, ast.Try) and _node_in_statements(child, parent.body):
            if any(_handler_catches_import_error(handler) for handler in parent.handlers):
                return True
        child = parent
        parent = getattr(parent, "parent", None)
    return False


def _local_module_exists(file_path: Path, module_name: str) -> bool:
    """Check for bare-module fallbacks resolved from the current file's package ancestry."""
    module_parts = module_name.split('.')
    search_dir = file_path.parent
    while True:
        target = search_dir.joinpath(*module_parts)
        if target.with_suffix('.py').exists() or (target / '__init__.py').exists():
            return True
        if not (search_dir / '__init__.py').exists() or search_dir.parent == search_dir:
            break
        search_dir = search_dir.parent
    return False


def _relative_import_exists(file_path: Path, node: ast.ImportFrom) -> bool:
    """Best-effort resolve a relative import against the current file path."""
    base_dir = file_path.parent

    # level=1 means current package, so only ascend for deeper relative imports.
    for _ in range(max(node.level - 1, 0)):
        base_dir = base_dir.parent

    candidate_dirs = [base_dir]
    ancestor = base_dir.parent
    while ancestor != ancestor.parent and (ancestor / '__init__.py').exists():
        candidate_dirs.append(ancestor)
        ancestor = ancestor.parent

    if node.module:
        for candidate_dir in candidate_dirs:
            target = candidate_dir.joinpath(*node.module.split('.'))
            if target.with_suffix('.py').exists() or (target / '__init__.py').exists():
                return True
        return False

    # `from . import foo` may refer to a sibling module or a package attribute.
    # Treat an existing package context as sufficient to avoid false positives.
    return (base_dir / '__init__.py').exists() or any(
        (base_dir / alias.name).with_suffix('.py').exists() or (base_dir / alias.name / '__init__.py').exists()
        for alias in node.names
    )


def check_imports(file_path: Path) -> Tuple[bool, List[str]]:
    """Check if all imports in a file are valid."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))
        _annotate_parents(tree)

        errors = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Check if module exists
                    if not (
                        _is_optional_module(alias.name)
                        or _is_optional_module(alias.name.split('.')[0])
                        or _module_exists(alias.name)
                        or _module_exists(alias.name.split('.')[0])
                        or _local_module_exists(file_path, alias.name)
                        or _is_guarded_optional_import(node)
                        or (_is_nested_import(node) and not _module_exists(alias.name.split('.')[0]))
                    ):
                        errors.append(f"Cannot import '{alias.name}'")

            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    if not _relative_import_exists(file_path, node) and not _is_guarded_optional_import(node):
                        module_name = node.module or ""
                        errors.append(f"Cannot import from relative module '{'.' * node.level}{module_name}'")
                    continue

                if node.module:
                    if not (
                        _is_optional_module(node.module)
                        or _is_optional_module(node.module.split('.')[0])
                        or _module_exists(node.module)
                        or _module_exists(node.module.split('.')[0])
                        or _local_module_exists(file_path, node.module)
                        or _is_guarded_optional_import(node)
                        or (_is_nested_import(node) and not _module_exists(node.module.split('.')[0]))
                    ):
                        errors.append(f"Cannot import from '{node.module}'")

        return len(errors) == 0, errors

    except SyntaxError as e:
        return False, [f"SyntaxError: {e.msg}"]
    except Exception as e:
        return False, [f"Error: {str(e)}"]


def find_python_files(directory: Path, exclude_patterns: List[str]) -> List[Path]:
    """Find all Python files in directory."""
    python_files = []

    for py_file in directory.glob('**/*.py'):
        if py_file.is_file():
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in str(py_file):
                    should_exclude = True
                    break

            if not should_exclude:
                python_files.append(py_file)

    return python_files


def main():
    parser = argparse.ArgumentParser(description="Check Python imports")
    parser.add_argument('--directory', required=True, help='Directory to scan')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument(
        '--exclude',
        nargs='*',
        default=['__pycache__', '.venv', 'venv', '.git', 'build', 'dist'],
        help='Patterns to exclude',
    )

    args = parser.parse_args()

    directory = Path(args.directory).resolve()

    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Checking imports in: {directory}")
    print(f"Excluding: {', '.join(args.exclude)}")
    print()

    python_files = find_python_files(directory, args.exclude)

    if not python_files:
        print("No Python files found")
        return 0

    print(f"Found {len(python_files)} Python files to check\n")

    successful = 0
    failed = 0

    for py_file in sorted(python_files):
        success, errors = check_imports(py_file)

        if success:
            successful += 1
            if args.verbose:
                print(f"✓ {py_file.relative_to(directory)}")
        else:
            failed += 1
            print(f"✗ {py_file.relative_to(directory)}")
            for error in errors:
                print(f"  {error}")

    print("\n" + "="*60)
    print("IMPORT CHECK SUMMARY")
    print("="*60)
    print(f"Total files checked: {len(python_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if len(python_files) > 0:
        print(f"Success rate: {successful/len(python_files)*100:.1f}%")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
