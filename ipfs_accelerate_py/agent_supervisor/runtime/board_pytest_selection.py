"""IVP-backed affected-test selection for hermetic board pytest.

Builds exact ``tested_by`` / ``imports`` edges from a bounded AST import cone
over dirty worktree paths, then asks ``select_affected_verification``. Unknown
or dynamic imports fail closed for that test file (run it). Full-suite
fallback from IVP also runs every collected item. No I/O beyond reading source
and git status.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

from ipfs_accelerate_py.agent_supervisor.verification.selection import (
    SelectionPolicy,
    select_affected_verification,
)

from .pytest_item_ledger import dirty_source_paths

UNAFFECTED_SKIP_REASON = "ivp-unaffected"
_MAX_IMPORT_FILES = 400
_MAX_FILE_BYTES = 2 * 1024 * 1024
_LOCAL_ROOTS = ("ipfs_accelerate_py", "test")


def _read_text(path: Path) -> str | None:
    try:
        if not path.is_file() or path.is_symlink():
            return None
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _relative(workspace: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _module_file(workspace: Path, dotted: str) -> str | None:
    if not dotted or dotted.startswith("."):
        return None
    root = dotted.split(".", 1)[0]
    if root not in _LOCAL_ROOTS:
        return None
    parts = dotted.split(".")
    base = workspace.joinpath(*parts)
    py_file = base.with_suffix(".py")
    init_file = base / "__init__.py"
    if py_file.is_file():
        return _relative(workspace, py_file)
    if init_file.is_file():
        return _relative(workspace, init_file)
    parent = workspace.joinpath(*parts[:-1]) if len(parts) > 1 else workspace
    sibling = parent.with_suffix(".py") if parent != workspace else None
    if sibling is not None and sibling.is_file():
        return _relative(workspace, sibling)
    return None


def _package_of(workspace: Path, relative: str) -> str:
    path = Path(relative)
    if path.name == "__init__.py":
        parts = path.parent.parts
    else:
        parts = path.with_suffix("").parts
    return ".".join(parts)


def _walk_imports(workspace: Path, relative: str) -> tuple[set[str], bool]:
    """Return imported local files and whether the file is uncertain."""

    text = _read_text(workspace / relative)
    if text is None:
        return set(), True
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set(), True
    package = _package_of(workspace, relative)
    found: set[str] = set()
    uncertain = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _module_file(workspace, alias.name)
                if resolved:
                    found.add(resolved)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                pkg_parts = package.split(".") if package else []
                if relative.endswith(".py") and not relative.endswith("__init__.py"):
                    pkg_parts = pkg_parts[:-1]
                prefix = pkg_parts[: max(0, len(pkg_parts) - (node.level - 1))]
                base = ".".join([*prefix, *( [node.module] if node.module else [] )])
            else:
                base = node.module or ""
            if node.names and any(alias.name == "*" for alias in node.names):
                uncertain = True
            resolved = _module_file(workspace, base) if base else None
            if resolved:
                found.add(resolved)
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = f"{base}.{alias.name}" if base else alias.name
                child_path = _module_file(workspace, child)
                if child_path:
                    found.add(child_path)
        elif isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in {"__import__", "import_module"}:
                uncertain = True
    return found, uncertain


def _import_cone(workspace: Path, start: str) -> tuple[set[str], bool]:
    seen: set[str] = set()
    queue = [start]
    uncertain = False
    while queue and len(seen) < _MAX_IMPORT_FILES:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if not current.endswith(".py"):
            continue
        imported, local_uncertain = _walk_imports(workspace, current)
        if local_uncertain:
            uncertain = True
        for item in imported:
            if item not in seen:
                queue.append(item)
    if queue:
        uncertain = True
    return seen, uncertain


def _path_mentioned(test_text: str, relative: str) -> bool:
    name = Path(relative).name
    return bool(name) and name in test_text


def select_affected_test_files(
    workspace: Path,
    test_files: Iterable[str],
) -> frozenset[str] | None:
    """Return affected test files, or ``None`` to run the whole collected set."""

    tests = tuple(dict.fromkeys(str(item) for item in test_files if str(item)))
    if not tests:
        return None
    dirty = dirty_source_paths(workspace)
    if not dirty:
        return None
    dirty_set = set(dirty)
    edges: list[dict[str, Any]] = []
    uncertain_tests: set[str] = set()
    for test_file in tests:
        cone, uncertain = _import_cone(workspace, test_file)
        if uncertain:
            uncertain_tests.add(test_file)
        text = _read_text(workspace / test_file) or ""
        for source in cone:
            if source == test_file:
                continue
            edges.append(
                {
                    "source": test_file,
                    "target": source,
                    "kind": "imports",
                    "disposition": "exact",
                    "critical": True,
                }
            )
            edges.append(
                {
                    "source": source,
                    "target": test_file,
                    "kind": "tested_by",
                    "disposition": "exact",
                    "critical": True,
                }
            )
        for relative in dirty:
            if not relative.endswith(".py") and _path_mentioned(text, relative):
                edges.append(
                    {
                        "source": relative,
                        "target": test_file,
                        "kind": "tested_by",
                        "disposition": "exact",
                        "critical": True,
                    }
                )
        if test_file in dirty_set:
            edges.append(
                {
                    "source": test_file,
                    "target": test_file,
                    "kind": "tested_by",
                    "disposition": "exact",
                    "critical": True,
                }
            )
    if not edges:
        return None
    try:
        selection = select_affected_verification(
            changed_paths=sorted(dirty_set),
            edges=edges,
            known_tests=tests,
            policy=SelectionPolicy(
                force_full_suite=False,
                broader_escalates_to_full_suite=False,
                critical_uncertainty_requires_full_suite=False,
                broader_includes_sibling_tests=True,
                policy_id="board-pytest-import-cone@1",
            ),
        )
    except Exception:
        return None
    if selection.full_suite_required:
        return None
    selected = set(selection.selected_tests)
    selected.update(uncertain_tests)
    selected.intersection_update(tests)
    if not selected:
        # Exact empty selection with dirty sources: fail closed, run all.
        if dirty_set:
            return None
        return frozenset()
    return frozenset(selected)
