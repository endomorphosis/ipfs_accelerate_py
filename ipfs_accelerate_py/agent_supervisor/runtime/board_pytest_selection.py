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


def _read_text(path: Path) -> str | None:
    try:
        if not path.is_file() or path.is_symlink():
            return None
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _relative(workspace: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _import_files(
    workspace: Path, dotted: str, changed_paths: frozenset[str],
) -> tuple[set[str], bool]:
    """Resolve local source and package initializers without importing anything.

    Local operators and namespace packages are dependencies too. A local path
    that cannot be inspected inside this workspace makes the caller uncertain;
    it must not be mistaken for an unrelated installed package.
    """
    if not dotted:
        return set(), False
    parts = dotted.split(".")
    if any(not part.isidentifier() for part in parts):
        return set(), True
    found: set[str] = set()
    try:
        root = workspace / parts[0]
        changed_root = any(
            path == parts[0] + ".py" or path.startswith(parts[0] + "/")
            for path in changed_paths
        )
        if not (changed_root or root.is_dir() or root.is_symlink()
                or root.with_suffix(".py").is_file()):
            return found, False
        workspace_root = workspace.resolve()
        for length in range(1, len(parts) + 1):
            base = workspace.joinpath(*parts[:length])
            if not base.resolve().is_relative_to(workspace_root):
                return found, True
            # Importing a.b executes a/__init__.py as well as b's source.
            # A from-import may name an attribute; its containing module is
            # still retained even when no child module exists for that name.
            for candidate in (base / "__init__.py", base.with_suffix(".py")):
                lexical = candidate.relative_to(workspace).as_posix()
                if lexical in changed_paths:
                    # Deleted modules still explain affected imports. Keeping
                    # the path also makes the bounded reader report uncertainty
                    # instead of letting a surviving __init__ hide the deletion.
                    found.add(lexical)
                if candidate.is_file():
                    relative = _relative(workspace, candidate)
                    if relative is None:
                        return found, True
                    found.add(relative)
        return found, not bool(found)
    except (OSError, RuntimeError, ValueError):
        return found, True


def _package_of(workspace: Path, relative: str) -> str:
    path = Path(relative)
    if path.name == "__init__.py":
        parts = path.parent.parts
    else:
        parts = path.with_suffix("").parts
    return ".".join(parts)


def _walk_imports(
    workspace: Path, relative: str, changed_paths: frozenset[str],
) -> tuple[set[str], bool]:
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
                resolved, unresolved = _import_files(workspace, alias.name, changed_paths)
                found.update(resolved)
                uncertain |= unresolved
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
            resolved, unresolved = _import_files(workspace, base, changed_paths)
            found.update(resolved)
            uncertain |= unresolved
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = f"{base}.{alias.name}" if base else alias.name
                resolved, unresolved = _import_files(workspace, child, changed_paths)
                found.update(resolved)
                uncertain |= unresolved
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


def _import_cone(
    workspace: Path, start: str, changed_paths: frozenset[str],
) -> tuple[set[str], bool]:
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
        imported, local_uncertain = _walk_imports(workspace, current, changed_paths)
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
    dirty_set = frozenset(dirty)
    edges: list[dict[str, Any]] = []
    uncertain_tests: set[str] = set()
    for test_file in tests:
        cone, uncertain = _import_cone(workspace, test_file, dirty_set)
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
