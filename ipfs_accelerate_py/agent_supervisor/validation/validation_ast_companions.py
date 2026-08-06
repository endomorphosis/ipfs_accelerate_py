"""AST-driven companion path discovery for validation-linked repairs.

CI re-enable boards often declare only the integration test as Predicted
files / Outputs, while the real fix lives in the production module the test
imports (for example ``src/handsfree/*_interop.py``). The agent supervisor
previously restricted edit and proposal scope to the declared write set, so
implementers could not lawfully update the modules that owned broken path
constants.

This helper walks validation commands, parses the referenced test modules
with the Python AST, and resolves imported modules to repository-relative
source files so those companions can enter:

* implementer ``edit_policy.allowed_paths``
* proposal scope / path admission
* rescue prompts (optional relocation hints for missing string paths)
"""

from __future__ import annotations

import ast
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .validation_commands import infer_validation_impact_paths

# Prefer first-party package roots before broader repository search.
_DEFAULT_PACKAGE_ROOTS: tuple[str, ...] = (
    "src",
    "lib",
    "python",
    "",
)

_MISSING_PATH_STRING_RE = re.compile(
    r"""(?P<path>(?:external|src|tests|swissknife|mobile|docs|data)/"""
    r"""[A-Za-z0-9_./+\-]+\.[A-Za-z0-9]+)"""
)


def _normalize_repo_relative(value: str) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    pure = PurePosixPath(path)
    if (
        not path
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() in {".", ""}
    ):
        return ""
    return pure.as_posix()


def _is_python_test_path(relative: str) -> bool:
    pure = PurePosixPath(relative)
    if pure.suffix not in {".py", ".pyi"}:
        return False
    name = pure.name
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part in {"test", "tests"} for part in pure.parts)


def validation_command_test_paths(commands: Sequence[Any]) -> tuple[str, ...]:
    """Return safe repository-relative Python test paths from validation cmds."""

    found: list[str] = []
    for command in commands or ():
        for raw in infer_validation_impact_paths(str(command or "")):
            relative = _normalize_repo_relative(raw)
            if not relative or not _is_python_test_path(relative):
                continue
            if relative not in found:
                found.append(relative)
    return tuple(found)


def _module_to_candidate_paths(module_name: str) -> tuple[str, ...]:
    """Map a dotted import module to candidate repository-relative files."""

    module = str(module_name or "").strip().strip(".")
    if not module or module.startswith("."):
        return ()
    parts = [part for part in module.split(".") if part and part != "*"]
    if not parts:
        return ()
    # Reject obvious stdlib / third-party top-levels we never edit for CIG.
    top = parts[0]
    if top in {
        "os",
        "sys",
        "json",
        "re",
        "pathlib",
        "typing",
        "collections",
        "dataclasses",
        "importlib",
        "pytest",
        "unittest",
        "jsonschema",
        "numpy",
        "pandas",
    }:
        return ()
    joined = "/".join(parts)
    candidates: list[str] = []
    for root in _DEFAULT_PACKAGE_ROOTS:
        prefix = f"{root}/" if root else ""
        candidates.append(f"{prefix}{joined}.py")
        candidates.append(f"{prefix}{joined}/__init__.py")
    # de-dupe preserve order
    return tuple(dict.fromkeys(candidates))


def _iter_imported_modules(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name or "").strip()
                if name:
                    yield name
        elif isinstance(node, ast.ImportFrom):
            if int(node.level or 0) > 0:
                # Relative imports inside a test package are usually fixtures;
                # still resolve when module is absolute-ish via remaining name.
                if not node.module:
                    continue
            module = str(node.module or "").strip()
            if not module:
                continue
            yield module
            # Also consider imported leaf modules: from pkg import leaf_mod
            # when leaf looks like a submodule (best-effort).
            for alias in node.names:
                leaf = str(alias.name or "").strip()
                if not leaf or leaf == "*":
                    continue
                if leaf[:1].islower() or "_" in leaf:
                    yield f"{module}.{leaf}"


def _resolve_existing_module_path(
    module_name: str,
    *,
    repo_root: Path,
) -> str:
    for candidate in _module_to_candidate_paths(module_name):
        relative = _normalize_repo_relative(candidate)
        if not relative:
            continue
        absolute = repo_root / relative
        try:
            if absolute.is_file() and not absolute.is_symlink():
                return relative
        except OSError:
            continue
    return ""


def _parse_python_source(path: Path) -> ast.AST | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None


def validation_ast_companion_paths(
    task: Any,
    *,
    repo_root: Path,
    max_depth: int = 3,
    max_paths: int = 48,
) -> tuple[str, ...]:
    """Return repository files imported (via AST) by validation test modules.

    Walks up to ``max_depth`` import hops starting from validation test files so
    thin test wrappers that import ``handsfree.*_interop`` modules expand edit
    authority to the production owner of broken contracts. Also admits existing
    first-party path constants referenced in those tests (for example
    ``src/runtime_router.py``) so re-enable boards can lawfully edit them when
    the suite asserts on monorepo-relative production modules.
    """

    root = Path(repo_root)
    commands = tuple(getattr(task, "validation", ()) or ())
    seeds = list(validation_command_test_paths(commands))
    # Also seed declared outputs that are tests.
    for raw in getattr(task, "outputs", ()) or ():
        relative = _normalize_repo_relative(str(raw))
        if relative and _is_python_test_path(relative) and relative not in seeds:
            seeds.append(relative)

    resolved: list[str] = []
    seen_files: set[str] = set()
    queue: list[tuple[str, int]] = [(seed, 0) for seed in seeds]

    while queue and len(resolved) < max_paths:
        relative, depth = queue.pop(0)
        if relative in seen_files:
            continue
        seen_files.add(relative)
        absolute = root / relative
        if not absolute.is_file():
            continue
        # Seeds (tests) are not automatically write-scope companions unless
        # already declared; we only add *imported* production modules.
        tree = _parse_python_source(absolute)
        if tree is None:
            continue
        for module_name in _iter_imported_modules(tree):
            companion = _resolve_existing_module_path(
                module_name,
                repo_root=root,
            )
            if not companion or companion in resolved:
                continue
            # Prefer first-party packages; skip pure third-party installs.
            if not companion.startswith(
                ("src/", "lib/", "python/", "swissknife/", "mobile/", "tests/")
            ) and "/" not in companion.strip("/"):
                continue
            resolved.append(companion)
            if depth + 1 < max_depth:
                queue.append((companion, depth + 1))
            if len(resolved) >= max_paths:
                break
        # Existing first-party path constants referenced by the test suite.
        # Include monorepo evidence/docs paths when tests assert on them
        # (e.g. objective heap strings for CIG interop suites).
        if depth == 0 and len(resolved) < max_paths:
            for literal in _string_path_literals(tree):
                if not literal.startswith(
                    (
                        "src/",
                        "swissknife/",
                        "mobile/",
                        "docs/",
                        "implementation_plan/",
                        "data/",
                        "hallucinate_app/",
                    )
                ):
                    continue
                if not (root / literal).is_file():
                    continue
                if literal in resolved or literal in seeds:
                    continue
                resolved.append(literal)
                if len(resolved) >= max_paths:
                    break

    return tuple(sorted(resolved))


def _string_path_literals(tree: ast.AST) -> tuple[str, ...]:
    found: list[str] = []
    for node in ast.walk(tree):
        value = ""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
        elif isinstance(node, ast.JoinedStr):
            # Skip f-strings; too dynamic for safe relocation hints.
            continue
        text = str(value or "").strip().replace("\\", "/")
        if not text or len(text) > 240:
            continue
        match = _MISSING_PATH_STRING_RE.search(text)
        if match:
            relative = _normalize_repo_relative(match.group("path"))
            if relative and relative not in found:
                found.append(relative)
            continue
        # Also accept bare relative constants used under a known root join.
        if text.endswith(
            (".json", ".py", ".md", ".ts", ".js", ".yaml", ".yml")
        ) and "/" in text:
            relative = _normalize_repo_relative(text)
            if relative and relative not in found:
                found.append(relative)
    return tuple(found)


def _is_actionable_missing_constant(path: str) -> bool:
    """Filter path constants that are useful for monorepo pin relocation."""

    relative = _normalize_repo_relative(path)
    if not relative:
        return False
    # Android sample-relative paths are joined under external/meta-wearables.
    if relative.startswith(("src/main/", "app/src/", "java/com/")):
        return False
    if relative.startswith(
        ("external/", "swissknife/", "mobile/", "src/handsfree/", "src/")
    ):
        return True
    if relative.startswith((".tools/", "docs/", "data/", "examples/", "ipfs_")):
        return True
    if "/.tools/" in f"/{relative}" or relative.endswith(
        (".schema.json", ".json", ".md", ".py")
    ):
        return True
    return False


def _missing_path_exists(root: Path, missing: str) -> bool:
    try:
        if (root / missing).is_file():
            return True
    except OSError:
        return False
    # Constants are often joined under external/* roots.
    for prefix in (
        "external/ipfs_datasets/",
        "external/ipfs_kit/",
        "external/ipfs_accelerate/",
        "external/meta-wearables-dat-android/",
        "external/meta-wearables-dat-ios/",
    ):
        try:
            if (root / prefix / missing).is_file():
                return True
        except OSError:
            continue
    return False


def _find_basename_candidate(root: Path, missing: str) -> str:
    basename = PurePosixPath(missing).name
    if not basename or basename in {".", ".."}:
        return ""
    # Prefer external/ipfs_kit when the constant still mentions ipfs_kit_py tools.
    preferred_roots = ("external", "src", "swissknife", "mobile")
    if "ipfs_kit" in missing or ".tools/" in missing:
        preferred_roots = ("external/ipfs_kit", "external", "src")
    for search_root_name in preferred_roots:
        search_root = root / search_root_name
        if not search_root.is_dir():
            continue
        try:
            for hit in search_root.rglob(basename):
                try:
                    if not hit.is_file() or hit.is_symlink():
                        continue
                    rel = hit.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                if rel == missing or rel.endswith("/" + missing):
                    continue
                return rel
        except OSError:
            continue
    return ""


def validation_ast_relocation_hints(
    task: Any,
    *,
    repo_root: Path,
    max_hints: int = 12,
) -> tuple[dict[str, str], ...]:
    """Return missing path constants with relocated basename candidates.

    Each hint maps ``missing`` → ``candidate`` when the constant path does not
    exist but a same-basename file exists under the repository (for example
    descriptors that moved from ``external/ipfs_datasets/.tools/ipfs_kit_py``
    to ``external/ipfs_kit``).
    """

    root = Path(repo_root)
    companions = validation_ast_companion_paths(task, repo_root=root)
    seeds = list(validation_command_test_paths(getattr(task, "validation", ()) or ()))
    scan_files = list(dict.fromkeys([*companions, *seeds]))
    hints: list[dict[str, str]] = []
    seen_missing: set[str] = set()

    for relative in scan_files:
        absolute = root / relative
        tree = _parse_python_source(absolute)
        if tree is None:
            continue
        for missing in _string_path_literals(tree):
            if missing in seen_missing or not _is_actionable_missing_constant(missing):
                continue
            if _missing_path_exists(root, missing):
                continue
            candidate = _find_basename_candidate(root, missing)
            if not candidate:
                continue
            seen_missing.add(missing)
            hints.append(
                {
                    "owner_module": relative,
                    "missing": missing,
                    "candidate": candidate,
                }
            )
            if len(hints) >= max_hints:
                return tuple(hints)
    return tuple(hints)


def format_relocation_hints_for_prompt(
    hints: Sequence[Mapping[str, str]] | Sequence[dict[str, str]],
) -> str:
    """Render relocation hints as compact implementer guidance."""

    if not hints:
        return ""
    lines = [
        "AST descriptor relocation hints (update owner modules; do not invent new roots):"
    ]
    for hint in hints:
        owner = str(hint.get("owner_module") or "").strip()
        missing = str(hint.get("missing") or "").strip()
        candidate = str(hint.get("candidate") or "").strip()
        if not (missing and candidate):
            continue
        owner_bit = f" in {owner}" if owner else ""
        lines.append(f"- missing `{missing}`{owner_bit} → use `{candidate}`")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)
