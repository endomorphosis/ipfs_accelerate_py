"""Safe generic ProjectAdapter (EAAEF-040).

Perform bounded, read-only language/build/test/static inventory of a directory
and return a typed support outcome.  This adapter never fabricates mutation
argv and never admits autonomous mutation.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final


PROJECT_ADAPTER_INVENTORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/project-adapter-inventory@1"
)
ADAPTER_ID: Final[str] = "generic@1"
INVENTORY_AUTHORIZES_MUTATION: Final[bool] = False
INVENTORY_IS_COMPLETION_EVIDENCE: Final[bool] = False

DEFAULT_MAX_FILES: Final[int] = 4_096
DEFAULT_MAX_DEPTH: Final[int] = 24
DEFAULT_MAX_FILE_BYTES: Final[int] = 65_536

_SKIP_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {
        ".agent-supervisor",
        ".aws",
        ".bzr",
        ".cache",
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pijul",
        ".pytest_cache",
        ".ruff_cache",
        ".ssh",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "CVS",
        "bower_components",
        "dist",
        "htmlcov",
        "node_modules",
        "site-packages",
        "target",
        "venv",
    }
)
_CREDENTIAL_FILENAMES: Final[frozenset[str]] = frozenset(
    {
        ".env",
        ".netrc",
        "_netrc",
        "credentials",
        "credentials.json",
        "credentials.yaml",
        "credentials.yml",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
    }
)
_CREDENTIAL_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".der", ".jks", ".key", ".keystore", ".p12", ".pem", ".pfx", ".pkcs12"}
)

_LANGUAGE_MANIFESTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "cargo.lock": "rust",
        "cargo.toml": "rust",
        "composer.json": "php",
        "gemfile": "ruby",
        "go.mod": "go",
        "go.sum": "go",
        "package-lock.json": "javascript",
        "package.json": "javascript",
        "pipfile": "python",
        "pipfile.lock": "python",
        "pnpm-lock.yaml": "javascript",
        "poetry.lock": "python",
        "pyproject.toml": "python",
        "requirements.txt": "python",
        "setup.cfg": "python",
        "setup.py": "python",
        "tsconfig.json": "typescript",
        "uv.lock": "python",
        "yarn.lock": "javascript",
        "build.gradle": "java",
        "build.gradle.kts": "java",
        "pom.xml": "java",
        "cmakelists.txt": "c",
        "mix.exs": "elixir",
    }
)
_LANGUAGE_EXTENSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".cxx": "cpp",
        ".go": "go",
        ".h": "c",
        ".hpp": "cpp",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".php": "php",
        ".py": "python",
        ".pyi": "python",
        ".rb": "ruby",
        ".rs": "rust",
        ".scala": "scala",
        ".swift": "swift",
        ".ts": "typescript",
        ".tsx": "typescript",
    }
)
_BUILD_MANIFESTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "build.gradle": "gradle",
        "build.gradle.kts": "gradle",
        "cargo.toml": "cargo",
        "cmakelists.txt": "cmake",
        "composer.json": "composer",
        "gemfile": "bundler",
        "gnumakefile": "make",
        "go.mod": "go",
        "makefile": "make",
        "meson.build": "meson",
        "mix.exs": "mix",
        "package.json": "npm",
        "pipfile": "pipenv",
        "pom.xml": "maven",
        "pyproject.toml": "python",
        "setup.cfg": "python",
        "setup.py": "python",
    }
)
_TEST_FILENAMES: Final[frozenset[str]] = frozenset(
    {
        "conftest.py",
        "jest.config.cjs",
        "jest.config.js",
        "jest.config.mjs",
        "jest.config.ts",
        "noxfile.py",
        "phpunit.xml",
        "pytest.ini",
        "tox.ini",
        "vitest.config.js",
        "vitest.config.ts",
    }
)
_TEST_DIRECTORIES: Final[frozenset[str]] = frozenset({"spec", "test", "tests"})
_STATIC_FILENAMES: Final[frozenset[str]] = frozenset(
    {
        ".eslintrc",
        ".eslintrc.cjs",
        ".eslintrc.js",
        ".eslintrc.json",
        ".eslintrc.yml",
        ".flake8",
        ".isort.cfg",
        ".mypy.ini",
        ".pylintrc",
        ".ruff.toml",
        "clippy.toml",
        "eslint.config.js",
        "eslint.config.mjs",
        "eslint.config.ts",
        "mypy.ini",
        "pylintrc",
        "pyrightconfig.json",
        "ruff.toml",
        "rustfmt.toml",
        ".golangci.yml",
        ".golangci.yaml",
    }
)
_PYPROJECT_TEST_TABLES: Final[tuple[str, ...]] = (
    "tool.pytest",
    "tool.pytest.ini_options",
    "tool.tox",
    "tool.nox",
    "tool.hatch.envs.default.scripts",
)
_PYPROJECT_STATIC_TABLES: Final[tuple[str, ...]] = (
    "tool.ruff",
    "tool.mypy",
    "tool.black",
    "tool.isort",
    "tool.pylint",
    "tool.pyright",
    "tool.flake8",
    "tool.bandit",
)


class SupportOutcome(str, Enum):
    """Typed generic-adapter support outcomes.

    ``insufficient_validation`` is the inventory spelling of the board outcome
    ``insufficient_validation_profile``.
    """

    PREVIEW_ONLY = "preview_only"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    UNSUPPORTED_BUILD_SYSTEM = "unsupported_build_system"
    UNSAFE_REPOSITORY = "unsafe_repository"
    INSUFFICIENT_VALIDATION = "insufficient_validation"
    MUTATION_NOT_ADMITTED = "mutation_not_admitted"
    SUPPORTED_INVENTORY = "supported_inventory"


class SignalKind(str, Enum):
    LANGUAGE = "language"
    BUILD = "build"
    TEST = "test"
    STATIC = "static"


@dataclass(frozen=True, order=True)
class InventorySignal:
    """One observed language, build, test, or static signal."""

    kind: SignalKind
    name: str
    path: str


@dataclass(frozen=True)
class InventoryBounds:
    """Hard ceilings for a single read-only walk."""

    max_files: int = DEFAULT_MAX_FILES
    max_depth: int = DEFAULT_MAX_DEPTH
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES

    def __post_init__(self) -> None:
        if type(self.max_files) is not int or self.max_files < 1:
            raise ValueError("max_files must be a positive integer")
        if type(self.max_depth) is not int or self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if type(self.max_file_bytes) is not int or self.max_file_bytes < 1:
            raise ValueError("max_file_bytes must be a positive integer")


@dataclass(frozen=True)
class ProjectSupport:
    """Typed inventory result. Mutation argv is always empty."""

    outcome: SupportOutcome
    languages: tuple[str, ...] = ()
    build_systems: tuple[str, ...] = ()
    test_signals: tuple[str, ...] = ()
    static_signals: tuple[str, ...] = ()
    signals: tuple[InventorySignal, ...] = ()
    skipped_paths: tuple[str, ...] = ()
    files_visited: int = 0
    mutation_admitted: bool = False
    mutation_argv: tuple[str, ...] = ()
    reason: str = ""
    adapter_id: str = ADAPTER_ID
    schema: str = PROJECT_ADAPTER_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.mutation_admitted:
            raise ValueError("generic inventory cannot admit mutation")
        if self.mutation_argv:
            raise ValueError("generic inventory cannot carry mutation argv")

    def as_mapping(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "adapter_id": self.adapter_id,
                "outcome": self.outcome.value,
                "languages": self.languages,
                "build_systems": self.build_systems,
                "test_signals": self.test_signals,
                "static_signals": self.static_signals,
                "signals": tuple(
                    {
                        "kind": item.kind.value,
                        "name": item.name,
                        "path": item.path,
                    }
                    for item in self.signals
                ),
                "skipped_paths": self.skipped_paths,
                "files_visited": self.files_visited,
                "mutation_admitted": False,
                "mutation_argv": (),
                "reason": self.reason,
                "inventory_authorizes_mutation": INVENTORY_AUTHORIZES_MUTATION,
            }
        )


class ProjectAdapter:
    """Bounded read-only project inventory. Never fabricates mutation argv."""

    adapter_id: str = ADAPTER_ID
    mutation_admitted: bool = False

    def inspect(
        self,
        root: str | os.PathLike[str],
        *,
        bounds: InventoryBounds | None = None,
        max_files: int | None = None,
        max_depth: int | None = None,
    ) -> ProjectSupport:
        limit = bounds or InventoryBounds()
        if max_files is not None or max_depth is not None:
            limit = InventoryBounds(
                max_files=limit.max_files if max_files is None else max_files,
                max_depth=limit.max_depth if max_depth is None else max_depth,
                max_file_bytes=limit.max_file_bytes,
            )
        return _inspect_root(Path(root), limit)

    def inventory(
        self,
        root: str | os.PathLike[str],
        *,
        bounds: InventoryBounds | None = None,
        max_files: int | None = None,
        max_depth: int | None = None,
    ) -> ProjectSupport:
        return self.inspect(
            root, bounds=bounds, max_files=max_files, max_depth=max_depth
        )

    def mutation_commands(self, inventory: ProjectSupport | None = None) -> tuple[str, ...]:
        del inventory
        return ()

    def admit_mutation(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        inventory: ProjectSupport | None = None,
        bounds: InventoryBounds | None = None,
    ) -> ProjectSupport:
        support = inventory if inventory is not None else self.inspect(root or ".", bounds=bounds)
        if support.outcome in {
            SupportOutcome.UNSAFE_REPOSITORY,
            SupportOutcome.UNSUPPORTED_LANGUAGE,
            SupportOutcome.UNSUPPORTED_BUILD_SYSTEM,
            SupportOutcome.PREVIEW_ONLY,
            SupportOutcome.INSUFFICIENT_VALIDATION,
        }:
            return support
        return ProjectSupport(
            outcome=SupportOutcome.MUTATION_NOT_ADMITTED,
            languages=support.languages,
            build_systems=support.build_systems,
            test_signals=support.test_signals,
            static_signals=support.static_signals,
            signals=support.signals,
            skipped_paths=support.skipped_paths,
            files_visited=support.files_visited,
            reason="generic adapter never admits mutation",
        )


GenericProjectAdapter = ProjectAdapter


def inspect_project(
    root: str | os.PathLike[str],
    *,
    bounds: InventoryBounds | None = None,
    max_files: int | None = None,
    max_depth: int | None = None,
) -> ProjectSupport:
    """Inspect ``root`` with the generic read-only adapter."""

    return ProjectAdapter().inspect(
        root, bounds=bounds, max_files=max_files, max_depth=max_depth
    )


def _inspect_root(root: Path, bounds: InventoryBounds) -> ProjectSupport:
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        return _refused(SupportOutcome.UNSAFE_REPOSITORY, "repository root does not exist")
    except OSError:
        return _refused(SupportOutcome.UNSAFE_REPOSITORY, "repository root is unreadable")
    if stat.S_ISLNK(root_stat.st_mode):
        return _refused(SupportOutcome.UNSAFE_REPOSITORY, "repository root is a symlink")
    if not stat.S_ISDIR(root_stat.st_mode):
        return _refused(SupportOutcome.UNSAFE_REPOSITORY, "repository root is not a directory")

    walk = _walk_repository(root, bounds)
    if walk.unsafe_reason:
        return _refused(
            SupportOutcome.UNSAFE_REPOSITORY,
            walk.unsafe_reason,
            skipped_paths=walk.skipped,
            files_visited=walk.files_visited,
        )
    if not walk.files:
        return ProjectSupport(
            outcome=SupportOutcome.PREVIEW_ONLY,
            skipped_paths=walk.skipped,
            files_visited=walk.files_visited,
            reason="repository has no inventoried source files",
        )

    signals = _collect_signals(root, walk.files, bounds.max_file_bytes)
    languages = _unique(signal.name for signal in signals if signal.kind is SignalKind.LANGUAGE)
    build_systems = _unique(signal.name for signal in signals if signal.kind is SignalKind.BUILD)
    test_signals = _unique(signal.name for signal in signals if signal.kind is SignalKind.TEST)
    static_signals = _unique(signal.name for signal in signals if signal.kind is SignalKind.STATIC)

    if not languages:
        outcome = SupportOutcome.UNSUPPORTED_LANGUAGE
        reason = "no recognized programming language"
    elif not build_systems:
        outcome = SupportOutcome.UNSUPPORTED_BUILD_SYSTEM
        reason = "recognized language has no known build system"
    elif not test_signals and not static_signals:
        outcome = SupportOutcome.INSUFFICIENT_VALIDATION
        reason = "language and build are present without test or static signals"
    else:
        outcome = SupportOutcome.SUPPORTED_INVENTORY
        reason = "bounded language/build/test/static inventory"

    return ProjectSupport(
        outcome=outcome,
        languages=languages,
        build_systems=build_systems,
        test_signals=test_signals,
        static_signals=static_signals,
        signals=signals,
        skipped_paths=walk.skipped,
        files_visited=walk.files_visited,
        reason=reason,
    )


@dataclass(frozen=True)
class _WalkResult:
    files: tuple[str, ...]
    skipped: tuple[str, ...]
    files_visited: int
    unsafe_reason: str = ""


def _walk_repository(root: Path, bounds: InventoryBounds) -> _WalkResult:
    files: list[str] = []
    skipped: list[str] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    root_resolved = root.resolve()

    while stack:
        current, depth = stack.pop()
        if depth > bounds.max_depth:
            return _WalkResult(
                files=tuple(files),
                skipped=tuple(skipped),
                files_visited=len(files),
                unsafe_reason="directory depth exceeds inventory bound",
            )
        try:
            entries = list(os.scandir(current))
        except OSError:
            return _WalkResult(
                files=tuple(files),
                skipped=tuple(skipped),
                files_visited=len(files),
                unsafe_reason="directory is unreadable",
            )
        for entry in sorted(entries, key=lambda item: item.name):
            relative = _relative_posix(root, Path(entry.path))
            if relative is None:
                return _WalkResult(
                    files=tuple(files),
                    skipped=tuple(skipped),
                    files_visited=len(files),
                    unsafe_reason="path escaped repository root",
                )
            try:
                is_symlink = entry.is_symlink()
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                return _WalkResult(
                    files=tuple(files),
                    skipped=tuple(skipped),
                    files_visited=len(files),
                    unsafe_reason=f"directory entry is unreadable: {relative}",
                )
            if is_dir and entry.name in _SKIP_DIRECTORIES:
                skipped.append(relative)
                continue
            if _is_secret_name(entry.name):
                skipped.append(relative)
                continue
            if is_symlink:
                if not _symlink_stays_inside(Path(entry.path), root_resolved):
                    return _WalkResult(
                        files=tuple(files),
                        skipped=tuple(skipped),
                        files_visited=len(files),
                        unsafe_reason=f"symlink escaped repository root: {relative}",
                    )
                return _WalkResult(
                    files=tuple(files),
                    skipped=tuple(skipped),
                    files_visited=len(files),
                    unsafe_reason=f"symlink refused: {relative}",
                )
            if is_dir:
                stack.append((Path(entry.path), depth + 1))
                continue
            if not is_file:
                return _WalkResult(
                    files=tuple(files),
                    skipped=tuple(skipped),
                    files_visited=len(files),
                    unsafe_reason=f"special file refused: {relative}",
                )
            files.append(relative)
            if len(files) > bounds.max_files:
                return _WalkResult(
                    files=tuple(files[: bounds.max_files]),
                    skipped=tuple(skipped),
                    files_visited=len(files),
                    unsafe_reason="file count exceeds inventory bound",
                )
    return _WalkResult(files=tuple(files), skipped=tuple(skipped), files_visited=len(files))


def _collect_signals(
    root: Path, files: tuple[str, ...], max_file_bytes: int
) -> tuple[InventorySignal, ...]:
    observed: set[InventorySignal] = set()
    directories = {str(PurePosixPath(path).parts[0]) for path in files if "/" in path}

    for relative in files:
        posix = PurePosixPath(relative)
        name = posix.name
        lowered = name.lower()
        parent = posix.parts[0] if len(posix.parts) > 1 else ""

        language = _LANGUAGE_MANIFESTS.get(lowered)
        if language is None:
            language = _LANGUAGE_EXTENSIONS.get(posix.suffix.lower())
        if language is not None:
            observed.add(InventorySignal(SignalKind.LANGUAGE, language, relative))

        build = _BUILD_MANIFESTS.get(lowered)
        if build is not None:
            observed.add(InventorySignal(SignalKind.BUILD, build, relative))

        if lowered in _TEST_FILENAMES or parent.lower() in _TEST_DIRECTORIES:
            observed.add(InventorySignal(SignalKind.TEST, name, relative))
        elif name.startswith("test_") and posix.suffix.lower() == ".py":
            observed.add(InventorySignal(SignalKind.TEST, "pytest", relative))
        elif name.endswith("_test.py") or name.endswith("_test.go"):
            observed.add(InventorySignal(SignalKind.TEST, name, relative))

        if lowered in _STATIC_FILENAMES:
            observed.add(InventorySignal(SignalKind.STATIC, name, relative))

        if lowered == "pyproject.toml":
            text = _bounded_text(root / relative, max_file_bytes)
            if any(_has_toml_table(text, table) for table in _PYPROJECT_TEST_TABLES):
                observed.add(InventorySignal(SignalKind.TEST, "pytest", relative))
            if any(_has_toml_table(text, table) for table in _PYPROJECT_STATIC_TABLES):
                observed.add(InventorySignal(SignalKind.STATIC, "ruff", relative))

    for directory in directories:
        if directory.lower() in _TEST_DIRECTORIES:
            observed.add(InventorySignal(SignalKind.TEST, directory, directory))

    return tuple(sorted(observed, key=lambda item: (item.kind.value, item.path, item.name)))


def _bounded_text(path: Path, max_file_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            payload = handle.read(max_file_bytes + 1)
    except OSError:
        return ""
    if len(payload) > max_file_bytes:
        payload = payload[:max_file_bytes]
    return payload.decode("utf-8", errors="replace")


def _has_toml_table(text: str, table: str) -> bool:
    marker = f"[{table}]"
    return marker in text


def _is_secret_name(name: str) -> bool:
    lowered = name.lower()
    if lowered in _CREDENTIAL_FILENAMES:
        return True
    if lowered.startswith(".env"):
        return True
    suffix = Path(name).suffix.lower()
    return suffix in _CREDENTIAL_SUFFIXES


def _relative_posix(root: Path, path: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _symlink_stays_inside(link: Path, root_resolved: Path) -> bool:
    try:
        resolved = link.resolve()
        resolved.relative_to(root_resolved)
    except (OSError, ValueError):
        return False
    return True


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _refused(
    outcome: SupportOutcome,
    reason: str,
    *,
    skipped_paths: tuple[str, ...] = (),
    files_visited: int = 0,
) -> ProjectSupport:
    return ProjectSupport(
        outcome=outcome,
        skipped_paths=skipped_paths,
        files_visited=files_visited,
        reason=reason,
    )


__all__ = (
    "ADAPTER_ID",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_FILE_BYTES",
    "GenericProjectAdapter",
    "INVENTORY_AUTHORIZES_MUTATION",
    "INVENTORY_IS_COMPLETION_EVIDENCE",
    "InventoryBounds",
    "InventorySignal",
    "PROJECT_ADAPTER_INVENTORY_SCHEMA",
    "ProjectAdapter",
    "ProjectSupport",
    "SignalKind",
    "SupportOutcome",
    "inspect_project",
)
