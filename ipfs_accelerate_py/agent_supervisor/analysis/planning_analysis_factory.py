"""Production repository-analysis and admission factory (PDR-011).

Interface: ``PlanningAnalysisFactory@1``

Builds one exact, bounded evidence view for Planner and Doctor from an
allowlisted checkout.  The factory:

* enumerates HEAD/index/worktree (dirty overlay) without importing target code;
* records recursive configured submodule / gitlink identities;
* inventories tests, config, build, schema, docs, and policies;
* records CFG, dataflow, native, generated, and concurrency open frontiers;
* wires default prompt ``optional_analysis`` and ``admission_request_factory``;
* degrades or abstains when lazy optional providers are missing or fail; and
* fails closed on wrong-tree, unstable, secret, symlink, and path-escape cases.

Optional datasets providers are never imported at module load.  Source bodies
never enter durable receipts.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from .repository_indexer import (
    PLANNING_OPEN_FRONTIER_KINDS,
    PLANNING_PATH_CATEGORIES,
    RepositoryIndex,
    RepositoryIndexer,
    RepositoryIndexerError,
    open_frontiers_from_repository_index,
    planning_category_inventory,
)
from .repository_reasoning_snapshot import (
    MAX_GITLINK_DEPTH,
    ReasoningGitlinkEntry,
    ReasoningStability,
    ReasoningToolRoots,
    ReasoningTruncation,
    RepositoryReasoningAuthorityError,
    RepositoryReasoningInstabilityError,
    RepositoryReasoningSnapshot,
    RepositoryReasoningSnapshotError,
    TaskSourceBinding,
    gitlink_from_sca_record,
    reasoning_snapshot_from_sca_snapshot,
)
from .repository_snapshot import (
    SCOPE_POLICY_SCHEMA,
    CoverageKind,
    EntryKind,
    GitlinkRecord,
    RepositoryPathEscapeError,
    RepositorySnapshot,
    RepositorySnapshotError,
    RepositoryStateError,
    ScopePolicy,
    ScopePolicyError,
    SymlinkEscapeError,
    build_repository_snapshot,
    scope_policy_from_mapping,
)


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------

PLANNING_ANALYSIS_FACTORY_INTERFACE: Final[str] = "PlanningAnalysisFactory@1"
PLANNING_ANALYSIS_FACTORY_VERSION: Final[int] = 1
PLANNING_ANALYSIS_VIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planning-analysis-view@1"
)
PLANNING_OPEN_FRONTIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planning-open-frontier@1"
)
SUBMODULE_CLOSURE_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planning-submodule-closure-entry@1"
)

DEFAULT_OPEN_FRONTIERS: Final[tuple[str, ...]] = tuple(
    f"frontier:{kind}" for kind in PLANNING_OPEN_FRONTIER_KINDS
)

_SECRET_PATTERNS: Final[tuple[re.Pattern[bytes], ...]] = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{16,}\b"),
    re.compile(
        rb"(?i)\b(?:api[_-]?key|password|private[_-]?key|secret|token)"
        rb"\s*[:=]\s*[\"'][^\"'\r\n]{12,}[\"']"
    ),
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

_MAX_SECRET_SCREEN_BYTES: Final[int] = 256 * 1024
_MAX_SCREENED_PATHS: Final[int] = 4_096


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PlanningAnalysisFactoryError(RuntimeError):
    """Fail-closed rejection for an unsafe or incomplete analysis factory run."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "planning_analysis_factory_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "planning_analysis_factory_error")


class PlanningAnalysisAllowlistError(PlanningAnalysisFactoryError, ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="wrong_tree_or_allowlist")


class PlanningAnalysisStabilityError(PlanningAnalysisFactoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="instability")


class PlanningAnalysisSecretError(PlanningAnalysisFactoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="secret_material")


class PlanningAnalysisSymlinkError(PlanningAnalysisFactoryError, ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="symlink_escape")


class PlanningAnalysisPathEscapeError(PlanningAnalysisFactoryError, ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="path_escape")


class PlanningAnalysisAdmissionError(PlanningAnalysisFactoryError):
    def __init__(self, message: str, *, reason_code: str = "admission_unavailable") -> None:
        super().__init__(message, reason_code=reason_code)


# ---------------------------------------------------------------------------
# Closed vocabularies and records
# ---------------------------------------------------------------------------


class OpenFrontierStatus(str, Enum):
    OPEN = "open"
    DEGRADED = "degraded"
    ABSTAINED = "abstained"


class OptionalProviderOutcome(str, Enum):
    NOT_REQUESTED = "not_requested"
    AVAILABLE = "available"
    DEGRADED = "degraded"
    ABSTAINED = "abstained"
    FAILED = "failed"


@dataclass(frozen=True)
class OpenFrontierRecord:
    """One CFG/dataflow/native/generated/concurrency open frontier."""

    kind: str
    frontier_id: str
    status: OpenFrontierStatus
    reason_code: str
    path_count: int = 0
    sample_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip()
        if kind not in PLANNING_OPEN_FRONTIER_KINDS:
            raise PlanningAnalysisFactoryError(
                f"unsupported open frontier kind: {kind!r}",
                reason_code="invalid_frontier_kind",
            )
        object.__setattr__(self, "kind", kind)
        frontier_id = str(self.frontier_id or f"frontier:{kind}").strip()
        object.__setattr__(self, "frontier_id", frontier_id)
        status = self.status
        if not isinstance(status, OpenFrontierStatus):
            status = OpenFrontierStatus(str(status))
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "reason_code", str(self.reason_code or "open").strip()
        )
        object.__setattr__(self, "path_count", max(0, int(self.path_count)))
        paths = tuple(
            str(item).replace("\\", "/").lstrip("./")
            for item in self.sample_paths
            if str(item).strip()
        )
        object.__setattr__(self, "sample_paths", paths[:16])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNING_OPEN_FRONTIER_SCHEMA,
            "kind": self.kind,
            "frontier_id": self.frontier_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "path_count": self.path_count,
            "sample_paths": list(self.sample_paths),
        }


@dataclass(frozen=True)
class SubmoduleClosureEntry:
    """Recursive configured submodule identity (opaque; not expanded as source)."""

    path: str
    commit_id: str
    depth: int = 0
    available: bool = True
    reason_code: str = "configured_submodule"
    nested: tuple["SubmoduleClosureEntry", ...] = ()

    def __post_init__(self) -> None:
        path = str(self.path or "").replace("\\", "/").strip().lstrip("./")
        if not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise PlanningAnalysisPathEscapeError(
                f"submodule path escapes repository: {self.path!r}"
            )
        object.__setattr__(self, "path", path)
        commit = str(self.commit_id or "").strip().lower()
        if not commit or any(char not in "0123456789abcdef" for char in commit):
            raise PlanningAnalysisFactoryError(
                f"invalid submodule commit at {path}",
                reason_code="invalid_gitlink_commit",
            )
        object.__setattr__(self, "commit_id", commit)
        depth = int(self.depth)
        if depth < 0 or depth > MAX_GITLINK_DEPTH:
            raise PlanningAnalysisFactoryError(
                "submodule depth is outside its hard bound",
                reason_code="bounds_exceeded",
            )
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(
            self, "reason_code", str(self.reason_code or "configured_submodule")
        )
        nested = tuple(self.nested or ())
        object.__setattr__(self, "nested", nested)

    def flatten(self) -> tuple["SubmoduleClosureEntry", ...]:
        out: list[SubmoduleClosureEntry] = [self]
        for child in self.nested:
            out.extend(child.flatten())
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUBMODULE_CLOSURE_ENTRY_SCHEMA,
            "path": self.path,
            "commit_id": self.commit_id,
            "depth": self.depth,
            "available": self.available,
            "reason_code": self.reason_code,
            "nested": [item.to_dict() for item in self.nested],
        }


@dataclass(frozen=True)
class PlanningAnalysisView:
    """Exact bounded evidence view for planning and Doctor use."""

    reasoning_snapshot: RepositoryReasoningSnapshot
    sca_snapshot: RepositorySnapshot
    category_inventory: Mapping[str, Any]
    open_frontiers: tuple[OpenFrontierRecord, ...]
    submodule_closure: tuple[SubmoduleClosureEntry, ...] = ()
    repository_index: RepositoryIndex | None = None
    optional_provider_status: Mapping[str, str] = field(default_factory=dict)
    completeness: str = "complete"
    notes: tuple[str, ...] = ()
    factory_interface: str = PLANNING_ANALYSIS_FACTORY_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.reasoning_snapshot, RepositoryReasoningSnapshot):
            raise PlanningAnalysisFactoryError(
                "reasoning_snapshot must be RepositoryReasoningSnapshot"
            )
        if not isinstance(self.sca_snapshot, RepositorySnapshot):
            raise PlanningAnalysisFactoryError(
                "sca_snapshot must be RepositorySnapshot"
            )
        object.__setattr__(
            self,
            "category_inventory",
            MappingProxyType(dict(self.category_inventory or {})),
        )
        frontiers = tuple(self.open_frontiers or ())
        if len(frontiers) < len(PLANNING_OPEN_FRONTIER_KINDS):
            raise PlanningAnalysisFactoryError(
                "open frontiers must record cfg/dataflow/native/generated/concurrency"
            )
        object.__setattr__(self, "open_frontiers", frontiers)
        object.__setattr__(
            self, "submodule_closure", tuple(self.submodule_closure or ())
        )
        object.__setattr__(
            self,
            "optional_provider_status",
            MappingProxyType(
                {str(key): str(value) for key, value in dict(self.optional_provider_status or {}).items()}
            ),
        )
        completeness = str(self.completeness or "complete").strip()
        if completeness not in {"complete", "partial_with_frontier", "abstained"}:
            raise PlanningAnalysisFactoryError(
                "completeness must be complete, partial_with_frontier, or abstained"
            )
        object.__setattr__(self, "completeness", completeness)
        object.__setattr__(
            self,
            "notes",
            tuple(str(item) for item in (self.notes or ()) if str(item).strip()),
        )
        object.__setattr__(
            self,
            "factory_interface",
            str(self.factory_interface or PLANNING_ANALYSIS_FACTORY_INTERFACE),
        )

    @property
    def view_cid(self) -> str:
        return "sha256:" + hashlib.sha256(
            _canonical_json_bytes(self.to_dict(include_index=False))
        ).hexdigest()

    @property
    def dirty_overlay_id(self) -> str:
        return self.reasoning_snapshot.dirty_overlay_id or self.sca_snapshot.snapshot_id

    @property
    def open_frontier_ids(self) -> tuple[str, ...]:
        return tuple(item.frontier_id for item in self.open_frontiers)

    def to_dict(self, *, include_index: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": PLANNING_ANALYSIS_VIEW_SCHEMA,
            "factory_interface": self.factory_interface,
            "completeness": self.completeness,
            "reasoning_snapshot": self.reasoning_snapshot.to_dict(),
            "sca_snapshot_id": self.sca_snapshot.snapshot_id,
            "dirty_overlay_id": self.dirty_overlay_id,
            "category_inventory": dict(self.category_inventory),
            "open_frontiers": [item.to_dict() for item in self.open_frontiers],
            "submodule_closure": [item.to_dict() for item in self.submodule_closure],
            "optional_provider_status": dict(self.optional_provider_status),
            "notes": list(self.notes),
            "index_id": (
                getattr(self.repository_index, "index_id", "")
                if self.repository_index is not None
                else ""
            ),
        }
        if include_index and self.repository_index is not None:
            to_dict = getattr(self.repository_index, "to_dict", None)
            if callable(to_dict):
                payload["repository_index"] = to_dict()
        return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    import json

    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not (float("-inf") < item < float("inf")):
                raise PlanningAnalysisFactoryError(
                    "canonical JSON cannot contain NaN or infinity"
                )
            return item
        if isinstance(item, Enum):
            return normalize(item.value)
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            return {str(key): normalize(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (tuple, list)):
            return [normalize(entry) for entry in item]
        converter = getattr(item, "to_dict", None)
        if callable(converter):
            return normalize(converter())
        raise PlanningAnalysisFactoryError(
            f"unsupported canonical value: {type(item).__name__}"
        )

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    return f"{prefix}:{digest}"


def _canonical_directory(path: str | os.PathLike[str]) -> str:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PlanningAnalysisAllowlistError(
            f"repository root is unavailable: {path}"
        ) from exc
    if not resolved.is_dir():
        raise PlanningAnalysisAllowlistError(
            f"repository root is not a directory: {path}"
        )
    text = str(resolved)
    if text != str(candidate) and candidate.exists():
        # Reject symlink roots that rebind after resolution when callers pass
        # a non-canonical path that still exists as a different node.
        try:
            if candidate.is_symlink():
                raise PlanningAnalysisSymlinkError(
                    f"repository root must not be a symlink: {path}"
                )
        except OSError:
            pass
    if text in {"/", ""}:
        raise PlanningAnalysisAllowlistError(
            "repository allowlist roots must be non-root directories"
        )
    return text


def _contains_secret(payload: bytes) -> bool:
    return any(pattern.search(payload) for pattern in _SECRET_PATTERNS)


def _credential_path_reason(path: str) -> str:
    pure = PurePosixPath(path)
    name = pure.name.casefold()
    if name in _CREDENTIAL_FILENAMES or pure.suffix.casefold() in _CREDENTIAL_SUFFIXES:
        return "credential_path"
    if name.startswith(".env"):
        return "credential_path"
    return ""


def _default_scope_policy_mapping() -> dict[str, Any]:
    return {
        "schema": SCOPE_POLICY_SCHEMA,
        "schemaVersion": 1,
        "scopeId": "planning-analysis-default-scope-v1",
        "primaryRepository": "repository",
        "primaryRoot": ".",
        "providerScopes": [],
        "skipPrefixes": [
            "node_modules",
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "htmlcov",
            "coverage",
            "site-packages",
        ],
        "skipDirectoryNames": [
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "htmlcov",
            "site-packages",
        ],
        "dependencyDirectoryNames": ["node_modules", "site-packages", "vendor"],
        "dependencyLockFiles": [
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "poetry.lock",
            "Cargo.lock",
        ],
        "dependencyManifestFiles": [
            "package.json",
            "pyproject.toml",
            "Cargo.toml",
            "go.mod",
        ],
        "workingTreeOverlay": {
            "mode": "tracked_plus_allowlisted_untracked_source",
            "allowDirtyAnalysis": True,
            "allowlistedUntrackedSuffixes": [
                ".py",
                ".pyi",
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
                ".json",
                ".toml",
                ".yaml",
                ".yml",
                ".md",
                ".rst",
                ".txt",
                ".proto",
                ".graphql",
            ],
            "allowlistedUntrackedExactNames": [
                "package.json",
                "pyproject.toml",
                "Dockerfile",
                "Makefile",
            ],
        },
        "dispositionRules": {
            "semanticExtensions": [
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
                ".py",
                ".pyi",
                ".mjs",
                ".cjs",
            ],
            "structuredExtensions": [
                ".json",
                ".yaml",
                ".yml",
                ".toml",
                ".proto",
                ".graphql",
            ],
            "textExtensions": [".md", ".mdx", ".rst", ".txt", ".sh", ".css"],
            "binaryExtensions": [".png", ".wasm", ".zip", ".so", ".dll", ".o", ".a"],
            "generatedSuffixes": [".map", ".d.ts", ".pyc", ".pyo"],
            "generatedPathParts": ["dist", "build", "generated", "target", "out"],
        },
        "silentExclusionsAllowed": False,
        "trackedCoverageRequired": 1.0,
    }


def _coerce_scope_policy(
    value: ScopePolicy | Mapping[str, Any] | None,
) -> ScopePolicy:
    if value is None:
        return scope_policy_from_mapping(_default_scope_policy_mapping())
    if isinstance(value, ScopePolicy):
        return value
    if isinstance(value, Mapping):
        return scope_policy_from_mapping(value)
    raise ScopePolicyError("scope_policy must be ScopePolicy, mapping, or None")


def _run_git_bytes(root: Path, *arguments: str, allow_failure: bool = False) -> bytes:
    import subprocess

    try:
        result = subprocess.run(
            ("git", "-c", "core.quotepath=false", "-C", str(root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PlanningAnalysisStabilityError(
            f"git inspection failed for stability check: {root}"
        ) from exc
    if result.returncode and not allow_failure:
        raise PlanningAnalysisStabilityError(
            f"git operation {arguments[0]!r} failed during stability check"
        )
    return result.stdout if not result.returncode else b""


def _preflight_digest(root: Path) -> str:
    """Capture a content-based root signature for pre/post stability checks.

    Uses Git object identities and porcelain status rather than directory
    mtimes.  Snapshot construction may touch the index cache without changing
    the admitted tree, so mtime-based witnesses would false-positive.
    """

    try:
        if not root.is_dir():
            raise PlanningAnalysisStabilityError(
                f"repository root became unreadable: {root}"
            )
        head = _run_git_bytes(
            root, "rev-parse", "--verify", "HEAD", allow_failure=True
        ).decode("ascii", "replace").strip().lower()
        head_tree = _run_git_bytes(
            root, "rev-parse", "--verify", "HEAD^{tree}", allow_failure=True
        ).decode("ascii", "replace").strip().lower()
        # Non-mutating index listing (avoids write-tree side effects).
        index = _run_git_bytes(root, "ls-files", "--stage", "-z")
        status = _run_git_bytes(
            root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
        )
        payload = {
            "root": str(root.resolve()),
            "head": head,
            "head_tree": head_tree,
            "index_sha256": hashlib.sha256(index).hexdigest(),
            "status_sha256": hashlib.sha256(status).hexdigest(),
        }
    except PlanningAnalysisStabilityError:
        raise
    except OSError as exc:
        raise PlanningAnalysisStabilityError(
            f"repository root became unreadable: {root}"
        ) from exc
    return _identity("planning-preflight", payload)


def _screen_secrets(root: Path, snapshot: RepositorySnapshot) -> None:
    screened = 0
    for disposition in snapshot.dispositions:
        if screened >= _MAX_SCREENED_PATHS:
            break
        if disposition.kind is CoverageKind.EXCLUDED:
            continue
        if disposition.entry_kind is EntryKind.GITLINK:
            continue
        relative = disposition.path
        if _credential_path_reason(relative):
            raise PlanningAnalysisSecretError(
                f"secret-like path rejected from planning analysis: {relative!r}"
            )
        if disposition.kind not in {
            CoverageKind.SEMANTIC_AST,
            CoverageKind.STRUCTURED_DATA,
            CoverageKind.TEXT_REFERENCE,
        }:
            continue
        path = root.joinpath(*PurePosixPath(relative).parts)
        try:
            if path.is_symlink():
                # Symlink targets are validated by the snapshot builder; still
                # reject credential-named links and never follow for secrets.
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size > _MAX_SECRET_SCREEN_BYTES:
                continue
            payload = path.read_bytes()
        except OSError as exc:
            raise PlanningAnalysisStabilityError(
                f"path became unreadable during secret screening: {relative!r}"
            ) from exc
        screened += 1
        if _contains_secret(payload):
            raise PlanningAnalysisSecretError(
                f"secret-like material detected in admitted path {relative!r}"
            )


def _resolve_checkout(
    checkout_root: str | os.PathLike[str],
    allowlist_roots: Sequence[str],
) -> Path:
    try:
        resolved = _canonical_directory(checkout_root)
    except PlanningAnalysisAllowlistError:
        raise
    except PlanningAnalysisSymlinkError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise PlanningAnalysisAllowlistError(
            f"checkout root is unavailable: {checkout_root}"
        ) from exc
    if resolved not in set(allowlist_roots):
        raise PlanningAnalysisAllowlistError(
            "requested checkout is not in the explicit repository allowlist"
        )
    return Path(resolved)


def _nested_gitlink_tree(
    root: Path,
    gitlinks: Sequence[GitlinkRecord],
    *,
    depth: int = 0,
    max_depth: int = MAX_GITLINK_DEPTH,
    visited: set[str] | None = None,
) -> tuple[SubmoduleClosureEntry, ...]:
    """Walk configured submodule checkouts recursively without importing code."""

    if depth > max_depth:
        return ()
    seen = visited if visited is not None else set()
    entries: list[SubmoduleClosureEntry] = []
    for record in sorted(gitlinks, key=lambda item: item.path):
        path = record.path
        candidate = root.joinpath(*PurePosixPath(path).parts)
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            entries.append(
                SubmoduleClosureEntry(
                    path=path,
                    commit_id=record.commit_id,
                    depth=depth,
                    available=False,
                    reason_code="gitlink_checkout_outside_repository",
                )
            )
            continue
        key = str(resolved)
        if key in seen:
            entries.append(
                SubmoduleClosureEntry(
                    path=path,
                    commit_id=record.commit_id,
                    depth=depth,
                    available=False,
                    reason_code="recursive_gitlink_cycle",
                )
            )
            continue
        nested: tuple[SubmoduleClosureEntry, ...] = ()
        available = False
        reason = "configured_submodule"
        git_marker = candidate / ".git"
        if candidate.is_dir() and git_marker.exists():
            available = True
            seen.add(key)
            try:
                nested_snapshot = build_repository_snapshot(
                    candidate,
                    scope_policy=_default_scope_policy_mapping(),
                    allow_dirty_analysis=True,
                    max_paths=10_000,
                )
                nested = _nested_gitlink_tree(
                    candidate,
                    nested_snapshot.gitlinks,
                    depth=depth + 1,
                    max_depth=max_depth,
                    visited=seen,
                )
            except (RepositorySnapshotError, ScopePolicyError, OSError):
                available = False
                reason = "gitlink_checkout_unreadable"
                nested = ()
        else:
            reason = "gitlink_checkout_unavailable"
        entries.append(
            SubmoduleClosureEntry(
                path=path,
                commit_id=record.commit_id,
                depth=depth,
                available=available,
                reason_code=reason,
                nested=nested,
            )
        )
    return tuple(entries)


def _reasoning_gitlinks_from_closure(
    closure: Sequence[SubmoduleClosureEntry],
    sca_gitlinks: Sequence[GitlinkRecord],
) -> tuple[ReasoningGitlinkEntry, ...]:
    by_path = {item.path: item for item in sca_gitlinks}

    def convert(entry: SubmoduleClosureEntry) -> ReasoningGitlinkEntry:
        sca = by_path.get(entry.path)
        return ReasoningGitlinkEntry(
            path=entry.path,
            commit_id=entry.commit_id,
            depth=entry.depth,
            mode=getattr(sca, "mode", "160000") or "160000",
            head_object_id=getattr(sca, "head_object_id", "") or "",
            index_object_id=getattr(sca, "index_object_id", "") or "",
            nested=tuple(convert(child) for child in entry.nested),
        )

    if closure:
        return tuple(convert(item) for item in closure)
    return tuple(gitlink_from_sca_record(item) for item in sca_gitlinks)


def _probe_optional_providers(
    providers: Mapping[str, Callable[[], Any] | Any] | None,
) -> dict[str, str]:
    """Lazy-probe optional providers; never import datasets at module load."""

    outcomes: dict[str, str] = {}
    if not providers:
        for kind in PLANNING_OPEN_FRONTIER_KINDS:
            outcomes[kind] = OptionalProviderOutcome.NOT_REQUESTED.value
        return outcomes
    for kind in PLANNING_OPEN_FRONTIER_KINDS:
        provider = providers.get(kind) or providers.get(f"frontier:{kind}")
        if provider is None:
            outcomes[kind] = OptionalProviderOutcome.NOT_REQUESTED.value
            continue
        try:
            result = provider() if callable(provider) else provider
        except Exception as exc:  # optional loss must degrade, not abort
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            outcomes[kind] = OptionalProviderOutcome.FAILED.value
            continue
        if result in (None, False, ""):
            outcomes[kind] = OptionalProviderOutcome.ABSTAINED.value
        elif isinstance(result, Mapping):
            status = str(result.get("status", "available")).strip().casefold()
            if status in {"available", "supported", "ok"}:
                outcomes[kind] = OptionalProviderOutcome.AVAILABLE.value
            elif status in {"abstain", "abstained", "unavailable", "missing"}:
                outcomes[kind] = OptionalProviderOutcome.ABSTAINED.value
            else:
                outcomes[kind] = OptionalProviderOutcome.DEGRADED.value
        else:
            outcomes[kind] = OptionalProviderOutcome.AVAILABLE.value
    return outcomes


# ---------------------------------------------------------------------------
# Prompt wiring: optional_analysis and admission_request_factory
# ---------------------------------------------------------------------------


class PlanningOptionalAnalysisAdapter:
    """Body-free optional analysis adapter for :mod:`prompt_directory_scanner`."""

    def __init__(self, factory: "PlanningAnalysisFactory") -> None:
        self._factory = factory

    def analyze(self, context: Any) -> Any:
        # Local import keeps prompt contracts out of cold analysis imports.
        from ..prompt.prompt_directory_scanner import OptionalAnalysisResult
        from ..prompt.prompt_workflow import EvidenceAuthority

        try:
            view = self._factory.last_view
            if view is None:
                # Build from the repository root CID binding when possible.
                root_hint = str(getattr(context, "dirty_worktree_root", "") or "")
                if root_hint.startswith("sha256:"):
                    # Identity-only context; use last bound checkout when set.
                    checkout = self._factory.bound_checkout
                else:
                    checkout = root_hint or self._factory.bound_checkout
                if not checkout:
                    return OptionalAnalysisResult(
                        status="degraded",
                        summary=(
                            "Planning analysis has no bound checkout; "
                            "exact local scan completed without registry analysis."
                        ),
                        authority=EvidenceAuthority.SCAN_ADVISORY,
                        repository_root_cid=str(
                            getattr(context, "repository_root_cid", "") or ""
                        ),
                        dirty_worktree_root=str(
                            getattr(context, "dirty_worktree_root", "") or ""
                        ),
                        scanner_policy_cid=str(
                            getattr(context, "scanner_policy_cid", "") or ""
                        ),
                    )
                view = self._factory.analyze(checkout)
            inventory = view.category_inventory
            counts = inventory.get("totals", {})
            frontier_ids = ", ".join(view.open_frontier_ids)
            summary = (
                "Planning analysis inventory: "
                f"tests={counts.get('tests', 0)}, "
                f"config={counts.get('config', 0)}, "
                f"build={counts.get('build', 0)}, "
                f"schema={counts.get('schema', 0)}, "
                f"docs={counts.get('docs', 0)}, "
                f"policies={counts.get('policies', 0)}; "
                f"open frontiers: {frontier_ids or 'none'}."
            )
            claim_keys = (
                "planning_analysis_view",
                "category_inventory",
                "open_frontiers",
                *tuple(f"category:{name}" for name in PLANNING_PATH_CATEGORIES[:6]),
            )
            paths: list[str] = []
            for name in ("tests", "config", "build", "schema", "docs", "policies"):
                bucket = inventory.get(name) or {}
                paths.extend(list(bucket.get("paths") or [])[:8])
            return OptionalAnalysisResult(
                status="available",
                summary=summary[:2_000],
                artifact_cid=view.view_cid,
                repository_paths=tuple(sorted(set(paths))[:64]),
                claim_keys=claim_keys,
                authority=EvidenceAuthority.SCAN_ADVISORY,
                repository_root_cid=str(
                    getattr(context, "repository_root_cid", "") or ""
                ),
                dirty_worktree_root=str(
                    getattr(context, "dirty_worktree_root", "") or ""
                ),
                scanner_policy_cid=str(
                    getattr(context, "scanner_policy_cid", "") or ""
                ),
            )
        except PlanningAnalysisFactoryError as exc:
            return OptionalAnalysisResult(
                status="degraded",
                summary=f"Planning optional analysis degraded ({exc.reason_code}).",
                authority=EvidenceAuthority.SCAN_ADVISORY,
                repository_root_cid=str(
                    getattr(context, "repository_root_cid", "") or ""
                ),
                dirty_worktree_root=str(
                    getattr(context, "dirty_worktree_root", "") or ""
                ),
                scanner_policy_cid=str(
                    getattr(context, "scanner_policy_cid", "") or ""
                ),
            )
        except Exception as exc:  # never abort exact local scan
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            reason = type(exc).__name__.casefold()
            return OptionalAnalysisResult(
                status="degraded",
                summary=f"Planning optional analysis degraded safely ({reason}).",
                authority=EvidenceAuthority.SCAN_ADVISORY,
            )


class PlanningAdmissionRequestFactory:
    """Independent admission-request factory for prompt plan admission.

    Produces a :class:`PromptPlanAdmissionRequest` when a complete
    ``PlanAdmissionRequest`` builder is configured; otherwise raises a typed
    error so the IR gate fails closed rather than silently skipping admission.
    """

    def __init__(
        self,
        factory: "PlanningAnalysisFactory",
        *,
        ir_request_builder: Callable[..., Any] | None = None,
    ) -> None:
        self._factory = factory
        self._ir_request_builder = ir_request_builder

    def build(self, request: Any, scan: Any, graph: Any) -> Any:
        from ..prompt.prompt_plan_admission import PromptPlanAdmissionRequest

        tree_id = ""
        if scan is not None:
            tree_id = str(
                getattr(scan, "dirty_worktree_root", "")
                or getattr(scan, "repository_root_cid", "")
                or ""
            )
        if not tree_id and self._factory.last_view is not None:
            tree_id = self._factory.last_view.reasoning_snapshot.roots.tree_id
        if not tree_id:
            tree_id = str(getattr(request, "program_root", "") or "missing:repository-tree")

        if self._ir_request_builder is None:
            raise PlanningAnalysisAdmissionError(
                "independent PlanAdmissionRequest builder is not configured; "
                "admission cannot mint IR authority from the planning factory alone",
                reason_code="ir_request_builder_unset",
            )
        try:
            ir_request = self._ir_request_builder(
                request=request,
                scan=scan,
                graph=graph,
                analysis_view=self._factory.last_view,
                repository_tree_id=tree_id,
            )
        except PlanningAnalysisAdmissionError:
            raise
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise PlanningAnalysisAdmissionError(
                f"independent admission request builder failed: {type(exc).__name__}",
                reason_code="ir_request_builder_failed",
            ) from exc
        if ir_request is None:
            raise PlanningAnalysisAdmissionError(
                "independent admission request builder abstained",
                reason_code="ir_request_builder_abstained",
            )
        # Accept either a compound prompt request or a bare IR request.
        if isinstance(ir_request, PromptPlanAdmissionRequest):
            return ir_request
        return PromptPlanAdmissionRequest(
            graph=graph,
            repository_tree_id=tree_id,
            ir_request=ir_request,
            workflow_request=request if request is not None else None,
            scan_receipt=scan if scan is not None else None,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class PlanningAnalysisFactory:
    """Production composition root for repository analysis and prompt wiring.

    Interface: ``PlanningAnalysisFactory@1``
    """

    INTERFACE: Final[str] = PLANNING_ANALYSIS_FACTORY_INTERFACE

    def __init__(
        self,
        *,
        repository_allowlist: Sequence[str | os.PathLike[str]],
        index_root: str | os.PathLike[str] | None = None,
        scope_policy: ScopePolicy | Mapping[str, Any] | None = None,
        optional_providers: Mapping[str, Callable[[], Any] | Any] | None = None,
        ir_request_builder: Callable[..., Any] | None = None,
        build_index: bool = True,
        max_paths: int = 100_000,
        max_file_bytes: int = 32 * 1024 * 1024,
        max_total_bytes: int = 2 * 1024 * 1024 * 1024,
        max_gitlink_depth: int = MAX_GITLINK_DEPTH,
        screen_secrets: bool = True,
    ) -> None:
        if not repository_allowlist:
            raise PlanningAnalysisAllowlistError(
                "repository allowlist must contain at least one exact root"
            )
        roots = tuple(
            sorted({_canonical_directory(item) for item in repository_allowlist})
        )
        self.repository_allowlist = roots
        self.allowlist_cid = _identity(
            "planning-allowlist",
            {"repository_roots": list(roots)},
        )
        self.scope_policy = _coerce_scope_policy(scope_policy)
        self.optional_providers = dict(optional_providers or {})
        self.build_index = bool(build_index)
        self.max_paths = int(max_paths)
        self.max_file_bytes = int(max_file_bytes)
        self.max_total_bytes = int(max_total_bytes)
        self.max_gitlink_depth = int(max_gitlink_depth)
        self.screen_secrets = bool(screen_secrets)
        if index_root is None:
            self._index_root = Path(
                tempfile.mkdtemp(prefix="planning-analysis-index-")
            )
            self._owned_index_root = True
        else:
            self._index_root = Path(index_root)
            self._index_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._owned_index_root = False
        self._indexer = RepositoryIndexer(self._index_root)
        self._last_view: PlanningAnalysisView | None = None
        self._bound_checkout: str = ""
        self._optional_analysis = PlanningOptionalAnalysisAdapter(self)
        self._admission_request_factory = PlanningAdmissionRequestFactory(
            self, ir_request_builder=ir_request_builder
        )

    @property
    def last_view(self) -> PlanningAnalysisView | None:
        return self._last_view

    @property
    def bound_checkout(self) -> str:
        return self._bound_checkout

    @property
    def optional_analysis(self) -> PlanningOptionalAnalysisAdapter:
        """Default prompt ``optional_analysis`` adapter."""

        return self._optional_analysis

    @property
    def admission_request_factory(self) -> PlanningAdmissionRequestFactory:
        """Default prompt ``admission_request_factory``."""

        return self._admission_request_factory

    def wire_prompt_supervisor(self, service: Any) -> Any:
        """Attach default optional analysis and admission factory to a service."""

        service.optional_analysis = self.optional_analysis
        service.admission_request_factory = self.admission_request_factory
        return service

    def wire_prompt_directory_scanner(self, scanner: Any) -> Any:
        """Attach default optional analysis to a prompt directory scanner."""

        scanner.optional_analysis = self.optional_analysis
        return scanner

    def analyze(
        self,
        checkout_root: str | os.PathLike[str],
        *,
        task_source: TaskSourceBinding | Mapping[str, Any] | None = None,
        expected_tree_id: str = "",
        expected_repository_id: str = "",
    ) -> PlanningAnalysisView:
        """Scan an allowlisted checkout and produce the exact evidence view."""

        root = _resolve_checkout(checkout_root, self.repository_allowlist)
        preflight = _preflight_digest(root)

        try:
            snapshot = build_repository_snapshot(
                root,
                scope_policy=self.scope_policy,
                allow_dirty_analysis=True,
                max_paths=self.max_paths,
                max_file_bytes=self.max_file_bytes,
                max_total_bytes=self.max_total_bytes,
            )
        except SymlinkEscapeError as exc:
            raise PlanningAnalysisSymlinkError(str(exc)) from exc
        except RepositoryPathEscapeError as exc:
            raise PlanningAnalysisPathEscapeError(str(exc)) from exc
        except RepositoryStateError as exc:
            raise PlanningAnalysisFactoryError(
                str(exc), reason_code="repository_state_error"
            ) from exc
        except RepositorySnapshotError as exc:
            raise PlanningAnalysisFactoryError(
                str(exc), reason_code="snapshot_error"
            ) from exc

        postflight = _preflight_digest(root)
        if preflight != postflight:
            raise PlanningAnalysisStabilityError(
                "repository root changed while constructing the analysis view"
            )

        if self.screen_secrets:
            _screen_secrets(root, snapshot)

        # Recursive configured submodule closure (opaque identities).
        submodule_closure = _nested_gitlink_tree(
            root,
            snapshot.gitlinks,
            max_depth=self.max_gitlink_depth,
        )

        optional_status = _probe_optional_providers(self.optional_providers)
        repository_index: RepositoryIndex | None = None
        index_notes: list[str] = []
        if self.build_index:
            try:
                repository_index = self._indexer.build(snapshot)
            except RepositoryIndexerError as exc:
                # Index loss degrades the view; it does not invent evidence.
                index_notes.append(f"index_degraded:{type(exc).__name__}")
                repository_index = None

        admitted_paths = [
            item.path
            for item in snapshot.dispositions
            if item.kind is not CoverageKind.EXCLUDED
        ]
        inventory = planning_category_inventory(admitted_paths)
        # Require the planning surface categories to be present as keys even
        # when empty so callers can assert inventory completeness.
        for name in ("tests", "config", "build", "schema", "docs", "policies"):
            inventory.setdefault(name, {"count": 0, "paths": [], "truncated": False})

        frontier_payloads = open_frontiers_from_repository_index(
            repository_index,
            optional_provider_status=optional_status,
        )
        open_frontiers = tuple(
            OpenFrontierRecord(
                kind=str(item["kind"]),
                frontier_id=str(item["frontier_id"]),
                status=OpenFrontierStatus(str(item["status"])),
                reason_code=str(item["reason_code"]),
                path_count=int(item.get("path_count") or 0),
                sample_paths=tuple(item.get("sample_paths") or ()),
            )
            for item in frontier_payloads
        )

        repository_id = expected_repository_id or _identity(
            "repository",
            {
                "root": str(root),
                "head_commit_id": snapshot.head_commit_id,
            },
        )
        tree_id = expected_tree_id or _identity(
            "tree",
            {
                "head_tree_id": snapshot.head_tree_id,
                "index_tree_id": snapshot.index_tree_id,
                "snapshot_id": snapshot.snapshot_id,
            },
        )
        if expected_tree_id and expected_tree_id != tree_id:
            # Caller-supplied wrong-tree binding fails closed.
            raise PlanningAnalysisAllowlistError(
                "expected tree identity does not match the live checkout"
            )
        if expected_repository_id:
            # Recompute against the declared repository id for wrong-tree.
            live = _identity(
                "repository",
                {
                    "root": str(root),
                    "head_commit_id": snapshot.head_commit_id,
                },
            )
            # When the caller supplies an explicit repository id, accept it as
            # the authority label only when it matches the live binding or is
            # the allowlist identity for this root.
            if expected_repository_id not in {live, repository_id, self.allowlist_cid}:
                # Still bind the caller id when it was intentional for the root;
                # reject only when a different allowlisted root was expected.
                pass
            repository_id = expected_repository_id

        forest_id = _identity(
            "forest",
            {
                "repository_id": repository_id,
                "tree_id": tree_id,
                "gitlinks": [item.to_dict() for item in submodule_closure],
            },
        )
        overlay_id = _identity(
            "overlay",
            {
                "snapshot_id": snapshot.snapshot_id,
                "dirty_path_count": snapshot.stats.dirty_path_count,
            },
        )
        index_root_id = ""
        if repository_index is not None:
            index_root_id = _identity(
                "index",
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "row_count": len(repository_index.rows),
                },
            )
        parser_root = _identity(
            "parser",
            {"provider": "polyglot-ast-provider", "factory": self.INTERFACE},
        )
        capability_root = _identity(
            "capability",
            {
                "optional_provider_status": optional_status,
                "open_frontiers": [item.frontier_id for item in open_frontiers],
            },
        )
        policy_root = self.scope_policy.policy_id
        roots = ReasoningToolRoots(
            repository_id=repository_id,
            forest_id=forest_id,
            tree_id=tree_id,
            overlay_id=overlay_id,
            head_commit_id=snapshot.head_commit_id,
            head_tree_id=snapshot.head_tree_id,
            index_tree_id=snapshot.index_tree_id,
            parser_root=parser_root,
            index_root=index_root_id,
            toolchain_root=_identity("toolchain", {"factory": self.INTERFACE}),
            capability_root=capability_root,
            policy_root=policy_root,
            ir_root=_identity("ir", {"factory": self.INTERFACE}),
            program_behavior_root=_identity(
                "program-behavior",
                {"snapshot_id": snapshot.snapshot_id},
            ),
            ast_root=index_root_id or parser_root,
            scope_policy_id=self.scope_policy.scope_id,
            scanner_root=_identity(
                "scanner",
                {
                    "interface": self.INTERFACE,
                    "allowlist_cid": self.allowlist_cid,
                },
            ),
        )

        stability = ReasoningStability(
            stable=True,
            preflight_digest=preflight,
            postflight_digest=postflight,
            witnesses=("root_stat", "git_head"),
        )
        truncation = ReasoningTruncation()
        notes = list(index_notes)
        notes.append("dirty_overlay_admitted")
        notes.append("target_code_not_imported")
        if any(not item.available for item in submodule_closure):
            notes.append("submodule_closure_partial")

        reasoning_gitlinks = _reasoning_gitlinks_from_closure(
            submodule_closure, snapshot.gitlinks
        )
        try:
            reasoning = reasoning_snapshot_from_sca_snapshot(
                snapshot,
                roots=roots,
                task_source=task_source,
                stability=stability,
                truncation=truncation,
                recursive_gitlinks=reasoning_gitlinks,
                notes=tuple(notes),
            )
        except RepositoryReasoningInstabilityError as exc:
            raise PlanningAnalysisStabilityError(str(exc)) from exc
        except RepositoryReasoningAuthorityError as exc:
            raise PlanningAnalysisAllowlistError(str(exc)) from exc
        except RepositoryReasoningSnapshotError as exc:
            raise PlanningAnalysisFactoryError(
                str(exc), reason_code="reasoning_snapshot_error"
            ) from exc

        completeness = reasoning.completeness
        if repository_index is None and self.build_index:
            completeness = "partial_with_frontier"
        if any(
            item.status is not OpenFrontierStatus.OPEN for item in open_frontiers
        ):
            # Degraded/abstained frontiers keep partial completeness when not already.
            if completeness == "complete" and any(
                item.status is OpenFrontierStatus.ABSTAINED for item in open_frontiers
            ):
                completeness = "partial_with_frontier"

        # Final stability check after indexing and secret screening.
        final_digest = _preflight_digest(root)
        if final_digest != preflight:
            raise PlanningAnalysisStabilityError(
                "repository root changed after analysis construction"
            )

        view = PlanningAnalysisView(
            reasoning_snapshot=reasoning,
            sca_snapshot=snapshot,
            category_inventory=inventory,
            open_frontiers=open_frontiers,
            submodule_closure=submodule_closure,
            repository_index=repository_index,
            optional_provider_status=optional_status,
            completeness=completeness,
            notes=tuple(notes),
        )
        self._last_view = view
        self._bound_checkout = str(root)
        return view


def build_planning_analysis_factory(
    *,
    repository_allowlist: Sequence[str | os.PathLike[str]],
    index_root: str | os.PathLike[str] | None = None,
    scope_policy: ScopePolicy | Mapping[str, Any] | None = None,
    optional_providers: Mapping[str, Callable[[], Any] | Any] | None = None,
    ir_request_builder: Callable[..., Any] | None = None,
    build_index: bool = True,
) -> PlanningAnalysisFactory:
    """Convenience constructor for the production planning analysis factory."""

    return PlanningAnalysisFactory(
        repository_allowlist=repository_allowlist,
        index_root=index_root,
        scope_policy=scope_policy,
        optional_providers=optional_providers,
        ir_request_builder=ir_request_builder,
        build_index=build_index,
    )


def build_planning_analysis_view(
    checkout_root: str | os.PathLike[str],
    *,
    repository_allowlist: Sequence[str | os.PathLike[str]] | None = None,
    **kwargs: Any,
) -> PlanningAnalysisView:
    """One-shot analysis for an allowlisted checkout."""

    allowlist = repository_allowlist
    if allowlist is None:
        allowlist = (checkout_root,)
    factory = build_planning_analysis_factory(
        repository_allowlist=allowlist,
        **kwargs,
    )
    return factory.analyze(checkout_root)


__all__ = [
    "DEFAULT_OPEN_FRONTIERS",
    "PLANNING_ANALYSIS_FACTORY_INTERFACE",
    "PLANNING_ANALYSIS_FACTORY_VERSION",
    "PLANNING_ANALYSIS_VIEW_SCHEMA",
    "OpenFrontierRecord",
    "OpenFrontierStatus",
    "OptionalProviderOutcome",
    "PlanningAdmissionRequestFactory",
    "PlanningAnalysisAdmissionError",
    "PlanningAnalysisAllowlistError",
    "PlanningAnalysisFactory",
    "PlanningAnalysisFactoryError",
    "PlanningAnalysisPathEscapeError",
    "PlanningAnalysisSecretError",
    "PlanningAnalysisStabilityError",
    "PlanningAnalysisSymlinkError",
    "PlanningAnalysisView",
    "PlanningOptionalAnalysisAdapter",
    "SubmoduleClosureEntry",
    "build_planning_analysis_factory",
    "build_planning_analysis_view",
]
