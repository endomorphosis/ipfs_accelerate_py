"""Exact repository snapshot and coverage disposition ledger for SCA.

Wave-0 SwissKnife symbolic contract assurance needs one content-addressed
identity for the Git-tracked primary tree plus any allowlisted working-tree
overlay.  Every tracked path under the reviewed scope policy must receive
exactly one :class:`CoverageDisposition`.  Missing dispositions are never
treated as clean coverage.

This module is deliberately independent of the decision-runtime
``program_behavior.RepositorySnapshot`` contract: it records gitlinks,
dependency lock/tool identities, and explicit exclusions rather than
rejecting them.  Source bodies are never embedded in the snapshot; only
digests, Git object IDs, and typed reasons are retained.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_SNAPSHOT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/sca-repository-snapshot@1"
)
COVERAGE_DISPOSITION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/sca-coverage-disposition@1"
)
SCOPE_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/swissknife-symbolic-contract-scope@1"
)
DEPENDENCY_IDENTITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/sca-dependency-identity@1"
)
GITLINK_RECORD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/sca-gitlink-record@1"
)

REPOSITORY_SNAPSHOT_SCHEMA_VERSION = 1
COVERAGE_DISPOSITION_SCHEMA_VERSION = 1

DEFAULT_SCOPE_CONFIG_RELATIVE = "config/swissknife_symbolic_contract_scope.json"

# Bounded defaults for fail-closed inventory.  Full SwissKnife trees (~6k
# tracked files) fit comfortably under these ceilings.
DEFAULT_MAX_PATHS = 100_000
DEFAULT_MAX_FILE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


class RepositorySnapshotError(RuntimeError):
    """Base exception for an unsafe or incomplete SCA repository snapshot."""


class RepositoryPathEscapeError(RepositorySnapshotError, ValueError):
    """A declared repository path is outside its repository root."""


class SymlinkEscapeError(RepositorySnapshotError, ValueError):
    """A symlink target resolves outside the declared repository root."""


class RepositoryStateError(RepositorySnapshotError):
    """Git metadata is missing, corrupt, or not representable exactly."""


class ScopePolicyError(RepositorySnapshotError, ValueError):
    """The reviewed scope policy is missing, invalid, or incomplete."""


class CoverageIncompleteError(RepositorySnapshotError):
    """At least one tracked path lacks a single explicit disposition."""


class CoverageKind(str, Enum):
    """Exactly-one coverage disposition vocabulary (plan SCA scope contract)."""

    SEMANTIC_AST = "semantic_ast"
    STRUCTURED_DATA = "structured_data"
    TEXT_REFERENCE = "text_reference"
    BINARY_OR_GENERATED = "binary_or_generated"
    DEPENDENCY_TOOL_IDENTITY = "dependency_tool_identity"
    UNSUPPORTED = "unsupported"
    PARSE_FAILURE = "parse_failure"
    EXCLUDED = "excluded"


class EntryKind(str, Enum):
    """Filesystem / Git object kind for one path."""

    REGULAR = "regular"
    SYMLINK = "symlink"
    GITLINK = "gitlink"


class GitStatus(str, Enum):
    """Working-tree / index status bound into the snapshot overlay."""

    CLEAN = "clean"
    MODIFIED = "modified"
    STAGED = "staged"
    STAGED_AND_MODIFIED = "staged_and_modified"
    DELETED = "deleted"
    STAGED_DELETION = "staged_deletion"
    UNTRACKED = "untracked"
    RENAMED = "renamed"
    MODE_CHANGED = "mode_changed"


class DependencyIdentityKind(str, Enum):
    LOCKFILE = "lockfile"
    MANIFEST = "manifest"
    TOOLCHAIN = "toolchain"
    GITLINK = "gitlink"
    DIRECTORY_MARKER = "directory_marker"


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RepositorySnapshotError(
            "snapshot values must be canonical JSON"
        ) from exc


def _identity(prefix: str, value: Any) -> str:
    return (
        f"{prefix}:sha256:"
        + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    )


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def repo_path(value: Any, *, allow_root: bool = False) -> str:
    """Normalize a repository-relative POSIX path; reject escapes."""

    raw = str(value if value is not None else "").replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if "\x00" in raw:
        raise RepositoryPathEscapeError(
            f"repository path contains NUL: {value!r}"
        )
    candidate = PurePosixPath(raw or ".")
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise RepositoryPathEscapeError(
            f"repository path escapes its root: {value!r}"
        )
    normalized = candidate.as_posix()
    if normalized == ".":
        if allow_root:
            return "."
        raise RepositoryPathEscapeError("repository entry path is required")
    # Reject non-canonical forms such as "a//b" or trailing slash after
    # PurePosixPath normalization that would otherwise collapse silently.
    stripped = raw.rstrip("/")
    if stripped and normalized != stripped and PurePosixPath(stripped).as_posix() != normalized:
        raise RepositoryPathEscapeError(
            f"repository path is not canonical: {value!r}"
        )
    return normalized


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _run_git(
    root: Path,
    arguments: Sequence[str],
    *,
    allow_failure: bool = False,
) -> bytes:
    command = (
        "git",
        "-c",
        "core.quotepath=false",
        "-C",
        str(root),
        *arguments,
    )
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise RepositoryStateError("git executable is unavailable") from exc
    if result.returncode and not allow_failure:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RepositoryStateError(
            f"git {' '.join(arguments)} failed: {detail or result.returncode}"
        )
    return result.stdout if not result.returncode else b""


def _suffix(path: str) -> str:
    name = PurePosixPath(path).name
    if name.startswith(".") and name.count(".") == 1:
        return name.lower()
    suffix = PurePosixPath(path).suffix.lower()
    return suffix


def _path_has_prefix(path: str, prefixes: Sequence[str]) -> str | None:
    candidate = PurePosixPath(path)
    for prefix in prefixes:
        if prefix in {".", ""}:
            return prefix
        root = PurePosixPath(prefix)
        if candidate == root or root in candidate.parents:
            return prefix
    return None


def primary_relative_prefixes(
    prefixes: Sequence[str],
    *,
    primary_root: str,
    primary_repository: str = "",
) -> tuple[str, ...]:
    """Expand superproject prefixes into primary-relative inventory forms.

    When inventory paths are rooted at ``swissknife/``, a policy prefix such as
    ``swissknife/node_modules`` must also match the primary-relative path
    ``node_modules/...``.  Unrelated superproject prefixes (for example
    ``hallucinate_app/node_modules``) are left unchanged and only match when
    the inventory path itself carries that prefix.
    """

    heads = {
        item
        for item in (primary_root, primary_repository)
        if item and item not in {".", ""}
    }
    expanded: set[str] = set()
    for prefix in prefixes:
        normalized = repo_path(prefix, allow_root=True)
        expanded.add(normalized)
        for head in heads:
            if normalized == head:
                continue
            if normalized.startswith(f"{head}/"):
                expanded.add(normalized[len(head) + 1 :])
    return tuple(sorted(expanded))


def _path_has_directory_name(path: str, names: Sequence[str]) -> str | None:
    parts = PurePosixPath(path).parts
    for name in names:
        if name in parts:
            return name
    return None


@dataclass(frozen=True)
class ScopePolicy:
    """Reviewed SwissKnife symbolic-contract scope policy."""

    scope_id: str
    primary_repository: str
    primary_root: str
    provider_scopes: tuple[str, ...]
    skip_prefixes: tuple[str, ...]
    skip_directory_names: frozenset[str]
    dependency_directory_names: frozenset[str]
    dependency_lock_files: frozenset[str]
    dependency_manifest_files: frozenset[str]
    semantic_extensions: frozenset[str]
    structured_extensions: frozenset[str]
    text_extensions: frozenset[str]
    binary_extensions: frozenset[str]
    generated_suffixes: frozenset[str]
    generated_path_parts: frozenset[str]
    allow_dirty_analysis: bool
    allowlisted_untracked_suffixes: frozenset[str]
    allowlisted_untracked_exact_names: frozenset[str]
    silent_exclusions_allowed: bool
    tracked_coverage_required: float
    working_tree_overlay_mode: str
    schema_version: int = 1
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope_id", str(self.scope_id or "").strip())
        if not self.scope_id:
            raise ScopePolicyError("scope_id is required")
        object.__setattr__(
            self, "primary_root", repo_path(self.primary_root, allow_root=True)
        )
        object.__setattr__(
            self,
            "provider_scopes",
            tuple(
                sorted(
                    {
                        repo_path(item, allow_root=True)
                        for item in self.provider_scopes
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "skip_prefixes",
            tuple(
                sorted(
                    {repo_path(item, allow_root=True) for item in self.skip_prefixes}
                )
            ),
        )
        if not 0.0 <= float(self.tracked_coverage_required) <= 1.0:
            raise ScopePolicyError("tracked_coverage_required must be in [0, 1]")

    @property
    def policy_id(self) -> str:
        payload = {
            "schema": SCOPE_POLICY_SCHEMA,
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "primary_repository": self.primary_repository,
            "primary_root": self.primary_root,
            "provider_scopes": list(self.provider_scopes),
            "skip_prefixes": list(self.skip_prefixes),
            "skip_directory_names": sorted(self.skip_directory_names),
            "dependency_directory_names": sorted(self.dependency_directory_names),
            "dependency_lock_files": sorted(self.dependency_lock_files),
            "dependency_manifest_files": sorted(self.dependency_manifest_files),
            "semantic_extensions": sorted(self.semantic_extensions),
            "structured_extensions": sorted(self.structured_extensions),
            "text_extensions": sorted(self.text_extensions),
            "binary_extensions": sorted(self.binary_extensions),
            "generated_suffixes": sorted(self.generated_suffixes),
            "generated_path_parts": sorted(self.generated_path_parts),
            "allow_dirty_analysis": self.allow_dirty_analysis,
            "allowlisted_untracked_suffixes": sorted(
                self.allowlisted_untracked_suffixes
            ),
            "allowlisted_untracked_exact_names": sorted(
                self.allowlisted_untracked_exact_names
            ),
            "silent_exclusions_allowed": self.silent_exclusions_allowed,
            "tracked_coverage_required": self.tracked_coverage_required,
            "working_tree_overlay_mode": self.working_tree_overlay_mode,
        }
        return _identity("sca-scope-policy", payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCOPE_POLICY_SCHEMA,
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "policy_id": self.policy_id,
            "primary_repository": self.primary_repository,
            "primary_root": self.primary_root,
            "provider_scopes": list(self.provider_scopes),
            "skip_prefixes": list(self.skip_prefixes),
            "skip_directory_names": sorted(self.skip_directory_names),
            "dependency_directory_names": sorted(self.dependency_directory_names),
            "dependency_lock_files": sorted(self.dependency_lock_files),
            "dependency_manifest_files": sorted(self.dependency_manifest_files),
            "disposition_rules": {
                "semantic_extensions": sorted(self.semantic_extensions),
                "structured_extensions": sorted(self.structured_extensions),
                "text_extensions": sorted(self.text_extensions),
                "binary_extensions": sorted(self.binary_extensions),
                "generated_suffixes": sorted(self.generated_suffixes),
                "generated_path_parts": sorted(self.generated_path_parts),
            },
            "working_tree_overlay": {
                "mode": self.working_tree_overlay_mode,
                "allow_dirty_analysis": self.allow_dirty_analysis,
                "allowlisted_untracked_suffixes": sorted(
                    self.allowlisted_untracked_suffixes
                ),
                "allowlisted_untracked_exact_names": sorted(
                    self.allowlisted_untracked_exact_names
                ),
            },
            "silent_exclusions_allowed": self.silent_exclusions_allowed,
            "tracked_coverage_required": self.tracked_coverage_required,
        }

    def untracked_allowed(self, path: str) -> bool:
        name = PurePosixPath(path).name
        if name in self.allowlisted_untracked_exact_names:
            return True
        suffix = _suffix(path)
        return suffix in self.allowlisted_untracked_suffixes


def _as_string_set(value: Any, *, field_name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        raise ScopePolicyError(f"{field_name} must be a list of strings")
    try:
        items = list(value)
    except TypeError as exc:
        raise ScopePolicyError(f"{field_name} must be a list of strings") from exc
    result: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        result.add(text)
    return frozenset(result)


def _as_extension_set(value: Any, *, field_name: str) -> frozenset[str]:
    result: set[str] = set()
    for item in _as_string_set(value, field_name=field_name):
        text = item.lower()
        if not text.startswith("."):
            text = f".{text}"
        result.add(text)
    return frozenset(result)


def scope_policy_from_mapping(value: Mapping[str, Any]) -> ScopePolicy:
    """Parse and validate a reviewed scope policy mapping."""

    if not isinstance(value, Mapping):
        raise ScopePolicyError("scope policy must be a mapping")
    schema = str(value.get("schema") or "").strip()
    if schema and schema != SCOPE_POLICY_SCHEMA:
        raise ScopePolicyError(f"unsupported scope policy schema: {schema!r}")
    schema_version = int(value.get("schemaVersion") or value.get("schema_version") or 1)
    if schema_version != 1:
        raise ScopePolicyError(
            f"unsupported scope policy schema version: {schema_version}"
        )
    overlay = value.get("workingTreeOverlay") or value.get("working_tree_overlay") or {}
    if overlay is None:
        overlay = {}
    if not isinstance(overlay, Mapping):
        raise ScopePolicyError("workingTreeOverlay must be a mapping")
    rules = value.get("dispositionRules") or value.get("disposition_rules") or {}
    if rules is None:
        rules = {}
    if not isinstance(rules, Mapping):
        raise ScopePolicyError("dispositionRules must be a mapping")

    allow_dirty = overlay.get("allowDirtyAnalysis")
    if allow_dirty is None:
        allow_dirty = overlay.get("allow_dirty_analysis", True)

    return ScopePolicy(
        scope_id=str(value.get("scopeId") or value.get("scope_id") or "").strip(),
        primary_repository=str(
            value.get("primaryRepository") or value.get("primary_repository") or ""
        ).strip(),
        primary_root=str(
            value.get("primaryRoot") or value.get("primary_root") or "."
        ).strip()
        or ".",
        provider_scopes=tuple(
            _as_string_set(
                value.get("providerScopes") or value.get("provider_scopes"),
                field_name="providerScopes",
            )
        ),
        skip_prefixes=tuple(
            _as_string_set(
                value.get("skipPrefixes") or value.get("skip_prefixes"),
                field_name="skipPrefixes",
            )
        ),
        skip_directory_names=_as_string_set(
            value.get("skipDirectoryNames") or value.get("skip_directory_names"),
            field_name="skipDirectoryNames",
        ),
        dependency_directory_names=_as_string_set(
            value.get("dependencyDirectoryNames")
            or value.get("dependency_directory_names"),
            field_name="dependencyDirectoryNames",
        ),
        dependency_lock_files=_as_string_set(
            value.get("dependencyLockFiles") or value.get("dependency_lock_files"),
            field_name="dependencyLockFiles",
        ),
        dependency_manifest_files=_as_string_set(
            value.get("dependencyManifestFiles")
            or value.get("dependency_manifest_files"),
            field_name="dependencyManifestFiles",
        ),
        semantic_extensions=_as_extension_set(
            rules.get("semanticExtensions") or rules.get("semantic_extensions"),
            field_name="semanticExtensions",
        ),
        structured_extensions=_as_extension_set(
            rules.get("structuredExtensions") or rules.get("structured_extensions"),
            field_name="structuredExtensions",
        ),
        text_extensions=_as_extension_set(
            rules.get("textExtensions") or rules.get("text_extensions"),
            field_name="textExtensions",
        ),
        binary_extensions=_as_extension_set(
            rules.get("binaryExtensions") or rules.get("binary_extensions"),
            field_name="binaryExtensions",
        ),
        generated_suffixes=frozenset(
            item.lower() if item.startswith(".") else f".{item.lower()}"
            for item in _as_string_set(
                rules.get("generatedSuffixes") or rules.get("generated_suffixes"),
                field_name="generatedSuffixes",
            )
        ),
        generated_path_parts=_as_string_set(
            rules.get("generatedPathParts") or rules.get("generated_path_parts"),
            field_name="generatedPathParts",
        ),
        allow_dirty_analysis=bool(allow_dirty),
        allowlisted_untracked_suffixes=_as_extension_set(
            overlay.get("allowlistedUntrackedSuffixes")
            or overlay.get("allowlisted_untracked_suffixes"),
            field_name="allowlistedUntrackedSuffixes",
        ),
        allowlisted_untracked_exact_names=_as_string_set(
            overlay.get("allowlistedUntrackedExactNames")
            or overlay.get("allowlisted_untracked_exact_names"),
            field_name="allowlistedUntrackedExactNames",
        ),
        silent_exclusions_allowed=bool(
            value.get("silentExclusionsAllowed")
            if value.get("silentExclusionsAllowed") is not None
            else value.get("silent_exclusions_allowed", False)
        ),
        tracked_coverage_required=float(
            value.get("trackedCoverageRequired")
            if value.get("trackedCoverageRequired") is not None
            else value.get("tracked_coverage_required", 1.0)
        ),
        working_tree_overlay_mode=str(
            overlay.get("mode")
            or "tracked_plus_allowlisted_untracked_source"
        ).strip(),
        schema_version=schema_version,
        raw=MappingProxyType(dict(value)),
    )


def load_scope_policy(path: str | os.PathLike[str] | None = None) -> ScopePolicy:
    """Load the reviewed scope policy JSON from disk."""

    candidate = Path(path) if path is not None else Path(DEFAULT_SCOPE_CONFIG_RELATIVE)
    try:
        text = candidate.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScopePolicyError(f"scope policy is unreadable: {candidate}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScopePolicyError(f"scope policy is not valid JSON: {candidate}") from exc
    if not isinstance(payload, Mapping):
        raise ScopePolicyError("scope policy root must be a JSON object")
    return scope_policy_from_mapping(payload)


def default_scope_policy_path(repository_root: str | os.PathLike[str]) -> Path:
    """Resolve the default scope config relative to a superproject root."""

    root = Path(repository_root)
    direct = root / DEFAULT_SCOPE_CONFIG_RELATIVE
    if direct.is_file():
        return direct
    # When the inventory root is the primary submodule itself, walk up to the
    # superproject config location used by the SCA supervisor profile.
    parent = root.parent / DEFAULT_SCOPE_CONFIG_RELATIVE
    if parent.is_file():
        return parent
    return direct


@dataclass(frozen=True)
class DependencyIdentity:
    """Lockfile, manifest, gitlink, or toolchain identity for dependencies."""

    kind: DependencyIdentityKind
    path: str
    digest: str
    tool_name: str = ""
    tool_version: str = ""
    git_object_id: str = ""
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", DependencyIdentityKind(self.kind))
        object.__setattr__(self, "path", repo_path(self.path))
        object.__setattr__(self, "digest", str(self.digest or "").strip())
        if not self.digest:
            raise RepositoryStateError(
                f"dependency identity requires a digest at {self.path}"
            )

    @property
    def identity_id(self) -> str:
        return _identity(
            "sca-dependency-identity",
            {
                "schema": DEPENDENCY_IDENTITY_SCHEMA,
                "kind": self.kind.value,
                "path": self.path,
                "digest": self.digest,
                "tool_name": self.tool_name,
                "tool_version": self.tool_version,
                "git_object_id": self.git_object_id,
                "reason_code": self.reason_code,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEPENDENCY_IDENTITY_SCHEMA,
            "identity_id": self.identity_id,
            "kind": self.kind.value,
            "path": self.path,
            "digest": self.digest,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "git_object_id": self.git_object_id,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class GitlinkRecord:
    """Recursive submodule / gitlink identity (not expanded as source)."""

    path: str
    commit_id: str
    mode: str = "160000"
    head_object_id: str = ""
    index_object_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", repo_path(self.path))
        commit = str(self.commit_id or self.head_object_id or "").strip().lower()
        if not commit or any(char not in "0123456789abcdef" for char in commit):
            raise RepositoryStateError(f"invalid gitlink commit at {self.path}")
        object.__setattr__(self, "commit_id", commit)
        object.__setattr__(self, "mode", str(self.mode or "160000"))

    @property
    def gitlink_id(self) -> str:
        return _identity(
            "sca-gitlink",
            {
                "schema": GITLINK_RECORD_SCHEMA,
                "path": self.path,
                "commit_id": self.commit_id,
                "mode": self.mode,
                "head_object_id": self.head_object_id,
                "index_object_id": self.index_object_id,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GITLINK_RECORD_SCHEMA,
            "gitlink_id": self.gitlink_id,
            "path": self.path,
            "commit_id": self.commit_id,
            "mode": self.mode,
            "head_object_id": self.head_object_id,
            "index_object_id": self.index_object_id,
        }


@dataclass(frozen=True)
class CoverageDisposition:
    """Exactly one coverage disposition for one repository path."""

    path: str
    kind: CoverageKind
    git_status: GitStatus
    entry_kind: EntryKind
    reason_code: str
    policy_rule: str
    content_digest: str = ""
    git_mode: str = ""
    git_object_id: str = ""
    rename_from: str = ""
    tracked: bool = True
    overlay: bool = False
    dependency_identity_id: str = ""
    schema_version: int = COVERAGE_DISPOSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", repo_path(self.path))
        object.__setattr__(self, "kind", CoverageKind(self.kind))
        object.__setattr__(self, "git_status", GitStatus(self.git_status))
        object.__setattr__(self, "entry_kind", EntryKind(self.entry_kind))
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        object.__setattr__(self, "policy_rule", str(self.policy_rule or "").strip())
        if not self.reason_code:
            raise RepositoryStateError(
                f"coverage disposition requires reason_code at {self.path}"
            )
        if not self.policy_rule:
            raise RepositoryStateError(
                f"coverage disposition requires policy_rule at {self.path}"
            )
        if self.rename_from:
            object.__setattr__(self, "rename_from", repo_path(self.rename_from))
        if int(self.schema_version) != COVERAGE_DISPOSITION_SCHEMA_VERSION:
            raise RepositoryStateError(
                f"unsupported coverage disposition version at {self.path}"
            )

    @property
    def disposition_id(self) -> str:
        return _identity(
            "sca-coverage-disposition",
            self._content_dict(),
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": COVERAGE_DISPOSITION_SCHEMA,
            "schema_version": self.schema_version,
            "path": self.path,
            "kind": self.kind.value,
            "git_status": self.git_status.value,
            "entry_kind": self.entry_kind.value,
            "reason_code": self.reason_code,
            "policy_rule": self.policy_rule,
            "content_digest": self.content_digest,
            "git_mode": self.git_mode,
            "git_object_id": self.git_object_id,
            "rename_from": self.rename_from,
            "tracked": bool(self.tracked),
            "overlay": bool(self.overlay),
            "dependency_identity_id": self.dependency_identity_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"disposition_id": self.disposition_id, **self._content_dict()}


@dataclass(frozen=True)
class RepositorySnapshotStats:
    tracked_path_count: int
    disposition_count: int
    overlay_path_count: int
    excluded_path_count: int
    dependency_identity_count: int
    gitlink_count: int
    dirty_path_count: int
    deleted_path_count: int
    untracked_path_count: int
    semantic_path_count: int
    unsupported_path_count: int
    hashed_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name)) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class RepositorySnapshot:
    """Canonical SCA snapshot identity and exhaustive disposition ledger."""

    primary_root: str
    head_commit_id: str
    head_tree_id: str
    index_tree_id: str
    scope_policy_id: str
    scope_id: str
    dispositions: tuple[CoverageDisposition, ...]
    dependency_identities: tuple[DependencyIdentity, ...]
    gitlinks: tuple[GitlinkRecord, ...]
    stats: RepositorySnapshotStats
    schema_version: int = REPOSITORY_SNAPSHOT_SCHEMA_VERSION
    repository_root: str = ""
    git_directory: str = ""
    allow_dirty_analysis: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "primary_root", repo_path(self.primary_root, allow_root=True)
        )
        dispositions = tuple(sorted(self.dispositions, key=lambda item: item.path))
        paths = [item.path for item in dispositions]
        if len(paths) != len(set(paths)):
            raise CoverageIncompleteError(
                "coverage dispositions must be unique per path"
            )
        object.__setattr__(self, "dispositions", dispositions)
        object.__setattr__(
            self,
            "dependency_identities",
            tuple(sorted(self.dependency_identities, key=lambda item: item.path)),
        )
        object.__setattr__(
            self,
            "gitlinks",
            tuple(sorted(self.gitlinks, key=lambda item: item.path)),
        )
        if int(self.schema_version) != REPOSITORY_SNAPSHOT_SCHEMA_VERSION:
            raise RepositoryStateError("unsupported repository snapshot version")

    @property
    def is_clean(self) -> bool:
        return all(item.git_status is GitStatus.CLEAN for item in self.dispositions)

    @property
    def dirty(self) -> bool:
        return not self.is_clean

    def disposition_for_path(self, path: str) -> CoverageDisposition | None:
        normalized = repo_path(path)
        for item in self.dispositions:
            if item.path == normalized:
                return item
        return None

    def tracked_dispositions(self) -> tuple[CoverageDisposition, ...]:
        return tuple(item for item in self.dispositions if item.tracked)

    def assert_exhaustive_tracked_coverage(self) -> None:
        """Fail closed when any tracked path lacks exactly one disposition.

        Callers that materialize dispositions through
        :func:`build_repository_snapshot` already satisfy this invariant; the
        method exists for reconstructed or hand-assembled ledgers.
        """

        tracked = [item for item in self.dispositions if item.tracked]
        if not tracked and self.stats.tracked_path_count:
            raise CoverageIncompleteError("tracked path ledger is empty")
        if len(tracked) != self.stats.tracked_path_count:
            raise CoverageIncompleteError(
                "tracked disposition count does not match inventory"
            )
        paths = [item.path for item in tracked]
        if len(paths) != len(set(paths)):
            raise CoverageIncompleteError("duplicate tracked dispositions")

    def _content_dict(self) -> dict[str, Any]:
        # Absolute filesystem locations are verification metadata only.
        return {
            "schema": REPOSITORY_SNAPSHOT_SCHEMA,
            "schema_version": self.schema_version,
            "primary_root": self.primary_root,
            "head_commit_id": self.head_commit_id,
            "head_tree_id": self.head_tree_id,
            "index_tree_id": self.index_tree_id,
            "scope_policy_id": self.scope_policy_id,
            "scope_id": self.scope_id,
            "allow_dirty_analysis": bool(self.allow_dirty_analysis),
            "dispositions": [item.to_dict() for item in self.dispositions],
            "dependency_identities": [
                item.to_dict() for item in self.dependency_identities
            ],
            "gitlinks": [item.to_dict() for item in self.gitlinks],
        }

    @property
    def snapshot_id(self) -> str:
        return _identity("sca-repository-snapshot", self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "snapshot_id": self.snapshot_id,
            "is_clean": self.is_clean,
            "stats": self.stats.to_dict(),
            "repository_root": self.repository_root,
            "git_directory": self.git_directory,
        }

    def to_json(self) -> str:
        return _canonical_json_bytes(self.to_dict()).decode("utf-8")

    def coverage_inventory(self) -> dict[str, Any]:
        """Compact inventory suitable for analyzer-health classification."""

        tracked = self.tracked_dispositions()
        excluded = sum(1 for item in tracked if item.kind is CoverageKind.EXCLUDED)
        parse_failures = sum(
            1 for item in tracked if item.kind is CoverageKind.PARSE_FAILURE
        )
        semantic = sum(
            1 for item in tracked if item.kind is CoverageKind.SEMANTIC_AST
        )
        return {
            "git_roots": 1 if self.head_commit_id else 0,
            "expected_git_root_count": 1,
            "tracked_files": len(tracked),
            "eligible_files": max(0, len(tracked) - excluded),
            "parsed_files": semantic,
            "cache_hits": 0,
            "excluded_files": excluded,
            "parser_failures": parse_failures,
            "raw_candidates": 0,
            "seen_candidates": 0,
            "deduplicated_candidates": 0,
            "rejected_candidates": 0,
            "appended_tasks": 0,
            "coverage_complete": len(tracked) == self.stats.tracked_path_count
            and len(tracked) == len({item.path for item in tracked}),
            "disposition_count": len(self.dispositions),
            "gitlink_count": len(self.gitlinks),
            "dependency_identity_count": len(self.dependency_identities),
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True)
class _GitEntry:
    mode: str
    object_id: str
    kind: EntryKind


@dataclass(frozen=True)
class _WorktreeEntry:
    mode: str
    kind: EntryKind
    digest: str
    size_bytes: int


def _entry_kind_for_mode(mode: str) -> EntryKind:
    if mode == "120000":
        return EntryKind.SYMLINK
    if mode == "160000":
        return EntryKind.GITLINK
    return EntryKind.REGULAR


def _parse_ls_tree(root: Path, treeish: str) -> dict[str, _GitEntry]:
    output = _run_git(
        root,
        ("ls-tree", "-rz", "--full-tree", treeish),
        allow_failure=True,
    )
    result: dict[str, _GitEntry] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RepositoryStateError(
                f"{treeish} contains an undecodable entry"
            ) from exc
        normalized = repo_path(path)
        if kind == "blob":
            entry_kind = _entry_kind_for_mode(mode)
        elif kind == "commit":
            entry_kind = EntryKind.GITLINK
            mode = "160000"
        elif kind == "tree":
            # Recursive ls-tree already expands trees; bare trees are not paths.
            continue
        else:
            raise RepositoryStateError(
                f"unsupported {treeish} entry kind {kind!r} at {path!r}"
            )
        result[normalized] = _GitEntry(mode, object_id.lower(), entry_kind)
    return result


def _parse_index_entries(root: Path) -> dict[str, _GitEntry]:
    output = _run_git(root, ("ls-files", "--stage", "-z"))
    result: dict[str, _GitEntry] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RepositoryStateError(
                "index contains an undecodable entry"
            ) from exc
        if stage != "0":
            raise RepositoryStateError(
                f"unmerged index entry is unsupported at {path!r}"
            )
        normalized = repo_path(path)
        result[normalized] = _GitEntry(
            mode, object_id.lower(), _entry_kind_for_mode(mode)
        )
    return result


def _parse_status_porcelain(root: Path) -> dict[str, dict[str, str]]:
    """Return path -> {xy, rename_from} from ``git status --porcelain=v1 -z``."""

    output = _run_git(root, ("status", "--porcelain=v1", "-z", "--untracked-files=all"))
    result: dict[str, dict[str, str]] = {}
    records = output.split(b"\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        try:
            text = record.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryStateError(
                "status contains an undecodable path"
            ) from exc
        if len(text) < 3:
            continue
        xy = text[:2]
        path_text = text[3:]
        rename_from = ""
        # Rename/copy records are followed by the source path in -z format.
        if xy[0] in {"R", "C"} or xy[1] in {"R", "C"}:
            if index < len(records) and records[index]:
                try:
                    rename_from = repo_path(records[index].decode("utf-8"))
                except UnicodeDecodeError as exc:
                    raise RepositoryStateError(
                        "status rename source is undecodable"
                    ) from exc
                index += 1
            # Porcelain path field for rename is the destination.
        if " -> " in path_text and not rename_from:
            source, dest = path_text.split(" -> ", 1)
            rename_from = repo_path(source.strip())
            path_text = dest.strip()
        path = repo_path(path_text)
        result[path] = {"xy": xy, "rename_from": rename_from}
    return result


def _stable_worktree_entry(
    root: Path,
    relative: str,
    *,
    max_file_bytes: int,
) -> _WorktreeEntry:
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        before = path.lstat()
    except OSError as exc:
        raise RepositoryStateError(
            f"required input is unreadable: {relative}"
        ) from exc
    if stat.S_ISLNK(before.st_mode):
        try:
            target = os.readlink(path)
            after = path.lstat()
        except OSError as exc:
            raise RepositoryStateError(
                f"required symlink is unreadable: {relative}"
            ) from exc
        if (
            before.st_ino,
            before.st_dev,
            before.st_mtime_ns,
            before.st_size,
        ) != (
            after.st_ino,
            after.st_dev,
            after.st_mtime_ns,
            after.st_size,
        ):
            raise RepositoryStateError(
                f"symlink changed while hashing: {relative}"
            )
        resolved = (path.parent / target).resolve(strict=False)
        if not _is_within(resolved, root):
            raise SymlinkEscapeError(
                f"symlink escapes repository root: {relative!r} -> {target!r}"
            )
        data = os.fsencode(target)
        if len(data) > max_file_bytes:
            raise RepositoryStateError(
                f"symlink target exceeds bound at {relative}"
            )
        return _WorktreeEntry(
            "120000",
            EntryKind.SYMLINK,
            _sha256_bytes(data),
            len(data),
        )
    if not stat.S_ISREG(before.st_mode):
        raise RepositoryStateError(
            f"required input is not a regular file or symlink: {relative}"
        )
    if before.st_size > max_file_bytes:
        raise RepositoryStateError(
            f"required input exceeds {max_file_bytes} bytes: {relative}"
        )
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_ino,
                opened.st_dev,
                opened.st_size,
                opened.st_mtime_ns,
            ) != (
                before.st_ino,
                before.st_dev,
                before.st_size,
                before.st_mtime_ns,
            ):
                raise RepositoryStateError(
                    f"file changed before hashing: {relative}"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor, min(1024 * 1024, max_file_bytes + 1 - total)
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > max_file_bytes:
                    raise RepositoryStateError(
                        f"required input exceeds {max_file_bytes} bytes: {relative}"
                    )
                chunks.append(chunk)
            data = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RepositoryStateError(
            f"required input is unreadable: {relative}"
        ) from exc
    if (
        after.st_ino,
        after.st_dev,
        after.st_size,
        after.st_mtime_ns,
    ) != (
        before.st_ino,
        before.st_dev,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise RepositoryStateError(f"file changed while hashing: {relative}")
    mode = "100755" if (before.st_mode & stat.S_IXUSR) else "100644"
    return _WorktreeEntry(
        mode,
        EntryKind.REGULAR,
        _sha256_bytes(data),
        len(data),
    )


def _git_blob_digest(root: Path, object_id: str) -> str:
    data = _run_git(root, ("cat-file", "-p", object_id))
    return _sha256_bytes(data)


def _status_from_xy(
    xy: str,
    *,
    rename_from: str,
    in_head: bool,
    in_index: bool,
    in_worktree: bool,
) -> GitStatus:
    if rename_from:
        return GitStatus.RENAMED
    x, y = (xy + "  ")[:2]
    if x == "?" and y == "?":
        return GitStatus.UNTRACKED
    if x == "D" and y == "D":
        return GitStatus.DELETED
    if x == "D" and y == " ":
        return GitStatus.STAGED_DELETION
    if x == " " and y == "D":
        return GitStatus.DELETED
    if x in {"A", "M", "T", "R", "C"} and y in {"M", "T", "D"}:
        if y == "D":
            return GitStatus.DELETED
        return GitStatus.STAGED_AND_MODIFIED
    if x in {"A", "M", "T", "R", "C"} and y == " ":
        return GitStatus.STAGED
    if x == " " and y in {"M", "T"}:
        return GitStatus.MODIFIED
    if not in_worktree and in_head:
        return GitStatus.DELETED
    if not in_head and in_worktree and not in_index:
        return GitStatus.UNTRACKED
    if in_head and in_index and in_worktree:
        return GitStatus.CLEAN
    return GitStatus.MODIFIED


def classify_coverage_kind(
    path: str,
    *,
    policy: ScopePolicy,
    entry_kind: EntryKind,
    basename: str | None = None,
    skip_prefixes: Sequence[str] | None = None,
) -> tuple[CoverageKind, str, str]:
    """Return (kind, reason_code, policy_rule) for one path under ``policy``."""

    name = basename if basename is not None else PurePosixPath(path).name
    effective_prefixes = (
        tuple(skip_prefixes)
        if skip_prefixes is not None
        else policy.skip_prefixes
    )
    skip_prefix = _path_has_prefix(path, effective_prefixes)
    if skip_prefix is not None:
        return (
            CoverageKind.EXCLUDED,
            "excluded_prefix",
            f"skip_prefixes:{skip_prefix}",
        )
    skip_dir = _path_has_directory_name(path, sorted(policy.skip_directory_names))
    if skip_dir is not None:
        return (
            CoverageKind.EXCLUDED,
            "excluded_directory",
            f"skip_directory_names:{skip_dir}",
        )
    if entry_kind is EntryKind.GITLINK:
        return (
            CoverageKind.DEPENDENCY_TOOL_IDENTITY,
            "gitlink_submodule",
            "entry_kind:gitlink",
        )
    dep_dir = _path_has_directory_name(
        path, sorted(policy.dependency_directory_names)
    )
    if dep_dir is not None:
        return (
            CoverageKind.DEPENDENCY_TOOL_IDENTITY,
            "dependency_directory",
            f"dependency_directory_names:{dep_dir}",
        )
    if name in policy.dependency_lock_files:
        return (
            CoverageKind.DEPENDENCY_TOOL_IDENTITY,
            "dependency_lockfile",
            f"dependency_lock_files:{name}",
        )
    if name in policy.dependency_manifest_files:
        # Manifests are structured package identity, not recursive deps.
        return (
            CoverageKind.DEPENDENCY_TOOL_IDENTITY,
            "dependency_manifest",
            f"dependency_manifest_files:{name}",
        )
    generated_part = _path_has_directory_name(
        path, sorted(policy.generated_path_parts)
    )
    if generated_part is not None:
        return (
            CoverageKind.BINARY_OR_GENERATED,
            "generated_path_part",
            f"generated_path_parts:{generated_part}",
        )
    # Multi-suffix generated markers such as ".d.ts" and ".js.map".
    lower_name = name.lower()
    for generated_suffix in sorted(policy.generated_suffixes, key=len, reverse=True):
        if lower_name.endswith(generated_suffix):
            return (
                CoverageKind.BINARY_OR_GENERATED,
                "generated_suffix",
                f"generated_suffixes:{generated_suffix}",
            )
    suffix = _suffix(path)
    if suffix in policy.binary_extensions:
        return (
            CoverageKind.BINARY_OR_GENERATED,
            "binary_extension",
            f"binary_extensions:{suffix}",
        )
    if suffix in policy.semantic_extensions:
        return (
            CoverageKind.SEMANTIC_AST,
            "semantic_extension",
            f"semantic_extensions:{suffix}",
        )
    if suffix in policy.structured_extensions:
        return (
            CoverageKind.STRUCTURED_DATA,
            "structured_extension",
            f"structured_extensions:{suffix}",
        )
    if suffix in policy.text_extensions:
        return (
            CoverageKind.TEXT_REFERENCE,
            "text_extension",
            f"text_extensions:{suffix}",
        )
    if entry_kind is EntryKind.SYMLINK:
        return (
            CoverageKind.TEXT_REFERENCE,
            "symlink_target_text",
            "entry_kind:symlink",
        )
    return (
        CoverageKind.UNSUPPORTED,
        "unsupported_extension",
        "disposition_rules:fallback_unsupported",
    )


def _resolve_inventory_root(
    repository_root: Path,
    policy: ScopePolicy,
) -> tuple[Path, str]:
    """Return (git_worktree_root, primary_root_label)."""

    try:
        super_root = repository_root.resolve(strict=True)
    except OSError as exc:
        raise RepositoryStateError("repository root is unreadable") from exc

    primary_rel = policy.primary_root
    if primary_rel in {".", ""}:
        primary = super_root
        label = "."
    else:
        primary = super_root.joinpath(*PurePosixPath(primary_rel).parts)
        if not primary.exists():
            # Allow inventory when the provided root *is* the primary checkout
            # (tests and leased swissknife worktrees).
            if super_root.name == policy.primary_repository or (
                super_root / ".git"
            ).exists():
                primary = super_root
                label = "."
            else:
                raise RepositoryStateError(
                    f"primary root does not exist: {primary_rel}"
                )
        else:
            label = primary_rel

    try:
        primary = primary.resolve(strict=True)
    except OSError as exc:
        raise RepositoryStateError(
            f"primary root is unreadable: {primary_rel}"
        ) from exc

    discovered = _run_git(primary, ("rev-parse", "--show-toplevel"))
    try:
        git_root = Path(discovered.decode("utf-8").strip()).resolve(strict=True)
    except (OSError, UnicodeDecodeError) as exc:
        raise RepositoryStateError(
            "could not resolve Git repository root for primary tree"
        ) from exc
    if git_root != primary:
        raise RepositoryPathEscapeError(
            "primary root must name the exact Git worktree root for inventory"
        )
    return primary, label


def build_repository_snapshot(
    repository_root: str | os.PathLike[str],
    *,
    scope_policy: ScopePolicy | Mapping[str, Any] | None = None,
    scope_config_path: str | os.PathLike[str] | None = None,
    allow_dirty_analysis: bool | None = None,
    max_paths: int = DEFAULT_MAX_PATHS,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    include_provider_scopes: bool = False,
) -> RepositorySnapshot:
    """Build an exact SCA snapshot and exhaustive coverage disposition ledger.

    Parameters
    ----------
    repository_root:
        Superproject root (containing ``swissknife/``) or the primary Git
        worktree itself when tests or leased checkouts inventory only that tree.
    scope_policy / scope_config_path:
        Reviewed policy.  When both are omitted, the default config path under
        the repository root is loaded.
    allow_dirty_analysis:
        When false, only clean HEAD/index tracked paths are inventoried and
        dirty overlays are rejected if present.
    include_provider_scopes:
        Reserved for multi-root inventory.  Provider roots are recorded in the
        policy identity but are not expanded as source authority by default.
    """

    del include_provider_scopes  # reserved; policy still binds provider scopes

    if max_paths < 1 or max_file_bytes < 1 or max_total_bytes < 1:
        raise ValueError("resource bounds must be positive")

    if scope_policy is None:
        config_path = (
            Path(scope_config_path)
            if scope_config_path is not None
            else default_scope_policy_path(repository_root)
        )
        policy = load_scope_policy(config_path)
    elif isinstance(scope_policy, ScopePolicy):
        policy = scope_policy
    else:
        policy = scope_policy_from_mapping(scope_policy)

    dirty_enabled = (
        policy.allow_dirty_analysis
        if allow_dirty_analysis is None
        else bool(allow_dirty_analysis)
    )

    primary, primary_label = _resolve_inventory_root(Path(repository_root), policy)
    effective_skip_prefixes = primary_relative_prefixes(
        policy.skip_prefixes,
        primary_root=policy.primary_root
        if primary_label != "."
        else primary_label,
        primary_repository=policy.primary_repository,
    )
    # When inventorying the primary worktree itself, also expand prefixes that
    # were authored relative to the superproject primary root name.
    if primary_label == ".":
        effective_skip_prefixes = primary_relative_prefixes(
            policy.skip_prefixes,
            primary_root=policy.primary_root,
            primary_repository=policy.primary_repository,
        )
    git_directory = (
        _run_git(primary, ("rev-parse", "--absolute-git-dir"))
        .decode("utf-8", "strict")
        .strip()
    )
    head_commit = (
        _run_git(primary, ("rev-parse", "--verify", "HEAD"), allow_failure=True)
        .decode("ascii", "strict")
        .strip()
        .lower()
    )
    head_tree = (
        _run_git(
            primary, ("rev-parse", "--verify", "HEAD^{tree}"), allow_failure=True
        )
        .decode("ascii", "strict")
        .strip()
        .lower()
    )
    index_tree = (
        _run_git(primary, ("write-tree",)).decode("ascii", "strict").strip().lower()
    )

    head = _parse_ls_tree(primary, "HEAD") if head_commit else {}
    index = _parse_index_entries(primary)
    # Always inspect porcelain status so clean-tree gates remain fail-closed
    # even when dirty overlays are not admitted into the snapshot identity.
    status = _parse_status_porcelain(primary)

    if not dirty_enabled and status:
        raise RepositoryStateError(
            "dirty working tree is not allowed when allow_dirty_analysis is false"
        )
    if not dirty_enabled:
        status = {}

    # Paths that participate in the tracked ledger: HEAD U index.
    tracked_paths = sorted(set(head) | set(index))
    if len(tracked_paths) > max_paths:
        raise RepositoryStateError(
            f"tracked inventory exceeds {max_paths} paths"
        )

    # Rename inference for cases porcelain did not mark explicitly.
    rename_from_map: dict[str, str] = {
        path: meta["rename_from"]
        for path, meta in status.items()
        if meta.get("rename_from")
    }

    worktree_cache: dict[str, _WorktreeEntry] = {}
    hashed_bytes = 0

    def worktree_entry(path: str) -> _WorktreeEntry | None:
        nonlocal hashed_bytes
        if path in worktree_cache:
            return worktree_cache[path]
        candidate = primary.joinpath(*PurePosixPath(path).parts)
        if not candidate.exists() and not candidate.is_symlink():
            return None
        entry = _stable_worktree_entry(
            primary, path, max_file_bytes=max_file_bytes
        )
        hashed_bytes += entry.size_bytes
        if hashed_bytes > max_total_bytes:
            raise RepositoryStateError(
                f"inventory exceeds {max_total_bytes} hashed bytes"
            )
        worktree_cache[path] = entry
        return entry

    # Digest cache for Git blobs used when worktree is absent or clean.
    blob_digest_cache: dict[str, str] = {}

    def object_digest(object_id: str) -> str:
        if object_id in blob_digest_cache:
            return blob_digest_cache[object_id]
        digest = _git_blob_digest(primary, object_id)
        blob_digest_cache[object_id] = digest
        return digest

    dispositions: list[CoverageDisposition] = []
    dependency_identities: list[DependencyIdentity] = []
    gitlinks: list[GitlinkRecord] = []
    dependency_ids_by_path: dict[str, str] = {}

    def remember_dependency(identity: DependencyIdentity) -> str:
        dependency_identities.append(identity)
        dependency_ids_by_path[identity.path] = identity.identity_id
        return identity.identity_id

    for path in tracked_paths:
        head_item = head.get(path)
        index_item = index.get(path)
        status_meta = status.get(path, {})
        xy = status_meta.get("xy", "  ")
        rename_from = rename_from_map.get(path, status_meta.get("rename_from", ""))

        entry_kind = (
            (index_item or head_item).kind
            if (index_item or head_item) is not None
            else EntryKind.REGULAR
        )
        git_mode = (index_item or head_item).mode if (index_item or head_item) else ""
        git_object_id = (
            (index_item or head_item).object_id if (index_item or head_item) else ""
        )

        in_head = head_item is not None
        in_index = index_item is not None
        wt = None
        if dirty_enabled and entry_kind is not EntryKind.GITLINK:
            # Only materialize worktree bytes for dirty or missing status paths.
            if path in status or not in_head or not in_index:
                wt = worktree_entry(path)
            elif xy.strip():
                wt = worktree_entry(path)
        in_worktree = wt is not None
        if not dirty_enabled:
            in_worktree = in_index or in_head

        if path not in status and in_head and in_index and head_item == index_item:
            git_status = GitStatus.CLEAN
        else:
            git_status = _status_from_xy(
                xy,
                rename_from=rename_from,
                in_head=in_head,
                in_index=in_index,
                in_worktree=in_worktree or (not dirty_enabled and in_index),
            )
            # Refine clean when porcelain is quiet but index diverged from HEAD.
            if path not in status and in_head and in_index and head_item != index_item:
                git_status = GitStatus.STAGED
            if path not in status and in_head and not in_index:
                git_status = GitStatus.DELETED
            if path not in status and not in_head and in_index:
                git_status = GitStatus.STAGED

        kind, reason_code, policy_rule = classify_coverage_kind(
            path,
            policy=policy,
            entry_kind=entry_kind,
            skip_prefixes=effective_skip_prefixes,
        )

        content_digest = ""
        dependency_identity_id = ""

        if entry_kind is EntryKind.GITLINK:
            commit_id = git_object_id or (
                head_item.object_id if head_item else ""
            )
            gitlink = GitlinkRecord(
                path=path,
                commit_id=commit_id,
                mode=git_mode or "160000",
                head_object_id=head_item.object_id if head_item else "",
                index_object_id=index_item.object_id if index_item else "",
            )
            gitlinks.append(gitlink)
            content_digest = f"gitlink:{commit_id}"
            dependency_identity_id = remember_dependency(
                DependencyIdentity(
                    kind=DependencyIdentityKind.GITLINK,
                    path=path,
                    digest=content_digest,
                    git_object_id=commit_id,
                    reason_code="gitlink_submodule",
                )
            )
        else:
            if wt is not None:
                content_digest = wt.digest
                git_mode = wt.mode or git_mode
            elif git_object_id and entry_kind is not EntryKind.GITLINK:
                # Bind exact blob bytes through Git without embedding them.
                content_digest = object_digest(git_object_id)
            elif head_item is not None:
                content_digest = object_digest(head_item.object_id)
                git_object_id = head_item.object_id
            if kind is CoverageKind.DEPENDENCY_TOOL_IDENTITY:
                dep_kind = DependencyIdentityKind.LOCKFILE
                if reason_code == "dependency_manifest":
                    dep_kind = DependencyIdentityKind.MANIFEST
                elif reason_code == "dependency_directory":
                    dep_kind = DependencyIdentityKind.DIRECTORY_MARKER
                dependency_identity_id = remember_dependency(
                    DependencyIdentity(
                        kind=dep_kind,
                        path=path,
                        digest=content_digest or f"git:{git_object_id}",
                        git_object_id=git_object_id,
                        reason_code=reason_code,
                    )
                )

        dispositions.append(
            CoverageDisposition(
                path=path,
                kind=kind,
                git_status=git_status,
                entry_kind=entry_kind,
                reason_code=reason_code,
                policy_rule=policy_rule,
                content_digest=content_digest,
                git_mode=git_mode,
                git_object_id=git_object_id,
                rename_from=rename_from,
                tracked=True,
                overlay=git_status is not GitStatus.CLEAN,
                dependency_identity_id=dependency_identity_id,
            )
        )

    # Allowlisted untracked overlays (dirty analysis only).
    if dirty_enabled:
        for path, meta in sorted(status.items()):
            if path in head or path in index:
                continue
            if meta.get("xy") != "??":
                # Untracked is "??"; other codes for untracked-like paths are rare.
                if path in {item.path for item in dispositions}:
                    continue
            if not policy.untracked_allowed(path):
                # Explicit non-admission: do not silently ignore; record as
                # excluded overlay only when under a skip rule, otherwise omit
                # from authority while remaining outside the tracked ledger.
                skip_prefix = _path_has_prefix(path, effective_skip_prefixes)
                skip_dir = _path_has_directory_name(
                    path, sorted(policy.skip_directory_names)
                )
                if skip_prefix is None and skip_dir is None:
                    continue
                kind = CoverageKind.EXCLUDED
                reason_code = (
                    "excluded_prefix" if skip_prefix else "excluded_directory"
                )
                policy_rule = (
                    f"skip_prefixes:{skip_prefix}"
                    if skip_prefix
                    else f"skip_directory_names:{skip_dir}"
                )
                dispositions.append(
                    CoverageDisposition(
                        path=path,
                        kind=kind,
                        git_status=GitStatus.UNTRACKED,
                        entry_kind=EntryKind.REGULAR,
                        reason_code=reason_code,
                        policy_rule=policy_rule,
                        tracked=False,
                        overlay=True,
                    )
                )
                continue
            wt = worktree_entry(path)
            if wt is None:
                continue
            kind, reason_code, policy_rule = classify_coverage_kind(
                path,
                policy=policy,
                entry_kind=wt.kind,
                skip_prefixes=effective_skip_prefixes,
            )
            dispositions.append(
                CoverageDisposition(
                    path=path,
                    kind=kind,
                    git_status=GitStatus.UNTRACKED,
                    entry_kind=wt.kind,
                    reason_code=reason_code,
                    policy_rule=policy_rule,
                    content_digest=wt.digest,
                    git_mode=wt.mode,
                    tracked=False,
                    overlay=True,
                )
            )

    # Exactly one disposition per path (including overlays).
    by_path: dict[str, CoverageDisposition] = {}
    for item in dispositions:
        if item.path in by_path:
            raise CoverageIncompleteError(
                f"duplicate disposition for path {item.path}"
            )
        by_path[item.path] = item
    ordered = tuple(by_path[path] for path in sorted(by_path))

    tracked = [item for item in ordered if item.tracked]
    if len(tracked) != len(tracked_paths):
        missing = sorted(set(tracked_paths) - {item.path for item in tracked})
        raise CoverageIncompleteError(
            f"tracked paths missing dispositions: {missing[:10]}"
        )
    if not policy.silent_exclusions_allowed:
        # Every tracked path must have an explicit kind and reason.
        for item in tracked:
            if not item.reason_code or not item.policy_rule:
                raise CoverageIncompleteError(
                    f"silent exclusion forbidden at {item.path}"
                )

    stats = RepositorySnapshotStats(
        tracked_path_count=len(tracked),
        disposition_count=len(ordered),
        overlay_path_count=sum(1 for item in ordered if item.overlay),
        excluded_path_count=sum(
            1 for item in ordered if item.kind is CoverageKind.EXCLUDED
        ),
        dependency_identity_count=len(dependency_identities),
        gitlink_count=len(gitlinks),
        dirty_path_count=sum(
            1 for item in ordered if item.git_status is not GitStatus.CLEAN
        ),
        deleted_path_count=sum(
            1
            for item in ordered
            if item.git_status
            in {GitStatus.DELETED, GitStatus.STAGED_DELETION}
        ),
        untracked_path_count=sum(
            1 for item in ordered if item.git_status is GitStatus.UNTRACKED
        ),
        semantic_path_count=sum(
            1 for item in ordered if item.kind is CoverageKind.SEMANTIC_AST
        ),
        unsupported_path_count=sum(
            1 for item in ordered if item.kind is CoverageKind.UNSUPPORTED
        ),
        hashed_bytes=hashed_bytes,
    )

    snapshot = RepositorySnapshot(
        primary_root=primary_label,
        head_commit_id=head_commit,
        head_tree_id=head_tree,
        index_tree_id=index_tree,
        scope_policy_id=policy.policy_id,
        scope_id=policy.scope_id,
        dispositions=ordered,
        dependency_identities=tuple(dependency_identities),
        gitlinks=tuple(gitlinks),
        stats=stats,
        repository_root=str(primary),
        git_directory=git_directory,
        allow_dirty_analysis=dirty_enabled,
    )
    snapshot.assert_exhaustive_tracked_coverage()
    return snapshot


def snapshot_analyzer_health_inventory(
    snapshot: RepositorySnapshot,
) -> dict[str, Any]:
    """Project a snapshot into the analyzer-health inventory shape."""

    return snapshot.coverage_inventory()


__all__ = [
    "COVERAGE_DISPOSITION_SCHEMA",
    "COVERAGE_DISPOSITION_SCHEMA_VERSION",
    "CoverageDisposition",
    "CoverageIncompleteError",
    "CoverageKind",
    "DEFAULT_SCOPE_CONFIG_RELATIVE",
    "DEPENDENCY_IDENTITY_SCHEMA",
    "DependencyIdentity",
    "DependencyIdentityKind",
    "EntryKind",
    "GITLINK_RECORD_SCHEMA",
    "GitStatus",
    "GitlinkRecord",
    "REPOSITORY_SNAPSHOT_SCHEMA",
    "REPOSITORY_SNAPSHOT_SCHEMA_VERSION",
    "RepositoryPathEscapeError",
    "RepositorySnapshot",
    "RepositorySnapshotError",
    "RepositorySnapshotStats",
    "RepositoryStateError",
    "SCOPE_POLICY_SCHEMA",
    "ScopePolicy",
    "ScopePolicyError",
    "SymlinkEscapeError",
    "build_repository_snapshot",
    "classify_coverage_kind",
    "default_scope_policy_path",
    "load_scope_policy",
    "primary_relative_prefixes",
    "repo_path",
    "scope_policy_from_mapping",
    "snapshot_analyzer_health_inventory",
]
