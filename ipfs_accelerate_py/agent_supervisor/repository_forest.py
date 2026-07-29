"""Independently bound repository descriptors and authority forests.

A repository forest is the unit of observation authority for multi-checkout
assurance work.  Each descriptor is derived from its own Git root; sibling
roots never share Git authority merely by co-location under a parent path.

Portable identity binds commit, tree, recursive gitlink closure, dirty overlay
digest, ignore policy, case/Unicode policy, and read/write authority.  Host
locators and credentials stay outside portable CIDs so equivalent relocations
replay with the same forest identity.

Root and path resolution is fail-closed: missing roots, escaping paths, and
symlink escapes reject rather than silently broaden scope.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .checkout_lock import checkout_repository_id
from .formal_verification_contracts import content_identity
from .task_identity import canonical_content_cid


REPOSITORY_FOREST_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.repository-forest@1"
)
REPOSITORY_DESCRIPTOR_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.repository-descriptor@1"
)
REPOSITORY_ID_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.repository-id@1"
)
PORTABLE_CLOSURE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.portable-git-closure@1"
)
LOCAL_LOCATOR_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.local-locator@1"
)
DIRTY_OVERLAY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.dirty-overlay@1"
)
IGNORE_POLICY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.ignore-policy@1"
)
CASE_UNICODE_POLICY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.case-unicode-policy@1"
)
AUTHORITY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.repository-authority@1"
)
FOREST_POLICY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.forest-policy@1"
)
GITLINK_ENTRY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.gitlink-closure-entry@1"
)
ANALYZER_PROFILE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.analyzer-profile@1"
)

DEFAULT_SWISSKNIFE_ROOT = "/home/barberb/swissknife"
DEFAULT_SWISSKNIFE_ALIAS = "swissknife"
DEFAULT_ACCELERATOR_ALIAS = "ipfs_accelerate_py"
DEFAULT_KIT_ALIAS = "ipfs_kit_py"
DEFAULT_DATASETS_ALIAS = "ipfs_datasets_py"

_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_ALIAS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}\Z")
_EMPTY_OVERLAY_DIGEST = content_identity(
    {
        "schema": DIRTY_OVERLAY_SCHEMA,
        "entries": (),
    }
)
_MAX_GITLINK_DEPTH = 16
_MAX_DIRTY_PATHS = 4096
_MAX_DIRTY_BYTES_PER_FILE = 1_048_576
_GIT_TIMEOUT_SECONDS = 30


class RepositoryForestError(ValueError):
    """Fail-closed rejection for repository forest construction or path use."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = str(reason_code or "repository_forest_error").strip()
        detail = str(message or "").strip()
        super().__init__(detail or self.reason_code)


class AuthorityMode(str, Enum):
    """Read/write authority granted to one repository descriptor."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class UnicodeNormalizationForm(str, Enum):
    """Allowed Unicode normalization forms for path comparison policy."""

    NONE = "none"
    NFC = "NFC"
    NFD = "NFD"
    NFKC = "NFKC"
    NFKD = "NFKD"


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise RepositoryForestError(
            "invalid_field_type",
            f"{field_name} must be a string",
        )
    if required and not text:
        raise RepositoryForestError(
            "missing_required_field",
            f"{field_name} is required",
        )
    return text


def _git_object(value: Any, *, field_name: str) -> str:
    text = _text(value, field_name=field_name).lower()
    if not _GIT_OBJECT_RE.fullmatch(text):
        raise RepositoryForestError(
            "invalid_git_object",
            f"{field_name} must be a full Git object identity",
        )
    return text


def _normalize_alias(value: Any) -> str:
    text = _text(value, field_name="alias")
    if not _ALIAS_RE.fullmatch(text):
        raise RepositoryForestError(
            "invalid_alias",
            "alias must be a short alphanumeric identifier",
        )
    return text


def _normalize_posix_relative(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        raise RepositoryForestError(
            "invalid_relative_path",
            f"{field_name} must be a non-empty relative path",
        )
    if text.startswith("/") or re.match(r"^[A-Za-z]:/", text):
        raise RepositoryForestError(
            "absolute_path_rejected",
            f"{field_name} must be relative",
        )
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise RepositoryForestError(
            "path_escape",
            f"{field_name} escapes its repository root",
        )
    return "/".join(parts)


def _normalize_remote_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Drop credentials from URL-shaped remotes without logging them.
    if "://" in text:
        scheme, remainder = text.split("://", 1)
        if "@" in remainder:
            remainder = remainder.rsplit("@", 1)[-1]
        text = f"{scheme}://{remainder}"
    elif text.startswith("git@") and ":" in text:
        # git@host:path -> host:path (no credentials beyond host key form)
        text = text[4:]
    return text.rstrip("/")


def _sorted_unique_strings(values: Iterable[Any], *, field_name: str) -> tuple[str, ...]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = _text(raw, field_name=field_name, required=True)
        if text not in seen:
            seen.add(text)
            items.append(text)
    return tuple(sorted(items))


def _git(
    repo_root: Path,
    *arguments: str,
    binary: bool = False,
) -> tuple[int, str | bytes]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            text=not binary,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 1, b"" if binary else ""
    output: str | bytes = completed.stdout
    if not binary:
        output = str(output).strip()
    return completed.returncode, output


def _status_porcelain(repo_root: Path) -> tuple[bool, list[tuple[str, str]]]:
    returncode, output = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        binary=True,
    )
    if returncode != 0 or not isinstance(output, bytes):
        return False, []
    entries: list[tuple[str, str]] = []
    chunks = output.split(b"\0")
    index = 0
    while index < len(chunks):
        raw = chunks[index]
        index += 1
        if not raw:
            continue
        if len(raw) < 3:
            continue
        code = raw[:2].decode("ascii", errors="replace")
        path = raw[3:].decode("utf-8", errors="surrogateescape")
        # Rename records carry the destination then the source as the next
        # null-terminated field.
        if code[0] == "R" or code[1] == "R" or code[0] == "C" or code[1] == "C":
            if index < len(chunks):
                source = chunks[index].decode("utf-8", errors="surrogateescape")
                index += 1
                path = f"{source}->{path}"
        entries.append((code, path.replace("\\", "/")))
    entries.sort(key=lambda item: (item[1], item[0]))
    return True, entries


def _gitlinks_at_commit(
    repo_root: Path,
    commit: str,
) -> tuple[bool, list[tuple[str, str]]]:
    returncode, output = _git(
        repo_root,
        "ls-tree",
        "-r",
        "-z",
        commit,
        binary=True,
    )
    if returncode != 0 or not isinstance(output, bytes):
        return False, []
    rows: list[tuple[str, str]] = []
    for raw in output.split(b"\0"):
        if not raw or b"\t" not in raw:
            continue
        metadata, raw_path = raw.split(b"\t", 1)
        parts = metadata.split()
        if len(parts) != 3 or parts[0] != b"160000":
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        commit_id = parts[2].decode("ascii", errors="replace")
        rows.append((relative.replace("\\", "/"), commit_id))
    return True, sorted(rows)


def _remote_origin_url(repo_root: Path) -> str:
    status, output = _git(repo_root, "config", "--get", "remote.origin.url")
    if status != 0:
        return ""
    return _normalize_remote_url(output)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            remaining = _MAX_DIRTY_BYTES_PER_FILE
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
            leftover = handle.read(1)
            if leftover:
                digest.update(b"|truncated")
    except OSError:
        return "unreadable"
    return digest.hexdigest()


@dataclass(frozen=True)
class IgnorePolicy:
    """Which paths participate in inventory and dirty-overlay digests."""

    schema: str = IGNORE_POLICY_SCHEMA
    include_gitignored: bool = False
    allow_dirty_overlay: bool = True
    exclude_patterns: tuple[str, ...] = ()
    include_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="ignore_policy.schema"),
        )
        if self.schema != IGNORE_POLICY_SCHEMA:
            raise RepositoryForestError("unsupported_ignore_policy_schema")
        if not isinstance(self.include_gitignored, bool):
            raise RepositoryForestError("invalid_ignore_policy")
        if not isinstance(self.allow_dirty_overlay, bool):
            raise RepositoryForestError("invalid_ignore_policy")
        excludes = _sorted_unique_strings(
            self.exclude_patterns,
            field_name="exclude_patterns",
        ) if self.exclude_patterns else ()
        includes = _sorted_unique_strings(
            self.include_patterns,
            field_name="include_patterns",
        ) if self.include_patterns else ()
        object.__setattr__(self, "exclude_patterns", excludes)
        object.__setattr__(self, "include_patterns", includes)

    @property
    def policy_cid(self) -> str:
        return content_identity(self.to_portable_dict())

    def to_portable_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "include_gitignored": self.include_gitignored,
            "allow_dirty_overlay": self.allow_dirty_overlay,
            "exclude_patterns": list(self.exclude_patterns),
            "include_patterns": list(self.include_patterns),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IgnorePolicy":
        return cls(
            schema=str(payload.get("schema") or IGNORE_POLICY_SCHEMA),
            include_gitignored=bool(payload.get("include_gitignored", False)),
            allow_dirty_overlay=bool(payload.get("allow_dirty_overlay", True)),
            exclude_patterns=tuple(payload.get("exclude_patterns") or ()),
            include_patterns=tuple(payload.get("include_patterns") or ()),
        )


@dataclass(frozen=True)
class CaseUnicodePolicy:
    """Filesystem case and Unicode comparison policy for path identity."""

    schema: str = CASE_UNICODE_POLICY_SCHEMA
    case_sensitive: bool = True
    unicode_normalization: str = UnicodeNormalizationForm.NFC.value
    reject_encoding_collisions: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="case_unicode_policy.schema"),
        )
        if self.schema != CASE_UNICODE_POLICY_SCHEMA:
            raise RepositoryForestError("unsupported_case_unicode_policy_schema")
        if not isinstance(self.case_sensitive, bool):
            raise RepositoryForestError("invalid_case_unicode_policy")
        if not isinstance(self.reject_encoding_collisions, bool):
            raise RepositoryForestError("invalid_case_unicode_policy")
        form = _text(
            self.unicode_normalization,
            field_name="unicode_normalization",
        )
        allowed = {item.value for item in UnicodeNormalizationForm}
        if form not in allowed:
            raise RepositoryForestError("unsupported_unicode_normalization")
        object.__setattr__(self, "unicode_normalization", form)

    @property
    def policy_cid(self) -> str:
        return content_identity(self.to_portable_dict())

    def normalize_path_text(self, value: str) -> str:
        text = str(value or "").replace("\\", "/")
        form = self.unicode_normalization
        if form != UnicodeNormalizationForm.NONE.value:
            text = unicodedata.normalize(form, text)
        if not self.case_sensitive:
            text = text.casefold()
        return text

    def to_portable_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "case_sensitive": self.case_sensitive,
            "unicode_normalization": self.unicode_normalization,
            "reject_encoding_collisions": self.reject_encoding_collisions,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaseUnicodePolicy":
        return cls(
            schema=str(payload.get("schema") or CASE_UNICODE_POLICY_SCHEMA),
            case_sensitive=bool(payload.get("case_sensitive", True)),
            unicode_normalization=str(
                payload.get("unicode_normalization")
                or UnicodeNormalizationForm.NFC.value
            ),
            reject_encoding_collisions=bool(
                payload.get("reject_encoding_collisions", True)
            ),
        )


@dataclass(frozen=True)
class RepositoryAuthority:
    """Read/write authority bound into a descriptor's identity."""

    schema: str = AUTHORITY_SCHEMA
    mode: str = AuthorityMode.READ_ONLY.value
    write_path_allowlist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="authority.schema"),
        )
        if self.schema != AUTHORITY_SCHEMA:
            raise RepositoryForestError("unsupported_authority_schema")
        mode = _text(self.mode, field_name="authority.mode")
        if mode not in {item.value for item in AuthorityMode}:
            raise RepositoryForestError("unsupported_authority_mode")
        object.__setattr__(self, "mode", mode)
        allowlist: list[str] = []
        seen: set[str] = set()
        for raw in self.write_path_allowlist or ():
            path = _normalize_posix_relative(raw, field_name="write_path_allowlist")
            if path not in seen:
                seen.add(path)
                allowlist.append(path)
        object.__setattr__(self, "write_path_allowlist", tuple(sorted(allowlist)))
        if mode == AuthorityMode.READ_ONLY.value and allowlist:
            raise RepositoryForestError(
                "read_only_write_allowlist",
                "read-only authority cannot carry a write path allowlist",
            )

    @property
    def authority_cid(self) -> str:
        return content_identity(self.to_portable_dict())

    @property
    def is_writable(self) -> bool:
        return self.mode == AuthorityMode.READ_WRITE.value

    def to_portable_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "write_path_allowlist": list(self.write_path_allowlist),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepositoryAuthority":
        return cls(
            schema=str(payload.get("schema") or AUTHORITY_SCHEMA),
            mode=str(payload.get("mode") or AuthorityMode.READ_ONLY.value),
            write_path_allowlist=tuple(payload.get("write_path_allowlist") or ()),
        )


@dataclass(frozen=True)
class LocalLocator:
    """Host-local checkout location; never folded into portable forest CIDs."""

    schema: str = LOCAL_LOCATOR_SCHEMA
    alias: str = ""
    root_path: str = ""
    resolved_root_path: str = ""
    local_repository_binding_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="local_locator.schema"),
        )
        if self.schema != LOCAL_LOCATOR_SCHEMA:
            raise RepositoryForestError("unsupported_local_locator_schema")
        object.__setattr__(self, "alias", _normalize_alias(self.alias))
        root = _text(self.root_path, field_name="root_path")
        resolved = _text(self.resolved_root_path, field_name="resolved_root_path")
        binding = _text(
            self.local_repository_binding_id,
            field_name="local_repository_binding_id",
            required=False,
        )
        object.__setattr__(self, "root_path", root)
        object.__setattr__(self, "resolved_root_path", resolved)
        object.__setattr__(self, "local_repository_binding_id", binding)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "alias": self.alias,
            "root_path": self.root_path,
            "resolved_root_path": self.resolved_root_path,
            "local_repository_binding_id": self.local_repository_binding_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LocalLocator":
        return cls(
            schema=str(payload.get("schema") or LOCAL_LOCATOR_SCHEMA),
            alias=str(payload.get("alias") or ""),
            root_path=str(payload.get("root_path") or ""),
            resolved_root_path=str(payload.get("resolved_root_path") or ""),
            local_repository_binding_id=str(
                payload.get("local_repository_binding_id") or ""
            ),
        )


@dataclass(frozen=True)
class GitlinkClosureEntry:
    """Opaque recursive gitlink identity without a persisted checkout path."""

    schema: str = GITLINK_ENTRY_SCHEMA
    gitlink_id: str = ""
    commit: str = ""
    tree: str = ""
    parent_gitlink_id: str = ""
    depth: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="gitlink.schema"),
        )
        if self.schema != GITLINK_ENTRY_SCHEMA:
            raise RepositoryForestError("unsupported_gitlink_schema")
        object.__setattr__(
            self,
            "gitlink_id",
            _text(self.gitlink_id, field_name="gitlink_id"),
        )
        object.__setattr__(
            self,
            "commit",
            _git_object(self.commit, field_name="gitlink.commit"),
        )
        object.__setattr__(
            self,
            "tree",
            _git_object(self.tree, field_name="gitlink.tree"),
        )
        parent = str(self.parent_gitlink_id or "").strip()
        depth = int(self.depth)
        if depth < 0:
            raise RepositoryForestError("invalid_gitlink_depth")
        if depth == 0 and parent:
            raise RepositoryForestError("top_level_gitlink_parent")
        if depth > 0 and not parent:
            raise RepositoryForestError("nested_gitlink_missing_parent")
        object.__setattr__(self, "parent_gitlink_id", parent)
        object.__setattr__(self, "depth", depth)

    def to_portable_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "gitlink_id": self.gitlink_id,
            "commit": self.commit,
            "tree": self.tree,
            "parent_gitlink_id": self.parent_gitlink_id,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GitlinkClosureEntry":
        return cls(
            schema=str(payload.get("schema") or GITLINK_ENTRY_SCHEMA),
            gitlink_id=str(payload.get("gitlink_id") or ""),
            commit=str(payload.get("commit") or ""),
            tree=str(payload.get("tree") or ""),
            parent_gitlink_id=str(payload.get("parent_gitlink_id") or ""),
            depth=int(payload.get("depth") or 0),
        )


@dataclass(frozen=True)
class PortableGitClosure:
    """Portable tree/commit/gitlink closure for one repository."""

    schema: str = PORTABLE_CLOSURE_SCHEMA
    commit: str = ""
    tree: str = ""
    gitlinks: tuple[GitlinkClosureEntry, ...] = ()
    gitlink_closure_complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="portable_closure.schema"),
        )
        if self.schema != PORTABLE_CLOSURE_SCHEMA:
            raise RepositoryForestError("unsupported_portable_closure_schema")
        object.__setattr__(
            self,
            "commit",
            _git_object(self.commit, field_name="commit"),
        )
        object.__setattr__(
            self,
            "tree",
            _git_object(self.tree, field_name="tree"),
        )
        entries = tuple(
            item
            if isinstance(item, GitlinkClosureEntry)
            else GitlinkClosureEntry.from_dict(item)
            for item in self.gitlinks
        )
        entries = tuple(sorted(entries, key=lambda item: item.gitlink_id))
        ids = [item.gitlink_id for item in entries]
        if len(ids) != len(set(ids)):
            raise RepositoryForestError("duplicate_gitlink_identity")
        if not isinstance(self.gitlink_closure_complete, bool):
            raise RepositoryForestError("invalid_gitlink_closure_flag")
        object.__setattr__(self, "gitlinks", entries)

    @property
    def gitlink_closure_cid(self) -> str:
        return content_identity(
            {
                "schema": PORTABLE_CLOSURE_SCHEMA + "/gitlinks",
                "complete": self.gitlink_closure_complete,
                "gitlinks": [item.to_portable_dict() for item in self.gitlinks],
            }
        )

    @property
    def closure_cid(self) -> str:
        return content_identity(self.to_portable_dict())

    def to_portable_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "commit": self.commit,
            "tree": self.tree,
            "gitlink_closure_complete": self.gitlink_closure_complete,
            "gitlink_closure_cid": self.gitlink_closure_cid,
            "gitlinks": [item.to_portable_dict() for item in self.gitlinks],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortableGitClosure":
        raw_gitlinks = payload.get("gitlinks") or ()
        if not isinstance(raw_gitlinks, Sequence) or isinstance(
            raw_gitlinks, (str, bytes, bytearray)
        ):
            raise RepositoryForestError("invalid_gitlinks")
        return cls(
            schema=str(payload.get("schema") or PORTABLE_CLOSURE_SCHEMA),
            commit=str(payload.get("commit") or ""),
            tree=str(payload.get("tree") or ""),
            gitlinks=tuple(raw_gitlinks),
            gitlink_closure_complete=bool(
                payload.get("gitlink_closure_complete", True)
            ),
        )


@dataclass(frozen=True)
class RepositoryIdentity:
    """Canonical repository ID independent of local checkout path."""

    schema: str = REPOSITORY_ID_SCHEMA
    repository_id: str = ""
    logical_name: str = ""
    remote_url: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="repository_identity.schema"),
        )
        if self.schema != REPOSITORY_ID_SCHEMA:
            raise RepositoryForestError("unsupported_repository_id_schema")
        logical = _normalize_alias(self.logical_name)
        remote = _normalize_remote_url(self.remote_url)
        object.__setattr__(self, "logical_name", logical)
        object.__setattr__(self, "remote_url", remote)
        expected = make_repository_id(logical_name=logical, remote_url=remote)
        provided = str(self.repository_id or "").strip()
        if provided and provided != expected:
            raise RepositoryForestError(
                "repository_id_mismatch",
                "repository_id does not match logical name and remote",
            )
        object.__setattr__(self, "repository_id", expected)

    def to_portable_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "repository_id": self.repository_id,
            "logical_name": self.logical_name,
            "remote_url": self.remote_url,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepositoryIdentity":
        return cls(
            schema=str(payload.get("schema") or REPOSITORY_ID_SCHEMA),
            repository_id=str(payload.get("repository_id") or ""),
            logical_name=str(payload.get("logical_name") or ""),
            remote_url=str(payload.get("remote_url") or ""),
        )


def make_repository_id(*, logical_name: str, remote_url: str = "") -> str:
    """Return a portable repository ID for a logical checkout identity."""

    name = _normalize_alias(logical_name)
    remote = _normalize_remote_url(remote_url)
    return "repository:" + content_identity(
        {
            "schema": REPOSITORY_ID_SCHEMA,
            "logical_name": name,
            "remote_url": remote,
        }
    )


@dataclass(frozen=True)
class RepositoryDescriptor:
    """One independently bound repository observation."""

    schema: str = REPOSITORY_DESCRIPTOR_SCHEMA
    identity: RepositoryIdentity = None  # type: ignore[assignment]
    portable_closure: PortableGitClosure = None  # type: ignore[assignment]
    local_locator: LocalLocator = None  # type: ignore[assignment]
    dirty: bool = False
    dirty_overlay_digest: str = _EMPTY_OVERLAY_DIGEST
    ignore_policy: IgnorePolicy = None  # type: ignore[assignment]
    case_unicode_policy: CaseUnicodePolicy = None  # type: ignore[assignment]
    authority: RepositoryAuthority = None  # type: ignore[assignment]
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="descriptor.schema"),
        )
        if self.schema != REPOSITORY_DESCRIPTOR_SCHEMA:
            raise RepositoryForestError("unsupported_descriptor_schema")
        identity = self.identity
        if not isinstance(identity, RepositoryIdentity):
            if isinstance(identity, Mapping):
                identity = RepositoryIdentity.from_dict(identity)
            else:
                raise RepositoryForestError("missing_repository_identity")
        closure = self.portable_closure
        if not isinstance(closure, PortableGitClosure):
            if isinstance(closure, Mapping):
                closure = PortableGitClosure.from_dict(closure)
            else:
                raise RepositoryForestError("missing_portable_closure")
        locator = self.local_locator
        if not isinstance(locator, LocalLocator):
            if isinstance(locator, Mapping):
                locator = LocalLocator.from_dict(locator)
            else:
                raise RepositoryForestError("missing_local_locator")
        ignore_policy = self.ignore_policy
        if not isinstance(ignore_policy, IgnorePolicy):
            if isinstance(ignore_policy, Mapping):
                ignore_policy = IgnorePolicy.from_dict(ignore_policy)
            else:
                ignore_policy = IgnorePolicy()
        case_policy = self.case_unicode_policy
        if not isinstance(case_policy, CaseUnicodePolicy):
            if isinstance(case_policy, Mapping):
                case_policy = CaseUnicodePolicy.from_dict(case_policy)
            else:
                case_policy = CaseUnicodePolicy()
        authority = self.authority
        if not isinstance(authority, RepositoryAuthority):
            if isinstance(authority, Mapping):
                authority = RepositoryAuthority.from_dict(authority)
            else:
                authority = RepositoryAuthority()
        if not isinstance(self.dirty, bool):
            raise RepositoryForestError("invalid_dirty_flag")
        overlay = _text(
            self.dirty_overlay_digest,
            field_name="dirty_overlay_digest",
        )
        if not self.dirty and overlay != _EMPTY_OVERLAY_DIGEST:
            raise RepositoryForestError(
                "clean_descriptor_dirty_overlay",
                "clean descriptors must use the empty overlay digest",
            )
        reasons = tuple(
            dict.fromkeys(
                _text(item, field_name="reason_codes")
                for item in (self.reason_codes or ())
            )
        )
        if locator.alias != identity.logical_name:
            raise RepositoryForestError(
                "alias_identity_mismatch",
                "local locator alias must match repository logical name",
            )
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "portable_closure", closure)
        object.__setattr__(self, "local_locator", locator)
        object.__setattr__(self, "ignore_policy", ignore_policy)
        object.__setattr__(self, "case_unicode_policy", case_policy)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "dirty_overlay_digest", overlay)
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def alias(self) -> str:
        return self.identity.logical_name

    @property
    def repository_id(self) -> str:
        return self.identity.repository_id

    @property
    def commit(self) -> str:
        return self.portable_closure.commit

    @property
    def tree(self) -> str:
        return self.portable_closure.tree

    @property
    def root_path(self) -> Path:
        return Path(self.local_locator.resolved_root_path)

    @property
    def descriptor_cid(self) -> str:
        return content_identity(self.to_portable_dict())

    def to_portable_dict(self) -> dict[str, Any]:
        """Portable projection: host locators intentionally omitted."""

        return {
            "schema": self.schema,
            "repository_id": self.repository_id,
            "logical_name": self.identity.logical_name,
            "remote_url": self.identity.remote_url,
            "commit": self.commit,
            "tree": self.tree,
            "gitlink_closure_cid": self.portable_closure.gitlink_closure_cid,
            "gitlink_closure_complete": (
                self.portable_closure.gitlink_closure_complete
            ),
            "gitlinks": [
                item.to_portable_dict() for item in self.portable_closure.gitlinks
            ],
            "dirty": self.dirty,
            "dirty_overlay_digest": self.dirty_overlay_digest,
            "ignore_policy_cid": self.ignore_policy.policy_cid,
            "ignore_policy": self.ignore_policy.to_portable_dict(),
            "case_unicode_policy_cid": self.case_unicode_policy.policy_cid,
            "case_unicode_policy": self.case_unicode_policy.to_portable_dict(),
            "authority_cid": self.authority.authority_cid,
            "authority": self.authority.to_portable_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_portable_dict()
        payload["local_locator"] = self.local_locator.to_dict()
        payload["reason_codes"] = list(self.reason_codes)
        payload["descriptor_cid"] = self.descriptor_cid
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepositoryDescriptor":
        identity = RepositoryIdentity(
            repository_id=str(payload.get("repository_id") or ""),
            logical_name=str(
                payload.get("logical_name")
                or (payload.get("identity") or {}).get("logical_name")
                or ""
            ),
            remote_url=str(
                payload.get("remote_url")
                or (payload.get("identity") or {}).get("remote_url")
                or ""
            ),
        )
        if "portable_closure" in payload:
            closure = PortableGitClosure.from_dict(payload["portable_closure"])
        else:
            closure = PortableGitClosure(
                commit=str(payload.get("commit") or ""),
                tree=str(payload.get("tree") or ""),
                gitlinks=tuple(payload.get("gitlinks") or ()),
                gitlink_closure_complete=bool(
                    payload.get("gitlink_closure_complete", True)
                ),
            )
        local_raw = payload.get("local_locator") or {}
        if not isinstance(local_raw, Mapping):
            raise RepositoryForestError("invalid_local_locator")
        return cls(
            schema=str(payload.get("schema") or REPOSITORY_DESCRIPTOR_SCHEMA),
            identity=identity,
            portable_closure=closure,
            local_locator=LocalLocator.from_dict(local_raw),
            dirty=bool(payload.get("dirty", False)),
            dirty_overlay_digest=str(
                payload.get("dirty_overlay_digest") or _EMPTY_OVERLAY_DIGEST
            ),
            ignore_policy=IgnorePolicy.from_dict(
                payload.get("ignore_policy") or {}
            ),
            case_unicode_policy=CaseUnicodePolicy.from_dict(
                payload.get("case_unicode_policy") or {}
            ),
            authority=RepositoryAuthority.from_dict(
                payload.get("authority") or {}
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class AnalyzerProfile:
    """Analyzer/parser/toolchain profile bound into forest portable identity.

    Version pins and configuration digests are portable; host paths, API keys,
    and environment secrets must never appear here.
    """

    schema: str = ANALYZER_PROFILE_SCHEMA
    profile_name: str = "default"
    analyzer_versions: tuple[tuple[str, str], ...] = ()
    parser_versions: tuple[tuple[str, str], ...] = ()
    toolchain_versions: tuple[tuple[str, str], ...] = ()
    configuration_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="analyzer_profile.schema"),
        )
        if self.schema != ANALYZER_PROFILE_SCHEMA:
            raise RepositoryForestError("unsupported_analyzer_profile_schema")
        name = _text(self.profile_name, field_name="profile_name")
        if not _ALIAS_RE.fullmatch(name):
            raise RepositoryForestError(
                "invalid_analyzer_profile_name",
                "profile_name must be a short alphanumeric identifier",
            )
        object.__setattr__(self, "profile_name", name)
        object.__setattr__(
            self,
            "analyzer_versions",
            _normalize_version_pairs(
                self.analyzer_versions,
                field_name="analyzer_versions",
            ),
        )
        object.__setattr__(
            self,
            "parser_versions",
            _normalize_version_pairs(
                self.parser_versions,
                field_name="parser_versions",
            ),
        )
        object.__setattr__(
            self,
            "toolchain_versions",
            _normalize_version_pairs(
                self.toolchain_versions,
                field_name="toolchain_versions",
            ),
        )
        digest = str(self.configuration_digest or "").strip()
        object.__setattr__(self, "configuration_digest", digest)

    @property
    def profile_cid(self) -> str:
        return content_identity(self.to_portable_dict())

    def to_portable_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile_name": self.profile_name,
            "analyzer_versions": [
                {"name": name, "version": version}
                for name, version in self.analyzer_versions
            ],
            "parser_versions": [
                {"name": name, "version": version}
                for name, version in self.parser_versions
            ],
            "toolchain_versions": [
                {"name": name, "version": version}
                for name, version in self.toolchain_versions
            ],
            "configuration_digest": self.configuration_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalyzerProfile":
        return cls(
            schema=str(payload.get("schema") or ANALYZER_PROFILE_SCHEMA),
            profile_name=str(payload.get("profile_name") or "default"),
            analyzer_versions=_pairs_from_payload(
                payload.get("analyzer_versions")
            ),
            parser_versions=_pairs_from_payload(payload.get("parser_versions")),
            toolchain_versions=_pairs_from_payload(
                payload.get("toolchain_versions")
            ),
            configuration_digest=str(
                payload.get("configuration_digest") or ""
            ),
        )


def _normalize_version_pairs(
    values: Iterable[Any],
    *,
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in values or ():
        if isinstance(raw, Mapping):
            name = _text(raw.get("name"), field_name=f"{field_name}.name")
            version = _text(
                raw.get("version"),
                field_name=f"{field_name}.version",
            )
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            name = _text(raw[0], field_name=f"{field_name}.name")
            version = _text(raw[1], field_name=f"{field_name}.version")
        else:
            raise RepositoryForestError(
                "invalid_version_pair",
                f"{field_name} entries must be name/version pairs",
            )
        if name in seen:
            raise RepositoryForestError(
                "duplicate_version_name",
                f"duplicate version name in {field_name}: {name}",
            )
        # Reject credential-shaped version strings without logging them.
        lowered = f"{name}={version}".lower()
        if any(
            marker in lowered
            for marker in (
                "password=",
                "secret=",
                "api_key=",
                "token=",
                "authorization=",
            )
        ):
            raise RepositoryForestError(
                "secret_material_rejected",
                f"{field_name} must not carry credential-like material",
            )
        seen.add(name)
        pairs.append((name, version))
    return tuple(sorted(pairs, key=lambda item: item[0]))


def _pairs_from_payload(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple((str(key), str(val)) for key, val in value.items())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(value)  # type: ignore[return-value]
    raise RepositoryForestError("invalid_version_pairs")


@dataclass(frozen=True)
class ForestRootSpec:
    """Caller-supplied root binding used to build one descriptor."""

    alias: str
    root_path: str | Path
    authority: RepositoryAuthority | Mapping[str, Any] | None = None
    ignore_policy: IgnorePolicy | Mapping[str, Any] | None = None
    case_unicode_policy: CaseUnicodePolicy | Mapping[str, Any] | None = None
    logical_name: str = ""
    remote_url: str = ""
    required: bool = True


@dataclass(frozen=True)
class ForestPolicy:
    """Authority and root policy for an entire forest."""

    schema: str = FOREST_POLICY_SCHEMA
    roots: tuple[ForestRootSpec, ...] = ()
    sole_write_alias: str = DEFAULT_ACCELERATOR_ALIAS
    analyzer_profile: AnalyzerProfile | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="forest_policy.schema"),
        )
        if self.schema != FOREST_POLICY_SCHEMA:
            raise RepositoryForestError("unsupported_forest_policy_schema")
        write_alias = _normalize_alias(self.sole_write_alias)
        object.__setattr__(self, "sole_write_alias", write_alias)
        profile = self.analyzer_profile
        if profile is None:
            profile = AnalyzerProfile()
        elif isinstance(profile, Mapping):
            profile = AnalyzerProfile.from_dict(profile)
        elif not isinstance(profile, AnalyzerProfile):
            raise RepositoryForestError("invalid_analyzer_profile")
        object.__setattr__(self, "analyzer_profile", profile)
        normalized: list[ForestRootSpec] = []
        aliases: set[str] = set()
        write_count = 0
        for raw in self.roots:
            if not isinstance(raw, ForestRootSpec):
                raise RepositoryForestError("invalid_forest_root_spec")
            alias = _normalize_alias(raw.alias)
            if alias in aliases:
                raise RepositoryForestError(
                    "duplicate_alias",
                    f"duplicate forest alias: {alias}",
                )
            aliases.add(alias)
            authority = raw.authority
            if authority is None:
                if alias == write_alias:
                    authority = RepositoryAuthority(
                        mode=AuthorityMode.READ_WRITE.value
                    )
                else:
                    authority = RepositoryAuthority(
                        mode=AuthorityMode.READ_ONLY.value
                    )
            elif isinstance(authority, Mapping):
                authority = RepositoryAuthority.from_dict(authority)
            elif not isinstance(authority, RepositoryAuthority):
                raise RepositoryForestError("invalid_authority")
            if authority.is_writable:
                write_count += 1
                if alias != write_alias:
                    raise RepositoryForestError(
                        "unexpected_write_root",
                        "only the sole write alias may be writable",
                    )
            elif alias == write_alias:
                raise RepositoryForestError(
                    "write_root_not_writable",
                    "sole write alias must carry read/write authority",
                )
            ignore_policy = raw.ignore_policy
            if ignore_policy is None:
                ignore_policy = IgnorePolicy()
            elif isinstance(ignore_policy, Mapping):
                ignore_policy = IgnorePolicy.from_dict(ignore_policy)
            case_policy = raw.case_unicode_policy
            if case_policy is None:
                case_policy = CaseUnicodePolicy()
            elif isinstance(case_policy, Mapping):
                case_policy = CaseUnicodePolicy.from_dict(case_policy)
            logical = _normalize_alias(raw.logical_name or alias)
            normalized.append(
                ForestRootSpec(
                    alias=alias,
                    root_path=raw.root_path,
                    authority=authority,
                    ignore_policy=ignore_policy,
                    case_unicode_policy=case_policy,
                    logical_name=logical,
                    remote_url=_normalize_remote_url(raw.remote_url),
                    required=bool(raw.required),
                )
            )
        if write_alias not in aliases:
            raise RepositoryForestError(
                "missing_write_root",
                "forest policy must include the sole write alias",
            )
        if write_count != 1:
            raise RepositoryForestError(
                "write_root_cardinality",
                "forest policy must grant write authority to exactly one root",
            )
        object.__setattr__(
            self,
            "roots",
            tuple(sorted(normalized, key=lambda item: item.alias)),
        )

    @property
    def policy_cid(self) -> str:
        return content_identity(self.to_portable_dict())

    def to_portable_dict(self) -> dict[str, Any]:
        profile = self.analyzer_profile
        if not isinstance(profile, AnalyzerProfile):
            profile = AnalyzerProfile.from_dict(profile or {})
        return {
            "schema": self.schema,
            "sole_write_alias": self.sole_write_alias,
            "analyzer_profile": profile.to_portable_dict(),
            "analyzer_profile_cid": profile.profile_cid,
            "roots": [
                {
                    "alias": root.alias,
                    "logical_name": root.logical_name,
                    "remote_url": root.remote_url,
                    "required": root.required,
                    "authority": (
                        root.authority.to_portable_dict()
                        if isinstance(root.authority, RepositoryAuthority)
                        else root.authority
                    ),
                    "ignore_policy": (
                        root.ignore_policy.to_portable_dict()
                        if isinstance(root.ignore_policy, IgnorePolicy)
                        else root.ignore_policy
                    ),
                    "case_unicode_policy": (
                        root.case_unicode_policy.to_portable_dict()
                        if isinstance(root.case_unicode_policy, CaseUnicodePolicy)
                        else root.case_unicode_policy
                    ),
                    # Host paths intentionally omitted from portable policy.
                }
                for root in self.roots
            ],
        }


@dataclass(frozen=True)
class RepositoryForest:
    """Independently bound multi-repository authority forest."""

    schema: str = REPOSITORY_FOREST_SCHEMA
    descriptors: tuple[RepositoryDescriptor, ...] = ()
    sole_write_alias: str = DEFAULT_ACCELERATOR_ALIAS
    policy_cid: str = ""
    analyzer_profile: AnalyzerProfile | Mapping[str, Any] | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="forest.schema"),
        )
        if self.schema != REPOSITORY_FOREST_SCHEMA:
            raise RepositoryForestError("unsupported_forest_schema")
        write_alias = _normalize_alias(self.sole_write_alias)
        object.__setattr__(self, "sole_write_alias", write_alias)
        profile = self.analyzer_profile
        if profile is None:
            profile = AnalyzerProfile()
        elif isinstance(profile, Mapping):
            profile = AnalyzerProfile.from_dict(profile)
        elif not isinstance(profile, AnalyzerProfile):
            raise RepositoryForestError("invalid_analyzer_profile")
        object.__setattr__(self, "analyzer_profile", profile)
        descriptors = tuple(
            item
            if isinstance(item, RepositoryDescriptor)
            else RepositoryDescriptor.from_dict(item)
            for item in self.descriptors
        )
        descriptors = tuple(sorted(descriptors, key=lambda item: item.alias))
        aliases = [item.alias for item in descriptors]
        if len(aliases) != len(set(aliases)):
            raise RepositoryForestError("duplicate_alias")
        repository_ids = [item.repository_id for item in descriptors]
        if len(repository_ids) != len(set(repository_ids)):
            raise RepositoryForestError("duplicate_repository_id")
        # Never treat co-located checkouts as one Git authority domain.
        binding_ids = [
            item.local_locator.local_repository_binding_id
            for item in descriptors
            if item.local_locator.local_repository_binding_id
        ]
        if len(binding_ids) != len(set(binding_ids)):
            raise RepositoryForestError(
                "shared_git_authority_rejected",
                "sibling descriptors must not share a Git common-directory binding",
            )
        writable = [item for item in descriptors if item.authority.is_writable]
        if len(writable) != 1:
            raise RepositoryForestError("write_root_cardinality")
        if writable[0].alias != write_alias:
            raise RepositoryForestError("unexpected_write_root")
        reasons = tuple(
            dict.fromkeys(
                _text(item, field_name="reason_codes")
                for item in (self.reason_codes or ())
            )
        )
        policy_cid = str(self.policy_cid or "").strip()
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "policy_cid", policy_cid)

    @property
    def forest_id(self) -> str:
        """Deterministic portable forest identity."""

        profile = self.analyzer_profile
        if not isinstance(profile, AnalyzerProfile):
            profile = AnalyzerProfile.from_dict(profile or {})
        return content_identity(
            {
                "schema": REPOSITORY_FOREST_SCHEMA + "/identity",
                "sole_write_alias": self.sole_write_alias,
                "policy_cid": self.policy_cid,
                "analyzer_profile_cid": profile.profile_cid,
                "descriptors": [
                    item.to_portable_dict() for item in self.descriptors
                ],
            }
        )

    def descriptor_for_alias(self, alias: str) -> RepositoryDescriptor:
        key = _normalize_alias(alias)
        for item in self.descriptors:
            if item.alias == key:
                return item
        raise RepositoryForestError(
            "unknown_alias",
            f"no descriptor for alias {key}",
        )

    def write_descriptor(self) -> RepositoryDescriptor:
        return self.descriptor_for_alias(self.sole_write_alias)

    def to_portable_dict(self) -> dict[str, Any]:
        profile = self.analyzer_profile
        if not isinstance(profile, AnalyzerProfile):
            profile = AnalyzerProfile.from_dict(profile or {})
        return {
            "schema": self.schema,
            "forest_id": self.forest_id,
            "sole_write_alias": self.sole_write_alias,
            "policy_cid": self.policy_cid,
            "analyzer_profile": profile.to_portable_dict(),
            "analyzer_profile_cid": profile.profile_cid,
            "descriptors": [item.to_portable_dict() for item in self.descriptors],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_portable_dict()
        payload["descriptors"] = [item.to_dict() for item in self.descriptors]
        payload["reason_codes"] = list(self.reason_codes)
        return payload

    @classmethod
    def from_portable_dict(cls, payload: Mapping[str, Any]) -> "RepositoryForest":
        """Replay a portable forest projection (local locators absent)."""

        raw_descriptors = payload.get("descriptors") or ()
        if not isinstance(raw_descriptors, Sequence) or isinstance(
            raw_descriptors, (str, bytes, bytearray)
        ):
            raise RepositoryForestError("invalid_portable_forest")
        rebuilt: list[RepositoryDescriptor] = []
        for raw in raw_descriptors:
            if not isinstance(raw, Mapping):
                raise RepositoryForestError("invalid_portable_descriptor")
            logical = str(raw.get("logical_name") or "")
            # Portable replay substitutes a sentinel locator; callers that need
            # host paths must re-bind through build_repository_forest.
            locator = LocalLocator(
                alias=logical,
                root_path=f"portable://{logical}",
                resolved_root_path=f"portable://{logical}",
                local_repository_binding_id=(
                    "portable-binding:"
                    + content_identity(
                        {
                            "schema": LOCAL_LOCATOR_SCHEMA + "/portable-binding",
                            "logical_name": logical,
                            "repository_id": str(raw.get("repository_id") or ""),
                        }
                    )
                ),
            )
            material = dict(raw)
            material["local_locator"] = locator.to_dict()
            rebuilt.append(RepositoryDescriptor.from_dict(material))
        forest = cls(
            schema=str(payload.get("schema") or REPOSITORY_FOREST_SCHEMA),
            descriptors=tuple(rebuilt),
            sole_write_alias=str(
                payload.get("sole_write_alias") or DEFAULT_ACCELERATOR_ALIAS
            ),
            policy_cid=str(payload.get("policy_cid") or ""),
            analyzer_profile=AnalyzerProfile.from_dict(
                payload.get("analyzer_profile") or {}
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        claimed = str(payload.get("forest_id") or "").strip()
        if claimed and claimed != forest.forest_id:
            raise RepositoryForestError(
                "forest_id_mismatch",
                "portable forest_id does not match recomputed identity",
            )
        return forest


def resolve_repository_root(
    root_path: str | Path,
    *,
    follow_symlinks: bool = True,
) -> Path:
    """Resolve a repository root fail-closed.

    Broken symlinks, missing paths, non-directories, and resolution errors
    raise :class:`RepositoryForestError` instead of returning a partial path.
    """

    raw = Path(root_path)
    if not str(raw):
        raise RepositoryForestError("missing_root", "root path is empty")
    try:
        if not raw.exists():
            raise RepositoryForestError("missing_root", f"root does not exist: {raw}")
    except OSError as exc:
        raise RepositoryForestError(
            "root_unresolvable",
            "root path could not be inspected",
        ) from exc
    try:
        if raw.is_symlink() and not follow_symlinks:
            raise RepositoryForestError(
                "symlink_root_rejected",
                "symlink repository roots are rejected by policy",
            )
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RepositoryForestError(
            "root_unresolvable",
            "failed to resolve repository root",
        ) from exc
    if not resolved.is_dir():
        raise RepositoryForestError(
            "root_not_directory",
            "repository root must be a directory",
        )
    return resolved


def path_within_repository(
    descriptor: RepositoryDescriptor,
    candidate: str | Path,
    *,
    require_existing: bool = False,
) -> Path:
    """Resolve ``candidate`` under ``descriptor`` or fail closed on escape."""

    root = descriptor.root_path
    raw = Path(candidate)
    try:
        if raw.is_absolute():
            target = raw
        else:
            relative = _normalize_posix_relative(
                raw.as_posix(),
                field_name="candidate",
            )
            target = root.joinpath(*PurePosixPath(relative).parts)
        # Resolve with strictness matching existence requirements, but always
        # re-check containment after symlink resolution.
        if require_existing:
            resolved = target.resolve(strict=True)
        else:
            # Resolve existing parents; reject if any symlink leaves the root.
            resolved = _resolve_under_root(root, target)
        resolved.relative_to(root)
    except RepositoryForestError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise RepositoryForestError(
            "path_escape",
            "path escapes repository descriptor root",
        ) from exc
    normalized = descriptor.case_unicode_policy.normalize_path_text(
        resolved.relative_to(root).as_posix()
    )
    # Recompute containment using policy-normalized relative form.
    if normalized.startswith("../") or normalized == "..":
        raise RepositoryForestError("path_escape")
    return resolved


def _resolve_under_root(root: Path, target: Path) -> Path:
    """Resolve ``target`` while forbidding symlink escapes outside ``root``."""

    try:
        root_resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RepositoryForestError("root_unresolvable") from exc

    current = root_resolved
    try:
        relative = target if target.is_absolute() else target
        if target.is_absolute():
            parts = target.parts
            # Absolute targets must still land under root after resolution.
            candidate = Path(*parts) if parts else Path("/")
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(root_resolved)
            return resolved
        parts = PurePosixPath(
            _normalize_posix_relative(target.as_posix(), field_name="candidate")
        ).parts
    except RepositoryForestError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise RepositoryForestError("path_escape") from exc

    for part in parts:
        next_path = current / part
        try:
            if next_path.is_symlink():
                linked = next_path.resolve(strict=False)
                linked.relative_to(root_resolved)
                current = linked
            elif next_path.exists():
                current = next_path.resolve(strict=False)
                current.relative_to(root_resolved)
            else:
                # Non-existent leaf: keep lexical join under resolved parents.
                current = next_path
                current.relative_to(root_resolved)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RepositoryForestError(
                "path_escape",
                "path escapes repository descriptor root",
            ) from exc
    return current


def compute_dirty_overlay_digest(
    repo_root: Path,
    *,
    ignore_policy: IgnorePolicy,
    status_entries: Sequence[tuple[str, str]] | None = None,
) -> tuple[bool, str, tuple[str, ...]]:
    """Return ``(dirty, digest, reason_codes)`` for the working tree overlay."""

    reasons: list[str] = []
    if status_entries is None:
        ok, entries = _status_porcelain(repo_root)
        if not ok:
            return True, _EMPTY_OVERLAY_DIGEST, ("status_unavailable",)
    else:
        entries = list(status_entries)
    if not entries:
        return False, _EMPTY_OVERLAY_DIGEST, ()
    if not ignore_policy.allow_dirty_overlay:
        return True, _EMPTY_OVERLAY_DIGEST, ("dirty_overlay_forbidden",)
    if len(entries) > _MAX_DIRTY_PATHS:
        reasons.append("dirty_overlay_truncated")
        entries = entries[:_MAX_DIRTY_PATHS]
    overlay_entries: list[dict[str, str]] = []
    for code, rel_path in entries:
        path_key = rel_path.split("->", 1)[-1]
        try:
            relative = _normalize_posix_relative(
                path_key,
                field_name="dirty_path",
            )
        except RepositoryForestError:
            reasons.append("dirty_path_escape")
            continue
        absolute = repo_root / relative
        if absolute.is_file():
            digest = _file_digest(absolute)
            kind = "file"
        elif absolute.is_dir():
            digest = "directory"
            kind = "directory"
        elif absolute.is_symlink():
            try:
                digest = hashlib.sha256(
                    os.readlink(absolute).encode("utf-8", errors="surrogateescape")
                ).hexdigest()
            except OSError:
                digest = "unreadable-symlink"
            kind = "symlink"
        else:
            digest = "missing"
            kind = "missing"
        overlay_entries.append(
            {
                "status": code,
                "path": relative,
                "kind": kind,
                "digest": digest,
            }
        )
    overlay_entries.sort(key=lambda item: (item["path"], item["status"]))
    digest = content_identity(
        {
            "schema": DIRTY_OVERLAY_SCHEMA,
            "entries": overlay_entries,
        }
    )
    return True, digest, tuple(dict.fromkeys(reasons))


def inspect_gitlink_closure(
    repo_root: Path,
    commit: str,
    *,
    max_depth: int = _MAX_GITLINK_DEPTH,
) -> tuple[tuple[GitlinkClosureEntry, ...], bool, tuple[str, ...]]:
    """Walk recursive gitlinks without persisting checkout paths."""

    reasons: list[str] = []
    entries: list[GitlinkClosureEntry] = []
    complete = True
    visited: set[tuple[str, str]] = set()
    root = repo_root.resolve()

    def walk(
        checkout: Path,
        parent_commit: str,
        *,
        parent_gitlink_id: str = "",
        depth: int = 0,
    ) -> None:
        nonlocal complete
        if depth > max(0, int(max_depth)):
            complete = False
            reasons.append("recursive_gitlink_depth_exceeded")
            return
        try:
            checkout_key = str(checkout.resolve())
        except (OSError, RuntimeError):
            complete = False
            reasons.append("gitlink_checkout_unresolvable")
            return
        repository_key = (checkout_key, parent_commit)
        if repository_key in visited:
            complete = False
            reasons.append("recursive_gitlink_cycle")
            return
        visited.add(repository_key)
        listed, gitlinks = _gitlinks_at_commit(checkout, parent_commit)
        if not listed:
            complete = False
            reasons.append("recursive_gitlink_map_unavailable")
            return
        for relative, recorded_commit in gitlinks:
            link_id = canonical_content_cid(
                {
                    "schema": GITLINK_ENTRY_SCHEMA + "/location",
                    "parent_commit": parent_commit,
                    "location": relative,
                }
            )
            candidate = checkout / relative
            try:
                resolved_candidate = candidate.resolve()
                resolved_candidate.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                complete = False
                reasons.append("gitlink_checkout_outside_repository")
                continue
            head_status, head_output = _git(candidate, "rev-parse", "HEAD")
            tree_status, tree_output = _git(candidate, "rev-parse", "HEAD^{tree}")
            if head_status != 0 or tree_status != 0:
                complete = False
                reasons.append("gitlink_checkout_unavailable")
                continue
            try:
                child_head = _git_object(head_output, field_name="gitlink head")
                child_tree = _git_object(tree_output, field_name="gitlink tree")
                recorded = _git_object(
                    recorded_commit,
                    field_name="gitlink recorded commit",
                )
            except RepositoryForestError:
                complete = False
                reasons.append("gitlink_identity_invalid")
                continue
            if child_head != recorded:
                reasons.append("gitlink_head_mismatch")
            entries.append(
                GitlinkClosureEntry(
                    gitlink_id=link_id,
                    commit=recorded,
                    tree=child_tree,
                    parent_gitlink_id=parent_gitlink_id,
                    depth=depth,
                )
            )
            walk(
                candidate,
                child_head,
                parent_gitlink_id=link_id,
                depth=depth + 1,
            )

    walk(root, commit)
    ordered = tuple(sorted(entries, key=lambda item: item.gitlink_id))
    return ordered, complete, tuple(dict.fromkeys(reasons))


def build_repository_descriptor(
    root_path: str | Path,
    *,
    alias: str,
    authority: RepositoryAuthority | Mapping[str, Any] | None = None,
    ignore_policy: IgnorePolicy | Mapping[str, Any] | None = None,
    case_unicode_policy: CaseUnicodePolicy | Mapping[str, Any] | None = None,
    logical_name: str = "",
    remote_url: str = "",
    follow_symlinks: bool = True,
    max_gitlink_depth: int = _MAX_GITLINK_DEPTH,
) -> RepositoryDescriptor:
    """Derive a fresh repository descriptor from a live checkout."""

    alias_key = _normalize_alias(alias)
    logical = _normalize_alias(logical_name or alias_key)
    if isinstance(authority, Mapping):
        authority_obj = RepositoryAuthority.from_dict(authority)
    elif authority is None:
        authority_obj = RepositoryAuthority()
    else:
        authority_obj = authority
    if isinstance(ignore_policy, Mapping):
        ignore_obj = IgnorePolicy.from_dict(ignore_policy)
    elif ignore_policy is None:
        ignore_obj = IgnorePolicy()
    else:
        ignore_obj = ignore_policy
    if isinstance(case_unicode_policy, Mapping):
        case_obj = CaseUnicodePolicy.from_dict(case_unicode_policy)
    elif case_unicode_policy is None:
        case_obj = CaseUnicodePolicy()
    else:
        case_obj = case_unicode_policy

    reasons: list[str] = []
    resolved = resolve_repository_root(
        root_path,
        follow_symlinks=follow_symlinks,
    )
    top_status, top_output = _git(resolved, "rev-parse", "--show-toplevel")
    if top_status != 0 or not str(top_output):
        raise RepositoryForestError(
            "not_a_git_repository",
            "descriptor root is not a Git repository",
        )
    try:
        top = Path(str(top_output)).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RepositoryForestError("root_unresolvable") from exc
    # Bind the Git toplevel only.  A nested path must never inherit parent Git
    # authority merely because it sits under another checkout.
    if top != resolved:
        raise RepositoryForestError(
            "nested_path_not_repository_root",
            "descriptor root must be a Git toplevel, not a nested path",
        )

    head_status, head_output = _git(top, "rev-parse", "HEAD")
    tree_status, tree_output = _git(top, "rev-parse", "HEAD^{tree}")
    if head_status != 0 or tree_status != 0:
        raise RepositoryForestError(
            "uncommitted_repository",
            "descriptor requires a committed Git HEAD",
        )
    commit = _git_object(head_output, field_name="commit")
    tree = _git_object(tree_output, field_name="tree")

    observed_remote = remote_url or _remote_origin_url(top)
    identity = RepositoryIdentity(
        logical_name=logical,
        remote_url=observed_remote,
    )
    gitlinks, complete, gitlink_reasons = inspect_gitlink_closure(
        top,
        commit,
        max_depth=max_gitlink_depth,
    )
    reasons.extend(gitlink_reasons)
    dirty, overlay_digest, overlay_reasons = compute_dirty_overlay_digest(
        top,
        ignore_policy=ignore_obj,
    )
    reasons.extend(overlay_reasons)
    try:
        local_binding = checkout_repository_id(top)
    except Exception:
        local_binding = "repository:" + content_identity(
            {
                "schema": LOCAL_LOCATOR_SCHEMA + "/fallback-binding",
                "path": str(top),
            }
        )
        reasons.append("local_binding_fallback")

    # Local locator alias tracks the forest alias; identity.logical_name is the
    # portable logical name.  They must match for the descriptor contract.
    if alias_key != logical:
        raise RepositoryForestError(
            "alias_logical_name_mismatch",
            "alias and logical_name must match for a descriptor binding",
        )
    locator = LocalLocator(
        alias=logical,
        root_path=str(Path(root_path)),
        resolved_root_path=str(top),
        local_repository_binding_id=local_binding,
    )

    return RepositoryDescriptor(
        identity=identity,
        portable_closure=PortableGitClosure(
            commit=commit,
            tree=tree,
            gitlinks=gitlinks,
            gitlink_closure_complete=complete,
        ),
        local_locator=locator,
        dirty=dirty,
        dirty_overlay_digest=overlay_digest,
        ignore_policy=ignore_obj,
        case_unicode_policy=case_obj,
        authority=authority_obj,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def initial_vfs_assurance_forest_policy(
    *,
    accelerator_root: str | Path,
    swissknife_root: str | Path = DEFAULT_SWISSKNIFE_ROOT,
    kit_root: str | Path | None = None,
    datasets_root: str | Path | None = None,
    include_optional_missing: bool = False,
    analyzer_profile: AnalyzerProfile | Mapping[str, Any] | None = None,
) -> ForestPolicy:
    """Initial policy: SwissKnife read-only; accelerator sole write root.

    Optional kit/datasets roots default to the accelerator submodule paths and
    remain read-only.  They may be omitted when absent unless
    ``include_optional_missing`` is true.
    """

    accelerator = Path(accelerator_root)
    roots: list[ForestRootSpec] = [
        ForestRootSpec(
            alias=DEFAULT_SWISSKNIFE_ALIAS,
            root_path=swissknife_root,
            authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
            required=True,
        ),
        ForestRootSpec(
            alias=DEFAULT_ACCELERATOR_ALIAS,
            root_path=accelerator,
            authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
            required=True,
        ),
    ]
    kit_path = Path(kit_root) if kit_root is not None else accelerator / "ipfs_kit_py"
    datasets_path = (
        Path(datasets_root)
        if datasets_root is not None
        else accelerator / "ipfs_datasets_py"
    )
    for alias, path in (
        (DEFAULT_KIT_ALIAS, kit_path),
        (DEFAULT_DATASETS_ALIAS, datasets_path),
    ):
        required = bool(include_optional_missing)
        if path.exists() or required:
            roots.append(
                ForestRootSpec(
                    alias=alias,
                    root_path=path,
                    authority=RepositoryAuthority(
                        mode=AuthorityMode.READ_ONLY.value
                    ),
                    required=required,
                )
            )
    return ForestPolicy(
        roots=tuple(roots),
        sole_write_alias=DEFAULT_ACCELERATOR_ALIAS,
        analyzer_profile=analyzer_profile,
    )


def build_repository_forest(
    policy: ForestPolicy | Sequence[ForestRootSpec],
    *,
    sole_write_alias: str = DEFAULT_ACCELERATOR_ALIAS,
    follow_symlinks: bool = True,
    max_gitlink_depth: int = _MAX_GITLINK_DEPTH,
    fail_on_missing_required: bool = True,
) -> RepositoryForest:
    """Build a forest of independently bound descriptors from policy roots."""

    if isinstance(policy, ForestPolicy):
        forest_policy = policy
    else:
        forest_policy = ForestPolicy(
            roots=tuple(policy),
            sole_write_alias=sole_write_alias,
        )

    descriptors: list[RepositoryDescriptor] = []
    reasons: list[str] = []
    for root in forest_policy.roots:
        try:
            descriptor = build_repository_descriptor(
                root.root_path,
                alias=root.alias,
                authority=root.authority
                if isinstance(root.authority, RepositoryAuthority)
                else None,
                ignore_policy=root.ignore_policy
                if isinstance(root.ignore_policy, IgnorePolicy)
                else None,
                case_unicode_policy=root.case_unicode_policy
                if isinstance(root.case_unicode_policy, CaseUnicodePolicy)
                else None,
                logical_name=root.logical_name,
                remote_url=root.remote_url,
                follow_symlinks=follow_symlinks,
                max_gitlink_depth=max_gitlink_depth,
            )
        except RepositoryForestError as exc:
            code = exc.reason_code
            reasons.append(f"{root.alias}:{code}")
            if root.required and fail_on_missing_required:
                raise RepositoryForestError(
                    code,
                    f"required root {root.alias!r} failed: {code}",
                ) from exc
            continue
        descriptors.append(descriptor)

    if not descriptors:
        raise RepositoryForestError(
            "empty_forest",
            "no repository descriptors could be built",
        )

    profile = forest_policy.analyzer_profile
    if not isinstance(profile, AnalyzerProfile):
        profile = AnalyzerProfile.from_dict(profile or {})
    return RepositoryForest(
        descriptors=tuple(descriptors),
        sole_write_alias=forest_policy.sole_write_alias,
        policy_cid=forest_policy.policy_cid,
        analyzer_profile=profile,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def forests_share_portable_identity(
    left: RepositoryForest,
    right: RepositoryForest,
) -> bool:
    """Return True when two forests share portable identity (ignoring hosts)."""

    return left.forest_id == right.forest_id


def empty_dirty_overlay_digest() -> str:
    """Return the canonical digest for a clean working tree overlay."""

    return _EMPTY_OVERLAY_DIGEST


__all__ = [
    "ANALYZER_PROFILE_SCHEMA",
    "AUTHORITY_SCHEMA",
    "AnalyzerProfile",
    "AuthorityMode",
    "CASE_UNICODE_POLICY_SCHEMA",
    "CaseUnicodePolicy",
    "DEFAULT_ACCELERATOR_ALIAS",
    "DEFAULT_DATASETS_ALIAS",
    "DEFAULT_KIT_ALIAS",
    "DEFAULT_SWISSKNIFE_ALIAS",
    "DEFAULT_SWISSKNIFE_ROOT",
    "DIRTY_OVERLAY_SCHEMA",
    "FOREST_POLICY_SCHEMA",
    "ForestPolicy",
    "ForestRootSpec",
    "GITLINK_ENTRY_SCHEMA",
    "GitlinkClosureEntry",
    "IGNORE_POLICY_SCHEMA",
    "IgnorePolicy",
    "LOCAL_LOCATOR_SCHEMA",
    "LocalLocator",
    "PORTABLE_CLOSURE_SCHEMA",
    "PortableGitClosure",
    "REPOSITORY_DESCRIPTOR_SCHEMA",
    "REPOSITORY_FOREST_SCHEMA",
    "REPOSITORY_ID_SCHEMA",
    "RepositoryAuthority",
    "RepositoryDescriptor",
    "RepositoryForest",
    "RepositoryForestError",
    "RepositoryIdentity",
    "UnicodeNormalizationForm",
    "build_repository_descriptor",
    "build_repository_forest",
    "compute_dirty_overlay_digest",
    "empty_dirty_overlay_digest",
    "forests_share_portable_identity",
    "initial_vfs_assurance_forest_policy",
    "inspect_gitlink_closure",
    "make_repository_id",
    "path_within_repository",
    "resolve_repository_root",
]
