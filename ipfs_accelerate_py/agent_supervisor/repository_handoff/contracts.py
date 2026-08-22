"""Transport-neutral repository handoff and overlay contract family (EAAEF-020).

These records are the shared serialization boundary for exact Git repository
transfer.  They are immutable, DAG-JSON compatible, content addressed, and
strictly versioned at major ``@1``.  Unknown schema names, unknown major
versions, floats, private material, hidden chain-of-thought, host paths, and
unsafe overlays are rejected.

A complete overlay accounts for HEAD, refs, the index, the worktree, untracked
paths, submodules, nested repositories, LFS pointers, sparse checkout, hooks,
attributes, modes, origin, and shallow/promisor bounds.  Public records carry
content-addressed references, not object bodies.  Hooks must be disabled on
import.  Symlink escape, enabled hooks, and unbounded objects are typed
refusals rather than reconstructed state.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, TypeAlias, TypeVar
from urllib.parse import urlsplit

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


REPOSITORY_HANDOFF_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = REPOSITORY_HANDOFF_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = REPOSITORY_HANDOFF_CONTRACT_VERSION

REPOSITORY_HANDOFF_REQUEST_INTERFACE: Final[str] = "RepositoryHandoffRequest@1"
REPOSITORY_OVERLAY_INTERFACE: Final[str] = "RepositoryOverlay@1"
SUBMODULE_RECORD_INTERFACE: Final[str] = "SubmoduleRecord@1"
NESTED_REPO_RECORD_INTERFACE: Final[str] = "NestedRepoRecord@1"
LFS_POINTER_RECORD_INTERFACE: Final[str] = "LfsPointerRecord@1"
SPARSE_CHECKOUT_RECORD_INTERFACE: Final[str] = "SparseCheckoutRecord@1"
HOOK_POLICY_INTERFACE: Final[str] = "HookPolicy@1"
ATTRIBUTE_AND_MODE_RECORD_INTERFACE: Final[str] = "AttributeAndModeRecord@1"
ORIGIN_AND_SHALLOW_BOUNDS_INTERFACE: Final[str] = "OriginAndShallowBounds@1"
REPOSITORY_HANDOFF_REFUSAL_INTERFACE: Final[str] = "RepositoryHandoffRefusal@1"
REPOSITORY_HANDOFF_BOUNDS_INTERFACE: Final[str] = "RepositoryHandoffBounds@1"

REPOSITORY_HANDOFF_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-handoff-request@1"
)
REPOSITORY_OVERLAY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-overlay@1"
)
SUBMODULE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/submodule-record@1"
)
NESTED_REPO_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/nested-repo-record@1"
)
LFS_POINTER_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/lfs-pointer-record@1"
)
SPARSE_CHECKOUT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/sparse-checkout-record@1"
)
HOOK_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/hook-policy@1"
)
ATTRIBUTE_AND_MODE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/attribute-and-mode-record@1"
)
ORIGIN_AND_SHALLOW_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/origin-and-shallow-bounds@1"
)
REPOSITORY_HANDOFF_REFUSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-handoff-refusal@1"
)
REPOSITORY_HANDOFF_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-handoff-bounds@1"
)

REPOSITORY_HANDOFF_CONTRACT_FAMILY: Final[Mapping[str, str]] = MappingProxyType(
    {
        "request": REPOSITORY_HANDOFF_REQUEST_INTERFACE,
        "overlay": REPOSITORY_OVERLAY_INTERFACE,
        "submodule": SUBMODULE_RECORD_INTERFACE,
        "nested_repo": NESTED_REPO_RECORD_INTERFACE,
        "lfs_pointer": LFS_POINTER_RECORD_INTERFACE,
        "sparse_checkout": SPARSE_CHECKOUT_RECORD_INTERFACE,
        "hook_policy": HOOK_POLICY_INTERFACE,
        "attribute_and_mode": ATTRIBUTE_AND_MODE_RECORD_INTERFACE,
        "origin_and_shallow": ORIGIN_AND_SHALLOW_BOUNDS_INTERFACE,
        "refusal": REPOSITORY_HANDOFF_REFUSAL_INTERFACE,
    }
)

ABSOLUTE_MAX_REFS: Final[int] = 4_096
ABSOLUTE_MAX_INDEX_ENTRIES: Final[int] = 65_536
ABSOLUTE_MAX_WORKTREE_ENTRIES: Final[int] = 65_536
ABSOLUTE_MAX_UNTRACKED: Final[int] = 16_384
ABSOLUTE_MAX_SUBMODULES: Final[int] = 1_024
ABSOLUTE_MAX_NESTED_REPOS: Final[int] = 256
ABSOLUTE_MAX_LFS_POINTERS: Final[int] = 16_384
ABSOLUTE_MAX_SPARSE_PATTERNS: Final[int] = 4_096
ABSOLUTE_MAX_HOOKS: Final[int] = 64
ABSOLUTE_MAX_ATTRIBUTES: Final[int] = 16_384
ABSOLUTE_MAX_OBJECTS: Final[int] = 1_000_000
ABSOLUTE_MAX_OBJECT_BYTES: Final[int] = 4 * 1024 * 1024 * 1024
ABSOLUTE_MAX_TEXT_BYTES: Final[int] = 65_536
ABSOLUTE_MAX_RECORD_BYTES: Final[int] = 1_048_576
ABSOLUTE_MAX_PATH_BYTES: Final[int] = 4_096
ABSOLUTE_MAX_ID_BYTES: Final[int] = 256
ABSOLUTE_MAX_REASON_BYTES: Final[int] = 256
ABSOLUTE_MAX_DEPTH: Final[int] = 16
ABSOLUTE_MAX_ITEMS: Final[int] = 4_096
ABSOLUTE_MAX_UNKNOWN_FIELDS: Final[int] = 32
ABSOLUTE_MAX_UNKNOWN_FIELD_BYTES: Final[int] = 8_192
ABSOLUTE_MAX_SHALLOW_DEPTH: Final[int] = 1_000_000

DEFAULT_MAX_REFS: Final[int] = 512
DEFAULT_MAX_INDEX_ENTRIES: Final[int] = 4_096
DEFAULT_MAX_WORKTREE_ENTRIES: Final[int] = 4_096
DEFAULT_MAX_UNTRACKED: Final[int] = 1_024
DEFAULT_MAX_SUBMODULES: Final[int] = 128
DEFAULT_MAX_NESTED_REPOS: Final[int] = 32
DEFAULT_MAX_LFS_POINTERS: Final[int] = 1_024
DEFAULT_MAX_SPARSE_PATTERNS: Final[int] = 256
DEFAULT_MAX_HOOKS: Final[int] = 32
DEFAULT_MAX_ATTRIBUTES: Final[int] = 4_096
DEFAULT_MAX_OBJECTS: Final[int] = 100_000
DEFAULT_MAX_OBJECT_BYTES: Final[int] = 512 * 1024 * 1024
DEFAULT_MAX_TEXT_BYTES: Final[int] = 16_384
DEFAULT_MAX_RECORD_BYTES: Final[int] = 65_536
DEFAULT_MAX_SERIALIZED_BYTES: Final[int] = 262_144
DEFAULT_MAX_PATH_BYTES: Final[int] = 1_024
DEFAULT_MAX_ID_BYTES: Final[int] = 128
DEFAULT_MAX_DEPTH: Final[int] = 8
DEFAULT_MAX_UNKNOWN_FIELDS: Final[int] = 16
DEFAULT_MAX_UNKNOWN_FIELD_BYTES: Final[int] = 2_048
DEFAULT_MAX_SHALLOW_DEPTH: Final[int] = 512

LFS_POINTER_VERSION: Final[str] = "https://git-lfs.github.com/spec/v1"

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_CIDV1_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{20,}$")
_HEX40_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_SCP_LIKE_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[A-Za-z0-9._-]+@)?[A-Za-z0-9.-]+:[\w./~+-]+$"
)
_RELATIVE_GIT_URL_RE: Final[re.Pattern[str]] = re.compile(r"^(?:\.\./)+[\w./~+-]+$")
_HOOK_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_ALLOWED_GIT_MODES: Final[frozenset[int]] = frozenset(
    {0o100644, 0o100755, 0o120000, 0o160000, 0o040000}
)
_GIT_MODE_SPELLINGS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "100644": 0o100644,
        "100755": 0o100755,
        "120000": 0o120000,
        "160000": 0o160000,
        "040000": 0o040000,
        "40000": 0o040000,
    }
)

_HIDDEN_CHAIN_OF_THOUGHT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "cot",
        "hidden_chain_of_thought",
        "hidden_cot",
        "hidden_reasoning",
        "hidden_thoughts",
        "internal_monologue",
        "model_thoughts",
        "private_reasoning",
        "private_thinking",
        "scratchpad",
        "thinking",
        "thinking_blocks",
        "thinking_private",
        "thinking_text",
    }
)
_PRIVATE_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "transcript_body",
        "witness",
    }
)
_BODY_FIELD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "blob_bytes",
        "body",
        "file_bytes",
        "object_bytes_payload",
        "raw_bytes",
        "raw_object",
        "worktree_bytes",
    }
)
_FORBIDDEN_ORIGIN_SCHEMES: Final[frozenset[str]] = frozenset(
    {"file", "ftp", "data", "javascript", "smb", "nfs"}
)
_ALLOWED_ORIGIN_SCHEMES: Final[frozenset[str]] = frozenset(
    {"https", "http", "git", "ssh"}
)

TEnum = TypeVar("TEnum", bound=Enum)


class RepositoryHandoffContractError(ContractValidationError):
    """Malformed or unsafe repository-handoff contract."""


class RepositoryHandoffBoundsError(RepositoryHandoffContractError):
    """A repository-handoff value exceeded a declared resource bound."""


class RepositoryHandoffIdentityError(RepositoryHandoffContractError):
    """A claimed content identity did not match its canonical payload."""


class RepositoryHandoffVersionError(RepositoryHandoffContractError):
    """Unsupported repository-handoff schema name or contract version."""


class RepositoryHandoffRefusalError(RepositoryHandoffContractError):
    """Typed refusal for an unsafe overlay or transfer descriptor."""

    def __init__(self, reason: "RefusalReason", message: str) -> None:
        super().__init__(message)
        self.reason = reason


class RefusalReason(str, Enum):
    """Closed vocabulary of unsafe-overlay refusals."""

    SYMLINK_ESCAPE = "symlink_escape"
    ENABLED_HOOKS = "enabled_hooks"
    UNBOUNDED_OBJECTS = "unbounded_objects"
    NESTED_GIT_ESCAPE = "nested_git_escape"
    HOST_PATH_ORIGIN = "host_path_origin"
    PRIVATE_MATERIAL = "private_material"


class FileKind(str, Enum):
    """Closed Git/worktree entry kinds."""

    REGULAR = "regular"
    EXECUTABLE = "executable"
    SYMLINK = "symlink"
    GITLINK = "gitlink"
    DIRECTORY = "directory"


class RepositoryHandoffMode(str, Enum):
    """Caller-requested transfer operation.  Not an authority grant."""

    PREVIEW = "preview"
    INSPECT = "inspect"
    RECONSTRUCT = "reconstruct"


class SubmoduleIgnore(str, Enum):
    """Closed submodule ignore vocabulary from ``.gitmodules``."""

    NONE = "none"
    UNTRACKED = "untracked"
    DIRTY = "dirty"
    ALL = "all"


class NestedGitDirKind(str, Enum):
    """How a nested repository's Git directory is represented."""

    GIT_DIR = "git_dir"
    GITFILE = "gitfile"


_FILE_KIND_BY_MODE: Final[Mapping[int, FileKind]] = MappingProxyType(
    {
        0o100644: FileKind.REGULAR,
        0o100755: FileKind.EXECUTABLE,
        0o120000: FileKind.SYMLINK,
        0o160000: FileKind.GITLINK,
        0o040000: FileKind.DIRECTORY,
    }
)


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _enum(value: Any, enum_type: type[TEnum], name: str) -> TEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise RepositoryHandoffContractError(f"{name} must be one of: {allowed}") from exc


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = ABSOLUTE_MAX_TEXT_BYTES,
) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise RepositoryHandoffContractError(f"{name} must be a string")
    else:
        result = value.strip()
    if required and not result:
        raise RepositoryHandoffContractError(f"{name} is required")
    if "\x00" in result:
        raise RepositoryHandoffContractError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > max_bytes:
        raise RepositoryHandoffBoundsError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RepositoryHandoffContractError(f"{name} must be a boolean")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RepositoryHandoffContractError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result < 1:
        raise RepositoryHandoffContractError(f"{name} must be at least 1")
    return result


def _major_version(name: str) -> int | None:
    if not isinstance(name, str) or "@" not in name:
        return None
    suffix = name.rsplit("@", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _require_versioned_name(name: str, expected: str, field_name: str) -> None:
    if name != expected:
        supplied_major = _major_version(name)
        expected_major = _major_version(expected) or REPOSITORY_HANDOFF_CONTRACT_VERSION
        if supplied_major is not None and supplied_major != expected_major:
            raise RepositoryHandoffVersionError(
                f"unsupported {field_name} {name!r}; rebuild with {expected}"
            )
        raise RepositoryHandoffVersionError(
            f"unsupported {field_name} {name!r}; expected {expected}"
        )


def _schema_and_version(
    payload: Mapping[str, Any],
    expected_schema: str,
    expected_interface: str,
    *,
    artifact_name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise RepositoryHandoffContractError(f"{artifact_name} payload must be an object")
    schema = payload.get("schema")
    if schema not in (None, "", expected_schema):
        _require_versioned_name(str(schema), expected_schema, "schema")
    interface = payload.get("interface")
    if interface not in (None, "", expected_interface):
        _require_versioned_name(str(interface), expected_interface, "interface")
    for key in ("contract_version", "schema_version"):
        version = payload.get(key)
        if version not in (None, "", REPOSITORY_HANDOFF_CONTRACT_VERSION):
            raise RepositoryHandoffVersionError(
                f"unsupported {artifact_name} contract version; rebuild with "
                f"{expected_interface}"
            )


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Iterable[str], *, artifact_name: str
) -> None:
    extra = set(payload).difference(allowed)
    if extra:
        raise RepositoryHandoffContractError(
            f"{artifact_name} contains unsupported fields; rebuild its canonical payload"
        )


def _claimed_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise RepositoryHandoffIdentityError(
                f"{artifact_name} content identity does not match payload"
            )


def _content_ref(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = ABSOLUTE_MAX_ID_BYTES,
) -> str:
    text = _text(value, name, required=required, max_bytes=max_bytes)
    if not text:
        return ""
    if _SHA256_RE.fullmatch(text) or _CIDV1_RE.fullmatch(text):
        return text
    raise RepositoryHandoffContractError(f"{name} must be a sha256 or CIDv1 identity")


def _git_oid(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, max_bytes=80)
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("sha1:"):
        lowered = lowered[5:]
    elif lowered.startswith("git:"):
        lowered = lowered[4:]
    if _HEX40_RE.fullmatch(lowered) or _HEX64_RE.fullmatch(lowered):
        return lowered
    raise RepositoryHandoffContractError(f"{name} must be a Git object id")


def _digest_sha256(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, max_bytes=80)
    if not text:
        return ""
    if _HEX64_RE.fullmatch(text):
        return f"sha256:{text}"
    if _SHA256_RE.fullmatch(text):
        return text
    raise RepositoryHandoffContractError(f"{name} must be a sha256 hex digest")


def _git_mode(value: Any, name: str) -> int:
    if isinstance(value, str):
        spelling = value.strip()
        if spelling in _GIT_MODE_SPELLINGS:
            return _GIT_MODE_SPELLINGS[spelling]
        raise RepositoryHandoffContractError(f"{name} must be a Git file mode")
    if isinstance(value, bool) or not isinstance(value, int):
        raise RepositoryHandoffContractError(f"{name} must be a Git file mode")
    if value not in _ALLOWED_GIT_MODES:
        raise RepositoryHandoffContractError(f"{name} must be a Git file mode")
    return value


def _file_kind_for_mode(mode: int, kind: Any, name: str) -> FileKind:
    expected = _FILE_KIND_BY_MODE[mode]
    if kind in (None, ""):
        return expected
    resolved = _enum(kind, FileKind, name)
    if resolved is not expected:
        raise RepositoryHandoffContractError(f"{name} does not match mode")
    return resolved


def _relative_path(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = ABSOLUTE_MAX_PATH_BYTES,
    allow_empty: bool = False,
) -> str:
    text = _text(value, name, required=required and not allow_empty, max_bytes=max_bytes)
    if not text:
        if required and not allow_empty:
            raise RepositoryHandoffContractError(f"{name} is required")
        return ""
    normalized = text.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise RepositoryHandoffContractError(f"{name} must be repository-relative")
    result = candidate.as_posix().removeprefix("./")
    if result in ("", "."):
        raise RepositoryHandoffContractError(f"{name} must not be empty")
    return result


def _git_dir_escape(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts) and parts[0] == ".git"


def _refuse(reason: RefusalReason, message: str) -> None:
    raise RepositoryHandoffRefusalError(reason, message)


def _reject_git_dir_path(path: str, name: str) -> None:
    if _git_dir_escape(path):
        _refuse(
            RefusalReason.NESTED_GIT_ESCAPE,
            f"{name} must not address the Git directory; nested git escape refused",
        )


def _symlink_target(value: Any, name: str, *, required: bool) -> str:
    text = _text(value, name, required=required, max_bytes=ABSOLUTE_MAX_PATH_BYTES)
    if not text:
        return ""
    normalized = text.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or (candidate.parts and candidate.parts[0].endswith(":"))
        or normalized.startswith("/")
    ):
        _refuse(
            RefusalReason.SYMLINK_ESCAPE,
            f"{name} is a symlink escape and cannot be reconstructed",
        )
    result = candidate.as_posix().removeprefix("./")
    if result in ("", "."):
        _refuse(
            RefusalReason.SYMLINK_ESCAPE,
            f"{name} is a symlink escape and cannot be reconstructed",
        )
    if _git_dir_escape(result):
        _refuse(
            RefusalReason.SYMLINK_ESCAPE,
            f"{name} is a symlink escape into the Git directory",
        )
    return result


def _git_ref_name(value: Any, name: str, *, allow_head: bool = False) -> str:
    text = _text(value, name, max_bytes=DEFAULT_MAX_PATH_BYTES)
    if allow_head and text == "HEAD":
        return text
    if text == "HEAD":
        raise RepositoryHandoffContractError(f"{name} must be a refs/ name")
    if (
        not text.startswith("refs/")
        or text.endswith("/")
        or "//" in text
        or ".." in text
        or "@{" in text
        or "\\" in text
        or " " in text
        or text.endswith(".lock")
        or any(part in ("", ".", "..") for part in text.split("/"))
    ):
        raise RepositoryHandoffContractError(f"{name} must be a valid Git ref")
    return text


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    max_items: int,
    max_bytes: int = ABSOLUTE_MAX_ID_BYTES,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Sequence
    ):
        raise RepositoryHandoffContractError(f"{name} must be a sequence of strings")
    else:
        items = values
    if len(items) > max_items:
        raise RepositoryHandoffBoundsError(f"{name} exceeds its item-count limit")
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _content_ref(item, name, max_bytes=max_bytes)
        if text in seen:
            raise RepositoryHandoffContractError(f"{name} must not contain duplicate identities")
        seen.add(text)
        result.append(text)
    if required and not result:
        raise RepositoryHandoffContractError(f"{name} must not be empty")
    return tuple(result)


def _key_is_forbidden(key: str) -> str | None:
    normalized = _normalize_key(key)
    if normalized in _HIDDEN_CHAIN_OF_THOUGHT_KEYS:
        return "hidden_chain_of_thought"
    if normalized in _BODY_FIELD_KEYS:
        return "object_body"
    if any(
        normalized == marker or normalized.endswith("_" + marker) or marker in normalized
        for marker in _PRIVATE_FIELD_MARKERS
    ):
        return "private_material"
    return None


def _reject_forbidden_keys(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            reason = _key_is_forbidden(str(raw_key))
            if reason == "hidden_chain_of_thought":
                raise RepositoryHandoffContractError(
                    f"{name} must not represent hidden chain-of-thought"
                )
            if reason == "object_body":
                raise RepositoryHandoffContractError(
                    f"{name} must not embed object bodies; use content-addressed references"
                )
            if reason == "private_material":
                _refuse(
                    RefusalReason.PRIVATE_MATERIAL,
                    f"{name} must not contain private material",
                )
            _reject_forbidden_keys(item, name=name)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            _reject_forbidden_keys(item, name=name)


def _freeze_bounded(
    value: Any,
    *,
    name: str,
    max_depth: int,
    max_items: int,
    max_text_bytes: int,
) -> Any:
    seen = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal seen
        seen += 1
        if seen > max_items:
            raise RepositoryHandoffBoundsError(f"{name} exceeds its item-count limit")
        if depth > max_depth:
            raise RepositoryHandoffBoundsError(f"{name} exceeds its nesting-depth limit")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            raise RepositoryHandoffContractError(f"{name} cannot contain floats")
        if isinstance(item, str):
            return _text(item, name, required=False, max_bytes=max_text_bytes)
        if isinstance(item, Enum):
            return visit(item.value, depth)
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise RepositoryHandoffContractError(f"{name} object keys must be strings")
            frozen: dict[str, Any] = {}
            for key in sorted(item):
                normalized_key = _text(key, f"{name} key", max_bytes=max_text_bytes)
                reason = _key_is_forbidden(normalized_key)
                if reason == "hidden_chain_of_thought":
                    raise RepositoryHandoffContractError(
                        f"{name} must not represent hidden chain-of-thought"
                    )
                if reason == "object_body":
                    raise RepositoryHandoffContractError(
                        f"{name} must not embed object bodies; use content-addressed references"
                    )
                if reason == "private_material":
                    _refuse(
                        RefusalReason.PRIVATE_MATERIAL,
                        f"{name} must not contain private material",
                    )
                frozen[normalized_key] = visit(item[key], depth + 1)
            return MappingProxyType(frozen)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return tuple(visit(member, depth + 1) for member in item)
        raise RepositoryHandoffContractError(
            f"{name} contains unsupported value type {type(item).__name__}"
        )

    return visit(value, 0)


def _distinct_identities(pairs: Sequence[tuple[str, str]]) -> None:
    seen: dict[str, str] = {}
    for name, identity in pairs:
        if not identity:
            continue
        previous = seen.get(identity)
        if previous is not None and previous != name:
            raise RepositoryHandoffIdentityError(
                f"{name} identity must be distinct from {previous}"
            )
        seen[identity] = name


def _envelope(interface: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interface": interface,
        "contract_version": REPOSITORY_HANDOFF_CONTRACT_VERSION,
        **dict(body),
    }


def _require_record_bound(
    record: CanonicalContract,
    *,
    artifact_name: str,
    bounds: RepositoryHandoffBounds | None = None,
    serialized: bool = False,
) -> None:
    size = len(record.canonical_bytes())
    if size > ABSOLUTE_MAX_RECORD_BYTES:
        raise RepositoryHandoffBoundsError(
            f"{artifact_name} exceeds the absolute record bound of "
            f"{ABSOLUTE_MAX_RECORD_BYTES} bytes"
        )
    if bounds is None:
        return
    limit = bounds.max_serialized_bytes if serialized else bounds.max_record_bytes
    limit_name = "max_serialized_bytes" if serialized else "max_record_bytes"
    if size > limit:
        raise RepositoryHandoffBoundsError(f"{artifact_name} exceeds {limit_name}")


_COMMON_WIRE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "contract_version",
        "schema_version",
        "content_id",
        "cid",
        "identity",
        "canonical_id",
    }
)


class _RepositoryHandoffCanonicalContract(CanonicalContract):
    """Canonical mixin that preserves repository-handoff error types on decode."""

    INTERFACE: ClassVar[str] = ""

    @property
    def schema_version(self) -> int:
        return REPOSITORY_HANDOFF_CONTRACT_VERSION

    @property
    def interface(self) -> str:
        return self.INTERFACE

    @classmethod
    def from_json(cls, payload: str) -> "_RepositoryHandoffCanonicalContract":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RepositoryHandoffContractError(
                "repository handoff contract JSON is malformed"
            ) from exc
        if not isinstance(value, Mapping):
            raise RepositoryHandoffContractError(
                "repository handoff contract JSON must contain an object"
            )
        decoder = getattr(cls, "from_dict", None)
        if decoder is None:
            raise RepositoryHandoffContractError(
                f"{cls.__name__} does not support from_dict"
            )
        return decoder(value)


@dataclass(frozen=True)
class RepositoryHandoffBounds(_RepositoryHandoffCanonicalContract):
    """Absolute and default count/byte/depth limits for one overlay family."""

    SCHEMA: ClassVar[str] = REPOSITORY_HANDOFF_BOUNDS_SCHEMA
    INTERFACE: ClassVar[str] = REPOSITORY_HANDOFF_BOUNDS_INTERFACE

    max_refs: int = DEFAULT_MAX_REFS
    max_index_entries: int = DEFAULT_MAX_INDEX_ENTRIES
    max_worktree_entries: int = DEFAULT_MAX_WORKTREE_ENTRIES
    max_untracked: int = DEFAULT_MAX_UNTRACKED
    max_submodules: int = DEFAULT_MAX_SUBMODULES
    max_nested_repos: int = DEFAULT_MAX_NESTED_REPOS
    max_lfs_pointers: int = DEFAULT_MAX_LFS_POINTERS
    max_sparse_patterns: int = DEFAULT_MAX_SPARSE_PATTERNS
    max_hooks: int = DEFAULT_MAX_HOOKS
    max_attributes: int = DEFAULT_MAX_ATTRIBUTES
    max_objects: int = DEFAULT_MAX_OBJECTS
    max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES
    max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES
    max_serialized_bytes: int = DEFAULT_MAX_SERIALIZED_BYTES
    max_path_bytes: int = DEFAULT_MAX_PATH_BYTES
    max_id_bytes: int = DEFAULT_MAX_ID_BYTES
    max_depth: int = DEFAULT_MAX_DEPTH
    max_unknown_fields: int = DEFAULT_MAX_UNKNOWN_FIELDS
    max_unknown_field_bytes: int = DEFAULT_MAX_UNKNOWN_FIELD_BYTES
    max_shallow_depth: int = DEFAULT_MAX_SHALLOW_DEPTH

    def __post_init__(self) -> None:
        limits = (
            ("max_refs", self.max_refs, ABSOLUTE_MAX_REFS),
            ("max_index_entries", self.max_index_entries, ABSOLUTE_MAX_INDEX_ENTRIES),
            (
                "max_worktree_entries",
                self.max_worktree_entries,
                ABSOLUTE_MAX_WORKTREE_ENTRIES,
            ),
            ("max_untracked", self.max_untracked, ABSOLUTE_MAX_UNTRACKED),
            ("max_submodules", self.max_submodules, ABSOLUTE_MAX_SUBMODULES),
            ("max_nested_repos", self.max_nested_repos, ABSOLUTE_MAX_NESTED_REPOS),
            ("max_lfs_pointers", self.max_lfs_pointers, ABSOLUTE_MAX_LFS_POINTERS),
            (
                "max_sparse_patterns",
                self.max_sparse_patterns,
                ABSOLUTE_MAX_SPARSE_PATTERNS,
            ),
            ("max_hooks", self.max_hooks, ABSOLUTE_MAX_HOOKS),
            ("max_attributes", self.max_attributes, ABSOLUTE_MAX_ATTRIBUTES),
            ("max_objects", self.max_objects, ABSOLUTE_MAX_OBJECTS),
            ("max_object_bytes", self.max_object_bytes, ABSOLUTE_MAX_OBJECT_BYTES),
            ("max_text_bytes", self.max_text_bytes, ABSOLUTE_MAX_TEXT_BYTES),
            ("max_record_bytes", self.max_record_bytes, ABSOLUTE_MAX_RECORD_BYTES),
            (
                "max_serialized_bytes",
                self.max_serialized_bytes,
                ABSOLUTE_MAX_RECORD_BYTES,
            ),
            ("max_path_bytes", self.max_path_bytes, ABSOLUTE_MAX_PATH_BYTES),
            ("max_id_bytes", self.max_id_bytes, ABSOLUTE_MAX_ID_BYTES),
            ("max_depth", self.max_depth, ABSOLUTE_MAX_DEPTH),
            (
                "max_unknown_fields",
                self.max_unknown_fields,
                ABSOLUTE_MAX_UNKNOWN_FIELDS,
            ),
            (
                "max_unknown_field_bytes",
                self.max_unknown_field_bytes,
                ABSOLUTE_MAX_UNKNOWN_FIELD_BYTES,
            ),
            (
                "max_shallow_depth",
                self.max_shallow_depth,
                ABSOLUTE_MAX_SHALLOW_DEPTH,
            ),
        )
        for field_name, value, absolute in limits:
            object.__setattr__(self, field_name, _positive_int(value, field_name))
            if getattr(self, field_name) > absolute:
                raise RepositoryHandoffBoundsError(
                    f"{field_name} exceeds the absolute limit"
                )
        if self.max_text_bytes > self.max_record_bytes:
            raise RepositoryHandoffBoundsError(
                "max_text_bytes cannot exceed max_record_bytes"
            )
        if self.max_record_bytes > self.max_serialized_bytes:
            raise RepositoryHandoffBoundsError(
                "max_record_bytes cannot exceed max_serialized_bytes"
            )
        if self.max_unknown_field_bytes > self.max_record_bytes:
            raise RepositoryHandoffBoundsError(
                "max_unknown_field_bytes cannot exceed max_record_bytes"
            )
        if self.max_path_bytes > self.max_text_bytes:
            raise RepositoryHandoffBoundsError(
                "max_path_bytes cannot exceed max_text_bytes"
            )

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "max_refs": self.max_refs,
                "max_index_entries": self.max_index_entries,
                "max_worktree_entries": self.max_worktree_entries,
                "max_untracked": self.max_untracked,
                "max_submodules": self.max_submodules,
                "max_nested_repos": self.max_nested_repos,
                "max_lfs_pointers": self.max_lfs_pointers,
                "max_sparse_patterns": self.max_sparse_patterns,
                "max_hooks": self.max_hooks,
                "max_attributes": self.max_attributes,
                "max_objects": self.max_objects,
                "max_object_bytes": self.max_object_bytes,
                "max_text_bytes": self.max_text_bytes,
                "max_record_bytes": self.max_record_bytes,
                "max_serialized_bytes": self.max_serialized_bytes,
                "max_path_bytes": self.max_path_bytes,
                "max_id_bytes": self.max_id_bytes,
                "max_depth": self.max_depth,
                "max_unknown_fields": self.max_unknown_fields,
                "max_unknown_field_bytes": self.max_unknown_field_bytes,
                "max_shallow_depth": self.max_shallow_depth,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "RepositoryHandoffBounds":
        if payload is None:
            return cls()
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="repository handoff bounds"
        )
        allowed = _COMMON_WIRE_FIELDS.union(
            {
                "max_refs",
                "max_index_entries",
                "max_worktree_entries",
                "max_untracked",
                "max_submodules",
                "max_nested_repos",
                "max_lfs_pointers",
                "max_sparse_patterns",
                "max_hooks",
                "max_attributes",
                "max_objects",
                "max_object_bytes",
                "max_text_bytes",
                "max_record_bytes",
                "max_serialized_bytes",
                "max_path_bytes",
                "max_id_bytes",
                "max_depth",
                "max_unknown_fields",
                "max_unknown_field_bytes",
                "max_shallow_depth",
            }
        )
        _reject_unknown(payload, allowed, artifact_name="repository handoff bounds")
        defaults = cls()
        result = cls(
            **{
                field: payload.get(field, getattr(defaults, field))
                for field in (
                    "max_refs",
                    "max_index_entries",
                    "max_worktree_entries",
                    "max_untracked",
                    "max_submodules",
                    "max_nested_repos",
                    "max_lfs_pointers",
                    "max_sparse_patterns",
                    "max_hooks",
                    "max_attributes",
                    "max_objects",
                    "max_object_bytes",
                    "max_text_bytes",
                    "max_record_bytes",
                    "max_serialized_bytes",
                    "max_path_bytes",
                    "max_id_bytes",
                    "max_depth",
                    "max_unknown_fields",
                    "max_unknown_field_bytes",
                    "max_shallow_depth",
                )
            }
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id"),
            artifact_name="repository handoff bounds",
        )
        return result


def _coerce_bounds(value: Any) -> RepositoryHandoffBounds:
    if value is None:
        return RepositoryHandoffBounds()
    if isinstance(value, RepositoryHandoffBounds):
        return value
    if isinstance(value, Mapping):
        return RepositoryHandoffBounds.from_dict(value)
    raise RepositoryHandoffContractError("bounds must be a RepositoryHandoffBounds object")


def _records(
    values: Any,
    cls: type[_RepositoryHandoffCanonicalContract],
    name: str,
    *,
    max_items: int,
) -> tuple[Any, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Sequence
    ):
        raise RepositoryHandoffContractError(f"{name} must be a sequence of records")
    else:
        items = values
    if len(items) > max_items:
        raise RepositoryHandoffBoundsError(f"{name} exceeds its item-count limit")
    result: list[Any] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, cls):
            record = item
        elif isinstance(item, Mapping):
            record = cls.from_dict(item)
        else:
            raise RepositoryHandoffContractError(f"{name} must contain {cls.__name__} objects")
        identity = record.content_id
        if identity in seen:
            raise RepositoryHandoffContractError(f"{name} must not contain duplicate identities")
        seen.add(identity)
        result.append(record)
    return tuple(result)


def _origin_url(value: Any, name: str, *, required: bool = False) -> str:
    text = _text(value, name, required=required, max_bytes=ABSOLUTE_MAX_TEXT_BYTES)
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("file:") or text.startswith(("/", "\\")) or (
        len(text) > 1 and text[1] == ":"
    ):
        _refuse(
            RefusalReason.HOST_PATH_ORIGIN,
            f"{name} must not be a host filesystem path",
        )
    parsed = urlsplit(text)
    if parsed.scheme:
        scheme = parsed.scheme.lower()
        if scheme in _FORBIDDEN_ORIGIN_SCHEMES or scheme not in _ALLOWED_ORIGIN_SCHEMES:
            _refuse(
                RefusalReason.HOST_PATH_ORIGIN,
                f"{name} must not use a host-path or forbidden URL scheme",
            )
        if parsed.netloc in ("", ".", "..") or parsed.path.startswith("//"):
            _refuse(
                RefusalReason.HOST_PATH_ORIGIN,
                f"{name} must not be a host filesystem path",
            )
        return text
    if _SCP_LIKE_URL_RE.fullmatch(text) or _RELATIVE_GIT_URL_RE.fullmatch(text):
        return text
    _refuse(
        RefusalReason.HOST_PATH_ORIGIN,
        f"{name} must be a Git URL, scp-like remote, or relative Git URL",
    )
    return text


@dataclass(frozen=True)
class RefEntry:
    """One HEAD or refs/ entry in the overlay."""

    name: str
    object_id: str
    symbolic_target: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _git_ref_name(self.name, "ref name", allow_head=True)
        )
        object.__setattr__(self, "object_id", _git_oid(self.object_id, "ref object_id"))
        target = _text(
            self.symbolic_target,
            "symbolic_target",
            required=False,
            max_bytes=DEFAULT_MAX_PATH_BYTES,
        )
        if target:
            target = _git_ref_name(target, "symbolic_target")
            if self.name != "HEAD" and not self.name.startswith("refs/heads/"):
                raise RepositoryHandoffContractError(
                    "only HEAD or a branch may be a symbolic ref"
                )
        object.__setattr__(self, "symbolic_target", target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "object_id": self.object_id,
            "symbolic_target": self.symbolic_target,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefEntry":
        if not isinstance(payload, Mapping):
            raise RepositoryHandoffContractError("ref entry must be an object")
        _reject_unknown(
            payload,
            {"name", "object_id", "symbolic_target"},
            artifact_name="ref entry",
        )
        return cls(
            name=payload.get("name", ""),
            object_id=payload.get("object_id", ""),
            symbolic_target=payload.get("symbolic_target", ""),
        )


@dataclass(frozen=True)
class IndexEntry:
    """One index (staging area) path, mode, stage, and Git object id."""

    path: str
    mode: int
    object_id: str
    stage: int = 0
    skip_worktree: bool = False
    intent_to_add: bool = False
    kind: FileKind | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, "index path"))
        _reject_git_dir_path(self.path, "index path")
        mode = _git_mode(self.mode, "index mode")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self, "kind", _file_kind_for_mode(mode, self.kind, "index kind")
        )
        object.__setattr__(self, "object_id", _git_oid(self.object_id, "index object_id"))
        stage = _nonnegative_int(self.stage, "stage")
        if stage > 3:
            raise RepositoryHandoffContractError("index stage must be 0..3")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(
            self, "skip_worktree", _bool(self.skip_worktree, "skip_worktree")
        )
        object.__setattr__(
            self, "intent_to_add", _bool(self.intent_to_add, "intent_to_add")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode,
            "kind": self.kind.value,
            "object_id": self.object_id,
            "stage": self.stage,
            "skip_worktree": self.skip_worktree,
            "intent_to_add": self.intent_to_add,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IndexEntry":
        if not isinstance(payload, Mapping):
            raise RepositoryHandoffContractError("index entry must be an object")
        _reject_unknown(
            payload,
            {
                "path",
                "mode",
                "kind",
                "object_id",
                "stage",
                "skip_worktree",
                "intent_to_add",
            },
            artifact_name="index entry",
        )
        return cls(
            path=payload.get("path", ""),
            mode=payload.get("mode", 0o100644),
            object_id=payload.get("object_id", ""),
            stage=payload.get("stage", 0),
            skip_worktree=payload.get("skip_worktree", False),
            intent_to_add=payload.get("intent_to_add", False),
            kind=payload.get("kind"),
        )


@dataclass(frozen=True)
class WorktreeEntry:
    """One worktree path with mode, kind, and content-addressed bytes."""

    path: str
    mode: int
    kind: FileKind | None = None
    object_id: str = ""
    content_id: str = ""
    symlink_target: str = ""
    byte_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, "worktree path"))
        _reject_git_dir_path(self.path, "worktree path")
        mode = _git_mode(self.mode, "worktree mode")
        object.__setattr__(self, "mode", mode)
        kind = _file_kind_for_mode(mode, self.kind, "worktree kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "object_id",
            _git_oid(self.object_id, "worktree object_id", required=False),
        )
        object.__setattr__(
            self,
            "content_id",
            _content_ref(self.content_id, "worktree content_id", required=False),
        )
        object.__setattr__(
            self, "byte_count", _nonnegative_int(self.byte_count, "byte_count")
        )
        if kind is FileKind.SYMLINK:
            object.__setattr__(
                self,
                "symlink_target",
                _symlink_target(self.symlink_target, "symlink_target", required=True),
            )
        else:
            if _text(self.symlink_target, "symlink_target", required=False):
                raise RepositoryHandoffContractError(
                    "symlink_target is only valid for symlink entries"
                )
            object.__setattr__(self, "symlink_target", "")
            if kind in {FileKind.REGULAR, FileKind.EXECUTABLE} and not self.content_id:
                raise RepositoryHandoffContractError(
                    "worktree file entries require a content-addressed content_id"
                )


    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode,
            "kind": self.kind.value,
            "object_id": self.object_id,
            "content_id": self.content_id,
            "symlink_target": self.symlink_target,
            "byte_count": self.byte_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorktreeEntry":
        if not isinstance(payload, Mapping):
            raise RepositoryHandoffContractError("worktree entry must be an object")
        _reject_unknown(
            payload,
            {
                "path",
                "mode",
                "kind",
                "object_id",
                "content_id",
                "symlink_target",
                "byte_count",
            },
            artifact_name="worktree entry",
        )
        return cls(
            path=payload.get("path", ""),
            mode=payload.get("mode", 0o100644),
            kind=payload.get("kind"),
            object_id=payload.get("object_id", ""),
            content_id=payload.get("content_id", ""),
            symlink_target=payload.get("symlink_target", ""),
            byte_count=payload.get("byte_count", 0),
        )


@dataclass(frozen=True)
class UntrackedEntry:
    """One untracked path.  Bodies stay behind content identities."""

    path: str
    mode: int
    kind: FileKind | None = None
    content_id: str = ""
    symlink_target: str = ""
    byte_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, "untracked path"))
        _reject_git_dir_path(self.path, "untracked path")
        mode = _git_mode(self.mode, "untracked mode")
        object.__setattr__(self, "mode", mode)
        kind = _file_kind_for_mode(mode, self.kind, "untracked kind")
        object.__setattr__(self, "kind", kind)
        if kind is FileKind.GITLINK:
            raise RepositoryHandoffContractError(
                "untracked entries cannot be gitlinks; use SubmoduleRecord"
            )
        object.__setattr__(
            self,
            "content_id",
            _content_ref(self.content_id, "untracked content_id", required=False),
        )
        object.__setattr__(
            self, "byte_count", _nonnegative_int(self.byte_count, "byte_count")
        )
        if kind is FileKind.SYMLINK:
            object.__setattr__(
                self,
                "symlink_target",
                _symlink_target(self.symlink_target, "symlink_target", required=True),
            )
        else:
            if _text(self.symlink_target, "symlink_target", required=False):
                raise RepositoryHandoffContractError(
                    "symlink_target is only valid for symlink entries"
                )
            object.__setattr__(self, "symlink_target", "")
            if kind in {FileKind.REGULAR, FileKind.EXECUTABLE} and not self.content_id:
                raise RepositoryHandoffContractError(
                    "untracked file entries require a content-addressed content_id"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode,
            "kind": self.kind.value,
            "content_id": self.content_id,
            "symlink_target": self.symlink_target,
            "byte_count": self.byte_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UntrackedEntry":
        if not isinstance(payload, Mapping):
            raise RepositoryHandoffContractError("untracked entry must be an object")
        _reject_unknown(
            payload,
            {"path", "mode", "kind", "content_id", "symlink_target", "byte_count"},
            artifact_name="untracked entry",
        )
        return cls(
            path=payload.get("path", ""),
            mode=payload.get("mode", 0o100644),
            kind=payload.get("kind"),
            content_id=payload.get("content_id", ""),
            symlink_target=payload.get("symlink_target", ""),
            byte_count=payload.get("byte_count", 0),
        )


def _entry_sequence(
    values: Any,
    cls: type[Any],
    name: str,
    *,
    max_items: int,
) -> tuple[Any, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Sequence
    ):
        raise RepositoryHandoffContractError(f"{name} must be a sequence")
    else:
        items = values
    if len(items) > max_items:
        raise RepositoryHandoffBoundsError(f"{name} exceeds its item-count limit")
    result: list[Any] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, cls):
            entry = item
        elif isinstance(item, Mapping):
            entry = cls.from_dict(item)
        else:
            raise RepositoryHandoffContractError(f"{name} must contain {cls.__name__} objects")
        key = getattr(entry, "name", None) or entry.path
        if cls is IndexEntry:
            key = f"{entry.path}:{entry.stage}"
        if key in seen:
            raise RepositoryHandoffContractError(f"{name} must not contain duplicate paths")
        seen.add(key)
        result.append(entry)
    return tuple(result)


@dataclass(frozen=True)
class RepositoryOverlay(_RepositoryHandoffCanonicalContract):
    """Exact HEAD, refs, index, worktree, and untracked overlay."""

    SCHEMA: ClassVar[str] = REPOSITORY_OVERLAY_SCHEMA
    INTERFACE: ClassVar[str] = REPOSITORY_OVERLAY_INTERFACE

    head_commit: str
    refs: tuple[RefEntry, ...]
    index: tuple[IndexEntry, ...] = ()
    worktree: tuple[WorktreeEntry, ...] = ()
    untracked: tuple[UntrackedEntry, ...] = ()
    head_name: str = "HEAD"
    head_ref: str = ""
    detached: bool = False
    object_count: int = 0
    object_bytes: int = 0
    unbounded_objects: bool = False
    bounds: RepositoryHandoffBounds = RepositoryHandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(
            self, "head_commit", _git_oid(self.head_commit, "head_commit")
        )
        object.__setattr__(
            self,
            "head_name",
            _git_ref_name(self.head_name, "head_name", allow_head=True),
        )
        if self.head_name != "HEAD":
            raise RepositoryHandoffContractError("head_name must be HEAD")
        object.__setattr__(self, "detached", _bool(self.detached, "detached"))
        head_ref = _text(
            self.head_ref, "head_ref", required=False, max_bytes=DEFAULT_MAX_PATH_BYTES
        )
        if head_ref:
            head_ref = _git_ref_name(head_ref, "head_ref")
        object.__setattr__(self, "head_ref", head_ref)
        object.__setattr__(
            self,
            "refs",
            _entry_sequence(self.refs, RefEntry, "refs", max_items=self.bounds.max_refs),
        )
        object.__setattr__(
            self,
            "index",
            _entry_sequence(
                self.index, IndexEntry, "index", max_items=self.bounds.max_index_entries
            ),
        )
        object.__setattr__(
            self,
            "worktree",
            _entry_sequence(
                self.worktree,
                WorktreeEntry,
                "worktree",
                max_items=self.bounds.max_worktree_entries,
            ),
        )
        object.__setattr__(
            self,
            "untracked",
            _entry_sequence(
                self.untracked,
                UntrackedEntry,
                "untracked",
                max_items=self.bounds.max_untracked,
            ),
        )
        object.__setattr__(
            self, "object_count", _nonnegative_int(self.object_count, "object_count")
        )
        object.__setattr__(
            self, "object_bytes", _nonnegative_int(self.object_bytes, "object_bytes")
        )
        object.__setattr__(
            self,
            "unbounded_objects",
            _bool(self.unbounded_objects, "unbounded_objects"),
        )
        if self.unbounded_objects:
            _refuse(
                RefusalReason.UNBOUNDED_OBJECTS,
                "overlay declares unbounded objects and is refused",
            )
        if self.object_count > self.bounds.max_objects or (
            self.object_bytes > self.bounds.max_object_bytes
        ):
            _refuse(
                RefusalReason.UNBOUNDED_OBJECTS,
                "overlay object count or bytes exceed declared bounds",
            )
        if not self.refs:
            raise RepositoryHandoffContractError("overlay refs must include HEAD")
        refs_by_name = {item.name: item for item in self.refs}
        head = refs_by_name.get("HEAD")
        if head is None:
            raise RepositoryHandoffContractError("overlay refs must include HEAD")
        if head.object_id != self.head_commit:
            raise RepositoryHandoffIdentityError(
                "HEAD object id must match head_commit"
            )
        if self.detached:
            if self.head_ref:
                raise RepositoryHandoffContractError(
                    "detached HEAD must not name a branch ref"
                )
            if head.symbolic_target:
                raise RepositoryHandoffContractError(
                    "detached HEAD must not be symbolic"
                )
        else:
            if not self.head_ref:
                raise RepositoryHandoffContractError(
                    "attached HEAD requires head_ref"
                )
            if head.symbolic_target != self.head_ref:
                raise RepositoryHandoffContractError(
                    "HEAD symbolic_target must match head_ref"
                )
            branch = refs_by_name.get(self.head_ref)
            if branch is None:
                raise RepositoryHandoffContractError(
                    "head_ref must be present in overlay refs"
                )
            if branch.object_id != self.head_commit:
                raise RepositoryHandoffIdentityError(
                    "head_ref object id must match head_commit"
                )
        index_paths = {item.path for item in self.index if item.stage == 0}
        worktree_paths = {item.path for item in self.worktree}
        untracked_paths = {item.path for item in self.untracked}
        overlap = worktree_paths.intersection(untracked_paths)
        if overlap:
            raise RepositoryHandoffContractError(
                "worktree and untracked paths must be disjoint"
            )
        tracked_untracked = index_paths.intersection(untracked_paths)
        if tracked_untracked:
            raise RepositoryHandoffContractError(
                "untracked paths must not also be indexed"
            )
        _reject_forbidden_keys(self.to_dict(), name="repository overlay")
        _require_record_bound(
            self, artifact_name="repository overlay", bounds=self.bounds, serialized=True
        )

    @property
    def overlay_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "head_name": self.head_name,
                "head_ref": self.head_ref,
                "head_commit": self.head_commit,
                "detached": self.detached,
                "refs": [item.to_dict() for item in self.refs],
                "index": [item.to_dict() for item in self.index],
                "worktree": [item.to_dict() for item in self.worktree],
                "untracked": [item.to_dict() for item in self.untracked],
                "object_count": self.object_count,
                "object_bytes": self.object_bytes,
                "unbounded_objects": False,
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepositoryOverlay":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="repository overlay"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "head_name",
                    "head_ref",
                    "head_commit",
                    "detached",
                    "refs",
                    "index",
                    "worktree",
                    "untracked",
                    "object_count",
                    "object_bytes",
                    "unbounded_objects",
                    "bounds",
                    "overlay_id",
                }
            ),
            artifact_name="repository overlay",
        )
        result = cls(
            head_commit=payload.get("head_commit", ""),
            refs=payload.get("refs", ()),
            index=payload.get("index", ()),
            worktree=payload.get("worktree", ()),
            untracked=payload.get("untracked", ()),
            head_name=payload.get("head_name", "HEAD"),
            head_ref=payload.get("head_ref", ""),
            detached=payload.get("detached", False),
            object_count=payload.get("object_count", 0),
            object_bytes=payload.get("object_bytes", 0),
            unbounded_objects=payload.get("unbounded_objects", False),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "overlay_id"),
            artifact_name="repository overlay",
        )
        return result


def _coerce_overlay(value: Any) -> RepositoryOverlay:
    if isinstance(value, RepositoryOverlay):
        return value
    if isinstance(value, Mapping):
        return RepositoryOverlay.from_dict(value)
    raise RepositoryHandoffContractError("overlay must be a RepositoryOverlay object")


@dataclass(frozen=True)
class SubmoduleRecord(_RepositoryHandoffCanonicalContract):
    """One recorded submodule gitlink.  URLs must not be host paths."""

    SCHEMA: ClassVar[str] = SUBMODULE_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = SUBMODULE_RECORD_INTERFACE

    path: str
    commit: str
    url: str = ""
    name: str = ""
    branch: str = ""
    parent_path: str = ""
    ignore: SubmoduleIgnore = SubmoduleIgnore.NONE
    recursive: bool = False
    shallow: bool = False
    fetch_recurse: bool = False
    bounds: RepositoryHandoffBounds = RepositoryHandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(self, "path", _relative_path(self.path, "submodule path"))
        _reject_git_dir_path(self.path, "submodule path")
        object.__setattr__(self, "commit", _git_oid(self.commit, "submodule commit"))
        object.__setattr__(
            self, "url", _origin_url(self.url, "submodule url", required=False)
        )
        object.__setattr__(
            self,
            "name",
            _text(
                self.name or self.path,
                "submodule name",
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "branch",
            _text(
                self.branch,
                "branch",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "parent_path",
            _relative_path(self.parent_path, "parent_path", required=False),
        )
        if self.parent_path:
            _reject_git_dir_path(self.parent_path, "parent_path")
        object.__setattr__(self, "ignore", _enum(self.ignore, SubmoduleIgnore, "ignore"))
        object.__setattr__(self, "recursive", _bool(self.recursive, "recursive"))
        object.__setattr__(self, "shallow", _bool(self.shallow, "shallow"))
        object.__setattr__(
            self, "fetch_recurse", _bool(self.fetch_recurse, "fetch_recurse")
        )
        _reject_forbidden_keys(self.to_dict(), name="submodule record")
        _require_record_bound(
            self, artifact_name="submodule record", bounds=self.bounds
        )

    @property
    def submodule_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "path": self.path,
                "commit": self.commit,
                "url": self.url,
                "name": self.name,
                "branch": self.branch,
                "parent_path": self.parent_path,
                "ignore": self.ignore.value,
                "recursive": self.recursive,
                "shallow": self.shallow,
                "fetch_recurse": self.fetch_recurse,
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubmoduleRecord":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="submodule record"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "path",
                    "commit",
                    "url",
                    "name",
                    "branch",
                    "parent_path",
                    "ignore",
                    "recursive",
                    "shallow",
                    "fetch_recurse",
                    "bounds",
                    "submodule_id",
                }
            ),
            artifact_name="submodule record",
        )
        result = cls(
            path=payload.get("path", ""),
            commit=payload.get("commit", ""),
            url=payload.get("url", ""),
            name=payload.get("name", ""),
            branch=payload.get("branch", ""),
            parent_path=payload.get("parent_path", ""),
            ignore=payload.get("ignore", SubmoduleIgnore.NONE),
            recursive=payload.get("recursive", False),
            shallow=payload.get("shallow", False),
            fetch_recurse=payload.get("fetch_recurse", False),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "submodule_id"),
            artifact_name="submodule record",
        )
        return result


@dataclass(frozen=True)
class NestedRepoRecord(_RepositoryHandoffCanonicalContract):
    """A nested ``.git`` directory that is not a recorded submodule."""

    SCHEMA: ClassVar[str] = NESTED_REPO_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = NESTED_REPO_RECORD_INTERFACE

    path: str
    head_commit: str
    git_dir_kind: NestedGitDirKind = NestedGitDirKind.GIT_DIR
    git_dir_path: str = ""
    registered_submodule: bool = False
    bounds: RepositoryHandoffBounds = RepositoryHandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(self, "path", _relative_path(self.path, "nested repo path"))
        _reject_git_dir_path(self.path, "nested repo path")
        object.__setattr__(
            self, "head_commit", _git_oid(self.head_commit, "nested head_commit")
        )
        object.__setattr__(
            self,
            "git_dir_kind",
            _enum(self.git_dir_kind, NestedGitDirKind, "git_dir_kind"),
        )
        git_dir_path = _relative_path(
            self.git_dir_path, "git_dir_path", required=False
        )
        if git_dir_path:
            candidate = PurePosixPath(git_dir_path)
            if ".." in candidate.parts or candidate.is_absolute():
                _refuse(
                    RefusalReason.NESTED_GIT_ESCAPE,
                    "nested git_dir_path is a nested git escape",
                )
        object.__setattr__(self, "git_dir_path", git_dir_path)
        object.__setattr__(
            self,
            "registered_submodule",
            _bool(self.registered_submodule, "registered_submodule"),
        )
        if self.git_dir_kind is NestedGitDirKind.GITFILE and not self.git_dir_path:
            raise RepositoryHandoffContractError(
                "gitfile nested repos require a repository-relative git_dir_path"
            )
        _reject_forbidden_keys(self.to_dict(), name="nested repo record")
        _require_record_bound(
            self, artifact_name="nested repo record", bounds=self.bounds
        )

    @property
    def nested_repo_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "path": self.path,
                "head_commit": self.head_commit,
                "git_dir_kind": self.git_dir_kind.value,
                "git_dir_path": self.git_dir_path,
                "registered_submodule": self.registered_submodule,
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NestedRepoRecord":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="nested repo record"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "path",
                    "head_commit",
                    "git_dir_kind",
                    "git_dir_path",
                    "registered_submodule",
                    "bounds",
                    "nested_repo_id",
                }
            ),
            artifact_name="nested repo record",
        )
        result = cls(
            path=payload.get("path", ""),
            head_commit=payload.get("head_commit", ""),
            git_dir_kind=payload.get("git_dir_kind", NestedGitDirKind.GIT_DIR),
            git_dir_path=payload.get("git_dir_path", ""),
            registered_submodule=payload.get("registered_submodule", False),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "nested_repo_id"),
            artifact_name="nested repo record",
        )
        return result


@dataclass(frozen=True)
class LfsPointerRecord(_RepositoryHandoffCanonicalContract):
    """One Git LFS pointer.  The large object stays content-addressed."""

    SCHEMA: ClassVar[str] = LFS_POINTER_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = LFS_POINTER_RECORD_INTERFACE

    path: str
    oid: str
    size_bytes: int
    version: str = LFS_POINTER_VERSION
    pointer_content_id: str = ""
    bounds: RepositoryHandoffBounds = RepositoryHandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(self, "path", _relative_path(self.path, "lfs path"))
        _reject_git_dir_path(self.path, "lfs path")
        object.__setattr__(self, "oid", _digest_sha256(self.oid, "lfs oid"))
        object.__setattr__(
            self, "size_bytes", _nonnegative_int(self.size_bytes, "size_bytes")
        )
        version = _text(self.version, "lfs version", max_bytes=self.bounds.max_id_bytes * 4)
        if version != LFS_POINTER_VERSION:
            raise RepositoryHandoffVersionError(
                f"unsupported LFS pointer version; expected {LFS_POINTER_VERSION}"
            )
        object.__setattr__(self, "version", version)
        object.__setattr__(
            self,
            "pointer_content_id",
            _content_ref(
                self.pointer_content_id, "pointer_content_id", required=False
            ),
        )
        if self.size_bytes > self.bounds.max_object_bytes:
            _refuse(
                RefusalReason.UNBOUNDED_OBJECTS,
                "LFS object exceeds declared object-byte bounds",
            )
        _reject_forbidden_keys(self.to_dict(), name="lfs pointer record")
        _require_record_bound(
            self, artifact_name="lfs pointer record", bounds=self.bounds
        )

    @property
    def lfs_pointer_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "path": self.path,
                "oid": self.oid,
                "size_bytes": self.size_bytes,
                "version": self.version,
                "pointer_content_id": self.pointer_content_id,
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LfsPointerRecord":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="lfs pointer record"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "path",
                    "oid",
                    "size_bytes",
                    "version",
                    "pointer_content_id",
                    "bounds",
                    "lfs_pointer_id",
                }
            ),
            artifact_name="lfs pointer record",
        )
        result = cls(
            path=payload.get("path", ""),
            oid=payload.get("oid", ""),
            size_bytes=payload.get("size_bytes", 0),
            version=payload.get("version", LFS_POINTER_VERSION),
            pointer_content_id=payload.get("pointer_content_id", ""),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "lfs_pointer_id"),
            artifact_name="lfs pointer record",
        )
        return result


@dataclass(frozen=True)
class SparseCheckoutRecord(_RepositoryHandoffCanonicalContract):
    """Sparse-checkout patterns, cone mode, and sparse-index flag."""

    SCHEMA: ClassVar[str] = SPARSE_CHECKOUT_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = SPARSE_CHECKOUT_RECORD_INTERFACE

    enabled: bool = False
    cone: bool = False
    sparse_index: bool = False
    patterns: tuple[str, ...] = ()
    skip_worktree_count: int = 0
    bounds: RepositoryHandoffBounds = RepositoryHandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(self, "enabled", _bool(self.enabled, "enabled"))
        object.__setattr__(self, "cone", _bool(self.cone, "cone"))
        object.__setattr__(
            self, "sparse_index", _bool(self.sparse_index, "sparse_index")
        )
        if self.patterns is None:
            patterns: Sequence[Any] = ()
        elif isinstance(self.patterns, (str, bytes, bytearray, memoryview)) or not isinstance(
            self.patterns, Sequence
        ):
            raise RepositoryHandoffContractError("patterns must be a sequence of paths")
        else:
            patterns = self.patterns
        if len(patterns) > self.bounds.max_sparse_patterns:
            raise RepositoryHandoffBoundsError("patterns exceeds its item-count limit")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in patterns:
            path = _relative_path(item, "sparse pattern")
            if path in seen:
                raise RepositoryHandoffContractError(
                    "patterns must not contain duplicate paths"
                )
            seen.add(path)
            normalized.append(path)
        object.__setattr__(self, "patterns", tuple(normalized))
        object.__setattr__(
            self,
            "skip_worktree_count",
            _nonnegative_int(self.skip_worktree_count, "skip_worktree_count"),
        )
        if self.skip_worktree_count > self.bounds.max_index_entries:
            raise RepositoryHandoffBoundsError(
                "skip_worktree_count exceeds max_index_entries"
            )
        if not self.enabled and (self.patterns or self.sparse_index or self.cone):
            raise RepositoryHandoffContractError(
                "disabled sparse checkout cannot declare cone, sparse-index, or patterns"
            )
        _reject_forbidden_keys(self.to_dict(), name="sparse checkout record")
        _require_record_bound(
            self, artifact_name="sparse checkout record", bounds=self.bounds
        )

    @property
    def sparse_checkout_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "enabled": self.enabled,
                "cone": self.cone,
                "sparse_index": self.sparse_index,
                "patterns": list(self.patterns),
                "skip_worktree_count": self.skip_worktree_count,
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SparseCheckoutRecord":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="sparse checkout record"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "enabled",
                    "cone",
                    "sparse_index",
                    "patterns",
                    "skip_worktree_count",
                    "bounds",
                    "sparse_checkout_id",
                }
            ),
            artifact_name="sparse checkout record",
        )
        result = cls(
            enabled=payload.get("enabled", False),
            cone=payload.get("cone", False),
            sparse_index=payload.get("sparse_index", False),
            patterns=payload.get("patterns", ()),
            skip_worktree_count=payload.get("skip_worktree_count", 0),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=(
                "content_id",
                "cid",
                "identity",
                "canonical_id",
                "sparse_checkout_id",
            ),
            artifact_name="sparse checkout record",
        )
        return result


def _coerce_sparse(value: Any) -> SparseCheckoutRecord:
    if value is None:
        return SparseCheckoutRecord()
    if isinstance(value, SparseCheckoutRecord):
        return value
    if isinstance(value, Mapping):
        return SparseCheckoutRecord.from_dict(value)
    raise RepositoryHandoffContractError(
        "sparse_checkout must be a SparseCheckoutRecord object"
    )


@dataclass(frozen=True)
class HookPolicy(_RepositoryHandoffCanonicalContract):
    """Hook inventory.  Hooks must be disabled on import."""

    SCHEMA: ClassVar[str] = HOOK_POLICY_SCHEMA
    INTERFACE: ClassVar[str] = HOOK_POLICY_INTERFACE

    present_hook_names: tuple[str, ...] = ()
    hooks_enabled: bool = False
    import_hooks_disabled: bool = True
    core_hooks_path: str = ""
    bounds: RepositoryHandoffBounds = RepositoryHandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(
            self, "hooks_enabled", _bool(self.hooks_enabled, "hooks_enabled")
        )
        object.__setattr__(
            self,
            "import_hooks_disabled",
            _bool(self.import_hooks_disabled, "import_hooks_disabled"),
        )
        if self.hooks_enabled or not self.import_hooks_disabled:
            _refuse(
                RefusalReason.ENABLED_HOOKS,
                "hooks must be disabled on import; enabled hooks are refused",
            )
        object.__setattr__(
            self,
            "core_hooks_path",
            _text(
                self.core_hooks_path,
                "core_hooks_path",
                required=False,
                max_bytes=self.bounds.max_path_bytes,
            ),
        )
        if self.core_hooks_path:
            _refuse(
                RefusalReason.ENABLED_HOOKS,
                "custom core.hooksPath must be disabled on import",
            )
        if self.present_hook_names is None:
            names: Sequence[Any] = ()
        elif isinstance(
            self.present_hook_names, (str, bytes, bytearray, memoryview)
        ) or not isinstance(self.present_hook_names, Sequence):
            raise RepositoryHandoffContractError(
                "present_hook_names must be a sequence of strings"
            )
        else:
            names = self.present_hook_names
        if len(names) > self.bounds.max_hooks:
            raise RepositoryHandoffBoundsError(
                "present_hook_names exceeds its item-count limit"
            )
        normalized: list[str] = []
        seen: set[str] = set()
        for item in names:
            text = _text(item, "hook name", max_bytes=self.bounds.max_id_bytes)
            if not _HOOK_NAME_RE.fullmatch(text) or text.endswith(".sample"):
                raise RepositoryHandoffContractError("hook name is not admitted")
            if text in seen:
                raise RepositoryHandoffContractError(
                    "present_hook_names must not contain duplicates"
                )
            seen.add(text)
            normalized.append(text)
        object.__setattr__(self, "present_hook_names", tuple(normalized))
        object.__setattr__(self, "hooks_enabled", False)
        object.__setattr__(self, "import_hooks_disabled", True)
        _reject_forbidden_keys(self.to_dict(), name="hook policy")
        _require_record_bound(self, artifact_name="hook policy", bounds=self.bounds)

    @property
    def hook_policy_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "present_hook_names": list(self.present_hook_names),
                "hooks_enabled": False,
                "import_hooks_disabled": True,
                "core_hooks_path": "",
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HookPolicy":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="hook policy"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "present_hook_names",
                    "hooks_enabled",
                    "import_hooks_disabled",
                    "core_hooks_path",
                    "bounds",
                    "hook_policy_id",
                }
            ),
            artifact_name="hook policy",
        )
        result = cls(
            present_hook_names=payload.get("present_hook_names", ()),
            hooks_enabled=payload.get("hooks_enabled", False),
            import_hooks_disabled=payload.get("import_hooks_disabled", True),
            core_hooks_path=payload.get("core_hooks_path", ""),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "hook_policy_id"),
            artifact_name="hook policy",
        )
        return result


def _coerce_hooks(value: Any) -> HookPolicy:
    if value is None:
        return HookPolicy()
    if isinstance(value, HookPolicy):
        return value
    if isinstance(value, Mapping):
        return HookPolicy.from_dict(value)
    raise RepositoryHandoffContractError("hook_policy must be a HookPolicy object")


@dataclass(frozen=True)
class AttributeAndModeRecord(_RepositoryHandoffCanonicalContract):
    """Exact path mode plus bounded gitattributes projection."""

    SCHEMA: ClassVar[str] = ATTRIBUTE_AND_MODE_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = ATTRIBUTE_AND_MODE_RECORD_INTERFACE

    path: str
    mode: int
    kind: FileKind | None = None
    attributes: Mapping[str, str] = MappingProxyType({})
    ident: bool = False
    bounds: RepositoryHandoffBounds = RepositoryHandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(self, "path", _relative_path(self.path, "attribute path"))
        _reject_git_dir_path(self.path, "attribute path")
        mode = _git_mode(self.mode, "attribute mode")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self, "kind", _file_kind_for_mode(mode, self.kind, "attribute kind")
        )
        object.__setattr__(self, "ident", _bool(self.ident, "ident"))
        if self.attributes is None:
            attributes: Any = {}
        else:
            attributes = self.attributes
        if not isinstance(attributes, Mapping):
            raise RepositoryHandoffContractError("attributes must be an object")
        frozen = _freeze_bounded(
            attributes,
            name="attributes",
            max_depth=self.bounds.max_depth,
            max_items=ABSOLUTE_MAX_ITEMS,
            max_text_bytes=self.bounds.max_text_bytes,
        )
        if not isinstance(frozen, Mapping):
            raise RepositoryHandoffContractError("attributes must be an object")
        for value in frozen.values():
            if not isinstance(value, str):
                raise RepositoryHandoffContractError(
                    "attribute values must be strings"
                )
        object.__setattr__(self, "attributes", frozen)
        _reject_forbidden_keys(self.to_dict(), name="attribute and mode record")
        _require_record_bound(
            self, artifact_name="attribute and mode record", bounds=self.bounds
        )

    @property
    def attribute_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "path": self.path,
                "mode": self.mode,
                "kind": self.kind.value,
                "attributes": dict(self.attributes),
                "ident": self.ident,
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttributeAndModeRecord":
        _schema_and_version(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="attribute and mode record",
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "path",
                    "mode",
                    "kind",
                    "attributes",
                    "ident",
                    "bounds",
                    "attribute_id",
                }
            ),
            artifact_name="attribute and mode record",
        )
        result = cls(
            path=payload.get("path", ""),
            mode=payload.get("mode", 0o100644),
            kind=payload.get("kind"),
            attributes=payload.get("attributes", {}),
            ident=payload.get("ident", False),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "attribute_id"),
            artifact_name="attribute and mode record",
        )
        return result


@dataclass(frozen=True)
class OriginAndShallowBounds(_RepositoryHandoffCanonicalContract):
    """Origin URL policy plus shallow/promisor object bounds."""

    SCHEMA: ClassVar[str] = ORIGIN_AND_SHALLOW_BOUNDS_SCHEMA
    INTERFACE: ClassVar[str] = ORIGIN_AND_SHALLOW_BOUNDS_INTERFACE

    origin_url: str = ""
    origin_name: str = "origin"
    shallow: bool = False
    depth: int = 0
    shallow_since_epoch_s: int = 0
    filter_spec: str = ""
    promisor: bool = False
    unshallow_required: bool = False
    object_count: int = 0
    object_bytes: int = 0
    max_objects: int = DEFAULT_MAX_OBJECTS
    max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES
    unbounded_objects: bool = False
    bounds: RepositoryHandoffBounds = RepositoryHandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(
            self, "origin_url", _origin_url(self.origin_url, "origin_url")
        )
        object.__setattr__(
            self,
            "origin_name",
            _text(
                self.origin_name,
                "origin_name",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            )
            or "origin",
        )
        object.__setattr__(self, "shallow", _bool(self.shallow, "shallow"))
        object.__setattr__(self, "depth", _nonnegative_int(self.depth, "depth"))
        object.__setattr__(
            self,
            "shallow_since_epoch_s",
            _nonnegative_int(self.shallow_since_epoch_s, "shallow_since_epoch_s"),
        )
        object.__setattr__(
            self,
            "filter_spec",
            _text(
                self.filter_spec,
                "filter_spec",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(self, "promisor", _bool(self.promisor, "promisor"))
        object.__setattr__(
            self,
            "unshallow_required",
            _bool(self.unshallow_required, "unshallow_required"),
        )
        object.__setattr__(
            self, "object_count", _nonnegative_int(self.object_count, "object_count")
        )
        object.__setattr__(
            self, "object_bytes", _nonnegative_int(self.object_bytes, "object_bytes")
        )
        object.__setattr__(
            self, "max_objects", _positive_int(self.max_objects, "max_objects")
        )
        object.__setattr__(
            self,
            "max_object_bytes",
            _positive_int(self.max_object_bytes, "max_object_bytes"),
        )
        object.__setattr__(
            self,
            "unbounded_objects",
            _bool(self.unbounded_objects, "unbounded_objects"),
        )
        if self.max_objects > self.bounds.max_objects:
            raise RepositoryHandoffBoundsError("max_objects exceeds overlay bounds")
        if self.max_object_bytes > self.bounds.max_object_bytes:
            raise RepositoryHandoffBoundsError(
                "max_object_bytes exceeds overlay bounds"
            )
        if self.depth > self.bounds.max_shallow_depth:
            raise RepositoryHandoffBoundsError("depth exceeds max_shallow_depth")
        if self.unbounded_objects:
            _refuse(
                RefusalReason.UNBOUNDED_OBJECTS,
                "origin declares unbounded objects and is refused",
            )
        if self.object_count > self.max_objects or self.object_bytes > self.max_object_bytes:
            _refuse(
                RefusalReason.UNBOUNDED_OBJECTS,
                "origin object count or bytes exceed declared bounds",
            )
        if self.shallow:
            if self.depth < 1 and self.shallow_since_epoch_s < 1:
                raise RepositoryHandoffContractError(
                    "shallow repositories require depth or shallow_since_epoch_s"
                )
        elif self.depth != 0:
            raise RepositoryHandoffContractError(
                "non-shallow origin must use depth 0"
            )
        if self.promisor and not self.filter_spec:
            raise RepositoryHandoffContractError(
                "promisor remotes require a filter_spec"
            )
        _reject_forbidden_keys(self.to_dict(), name="origin and shallow bounds")
        _require_record_bound(
            self, artifact_name="origin and shallow bounds", bounds=self.bounds
        )

    @property
    def origin_bounds_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "origin_url": self.origin_url,
                "origin_name": self.origin_name,
                "shallow": self.shallow,
                "depth": self.depth,
                "shallow_since_epoch_s": self.shallow_since_epoch_s,
                "filter_spec": self.filter_spec,
                "promisor": self.promisor,
                "unshallow_required": self.unshallow_required,
                "object_count": self.object_count,
                "object_bytes": self.object_bytes,
                "max_objects": self.max_objects,
                "max_object_bytes": self.max_object_bytes,
                "unbounded_objects": False,
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OriginAndShallowBounds":
        _schema_and_version(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="origin and shallow bounds",
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "origin_url",
                    "origin_name",
                    "shallow",
                    "depth",
                    "shallow_since_epoch_s",
                    "filter_spec",
                    "promisor",
                    "unshallow_required",
                    "object_count",
                    "object_bytes",
                    "max_objects",
                    "max_object_bytes",
                    "unbounded_objects",
                    "bounds",
                    "origin_bounds_id",
                }
            ),
            artifact_name="origin and shallow bounds",
        )
        defaults = cls()
        result = cls(
            origin_url=payload.get("origin_url", ""),
            origin_name=payload.get("origin_name", "origin"),
            shallow=payload.get("shallow", False),
            depth=payload.get("depth", 0),
            shallow_since_epoch_s=payload.get("shallow_since_epoch_s", 0),
            filter_spec=payload.get("filter_spec", ""),
            promisor=payload.get("promisor", False),
            unshallow_required=payload.get("unshallow_required", False),
            object_count=payload.get("object_count", 0),
            object_bytes=payload.get("object_bytes", 0),
            max_objects=payload.get("max_objects", defaults.max_objects),
            max_object_bytes=payload.get(
                "max_object_bytes", defaults.max_object_bytes
            ),
            unbounded_objects=payload.get("unbounded_objects", False),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "origin_bounds_id"),
            artifact_name="origin and shallow bounds",
        )
        return result


def _coerce_origin(value: Any) -> OriginAndShallowBounds:
    if value is None:
        return OriginAndShallowBounds()
    if isinstance(value, OriginAndShallowBounds):
        return value
    if isinstance(value, Mapping):
        return OriginAndShallowBounds.from_dict(value)
    raise RepositoryHandoffContractError(
        "origin_and_shallow must be an OriginAndShallowBounds object"
    )


@dataclass(frozen=True)
class RepositoryHandoffRefusal(_RepositoryHandoffCanonicalContract):
    """Typed refusal for an unsafe overlay.  Not reconstructed state."""

    SCHEMA: ClassVar[str] = REPOSITORY_HANDOFF_REFUSAL_SCHEMA
    INTERFACE: ClassVar[str] = REPOSITORY_HANDOFF_REFUSAL_INTERFACE

    reason: RefusalReason
    message: str
    overlay_id: str = ""
    created_at_ms: int = 0
    bounds: RepositoryHandoffBounds = RepositoryHandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(self, "reason", _enum(self.reason, RefusalReason, "reason"))
        object.__setattr__(
            self,
            "message",
            _text(self.message, "message", max_bytes=ABSOLUTE_MAX_REASON_BYTES),
        )
        object.__setattr__(
            self,
            "overlay_id",
            _content_ref(self.overlay_id, "overlay_id", required=False),
        )
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        _reject_forbidden_keys(self.to_dict(), name="repository handoff refusal")
        _require_record_bound(
            self, artifact_name="repository handoff refusal", bounds=self.bounds
        )

    @property
    def refusal_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "reason": self.reason.value,
                "message": self.message,
                "overlay_id": self.overlay_id,
                "created_at_ms": self.created_at_ms,
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepositoryHandoffRefusal":
        _schema_and_version(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="repository handoff refusal",
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "reason",
                    "message",
                    "overlay_id",
                    "created_at_ms",
                    "bounds",
                    "refusal_id",
                }
            ),
            artifact_name="repository handoff refusal",
        )
        result = cls(
            reason=payload.get("reason"),
            message=payload.get("message", ""),
            overlay_id=payload.get("overlay_id", ""),
            created_at_ms=payload.get("created_at_ms", 0),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "refusal_id"),
            artifact_name="repository handoff refusal",
        )
        return result


def refusal_from_error(
    error: RepositoryHandoffRefusalError,
    *,
    overlay_id: str = "",
    created_at_ms: int = 0,
) -> RepositoryHandoffRefusal:
    """Project a typed refusal exception into a content-addressed record."""

    return RepositoryHandoffRefusal(
        reason=error.reason,
        message=str(error)[:ABSOLUTE_MAX_REASON_BYTES],
        overlay_id=overlay_id,
        created_at_ms=created_at_ms,
    )


@dataclass(frozen=True)
class RepositoryHandoffRequest(_RepositoryHandoffCanonicalContract):
    """Complete repository handoff request binding overlay and related records."""

    SCHEMA: ClassVar[str] = REPOSITORY_HANDOFF_REQUEST_SCHEMA
    INTERFACE: ClassVar[str] = REPOSITORY_HANDOFF_REQUEST_INTERFACE

    overlay: RepositoryOverlay
    caller_principal_id: str
    idempotency_key: str
    hook_policy: HookPolicy = HookPolicy()
    origin_and_shallow: OriginAndShallowBounds = OriginAndShallowBounds()
    sparse_checkout: SparseCheckoutRecord = SparseCheckoutRecord()
    submodules: tuple[SubmoduleRecord, ...] = ()
    nested_repos: tuple[NestedRepoRecord, ...] = ()
    lfs_pointers: tuple[LfsPointerRecord, ...] = ()
    attributes_and_modes: tuple[AttributeAndModeRecord, ...] = ()
    session_id: str = ""
    object_bundle_id: str = ""
    mode: RepositoryHandoffMode = RepositoryHandoffMode.PREVIEW
    created_at_ms: int = 0
    bounds: RepositoryHandoffBounds = RepositoryHandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(self, "overlay", _coerce_overlay(self.overlay))
        object.__setattr__(self, "hook_policy", _coerce_hooks(self.hook_policy))
        object.__setattr__(
            self, "origin_and_shallow", _coerce_origin(self.origin_and_shallow)
        )
        object.__setattr__(
            self, "sparse_checkout", _coerce_sparse(self.sparse_checkout)
        )
        object.__setattr__(
            self,
            "caller_principal_id",
            _text(
                self.caller_principal_id,
                "caller_principal_id",
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _text(
                self.idempotency_key,
                "idempotency_key",
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "submodules",
            _records(
                self.submodules,
                SubmoduleRecord,
                "submodules",
                max_items=self.bounds.max_submodules,
            ),
        )
        object.__setattr__(
            self,
            "nested_repos",
            _records(
                self.nested_repos,
                NestedRepoRecord,
                "nested_repos",
                max_items=self.bounds.max_nested_repos,
            ),
        )
        object.__setattr__(
            self,
            "lfs_pointers",
            _records(
                self.lfs_pointers,
                LfsPointerRecord,
                "lfs_pointers",
                max_items=self.bounds.max_lfs_pointers,
            ),
        )
        object.__setattr__(
            self,
            "attributes_and_modes",
            _records(
                self.attributes_and_modes,
                AttributeAndModeRecord,
                "attributes_and_modes",
                max_items=self.bounds.max_attributes,
            ),
        )
        object.__setattr__(
            self,
            "session_id",
            _content_ref(self.session_id, "session_id", required=False),
        )
        object.__setattr__(
            self,
            "object_bundle_id",
            _content_ref(self.object_bundle_id, "object_bundle_id", required=False),
        )
        object.__setattr__(
            self, "mode", _enum(self.mode, RepositoryHandoffMode, "mode")
        )
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        if (
            self.overlay.object_count > self.origin_and_shallow.max_objects
            or self.overlay.object_bytes > self.origin_and_shallow.max_object_bytes
        ):
            _refuse(
                RefusalReason.UNBOUNDED_OBJECTS,
                "overlay objects exceed origin-and-shallow bounds",
            )
        gitlink_paths = {
            item.path for item in self.overlay.index if item.kind is FileKind.GITLINK
        }
        submodule_paths = {item.path for item in self.submodules}
        if not gitlink_paths.issuperset(submodule_paths) and self.submodules:
            missing = sorted(submodule_paths.difference(gitlink_paths))
            if missing:
                raise RepositoryHandoffContractError(
                    "submodule paths must match index gitlinks"
                )
        nested_paths = {item.path for item in self.nested_repos}
        overlap = submodule_paths.intersection(nested_paths)
        if overlap:
            raise RepositoryHandoffContractError(
                "nested repos must not reuse submodule paths"
            )
        attribute_paths: dict[str, AttributeAndModeRecord] = {}
        for record in self.attributes_and_modes:
            if record.path in attribute_paths:
                raise RepositoryHandoffContractError(
                    "attributes_and_modes must not contain duplicate paths"
                )
            attribute_paths[record.path] = record
        worktree_by_path = {item.path: item for item in self.overlay.worktree}
        for path, record in attribute_paths.items():
            worktree = worktree_by_path.get(path)
            if worktree is not None and (
                worktree.mode != record.mode or worktree.kind is not record.kind
            ):
                raise RepositoryHandoffContractError(
                    "attribute mode must match overlay worktree mode"
                )
        lfs_paths: set[str] = set()
        for record in self.lfs_pointers:
            if record.path in lfs_paths:
                raise RepositoryHandoffContractError(
                    "lfs_pointers must not contain duplicate paths"
                )
            lfs_paths.add(record.path)
        if not self.hook_policy.import_hooks_disabled or self.hook_policy.hooks_enabled:
            _refuse(
                RefusalReason.ENABLED_HOOKS,
                "hooks must be disabled on import",
            )
        _distinct_identities(
            (
                ("overlay", self.overlay.overlay_id),
                ("hook_policy", self.hook_policy.hook_policy_id),
                ("origin_and_shallow", self.origin_and_shallow.origin_bounds_id),
                ("sparse_checkout", self.sparse_checkout.sparse_checkout_id),
                ("session", self.session_id),
                ("object_bundle", self.object_bundle_id),
                ("caller", self.caller_principal_id),
            )
        )
        _reject_forbidden_keys(self.to_dict(), name="repository handoff request")
        _require_record_bound(
            self,
            artifact_name="repository handoff request",
            bounds=self.bounds,
            serialized=True,
        )

    @property
    def request_id(self) -> str:
        return self.content_id

    @property
    def overlay_id(self) -> str:
        return self.overlay.overlay_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "overlay": self.overlay.to_dict(),
                "hook_policy": self.hook_policy.to_dict(),
                "origin_and_shallow": self.origin_and_shallow.to_dict(),
                "sparse_checkout": self.sparse_checkout.to_dict(),
                "submodules": [item.to_dict() for item in self.submodules],
                "nested_repos": [item.to_dict() for item in self.nested_repos],
                "lfs_pointers": [item.to_dict() for item in self.lfs_pointers],
                "attributes_and_modes": [
                    item.to_dict() for item in self.attributes_and_modes
                ],
                "caller_principal_id": self.caller_principal_id,
                "idempotency_key": self.idempotency_key,
                "session_id": self.session_id,
                "object_bundle_id": self.object_bundle_id,
                "mode": self.mode.value,
                "created_at_ms": self.created_at_ms,
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepositoryHandoffRequest":
        _schema_and_version(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="repository handoff request",
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "overlay",
                    "hook_policy",
                    "origin_and_shallow",
                    "sparse_checkout",
                    "submodules",
                    "nested_repos",
                    "lfs_pointers",
                    "attributes_and_modes",
                    "caller_principal_id",
                    "idempotency_key",
                    "session_id",
                    "object_bundle_id",
                    "mode",
                    "created_at_ms",
                    "bounds",
                    "request_id",
                    "provider_id",
                    "provider_route",
                    "host_path",
                }
            ),
            artifact_name="repository handoff request",
        )
        if payload.get("provider_id") not in (None, "") or payload.get(
            "provider_route"
        ) not in (None, ""):
            raise RepositoryHandoffContractError(
                "imported history cannot select a provider on a repository handoff request"
            )
        if payload.get("host_path") not in (None, ""):
            _refuse(
                RefusalReason.HOST_PATH_ORIGIN,
                "repository handoff request must not accept a host filesystem path",
            )
        result = cls(
            overlay=payload.get("overlay"),
            caller_principal_id=payload.get("caller_principal_id", ""),
            idempotency_key=payload.get("idempotency_key", ""),
            hook_policy=payload.get("hook_policy"),
            origin_and_shallow=payload.get("origin_and_shallow"),
            sparse_checkout=payload.get("sparse_checkout"),
            submodules=payload.get("submodules", ()),
            nested_repos=payload.get("nested_repos", ()),
            lfs_pointers=payload.get("lfs_pointers", ()),
            attributes_and_modes=payload.get("attributes_and_modes", ()),
            session_id=payload.get("session_id", ""),
            object_bundle_id=payload.get("object_bundle_id", ""),
            mode=payload.get("mode", RepositoryHandoffMode.PREVIEW),
            created_at_ms=payload.get("created_at_ms", 0),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "request_id"),
            artifact_name="repository handoff request",
        )
        return result


_RECORD_DECODERS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        REPOSITORY_HANDOFF_BOUNDS_SCHEMA: RepositoryHandoffBounds.from_dict,
        REPOSITORY_HANDOFF_BOUNDS_INTERFACE: RepositoryHandoffBounds.from_dict,
        REPOSITORY_OVERLAY_SCHEMA: RepositoryOverlay.from_dict,
        REPOSITORY_OVERLAY_INTERFACE: RepositoryOverlay.from_dict,
        SUBMODULE_RECORD_SCHEMA: SubmoduleRecord.from_dict,
        SUBMODULE_RECORD_INTERFACE: SubmoduleRecord.from_dict,
        NESTED_REPO_RECORD_SCHEMA: NestedRepoRecord.from_dict,
        NESTED_REPO_RECORD_INTERFACE: NestedRepoRecord.from_dict,
        LFS_POINTER_RECORD_SCHEMA: LfsPointerRecord.from_dict,
        LFS_POINTER_RECORD_INTERFACE: LfsPointerRecord.from_dict,
        SPARSE_CHECKOUT_RECORD_SCHEMA: SparseCheckoutRecord.from_dict,
        SPARSE_CHECKOUT_RECORD_INTERFACE: SparseCheckoutRecord.from_dict,
        HOOK_POLICY_SCHEMA: HookPolicy.from_dict,
        HOOK_POLICY_INTERFACE: HookPolicy.from_dict,
        ATTRIBUTE_AND_MODE_RECORD_SCHEMA: AttributeAndModeRecord.from_dict,
        ATTRIBUTE_AND_MODE_RECORD_INTERFACE: AttributeAndModeRecord.from_dict,
        ORIGIN_AND_SHALLOW_BOUNDS_SCHEMA: OriginAndShallowBounds.from_dict,
        ORIGIN_AND_SHALLOW_BOUNDS_INTERFACE: OriginAndShallowBounds.from_dict,
        REPOSITORY_HANDOFF_REFUSAL_SCHEMA: RepositoryHandoffRefusal.from_dict,
        REPOSITORY_HANDOFF_REFUSAL_INTERFACE: RepositoryHandoffRefusal.from_dict,
        REPOSITORY_HANDOFF_REQUEST_SCHEMA: RepositoryHandoffRequest.from_dict,
        REPOSITORY_HANDOFF_REQUEST_INTERFACE: RepositoryHandoffRequest.from_dict,
    }
)

RepositoryHandoffRecord: TypeAlias = (
    RepositoryHandoffBounds
    | RepositoryOverlay
    | SubmoduleRecord
    | NestedRepoRecord
    | LfsPointerRecord
    | SparseCheckoutRecord
    | HookPolicy
    | AttributeAndModeRecord
    | OriginAndShallowBounds
    | RepositoryHandoffRefusal
    | RepositoryHandoffRequest
)


def decode_repository_handoff_contract(
    payload: Mapping[str, Any] | RepositoryHandoffRecord,
) -> RepositoryHandoffRecord:
    """Decode any strictly versioned repository-handoff family record."""

    if isinstance(payload, CanonicalContract):
        return payload  # type: ignore[return-value]
    if not isinstance(payload, Mapping):
        raise RepositoryHandoffContractError(
            "repository handoff contract payload must be an object"
        )
    for key in (payload.get("schema"), payload.get("interface")):
        decoder = _RECORD_DECODERS.get(str(key) if key is not None else "")
        if decoder is not None:
            return decoder(payload)
    raise RepositoryHandoffVersionError("unsupported repository handoff contract schema")


def canonical_repository_handoff_json_bytes(value: Any) -> bytes:
    """Encode one repository-handoff value as canonical DAG-JSON UTF-8 bytes."""

    if isinstance(value, CanonicalContract):
        return value.canonical_bytes()
    return canonical_json_bytes(value)


__all__ = (
    "ABSOLUTE_MAX_OBJECTS",
    "ABSOLUTE_MAX_OBJECT_BYTES",
    "ABSOLUTE_MAX_RECORD_BYTES",
    "ATTRIBUTE_AND_MODE_RECORD_INTERFACE",
    "ATTRIBUTE_AND_MODE_RECORD_SCHEMA",
    "CONTRACT_VERSION",
    "HOOK_POLICY_INTERFACE",
    "HOOK_POLICY_SCHEMA",
    "LFS_POINTER_RECORD_INTERFACE",
    "LFS_POINTER_RECORD_SCHEMA",
    "LFS_POINTER_VERSION",
    "NESTED_REPO_RECORD_INTERFACE",
    "NESTED_REPO_RECORD_SCHEMA",
    "ORIGIN_AND_SHALLOW_BOUNDS_INTERFACE",
    "ORIGIN_AND_SHALLOW_BOUNDS_SCHEMA",
    "REPOSITORY_HANDOFF_BOUNDS_INTERFACE",
    "REPOSITORY_HANDOFF_BOUNDS_SCHEMA",
    "REPOSITORY_HANDOFF_CONTRACT_FAMILY",
    "REPOSITORY_HANDOFF_CONTRACT_VERSION",
    "REPOSITORY_HANDOFF_REFUSAL_INTERFACE",
    "REPOSITORY_HANDOFF_REFUSAL_SCHEMA",
    "REPOSITORY_HANDOFF_REQUEST_INTERFACE",
    "REPOSITORY_HANDOFF_REQUEST_SCHEMA",
    "REPOSITORY_OVERLAY_INTERFACE",
    "REPOSITORY_OVERLAY_SCHEMA",
    "SCHEMA_VERSION",
    "SPARSE_CHECKOUT_RECORD_INTERFACE",
    "SPARSE_CHECKOUT_RECORD_SCHEMA",
    "SUBMODULE_RECORD_INTERFACE",
    "SUBMODULE_RECORD_SCHEMA",
    "AttributeAndModeRecord",
    "FileKind",
    "HookPolicy",
    "IndexEntry",
    "LfsPointerRecord",
    "NestedGitDirKind",
    "NestedRepoRecord",
    "OriginAndShallowBounds",
    "RefEntry",
    "RefusalReason",
    "RepositoryHandoffBounds",
    "RepositoryHandoffBoundsError",
    "RepositoryHandoffContractError",
    "RepositoryHandoffIdentityError",
    "RepositoryHandoffMode",
    "RepositoryHandoffRecord",
    "RepositoryHandoffRefusal",
    "RepositoryHandoffRefusalError",
    "RepositoryHandoffRequest",
    "RepositoryHandoffVersionError",
    "RepositoryOverlay",
    "SparseCheckoutRecord",
    "SubmoduleIgnore",
    "SubmoduleRecord",
    "UntrackedEntry",
    "WorktreeEntry",
    "canonical_repository_handoff_json_bytes",
    "decode_repository_handoff_contract",
    "refusal_from_error",
)
