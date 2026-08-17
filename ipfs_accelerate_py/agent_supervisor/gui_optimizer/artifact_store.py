"""Host-owned content-addressed evidence CAS for VerifiedGuiOptimizer (VGO-054).

Interfaces owned by this module:

* ``GuiEvidenceArtifactStore@1`` — immutable screenshot, accessibility,
  trace, baseline, receipt, and manifest bytes
* ``GuiEvidenceArtifactManifest@1`` — closed CID inventory bound to exact
  reuse identities

This store is a narrow evidence CAS.  It is not a semantic index or proof
cache and cannot turn an old verification into current authority.  Bytes
resolve only through verified CIDs under a fixed host root.  Browser
content cannot select host paths.

Fail-closed invariants:

* get/put never accept a caller- or browser-selected filesystem path;
* stored bytes rehash to their CID before any return;
* reuse requires exact repository, component, scenario, extractor, and
  checker identities;
* a successful reuse is never current verification authority;
* corrupt, truncated, or path-escaping state rejects.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_datasets_py.logic.gui_optimizer.identity import (
    DOMAIN_ARTIFACT,
    GuiCanonicalIdentity,
    GuiIdentityError,
    canonical_identity,
    cid_v1,
    parse_cid_v1,
    sha256_digest,
)

from .authority import (
    AuthorityReasonCode,
    GuiAuthorityError,
    GuiHostBoundaryPolicy,
)

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

GUI_EVIDENCE_ARTIFACT_STORE_INTERFACE: Final[str] = "GuiEvidenceArtifactStore@1"
GUI_EVIDENCE_ARTIFACT_MANIFEST_INTERFACE: Final[str] = (
    "GuiEvidenceArtifactManifest@1"
)
GUI_EVIDENCE_ARTIFACT_STORE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
    "evidence-artifact-store@1"
)
GUI_EVIDENCE_ARTIFACT_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
    "evidence-artifact-manifest@1"
)
GUI_EVIDENCE_ARTIFACT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
    "evidence-artifact-record@1"
)
GUI_ARTIFACT_REUSE_GATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/artifact-reuse-gate@1"
)

DOMAIN_EVIDENCE_ARTIFACT: Final[str] = "gui.evidence-artifact"
DOMAIN_EVIDENCE_MANIFEST: Final[str] = "gui.evidence-artifact-manifest"
DOMAIN_REUSE_GATE: Final[str] = "gui.evidence-artifact-reuse-gate"

DEFAULT_HOST_ROOT_RELATIVE: Final[str] = (
    "data/agent_supervisor/verified_gui_optimizer/artifacts"
)
DEFAULT_MAX_ARTIFACT_BYTES: Final[int] = 32 * 1024 * 1024
ABSOLUTE_MAX_ARTIFACT_BYTES: Final[int] = 64 * 1024 * 1024

_CID_RE = re.compile(r"^b[a-z2-7]{50,80}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_URI_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:")
_COMMAND_META_RE = re.compile(r"[;&|`$<>\n]|\$\(|\)")

BROAD_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "/",
        "/dev",
        "/etc",
        "/home",
        "/media",
        "/mnt",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sys",
        "/tmp",
        "/usr",
        "/var",
        "/var/tmp",
    }
)

_PUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "binding",
        "bytes",
        "kind",
        "media_type",
        "payload",
        "text",
    }
)
_GET_KEYS: Final[frozenset[str]] = frozenset({"cid", "kind", "required_gate"})
_GATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "checker_id",
        "checker_version",
        "component_id",
        "extractor_id",
        "extractor_version",
        "repository_id",
        "repository_revision",
        "scenario_id",
    }
)
_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact_cids",
        "entries",
        "interface",
        "run_id",
        "schema_version",
    }
)
_FORBIDDEN_PATH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "browser_input",
        "command",
        "commands",
        "cwd",
        "file_path",
        "filesystem_path",
        "host_path",
        "path",
        "selected_host_paths",
        "working_directory",
    }
)

_KIND_MEDIA_TYPES: Final[Mapping[str, str]] = {
    "screenshot": "image/png",
    "accessibility": "application/json",
    "trace": "application/json",
    "baseline": "application/json",
    "receipt": "application/json",
    "manifest": "application/json",
    "checkpoint": "application/json",
    "journal_record": "application/json",
}


class GuiArtifactStoreError(GuiAuthorityError):
    """Malformed or unsafe evidence-store input.  Never grants reuse."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_artifact_store_input",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


class ArtifactKind(str, Enum):
    """Closed evidence kinds stored by ``GuiEvidenceArtifactStore@1``."""

    SCREENSHOT = "screenshot"
    ACCESSIBILITY = "accessibility"
    TRACE = "trace"
    BASELINE = "baseline"
    RECEIPT = "receipt"
    MANIFEST = "manifest"
    CHECKPOINT = "checkpoint"
    JOURNAL_RECORD = "journal_record"


class ArtifactStoreReasonCode(str, Enum):
    """Stable reason codes for the host-owned evidence CAS."""

    STORED = "stored"
    RESOLVED = "resolved"
    REHASH_MISMATCH = "artifact_rehash_mismatch"
    CID_MISMATCH = "cid_mismatch"
    MISSING_ARTIFACT = "missing_artifact"
    CORRUPT_ARTIFACT = "corrupt_artifact"
    TRUNCATED_ARTIFACT = "truncated_artifact"
    PATH_ESCAPE = "path_escape"
    BROWSER_PATH_FORBIDDEN = (
        AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value
    )
    COMMAND_STRING_FORBIDDEN = (
        AuthorityReasonCode.BROWSER_COMMAND_FORBIDDEN.value
    )
    BROAD_ROOT_FORBIDDEN = "broad_root_forbidden"
    HOST_ROOT_INVALID = "host_root_invalid"
    REUSE_GATE_MISMATCH = "reuse_gate_mismatch"
    REUSE_NOT_AUTHORITY = "reuse_not_current_authority"
    UNKNOWN_KIND = "unknown_artifact_kind"
    ARTIFACT_TOO_LARGE = "artifact_too_large"
    UNKNOWN_FIELD = AuthorityReasonCode.UNKNOWN_FIELD.value
    INVALID_COLLECTION_TYPE = (
        AuthorityReasonCode.INVALID_COLLECTION_TYPE.value
    )
    INVALID_ARTIFACT_STORE_INPUT = "invalid_artifact_store_input"
    PATH_ABSOLUTE_OR_TRAVERSAL = (
        AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    )


# ---------------------------------------------------------------------------
# Closed input helpers
# ---------------------------------------------------------------------------


def _exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise GuiArtifactStoreError(
            f"{name} must be a string",
            reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiArtifactStoreError(
            f"{name} must not contain NUL",
            reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            details={"field": name},
        )
    text = text_value.strip()
    if required and not text:
        raise GuiArtifactStoreError(
            f"{name} must not be empty",
            reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            details={"field": name},
        )
    return text


def _identifier(value: Any, name: str) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiArtifactStoreError(
            f"{name} must not contain NUL",
            reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            details={"field": name},
        )
    if text_value == "" or text_value != text_value.strip():
        raise GuiArtifactStoreError(
            f"{name} must be a canonical nonempty string identifier",
            reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            details={"field": name},
        )
    return text_value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GuiArtifactStoreError(
            f"{name} must be a JSON object",
            reason_code=ArtifactStoreReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    for key in value:
        if type(key) is not str:
            raise GuiArtifactStoreError(
                f"{name} keys must be strings",
                reason_code=ArtifactStoreReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": name, "key_type": type(key).__name__},
            )
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise GuiArtifactStoreError(
            f"{name} must be a JSON array",
            reason_code=ArtifactStoreReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _reject_unknown(
    payload: Mapping[str, Any], allowed: frozenset[str], noun: str
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise GuiArtifactStoreError(
            f"{noun} contains unknown fields: {unknown}",
            reason_code=ArtifactStoreReasonCode.UNKNOWN_FIELD.value,
            details={"noun": noun, "unknown_fields": unknown},
        )


def _reject_forbidden_path_keys(payload: Mapping[str, Any], noun: str) -> None:
    forbidden = sorted(set(payload) & set(_FORBIDDEN_PATH_KEYS))
    if forbidden:
        raise GuiArtifactStoreError(
            f"{noun} contains forbidden host-path fields: {forbidden}",
            reason_code=ArtifactStoreReasonCode.BROWSER_PATH_FORBIDDEN.value,
            details={"noun": noun, "forbidden_fields": forbidden},
        )


def _reject_present_null(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is None:
        raise GuiArtifactStoreError(
            f"{key} must not be null when present",
            reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            details={"field": key, "value_type": "NoneType"},
        )


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(_require_mapping(value, "details")))


def _looks_like_browser_or_command_path(value: str) -> str:
    lowered = value.strip().lower()
    if lowered.startswith(
        ("file:", "http:", "https:", "blob:", "data:", "about:")
    ):
        return ArtifactStoreReasonCode.BROWSER_PATH_FORBIDDEN.value
    if _URI_RE.match(value) and not value.startswith("/"):
        return ArtifactStoreReasonCode.BROWSER_PATH_FORBIDDEN.value
    if _WINDOWS_DRIVE_RE.match(value) or value.startswith("//"):
        return ArtifactStoreReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    if _COMMAND_META_RE.search(value):
        return ArtifactStoreReasonCode.COMMAND_STRING_FORBIDDEN.value
    return ""


def _as_kind(value: Any) -> ArtifactKind:
    if type(value) is ArtifactKind:
        return value
    text = _text(value, "kind")
    try:
        return ArtifactKind(text)
    except ValueError as exc:
        raise GuiArtifactStoreError(
            f"unknown artifact kind: {text}",
            reason_code=ArtifactStoreReasonCode.UNKNOWN_KIND.value,
            details={"kind": text},
        ) from exc


def _require_cid(value: Any, name: str = "cid") -> str:
    text = _identifier(value, name)
    if not _CID_RE.fullmatch(text):
        raise GuiArtifactStoreError(
            f"{name} must be a CIDv1 raw/sha2-256 base32 string",
            reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            details={"field": name, "value": text},
        )
    try:
        parsed = parse_cid_v1(text)
    except GuiIdentityError as exc:
        raise GuiArtifactStoreError(
            f"{name} is not a verified GUI CIDv1",
            reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            details={"field": name, "value": text},
        ) from exc
    if parsed["cid"] != text:
        raise GuiArtifactStoreError(
            f"{name} is not in canonical CID form",
            reason_code=ArtifactStoreReasonCode.CID_MISMATCH.value,
            details={"field": name, "value": text},
        )
    return text


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(payload)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".part", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    atomic_write_bytes(path, payload)


def resolve_host_root(value: Any, *, create: bool = True) -> Path:
    """Resolve a host-owned store root.  Browser and broad roots reject."""

    if isinstance(value, Path):
        rendered_like = os.fspath(value)
        if type(rendered_like) is not str:
            raise GuiArtifactStoreError(
                "host_root must be a string or host Path",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
                details={
                    "field": "host_root",
                    "value_type": type(value).__name__,
                },
            )
        raw = _text(rendered_like, "host_root")
    else:
        raw = _text(value, "host_root")
    injected = _looks_like_browser_or_command_path(raw)
    if injected:
        raise GuiArtifactStoreError(
            "host_root is not an explicit host directory",
            reason_code=injected,
            details={"field": "host_root", "value": raw},
        )
    candidate = Path(raw)
    if ".." in candidate.parts:
        raise GuiArtifactStoreError(
            "host_root must not contain parent-directory segments",
            reason_code=ArtifactStoreReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value,
            details={"field": "host_root", "value": raw},
        )
    try:
        resolved = candidate.expanduser().resolve(strict=False)
    except OSError as exc:
        raise GuiArtifactStoreError(
            "host_root could not be resolved",
            reason_code=ArtifactStoreReasonCode.HOST_ROOT_INVALID.value,
            details={"field": "host_root", "error": str(exc)},
        ) from exc
    rendered = str(resolved)
    if rendered in BROAD_ROOTS:
        raise GuiArtifactStoreError(
            "host_root must not be a broad filesystem root",
            reason_code=ArtifactStoreReasonCode.BROAD_ROOT_FORBIDDEN.value,
            details={"field": "host_root", "value": rendered},
        )
    if not resolved.is_absolute():
        raise GuiArtifactStoreError(
            "host_root must be an absolute host path",
            reason_code=ArtifactStoreReasonCode.HOST_ROOT_INVALID.value,
            details={"field": "host_root", "value": rendered},
        )
    if create:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise GuiArtifactStoreError(
                "host_root could not be created",
                reason_code=ArtifactStoreReasonCode.HOST_ROOT_INVALID.value,
                details={"field": "host_root", "error": str(exc)},
            ) from exc
    if not resolved.is_dir():
        raise GuiArtifactStoreError(
            "host_root must be an existing directory",
            reason_code=ArtifactStoreReasonCode.HOST_ROOT_INVALID.value,
            details={"field": "host_root", "value": rendered},
        )
    return resolved


def artifact_cid_for_bytes(payload: bytes) -> str:
    """Return the verified CIDv1 of exact artifact bytes."""

    if type(payload) is not bytes:
        raise GuiArtifactStoreError(
            "artifact payload must be raw bytes",
            reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            details={"value_type": type(payload).__name__},
        )
    return cid_v1(payload)


def artifact_digest_for_bytes(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise GuiArtifactStoreError(
            "artifact payload must be raw bytes",
            reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            details={"value_type": type(payload).__name__},
        )
    return sha256_digest(payload)


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactReuseGate:
    """Exact identities required before stored bytes may be reused."""

    repository_id: str
    repository_revision: str
    component_id: str
    scenario_id: str
    extractor_id: str
    extractor_version: str
    checker_id: str
    checker_version: str
    interface: str = "GuiArtifactReuseGate@1"
    schema_version: str = GUI_ARTIFACT_REUSE_GATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        revision = _identifier(self.repository_revision, "repository_revision")
        if not (
            _FULL_SHA_RE.fullmatch(revision) or _DIGEST_RE.fullmatch(revision)
        ):
            raise GuiArtifactStoreError(
                "repository_revision must be a 40-character SHA-1 or sha256 digest",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
                details={"field": "repository_revision"},
            )
        object.__setattr__(self, "repository_revision", revision)
        object.__setattr__(
            self, "component_id", _identifier(self.component_id, "component_id")
        )
        object.__setattr__(
            self, "scenario_id", _identifier(self.scenario_id, "scenario_id")
        )
        object.__setattr__(
            self, "extractor_id", _identifier(self.extractor_id, "extractor_id")
        )
        object.__setattr__(
            self,
            "extractor_version",
            _identifier(self.extractor_version, "extractor_version"),
        )
        object.__setattr__(
            self, "checker_id", _identifier(self.checker_id, "checker_id")
        )
        object.__setattr__(
            self,
            "checker_version",
            _identifier(self.checker_version, "checker_version"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "checker_id": self.checker_id,
            "checker_version": self.checker_version,
            "component_id": self.component_id,
            "extractor_id": self.extractor_id,
            "extractor_version": self.extractor_version,
            "interface": self.interface,
            "repository_id": self.repository_id,
            "repository_revision": self.repository_revision,
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
        }

    def matches(self, other: "ArtifactReuseGate") -> bool:
        return (
            self.repository_id == other.repository_id
            and self.repository_revision == other.repository_revision
            and self.component_id == other.component_id
            and self.scenario_id == other.scenario_id
            and self.extractor_id == other.extractor_id
            and self.extractor_version == other.extractor_version
            and self.checker_id == other.checker_id
            and self.checker_version == other.checker_version
        )

    def identity(self) -> GuiCanonicalIdentity:
        return canonical_identity(
            {
                "checker_id": self.checker_id,
                "checker_version": self.checker_version,
                "component_id": self.component_id,
                "extractor_id": self.extractor_id,
                "extractor_version": self.extractor_version,
                "repository_id": self.repository_id,
                "repository_revision": self.repository_revision,
                "scenario_id": self.scenario_id,
            },
            domain=DOMAIN_REUSE_GATE,
            schema_version=self.schema_version,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ArtifactReuseGate":
        payload = _require_mapping(raw, "binding")
        _reject_forbidden_path_keys(payload, "binding")
        allowed = set(_GATE_KEYS) | {"interface", "schema_version"}
        _reject_unknown(payload, frozenset(allowed), "binding")
        for key in _GATE_KEYS:
            if key not in payload:
                raise GuiArtifactStoreError(
                    f"binding.{key} is required",
                    reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
                    details={"field": key},
                )
            _reject_present_null(payload, key)
        return cls(
            repository_id=payload["repository_id"],
            repository_revision=payload["repository_revision"],
            component_id=payload["component_id"],
            scenario_id=payload["scenario_id"],
            extractor_id=payload["extractor_id"],
            extractor_version=payload["extractor_version"],
            checker_id=payload["checker_id"],
            checker_version=payload["checker_version"],
        )

    @classmethod
    def from_any(cls, value: Any) -> "ArtifactReuseGate":
        if type(value) is cls:
            return value
        if type(value) is dict:
            return cls.from_mapping(value)
        raise GuiArtifactStoreError(
            "binding must be an ArtifactReuseGate or JSON object",
            reason_code=ArtifactStoreReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"value_type": type(value).__name__},
        )


@dataclass(frozen=True)
class StoredArtifact:
    """Verified on-disk evidence record.  Never current authority."""

    cid: str
    digest: str
    kind: ArtifactKind
    media_type: str
    size_bytes: int
    binding: ArtifactReuseGate
    host_relative_path: str
    interface: str = GUI_EVIDENCE_ARTIFACT_STORE_INTERFACE
    schema_version: str = GUI_EVIDENCE_ARTIFACT_RECORD_SCHEMA
    is_current_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cid", _require_cid(self.cid))
        digest = _identifier(self.digest, "digest")
        if not _DIGEST_RE.fullmatch(digest):
            raise GuiArtifactStoreError(
                "digest must be sha256:<hex>",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
                details={"field": "digest"},
            )
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "kind", _as_kind(self.kind))
        object.__setattr__(
            self, "media_type", _identifier(self.media_type, "media_type")
        )
        if type(self.size_bytes) is not int or type(self.size_bytes) is bool:
            raise GuiArtifactStoreError(
                "size_bytes must be an integer",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
                details={"field": "size_bytes"},
            )
        if self.size_bytes < 1:
            raise GuiArtifactStoreError(
                "size_bytes must be positive",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
                details={"field": "size_bytes"},
            )
        if type(self.binding) is not ArtifactReuseGate:
            raise GuiArtifactStoreError(
                "binding must be an ArtifactReuseGate",
                reason_code=ArtifactStoreReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"value_type": type(self.binding).__name__},
            )
        object.__setattr__(
            self,
            "host_relative_path",
            _identifier(self.host_relative_path, "host_relative_path"),
        )
        object.__setattr__(self, "is_current_authority", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding": self.binding.to_dict(),
            "cid": self.cid,
            "digest": self.digest,
            "host_relative_path": self.host_relative_path,
            "interface": self.interface,
            "is_current_authority": False,
            "kind": self.kind.value,
            "media_type": self.media_type,
            "schema_version": self.schema_version,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ArtifactManifest:
    """Closed inventory of verified artifact CIDs for one run."""

    run_id: str
    entries: tuple[StoredArtifact, ...]
    cid: str
    digest: str
    interface: str = GUI_EVIDENCE_ARTIFACT_MANIFEST_INTERFACE
    schema_version: str = GUI_EVIDENCE_ARTIFACT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "cid", _require_cid(self.cid))
        digest = _identifier(self.digest, "digest")
        if not _DIGEST_RE.fullmatch(digest):
            raise GuiArtifactStoreError(
                "manifest digest must be sha256:<hex>",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            )
        object.__setattr__(self, "digest", digest)

    @property
    def artifact_cids(self) -> tuple[str, ...]:
        return tuple(entry.cid for entry in self.entries)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "artifact_cids": list(self.artifact_cids),
            "entries": [
                {
                    "binding": entry.binding.to_dict(),
                    "cid": entry.cid,
                    "digest": entry.digest,
                    "kind": entry.kind.value,
                    "media_type": entry.media_type,
                    "size_bytes": entry.size_bytes,
                }
                for entry in self.entries
            ],
            "interface": self.interface,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["cid"] = self.cid
        payload["digest"] = self.digest
        return payload


# ---------------------------------------------------------------------------
# GuiEvidenceArtifactStore@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiEvidenceArtifactStore:
    """Immutable host-owned CAS.  Interface: ``GuiEvidenceArtifactStore@1``."""

    host_root: Path
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
    host_boundary: GuiHostBoundaryPolicy = field(
        default_factory=GuiHostBoundaryPolicy
    )
    interface: str = GUI_EVIDENCE_ARTIFACT_STORE_INTERFACE
    schema: str = GUI_EVIDENCE_ARTIFACT_STORE_SCHEMA

    def __post_init__(self) -> None:
        root = resolve_host_root(self.host_root, create=True)
        object.__setattr__(self, "host_root", root)
        if (
            type(self.max_artifact_bytes) is not int
            or type(self.max_artifact_bytes) is bool
        ):
            raise GuiArtifactStoreError(
                "max_artifact_bytes must be an integer",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            )
        if (
            self.max_artifact_bytes < 1
            or self.max_artifact_bytes > ABSOLUTE_MAX_ARTIFACT_BYTES
        ):
            raise GuiArtifactStoreError(
                "max_artifact_bytes is outside the sealed bound",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
                details={"max_artifact_bytes": self.max_artifact_bytes},
            )
        if type(self.host_boundary) is not GuiHostBoundaryPolicy:
            raise GuiArtifactStoreError(
                "host_boundary must be a GuiHostBoundaryPolicy",
                reason_code=ArtifactStoreReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"value_type": type(self.host_boundary).__name__},
            )
        cas = root / "cas"
        cas.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))

    @property
    def cas_root(self) -> Path:
        return self.host_root / "cas"

    def put(
        self,
        payload: bytes | str | Mapping[str, Any],
        *,
        kind: ArtifactKind | str,
        binding: ArtifactReuseGate | Mapping[str, Any],
        media_type: str = "",
    ) -> StoredArtifact:
        """Store exact bytes and return their verified CID record."""

        typed_kind = _as_kind(kind)
        typed_binding = ArtifactReuseGate.from_any(binding)
        body = self._coerce_payload(payload)
        if len(body) < 1:
            raise GuiArtifactStoreError(
                "artifact payload must not be empty",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            )
        if len(body) > self.max_artifact_bytes:
            raise GuiArtifactStoreError(
                "artifact exceeds the sealed size bound",
                reason_code=ArtifactStoreReasonCode.ARTIFACT_TOO_LARGE.value,
                details={
                    "size_bytes": len(body),
                    "max_artifact_bytes": self.max_artifact_bytes,
                },
            )
        typed_media = (
            _identifier(media_type, "media_type")
            if media_type
            else _KIND_MEDIA_TYPES[typed_kind.value]
        )
        cid = artifact_cid_for_bytes(body)
        digest = artifact_digest_for_bytes(body)
        blob_path = self._blob_path(cid)
        self._confine(blob_path)
        if blob_path.exists():
            existing = self._read_verified_bytes(cid)
            if existing != body:
                raise GuiArtifactStoreError(
                    "existing CID payload does not rehash to the claimed bytes",
                    reason_code=ArtifactStoreReasonCode.REHASH_MISMATCH.value,
                    details={"cid": cid},
                )
        else:
            _atomic_write_bytes(blob_path, body)
            written = self._read_verified_bytes(cid)
            if written != body or artifact_cid_for_bytes(written) != cid:
                raise GuiArtifactStoreError(
                    "stored artifact failed post-write rehash",
                    reason_code=ArtifactStoreReasonCode.REHASH_MISMATCH.value,
                    details={"cid": cid},
                )
        record = StoredArtifact(
            cid=cid,
            digest=digest,
            kind=typed_kind,
            media_type=typed_media,
            size_bytes=len(body),
            binding=typed_binding,
            host_relative_path=self._relative_blob(cid),
        )
        self._write_record(record)
        return record

    def put_from_mapping(self, raw: Mapping[str, Any]) -> StoredArtifact:
        payload = _require_mapping(raw, "put")
        _reject_forbidden_path_keys(payload, "put")
        _reject_unknown(payload, _PUT_KEYS, "put")
        if "kind" not in payload or "binding" not in payload:
            raise GuiArtifactStoreError(
                "put.kind and put.binding are required",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            )
        body: Any
        present = [key for key in ("bytes", "payload", "text") if key in payload]
        if len(present) != 1:
            raise GuiArtifactStoreError(
                "put must include exactly one of bytes, payload, or text",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
                details={"present": present},
            )
        body = payload[present[0]]
        media = payload["media_type"] if "media_type" in payload else ""
        if "media_type" in payload:
            _reject_present_null(payload, "media_type")
        return self.put(
            body,
            kind=payload["kind"],
            binding=payload["binding"],
            media_type=media,
        )

    def get(
        self,
        cid: str,
        *,
        kind: ArtifactKind | str | None = None,
        required_gate: ArtifactReuseGate | Mapping[str, Any] | None = None,
    ) -> tuple[bytes, StoredArtifact]:
        """Resolve bytes only through a verified CID under the fixed root."""

        typed_cid = _require_cid(cid)
        body = self._read_verified_bytes(typed_cid)
        record = self._read_record(typed_cid, expected_size=len(body))
        if artifact_digest_for_bytes(body) != record.digest:
            raise GuiArtifactStoreError(
                "artifact digest does not rehash from stored bytes",
                reason_code=ArtifactStoreReasonCode.REHASH_MISMATCH.value,
                details={"cid": typed_cid},
            )
        if kind is not None and record.kind is not _as_kind(kind):
            raise GuiArtifactStoreError(
                "stored artifact kind does not match the requested kind",
                reason_code=ArtifactStoreReasonCode.UNKNOWN_KIND.value,
                details={
                    "cid": typed_cid,
                    "stored_kind": record.kind.value,
                    "requested_kind": _as_kind(kind).value,
                },
            )
        if required_gate is not None:
            gate = ArtifactReuseGate.from_any(required_gate)
            if not record.binding.matches(gate):
                raise GuiArtifactStoreError(
                    "artifact reuse gate does not match exactly",
                    reason_code=ArtifactStoreReasonCode.REUSE_GATE_MISMATCH.value,
                    details={
                        "cid": typed_cid,
                        "stored_gate": record.binding.to_dict(),
                        "required_gate": gate.to_dict(),
                    },
                )
        return body, record

    def get_from_mapping(
        self, raw: Mapping[str, Any]
    ) -> tuple[bytes, StoredArtifact]:
        payload = _require_mapping(raw, "get")
        _reject_forbidden_path_keys(payload, "get")
        _reject_unknown(payload, _GET_KEYS, "get")
        if "cid" not in payload:
            raise GuiArtifactStoreError(
                "get.cid is required",
                reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
            )
        _reject_present_null(payload, "cid")
        kind = payload["kind"] if "kind" in payload else None
        if "kind" in payload:
            _reject_present_null(payload, "kind")
        gate = payload["required_gate"] if "required_gate" in payload else None
        if "required_gate" in payload:
            _reject_present_null(payload, "required_gate")
        return self.get(payload["cid"], kind=kind, required_gate=gate)

    def rehash(self, cid: str) -> StoredArtifact:
        """Fail closed unless stored bytes still produce ``cid``."""

        _body, record = self.get(cid)
        return record

    def reuse(
        self,
        cid: str,
        required_gate: ArtifactReuseGate | Mapping[str, Any],
        *,
        kind: ArtifactKind | str | None = None,
    ) -> tuple[bytes, StoredArtifact]:
        """Return bytes only when every reuse identity matches exactly.

        A successful reuse is still not current verification authority.
        """

        body, record = self.get(cid, kind=kind, required_gate=required_gate)
        if record.is_current_authority:
            raise GuiArtifactStoreError(
                "evidence CAS cannot grant current verification authority",
                reason_code=ArtifactStoreReasonCode.REUSE_NOT_AUTHORITY.value,
                details={"cid": record.cid},
            )
        return body, record

    def put_manifest(
        self,
        *,
        run_id: str,
        artifacts: Sequence[StoredArtifact | Mapping[str, Any] | str],
        binding: ArtifactReuseGate | Mapping[str, Any],
    ) -> ArtifactManifest:
        """Persist a closed CID inventory after rehashing every entry."""

        typed_run = _identifier(run_id, "run_id")
        typed_binding = ArtifactReuseGate.from_any(binding)
        entries: list[StoredArtifact] = []
        seen: set[str] = set()
        for item in artifacts:
            if type(item) is StoredArtifact:
                record = self.rehash(item.cid)
            elif type(item) is str:
                record = self.rehash(item)
            elif type(item) is dict:
                mapping = _require_mapping(item, "manifest entry")
                if "cid" not in mapping:
                    raise GuiArtifactStoreError(
                        "manifest entry.cid is required",
                        reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
                    )
                record = self.rehash(mapping["cid"])
            else:
                raise GuiArtifactStoreError(
                    "manifest entries must be CIDs or stored artifacts",
                    reason_code=ArtifactStoreReasonCode.INVALID_COLLECTION_TYPE.value,
                    details={"value_type": type(item).__name__},
                )
            if record.cid in seen:
                continue
            seen.add(record.cid)
            entries.append(record)
        entries = sorted(entries, key=lambda item: item.cid)
        identity_payload = {
            "artifact_cids": [entry.cid for entry in entries],
            "entries": [
                {
                    "binding": entry.binding.to_dict(),
                    "cid": entry.cid,
                    "digest": entry.digest,
                    "kind": entry.kind.value,
                    "media_type": entry.media_type,
                    "size_bytes": entry.size_bytes,
                }
                for entry in entries
            ],
            "interface": GUI_EVIDENCE_ARTIFACT_MANIFEST_INTERFACE,
            "run_id": typed_run,
            "schema_version": GUI_EVIDENCE_ARTIFACT_MANIFEST_SCHEMA,
        }
        identity = canonical_identity(
            identity_payload,
            domain=DOMAIN_EVIDENCE_MANIFEST,
            schema_version=GUI_EVIDENCE_ARTIFACT_MANIFEST_SCHEMA,
        )
        stored = self.put(
            identity_payload,
            kind=ArtifactKind.MANIFEST,
            binding=typed_binding,
            media_type="application/json",
        )
        return ArtifactManifest(
            run_id=typed_run,
            entries=tuple(entries),
            cid=stored.cid,
            digest=identity.digest,
        )

    def get_manifest(
        self,
        cid: str,
        *,
        required_gate: ArtifactReuseGate | Mapping[str, Any] | None = None,
    ) -> ArtifactManifest:
        body, record = self.get(
            cid, kind=ArtifactKind.MANIFEST, required_gate=required_gate
        )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuiArtifactStoreError(
                "manifest bytes are not canonical JSON",
                reason_code=ArtifactStoreReasonCode.CORRUPT_ARTIFACT.value,
                details={"cid": cid},
            ) from exc
        mapping = _require_mapping(payload, "manifest")
        _reject_unknown(
            mapping,
            _MANIFEST_KEYS,
            "manifest",
        )
        for key in ("run_id", "artifact_cids", "entries"):
            if key not in mapping:
                raise GuiArtifactStoreError(
                    f"manifest.{key} is required",
                    reason_code=ArtifactStoreReasonCode.CORRUPT_ARTIFACT.value,
                    details={"field": key},
                )
        raw_entries = _require_list(mapping["entries"], "entries")
        entry_cids: list[str] = []
        for item in raw_entries:
            if type(item) is not dict:
                raise GuiArtifactStoreError(
                    "manifest entries must be JSON objects",
                    reason_code=ArtifactStoreReasonCode.CORRUPT_ARTIFACT.value,
                    details={"cid": cid},
                )
            entry_cids.append(_require_cid(item["cid"], "entry.cid"))
        claimed = [
            _require_cid(item, "artifact_cids[]")
            for item in _require_list(mapping["artifact_cids"], "artifact_cids")
        ]
        if entry_cids != claimed:
            raise GuiArtifactStoreError(
                "manifest artifact_cids do not match entries",
                reason_code=ArtifactStoreReasonCode.CORRUPT_ARTIFACT.value,
                details={"cid": cid},
            )
        entries = tuple(self.rehash(item) for item in claimed)
        reconstructed = [
            {
                "binding": entry.binding.to_dict(),
                "cid": entry.cid,
                "digest": entry.digest,
                "kind": entry.kind.value,
                "media_type": entry.media_type,
                "size_bytes": entry.size_bytes,
            }
            for entry in entries
        ]
        if reconstructed != raw_entries:
            raise GuiArtifactStoreError(
                "manifest entries do not rehash to stored artifact metadata",
                reason_code=ArtifactStoreReasonCode.REHASH_MISMATCH.value,
                details={"cid": cid},
            )
        identity = canonical_identity(
            {
                "artifact_cids": list(claimed),
                "entries": reconstructed,
                "interface": mapping.get(
                    "interface", GUI_EVIDENCE_ARTIFACT_MANIFEST_INTERFACE
                ),
                "run_id": mapping["run_id"],
                "schema_version": mapping.get(
                    "schema_version", GUI_EVIDENCE_ARTIFACT_MANIFEST_SCHEMA
                ),
            },
            domain=DOMAIN_EVIDENCE_MANIFEST,
            schema_version=GUI_EVIDENCE_ARTIFACT_MANIFEST_SCHEMA,
        )
        return ArtifactManifest(
            run_id=_identifier(mapping["run_id"], "run_id"),
            entries=entries,
            cid=record.cid,
            digest=identity.digest,
        )

    def host_path_for_cid(self, cid: str) -> Path:
        """Return the confined host path derived only from ``cid``."""

        return self._confine(self._blob_path(_require_cid(cid)))

    def is_current_authority(self, cid: str) -> bool:
        """Evidence CAS never elevates stored bytes into current authority."""

        self.rehash(cid)
        return False

    def _coerce_payload(self, payload: Any) -> bytes:
        if type(payload) is bytes:
            return payload
        if type(payload) is str:
            text = _exact_str(payload, "text")
            if "\x00" in text:
                raise GuiArtifactStoreError(
                    "text payload must not contain NUL",
                    reason_code=ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value,
                )
            return text.encode("utf-8")
        if type(payload) is dict:
            mapping = _require_mapping(payload, "payload")
            _reject_forbidden_path_keys(mapping, "payload")
            return _canonical_json_bytes(mapping)
        raise GuiArtifactStoreError(
            "payload must be bytes, text, or a JSON object",
            reason_code=ArtifactStoreReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"value_type": type(payload).__name__},
        )

    def _relative_blob(self, cid: str) -> str:
        return f"cas/{cid[1:3]}/{cid}.bin"

    def _blob_path(self, cid: str) -> Path:
        return self.host_root / self._relative_blob(cid)

    def _record_path(self, cid: str) -> Path:
        return self.host_root / "cas" / cid[1:3] / f"{cid}.meta.json"

    def _confine(self, path: Path) -> Path:
        try:
            resolved = path.resolve(strict=False)
        except OSError as exc:
            raise GuiArtifactStoreError(
                "artifact path could not be resolved",
                reason_code=ArtifactStoreReasonCode.PATH_ESCAPE.value,
                details={"path": str(path), "error": str(exc)},
            ) from exc
        root = self.host_root.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise GuiArtifactStoreError(
                "artifact path escapes the fixed host root",
                reason_code=ArtifactStoreReasonCode.PATH_ESCAPE.value,
                details={"path": str(resolved), "host_root": str(root)},
            ) from exc
        if ".." in path.parts:
            raise GuiArtifactStoreError(
                "artifact path must not contain parent-directory segments",
                reason_code=ArtifactStoreReasonCode.PATH_ESCAPE.value,
                details={"path": str(path)},
            )
        return resolved

    def _read_verified_bytes(self, cid: str) -> bytes:
        path = self._confine(self._blob_path(cid))
        if not path.is_file():
            raise GuiArtifactStoreError(
                "artifact CID is not present under the fixed host root",
                reason_code=ArtifactStoreReasonCode.MISSING_ARTIFACT.value,
                details={"cid": cid},
            )
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise GuiArtifactStoreError(
                "artifact bytes could not be read",
                reason_code=ArtifactStoreReasonCode.CORRUPT_ARTIFACT.value,
                details={"cid": cid, "error": str(exc)},
            ) from exc
        if not body:
            raise GuiArtifactStoreError(
                "artifact bytes are truncated",
                reason_code=ArtifactStoreReasonCode.TRUNCATED_ARTIFACT.value,
                details={"cid": cid},
            )
        recomputed = artifact_cid_for_bytes(body)
        if recomputed != cid:
            raise GuiArtifactStoreError(
                "stored artifact bytes do not rehash to the requested CID",
                reason_code=ArtifactStoreReasonCode.REHASH_MISMATCH.value,
                details={"cid": cid, "recomputed_cid": recomputed},
            )
        return body

    def _write_record(self, record: StoredArtifact) -> None:
        path = self._confine(self._record_path(record.cid))
        _atomic_write_bytes(path, _canonical_json_bytes(record.to_dict()))

    def _read_record(self, cid: str, *, expected_size: int) -> StoredArtifact:
        path = self._confine(self._record_path(cid))
        if not path.is_file():
            raise GuiArtifactStoreError(
                "artifact metadata is missing",
                reason_code=ArtifactStoreReasonCode.CORRUPT_ARTIFACT.value,
                details={"cid": cid},
            )
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuiArtifactStoreError(
                "artifact metadata is corrupt",
                reason_code=ArtifactStoreReasonCode.CORRUPT_ARTIFACT.value,
                details={"cid": cid},
            ) from exc
        mapping = _require_mapping(payload, "artifact record")
        if mapping.get("cid") != cid:
            raise GuiArtifactStoreError(
                "artifact metadata CID does not match the locator",
                reason_code=ArtifactStoreReasonCode.CID_MISMATCH.value,
                details={"cid": cid, "recorded_cid": mapping.get("cid")},
            )
        if mapping.get("size_bytes") != expected_size:
            raise GuiArtifactStoreError(
                "artifact metadata size does not match stored bytes",
                reason_code=ArtifactStoreReasonCode.CORRUPT_ARTIFACT.value,
                details={
                    "cid": cid,
                    "recorded_size": mapping.get("size_bytes"),
                    "actual_size": expected_size,
                },
            )
        try:
            binding = ArtifactReuseGate.from_any(mapping["binding"])
            return StoredArtifact(
                cid=mapping["cid"],
                digest=mapping["digest"],
                kind=mapping["kind"],
                media_type=mapping["media_type"],
                size_bytes=mapping["size_bytes"],
                binding=binding,
                host_relative_path=mapping["host_relative_path"],
            )
        except (KeyError, GuiArtifactStoreError) as exc:
            raise GuiArtifactStoreError(
                "artifact metadata failed closed validation",
                reason_code=ArtifactStoreReasonCode.CORRUPT_ARTIFACT.value,
                details={"cid": cid},
            ) from exc


def default_evidence_artifact_store(
    host_root: Path | str,
    *,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> GuiEvidenceArtifactStore:
    """Construct a host-owned evidence CAS under ``host_root``."""

    return GuiEvidenceArtifactStore(
        host_root=Path(host_root),
        max_artifact_bytes=max_artifact_bytes,
    )


__all__ = (
    "ABSOLUTE_MAX_ARTIFACT_BYTES",
    "ArtifactKind",
    "ArtifactManifest",
    "ArtifactReuseGate",
    "ArtifactStoreReasonCode",
    "BROAD_ROOTS",
    "DEFAULT_HOST_ROOT_RELATIVE",
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "DOMAIN_ARTIFACT",
    "DOMAIN_EVIDENCE_ARTIFACT",
    "DOMAIN_EVIDENCE_MANIFEST",
    "DOMAIN_REUSE_GATE",
    "GUI_ARTIFACT_REUSE_GATE_SCHEMA",
    "GUI_EVIDENCE_ARTIFACT_MANIFEST_INTERFACE",
    "GUI_EVIDENCE_ARTIFACT_MANIFEST_SCHEMA",
    "GUI_EVIDENCE_ARTIFACT_RECORD_SCHEMA",
    "GUI_EVIDENCE_ARTIFACT_STORE_INTERFACE",
    "GUI_EVIDENCE_ARTIFACT_STORE_SCHEMA",
    "GuiArtifactStoreError",
    "GuiEvidenceArtifactStore",
    "StoredArtifact",
    "artifact_cid_for_bytes",
    "artifact_digest_for_bytes",
    "atomic_write_bytes",
    "canonical_json_bytes",
    "default_evidence_artifact_store",
    "resolve_host_root",
)
