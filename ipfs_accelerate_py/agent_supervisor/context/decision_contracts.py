"""Canonical, provider-free contracts for proof-directed decisions.

The records in this module are deliberately only data and validation.  They do
not resolve providers, inspect a checkout, grant authority, or dispatch tools.
Callers must compute and pin every decision-changing input before constructing
a :class:`DecisionRequest`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import posixpath
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
)


DECISION_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = DECISION_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = DECISION_CONTRACT_VERSION

PINNED_ARTIFACT_REF_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/pinned-artifact-ref@1"
)
SEMANTIC_ROOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-root@1"
)
DECISION_TARGET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/decision-target@1"
)
ACTION_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/action-envelope@1"
)
EFFECT_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/effect-envelope@1"
)
APPLICABILITY_FACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/applicability-fact@1"
)
CAPABILITY_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/capability-envelope@1"
)
DECISION_BUDGET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/decision-budget@1"
)
AUTHORITY_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/authority-envelope@1"
)
DECISION_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/decision-request@1"
)

ABSOLUTE_MAX_DECISION_BYTES: Final[int] = 1_048_576
ABSOLUTE_MAX_ARTIFACT_BYTES: Final[int] = 64 * 1024 * 1024
ABSOLUTE_MAX_ITEMS: Final[int] = 4_096
ABSOLUTE_MAX_DEPTH: Final[int] = 32
ABSOLUTE_MAX_TEXT_BYTES: Final[int] = 65_536
ABSOLUTE_MAX_ACTIONS: Final[int] = 64
ABSOLUTE_MAX_EFFECTS: Final[int] = 128
ABSOLUTE_MAX_FACTS: Final[int] = 256
ABSOLUTE_MAX_CAPABILITIES: Final[int] = 256
ABSOLUTE_MAX_TARGETS: Final[int] = 256
ABSOLUTE_MAX_PATHS: Final[int] = 1_024
ABSOLUTE_MAX_LATENCY_MS: Final[int] = 86_400_000

_SHA256 = "sha256:"
_CIDV1_DAG_JSON_SHA256_PREFIX = b"\x01\xa9\x02\x12\x20"


class DecisionContractError(ContractValidationError):
    """Base exception for malformed decision contracts."""


class DecisionBoundsError(DecisionContractError):
    """A decision value exceeds a declared hard bound."""


class DecisionIdentityError(DecisionContractError):
    """A claimed CID, digest, or contract identity is inconsistent."""


class MissingSemanticRootError(DecisionContractError):
    """A mandatory semantic root is absent."""


class DuplicateReferenceError(DecisionContractError):
    """A request contains duplicate or conflicting pinned references."""


class UnknownAuthorityError(DecisionContractError):
    """An authority value is outside the closed vocabulary."""


class DecisionPathEscapeError(DecisionContractError):
    """An absolute root or repository-relative path can escape its scope."""


class DecisionBindingError(DecisionContractError):
    """Two otherwise valid envelopes have inconsistent decision bindings."""


class NonCanonicalDecisionError(DecisionContractError):
    """Serialized input changes when decoded and canonically re-encoded."""


class DecisionKind(str, Enum):
    ANALYZE = "analyze"
    PLAN = "plan"
    AUTHORIZE = "authorize"
    EXECUTE = "execute"
    VALIDATE = "validate"
    COMMIT = "commit"
    MERGE = "merge"
    COMPLETE = "complete"


class DecisionStage(str, Enum):
    ANALYSIS = "analysis"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    COMMIT = "commit"
    MERGE = "merge"
    COMPLETION = "completion"


class DecisionAuthority(str, Enum):
    READ = "read"
    PROPOSAL = "proposal"
    MUTATION = "mutation"

    @property
    def rank(self) -> int:
        return {
            DecisionAuthority.READ: 0,
            DecisionAuthority.PROPOSAL: 1,
            DecisionAuthority.MUTATION: 2,
        }[self]

    def allows(self, other: "DecisionAuthority | str") -> bool:
        return self.rank >= _authority(other).rank


class ReferenceAuthority(str, Enum):
    """Authority asserted by the producer of a pinned semantic input."""

    AUTHORITATIVE = "authoritative"
    VERIFIED = "verified"
    ADVISORY = "advisory"
    UNTRUSTED = "untrusted"

    @property
    def usable_as_root(self) -> bool:
        return self in {
            ReferenceAuthority.AUTHORITATIVE,
            ReferenceAuthority.VERIFIED,
        }


class SemanticRootKind(str, Enum):
    REPOSITORY = "repository"
    DIRTY_WORKTREE = "dirty_worktree"
    INTENT_IR = "intent_ir"
    LEGAL_IR = "legal_ir"
    SECURITY_IR = "security_ir"
    PROGRAM = "program"
    AST = "program"  # compatibility spelling
    TOOL_CATALOG = "tool_catalog"
    POLICY = "policy"


MANDATORY_SEMANTIC_ROOT_KINDS: Final[frozenset[SemanticRootKind]] = frozenset(
    SemanticRootKind
)


class WorktreeCoverage(str, Enum):
    TRACKED = "tracked"
    MODIFIED = "modified"
    STAGED = "staged"
    DELETED = "deleted"
    UNTRACKED = "untracked"


REQUIRED_DIRTY_WORKTREE_COVERAGE: Final[frozenset[WorktreeCoverage]] = frozenset(
    WorktreeCoverage
)


class EffectKind(str, Enum):
    OBSERVE = "observe"
    PROPOSE = "propose"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    START_PROCESS = "start_process"
    STOP_PROCESS = "stop_process"
    COMMIT = "commit"
    MERGE = "merge"
    EMIT_RECEIPT = "emit_receipt"

    @property
    def authority(self) -> DecisionAuthority:
        if self in {EffectKind.OBSERVE, EffectKind.READ}:
            return DecisionAuthority.READ
        if self is EffectKind.PROPOSE:
            return DecisionAuthority.PROPOSAL
        return DecisionAuthority.MUTATION


class ApplicabilityFactKind(str, Enum):
    JURISDICTION = "jurisdiction"
    EFFECTIVE_TIME = "effective_time"
    ENVIRONMENT = "environment"
    CAPABILITY = "capability"
    MODEL = "model"
    TOOLCHAIN = "toolchain"
    PRINCIPAL = "principal"
    RESOURCE = "resource"
    OTHER = "other"


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(value))
    except (TypeError, ValueError) as exc:
        if kind is DecisionAuthority:
            raise UnknownAuthorityError(
                f"unknown authority {value!r}; expected read, proposal, or mutation"
            ) from exc
        allowed = ", ".join(sorted({str(item.value) for item in kind}))
        raise DecisionContractError(f"{name} must be one of: {allowed}") from exc


def _authority(value: Any) -> DecisionAuthority:
    return _enum(value, DecisionAuthority, "authority")


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = ABSOLUTE_MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise DecisionContractError(f"{name} must be a string")
    if value != value.strip():
        raise NonCanonicalDecisionError(
            f"{name} has leading or trailing whitespace"
        )
    if required and not value:
        raise DecisionContractError(f"{name} must not be empty")
    if "\x00" in value:
        raise DecisionContractError(f"{name} must not contain NUL")
    if len(value.encode("utf-8")) > maximum:
        raise DecisionBoundsError(f"{name} exceeds {maximum} UTF-8 bytes")
    return value


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    # Reject bool, floats (including NaN/inf), Decimal, and integer-like
    # strings.  A budget must never acquire host-language infinity semantics.
    if isinstance(value, bool) or not isinstance(value, int):
        raise DecisionContractError(f"{name} must be a finite integer")
    if value < minimum:
        raise DecisionContractError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise DecisionBoundsError(f"{name} exceeds its absolute limit")
    return value


def _optional_integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _integer(value, name, minimum=minimum, maximum=maximum)


def _absolute_root(value: Any, name: str) -> str:
    result = _text(value, name)
    if "\\" in result or not result.startswith("/"):
        raise DecisionPathEscapeError(f"{name} must be a canonical absolute path")
    if ".." in PurePosixPath(result).parts:
        raise DecisionPathEscapeError(f"{name} must not traverse a parent")
    normalized = posixpath.normpath(result)
    if normalized == "/":
        raise DecisionPathEscapeError(f"{name} must not be the filesystem root")
    if normalized != result:
        raise NonCanonicalDecisionError(f"{name} is not a canonical path")
    return result


def _relative_path(value: Any, name: str) -> str:
    result = _text(value, name)
    candidate = PurePosixPath(result)
    if (
        "\\" in result
        or candidate.is_absolute()
        or ".." in candidate.parts
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise DecisionPathEscapeError(f"{name} must be repository-relative")
    normalized = candidate.as_posix()
    if normalized in ("", ".") or normalized != result:
        raise NonCanonicalDecisionError(f"{name} is not a canonical path")
    return result


def _strings(
    values: Any,
    name: str,
    *,
    required: bool,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise DecisionContractError(f"{name} must be a sequence of strings")
    if len(values) > maximum:
        raise DecisionBoundsError(f"{name} exceeds its count bound")
    result = tuple(_text(item, name) for item in values)
    if required and not result:
        raise DecisionContractError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise DuplicateReferenceError(f"{name} contains duplicates")
    canonical = tuple(sorted(result))
    if result != canonical:
        raise NonCanonicalDecisionError(f"{name} must be canonically sorted")
    return result


_PATH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "path",
        "root",
        "cwd",
        "repository_path",
        "repository_root",
        "state_path",
        "state_root",
        "target_path",
        "artifact_path",
        "worktree_path",
        "worktree_root",
    }
)
_PATHS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "paths",
        "roots",
        "repository_paths",
        "repository_roots",
        "state_paths",
        "state_roots",
        "target_paths",
        "artifact_paths",
        "worktree_paths",
        "worktree_roots",
    }
)


def _freeze_value(
    value: Any,
    *,
    name: str,
    max_depth: int,
    max_items: int,
    max_text_bytes: int,
    check_paths: bool = True,
) -> Any:
    """Validate, path-check, bound, and deeply freeze a canonical JSON value."""

    seen = 0

    def visit(item: Any, depth: int, key_name: str = "") -> Any:
        nonlocal seen
        seen += 1
        if seen > max_items:
            raise DecisionBoundsError(f"{name} exceeds its item-count bound")
        if depth > max_depth:
            raise DecisionBoundsError(f"{name} exceeds its nesting-depth bound")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            raise DecisionContractError(
                f"{name} cannot contain finite or non-finite floating values"
            )
        if isinstance(item, str):
            result = _text(
                item, name, required=False, maximum=max_text_bytes
            )
            if check_paths and key_name in _PATH_KEYS:
                return _relative_path(result, key_name)
            return result
        if isinstance(item, Enum):
            return visit(item.value, depth, key_name)
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise DecisionContractError(f"{name} object keys must be strings")
            frozen: dict[str, Any] = {}
            previous = ""
            for key in sorted(item):
                normalized_key = _text(
                    key, f"{name} key", maximum=max_text_bytes
                )
                if normalized_key in frozen or (previous and normalized_key <= previous):
                    raise DuplicateReferenceError(
                        f"{name} contains duplicate or conflicting keys"
                    )
                previous = normalized_key
                raw = item[key]
                if check_paths and normalized_key in _PATHS_KEYS:
                    if isinstance(raw, str) or not isinstance(raw, Sequence):
                        raise DecisionContractError(
                            f"{normalized_key} must be a sequence"
                        )
                    paths = tuple(
                        _relative_path(member, normalized_key) for member in raw
                    )
                    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
                        raise NonCanonicalDecisionError(
                            f"{normalized_key} must be unique and sorted"
                        )
                    frozen[normalized_key] = paths
                else:
                    frozen[normalized_key] = visit(
                        raw, depth + 1, normalized_key
                    )
            return MappingProxyType(frozen)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return tuple(visit(member, depth + 1, key_name) for member in item)
        raise DecisionContractError(
            f"{name} contains unsupported value type {type(item).__name__}"
        )

    return visit(value, 0)


def _schema(payload: Mapping[str, Any], expected: str, noun: str) -> None:
    if not isinstance(payload, Mapping):
        raise DecisionContractError(f"{noun} must be an object")
    if payload.get("schema") != expected:
        raise DecisionContractError(
            f"{noun} requires exact schema {expected!r}"
        )
    if payload.get("contract_version") != DECISION_CONTRACT_VERSION:
        raise DecisionContractError(
            f"{noun} requires contract_version {DECISION_CONTRACT_VERSION}"
        )


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Iterable[str], noun: str
) -> None:
    unknown = set(payload).difference(allowed)
    if unknown:
        raise DecisionContractError(
            f"{noun} contains unsupported fields: {', '.join(sorted(unknown))}"
        )


def _required(payload: Mapping[str, Any], name: str, noun: str) -> Any:
    if name not in payload:
        raise DecisionContractError(f"{noun} is missing required field {name}")
    return payload[name]


def _identity(payload: Mapping[str, Any], actual: str, noun: str) -> None:
    claimed = payload.get("content_id")
    if claimed is not None and claimed != actual:
        raise DecisionIdentityError(f"{noun} identity does not match payload")


def _decode_json_object(payload: str, noun: str) -> Mapping[str, Any]:
    if not isinstance(payload, str):
        raise DecisionContractError(f"{noun} JSON must be text")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateReferenceError(
                    f"{noun} JSON contains duplicate object keys"
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs_hook)
    except DuplicateReferenceError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise DecisionContractError(f"{noun} JSON is malformed") from exc
    if not isinstance(value, Mapping):
        raise DecisionContractError(f"{noun} JSON must contain an object")
    try:
        encoded = canonical_json_bytes(value)
    except ContractValidationError as exc:
        raise DecisionContractError(f"{noun} JSON is not canonicalizable") from exc
    if encoded != payload.encode("utf-8"):
        raise NonCanonicalDecisionError(
            f"{noun} JSON changes during canonical round trip"
        )
    return value


class _DecisionCanonicalContract(CanonicalContract):
    """Canonical mixin with strict, unchanged JSON round trips."""

    @classmethod
    def from_json(cls, payload: str) -> "_DecisionCanonicalContract":
        value = _decode_json_object(payload, cls.__name__)
        decoder = getattr(cls, "from_dict", None)
        if decoder is None:
            raise DecisionContractError(f"{cls.__name__} has no decoder")
        result = decoder(value)
        if result.to_json() != payload:
            raise NonCanonicalDecisionError(
                f"{cls.__name__} changed during canonical round trip"
            )
        return result


def supervisor_digest_for_bytes(value: bytes) -> str:
    """Return the supervisor SHA-256 digest for exact canonical bytes."""

    if not isinstance(value, bytes):
        raise DecisionContractError("artifact bytes must be bytes")
    return _SHA256 + hashlib.sha256(value).hexdigest()


def cidv1_for_canonical_bytes(value: bytes) -> str:
    """Return a CIDv1 dag-json/sha2-256 identity for exact canonical bytes."""

    if not isinstance(value, bytes):
        raise DecisionContractError("artifact bytes must be bytes")
    raw = _CIDV1_DAG_JSON_SHA256_PREFIX + hashlib.sha256(value).digest()
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def canonical_artifact_bytes(value: Any) -> bytes:
    """Encode one bounded DAG-JSON artifact for pinning."""

    try:
        encoded = canonical_json_bytes(value)
    except ContractValidationError as exc:
        raise DecisionContractError("artifact is not canonical DAG-JSON") from exc
    if len(encoded) > ABSOLUTE_MAX_ARTIFACT_BYTES:
        raise DecisionBoundsError("artifact exceeds the absolute byte limit")
    return encoded


def _cid_digest(cid: Any) -> bytes:
    value = _text(cid, "cid_v1")
    if not value.startswith("b") or value != value.lower():
        raise DecisionIdentityError("cid_v1 must be canonical lowercase base32 CIDv1")
    try:
        padding = "=" * ((8 - len(value[1:]) % 8) % 8)
        decoded = base64.b32decode((value[1:].upper() + padding).encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise DecisionIdentityError("cid_v1 is malformed") from exc
    if (
        len(decoded) != len(_CIDV1_DAG_JSON_SHA256_PREFIX) + 32
        or not decoded.startswith(_CIDV1_DAG_JSON_SHA256_PREFIX)
    ):
        raise DecisionIdentityError(
            "cid_v1 must use CIDv1 dag-json with sha2-256"
        )
    canonical = "b" + base64.b32encode(decoded).decode("ascii").rstrip("=").lower()
    if canonical != value:
        raise DecisionIdentityError("cid_v1 is not canonically encoded")
    return decoded[len(_CIDV1_DAG_JSON_SHA256_PREFIX) :]


def _digest_bytes(value: Any) -> bytes:
    digest = _text(value, "supervisor_digest")
    if not digest.startswith(_SHA256) or len(digest) != len(_SHA256) + 64:
        raise DecisionIdentityError(
            "supervisor_digest must be sha256:<64 lowercase hex>"
        )
    try:
        result = bytes.fromhex(digest.removeprefix(_SHA256))
    except ValueError as exc:
        raise DecisionIdentityError("supervisor_digest is malformed") from exc
    if digest != _SHA256 + result.hex():
        raise DecisionIdentityError(
            "supervisor_digest must use canonical lowercase hex"
        )
    return result


@dataclass(frozen=True)
class PinnedArtifactRef(_DecisionCanonicalContract):
    """Body-free reference whose two identities bind the same canonical bytes."""

    SCHEMA: ClassVar[str] = PINNED_ARTIFACT_REF_SCHEMA

    artifact_id: str
    artifact_kind: str
    artifact_schema: str
    artifact_schema_version: str
    cid_v1: str
    supervisor_digest: str
    size_bytes: int
    producer_id: str
    authority: ReferenceAuthority

    def __post_init__(self) -> None:
        for name in (
            "artifact_id",
            "artifact_kind",
            "artifact_schema",
            "artifact_schema_version",
            "producer_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, ReferenceAuthority, "reference authority"),
        )
        cid_digest = _cid_digest(self.cid_v1)
        supervisor_digest = _digest_bytes(self.supervisor_digest)
        if cid_digest != supervisor_digest:
            raise DecisionIdentityError(
                "CIDv1 and supervisor digest do not identify the same bytes"
            )
        object.__setattr__(
            self,
            "size_bytes",
            _integer(
                self.size_bytes,
                "size_bytes",
                minimum=1,
                maximum=ABSOLUTE_MAX_ARTIFACT_BYTES,
            ),
        )
        if len(self.canonical_bytes()) > ABSOLUTE_MAX_TEXT_BYTES:
            raise DecisionBoundsError("pinned artifact reference is too large")

    @property
    def cid(self) -> str:
        return self.cid_v1

    @property
    def digest(self) -> str:
        return self.supervisor_digest

    @property
    def kind(self) -> str:
        return self.artifact_kind

    def verify_canonical_bytes(self, value: bytes) -> bool:
        """Verify byte count, canonical JSON encoding, CID, and digest."""

        if not isinstance(value, bytes) or len(value) != self.size_bytes:
            return False
        try:
            decoded = json.loads(value)
            canonical = canonical_artifact_bytes(decoded)
        except (DecisionContractError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        return (
            canonical == value
            and cidv1_for_canonical_bytes(value) == self.cid_v1
            and supervisor_digest_for_bytes(value) == self.supervisor_digest
        )

    verify = verify_canonical_bytes

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        artifact_id: str,
        artifact_kind: str,
        artifact_schema: str,
        artifact_schema_version: str,
        producer_id: str,
        authority: ReferenceAuthority | str,
    ) -> "PinnedArtifactRef":
        encoded = canonical_artifact_bytes(value)
        return cls.from_canonical_bytes(
            encoded,
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            artifact_schema=artifact_schema,
            artifact_schema_version=artifact_schema_version,
            producer_id=producer_id,
            authority=authority,
        )

    @classmethod
    def from_canonical_bytes(
        cls,
        value: bytes,
        *,
        artifact_id: str,
        artifact_kind: str,
        artifact_schema: str,
        artifact_schema_version: str,
        producer_id: str,
        authority: ReferenceAuthority | str,
    ) -> "PinnedArtifactRef":
        if not isinstance(value, bytes):
            raise DecisionContractError("canonical artifact bytes must be bytes")
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise NonCanonicalDecisionError(
                "artifact bytes must contain canonical JSON"
            ) from exc
        if canonical_artifact_bytes(decoded) != value:
            raise NonCanonicalDecisionError(
                "artifact bytes change during canonical round trip"
            )
        return cls(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            artifact_schema=artifact_schema,
            artifact_schema_version=artifact_schema_version,
            cid_v1=cidv1_for_canonical_bytes(value),
            supervisor_digest=supervisor_digest_for_bytes(value),
            size_bytes=len(value),
            producer_id=producer_id,
            authority=authority,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTRACT_VERSION,
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "artifact_schema": self.artifact_schema,
            "artifact_schema_version": self.artifact_schema_version,
            "cid_v1": self.cid_v1,
            "supervisor_digest": self.supervisor_digest,
            "size_bytes": self.size_bytes,
            "producer_id": self.producer_id,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PinnedArtifactRef":
        _schema(payload, cls.SCHEMA, "pinned artifact reference")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "artifact_id",
                "artifact_kind",
                "artifact_schema",
                "artifact_schema_version",
                "cid_v1",
                "supervisor_digest",
                "size_bytes",
                "producer_id",
                "authority",
                "content_id",
            },
            "pinned artifact reference",
        )
        names = (
            "artifact_id",
            "artifact_kind",
            "artifact_schema",
            "artifact_schema_version",
            "cid_v1",
            "supervisor_digest",
            "size_bytes",
            "producer_id",
            "authority",
        )
        values = {name: _required(payload, name, "pinned artifact reference") for name in names}
        result = cls(**values)
        _identity(payload, result.content_id, "pinned artifact reference")
        return result


@dataclass(frozen=True)
class SemanticRoot(_DecisionCanonicalContract):
    """A pinned root with an unambiguous semantic role."""

    SCHEMA: ClassVar[str] = SEMANTIC_ROOT_SCHEMA

    kind: SemanticRootKind
    artifact: PinnedArtifactRef
    coverage: tuple[WorktreeCoverage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, SemanticRootKind, "semantic root kind")
        )
        artifact = self.artifact
        if not isinstance(artifact, PinnedArtifactRef):
            if not isinstance(artifact, Mapping):
                raise DecisionContractError(
                    "semantic root artifact must be a PinnedArtifactRef"
                )
            artifact = PinnedArtifactRef.from_dict(artifact)
        if not artifact.authority.usable_as_root:
            raise UnknownAuthorityError(
                "semantic roots require authoritative or verified references"
            )
        object.__setattr__(self, "artifact", artifact)
        if isinstance(self.coverage, str) or not isinstance(self.coverage, Sequence):
            raise DecisionContractError("semantic root coverage must be a sequence")
        coverage = tuple(
            _enum(item, WorktreeCoverage, "worktree coverage")
            for item in self.coverage
        )
        if len(coverage) != len(set(coverage)):
            raise DuplicateReferenceError("semantic root coverage contains duplicates")
        if coverage != tuple(sorted(coverage, key=lambda item: item.value)):
            raise NonCanonicalDecisionError(
                "semantic root coverage must be canonically sorted"
            )
        if self.kind is SemanticRootKind.DIRTY_WORKTREE:
            if set(coverage) != set(REQUIRED_DIRTY_WORKTREE_COVERAGE):
                raise MissingSemanticRootError(
                    "dirty_worktree root must cover tracked, modified, staged, "
                    "deleted, and untracked inputs"
                )
        elif coverage:
            raise DecisionContractError(
                "coverage is only valid for the dirty_worktree root"
            )
        object.__setattr__(self, "coverage", coverage)

    @property
    def reference(self) -> PinnedArtifactRef:
        return self.artifact

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTRACT_VERSION,
            "kind": self.kind,
            "artifact": self.artifact.to_dict(),
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticRoot":
        _schema(payload, cls.SCHEMA, "semantic root")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "kind",
                "artifact",
                "coverage",
                "content_id",
            },
            "semantic root",
        )
        result = cls(
            kind=_required(payload, "kind", "semantic root"),
            artifact=_required(payload, "artifact", "semantic root"),
            coverage=_required(payload, "coverage", "semantic root"),
        )
        _identity(payload, result.content_id, "semantic root")
        return result


@dataclass(frozen=True)
class DecisionTarget(_DecisionCanonicalContract):
    """One exact resource and repository-relative path scope."""

    SCHEMA: ClassVar[str] = DECISION_TARGET_SCHEMA

    target_id: str
    resource_type: str
    repository_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        object.__setattr__(
            self, "resource_type", _text(self.resource_type, "resource_type")
        )
        if isinstance(self.repository_paths, str) or not isinstance(
            self.repository_paths, Sequence
        ):
            raise DecisionContractError("repository_paths must be a sequence")
        paths = tuple(
            _relative_path(item, "repository_paths")
            for item in self.repository_paths
        )
        if len(paths) > ABSOLUTE_MAX_PATHS:
            raise DecisionBoundsError("repository_paths exceeds its count bound")
        if len(paths) != len(set(paths)):
            raise DuplicateReferenceError("repository_paths contains duplicates")
        if paths != tuple(sorted(paths)):
            raise NonCanonicalDecisionError(
                "repository_paths must be canonically sorted"
            )
        object.__setattr__(self, "repository_paths", paths)

    @property
    def paths(self) -> tuple[str, ...]:
        return self.repository_paths

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTRACT_VERSION,
            "target_id": self.target_id,
            "resource_type": self.resource_type,
            "repository_paths": self.repository_paths,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionTarget":
        _schema(payload, cls.SCHEMA, "decision target")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "target_id",
                "resource_type",
                "repository_paths",
                "content_id",
            },
            "decision target",
        )
        result = cls(
            target_id=_required(payload, "target_id", "decision target"),
            resource_type=_required(payload, "resource_type", "decision target"),
            repository_paths=_required(
                payload, "repository_paths", "decision target"
            ),
        )
        _identity(payload, result.content_id, "decision target")
        return result


def _coerce_targets(value: Any) -> tuple[DecisionTarget, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DecisionContractError("action targets must be a sequence")
    if not value:
        raise DecisionContractError("action targets must not be empty")
    if len(value) > ABSOLUTE_MAX_TARGETS:
        raise DecisionBoundsError("action targets exceeds its count bound")
    result = tuple(
        item
        if isinstance(item, DecisionTarget)
        else DecisionTarget.from_dict(item)
        for item in value
    )
    ids = tuple(item.target_id for item in result)
    if len(ids) != len(set(ids)):
        raise DuplicateReferenceError("action target IDs must be unique")
    if result != tuple(sorted(result, key=lambda item: item.target_id)):
        raise NonCanonicalDecisionError("action targets must be canonically sorted")
    return result


@dataclass(frozen=True)
class ActionEnvelope(_DecisionCanonicalContract):
    """Exact action/tool invocation proposed by the decision."""

    SCHEMA: ClassVar[str] = ACTION_ENVELOPE_SCHEMA

    action_id: str
    action: str
    tool_id: str
    authority: DecisionAuthority
    arguments: Mapping[str, Any]
    targets: tuple[DecisionTarget, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id"))
        object.__setattr__(self, "action", _text(self.action, "action"))
        object.__setattr__(self, "tool_id", _text(self.tool_id, "tool_id"))
        object.__setattr__(self, "authority", _authority(self.authority))
        if not isinstance(self.arguments, Mapping):
            raise DecisionContractError("action arguments must be an object")
        object.__setattr__(
            self,
            "arguments",
            _freeze_value(
                self.arguments,
                name="action arguments",
                max_depth=ABSOLUTE_MAX_DEPTH,
                max_items=ABSOLUTE_MAX_ITEMS,
                max_text_bytes=ABSOLUTE_MAX_TEXT_BYTES,
            ),
        )
        object.__setattr__(self, "targets", _coerce_targets(self.targets))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTRACT_VERSION,
            "action_id": self.action_id,
            "action": self.action,
            "tool_id": self.tool_id,
            "authority": self.authority,
            "arguments": self.arguments,
            "targets": tuple(item.to_dict() for item in self.targets),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionEnvelope":
        _schema(payload, cls.SCHEMA, "action envelope")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "action_id",
                "action",
                "tool_id",
                "authority",
                "arguments",
                "targets",
                "content_id",
            },
            "action envelope",
        )
        result = cls(
            action_id=_required(payload, "action_id", "action envelope"),
            action=_required(payload, "action", "action envelope"),
            tool_id=_required(payload, "tool_id", "action envelope"),
            authority=_required(payload, "authority", "action envelope"),
            arguments=_required(payload, "arguments", "action envelope"),
            targets=_required(payload, "targets", "action envelope"),
        )
        _identity(payload, result.content_id, "action envelope")
        return result


@dataclass(frozen=True)
class EffectEnvelope(_DecisionCanonicalContract):
    """One exact expected effect and its verification obligation."""

    SCHEMA: ClassVar[str] = EFFECT_ENVELOPE_SCHEMA

    effect_id: str
    kind: EffectKind
    authority: DecisionAuthority
    target_ids: tuple[str, ...]
    repository_paths: tuple[str, ...]
    description: str
    verification: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_id", _text(self.effect_id, "effect_id"))
        object.__setattr__(self, "kind", _enum(self.kind, EffectKind, "effect kind"))
        object.__setattr__(self, "authority", _authority(self.authority))
        if self.authority is not self.kind.authority:
            raise UnknownAuthorityError(
                "effect authority does not match the effect kind"
            )
        object.__setattr__(
            self,
            "target_ids",
            _strings(
                self.target_ids,
                "effect target_ids",
                required=True,
                maximum=ABSOLUTE_MAX_TARGETS,
            ),
        )
        if isinstance(self.repository_paths, str) or not isinstance(
            self.repository_paths, Sequence
        ):
            raise DecisionContractError("effect repository_paths must be a sequence")
        paths = tuple(
            _relative_path(item, "effect repository_paths")
            for item in self.repository_paths
        )
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise NonCanonicalDecisionError(
                "effect repository_paths must be unique and sorted"
            )
        object.__setattr__(self, "repository_paths", paths)
        object.__setattr__(
            self, "description", _text(self.description, "effect description")
        )
        if not isinstance(self.verification, Mapping) or not self.verification:
            raise DecisionContractError(
                "effect verification must be a non-empty object"
            )
        object.__setattr__(
            self,
            "verification",
            _freeze_value(
                self.verification,
                name="effect verification",
                max_depth=ABSOLUTE_MAX_DEPTH,
                max_items=ABSOLUTE_MAX_ITEMS,
                max_text_bytes=ABSOLUTE_MAX_TEXT_BYTES,
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTRACT_VERSION,
            "effect_id": self.effect_id,
            "kind": self.kind,
            "authority": self.authority,
            "target_ids": self.target_ids,
            "repository_paths": self.repository_paths,
            "description": self.description,
            "verification": self.verification,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EffectEnvelope":
        _schema(payload, cls.SCHEMA, "effect envelope")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "effect_id",
                "kind",
                "authority",
                "target_ids",
                "repository_paths",
                "description",
                "verification",
                "content_id",
            },
            "effect envelope",
        )
        names = (
            "effect_id",
            "kind",
            "authority",
            "target_ids",
            "repository_paths",
            "description",
            "verification",
        )
        values = {name: _required(payload, name, "effect envelope") for name in names}
        result = cls(**values)
        _identity(payload, result.content_id, "effect envelope")
        return result


@dataclass(frozen=True)
class ApplicabilityFact(_DecisionCanonicalContract):
    """A sourced fact that may change legal, security, or tool applicability."""

    SCHEMA: ClassVar[str] = APPLICABILITY_FACT_SCHEMA

    fact_id: str
    kind: ApplicabilityFactKind
    predicate: str
    value: Mapping[str, Any]
    source: PinnedArtifactRef
    jurisdiction: str
    effective_from_ms: int | None
    effective_until_ms: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _text(self.fact_id, "fact_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ApplicabilityFactKind, "fact kind")
        )
        object.__setattr__(self, "predicate", _text(self.predicate, "predicate"))
        if not isinstance(self.value, Mapping) or not self.value:
            raise DecisionContractError("applicability fact value must be non-empty")
        object.__setattr__(
            self,
            "value",
            _freeze_value(
                self.value,
                name="applicability fact value",
                max_depth=ABSOLUTE_MAX_DEPTH,
                max_items=ABSOLUTE_MAX_ITEMS,
                max_text_bytes=ABSOLUTE_MAX_TEXT_BYTES,
            ),
        )
        source = self.source
        if not isinstance(source, PinnedArtifactRef):
            if not isinstance(source, Mapping):
                raise DecisionContractError(
                    "applicability source must be a PinnedArtifactRef"
                )
            source = PinnedArtifactRef.from_dict(source)
        if not source.authority.usable_as_root:
            raise UnknownAuthorityError(
                "applicability facts require an authoritative or verified source"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "jurisdiction",
            _text(self.jurisdiction, "jurisdiction", required=False),
        )
        start = _optional_integer(
            self.effective_from_ms, "effective_from_ms", minimum=0
        )
        end = _optional_integer(
            self.effective_until_ms, "effective_until_ms", minimum=0
        )
        if start is not None and end is not None and end <= start:
            raise DecisionContractError(
                "effective_until_ms must be greater than effective_from_ms"
            )
        if self.kind is ApplicabilityFactKind.JURISDICTION and not self.jurisdiction:
            raise DecisionContractError(
                "jurisdiction facts require an explicit jurisdiction"
            )
        if self.kind is ApplicabilityFactKind.EFFECTIVE_TIME and (
            start is None or end is None
        ):
            raise DecisionContractError(
                "effective-time facts require an explicit bounded interval"
            )
        object.__setattr__(self, "effective_from_ms", start)
        object.__setattr__(self, "effective_until_ms", end)

    def applies_at(self, effective_at_ms: int, jurisdiction: str) -> bool:
        moment = _integer(effective_at_ms, "effective_at_ms", minimum=0)
        if self.jurisdiction and self.jurisdiction != jurisdiction:
            return False
        if self.effective_from_ms is not None and moment < self.effective_from_ms:
            return False
        return self.effective_until_ms is None or moment < self.effective_until_ms

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTRACT_VERSION,
            "fact_id": self.fact_id,
            "kind": self.kind,
            "predicate": self.predicate,
            "value": self.value,
            "source": self.source.to_dict(),
            "jurisdiction": self.jurisdiction,
            "effective_from_ms": self.effective_from_ms,
            "effective_until_ms": self.effective_until_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ApplicabilityFact":
        _schema(payload, cls.SCHEMA, "applicability fact")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "fact_id",
                "kind",
                "predicate",
                "value",
                "source",
                "jurisdiction",
                "effective_from_ms",
                "effective_until_ms",
                "content_id",
            },
            "applicability fact",
        )
        names = (
            "fact_id",
            "kind",
            "predicate",
            "value",
            "source",
            "jurisdiction",
            "effective_from_ms",
            "effective_until_ms",
        )
        values = {
            name: _required(payload, name, "applicability fact") for name in names
        }
        result = cls(**values)
        _identity(payload, result.content_id, "applicability fact")
        return result


@dataclass(frozen=True)
class CapabilityEnvelope(_DecisionCanonicalContract):
    """One exact runtime capability and its pinned configuration."""

    SCHEMA: ClassVar[str] = CAPABILITY_ENVELOPE_SCHEMA

    capability_id: str
    provider_id: str
    version: str
    configuration: PinnedArtifactRef

    def __post_init__(self) -> None:
        for name in ("capability_id", "provider_id", "version"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        configuration = self.configuration
        if not isinstance(configuration, PinnedArtifactRef):
            if not isinstance(configuration, Mapping):
                raise DecisionContractError(
                    "capability configuration must be a PinnedArtifactRef"
                )
            configuration = PinnedArtifactRef.from_dict(configuration)
        if not configuration.authority.usable_as_root:
            raise UnknownAuthorityError(
                "capability configuration must be verified"
            )
        object.__setattr__(self, "configuration", configuration)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTRACT_VERSION,
            "capability_id": self.capability_id,
            "provider_id": self.provider_id,
            "version": self.version,
            "configuration": self.configuration.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityEnvelope":
        _schema(payload, cls.SCHEMA, "capability envelope")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "capability_id",
                "provider_id",
                "version",
                "configuration",
                "content_id",
            },
            "capability envelope",
        )
        result = cls(
            capability_id=_required(payload, "capability_id", "capability envelope"),
            provider_id=_required(payload, "provider_id", "capability envelope"),
            version=_required(payload, "version", "capability envelope"),
            configuration=_required(
                payload, "configuration", "capability envelope"
            ),
        )
        _identity(payload, result.content_id, "capability envelope")
        return result


@dataclass(frozen=True)
class DecisionBudget(_DecisionCanonicalContract):
    """Hard limits carried in, and therefore identified by, every decision."""

    SCHEMA: ClassVar[str] = DECISION_BUDGET_SCHEMA

    max_input_tokens: int
    max_output_tokens: int
    max_serialized_bytes: int
    max_artifact_bytes: int
    max_graph_hops: int
    max_retrieval_results: int
    max_proof_attempts: int
    max_latency_ms: int
    max_expansions: int
    max_items: int
    max_depth: int
    max_text_bytes: int
    max_actions: int
    max_effects: int
    max_facts: int
    max_capabilities: int

    def __post_init__(self) -> None:
        maxima = {
            "max_input_tokens": 10_000_000,
            "max_output_tokens": 10_000_000,
            "max_serialized_bytes": ABSOLUTE_MAX_DECISION_BYTES,
            "max_artifact_bytes": ABSOLUTE_MAX_ARTIFACT_BYTES,
            "max_graph_hops": ABSOLUTE_MAX_DEPTH,
            "max_retrieval_results": ABSOLUTE_MAX_ITEMS,
            "max_proof_attempts": ABSOLUTE_MAX_ITEMS,
            "max_latency_ms": ABSOLUTE_MAX_LATENCY_MS,
            "max_expansions": ABSOLUTE_MAX_ITEMS,
            "max_items": ABSOLUTE_MAX_ITEMS,
            "max_depth": ABSOLUTE_MAX_DEPTH,
            "max_text_bytes": ABSOLUTE_MAX_TEXT_BYTES,
            "max_actions": ABSOLUTE_MAX_ACTIONS,
            "max_effects": ABSOLUTE_MAX_EFFECTS,
            "max_facts": ABSOLUTE_MAX_FACTS,
            "max_capabilities": ABSOLUTE_MAX_CAPABILITIES,
        }
        for name, maximum in maxima.items():
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=1, maximum=maximum),
            )
        for name in (
            "max_actions",
            "max_effects",
            "max_facts",
            "max_capabilities",
        ):
            if getattr(self, name) > self.max_items:
                raise DecisionBoundsError(f"{name} cannot exceed max_items")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTRACT_VERSION,
            **{
                name: getattr(self, name)
                for name in (
                    "max_input_tokens",
                    "max_output_tokens",
                    "max_serialized_bytes",
                    "max_artifact_bytes",
                    "max_graph_hops",
                    "max_retrieval_results",
                    "max_proof_attempts",
                    "max_latency_ms",
                    "max_expansions",
                    "max_items",
                    "max_depth",
                    "max_text_bytes",
                    "max_actions",
                    "max_effects",
                    "max_facts",
                    "max_capabilities",
                )
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionBudget":
        _schema(payload, cls.SCHEMA, "decision budget")
        fields = set(cls.__dataclass_fields__).difference({"SCHEMA"})
        allowed = {"schema", "contract_version", "content_id", *fields}
        _reject_unknown(payload, allowed, "decision budget")
        values = {
            name: _required(payload, name, "decision budget") for name in fields
        }
        result = cls(**values)
        _identity(payload, result.content_id, "decision budget")
        return result


@dataclass(frozen=True)
class AuthorityEnvelope(_DecisionCanonicalContract):
    """Principal, authority, capability, lease, fence, and replay binding."""

    SCHEMA: ClassVar[str] = AUTHORITY_ENVELOPE_SCHEMA

    principal_id: str
    requested_authority: DecisionAuthority
    capability_ids: tuple[str, ...]
    lease_id: str | None
    fencing_epoch: int | None
    idempotency_key: str | None
    authorization: PinnedArtifactRef | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "principal_id", _text(self.principal_id, "principal_id")
        )
        object.__setattr__(
            self,
            "requested_authority",
            _authority(self.requested_authority),
        )
        object.__setattr__(
            self,
            "capability_ids",
            _strings(
                self.capability_ids,
                "capability_ids",
                required=True,
                maximum=ABSOLUTE_MAX_CAPABILITIES,
            ),
        )
        lease = (
            None
            if self.lease_id is None
            else _text(self.lease_id, "lease_id")
        )
        fence = _optional_integer(self.fencing_epoch, "fencing_epoch", minimum=0)
        replay = (
            None
            if self.idempotency_key is None
            else _text(self.idempotency_key, "idempotency_key", maximum=256)
        )
        authorization = self.authorization
        if authorization is not None and not isinstance(
            authorization, PinnedArtifactRef
        ):
            if not isinstance(authorization, Mapping):
                raise DecisionContractError(
                    "authorization must be a PinnedArtifactRef or null"
                )
            authorization = PinnedArtifactRef.from_dict(authorization)
        if authorization is not None and not authorization.authority.usable_as_root:
            raise UnknownAuthorityError("authorization reference must be verified")
        if self.requested_authority is DecisionAuthority.MUTATION:
            if lease is None or fence is None or replay is None or authorization is None:
                raise DecisionBindingError(
                    "mutation authority requires authorization, lease, fence, "
                    "and idempotency"
                )
        object.__setattr__(self, "lease_id", lease)
        object.__setattr__(self, "fencing_epoch", fence)
        object.__setattr__(self, "idempotency_key", replay)
        object.__setattr__(self, "authorization", authorization)

    @property
    def principal(self) -> str:
        return self.principal_id

    @property
    def authority(self) -> DecisionAuthority:
        return self.requested_authority

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTRACT_VERSION,
            "principal_id": self.principal_id,
            "requested_authority": self.requested_authority,
            "capability_ids": self.capability_ids,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "idempotency_key": self.idempotency_key,
            "authorization": (
                self.authorization.to_dict() if self.authorization else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorityEnvelope":
        _schema(payload, cls.SCHEMA, "authority envelope")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "principal_id",
                "requested_authority",
                "capability_ids",
                "lease_id",
                "fencing_epoch",
                "idempotency_key",
                "authorization",
                "content_id",
            },
            "authority envelope",
        )
        names = (
            "principal_id",
            "requested_authority",
            "capability_ids",
            "lease_id",
            "fencing_epoch",
            "idempotency_key",
            "authorization",
        )
        values = {name: _required(payload, name, "authority envelope") for name in names}
        result = cls(**values)
        _identity(payload, result.content_id, "authority envelope")
        return result


def _coerce_sequence(
    value: Any,
    *,
    kind: type,
    decoder: Any,
    name: str,
    required: bool,
    maximum: int,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DecisionContractError(f"{name} must be a sequence")
    if required and not value:
        raise DecisionContractError(f"{name} must not be empty")
    if len(value) > maximum:
        raise DecisionBoundsError(f"{name} exceeds its count bound")
    return tuple(item if isinstance(item, kind) else decoder(item) for item in value)


@dataclass(frozen=True)
class DecisionRequest(_DecisionCanonicalContract):
    """Complete canonical input to one proof-directed decision."""

    SCHEMA: ClassVar[str] = DECISION_REQUEST_SCHEMA

    decision_kind: DecisionKind
    stage: DecisionStage
    objective_id: str
    objective_revision: str
    acceptance_id: str
    repository_id: str
    repository_path: str
    jurisdiction: str
    effective_at_ms: int | None
    environment_id: str
    model_id: str
    toolchain_id: str
    authority: AuthorityEnvelope
    budget: DecisionBudget
    action: ActionEnvelope
    expected_effects: tuple[EffectEnvelope, ...]
    semantic_roots: tuple[SemanticRoot, ...]
    applicability_facts: tuple[ApplicabilityFact, ...]
    capabilities: tuple[CapabilityEnvelope, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_kind",
            _enum(self.decision_kind, DecisionKind, "decision kind"),
        )
        object.__setattr__(
            self, "stage", _enum(self.stage, DecisionStage, "decision stage")
        )
        for name in (
            "objective_id",
            "objective_revision",
            "acceptance_id",
            "repository_id",
            "environment_id",
            "model_id",
            "toolchain_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "repository_path",
            _absolute_root(self.repository_path, "repository_path"),
        )
        object.__setattr__(
            self,
            "jurisdiction",
            _text(self.jurisdiction, "jurisdiction", required=False),
        )
        object.__setattr__(
            self,
            "effective_at_ms",
            _optional_integer(self.effective_at_ms, "effective_at_ms", minimum=0),
        )
        authority = self.authority
        if not isinstance(authority, AuthorityEnvelope):
            if not isinstance(authority, Mapping):
                raise DecisionContractError(
                    "authority must be an AuthorityEnvelope"
                )
            authority = AuthorityEnvelope.from_dict(authority)
        object.__setattr__(self, "authority", authority)
        budget = self.budget
        if not isinstance(budget, DecisionBudget):
            if not isinstance(budget, Mapping):
                raise DecisionContractError("budget must be a DecisionBudget")
            budget = DecisionBudget.from_dict(budget)
        object.__setattr__(self, "budget", budget)
        action = self.action
        if not isinstance(action, ActionEnvelope):
            if not isinstance(action, Mapping):
                raise DecisionContractError("action must be an ActionEnvelope")
            action = ActionEnvelope.from_dict(action)
        object.__setattr__(self, "action", action)
        if not authority.requested_authority.allows(action.authority):
            raise UnknownAuthorityError(
                "requested authority does not cover the action"
            )

        effects = _coerce_sequence(
            self.expected_effects,
            kind=EffectEnvelope,
            decoder=EffectEnvelope.from_dict,
            name="expected_effects",
            required=True,
            maximum=budget.max_effects,
        )
        effect_ids = tuple(item.effect_id for item in effects)
        if len(effect_ids) != len(set(effect_ids)):
            raise DuplicateReferenceError("expected effect IDs must be unique")
        if effects != tuple(sorted(effects, key=lambda item: item.effect_id)):
            raise NonCanonicalDecisionError(
                "expected_effects must be canonically sorted"
            )
        target_ids = {item.target_id for item in action.targets}
        for effect in effects:
            if not authority.requested_authority.allows(effect.authority):
                raise UnknownAuthorityError(
                    "requested authority does not cover an expected effect"
                )
            if not set(effect.target_ids).issubset(target_ids):
                raise DecisionBindingError(
                    "expected effect references an undeclared action target"
                )
        object.__setattr__(self, "expected_effects", effects)

        roots = _coerce_sequence(
            self.semantic_roots,
            kind=SemanticRoot,
            decoder=SemanticRoot.from_dict,
            name="semantic_roots",
            required=True,
            # Inspect the bounded population before enforcing the exact closed
            # role set so duplicates receive their stable semantic rejection
            # rather than an incidental count error.
            maximum=ABSOLUTE_MAX_ITEMS,
        )
        root_kinds = tuple(item.kind for item in roots)
        if len(root_kinds) != len(set(root_kinds)):
            raise DuplicateReferenceError(
                "semantic roots contain duplicate or conflicting roles"
            )
        missing = MANDATORY_SEMANTIC_ROOT_KINDS.difference(root_kinds)
        if missing:
            raise MissingSemanticRootError(
                "missing semantic roots: "
                + ", ".join(sorted(item.value for item in missing))
            )
        if roots != tuple(sorted(roots, key=lambda item: item.kind.value)):
            raise NonCanonicalDecisionError(
                "semantic_roots must be canonically sorted"
            )
        object.__setattr__(self, "semantic_roots", roots)

        facts = _coerce_sequence(
            self.applicability_facts,
            kind=ApplicabilityFact,
            decoder=ApplicabilityFact.from_dict,
            name="applicability_facts",
            required=False,
            maximum=budget.max_facts,
        )
        fact_ids = tuple(item.fact_id for item in facts)
        if len(fact_ids) != len(set(fact_ids)):
            raise DuplicateReferenceError("applicability fact IDs must be unique")
        if facts != tuple(sorted(facts, key=lambda item: item.fact_id)):
            raise NonCanonicalDecisionError(
                "applicability_facts must be canonically sorted"
            )
        if any(item.jurisdiction for item in facts) and not self.jurisdiction:
            raise DecisionBindingError(
                "jurisdictional facts require request jurisdiction"
            )
        if any(
            item.effective_from_ms is not None or item.effective_until_ms is not None
            for item in facts
        ) and self.effective_at_ms is None:
            raise DecisionBindingError(
                "temporal facts require request effective_at_ms"
            )
        if self.effective_at_ms is not None and any(
            not item.applies_at(self.effective_at_ms, self.jurisdiction)
            for item in facts
        ):
            raise DecisionBindingError(
                "applicability fact is outside the request jurisdiction or time"
            )
        object.__setattr__(self, "applicability_facts", facts)

        capabilities = _coerce_sequence(
            self.capabilities,
            kind=CapabilityEnvelope,
            decoder=CapabilityEnvelope.from_dict,
            name="capabilities",
            required=True,
            maximum=budget.max_capabilities,
        )
        capability_ids = tuple(item.capability_id for item in capabilities)
        if len(capability_ids) != len(set(capability_ids)):
            raise DuplicateReferenceError("capability IDs must be unique")
        if capabilities != tuple(
            sorted(capabilities, key=lambda item: item.capability_id)
        ):
            raise NonCanonicalDecisionError(
                "capabilities must be canonically sorted"
            )
        if set(authority.capability_ids) != set(capability_ids):
            raise DecisionBindingError(
                "authority capability IDs must exactly match capability envelopes"
            )
        if action.tool_id not in set(capability_ids):
            raise DecisionBindingError(
                "action tool_id must name a declared capability"
            )
        object.__setattr__(self, "capabilities", capabilities)

        references = [
            *(root.artifact for root in roots),
            *(fact.source for fact in facts),
            *(capability.configuration for capability in capabilities),
        ]
        if authority.authorization is not None:
            references.append(authority.authorization)
        by_artifact: dict[str, PinnedArtifactRef] = {}
        by_cid: dict[str, PinnedArtifactRef] = {}
        for reference in references:
            if reference.size_bytes > budget.max_artifact_bytes:
                raise DecisionBoundsError(
                    "pinned artifact exceeds max_artifact_bytes"
                )
            old_artifact = by_artifact.get(reference.artifact_id)
            old_cid = by_cid.get(reference.cid_v1)
            if old_artifact is not None or old_cid is not None:
                previous = old_artifact or old_cid
                assert previous is not None
                if previous != reference:
                    raise DuplicateReferenceError(
                        "conflicting pinned artifact references"
                    )
                raise DuplicateReferenceError("duplicate pinned artifact reference")
            by_artifact[reference.artifact_id] = reference
            by_cid[reference.cid_v1] = reference

        # Re-measure the complete assembled record with the caller's hard
        # limits.  Nested envelopes use absolute bounds while being built so
        # they can stand alone; a DecisionRequest must additionally fit its
        # stricter aggregate budget.
        _freeze_value(
            self._payload(),
            name="decision request",
            max_depth=budget.max_depth,
            max_items=budget.max_items,
            max_text_bytes=budget.max_text_bytes,
            check_paths=False,
        )
        if len(self.canonical_bytes()) > budget.max_serialized_bytes:
            raise DecisionBoundsError(
                "decision request exceeds max_serialized_bytes"
            )

    @property
    def request_id(self) -> str:
        return self.content_id

    @property
    def principal(self) -> str:
        return self.authority.principal_id

    @property
    def requested_authority(self) -> DecisionAuthority:
        return self.authority.requested_authority

    @property
    def lease_id(self) -> str | None:
        return self.authority.lease_id

    @property
    def fencing_epoch(self) -> int | None:
        return self.authority.fencing_epoch

    @property
    def idempotency_key(self) -> str | None:
        return self.authority.idempotency_key

    @property
    def roots_by_kind(self) -> Mapping[SemanticRootKind, SemanticRoot]:
        return MappingProxyType({root.kind: root for root in self.semantic_roots})

    def root(self, kind: SemanticRootKind | str) -> SemanticRoot:
        normalized = _enum(kind, SemanticRootKind, "semantic root kind")
        return self.roots_by_kind[normalized]

    def artifact_root(
        self, kind: SemanticRootKind | str
    ) -> PinnedArtifactRef:
        """Return the pinned artifact for one mandatory semantic role."""

        return self.root(kind).artifact

    @property
    def repository_root(self) -> PinnedArtifactRef:
        return self.artifact_root(SemanticRootKind.REPOSITORY)

    @property
    def dirty_worktree_root(self) -> PinnedArtifactRef:
        return self.artifact_root(SemanticRootKind.DIRTY_WORKTREE)

    @property
    def intent_ir_root(self) -> PinnedArtifactRef:
        return self.artifact_root(SemanticRootKind.INTENT_IR)

    @property
    def legal_ir_root(self) -> PinnedArtifactRef:
        return self.artifact_root(SemanticRootKind.LEGAL_IR)

    @property
    def security_ir_root(self) -> PinnedArtifactRef:
        return self.artifact_root(SemanticRootKind.SECURITY_IR)

    @property
    def program_root(self) -> PinnedArtifactRef:
        return self.artifact_root(SemanticRootKind.PROGRAM)

    @property
    def ast_root(self) -> PinnedArtifactRef:
        return self.program_root

    @property
    def tool_catalog_root(self) -> PinnedArtifactRef:
        return self.artifact_root(SemanticRootKind.TOOL_CATALOG)

    @property
    def policy_root(self) -> PinnedArtifactRef:
        return self.artifact_root(SemanticRootKind.POLICY)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTRACT_VERSION,
            "decision_kind": self.decision_kind,
            "stage": self.stage,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "acceptance_id": self.acceptance_id,
            "repository_id": self.repository_id,
            "repository_path": self.repository_path,
            "jurisdiction": self.jurisdiction,
            "effective_at_ms": self.effective_at_ms,
            "environment_id": self.environment_id,
            "model_id": self.model_id,
            "toolchain_id": self.toolchain_id,
            "authority": self.authority.to_dict(),
            "budget": self.budget.to_dict(),
            "action": self.action.to_dict(),
            "expected_effects": tuple(
                item.to_dict() for item in self.expected_effects
            ),
            "semantic_roots": tuple(item.to_dict() for item in self.semantic_roots),
            "applicability_facts": tuple(
                item.to_dict() for item in self.applicability_facts
            ),
            "capabilities": tuple(item.to_dict() for item in self.capabilities),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionRequest":
        _schema(payload, cls.SCHEMA, "decision request")
        names = (
            "decision_kind",
            "stage",
            "objective_id",
            "objective_revision",
            "acceptance_id",
            "repository_id",
            "repository_path",
            "jurisdiction",
            "effective_at_ms",
            "environment_id",
            "model_id",
            "toolchain_id",
            "authority",
            "budget",
            "action",
            "expected_effects",
            "semantic_roots",
            "applicability_facts",
            "capabilities",
        )
        _reject_unknown(
            payload,
            {"schema", "contract_version", "content_id", *names},
            "decision request",
        )
        values = {
            name: _required(payload, name, "decision request") for name in names
        }
        result = cls(**values)
        _identity(payload, result.content_id, "decision request")
        return result


def canonical_decision_json_bytes(value: Any) -> bytes:
    """Return the canonical DAG-JSON bytes used by decision contracts."""

    try:
        return canonical_json_bytes(value)
    except ContractValidationError as exc:
        raise DecisionContractError("value is not canonical decision JSON") from exc


def decode_decision_request(payload: Mapping[str, Any]) -> DecisionRequest:
    """Decode the strict pre-resolution decision boundary."""

    return DecisionRequest.from_dict(payload)


# Readable compatibility names for consumers that use domain rather than wire
# terminology.  They remain aliases, so canonical identities are unaffected.
Action = ActionEnvelope
DecisionAction = ActionEnvelope
ExpectedEffect = EffectEnvelope
DecisionEffect = EffectEnvelope
TargetEnvelope = DecisionTarget
BudgetEnvelope = DecisionBudget
DecisionBudgetEnvelope = DecisionBudget
AuthorityBinding = AuthorityEnvelope
DecisionAuthorityEnvelope = AuthorityEnvelope
SemanticRootEnvelope = SemanticRoot
ApplicabilityFactEnvelope = ApplicabilityFact
ArtifactAuthority = ReferenceAuthority
PinnedArtifactReference = PinnedArtifactRef


__all__ = [
    "ABSOLUTE_MAX_ARTIFACT_BYTES",
    "ABSOLUTE_MAX_DECISION_BYTES",
    "ACTION_ENVELOPE_SCHEMA",
    "APPLICABILITY_FACT_SCHEMA",
    "Action",
    "ActionEnvelope",
    "ApplicabilityFact",
    "ApplicabilityFactEnvelope",
    "ApplicabilityFactKind",
    "ArtifactAuthority",
    "AuthorityBinding",
    "AuthorityEnvelope",
    "BudgetEnvelope",
    "CAPABILITY_ENVELOPE_SCHEMA",
    "CONTRACT_VERSION",
    "CapabilityEnvelope",
    "DECISION_BUDGET_SCHEMA",
    "DECISION_CONTRACT_VERSION",
    "DECISION_REQUEST_SCHEMA",
    "DecisionAction",
    "DecisionAuthority",
    "DecisionAuthorityEnvelope",
    "DecisionBindingError",
    "DecisionBoundsError",
    "DecisionBudget",
    "DecisionBudgetEnvelope",
    "DecisionContractError",
    "DecisionEffect",
    "DecisionIdentityError",
    "DecisionKind",
    "DecisionPathEscapeError",
    "DecisionRequest",
    "DecisionStage",
    "DecisionTarget",
    "DuplicateReferenceError",
    "EFFECT_ENVELOPE_SCHEMA",
    "EffectEnvelope",
    "EffectKind",
    "ExpectedEffect",
    "MANDATORY_SEMANTIC_ROOT_KINDS",
    "MissingSemanticRootError",
    "NonCanonicalDecisionError",
    "PINNED_ARTIFACT_REF_SCHEMA",
    "PinnedArtifactRef",
    "PinnedArtifactReference",
    "REQUIRED_DIRTY_WORKTREE_COVERAGE",
    "ReferenceAuthority",
    "SCHEMA_VERSION",
    "SEMANTIC_ROOT_SCHEMA",
    "SemanticRoot",
    "SemanticRootEnvelope",
    "SemanticRootKind",
    "TargetEnvelope",
    "UnknownAuthorityError",
    "WorktreeCoverage",
    "canonical_artifact_bytes",
    "canonical_decision_json_bytes",
    "cidv1_for_canonical_bytes",
    "decode_decision_request",
    "supervisor_digest_for_bytes",
]
