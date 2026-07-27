"""Strict generation-2 contracts for the agent supervisor.

This module is deliberately provider- and transport-free.  It defines the
content-addressed identities and bounded receipts shared by later generation-2
runtime components without granting those components authority by discovery.

Result-bearing records all carry :class:`ResultBinding`.  The binding freezes
the repository/tree, objective/task, policy, producer, capability,
environment, and complete semantic dependency population.  Identities and
derived summaries are recomputed while decoding; callers cannot supply a
different identity or upgrade a record's authority.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import MISSING, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, TypeVar

from .formal_verification_contracts import (
    CanonicalContract,
    canonical_json_bytes,
    content_identity,
)


SUPERVISOR_V2_CONTRACT_VERSION: Final[int] = 2
CONTRACT_VERSION: Final[int] = SUPERVISOR_V2_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = SUPERVISOR_V2_CONTRACT_VERSION

MAX_RECEIPT_BYTES: Final[int] = 262_144
MAX_PROJECTION_BYTES: Final[int] = 1_048_576
MAX_TEXT_BYTES: Final[int] = 16_384
MAX_COLLECTION_ITEMS: Final[int] = 4_096
MAX_PAYLOAD_DEPTH: Final[int] = 16
MAX_REFILL_GOALS: Final[int] = 8
MAX_REFILL_TASKS: Final[int] = 24
MIN_REFILL_COOLDOWN_SECONDS: Final[int] = 6 * 60 * 60
MILLION: Final[int] = 1_000_000

# Stable producer routing identity declared by the ASI-G200 objective.  The
# identifier is metadata, not proof by itself; later rollout assembly must
# obtain a fresh current-tree receipt before it can qualify.
V2_CONTRACT_INTEGRITY_REQUIREMENT_ID: Final[str] = (
    "66755390419724488747029814613031064528"
)

SEMANTIC_DEPENDENCY_IDENTITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/v2/semantic-dependency-identity@2"
)
RESULT_BINDING_SCHEMA = "ipfs_accelerate_py/agent-supervisor/v2/result-binding@2"
ARTIFACT_BOUNDS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/v2/artifact-bounds@2"
EVIDENCE_REFERENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/v2/evidence-reference@2"
)
STAGE_EVENT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/v2/stage-event@2"
STAGE_RECEIPT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/v2/stage-receipt@2"
OPERATION_CAPABILITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/v2/operation-capability@2"
)
UNCERTAINTY_RECORD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/v2/uncertainty-record@2"
)
DISAGREEMENT_RECORD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/v2/disagreement-record@2"
)
PROMOTION_VECTOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/v2/promotion-vector@2"
)
SUPERVISOR_V2_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/v2/policy@2"
)
TARGET_DESCRIPTOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/v2/target-descriptor@2"
)
REFILL_EPOCH_SCHEMA = "ipfs_accelerate_py/agent-supervisor/v2/refill-epoch@2"
TYPED_FAILURE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/v2/typed-failure@2"

NON_COMPENSABLE_GATES: Final[tuple[str, ...]] = (
    "artifact_bounds",
    "authority",
    "escaped_defects",
    "freshness",
    "idempotency",
    "population",
    "safety",
)


class SupervisorV2ContractError(ValueError):
    """Base error for malformed or unsafe generation-2 contracts."""


class UnknownFieldError(SupervisorV2ContractError):
    """A closed contract contained a field outside its schema."""


class ContractBoundsError(SupervisorV2ContractError):
    """A contract exceeded a byte, item, text, or nesting bound."""


class DetachedReferenceError(SupervisorV2ContractError):
    """A reference was absent from, or foreign to, its containing receipt."""


class ForgedSummaryError(SupervisorV2ContractError):
    """A claimed summary or content identity did not match canonical content."""


class PathEscapeError(SupervisorV2ContractError):
    """A target or artifact path could escape an explicitly allowed root."""


class PromotionGateError(SupervisorV2ContractError):
    """A promotion attempted to compensate for a hard-gate failure."""


class AuthorityClass(str, Enum):
    """Closed semantic authority vocabulary.

    The values are intentionally classes, not an ordering.  A proof cannot be
    converted into merge authority by comparing ranks, and a proposal cannot
    claim mutation or completion authority.
    """

    DIAGNOSTIC = "diagnostic"
    PROPOSAL = "proposal"
    VALIDATION = "validation"
    PROOF = "proof"
    MERGE = "merge"
    MUTATION = "mutation"
    COMPLETION = "completion"


class StageEventKind(str, Enum):
    STARTED = "started"
    PROGRESSED = "progressed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"

    @property
    def terminal(self) -> bool:
        return self not in {StageEventKind.STARTED, StageEventKind.PROGRESSED}


class EvidenceFreshness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


class UncertaintyDisposition(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    ACCEPTED_RISK = "accepted_risk"
    BLOCKED = "blocked"


class DisagreementResolution(str, Enum):
    UNRESOLVED = "unresolved"
    DETERMINISTIC_POLICY = "deterministic_policy"
    INDEPENDENT_VALIDATION = "independent_validation"
    EXPLICIT_UNCERTAINTY = "explicit_uncertainty"


class PromotionDecision(str, Enum):
    SHADOW = "shadow"
    PROVISIONAL = "provisional"
    PROMOTE = "promote"


class TargetKind(str, Enum):
    REPOSITORY = "repository"
    STATE = "state"
    REPOSITORY_AND_STATE = "repository_and_state"
    ARTIFACT = "artifact"


class FailureCode(str, Enum):
    INVALID_CONTRACT = "invalid_contract"
    UNKNOWN_FIELD = "unknown_field"
    DETACHED_REFERENCE = "detached_reference"
    FORGED_SUMMARY = "forged_summary"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    PATH_ESCAPE = "path_escape"
    AUTHORITY_VIOLATION = "authority_violation"
    STALE_TREE = "stale_tree"
    STALE_EVIDENCE = "stale_evidence"
    STALE_LEASE = "stale_lease"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    VALIDATION_FAILED = "validation_failed"
    PROOF_FAILED = "proof_failed"
    MERGE_CONFLICT = "merge_conflict"
    MUTATION_FAILED = "mutation_failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class RetryDisposition(str, Enum):
    NEVER = "never"
    SAME_BINDING = "same_binding"
    NEW_BINDING = "new_binding"
    AFTER_EXTERNAL_CHANGE = "after_external_change"


class RefillEpochStatus(str, Enum):
    SHADOW = "shadow"
    HEALTHY_EXHAUSTION = "healthy_exhaustion"
    PROPOSED = "proposed"
    MATERIALIZED = "materialized"
    REJECTED = "rejected"


T = TypeVar("T")


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    max_bytes: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise SupervisorV2ContractError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise SupervisorV2ContractError(f"{field_name} must not be empty")
    if "\x00" in result:
        raise SupervisorV2ContractError(f"{field_name} must not contain NUL")
    if len(result.encode("utf-8")) > max_bytes:
        raise ContractBoundsError(f"{field_name} exceeds {max_bytes} UTF-8 bytes")
    return result


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SupervisorV2ContractError(f"{field_name} must be a boolean")
    return value


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SupervisorV2ContractError(f"{field_name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise ContractBoundsError(
            f"{field_name} must be at least {minimum}{suffix}"
        )
    return value


def _enum(value: Any, enum_type: type[T], *, field_name: str) -> T:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise SupervisorV2ContractError(
            f"{field_name} is outside the closed {enum_type.__name__} vocabulary"
        ) from exc


def _strings(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    preserve_order: bool = False,
    maximum: int = MAX_COLLECTION_ITEMS,
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise SupervisorV2ContractError(f"{field_name} must be a sequence")
    if len(values) > maximum:
        raise ContractBoundsError(f"{field_name} exceeds {maximum} items")
    result: list[str] = []
    for index, value in enumerate(values):
        item = _text(value, field_name=f"{field_name}[{index}]")
        if item in result:
            raise SupervisorV2ContractError(
                f"{field_name} must not contain duplicates"
            )
        result.append(item)
    if required and not result:
        raise SupervisorV2ContractError(f"{field_name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _records(
    values: Any,
    record_type: type[T],
    *,
    field_name: str,
    required: bool = False,
    maximum: int = MAX_COLLECTION_ITEMS,
) -> tuple[T, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise SupervisorV2ContractError(f"{field_name} must be a sequence")
    if len(values) > maximum:
        raise ContractBoundsError(f"{field_name} exceeds {maximum} items")
    result: list[T] = []
    decoder = getattr(record_type, "from_dict")
    for item in values:
        if isinstance(item, record_type):
            value = item
        elif isinstance(item, Mapping):
            value = decoder(item)
        else:
            raise SupervisorV2ContractError(
                f"{field_name} entries must be {record_type.__name__} records"
            )
        result.append(value)
    if required and not result:
        raise SupervisorV2ContractError(f"{field_name} must not be empty")
    identities = [getattr(item, "content_id") for item in result]
    if len(set(identities)) != len(identities):
        raise SupervisorV2ContractError(f"{field_name} contains duplicate identities")
    return tuple(sorted(result, key=lambda item: getattr(item, "content_id")))


def _timestamp(value: Any, *, field_name: str, required: bool = True) -> str:
    if not value and not required:
        return ""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise SupervisorV2ContractError(
                f"{field_name} must be an ISO-8601 timestamp"
            ) from exc
    else:
        raise SupervisorV2ContractError(
            f"{field_name} must be a datetime or ISO-8601 string"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SupervisorV2ContractError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat()


def _sha256(value: Any, *, field_name: str, required: bool = True) -> str:
    result = _text(value, field_name=field_name, required=required).lower()
    if not result and not required:
        return ""
    digest = result.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SupervisorV2ContractError(
            f"{field_name} must be a SHA-256 digest"
        )
    return f"sha256:{digest}"


def _summary_digest(summary: str) -> str:
    return "sha256:" + hashlib.sha256(summary.encode("utf-8")).hexdigest()


def _relative_path(value: Any, *, field_name: str) -> str:
    result = _text(value, field_name=field_name).replace("\\", "/")
    candidate = PurePosixPath(result)
    windows_candidate = PureWindowsPath(result)
    normalized = posixpath.normpath(result)
    if (
        candidate.is_absolute()
        or bool(windows_candidate.drive)
        or normalized in {"", ".", ".."}
        or normalized.startswith("../")
        or "//" in result
        or normalized != result
        or ".." in candidate.parts
    ):
        raise PathEscapeError(
            f"{field_name} must be a normalized repository-relative path"
        )
    return result


def _root_path(value: Any, *, field_name: str) -> str:
    result = _text(value, field_name=field_name).replace("\\", "/")
    normalized = posixpath.normpath(result)
    if (
        not result.startswith("/")
        or result == "/"
        or result.startswith("//")
        or normalized != result
        or ".." in PurePosixPath(result).parts
    ):
        raise PathEscapeError(
            f"{field_name} must be a normalized, non-root absolute path"
        )
    return result


def _payload_depth(value: Any, depth: int = 1) -> int:
    # Stop as soon as the absolute contract limit is crossed.  Besides being
    # cheaper, this prevents an adversarial deeply nested JSON value from
    # exhausting Python's recursion limit merely while we reject it.
    if depth > MAX_PAYLOAD_DEPTH:
        return depth
    if not isinstance(value, (Mapping, list, tuple)):
        return depth
    children = value.values() if isinstance(value, Mapping) else value
    return max((depth, *(_payload_depth(item, depth + 1) for item in children)))


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Iterable[str], *, artifact_name: str
) -> None:
    if not isinstance(payload, Mapping):
        raise SupervisorV2ContractError(f"{artifact_name} must be an object")
    if set(payload).difference(allowed):
        raise UnknownFieldError(
            f"{artifact_name} contains unsupported fields; rebuild its canonical payload"
        )


def _check_header(payload: Mapping[str, Any], schema: str) -> None:
    if not isinstance(payload, Mapping):
        raise SupervisorV2ContractError("generation-2 contract must be an object")
    supplied_schema = payload.get("schema")
    if supplied_schema not in (None, schema):
        raise SupervisorV2ContractError(f"unsupported schema; expected {schema}")
    for version_field in ("contract_version", "schema_version"):
        supplied_version = payload.get(version_field)
        if supplied_version not in (None, SUPERVISOR_V2_CONTRACT_VERSION):
            raise SupervisorV2ContractError(
                "unsupported generation-2 contract version"
            )


def _check_claim(
    payload: Mapping[str, Any],
    expected: str,
    *,
    names: Sequence[str] = ("content_id",),
    artifact_name: str,
) -> None:
    for name in names:
        supplied = payload.get(name)
        if supplied not in (None, "", expected):
            raise ForgedSummaryError(
                f"{artifact_name} content identity does not match canonical content"
            )


def _bounded(value: Any, *, maximum: int, artifact_name: str) -> None:
    payload = value.to_dict() if isinstance(value, CanonicalContract) else value
    if _payload_depth(payload) > MAX_PAYLOAD_DEPTH:
        raise ContractBoundsError(
            f"{artifact_name} exceeds maximum depth {MAX_PAYLOAD_DEPTH}"
        )
    size = len(canonical_json_bytes(payload))
    if size > maximum:
        raise ContractBoundsError(f"{artifact_name} exceeds {maximum} bytes")


class _V2Contract(CanonicalContract):
    @property
    def schema_version(self) -> int:
        return SUPERVISOR_V2_CONTRACT_VERSION

    @classmethod
    def from_json(cls, payload: str) -> "_V2Contract":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError, RecursionError) as exc:
            raise SupervisorV2ContractError(
                "generation-2 contract JSON is malformed"
            ) from exc
        if not isinstance(value, Mapping):
            raise SupervisorV2ContractError(
                "generation-2 contract JSON must contain an object"
            )
        _bounded(value, maximum=MAX_PROJECTION_BYTES, artifact_name=cls.__name__)
        return cls.from_dict(value)  # type: ignore[attr-defined,no-any-return]


@dataclass(frozen=True)
class SemanticDependencyIdentity(_V2Contract):
    """Identity of one input whose semantic change invalidates a result."""

    SCHEMA: ClassVar[str] = SEMANTIC_DEPENDENCY_IDENTITY_SCHEMA

    namespace: str
    key: str
    revision: str
    digest: str

    def __post_init__(self) -> None:
        for name in ("namespace", "key", "revision"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(self, "digest", _sha256(self.digest, field_name="digest"))
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="semantic dependency")

    @property
    def dependency_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "namespace": self.namespace,
            "key": self.key,
            "revision": self.revision,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticDependencyIdentity":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "namespace",
                "key",
                "revision",
                "digest",
                "dependency_id",
                "content_id",
            },
            artifact_name="semantic dependency",
        )
        result = cls(
            namespace=payload.get("namespace", ""),
            key=payload.get("key", ""),
            revision=payload.get("revision", ""),
            digest=payload.get("digest", ""),
        )
        _check_claim(
            payload,
            result.content_id,
            names=("dependency_id", "content_id"),
            artifact_name="semantic dependency",
        )
        return result


@dataclass(frozen=True)
class ResultBinding(_V2Contract):
    """Complete semantic binding required by every result-bearing record."""

    SCHEMA: ClassVar[str] = RESULT_BINDING_SCHEMA

    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    task_id: str
    task_revision: str
    policy_id: str
    policy_revision: str
    producer_id: str
    producer_revision: str
    capability_id: str
    capability_revision: str
    environment_id: str
    environment_revision: str
    semantic_dependencies: tuple[SemanticDependencyIdentity, ...]

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "task_id",
            "task_revision",
            "policy_id",
            "policy_revision",
            "producer_id",
            "producer_revision",
            "capability_id",
            "capability_revision",
            "environment_id",
            "environment_revision",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        dependencies = _records(
            self.semantic_dependencies,
            SemanticDependencyIdentity,
            field_name="semantic_dependencies",
            required=True,
            maximum=256,
        )
        keys = [(item.namespace, item.key) for item in dependencies]
        if len(keys) != len(set(keys)):
            raise SupervisorV2ContractError(
                "semantic_dependencies contains duplicate namespace/key identities"
            )
        object.__setattr__(self, "semantic_dependencies", dependencies)
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="result binding")

    @property
    def binding_id(self) -> str:
        return self.content_id

    @property
    def semantic_dependency_ids(self) -> tuple[str, ...]:
        return tuple(item.dependency_id for item in self.semantic_dependencies)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "task_id": self.task_id,
            "task_revision": self.task_revision,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "producer_id": self.producer_id,
            "producer_revision": self.producer_revision,
            "capability_id": self.capability_id,
            "capability_revision": self.capability_revision,
            "environment_id": self.environment_id,
            "environment_revision": self.environment_revision,
            "semantic_dependencies": tuple(
                item.to_record() for item in self.semantic_dependencies
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResultBinding":
        _check_header(payload, cls.SCHEMA)
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "task_id",
            "task_revision",
            "policy_id",
            "policy_revision",
            "producer_id",
            "producer_revision",
            "capability_id",
            "capability_revision",
            "environment_id",
            "environment_revision",
            "semantic_dependencies",
            "semantic_dependency_ids",
            "binding_id",
            "content_id",
        }
        _reject_unknown(payload, allowed, artifact_name="result binding")
        result = cls(
            **{
                name: payload.get(name, "")
                for name in (
                    "repository_id",
                    "tree_id",
                    "objective_id",
                    "objective_revision",
                    "task_id",
                    "task_revision",
                    "policy_id",
                    "policy_revision",
                    "producer_id",
                    "producer_revision",
                    "capability_id",
                    "capability_revision",
                    "environment_id",
                    "environment_revision",
                )
            },
            semantic_dependencies=payload.get("semantic_dependencies", ()),
        )
        claimed_dependencies = payload.get("semantic_dependency_ids")
        if claimed_dependencies is not None and tuple(claimed_dependencies) != result.semantic_dependency_ids:
            raise DetachedReferenceError(
                "semantic_dependency_ids do not match embedded dependencies"
            )
        _check_claim(
            payload,
            result.content_id,
            names=("binding_id", "content_id"),
            artifact_name="result binding",
        )
        return result


@dataclass(frozen=True)
class ArtifactBounds(_V2Contract):
    """Policy bounds for receipts, projections, references, and nesting."""

    SCHEMA: ClassVar[str] = ARTIFACT_BOUNDS_SCHEMA

    max_receipt_bytes: int = MAX_RECEIPT_BYTES
    max_projection_bytes: int = MAX_PROJECTION_BYTES
    max_reference_bytes: int = 8_192
    max_text_bytes: int = MAX_TEXT_BYTES
    max_depth: int = MAX_PAYLOAD_DEPTH
    max_references: int = 256

    def __post_init__(self) -> None:
        limits = (
            ("max_receipt_bytes", self.max_receipt_bytes, MAX_RECEIPT_BYTES),
            ("max_projection_bytes", self.max_projection_bytes, MAX_PROJECTION_BYTES),
            ("max_reference_bytes", self.max_reference_bytes, MAX_RECEIPT_BYTES),
            ("max_text_bytes", self.max_text_bytes, MAX_TEXT_BYTES),
            ("max_depth", self.max_depth, MAX_PAYLOAD_DEPTH),
            ("max_references", self.max_references, MAX_COLLECTION_ITEMS),
        )
        for name, value, maximum in limits:
            object.__setattr__(
                self,
                name,
                _integer(value, field_name=name, minimum=1, maximum=maximum),
            )
        if self.max_reference_bytes > self.max_receipt_bytes:
            raise ContractBoundsError(
                "max_reference_bytes cannot exceed max_receipt_bytes"
            )
        if self.max_receipt_bytes > self.max_projection_bytes:
            raise ContractBoundsError(
                "max_receipt_bytes cannot exceed max_projection_bytes"
            )

    def validate(self, value: Any, *, projection: bool = False) -> None:
        payload = value.to_dict() if isinstance(value, CanonicalContract) else value
        depth = _payload_depth(payload)
        if depth > self.max_depth:
            raise ContractBoundsError(
                f"payload depth {depth} exceeds configured maximum {self.max_depth}"
            )
        maximum = self.max_projection_bytes if projection else self.max_receipt_bytes
        if len(canonical_json_bytes(payload)) > maximum:
            raise ContractBoundsError(f"payload exceeds configured {maximum} byte bound")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "max_receipt_bytes": self.max_receipt_bytes,
            "max_projection_bytes": self.max_projection_bytes,
            "max_reference_bytes": self.max_reference_bytes,
            "max_text_bytes": self.max_text_bytes,
            "max_depth": self.max_depth,
            "max_references": self.max_references,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactBounds":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "max_receipt_bytes",
                "max_projection_bytes",
                "max_reference_bytes",
                "max_text_bytes",
                "max_depth",
                "max_references",
                "content_id",
            },
            artifact_name="artifact bounds",
        )
        defaults = cls()
        result = cls(
            **{
                name: payload.get(name, getattr(defaults, name))
                for name in (
                    "max_receipt_bytes",
                    "max_projection_bytes",
                    "max_reference_bytes",
                    "max_text_bytes",
                    "max_depth",
                    "max_references",
                )
            }
        )
        _check_claim(
            payload, result.content_id, artifact_name="artifact bounds"
        )
        return result


def _coerce_binding(value: ResultBinding | Mapping[str, Any]) -> ResultBinding:
    if isinstance(value, ResultBinding):
        return value
    if isinstance(value, Mapping):
        return ResultBinding.from_dict(value)
    raise SupervisorV2ContractError("binding must be a ResultBinding")


def _same_semantic_scope(first: ResultBinding, second: ResultBinding) -> bool:
    """Return whether results share the same authority-relevant input scope.

    Producer and capability identities are deliberately excluded: independent
    producers are allowed (and required for disagreement resolution), while
    repository, objective, policy, environment, task, and semantic inputs may
    never drift.
    """

    names = (
        "repository_id",
        "tree_id",
        "objective_id",
        "objective_revision",
        "task_id",
        "task_revision",
        "policy_id",
        "policy_revision",
        "environment_id",
        "environment_revision",
        "semantic_dependency_ids",
    )
    return all(getattr(first, name) == getattr(second, name) for name in names)


@dataclass(frozen=True)
class EvidenceReference(_V2Contract):
    """Bounded content-addressed evidence pointer with a verified summary."""

    SCHEMA: ClassVar[str] = EVIDENCE_REFERENCE_SCHEMA

    binding: ResultBinding
    kind: str
    authority: AuthorityClass
    artifact_uri: str
    artifact_content_id: str
    sha256: str
    byte_count: int
    media_type: str
    summary: str
    summary_sha256: str = ""
    freshness: EvidenceFreshness = EvidenceFreshness.FRESH

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding", _coerce_binding(self.binding))
        object.__setattr__(self, "kind", _text(self.kind, field_name="kind"))
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, AuthorityClass, field_name="authority"),
        )
        uri = _text(self.artifact_uri, field_name="artifact_uri")
        if "://" not in uri:
            uri = _relative_path(uri, field_name="artifact_uri")
        else:
            if not uri.startswith(("cas://", "ipfs://")):
                raise PathEscapeError(
                    "artifact_uri must be repository-relative or a cas/ipfs URI"
                )
            scheme, location = uri.split("://", 1)
            if not location:
                raise PathEscapeError(f"{scheme} artifact URI must identify content")
            parts = location.split("/")
            if (
                any(part in {"", ".", ".."} for part in parts)
                or "\\" in location
                or "?" in location
                or "#" in location
            ):
                raise PathEscapeError(
                    f"{scheme} artifact URI contains an unsafe content path"
                )
        object.__setattr__(self, "artifact_uri", uri)
        object.__setattr__(
            self,
            "artifact_content_id",
            _text(self.artifact_content_id, field_name="artifact_content_id"),
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, field_name="sha256"))
        object.__setattr__(
            self,
            "byte_count",
            _integer(self.byte_count, field_name="byte_count", minimum=1),
        )
        object.__setattr__(
            self, "media_type", _text(self.media_type, field_name="media_type")
        )
        summary = _text(
            self.summary,
            field_name="summary",
            max_bytes=4_096,
        )
        object.__setattr__(self, "summary", summary)
        expected_summary_digest = _summary_digest(summary)
        if self.summary_sha256 and _sha256(
            self.summary_sha256, field_name="summary_sha256"
        ) != expected_summary_digest:
            raise ForgedSummaryError("evidence summary digest does not match summary")
        object.__setattr__(self, "summary_sha256", expected_summary_digest)
        object.__setattr__(
            self,
            "freshness",
            _enum(self.freshness, EvidenceFreshness, field_name="freshness"),
        )
        _bounded(self, maximum=8_192, artifact_name="evidence reference")

    @property
    def reference_id(self) -> str:
        return self.content_id

    @property
    def completion_authoritative(self) -> bool:
        return (
            self.authority is AuthorityClass.COMPLETION
            and self.freshness is EvidenceFreshness.FRESH
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "kind": self.kind,
            "authority": self.authority,
            "artifact_uri": self.artifact_uri,
            "artifact_content_id": self.artifact_content_id,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
            "summary": self.summary,
            "summary_sha256": self.summary_sha256,
            "freshness": self.freshness,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceReference":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "binding",
                "kind",
                "authority",
                "artifact_uri",
                "artifact_content_id",
                "sha256",
                "byte_count",
                "media_type",
                "summary",
                "summary_sha256",
                "freshness",
                "reference_id",
                "content_id",
                "completion_authoritative",
            },
            artifact_name="evidence reference",
        )
        result = cls(
            binding=payload.get("binding", {}),
            kind=payload.get("kind", ""),
            authority=payload.get("authority", ""),
            artifact_uri=payload.get("artifact_uri", ""),
            artifact_content_id=payload.get("artifact_content_id", ""),
            sha256=payload.get("sha256", ""),
            byte_count=payload.get("byte_count", 0),
            media_type=payload.get("media_type", ""),
            summary=payload.get("summary", ""),
            summary_sha256=payload.get("summary_sha256", ""),
            freshness=payload.get("freshness", EvidenceFreshness.FRESH),
        )
        if "completion_authoritative" in payload and payload[
            "completion_authoritative"
        ] is not result.completion_authoritative:
            raise ForgedSummaryError(
                "completion_authoritative is derived from authority and freshness"
            )
        _check_claim(
            payload,
            result.content_id,
            names=("reference_id", "content_id"),
            artifact_name="evidence reference",
        )
        return result


@dataclass(frozen=True)
class StageEvent(_V2Contract):
    """One compact, causally ordered stage transition."""

    SCHEMA: ClassVar[str] = STAGE_EVENT_SCHEMA

    binding: ResultBinding
    stage: str
    attempt: int
    sequence: int
    kind: StageEventKind
    authority: AuthorityClass
    occurred_at: str | datetime
    evidence_references: tuple[EvidenceReference, ...] = ()
    reason_code: str = ""

    def __post_init__(self) -> None:
        binding = _coerce_binding(self.binding)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "stage", _text(self.stage, field_name="stage"))
        object.__setattr__(
            self, "attempt", _integer(self.attempt, field_name="attempt", minimum=1)
        )
        object.__setattr__(
            self, "sequence", _integer(self.sequence, field_name="sequence")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, StageEventKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, AuthorityClass, field_name="authority"),
        )
        object.__setattr__(
            self,
            "occurred_at",
            _timestamp(self.occurred_at, field_name="occurred_at"),
        )
        evidence = _records(
            self.evidence_references,
            EvidenceReference,
            field_name="evidence_references",
            maximum=256,
        )
        for reference in evidence:
            if reference.binding.binding_id != binding.binding_id:
                raise DetachedReferenceError(
                    "stage event evidence is detached from the event binding"
                )
            if reference.authority is not self.authority:
                raise AuthorityClassError(
                    "stage event authority must match all emitted evidence authority"
                )
        object.__setattr__(self, "evidence_references", evidence)
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, field_name="reason_code", required=False),
        )
        if self.kind in {
            StageEventKind.FAILED,
            StageEventKind.CANCELLED,
            StageEventKind.TIMED_OUT,
            StageEventKind.SKIPPED,
        } and not self.reason_code:
            raise SupervisorV2ContractError(
                f"{self.kind.value} events require reason_code"
            )
        if not self.kind.terminal and self.authority in {
            AuthorityClass.MERGE,
            AuthorityClass.MUTATION,
            AuthorityClass.COMPLETION,
        }:
            raise SupervisorV2ContractError(
                "non-terminal events cannot claim merge, mutation, or completion authority"
            )
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="stage event")

    @property
    def event_id(self) -> str:
        return self.content_id

    @property
    def evidence_reference_ids(self) -> tuple[str, ...]:
        return tuple(item.reference_id for item in self.evidence_references)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "stage": self.stage,
            "attempt": self.attempt,
            "sequence": self.sequence,
            "kind": self.kind,
            "authority": self.authority,
            "occurred_at": self.occurred_at,
            "evidence_references": tuple(
                item.to_record() for item in self.evidence_references
            ),
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageEvent":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "binding",
                "stage",
                "attempt",
                "sequence",
                "kind",
                "authority",
                "occurred_at",
                "evidence_references",
                "evidence_reference_ids",
                "reason_code",
                "event_id",
                "content_id",
            },
            artifact_name="stage event",
        )
        result = cls(
            binding=payload.get("binding", {}),
            stage=payload.get("stage", ""),
            attempt=payload.get("attempt", 0),
            sequence=payload.get("sequence", 0),
            kind=payload.get("kind", ""),
            authority=payload.get("authority", ""),
            occurred_at=payload.get("occurred_at", ""),
            evidence_references=payload.get("evidence_references", ()),
            reason_code=payload.get("reason_code", ""),
        )
        claimed = payload.get("evidence_reference_ids")
        if claimed is not None and tuple(claimed) != result.evidence_reference_ids:
            raise DetachedReferenceError(
                "stage event evidence_reference_ids do not match embedded references"
            )
        _check_claim(
            payload,
            result.content_id,
            names=("event_id", "content_id"),
            artifact_name="stage event",
        )
        return result


class AuthorityClassError(SupervisorV2ContractError):
    """A record combined incompatible semantic authority classes."""


@dataclass(frozen=True)
class StageReceipt(_V2Contract):
    """Bounded ordered stage history with no detached event or evidence IDs."""

    SCHEMA: ClassVar[str] = STAGE_RECEIPT_SCHEMA

    binding: ResultBinding
    stage: str
    attempt: int
    authority: AuthorityClass
    events: tuple[StageEvent, ...]
    summary: str
    summary_sha256: str = ""

    def __post_init__(self) -> None:
        binding = _coerce_binding(self.binding)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "stage", _text(self.stage, field_name="stage"))
        object.__setattr__(
            self, "attempt", _integer(self.attempt, field_name="attempt", minimum=1)
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, AuthorityClass, field_name="authority"),
        )
        events = _records(
            self.events,
            StageEvent,
            field_name="events",
            required=True,
            maximum=1_024,
        )
        ordered = tuple(sorted(events, key=lambda item: item.sequence))
        sequences = tuple(item.sequence for item in ordered)
        if sequences != tuple(range(len(ordered))):
            raise SupervisorV2ContractError(
                "stage receipt event sequences must be contiguous from zero"
            )
        for event in ordered:
            if (
                event.binding.binding_id != binding.binding_id
                or event.stage != self.stage
                or event.attempt != self.attempt
            ):
                raise DetachedReferenceError(
                    "stage receipt contains a foreign or detached event"
                )
            if event.authority is not self.authority:
                raise AuthorityClassError(
                    "stage receipt and event authority must match"
                )
        if any(event.kind.terminal for event in ordered[:-1]):
            raise SupervisorV2ContractError(
                "stage receipt cannot contain events after a terminal event"
            )
        if not ordered[-1].kind.terminal:
            raise SupervisorV2ContractError(
                "stage receipt must end with a terminal event"
            )
        if any(
            previous.occurred_at > current.occurred_at
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise SupervisorV2ContractError(
                "stage receipt event timestamps must be nondecreasing"
            )
        object.__setattr__(self, "events", ordered)
        summary = _text(self.summary, field_name="summary", max_bytes=4_096)
        object.__setattr__(self, "summary", summary)
        expected = _summary_digest(summary)
        if self.summary_sha256 and _sha256(
            self.summary_sha256, field_name="summary_sha256"
        ) != expected:
            raise ForgedSummaryError("stage receipt summary digest does not match")
        object.__setattr__(self, "summary_sha256", expected)
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="stage receipt")

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def terminal_kind(self) -> StageEventKind:
        return self.events[-1].kind

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "stage": self.stage,
            "attempt": self.attempt,
            "authority": self.authority,
            "events": tuple(item.to_record() for item in self.events),
            "summary": self.summary,
            "summary_sha256": self.summary_sha256,
            "terminal_kind": self.terminal_kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageReceipt":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "binding",
                "stage",
                "attempt",
                "authority",
                "events",
                "summary",
                "summary_sha256",
                "terminal_kind",
                "receipt_id",
                "content_id",
            },
            artifact_name="stage receipt",
        )
        result = cls(
            binding=payload.get("binding", {}),
            stage=payload.get("stage", ""),
            attempt=payload.get("attempt", 0),
            authority=payload.get("authority", ""),
            events=payload.get("events", ()),
            summary=payload.get("summary", ""),
            summary_sha256=payload.get("summary_sha256", ""),
        )
        if "terminal_kind" in payload and payload["terminal_kind"] != result.terminal_kind.value:
            raise ForgedSummaryError("terminal_kind does not match the final event")
        _check_claim(
            payload,
            result.content_id,
            names=("receipt_id", "content_id"),
            artifact_name="stage receipt",
        )
        return result


@dataclass(frozen=True)
class OperationCapability(_V2Contract):
    """One catalog operation and its maximum authority and safety semantics."""

    SCHEMA: ClassVar[str] = OPERATION_CAPABILITY_SCHEMA

    operation: str
    capability_id: str
    capability_revision: str
    authority: AuthorityClass
    request_schema: str
    result_schema: str
    target_kinds: tuple[TargetKind, ...]
    allowed_roots: tuple[str, ...] = ()
    max_result_bytes: int = MAX_RECEIPT_BYTES
    supports_dry_run: bool = False
    requires_idempotency: bool = False
    requires_authorization: bool = False
    requires_lease: bool = False
    requires_fencing: bool = False
    backend_capabilities: tuple[str, ...] = ()
    degradation_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "operation",
            "capability_id",
            "capability_revision",
            "request_schema",
            "result_schema",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, AuthorityClass, field_name="authority"),
        )
        if not isinstance(self.target_kinds, Sequence):
            raise SupervisorV2ContractError("target_kinds must be a sequence")
        target_kind_values = tuple(
            _enum(item, TargetKind, field_name="target_kinds")
            for item in self.target_kinds
        )
        if len(target_kind_values) != len(set(target_kind_values)):
            raise SupervisorV2ContractError("target_kinds contains duplicates")
        target_kinds = tuple(sorted(target_kind_values, key=lambda item: item.value))
        if not target_kinds:
            raise SupervisorV2ContractError("target_kinds must not be empty")
        object.__setattr__(self, "target_kinds", target_kinds)
        roots = tuple(
            sorted(
                _root_path(item, field_name="allowed_roots")
                for item in self.allowed_roots
            )
        )
        if len(set(roots)) != len(roots):
            raise SupervisorV2ContractError("allowed_roots contains duplicates")
        object.__setattr__(self, "allowed_roots", roots)
        object.__setattr__(
            self,
            "max_result_bytes",
            _integer(
                self.max_result_bytes,
                field_name="max_result_bytes",
                minimum=1,
                maximum=MAX_PROJECTION_BYTES,
            ),
        )
        for name in (
            "supports_dry_run",
            "requires_idempotency",
            "requires_authorization",
            "requires_lease",
            "requires_fencing",
        ):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "backend_capabilities",
            _strings(
                self.backend_capabilities,
                field_name="backend_capabilities",
            ),
        )
        object.__setattr__(
            self,
            "degradation_rules",
            _strings(
                self.degradation_rules,
                field_name="degradation_rules",
            ),
        )
        if self.authority is AuthorityClass.MUTATION:
            required = (
                self.supports_dry_run,
                self.requires_idempotency,
                self.requires_authorization,
                self.requires_lease,
                self.requires_fencing,
                bool(self.allowed_roots),
            )
            if not all(required):
                raise AuthorityClassError(
                    "mutation capabilities require dry-run, idempotency, "
                    "authorization, lease, fencing, and allowed roots"
                )
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="operation capability")

    @property
    def operation_capability_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "operation": self.operation,
            "capability_id": self.capability_id,
            "capability_revision": self.capability_revision,
            "authority": self.authority,
            "request_schema": self.request_schema,
            "result_schema": self.result_schema,
            "target_kinds": self.target_kinds,
            "allowed_roots": self.allowed_roots,
            "max_result_bytes": self.max_result_bytes,
            "supports_dry_run": self.supports_dry_run,
            "requires_idempotency": self.requires_idempotency,
            "requires_authorization": self.requires_authorization,
            "requires_lease": self.requires_lease,
            "requires_fencing": self.requires_fencing,
            "backend_capabilities": self.backend_capabilities,
            "degradation_rules": self.degradation_rules,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationCapability":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "operation",
            "capability_id",
            "capability_revision",
            "authority",
            "request_schema",
            "result_schema",
            "target_kinds",
            "allowed_roots",
            "max_result_bytes",
            "supports_dry_run",
            "requires_idempotency",
            "requires_authorization",
            "requires_lease",
            "requires_fencing",
            "backend_capabilities",
            "degradation_rules",
        }
        _reject_unknown(
            payload,
            fields
            | {
                "schema",
                "schema_version",
                "contract_version",
                "operation_capability_id",
                "content_id",
            },
            artifact_name="operation capability",
        )
        defaults = cls.__dataclass_fields__
        values: dict[str, Any] = {}
        for name in fields:
            field = defaults[name]
            if name in payload:
                values[name] = payload[name]
            elif field.default is not MISSING:
                values[name] = field.default
            elif field.default_factory is not MISSING:
                values[name] = field.default_factory()
            else:
                values[name] = ""
        result = cls(**values)
        _check_claim(
            payload,
            result.content_id,
            names=("operation_capability_id", "content_id"),
            artifact_name="operation capability",
        )
        return result


@dataclass(frozen=True)
class UncertaintyRecord(_V2Contract):
    """Explicit bounded uncertainty which can never masquerade as proof."""

    SCHEMA: ClassVar[str] = UNCERTAINTY_RECORD_SCHEMA

    binding: ResultBinding
    subject: str
    statement: str
    disposition: UncertaintyDisposition
    probability_lower_millionths: int
    probability_upper_millionths: int
    evidence_references: tuple[EvidenceReference, ...] = ()
    resolution_code: str = ""

    def __post_init__(self) -> None:
        binding = _coerce_binding(self.binding)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "subject", _text(self.subject, field_name="subject"))
        object.__setattr__(
            self, "statement", _text(self.statement, field_name="statement")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(
                self.disposition,
                UncertaintyDisposition,
                field_name="disposition",
            ),
        )
        for name in (
            "probability_lower_millionths",
            "probability_upper_millionths",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    maximum=MILLION,
                ),
            )
        if self.probability_lower_millionths > self.probability_upper_millionths:
            raise SupervisorV2ContractError(
                "uncertainty probability lower bound exceeds upper bound"
            )
        references = _records(
            self.evidence_references,
            EvidenceReference,
            field_name="evidence_references",
            maximum=256,
        )
        for reference in references:
            if not _same_semantic_scope(reference.binding, binding):
                raise DetachedReferenceError(
                    "uncertainty evidence is detached from its binding"
                )
            if reference.authority is AuthorityClass.COMPLETION:
                raise AuthorityClassError(
                    "uncertainty cannot consume completion authority as uncertainty"
                )
        object.__setattr__(self, "evidence_references", references)
        object.__setattr__(
            self,
            "resolution_code",
            _text(
                self.resolution_code,
                field_name="resolution_code",
                required=False,
            ),
        )
        if self.disposition is UncertaintyDisposition.RESOLVED and not self.resolution_code:
            raise SupervisorV2ContractError(
                "resolved uncertainty requires resolution_code"
            )
        if (
            self.disposition is UncertaintyDisposition.OPEN
            and self.probability_lower_millionths
            == self.probability_upper_millionths
        ):
            raise SupervisorV2ContractError(
                "open uncertainty must retain a non-zero interval"
            )
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="uncertainty record")

    @property
    def uncertainty_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "subject": self.subject,
            "statement": self.statement,
            "disposition": self.disposition,
            "probability_lower_millionths": self.probability_lower_millionths,
            "probability_upper_millionths": self.probability_upper_millionths,
            "evidence_references": tuple(
                item.to_record() for item in self.evidence_references
            ),
            "resolution_code": self.resolution_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UncertaintyRecord":
        _check_header(payload, cls.SCHEMA)
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "binding",
            "subject",
            "statement",
            "disposition",
            "probability_lower_millionths",
            "probability_upper_millionths",
            "evidence_references",
            "resolution_code",
            "uncertainty_id",
            "content_id",
        }
        _reject_unknown(payload, allowed, artifact_name="uncertainty record")
        result = cls(
            binding=payload.get("binding", {}),
            subject=payload.get("subject", ""),
            statement=payload.get("statement", ""),
            disposition=payload.get("disposition", ""),
            probability_lower_millionths=payload.get(
                "probability_lower_millionths", -1
            ),
            probability_upper_millionths=payload.get(
                "probability_upper_millionths", -1
            ),
            evidence_references=payload.get("evidence_references", ()),
            resolution_code=payload.get("resolution_code", ""),
        )
        _check_claim(
            payload,
            result.content_id,
            names=("uncertainty_id", "content_id"),
            artifact_name="uncertainty record",
        )
        return result


@dataclass(frozen=True)
class DisagreementRecord(_V2Contract):
    """Persisted disagreement between independent, provenance-bearing sources."""

    SCHEMA: ClassVar[str] = DISAGREEMENT_RECORD_SCHEMA

    binding: ResultBinding
    subject: str
    claims: tuple[EvidenceReference, ...]
    resolution: DisagreementResolution = DisagreementResolution.UNRESOLVED
    selected_reference_id: str = ""
    resolver_reference: EvidenceReference | None = None

    def __post_init__(self) -> None:
        binding = _coerce_binding(self.binding)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "subject", _text(self.subject, field_name="subject"))
        claims = _records(
            self.claims,
            EvidenceReference,
            field_name="claims",
            required=True,
            maximum=32,
        )
        if len(claims) < 2:
            raise SupervisorV2ContractError(
                "disagreement requires at least two evidence claims"
            )
        producers = {item.binding.producer_id for item in claims}
        if len(producers) < 2:
            raise SupervisorV2ContractError(
                "disagreement claims require independent producers"
            )
        for claim in claims:
            if not _same_semantic_scope(claim.binding, binding):
                raise DetachedReferenceError(
                    "disagreement claim is detached from its binding"
                )
            if claim.authority is AuthorityClass.COMPLETION:
                raise AuthorityClassError(
                    "a disagreement claim cannot itself have completion authority"
                )
        object.__setattr__(self, "claims", claims)
        object.__setattr__(
            self,
            "resolution",
            _enum(
                self.resolution,
                DisagreementResolution,
                field_name="resolution",
            ),
        )
        selected = _text(
            self.selected_reference_id,
            field_name="selected_reference_id",
            required=False,
        )
        resolver = self.resolver_reference
        if isinstance(resolver, Mapping):
            resolver = EvidenceReference.from_dict(resolver)
        if resolver is not None:
            if not isinstance(resolver, EvidenceReference):
                raise SupervisorV2ContractError(
                    "resolver_reference must be an EvidenceReference"
                )
            if not _same_semantic_scope(resolver.binding, binding):
                raise DetachedReferenceError(
                    "disagreement resolver is detached from its binding"
                )
            if resolver.authority not in {
                AuthorityClass.VALIDATION,
                AuthorityClass.PROOF,
            }:
                raise AuthorityClassError(
                    "independent resolution requires validation or proof authority"
                )
        object.__setattr__(self, "resolver_reference", resolver)
        claim_ids = {item.reference_id for item in claims}
        if self.resolution is DisagreementResolution.UNRESOLVED:
            if selected or resolver is not None:
                raise SupervisorV2ContractError(
                    "unresolved disagreement cannot select or resolve a claim"
                )
        elif self.resolution is DisagreementResolution.EXPLICIT_UNCERTAINTY:
            if selected:
                raise SupervisorV2ContractError(
                    "explicit uncertainty cannot silently select a claim"
                )
        else:
            if selected not in claim_ids:
                raise DetachedReferenceError(
                    "selected disagreement reference is not an embedded claim"
                )
            if (
                self.resolution is DisagreementResolution.INDEPENDENT_VALIDATION
                and resolver is None
            ):
                raise DetachedReferenceError(
                    "independent validation requires an embedded resolver reference"
                )
            if (
                self.resolution is DisagreementResolution.INDEPENDENT_VALIDATION
                and resolver is not None
                and resolver.binding.producer_id in producers
            ):
                raise SupervisorV2ContractError(
                    "independent validation requires a producer independent of all claims"
                )
        object.__setattr__(self, "selected_reference_id", selected)
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="disagreement record")

    @property
    def disagreement_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "subject": self.subject,
            "claims": tuple(item.to_record() for item in self.claims),
            "resolution": self.resolution,
            "selected_reference_id": self.selected_reference_id,
            "resolver_reference": (
                self.resolver_reference.to_record()
                if self.resolver_reference is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DisagreementRecord":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "binding",
                "subject",
                "claims",
                "resolution",
                "selected_reference_id",
                "resolver_reference",
                "disagreement_id",
                "content_id",
            },
            artifact_name="disagreement record",
        )
        result = cls(
            binding=payload.get("binding", {}),
            subject=payload.get("subject", ""),
            claims=payload.get("claims", ()),
            resolution=payload.get(
                "resolution", DisagreementResolution.UNRESOLVED
            ),
            selected_reference_id=payload.get("selected_reference_id", ""),
            resolver_reference=payload.get("resolver_reference"),
        )
        _check_claim(
            payload,
            result.content_id,
            names=("disagreement_id", "content_id"),
            artifact_name="disagreement record",
        )
        return result


@dataclass(frozen=True)
class PromotionVector(_V2Contract):
    """Pareto metrics plus non-compensable generation-2 safety gates."""

    SCHEMA: ClassVar[str] = PROMOTION_VECTOR_SCHEMA

    binding: ResultBinding
    safety_gates: Mapping[str, bool]
    metrics_millionths: Mapping[str, int]
    decision: PromotionDecision = PromotionDecision.SHADOW
    composite_score_millionths: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding", _coerce_binding(self.binding))
        if not isinstance(self.safety_gates, Mapping):
            raise SupervisorV2ContractError("safety_gates must be a mapping")
        gates: dict[str, bool] = {}
        for name, value in self.safety_gates.items():
            normalized_name = _text(name, field_name="safety gate")
            if normalized_name in gates:
                raise SupervisorV2ContractError(
                    "safety_gates contains duplicate normalized names"
                )
            gates[normalized_name] = _boolean(
                value, field_name=f"safety_gates.{normalized_name}"
            )
        if tuple(sorted(gates)) != NON_COMPENSABLE_GATES:
            raise SupervisorV2ContractError(
                "safety_gates must contain the exact closed non-compensable population"
            )
        object.__setattr__(
            self,
            "safety_gates",
            MappingProxyType(dict(sorted(gates.items()))),
        )
        if not isinstance(self.metrics_millionths, Mapping):
            raise SupervisorV2ContractError("metrics_millionths must be a mapping")
        metrics: dict[str, int] = {}
        for name, value in self.metrics_millionths.items():
            normalized_name = _text(name, field_name="metric name")
            if normalized_name in metrics:
                raise SupervisorV2ContractError(
                    "metrics_millionths contains duplicate normalized names"
                )
            metrics[normalized_name] = _integer(
                value,
                field_name=f"metrics_millionths.{normalized_name}",
                maximum=1_000 * MILLION,
            )
        if not metrics:
            raise SupervisorV2ContractError("metrics_millionths must not be empty")
        object.__setattr__(
            self,
            "metrics_millionths",
            MappingProxyType(dict(sorted(metrics.items()))),
        )
        object.__setattr__(
            self,
            "decision",
            _enum(self.decision, PromotionDecision, field_name="decision"),
        )
        object.__setattr__(
            self,
            "composite_score_millionths",
            _integer(
                self.composite_score_millionths,
                field_name="composite_score_millionths",
                maximum=MILLION,
            ),
        )
        if not self.hard_gates_pass:
            if (
                self.decision is not PromotionDecision.SHADOW
                or self.composite_score_millionths != 0
            ):
                raise PromotionGateError(
                    "failed safety gate is non-compensable; decision must be "
                    "shadow and composite score must be zero"
                )
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="promotion vector")

    @property
    def hard_gates_pass(self) -> bool:
        return all(self.safety_gates.values())

    @property
    def promotion_eligible(self) -> bool:
        return self.hard_gates_pass and self.decision in {
            PromotionDecision.PROVISIONAL,
            PromotionDecision.PROMOTE,
        }

    @property
    def vector_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "safety_gates": self.safety_gates,
            "metrics_millionths": self.metrics_millionths,
            "decision": self.decision,
            "composite_score_millionths": self.composite_score_millionths,
            "hard_gates_pass": self.hard_gates_pass,
            "promotion_eligible": self.promotion_eligible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionVector":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "binding",
                "safety_gates",
                "metrics_millionths",
                "decision",
                "composite_score_millionths",
                "hard_gates_pass",
                "promotion_eligible",
                "vector_id",
                "content_id",
            },
            artifact_name="promotion vector",
        )
        result = cls(
            binding=payload.get("binding", {}),
            safety_gates=payload.get("safety_gates", {}),
            metrics_millionths=payload.get("metrics_millionths", {}),
            decision=payload.get("decision", PromotionDecision.SHADOW),
            composite_score_millionths=payload.get(
                "composite_score_millionths", 0
            ),
        )
        for name in ("hard_gates_pass", "promotion_eligible"):
            if name in payload and payload[name] is not getattr(result, name):
                raise ForgedSummaryError(f"{name} is a derived promotion value")
        _check_claim(
            payload,
            result.content_id,
            names=("vector_id", "content_id"),
            artifact_name="promotion vector",
        )
        return result


@dataclass(frozen=True)
class SupervisorV2Policy(_V2Contract):
    """Versioned limits and authority policy shared by v2 producers."""

    SCHEMA: ClassVar[str] = SUPERVISOR_V2_POLICY_SCHEMA

    policy_id: str
    policy_revision: str
    artifact_bounds: ArtifactBounds = ArtifactBounds()
    allowed_authorities: tuple[AuthorityClass, ...] = tuple(AuthorityClass)
    refill_max_goals: int = MAX_REFILL_GOALS
    refill_max_tasks: int = MAX_REFILL_TASKS
    refill_cooldown_seconds: int = MIN_REFILL_COOLDOWN_SECONDS
    require_fresh_evidence: bool = True
    require_independent_disagreement_resolution: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, field_name="policy_id")
        )
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, field_name="policy_revision"),
        )
        bounds = self.artifact_bounds
        if isinstance(bounds, Mapping):
            bounds = ArtifactBounds.from_dict(bounds)
        if not isinstance(bounds, ArtifactBounds):
            raise SupervisorV2ContractError(
                "artifact_bounds must be ArtifactBounds"
            )
        object.__setattr__(self, "artifact_bounds", bounds)
        authority_values = tuple(
            _enum(item, AuthorityClass, field_name="allowed_authorities")
            for item in self.allowed_authorities
        )
        if len(authority_values) != len(set(authority_values)):
            raise SupervisorV2ContractError(
                "allowed_authorities contains duplicates"
            )
        authorities = tuple(sorted(authority_values, key=lambda item: item.value))
        if not authorities:
            raise SupervisorV2ContractError("allowed_authorities must not be empty")
        object.__setattr__(self, "allowed_authorities", authorities)
        object.__setattr__(
            self,
            "refill_max_goals",
            _integer(
                self.refill_max_goals,
                field_name="refill_max_goals",
                maximum=MAX_REFILL_GOALS,
            ),
        )
        object.__setattr__(
            self,
            "refill_max_tasks",
            _integer(
                self.refill_max_tasks,
                field_name="refill_max_tasks",
                maximum=MAX_REFILL_TASKS,
            ),
        )
        object.__setattr__(
            self,
            "refill_cooldown_seconds",
            _integer(
                self.refill_cooldown_seconds,
                field_name="refill_cooldown_seconds",
                minimum=MIN_REFILL_COOLDOWN_SECONDS,
            ),
        )
        for name in (
            "require_fresh_evidence",
            "require_independent_disagreement_resolution",
        ):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), field_name=name)
            )

    @property
    def policy_content_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "artifact_bounds": self.artifact_bounds.to_record(),
            "allowed_authorities": self.allowed_authorities,
            "refill_max_goals": self.refill_max_goals,
            "refill_max_tasks": self.refill_max_tasks,
            "refill_cooldown_seconds": self.refill_cooldown_seconds,
            "require_fresh_evidence": self.require_fresh_evidence,
            "require_independent_disagreement_resolution": (
                self.require_independent_disagreement_resolution
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorV2Policy":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "policy_id",
            "policy_revision",
            "artifact_bounds",
            "allowed_authorities",
            "refill_max_goals",
            "refill_max_tasks",
            "refill_cooldown_seconds",
            "require_fresh_evidence",
            "require_independent_disagreement_resolution",
        }
        _reject_unknown(
            payload,
            fields
            | {
                "schema",
                "schema_version",
                "contract_version",
                "policy_content_id",
                "content_id",
            },
            artifact_name="supervisor v2 policy",
        )
        defaults = cls("policy:default", "policy:default@2")
        result = cls(
            **{
                name: payload.get(name, getattr(defaults, name))
                for name in fields
            }
        )
        _check_claim(
            payload,
            result.content_id,
            names=("policy_content_id", "content_id"),
            artifact_name="supervisor v2 policy",
        )
        return result


@dataclass(frozen=True)
class TargetDescriptor(_V2Contract):
    """Exact repository/state target with normalized bounded paths."""

    SCHEMA: ClassVar[str] = TARGET_DESCRIPTOR_SCHEMA

    repository_id: str
    tree_id: str
    state_revision: str
    kind: TargetKind
    repository_root: str
    state_root: str
    relative_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("repository_id", "tree_id", "state_revision"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self, "kind", _enum(self.kind, TargetKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "repository_root",
            _root_path(self.repository_root, field_name="repository_root"),
        )
        object.__setattr__(
            self, "state_root", _root_path(self.state_root, field_name="state_root")
        )
        paths = tuple(
            sorted(
                _relative_path(item, field_name="relative_paths")
                for item in self.relative_paths
            )
        )
        if len(paths) != len(set(paths)):
            raise SupervisorV2ContractError("relative_paths contains duplicates")
        object.__setattr__(self, "relative_paths", paths)
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="target descriptor")

    @property
    def target_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "state_revision": self.state_revision,
            "kind": self.kind,
            "repository_root": self.repository_root,
            "state_root": self.state_root,
            "relative_paths": self.relative_paths,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetDescriptor":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "repository_id",
                "tree_id",
                "state_revision",
                "kind",
                "repository_root",
                "state_root",
                "relative_paths",
                "target_id",
                "content_id",
            },
            artifact_name="target descriptor",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            state_revision=payload.get("state_revision", ""),
            kind=payload.get("kind", ""),
            repository_root=payload.get("repository_root", ""),
            state_root=payload.get("state_root", ""),
            relative_paths=payload.get("relative_paths", ()),
        )
        _check_claim(
            payload,
            result.content_id,
            names=("target_id", "content_id"),
            artifact_name="target descriptor",
        )
        return result


@dataclass(frozen=True)
class TypedFailure(_V2Contract):
    """Stable, redacted failure result with deterministic retry semantics."""

    SCHEMA: ClassVar[str] = TYPED_FAILURE_SCHEMA

    binding: ResultBinding
    code: FailureCode
    authority: AuthorityClass
    retry: RetryDisposition
    reason_code: str
    public_message: str
    occurred_at: str | datetime
    evidence_references: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        binding = _coerce_binding(self.binding)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "code", _enum(self.code, FailureCode, field_name="code"))
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, AuthorityClass, field_name="authority"),
        )
        object.__setattr__(
            self, "retry", _enum(self.retry, RetryDisposition, field_name="retry")
        )
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, field_name="reason_code")
        )
        message = _text(
            self.public_message,
            field_name="public_message",
            max_bytes=512,
        )
        lowered = message.lower()
        if any(
            marker in lowered
            for marker in (
                "api_key",
                "authorization:",
                "password=",
                "private_key",
                "secret=",
                "token=",
            )
        ):
            raise SupervisorV2ContractError(
                "public_message must not contain credential material"
            )
        object.__setattr__(self, "public_message", message)
        object.__setattr__(
            self,
            "occurred_at",
            _timestamp(self.occurred_at, field_name="occurred_at"),
        )
        references = _records(
            self.evidence_references,
            EvidenceReference,
            field_name="evidence_references",
            maximum=64,
        )
        for reference in references:
            if reference.binding.binding_id != binding.binding_id:
                raise DetachedReferenceError(
                    "failure evidence is detached from its binding"
                )
        object.__setattr__(self, "evidence_references", references)
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="typed failure")

    @property
    def failure_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "code": self.code,
            "authority": self.authority,
            "retry": self.retry,
            "reason_code": self.reason_code,
            "public_message": self.public_message,
            "occurred_at": self.occurred_at,
            "evidence_references": tuple(
                item.to_record() for item in self.evidence_references
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypedFailure":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "binding",
                "code",
                "authority",
                "retry",
                "reason_code",
                "public_message",
                "occurred_at",
                "evidence_references",
                "failure_id",
                "content_id",
            },
            artifact_name="typed failure",
        )
        result = cls(
            binding=payload.get("binding", {}),
            code=payload.get("code", ""),
            authority=payload.get("authority", ""),
            retry=payload.get("retry", ""),
            reason_code=payload.get("reason_code", ""),
            public_message=payload.get("public_message", ""),
            occurred_at=payload.get("occurred_at", ""),
            evidence_references=payload.get("evidence_references", ()),
        )
        _check_claim(
            payload,
            result.content_id,
            names=("failure_id", "content_id"),
            artifact_name="typed failure",
        )
        return result


@dataclass(frozen=True)
class RefillEpoch(_V2Contract):
    """Content-addressed, bounded generation-2 successor refill epoch."""

    SCHEMA: ClassVar[str] = REFILL_EPOCH_SCHEMA

    binding: ResultBinding
    target: TargetDescriptor
    board_revision: str
    operation_catalog_id: str
    artifact_store_policy_id: str
    observation_window_start: str | datetime
    observation_window_end: str | datetime
    status: RefillEpochStatus
    successor_goal_ids: tuple[str, ...] = ()
    successor_task_ids: tuple[str, ...] = ()
    trigger_dependency_ids: tuple[str, ...] = ()
    promotion_vector: PromotionVector | None = None
    previous_epoch_id: str = ""

    def __post_init__(self) -> None:
        binding = _coerce_binding(self.binding)
        object.__setattr__(self, "binding", binding)
        target = self.target
        if isinstance(target, Mapping):
            target = TargetDescriptor.from_dict(target)
        if not isinstance(target, TargetDescriptor):
            raise SupervisorV2ContractError("target must be a TargetDescriptor")
        if (
            target.repository_id != binding.repository_id
            or target.tree_id != binding.tree_id
        ):
            raise DetachedReferenceError(
                "refill target is detached from repository/tree binding"
            )
        object.__setattr__(self, "target", target)
        for name in (
            "board_revision",
            "operation_catalog_id",
            "artifact_store_policy_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        start = _timestamp(
            self.observation_window_start, field_name="observation_window_start"
        )
        end = _timestamp(
            self.observation_window_end, field_name="observation_window_end"
        )
        if end <= start:
            raise SupervisorV2ContractError(
                "observation window end must be later than start"
            )
        object.__setattr__(self, "observation_window_start", start)
        object.__setattr__(self, "observation_window_end", end)
        object.__setattr__(
            self,
            "status",
            _enum(self.status, RefillEpochStatus, field_name="status"),
        )
        goals = _strings(
            self.successor_goal_ids,
            field_name="successor_goal_ids",
            maximum=MAX_REFILL_GOALS,
        )
        tasks = _strings(
            self.successor_task_ids,
            field_name="successor_task_ids",
            maximum=MAX_REFILL_TASKS,
        )
        object.__setattr__(self, "successor_goal_ids", goals)
        object.__setattr__(self, "successor_task_ids", tasks)
        triggers = _strings(
            self.trigger_dependency_ids,
            field_name="trigger_dependency_ids",
            required=True,
            maximum=256,
        )
        known_dependencies = set(binding.semantic_dependency_ids)
        if not set(triggers).issubset(known_dependencies):
            raise DetachedReferenceError(
                "refill trigger_dependency_ids must reference embedded semantic dependencies"
            )
        object.__setattr__(self, "trigger_dependency_ids", triggers)
        vector = self.promotion_vector
        if isinstance(vector, Mapping):
            vector = PromotionVector.from_dict(vector)
        if vector is not None:
            if not isinstance(vector, PromotionVector):
                raise SupervisorV2ContractError(
                    "promotion_vector must be PromotionVector"
                )
            if vector.binding.binding_id != binding.binding_id:
                raise DetachedReferenceError(
                    "refill promotion vector is detached from epoch binding"
                )
        object.__setattr__(self, "promotion_vector", vector)
        previous = _text(
            self.previous_epoch_id,
            field_name="previous_epoch_id",
            required=False,
        )
        object.__setattr__(self, "previous_epoch_id", previous)
        has_work = bool(goals or tasks)
        if self.status is RefillEpochStatus.HEALTHY_EXHAUSTION and has_work:
            raise SupervisorV2ContractError(
                "healthy exhaustion cannot contain successor work"
            )
        if self.status in {
            RefillEpochStatus.PROPOSED,
            RefillEpochStatus.MATERIALIZED,
        } and not has_work:
            raise SupervisorV2ContractError(
                f"{self.status.value} refill epoch requires successor work"
            )
        if self.status is RefillEpochStatus.MATERIALIZED:
            if vector is None or not vector.promotion_eligible:
                raise PromotionGateError(
                    "materialized refill requires an eligible promotion vector"
                )
        _bounded(self, maximum=MAX_RECEIPT_BYTES, artifact_name="refill epoch")

    @property
    def epoch_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SUPERVISOR_V2_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "target": self.target.to_record(),
            "board_revision": self.board_revision,
            "operation_catalog_id": self.operation_catalog_id,
            "artifact_store_policy_id": self.artifact_store_policy_id,
            "observation_window_start": self.observation_window_start,
            "observation_window_end": self.observation_window_end,
            "status": self.status,
            "successor_goal_ids": self.successor_goal_ids,
            "successor_task_ids": self.successor_task_ids,
            "trigger_dependency_ids": self.trigger_dependency_ids,
            "promotion_vector": (
                self.promotion_vector.to_record()
                if self.promotion_vector is not None
                else None
            ),
            "previous_epoch_id": self.previous_epoch_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefillEpoch":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "binding",
                "target",
                "board_revision",
                "operation_catalog_id",
                "artifact_store_policy_id",
                "observation_window_start",
                "observation_window_end",
                "status",
                "successor_goal_ids",
                "successor_task_ids",
                "trigger_dependency_ids",
                "promotion_vector",
                "previous_epoch_id",
                "epoch_id",
                "content_id",
            },
            artifact_name="refill epoch",
        )
        result = cls(
            binding=payload.get("binding", {}),
            target=payload.get("target", {}),
            board_revision=payload.get("board_revision", ""),
            operation_catalog_id=payload.get("operation_catalog_id", ""),
            artifact_store_policy_id=payload.get("artifact_store_policy_id", ""),
            observation_window_start=payload.get("observation_window_start", ""),
            observation_window_end=payload.get("observation_window_end", ""),
            status=payload.get("status", ""),
            successor_goal_ids=payload.get("successor_goal_ids", ()),
            successor_task_ids=payload.get("successor_task_ids", ()),
            trigger_dependency_ids=payload.get("trigger_dependency_ids", ()),
            promotion_vector=payload.get("promotion_vector"),
            previous_epoch_id=payload.get("previous_epoch_id", ""),
        )
        _check_claim(
            payload,
            result.content_id,
            names=("epoch_id", "content_id"),
            artifact_name="refill epoch",
        )
        return result


# Clear compatibility names for callers that use receipt/policy terminology.
SemanticDependency = SemanticDependencyIdentity
BindingIdentity = ResultBinding
EvidenceRef = EvidenceReference
V2Policy = SupervisorV2Policy
PromotionGateVector = PromotionVector
FailureReceipt = TypedFailure


def canonical_v2_json_bytes(value: Any) -> bytes:
    """Return canonical generation-2 JSON after enforcing projection bounds."""

    payload = value.to_dict() if isinstance(value, CanonicalContract) else value
    _bounded(payload, maximum=MAX_PROJECTION_BYTES, artifact_name="v2 payload")
    return canonical_json_bytes(payload)


def semantic_dependency_set_id(
    dependencies: Sequence[SemanticDependencyIdentity | Mapping[str, Any]],
) -> str:
    """Return the stable identity of a complete, order-independent dependency set."""

    normalized = _records(
        dependencies,
        SemanticDependencyIdentity,
        field_name="dependencies",
        required=True,
        maximum=256,
    )
    keys = [(item.namespace, item.key) for item in normalized]
    if len(keys) != len(set(keys)):
        raise SupervisorV2ContractError(
            "dependencies contains duplicate namespace/key identities"
        )
    return content_identity(tuple(item.to_record() for item in normalized))


__all__ = [
    "ARTIFACT_BOUNDS_SCHEMA",
    "CONTRACT_VERSION",
    "DISAGREEMENT_RECORD_SCHEMA",
    "EVIDENCE_REFERENCE_SCHEMA",
    "MAX_PAYLOAD_DEPTH",
    "MAX_PROJECTION_BYTES",
    "MAX_RECEIPT_BYTES",
    "MAX_REFILL_GOALS",
    "MAX_REFILL_TASKS",
    "NON_COMPENSABLE_GATES",
    "OPERATION_CAPABILITY_SCHEMA",
    "PROMOTION_VECTOR_SCHEMA",
    "REFILL_EPOCH_SCHEMA",
    "RESULT_BINDING_SCHEMA",
    "SCHEMA_VERSION",
    "SEMANTIC_DEPENDENCY_IDENTITY_SCHEMA",
    "STAGE_EVENT_SCHEMA",
    "STAGE_RECEIPT_SCHEMA",
    "SUPERVISOR_V2_CONTRACT_VERSION",
    "SUPERVISOR_V2_POLICY_SCHEMA",
    "TARGET_DESCRIPTOR_SCHEMA",
    "TYPED_FAILURE_SCHEMA",
    "UNCERTAINTY_RECORD_SCHEMA",
    "V2_CONTRACT_INTEGRITY_REQUIREMENT_ID",
    "ArtifactBounds",
    "AuthorityClass",
    "AuthorityClassError",
    "BindingIdentity",
    "ContractBoundsError",
    "DetachedReferenceError",
    "DisagreementRecord",
    "DisagreementResolution",
    "EvidenceFreshness",
    "EvidenceRef",
    "EvidenceReference",
    "FailureCode",
    "FailureReceipt",
    "ForgedSummaryError",
    "OperationCapability",
    "PathEscapeError",
    "PromotionDecision",
    "PromotionGateError",
    "PromotionGateVector",
    "PromotionVector",
    "RefillEpoch",
    "RefillEpochStatus",
    "ResultBinding",
    "RetryDisposition",
    "SemanticDependency",
    "SemanticDependencyIdentity",
    "StageEvent",
    "StageEventKind",
    "StageReceipt",
    "SupervisorV2ContractError",
    "SupervisorV2Policy",
    "TargetDescriptor",
    "TargetKind",
    "TypedFailure",
    "UncertaintyDisposition",
    "UncertaintyRecord",
    "UnknownFieldError",
    "V2Policy",
    "canonical_v2_json_bytes",
    "semantic_dependency_set_id",
]
