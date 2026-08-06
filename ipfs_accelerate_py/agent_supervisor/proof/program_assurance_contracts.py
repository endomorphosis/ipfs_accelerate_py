"""Immutable contracts for content-bound program-assurance evidence.

These records are the narrow serialization boundary between repository
observation, static analysis, model checking, runtime validation, ZK trace
attestation, and finding persistence.  They intentionally contain compact
facts and references only.  Source text, ASTs, proof objects, witnesses,
traces, and model output must live in immutable artifacts.

Claim levels are a closed, non-hierarchical vocabulary.  In particular,
``zk_trace_attested`` establishes only the configured trace statement and can
never be decoded or promoted as semantic proof.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, ClassVar, Final, TypeVar

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


PROGRAM_ASSURANCE_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = PROGRAM_ASSURANCE_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = PROGRAM_ASSURANCE_CONTRACT_VERSION

MAX_TEXT_BYTES: Final[int] = 8_192
MAX_CLAUSE_BYTES: Final[int] = 4_096
MAX_COLLECTION_ITEMS: Final[int] = 256
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_STAGE_RECEIPT_BYTES: Final[int] = 1_048_576
MILLION: Final[int] = 1_000_000

ARTIFACT_REFERENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-assurance/artifact-reference@1"
)
REPOSITORY_OBSERVATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-assurance/repository-observation@1"
)
EXPECTED_CONTRACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-assurance/expected-contract@1"
)
OBSERVED_CONTRACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-assurance/observed-contract@1"
)
COUNTEREXAMPLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-assurance/counterexample@1"
)
ASSURANCE_CLAIM_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-assurance/claim@1"
)
FINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-assurance/finding@1"
)
ASSURANCE_LIMITS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-assurance/limits@1"
)
STAGE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-assurance/stage-receipt@1"
)


class ProgramAssuranceContractError(ContractValidationError):
    """Base error for malformed or unsafe program-assurance records."""


class ContractBoundsError(ProgramAssuranceContractError):
    """A compact record exceeded an explicit item, text, or byte bound."""


class ForgedIdentityError(ProgramAssuranceContractError):
    """A caller-supplied identity or derived projection was forged."""


class StaleAuthorityError(ProgramAssuranceContractError):
    """Stale evidence was presented as current authority."""


class ClaimPromotionError(ProgramAssuranceContractError):
    """One claim level was incorrectly treated as implying another."""


class SemanticAuthorityError(ProgramAssuranceContractError):
    """Evidence was presented outside the semantics it can establish."""


class ClaimLevel(str, Enum):
    """Exact, intentionally non-ordered program-assurance claim classes."""

    OBSERVED_SYNTAX = "observed_syntax"
    RESOLVED_STATIC = "resolved_static"
    MODEL_PROVED = "model_proved"
    MODEL_DISPROVED = "model_disproved"
    RUNTIME_WITNESSED = "runtime_witnessed"
    ZK_TRACE_ATTESTED = "zk_trace_attested"

    def permits(self, required: "ClaimLevel | str") -> bool:
        """Return true only for the same exact claim level."""

        return self is _enum(required, ClaimLevel, field_name="required")

    def require(self, required: "ClaimLevel | str") -> None:
        if not self.permits(required):
            raise ClaimPromotionError(
                f"{self.value} cannot be promoted to "
                f"{_enum(required, ClaimLevel, field_name='required').value}"
            )


class InconclusiveState(str, Enum):
    """Why a record cannot carry conclusive semantic authority."""

    NONE = "none"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    STALE = "stale"
    NEGATIVE_RESULT = "negative_result"
    CONFLICTING_EXPECTATIONS = "conflicting_expectations"
    UNVERIFIED_TRANSLATION = "unverified_translation"

    @property
    def conclusive(self) -> bool:
        return self is InconclusiveState.NONE


class EvidenceFreshness(str, Enum):
    CURRENT = "current"
    STALE = "stale"


class ClaimVerdict(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    INCONCLUSIVE = "inconclusive"

    @property
    def conclusive(self) -> bool:
        return self is not ClaimVerdict.INCONCLUSIVE


class AuthorityKind(str, Enum):
    """Exact producer boundary required by a claim level."""

    PARSER = "parser"
    STATIC_RESOLVER = "static_resolver"
    PROOF_KERNEL = "proof_kernel"
    RUNTIME_RUNNER = "runtime_runner"
    ZK_VERIFIER = "zk_verifier"


class ContractPrecedence(str, Enum):
    REVIEWED_INTERFACE = "reviewed_interface"
    PUBLIC_SIGNATURE = "public_signature"
    CONTRACT_TEST = "contract_test"
    NORMATIVE_DOCUMENTATION = "normative_documentation"
    COMPATIBILITY_MANIFEST = "compatibility_manifest"


class FindingStatus(str, Enum):
    CONTRACT_BROKEN = "contract_broken"
    SUSPECTED = "suspected"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"
    STALE = "stale"


class FindingSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StageStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"
    NEGATIVE = "negative"

    @property
    def successful(self) -> bool:
        return self is StageStatus.COMPLETED


T = TypeVar("T")


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise ProgramAssuranceContractError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise ProgramAssuranceContractError(f"{field_name} must not be empty")
    if "\x00" in result:
        raise ProgramAssuranceContractError(f"{field_name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise ContractBoundsError(
            f"{field_name} exceeds {maximum} UTF-8 bytes"
        )
    return result


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramAssuranceContractError(f"{field_name} must be a boolean")
    return value


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramAssuranceContractError(f"{field_name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise ContractBoundsError(
            f"{field_name} must be at least {minimum}{suffix}"
        )
    return value


def _enum(value: Any, enum_type: type[T], *, field_name: str) -> T:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(raw)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ProgramAssuranceContractError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _strings(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    preserve_order: bool = False,
    maximum: int = MAX_COLLECTION_ITEMS,
    item_bytes: int = MAX_TEXT_BYTES,
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ProgramAssuranceContractError(f"{field_name} must be a sequence")
    if len(values) > maximum:
        raise ContractBoundsError(f"{field_name} exceeds {maximum} items")
    result: list[str] = []
    for index, value in enumerate(values):
        item = _text(
            value,
            field_name=f"{field_name}[{index}]",
            maximum=item_bytes,
        )
        if item in result:
            raise ProgramAssuranceContractError(
                f"{field_name} must not contain duplicates"
            )
        result.append(item)
    if required and not result:
        raise ProgramAssuranceContractError(f"{field_name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _relative_paths(values: Any, *, field_name: str) -> tuple[str, ...]:
    paths = _strings(values, field_name=field_name)
    result: list[str] = []
    for value in paths:
        normalized = value.replace("\\", "/")
        posix = PurePosixPath(normalized)
        windows = PureWindowsPath(normalized)
        if (
            posix.is_absolute()
            or bool(windows.drive)
            or ".." in posix.parts
            or normalized.startswith("//")
        ):
            raise ProgramAssuranceContractError(
                f"{field_name} must contain repository-relative paths"
            )
        result.append(normalized)
    return tuple(sorted(result))


def _timestamp(value: Any, *, field_name: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ProgramAssuranceContractError(
                f"{field_name} must be an ISO-8601 timestamp"
            ) from exc
    else:
        raise ProgramAssuranceContractError(
            f"{field_name} must be a datetime or ISO-8601 string"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProgramAssuranceContractError(
            f"{field_name} must be timezone-aware"
        )
    return parsed.astimezone(timezone.utc).isoformat()


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _sha256(value: Any, *, field_name: str, required: bool = False) -> str:
    result = _text(value, field_name=field_name, required=required).lower()
    if not result:
        return ""
    digest = result.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ProgramAssuranceContractError(
            f"{field_name} must be a SHA-256 digest"
        )
    return f"sha256:{digest}"


def _check_header(payload: Mapping[str, Any], expected_schema: str) -> None:
    if not isinstance(payload, Mapping):
        raise ProgramAssuranceContractError("contract payload must be an object")
    if payload.get("schema") not in (None, "", expected_schema):
        raise ProgramAssuranceContractError(
            f"unsupported schema; expected {expected_schema}"
        )
    version = payload.get("contract_version", payload.get("schema_version"))
    if version not in (None, PROGRAM_ASSURANCE_CONTRACT_VERSION):
        raise ProgramAssuranceContractError(
            "unsupported program-assurance contract version"
        )


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: Iterable[str],
    *,
    artifact_name: str,
) -> None:
    if set(payload).difference(allowed):
        raise ProgramAssuranceContractError(
            f"{artifact_name} contains unsupported fields"
        )


def _check_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise ForgedIdentityError(
                f"{artifact_name} content identity does not match payload"
            )


def _bounded(
    value: CanonicalContract,
    *,
    maximum: int = MAX_RECORD_BYTES,
    artifact_name: str,
) -> None:
    if len(value.canonical_bytes()) > maximum:
        raise ContractBoundsError(
            f"{artifact_name} exceeds {maximum} canonical bytes"
        )


def _record(
    value: Any,
    record_type: type[T],
    *,
    field_name: str,
    optional: bool = False,
) -> T | None:
    if value is None and optional:
        return None
    if isinstance(value, record_type):
        return value
    if isinstance(value, Mapping):
        return record_type.from_dict(value)
    raise ProgramAssuranceContractError(
        f"{field_name} must be a {record_type.__name__} record"
    )


def _records(
    values: Any,
    record_type: type[T],
    *,
    field_name: str,
    maximum: int = MAX_COLLECTION_ITEMS,
) -> tuple[T, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise ProgramAssuranceContractError(f"{field_name} must be a sequence")
    if len(values) > maximum:
        raise ContractBoundsError(f"{field_name} exceeds {maximum} items")
    normalized = tuple(
        _record(item, record_type, field_name=field_name) for item in values
    )
    identities = tuple(item.content_id for item in normalized)
    if len(identities) != len(set(identities)):
        raise ProgramAssuranceContractError(
            f"{field_name} contains duplicate identities"
        )
    return tuple(sorted(normalized, key=lambda item: item.content_id))


def _same_scope(*values: Any) -> bool:
    if len(values) < 2:
        return True
    fields = ("repository_id", "tree_id", "symbol", "interface", "policy_revision")
    first = values[0]
    return all(
        all(getattr(value, name) == getattr(first, name) for name in fields)
        for value in values[1:]
    )


def _verify_projection(
    payload: Mapping[str, Any],
    *,
    name: str,
    actual: bool,
    stale: bool = False,
) -> None:
    if name in payload and (
        not isinstance(payload[name], bool) or payload[name] is not actual
    ):
        if stale and bool(payload[name]):
            raise StaleAuthorityError("stale evidence cannot carry authority")
        raise ForgedIdentityError(f"{name} does not match derived state")


class _AssuranceContract(CanonicalContract):
    """Shared record helpers without any ambient-state behavior."""

    @property
    def schema_version(self) -> int:
        return PROGRAM_ASSURANCE_CONTRACT_VERSION


@dataclass(frozen=True)
class ArtifactReference(_AssuranceContract):
    """Compact reference to immutable bytes stored outside these contracts."""

    SCHEMA: ClassVar[str] = ARTIFACT_REFERENCE_SCHEMA

    artifact_id: str
    kind: str
    content_cid: str = ""
    sha256: str = ""
    media_type: str = "application/octet-stream"
    byte_count: int = 0
    uri: str = ""

    def __post_init__(self) -> None:
        for name, required in (
            ("artifact_id", True),
            ("kind", True),
            ("content_cid", False),
            ("media_type", True),
            ("uri", False),
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=required),
            )
        object.__setattr__(
            self, "sha256", _sha256(self.sha256, field_name="sha256")
        )
        object.__setattr__(
            self,
            "byte_count",
            _integer(self.byte_count, field_name="byte_count"),
        )
        if not (self.content_cid or self.sha256):
            raise ProgramAssuranceContractError(
                "artifact references require content_cid or sha256"
            )
        _bounded(self, artifact_name="artifact reference")

    @property
    def reference_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "content_cid": self.content_cid,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "byte_count": self.byte_count,
            "uri": self.uri,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "reference_id": self.reference_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactReference":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "artifact_id",
                "kind",
                "content_cid",
                "cid",
                "sha256",
                "media_type",
                "byte_count",
                "uri",
                "reference_id",
                "content_id",
            },
            artifact_name="artifact reference",
        )
        result = cls(
            artifact_id=payload.get("artifact_id", ""),
            kind=payload.get("kind", ""),
            content_cid=payload.get("content_cid", payload.get("cid", "")),
            sha256=payload.get("sha256", ""),
            media_type=payload.get("media_type", "application/octet-stream"),
            byte_count=payload.get("byte_count", 0),
            uri=payload.get("uri", ""),
        )
        _check_identity(
            payload,
            result.reference_id,
            names=("reference_id", "content_id"),
            artifact_name="artifact reference",
        )
        return result


@dataclass(frozen=True)
class RepositoryObservation(_AssuranceContract):
    """One time-bounded observation of an independently identified Git tree."""

    SCHEMA: ClassVar[str] = REPOSITORY_OBSERVATION_SCHEMA

    repository_id: str
    tree_id: str
    resolved_root: str
    commit_id: str
    observed_at: str
    authority_expires_at: str
    analyzer_id: str
    analyzer_version: str
    policy_revision: str
    remote_id: str = ""
    dirty: bool = False
    dirty_diff_digest: str = ""
    gitlink_tree_ids: tuple[str, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        for name, required in (
            ("repository_id", True),
            ("tree_id", True),
            ("resolved_root", True),
            ("commit_id", True),
            ("analyzer_id", True),
            ("analyzer_version", True),
            ("policy_revision", True),
            ("remote_id", False),
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=required),
            )
        root = self.resolved_root.replace("\\", "/")
        if (
            not PurePosixPath(root).is_absolute()
            or root == "/"
            or ".." in PurePosixPath(root).parts
        ):
            raise ProgramAssuranceContractError(
                "resolved_root must be a normalized, non-root absolute path"
            )
        object.__setattr__(self, "resolved_root", root)
        object.__setattr__(
            self, "observed_at", _timestamp(self.observed_at, field_name="observed_at")
        )
        object.__setattr__(
            self,
            "authority_expires_at",
            _timestamp(
                self.authority_expires_at, field_name="authority_expires_at"
            ),
        )
        if _datetime(self.authority_expires_at) <= _datetime(self.observed_at):
            raise ProgramAssuranceContractError(
                "authority_expires_at must be later than observed_at"
            )
        object.__setattr__(self, "dirty", _boolean(self.dirty, field_name="dirty"))
        object.__setattr__(
            self,
            "dirty_diff_digest",
            _sha256(
                self.dirty_diff_digest,
                field_name="dirty_diff_digest",
                required=self.dirty,
            ),
        )
        if not self.dirty and self.dirty_diff_digest:
            raise ProgramAssuranceContractError(
                "clean observations cannot carry dirty_diff_digest"
            )
        object.__setattr__(
            self,
            "gitlink_tree_ids",
            _strings(self.gitlink_tree_ids, field_name="gitlink_tree_ids"),
        )
        object.__setattr__(
            self,
            "artifacts",
            _records(self.artifacts, ArtifactReference, field_name="artifacts"),
        )
        _bounded(self, artifact_name="repository observation")

    @property
    def observation_id(self) -> str:
        return self.content_id

    def freshness_at(self, evaluated_at: str | datetime) -> EvidenceFreshness:
        evaluated = _datetime(_timestamp(evaluated_at, field_name="evaluated_at"))
        return (
            EvidenceFreshness.CURRENT
            if _datetime(self.observed_at)
            <= evaluated
            < _datetime(self.authority_expires_at)
            else EvidenceFreshness.STALE
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "resolved_root": self.resolved_root,
            "remote_id": self.remote_id,
            "commit_id": self.commit_id,
            "dirty": self.dirty,
            "dirty_diff_digest": self.dirty_diff_digest,
            "gitlink_tree_ids": self.gitlink_tree_ids,
            "observed_at": self.observed_at,
            "authority_expires_at": self.authority_expires_at,
            "analyzer_id": self.analyzer_id,
            "analyzer_version": self.analyzer_version,
            "policy_revision": self.policy_revision,
            "artifacts": tuple(item.to_record() for item in self.artifacts),
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "observation_id": self.observation_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepositoryObservation":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "repository_id",
            "tree_id",
            "resolved_root",
            "remote_id",
            "commit_id",
            "dirty",
            "dirty_diff_digest",
            "gitlink_tree_ids",
            "observed_at",
            "authority_expires_at",
            "analyzer_id",
            "analyzer_version",
            "policy_revision",
            "artifacts",
        }
        _reject_unknown(
            payload,
            fields
            | {
                "schema",
                "schema_version",
                "contract_version",
                "observation_id",
                "content_id",
            },
            artifact_name="repository observation",
        )
        result = cls(
            **{
                name: payload.get(name, ())
                if name in {"gitlink_tree_ids", "artifacts"}
                else payload.get(name, False)
                if name == "dirty"
                else payload.get(name, "")
                for name in fields
            }
        )
        _check_identity(
            payload,
            result.observation_id,
            names=("observation_id", "content_id"),
            artifact_name="repository observation",
        )
        return result


@dataclass(frozen=True)
class ExpectedContract(_AssuranceContract):
    """Typed expectation selected under an explicit precedence policy."""

    SCHEMA: ClassVar[str] = EXPECTED_CONTRACT_SCHEMA

    repository_id: str
    tree_id: str
    symbol: str
    interface: str
    policy_revision: str
    precedence: ContractPrecedence
    summary: str
    clauses: tuple[str, ...]
    source_artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "symbol",
            "interface",
            "policy_revision",
            "summary",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "precedence",
            _enum(self.precedence, ContractPrecedence, field_name="precedence"),
        )
        object.__setattr__(
            self,
            "clauses",
            _strings(
                self.clauses,
                field_name="clauses",
                required=True,
                preserve_order=True,
                maximum=64,
                item_bytes=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "source_artifact_ids",
            _strings(
                self.source_artifact_ids,
                field_name="source_artifact_ids",
                required=True,
            ),
        )
        _bounded(self, artifact_name="expected contract")

    @property
    def expected_contract_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "symbol": self.symbol,
            "interface": self.interface,
            "policy_revision": self.policy_revision,
            "precedence": self.precedence,
            "summary": self.summary,
            "clauses": self.clauses,
            "source_artifact_ids": self.source_artifact_ids,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "expected_contract_id": self.expected_contract_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpectedContract":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "repository_id",
            "tree_id",
            "symbol",
            "interface",
            "policy_revision",
            "precedence",
            "summary",
            "clauses",
            "source_artifact_ids",
        }
        _reject_unknown(
            payload,
            fields
            | {
                "schema",
                "schema_version",
                "contract_version",
                "expected_contract_id",
                "content_id",
            },
            artifact_name="expected contract",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            symbol=payload.get("symbol", ""),
            interface=payload.get("interface", ""),
            policy_revision=payload.get("policy_revision", ""),
            precedence=payload.get("precedence", ""),
            summary=payload.get("summary", ""),
            clauses=tuple(payload.get("clauses") or ()),
            source_artifact_ids=tuple(payload.get("source_artifact_ids") or ()),
        )
        _check_identity(
            payload,
            result.expected_contract_id,
            names=("expected_contract_id", "content_id"),
            artifact_name="expected contract",
        )
        return result


@dataclass(frozen=True)
class ObservedContract(_AssuranceContract):
    """Compact behavior observed at one exact repository observation."""

    SCHEMA: ClassVar[str] = OBSERVED_CONTRACT_SCHEMA

    repository_id: str
    tree_id: str
    symbol: str
    interface: str
    policy_revision: str
    repository_observation_id: str
    summary: str
    clauses: tuple[str, ...]
    source_artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "symbol",
            "interface",
            "policy_revision",
            "repository_observation_id",
            "summary",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "clauses",
            _strings(
                self.clauses,
                field_name="clauses",
                required=True,
                preserve_order=True,
                maximum=64,
                item_bytes=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "source_artifact_ids",
            _strings(
                self.source_artifact_ids,
                field_name="source_artifact_ids",
                required=True,
            ),
        )
        _bounded(self, artifact_name="observed contract")

    @property
    def observed_contract_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "symbol": self.symbol,
            "interface": self.interface,
            "policy_revision": self.policy_revision,
            "repository_observation_id": self.repository_observation_id,
            "summary": self.summary,
            "clauses": self.clauses,
            "source_artifact_ids": self.source_artifact_ids,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "observed_contract_id": self.observed_contract_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservedContract":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "repository_id",
            "tree_id",
            "symbol",
            "interface",
            "policy_revision",
            "repository_observation_id",
            "summary",
            "clauses",
            "source_artifact_ids",
        }
        _reject_unknown(
            payload,
            fields
            | {
                "schema",
                "schema_version",
                "contract_version",
                "observed_contract_id",
                "content_id",
            },
            artifact_name="observed contract",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            symbol=payload.get("symbol", ""),
            interface=payload.get("interface", ""),
            policy_revision=payload.get("policy_revision", ""),
            repository_observation_id=payload.get(
                "repository_observation_id", ""
            ),
            summary=payload.get("summary", ""),
            clauses=tuple(payload.get("clauses") or ()),
            source_artifact_ids=tuple(payload.get("source_artifact_ids") or ()),
        )
        _check_identity(
            payload,
            result.observed_contract_id,
            names=("observed_contract_id", "content_id"),
            artifact_name="observed contract",
        )
        return result


@dataclass(frozen=True)
class Counterexample(_AssuranceContract):
    """A bounded contradiction whose detailed witness is artifact-addressed."""

    SCHEMA: ClassVar[str] = COUNTEREXAMPLE_SCHEMA

    repository_id: str
    tree_id: str
    symbol: str
    interface: str
    policy_revision: str
    expected_contract_id: str
    observed_contract_id: str
    summary: str
    witness_steps: tuple[str, ...]
    artifacts: tuple[ArtifactReference, ...]
    evaluated_at: str
    authority_expires_at: str
    conclusive: bool = True

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "symbol",
            "interface",
            "policy_revision",
            "expected_contract_id",
            "observed_contract_id",
            "summary",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "witness_steps",
            _strings(
                self.witness_steps,
                field_name="witness_steps",
                required=True,
                preserve_order=True,
                maximum=64,
                item_bytes=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "artifacts",
            _records(self.artifacts, ArtifactReference, field_name="artifacts"),
        )
        if not self.artifacts:
            raise ProgramAssuranceContractError(
                "counterexamples require an immutable witness artifact"
            )
        object.__setattr__(
            self, "evaluated_at", _timestamp(self.evaluated_at, field_name="evaluated_at")
        )
        object.__setattr__(
            self,
            "authority_expires_at",
            _timestamp(
                self.authority_expires_at, field_name="authority_expires_at"
            ),
        )
        object.__setattr__(
            self, "conclusive", _boolean(self.conclusive, field_name="conclusive")
        )
        if self.conclusive and self.freshness is EvidenceFreshness.STALE:
            raise StaleAuthorityError(
                "a conclusive counterexample cannot have stale authority"
            )
        _bounded(self, artifact_name="counterexample")

    @property
    def freshness(self) -> EvidenceFreshness:
        return (
            EvidenceFreshness.CURRENT
            if _datetime(self.evaluated_at) < _datetime(self.authority_expires_at)
            else EvidenceFreshness.STALE
        )

    @property
    def counterexample_id(self) -> str:
        return self.content_id

    @property
    def authoritative(self) -> bool:
        return self.conclusive and self.freshness is EvidenceFreshness.CURRENT

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "symbol": self.symbol,
            "interface": self.interface,
            "policy_revision": self.policy_revision,
            "expected_contract_id": self.expected_contract_id,
            "observed_contract_id": self.observed_contract_id,
            "summary": self.summary,
            "witness_steps": self.witness_steps,
            "artifacts": tuple(item.to_record() for item in self.artifacts),
            "evaluated_at": self.evaluated_at,
            "authority_expires_at": self.authority_expires_at,
            "conclusive": self.conclusive,
            "freshness": self.freshness,
            "authoritative": self.authoritative,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "counterexample_id": self.counterexample_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Counterexample":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "repository_id",
            "tree_id",
            "symbol",
            "interface",
            "policy_revision",
            "expected_contract_id",
            "observed_contract_id",
            "summary",
            "witness_steps",
            "artifacts",
            "evaluated_at",
            "authority_expires_at",
            "conclusive",
        }
        _reject_unknown(
            payload,
            fields
            | {
                "schema",
                "schema_version",
                "contract_version",
                "freshness",
                "authoritative",
                "counterexample_id",
                "content_id",
            },
            artifact_name="counterexample",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            symbol=payload.get("symbol", ""),
            interface=payload.get("interface", ""),
            policy_revision=payload.get("policy_revision", ""),
            expected_contract_id=payload.get("expected_contract_id", ""),
            observed_contract_id=payload.get("observed_contract_id", ""),
            summary=payload.get("summary", ""),
            witness_steps=tuple(payload.get("witness_steps") or ()),
            artifacts=tuple(payload.get("artifacts") or ()),
            evaluated_at=payload.get("evaluated_at", ""),
            authority_expires_at=payload.get("authority_expires_at", ""),
            conclusive=payload.get("conclusive", True),
        )
        if "freshness" in payload and _enum(
            payload["freshness"], EvidenceFreshness, field_name="freshness"
        ) is not result.freshness:
            raise ForgedIdentityError("counterexample freshness does not match timestamps")
        _verify_projection(
            payload,
            name="authoritative",
            actual=result.authoritative,
            stale=result.freshness is EvidenceFreshness.STALE,
        )
        _check_identity(
            payload,
            result.counterexample_id,
            names=("counterexample_id", "content_id"),
            artifact_name="counterexample",
        )
        return result


_LEVEL_AUTHORITY: Final[dict[ClaimLevel, AuthorityKind]] = {
    ClaimLevel.OBSERVED_SYNTAX: AuthorityKind.PARSER,
    ClaimLevel.RESOLVED_STATIC: AuthorityKind.STATIC_RESOLVER,
    ClaimLevel.MODEL_PROVED: AuthorityKind.PROOF_KERNEL,
    ClaimLevel.MODEL_DISPROVED: AuthorityKind.PROOF_KERNEL,
    ClaimLevel.RUNTIME_WITNESSED: AuthorityKind.RUNTIME_RUNNER,
    ClaimLevel.ZK_TRACE_ATTESTED: AuthorityKind.ZK_VERIFIER,
}


def validate_claim_promotion(
    source: ClaimLevel | str, target: ClaimLevel | str
) -> None:
    """Reject treating one claim class as authority for a different class."""

    _enum(source, ClaimLevel, field_name="source").require(
        _enum(target, ClaimLevel, field_name="target")
    )


@dataclass(frozen=True)
class AssuranceClaim(_AssuranceContract):
    """One exact, bounded claim with independently derived authority."""

    SCHEMA: ClassVar[str] = ASSURANCE_CLAIM_SCHEMA

    repository_id: str
    tree_id: str
    symbol: str
    interface: str
    policy_revision: str
    repository_observation_id: str
    claim_level: ClaimLevel
    verdict: ClaimVerdict
    inconclusive_state: InconclusiveState
    authority_kind: AuthorityKind
    producer_id: str
    producer_version: str
    evaluated_at: str
    authority_expires_at: str
    expected_contract_id: str = ""
    observed_contract_id: str = ""
    counterexample_id: str = ""
    assumptions: tuple[str, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()
    confidence_millionths: int = 0
    source_claim_level: ClaimLevel | None = None
    semantic_proof: bool = False

    def __post_init__(self) -> None:
        for name, required in (
            ("repository_id", True),
            ("tree_id", True),
            ("symbol", True),
            ("interface", True),
            ("policy_revision", True),
            ("repository_observation_id", True),
            ("producer_id", True),
            ("producer_version", True),
            ("expected_contract_id", False),
            ("observed_contract_id", False),
            ("counterexample_id", False),
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=required),
            )
        object.__setattr__(
            self,
            "claim_level",
            _enum(self.claim_level, ClaimLevel, field_name="claim_level"),
        )
        object.__setattr__(
            self,
            "verdict",
            _enum(self.verdict, ClaimVerdict, field_name="verdict"),
        )
        object.__setattr__(
            self,
            "inconclusive_state",
            _enum(
                self.inconclusive_state,
                InconclusiveState,
                field_name="inconclusive_state",
            ),
        )
        object.__setattr__(
            self,
            "authority_kind",
            _enum(self.authority_kind, AuthorityKind, field_name="authority_kind"),
        )
        expected_authority = _LEVEL_AUTHORITY[self.claim_level]
        if self.authority_kind is not expected_authority:
            raise SemanticAuthorityError(
                f"{self.claim_level.value} requires {expected_authority.value} authority"
            )
        object.__setattr__(
            self, "evaluated_at", _timestamp(self.evaluated_at, field_name="evaluated_at")
        )
        object.__setattr__(
            self,
            "authority_expires_at",
            _timestamp(
                self.authority_expires_at, field_name="authority_expires_at"
            ),
        )
        object.__setattr__(
            self,
            "assumptions",
            _strings(self.assumptions, field_name="assumptions", maximum=64),
        )
        object.__setattr__(
            self,
            "artifacts",
            _records(self.artifacts, ArtifactReference, field_name="artifacts"),
        )
        if not self.artifacts:
            raise ProgramAssuranceContractError(
                "assurance claims require at least one evidence artifact"
            )
        object.__setattr__(
            self,
            "confidence_millionths",
            _integer(
                self.confidence_millionths,
                field_name="confidence_millionths",
                maximum=MILLION,
            ),
        )
        if self.source_claim_level is not None:
            source = _enum(
                self.source_claim_level,
                ClaimLevel,
                field_name="source_claim_level",
            )
            validate_claim_promotion(source, self.claim_level)
            object.__setattr__(self, "source_claim_level", source)
        object.__setattr__(
            self,
            "semantic_proof",
            _boolean(self.semantic_proof, field_name="semantic_proof"),
        )
        if (
            self.claim_level is ClaimLevel.ZK_TRACE_ATTESTED
            and self.semantic_proof
        ):
            raise SemanticAuthorityError(
                "a ZK trace attestation cannot be presented as semantic proof"
            )
        if self.semantic_proof and self.claim_level not in {
            ClaimLevel.MODEL_PROVED,
            ClaimLevel.MODEL_DISPROVED,
            ClaimLevel.RUNTIME_WITNESSED,
        }:
            raise SemanticAuthorityError(
                f"{self.claim_level.value} cannot be presented as semantic proof"
            )
        if self.verdict.conclusive != self.inconclusive_state.conclusive:
            raise ProgramAssuranceContractError(
                "verdict and inconclusive_state disagree"
            )
        if self.freshness is EvidenceFreshness.STALE:
            if self.verdict.conclusive or self.inconclusive_state is not InconclusiveState.STALE:
                raise StaleAuthorityError(
                    "stale claims must be explicitly inconclusive with state stale"
                )
        if self.claim_level is ClaimLevel.MODEL_PROVED:
            if (
                self.verdict is not ClaimVerdict.SATISFIED
                or not self.semantic_proof
            ):
                raise SemanticAuthorityError(
                    "model_proved requires a satisfied semantic proof"
                )
        if self.claim_level is ClaimLevel.MODEL_DISPROVED:
            if (
                self.verdict is not ClaimVerdict.VIOLATED
                or not self.semantic_proof
                or not self.counterexample_id
            ):
                raise SemanticAuthorityError(
                    "model_disproved requires a semantic violation and counterexample"
                )
        _bounded(self, artifact_name="assurance claim")

    @property
    def freshness(self) -> EvidenceFreshness:
        return (
            EvidenceFreshness.CURRENT
            if _datetime(self.evaluated_at) < _datetime(self.authority_expires_at)
            else EvidenceFreshness.STALE
        )

    @property
    def authoritative(self) -> bool:
        return (
            self.verdict.conclusive
            and self.inconclusive_state is InconclusiveState.NONE
            and self.freshness is EvidenceFreshness.CURRENT
        )

    @property
    def semantic_authority(self) -> bool:
        return self.authoritative and self.semantic_proof and (
            self.claim_level is not ClaimLevel.ZK_TRACE_ATTESTED
        )

    @property
    def claim_id(self) -> str:
        return self.content_id

    def require_level(self, required: ClaimLevel | str) -> None:
        self.claim_level.require(required)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "symbol": self.symbol,
            "interface": self.interface,
            "policy_revision": self.policy_revision,
            "repository_observation_id": self.repository_observation_id,
            "claim_level": self.claim_level,
            "verdict": self.verdict,
            "inconclusive_state": self.inconclusive_state,
            "authority_kind": self.authority_kind,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "evaluated_at": self.evaluated_at,
            "authority_expires_at": self.authority_expires_at,
            "expected_contract_id": self.expected_contract_id,
            "observed_contract_id": self.observed_contract_id,
            "counterexample_id": self.counterexample_id,
            "assumptions": self.assumptions,
            "artifacts": tuple(item.to_record() for item in self.artifacts),
            "confidence_millionths": self.confidence_millionths,
            "source_claim_level": self.source_claim_level,
            "semantic_proof": self.semantic_proof,
            "freshness": self.freshness,
            "authoritative": self.authoritative,
            "semantic_authority": self.semantic_authority,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "claim_id": self.claim_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssuranceClaim":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "repository_id",
            "tree_id",
            "symbol",
            "interface",
            "policy_revision",
            "repository_observation_id",
            "claim_level",
            "verdict",
            "inconclusive_state",
            "authority_kind",
            "producer_id",
            "producer_version",
            "evaluated_at",
            "authority_expires_at",
            "expected_contract_id",
            "observed_contract_id",
            "counterexample_id",
            "assumptions",
            "artifacts",
            "confidence_millionths",
            "source_claim_level",
            "semantic_proof",
        }
        _reject_unknown(
            payload,
            fields
            | {
                "schema",
                "schema_version",
                "contract_version",
                "freshness",
                "authoritative",
                "semantic_authority",
                "claim_id",
                "content_id",
            },
            artifact_name="assurance claim",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            symbol=payload.get("symbol", ""),
            interface=payload.get("interface", ""),
            policy_revision=payload.get("policy_revision", ""),
            repository_observation_id=payload.get(
                "repository_observation_id", ""
            ),
            claim_level=payload.get("claim_level", ""),
            verdict=payload.get("verdict", ""),
            inconclusive_state=payload.get("inconclusive_state", ""),
            authority_kind=payload.get("authority_kind", ""),
            producer_id=payload.get("producer_id", ""),
            producer_version=payload.get("producer_version", ""),
            evaluated_at=payload.get("evaluated_at", ""),
            authority_expires_at=payload.get("authority_expires_at", ""),
            expected_contract_id=payload.get("expected_contract_id", ""),
            observed_contract_id=payload.get("observed_contract_id", ""),
            counterexample_id=payload.get("counterexample_id", ""),
            assumptions=tuple(payload.get("assumptions") or ()),
            artifacts=tuple(payload.get("artifacts") or ()),
            confidence_millionths=payload.get("confidence_millionths", 0),
            source_claim_level=payload.get("source_claim_level"),
            semantic_proof=payload.get("semantic_proof", False),
        )
        if "freshness" in payload and _enum(
            payload["freshness"], EvidenceFreshness, field_name="freshness"
        ) is not result.freshness:
            raise ForgedIdentityError("claim freshness does not match timestamps")
        _verify_projection(
            payload,
            name="authoritative",
            actual=result.authoritative,
            stale=result.freshness is EvidenceFreshness.STALE,
        )
        _verify_projection(
            payload,
            name="semantic_authority",
            actual=result.semantic_authority,
        )
        _check_identity(
            payload,
            result.claim_id,
            names=("claim_id", "content_id"),
            artifact_name="assurance claim",
        )
        return result


@dataclass(frozen=True)
class Finding(_AssuranceContract):
    """A bounded finding whose strongest status is derived from exact evidence."""

    SCHEMA: ClassVar[str] = FINDING_SCHEMA

    status: FindingStatus
    severity: FindingSeverity
    summary: str
    claim: AssuranceClaim
    expected_contract: ExpectedContract | None = None
    observed_contract: ObservedContract | None = None
    counterexample: Counterexample | None = None
    affected_paths: tuple[str, ...] = ()
    remediation_scope: tuple[str, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, FindingStatus, field_name="status")
        )
        object.__setattr__(
            self,
            "severity",
            _enum(self.severity, FindingSeverity, field_name="severity"),
        )
        object.__setattr__(
            self, "summary", _text(self.summary, field_name="summary")
        )
        claim = _record(self.claim, AssuranceClaim, field_name="claim")
        expected = _record(
            self.expected_contract,
            ExpectedContract,
            field_name="expected_contract",
            optional=True,
        )
        observed = _record(
            self.observed_contract,
            ObservedContract,
            field_name="observed_contract",
            optional=True,
        )
        counterexample = _record(
            self.counterexample,
            Counterexample,
            field_name="counterexample",
            optional=True,
        )
        object.__setattr__(self, "claim", claim)
        object.__setattr__(self, "expected_contract", expected)
        object.__setattr__(self, "observed_contract", observed)
        object.__setattr__(self, "counterexample", counterexample)
        object.__setattr__(
            self,
            "affected_paths",
            _relative_paths(self.affected_paths, field_name="affected_paths"),
        )
        object.__setattr__(
            self,
            "remediation_scope",
            _strings(
                self.remediation_scope,
                field_name="remediation_scope",
                maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "artifacts",
            _records(self.artifacts, ArtifactReference, field_name="artifacts"),
        )
        if self.status is FindingStatus.CONTRACT_BROKEN:
            if expected is None or observed is None or counterexample is None:
                raise SemanticAuthorityError(
                    "contract_broken requires expected, observed, and counterexample records"
                )
            if not _same_scope(claim, expected, observed, counterexample):
                raise SemanticAuthorityError(
                    "contract_broken evidence must share one exact semantic scope"
                )
            if observed.repository_observation_id != claim.repository_observation_id:
                raise SemanticAuthorityError(
                    "observed contract is detached from the claim observation"
                )
            if (
                claim.expected_contract_id != expected.expected_contract_id
                or claim.observed_contract_id != observed.observed_contract_id
                or claim.counterexample_id != counterexample.counterexample_id
            ):
                raise SemanticAuthorityError(
                    "contract_broken claim references do not match embedded evidence"
                )
            if (
                claim.evaluated_at != counterexample.evaluated_at
                or claim.authority_expires_at
                != counterexample.authority_expires_at
            ):
                raise SemanticAuthorityError(
                    "contract_broken claim and counterexample must share one freshness binding"
                )
            if (
                claim.claim_level
                not in {
                    ClaimLevel.MODEL_DISPROVED,
                    ClaimLevel.RUNTIME_WITNESSED,
                }
                or claim.verdict is not ClaimVerdict.VIOLATED
                or not claim.semantic_authority
                or not counterexample.authoritative
            ):
                raise SemanticAuthorityError(
                    "contract_broken requires a fresh semantic violation and conclusive counterexample"
                )
        if self.status is FindingStatus.AMBIGUOUS and (
            claim.inconclusive_state is not InconclusiveState.AMBIGUOUS
        ):
            raise ProgramAssuranceContractError(
                "ambiguous findings require an ambiguous claim"
            )
        if self.status is FindingStatus.UNSUPPORTED and (
            claim.inconclusive_state is not InconclusiveState.UNSUPPORTED
        ):
            raise ProgramAssuranceContractError(
                "unsupported findings require an unsupported claim"
            )
        if self.status is FindingStatus.STALE and (
            claim.inconclusive_state is not InconclusiveState.STALE
        ):
            raise ProgramAssuranceContractError(
                "stale findings require an explicitly stale claim"
            )
        _bounded(self, artifact_name="finding")

    @property
    def finding_id(self) -> str:
        return self.content_id

    @property
    def repository_id(self) -> str:
        return self.claim.repository_id

    @property
    def tree_id(self) -> str:
        return self.claim.tree_id

    @property
    def symbol(self) -> str:
        return self.claim.symbol

    @property
    def interface(self) -> str:
        return self.claim.interface

    @property
    def policy_revision(self) -> str:
        return self.claim.policy_revision

    @property
    def claim_level(self) -> ClaimLevel:
        return self.claim.claim_level

    @property
    def confidence_millionths(self) -> int:
        return self.claim.confidence_millionths

    @property
    def freshness(self) -> EvidenceFreshness:
        return self.claim.freshness

    @property
    def actionable(self) -> bool:
        return self.status is FindingStatus.CONTRACT_BROKEN

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "status": self.status,
            "severity": self.severity,
            "summary": self.summary,
            "claim": self.claim.to_record(),
            "expected_contract": (
                self.expected_contract.to_record()
                if self.expected_contract is not None
                else None
            ),
            "observed_contract": (
                self.observed_contract.to_record()
                if self.observed_contract is not None
                else None
            ),
            "counterexample": (
                self.counterexample.to_record()
                if self.counterexample is not None
                else None
            ),
            "affected_paths": self.affected_paths,
            "remediation_scope": self.remediation_scope,
            "artifacts": tuple(item.to_record() for item in self.artifacts),
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "symbol": self.symbol,
            "interface": self.interface,
            "policy_revision": self.policy_revision,
            "claim_level": self.claim_level,
            "confidence_millionths": self.confidence_millionths,
            "freshness": self.freshness,
            "actionable": self.actionable,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "finding_id": self.finding_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Finding":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "status",
            "severity",
            "summary",
            "claim",
            "expected_contract",
            "observed_contract",
            "counterexample",
            "affected_paths",
            "remediation_scope",
            "artifacts",
        }
        _reject_unknown(
            payload,
            fields
            | {
                "schema",
                "schema_version",
                "contract_version",
                "repository_id",
                "tree_id",
                "symbol",
                "interface",
                "policy_revision",
                "claim_level",
                "confidence_millionths",
                "freshness",
                "actionable",
                "finding_id",
                "content_id",
            },
            artifact_name="finding",
        )
        result = cls(
            status=payload.get("status", ""),
            severity=payload.get("severity", ""),
            summary=payload.get("summary", ""),
            claim=payload.get("claim", {}),
            expected_contract=payload.get("expected_contract"),
            observed_contract=payload.get("observed_contract"),
            counterexample=payload.get("counterexample"),
            affected_paths=tuple(payload.get("affected_paths") or ()),
            remediation_scope=tuple(payload.get("remediation_scope") or ()),
            artifacts=tuple(payload.get("artifacts") or ()),
        )
        _verify_projection(
            payload, name="actionable", actual=result.actionable
        )
        for name in (
            "repository_id",
            "tree_id",
            "symbol",
            "interface",
            "policy_revision",
        ):
            if name in payload and payload[name] != getattr(result, name):
                raise ForgedIdentityError(
                    f"finding {name} does not match its claim"
                )
        if "claim_level" in payload and _enum(
            payload["claim_level"], ClaimLevel, field_name="claim_level"
        ) is not result.claim_level:
            raise ForgedIdentityError(
                "finding claim_level does not match its claim"
            )
        if (
            "confidence_millionths" in payload
            and payload["confidence_millionths"] != result.confidence_millionths
        ):
            raise ForgedIdentityError(
                "finding confidence does not match its claim"
            )
        if "freshness" in payload and _enum(
            payload["freshness"], EvidenceFreshness, field_name="freshness"
        ) is not result.freshness:
            raise ForgedIdentityError(
                "finding freshness does not match its claim"
            )
        _check_identity(
            payload,
            result.finding_id,
            names=("finding_id", "content_id"),
            artifact_name="finding",
        )
        return result


@dataclass(frozen=True)
class AssuranceLimits(_AssuranceContract):
    """Explicit collection and serialized-size bounds for a stage receipt."""

    SCHEMA: ClassVar[str] = ASSURANCE_LIMITS_SCHEMA

    max_claims: int = 64
    max_findings: int = 64
    max_artifacts: int = 128
    max_record_bytes: int = MAX_RECORD_BYTES
    max_receipt_bytes: int = MAX_STAGE_RECEIPT_BYTES

    def __post_init__(self) -> None:
        for name in (
            "max_claims",
            "max_findings",
            "max_artifacts",
            "max_record_bytes",
            "max_receipt_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    minimum=1,
                    maximum=MAX_STAGE_RECEIPT_BYTES,
                ),
            )
        if self.max_record_bytes > self.max_receipt_bytes:
            raise ContractBoundsError(
                "max_record_bytes cannot exceed max_receipt_bytes"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "max_claims": self.max_claims,
            "max_findings": self.max_findings,
            "max_artifacts": self.max_artifacts,
            "max_record_bytes": self.max_record_bytes,
            "max_receipt_bytes": self.max_receipt_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssuranceLimits":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "max_claims",
            "max_findings",
            "max_artifacts",
            "max_record_bytes",
            "max_receipt_bytes",
        }
        _reject_unknown(
            payload,
            fields
            | {
                "schema",
                "schema_version",
                "contract_version",
                "content_id",
            },
            artifact_name="assurance limits",
        )
        defaults = cls()
        result = cls(
            **{name: payload.get(name, getattr(defaults, name)) for name in fields}
        )
        _check_identity(
            payload,
            result.content_id,
            names=("content_id",),
            artifact_name="assurance limits",
        )
        return result


@dataclass(frozen=True)
class StageReceipt(_AssuranceContract):
    """One bounded stage result with fail-closed derived authority."""

    SCHEMA: ClassVar[str] = STAGE_RECEIPT_SCHEMA

    stage: str
    status: StageStatus
    claim_level: ClaimLevel
    inconclusive_state: InconclusiveState
    observation: RepositoryObservation
    analyzer_id: str
    analyzer_version: str
    objective_revision: str
    policy_revision: str
    configuration_digest: str
    evaluated_at: str
    authority_expires_at: str
    coverage_complete: bool
    truncated: bool
    query_digest: str = ""
    capability_snapshot_id: str = ""
    redaction_policy_revision: str = ""
    dependency_ids: tuple[str, ...] = ()
    toolchain_ids: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    claims: tuple[AssuranceClaim, ...] = ()
    findings: tuple[Finding, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()
    limits: AssuranceLimits = field(default_factory=AssuranceLimits)

    def __post_init__(self) -> None:
        for name in (
            "stage",
            "analyzer_id",
            "analyzer_version",
            "objective_revision",
            "policy_revision",
            "configuration_digest",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        for name in (
            "query_digest",
            "capability_snapshot_id",
            "redaction_policy_revision",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    field_name=name,
                    required=False,
                ),
            )
        for name in ("dependency_ids", "toolchain_ids", "assumptions"):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    maximum=128,
                ),
            )
        object.__setattr__(
            self, "status", _enum(self.status, StageStatus, field_name="status")
        )
        object.__setattr__(
            self,
            "claim_level",
            _enum(self.claim_level, ClaimLevel, field_name="claim_level"),
        )
        object.__setattr__(
            self,
            "inconclusive_state",
            _enum(
                self.inconclusive_state,
                InconclusiveState,
                field_name="inconclusive_state",
            ),
        )
        observation = _record(
            self.observation,
            RepositoryObservation,
            field_name="observation",
        )
        object.__setattr__(self, "observation", observation)
        object.__setattr__(
            self, "evaluated_at", _timestamp(self.evaluated_at, field_name="evaluated_at")
        )
        object.__setattr__(
            self,
            "authority_expires_at",
            _timestamp(
                self.authority_expires_at, field_name="authority_expires_at"
            ),
        )
        object.__setattr__(
            self,
            "coverage_complete",
            _boolean(self.coverage_complete, field_name="coverage_complete"),
        )
        object.__setattr__(
            self, "truncated", _boolean(self.truncated, field_name="truncated")
        )
        claims = _records(self.claims, AssuranceClaim, field_name="claims")
        findings = _records(self.findings, Finding, field_name="findings")
        artifacts = _records(
            self.artifacts, ArtifactReference, field_name="artifacts"
        )
        limits = _record(self.limits, AssuranceLimits, field_name="limits")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "limits", limits)
        if self.policy_revision != observation.policy_revision:
            raise SemanticAuthorityError(
                "stage policy_revision differs from repository observation"
            )
        for claim in claims:
            if claim.claim_level is not self.claim_level:
                raise ClaimPromotionError(
                    "stage receipt cannot combine or promote claim levels"
                )
            if (
                claim.repository_id != observation.repository_id
                or claim.tree_id != observation.tree_id
                or claim.repository_observation_id != observation.observation_id
                or claim.policy_revision != self.policy_revision
            ):
                raise SemanticAuthorityError(
                    "claim is detached from the stage observation"
                )
            if (
                claim.evaluated_at != self.evaluated_at
                or claim.authority_expires_at != self.authority_expires_at
            ):
                raise SemanticAuthorityError(
                    "claim and stage receipt must share one freshness binding"
                )
        for finding in findings:
            if finding.claim.claim_id not in {item.claim_id for item in claims}:
                raise SemanticAuthorityError(
                    "finding claim is not embedded in the stage receipt"
                )
        if self.status.successful and self.inconclusive_state.conclusive:
            if not claims:
                raise ProgramAssuranceContractError(
                    "a conclusive completed stage requires at least one claim"
                )
        else:
            if self.inconclusive_state is InconclusiveState.NONE:
                raise ProgramAssuranceContractError(
                    "non-completed stages require an explicit inconclusive state"
                )
        if self.freshness is EvidenceFreshness.STALE and (
            self.status.successful or self.inconclusive_state is not InconclusiveState.STALE
        ):
            raise StaleAuthorityError(
                "stale stage receipts must be explicitly inconclusive"
            )
        if len(claims) > limits.max_claims:
            raise ContractBoundsError("claims exceed the configured stage limit")
        if len(findings) > limits.max_findings:
            raise ContractBoundsError("findings exceed the configured stage limit")
        nested_artifact_count = (
            len(observation.artifacts)
            + len(artifacts)
            + sum(len(item.artifacts) for item in claims)
            + sum(len(item.artifacts) for item in findings)
            + sum(
                len(item.counterexample.artifacts)
                for item in findings
                if item.counterexample is not None
            )
        )
        if nested_artifact_count > limits.max_artifacts:
            raise ContractBoundsError("artifacts exceed the configured stage limit")
        for record in (*claims, *findings, *artifacts):
            if len(record.canonical_bytes()) > limits.max_record_bytes:
                raise ContractBoundsError(
                    f"{type(record).__name__} exceeds max_record_bytes"
                )
        _bounded(
            self,
            maximum=limits.max_receipt_bytes,
            artifact_name="stage receipt",
        )

    @property
    def freshness(self) -> EvidenceFreshness:
        evaluated = _datetime(self.evaluated_at)
        current = (
            evaluated < _datetime(self.authority_expires_at)
            and self.observation.freshness_at(self.evaluated_at)
            is EvidenceFreshness.CURRENT
        )
        return EvidenceFreshness.CURRENT if current else EvidenceFreshness.STALE

    @property
    def authoritative(self) -> bool:
        return (
            self.status.successful
            and self.inconclusive_state is InconclusiveState.NONE
            and self.freshness is EvidenceFreshness.CURRENT
            and self.coverage_complete
            and not self.truncated
            and bool(self.claims)
            and all(item.authoritative for item in self.claims)
        )

    @property
    def safe_for_semantic_reasoning(self) -> bool:
        return (
            self.authoritative
            and self.claim_level is not ClaimLevel.ZK_TRACE_ATTESTED
            and all(item.semantic_authority for item in self.claims)
        )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "stage": self.stage,
            "status": self.status,
            "claim_level": self.claim_level,
            "inconclusive_state": self.inconclusive_state,
            "observation": self.observation.to_record(),
            "analyzer_id": self.analyzer_id,
            "analyzer_version": self.analyzer_version,
            "objective_revision": self.objective_revision,
            "policy_revision": self.policy_revision,
            "configuration_digest": self.configuration_digest,
            "query_digest": self.query_digest,
            "capability_snapshot_id": self.capability_snapshot_id,
            "redaction_policy_revision": self.redaction_policy_revision,
            "dependency_ids": self.dependency_ids,
            "toolchain_ids": self.toolchain_ids,
            "assumptions": self.assumptions,
            "evaluated_at": self.evaluated_at,
            "authority_expires_at": self.authority_expires_at,
            "coverage_complete": self.coverage_complete,
            "truncated": self.truncated,
            "claims": tuple(item.to_record() for item in self.claims),
            "findings": tuple(item.to_record() for item in self.findings),
            "artifacts": tuple(item.to_record() for item in self.artifacts),
            "limits": self.limits.to_record(),
            "freshness": self.freshness,
            "authoritative": self.authoritative,
            "safe_for_semantic_reasoning": self.safe_for_semantic_reasoning,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageReceipt":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "stage",
            "status",
            "claim_level",
            "inconclusive_state",
            "observation",
            "analyzer_id",
            "analyzer_version",
            "objective_revision",
            "policy_revision",
            "configuration_digest",
            "query_digest",
            "capability_snapshot_id",
            "redaction_policy_revision",
            "dependency_ids",
            "toolchain_ids",
            "assumptions",
            "evaluated_at",
            "authority_expires_at",
            "coverage_complete",
            "truncated",
            "claims",
            "findings",
            "artifacts",
            "limits",
        }
        _reject_unknown(
            payload,
            fields
            | {
                "schema",
                "schema_version",
                "contract_version",
                "freshness",
                "authoritative",
                "safe_for_semantic_reasoning",
                "receipt_id",
                "content_id",
            },
            artifact_name="stage receipt",
        )
        result = cls(
            stage=payload.get("stage", ""),
            status=payload.get("status", ""),
            claim_level=payload.get("claim_level", ""),
            inconclusive_state=payload.get("inconclusive_state", ""),
            observation=payload.get("observation", {}),
            analyzer_id=payload.get("analyzer_id", ""),
            analyzer_version=payload.get("analyzer_version", ""),
            objective_revision=payload.get("objective_revision", ""),
            policy_revision=payload.get("policy_revision", ""),
            configuration_digest=payload.get("configuration_digest", ""),
            query_digest=payload.get("query_digest", ""),
            capability_snapshot_id=payload.get("capability_snapshot_id", ""),
            redaction_policy_revision=payload.get(
                "redaction_policy_revision", ""
            ),
            dependency_ids=tuple(payload.get("dependency_ids") or ()),
            toolchain_ids=tuple(payload.get("toolchain_ids") or ()),
            assumptions=tuple(payload.get("assumptions") or ()),
            evaluated_at=payload.get("evaluated_at", ""),
            authority_expires_at=payload.get("authority_expires_at", ""),
            coverage_complete=payload.get("coverage_complete", False),
            truncated=payload.get("truncated", False),
            claims=tuple(payload.get("claims") or ()),
            findings=tuple(payload.get("findings") or ()),
            artifacts=tuple(payload.get("artifacts") or ()),
            limits=payload.get("limits") or {},
        )
        if "freshness" in payload and _enum(
            payload["freshness"], EvidenceFreshness, field_name="freshness"
        ) is not result.freshness:
            raise ForgedIdentityError("stage freshness does not match timestamps")
        _verify_projection(
            payload,
            name="authoritative",
            actual=result.authoritative,
            stale=result.freshness is EvidenceFreshness.STALE,
        )
        _verify_projection(
            payload,
            name="safe_for_semantic_reasoning",
            actual=result.safe_for_semantic_reasoning,
        )
        _check_identity(
            payload,
            result.receipt_id,
            names=("receipt_id", "content_id"),
            artifact_name="stage receipt",
        )
        return result


def canonical_program_assurance_json_bytes(
    value: CanonicalContract | Mapping[str, Any],
) -> bytes:
    """Return bounded deterministic bytes, rejecting floats and non-finite data."""

    payload = value.to_dict() if isinstance(value, CanonicalContract) else value
    encoded = canonical_json_bytes(payload)
    if len(encoded) > MAX_STAGE_RECEIPT_BYTES:
        raise ContractBoundsError(
            f"program-assurance payload exceeds {MAX_STAGE_RECEIPT_BYTES} bytes"
        )
    return encoded


def program_assurance_content_identity(
    value: CanonicalContract | Mapping[str, Any],
) -> str:
    """Return the established CIDv1 identity for a canonical assurance value."""

    payload = value.to_dict() if isinstance(value, CanonicalContract) else value
    canonical_program_assurance_json_bytes(payload)
    return content_identity(payload)


# Clear compatibility spellings for downstream adapters.
ProgramAssuranceArtifactReference = ArtifactReference
ProgramAssuranceStageReceipt = StageReceipt
ProgramAssuranceClaim = AssuranceClaim
ProgramAssuranceFinding = Finding
ProgramAssuranceLimits = AssuranceLimits
ProgramAssuranceContractValidationError = ProgramAssuranceContractError
AssuranceContractValidationError = ProgramAssuranceContractError
AssuranceFinding = Finding
AssuranceStageReceipt = StageReceipt
EvidenceClaim = AssuranceClaim
RepositoryObservationRecord = RepositoryObservation
CounterexampleRecord = Counterexample
ContractExpectation = ExpectedContract
ContractObservation = ObservedContract
ContractFinding = Finding
Claim = AssuranceClaim
Freshness = EvidenceFreshness
FindingDisposition = FindingStatus
StageReceiptStatus = StageStatus
ExplicitInconclusiveState = InconclusiveState


__all__ = [
    "ARTIFACT_REFERENCE_SCHEMA",
    "ASSURANCE_CLAIM_SCHEMA",
    "ASSURANCE_LIMITS_SCHEMA",
    "CONTRACT_VERSION",
    "COUNTEREXAMPLE_SCHEMA",
    "EXPECTED_CONTRACT_SCHEMA",
    "FINDING_SCHEMA",
    "MAX_CLAUSE_BYTES",
    "MAX_COLLECTION_ITEMS",
    "MAX_RECORD_BYTES",
    "MAX_STAGE_RECEIPT_BYTES",
    "MAX_TEXT_BYTES",
    "MILLION",
    "OBSERVED_CONTRACT_SCHEMA",
    "PROGRAM_ASSURANCE_CONTRACT_VERSION",
    "REPOSITORY_OBSERVATION_SCHEMA",
    "SCHEMA_VERSION",
    "STAGE_RECEIPT_SCHEMA",
    "ArtifactReference",
    "AssuranceClaim",
    "AssuranceContractValidationError",
    "AssuranceFinding",
    "AssuranceLimits",
    "AssuranceStageReceipt",
    "AuthorityKind",
    "Claim",
    "ClaimLevel",
    "ClaimPromotionError",
    "ClaimVerdict",
    "ContractBoundsError",
    "ContractFinding",
    "ContractExpectation",
    "ContractObservation",
    "ContractPrecedence",
    "Counterexample",
    "CounterexampleRecord",
    "EvidenceClaim",
    "EvidenceFreshness",
    "ExpectedContract",
    "Finding",
    "FindingSeverity",
    "FindingStatus",
    "FindingDisposition",
    "ForgedIdentityError",
    "Freshness",
    "InconclusiveState",
    "ExplicitInconclusiveState",
    "ObservedContract",
    "ProgramAssuranceArtifactReference",
    "ProgramAssuranceClaim",
    "ProgramAssuranceContractError",
    "ProgramAssuranceContractValidationError",
    "ProgramAssuranceFinding",
    "ProgramAssuranceLimits",
    "ProgramAssuranceStageReceipt",
    "RepositoryObservation",
    "RepositoryObservationRecord",
    "SemanticAuthorityError",
    "StageReceipt",
    "StageReceiptStatus",
    "StageStatus",
    "StaleAuthorityError",
    "canonical_program_assurance_json_bytes",
    "program_assurance_content_identity",
    "validate_claim_promotion",
]
