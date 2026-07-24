"""Deterministic, bounded end-to-end supervisor efficiency receipts.

The supervisor emits many useful local metrics, but an optimization decision
needs one record which follows a task all the way to a terminal outcome.  This
module defines that record and the aggregation rules used by paired
benchmarks.

The wire contract intentionally contains no prompt, source, decoded model
response, command output, or recursively embedded artifact.  Potentially large
or sensitive values cross this boundary only as SHA-256 digests and small,
typed references.  All accounting uses bounded integers; ratios are retained
as exact numerator/denominator pairs and exposed as convenience floats only at
the Python API boundary.

Aggregation is deliberately fail closed:

* every terminal attempt contributes tokens and cost;
* only accepted attempts contribute evidence gain; and
* accepted tasks are counted by stable task reference, so a failed attempt
  followed by a repair has one accepted-task denominator.

These rules prevent cheap failures, retry churn, or duplicate acceptance
receipts from making an implementation appear more efficient.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
)


EFFICIENCY_CONTRACT_VERSION = 1
SCHEMA_VERSION = EFFICIENCY_CONTRACT_VERSION
EFFICIENCY_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-efficiency-receipt@1"
)
EFFICIENCY_REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-efficiency-report@1"
)
TOKEN_USAGE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/efficiency-token-usage@1"
)
STAGE_TIMING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/efficiency-stage-timing@1"
)
CACHE_OBSERVATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/efficiency-cache-observation@1"
)
RETRY_OBSERVATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/efficiency-retry-observation@1"
)
WORK_COST_SCHEMA = "ipfs_accelerate_py/agent-supervisor/efficiency-work-cost@1"
CHANGED_SCOPE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/efficiency-changed-scope@1"
)
ARTIFACT_REFERENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/efficiency-artifact-reference@1"
)
EVIDENCE_DELTA_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/efficiency-evidence-delta@1"
)
TERMINAL_ACCEPTANCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/efficiency-terminal-acceptance@1"
)
EXACT_RATIO_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/efficiency-exact-ratio@1"
)

# Absolute construction bounds.  They are intentionally generous enough for a
# full supervisor run but small enough that a receipt cannot become a hidden
# prompt, source archive, or artifact graph.
MAX_TEXT_BYTES = 512
MAX_REFERENCE_BYTES = 256
MAX_REASON_CODES = 32
MAX_STAGES = 32
MAX_CACHE_OBSERVATIONS = 64
MAX_RETRIES = 32
MAX_CHANGED_PATHS = 128
MAX_CHANGED_SYMBOLS = 256
MAX_ARTIFACT_REFERENCES = 64
MAX_EVIDENCE_REFERENCES = 256
MAX_RELATED_TASK_REFERENCES = 64
MAX_CONFLICT_REFERENCES = 64
MAX_RECEIPTS_PER_REPORT = 10_000
MAX_SERIALIZED_RECEIPT_BYTES = 262_144
MAX_SERIALIZED_REPORT_BYTES = 1_048_576
MAX_DURATION_MS = 31 * 24 * 60 * 60 * 1000
MAX_TOKENS = 1_000_000_000
MAX_BYTES = 1_000_000_000_000
MAX_COST_MICROUNITS = 1_000_000_000_000_000
MAX_CHANGE_LINES = 100_000_000
MAX_OPERATIONS = 1_000_000

_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_CODE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,95}$")


EfficiencyValidationError = ContractValidationError


class StageName(str, Enum):
    """Known stages in one supervisor task lifecycle."""

    ADMISSION = "admission"
    ANALYSIS = "analysis"
    CONTEXT = "context"
    PLANNING = "planning"
    INFERENCE = "inference"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    PROOF = "proof"
    MERGE = "merge"
    ACCEPTANCE = "acceptance"


class CacheDisposition(str, Enum):
    """Result of one namespaced cache lookup."""

    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"
    INVALIDATED = "invalidated"
    ERROR = "error"


class WorkStatus(str, Enum):
    """Outcome of validation or proof work."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_REQUIRED = "not_required"


class TerminalOutcome(str, Enum):
    """Terminal task result.  There are deliberately no in-progress values."""

    ACCEPTED = "accepted"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    CONFLICTED = "conflicted"

    @property
    def accepted(self) -> bool:
        return self is TerminalOutcome.ACCEPTED


class EfficiencyScenario(str, Enum):
    """Fixture/benchmark classification without changing outcome authority."""

    OBSERVED = "observed"
    COLD = "cold"
    WARM = "warm"
    FAILED = "failed"
    REPAIRED = "repaired"
    PARALLEL_INDEPENDENT = "parallel-independent"
    CONFLICTING = "conflicting"


def _enum(value: Any, enum_type: type[Enum], *, field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContractValidationError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    max_bytes: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string")
    else:
        result = value.strip()
    if required and not result:
        raise ContractValidationError(f"{field_name} is required")
    if "\x00" in result:
        raise ContractValidationError(f"{field_name} must not contain NUL bytes")
    try:
        encoded = result.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractValidationError(
            f"{field_name} must contain valid Unicode text"
        ) from exc
    if len(encoded) > max_bytes:
        raise ContractValidationError(
            f"{field_name} exceeds the {max_bytes}-byte bound"
        )
    return result


def _code(value: Any, *, field_name: str) -> str:
    result = _text(
        value,
        field_name=field_name,
        required=True,
        max_bytes=96,
    ).lower()
    if not _CODE_RE.fullmatch(result):
        raise ContractValidationError(
            f"{field_name} must be a bounded machine-readable code"
        )
    return result


def _integer(
    value: Any,
    *,
    field_name: str,
    maximum: int,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise ContractValidationError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return value


def _digest(value: Any, *, field_name: str, required: bool = True) -> str:
    result = _text(
        value,
        field_name=field_name,
        required=required,
        max_bytes=71,
    )
    if not result:
        return ""
    match = _SHA256_RE.fullmatch(result)
    if match is None:
        raise ContractValidationError(
            f"{field_name} must be a SHA-256 digest"
        )
    return "sha256:" + match.group(1).lower()


def _strings(
    values: Any,
    *,
    field_name: str,
    maximum: int,
    max_item_bytes: int = MAX_REFERENCE_BYTES,
    code: bool = False,
) -> tuple[str, ...]:
    if values is None:
        source: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise ContractValidationError(f"{field_name} must be a sequence")
    if len(source) > maximum:
        raise ContractValidationError(
            f"{field_name} exceeds the {maximum}-item bound"
        )
    result: list[str] = []
    for index, item in enumerate(source):
        normalized = (
            _code(item, field_name=f"{field_name}[{index}]")
            if code
            else _text(
                item,
                field_name=f"{field_name}[{index}]",
                required=True,
                max_bytes=max_item_bytes,
            )
        )
        if normalized not in result:
            result.append(normalized)
    return tuple(sorted(result))


def _repo_paths(values: Any, *, field_name: str) -> tuple[str, ...]:
    paths = _strings(
        values,
        field_name=field_name,
        maximum=MAX_CHANGED_PATHS,
        max_item_bytes=MAX_TEXT_BYTES,
    )
    normalized: list[str] = []
    for path in paths:
        candidate_text = path.replace("\\", "/")
        candidate = PurePosixPath(candidate_text)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or candidate_text in {"", "."}
        ):
            raise ContractValidationError(
                f"{field_name} must contain repository-relative paths"
            )
        normalized.append(candidate_text)
    return tuple(sorted(normalized))


def _schema(payload: Mapping[str, Any], expected: str, artifact_name: str) -> None:
    if not isinstance(payload, Mapping):
        raise ContractValidationError(f"{artifact_name} must be an object")
    if payload.get("schema") != expected:
        raise ContractValidationError(
            f"unsupported {artifact_name} schema; expected {expected}"
        )
    version = payload.get("contract_version", payload.get("schema_version"))
    if version != EFFICIENCY_CONTRACT_VERSION:
        raise ContractValidationError(
            f"unsupported {artifact_name} contract version"
        )


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: Iterable[str],
    *,
    artifact_name: str,
) -> None:
    if set(payload).difference(allowed):
        raise ContractValidationError(
            f"{artifact_name} contains unsupported fields"
        )


def _claim(payload: Mapping[str, Any], actual: str, *names: str) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise ContractValidationError(
                "content identity does not match the canonical payload"
            )


def _coerce_records(
    values: Any,
    expected_type: type,
    decoder: Any,
    *,
    field_name: str,
    maximum: int,
) -> tuple[Any, ...]:
    if values is None:
        source: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise ContractValidationError(f"{field_name} must be a sequence")
    if len(source) > maximum:
        raise ContractValidationError(
            f"{field_name} exceeds the {maximum}-item bound"
        )
    records: list[Any] = []
    for index, value in enumerate(source):
        if isinstance(value, expected_type):
            record = value
        elif isinstance(value, Mapping):
            record = decoder(value)
        else:
            raise ContractValidationError(
                f"{field_name}[{index}] must be a {expected_type.__name__}"
            )
        records.append(record)
    return tuple(records)


def _record_size(record: CanonicalContract, *, maximum: int, name: str) -> None:
    if len(record.canonical_bytes()) > maximum:
        raise ContractValidationError(
            f"{name} exceeds the {maximum}-byte serialized bound"
        )


def _load_json(value: str | bytes | bytearray, *, artifact_name: str) -> Any:
    """Decode strict JSON, rejecting invalid UTF-8 and duplicate object keys."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ContractValidationError(
                    f"{artifact_name} JSON contains duplicate object keys"
                )
            result[key] = item
        return result

    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        if not isinstance(value, str):
            raise ContractValidationError(
                f"{artifact_name} JSON must be text or bytes"
            )
        return json.loads(value, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(
            f"{artifact_name} JSON is invalid"
        ) from exc


@dataclass(frozen=True)
class TokenUsage(CanonicalContract):
    """Model-token accounting, with reused tokens a subset of input tokens."""

    SCHEMA: ClassVar[str] = TOKEN_USAGE_SCHEMA

    input_tokens: int = 0
    output_tokens: int = 0
    reused_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "reused_tokens"):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    maximum=MAX_TOKENS,
                ),
            )
        if self.reused_tokens > self.input_tokens:
            raise ContractValidationError(
                "reused_tokens cannot exceed input_tokens"
            )

    @property
    def fresh_input_tokens(self) -> int:
        return self.input_tokens - self.reused_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reused_tokens": self.reused_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TokenUsage":
        _schema(payload, cls.SCHEMA, "token usage")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "input_tokens",
                "output_tokens",
                "reused_tokens",
                "content_id",
            },
            artifact_name="token usage",
        )
        result = cls(
            input_tokens=payload.get("input_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
            reused_tokens=payload.get("reused_tokens", 0),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class StageTiming(CanonicalContract):
    """Latency and invocation count for one lifecycle stage."""

    SCHEMA: ClassVar[str] = STAGE_TIMING_SCHEMA

    stage: StageName
    latency_ms: int
    invocation_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage", _enum(self.stage, StageName, field_name="stage")
        )
        object.__setattr__(
            self,
            "latency_ms",
            _integer(
                self.latency_ms,
                field_name="latency_ms",
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "invocation_count",
            _integer(
                self.invocation_count,
                field_name="invocation_count",
                minimum=1,
                maximum=MAX_OPERATIONS,
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "stage": self.stage,
            "latency_ms": self.latency_ms,
            "invocation_count": self.invocation_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageTiming":
        _schema(payload, cls.SCHEMA, "stage timing")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "stage",
                "latency_ms",
                "invocation_count",
                "content_id",
            },
            artifact_name="stage timing",
        )
        result = cls(
            stage=payload.get("stage", ""),
            latency_ms=payload.get("latency_ms", 0),
            invocation_count=payload.get("invocation_count", 1),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class CacheObservation(CanonicalContract):
    """One compact cache lookup outcome; cache keys are represented by digests."""

    SCHEMA: ClassVar[str] = CACHE_OBSERVATION_SCHEMA

    namespace: str
    disposition: CacheDisposition
    key_digest: str
    lookup_latency_ms: int = 0
    bytes_reused: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "namespace",
            _code(self.namespace, field_name="namespace"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(
                self.disposition,
                CacheDisposition,
                field_name="disposition",
            ),
        )
        object.__setattr__(
            self,
            "key_digest",
            _digest(self.key_digest, field_name="key_digest"),
        )
        object.__setattr__(
            self,
            "lookup_latency_ms",
            _integer(
                self.lookup_latency_ms,
                field_name="lookup_latency_ms",
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "bytes_reused",
            _integer(
                self.bytes_reused,
                field_name="bytes_reused",
                maximum=MAX_BYTES,
            ),
        )
        if (
            self.disposition is not CacheDisposition.HIT
            and self.bytes_reused
        ):
            raise ContractValidationError(
                "only a cache hit can report bytes_reused"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "namespace": self.namespace,
            "disposition": self.disposition,
            "key_digest": self.key_digest,
            "lookup_latency_ms": self.lookup_latency_ms,
            "bytes_reused": self.bytes_reused,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CacheObservation":
        _schema(payload, cls.SCHEMA, "cache observation")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "namespace",
                "disposition",
                "key_digest",
                "lookup_latency_ms",
                "bytes_reused",
                "content_id",
            },
            artifact_name="cache observation",
        )
        result = cls(
            namespace=payload.get("namespace", ""),
            disposition=payload.get("disposition", ""),
            key_digest=payload.get("key_digest", ""),
            lookup_latency_ms=payload.get("lookup_latency_ms", 0),
            bytes_reused=payload.get("bytes_reused", 0),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class RetryObservation(CanonicalContract):
    """Compact retry accounting without replaying original or delta context."""

    SCHEMA: ClassVar[str] = RETRY_OBSERVATION_SCHEMA

    attempt: int
    reason_code: str
    diagnostic_digest: str
    delta_context_digest: str = ""
    tokens: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt",
            _integer(
                self.attempt,
                field_name="attempt",
                minimum=2,
                maximum=MAX_RETRIES + 1,
            ),
        )
        object.__setattr__(
            self,
            "reason_code",
            _code(self.reason_code, field_name="reason_code"),
        )
        object.__setattr__(
            self,
            "diagnostic_digest",
            _digest(
                self.diagnostic_digest,
                field_name="diagnostic_digest",
            ),
        )
        object.__setattr__(
            self,
            "delta_context_digest",
            _digest(
                self.delta_context_digest,
                field_name="delta_context_digest",
                required=False,
            ),
        )
        tokens = self.tokens
        if isinstance(tokens, Mapping):
            tokens = TokenUsage.from_dict(tokens)
        if not isinstance(tokens, TokenUsage):
            raise ContractValidationError("tokens must be TokenUsage")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(
            self,
            "latency_ms",
            _integer(
                self.latency_ms,
                field_name="latency_ms",
                maximum=MAX_DURATION_MS,
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "attempt": self.attempt,
            "reason_code": self.reason_code,
            "diagnostic_digest": self.diagnostic_digest,
            "delta_context_digest": self.delta_context_digest,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RetryObservation":
        _schema(payload, cls.SCHEMA, "retry observation")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "attempt",
                "reason_code",
                "diagnostic_digest",
                "delta_context_digest",
                "tokens",
                "latency_ms",
                "content_id",
            },
            artifact_name="retry observation",
        )
        result = cls(
            attempt=payload.get("attempt", 0),
            reason_code=payload.get("reason_code", ""),
            diagnostic_digest=payload.get("diagnostic_digest", ""),
            delta_context_digest=payload.get("delta_context_digest", ""),
            tokens=TokenUsage.from_dict(payload.get("tokens") or {}),
            latency_ms=payload.get("latency_ms", 0),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class ArtifactReference(CanonicalContract):
    """Bounded content-addressed artifact pointer, never an artifact body."""

    SCHEMA: ClassVar[str] = ARTIFACT_REFERENCE_SCHEMA

    reference_id: str
    digest: str
    kind: str
    byte_count: int = 0
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_id",
            _text(
                self.reference_id,
                field_name="reference_id",
                required=True,
                max_bytes=MAX_REFERENCE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, field_name="digest"),
        )
        object.__setattr__(
            self, "kind", _code(self.kind, field_name="kind")
        )
        object.__setattr__(
            self,
            "byte_count",
            _integer(
                self.byte_count,
                field_name="byte_count",
                maximum=MAX_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "media_type",
            _text(
                self.media_type,
                field_name="media_type",
                required=True,
                max_bytes=128,
            ).lower(),
        )
        if "/" not in self.media_type:
            raise ContractValidationError(
                "media_type must be an Internet media type"
            )

    @property
    def artifact_id(self) -> str:
        return self.reference_id

    @property
    def sha256(self) -> str:
        return self.digest.removeprefix("sha256:")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "reference_id": self.reference_id,
            "digest": self.digest,
            "kind": self.kind,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactReference":
        _schema(payload, cls.SCHEMA, "artifact reference")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "reference_id",
                "digest",
                "kind",
                "byte_count",
                "media_type",
                "content_id",
            },
            artifact_name="artifact reference",
        )
        result = cls(
            reference_id=payload.get("reference_id", ""),
            digest=payload.get("digest", ""),
            kind=payload.get("kind", ""),
            byte_count=payload.get("byte_count", 0),
            media_type=payload.get(
                "media_type", "application/octet-stream"
            ),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class WorkCost(CanonicalContract):
    """Validation/proof cost with compact evidence references."""

    SCHEMA: ClassVar[str] = WORK_COST_SCHEMA

    status: WorkStatus = WorkStatus.NOT_REQUIRED
    duration_ms: int = 0
    cost_microunits: int = 0
    operation_count: int = 0
    evidence_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, WorkStatus, field_name="status")
        )
        object.__setattr__(
            self,
            "duration_ms",
            _integer(
                self.duration_ms,
                field_name="duration_ms",
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "cost_microunits",
            _integer(
                self.cost_microunits,
                field_name="cost_microunits",
                maximum=MAX_COST_MICROUNITS,
            ),
        )
        object.__setattr__(
            self,
            "operation_count",
            _integer(
                self.operation_count,
                field_name="operation_count",
                maximum=MAX_OPERATIONS,
            ),
        )
        object.__setattr__(
            self,
            "evidence_references",
            _strings(
                self.evidence_references,
                field_name="evidence_references",
                maximum=MAX_EVIDENCE_REFERENCES,
            ),
        )
        if self.status is WorkStatus.NOT_REQUIRED and (
            self.duration_ms
            or self.cost_microunits
            or self.operation_count
            or self.evidence_references
        ):
            raise ContractValidationError(
                "not_required work cannot report cost or evidence"
            )
        if self.status in {WorkStatus.PASSED, WorkStatus.FAILED}:
            if self.operation_count < 1:
                raise ContractValidationError(
                    "executed work must report operation_count"
                )
        if self.status is WorkStatus.PASSED and not self.evidence_references:
            raise ContractValidationError(
                "passed work must reference its evidence receipt"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "cost_microunits": self.cost_microunits,
            "operation_count": self.operation_count,
            "evidence_references": self.evidence_references,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkCost":
        _schema(payload, cls.SCHEMA, "work cost")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "status",
                "duration_ms",
                "cost_microunits",
                "operation_count",
                "evidence_references",
                "content_id",
            },
            artifact_name="work cost",
        )
        result = cls(
            status=payload.get("status", WorkStatus.NOT_REQUIRED),
            duration_ms=payload.get("duration_ms", 0),
            cost_microunits=payload.get("cost_microunits", 0),
            operation_count=payload.get("operation_count", 0),
            evidence_references=tuple(
                payload.get("evidence_references") or ()
            ),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class ChangedScope(CanonicalContract):
    """Compact changed-code scope; source and diffs remain in artifact storage."""

    SCHEMA: ClassVar[str] = CHANGED_SCOPE_SCHEMA

    paths: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    lines_added: int = 0
    lines_deleted: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "paths", _repo_paths(self.paths, field_name="paths")
        )
        object.__setattr__(
            self,
            "symbols",
            _strings(
                self.symbols,
                field_name="symbols",
                maximum=MAX_CHANGED_SYMBOLS,
                max_item_bytes=MAX_TEXT_BYTES,
            ),
        )
        for name in ("lines_added", "lines_deleted"):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    maximum=MAX_CHANGE_LINES,
                ),
            )

    @property
    def changed_file_count(self) -> int:
        return len(self.paths)

    @property
    def changed_symbol_count(self) -> int:
        return len(self.symbols)

    @property
    def changed_line_count(self) -> int:
        return self.lines_added + self.lines_deleted

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "paths": self.paths,
            "symbols": self.symbols,
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChangedScope":
        _schema(payload, cls.SCHEMA, "changed scope")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "paths",
                "symbols",
                "lines_added",
                "lines_deleted",
                "content_id",
            },
            artifact_name="changed scope",
        )
        result = cls(
            paths=tuple(payload.get("paths") or ()),
            symbols=tuple(payload.get("symbols") or ()),
            lines_added=payload.get("lines_added", 0),
            lines_deleted=payload.get("lines_deleted", 0),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class EvidenceDelta(CanonicalContract):
    """Evidence-set change represented by stable bounded references."""

    SCHEMA: ClassVar[str] = EVIDENCE_DELTA_SCHEMA

    baseline_references: tuple[str, ...] = ()
    terminal_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_references",
            _strings(
                self.baseline_references,
                field_name="baseline_references",
                maximum=MAX_EVIDENCE_REFERENCES,
            ),
        )
        object.__setattr__(
            self,
            "terminal_references",
            _strings(
                self.terminal_references,
                field_name="terminal_references",
                maximum=MAX_EVIDENCE_REFERENCES,
            ),
        )

    @property
    def gained_references(self) -> tuple[str, ...]:
        return tuple(
            sorted(set(self.terminal_references) - set(self.baseline_references))
        )

    @property
    def lost_references(self) -> tuple[str, ...]:
        return tuple(
            sorted(set(self.baseline_references) - set(self.terminal_references))
        )

    @property
    def gain(self) -> int:
        return len(self.gained_references)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "baseline_references": self.baseline_references,
            "terminal_references": self.terminal_references,
            "gain": self.gain,
            "gained_references": self.gained_references,
            "lost_references": self.lost_references,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceDelta":
        _schema(payload, cls.SCHEMA, "evidence delta")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "baseline_references",
                "terminal_references",
                "gain",
                "gained_references",
                "lost_references",
                "content_id",
            },
            artifact_name="evidence delta",
        )
        result = cls(
            baseline_references=tuple(
                payload.get("baseline_references") or ()
            ),
            terminal_references=tuple(
                payload.get("terminal_references") or ()
            ),
        )
        if payload.get("gain", result.gain) != result.gain:
            raise ContractValidationError("evidence gain claim does not match sets")
        if tuple(payload.get("gained_references", result.gained_references)) != (
            result.gained_references
        ):
            raise ContractValidationError(
                "gained evidence claim does not match sets"
            )
        if tuple(payload.get("lost_references", result.lost_references)) != (
            result.lost_references
        ):
            raise ContractValidationError(
                "lost evidence claim does not match sets"
            )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class TerminalAcceptance(CanonicalContract):
    """Fail-closed terminal outcome and acceptance receipt digest."""

    SCHEMA: ClassVar[str] = TERMINAL_ACCEPTANCE_SCHEMA

    outcome: TerminalOutcome
    reason_codes: tuple[str, ...]
    acceptance_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome",
            _enum(self.outcome, TerminalOutcome, field_name="outcome"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _strings(
                self.reason_codes,
                field_name="reason_codes",
                maximum=MAX_REASON_CODES,
                code=True,
            ),
        )
        if not self.reason_codes:
            raise ContractValidationError(
                "terminal acceptance requires at least one reason code"
            )
        object.__setattr__(
            self,
            "acceptance_digest",
            _digest(
                self.acceptance_digest,
                field_name="acceptance_digest",
                required=self.outcome.accepted,
            ),
        )
        if not self.outcome.accepted and self.acceptance_digest:
            raise ContractValidationError(
                "a non-accepted outcome cannot carry an acceptance digest"
            )

    @property
    def accepted(self) -> bool:
        return self.outcome.accepted

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "outcome": self.outcome,
            "accepted": self.accepted,
            "reason_codes": self.reason_codes,
            "acceptance_digest": self.acceptance_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TerminalAcceptance":
        _schema(payload, cls.SCHEMA, "terminal acceptance")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "outcome",
                "accepted",
                "reason_codes",
                "acceptance_digest",
                "content_id",
            },
            artifact_name="terminal acceptance",
        )
        result = cls(
            outcome=payload.get("outcome", ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            acceptance_digest=payload.get("acceptance_digest", ""),
        )
        if payload.get("accepted", result.accepted) is not result.accepted:
            raise ContractValidationError(
                "accepted claim does not match terminal outcome"
            )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class ExactRatio(CanonicalContract):
    """Exact ratio used for deterministic aggregate metric serialization."""

    SCHEMA: ClassVar[str] = EXACT_RATIO_SCHEMA

    numerator: int
    denominator: int
    multiplier: int = 1

    def __post_init__(self) -> None:
        for name, maximum in (
            (
                "numerator",
                MAX_COST_MICROUNITS * 3 * MAX_RECEIPTS_PER_REPORT,
            ),
            ("denominator", MAX_TOKENS * MAX_RECEIPTS_PER_REPORT),
            ("multiplier", 1_000_000_000),
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    maximum=maximum,
                ),
            )

    @property
    def defined(self) -> bool:
        return self.denominator > 0

    @property
    def value(self) -> float:
        if not self.denominator:
            return 0.0
        return self.numerator * self.multiplier / self.denominator

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "multiplier": self.multiplier,
            "defined": self.defined,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExactRatio":
        _schema(payload, cls.SCHEMA, "exact ratio")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "numerator",
                "denominator",
                "multiplier",
                "defined",
                "content_id",
            },
            artifact_name="exact ratio",
        )
        result = cls(
            numerator=payload.get("numerator", 0),
            denominator=payload.get("denominator", 0),
            multiplier=payload.get("multiplier", 1),
        )
        if payload.get("defined", result.defined) is not result.defined:
            raise ContractValidationError("ratio defined claim is inconsistent")
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class EfficiencyReceipt(CanonicalContract):
    """One complete terminal supervisor-attempt efficiency receipt."""

    SCHEMA: ClassVar[str] = EFFICIENCY_RECEIPT_SCHEMA

    task_reference: str
    goal_reference: str
    attempt: int
    scenario: EfficiencyScenario
    repository_tree_digest: str
    policy_digest: str
    context_digest: str
    input_digest: str
    output_digest: str
    elapsed_ms: int
    queue_delay_ms: int
    stages: tuple[StageTiming, ...]
    tokens: TokenUsage
    cache_observations: tuple[CacheObservation, ...]
    retries: tuple[RetryObservation, ...]
    inference_cost_microunits: int
    validation: WorkCost
    proof: WorkCost
    changed_scope: ChangedScope
    artifacts: tuple[ArtifactReference, ...]
    evidence: EvidenceDelta
    terminal: TerminalAcceptance
    provider_reference: str = ""
    related_task_references: tuple[str, ...] = ()
    conflict_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, required in (
            ("task_reference", True),
            ("goal_reference", True),
            ("provider_reference", False),
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    field_name=name,
                    required=required,
                    max_bytes=MAX_REFERENCE_BYTES,
                ),
            )
        object.__setattr__(
            self,
            "attempt",
            _integer(
                self.attempt,
                field_name="attempt",
                minimum=1,
                maximum=MAX_RETRIES + 1,
            ),
        )
        object.__setattr__(
            self,
            "scenario",
            _enum(self.scenario, EfficiencyScenario, field_name="scenario"),
        )
        for name, required in (
            ("repository_tree_digest", True),
            ("policy_digest", True),
            ("context_digest", True),
            ("input_digest", False),
            ("output_digest", False),
        ):
            object.__setattr__(
                self,
                name,
                _digest(
                    getattr(self, name),
                    field_name=name,
                    required=required,
                ),
            )
        for name in ("elapsed_ms", "queue_delay_ms"):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    maximum=MAX_DURATION_MS,
                ),
            )
        if self.queue_delay_ms > self.elapsed_ms:
            raise ContractValidationError(
                "queue_delay_ms cannot exceed elapsed_ms"
            )

        stages = _coerce_records(
            self.stages,
            StageTiming,
            StageTiming.from_dict,
            field_name="stages",
            maximum=MAX_STAGES,
        )
        stages = tuple(sorted(stages, key=lambda item: item.stage.value))
        if len({item.stage for item in stages}) != len(stages):
            raise ContractValidationError("stages must contain unique stage names")
        if any(item.latency_ms > self.elapsed_ms for item in stages):
            raise ContractValidationError(
                "individual stage latency cannot exceed elapsed_ms"
            )
        object.__setattr__(self, "stages", stages)

        tokens = self.tokens
        if isinstance(tokens, Mapping):
            tokens = TokenUsage.from_dict(tokens)
        if not isinstance(tokens, TokenUsage):
            raise ContractValidationError("tokens must be TokenUsage")
        object.__setattr__(self, "tokens", tokens)
        if tokens.input_tokens and not self.input_digest:
            raise ContractValidationError(
                "input tokens require an input_digest"
            )
        if tokens.output_tokens and not self.output_digest:
            raise ContractValidationError(
                "output tokens require an output_digest"
            )

        cache_observations = _coerce_records(
            self.cache_observations,
            CacheObservation,
            CacheObservation.from_dict,
            field_name="cache_observations",
            maximum=MAX_CACHE_OBSERVATIONS,
        )
        cache_observations = tuple(
            sorted(
                cache_observations,
                key=lambda item: (
                    item.namespace,
                    item.key_digest,
                    item.disposition.value,
                ),
            )
        )
        keys = [
            (item.namespace, item.key_digest) for item in cache_observations
        ]
        if len(set(keys)) != len(keys):
            raise ContractValidationError(
                "cache observations must be unique by namespace and key digest"
            )
        object.__setattr__(
            self, "cache_observations", cache_observations
        )

        retries = _coerce_records(
            self.retries,
            RetryObservation,
            RetryObservation.from_dict,
            field_name="retries",
            maximum=MAX_RETRIES,
        )
        retries = tuple(sorted(retries, key=lambda item: item.attempt))
        expected_attempts = tuple(range(2, len(retries) + 2))
        if tuple(item.attempt for item in retries) != expected_attempts:
            raise ContractValidationError(
                "retry attempt numbers must be unique and contiguous from 2"
            )
        if self.attempt != len(retries) + 1:
            raise ContractValidationError(
                "attempt must equal the initial attempt plus retry count"
            )
        retry_input = sum(item.tokens.input_tokens for item in retries)
        retry_output = sum(item.tokens.output_tokens for item in retries)
        retry_reused = sum(item.tokens.reused_tokens for item in retries)
        if (
            retry_input > tokens.input_tokens
            or retry_output > tokens.output_tokens
            or retry_reused > tokens.reused_tokens
        ):
            raise ContractValidationError(
                "retry token accounting cannot exceed total token accounting"
            )
        object.__setattr__(self, "retries", retries)

        object.__setattr__(
            self,
            "inference_cost_microunits",
            _integer(
                self.inference_cost_microunits,
                field_name="inference_cost_microunits",
                maximum=MAX_COST_MICROUNITS,
            ),
        )
        for name in ("validation", "proof"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = WorkCost.from_dict(value)
            if not isinstance(value, WorkCost):
                raise ContractValidationError(f"{name} must be WorkCost")
            if value.duration_ms > self.elapsed_ms:
                raise ContractValidationError(
                    f"{name} duration cannot exceed elapsed_ms"
                )
            object.__setattr__(self, name, value)

        changed_scope = self.changed_scope
        if isinstance(changed_scope, Mapping):
            changed_scope = ChangedScope.from_dict(changed_scope)
        if not isinstance(changed_scope, ChangedScope):
            raise ContractValidationError(
                "changed_scope must be ChangedScope"
            )
        object.__setattr__(self, "changed_scope", changed_scope)

        artifacts = _coerce_records(
            self.artifacts,
            ArtifactReference,
            ArtifactReference.from_dict,
            field_name="artifacts",
            maximum=MAX_ARTIFACT_REFERENCES,
        )
        artifacts = tuple(
            sorted(artifacts, key=lambda item: item.reference_id)
        )
        if len({item.reference_id for item in artifacts}) != len(artifacts):
            raise ContractValidationError(
                "artifact reference IDs must be unique"
            )
        object.__setattr__(self, "artifacts", artifacts)

        evidence = self.evidence
        if isinstance(evidence, Mapping):
            evidence = EvidenceDelta.from_dict(evidence)
        if not isinstance(evidence, EvidenceDelta):
            raise ContractValidationError("evidence must be EvidenceDelta")
        object.__setattr__(self, "evidence", evidence)

        terminal = self.terminal
        if isinstance(terminal, Mapping):
            terminal = TerminalAcceptance.from_dict(terminal)
        if not isinstance(terminal, TerminalAcceptance):
            raise ContractValidationError(
                "terminal must be TerminalAcceptance"
            )
        object.__setattr__(self, "terminal", terminal)

        object.__setattr__(
            self,
            "related_task_references",
            _strings(
                self.related_task_references,
                field_name="related_task_references",
                maximum=MAX_RELATED_TASK_REFERENCES,
            ),
        )
        object.__setattr__(
            self,
            "conflict_references",
            _strings(
                self.conflict_references,
                field_name="conflict_references",
                maximum=MAX_CONFLICT_REFERENCES,
            ),
        )
        if self.task_reference in self.related_task_references:
            raise ContractValidationError(
                "related_task_references cannot contain the current task"
            )

        self._validate_state()
        _record_size(
            self,
            maximum=MAX_SERIALIZED_RECEIPT_BYTES,
            name="efficiency receipt",
        )

    def _validate_state(self) -> None:
        if self.terminal.accepted:
            if self.validation.status is not WorkStatus.PASSED:
                raise ContractValidationError(
                    "accepted work requires passed validation"
                )
            if self.proof.status not in {
                WorkStatus.PASSED,
                WorkStatus.NOT_REQUIRED,
            }:
                raise ContractValidationError(
                    "accepted work cannot have failed or skipped required proof"
                )
            if not self.artifacts:
                raise ContractValidationError(
                    "accepted work requires at least one artifact reference"
                )
            if not self.evidence.terminal_references:
                raise ContractValidationError(
                    "accepted work requires terminal evidence references"
                )

        if self.scenario is EfficiencyScenario.COLD:
            if not self.cache_observations or any(
                item.disposition is CacheDisposition.HIT
                for item in self.cache_observations
            ):
                raise ContractValidationError(
                    "cold scenario requires non-hit cache observations"
                )
            if self.tokens.reused_tokens:
                raise ContractValidationError(
                    "cold scenario cannot report reused tokens"
                )
        elif self.scenario is EfficiencyScenario.WARM:
            if not any(
                item.disposition is CacheDisposition.HIT
                for item in self.cache_observations
            ):
                raise ContractValidationError(
                    "warm scenario requires a cache hit"
                )
            if not self.tokens.reused_tokens:
                raise ContractValidationError(
                    "warm scenario requires reused input tokens"
                )
        elif self.scenario is EfficiencyScenario.FAILED:
            if self.terminal.outcome is not TerminalOutcome.FAILED:
                raise ContractValidationError(
                    "failed scenario requires a failed terminal outcome"
                )
        elif self.scenario is EfficiencyScenario.REPAIRED:
            if not self.retries or not self.terminal.accepted:
                raise ContractValidationError(
                    "repaired scenario requires retries and acceptance"
                )
        elif self.scenario is EfficiencyScenario.PARALLEL_INDEPENDENT:
            if not self.related_task_references or self.conflict_references:
                raise ContractValidationError(
                    "parallel-independent scenario requires related tasks and no conflicts"
                )
        elif self.scenario is EfficiencyScenario.CONFLICTING:
            if not self.conflict_references:
                raise ContractValidationError(
                    "conflicting scenario requires conflict references"
                )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def accepted(self) -> bool:
        return self.terminal.accepted

    @property
    def retry_count(self) -> int:
        return len(self.retries)

    @property
    def input_tokens(self) -> int:
        return self.tokens.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.tokens.output_tokens

    @property
    def reused_tokens(self) -> int:
        return self.tokens.reused_tokens

    @property
    def total_cost_microunits(self) -> int:
        return (
            self.inference_cost_microunits
            + self.validation.cost_microunits
            + self.proof.cost_microunits
        )

    @property
    def raw_evidence_gain(self) -> int:
        return self.evidence.gain

    @property
    def accepted_evidence_gain(self) -> int:
        return self.evidence.gain if self.accepted else 0

    @property
    def evidence_gain_per_thousand_input_tokens(self) -> float:
        if not self.input_tokens:
            return 0.0
        return self.accepted_evidence_gain * 1000 / self.input_tokens

    def stage_latency_ms(self, stage: StageName | str) -> int:
        normalized = _enum(stage, StageName, field_name="stage")
        for timing in self.stages:
            if timing.stage is normalized:
                return timing.latency_ms
        return 0

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "task_reference": self.task_reference,
            "goal_reference": self.goal_reference,
            "provider_reference": self.provider_reference,
            "attempt": self.attempt,
            "scenario": self.scenario,
            "repository_tree_digest": self.repository_tree_digest,
            "policy_digest": self.policy_digest,
            "context_digest": self.context_digest,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "elapsed_ms": self.elapsed_ms,
            "queue_delay_ms": self.queue_delay_ms,
            "stages": self.stages,
            "tokens": self.tokens,
            "cache_observations": self.cache_observations,
            "retries": self.retries,
            "retry_count": self.retry_count,
            "inference_cost_microunits": self.inference_cost_microunits,
            "validation": self.validation,
            "proof": self.proof,
            "total_cost_microunits": self.total_cost_microunits,
            "changed_scope": self.changed_scope,
            "artifacts": self.artifacts,
            "evidence": self.evidence,
            "accepted_evidence_gain": self.accepted_evidence_gain,
            "terminal": self.terminal,
            "related_task_references": self.related_task_references,
            "conflict_references": self.conflict_references,
        }

    def to_dict(self, *, include_receipt_id: bool = False) -> dict[str, Any]:
        payload = super().to_dict()
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EfficiencyReceipt":
        _schema(payload, cls.SCHEMA, "efficiency receipt")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "task_reference",
                "goal_reference",
                "provider_reference",
                "attempt",
                "scenario",
                "repository_tree_digest",
                "policy_digest",
                "context_digest",
                "input_digest",
                "output_digest",
                "elapsed_ms",
                "queue_delay_ms",
                "stages",
                "tokens",
                "cache_observations",
                "retries",
                "retry_count",
                "inference_cost_microunits",
                "validation",
                "proof",
                "total_cost_microunits",
                "changed_scope",
                "artifacts",
                "evidence",
                "accepted_evidence_gain",
                "terminal",
                "related_task_references",
                "conflict_references",
                "receipt_id",
                "content_id",
            },
            artifact_name="efficiency receipt",
        )
        result = cls(
            task_reference=payload.get("task_reference", ""),
            goal_reference=payload.get("goal_reference", ""),
            provider_reference=payload.get("provider_reference", ""),
            attempt=payload.get("attempt", 0),
            scenario=payload.get("scenario", EfficiencyScenario.OBSERVED),
            repository_tree_digest=payload.get(
                "repository_tree_digest", ""
            ),
            policy_digest=payload.get("policy_digest", ""),
            context_digest=payload.get("context_digest", ""),
            input_digest=payload.get("input_digest", ""),
            output_digest=payload.get("output_digest", ""),
            elapsed_ms=payload.get("elapsed_ms", 0),
            queue_delay_ms=payload.get("queue_delay_ms", 0),
            stages=tuple(
                StageTiming.from_dict(item)
                for item in payload.get("stages") or ()
            ),
            tokens=TokenUsage.from_dict(payload.get("tokens") or {}),
            cache_observations=tuple(
                CacheObservation.from_dict(item)
                for item in payload.get("cache_observations") or ()
            ),
            retries=tuple(
                RetryObservation.from_dict(item)
                for item in payload.get("retries") or ()
            ),
            inference_cost_microunits=payload.get(
                "inference_cost_microunits", 0
            ),
            validation=WorkCost.from_dict(payload.get("validation") or {}),
            proof=WorkCost.from_dict(payload.get("proof") or {}),
            changed_scope=ChangedScope.from_dict(
                payload.get("changed_scope") or {}
            ),
            artifacts=tuple(
                ArtifactReference.from_dict(item)
                for item in payload.get("artifacts") or ()
            ),
            evidence=EvidenceDelta.from_dict(payload.get("evidence") or {}),
            terminal=TerminalAcceptance.from_dict(
                payload.get("terminal") or {}
            ),
            related_task_references=tuple(
                payload.get("related_task_references") or ()
            ),
            conflict_references=tuple(
                payload.get("conflict_references") or ()
            ),
        )
        claims = {
            "retry_count": result.retry_count,
            "total_cost_microunits": result.total_cost_microunits,
            "accepted_evidence_gain": result.accepted_evidence_gain,
        }
        for name, actual in claims.items():
            if payload.get(name, actual) != actual:
                raise ContractValidationError(
                    f"{name} claim does not match receipt contents"
                )
        _claim(payload, result.receipt_id, "receipt_id", "content_id")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "EfficiencyReceipt":
        return cls.from_dict(_load_json(value, artifact_name="receipt"))


@dataclass(frozen=True)
class EfficiencyReport(CanonicalContract):
    """Deterministic aggregate of unique efficiency receipts."""

    SCHEMA: ClassVar[str] = EFFICIENCY_REPORT_SCHEMA

    receipt_ids: tuple[str, ...]
    task_references: tuple[str, ...]
    accepted_task_references: tuple[str, ...]
    receipt_count: int
    accepted_receipt_count: int
    total_elapsed_ms: int
    total_queue_delay_ms: int
    total_input_tokens: int
    total_output_tokens: int
    total_reused_tokens: int
    total_retry_count: int
    stage_latency_ms: Mapping[str, int]
    stage_invocation_counts: Mapping[str, int]
    cache_outcome_counts: Mapping[str, int]
    total_cache_bytes_reused: int
    total_validation_duration_ms: int
    total_proof_duration_ms: int
    total_inference_cost_microunits: int
    total_validation_cost_microunits: int
    total_proof_cost_microunits: int
    total_cost_microunits: int
    total_changed_file_count: int
    total_changed_symbol_count: int
    total_lines_added: int
    total_lines_deleted: int
    artifact_reference_count: int
    accepted_evidence_gain: int
    cost_per_accepted_task_ratio: ExactRatio
    evidence_gain_per_thousand_input_tokens_ratio: ExactRatio

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt_ids",
            _strings(
                self.receipt_ids,
                field_name="receipt_ids",
                maximum=MAX_RECEIPTS_PER_REPORT,
                max_item_bytes=MAX_REFERENCE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "task_references",
            _strings(
                self.task_references,
                field_name="task_references",
                maximum=MAX_RECEIPTS_PER_REPORT,
            ),
        )
        object.__setattr__(
            self,
            "accepted_task_references",
            _strings(
                self.accepted_task_references,
                field_name="accepted_task_references",
                maximum=MAX_RECEIPTS_PER_REPORT,
            ),
        )
        numeric_limits = {
            "receipt_count": MAX_RECEIPTS_PER_REPORT,
            "accepted_receipt_count": MAX_RECEIPTS_PER_REPORT,
            "total_elapsed_ms": MAX_DURATION_MS * MAX_RECEIPTS_PER_REPORT,
            "total_queue_delay_ms": MAX_DURATION_MS * MAX_RECEIPTS_PER_REPORT,
            "total_input_tokens": MAX_TOKENS * MAX_RECEIPTS_PER_REPORT,
            "total_output_tokens": MAX_TOKENS * MAX_RECEIPTS_PER_REPORT,
            "total_reused_tokens": MAX_TOKENS * MAX_RECEIPTS_PER_REPORT,
            "total_retry_count": MAX_RETRIES * MAX_RECEIPTS_PER_REPORT,
            "total_cache_bytes_reused": MAX_BYTES * MAX_RECEIPTS_PER_REPORT,
            "total_validation_duration_ms": (
                MAX_DURATION_MS * MAX_RECEIPTS_PER_REPORT
            ),
            "total_proof_duration_ms": (
                MAX_DURATION_MS * MAX_RECEIPTS_PER_REPORT
            ),
            "total_inference_cost_microunits": (
                MAX_COST_MICROUNITS * MAX_RECEIPTS_PER_REPORT
            ),
            "total_validation_cost_microunits": (
                MAX_COST_MICROUNITS * MAX_RECEIPTS_PER_REPORT
            ),
            "total_proof_cost_microunits": (
                MAX_COST_MICROUNITS * MAX_RECEIPTS_PER_REPORT
            ),
            "total_cost_microunits": (
                MAX_COST_MICROUNITS * 3 * MAX_RECEIPTS_PER_REPORT
            ),
            "total_changed_file_count": (
                MAX_CHANGED_PATHS * MAX_RECEIPTS_PER_REPORT
            ),
            "total_changed_symbol_count": (
                MAX_CHANGED_SYMBOLS * MAX_RECEIPTS_PER_REPORT
            ),
            "total_lines_added": MAX_CHANGE_LINES * MAX_RECEIPTS_PER_REPORT,
            "total_lines_deleted": MAX_CHANGE_LINES * MAX_RECEIPTS_PER_REPORT,
            "artifact_reference_count": (
                MAX_ARTIFACT_REFERENCES * MAX_RECEIPTS_PER_REPORT
            ),
            "accepted_evidence_gain": (
                MAX_EVIDENCE_REFERENCES * MAX_RECEIPTS_PER_REPORT
            ),
        }
        for name, maximum in numeric_limits.items():
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    maximum=maximum,
                ),
            )
        if self.receipt_count != len(self.receipt_ids):
            raise ContractValidationError(
                "receipt_count must match unique receipt IDs"
            )
        if self.accepted_receipt_count < len(
            self.accepted_task_references
        ):
            raise ContractValidationError(
                "accepted receipt count cannot be below accepted task count"
            )
        if not set(self.accepted_task_references).issubset(
            self.task_references
        ):
            raise ContractValidationError(
                "accepted tasks must be present in task references"
            )
        if self.total_reused_tokens > self.total_input_tokens:
            raise ContractValidationError(
                "total reused tokens cannot exceed total input tokens"
            )
        if self.total_cost_microunits != (
            self.total_inference_cost_microunits
            + self.total_validation_cost_microunits
            + self.total_proof_cost_microunits
        ):
            raise ContractValidationError(
                "total cost must equal inference, validation, and proof cost"
            )

        if not isinstance(self.cache_outcome_counts, Mapping):
            raise ContractValidationError(
                "cache_outcome_counts must be an object"
            )
        expected_keys = {item.value for item in CacheDisposition}
        if set(self.cache_outcome_counts) != expected_keys:
            raise ContractValidationError(
                "cache_outcome_counts must contain every cache disposition"
            )
        normalized_counts: dict[str, int] = {}
        for key in sorted(expected_keys):
            normalized_counts[key] = _integer(
                self.cache_outcome_counts[key],
                field_name=f"cache_outcome_counts.{key}",
                maximum=MAX_CACHE_OBSERVATIONS * MAX_RECEIPTS_PER_REPORT,
            )
        object.__setattr__(self, "cache_outcome_counts", normalized_counts)

        stage_keys = {item.value for item in StageName}
        for name in ("stage_latency_ms", "stage_invocation_counts"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or set(value) != stage_keys:
                raise ContractValidationError(
                    f"{name} must contain every known stage"
                )
            maximum = (
                MAX_DURATION_MS * MAX_RECEIPTS_PER_REPORT
                if name == "stage_latency_ms"
                else MAX_OPERATIONS * MAX_RECEIPTS_PER_REPORT
            )
            normalized_stages: dict[str, int] = {}
            for key in sorted(stage_keys):
                normalized_stages[key] = _integer(
                    value[key],
                    field_name=f"{name}.{key}",
                    maximum=maximum,
                )
            object.__setattr__(self, name, normalized_stages)

        for name in (
            "cost_per_accepted_task_ratio",
            "evidence_gain_per_thousand_input_tokens_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = ExactRatio.from_dict(value)
            if not isinstance(value, ExactRatio):
                raise ContractValidationError(f"{name} must be ExactRatio")
            object.__setattr__(self, name, value)

        expected_cost_ratio = ExactRatio(
            self.total_cost_microunits,
            len(self.accepted_task_references),
            1,
        )
        expected_evidence_ratio = ExactRatio(
            self.accepted_evidence_gain,
            self.total_input_tokens,
            1000,
        )
        if self.cost_per_accepted_task_ratio != expected_cost_ratio:
            raise ContractValidationError(
                "cost-per-accepted-task ratio is inconsistent"
            )
        if (
            self.evidence_gain_per_thousand_input_tokens_ratio
            != expected_evidence_ratio
        ):
            raise ContractValidationError(
                "evidence-gain ratio is inconsistent"
            )
        _record_size(
            self,
            maximum=MAX_SERIALIZED_REPORT_BYTES,
            name="efficiency report",
        )

    @property
    def report_id(self) -> str:
        return self.content_id

    @property
    def accepted_task_count(self) -> int:
        return len(self.accepted_task_references)

    @property
    def cost_per_accepted_task_microunits(self) -> float:
        return self.cost_per_accepted_task_ratio.value

    @property
    def evidence_gain_per_thousand_input_tokens(self) -> float:
        return self.evidence_gain_per_thousand_input_tokens_ratio.value

    @property
    def cost_per_accepted_task(self) -> float:
        """Compatibility spelling; the value remains in integer microunits."""

        return self.cost_per_accepted_task_microunits

    @property
    def evidence_gain_per_1k_input_tokens(self) -> float:
        return self.evidence_gain_per_thousand_input_tokens

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "receipt_ids": self.receipt_ids,
            "task_references": self.task_references,
            "accepted_task_references": self.accepted_task_references,
            "receipt_count": self.receipt_count,
            "accepted_receipt_count": self.accepted_receipt_count,
            "accepted_task_count": self.accepted_task_count,
            "total_elapsed_ms": self.total_elapsed_ms,
            "total_queue_delay_ms": self.total_queue_delay_ms,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_reused_tokens": self.total_reused_tokens,
            "total_retry_count": self.total_retry_count,
            "stage_latency_ms": self.stage_latency_ms,
            "stage_invocation_counts": self.stage_invocation_counts,
            "cache_outcome_counts": self.cache_outcome_counts,
            "total_cache_bytes_reused": self.total_cache_bytes_reused,
            "total_validation_duration_ms": (
                self.total_validation_duration_ms
            ),
            "total_proof_duration_ms": self.total_proof_duration_ms,
            "total_inference_cost_microunits": (
                self.total_inference_cost_microunits
            ),
            "total_validation_cost_microunits": (
                self.total_validation_cost_microunits
            ),
            "total_proof_cost_microunits": self.total_proof_cost_microunits,
            "total_cost_microunits": self.total_cost_microunits,
            "total_changed_file_count": self.total_changed_file_count,
            "total_changed_symbol_count": self.total_changed_symbol_count,
            "total_lines_added": self.total_lines_added,
            "total_lines_deleted": self.total_lines_deleted,
            "artifact_reference_count": self.artifact_reference_count,
            "accepted_evidence_gain": self.accepted_evidence_gain,
            "cost_per_accepted_task_ratio": (
                self.cost_per_accepted_task_ratio
            ),
            "evidence_gain_per_thousand_input_tokens_ratio": (
                self.evidence_gain_per_thousand_input_tokens_ratio
            ),
        }

    def to_dict(self, *, include_report_id: bool = False) -> dict[str, Any]:
        payload = super().to_dict()
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EfficiencyReport":
        _schema(payload, cls.SCHEMA, "efficiency report")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "receipt_ids",
                "task_references",
                "accepted_task_references",
                "receipt_count",
                "accepted_receipt_count",
                "accepted_task_count",
                "total_elapsed_ms",
                "total_queue_delay_ms",
                "total_input_tokens",
                "total_output_tokens",
                "total_reused_tokens",
                "total_retry_count",
                "stage_latency_ms",
                "stage_invocation_counts",
                "cache_outcome_counts",
                "total_cache_bytes_reused",
                "total_validation_duration_ms",
                "total_proof_duration_ms",
                "total_inference_cost_microunits",
                "total_validation_cost_microunits",
                "total_proof_cost_microunits",
                "total_cost_microunits",
                "total_changed_file_count",
                "total_changed_symbol_count",
                "total_lines_added",
                "total_lines_deleted",
                "artifact_reference_count",
                "accepted_evidence_gain",
                "cost_per_accepted_task_ratio",
                "evidence_gain_per_thousand_input_tokens_ratio",
                "report_id",
                "content_id",
            },
            artifact_name="efficiency report",
        )
        result = cls(
            receipt_ids=tuple(payload.get("receipt_ids") or ()),
            task_references=tuple(payload.get("task_references") or ()),
            accepted_task_references=tuple(
                payload.get("accepted_task_references") or ()
            ),
            receipt_count=payload.get("receipt_count", 0),
            accepted_receipt_count=payload.get(
                "accepted_receipt_count", 0
            ),
            total_elapsed_ms=payload.get("total_elapsed_ms", 0),
            total_queue_delay_ms=payload.get("total_queue_delay_ms", 0),
            total_input_tokens=payload.get("total_input_tokens", 0),
            total_output_tokens=payload.get("total_output_tokens", 0),
            total_reused_tokens=payload.get("total_reused_tokens", 0),
            total_retry_count=payload.get("total_retry_count", 0),
            stage_latency_ms=payload.get("stage_latency_ms") or {},
            stage_invocation_counts=payload.get(
                "stage_invocation_counts"
            )
            or {},
            cache_outcome_counts=payload.get("cache_outcome_counts") or {},
            total_cache_bytes_reused=payload.get(
                "total_cache_bytes_reused", 0
            ),
            total_validation_duration_ms=payload.get(
                "total_validation_duration_ms", 0
            ),
            total_proof_duration_ms=payload.get(
                "total_proof_duration_ms", 0
            ),
            total_inference_cost_microunits=payload.get(
                "total_inference_cost_microunits", 0
            ),
            total_validation_cost_microunits=payload.get(
                "total_validation_cost_microunits", 0
            ),
            total_proof_cost_microunits=payload.get(
                "total_proof_cost_microunits", 0
            ),
            total_cost_microunits=payload.get(
                "total_cost_microunits", 0
            ),
            total_changed_file_count=payload.get(
                "total_changed_file_count", 0
            ),
            total_changed_symbol_count=payload.get(
                "total_changed_symbol_count", 0
            ),
            total_lines_added=payload.get("total_lines_added", 0),
            total_lines_deleted=payload.get("total_lines_deleted", 0),
            artifact_reference_count=payload.get(
                "artifact_reference_count", 0
            ),
            accepted_evidence_gain=payload.get(
                "accepted_evidence_gain", 0
            ),
            cost_per_accepted_task_ratio=ExactRatio.from_dict(
                payload.get("cost_per_accepted_task_ratio") or {}
            ),
            evidence_gain_per_thousand_input_tokens_ratio=(
                ExactRatio.from_dict(
                    payload.get(
                        "evidence_gain_per_thousand_input_tokens_ratio"
                    )
                    or {}
                )
            ),
        )
        if payload.get("accepted_task_count", result.accepted_task_count) != (
            result.accepted_task_count
        ):
            raise ContractValidationError(
                "accepted_task_count claim does not match task references"
            )
        _claim(payload, result.report_id, "report_id", "content_id")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "EfficiencyReport":
        return cls.from_dict(_load_json(value, artifact_name="report"))


def aggregate_efficiency_receipts(
    receipts: Iterable[EfficiencyReceipt | Mapping[str, Any]],
) -> EfficiencyReport:
    """Aggregate unique receipts without dropping failed/retried attempt cost."""

    normalized: list[EfficiencyReceipt] = []
    for index, value in enumerate(receipts):
        if index >= MAX_RECEIPTS_PER_REPORT:
            raise ContractValidationError(
                f"receipt aggregate exceeds {MAX_RECEIPTS_PER_REPORT} items"
            )
        if isinstance(value, EfficiencyReceipt):
            receipt = value
        elif isinstance(value, Mapping):
            receipt = EfficiencyReceipt.from_dict(value)
        else:
            raise ContractValidationError(
                f"receipts[{index}] must be an EfficiencyReceipt"
            )
        normalized.append(receipt)

    ids = [item.receipt_id for item in normalized]
    if len(ids) != len(set(ids)):
        raise ContractValidationError(
            "duplicate receipt identity in efficiency aggregate"
        )

    # Multiple accepted receipts for one task would double count evidence while
    # retaining a one-task denominator.  Reject that invalid state explicitly.
    accepted_by_task: dict[str, EfficiencyReceipt] = {}
    for receipt in normalized:
        if not receipt.accepted:
            continue
        if receipt.task_reference in accepted_by_task:
            raise ContractValidationError(
                "a task may have only one accepted receipt in an aggregate"
            )
        accepted_by_task[receipt.task_reference] = receipt

    cache_counts = {item.value: 0 for item in CacheDisposition}
    stage_latency = {item.value: 0 for item in StageName}
    stage_invocations = {item.value: 0 for item in StageName}
    for receipt in normalized:
        for observation in receipt.cache_observations:
            cache_counts[observation.disposition.value] += 1
        for timing in receipt.stages:
            stage_latency[timing.stage.value] += timing.latency_ms
            stage_invocations[timing.stage.value] += timing.invocation_count

    total_inference = sum(
        item.inference_cost_microunits for item in normalized
    )
    total_validation = sum(
        item.validation.cost_microunits for item in normalized
    )
    total_proof = sum(item.proof.cost_microunits for item in normalized)
    total_cost = total_inference + total_validation + total_proof
    total_input = sum(item.input_tokens for item in normalized)
    evidence_gain = sum(
        item.accepted_evidence_gain for item in normalized
    )

    return EfficiencyReport(
        receipt_ids=tuple(ids),
        task_references=tuple(item.task_reference for item in normalized),
        accepted_task_references=tuple(accepted_by_task),
        receipt_count=len(normalized),
        accepted_receipt_count=sum(item.accepted for item in normalized),
        total_elapsed_ms=sum(item.elapsed_ms for item in normalized),
        total_queue_delay_ms=sum(item.queue_delay_ms for item in normalized),
        total_input_tokens=total_input,
        total_output_tokens=sum(item.output_tokens for item in normalized),
        total_reused_tokens=sum(item.reused_tokens for item in normalized),
        total_retry_count=sum(item.retry_count for item in normalized),
        stage_latency_ms=stage_latency,
        stage_invocation_counts=stage_invocations,
        cache_outcome_counts=cache_counts,
        total_cache_bytes_reused=sum(
            observation.bytes_reused
            for item in normalized
            for observation in item.cache_observations
        ),
        total_validation_duration_ms=sum(
            item.validation.duration_ms for item in normalized
        ),
        total_proof_duration_ms=sum(
            item.proof.duration_ms for item in normalized
        ),
        total_inference_cost_microunits=total_inference,
        total_validation_cost_microunits=total_validation,
        total_proof_cost_microunits=total_proof,
        total_cost_microunits=total_cost,
        total_changed_file_count=sum(
            item.changed_scope.changed_file_count for item in normalized
        ),
        total_changed_symbol_count=sum(
            item.changed_scope.changed_symbol_count for item in normalized
        ),
        total_lines_added=sum(
            item.changed_scope.lines_added for item in normalized
        ),
        total_lines_deleted=sum(
            item.changed_scope.lines_deleted for item in normalized
        ),
        artifact_reference_count=sum(
            len(item.artifacts) for item in normalized
        ),
        accepted_evidence_gain=evidence_gain,
        cost_per_accepted_task_ratio=ExactRatio(
            total_cost, len(accepted_by_task), 1
        ),
        evidence_gain_per_thousand_input_tokens_ratio=ExactRatio(
            evidence_gain, total_input, 1000
        ),
    )


def _fixture_digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixture_work(
    kind: str,
    *,
    passed: bool = True,
    required: bool = True,
    duration_ms: int,
    cost_microunits: int,
) -> WorkCost:
    if not required:
        return WorkCost()
    return WorkCost(
        status=WorkStatus.PASSED if passed else WorkStatus.FAILED,
        duration_ms=duration_ms,
        cost_microunits=cost_microunits,
        operation_count=1,
        evidence_references=(f"receipt:{kind}",) if passed else (),
    )


def _baseline_receipt(
    scenario: EfficiencyScenario,
    *,
    task: str,
    outcome: TerminalOutcome = TerminalOutcome.ACCEPTED,
    input_tokens: int = 4_000,
    output_tokens: int = 600,
    reused_tokens: int = 0,
    cache_disposition: CacheDisposition = CacheDisposition.MISS,
    retries: tuple[RetryObservation, ...] = (),
    validation_passed: bool = True,
    related: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
) -> EfficiencyReceipt:
    accepted = outcome is TerminalOutcome.ACCEPTED
    evidence_terminal = (
        ("evidence:syntax", "evidence:unit", "evidence:acceptance")
        if accepted
        else ("evidence:diagnostic",)
    )
    return EfficiencyReceipt(
        task_reference=task,
        goal_reference="goal:asi-g010",
        provider_reference="provider:fixture",
        attempt=len(retries) + 1,
        scenario=scenario,
        repository_tree_digest=_fixture_digest(f"{task}:tree"),
        policy_digest=_fixture_digest("baseline-policy-v1"),
        context_digest=_fixture_digest(f"{task}:context"),
        input_digest=_fixture_digest(f"{task}:input"),
        output_digest=_fixture_digest(f"{task}:output"),
        elapsed_ms=12_000 + len(retries) * 2_000,
        queue_delay_ms=1_000,
        stages=(
            StageTiming(StageName.ANALYSIS, 1_500),
            StageTiming(StageName.INFERENCE, 4_000),
            StageTiming(StageName.IMPLEMENTATION, 3_000),
            StageTiming(StageName.VALIDATION, 2_000),
        ),
        tokens=TokenUsage(input_tokens, output_tokens, reused_tokens),
        cache_observations=(
            CacheObservation(
                namespace="analysis",
                disposition=cache_disposition,
                key_digest=_fixture_digest(f"{task}:cache-key"),
                lookup_latency_ms=5 if cache_disposition is CacheDisposition.HIT else 20,
                bytes_reused=2048 if cache_disposition is CacheDisposition.HIT else 0,
            ),
        ),
        retries=retries,
        inference_cost_microunits=input_tokens + output_tokens * 2,
        validation=_fixture_work(
            "validation",
            passed=validation_passed,
            duration_ms=2_000,
            cost_microunits=400,
        ),
        proof=_fixture_work(
            "proof",
            required=False,
            duration_ms=0,
            cost_microunits=0,
        ),
        changed_scope=ChangedScope(
            paths=(f"src/{task.split(':')[-1]}.py",) if accepted else (),
            symbols=(f"{task.split(':')[-1]}.run",) if accepted else (),
            lines_added=20 if accepted else 0,
            lines_deleted=3 if accepted else 0,
        ),
        artifacts=(
            ArtifactReference(
                reference_id=f"artifact:{task}",
                digest=_fixture_digest(f"{task}:artifact"),
                kind="patch",
                byte_count=1024,
                media_type="text/x-diff",
            ),
        )
        if accepted
        else (),
        evidence=EvidenceDelta(
            baseline_references=("evidence:syntax",),
            terminal_references=evidence_terminal,
        ),
        terminal=TerminalAcceptance(
            outcome=outcome,
            reason_codes=("accepted",) if accepted else (outcome.value,),
            acceptance_digest=(
                _fixture_digest(f"{task}:acceptance") if accepted else ""
            ),
        ),
        related_task_references=related,
        conflict_references=conflicts,
    )


def build_efficiency_baseline_fixtures() -> dict[str, EfficiencyReceipt]:
    """Return deterministic cold/warm/failure/repair/parallel/conflict fixtures."""

    repair_retry = RetryObservation(
        attempt=2,
        reason_code="validation_failure",
        diagnostic_digest=_fixture_digest("repaired:diagnostic"),
        delta_context_digest=_fixture_digest("repaired:delta-context"),
        tokens=TokenUsage(900, 180, 300),
        latency_ms=2_000,
    )
    fixtures = {
        "cold": _baseline_receipt(
            EfficiencyScenario.COLD,
            task="task:cold",
        ),
        "warm": _baseline_receipt(
            EfficiencyScenario.WARM,
            task="task:warm",
            input_tokens=2_000,
            output_tokens=450,
            reused_tokens=1_400,
            cache_disposition=CacheDisposition.HIT,
        ),
        "failed": _baseline_receipt(
            EfficiencyScenario.FAILED,
            task="task:failed",
            outcome=TerminalOutcome.FAILED,
            validation_passed=False,
        ),
        "repaired": _baseline_receipt(
            EfficiencyScenario.REPAIRED,
            task="task:repaired",
            input_tokens=4_900,
            output_tokens=780,
            reused_tokens=300,
            retries=(repair_retry,),
        ),
        "parallel-independent": _baseline_receipt(
            EfficiencyScenario.PARALLEL_INDEPENDENT,
            task="task:parallel-a",
            related=("task:parallel-b",),
        ),
        "conflicting": _baseline_receipt(
            EfficiencyScenario.CONFLICTING,
            task="task:conflicting",
            outcome=TerminalOutcome.CONFLICTED,
            validation_passed=False,
            related=("task:conflict-peer",),
            conflicts=("conflict:path-overlap",),
        ),
    }
    return fixtures


# Compatibility-oriented names make the standalone contract easy to discover
# without requiring the package export changes intentionally deferred by the
# task board.
SupervisorEfficiencyReceipt = EfficiencyReceipt
SupervisorEfficiencyReport = EfficiencyReport
EfficiencyAggregate = EfficiencyReport
aggregate_receipts = aggregate_efficiency_receipts
make_baseline_fixtures = build_efficiency_baseline_fixtures


__all__ = [
    "ARTIFACT_REFERENCE_SCHEMA",
    "CACHE_OBSERVATION_SCHEMA",
    "CHANGED_SCOPE_SCHEMA",
    "EFFICIENCY_CONTRACT_VERSION",
    "EFFICIENCY_RECEIPT_SCHEMA",
    "EFFICIENCY_REPORT_SCHEMA",
    "EVIDENCE_DELTA_SCHEMA",
    "EXACT_RATIO_SCHEMA",
    "MAX_ARTIFACT_REFERENCES",
    "MAX_CACHE_OBSERVATIONS",
    "MAX_CHANGED_PATHS",
    "MAX_CHANGED_SYMBOLS",
    "MAX_DURATION_MS",
    "MAX_EVIDENCE_REFERENCES",
    "MAX_RECEIPTS_PER_REPORT",
    "MAX_RETRIES",
    "MAX_SERIALIZED_RECEIPT_BYTES",
    "MAX_STAGES",
    "MAX_TEXT_BYTES",
    "MAX_TOKENS",
    "RETRY_OBSERVATION_SCHEMA",
    "SCHEMA_VERSION",
    "STAGE_TIMING_SCHEMA",
    "TERMINAL_ACCEPTANCE_SCHEMA",
    "TOKEN_USAGE_SCHEMA",
    "WORK_COST_SCHEMA",
    "ArtifactReference",
    "CacheDisposition",
    "CacheObservation",
    "ChangedScope",
    "EfficiencyAggregate",
    "EfficiencyReceipt",
    "EfficiencyReport",
    "EfficiencyScenario",
    "EfficiencyValidationError",
    "EvidenceDelta",
    "ExactRatio",
    "RetryObservation",
    "StageName",
    "StageTiming",
    "SupervisorEfficiencyReceipt",
    "SupervisorEfficiencyReport",
    "TerminalAcceptance",
    "TerminalOutcome",
    "TokenUsage",
    "WorkCost",
    "WorkStatus",
    "aggregate_efficiency_receipts",
    "aggregate_receipts",
    "build_efficiency_baseline_fixtures",
    "make_baseline_fixtures",
]
