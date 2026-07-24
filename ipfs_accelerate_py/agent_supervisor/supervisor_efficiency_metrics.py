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
PAIRED_EFFICIENCY_CASE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/paired-efficiency-case@1"
)
PAIRED_EFFICIENCY_REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/paired-efficiency-report@3"
)
TERMINAL_ACCEPTED_WORK_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "terminal-accepted-work-evidence@1"
)
REQUIRED_CONTEXT_PROOF_BINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/required-context-proof-binding@1"
)
REQUIRED_CONTEXT_PROMOTION_REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "required-context-promotion-report@2"
)
DELTA_RETRY_PROOF_BINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/delta-retry-proof-binding@2"
)
DELTA_RETRY_PROMOTION_REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/delta-retry-promotion-report@3"
)

BASIS_POINTS = 10_000
DEFAULT_MINIMUM_INPUT_TOKEN_REDUCTION_BPS = 3_500
# Stable objective evidence term emitted by the accepted-work population gate.
# Context-budget and retry-delta terms are owned by their respective contracts;
# this module must not claim them from token receipts alone.
TERMINAL_ACCEPTED_WORK_EVIDENCE_ID = (
    "248026856102230635452423769994290240744"
)
# Verification-side binding to
# context_compiler.REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID.  Token measurements
# alone cannot emit this term; the promotion report must consume a typed,
# capsule-verified compiler result for every paired task.
REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID = (
    "208290439421789408250562066350459701853"
)
# Verification-side binding to context_compiler.DELTA_RETRY_EVIDENCE_ID.
# This module never emits the term from token measurements alone: a promotion
# report can cover it only after consuming a typed ContextDeltaReceipt whose
# content-addressed witness carries the same identifier.
DELTA_RETRY_CONTEXT_EVIDENCE_ID = (
    "306437607356117177048620815571362227127"
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


def _canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest for a bounded JSON-compatible value."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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


def _median_integer(values: Sequence[int]) -> int:
    """Return the deterministic integer median, flooring an even midpoint."""

    if not values:
        return 0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


@dataclass(frozen=True)
class PairedEfficiencyCase(CanonicalContract):
    """One same-task baseline/candidate accepted-work comparison.

    ``*_receipt_ids`` contain every attempt charged to the accepted task, not
    only the final acceptance receipt.  The separate terminal IDs prove that
    both arms reached an accepted terminal state.  This keeps retry and failed
    attempt tokens in the numerator without allowing failed-only tasks into
    the paired population.
    """

    SCHEMA: ClassVar[str] = PAIRED_EFFICIENCY_CASE_SCHEMA

    task_reference: str
    goal_reference: str
    repository_tree_digest: str
    policy_digest: str
    baseline_receipt_ids: tuple[str, ...]
    candidate_receipt_ids: tuple[str, ...]
    baseline_terminal_receipt_id: str
    candidate_terminal_receipt_id: str
    baseline_input_tokens: int
    candidate_input_tokens: int
    required_evidence_references: tuple[str, ...]
    baseline_covered_evidence_references: tuple[str, ...]
    candidate_covered_evidence_references: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("task_reference", "goal_reference"):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    field_name=name,
                    required=True,
                    max_bytes=MAX_REFERENCE_BYTES,
                ),
            )
        for name in ("repository_tree_digest", "policy_digest"):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), field_name=name),
            )
        for name in ("baseline_receipt_ids", "candidate_receipt_ids"):
            value = _strings(
                getattr(self, name),
                field_name=name,
                maximum=MAX_RECEIPTS_PER_REPORT,
                max_item_bytes=MAX_REFERENCE_BYTES,
            )
            if not value:
                raise ContractValidationError(
                    f"{name} must contain at least one charged attempt"
                )
            object.__setattr__(self, name, value)
        for name, receipt_ids in (
            ("baseline_terminal_receipt_id", self.baseline_receipt_ids),
            ("candidate_terminal_receipt_id", self.candidate_receipt_ids),
        ):
            value = _text(
                getattr(self, name),
                field_name=name,
                required=True,
                max_bytes=MAX_REFERENCE_BYTES,
            )
            if value not in receipt_ids:
                raise ContractValidationError(
                    f"{name} must identify one of its arm's charged receipts"
                )
            object.__setattr__(self, name, value)
        for name in ("baseline_input_tokens", "candidate_input_tokens"):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    maximum=MAX_TOKENS * MAX_RECEIPTS_PER_REPORT,
                ),
            )
        for name in (
            "required_evidence_references",
            "baseline_covered_evidence_references",
            "candidate_covered_evidence_references",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    maximum=MAX_EVIDENCE_REFERENCES,
                ),
            )
        required = set(self.required_evidence_references)
        if not required:
            raise ContractValidationError(
                "paired efficiency cases require authoritative evidence references"
            )
        if not set(self.baseline_covered_evidence_references).issubset(required):
            raise ContractValidationError(
                "baseline covered evidence must be a subset of required evidence"
            )
        if not set(self.candidate_covered_evidence_references).issubset(required):
            raise ContractValidationError(
                "candidate covered evidence must be a subset of required evidence"
            )
        _record_size(
            self,
            maximum=MAX_SERIALIZED_RECEIPT_BYTES,
            name="paired efficiency case",
        )

    @property
    def case_id(self) -> str:
        return self.content_id

    @property
    def required_evidence_count(self) -> int:
        return len(self.required_evidence_references)

    @property
    def baseline_covered_evidence_count(self) -> int:
        return len(self.baseline_covered_evidence_references)

    @property
    def candidate_covered_evidence_count(self) -> int:
        return len(self.candidate_covered_evidence_references)

    @property
    def baseline_coverage_bps(self) -> int:
        if not self.required_evidence_count:
            return BASIS_POINTS
        return (
            self.baseline_covered_evidence_count
            * BASIS_POINTS
            // self.required_evidence_count
        )

    @property
    def candidate_coverage_bps(self) -> int:
        if not self.required_evidence_count:
            return BASIS_POINTS
        return (
            self.candidate_covered_evidence_count
            * BASIS_POINTS
            // self.required_evidence_count
        )

    @property
    def coverage_preserved(self) -> bool:
        return set(self.baseline_covered_evidence_references).issubset(
            self.candidate_covered_evidence_references
        )

    @property
    def candidate_has_full_required_coverage(self) -> bool:
        return self.candidate_covered_evidence_count == self.required_evidence_count

    @property
    def input_token_reduction_bps(self) -> int:
        if not self.baseline_input_tokens:
            return 0
        return (
            (self.baseline_input_tokens - self.candidate_input_tokens)
            * BASIS_POINTS
            // self.baseline_input_tokens
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "task_reference": self.task_reference,
            "goal_reference": self.goal_reference,
            "repository_tree_digest": self.repository_tree_digest,
            "policy_digest": self.policy_digest,
            "baseline_receipt_ids": self.baseline_receipt_ids,
            "candidate_receipt_ids": self.candidate_receipt_ids,
            "baseline_terminal_receipt_id": self.baseline_terminal_receipt_id,
            "candidate_terminal_receipt_id": self.candidate_terminal_receipt_id,
            "baseline_input_tokens": self.baseline_input_tokens,
            "candidate_input_tokens": self.candidate_input_tokens,
            "required_evidence_references": self.required_evidence_references,
            "required_evidence_count": self.required_evidence_count,
            "baseline_covered_evidence_references": (
                self.baseline_covered_evidence_references
            ),
            "baseline_covered_evidence_count": (
                self.baseline_covered_evidence_count
            ),
            "baseline_coverage_bps": self.baseline_coverage_bps,
            "candidate_covered_evidence_references": (
                self.candidate_covered_evidence_references
            ),
            "candidate_covered_evidence_count": (
                self.candidate_covered_evidence_count
            ),
            "candidate_coverage_bps": self.candidate_coverage_bps,
            "coverage_preserved": self.coverage_preserved,
            "candidate_has_full_required_coverage": (
                self.candidate_has_full_required_coverage
            ),
            "input_token_reduction_bps": self.input_token_reduction_bps,
        }

    def to_dict(self, *, include_case_id: bool = False) -> dict[str, Any]:
        payload = super().to_dict()
        if include_case_id:
            payload["case_id"] = self.case_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairedEfficiencyCase":
        _schema(payload, cls.SCHEMA, "paired efficiency case")
        derived = {
            "required_evidence_count",
            "baseline_covered_evidence_count",
            "baseline_coverage_bps",
            "candidate_covered_evidence_count",
            "candidate_coverage_bps",
            "coverage_preserved",
            "candidate_has_full_required_coverage",
            "input_token_reduction_bps",
        }
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "task_reference",
                "goal_reference",
                "repository_tree_digest",
                "policy_digest",
                "baseline_receipt_ids",
                "candidate_receipt_ids",
                "baseline_terminal_receipt_id",
                "candidate_terminal_receipt_id",
                "baseline_input_tokens",
                "candidate_input_tokens",
                "required_evidence_references",
                "baseline_covered_evidence_references",
                "candidate_covered_evidence_references",
                "case_id",
                "content_id",
                *derived,
            },
            artifact_name="paired efficiency case",
        )
        result = cls(
            task_reference=payload.get("task_reference", ""),
            goal_reference=payload.get("goal_reference", ""),
            repository_tree_digest=payload.get("repository_tree_digest", ""),
            policy_digest=payload.get("policy_digest", ""),
            baseline_receipt_ids=tuple(payload.get("baseline_receipt_ids") or ()),
            candidate_receipt_ids=tuple(
                payload.get("candidate_receipt_ids") or ()
            ),
            baseline_terminal_receipt_id=payload.get(
                "baseline_terminal_receipt_id", ""
            ),
            candidate_terminal_receipt_id=payload.get(
                "candidate_terminal_receipt_id", ""
            ),
            baseline_input_tokens=payload.get("baseline_input_tokens", 0),
            candidate_input_tokens=payload.get("candidate_input_tokens", 0),
            required_evidence_references=tuple(
                payload.get("required_evidence_references") or ()
            ),
            baseline_covered_evidence_references=tuple(
                payload.get("baseline_covered_evidence_references") or ()
            ),
            candidate_covered_evidence_references=tuple(
                payload.get("candidate_covered_evidence_references") or ()
            ),
        )
        for name in derived:
            if payload.get(name, getattr(result, name)) != getattr(result, name):
                raise ContractValidationError(
                    f"{name} claim does not match paired efficiency case"
                )
        _claim(payload, result.case_id, "case_id", "content_id")
        return result


@dataclass(frozen=True)
class PairedEfficiencyReport(CanonicalContract):
    """Detached paired token/coverage calculation.

    The report deliberately does not claim objective evidence by itself.  It
    contains compact receipt identifiers and derived totals, so a persisted
    report cannot independently establish that those identifiers name real
    terminal receipts or that its source population was exhaustive.  Use
    :class:`TerminalAcceptedWorkEvidence` for an authority-bearing,
    source-replayable benchmark receipt.
    """

    SCHEMA: ClassVar[str] = PAIRED_EFFICIENCY_REPORT_SCHEMA

    cases: tuple[PairedEfficiencyCase, ...] = ()
    baseline_unpaired_accepted_task_references: tuple[str, ...] = ()
    candidate_unpaired_accepted_task_references: tuple[str, ...] = ()
    minimum_input_token_reduction_bps: int = (
        DEFAULT_MINIMUM_INPUT_TOKEN_REDUCTION_BPS
    )

    def __post_init__(self) -> None:
        cases = _coerce_records(
            self.cases,
            PairedEfficiencyCase,
            PairedEfficiencyCase.from_dict,
            field_name="cases",
            maximum=MAX_RECEIPTS_PER_REPORT,
        )
        cases = tuple(sorted(cases, key=lambda item: item.task_reference))
        if len({item.task_reference for item in cases}) != len(cases):
            raise ContractValidationError(
                "paired efficiency cases must have unique task references"
            )
        object.__setattr__(self, "cases", cases)
        for name in (
            "baseline_unpaired_accepted_task_references",
            "candidate_unpaired_accepted_task_references",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    maximum=MAX_RECEIPTS_PER_REPORT,
                ),
            )
        paired = {item.task_reference for item in cases}
        unpaired = set(self.baseline_unpaired_accepted_task_references) | set(
            self.candidate_unpaired_accepted_task_references
        )
        if paired.intersection(unpaired):
            raise ContractValidationError(
                "paired tasks cannot also be reported as unpaired"
            )
        object.__setattr__(
            self,
            "minimum_input_token_reduction_bps",
            _integer(
                self.minimum_input_token_reduction_bps,
                field_name="minimum_input_token_reduction_bps",
                maximum=BASIS_POINTS,
            ),
        )
        _record_size(
            self,
            maximum=MAX_SERIALIZED_REPORT_BYTES,
            name="paired efficiency report",
        )

    @property
    def report_id(self) -> str:
        return self.content_id

    @property
    def paired_task_count(self) -> int:
        return len(self.cases)

    @property
    def median_baseline_input_tokens(self) -> int:
        return _median_integer(
            tuple(item.baseline_input_tokens for item in self.cases)
        )

    @property
    def median_candidate_input_tokens(self) -> int:
        return _median_integer(
            tuple(item.candidate_input_tokens for item in self.cases)
        )

    @property
    def median_input_token_reduction_bps(self) -> int:
        # Preserve the pairing when aggregating.  A ratio of the two arm
        # medians can combine different tasks and report a gate-passing
        # reduction even when the median same-task reduction fails.
        return _median_integer(
            tuple(item.input_token_reduction_bps for item in self.cases)
        )

    @property
    def coverage_regression_count(self) -> int:
        return sum(not item.coverage_preserved for item in self.cases)

    @property
    def candidate_incomplete_coverage_count(self) -> int:
        return sum(
            not item.candidate_has_full_required_coverage
            for item in self.cases
        )

    @property
    def population_complete(self) -> bool:
        return not (
            self.baseline_unpaired_accepted_task_references
            or self.candidate_unpaired_accepted_task_references
        )

    @property
    def terminal_accepted_work_accounting_proven(self) -> bool:
        return bool(self.cases) and self.population_complete

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        # A detached calculation is diagnostic, never completion evidence.
        # Keeping this property explicit prevents older consumers from
        # interpreting structural arithmetic as a verified source population.
        return ()

    @property
    def token_gate_passed(self) -> bool:
        return (
            bool(self.cases)
            and self.median_input_token_reduction_bps
            >= self.minimum_input_token_reduction_bps
        )

    @property
    def coverage_gate_passed(self) -> bool:
        return (
            bool(self.cases)
            and self.coverage_regression_count == 0
            and self.candidate_incomplete_coverage_count == 0
        )

    @property
    def passed(self) -> bool:
        return (
            self.population_complete
            and self.token_gate_passed
            and self.coverage_gate_passed
        )

    @property
    def promotion_eligible(self) -> bool:
        return self.passed

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "cases": self.cases,
            "baseline_unpaired_accepted_task_references": (
                self.baseline_unpaired_accepted_task_references
            ),
            "candidate_unpaired_accepted_task_references": (
                self.candidate_unpaired_accepted_task_references
            ),
            "minimum_input_token_reduction_bps": (
                self.minimum_input_token_reduction_bps
            ),
            "paired_task_count": self.paired_task_count,
            "median_baseline_input_tokens": (
                self.median_baseline_input_tokens
            ),
            "median_candidate_input_tokens": (
                self.median_candidate_input_tokens
            ),
            "median_input_token_reduction_bps": (
                self.median_input_token_reduction_bps
            ),
            "coverage_regression_count": self.coverage_regression_count,
            "candidate_incomplete_coverage_count": (
                self.candidate_incomplete_coverage_count
            ),
            "population_complete": self.population_complete,
            "terminal_accepted_work_accounting_proven": (
                self.terminal_accepted_work_accounting_proven
            ),
            "evidence_claim_references": self.evidence_claim_references,
            "token_gate_passed": self.token_gate_passed,
            "coverage_gate_passed": self.coverage_gate_passed,
            "passed": self.passed,
        }

    def to_dict(self, *, include_report_id: bool = False) -> dict[str, Any]:
        payload = super().to_dict()
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairedEfficiencyReport":
        _schema(payload, cls.SCHEMA, "paired efficiency report")
        derived = {
            "paired_task_count",
            "median_baseline_input_tokens",
            "median_candidate_input_tokens",
            "median_input_token_reduction_bps",
            "coverage_regression_count",
            "candidate_incomplete_coverage_count",
            "population_complete",
            "terminal_accepted_work_accounting_proven",
            "token_gate_passed",
            "coverage_gate_passed",
            "passed",
        }
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "cases",
                "baseline_unpaired_accepted_task_references",
                "candidate_unpaired_accepted_task_references",
                "minimum_input_token_reduction_bps",
                "evidence_claim_references",
                "report_id",
                "content_id",
                *derived,
            },
            artifact_name="paired efficiency report",
        )
        result = cls(
            cases=tuple(
                PairedEfficiencyCase.from_dict(item)
                for item in payload.get("cases") or ()
            ),
            baseline_unpaired_accepted_task_references=tuple(
                payload.get(
                    "baseline_unpaired_accepted_task_references"
                )
                or ()
            ),
            candidate_unpaired_accepted_task_references=tuple(
                payload.get(
                    "candidate_unpaired_accepted_task_references"
                )
                or ()
            ),
            minimum_input_token_reduction_bps=payload.get(
                "minimum_input_token_reduction_bps",
                DEFAULT_MINIMUM_INPUT_TOKEN_REDUCTION_BPS,
            ),
        )
        for name in derived:
            if payload.get(name, getattr(result, name)) != getattr(result, name):
                raise ContractValidationError(
                    f"{name} claim does not match paired efficiency report"
                )
        if tuple(
            payload.get(
                "evidence_claim_references",
                result.evidence_claim_references,
            )
        ) != result.evidence_claim_references:
            raise ContractValidationError(
                "evidence_claim_references claim does not match paired "
                "efficiency report"
            )
        _claim(payload, result.report_id, "report_id", "content_id")
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "PairedEfficiencyReport":
        return cls.from_dict(
            _load_json(value, artifact_name="paired efficiency report")
        )


@dataclass(frozen=True)
class TerminalAcceptedWorkEvidence(CanonicalContract):
    """Replayable proof of the terminal accepted-work accounting boundary.

    A compact :class:`PairedEfficiencyReport` is useful for dashboards but
    cannot prove what its receipt IDs contained.  This evidence artifact
    carries both complete, typed source populations and independently rebuilds
    the paired report whenever it is constructed or decoded.  Consequently an
    omitted attempt, forged token total, non-accepted terminal ID, unpaired
    accepted task, or stale tree/policy binding fails before the objective
    requirement can be claimed.

    The proof freezes one goal, repository tree, and policy.  Failed-only work
    may be present in the source population for transparency, but it never
    enters the accepted-task denominator.
    """

    SCHEMA: ClassVar[str] = TERMINAL_ACCEPTED_WORK_EVIDENCE_SCHEMA

    paired_report: PairedEfficiencyReport
    baseline_receipts: tuple[EfficiencyReceipt, ...]
    candidate_receipts: tuple[EfficiencyReceipt, ...]
    requirement_id: str = TERMINAL_ACCEPTED_WORK_EVIDENCE_ID
    result: str = "passed"

    def __post_init__(self) -> None:
        paired = self.paired_report
        if isinstance(paired, Mapping):
            paired = PairedEfficiencyReport.from_dict(paired)
        if not isinstance(paired, PairedEfficiencyReport):
            raise ContractValidationError(
                "terminal accepted-work evidence requires a paired report"
            )
        object.__setattr__(self, "paired_report", paired)

        for name in ("baseline_receipts", "candidate_receipts"):
            receipts = _coerce_records(
                getattr(self, name),
                EfficiencyReceipt,
                EfficiencyReceipt.from_dict,
                field_name=name,
                maximum=MAX_RECEIPTS_PER_REPORT,
            )
            receipts = tuple(sorted(receipts, key=lambda item: item.receipt_id))
            if len({item.receipt_id for item in receipts}) != len(receipts):
                raise ContractValidationError(
                    f"{name} contains duplicate receipt identities"
                )
            object.__setattr__(self, name, receipts)

        if self.requirement_id != TERMINAL_ACCEPTED_WORK_EVIDENCE_ID:
            raise ContractValidationError(
                "terminal accepted-work evidence carries an unexpected "
                "requirement ID"
            )
        if self.result != "passed":
            raise ContractValidationError(
                "terminal accepted-work evidence result must be passed"
            )
        if not self.baseline_receipts or not self.candidate_receipts:
            raise ContractValidationError(
                "terminal accepted-work evidence requires both source arms"
            )

        bindings = {
            (
                item.goal_reference,
                item.repository_tree_digest,
                item.policy_digest,
            )
            for item in self.baseline_receipts + self.candidate_receipts
        }
        if len(bindings) != 1:
            raise ContractValidationError(
                "terminal accepted-work source receipts must freeze one "
                "goal, repository tree, and policy"
            )

        required_evidence_by_task = {
            case.task_reference: case.required_evidence_references
            for case in paired.cases
        }
        rebuilt = build_paired_efficiency_report(
            self.baseline_receipts,
            self.candidate_receipts,
            required_evidence_by_task=required_evidence_by_task,
            minimum_input_token_reduction_bps=(
                paired.minimum_input_token_reduction_bps
            ),
        )
        if rebuilt != paired:
            raise ContractValidationError(
                "terminal accepted-work report does not match replayed "
                "source receipt populations"
            )
        if not paired.terminal_accepted_work_accounting_proven:
            raise ContractValidationError(
                "terminal accepted-work evidence requires a non-empty, "
                "population-complete accepted-task comparison"
            )

        _record_size(
            self,
            maximum=MAX_SERIALIZED_REPORT_BYTES,
            name="terminal accepted-work evidence",
        )

    @property
    def evidence_id(self) -> str:
        return self.content_id

    @property
    def report_id(self) -> str:
        return self.paired_report.report_id

    @property
    def baseline_receipt_ids(self) -> tuple[str, ...]:
        return tuple(item.receipt_id for item in self.baseline_receipts)

    @property
    def candidate_receipt_ids(self) -> tuple[str, ...]:
        return tuple(item.receipt_id for item in self.candidate_receipts)

    @property
    def source_receipt_count(self) -> int:
        return len(self.baseline_receipts) + len(self.candidate_receipts)

    @property
    def task_references(self) -> tuple[str, ...]:
        return tuple(item.task_reference for item in self.paired_report.cases)

    @property
    def goal_reference(self) -> str:
        return self.paired_report.cases[0].goal_reference

    @property
    def repository_tree_digest(self) -> str:
        return self.paired_report.cases[0].repository_tree_digest

    @property
    def policy_digest(self) -> str:
        return self.paired_report.cases[0].policy_digest

    @property
    def benchmark_input_digest(self) -> str:
        return _canonical_sha256(
            {
                "baseline_receipt_ids": self.baseline_receipt_ids,
                "candidate_receipt_ids": self.candidate_receipt_ids,
                "paired_report_id": self.report_id,
            }
        )

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (self.requirement_id,)

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        return self.proved_requirement_ids

    @property
    def promotion_eligible(self) -> bool:
        """Whether the separate token-reduction and coverage gates also pass."""

        return self.paired_report.passed

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "requirement_id": self.requirement_id,
            "goal_reference": self.goal_reference,
            "repository_tree_digest": self.repository_tree_digest,
            "policy_digest": self.policy_digest,
            "paired_report": self.paired_report,
            "report_id": self.report_id,
            "baseline_receipts": self.baseline_receipts,
            "candidate_receipts": self.candidate_receipts,
            "baseline_receipt_ids": self.baseline_receipt_ids,
            "candidate_receipt_ids": self.candidate_receipt_ids,
            "source_receipt_count": self.source_receipt_count,
            "task_references": self.task_references,
            "benchmark_input_digest": self.benchmark_input_digest,
            "result": self.result,
            "proved_requirement_ids": self.proved_requirement_ids,
            "evidence_claim_references": self.evidence_claim_references,
            "promotion_eligible": self.promotion_eligible,
        }

    def to_dict(self, *, include_evidence_id: bool = False) -> dict[str, Any]:
        payload = super().to_dict()
        if include_evidence_id:
            payload["evidence_id"] = self.evidence_id
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "TerminalAcceptedWorkEvidence":
        _schema(payload, cls.SCHEMA, "terminal accepted-work evidence")
        derived = {
            "goal_reference",
            "repository_tree_digest",
            "policy_digest",
            "report_id",
            "baseline_receipt_ids",
            "candidate_receipt_ids",
            "source_receipt_count",
            "task_references",
            "benchmark_input_digest",
            "proved_requirement_ids",
            "evidence_claim_references",
            "promotion_eligible",
        }
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "requirement_id",
                "paired_report",
                "baseline_receipts",
                "candidate_receipts",
                "result",
                "evidence_id",
                "content_id",
                *derived,
            },
            artifact_name="terminal accepted-work evidence",
        )
        paired_payload = payload.get("paired_report")
        if not isinstance(paired_payload, Mapping):
            raise ContractValidationError(
                "terminal accepted-work evidence requires a paired report"
            )
        result = cls(
            paired_report=PairedEfficiencyReport.from_dict(paired_payload),
            baseline_receipts=tuple(
                EfficiencyReceipt.from_dict(item)
                for item in payload.get("baseline_receipts") or ()
            ),
            candidate_receipts=tuple(
                EfficiencyReceipt.from_dict(item)
                for item in payload.get("candidate_receipts") or ()
            ),
            requirement_id=payload.get("requirement_id", ""),
            result=payload.get("result", ""),
        )
        for name in derived:
            expected = getattr(result, name)
            claimed = payload.get(name, expected)
            if isinstance(expected, tuple):
                claimed = tuple(claimed)
            if claimed != expected:
                raise ContractValidationError(
                    f"{name} claim does not match terminal accepted-work evidence"
                )
        _claim(payload, result.evidence_id, "evidence_id", "content_id")
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "TerminalAcceptedWorkEvidence":
        return cls.from_dict(
            _load_json(value, artifact_name="terminal accepted-work evidence")
        )


@dataclass(frozen=True)
class RequiredContextProofBinding(CanonicalContract):
    """Capsule-verified projection of one required-context compiler result."""

    SCHEMA: ClassVar[str] = REQUIRED_CONTEXT_PROOF_BINDING_SCHEMA

    task_reference: str
    context_capsule: Any
    context_compilation_receipt: Any
    capsule_id: str
    receipt_id: str
    evidence_id: str
    repository_id: str
    tree_id: str
    objective_id: str
    policy_id: str
    policy_revision: str
    effective_input_limit: int
    input_tokens: int
    required_reference_ids: tuple[str, ...]
    selected_reference_ids: tuple[str, ...]
    required_coverage_ids: tuple[str, ...]
    selected_coverage_ids: tuple[str, ...]
    required_fields: tuple[str, ...]
    artifact_digest: str
    requirement_id: str = REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID

    def __post_init__(self) -> None:
        from .context_compiler import (
            REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID as COMPILER_REQUIREMENT_ID,
            ContextCompilationReceipt,
            ContextCompileResult,
        )
        from .context_contracts import ContextCapsule

        capsule = self.context_capsule
        if isinstance(capsule, Mapping):
            capsule = ContextCapsule.from_dict(capsule)
        receipt = self.context_compilation_receipt
        if isinstance(receipt, Mapping):
            receipt = ContextCompilationReceipt.from_dict(receipt)
        if not isinstance(capsule, ContextCapsule) or not isinstance(
            receipt, ContextCompilationReceipt
        ):
            raise ContractValidationError(
                "required context proof requires a typed capsule and receipt"
            )
        # This performs the strong capsule/receipt/witness/decision/digest
        # cross-check.  A receipt alone is intentionally not sufficient.
        ContextCompileResult(capsule, receipt, receipt.decisions)
        evidence = receipt.evidence
        if (
            evidence is None
            or COMPILER_REQUIREMENT_ID != REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID
            or receipt.evidence_claim_references
            != (REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID,)
        ):
            raise ContractValidationError(
                "context compilation receipt lacks qualifying typed evidence"
            )
        object.__setattr__(self, "context_capsule", capsule)
        object.__setattr__(self, "context_compilation_receipt", receipt)
        for name in (
            "task_reference",
            "capsule_id",
            "receipt_id",
            "evidence_id",
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "policy_revision",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    field_name=name,
                    required=True,
                    max_bytes=MAX_REFERENCE_BYTES,
                ),
            )
        if self.requirement_id != REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID:
            raise ContractValidationError(
                "required context proof carries an unexpected requirement ID"
            )
        for name in ("effective_input_limit", "input_tokens"):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    minimum=1,
                    maximum=MAX_TOKENS,
                ),
            )
        if self.input_tokens > self.effective_input_limit:
            raise ContractValidationError(
                "required context proof exceeds the effective input budget"
            )
        for name in (
            "required_reference_ids",
            "selected_reference_ids",
            "required_coverage_ids",
            "selected_coverage_ids",
            "required_fields",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    maximum=MAX_EVIDENCE_REFERENCES,
                ),
            )
        if not self.required_references_preserved:
            raise ContractValidationError(
                "required context proof loses required references"
            )
        if not self.required_coverage_preserved:
            raise ContractValidationError(
                "required context proof loses required coverage"
            )
        if self.required_fields != (
            "acceptance",
            "authority",
            "goal",
            "scope",
        ):
            raise ContractValidationError(
                "required context proof must preserve every invariant field"
            )
        required_coverage = tuple(
            sorted(
                {
                    coverage_id
                    for reference in capsule.evidence
                    if reference.required
                    for coverage_id in reference.coverage_ids
                }
            )
        )
        source_claims = {
            "capsule_id": capsule.capsule_id,
            "receipt_id": receipt.receipt_id,
            "evidence_id": evidence.content_id,
            "repository_id": receipt.repository_id,
            "tree_id": receipt.tree_id,
            "objective_id": receipt.objective_id,
            "policy_id": receipt.policy_id,
            "policy_revision": receipt.policy_revision,
            "effective_input_limit": receipt.effective_input_limit,
            "input_tokens": receipt.input_tokens,
            "required_reference_ids": evidence.required_reference_ids,
            "selected_reference_ids": evidence.selected_reference_ids,
            "required_coverage_ids": required_coverage,
            "selected_coverage_ids": capsule.evidence_coverage_ids,
            "required_fields": evidence.required_fields,
            "artifact_digest": evidence.artifact_digest,
            "requirement_id": evidence.requirement_id,
        }
        if any(
            getattr(self, name) != value
            for name, value in source_claims.items()
        ):
            raise ContractValidationError(
                "required context proof projection is not bound to its "
                "typed compiler result"
            )
        object.__setattr__(
            self,
            "artifact_digest",
            _digest(self.artifact_digest, field_name="artifact_digest"),
        )
        _record_size(
            self,
            maximum=MAX_SERIALIZED_RECEIPT_BYTES * 2,
            name="required context proof binding",
        )

    @property
    def binding_id(self) -> str:
        return self.content_id

    @property
    def required_references_preserved(self) -> bool:
        return set(self.required_reference_ids).issubset(
            self.selected_reference_ids
        )

    @property
    def required_coverage_preserved(self) -> bool:
        return set(self.required_coverage_ids).issubset(
            self.selected_coverage_ids
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "task_reference": self.task_reference,
            "context_capsule": self.context_capsule,
            "context_compilation_receipt": self.context_compilation_receipt,
            "capsule_id": self.capsule_id,
            "receipt_id": self.receipt_id,
            "evidence_id": self.evidence_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "effective_input_limit": self.effective_input_limit,
            "input_tokens": self.input_tokens,
            "required_reference_ids": self.required_reference_ids,
            "selected_reference_ids": self.selected_reference_ids,
            "required_coverage_ids": self.required_coverage_ids,
            "selected_coverage_ids": self.selected_coverage_ids,
            "required_fields": self.required_fields,
            "artifact_digest": self.artifact_digest,
            "requirement_id": self.requirement_id,
            "required_references_preserved": (
                self.required_references_preserved
            ),
            "required_coverage_preserved": self.required_coverage_preserved,
        }

    @classmethod
    def from_context_compile_result(
        cls,
        task_reference: str,
        result: Any,
    ) -> "RequiredContextProofBinding":
        from .context_compiler import ContextCompileResult

        if not isinstance(result, ContextCompileResult):
            raise ContractValidationError(
                "required context proofs must be ContextCompileResult values"
            )
        # Re-run result validation even if a caller constructed the dataclass
        # before crossing this promotion boundary.
        ContextCompileResult(result.capsule, result.receipt, result.decisions)
        evidence = result.receipt.evidence
        if evidence is None:
            raise ContractValidationError(
                "context compilation result lacks qualifying evidence"
            )
        required_coverage = tuple(
            sorted(
                {
                    coverage_id
                    for reference in result.capsule.evidence
                    if reference.required
                    for coverage_id in reference.coverage_ids
                }
            )
        )
        return cls(
            task_reference=task_reference,
            context_capsule=result.capsule,
            context_compilation_receipt=result.receipt,
            capsule_id=result.capsule.capsule_id,
            receipt_id=result.receipt.receipt_id,
            evidence_id=evidence.content_id,
            repository_id=result.receipt.repository_id,
            tree_id=result.receipt.tree_id,
            objective_id=result.receipt.objective_id,
            policy_id=result.receipt.policy_id,
            policy_revision=result.receipt.policy_revision,
            effective_input_limit=result.receipt.effective_input_limit,
            input_tokens=result.receipt.input_tokens,
            required_reference_ids=evidence.required_reference_ids,
            selected_reference_ids=evidence.selected_reference_ids,
            required_coverage_ids=required_coverage,
            selected_coverage_ids=result.capsule.evidence_coverage_ids,
            required_fields=evidence.required_fields,
            artifact_digest=evidence.artifact_digest,
            requirement_id=evidence.requirement_id,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequiredContextProofBinding":
        _schema(payload, cls.SCHEMA, "required context proof binding")
        derived = {
            "required_references_preserved",
            "required_coverage_preserved",
        }
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "task_reference",
                "context_capsule",
                "context_compilation_receipt",
                "capsule_id",
                "receipt_id",
                "evidence_id",
                "repository_id",
                "tree_id",
                "objective_id",
                "policy_id",
                "policy_revision",
                "effective_input_limit",
                "input_tokens",
                "required_reference_ids",
                "selected_reference_ids",
                "required_coverage_ids",
                "selected_coverage_ids",
                "required_fields",
                "artifact_digest",
                "requirement_id",
                "binding_id",
                "content_id",
                *derived,
            },
            artifact_name="required context proof binding",
        )
        result = cls(
            task_reference=payload.get("task_reference", ""),
            context_capsule=payload.get("context_capsule"),
            context_compilation_receipt=payload.get(
                "context_compilation_receipt"
            ),
            capsule_id=payload.get("capsule_id", ""),
            receipt_id=payload.get("receipt_id", ""),
            evidence_id=payload.get("evidence_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            effective_input_limit=payload.get("effective_input_limit", 0),
            input_tokens=payload.get("input_tokens", 0),
            required_reference_ids=tuple(
                payload.get("required_reference_ids") or ()
            ),
            selected_reference_ids=tuple(
                payload.get("selected_reference_ids") or ()
            ),
            required_coverage_ids=tuple(
                payload.get("required_coverage_ids") or ()
            ),
            selected_coverage_ids=tuple(
                payload.get("selected_coverage_ids") or ()
            ),
            required_fields=tuple(payload.get("required_fields") or ()),
            artifact_digest=payload.get("artifact_digest", ""),
            requirement_id=payload.get("requirement_id", ""),
        )
        for name in derived:
            if payload.get(name, getattr(result, name)) != getattr(result, name):
                raise ContractValidationError(
                    f"{name} claim does not match required context proof"
                )
        _claim(payload, result.binding_id, "binding_id", "content_id")
        return result


@dataclass(frozen=True)
class RequiredContextPromotionReport(CanonicalContract):
    """Join paired accepted-work measurement to verified base contexts."""

    SCHEMA: ClassVar[str] = REQUIRED_CONTEXT_PROMOTION_REPORT_SCHEMA

    paired_report: PairedEfficiencyReport
    proof_bindings: tuple[RequiredContextProofBinding, ...] = ()
    terminal_work_evidence: TerminalAcceptedWorkEvidence | None = None

    def __post_init__(self) -> None:
        paired = self.paired_report
        if isinstance(paired, Mapping):
            paired = PairedEfficiencyReport.from_dict(paired)
        if not isinstance(paired, PairedEfficiencyReport):
            raise ContractValidationError(
                "paired_report must be a PairedEfficiencyReport"
            )
        object.__setattr__(self, "paired_report", paired)
        terminal_evidence = self.terminal_work_evidence
        if isinstance(terminal_evidence, Mapping):
            terminal_evidence = TerminalAcceptedWorkEvidence.from_dict(
                terminal_evidence
            )
        if terminal_evidence is not None:
            if not isinstance(
                terminal_evidence, TerminalAcceptedWorkEvidence
            ):
                raise ContractValidationError(
                    "terminal_work_evidence must be typed accepted-work evidence"
                )
            if terminal_evidence.paired_report != paired:
                raise ContractValidationError(
                    "terminal accepted-work evidence does not bind the paired report"
                )
        object.__setattr__(
            self, "terminal_work_evidence", terminal_evidence
        )
        bindings = _coerce_records(
            self.proof_bindings,
            RequiredContextProofBinding,
            RequiredContextProofBinding.from_dict,
            field_name="proof_bindings",
            maximum=MAX_RECEIPTS_PER_REPORT,
        )
        bindings = tuple(
            sorted(
                bindings,
                key=lambda item: (item.task_reference, item.receipt_id),
            )
        )
        if len({item.receipt_id for item in bindings}) != len(bindings):
            raise ContractValidationError(
                "required context report contains duplicate receipt IDs"
            )
        object.__setattr__(self, "proof_bindings", bindings)
        _record_size(
            self,
            maximum=MAX_SERIALIZED_REPORT_BYTES,
            name="required context promotion report",
        )

    @property
    def report_id(self) -> str:
        return self.content_id

    @property
    def proof_task_references(self) -> tuple[str, ...]:
        return tuple(
            sorted({item.task_reference for item in self.proof_bindings})
        )

    @property
    def missing_proof_task_references(self) -> tuple[str, ...]:
        paired = {item.task_reference for item in self.paired_report.cases}
        return tuple(sorted(paired.difference(self.proof_task_references)))

    @property
    def unexpected_proof_task_references(self) -> tuple[str, ...]:
        paired = {item.task_reference for item in self.paired_report.cases}
        return tuple(
            sorted(set(self.proof_task_references).difference(paired))
        )

    @property
    def proof_population_complete(self) -> bool:
        return bool(self.paired_report.cases) and not (
            self.missing_proof_task_references
            or self.unexpected_proof_task_references
        )

    @property
    def context_receipt_count(self) -> int:
        return len(self.proof_bindings)

    def _coverage_by_task(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for binding in self.proof_bindings:
            result.setdefault(binding.task_reference, set()).update(
                binding.required_coverage_ids
            )
        return result

    @property
    def coverage_requirements_consistent(self) -> bool:
        case_by_task = {
            item.task_reference: item for item in self.paired_report.cases
        }
        coverage = self._coverage_by_task()
        return all(
            task in coverage
            and coverage[task] == set(case.required_evidence_references)
            for task, case in case_by_task.items()
        ) and not set(coverage).difference(case_by_task)

    @property
    def token_accounting_consistent(self) -> bool:
        """Require exact reconciliation with charged candidate input."""

        case_by_task = {
            item.task_reference: item for item in self.paired_report.cases
        }
        totals: dict[str, int] = {}
        for binding in self.proof_bindings:
            totals[binding.task_reference] = (
                totals.get(binding.task_reference, 0) + binding.input_tokens
            )
        return all(
            task in totals
            and totals[task] == case.candidate_input_tokens
            for task, case in case_by_task.items()
        ) and not set(totals).difference(case_by_task)

    @property
    def typed_context_gate_passed(self) -> bool:
        return (
            self.proof_population_complete
            and bool(self.proof_bindings)
            and all(
                item.required_references_preserved
                and item.required_coverage_preserved
                and item.input_tokens <= item.effective_input_limit
                for item in self.proof_bindings
            )
            and self.coverage_requirements_consistent
            and self.token_accounting_consistent
        )

    @property
    def paired_efficiency_gate_passed(self) -> bool:
        return bool(
            self.terminal_work_evidence is not None
            and self.terminal_work_evidence.promotion_eligible
        )

    @property
    def passed(self) -> bool:
        return (
            self.paired_efficiency_gate_passed
            and self.typed_context_gate_passed
        )

    @property
    def promotion_eligible(self) -> bool:
        return self.passed

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        if not self.passed:
            return ()
        return (REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID,)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "paired_report": self.paired_report,
            "terminal_work_evidence": self.terminal_work_evidence,
            "proof_bindings": self.proof_bindings,
            "proof_task_references": self.proof_task_references,
            "missing_proof_task_references": (
                self.missing_proof_task_references
            ),
            "unexpected_proof_task_references": (
                self.unexpected_proof_task_references
            ),
            "proof_population_complete": self.proof_population_complete,
            "context_receipt_count": self.context_receipt_count,
            "coverage_requirements_consistent": (
                self.coverage_requirements_consistent
            ),
            "token_accounting_consistent": self.token_accounting_consistent,
            "typed_context_gate_passed": self.typed_context_gate_passed,
            "paired_efficiency_gate_passed": (
                self.paired_efficiency_gate_passed
            ),
            "evidence_claim_references": self.evidence_claim_references,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
        }

    def to_dict(self, *, include_report_id: bool = False) -> dict[str, Any]:
        payload = super().to_dict()
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "RequiredContextPromotionReport":
        _schema(payload, cls.SCHEMA, "required context promotion report")
        derived = {
            "proof_task_references",
            "missing_proof_task_references",
            "unexpected_proof_task_references",
            "proof_population_complete",
            "context_receipt_count",
            "coverage_requirements_consistent",
            "token_accounting_consistent",
            "typed_context_gate_passed",
            "paired_efficiency_gate_passed",
            "evidence_claim_references",
            "passed",
            "promotion_eligible",
        }
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "paired_report",
                "terminal_work_evidence",
                "proof_bindings",
                "report_id",
                "content_id",
                *derived,
            },
            artifact_name="required context promotion report",
        )
        paired_payload = payload.get("paired_report")
        if not isinstance(paired_payload, Mapping):
            raise ContractValidationError(
                "required context promotion report requires a paired report"
            )
        terminal_payload = payload.get("terminal_work_evidence")
        if terminal_payload is not None and not isinstance(
            terminal_payload, Mapping
        ):
            raise ContractValidationError(
                "terminal_work_evidence must be an object"
            )
        result = cls(
            paired_report=PairedEfficiencyReport.from_dict(paired_payload),
            terminal_work_evidence=(
                TerminalAcceptedWorkEvidence.from_dict(
                    terminal_payload
                )
                if terminal_payload is not None
                else None
            ),
            proof_bindings=tuple(
                RequiredContextProofBinding.from_dict(item)
                for item in payload.get("proof_bindings") or ()
            ),
        )
        for name in derived:
            claimed = payload.get(name, getattr(result, name))
            if isinstance(getattr(result, name), tuple):
                claimed = tuple(claimed)
            if claimed != getattr(result, name):
                raise ContractValidationError(
                    f"{name} claim does not match required context report"
                )
        _claim(payload, result.report_id, "report_id", "content_id")
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "RequiredContextPromotionReport":
        return cls.from_dict(
            _load_json(value, artifact_name="required context promotion report")
        )


@dataclass(frozen=True)
class DeltaRetryProofBinding(CanonicalContract):
    """Bound projection of one capsule-verified context-delta result.

    A receipt by itself is not proof that its claimed artifact digest describes
    a real parent-bound delta and reconstructed context.  The binding therefore
    carries all three bounded capsules as well as the receipt and reruns
    ``ContextDeltaResult`` validation whenever it is constructed or decoded
    from the wire.
    """

    SCHEMA: ClassVar[str] = DELTA_RETRY_PROOF_BINDING_SCHEMA

    task_reference: str
    parent_context_capsule: Any
    context_delta_capsule: Any
    reconstructed_context_capsule: Any
    context_delta_receipt: Any
    receipt_id: str
    evidence_id: str
    repository_id: str
    tree_id: str
    objective_id: str
    policy_id: str
    policy_revision: str
    parent_capsule_id: str
    delta_capsule_id: str
    reconstructed_capsule_id: str
    full_replay_tokens: int
    delta_tokens: int
    required_coverage_ids: tuple[str, ...]
    reconstructed_coverage_ids: tuple[str, ...]
    changed_reference_ids: tuple[str, ...]
    requested_reference_ids: tuple[str, ...]
    retained_reference_ids: tuple[str, ...]
    required_fields: tuple[str, ...]
    artifact_digest: str
    requirement_id: str = DELTA_RETRY_CONTEXT_EVIDENCE_ID
    verifier: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        from .context_compiler import (
            DELTA_RETRY_EVIDENCE_ID,
            ContextDeltaReceipt,
            ContextDeltaResult,
        )
        from .context_contracts import ContextCapsule, ContextDeltaCapsule

        parent_capsule = self.parent_context_capsule
        if isinstance(parent_capsule, Mapping):
            parent_capsule = ContextCapsule.from_dict(parent_capsule)
        delta_capsule = self.context_delta_capsule
        if isinstance(delta_capsule, Mapping):
            delta_capsule = ContextDeltaCapsule.from_dict(delta_capsule)
        reconstructed_capsule = self.reconstructed_context_capsule
        if isinstance(reconstructed_capsule, Mapping):
            reconstructed_capsule = ContextCapsule.from_dict(
                reconstructed_capsule
            )
        receipt = self.context_delta_receipt
        if isinstance(receipt, Mapping):
            receipt = ContextDeltaReceipt.from_dict(receipt)
        if (
            not isinstance(parent_capsule, ContextCapsule)
            or not isinstance(delta_capsule, ContextDeltaCapsule)
            or not isinstance(reconstructed_capsule, ContextCapsule)
            or not isinstance(receipt, ContextDeltaReceipt)
        ):
            raise ContractValidationError(
                "delta retry proof requires typed parent, delta, "
                "reconstructed, and receipt artifacts"
            )
        # Re-run the producer's complete receipt/capsule/witness/digest checks.
        # This is deliberately stronger than accepting an independently
        # constructible ContextDeltaReceipt.
        structural_result = ContextDeltaResult(
            parent_capsule=parent_capsule,
            delta_capsule=delta_capsule,
            reconstructed_capsule=reconstructed_capsule,
            receipt=receipt,
            decisions=receipt.decisions,
            verifier=None,
        )
        if self.verifier is not None:
            from .context_compiler import ContextCompiler

            if not isinstance(self.verifier, ContextCompiler):
                raise ContractValidationError(
                    "delta retry proof verifier must be a ContextCompiler"
                )
            self.verifier.verify_delta_result(structural_result)
        evidence = receipt.evidence
        if (
            evidence is None
            or DELTA_RETRY_EVIDENCE_ID
            != DELTA_RETRY_CONTEXT_EVIDENCE_ID
            or receipt.evidence_claim_references
            != (DELTA_RETRY_CONTEXT_EVIDENCE_ID,)
        ):
            raise ContractValidationError(
                "context delta result does not carry qualifying typed evidence"
            )
        object.__setattr__(self, "parent_context_capsule", parent_capsule)
        object.__setattr__(self, "context_delta_capsule", delta_capsule)
        object.__setattr__(
            self, "reconstructed_context_capsule", reconstructed_capsule
        )
        object.__setattr__(self, "context_delta_receipt", receipt)
        for name in (
            "task_reference",
            "receipt_id",
            "evidence_id",
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "policy_revision",
            "parent_capsule_id",
            "delta_capsule_id",
            "reconstructed_capsule_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    field_name=name,
                    required=True,
                    max_bytes=MAX_REFERENCE_BYTES,
                ),
            )
        if self.requirement_id != DELTA_RETRY_CONTEXT_EVIDENCE_ID:
            raise ContractValidationError(
                "delta retry proof carries an unexpected requirement ID"
            )
        for name in ("full_replay_tokens", "delta_tokens"):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    minimum=1,
                    maximum=MAX_TOKENS,
                ),
            )
        if self.delta_tokens >= self.full_replay_tokens:
            raise ContractValidationError(
                "delta retry proof must transmit fewer tokens than full replay"
            )
        for name in (
            "required_coverage_ids",
            "reconstructed_coverage_ids",
            "changed_reference_ids",
            "requested_reference_ids",
            "retained_reference_ids",
            "required_fields",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    maximum=MAX_EVIDENCE_REFERENCES,
                ),
            )
        if not self.required_coverage_ids:
            raise ContractValidationError(
                "delta retry proof requires authoritative coverage"
            )
        if not self.coverage_preserved:
            raise ContractValidationError(
                "delta retry proof loses required coverage"
            )
        if not (
            self.changed_reference_ids or self.requested_reference_ids
        ):
            raise ContractValidationError(
                "delta retry proof must identify changed or requested evidence"
            )
        if set(self.changed_reference_ids).intersection(
            self.requested_reference_ids
        ):
            raise ContractValidationError(
                "changed and requested-only proof references must be disjoint"
            )
        if set(self.changed_reference_ids).union(
            self.requested_reference_ids
        ).intersection(self.retained_reference_ids):
            raise ContractValidationError(
                "transmitted and retained proof references must be disjoint"
            )
        if self.required_fields != (
            "acceptance",
            "authority",
            "goal",
            "scope",
        ):
            raise ContractValidationError(
                "delta retry proof must preserve every invariant context field"
            )
        source_claims = {
            "receipt_id": receipt.receipt_id,
            "evidence_id": evidence.content_id,
            "repository_id": receipt.repository_id,
            "tree_id": receipt.tree_id,
            "objective_id": receipt.objective_id,
            "policy_id": receipt.policy_id,
            "policy_revision": receipt.policy_revision,
            "parent_capsule_id": parent_capsule.capsule_id,
            "delta_capsule_id": delta_capsule.capsule_id,
            "reconstructed_capsule_id": reconstructed_capsule.capsule_id,
            "full_replay_tokens": receipt.full_replay_tokens,
            "delta_tokens": receipt.delta_tokens,
            "required_coverage_ids": evidence.required_coverage_ids,
            "reconstructed_coverage_ids": (
                evidence.reconstructed_coverage_ids
            ),
            "changed_reference_ids": evidence.changed_reference_ids,
            "requested_reference_ids": evidence.requested_reference_ids,
            "retained_reference_ids": evidence.retained_reference_ids,
            "required_fields": evidence.required_fields,
            "artifact_digest": evidence.artifact_digest,
            "requirement_id": evidence.requirement_id,
        }
        if any(
            getattr(self, name) != value
            for name, value in source_claims.items()
        ):
            raise ContractValidationError(
                "delta retry proof projection is not bound to its typed receipt"
            )
        object.__setattr__(
            self,
            "artifact_digest",
            _digest(self.artifact_digest, field_name="artifact_digest"),
        )
        _record_size(
            self,
            maximum=MAX_SERIALIZED_REPORT_BYTES,
            name="delta retry proof binding",
        )

    @property
    def binding_id(self) -> str:
        return self.content_id

    @property
    def coverage_preserved(self) -> bool:
        return set(self.required_coverage_ids).issubset(
            self.reconstructed_coverage_ids
        )

    @property
    def input_token_reduction_bps(self) -> int:
        return (
            (self.full_replay_tokens - self.delta_tokens)
            * BASIS_POINTS
            // self.full_replay_tokens
        )

    @property
    def provider_tokens_verified(self) -> bool:
        from .context_compiler import ContextCompiler

        return isinstance(self.verifier, ContextCompiler)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "task_reference": self.task_reference,
            "parent_context_capsule": self.parent_context_capsule,
            "context_delta_capsule": self.context_delta_capsule,
            "reconstructed_context_capsule": (
                self.reconstructed_context_capsule
            ),
            "context_delta_receipt": self.context_delta_receipt,
            "receipt_id": self.receipt_id,
            "evidence_id": self.evidence_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "parent_capsule_id": self.parent_capsule_id,
            "delta_capsule_id": self.delta_capsule_id,
            "reconstructed_capsule_id": self.reconstructed_capsule_id,
            "full_replay_tokens": self.full_replay_tokens,
            "delta_tokens": self.delta_tokens,
            "required_coverage_ids": self.required_coverage_ids,
            "reconstructed_coverage_ids": self.reconstructed_coverage_ids,
            "changed_reference_ids": self.changed_reference_ids,
            "requested_reference_ids": self.requested_reference_ids,
            "retained_reference_ids": self.retained_reference_ids,
            "required_fields": self.required_fields,
            "artifact_digest": self.artifact_digest,
            "requirement_id": self.requirement_id,
            "coverage_preserved": self.coverage_preserved,
            "input_token_reduction_bps": self.input_token_reduction_bps,
            "provider_tokens_verified": self.provider_tokens_verified,
        }

    @classmethod
    def from_context_delta_result(
        cls,
        task_reference: str,
        result: Any,
    ) -> "DeltaRetryProofBinding":
        """Verify and project a complete ``ContextDeltaResult`` instance."""

        # Local import keeps the foundational efficiency receipt contract
        # usable without importing the context compiler during module import.
        from .context_compiler import (
            DELTA_RETRY_EVIDENCE_ID,
            ContextCompiler,
            ContextDeltaResult,
        )

        if DELTA_RETRY_EVIDENCE_ID != DELTA_RETRY_CONTEXT_EVIDENCE_ID:
            raise ContractValidationError(
                "context compiler and efficiency verifier disagree on the "
                "delta retry requirement ID"
            )
        if not isinstance(result, ContextDeltaResult):
            raise ContractValidationError(
                "delta retry proofs must be typed ContextDeltaResult values"
            )
        if not isinstance(result.verifier, ContextCompiler):
            raise ContractValidationError(
                "delta retry proof requires its provider-token verifier"
            )
        # Revalidate both the structural artifacts and canonical provider-token
        # measurements at the promotion trust boundary.
        ContextDeltaResult(
            parent_capsule=result.parent_capsule,
            delta_capsule=result.delta_capsule,
            reconstructed_capsule=result.reconstructed_capsule,
            receipt=result.receipt,
            decisions=result.decisions,
            verifier=result.verifier,
        )
        receipt = result.receipt
        evidence = receipt.evidence
        if evidence is None or receipt.evidence_claim_references != (
            DELTA_RETRY_CONTEXT_EVIDENCE_ID,
        ):
            raise ContractValidationError(
                "context delta receipt does not carry qualifying typed evidence"
            )
        return cls(
            task_reference=task_reference,
            parent_context_capsule=result.parent_capsule,
            context_delta_capsule=result.delta_capsule,
            reconstructed_context_capsule=result.reconstructed_capsule,
            context_delta_receipt=receipt,
            receipt_id=receipt.receipt_id,
            evidence_id=evidence.content_id,
            repository_id=receipt.repository_id,
            tree_id=receipt.tree_id,
            objective_id=receipt.objective_id,
            policy_id=receipt.policy_id,
            policy_revision=receipt.policy_revision,
            parent_capsule_id=receipt.parent_capsule_id,
            delta_capsule_id=receipt.delta_capsule_id,
            reconstructed_capsule_id=receipt.reconstructed_capsule_id,
            full_replay_tokens=receipt.full_replay_tokens,
            delta_tokens=receipt.delta_tokens,
            required_coverage_ids=evidence.required_coverage_ids,
            reconstructed_coverage_ids=evidence.reconstructed_coverage_ids,
            changed_reference_ids=evidence.changed_reference_ids,
            requested_reference_ids=evidence.requested_reference_ids,
            retained_reference_ids=evidence.retained_reference_ids,
            required_fields=evidence.required_fields,
            artifact_digest=evidence.artifact_digest,
            requirement_id=evidence.requirement_id,
            verifier=result.verifier,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        verifier: Any = None,
    ) -> "DeltaRetryProofBinding":
        _schema(payload, cls.SCHEMA, "delta retry proof binding")
        derived = {
            "coverage_preserved",
            "input_token_reduction_bps",
            "provider_tokens_verified",
        }
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "task_reference",
                "parent_context_capsule",
                "context_delta_capsule",
                "reconstructed_context_capsule",
                "context_delta_receipt",
                "receipt_id",
                "evidence_id",
                "repository_id",
                "tree_id",
                "objective_id",
                "policy_id",
                "policy_revision",
                "parent_capsule_id",
                "delta_capsule_id",
                "reconstructed_capsule_id",
                "full_replay_tokens",
                "delta_tokens",
                "required_coverage_ids",
                "reconstructed_coverage_ids",
                "changed_reference_ids",
                "requested_reference_ids",
                "retained_reference_ids",
                "required_fields",
                "artifact_digest",
                "requirement_id",
                "binding_id",
                "content_id",
                *derived,
            },
            artifact_name="delta retry proof binding",
        )
        result = cls(
            task_reference=payload.get("task_reference", ""),
            parent_context_capsule=payload.get("parent_context_capsule"),
            context_delta_capsule=payload.get("context_delta_capsule"),
            reconstructed_context_capsule=payload.get(
                "reconstructed_context_capsule"
            ),
            context_delta_receipt=payload.get("context_delta_receipt"),
            receipt_id=payload.get("receipt_id", ""),
            evidence_id=payload.get("evidence_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            parent_capsule_id=payload.get("parent_capsule_id", ""),
            delta_capsule_id=payload.get("delta_capsule_id", ""),
            reconstructed_capsule_id=payload.get(
                "reconstructed_capsule_id", ""
            ),
            full_replay_tokens=payload.get("full_replay_tokens", 0),
            delta_tokens=payload.get("delta_tokens", 0),
            required_coverage_ids=tuple(
                payload.get("required_coverage_ids") or ()
            ),
            reconstructed_coverage_ids=tuple(
                payload.get("reconstructed_coverage_ids") or ()
            ),
            changed_reference_ids=tuple(
                payload.get("changed_reference_ids") or ()
            ),
            requested_reference_ids=tuple(
                payload.get("requested_reference_ids") or ()
            ),
            retained_reference_ids=tuple(
                payload.get("retained_reference_ids") or ()
            ),
            required_fields=tuple(payload.get("required_fields") or ()),
            artifact_digest=payload.get("artifact_digest", ""),
            requirement_id=payload.get("requirement_id", ""),
            verifier=verifier,
        )
        for name in derived:
            if payload.get(name, getattr(result, name)) != getattr(result, name):
                raise ContractValidationError(
                    f"{name} claim does not match delta retry proof binding"
                )
        _claim(payload, result.binding_id, "binding_id", "content_id")
        return result


@dataclass(frozen=True)
class DeltaRetryPromotionReport(CanonicalContract):
    """Fail-closed join of paired efficiency and typed retry-delta proofs."""

    SCHEMA: ClassVar[str] = DELTA_RETRY_PROMOTION_REPORT_SCHEMA

    paired_report: PairedEfficiencyReport
    proof_bindings: tuple[DeltaRetryProofBinding, ...] = ()
    terminal_work_evidence: TerminalAcceptedWorkEvidence | None = None

    def __post_init__(self) -> None:
        paired = self.paired_report
        if isinstance(paired, Mapping):
            paired = PairedEfficiencyReport.from_dict(paired)
        if not isinstance(paired, PairedEfficiencyReport):
            raise ContractValidationError(
                "paired_report must be a PairedEfficiencyReport"
            )
        object.__setattr__(self, "paired_report", paired)
        terminal_evidence = self.terminal_work_evidence
        if isinstance(terminal_evidence, Mapping):
            terminal_evidence = TerminalAcceptedWorkEvidence.from_dict(
                terminal_evidence
            )
        if terminal_evidence is not None:
            if not isinstance(
                terminal_evidence, TerminalAcceptedWorkEvidence
            ):
                raise ContractValidationError(
                    "terminal_work_evidence must be typed accepted-work evidence"
                )
            if terminal_evidence.paired_report != paired:
                raise ContractValidationError(
                    "terminal accepted-work evidence does not bind the paired report"
                )
        object.__setattr__(
            self, "terminal_work_evidence", terminal_evidence
        )
        bindings = _coerce_records(
            self.proof_bindings,
            DeltaRetryProofBinding,
            DeltaRetryProofBinding.from_dict,
            field_name="proof_bindings",
            maximum=MAX_RECEIPTS_PER_REPORT,
        )
        bindings = tuple(
            sorted(
                bindings,
                key=lambda item: (item.task_reference, item.receipt_id),
            )
        )
        if len({item.receipt_id for item in bindings}) != len(bindings):
            raise ContractValidationError(
                "delta retry promotion report contains duplicate receipt IDs"
            )
        object.__setattr__(self, "proof_bindings", bindings)
        _record_size(
            self,
            maximum=MAX_SERIALIZED_REPORT_BYTES,
            name="delta retry promotion report",
        )

    @property
    def report_id(self) -> str:
        return self.content_id

    @property
    def proof_task_references(self) -> tuple[str, ...]:
        return tuple(
            sorted({item.task_reference for item in self.proof_bindings})
        )

    @property
    def missing_proof_task_references(self) -> tuple[str, ...]:
        paired = {item.task_reference for item in self.paired_report.cases}
        return tuple(sorted(paired.difference(self.proof_task_references)))

    @property
    def unexpected_proof_task_references(self) -> tuple[str, ...]:
        paired = {item.task_reference for item in self.paired_report.cases}
        return tuple(sorted(set(self.proof_task_references).difference(paired)))

    @property
    def proof_population_complete(self) -> bool:
        return bool(self.paired_report.cases) and not (
            self.missing_proof_task_references
            or self.unexpected_proof_task_references
        )

    @property
    def delta_receipt_count(self) -> int:
        return len(self.proof_bindings)

    def _per_task_tokens(self, name: str) -> tuple[int, ...]:
        totals: dict[str, int] = {}
        for binding in self.proof_bindings:
            totals[binding.task_reference] = (
                totals.get(binding.task_reference, 0)
                + getattr(binding, name)
            )
        return tuple(totals[task] for task in sorted(totals))

    @property
    def median_full_replay_input_tokens(self) -> int:
        return _median_integer(self._per_task_tokens("full_replay_tokens"))

    @property
    def median_delta_input_tokens(self) -> int:
        return _median_integer(self._per_task_tokens("delta_tokens"))

    @property
    def median_delta_input_token_reduction_bps(self) -> int:
        totals: dict[str, list[int]] = {}
        for binding in self.proof_bindings:
            task = totals.setdefault(binding.task_reference, [0, 0])
            task[0] += binding.full_replay_tokens
            task[1] += binding.delta_tokens
        return _median_integer(
            tuple(
                (full - delta) * BASIS_POINTS // full
                for full, delta in (
                    totals[task] for task in sorted(totals)
                )
                if full
            )
        )

    @property
    def token_accounting_consistent(self) -> bool:
        """Whether every charged lifecycle input token has a retry proof."""

        case_by_task = {
            item.task_reference: item for item in self.paired_report.cases
        }
        full_by_task: dict[str, int] = {}
        delta_by_task: dict[str, int] = {}
        for binding in self.proof_bindings:
            full_by_task[binding.task_reference] = (
                full_by_task.get(binding.task_reference, 0)
                + binding.full_replay_tokens
            )
            delta_by_task[binding.task_reference] = (
                delta_by_task.get(binding.task_reference, 0)
                + binding.delta_tokens
            )
        return all(
            task in case_by_task
            and full_by_task[task]
            == case_by_task[task].baseline_input_tokens
            and delta_by_task[task]
            == case_by_task[task].candidate_input_tokens
            for task in set(full_by_task) | set(delta_by_task)
        )

    @property
    def typed_delta_gate_passed(self) -> bool:
        return (
            self.proof_population_complete
            and bool(self.proof_bindings)
            and all(
                item.provider_tokens_verified
                for item in self.proof_bindings
            )
            and all(item.coverage_preserved for item in self.proof_bindings)
            and all(
                item.delta_tokens < item.full_replay_tokens
                for item in self.proof_bindings
            )
            and self.median_delta_input_token_reduction_bps
            >= self.paired_report.minimum_input_token_reduction_bps
            and self.token_accounting_consistent
        )

    @property
    def paired_efficiency_gate_passed(self) -> bool:
        return bool(
            self.terminal_work_evidence is not None
            and self.terminal_work_evidence.promotion_eligible
        )

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        if not self.passed:
            return ()
        return (DELTA_RETRY_CONTEXT_EVIDENCE_ID,)

    @property
    def passed(self) -> bool:
        return (
            self.paired_efficiency_gate_passed
            and self.typed_delta_gate_passed
        )

    @property
    def promotion_eligible(self) -> bool:
        return self.passed

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": EFFICIENCY_CONTRACT_VERSION,
            "paired_report": self.paired_report,
            "terminal_work_evidence": self.terminal_work_evidence,
            "proof_bindings": self.proof_bindings,
            "proof_task_references": self.proof_task_references,
            "missing_proof_task_references": (
                self.missing_proof_task_references
            ),
            "unexpected_proof_task_references": (
                self.unexpected_proof_task_references
            ),
            "proof_population_complete": self.proof_population_complete,
            "delta_receipt_count": self.delta_receipt_count,
            "median_full_replay_input_tokens": (
                self.median_full_replay_input_tokens
            ),
            "median_delta_input_tokens": self.median_delta_input_tokens,
            "median_delta_input_token_reduction_bps": (
                self.median_delta_input_token_reduction_bps
            ),
            "token_accounting_consistent": self.token_accounting_consistent,
            "typed_delta_gate_passed": self.typed_delta_gate_passed,
            "paired_efficiency_gate_passed": (
                self.paired_efficiency_gate_passed
            ),
            "evidence_claim_references": self.evidence_claim_references,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
        }

    def to_dict(self, *, include_report_id: bool = False) -> dict[str, Any]:
        payload = super().to_dict()
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        verifiers_by_receipt: Mapping[str, Any] | None = None,
    ) -> "DeltaRetryPromotionReport":
        _schema(payload, cls.SCHEMA, "delta retry promotion report")
        if verifiers_by_receipt is not None and not isinstance(
            verifiers_by_receipt, Mapping
        ):
            raise ContractValidationError(
                "verifiers_by_receipt must be an object"
            )
        derived = {
            "proof_task_references",
            "missing_proof_task_references",
            "unexpected_proof_task_references",
            "proof_population_complete",
            "delta_receipt_count",
            "median_full_replay_input_tokens",
            "median_delta_input_tokens",
            "median_delta_input_token_reduction_bps",
            "token_accounting_consistent",
            "typed_delta_gate_passed",
            "paired_efficiency_gate_passed",
            "evidence_claim_references",
            "passed",
            "promotion_eligible",
        }
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "paired_report",
                "terminal_work_evidence",
                "proof_bindings",
                "report_id",
                "content_id",
                *derived,
            },
            artifact_name="delta retry promotion report",
        )
        paired_payload = payload.get("paired_report")
        if not isinstance(paired_payload, Mapping):
            raise ContractValidationError(
                "delta retry promotion report requires a paired report"
            )
        terminal_payload = payload.get("terminal_work_evidence")
        if terminal_payload is not None and not isinstance(
            terminal_payload, Mapping
        ):
            raise ContractValidationError(
                "terminal_work_evidence must be an object"
            )
        binding_payloads = payload.get("proof_bindings") or ()
        bindings: list[DeltaRetryProofBinding] = []
        for item in binding_payloads:
            if not isinstance(item, Mapping):
                raise ContractValidationError(
                    "delta retry proof binding must be an object"
                )
            receipt_id = item.get("receipt_id", "")
            verifier = (
                verifiers_by_receipt.get(receipt_id)
                if verifiers_by_receipt is not None
                else None
            )
            bindings.append(
                DeltaRetryProofBinding.from_dict(
                    item,
                    verifier=verifier,
                )
            )
        result = cls(
            paired_report=PairedEfficiencyReport.from_dict(paired_payload),
            terminal_work_evidence=(
                TerminalAcceptedWorkEvidence.from_dict(
                    terminal_payload
                )
                if terminal_payload is not None
                else None
            ),
            proof_bindings=tuple(bindings),
        )
        for name in derived:
            claimed = payload.get(name, getattr(result, name))
            if isinstance(getattr(result, name), tuple):
                claimed = tuple(claimed)
            if claimed != getattr(result, name):
                raise ContractValidationError(
                    f"{name} claim does not match delta retry promotion report"
                )
        _claim(payload, result.report_id, "report_id", "content_id")
        return result

    @classmethod
    def from_json(
        cls,
        value: str | bytes | bytearray,
        *,
        verifiers_by_receipt: Mapping[str, Any] | None = None,
    ) -> "DeltaRetryPromotionReport":
        return cls.from_dict(
            _load_json(value, artifact_name="delta retry promotion report"),
            verifiers_by_receipt=verifiers_by_receipt,
        )


def _coerce_paired_measurement(
    value: (
        PairedEfficiencyReport
        | TerminalAcceptedWorkEvidence
        | Mapping[str, Any]
    ),
) -> tuple[PairedEfficiencyReport, TerminalAcceptedWorkEvidence | None]:
    """Normalize a diagnostic report or an authority-bearing measurement."""

    if isinstance(value, TerminalAcceptedWorkEvidence):
        return value.paired_report, value
    if isinstance(value, Mapping):
        if value.get("schema") == TERMINAL_ACCEPTED_WORK_EVIDENCE_SCHEMA:
            evidence = TerminalAcceptedWorkEvidence.from_dict(value)
            return evidence.paired_report, evidence
        value = PairedEfficiencyReport.from_dict(value)
    if not isinstance(value, PairedEfficiencyReport):
        raise ContractValidationError(
            "paired measurement must be a PairedEfficiencyReport or "
            "TerminalAcceptedWorkEvidence"
        )
    return value, None


def build_required_context_promotion_report(
    paired_report: (
        PairedEfficiencyReport
        | TerminalAcceptedWorkEvidence
        | Mapping[str, Any]
    ),
    context_results_by_task: Mapping[str, Sequence[Any]],
) -> RequiredContextPromotionReport:
    """Join paired terminal work to capsule-verified context compilations.

    Every provider input charged to the candidate arm must have a typed
    compiler result.  Their exact input-token sum, repository tree, policy,
    objective, and required-coverage population are reconciled with the
    corresponding paired task before the requirement ID can be emitted.
    """

    paired_report, terminal_evidence = _coerce_paired_measurement(
        paired_report
    )
    if not isinstance(context_results_by_task, Mapping):
        raise ContractValidationError(
            "context_results_by_task must be an object"
        )
    cases = {item.task_reference: item for item in paired_report.cases}
    unknown = set(context_results_by_task).difference(cases)
    if unknown:
        raise ContractValidationError(
            "required context proofs contain tasks outside the paired population"
        )
    bindings: list[RequiredContextProofBinding] = []
    for task_reference in sorted(context_results_by_task):
        values = context_results_by_task[task_reference]
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes, bytearray, memoryview)
        ):
            raise ContractValidationError(
                "each required context proof population must be a sequence"
            )
        if len(values) > MAX_STAGES + MAX_RETRIES:
            raise ContractValidationError(
                "a task carries too many context compilation results"
            )
        case = cases[task_reference]
        task_bindings: list[RequiredContextProofBinding] = []
        for result in values:
            binding = RequiredContextProofBinding.from_context_compile_result(
                task_reference,
                result,
            )
            if binding.objective_id != case.goal_reference:
                raise ContractValidationError(
                    "required context objective does not match its paired task"
                )
            if binding.tree_id != case.repository_tree_digest:
                raise ContractValidationError(
                    "required context tree does not match its paired task"
                )
            if binding.policy_revision != case.policy_digest:
                raise ContractValidationError(
                    "required context policy does not match its paired task"
                )
            task_bindings.append(binding)
        required_coverage = {
            coverage_id
            for binding in task_bindings
            for coverage_id in binding.required_coverage_ids
        }
        if task_bindings and required_coverage != set(
            case.required_evidence_references
        ):
            raise ContractValidationError(
                "required context coverage does not match the paired task's "
                "authoritative evidence population"
            )
        bindings.extend(task_bindings)
    return RequiredContextPromotionReport(
        paired_report=paired_report,
        proof_bindings=tuple(bindings),
        terminal_work_evidence=terminal_evidence,
    )


def build_delta_retry_promotion_report(
    paired_report: (
        PairedEfficiencyReport
        | TerminalAcceptedWorkEvidence
        | Mapping[str, Any]
    ),
    delta_results_by_task: Mapping[str, Sequence[Any]],
) -> DeltaRetryPromotionReport:
    """Join a paired report to capsule-verified delta results by task.

    Missing task proofs produce a deterministic, promotion-ineligible report
    so operators can diagnose incomplete benchmark collection.  Unknown task
    keys and identity mismatches are rejected because they indicate a stale or
    ambiguously joined evidence population.  Receipt-only inputs are rejected:
    promotion requires the delta and reconstructed capsules so the producer's
    artifact digest and reconstruction checks can be rerun.
    """

    paired_report, terminal_evidence = _coerce_paired_measurement(
        paired_report
    )
    if not isinstance(delta_results_by_task, Mapping):
        raise ContractValidationError(
            "delta_results_by_task must be an object"
        )
    cases = {item.task_reference: item for item in paired_report.cases}
    unknown = set(delta_results_by_task).difference(cases)
    if unknown:
        raise ContractValidationError(
            "delta retry proofs contain tasks outside the paired population"
        )
    bindings: list[DeltaRetryProofBinding] = []
    for task_reference in sorted(delta_results_by_task):
        values = delta_results_by_task[task_reference]
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes, bytearray, memoryview)
        ):
            raise ContractValidationError(
                "each delta retry proof population must be a sequence"
            )
        if len(values) > MAX_RETRIES:
            raise ContractValidationError(
                f"a task cannot carry more than {MAX_RETRIES} delta receipts"
            )
        case = cases[task_reference]
        for result in values:
            binding = DeltaRetryProofBinding.from_context_delta_result(
                task_reference,
                result,
            )
            if binding.objective_id != case.goal_reference:
                raise ContractValidationError(
                    "context delta objective does not match its paired task"
                )
            if binding.tree_id != case.repository_tree_digest:
                raise ContractValidationError(
                    "context delta tree does not match its paired task"
                )
            if binding.policy_revision != case.policy_digest:
                raise ContractValidationError(
                    "context delta policy does not match its paired task"
                )
            if set(binding.required_coverage_ids) != set(
                case.required_evidence_references
            ):
                raise ContractValidationError(
                    "context delta requirements do not match the paired "
                    "task's authoritative evidence population"
                )
            bindings.append(binding)
    return DeltaRetryPromotionReport(
        paired_report=paired_report,
        proof_bindings=tuple(bindings),
        terminal_work_evidence=terminal_evidence,
    )


def _normalize_efficiency_arm(
    receipts: Iterable[EfficiencyReceipt | Mapping[str, Any]],
    *,
    arm_name: str,
) -> tuple[
    dict[str, tuple[EfficiencyReceipt, ...]],
    dict[str, EfficiencyReceipt],
]:
    by_task: dict[str, list[EfficiencyReceipt]] = {}
    accepted_by_task: dict[str, EfficiencyReceipt] = {}
    receipt_ids: set[str] = set()
    for index, value in enumerate(receipts):
        if index >= MAX_RECEIPTS_PER_REPORT:
            raise ContractValidationError(
                f"{arm_name} receipt population exceeds "
                f"{MAX_RECEIPTS_PER_REPORT} items"
            )
        if isinstance(value, EfficiencyReceipt):
            receipt = value
        elif isinstance(value, Mapping):
            receipt = EfficiencyReceipt.from_dict(value)
        else:
            raise ContractValidationError(
                f"{arm_name}_receipts[{index}] must be an EfficiencyReceipt"
            )
        if receipt.receipt_id in receipt_ids:
            raise ContractValidationError(
                f"duplicate receipt identity in {arm_name} efficiency arm"
            )
        receipt_ids.add(receipt.receipt_id)
        by_task.setdefault(receipt.task_reference, []).append(receipt)
        if receipt.accepted:
            if receipt.task_reference in accepted_by_task:
                raise ContractValidationError(
                    f"a task may have only one accepted receipt in the "
                    f"{arm_name} efficiency arm"
                )
            accepted_by_task[receipt.task_reference] = receipt
    return (
        {
            task: tuple(sorted(items, key=lambda item: item.receipt_id))
            for task, items in by_task.items()
        },
        accepted_by_task,
    )


def build_paired_efficiency_report(
    baseline_receipts: Iterable[EfficiencyReceipt | Mapping[str, Any]],
    candidate_receipts: Iterable[EfficiencyReceipt | Mapping[str, Any]],
    *,
    required_evidence_by_task: Mapping[str, Sequence[str]] | None = None,
    minimum_input_token_reduction_bps: int = (
        DEFAULT_MINIMUM_INPUT_TOKEN_REDUCTION_BPS
    ),
) -> PairedEfficiencyReport:
    """Compare identical terminal accepted tasks for token/coverage regressions.

    Failed-only tasks never enter the paired denominator.  For a task that
    eventually succeeds, however, all receipts sharing that stable task
    reference are charged to the arm.  Unpaired accepted tasks are disclosed
    and fail the report gate rather than being silently dropped.

    When ``required_evidence_by_task`` is omitted, the baseline terminal
    evidence set is the required set.  An explicit mapping must define every
    paired task and is useful when the benchmark owns a smaller authoritative
    requirement set.
    """

    baseline_by_task, baseline_accepted = _normalize_efficiency_arm(
        baseline_receipts, arm_name="baseline"
    )
    candidate_by_task, candidate_accepted = _normalize_efficiency_arm(
        candidate_receipts, arm_name="candidate"
    )
    baseline_tasks = set(baseline_accepted)
    candidate_tasks = set(candidate_accepted)
    paired_tasks = baseline_tasks & candidate_tasks

    if required_evidence_by_task is not None and not isinstance(
        required_evidence_by_task, Mapping
    ):
        raise ContractValidationError(
            "required_evidence_by_task must be an object"
        )
    cases: list[PairedEfficiencyCase] = []
    for task_reference in sorted(paired_tasks):
        baseline_terminal = baseline_accepted[task_reference]
        candidate_terminal = candidate_accepted[task_reference]
        for name in (
            "goal_reference",
            "repository_tree_digest",
            "policy_digest",
        ):
            if getattr(baseline_terminal, name) != getattr(
                candidate_terminal, name
            ):
                raise ContractValidationError(
                    f"paired receipts must bind the same {name}"
                )
        if required_evidence_by_task is None:
            required = baseline_terminal.evidence.terminal_references
        else:
            if task_reference not in required_evidence_by_task:
                raise ContractValidationError(
                    "required_evidence_by_task must define every paired task"
                )
            required = _strings(
                required_evidence_by_task[task_reference],
                field_name=f"required_evidence_by_task.{task_reference}",
                maximum=MAX_EVIDENCE_REFERENCES,
            )
        required_set = set(required)
        baseline_covered = tuple(
            sorted(
                required_set.intersection(
                    baseline_terminal.evidence.terminal_references
                )
            )
        )
        candidate_covered = tuple(
            sorted(
                required_set.intersection(
                    candidate_terminal.evidence.terminal_references
                )
            )
        )
        baseline_attempts = baseline_by_task[task_reference]
        candidate_attempts = candidate_by_task[task_reference]
        for arm_name, terminal, attempts in (
            ("baseline", baseline_terminal, baseline_attempts),
            ("candidate", candidate_terminal, candidate_attempts),
        ):
            for attempt in attempts:
                if any(
                    getattr(attempt, name) != getattr(terminal, name)
                    for name in (
                        "goal_reference",
                        "repository_tree_digest",
                        "policy_digest",
                    )
                ):
                    raise ContractValidationError(
                        f"all charged {arm_name} attempts must bind the "
                        "accepted task's frozen goal, repository tree, and policy"
                    )
        cases.append(
            PairedEfficiencyCase(
                task_reference=task_reference,
                goal_reference=baseline_terminal.goal_reference,
                repository_tree_digest=(
                    baseline_terminal.repository_tree_digest
                ),
                policy_digest=baseline_terminal.policy_digest,
                baseline_receipt_ids=tuple(
                    item.receipt_id for item in baseline_attempts
                ),
                candidate_receipt_ids=tuple(
                    item.receipt_id for item in candidate_attempts
                ),
                baseline_terminal_receipt_id=baseline_terminal.receipt_id,
                candidate_terminal_receipt_id=candidate_terminal.receipt_id,
                baseline_input_tokens=sum(
                    item.input_tokens for item in baseline_attempts
                ),
                candidate_input_tokens=sum(
                    item.input_tokens for item in candidate_attempts
                ),
                required_evidence_references=tuple(required),
                baseline_covered_evidence_references=baseline_covered,
                candidate_covered_evidence_references=candidate_covered,
            )
        )
    return PairedEfficiencyReport(
        cases=tuple(cases),
        baseline_unpaired_accepted_task_references=tuple(
            baseline_tasks - candidate_tasks
        ),
        candidate_unpaired_accepted_task_references=tuple(
            candidate_tasks - baseline_tasks
        ),
        minimum_input_token_reduction_bps=(
            minimum_input_token_reduction_bps
        ),
    )


def build_terminal_accepted_work_evidence(
    baseline_receipts: Iterable[EfficiencyReceipt | Mapping[str, Any]],
    candidate_receipts: Iterable[EfficiencyReceipt | Mapping[str, Any]],
    *,
    required_evidence_by_task: Mapping[str, Sequence[str]] | None = None,
    minimum_input_token_reduction_bps: int = (
        DEFAULT_MINIMUM_INPUT_TOKEN_REDUCTION_BPS
    ),
) -> TerminalAcceptedWorkEvidence:
    """Build a replayable ASI-G093 benchmark receipt.

    Both iterables are materialized exactly once, retained as typed source
    records, and replayed by :class:`TerminalAcceptedWorkEvidence`.  The
    resulting requirement claim proves the accepted-work accounting boundary;
    ``promotion_eligible`` additionally reports the independent token and
    coverage gates.
    """

    baseline = tuple(baseline_receipts)
    candidate = tuple(candidate_receipts)
    report = build_paired_efficiency_report(
        baseline,
        candidate,
        required_evidence_by_task=required_evidence_by_task,
        minimum_input_token_reduction_bps=(
            minimum_input_token_reduction_bps
        ),
    )
    return TerminalAcceptedWorkEvidence(
        paired_report=report,
        baseline_receipts=baseline,
        candidate_receipts=candidate,
    )


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
compare_paired_efficiency_receipts = build_paired_efficiency_report
PairedSupervisorEfficiencyCase = PairedEfficiencyCase
PairedSupervisorEfficiencyReport = PairedEfficiencyReport


__all__ = [
    "ARTIFACT_REFERENCE_SCHEMA",
    "CACHE_OBSERVATION_SCHEMA",
    "CHANGED_SCOPE_SCHEMA",
    "DEFAULT_MINIMUM_INPUT_TOKEN_REDUCTION_BPS",
    "DELTA_RETRY_CONTEXT_EVIDENCE_ID",
    "DELTA_RETRY_PROMOTION_REPORT_SCHEMA",
    "DELTA_RETRY_PROOF_BINDING_SCHEMA",
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
    "PAIRED_EFFICIENCY_CASE_SCHEMA",
    "PAIRED_EFFICIENCY_REPORT_SCHEMA",
    "REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID",
    "REQUIRED_CONTEXT_PROOF_BINDING_SCHEMA",
    "REQUIRED_CONTEXT_PROMOTION_REPORT_SCHEMA",
    "RETRY_OBSERVATION_SCHEMA",
    "SCHEMA_VERSION",
    "STAGE_TIMING_SCHEMA",
    "TERMINAL_ACCEPTANCE_SCHEMA",
    "TERMINAL_ACCEPTED_WORK_EVIDENCE_SCHEMA",
    "TOKEN_USAGE_SCHEMA",
    "WORK_COST_SCHEMA",
    "ArtifactReference",
    "CacheDisposition",
    "CacheObservation",
    "ChangedScope",
    "DeltaRetryProofBinding",
    "DeltaRetryPromotionReport",
    "EfficiencyAggregate",
    "EfficiencyReceipt",
    "EfficiencyReport",
    "EfficiencyScenario",
    "EfficiencyValidationError",
    "EvidenceDelta",
    "ExactRatio",
    "RetryObservation",
    "RequiredContextProofBinding",
    "RequiredContextPromotionReport",
    "StageName",
    "StageTiming",
    "SupervisorEfficiencyReceipt",
    "SupervisorEfficiencyReport",
    "TERMINAL_ACCEPTED_WORK_EVIDENCE_ID",
    "TerminalAcceptance",
    "TerminalAcceptedWorkEvidence",
    "TerminalOutcome",
    "TokenUsage",
    "WorkCost",
    "WorkStatus",
    "aggregate_efficiency_receipts",
    "aggregate_receipts",
    "build_paired_efficiency_report",
    "build_terminal_accepted_work_evidence",
    "build_required_context_promotion_report",
    "build_delta_retry_promotion_report",
    "build_efficiency_baseline_fixtures",
    "compare_paired_efficiency_receipts",
    "make_baseline_fixtures",
    "PairedEfficiencyCase",
    "PairedEfficiencyReport",
    "PairedSupervisorEfficiencyCase",
    "PairedSupervisorEfficiencyReport",
]
