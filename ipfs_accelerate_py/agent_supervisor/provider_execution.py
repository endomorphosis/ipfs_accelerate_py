"""Reservation-aware supervisor provider execution gateway (ASI-166).

Lifecycle for every chargeable supervisor provider call:

    estimate → exact reserve → invoke (router/typed adapter) → settle/reconcile
    → redacted endpoint + supervisor attribution receipt

Exact request/attempt replay cannot reinvoke or recharge a terminal outcome.
Pre-dispatch cancellation releases the reservation; post-dispatch timeout or
cancel conservatively settles because the provider may still charge.  Enforce
mode fails closed on unknown or stale coordination unless a reviewed degraded
budget permits local/deterministic fallback.

This module is pure on cold import: no network, provider, process, database,
or secret-store I/O.  Usage receipts remain operational evidence only and never
authorize usage, rewrite provider settlement, or prove completion/correctness.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final, Optional, Protocol

from ipfs_accelerate_py.endpoint_usage.identity import (
    assert_no_prompt_media_or_output,
    contains_bearer_url,
    contains_raw_endpoint,
    is_secret_key,
    is_secret_value,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    EstimateMethod,
    Quantity,
    QuantityKind,
    SchemaValidationError,
    UsageDimension,
    UsageEstimate,
    UsageVector,
    UsageVectorEntry,
)

from .formal_verification_contracts import CanonicalContract
from .provider_usage import (
    BRIDGE_AUTHORIZES_USAGE,
    BRIDGE_IS_COMPLETION_EVIDENCE,
    BRIDGE_IS_CORRECTNESS_EVIDENCE,
    BRIDGE_REWRITES_PROVIDER_SETTLEMENT,
    ProviderUsageValidationError,
    SupervisorToEndpointRequest,
    SupervisorUsageAttribution,
    SupervisorUsageEnvelope,
    SupervisorUsageFinalStatus,
    SupervisorUsageLevel,
    SupervisorUsageReceipt,
    SupervisorUsageScope,
    attribute_endpoint_events,
    consume_reconciled_endpoint_events_exactly_once,
)


RESERVATION_AWARE_PROVIDER_EXECUTION_REQUIREMENT_ID: Final[str] = (
    "requirement:reservation-aware-provider-execution.v1"
)
PROVIDER_EXECUTION_GOAL_ID: Final[str] = "ASI-G510"
PROVIDER_EXECUTION_CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = PROVIDER_EXECUTION_CONTRACT_VERSION

PROVIDER_EXECUTION_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-execution-request@1"
)
PROVIDER_EXECUTION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-execution-result@1"
)
PROVIDER_EXECUTION_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-execution-observation@1"
)

MAX_TEXT_BYTES: Final[int] = 512
MAX_REASON_CODES: Final[int] = 64
MAX_METADATA_KEYS: Final[int] = 32
MAX_OBSERVATION_KEYS: Final[int] = 32
MAX_SERIALIZED_BYTES: Final[int] = 4 * 1024 * 1024
MAX_ABS_CEILING: Final[int] = (1 << 63) - 1

GATEWAY_AUTHORIZES_USAGE: Final[bool] = False
GATEWAY_REWRITES_PROVIDER_SETTLEMENT: Final[bool] = False
GATEWAY_IS_COMPLETION_EVIDENCE: Final[bool] = False
GATEWAY_IS_CORRECTNESS_EVIDENCE: Final[bool] = False

_TEXT_SAFE = re.compile(r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+$")
_NAME = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")

_FORBIDDEN_OBSERVATION_KEYS = frozenset(
    {
        "prompt",
        "messages",
        "message",
        "source",
        "media",
        "image_data",
        "audio_data",
        "video_data",
        "output",
        "output_text",
        "completion",
        "payload",
        "raw_headers",
        "raw_body",
        "response_body",
        "credential",
        "credentials",
        "password",
        "secret",
        "api_key",
        "authorization",
        "token",
        "endpoint",
        "url",
        "uri",
    }
)


class ProviderExecutionError(ValueError):
    """Fail-closed gateway contract or coordination error."""

    def __init__(
        self,
        message: str,
        *,
        reason_codes: Sequence[str] = (),
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason_codes = tuple(
            _reason_code(code) for code in reason_codes if str(code).strip()
        )
        self.retryable = bool(retryable)


class ProviderExecutionMode(str, Enum):
    """Promotion modes for the execution gateway."""

    OFF = "off"
    OBSERVE = "observe"
    SHADOW = "shadow"
    ASSIST = "assist"
    ENFORCE = "enforce"


class SideEffectBoundary(str, Enum):
    """Expected provider side-effect class for the request."""

    NONE = "none"
    READ_ONLY = "read_only"
    IDEMPOTENT = "idempotent"
    SIDE_EFFECTING = "side_effecting"


class ProviderExecutionPhase(str, Enum):
    """Lifecycle phase recorded on a result."""

    ACCEPTED = "accepted"
    ESTIMATED = "estimated"
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    SETTLED = "settled"
    RELEASED = "released"
    DENIED = "denied"
    CANCELLED = "cancelled"
    FAILED = "failed"
    DEGRADED = "degraded"
    REPLAYED = "replayed"


class CoordinationState(str, Enum):
    """Health of the injected usage coordinator path."""

    AVAILABLE = "available"
    UNKNOWN = "unknown"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    SIMULATED = "simulated"


def _fail(
    message: str,
    *,
    reason_codes: Sequence[str] = (),
    retryable: bool = False,
) -> None:
    raise ProviderExecutionError(
        message, reason_codes=reason_codes, retryable=retryable
    )


def _reason_code(value: Any) -> str:
    text = str(value or "").strip().casefold().replace(" ", "_")
    if not text or not _NAME.fullmatch(text):
        return "invalid_reason"
    if len(text) > 64:
        return text[:64]
    return text


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        if required:
            _fail(f"{name} must be text", reason_codes=("missing_field", name))
        return ""
    if not isinstance(value, str):
        _fail(f"{name} must be text", reason_codes=("invalid_field", name))
    result = value.strip()
    if required and not result:
        _fail(f"{name} must not be empty", reason_codes=("missing_field", name))
    if not required and not result:
        return ""
    if len(result.encode("utf-8")) > maximum:
        _fail(f"{name} is too large", reason_codes=("field_too_large", name))
    if not _TEXT_SAFE.fullmatch(result):
        _fail(
            f"{name} contains control characters",
            reason_codes=("control_characters", name),
        )
    if is_secret_value(result) or contains_bearer_url(result):
        _fail(
            f"{name} contains credential-shaped data",
            reason_codes=("credential_shaped", name),
        )
    if contains_raw_endpoint(result):
        _fail(
            f"{name} must not embed a raw endpoint or URL",
            reason_codes=("raw_endpoint", name),
        )
    return result


def _optional_text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    return _text(value, name, required=False, maximum=maximum)


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_ABS_CEILING,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer", reason_codes=("invalid_field", name))
    if value < minimum or value > maximum:
        _fail(
            f"{name} must be between {minimum} and {maximum}",
            reason_codes=("out_of_range", name),
        )
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(raw)
    except (TypeError, ValueError) as exc:
        raise ProviderExecutionError(
            f"{name} is not a supported {enum_type.__name__}",
            reason_codes=("invalid_enum", name),
        ) from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{name} must be boolean", reason_codes=("invalid_field", name))
    return value


def _usage_vector(value: Any) -> UsageVector:
    if isinstance(value, UsageVector):
        return value
    if value is None:
        return UsageVector()
    if isinstance(value, Mapping):
        # Accept either the canonical {"entries": [...]} form or a compact
        # dimension->amount mapping used by typed adapters/tests.
        if "entries" in value or "schema" in value or "schema_version" in value:
            try:
                return UsageVector.from_dict(value)
            except (SchemaValidationError, TypeError, ValueError) as exc:
                raise ProviderExecutionError(
                    "usage vector is malformed",
                    reason_codes=("malformed_usage_vector",),
                ) from exc
        compact: dict[str, int] = {}
        currency: Optional[str] = None
        for key, item in value.items():
            name = str(key)
            if name == "currency" and isinstance(item, str):
                currency = item
                continue
            if isinstance(item, bool) or not isinstance(item, int):
                _fail(
                    "usage vector is malformed",
                    reason_codes=("malformed_usage_vector",),
                )
            compact[name] = int(item)
        try:
            return UsageVector.of(currency=currency, **compact)
        except (SchemaValidationError, TypeError, ValueError) as exc:
            raise ProviderExecutionError(
                "usage vector is malformed",
                reason_codes=("malformed_usage_vector",),
            ) from exc
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return UsageVector.from_dict(value)
        except (SchemaValidationError, TypeError, ValueError) as exc:
            raise ProviderExecutionError(
                "usage vector is malformed",
                reason_codes=("malformed_usage_vector",),
            ) from exc
    _fail("usage vector is malformed", reason_codes=("malformed_usage_vector",))


def _closed(
    payload: Mapping[str, Any],
    *,
    schema: str,
    allowed: Sequence[str],
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        _fail(f"{name} must be an object", reason_codes=("invalid_payload",))
    if payload.get("schema") != schema:
        _fail(f"unsupported {name} schema", reason_codes=("unsupported_schema",))
    version = payload.get("contract_version", payload.get("schema_version"))
    if version != PROVIDER_EXECUTION_CONTRACT_VERSION:
        _fail(f"unsupported {name} version", reason_codes=("unsupported_version",))
    unknown = set(payload).difference(allowed)
    if unknown:
        _fail(
            f"{name} contains unknown fields: {sorted(unknown)}",
            reason_codes=("unknown_fields",),
        )


def _claim(payload: Mapping[str, Any], actual: str, *names: str) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "", actual):
            _fail(
                "content identity does not match canonical contents",
                reason_codes=("content_identity_mismatch",),
            )


def _reject_forbidden_payload(payload: Mapping[str, Any]) -> None:
    try:
        assert_no_prompt_media_or_output(dict(payload))
    except Exception as exc:
        raise ProviderExecutionError(
            str(exc), reason_codes=("forbidden_payload",)
        ) from exc
    for key in payload:
        key_text = str(key)
        if is_secret_key(key_text) and "pseudonym" not in key_text.casefold():
            _fail(
                f"forbidden credential field: {key}",
                reason_codes=("forbidden_credential_field",),
            )


def _reason_codes(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        _fail("reason_codes must be a sequence", reason_codes=("invalid_reason_codes",))
    if len(values) > MAX_REASON_CODES:
        _fail("reason_codes exceeds maximum count", reason_codes=("too_many_reasons",))
    out = [_reason_code(item) for item in values]
    return tuple(sorted(set(out)))


def _safe_metadata(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _fail("metadata must be an object", reason_codes=("invalid_metadata",))
    if len(value) > MAX_METADATA_KEYS:
        _fail("metadata exceeds key bound", reason_codes=("metadata_too_large",))
    out: dict[str, str] = {}
    for key, item in value.items():
        name = _text(str(key), "metadata_key", maximum=64)
        lowered = name.casefold()
        if lowered in _FORBIDDEN_OBSERVATION_KEYS or is_secret_key(name):
            _fail(
                f"forbidden metadata key: {name}",
                reason_codes=("forbidden_metadata",),
            )
        if isinstance(item, bool):
            text = "true" if item else "false"
        elif isinstance(item, int) and not isinstance(item, bool):
            text = str(item)
        elif isinstance(item, str):
            text = _text(item, f"metadata.{name}", maximum=256)
        elif item is None:
            continue
        else:
            _fail(
                f"metadata.{name} must be a scalar",
                reason_codes=("invalid_metadata",),
            )
        out[name] = text
    return out


def _sanitize_observation(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _fail(
            "observation must be an object",
            reason_codes=("invalid_observation",),
        )
    if len(value) > MAX_OBSERVATION_KEYS:
        _fail(
            "observation exceeds key bound",
            reason_codes=("observation_too_large",),
        )
    out: dict[str, Any] = {}
    for key, item in value.items():
        name = str(key).strip().casefold()
        if not name or not _NAME.fullmatch(name):
            continue
        if name in _FORBIDDEN_OBSERVATION_KEYS or is_secret_key(name):
            continue
        if isinstance(item, bool):
            out[name] = item
        elif isinstance(item, int) and not isinstance(item, bool):
            if abs(item) > MAX_ABS_CEILING:
                continue
            out[name] = item
        elif isinstance(item, str):
            try:
                out[name] = _text(item, f"observation.{name}", maximum=256)
            except ProviderExecutionError:
                continue
        elif isinstance(item, Mapping) and name == "units":
            out[name] = _usage_vector(item).to_dict()
        elif item is None:
            continue
    return out


def _redact_endpoint(endpoint: str) -> str:
    text = str(endpoint or "").strip()
    if not text:
        return "endpoint:redacted"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"endpoint:{digest}"


def _finite_amount(vector: UsageVector, dimension: UsageDimension) -> int:
    entry = vector.get(dimension)
    if entry is None or entry.amount.kind is not QuantityKind.FINITE:
        return 0
    return int(entry.amount.value or 0)


def _stable_scope_id(scope_id: str) -> str:
    """Normalize free-form endpoint binding labels to catalog-safe scope ids."""

    text = _text(scope_id, "scope_id")
    # Endpoint usage identities require a stable pseudonym/catalog form.
    if re.fullmatch(r"[a-z][a-z0-9_]{2,127}", text):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"scope_{digest}"


def conservative_estimate(
    *,
    scope_id: str,
    operation: str,
    requested: UsageVector | Mapping[str, Any] | None = None,
    method: EstimateMethod | str = EstimateMethod.CONSERVATIVE,
    estimated_at: str = "",
) -> UsageEstimate:
    """Build a conservative, operation-bound estimate (never raises ceilings)."""

    vector = _usage_vector(requested)
    if not vector.entries:
        vector = UsageVector(
            entries=(
                UsageVectorEntry(
                    dimension=UsageDimension.REQUESTS,
                    amount=Quantity.finite(1),
                ),
            )
        )
    method_value = _enum(method, EstimateMethod, "method")
    return UsageEstimate(
        scope_id=_stable_scope_id(scope_id),
        operation=_text(operation, "operation", maximum=128),
        requested=vector,
        method=method_value,
        estimated_at=_optional_text(estimated_at, "estimated_at") or None,
    )


class _ExecutionContract(CanonicalContract):
    @property
    def schema_version(self) -> int:
        return PROVIDER_EXECUTION_CONTRACT_VERSION

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "_ExecutionContract":
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        try:
            decoded = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderExecutionError(
                "execution contract JSON is malformed",
                reason_codes=("malformed_json",),
            ) from exc
        if not isinstance(decoded, Mapping):
            _fail(
                "execution contract JSON must contain an object",
                reason_codes=("malformed_json",),
            )
        return cls.from_dict(decoded)  # type: ignore[attr-defined,no-any-return]


@dataclass(frozen=True)
class ProviderExecutionRequest(_ExecutionContract):
    """Exact, idempotent supervisor provider execution request.

    Binds supervisor scope/envelope lineage, attempt/idempotency, catalog and
    usage revisions, endpoint binding, deadline, cancellation, lease/fence, and
    the expected provider side-effect boundary.
    """

    SCHEMA: ClassVar[str] = PROVIDER_EXECUTION_REQUEST_SCHEMA

    bridge: SupervisorToEndpointRequest
    envelope: SupervisorUsageEnvelope
    provider_id: str
    modality: str
    side_effect_boundary: SideEffectBoundary
    operation: str
    mode: ProviderExecutionMode = ProviderExecutionMode.ENFORCE
    cancelled: bool = False
    post_dispatch: bool = False
    timeout_expired: bool = False
    degraded_budget_id: str = ""
    coordination_state: CoordinationState = CoordinationState.AVAILABLE
    metadata: Mapping[str, str] = field(default_factory=dict)
    estimate: Optional[UsageEstimate] = None

    def __post_init__(self) -> None:
        if not isinstance(self.bridge, SupervisorToEndpointRequest):
            if isinstance(self.bridge, Mapping):
                try:
                    object.__setattr__(
                        self, "bridge", SupervisorToEndpointRequest.from_dict(self.bridge)
                    )
                except ProviderUsageValidationError as exc:
                    raise ProviderExecutionError(
                        str(exc), reason_codes=("invalid_bridge",)
                    ) from exc
            else:
                _fail(
                    "bridge must be SupervisorToEndpointRequest",
                    reason_codes=("invalid_bridge",),
                )
        if not isinstance(self.envelope, SupervisorUsageEnvelope):
            if isinstance(self.envelope, Mapping):
                try:
                    object.__setattr__(
                        self,
                        "envelope",
                        SupervisorUsageEnvelope.from_dict(self.envelope),
                    )
                except ProviderUsageValidationError as exc:
                    raise ProviderExecutionError(
                        str(exc), reason_codes=("invalid_envelope",)
                    ) from exc
            else:
                _fail(
                    "envelope must be SupervisorUsageEnvelope",
                    reason_codes=("invalid_envelope",),
                )
        if self.envelope.scope.level is not SupervisorUsageLevel.REQUEST:
            # Prefer the request-level node from the lineage when the root is
            # a parent envelope.
            request_nodes = [
                node
                for node in self.envelope.walk()
                if node.scope.level is SupervisorUsageLevel.REQUEST
                and node.scope.request_id == self.bridge.request_id
            ]
            if not request_nodes and self.envelope.scope.request_id != self.bridge.request_id:
                _fail(
                    "envelope lineage missing request-level scope",
                    reason_codes=("envelope_mismatch",),
                )
        if self.envelope.envelope_id != self.bridge.envelope_id:
            # Allow parent envelopes that contain the request child.
            child_ids = {node.envelope_id for node in self.envelope.walk()}
            if self.bridge.envelope_id not in child_ids:
                _fail(
                    "envelope_id foreign to bridge request",
                    reason_codes=("envelope_mismatch",),
                )
        bridge_scope = self.bridge.scope
        if bridge_scope.request_id != self.bridge.request_id:
            _fail("request_id foreign to bridge scope", reason_codes=("scope_mismatch",))
        if bridge_scope.idempotency_key != self.bridge.idempotency_key:
            _fail(
                "idempotency_key foreign to bridge scope",
                reason_codes=("scope_mismatch",),
            )
        if bridge_scope.catalog_revision != self.bridge.catalog_revision:
            _fail(
                "catalog_revision stale relative to bridge scope",
                reason_codes=("stale_catalog_revision",),
            )
        if bridge_scope.usage_revision != self.bridge.usage_revision:
            _fail(
                "usage_revision stale relative to bridge scope",
                reason_codes=("stale_usage_revision",),
            )
        if bridge_scope.lease_id != self.bridge.lease_id or bridge_scope.fence_id != self.bridge.fence_id:
            _fail(
                "lease or fence foreign to bridge scope",
                reason_codes=("lease_fence_mismatch",),
            )
        if not bridge_scope.deadline_at:
            _fail("deadline_at is required", reason_codes=("missing_deadline",))
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id"))
        object.__setattr__(self, "modality", _text(self.modality, "modality", maximum=64))
        object.__setattr__(
            self,
            "side_effect_boundary",
            _enum(self.side_effect_boundary, SideEffectBoundary, "side_effect_boundary"),
        )
        object.__setattr__(
            self, "operation", _text(self.operation, "operation", maximum=128)
        )
        object.__setattr__(
            self, "mode", _enum(self.mode, ProviderExecutionMode, "mode")
        )
        object.__setattr__(self, "cancelled", _bool(self.cancelled, "cancelled"))
        object.__setattr__(
            self, "post_dispatch", _bool(self.post_dispatch, "post_dispatch")
        )
        object.__setattr__(
            self, "timeout_expired", _bool(self.timeout_expired, "timeout_expired")
        )
        object.__setattr__(
            self,
            "degraded_budget_id",
            _optional_text(self.degraded_budget_id, "degraded_budget_id"),
        )
        object.__setattr__(
            self,
            "coordination_state",
            _enum(self.coordination_state, CoordinationState, "coordination_state"),
        )
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata))
        if self.estimate is not None and not isinstance(self.estimate, UsageEstimate):
            if isinstance(self.estimate, Mapping):
                try:
                    object.__setattr__(
                        self, "estimate", UsageEstimate.from_dict(self.estimate)
                    )
                except (SchemaValidationError, TypeError, ValueError) as exc:
                    raise ProviderExecutionError(
                        "estimate is malformed",
                        reason_codes=("malformed_estimate",),
                    ) from exc
            else:
                _fail("estimate is malformed", reason_codes=("malformed_estimate",))
        _reject_forbidden_payload(self._payload())

    @property
    def request_key(self) -> str:
        return self.bridge.idempotency_key

    @property
    def attempt_key(self) -> str:
        return f"{self.bridge.request_id}#{self.bridge.attempt}"

    @property
    def request_content_id(self) -> str:
        return self.content_id

    def effective_estimate(self) -> UsageEstimate:
        if self.estimate is not None:
            return self.estimate
        return conservative_estimate(
            scope_id=self.bridge.endpoint_scope_id,
            operation=self.operation,
            requested=self.bridge.estimated,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_EXECUTION_CONTRACT_VERSION,
            "bridge": self.bridge.to_record(),
            "envelope": self.envelope.to_record(),
            "envelope_id": self.bridge.envelope_id,
            "lineage_envelope_id": self.envelope.envelope_id,
            "provider_id": self.provider_id,
            "modality": self.modality,
            "side_effect_boundary": self.side_effect_boundary.value,
            "operation": self.operation,
            "mode": self.mode.value,
            "cancelled": self.cancelled,
            "post_dispatch": self.post_dispatch,
            "timeout_expired": self.timeout_expired,
            "degraded_budget_id": self.degraded_budget_id,
            "coordination_state": self.coordination_state.value,
            "metadata": dict(self.metadata),
            "estimate": self.estimate.to_dict() if self.estimate is not None else None,
            "request_key": self.request_key,
            "attempt_key": self.attempt_key,
            "catalog_revision": self.bridge.catalog_revision,
            "usage_revision": self.bridge.usage_revision,
            "endpoint_scope_id": self.bridge.endpoint_scope_id,
            "deadline_at": self.bridge.deadline_at,
            "lease_id": self.bridge.lease_id,
            "fence_id": self.bridge.fence_id,
            "attempt": self.bridge.attempt,
            "idempotency_key": self.bridge.idempotency_key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderExecutionRequest":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "bridge",
            "envelope",
            "envelope_id",
            "lineage_envelope_id",
            "provider_id",
            "modality",
            "side_effect_boundary",
            "operation",
            "mode",
            "cancelled",
            "post_dispatch",
            "timeout_expired",
            "degraded_budget_id",
            "coordination_state",
            "metadata",
            "estimate",
            "request_key",
            "attempt_key",
            "catalog_revision",
            "usage_revision",
            "endpoint_scope_id",
            "deadline_at",
            "lease_id",
            "fence_id",
            "attempt",
            "idempotency_key",
            "content_id",
            "request_content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=tuple(allowed),
            name="provider execution request",
        )
        envelope = payload.get("envelope")
        if envelope is None:
            _fail(
                "envelope is required for deserialization",
                reason_codes=("missing_envelope",),
            )
        result = cls(
            bridge=payload.get("bridge", {}),
            envelope=envelope,
            provider_id=payload.get("provider_id", ""),
            modality=payload.get("modality", ""),
            side_effect_boundary=payload.get("side_effect_boundary", ""),
            operation=payload.get("operation", ""),
            mode=payload.get("mode", ProviderExecutionMode.ENFORCE),
            cancelled=bool(payload.get("cancelled", False)),
            post_dispatch=bool(payload.get("post_dispatch", False)),
            timeout_expired=bool(payload.get("timeout_expired", False)),
            degraded_budget_id=payload.get("degraded_budget_id", ""),
            coordination_state=payload.get(
                "coordination_state", CoordinationState.AVAILABLE
            ),
            metadata=payload.get("metadata", {}),
            estimate=payload.get("estimate"),
        )
        _claim(payload, result.content_id, "content_id", "request_content_id")
        return result


@dataclass(frozen=True)
class ProviderExecutionResult(_ExecutionContract):
    """Normalized gateway outcome with operational receipts only."""

    SCHEMA: ClassVar[str] = PROVIDER_EXECUTION_RESULT_SCHEMA

    phase: ProviderExecutionPhase
    final_status: SupervisorUsageFinalStatus
    granted: bool
    reservation_id: str
    usage_revision: str
    catalog_revision: str
    provider_id: str
    redacted_endpoint: str
    attempt_key: str
    request_key: str
    request_id: str
    reason_codes: tuple[str, ...] = ()
    observation: Mapping[str, Any] = field(default_factory=dict)
    settled: UsageVector = field(default_factory=UsageVector)
    receipt: Optional[SupervisorUsageReceipt] = None
    endpoint_receipt_id: str = ""
    supervisor_receipt_id: str = ""
    attribution: Optional[SupervisorUsageAttribution] = None
    replayed: bool = False
    coordination_state: CoordinationState = CoordinationState.AVAILABLE
    mode: ProviderExecutionMode = ProviderExecutionMode.ENFORCE
    authorizes_usage: bool = False
    rewrites_provider_settlement: bool = False
    is_completion_evidence: bool = False
    is_correctness_evidence: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "phase", _enum(self.phase, ProviderExecutionPhase, "phase")
        )
        object.__setattr__(
            self,
            "final_status",
            _enum(self.final_status, SupervisorUsageFinalStatus, "final_status"),
        )
        object.__setattr__(self, "granted", _bool(self.granted, "granted"))
        object.__setattr__(
            self, "reservation_id", _optional_text(self.reservation_id, "reservation_id")
        )
        object.__setattr__(
            self, "usage_revision", _optional_text(self.usage_revision, "usage_revision")
        )
        object.__setattr__(
            self,
            "catalog_revision",
            _optional_text(self.catalog_revision, "catalog_revision"),
        )
        object.__setattr__(
            self, "provider_id", _optional_text(self.provider_id, "provider_id")
        )
        object.__setattr__(
            self,
            "redacted_endpoint",
            _optional_text(self.redacted_endpoint, "redacted_endpoint")
            or "endpoint:redacted",
        )
        object.__setattr__(
            self, "attempt_key", _optional_text(self.attempt_key, "attempt_key")
        )
        object.__setattr__(
            self, "request_key", _optional_text(self.request_key, "request_key")
        )
        object.__setattr__(
            self, "request_id", _optional_text(self.request_id, "request_id")
        )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(
            self, "observation", _sanitize_observation(self.observation)
        )
        object.__setattr__(self, "settled", _usage_vector(self.settled))
        if self.receipt is not None and not isinstance(
            self.receipt, SupervisorUsageReceipt
        ):
            if isinstance(self.receipt, Mapping):
                object.__setattr__(
                    self, "receipt", SupervisorUsageReceipt.from_dict(self.receipt)
                )
            else:
                _fail("receipt is malformed", reason_codes=("invalid_receipt",))
        object.__setattr__(
            self,
            "endpoint_receipt_id",
            _optional_text(self.endpoint_receipt_id, "endpoint_receipt_id"),
        )
        object.__setattr__(
            self,
            "supervisor_receipt_id",
            _optional_text(self.supervisor_receipt_id, "supervisor_receipt_id")
            or (self.receipt.receipt_id if self.receipt is not None else ""),
        )
        if self.attribution is not None and not isinstance(
            self.attribution, SupervisorUsageAttribution
        ):
            if isinstance(self.attribution, Mapping):
                object.__setattr__(
                    self,
                    "attribution",
                    SupervisorUsageAttribution.from_dict(self.attribution),
                )
            else:
                _fail(
                    "attribution is malformed",
                    reason_codes=("invalid_attribution",),
                )
        object.__setattr__(self, "replayed", _bool(self.replayed, "replayed"))
        object.__setattr__(
            self,
            "coordination_state",
            _enum(self.coordination_state, CoordinationState, "coordination_state"),
        )
        object.__setattr__(
            self, "mode", _enum(self.mode, ProviderExecutionMode, "mode")
        )
        for flag_name, expected in (
            ("authorizes_usage", GATEWAY_AUTHORIZES_USAGE),
            ("rewrites_provider_settlement", GATEWAY_REWRITES_PROVIDER_SETTLEMENT),
            ("is_completion_evidence", GATEWAY_IS_COMPLETION_EVIDENCE),
            ("is_correctness_evidence", GATEWAY_IS_CORRECTNESS_EVIDENCE),
        ):
            value = getattr(self, flag_name)
            if not isinstance(value, bool):
                _fail(f"{flag_name} must be boolean", reason_codes=("invalid_field",))
            if value is not expected:
                _fail(
                    f"{flag_name} cannot be true; execution results are operational only",
                    reason_codes=("authority_boundary",),
                )
        _reject_forbidden_payload(self._payload())

    @property
    def result_id(self) -> str:
        return self.content_id

    @property
    def success(self) -> bool:
        return self.phase in {
            ProviderExecutionPhase.SETTLED,
            ProviderExecutionPhase.DEGRADED,
            ProviderExecutionPhase.REPLAYED,
        } and self.final_status in {
            SupervisorUsageFinalStatus.COMMITTED,
            SupervisorUsageFinalStatus.RELEASED,
            SupervisorUsageFinalStatus.CANCELLED,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_EXECUTION_CONTRACT_VERSION,
            "phase": self.phase.value,
            "final_status": self.final_status.value,
            "granted": self.granted,
            "reservation_id": self.reservation_id,
            "usage_revision": self.usage_revision,
            "catalog_revision": self.catalog_revision,
            "provider_id": self.provider_id,
            "redacted_endpoint": self.redacted_endpoint,
            "attempt_key": self.attempt_key,
            "request_key": self.request_key,
            "request_id": self.request_id,
            "reason_codes": self.reason_codes,
            "observation": dict(self.observation),
            "settled": self.settled.to_dict(),
            "receipt": self.receipt.to_record() if self.receipt is not None else None,
            "endpoint_receipt_id": self.endpoint_receipt_id,
            "supervisor_receipt_id": self.supervisor_receipt_id,
            "attribution": (
                self.attribution.to_record() if self.attribution is not None else None
            ),
            "replayed": self.replayed,
            "coordination_state": self.coordination_state.value,
            "mode": self.mode.value,
            "authorizes_usage": False,
            "rewrites_provider_settlement": False,
            "is_completion_evidence": False,
            "is_correctness_evidence": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderExecutionResult":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "phase",
            "final_status",
            "granted",
            "reservation_id",
            "usage_revision",
            "catalog_revision",
            "provider_id",
            "redacted_endpoint",
            "attempt_key",
            "request_key",
            "request_id",
            "reason_codes",
            "observation",
            "settled",
            "receipt",
            "endpoint_receipt_id",
            "supervisor_receipt_id",
            "attribution",
            "replayed",
            "coordination_state",
            "mode",
            "authorizes_usage",
            "rewrites_provider_settlement",
            "is_completion_evidence",
            "is_correctness_evidence",
            "content_id",
            "result_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=tuple(allowed),
            name="provider execution result",
        )
        result = cls(
            phase=payload.get("phase", ""),
            final_status=payload.get("final_status", ""),
            granted=bool(payload.get("granted", False)),
            reservation_id=payload.get("reservation_id", ""),
            usage_revision=payload.get("usage_revision", ""),
            catalog_revision=payload.get("catalog_revision", ""),
            provider_id=payload.get("provider_id", ""),
            redacted_endpoint=payload.get("redacted_endpoint", ""),
            attempt_key=payload.get("attempt_key", ""),
            request_key=payload.get("request_key", ""),
            request_id=payload.get("request_id", ""),
            reason_codes=payload.get("reason_codes", ()),
            observation=payload.get("observation", {}),
            settled=payload.get("settled", {}),
            receipt=payload.get("receipt"),
            endpoint_receipt_id=payload.get("endpoint_receipt_id", ""),
            supervisor_receipt_id=payload.get("supervisor_receipt_id", ""),
            attribution=payload.get("attribution"),
            replayed=bool(payload.get("replayed", False)),
            coordination_state=payload.get(
                "coordination_state", CoordinationState.AVAILABLE
            ),
            mode=payload.get("mode", ProviderExecutionMode.ENFORCE),
            authorizes_usage=bool(payload.get("authorizes_usage", False)),
            rewrites_provider_settlement=bool(
                payload.get("rewrites_provider_settlement", False)
            ),
            is_completion_evidence=bool(payload.get("is_completion_evidence", False)),
            is_correctness_evidence=bool(
                payload.get("is_correctness_evidence", False)
            ),
        )
        _claim(payload, result.content_id, "content_id", "result_id")
        return result


class ProviderInvoker(Protocol):
    """Typed adapter that performs the provider side-effect after reservation."""

    def __call__(self, request: ProviderExecutionRequest) -> Mapping[str, Any]:
        ...


class UsageCoordinatorLike(Protocol):
    """Minimal coordinator surface used by the gateway (injection boundary)."""

    def reserve(
        self,
        scope_id: str,
        requested: Any,
        *,
        request_id: str,
        attempt_id: str = "1",
        idempotency_key: str,
        owner_id: str,
        lease_id: Optional[str] = None,
        estimate: Any = None,
        expected_usage_revision: Optional[str] = None,
        ttl_ms: int = 30_000,
        caller_budget: Any = None,
        fence: Optional[int] = None,
    ) -> Any:
        ...

    def mark_dispatched(self, reservation_id: str) -> Any:
        ...

    def cancel(self, reservation_id: str, *, reason: str = "cancelled") -> Any:
        ...

    def release(self, reservation_id: str, *, reason: str = "released") -> Any:
        ...

    def commit(
        self,
        reservation_id: str,
        actual: Any = None,
        *,
        observation_id: Optional[str] = None,
        release_unused: bool = True,
    ) -> Any:
        ...


@dataclass
class _TerminalRecord:
    result: ProviderExecutionResult
    finished_at: float
    invoke_count: int = 0


class ProviderExecutionGateway:
    """Atomic estimate → reserve → invoke → settle gateway.

    Exact attempt replay returns the terminal result without reinvoke/recharge.
    Inject a coordinator and invoker for live paths; omit them for hermetic
    offline simulation (simulated reserve only, no network).
    """

    requirement_id = RESERVATION_AWARE_PROVIDER_EXECUTION_REQUIREMENT_ID

    def __init__(
        self,
        *,
        coordinator: Optional[UsageCoordinatorLike] = None,
        invoker: Optional[ProviderInvoker] = None,
        owner_id: str = "supervisor-provider-execution",
        reservation_ttl_ms: int = 60_000,
        clock: Optional[Callable[[], float]] = None,
        single_flight_outcomes: Optional[MutableMapping[str, ProviderExecutionResult]] = None,
    ) -> None:
        self._coordinator = coordinator
        self._invoker = invoker
        self._owner_id = _text(owner_id, "owner_id")
        self._reservation_ttl_ms = max(1, int(reservation_ttl_ms))
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._terminals: MutableMapping[str, _TerminalRecord] = {}
        self._in_flight: set[str] = set()
        self._invoke_counts: MutableMapping[str, int] = {}
        self._single_flight = single_flight_outcomes if single_flight_outcomes is not None else {}

    @property
    def coordinator(self) -> Optional[UsageCoordinatorLike]:
        return self._coordinator

    def invoke_count(self, attempt_key: str) -> int:
        with self._lock:
            return int(self._invoke_counts.get(attempt_key, 0))

    def execute(self, request: ProviderExecutionRequest) -> ProviderExecutionResult:
        if not isinstance(request, ProviderExecutionRequest):
            _fail(
                "request must be ProviderExecutionRequest",
                reason_codes=("invalid_request",),
            )
        attempt_key = request.attempt_key
        request_key = request.request_key

        with self._lock:
            # Cache/batch/single-flight: identical request keys share outcome
            # metadata so remote charge cannot be duplicated.
            if request_key in self._single_flight:
                cached = self._single_flight[request_key]
                return self._replay_result(cached, extra_reasons=("single_flight",))
            prior = self._terminals.get(attempt_key)
            if prior is not None:
                return self._replay_result(prior.result, extra_reasons=("exact_replay",))
            if attempt_key in self._in_flight:
                raise ProviderExecutionError(
                    "attempt already in flight",
                    reason_codes=("attempt_in_flight",),
                    retryable=True,
                )
            self._in_flight.add(attempt_key)

        try:
            result = self._execute_unlocked(request)
            with self._lock:
                self._single_flight[request_key] = result
            return result
        finally:
            with self._lock:
                self._in_flight.discard(attempt_key)

    def _replay_result(
        self,
        prior: ProviderExecutionResult,
        *,
        extra_reasons: Sequence[str],
    ) -> ProviderExecutionResult:
        return ProviderExecutionResult(
            phase=ProviderExecutionPhase.REPLAYED
            if prior.phase
            not in {
                ProviderExecutionPhase.DENIED,
                ProviderExecutionPhase.CANCELLED,
                ProviderExecutionPhase.FAILED,
            }
            else prior.phase,
            final_status=prior.final_status,
            granted=prior.granted,
            reservation_id=prior.reservation_id,
            usage_revision=prior.usage_revision,
            catalog_revision=prior.catalog_revision,
            provider_id=prior.provider_id,
            redacted_endpoint=prior.redacted_endpoint,
            attempt_key=prior.attempt_key,
            request_key=prior.request_key,
            request_id=prior.request_id,
            reason_codes=tuple(prior.reason_codes) + tuple(extra_reasons),
            observation=prior.observation,
            settled=prior.settled,
            receipt=prior.receipt,
            endpoint_receipt_id=prior.endpoint_receipt_id,
            supervisor_receipt_id=prior.supervisor_receipt_id,
            attribution=prior.attribution,
            replayed=True,
            coordination_state=prior.coordination_state,
            mode=prior.mode,
        )

    def _execute_unlocked(
        self, request: ProviderExecutionRequest
    ) -> ProviderExecutionResult:
        mode = request.mode
        coordination = self._resolve_coordination(request)

        if mode is ProviderExecutionMode.OFF:
            return self._off_mode_execute(request)

        if coordination in {
            CoordinationState.UNKNOWN,
            CoordinationState.STALE,
            CoordinationState.UNAVAILABLE,
        }:
            if mode is ProviderExecutionMode.ENFORCE:
                if not request.degraded_budget_id:
                    return self._terminal(
                        request,
                        phase=ProviderExecutionPhase.DENIED,
                        final_status=SupervisorUsageFinalStatus.CAPACITY_UNAVAILABLE,
                        granted=False,
                        reason_codes=(
                            "coordination_fail_closed",
                            coordination.value,
                        ),
                        coordination_state=coordination,
                    )
                return self._degraded_local_execute(request, coordination)
            # observe/shadow/assist: continue with simulated coordination.
            coordination = CoordinationState.SIMULATED

        if request.cancelled and not request.post_dispatch:
            return self._terminal(
                request,
                phase=ProviderExecutionPhase.CANCELLED,
                final_status=SupervisorUsageFinalStatus.CANCELLED,
                granted=False,
                reason_codes=("pre_dispatch_cancelled",),
                coordination_state=coordination,
            )

        estimate = request.effective_estimate()
        decision = self._reserve(request, estimate, coordination)
        if not decision.get("granted"):
            return self._terminal(
                request,
                phase=ProviderExecutionPhase.DENIED,
                final_status=SupervisorUsageFinalStatus.CAPACITY_UNAVAILABLE
                if "capacity" in " ".join(decision.get("reason_codes") or ())
                else SupervisorUsageFinalStatus.REJECTED,
                granted=False,
                reservation_id=str(decision.get("reservation_id") or ""),
                usage_revision=str(
                    decision.get("usage_revision") or request.bridge.usage_revision
                ),
                reason_codes=tuple(decision.get("reason_codes") or ("capacity_denied",)),
                coordination_state=coordination,
            )

        reservation_id = str(decision.get("reservation_id") or "")
        usage_revision = str(
            decision.get("usage_revision") or request.bridge.usage_revision
        )

        # Pre-dispatch cancel after reserve: full release.
        if request.cancelled and not request.post_dispatch:
            self._release_or_cancel(
                reservation_id,
                reason="pre_dispatch_cancelled",
                post_dispatch=False,
                coordination=coordination,
            )
            receipt = self._build_receipt(
                request,
                reservation_id=reservation_id,
                usage_revision=usage_revision,
                units=UsageVector(),
                final_status=SupervisorUsageFinalStatus.RELEASED,
                event_ids=(),
            )
            return self._terminal(
                request,
                phase=ProviderExecutionPhase.RELEASED,
                final_status=SupervisorUsageFinalStatus.RELEASED,
                granted=True,
                reservation_id=reservation_id,
                usage_revision=usage_revision,
                reason_codes=("pre_dispatch_cancelled", "released"),
                receipt=receipt,
                coordination_state=coordination,
            )

        dispatched = self._mark_dispatched(reservation_id, coordination)

        # Post-dispatch cancel/timeout: conservative settle (provider may charge).
        if request.cancelled or request.timeout_expired:
            settlement = self._conservative_settle(
                reservation_id,
                estimate=estimate,
                reason=(
                    "post_dispatch_timeout"
                    if request.timeout_expired
                    else "post_dispatch_cancelled"
                ),
                coordination=coordination,
            )
            units = settlement.get("charged") or estimate.requested
            event_ids = settlement.get("event_ids") or ()
            receipt = self._build_receipt(
                request,
                reservation_id=reservation_id,
                usage_revision=str(
                    settlement.get("usage_revision") or usage_revision
                ),
                units=units,
                final_status=SupervisorUsageFinalStatus.COMMITTED,
                event_ids=tuple(event_ids),
            )
            return self._terminal(
                request,
                phase=ProviderExecutionPhase.SETTLED,
                final_status=SupervisorUsageFinalStatus.COMMITTED,
                granted=True,
                reservation_id=reservation_id,
                usage_revision=str(
                    settlement.get("usage_revision") or usage_revision
                ),
                reason_codes=(
                    "post_dispatch_timeout"
                    if request.timeout_expired
                    else "post_dispatch_cancelled",
                    "conservative_settle",
                    "dispatched" if dispatched else "dispatch_mark_skipped",
                ),
                settled=units,
                receipt=receipt,
                coordination_state=coordination,
            )

        try:
            observation = self._invoke(request)
        except Exception as exc:
            # Unknown side effects after dispatch: cancel conservatively.
            settlement = self._conservative_settle(
                reservation_id,
                estimate=estimate,
                reason="invoke_failed",
                coordination=coordination,
            )
            units = settlement.get("charged") or UsageVector()
            return self._terminal(
                request,
                phase=ProviderExecutionPhase.FAILED,
                final_status=SupervisorUsageFinalStatus.FAILED,
                granted=True,
                reservation_id=reservation_id,
                usage_revision=str(
                    settlement.get("usage_revision") or usage_revision
                ),
                reason_codes=("invoke_failed", type(exc).__name__.casefold()),
                observation={"error_class": type(exc).__name__},
                settled=units,
                coordination_state=coordination,
            )

        settled_units = self._observation_units(observation, estimate)
        settlement = self._commit(
            reservation_id,
            units=settled_units,
            observation_id=str(observation.get("observation_id") or "") or None,
            coordination=coordination,
        )
        usage_revision = str(settlement.get("usage_revision") or usage_revision)
        event_ids = tuple(settlement.get("event_ids") or ())
        if settlement.get("charged") is not None:
            settled_units = settlement["charged"]

        receipt = self._build_receipt(
            request,
            reservation_id=reservation_id,
            usage_revision=usage_revision,
            units=settled_units,
            final_status=SupervisorUsageFinalStatus.COMMITTED,
            event_ids=event_ids,
        )
        attribution = self._attribute(request, receipt, settled_units, event_ids)
        redacted = _redact_endpoint(
            str(
                observation.get("endpoint_scope_id")
                or request.bridge.endpoint_scope_id
            )
        )
        endpoint_receipt_id = str(
            observation.get("endpoint_receipt_id")
            or observation.get("routing_receipt_id")
            or (event_ids[0] if event_ids else "")
        )
        return self._terminal(
            request,
            phase=ProviderExecutionPhase.SETTLED,
            final_status=SupervisorUsageFinalStatus.COMMITTED,
            granted=True,
            reservation_id=reservation_id,
            usage_revision=usage_revision,
            reason_codes=("settled", "dispatched" if dispatched else "dispatch_mark_skipped"),
            observation=observation,
            settled=settled_units,
            receipt=receipt,
            attribution=attribution,
            redacted_endpoint=redacted,
            endpoint_receipt_id=endpoint_receipt_id,
            coordination_state=coordination,
        )

    def _resolve_coordination(
        self, request: ProviderExecutionRequest
    ) -> CoordinationState:
        if request.coordination_state is not CoordinationState.AVAILABLE:
            return request.coordination_state
        if self._coordinator is None:
            return CoordinationState.SIMULATED
        return CoordinationState.AVAILABLE

    def _off_mode_execute(
        self, request: ProviderExecutionRequest
    ) -> ProviderExecutionResult:
        """Legacy-compatible path: invoke without reservation/settlement."""

        if request.cancelled:
            return self._terminal(
                request,
                phase=ProviderExecutionPhase.CANCELLED,
                final_status=SupervisorUsageFinalStatus.CANCELLED,
                granted=False,
                reason_codes=("off_mode", "cancelled"),
                coordination_state=CoordinationState.SIMULATED,
            )
        try:
            observation = self._invoke(request)
        except Exception as exc:
            return self._terminal(
                request,
                phase=ProviderExecutionPhase.FAILED,
                final_status=SupervisorUsageFinalStatus.FAILED,
                granted=False,
                reason_codes=("off_mode", "invoke_failed", type(exc).__name__.casefold()),
                observation={"error_class": type(exc).__name__},
                coordination_state=CoordinationState.SIMULATED,
            )
        return self._terminal(
            request,
            phase=ProviderExecutionPhase.SETTLED,
            final_status=SupervisorUsageFinalStatus.UNKNOWN,
            granted=True,
            reason_codes=("off_mode", "invoked_without_reservation"),
            observation=observation,
            settled=UsageVector(),
            coordination_state=CoordinationState.SIMULATED,
        )

    def _degraded_local_execute(
        self,
        request: ProviderExecutionRequest,
        coordination: CoordinationState,
    ) -> ProviderExecutionResult:
        if not request.degraded_budget_id:
            return self._terminal(
                request,
                phase=ProviderExecutionPhase.DENIED,
                final_status=SupervisorUsageFinalStatus.CAPACITY_UNAVAILABLE,
                granted=False,
                reason_codes=("degraded_budget_required", coordination.value),
                coordination_state=coordination,
            )
        try:
            observation = self._invoke(request)
        except Exception as exc:
            return self._terminal(
                request,
                phase=ProviderExecutionPhase.FAILED,
                final_status=SupervisorUsageFinalStatus.FAILED,
                granted=False,
                reason_codes=(
                    "degraded_local_fallback",
                    "invoke_failed",
                    type(exc).__name__.casefold(),
                ),
                observation={"error_class": type(exc).__name__},
                coordination_state=coordination,
            )
        receipt = self._build_receipt(
            request,
            reservation_id=f"degraded:{request.attempt_key}",
            usage_revision=request.bridge.usage_revision,
            units=UsageVector(),
            final_status=SupervisorUsageFinalStatus.COMMITTED,
            event_ids=(f"degraded-event:{request.attempt_key}",),
        )
        return self._terminal(
            request,
            phase=ProviderExecutionPhase.DEGRADED,
            final_status=SupervisorUsageFinalStatus.COMMITTED,
            granted=True,
            reservation_id=f"degraded:{request.attempt_key}",
            usage_revision=request.bridge.usage_revision,
            reason_codes=(
                "degraded_local_fallback",
                "reviewed_degraded_budget",
                coordination.value,
            ),
            observation=observation,
            settled=UsageVector(),
            receipt=receipt,
            coordination_state=coordination,
        )

    def _reserve(
        self,
        request: ProviderExecutionRequest,
        estimate: UsageEstimate,
        coordination: CoordinationState,
    ) -> dict[str, Any]:
        if self._coordinator is None or coordination is CoordinationState.SIMULATED:
            return {
                "granted": True,
                "reservation_id": f"sim:{request.attempt_key}",
                "usage_revision": request.bridge.usage_revision,
                "reason_codes": ("simulated_reserve",),
            }
        fence_value: Optional[int] = None
        try:
            fence_value = int(request.bridge.fence_id)
        except (TypeError, ValueError):
            fence_value = None
        try:
            decision = self._coordinator.reserve(
                request.bridge.endpoint_scope_id,
                estimate.requested,
                request_id=request.bridge.request_id,
                attempt_id=str(request.bridge.attempt),
                idempotency_key=request.bridge.idempotency_key,
                owner_id=self._owner_id,
                lease_id=request.bridge.lease_id or None,
                estimate=estimate,
                expected_usage_revision=request.bridge.usage_revision,
                ttl_ms=self._reservation_ttl_ms,
                caller_budget=estimate.requested,
                fence=fence_value,
            )
        except Exception as exc:
            name = type(exc).__name__.casefold()
            if "stale" in name or "stale" in str(exc).casefold():
                raise ProviderExecutionError(
                    f"stale coordination: {exc}",
                    reason_codes=("stale_coordination", name),
                    retryable=True,
                ) from exc
            return {
                "granted": False,
                "reservation_id": "",
                "usage_revision": request.bridge.usage_revision,
                "reason_codes": ("reserve_failed", name),
            }
        granted = bool(getattr(decision, "granted", False))
        return {
            "granted": granted,
            "reservation_id": str(
                getattr(decision, "reservation_id", None) or ""
            ),
            "usage_revision": str(
                getattr(decision, "usage_revision", None)
                or request.bridge.usage_revision
            ),
            "reason_codes": tuple(getattr(decision, "reason_codes", ()) or ()),
        }

    def _mark_dispatched(
        self, reservation_id: str, coordination: CoordinationState
    ) -> bool:
        if (
            self._coordinator is None
            or not reservation_id
            or reservation_id.startswith(("sim:", "degraded:"))
            or coordination is CoordinationState.SIMULATED
        ):
            return False
        try:
            self._coordinator.mark_dispatched(reservation_id)
            return True
        except Exception:
            return False

    def _release_or_cancel(
        self,
        reservation_id: str,
        *,
        reason: str,
        post_dispatch: bool,
        coordination: CoordinationState,
    ) -> dict[str, Any]:
        if (
            self._coordinator is None
            or not reservation_id
            or reservation_id.startswith(("sim:", "degraded:"))
            or coordination is CoordinationState.SIMULATED
        ):
            return {
                "charged": UsageVector(),
                "event_ids": (),
                "usage_revision": "",
                "state": "released",
            }
        try:
            if post_dispatch:
                settlement = self._coordinator.cancel(
                    reservation_id, reason=reason
                )
            else:
                try:
                    settlement = self._coordinator.release(
                        reservation_id, reason=reason
                    )
                except Exception:
                    settlement = self._coordinator.cancel(
                        reservation_id, reason=reason
                    )
            return {
                "charged": getattr(settlement, "charged", UsageVector())
                or UsageVector(),
                "event_ids": (
                    (str(getattr(settlement, "event_id", "")),)
                    if getattr(settlement, "event_id", None)
                    else ()
                ),
                "usage_revision": str(
                    getattr(settlement, "usage_revision", "") or ""
                ),
                "state": str(
                    getattr(getattr(settlement, "state", None), "value", "")
                    or getattr(settlement, "state", "")
                    or ""
                ),
            }
        except Exception:
            return {
                "charged": UsageVector(),
                "event_ids": (),
                "usage_revision": "",
                "state": "failed",
            }

    def _conservative_settle(
        self,
        reservation_id: str,
        *,
        estimate: UsageEstimate,
        reason: str,
        coordination: CoordinationState,
    ) -> dict[str, Any]:
        """Post-dispatch cancel/timeout: settle reserved/chargeable amounts."""

        if (
            self._coordinator is None
            or not reservation_id
            or reservation_id.startswith(("sim:", "degraded:"))
            or coordination is CoordinationState.SIMULATED
        ):
            return {
                "charged": estimate.requested,
                "event_ids": (f"sim-event:{reservation_id}:{reason}",),
                "usage_revision": "",
            }
        try:
            settlement = self._coordinator.cancel(reservation_id, reason=reason)
            charged = getattr(settlement, "charged", None) or estimate.requested
            return {
                "charged": charged if isinstance(charged, UsageVector) else _usage_vector(charged),
                "event_ids": (
                    (str(getattr(settlement, "event_id", "")),)
                    if getattr(settlement, "event_id", None)
                    else ()
                ),
                "usage_revision": str(
                    getattr(settlement, "usage_revision", "") or ""
                ),
            }
        except Exception:
            try:
                settlement = self._coordinator.commit(
                    reservation_id, actual=estimate.requested
                )
                charged = getattr(settlement, "charged", None) or estimate.requested
                return {
                    "charged": charged
                    if isinstance(charged, UsageVector)
                    else _usage_vector(charged),
                    "event_ids": (
                        (str(getattr(settlement, "event_id", "")),)
                        if getattr(settlement, "event_id", None)
                        else ()
                    ),
                    "usage_revision": str(
                        getattr(settlement, "usage_revision", "") or ""
                    ),
                }
            except Exception:
                return {
                    "charged": estimate.requested,
                    "event_ids": (),
                    "usage_revision": "",
                }

    def _commit(
        self,
        reservation_id: str,
        *,
        units: UsageVector,
        observation_id: Optional[str],
        coordination: CoordinationState,
    ) -> dict[str, Any]:
        if (
            self._coordinator is None
            or not reservation_id
            or reservation_id.startswith(("sim:", "degraded:"))
            or coordination is CoordinationState.SIMULATED
        ):
            return {
                "charged": units,
                "event_ids": (f"sim-event:{reservation_id}",),
                "usage_revision": "",
            }
        try:
            settlement = self._coordinator.commit(
                reservation_id,
                actual=units,
                observation_id=observation_id,
            )
            charged = getattr(settlement, "charged", None) or units
            return {
                "charged": charged
                if isinstance(charged, UsageVector)
                else _usage_vector(charged),
                "event_ids": (
                    (str(getattr(settlement, "event_id", "")),)
                    if getattr(settlement, "event_id", None)
                    else ()
                ),
                "usage_revision": str(
                    getattr(settlement, "usage_revision", "") or ""
                ),
            }
        except Exception:
            # Residual hold must not leak.
            try:
                self._coordinator.release(reservation_id, reason="settle_fallback")
            except Exception:
                pass
            return {
                "charged": units,
                "event_ids": (),
                "usage_revision": "",
            }

    def _invoke(self, request: ProviderExecutionRequest) -> dict[str, Any]:
        with self._lock:
            self._invoke_counts[request.attempt_key] = (
                int(self._invoke_counts.get(request.attempt_key, 0)) + 1
            )
        if self._invoker is None:
            return {
                "provider_id": request.provider_id,
                "endpoint_scope_id": request.bridge.endpoint_scope_id,
                "status": "simulated_ok",
                "units": request.effective_estimate().requested.to_dict(),
            }
        raw = self._invoker(request)
        if not isinstance(raw, Mapping):
            _fail(
                "invoker must return a mapping observation",
                reason_codes=("invalid_observation",),
            )
        return _sanitize_observation(raw)

    def _observation_units(
        self,
        observation: Mapping[str, Any],
        estimate: UsageEstimate,
    ) -> UsageVector:
        units = observation.get("units")
        if units is None:
            return estimate.requested
        return _usage_vector(units)

    def _build_receipt(
        self,
        request: ProviderExecutionRequest,
        *,
        reservation_id: str,
        usage_revision: str,
        units: UsageVector,
        final_status: SupervisorUsageFinalStatus,
        event_ids: Sequence[str],
    ) -> SupervisorUsageReceipt:
        ids = tuple(
            _text(item, "endpoint_event_id")
            for item in event_ids
            if str(item or "").strip()
        )
        if not ids:
            # Receipts require a reservation identity; synthetic when simulated.
            ids = (f"event:{reservation_id or request.attempt_key}",)
        # SupervisorUsageReceipt rejects empty endpoint_event_ids? No - empty OK.
        # But reservation_id must be non-empty text.
        # Receipt identity is bound to the request-time scope revisions. The
        # coordinator may advance usage_revision after settlement; that value
        # is retained on the gateway result, not by mutating the scope binding.
        bound_usage_revision = request.bridge.usage_revision
        return SupervisorUsageReceipt(
            scope=request.bridge.scope,
            envelope_id=request.bridge.envelope_id,
            request_id=request.bridge.request_id,
            endpoint_scope_id=request.bridge.endpoint_scope_id,
            catalog_revision=request.bridge.catalog_revision,
            usage_revision=bound_usage_revision,
            reservation_id=reservation_id or f"none:{request.attempt_key}",
            endpoint_event_ids=ids,
            settled=units,
            final_status=final_status,
        )

    def _attribute(
        self,
        request: ProviderExecutionRequest,
        receipt: Optional[SupervisorUsageReceipt],
        settled: UsageVector,
        event_ids: Sequence[str],
    ) -> Optional[SupervisorUsageAttribution]:
        if receipt is None or not event_ids:
            return None
        try:
            # Build synthetic events for exact-once attribution when coordinator
            # did not surface full UsageEvent objects.
            from ipfs_accelerate_py.endpoint_usage.schema import (
                UsageEvent,
                UsageEventKind,
            )

            events = []
            for index, event_id in enumerate(event_ids, start=1):
                events.append(
                    UsageEvent(
                        kind=UsageEventKind.COMMIT,
                        scope_id=request.bridge.endpoint_scope_id,
                        request_id=request.bridge.request_id,
                        sequence=index,
                        occurred_at=request.bridge.deadline_at,
                        units=settled,
                        reservation_id=receipt.reservation_id,
                        event_id=event_id,
                    )
                )
            # Ensure event_ids present (schema may auto-generate if empty).
            normalized = []
            for event, event_id in zip(events, event_ids):
                payload = event.to_dict()
                payload["event_id"] = event_id
                normalized.append(UsageEvent.from_dict(payload))
            attributions = attribute_endpoint_events(
                scope=request.bridge.scope,
                events=normalized,
                lifecycle_event_ids=tuple(
                    f"lifecycle:{request.attempt_key}:{i}"
                    for i in range(len(normalized))
                ),
            )
            return attributions[0] if attributions else None
        except Exception:
            return None

    def _terminal(
        self,
        request: ProviderExecutionRequest,
        *,
        phase: ProviderExecutionPhase,
        final_status: SupervisorUsageFinalStatus,
        granted: bool,
        reason_codes: Sequence[str],
        reservation_id: str = "",
        usage_revision: str = "",
        observation: Optional[Mapping[str, Any]] = None,
        settled: Optional[UsageVector] = None,
        receipt: Optional[SupervisorUsageReceipt] = None,
        attribution: Optional[SupervisorUsageAttribution] = None,
        redacted_endpoint: str = "",
        endpoint_receipt_id: str = "",
        coordination_state: CoordinationState = CoordinationState.AVAILABLE,
    ) -> ProviderExecutionResult:
        result = ProviderExecutionResult(
            phase=phase,
            final_status=final_status,
            granted=granted,
            reservation_id=reservation_id,
            usage_revision=usage_revision or request.bridge.usage_revision,
            catalog_revision=request.bridge.catalog_revision,
            provider_id=request.provider_id,
            redacted_endpoint=redacted_endpoint
            or _redact_endpoint(request.bridge.endpoint_scope_id),
            attempt_key=request.attempt_key,
            request_key=request.request_key,
            request_id=request.bridge.request_id,
            reason_codes=tuple(reason_codes),
            observation=dict(observation or {}),
            settled=settled if settled is not None else UsageVector(),
            receipt=receipt,
            endpoint_receipt_id=endpoint_receipt_id,
            supervisor_receipt_id=receipt.receipt_id if receipt is not None else "",
            attribution=attribution,
            replayed=False,
            coordination_state=coordination_state,
            mode=request.mode,
        )
        with self._lock:
            self._terminals[request.attempt_key] = _TerminalRecord(
                result=result,
                finished_at=float(self._clock()),
                invoke_count=int(self._invoke_counts.get(request.attempt_key, 0)),
            )
        return result


def new_attempt_idempotency_key(base_key: str, attempt: int) -> str:
    """Derive a fresh idempotency key for retry/fallback attempts."""

    base = _text(base_key, "base_key")
    attempt_value = _integer(attempt, "attempt", minimum=1, maximum=100_000)
    return f"{base}#attempt-{attempt_value}"


def build_execution_request(
    *,
    bridge: SupervisorToEndpointRequest,
    envelope: SupervisorUsageEnvelope,
    provider_id: str,
    modality: str = "text",
    side_effect_boundary: SideEffectBoundary | str = SideEffectBoundary.IDEMPOTENT,
    operation: str = "text.generate",
    mode: ProviderExecutionMode | str = ProviderExecutionMode.ENFORCE,
    cancelled: bool = False,
    post_dispatch: bool = False,
    timeout_expired: bool = False,
    degraded_budget_id: str = "",
    coordination_state: CoordinationState | str = CoordinationState.AVAILABLE,
    metadata: Optional[Mapping[str, Any]] = None,
    estimate: Optional[UsageEstimate] = None,
) -> ProviderExecutionRequest:
    """Convenience constructor used by callers and tests."""

    return ProviderExecutionRequest(
        bridge=bridge,
        envelope=envelope,
        provider_id=provider_id,
        modality=modality,
        side_effect_boundary=side_effect_boundary,
        operation=operation,
        mode=mode,
        cancelled=cancelled,
        post_dispatch=post_dispatch,
        timeout_expired=timeout_expired,
        degraded_budget_id=degraded_budget_id,
        coordination_state=coordination_state,
        metadata=dict(metadata or {}),
        estimate=estimate,
    )


def discover_schemas() -> dict[str, str]:
    """Provider-free schema discovery for the execution gateway."""

    return {
        "requirement_id": RESERVATION_AWARE_PROVIDER_EXECUTION_REQUIREMENT_ID,
        "goal_id": PROVIDER_EXECUTION_GOAL_ID,
        "contract_version": str(PROVIDER_EXECUTION_CONTRACT_VERSION),
        "request": PROVIDER_EXECUTION_REQUEST_SCHEMA,
        "result": PROVIDER_EXECUTION_RESULT_SCHEMA,
        "observation": PROVIDER_EXECUTION_OBSERVATION_SCHEMA,
        "authorizes_usage": str(GATEWAY_AUTHORIZES_USAGE).lower(),
        "rewrites_provider_settlement": str(
            GATEWAY_REWRITES_PROVIDER_SETTLEMENT
        ).lower(),
        "is_completion_evidence": str(GATEWAY_IS_COMPLETION_EVIDENCE).lower(),
        "is_correctness_evidence": str(GATEWAY_IS_CORRECTNESS_EVIDENCE).lower(),
        "bridge_authorizes_usage": str(BRIDGE_AUTHORIZES_USAGE).lower(),
        "bridge_is_completion_evidence": str(BRIDGE_IS_COMPLETION_EVIDENCE).lower(),
        "bridge_is_correctness_evidence": str(BRIDGE_IS_CORRECTNESS_EVIDENCE).lower(),
        "bridge_rewrites_provider_settlement": str(
            BRIDGE_REWRITES_PROVIDER_SETTLEMENT
        ).lower(),
    }


def accounting_bounds() -> dict[str, bool]:
    """Explicit non-authority bounds for gateway consumers."""

    return {
        "authorizes_usage": GATEWAY_AUTHORIZES_USAGE,
        "rewrites_provider_settlement": GATEWAY_REWRITES_PROVIDER_SETTLEMENT,
        "is_completion_evidence": GATEWAY_IS_COMPLETION_EVIDENCE,
        "is_correctness_evidence": GATEWAY_IS_CORRECTNESS_EVIDENCE,
    }


__all__ = [
    "GATEWAY_AUTHORIZES_USAGE",
    "GATEWAY_IS_COMPLETION_EVIDENCE",
    "GATEWAY_IS_CORRECTNESS_EVIDENCE",
    "GATEWAY_REWRITES_PROVIDER_SETTLEMENT",
    "PROVIDER_EXECUTION_CONTRACT_VERSION",
    "PROVIDER_EXECUTION_GOAL_ID",
    "PROVIDER_EXECUTION_OBSERVATION_SCHEMA",
    "PROVIDER_EXECUTION_REQUEST_SCHEMA",
    "PROVIDER_EXECUTION_RESULT_SCHEMA",
    "RESERVATION_AWARE_PROVIDER_EXECUTION_REQUIREMENT_ID",
    "SCHEMA_VERSION",
    "CoordinationState",
    "ProviderExecutionError",
    "ProviderExecutionGateway",
    "ProviderExecutionMode",
    "ProviderExecutionPhase",
    "ProviderExecutionRequest",
    "ProviderExecutionResult",
    "ProviderInvoker",
    "SideEffectBoundary",
    "UsageCoordinatorLike",
    "accounting_bounds",
    "build_execution_request",
    "conservative_estimate",
    "discover_schemas",
    "new_attempt_idempotency_key",
]
