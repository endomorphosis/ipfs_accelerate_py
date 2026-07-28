"""Normalize configured and provider-observed usage metadata.

Adapters parse only metadata from the exact local invocation (headers, structured
usage/error bodies, CLI reset metadata, or local capacity signals). They never
store raw payloads, never probe the network, and never raise a policy ceiling
from an untrusted observation. A valid restrictive cooldown is retained even
when unrelated field parsing fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .identity import (
    content_cid,
    is_secret_key,
    is_secret_value,
    redact_secrets,
)
from .provider_registry import (
    ADAPTER_PARSER_VERSION,
    PROVIDER_USAGE_ADAPTER_REQUIREMENT_ID,
    AdapterError,
    AdapterFamily,
    resolve_adapter_family,
)
from .schema import (
    MAX_ABS_INTEGER,
    MAX_LIMITS,
    MAX_REASON_CODES,
    MAX_STRING_BYTES,
    MAX_VECTOR_ENTRIES,
    MAX_WINDOW_MS,
    ConfidenceLevel,
    EndpointUsageScope,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    ProviderUsageObservation,
    Provenance,
    Quantity,
    QuantityKind,
    SchemaValidationError,
    UsageDimension,
    UsageLimit,
    UsageVector,
    UsageVectorEntry,
    WindowKind,
)

# ---------------------------------------------------------------------------
# Bounds (fail-closed clamps)
# ---------------------------------------------------------------------------

MAX_HEADERS = 64
MAX_HEADER_NAME_BYTES = 128
MAX_HEADER_VALUE_BYTES = 512
MAX_BODY_KEYS = 64
MAX_NESTING_DEPTH = 8
MAX_CLI_KEYS = 32
MAX_LOCAL_KEYS = 32
MAX_REASON_CODE_INPUT = MAX_REASON_CODES
MAX_CLOCK_SKEW_PAST_MS = 86_400_000  # 24h
MAX_RESET_FUTURE_MS = MAX_WINDOW_MS
DEFAULT_RETRY_MS = 60_000
MIN_RETRY_MS = 0
RESET_CONFLICT_TOLERANCE_MS = 2_000
DEFAULT_REQUEST_WINDOW_MS = 60_000
DEFAULT_TOKEN_WINDOW_MS = 60_000

_REASON = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_HTTP_DATE = re.compile(r"^[A-Za-z]{3},")

# OpenAI-compatible rate-limit header names (case-insensitive match).
_OPENAI_HEADER_MAP = {
    "x-ratelimit-limit-requests": ("requests", "limit"),
    "x-ratelimit-remaining-requests": ("requests", "remaining"),
    "x-ratelimit-reset-requests": ("requests", "reset"),
    "x-ratelimit-limit-tokens": ("total_tokens", "limit"),
    "x-ratelimit-remaining-tokens": ("total_tokens", "remaining"),
    "x-ratelimit-reset-tokens": ("total_tokens", "reset"),
    "x-request-id": ("_meta", "request_id"),
    "x-openai-request-id": ("_meta", "request_id"),
    "request-id": ("_meta", "request_id"),
    "retry-after": ("_meta", "retry_after"),
}

_ANTHROPIC_HEADER_MAP = {
    "anthropic-ratelimit-requests-limit": ("requests", "limit"),
    "anthropic-ratelimit-requests-remaining": ("requests", "remaining"),
    "anthropic-ratelimit-requests-reset": ("requests", "reset"),
    "anthropic-ratelimit-tokens-limit": ("total_tokens", "limit"),
    "anthropic-ratelimit-tokens-remaining": ("total_tokens", "remaining"),
    "anthropic-ratelimit-tokens-reset": ("total_tokens", "reset"),
    "anthropic-ratelimit-input-tokens-limit": ("input_tokens", "limit"),
    "anthropic-ratelimit-input-tokens-remaining": ("input_tokens", "remaining"),
    "anthropic-ratelimit-input-tokens-reset": ("input_tokens", "reset"),
    "anthropic-ratelimit-output-tokens-limit": ("output_tokens", "limit"),
    "anthropic-ratelimit-output-tokens-remaining": ("output_tokens", "remaining"),
    "anthropic-ratelimit-output-tokens-reset": ("output_tokens", "reset"),
    "request-id": ("_meta", "request_id"),
    "x-request-id": ("_meta", "request_id"),
    "retry-after": ("_meta", "retry_after"),
}

_HF_HEADER_MAP = {
    "x-ratelimit-limit": ("requests", "limit"),
    "x-ratelimit-remaining": ("requests", "remaining"),
    "x-ratelimit-reset": ("requests", "reset"),
    "ratelimit-limit": ("requests", "limit"),
    "ratelimit-remaining": ("requests", "remaining"),
    "ratelimit-reset": ("requests", "reset"),
    "x-request-id": ("_meta", "request_id"),
    "request-id": ("_meta", "request_id"),
    "retry-after": ("_meta", "retry_after"),
}

_USAGE_BODY_KEYS = {
    "prompt_tokens": UsageDimension.INPUT_TOKENS,
    "input_tokens": UsageDimension.INPUT_TOKENS,
    "completion_tokens": UsageDimension.OUTPUT_TOKENS,
    "output_tokens": UsageDimension.OUTPUT_TOKENS,
    "total_tokens": UsageDimension.TOTAL_TOKENS,
    "embedding_tokens": UsageDimension.EMBEDDING_TOKENS,
    "prompt_tokens_details": None,  # nested ignored except known counters
    "completion_tokens_details": None,
    "cache_read_input_tokens": UsageDimension.INPUT_TOKENS,
    "cache_creation_input_tokens": UsageDimension.INPUT_TOKENS,
    "requests": UsageDimension.REQUESTS,
    "batch_items": UsageDimension.BATCH_ITEMS,
    "images": UsageDimension.IMAGES,
    "characters": UsageDimension.CHARACTERS,
    "audio_seconds": UsageDimension.AUDIO_SECONDS,
    "cost_micros": UsageDimension.COST_MICROS,
    "n_prompt_tokens": UsageDimension.INPUT_TOKENS,
    "n_completion_tokens": UsageDimension.OUTPUT_TOKENS,
    "n_tokens": UsageDimension.TOTAL_TOKENS,
}

_BILLING_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
    "quota has been exceeded",
    "billing_not_active",
    "billing hard limit",
    "billing limit",
    "check your plan and billing",
    "add a payment method",
    "payment required",
    "account is not active",
    "credit balance",
    "out of credits",
    "add more credits",
)

_USAGE_LIMIT_MARKERS = (
    "usage_limit",
    "usage limit",
    "rate_limit",
    "rate limit",
    "rate_limit_exceeded",
    "too many requests",
    "tokens per minute",
    "requests per minute",
    "quota exceeded",
    "capacity",
    "overloaded",
)

class AdapterParseError(AdapterError):
    """Input was rejected as negative, overflowing, credential-bearing, etc."""


class AdapterScopeError(AdapterParseError):
    """Observation is not bound to the expected endpoint usage scope."""


@dataclass
class _ParseState:
    """Mutable collector used while parsing; never exposed or persisted raw."""

    family: AdapterFamily
    scope: EndpointUsageScope
    request_id: str
    observed_at: datetime
    now: datetime
    reason_codes: List[str] = field(default_factory=list)
    usage_entries: List[UsageVectorEntry] = field(default_factory=list)
    limits: List[UsageLimit] = field(default_factory=list)
    retry_after_ms: Optional[int] = None
    reset_at: Optional[datetime] = None
    http_status: Optional[int] = None
    provider_request_id: Optional[str] = None
    sources: Set[str] = field(default_factory=set)
    restrictive: bool = False
    billing_exhausted: bool = False
    parse_failures: List[str] = field(default_factory=list)
    policy_ceilings: Dict[str, int] = field(default_factory=dict)
    configured_limits: Tuple[UsageLimit, ...] = ()
    header_buckets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    custom_field_map: Mapping[str, str] = field(default_factory=dict)

    def add_reason(self, code: str) -> None:
        normalized = _reason_code(code)
        if normalized and normalized not in self.reason_codes:
            if len(self.reason_codes) < MAX_REASON_CODE_INPUT:
                self.reason_codes.append(normalized)

    def note_failure(self, code: str) -> None:
        self.parse_failures.append(code)
        self.add_reason(code)


# ---------------------------------------------------------------------------
# Public configured-limit normalization
# ---------------------------------------------------------------------------


def normalize_configured_limits(
    scope: Union[EndpointUsageScope, Mapping[str, Any]],
    limits: Sequence[Any],
    *,
    source: LimitSource = LimitSource.CONFIGURED,
    observed_at: Optional[Union[str, datetime]] = None,
    parser_version: str = ADAPTER_PARSER_VERSION,
) -> Tuple[UsageLimit, ...]:
    """Normalize operator/configured limits into typed :class:`UsageLimit`s.

    Rejects credential-bearing material, negative/overflow values, invalid
    unit/window pairs, and scope mismatches. Does not contact providers.
    """

    scope_obj = _coerce_scope(scope)
    if not isinstance(limits, (list, tuple)):
        raise AdapterParseError("limits must be a sequence")
    if len(limits) > MAX_LIMITS:
        raise AdapterParseError("limits exceeds maximum count")

    source_enum = (
        source if isinstance(source, LimitSource) else LimitSource(str(source))
    )
    if source_enum not in (
        LimitSource.POLICY,
        LimitSource.CONFIGURED,
        LimitSource.RECONCILED,
    ):
        raise AdapterParseError(
            "configured limit source must be policy, configured, or reconciled"
        )
    observed = _coerce_timestamp(observed_at, "observed_at") or _utcnow()
    observed_text = _format_ts(observed)
    out: List[UsageLimit] = []
    for index, item in enumerate(limits):
        try:
            limit = _normalize_one_configured_limit(
                scope_obj,
                item,
                source=source_enum,
                observed_at=observed_text,
                parser_version=parser_version,
            )
        except (AdapterParseError, SchemaValidationError, AdapterError) as exc:
            raise AdapterParseError(
                "configured limit[%d] rejected: %s" % (index, exc)
            ) from exc
        out.append(limit)
    return tuple(out)


def _normalize_one_configured_limit(
    scope: EndpointUsageScope,
    item: Any,
    *,
    source: LimitSource,
    observed_at: str,
    parser_version: str,
) -> UsageLimit:
    if isinstance(item, UsageLimit):
        if item.scope_id != scope.scope_id:
            raise AdapterScopeError("configured limit scope_id mismatch")
        return item
    if not isinstance(item, Mapping):
        raise AdapterParseError("configured limit must be a mapping or UsageLimit")
    _reject_secret_keys(item, path="configured_limit")
    data = dict(item)
    scope_id = data.get("scope_id", scope.scope_id)
    if scope_id != scope.scope_id:
        raise AdapterScopeError("configured limit scope_id mismatch")
    dimension = data.get("dimension")
    ceiling = data.get("ceiling")
    window = data.get("window")
    if dimension is None or ceiling is None or window is None:
        raise AdapterParseError("configured limit requires dimension, ceiling, window")
    remaining = data.get("remaining")
    used = data.get("used")
    enforcement = data.get("enforcement", LimitEnforcement.HARD)
    confidence = data.get("confidence", ConfidenceLevel.HIGH)
    confidence_micros = data.get("confidence_micros", 900_000)
    currency = data.get("currency")
    reason_codes = list(data.get("reason_codes") or ())
    reason_codes.append("source.%s" % source.value)
    provenance = Provenance(
        source=source,
        parser_version=parser_version,
        observed_at=observed_at,
        reason_codes=tuple(reason_codes[:MAX_REASON_CODES]),
    )
    return UsageLimit(
        scope_id=scope.scope_id,
        dimension=dimension,
        ceiling=_coerce_quantity(ceiling, "ceiling"),
        remaining=_coerce_quantity(remaining, "remaining", default_unknown=True),
        used=_coerce_quantity(used, "used", default_unknown=True),
        window=window if isinstance(window, LimitWindow) else LimitWindow.from_dict(window),
        enforcement=enforcement,
        confidence=confidence,
        confidence_micros=confidence_micros,
        provenance=provenance,
        currency=currency,
    )


def _coerce_quantity(
    value: Any, field_name: str, *, default_unknown: bool = False
) -> Quantity:
    if value is None:
        if default_unknown:
            return Quantity.unknown()
        raise AdapterParseError("%s is required" % field_name)
    if isinstance(value, Quantity):
        return value
    if isinstance(value, Mapping):
        return Quantity.from_dict(value)
    return Quantity.finite(_require_non_negative_int(value, field_name))


# ---------------------------------------------------------------------------
# Public observation parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationInput:
    """Bounded invocation metadata for observation adapters.

    Accepts structured fields only. Callers must not pass prompts, media, or
    full raw response bodies; secret-shaped keys and values are rejected.
    """

    scope: EndpointUsageScope
    request_id: str
    http_status: Optional[int] = None
    headers: Optional[Mapping[str, Any]] = None
    usage_body: Optional[Mapping[str, Any]] = None
    error_body: Optional[Mapping[str, Any]] = None
    cli_metadata: Optional[Mapping[str, Any]] = None
    local_capacity: Optional[Mapping[str, Any]] = None
    observed_at: Optional[Union[str, datetime]] = None
    now: Optional[Union[str, datetime]] = None
    adapter_family: Optional[Union[str, AdapterFamily]] = None
    policy_ceilings: Optional[Mapping[str, Any]] = None
    configured_limits: Optional[Sequence[Any]] = None
    custom_field_map: Optional[Mapping[str, str]] = None
    claimed_scope_id: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ObservationInput":
        if not isinstance(data, Mapping):
            raise AdapterParseError("observation input must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            # Unknown top-level keys become reason codes later via adapters; hard
            # reject only credential-shaped keys here.
            for key in unknown:
                if is_secret_key(str(key)):
                    raise AdapterParseError(
                        "credential-bearing field is forbidden: %s" % key
                    )
        scope = data.get("scope")
        request_id = data.get("request_id")
        if scope is None or request_id is None:
            raise AdapterParseError("scope and request_id are required")
        return cls(
            scope=_coerce_scope(scope),
            request_id=str(request_id),
            http_status=data.get("http_status"),
            headers=data.get("headers"),
            usage_body=data.get("usage_body"),
            error_body=data.get("error_body"),
            cli_metadata=data.get("cli_metadata"),
            local_capacity=data.get("local_capacity"),
            observed_at=data.get("observed_at"),
            now=data.get("now"),
            adapter_family=data.get("adapter_family"),
            policy_ceilings=data.get("policy_ceilings"),
            configured_limits=data.get("configured_limits"),
            custom_field_map=data.get("custom_field_map"),
            claimed_scope_id=data.get("claimed_scope_id"),
        )


def parse_provider_observation(
    payload: Union[ObservationInput, Mapping[str, Any]],
    *,
    family: Optional[Union[str, AdapterFamily]] = None,
) -> ProviderUsageObservation:
    """Parse invocation metadata into a :class:`ProviderUsageObservation`.

    Settlement observations are bound to the exact ``scope.scope_id`` and
    ``request_id``. Raw headers/bodies are not retained on the observation.
    """

    if isinstance(payload, ObservationInput):
        inp = payload
    elif isinstance(payload, Mapping):
        inp = ObservationInput.from_mapping(payload)
    else:
        raise AdapterParseError("observation payload must be ObservationInput or mapping")

    scope = inp.scope
    if not isinstance(scope, EndpointUsageScope):
        scope = _coerce_scope(scope)

    if inp.claimed_scope_id is not None and inp.claimed_scope_id != scope.scope_id:
        raise AdapterScopeError("observation scope_id does not match endpoint scope")

    request_id = _require_text(inp.request_id, "request_id", MAX_STRING_BYTES)
    observed_at = _coerce_timestamp(inp.observed_at, "observed_at") or _utcnow()
    now = _coerce_timestamp(inp.now, "now") or observed_at

    family_hint = family if family is not None else inp.adapter_family
    resolved_family = resolve_adapter_family(
        family_hint,
        protocol=scope.protocol,
        default=AdapterFamily.UNKNOWN,
    )

    state = _ParseState(
        family=resolved_family,
        scope=scope,
        request_id=request_id,
        observed_at=observed_at,
        now=now,
        http_status=_optional_http_status(inp.http_status),
        policy_ceilings=_normalize_policy_ceilings(inp.policy_ceilings),
        custom_field_map=dict(inp.custom_field_map or {}),
    )
    state.add_reason("adapter.%s" % resolved_family.value)
    state.add_reason("scope.endpoint")
    if scope.account_pseudonym:
        state.add_reason("scope.account")
    if scope.project_pseudonym:
        state.add_reason("scope.project")
    if scope.credential_pseudonym:
        state.add_reason("scope.credential")
    if scope.model_id:
        state.add_reason("scope.model")
    if scope.operation:
        state.add_reason("scope.operation")

    if inp.configured_limits is not None:
        try:
            state.configured_limits = normalize_configured_limits(
                scope,
                inp.configured_limits,
                source=LimitSource.CONFIGURED,
                observed_at=observed_at,
            )
            for limit in state.configured_limits:
                if limit.ceiling.kind is QuantityKind.FINITE and limit.ceiling.value is not None:
                    state.policy_ceilings.setdefault(
                        limit.dimension.value, limit.ceiling.value
                    )
        except AdapterParseError as exc:
            state.note_failure("configured.limits_rejected")
            # Continue; configured limits are optional for observation.

    # HTTP status may already indicate restriction. Do not invent a retry
    # duration yet — explicit Retry-After / reset fields win; finalization
    # defaults only when no explicit signal was parsed.
    if state.http_status in (429, 503):
        state.restrictive = True
        state.add_reason(
            "http.%d" % state.http_status
            if state.http_status is not None
            else "http.restrictive"
        )

    # Parse channels; each channel is isolated so partial failure still allows
    # restrictive cooldowns from other channels. Hard rejects (credentials,
    # bounds, scope, structural type errors) still propagate.
    if inp.headers is not None:
        try:
            _parse_headers(state, inp.headers)
        except AdapterParseError as exc:
            state.note_failure("headers.rejected")
            if _is_hard_reject(exc):
                raise
        except Exception:
            state.note_failure("headers.parse_failed")

    if inp.usage_body is not None:
        try:
            _parse_usage_body(state, inp.usage_body)
        except AdapterParseError as exc:
            state.note_failure("usage_body.rejected")
            if _is_hard_reject(exc):
                raise
        except Exception:
            state.note_failure("usage_body.parse_failed")

    if inp.error_body is not None:
        try:
            _parse_error_body(state, inp.error_body)
        except AdapterParseError as exc:
            state.note_failure("error_body.rejected")
            if _is_hard_reject(exc):
                raise
        except Exception:
            state.note_failure("error_body.parse_failed")

    if inp.cli_metadata is not None:
        try:
            _parse_cli_metadata(state, inp.cli_metadata)
        except AdapterParseError as exc:
            state.note_failure("cli_metadata.rejected")
            if _is_hard_reject(exc):
                raise
        except Exception:
            state.note_failure("cli_metadata.parse_failed")

    if inp.local_capacity is not None:
        try:
            _parse_local_capacity(state, inp.local_capacity)
        except AdapterParseError as exc:
            state.note_failure("local_capacity.rejected")
            if _is_hard_reject(exc):
                raise
        except Exception:
            state.note_failure("local_capacity.parse_failed")

    # Family-specific post-processing (header buckets → limits).
    try:
        _materialize_header_limits(state)
    except Exception:
        state.note_failure("limits.materialize_failed")

    # Validate reset / retry coherence.
    _finalize_reset_and_retry(state)

    # Policy ceiling guard: never raise available ceiling from untrusted data.
    state.limits = list(_apply_policy_ceiling_guard(state.limits, state.policy_ceilings))

    # If restrictive and we lost everything else, still emit a cooldown limit.
    if state.restrictive and not state.limits and state.retry_after_ms is not None:
        try:
            state.limits.append(
                _build_limit(
                    state,
                    dimension=UsageDimension.REQUESTS,
                    ceiling=None,
                    remaining=Quantity.finite(0),
                    used=None,
                    window_kind=WindowKind.FIXED,
                    length_ms=state.retry_after_ms,
                    reset_at=state.reset_at,
                    source=LimitSource.ERROR,
                    reason_codes=("cooldown.restrictive",),
                )
            )
            state.add_reason("cooldown.retained")
        except Exception:
            state.note_failure("cooldown.materialize_failed")

    usage = UsageVector(entries=tuple(state.usage_entries[:MAX_VECTOR_ENTRIES]))
    limits = tuple(state.limits[:MAX_LIMITS])
    confidence, confidence_micros = _confidence_for(state)
    provenance_source = _provenance_source(state)
    digest = _observation_digest(
        scope_id=scope.scope_id,
        request_id=request_id,
        usage=usage,
        limits=limits,
        http_status=state.http_status,
        provider_request_id=state.provider_request_id,
        family=resolved_family,
    )
    provenance = Provenance(
        source=provenance_source,
        parser_version=ADAPTER_PARSER_VERSION,
        observed_at=_format_ts(state.observed_at),
        expires_at=_format_ts(state.reset_at) if state.reset_at is not None else None,
        digest=digest,
        reason_codes=tuple(state.reason_codes[:MAX_REASON_CODES]),
    )

    # Build observation; schema asserts no prompt/media/raw payload keys.
    observation = ProviderUsageObservation(
        scope_id=scope.scope_id,
        request_id=request_id,
        usage=usage,
        limits=limits,
        retry_after_ms=state.retry_after_ms,
        reset_at=_format_ts(state.reset_at) if state.reset_at is not None else None,
        http_status=state.http_status,
        provider_request_id=state.provider_request_id,
        confidence=confidence,
        confidence_micros=confidence_micros,
        provenance=provenance,
        reason_codes=tuple(state.reason_codes[:MAX_REASON_CODES]),
    )
    _assert_no_raw_payload(observation)
    return observation


def apply_policy_ceiling_guard(
    observed_limits: Sequence[UsageLimit],
    policy_ceilings: Optional[Mapping[str, Any]] = None,
    *,
    policy_limits: Optional[Sequence[UsageLimit]] = None,
) -> Tuple[UsageLimit, ...]:
    """Clamp observed ceilings so untrusted data cannot raise policy limits."""

    ceilings = _normalize_policy_ceilings(policy_ceilings)
    if policy_limits:
        for limit in policy_limits:
            if (
                isinstance(limit, UsageLimit)
                and limit.ceiling.kind is QuantityKind.FINITE
                and limit.ceiling.value is not None
            ):
                ceilings.setdefault(limit.dimension.value, limit.ceiling.value)
    return _apply_policy_ceiling_guard(list(observed_limits), ceilings)


def retain_restrictive_cooldown(
    *,
    scope: Union[EndpointUsageScope, Mapping[str, Any]],
    request_id: str,
    retry_after_ms: Optional[int] = None,
    reset_at: Optional[Union[str, datetime]] = None,
    http_status: Optional[int] = None,
    reason_codes: Sequence[str] = (),
    observed_at: Optional[Union[str, datetime]] = None,
) -> ProviderUsageObservation:
    """Build a minimal observation that preserves a valid restrictive cooldown.

    Used when broader parsing fails but 429/503/Retry-After evidence remains.
    """

    scope_obj = _coerce_scope(scope)
    observed = _coerce_timestamp(observed_at, "observed_at") or _utcnow()
    status = _optional_http_status(http_status)
    retry = _clamp_retry_ms(retry_after_ms)
    if retry is None and status in (429, 503):
        retry = DEFAULT_RETRY_MS
    reset = _coerce_timestamp(reset_at, "reset_at")
    if reset is None and retry is not None:
        reset = observed + timedelta(milliseconds=retry)
    codes = ["cooldown.retained", "parse.partial_failure"]
    for code in reason_codes:
        try:
            codes.append(_reason_code(code))
        except AdapterParseError:
            continue
    if status is not None:
        codes.append("http.%d" % status)
    state_like_family = resolve_adapter_family(None, protocol=scope_obj.protocol)
    limits: List[UsageLimit] = []
    if retry is not None:
        limits.append(
            UsageLimit(
                scope_id=scope_obj.scope_id,
                dimension=UsageDimension.REQUESTS,
                ceiling=Quantity.unknown(),
                remaining=Quantity.finite(0),
                used=Quantity.unknown(),
                window=LimitWindow(
                    kind=WindowKind.FIXED,
                    length_ms=retry,
                    reset_at=_format_ts(reset) if reset is not None else None,
                ),
                enforcement=LimitEnforcement.HARD,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_micros=500_000,
                provenance=Provenance(
                    source=LimitSource.ERROR,
                    parser_version=ADAPTER_PARSER_VERSION,
                    observed_at=_format_ts(observed),
                    reason_codes=tuple(codes[:MAX_REASON_CODES]),
                ),
            )
        )
    usage = UsageVector()
    digest = _observation_digest(
        scope_id=scope_obj.scope_id,
        request_id=_require_text(request_id, "request_id", MAX_STRING_BYTES),
        usage=usage,
        limits=tuple(limits),
        http_status=status,
        provider_request_id=None,
        family=state_like_family,
    )
    return ProviderUsageObservation(
        scope_id=scope_obj.scope_id,
        request_id=_require_text(request_id, "request_id", MAX_STRING_BYTES),
        usage=usage,
        limits=tuple(limits),
        retry_after_ms=retry,
        reset_at=_format_ts(reset) if reset is not None else None,
        http_status=status,
        confidence=ConfidenceLevel.MEDIUM,
        confidence_micros=500_000,
        provenance=Provenance(
            source=LimitSource.ERROR,
            parser_version=ADAPTER_PARSER_VERSION,
            observed_at=_format_ts(observed),
            expires_at=_format_ts(reset) if reset is not None else None,
            digest=digest,
            reason_codes=tuple(codes[:MAX_REASON_CODES]),
        ),
        reason_codes=tuple(codes[:MAX_REASON_CODES]),
    )


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def _parse_headers(state: _ParseState, headers: Any) -> None:
    normalized = _normalize_headers(headers)
    header_map = _header_map_for(state.family)
    # Always honor Retry-After and request-id across families.
    for name, value in normalized.items():
        if name in ("retry-after",):
            _ingest_retry_after(state, value, source="header")
            state.sources.add("response_header")
        if name in ("x-request-id", "request-id", "x-openai-request-id"):
            state.provider_request_id = _clamp_string(value, MAX_STRING_BYTES)
            state.sources.add("response_header")
        mapped = header_map.get(name)
        if mapped is None and state.family is AdapterFamily.CUSTOM:
            mapped = _custom_header_map(state).get(name)
        if mapped is None:
            if name.startswith("x-ratelimit") or name.startswith("anthropic-ratelimit"):
                state.add_reason("header.unknown_%s" % _safe_token(name))
            continue
        dim_name, field_name = mapped
        if dim_name == "_meta":
            if field_name == "request_id":
                state.provider_request_id = _clamp_string(value, MAX_STRING_BYTES)
            elif field_name == "retry_after":
                _ingest_retry_after(state, value, source="header")
            continue
        bucket = state.header_buckets.setdefault(dim_name, {})
        if field_name == "reset":
            reset_dt = _parse_reset_value(value, now=state.now)
            if reset_dt is None:
                state.note_failure("header.reset_invalid")
                continue
            existing = bucket.get("reset")
            if existing is not None and isinstance(existing, datetime):
                if abs(int((existing - reset_dt).total_seconds() * 1000)) > RESET_CONFLICT_TOLERANCE_MS:
                    state.note_failure("header.reset_conflict")
                    # Prefer the more restrictive (earlier) reset.
                    if reset_dt < existing:
                        bucket["reset"] = reset_dt
                    continue
            bucket["reset"] = reset_dt
            state.sources.add("response_header")
            continue
        number = _parse_bounded_int(value, field_name)
        if number is None:
            state.note_failure("header.%s_invalid" % field_name)
            continue
        bucket[field_name] = number
        state.sources.add("response_header")
        if field_name == "remaining" and number == 0:
            state.restrictive = True
            state.add_reason("limit.remaining_zero")


def _header_map_for(family: AdapterFamily) -> Mapping[str, Tuple[str, str]]:
    if family is AdapterFamily.OPENAI_COMPATIBLE:
        return _OPENAI_HEADER_MAP
    if family is AdapterFamily.ANTHROPIC:
        return _ANTHROPIC_HEADER_MAP
    if family is AdapterFamily.HUGGINGFACE:
        return _HF_HEADER_MAP
    if family is AdapterFamily.UNKNOWN:
        # Conservative: only request-id and retry-after style headers.
        return {
            "retry-after": ("_meta", "retry_after"),
            "x-request-id": ("_meta", "request_id"),
            "request-id": ("_meta", "request_id"),
            "x-ratelimit-remaining-requests": ("requests", "remaining"),
            "x-ratelimit-limit-requests": ("requests", "limit"),
            "x-ratelimit-reset-requests": ("requests", "reset"),
        }
    if family is AdapterFamily.CUSTOM:
        return {
            "retry-after": ("_meta", "retry_after"),
            "x-request-id": ("_meta", "request_id"),
            "request-id": ("_meta", "request_id"),
        }
    return {
        "retry-after": ("_meta", "retry_after"),
        "x-request-id": ("_meta", "request_id"),
        "request-id": ("_meta", "request_id"),
    }


def _custom_header_map(state: _ParseState) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for key, value in state.custom_field_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        name = key.casefold().strip()
        # value format: "dimension.field" e.g. "requests.limit"
        parts = value.casefold().strip().split(".")
        if len(parts) != 2:
            state.add_reason("custom.map_invalid")
            continue
        out[name] = (parts[0], parts[1])
    return out


def _materialize_header_limits(state: _ParseState) -> None:
    for dim_name, bucket in state.header_buckets.items():
        try:
            dimension = UsageDimension(dim_name)
        except ValueError:
            state.add_reason("header.unknown_dimension_%s" % _safe_token(dim_name))
            continue
        limit_val = bucket.get("limit")
        remaining_val = bucket.get("remaining")
        reset_dt = bucket.get("reset")
        if isinstance(reset_dt, datetime):
            if not _validate_reset_clock(state, reset_dt):
                reset_dt = None
            else:
                _merge_reset(state, reset_dt)
        ceiling = (
            Quantity.finite(limit_val)
            if isinstance(limit_val, int)
            else Quantity.unknown()
        )
        remaining = (
            Quantity.finite(remaining_val)
            if isinstance(remaining_val, int)
            else Quantity.unknown()
        )
        used = Quantity.unknown()
        if (
            ceiling.kind is QuantityKind.FINITE
            and remaining.kind is QuantityKind.FINITE
            and ceiling.value is not None
            and remaining.value is not None
            and remaining.value <= ceiling.value
        ):
            used = Quantity.finite(ceiling.value - remaining.value)
        length_ms = DEFAULT_REQUEST_WINDOW_MS
        if dimension in (
            UsageDimension.INPUT_TOKENS,
            UsageDimension.OUTPUT_TOKENS,
            UsageDimension.TOTAL_TOKENS,
        ):
            length_ms = DEFAULT_TOKEN_WINDOW_MS
        if isinstance(reset_dt, datetime):
            delta = int((reset_dt - state.observed_at).total_seconds() * 1000)
            if 0 <= delta <= MAX_WINDOW_MS:
                length_ms = max(delta, 1)
        try:
            state.limits.append(
                _build_limit(
                    state,
                    dimension=dimension,
                    ceiling=ceiling,
                    remaining=remaining,
                    used=used,
                    window_kind=WindowKind.FIXED,
                    length_ms=length_ms,
                    reset_at=reset_dt if isinstance(reset_dt, datetime) else None,
                    source=LimitSource.RESPONSE_HEADER,
                    reason_codes=("header.ratelimit",),
                )
            )
        except (AdapterParseError, SchemaValidationError):
            state.note_failure("header.limit_rejected")


# ---------------------------------------------------------------------------
# Body / error / CLI / local parsers
# ---------------------------------------------------------------------------


def _parse_usage_body(state: _ParseState, body: Any) -> None:
    mapping = _require_mapping(body, "usage_body")
    _reject_secret_keys(mapping, path="usage_body")
    if len(mapping) > MAX_BODY_KEYS:
        raise AdapterParseError("usage_body exceeds key bound")
    usage_node = mapping
    if "usage" in mapping and isinstance(mapping["usage"], Mapping):
        usage_node = mapping["usage"]
        _reject_secret_keys(usage_node, path="usage_body.usage")
    # OpenAI id on body
    for key in ("id", "request_id"):
        if key in mapping and state.provider_request_id is None:
            value = mapping[key]
            if isinstance(value, str) and value.strip():
                state.provider_request_id = _clamp_string(value, MAX_STRING_BYTES)
    _walk_usage_counters(state, usage_node, depth=0)
    state.sources.add("response_body")

    # Nested rate limit objects some providers return.
    for key in ("rate_limit", "rate_limits", "limits"):
        node = mapping.get(key)
        if isinstance(node, Mapping):
            _parse_structured_limits(state, node, source=LimitSource.RESPONSE_BODY)


def _walk_usage_counters(state: _ParseState, node: Mapping[str, Any], depth: int) -> None:
    if depth > MAX_NESTING_DEPTH:
        state.note_failure("usage.excessive_nesting")
        return
    for key, value in node.items():
        if not isinstance(key, str):
            continue
        if is_secret_key(key):
            raise AdapterParseError("credential-bearing field is forbidden: %s" % key)
        key_cf = key.casefold()
        dimension = _USAGE_BODY_KEYS.get(key_cf)
        if dimension is None:
            if isinstance(value, Mapping):
                # Walk known nested detail objects only one level of counters.
                _walk_usage_counters(state, value, depth + 1)
            elif key_cf not in ("currency", "unit", "type", "object"):
                state.add_reason("usage.unknown_%s" % _safe_token(key_cf))
            continue
        if isinstance(value, Mapping):
            # Aggregate nested detail counters conservatively if present.
            continue
        number = _parse_bounded_int(value, key_cf)
        if number is None:
            state.note_failure("usage.invalid_%s" % _safe_token(key_cf))
            continue
        currency = None
        if dimension is UsageDimension.COST_MICROS:
            raw_currency = node.get("currency") or "USD"
            if not isinstance(raw_currency, str):
                state.note_failure("usage.currency_invalid")
                continue
            currency = raw_currency.strip().upper()
        try:
            entry = UsageVectorEntry(
                dimension=dimension,
                amount=Quantity.finite(number),
                currency=currency,
            )
        except SchemaValidationError:
            state.note_failure("usage.entry_rejected")
            continue
        # Merge same dimension by summing only for cache-like dual inputs;
        # otherwise last finite wins if equal, conflict notes reason.
        existing = None
        for idx, prior in enumerate(state.usage_entries):
            if prior.dimension == dimension and prior.currency == currency:
                existing = idx
                break
        if existing is None:
            state.usage_entries.append(entry)
        else:
            prior = state.usage_entries[existing]
            if prior.amount != entry.amount:
                # Prefer max for settlement-safe accounting on conflicting reports.
                if (
                    prior.amount.kind is QuantityKind.FINITE
                    and entry.amount.kind is QuantityKind.FINITE
                    and prior.amount.value is not None
                    and entry.amount.value is not None
                ):
                    merged = max(prior.amount.value, entry.amount.value)
                    state.usage_entries[existing] = UsageVectorEntry(
                        dimension=dimension,
                        amount=Quantity.finite(merged),
                        currency=currency,
                    )
                    state.add_reason("usage.conflict_max_%s" % dimension.value)
                else:
                    state.note_failure("usage.conflict_%s" % dimension.value)


def _parse_error_body(state: _ParseState, body: Any) -> None:
    mapping = _require_mapping(body, "error_body")
    _reject_secret_keys(mapping, path="error_body")
    if len(mapping) > MAX_BODY_KEYS:
        raise AdapterParseError("error_body exceeds key bound")

    # Unwrap common {"error": {...}} envelopes without retaining messages.
    error_node = mapping.get("error") if isinstance(mapping.get("error"), Mapping) else mapping
    if not isinstance(error_node, Mapping):
        error_node = mapping
    _reject_secret_keys(error_node, path="error_body.error")

    code = error_node.get("code") or error_node.get("type") or mapping.get("code")
    code_text = str(code).casefold() if code is not None else ""
    message = error_node.get("message") or mapping.get("message") or ""
    message_text = str(message).casefold() if message is not None else ""
    # Never store message text; only classify.
    if any(marker in code_text or marker in message_text for marker in _BILLING_MARKERS):
        state.billing_exhausted = True
        state.restrictive = True
        state.add_reason("billing.exhausted")
        state.sources.add("error")
    if any(marker in code_text or marker in message_text for marker in _USAGE_LIMIT_MARKERS):
        state.restrictive = True
        state.add_reason("subscription.usage_limit")
        state.sources.add("error")
    if "overloaded" in code_text or "overloaded" in message_text:
        state.restrictive = True
        state.add_reason("http.overloaded")
        if state.http_status is None:
            state.http_status = 503

    for key in (
        "retry_after",
        "retry_after_ms",
        "retry_after_seconds",
        "resets_in_seconds",
        "reset_in_seconds",
        "estimated_time",
    ):
        if key in error_node:
            _ingest_retry_after(state, error_node[key], source="error", key=key)
        elif key in mapping:
            _ingest_retry_after(state, mapping[key], source="error", key=key)

    for key in ("reset_at", "resets_at", "rate_limit_reset"):
        if key in error_node:
            reset = _parse_reset_value(error_node[key], now=state.now)
            if reset is not None and _validate_reset_clock(state, reset):
                _merge_reset(state, reset)
        elif key in mapping:
            reset = _parse_reset_value(mapping[key], now=state.now)
            if reset is not None and _validate_reset_clock(state, reset):
                _merge_reset(state, reset)

    if state.restrictive and state.retry_after_ms is None and state.reset_at is None:
        state.retry_after_ms = DEFAULT_RETRY_MS
        state.add_reason("cooldown.defaulted")


def _parse_cli_metadata(state: _ParseState, metadata: Any) -> None:
    mapping = _require_mapping(metadata, "cli_metadata")
    _reject_secret_keys(mapping, path="cli_metadata")
    if len(mapping) > MAX_CLI_KEYS:
        raise AdapterParseError("cli_metadata exceeds key bound")
    if state.family not in (
        AdapterFamily.CLI,
        AdapterFamily.CUSTOM,
        AdapterFamily.UNKNOWN,
    ):
        # Still parse structured reset metadata when present.
        state.add_reason("cli.family_mismatch")

    kind = str(mapping.get("kind") or mapping.get("error_kind") or "").casefold()
    provider = str(mapping.get("provider") or mapping.get("cli") or "").casefold()
    if provider:
        state.add_reason("cli.%s" % _safe_token(provider))

    if kind in ("quota_exceeded", "billing", "insufficient_quota"):
        state.billing_exhausted = True
        state.restrictive = True
        state.add_reason("billing.exhausted")
    if kind in ("usage_limit", "rate_limit", "capacity"):
        state.restrictive = True
        state.add_reason("subscription.usage_limit")

    for key in (
        "resets_in_seconds",
        "reset_in_seconds",
        "retry_after_seconds",
        "retry_after",
        "retry_after_ms",
    ):
        if key in mapping:
            _ingest_retry_after(state, mapping[key], source="cli", key=key)

    if "reset_at" in mapping:
        reset = _parse_reset_value(mapping["reset_at"], now=state.now)
        if reset is not None and _validate_reset_clock(state, reset):
            _merge_reset(state, reset)

    # Optional structured usage from CLI JSON.
    usage = mapping.get("usage")
    if isinstance(usage, Mapping):
        _walk_usage_counters(state, usage, depth=0)

    # Nested JSONL-style objects: only look at known reset keys, never messages.
    nested = mapping.get("events")
    if isinstance(nested, (list, tuple)):
        for item in nested[:16]:
            if not isinstance(item, Mapping):
                continue
            for key in (
                "resets_in_seconds",
                "reset_in_seconds",
                "retry_after_seconds",
                "retry_after",
            ):
                if key in item:
                    _ingest_retry_after(state, item[key], source="cli", key=key)

    if state.restrictive or state.retry_after_ms is not None:
        state.sources.add("error" if state.restrictive else "response_body")
        if state.restrictive and state.retry_after_ms is None:
            state.retry_after_ms = DEFAULT_RETRY_MS
            state.add_reason("cooldown.defaulted")


def _parse_local_capacity(state: _ParseState, capacity: Any) -> None:
    mapping = _require_mapping(capacity, "local_capacity")
    _reject_secret_keys(mapping, path="local_capacity")
    if len(mapping) > MAX_LOCAL_KEYS:
        raise AdapterParseError("local_capacity exceeds key bound")
    state.sources.add("local_observation")
    state.add_reason("scope.concurrent")

    concurrent = mapping.get("max_concurrent_requests", mapping.get("concurrent_requests"))
    in_flight = mapping.get("in_flight_requests", mapping.get("active_requests"))
    if concurrent is not None:
        ceiling_v = _parse_bounded_int(concurrent, "concurrent_requests")
        if ceiling_v is None:
            state.note_failure("local.concurrent_invalid")
        else:
            remaining_q: Quantity = Quantity.unknown()
            used_q: Quantity = Quantity.unknown()
            if in_flight is not None:
                used_v = _parse_bounded_int(in_flight, "in_flight_requests")
                if used_v is None:
                    state.note_failure("local.in_flight_invalid")
                elif used_v > ceiling_v:
                    state.note_failure("local.in_flight_exceeds_ceiling")
                else:
                    used_q = Quantity.finite(used_v)
                    remaining_q = Quantity.finite(ceiling_v - used_v)
            try:
                state.limits.append(
                    _build_limit(
                        state,
                        dimension=UsageDimension.CONCURRENT_REQUESTS,
                        ceiling=Quantity.finite(ceiling_v),
                        remaining=remaining_q,
                        used=used_q,
                        window_kind=WindowKind.CONCURRENT,
                        length_ms=None,
                        reset_at=None,
                        source=LimitSource.LOCAL_OBSERVATION,
                        reason_codes=("local.concurrency",),
                    )
                )
            except (AdapterParseError, SchemaValidationError):
                state.note_failure("local.concurrent_limit_rejected")

    streams = mapping.get("max_concurrent_streams", mapping.get("concurrent_streams"))
    if streams is not None:
        ceiling_v = _parse_bounded_int(streams, "concurrent_streams")
        if ceiling_v is not None:
            try:
                state.limits.append(
                    _build_limit(
                        state,
                        dimension=UsageDimension.CONCURRENT_STREAMS,
                        ceiling=Quantity.finite(ceiling_v),
                        remaining=Quantity.unknown(),
                        used=Quantity.unknown(),
                        window_kind=WindowKind.CONCURRENT,
                        length_ms=None,
                        reset_at=None,
                        source=LimitSource.LOCAL_OBSERVATION,
                        reason_codes=("local.concurrency",),
                    )
                )
            except (AdapterParseError, SchemaValidationError):
                state.note_failure("local.stream_limit_rejected")

    # Memory ceilings are expressed as media_bytes when finite.
    memory = mapping.get("max_memory_bytes", mapping.get("memory_bytes_ceiling"))
    memory_used = mapping.get("memory_bytes_used")
    if memory is not None:
        ceiling_v = _parse_bounded_int(memory, "memory_bytes")
        if ceiling_v is not None:
            remaining_q = Quantity.unknown()
            used_q = Quantity.unknown()
            if memory_used is not None:
                used_v = _parse_bounded_int(memory_used, "memory_bytes_used")
                if used_v is not None and used_v <= ceiling_v:
                    used_q = Quantity.finite(used_v)
                    remaining_q = Quantity.finite(ceiling_v - used_v)
                elif used_v is not None:
                    state.note_failure("local.memory_used_exceeds_ceiling")
            try:
                state.limits.append(
                    _build_limit(
                        state,
                        dimension=UsageDimension.MEDIA_BYTES,
                        ceiling=Quantity.finite(ceiling_v),
                        remaining=remaining_q,
                        used=used_q,
                        window_kind=WindowKind.LIFETIME,
                        length_ms=None,
                        reset_at=None,
                        source=LimitSource.LOCAL_OBSERVATION,
                        reason_codes=("local.memory",),
                    )
                )
            except (AdapterParseError, SchemaValidationError):
                state.note_failure("local.memory_limit_rejected")


def _parse_structured_limits(
    state: _ParseState, node: Mapping[str, Any], *, source: LimitSource
) -> None:
    """Parse a compact structured limits object (not raw provider payload dump)."""

    for key, value in node.items():
        if not isinstance(key, str) or is_secret_key(key):
            if isinstance(key, str) and is_secret_key(key):
                raise AdapterParseError(
                    "credential-bearing field is forbidden: %s" % key
                )
            continue
        key_cf = key.casefold()
        try:
            dimension = UsageDimension(key_cf)
        except ValueError:
            state.add_reason("limits.unknown_%s" % _safe_token(key_cf))
            continue
        if isinstance(value, Mapping):
            limit_v = value.get("limit", value.get("ceiling"))
            remaining_v = value.get("remaining")
            reset_v = value.get("reset", value.get("reset_at"))
        else:
            limit_v = value
            remaining_v = None
            reset_v = None
        ceiling = (
            Quantity.finite(_parse_bounded_int(limit_v, "limit"))
            if limit_v is not None and _parse_bounded_int(limit_v, "limit") is not None
            else Quantity.unknown()
        )
        remaining = (
            Quantity.finite(_parse_bounded_int(remaining_v, "remaining"))
            if remaining_v is not None
            and _parse_bounded_int(remaining_v, "remaining") is not None
            else Quantity.unknown()
        )
        reset_dt = (
            _parse_reset_value(reset_v, now=state.now) if reset_v is not None else None
        )
        if reset_dt is not None and not _validate_reset_clock(state, reset_dt):
            reset_dt = None
        if reset_dt is not None:
            _merge_reset(state, reset_dt)
        try:
            state.limits.append(
                _build_limit(
                    state,
                    dimension=dimension,
                    ceiling=ceiling,
                    remaining=remaining,
                    used=Quantity.unknown(),
                    window_kind=WindowKind.FIXED,
                    length_ms=DEFAULT_REQUEST_WINDOW_MS,
                    reset_at=reset_dt,
                    source=source,
                    reason_codes=("body.limits",),
                )
            )
        except (AdapterParseError, SchemaValidationError):
            state.note_failure("body.limit_rejected")


# ---------------------------------------------------------------------------
# Limit construction and policy guard
# ---------------------------------------------------------------------------


def _build_limit(
    state: _ParseState,
    *,
    dimension: UsageDimension,
    ceiling: Optional[Quantity],
    remaining: Optional[Quantity],
    used: Optional[Quantity],
    window_kind: WindowKind,
    length_ms: Optional[int],
    reset_at: Optional[datetime],
    source: LimitSource,
    reason_codes: Sequence[str],
) -> UsageLimit:
    ceiling_q = ceiling if ceiling is not None else Quantity.unknown()
    remaining_q = remaining if remaining is not None else Quantity.unknown()
    used_q = used if used is not None else Quantity.unknown()

    # Never raise policy ceiling.
    policy_cap = state.policy_ceilings.get(dimension.value)
    clamped = False
    if (
        policy_cap is not None
        and ceiling_q.kind is QuantityKind.FINITE
        and ceiling_q.value is not None
        and ceiling_q.value > policy_cap
    ):
        state.add_reason("policy.ceiling_clamped")
        ceiling_q = Quantity.finite(policy_cap)
        clamped = True
        if (
            remaining_q.kind is QuantityKind.FINITE
            and remaining_q.value is not None
            and remaining_q.value > policy_cap
        ):
            remaining_q = Quantity.finite(policy_cap)

    if window_kind is WindowKind.CONCURRENT:
        window = LimitWindow(kind=WindowKind.CONCURRENT)
    elif window_kind is WindowKind.LIFETIME:
        window = LimitWindow(kind=WindowKind.LIFETIME)
    elif window_kind is WindowKind.BILLING:
        window = LimitWindow(
            kind=WindowKind.BILLING,
            reset_at=_format_ts(reset_at) if reset_at is not None else None,
            anchor_at=_format_ts(state.observed_at),
        )
    else:
        length = length_ms if length_ms is not None else DEFAULT_REQUEST_WINDOW_MS
        length = max(0, min(int(length), MAX_WINDOW_MS))
        window = LimitWindow(
            kind=window_kind,
            length_ms=length if length > 0 else 1,
            reset_at=_format_ts(reset_at) if reset_at is not None else None,
        )

    conf = ConfidenceLevel.HIGH if source is LimitSource.RESPONSE_HEADER else ConfidenceLevel.MEDIUM
    conf_micros = 800_000 if conf is ConfidenceLevel.HIGH else 500_000
    if state.parse_failures:
        conf = ConfidenceLevel.LOW
        conf_micros = 200_000

    codes = list(reason_codes) + ["source.%s" % source.value]
    if clamped:
        codes.append("policy.ceiling_clamped")
    return UsageLimit(
        scope_id=state.scope.scope_id,
        dimension=dimension,
        ceiling=ceiling_q,
        remaining=remaining_q,
        used=used_q,
        window=window,
        enforcement=LimitEnforcement.HARD,
        confidence=conf,
        confidence_micros=conf_micros,
        provenance=Provenance(
            source=source,
            parser_version=ADAPTER_PARSER_VERSION,
            observed_at=_format_ts(state.observed_at),
            expires_at=_format_ts(reset_at) if reset_at is not None else None,
            reason_codes=tuple(codes[:MAX_REASON_CODES]),
        ),
    )


def _apply_policy_ceiling_guard(
    limits: Sequence[UsageLimit],
    policy_ceilings: Mapping[str, int],
) -> Tuple[UsageLimit, ...]:
    if not policy_ceilings:
        return tuple(limits)
    out: List[UsageLimit] = []
    for limit in limits:
        cap = policy_ceilings.get(limit.dimension.value)
        if (
            cap is None
            or limit.ceiling.kind is not QuantityKind.FINITE
            or limit.ceiling.value is None
            or limit.ceiling.value <= cap
        ):
            out.append(limit)
            continue
        remaining = limit.remaining
        if (
            remaining.kind is QuantityKind.FINITE
            and remaining.value is not None
            and remaining.value > cap
        ):
            remaining = Quantity.finite(cap)
        used = limit.used
        if used.kind is QuantityKind.FINITE and used.value is not None and used.value > cap:
            used = Quantity.finite(cap)
        reason = list(limit.provenance.reason_codes) + ["policy.ceiling_clamped"]
        out.append(
            UsageLimit(
                scope_id=limit.scope_id,
                dimension=limit.dimension,
                ceiling=Quantity.finite(cap),
                remaining=remaining,
                used=used,
                window=limit.window,
                enforcement=limit.enforcement,
                confidence=limit.confidence,
                confidence_micros=limit.confidence_micros,
                provenance=Provenance(
                    source=limit.provenance.source,
                    parser_version=limit.provenance.parser_version,
                    observed_at=limit.provenance.observed_at,
                    expires_at=limit.provenance.expires_at,
                    digest=limit.provenance.digest,
                    reason_codes=tuple(reason[:MAX_REASON_CODES]),
                ),
                currency=limit.currency,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Retry / reset helpers
# ---------------------------------------------------------------------------


def _ingest_retry_after(
    state: _ParseState,
    value: Any,
    *,
    source: str,
    key: str = "retry_after",
) -> None:
    key_cf = key.casefold()
    if key_cf.endswith("_ms") or key_cf == "retry_after_ms":
        number = _parse_bounded_int(value, key)
        if number is None:
            state.note_failure("%s.retry_invalid" % source)
            return
        ms = _clamp_retry_ms(number)
    elif key_cf in (
        "retry_after",
        "retry_after_seconds",
        "resets_in_seconds",
        "reset_in_seconds",
        "estimated_time",
    ):
        # Retry-After may be seconds or HTTP-date.
        if isinstance(value, str) and _HTTP_DATE.match(value.strip()):
            try:
                dt = parsedate_to_datetime(value.strip())
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ms = int((dt - state.now).total_seconds() * 1000)
                ms = _clamp_retry_ms(max(ms, 0))
                if _validate_reset_clock(state, dt):
                    _merge_reset(state, dt.astimezone(timezone.utc))
            except (TypeError, ValueError, OverflowError):
                state.note_failure("%s.retry_date_invalid" % source)
                return
        else:
            number = _parse_bounded_int(value, key)
            if number is None:
                # Numeric strings with units are rejected rather than guessed.
                state.note_failure("%s.retry_invalid" % source)
                return
            if key_cf.endswith("_ms"):
                ms = _clamp_retry_ms(number)
            else:
                # Treat as seconds when reasonable; values that already look like
                # ms (>1 day in seconds) get clamped by ms bound after conversion.
                if number > MAX_WINDOW_MS:
                    state.note_failure("%s.retry_overflow" % source)
                    return
                ms = _clamp_retry_ms(number * 1000 if number < MAX_WINDOW_MS // 1000 else number)
    else:
        number = _parse_bounded_int(value, key)
        if number is None:
            state.note_failure("%s.retry_invalid" % source)
            return
        ms = _clamp_retry_ms(number * 1000)

    if ms is None:
        return
    if state.retry_after_ms is None:
        state.retry_after_ms = ms
    else:
        # Prefer the more restrictive (longer) cooldown when combining signals.
        state.retry_after_ms = max(state.retry_after_ms, ms)
    state.restrictive = True
    state.add_reason("%s.retry_after" % source)
    candidate_reset = state.observed_at + timedelta(milliseconds=ms)
    if state.reset_at is None:
        state.reset_at = candidate_reset
    else:
        _merge_reset(state, candidate_reset)


def _parse_reset_value(value: Any, *, now: datetime) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        # Heuristic: values that look like unix seconds vs ms vs delta seconds.
        if number < 0:
            return None
        if number > MAX_ABS_INTEGER:
            return None
        # Absolute unix ms
        if number > 10_000_000_000:  # ms epoch-ish
            try:
                return datetime.fromtimestamp(number / 1000.0, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        # Absolute unix seconds
        if number > 1_000_000_000:
            try:
                return datetime.fromtimestamp(number, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        # Relative seconds from now
        if number * 1000 > MAX_WINDOW_MS:
            return None
        return now + timedelta(seconds=number)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if is_secret_value(text):
            raise AdapterParseError("credential-bearing reset value is forbidden")
        # Duration like "1s", "6m0s", "1h" (OpenAI float duration strings).
        duration_ms = _parse_duration_string(text)
        if duration_ms is not None:
            return now + timedelta(milliseconds=duration_ms)
        if _HTTP_DATE.match(text):
            try:
                dt = parsedate_to_datetime(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                return None
        # RFC3339
        try:
            raw = text[:-1] + "+00:00" if text.endswith("Z") else text
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                return None
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
        # Pure integer string
        if re.fullmatch(r"\d+", text):
            return _parse_reset_value(int(text), now=now)
    return None


def _parse_duration_string(text: str) -> Optional[int]:
    """Parse OpenAI-style duration strings (e.g. ``6m0s``, ``1s``, ``1h``) to ms."""

    if not re.fullmatch(r"(?:\d+(?:\.\d+)?[hms])+", text.casefold()):
        # Also allow plain float seconds with trailing s only already covered.
        return None
    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", text.casefold()):
        value = float(amount)
        if unit == "h":
            total += value * 3_600_000
        elif unit == "m":
            total += value * 60_000
        else:
            total += value * 1_000
    if total < 0 or total > MAX_WINDOW_MS:
        return None
    return int(total)


def _validate_reset_clock(state: _ParseState, reset: datetime) -> bool:
    """Reject stale or absurd resets relative to observed_at/now."""

    if reset.tzinfo is None:
        state.note_failure("reset.missing_timezone")
        return False
    reset = reset.astimezone(timezone.utc)
    # Too far in the past relative to now → stale.
    past_ms = int((state.now - reset).total_seconds() * 1000)
    if past_ms > MAX_CLOCK_SKEW_PAST_MS:
        state.note_failure("reset.stale")
        return False
    # Too far in the future.
    future_ms = int((reset - state.now).total_seconds() * 1000)
    if future_ms > MAX_RESET_FUTURE_MS:
        state.note_failure("reset.future_overflow")
        return False
    return True


def _merge_reset(state: _ParseState, reset: datetime) -> None:
    reset = reset.astimezone(timezone.utc)
    if state.reset_at is None:
        state.reset_at = reset
        return
    delta_ms = abs(int((state.reset_at - reset).total_seconds() * 1000))
    if delta_ms > RESET_CONFLICT_TOLERANCE_MS:
        state.add_reason("reset.conflict")
        # Prefer earlier (more restrictive) reset.
        if reset < state.reset_at:
            state.reset_at = reset
    else:
        # Keep existing; already aligned.
        pass


def _finalize_reset_and_retry(state: _ParseState) -> None:
    if state.reset_at is not None:
        if not _validate_reset_clock(state, state.reset_at):
            state.reset_at = None
        elif state.retry_after_ms is None:
            delta = int((state.reset_at - state.observed_at).total_seconds() * 1000)
            if delta >= 0:
                state.retry_after_ms = _clamp_retry_ms(delta)
    if state.billing_exhausted:
        # Billing exhaustion is not a short cooldown; leave retry as-is but mark.
        state.add_reason("billing.no_short_reset")
    if state.restrictive and state.retry_after_ms is None:
        state.retry_after_ms = DEFAULT_RETRY_MS
        state.add_reason("cooldown.defaulted")
    if state.retry_after_ms is not None:
        state.retry_after_ms = _clamp_retry_ms(state.retry_after_ms)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _coerce_scope(scope: Any) -> EndpointUsageScope:
    if isinstance(scope, EndpointUsageScope):
        return scope
    if isinstance(scope, Mapping):
        return EndpointUsageScope.from_dict(scope)
    raise AdapterParseError("scope must be EndpointUsageScope or mapping")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _format_ts(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _coerce_timestamp(value: Any, field_name: str) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise AdapterParseError("%s must include a timezone" % field_name)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise AdapterParseError("%s must not be empty" % field_name)
        try:
            raw = text[:-1] + "+00:00" if text.endswith("Z") else text
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise AdapterParseError(
                "%s must be an RFC 3339 timestamp" % field_name
            ) from exc
        if parsed.tzinfo is None:
            raise AdapterParseError("%s must include a timezone" % field_name)
        return parsed.astimezone(timezone.utc)
    raise AdapterParseError("%s must be an RFC 3339 string or datetime" % field_name)


def _require_text(value: Any, field_name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise AdapterParseError("%s must be a string" % field_name)
    text = value.strip()
    if not text:
        raise AdapterParseError("%s must not be empty" % field_name)
    if len(text.encode("utf-8")) > maximum:
        raise AdapterParseError("%s exceeds bound" % field_name)
    if is_secret_value(text):
        raise AdapterParseError("%s is credential-shaped" % field_name)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise AdapterParseError("%s contains control characters" % field_name)
    return text


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        # Allow numeric strings only when pure digits.
        if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
            value = int(value.strip())
        else:
            raise AdapterParseError("%s must be an integer" % field_name)
    if value < 0:
        raise AdapterParseError("%s must be non-negative" % field_name)
    if value > MAX_ABS_INTEGER:
        raise AdapterParseError("%s overflows the allowed bound" % field_name)
    return value


def _parse_bounded_int(value: Any, field_name: str) -> Optional[int]:
    try:
        return _require_non_negative_int(value, field_name)
    except AdapterParseError:
        return None


def _optional_http_status(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterParseError("http_status must be an integer")
    if value < 100 or value > 599:
        raise AdapterParseError("http_status must be a valid HTTP status")
    return value


def _clamp_retry_ms(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < MIN_RETRY_MS:
        return None
    if value > MAX_WINDOW_MS:
        return MAX_WINDOW_MS
    return value


def _clamp_string(value: Any, maximum: int) -> str:
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    if is_secret_value(text):
        raise AdapterParseError("credential-shaped string is forbidden")
    encoded = text.encode("utf-8")
    if len(encoded) > maximum:
        text = encoded[:maximum].decode("utf-8", errors="ignore")
    return text


def _reason_code(value: Any) -> str:
    if not isinstance(value, str):
        raise AdapterParseError("reason code must be a string")
    text = value.casefold().strip().replace(" ", "_")
    text = re.sub(r"[^a-z0-9._-]", "", text)
    if not text:
        raise AdapterParseError("reason code is empty")
    if not _REASON.fullmatch(text):
        # Truncate / reshape into a safe token.
        text = _safe_token(text)
    if len(text.encode("utf-8")) > 64:
        text = text[:64]
    if not _REASON.fullmatch(text):
        raise AdapterParseError("reason code is not canonical")
    return text


def _safe_token(value: str) -> str:
    text = re.sub(r"[^a-z0-9._-]", "", value.casefold())
    if not text:
        return "unknown"
    if text[0] not in "abcdefghijklmnopqrstuvwxyz":
        text = "x." + text
    return text[:64]


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterParseError("%s must be a mapping" % field_name)
    if any(not isinstance(key, str) for key in value):
        raise AdapterParseError("%s keys must be strings" % field_name)
    return value


def _reject_secret_keys(mapping: Mapping[str, Any], *, path: str) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise AdapterParseError("%s keys must be strings" % path)
        if is_secret_key(key):
            raise AdapterParseError(
                "credential-bearing field is forbidden: %s" % key
            )
        if isinstance(value, str) and is_secret_value(value):
            raise AdapterParseError(
                "credential-bearing value is forbidden at %s.%s" % (path, key)
            )


def _normalize_headers(headers: Any) -> Dict[str, str]:
    if not isinstance(headers, Mapping):
        raise AdapterParseError("headers must be a mapping")
    if len(headers) > MAX_HEADERS:
        raise AdapterParseError("headers exceeds maximum count")
    out: Dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str):
            raise AdapterParseError("header names must be strings")
        name = key.strip().casefold()
        if not name:
            raise AdapterParseError("header name must not be empty")
        if len(name.encode("utf-8")) > MAX_HEADER_NAME_BYTES:
            raise AdapterParseError("header name exceeds bound")
        if is_secret_key(name) or name in (
            "authorization",
            "proxy-authorization",
            "x-api-key",
            "api-key",
        ):
            raise AdapterParseError(
                "credential-bearing header is forbidden: %s" % name
            )
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        if is_secret_value(value):
            raise AdapterParseError(
                "credential-bearing header value is forbidden: %s" % name
            )
        if len(value.encode("utf-8")) > MAX_HEADER_VALUE_BYTES:
            raise AdapterParseError("header value exceeds bound: %s" % name)
        out[name] = value.strip()
    return out


def _normalize_policy_ceilings(
    value: Optional[Mapping[str, Any]],
) -> Dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AdapterParseError("policy_ceilings must be a mapping")
    out: Dict[str, int] = {}
    for key, amount in value.items():
        if not isinstance(key, str):
            raise AdapterParseError("policy ceiling keys must be strings")
        try:
            dimension = UsageDimension(key.casefold())
        except ValueError as exc:
            raise AdapterParseError("unknown policy ceiling dimension") from exc
        out[dimension.value] = _require_non_negative_int(amount, "policy_ceiling")
    return out


def _is_hard_reject(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return any(
        token in message
        for token in (
            "credential",
            "forbidden",
            "scope_id mismatch",
            "scope_id does not match",
            "must be non-negative",
            "overflow",
            "exceeds",
            "must be a mapping",
            "must be a sequence",
            "must be an integer",
            "must be a string",
            "must be an object",
            "valid http status",
            "control characters",
            "required",
        )
    )


def _confidence_for(state: _ParseState) -> Tuple[ConfidenceLevel, int]:
    if state.parse_failures and state.restrictive:
        return ConfidenceLevel.LOW, 250_000
    if state.parse_failures:
        return ConfidenceLevel.LOW, 200_000
    if "response_header" in state.sources and "response_body" in state.sources:
        return ConfidenceLevel.AUTHORITATIVE, 950_000
    if "response_header" in state.sources or "response_body" in state.sources:
        return ConfidenceLevel.HIGH, 800_000
    if "local_observation" in state.sources:
        return ConfidenceLevel.HIGH, 750_000
    if "error" in state.sources:
        return ConfidenceLevel.MEDIUM, 500_000
    return ConfidenceLevel.LOW, 100_000


def _provenance_source(state: _ParseState) -> LimitSource:
    if "response_header" in state.sources:
        return LimitSource.RESPONSE_HEADER
    if "response_body" in state.sources:
        return LimitSource.RESPONSE_BODY
    if "local_observation" in state.sources:
        return LimitSource.LOCAL_OBSERVATION
    if "error" in state.sources:
        return LimitSource.ERROR
    return LimitSource.UNKNOWN


def _observation_digest(
    *,
    scope_id: str,
    request_id: str,
    usage: UsageVector,
    limits: Sequence[UsageLimit],
    http_status: Optional[int],
    provider_request_id: Optional[str],
    family: AdapterFamily,
) -> str:
    material = {
        "scope_id": scope_id,
        "request_id": request_id,
        "usage": usage.to_dict(),
        "limit_ids": [item.limit_id for item in limits],
        "http_status": http_status,
        "provider_request_id": provider_request_id,
        "family": family.value,
        "parser_version": ADAPTER_PARSER_VERSION,
    }
    # content_cid returns a multiformats-style cid; provenance digest wants hex.
    # Use the stable hex portion of content_cid input hash via content_cid on a
    # framed object and strip to 64 hex when possible; otherwise hash canonical.
    cid = content_cid(material)
    # Prefer trailing 64 hex chars when present.
    hex_part = re.sub(r"[^0-9a-f]", "", cid)[-64:]
    if len(hex_part) == 64:
        return hex_part
    import hashlib

    return hashlib.sha256(repr(material).encode("utf-8")).hexdigest()


def _assert_no_raw_payload(observation: ProviderUsageObservation) -> None:
    payload = observation.to_dict()
    forbidden = (
        "raw_headers",
        "raw_body",
        "response_body",
        "prompt",
        "messages",
        "authorization",
    )
    encoded = repr(payload).casefold()
    for key in forbidden:
        if key in payload:
            raise AdapterParseError("observation retained forbidden field: %s" % key)
    # Redact check: no bearer-shaped leftovers.
    redacted = redact_secrets(payload)
    if redacted != payload:
        raise AdapterParseError("observation contained secret-shaped material")
    if "bearer " in encoded:
        raise AdapterParseError("observation contained bearer material")


def parse_openai_compatible_observation(
    payload: Union[ObservationInput, Mapping[str, Any]],
) -> ProviderUsageObservation:
    """Convenience wrapper forcing the OpenAI-compatible family."""

    return parse_provider_observation(payload, family=AdapterFamily.OPENAI_COMPATIBLE)


def parse_anthropic_observation(
    payload: Union[ObservationInput, Mapping[str, Any]],
) -> ProviderUsageObservation:
    return parse_provider_observation(payload, family=AdapterFamily.ANTHROPIC)


def parse_huggingface_observation(
    payload: Union[ObservationInput, Mapping[str, Any]],
) -> ProviderUsageObservation:
    return parse_provider_observation(payload, family=AdapterFamily.HUGGINGFACE)


def parse_cli_observation(
    payload: Union[ObservationInput, Mapping[str, Any]],
) -> ProviderUsageObservation:
    return parse_provider_observation(payload, family=AdapterFamily.CLI)


def parse_local_observation(
    payload: Union[ObservationInput, Mapping[str, Any]],
) -> ProviderUsageObservation:
    return parse_provider_observation(payload, family=AdapterFamily.LOCAL)


def parse_custom_observation(
    payload: Union[ObservationInput, Mapping[str, Any]],
) -> ProviderUsageObservation:
    return parse_provider_observation(payload, family=AdapterFamily.CUSTOM)


__all__ = [
    "ADAPTER_PARSER_VERSION",
    "AdapterError",
    "AdapterParseError",
    "AdapterScopeError",
    "ObservationInput",
    "PROVIDER_USAGE_ADAPTER_REQUIREMENT_ID",
    "apply_policy_ceiling_guard",
    "normalize_configured_limits",
    "parse_anthropic_observation",
    "parse_cli_observation",
    "parse_custom_observation",
    "parse_huggingface_observation",
    "parse_local_observation",
    "parse_openai_compatible_observation",
    "parse_provider_observation",
    "retain_restrictive_cooldown",
]
