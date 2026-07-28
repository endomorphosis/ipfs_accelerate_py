"""Version 1 contracts for endpoint-scoped usage, limits, events, and receipts.

These types are provider-neutral and pure: constructing or serializing them
never contacts a network, provider, process, secret store, model loader, or
database.  Unknown is distinct from unlimited.  Canonical serialization is
bounded, deterministic, round-trippable, and fail-closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

from .identity import (
    IDENTITY_POLICY_VERSION,
    CanonicalizationError,
    UsageIdentityError,
    assert_no_prompt_media_or_output,
    canonical_json,
    contains_bearer_url,
    contains_raw_endpoint,
    content_cid,
    endpoint_usage_scope_identity,
    event_identity,
    is_secret_key,
    is_secret_value,
    receipt_identity,
    require_catalog_id,
    reservation_identity,
    scope_identity_components,
    snapshot_identity,
    stable_id,
)

ENDPOINT_USAGE_CONTRACT_REQUIREMENT_ID = "requirement:endpoint-usage-contract.v1"
SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset((SCHEMA_VERSION,))

MAX_REASON_CODES = 32
MAX_REASON_CODE_BYTES = 64
MAX_VECTOR_ENTRIES = 32
MAX_LIMITS = 64
MAX_RESERVATIONS = 256
MAX_CANDIDATES = 128
MAX_RANKING_INPUTS = 64
MAX_PROVENANCE = 32
MAX_LABELS = 32
MAX_STRING_BYTES = 512
MAX_DESCRIPTION_BYTES = 1024
MAX_CURRENCY_BYTES = 8
MAX_ABS_INTEGER = (1 << 63) - 1
MAX_WINDOW_MS = 31_622_400_000
MAX_CONFIDENCE = 1_000_000
MAX_NESTING_HINT = 24

_NAME = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_REASON = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
_OPERATION = re.compile(r"^[A-Za-z0-9._\-/]{1,64}$")
_REGION = re.compile(r"^[A-Za-z0-9._\-/]{1,64}$")
_PSEUDONYM_OR_ID = re.compile(r"^[a-z]+_[0-9a-f]{64}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


class SchemaValidationError(ValueError):
    """A usage contract record violates the v1 schema."""


class UsageErrorCode(str, Enum):
    """Typed error codes for usage contract and admission failures."""

    INVALID_SCOPE = "invalid_scope"
    UNKNOWN_SCOPE = "unknown_scope"
    UNKNOWN_FIELD = "unknown_field"
    NEGATIVE_VALUE = "negative_value"
    OVERFLOW = "overflow"
    INVALID_UNIT_WINDOW = "invalid_unit_window"
    CREDENTIAL_SHAPED = "credential_shaped"
    BEARER_URL = "bearer_url"
    PROMPT_OR_MEDIA = "prompt_or_media"
    EXCESSIVE_NESTING = "excessive_nesting"
    LIMIT_EXHAUSTED = "limit_exhausted"
    RESERVATION_CONFLICT = "reservation_conflict"
    STALE_SNAPSHOT = "stale_snapshot"
    CAPACITY_UNAVAILABLE = "usage_capacity_unavailable"
    POLICY_DENIED = "policy_denied"
    UNSUPPORTED_SCHEMA = "unsupported_schema"


class UsageDimension(str, Enum):
    """Typed usage dimensions; never collapsed into a fictional universal unit."""

    REQUESTS = "requests"
    BATCH_ITEMS = "batch_items"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    TOTAL_TOKENS = "total_tokens"
    EMBEDDING_INPUTS = "embedding_inputs"
    EMBEDDING_TOKENS = "embedding_tokens"
    VECTORS = "vectors"
    IMAGES = "images"
    PIXELS = "pixels"
    MEDIA_BYTES = "media_bytes"
    AUDIO_SECONDS = "audio_seconds"
    CHARACTERS = "characters"
    CONCURRENT_REQUESTS = "concurrent_requests"
    CONCURRENT_STREAMS = "concurrent_streams"
    COST_MICROS = "cost_micros"


class QuantityKind(str, Enum):
    """Distinguish unknown from unlimited from a known finite amount."""

    FINITE = "finite"
    UNKNOWN = "unknown"
    UNLIMITED = "unlimited"


class WindowKind(str, Enum):
    FIXED = "fixed"
    SLIDING = "sliding"
    TOKEN_BUCKET = "token_bucket"
    CONCURRENT = "concurrent"
    BILLING = "billing"
    LIFETIME = "lifetime"


class LimitSource(str, Enum):
    POLICY = "policy"
    CONFIGURED = "configured"
    RESPONSE_HEADER = "response_header"
    RESPONSE_BODY = "response_body"
    ERROR = "error"
    RECONCILED = "reconciled"
    LOCAL_OBSERVATION = "local_observation"
    UNKNOWN = "unknown"


class LimitEnforcement(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    DIAGNOSTIC = "diagnostic"


class ProtocolKind(str, Enum):
    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    CLI = "cli"
    LOCAL = "local"
    UNIX = "unix"
    UNKNOWN = "unknown"


class AvailabilityState(str, Enum):
    AVAILABLE = "available"
    NEAR_LIMIT = "near_limit"
    EXHAUSTED = "exhausted"
    COOLING_DOWN = "cooling_down"
    STALE = "stale"
    UNKNOWN = "unknown"
    DISABLED = "disabled"
    UNROUTABLE = "unroutable"


class UsageEventKind(str, Enum):
    ESTIMATE = "estimate"
    RESERVATION = "reservation"
    STREAM_SETTLEMENT = "stream_settlement"
    OBSERVATION_SUCCESS = "observation_success"
    OBSERVATION_FAILURE = "observation_failure"
    COMMIT = "commit"
    RELEASE = "release"
    EXPIRY_RECOVERY = "expiry_recovery"
    REFUND = "refund"
    CORRECTION = "correction"


class ReservationState(str, Enum):
    PENDING = "pending"
    HELD = "held"
    COMMITTED = "committed"
    RELEASED = "released"
    EXPIRED = "expired"
    REJECTED = "rejected"


class FallbackClass(str, Enum):
    NONE = "none"
    SAME_DEPLOYMENT = "same_deployment"
    SAME_PROVIDER = "same_provider"
    SAME_MODEL = "same_model"
    EQUIVALENT_MODEL = "equivalent_model"
    CROSS_PROVIDER = "cross_provider"


class RoutingMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    SHADOW = "shadow"
    ASSIST = "assist"
    ENFORCE = "enforce"


class EstimateMethod(str, Enum):
    STATIC = "static"
    HEURISTIC = "heuristic"
    TOKENIZER = "tokenizer"
    PROVIDER_HINT = "provider_hint"
    CONSERVATIVE = "conservative"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTHORITATIVE = "authoritative"


_CONCURRENT_DIMENSIONS = frozenset(
    (UsageDimension.CONCURRENT_REQUESTS, UsageDimension.CONCURRENT_STREAMS)
)


def _fail(message: str) -> None:
    raise SchemaValidationError(message)


def _version(value: Any) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_SCHEMA_VERSIONS:
        _fail("unsupported schema_version: %r" % (value,))
    return value


def _text(
    value: Any,
    field_name: str,
    maximum: int,
    *,
    empty: bool = False,
    allow_raw_endpoint: bool = False,
) -> str:
    if not isinstance(value, str):
        _fail("%s must be a string" % field_name)
    stripped = value.strip()
    if value != stripped and stripped:
        _fail("%s must not contain surrounding whitespace" % field_name)
    if empty:
        value = stripped
    else:
        value = stripped
        if not value:
            _fail("%s must not be empty" % field_name)
    if len(value.encode("utf-8")) > maximum:
        _fail("%s exceeds %d UTF-8 bytes" % (field_name, maximum))
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _fail("%s contains control characters" % field_name)
    if value and (is_secret_value(value) or contains_bearer_url(value)):
        _fail("%s contains credential-shaped data" % field_name)
    if value and not allow_raw_endpoint and contains_raw_endpoint(value):
        _fail("%s must not embed a raw endpoint or URL" % field_name)
    return value


def _name(value: Any, field_name: str = "name") -> str:
    value = _text(value, field_name, 64).casefold()
    if not _NAME.fullmatch(value):
        _fail("%s is not a canonical name" % field_name)
    return value


def _reason_code(value: Any) -> str:
    value = _text(value, "reason_code", MAX_REASON_CODE_BYTES).casefold()
    if not _REASON.fullmatch(value):
        _fail("reason_code is not canonical")
    return value


def _enum(value: Any, enum_type: Type[_T], field_name: str) -> _T:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        _fail("%s must be a string" % field_name)
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError:
        _fail("unknown %s: %r" % (field_name, value))
    raise AssertionError("unreachable")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("%s must be an integer" % field_name)
    if value < 0:
        _fail("%s must be non-negative" % field_name)
    if value > MAX_ABS_INTEGER:
        _fail("value overflows the allowed bound")
    return value


def _optional_non_negative_int(
    value: Any, field_name: str, *, maximum: Optional[int] = None
) -> Optional[int]:
    if value is None:
        return None
    parsed = _non_negative_int(value, field_name)
    if maximum is not None and parsed > maximum:
        _fail("%s exceeds maximum %d" % (field_name, maximum))
    return parsed


def _confidence_micros(value: Any) -> int:
    parsed = _non_negative_int(value, "confidence_micros")
    if parsed > MAX_CONFIDENCE:
        _fail("confidence_micros must be between 0 and %d" % MAX_CONFIDENCE)
    return parsed


def _timestamp(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            _fail("%s must not be empty" % field_name)
        if not _RFC3339.fullmatch(raw):
            _fail("%s must be an RFC 3339 timestamp" % field_name)
        try:
            parsed = datetime.fromisoformat(
                raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            )
        except ValueError:
            _fail("%s must be an RFC 3339 timestamp" % field_name)
    else:
        _fail("%s must be an RFC 3339 string" % field_name)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail("%s must include a timezone" % field_name)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _strict_mapping(
    data: Any,
    allowed: Iterable[str],
    required: Iterable[str],
    name: str,
) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        _fail("%s must be an object" % name)
    keys = set(data)
    if any(not isinstance(key, str) for key in keys):
        _fail("%s keys must be strings" % name)
    allowed_set = set(allowed)
    unknown = keys - allowed_set
    if unknown:
        _fail("%s has unknown fields: %s" % (name, ", ".join(sorted(unknown))))
    missing = set(required) - keys
    if missing:
        _fail("%s missing required fields: %s" % (name, ", ".join(sorted(missing))))
    for key in keys:
        if is_secret_key(key):
            _fail("credential-bearing field is forbidden: %s" % key)
    return dict(data)


def _currency(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = _text(value, "currency", MAX_CURRENCY_BYTES).upper()
    if not _CURRENCY.fullmatch(text):
        _fail("currency must be a 3-letter ISO code")
    return text


def _catalog_id(value: Any, field_name: str) -> str:
    try:
        return require_catalog_id(value, field_name)
    except UsageIdentityError as exc:
        _fail(str(exc))
    raise AssertionError("unreachable")


def _pseudonym(
    value: Any, field_name: str, *, optional: bool = True
) -> Optional[str]:
    if value is None:
        if optional:
            return None
        _fail("%s is required" % field_name)
    text = _text(value, field_name, 128)
    if not _PSEUDONYM_OR_ID.fullmatch(text):
        _fail("%s is not a stable pseudonym or catalog identity" % field_name)
    return text


def _reason_codes(values: Any) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        _fail("reason_codes must be an array")
    if len(values) > MAX_REASON_CODES:
        _fail("reason_codes exceeds maximum count")
    return tuple(sorted({_reason_code(item) for item in values}))


def _normalize_labels(value: Any) -> Tuple[Tuple[str, str], ...]:
    if value is None or value == () or value == {}:
        return ()
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, (list, tuple)):
        items = []
        for entry in value:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                items.append((entry[0], entry[1]))
            else:
                _fail("labels entries must be name/value pairs")
    else:
        _fail("labels must be an object or array")
    if len(items) > MAX_LABELS:
        _fail("labels exceeds maximum count")
    normalized = []
    for key, item in items:
        name = _name(key, "label")
        if is_secret_key(name):
            _fail("credential-bearing label keys are forbidden")
        normalized.append((name, _text(item, "label_value", MAX_STRING_BYTES)))
    return tuple(sorted({pair for pair in normalized}, key=lambda pair: pair[0]))


def _validate_dimension_window(
    dimension: UsageDimension, window: "LimitWindow"
) -> None:
    if window.kind is WindowKind.CONCURRENT:
        if dimension not in _CONCURRENT_DIMENSIONS:
            _fail(
                "invalid unit/window combination: concurrent requires concurrent dimension"
            )
        return
    if dimension in _CONCURRENT_DIMENSIONS:
        if window.kind not in (WindowKind.CONCURRENT, WindowKind.LIFETIME):
            _fail(
                "invalid unit/window combination: concurrent dimensions require concurrent or lifetime window"
            )


def _limit_tuple(values: Any, field_name: str = "limits") -> Tuple["UsageLimit", ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        _fail("%s must be an array" % field_name)
    if len(values) > MAX_LIMITS:
        _fail("%s exceeds maximum count" % field_name)
    parsed = []
    for item in values:
        if isinstance(item, UsageLimit):
            parsed.append(item)
        else:
            parsed.append(UsageLimit.from_dict(item))
    return tuple(sorted(parsed, key=lambda item: item.limit_id or ""))


def _candidate_tuple(
    values: Any, field_name: str
) -> Tuple["ResolutionCandidate", ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        _fail("%s must be an array" % field_name)
    if len(values) > MAX_CANDIDATES:
        _fail("%s exceeds maximum count" % field_name)
    parsed = []
    for item in values:
        if isinstance(item, ResolutionCandidate):
            parsed.append(item)
        else:
            parsed.append(ResolutionCandidate.from_dict(item))
    return tuple(sorted(parsed, key=lambda item: (item.rank, item.binding_id)))


def _ranking_inputs(
    values: Any,
) -> Tuple[Tuple[str, Union[int, float, str, bool, None]], ...]:
    if values is None or values == () or values == {}:
        return ()
    pairs = []
    if isinstance(values, Mapping):
        pairs = list(values.items())
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for entry in values:
            if isinstance(entry, Mapping):
                if "name" not in entry or "value" not in entry:
                    _fail("ranking_inputs entries must be name/value pairs")
                pairs.append((entry["name"], entry["value"]))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                pairs.append((entry[0], entry[1]))
            else:
                _fail("ranking_inputs entries must be name/value pairs")
    else:
        _fail("ranking_inputs must be an array or object")
    if len(pairs) > MAX_RANKING_INPUTS:
        _fail("ranking_inputs exceeds maximum count")
    normalized = []
    for key, item in pairs:
        name = _name(key, "ranking_input")
        if is_secret_key(name):
            _fail("credential-bearing ranking input names are forbidden")
        if item is not None and not isinstance(item, (bool, int, float, str)):
            _fail("ranking input values must be scalars")
        if isinstance(item, bool):
            pass
        elif isinstance(item, int):
            if abs(item) > MAX_ABS_INTEGER:
                _fail("ranking input integer overflows")
        elif isinstance(item, float):
            if item != item or item in (float("inf"), float("-inf")):
                _fail("ranking input float must be finite")
        elif isinstance(item, str):
            item = _text(item, "ranking_input_value", MAX_STRING_BYTES)
        normalized.append((name, item))
    return tuple(sorted(normalized, key=lambda pair: pair[0]))


@dataclass(frozen=True)
class Quantity:
    """A non-negative amount that may also be unknown or unlimited.

    ``unknown`` is never treated as ``unlimited``.  Finite values must be
    non-negative integers within the 64-bit bound.
    """

    kind: QuantityKind = QuantityKind.UNKNOWN
    value: Optional[int] = None

    def __post_init__(self) -> None:
        kind = _enum(self.kind, QuantityKind, "kind")
        object.__setattr__(self, "kind", kind)
        if kind is QuantityKind.FINITE:
            if self.value is None:
                _fail("finite quantity requires value")
            object.__setattr__(self, "value", _non_negative_int(self.value, "value"))
        else:
            if self.value is not None:
                _fail("%s quantity must not set value" % kind.value)
            object.__setattr__(self, "value", None)

    @classmethod
    def finite(cls, value: int) -> "Quantity":
        return cls(kind=QuantityKind.FINITE, value=value)

    @classmethod
    def unknown(cls) -> "Quantity":
        return cls(kind=QuantityKind.UNKNOWN)

    @classmethod
    def unlimited(cls) -> "Quantity":
        return cls(kind=QuantityKind.UNLIMITED)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"kind": self.kind.value}
        if self.kind is QuantityKind.FINITE:
            payload["value"] = self.value
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> "Quantity":
        values = _strict_mapping(data, ("kind", "value"), ("kind",), "Quantity")
        return cls(kind=values.get("kind"), value=values.get("value"))

    def __int__(self) -> int:
        if self.kind is not QuantityKind.FINITE or self.value is None:
            raise TypeError("only finite quantities convert to int")
        return self.value


@dataclass(frozen=True)
class UsageVectorEntry:
    """One typed dimension contribution, optionally currency-tagged for cost."""

    dimension: UsageDimension
    amount: Quantity
    currency: Optional[str] = None

    def __post_init__(self) -> None:
        dimension = _enum(self.dimension, UsageDimension, "dimension")
        object.__setattr__(self, "dimension", dimension)
        amount = (
            self.amount
            if isinstance(self.amount, Quantity)
            else Quantity.from_dict(self.amount)
        )
        object.__setattr__(self, "amount", amount)
        currency = _currency(self.currency)
        object.__setattr__(self, "currency", currency)
        if dimension is UsageDimension.COST_MICROS:
            if currency is None:
                _fail("currency is required for cost_micros")
        elif currency is not None:
            _fail("currency is only valid for cost_micros")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "amount": self.amount.to_dict(),
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageVectorEntry":
        values = _strict_mapping(
            data,
            ("dimension", "amount", "currency"),
            ("dimension", "amount"),
            "UsageVectorEntry",
        )
        return cls(
            dimension=values.get("dimension"),
            amount=values.get("amount"),
            currency=values.get("currency"),
        )


@dataclass(frozen=True)
class UsageVector:
    """Bounded multi-dimension usage vector."""

    entries: Tuple[UsageVectorEntry, ...] = ()

    def __post_init__(self) -> None:
        raw = self.entries
        if raw is None:
            raw = ()
        if isinstance(raw, (str, bytes, Mapping)) or not isinstance(
            raw, (Sequence, set, frozenset)
        ):
            _fail("entries must be an array")
        if len(raw) > MAX_VECTOR_ENTRIES:
            _fail("usage vector exceeds maximum entries")
        parsed = []
        for item in raw:
            if isinstance(item, UsageVectorEntry):
                parsed.append(item)
            else:
                parsed.append(UsageVectorEntry.from_dict(item))
        parsed.sort(
            key=lambda entry: (
                entry.dimension.value,
                entry.currency or "",
                entry.amount.kind.value,
                -1 if entry.amount.value is None else entry.amount.value,
            )
        )
        unique = []
        seen: Dict[Tuple[str, Optional[str]], Quantity] = {}
        for entry in parsed:
            key = (entry.dimension.value, entry.currency)
            if key in seen:
                if seen[key] != entry.amount:
                    _fail(
                        "duplicate conflicting entry for dimension %s"
                        % entry.dimension.value
                    )
                continue
            seen[key] = entry.amount
            unique.append(entry)
        object.__setattr__(self, "entries", tuple(unique))

    def get(
        self, dimension: UsageDimension, *, currency: Optional[str] = None
    ) -> Optional[UsageVectorEntry]:
        for entry in self.entries:
            if entry.dimension == dimension and entry.currency == currency:
                return entry
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, data: Any) -> "UsageVector":
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
            return cls(entries=tuple(data))
        values = _strict_mapping(data, ("entries",), (), "UsageVector")
        return cls(entries=tuple(values.get("entries") or ()))

    @classmethod
    def of(cls, **kwargs: Any) -> "UsageVector":
        """Build a vector from dimension-name kwargs with finite amounts."""

        currency = kwargs.pop("currency", None)
        entries = []
        for name, amount in kwargs.items():
            dimension = UsageDimension(name)
            if dimension is UsageDimension.COST_MICROS:
                entries.append(
                    UsageVectorEntry(
                        dimension=dimension,
                        amount=Quantity.finite(int(amount)),
                        currency=currency,
                    )
                )
            else:
                entries.append(
                    UsageVectorEntry(
                        dimension=dimension, amount=Quantity.finite(int(amount))
                    )
                )
        return cls(entries=tuple(entries))


@dataclass(frozen=True)
class LimitWindow:
    """Time or concurrency window that bounds a :class:`UsageLimit`."""

    kind: WindowKind
    length_ms: Optional[int] = None
    anchor_at: Optional[str] = None
    reset_at: Optional[str] = None
    refill_per_second: Optional[int] = None
    burst: Optional[int] = None
    safety_reserve: Optional[int] = None

    def __post_init__(self) -> None:
        kind = _enum(self.kind, WindowKind, "kind")
        object.__setattr__(self, "kind", kind)
        length = _optional_non_negative_int(
            self.length_ms, "length_ms", maximum=MAX_WINDOW_MS
        )
        anchor = _timestamp(self.anchor_at, "anchor_at")
        reset = _timestamp(self.reset_at, "reset_at")
        refill = _optional_non_negative_int(self.refill_per_second, "refill_per_second")
        burst = _optional_non_negative_int(self.burst, "burst")
        reserve = _optional_non_negative_int(self.safety_reserve, "safety_reserve")

        if kind is WindowKind.CONCURRENT:
            if length is not None:
                _fail("concurrent windows must not set length_ms")
            if refill is not None:
                _fail("concurrent windows must not set refill_per_second")
            if anchor is not None or reset is not None:
                _fail("concurrent windows must not set anchor_at or reset_at")
        elif kind is WindowKind.LIFETIME:
            if length is not None:
                _fail("lifetime windows must not set length_ms")
            if refill is not None:
                _fail("lifetime windows must not set refill_per_second")
        elif kind is WindowKind.TOKEN_BUCKET:
            if refill is None:
                _fail("token_bucket windows require refill_per_second")
            if burst is None:
                _fail("token_bucket windows require burst")
            if length is None and reset is None:
                _fail("token_bucket windows require length_ms or reset_at")
        elif kind in (WindowKind.FIXED, WindowKind.SLIDING):
            if length is None:
                _fail("%s windows require length_ms" % kind.value)
            if refill is not None:
                _fail("%s windows must not set refill_per_second" % kind.value)
        elif kind is WindowKind.BILLING:
            if anchor is None and reset is None:
                _fail("billing windows require anchor_at or reset_at")
            if refill is not None:
                _fail("billing windows must not set refill_per_second")

        if anchor is not None and reset is not None and reset < anchor:
            _fail("reset_at must not precede anchor_at")

        object.__setattr__(self, "length_ms", length)
        object.__setattr__(self, "anchor_at", anchor)
        object.__setattr__(self, "reset_at", reset)
        object.__setattr__(self, "refill_per_second", refill)
        object.__setattr__(self, "burst", burst)
        object.__setattr__(self, "safety_reserve", reserve)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "length_ms": self.length_ms,
            "anchor_at": self.anchor_at,
            "reset_at": self.reset_at,
            "refill_per_second": self.refill_per_second,
            "burst": self.burst,
            "safety_reserve": self.safety_reserve,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LimitWindow":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(data, fields, ("kind",), "LimitWindow")
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class Provenance:
    """Bounded provenance for a limit or observation."""

    source: LimitSource = LimitSource.UNKNOWN
    parser_version: str = "1.0"
    observed_at: Optional[str] = None
    expires_at: Optional[str] = None
    digest: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _enum(self.source, LimitSource, "source"))
        object.__setattr__(
            self,
            "parser_version",
            _text(self.parser_version, "parser_version", 64),
        )
        observed = _timestamp(self.observed_at, "observed_at")
        expires = _timestamp(self.expires_at, "expires_at")
        if observed is not None and expires is not None and expires < observed:
            _fail("expires_at must be later than observed_at")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "expires_at", expires)
        digest = self.digest
        if digest is not None:
            digest = _text(digest, "digest", 64)
            if not _DIGEST.fullmatch(digest):
                _fail("digest is not a canonical provenance digest")
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "parser_version": self.parser_version,
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "digest": self.digest,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Provenance":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(data, fields, (), "Provenance")
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class EndpointUsageScope:
    """Exact non-secret owner of a limit or usage counter.

    Binds provider, deployment or endpoint fingerprint, protocol, operation,
    optional provider-scoped model/account/project/region/organization, and a
    keyed local credential-configuration pseudonym.
    """

    provider_id: str
    protocol: ProtocolKind
    operation: str
    schema_version: str = SCHEMA_VERSION
    identity_policy_version: str = IDENTITY_POLICY_VERSION
    deployment_id: Optional[str] = None
    endpoint_fingerprint: Optional[str] = None
    model_id: Optional[str] = None
    account_pseudonym: Optional[str] = None
    project_pseudonym: Optional[str] = None
    organization_pseudonym: Optional[str] = None
    region: Optional[str] = None
    credential_pseudonym: Optional[str] = None
    unknown_scope: bool = False
    scope_id: Optional[str] = None
    labels: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        policy = _text(self.identity_policy_version, "identity_policy_version", 32)
        if policy != IDENTITY_POLICY_VERSION:
            _fail("unsupported identity_policy_version")
        object.__setattr__(self, "identity_policy_version", policy)
        provider_id = _catalog_id(self.provider_id, "provider_id")
        object.__setattr__(self, "provider_id", provider_id)
        protocol = _enum(self.protocol, ProtocolKind, "protocol")
        object.__setattr__(self, "protocol", protocol)
        operation = _text(self.operation, "operation", 64).casefold()
        if not _OPERATION.fullmatch(operation):
            _fail("operation is not canonical")
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.unknown_scope, bool):
            _fail("unknown_scope must be a boolean")
        deployment_id = _pseudonym(self.deployment_id, "deployment_id")
        endpoint_fp = _pseudonym(self.endpoint_fingerprint, "endpoint_fingerprint")
        model_id = _pseudonym(self.model_id, "model_id")
        account = _pseudonym(self.account_pseudonym, "account_pseudonym")
        project = _pseudonym(self.project_pseudonym, "project_pseudonym")
        organization = _pseudonym(
            self.organization_pseudonym, "organization_pseudonym"
        )
        region = self.region
        if region is not None:
            region = _text(region, "region", 64)
            if not _REGION.fullmatch(region):
                _fail("region is not canonical")
        credential = _pseudonym(self.credential_pseudonym, "credential_pseudonym")
        if self.unknown_scope:
            if deployment_id is not None or endpoint_fp is not None:
                _fail("unknown_scope must not set deployment or endpoint identity")
        else:
            if deployment_id is None and endpoint_fp is None:
                _fail("scope requires deployment_id or endpoint_fingerprint")
            if credential is None:
                _fail("credential_pseudonym is required for established scopes")
        labels = _normalize_labels(self.labels)
        object.__setattr__(self, "deployment_id", deployment_id)
        object.__setattr__(self, "endpoint_fingerprint", endpoint_fp)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "account_pseudonym", account)
        object.__setattr__(self, "project_pseudonym", project)
        object.__setattr__(self, "organization_pseudonym", organization)
        object.__setattr__(self, "region", region)
        object.__setattr__(self, "credential_pseudonym", credential)
        object.__setattr__(self, "labels", labels)
        components = scope_identity_components(
            provider_id=provider_id,
            protocol=protocol.value,
            operation=operation,
            deployment_id=deployment_id,
            endpoint_fingerprint_value=endpoint_fp,
            model_id=model_id,
            account_pseudonym_value=account,
            project_pseudonym_value=project,
            organization_pseudonym_value=organization,
            region=region,
            credential_pseudonym_value=credential,
            unknown_scope=self.unknown_scope,
        )
        expected = endpoint_usage_scope_identity(components)
        if self.scope_id is not None and self.scope_id != expected:
            _fail("scope_id does not match canonical identity fields")
        object.__setattr__(self, "scope_id", expected)

    def to_dict(self) -> Dict[str, Any]:
        labels = {key: value for key, value in self.labels}
        return {
            "schema_version": self.schema_version,
            "identity_policy_version": self.identity_policy_version,
            "provider_id": self.provider_id,
            "deployment_id": self.deployment_id,
            "endpoint_fingerprint": self.endpoint_fingerprint,
            "protocol": self.protocol.value,
            "operation": self.operation,
            "model_id": self.model_id,
            "account_pseudonym": self.account_pseudonym,
            "project_pseudonym": self.project_pseudonym,
            "organization_pseudonym": self.organization_pseudonym,
            "region": self.region,
            "credential_pseudonym": self.credential_pseudonym,
            "unknown_scope": self.unknown_scope,
            "scope_id": self.scope_id,
            "labels": labels,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "EndpointUsageScope":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(data, fields, ("provider_id", "protocol", "operation"), "EndpointUsageScope")
        labels = values.get("labels")
        if isinstance(labels, Mapping):
            values = dict(values)
            values["labels"] = tuple(labels.items())
        return cls(**{key: values.get(key) for key in fields})

    @property
    def cid(self) -> str:
        return content_cid(self.to_dict())


@dataclass(frozen=True)
class UsageLimit:
    """A single typed limit attached to one :class:`EndpointUsageScope`."""

    scope_id: str
    dimension: UsageDimension
    ceiling: Quantity
    window: LimitWindow
    schema_version: str = SCHEMA_VERSION
    remaining: Quantity = field(default_factory=Quantity.unknown)
    used: Quantity = field(default_factory=Quantity.unknown)
    enforcement: LimitEnforcement = LimitEnforcement.HARD
    confidence_micros: int = 0
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    provenance: Provenance = field(default_factory=Provenance)
    currency: Optional[str] = None
    limit_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        scope_id = _pseudonym(self.scope_id, "scope_id", optional=False)
        object.__setattr__(self, "scope_id", scope_id)
        dimension = _enum(self.dimension, UsageDimension, "dimension")
        object.__setattr__(self, "dimension", dimension)
        ceiling = (
            self.ceiling
            if isinstance(self.ceiling, Quantity)
            else Quantity.from_dict(self.ceiling)
        )
        remaining = (
            self.remaining
            if isinstance(self.remaining, Quantity)
            else Quantity.from_dict(self.remaining)
        )
        used = (
            self.used if isinstance(self.used, Quantity) else Quantity.from_dict(self.used)
        )
        window = (
            self.window
            if isinstance(self.window, LimitWindow)
            else LimitWindow.from_dict(self.window)
        )
        object.__setattr__(self, "ceiling", ceiling)
        object.__setattr__(self, "remaining", remaining)
        object.__setattr__(self, "used", used)
        object.__setattr__(self, "window", window)
        object.__setattr__(
            self, "enforcement", _enum(self.enforcement, LimitEnforcement, "enforcement")
        )
        object.__setattr__(
            self, "confidence_micros", _confidence_micros(self.confidence_micros)
        )
        object.__setattr__(
            self, "confidence", _enum(self.confidence, ConfidenceLevel, "confidence")
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, Provenance)
            else Provenance.from_dict(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        currency = _currency(self.currency)
        if dimension is UsageDimension.COST_MICROS:
            if currency is None:
                _fail("currency is required for cost_micros")
        elif currency is not None:
            _fail("currency is only valid for cost_micros")
        object.__setattr__(self, "currency", currency)
        _validate_dimension_window(dimension, window)
        if (
            ceiling.kind is QuantityKind.FINITE
            and remaining.kind is QuantityKind.FINITE
            and remaining.value is not None
            and ceiling.value is not None
            and remaining.value > ceiling.value
        ):
            _fail("remaining exceeds ceiling")
        if (
            ceiling.kind is QuantityKind.FINITE
            and used.kind is QuantityKind.FINITE
            and used.value is not None
            and ceiling.value is not None
            and used.value > ceiling.value
        ):
            _fail("used exceeds ceiling")
        if (
            ceiling.kind is QuantityKind.UNKNOWN
            and remaining.kind is QuantityKind.UNLIMITED
        ):
            _fail("unknown ceiling cannot imply unlimited remaining")
        expected_limit_id = stable_id(
            "limit",
            scope_id,
            dimension.value,
            window.to_dict(),
            currency,
            self.enforcement.value,
        )
        if self.limit_id is not None and self.limit_id != expected_limit_id:
            _fail("limit_id does not match canonical identity fields")
        object.__setattr__(self, "limit_id", expected_limit_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "limit_id": self.limit_id,
            "scope_id": self.scope_id,
            "dimension": self.dimension.value,
            "ceiling": self.ceiling.to_dict(),
            "remaining": self.remaining.to_dict(),
            "used": self.used.to_dict(),
            "window": self.window.to_dict(),
            "enforcement": self.enforcement.value,
            "confidence_micros": self.confidence_micros,
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict(),
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageLimit":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(
            data, fields, ("scope_id", "dimension", "ceiling", "window"), "UsageLimit"
        )
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class DimensionHeadroom:
    """Headroom for one dimension within a snapshot."""

    dimension: UsageDimension
    available: Quantity
    ceiling: Quantity
    reserved: Quantity = field(default_factory=lambda: Quantity.finite(0))
    currency: Optional[str] = None
    state: AvailabilityState = AvailabilityState.UNKNOWN
    next_eligible_at: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dimension", _enum(self.dimension, UsageDimension, "dimension")
        )
        for name in ("available", "ceiling", "reserved"):
            value = getattr(self, name)
            if not isinstance(value, Quantity):
                value = Quantity.from_dict(value)
            object.__setattr__(self, name, value)
        currency = _currency(self.currency)
        if self.dimension is UsageDimension.COST_MICROS:
            if currency is None:
                _fail("currency is required for cost_micros")
        elif currency is not None:
            _fail("currency is only valid for cost_micros")
        object.__setattr__(self, "currency", currency)
        object.__setattr__(
            self, "state", _enum(self.state, AvailabilityState, "state")
        )
        object.__setattr__(
            self, "next_eligible_at", _timestamp(self.next_eligible_at, "next_eligible_at")
        )
        if (
            self.ceiling.kind is QuantityKind.UNKNOWN
            and self.available.kind is QuantityKind.UNLIMITED
        ):
            _fail("unknown ceiling cannot imply unlimited available")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "available": self.available.to_dict(),
            "ceiling": self.ceiling.to_dict(),
            "reserved": self.reserved.to_dict(),
            "currency": self.currency,
            "state": self.state.value,
            "next_eligible_at": self.next_eligible_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DimensionHeadroom":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(
            data, fields, ("dimension", "available", "ceiling"), "DimensionHeadroom"
        )
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class UsageEstimate:
    """Conservative pre-invocation estimate of required units."""

    scope_id: str
    operation: str
    requested: UsageVector
    schema_version: str = SCHEMA_VERSION
    method: EstimateMethod = EstimateMethod.CONSERVATIVE
    method_version: str = "1.0"
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    confidence_micros: int = 100_000
    estimated_at: Optional[str] = None
    estimate_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        scope_id = _pseudonym(self.scope_id, "scope_id", optional=False)
        object.__setattr__(self, "scope_id", scope_id)
        operation = _text(self.operation, "operation", 64).casefold()
        if not _OPERATION.fullmatch(operation):
            _fail("operation is not canonical")
        object.__setattr__(self, "operation", operation)
        requested = (
            self.requested
            if isinstance(self.requested, UsageVector)
            else UsageVector.from_dict(self.requested)
        )
        if not requested.entries:
            _fail("requested estimate vector must not be empty")
        for entry in requested.entries:
            if entry.amount.kind is QuantityKind.UNLIMITED:
                _fail("estimate amounts must not be unlimited")
            if entry.amount.kind is QuantityKind.UNKNOWN:
                _fail("estimate amounts must be finite (unknown is not a request)")
        object.__setattr__(self, "requested", requested)
        object.__setattr__(
            self, "method", _enum(self.method, EstimateMethod, "method")
        )
        object.__setattr__(
            self,
            "method_version",
            _text(self.method_version, "method_version", 64),
        )
        object.__setattr__(
            self, "confidence", _enum(self.confidence, ConfidenceLevel, "confidence")
        )
        object.__setattr__(
            self, "confidence_micros", _confidence_micros(self.confidence_micros)
        )
        object.__setattr__(
            self, "estimated_at", _timestamp(self.estimated_at, "estimated_at")
        )
        expected = stable_id(
            "uest",
            scope_id,
            operation,
            requested.to_dict(),
            self.method.value,
            self.method_version,
        )
        if self.estimate_id is not None and self.estimate_id != expected:
            _fail("estimate_id does not match canonical identity fields")
        object.__setattr__(self, "estimate_id", expected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "estimate_id": self.estimate_id,
            "scope_id": self.scope_id,
            "operation": self.operation,
            "requested": self.requested.to_dict(),
            "method": self.method.value,
            "method_version": self.method_version,
            "confidence": self.confidence.value,
            "confidence_micros": self.confidence_micros,
            "estimated_at": self.estimated_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageEstimate":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(
            data, fields, ("scope_id", "operation", "requested"), "UsageEstimate"
        )
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class ProviderUsageObservation:
    """Bounded metadata observed from one exact local invocation response."""

    scope_id: str
    request_id: str
    schema_version: str = SCHEMA_VERSION
    usage: UsageVector = field(default_factory=UsageVector)
    limits: Tuple[UsageLimit, ...] = ()
    retry_after_ms: Optional[int] = None
    reset_at: Optional[str] = None
    http_status: Optional[int] = None
    provider_request_id: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_micros: int = 500_000
    provenance: Provenance = field(default_factory=Provenance)
    reason_codes: Tuple[str, ...] = ()
    observation_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        scope_id = _pseudonym(self.scope_id, "scope_id", optional=False)
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(
            self, "request_id", _text(self.request_id, "request_id", MAX_STRING_BYTES)
        )
        usage = (
            self.usage
            if isinstance(self.usage, UsageVector)
            else UsageVector.from_dict(self.usage)
        )
        object.__setattr__(self, "usage", usage)
        limits = _limit_tuple(self.limits)
        object.__setattr__(self, "limits", limits)
        object.__setattr__(
            self,
            "retry_after_ms",
            _optional_non_negative_int(
                self.retry_after_ms, "retry_after_ms", maximum=MAX_WINDOW_MS
            ),
        )
        object.__setattr__(self, "reset_at", _timestamp(self.reset_at, "reset_at"))
        status = self.http_status
        if status is not None:
            status = _non_negative_int(status, "http_status")
            if status < 100 or status > 599:
                _fail("http_status must be a valid HTTP status")
        object.__setattr__(self, "http_status", status)
        provider_request_id = self.provider_request_id
        if provider_request_id is not None:
            provider_request_id = _text(
                provider_request_id, "provider_request_id", MAX_STRING_BYTES
            )
        object.__setattr__(self, "provider_request_id", provider_request_id)
        object.__setattr__(
            self, "confidence", _enum(self.confidence, ConfidenceLevel, "confidence")
        )
        object.__setattr__(
            self, "confidence_micros", _confidence_micros(self.confidence_micros)
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, Provenance)
            else Provenance.from_dict(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        material = content_cid(
            {
                "scope_id": scope_id,
                "request_id": self.request_id,
                "usage": usage.to_dict(),
                "limits": [item.limit_id for item in limits],
                "http_status": status,
                "provider_request_id": provider_request_id,
            }
        )
        expected_id = stable_id("uobs", material)
        if self.observation_id is not None and self.observation_id != expected_id:
            _fail("observation_id does not match canonical identity fields")
        object.__setattr__(self, "observation_id", expected_id)
        assert_no_prompt_media_or_output(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "scope_id": self.scope_id,
            "request_id": self.request_id,
            "usage": self.usage.to_dict(),
            "limits": [item.to_dict() for item in self.limits],
            "retry_after_ms": self.retry_after_ms,
            "reset_at": self.reset_at,
            "http_status": self.http_status,
            "provider_request_id": self.provider_request_id,
            "confidence": self.confidence.value,
            "confidence_micros": self.confidence_micros,
            "provenance": self.provenance.to_dict(),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ProviderUsageObservation":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(
            data, fields, ("scope_id", "request_id"), "ProviderUsageObservation"
        )
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class UsageEvent:
    """Immutable, content-addressed ledger event."""

    kind: UsageEventKind
    scope_id: str
    schema_version: str = SCHEMA_VERSION
    event_id: Optional[str] = None
    sequence: Optional[int] = None
    occurred_at: Optional[str] = None
    request_id: Optional[str] = None
    reservation_id: Optional[str] = None
    estimate_id: Optional[str] = None
    observation_id: Optional[str] = None
    supersedes_event_id: Optional[str] = None
    units: UsageVector = field(default_factory=UsageVector)
    reason_codes: Tuple[str, ...] = ()
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        object.__setattr__(self, "kind", _enum(self.kind, UsageEventKind, "kind"))
        object.__setattr__(
            self, "scope_id", _pseudonym(self.scope_id, "scope_id", optional=False)
        )
        sequence = self.sequence
        if sequence is not None:
            sequence = _non_negative_int(sequence, "sequence")
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(
            self, "occurred_at", _timestamp(self.occurred_at, "occurred_at")
        )
        for name in (
            "request_id",
            "reservation_id",
            "estimate_id",
            "observation_id",
            "supersedes_event_id",
        ):
            value = getattr(self, name)
            if value is not None:
                value = _text(value, name, 128)
            object.__setattr__(self, name, value)
        units = (
            self.units
            if isinstance(self.units, UsageVector)
            else UsageVector.from_dict(self.units)
        )
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        provenance = (
            self.provenance
            if isinstance(self.provenance, Provenance)
            else Provenance.from_dict(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        if self.kind is UsageEventKind.CORRECTION and not self.supersedes_event_id:
            _fail("correction events require supersedes_event_id")
        expected = event_identity(
            {
                "kind": self.kind.value,
                "scope_id": self.scope_id,
                "sequence": self.sequence,
                "occurred_at": self.occurred_at,
                "request_id": self.request_id,
                "reservation_id": self.reservation_id,
                "estimate_id": self.estimate_id,
                "observation_id": self.observation_id,
                "supersedes_event_id": self.supersedes_event_id,
                "units": units.to_dict(),
                "reason_codes": list(self.reason_codes),
            }
        )
        if self.event_id is not None and self.event_id != expected:
            _fail("event_id does not match canonical identity fields")
        object.__setattr__(self, "event_id", expected)
        assert_no_prompt_media_or_output(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "kind": self.kind.value,
            "scope_id": self.scope_id,
            "sequence": self.sequence,
            "occurred_at": self.occurred_at,
            "request_id": self.request_id,
            "reservation_id": self.reservation_id,
            "estimate_id": self.estimate_id,
            "observation_id": self.observation_id,
            "supersedes_event_id": self.supersedes_event_id,
            "units": self.units.to_dict(),
            "reason_codes": list(self.reason_codes),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageEvent":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(data, fields, ("kind", "scope_id"), "UsageEvent")
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class UsageReservation:
    """Atomic multi-dimension reservation against one scope."""

    scope_id: str
    reserved: UsageVector
    state: ReservationState
    schema_version: str = SCHEMA_VERSION
    reservation_id: Optional[str] = None
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    owner_id: Optional[str] = None
    lease_id: Optional[str] = None
    fence: Optional[int] = None
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    estimate_id: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        object.__setattr__(
            self, "scope_id", _pseudonym(self.scope_id, "scope_id", optional=False)
        )
        reserved = (
            self.reserved
            if isinstance(self.reserved, UsageVector)
            else UsageVector.from_dict(self.reserved)
        )
        if not reserved.entries:
            _fail("reserved vector must not be empty")
        for entry in reserved.entries:
            if entry.amount.kind is not QuantityKind.FINITE:
                _fail("reserved amounts must be finite")
        object.__setattr__(self, "reserved", reserved)
        object.__setattr__(
            self, "state", _enum(self.state, ReservationState, "state")
        )
        for name in ("request_id", "idempotency_key", "owner_id", "lease_id", "estimate_id"):
            value = getattr(self, name)
            if value is not None:
                value = _text(value, name, 128)
            object.__setattr__(self, name, value)
        fence = self.fence
        if fence is not None:
            fence = _non_negative_int(fence, "fence")
        object.__setattr__(self, "fence", fence)
        created = _timestamp(self.created_at, "created_at")
        expires = _timestamp(self.expires_at, "expires_at")
        if created is not None and expires is not None and expires < created:
            _fail("expires_at must not precede created_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        expected = reservation_identity(
            {
                "scope_id": self.scope_id,
                "reserved": reserved.to_dict(),
                "request_id": self.request_id,
                "idempotency_key": self.idempotency_key,
                "owner_id": self.owner_id,
                "lease_id": self.lease_id,
                "fence": self.fence,
            }
        )
        if self.reservation_id is not None and self.reservation_id != expected:
            _fail("reservation_id does not match canonical identity fields")
        object.__setattr__(self, "reservation_id", expected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reservation_id": self.reservation_id,
            "scope_id": self.scope_id,
            "reserved": self.reserved.to_dict(),
            "state": self.state.value,
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "owner_id": self.owner_id,
            "lease_id": self.lease_id,
            "fence": self.fence,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "estimate_id": self.estimate_id,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageReservation":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(
            data, fields, ("scope_id", "reserved", "state"), "UsageReservation"
        )
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class UsageSnapshot:
    """Immutable materialized view of limits and in-flight reservations."""

    scope_id: str
    schema_version: str = SCHEMA_VERSION
    usage_revision: Optional[str] = None
    observed_at: Optional[str] = None
    fresh_until: Optional[str] = None
    state: AvailabilityState = AvailabilityState.UNKNOWN
    limits: Tuple[UsageLimit, ...] = ()
    headroom: Tuple[DimensionHeadroom, ...] = ()
    reservations: Tuple[UsageReservation, ...] = ()
    next_eligible_at: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        object.__setattr__(
            self, "scope_id", _pseudonym(self.scope_id, "scope_id", optional=False)
        )
        observed = _timestamp(self.observed_at, "observed_at")
        fresh = _timestamp(self.fresh_until, "fresh_until")
        if observed is not None and fresh is not None and fresh < observed:
            _fail("fresh_until must not precede observed_at")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "fresh_until", fresh)
        object.__setattr__(
            self, "state", _enum(self.state, AvailabilityState, "state")
        )
        limits = _limit_tuple(self.limits)
        object.__setattr__(self, "limits", limits)
        if isinstance(self.headroom, (str, bytes, Mapping)) or not isinstance(
            self.headroom, (Sequence, set, frozenset)
        ):
            if self.headroom in (None, ()):
                headroom: Tuple[DimensionHeadroom, ...] = ()
            else:
                _fail("headroom must be an array")
                headroom = ()
        else:
            if len(self.headroom) > MAX_VECTOR_ENTRIES:
                _fail("headroom exceeds maximum count")
            headroom = tuple(
                item
                if isinstance(item, DimensionHeadroom)
                else DimensionHeadroom.from_dict(item)
                for item in self.headroom
            )
            headroom = tuple(
                sorted(
                    headroom,
                    key=lambda item: (item.dimension.value, item.currency or ""),
                )
            )
        object.__setattr__(self, "headroom", headroom)
        if isinstance(self.reservations, (str, bytes, Mapping)) or not isinstance(
            self.reservations, (Sequence, set, frozenset)
        ):
            if self.reservations in (None, ()):
                reservations: Tuple[UsageReservation, ...] = ()
            else:
                _fail("reservations must be an array")
                reservations = ()
        else:
            if len(self.reservations) > MAX_RESERVATIONS:
                _fail("reservations exceeds maximum count")
            reservations = tuple(
                item
                if isinstance(item, UsageReservation)
                else UsageReservation.from_dict(item)
                for item in self.reservations
            )
            reservation_ids = [item.reservation_id for item in reservations]
            if len(reservation_ids) != len(set(reservation_ids)):
                _fail("reservations contains duplicates")
            reservations = tuple(
                sorted(reservations, key=lambda item: item.reservation_id or "")
            )
        object.__setattr__(self, "reservations", reservations)
        object.__setattr__(
            self,
            "next_eligible_at",
            _timestamp(self.next_eligible_at, "next_eligible_at"),
        )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        expected = snapshot_identity(
            {
                "scope_id": self.scope_id,
                "observed_at": self.observed_at,
                "state": self.state.value,
                "limits": [item.limit_id for item in limits],
                "reservations": [item.reservation_id for item in reservations],
                "reason_codes": list(self.reason_codes),
            }
        )
        if self.usage_revision is not None and self.usage_revision != expected:
            _fail("usage_revision does not match canonical identity fields")
        object.__setattr__(self, "usage_revision", expected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "usage_revision": self.usage_revision,
            "scope_id": self.scope_id,
            "observed_at": self.observed_at,
            "fresh_until": self.fresh_until,
            "state": self.state.value,
            "limits": [item.to_dict() for item in self.limits],
            "headroom": [item.to_dict() for item in self.headroom],
            "reservations": [item.to_dict() for item in self.reservations],
            "next_eligible_at": self.next_eligible_at,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageSnapshot":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(data, fields, ("scope_id",), "UsageSnapshot")
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class RoutingPolicy:
    """Explicit fallback and enforcement policy for usage-aware routing."""

    schema_version: str = SCHEMA_VERSION
    mode: RoutingMode = RoutingMode.OFF
    fallback: FallbackClass = FallbackClass.NONE
    max_attempts: int = 1
    deadline_ms: Optional[int] = None
    allow_wait: bool = False
    max_wait_ms: Optional[int] = None
    prefer_local: bool = False
    cost_ceiling_micros: Optional[int] = None
    cost_currency: Optional[str] = None
    policy_id: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        object.__setattr__(self, "mode", _enum(self.mode, RoutingMode, "mode"))
        object.__setattr__(
            self, "fallback", _enum(self.fallback, FallbackClass, "fallback")
        )
        object.__setattr__(
            self, "max_attempts", _non_negative_int(self.max_attempts, "max_attempts")
        )
        if self.max_attempts < 1:
            _fail("max_attempts must be at least 1")
        object.__setattr__(
            self,
            "deadline_ms",
            _optional_non_negative_int(
                self.deadline_ms, "deadline_ms", maximum=MAX_WINDOW_MS
            ),
        )
        if not isinstance(self.allow_wait, bool):
            _fail("allow_wait must be a boolean")
        if not isinstance(self.prefer_local, bool):
            _fail("prefer_local must be a boolean")
        object.__setattr__(
            self,
            "max_wait_ms",
            _optional_non_negative_int(
                self.max_wait_ms, "max_wait_ms", maximum=MAX_WINDOW_MS
            ),
        )
        if self.allow_wait and self.max_wait_ms is None:
            _fail("allow_wait requires max_wait_ms")
        object.__setattr__(
            self,
            "cost_ceiling_micros",
            _optional_non_negative_int(self.cost_ceiling_micros, "cost_ceiling_micros"),
        )
        cost_currency = _currency(self.cost_currency)
        if cost_currency is not None and self.cost_ceiling_micros is None:
            _fail("cost_currency requires cost_ceiling_micros")
        object.__setattr__(self, "cost_currency", cost_currency)
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        expected = stable_id(
            "upol",
            {
                "mode": self.mode.value,
                "fallback": self.fallback.value,
                "max_attempts": self.max_attempts,
                "deadline_ms": self.deadline_ms,
                "allow_wait": self.allow_wait,
                "max_wait_ms": self.max_wait_ms,
                "prefer_local": self.prefer_local,
                "cost_ceiling_micros": self.cost_ceiling_micros,
                "cost_currency": self.cost_currency,
            },
        )
        if self.policy_id is not None and self.policy_id != expected:
            _fail("policy_id does not match canonical identity fields")
        object.__setattr__(self, "policy_id", expected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "mode": self.mode.value,
            "fallback": self.fallback.value,
            "max_attempts": self.max_attempts,
            "deadline_ms": self.deadline_ms,
            "allow_wait": self.allow_wait,
            "max_wait_ms": self.max_wait_ms,
            "prefer_local": self.prefer_local,
            "cost_ceiling_micros": self.cost_ceiling_micros,
            "cost_currency": self.cost_currency,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RoutingPolicy":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(data, fields, (), "RoutingPolicy")
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class ResolutionCandidate:
    """One statically eligible binding with usage headroom facts."""

    binding_id: str
    scope_id: str
    rank: int = 0
    state: AvailabilityState = AvailabilityState.UNKNOWN
    headroom: Tuple[DimensionHeadroom, ...] = ()
    rejection_reasons: Tuple[str, ...] = ()
    ranking_inputs: Tuple[Tuple[str, Union[int, float, str, bool, None]], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binding_id", _pseudonym(self.binding_id, "binding_id", optional=False)
        )
        object.__setattr__(
            self, "scope_id", _pseudonym(self.scope_id, "scope_id", optional=False)
        )
        object.__setattr__(self, "rank", _non_negative_int(self.rank, "rank"))
        object.__setattr__(
            self, "state", _enum(self.state, AvailabilityState, "state")
        )
        if isinstance(self.headroom, (str, bytes, Mapping)) or not isinstance(
            self.headroom, (Sequence, set, frozenset)
        ):
            if self.headroom in (None, ()):
                headroom: Tuple[DimensionHeadroom, ...] = ()
            else:
                _fail("headroom must be an array")
                headroom = ()
        else:
            headroom = tuple(
                item
                if isinstance(item, DimensionHeadroom)
                else DimensionHeadroom.from_dict(item)
                for item in self.headroom
            )
            headroom = tuple(
                sorted(
                    headroom,
                    key=lambda item: (item.dimension.value, item.currency or ""),
                )
            )
        object.__setattr__(self, "headroom", headroom)
        object.__setattr__(
            self, "rejection_reasons", _reason_codes(self.rejection_reasons)
        )
        object.__setattr__(
            self, "ranking_inputs", _ranking_inputs(self.ranking_inputs)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "scope_id": self.scope_id,
            "rank": self.rank,
            "state": self.state.value,
            "headroom": [item.to_dict() for item in self.headroom],
            "rejection_reasons": list(self.rejection_reasons),
            "ranking_inputs": [
                {"name": name, "value": value} for name, value in self.ranking_inputs
            ],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ResolutionCandidate":
        values = _strict_mapping(
            data,
            (
                "binding_id",
                "scope_id",
                "rank",
                "state",
                "headroom",
                "rejection_reasons",
                "ranking_inputs",
            ),
            ("binding_id", "scope_id"),
            "ResolutionCandidate",
        )
        ranking = values.get("ranking_inputs")
        if isinstance(ranking, Sequence) and ranking and isinstance(ranking[0], Mapping):
            values = dict(values)
            values["ranking_inputs"] = [
                (item.get("name"), item.get("value")) for item in ranking
            ]
        return cls(
            binding_id=values.get("binding_id"),
            scope_id=values.get("scope_id"),
            rank=values.get("rank", 0),
            state=values.get("state", AvailabilityState.UNKNOWN),
            headroom=tuple(values.get("headroom") or ()),
            rejection_reasons=tuple(values.get("rejection_reasons") or ()),
            ranking_inputs=tuple(values.get("ranking_inputs") or ()),
        )


@dataclass(frozen=True)
class UsageAwareResolution:
    """Planning result over one catalog revision and one usage revision.

    Does not reserve capacity or invoke a provider.
    """

    catalog_revision: str
    usage_revision: str
    schema_version: str = SCHEMA_VERSION
    policy_id: Optional[str] = None
    candidates: Tuple[ResolutionCandidate, ...] = ()
    rejected: Tuple[ResolutionCandidate, ...] = ()
    selected_binding_id: Optional[str] = None
    next_eligible_at: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()
    resolution_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        object.__setattr__(
            self,
            "catalog_revision",
            _text(self.catalog_revision, "catalog_revision", 128),
        )
        object.__setattr__(
            self, "usage_revision", _text(self.usage_revision, "usage_revision", 128)
        )
        policy_id = self.policy_id
        if policy_id is not None:
            policy_id = _text(policy_id, "policy_id", 128)
        object.__setattr__(self, "policy_id", policy_id)
        candidates = _candidate_tuple(self.candidates, "candidates")
        rejected = _candidate_tuple(self.rejected, "rejected")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "rejected", rejected)
        selected = self.selected_binding_id
        if selected is not None:
            selected = _pseudonym(selected, "selected_binding_id", optional=False)
            if selected not in {item.binding_id for item in candidates}:
                _fail("selected_binding_id must reference a candidate")
        object.__setattr__(self, "selected_binding_id", selected)
        object.__setattr__(
            self,
            "next_eligible_at",
            _timestamp(self.next_eligible_at, "next_eligible_at"),
        )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        expected = stable_id(
            "uresol",
            {
                "catalog_revision": self.catalog_revision,
                "usage_revision": self.usage_revision,
                "policy_id": self.policy_id,
                "candidates": [item.to_dict() for item in candidates],
                "rejected": [item.to_dict() for item in rejected],
                "selected_binding_id": self.selected_binding_id,
                "next_eligible_at": self.next_eligible_at,
                "reason_codes": list(self.reason_codes),
            },
        )
        if self.resolution_id is not None and self.resolution_id != expected:
            _fail("resolution_id does not match canonical identity fields")
        object.__setattr__(self, "resolution_id", expected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "resolution_id": self.resolution_id,
            "catalog_revision": self.catalog_revision,
            "usage_revision": self.usage_revision,
            "policy_id": self.policy_id,
            "candidates": [item.to_dict() for item in self.candidates],
            "rejected": [item.to_dict() for item in self.rejected],
            "selected_binding_id": self.selected_binding_id,
            "next_eligible_at": self.next_eligible_at,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageAwareResolution":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(
            data, fields, ("catalog_revision", "usage_revision"), "UsageAwareResolution"
        )
        return cls(**{key: values.get(key) for key in fields})


@dataclass(frozen=True)
class UsageRoutingReceipt:
    """Bounded route/settlement receipt with digests and IDs only."""

    catalog_revision: str
    usage_revision: str
    schema_version: str = SCHEMA_VERSION
    receipt_id: Optional[str] = None
    request_id: Optional[str] = None
    attempt_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    caller_id: Optional[str] = None
    operation: Optional[str] = None
    policy_id: Optional[str] = None
    resolution_id: Optional[str] = None
    selected_binding_id: Optional[str] = None
    scope_id: Optional[str] = None
    reservation_id: Optional[str] = None
    estimate_id: Optional[str] = None
    observation_id: Optional[str] = None
    estimated: UsageVector = field(default_factory=UsageVector)
    settled: UsageVector = field(default_factory=UsageVector)
    fallback_class: FallbackClass = FallbackClass.NONE
    final_status: str = "unknown"
    next_eligible_at: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        object.__setattr__(
            self,
            "catalog_revision",
            _text(self.catalog_revision, "catalog_revision", 128),
        )
        object.__setattr__(
            self, "usage_revision", _text(self.usage_revision, "usage_revision", 128)
        )
        for name in (
            "request_id",
            "attempt_id",
            "idempotency_key",
            "caller_id",
            "policy_id",
            "resolution_id",
            "selected_binding_id",
            "scope_id",
            "reservation_id",
            "estimate_id",
            "observation_id",
        ):
            value = getattr(self, name)
            if value is not None:
                value = _text(value, name, 128)
            object.__setattr__(self, name, value)
        operation = self.operation
        if operation is not None:
            operation = _text(operation, "operation", 64).casefold()
            if not _OPERATION.fullmatch(operation):
                _fail("operation is not canonical")
        object.__setattr__(self, "operation", operation)
        estimated = (
            self.estimated
            if isinstance(self.estimated, UsageVector)
            else UsageVector.from_dict(self.estimated)
        )
        settled = (
            self.settled
            if isinstance(self.settled, UsageVector)
            else UsageVector.from_dict(self.settled)
        )
        object.__setattr__(self, "estimated", estimated)
        object.__setattr__(self, "settled", settled)
        object.__setattr__(
            self,
            "fallback_class",
            _enum(self.fallback_class, FallbackClass, "fallback_class"),
        )
        object.__setattr__(
            self, "final_status", _name(self.final_status, "final_status")
        )
        object.__setattr__(
            self,
            "next_eligible_at",
            _timestamp(self.next_eligible_at, "next_eligible_at"),
        )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(
            self, "created_at", _timestamp(self.created_at, "created_at")
        )
        payload = {
            "catalog_revision": self.catalog_revision,
            "usage_revision": self.usage_revision,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
            "idempotency_key": self.idempotency_key,
            "caller_id": self.caller_id,
            "operation": self.operation,
            "policy_id": self.policy_id,
            "resolution_id": self.resolution_id,
            "selected_binding_id": self.selected_binding_id,
            "scope_id": self.scope_id,
            "reservation_id": self.reservation_id,
            "estimate_id": self.estimate_id,
            "observation_id": self.observation_id,
            "estimated": estimated.to_dict(),
            "settled": settled.to_dict(),
            "fallback_class": self.fallback_class.value,
            "final_status": self.final_status,
            "next_eligible_at": self.next_eligible_at,
            "reason_codes": list(self.reason_codes),
            "created_at": self.created_at,
        }
        expected = receipt_identity(payload)
        if self.receipt_id is not None and self.receipt_id != expected:
            _fail("receipt_id does not match canonical identity fields")
        object.__setattr__(self, "receipt_id", expected)
        assert_no_prompt_media_or_output(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "catalog_revision": self.catalog_revision,
            "usage_revision": self.usage_revision,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
            "idempotency_key": self.idempotency_key,
            "caller_id": self.caller_id,
            "operation": self.operation,
            "policy_id": self.policy_id,
            "resolution_id": self.resolution_id,
            "selected_binding_id": self.selected_binding_id,
            "scope_id": self.scope_id,
            "reservation_id": self.reservation_id,
            "estimate_id": self.estimate_id,
            "observation_id": self.observation_id,
            "estimated": self.estimated.to_dict(),
            "settled": self.settled.to_dict(),
            "fallback_class": self.fallback_class.value,
            "final_status": self.final_status,
            "next_eligible_at": self.next_eligible_at,
            "reason_codes": list(self.reason_codes),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageRoutingReceipt":
        fields = tuple(cls.__dataclass_fields__)
        values = _strict_mapping(
            data,
            fields,
            ("catalog_revision", "usage_revision"),
            "UsageRoutingReceipt",
        )
        return cls(**{key: values.get(key) for key in fields})


def validate_canonical_record(record: Any) -> str:
    """Serialize *record* to canonical JSON after forbidding unsafe payloads."""

    if hasattr(record, "to_dict"):
        payload = record.to_dict()
    elif isinstance(record, Mapping):
        payload = dict(record)
    else:
        _fail("record must provide to_dict() or be a mapping")
    try:
        assert_no_prompt_media_or_output(payload)
        return canonical_json(payload)
    except (CanonicalizationError, UsageIdentityError) as exc:
        _fail(str(exc))
    raise AssertionError("unreachable")


__all__ = [
    "ENDPOINT_USAGE_CONTRACT_REQUIREMENT_ID",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "AvailabilityState",
    "ConfidenceLevel",
    "DimensionHeadroom",
    "EndpointUsageScope",
    "EstimateMethod",
    "FallbackClass",
    "LimitEnforcement",
    "LimitSource",
    "LimitWindow",
    "ProtocolKind",
    "ProviderUsageObservation",
    "Provenance",
    "Quantity",
    "QuantityKind",
    "ReservationState",
    "ResolutionCandidate",
    "RoutingMode",
    "RoutingPolicy",
    "SchemaValidationError",
    "UsageAwareResolution",
    "UsageDimension",
    "UsageErrorCode",
    "UsageEstimate",
    "UsageEvent",
    "UsageEventKind",
    "UsageLimit",
    "UsageReservation",
    "UsageRoutingReceipt",
    "UsageSnapshot",
    "UsageVector",
    "UsageVectorEntry",
    "WindowKind",
    "validate_canonical_record",
]
