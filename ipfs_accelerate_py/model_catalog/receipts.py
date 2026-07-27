"""Bounded, endpoint-free receipts for deterministic catalog selection.

Receipts explain *why a binding was selected* without becoming inference logs.
They contain stable catalog identities and ranking facts only: never prompts,
media, provider output, credentials, request headers, or raw endpoint URIs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .identity import content_cid, is_secret_key, is_secret_value
from .resolver import ResolutionCandidate, ResolutionResult
from .schema import OperationalState, Provenance, RouterBinding


MAX_RECEIPT_CANDIDATES = 128
MAX_RECEIPT_FILTERS = 128
MAX_RANKING_INPUTS = 64
MAX_FALLBACK_BOUNDARIES = 128
MAX_RECEIPT_PROVENANCE = 256
MAX_RECEIPT_STRING_BYTES = 512

_FIELD = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9._:-]{0,511}$")
_RAW_ENDPOINT = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://|(?:^|[.@/])(?:localhost|"
    r"(?:\d{1,3}\.){3}\d{1,3})(?::\d+)?(?:/|$))"
)
_FORBIDDEN_FIELD = re.compile(
    r"(?i)(?:prompt|message|media|image|audio|video|output|response|"
    r"endpoint|url|uri|header|body|payload|credential|secret|password|token|"
    r"authorization|api[_-]?key)"
)


class ReceiptValidationError(ValueError):
    """A proposed receipt contained unsafe or unbounded data."""


def _text(
    value: Any,
    field: str,
    *,
    maximum: int = MAX_RECEIPT_STRING_BYTES,
    pattern: Optional[re.Pattern] = None,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > maximum
        or is_secret_value(value)
        or _RAW_ENDPOINT.search(value)
    ):
        raise ReceiptValidationError("%s is invalid or unsafe" % field)
    if pattern is not None and not pattern.fullmatch(value):
        raise ReceiptValidationError("%s is not canonical" % field)
    return value


def _optional_text(
    value: Any,
    field: str,
    *,
    maximum: int = MAX_RECEIPT_STRING_BYTES,
    pattern: Optional[re.Pattern] = None,
) -> Optional[str]:
    return (
        None
        if value is None
        else _text(value, field, maximum=maximum, pattern=pattern)
    )


def _field(value: Any, field_name: str = "field") -> str:
    selected = _text(value, field_name, maximum=64, pattern=_FIELD).casefold()
    if is_secret_key(selected) or _FORBIDDEN_FIELD.search(selected):
        raise ReceiptValidationError("%s is forbidden in selection receipts" % field_name)
    return selected


def _identifier(value: Any, field: str) -> str:
    return _text(value, field, pattern=_IDENTIFIER)


def _scalar(value: Any, field: str) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > (1 << 63) - 1:
            raise ReceiptValidationError("%s exceeds the integer bound" % field)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReceiptValidationError("%s must be finite" % field)
        return value
    if isinstance(value, str):
        return _text(value, field, maximum=256)
    raise ReceiptValidationError("%s must be a bounded scalar" % field)


def _timestamp(value: Any, field: str) -> str:
    if isinstance(value, datetime):
        selected = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        selected = datetime.fromtimestamp(float(value), timezone.utc)
    elif isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            selected = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ReceiptValidationError("%s is not a timestamp" % field) from exc
    else:
        raise ReceiptValidationError("%s is not a timestamp" % field)
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise ReceiptValidationError("%s must be timezone-aware" % field)
    return (
        selected.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _now(clock: Optional[Callable[[], Any]]) -> str:
    return _timestamp(
        datetime.now(timezone.utc) if clock is None else clock(),
        "receipt clock",
    )


@dataclass(frozen=True)
class PolicyFilter:
    """One safe constraint and whether it admitted a candidate set."""

    name: str
    value: Any = None
    matched: bool = True
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _field(self.name, "filter name"))
        object.__setattr__(self, "value", _scalar(self.value, "filter value"))
        if not isinstance(self.matched, bool):
            raise ReceiptValidationError("filter matched must be boolean")
        object.__setattr__(
            self,
            "reason",
            _optional_text(self.reason, "filter reason", maximum=256),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "matched": self.matched,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyFilter":
        if not isinstance(data, Mapping) or set(data) != {
            "name",
            "value",
            "matched",
            "reason",
        }:
            raise ReceiptValidationError("PolicyFilter has missing or unknown fields")
        return cls(**dict(data))


FilterTrace = PolicyFilter


def _ranking(values: Any) -> Tuple[Tuple[str, Any], ...]:
    if values is None:
        return ()
    if isinstance(values, Mapping):
        values = tuple(values.items())
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        values = tuple(values)
    else:
        raise ReceiptValidationError("ranking_inputs must be key/value pairs")
    if len(values) > MAX_RANKING_INPUTS:
        raise ReceiptValidationError("ranking_inputs exceed the receipt bound")
    result = []
    for pair in values:
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(pair) != 2
        ):
            raise ReceiptValidationError("ranking input must be a key/value pair")
        key = _field(pair[0], "ranking input name")
        result.append((key, _scalar(pair[1], "ranking input value")))
    if len({item[0] for item in result}) != len(result):
        raise ReceiptValidationError("ranking_inputs contain duplicate names")
    return tuple(sorted(result))


@dataclass(frozen=True)
class CandidateTrace:
    """A ranked candidate represented only by stable catalog identities."""

    binding_id: str
    provider_id: str
    model_id: Optional[str]
    deployment_id: Optional[str]
    rank: int
    score: int
    ranking_inputs: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for field in ("binding_id", "provider_id"):
            object.__setattr__(self, field, _identifier(getattr(self, field), field))
        for field in ("model_id", "deployment_id"):
            object.__setattr__(
                self, field, _optional_text(getattr(self, field), field, pattern=_IDENTIFIER)
            )
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 0 <= self.rank < MAX_RECEIPT_CANDIDATES
        ):
            raise ReceiptValidationError("candidate rank is invalid")
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise ReceiptValidationError("candidate score is invalid")
        object.__setattr__(self, "ranking_inputs", _ranking(self.ranking_inputs))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
            "rank": self.rank,
            "score": self.score,
            "ranking_inputs": dict(self.ranking_inputs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateTrace":
        fields = {
            "binding_id",
            "provider_id",
            "model_id",
            "deployment_id",
            "rank",
            "score",
            "ranking_inputs",
        }
        if not isinstance(data, Mapping) or set(data) != fields:
            raise ReceiptValidationError("CandidateTrace has missing or unknown fields")
        return cls(**dict(data))


SelectionCandidate = CandidateTrace
CandidateReceipt = CandidateTrace


@dataclass(frozen=True)
class FallbackBoundary:
    """A router-safe boundary between successive ranked bindings."""

    position: int
    binding_id: str
    boundary: str
    allowed: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.position, bool)
            or not isinstance(self.position, int)
            or not 0 <= self.position < MAX_FALLBACK_BOUNDARIES
        ):
            raise ReceiptValidationError("fallback position is invalid")
        object.__setattr__(
            self, "binding_id", _identifier(self.binding_id, "fallback binding_id")
        )
        object.__setattr__(self, "boundary", _field(self.boundary, "fallback boundary"))
        if not isinstance(self.allowed, bool):
            raise ReceiptValidationError("fallback allowed must be boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "binding_id": self.binding_id,
            "boundary": self.boundary,
            "allowed": self.allowed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FallbackBoundary":
        if not isinstance(data, Mapping) or set(data) != {
            "position",
            "binding_id",
            "boundary",
            "allowed",
        }:
            raise ReceiptValidationError(
                "FallbackBoundary has missing or unknown fields"
            )
        return cls(**dict(data))


@dataclass(frozen=True)
class SourceProvenance:
    """Bounded provenance without source-controlled record bodies."""

    source: str
    observed_at: Optional[str] = None
    expires_at: Optional[str] = None
    issuer: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source", _text(self.source, "provenance source", maximum=128)
        )
        for field in ("observed_at", "expires_at"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _timestamp(value, field))
        if self.observed_at and self.expires_at and self.expires_at <= self.observed_at:
            raise ReceiptValidationError("provenance expiry must follow observation")
        object.__setattr__(
            self,
            "issuer",
            _optional_text(self.issuer, "provenance issuer", maximum=256),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "issuer": self.issuer,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceProvenance":
        if not isinstance(data, Mapping) or set(data) != {
            "source",
            "observed_at",
            "expires_at",
            "issuer",
        }:
            raise ReceiptValidationError(
                "SourceProvenance has missing or unknown fields"
            )
        return cls(**dict(data))


ProvenanceTrace = SourceProvenance


def _typed_tuple(
    values: Any, cls: Any, maximum: int, field: str
) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, Sequence
    ):
        raise ReceiptValidationError("%s must be a bounded array" % field)
    if len(values) > maximum:
        raise ReceiptValidationError("%s exceeds the receipt bound" % field)
    return tuple(
        item if isinstance(item, cls) else cls.from_dict(item) for item in values
    )


@dataclass(frozen=True)
class SelectionReceipt:
    """Complete, deterministic explanation of one catalog selection."""

    candidates: Tuple[CandidateTrace, ...]
    policy_filters: Tuple[PolicyFilter, ...]
    selected_binding: Optional[str]
    fallback_boundaries: Tuple[FallbackBoundary, ...]
    catalog_revision: str
    source_provenance: Tuple[SourceProvenance, ...]
    started_at: str
    decided_at: str
    total_candidates: int
    receipt_id: Optional[str] = None

    def __post_init__(self) -> None:
        candidates = _typed_tuple(
            self.candidates,
            CandidateTrace,
            MAX_RECEIPT_CANDIDATES,
            "candidates",
        )
        if tuple(item.rank for item in candidates) != tuple(range(len(candidates))):
            raise ReceiptValidationError("candidate ranks must be contiguous")
        if len({item.binding_id for item in candidates}) != len(candidates):
            raise ReceiptValidationError("candidate bindings must be unique")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "policy_filters",
            _typed_tuple(
                self.policy_filters,
                PolicyFilter,
                MAX_RECEIPT_FILTERS,
                "policy_filters",
            ),
        )
        selected = _optional_text(
            self.selected_binding, "selected_binding", pattern=_IDENTIFIER
        )
        if selected is not None and selected not in {
            item.binding_id for item in candidates
        }:
            raise ReceiptValidationError(
                "selected binding must be present in receipt candidates"
            )
        object.__setattr__(self, "selected_binding", selected)
        boundaries = _typed_tuple(
            self.fallback_boundaries,
            FallbackBoundary,
            MAX_FALLBACK_BOUNDARIES,
            "fallback_boundaries",
        )
        if tuple(item.position for item in boundaries) != tuple(
            range(len(boundaries))
        ):
            raise ReceiptValidationError(
                "fallback boundary positions must be contiguous"
            )
        if any(
            item.binding_id not in {candidate.binding_id for candidate in candidates}
            for item in boundaries
        ):
            raise ReceiptValidationError(
                "fallback boundaries must reference receipt candidates"
            )
        object.__setattr__(self, "fallback_boundaries", boundaries)
        object.__setattr__(
            self,
            "catalog_revision",
            _identifier(self.catalog_revision, "catalog_revision"),
        )
        provenance = _typed_tuple(
            self.source_provenance,
            SourceProvenance,
            MAX_RECEIPT_PROVENANCE,
            "source_provenance",
        )
        # Stable deduplication is important when the same source is asserted at
        # provider, model, deployment, and binding levels.
        unique = {
            (
                item.source,
                item.observed_at,
                item.expires_at,
                item.issuer,
            ): item
            for item in provenance
        }
        provenance = tuple(
            unique[key]
            for key in sorted(
                unique,
                key=lambda item: tuple("" if value is None else value for value in item),
            )
        )
        object.__setattr__(self, "source_provenance", provenance)
        started = _timestamp(self.started_at, "started_at")
        decided = _timestamp(self.decided_at, "decided_at")
        if decided < started:
            raise ReceiptValidationError("decided_at must not precede started_at")
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "decided_at", decided)
        if (
            isinstance(self.total_candidates, bool)
            or not isinstance(self.total_candidates, int)
            or self.total_candidates < len(candidates)
            or self.total_candidates > 1_000_000
        ):
            raise ReceiptValidationError("total_candidates is invalid")
        expected = content_cid(self.content_dict())
        if self.receipt_id is not None and self.receipt_id != expected:
            raise ReceiptValidationError("receipt_id does not match receipt content")
        object.__setattr__(self, "receipt_id", expected)

    @property
    def selected_binding_id(self) -> Optional[str]:
        return self.selected_binding

    @property
    def timestamp(self) -> str:
        return self.decided_at

    def content_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "policy_filters": [item.to_dict() for item in self.policy_filters],
            "selected_binding": self.selected_binding,
            "fallback_boundaries": [
                item.to_dict() for item in self.fallback_boundaries
            ],
            "catalog_revision": self.catalog_revision,
            "source_provenance": [
                item.to_dict() for item in self.source_provenance
            ],
            "started_at": self.started_at,
            "decided_at": self.decided_at,
            "total_candidates": self.total_candidates,
        }

    def to_dict(self) -> Dict[str, Any]:
        result = self.content_dict()
        result["receipt_id"] = self.receipt_id
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectionReceipt":
        fields = {
            "candidates",
            "policy_filters",
            "selected_binding",
            "fallback_boundaries",
            "catalog_revision",
            "source_provenance",
            "started_at",
            "decided_at",
            "total_candidates",
            "receipt_id",
        }
        if not isinstance(data, Mapping) or set(data) != fields:
            raise ReceiptValidationError(
                "SelectionReceipt has missing or unknown fields"
            )
        return cls(**dict(data))

    @classmethod
    def from_resolution(
        cls, result: ResolutionResult, **kwargs: Any
    ) -> "SelectionReceipt":
        return create_selection_receipt(result, **kwargs)


RoutingReceipt = SelectionReceipt


def _effective_state(candidate: ResolutionCandidate) -> OperationalState:
    levels = [candidate.binding.state]
    if candidate.deployment is not None:
        levels.append(candidate.deployment.state)
    if candidate.model is not None:
        levels.append(candidate.model.state)
    levels.append(candidate.provider.state)
    values = {}
    for name in OperationalState.__dataclass_fields__:  # type: ignore[attr-defined]
        values[name] = next(
            (
                getattr(state, name)
                for state in levels
                if getattr(state, name) is not None
            ),
            None,
        )
    return OperationalState(**values)


def _candidate_trace(candidate: ResolutionCandidate, rank: int) -> CandidateTrace:
    state = _effective_state(candidate)
    ranking = {
        "binding.priority": candidate.binding.priority,
        "resolution.score": candidate.score,
    }
    for name in OperationalState.__dataclass_fields__:  # type: ignore[attr-defined]
        ranking["state.%s" % name] = getattr(state, name)
    for prefix, record in (
        ("provider", candidate.provider),
        ("model", candidate.model),
        ("deployment", candidate.deployment),
    ):
        if record is not None:
            ranking["%s.lifecycle" % prefix] = record.lifecycle.value
    return CandidateTrace(
        binding_id=candidate.binding_id,
        provider_id=candidate.provider_id,
        model_id=candidate.model_id,
        deployment_id=candidate.deployment_id,
        rank=rank,
        score=candidate.score,
        ranking_inputs=tuple(ranking.items()),
    )


def _filters(result: ResolutionResult) -> Tuple[PolicyFilter, ...]:
    request = result.request
    values = {
        "operation": request.operation.value,
        "modality": None if request.modality is None else request.modality.value,
        "model": request.model,
        "provider": request.provider,
        "deployment": request.deployment,
        "device": request.device,
        "context": request.context,
        "health": request.health,
        "locality": request.locality,
        "configured": request.configured,
        "authorized": request.authorized,
        "reachable": request.reachable,
        "routable": request.routable,
    }
    filters = [
        PolicyFilter(name=name, value=value, matched=result.found)
        for name, value in values.items()
        if value is not None
    ]
    for name, value in request.policy:
        selected_name = "policy.%s" % name
        if (
            is_secret_key(name)
            or _FORBIDDEN_FIELD.search(name)
            or is_secret_value(value)
            or _RAW_ENDPOINT.search(value)
        ):
            # Preserve the existence and outcome of a policy boundary without
            # retaining its credential-bearing name or value.
            filters.append(
                PolicyFilter(
                    name="policy.redacted",
                    value=None,
                    matched=result.found,
                    reason="sensitive_filter_omitted",
                )
            )
        else:
            filters.append(
                PolicyFilter(
                    name=selected_name,
                    value=value,
                    matched=result.found,
                )
            )
    return tuple(filters[:MAX_RECEIPT_FILTERS])


def _provenance(
    candidates: Iterable[ResolutionCandidate],
) -> Tuple[SourceProvenance, ...]:
    result = []
    for candidate in candidates:
        records = (
            candidate.provider,
            candidate.model,
            candidate.deployment,
            candidate.binding,
        )
        for record in records:
            if record is None:
                continue
            for item in record.provenance:
                if not isinstance(item, Provenance):
                    continue
                try:
                    result.append(
                        SourceProvenance(
                            source=item.source,
                            observed_at=item.observed_at,
                            expires_at=item.expires_at,
                            issuer=item.issuer,
                        )
                    )
                except ReceiptValidationError:
                    # Untrusted source-controlled detail is omitted rather than
                    # copied or redacted into the debugging contract.
                    continue
                if len(result) >= MAX_RECEIPT_PROVENANCE:
                    return tuple(result)
    return tuple(result)


def _selected_id(
    selected: Any, candidates: Sequence[ResolutionCandidate]
) -> Optional[str]:
    if selected is None:
        return candidates[0].binding_id if candidates else None
    if isinstance(selected, ResolutionCandidate):
        return selected.binding_id
    if isinstance(selected, RouterBinding):
        return selected.binding_id
    if isinstance(selected, str):
        return selected
    raise ReceiptValidationError("selected binding is invalid")


def _fallbacks(
    candidates: Sequence[ResolutionCandidate],
) -> Tuple[FallbackBoundary, ...]:
    result = []
    previous = None
    for position, candidate in enumerate(candidates[:MAX_FALLBACK_BOUNDARIES]):
        if position == 0:
            boundary = "primary"
        elif previous is not None and candidate.provider_id != previous.provider_id:
            boundary = "provider"
        elif (
            previous is not None
            and candidate.binding.router != previous.binding.router
        ):
            boundary = "router"
        else:
            boundary = "binding"
        result.append(
            FallbackBoundary(
                position=position,
                binding_id=candidate.binding_id,
                boundary=boundary,
                allowed=True,
            )
        )
        previous = candidate
    return tuple(result)


def create_selection_receipt(
    result: ResolutionResult,
    *,
    selected_binding: Any = None,
    started_at: Optional[Any] = None,
    decided_at: Optional[Any] = None,
    clock: Optional[Callable[[], Any]] = None,
) -> SelectionReceipt:
    """Create a safe receipt from a deterministic resolver result."""

    if not isinstance(result, ResolutionResult):
        raise TypeError("result must be a ResolutionResult")
    selected = _selected_id(selected_binding, result.candidates)
    included = list(result.candidates[:MAX_RECEIPT_CANDIDATES])
    if selected is not None and selected not in {
        item.binding_id for item in included
    }:
        match = next(
            (item for item in result.candidates if item.binding_id == selected),
            None,
        )
        if match is None:
            raise ReceiptValidationError(
                "selected binding is not a resolution candidate"
            )
        if included:
            included[-1] = match
        else:
            included.append(match)
    begun = _now(clock) if started_at is None else _timestamp(started_at, "started_at")
    ended = _now(clock) if decided_at is None else _timestamp(decided_at, "decided_at")
    return SelectionReceipt(
        candidates=tuple(
            _candidate_trace(candidate, rank)
            for rank, candidate in enumerate(included)
        ),
        policy_filters=_filters(result),
        selected_binding=selected,
        fallback_boundaries=_fallbacks(included),
        catalog_revision=result.snapshot_revision,
        source_provenance=_provenance(included),
        started_at=begun,
        decided_at=ended,
        total_candidates=result.total_candidates,
    )


selection_receipt = create_selection_receipt
build_selection_receipt = create_selection_receipt
receipt_from_resolution = create_selection_receipt


class SelectionReceiptBuilder:
    """Small injectable-clock facade for routing integrations."""

    def __init__(self, clock: Optional[Callable[[], Any]] = None) -> None:
        if clock is not None and not callable(clock):
            raise TypeError("receipt clock must be callable")
        self.clock = clock

    def build(self, result: ResolutionResult, **kwargs: Any) -> SelectionReceipt:
        kwargs.setdefault("clock", self.clock)
        return create_selection_receipt(result, **kwargs)

    create = build


ReceiptBuilder = SelectionReceiptBuilder


__all__ = [
    "CandidateReceipt",
    "CandidateTrace",
    "FallbackBoundary",
    "FilterTrace",
    "MAX_FALLBACK_BOUNDARIES",
    "MAX_RANKING_INPUTS",
    "MAX_RECEIPT_CANDIDATES",
    "MAX_RECEIPT_FILTERS",
    "MAX_RECEIPT_PROVENANCE",
    "PolicyFilter",
    "ProvenanceTrace",
    "ReceiptBuilder",
    "ReceiptValidationError",
    "RoutingReceipt",
    "SelectionCandidate",
    "SelectionReceipt",
    "SelectionReceiptBuilder",
    "SourceProvenance",
    "build_selection_receipt",
    "create_selection_receipt",
    "receipt_from_resolution",
    "selection_receipt",
]
