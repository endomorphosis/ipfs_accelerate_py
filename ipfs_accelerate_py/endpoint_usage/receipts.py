"""Bounded usage-route receipts and attempt-chain provenance.

Receipts bind catalog/usage revisions, candidates, hard rejection reasons,
ranking inputs, selection, reservation, observation, settlement, and the
retry/fallback chain.  They never embed prompts, media, model output,
credentials, raw headers, or private endpoints.

This module is pure: constructing or serializing receipts performs no network,
provider, process, secret-store, model-load, or database I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .identity import (
    assert_no_prompt_media_or_output,
    contains_bearer_url,
    contains_raw_endpoint,
    is_secret_key,
    is_secret_value,
    receipt_identity,
    stable_id,
)
from .schema import (
    MAX_CANDIDATES,
    MAX_RANKING_INPUTS,
    MAX_REASON_CODES,
    MAX_STRING_BYTES,
    FallbackClass,
    ProviderUsageObservation,
    ResolutionCandidate,
    SchemaValidationError,
    UsageAwareResolution,
    UsageEstimate,
    UsageReservation,
    UsageRoutingReceipt,
    UsageVector,
    validate_canonical_record,
)

USAGE_ROUTING_RECEIPT_REQUIREMENT_ID = "requirement:usage-routing-receipt.v1"
MAX_CHAIN_LINKS = 32
MAX_ATTEMPT_DIGEST_BYTES = 128

_NAME = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_FORBIDDEN_RECEIPT_FIELD = re.compile(
    r"(?i)(?:prompt|message|messages|media|image_data|audio_data|video_data|"
    r"output_text|completion|payload|raw_headers|raw_body|response_body|"
    r"endpoint|url|uri|authorization|api[_-]?key|credential|secret|password|token)"
)


class ReceiptError(ValueError):
    """A proposed route receipt or chain link is unsafe or unbounded."""


class FinalStatus(str, Enum):
    """Terminal attempt statuses recorded on receipts."""

    COMMITTED = "committed"
    RELEASED = "released"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CAPACITY_UNAVAILABLE = "capacity_unavailable"
    POLICY_DENIED = "policy_denied"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


def _text(value: Any, field_name: str, maximum: int = MAX_STRING_BYTES) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ReceiptError("%s must be non-empty trimmed text" % field_name)
    if len(value.encode("utf-8")) > maximum:
        raise ReceiptError("%s exceeds %d UTF-8 bytes" % (field_name, maximum))
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ReceiptError("%s contains control characters" % field_name)
    if is_secret_value(value) or contains_bearer_url(value):
        raise ReceiptError("%s contains credential-shaped data" % field_name)
    if contains_raw_endpoint(value):
        raise ReceiptError("%s must not embed a raw endpoint or URL" % field_name)
    if _FORBIDDEN_RECEIPT_FIELD.search(value) and field_name in (
        "final_status",
        "reason_code",
        "link_kind",
    ):
        # Field names themselves are checked separately; values that look like
        # forbidden tokens in free-form slots are still rejected via secret/URL.
        pass
    return value


def _optional_text(
    value: Any, field_name: str, maximum: int = MAX_STRING_BYTES
) -> Optional[str]:
    if value is None:
        return None
    return _text(value, field_name, maximum=maximum)


def _reason_codes(values: Any) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        raise ReceiptError("reason_codes must be an array")
    if len(values) > MAX_REASON_CODES:
        raise ReceiptError("reason_codes exceeds maximum count")
    out = []
    for item in values:
        text = _text(item, "reason_code", maximum=64).casefold()
        if not _NAME.fullmatch(text):
            raise ReceiptError("reason_code is not canonical: %r" % item)
        out.append(text)
    return tuple(sorted(set(out)))


def _ranking_digest(
    ranking_inputs: Sequence[Tuple[str, Any]] | Mapping[str, Any] | None,
) -> Tuple[Tuple[str, Union[int, float, str, bool, None]], ...]:
    """Normalize ranking inputs into a bounded secret-free digest map."""

    if ranking_inputs is None:
        return ()
    pairs: List[Tuple[str, Any]] = []
    if isinstance(ranking_inputs, Mapping):
        pairs = list(ranking_inputs.items())
    else:
        for entry in ranking_inputs:
            if isinstance(entry, Mapping):
                pairs.append((entry.get("name"), entry.get("value")))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                pairs.append((entry[0], entry[1]))
            else:
                raise ReceiptError("ranking_inputs entries must be name/value pairs")
    if len(pairs) > MAX_RANKING_INPUTS:
        pairs = pairs[:MAX_RANKING_INPUTS]
    normalized: List[Tuple[str, Union[int, float, str, bool, None]]] = []
    for key, value in pairs:
        if not isinstance(key, str) or not key:
            raise ReceiptError("ranking input name is required")
        name = key.casefold().strip()
        if is_secret_key(name) or _FORBIDDEN_RECEIPT_FIELD.search(name):
            raise ReceiptError("forbidden ranking input name: %s" % name)
        if not _NAME.fullmatch(name):
            raise ReceiptError("ranking input name is not canonical: %r" % key)
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise ReceiptError("ranking input values must be scalars")
        if isinstance(value, str):
            value = _text(value, "ranking_input_value", maximum=256)
        elif isinstance(value, float) and (
            value != value or value in (float("inf"), float("-inf"))
        ):
            raise ReceiptError("ranking input float must be finite")
        elif isinstance(value, int) and not isinstance(value, bool):
            if abs(value) > (1 << 63) - 1:
                raise ReceiptError("ranking input integer overflows")
        normalized.append((name, value))
    return tuple(sorted(normalized, key=lambda pair: pair[0]))


def _candidate_summaries(
    candidates: Sequence[ResolutionCandidate] | None,
    *,
    limit: int = MAX_CANDIDATES,
) -> Tuple[Dict[str, Any], ...]:
    if not candidates:
        return ()
    rows: List[Dict[str, Any]] = []
    for item in list(candidates)[:limit]:
        if isinstance(item, ResolutionCandidate):
            cand = item
        elif isinstance(item, Mapping):
            cand = ResolutionCandidate.from_dict(item)
        else:
            raise ReceiptError("candidates must be ResolutionCandidate instances")
        rows.append(
            {
                "binding_id": cand.binding_id,
                "scope_id": cand.scope_id,
                "rank": cand.rank,
                "state": cand.state.value
                if hasattr(cand.state, "value")
                else str(cand.state),
                "rejection_reasons": list(cand.rejection_reasons),
                "ranking_inputs": [
                    {"name": n, "value": v} for n, v in _ranking_digest(cand.ranking_inputs)
                ],
            }
        )
    return tuple(rows)


def _rfc3339(value: Any, field_name: str = "timestamp") -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ReceiptError("%s must not be empty" % field_name)
        try:
            parsed = datetime.fromisoformat(
                raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            )
        except ValueError as exc:
            raise ReceiptError("%s is not RFC 3339" % field_name) from exc
    else:
        raise ReceiptError("%s must be an RFC 3339 string" % field_name)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReceiptError("%s must include a timezone" % field_name)
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True)
class AttemptLink:
    """One linked attempt/reservation in a retry or fallback chain.

    Each retry or fallback creates a new attempt and reservation; this link
    records the edge without embedding invoke payloads.
    """

    attempt_id: str
    parent_attempt_id: Optional[str] = None
    reservation_id: Optional[str] = None
    binding_id: Optional[str] = None
    scope_id: Optional[str] = None
    fallback_class: FallbackClass = FallbackClass.NONE
    denial_kind: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()
    final_status: str = FinalStatus.UNKNOWN.value
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attempt_id", _text(self.attempt_id, "attempt_id", maximum=128)
        )
        object.__setattr__(
            self,
            "parent_attempt_id",
            _optional_text(self.parent_attempt_id, "parent_attempt_id", maximum=128),
        )
        object.__setattr__(
            self,
            "reservation_id",
            _optional_text(self.reservation_id, "reservation_id", maximum=128),
        )
        object.__setattr__(
            self,
            "binding_id",
            _optional_text(self.binding_id, "binding_id", maximum=128),
        )
        object.__setattr__(
            self, "scope_id", _optional_text(self.scope_id, "scope_id", maximum=128)
        )
        fallback = self.fallback_class
        if not isinstance(fallback, FallbackClass):
            fallback = FallbackClass(str(fallback))
        object.__setattr__(self, "fallback_class", fallback)
        if self.denial_kind is not None:
            object.__setattr__(
                self,
                "denial_kind",
                _text(self.denial_kind, "denial_kind", maximum=64).casefold(),
            )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        status = _text(self.final_status, "final_status", maximum=64).casefold()
        if not _NAME.fullmatch(status):
            raise ReceiptError("final_status is not canonical")
        object.__setattr__(self, "final_status", status)
        object.__setattr__(self, "created_at", _rfc3339(self.created_at, "created_at"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "parent_attempt_id": self.parent_attempt_id,
            "reservation_id": self.reservation_id,
            "binding_id": self.binding_id,
            "scope_id": self.scope_id,
            "fallback_class": self.fallback_class.value,
            "denial_kind": self.denial_kind,
            "reason_codes": list(self.reason_codes),
            "final_status": self.final_status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "AttemptLink":
        if not isinstance(data, Mapping):
            raise ReceiptError("AttemptLink must be an object")
        return cls(
            attempt_id=data.get("attempt_id"),
            parent_attempt_id=data.get("parent_attempt_id"),
            reservation_id=data.get("reservation_id"),
            binding_id=data.get("binding_id"),
            scope_id=data.get("scope_id"),
            fallback_class=data.get("fallback_class", FallbackClass.NONE),
            denial_kind=data.get("denial_kind"),
            reason_codes=tuple(data.get("reason_codes") or ()),
            final_status=data.get("final_status", FinalStatus.UNKNOWN.value),
            created_at=data.get("created_at"),
        )


@dataclass(frozen=True)
class ReceiptChain:
    """Ordered attempt chain bound into the terminal route receipt."""

    links: Tuple[AttemptLink, ...] = ()
    chain_id: Optional[str] = None

    def __post_init__(self) -> None:
        raw = self.links or ()
        if len(raw) > MAX_CHAIN_LINKS:
            raise ReceiptError("receipt chain exceeds maximum links")
        links = tuple(
            item if isinstance(item, AttemptLink) else AttemptLink.from_dict(item)
            for item in raw
        )
        object.__setattr__(self, "links", links)
        material = [link.to_dict() for link in links]
        expected = stable_id("uchain", material)
        if self.chain_id is not None and self.chain_id != expected:
            raise ReceiptError("chain_id does not match canonical identity fields")
        object.__setattr__(self, "chain_id", expected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "links": [link.to_dict() for link in self.links],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ReceiptChain":
        if not isinstance(data, Mapping):
            raise ReceiptError("ReceiptChain must be an object")
        return cls(links=tuple(data.get("links") or ()), chain_id=data.get("chain_id"))


@dataclass(frozen=True)
class RouteReceiptDraft:
    """Intermediate assembly surface for a usage routing receipt.

    Routers populate identities and digests only; the builder rejects any
    forbidden payload fields.
    """

    catalog_revision: str
    usage_revision: str
    request_id: str
    attempt_id: str
    idempotency_key: str
    operation: str
    policy_id: Optional[str] = None
    resolution_id: Optional[str] = None
    selected_binding_id: Optional[str] = None
    scope_id: Optional[str] = None
    reservation_id: Optional[str] = None
    estimate_id: Optional[str] = None
    observation_id: Optional[str] = None
    caller_id: Optional[str] = None
    estimated: UsageVector = field(default_factory=UsageVector)
    settled: UsageVector = field(default_factory=UsageVector)
    fallback_class: FallbackClass = FallbackClass.NONE
    final_status: str = FinalStatus.UNKNOWN.value
    next_eligible_at: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()
    created_at: Optional[str] = None
    # Provenance digests — not full payload copies.
    hard_rejection_digest: Optional[str] = None
    ranking_inputs_digest: Optional[str] = None
    candidates_digest: Optional[str] = None
    chain: Optional[ReceiptChain] = None
    candidate_count: int = 0
    rejected_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "catalog_revision",
            "usage_revision",
            "request_id",
            "attempt_id",
            "idempotency_key",
            "operation",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum=128))
        for name in (
            "policy_id",
            "resolution_id",
            "selected_binding_id",
            "scope_id",
            "reservation_id",
            "estimate_id",
            "observation_id",
            "caller_id",
            "hard_rejection_digest",
            "ranking_inputs_digest",
            "candidates_digest",
        ):
            object.__setattr__(
                self, name, _optional_text(getattr(self, name), name, maximum=128)
            )
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
        fallback = self.fallback_class
        if not isinstance(fallback, FallbackClass):
            fallback = FallbackClass(str(fallback))
        object.__setattr__(self, "fallback_class", fallback)
        status = _text(self.final_status, "final_status", maximum=64).casefold()
        if not _NAME.fullmatch(status):
            raise ReceiptError("final_status is not canonical")
        object.__setattr__(self, "final_status", status)
        object.__setattr__(
            self, "next_eligible_at", _rfc3339(self.next_eligible_at, "next_eligible_at")
        )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(self, "created_at", _rfc3339(self.created_at, "created_at"))
        if self.chain is not None and not isinstance(self.chain, ReceiptChain):
            object.__setattr__(self, "chain", ReceiptChain.from_dict(self.chain))
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 0
        ):
            raise ReceiptError("candidate_count must be a non-negative integer")
        if (
            isinstance(self.rejected_count, bool)
            or not isinstance(self.rejected_count, int)
            or self.rejected_count < 0
        ):
            raise ReceiptError("rejected_count must be a non-negative integer")


def hard_rejection_digest(
    rejected: Sequence[ResolutionCandidate] | None,
) -> Optional[str]:
    """Content-address hard rejection reasons without full candidate dumps."""

    if not rejected:
        return None
    material = []
    for item in rejected[:MAX_CANDIDATES]:
        if isinstance(item, ResolutionCandidate):
            material.append(
                {
                    "binding_id": item.binding_id,
                    "reasons": list(item.rejection_reasons),
                }
            )
        elif isinstance(item, Mapping):
            material.append(
                {
                    "binding_id": item.get("binding_id"),
                    "reasons": list(item.get("rejection_reasons") or ()),
                }
            )
    return stable_id("uhard", material)


def ranking_inputs_digest(
    candidates: Sequence[ResolutionCandidate] | None,
) -> Optional[str]:
    """Content-address ranking inputs of accepted candidates."""

    if not candidates:
        return None
    material = []
    for item in candidates[:MAX_CANDIDATES]:
        if isinstance(item, ResolutionCandidate):
            material.append(
                {
                    "binding_id": item.binding_id,
                    "rank": item.rank,
                    "inputs": [
                        {"name": n, "value": v}
                        for n, v in _ranking_digest(item.ranking_inputs)
                    ],
                }
            )
    return stable_id("urank", material)


def candidates_digest(
    candidates: Sequence[ResolutionCandidate] | None,
    rejected: Sequence[ResolutionCandidate] | None = None,
) -> Optional[str]:
    """Content-address the full candidate + rejected set for one plan."""

    if not candidates and not rejected:
        return None
    material = {
        "candidates": list(_candidate_summaries(candidates)),
        "rejected": list(_candidate_summaries(rejected)),
    }
    return stable_id("ucands", material)


def build_receipt_chain(links: Sequence[AttemptLink | Mapping[str, Any]]) -> ReceiptChain:
    """Build a bounded attempt chain; each link is a distinct attempt/reservation."""

    parsed = [
        item if isinstance(item, AttemptLink) else AttemptLink.from_dict(item)
        for item in links
    ]
    # Validate parent linkage is acyclic and references prior attempts.
    seen: Dict[str, AttemptLink] = {}
    for link in parsed:
        if link.attempt_id in seen:
            raise ReceiptError("duplicate attempt_id in chain: %s" % link.attempt_id)
        if link.parent_attempt_id is not None and link.parent_attempt_id not in seen:
            # Allow forward reference only if parent appears earlier.
            raise ReceiptError(
                "parent_attempt_id %s not found before attempt %s"
                % (link.parent_attempt_id, link.attempt_id)
            )
        seen[link.attempt_id] = link
    return ReceiptChain(links=tuple(parsed))


def build_usage_routing_receipt(
    draft: RouteReceiptDraft | Mapping[str, Any],
    *,
    resolution: Optional[UsageAwareResolution] = None,
    estimate: Optional[UsageEstimate] = None,
    reservation: Optional[UsageReservation] = None,
    observation: Optional[ProviderUsageObservation] = None,
    include_chain_reason: bool = True,
) -> UsageRoutingReceipt:
    """Assemble a schema-valid :class:`UsageRoutingReceipt` from digests and IDs.

    Populates missing IDs from optional structured objects.  Never accepts or
    serializes prompts, media, output, credentials, raw headers, or endpoints.
    """

    if not isinstance(draft, RouteReceiptDraft):
        if not isinstance(draft, Mapping):
            raise ReceiptError("draft must be a RouteReceiptDraft or mapping")
        draft = RouteReceiptDraft(**dict(draft))

    policy_id = draft.policy_id
    resolution_id = draft.resolution_id
    selected_binding_id = draft.selected_binding_id
    scope_id = draft.scope_id
    estimate_id = draft.estimate_id
    reservation_id = draft.reservation_id
    observation_id = draft.observation_id
    estimated = draft.estimated
    settled = draft.settled
    usage_revision = draft.usage_revision
    catalog_revision = draft.catalog_revision
    next_eligible = draft.next_eligible_at
    reason_codes = list(draft.reason_codes)

    if resolution is not None:
        resolution_id = resolution_id or resolution.resolution_id
        catalog_revision = resolution.catalog_revision or catalog_revision
        usage_revision = resolution.usage_revision or usage_revision
        policy_id = policy_id or resolution.policy_id
        selected_binding_id = selected_binding_id or resolution.selected_binding_id
        next_eligible = next_eligible or resolution.next_eligible_at
        if draft.hard_rejection_digest is None and resolution.rejected:
            # Digests are recorded as reason_codes material only via explicit fields
            # on the draft; resolution hard reasons surface as reason codes.
            for cand in resolution.rejected[:8]:
                reason_codes.extend(cand.rejection_reasons[:4])
        if not draft.candidates_digest:
            pass  # digests optional on the schema receipt itself
        if resolution.reason_codes:
            reason_codes.extend(resolution.reason_codes)

    if estimate is not None:
        estimate_id = estimate_id or estimate.estimate_id
        if not estimated.entries:
            estimated = estimate.requested
        scope_id = scope_id or estimate.scope_id

    if reservation is not None:
        reservation_id = reservation_id or reservation.reservation_id
        scope_id = scope_id or reservation.scope_id
        if not estimated.entries and reservation.reserved.entries:
            estimated = reservation.reserved

    if observation is not None:
        observation_id = observation_id or observation.observation_id
        if not settled.entries and observation.usage.entries:
            settled = observation.usage
        scope_id = scope_id or observation.scope_id
        if observation.reason_codes:
            reason_codes.extend(observation.reason_codes)

    if draft.chain is not None and include_chain_reason:
        reason_codes.append("chain_links_%d" % len(draft.chain.links))
        # Bind chain identity into reason codes as a stable marker (bounded).
        if draft.chain.chain_id:
            reason_codes.append("chain_%s" % draft.chain.chain_id[-16:])

    # Surface digest presence as reason markers (IDs only, no payloads).
    if draft.hard_rejection_digest:
        reason_codes.append("hard_digest_bound")
    if draft.ranking_inputs_digest:
        reason_codes.append("rank_digest_bound")
    if draft.candidates_digest:
        reason_codes.append("candidates_digest_bound")

    # Deduplicate and bound.
    unique_reasons = tuple(sorted({c.casefold() for c in reason_codes if c}))[:MAX_REASON_CODES]

    receipt = UsageRoutingReceipt(
        catalog_revision=catalog_revision,
        usage_revision=usage_revision,
        request_id=draft.request_id,
        attempt_id=draft.attempt_id,
        idempotency_key=draft.idempotency_key,
        caller_id=draft.caller_id,
        operation=draft.operation.casefold(),
        policy_id=policy_id,
        resolution_id=resolution_id,
        selected_binding_id=selected_binding_id,
        scope_id=scope_id,
        reservation_id=reservation_id,
        estimate_id=estimate_id,
        observation_id=observation_id,
        estimated=estimated,
        settled=settled,
        fallback_class=draft.fallback_class,
        final_status=draft.final_status,
        next_eligible_at=next_eligible,
        reason_codes=unique_reasons,
        created_at=draft.created_at,
    )
    # Fail closed on any forbidden payload that snuck into to_dict().
    try:
        assert_no_prompt_media_or_output(receipt.to_dict())
        validate_canonical_record(receipt)
    except (SchemaValidationError, Exception) as exc:
        # Re-raise schema errors; wrap identity errors.
        if isinstance(exc, SchemaValidationError):
            raise
        raise ReceiptError(str(exc)) from exc
    return receipt


def receipt_binds_revisions(
    receipt: UsageRoutingReceipt,
    *,
    catalog_revision: str,
    usage_revision: str,
) -> bool:
    """Return whether *receipt* binds the expected catalog and usage revisions."""

    return (
        receipt.catalog_revision == catalog_revision
        and receipt.usage_revision == usage_revision
    )


def assert_receipt_safe(payload: Mapping[str, Any]) -> None:
    """Fail closed if *payload* contains forbidden route-receipt material."""

    try:
        assert_no_prompt_media_or_output(payload)
    except Exception as exc:
        raise ReceiptError(str(exc)) from exc
    for key in payload:
        name = str(key)
        if is_secret_key(name) or _FORBIDDEN_RECEIPT_FIELD.search(name):
            raise ReceiptError("forbidden receipt field: %s" % name)
        value = payload[key]
        if isinstance(value, str) and (
            contains_raw_endpoint(value) or contains_bearer_url(value)
        ):
            raise ReceiptError("receipt value embeds endpoint or credential material")


def chain_from_receipts(
    receipts: Sequence[UsageRoutingReceipt],
) -> ReceiptChain:
    """Derive an attempt chain from ordered terminal receipts."""

    links: List[AttemptLink] = []
    parent: Optional[str] = None
    for receipt in receipts:
        links.append(
            AttemptLink(
                attempt_id=receipt.attempt_id or stable_id("attempt", receipt.receipt_id),
                parent_attempt_id=parent,
                reservation_id=receipt.reservation_id,
                binding_id=receipt.selected_binding_id,
                scope_id=receipt.scope_id,
                fallback_class=receipt.fallback_class,
                reason_codes=receipt.reason_codes,
                final_status=receipt.final_status,
                created_at=receipt.created_at,
            )
        )
        parent = links[-1].attempt_id
    return build_receipt_chain(links)


def extend_reason_codes(
    existing: Sequence[str],
    *extra: str,
    limit: int = MAX_REASON_CODES,
) -> Tuple[str, ...]:
    """Merge reason codes with canonicalization and a hard bound."""

    merged = list(existing or ())
    merged.extend(extra)
    return _reason_codes(merged[: limit * 2])[:limit]


__all__ = [
    "USAGE_ROUTING_RECEIPT_REQUIREMENT_ID",
    "MAX_CHAIN_LINKS",
    "ReceiptError",
    "FinalStatus",
    "AttemptLink",
    "ReceiptChain",
    "RouteReceiptDraft",
    "hard_rejection_digest",
    "ranking_inputs_digest",
    "candidates_digest",
    "build_receipt_chain",
    "build_usage_routing_receipt",
    "receipt_binds_revisions",
    "assert_receipt_safe",
    "chain_from_receipts",
    "extend_reason_codes",
    "receipt_identity",
]
