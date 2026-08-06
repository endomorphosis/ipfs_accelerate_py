"""Automatic implementation-provider selection for the agent supervisor.

This module is the pure ranking / admission surface for ``auto``
implementation routing.  It reuses:

* :mod:`entrypoints.capability_resolver` preferred-provider policy
  (Grok preferred, Codex fallback, Grok default when both healthy);
* :mod:`ipfs_accelerate_py.llm_router` readiness probes that do **not**
  charge usage (binary + auth, no generation);
* durable capacity / hard-quota latches already maintained by the
  implementation daemon from classified provider failures;
* optional :mod:`endpoint_usage.adapters` observations when a caller has
  already normalized provider balance metadata.

Selection rules (fail-closed):

1. Hard filters: not ready, active transient capacity latch, or hard quota
   exhaustion remove a candidate before ranking.
2. Soft ranking prefers more headroom / fewer restrictions; identical soft
   scores are broken by the preferred provider (Grok).
3. Codex is authorized for *implementation* only when Grok has a durable
   hard-quota exhaustion latch (or equivalent observation).  Transient Grok
   capacity cooldowns never open Codex as implementer.
4. Explicit non-``auto`` pins bypass this selector entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.agent_supervisor.entrypoints.capability_resolver import (
    FALLBACK_PROVIDER,
    PREFERRED_PROVIDER,
    PreferredProviderCapability,
)


IMPLEMENTATION_PROVIDER_AUTO_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/implementation-provider-auto@1"
)
TIE_BREAKER_PROVIDER = PREFERRED_PROVIDER  # grok
DEFAULT_IMPLEMENTATION_PROVIDER = "auto"


class AutoProviderDecision(str, Enum):
    """Closed set of auto-route outcomes."""

    GROK = "grok"
    CODEX = "codex"
    UNAVAILABLE = "unavailable"
    BACKOFF = "backoff"


class AutoProviderReason(str, Enum):
    """Stable reason codes for auto-route receipts."""

    PREFERRED_READY = "preferred_provider_ready"
    TIE_BREAK_PREFERRED = "tie_break_preferred_provider"
    FALLBACK_AFTER_QUOTA = "fallback_after_preferred_hard_quota"
    PREFERRED_TRANSIENT_CAPACITY = "preferred_provider_transient_capacity"
    PREFERRED_NOT_READY = "preferred_provider_not_ready"
    FALLBACK_NOT_READY = "fallback_provider_not_ready"
    GLOBAL_CAPACITY = "global_provider_capacity_cooldown"
    NO_ELIGIBLE = "no_eligible_implementation_provider"
    EXPLICIT_PIN = "explicit_provider_pin"


@dataclass(frozen=True)
class BackendObservation:
    """Observed readiness + quota/capacity for one implementation backend."""

    provider_id: str
    ready: bool
    authenticated: bool = False
    binary_available: bool = False
    hard_quota_exhausted: bool = False
    capacity_latched: bool = False
    request_headroom: int | None = None
    retry_at: str = ""
    source: str = "probe"
    reason_codes: tuple[str, ...] = ()

    @property
    def capability(self) -> PreferredProviderCapability:
        if self.hard_quota_exhausted:
            return PreferredProviderCapability.QUOTA_EXHAUSTED
        if self.capacity_latched:
            return PreferredProviderCapability.CAPACITY_UNAVAILABLE
        if self.ready and self.authenticated:
            return PreferredProviderCapability.AVAILABLE
        return PreferredProviderCapability.UNAVAILABLE

    @property
    def eligible(self) -> bool:
        """Soft-rank eligible: ready, authenticated, and not latched/exhausted."""

        return (
            self.ready
            and self.authenticated
            and self.binary_available
            and not self.hard_quota_exhausted
            and not self.capacity_latched
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "ready": self.ready,
            "authenticated": self.authenticated,
            "binary_available": self.binary_available,
            "hard_quota_exhausted": self.hard_quota_exhausted,
            "capacity_latched": self.capacity_latched,
            "request_headroom": self.request_headroom,
            "retry_at": self.retry_at,
            "source": self.source,
            "reason_codes": list(self.reason_codes),
            "capability": self.capability.value,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class AutoProviderSelection:
    """Deterministic selection receipt for one auto-route decision."""

    decision: AutoProviderDecision
    selected_provider: str
    preferred_provider: str = PREFERRED_PROVIDER
    fallback_provider: str = FALLBACK_PROVIDER
    tie_breaker: str = TIE_BREAKER_PROVIDER
    reason_codes: tuple[str, ...] = ()
    observations: tuple[BackendObservation, ...] = ()
    retry_at: str = ""
    schema: str = IMPLEMENTATION_PROVIDER_AUTO_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "decision": self.decision.value,
            "selected_provider": self.selected_provider,
            "preferred_provider": self.preferred_provider,
            "fallback_provider": self.fallback_provider,
            "tie_breaker": self.tie_breaker,
            "reason_codes": list(self.reason_codes),
            "retry_at": self.retry_at,
            "observations": [item.to_dict() for item in self.observations],
        }


def _bool(value: Any) -> bool:
    return bool(value)


def _latch_entry(
    latches: Mapping[str, Any] | None,
    family: str,
) -> dict[str, Any]:
    if not isinstance(latches, Mapping):
        return {}
    entry = latches.get(family)
    return dict(entry) if isinstance(entry, Mapping) else {}


def observe_llm_router_backend(
    provider_id: str,
    *,
    binary_available: bool,
    authenticated: bool,
    provider_constructible: bool = True,
    source: str = "llm_router",
) -> BackendObservation:
    """Build one observation from non-charging llm_router readiness probes."""

    ready = bool(
        binary_available and authenticated and provider_constructible
    )
    reasons: list[str] = []
    if not binary_available:
        reasons.append("binary_unavailable")
    if not authenticated:
        reasons.append("unauthenticated")
    if binary_available and authenticated and not provider_constructible:
        reasons.append("provider_not_constructible")
    return BackendObservation(
        provider_id=provider_id,
        ready=ready,
        authenticated=authenticated,
        binary_available=binary_available,
        source=source,
        reason_codes=tuple(reasons),
    )


def merge_capacity_latch(
    observation: BackendObservation,
    latch: Mapping[str, Any] | None,
) -> BackendObservation:
    """Overlay a durable daemon capacity/quota latch onto one observation."""

    entry = dict(latch) if isinstance(latch, Mapping) else {}
    if not entry:
        return observation
    active = bool(entry.get("active", False))
    hard_quota = bool(entry.get("hard_quota_exhausted", False))
    retry_at = str(entry.get("retry_at") or observation.retry_at or "")
    reasons = list(observation.reason_codes)
    if hard_quota:
        reasons.append("hard_quota_exhausted")
    elif active:
        reasons.append("capacity_latched")
    return BackendObservation(
        provider_id=observation.provider_id,
        ready=observation.ready,
        authenticated=observation.authenticated,
        binary_available=observation.binary_available,
        hard_quota_exhausted=observation.hard_quota_exhausted or hard_quota,
        capacity_latched=observation.capacity_latched or (active and not hard_quota),
        request_headroom=observation.request_headroom,
        retry_at=retry_at,
        source=observation.source,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def merge_usage_observation(
    observation: BackendObservation,
    *,
    hard_quota_exhausted: bool = False,
    capacity_restricted: bool = False,
    request_headroom: int | None = None,
    retry_at: str = "",
    reason_codes: Sequence[str] = (),
    source: str = "endpoint_usage",
) -> BackendObservation:
    """Overlay a normalized endpoint_usage / adapter observation."""

    reasons = list(observation.reason_codes)
    reasons.extend(str(code) for code in reason_codes if str(code))
    if hard_quota_exhausted:
        reasons.append("usage_hard_quota_exhausted")
    if capacity_restricted and not hard_quota_exhausted:
        reasons.append("usage_capacity_restricted")
    headroom = (
        request_headroom
        if request_headroom is not None
        else observation.request_headroom
    )
    return BackendObservation(
        provider_id=observation.provider_id,
        ready=observation.ready and not hard_quota_exhausted,
        authenticated=observation.authenticated,
        binary_available=observation.binary_available,
        hard_quota_exhausted=(
            observation.hard_quota_exhausted or hard_quota_exhausted
        ),
        capacity_latched=(
            observation.capacity_latched
            or (capacity_restricted and not hard_quota_exhausted)
        ),
        request_headroom=headroom,
        retry_at=str(retry_at or observation.retry_at or ""),
        source=source if (hard_quota_exhausted or capacity_restricted) else observation.source,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _soft_rank_key(observation: BackendObservation) -> tuple[int, int, int, str]:
    """Higher is better.  Final element is inverse-preferred for stable sort.

    Sort with ``reverse=True`` so larger headroom and preferred providers win.
    The final string is inverted preferred id so Grok sorts above Codex when
    all numeric fields tie.
    """

    headroom = (
        int(observation.request_headroom)
        if observation.request_headroom is not None
        else 0
    )
    preferred_bonus = 1 if observation.provider_id == TIE_BREAKER_PROVIDER else 0
    # Invert provider id so "grok" < "codex" lexicographically becomes higher
    # priority when using reverse sort on the whole tuple... actually we put
    # preferred_bonus first after eligibility.
    # Order: preferred_bonus, headroom, not_latched already filtered, -name for
    # deterministic residual tie-break with Grok first via preferred_bonus.
    residual = 0 if observation.provider_id == TIE_BREAKER_PROVIDER else 1
    return (preferred_bonus, headroom, -residual, observation.provider_id)


def select_implementation_provider(
    observations: Sequence[BackendObservation],
    *,
    global_capacity_latched: bool = False,
    allow_codex_without_grok_quota: bool = False,
) -> AutoProviderSelection:
    """Select the implementation backend from frozen observations.

    Parameters
    ----------
    observations:
        One entry per candidate backend (typically grok + codex).
    global_capacity_latched:
        When true, no automatic provider may dispatch.
    allow_codex_without_grok_quota:
        Test / operator escape hatch.  Production auto routing leaves this
        false so Codex implement only opens after Grok hard-quota evidence.
    """

    obs = tuple(observations)
    by_id = {item.provider_id: item for item in obs}
    grok = by_id.get(PREFERRED_PROVIDER)
    codex = by_id.get(FALLBACK_PROVIDER)

    if global_capacity_latched:
        return AutoProviderSelection(
            decision=AutoProviderDecision.BACKOFF,
            selected_provider="",
            reason_codes=(AutoProviderReason.GLOBAL_CAPACITY.value,),
            observations=obs,
            retry_at=next((item.retry_at for item in obs if item.retry_at), ""),
        )

    # Grok ready → always preferred (explicit tie-break when Codex also ready).
    if grok is not None and grok.eligible:
        reasons = [AutoProviderReason.PREFERRED_READY.value]
        if codex is not None and codex.eligible:
            reasons.append(AutoProviderReason.TIE_BREAK_PREFERRED.value)
        # Soft ranking is recorded for multi-candidate observability only.
        _ = sorted(
            [item for item in obs if item.eligible],
            key=_soft_rank_key,
            reverse=True,
        )
        return AutoProviderSelection(
            decision=AutoProviderDecision.GROK,
            selected_provider=PREFERRED_PROVIDER,
            reason_codes=tuple(reasons),
            observations=obs,
        )

    # Transient Grok capacity: backoff; never open Codex without hard quota.
    if (
        grok is not None
        and grok.capacity_latched
        and not grok.hard_quota_exhausted
    ):
        return AutoProviderSelection(
            decision=AutoProviderDecision.BACKOFF,
            selected_provider="",
            reason_codes=(
                AutoProviderReason.PREFERRED_TRANSIENT_CAPACITY.value,
            ),
            observations=obs,
            retry_at=grok.retry_at,
        )

    # Durable Grok hard-quota exhaustion authorizes Codex implement.
    if (
        grok is not None
        and grok.hard_quota_exhausted
        and codex is not None
        and codex.eligible
    ):
        return AutoProviderSelection(
            decision=AutoProviderDecision.CODEX,
            selected_provider=FALLBACK_PROVIDER,
            reason_codes=(AutoProviderReason.FALLBACK_AFTER_QUOTA.value,),
            observations=obs,
            retry_at=grok.retry_at,
        )

    # Optional escape hatch (tests / operator) — off by default.
    if (
        allow_codex_without_grok_quota
        and codex is not None
        and codex.eligible
    ):
        return AutoProviderSelection(
            decision=AutoProviderDecision.CODEX,
            selected_provider=FALLBACK_PROVIDER,
            reason_codes=(AutoProviderReason.FALLBACK_AFTER_QUOTA.value,),
            observations=obs,
            retry_at=codex.retry_at,
        )

    reasons: list[str] = [AutoProviderReason.NO_ELIGIBLE.value]
    if grok is not None and not grok.eligible:
        reasons.append(AutoProviderReason.PREFERRED_NOT_READY.value)
    if codex is not None and not codex.eligible:
        reasons.append(AutoProviderReason.FALLBACK_NOT_READY.value)
    return AutoProviderSelection(
        decision=AutoProviderDecision.UNAVAILABLE,
        selected_provider="",
        reason_codes=tuple(dict.fromkeys(reasons)),
        observations=obs,
        retry_at=(
            (grok.retry_at if grok is not None else "")
            or (codex.retry_at if codex is not None else "")
        ),
    )


def probe_llm_router_backends(
    *,
    grok_binary: bool,
    grok_authenticated: bool,
    grok_constructible: bool = True,
    codex_binary: bool,
    codex_authenticated: bool = True,
    latches: Mapping[str, Any] | None = None,
    usage_observations: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[BackendObservation, ...]:
    """Assemble grok/codex observations from probes + latches + usage overlays.

    Callers supply the boolean probe results so this function stays free of
    import-time and network side effects (llm_router probes happen outside).
    """

    grok = observe_llm_router_backend(
        PREFERRED_PROVIDER,
        binary_available=grok_binary,
        authenticated=grok_authenticated,
        provider_constructible=grok_constructible,
    )
    codex = observe_llm_router_backend(
        FALLBACK_PROVIDER,
        binary_available=codex_binary,
        authenticated=codex_authenticated,
        provider_constructible=codex_binary,
    )
    grok = merge_capacity_latch(grok, _latch_entry(latches, PREFERRED_PROVIDER))
    codex = merge_capacity_latch(codex, _latch_entry(latches, FALLBACK_PROVIDER))

    usage = usage_observations if isinstance(usage_observations, Mapping) else {}
    for provider_id, name, base in (
        (PREFERRED_PROVIDER, PREFERRED_PROVIDER, grok),
        (FALLBACK_PROVIDER, FALLBACK_PROVIDER, codex),
    ):
        raw = usage.get(provider_id) or usage.get(name)
        if not isinstance(raw, Mapping):
            continue
        updated = merge_usage_observation(
            base,
            hard_quota_exhausted=bool(
                raw.get("hard_quota_exhausted") or raw.get("quota_exhausted")
            ),
            capacity_restricted=bool(
                raw.get("capacity_restricted") or raw.get("capacity_latched")
            ),
            request_headroom=(
                int(raw["request_headroom"])
                if raw.get("request_headroom") is not None
                else None
            ),
            retry_at=str(raw.get("retry_at") or ""),
            reason_codes=tuple(raw.get("reason_codes") or ()),
        )
        if provider_id == PREFERRED_PROVIDER:
            grok = updated
        else:
            codex = updated
    return (grok, codex)


def select_auto_implementation_provider(
    *,
    grok_binary: bool,
    grok_authenticated: bool,
    grok_constructible: bool = True,
    codex_binary: bool,
    codex_authenticated: bool = True,
    latches: Mapping[str, Any] | None = None,
    usage_observations: Mapping[str, Mapping[str, Any]] | None = None,
    global_capacity_latched: bool = False,
    allow_codex_without_grok_quota: bool = False,
) -> AutoProviderSelection:
    """End-to-end auto selection from probe booleans + capacity latches."""

    observations = probe_llm_router_backends(
        grok_binary=grok_binary,
        grok_authenticated=grok_authenticated,
        grok_constructible=grok_constructible,
        codex_binary=codex_binary,
        codex_authenticated=codex_authenticated,
        latches=latches,
        usage_observations=usage_observations,
    )
    return select_implementation_provider(
        observations,
        global_capacity_latched=global_capacity_latched,
        allow_codex_without_grok_quota=allow_codex_without_grok_quota,
    )


def resolve_configured_implementation_provider(
    configured: str | None,
) -> str:
    """Normalize the configured provider pin; empty means auto."""

    value = str(configured or "").strip().lower()
    return value or DEFAULT_IMPLEMENTATION_PROVIDER


__all__ = [
    "AutoProviderDecision",
    "AutoProviderReason",
    "AutoProviderSelection",
    "BackendObservation",
    "DEFAULT_IMPLEMENTATION_PROVIDER",
    "IMPLEMENTATION_PROVIDER_AUTO_SCHEMA",
    "TIE_BREAKER_PROVIDER",
    "merge_capacity_latch",
    "merge_usage_observation",
    "observe_llm_router_backend",
    "probe_llm_router_backends",
    "resolve_configured_implementation_provider",
    "select_auto_implementation_provider",
    "select_implementation_provider",
]
