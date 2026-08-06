"""Automatic implementation-provider selection for the agent supervisor.

This module is the pure ranking / admission surface for ``auto``
implementation routing.  It reuses:

* :mod:`entrypoints.capability_resolver` preferred-provider policy
  (Grok preferred; Grok is the default tie-breaker when multiple backends
  are healthy);
* :mod:`ipfs_accelerate_py.llm_router` readiness probes that do **not**
  charge usage (binary + auth, no generation);
* :mod:`cli_provider_balance` for Anthropic Claude CLI and Google Gemini CLI
  readiness plus quota/balance classification;
* durable capacity / hard-quota latches already maintained by the
  implementation daemon from classified provider failures;
* optional :mod:`endpoint_usage.adapters` observations when a caller has
  already normalized provider balance metadata.

Selection rules (fail-closed):

1. Hard filters: not ready, active transient capacity latch, or hard quota
   exhaustion remove a candidate before ranking.
2. Soft ranking prefers more headroom / higher preference rank; identical soft
   scores are broken by the preferred provider (Grok).
3. Secondary backends (Codex, Claude, Gemini) are authorized for
   *implementation* only when Grok has a durable hard-quota exhaustion latch
   (or an operator escape hatch).  Transient Grok capacity cooldowns never
   open a secondary implementer.
4. Explicit non-``auto`` pins bypass this selector entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.entrypoints.capability_resolver import (
    FALLBACK_PROVIDER,
    PREFERRED_PROVIDER,
    PreferredProviderCapability,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.cli_provider_balance import (
    CLAUDE_PROVIDER_ID,
    GEMINI_PROVIDER_ID,
    SECONDARY_IMPLEMENTATION_PREFERENCE,
)


IMPLEMENTATION_PROVIDER_AUTO_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/implementation-provider-auto@1"
)
TIE_BREAKER_PROVIDER = PREFERRED_PROVIDER  # grok
DEFAULT_IMPLEMENTATION_PROVIDER = "auto"

# Full observation set for auto routing (preferred + secondaries).
AUTO_OBSERVED_PROVIDERS: tuple[str, ...] = (
    PREFERRED_PROVIDER,
    CLAUDE_PROVIDER_ID,
    GEMINI_PROVIDER_ID,
    FALLBACK_PROVIDER,
)

# Preference rank: lower is better.  Grok is always first when eligible.
_PROVIDER_PREFERENCE_RANK: dict[str, int] = {
    PREFERRED_PROVIDER: 0,
    **{
        provider: index + 1
        for index, provider in enumerate(SECONDARY_IMPLEMENTATION_PREFERENCE)
    },
}


class AutoProviderDecision(str, Enum):
    """Closed set of auto-route outcomes."""

    GROK = "grok"
    CODEX = "codex"
    CLAUDE = "claude"
    GEMINI = "gemini"
    UNAVAILABLE = "unavailable"
    BACKOFF = "backoff"


class AutoProviderReason(str, Enum):
    """Stable reason codes for auto-route receipts."""

    PREFERRED_READY = "preferred_provider_ready"
    TIE_BREAK_PREFERRED = "tie_break_preferred_provider"
    FALLBACK_AFTER_QUOTA = "fallback_after_preferred_hard_quota"
    SECONDARY_AFTER_QUOTA = "secondary_after_preferred_hard_quota"
    PREFERRED_TRANSIENT_CAPACITY = "preferred_provider_transient_capacity"
    PREFERRED_NOT_READY = "preferred_provider_not_ready"
    FALLBACK_NOT_READY = "fallback_provider_not_ready"
    SECONDARY_NOT_READY = "secondary_providers_not_ready"
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
    secondary_providers: tuple[str, ...] = SECONDARY_IMPLEMENTATION_PREFERENCE
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
            "secondary_providers": list(self.secondary_providers),
            "tie_breaker": self.tie_breaker,
            "reason_codes": list(self.reason_codes),
            "retry_at": self.retry_at,
            "observations": [item.to_dict() for item in self.observations],
        }


def _decision_for_provider(provider_id: str) -> AutoProviderDecision:
    mapping = {
        PREFERRED_PROVIDER: AutoProviderDecision.GROK,
        FALLBACK_PROVIDER: AutoProviderDecision.CODEX,
        CLAUDE_PROVIDER_ID: AutoProviderDecision.CLAUDE,
        GEMINI_PROVIDER_ID: AutoProviderDecision.GEMINI,
    }
    return mapping.get(provider_id, AutoProviderDecision.UNAVAILABLE)


def observe_llm_router_backend(
    provider_id: str,
    *,
    binary_available: bool,
    authenticated: bool,
    provider_constructible: bool = True,
    source: str = "llm_router",
) -> BackendObservation:
    """Build one observation from non-charging llm_router / CLI readiness probes."""

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


def _latch_entry(
    latches: Mapping[str, Any] | None,
    family: str,
) -> dict[str, Any]:
    if not isinstance(latches, Mapping):
        return {}
    entry = latches.get(family)
    return dict(entry) if isinstance(entry, Mapping) else {}


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
        source=(
            source
            if (hard_quota_exhausted or capacity_restricted)
            else observation.source
        ),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _soft_rank_key(observation: BackendObservation) -> tuple[int, int, int]:
    """Lower is better (for ascending sort).

    Order: preference rank (Grok=0), inverted headroom, residual name.
    """

    rank = _PROVIDER_PREFERENCE_RANK.get(observation.provider_id, 100)
    headroom = (
        int(observation.request_headroom)
        if observation.request_headroom is not None
        else 0
    )
    return (rank, -headroom, observation.provider_id)


def _best_secondary(eligible: Sequence[BackendObservation]) -> BackendObservation | None:
    secondaries = [
        item
        for item in eligible
        if item.provider_id in SECONDARY_IMPLEMENTATION_PREFERENCE
    ]
    if not secondaries:
        return None
    return sorted(secondaries, key=_soft_rank_key)[0]


def select_implementation_provider(
    observations: Sequence[BackendObservation],
    *,
    global_capacity_latched: bool = False,
    allow_secondary_without_grok_quota: bool = False,
    allow_codex_without_grok_quota: bool = False,
) -> AutoProviderSelection:
    """Select the implementation backend from frozen observations.

    Parameters
    ----------
    observations:
        One entry per candidate backend (grok, claude, gemini, codex, …).
    global_capacity_latched:
        When true, no automatic provider may dispatch.
    allow_secondary_without_grok_quota / allow_codex_without_grok_quota:
        Test / operator escape hatches.  Production auto routing leaves these
        false so secondaries open only after Grok hard-quota evidence.
    """

    # Back-compat alias used by older call sites / tests.
    allow_secondary = bool(
        allow_secondary_without_grok_quota or allow_codex_without_grok_quota
    )

    obs = tuple(observations)
    by_id = {item.provider_id: item for item in obs}
    grok = by_id.get(PREFERRED_PROVIDER)

    if global_capacity_latched:
        return AutoProviderSelection(
            decision=AutoProviderDecision.BACKOFF,
            selected_provider="",
            reason_codes=(AutoProviderReason.GLOBAL_CAPACITY.value,),
            observations=obs,
            retry_at=next((item.retry_at for item in obs if item.retry_at), ""),
        )

    eligible = [item for item in obs if item.eligible]
    other_eligible = [
        item for item in eligible if item.provider_id != PREFERRED_PROVIDER
    ]

    # Grok ready → always preferred (explicit tie-break when others ready).
    if grok is not None and grok.eligible:
        reasons = [AutoProviderReason.PREFERRED_READY.value]
        if other_eligible:
            reasons.append(AutoProviderReason.TIE_BREAK_PREFERRED.value)
        return AutoProviderSelection(
            decision=AutoProviderDecision.GROK,
            selected_provider=PREFERRED_PROVIDER,
            reason_codes=tuple(reasons),
            observations=obs,
        )

    # Transient Grok capacity: backoff; never open secondaries without hard quota.
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

    # Durable Grok hard-quota exhaustion authorizes secondary implementers.
    if grok is not None and grok.hard_quota_exhausted:
        secondary = _best_secondary(eligible)
        if secondary is not None:
            return AutoProviderSelection(
                decision=_decision_for_provider(secondary.provider_id),
                selected_provider=secondary.provider_id,
                reason_codes=(
                    AutoProviderReason.FALLBACK_AFTER_QUOTA.value,
                    AutoProviderReason.SECONDARY_AFTER_QUOTA.value,
                ),
                observations=obs,
                retry_at=grok.retry_at,
            )

    # Optional escape hatch (tests / operator) — off by default.
    if allow_secondary:
        secondary = _best_secondary(eligible)
        if secondary is not None:
            return AutoProviderSelection(
                decision=_decision_for_provider(secondary.provider_id),
                selected_provider=secondary.provider_id,
                reason_codes=(AutoProviderReason.FALLBACK_AFTER_QUOTA.value,),
                observations=obs,
                retry_at=secondary.retry_at,
            )

    reasons: list[str] = [AutoProviderReason.NO_ELIGIBLE.value]
    if grok is not None and not grok.eligible:
        reasons.append(AutoProviderReason.PREFERRED_NOT_READY.value)
    if not any(
        item.eligible and item.provider_id in SECONDARY_IMPLEMENTATION_PREFERENCE
        for item in obs
    ):
        reasons.append(AutoProviderReason.SECONDARY_NOT_READY.value)
        reasons.append(AutoProviderReason.FALLBACK_NOT_READY.value)
    return AutoProviderSelection(
        decision=AutoProviderDecision.UNAVAILABLE,
        selected_provider="",
        reason_codes=tuple(dict.fromkeys(reasons)),
        observations=obs,
        retry_at=(
            (grok.retry_at if grok is not None else "")
            or next((item.retry_at for item in obs if item.retry_at), "")
        ),
    )


def _apply_usage_overlay(
    base: BackendObservation,
    usage: Mapping[str, Any],
    provider_id: str,
) -> BackendObservation:
    raw = usage.get(provider_id)
    if not isinstance(raw, Mapping):
        return base
    retry_at = str(raw.get("retry_at") or "")
    if not retry_at and raw.get("retry_after_seconds") is not None:
        # Leave as empty string; latches usually supply RFC3339 retry_at.
        retry_at = ""
    return merge_usage_observation(
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
        retry_at=retry_at,
        reason_codes=tuple(raw.get("reason_codes") or ()),
        source=str(raw.get("source") or "endpoint_usage"),
    )


def probe_llm_router_backends(
    *,
    grok_binary: bool,
    grok_authenticated: bool,
    grok_constructible: bool = True,
    codex_binary: bool,
    codex_authenticated: bool = True,
    claude_binary: bool = False,
    claude_authenticated: bool = False,
    gemini_binary: bool = False,
    gemini_authenticated: bool = False,
    latches: Mapping[str, Any] | None = None,
    usage_observations: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[BackendObservation, ...]:
    """Assemble multi-backend observations from probes + latches + usage overlays.

    Callers supply boolean probe results so this function stays free of
    import-time and network side effects.
    """

    specs: list[tuple[str, bool, bool, bool, str]] = [
        (
            PREFERRED_PROVIDER,
            grok_binary,
            grok_authenticated,
            grok_constructible,
            "llm_router",
        ),
        (
            CLAUDE_PROVIDER_ID,
            claude_binary,
            claude_authenticated,
            claude_binary and claude_authenticated,
            "cli_probe",
        ),
        (
            GEMINI_PROVIDER_ID,
            gemini_binary,
            gemini_authenticated,
            gemini_binary and gemini_authenticated,
            "cli_probe",
        ),
        (
            FALLBACK_PROVIDER,
            codex_binary,
            codex_authenticated,
            codex_binary,
            "llm_router",
        ),
    ]
    observations: list[BackendObservation] = []
    usage = usage_observations if isinstance(usage_observations, Mapping) else {}
    for provider_id, binary, auth, constructible, source in specs:
        base = observe_llm_router_backend(
            provider_id,
            binary_available=binary,
            authenticated=auth,
            provider_constructible=constructible,
            source=source,
        )
        base = merge_capacity_latch(base, _latch_entry(latches, provider_id))
        # Also accept family aliases for claude/gemini latches.
        if provider_id == CLAUDE_PROVIDER_ID:
            base = merge_capacity_latch(
                base, _latch_entry(latches, "claude_code")
            )
            base = merge_capacity_latch(
                base, _latch_entry(latches, "anthropic")
            )
        if provider_id == GEMINI_PROVIDER_ID:
            base = merge_capacity_latch(
                base, _latch_entry(latches, "gemini_cli")
            )
        base = _apply_usage_overlay(base, usage, provider_id)
        observations.append(base)
    return tuple(observations)


def select_auto_implementation_provider(
    *,
    grok_binary: bool,
    grok_authenticated: bool,
    grok_constructible: bool = True,
    codex_binary: bool,
    codex_authenticated: bool = True,
    claude_binary: bool = False,
    claude_authenticated: bool = False,
    gemini_binary: bool = False,
    gemini_authenticated: bool = False,
    latches: Mapping[str, Any] | None = None,
    usage_observations: Mapping[str, Mapping[str, Any]] | None = None,
    global_capacity_latched: bool = False,
    allow_secondary_without_grok_quota: bool = False,
    allow_codex_without_grok_quota: bool = False,
) -> AutoProviderSelection:
    """End-to-end auto selection from probe booleans + capacity latches."""

    observations = probe_llm_router_backends(
        grok_binary=grok_binary,
        grok_authenticated=grok_authenticated,
        grok_constructible=grok_constructible,
        codex_binary=codex_binary,
        codex_authenticated=codex_authenticated,
        claude_binary=claude_binary,
        claude_authenticated=claude_authenticated,
        gemini_binary=gemini_binary,
        gemini_authenticated=gemini_authenticated,
        latches=latches,
        usage_observations=usage_observations,
    )
    return select_implementation_provider(
        observations,
        global_capacity_latched=global_capacity_latched,
        allow_secondary_without_grok_quota=allow_secondary_without_grok_quota,
        allow_codex_without_grok_quota=allow_codex_without_grok_quota,
    )


def resolve_configured_implementation_provider(
    configured: str | None,
) -> str:
    """Normalize the configured provider pin; empty means auto."""

    value = str(configured or "").strip().lower()
    return value or DEFAULT_IMPLEMENTATION_PROVIDER


__all__ = [
    "AUTO_OBSERVED_PROVIDERS",
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
