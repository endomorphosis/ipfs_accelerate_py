"""Automatic implementation-provider selection for the agent supervisor.

This module normalizes non-authoritative backend observations for ``auto``
implementation routing and adapts the sole router-owned decision from
:mod:`ipfs_accelerate_py.llm_router`.  It reuses:

* :mod:`ipfs_accelerate_py.llm_router` for eligibility, ranking, freshness,
  authentication/quota/capacity classification, authorization, and final
  allow/deny via :func:`decide_router_owned_implementation_provider`;
* readiness probes that do **not** charge usage (binary + auth, no generation);
* :mod:`cli_provider_balance` for Claude, Gemini, Meta Spark, Mistral, and
  Copilot readiness plus quota/balance *observation* helpers;
* durable capacity / hard-quota latches already maintained by the
  implementation daemon from classified provider failures;
* optional :mod:`endpoint_usage.adapters` observations when a caller has
  already normalized provider balance metadata.

This module must not retain an independent provider/model/trigger/effort
tuple, preferred-provider rank or tie-break table, authentication/quota
classifier, freshness rule, authorization branch, or final allow/deny path.
Explicit non-``auto`` pins bypass this selector entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.control.capability_resolver import (
    FALLBACK_PROVIDER,
    PREFERRED_PROVIDER,
    PreferredProviderCapability,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.cli_provider_balance import (
    CLAUDE_PROVIDER_ID,
    COPILOT_PROVIDER_ID,
    GEMINI_PROVIDER_ID,
    META_SPARK_PROVIDER_ID,
    MISTRAL_PROVIDER_ID,
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
    COPILOT_PROVIDER_ID,
    META_SPARK_PROVIDER_ID,
    MISTRAL_PROVIDER_ID,
    FALLBACK_PROVIDER,
)

class AutoProviderDecision(str, Enum):
    """Closed set of auto-route outcomes."""

    GROK = "grok"
    CODEX = "codex"
    CLAUDE = "claude"
    GEMINI = "gemini"
    COPILOT = "copilot"
    META_SPARK = "meta_spark"
    MISTRAL = "mistral"
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
        COPILOT_PROVIDER_ID: AutoProviderDecision.COPILOT,
        META_SPARK_PROVIDER_ID: AutoProviderDecision.META_SPARK,
        MISTRAL_PROVIDER_ID: AutoProviderDecision.MISTRAL,
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


def select_implementation_provider(
    observations: Sequence[BackendObservation],
    *,
    global_capacity_latched: bool = False,
    allow_secondary_without_grok_quota: bool = False,
    allow_codex_without_grok_quota: bool = False,
) -> AutoProviderSelection:
    """Select the implementation backend from frozen observations.

    Observation normalization stays local; eligibility, ranking, freshness,
    classification, authorization, and final allow/deny are owned solely by
    :func:`ipfs_accelerate_py.llm_router.decide_router_owned_implementation_provider`.

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

    from ipfs_accelerate_py.llm_router import (
        ROUTER_OWNED_COMPATIBILITY_LEGACY_AUTO,
        LegacyAutoProviderCompatibilityAdapter,
        RouterOwnedProviderObservation,
        decide_router_owned_implementation_provider,
    )

    # Back-compat alias used by older call sites / tests.
    allow_secondary = bool(
        allow_secondary_without_grok_quota or allow_codex_without_grok_quota
    )

    obs = tuple(observations)
    router_observations = tuple(
        RouterOwnedProviderObservation(
            provider_id=item.provider_id,
            ready=item.ready,
            authenticated=item.authenticated,
            binary_available=item.binary_available,
            hard_quota_exhausted=item.hard_quota_exhausted,
            capacity_latched=item.capacity_latched,
            request_headroom=item.request_headroom,
            retry_at=item.retry_at,
            source=item.source,
            reason_codes=item.reason_codes,
        )
        for item in obs
    )
    decision = decide_router_owned_implementation_provider(
        router_observations,
        preferred_provider=PREFERRED_PROVIDER,
        fallback_provider=FALLBACK_PROVIDER,
        secondary_providers=SECONDARY_IMPLEMENTATION_PREFERENCE,
        global_capacity_latched=global_capacity_latched,
        allow_secondary_without_preferred_quota=allow_secondary,
        compatibility_mode=ROUTER_OWNED_COMPATIBILITY_LEGACY_AUTO,
    )
    adapted = LegacyAutoProviderCompatibilityAdapter.to_selection_fields(decision)
    selected_provider = str(adapted["selected_provider"] or "")
    mapped_decision = str(adapted["decision"] or "unavailable")
    if mapped_decision == "backoff":
        auto_decision = AutoProviderDecision.BACKOFF
    elif mapped_decision == "unavailable" or not selected_provider:
        auto_decision = AutoProviderDecision.UNAVAILABLE
    else:
        auto_decision = _decision_for_provider(selected_provider)
    return AutoProviderSelection(
        decision=auto_decision,
        selected_provider=selected_provider,
        preferred_provider=str(
            adapted.get("preferred_provider") or PREFERRED_PROVIDER
        ),
        fallback_provider=str(
            adapted.get("fallback_provider") or FALLBACK_PROVIDER
        ),
        secondary_providers=tuple(
            adapted.get("secondary_providers")
            or SECONDARY_IMPLEMENTATION_PREFERENCE
        ),
        reason_codes=tuple(adapted.get("reason_codes") or ()),
        observations=obs,
        retry_at=str(adapted.get("retry_at") or ""),
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


_LATCH_ALIASES: dict[str, tuple[str, ...]] = {
    CLAUDE_PROVIDER_ID: ("claude_code", "anthropic"),
    GEMINI_PROVIDER_ID: ("gemini_cli",),
    META_SPARK_PROVIDER_ID: ("goose", "meta", "muse", "muse_spark", "spark"),
    MISTRAL_PROVIDER_ID: ("mistral_vibe", "vibe"),
    COPILOT_PROVIDER_ID: ("github_copilot",),
}


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
    copilot_binary: bool = False,
    copilot_authenticated: bool = False,
    meta_spark_binary: bool = False,
    meta_spark_authenticated: bool = False,
    mistral_binary: bool = False,
    mistral_authenticated: bool = False,
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
            COPILOT_PROVIDER_ID,
            copilot_binary,
            copilot_authenticated,
            copilot_binary and copilot_authenticated,
            "cli_probe",
        ),
        (
            META_SPARK_PROVIDER_ID,
            meta_spark_binary,
            meta_spark_authenticated,
            meta_spark_binary and meta_spark_authenticated,
            "cli_probe",
        ),
        (
            MISTRAL_PROVIDER_ID,
            mistral_binary,
            mistral_authenticated,
            mistral_binary and mistral_authenticated,
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
        for alias in _LATCH_ALIASES.get(provider_id, ()):
            base = merge_capacity_latch(base, _latch_entry(latches, alias))
        base = _apply_usage_overlay(base, usage, provider_id)
        for alias in _LATCH_ALIASES.get(provider_id, ()):
            base = _apply_usage_overlay(base, usage, alias)
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
    copilot_binary: bool = False,
    copilot_authenticated: bool = False,
    meta_spark_binary: bool = False,
    meta_spark_authenticated: bool = False,
    mistral_binary: bool = False,
    mistral_authenticated: bool = False,
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
        copilot_binary=copilot_binary,
        copilot_authenticated=copilot_authenticated,
        meta_spark_binary=meta_spark_binary,
        meta_spark_authenticated=meta_spark_authenticated,
        mistral_binary=mistral_binary,
        mistral_authenticated=mistral_authenticated,
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
