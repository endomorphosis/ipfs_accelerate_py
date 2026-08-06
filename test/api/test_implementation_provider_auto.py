"""Auto implementation-provider selection: Grok default, Codex after quota."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.entrypoints.capability_resolver import (
    FALLBACK_PROVIDER,
    PREFERRED_PROVIDER,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_provider_auto import (
    AutoProviderDecision,
    AutoProviderReason,
    BackendObservation,
    merge_capacity_latch,
    select_auto_implementation_provider,
    select_implementation_provider,
)


def _obs(
    provider_id: str,
    *,
    ready: bool = True,
    authenticated: bool = True,
    binary_available: bool = True,
    hard_quota_exhausted: bool = False,
    capacity_latched: bool = False,
    request_headroom: int | None = 10,
    retry_at: str = "",
) -> BackendObservation:
    return BackendObservation(
        provider_id=provider_id,
        ready=ready,
        authenticated=authenticated,
        binary_available=binary_available,
        hard_quota_exhausted=hard_quota_exhausted,
        capacity_latched=capacity_latched,
        request_headroom=request_headroom,
        retry_at=retry_at,
    )


def test_both_ready_tie_breaks_to_grok() -> None:
    selection = select_implementation_provider(
        (
            _obs(PREFERRED_PROVIDER, request_headroom=1),
            _obs(FALLBACK_PROVIDER, request_headroom=100),
        )
    )
    assert selection.decision is AutoProviderDecision.GROK
    assert selection.selected_provider == PREFERRED_PROVIDER
    assert AutoProviderReason.TIE_BREAK_PREFERRED.value in selection.reason_codes
    assert selection.tie_breaker == PREFERRED_PROVIDER


def test_grok_only_ready_selects_grok() -> None:
    selection = select_implementation_provider(
        (
            _obs(PREFERRED_PROVIDER),
            _obs(FALLBACK_PROVIDER, ready=False, binary_available=False),
        )
    )
    assert selection.decision is AutoProviderDecision.GROK
    assert AutoProviderReason.PREFERRED_READY.value in selection.reason_codes


def test_grok_hard_quota_opens_codex() -> None:
    selection = select_implementation_provider(
        (
            _obs(
                PREFERRED_PROVIDER,
                ready=False,
                hard_quota_exhausted=True,
                capacity_latched=True,
            ),
            _obs(FALLBACK_PROVIDER),
        )
    )
    assert selection.decision is AutoProviderDecision.CODEX
    assert selection.selected_provider == FALLBACK_PROVIDER
    assert (
        AutoProviderReason.FALLBACK_AFTER_QUOTA.value in selection.reason_codes
    )


def test_grok_transient_capacity_does_not_open_codex() -> None:
    selection = select_implementation_provider(
        (
            _obs(
                PREFERRED_PROVIDER,
                ready=False,
                capacity_latched=True,
                hard_quota_exhausted=False,
                retry_at="2026-08-10T17:16:00+00:00",
            ),
            _obs(FALLBACK_PROVIDER),
        )
    )
    assert selection.decision is AutoProviderDecision.BACKOFF
    assert selection.selected_provider == ""
    assert (
        AutoProviderReason.PREFERRED_TRANSIENT_CAPACITY.value
        in selection.reason_codes
    )
    assert selection.retry_at == "2026-08-10T17:16:00+00:00"


def test_global_capacity_backoff() -> None:
    selection = select_implementation_provider(
        (_obs(PREFERRED_PROVIDER), _obs(FALLBACK_PROVIDER)),
        global_capacity_latched=True,
    )
    assert selection.decision is AutoProviderDecision.BACKOFF
    assert AutoProviderReason.GLOBAL_CAPACITY.value in selection.reason_codes


def test_neither_ready_unavailable() -> None:
    selection = select_implementation_provider(
        (
            _obs(PREFERRED_PROVIDER, ready=False, authenticated=False),
            _obs(FALLBACK_PROVIDER, ready=False, binary_available=False),
        )
    )
    assert selection.decision is AutoProviderDecision.UNAVAILABLE
    assert AutoProviderReason.NO_ELIGIBLE.value in selection.reason_codes


def test_merge_capacity_latch_marks_hard_quota() -> None:
    base = _obs(PREFERRED_PROVIDER)
    merged = merge_capacity_latch(
        base,
        {
            "active": True,
            "hard_quota_exhausted": True,
            "retry_at": "2026-08-10T17:16:00+00:00",
        },
    )
    assert merged.hard_quota_exhausted is True
    assert merged.capacity_latched is False  # hard quota, not transient
    assert merged.retry_at.startswith("2026-08-10")


def test_select_auto_from_probe_booleans_prefers_grok() -> None:
    selection = select_auto_implementation_provider(
        grok_binary=True,
        grok_authenticated=True,
        grok_constructible=True,
        codex_binary=True,
        latches={},
    )
    assert selection.decision is AutoProviderDecision.GROK
    assert selection.selected_provider == "grok"


def test_select_auto_codex_after_grok_quota_latch() -> None:
    selection = select_auto_implementation_provider(
        grok_binary=True,
        grok_authenticated=True,
        grok_constructible=True,
        codex_binary=True,
        latches={
            "grok": {
                "active": True,
                "hard_quota_exhausted": True,
                "retry_at": "2026-08-10T17:16:00+00:00",
            }
        },
    )
    assert selection.decision is AutoProviderDecision.CODEX
    assert selection.selected_provider == "codex"


def test_codex_usage_limit_does_not_override_ready_grok() -> None:
    """Codex being quota-exhausted must not prevent Grok selection."""

    selection = select_auto_implementation_provider(
        grok_binary=True,
        grok_authenticated=True,
        grok_constructible=True,
        codex_binary=True,
        latches={
            "codex": {
                "active": True,
                "hard_quota_exhausted": True,
                "retry_at": "2026-08-10T17:16:00+00:00",
            }
        },
    )
    assert selection.decision is AutoProviderDecision.GROK


def test_all_secondaries_ready_still_tie_breaks_to_grok() -> None:
    selection = select_implementation_provider(
        (
            _obs(PREFERRED_PROVIDER, request_headroom=1),
            _obs("claude", request_headroom=1000),
            _obs("gemini", request_headroom=1000),
            _obs(FALLBACK_PROVIDER, request_headroom=1000),
        )
    )
    assert selection.decision is AutoProviderDecision.GROK
    assert AutoProviderReason.TIE_BREAK_PREFERRED.value in selection.reason_codes
    assert {item.provider_id for item in selection.observations} >= {
        "grok",
        "claude",
        "gemini",
        "codex",
    }


def test_grok_hard_quota_prefers_codex_over_claude_when_both_ready() -> None:
    selection = select_implementation_provider(
        (
            _obs(
                PREFERRED_PROVIDER,
                ready=False,
                hard_quota_exhausted=True,
            ),
            _obs("claude", request_headroom=50),
            _obs(FALLBACK_PROVIDER, request_headroom=10),
        )
    )
    assert selection.decision is AutoProviderDecision.CODEX
    assert selection.selected_provider == "codex"


def test_grok_hard_quota_opens_claude_when_codex_latched() -> None:
    selection = select_implementation_provider(
        (
            _obs(
                PREFERRED_PROVIDER,
                ready=False,
                hard_quota_exhausted=True,
            ),
            _obs(
                FALLBACK_PROVIDER,
                ready=False,
                capacity_latched=True,
            ),
            _obs("claude"),
            _obs("gemini", ready=False),
        )
    )
    assert selection.decision is AutoProviderDecision.CLAUDE
    assert selection.selected_provider == "claude"
    assert AutoProviderReason.SECONDARY_AFTER_QUOTA.value in selection.reason_codes


def test_grok_hard_quota_opens_gemini_when_higher_secondaries_unavailable() -> None:
    selection = select_implementation_provider(
        (
            _obs(
                PREFERRED_PROVIDER,
                ready=False,
                hard_quota_exhausted=True,
            ),
            _obs(FALLBACK_PROVIDER, ready=False, binary_available=False),
            _obs("claude", ready=False, capacity_latched=True),
            _obs("gemini"),
        )
    )
    assert selection.decision is AutoProviderDecision.GEMINI
    assert selection.selected_provider == "gemini"


def test_select_auto_includes_claude_gemini_observations() -> None:
    selection = select_auto_implementation_provider(
        grok_binary=True,
        grok_authenticated=True,
        codex_binary=True,
        claude_binary=True,
        claude_authenticated=True,
        gemini_binary=True,
        gemini_authenticated=True,
        latches={
            "codex": {"active": True, "hard_quota_exhausted": False},
        },
    )
    assert selection.decision is AutoProviderDecision.GROK
    by_id = {item.provider_id: item for item in selection.observations}
    assert by_id["claude"].eligible is True
    assert by_id["gemini"].eligible is True
    assert by_id["codex"].capacity_latched is True


def test_claude_usage_observation_overlay_restricts_eligibility() -> None:
    selection = select_auto_implementation_provider(
        grok_binary=True,
        grok_authenticated=True,
        grok_constructible=True,
        codex_binary=False,
        claude_binary=True,
        claude_authenticated=True,
        latches={
            "grok": {
                "active": True,
                "hard_quota_exhausted": True,
            }
        },
        usage_observations={
            "claude": {
                "capacity_restricted": True,
                "reason_codes": ["subscription.usage_limit", "cli.claude"],
            }
        },
    )
    # Claude restricted; no other secondary → unavailable
    assert selection.decision is AutoProviderDecision.UNAVAILABLE
    claude = next(
        item
        for item in selection.observations
        if item.provider_id == "claude"
    )
    assert claude.eligible is False
    assert claude.capacity_latched is True
