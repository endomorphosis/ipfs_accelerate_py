"""Claude / Gemini CLI quota-balance classification and readiness probes."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.todo_daemon.cli_provider_balance import (
    CLAUDE_PROVIDER_ID,
    GEMINI_PROVIDER_ID,
    classify_claude_cli_text,
    classify_gemini_cli_text,
    parse_cli_balance_observation,
    probe_claude_cli_readiness,
    probe_gemini_cli_readiness,
)


def test_claude_rate_limit_is_capacity_not_hard_quota() -> None:
    classified = classify_claude_cli_text(
        "Error: rate_limit_error: You've hit your usage limit. Retry after 60s"
    )
    assert classified.provider_id == CLAUDE_PROVIDER_ID
    assert classified.hard_quota_exhausted is False
    assert classified.capacity_restricted is True
    assert classified.failure_class == "rate_limited"
    assert classified.retry_after_seconds == 60
    assert any("usage_limit" in code for code in classified.reason_codes)


def test_claude_credit_balance_is_hard_quota() -> None:
    classified = classify_claude_cli_text(
        "API Error: Your credit balance is too low to access the Anthropic API"
    )
    assert classified.hard_quota_exhausted is True
    assert classified.capacity_restricted is True
    assert classified.failure_class == "hard_quota_exhausted"
    assert any("billing.exhausted" in code for code in classified.reason_codes)


def test_gemini_resource_exhausted_is_rate_limit() -> None:
    classified = classify_gemini_cli_text(
        "Error: 429 RESOURCE_EXHAUSTED: Quota exceeded for metric "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests"
    )
    assert classified.provider_id == GEMINI_PROVIDER_ID
    assert classified.hard_quota_exhausted is False
    assert classified.capacity_restricted is True
    assert classified.failure_class == "rate_limited"


def test_gemini_billing_disabled_is_hard_quota() -> None:
    classified = classify_gemini_cli_text(
        "Billing account is disabled for this project; payment required"
    )
    assert classified.hard_quota_exhausted is True
    assert classified.failure_class == "hard_quota_exhausted"


def test_gemini_auth_failure_not_quota() -> None:
    classified = classify_gemini_cli_text(
        "Error: API key not valid. Please pass a valid API key."
    )
    assert classified.authenticated_failure is True
    assert classified.hard_quota_exhausted is False
    assert classified.capacity_restricted is False
    assert classified.failure_class == "authentication"


def test_parse_cli_balance_observation_structured_claude() -> None:
    observation = parse_cli_balance_observation(
        "claude",
        kind="usage_limit",
        resets_in_seconds=120,
        usage={"input_tokens": 10, "output_tokens": 2},
    )
    assert observation["provider_id"] == CLAUDE_PROVIDER_ID
    assert observation["hard_quota_exhausted"] is False
    assert observation.get("capacity_restricted") or observation.get(
        "capacity_latched"
    )
    assert observation.get("retry_after_seconds") == 120
    assert any(
        "usage_limit" in str(code) for code in observation.get("reason_codes", [])
    )


def test_parse_cli_balance_observation_structured_gemini_billing() -> None:
    observation = parse_cli_balance_observation(
        "gemini",
        kind="quota_exceeded",
        resets_in_seconds=0,
    )
    assert observation["provider_id"] == GEMINI_PROVIDER_ID
    assert observation["hard_quota_exhausted"] is True


def test_readiness_probes_are_side_effect_free_dicts() -> None:
    claude = probe_claude_cli_readiness()
    gemini = probe_gemini_cli_readiness()
    assert claude["provider_id"] == CLAUDE_PROVIDER_ID
    assert gemini["provider_id"] == GEMINI_PROVIDER_ID
    assert "binary_available" in claude and "authenticated" in claude
    assert "binary_available" in gemini and "authenticated" in gemini
    assert "ready" in claude and "ready" in gemini
