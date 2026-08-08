"""Focused fail-closed contracts for the canonical implementation route."""
from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py import llm_router


def test_high_route_never_accepts_an_ambient_six_field_profile() -> None:
    with pytest.raises(ValueError, match="authorization"):
        llm_router.resolve_agent_implementation_route(
            primary_provider_id="grok_cli",
            primary_model_id="grok-4.5",
            fallback_provider_id="codex",
            fallback_model_id="gpt-5.6-terra",
            fallback_trigger="primary_quota_or_auth_unavailable",
            fallback_reasoning_effort="high",
        )


def test_protected_bootstrap_artifact_is_denied_until_it_binds_a_reviewer() -> None:
    with pytest.raises(ValueError):
        llm_router.load_agent_implementation_route_authorization(
            repo_root=Path.cwd(),
            artifact_path=(
                "data/agent_supervisor/prompt_only_self_improvement_v3/"
                "convergence/provider_fallback_policy_authorization_20260808.json"
            ),
            board_namespace="agent-supervisor-prompt-only-self-improvement-v3",
        )
