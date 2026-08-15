import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.provider_route import (
    FALLBACK_MODEL_ID, FALLBACK_REASONING_EFFORT, ProviderAttemptReceipt,
    ProviderRouteError, QuotaExhaustionEvidence, default_provider_route_policy,
    evaluate_quota_fallback,
)

def _evidence():
    return QuotaExhaustionEvidence("grok", "grok-4.6", "usage:one", "cap:one", 1, task_revision_cid="task:one", worktree_cid="worktree:one", budget_cid="budget:one", scope_cid="scope:one")

def _attempt():
    return ProviderAttemptReceipt(default_provider_route_policy().content_id, "codex", FALLBACK_MODEL_ID, "attempt:one", "worktree:one", "task:one", "process:one", FALLBACK_REASONING_EFFORT, budget_cid="budget:one", scope_cid="scope:one")

def test_only_fresh_equal_pre_effect_quota_evidence_admits_one_fallback():
    result = evaluate_quota_fallback(quota_evidence=_evidence(), attempt=_attempt(), now_ms=2)
    assert result.admitted and result.selected_provider.value == "codex"
    for kwargs in ({"prompt_selected_fallback": True}, {"prior_fallback_dispatches": 1}, {"repository_effect_observed": True}, {"now_ms": 400_000}):
        with pytest.raises(ProviderRouteError): evaluate_quota_fallback(quota_evidence=_evidence(), attempt=_attempt(), **kwargs)

def test_mismatched_scope_fails_closed():
    attempt = ProviderAttemptReceipt(default_provider_route_policy().content_id, "codex", FALLBACK_MODEL_ID, "attempt:one", "worktree:one", "task:one", "process:one", FALLBACK_REASONING_EFFORT, budget_cid="budget:one", scope_cid="scope:other")
    with pytest.raises(ProviderRouteError, match="exactly match"):
        evaluate_quota_fallback(quota_evidence=_evidence(), attempt=attempt)
