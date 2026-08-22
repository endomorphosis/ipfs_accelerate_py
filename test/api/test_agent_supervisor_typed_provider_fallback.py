"""ASE2-002: exact quota-only Grok→Codex fallback policy and attempt evidence."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints import provider_route as route
from ipfs_accelerate_py.agent_supervisor.entrypoints.capability_resolver import (
    ProviderFallbackReceipt,
    _cid,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    ProviderFallbackReason,
    ProviderSelection,
)


def _quota_evidence(**overrides) -> route.QuotaExhaustionEvidence:
    payload = {
        "preferred_provider": "grok",
        "preferred_model_id": route.PRIMARY_MODEL_ID,
        "usage_evidence_cid": _cid("usage"),
        "observed_capability_cid": _cid("capability"),
        "observed_at_ms": 1_700_000_000_000,
    }
    payload.update(overrides)
    return route.QuotaExhaustionEvidence(**payload)


def test_default_policy_binds_exact_model_identities() -> None:
    policy = route.default_provider_route_policy()
    assert policy.preferred_model_id == "grok-4.5"
    assert policy.fallback_model_id == "gpt-5.6-terra"
    assert policy.fallback_reasoning_effort == "medium"
    assert policy.maximum_fallback_dispatches == 1
    assert policy.quota_only_fallback is True
    assert policy.pre_effect_only is True
    assert policy.content_id.startswith("b")


def test_policy_rejects_model_or_effort_drift() -> None:
    with pytest.raises(route.ProviderRouteError) as model_exc:
        route.ProviderRoutePolicy(preferred_model_id="grok-not-4.5")
    assert model_exc.value.reason_code == "model_identity_invalid"

    with pytest.raises(route.ProviderRouteError) as effort_exc:
        route.ProviderRoutePolicy(fallback_reasoning_effort="high")
    assert effort_exc.value.reason_code == "effort_identity_invalid"


def test_healthy_preferred_route_admits_grok() -> None:
    evaluation = route.evaluate_preferred_route()
    assert evaluation.admitted is True
    assert evaluation.selected_provider is ProviderSelection.GROK
    assert evaluation.selected_model_id == "grok-4.5"
    assert evaluation.fallback_reason is ProviderFallbackReason.NONE
    assert evaluation.reason_code == "admitted:grok-implement"


@pytest.mark.parametrize(
    "failure",
    sorted(route.FAIL_CLOSED_PREFERRED_FAILURES, key=lambda item: item.value),
)
def test_non_quota_preferred_failures_fail_closed(
    failure: route.PreferredFailureClass,
) -> None:
    evaluation = route.evaluate_preferred_route(
        preferred_healthy=False,
        preferred_failure=failure,
    )
    assert evaluation.admitted is False
    assert evaluation.selected_provider is ProviderSelection.UNAVAILABLE
    assert evaluation.reason_code == f"fail_closed:{failure.value}"


def test_quota_fallback_admits_exact_codex_identities() -> None:
    evaluation = route.evaluate_quota_fallback(quota_evidence=_quota_evidence())
    assert evaluation.admitted is True
    assert evaluation.selected_provider is ProviderSelection.CODEX
    assert evaluation.selected_model_id == "gpt-5.6-terra"
    assert evaluation.selected_reasoning_effort == "medium"
    assert (
        evaluation.fallback_reason
        is ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED
    )


def test_quota_fallback_rejects_post_effect_and_repeated() -> None:
    with pytest.raises(route.ProviderRouteError) as post_exc:
        route.evaluate_quota_fallback(
            quota_evidence=_quota_evidence(),
            repository_effect_observed=True,
        )
    assert post_exc.value.reason_code == "post_effect"

    with pytest.raises(route.ProviderRouteError) as repeated_exc:
        route.evaluate_quota_fallback(
            quota_evidence=_quota_evidence(),
            prior_fallback_dispatches=1,
        )
    assert repeated_exc.value.reason_code == "repeated_fallback"


def test_quota_fallback_rejects_prompt_selected_and_scope_widening() -> None:
    with pytest.raises(route.ProviderRouteError) as prompt_exc:
        route.evaluate_quota_fallback(
            quota_evidence=_quota_evidence(),
            prompt_selected_fallback=True,
        )
    assert prompt_exc.value.reason_code == "prompt_selected"

    with pytest.raises(route.ProviderRouteError) as scope_exc:
        route.evaluate_quota_fallback(
            quota_evidence=_quota_evidence(),
            scope_widened=True,
        )
    assert scope_exc.value.reason_code == "scope_widening"


def test_quota_fallback_rejects_model_and_effort_drift() -> None:
    with pytest.raises(route.ProviderRouteError) as model_exc:
        route.evaluate_quota_fallback(
            quota_evidence=_quota_evidence(),
            fallback_model_id="gpt-5.6-sol",
        )
    assert model_exc.value.reason_code == "model_drift"

    with pytest.raises(route.ProviderRouteError) as effort_exc:
        route.evaluate_quota_fallback(
            quota_evidence=_quota_evidence(),
            fallback_reasoning_effort="high",
        )
    assert effort_exc.value.reason_code == "effort_drift"


def test_stale_or_post_effect_quota_evidence_is_rejected() -> None:
    with pytest.raises(route.ProviderRouteError) as stale_exc:
        _quota_evidence(fresh=False)
    assert stale_exc.value.reason_code == "quota_evidence_stale"

    with pytest.raises(route.ProviderRouteError) as post_exc:
        _quota_evidence(post_effect=True)
    assert post_exc.value.reason_code == "quota_evidence_post_effect"


def test_fallback_receipt_and_review_continuation_forbid_self_review() -> None:
    receipt = route.build_fallback_receipt(
        quota_evidence=_quota_evidence(),
        task_revision_cid=_cid("task-rev"),
        budget_cid=_cid("budget"),
        attempt_id=_cid("attempt"),
        worktree_cid=_cid("worktree"),
        implementer_process_identity=_cid("implementer-proc"),
        review_authorization=_cid("review-auth"),
    )
    assert isinstance(receipt, ProviderFallbackReceipt)
    assert receipt.can_self_satisfy_independent_review() is False
    assert receipt.reason_code is ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED
    assert receipt.committed_before_dispatch is True

    continuation = route.build_independent_review_continuation(
        implementation_attempt_cid=_cid("attempt"),
        review_authorization=_cid("review-auth"),
        implementer_process_identity=_cid("implementer-proc"),
    )
    assert continuation.self_review_forbidden is True

    with pytest.raises(route.ProviderRouteError) as self_exc:
        route.build_independent_review_continuation(
            implementation_attempt_cid=_cid("same"),
            review_authorization=_cid("same"),
        )
    assert self_exc.value.reason_code == "self_review"


def test_provider_attempt_receipt_binds_policy_and_identities() -> None:
    policy = route.default_provider_route_policy()
    grok_attempt = route.ProviderAttemptReceipt(
        policy_cid=policy.content_id,
        provider="grok",
        model_id="grok-4.5",
        attempt_cid=_cid("grok-attempt"),
        worktree_cid=_cid("worktree"),
        task_revision_cid=_cid("task-rev"),
        process_identity=_cid("grok-proc"),
    )
    assert grok_attempt.provider == "grok"
    assert grok_attempt.reasoning_effort == ""

    codex_attempt = route.ProviderAttemptReceipt(
        policy_cid=policy.content_id,
        provider="codex",
        model_id="gpt-5.6-terra",
        attempt_cid=_cid("codex-attempt"),
        worktree_cid=_cid("worktree"),
        task_revision_cid=_cid("task-rev"),
        process_identity=_cid("codex-proc"),
        reasoning_effort="medium",
    )
    assert codex_attempt.reasoning_effort == "medium"

    with pytest.raises(route.ProviderRouteError) as drift_exc:
        route.ProviderAttemptReceipt(
            policy_cid=policy.content_id,
            provider="codex",
            model_id="gpt-5.6-sol",
            attempt_cid=_cid("bad"),
            worktree_cid=_cid("worktree"),
            task_revision_cid=_cid("task-rev"),
            process_identity=_cid("proc"),
            reasoning_effort="medium",
        )
    assert drift_exc.value.reason_code == "model_drift"


def test_provider_route_provenance_for_grok_and_codex() -> None:
    grok = route.build_provider_route_provenance(
        selected=ProviderSelection.GROK,
        observed_capability_cid=_cid("capability"),
        usage_evidence_cid=_cid("usage"),
        budget_cid=_cid("budget"),
        task_revision_cid=_cid("task-rev"),
        attempt_cid=_cid("attempt"),
        worktree_cid=_cid("worktree"),
    )
    assert grok.selected_provider is ProviderSelection.GROK
    assert grok.fallback_reason is ProviderFallbackReason.NONE

    receipt = route.build_fallback_receipt(
        quota_evidence=_quota_evidence(),
        task_revision_cid=_cid("task-rev"),
        budget_cid=_cid("budget"),
        attempt_id=_cid("attempt"),
        worktree_cid=_cid("worktree"),
        implementer_process_identity=_cid("implementer-proc"),
        review_authorization=_cid("review-auth"),
    )
    codex = route.build_provider_route_provenance(
        selected=ProviderSelection.CODEX,
        fallback_reason=ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED,
        fallback_receipt_cid=receipt.content_id,
        observed_capability_cid=_cid("capability"),
        usage_evidence_cid=_cid("usage"),
        budget_cid=_cid("budget"),
        task_revision_cid=_cid("task-rev"),
        attempt_cid=_cid("attempt"),
        worktree_cid=_cid("worktree"),
    )
    assert codex.selected_provider is ProviderSelection.CODEX
    assert codex.independent_review_required is True


def test_requirement_ids_are_stable_evidence_anchors() -> None:
    assert (
        route.EXACT_QUOTA_FALLBACK_REQUIREMENT_ID
        == "provider_route.EXACT_QUOTA_FALLBACK_REQUIREMENT_ID"
    )
    assert (
        route.TYPED_GROK_CODEX_FALLBACK_REQUIREMENT_ID
        == "provider_route.TYPED_GROK_CODEX_FALLBACK_REQUIREMENT_ID"
    )
