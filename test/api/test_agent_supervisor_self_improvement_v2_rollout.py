from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.self_improvement_v2 import (
    REQUIRED_V2_OBJECTIVE_DIMENSIONS,
    V2ObjectiveDimension,
    build_frozen_v2_ablation_receipts,
    build_frozen_v2_self_evaluation_inputs,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.self_improvement_v2_rollout import (
    V2RolloutBinding,
    V2RolloutError,
    V2RolloutEvaluation,
    V2RolloutMode,
    V2RolloutPolicy,
    V2RolloutReport,
    evaluate_v2_rollout,
    recompute_v2_rollout_evaluation,
    verify_v2_rollout_report,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_v2_benchmark import (
    V2BenchmarkArm,
    V2FixtureKind,
    build_frozen_v2_paired_corpus,
    replace_v2_candidate_metrics,
)


QUALIFIED_AT = "2026-07-26T00:00:00Z"
CURRENT_AT = "2026-07-26T01:00:00Z"


def _evaluation(
    evaluation_id: str,
    evaluated_at: str,
    *,
    corpus=None,
    producers=None,
    ablations=None,
) -> V2RolloutEvaluation:
    if corpus is None:
        corpus = build_frozen_v2_paired_corpus()
    if producers is None:
        producers, default_ablations = (
            build_frozen_v2_self_evaluation_inputs(corpus)
        )
        if ablations is None:
            ablations = default_ablations
    if ablations is None:
        ablations = build_frozen_v2_ablation_receipts(corpus, producers)
    return V2RolloutEvaluation(
        evaluation_id=evaluation_id,
        evaluated_at=evaluated_at,
        corpus=corpus,
        producer_receipts=producers,
        ablation_receipts=ablations,
    )


def _inputs():
    qualification = _evaluation(
        "evaluation:qualification@1", QUALIFIED_AT
    )
    current = _evaluation("evaluation:current@1", CURRENT_AT)
    binding = V2RolloutBinding.from_corpus(qualification.corpus)
    automatic_policy = V2RolloutPolicy(
        allowed_modes=tuple(V2RolloutMode)
    )
    return qualification, current, binding, automatic_policy


def test_complete_report_recomputes_all_zero_and_threshold_gates():
    qualification, _, _, _ = _inputs()

    result = recompute_v2_rollout_evaluation(qualification)

    assert result.passed
    assert set(result.zero_failure_counts) == {
        "safety",
        "authority",
        "escaped-defect",
        "stale-hit",
        "idempotency",
        "population",
        "artifact-bound",
    }
    assert not any(result.zero_failure_counts.values())
    assert tuple(result.threshold_status) == (
        REQUIRED_V2_OBJECTIVE_DIMENSIONS
    )
    assert all(result.threshold_status.values())
    assert set(result.threshold_status) == {
        V2ObjectiveDimension.SAFETY,
        V2ObjectiveDimension.TOKENS,
        V2ObjectiveDimension.CONTEXT_REUSE,
        V2ObjectiveDimension.PLANNING,
        V2ObjectiveDimension.ANALYSIS,
        V2ObjectiveDimension.CACHE,
        V2ObjectiveDimension.VALIDATION,
        V2ObjectiveDimension.TASK_QUALITY,
        V2ObjectiveDimension.THROUGHPUT,
        V2ObjectiveDimension.PERSISTENCE,
        V2ObjectiveDimension.IDLE_RELIABILITY,
        V2ObjectiveDimension.CONTROL,
        V2ObjectiveDimension.REFILL,
    }
    assert result.failure_codes == ()


@pytest.mark.parametrize(
    ("desired", "expected"),
    (
        (V2RolloutMode.OFF, V2RolloutMode.OFF),
        (V2RolloutMode.SHADOW, V2RolloutMode.SHADOW),
        (V2RolloutMode.ASSIST, V2RolloutMode.ASSIST),
    ),
)
def test_off_shadow_and_assist_have_bound_deterministic_modes(
    desired, expected
):
    qualification, _, binding, policy = _inputs()

    report = evaluate_v2_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=desired,
    )

    assert report.desired_mode is desired
    assert report.effective_mode is expected
    assert report.qualification_gate_passed
    assert report.desired_binding_id != report.effective_binding_id or (
        desired is expected
    )
    assert report.to_dict()["binding"]["policy_id"] == binding.policy_id
    assert (
        report.to_dict()["binding"]["capability_id"]
        == binding.capability_id
    )


def test_automatic_requires_explicit_policy_approval():
    qualification, current, binding, _ = _inputs()

    report = evaluate_v2_rollout(
        qualification,
        binding=binding,
        desired_mode=V2RolloutMode.AUTOMATIC,
        current_evaluation=current,
    )

    assert report.effective_mode is V2RolloutMode.SHADOW
    assert not report.automatic_ready
    assert "policy-mode-not-approved:automatic" in report.reason_codes


def test_automatic_requires_a_later_separate_current_tree_evaluation():
    qualification, current, binding, policy = _inputs()

    missing = evaluate_v2_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=V2RolloutMode.AUTOMATIC,
    )
    replayed = evaluate_v2_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=V2RolloutMode.AUTOMATIC,
        current_evaluation=qualification,
    )
    promoted = evaluate_v2_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=V2RolloutMode.AUTOMATIC,
        current_evaluation=current,
    )

    assert missing.effective_mode is V2RolloutMode.SHADOW
    assert "current-tree-evaluation-required" in missing.reason_codes
    assert replayed.effective_mode is V2RolloutMode.SHADOW
    assert "current-evaluation-not-separate" in replayed.reason_codes
    assert "current-evaluation-not-later" in replayed.reason_codes
    assert promoted.effective_mode is V2RolloutMode.AUTOMATIC
    assert promoted.automatic_ready
    assert promoted.current_tree_gate_passed
    assert promoted.reason_codes == ()


def test_stale_policy_capability_or_tree_binding_rolls_back_to_shadow():
    qualification, current, binding, _ = _inputs()
    stale_binding = replace(
        binding,
        tree_id="sha256:" + "f" * 64,
        policy_revision="sha256:" + "e" * 64,
        capability_revision="sha256:" + "d" * 64,
    )
    stale_policy = V2RolloutPolicy(
        policy_id=stale_binding.policy_id,
        policy_revision=stale_binding.policy_revision,
        approved_capability_ids=(stale_binding.capability_id,),
        approved_behavior_ids=(stale_binding.behavior_id,),
        allowed_modes=tuple(V2RolloutMode),
    )

    report = evaluate_v2_rollout(
        qualification,
        binding=stale_binding,
        policy=stale_policy,
        desired_mode=V2RolloutMode.AUTOMATIC,
        current_evaluation=current,
    )

    assert report.effective_mode is V2RolloutMode.SHADOW
    assert report.rollback_applied
    assert "stale-binding:qualification" in report.reason_codes
    assert "stale-binding:current" in report.reason_codes
    assert report.to_dict()["affected_behavior_ids"] == [
        binding.behavior_id
    ]


def test_later_metric_deterioration_rolls_back_even_if_threshold_still_passes():
    qualification, _, binding, policy = _inputs()
    producers = []
    for receipt in qualification.producer_receipts:
        if (
            receipt.dimension is V2ObjectiveDimension.CACHE
            and receipt.arm is V2BenchmarkArm.CANDIDATE
        ):
            samples = dict(receipt.metric_samples)
            samples["warm-exact-reuse-rate"] = replace(
                samples["warm-exact-reuse-rate"], numerator=84
            )
            receipt = replace(receipt, metric_samples=samples)
        producers.append(receipt)
    current = _evaluation(
        "evaluation:current-regressed@1",
        CURRENT_AT,
        corpus=qualification.corpus,
        producers=tuple(producers),
    )

    assert recompute_v2_rollout_evaluation(current).passed
    report = evaluate_v2_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=V2RolloutMode.AUTOMATIC,
        current_evaluation=current,
    )

    assert report.effective_mode is V2RolloutMode.SHADOW
    assert report.rollback_applied
    assert (
        "regression:cache:warm-exact-reuse-rate"
        in report.reason_codes
    )


def test_any_noncompensable_failure_forces_shadow():
    qualification, _, binding, policy = _inputs()
    corpus = replace_v2_candidate_metrics(
        qualification.corpus,
        V2FixtureKind.STALE_CACHE,
        stale_authoritative_cache_hit_count=1,
    )
    current = _evaluation(
        "evaluation:current-stale-hit@1",
        CURRENT_AT,
        corpus=corpus,
    )

    report = evaluate_v2_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=V2RolloutMode.AUTOMATIC,
        current_evaluation=current,
    )

    assert report.effective_mode is V2RolloutMode.SHADOW
    assert report.rollback_applied
    assert report.current.zero_failure_counts["stale-hit"] > 0
    assert "current:zero-failure:stale-hit" in report.reason_codes
    assert "current:benchmark:cache-authority" in report.reason_codes


def test_report_restore_replays_sources_and_rejects_tampering():
    qualification, current, binding, policy = _inputs()
    report = evaluate_v2_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=V2RolloutMode.AUTOMATIC,
        current_evaluation=current,
    )

    restored = V2RolloutReport.from_json(
        report.to_json(),
        qualification=qualification,
        current=current,
    )
    assert restored == report
    assert verify_v2_rollout_report(
        report,
        qualification,
        current_evaluation=current,
    ) == report
    tampered = report.to_dict(include_report_id=True)
    tampered["effective_mode"] = "shadow"
    with pytest.raises(V2RolloutError, match="source replay"):
        V2RolloutReport.from_dict(
            tampered,
            qualification=qualification,
            current=current,
        )


def test_inconsistent_cross_fixture_identity_is_rejected():
    qualification, _, _, _ = _inputs()
    cases = list(qualification.corpus.cases)
    first = cases[0]
    changed_identity = replace(
        first.baseline.identity,
        capability_revision="sha256:" + "a" * 64,
    )
    baseline = replace(first.baseline, identity=changed_identity)
    candidate = replace(
        first.candidate,
        identity=replace(
            first.candidate.identity,
            capability_revision=changed_identity.capability_revision,
        ),
        causal_parent_ids=(baseline.receipt_id,),
    )
    cases[0] = replace(first, baseline=baseline, candidate=candidate)
    corpus = replace(qualification.corpus, cases=tuple(cases))
    producers, ablations = build_frozen_v2_self_evaluation_inputs(corpus)

    with pytest.raises(V2RolloutError, match="inconsistent"):
        V2RolloutEvaluation(
            evaluation_id="evaluation:identity-drift@1",
            evaluated_at=CURRENT_AT,
            corpus=corpus,
            producer_receipts=producers,
            ablation_receipts=ablations,
        )
