from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.decision_runtime_benchmark import (
    DecisionRuntimeBenchmark,
    build_frozen_decision_runtime_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.decision_runtime_rollout import (
    DecisionRuntimeRolloutBinding,
    DecisionRuntimeRolloutEvaluation,
    DecisionRuntimeRolloutMode,
    DecisionRuntimeRolloutPolicy,
    evaluate_decision_runtime_rollout,
    verify_decision_runtime_rollout,
)


def _inputs():
    qualification = DecisionRuntimeRolloutEvaluation(
        "evaluation:qualification@1",
        "2026-01-01T00:00:00Z",
        build_frozen_decision_runtime_benchmark(observation_label="qualification"),
    )
    current = DecisionRuntimeRolloutEvaluation(
        "evaluation:current@1",
        "2026-01-02T00:00:00Z",
        build_frozen_decision_runtime_benchmark(observation_label="current"),
    )
    binding = DecisionRuntimeRolloutBinding(
        repository_id="repository:proof-runtime-benchmark@1",
        tree_id="sha256:frozen-proof-runtime-tree",
        behavior_id="behavior:proof-directed-decision-runtime",
        objective_id="ASI-G360",
        objective_revision="sha256:frozen-objective",
        policy_id="policy:proof-runtime-rollout@1",
        policy_revision="sha256:frozen-policy",
        capability_id="capability:proof-runtime-local@1",
        capability_revision="sha256:frozen-capability",
    )
    policy = DecisionRuntimeRolloutPolicy(
        policy_id=binding.policy_id,
        policy_revision=binding.policy_revision,
        approved_behavior_ids=(binding.behavior_id,),
        approved_modes=tuple(DecisionRuntimeRolloutMode),
    )
    return qualification, current, binding, policy


def test_automatic_requires_policy_and_a_later_separate_current_root():
    qualification, current, binding, policy = _inputs()

    missing = evaluate_decision_runtime_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="automatic",
    )
    assert missing.effective_mode is DecisionRuntimeRolloutMode.SHADOW
    assert missing.rollback_applied
    assert "current-evaluation-required" in missing.reason_codes

    admitted = evaluate_decision_runtime_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="automatic",
        current_evaluation=current,
    )
    assert admitted.effective_mode is DecisionRuntimeRolloutMode.AUTOMATIC
    assert admitted.automatic_ready
    assert not admitted.authoritative
    assert verify_decision_runtime_rollout(
        admitted,
        qualification,
        binding=binding,
        policy=policy,
        current_evaluation=current,
    )


def test_binding_or_safety_regression_returns_affected_behavior_to_shadow():
    qualification, current, binding, policy = _inputs()
    stale = replace(binding, tree_id="sha256:different-current-tree")
    decision = evaluate_decision_runtime_rollout(
        qualification,
        binding=stale,
        policy=policy,
        desired_mode="automatic",
        current_evaluation=current,
    )
    assert decision.effective_mode is DecisionRuntimeRolloutMode.SHADOW
    assert decision.affected_behavior_ids == (binding.behavior_id,)
    assert "stale-binding:tree_id" in decision.reason_codes

    escaped = next(
        item
        for item in current.benchmark.receipts
        if item.adversarial_fixture is not None
    )
    unsafe_benchmark = DecisionRuntimeBenchmark(
        tuple(
            replace(item, escape_count=1)
            if item.receipt_id == escaped.receipt_id
            else item
            for item in current.benchmark.receipts
        )
    )
    unsafe_current = replace(current, benchmark=unsafe_benchmark)
    unsafe = evaluate_decision_runtime_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="automatic",
        current_evaluation=unsafe_current,
    )
    assert unsafe.effective_mode is DecisionRuntimeRolloutMode.SHADOW
    assert any(code.startswith("current:adversarial-escape:") for code in unsafe.reason_codes)
