"""Rollout gate for prompt bootstrap and rescue promotion controls."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.prompt_workflow_benchmark import (
    PromptWorkflowBenchmark,
    build_frozen_prompt_workflow_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.prompt_workflow_rollout import (
    PROMPT_WORKFLOW_BEHAVIOR_ID,
    PromptWorkflowRolloutEvaluation,
    PromptWorkflowRolloutMode,
    PromptWorkflowRolloutPolicy,
    build_default_prompt_workflow_binding,
    build_default_prompt_workflow_policy,
    evaluate_prompt_workflow_rollout,
    verify_prompt_workflow_rollout,
)


def _inputs():
    qualification = PromptWorkflowRolloutEvaluation(
        "evaluation:qualification@1",
        "2026-01-01T00:00:00Z",
        build_frozen_prompt_workflow_benchmark(observation_label="qualification"),
    )
    current = PromptWorkflowRolloutEvaluation(
        "evaluation:current@1",
        "2026-01-02T00:00:00Z",
        build_frozen_prompt_workflow_benchmark(observation_label="current"),
    )
    binding = build_default_prompt_workflow_binding()
    policy = build_default_prompt_workflow_policy(approve_automatic=True)
    return qualification, current, binding, policy


def test_automatic_requires_policy_and_a_later_separate_current_root():
    qualification, current, binding, policy = _inputs()

    missing = evaluate_prompt_workflow_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="automatic",
    )
    assert missing.effective_mode is PromptWorkflowRolloutMode.SHADOW
    assert missing.rollback_applied
    assert "current-evaluation-required" in missing.reason_codes

    unapproved = evaluate_prompt_workflow_rollout(
        qualification,
        binding=binding,
        policy=build_default_prompt_workflow_policy(approve_automatic=False),
        desired_mode="automatic",
        current_evaluation=current,
    )
    assert unapproved.effective_mode is PromptWorkflowRolloutMode.SHADOW
    assert "mode-not-policy-approved" in unapproved.reason_codes

    admitted = evaluate_prompt_workflow_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="automatic",
        current_evaluation=current,
    )
    assert admitted.effective_mode is PromptWorkflowRolloutMode.AUTOMATIC
    assert admitted.automatic_ready
    assert not admitted.authoritative
    assert not admitted.completion_authoritative
    assert verify_prompt_workflow_rollout(
        admitted,
        qualification,
        binding=binding,
        policy=policy,
        current_evaluation=current,
    )


def test_binding_or_safety_regression_returns_affected_behavior_to_shadow():
    qualification, current, binding, policy = _inputs()
    stale = replace(binding, tree_id="sha256:different-current-tree")
    decision = evaluate_prompt_workflow_rollout(
        qualification,
        binding=stale,
        policy=policy,
        desired_mode="automatic",
        current_evaluation=current,
    )
    assert decision.effective_mode is PromptWorkflowRolloutMode.SHADOW
    assert decision.affected_behavior_ids == (PROMPT_WORKFLOW_BEHAVIOR_ID,)
    assert "stale-binding:tree_id" in decision.reason_codes

    escaped = next(
        item
        for item in current.benchmark.receipts
        if item.adversarial_fixture is not None
    )
    unsafe_benchmark = PromptWorkflowBenchmark(
        tuple(
            replace(
                item,
                metrics=replace(item.metrics, escape_count=1),
            )
            if item.receipt_id == escaped.receipt_id
            else item
            for item in current.benchmark.receipts
        )
    )
    unsafe_current = replace(current, benchmark=unsafe_benchmark)
    unsafe = evaluate_prompt_workflow_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="automatic",
        current_evaluation=unsafe_current,
    )
    assert unsafe.effective_mode is PromptWorkflowRolloutMode.SHADOW
    assert any(
        code.startswith("current:adversarial-escape:")
        for code in unsafe.reason_codes
    )


def test_off_shadow_assist_and_population_narrowing():
    qualification, current, binding, policy = _inputs()

    off = evaluate_prompt_workflow_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=PromptWorkflowRolloutMode.OFF,
    )
    assert off.effective_mode is PromptWorkflowRolloutMode.OFF
    assert not off.rollback_applied

    shadow = evaluate_prompt_workflow_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=PromptWorkflowRolloutMode.SHADOW,
    )
    assert shadow.effective_mode is PromptWorkflowRolloutMode.SHADOW

    assist = evaluate_prompt_workflow_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=PromptWorkflowRolloutMode.ASSIST,
    )
    assert assist.effective_mode is PromptWorkflowRolloutMode.ASSIST
    assert assist.qualification_passed

    narrowed_benchmark = PromptWorkflowBenchmark(
        tuple(
            receipt
            for receipt in current.benchmark.receipts
            if receipt.adversarial_fixture is None
            or receipt.adversarial_fixture.value != "prompt-injection"
        )
    )
    narrowed = replace(current, benchmark=narrowed_benchmark)
    blocked = evaluate_prompt_workflow_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="automatic",
        current_evaluation=narrowed,
    )
    assert blocked.effective_mode is PromptWorkflowRolloutMode.SHADOW
    assert "benchmark-population-narrowed" in blocked.reason_codes or any(
        code.startswith("current:missing-adversarial-fixture:")
        for code in blocked.reason_codes
    )


def test_metric_regression_rolls_back_without_waiving_safety():
    qualification, current, binding, policy = _inputs()
    # Inflate model spend on the current observation relative to qualification.
    inflated = PromptWorkflowBenchmark(
        tuple(
            replace(
                receipt,
                metrics=replace(
                    receipt.metrics,
                    model_calls=receipt.metrics.model_calls + 5,
                    provider_input_tokens=receipt.metrics.provider_input_tokens
                    + 100,
                ),
            )
            if receipt.is_paired_path
            and receipt.planning_mode.value == "model"
            else receipt
            for receipt in current.benchmark.receipts
        )
    )
    regressed = replace(current, benchmark=inflated)
    decision = evaluate_prompt_workflow_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="automatic",
        current_evaluation=regressed,
    )
    assert decision.effective_mode is PromptWorkflowRolloutMode.SHADOW
    assert decision.rollback_applied
    assert any(
        code.startswith("metric-regression:") for code in decision.reason_codes
    )

    no_rollback_policy = PromptWorkflowRolloutPolicy(
        policy_id=policy.policy_id,
        policy_revision=policy.policy_revision,
        approved_behavior_ids=policy.approved_behavior_ids,
        approved_modes=policy.approved_modes,
        require_distinct_current_evaluation=True,
        rollback_on_metric_regression=False,
    )
    # Even without metric rollback, safety failure still returns to shadow.
    unsafe_item = next(
        item
        for item in regressed.benchmark.receipts
        if item.adversarial_fixture is not None
    )
    unsafe = PromptWorkflowBenchmark(
        tuple(
            replace(
                item,
                metrics=replace(item.metrics, escape_count=1),
            )
            if item.receipt_id == unsafe_item.receipt_id
            else item
            for item in regressed.benchmark.receipts
        )
    )
    still_blocked = evaluate_prompt_workflow_rollout(
        qualification,
        binding=binding,
        policy=no_rollback_policy,
        desired_mode="automatic",
        current_evaluation=replace(regressed, benchmark=unsafe),
    )
    assert still_blocked.effective_mode is PromptWorkflowRolloutMode.SHADOW
