from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AttributionCause,
    AutonomyContractError,
    CausalAttribution,
    DistillationStatus,
    DistilledDecisionRule,
    ExperienceEpisode,
    MetaAction,
    TerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.policy_distillation import (
    DECLARATIVE_WHEN_KEYS,
    DISTILLED_DECISION_RULE_INTERFACE,
    MIN_INDEPENDENT_DEVELOPMENT_EXAMPLES,
    POLICY_DISTILLER_INTERFACE,
    AdversarialMutation,
    DistillationDisposition,
    DistillationExample,
    DistillationGate,
    DistillationPartition,
    PolicyDistillationError,
    PolicyDistiller,
    common_feature_conjunction,
    field_is_forbidden,
    generate_aae_mutations,
    rule_is_narrower_than_evidence,
)


def _episode(**overrides: Any) -> ExperienceEpisode:
    values: dict[str, Any] = {
        "frozen_input_ids": ("tree-1", "objective-rev-1"),
        "question_feature_ids": ("feature-cache",),
        "selected_action": MetaAction.READ_CACHED_RECEIPT,
        "selection_policy_id": "policy-1",
        "selection_policy_version": "policy-rev-1",
        "terminal_status": TerminalStatus.SUCCEEDED,
        "context_metrics": {"input_tokens": 12, "prefix_reused_tokens": 4},
        "evidence_ids": ("evidence-static-1",),
        "accepted_criterion_ids": ("AC-1",),
        "validation_receipt_ids": ("validation-1",),
        "latency_ms": 40,
        "cost_micros": 0,
    }
    values.update(overrides)
    return ExperienceEpisode(**values)


def _attribution(episode: ExperienceEpisode, **overrides: Any) -> CausalAttribution:
    values: dict[str, Any] = {
        "episode_ids": (episode.episode_id,),
        "primary_cause": AttributionCause.INCORRECT_CACHE_REUSE,
        "evidence_ids": ("attribution-evidence-1",),
        "confidence_bp": 9_000,
        "shadow_only": True,
    }
    values.update(overrides)
    return CausalAttribution(**values)


def _features(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "risk_class": "R1_READ_ONLY",
        "context_confidence": "current",
        "task_class": "cache_reuse",
        "language": "python",
        "repository_family": "ipfs_accelerate_py",
    }
    values.update(overrides)
    return values


def _example(
    *,
    index: int,
    partition: DistillationPartition = DistillationPartition.DEVELOPMENT,
    features: dict[str, Any] | None = None,
    action: MetaAction = MetaAction.READ_CACHED_RECEIPT,
    decision_class: str = "cache_reuse",
    independence_id: str | None = None,
    **episode_overrides: Any,
) -> DistillationExample:
    frozen = episode_overrides.pop("frozen_input_ids", (f"tree-{index}", "objective-rev-1"))
    validation_receipt_ids = episode_overrides.pop(
        "validation_receipt_ids", (f"validation-{index}",)
    )
    evidence_ids = episode_overrides.pop("evidence_ids", (f"evidence-{index}",))
    episode = _episode(
        frozen_input_ids=frozen,
        selected_action=action,
        validation_receipt_ids=validation_receipt_ids,
        evidence_ids=evidence_ids,
        **episode_overrides,
    )
    return DistillationExample(
        example_id=f"example-{partition.value}-{index}",
        decision_class=decision_class,
        features=features or _features(),
        action=action,
        episode=episode,
        attribution=_attribution(episode, evidence_ids=(f"attribution-{index}",)),
        partition=partition,
        independence_id=independence_id or f"independent-{index}",
    )


def _stable_examples(
    *,
    development: int = 3,
    held_out: int = 1,
    counterexamples: int = 0,
    features: dict[str, Any] | None = None,
    action: MetaAction = MetaAction.READ_CACHED_RECEIPT,
    decision_class: str = "cache_reuse",
) -> list[DistillationExample]:
    items = [
        _example(
            index=index,
            partition=DistillationPartition.DEVELOPMENT,
            features=features,
            action=action,
            decision_class=decision_class,
        )
        for index in range(development)
    ]
    items.extend(
        _example(
            index=100 + index,
            partition=DistillationPartition.HELD_OUT,
            features=features,
            action=action,
            decision_class=decision_class,
        )
        for index in range(held_out)
    )
    items.extend(
        _example(
            index=200 + index,
            partition=DistillationPartition.COUNTEREXAMPLE,
            features=features,
            action=action,
            decision_class=decision_class,
            terminal_status=TerminalStatus.FAILED,
            accepted_criterion_ids=(),
            validation_receipt_ids=(),
            failure_signature="counterexample",
        )
        for index in range(counterexamples)
    )
    return items


def test_interfaces_are_versioned_and_when_keys_are_closed() -> None:
    distiller = PolicyDistiller()
    assert POLICY_DISTILLER_INTERFACE == "PolicyDistiller@1"
    assert DISTILLED_DECISION_RULE_INTERFACE == "DistilledDecisionRule@1"
    assert distiller.INTERFACE == POLICY_DISTILLER_INTERFACE
    assert distiller.RULE_INTERFACE == DISTILLED_DECISION_RULE_INTERFACE
    assert DistilledDecisionRule.SCHEMA.endswith("distilled-decision-rule@1")
    assert DECLARATIVE_WHEN_KEYS == {
        "context_confidence",
        "failure_signature",
        "language",
        "proof_requirements",
        "provider_health",
        "repository_family",
        "required_capabilities",
        "risk_class",
        "task_class",
        "token_budget",
    }
    assert MIN_INDEPENDENT_DEVELOPMENT_EXAMPLES >= 3


def test_stable_class_emits_shadow_rule_narrower_than_evidence_with_fallback() -> None:
    distiller = PolicyDistiller()
    examples = _stable_examples()
    detected = distiller.detect_stable_classes(examples)
    assert [item.decision_class for item in detected] == ["cache_reuse"]
    result = distiller.distill(examples)
    assert result.disposition is DistillationDisposition.SHADOW_CANDIDATE
    assert result.passed is True
    assert result.rule is not None
    assert result.candidate is not None
    assert result.candidate.status is DistillationStatus.SHADOW
    assert result.rule.shadow_only is True
    assert result.rule.authorized_promotion_id == ""
    assert result.rule.action is MetaAction.READ_CACHED_RECEIPT
    assert result.rule.fallback is MetaAction.CALL_LOCAL_SMALL_MODEL
    assert result.rule.action is not result.rule.fallback
    assert "cannot_self_promote" in result.reason_codes
    assert "narrower_than_evidence" in result.reason_codes
    assert "out_of_domain_fallback" in result.reason_codes
    positives = [item.features for item in examples if item.partition is DistillationPartition.DEVELOPMENT]
    assert rule_is_narrower_than_evidence(result.rule.when, positives)
    for key, value in common_feature_conjunction(positives).items():
        assert result.rule.when[key] == value
    replayed = DistilledDecisionRule.from_dict(result.rule.to_dict())
    assert replayed.rule_id == result.rule.rule_id
    assert replayed.shadow_only is True


def test_independent_example_threshold_blocks_distillation() -> None:
    distiller = PolicyDistiller()
    examples = _stable_examples(development=2, held_out=1)
    result = distiller.distill(examples)
    assert result.disposition is DistillationDisposition.INSUFFICIENT_EVIDENCE
    assert "independent_example_threshold" in result.reason_codes
    assert result.rule is None
    assert distiller.detect_stable_classes(examples) == ()

    first = _example(index=0)
    clone = _example(index=1, independence_id=first.independence_id)
    third = _example(index=2)
    held = _example(index=100, partition=DistillationPartition.HELD_OUT)
    result = distiller.distill((first, clone, third, held))
    assert result.disposition is DistillationDisposition.INSUFFICIENT_EVIDENCE
    assert "independent_example_threshold" in result.reason_codes


def test_unstable_output_and_unstable_features_are_rejected() -> None:
    distiller = PolicyDistiller()
    mixed = _stable_examples()
    mixed[2] = _example(index=2, action=MetaAction.RUN_SELECTED_TEST)
    result = distiller.distill(mixed)
    assert result.disposition is DistillationDisposition.DEVELOPMENT_FAILED
    assert "unstable_output" in result.reason_codes
    assert result.rule is None

    drifting = [
        _example(
            index=0,
            features=_features(
                language="python",
                risk_class="R1_READ_ONLY",
                task_class="cache_reuse",
                context_confidence="current",
                repository_family="ipfs_accelerate_py",
            ),
        ),
        _example(
            index=1,
            features=_features(
                language="go",
                risk_class="R2_REVERSIBLE_LOCAL",
                task_class="repair",
                context_confidence="stale",
                repository_family="other_family",
            ),
        ),
        _example(
            index=2,
            features=_features(
                language="rust",
                risk_class="R3_BOUNDED_REPOSITORY_MUTATION",
                task_class="proof",
                context_confidence="unknown",
                repository_family="third_family",
            ),
        ),
        _example(index=100, partition=DistillationPartition.HELD_OUT, features=_features(language="python")),
    ]
    result = distiller.distill(drifting)
    assert result.disposition is DistillationDisposition.INSUFFICIENT_EVIDENCE
    assert "unstable_features" in result.reason_codes


@pytest.mark.parametrize(
    "payload",
    (
        {"when": {"python_code": "import os"}, "action": MetaAction.NO_OP.value},
        {"kind": "python", "body": "def decide(x):\n    return x"},
        {"when": {"risk_class": "lambda features: True"}, "action": MetaAction.NO_OP.value},
        {"executable_code": "print(1)", "when": {"risk_class": "R1_READ_ONLY"}},
        {"when": {"eval": "os.system"}, "action": MetaAction.NO_OP.value},
        {"language": "python", "source_body": "import subprocess"},
    ),
)
def test_arbitrary_python_and_model_generated_executable_policy_is_rejected(payload: dict[str, Any]) -> None:
    distiller = PolicyDistiller()
    result = distiller.distill(_stable_examples(), proposed_rule=payload)
    assert result.disposition is DistillationDisposition.REJECTED
    assert "dsl_rejected" in result.reason_codes or "executable_policy_rejected" in result.reason_codes
    assert result.rule is None
    assert result.candidate is None


def test_unsupported_when_keys_are_dsl_rejected() -> None:
    distiller = PolicyDistiller()
    result = distiller.distill(
        _stable_examples(),
        proposed_rule={"when": {"arbitrary_predicate": "yes"}, "action": MetaAction.READ_CACHED_RECEIPT.value},
    )
    assert result.disposition is DistillationDisposition.REJECTED
    assert "dsl_rejected" in result.reason_codes
    with pytest.raises(AutonomyContractError, match="declarative"):
        DistilledDecisionRule(
            version="v1",
            when={"python_code": "import os"},
            action=MetaAction.NO_OP,
            required_validation_ids=("validation-1",),
            fallback=MetaAction.REQUEST_HUMAN_DECISION,
            scope={},
            source_episode_ids=("episode-1",),
            held_out_evaluation_ids=("held-out-1",),
        )


def test_broad_proposal_is_narrowed_to_evidence_conjunction() -> None:
    distiller = PolicyDistiller()
    examples = _stable_examples()
    result = distiller.distill(
        examples,
        proposed_rule={
            "when": {"risk_class": "R1_READ_ONLY"},
            "action": MetaAction.READ_CACHED_RECEIPT.value,
            "fallback": MetaAction.CALL_LOCAL_SMALL_MODEL.value,
        },
    )
    assert result.disposition is DistillationDisposition.SHADOW_CANDIDATE
    assert result.rule is not None
    assert "proposal_was_broader_than_evidence" in result.gate_receipts[1].reason_codes or (
        "counterexample_narrowed" in result.gate_receipts[1].reason_codes
    )
    assert set(result.rule.when) >= {
        "risk_class",
        "context_confidence",
        "task_class",
        "language",
        "repository_family",
    }
    positives = [item.features for item in examples if item.partition is DistillationPartition.DEVELOPMENT]
    assert rule_is_narrower_than_evidence(result.rule.when, positives)
    assert not rule_is_narrower_than_evidence({"risk_class": "R1_READ_ONLY"}, positives)


def test_counterexample_narrows_or_rejects_indistinguishable_cases() -> None:
    distiller = PolicyDistiller()
    counter = _example(
        index=200,
        partition=DistillationPartition.COUNTEREXAMPLE,
        features=_features(provider_health="degraded"),
        action=MetaAction.RUN_SELECTED_TEST,
        terminal_status=TerminalStatus.FAILED,
        accepted_criterion_ids=(),
        validation_receipt_ids=(),
        failure_signature="provider-degraded",
    )
    # Development examples share provider_health=healthy so CEGIS can add it.
    development = [
        _example(index=index, features=_features(provider_health="healthy"))
        for index in range(3)
    ]
    held = _example(
        index=100,
        partition=DistillationPartition.HELD_OUT,
        features=_features(provider_health="healthy"),
    )
    result = distiller.distill(
        (*development, held, counter),
        proposed_rule={
            "when": {
                "risk_class": "R1_READ_ONLY",
                "context_confidence": "current",
                "task_class": "cache_reuse",
                "language": "python",
                "repository_family": "ipfs_accelerate_py",
            },
            "action": MetaAction.READ_CACHED_RECEIPT.value,
        },
    )
    assert result.disposition is DistillationDisposition.SHADOW_CANDIDATE
    assert result.rule is not None
    assert result.rule.when["provider_health"] == "healthy"
    assert "counterexample_narrowed" in result.gate_receipts[1].reason_codes
    application = distiller.apply(result.rule, counter.features)
    assert application.used_fallback is True
    assert application.selected_action is result.rule.fallback

    colliding = _example(
        index=201,
        partition=DistillationPartition.COUNTEREXAMPLE,
        features=_features(),
        action=MetaAction.RUN_FULL_VALIDATION,
        terminal_status=TerminalStatus.FAILED,
        accepted_criterion_ids=(),
        validation_receipt_ids=(),
        failure_signature="same-features-other-action",
    )
    rejected = distiller.distill((*_stable_examples(), colliding))
    assert rejected.disposition is DistillationDisposition.REJECTED
    assert "counterexample_not_distinguishable" in rejected.reason_codes


def test_held_out_failure_does_not_emit_shadow_candidate() -> None:
    distiller = PolicyDistiller()
    examples = _stable_examples()
    examples[-1] = _example(
        index=100,
        partition=DistillationPartition.HELD_OUT,
        action=MetaAction.RUN_SELECTED_TEST,
    )
    result = distiller.distill(examples)
    assert result.disposition is DistillationDisposition.HELD_OUT_FAILED
    assert "held_out_failed" in result.reason_codes
    assert result.candidate is not None
    assert result.candidate.status is DistillationStatus.HELD_OUT_FAILED
    assert result.rule is not None
    assert result.rule.shadow_only is True


def test_missing_held_out_and_validation_are_insufficient() -> None:
    distiller = PolicyDistiller()
    development_only = [_example(index=index) for index in range(3)]
    result = distiller.distill(development_only)
    assert result.disposition is DistillationDisposition.INSUFFICIENT_EVIDENCE
    assert "missing_held_out_partition" in result.reason_codes

    unvalidated = _stable_examples()
    unvalidated[0] = _example(
        index=0,
        accepted_criterion_ids=(),
        validation_receipt_ids=(),
        terminal_status=TerminalStatus.FAILED,
        failure_signature="unvalidated",
    )
    result = distiller.distill(unvalidated)
    assert result.disposition is DistillationDisposition.INSUFFICIENT_EVIDENCE
    assert "missing_validation_receipt" in result.reason_codes


def test_aae_mutations_are_out_of_domain_and_use_fallback() -> None:
    distiller = PolicyDistiller()
    result = distiller.distill(_stable_examples())
    assert result.rule is not None
    mutants = generate_aae_mutations(result.rule.when)
    assert mutants
    for mutant in mutants:
        application = distiller.apply(result.rule, mutant.features)
        assert application.in_domain is False
        assert application.used_fallback is True
        assert application.selected_action is result.rule.fallback
        assert application.selected_action is not result.rule.action
        assert "out_of_domain" in application.reason_codes
    aae_receipt = next(item for item in result.gate_receipts if item.gate is DistillationGate.AAE)
    assert aae_receipt.passed is True

    colliding = AdversarialMutation(
        mutation_id="aae-same-as-evidence",
        features=_features(),
        expected_in_domain=False,
    )
    failed = distiller.distill(_stable_examples(), aae_mutations=(colliding,))
    assert failed.disposition is DistillationDisposition.REJECTED
    assert "aae_gate_failed" in failed.reason_codes


def test_out_of_domain_features_keep_fallback_and_never_self_promote() -> None:
    distiller = PolicyDistiller()
    result = distiller.distill(_stable_examples())
    assert result.rule is not None
    in_domain = distiller.apply(result.rule, _features())
    assert in_domain.in_domain is True
    assert in_domain.used_fallback is False
    assert in_domain.selected_action is result.rule.action
    ood = distiller.apply(result.rule, _features(task_class="merge_conflict"))
    assert ood.used_fallback is True
    assert ood.selected_action is result.rule.fallback
    missing = distiller.apply(result.rule, {"risk_class": "R1_READ_ONLY"})
    assert missing.used_fallback is True
    promotion = distiller.promote(result.rule, authorization_id=result.rule.rule_id)
    assert promotion.disposition is DistillationDisposition.REJECTED
    assert "cannot_self_promote" in promotion.reason_codes
    assert promotion.rule is None
    assert result.candidate is not None
    assert result.candidate.status is not DistillationStatus.PROMOTED
    assert all(item.status is not DistillationStatus.PROMOTED for item in (result.candidate,))


def test_proposed_self_authorizing_rule_is_rejected() -> None:
    distiller = PolicyDistiller()
    shadow = distiller.distill(_stable_examples())
    assert shadow.rule is not None
    promoted = DistilledDecisionRule(
        version="shadow-v1",
        when=dict(shadow.rule.when),
        action=shadow.rule.action,
        required_validation_ids=shadow.rule.required_validation_ids,
        fallback=shadow.rule.fallback,
        scope=dict(shadow.rule.scope),
        source_episode_ids=shadow.rule.source_episode_ids,
        held_out_evaluation_ids=shadow.rule.held_out_evaluation_ids,
        shadow_only=False,
        authorized_promotion_id="self-signed",
    )
    result = distiller.distill(_stable_examples(), proposed_rule=promoted)
    assert result.disposition is DistillationDisposition.REJECTED
    assert "cannot_self_promote" in result.reason_codes


def test_rollback_retains_history_and_cannot_reactivate() -> None:
    distiller = PolicyDistiller()
    shadow = distiller.distill(_stable_examples())
    assert shadow.rule is not None
    rolled = distiller.rollback(shadow.rule, reason_codes=("aae_regression",))
    assert rolled.disposition is DistillationDisposition.ROLLED_BACK
    assert rolled.candidate is not None
    assert rolled.candidate.status is DistillationStatus.ROLLED_BACK
    assert rolled.candidate.proposed_rule_id == shadow.rule.rule_id
    assert rolled.rolled_back_rule_id == shadow.rule.rule_id
    assert rolled.rule is not None
    assert rolled.rule.rule_id == shadow.rule.rule_id
    assert "history_retained" in rolled.reason_codes
    failed = distiller.distill(
        _stable_examples(development=2),
        current_shadow_rule=shadow.rule,
    )
    assert failed.disposition is DistillationDisposition.ROLLED_BACK
    assert failed.rolled_back_rule_id == shadow.rule.rule_id


def test_forbidden_fields_and_attribution_coverage_are_rejected() -> None:
    assert field_is_forbidden("python_code")
    assert field_is_forbidden("executable_code")
    assert field_is_forbidden("source_body")
    episode = _episode()
    attribution = _attribution(episode)
    with pytest.raises(PolicyDistillationError, match="forbidden|executable"):
        DistillationExample(
            example_id="bad",
            decision_class="cache_reuse",
            features={"risk_class": "R1_READ_ONLY", "python_code": "import os"},
            action=MetaAction.READ_CACHED_RECEIPT,
            episode=episode,
            attribution=attribution,
            partition=DistillationPartition.DEVELOPMENT,
        )
    other = _episode(frozen_input_ids=("tree-other", "objective-rev-1"))
    with pytest.raises(PolicyDistillationError, match="attribution"):
        DistillationExample(
            example_id="uncovered",
            decision_class="cache_reuse",
            features=_features(),
            action=MetaAction.READ_CACHED_RECEIPT,
            episode=other,
            attribution=attribution,
            partition=DistillationPartition.DEVELOPMENT,
        )


def test_results_and_rules_are_frozen_and_shadow_only() -> None:
    distiller = PolicyDistiller()
    result = distiller.distill(_stable_examples())
    assert result.rule is not None
    with pytest.raises(FrozenInstanceError):
        result.rule.shadow_only = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.disposition = DistillationDisposition.REJECTED  # type: ignore[misc]
    encoded = result.to_dict()
    assert encoded["shadow_only"] is True
    assert encoded["self_promoted"] is False
    assert result.result_id == result.to_dict()["result_id"]
    assert result.candidate is not None
    assert result.candidate.proposed_rule_id == result.rule.rule_id
    gates = {item.gate for item in result.gate_receipts}
    assert DistillationGate.DEVELOPMENT in gates
    assert DistillationGate.COUNTEREXAMPLE in gates
    assert DistillationGate.HELD_OUT in gates
    assert DistillationGate.AAE in gates
    assert DistillationGate.SHADOW in gates
    assert DistillationGate.PROMOTION_FORBIDDEN in gates
