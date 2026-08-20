from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.causal_attribution import (
    CAUSAL_ATTRIBUTION_ENGINE_INTERFACE,
    CAUSAL_ATTRIBUTION_INTERFACE,
    AblationFactor,
    AttributionDisposition,
    AttributionEvidenceKind,
    AttributionObservation,
    CausalAttributionEngine,
    CausalAttributionError,
    CausalAttributionResult,
    ControlledAblationProposal,
    ablation_may_affect_production_acceptance,
    confounder_pair,
    field_is_forbidden,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AttributionCause,
    CausalAttribution,
    ExperienceEpisode,
    MetaAction,
    TerminalStatus,
)


def _episode(**overrides: Any) -> ExperienceEpisode:
    values: dict[str, Any] = {
        "frozen_input_ids": ("tree-1", "objective-rev-1"),
        "question_feature_ids": ("feature-cache",),
        "selected_action": MetaAction.CALL_REMOTE_STANDARD_MODEL,
        "selection_policy_id": "policy-1",
        "selection_policy_version": "policy-rev-1",
        "terminal_status": TerminalStatus.FAILED,
        "context_metrics": {"input_tokens": 12, "prefix_reused_tokens": 0},
        "evidence_ids": ("evidence-model-1",),
        "accepted_criterion_ids": (),
        "validation_receipt_ids": (),
        "latency_ms": 40,
        "cost_micros": 0,
        "provider_id": "provider-remote",
        "model_id": "model-standard",
        "failure_signature": "failed-task",
    }
    values.update(overrides)
    return ExperienceEpisode(**values)


def _observation(**overrides: Any) -> AttributionObservation:
    values: dict[str, Any] = {
        "observation_id": "obs-1",
        "kind": AttributionEvidenceKind.COMPLETENESS_WITNESS,
        "evidence_ids": ("witness-1",),
        "episode_ids": (),
        "omitted_reference_ids": ("source-ref-missing",),
    }
    values.update(overrides)
    return AttributionObservation(**values)


def _paired(
    *,
    factor: AblationFactor,
    baseline: ExperienceEpisode,
    contrast: ExperienceEpisode,
    observation_id: str = "obs-paired",
    evidence_id: str = "ablation-receipt-1",
) -> AttributionObservation:
    return AttributionObservation(
        observation_id=observation_id,
        kind=AttributionEvidenceKind.PAIRED_COMPARISON,
        evidence_ids=(evidence_id,),
        episode_ids=(baseline.episode_id,),
        contrast_episode_ids=(contrast.episode_id,),
        factor=factor,
        baseline_terminal_status=baseline.terminal_status,
        contrast_terminal_status=contrast.terminal_status,
        shadow_only=True,
        production_acceptance=False,
    )


def test_interfaces_are_versioned_and_causes_are_closed() -> None:
    engine = CausalAttributionEngine()
    assert CAUSAL_ATTRIBUTION_ENGINE_INTERFACE == "CausalAttributionEngine@1"
    assert CAUSAL_ATTRIBUTION_INTERFACE == "CausalAttribution@1"
    assert engine.INTERFACE == CAUSAL_ATTRIBUTION_ENGINE_INTERFACE
    assert engine.ATTRIBUTION_INTERFACE == CAUSAL_ATTRIBUTION_INTERFACE
    assert {item.value for item in AttributionCause} == {
        "context_omission",
        "model_capability_failure",
        "provider_failure",
        "bad_task_decomposition",
        "bad_plan_branch",
        "stale_evidence",
        "incorrect_cache_reuse",
        "validation_selection_failure",
        "proof_selection_failure",
        "merge_conflict",
        "environment_failure",
        "human_policy_blocker",
    }
    assert confounder_pair(
        AttributionCause.CONTEXT_OMISSION, AttributionCause.MODEL_CAPABILITY_FAILURE
    )
    assert not confounder_pair(
        AttributionCause.CONTEXT_OMISSION, AttributionCause.MERGE_CONFLICT
    )


def test_completeness_witness_assigns_context_omission_not_model() -> None:
    episode = _episode()
    result = CausalAttributionEngine().attribute(episode, (_observation(episode_ids=(episode.episode_id,)),))
    assert result.disposition is AttributionDisposition.ATTRIBUTED
    assert result.primary_cause is AttributionCause.CONTEXT_OMISSION
    assert result.attribution is not None
    assert result.attribution.shadow_only is True
    assert "witness-1" in result.attribution.evidence_ids
    assert "model_not_blamed_for_omitted_source" in result.reason_codes
    assert result.affects_production_acceptance is False
    assert result.proposed_ablations == ()


def test_claimed_model_failure_is_rejected_when_source_was_omitted() -> None:
    episode = _episode()
    result = CausalAttributionEngine().attribute(
        episode,
        (_observation(episode_ids=(episode.episode_id,)),),
        claimed_cause=AttributionCause.MODEL_CAPABILITY_FAILURE,
    )
    assert result.primary_cause is AttributionCause.CONTEXT_OMISSION
    assert result.attribution is not None
    assert result.attribution.primary_cause is not AttributionCause.MODEL_CAPABILITY_FAILURE


def test_model_failure_requires_complete_context_evidence() -> None:
    episode = _episode()
    engine = CausalAttributionEngine()
    guessed = engine.attribute(episode)
    assert guessed.disposition is AttributionDisposition.ABLATION_REQUIRED
    assert guessed.attribution is None
    assert guessed.primary_cause is None
    assert any(
        item.hypothesized_cause is AttributionCause.CONTEXT_OMISSION
        for item in guessed.proposed_ablations
    )

    sufficient = _observation(
        observation_id="obs-sufficient",
        kind=AttributionEvidenceKind.CONTEXT_SUFFICIENT,
        evidence_ids=("completeness-ok",),
        omitted_reference_ids=(),
        episode_ids=(episode.episode_id,),
    )
    attributed = engine.attribute(episode, (sufficient,))
    assert attributed.disposition is AttributionDisposition.ATTRIBUTED
    assert attributed.primary_cause is AttributionCause.MODEL_CAPABILITY_FAILURE
    assert "complete_context_model_failure" in attributed.reason_codes


def test_paired_complete_context_still_failing_isolates_model() -> None:
    baseline = _episode()
    contrast = _episode(
        frozen_input_ids=("tree-1", "objective-rev-1", "complete-context"),
        evidence_ids=("evidence-model-complete",),
        terminal_status=TerminalStatus.FAILED,
    )
    sufficient = _observation(
        observation_id="obs-sufficient",
        kind=AttributionEvidenceKind.CONTEXT_SUFFICIENT,
        evidence_ids=("completeness-ok",),
        omitted_reference_ids=(),
        episode_ids=(baseline.episode_id, contrast.episode_id),
    )
    paired = _paired(
        factor=AblationFactor.CONTEXT_COMPLETENESS,
        baseline=baseline,
        contrast=contrast,
    )
    result = CausalAttributionEngine().attribute((baseline, contrast), (sufficient, paired))
    assert result.primary_cause is AttributionCause.MODEL_CAPABILITY_FAILURE
    assert result.attribution is not None
    assert result.attribution.controlled_ablation_ids
    assert result.attribution.shadow_only is True


def test_omitted_reference_metric_is_not_discriminating() -> None:
    episode = _episode(context_metrics={"input_tokens": 12, "omitted_reference_count": 1})
    result = CausalAttributionEngine().attribute(episode)
    assert result.disposition is AttributionDisposition.ABLATION_REQUIRED
    assert result.attribution is None
    assert AttributionCause.CONTEXT_OMISSION in result.competing_causes
    assert AttributionCause.MODEL_CAPABILITY_FAILURE not in result.competing_causes


def test_provider_failure_is_isolated_from_environment() -> None:
    episode = _episode(
        selected_action=MetaAction.CALL_REMOTE_STANDARD_MODEL,
        terminal_status=TerminalStatus.UNAVAILABLE,
        evidence_ids=("evidence-provider-1",),
    )
    observations = (
        _observation(
            observation_id="obs-provider",
            kind=AttributionEvidenceKind.PROVIDER_UNAVAILABLE,
            evidence_ids=("provider-down",),
            omitted_reference_ids=(),
            episode_ids=(episode.episode_id,),
        ),
        _observation(
            observation_id="obs-env-ok",
            kind=AttributionEvidenceKind.ENVIRONMENT_PROBE_SUCCESS,
            evidence_ids=("env-ok",),
            omitted_reference_ids=(),
            episode_ids=(episode.episode_id,),
        ),
    )
    result = CausalAttributionEngine().attribute(episode, observations)
    assert result.primary_cause is AttributionCause.PROVIDER_FAILURE
    assert "provider_unavailable_isolated" in result.reason_codes


def test_environment_failure_is_isolated_from_provider() -> None:
    episode = _episode(
        selected_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        provider_id="",
        model_id="",
        evidence_ids=("evidence-env-1",),
        terminal_status=TerminalStatus.FAILED,
    )
    result = CausalAttributionEngine().attribute(
        episode,
        (
            _observation(
                observation_id="obs-env",
                kind=AttributionEvidenceKind.ENVIRONMENT_PROBE_FAILURE,
                evidence_ids=("env-down",),
                omitted_reference_ids=(),
                episode_ids=(episode.episode_id,),
            ),
        ),
    )
    assert result.primary_cause is AttributionCause.ENVIRONMENT_FAILURE


def test_provider_and_environment_without_isolation_abstain() -> None:
    episode = _episode(terminal_status=TerminalStatus.UNAVAILABLE)
    result = CausalAttributionEngine().attribute(
        episode,
        (
            _observation(
                observation_id="obs-provider",
                kind=AttributionEvidenceKind.PROVIDER_UNAVAILABLE,
                evidence_ids=("provider-down",),
                omitted_reference_ids=(),
                episode_ids=(episode.episode_id,),
            ),
            _observation(
                observation_id="obs-env",
                kind=AttributionEvidenceKind.ENVIRONMENT_PROBE_FAILURE,
                evidence_ids=("env-down",),
                omitted_reference_ids=(),
                episode_ids=(episode.episode_id,),
            ),
        ),
    )
    assert result.disposition is AttributionDisposition.CONFOUNDER_PRESENT
    assert result.attribution is None
    assert set(result.competing_causes) >= {
        AttributionCause.PROVIDER_FAILURE,
        AttributionCause.ENVIRONMENT_FAILURE,
    }


def test_stale_evidence_versus_incorrect_cache_reuse() -> None:
    stale_episode = _episode(
        selected_action=MetaAction.RUN_SELECTED_TEST,
        provider_id="",
        model_id="",
        evidence_ids=("evidence-stale",),
        validation_receipt_ids=("validation-stale",),
    )
    stale = CausalAttributionEngine().attribute(
        stale_episode,
        (
            _observation(
                observation_id="obs-stale",
                kind=AttributionEvidenceKind.STALE_IDENTITY,
                evidence_ids=("stale-tree",),
                omitted_reference_ids=(),
                episode_ids=(stale_episode.episode_id,),
            ),
        ),
    )
    assert stale.primary_cause is AttributionCause.STALE_EVIDENCE

    cache_episode = _episode(
        selected_action=MetaAction.READ_CACHED_RECEIPT,
        provider_id="",
        model_id="",
        evidence_ids=("evidence-cache",),
    )
    cache = CausalAttributionEngine().attribute(
        cache_episode,
        (
            _observation(
                observation_id="obs-cache",
                kind=AttributionEvidenceKind.CACHE_BINDING_MISMATCH,
                evidence_ids=("cache-mismatch",),
                omitted_reference_ids=(),
                episode_ids=(cache_episode.episode_id,),
            ),
        ),
    )
    assert cache.primary_cause is AttributionCause.INCORRECT_CACHE_REUSE

    mixed = CausalAttributionEngine().attribute(
        cache_episode,
        (
            _observation(
                observation_id="obs-stale",
                kind=AttributionEvidenceKind.STALE_IDENTITY,
                evidence_ids=("stale-tree",),
                omitted_reference_ids=(),
                episode_ids=(cache_episode.episode_id,),
            ),
            _observation(
                observation_id="obs-cache",
                kind=AttributionEvidenceKind.CACHE_BINDING_MISMATCH,
                evidence_ids=("cache-mismatch",),
                omitted_reference_ids=(),
                episode_ids=(cache_episode.episode_id,),
            ),
        ),
    )
    assert mixed.disposition is AttributionDisposition.CONFOUNDER_PRESENT
    assert mixed.attribution is None


def test_plan_branch_versus_task_decomposition() -> None:
    plan_episode = _episode(
        selected_action=MetaAction.REPLAN_AFFECTED_SUFFIX,
        provider_id="",
        model_id="",
        evidence_ids=("evidence-plan",),
    )
    plan = CausalAttributionEngine().attribute(
        plan_episode,
        (
            _observation(
                observation_id="obs-plan",
                kind=AttributionEvidenceKind.PLAN_BRANCH_FAILURE,
                evidence_ids=("branch-fail",),
                omitted_reference_ids=(),
                episode_ids=(plan_episode.episode_id,),
            ),
        ),
    )
    assert plan.primary_cause is AttributionCause.BAD_PLAN_BRANCH

    decomp_episode = _episode(
        selected_action=MetaAction.GENERATE_BOUNDED_REPAIR,
        provider_id="",
        model_id="",
        evidence_ids=("evidence-decomp",),
    )
    decomp = CausalAttributionEngine().attribute(
        decomp_episode,
        (
            _observation(
                observation_id="obs-decomp",
                kind=AttributionEvidenceKind.DECOMPOSITION_FAILURE,
                evidence_ids=("decomp-fail",),
                omitted_reference_ids=(),
            ),
        ),
    )
    assert decomp.primary_cause is AttributionCause.BAD_TASK_DECOMPOSITION

    mixed = CausalAttributionEngine().attribute(
        plan_episode,
        (
            _observation(
                observation_id="obs-plan",
                kind=AttributionEvidenceKind.PLAN_BRANCH_FAILURE,
                evidence_ids=("branch-fail",),
                omitted_reference_ids=(),
            ),
            _observation(
                observation_id="obs-decomp",
                kind=AttributionEvidenceKind.DECOMPOSITION_FAILURE,
                evidence_ids=("decomp-fail",),
                omitted_reference_ids=(),
            ),
        ),
    )
    assert mixed.disposition is AttributionDisposition.CONFOUNDER_PRESENT


def test_validation_proof_merge_and_human_policy_cases() -> None:
    engine = CausalAttributionEngine()
    validation_episode = _episode(
        selected_action=MetaAction.RUN_SELECTED_TEST,
        provider_id="",
        model_id="",
        evidence_ids=("evidence-val",),
        validation_receipt_ids=("validation-wrong-selector",),
    )
    validation = engine.attribute(
        validation_episode,
        (
            _observation(
                observation_id="obs-val",
                kind=AttributionEvidenceKind.VALIDATION_SELECTOR_MISMATCH,
                evidence_ids=("selector-mismatch",),
                omitted_reference_ids=(),
            ),
        ),
    )
    assert validation.primary_cause is AttributionCause.VALIDATION_SELECTION_FAILURE

    proof_episode = _episode(
        selected_action=MetaAction.RUN_SMT_OR_PROVER,
        provider_id="",
        model_id="",
        evidence_ids=("evidence-proof",),
        proof_receipt_ids=("proof-wrong-obligation",),
    )
    proof = engine.attribute(
        proof_episode,
        (
            _observation(
                observation_id="obs-proof",
                kind=AttributionEvidenceKind.PROOF_SELECTOR_MISMATCH,
                evidence_ids=("proof-selector-mismatch",),
                omitted_reference_ids=(),
            ),
        ),
    )
    assert proof.primary_cause is AttributionCause.PROOF_SELECTION_FAILURE

    merge_episode = _episode(
        selected_action=MetaAction.QUARANTINE_TASK,
        provider_id="",
        model_id="",
        evidence_ids=("evidence-merge",),
        merge_receipt_ids=("merge-conflict-1",),
        terminal_status=TerminalStatus.BLOCKED,
    )
    merge = engine.attribute(
        merge_episode,
        (
            _observation(
                observation_id="obs-merge",
                kind=AttributionEvidenceKind.MERGE_CONFLICT,
                evidence_ids=("merge-conflict-receipt",),
                omitted_reference_ids=(),
            ),
        ),
    )
    assert merge.primary_cause is AttributionCause.MERGE_CONFLICT

    policy_episode = _episode(
        selected_action=MetaAction.REQUEST_HUMAN_DECISION,
        provider_id="",
        model_id="",
        evidence_ids=("evidence-policy",),
        human_intervention_ids=("human-1",),
        terminal_status=TerminalStatus.BLOCKED,
    )
    policy = engine.attribute(
        policy_episode,
        (
            _observation(
                observation_id="obs-policy",
                kind=AttributionEvidenceKind.HUMAN_POLICY_BLOCK,
                evidence_ids=("policy-block",),
                omitted_reference_ids=(),
            ),
        ),
    )
    assert policy.primary_cause is AttributionCause.HUMAN_POLICY_BLOCKER


def test_compression_is_not_credited_from_one_pass() -> None:
    episode = _episode(
        terminal_status=TerminalStatus.SUCCEEDED,
        accepted_criterion_ids=("AC-1",),
        validation_receipt_ids=("validation-1",),
        context_metrics={"input_tokens": 8, "prefix_reused_tokens": 6, "compressed": 1},
        failure_signature="",
    )
    credit = CausalAttributionEngine().compression_credit(episode)
    assert credit.credited is False
    assert "single_pass_insufficient" in credit.reason_codes
    assert credit.proposed_ablations
    assert all(item.shadow_only for item in credit.proposed_ablations)
    assert all(not item.affects_production_acceptance for item in credit.proposed_ablations)
    assert ablation_may_affect_production_acceptance(credit.proposed_ablations[0]) is False

    attribution = CausalAttributionEngine().attribute(episode)
    assert attribution.attribution is None
    assert attribution.disposition is AttributionDisposition.INSUFFICIENT_EVIDENCE


def test_paired_equivalent_runs_may_credit_compression() -> None:
    compressed = _episode(
        frozen_input_ids=("tree-1", "compressed"),
        terminal_status=TerminalStatus.SUCCEEDED,
        accepted_criterion_ids=("AC-1",),
        validation_receipt_ids=("validation-compressed",),
        context_metrics={"input_tokens": 8, "prefix_reused_tokens": 6, "compressed": 1},
        failure_signature="",
    )
    uncompressed = _episode(
        frozen_input_ids=("tree-1", "uncompressed"),
        terminal_status=TerminalStatus.SUCCEEDED,
        accepted_criterion_ids=("AC-1",),
        validation_receipt_ids=("validation-uncompressed",),
        context_metrics={"input_tokens": 20, "prefix_reused_tokens": 0, "compressed": 0},
        evidence_ids=("evidence-uncompressed",),
        failure_signature="",
    )
    credit = CausalAttributionEngine().compression_credit(
        (compressed, uncompressed),
        (_paired(factor=AblationFactor.COMPRESSION, baseline=compressed, contrast=uncompressed),),
    )
    assert credit.credited is True
    assert "paired_compression_equivalent" in credit.reason_codes
    assert credit.proposed_ablations == ()


def test_both_failed_compression_pair_cannot_credit_or_blame_compression() -> None:
    compressed = _episode(
        frozen_input_ids=("tree-1", "compressed"),
        context_metrics={"input_tokens": 8, "compressed": 1},
        evidence_ids=("evidence-compressed",),
    )
    uncompressed = _episode(
        frozen_input_ids=("tree-1", "uncompressed"),
        context_metrics={"input_tokens": 20, "compressed": 0},
        evidence_ids=("evidence-uncompressed",),
    )
    paired = _paired(factor=AblationFactor.COMPRESSION, baseline=compressed, contrast=uncompressed)
    credit = CausalAttributionEngine().compression_credit((compressed, uncompressed), (paired,))
    assert credit.credited is False
    assert "both_failed_cannot_blame_compression" in credit.reason_codes
    result = CausalAttributionEngine().attribute((compressed, uncompressed), (paired,))
    assert result.primary_cause is not AttributionCause.CONTEXT_OMISSION or result.attribution is None
    assert result.disposition in {
        AttributionDisposition.INSUFFICIENT_EVIDENCE,
        AttributionDisposition.ABLATION_REQUIRED,
    }


def test_expanded_success_after_compressed_failure_is_omission_not_credit() -> None:
    compressed = _episode(
        frozen_input_ids=("tree-1", "compressed"),
        context_metrics={"input_tokens": 8, "compressed": 1},
        evidence_ids=("evidence-compressed",),
    )
    expanded = _episode(
        frozen_input_ids=("tree-1", "expanded"),
        terminal_status=TerminalStatus.SUCCEEDED,
        accepted_criterion_ids=("AC-1",),
        validation_receipt_ids=("validation-expanded",),
        context_metrics={"input_tokens": 20, "compressed": 0},
        evidence_ids=("evidence-expanded",),
        failure_signature="",
    )
    paired = _paired(factor=AblationFactor.COMPRESSION, baseline=compressed, contrast=expanded)
    credit = CausalAttributionEngine().compression_credit((compressed, expanded), (paired,))
    assert credit.credited is False
    result = CausalAttributionEngine().attribute((compressed, expanded), (paired,))
    assert result.primary_cause is AttributionCause.CONTEXT_OMISSION
    assert result.attribution is not None
    assert result.attribution.shadow_only is True


def test_ablations_cannot_affect_production_acceptance() -> None:
    episode = _episode()
    proposals = CausalAttributionEngine().propose_ablations(episode)
    assert proposals
    for proposal in proposals:
        assert proposal.shadow_only is True
        assert proposal.affects_production_acceptance is False
        assert ablation_may_affect_production_acceptance(proposal) is False
        replayed = ControlledAblationProposal.from_dict(proposal.to_dict())
        assert replayed.proposal_id == proposal.proposal_id
        assert replayed.shadow_only is True

    with pytest.raises(CausalAttributionError, match="production acceptance"):
        AttributionObservation(
            observation_id="obs-bad",
            kind=AttributionEvidenceKind.PAIRED_COMPARISON,
            evidence_ids=("ablation-1",),
            episode_ids=("episode-a", "episode-b"),
            factor=AblationFactor.COMPRESSION,
            baseline_terminal_status=TerminalStatus.FAILED,
            contrast_terminal_status=TerminalStatus.SUCCEEDED,
            shadow_only=True,
            production_acceptance=True,
        )
    with pytest.raises(CausalAttributionError, match="shadow-only"):
        AttributionObservation(
            observation_id="obs-live",
            kind=AttributionEvidenceKind.PAIRED_COMPARISON,
            evidence_ids=("ablation-1",),
            episode_ids=("episode-a", "episode-b"),
            factor=AblationFactor.COMPRESSION,
            baseline_terminal_status=TerminalStatus.FAILED,
            contrast_terminal_status=TerminalStatus.SUCCEEDED,
            shadow_only=False,
        )
    with pytest.raises(CausalAttributionError, match="production acceptance"):
        ControlledAblationProposal(
            factor=AblationFactor.COMPRESSION,
            hypothesized_cause=AttributionCause.CONTEXT_OMISSION,
            baseline_episode_ids=(episode.episode_id,),
            contrast_action=MetaAction.EXPAND_CONTEXT_REFERENCE,
            expected_evidence_ids=(),
            reason_codes=("single_pass_insufficient",),
            shadow_only=False,
        )


def test_correlation_and_empty_evidence_do_not_assign_a_cause() -> None:
    episode = _episode(
        selected_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        provider_id="",
        model_id="",
        evidence_ids=("evidence-static",),
    )
    result = CausalAttributionEngine().attribute(episode)
    assert result.disposition is AttributionDisposition.INSUFFICIENT_EVIDENCE
    assert result.attribution is None
    assert result.primary_cause is None


def test_forbidden_fields_and_floats_fail_closed() -> None:
    assert field_is_forbidden("raw_prompt")
    assert field_is_forbidden("source_body")
    engine = CausalAttributionEngine()
    with pytest.raises(CausalAttributionError, match="forbidden"):
        engine.attribute(_episode(), ({"observation_id": "x", "kind": "completeness_witness", "evidence_ids": ("e",), "prompt": "secret"},))
    with pytest.raises(CausalAttributionError, match="floats"):
        payload = dict(_episode().to_dict())
        payload["context_metrics"] = {"input_tokens": 1.5}
        engine.attribute(payload)
    with pytest.raises(CausalAttributionError, match="unsupported fields"):
        AttributionObservation.from_dict(
            {
                "observation_id": "obs-extra",
                "kind": "completeness_witness",
                "evidence_ids": ("witness-1",),
                "omitted_reference_ids": ("source-ref-missing",),
                "unexpected_claim": "authoritative",
            }
        )
    with pytest.raises(CausalAttributionError, match="must not be empty"):
        engine.attribute(())


def test_canonical_result_is_idempotent_and_immutable() -> None:
    episode = _episode()
    observation = _observation(episode_ids=(episode.episode_id,))
    engine = CausalAttributionEngine()
    first = engine.attribute(episode, (observation,))
    second = engine.attribute(episode, (observation.to_dict(),))
    assert first.result_id == second.result_id
    assert first.to_dict() == second.to_dict()
    assert first.attribution is not None
    replayed = CausalAttribution.from_dict(first.attribution.to_dict())
    assert replayed.attribution_id == first.attribution.attribution_id
    with pytest.raises(FrozenInstanceError):
        first.affects_production_acceptance = True  # type: ignore[misc]


def test_result_rejects_cause_without_attribution_contract() -> None:
    episode = _episode()
    with pytest.raises(CausalAttributionError, match="discriminating evidence"):
        CausalAttributionResult(
            disposition=AttributionDisposition.INSUFFICIENT_EVIDENCE,
            reason_codes=("insufficient_discriminating_evidence",),
            episode_ids=(episode.episode_id,),
            attribution=CausalAttribution(
                episode_ids=(episode.episode_id,),
                primary_cause=AttributionCause.MODEL_CAPABILITY_FAILURE,
                evidence_ids=("guess",),
                confidence_bp=1_000,
            ),
        )
    with pytest.raises(CausalAttributionError, match="production acceptance"):
        CausalAttributionResult(
            disposition=AttributionDisposition.INSUFFICIENT_EVIDENCE,
            reason_codes=("insufficient_discriminating_evidence",),
            episode_ids=(episode.episode_id,),
            affects_production_acceptance=True,
        )
