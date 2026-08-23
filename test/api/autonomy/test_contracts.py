from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy import contracts as contracts_module
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
    AttributionCause,
    AuthorityClass,
    AutonomousRepairPlan,
    AutonomousRepairReceipt,
    AutonomyContractError,
    AutonomyEnvelope,
    AutonomyLevel,
    AutonomyPolicy,
    AutonomyPromotionReceipt,
    AutonomyRunReceipt,
    BeliefFact,
    BeliefState,
    BudgetLedger,
    BudgetPurpose,
    BudgetReservation,
    BudgetReservationStatus,
    CancellationBehavior,
    CausalAttribution,
    CognitiveBudget,
    DecisionGraph,
    DecisionQuestion,
    DecisionQuestionType,
    DistillationCandidate,
    DistilledDecisionRule,
    EvidenceFreshness,
    ExperienceEpisode,
    HumanEscalationPacket,
    MetaAction,
    MetaDecision,
    MetaDecisionDisposition,
    PolicyObservation,
    PrivacyClass,
    PromotionStatus,
    QuestionDisposition,
    RepairTier,
    ResolutionAction,
    ResolutionCandidate,
    ResolutionEvidenceKind,
    RiskAssessment,
    RiskClass,
    RoutePolicyCandidate,
    SupervisorSkill,
    TerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    CanonicalContract,
    content_identity,
)

SAFETY_GATES = {
    "false_completions": True,
    "unauthorized_mutations": True,
    "simulated_as_live": True,
    "stale_authoritative_cache_hits": True,
    "confirmation_replays": True,
    "path_or_scope_escapes": True,
    "hidden_validation_reductions": True,
    "escaped_critical_seeded_defects": True,
    "self_authorized_policy_promotions": True,
}


def _budget(**overrides: int) -> CognitiveBudget:
    values = {
        "max_total_model_calls": 4,
        "max_strong_model_calls": 1,
        "max_input_tokens": 10_000,
        "max_output_tokens": 2_000,
        "max_provider_spend_micros": 500_000,
        "max_proof_time_ms": 60_000,
        "max_validation_time_ms": 120_000,
        "max_human_questions": 1,
        "max_repair_rounds": 2,
        "max_plan_branches": 3,
        "max_context_expansions": 4,
        "max_wall_time_ms": 300_000,
        "validation_reserve_ms": 30_000,
        "proof_reserve_ms": 10_000,
    }
    values.update(overrides)
    return CognitiveBudget(**values)


def _risk(**overrides: object) -> RiskAssessment:
    values: dict[str, object] = {
        "risk_class": RiskClass.R2_REVERSIBLE_LOCAL,
        "reversible": True,
        "blast_radius_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "blast_radius_symbols": ("AutonomousMetaController",),
        "evidence_ids": ("evidence-risk",),
        "reason_codes": ("bounded_local_change",),
    }
    values.update(overrides)
    return RiskAssessment(**values)


def _question(**overrides: object) -> DecisionQuestion:
    values: dict[str, object] = {
        "objective_id": "APMC-G000",
        "acceptance_criterion_ids": ("AC-1",),
        "question_type": DecisionQuestionType.WHETHER_CACHE_IS_REUSABLE,
        "current_alternatives": ("reuse", "recompute"),
        "required_evidence_ids": ("required-tree-binding",),
        "known_evidence_ids": ("current-tree-receipt",),
        "contradictory_evidence_ids": (),
        "residual_uncertainty_bp": 0,
        "decision_deadline_ms": 1_000,
        "risk_if_incorrect": RiskClass.R2_REVERSIBLE_LOCAL,
        "risk_if_left_unresolved": RiskClass.R1_READ_ONLY,
        "possible_resolution_action_ids": ("action-static",),
        "dependency_question_ids": (),
        "terminal_decision_rule": "tree_identity_matches",
        "mandatory": True,
        "disposition": QuestionDisposition.RESOLVED,
        "terminal_answer": "reuse",
    }
    values.update(overrides)
    return DecisionQuestion(**values)


def _action(**overrides: object) -> ResolutionAction:
    values: dict[str, object] = {
        "action": MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        "precondition_ids": ("tree-current",),
        "expected_evidence_kind": ResolutionEvidenceKind.STATIC_ANALYSIS,
        "expected_uncertainty_reduction_bp": 8_000,
        "token_cost": 0,
        "latency_cost_ms": 100,
        "provider_cost_micros": 0,
        "resource_cost_units": 10,
        "invalidation_cost_units": 1,
        "privacy_cost_units": 0,
        "privacy_class": PrivacyClass.LOCAL_ONLY,
        "risk_class": RiskClass.R1_READ_ONLY,
        "cancellation_behavior": CancellationBehavior.COOPERATIVE,
        "cacheable": True,
        "authority_class": AuthorityClass.VERIFIED,
        "can_change_decision": True,
        "accepted_as_authority": True,
    }
    values.update(overrides)
    return ResolutionAction(**values)


def _samples() -> tuple[CanonicalContract, ...]:
    budget = _budget()
    risk = _risk()
    policy = AutonomyPolicy(
        policy_revision="policy-rev-1",
        authority_id="operator-policy-authority",
        human_escalation_policy_id="human-policy-1",
    )
    envelope = AutonomyEnvelope(
        repository_id="repo-1",
        tree_id="tree-1",
        objective_id="APMC-G000",
        objective_revision="objective-rev-1",
        task_id="APMC-001",
        acceptance_criterion_ids=("AC-1",),
        risk_assessment=risk,
        autonomy_level=AutonomyLevel.EXECUTE_REVERSIBLE,
        cognitive_budget=budget,
        allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
        allowed_symbols=("AutonomousMetaController",),
        required_test_ids=("test-contracts",),
        required_proof_ids=(),
        authority_id="operator-policy-authority",
        policy_id=policy.policy_id,
        provider_usage_envelope_id="provider-envelope-1",
        resource_budget_id="resource-budget-1",
        human_escalation_policy_id="human-policy-1",
        expiry_ms=10_000,
        reversible=True,
        blast_radius={"max_files": 3},
    )
    first_question = _question()
    second_question = _question(
        question_type=DecisionQuestionType.WHETHER_REPLAN_IS_REQUIRED,
        current_alternatives=("preserve", "replan_suffix"),
        required_evidence_ids=("failure-dependency-receipt",),
        known_evidence_ids=("failure-dependency-receipt",),
        possible_resolution_action_ids=("action-replan",),
        dependency_question_ids=(first_question.question_id,),
        terminal_decision_rule="affected_suffix_is_nonempty",
        terminal_answer="preserve",
    )
    graph = DecisionGraph(
        repository_id="repo-1",
        tree_id="tree-1",
        objective_id="APMC-G000",
        objective_revision="objective-rev-1",
        graph_revision=1,
        questions=(first_question, second_question),
        evidence_dependencies={first_question.question_id: ("tree-1",)},
    )
    fact = BeliefFact(
        subject_question_id=first_question.question_id,
        predicate="cache_binding_matches",
        value={"matches": True, "revision": 1},
        evidence_ids=("current-tree-receipt",),
        authority_class=AuthorityClass.AUTHORITATIVE,
        freshness=EvidenceFreshness.CURRENT,
        confidence_bp=10_000,
        observed_tree_id="tree-1",
    )
    belief = BeliefState(
        objective_id="APMC-G000",
        objective_revision="objective-rev-1",
        current_tree_id="tree-1",
        revision=1,
        facts=(fact,),
    )
    action = _action()
    candidate = ResolutionCandidate(
        question_id=first_question.question_id,
        resolution_action=action,
        expected_decision_value=900,
        admissible=True,
        policy_id=policy.policy_id,
        reason_codes=("software_first",),
        evidence_ids=("current-tree-receipt",),
    )
    reservation = BudgetReservation(
        budget_id=budget.budget_id,
        idempotency_key="reserve-static-analysis-1",
        question_id=first_question.question_id,
        action_id=action.action_id,
        purpose=BudgetPurpose.ANALYSIS,
        status=BudgetReservationStatus.RESERVED,
        max_wall_time_ms=1_000,
    )
    ledger = BudgetLedger(
        budget=budget,
        epoch=1,
        reservations=(reservation,),
    )
    decision = MetaDecision(
        question_id=first_question.question_id,
        selected_candidate_id=candidate.candidate_id,
        selected_action=action.action,
        considered_candidate_ids=(candidate.candidate_id,),
        rejected_candidate_ids=(),
        evidence_ids=("current-tree-receipt",),
        reservation_id=reservation.reservation_id,
        policy_id=policy.policy_id,
        disposition=MetaDecisionDisposition.SELECTED,
        reason_codes=("highest_admissible_utility",),
    )
    episode = ExperienceEpisode(
        frozen_input_ids=("tree-1", "objective-rev-1"),
        question_feature_ids=("feature-cache",),
        selected_action=action.action,
        selection_policy_id=policy.policy_id,
        selection_policy_version="policy-rev-1",
        terminal_status=TerminalStatus.SUCCEEDED,
        context_metrics={"input_tokens": 0, "prefix_reused_tokens": 0},
        evidence_ids=("current-tree-receipt",),
        accepted_criterion_ids=("AC-1",),
        validation_receipt_ids=("validation-1",),
        latency_ms=100,
    )
    attribution = CausalAttribution(
        episode_ids=(episode.episode_id,),
        primary_cause=AttributionCause.STALE_EVIDENCE,
        evidence_ids=("ablation-result-1",),
        confidence_bp=8_500,
        controlled_ablation_ids=("ablation-1",),
    )
    observation = PolicyObservation(
        episode_id=episode.episode_id,
        route_policy_id=policy.policy_id,
        selected_action=action.action,
        selection_reason_codes=("deterministic_authority",),
        feature_ids=("feature-cache",),
        terminal_status=TerminalStatus.SUCCEEDED,
        accepted_criterion_ids=("AC-1",),
        evidence_gain_bp=9_000,
    )
    route = RoutePolicyCandidate(
        parent_policy_id=policy.policy_id,
        policy_version="route-candidate-1",
        allowed_actions=(MetaAction.RUN_LOCAL_STATIC_ANALYSIS, MetaAction.RUN_SELECTED_TEST),
        feature_names=("risk_rank", "context_sufficient"),
        integer_weights={"risk_rank": -2, "context_sufficient": 5},
        training_observation_ids=(observation.observation_id,),
        held_out_evaluation_ids=("held-out-route-1",),
        safety_gate_receipt_ids=("safety-route-1",),
        selection_reason="linear_score_v1",
    )
    distillation = DistillationCandidate(
        decision_class="cache_reuse",
        episode_ids=(episode.episode_id,),
        input_feature_names=("risk_class", "context_confidence"),
        output_actions=(MetaAction.READ_CACHED_RECEIPT,),
        development_example_ids=("development-1",),
        held_out_example_ids=("held-out-1",),
        proposed_rule_id="rule-proposal-1",
    )
    rule = DistilledDecisionRule(
        version="rule-v1",
        when={"risk_class": "R1_READ_ONLY", "context_confidence": "current"},
        action=MetaAction.READ_CACHED_RECEIPT,
        required_validation_ids=("validate-tree-binding",),
        fallback=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        scope={"repository_family": "ipfs_accelerate_py"},
        source_episode_ids=(episode.episode_id,),
        held_out_evaluation_ids=("held-out-1",),
    )
    skill = SupervisorSkill(
        version="skill-v1",
        precondition_ids=("tree-current",),
        input_schema_id="schema-static-analysis",
        effect_class="read_only_analysis",
        steps=(MetaAction.RUN_LOCAL_STATIC_ANALYSIS, MetaAction.RUN_SELECTED_TEST),
        postcondition_ids=("question-resolved",),
        validation_ids=("validate-tree-binding",),
        rollback_action_ids=("release-reservation",),
        fallback=MetaAction.QUARANTINE_TASK,
        scope_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
        scope_symbols=("DecisionQuestion",),
        risk_class=RiskClass.R1_READ_ONLY,
    )
    escalation = HumanEscalationPacket(
        objective_id="APMC-G000",
        blocked_criterion_ids=("AC-operator-choice",),
        question="Which externally authorized release policy should apply?",
        options=("keep_shadow", "request_review"),
        recommended_option="keep_shadow",
        predicted_consequences={
            "keep_shadow": "No live routing change.",
            "request_review": "An authorized review is queued.",
        },
        cost_and_risk={"keep_shadow": "low", "request_review": "bounded"},
        evidence_ids=("held-out-route-1",),
        continuation_by_option={
            "keep_shadow": "record_non_promotion",
            "request_review": "await_authority",
        },
        expires_at_ms=10_000,
    )
    repair_plan = AutonomousRepairPlan(
        objective_id="APMC-G000",
        task_id="APMC-001",
        repair_tier=RepairTier.TEMPLATE_CONSTRAINED,
        predicted_files=("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
        predicted_symbols=("AutonomyEnvelope",),
        patch_envelope_id=envelope.envelope_id,
        context_reference_ids=("context-ref-1",),
        required_test_ids=("test-contracts",),
        required_proof_ids=(),
        worktree_id="worktree-1",
        allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
        forbidden_symbols=("trusted_keys",),
        rollback_plan_id="rollback-plan-1",
        risk_class=RiskClass.R2_REVERSIBLE_LOCAL,
        max_changed_files=1,
        max_changed_lines=100,
    )
    repair_receipt = AutonomousRepairReceipt(
        plan_id=repair_plan.plan_id,
        envelope_id=envelope.envelope_id,
        terminal_status=TerminalStatus.SUCCEEDED,
        changed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
        validation_receipt_ids=("validation-1",),
        proof_receipt_ids=(),
        adversarial_assurance_receipt_ids=("assurance-1",),
    )
    run = AutonomyRunReceipt(
        envelope_id=envelope.envelope_id,
        policy_id=policy.policy_id,
        graph_id=graph.graph_id,
        budget_ledger_id=ledger.ledger_id,
        terminal_status=TerminalStatus.SUCCEEDED,
        safety_gate_results=SAFETY_GATES,
        meta_decision_ids=(decision.decision_id,),
        action_receipt_ids=(repair_receipt.receipt_id,),
        accepted_criterion_ids=("AC-1",),
    )
    promotion = AutonomyPromotionReceipt(
        candidate_policy_id=route.candidate_id,
        expected_old_policy_id=policy.policy_id,
        resulting_policy_id="route-policy-live-1",
        status=PromotionStatus.PROMOTED,
        safety_gate_results=SAFETY_GATES,
        held_out_evaluation_ids=("held-out-route-1",),
        safety_gate_receipt_ids=("safety-route-1",),
        authorization_id="operator-authorization-1",
        compare_and_swap_receipt_id="cas-1",
        rollback_policy_id=policy.policy_id,
    )
    return (
        policy,
        risk,
        budget,
        envelope,
        first_question,
        graph,
        fact,
        belief,
        action,
        candidate,
        decision,
        reservation,
        ledger,
        episode,
        attribution,
        observation,
        route,
        distillation,
        rule,
        skill,
        escalation,
        repair_plan,
        repair_receipt,
        run,
        promotion,
    )


def test_all_required_named_contracts_are_exported() -> None:
    required = {
        "AutonomyPolicy",
        "AutonomyEnvelope",
        "AutonomyLevel",
        "RiskAssessment",
        "DecisionQuestion",
        "DecisionGraph",
        "BeliefFact",
        "BeliefState",
        "ResolutionAction",
        "ResolutionCandidate",
        "MetaDecision",
        "CognitiveBudget",
        "BudgetReservation",
        "BudgetLedger",
        "ExperienceEpisode",
        "CausalAttribution",
        "PolicyObservation",
        "RoutePolicyCandidate",
        "DistillationCandidate",
        "DistilledDecisionRule",
        "SupervisorSkill",
        "HumanEscalationPacket",
        "AutonomousRepairPlan",
        "AutonomousRepairReceipt",
        "AutonomyRunReceipt",
        "AutonomyPromotionReceipt",
    }
    assert required.issubset(set(contracts_module.__all__))


@pytest.mark.parametrize("contract", _samples(), ids=lambda item: type(item).__name__)
def test_contract_round_trip_identity_unknown_fields_and_immutability(
    contract: CanonicalContract,
) -> None:
    assert isinstance(contract, CanonicalContract)
    rebuilt = type(contract).from_dict(contract.to_dict())
    assert rebuilt.to_dict() == contract.to_dict()
    assert rebuilt.content_id == contract.content_id
    assert content_identity(contract.to_dict()) == contract.content_id
    assert type(contract).from_json(contract.to_json()).content_id == contract.content_id
    assert type(contract).from_dict(contract.to_record()).content_id == contract.content_id

    unknown = dict(contract.to_dict())
    unknown["unexpected_model_claim"] = "authoritative"
    with pytest.raises(AutonomyContractError, match="unsupported fields"):
        type(contract).from_dict(unknown)

    with pytest.raises(FrozenInstanceError):
        contract.schema = "changed"  # type: ignore[misc]


def test_closed_vocabulary_is_exact_and_has_no_unrestricted_autonomy() -> None:
    assert {item.value for item in AutonomyLevel} == {
        "observe_only",
        "recommend",
        "dry_run",
        "execute_reversible",
        "execute_bounded_mutation",
        "self_repair_isolated",
    }
    assert {item.value for item in RiskClass} == {
        "R0_PURE",
        "R1_READ_ONLY",
        "R2_REVERSIBLE_LOCAL",
        "R3_BOUNDED_REPOSITORY_MUTATION",
        "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL",
    }
    assert {item.value for item in MetaAction} == {
        "NO_OP",
        "READ_CACHED_RECEIPT",
        "RUN_LOCAL_STATIC_ANALYSIS",
        "RUN_INCREMENTAL_INDEX_QUERY",
        "RUN_GRAPH_RETRIEVAL",
        "EXPAND_CONTEXT_REFERENCE",
        "RUN_SCHEMA_VALIDATION",
        "RUN_TYPE_CHECK",
        "RUN_SELECTED_TEST",
        "RUN_FULL_VALIDATION",
        "RUN_SMT_OR_PROVER",
        "CALL_LOCAL_SMALL_MODEL",
        "CALL_REMOTE_STANDARD_MODEL",
        "CALL_REMOTE_STRONG_MODEL",
        "REQUEST_HUMAN_DECISION",
        "GENERATE_BOUNDED_REPAIR",
        "REPLAN_AFFECTED_SUFFIX",
        "QUARANTINE_TASK",
    }
    with pytest.raises(ValueError):
        AutonomyLevel("fully_autonomous")


def test_canonical_identity_ignores_input_mapping_and_id_order() -> None:
    first = BeliefFact(
        subject_question_id="question-1",
        predicate="typed_result",
        value={"b": 2, "a": 1},
        evidence_ids=("evidence-b", "evidence-a"),
        authority_class=AuthorityClass.VERIFIED,
        freshness=EvidenceFreshness.CURRENT,
        confidence_bp=9_000,
        observed_tree_id="tree-1",
    )
    second = BeliefFact(
        subject_question_id="question-1",
        predicate="typed_result",
        value={"a": 1, "b": 2},
        evidence_ids=("evidence-a", "evidence-b"),
        authority_class=AuthorityClass.VERIFIED,
        freshness=EvidenceFreshness.CURRENT,
        confidence_bp=9_000,
        observed_tree_id="tree-1",
    )
    assert first.content_id == second.content_id
    with pytest.raises(TypeError):
        first.value["new"] = 3  # type: ignore[index]


def test_identity_claim_and_duplicate_json_fail_closed() -> None:
    question = _question()
    forged = question.to_record()
    forged["content_id"] = "forged"
    with pytest.raises(AutonomyContractError, match="identity"):
        DecisionQuestion.from_dict(forged)
    with pytest.raises(AutonomyContractError, match="duplicate"):
        DecisionQuestion.from_json('{"schema":"x","schema":"x"}')


def test_payload_bounds_private_reasoning_and_floats_are_rejected() -> None:
    with pytest.raises(AutonomyContractError, match="bounded size"):
        _question(current_alternatives=("x" * 8_193,))
    with pytest.raises(AutonomyContractError, match="too many"):
        _question(current_alternatives=tuple(f"option-{index}" for index in range(1_025)))
    with pytest.raises(AutonomyContractError, match="forbidden"):
        ExperienceEpisode(
            frozen_input_ids=("tree-1",),
            question_feature_ids=("feature-1",),
            selected_action=MetaAction.NO_OP,
            selection_policy_id="policy-1",
            selection_policy_version="v1",
            terminal_status=TerminalStatus.SUCCEEDED,
            context_metrics={"raw_prompt": "do not persist this"},
        )
    with pytest.raises(AutonomyContractError, match="floats"):
        BeliefFact(
            subject_question_id="question-1",
            predicate="score",
            value=0.5,
            evidence_ids=("evidence-1",),
            authority_class=AuthorityClass.ADVISORY,
            freshness=EvidenceFreshness.CURRENT,
            confidence_bp=5_000,
            observed_tree_id="tree-1",
        )


def test_stale_authority_and_contradictory_question_evidence_fail_closed() -> None:
    with pytest.raises(AutonomyContractError, match="current evidence"):
        BeliefFact(
            subject_question_id="question-1",
            predicate="capability_live",
            value=True,
            evidence_ids=("old-capability",),
            authority_class=AuthorityClass.AUTHORITATIVE,
            freshness=EvidenceFreshness.STALE,
            confidence_bp=10_000,
            observed_tree_id="tree-old",
        )

    old_tree_fact = BeliefFact(
        subject_question_id="question-1",
        predicate="tree_matches",
        value=True,
        evidence_ids=("evidence-old-tree",),
        authority_class=AuthorityClass.AUTHORITATIVE,
        freshness=EvidenceFreshness.CURRENT,
        confidence_bp=10_000,
        observed_tree_id="tree-old",
    )
    with pytest.raises(AutonomyContractError, match="current tree"):
        BeliefState(
            objective_id="APMC-G000",
            objective_revision="revision-1",
            current_tree_id="tree-current",
            revision=1,
            facts=(old_tree_fact,),
        )
    with pytest.raises(AutonomyContractError, match="disjoint"):
        _question(
            known_evidence_ids=("evidence-1",),
            contradictory_evidence_ids=("evidence-1",),
        )


def test_remote_model_privacy_and_advisory_authority_are_rejected() -> None:
    with pytest.raises(AutonomyContractError, match="privacy"):
        _action(
            action=MetaAction.CALL_REMOTE_STRONG_MODEL,
            expected_evidence_kind=ResolutionEvidenceKind.MODEL_ADVICE,
            privacy_class=PrivacyClass.LOCAL_ONLY,
        )
    with pytest.raises(AutonomyContractError, match="advisory"):
        _action(authority_class=AuthorityClass.ADVISORY, accepted_as_authority=True)


def test_risk_and_autonomy_envelope_do_not_self_raise_authority() -> None:
    with pytest.raises(AutonomyContractError, match="R5"):
        _risk(
            risk_class=RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL,
            reversible=True,
            irreversible_external_effect=True,
        )
    risk = _risk(
        risk_class=RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL,
        reversible=False,
        irreversible_external_effect=True,
    )
    policy = AutonomyPolicy("v1", "authority-1", "human-policy-1")
    with pytest.raises(AutonomyContractError, match="cannot authorize execution"):
        AutonomyEnvelope(
            repository_id="repo-1",
            tree_id="tree-1",
            objective_id="APMC-G000",
            objective_revision="rev-1",
            task_id="APMC-001",
            acceptance_criterion_ids=("AC-1",),
            risk_assessment=risk,
            autonomy_level=AutonomyLevel.EXECUTE_REVERSIBLE,
            cognitive_budget=_budget(),
            allowed_paths=(),
            allowed_symbols=(),
            required_test_ids=(),
            required_proof_ids=(),
            authority_id="authority-1",
            policy_id=policy.policy_id,
            provider_usage_envelope_id="provider-budget-1",
            resource_budget_id="resource-budget-1",
            human_escalation_policy_id="human-policy-1",
            expiry_ms=10,
            reversible=False,
        )


@pytest.mark.parametrize("risk_class", tuple(RiskClass))
def test_observe_only_policy_cannot_self_raise_at_any_risk(
    risk_class: RiskClass,
) -> None:
    policy = AutonomyPolicy("v1", "authority-1", "human-policy-1")

    assert policy.allows(AutonomyLevel.OBSERVE_ONLY, risk_class)
    assert all(
        not policy.allows(level, risk_class)
        for level in AutonomyLevel
        if level.rank > AutonomyLevel.OBSERVE_ONLY.rank
    )


@pytest.mark.parametrize(
    ("risk_class", "highest_allowed", "first_forbidden"),
    (
        (
            RiskClass.R0_PURE,
            AutonomyLevel.EXECUTE_REVERSIBLE,
            AutonomyLevel.EXECUTE_BOUNDED_MUTATION,
        ),
        (
            RiskClass.R1_READ_ONLY,
            AutonomyLevel.EXECUTE_REVERSIBLE,
            AutonomyLevel.EXECUTE_BOUNDED_MUTATION,
        ),
        (
            RiskClass.R2_REVERSIBLE_LOCAL,
            AutonomyLevel.EXECUTE_REVERSIBLE,
            AutonomyLevel.EXECUTE_BOUNDED_MUTATION,
        ),
        (
            RiskClass.R3_BOUNDED_REPOSITORY_MUTATION,
            AutonomyLevel.EXECUTE_BOUNDED_MUTATION,
            AutonomyLevel.SELF_REPAIR_ISOLATED,
        ),
        (
            RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE,
            AutonomyLevel.DRY_RUN,
            AutonomyLevel.EXECUTE_REVERSIBLE,
        ),
        (
            RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL,
            AutonomyLevel.RECOMMEND,
            AutonomyLevel.DRY_RUN,
        ),
    ),
)
def test_active_policy_and_risk_ceiling_are_both_enforced(
    risk_class: RiskClass,
    highest_allowed: AutonomyLevel,
    first_forbidden: AutonomyLevel,
) -> None:
    policy = AutonomyPolicy(
        "v1",
        "authority-1",
        "human-policy-1",
        default_level=AutonomyLevel.EXECUTE_BOUNDED_MUTATION,
    )

    assert policy.allows(highest_allowed, risk_class)
    assert not policy.allows(first_forbidden, risk_class)


def test_budget_ledger_rejects_overspend_and_unattributed_tokens() -> None:
    budget = _budget(max_input_tokens=10)
    with pytest.raises(AutonomyContractError, match="attributed token"):
        BudgetLedger(budget=budget, epoch=1, committed_input_tokens=1)
    reservation = BudgetReservation(
        budget_id=budget.budget_id,
        idempotency_key="overrun-1",
        question_id="question-1",
        action_id="action-1",
        purpose=BudgetPurpose.ANALYSIS,
        status=BudgetReservationStatus.RECONCILED,
        max_input_tokens=10,
        actual_input_tokens=11,
        token_measurement_ids=("usage-1",),
    )
    with pytest.raises(AutonomyContractError, match="max_input_tokens"):
        BudgetLedger(
            budget=budget,
            epoch=1,
            reservations=(reservation,),
            committed_input_tokens=11,
            token_measurement_ids=("usage-1",),
        )


def test_route_policy_and_distilled_rules_cannot_self_promote_or_execute_code() -> None:
    with pytest.raises(AutonomyContractError, match="self|authorize"):
        RoutePolicyCandidate(
            parent_policy_id="policy-1",
            policy_version="v2",
            allowed_actions=(MetaAction.NO_OP,),
            feature_names=("risk",),
            integer_weights={"risk": 1},
            training_observation_ids=("observation-1",),
            held_out_evaluation_ids=("evaluation-1",),
            safety_gate_receipt_ids=("safety-1",),
            selection_reason="linear_score",
            external_authorization_id="candidate-says-yes",
        )
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


def test_human_packet_is_bounded_and_repair_scope_cannot_escape() -> None:
    with pytest.raises(AutonomyContractError, match="2 to 4"):
        HumanEscalationPacket(
            objective_id="APMC-G000",
            blocked_criterion_ids=("AC-1",),
            question="Choose",
            options=("a",),
            recommended_option="a",
            predicted_consequences={"a": "wait"},
            cost_and_risk={"a": "low"},
            evidence_ids=("evidence-1",),
            continuation_by_option={"a": "wait"},
            expires_at_ms=10,
        )
    with pytest.raises(AutonomyContractError, match="escapes allowed paths"):
        AutonomousRepairPlan(
            objective_id="APMC-G000",
            task_id="APMC-001",
            repair_tier=RepairTier.DETERMINISTIC,
            predicted_files=("docs/outside.md",),
            predicted_symbols=("heading",),
            patch_envelope_id="envelope-1",
            context_reference_ids=("context-1",),
            required_test_ids=(),
            required_proof_ids=(),
            worktree_id="worktree-1",
            allowed_paths=("ipfs_accelerate_py",),
            forbidden_symbols=(),
            rollback_plan_id="rollback-1",
            risk_class=RiskClass.R2_REVERSIBLE_LOCAL,
            max_changed_files=1,
            max_changed_lines=10,
        )


def test_receipts_cannot_claim_false_completion_merge_or_self_promotion() -> None:
    failed_gate = dict(SAFETY_GATES)
    failed_gate["false_completions"] = False
    with pytest.raises(AutonomyContractError, match="failed safety"):
        AutonomyRunReceipt(
            envelope_id="envelope-1",
            policy_id="policy-1",
            graph_id="graph-1",
            budget_ledger_id="ledger-1",
            terminal_status=TerminalStatus.SUCCEEDED,
            safety_gate_results=failed_gate,
        )
    with pytest.raises(AutonomyContractError, match="authorize merge"):
        AutonomousRepairReceipt(
            plan_id="plan-1",
            envelope_id="envelope-1",
            terminal_status=TerminalStatus.SUCCEEDED,
            changed_paths=(),
            validation_receipt_ids=("validation-1",),
            proof_receipt_ids=(),
            adversarial_assurance_receipt_ids=(),
            authorizes_merge=True,
        )
    with pytest.raises(AutonomyContractError, match="own promotion"):
        AutonomyPromotionReceipt(
            candidate_policy_id="candidate-1",
            expected_old_policy_id="policy-1",
            resulting_policy_id="policy-2",
            status=PromotionStatus.PROMOTED,
            safety_gate_results=SAFETY_GATES,
            held_out_evaluation_ids=("held-out-1",),
            safety_gate_receipt_ids=("safety-1",),
            authorization_id="candidate-authorization",
            compare_and_swap_receipt_id="cas-1",
            rollback_policy_id="policy-1",
            self_authorized=True,
        )


def test_program_identity_is_exact() -> None:
    assert AUTONOMOUS_META_CONTROLLER_PROGRAM_ID == "agent-supervisor-autonomous-meta-controller-v1"
