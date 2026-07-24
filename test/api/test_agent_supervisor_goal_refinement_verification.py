from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_logic_vocabulary import (
    LOGIC_VOCABULARY_VERSION,
    ReviewedPredicate,
    TDFOLVocabulary,
    TermSort,
    atom,
    constant,
)
from ipfs_accelerate_py.agent_supervisor.formal_planning_contracts import (
    Actor,
    ActorKind,
    Effect,
    EffectOperation,
    EventKind,
    EvidenceRequirement,
    EvidenceRequirementKind,
    Fluent,
    FluentValueType,
    FormalWorkPlan,
    Goal,
    Norm,
    NormKind,
    PlanEvent,
    PlanTask,
    RefinementMode,
    Subgoal,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    ContractValidationError,
)
from ipfs_accelerate_py.agent_supervisor.goal_refinement_verification import (
    MAX_LEANSTRAL_REPAIR_ROUNDS,
    FrozenRefinementContext,
    GoalRefinementVerifier,
    JsonlRefinementAuditStore,
    RefinementCounterexample,
    RefinementObligation,
    RefinementObligationKind,
    RefinementRepairCandidate,
    RefinementVerificationAttempt,
    RefinementVerificationPolicy,
    RefinementVerificationResult,
    RefinementVerificationStatus,
    RepairImmutabilityReceipt,
    derive_refinement_obligations,
)
from ipfs_accelerate_py.agent_supervisor.multi_prover_router import (
    AttemptOutcome,
    MultiProverRouter,
    PropertyKind,
    PropertyPolicy,
    ProverLane,
    ProverOutput,
    ProverRole,
)


def _plan(*, equivalent: bool = False) -> FormalWorkPlan:
    ready = atom(
        ReviewedPredicate.TASK_READY,
        constant(TermSort.TASK, "task:implement"),
    )
    goal_done = atom(
        ReviewedPredicate.GOAL_SATISFIED,
        constant(TermSort.GOAL, "goal:root"),
    )
    subgoal_done = TDFOLVocabulary.subgoal_satisfaction("subgoal:child", 4)
    evidence = EvidenceRequirement(
        requirement_id="evidence:test",
        kind=EvidenceRequirementKind.TEST,
        subject_ids=("task:implement", "subgoal:child"),
        freshness_seconds=60,
        fallback_check_ids=("pytest:refinement",),
    )
    return FormalWorkPlan(
        vocabulary_profile_id="reviewed-tdfol",
        vocabulary_version=LOGIC_VOCABULARY_VERSION,
        source_ids=("source:objective",),
        repository_tree_id="tree:1",
        trace_bound=4,
        actors=(
            Actor(
                "actor:supervisor",
                ActorKind.SUPERVISOR,
                authority_ids=("authority:assign",),
            ),
            Actor(
                "actor:agent",
                ActorKind.AGENT,
                capabilities=("python",),
                authority_ids=("authority:implement",),
            ),
        ),
        goals=(
            Goal(
                "goal:root",
                "actor:supervisor",
                goal_done.formula_id,
                evidence_requirement_ids=("evidence:test",),
            ),
        ),
        subgoals=(
            Subgoal(
                "subgoal:child",
                "goal:root",
                subgoal_done.formula_id,
                refinement_mode=(
                    RefinementMode.EQUIVALENT
                    if equivalent
                    else RefinementMode.SUFFICIENT
                ),
                evidence_requirement_ids=("evidence:test",),
            ),
        ),
        tasks=(
            PlanTask(
                "task:implement",
                "goal:root",
                subgoal_id="subgoal:child",
                actor_ids=("actor:agent",),
                effect_ids=("effect:done",),
                event_ids=("event:execute",),
                evidence_requirement_ids=("evidence:test",),
                metadata={
                    "required_authority_ids": ["authority:implement"],
                    "resource_ids": ["cpu"],
                },
            ),
        ),
        events=(
            PlanEvent(
                "event:execute",
                EventKind.EXECUTED,
                "actor:agent",
                "task:implement",
                2,
            ),
        ),
        fluents=(
            Fluent("fluent:done", FluentValueType.BOOLEAN, initial_value=False),
        ),
        preconditions=(),
        effects=(
            Effect(
                "effect:done",
                EffectOperation.ASSIGN,
                "task:implement",
                fluent_id="fluent:done",
                event_id="event:execute",
                value=True,
            ),
        ),
        norms=(
            Norm(
                "norm:implement",
                NormKind.OBLIGATION,
                "actor:agent",
                "task:implement",
                issuer_actor_id="actor:supervisor",
                activation_formula_id=ready.formula_id,
                valid_from=0,
                valid_until=4,
            ),
        ),
        temporal_constraints=(),
        evidence_requirements=(evidence,),
        formulas=(ready, goal_done, subgoal_done),
        metadata={"resource_ids": ["cpu", "memory"]},
    )


def test_derives_all_required_obligations_deterministically_and_classifies_them():
    plan = _plan(equivalent=True)

    first = derive_refinement_obligations(
        plan, assumption_ids=("assumption:b", "assumption:a")
    )
    second = derive_refinement_obligations(
        plan, assumption_ids=("assumption:a", "assumption:b")
    )

    assert first == second
    assert {item.kind for item in first} == set(RefinementObligationKind)
    assert {item.property_kind for item in first} == {
        PropertyKind.TYPED_PLANNING,
        PropertyKind.TEMPORAL_DEONTIC,
        PropertyKind.FINITE_CONSTRAINT,
        PropertyKind.STATE_MACHINE,
        PropertyKind.FIRST_ORDER_THEOREM,
    }
    refinement = next(
        item for item in first if item.kind is RefinementObligationKind.CHILD_TO_PARENT
    )
    assert refinement.statement_model["directions"] == [
        "child_implies_parent",
        "parent_implies_child",
    ]
    assert refinement.assumption_ids == ("assumption:a", "assumption:b")
    assert RefinementObligation.from_dict(refinement.to_record()) == refinement
    assert refinement.to_property_obligation().metadata["plan_id"] == plan.content_id


def test_sufficient_refinement_never_silently_adds_reverse_implication():
    obligation = next(
        item
        for item in derive_refinement_obligations(_plan())
        if item.kind is RefinementObligationKind.CHILD_TO_PARENT
    )

    assert obligation.statement_model["relation"] == "sufficient"
    assert obligation.statement_model["directions"] == ["child_implies_parent"]


def test_leanstral_lane_is_candidate_only_and_cannot_claim_authority():
    with pytest.raises(ContractValidationError, match="model-assistant"):
        ProverLane(
            "leanstral",
            ProverRole.MODEL_CHECKER,
            authority_capability="finite_constraint_satisfiability",
        )

    policies = dict(MultiProverRouter().policies)
    policies[PropertyKind.FINITE_CONSTRAINT] = PropertyPolicy(
        PropertyKind.FINITE_CONSTRAINT,
        (
            ProverLane("leanstral", ProverRole.MODEL_ASSISTANT, stage=0),
            ProverLane(
                "z3",
                ProverRole.MODEL_CHECKER,
                stage=1,
                authority_capability="finite_constraint_satisfiability",
            ),
        ),
    )
    obligation = next(
        item
        for item in derive_refinement_obligations(_plan())
        if item.kind is RefinementObligationKind.AUTHORITY
    )
    result = MultiProverRouter(policies).execute(
        obligation.to_property_obligation(),
        lambda request, cancellation: ProverOutput(AttemptOutcome.VERIFIED),
    )

    assert result.proved
    assistant = result.attempts[0]
    assert assistant.role is ProverRole.MODEL_ASSISTANT
    assert assistant.effective_outcome is AttemptOutcome.CANDIDATE
    assert not assistant.authoritative
    assert result.attempts[1].authoritative


def test_high_assurance_obligations_require_independent_kernel_reconstruction():
    plan = _plan()

    def runner(request, cancellation):
        if request.lane.role is ProverRole.KERNEL:
            return ProverOutput(AttemptOutcome.UNKNOWN)
        return ProverOutput(AttemptOutcome.VERIFIED)

    result = GoalRefinementVerifier().verify(plan, runner)

    assert result.status is RefinementVerificationStatus.INCONCLUSIVE
    logical = [
        portfolio
        for portfolio in result.rounds[0].portfolio_results
        if portfolio.plan.obligation.property_kind
        in (
            PropertyKind.TYPED_PLANNING,
            PropertyKind.TEMPORAL_DEONTIC,
            PropertyKind.FIRST_ORDER_THEOREM,
        )
    ]
    assert logical
    assert all(not item.proved for item in logical)
    assert all(
        not attempt.authoritative
        for item in logical
        for attempt in item.attempts
        if attempt.role
        in (
            ProverRole.MODEL_ASSISTANT,
            ProverRole.DOMAIN_REASONER,
            ProverRole.ORCHESTRATOR,
            ProverRole.CANDIDATE,
        )
    )


def test_persists_every_attempt_and_bounded_counterexample(tmp_path):
    store = JsonlRefinementAuditStore(tmp_path / "refinement-attempts.jsonl")

    def runner(request, cancellation):
        if request.obligation.property_kind is PropertyKind.FINITE_CONSTRAINT:
            return ProverOutput(
                AttemptOutcome.COUNTEREXAMPLE,
                evidence={"model": {"authorized": False}},
                conclusive=True,
            )
        return ProverOutput(AttemptOutcome.UNKNOWN)

    result = GoalRefinementVerifier(audit_store=store).verify(_plan(), runner)
    records = store.read_records()

    assert result.status is RefinementVerificationStatus.DISPROVED
    attempt_records = [
        item for item in records if item["schema"] == RefinementVerificationAttempt.SCHEMA
    ]
    counterexample_records = [
        item for item in records if item["schema"] == RefinementCounterexample.SCHEMA
    ]
    assert len(attempt_records) == len(result.all_attempts)
    assert counterexample_records
    assert all(item["bounds"] == {"trace_bound": 4} for item in counterexample_records)
    assert all(item["content_id"] for item in records)
    assert len(store.path.read_text(encoding="utf-8").splitlines()) == len(records)


def test_two_repair_round_cap_and_frozen_root_assumption_receipts(tmp_path):
    plan = _plan()
    repaired = replace(plan, metadata={**plan.metadata, "repair": 1})
    policy = RefinementVerificationPolicy(
        allow_leanstral_repairs=True,
        max_repair_rounds=MAX_LEANSTRAL_REPAIR_ROUNDS,
    )
    store = JsonlRefinementAuditStore(tmp_path / "repairs.jsonl")
    repair_calls = []

    def runner(request, cancellation):
        if request.obligation.metadata["plan_id"] == plan.content_id:
            return ProverOutput(AttemptOutcome.UNKNOWN)
        return ProverOutput(AttemptOutcome.VERIFIED)

    def repairer(request):
        repair_calls.append(request)
        return RefinementRepairCandidate(repaired, request.frozen_context)

    result = GoalRefinementVerifier(
        policy=policy, audit_store=store
    ).verify(
        plan,
        runner,
        assumption_ids=("assumption:reviewed",),
        repairer=repairer,
    )

    assert result.verified
    assert RefinementVerificationResult.from_dict(result.to_record()) == result
    assert len(result.rounds) == 2
    assert len(repair_calls) == 1
    assert len(result.repair_receipts) == 1
    receipt = result.repair_receipts[0]
    assert receipt.root_unchanged and receipt.assumptions_unchanged
    assert receipt.accepted
    assert any(
        item["schema"] == RepairImmutabilityReceipt.SCHEMA
        for item in store.read_records()
    )
    with pytest.raises(ContractValidationError, match="must not exceed"):
        RefinementVerificationPolicy(
            allow_leanstral_repairs=True,
            max_repair_rounds=MAX_LEANSTRAL_REPAIR_ROUNDS + 1,
        )


def test_repair_that_changes_root_or_assumptions_fails_closed_and_is_audited(
    tmp_path,
):
    plan = _plan()
    changed_goal = replace(plan.goals[0], metadata={"mutation": True})
    changed_plan = replace(plan, goals=(changed_goal,))
    policy = RefinementVerificationPolicy(
        allow_leanstral_repairs=True, max_repair_rounds=1
    )
    store = JsonlRefinementAuditStore(tmp_path / "rejected-repair.jsonl")

    def repairer(request):
        changed_context = FrozenRefinementContext(
            request.frozen_context.root_goal_id,
            changed_goal.content_id,
            ("assumption:hidden",),
        )
        return RefinementRepairCandidate(changed_plan, changed_context)

    result = GoalRefinementVerifier(
        policy=policy, audit_store=store
    ).verify(
        plan,
        lambda request, cancellation: ProverOutput(AttemptOutcome.UNKNOWN),
        assumption_ids=("assumption:reviewed",),
        repairer=repairer,
    )

    assert result.status is RefinementVerificationStatus.ERROR
    assert len(result.rounds) == 1
    assert len(result.repair_receipts) == 1
    assert not result.repair_receipts[0].accepted
    assert not result.repair_receipts[0].root_unchanged
    assert not result.repair_receipts[0].assumptions_unchanged
    journal = [
        json.loads(line)
        for line in store.path.read_text(encoding="utf-8").splitlines()
    ]
    assert journal[-1]["schema"] == RepairImmutabilityReceipt.SCHEMA
    assert journal[-1]["accepted"] is False
