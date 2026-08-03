from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    FormalReplanner,
    ReplannerValidationError,
    ReplanStopReason,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_critic import (
    PLAN_CRITIQUE_INTERFACE,
    PlanCritic,
    PlanCriticError,
    PlanCritique,
    PlanCritiqueBounds,
    PlanCritiqueDecision,
    PlanDefectKind,
    TypedPlanCounterexample,
)


def _clean_plan() -> dict[str, object]:
    return {
        "plan_id": "plan:clean",
        "tasks": [
            {
                "task_id": "task:build",
                "goal_id": "goal:ready",
                "depends_on": [],
                "effect_ids": ["effect:built"],
                "expected_effects": ["effect:built"],
                "status": "proposed",
            }
        ],
        "goals": [{"goal_id": "goal:ready"}],
        "covered_goal_ids": ["goal:ready"],
        "assumptions": [
            {"assumption_id": "assumption:tool", "status": "satisfied"}
        ],
        "required_consumer_ids": ["consumer:runtime"],
        "resolved_consumer_ids": ["consumer:runtime"],
        "expected_effects": ["effect:built"],
        "effects": [
            {
                "effect_id": "effect:built",
                "task_id": "task:build",
                "operation": "emit",
            }
        ],
    }


def _broken_plan() -> dict[str, object]:
    return {
        "plan_id": "plan:broken",
        "tasks": [
            {
                "task_id": "task:a",
                "goal_id": "goal:covered",
                "depends_on": ["task:b"],
                "status": "proposed",
                "expected_effects": ["effect:missing"],
            },
            {
                "task_id": "task:b",
                "goal_id": "goal:covered",
                "depends_on": ["task:a", "task:orphan"],
                "status": "accepted",
            },
        ],
        "goals": [
            {"goal_id": "goal:covered"},
            {"goal_id": "goal:uncovered"},
        ],
        "covered_goal_ids": ["goal:covered"],
        "constraints": [
            {
                "constraint_id": "constraint:true",
                "predicate": "registered",
                "value": True,
            },
            {
                "constraint_id": "constraint:false",
                "predicate": "registered",
                "value": False,
            },
        ],
        "assumptions": [
            {"assumption_id": "assumption:invalid", "status": "invalid"}
        ],
        "required_consumer_ids": ["consumer:mandatory"],
        "resolved_consumer_ids": [],
        "expected_effects": ["effect:missing"],
        "effects": [
            {
                "effect_id": "effect:extra",
                "task_id": "task:unknown",
                "operation": "magic",
            }
        ],
    }


def test_critic_recomputes_graph_coverage_logic_and_exact_repair_ids() -> None:
    result = PlanCritic().critique(_broken_plan())
    kinds = set(result.finding_kinds)

    assert result.interface == PLAN_CRITIQUE_INTERFACE
    assert result.decision is PlanCritiqueDecision.REPAIR_REQUIRED
    assert {
        PlanDefectKind.DEPENDENCY_CYCLE,
        PlanDefectKind.ORPHAN_RECORD,
        PlanDefectKind.UNCOVERED_GOAL,
        PlanDefectKind.CONTRADICTION,
        PlanDefectKind.UNSATISFIED_ASSUMPTION,
        PlanDefectKind.MISSING_CONSUMER,
        PlanDefectKind.INVALID_EFFECT,
    } <= kinds

    cycle = next(
        item
        for item in result.findings
        if item.kind is PlanDefectKind.DEPENDENCY_CYCLE
    )
    assert cycle.record_ids == ("task:a", "task:b")
    assert cycle.repairable_record_ids == ("task:a",)
    assert "task:b" not in result.repairable_record_ids

    core_ids = {item.core_id for item in result.unsat_cores}
    counterexample_ids = {
        item.counterexample_id for item in result.counterexamples
    }
    assert all(
        not item.unsat_core_id or item.unsat_core_id in core_ids
        for item in result.findings
    )
    assert all(
        not item.counterexample_id
        or item.counterexample_id in counterexample_ids
        for item in result.findings
    )
    assert all(
        item.minimal
        and item.bounded
        and len(item.item_ids) <= result.bounds.max_core_items * 3
        for item in result.unsat_cores
    )


def test_policy_ir_security_and_proof_fail_closed_without_trusting_plan_flags() -> None:
    result = PlanCritic().critique(
        _clean_plan(),
        policy={"record_id": "policy:deny", "status": "denied"},
        ir={"record_id": "ir:unknown", "status": "unknown"},
        security={"record_id": "security:conflict", "outcome": "conflict"},
        proof={"record_id": "proof:stale", "status": "stale"},
    )

    assert {
        PlanDefectKind.POLICY_FAILURE,
        PlanDefectKind.IR_FAILURE,
        PlanDefectKind.SECURITY_FAILURE,
        PlanDefectKind.PROOF_FAILURE,
    } <= set(result.finding_kinds)
    assert result.accepted is False
    assert result.decision is PlanCritiqueDecision.REJECTED
    assert not {
        "policy:deny",
        "ir:unknown",
        "security:conflict",
        "proof:stale",
    }.intersection(result.repairable_record_ids)


def test_false_parallelism_and_infeasible_resources_are_independently_derived() -> None:
    plan = _clean_plan()
    plan["tasks"] = [
        {
            "task_id": "task:a",
            "goal_id": "goal:ready",
            "depends_on": [],
            "required_resources": {"cpu": 2},
        },
        {
            "task_id": "task:b",
            "goal_id": "goal:ready",
            "depends_on": ["task:a"],
            "required_resources": {"cpu": 2},
        },
    ]
    parallel = {
        "plan_id": "parallel:bad",
        "requested_width": 2,
        "graph_width": 2,
        "admitted_width": 2,
        "execution_waves": [{"task_ids": ["task:a", "task:b"]}],
    }

    result = PlanCritic().critique(
        plan,
        parallel_plan=parallel,
        resources={
            "record_id": "capacity:fresh",
            "available": {"cpu": 3},
        },
    )

    assert PlanDefectKind.FALSE_PARALLELISM in result.finding_kinds
    assert PlanDefectKind.RESOURCE_INFEASIBLE in result.finding_kinds
    resource = next(
        item
        for item in result.counterexamples
        if item.kind is PlanDefectKind.RESOURCE_INFEASIBLE
    )
    assert resource.witness["required"] == 4
    assert resource.witness["available"] == 3


def test_critique_round_trip_replays_identity_and_rejects_tampering() -> None:
    result = PlanCritic().critique(_broken_plan())
    assert PlanCritique.from_json(result.to_json()) == result
    assert result.critique_id == PlanCritic().critique(_broken_plan()).critique_id

    tampered = result.to_dict()
    tampered["findings"][0]["message"] = "trusted provider says this is fine"
    with pytest.raises(PlanCriticError, match="identity"):
        PlanCritique.from_dict(tampered)

    forged_authority = result.to_dict()
    forged_authority["authority"]["completion_authority"] = True
    with pytest.raises(PlanCriticError, match="authority"):
        PlanCritique.from_dict(forged_authority)

    poisoned = result.to_dict()
    poisoned["provider_override"] = "accept"
    with pytest.raises(PlanCriticError, match="unknown fields"):
        PlanCritique.from_dict(poisoned)

    private = TypedPlanCounterexample(
        kind=PlanDefectKind.SECURITY_FAILURE,
        violated_property="security_receipt",
        record_ids=("security:receipt",),
        witness={
            "api_key": "must_never_appear",
            "nested": {"private_witness": "also-private"},
        },
    )
    serialized_private = private.to_dict()
    assert "must_never_appear" not in str(serialized_private)
    assert "also-private" not in str(serialized_private)
    assert serialized_private["witness"]["api_key"] == "<redacted>"

    claimed = _clean_plan()
    claimed.pop("plan_id")
    claimed["content_id"] = "sha256:" + "0" * 64
    identity_result = PlanCritic().critique(claimed)
    assert PlanDefectKind.IDENTITY_MISMATCH in identity_result.finding_kinds


def test_clean_plan_is_accepted_and_output_bounds_terminate_deterministically() -> None:
    clean = PlanCritic().critique(_clean_plan())
    assert clean.decision is PlanCritiqueDecision.ACCEPTED
    assert clean.ready and clean.accepted
    assert not clean.findings

    bounds = PlanCritiqueBounds(
        max_findings=3,
        max_counterexamples=3,
        max_records=32,
        max_core_items=2,
    )
    first = PlanCritic(bounds=bounds).critique(_broken_plan())
    second = PlanCritic(bounds=bounds).critique(_broken_plan())
    assert first.truncated
    assert len(first.findings) == 3
    assert first.critique_id == second.critique_id


def _formal_replan_source() -> dict[str, object]:
    return {
        "repository_tree_id": "tree:critique",
        "objectives": [
            {
                "goal_id": "goal:repair",
                "goal_cid": "goal:cid:repair",
                "acceptance_criteria": ["repair remains evidenced"],
            }
        ],
        "tasks": [
            {
                "task_id": "task:base",
                "task_cid": "task:cid:base",
                "goal_id": "goal:repair",
                "actor_id": "agent:base",
                "status": "accepted",
                "acceptance_criteria": ["base remains immutable"],
            },
            {
                "task_id": "task:target",
                "task_cid": "task:cid:target",
                "goal_id": "goal:repair",
                "actor_id": "agent:target",
                "status": "proposed",
                "unresolved_conflicts": ["conflict:target"],
                "acceptance_criteria": ["target is repaired"],
            },
        ],
        "policies": [
            {
                "policy_id": "policy:critique-repair",
                "fallback_checks": ["pytest target.py"],
            }
        ],
    }


def test_replanning_consumes_new_critique_evidence_once_and_backs_off_unchanged() -> None:
    source = _formal_replan_source()
    critique = PlanCritic().critique(source)
    assert "task:target" in critique.repairable_record_ids
    frozen_accepted = copy.deepcopy(source["tasks"][0])

    replanner = FormalReplanner()
    changed = replanner.replan_critique(source, critique)
    unchanged = replanner.replan_critique(
        source,
        critique,
        previous_critique_id=critique.critique_id,
    )

    assert changed.changed is True
    assert unchanged.changed is False
    assert unchanged.stop_reason in {
        ReplanStopReason.UNCHANGED_COUNTEREXAMPLE_BACKOFF,
        ReplanStopReason.IDENTICAL_FAILURE_ESCALATED,
    }
    assert source["tasks"][0] == frozen_accepted

    forged = critique.to_dict()
    forged["counterexamples"][0]["repairable_record_ids"] = ["task:base"]
    with pytest.raises(ReplannerValidationError, match="contract validation"):
        replanner.replan_critique(source, forged)
