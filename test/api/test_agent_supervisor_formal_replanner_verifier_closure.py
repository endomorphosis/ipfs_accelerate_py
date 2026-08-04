"""Verifier-backed counterexample closure for formal replanning (FVT-G008).

Structural plan repairs may be admissible without claiming semantic success.
Only a fresh verifier receipt matching the repaired tree, property, assumptions,
tool, policy, and bounds may reduce the open-witness count to zero.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import (
    compile_formal_plan,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    VERIFIER_BACKED_REPAIR_CLOSURE_SCHEMA,
    FormalReplanner,
    RepairCandidateStatus,
    RepairOperation,
    RepairRuleKind,
    ReplanStopReason,
    VerifierBackedRepairClosure,
    VerifierClosureReceipt,
    WitnessClosureStatus,
    _bound_digest,
    _closure_binding,
    evaluate_verifier_backed_closure,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_counterexamples import (
    CounterexampleKind,
    RepairClass,
    normalize_counterexample,
)


def _source() -> dict[str, object]:
    return {
        "repository_tree_id": "tree:repair-v1",
        "objectives": [
            {
                "goal_id": "G12.S4",
                "goal_cid": "goal:cid:g12-s4",
                "acceptance_criteria": ["The intended transition is evidenced."],
            }
        ],
        "tasks": [
            {
                "task_id": "REF-BASE",
                "task_cid": "task:cid:base",
                "goal_id": "G12.S4",
                "actor_id": "agent:base",
                "changed_ast_scopes": ["symbol:base"],
                "acceptance_criteria": ["base test"],
            },
            {
                "task_id": "REF-TARGET",
                "task_cid": "task:cid:target",
                "goal_id": "G12.S4",
                "actor_id": "agent:target",
                "changed_ast_scopes": ["symbol:target", "symbol:unrelated"],
                "effects": [
                    {
                        "operation": "assign",
                        "fluent_id": "target:built",
                        "value": True,
                    },
                    {
                        "operation": "assign",
                        "fluent_id": "target:tested",
                        "value": True,
                    },
                ],
                "acceptance_criteria": ["target test"],
            },
        ],
        "policies": [
            {
                "policy_id": "policy:formal-repair",
                "fallback_checks": ["pytest baseline.py"],
            }
        ],
    }


def _counterexample(
    source: dict[str, object] | None = None,
    *,
    repair_classes: tuple[RepairClass, ...] = (RepairClass.ADD_DEPENDENCY,),
):
    active_source = source or _source()
    compiled = compile_formal_plan(active_source)
    assert compiled.plan is not None
    return normalize_counterexample(
        {
            "kind": CounterexampleKind.GENERIC_FAILURE.value,
            "failure": {"code": "focused-repair-required"},
        },
        kind=CounterexampleKind.GENERIC_FAILURE,
        violated_property="goal transition must remain valid",
        bindings={
            "plan_id": compiled.plan_id,
            "task_id": "REF-TARGET",
            "ast_scope_id": "symbol:target",
            "tree_id": str(active_source["repository_tree_id"]),
            "assumption_id": "assumption:dep-ready",
            "provider_id": "tool:z3",
            "policy_id": "policy:formal-repair",
        },
        finite_bounds={"portfolio_width": 1, "deadline": 20},
        repair_classes=repair_classes,
    )


def _operation(counterexample_id: str) -> RepairOperation:
    return RepairOperation(
        kind=RepairRuleKind.ADD_DEPENDENCY,
        target_task_id="REF-TARGET",
        parameters={"dependency_task_id": "REF-BASE"},
        counterexample_id=counterexample_id,
    )


def _matching_receipt(
    source: dict[str, object],
    counterexample,
    *,
    repaired_plan_id: str,
    outcome: str = "verified",
    freshness: str = "current",
    available: bool = True,
    **overrides: Any,
) -> VerifierClosureReceipt:
    binding = _closure_binding(
        source,
        counterexample,
        repaired_plan_id=repaired_plan_id,
        tool_id="tool:z3",
        policy_id="policy:formal-repair",
    )
    payload = {
        "receipt_id": "receipt:fresh-verifier-1",
        "counterexample_id": binding["counterexample_id"],
        "repository_tree_id": binding["repository_tree_id"],
        "property_id": binding["property_id"],
        "assumption_ids": list(binding["assumption_ids"]),
        "bound_digest": binding["bound_digest"],
        "tool_id": binding["tool_id"] or "tool:z3",
        "policy_id": binding["policy_id"] or "policy:formal-repair",
        "repaired_plan_id": repaired_plan_id,
        "freshness": freshness,
        "outcome": outcome,
        "available": available,
    }
    payload.update(overrides)
    return VerifierClosureReceipt.from_dict(payload)


def test_addresses_counterexample_alone_never_zeros_open_count() -> None:
    source = _source()
    counterexample = _counterexample(source)
    operation = _operation(counterexample.semantic_id)
    repaired, _changed, _generated, _taskboard = FormalReplanner()._apply(
        source, operation
    )
    assert FormalReplanner._addresses_counterexample(
        repaired, operation, counterexample
    )
    # Direct structural postcondition check is not semantic verification.
    structural = evaluate_verifier_backed_closure(
        counterexample_id=counterexample.semantic_id,
        structural_addressed=True,
    )
    assert structural.open_counterexamples == 1
    assert structural.status is WitnessClosureStatus.UNKNOWN
    assert not structural.closed
    assert structural.verifier_receipt_id == ""

    result = FormalReplanner().replan(
        source,
        counterexample,
        candidate_repairs=(operation,),
    )
    assert result.stop_reason is ReplanStopReason.ADMITTED
    assert result.selected is not None
    assert result.selected.status is RepairCandidateStatus.ADMITTED
    progress = result.selected.transition.progress
    assert progress.before_open_counterexamples == 1
    assert progress.after_open_counterexamples == 1
    assert progress.improved
    assert result.selected.closure is not None
    assert result.selected.closure.open_counterexamples == 1
    assert not result.selected.verifier_confirmed


def test_no_verifier_leaves_witness_unknown() -> None:
    closure = evaluate_verifier_backed_closure(
        counterexample_id="cx:1",
        structural_addressed=True,
        verifier_available=None,
        receipt=None,
    )
    assert closure.status is WitnessClosureStatus.UNKNOWN
    assert closure.open_counterexamples == 1
    assert closure.reason_code == "no_verifier_receipt"


def test_unavailable_verifier_leaves_witness_unknown() -> None:
    source = _source()
    counterexample = _counterexample(source)
    operation = _operation(counterexample.semantic_id)

    def _unavailable(_binding: dict[str, Any]) -> None:
        raise RuntimeError("backend offline")

    result = FormalReplanner(
        verifier=_unavailable,
        verifier_available=True,
    ).replan(source, counterexample, candidate_repairs=(operation,))
    assert result.selected is not None
    closure = result.selected.closure
    assert closure is not None
    assert closure.status is WitnessClosureStatus.UNKNOWN
    assert closure.open_counterexamples == 1
    assert closure.reason_code == "verifier_unavailable"
    assert result.selected.transition.progress.after_open_counterexamples == 1

    explicit = evaluate_verifier_backed_closure(
        counterexample_id=counterexample.semantic_id,
        structural_addressed=True,
        verifier_available=False,
    )
    assert explicit.status is WitnessClosureStatus.UNKNOWN
    assert explicit.reason_code == "verifier_unavailable"


def test_stale_receipt_leaves_witness_open() -> None:
    source = _source()
    counterexample = _counterexample(source)
    compiled = compile_formal_plan(source)
    receipt = _matching_receipt(
        source,
        counterexample,
        repaired_plan_id=compiled.plan_id,
        freshness="stale",
    )
    closure = evaluate_verifier_backed_closure(
        counterexample_id=counterexample.semantic_id,
        structural_addressed=True,
        repository_tree_id=receipt.repository_tree_id,
        property_id=receipt.property_id,
        assumption_ids=receipt.assumption_ids,
        bound_digest=receipt.bound_digest,
        tool_id=receipt.tool_id,
        policy_id=receipt.policy_id,
        repaired_plan_id=receipt.repaired_plan_id,
        receipt=receipt,
    )
    assert closure.status is WitnessClosureStatus.OPEN
    assert closure.open_counterexamples == 1
    assert closure.reason_code == "stale_receipt"
    assert not closure.closed


@pytest.mark.parametrize(
    ("field", "value", "token"),
    [
        ("repository_tree_id", "tree:other", "tree"),
        ("property_id", "different property", "property"),
        ("assumption_ids", ("assumption:other",), "assumption"),
        ("bound_digest", "sha256:wrong-bounds", "bound"),
        ("tool_id", "tool:cvc5", "tool"),
        ("policy_id", "policy:other", "policy"),
        ("repaired_plan_id", "plan:other", "plan"),
    ],
)
def test_changed_binding_leaves_witness_open(
    field: str, value: object, token: str
) -> None:
    source = _source()
    counterexample = _counterexample(source)
    binding = _closure_binding(
        source,
        counterexample,
        repaired_plan_id="plan:repaired",
        tool_id="tool:z3",
        policy_id="policy:formal-repair",
    )
    receipt_kwargs: dict[str, Any] = {"repaired_plan_id": "plan:repaired"}
    receipt_kwargs[field] = value
    receipt = _matching_receipt(
        source,
        counterexample,
        **receipt_kwargs,
    )
    closure = evaluate_verifier_backed_closure(
        structural_addressed=True,
        receipt=receipt,
        **binding,
    )
    assert closure.status is WitnessClosureStatus.OPEN
    assert closure.open_counterexamples == 1
    assert closure.reason_code.startswith("binding_mismatch:")
    assert token in closure.reason_code


def test_timeout_and_disagreement_leave_witness_open_or_unknown() -> None:
    source = _source()
    counterexample = _counterexample(source)
    binding = _closure_binding(
        source,
        counterexample,
        repaired_plan_id="plan:repaired",
        tool_id="tool:z3",
        policy_id="policy:formal-repair",
    )
    timeout = evaluate_verifier_backed_closure(
        structural_addressed=True,
        receipt=_matching_receipt(
            source,
            counterexample,
            repaired_plan_id="plan:repaired",
            outcome="timeout",
        ),
        **binding,
    )
    assert timeout.status is WitnessClosureStatus.UNKNOWN
    assert timeout.open_counterexamples == 1
    assert timeout.reason_code == "verifier_timeout"

    disagreement = evaluate_verifier_backed_closure(
        structural_addressed=True,
        receipt=_matching_receipt(
            source,
            counterexample,
            repaired_plan_id="plan:repaired",
            outcome="disagreement",
        ),
        **binding,
    )
    assert disagreement.status is WitnessClosureStatus.OPEN
    assert disagreement.open_counterexamples == 1
    assert disagreement.reason_code == "verifier_disagreement"


def test_fresh_matching_receipt_closes_and_names_receipt() -> None:
    source = _source()
    counterexample = _counterexample(source)
    operation = _operation(counterexample.semantic_id)

    def _verifier(binding: dict[str, Any]) -> VerifierClosureReceipt:
        return VerifierClosureReceipt(
            receipt_id="receipt:fresh-closed",
            counterexample_id=binding["counterexample_id"],
            repository_tree_id=binding["repository_tree_id"],
            property_id=binding["property_id"],
            assumption_ids=tuple(binding["assumption_ids"]),
            bound_digest=binding["bound_digest"],
            tool_id=binding["tool_id"] or "tool:z3",
            policy_id=binding["policy_id"] or "policy:formal-repair",
            repaired_plan_id=binding["repaired_plan_id"],
            freshness="current",
            outcome="verified",
            available=True,
        )

    result = FormalReplanner(verifier=_verifier).replan(
        source,
        counterexample,
        candidate_repairs=(operation,),
    )
    assert result.stop_reason is ReplanStopReason.ADMITTED
    assert result.selected is not None
    closure = result.selected.closure
    assert closure is not None
    assert closure.status is WitnessClosureStatus.CLOSED
    assert closure.open_counterexamples == 0
    assert closure.closed
    assert closure.verifier_receipt_id == "receipt:fresh-closed"
    assert closure.reason_code == "fresh_matching_verifier_receipt"
    assert result.selected.verifier_confirmed
    assert result.selected.transition.progress.after_open_counterexamples == 0
    assert result.selected.transition.progress.improved
    payload = closure.to_dict()
    assert payload["schema"] == VERIFIER_BACKED_REPAIR_CLOSURE_SCHEMA
    assert VerifierBackedRepairClosure.from_dict(payload) == closure


def test_structurally_admissible_differs_from_verifier_confirmed() -> None:
    source = _source()
    counterexample = _counterexample(source)
    operation = _operation(counterexample.semantic_id)

    structural = FormalReplanner().replan(
        source, counterexample, candidate_repairs=(operation,)
    )
    assert structural.selected is not None
    assert structural.selected.admissible
    assert not structural.selected.verifier_confirmed
    assert structural.selected.transition.progress.after_open_counterexamples == 1

    def _verifier(binding: dict[str, Any]) -> dict[str, Any]:
        return {
            "receipt_id": "receipt:confirmed",
            "counterexample_id": binding["counterexample_id"],
            "repository_tree_id": binding["repository_tree_id"],
            "property_id": binding["property_id"],
            "assumption_ids": list(binding["assumption_ids"]),
            "bound_digest": binding["bound_digest"],
            "tool_id": binding["tool_id"] or "tool:z3",
            "policy_id": binding["policy_id"] or "policy:formal-repair",
            "repaired_plan_id": binding["repaired_plan_id"],
            "freshness": "current",
            "outcome": "verified",
            "available": True,
        }

    confirmed = FormalReplanner(verifier=_verifier).replan(
        source, counterexample, candidate_repairs=(operation,)
    )
    assert confirmed.selected is not None
    assert confirmed.selected.admissible
    assert confirmed.selected.verifier_confirmed
    assert confirmed.selected.transition.progress.after_open_counterexamples == 0
    assert (
        structural.selected.transition.progress.after_open_counterexamples
        != confirmed.selected.transition.progress.after_open_counterexamples
    )


def test_closed_closure_requires_named_receipt() -> None:
    with pytest.raises(Exception, match="fresh matching verifier receipt"):
        VerifierBackedRepairClosure(
            counterexample_id="cx:1",
            status=WitnessClosureStatus.CLOSED,
            open_counterexamples=0,
            structural_addressed=True,
            reason_code="fresh_matching_verifier_receipt",
            verifier_receipt_id="",
        )


def test_bound_digest_is_stable_for_finite_bounds() -> None:
    first = _bound_digest({"portfolio_width": 1, "deadline": 20})
    second = _bound_digest({"deadline": 20, "portfolio_width": 1})
    assert first == second
    assert first != _bound_digest({"portfolio_width": 2})
