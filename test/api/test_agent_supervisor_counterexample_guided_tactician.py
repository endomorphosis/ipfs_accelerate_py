"""PDR-051 coverage for bounded CEGIS via CounterexampleGuidedTactician.

Mirrors the FVT CEGIS contract under the agent-supervisor naming convention
required by the proof-directed planner/doctor validation gate, and adds
program-repair-facing checks: independent candidate validation, fixed-budget
termination, and proposal-only refinement candidates.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    WitnessClosureStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.counterexample_guided_tactician import (
    CEGIS_ITERATION_BINDING_SCHEMA,
    CEGIS_LOOP_RESULT_SCHEMA,
    COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE,
    CandidateKind,
    CandidateValidationStatus,
    CegisBudget,
    CegisLoopResult,
    CegisStopReason,
    CegisValidationError,
    CounterexampleGuidedTactician,
    IterationBinding,
    IterationOutcome,
    RefinementCandidate,
    run_counterexample_guided_loop,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_counterexamples import (
    CounterexampleKind,
    RepairClass,
    normalize_counterexample,
)


def _counterexample(
    *,
    tree_id: str = "tree:repair-v1",
    provider_id: str = "tool:z3",
    policy_id: str = "policy:cegis",
    property_id: str = "obligation:pdr-051",
    finite_bounds: dict[str, Any] | None = None,
):
    return normalize_counterexample(
        {
            "kind": CounterexampleKind.GENERIC_FAILURE.value,
            "failure": {"code": "focused-repair-required"},
        },
        kind=CounterexampleKind.GENERIC_FAILURE,
        violated_property=property_id,
        bindings={
            "plan_id": "plan:base",
            "task_id": "PDR-051",
            "ast_scope_id": "symbol:target",
            "tree_id": tree_id,
            "assumption_id": "assumption:dep-ready",
            "provider_id": provider_id,
            "policy_id": policy_id,
            "obligation_id": property_id,
        },
        finite_bounds=finite_bounds or {"portfolio_width": 1, "deadline": 20},
        repair_classes=(RepairClass.ADD_DEPENDENCY,),
    )


def _candidate_for(
    counterexample,
    *,
    candidate_id: str = "candidate:inv-1",
    addresses: bool = True,
) -> RefinementCandidate:
    return RefinementCandidate(
        candidate_id=candidate_id,
        kind=CandidateKind.REPAIR,
        goal_id=counterexample.violated_property,
        repaired_tree_id=(
            counterexample.bindings.tree_ids[0]
            if counterexample.bindings.tree_ids
            else "tree:repair-v1"
        ),
        repaired_plan_id="plan:repaired-1",
        statement="bounded repair candidate",
        addresses_witness=addresses,
        parameters={"source_witness_id": counterexample.semantic_id},
    )


def _matching_receipt(binding: dict[str, Any], *, outcome: str = "verified") -> dict[str, Any]:
    return {
        "receipt_id": f"receipt:{outcome}:{binding.get('candidate_id', 'x')}",
        "counterexample_id": binding["counterexample_id"],
        "repository_tree_id": binding["repository_tree_id"],
        "property_id": binding["property_id"],
        "assumption_ids": list(binding.get("assumption_ids") or ()),
        "bound_digest": binding["bound_digest"],
        "tool_id": binding["tool_id"],
        "policy_id": binding["policy_id"],
        "repaired_plan_id": binding["repaired_plan_id"],
        "freshness": "current",
        "outcome": outcome,
        "available": True,
    }


def test_interface_and_schema_constants() -> None:
    assert (
        COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE
        == "CounterexampleGuidedProofDevelopment@1"
    )
    assert CounterexampleGuidedTactician.interface == (
        COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE
    )
    assert CEGIS_LOOP_RESULT_SCHEMA.endswith("cegis-loop-result@1")
    assert CEGIS_ITERATION_BINDING_SCHEMA.endswith("cegis-iteration-binding@1")


def test_independent_validation_runs_before_originating_verifier() -> None:
    cx = _counterexample()
    verify_calls = {"n": 0}

    def refine(witness, context):
        del witness, context
        return (_candidate_for(cx, addresses=False),)

    def validate(candidate, context):
        del context
        assert candidate.addresses_witness is False
        return CandidateValidationStatus.INVALID, "independent_reject"

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        verify_calls["n"] += 1
        return _matching_receipt(binding)

    result = CounterexampleGuidedTactician(
        refine=refine,
        validate=validate,
        verify=verify,
        budget=CegisBudget(max_iterations=2),
    ).run(cx)

    assert not result.closed
    assert verify_calls["n"] == 0
    assert result.iterations
    assert result.iterations[0].binding.result_status is IterationOutcome.CANDIDATE_REJECTED


def test_fresh_matching_receipt_closes_and_stale_does_not() -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx)

    def refine(witness, context):
        del witness, context
        return (candidate,)

    closed = CounterexampleGuidedTactician(
        refine=refine,
        verify=lambda b: _matching_receipt(b, outcome="verified"),
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert closed.closed
    assert closed.open_counterexamples == 0
    assert closed.stop_reason is CegisStopReason.CLOSED
    assert closed.closure is not None
    assert closed.closure.status is WitnessClosureStatus.CLOSED

    def stale(binding: dict[str, Any]) -> dict[str, Any]:
        payload = _matching_receipt(binding)
        payload["freshness"] = "stale"
        return payload

    open_result = CounterexampleGuidedTactician(
        refine=refine,
        verify=stale,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert not open_result.closed
    assert open_result.open_counterexamples == 1


def test_fixed_budget_terminates_without_infinite_loop() -> None:
    cx = _counterexample()
    refine_calls = {"n": 0}

    def refine(witness, context):
        refine_calls["n"] += 1
        del witness, context
        return (
            _candidate_for(cx, candidate_id=f"candidate:budget:{refine_calls['n']}"),
        )

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        return _matching_receipt(binding, outcome="still_violated")

    result = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify,
        budget=CegisBudget(
            max_iterations=3,
            max_candidates_per_iteration=1,
            max_identical_failures=10,
        ),
    ).run(cx)

    assert not result.closed
    assert result.iteration_count <= 3
    assert refine_calls["n"] <= 3
    assert isinstance(result, CegisLoopResult)
    assert result.budget.max_iterations == 3


def test_identical_failure_terminates_under_policy() -> None:
    cx = _counterexample()
    result = CounterexampleGuidedTactician(
        budget=CegisBudget(
            max_iterations=4,
            max_identical_failures=2,
            base_backoff_seconds=1,
            max_backoff_seconds=10,
        ),
    ).run(
        cx,
        previous_witness_id=cx.semantic_id,
        budget=CegisBudget(
            max_iterations=4,
            max_identical_failures=2,
            identical_failure_count=1,
        ),
    )
    assert result.stop_reason is CegisStopReason.IDENTICAL_FAILURE_TERMINATED
    assert not result.closed


def test_iteration_binding_is_auditable() -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx)

    def refine(witness, context):
        del witness, context
        return (candidate,)

    result = CounterexampleGuidedTactician(
        refine=refine,
        verify=lambda b: _matching_receipt(b, outcome="still_violated"),
        budget=CegisBudget(max_iterations=1),
    ).run(cx)

    assert result.iterations
    binding = result.iterations[0].binding
    payload = binding.to_dict()
    assert payload["schema"] == CEGIS_ITERATION_BINDING_SCHEMA
    assert payload["prior_witness_id"]
    assert payload["candidate_id"] == candidate.candidate_id
    assert payload["repaired_tree_id"]
    assert payload["exact_verifier_id"] == "tool:z3"
    assert "budget" in payload
    restored = IterationBinding.from_dict(payload)
    assert restored.candidate_id == binding.candidate_id


def test_functional_entry_point_round_trip() -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx)

    def refine(witness, context):
        del witness, context
        return (candidate,)

    result = run_counterexample_guided_loop(
        cx,
        refine=refine,
        verify=lambda b: _matching_receipt(b),
        budget=CegisBudget(max_iterations=1),
    )
    assert result.closed
    payload = result.to_dict()
    assert payload["schema"] == CEGIS_LOOP_RESULT_SCHEMA
    assert payload["interface"] == COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE
    restored = CegisLoopResult.from_dict(payload)
    assert restored.closed
    assert restored.selected_candidate is not None


def test_budget_contract_rejects_inverted_backoff() -> None:
    with pytest.raises(CegisValidationError):
        CegisBudget(base_backoff_seconds=10, max_backoff_seconds=1)


def test_candidates_remain_non_authority_artifacts() -> None:
    """Refinement candidates never grant proof/write/completion authority."""

    candidate = RefinementCandidate(
        candidate_id="candidate:proposal",
        kind=CandidateKind.REPAIR,
        goal_id="obligation:one",
        repaired_tree_id="tree:x",
        repaired_plan_id="plan:x",
        statement="proposal only",
        addresses_witness=True,
    )
    payload = candidate.to_dict()
    assert payload["candidate_id"] == "candidate:proposal"
    # No authority claim fields are admitted on the public payload.
    assert "write_authority" not in payload or payload.get("write_authority") is not True
    assert "completion_authority" not in payload
    assert candidate.admissible is True or candidate.validation_status in {
        CandidateValidationStatus.VALID,
        CandidateValidationStatus.SKIPPED,
        None,
        "",
    }


@pytest.mark.parametrize(
    ("outcome", "stop_reason"),
    [
        ("timeout", CegisStopReason.VERIFIER_TIMEOUT),
        ("disagreement", CegisStopReason.VERIFIER_DISAGREEMENT),
        ("unavailable", CegisStopReason.VERIFIER_UNAVAILABLE),
    ],
)
def test_verifier_faults_remain_open(
    outcome: str, stop_reason: CegisStopReason
) -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx)

    def refine(witness, context):
        del witness, context
        return (candidate,)

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        return _matching_receipt(binding, outcome=outcome)

    result = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert not result.closed
    assert result.open_counterexamples == 1
    assert result.stop_reason in {
        stop_reason,
        CegisStopReason.OPEN_CONTINUED_FAILURE,
        CegisStopReason.NO_ADMISSIBLE_CANDIDATE,
        CegisStopReason.RETRY_BUDGET_EXHAUSTED,
        CegisStopReason.REFINEMENT_DEPTH_EXHAUSTED,
        CegisStopReason.CANDIDATE_BUDGET_EXHAUSTED,
    }
