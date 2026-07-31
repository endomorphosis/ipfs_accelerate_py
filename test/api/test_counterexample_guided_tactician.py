"""Verifier-backed CEGIS/CEGAR loop (FVT-G044 / FVT-028).

Acceptance (CounterexampleGuidedProofDevelopment@1):

* each iteration binds prior witness, candidate, repaired tree/goal, exact
  verifier, budget, and result;
* only fresh success closes;
* unchanged witnesses back off;
* repeated failure terminates under policy;
* disagreement / timeout / unavailable / bound change remains open or unknown.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    VERIFIER_BACKED_REPAIR_CLOSURE_SCHEMA,
    WitnessClosureStatus,
    _bound_digest,
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
    property_id: str = "goal transition must remain valid",
    assumption_id: str = "assumption:dep-ready",
    plan_id: str = "plan:base",
    finite_bounds: dict[str, Any] | None = None,
    failure_code: str = "focused-repair-required",
):
    return normalize_counterexample(
        {
            "kind": CounterexampleKind.GENERIC_FAILURE.value,
            "failure": {"code": failure_code},
        },
        kind=CounterexampleKind.GENERIC_FAILURE,
        violated_property=property_id,
        bindings={
            "plan_id": plan_id,
            "task_id": "REF-TARGET",
            "ast_scope_id": "symbol:target",
            "tree_id": tree_id,
            "assumption_id": assumption_id,
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
    tree_id: str | None = None,
    goal_id: str | None = None,
    plan_id: str = "plan:repaired-1",
    addresses: bool = True,
) -> RefinementCandidate:
    return RefinementCandidate(
        candidate_id=candidate_id,
        kind=CandidateKind.INVARIANT,
        goal_id=goal_id or counterexample.violated_property,
        repaired_tree_id=tree_id
        or (
            counterexample.bindings.tree_ids[0]
            if counterexample.bindings.tree_ids
            else "tree:repair-v1"
        ),
        repaired_plan_id=plan_id,
        statement="x >= 0 after init",
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


def test_interface_identity_and_schema_constants() -> None:
    assert (
        COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE
        == "CounterexampleGuidedProofDevelopment@1"
    )
    assert CounterexampleGuidedTactician.interface == (
        COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE
    )
    assert CEGIS_LOOP_RESULT_SCHEMA.endswith("cegis-loop-result@1")
    assert CEGIS_ITERATION_BINDING_SCHEMA.endswith("cegis-iteration-binding@1")


def test_each_iteration_binds_witness_candidate_tree_goal_verifier_budget_result() -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx)

    def refine(witness, context):
        del witness, context
        return (candidate,)

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        return _matching_receipt(binding, outcome="still_violated")

    result = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify,
        budget=CegisBudget(max_iterations=2, max_identical_failures=2),
    ).run(cx, previous_witness_id=None)

    assert result.iterations
    binding = result.iterations[0].binding
    assert binding.prior_witness_id == cx.semantic_id
    assert binding.candidate_id == candidate.candidate_id
    assert binding.repaired_tree_id == candidate.repaired_tree_id
    assert binding.repaired_goal_id == candidate.goal_id
    assert binding.exact_verifier_id == "tool:z3"
    assert isinstance(binding.budget, CegisBudget)
    assert binding.budget.bound_digest == _bound_digest(cx.finite_bounds)
    assert binding.result_status in {
        IterationOutcome.STILL_OPEN,
        IterationOutcome.UNKNOWN,
        IterationOutcome.CLOSED,
    }
    payload = binding.to_dict()
    assert payload["schema"] == CEGIS_ITERATION_BINDING_SCHEMA
    assert payload["prior_witness_id"]
    assert payload["candidate_id"]
    assert payload["repaired_tree_id"]
    assert payload["repaired_goal_id"]
    assert payload["exact_verifier_id"]
    assert "budget" in payload
    assert "result_status" in payload
    assert IterationBinding.from_dict(payload).prior_witness_id == binding.prior_witness_id


def test_only_fresh_matching_success_closes() -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx)

    def refine(witness, context):
        del witness, context
        return (candidate,)

    # Structural path without a verifier never closes.
    structural = CounterexampleGuidedTactician(
        refine=refine,
        verify=None,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert not structural.closed
    assert structural.open_counterexamples == 1
    assert structural.stop_reason is not CegisStopReason.CLOSED
    if structural.closure is not None:
        assert structural.closure.status is not WitnessClosureStatus.CLOSED
        assert structural.closure.open_counterexamples == 1

    def verify_ok(binding: dict[str, Any]) -> dict[str, Any]:
        return _matching_receipt(binding, outcome="verified")

    closed = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify_ok,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert closed.closed
    assert closed.open_counterexamples == 0
    assert closed.stop_reason is CegisStopReason.CLOSED
    assert closed.closure is not None
    assert closed.closure.closed
    assert closed.closure.status is WitnessClosureStatus.CLOSED
    assert closed.closure.verifier_receipt_id
    assert closed.closure.reason_code == "fresh_matching_verifier_receipt"
    assert closed.closure.to_dict()["schema"] == VERIFIER_BACKED_REPAIR_CLOSURE_SCHEMA
    assert closed.selected_candidate is not None
    assert closed.selected_candidate.candidate_id == candidate.candidate_id
    assert closed.iterations[0].binding.closed
    assert closed.iterations[0].binding.verifier_receipt_id


def test_stale_or_mismatched_receipt_does_not_close() -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx)

    def refine(witness, context):
        del witness, context
        return (candidate,)

    def stale(binding: dict[str, Any]) -> dict[str, Any]:
        payload = _matching_receipt(binding)
        payload["freshness"] = "stale"
        return payload

    result = CounterexampleGuidedTactician(
        refine=refine,
        verify=stale,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert not result.closed
    assert result.open_counterexamples == 1
    assert result.closure is not None
    assert result.closure.reason_code == "stale_receipt"

    def wrong_tool(binding: dict[str, Any]) -> dict[str, Any]:
        payload = _matching_receipt(binding)
        payload["tool_id"] = "tool:cvc5"
        return payload

    mismatched = CounterexampleGuidedTactician(
        refine=refine,
        verify=wrong_tool,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert not mismatched.closed
    assert mismatched.open_counterexamples == 1
    assert mismatched.closure is not None
    assert "binding_mismatch" in mismatched.closure.reason_code
    assert "tool" in mismatched.closure.reason_code


def test_unchanged_witness_backs_off() -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx)
    refine_calls = {"n": 0}

    def refine(witness, context):
        refine_calls["n"] += 1
        del witness, context
        return (candidate,)

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        return _matching_receipt(binding, outcome="still_violated")

    tactician = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify,
        budget=CegisBudget(
            max_iterations=4,
            max_identical_failures=3,
            base_backoff_seconds=2,
            max_backoff_seconds=100,
        ),
    )
    # First run attempts a candidate and leaves the witness open.
    first = tactician.run(cx, previous_witness_id=None)
    assert not first.closed
    assert first.iterations
    assert first.iterations[0].binding.candidate_id == candidate.candidate_id
    assert refine_calls["n"] == 1

    # Re-entering with the same witness identity must back off without refine.
    second = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify,
        budget=CegisBudget(
            max_iterations=4,
            max_identical_failures=3,
            base_backoff_seconds=2,
            max_backoff_seconds=100,
        ),
    ).run(cx, previous_witness_id=cx.semantic_id)
    assert second.stop_reason is CegisStopReason.UNCHANGED_WITNESS_BACKOFF
    assert not second.closed
    assert second.open_counterexamples == 1
    assert second.iterations
    backoff = second.iterations[0]
    assert backoff.binding.result_status is IterationOutcome.BACKED_OFF
    assert backoff.binding.reason_code == "unchanged_witness_backoff"
    assert backoff.backoff_seconds == 2
    assert refine_calls["n"] == 1  # refine not invoked on backoff path


def test_repeated_identical_failure_terminates_under_policy() -> None:
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
        # identical_failure_count already at threshold-1 via budget usage
        budget=CegisBudget(
            max_iterations=4,
            max_identical_failures=2,
            identical_failure_count=1,
        ),
    )
    assert result.stop_reason is CegisStopReason.IDENTICAL_FAILURE_TERMINATED
    assert not result.closed
    assert result.open_counterexamples == 1
    assert result.iterations
    assert result.iterations[0].binding.reason_code == "identical_failure_terminated"
    assert result.reason_code == "identical_failure_terminated"


@pytest.mark.parametrize(
    ("outcome", "stop_reason", "closure_token"),
    [
        ("timeout", CegisStopReason.VERIFIER_TIMEOUT, "verifier_timeout"),
        ("disagreement", CegisStopReason.VERIFIER_DISAGREEMENT, "verifier_disagreement"),
        ("unavailable", CegisStopReason.VERIFIER_UNAVAILABLE, "verifier_unavailable"),
    ],
)
def test_timeout_disagreement_unavailable_remain_open_or_unknown(
    outcome: str,
    stop_reason: CegisStopReason,
    closure_token: str,
) -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx)

    def refine(witness, context):
        del witness, context
        return (candidate,)

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        payload = _matching_receipt(binding, outcome=outcome)
        if outcome == "unavailable":
            payload["available"] = False
        return payload

    result = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert not result.closed
    assert result.open_counterexamples == 1
    assert result.stop_reason is stop_reason
    assert result.closure is not None
    assert result.closure.open_counterexamples == 1
    assert result.closure.status in {
        WitnessClosureStatus.OPEN,
        WitnessClosureStatus.UNKNOWN,
    }
    assert closure_token in result.closure.reason_code
    assert result.iterations[0].binding.closure_status in {
        WitnessClosureStatus.OPEN,
        WitnessClosureStatus.UNKNOWN,
    }


def test_bound_change_leaves_witness_open() -> None:
    cx = _counterexample(finite_bounds={"portfolio_width": 1, "deadline": 20})
    candidate = _candidate_for(cx)

    def refine(witness, context):
        del witness, context
        return (candidate,)

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        payload = _matching_receipt(binding)
        payload["bound_digest"] = _bound_digest({"portfolio_width": 99})
        return payload

    result = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify,
        budget=CegisBudget(
            max_iterations=1,
            finite_bounds=dict(cx.finite_bounds),
        ),
    ).run(cx)
    assert not result.closed
    assert result.open_counterexamples == 1
    assert result.stop_reason is CegisStopReason.BOUND_CHANGED
    assert result.closure is not None
    assert "bound" in result.closure.reason_code


def test_independent_validation_rejects_before_verifier() -> None:
    cx = _counterexample()
    bad = _candidate_for(cx, candidate_id="candidate:bad", addresses=False)
    verify_calls = {"n": 0}

    def refine(witness, context):
        del witness, context
        return (bad,)

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        verify_calls["n"] += 1
        return _matching_receipt(binding)

    result = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert not result.closed
    assert verify_calls["n"] == 0
    assert result.stop_reason is CegisStopReason.NO_ADMISSIBLE_CANDIDATE
    assert result.iterations[0].binding.result_status is (
        IterationOutcome.CANDIDATE_REJECTED
    )


def test_exact_originating_verifier_is_bound_into_request() -> None:
    cx = _counterexample(provider_id="tool:origin-cvc5")
    candidate = _candidate_for(cx)
    seen: dict[str, Any] = {}

    def refine(witness, context):
        del witness, context
        return (candidate,)

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        seen.update(binding)
        return _matching_receipt(binding)

    result = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert result.closed
    assert result.exact_verifier_id == "tool:origin-cvc5"
    assert seen["tool_id"] == "tool:origin-cvc5"
    assert seen["counterexample_id"] == cx.semantic_id
    assert seen["repository_tree_id"] == candidate.repaired_tree_id
    assert seen["property_id"] == cx.violated_property
    assert seen["bound_digest"] == _bound_digest(cx.finite_bounds)


def test_cancelled_before_start_remains_open() -> None:
    cx = _counterexample()
    result = CounterexampleGuidedTactician().run(cx, cancelled=True)
    assert result.stop_reason is CegisStopReason.CANCELLED
    assert not result.closed
    assert result.open_counterexamples == 1
    assert result.iteration_count == 0


def test_verifier_exception_treated_as_unavailable() -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx)

    def refine(witness, context):
        del witness, context
        return (candidate,)

    def verify(_binding: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("backend offline")

    result = CounterexampleGuidedTactician(
        refine=refine,
        verify=verify,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert not result.closed
    assert result.open_counterexamples == 1
    assert result.stop_reason is CegisStopReason.VERIFIER_UNAVAILABLE
    assert result.closure is not None
    assert result.closure.status is WitnessClosureStatus.UNKNOWN
    assert result.closure.reason_code == "verifier_unavailable"


def test_functional_entry_point_and_round_trip() -> None:
    cx = _counterexample()
    candidate = _candidate_for(cx, candidate_id="candidate:rt")

    def refine(witness, context):
        del witness, context
        return (candidate,)

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        return _matching_receipt(binding)

    result = run_counterexample_guided_loop(
        cx,
        refine=refine,
        verify=verify,
        budget={"max_iterations": 2},
    )
    assert result.closed
    payload = result.to_dict()
    assert payload["schema"] == CEGIS_LOOP_RESULT_SCHEMA
    assert payload["interface"] == COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE
    restored = CegisLoopResult.from_dict(payload)
    assert restored.closed == result.closed
    assert restored.stop_reason is result.stop_reason
    assert restored.initial_witness_id == result.initial_witness_id
    assert restored.iteration_count == result.iteration_count
    assert restored.selected_candidate is not None
    assert restored.selected_candidate.candidate_id == candidate.candidate_id


def test_replay_provider_failure_is_honest_open() -> None:
    cx = _counterexample()

    def replay(witness, context):
        del witness, context
        raise ValueError("replay oracle offline")

    result = CounterexampleGuidedTactician(
        replay=replay,
        budget=CegisBudget(max_iterations=1),
    ).run(cx)
    assert not result.closed
    assert result.open_counterexamples == 1
    assert result.stop_reason is CegisStopReason.OPEN_CONTINUED_FAILURE
    assert result.reason_code.startswith("replay_failed:")


def test_default_refine_and_validate_path_without_custom_providers() -> None:
    cx = _counterexample()

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        return _matching_receipt(binding)

    result = CounterexampleGuidedTactician(
        verify=verify,
        budget=CegisBudget(max_iterations=1),
    ).run(cx, repository_tree_id="tree:repair-v1", goal_id="G-close")
    assert result.closed
    assert result.selected_candidate is not None
    assert result.selected_candidate.validation_status is (
        CandidateValidationStatus.VALID
    )
    assert result.selected_candidate.kind in set(CandidateKind)


def test_budget_contract_rejects_inverted_backoff() -> None:
    with pytest.raises(CegisValidationError, match="base_backoff_seconds"):
        CegisBudget(base_backoff_seconds=10, max_backoff_seconds=1)


def test_closed_result_requires_zero_open_and_named_closure() -> None:
    cx = _counterexample()
    with pytest.raises(CegisValidationError):
        CegisLoopResult(
            stop_reason=CegisStopReason.CLOSED,
            initial_witness_id=cx.semantic_id,
            final_witness_id=cx.semantic_id,
            exact_verifier_id="tool:z3",
            property_id=cx.violated_property,
            iterations=(),
            budget=CegisBudget(max_iterations=1),
            open_counterexamples=0,
            closed=True,
            closure=None,
        )
