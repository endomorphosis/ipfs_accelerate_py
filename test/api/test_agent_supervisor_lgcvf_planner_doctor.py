"""LGCVF-100: semantic discharge in Planner/Doctor admission and fixed points.

Required evidence: admission, completion, fixed-point, second-order, and
oscillation coverage. Missing semantic coverage blocks. Plan ancestry is
preserved.
"""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    ObligationGraphDecision,
    SemanticDischargeEvidence,
    SemanticDischargeReason,
    apply_semantic_discharge,
)
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_live_fixed_point import (
    LiveFixedPointAbortReason,
)
from test.api.test_agent_supervisor_deterministic_doctor_live_fixed_point import (
    _admitted_plan,
    _committed_report,
    _happy_request,
    _live_runner,
    roots,
)


def _evidence(**overrides: object) -> SemanticDischargeEvidence:
    values: dict[str, object] = {
        "discharge_refs": ("discharge:obligation:one",),
        "invalidation_refs": ("invalidate:delta:plan",),
        "covered_obligation_ids": ("obligation:one",),
        "current_tree_id": "tree:candidate",
        "evidence_tree_id": "tree:candidate",
    }
    values.update(overrides)
    return SemanticDischargeEvidence(**values)  # type: ignore[arg-type]


def test_admission_consumes_current_discharge_and_invalidation() -> None:
    ancestry = ("plan:parent", "plan:child")
    decision = apply_semantic_discharge(
        _evidence(),
        required_obligation_ids=("obligation:one",),
        plan_ancestry=ancestry,
    )
    assert decision.admitted
    assert decision.complete
    assert not decision.blocked
    assert decision.plan_ancestry == ancestry
    assert SemanticDischargeReason.ANCESTRY_PRESERVED.value in decision.reason_codes
    assert decision.graph_decision is ObligationGraphDecision.READY


def test_missing_coverage_blocks_admission_and_completion() -> None:
    decision = apply_semantic_discharge(
        _evidence(covered_obligation_ids=()),
        required_obligation_ids=("obligation:one", "obligation:two"),
        plan_ancestry=("plan:parent",),
    )
    assert decision.blocked
    assert not decision.admitted
    assert not decision.complete
    assert decision.missing_coverage == ("obligation:one", "obligation:two")
    assert SemanticDischargeReason.MISSING_COVERAGE.value in decision.reason_codes
    assert decision.plan_ancestry == ("plan:parent",)


def test_unsat_cores_counterexamples_and_validated_interpolants_create_successors() -> None:
    decision = apply_semantic_discharge(
        _evidence(
            unsat_core_refs=("core:narrow",),
            counterexample_refs=("cex:1",),
            interpolant_refs=("interpolant:safe",),
            interpolants_independently_validated=True,
        ),
        required_obligation_ids=("obligation:one",),
    )
    assert decision.admitted
    assert not decision.complete
    kinds = {item.kind for item in decision.successors}
    assert kinds == {"unsat_core", "counterexample", "interpolant"}
    assert all(item.minimal for item in decision.successors)
    assert SemanticDischargeReason.SUCCESSORS_OPEN.value in decision.reason_codes


def test_unvalidated_interpolants_fail_closed() -> None:
    decision = apply_semantic_discharge(
        _evidence(
            interpolant_refs=("interpolant:unvalidated",),
            interpolants_independently_validated=False,
        ),
        required_obligation_ids=("obligation:one",),
    )
    assert decision.blocked
    assert not decision.successors
    assert SemanticDischargeReason.UNVALIDATED_INTERPOLANT.value in decision.reason_codes


def test_stale_discharge_evidence_blocks() -> None:
    decision = apply_semantic_discharge(
        _evidence(evidence_tree_id="tree:stale"),
        required_obligation_ids=("obligation:one",),
    )
    assert decision.blocked
    assert SemanticDischargeReason.STALE_EVIDENCE.value in decision.reason_codes


def test_repeated_successors_are_oscillation() -> None:
    first = apply_semantic_discharge(
        _evidence(unsat_core_refs=("core:same",)),
        required_obligation_ids=("obligation:one",),
    )
    assert first.successors
    second = apply_semantic_discharge(
        _evidence(
            unsat_core_refs=("core:same",),
            prior_successor_fingerprint=first.successor_fingerprint,
        ),
        required_obligation_ids=("obligation:one",),
    )
    assert second.blocked
    assert SemanticDischargeReason.OSCILLATION.value in second.reason_codes
    assert not second.successors


def test_live_fixed_point_blocks_on_missing_semantic_coverage() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    request = _happy_request(
        required_obligation_ids=("obligation:one",),
        semantic_discharge=_evidence(covered_obligation_ids=()),
        plan_ancestry=("plan:parent",),
    )
    outcome = _live_runner().run(plan, report, request)
    assert not outcome.complete
    assert LiveFixedPointAbortReason.MISSING_SEMANTIC_COVERAGE.value in (
        set(outcome.report.reason_codes)
        | set(getattr(outcome, "reason_codes", ()) or ())
    ) or any(
        "missing_semantic_coverage" in str(code)
        for code in outcome.report.reason_codes
    )


def test_live_fixed_point_blocks_on_unvalidated_interpolant() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    request = _happy_request(
        required_obligation_ids=("obligation:one",),
        semantic_discharge=_evidence(
            interpolant_refs=("interpolant:open",),
            interpolants_independently_validated=False,
        ),
    )
    outcome = _live_runner().run(plan, report, request)
    assert not outcome.complete


def test_live_fixed_point_still_completes_without_semantic_gate() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    outcome = _live_runner().run(plan, report, _happy_request())
    assert outcome.complete, outcome.report.reason_codes
