from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.procedure_compiler.certificate import (
    PromotionRequest,
    ProcedureCertificateError,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.metrics import (
    AmortizationReport,
    MetricPopulation,
    MetricReason,
    ProcedureMetrics,
    ProcedurePromotionGate,
    PromotionMetricsError,
    QualifiedBaseline,
    REQUIRED_COST_KINDS,
    SAFETY_GATES,
)


def population(*, eligible: int = 10, successful: int = 10) -> MetricPopulation:
    # For count-valued token/model measurements the observed value is itself
    # the complete declared population.  Ratio metrics pass an explicit
    # eligible population and use a smaller success count.
    if successful > eligible:
        return MetricPopulation(eligible=successful, covered=successful, successful=successful)
    return MetricPopulation(eligible=eligible, covered=eligible, successful=successful)


def passing_metrics(**changes: object) -> ProcedureMetrics:
    baseline = QualifiedBaseline(True, 100, 1_000, 100, 1_000, 100)
    values: dict[str, object] = {
        "safety_violations": {name: 0 for name in SAFETY_GATES},
        "required_postconditions": population(), "validation_retention": population(),
        "boundary_rejection": population(), "proof_coverage": population(), "test_coverage": population(),
        "post_merge_regressions": 0, "baseline_post_merge_regressions": 0,
        "planning_tokens": population(eligible=1, successful=50),
        "model_input_tokens": population(eligible=1, successful=600),
        "remote_model_calls": population(eligible=1, successful=40),
        "retry_tokens": population(eligible=1, successful=300),
        "recurring_without_remote": population(eligible=10, successful=6),
        "deterministic_repair_without_model": population(eligible=10, successful=8),
        "accepted_via_verified_procedure": population(eligible=10, successful=3),
        "human_interventions": 75, "baseline": baseline,
        "held_out_results_present": True, "unsafe_cross_repository_transfers": 0,
        "cost_by_kind": {name: 1 for name in REQUIRED_COST_KINDS},
        "amortization": AmortizationReport(100, 10, 10),
    }
    values.update(changes)
    return ProcedureMetrics(**values)  # type: ignore[arg-type]


def test_every_exact_threshold_passes_and_gate_never_authorizes() -> None:
    result = ProcedurePromotionGate().evaluate(passing_metrics())
    assert result.eligible
    assert result.reasons == (MetricReason.PASS,)
    assert result.grants_promotion is False


@pytest.mark.parametrize("safety_name", SAFETY_GATES)
def test_each_safety_floor_is_non_compensable(safety_name: str) -> None:
    metrics = passing_metrics(safety_violations={name: int(name == safety_name) for name in SAFETY_GATES})
    result = ProcedurePromotionGate().evaluate(metrics)
    assert not result.eligible
    assert MetricReason.SAFETY_FLOOR_FAILED in result.reasons


def test_correctness_transfer_and_denominators_fail_closed() -> None:
    gate = ProcedurePromotionGate()
    partial = replace(passing_metrics(), validation_retention=MetricPopulation(10, 9, 9))
    assert MetricReason.INCOMPLETE_DENOMINATOR in gate.evaluate(partial).reasons
    wrong = replace(passing_metrics(), boundary_rejection=population(successful=9))
    assert MetricReason.CORRECTNESS_FLOOR_FAILED in gate.evaluate(wrong).reasons
    transfer = replace(passing_metrics(), unsafe_cross_repository_transfers=1)
    assert MetricReason.TRANSFER_GATE_FAILED in gate.evaluate(transfer).reasons


def test_baseline_and_complete_cost_ledger_are_required() -> None:
    gate = ProcedurePromotionGate()
    missing = replace(passing_metrics(), baseline=None)
    assert MetricReason.MISSING_QUALIFIED_BASELINE in gate.evaluate(missing).reasons
    with pytest.raises(PromotionMetricsError, match="cost_by_kind"):
        passing_metrics(cost_by_kind={"match": 1})


def test_token_autonomy_and_amortization_gates_use_exact_thresholds() -> None:
    gate = ProcedurePromotionGate()
    token_fail = replace(passing_metrics(), planning_tokens=population(eligible=1, successful=51))
    assert MetricReason.TOKEN_GATE_FAILED in gate.evaluate(token_fail).reasons
    autonomy_fail = replace(passing_metrics(), recurring_without_remote=population(successful=5))
    assert MetricReason.AUTONOMY_GATE_FAILED in gate.evaluate(autonomy_fail).reasons
    amortization_fail = replace(passing_metrics(), amortization=AmortizationReport(101, 10, 10))
    assert MetricReason.AMORTIZATION_FAILED in gate.evaluate(amortization_fail).reasons


def test_request_requires_passing_gate_expected_old_and_exact_rollback() -> None:
    verdict = ProcedurePromotionGate().evaluate(passing_metrics())
    request = PromotionRequest("procedure", "procedure-cid", "certificate-cid", "old-revision", "rollback-revision", verdict)
    assert request.grants_promotion is False
    with pytest.raises(ProcedureCertificateError):
        PromotionRequest("procedure", "procedure-cid", "certificate-cid", "", "", verdict)
    with pytest.raises(ProcedureCertificateError):
        PromotionRequest("procedure", "procedure-cid", "certificate-cid", "old", "rollback", replace(verdict, eligible=False))
