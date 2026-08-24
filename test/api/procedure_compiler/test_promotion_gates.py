from ipfs_accelerate_py.agent_supervisor.procedure_compiler.metrics import (
    AmortizationReport,
    ProcedureMetrics,
    ProcedurePromotionGate,
    PromotionGateReason,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.certificate import (
    CertificateAdmission, CertificateReasonCode, CertificateVerificationStatus,
)


def certificate_admission() -> CertificateAdmission:
    return CertificateAdmission(
        status=CertificateVerificationStatus.ACCEPTED, reason_code=CertificateReasonCode.ACCEPTED,
        certificate_cid="certificate-cid", issuer="independent-issuer", accepted=True,
        usable=True, grants_authority=False, grants_promotion=False,
    )


def qualified_metrics(**changes):
    values = dict(
        planning_token_samples=(100, 100, 100), total_model_input_tokens=1000,
        remote_model_calls=10, retry_tokens=100, eligible_recurring_tasks=10,
        recurring_tasks_without_remote_model=8, deterministic_repair_tasks=10,
        deterministic_repairs_without_model=9, accepted_benchmark_work=10,
        accepted_benchmark_work_via_verified_procedures=4, eligible_human_interventions=10,
        human_interventions=5, required_postconditions=10,
        satisfied_required_postconditions=10, required_validations=10, retained_validations=10,
        known_boundary_counterexamples=10, rejected_boundary_counterexamples=10,
        proof_coverage=10, test_coverage=10, held_out_transfer_results=1,
    )
    values.update(changes)
    return ProcedureMetrics(**values)


def test_promotion_passes_only_all_exact_thresholds() -> None:
    report = ProcedurePromotionGate().evaluate(
        qualified_metrics(), baseline=qualified_metrics(
            planning_token_samples=(200, 200, 200), total_model_input_tokens=2000,
            remote_model_calls=25, retry_tokens=400, human_interventions=8),
        baseline_qualified=True, amortization=AmortizationReport(100, 20, 5),
        certificate_admission=certificate_admission(),
        expected_old_revision_id="prior-revision", rollback_target_revision_id="exact-revision",
    )
    assert report.accepted
    assert report.reasons == (PromotionGateReason.ACCEPTED,)
    assert not report.promotion_authorized


def test_safety_and_correctness_cannot_be_compensated() -> None:
    report = ProcedurePromotionGate().evaluate(
        qualified_metrics(unauthorized_effects=1, retained_validations=9),
        baseline=qualified_metrics(planning_token_samples=(200,), total_model_input_tokens=2000,
                                 remote_model_calls=25, retry_tokens=400, human_interventions=8),
        baseline_qualified=True, amortization=AmortizationReport(1, 1, 1),
        certificate_admission=certificate_admission(),
        expected_old_revision_id="prior", rollback_target_revision_id="rollback",
    )
    assert not report.accepted
    assert PromotionGateReason.SAFETY_FAILURE in report.reasons
    assert PromotionGateReason.CORRECTNESS_FAILURE in report.reasons


def test_missing_denominators_or_baseline_fail_closed() -> None:
    report = ProcedurePromotionGate().evaluate(
        ProcedureMetrics(), baseline=None, baseline_qualified=False,
        amortization=AmortizationReport(1, 1, 1), expected_old_revision_id="prior",
        rollback_target_revision_id="rollback", certificate_admission=certificate_admission(),
    )
    assert not report.accepted
    assert PromotionGateReason.INCOMPLETE_DENOMINATOR in report.reasons
    assert PromotionGateReason.MISSING_QUALIFIED_BASELINE in report.reasons


def test_exact_token_thresholds_and_cas_rollback_are_enforced() -> None:
    baseline = qualified_metrics(planning_token_samples=(200,), total_model_input_tokens=1000,
                               remote_model_calls=10, retry_tokens=100, human_interventions=8)
    report = ProcedurePromotionGate().evaluate(
        qualified_metrics(total_model_input_tokens=601), baseline=baseline,
        baseline_qualified=True, amortization=AmortizationReport(1, 1, 1), certificate_admission=certificate_admission(),
    )
    assert PromotionGateReason.TOKEN_EFFICIENCY_FAILURE in report.reasons
    assert PromotionGateReason.CAS_OR_ROLLBACK_MISSING in report.reasons


def test_transfer_and_break_even_require_complete_evidence() -> None:
    report = ProcedurePromotionGate().evaluate(
        qualified_metrics(held_out_transfer_results=0, transfer_assumption_mismatches=1,
                          typed_transfer_refusals=0),
        baseline=qualified_metrics(planning_token_samples=(200,), total_model_input_tokens=2000,
                                 remote_model_calls=25, retry_tokens=400, human_interventions=8),
        baseline_qualified=True, amortization=AmortizationReport(100, 20, 4),
        certificate_admission=certificate_admission(),
        expected_old_revision_id="prior", rollback_target_revision_id="rollback",
    )
    assert PromotionGateReason.TRANSFER_FAILURE in report.reasons
    assert PromotionGateReason.AMORTIZATION_FAILURE in report.reasons


def test_missing_certificate_is_a_release_failure() -> None:
    report = ProcedurePromotionGate().evaluate(
        qualified_metrics(), baseline=qualified_metrics(planning_token_samples=(200,),
            total_model_input_tokens=2000, remote_model_calls=25, retry_tokens=400,
            human_interventions=8), baseline_qualified=True,
        amortization=AmortizationReport(1, 1, 1), expected_old_revision_id="prior",
        rollback_target_revision_id="rollback",
    )
    assert PromotionGateReason.CERTIFICATE_FAILURE in report.reasons
