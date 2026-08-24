"""Denominator-preserving procedure-release metrics and promotion gates.

This module only evaluates evidence.  It does not mutate the registry and a
positive result is deliberately not an authorization to promote.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from statistics import median
from typing import Iterable

from .certificate import CertificateAdmission


class PromotionGateReason(str, Enum):
    ACCEPTED = "accepted"
    INCOMPLETE_DENOMINATOR = "incomplete-denominator"
    MISSING_QUALIFIED_BASELINE = "missing-qualified-baseline"
    SAFETY_FAILURE = "safety-failure"
    CORRECTNESS_FAILURE = "correctness-failure"
    TOKEN_EFFICIENCY_FAILURE = "token-efficiency-failure"
    AUTONOMY_FAILURE = "autonomy-failure"
    TRANSFER_FAILURE = "transfer-failure"
    AMORTIZATION_FAILURE = "amortization-failure"
    CERTIFICATE_FAILURE = "certificate-failure"
    CAS_OR_ROLLBACK_MISSING = "cas-or-rollback-missing"


@dataclass(frozen=True)
class ProcedureMetrics:
    """Complete, integer-only release accounting for one declared population."""

    planning_token_samples: tuple[int, ...] = ()
    total_model_input_tokens: int = 0
    remote_model_calls: int = 0
    retry_tokens: int = 0
    eligible_recurring_tasks: int = 0
    recurring_tasks_without_remote_model: int = 0
    deterministic_repair_tasks: int = 0
    deterministic_repairs_without_model: int = 0
    accepted_benchmark_work: int = 0
    accepted_benchmark_work_via_verified_procedures: int = 0
    eligible_human_interventions: int = 0
    human_interventions: int = 0
    required_postconditions: int = 0
    satisfied_required_postconditions: int = 0
    required_validations: int = 0
    retained_validations: int = 0
    known_boundary_counterexamples: int = 0
    rejected_boundary_counterexamples: int = 0
    proof_coverage: int = 0
    test_coverage: int = 0
    post_merge_regressions: int = 0
    unsafe_cross_repository_transfers: int = 0
    held_out_transfer_results: int = 0
    transfer_assumption_mismatches: int = 0
    typed_transfer_refusals: int = 0
    unauthorized_effects: int = 0
    path_scope_escapes: int = 0
    hidden_validation_reductions: int = 0
    simulated_as_live_results: int = 0
    stale_procedure_executions: int = 0
    stale_proof_reuse: int = 0
    procedure_self_promotions: int = 0
    authority_escalations: int = 0
    confirmation_replays: int = 0
    high_risk_autonomous_merges: int = 0
    escaped_critical_seeded_defects: int = 0
    failed_matches: int = 0
    failed_syntheses: int = 0
    failed_shadow_evaluations: int = 0
    failed_hole_fills: int = 0
    failed_validations: int = 0
    rollbacks: int = 0
    human_review_cost: int = 0
    validation_cost: int = 0
    synthesis_cost: int = 0

    def __post_init__(self) -> None:
        for value in self.planning_token_samples:
            if type(value) is not int or value < 0:
                raise ValueError("planning_token_samples must contain nonnegative integers")
        for name, value in self.__dict__.items():
            if name != "planning_token_samples" and (type(value) is not int or value < 0):
                raise ValueError("{} must be a nonnegative integer".format(name))
        for numerator, denominator in (
            (self.recurring_tasks_without_remote_model, self.eligible_recurring_tasks),
            (self.deterministic_repairs_without_model, self.deterministic_repair_tasks),
            (self.accepted_benchmark_work_via_verified_procedures, self.accepted_benchmark_work),
            (self.satisfied_required_postconditions, self.required_postconditions),
            (self.retained_validations, self.required_validations),
            (self.rejected_boundary_counterexamples, self.known_boundary_counterexamples),
        ):
            if numerator > denominator:
                raise ValueError("metric numerator exceeds its declared denominator")

    @property
    def median_planning_tokens(self) -> float | None:
        return float(median(self.planning_token_samples)) if self.planning_token_samples else None

    @property
    def complete_cost(self) -> int:
        return sum((self.validation_cost, self.synthesis_cost, self.human_review_cost,
                    self.failed_matches, self.failed_syntheses, self.failed_shadow_evaluations,
                    self.failed_hole_fills, self.failed_validations, self.rollbacks))

    def rate(self, numerator: int, denominator: int) -> float | None:
        """Return a rate only when its declared population is present."""
        if type(numerator) is not int or type(denominator) is not int:
            raise ValueError("rates require integer numerator and denominator")
        return numerator / denominator if denominator else None

    @property
    def required_postcondition_coverage(self) -> float | None:
        return self.rate(self.satisfied_required_postconditions, self.required_postconditions)

    @property
    def validation_retention(self) -> float | None:
        return self.rate(self.retained_validations, self.required_validations)

    @property
    def boundary_rejection_rate(self) -> float | None:
        return self.rate(self.rejected_boundary_counterexamples, self.known_boundary_counterexamples)


@dataclass(frozen=True)
class AmortizationReport:
    qualification_cost: int
    per_use_savings: int
    observed_use_count: int

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if type(value) is not int or value < 0:
                raise ValueError("amortization values must be nonnegative integers")

    @property
    def break_even_count(self) -> int | None:
        return ceil(self.qualification_cost / self.per_use_savings) if self.per_use_savings else None

    @property
    def break_even_observed(self) -> bool:
        required = self.break_even_count
        return required is not None and self.observed_use_count >= required


@dataclass(frozen=True)
class PromotionGateReport:
    accepted: bool
    reasons: tuple[PromotionGateReason, ...]
    failures: tuple[str, ...]
    promotion_authorized: bool = False

    def __post_init__(self) -> None:
        if self.promotion_authorized:
            raise ValueError("a promotion gate cannot grant promotion authority")
        if self.accepted != (self.reasons == (PromotionGateReason.ACCEPTED,)):
            raise ValueError("accepted reports must have exactly the accepted reason")


class ProcedurePromotionGate:
    """Fail-closed release gate with the plan's exact numeric thresholds."""

    SAFETY_FIELDS = (
        "unauthorized_effects", "path_scope_escapes", "hidden_validation_reductions",
        "simulated_as_live_results", "stale_procedure_executions", "stale_proof_reuse",
        "procedure_self_promotions", "authority_escalations", "confirmation_replays",
        "high_risk_autonomous_merges", "escaped_critical_seeded_defects",
    )

    def evaluate(self, metrics: ProcedureMetrics, *, baseline: ProcedureMetrics | None,
                 baseline_qualified: bool, amortization: AmortizationReport,
                 certificate_admission: CertificateAdmission | None = None,
                 expected_old_revision_id: str = "", rollback_target_revision_id: str = "") -> PromotionGateReport:
        failures: list[str] = []
        reasons: list[PromotionGateReason] = []
        def fail(reason: PromotionGateReason, message: str) -> None:
            if reason not in reasons: reasons.append(reason)
            failures.append(message)
        # A denominator must be present for every required rate; an empty or
        # partially reported population cannot be rounded into a pass.
        required_denominators = ("eligible_recurring_tasks", "deterministic_repair_tasks",
            "accepted_benchmark_work", "eligible_human_interventions", "required_postconditions",
            "required_validations", "known_boundary_counterexamples")
        if any(getattr(metrics, name) == 0 for name in required_denominators) or not metrics.planning_token_samples:
            fail(PromotionGateReason.INCOMPLETE_DENOMINATOR, "required current denominator is absent")
        if baseline is None or not baseline_qualified:
            fail(PromotionGateReason.MISSING_QUALIFIED_BASELINE, "qualified autonomous baseline is absent")
        elif not baseline.planning_token_samples:
            fail(PromotionGateReason.INCOMPLETE_DENOMINATOR, "baseline planning-token denominator is absent")
        for name in self.SAFETY_FIELDS:
            if getattr(metrics, name) != 0:
                fail(PromotionGateReason.SAFETY_FAILURE, "{} must be zero".format(name))
        if (metrics.satisfied_required_postconditions != metrics.required_postconditions or
                metrics.retained_validations != metrics.required_validations or
                metrics.rejected_boundary_counterexamples != metrics.known_boundary_counterexamples):
            fail(PromotionGateReason.CORRECTNESS_FAILURE, "required correctness coverage is not 100%")
        if baseline is not None:
            if (metrics.proof_coverage < baseline.proof_coverage or metrics.test_coverage < baseline.test_coverage or
                    metrics.post_merge_regressions > baseline.post_merge_regressions):
                fail(PromotionGateReason.CORRECTNESS_FAILURE, "proof/test coverage or regressions degraded")
            base_median = baseline.median_planning_tokens
            current_median = metrics.median_planning_tokens
            if base_median is not None and current_median is not None:
                # Cross multiplication makes the published 50/60/40/30
                # thresholds exact rather than susceptible to float rounding.
                comparisons = (
                    (current_median * 2, base_median, "median planning tokens"),
                    (metrics.total_model_input_tokens * 5, baseline.total_model_input_tokens * 3, "total model input tokens"),
                    (metrics.remote_model_calls * 5, baseline.remote_model_calls * 2, "remote-model calls"),
                    (metrics.retry_tokens * 10, baseline.retry_tokens * 3, "retry tokens"),
                )
                for current, limit, label in comparisons:
                    if current > limit:
                        fail(PromotionGateReason.TOKEN_EFFICIENCY_FAILURE, "{} exceeds threshold".format(label))
            if (metrics.recurring_tasks_without_remote_model * 100 < metrics.eligible_recurring_tasks * 60 or
                metrics.deterministic_repairs_without_model * 100 < metrics.deterministic_repair_tasks * 80 or
                metrics.accepted_benchmark_work_via_verified_procedures * 100 < metrics.accepted_benchmark_work * 30 or
                metrics.human_interventions * 100 > baseline.human_interventions * 75):
                fail(PromotionGateReason.AUTONOMY_FAILURE, "autonomy threshold is not met")
        if (metrics.unsafe_cross_repository_transfers != 0 or metrics.held_out_transfer_results == 0 or
                metrics.typed_transfer_refusals < metrics.transfer_assumption_mismatches):
            fail(PromotionGateReason.TRANSFER_FAILURE, "transfer safety, held-out result, or typed refusal is missing")
        if not amortization.break_even_observed:
            fail(PromotionGateReason.AMORTIZATION_FAILURE, "break-even has not been observed")
        if certificate_admission is None or not (certificate_admission.accepted and certificate_admission.usable):
            fail(PromotionGateReason.CERTIFICATE_FAILURE, "certificate is not independently admissible")
        if not isinstance(expected_old_revision_id, str) or not expected_old_revision_id or not isinstance(rollback_target_revision_id, str) or not rollback_target_revision_id:
            fail(PromotionGateReason.CAS_OR_ROLLBACK_MISSING, "expected-old CAS and exact rollback target are required")
        if failures:
            return PromotionGateReport(False, tuple(reasons), tuple(failures))
        return PromotionGateReport(True, (PromotionGateReason.ACCEPTED,), ())


__all__ = [
    "AmortizationReport",
    "ProcedureMetrics",
    "ProcedurePromotionGate",
    "PromotionGateReason",
    "PromotionGateReport",
]
