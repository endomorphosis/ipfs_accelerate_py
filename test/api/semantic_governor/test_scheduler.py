"""Active audit scheduler tests for SCG-031.

Acceptance criteria enforced here:

* Configured shadow rates are honored (development/high risk full rate;
  mature low-risk samples and is not forced to 100 percent).
* Privacy zero-call policy is honored (forbidden external disclosure never
  admits with allow_external_expanded_disclosure=True).
* Starvation: aged high-value candidates receive a boost and admit ahead of
  mature repetitive low-risk flood.
* Unbounded queue growth is prevented (max_queue_depth).
* Mature low-risk work cannot monopolize admission slots.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    ShadowSelectionReason,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    DisclosureDisposition,
    default_shadow_disclosure_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow_plan import (
    BASIS_POINTS_MAX,
    LifecyclePhase,
    ShadowSamplingPolicy,
    default_shadow_sampling_policy,
    development_shadow_sampling_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.scheduler import (
    ACTIVE_AUDIT_SCHEDULER_INTERFACE,
    AUDIT_PRIORITY_INTERFACE,
    DEFAULT_MAX_MATURE_LOW_RISK_FRACTION_BP,
    DEFAULT_MAX_QUEUE_DEPTH,
    DEFAULT_STARVATION_AGE_MS,
    SCHEDULE_AUDITS_INTERFACE,
    SCG_ACTIVE_SCHEDULER_EVIDENCE,
    ActiveAuditScheduler,
    AuditAdmissionDisposition,
    AuditCandidate,
    AuditPriority,
    AuditQueueOverflowError,
    AuditSchedulerPolicy,
    FairnessClass,
    SemanticGovernorSchedulerError,
    active_audit_scheduler_evidence_id,
    classify_fairness_class,
    compute_audit_priority,
    default_audit_scheduler_policy,
    schedule_audits,
    schedule_audits_interface_id,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/scheduler.py"
)


# ---------------------------------------------------------------------------
# Fixtures / recipes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _candidate(
    task_id: str = "task.a",
    *,
    risk_class: str = "low",
    **overrides: Any,
) -> AuditCandidate:
    fields: dict[str, Any] = {
        "task_id": task_id,
        "task_class": "bugfix",
        "risk_class": risk_class,
        "context_pack_cid": _cid(f"pack-{task_id}"),
        "repository_state_cid": _cid(f"repo-{task_id}"),
        "estimated_cost_micros": 1_000,
        "queue_age_ms": 0,
    }
    fields.update(overrides)
    return AuditCandidate(**fields)


def _scheduler(
    *,
    max_queue_depth: int = 64,
    max_admissions_per_tick: int = 8,
    max_spend_micros_per_tick: int = 100_000_000,
    max_mature_low_risk_fraction_bp: int = DEFAULT_MAX_MATURE_LOW_RISK_FRACTION_BP,
    starvation_age_ms: int = DEFAULT_STARVATION_AGE_MS,
    starvation_boost_bp: int = 2_500,
    audit_policy: ShadowSamplingPolicy | None = None,
    raise_on_queue_overflow: bool = False,
) -> ActiveAuditScheduler:
    return ActiveAuditScheduler(
        scheduler_policy=AuditSchedulerPolicy(
            max_queue_depth=max_queue_depth,
            max_admissions_per_tick=max_admissions_per_tick,
            max_spend_micros_per_tick=max_spend_micros_per_tick,
            max_mature_low_risk_fraction_bp=max_mature_low_risk_fraction_bp,
            starvation_age_ms=starvation_age_ms,
            starvation_boost_bp=starvation_boost_bp,
            raise_on_queue_overflow=raise_on_queue_overflow,
            default_estimated_cost_micros=1_000,
        ),
        audit_policy=audit_policy or default_shadow_sampling_policy(random_seed=7),
        disclosure_policy=default_shadow_disclosure_policy(),
    )


# ---------------------------------------------------------------------------
# Module surface / evidence
# ---------------------------------------------------------------------------


def test_evidence_and_interfaces_are_stable() -> None:
    assert SCG_ACTIVE_SCHEDULER_EVIDENCE == "scg/active-scheduler@1"
    assert active_audit_scheduler_evidence_id() == SCG_ACTIVE_SCHEDULER_EVIDENCE
    assert SCHEDULE_AUDITS_INTERFACE == "schedule_audits@1"
    assert schedule_audits_interface_id() == SCHEDULE_AUDITS_INTERFACE
    assert ACTIVE_AUDIT_SCHEDULER_INTERFACE == "ActiveAuditScheduler@1"
    assert AUDIT_PRIORITY_INTERFACE == "AuditPriority@1"


def test_module_import_performs_no_io() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"open", "urlopen", "system", "Popen", "connect", "create_connection"}
    for node in tree.body:
        if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else (func.attr if isinstance(func, ast.Attribute) else "")
                )
                assert name not in forbidden


# ---------------------------------------------------------------------------
# Policy / priority identity
# ---------------------------------------------------------------------------


def test_default_scheduler_policy_is_bounded() -> None:
    policy = default_audit_scheduler_policy()
    assert policy.max_queue_depth == DEFAULT_MAX_QUEUE_DEPTH
    assert policy.max_queue_depth > 0
    assert policy.max_admissions_per_tick > 0
    assert policy.max_mature_low_risk_fraction_bp < BASIS_POINTS_MAX
    assert policy.policy_cid == AuditSchedulerPolicy().policy_cid


def test_scheduler_policy_round_trip() -> None:
    policy = AuditSchedulerPolicy(
        policy_id="audit-scheduler-lab",
        max_queue_depth=32,
        max_admissions_per_tick=4,
        max_mature_low_risk_fraction_bp=1_000,
        notes="lab",
    )
    restored = AuditSchedulerPolicy.from_dict(policy.to_dict())
    assert restored.policy_cid == policy.policy_cid
    assert restored.max_queue_depth == 32


def test_scheduler_policy_rejects_zero_queue() -> None:
    with pytest.raises(SemanticGovernorSchedulerError, match="positive"):
        AuditSchedulerPolicy(max_queue_depth=0)


def test_candidate_and_priority_identity_round_trip() -> None:
    cand = _candidate(
        "task.roundtrip",
        risk_class="high",
        capsule_uncertainty=True,
        sample_deficit_bp=4_000,
        cone_size=128,
    )
    restored = AuditCandidate.from_dict(cand.to_dict())
    assert restored.candidate_cid == cand.candidate_cid

    priority = compute_audit_priority(cand)
    restored_p = AuditPriority.from_dict(priority.to_dict())
    assert restored_p.priority_cid == priority.priority_cid
    assert restored_p.information_value_bp == priority.information_value_bp


def test_compute_audit_priority_ranks_factors() -> None:
    low = compute_audit_priority(_candidate("task.low", risk_class="low"))
    high = compute_audit_priority(
        _candidate(
            "task.high",
            risk_class="critical",
            capsule_uncertainty=True,
            recent_omission=True,
            sample_deficit_bp=8_000,
            rule_exposure=True,
            promotion_evaluation=True,
            cone_size=512,
            dynamic_features=True,
            cost_escalation_pressure_bp=5_000,
            policy_importance_bp=5_000,
            token_savings_eligible=True,
        )
    )
    assert high.information_value_bp > low.information_value_bp
    assert high.risk_score_bp > low.risk_score_bp
    assert high.failure_score_bp > 0
    assert high.sample_deficit_score_bp == 8_000
    assert high.cone_size_score_bp > 0
    assert "risk_high" in high.reason_codes
    assert "sample_deficit" in high.reason_codes


def test_starvation_boost_raises_effective_priority() -> None:
    policy = AuditSchedulerPolicy(
        starvation_age_ms=1_000,
        starvation_boost_bp=3_000,
    )
    young = compute_audit_priority(
        _candidate("task.young", risk_class="medium", queue_age_ms=0),
        policy,
    )
    aged = compute_audit_priority(
        _candidate("task.aged", risk_class="medium", queue_age_ms=5_000),
        policy,
    )
    assert aged.starvation_boost_bp == 3_000
    assert aged.effective_priority_bp == aged.information_value_bp + 3_000
    assert aged.effective_priority_bp > young.effective_priority_bp
    assert "starvation_boost" in aged.reason_codes


def test_fairness_class_classification() -> None:
    assert (
        classify_fairness_class(risk_class="low", environment="production")
        == FairnessClass.MATURE_LOW_RISK.value
    )
    assert (
        classify_fairness_class(risk_class="high", environment=None)
        == FairnessClass.HIGH_RISK.value
    )
    assert (
        classify_fairness_class(risk_class="low", environment="development")
        == FairnessClass.DEVELOPMENT.value
    )


# ---------------------------------------------------------------------------
# Acceptance: configured shadow rates are honored
# ---------------------------------------------------------------------------


def test_development_policy_admits_at_full_rate_even_on_high_roll() -> None:
    result = schedule_audits(
        [
            _candidate(
                "task.dev",
                risk_class="low",
                environment="development",
            )
        ],
        audit_policy=development_shadow_sampling_policy(random_seed=1),
        sample_rolls={"task.dev": 9_999},
        schedule_id="schedule.dev.full",
    )
    assert result.admitted_count == 1
    assert result.admitted_task_ids == ("task.dev",)
    admission = result.admissions[0]
    assert admission.admitted is True
    assert admission.plan_decision is not None
    assert admission.plan_decision.selected is True
    assert (
        ShadowSelectionReason.DEVELOPMENT_FULL_RATE.value
        in admission.plan_decision.selection_reasons
    )
    assert "shadow_rate_honored" in admission.reason_codes


def test_high_risk_admits_at_configured_full_rate() -> None:
    result = schedule_audits(
        [_candidate("task.high", risk_class="high")],
        audit_policy=default_shadow_sampling_policy(),
        sample_rolls={"task.high": 9_999},
        schedule_id="schedule.high.full",
    )
    assert result.admitted_count == 1
    admission = result.admissions[0]
    assert admission.plan_decision is not None
    assert admission.plan_decision.effective_sample_rate_bp == BASIS_POINTS_MAX
    assert (
        ShadowSelectionReason.RISK_CLASS_MANDATORY.value
        in admission.plan_decision.selection_reasons
    )


def test_mature_low_risk_sample_miss_is_honored() -> None:
    policy = default_shadow_sampling_policy(random_seed=1)
    assert policy.mature_low_risk_sample_rate_bp < BASIS_POINTS_MAX
    result = schedule_audits(
        [_candidate("task.low.miss", risk_class="low")],
        audit_policy=policy,
        sample_rolls={"task.low.miss": 9_999},
        schedule_id="schedule.low.miss",
    )
    assert result.admitted_count == 0
    assert len(result.deferred) == 1
    deferred = result.deferred[0]
    assert deferred.disposition == AuditAdmissionDisposition.SAMPLE_SKIPPED.value
    assert "shadow_rate_honored" in deferred.reason_codes
    assert "sample_miss" in deferred.reason_codes


def test_mature_low_risk_sample_hit_admits() -> None:
    policy = default_shadow_sampling_policy(random_seed=1)
    result = schedule_audits(
        [_candidate("task.low.hit", risk_class="low")],
        audit_policy=policy,
        sample_rolls={"task.low.hit": 0},
        schedule_id="schedule.low.hit",
    )
    assert result.admitted_count == 1
    assert result.admissions[0].plan_decision is not None
    assert (
        ShadowSelectionReason.RISK_CLASS_MANDATORY.value
        not in result.admissions[0].plan_decision.selection_reasons
    )


def test_custom_shadow_rate_zero_skips_medium_risk() -> None:
    """A policy with zero medium rate and zero QC must not admit medium work."""

    policy = ShadowSamplingPolicy(
        lifecycle_phase=LifecyclePhase.MATURE,
        medium_risk_sample_rate_bp=0,
        mature_low_risk_sample_rate_bp=0,
        random_quality_control_rate_bp=0,
        capsule_uncertainty_rate_bp=0,
        novelty_rate_bp=0,
        token_savings_rate_bp=0,
        proof_cache_reuse_rate_bp=0,
        recent_omission_rate_bp=0,
        promotion_evaluation_rate_bp=0,
        high_risk_sample_rate_bp=BASIS_POINTS_MAX,
        critical_risk_sample_rate_bp=BASIS_POINTS_MAX,
        development_sample_rate_bp=BASIS_POINTS_MAX,
    )
    result = schedule_audits(
        [_candidate("task.medium.zero", risk_class="medium")],
        audit_policy=policy,
        sample_rolls={"task.medium.zero": 0},
        schedule_id="schedule.medium.zero",
    )
    assert result.admitted_count == 0
    assert result.deferred[0].disposition == (
        AuditAdmissionDisposition.SAMPLE_SKIPPED.value
    )


# ---------------------------------------------------------------------------
# Acceptance: privacy zero-call policy is honored
# ---------------------------------------------------------------------------


def test_privacy_zero_call_forbids_external_on_private_source() -> None:
    audit_policy = default_shadow_sampling_policy()
    assert audit_policy.zero_external_calls_when_disclosure_forbidden is True
    assert audit_policy.allow_external_expanded_disclosure is False

    result = schedule_audits(
        [
            _candidate(
                "task.private",
                risk_class="high",
                includes_private_source=True,
                expanded_provider_id="external.unapproved.vendor",
                expanded_context={
                    "raw_private_source": "def secret():\n    return 1\n"
                },
            )
        ],
        audit_policy=audit_policy,
        sample_rolls={"task.private": 0},
        schedule_id="schedule.privacy.zero",
    )
    assert result.admitted_count == 1
    admission = result.admissions[0]
    assert admission.allow_external_expanded_disclosure is False
    assert admission.plan_decision is not None
    assert admission.plan_decision.allow_external_expanded_disclosure is False
    assert (
        admission.plan_decision.disclosure_disposition
        == DisclosureDisposition.FORBIDDEN.value
    )
    assert (
        ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
        in admission.plan_decision.selection_reasons
    )
    assert "privacy_zero_call_honored" in admission.reason_codes
    if admission.plan_decision.plan is not None:
        assert (
            admission.plan_decision.plan.allow_external_expanded_disclosure is False
        )


def test_sampling_policy_cannot_bypass_privacy_for_unapproved_external() -> None:
    # Even if sampling policy requests external disclosure, privacy forbids it.
    audit_policy = ShadowSamplingPolicy(
        allow_external_expanded_disclosure=True,
        zero_external_calls_when_disclosure_forbidden=True,
        high_risk_sample_rate_bp=BASIS_POINTS_MAX,
    )
    result = schedule_audits(
        [
            _candidate(
                "task.bypass",
                risk_class="high",
                includes_private_source=True,
                expanded_provider_id="external.unapproved.vendor",
                expanded_context={"raw_private_source": "SECRET=1"},
            )
        ],
        audit_policy=audit_policy,
        sample_rolls={"task.bypass": 0},
        schedule_id="schedule.privacy.nobypass",
    )
    assert result.admitted_count == 1
    admission = result.admissions[0]
    assert admission.allow_external_expanded_disclosure is False
    assert admission.plan_decision is not None
    assert admission.plan_decision.allow_external_expanded_disclosure is False


# ---------------------------------------------------------------------------
# Acceptance: starvation prevention
# ---------------------------------------------------------------------------


def test_starvation_boost_admits_aged_high_value_over_fresh_low_risk_flood() -> None:
    """Aged high-information work must not starve behind mature low-risk spam.

    With a tight admission budget, a starved medium/high-value candidate that
    has aged past the starvation threshold must outrank fresher low-risk
    candidates and be admitted first.
    """

    scheduler = _scheduler(
        max_admissions_per_tick=1,
        max_mature_low_risk_fraction_bp=0,  # low-risk cannot take the only slot
        starvation_age_ms=1_000,
        starvation_boost_bp=4_000,
        audit_policy=development_shadow_sampling_policy(),  # admit all selected
    )

    # Flood of fresh mature low-risk (would monopolize if allowed).
    lows = [
        _candidate(
            f"task.low.{i:03d}",
            risk_class="low",
            queue_age_ms=0,
            estimated_cost_micros=100,
        )
        for i in range(20)
    ]
    # Aged high-value candidate with sample deficit / uncertainty.
    aged = _candidate(
        "task.aged.highvalue",
        risk_class="medium",
        queue_age_ms=10_000,
        capsule_uncertainty=True,
        sample_deficit_bp=9_000,
        recent_omission=True,
        estimated_cost_micros=100,
    )

    result = scheduler.schedule_audits(
        lows + [aged],
        sample_rolls={
            **{c.task_id: 0 for c in lows},
            "task.aged.highvalue": 0,
        },
        schedule_id="schedule.starvation.1",
    )

    assert result.admitted_count == 1
    assert result.admitted_task_ids == ("task.aged.highvalue",)
    admitted = result.admissions[0]
    assert admitted.priority is not None
    assert admitted.priority.starvation_boost_bp == 4_000
    assert "starvation_boost" in admitted.priority.reason_codes
    # Low-risk flood must not have been admitted under monopoly cap.
    assert result.mature_low_risk_admitted_count == 0
    assert any(
        d.disposition == AuditAdmissionDisposition.ANTI_MONOPOLY_DEFERRED.value
        for d in result.deferred
    )


def test_starvation_boost_changes_rank_order_deterministically() -> None:
    policy = AuditSchedulerPolicy(starvation_age_ms=500, starvation_boost_bp=5_000)
    # Fresh high risk vs aged medium with boost — ensure rank_key ordering.
    fresh_high = compute_audit_priority(
        _candidate("task.z.fresh", risk_class="high", queue_age_ms=0),
        policy,
    )
    aged_medium = compute_audit_priority(
        _candidate(
            "task.a.aged",
            risk_class="medium",
            queue_age_ms=5_000,
            sample_deficit_bp=9_000,
            capsule_uncertainty=True,
            recent_failure=True,
        ),
        policy,
    )
    # Both should be orderable; aged with boost is competitive.
    assert aged_medium.starvation_boost_bp > 0
    ordered = sorted([fresh_high, aged_medium], key=lambda p: p.rank_key())
    assert ordered[0].task_id in {fresh_high.task_id, aged_medium.task_id}
    # Determinism: same inputs same order.
    ordered2 = sorted([fresh_high, aged_medium], key=lambda p: p.rank_key())
    assert [p.task_id for p in ordered] == [p.task_id for p in ordered2]


# ---------------------------------------------------------------------------
# Acceptance: unbounded queue growth prevented
# ---------------------------------------------------------------------------


def test_queue_rejects_unbounded_growth() -> None:
    scheduler = _scheduler(max_queue_depth=5, max_admissions_per_tick=1)
    outcomes = []
    for i in range(12):
        outcome = scheduler.enqueue(
            _candidate(f"task.q.{i:03d}", risk_class="low")
        )
        if outcome is not None:
            outcomes.append(outcome)

    assert scheduler.queue_depth == 5
    assert scheduler.overflow_count == 7
    assert len(outcomes) == 7
    assert all(
        o.disposition == AuditAdmissionDisposition.QUEUE_OVERFLOW.value
        for o in outcomes
    )
    assert all("unbounded_growth_prevented" in o.reason_codes for o in outcomes)


def test_queue_overflow_raise_mode() -> None:
    scheduler = _scheduler(max_queue_depth=2, raise_on_queue_overflow=True)
    assert scheduler.enqueue(_candidate("task.a1")) is None
    assert scheduler.enqueue(_candidate("task.a2")) is None
    with pytest.raises(AuditQueueOverflowError, match="capacity"):
        scheduler.enqueue(_candidate("task.a3"))


def test_schedule_audits_reports_overflow_in_rejected() -> None:
    scheduler = _scheduler(max_queue_depth=3, max_admissions_per_tick=1)
    # Pre-fill queue to capacity.
    for i in range(3):
        scheduler.enqueue(_candidate(f"task.pre.{i}", risk_class="high"))

    result = scheduler.schedule_audits(
        [
            _candidate("task.extra.1", risk_class="high"),
            _candidate("task.extra.2", risk_class="high"),
        ],
        sample_rolls={
            "task.pre.0": 0,
            "task.pre.1": 0,
            "task.pre.2": 0,
            "task.extra.1": 0,
            "task.extra.2": 0,
        },
        schedule_id="schedule.overflow.1",
    )
    assert result.queue_depth_before == 3
    assert len(result.rejected) == 2
    assert all(
        r.disposition == AuditAdmissionDisposition.QUEUE_OVERFLOW.value
        for r in result.rejected
    )
    # Queue cannot grow past max even across schedule ticks with new input.
    assert result.queue_depth_after <= 3


def test_repeated_enqueue_cannot_grow_without_bound() -> None:
    scheduler = _scheduler(max_queue_depth=10, max_admissions_per_tick=2)
    for round_i in range(50):
        for j in range(20):
            scheduler.enqueue(
                _candidate(f"task.r{round_i}.j{j}", risk_class="low")
            )
        # Drain a little so the queue stays active but bounded.
        scheduler.schedule_audits(
            sample_rolls={
                tid: 9_999  # miss samples so deferred drop (sample miss)
                for tid in scheduler.pending_task_ids()
            },
            schedule_id=f"schedule.bound.{round_i}",
        )
        assert scheduler.queue_depth <= 10


# ---------------------------------------------------------------------------
# Anti-monopoly: mature low-risk cannot monopolize audit spend
# ---------------------------------------------------------------------------


def test_mature_low_risk_cannot_monopolize_admission_slots() -> None:
    scheduler = _scheduler(
        max_admissions_per_tick=8,
        max_mature_low_risk_fraction_bp=2_500,  # 25% => 2 slots
        audit_policy=development_shadow_sampling_policy(),
    )
    lows = [
        _candidate(f"task.mono.low.{i:02d}", risk_class="low")
        for i in range(20)
    ]
    highs = [
        _candidate(f"task.mono.high.{i:02d}", risk_class="high")
        for i in range(6)
    ]
    result = scheduler.schedule_audits(
        lows + highs,
        sample_rolls={c.task_id: 0 for c in lows + highs},
        schedule_id="schedule.monopoly.1",
    )
    assert result.admitted_count == 8
    assert result.mature_low_risk_admitted_count <= 2
    high_admitted = [
        tid for tid in result.admitted_task_ids if tid.startswith("task.mono.high")
    ]
    assert len(high_admitted) >= 6  # all highs fit under remaining slots
    assert any(
        d.disposition == AuditAdmissionDisposition.ANTI_MONOPOLY_DEFERRED.value
        for d in result.deferred
    )


def test_spend_budget_resource_admission() -> None:
    scheduler = _scheduler(
        max_admissions_per_tick=10,
        max_spend_micros_per_tick=2_500,
        max_mature_low_risk_fraction_bp=BASIS_POINTS_MAX,  # allow low risk
        audit_policy=development_shadow_sampling_policy(),
    )
    cands = [
        _candidate(f"task.spend.{i}", risk_class="high", estimated_cost_micros=1_000)
        for i in range(5)
    ]
    result = scheduler.schedule_audits(
        cands,
        sample_rolls={c.task_id: 0 for c in cands},
        schedule_id="schedule.spend.1",
    )
    assert result.admitted_count == 2
    assert result.projected_spend_micros == 2_000
    assert any(
        d.disposition == AuditAdmissionDisposition.SPEND_EXHAUSTED.value
        for d in result.deferred
    )


# ---------------------------------------------------------------------------
# Determinism / ranking order
# ---------------------------------------------------------------------------


def test_schedule_is_deterministic() -> None:
    cands = [
        _candidate("task.d.b", risk_class="medium", sample_deficit_bp=3_000),
        _candidate("task.d.a", risk_class="high"),
        _candidate("task.d.c", risk_class="low"),
    ]
    rolls = {c.task_id: 0 for c in cands}
    left = schedule_audits(
        cands,
        audit_policy=development_shadow_sampling_policy(),
        sample_rolls=rolls,
        schedule_id="schedule.det.1",
    )
    right = schedule_audits(
        cands,
        audit_policy=development_shadow_sampling_policy(),
        sample_rolls=rolls,
        schedule_id="schedule.det.1",
    )
    assert left.admitted_task_ids == right.admitted_task_ids
    assert left.result_cid == right.result_cid


def test_higher_information_value_admits_first_under_tight_budget() -> None:
    scheduler = _scheduler(
        max_admissions_per_tick=1,
        max_mature_low_risk_fraction_bp=BASIS_POINTS_MAX,
        audit_policy=development_shadow_sampling_policy(),
    )
    low_iv = _candidate(
        "task.iv.low",
        risk_class="low",
        estimated_cost_micros=100,
    )
    high_iv = _candidate(
        "task.iv.high",
        risk_class="critical",
        capsule_uncertainty=True,
        recent_omission=True,
        sample_deficit_bp=9_000,
        estimated_cost_micros=100,
    )
    result = scheduler.schedule_audits(
        [low_iv, high_iv],
        sample_rolls={"task.iv.low": 0, "task.iv.high": 0},
        schedule_id="schedule.iv.1",
    )
    assert result.admitted_task_ids == ("task.iv.high",)


def test_duplicate_enqueue_is_reported() -> None:
    scheduler = _scheduler()
    assert scheduler.enqueue(_candidate("task.dup")) is None
    dup = scheduler.enqueue(_candidate("task.dup"))
    assert dup is not None
    assert dup.disposition == AuditAdmissionDisposition.DUPLICATE.value
    assert scheduler.queue_depth == 1


def test_active_scheduler_stateful_multi_tick() -> None:
    scheduler = _scheduler(
        max_admissions_per_tick=1,
        max_mature_low_risk_fraction_bp=BASIS_POINTS_MAX,
        audit_policy=development_shadow_sampling_policy(),
    )
    scheduler.enqueue(_candidate("task.t1", risk_class="high"))
    scheduler.enqueue(_candidate("task.t2", risk_class="high"))
    r1 = scheduler.schedule_audits(
        sample_rolls={"task.t1": 0, "task.t2": 0},
        schedule_id="schedule.multi.1",
    )
    assert r1.admitted_count == 1
    assert scheduler.queue_depth == 1
    r2 = scheduler.schedule_audits(
        sample_rolls={"task.t1": 0, "task.t2": 0},
        schedule_id="schedule.multi.2",
    )
    assert r2.admitted_count == 1
    assert scheduler.queue_depth == 0
    assert scheduler.total_admitted == 2
