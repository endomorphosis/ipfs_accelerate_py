"""PTR-101: staged proof-reuse promotion, sampling, and rollback."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
    run_proof_reuse_benchmark,
)
from ipfs_accelerate_py.testing.proof_reuse.config import (
    ProofReuseConfig,
    ProofReuseMode,
)
from ipfs_accelerate_py.testing.proof_reuse.rollout import (
    BENCHMARK_RECEIPT_INTERFACE,
    PROOF_REUSE_CONFIG_INTERFACE,
    PROOF_REUSE_METRICS_INTERFACE,
    PROOF_REUSE_ROLLOUT_DECISION_INTERFACE,
    ForcedRerunObservation,
    ForcedRerunOutcome,
    ForcedRerunSampler,
    ForcedRerunSummary,
    ProofReusePromotionEvidence,
    ProofReuseRollbackDecision,
    ProofReuseRolloutError,
    ProofReuseRolloutPolicy,
    ProofReuseRolloutStage,
    ProofReuseSafetySignals,
    RollbackTrigger,
    RolloutDisposition,
)


NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
REPOSITORY_ID = "repository:proof-reuse"
TREE_ID = "tree:current"
POLICY_ID = "policy:proof-reuse-rollout"
POLICY_REVISION = "revision:7"


def _policy(
    *,
    approved_stages: tuple[ProofReuseRolloutStage, ...] = tuple(
        ProofReuseRolloutStage
    ),
    allow_eligible_default: bool = True,
    min_forced_reruns: int = 2,
) -> ProofReuseRolloutPolicy:
    return ProofReuseRolloutPolicy(
        policy_id=POLICY_ID,
        policy_revision=POLICY_REVISION,
        approved_stages=approved_stages,
        max_evidence_age_seconds=3600,
        max_future_skew_seconds=5,
        min_forced_reruns=min_forced_reruns,
        forced_rerun_sample_bps=500,
        allow_eligible_default=allow_eligible_default,
    )


def _clean_reruns(count: int = 2) -> ForcedRerunSummary:
    return ForcedRerunSummary(
        selected=count,
        completed=count,
        matched=count,
    )


def _evidence(
    current: ProofReuseRolloutStage,
    target: ProofReuseRolloutStage,
    **changes: object,
) -> ProofReusePromotionEvidence:
    benchmark = run_proof_reuse_benchmark()
    metrics = benchmark.scenario_summaries[0].metrics
    values: dict[str, object] = {
        "observed_at": NOW,
        "repository_id": REPOSITORY_ID,
        "tree_id": TREE_ID,
        "policy_id": POLICY_ID,
        "policy_revision": POLICY_REVISION,
        "current_stage": current,
        "target_stage": target,
        "benchmark_receipt": benchmark,
        "metrics_snapshot": metrics,
        "forced_reruns": _clean_reruns(),
        "mutation_false_skips": 0,
        "degradation_false_skips": 0,
        "authority_contradictions": 0,
        "corruption_spike": False,
        "stale_keys": 0,
        "key_health_ok": True,
        "revocation_health_ok": True,
        "operator_approval_id": "approval:change-42",
        "controlled_issuer": True,
        "current_tree_gate_passed": True,
        "all_repositories_passed": True,
    }
    values.update(changes)
    return ProofReusePromotionEvidence(**values)


def _promote(
    policy: ProofReuseRolloutPolicy,
    evidence: ProofReusePromotionEvidence,
    **changes: object,
):
    values: dict[str, object] = {
        "current_repository_id": REPOSITORY_ID,
        "current_tree_id": TREE_ID,
        "now": NOW,
    }
    values.update(changes)
    return policy.evaluate_promotion(evidence, **values)


def _clean_signals(**changes: object) -> ProofReuseSafetySignals:
    values: dict[str, object] = {
        "false_skips": 0,
        "authority_contradictions": 0,
        "corruption_spike": False,
        "stale_keys": 0,
        "unexplained_mismatches": 0,
    }
    values.update(changes)
    return ProofReuseSafetySignals(**values)


def test_defaults_remain_off_and_default_policy_cannot_grant_read() -> None:
    policy = ProofReuseRolloutPolicy()

    assert policy.default_stage is ProofReuseRolloutStage.OFF
    assert policy.to_dict()["interface"] == PROOF_REUSE_CONFIG_INTERFACE
    assert policy.config_for(policy.default_stage) == ProofReuseConfig(
        mode=ProofReuseMode.OFF,
        source="rollout:off",
    )

    evidence = _evidence(
        ProofReuseRolloutStage.SHADOW,
        ProofReuseRolloutStage.READ,
        policy_id=policy.policy_id,
        policy_revision=policy.policy_revision,
    )
    decision = policy.evaluate_promotion(
        evidence,
        current_repository_id=REPOSITORY_ID,
        current_tree_id=TREE_ID,
        now=NOW,
    )
    assert not decision.promoted
    assert "policy_approved" in decision.reason_codes
    assert decision.effective_stage is ProofReuseRolloutStage.SHADOW


def test_each_adjacent_stage_promotes_with_explicit_fresh_gates() -> None:
    policy = _policy()
    transitions = (
        (ProofReuseRolloutStage.OFF, ProofReuseRolloutStage.SHADOW),
        (ProofReuseRolloutStage.SHADOW, ProofReuseRolloutStage.READ),
        (
            ProofReuseRolloutStage.READ,
            ProofReuseRolloutStage.OPT_IN_READWRITE,
        ),
        (
            ProofReuseRolloutStage.OPT_IN_READWRITE,
            ProofReuseRolloutStage.ELIGIBLE_DEFAULT,
        ),
    )

    for current, target in transitions:
        decision = _promote(policy, _evidence(current, target))
        assert decision.promoted
        assert decision.disposition is RolloutDisposition.PROMOTE
        assert decision.effective_stage is target
        assert all(gate.passed for gate in decision.gates)
        assert decision.decision_id.startswith("sha256:")
        assert decision.to_dict()["interface"] == (
            PROOF_REUSE_ROLLOUT_DECISION_INTERFACE
        )


def test_promotion_cannot_jump_a_stage_or_reuse_evidence_for_another_stage() -> None:
    policy = _policy()
    jumped = _evidence(
        ProofReuseRolloutStage.OFF,
        ProofReuseRolloutStage.READ,
    )
    decision = _promote(policy, jumped)
    assert not decision.promoted
    assert decision.reason_codes == ("adjacent_stage",)

    valid = _evidence(
        ProofReuseRolloutStage.SHADOW,
        ProofReuseRolloutStage.READ,
    )
    rebound = _promote(
        policy,
        valid,
        target_stage=ProofReuseRolloutStage.OPT_IN_READWRITE,
    )
    assert not rebound.promoted
    assert "adjacent_stage" in rebound.reason_codes


@pytest.mark.parametrize(
    ("observed_at", "now"),
    (
        (NOW - timedelta(seconds=3601), NOW),
        (NOW + timedelta(seconds=6), NOW),
    ),
)
def test_stale_or_implausibly_future_evidence_holds(
    observed_at: datetime, now: datetime
) -> None:
    policy = _policy()
    evidence = _evidence(
        ProofReuseRolloutStage.SHADOW,
        ProofReuseRolloutStage.READ,
        observed_at=observed_at,
    )
    decision = _promote(policy, evidence, now=now)
    assert not decision.promoted
    assert "evidence_fresh" in decision.reason_codes


def test_current_repository_tree_and_policy_bindings_are_non_waivable() -> None:
    policy = _policy()
    evidence = _evidence(
        ProofReuseRolloutStage.SHADOW,
        ProofReuseRolloutStage.READ,
    )

    missing_current = policy.evaluate_promotion(evidence, now=NOW)
    assert "deployment_binding_current" in missing_current.reason_codes

    stale_tree = _promote(policy, evidence, current_tree_id="tree:stale")
    assert "deployment_binding_current" in stale_tree.reason_codes

    stale_policy = _promote(
        policy,
        replace(evidence, policy_revision="revision:stale"),
    )
    assert "policy_binding_current" in stale_policy.reason_codes


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ({"operator_approval_id": ""}, "operator_approved"),
        ({"benchmark_receipt": None}, "benchmark_passed"),
        ({"metrics_snapshot": None}, "metrics_interface_current"),
        ({"mutation_false_skips": None}, "mutation_degradation_clean"),
        ({"degradation_false_skips": 1}, "mutation_degradation_clean"),
        ({"authority_contradictions": 1}, "authority_consistent"),
        ({"corruption_spike": True}, "corruption_stable"),
        ({"stale_keys": 1}, "key_revocation_healthy"),
        ({"key_health_ok": None}, "key_revocation_healthy"),
        ({"revocation_health_ok": False}, "key_revocation_healthy"),
    ),
)
def test_missing_or_unsafe_promotion_evidence_fails_closed(
    change: dict[str, object], reason: str
) -> None:
    policy = _policy()
    evidence = _evidence(
        ProofReuseRolloutStage.SHADOW,
        ProofReuseRolloutStage.READ,
        **change,
    )
    decision = _promote(policy, evidence)
    assert not decision.promoted
    assert reason in decision.reason_codes


def test_benchmark_receipt_must_bind_expected_interfaces_and_all_gates() -> None:
    policy = _policy()
    evidence = _evidence(
        ProofReuseRolloutStage.SHADOW,
        ProofReuseRolloutStage.READ,
    )
    receipt = dict(evidence.benchmark_receipt)
    assert receipt["interface"] == BENCHMARK_RECEIPT_INTERFACE
    assert receipt["metrics_interface"] == PROOF_REUSE_METRICS_INTERFACE

    receipt["metrics_interface"] = "ProofReuseMetrics@0"
    wrong_interface = _promote(
        policy, replace(evidence, benchmark_receipt=receipt)
    )
    assert "benchmark_passed" in wrong_interface.reason_codes

    receipt = dict(evidence.benchmark_receipt)
    receipt["gates"] = [dict(receipt["gates"][0], passed=False)]
    failed_gate = _promote(
        policy, replace(evidence, benchmark_receipt=receipt)
    )
    assert "benchmark_passed" in failed_gate.reason_codes


def test_forced_rerun_sampler_is_stable_private_and_compares_actual_outcome() -> None:
    sampler = ForcedRerunSampler(sample_rate_bps=10_000, seed="ci:seed-7")
    identity = "private/test_file.py::test_secret[param]"

    assert sampler.should_sample(identity)
    assert sampler.should_force_rerun(identity)
    assert sampler.sample_id(identity) == sampler.sample_id(identity)
    assert identity not in sampler.sample_id(identity)

    match = sampler.compare(identity, ForcedRerunOutcome.PASS, True)
    mismatch = sampler.compare(identity, "pass", "fail")
    assert not match.mismatch
    assert not match.false_skip
    assert mismatch.mismatch
    assert mismatch.false_skip
    assert mismatch.unexplained_mismatch
    assert identity not in str(mismatch.to_dict())

    summary = ForcedRerunSummary.from_observations((match, mismatch))
    assert summary.completed == 2
    assert summary.matched == 1
    assert summary.false_skips == 1
    assert summary.unexplained_mismatches == 1
    assert not summary.clean


def test_sampler_rejects_unselected_comparisons_and_invalid_rates() -> None:
    sampler = ForcedRerunSampler(sample_rate_bps=0, seed="ci:seed")
    assert not sampler.should_sample("execution:key")
    with pytest.raises(ProofReuseRolloutError, match="not selected"):
        sampler.compare("execution:key", "pass", "pass")
    with pytest.raises(ProofReuseRolloutError):
        ForcedRerunSampler(sample_rate_bps=10_001, seed="ci:seed")


def test_read_promotion_requires_complete_clean_forced_reruns() -> None:
    policy = _policy(min_forced_reruns=2)
    evidence = _evidence(
        ProofReuseRolloutStage.SHADOW,
        ProofReuseRolloutStage.READ,
    )

    missing = _promote(policy, replace(evidence, forced_reruns=None))
    assert "forced_reruns_clean" in missing.reason_codes

    incomplete = _promote(
        policy,
        replace(
            evidence,
            forced_reruns=ForcedRerunSummary(
                selected=2,
                completed=1,
                matched=1,
            ),
        ),
    )
    assert "forced_reruns_clean" in incomplete.reason_codes

    false_skip = _promote(
        policy,
        replace(
            evidence,
            forced_reruns=ForcedRerunSummary(
                selected=2,
                completed=2,
                matched=1,
                unexplained_mismatches=1,
                false_skips=1,
            ),
        ),
    )
    assert "forced_reruns_clean" in false_skip.reason_codes


def test_shadow_promotion_records_sampling_but_does_not_require_minimum() -> None:
    policy = _policy(min_forced_reruns=100)
    evidence = _evidence(
        ProofReuseRolloutStage.OFF,
        ProofReuseRolloutStage.SHADOW,
        forced_reruns=None,
    )
    decision = _promote(policy, evidence)
    assert decision.promoted


def test_write_and_eligible_default_have_additional_explicit_gates() -> None:
    policy = _policy()
    write = _evidence(
        ProofReuseRolloutStage.READ,
        ProofReuseRolloutStage.OPT_IN_READWRITE,
        controlled_issuer=False,
    )
    assert "controlled_issuer" in _promote(policy, write).reason_codes

    default = _evidence(
        ProofReuseRolloutStage.OPT_IN_READWRITE,
        ProofReuseRolloutStage.ELIGIBLE_DEFAULT,
        all_repositories_passed=False,
    )
    assert "eligible_default_current_tree" in _promote(
        policy, default
    ).reason_codes

    policy_disallows = _policy(allow_eligible_default=False)
    asserted = replace(default, all_repositories_passed=True)
    assert "eligible_default_current_tree" in _promote(
        policy_disallows, asserted
    ).reason_codes


def test_promoted_stage_only_narrows_reviewed_config_authority() -> None:
    policy = _policy()

    assert policy.mode_for("off", eligible=True) is ProofReuseMode.OFF
    assert policy.mode_for("shadow", eligible=False) is ProofReuseMode.SHADOW
    assert policy.mode_for("read", eligible=False) is ProofReuseMode.OFF
    assert policy.mode_for("read", eligible=True) is ProofReuseMode.READ
    assert (
        policy.mode_for(
            "opt_in_readwrite",
            eligible=True,
            readwrite_opt_in=False,
        )
        is ProofReuseMode.READ
    )
    assert (
        policy.mode_for(
            "opt_in_readwrite",
            eligible=True,
            readwrite_opt_in=True,
        )
        is ProofReuseMode.READWRITE
    )
    assert (
        policy.mode_for(
            "eligible_default",
            eligible=True,
            readwrite_opt_in=False,
        )
        is ProofReuseMode.READ
    )
    assert policy.mode_for("eligible_default", eligible=False) is ProofReuseMode.OFF


@pytest.mark.parametrize(
    ("changes", "trigger"),
    (
        ({"false_skips": 1}, RollbackTrigger.FALSE_SKIP),
        (
            {"authority_contradictions": 1},
            RollbackTrigger.AUTHORITY_CONTRADICTION,
        ),
        ({"corruption_spike": True}, RollbackTrigger.CORRUPTION_SPIKE),
        ({"stale_keys": 1}, RollbackTrigger.STALE_KEY),
        (
            {"unexplained_mismatches": 1},
            RollbackTrigger.UNEXPLAINED_MISMATCH,
        ),
    ),
)
def test_every_required_hazard_automatically_rolls_back(
    changes: dict[str, object], trigger: RollbackTrigger
) -> None:
    decision = ProofReuseRollbackDecision.evaluate(
        ProofReuseRolloutStage.ELIGIBLE_DEFAULT,
        _clean_signals(**changes),
    )
    assert decision.triggered
    assert decision.rolled_back
    assert decision.automatic
    assert trigger in decision.triggers
    assert decision.effective_stage in (
        ProofReuseRolloutStage.SHADOW,
        ProofReuseRolloutStage.OFF,
    )


def test_severe_authority_hazards_go_off_and_diagnostic_hazards_go_shadow() -> None:
    for changes in (
        {"false_skips": 1},
        {"authority_contradictions": 1},
        {"stale_keys": 1},
    ):
        decision = ProofReuseRollbackDecision.evaluate(
            ProofReuseRolloutStage.READ,
            _clean_signals(**changes),
        )
        assert decision.effective_stage is ProofReuseRolloutStage.OFF

    for changes in (
        {"corruption_spike": True},
        {"unexplained_mismatches": 1},
    ):
        decision = ProofReuseRollbackDecision.evaluate(
            ProofReuseRolloutStage.READ,
            _clean_signals(**changes),
        )
        assert decision.effective_stage is ProofReuseRolloutStage.SHADOW

    shadow_hazard = ProofReuseRollbackDecision.evaluate(
        ProofReuseRolloutStage.SHADOW,
        _clean_signals(corruption_spike=True),
    )
    assert shadow_hazard.effective_stage is ProofReuseRolloutStage.OFF


def test_missing_safety_monitoring_fails_closed_and_clean_signals_hold_stage() -> None:
    missing = ProofReuseRollbackDecision.evaluate(
        ProofReuseRolloutStage.READ,
        ProofReuseSafetySignals(),
    )
    assert missing.triggers == (RollbackTrigger.UNEXPLAINED_MISMATCH,)
    assert missing.effective_stage is ProofReuseRolloutStage.SHADOW

    clean = ProofReuseRollbackDecision.evaluate(
        ProofReuseRolloutStage.READ,
        _clean_signals(),
    )
    assert not clean.triggered
    assert not clean.rolled_back
    assert clean.effective_stage is ProofReuseRolloutStage.READ

    incomplete_from_reruns = ProofReuseSafetySignals.from_forced_reruns(
        _clean_reruns()
    )
    incomplete_decision = ProofReuseRollbackDecision.evaluate(
        ProofReuseRolloutStage.READ,
        incomplete_from_reruns,
    )
    assert incomplete_decision.triggers == (
        RollbackTrigger.UNEXPLAINED_MISMATCH,
    )
    assert incomplete_decision.effective_stage is ProofReuseRolloutStage.SHADOW


def test_forced_rerun_summary_feeds_rollback_without_raw_test_identity() -> None:
    summary = ForcedRerunSummary(
        selected=3,
        completed=3,
        matched=2,
        unexplained_mismatches=1,
        false_skips=1,
    )
    signals = ProofReuseSafetySignals.from_forced_reruns(summary)
    decision = _policy().evaluate_rollback(
        ProofReuseRolloutStage.READ,
        signals,
    )
    assert decision.effective_stage is ProofReuseRolloutStage.OFF
    assert set(decision.triggers) == {
        RollbackTrigger.FALSE_SKIP,
        RollbackTrigger.UNEXPLAINED_MISMATCH,
    }
    assert "test_" not in str(decision.to_dict())


def test_evidence_and_decisions_are_reproducible_and_aggregate_only() -> None:
    policy = _policy()
    evidence = _evidence(
        ProofReuseRolloutStage.SHADOW,
        ProofReuseRolloutStage.READ,
    )
    first = _promote(policy, evidence)
    second = _promote(policy, evidence)

    assert evidence.evidence_id == evidence.evidence_id
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()
    assert first.decision_id == second.decision_id
    assert "nodeid" not in first.to_json()
    assert "stdout" not in first.to_json()


def test_summary_and_evidence_reject_inconsistent_or_unsafe_values() -> None:
    with pytest.raises(ProofReuseRolloutError, match="sha256"):
        ForcedRerunObservation(
            sample_id="sha256:" + ("z" * 64),
            predicted_outcome=ForcedRerunOutcome.PASS,
            actual_outcome=ForcedRerunOutcome.PASS,
        )
    with pytest.raises(ProofReuseRolloutError):
        ForcedRerunSummary(selected=1, completed=2, matched=2)
    with pytest.raises(ProofReuseRolloutError):
        ForcedRerunSummary(
            selected=2,
            completed=2,
            matched=2,
            false_skips=1,
        )
    with pytest.raises(ProofReuseRolloutError):
        _evidence(
            ProofReuseRolloutStage.SHADOW,
            ProofReuseRolloutStage.READ,
            stale_keys=-1,
        )
