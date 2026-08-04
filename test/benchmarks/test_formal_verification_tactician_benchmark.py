"""FVT-033 / FVT-G063: GoalTacticianBenchmark@1 and GoalTacticianMetrics@1.

Validates that formal verification tactician quality metrics:

* are derived from actual cohort receipts (not synthetic distributions);
* enforce 100 percent hard correctness, privacy, and authority gates;
* treat timing as observational unless calibrated;
* require cache hits to preserve authority and exact identity; and
* expose supervisor progress (unresolved holes, witnesses, critical path,
  budgets, next actions).

Also locks the architecture report at
``docs/architecture/formal_verification_tactician_benchmark.json``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.goal_tactician_metrics import (
    BASIS_POINTS,
    CacheOutcome,
    EvidenceClass,
    GOAL_TACTICIAN_BENCHMARK_INTERFACE,
    GOAL_TACTICIAN_BENCHMARK_SCHEMA,
    GOAL_TACTICIAN_METRICS_INTERFACE,
    GOAL_TACTICIAN_METRICS_SCHEMA,
    HARD_GATE_NAMES,
    OBSERVATIONAL_METRIC_NAMES,
    GoalTacticianMetricsError,
    GoalTacticianRunReceipt,
    architecture_benchmark_document,
    build_goal_tactician_benchmark_report,
    derive_goal_tactician_metrics,
    fixture_cohort_receipts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_benchmark.json"
)
OBJECTIVES_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_readiness.objectives.md"
)
METRICS_MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "proof"
    / "goal_tactician_metrics.py"
)

GOAL_ID = "FVT-G063"
TASK_ID = "FVT-033"
SCHEMA_VERSION = "formal-verification-tactician-benchmark/v1"


def _load_benchmark_document() -> dict[str, Any]:
    assert BENCHMARK_PATH.is_file(), f"missing benchmark report: {BENCHMARK_PATH}"
    payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _receipt(**changes: Any) -> GoalTacticianRunReceipt:
    base = fixture_cohort_receipts()[0]
    payload = base.to_dict()
    payload.update(changes)
    return GoalTacticianRunReceipt.from_dict(payload)


# ---------------------------------------------------------------------------
# Architecture document contract
# ---------------------------------------------------------------------------


def test_architecture_benchmark_document_exists_and_binds_goal() -> None:
    doc = _load_benchmark_document()
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["schema"] == GOAL_TACTICIAN_BENCHMARK_SCHEMA
    assert doc["interface"] == GOAL_TACTICIAN_BENCHMARK_INTERFACE
    assert doc["metrics_interface"] == GOAL_TACTICIAN_METRICS_INTERFACE
    assert doc["goal_id"] == GOAL_ID
    assert doc["task_id"] == TASK_ID
    assert doc["synthetic_distributions"] is False
    assert doc["source"] == "cohort_receipts"
    assert METRICS_MODULE_PATH.is_file()


def test_architecture_document_matches_fixture_cohort_derivation() -> None:
    doc = _load_benchmark_document()
    rebuilt = architecture_benchmark_document(
        build_goal_tactician_benchmark_report(fixture_cohort_receipts())
    )
    # Stable identity fields must match the checked-in report projection.
    assert doc["interface"] == rebuilt["interface"]
    assert doc["goal_id"] == rebuilt["goal_id"]
    assert doc["report"]["receipt_count"] == rebuilt["report"]["receipt_count"]
    assert (
        doc["report"]["metrics"]["hard_gates"]["passed"]
        is rebuilt["report"]["metrics"]["hard_gates"]["passed"]
        is True
    )
    assert (
        doc["report"]["metrics"]["cache"]["hits_preserve_authority_and_identity"]
        is True
    )
    assert (
        doc["report"]["metrics"]["resources"]["timing_role"] == "observational"
    )


def test_architecture_document_covers_required_metric_dimensions() -> None:
    doc = _load_benchmark_document()
    required = {
        "formalization",
        "proof_gap_recall_precision",
        "plan_solvability",
        "proof_authority",
        "counterexample_replay_reduction_explanation",
        "provider_agreement",
        "resources",
        "cancellation",
        "cache_correctness",
        "supervisor_progress",
    }
    assert required.issubset(set(doc["metric_dimensions"]))
    assert list(doc["hard_gates"]) == list(HARD_GATE_NAMES)
    assert set(OBSERVATIONAL_METRIC_NAMES).issubset(set(doc["observational_fields"]))

    metrics = doc["report"]["metrics"]
    for section in (
        "formalization",
        "proof_gap",
        "plan_solvability",
        "proof_authority",
        "counterexamples",
        "provider_agreement",
        "resources",
        "cache",
        "hard_gates",
        "progress",
    ):
        assert section in metrics


def test_objectives_heap_lists_benchmark_evidence() -> None:
    text = OBJECTIVES_PATH.read_text(encoding="utf-8")
    assert GOAL_ID in text
    assert "test/benchmarks/test_formal_verification_tactician_benchmark.py" in text
    assert "docs/architecture/formal_verification_tactician_benchmark.json" in text
    assert "ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_metrics.py" in text


# ---------------------------------------------------------------------------
# Metrics derivation from cohort receipts
# ---------------------------------------------------------------------------


def test_metrics_derive_from_cohort_receipts_not_synthetic() -> None:
    metrics = derive_goal_tactician_metrics(fixture_cohort_receipts())
    payload = metrics.to_dict()
    assert payload["schema"] == GOAL_TACTICIAN_METRICS_SCHEMA
    assert payload["interface"] == GOAL_TACTICIAN_METRICS_INTERFACE
    assert payload["source"] == "cohort_receipts"
    assert payload["synthetic_distributions"] is False
    assert metrics.receipt_count == 3
    assert metrics.proof_gap_true_positive == 4 + 5 + 1
    assert metrics.proof_gap_precision_bps == BASIS_POINTS  # no false positives
    assert 0 < metrics.proof_gap_recall_bps <= BASIS_POINTS
    assert metrics.plan_steps_solvable == 5 + 6 + 2
    assert metrics.formalization_succeeded_count == 3


def test_hard_gates_require_one_hundred_percent() -> None:
    good = derive_goal_tactician_metrics(fixture_cohort_receipts())
    assert good.hard_gates_passed is True
    assert good.hard_gate_correctness_bps == BASIS_POINTS
    assert good.hard_gate_privacy_bps == BASIS_POINTS
    assert good.hard_gate_authority_bps == BASIS_POINTS

    bad_completion = _receipt(
        receipt_id="receipt:bad-completion",
        false_completion=True,
    )
    bad = derive_goal_tactician_metrics(
        (*fixture_cohort_receipts()[1:], bad_completion)
    )
    assert bad.hard_gates_passed is False
    assert bad.hard_gate_correctness_bps == 0
    assert bad.false_completion_count == 1

    bad_privacy = _receipt(
        receipt_id="receipt:bad-privacy",
        privacy_violation=True,
    )
    privacy = derive_goal_tactician_metrics((bad_privacy,))
    assert privacy.hard_gate_privacy_bps == 0
    assert privacy.hard_gates_passed is False

    bad_authority = _receipt(
        receipt_id="receipt:bad-authority",
        claimed_assurance="attested",
        authoritative_assurance="candidate",
        authority_boundary_violation=True,
    )
    authority = derive_goal_tactician_metrics((bad_authority,))
    assert authority.hard_gate_authority_bps == 0
    assert authority.hard_gates_passed is False


def test_timing_is_observational_unless_calibrated() -> None:
    metrics = derive_goal_tactician_metrics(fixture_cohort_receipts())
    resources = metrics.to_dict()["resources"]
    assert resources["calibrated_timing"] is False
    assert resources["timing_role"] == "observational"
    assert "wall_time_ms" in resources["observational_fields"]
    assert metrics.wall_time_total_ms == 1_200 + 450 + 80

    calibrated = _receipt(
        receipt_id="receipt:calibrated",
        evidence_class=EvidenceClass.CALIBRATED.value,
        calibration_receipt_id="calibration:host-2026-07-30",
        wall_time_ms=100,
    )
    cal_metrics = derive_goal_tactician_metrics((calibrated,))
    assert cal_metrics.calibrated_timing is True
    assert cal_metrics.to_dict()["resources"]["timing_role"] == "calibrated_gate_eligible"


def test_cache_hits_must_preserve_authority_and_identity() -> None:
    hit = _receipt(
        receipt_id="receipt:warm-hit",
        cache_outcome=CacheOutcome.HIT.value,
        cache_key="cache:exact-1",
        cache_authority_preserved=True,
        cache_identity_preserved=True,
    )
    metrics = derive_goal_tactician_metrics((hit,))
    cache = metrics.to_dict()["cache"]
    assert cache["hit_count"] == 1
    assert cache["hits_preserve_authority_and_identity"] is True

    with pytest.raises(GoalTacticianMetricsError, match="preserve authority"):
        _receipt(
            receipt_id="receipt:poisoned-hit",
            cache_outcome=CacheOutcome.HIT.value,
            cache_key="cache:poison",
            cache_authority_preserved=False,
            cache_identity_preserved=True,
        )

    with pytest.raises(GoalTacticianMetricsError, match="preserve authority"):
        _receipt(
            receipt_id="receipt:identity-drift-hit",
            cache_outcome=CacheOutcome.HIT.value,
            cache_key="cache:drift",
            cache_authority_preserved=True,
            cache_identity_preserved=False,
        )

    with pytest.raises(GoalTacticianMetricsError, match="cache_key"):
        _receipt(
            receipt_id="receipt:hit-no-key",
            cache_outcome=CacheOutcome.HIT.value,
            cache_key="",
            cache_authority_preserved=True,
            cache_identity_preserved=True,
        )


def test_progress_exposes_holes_witnesses_critical_path_budgets_next_actions() -> None:
    metrics = derive_goal_tactician_metrics(fixture_cohort_receipts())
    progress = metrics.to_dict()["progress"]
    assert "hole:invariant-strengthening" in progress["unresolved_hole_ids"]
    assert "hole:budget-exhausted" in progress["unresolved_hole_ids"]
    assert "witness:cx-1" in progress["witness_ids"]
    assert progress["critical_path_step_ids"]
    assert "budgets" in progress
    assert progress["budgets"]["cpu_ms_remaining"] == 0  # cancelled run is tightest
    assert progress["next_actions"]
    assert "resume-from-checkpoint" in progress["next_actions"]
    assert progress["cancelled_run_count"] == 1
    assert progress["open_plan_count"] >= 1


def test_counterexample_and_provider_agreement_rates() -> None:
    metrics = derive_goal_tactician_metrics(fixture_cohort_receipts())
    cx = metrics.to_dict()["counterexamples"]
    assert cx["count"] == 3
    assert cx["replayable_count"] == 3
    assert cx["replay_bps"] == BASIS_POINTS
    assert cx["reduced_count"] == 3
    assert 0 < cx["explanation_bps"] <= BASIS_POINTS

    agreement = metrics.to_dict()["provider_agreement"]
    assert agreement["query_pairs"] == 2 + 2 + 1
    assert agreement["agreement_pairs"] == 2 + 2 + 1
    assert agreement["agreement_bps"] == BASIS_POINTS


def test_cancellation_honor_rate_from_receipts() -> None:
    metrics = derive_goal_tactician_metrics(fixture_cohort_receipts())
    assert metrics.cancelled_count == 1
    assert metrics.cancellation_honored_count == 1
    assert metrics.cancellation_honor_bps == BASIS_POINTS

    ignored = _receipt(
        receipt_id="receipt:cancel-ignored",
        cancelled=True,
        cancellation_honored=False,
        plan_admitted=False,
        unresolved_hole_ids=("hole:cancel",),
    )
    bad = derive_goal_tactician_metrics((ignored,))
    assert bad.cancellation_honor_bps == 0


def test_private_material_rejected_from_receipts_and_reports() -> None:
    with pytest.raises(GoalTacticianMetricsError, match="private material"):
        GoalTacticianRunReceipt.from_dict(
            {
                **fixture_cohort_receipts()[0].to_dict(),
                "receipt_id": "receipt:leaky",
                "proof_body": "secret-term",
            }
        )

    report = build_goal_tactician_benchmark_report(fixture_cohort_receipts())
    rendered = json.dumps(report.to_dict())
    assert "proof_body" not in rendered
    assert report["contains_prompts"] is False
    assert report["contains_proof_transcripts"] is False
    assert report["contains_private_witnesses"] is False


def test_claimed_assurance_cannot_exceed_authoritative_without_violation_flag() -> None:
    with pytest.raises(GoalTacticianMetricsError, match="claimed_assurance exceeds"):
        _receipt(
            receipt_id="receipt:overclaim",
            claimed_assurance="attested",
            authoritative_assurance="solver_checked",
            authority_boundary_violation=False,
        )


def test_empty_cohort_and_duplicate_receipts_fail_closed() -> None:
    with pytest.raises(GoalTacticianMetricsError, match="at least one"):
        derive_goal_tactician_metrics(())

    twin = fixture_cohort_receipts()[0]
    with pytest.raises(GoalTacticianMetricsError, match="unique"):
        derive_goal_tactician_metrics((twin, twin))


def test_benchmark_report_gates_and_acceptance_block() -> None:
    report = build_goal_tactician_benchmark_report(fixture_cohort_receipts())
    assert report.hard_gates_passed is True
    acceptance = report["acceptance"]
    assert acceptance["metrics_from_cohort_receipts"] is True
    assert acceptance["hard_correctness_privacy_authority_100_percent"] is True
    assert acceptance["timing_observational_unless_calibrated"] is True
    assert acceptance["cache_hits_preserve_authority_and_identity"] is True
    assert acceptance[
        "progress_exposes_holes_witnesses_critical_path_budgets_next_actions"
    ] is True

    gates = report["gates"]
    for name in HARD_GATE_NAMES:
        assert gates["hard"][name]["status"] == "pass"
        assert gates["hard"][name]["actual_bps"] == BASIS_POINTS
    assert gates["timing"]["status"] == "observational"
    assert gates["tool_availability"]["status"] == "not_applicable"

    # Failing hard gate flips report acceptance.
    poisoned = _receipt(
        receipt_id="receipt:poison-authority",
        claimed_assurance="kernel_verified",
        authoritative_assurance="unverified",
        authority_boundary_violation=True,
    )
    failing = build_goal_tactician_benchmark_report((poisoned,))
    assert failing.hard_gates_passed is False
    assert failing["acceptance"]["hard_correctness_privacy_authority_100_percent"] is False


def test_architecture_document_rejects_synthetic_flag() -> None:
    report = build_goal_tactician_benchmark_report(fixture_cohort_receipts())
    payload = report.to_dict()
    payload["synthetic_distributions"] = True
    with pytest.raises(GoalTacticianMetricsError, match="synthetic"):
        architecture_benchmark_document(
            type(report)(payload)  # type: ignore[misc]
        )


def test_rates_recomputed_from_additive_counts() -> None:
    metrics = derive_goal_tactician_metrics(fixture_cohort_receipts())
    payload = metrics.to_dict()
    precision = payload["proof_gap"]["precision_bps"]
    recomputed = (
        metrics.proof_gap_true_positive
        * BASIS_POINTS
        // (
            metrics.proof_gap_true_positive + metrics.proof_gap_false_positive
        )
    )
    assert precision == recomputed
    assert payload["proof_gap"]["precision_bps"] == metrics.proof_gap_precision_bps
    assert payload["cache"]["hit_rate_bps"] == metrics.cache_hit_rate_bps
    assert metrics.cache_hit_count == 1
    assert metrics.cache_miss_count == 1


def test_replace_cannot_introduce_synthetic_metrics() -> None:
    metrics = derive_goal_tactician_metrics(fixture_cohort_receipts())
    with pytest.raises(GoalTacticianMetricsError, match="synthetic"):
        replace(metrics, synthetic_distributions=True)
    with pytest.raises(GoalTacticianMetricsError, match="cohort_receipts"):
        replace(metrics, source="monte-carlo")
