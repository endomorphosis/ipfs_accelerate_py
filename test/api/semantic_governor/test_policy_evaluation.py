"""Tests for SCG-033 held-out policy evaluation.

Acceptance criteria enforced here:

* Missing held-out data rejects.
* Overlapping held-out / calibration / development / candidate-generating
  identities reject.
* Critical omission detection cannot regress versus baseline.
* Stale rejection cannot regress versus baseline.
* Hidden accepted regressions block.
* Evaluation emits a reproducible report without mutation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AssumptionKind,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.calibration_contracts import (
    EvidencePartition,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.policy_contracts import (
    CompressionPolicyCandidate,
    EvaluationVerdict,
    ProtectedThresholds,
    RuleEvaluationReport,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.policy_evaluation import (
    EVALUATE_RULE_CANDIDATE_INTERFACE,
    REASON_CANDIDATE_GENERATING_OVERLAP,
    REASON_HIDDEN_REGRESSION,
    REASON_MISSING_HELD_OUT,
    REASON_OMISSION_REGRESSION,
    REASON_OVERLAP,
    REASON_STALE_REGRESSION,
    SCG_HELD_OUT_EVALUATION_EVIDENCE,
    HeldOutBenchmark,
    HeldOutCaseOutcome,
    PolicyEvaluationError,
    compute_held_out_metrics,
    evaluate_rule_candidate,
    verify_evaluation_report_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/policy_evaluation.py"
)


# ---------------------------------------------------------------------------
# Fixtures / recipes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _header(artifact_kind: str, **overrides: Any) -> GovernorArtifactHeader:
    fields: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "repository_state_cid": _cid("repo-state"),
        "context_pack_cid": _cid("context-pack"),
        "verification_bundle_cid": _cid("verification-bundle"),
        "generator": GeneratorIdentity(
            generator_id="policy_contracts",
            generator_version="1.0.0",
            interface_id="propose_rule_change@1",
        ),
        "provenance": ArtifactProvenance(
            producer_id="semantic_governor",
            producer_version="1",
            execution_mode=ExecutionMode.LIVE,
            authority_source=AuthoritySource.DETERMINISTIC,
            input_cids=(_cid("input-a"),),
            tool_ids=("policy.v1",),
            policy_cid=_cid("policy-v1"),
            notes=None,
        ),
        "terminal_status": GovernorTerminalStatus.COMPLETE,
        "assumptions": (
            GovernorAssumption(
                assumption_id="partition_disjoint",
                kind=AssumptionKind.VERIFICATION,
                statement="Held-out partition is disjoint from calibration",
                supporting_cids=(_cid("partition"),),
            ),
        ),
        "metadata": {"track": "policy_evaluation"},
    }
    fields.update(overrides)
    return GovernorArtifactHeader(**fields)


def _thresholds(**overrides: Any) -> ProtectedThresholds:
    fields = {
        "min_critical_omission_detection_bp": 9_500,
        "max_critical_omission_accepted": 0,
        "min_median_context_reduction_bp": 5_000,
        "max_accepted_regression_bp": 0,
        "min_shadow_sample_rate_bp": 100,
        "require_full_suite_fallback": True,
        "allow_heuristic_as_exact": False,
        "allow_assurance_reduction": False,
    }
    fields.update(overrides)
    return ProtectedThresholds(**fields)


def _candidate(**overrides: Any) -> CompressionPolicyCandidate:
    fields: dict[str, Any] = {
        "header": _header("compression_policy_candidate"),
        "candidate_id": "cand_ok",
        "base_policy_cid": _cid("policy-v1"),
        "base_policy_version": "1.0.0",
        "proposal_cid": _cid("proposal-1"),
        "proposed_policy_cid": _cid("policy-v2"),
        "proposed_protected_thresholds": _thresholds(),
        "baseline_protected_thresholds": _thresholds(),
        "evaluation_partition": EvidencePartition.HELD_OUT,
        "external_authorization_cid": None,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return CompressionPolicyCandidate(**fields)


def _case(
    label: str,
    *,
    critical_omission_present: bool = False,
    critical_omission_detected: bool = False,
    critical_omission_accepted: bool = False,
    stale_artifact_present: bool = False,
    stale_artifact_rejected: bool = False,
    accepted_regression: bool = False,
    context_reduction_bp: int = 6_000,
) -> HeldOutCaseOutcome:
    return HeldOutCaseOutcome(
        case_id=f"case_{label}",
        case_cid=_cid(f"case-{label}"),
        partition=EvidencePartition.HELD_OUT,
        critical_omission_present=critical_omission_present,
        critical_omission_detected=critical_omission_detected,
        critical_omission_accepted=critical_omission_accepted,
        stale_artifact_present=stale_artifact_present,
        stale_artifact_rejected=stale_artifact_rejected,
        accepted_regression=accepted_regression,
        context_reduction_bp=context_reduction_bp,
    )


def _passing_cases() -> tuple[HeldOutCaseOutcome, ...]:
    """20 held-out cases: 20 omission probes all detected; 20 stale all rejected.

    20/20 detection = 10000 bp (>= 9500). Median context reduction 6000.
    Zero accepted regressions.
    """

    cases: list[HeldOutCaseOutcome] = []
    for index in range(20):
        cases.append(
            _case(
                f"omit_{index:02d}",
                critical_omission_present=True,
                critical_omission_detected=True,
                stale_artifact_present=True,
                stale_artifact_rejected=True,
                context_reduction_bp=6_000,
            )
        )
    return tuple(cases)


def _benchmark(
    cases: SequenceLike | None = None,
    **overrides: Any,
) -> HeldOutBenchmark:
    fields: dict[str, Any] = {
        "benchmark_id": "held_out_v1",
        "partition": EvidencePartition.HELD_OUT,
        "case_outcomes": list(cases if cases is not None else _passing_cases()),
        "calibration_case_cids": (_cid("cal-case-1"), _cid("cal-case-2")),
        "development_case_cids": (_cid("dev-case-1"),),
        "candidate_generating_case_cids": (_cid("cal-case-1"),),
        "baseline_critical_omission_detection_bp": 9_500,
        "baseline_stale_rejection_rate_bp": 10_000,
        "baseline_accepted_regression_bp": 0,
        "baseline_policy_cid": _cid("policy-v1"),
        "repository_state_cid": _cid("repo-state"),
        "context_pack_cid": _cid("context-pack"),
        "verification_bundle_cid": _cid("verification-bundle"),
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return HeldOutBenchmark(**fields)


# Type alias for readability in helpers.
SequenceLike = tuple[HeldOutCaseOutcome, ...] | list[HeldOutCaseOutcome]


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exports_evaluate_rule_candidate() -> None:
    assert MODULE_PATH.is_file()
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "def evaluate_rule_candidate" in source
    assert EVALUATE_RULE_CANDIDATE_INTERFACE == "evaluate_rule_candidate@1"
    assert SCG_HELD_OUT_EVALUATION_EVIDENCE == "scg/held-out-evaluation@1"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_evaluate_passing_candidate_on_disjoint_held_out() -> None:
    candidate = _candidate()
    benchmark = _benchmark()
    report = evaluate_rule_candidate(candidate, benchmark)

    assert isinstance(report, RuleEvaluationReport)
    assert report.verdict == EvaluationVerdict.PASS.value
    assert report.partition == EvidencePartition.HELD_OUT.value
    assert report.candidate_cid == candidate.candidate_cid
    assert report.held_out_benchmark_cid == benchmark.benchmark_cid
    assert report.baseline_policy_cid == candidate.base_policy_cid
    assert report.critical_omission_detection_bp == 10_000
    assert report.stale_rejection_rate_bp == 10_000
    assert report.accepted_regression_bp == 0
    assert report.high_risk_assurance_reduced is False
    assert report.declared_thresholds_applied is True
    assert report.blocking_reasons == ()
    assert report.metadata["evidence"] == SCG_HELD_OUT_EVALUATION_EVIDENCE


def test_evaluation_is_deterministic_and_idempotent() -> None:
    candidate = _candidate()
    benchmark = _benchmark()
    first = evaluate_rule_candidate(candidate, benchmark)
    second = evaluate_rule_candidate(candidate, benchmark)
    assert first.report_cid == second.report_cid
    assert first.to_dict() == second.to_dict()
    restored = RuleEvaluationReport.from_dict(first.to_dict())
    assert restored.report_cid == first.report_cid
    assert verify_evaluation_report_identity(first) == first.report_cid


def test_evaluation_does_not_mutate_inputs() -> None:
    candidate = _candidate()
    benchmark = _benchmark()
    cand_before = candidate.to_dict()
    bench_before = benchmark.to_dict()
    evaluate_rule_candidate(candidate, benchmark)
    assert candidate.to_dict() == cand_before
    assert benchmark.to_dict() == bench_before


def test_accepts_mapping_inputs() -> None:
    candidate = _candidate()
    benchmark = _benchmark()
    report = evaluate_rule_candidate(candidate.to_dict(), benchmark.to_dict())
    assert report.verdict == EvaluationVerdict.PASS.value


# ---------------------------------------------------------------------------
# Missing / overlapping held-out data
# ---------------------------------------------------------------------------


def test_missing_held_out_cases_rejects() -> None:
    report = evaluate_rule_candidate(
        _candidate(),
        _benchmark(cases=()),
    )
    assert report.verdict == EvaluationVerdict.REJECTED.value
    assert REASON_MISSING_HELD_OUT in report.blocking_reasons


def test_overlap_with_calibration_rejects() -> None:
    held = _case(
        "overlap_cal",
        critical_omission_present=True,
        critical_omission_detected=True,
        stale_artifact_present=True,
        stale_artifact_rejected=True,
    )
    # Force case_cid to match a calibration identity.
    overlapping = HeldOutCaseOutcome(
        case_id="case_overlap_cal",
        case_cid=_cid("cal-case-1"),
        partition=EvidencePartition.HELD_OUT,
        critical_omission_present=True,
        critical_omission_detected=True,
        stale_artifact_present=True,
        stale_artifact_rejected=True,
        context_reduction_bp=6_000,
    )
    report = evaluate_rule_candidate(
        _candidate(),
        _benchmark(cases=(overlapping, *list(_passing_cases())[:5])),
    )
    assert report.verdict == EvaluationVerdict.REJECTED.value
    assert REASON_OVERLAP in report.blocking_reasons
    # Also generating overlap (cal-case-1 is generating).
    assert REASON_CANDIDATE_GENERATING_OVERLAP in report.blocking_reasons


def test_overlap_with_development_rejects() -> None:
    overlapping = HeldOutCaseOutcome(
        case_id="case_overlap_dev",
        case_cid=_cid("dev-case-1"),
        partition=EvidencePartition.HELD_OUT,
        critical_omission_present=True,
        critical_omission_detected=True,
        stale_artifact_present=True,
        stale_artifact_rejected=True,
        context_reduction_bp=6_000,
    )
    report = evaluate_rule_candidate(
        _candidate(),
        _benchmark(cases=(overlapping,)),
    )
    assert report.verdict == EvaluationVerdict.REJECTED.value
    assert REASON_OVERLAP in report.blocking_reasons


def test_candidate_generating_case_cannot_score_promotion() -> None:
    generating = HeldOutCaseOutcome(
        case_id="case_gen",
        case_cid=_cid("gen-only"),
        partition=EvidencePartition.HELD_OUT,
        critical_omission_present=True,
        critical_omission_detected=True,
        stale_artifact_present=True,
        stale_artifact_rejected=True,
        context_reduction_bp=6_000,
    )
    report = evaluate_rule_candidate(
        _candidate(),
        _benchmark(
            cases=(generating,),
            candidate_generating_case_cids=(_cid("gen-only"),),
        ),
    )
    assert report.verdict == EvaluationVerdict.REJECTED.value
    assert REASON_CANDIDATE_GENERATING_OVERLAP in report.blocking_reasons


def test_case_partition_must_be_held_out() -> None:
    with pytest.raises(PolicyEvaluationError, match="held_out"):
        HeldOutCaseOutcome(
            case_id="case_dev",
            case_cid=_cid("case-dev"),
            partition=EvidencePartition.DEVELOPMENT,
        )


def test_benchmark_partition_must_be_held_out() -> None:
    with pytest.raises(PolicyEvaluationError, match="held_out"):
        HeldOutBenchmark(
            benchmark_id="bad",
            partition=EvidencePartition.CALIBRATION,
            case_outcomes=list(_passing_cases()[:1]),
        )


def test_duplicate_held_out_case_identities_reject() -> None:
    case_a = _case(
        "dup",
        critical_omission_present=True,
        critical_omission_detected=True,
        stale_artifact_present=True,
        stale_artifact_rejected=True,
    )
    # Same case_cid, different case_id not allowed in sorted set for cids.
    case_b = HeldOutCaseOutcome(
        case_id="case_dup_b",
        case_cid=case_a.case_cid,
        partition=EvidencePartition.HELD_OUT,
        critical_omission_present=True,
        critical_omission_detected=True,
        stale_artifact_present=True,
        stale_artifact_rejected=True,
        context_reduction_bp=6_000,
    )
    report = evaluate_rule_candidate(
        _candidate(),
        _benchmark(cases=(case_a, case_b)),
    )
    assert report.verdict == EvaluationVerdict.REJECTED.value
    assert "duplicate_held_out_case_identity" in report.blocking_reasons


# ---------------------------------------------------------------------------
# Non-regression: omission detection and stale rejection
# ---------------------------------------------------------------------------


def test_critical_omission_detection_regression_blocks() -> None:
    # 18/20 = 9000 bp < baseline 9500.
    cases: list[HeldOutCaseOutcome] = []
    for index in range(20):
        detected = index < 18
        cases.append(
            _case(
                f"omit_reg_{index:02d}",
                critical_omission_present=True,
                critical_omission_detected=detected,
                critical_omission_accepted=not detected,
                stale_artifact_present=True,
                stale_artifact_rejected=True,
                context_reduction_bp=6_000,
            )
        )
    report = evaluate_rule_candidate(_candidate(), _benchmark(cases=cases))
    assert report.verdict == EvaluationVerdict.FAIL.value
    assert REASON_OMISSION_REGRESSION in report.blocking_reasons
    assert report.critical_omission_detection_bp == 9_000


def test_stale_rejection_regression_blocks() -> None:
    # Omission perfect; stale 19/20 = 9500 < baseline 10000.
    cases: list[HeldOutCaseOutcome] = []
    for index in range(20):
        rejected = index < 19
        cases.append(
            _case(
                f"stale_reg_{index:02d}",
                critical_omission_present=True,
                critical_omission_detected=True,
                stale_artifact_present=True,
                stale_artifact_rejected=rejected,
                context_reduction_bp=6_000,
            )
        )
    report = evaluate_rule_candidate(_candidate(), _benchmark(cases=cases))
    assert report.verdict == EvaluationVerdict.FAIL.value
    assert REASON_STALE_REGRESSION in report.blocking_reasons
    assert report.stale_rejection_rate_bp == 9_500


def test_omission_and_stale_non_regression_pass_at_baseline() -> None:
    # Exactly meet baseline 9500 omission (19/20) with baseline lowered.
    cases: list[HeldOutCaseOutcome] = []
    for index in range(20):
        detected = index < 19
        cases.append(
            _case(
                f"edge_{index:02d}",
                critical_omission_present=True,
                critical_omission_detected=detected,
                critical_omission_accepted=not detected,
                stale_artifact_present=True,
                stale_artifact_rejected=True,
                context_reduction_bp=6_000,
            )
        )
    # Thresholds and baseline both at 9500; one accepted omission fails
    # max_critical_omission_accepted=0.
    report = evaluate_rule_candidate(
        _candidate(
            proposed_protected_thresholds=_thresholds(
                min_critical_omission_detection_bp=9_500,
                max_critical_omission_accepted=1,
            ),
            baseline_protected_thresholds=_thresholds(
                min_critical_omission_detection_bp=9_500,
                max_critical_omission_accepted=1,
            ),
        ),
        _benchmark(
            cases=cases,
            baseline_critical_omission_detection_bp=9_500,
            baseline_stale_rejection_rate_bp=10_000,
        ),
    )
    assert report.critical_omission_detection_bp == 9_500
    assert report.stale_rejection_rate_bp == 10_000
    assert report.verdict == EvaluationVerdict.PASS.value


# ---------------------------------------------------------------------------
# Hidden accepted regressions
# ---------------------------------------------------------------------------


def test_hidden_accepted_regression_blocks() -> None:
    cases = list(_passing_cases())
    cases[0] = HeldOutCaseOutcome(
        case_id=cases[0].case_id,
        case_cid=cases[0].case_cid,
        partition=EvidencePartition.HELD_OUT,
        critical_omission_present=True,
        critical_omission_detected=True,
        stale_artifact_present=True,
        stale_artifact_rejected=True,
        accepted_regression=True,
        context_reduction_bp=6_000,
    )
    report = evaluate_rule_candidate(_candidate(), _benchmark(cases=cases))
    assert report.verdict == EvaluationVerdict.FAIL.value
    assert REASON_HIDDEN_REGRESSION in report.blocking_reasons
    assert report.accepted_regression_bp > 0


def test_even_single_hidden_regression_blocks_with_zero_threshold() -> None:
    cases = list(_passing_cases())
    # Inject regression into last case.
    last = cases[-1]
    cases[-1] = HeldOutCaseOutcome(
        case_id=last.case_id,
        case_cid=last.case_cid,
        partition=EvidencePartition.HELD_OUT,
        critical_omission_present=True,
        critical_omission_detected=True,
        stale_artifact_present=True,
        stale_artifact_rejected=True,
        accepted_regression=True,
        context_reduction_bp=6_000,
    )
    report = evaluate_rule_candidate(_candidate(), _benchmark(cases=cases))
    assert REASON_HIDDEN_REGRESSION in report.blocking_reasons
    assert report.verdict != EvaluationVerdict.PASS.value


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def test_compute_held_out_metrics_rates() -> None:
    cases = (
        _case(
            "a",
            critical_omission_present=True,
            critical_omission_detected=True,
            stale_artifact_present=True,
            stale_artifact_rejected=True,
            context_reduction_bp=4_000,
        ),
        _case(
            "b",
            critical_omission_present=True,
            critical_omission_detected=False,
            critical_omission_accepted=True,
            stale_artifact_present=True,
            stale_artifact_rejected=False,
            accepted_regression=True,
            context_reduction_bp=6_000,
        ),
    )
    metrics = compute_held_out_metrics(cases)
    assert metrics.case_count == 2
    assert metrics.critical_omission_detection_bp == 5_000
    assert metrics.stale_rejection_rate_bp == 5_000
    assert metrics.accepted_regression_bp == 5_000
    assert metrics.critical_omission_accepted_count == 1
    assert metrics.median_context_reduction_bp == 5_000


# ---------------------------------------------------------------------------
# Integrity / contract edges
# ---------------------------------------------------------------------------


def test_baseline_policy_cid_mismatch_raises() -> None:
    with pytest.raises(PolicyEvaluationError, match="baseline_policy_cid"):
        evaluate_rule_candidate(
            _candidate(),
            _benchmark(baseline_policy_cid=_cid("other-policy")),
        )


def test_benchmark_round_trip_identity() -> None:
    bench = _benchmark()
    restored = HeldOutBenchmark.from_dict(bench.to_dict())
    assert restored.benchmark_cid == bench.benchmark_cid


def test_forged_benchmark_cid_fails_closed() -> None:
    bench = _benchmark()
    payload = bench.to_dict()
    payload["benchmark_cid"] = _cid("forged")
    with pytest.raises(PolicyEvaluationError, match="does not verify"):
        HeldOutBenchmark.from_dict(payload)


def test_context_reduction_below_threshold_fails() -> None:
    cases = [
        _case(
            f"cheap_{index:02d}",
            critical_omission_present=True,
            critical_omission_detected=True,
            stale_artifact_present=True,
            stale_artifact_rejected=True,
            context_reduction_bp=1_000,
        )
        for index in range(20)
    ]
    report = evaluate_rule_candidate(_candidate(), _benchmark(cases=cases))
    assert report.verdict == EvaluationVerdict.FAIL.value
    assert "median_context_reduction_below_threshold" in report.blocking_reasons


def test_critical_omission_accepted_blocks() -> None:
    cases = list(_passing_cases())
    first = cases[0]
    cases[0] = HeldOutCaseOutcome(
        case_id=first.case_id,
        case_cid=first.case_cid,
        partition=EvidencePartition.HELD_OUT,
        critical_omission_present=True,
        critical_omission_detected=False,
        critical_omission_accepted=True,
        stale_artifact_present=True,
        stale_artifact_rejected=True,
        context_reduction_bp=6_000,
    )
    report = evaluate_rule_candidate(_candidate(), _benchmark(cases=cases))
    assert report.verdict == EvaluationVerdict.FAIL.value
    assert "critical_omission_accepted" in report.blocking_reasons


def test_invalid_candidate_type_raises() -> None:
    with pytest.raises(PolicyEvaluationError, match="candidate must be"):
        evaluate_rule_candidate("not-a-candidate", _benchmark())  # type: ignore[arg-type]


def test_invalid_benchmark_type_raises() -> None:
    with pytest.raises(PolicyEvaluationError, match="held_out_benchmark must be"):
        evaluate_rule_candidate(_candidate(), "not-a-bench")  # type: ignore[arg-type]


def test_case_outcome_rejects_detected_without_present() -> None:
    with pytest.raises(PolicyEvaluationError, match="critical_omission_detected"):
        HeldOutCaseOutcome(
            case_id="bad",
            case_cid=_cid("bad"),
            partition=EvidencePartition.HELD_OUT,
            critical_omission_present=False,
            critical_omission_detected=True,
        )


def test_report_metadata_includes_metrics() -> None:
    report = evaluate_rule_candidate(_candidate(), _benchmark())
    metrics = report.metadata["metrics"]
    assert metrics["case_count"] == 20
    assert metrics["critical_omission_detection_bp"] == 10_000
    assert metrics["stale_rejection_rate_bp"] == 10_000


def test_deep_copy_benchmark_still_evaluates() -> None:
    """Ensure mapping path does not rely on object identity mutation."""

    benchmark = _benchmark()
    payload = copy.deepcopy(benchmark.to_dict())
    report = evaluate_rule_candidate(_candidate().to_dict(), payload)
    assert report.verdict == EvaluationVerdict.PASS.value
