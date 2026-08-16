"""SCA-181 runtime contract evaluation (mutations, ZK, release aggregation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.runtime_contract_evaluation import (
    APPROVED_ZK_PREDICATE,
    ChildDisposition,
    DetectionOutcome,
    GOAL_ID,
    HeldOutMutationCase,
    MutationKind,
    RUNTIME_CONTRACT_EVALUATION_INTERFACE,
    RUNTIME_REPORT_SCHEMA,
    ReleaseChild,
    ReleaseVerdict,
    RuntimeContractEvaluationError,
    TASK_ID,
    ZkAttestationAttempt,
    ZkBackendKind,
    aggregate_release,
    default_held_out_suite,
    evaluate_mutation,
    evaluate_runtime_contracts,
    evaluate_zk_attestation,
    write_runtime_report,
)


def _superproject_root() -> Path | None:
    for candidate in (Path.cwd().resolve(), *Path(__file__).resolve().parents):
        if (candidate / "config/swissknife_symbolic_contract_scope.json").is_file():
            return candidate
    return None


REPO = _superproject_root()
EVAL_DIR = (
    (REPO or Path("/__missing__"))
    / "data/agent_supervisor/swissknife_contract_assurance/evaluation"
)
RUNTIME_REPORT = EVAL_DIR / "runtime_report.json"
BASELINE_SUMMARY = (
    (REPO or Path("/__missing__"))
    / "data/agent_supervisor/swissknife_contract_assurance/baseline/"
    "runtime_components/summary.json"
)

requires_repo = pytest.mark.skipif(REPO is None, reason="needs SwissKnife superproject")


def test_default_held_out_suite_covers_mandatory_mutations() -> None:
    suite = default_held_out_suite()
    kinds = {case.mutation for case in suite}
    for required in MutationKind:
        assert required in kinds
    assert all(case.partition == "held_out" for case in suite)
    assert all(not case.to_dict()["provider_context_includes_witness"] for case in suite)
    assert all(case.private_witness_ref.startswith("private://") for case in suite)


def test_detector_matches_expected_and_never_calls_models() -> None:
    case = HeldOutMutationCase(
        case_id="mut:policy",
        mutation=MutationKind.POLICY,
        expected_detection=DetectionOutcome.DETECTED,
        private_witness_ref="private://held-out/mut:policy",
    )
    obs = evaluate_mutation(case)
    assert obs.outcome is DetectionOutcome.DETECTED
    assert obs.model_call_count == 0
    assert obs.provider_call_count == 0
    assert obs.llm_call_count == 0
    assert obs.authority_granted is False


def test_missed_mandatory_detection_is_recorded() -> None:
    case = HeldOutMutationCase(
        case_id="mut:forged-receipt",
        mutation=MutationKind.FORGED_RECEIPT,
        expected_detection=DetectionOutcome.DETECTED,
    )
    obs = evaluate_mutation(case, detector_outcome=DetectionOutcome.MISSED)
    assert obs.outcome is DetectionOutcome.MISSED
    assert "mandatory_held_out_missed" in obs.reason_codes


def test_mock_evidence_cannot_grant_authority() -> None:
    case = HeldOutMutationCase(
        case_id="mut:mock-evidence",
        mutation=MutationKind.MOCK_EVIDENCE,
        expected_detection=DetectionOutcome.DETECTED,
    )
    obs = evaluate_mutation(
        case,
        detector_outcome=DetectionOutcome.DETECTED,
        authority_granted=True,
    )
    # Detection is fine, but mock authority is stripped / marked false admission
    # when mutation is mock_evidence with authority_granted.
    # Our policy forces FALSE_AUTHORITY for mock+authority.
    assert obs.outcome is DetectionOutcome.FALSE_AUTHORITY
    assert obs.authority_granted is False


def test_nonzero_model_calls_fail_closed() -> None:
    case = default_held_out_suite()[0]
    with pytest.raises(RuntimeContractEvaluationError, match="nonzero"):
        evaluate_mutation(case, model_call_count=1)


def test_simulated_zk_never_attests() -> None:
    result = evaluate_zk_attestation(
        ZkAttestationAttempt(
            backend=ZkBackendKind.SIMULATED,
            predicate=APPROVED_ZK_PREDICATE,
            receipt_root="baguqeera:test",
            required=False,
            capability_ready=True,
        )
    )
    assert result.attested is False
    assert result.blocks_release is False
    assert "simulated_zk_non_attested" in result.reason_codes


def test_required_simulated_zk_blocks_release() -> None:
    result = evaluate_zk_attestation(
        ZkAttestationAttempt(
            backend=ZkBackendKind.SIMULATED,
            predicate=APPROVED_ZK_PREDICATE,
            receipt_root="baguqeera:test",
            required=True,
        )
    )
    assert result.attested is False
    assert result.blocks_release is True


def test_real_zk_only_attests_verified_receipt_when_ready() -> None:
    bad = evaluate_zk_attestation(
        ZkAttestationAttempt(
            backend=ZkBackendKind.REAL,
            predicate="anything_else",
            receipt_root="baguqeera:test",
            capability_ready=True,
        )
    )
    assert bad.attested is False

    good = evaluate_zk_attestation(
        ZkAttestationAttempt(
            backend=ZkBackendKind.REAL,
            predicate=APPROVED_ZK_PREDICATE,
            receipt_root="baguqeera:test",
            capability_ready=True,
        )
    )
    assert good.attested is True
    assert good.blocks_release is False


def test_release_aggregation_fails_closed_on_stale_mock_degraded_nogo() -> None:
    root = "content-root:sca-181"
    for disposition in (
        ChildDisposition.NO_GO,
        ChildDisposition.STALE,
        ChildDisposition.MOCK,
        ChildDisposition.DEGRADED,
    ):
        result = aggregate_release(
            (
                ReleaseChild("ok", ChildDisposition.GO, content_root=root),
                ReleaseChild("bad", disposition, content_root=root),
            ),
            content_root=root,
        )
        assert result.verdict is ReleaseVerdict.NO_GO
        assert any(disposition.value in code for code in result.reason_codes)


def test_release_go_only_when_all_children_go() -> None:
    root = "content-root:sca-181"
    result = aggregate_release(
        (
            ReleaseChild("a", ChildDisposition.GO, content_root=root),
            ReleaseChild("b", ChildDisposition.GO, content_root=root),
        ),
        content_root=root,
    )
    assert result.verdict is ReleaseVerdict.GO
    assert result.reason_codes == ()


def test_cross_root_child_fails_closed() -> None:
    result = aggregate_release(
        (ReleaseChild("a", ChildDisposition.GO, content_root="other-root"),),
        content_root="content-root:sca-181",
    )
    assert result.verdict is ReleaseVerdict.NO_GO
    assert any("cross_root" in code for code in result.reason_codes)


def test_full_evaluation_passes_on_healthy_defaults() -> None:
    report = evaluate_runtime_contracts(content_root="content-root:sca-181-healthy")
    assert report.passed is True
    assert report.reason_codes == ()
    assert report.release.verdict is ReleaseVerdict.GO
    assert report.model_call_count == 0
    payload = report.to_dict()
    assert payload["schema"] == RUNTIME_REPORT_SCHEMA
    assert payload["interface"] == RUNTIME_CONTRACT_EVALUATION_INTERFACE
    assert payload["goal_id"] == GOAL_ID
    assert payload["task_id"] == TASK_ID
    gates = payload["safety_gates"]
    assert gates["all_mandatory_held_out_detected_or_unsupported"] is True
    assert gates["zero_false_authoritative_admissions"] is True
    assert gates["release_fail_closed_on_bad_children"] is True
    assert gates["simulated_zk_never_attests"] is True


def test_full_evaluation_fails_on_missed_held_out() -> None:
    suite = default_held_out_suite()[:1]
    obs = [
        evaluate_mutation(
            suite[0],
            detector_outcome=DetectionOutcome.MISSED,
        )
    ]
    report = evaluate_runtime_contracts(
        content_root="content-root:sca-181",
        cases=suite,
        observations=obs,
    )
    assert report.passed is False
    assert any("missed" in code for code in report.reason_codes)


def test_full_evaluation_with_simulated_required_zk_is_no_go() -> None:
    report = evaluate_runtime_contracts(
        content_root="content-root:sca-181",
        zk_attempt=ZkAttestationAttempt(
            backend=ZkBackendKind.SIMULATED,
            predicate=APPROVED_ZK_PREDICATE,
            receipt_root="baguqeera:test",
            required=True,
        ),
    )
    # Release is NO_GO due to required simulated zk, but safety gate for
    # "simulated never attests" still holds (attested=false).
    assert report.release.verdict is ReleaseVerdict.NO_GO
    assert report.release.zk is not None
    assert report.release.zk.attested is False
    assert report.to_dict()["safety_gates"]["simulated_zk_never_attests"] is True


@requires_repo
def test_publish_runtime_report_against_current_baseline(tmp_path: Path) -> None:
    content_root = "content-root:sca-181-current"
    if BASELINE_SUMMARY.is_file():
        summary = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
        content_root = str(
            summary.get("snapshot_id")
            or summary.get("index_id")
            or content_root
        )
        assert summary.get("health_status") == "healthy"
        assert summary.get("llm_call_count") == 0

    report = evaluate_runtime_contracts(
        content_root=content_root,
        zk_attempt=ZkAttestationAttempt(
            backend=ZkBackendKind.UNAVAILABLE,
            predicate=APPROVED_ZK_PREDICATE,
            receipt_root="",
            required=False,
            capability_ready=False,
        ),
        extra={
            "baseline_runtime_components": str(BASELINE_SUMMARY),
            "baseline_health": (
                json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8")).get(
                    "health_status"
                )
                if BASELINE_SUMMARY.is_file()
                else "unknown"
            ),
        },
    )
    assert report.passed is True

    # Write canonical evaluation artifact
    EVAL_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_runtime_report(RUNTIME_REPORT, report)
    assert RUNTIME_REPORT.is_file()
    sealed = json.loads(RUNTIME_REPORT.read_text(encoding="utf-8"))
    assert sealed["passed"] is True
    assert sealed["schema"] == RUNTIME_REPORT_SCHEMA
    assert sealed["isolation_audit"]["held_out_witnesses_in_provider_context"] is False
    assert sealed["isolation_audit"]["llm_call_count"] == 0
    assert sealed["report_id"].startswith("sca-runtime-eval:sha256:")

    # Also prove write path works under tmp
    alt = tmp_path / "runtime_report.json"
    write_runtime_report(alt, report)
    assert alt.is_file()
