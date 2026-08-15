"""Tests for individual mutant execution and closed outcome classification (AAE-044).

Acceptance criteria enforced here:

* Unmutated baseline is green or explicitly blocked.
* Predicted checks run first.
* Broader fallback is policy-bound (never silent).
* Observed detectors and one closed terminal outcome are persisted honestly.
* Invalid/uncompilable/infrastructure/timeout/equivalent never count as killed.
* Cold import is side-effect free; no production policy change.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.execution import (
    AAE_OUTCOME_EVIDENCE,
    CLASSIFY_MUTATION_OUTCOME_INTERFACE,
    EXECUTE_MUTATION_INTERFACE,
    FULL_SUITE_DETECTOR_ID,
    REASON_BASELINE_BLOCKED,
    REASON_BASELINE_GREEN,
    REASON_FALLBACK_DISABLED,
    REASON_FALLBACK_POLICY_BOUND,
    REASON_FALLBACK_SKIPPED_KILL,
    REASON_OBSERVED_HONEST,
    REASON_ONE_TERMINAL_OUTCOME,
    REASON_PREDICTED_FIRST,
    BaselineGateError,
    BaselineGateStatus,
    DetectorRunObservation,
    DetectorRunStatus,
    ExecutionBaseline,
    ExecutionDisposition,
    ExecutionFallbackPolicy,
    ExecutionPhase,
    MutationExecutionError,
    MutationExecutionReport,
    classify_mutation_outcome,
    closed_mutation_outcome_statuses,
    evaluate_baseline_gate,
    execute_mutation,
    execute_mutation_descriptor,
    resolve_execution_fallback,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.incremental import (
    BroadeningMode,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    ArtifactProvenance,
    AssuranceArtifactHeader,
    AssuranceTerminalStatus,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    VersionBinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    DetectorKind,
    DetectorPrediction,
    DetectorStrength,
    ExpectedDetectionSet,
    MutationOutcomeStatus,
    assert_outcome_never_false_kill,
    counts_as_killed,
    verify_outcome_identity,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes


REPO_ROOT = Path(__file__).resolve().parents[3]
EXECUTION_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/execution.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


REPO_ID = "repository:sha256:test-repo-identity-aae044"
REPO_STATE = _cid("repo-state-aae044")
ENV_CID = _cid("environment-aae044")
DEP_LOCK = _cid("dependency-lock-aae044")
BASELINE_RECEIPT = _cid("baseline-receipt-aae044")
POLICY_CID = _cid("policy-aae044")
CANDIDATE_CID = _cid("candidate-aae044")
MUTANT_CID = _cid("mutant-identity-aae044")
CANDIDATE_ID = "cand_control_flow_invert_0"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _generator(**overrides: object) -> GeneratorIdentity:
    fields = {
        "generator_id": "mutation_execution",
        "generator_version": "1.0.0",
        "interface_id": EXECUTE_MUTATION_INTERFACE,
    }
    fields.update(overrides)
    return GeneratorIdentity(**fields)  # type: ignore[arg-type]


def _versions(**overrides: object) -> VersionBinding:
    fields = {
        "operator_id": "control_flow_invert",
        "operator_version": "1",
        "campaign_policy_id": "default_campaign",
        "campaign_policy_version": "1.0.0",
        "generator": _generator(),
    }
    fields.update(overrides)
    return VersionBinding(**fields)  # type: ignore[arg-type]


def _provenance(**overrides: object) -> ArtifactProvenance:
    fields = {
        "producer_id": "adversarial_assurance",
        "producer_version": "1",
        "execution_mode": ExecutionMode.LIVE,
        "authority_source": AuthoritySource.OBSERVED,
        "input_cids": (_cid("input-a"),),
        "tool_ids": ("mutation_executor.v1",),
        "policy_cid": POLICY_CID,
        "notes": None,
    }
    fields.update(overrides)
    return ArtifactProvenance(**fields)  # type: ignore[arg-type]


def _header(artifact_kind: str, **overrides: object) -> AssuranceArtifactHeader:
    fields = {
        "artifact_kind": artifact_kind,
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "target_symbol_ids": ("mod.fn",),
        "target_artifact_cids": (_cid("artifact-a"),),
        "capsule_cids": (_cid("capsule-a"),),
        "proof_unit_cids": (_cid("proof-unit-a"),),
        "environment_cid": ENV_CID,
        "dependency_lock_cid": DEP_LOCK,
        "versions": _versions(),
        "provenance": _provenance(),
        "terminal_status": AssuranceTerminalStatus.COMPLETE,
        "receipt_cids": (BASELINE_RECEIPT,),
        "proof_cids": (_cid("proof-a"),),
        "metadata": {"risk_class": "local_bug"},
    }
    fields.update(overrides)
    return AssuranceArtifactHeader(**fields)  # type: ignore[arg-type]


def _prediction(
    detector_id: str = "unit.test_branch",
    kind: DetectorKind = DetectorKind.UNIT_TEST,
    **overrides: object,
) -> DetectorPrediction:
    fields = {
        "detector_id": detector_id,
        "detector_kind": kind,
        "violated_claim": "branch predicate must preserve control invariant",
        "observation_rationale": "test asserts inverted branch is rejected",
        "dependency_path": ("mod.fn", detector_id),
        "strength": DetectorStrength.REQUIRED,
        "expected_terminal_status": AssuranceTerminalStatus.COMPLETE,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return DetectorPrediction(**fields)  # type: ignore[arg-type]


def _detection_set(
    *detectors: DetectorPrediction, **overrides: object
) -> ExpectedDetectionSet:
    preds = detectors or (
        _prediction("static.authz_rule", DetectorKind.STATIC_RULE),
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    fields = {
        "header": _header("expected_detection_set"),
        "detection_set_id": "eds_cand_1",
        "candidate_id": CANDIDATE_ID,
        "candidate_cid": CANDIDATE_CID,
        "predicted_detectors": preds,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return ExpectedDetectionSet(**fields)  # type: ignore[arg-type]


def _green_baseline(**overrides: object) -> ExecutionBaseline:
    fields = {
        "baseline_receipt_cid": BASELINE_RECEIPT,
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "status": BaselineGateStatus.GREEN,
        "unmutated": True,
        "verification_green": True,
        "observation_complete": True,
    }
    fields.update(overrides)
    return ExecutionBaseline(**fields)  # type: ignore[arg-type]


def _blocked_baseline(**overrides: object) -> ExecutionBaseline:
    fields = {
        "baseline_receipt_cid": BASELINE_RECEIPT,
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "status": BaselineGateStatus.BLOCKED,
        "unmutated": True,
        "verification_green": False,
        "observation_complete": True,
        "block_reason": "unmutated suite failed on baseline target",
    }
    fields.update(overrides)
    return ExecutionBaseline(**fields)  # type: ignore[arg-type]


def _obs(
    detector_id: str,
    kind: DetectorKind | str,
    status: DetectorRunStatus | str,
    *,
    phase: ExecutionPhase = ExecutionPhase.PREDICTED,
    cost: int = 1,
) -> DetectorRunObservation:
    return DetectorRunObservation(
        detector_id=detector_id,
        detector_kind=kind,
        status=status,
        phase=phase,
        cost_units=cost,
        execution_seconds=1,
    )


def _runner(results: dict[str, DetectorRunStatus | str]):
    def run(
        detector_id: str,
        kind: DetectorKind | str,
        phase: ExecutionPhase,
    ) -> DetectorRunObservation:
        status = results.get(detector_id, DetectorRunStatus.PASSED)
        return DetectorRunObservation(
            detector_id=detector_id,
            detector_kind=kind,
            status=status,
            phase=phase,
            cost_units=2,
            execution_seconds=1,
        )

    return run


# ---------------------------------------------------------------------------
# Cold import / descriptors
# ---------------------------------------------------------------------------


def test_cold_import_is_side_effect_free() -> None:
    source = EXECUTION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Module body must not call network/process/filesystem helpers at import.
    forbidden_calls = {
        "open",
        "Popen",
        "run",
        "urlopen",
        "Thread",
        "Process",
    }
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            assert name not in forbidden_calls


def test_descriptor_exposes_interfaces_and_evidence() -> None:
    desc = execute_mutation_descriptor()
    assert desc["interface_id"] == EXECUTE_MUTATION_INTERFACE
    assert desc["classify_interface_id"] == CLASSIFY_MUTATION_OUTCOME_INTERFACE
    assert desc["evidence_subset"] == AAE_OUTCOME_EVIDENCE
    assert desc["production_policy_changed"] is False
    assert "predicted_checks_first" in desc["acceptance"]
    statuses = closed_mutation_outcome_statuses()
    assert "killed_by_test" in statuses
    assert "survived_selected_verification" in statuses
    assert "infrastructure_failure" in statuses


# ---------------------------------------------------------------------------
# Baseline gate: green or explicitly blocked
# ---------------------------------------------------------------------------


def test_green_baseline_passes_gate() -> None:
    sealed, reasons = evaluate_baseline_gate(_green_baseline())
    assert sealed.is_green is True
    assert sealed.is_blocked is False
    assert REASON_BASELINE_GREEN in reasons
    assert sealed.baseline_cid


def test_explicitly_blocked_baseline_is_admitted_as_blocked() -> None:
    sealed, reasons = evaluate_baseline_gate(_blocked_baseline())
    assert sealed.is_blocked is True
    assert sealed.is_green is False
    assert REASON_BASELINE_BLOCKED in reasons
    assert sealed.block_reason is not None


def test_incomplete_or_false_green_baseline_fails_closed() -> None:
    with pytest.raises(BaselineGateError, match="complete"):
        ExecutionBaseline(
            baseline_receipt_cid=BASELINE_RECEIPT,
            repository_id=REPO_ID,
            repository_state_cid=REPO_STATE,
            status=BaselineGateStatus.GREEN,
            observation_complete=False,
        )
    with pytest.raises(BaselineGateError, match="unmutated"):
        _green_baseline(unmutated=False)
    with pytest.raises(BaselineGateError, match="green"):
        _green_baseline(verification_green=False)
    with pytest.raises(BaselineGateError, match="block_reason"):
        ExecutionBaseline(
            baseline_receipt_cid=BASELINE_RECEIPT,
            repository_id=REPO_ID,
            repository_state_cid=REPO_STATE,
            status=BaselineGateStatus.BLOCKED,
            verification_green=False,
            block_reason=None,
        )
    with pytest.raises(BaselineGateError, match="verification_green"):
        _blocked_baseline(verification_green=True)


def test_execute_mutation_blocks_when_baseline_blocked() -> None:
    report = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_blocked_baseline(),
        expected_detection_set=_detection_set(),
        observations=(),
    )
    assert report.disposition == ExecutionDisposition.BLOCKED.value
    assert report.outcome is not None
    assert report.outcome.outcome_status == (
        MutationOutcomeStatus.INCONCLUSIVE.value
    )
    assert report.outcome.counts_as_killed is False
    assert report.observations == ()
    assert REASON_BASELINE_BLOCKED in report.reason_codes
    assert ExecutionPhase.CLASSIFIED.value in report.phases
    # Predicted phase must not run when blocked.
    assert ExecutionPhase.PREDICTED.value not in report.phases


# ---------------------------------------------------------------------------
# Predicted checks first; kill classification
# ---------------------------------------------------------------------------


def test_predicted_checks_run_first_and_kill_by_test() -> None:
    eds = _detection_set(
        _prediction("static.authz_rule", DetectorKind.STATIC_RULE),
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    order: list[str] = []

    def runner(detector_id: str, kind, phase):
        order.append(f"{phase.value}:{detector_id}")
        status = (
            DetectorRunStatus.DETECTED
            if detector_id == "unit.test_branch"
            else DetectorRunStatus.PASSED
        )
        return DetectorRunObservation(
            detector_id=detector_id,
            detector_kind=kind,
            status=status,
            phase=phase,
        )

    report = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=runner,
        fallback_policy=ExecutionFallbackPolicy(
            enable_broader_fallback=True,
            enable_full_suite_fallback=True,
        ),
        broader_detector_ids=("unit.extra_suite",),
    )
    assert order[0].startswith("predicted:")
    assert order[0].endswith("static.authz_rule") or order[0].endswith(
        "unit.test_branch"
    )
    # Both predicted detectors should run before any broader id when kill is
    # late in predicted set; static passes first, unit kills, broader skipped.
    assert all(not item.startswith("broader:") for item in order)
    assert REASON_PREDICTED_FIRST in report.reason_codes
    assert REASON_FALLBACK_SKIPPED_KILL in report.reason_codes
    assert report.outcome is not None
    assert report.outcome.outcome_status == MutationOutcomeStatus.KILLED_BY_TEST.value
    assert report.outcome.counts_as_killed is True
    assert report.outcome.killing_detector_id == "unit.test_branch"
    assert report.outcome.killing_detector_kind == DetectorKind.UNIT_TEST.value
    assert_outcome_never_false_kill(report.outcome)
    verify_outcome_identity(report.outcome)
    # Honest roles: observed ⊆ executed ⊆ selected
    clf = report.outcome.detector_classification
    assert "unit.test_branch" in clf.observed_detector_ids
    assert set(clf.observed_detector_ids) <= set(clf.executed_detector_ids)
    assert set(clf.executed_detector_ids) <= set(clf.selected_detector_ids)
    assert REASON_OBSERVED_HONEST in report.classification.reason_codes  # type: ignore[union-attr]
    assert report.production_policy_changed is False


def test_killed_by_static_analysis_from_first_observation() -> None:
    eds = _detection_set(
        _prediction("static.authz_rule", DetectorKind.STATIC_RULE),
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    report = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=_runner(
            {
                "static.authz_rule": DetectorRunStatus.DETECTED,
                "unit.test_branch": DetectorRunStatus.PASSED,
            }
        ),
    )
    assert report.outcome is not None
    assert report.outcome.outcome_status == (
        MutationOutcomeStatus.KILLED_BY_STATIC_ANALYSIS.value
    )
    assert report.outcome.killing_detector_id == "static.authz_rule"
    # Unit test must not execute after static kill within the phase batch stop.
    executed = report.outcome.detector_classification.executed_detector_ids
    assert "static.authz_rule" in executed
    assert "unit.test_branch" not in executed


def test_kill_kind_mapping_for_policy_and_proof() -> None:
    for kind, status in (
        (DetectorKind.POLICY_RULE, MutationOutcomeStatus.KILLED_BY_POLICY),
        (
            DetectorKind.FORMAL_OBLIGATION,
            MutationOutcomeStatus.KILLED_BY_FORMAL_PROOF,
        ),
        (
            DetectorKind.RUNTIME_INVARIANT,
            MutationOutcomeStatus.KILLED_BY_RUNTIME_INVARIANT,
        ),
        (DetectorKind.TYPE_CHECK, MutationOutcomeStatus.KILLED_BY_TYPE_CHECK),
    ):
        detector_id = f"det.{kind.value}"
        # detector_id must match token regex (lowercase)
        detector_id = f"det.{kind.value}"
        eds = _detection_set(_prediction(detector_id, kind))
        classification = classify_mutation_outcome(
            predicted_detector_ids=(detector_id,),
            selected_detector_ids=(detector_id,),
            observations=(
                _obs(detector_id, kind, DetectorRunStatus.DETECTED),
            ),
        )
        assert classification.outcome_status == status.value
        assert classification.counts_as_killed is True
        assert classification.killing_detector_kind == kind.value


# ---------------------------------------------------------------------------
# Broader fallback is policy-bound
# ---------------------------------------------------------------------------


def test_survivor_broadens_only_when_policy_allows() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    broader_ran: list[str] = []

    def runner(detector_id: str, kind, phase):
        if phase is ExecutionPhase.BROADER:
            broader_ran.append(detector_id)
        return DetectorRunObservation(
            detector_id=detector_id,
            detector_kind=kind,
            status=DetectorRunStatus.PASSED,
            phase=phase,
        )

    report = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=runner,
        fallback_policy=ExecutionFallbackPolicy(
            enable_broader_fallback=True,
            enable_full_suite_fallback=False,
        ),
        broader_detector_ids=("unit.related_case", "unit.another_case"),
        risk_class="local_bug",
    )
    assert report.broadening_mode is BroadeningMode.BROADER
    assert REASON_FALLBACK_POLICY_BOUND in report.reason_codes
    assert broader_ran == ["unit.related_case", "unit.another_case"]
    assert ExecutionPhase.PREDICTED.value in report.phases
    assert ExecutionPhase.BROADER.value in report.phases
    assert report.outcome is not None
    assert report.outcome.outcome_status == (
        MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value
    )
    assert report.outcome.counts_as_killed is False
    assert "unit.related_case" in (
        report.outcome.detector_classification.executed_detector_ids
    )


def test_broader_fallback_disabled_stays_predicted_only() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    broader_ran: list[str] = []

    def runner(detector_id: str, kind, phase):
        if phase is ExecutionPhase.BROADER or phase is ExecutionPhase.FULL_SUITE:
            broader_ran.append(detector_id)
        return DetectorRunObservation(
            detector_id=detector_id,
            detector_kind=kind,
            status=DetectorRunStatus.PASSED,
            phase=phase,
        )

    report = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=runner,
        fallback_policy=ExecutionFallbackPolicy(
            enable_broader_fallback=False,
            enable_full_suite_fallback=False,
        ),
        broader_detector_ids=("unit.related_case",),
        risk_class="local_bug",
    )
    assert report.broadening_mode is BroadeningMode.NONE
    assert broader_ran == []
    assert REASON_FALLBACK_DISABLED in report.reason_codes or any(
        "disabled" in code or "fallback" in code for code in report.reason_codes
    )
    assert ExecutionPhase.BROADER.value not in report.phases
    assert report.outcome is not None
    assert report.outcome.outcome_status == (
        MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value
    )


def test_high_risk_full_suite_requires_policy_enablement() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    phases_seen: list[str] = []

    def runner(detector_id: str, kind, phase):
        phases_seen.append(phase.value)
        return DetectorRunObservation(
            detector_id=detector_id,
            detector_kind=kind,
            status=DetectorRunStatus.PASSED,
            phase=phase,
        )

    # Without full-suite enablement, high risk does not force full suite.
    report_no = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=runner,
        fallback_policy=ExecutionFallbackPolicy(
            enable_broader_fallback=True,
            enable_full_suite_fallback=False,
            full_suite_on_high_risk=True,
            high_risk_classes=("critical_security",),
        ),
        broader_detector_ids=("unit.related_case",),
        risk_class="critical_security",
    )
    assert report_no.broadening_mode is BroadeningMode.BROADER
    assert ExecutionPhase.FULL_SUITE.value not in report_no.phases

    # With full-suite enablement + high risk → full suite.
    report_yes = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=_runner(
            {
                "unit.test_branch": DetectorRunStatus.PASSED,
                FULL_SUITE_DETECTOR_ID: DetectorRunStatus.DETECTED,
            }
        ),
        fallback_policy=ExecutionFallbackPolicy(
            enable_broader_fallback=True,
            enable_full_suite_fallback=True,
            full_suite_on_high_risk=True,
            high_risk_classes=("critical_security",),
        ),
        risk_class="critical_security",
    )
    assert report_yes.broadening_mode is BroadeningMode.FULL_SUITE
    assert ExecutionPhase.PREDICTED.value in report_yes.phases
    assert ExecutionPhase.FULL_SUITE.value in report_yes.phases
    assert report_yes.outcome is not None
    assert report_yes.outcome.outcome_status == (
        MutationOutcomeStatus.KILLED_BY_FULL_SUITE.value
    )
    assert report_yes.outcome.counts_as_killed is True


def test_resolve_execution_fallback_skips_when_already_killed() -> None:
    mode, reasons = resolve_execution_fallback(
        survived_predicted=False,
        policy=ExecutionFallbackPolicy(enable_broader_fallback=True),
    )
    assert mode is BroadeningMode.NONE
    assert REASON_FALLBACK_SKIPPED_KILL in reasons


def test_always_full_suite_policy() -> None:
    mode, reasons = resolve_execution_fallback(
        survived_predicted=True,
        policy=ExecutionFallbackPolicy(always_full_suite=True),
    )
    assert mode is BroadeningMode.FULL_SUITE
    assert REASON_FALLBACK_POLICY_BOUND in reasons


# ---------------------------------------------------------------------------
# Survival and non-kill statuses
# ---------------------------------------------------------------------------


def test_survived_selected_when_predicted_pass() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    report = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=_runner(
            {"unit.test_branch": DetectorRunStatus.PASSED}
        ),
        fallback_policy=ExecutionFallbackPolicy(
            enable_broader_fallback=False,
            enable_full_suite_fallback=False,
        ),
    )
    assert report.outcome is not None
    assert report.outcome.outcome_status == (
        MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value
    )
    assert report.outcome.counts_as_killed is False
    assert report.outcome.killing_detector_id is None
    assert report.receipt is not None
    assert report.receipt.observed_detector_ids == ()
    assert "unit.test_branch" in report.receipt.executed_detector_ids
    # Missed predicted detectors are derived honestly.
    assert "unit.test_branch" in report.receipt.missed_detector_ids


def test_survived_full_verification() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    report = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=_runner(
            {
                "unit.test_branch": DetectorRunStatus.PASSED,
                FULL_SUITE_DETECTOR_ID: DetectorRunStatus.PASSED,
            }
        ),
        fallback_policy=ExecutionFallbackPolicy(always_full_suite=True),
    )
    assert report.outcome is not None
    assert report.outcome.outcome_status == (
        MutationOutcomeStatus.SURVIVED_FULL_VERIFICATION.value
    )
    assert report.outcome.counts_as_killed is False
    assert report.classification is not None
    assert report.classification.full_suite_executed is True


def test_invalid_uncompilable_timeout_infra_never_count_as_killed() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    for kwargs, status in (
        ({"invalid_mutant": True}, MutationOutcomeStatus.INVALID_MUTANT),
        ({"uncompilable": True}, MutationOutcomeStatus.UNCOMPILABLE),
        (
            {"infrastructure_ok": False},
            MutationOutcomeStatus.INFRASTRUCTURE_FAILURE,
        ),
    ):
        report = execute_mutation(
            candidate_id=CANDIDATE_ID,
            candidate_cid=CANDIDATE_CID,
            mutant_identity_cid=MUTANT_CID,
            baseline=_green_baseline(),
            expected_detection_set=eds,
            **kwargs,
        )
        assert report.outcome is not None
        assert report.outcome.outcome_status == status.value
        assert report.outcome.counts_as_killed is False
        assert_outcome_never_false_kill(report.outcome)

    # Timeout via observation
    report_t = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=_runner(
            {"unit.test_branch": DetectorRunStatus.TIMEOUT}
        ),
        fallback_policy=ExecutionFallbackPolicy(enable_broader_fallback=False),
    )
    assert report_t.outcome is not None
    assert report_t.outcome.outcome_status == MutationOutcomeStatus.TIMEOUT.value
    assert report_t.outcome.counts_as_killed is False
    assert report_t.receipt is not None
    assert report_t.receipt.timed_out is True


def test_equivalent_requires_assessment_cid_and_never_killed() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    eq_cid = _cid("equivalence-assessment")
    report = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=_runner(
            {"unit.test_branch": DetectorRunStatus.PASSED}
        ),
        fallback_policy=ExecutionFallbackPolicy(enable_broader_fallback=False),
        equivalence_status="equivalent",
        equivalence_assessment_cid=eq_cid,
    )
    assert report.outcome is not None
    assert report.outcome.outcome_status == MutationOutcomeStatus.EQUIVALENT.value
    assert report.outcome.counts_as_killed is False
    assert report.outcome.equivalence_assessment_cid == eq_cid

    with pytest.raises(MutationExecutionError, match="equivalence_assessment_cid"):
        execute_mutation(
            candidate_id=CANDIDATE_ID,
            candidate_cid=CANDIDATE_CID,
            mutant_identity_cid=MUTANT_CID,
            baseline=_green_baseline(),
            expected_detection_set=eds,
            detector_runner=_runner(
                {"unit.test_branch": DetectorRunStatus.PASSED}
            ),
            fallback_policy=ExecutionFallbackPolicy(
                enable_broader_fallback=False
            ),
            equivalence_status="equivalent",
        )


# ---------------------------------------------------------------------------
# classify_mutation_outcome honesty
# ---------------------------------------------------------------------------


def test_classify_requires_observed_detector_for_kill() -> None:
    classification = classify_mutation_outcome(
        predicted_detector_ids=("unit.test_branch",),
        selected_detector_ids=("unit.test_branch",),
        observations=(
            _obs(
                "unit.test_branch",
                DetectorKind.UNIT_TEST,
                DetectorRunStatus.PASSED,
            ),
        ),
    )
    assert classification.counts_as_killed is False
    assert classification.outcome_status == (
        MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value
    )

    with pytest.raises(MutationExecutionError, match="dishonest|subset"):
        # observed without executed is impossible via observations path;
        # force via explicit lists that violate nesting.
        classify_mutation_outcome(
            predicted_detector_ids=("unit.test_branch",),
            selected_detector_ids=("unit.test_branch",),
            executed_detector_ids=(),
            observed_detector_ids=("unit.test_branch",),
            detector_kinds={"unit.test_branch": DetectorKind.UNIT_TEST},
        )


def test_classify_one_closed_terminal_outcome_only() -> None:
    classification = classify_mutation_outcome(
        predicted_detector_ids=("unit.a", "unit.b"),
        selected_detector_ids=("unit.a", "unit.b"),
        observations=(
            _obs("unit.a", DetectorKind.UNIT_TEST, DetectorRunStatus.DETECTED),
            _obs("unit.b", DetectorKind.UNIT_TEST, DetectorRunStatus.DETECTED),
        ),
    )
    assert classification.outcome_status == MutationOutcomeStatus.KILLED_BY_TEST.value
    # First observation in order is the killing detector.
    assert classification.killing_detector_id == "unit.a"
    assert REASON_ONE_TERMINAL_OUTCOME in classification.reason_codes
    assert counts_as_killed(classification.outcome_status) is True


def test_classify_baseline_blocked_inconclusive() -> None:
    classification = classify_mutation_outcome(
        predicted_detector_ids=("unit.test_branch",),
        selected_detector_ids=("unit.test_branch",),
        baseline_blocked=True,
    )
    assert classification.outcome_status == (
        MutationOutcomeStatus.INCONCLUSIVE.value
    )
    assert classification.counts_as_killed is False
    assert REASON_BASELINE_BLOCKED in classification.reason_codes


def test_report_persists_receipt_and_outcome_with_matching_identities() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    report = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=_runner(
            {"unit.test_branch": DetectorRunStatus.DETECTED}
        ),
    )
    assert isinstance(report, MutationExecutionReport)
    assert report.receipt is not None
    assert report.outcome is not None
    assert report.outcome.receipt_cid == report.receipt.receipt_cid
    assert report.outcome.expected_detection_set_cid == eds.detection_set_cid
    assert report.detection_set_cid == eds.detection_set_cid
    assert report.report_cid
    # Round-trip identity payload is stable.
    again = report.compute_report_cid()
    assert again == report.report_cid
    payload = report.to_dict()
    assert payload["production_policy_changed"] is False
    assert payload["outcome"]["counts_as_killed"] is True


def test_missing_observation_without_runner_fails_closed() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    with pytest.raises(MutationExecutionError, match="no observation or runner"):
        execute_mutation(
            candidate_id=CANDIDATE_ID,
            candidate_cid=CANDIDATE_CID,
            mutant_identity_cid=MUTANT_CID,
            baseline=_green_baseline(),
            expected_detection_set=eds,
            observations=(),
        )


def test_candidate_identity_mismatch_fails_closed() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    with pytest.raises(MutationExecutionError, match="candidate_id"):
        execute_mutation(
            candidate_id="cand_other_id",
            candidate_cid=CANDIDATE_CID,
            mutant_identity_cid=MUTANT_CID,
            baseline=_green_baseline(),
            expected_detection_set=eds,
            detector_runner=_runner(
                {"unit.test_branch": DetectorRunStatus.PASSED}
            ),
        )


def test_fallback_policy_requires_disposable_worktree_and_no_network() -> None:
    with pytest.raises(MutationExecutionError, match="disposable"):
        ExecutionFallbackPolicy(require_disposable_worktree=False)
    with pytest.raises(MutationExecutionError, match="network"):
        ExecutionFallbackPolicy(require_network_disabled=False)


def test_phases_order_predicted_before_broader() -> None:
    eds = _detection_set(
        _prediction("unit.test_branch", DetectorKind.UNIT_TEST),
    )
    report = execute_mutation(
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        mutant_identity_cid=MUTANT_CID,
        baseline=_green_baseline(),
        expected_detection_set=eds,
        detector_runner=_runner(
            {
                "unit.test_branch": DetectorRunStatus.PASSED,
                "unit.related_case": DetectorRunStatus.PASSED,
            }
        ),
        fallback_policy=ExecutionFallbackPolicy(enable_broader_fallback=True),
        broader_detector_ids=("unit.related_case",),
    )
    phases = list(report.phases)
    assert phases.index(ExecutionPhase.BASELINE.value) < phases.index(
        ExecutionPhase.PREDICTED.value
    )
    assert phases.index(ExecutionPhase.PREDICTED.value) < phases.index(
        ExecutionPhase.BROADER.value
    )
    assert phases.index(ExecutionPhase.BROADER.value) < phases.index(
        ExecutionPhase.CLASSIFIED.value
    )
    # Observation phases also respect order.
    obs_phases = [obs.phase for obs in report.observations]
    if len(obs_phases) >= 2:
        assert obs_phases[0] == ExecutionPhase.PREDICTED.value
