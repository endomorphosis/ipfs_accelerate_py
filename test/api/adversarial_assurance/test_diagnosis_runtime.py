"""Tests for bounded counterexample minimization and survivor diagnosis (AAE-045).

Acceptance criteria enforced here:

* Existing CounterexampleMinimizer and semantic diagnostics produce bounded
  reproductions.
* Minimization failure is explicit (failed flag + reason + bounded digest).
* Every high-risk survivor always persists an AssuranceGap.
* Human review accompanies an unknown gap rather than replacing it.
* Cold import is side-effect free; no production policy change.
"""

from __future__ import annotations

import ast
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.diagnosis import (
    AAE_DIAGNOSIS_RUN_EVIDENCE,
    DIAGNOSE_SURVIVING_MUTANT_INTERFACE,
    REASON_BOUNDED_REPRODUCTION,
    REASON_COUNTEREXAMPLE_MINIMIZER,
    REASON_GAP_IN_MEMORY,
    REASON_GAP_PERSISTED,
    REASON_HIGH_RISK_GAP_REQUIRED,
    REASON_HUMAN_REVIEW_ACCOMPANIES_GAP,
    REASON_LOGS_BOUNDED,
    REASON_MINIMIZATION_FAILED,
    REASON_MINIMIZATION_SUCCEEDED,
    REASON_NO_PRODUCTION_POLICY_CHANGE,
    REASON_SEMANTIC_DIAGNOSIS,
    REASON_UNKNOWN_GAP_HUMAN_REVIEW,
    BoundedReproduction,
    DiagnosisRuntimeError,
    GapPersistRequest,
    GapPersistStatus,
    HighRiskGapMissingError,
    InMemoryAssuranceGapRepository,
    SurvivorDiagnosisRun,
    diagnose_surviving_mutant,
    diagnose_surviving_mutant_descriptor,
    high_risk_classes,
    is_high_risk,
    run_counterexample_minimization,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    TerminalStatus,
    TestReceipt,
    VerificationReceiptKind,
)
from ipfs_accelerate_py.agent_supervisor.verification.counterexamples import (
    COUNTEREXAMPLE_MINIMIZER_INTERFACE,
    FailureMaterial,
    MinimizationBudget,
    MinimizationGuarantee,
    MinimizationRequest,
    RerunObservation,
    extract_failure_material_from_pytest_output,
    minimize_counterexample,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.analysis_contracts import (
    AssuranceGapClass,
    MinimizedEvidenceBinding,
    SourceSpan,
    SurvivorRiskClass,
    verify_gap_identity,
    verify_survivor_report_identity,
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
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.diagnosis import (
    DIAGNOSIS_STEP_ORDER,
    DiagnosisDisposition,
    DiagnosisMutationBinding,
    DiagnosisOutcomeBinding,
    DiagnosisSignals,
    DiagnosisStepId,
    verify_survivor_diagnosis_identity,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    MutationOutcomeStatus,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.minimization import (
    BoundedLogDigest,
    MinimizationStatus,
    SurvivorMinimizationSubject,
    logs_remain_bounded,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from test.api.test_agent_supervisor_verification_contracts import (
    _artifact,
    _key,
    _observation,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DIAGNOSIS_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/diagnosis.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


REPO_ID = "repository:sha256:test-repo-identity-aae045"
REPO_STATE = _cid("repo-state-aae045")
ENV_CID = _cid("environment-aae045")
DEP_LOCK = _cid("dependency-lock-aae045")
POLICY_CID = _cid("policy-aae045")
CANDIDATE_CID = _cid("candidate-aae045")
OUTCOME_CID = _cid("outcome-aae045")
CANDIDATE_ID = "cand_control_flow_invert_0"

NOISY_PYTEST_OUTPUT = """\
============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-8.0.0
F                                                                        [100%]
=================================== FAILURES ===================================
______________________ test_calculate_returns_string ___________________________
/usr/lib/python3.12/site-packages/_pytest/runner.py:120: in from_call
    result = call()
/home/user/project/src/mod.py:12: in test_guard
    assert result == "deny"
E   AssertionError: expected deny
E   assert 'allow' == 'deny'
=========================== short test summary info ============================
FAILED src/mod.py::test_guard - AssertionError: expected deny
1 failed in 0.04s
"""

ORIGINAL_ARGV = (
    "/usr/bin/python3.12",
    "-m",
    "pytest",
    "-q",
    "src/mod.py::test_guard",
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _generator(**overrides: object) -> GeneratorIdentity:
    fields = {
        "generator_id": "diagnosis_orchestration",
        "generator_version": "1.0.0",
        "interface_id": DIAGNOSE_SURVIVING_MUTANT_INTERFACE,
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
        "tool_ids": ("diagnosis_runtime.v1",),
        "policy_cid": POLICY_CID,
        "notes": None,
    }
    fields.update(overrides)
    return ArtifactProvenance(**fields)  # type: ignore[arg-type]


def _header(artifact_kind: str = "mutation_candidate", **overrides: object) -> AssuranceArtifactHeader:
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
        "receipt_cids": (_cid("receipt-a"),),
        "proof_cids": (_cid("proof-a"),),
        "metadata": {},
    }
    fields.update(overrides)
    return AssuranceArtifactHeader(**fields)  # type: ignore[arg-type]


def _span(**overrides: object) -> SourceSpan:
    fields = {
        "path": "src/mod.py",
        "start_line": 10,
        "end_line": 12,
        "start_col": 0,
        "end_col": 40,
    }
    fields.update(overrides)
    return SourceSpan(**fields)  # type: ignore[arg-type]


def _evidence(**overrides: object) -> MinimizedEvidenceBinding:
    fields = {
        "evidence_cids": (_cid("min-evidence-1"),),
        "minimized": True,
        "minimization_failed": False,
        "reproduction_input_cid": _cid("repro-input"),
        "notes": None,
    }
    fields.update(overrides)
    return MinimizedEvidenceBinding(**fields)  # type: ignore[arg-type]


def _mutation(**overrides: object) -> DiagnosisMutationBinding:
    fields = {
        "candidate_id": CANDIDATE_ID,
        "candidate_cid": CANDIDATE_CID,
        "risk_class": SurvivorRiskClass.CRITICAL_SECURITY,
        "symbol_ids": ("mod.fn",),
        "violated_or_missing_property": "authz.must_deny_cross_tenant",
        "source_spans": (_span(),),
        "dependency_path": ("mod.fn", "authz.check"),
        "header": _header(),
        "transformation_summary": "invert tenant guard",
        "likely_equivalent": False,
    }
    fields.update(overrides)
    return DiagnosisMutationBinding(**fields)  # type: ignore[arg-type]


def _outcome(**overrides: object) -> DiagnosisOutcomeBinding:
    fields = {
        "outcome_id": f"{CANDIDATE_ID}.outcome",
        "outcome_cid": OUTCOME_CID,
        "outcome_status": MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION,
        "candidate_id": CANDIDATE_ID,
        "candidate_cid": CANDIDATE_CID,
        "expected_detection_set_cid": _cid("expected-detection"),
    }
    fields.update(overrides)
    return DiagnosisOutcomeBinding(**fields)  # type: ignore[arg-type]


def _signals(**overrides: object) -> DiagnosisSignals:
    fields = {
        "observation_complete": True,
        "not_selected_detector_ids": ("static.authz_rule",),
        "minimized_evidence": _evidence(),
    }
    fields.update(overrides)
    return DiagnosisSignals(**fields)  # type: ignore[arg-type]


def _digest(**overrides: object) -> BoundedLogDigest:
    fields = {
        "digest_cid": _cid("log-digest"),
        "byte_count": 4096,
        "truncated": True,
        "full_log_excluded": True,
        "notes": "first 4 KiB retained under budget",
    }
    fields.update(overrides)
    return BoundedLogDigest(**fields)  # type: ignore[arg-type]


def _failed_test(
    *,
    label: str = "pytest-fail",
    command_argv: Sequence[str] | None = None,
) -> TestReceipt:
    argv = tuple(command_argv) if command_argv is not None else ORIGINAL_ARGV
    key = _key(VerificationReceiptKind.TEST, selector_argv=argv)
    observation = _observation(
        key,
        TerminalStatus.FAILED,
        label=label,
        command_argv=argv,
    )
    return TestReceipt(key, observation)


def _material(receipt: TestReceipt) -> FailureMaterial:
    return extract_failure_material_from_pytest_output(
        NOISY_PYTEST_OUTPUT,
        stdout_artifact_cid=receipt.execution.stdout_artifact_cid,
        stderr_artifact_cid=receipt.execution.stderr_artifact_cid,
        extra_artifact_cids=tuple(receipt.artifact_cids),
        relevant_paths=("src/mod.py",),
        relevant_symbols=("mod.fn",),
        relevant_input={"caller": "cross_tenant", "action": "read"},
        expected_output="deny",
        observed_output="allow",
        source_spans=(
            {
                "path": "src/mod.py",
                "start_line": 12,
                "end_line": 12,
                "artifact_cid": _artifact("span-mod"),
                "symbol": "mod.fn",
            },
        ),
    )


def _oracle_preserving(
    material: FailureMaterial,
    *,
    lease_prefix: str = "resource-lease:aae045",
    pass_instead: bool = False,
    counter: itertools.count | None = None,
) -> Any:
    counter = counter or itertools.count(1)
    assertion = material.assertion_message
    exception_type = material.exception_type
    node_id = material.node_id
    primary = "src/mod.py:12"

    def _run(argv: Sequence[str]) -> RerunObservation:
        lease_id = f"{lease_prefix}-{next(counter)}"
        if pass_instead:
            return RerunObservation(
                terminal_status=TerminalStatus.PASSED,
                exit_code=0,
                lease_id=lease_id,
                command_argv=tuple(argv),
                stdout_preview=".\n",
                stdout_artifact_cid=_artifact(f"rerun-stdout-{lease_id}"),
                stderr_artifact_cid=_artifact(f"rerun-stderr-{lease_id}"),
            )
        body = (
            f"FAILED {node_id} - {exception_type}: {assertion}\n"
            f"{primary}: in test_guard\n"
            f"    assert result == 'deny'\n"
            f"E   {exception_type}: {assertion}\n"
            f"E   assert 'allow' == 'deny'\n"
        )
        return RerunObservation(
            terminal_status=TerminalStatus.FAILED,
            exit_code=1,
            lease_id=lease_id,
            command_argv=tuple(argv),
            stdout_preview=body,
            stderr_preview="",
            stdout_artifact_cid=_artifact(f"rerun-stdout-{lease_id}"),
            stderr_artifact_cid=_artifact(f"rerun-stderr-{lease_id}"),
            combined_output=body,
        )

    return _run


def _run(**overrides: object) -> SurvivorDiagnosisRun:
    mutation = overrides.pop("mutation", _mutation())
    outcome = overrides.pop("outcome", _outcome())
    repository_state = overrides.pop("repository_state", REPO_STATE)
    signals = overrides.pop("signals", None)
    if signals is None and "minimized_evidence" not in overrides and (
        "failed_receipt" not in overrides
        and "minimization_request" not in overrides
        and "minimization_subject" not in overrides
    ):
        signals = _signals()
    return diagnose_surviving_mutant(
        mutation,  # type: ignore[arg-type]
        outcome,  # type: ignore[arg-type]
        repository_state,  # type: ignore[arg-type]
        signals=signals,  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Static / cold-import
# ---------------------------------------------------------------------------


def test_module_ast_parses_and_exports_interface() -> None:
    source = DIAGNOSIS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module)
    assert "diagnose_surviving_mutant" in source
    assert "CounterexampleMinimizer" in source
    assert "AssuranceGap" in source
    assert "aae/diagnosis-run@1" in source


def test_descriptor_binds_released_authorities() -> None:
    desc = diagnose_surviving_mutant_descriptor()
    assert desc["interface_id"] == DIAGNOSE_SURVIVING_MUTANT_INTERFACE
    assert desc["evidence"] == AAE_DIAGNOSIS_RUN_EVIDENCE
    assert COUNTEREXAMPLE_MINIMIZER_INTERFACE in desc["depends_on"]
    assert desc["production_policy_change"] is False
    assert "critical_security" in desc["high_risk_classes"]


def test_high_risk_vocabulary() -> None:
    assert is_high_risk(SurvivorRiskClass.CRITICAL_SECURITY)
    assert is_high_risk("authorization")
    assert is_high_risk(SurvivorRiskClass.HIGH)
    assert not is_high_risk(SurvivorRiskClass.LOW)
    assert not is_high_risk(SurvivorRiskClass.LOCAL_BUG)
    assert "financial_legal" in high_risk_classes()


# ---------------------------------------------------------------------------
# Bounded counterexample minimization
# ---------------------------------------------------------------------------


def test_counterexample_minimizer_produces_bounded_reproduction() -> None:
    receipt = _failed_test()
    material = _material(receipt)
    oracle = _oracle_preserving(material)
    reproduction, result = run_counterexample_minimization(
        failed_receipt=receipt,
        failure_material=material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=oracle,
        semantic_cone_paths=("src/mod.py",),
    )
    assert reproduction.used_counterexample_minimizer is True
    assert reproduction.logs_bounded is True
    assert reproduction.minimization_failed is False
    assert reproduction.evidence is not None
    assert reproduction.evidence.minimized is True
    assert reproduction.evidence.minimization_failed is False
    assert reproduction.evidence.reproduction_input_cid
    assert reproduction.reproduction_argv
    assert "full_log" not in reproduction.to_dict()
    assert result.receipt.minimized is True
    assert result.quality.guarantee in {
        MinimizationGuarantee.RERUN_VALIDATED,
        MinimizationGuarantee.BOUNDED,
    }
    assert REASON_COUNTEREXAMPLE_MINIMIZER in reproduction.reason_codes


def test_minimization_failure_is_explicit_without_oracle() -> None:
    """Without a lease rerun oracle, minimizer must not claim success."""

    receipt = _failed_test(label="no-oracle")
    material = _material(receipt)
    reproduction, result = run_counterexample_minimization(
        failed_receipt=receipt,
        failure_material=material,
        reproduction_argv=ORIGINAL_ARGV,
        semantic_cone_paths=("src/mod.py",),
    )
    assert reproduction.minimization_failed is True
    assert reproduction.minimization_status == MinimizationStatus.FAILED.value
    assert reproduction.minimization_failure_reason
    assert reproduction.bounded_log_digest is not None
    assert reproduction.bounded_log_digest.full_log_excluded is True
    assert reproduction.evidence is not None
    assert reproduction.evidence.minimization_failed is True
    assert reproduction.evidence.minimized is False
    assert result.receipt.minimized is False
    assert REASON_MINIMIZATION_FAILED in reproduction.reason_codes


def test_diagnose_uses_counterexample_minimizer_end_to_end() -> None:
    receipt = _failed_test(label="e2e-min")
    material = _material(receipt)
    oracle = _oracle_preserving(material)
    run = diagnose_surviving_mutant(
        _mutation(risk_class=SurvivorRiskClass.AUTHORIZATION),
        _outcome(),
        REPO_STATE,
        signals=_signals(minimized_evidence=None, not_selected_detector_ids=("static.authz_rule",)),
        failed_receipt=receipt,
        failure_material=material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=oracle,
        semantic_cone_paths=("src/mod.py",),
        detectors_run=("unit.test_guard",),
        detectors_omitted=("static.authz_rule",),
        expected_behavior="deny unauthorized caller",
        observed_behavior="unauthorized caller accepted",
    )
    assert run.production_policy_changed is False
    assert run.logs_bounded is True
    assert run.minimization_failed is False
    assert run.reproduction is not None
    assert run.reproduction.used_counterexample_minimizer is True
    assert run.survivor_report is not None
    verify_survivor_report_identity(run.survivor_report)
    assert logs_remain_bounded(run.survivor_report)
    assert run.diagnosis is not None
    verify_survivor_diagnosis_identity(run.diagnosis)
    assert len(run.diagnosis.steps) == 9
    assert [step.step_id for step in run.diagnosis.steps] == list(DIAGNOSIS_STEP_ORDER)
    assert REASON_SEMANTIC_DIAGNOSIS in run.reason_codes
    assert REASON_BOUNDED_REPRODUCTION in run.reason_codes
    assert REASON_LOGS_BOUNDED in run.reason_codes
    assert REASON_NO_PRODUCTION_POLICY_CHANGE in run.reason_codes


def test_diagnose_with_minimization_request() -> None:
    receipt = _failed_test(label="req-min")
    material = _material(receipt)
    request = MinimizationRequest(
        failed_receipt=receipt,
        material=material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(material),
        budget=MinimizationBudget(),
        semantic_cone_paths=("src/mod.py",),
    )
    run = diagnose_surviving_mutant(
        _mutation(),
        _outcome(),
        {"repository_state_cid": REPO_STATE},
        signals=_signals(
            minimized_evidence=None,
            not_executed_detector_ids=("unit.test_guard",),
        ),
        minimization_request=request,
        detectors_run=("unit.test_guard",),
        expected_behavior="deny",
        observed_behavior="allow",
    )
    assert run.minimization_failed is False
    assert run.high_risk is True
    assert run.assurance_gap is not None


# ---------------------------------------------------------------------------
# Explicit minimization failure through full diagnose path
# ---------------------------------------------------------------------------


def test_diagnose_records_explicit_minimization_failure() -> None:
    receipt = _failed_test(label="fail-min")
    material = _material(receipt)
    run = diagnose_surviving_mutant(
        _mutation(risk_class=SurvivorRiskClass.HIGH),
        _outcome(),
        REPO_STATE,
        signals=_signals(minimized_evidence=None),
        failed_receipt=receipt,
        failure_material=material,
        reproduction_argv=ORIGINAL_ARGV,
        # no oracle → explicit failure
        detectors_run=("unit.test_guard",),
        expected_behavior="deny",
        observed_behavior="allow",
    )
    assert run.minimization_failed is True
    assert run.reproduction is not None
    assert run.reproduction.minimization_failure_reason
    assert run.reproduction.bounded_log_digest is not None
    assert run.survivor_report is not None
    assert run.survivor_report.minimized_evidence.minimization_failed is True
    assert run.survivor_report.minimized_evidence.minimized is False
    assert REASON_MINIMIZATION_FAILED in run.reason_codes
    # Full logs must not appear on durable surfaces.
    report_dict = run.survivor_report.to_dict()
    assert "full_log" not in report_dict
    assert "raw_traceback" not in report_dict
    assert logs_remain_bounded(run.survivor_report)


def test_prebuilt_failed_subject_is_explicit() -> None:
    subject = SurvivorMinimizationSubject(
        subject_id="subj.survivor.fail",
        report_id="survivor_fail",
        candidate_id=CANDIDATE_ID,
        candidate_cid=CANDIDATE_CID,
        outcome_cid=OUTCOME_CID,
        risk_class=SurvivorRiskClass.AUTHORIZATION,
        symbol_ids=("mod.fn",),
        violated_or_missing_property="authz.must_deny_cross_tenant",
        source_spans=(_span(),),
        detectors_run=("unit.test_guard",),
        detectors_omitted=("static.authz_rule",),
        expected_behavior="deny",
        observed_behavior="allow",
        dependency_path=("mod.fn", "authz.check"),
        reproduction_command="pytest -q src/mod.py::test_guard",
        evidence_cids=(),
        minimization_status=MinimizationStatus.FAILED,
        minimization_failure_reason="counterexample minimizer exhausted budget",
        bounded_log_digest=_digest(),
        observation_complete=True,
        repository_state_cid=REPO_STATE,
    )
    run = diagnose_surviving_mutant(
        _mutation(risk_class=SurvivorRiskClass.AUTHORIZATION),
        _outcome(),
        REPO_STATE,
        signals=_signals(minimized_evidence=None),
        minimization_subject=subject,
    )
    assert run.minimization_failed is True
    assert run.reproduction is not None
    assert "exhausted budget" in (run.reproduction.minimization_failure_reason or "")
    assert run.assurance_gap is not None
    assert run.assurance_gap.minimized_evidence.minimization_failed is True


# ---------------------------------------------------------------------------
# High-risk survivors always persist AssuranceGap
# ---------------------------------------------------------------------------


def test_high_risk_survivor_always_persists_gap() -> None:
    repo = InMemoryAssuranceGapRepository()
    run = _run(
        mutation=_mutation(risk_class=SurvivorRiskClass.CRITICAL_SECURITY),
        signals=_signals(not_selected_detector_ids=("static.authz_rule",)),
        gap_repository=repo,
        gap_persist_request=GapPersistRequest(
            workspace="campaign-aae045",
            artifact_operation_id="gap_art_1",
            history_operation_id="gap_hist_1",
        ),
        detectors_run=("unit.test_guard",),
        detectors_omitted=("static.authz_rule",),
        expected_behavior="deny",
        observed_behavior="allow",
    )
    assert run.high_risk is True
    assert run.assurance_gap is not None
    assert run.gap_cid == run.assurance_gap.gap_cid
    verify_gap_identity(run.assurance_gap)
    assert run.gap_persist_status == GapPersistStatus.PERSISTED.value
    assert REASON_HIGH_RISK_GAP_REQUIRED in run.reason_codes
    assert REASON_GAP_PERSISTED in run.reason_codes
    assert any(run.gap_cid in key for key in repo.gaps)
    # Gap is durable authority — not replaced by human-review-only disposition.
    assert run.assurance_gap.candidate_id == CANDIDATE_ID


def test_high_risk_gap_sealed_without_repository() -> None:
    run = _run(
        mutation=_mutation(risk_class=SurvivorRiskClass.PROOF_RECEIPT_TRUST),
        signals=_signals(not_executed_detector_ids=("unit.test_guard",)),
        detectors_run=("unit.test_guard",),
        expected_behavior="seal binds receipt",
        observed_behavior="receipt accepted without seal",
    )
    assert run.high_risk is True
    assert run.assurance_gap is not None
    assert run.gap_persist_status == GapPersistStatus.SEALED_ONLY.value
    assert REASON_GAP_IN_MEMORY in run.reason_codes
    assert run.gap_cid is not None


@pytest.mark.parametrize(
    "risk",
    [
        SurvivorRiskClass.CRITICAL_SECURITY,
        SurvivorRiskClass.AUTHORIZATION,
        SurvivorRiskClass.FINANCIAL_LEGAL,
        SurvivorRiskClass.DURABILITY,
        SurvivorRiskClass.DISTRIBUTED_TRANSITION,
        SurvivorRiskClass.PROOF_RECEIPT_TRUST,
        SurvivorRiskClass.CRITICAL_INVARIANT,
        SurvivorRiskClass.HIGH,
    ],
)
def test_all_default_high_risk_classes_require_gap(risk: SurvivorRiskClass) -> None:
    run = _run(
        mutation=_mutation(risk_class=risk),
        signals=_signals(not_selected_detector_ids=("static.authz_rule",)),
        detectors_run=("unit.test_guard",),
        expected_behavior="deny",
        observed_behavior="allow",
    )
    assert run.high_risk is True
    assert run.assurance_gap is not None
    assert run.gap_cid is not None


def test_low_risk_does_not_require_gap_by_default() -> None:
    run = _run(
        mutation=_mutation(risk_class=SurvivorRiskClass.LOW),
        signals=_signals(not_selected_detector_ids=("static.authz_rule",)),
        detectors_run=("unit.test_guard",),
        expected_behavior="local behavior",
        observed_behavior="local survivor",
    )
    assert run.high_risk is False
    assert run.assurance_gap is None
    assert run.gap_persist_status == GapPersistStatus.NOT_REQUIRED.value
    assert run.diagnosis is not None


def test_always_persist_gap_overrides_low_risk() -> None:
    run = _run(
        mutation=_mutation(risk_class=SurvivorRiskClass.LOCAL_BUG),
        signals=_signals(not_selected_detector_ids=("static.authz_rule",)),
        always_persist_gap=True,
        detectors_run=("unit.test_guard",),
        expected_behavior="x",
        observed_behavior="y",
    )
    assert run.high_risk is False
    assert run.assurance_gap is not None
    assert run.gap_cid is not None


def test_high_risk_run_constructor_rejects_missing_gap() -> None:
    with pytest.raises(HighRiskGapMissingError):
        SurvivorDiagnosisRun(
            candidate_id=CANDIDATE_ID,
            candidate_cid=CANDIDATE_CID,
            outcome_cid=OUTCOME_CID,
            repository_state_cid=REPO_STATE,
            risk_class=SurvivorRiskClass.CRITICAL_SECURITY.value,
            high_risk=True,
            assurance_gap=None,
            gap_cid=None,
        )


# ---------------------------------------------------------------------------
# Unknown gap + human review accompaniment
# ---------------------------------------------------------------------------


def test_unknown_gap_requires_human_review_accompanying_gap() -> None:
    # Residual unknown path: no detector-partition signals.
    run = _run(
        mutation=_mutation(risk_class=SurvivorRiskClass.CRITICAL_SECURITY),
        signals=_signals(
            not_selected_detector_ids=(),
            not_executed_detector_ids=(),
            path_unobserved_detector_ids=(),
            weak_property_detector_ids=(),
            dependency_omission_detector_ids=(),
            capsule_omission_detector_ids=(),
        ),
        detectors_run=("unit.test_guard",),
        expected_behavior="deny",
        observed_behavior="allow",
    )
    assert run.diagnosis is not None
    assert run.diagnosis.disposition in {
        DiagnosisDisposition.UNKNOWN.value,
        DiagnosisDisposition.ASSURANCE_GAP.value,
    }
    assert run.assurance_gap is not None
    # Whether disposition is unknown residual or another gap class, when the
    # sealed gap is unknown it must require human review *and* still exist.
    if run.assurance_gap.gap_class == AssuranceGapClass.UNKNOWN.value:
        assert run.assurance_gap.requires_human_review is True
        assert run.requires_human_review is True
        assert run.human_review_accompanies_gap is True
        assert REASON_UNKNOWN_GAP_HUMAN_REVIEW in run.reason_codes
        assert REASON_HUMAN_REVIEW_ACCOMPANIES_GAP in run.reason_codes
    # Gap is never replaced by a human-review-only outcome.
    assert run.gap_cid is not None
    assert run.assurance_gap.gap_cid == run.gap_cid


def test_probably_equivalent_high_risk_still_persists_gap() -> None:
    run = _run(
        mutation=_mutation(risk_class=SurvivorRiskClass.AUTHORIZATION),
        outcome=_outcome(
            outcome_status=MutationOutcomeStatus.PROBABLY_EQUIVALENT
        ),
        signals=_signals(
            equivalence_status="probably_equivalent",
            not_selected_detector_ids=(),
        ),
        detectors_run=("unit.test_guard",),
        expected_behavior="equivalent under bound",
        observed_behavior="probably equivalent",
    )
    assert run.high_risk is True
    assert run.assurance_gap is not None
    assert run.requires_human_review is True
    # Human review accompanies gap — gap remains.
    assert run.human_review_accompanies_gap is True
    assert run.gap_cid is not None
    assert run.diagnosis is not None
    assert run.diagnosis.disposition == (
        DiagnosisDisposition.PROBABLY_EQUIVALENT.value
    )


def test_unknown_gap_without_review_rejected_on_run() -> None:
    # Build a valid unknown gap, then try to seal a run that drops review.
    good = _run(
        mutation=_mutation(risk_class=SurvivorRiskClass.HIGH),
        signals=_signals(
            not_selected_detector_ids=(),
            not_executed_detector_ids=(),
            path_unobserved_detector_ids=(),
            weak_property_detector_ids=(),
            dependency_omission_detector_ids=(),
            capsule_omission_detector_ids=(),
        ),
        detectors_run=("unit.test_guard",),
        expected_behavior="a",
        observed_behavior="b",
    )
    assert good.assurance_gap is not None
    # Force a run that claims unknown gap without review accompaniment.
    if good.assurance_gap.gap_class == AssuranceGapClass.UNKNOWN.value:
        with pytest.raises(DiagnosisRuntimeError):
            SurvivorDiagnosisRun(
                candidate_id=good.candidate_id,
                candidate_cid=good.candidate_cid,
                outcome_cid=good.outcome_cid,
                repository_state_cid=good.repository_state_cid,
                risk_class=good.risk_class,
                high_risk=True,
                assurance_gap=good.assurance_gap,
                gap_cid=good.gap_cid,
                requires_human_review=False,
                human_review_accompanies_gap=False,
            )


# ---------------------------------------------------------------------------
# Semantic diagnosis integration
# ---------------------------------------------------------------------------


def test_assurance_gap_disposition_from_not_selected() -> None:
    run = _run(
        mutation=_mutation(risk_class=SurvivorRiskClass.AUTHORIZATION),
        signals=_signals(not_selected_detector_ids=("static.authz_rule",)),
        detectors_run=("unit.test_guard",),
        detectors_omitted=("static.authz_rule",),
        expected_behavior="deny",
        observed_behavior="allow",
    )
    assert run.diagnosis is not None
    assert run.diagnosis.disposition == DiagnosisDisposition.ASSURANCE_GAP.value
    assert run.diagnosis.deciding_step_id == (
        DiagnosisStepId.DETECTOR_SELECTION.value
    )
    assert run.diagnosis.gap_class == AssuranceGapClass.TEST_SELECTION_FAILURE.value
    assert run.assurance_gap is not None
    assert run.assurance_gap.gap_class == AssuranceGapClass.TEST_SELECTION_FAILURE.value


def test_nine_step_path_always_recorded() -> None:
    run = _run(
        signals=_signals(not_executed_detector_ids=("unit.test_guard",)),
        detectors_run=("unit.test_guard",),
        expected_behavior="deny",
        observed_behavior="allow",
    )
    assert run.diagnosis is not None
    assert [s.step_id for s in run.diagnosis.steps] == list(DIAGNOSIS_STEP_ORDER)
    assert run.diagnosis.difficulty_to_kill_not_evidence is True
    assert run.diagnosis.survivor_not_automatically_product_defect is True


def test_survivor_report_binds_required_surface() -> None:
    run = _run(
        detectors_run=("unit.test_guard", "static.authz_rule"),
        detectors_omitted=("proof.obligation",),
        expected_behavior="deny unauthorized",
        observed_behavior="allow unauthorized",
        reproduction_command="pytest -q src/mod.py::test_guard",
    )
    report = run.survivor_report
    assert report is not None
    assert report.candidate_id == CANDIDATE_ID
    assert report.risk_class == SurvivorRiskClass.CRITICAL_SECURITY.value
    assert report.symbol_ids
    assert report.violated_or_missing_property
    assert report.detectors_run
    assert report.expected_behavior
    assert report.observed_behavior
    assert report.source_spans
    assert report.dependency_path
    assert report.reproduction_command
    assert report.minimized_evidence is not None


# ---------------------------------------------------------------------------
# Fail-closed negatives
# ---------------------------------------------------------------------------


def test_incomplete_observation_fails_closed() -> None:
    with pytest.raises(DiagnosisRuntimeError, match="observation_complete"):
        diagnose_surviving_mutant(
            _mutation(),
            _outcome(),
            REPO_STATE,
            signals=_signals(observation_complete=False),
        )


def test_candidate_mismatch_fails_closed() -> None:
    with pytest.raises(DiagnosisRuntimeError, match="candidate_id"):
        diagnose_surviving_mutant(
            _mutation(),
            _outcome(candidate_id="cand_other_0"),
            REPO_STATE,
            signals=_signals(),
        )


def test_missing_minimization_evidence_fails_closed() -> None:
    with pytest.raises(DiagnosisRuntimeError, match="minimization"):
        diagnose_surviving_mutant(
            _mutation(),
            _outcome(),
            REPO_STATE,
            signals=_signals(minimized_evidence=None),
        )


def test_mapping_inputs_accepted() -> None:
    run = diagnose_surviving_mutant(
        _mutation().to_dict(),
        _outcome().to_dict(),
        REPO_STATE,
        signals=_signals(not_selected_detector_ids=("static.authz_rule",)).to_dict(),
        detectors_run=("unit.test_guard",),
        expected_behavior="deny",
        observed_behavior="allow",
    )
    assert run.diagnosis is not None
    assert run.run_cid
    # Round-trip identity of nested diagnosis.
    verify_survivor_diagnosis_identity(run.diagnosis)


def test_run_is_deterministic() -> None:
    kwargs = dict(
        mutation=_mutation(),
        outcome=_outcome(),
        repository_state=REPO_STATE,
        signals=_signals(not_selected_detector_ids=("static.authz_rule",)),
        detectors_run=("unit.test_guard",),
        detectors_omitted=("static.authz_rule",),
        expected_behavior="deny",
        observed_behavior="allow",
    )
    a = diagnose_surviving_mutant(**kwargs)  # type: ignore[arg-type]
    b = diagnose_surviving_mutant(**kwargs)  # type: ignore[arg-type]
    assert a.run_cid == b.run_cid
    assert a.diagnosis_cid == b.diagnosis_cid
    assert a.gap_cid == b.gap_cid
    assert a.survivor_report_cid == b.survivor_report_cid


def test_production_policy_never_changed() -> None:
    run = _run(
        detectors_run=("unit.test_guard",),
        expected_behavior="deny",
        observed_behavior="allow",
    )
    assert run.production_policy_changed is False
    payload = run.to_dict()
    assert payload["production_policy_changed"] is False


def test_bounded_reproduction_rejects_success_without_evidence() -> None:
    with pytest.raises(DiagnosisRuntimeError):
        BoundedReproduction(
            minimization_status=MinimizationStatus.MINIMIZED.value,
            minimization_failed=False,
            evidence=None,
            reproduction_command="pytest -q",
        )


def test_minimization_failed_requires_reason_and_digest() -> None:
    with pytest.raises(DiagnosisRuntimeError, match="minimization_failure_reason"):
        BoundedReproduction(
            minimization_status=MinimizationStatus.FAILED.value,
            minimization_failed=True,
            evidence=MinimizedEvidenceBinding(
                evidence_cids=(),
                minimized=False,
                minimization_failed=True,
            ),
            reproduction_command="pytest -q",
            minimization_failure_reason=None,
            bounded_log_digest=_digest(),
        )
    with pytest.raises(DiagnosisRuntimeError, match="bounded_log_digest"):
        BoundedReproduction(
            minimization_status=MinimizationStatus.FAILED.value,
            minimization_failed=True,
            evidence=MinimizedEvidenceBinding(
                evidence_cids=(),
                minimized=False,
                minimization_failed=True,
            ),
            reproduction_command="pytest -q",
            minimization_failure_reason="budget exhausted",
            bounded_log_digest=None,
        )


def test_direct_minimize_counterexample_still_available() -> None:
    """Sanity: released minimizer remains usable independently of orchestration."""

    receipt = _failed_test(label="direct")
    material = _material(receipt)
    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(material),
        semantic_cone_paths=("src/mod.py",),
    )
    assert result.receipt.minimized is True
