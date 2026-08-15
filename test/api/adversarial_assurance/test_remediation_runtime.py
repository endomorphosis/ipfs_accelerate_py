"""Tests for candidate generation and held-out remediation evaluation (AAE-046).

Acceptance criteria enforced here:

* Evaluation covers original (unmutated), diagnosis, development, held-out,
  unrelated, performance, false-positive, overconstraint, and safety behavior.
* One-mutant overfit is rejected.
* Mock bypass is rejected.
* Cold import is side-effect free; no production policy change.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.remediation import (
    AAE046_EVALUATION_PARTITIONS,
    AAE_REMEDIATION_EVALUATION_EVIDENCE,
    ADAPTER_ID,
    EVALUATE_REMEDIATION_INTERFACE,
    GENERATOR_ID,
    ORIGINAL_PARTITION,
    PERFORMANCE_PARTITION,
    PROPOSE_GAP_REMEDIATION_INTERFACE,
    REASON_MOCK_BYPASS,
    REASON_NO_PRODUCTION_POLICY_CHANGE,
    REASON_ONE_MUTANT_OVERFIT,
    REASON_ORIGINAL_EVALUATED,
    REASON_QUALIFIED,
    CampaignPartitionResult,
    HeldOutCampaign,
    MockBypassError,
    OneMutantOverfitError,
    RemediationEvaluationRun,
    RemediationProposalRun,
    RemediationRuntimeError,
    aae046_evaluation_partitions,
    detect_mock_bypass,
    detect_one_mutant_overfit,
    evaluate_remediation,
    evaluate_remediation_descriptor,
    evaluation_covers_acceptance_partitions,
    propose_gap_remediation,
    propose_gap_remediation_descriptor,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.analysis_contracts import (
    AssuranceGap,
    AssuranceGapClass,
    GapSeverity,
    MinimizedEvidenceBinding,
    SourceSpan,
    SurvivingMutantReport,
    SurvivorRiskClass,
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
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.held_out import (
    REQUIRED_EVALUATION_PARTITIONS,
    QualificationDisposition,
    partition_mutants,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.remediation_contracts import (
    EvaluationPartition,
    EvaluationVerdict,
    RejectionReason,
    RemediationPlanStatus,
    verify_evaluation_report_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REMEDIATION_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/remediation.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


REPO_ID = "repository:sha256:test-repo-identity-aae046"
REPO_STATE = _cid("repo-state-aae046")
ENV_CID = _cid("environment-aae046")
DEP_LOCK = _cid("dependency-lock-aae046")
POLICY_CID = _cid("policy-aae046")
CANDIDATE_ID = "cand_authz_invert_0"
CANDIDATE_CID = _cid("candidate-aae046")
OUTCOME_CID = _cid("outcome-aae046")
PROPERTY = "authorization check must reject unauthorized callers"
EXPECTED = "reject unauthorized caller for protected action"
OBSERVED = "unauthorized caller accepted and side effect applied"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _generator(**overrides: object) -> GeneratorIdentity:
    fields = {
        "generator_id": "remediation_evaluation",
        "generator_version": "1.0.0",
        "interface_id": EVALUATE_REMEDIATION_INTERFACE,
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
        "tool_ids": ("remediation_runtime.v1",),
        "policy_cid": POLICY_CID,
        "notes": None,
    }
    fields.update(overrides)
    return ArtifactProvenance(**fields)  # type: ignore[arg-type]


def _header(artifact_kind: str = "mutation_campaign_plan", **overrides: object) -> AssuranceArtifactHeader:
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


def _survivor(**overrides: object) -> SurvivingMutantReport:
    fields = {
        "header": _header("surviving_mutant_report"),
        "report_id": "survivor_authz_1",
        "candidate_id": CANDIDATE_ID,
        "candidate_cid": CANDIDATE_CID,
        "outcome_cid": OUTCOME_CID,
        "risk_class": SurvivorRiskClass.AUTHORIZATION,
        "symbol_ids": ("mod.fn",),
        "violated_or_missing_property": PROPERTY,
        "detectors_run": ("unit.test_branch",),
        "detectors_omitted": ("static.authz_rule",),
        "expected_behavior": EXPECTED,
        "observed_behavior": OBSERVED,
        "source_spans": (_span(),),
        "dependency_path": ("mod.fn", "authz.check"),
        "reproduction_command": "pytest -q tests/test_authz.py::test_reject",
        "minimized_evidence": _evidence(),
        "proof_cids": (_cid("proof-a"),),
        "receipt_cids": (_cid("receipt-a"),),
        "equivalence_assessment_cid": None,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return SurvivingMutantReport(**fields)  # type: ignore[arg-type]


def _gap(
    gap_class: AssuranceGapClass | str = AssuranceGapClass.MISSING_TEST,
    **overrides: object,
) -> AssuranceGap:
    survivor = overrides.pop("survivor", None)
    if survivor is None:
        survivor = _survivor()
    requires_review = gap_class in {
        AssuranceGapClass.UNKNOWN.value,
        AssuranceGapClass.UNKNOWN,
        AssuranceGapClass.SPECIFICATION_AMBIGUITY.value,
        AssuranceGapClass.SPECIFICATION_AMBIGUITY,
        AssuranceGapClass.INTENTIONALLY_UNCONSTRAINED.value,
        AssuranceGapClass.INTENTIONALLY_UNCONSTRAINED,
        AssuranceGapClass.PROBABLY_EQUIVALENT.value,
        AssuranceGapClass.PROBABLY_EQUIVALENT,
    }
    fields = {
        "header": _header("assurance_gap"),
        "gap_id": "gap_authz_missing_test",
        "gap_class": gap_class,
        "severity": GapSeverity.CRITICAL,
        "risk_class": survivor.risk_class,
        "summary": f"assurance gap {gap_class} for {PROPERTY}",
        "candidate_id": survivor.candidate_id,
        "candidate_cid": survivor.candidate_cid,
        "survivor_report_cid": survivor.report_cid,
        "violated_or_missing_property": PROPERTY,
        "symbol_ids": survivor.symbol_ids,
        "source_spans": survivor.source_spans,
        "dependency_path": survivor.dependency_path,
        "minimized_evidence": survivor.minimized_evidence,
        "requires_human_review": requires_review,
        "detection_failure_cids": (),
        "vacuity_finding_cids": (),
        "notes": None,
        "metadata": {
            "requirement_id": "req_authz_reject",
            "requirement_source_id": "spec_authz_v1",
            "requirement_cid": _cid("req-doc"),
            "requirement_source_path": "docs/requirements/authz.md",
        },
    }
    fields.update(overrides)
    return AssuranceGap(**fields)  # type: ignore[arg-type]


def _partition(
    partition: EvaluationPartition | str,
    *,
    passed: bool = True,
    mutant_ids: tuple[str, ...] = (),
    killed_mutant_ids: tuple[str, ...] | None = None,
    mock_bypass: bool = False,
    freezes_implementation: bool = False,
    one_mutant_only: bool = False,
) -> CampaignPartitionResult:
    if killed_mutant_ids is None:
        if passed and mutant_ids:
            killed_mutant_ids = mutant_ids
        else:
            killed_mutant_ids = ()
    return CampaignPartitionResult(
        partition=partition,
        passed=passed,
        mutant_ids=mutant_ids,
        killed_mutant_ids=killed_mutant_ids,
        mock_bypass=mock_bypass,
        freezes_implementation=freezes_implementation,
        one_mutant_only=one_mutant_only,
        evidence_cids=(_cid(f"ev-{partition}"),),
    )


def _all_partition_results(
    *,
    diagnosis_ids: tuple[str, ...] = ("mut_diag_1",),
    development_ids: tuple[str, ...] = ("mut_dev_1",),
    held_out_ids: tuple[str, ...] = ("mut_hold_1", "mut_hold_2"),
    fail: frozenset[str] | None = None,
    mock_bypass_partitions: frozenset[str] | None = None,
    overfit_one_mutant: bool = False,
) -> tuple[CampaignPartitionResult, ...]:
    fail = fail or frozenset()
    mock_bypass_partitions = mock_bypass_partitions or frozenset()
    specs = (
        (EvaluationPartition.UNMUTATED, ()),
        (EvaluationPartition.DIAGNOSIS, diagnosis_ids),
        (EvaluationPartition.DEVELOPMENT, development_ids),
        (EvaluationPartition.HELD_OUT, held_out_ids),
        (EvaluationPartition.UNRELATED, ()),
        (EvaluationPartition.PERFORMANCE_COST, ()),
        (EvaluationPartition.FALSE_POSITIVE, ()),
        (EvaluationPartition.OVERCONSTRAINT, ()),
        (EvaluationPartition.REGRESSION, ()),
        (EvaluationPartition.SAFETY, ()),
    )
    results: list[CampaignPartitionResult] = []
    for partition, mutant_ids in specs:
        value = partition.value if isinstance(partition, EvaluationPartition) else partition
        passed = value not in fail and value not in mock_bypass_partitions
        if overfit_one_mutant and value == EvaluationPartition.HELD_OUT.value:
            # Diagnosis-only kill: held-out mutants survive.
            results.append(
                _partition(
                    partition,
                    passed=False,
                    mutant_ids=mutant_ids,
                    killed_mutant_ids=(),
                )
            )
            continue
        if overfit_one_mutant and value == EvaluationPartition.DEVELOPMENT.value:
            results.append(
                _partition(
                    partition,
                    passed=False,
                    mutant_ids=mutant_ids,
                    killed_mutant_ids=(),
                )
            )
            continue
        if overfit_one_mutant and value == EvaluationPartition.DIAGNOSIS.value:
            results.append(
                _partition(
                    partition,
                    passed=True,
                    mutant_ids=mutant_ids,
                    killed_mutant_ids=mutant_ids,
                    one_mutant_only=True,
                )
            )
            continue
        results.append(
            _partition(
                partition,
                passed=passed,
                mutant_ids=mutant_ids,
                mock_bypass=value in mock_bypass_partitions,
            )
        )
    return tuple(results)


def _build_partition_plan(campaign_id: str):
    """Build a leakage-resistant plan with nonempty development and held-out."""

    mutants = [
        {
            "mutant_id": "mut_diag_1",
            "candidate_cid": _cid("cand-mut_diag_1"),
            "operator_id": "authz_invert",
            "target_id": "mod_fn",
            "used_for_candidate_generation": True,
        },
        *[
            {
                "mutant_id": f"mut_x{index:02d}",
                "candidate_cid": _cid(f"cand-mut_x{index:02d}"),
                "operator_id": "authz_invert",
                "target_id": "mod_fn",
                "used_for_candidate_generation": False,
            }
            for index in range(12)
        ],
    ]
    # Seed until both development and held-out are nonempty (leakage-resistant).
    for seed in range(32):
        plan = partition_mutants(
            mutants,
            ("mut_diag_1",),
            header=_header("mutation_campaign_plan"),
            campaign_id=campaign_id,
            partition_seed=seed,
            development_ratio_bp=4_000,
            held_out_ratio_bp=6_000,
        )
        if plan.development_mutant_ids and plan.held_out_mutant_ids:
            return plan
    raise RuntimeError("unable to build nonempty development/held-out partition plan")


def _campaign(
    *,
    campaign_id: str = "camp_held_out_1",
    fail: frozenset[str] | None = None,
    mock_bypass_partitions: frozenset[str] | None = None,
    overfit_one_mutant: bool = False,
    cost_delta_basis_points: int = 50,
    partition_results: tuple[CampaignPartitionResult, ...] | None = None,
    with_partition_plan: bool = True,
) -> HeldOutCampaign:
    plan = _build_partition_plan(campaign_id) if with_partition_plan else None
    if partition_results is None:
        if plan is not None:
            diagnosis_ids = plan.diagnosis_mutant_ids
            development_ids = plan.development_mutant_ids[:2] or plan.development_mutant_ids
            held_out_ids = plan.held_out_mutant_ids[:2] or plan.held_out_mutant_ids
        else:
            diagnosis_ids = ("mut_diag_1",)
            development_ids = ("mut_dev_1",)
            held_out_ids = ("mut_hold_1", "mut_hold_2")
        results = _all_partition_results(
            diagnosis_ids=tuple(diagnosis_ids),
            development_ids=tuple(development_ids),
            held_out_ids=tuple(held_out_ids),
            fail=fail,
            mock_bypass_partitions=mock_bypass_partitions,
            overfit_one_mutant=overfit_one_mutant,
        )
    else:
        results = partition_results
    return HeldOutCampaign(
        campaign_id=campaign_id,
        header=_header("mutation_campaign_plan"),
        partition_results=results,
        partition_plan=plan,
        cost_delta_basis_points=cost_delta_basis_points,
        notes=None,
        metadata={"fixture": "aae046"},
    )


def _propose() -> RemediationProposalRun:
    return propose_gap_remediation(_survivor(), _gap())


# ---------------------------------------------------------------------------
# Module / descriptor / vocabulary
# ---------------------------------------------------------------------------


def test_module_ast_parses_and_exports_interfaces() -> None:
    source = REMEDIATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "propose_gap_remediation" in names
    assert "evaluate_remediation" in names
    assert "HeldOutCampaign" in names
    assert "RemediationEvaluationRun" in names


def test_descriptors_bind_released_authorities() -> None:
    propose_desc = propose_gap_remediation_descriptor()
    assert propose_desc["interface_id"] == PROPOSE_GAP_REMEDIATION_INTERFACE
    assert propose_desc["production_policy_change"] is False
    assert propose_desc["requires_held_out_evaluation"] is True

    eval_desc = evaluate_remediation_descriptor()
    assert eval_desc["interface_id"] == EVALUATE_REMEDIATION_INTERFACE
    assert eval_desc["evidence"] == AAE_REMEDIATION_EVALUATION_EVIDENCE
    assert eval_desc["generator_id"] == GENERATOR_ID
    assert eval_desc["production_policy_change"] is False
    assert RejectionReason.MOCK_BYPASS.value in eval_desc["rejects"]
    assert (
        RejectionReason.OVERFIT_IMPLEMENTATION_ASSERTION.value
        in eval_desc["rejects"]
    )
    assert ORIGINAL_PARTITION == EvaluationPartition.UNMUTATED.value
    assert PERFORMANCE_PARTITION == EvaluationPartition.PERFORMANCE_COST.value


def test_aae046_partitions_cover_acceptance_list() -> None:
    parts = aae046_evaluation_partitions()
    assert parts == AAE046_EVALUATION_PARTITIONS
    assert len(parts) == 10
    # Acceptance wording maps onto closed tokens.
    assert EvaluationPartition.UNMUTATED.value in parts  # original
    assert EvaluationPartition.DIAGNOSIS.value in parts
    assert EvaluationPartition.DEVELOPMENT.value in parts
    assert EvaluationPartition.HELD_OUT.value in parts
    assert EvaluationPartition.UNRELATED.value in parts
    assert EvaluationPartition.PERFORMANCE_COST.value in parts  # performance
    assert EvaluationPartition.FALSE_POSITIVE.value in parts
    assert EvaluationPartition.OVERCONSTRAINT.value in parts
    assert EvaluationPartition.SAFETY.value in parts
    # Full qualification set matches AAE-033 required partitions.
    assert set(parts) == set(REQUIRED_EVALUATION_PARTITIONS)


# ---------------------------------------------------------------------------
# propose_gap_remediation
# ---------------------------------------------------------------------------


def test_propose_gap_remediation_produces_heuristic_candidates() -> None:
    run = _propose()
    assert isinstance(run, RemediationProposalRun)
    assert run.proposal is not None
    assert run.all_heuristic is True
    assert run.requires_held_out_evaluation is True
    assert run.production_policy_changed is False
    assert run.candidate_cids
    assert run.plan_cid
    assert run.gap_cid
    assert REASON_NO_PRODUCTION_POLICY_CHANGE in run.reason_codes
    assert run.proposal.plan.plan_status == RemediationPlanStatus.DRAFT.value
    for test in run.proposal.candidate_tests:
        assert test.freezes_implementation is False
        assert test.requirement_provenances


def test_propose_accepts_mapping_inputs() -> None:
    run = propose_gap_remediation(_survivor().to_dict(), _gap().to_dict())
    assert run.proposal_cid
    assert run.proposal is not None


def test_propose_is_deterministic() -> None:
    first = propose_gap_remediation(_survivor(), _gap())
    second = propose_gap_remediation(_survivor(), _gap())
    assert first.run_cid == second.run_cid
    assert first.proposal_cid == second.proposal_cid
    assert first.plan_cid == second.plan_cid


def test_propose_rejects_non_remediable_gap() -> None:
    with pytest.raises(RemediationRuntimeError, match="propose_gap_remediation failed"):
        propose_gap_remediation(
            _survivor(),
            _gap(AssuranceGapClass.UNKNOWN, requires_human_review=True),
        )


# ---------------------------------------------------------------------------
# evaluate_remediation — happy path covers all partitions
# ---------------------------------------------------------------------------


def test_evaluate_qualifies_complete_held_out_campaign() -> None:
    proposal = _propose()
    campaign = _campaign()
    run = evaluate_remediation(proposal, campaign)

    assert isinstance(run, RemediationEvaluationRun)
    assert run.qualified is True
    assert run.disposition == QualificationDisposition.QUALIFIED.value
    assert run.verdict == EvaluationVerdict.QUALIFIED.value
    assert run.one_mutant_overfit is False
    assert run.mock_bypass is False
    assert run.production_policy_changed is False
    assert run.missing_partitions == ()
    assert evaluation_covers_acceptance_partitions(run)
    assert REASON_ORIGINAL_EVALUATED in run.reason_codes
    assert REASON_QUALIFIED in run.reason_codes
    assert REASON_NO_PRODUCTION_POLICY_CHANGE in run.reason_codes
    assert run.evaluation_report is not None
    assert run.qualification is not None
    assert run.qualification.disposition == QualificationDisposition.QUALIFIED.value
    assert run.metadata["adapter_id"] == ADAPTER_ID

    # Explicit acceptance partition coverage.
    covered = set(run.partitions_covered)
    for name in (
        "unmutated",
        "diagnosis",
        "development",
        "held_out",
        "unrelated",
        "performance_cost",
        "false_positive",
        "overconstraint",
        "safety",
    ):
        assert name in covered

    report = run.evaluation_report
    assert report.unmutated_suite_passed is True
    assert report.diagnosis_killed is True
    assert report.development_killed is True
    assert report.held_out_killed is True
    assert report.unrelated_behavior_preserved is True
    assert report.safety_preserved is True
    assert report.regression_detected is False
    assert report.overconstraint_detected is False
    assert report.false_positive_detected is False
    verify_evaluation_report_identity(report)


def test_evaluate_accepts_plan_and_mapping_campaign() -> None:
    proposal = _propose()
    campaign = _campaign()
    run = evaluate_remediation(proposal.proposal.plan, campaign.to_dict())
    assert run.qualified is True
    assert run.plan_cid == proposal.plan_cid


def test_evaluate_accepts_proposal_dict() -> None:
    proposal = _propose()
    run = evaluate_remediation(proposal.proposal.to_dict(), _campaign())
    assert run.qualified is True


def test_evaluate_is_deterministic() -> None:
    proposal = _propose()
    campaign = _campaign()
    first = evaluate_remediation(proposal, campaign)
    second = evaluate_remediation(proposal, campaign)
    assert first.run_cid == second.run_cid
    assert first.evaluation_report_cid == second.evaluation_report_cid
    assert first.qualification_cid == second.qualification_cid


# ---------------------------------------------------------------------------
# One-mutant overfit rejection
# ---------------------------------------------------------------------------


def test_evaluate_rejects_one_mutant_overfit() -> None:
    proposal = _propose()
    campaign = _campaign(overfit_one_mutant=True, campaign_id="camp_overfit")
    run = evaluate_remediation(proposal, campaign)

    assert run.qualified is False
    assert run.one_mutant_overfit is True
    assert run.disposition == QualificationDisposition.REJECTED.value
    assert run.verdict == EvaluationVerdict.OVERFIT.value
    assert (
        RejectionReason.OVERFIT_IMPLEMENTATION_ASSERTION.value
        in run.rejection_reasons
    )
    assert REASON_ONE_MUTANT_OVERFIT in run.reason_codes
    assert run.production_policy_changed is False


def test_detect_one_mutant_overfit_diagnosis_without_held_out() -> None:
    campaign = _campaign(overfit_one_mutant=True)
    hit, note = detect_one_mutant_overfit(campaign)
    assert hit is True
    assert note is not None


def test_evaluate_raise_on_hard_reject_overfit() -> None:
    proposal = _propose()
    campaign = _campaign(overfit_one_mutant=True, campaign_id="camp_overfit_raise")
    with pytest.raises(OneMutantOverfitError):
        evaluate_remediation(proposal, campaign, raise_on_hard_reject=True)


def test_explicit_one_mutant_only_flag_is_overfit() -> None:
    base = _campaign(campaign_id="camp_flag_overfit")
    # Force diagnosis-only pattern via flag even if held-out would pass.
    rewritten: list[CampaignPartitionResult] = []
    for item in base.partition_results:
        if item.partition == EvaluationPartition.DIAGNOSIS.value:
            rewritten.append(
                _partition(
                    item.partition,
                    passed=True,
                    mutant_ids=item.mutant_ids,
                    killed_mutant_ids=item.mutant_ids,
                    one_mutant_only=True,
                )
            )
        else:
            rewritten.append(item)
    campaign = HeldOutCampaign(
        campaign_id=base.campaign_id,
        header=base.header,
        partition_results=tuple(rewritten),
        partition_plan=base.partition_plan,
        cost_delta_basis_points=base.cost_delta_basis_points,
        metadata=dict(base.metadata),
    )
    hit, _ = detect_one_mutant_overfit(campaign)
    assert hit is True
    run = evaluate_remediation(_propose(), campaign)
    assert run.one_mutant_overfit is True
    assert run.qualified is False


# ---------------------------------------------------------------------------
# Mock bypass rejection
# ---------------------------------------------------------------------------


def test_evaluate_rejects_mock_bypass() -> None:
    proposal = _propose()
    campaign = _campaign(
        campaign_id="camp_mock",
        mock_bypass_partitions=frozenset({EvaluationPartition.DIAGNOSIS.value}),
    )
    run = evaluate_remediation(proposal, campaign)

    assert run.qualified is False
    assert run.mock_bypass is True
    assert run.disposition == QualificationDisposition.REJECTED.value
    assert RejectionReason.MOCK_BYPASS.value in run.rejection_reasons
    assert REASON_MOCK_BYPASS in run.reason_codes
    assert run.production_policy_changed is False


def test_detect_mock_bypass_reports_partitions() -> None:
    campaign = _campaign(
        campaign_id="camp_mock2",
        mock_bypass_partitions=frozenset(
            {
                EvaluationPartition.HELD_OUT.value,
                EvaluationPartition.SAFETY.value,
            }
        ),
    )
    hit, parts = detect_mock_bypass(campaign)
    assert hit is True
    assert EvaluationPartition.HELD_OUT.value in parts
    assert EvaluationPartition.SAFETY.value in parts


def test_mock_bypass_cannot_claim_passed() -> None:
    with pytest.raises(RemediationRuntimeError, match="mock_bypass"):
        CampaignPartitionResult(
            partition=EvaluationPartition.DIAGNOSIS,
            passed=True,
            mock_bypass=True,
            mutant_ids=("mut_diag_1",),
        )


def test_evaluate_raise_on_hard_reject_mock_bypass() -> None:
    proposal = _propose()
    campaign = _campaign(
        campaign_id="camp_mock_raise",
        mock_bypass_partitions=frozenset({EvaluationPartition.UNMUTATED.value}),
    )
    with pytest.raises(MockBypassError):
        evaluate_remediation(proposal, campaign, raise_on_hard_reject=True)


# ---------------------------------------------------------------------------
# Other partition failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "partition,reason",
    [
        (EvaluationPartition.UNMUTATED.value, RejectionReason.UNMUTATED_SUITE_FAILED.value),
        (EvaluationPartition.DIAGNOSIS.value, RejectionReason.DIAGNOSIS_NOT_KILLED.value),
        (EvaluationPartition.HELD_OUT.value, RejectionReason.HELD_OUT_FAILURE.value),
        (EvaluationPartition.UNRELATED.value, RejectionReason.UNRELATED_BEHAVIOR_BROKEN.value),
        (EvaluationPartition.SAFETY.value, RejectionReason.SAFETY_WEAKENING.value),
        (EvaluationPartition.REGRESSION.value, RejectionReason.REGRESSION.value),
        (EvaluationPartition.OVERCONSTRAINT.value, RejectionReason.OVERCONSTRAINT.value),
        (EvaluationPartition.FALSE_POSITIVE.value, RejectionReason.FALSE_POSITIVE.value),
        (
            EvaluationPartition.PERFORMANCE_COST.value,
            RejectionReason.UNAPPROVED_COST_INCREASE.value,
        ),
    ],
)
def test_evaluate_rejects_failed_partition(partition: str, reason: str) -> None:
    proposal = _propose()
    campaign = _campaign(
        campaign_id=f"camp_fail_{partition}",
        fail=frozenset({partition}),
    )
    # Cost failure also needs elevated cost when performance_cost "passes" flags
    # are derived from partition.passed — fail set already forces passed=False.
    run = evaluate_remediation(proposal, campaign)
    assert run.qualified is False
    assert reason in run.rejection_reasons
    assert partition in run.failed_partitions or partition in {
        # cost reason can also fire from cost_delta comparison path
        EvaluationPartition.PERFORMANCE_COST.value,
    }


def test_evaluate_rejects_missing_partition() -> None:
    results = [
        item
        for item in _all_partition_results()
        if item.partition != EvaluationPartition.SAFETY.value
    ]
    campaign = _campaign(
        campaign_id="camp_missing_safety",
        partition_results=tuple(results),
        with_partition_plan=False,
    )
    run = evaluate_remediation(_propose(), campaign)
    assert run.qualified is False
    assert EvaluationPartition.SAFETY.value in run.missing_partitions
    assert RejectionReason.HELD_OUT_FAILURE.value in run.rejection_reasons
    assert run.evaluation_report is None


def test_evaluate_rejects_cost_exceeded() -> None:
    proposal = _propose()
    campaign = _campaign(campaign_id="camp_cost", cost_delta_basis_points=5_000)
    run = evaluate_remediation(proposal, campaign, max_cost_delta_bp=100)
    assert run.qualified is False
    assert RejectionReason.UNAPPROVED_COST_INCREASE.value in run.rejection_reasons


def test_partition_aliases_original_and_performance() -> None:
    result_original = CampaignPartitionResult(partition="original", passed=True)
    result_perf = CampaignPartitionResult(partition="performance", passed=True)
    assert result_original.partition == EvaluationPartition.UNMUTATED.value
    assert result_perf.partition == EvaluationPartition.PERFORMANCE_COST.value


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_production_policy_never_changed() -> None:
    proposal = _propose()
    good = evaluate_remediation(proposal, _campaign())
    bad = evaluate_remediation(
        proposal, _campaign(overfit_one_mutant=True, campaign_id="camp_pol")
    )
    assert good.production_policy_changed is False
    assert bad.production_policy_changed is False
    assert proposal.production_policy_changed is False
    assert good.metadata["production_policy_changed"] is False


def test_run_round_trip_identity_fields() -> None:
    run = evaluate_remediation(_propose(), _campaign())
    payload = run.to_dict()
    assert payload["run_cid"] == run.run_cid
    assert payload["qualified"] is True
    assert payload["evaluation_report"]["report_cid"] == run.evaluation_report_cid
    assert payload["qualification"]["result_cid"] == run.qualification_cid
    # Recompute identity without full objects.
    restored_identity = RemediationEvaluationRun(
        campaign_id=run.campaign_id,
        campaign_cid=run.campaign_cid,
        plan_cid=run.plan_cid,
        candidate_cids=run.candidate_cids,
        phases=run.phases,
        evaluation_report_cid=run.evaluation_report_cid,
        qualification_cid=run.qualification_cid,
        disposition=run.disposition,
        verdict=run.verdict,
        partitions_covered=run.partitions_covered,
        missing_partitions=run.missing_partitions,
        failed_partitions=run.failed_partitions,
        one_mutant_overfit=run.one_mutant_overfit,
        mock_bypass=run.mock_bypass,
        qualified=run.qualified,
        reason_codes=run.reason_codes,
        rejection_reasons=run.rejection_reasons,
        diagnostic=run.diagnostic,
        metadata=dict(run.metadata),
    )
    assert restored_identity.run_cid == run.run_cid


def test_campaign_identity_is_stable() -> None:
    first = _campaign()
    second = _campaign()
    assert first.campaign_cid == second.campaign_cid


def test_held_out_campaign_from_dict_round_trip() -> None:
    campaign = _campaign()
    restored = HeldOutCampaign.from_dict(campaign.to_dict())
    assert restored.campaign_id == campaign.campaign_id
    assert restored.campaign_cid == campaign.campaign_cid
    assert len(restored.partition_results) == len(campaign.partition_results)


def test_interfaces_are_versioned_pins() -> None:
    assert PROPOSE_GAP_REMEDIATION_INTERFACE.endswith("@1")
    assert EVALUATE_REMEDIATION_INTERFACE.endswith("@1")
    assert "@" in PROPOSE_GAP_REMEDIATION_INTERFACE
