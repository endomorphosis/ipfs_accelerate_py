"""AAE-061: Qualify held-out remediation, controlled promotion, and initial success targets.

Validates ``AAERemediationPromotionQualification@1`` / ``aae/promotion-e2e@1``:

* Held-out remediation evaluation is mandatory before any promotion attempt.
* Every promoted candidate carries held-out evidence and a qualified evaluation.
* Critical regression and new vacuity block promotion (head unchanged).
* Cost/coverage declaration, external authorization, expected-old CAS, and a
  released incremental seal are mandatory for ``promoted`` outcomes.
* Candidates cannot self-promote; unauthorized attempts leave the policy head
  unchanged.
* Plan §15 success targets (zero / 90% / 50%) are goals, never fabricated
  passes: unmet targets remain explicit results with ``fabricated_pass=False``.
* Disposable coordination-store CAS only; no production policy change from
  pure fixture campaigns.

Depends on AAE-046 (evaluate_remediation), AAE-047 (promote_assurance_policy),
AAE-058 (metrics/success targets), AAE-059/060 isolation/CAS invariants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.metrics import (
    BASIS_POINTS,
    AssuranceMetrics,
    compute_assurance_metrics,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.promotion import (
    AAE_PROMOTION_EVIDENCE,
    PROMOTE_ASSURANCE_POLICY_INTERFACE,
    AssurancePolicyPromotionResult,
    PromotionStatus,
    REASON_ABSENT_AUTHORIZATION,
    REASON_ABSENT_SEAL,
    REASON_COST_NOT_DECLARED,
    REASON_COVERAGE_NOT_DECLARED,
    REASON_EVALUATION_NOT_PASS,
    REASON_HELD_OUT_NOT_PASS,
    REASON_REGRESSION_DETECTED,
    REASON_SEAL_UNAVAILABLE,
    REASON_SELF_PROMOTION,
    REASON_VACUITY_DETECTED,
    promote_assurance_policy,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.remediation import (
    AAE_REMEDIATION_EVALUATION_EVIDENCE,
    CampaignPartitionResult,
    HeldOutCampaign,
    RemediationEvaluationRun,
    RemediationProposalRun,
    evaluate_remediation,
    propose_gap_remediation,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.reporting import (
    DEFAULT_SUCCESS_TARGETS,
    build_assurance_report,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
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
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    MutationOutcomeStatus,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.held_out import (
    QualificationDisposition,
    partition_mutants,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.receipt_contracts import (
    EXISTING_SIGNATURE_ALGORITHM,
    EXISTING_SIGNATURE_AUTHORITY,
    AssuranceCampaignReceipt,
    HeldOutResult,
    ReceiptAction,
    ReceiptSignatureBinding,
    SealAvailabilityStatus,
    SealScopeItem,
    SignatureVerificationStatus,
    verify_campaign_receipt_identity,
    verify_promotion_receipt_identity,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.remediation_contracts import (
    EvaluationPartition,
    EvaluationVerdict,
    verify_evaluation_report_identity,
)
from ipfs_kit_py.adversarial_assurance_store.policy import (
    DurableAssurancePolicyRepository,
)
from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
    DurableCoordinationStore,
    cid_for_artifact,
)


# ---------------------------------------------------------------------------
# Qualification constants
# ---------------------------------------------------------------------------

INTERFACE = "AAERemediationPromotionQualification@1"
EVIDENCE = "aae/promotion-e2e@1"
TASK_ID = "AAE-061"
BUNDLE = "adversarial-assurance/promotion-e2e"
WORKSPACE = "aae061-promotion-e2e"

REPO_ROOT = Path(__file__).resolve().parents[3]
REMEDIATION_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/remediation.py"
)
PROMOTION_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/promotion.py"
)

_SIGNER = "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
_SIGNATURE = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAA"
)

# Plan §15 numeric goals (basis points where applicable).
ZERO_TARGET_COUNT = 0
HIGH_RISK_DETECTION_TARGET_BP = 9_000  # 90 percent
COMPUTE_SAVINGS_TARGET_BP = 5_000  # 50 percent

REQUIRED_PROMOTION_GATES = frozenset(
    {
        "held_out_evidence",
        "no_critical_regression",
        "no_new_vacuity",
        "cost_declared",
        "coverage_declared",
        "authorization",
        "expected_old_cas",
        "released_incremental_seal",
    }
)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    """Canonical dag-json CID (store CAS requires dag-json profile)."""

    return cid_for_structured({"test_label": label, "schema": "test/aae061@1"})


def _block(store: DurableCoordinationStore, name: str, **extra: Any) -> str:
    payload = {"schema": "example/assurance-policy@1", "name": name, "task": TASK_ID}
    payload.update(extra)
    return store.put(payload, expected_cid=cid_for_artifact(payload), replicate=False)[
        "cid"
    ]


def _generator(**overrides: object) -> GeneratorIdentity:
    fields = {
        "generator_id": "remediation_promotion_e2e",
        "generator_version": "1.0.0",
        "interface_id": "evaluate_remediation@1",
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
        "tool_ids": ("remediation_promotion_e2e.v1",),
        "policy_cid": _cid("policy-baseline"),
        "notes": None,
    }
    fields.update(overrides)
    return ArtifactProvenance(**fields)  # type: ignore[arg-type]


def _header(artifact_kind: str = "mutation_campaign_plan", **overrides: object) -> AssuranceArtifactHeader:
    fields = {
        "artifact_kind": artifact_kind,
        "repository_id": "repository:sha256:test-repo-identity-aae061",
        "repository_state_cid": _cid("repo-state-aae061"),
        "target_symbol_ids": ("mod.fn",),
        "target_artifact_cids": (_cid("artifact-a"),),
        "capsule_cids": (_cid("capsule-a"),),
        "proof_unit_cids": (_cid("proof-unit-a"),),
        "environment_cid": _cid("environment-aae061"),
        "dependency_lock_cid": _cid("dependency-lock-aae061"),
        "versions": _versions(),
        "provenance": _provenance(),
        "terminal_status": AssuranceTerminalStatus.COMPLETE,
        "receipt_cids": (_cid("receipt-a"),),
        "proof_cids": (_cid("proof-a"),),
        "metadata": {"task_id": TASK_ID, "evidence": EVIDENCE},
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
        "report_id": "survivor_authz_aae061",
        "candidate_id": "cand_authz_invert_0",
        "candidate_cid": _cid("candidate-aae061"),
        "outcome_cid": _cid("outcome-aae061"),
        "risk_class": SurvivorRiskClass.AUTHORIZATION,
        "symbol_ids": ("mod.fn",),
        "violated_or_missing_property": (
            "authorization check must reject unauthorized callers"
        ),
        "detectors_run": ("unit.test_branch",),
        "detectors_omitted": ("static.authz_rule",),
        "expected_behavior": "reject unauthorized caller for protected action",
        "observed_behavior": "unauthorized caller accepted and side effect applied",
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
    fields = {
        "header": _header("assurance_gap"),
        "gap_id": "gap_authz_missing_test_aae061",
        "gap_class": gap_class,
        "severity": GapSeverity.CRITICAL,
        "risk_class": survivor.risk_class,
        "summary": f"assurance gap {gap_class} for authorization",
        "candidate_id": survivor.candidate_id,
        "candidate_cid": survivor.candidate_cid,
        "survivor_report_cid": survivor.report_cid,
        "violated_or_missing_property": survivor.violated_or_missing_property,
        "symbol_ids": survivor.symbol_ids,
        "source_spans": survivor.source_spans,
        "dependency_path": survivor.dependency_path,
        "minimized_evidence": survivor.minimized_evidence,
        "requires_human_review": False,
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
    one_mutant_only: bool = False,
) -> CampaignPartitionResult:
    if killed_mutant_ids is None:
        killed_mutant_ids = mutant_ids if passed and mutant_ids else ()
    return CampaignPartitionResult(
        partition=partition,
        passed=passed,
        mutant_ids=mutant_ids,
        killed_mutant_ids=killed_mutant_ids,
        mock_bypass=mock_bypass,
        freezes_implementation=False,
        one_mutant_only=one_mutant_only,
        evidence_cids=(_cid(f"ev-{partition}"),),
    )


def _all_partition_results(
    *,
    diagnosis_ids: tuple[str, ...] = ("mut_diag_1",),
    development_ids: tuple[str, ...] = ("mut_dev_1",),
    held_out_ids: tuple[str, ...] = ("mut_hold_1", "mut_hold_2"),
    fail: frozenset[str] | None = None,
    overfit_one_mutant: bool = False,
) -> tuple[CampaignPartitionResult, ...]:
    fail = fail or frozenset()
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
        if overfit_one_mutant and value == EvaluationPartition.HELD_OUT.value:
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
                passed=value not in fail,
                mutant_ids=mutant_ids,
            )
        )
    return tuple(results)


def _build_partition_plan(campaign_id: str):
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


def _held_out_campaign(
    *,
    campaign_id: str = "camp_aae061_held_out",
    fail: frozenset[str] | None = None,
    overfit_one_mutant: bool = False,
    cost_delta_basis_points: int = 50,
) -> HeldOutCampaign:
    plan = _build_partition_plan(campaign_id)
    diagnosis_ids = plan.diagnosis_mutant_ids
    development_ids = plan.development_mutant_ids[:2] or plan.development_mutant_ids
    held_out_ids = plan.held_out_mutant_ids[:2] or plan.held_out_mutant_ids
    results = _all_partition_results(
        diagnosis_ids=tuple(diagnosis_ids),
        development_ids=tuple(development_ids),
        held_out_ids=tuple(held_out_ids),
        fail=fail,
        overfit_one_mutant=overfit_one_mutant,
    )
    return HeldOutCampaign(
        campaign_id=campaign_id,
        header=_header("mutation_campaign_plan"),
        partition_results=results,
        partition_plan=plan,
        cost_delta_basis_points=cost_delta_basis_points,
        notes=None,
        metadata={"fixture": "aae061", "evidence": EVIDENCE},
    )


def _signature(**overrides: object) -> ReceiptSignatureBinding:
    fields = {
        "signer_identity": _SIGNER,
        "key_identity": _SIGNER,
        "audience": "adversarial_assurance.store",
        "action": ReceiptAction.COMPLETE_CAMPAIGN,
        "signature": _SIGNATURE,
        "signature_verification_status": SignatureVerificationStatus.VERIFIED,
        "signature_algorithm": EXISTING_SIGNATURE_ALGORITHM,
        "signature_authority": EXISTING_SIGNATURE_AUTHORITY,
    }
    fields.update(overrides)
    return ReceiptSignatureBinding(**fields)  # type: ignore[arg-type]


def _campaign_scope() -> tuple[str, ...]:
    return (
        SealScopeItem.OPERATOR_VERSIONS.value,
        SealScopeItem.CAMPAIGN_POLICY.value,
        SealScopeItem.ADMITTED_SET.value,
        SealScopeItem.EXPECTED_DETECTION_SETS.value,
        SealScopeItem.OUTCOMES.value,
        SealScopeItem.SURVIVOR_REPORTS.value,
        SealScopeItem.VACUITY_FINDINGS.value,
        SealScopeItem.HELD_OUT_EVALUATIONS.value,
        SealScopeItem.CAMPAIGN_ARTIFACTS.value,
        SealScopeItem.DECLARED_RESULT_COMPLETENESS.value,
        SealScopeItem.CAMPAIGN_RECEIPT.value,
    )


def _campaign_receipt(**overrides: object) -> AssuranceCampaignReceipt:
    fields = {
        "header": _header("assurance_campaign_receipt"),
        "receipt_id": "campaign_receipt_aae061",
        "campaign_plan_cid": _cid("plan"),
        "campaign_policy_cid": _cid("campaign-policy"),
        "campaign_policy_version": "1.0.0",
        "admitted_set_cid": _cid("admitted"),
        "expected_detection_sets_cid": _cid("expected-detection"),
        "outcomes_cid": _cid("outcomes"),
        "survivor_reports_cid": _cid("survivors"),
        "vacuity_findings_cid": _cid("vacuity"),
        "held_out_evaluation_cid": _cid("held-out-eval"),
        "held_out_result": HeldOutResult.PASSED,
        "authorization_cid": _cid("campaign-external-authorization"),
        "expected_old_revision": "0.9.0",
        "seal_scope": _campaign_scope(),
        "seal_status": SealAvailabilityStatus.BOUND,
        "seal_evidence_cid": _cid("campaign-seal-evidence"),
        "gap_reports_cid": _cid("gaps"),
        "input_artifact_cids": (_cid("input-plan"), _cid("input-policy")),
        "signature": _signature(action=ReceiptAction.COMPLETE_CAMPAIGN),
        "notes": None,
        "metadata": {"task_id": TASK_ID},
    }
    fields.update(overrides)
    return AssuranceCampaignReceipt(**fields)  # type: ignore[arg-type]


def _promo_signature(**overrides: object) -> ReceiptSignatureBinding:
    return _signature(action=ReceiptAction.PROMOTE_POLICY, **overrides)


def _propose() -> RemediationProposalRun:
    return propose_gap_remediation(_survivor(), _gap())


def _evaluate(
    proposal: RemediationProposalRun | None = None,
    campaign: HeldOutCampaign | None = None,
    **kwargs: Any,
) -> RemediationEvaluationRun:
    proposal = proposal if proposal is not None else _propose()
    campaign = campaign if campaign is not None else _held_out_campaign()
    return evaluate_remediation(proposal, campaign, **kwargs)


def _promotion_candidate(
    *,
    proposal: RemediationProposalRun,
    evaluation: RemediationEvaluationRun,
    baseline: str,
    promoted: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Bridge held-out remediation identity into a promotion candidate pin set."""

    candidate_cid = (
        proposal.candidate_cids[0]
        if proposal.candidate_cids
        else proposal.plan_cid
    )
    cost = (
        evaluation.evaluation_report.cost_delta_basis_points
        if evaluation.evaluation_report is not None
        else 50
    )
    partitions = tuple(evaluation.partitions_covered) or (
        "unmutated",
        "held_out",
        "regression",
        "performance_cost",
    )
    fields: dict[str, Any] = {
        "candidate_cid": candidate_cid,
        "plan_cid": proposal.plan_cid,
        "proposed_policy_cid": promoted,
        "base_policy_cid": baseline,
        "base_policy_version": "1.0.0",
        "proposed_policy_version": "1.0.1",
        "cost_delta_basis_points": cost,
        "coverage_declared": True,
        "coverage_partitions": partitions,
        "metadata": {
            "task_id": TASK_ID,
            "held_out_evaluation_cid": evaluation.evaluation_report_cid,
            "qualification_cid": evaluation.qualification_cid,
            "campaign_id": evaluation.campaign_id,
        },
    }
    fields.update(overrides)
    return fields


def _assert_no_mutation(
    result: AssurancePolicyPromotionResult,
    policy_repo: DurableAssurancePolicyRepository,
    *,
    expected_cid: str,
    expected_generation: int,
) -> None:
    assert result.head_mutated is False
    assert result.status != PromotionStatus.PROMOTED.value
    head = policy_repo.current_policy(WORKSPACE)
    assert head.policy_cid == expected_cid
    assert head.generation == expected_generation


# ---------------------------------------------------------------------------
# Success-target evaluation (goals, never fabricated results)
# ---------------------------------------------------------------------------


def evaluate_initial_success_targets(
    *,
    metrics: AssuranceMetrics,
    promotions: Sequence[Mapping[str, Any]],
    critical_security_survivors_after_remediation: int,
    accepted_stale_proof_or_seal_mutants: int,
    high_risk_survivors_without_gap: int,
    vacuous_proof_claims_meaningful: int,
    campaign_ids: Sequence[str],
    worktree_escape_count: int,
    unauthorized_production_policy_changes: int,
) -> dict[str, Any]:
    """Honest plan §15 target evaluation for AAE-061.

    Returns explicit per-target results. Unmet zero/90/50-percent goals are
    reported as ``unmet`` / ``unavailable`` — never as fabricated passes.
    """

    goals = dict(DEFAULT_SUCCESS_TARGETS)
    assert goals["targets_are_goals_not_results"] is True

    # Zero-rate targets (counts must be zero).
    zero_security = {
        "target_id": "zero_controlled_critical_security_survivors_after_remediation",
        "goal": True,
        "observed_count": int(critical_security_survivors_after_remediation),
        "target_count": ZERO_TARGET_COUNT,
        "met": critical_security_survivors_after_remediation == ZERO_TARGET_COUNT,
    }
    zero_stale = {
        "target_id": "zero_accepted_stale_proof_or_seal_integrity_mutants",
        "goal": True,
        "observed_count": int(accepted_stale_proof_or_seal_mutants),
        "target_count": ZERO_TARGET_COUNT,
        "met": accepted_stale_proof_or_seal_mutants == ZERO_TARGET_COUNT,
    }

    # 90 percent high-risk semantic detection (risk-weighted kill score).
    observed_detection_bp = metrics.mutation_coverage.risk_weighted_score_bp
    detection_met: bool | None
    detection_status: str
    if observed_detection_bp is None:
        detection_met = None
        detection_status = "unavailable"
    elif observed_detection_bp >= HIGH_RISK_DETECTION_TARGET_BP:
        detection_met = True
        detection_status = "met"
    else:
        detection_met = False
        detection_status = "unmet"
    ninety = {
        "target_id": "high_risk_semantic_detection_min_bp",
        "goal_bp": HIGH_RISK_DETECTION_TARGET_BP,
        "observed_bp": observed_detection_bp,
        "met": detection_met,
        "status": detection_status,
        "percent_goal": 90,
    }

    # 50 percent compute savings.
    observed_savings_bp = metrics.economics.savings_rate_bp
    savings_met: bool | None
    savings_status: str
    if observed_savings_bp is None:
        savings_met = None
        savings_status = "unavailable"
    elif observed_savings_bp >= COMPUTE_SAVINGS_TARGET_BP:
        savings_met = True
        savings_status = "met"
    else:
        savings_met = False
        savings_status = "unmet"
    fifty = {
        "target_id": "compute_savings_min_bp",
        "goal_bp": COMPUTE_SAVINGS_TARGET_BP,
        "observed_bp": observed_savings_bp,
        "met": savings_met,
        "status": savings_status,
        "percent_goal": 50,
    }

    # Held-out for every promotion (acceptance hard gate, also a success target).
    promo_rows: list[dict[str, Any]] = []
    all_held_out = True
    for index, promo in enumerate(promotions):
        held_out = str(promo.get("held_out_result") or "")
        status = str(promo.get("status") or "")
        is_promoted = status == PromotionStatus.PROMOTED.value
        has_held_out = held_out == HeldOutResult.PASSED.value and bool(
            promo.get("evaluation_report_cid") or promo.get("held_out_evidence")
        )
        if is_promoted and not has_held_out:
            all_held_out = False
        promo_rows.append(
            {
                "index": index,
                "status": status,
                "held_out_result": held_out or None,
                "held_out_evidence_present": has_held_out,
                "promoted": is_promoted,
            }
        )
    held_out_target = {
        "target_id": "held_out_evaluation_for_every_promotion",
        "goal": True,
        "promotions": promo_rows,
        "met": all_held_out,
    }

    explicit_gap = {
        "target_id": "explicit_gap_for_every_high_risk_survivor",
        "goal": True,
        "high_risk_survivors_without_gap": int(high_risk_survivors_without_gap),
        "met": high_risk_survivors_without_gap == 0,
    }
    vacuity_claim = {
        "target_id": "no_meaningful_claim_for_vacuous_proof",
        "goal": True,
        "vacuous_proof_claims_meaningful": int(vacuous_proof_claims_meaningful),
        "met": vacuous_proof_claims_meaningful == 0,
    }
    deterministic_ids = {
        "target_id": "deterministic_campaign_ids",
        "goal": True,
        "campaign_ids": list(campaign_ids),
        "met": len(campaign_ids) == len(set(campaign_ids)) and all(campaign_ids),
    }
    worktree = {
        "target_id": "no_worktree_escape",
        "goal": True,
        "escape_count": int(worktree_escape_count),
        "met": worktree_escape_count == 0,
    }
    unauthorized = {
        "target_id": "no_unauthorized_production_policy_change",
        "goal": True,
        "unauthorized_changes": int(unauthorized_production_policy_changes),
        "met": unauthorized_production_policy_changes == 0,
    }

    target_results = {
        zero_security["target_id"]: zero_security,
        zero_stale["target_id"]: zero_stale,
        ninety["target_id"]: ninety,
        fifty["target_id"]: fifty,
        held_out_target["target_id"]: held_out_target,
        explicit_gap["target_id"]: explicit_gap,
        vacuity_claim["target_id"]: vacuity_claim,
        deterministic_ids["target_id"]: deterministic_ids,
        worktree["target_id"]: worktree,
        unauthorized["target_id"]: unauthorized,
    }

    unmet: list[str] = []
    unavailable: list[str] = []
    met: list[str] = []
    for tid, row in target_results.items():
        if row.get("status") == "unavailable" or row.get("met") is None:
            unavailable.append(tid)
        elif row.get("met") is True:
            met.append(tid)
        else:
            unmet.append(tid)

    # Never promote unmet/unavailable into a fabricated pass.
    overall_pass = not unmet and not unavailable
    return {
        "interface": INTERFACE,
        "evidence": EVIDENCE,
        "task_id": TASK_ID,
        "schema": goals.get("schema"),
        "targets_are_goals_not_results": True,
        "fabricated_pass": False,
        "goals": {
            "zero_controlled_critical_security_survivors_after_remediation": True,
            "zero_accepted_stale_proof_or_seal_integrity_mutants": True,
            "high_risk_semantic_detection_min_bp": HIGH_RISK_DETECTION_TARGET_BP,
            "compute_savings_min_bp": COMPUTE_SAVINGS_TARGET_BP,
        },
        "target_results": target_results,
        "met_targets": sorted(met),
        "unmet_targets": sorted(unmet),
        "unavailable_targets": sorted(unavailable),
        "overall_pass": overall_pass,
        "metrics_cid": metrics.metrics_cid,
        "production_policy_changed": False,
    }


def _outcome(
    candidate_id: str,
    status: str,
    *,
    operator_class: str = "authorization",
    risk_weight_bp: int = 5_000,
    observed: list[str] | None = None,
    killing_id: str | None = None,
    killing_kind: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "outcome_status": status,
        "operator_class": operator_class,
        "risk_weight_bp": risk_weight_bp,
        "predicted_detector_ids": ["det_test"],
        "selected_detector_ids": ["det_test"],
        "executed_detector_ids": ["det_test"],
        "observed_detector_ids": list(observed or []),
        "detector_kinds": {"det_test": "unit_test"},
    }
    if killing_id is not None:
        payload["killing_detector_id"] = killing_id
        payload["killing_detector_kind"] = killing_kind
    return payload


def _unmet_target_campaign_metrics() -> AssuranceMetrics:
    """Fixture metrics that intentionally miss 90% detection and 50% savings."""

    outcomes = [
        # 1 kill + 3 high-weight survivors → risk-weighted well below 90%.
        _outcome(
            "cand_kill",
            MutationOutcomeStatus.KILLED_BY_TEST.value,
            risk_weight_bp=5_000,
            observed=["det_test"],
            killing_id="det_test",
            killing_kind="unit_test",
        ),
        _outcome(
            "cand_surv_1",
            MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value,
            risk_weight_bp=9_000,
        ),
        _outcome(
            "cand_surv_2",
            MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value,
            risk_weight_bp=9_000,
        ),
        _outcome(
            "cand_surv_3",
            MutationOutcomeStatus.SURVIVED_FULL_VERIFICATION.value,
            risk_weight_bp=9_000,
        ),
        # Critical security survivor remaining after "remediation" attempt.
        _outcome(
            "cand_crit_surv",
            MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value,
            operator_class="authorization",
            risk_weight_bp=10_000,
        ),
    ]
    gaps = [
        {
            "gap_id": "gap_crit_surv",
            "gap_class": "missing_test",
            "risk_class": "critical_security",
            "high_risk": True,
        },
    ]
    remediations = [
        {
            "remediation_id": "rem_fail_held_out",
            "disposition": "rejected",
            "held_out_kill_count": 0,
            "evaluated": True,
            "cost_cpu_ms": 100,
        },
    ]
    # Savings: full=1000, incremental=800 → 20% (< 50% target).
    economics_records = [
        {
            "economics_id": "eco_low_savings",
            "full_cpu_ms": 1_000,
            "incremental_cpu_ms": 800,
            "full_wall_ms": 1_000,
            "incremental_wall_ms": 800,
            "cache_hits": 1,
            "cache_misses": 4,
            "model_calls": 0,
            "model_tokens": 0,
        },
    ]
    return compute_assurance_metrics(
        campaign_id="camp_aae061_unmet_targets",
        outcomes=outcomes,
        gaps=gaps,
        remediations=remediations,
        economics_records=economics_records,
        plan_id="plan_aae061_unmet",
        plan_cid=_cid("plan-unmet"),
        result_cid=_cid("result-unmet"),
        repository_state_cid=_cid("repo-state-unmet"),
        generated_count=5,
        admitted_count=5,
        notes="aae-061 intentional unmet zero/90/50 targets",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    return tmp_path / "aae061-policy-cas"


@pytest.fixture()
def coordination(store_dir: Path) -> DurableCoordinationStore:
    root = DurableCoordinationStore(store_dir)
    yield root
    root.close()


@pytest.fixture()
def policy_repo(
    coordination: DurableCoordinationStore,
) -> DurableAssurancePolicyRepository:
    return DurableAssurancePolicyRepository(coordination)


@pytest.fixture()
def seeded_policies(
    coordination: DurableCoordinationStore,
    policy_repo: DurableAssurancePolicyRepository,
) -> dict[str, str]:
    baseline = _block(coordination, "policy-baseline-v1")
    promoted = _block(coordination, "policy-promoted-v2")
    other = _block(coordination, "policy-other-v3")
    cas = policy_repo.compare_and_swap_policy(
        WORKSPACE,
        expected_generation=0,
        expected_policy_cid=None,
        new_policy_cid=baseline,
        operation_id="seed-baseline-aae061",
    )
    assert cas.status.value == "updated"
    return {"baseline": baseline, "promoted": promoted, "other": other}


# ---------------------------------------------------------------------------
# Interface / surface
# ---------------------------------------------------------------------------


def test_qualification_interface_and_evidence_pins() -> None:
    assert INTERFACE == "AAERemediationPromotionQualification@1"
    assert EVIDENCE == "aae/promotion-e2e@1"
    assert TASK_ID == "AAE-061"
    assert BUNDLE == "adversarial-assurance/promotion-e2e"
    assert PROMOTE_ASSURANCE_POLICY_INTERFACE == "promote_assurance_policy@1"
    assert AAE_PROMOTION_EVIDENCE == "aae/promotion@1"
    assert AAE_REMEDIATION_EVALUATION_EVIDENCE == "aae/remediation-evaluation@1"
    assert REMEDIATION_PATH.is_file()
    assert PROMOTION_PATH.is_file()
    assert DEFAULT_SUCCESS_TARGETS["targets_are_goals_not_results"] is True
    assert (
        DEFAULT_SUCCESS_TARGETS["high_risk_semantic_detection_min_bp"]
        == HIGH_RISK_DETECTION_TARGET_BP
    )
    assert DEFAULT_SUCCESS_TARGETS["compute_savings_min_bp"] == COMPUTE_SAVINGS_TARGET_BP
    assert REQUIRED_PROMOTION_GATES == {
        "held_out_evidence",
        "no_critical_regression",
        "no_new_vacuity",
        "cost_declared",
        "coverage_declared",
        "authorization",
        "expected_old_cas",
        "released_incremental_seal",
    }


# ---------------------------------------------------------------------------
# Happy path: held-out remediation → controlled promotion
# ---------------------------------------------------------------------------


def test_happy_path_held_out_remediation_then_controlled_promotion(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]

    proposal = _propose()
    assert proposal.requires_held_out_evaluation is True
    assert proposal.production_policy_changed is False
    assert proposal.all_heuristic is True

    campaign = _held_out_campaign(campaign_id="camp_aae061_happy")
    evaluation = evaluate_remediation(proposal, campaign)

    assert evaluation.qualified is True
    assert evaluation.disposition == QualificationDisposition.QUALIFIED.value
    assert evaluation.verdict == EvaluationVerdict.QUALIFIED.value
    assert evaluation.one_mutant_overfit is False
    assert evaluation.mock_bypass is False
    assert evaluation.production_policy_changed is False
    assert EvaluationPartition.HELD_OUT.value in evaluation.partitions_covered
    assert evaluation.evaluation_report is not None
    assert evaluation.evaluation_report.held_out_killed is True
    assert evaluation.evaluation_report.regression_detected is False
    verify_evaluation_report_identity(evaluation.evaluation_report)

    candidate = _promotion_candidate(
        proposal=proposal,
        evaluation=evaluation,
        baseline=baseline,
        promoted=promoted,
    )
    auth = _cid("external-operator-authorization-happy")
    seal = _cid("released-incremental-seal-happy")
    campaign_receipt = _campaign_receipt(
        held_out_evaluation_cid=evaluation.evaluation_report_cid
        or _cid("held-out-eval-happy")
    )

    result = promote_assurance_policy(
        candidate,
        evaluation,
        auth,
        campaign_receipt=campaign_receipt,
        policy_repository=policy_repo,
        operation_id="promote-aae061-happy",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=seal,
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
        expected_generation=1,
        expected_policy_cid=baseline,
    )

    assert result.status == PromotionStatus.PROMOTED.value
    assert result.head_mutated is True
    assert result.blocking_reasons == ()
    assert result.promoted_policy_cid == promoted
    assert result.candidate_cid == candidate["candidate_cid"]
    assert result.authorization_cid == auth
    assert result.seal_evidence_cid == seal
    assert result.held_out_result == HeldOutResult.PASSED.value
    assert result.evaluation_report_cid == evaluation.evaluation_report_cid
    assert result.receipt is not None
    assert result.receipt.signature.action == ReceiptAction.PROMOTE_POLICY.value
    assert (
        result.receipt.signature.signature_verification_status
        == SignatureVerificationStatus.VERIFIED.value
    )
    assert result.receipt.seal_status == SealAvailabilityStatus.RELEASED.value
    assert result.receipt.expected_old_policy_cid == baseline
    assert result.policy_cas is not None
    assert result.policy_cas["status"] == "updated"
    assert policy_repo.current_policy(WORKSPACE).policy_cid == promoted
    assert policy_repo.current_policy(WORKSPACE).generation == 2
    verify_promotion_receipt_identity(result.receipt)
    verify_campaign_receipt_identity(campaign_receipt)

    # Qualification envelope: promoted path carries held-out evidence + gates.
    qualification = {
        "interface": INTERFACE,
        "evidence": EVIDENCE,
        "task_id": TASK_ID,
        "promoted": True,
        "held_out_evidence": True,
        "held_out_result": result.held_out_result,
        "evaluation_report_cid": result.evaluation_report_cid,
        "qualification_cid": evaluation.qualification_cid,
        "regression_detected": False,
        "vacuity_detected": False,
        "cost_declared": True,
        "coverage_declared": True,
        "authorization_present": True,
        "cas_updated": True,
        "seal_released": True,
        "production_policy_changed": False,
        "gates_satisfied": sorted(REQUIRED_PROMOTION_GATES),
    }
    assert qualification["held_out_evidence"] is True
    assert set(qualification["gates_satisfied"]) == REQUIRED_PROMOTION_GATES


def test_every_promoted_candidate_requires_held_out_evidence(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    proposal = _propose()

    # Held-out failure: evaluation rejects → promotion must not mutate head.
    failed_campaign = _held_out_campaign(
        campaign_id="camp_aae061_held_out_fail",
        fail=frozenset({EvaluationPartition.HELD_OUT.value}),
    )
    failed_eval = evaluate_remediation(proposal, failed_campaign)
    assert failed_eval.qualified is False
    assert EvaluationPartition.HELD_OUT.value in failed_eval.failed_partitions or (
        failed_eval.evaluation_report is not None
        and failed_eval.evaluation_report.held_out_killed is False
    )

    # Campaign receipt remains complete (held-out failure lives on the
    # remediation evaluation, not the campaign receipt completeness claim).
    rejected = promote_assurance_policy(
        _promotion_candidate(
            proposal=proposal,
            evaluation=failed_eval,
            baseline=baseline,
            promoted=promoted,
        ),
        failed_eval,
        _cid("auth-held-out-fail"),
        campaign_receipt=_campaign_receipt(
            held_out_evaluation_cid=failed_eval.evaluation_report_cid
            or _cid("held-out-fail"),
        ),
        policy_repository=policy_repo,
        operation_id="promote-aae061-held-out-fail",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-held-out-fail"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
        expected_generation=1,
        expected_policy_cid=baseline,
    )
    assert REASON_HELD_OUT_NOT_PASS in rejected.blocking_reasons or (
        REASON_EVALUATION_NOT_PASS in rejected.blocking_reasons
    )
    _assert_no_mutation(
        rejected, policy_repo, expected_cid=baseline, expected_generation=1
    )

    # One-mutant overfit also fails held-out generalization.
    overfit = evaluate_remediation(
        proposal,
        _held_out_campaign(
            campaign_id="camp_aae061_overfit",
            overfit_one_mutant=True,
        ),
    )
    assert overfit.qualified is False
    assert overfit.one_mutant_overfit is True
    overfit_promo = promote_assurance_policy(
        _promotion_candidate(
            proposal=proposal,
            evaluation=overfit,
            baseline=baseline,
            promoted=promoted,
        ),
        overfit,
        _cid("auth-overfit"),
        campaign_receipt=_campaign_receipt(),
        policy_repository=policy_repo,
        operation_id="promote-aae061-overfit",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-overfit"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
    )
    assert overfit_promo.head_mutated is False
    assert overfit_promo.status != PromotionStatus.PROMOTED.value


# ---------------------------------------------------------------------------
# Regression / vacuity / cost / coverage / auth / seal / CAS
# ---------------------------------------------------------------------------


def test_regression_and_vacuity_block_promotion(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    proposal = _propose()
    evaluation = _evaluate(
        proposal,
        _held_out_campaign(campaign_id="camp_aae061_reg_vac"),
    )
    assert evaluation.qualified is True

    # Critical regression injected into the evaluation projection for promotion.
    regression_eval = {
        "evaluation_report_cid": evaluation.evaluation_report_cid,
        "verdict": EvaluationVerdict.REGRESSION.value,
        "held_out_killed": True,
        "held_out_result": HeldOutResult.PASSED.value,
        "regression_detected": True,
        "vacuity_detected": False,
        "cost_delta_basis_points": 50,
        "coverage_declared": True,
        "coverage_partitions": list(evaluation.partitions_covered),
        "qualification_cid": evaluation.qualification_cid,
        "disposition": QualificationDisposition.REJECTED.value,
        "qualified": False,
    }
    reg = promote_assurance_policy(
        _promotion_candidate(
            proposal=proposal,
            evaluation=evaluation,
            baseline=baseline,
            promoted=promoted,
        ),
        regression_eval,
        _cid("auth-regression"),
        campaign_receipt=_campaign_receipt(),
        policy_repository=policy_repo,
        operation_id="promote-aae061-regression",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-regression"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
    )
    assert REASON_REGRESSION_DETECTED in reg.blocking_reasons
    _assert_no_mutation(reg, policy_repo, expected_cid=baseline, expected_generation=1)

    vacuity_eval = {
        "evaluation_report_cid": evaluation.evaluation_report_cid,
        "verdict": EvaluationVerdict.QUALIFIED.value,
        "held_out_killed": True,
        "held_out_result": HeldOutResult.PASSED.value,
        "regression_detected": False,
        "vacuity_detected": True,
        "cost_delta_basis_points": 50,
        "coverage_declared": True,
        "coverage_partitions": list(evaluation.partitions_covered),
        "qualification_cid": evaluation.qualification_cid,
        "disposition": QualificationDisposition.QUALIFIED.value,
        "qualified": True,
        "metadata": {"vacuity_detected": True, "new_vacuity_detected": True},
    }
    vac = promote_assurance_policy(
        _promotion_candidate(
            proposal=proposal,
            evaluation=evaluation,
            baseline=baseline,
            promoted=promoted,
        ),
        vacuity_eval,
        _cid("auth-vacuity"),
        campaign_receipt=_campaign_receipt(),
        policy_repository=policy_repo,
        operation_id="promote-aae061-vacuity",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-vacuity"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
    )
    assert REASON_VACUITY_DETECTED in vac.blocking_reasons
    _assert_no_mutation(vac, policy_repo, expected_cid=baseline, expected_generation=1)


def test_cost_coverage_authorization_cas_and_seal_gates(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    proposal = _propose()
    evaluation = _evaluate(
        proposal,
        _held_out_campaign(campaign_id="camp_aae061_gates"),
    )
    assert evaluation.qualified is True
    base_candidate = _promotion_candidate(
        proposal=proposal,
        evaluation=evaluation,
        baseline=baseline,
        promoted=promoted,
    )

    # Cost undeclared.
    no_cost = dict(base_candidate)
    del no_cost["cost_delta_basis_points"]
    cost_eval = {
        "evaluation_report_cid": evaluation.evaluation_report_cid,
        "verdict": EvaluationVerdict.QUALIFIED.value,
        "held_out_killed": True,
        "held_out_result": HeldOutResult.PASSED.value,
        "regression_detected": False,
        "vacuity_detected": False,
        "coverage_declared": True,
        "coverage_partitions": list(evaluation.partitions_covered),
        "qualified": True,
        "disposition": QualificationDisposition.QUALIFIED.value,
    }
    cost_result = promote_assurance_policy(
        no_cost,
        cost_eval,
        _cid("auth-cost"),
        campaign_receipt=_campaign_receipt(),
        policy_repository=policy_repo,
        operation_id="promote-aae061-cost",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-cost"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
    )
    assert REASON_COST_NOT_DECLARED in cost_result.blocking_reasons
    _assert_no_mutation(
        cost_result, policy_repo, expected_cid=baseline, expected_generation=1
    )

    # Coverage undeclared.
    no_cov = dict(base_candidate)
    no_cov["coverage_declared"] = False
    no_cov["coverage_partitions"] = ()
    cov_eval = {
        "evaluation_report_cid": evaluation.evaluation_report_cid,
        "verdict": EvaluationVerdict.QUALIFIED.value,
        "held_out_killed": True,
        "held_out_result": HeldOutResult.PASSED.value,
        "regression_detected": False,
        "vacuity_detected": False,
        "cost_delta_basis_points": 50,
        "coverage_declared": False,
        "coverage_partitions": (),
        "qualified": True,
        "disposition": QualificationDisposition.QUALIFIED.value,
    }
    cov_result = promote_assurance_policy(
        no_cov,
        cov_eval,
        _cid("auth-coverage"),
        campaign_receipt=_campaign_receipt(),
        policy_repository=policy_repo,
        operation_id="promote-aae061-coverage",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-coverage"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
    )
    assert REASON_COVERAGE_NOT_DECLARED in cov_result.blocking_reasons
    _assert_no_mutation(
        cov_result, policy_repo, expected_cid=baseline, expected_generation=1
    )

    # Absent authorization.
    no_auth = promote_assurance_policy(
        base_candidate,
        evaluation,
        None,
        campaign_receipt=_campaign_receipt(),
        policy_repository=policy_repo,
        operation_id="promote-aae061-no-auth",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-no-auth"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
    )
    assert REASON_ABSENT_AUTHORIZATION in no_auth.blocking_reasons
    _assert_no_mutation(
        no_auth, policy_repo, expected_cid=baseline, expected_generation=1
    )

    # Self-promotion forbidden.
    self_promo = promote_assurance_policy(
        base_candidate,
        evaluation,
        base_candidate["candidate_cid"],
        campaign_receipt=_campaign_receipt(),
        policy_repository=policy_repo,
        operation_id="promote-aae061-self",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-self"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
    )
    assert REASON_SELF_PROMOTION in self_promo.blocking_reasons
    _assert_no_mutation(
        self_promo, policy_repo, expected_cid=baseline, expected_generation=1
    )

    # Seal not released.
    no_seal = promote_assurance_policy(
        base_candidate,
        evaluation,
        _cid("auth-seal"),
        campaign_receipt=_campaign_receipt(),
        policy_repository=policy_repo,
        operation_id="promote-aae061-no-seal",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-bound-only"),
        seal_status=SealAvailabilityStatus.BOUND,
        workspace=WORKSPACE,
    )
    assert REASON_SEAL_UNAVAILABLE in no_seal.blocking_reasons or (
        REASON_ABSENT_SEAL in no_seal.blocking_reasons
    )
    _assert_no_mutation(
        no_seal, policy_repo, expected_cid=baseline, expected_generation=1
    )

    # Expected-old CAS conflict (stale writer vs concurrent advance).
    advanced = policy_repo.compare_and_swap_policy(
        WORKSPACE,
        expected_generation=1,
        expected_policy_cid=baseline,
        new_policy_cid=seeded_policies["other"],
        operation_id="concurrent-writer-aae061",
    )
    assert advanced.status.value == "updated"
    conflict = promote_assurance_policy(
        base_candidate,
        evaluation,
        _cid("auth-conflict"),
        campaign_receipt=_campaign_receipt(),
        policy_repository=policy_repo,
        operation_id="promote-aae061-conflict",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-conflict"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
        expected_generation=1,
        expected_policy_cid=baseline,
    )
    assert conflict.head_mutated is False
    assert conflict.status in {
        PromotionStatus.REJECTED.value,
        PromotionStatus.CONFLICT.value,
    }
    assert policy_repo.current_policy(WORKSPACE).policy_cid == seeded_policies["other"]


# ---------------------------------------------------------------------------
# Success targets: unmet zero/90/50 remain explicit results
# ---------------------------------------------------------------------------


def test_unmet_zero_ninety_fifty_targets_remain_explicit_results() -> None:
    metrics = _unmet_target_campaign_metrics()
    cov = metrics.mutation_coverage
    eco = metrics.economics

    # Sanity: fixture is below the 90% and 50% goals.
    assert cov.risk_weighted_score_bp is not None
    assert cov.risk_weighted_score_bp < HIGH_RISK_DETECTION_TARGET_BP
    assert eco.savings_rate_bp is not None
    assert eco.savings_rate_bp < COMPUTE_SAVINGS_TARGET_BP
    # Risk-weighted: killed 5k of (5+9+9+9+10)k = 5/42 → ~1190 bp.
    expected_rw = (5_000 * BASIS_POINTS) // (5_000 + 9_000 + 9_000 + 9_000 + 10_000)
    assert cov.risk_weighted_score_bp == expected_rw
    # Savings: 200/1000 = 2000 bp.
    assert eco.savings_rate_bp == (200 * BASIS_POINTS) // 1_000

    # No promoted candidates in this unmet campaign (held-out rejected only).
    promotions: list[dict[str, Any]] = [
        {
            "status": PromotionStatus.REJECTED.value,
            "held_out_result": HeldOutResult.FAILED.value,
            "evaluation_report_cid": _cid("eval-rejected"),
            "held_out_evidence": False,
        }
    ]

    report = evaluate_initial_success_targets(
        metrics=metrics,
        promotions=promotions,
        critical_security_survivors_after_remediation=1,  # zero target unmet
        accepted_stale_proof_or_seal_mutants=1,  # zero target unmet
        high_risk_survivors_without_gap=0,
        vacuous_proof_claims_meaningful=0,
        campaign_ids=["camp_aae061_unmet_targets"],
        worktree_escape_count=0,
        unauthorized_production_policy_changes=0,
    )

    assert report["interface"] == INTERFACE
    assert report["evidence"] == EVIDENCE
    assert report["targets_are_goals_not_results"] is True
    assert report["fabricated_pass"] is False
    assert report["production_policy_changed"] is False
    assert report["overall_pass"] is False

    results = report["target_results"]
    zero_sec = results["zero_controlled_critical_security_survivors_after_remediation"]
    zero_stale = results["zero_accepted_stale_proof_or_seal_integrity_mutants"]
    ninety = results["high_risk_semantic_detection_min_bp"]
    fifty = results["compute_savings_min_bp"]

    assert zero_sec["met"] is False
    assert zero_sec["observed_count"] == 1
    assert zero_stale["met"] is False
    assert zero_stale["observed_count"] == 1
    assert ninety["status"] == "unmet"
    assert ninety["met"] is False
    assert ninety["observed_bp"] == cov.risk_weighted_score_bp
    assert ninety["goal_bp"] == HIGH_RISK_DETECTION_TARGET_BP
    assert fifty["status"] == "unmet"
    assert fifty["met"] is False
    assert fifty["observed_bp"] == eco.savings_rate_bp
    assert fifty["goal_bp"] == COMPUTE_SAVINGS_TARGET_BP

    # Explicit lists — not rebranded as passes.
    assert "zero_controlled_critical_security_survivors_after_remediation" in report[
        "unmet_targets"
    ]
    assert "zero_accepted_stale_proof_or_seal_integrity_mutants" in report[
        "unmet_targets"
    ]
    assert "high_risk_semantic_detection_min_bp" in report["unmet_targets"]
    assert "compute_savings_min_bp" in report["unmet_targets"]

    # No fabricated overall success claim.
    assert report["fabricated_pass"] is False
    assert "passed" not in report["unmet_targets"]
    for tid in report["unmet_targets"]:
        row = results[tid]
        assert row.get("met") is not True
        assert row.get("status", "unmet") != "met"


def test_success_targets_on_report_are_goals_not_results() -> None:
    metrics = _unmet_target_campaign_metrics()
    report = build_assurance_report(
        {
            "plan_id": "plan_aae061_targets",
            "plan_cid": _cid("plan-targets"),
            "result_cid": _cid("result-targets"),
            "repository_state_cid": _cid("repo-state-targets"),
            "verification_policy_cid": _cid("policy-targets"),
            "terminal_status": "complete",
            "candidate_reports": [],
            "require_sandbox": True,
            "network_disabled": True,
            "production_policy_changed": False,
        },
        include_metrics=False,
        notes="aae-061 success targets are goals",
    )
    assert report.success_targets["targets_are_goals_not_results"] is True
    assert (
        report.success_targets["high_risk_semantic_detection_min_bp"]
        == HIGH_RISK_DETECTION_TARGET_BP
    )
    assert report.success_targets["compute_savings_min_bp"] == COMPUTE_SAVINGS_TARGET_BP
    assert report.success_targets[
        "zero_controlled_critical_security_survivors_after_remediation"
    ] is True
    # Summary must not rebrand goals as measured results.
    assert "targets_are_goals_not_results" not in (report.summary or "")
    # Metrics still show the honest under-target observations.
    assert metrics.mutation_coverage.risk_weighted_score_bp < HIGH_RISK_DETECTION_TARGET_BP
    assert metrics.economics.savings_rate_bp is not None
    assert metrics.economics.savings_rate_bp < COMPUTE_SAVINGS_TARGET_BP


def test_promoted_path_records_held_out_in_success_target_evaluation(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    """When promotion succeeds, held-out target is met; numeric goals stay honest."""

    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    proposal = _propose()
    evaluation = _evaluate(
        proposal,
        _held_out_campaign(campaign_id="camp_aae061_promo_targets"),
    )
    result = promote_assurance_policy(
        _promotion_candidate(
            proposal=proposal,
            evaluation=evaluation,
            baseline=baseline,
            promoted=promoted,
        ),
        evaluation,
        _cid("auth-promo-targets"),
        campaign_receipt=_campaign_receipt(
            held_out_evaluation_cid=evaluation.evaluation_report_cid
            or _cid("held-out-promo-targets")
        ),
        policy_repository=policy_repo,
        operation_id="promote-aae061-promo-targets",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-promo-targets"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
        expected_generation=1,
        expected_policy_cid=baseline,
    )
    assert result.status == PromotionStatus.PROMOTED.value
    assert result.held_out_result == HeldOutResult.PASSED.value

    # Combine a successful promotion with still-unmet economics/detection goals.
    metrics = _unmet_target_campaign_metrics()
    target_report = evaluate_initial_success_targets(
        metrics=metrics,
        promotions=[
            {
                "status": result.status,
                "held_out_result": result.held_out_result,
                "evaluation_report_cid": result.evaluation_report_cid,
                "held_out_evidence": True,
            }
        ],
        critical_security_survivors_after_remediation=0,
        accepted_stale_proof_or_seal_mutants=0,
        high_risk_survivors_without_gap=0,
        vacuous_proof_claims_meaningful=0,
        campaign_ids=["camp_aae061_promo_targets", "camp_aae061_unmet_targets"],
        worktree_escape_count=0,
        unauthorized_production_policy_changes=0,
    )
    held = target_report["target_results"]["held_out_evaluation_for_every_promotion"]
    assert held["met"] is True
    assert target_report["fabricated_pass"] is False
    # 90/50 remain unmet despite a valid promotion — not rewritten as passes.
    assert "high_risk_semantic_detection_min_bp" in target_report["unmet_targets"]
    assert "compute_savings_min_bp" in target_report["unmet_targets"]
    assert target_report["overall_pass"] is False


# ---------------------------------------------------------------------------
# End-to-end qualification envelope
# ---------------------------------------------------------------------------


def test_qualification_envelope_end_to_end(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    """Full propose → evaluate → promote → target evaluation envelope."""

    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]

    proposal = _propose()
    campaign = _held_out_campaign(campaign_id="camp_aae061_envelope")
    evaluation = evaluate_remediation(proposal, campaign)
    assert evaluation.qualified is True

    promo = promote_assurance_policy(
        _promotion_candidate(
            proposal=proposal,
            evaluation=evaluation,
            baseline=baseline,
            promoted=promoted,
        ),
        evaluation,
        _cid("auth-envelope"),
        campaign_receipt=_campaign_receipt(
            held_out_evaluation_cid=evaluation.evaluation_report_cid
            or _cid("held-out-envelope")
        ),
        policy_repository=policy_repo,
        operation_id="promote-aae061-envelope",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-envelope"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
        expected_generation=1,
        expected_policy_cid=baseline,
    )
    assert promo.status == PromotionStatus.PROMOTED.value

    metrics = compute_assurance_metrics(
        campaign_id="camp_aae061_envelope",
        outcomes=[
            _outcome(
                "cand_kill",
                MutationOutcomeStatus.KILLED_BY_TEST.value,
                risk_weight_bp=8_000,
                observed=["det_test"],
                killing_id="det_test",
                killing_kind="unit_test",
            ),
            _outcome(
                "cand_surv",
                MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value,
                risk_weight_bp=2_000,
            ),
        ],
        gaps=[
            {
                "gap_id": "gap_surv",
                "gap_class": "missing_test",
                "risk_class": "authorization",
                "high_risk": True,
            }
        ],
        remediations=[
            {
                "remediation_id": "rem_promoted",
                "disposition": "promoted",
                "held_out_kill_count": 2,
                "evaluated": True,
                "cost_cpu_ms": 50,
            }
        ],
        economics_records=[
            {
                "economics_id": "eco_ok",
                "full_cpu_ms": 1_000,
                "incremental_cpu_ms": 400,
                "full_wall_ms": 1_000,
                "incremental_wall_ms": 400,
                "cache_hits": 3,
                "cache_misses": 1,
            }
        ],
        plan_id="plan_envelope",
        plan_cid=_cid("plan-envelope"),
        result_cid=_cid("result-envelope"),
        repository_state_cid=_cid("repo-state-envelope"),
    )

    targets = evaluate_initial_success_targets(
        metrics=metrics,
        promotions=[
            {
                "status": promo.status,
                "held_out_result": promo.held_out_result,
                "evaluation_report_cid": promo.evaluation_report_cid,
                "held_out_evidence": True,
            }
        ],
        critical_security_survivors_after_remediation=0,
        accepted_stale_proof_or_seal_mutants=0,
        high_risk_survivors_without_gap=0,
        vacuous_proof_claims_meaningful=0,
        campaign_ids=["camp_aae061_envelope"],
        worktree_escape_count=0,
        unauthorized_production_policy_changes=0,
    )

    envelope = {
        "interface": INTERFACE,
        "evidence": EVIDENCE,
        "task_id": TASK_ID,
        "bundle": BUNDLE,
        "proposal_cid": proposal.proposal_cid,
        "evaluation_run_cid": evaluation.run_cid,
        "evaluation_report_cid": evaluation.evaluation_report_cid,
        "qualification_cid": evaluation.qualification_cid,
        "promotion_status": promo.status,
        "promoted_policy_cid": promo.promoted_policy_cid,
        "held_out_result": promo.held_out_result,
        "head_mutated": promo.head_mutated,
        "production_policy_changed": False,
        "success_targets": targets,
        "gates": {
            "held_out_evidence": promo.held_out_result == HeldOutResult.PASSED.value,
            "no_critical_regression": True,
            "no_new_vacuity": True,
            "cost_declared": True,
            "coverage_declared": True,
            "authorization": promo.authorization_cid is not None,
            "expected_old_cas": promo.policy_cas is not None
            and promo.policy_cas.get("status") == "updated",
            "released_incremental_seal": promo.seal_evidence_cid is not None,
        },
    }

    assert envelope["interface"] == INTERFACE
    assert envelope["evidence"] == EVIDENCE
    assert envelope["promotion_status"] == PromotionStatus.PROMOTED.value
    assert envelope["held_out_result"] == HeldOutResult.PASSED.value
    assert envelope["production_policy_changed"] is False
    assert all(envelope["gates"].values())
    assert set(envelope["gates"]) == REQUIRED_PROMOTION_GATES
    assert targets["fabricated_pass"] is False
    assert targets["targets_are_goals_not_results"] is True
    # Risk-weighted 8000/10000 = 8000 bp < 9000 → 90% still explicit unmet.
    assert metrics.mutation_coverage.risk_weighted_score_bp == 8_000
    assert "high_risk_semantic_detection_min_bp" in targets["unmet_targets"]
    # Savings 600/1000 = 6000 bp >= 5000 → 50% met.
    assert metrics.economics.savings_rate_bp == 6_000
    assert "compute_savings_min_bp" in targets["met_targets"]
    assert "held_out_evaluation_for_every_promotion" in targets["met_targets"]


def test_rejected_remediation_never_self_promotes(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    proposal = _propose()
    evaluation = evaluate_remediation(
        proposal,
        _held_out_campaign(
            campaign_id="camp_aae061_reject",
            fail=frozenset({EvaluationPartition.SAFETY.value}),
        ),
    )
    assert evaluation.qualified is False
    assert evaluation.production_policy_changed is False

    # Even with external-looking inputs, rejected evaluation cannot promote.
    result = promote_assurance_policy(
        _promotion_candidate(
            proposal=proposal,
            evaluation=evaluation,
            baseline=baseline,
            promoted=promoted,
        ),
        evaluation,
        _cid("auth-reject"),
        campaign_receipt=_campaign_receipt(),
        policy_repository=policy_repo,
        operation_id="promote-aae061-reject",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-reject"),
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
    )
    assert result.head_mutated is False
    assert result.status != PromotionStatus.PROMOTED.value
    assert policy_repo.current_policy(WORKSPACE).policy_cid == baseline
    assert evaluation.production_policy_changed is False
    assert proposal.production_policy_changed is False
