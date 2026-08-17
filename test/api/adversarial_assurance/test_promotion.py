"""Tests for authorized assurance-policy promotion, CAS, and seal (AAE-047).

Acceptance criteria enforced here:

* Canonical candidate identity is required.
* Held-out pass is mandatory.
* Regression and vacuity block promotion.
* Cost and coverage must be declared.
* Authorization is mandatory; candidates cannot self-promote.
* Campaign and promotion receipts require verified
  signer/key/audience/action bindings before CAS.
* Expected-old policy CAS is required; stale writers do not overwrite.
* Released incremental seal evidence is mandatory.
* Cold import is side-effect free.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    ArtifactProvenance,
    AssuranceArtifactHeader,
    AssuranceTerminalStatus,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    VersionBinding,
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
    EvaluationVerdict,
)

from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
    DurableCoordinationStore,
    cid_for_artifact,
)
from ipfs_kit_py.adversarial_assurance_store.policy import (
    DurableAssurancePolicyRepository,
)

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.promotion import (
    AAE_PROMOTION_EVIDENCE,
    ADAPTER_ID,
    PROMOTE_ASSURANCE_POLICY_INTERFACE,
    AssurancePolicyPromotionResult,
    PromotionStatus,
    REASON_ABSENT_AUTHORIZATION,
    REASON_ABSENT_CAMPAIGN_RECEIPT,
    REASON_ABSENT_SEAL,
    REASON_CAS_CONFLICT,
    REASON_COST_NOT_DECLARED,
    REASON_COVERAGE_NOT_DECLARED,
    REASON_EVALUATION_NOT_PASS,
    REASON_HELD_OUT_NOT_PASS,
    REASON_INVALID_SIGNATURE_BINDINGS,
    REASON_MISSING_REPOSITORY,
    REASON_REGRESSION_DETECTED,
    REASON_SEAL_UNAVAILABLE,
    REASON_SELF_PROMOTION,
    REASON_STALE_CANDIDATE,
    REASON_UNVERIFIED_CAMPAIGN_RECEIPT,
    REASON_UNVERIFIED_PROMOTION_RECEIPT,
    REASON_VACUITY_DETECTED,
    evaluate_promotion_gates,
    promote_assurance_policy,
    promote_assurance_policy_descriptor,
    verify_receipt_signature_bindings,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/promotion.py"
)
WORKSPACE = "aae047-worker"

_SIGNER = "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
_SIGNATURE = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAA"
)


# ---------------------------------------------------------------------------
# Fixtures / recipes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    """Canonical dag-json CID (store CAS requires dag-json profile)."""

    return cid_for_structured({"test_label": label, "schema": "test/aae047@1"})


def _block(store: DurableCoordinationStore, name: str, **extra: Any) -> str:
    payload = {"schema": "example/assurance-policy@1", "name": name}
    payload.update(extra)
    return store.put(payload, expected_cid=cid_for_artifact(payload), replicate=False)[
        "cid"
    ]


def _generator(**overrides: object) -> GeneratorIdentity:
    fields = {
        "generator_id": "campaign_sealer",
        "generator_version": "1.0.0",
        "interface_id": "seal_campaign@1",
    }
    fields.update(overrides)
    return GeneratorIdentity(**fields)  # type: ignore[arg-type]


def _versions(**overrides: object) -> VersionBinding:
    fields = {
        "operator_id": "campaign_operator",
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
        "authority_source": AuthoritySource.RECEIPT,
        "input_cids": (_cid("input-a"),),
        "tool_ids": ("campaign.sealer.v1",),
        "policy_cid": _cid("policy-baseline"),
        "notes": None,
    }
    fields.update(overrides)
    return ArtifactProvenance(**fields)  # type: ignore[arg-type]


def _header(artifact_kind: str, **overrides: object) -> AssuranceArtifactHeader:
    fields = {
        "artifact_kind": artifact_kind,
        "repository_id": "repository:sha256:test-repo-identity-aae047",
        "repository_state_cid": _cid("repo-state"),
        "target_symbol_ids": ("mod.fn",),
        "target_artifact_cids": (_cid("artifact-a"),),
        "capsule_cids": (_cid("capsule-a"),),
        "proof_unit_cids": (_cid("proof-unit-a"),),
        "environment_cid": _cid("environment"),
        "dependency_lock_cid": _cid("dependency-lock"),
        "versions": _versions(),
        "provenance": _provenance(),
        "terminal_status": AssuranceTerminalStatus.COMPLETE,
        "receipt_cids": (),
        "proof_cids": (),
        "metadata": {},
    }
    fields.update(overrides)
    return AssuranceArtifactHeader(**fields)  # type: ignore[arg-type]


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


def _campaign(**overrides: object) -> AssuranceCampaignReceipt:
    fields = {
        "header": _header("assurance_campaign_receipt"),
        "receipt_id": "campaign_receipt_aae047",
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
        "metadata": {},
    }
    fields.update(overrides)
    return AssuranceCampaignReceipt(**fields)  # type: ignore[arg-type]


def _candidate(
    *,
    base_policy_cid: str,
    proposed_policy_cid: str,
    **overrides: Any,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "candidate_cid": _cid("canonical-candidate"),
        "plan_cid": _cid("remediation-plan"),
        "proposed_policy_cid": proposed_policy_cid,
        "base_policy_cid": base_policy_cid,
        "base_policy_version": "1.0.0",
        "proposed_policy_version": "1.0.1",
        "cost_delta_basis_points": 50,
        "coverage_declared": True,
        "coverage_partitions": (
            "unmutated",
            "held_out",
            "regression",
            "performance_cost",
        ),
    }
    fields.update(overrides)
    return fields


def _evaluation(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "evaluation_report_cid": _cid("evaluation-report"),
        "verdict": EvaluationVerdict.QUALIFIED.value,
        "held_out_killed": True,
        "held_out_result": HeldOutResult.PASSED.value,
        "regression_detected": False,
        "vacuity_detected": False,
        "cost_delta_basis_points": 50,
        "coverage_declared": True,
        "coverage_partitions": (
            "unmutated",
            "held_out",
            "regression",
            "performance_cost",
        ),
        "qualified": True,
        "disposition": "qualified",
    }
    fields.update(overrides)
    return fields


def _promo_signature(**overrides: object) -> ReceiptSignatureBinding:
    return _signature(action=ReceiptAction.PROMOTE_POLICY, **overrides)


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    return tmp_path / "aae047-policy-cas"


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
    """Publish a baseline policy head so promotions have a non-zero expected-old."""

    baseline = _block(coordination, "policy-baseline-v1")
    promoted = _block(coordination, "policy-promoted-v2")
    other = _block(coordination, "policy-other-v3")
    cas = policy_repo.compare_and_swap_policy(
        WORKSPACE,
        expected_generation=0,
        expected_policy_cid=None,
        new_policy_cid=baseline,
        operation_id="seed-baseline",
    )
    assert cas.status.value == "updated"
    return {
        "baseline": baseline,
        "promoted": promoted,
        "other": other,
    }


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exists_and_exports_interfaces() -> None:
    assert MODULE_PATH.is_file()
    assert PROMOTE_ASSURANCE_POLICY_INTERFACE == "promote_assurance_policy@1"
    assert AAE_PROMOTION_EVIDENCE == "aae/promotion@1"
    assert ADAPTER_ID.startswith("aae-047")
    descriptor = promote_assurance_policy_descriptor()
    assert descriptor["interface_id"] == PROMOTE_ASSURANCE_POLICY_INTERFACE
    assert descriptor["self_promotion_forbidden"] is True
    assert "held_out_pass" in descriptor["mandatory_gates"]
    assert "released_incremental_seal" in descriptor["mandatory_gates"]
    assert "no_self_promotion" in descriptor["mandatory_gates"]


def test_cold_import_is_side_effect_free() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Module body must not call network/store openers at import time.
    forbidden_calls = {
        "open",
        "urlopen",
        "DurableCoordinationStore",
        "promote_assurance_policy",
    }
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            assert name not in forbidden_calls


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_promotes_with_expected_old_cas(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    campaign = _campaign()
    auth = _cid("external-operator-authorization")
    seal = _cid("released-incremental-seal")
    candidate = _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted)
    evaluation = _evaluation()

    result = promote_assurance_policy(
        candidate,
        evaluation,
        auth,
        campaign_receipt=campaign,
        policy_repository=policy_repo,
        operation_id="promote-happy-1",
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
    assert result.receipt is not None
    assert result.receipt.signature.signer_identity == _SIGNER
    assert result.receipt.signature.key_identity == _SIGNER
    assert result.receipt.signature.audience == "adversarial_assurance.store"
    assert result.receipt.signature.action == ReceiptAction.PROMOTE_POLICY.value
    assert (
        result.receipt.signature.signature_verification_status
        == SignatureVerificationStatus.VERIFIED.value
    )
    assert result.receipt.seal_status == SealAvailabilityStatus.RELEASED.value
    assert result.receipt.expected_old_policy_cid == baseline
    assert result.receipt.cas_expected_version == "1.0.0"
    assert result.policy_cas is not None
    assert result.policy_cas["status"] == "updated"
    assert policy_repo.current_policy(WORKSPACE).policy_cid == promoted
    assert policy_repo.current_policy(WORKSPACE).generation == 2
    verify_promotion_receipt_identity(result.receipt)
    verify_campaign_receipt_identity(campaign)

    # Idempotent replay of the same operation_id does not double-advance.
    replay = promote_assurance_policy(
        candidate,
        evaluation,
        auth,
        campaign_receipt=campaign,
        policy_repository=policy_repo,
        operation_id="promote-happy-1",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=seal,
        seal_status=SealAvailabilityStatus.RELEASED,
        workspace=WORKSPACE,
        expected_generation=1,
        expected_policy_cid=baseline,
    )
    assert replay.head_mutated is False
    assert replay.status in {
        PromotionStatus.UNCHANGED.value,
        PromotionStatus.CONFLICT.value,
    }
    assert policy_repo.current_policy(WORKSPACE).generation == 2


# ---------------------------------------------------------------------------
# Mandatory gates — each blocks without mutating the head
# ---------------------------------------------------------------------------


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


def test_stale_candidate_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    stale_base = seeded_policies["other"]
    result = promote_assurance_policy(
        _candidate(base_policy_cid=stale_base, proposed_policy_cid=promoted),
        _evaluation(),
        _cid("auth-stale"),
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-stale",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-stale"),
        workspace=WORKSPACE,
    )
    assert REASON_STALE_CANDIDATE in result.blocking_reasons
    _assert_no_mutation(
        result, policy_repo, expected_cid=baseline, expected_generation=1
    )


def test_held_out_failure_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    result = promote_assurance_policy(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted),
        _evaluation(
            held_out_killed=False,
            held_out_result=HeldOutResult.FAILED.value,
            qualified=False,
            verdict=EvaluationVerdict.REJECTED.value,
            disposition="rejected",
        ),
        _cid("auth-held-out"),
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-held-out-fail",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-held-out"),
        workspace=WORKSPACE,
    )
    assert REASON_HELD_OUT_NOT_PASS in result.blocking_reasons
    assert REASON_EVALUATION_NOT_PASS in result.blocking_reasons
    _assert_no_mutation(
        result, policy_repo, expected_cid=baseline, expected_generation=1
    )


def test_regression_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    result = promote_assurance_policy(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted),
        _evaluation(regression_detected=True, qualified=False, verdict="regression"),
        _cid("auth-regression"),
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-regression",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-regression"),
        workspace=WORKSPACE,
    )
    assert REASON_REGRESSION_DETECTED in result.blocking_reasons
    _assert_no_mutation(
        result, policy_repo, expected_cid=baseline, expected_generation=1
    )


def test_vacuity_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    result = promote_assurance_policy(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted),
        _evaluation(vacuity_detected=True),
        _cid("auth-vacuity"),
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-vacuity",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-vacuity"),
        workspace=WORKSPACE,
    )
    assert REASON_VACUITY_DETECTED in result.blocking_reasons
    _assert_no_mutation(
        result, policy_repo, expected_cid=baseline, expected_generation=1
    )


def test_undeclared_cost_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    candidate = _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted)
    del candidate["cost_delta_basis_points"]
    evaluation = _evaluation()
    del evaluation["cost_delta_basis_points"]
    result = promote_assurance_policy(
        candidate,
        evaluation,
        _cid("auth-cost"),
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-cost",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-cost"),
        workspace=WORKSPACE,
    )
    assert REASON_COST_NOT_DECLARED in result.blocking_reasons
    _assert_no_mutation(
        result, policy_repo, expected_cid=baseline, expected_generation=1
    )


def test_undeclared_coverage_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    candidate = _candidate(
        base_policy_cid=baseline,
        proposed_policy_cid=promoted,
        coverage_declared=False,
        coverage_partitions=(),
    )
    evaluation = _evaluation(
        coverage_declared=False,
        coverage_partitions=(),
    )
    result = promote_assurance_policy(
        candidate,
        evaluation,
        _cid("auth-coverage"),
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-coverage",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-coverage"),
        workspace=WORKSPACE,
    )
    assert REASON_COVERAGE_NOT_DECLARED in result.blocking_reasons
    _assert_no_mutation(
        result, policy_repo, expected_cid=baseline, expected_generation=1
    )


def test_absent_authorization_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    result = promote_assurance_policy(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted),
        _evaluation(),
        None,
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-no-auth",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-no-auth"),
        workspace=WORKSPACE,
    )
    assert REASON_ABSENT_AUTHORIZATION in result.blocking_reasons
    _assert_no_mutation(
        result, policy_repo, expected_cid=baseline, expected_generation=1
    )


def test_self_promotion_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    candidate = _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted)
    evaluation = _evaluation()

    # Candidate authorizes itself.
    r1 = promote_assurance_policy(
        candidate,
        evaluation,
        candidate["candidate_cid"],
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-self-cand",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-self-1"),
        workspace=WORKSPACE,
    )
    assert REASON_SELF_PROMOTION in r1.blocking_reasons
    _assert_no_mutation(r1, policy_repo, expected_cid=baseline, expected_generation=1)

    # Evaluation authorizes itself.
    r2 = promote_assurance_policy(
        candidate,
        evaluation,
        evaluation["evaluation_report_cid"],
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-self-eval",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-self-2"),
        workspace=WORKSPACE,
    )
    assert REASON_SELF_PROMOTION in r2.blocking_reasons
    _assert_no_mutation(r2, policy_repo, expected_cid=baseline, expected_generation=1)

    # Promoted policy authorizes itself.
    r3 = promote_assurance_policy(
        candidate,
        evaluation,
        promoted,
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-self-policy",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-self-3"),
        workspace=WORKSPACE,
    )
    assert REASON_SELF_PROMOTION in r3.blocking_reasons
    _assert_no_mutation(r3, policy_repo, expected_cid=baseline, expected_generation=1)


def test_unverified_campaign_receipt_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    campaign = _campaign(
        header=_header(
            "assurance_campaign_receipt",
            terminal_status=AssuranceTerminalStatus.INCONCLUSIVE,
        ),
        signature=_signature(
            signature="",
            signature_verification_status=SignatureVerificationStatus.UNAVAILABLE,
            action=ReceiptAction.COMPLETE_CAMPAIGN,
        ),
    )
    result = promote_assurance_policy(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted),
        _evaluation(),
        _cid("auth-campaign-sig"),
        campaign_receipt=campaign,
        policy_repository=policy_repo,
        operation_id="promote-campaign-sig",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-campaign-sig"),
        workspace=WORKSPACE,
    )
    assert (
        REASON_UNVERIFIED_CAMPAIGN_RECEIPT in result.blocking_reasons
        or REASON_INVALID_SIGNATURE_BINDINGS in result.blocking_reasons
    )
    _assert_no_mutation(
        result, policy_repo, expected_cid=baseline, expected_generation=1
    )


def test_unverified_promotion_signature_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    bad_sig = _signature(
        action=ReceiptAction.PROMOTE_POLICY,
        signature="",
        signature_verification_status=SignatureVerificationStatus.UNAVAILABLE,
    )
    result = promote_assurance_policy(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted),
        _evaluation(),
        _cid("auth-promo-sig"),
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-promo-sig",
        promotion_signature=bad_sig,
        seal_evidence_cid=_cid("seal-promo-sig"),
        workspace=WORKSPACE,
    )
    assert (
        REASON_UNVERIFIED_PROMOTION_RECEIPT in result.blocking_reasons
        or REASON_INVALID_SIGNATURE_BINDINGS in result.blocking_reasons
    )
    _assert_no_mutation(
        result, policy_repo, expected_cid=baseline, expected_generation=1
    )


def test_missing_released_seal_cannot_mutate_head(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    result = promote_assurance_policy(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted),
        _evaluation(),
        _cid("auth-seal"),
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-no-seal",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-bound-only"),
        seal_status=SealAvailabilityStatus.BOUND,
        workspace=WORKSPACE,
    )
    assert REASON_SEAL_UNAVAILABLE in result.blocking_reasons
    _assert_no_mutation(
        result, policy_repo, expected_cid=baseline, expected_generation=1
    )


def test_cas_conflict_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    other = seeded_policies["other"]
    # Concurrent writer advances the head first.
    advanced = policy_repo.compare_and_swap_policy(
        WORKSPACE,
        expected_generation=1,
        expected_policy_cid=baseline,
        new_policy_cid=other,
        operation_id="concurrent-writer",
    )
    assert advanced.status.value == "updated"

    result = promote_assurance_policy(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted),
        _evaluation(),
        _cid("auth-conflict"),
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-conflict",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-conflict"),
        workspace=WORKSPACE,
        expected_generation=1,
        expected_policy_cid=baseline,
    )
    # Either stale-candidate gate or CAS conflict; head stays at concurrent write.
    assert result.head_mutated is False
    assert result.status in {
        PromotionStatus.REJECTED.value,
        PromotionStatus.CONFLICT.value,
    }
    assert (
        REASON_STALE_CANDIDATE in result.blocking_reasons
        or REASON_CAS_CONFLICT in result.blocking_reasons
        or "policy_head_expectation_mismatch" in result.blocking_reasons
    )
    assert policy_repo.current_policy(WORKSPACE).policy_cid == other


def test_missing_repository_rejects_without_mutation() -> None:
    result = promote_assurance_policy(
        _candidate(base_policy_cid=_cid("b"), proposed_policy_cid=_cid("p")),
        _evaluation(),
        _cid("auth"),
        campaign_receipt=_campaign(),
        policy_repository=None,
        operation_id="promote-no-repo",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal"),
        workspace=WORKSPACE,
    )
    assert result.status == PromotionStatus.REJECTED.value
    assert REASON_MISSING_REPOSITORY in result.blocking_reasons
    assert result.head_mutated is False


# ---------------------------------------------------------------------------
# Signature binding helpers
# ---------------------------------------------------------------------------


def test_verify_receipt_signature_bindings_requires_signer_key_audience_action() -> None:
    binding = _promo_signature()
    sealed = verify_receipt_signature_bindings(
        binding,
        expected_action=ReceiptAction.PROMOTE_POLICY,
        require_verified=True,
    )
    assert sealed.signer_identity == _SIGNER
    assert sealed.key_identity == _SIGNER
    assert sealed.audience == "adversarial_assurance.store"
    assert sealed.action == ReceiptAction.PROMOTE_POLICY.value

    with pytest.raises(Exception, match="verified|signature"):
        verify_receipt_signature_bindings(
            _signature(
                action=ReceiptAction.PROMOTE_POLICY,
                signature="",
                signature_verification_status=SignatureVerificationStatus.UNAVAILABLE,
            ),
            expected_action=ReceiptAction.PROMOTE_POLICY,
            require_verified=True,
        )


def test_evaluate_promotion_gates_covers_acceptance_reasons(
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.promotion import (
        _normalize_candidate,
        _normalize_evaluation,
    )

    cand = _normalize_candidate(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted)
    )
    evaluation = _normalize_evaluation(_evaluation())
    campaign = _campaign()
    auth = _cid("auth-gates")
    seal = _cid("seal-gates")

    ok = evaluate_promotion_gates(
        cand,
        evaluation,
        auth,
        campaign_receipt=campaign,
        seal_status=SealAvailabilityStatus.RELEASED,
        seal_evidence_cid=seal,
        current_policy_cid=baseline,
        current_generation=1,
        expected_generation=1,
        expected_policy_cid=baseline,
        promotion_signature=_promo_signature(),
    )
    assert ok == ()

    blocked = evaluate_promotion_gates(
        cand,
        evaluation,
        cand.candidate_cid,
        campaign_receipt=None,
        seal_status=SealAvailabilityStatus.UNAVAILABLE,
        seal_evidence_cid=None,
        current_policy_cid=baseline,
        current_generation=1,
        promotion_signature=None,
    )
    assert REASON_SELF_PROMOTION in blocked
    assert REASON_ABSENT_CAMPAIGN_RECEIPT in blocked
    assert REASON_SEAL_UNAVAILABLE in blocked or REASON_ABSENT_SEAL in blocked
    assert REASON_UNVERIFIED_PROMOTION_RECEIPT in blocked


def test_promotion_result_identity_is_stable(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    result = promote_assurance_policy(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted),
        _evaluation(),
        _cid("auth-identity"),
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-identity",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-identity"),
        workspace=WORKSPACE,
    )
    assert result.status == PromotionStatus.PROMOTED.value
    payload = result.to_dict()
    assert payload["result_cid"] == result.result_cid
    again = AssurancePolicyPromotionResult(
        status=result.status,
        head_mutated=result.head_mutated,
        blocking_reasons=result.blocking_reasons,
        workspace=result.workspace,
        operation_id=result.operation_id,
        candidate_cid=result.candidate_cid,
        evaluation_report_cid=result.evaluation_report_cid,
        authorization_cid=result.authorization_cid,
        campaign_receipt_cid=result.campaign_receipt_cid,
        seal_evidence_cid=result.seal_evidence_cid,
        expected_generation=result.expected_generation,
        expected_policy_cid=result.expected_policy_cid,
        promoted_policy_cid=result.promoted_policy_cid,
        held_out_result=result.held_out_result,
        receipt=result.receipt,
        policy_cas=dict(result.policy_cas) if result.policy_cas else None,
        promotion_cas=dict(result.promotion_cas) if result.promotion_cas else None,
        diagnostic=result.diagnostic,
        metadata=dict(result.metadata),
    )
    assert again.result_cid == result.result_cid


def test_rejected_result_never_claims_mutation() -> None:
    result = AssurancePolicyPromotionResult(
        status=PromotionStatus.REJECTED.value,
        head_mutated=False,
        blocking_reasons=(REASON_ABSENT_AUTHORIZATION,),
        workspace=WORKSPACE,
        operation_id="reject-only",
        candidate_cid=_cid("c"),
        evaluation_report_cid=_cid("e"),
        authorization_cid=None,
        campaign_receipt_cid=None,
        seal_evidence_cid=None,
        expected_generation=1,
        expected_policy_cid=_cid("old"),
        promoted_policy_cid=_cid("new"),
        held_out_result=HeldOutResult.FAILED.value,
        receipt=None,
        policy_cas=None,
        promotion_cas=None,
    )
    assert result.head_mutated is False
    with pytest.raises(Exception):
        AssurancePolicyPromotionResult(
            status=PromotionStatus.PROMOTED.value,
            head_mutated=False,
            blocking_reasons=(),
            workspace=WORKSPACE,
            operation_id="bad-promoted",
            candidate_cid=None,
            evaluation_report_cid=None,
            authorization_cid=None,
            campaign_receipt_cid=None,
            seal_evidence_cid=None,
            expected_generation=None,
            expected_policy_cid=None,
            promoted_policy_cid=None,
            held_out_result=None,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
        )


def test_authorization_mapping_form(
    policy_repo: DurableAssurancePolicyRepository,
    seeded_policies: dict[str, str],
) -> None:
    baseline = seeded_policies["baseline"]
    promoted = seeded_policies["promoted"]
    auth_cid = _cid("auth-mapping")
    result = promote_assurance_policy(
        _candidate(base_policy_cid=baseline, proposed_policy_cid=promoted),
        _evaluation(),
        {"authorization_cid": auth_cid},
        campaign_receipt=_campaign(),
        policy_repository=policy_repo,
        operation_id="promote-auth-map",
        promotion_signature=_promo_signature(),
        seal_evidence_cid=_cid("seal-auth-map"),
        workspace=WORKSPACE,
    )
    assert result.status == PromotionStatus.PROMOTED.value
    assert result.authorization_cid == auth_cid
    assert result.receipt is not None
    assert result.receipt.signature.action == "promote_policy"
