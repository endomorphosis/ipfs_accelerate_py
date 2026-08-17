"""Tests for SCG-035 release qualification and incremental seal binding.

Acceptance criteria enforced here:

* Missing sealer is typed unavailable and never replaced by
  VerificationCommitment.
* Promotion remains blocked unless the independently authorized
  release-qualification path passes (or released incremental-seal evidence
  is present).
* Signed/sealed artifacts bind evaluated policy and encode only the closed
  bounded claim set (no semantic-sufficiency / ZK overclaim).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_governor.policy_contracts import (
    EvaluationVerdict,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.adapters import (
    EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
    EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
    GovernorCapabilityUnavailable,
    SealStatus,
    probe_incremental_sealer_capability,
    sealer_capability_from_evidence,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.sealing import (
    ARTIFACT_ROLE_CALIBRATION_PROFILE,
    ARTIFACT_ROLE_CONTEXT_PACK,
    ARTIFACT_ROLE_DIFFERENTIAL_REPORT,
    ARTIFACT_ROLE_PROMOTION_DECISION,
    BOUNDED_CLAIM_SET_INTERFACE,
    BoundedClaimKind,
    DEFAULT_BOUNDED_CLAIMS,
    FORBIDDEN_CLAIM_KINDS,
    GOVERNOR_SEAL_INTERFACE,
    GOVERNOR_SEAL_SCHEMA,
    GovernorSeal,
    QUALIFY_POLICY_CANDIDATE_INTERFACE,
    QualificationPath,
    REASON_BUNDLE_IS_COMMITMENT,
    REASON_EVALUATION_NOT_PASS,
    REASON_IVP_COMMITMENT_NOT_SEALER,
    REASON_MISSING_QUALIFICATION_AUTHORIZATION,
    REASON_MISSING_RELEASE_QUALIFICATION,
    REASON_MISSING_VERIFICATION_BUNDLE,
    REASON_OVERCLAIM,
    REASON_PROMOTION_BLOCKED,
    REASON_SELF_AUTHORIZATION,
    REASON_UNAUTHORIZED_CLAIM,
    RELEASE_QUALIFICATION_INTERFACE,
    RELEASE_QUALIFICATION_SCHEMA,
    ReleaseQualification,
    SCG_RELEASE_QUALIFICATION_EVIDENCE,
    SCG_SEAL_BINDING_EVIDENCE,
    SEAL_GOVERNOR_RUN_INTERFACE,
    SEMANTIC_GOVERNOR_SEAL_ADAPTER_INTERFACE,
    SealArtifactBinding,
    SealingError,
    SemanticGovernorSealAdapter,
    VERIFY_GOVERNOR_SEAL_INTERFACE,
    load_seal_adapter,
    qualify_policy_candidate,
    seal_governor_run,
    verify_governor_seal,
)


# ---------------------------------------------------------------------------
# Fixtures / recipes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _pass_evaluation(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "report_cid": _cid("eval-report-pass"),
        "candidate_cid": _cid("candidate-1"),
        "held_out_benchmark_cid": _cid("benchmark-held-out"),
        "baseline_policy_cid": _cid("policy-v1"),
        "verdict": EvaluationVerdict.PASS.value,
        "declared_thresholds_applied": True,
        "blocking_reasons": (),
        "high_risk_assurance_reduced": False,
    }
    fields.update(overrides)
    return fields


def _fail_evaluation(**overrides: Any) -> dict[str, Any]:
    fields = _pass_evaluation(
        report_cid=_cid("eval-report-fail"),
        verdict=EvaluationVerdict.FAIL.value,
        blocking_reasons=("critical_omission_detection_below_threshold",),
    )
    fields.update(overrides)
    return fields


def _released_sealer_surface(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "__name__": "fake.released.proof_sealer",
        "IncrementalProofSealer": object,
        "DeltaSeal": object,
        "build_delta_seal": lambda *a, **k: {"delta": True},
        "publish_delta_seal": lambda *a, **k: {"published": True},
        "FullCheckpointSeal": object,
        "create_full_checkpoint": lambda *a, **k: {"full": True},
        "publish_full_checkpoint": lambda *a, **k: {"published": True},
        "INCREMENTAL_PROOF_SEALER_INTERFACE": "IncrementalProofSealer@1",
        "IS_ZK_SEALER": True,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _verification_bundle(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/verification-bundle@1",
        "interface_id": "VerificationBundle@1",
        "kind": "verification_bundle",
        "verification_plan": {"plan_id": "plan-release-qual"},
        "receipts": (
            {
                "receipt_id": "r1",
                "status": "passed",
            },
        ),
        "unresolved_requirement_ids": (),
        "repository_tree_cid": _cid("repo-tree"),
        "environment_cid": _cid("env"),
    }
    fields.update(overrides)
    return fields


def _auth_cid() -> str:
    return _cid("external-release-qualification-auth")


# ---------------------------------------------------------------------------
# Constants / structural
# ---------------------------------------------------------------------------


def test_evidence_and_interface_constants() -> None:
    assert SCG_SEAL_BINDING_EVIDENCE == "scg/seal-binding@1"
    assert SCG_RELEASE_QUALIFICATION_EVIDENCE == "scg/release-qualification@1"
    assert (
        SEMANTIC_GOVERNOR_SEAL_ADAPTER_INTERFACE
        == "SemanticGovernorSealAdapter@1"
    )
    assert QUALIFY_POLICY_CANDIDATE_INTERFACE == "qualify_policy_candidate@1"
    assert SEAL_GOVERNOR_RUN_INTERFACE == "seal_governor_run@1"
    assert VERIFY_GOVERNOR_SEAL_INTERFACE == "verify_governor_seal@1"
    assert RELEASE_QUALIFICATION_INTERFACE == "ReleaseQualification@1"
    assert GOVERNOR_SEAL_INTERFACE == "GovernorSeal@1"
    assert BOUNDED_CLAIM_SET_INTERFACE == "BoundedSealClaimSet@1"
    assert RELEASE_QUALIFICATION_SCHEMA.endswith("release-qualification@1")
    assert GOVERNOR_SEAL_SCHEMA.endswith("governor-seal@1")


def test_bounded_claims_are_closed_and_exclude_forbidden() -> None:
    allowed = {item.value for item in BoundedClaimKind}
    assert allowed == set(DEFAULT_BOUNDED_CLAIMS)
    assert "semantic_sufficiency" in FORBIDDEN_CLAIM_KINDS
    assert "zk_proof" in FORBIDDEN_CLAIM_KINDS
    assert "execution_proof" in FORBIDDEN_CLAIM_KINDS
    assert "ivp_commitment_is_sealer" in FORBIDDEN_CLAIM_KINDS
    assert allowed.isdisjoint(FORBIDDEN_CLAIM_KINDS)


def test_qualification_paths_are_closed() -> None:
    assert {p.value for p in QualificationPath} == {
        "incremental_seal",
        "authorized_release_qualification",
        "blocked",
    }


# ---------------------------------------------------------------------------
# Missing sealer is typed unavailable; never VerificationCommitment
# ---------------------------------------------------------------------------


def test_missing_sealer_is_typed_unavailable_by_default() -> None:
    adapter = load_seal_adapter()
    cap = adapter.capability
    assert cap.available is False
    assert cap.seal_status == SealStatus.UNAVAILABLE.value
    assert cap.can_be_satisfied_by_ivp_commitment is False
    assert cap.is_zk is False
    with pytest.raises(GovernorCapabilityUnavailable):
        adapter.require_sealer()


def test_ivp_commitment_never_satisfies_sealer_capability() -> None:
    class VerificationCommitment:
        IS_ZERO_KNOWLEDGE_PROOF = False

    for evidence in (
        VerificationCommitment,
        VerificationCommitment(),
        "VerificationCommitment",
        "build_verification_commitment",
        EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
        {
            "schema": EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            "interface_id": EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
            "kind": "verification_commitment",
        },
    ):
        cap = sealer_capability_from_evidence(evidence)
        assert cap.available is False, evidence
        assert cap.reason_code == "ivp_commitment_not_sealer", evidence
        assert cap.can_be_satisfied_by_ivp_commitment is False, evidence
        assert cap.is_zk is False, evidence


def test_adapter_rejects_commitment_as_sealer() -> None:
    adapter = SemanticGovernorSealAdapter()
    with pytest.raises(GovernorCapabilityUnavailable) as excinfo:
        adapter.reject_commitment_as_sealer("VerificationCommitment")
    assert excinfo.value.reason_code == "ivp_commitment_not_sealer"


def test_qualify_rejects_ivp_commitment_as_incremental_seal_evidence() -> None:
    evaluation = _pass_evaluation()
    result = qualify_policy_candidate(
        evaluation,
        incremental_seal_evidence={
            "schema": EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            "kind": "verification_commitment",
        },
    )
    assert result.promotion_allowed is False
    assert result.path == QualificationPath.BLOCKED.value
    assert result.seal_status == SealStatus.UNAVAILABLE.value
    assert REASON_IVP_COMMITMENT_NOT_SEALER in result.blocking_reasons
    assert result.sealer_capability["can_be_satisfied_by_ivp_commitment"] is False


def test_qualify_rejects_ivp_commitment_as_release_bundle() -> None:
    evaluation = _pass_evaluation()
    result = qualify_policy_candidate(
        evaluation,
        sealer_surface=SimpleNamespace(__name__="empty"),
        release_qualification_authorization_cid=_auth_cid(),
        release_qualification_bundle={
            "schema": EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            "interface_id": EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
            "kind": "verification_commitment",
        },
    )
    assert result.promotion_allowed is False
    assert result.path == QualificationPath.BLOCKED.value
    assert REASON_BUNDLE_IS_COMMITMENT in result.blocking_reasons
    assert REASON_IVP_COMMITMENT_NOT_SEALER in result.blocking_reasons
    assert result.sealer_available is False
    assert result.seal_status == SealStatus.UNAVAILABLE.value


def test_probe_with_commitment_surface_stays_unavailable() -> None:
    class VerificationCommitment:
        pass

    adapter = SemanticGovernorSealAdapter(
        sealer_surface=VerificationCommitment()
    )
    cap = adapter.probe()
    assert cap.available is False
    assert cap.reason_code == "ivp_commitment_not_sealer"
    assert adapter.sealer_status() == SealStatus.UNAVAILABLE.value


# ---------------------------------------------------------------------------
# Promotion blocked without release qualification
# ---------------------------------------------------------------------------


def test_promotion_blocked_when_sealer_missing_and_no_qualification() -> None:
    evaluation = _pass_evaluation()
    result = qualify_policy_candidate(
        evaluation,
        sealer_surface=SimpleNamespace(__name__="empty.module"),
    )
    assert result.promotion_allowed is False
    assert result.path == QualificationPath.BLOCKED.value
    assert result.seal_status == SealStatus.UNAVAILABLE.value
    assert result.sealer_available is False
    assert REASON_MISSING_QUALIFICATION_AUTHORIZATION in result.blocking_reasons
    assert REASON_MISSING_VERIFICATION_BUNDLE in result.blocking_reasons
    assert REASON_MISSING_RELEASE_QUALIFICATION in result.blocking_reasons
    assert REASON_PROMOTION_BLOCKED in result.blocking_reasons
    with pytest.raises(SealingError) as excinfo:
        result.require_promotion_allowed()
    assert REASON_PROMOTION_BLOCKED in str(excinfo.value)


def test_promotion_blocked_without_authorization_even_with_bundle() -> None:
    evaluation = _pass_evaluation()
    result = qualify_policy_candidate(
        evaluation,
        release_qualification_bundle=_verification_bundle(),
    )
    assert result.promotion_allowed is False
    assert REASON_MISSING_QUALIFICATION_AUTHORIZATION in result.blocking_reasons


def test_promotion_blocked_without_bundle_even_with_authorization() -> None:
    evaluation = _pass_evaluation()
    result = qualify_policy_candidate(
        evaluation,
        release_qualification_authorization_cid=_auth_cid(),
    )
    assert result.promotion_allowed is False
    assert REASON_MISSING_VERIFICATION_BUNDLE in result.blocking_reasons


def test_promotion_blocked_on_self_authorization() -> None:
    evaluation = _pass_evaluation()
    result = qualify_policy_candidate(
        evaluation,
        release_qualification_authorization_cid=evaluation["report_cid"],
        release_qualification_bundle=_verification_bundle(),
    )
    assert result.promotion_allowed is False
    assert REASON_SELF_AUTHORIZATION in result.blocking_reasons


def test_promotion_blocked_when_evaluation_not_pass() -> None:
    evaluation = _fail_evaluation()
    result = qualify_policy_candidate(
        evaluation,
        release_qualification_authorization_cid=_auth_cid(),
        release_qualification_bundle=_verification_bundle(),
    )
    assert result.promotion_allowed is False
    assert REASON_EVALUATION_NOT_PASS in result.blocking_reasons
    assert result.path == QualificationPath.BLOCKED.value


def test_require_promotion_allowed_on_seal_fails_when_blocked() -> None:
    evaluation = _pass_evaluation()
    with pytest.raises(SealingError) as excinfo:
        seal_governor_run(
            evaluation,
            require_promotion_allowed=True,
        )
    assert REASON_PROMOTION_BLOCKED in str(excinfo.value)


# ---------------------------------------------------------------------------
# Authorized release-qualification path (sealer unavailable)
# ---------------------------------------------------------------------------


def test_authorized_release_qualification_allows_promotion_when_sealer_unavailable() -> None:
    evaluation = _pass_evaluation()
    auth = _auth_cid()
    bundle = _verification_bundle()
    result = qualify_policy_candidate(
        evaluation,
        sealer_surface=SimpleNamespace(__name__="empty"),
        release_qualification_authorization_cid=auth,
        release_qualification_bundle=bundle,
    )
    assert result.promotion_allowed is True
    assert (
        result.path
        == QualificationPath.AUTHORIZED_RELEASE_QUALIFICATION.value
    )
    assert result.sealer_available is False
    assert result.seal_status == SealStatus.UNAVAILABLE.value
    assert result.authorization_cid == auth
    assert result.verification_bundle_cid is not None
    assert result.blocking_reasons == ()
    assert result.evaluation_report_cid == evaluation["report_cid"]
    assert result.candidate_cid == evaluation["candidate_cid"]
    assert result.baseline_policy_cid == evaluation["baseline_policy_cid"]
    # Sealer still cannot be satisfied by IVP commitment.
    assert result.sealer_capability["can_be_satisfied_by_ivp_commitment"] is False
    assert result.sealer_capability["is_zk"] is False
    result.require_promotion_allowed()  # does not raise


def test_authorized_release_qualification_accepts_bundle_cid_string() -> None:
    evaluation = _pass_evaluation()
    bundle_cid = _cid("bundle-ref")
    result = qualify_policy_candidate(
        evaluation,
        release_qualification_authorization_cid=_auth_cid(),
        release_qualification_bundle=bundle_cid,
    )
    assert result.promotion_allowed is True
    assert result.verification_bundle_cid == bundle_cid
    assert result.seal_status == SealStatus.UNAVAILABLE.value


def test_release_qualification_round_trip_identity() -> None:
    evaluation = _pass_evaluation()
    result = qualify_policy_candidate(
        evaluation,
        release_qualification_authorization_cid=_auth_cid(),
        release_qualification_bundle=_verification_bundle(),
    )
    restored = ReleaseQualification.from_dict(result.to_dict())
    assert restored.qualification_cid == result.qualification_cid
    assert restored.promotion_allowed is True
    assert restored.path == result.path


# ---------------------------------------------------------------------------
# Released incremental sealer path
# ---------------------------------------------------------------------------


def test_released_sealer_with_seal_evidence_allows_promotion() -> None:
    evaluation = _pass_evaluation()
    surface = _released_sealer_surface()
    seal_evidence = {"seal_cid": _cid("delta-seal-1"), "kind": "delta_seal"}
    result = qualify_policy_candidate(
        evaluation,
        sealer_surface=surface,
        incremental_seal_evidence=seal_evidence,
    )
    assert result.promotion_allowed is True
    assert result.path == QualificationPath.INCREMENTAL_SEAL.value
    assert result.sealer_available is True
    assert result.seal_status == SealStatus.AVAILABLE.value
    assert result.incremental_seal_cid == seal_evidence["seal_cid"]
    assert result.blocking_reasons == ()
    assert result.sealer_capability["is_full_or_delta_seal"] is True
    assert result.sealer_capability["is_zk"] is True


def test_released_sealer_without_seal_evidence_blocks() -> None:
    evaluation = _pass_evaluation()
    result = qualify_policy_candidate(
        evaluation,
        sealer_surface=_released_sealer_surface(),
    )
    assert result.promotion_allowed is False
    assert result.path == QualificationPath.BLOCKED.value
    assert "missing_incremental_seal_evidence" in result.blocking_reasons


def test_released_sealer_rejects_commitment_as_seal_evidence() -> None:
    evaluation = _pass_evaluation()
    # Even with a released sealer, IVP commitment must not qualify as seal
    # evidence. The early IVP gate returns blocked before capability resolve.
    result = qualify_policy_candidate(
        evaluation,
        sealer_surface=_released_sealer_surface(),
        incremental_seal_evidence="VerificationCommitment",
    )
    assert result.promotion_allowed is False
    assert REASON_IVP_COMMITMENT_NOT_SEALER in result.blocking_reasons


# ---------------------------------------------------------------------------
# Seal binding: policy identities + bounded claims only
# ---------------------------------------------------------------------------


def test_seal_binds_evaluation_policy_and_default_artifacts() -> None:
    evaluation = _pass_evaluation()
    auth = _auth_cid()
    qual = qualify_policy_candidate(
        evaluation,
        release_qualification_authorization_cid=auth,
        release_qualification_bundle=_verification_bundle(),
    )
    seal = seal_governor_run(
        evaluation,
        qualification=qual,
        bindings=(
            {
                "role": ARTIFACT_ROLE_CONTEXT_PACK,
                "artifact_cid": _cid("context-pack-1"),
            },
            {
                "role": ARTIFACT_ROLE_DIFFERENTIAL_REPORT,
                "artifact_cid": _cid("diff-report-1"),
            },
            {
                "role": ARTIFACT_ROLE_CALIBRATION_PROFILE,
                "artifact_cid": _cid("calib-1"),
            },
            {
                "role": ARTIFACT_ROLE_PROMOTION_DECISION,
                "artifact_cid": _cid("promo-decision-1"),
            },
        ),
    )
    assert seal.evaluation_report_cid == evaluation["report_cid"]
    assert seal.candidate_cid == evaluation["candidate_cid"]
    assert seal.baseline_policy_cid == evaluation["baseline_policy_cid"]
    assert seal.qualification_cid == qual.qualification_cid
    assert seal.authorization_cid == auth
    assert seal.sealer_available is False
    assert seal.seal_status == SealStatus.UNAVAILABLE.value
    assert seal.is_zk is False
    assert set(seal.claims) == set(DEFAULT_BOUNDED_CLAIMS)
    roles = {b.role for b in seal.bindings}
    assert ARTIFACT_ROLE_CONTEXT_PACK in roles
    assert ARTIFACT_ROLE_DIFFERENTIAL_REPORT in roles
    assert ARTIFACT_ROLE_CALIBRATION_PROFILE in roles
    assert ARTIFACT_ROLE_PROMOTION_DECISION in roles
    assert "evaluation_report" in roles
    assert "candidate" in roles
    assert "policy" in roles
    assert "benchmark" in roles
    # Every binding carries evaluated policy/evaluation identities.
    for binding in seal.bindings:
        assert binding.policy_cid == evaluation["baseline_policy_cid"]
        assert binding.evaluation_report_cid == evaluation["report_cid"]
    # Explicit non-claims are bound into the identity payload.
    payload = seal.identity_payload()
    assert "semantic_sufficiency" in payload["non_claims"]
    assert "zk_proof" in payload["non_claims"]
    assert seal.metadata["promotion_allowed"] is True


def test_seal_with_released_sealer_is_available_and_may_claim_zk() -> None:
    evaluation = _pass_evaluation()
    surface = _released_sealer_surface()
    seal_evidence = {"seal_cid": _cid("full-checkpoint-1")}
    qual = qualify_policy_candidate(
        evaluation,
        sealer_surface=surface,
        incremental_seal_evidence=seal_evidence,
    )
    seal = seal_governor_run(
        evaluation,
        qualification=qual,
        sealer_surface=surface,
    )
    assert seal.sealer_available is True
    assert seal.seal_status == SealStatus.AVAILABLE.value
    assert seal.is_zk is True
    assert seal.incremental_seal_cid == seal_evidence["seal_cid"]
    assert seal.qualification_path == QualificationPath.INCREMENTAL_SEAL.value


def test_seal_never_upgrades_unavailable_sealer_to_zk() -> None:
    evaluation = _pass_evaluation()
    seal = seal_governor_run(
        evaluation,
        qualification=qualify_policy_candidate(
            evaluation,
            release_qualification_authorization_cid=_auth_cid(),
            release_qualification_bundle=_verification_bundle(),
        ),
    )
    assert seal.sealer_available is False
    assert seal.is_zk is False
    assert seal.seal_status == SealStatus.UNAVAILABLE.value
    # Forging is_zk on an unavailable seal must fail.
    with pytest.raises(SealingError):
        GovernorSeal(
            seal_id=seal.seal_id,
            seal_status=SealStatus.UNAVAILABLE.value,
            claims=seal.claims,
            evaluation_report_cid=seal.evaluation_report_cid,
            candidate_cid=seal.candidate_cid,
            baseline_policy_cid=seal.baseline_policy_cid,
            qualification_cid=seal.qualification_cid,
            qualification_path=seal.qualification_path,
            sealer_available=False,
            is_zk=True,
            bindings=seal.bindings,
        )


def test_overclaim_rejected_at_seal_time() -> None:
    evaluation = _pass_evaluation()
    with pytest.raises(SealingError) as excinfo:
        seal_governor_run(
            evaluation,
            claims=("semantic_sufficiency",),
            qualification=qualify_policy_candidate(
                evaluation,
                release_qualification_authorization_cid=_auth_cid(),
                release_qualification_bundle=_verification_bundle(),
                claims=DEFAULT_BOUNDED_CLAIMS,
            ),
        )
    assert REASON_OVERCLAIM in str(excinfo.value) or REASON_UNAUTHORIZED_CLAIM in str(
        excinfo.value
    )


def test_unknown_claim_kind_rejected() -> None:
    evaluation = _pass_evaluation()
    with pytest.raises(SealingError) as excinfo:
        qualify_policy_candidate(
            evaluation,
            claims=("not_a_real_claim",),
            release_qualification_authorization_cid=_auth_cid(),
            release_qualification_bundle=_verification_bundle(),
        )
    assert REASON_UNAUTHORIZED_CLAIM in str(excinfo.value)


def test_binding_policy_mismatch_fails_closed() -> None:
    evaluation = _pass_evaluation()
    with pytest.raises(SealingError) as excinfo:
        seal_governor_run(
            evaluation,
            qualification=qualify_policy_candidate(
                evaluation,
                release_qualification_authorization_cid=_auth_cid(),
                release_qualification_bundle=_verification_bundle(),
            ),
            bindings=(
                SealArtifactBinding(
                    role=ARTIFACT_ROLE_CONTEXT_PACK,
                    artifact_cid=_cid("ctx"),
                    policy_cid=_cid("other-policy"),
                    evaluation_report_cid=evaluation["report_cid"],
                ),
            ),
        )
    assert "binding" in str(excinfo.value).lower() or "mismatch" in str(
        excinfo.value
    ).lower()


# ---------------------------------------------------------------------------
# verify_governor_seal
# ---------------------------------------------------------------------------


def test_verify_governor_seal_recomputes_identity() -> None:
    evaluation = _pass_evaluation()
    seal = seal_governor_run(
        evaluation,
        qualification=qualify_policy_candidate(
            evaluation,
            release_qualification_authorization_cid=_auth_cid(),
            release_qualification_bundle=_verification_bundle(),
        ),
    )
    cid = verify_governor_seal(
        seal,
        expected_evaluation_report_cid=evaluation["report_cid"],
        expected_candidate_cid=evaluation["candidate_cid"],
        expected_policy_cid=evaluation["baseline_policy_cid"],
    )
    assert cid == seal.seal_cid
    # Mapping round-trip
    cid2 = verify_governor_seal(seal.to_dict())
    assert cid2 == seal.seal_cid


def test_verify_detects_tampered_seal_cid() -> None:
    evaluation = _pass_evaluation()
    seal = seal_governor_run(
        evaluation,
        qualification=qualify_policy_candidate(
            evaluation,
            release_qualification_authorization_cid=_auth_cid(),
            release_qualification_bundle=_verification_bundle(),
        ),
    )
    payload = seal.to_dict()
    payload["seal_cid"] = _cid("tampered")
    with pytest.raises(SealingError):
        verify_governor_seal(payload)


def test_verify_rejects_identity_mismatch() -> None:
    evaluation = _pass_evaluation()
    seal = seal_governor_run(
        evaluation,
        qualification=qualify_policy_candidate(
            evaluation,
            release_qualification_authorization_cid=_auth_cid(),
            release_qualification_bundle=_verification_bundle(),
        ),
    )
    with pytest.raises(SealingError):
        verify_governor_seal(
            seal,
            expected_evaluation_report_cid=_cid("other-report"),
        )


def test_verify_can_require_available_sealer() -> None:
    evaluation = _pass_evaluation()
    seal = seal_governor_run(
        evaluation,
        qualification=qualify_policy_candidate(
            evaluation,
            release_qualification_authorization_cid=_auth_cid(),
            release_qualification_bundle=_verification_bundle(),
        ),
    )
    assert seal.seal_status == SealStatus.UNAVAILABLE.value
    with pytest.raises(SealingError):
        verify_governor_seal(seal, allow_unavailable=False)


# ---------------------------------------------------------------------------
# Adapter surface
# ---------------------------------------------------------------------------


def test_adapter_qualify_and_seal_end_to_end() -> None:
    adapter = load_seal_adapter()
    evaluation = _pass_evaluation()
    # Blocked path
    blocked = adapter.qualify_policy_candidate(evaluation)
    assert blocked.promotion_allowed is False
    # Authorized release qualification
    allowed = adapter.qualify_policy_candidate(
        evaluation,
        release_qualification_authorization_cid=_auth_cid(),
        release_qualification_bundle=_verification_bundle(),
    )
    assert allowed.promotion_allowed is True
    seal = adapter.seal_governor_run(
        evaluation,
        qualification=allowed,
        bindings=(
            {
                "role": ARTIFACT_ROLE_CONTEXT_PACK,
                "artifact_cid": _cid("ctx-e2e"),
            },
        ),
    )
    assert adapter.verify_governor_seal(seal) == seal.seal_cid
    view = adapter.runtime_view()
    assert view["can_be_satisfied_by_ivp_commitment"] is False
    assert view["interface_id"] == SEMANTIC_GOVERNOR_SEAL_ADAPTER_INTERFACE
    assert BoundedClaimKind.EXACT_ARTIFACTS_EVALUATED.value in view["bounded_claims"]


def test_adapter_with_released_sealer() -> None:
    surface = _released_sealer_surface()
    adapter = load_seal_adapter(surface, require_sealer=True)
    assert adapter.sealer_is_available() is True
    evaluation = _pass_evaluation()
    qual = adapter.qualify_policy_candidate(
        evaluation,
        incremental_seal_evidence={"seal_cid": _cid("delta-e2e")},
    )
    assert qual.promotion_allowed is True
    assert qual.path == QualificationPath.INCREMENTAL_SEAL.value
    seal = adapter.seal_governor_run(evaluation, qualification=qual)
    assert seal.seal_status == SealStatus.AVAILABLE.value
    assert seal.is_zk is True


def test_load_seal_adapter_require_sealer_fails_closed() -> None:
    with pytest.raises(GovernorCapabilityUnavailable):
        load_seal_adapter(
            SimpleNamespace(__name__="empty"),
            require_sealer=True,
        )


def test_seal_may_be_emitted_while_promotion_blocked() -> None:
    """Post-decision sealing binds but never upgrades evidence / unblocks promotion."""

    evaluation = _pass_evaluation()
    seal = seal_governor_run(evaluation)
    assert seal.metadata["promotion_allowed"] is False
    assert seal.seal_status == SealStatus.UNAVAILABLE.value
    assert seal.qualification_path == QualificationPath.BLOCKED.value
    # Seal still binds policy/evaluation identities.
    assert seal.evaluation_report_cid == evaluation["report_cid"]
    assert seal.baseline_policy_cid == evaluation["baseline_policy_cid"]
    verify_governor_seal(seal)


def test_live_sealer_probe_unavailable_on_current_tree() -> None:
    """On the current SCG tree the released sealer public API is absent."""

    cap = probe_incremental_sealer_capability(
        candidate_modules=(
            "ipfs_accelerate_py.agent_supervisor.proof_sealer",
            "ipfs_accelerate_py.agent_supervisor.incremental_proof_sealer",
            "ipfs_kit_py.proof_sealer",
            "ipfs_kit_py.incremental_proof_sealer",
        )
    )
    assert cap.available is False
    assert cap.seal_status == SealStatus.UNAVAILABLE.value
    adapter = load_seal_adapter()
    assert adapter.sealer_is_available() is False
    # Qualification without independent auth remains blocked.
    result = adapter.qualify_policy_candidate(_pass_evaluation())
    assert result.promotion_allowed is False
    assert result.seal_status == SealStatus.UNAVAILABLE.value
