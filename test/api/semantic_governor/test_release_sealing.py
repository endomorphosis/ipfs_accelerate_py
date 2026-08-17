"""SCG-047: qualify released IncrementalProofSealer binding and rollback evidence.

Acceptance criteria enforced here:

* Seal scope is precise (closed bounded claims only; no overclaim / ZK substitution).
* Stale, corrupt, or mismatched candidates fail closed without head mutation.
* Authorized promotion + rollback CAS is reproducible in a hermetic namespace.
* An unavailable released sealer never stalls unrelated governor qualification
  (authorized VerificationBundle release-qualification path still completes).

Conflict policy: if the canonical sealer has not released on this tree, record
a truthful unavailable qualification and never claim proof-backed promotion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
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
)

from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
    DurableCoordinationStore,
    cid_for_artifact,
)
from ipfs_kit_py.semantic_governor_store.contracts import GovernorStoreStatus
from ipfs_kit_py.semantic_governor_store.policy import DurablePolicyCASRepositories

from ipfs_accelerate_py.agent_supervisor.semantic_governor.adapters import (
    SealStatus,
    load_runtime_adapters,
    probe_incremental_sealer_capability,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.promotion import (
    PromotionStatus,
    REASON_CAS_CONFLICT,
    REASON_MISMATCHED_EVALUATION,
    REASON_STALE_CANDIDATE,
    RollbackStatus,
    promote_compression_policy,
    rollback_compression_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.sealing import (
    ARTIFACT_ROLE_CALIBRATION_PROFILE,
    ARTIFACT_ROLE_CONTEXT_PACK,
    ARTIFACT_ROLE_DIFFERENTIAL_REPORT,
    ARTIFACT_ROLE_INCREMENTAL_SEAL,
    ARTIFACT_ROLE_PROMOTION_DECISION,
    BOUNDED_CLAIM_SET_INTERFACE,
    DEFAULT_BOUNDED_CLAIMS,
    FORBIDDEN_CLAIM_KINDS,
    GOVERNOR_SEAL_INTERFACE,
    QualificationPath,
    REASON_BINDING_MISMATCH,
    REASON_IDENTITY_MISMATCH,
    REASON_IVP_COMMITMENT_NOT_SEALER,
    REASON_OVERCLAIM,
    REASON_PROMOTION_BLOCKED,
    REASON_STALE_SEAL,
    REASON_UNAUTHORIZED_CLAIM,
    RELEASE_QUALIFICATION_INTERFACE,
    SCG_RELEASE_QUALIFICATION_EVIDENCE,
    SCG_SEAL_BINDING_EVIDENCE,
    SealArtifactBinding,
    SealingError,
    load_seal_adapter,
    qualify_policy_candidate,
    seal_governor_run,
    verify_governor_seal,
)


# ---------------------------------------------------------------------------
# Paths / evidence constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = (
    REPO_ROOT / "artifacts" / "agent_supervisor" / "semantic_compression_governor"
)
SEAL_QUALIFICATION_PATH = ARTIFACT_DIR / "seal_qualification.json"
ROLLBACK_PATH = ARTIFACT_DIR / "rollback.json"

TASK_ID = "SCG-047"
GOAL_ID = "SCG-G090"
SCG_INCREMENTAL_SEAL_QUALIFICATION_EVIDENCE = "scg/incremental-seal-qualification@1"
SCG_ROLLBACK_EVIDENCE = "scg/rollback@1"
SEAL_QUALIFICATION_INTERFACE = "SemanticGovernorSealQualification@1"
SEAL_QUALIFICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "seal-qualification@1"
)
ROLLBACK_QUALIFICATION_INTERFACE = "SemanticGovernorRollbackQualification@1"
ROLLBACK_QUALIFICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "rollback-qualification@1"
)

HERMETIC_WORKSPACE = "scg047-release-seal"


# ---------------------------------------------------------------------------
# Content-address helpers
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    """Canonical dag-json CID (store CAS requires dag-json profile)."""

    return cid_for_structured({"test_label": label, "schema": "test/label@1"})


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in value.items() if k != "content_id"}
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _structural_outcome_cid(fields: Mapping[str, Any]) -> str:
    """Content-addressed fingerprint of stable CAS outcome fields only.

    Live PromotionResult/PolicyRollbackResult.result_cid embeds policy_cas
    transition CIDs that vary across hermetic store instances. Qualification
    evidence records structural outcomes so rollback remains reproducible.
    """

    return cid_for_structured(
        {
            "schema": "scg/structural-cas-outcome@1",
            "fields": dict(fields),
        }
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(encoded, encoding="utf-8")
    tmp.replace(path)
    return dict(payload)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return data


# ---------------------------------------------------------------------------
# Evaluation / sealer recipes
# ---------------------------------------------------------------------------


def _pass_evaluation(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "report_cid": _cid("scg047-eval-report-pass"),
        "candidate_cid": _cid("scg047-candidate-1"),
        "held_out_benchmark_cid": _cid("scg047-benchmark-held-out"),
        "baseline_policy_cid": _cid("scg047-policy-v1"),
        "verdict": EvaluationVerdict.PASS.value,
        "declared_thresholds_applied": True,
        "blocking_reasons": (),
        "high_risk_assurance_reduced": False,
    }
    fields.update(overrides)
    return fields


def _verification_bundle(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/verification-bundle@1",
        "interface_id": "VerificationBundle@1",
        "kind": "verification_bundle",
        "verification_plan": {"plan_id": "plan-scg047-release-qual"},
        "receipts": ({"receipt_id": "r-scg047-1", "status": "passed"},),
        "unresolved_requirement_ids": (),
        "repository_tree_cid": _cid("scg047-repo-tree"),
        "environment_cid": _cid("scg047-env"),
    }
    fields.update(overrides)
    return fields


def _auth_cid(label: str = "scg047-external-release-qual-auth") -> str:
    return _cid(label)


def _released_sealer_surface(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "__name__": "fake.released.proof_sealer.scg047",
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


# ---------------------------------------------------------------------------
# Promotion recipes (hermetic CAS)
# ---------------------------------------------------------------------------


def _header(artifact_kind: str, **overrides: Any) -> GovernorArtifactHeader:
    fields: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "repository_state_cid": _cid("scg047-repo-state"),
        "context_pack_cid": _cid("scg047-context-pack"),
        "verification_bundle_cid": _cid("scg047-verification-bundle"),
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
            input_cids=(_cid("scg047-input-a"),),
            tool_ids=("policy.v1",),
            policy_cid=_cid("scg047-policy-v1"),
            notes=None,
        ),
        "terminal_status": GovernorTerminalStatus.COMPLETE,
        "assumptions": (
            GovernorAssumption(
                assumption_id="partition_disjoint",
                kind=AssumptionKind.VERIFICATION,
                statement="Held-out partition is disjoint from calibration",
                supporting_cids=(_cid("scg047-partition"),),
            ),
        ),
        "metadata": {"track": "release-sealing", "task_id": TASK_ID},
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
        "candidate_id": "cand_scg047",
        "base_policy_cid": _cid("scg047-policy-v1"),
        "base_policy_version": "1.0.0",
        "proposal_cid": _cid("scg047-proposal-1"),
        "proposed_policy_cid": _cid("scg047-policy-v2"),
        "proposed_protected_thresholds": _thresholds(),
        "baseline_protected_thresholds": _thresholds(),
        "evaluation_partition": EvidencePartition.HELD_OUT,
        "external_authorization_cid": None,
        "notes": None,
        "metadata": {"task_id": TASK_ID},
    }
    fields.update(overrides)
    return CompressionPolicyCandidate(**fields)


def _promo_evaluation(
    candidate: CompressionPolicyCandidate, **overrides: Any
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "report_cid": _cid(f"scg047-eval-pass-{candidate.candidate_id}"),
        "candidate_cid": candidate.candidate_cid,
        "held_out_benchmark_cid": _cid("scg047-benchmark-held-out"),
        "baseline_policy_cid": candidate.base_policy_cid,
        "verdict": EvaluationVerdict.PASS.value,
        "partition": EvidencePartition.HELD_OUT.value,
        "declared_thresholds_applied": True,
        "blocking_reasons": (),
        "high_risk_assurance_reduced": False,
    }
    fields.update(overrides)
    return fields


def _store_block(store: DurableCoordinationStore, name: str) -> str:
    payload = {
        "schema": "example/governor-policy@1",
        "name": name,
        "task_id": TASK_ID,
    }
    return store.put(payload, expected_cid=cid_for_artifact(payload), replicate=False)[
        "cid"
    ]


@pytest.fixture()
def coordination(tmp_path: Path) -> DurableCoordinationStore:
    root = DurableCoordinationStore(tmp_path / "scg047-store")
    yield root
    root.close()


@pytest.fixture()
def cas(coordination: DurableCoordinationStore) -> DurablePolicyCASRepositories:
    return DurablePolicyCASRepositories(coordination)


def _seed_policy_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
    *,
    name: str = "scg047-policy-v1",
) -> str:
    cid = _store_block(coordination, name)
    result = cas.policy.compare_and_swap_policy(
        HERMETIC_WORKSPACE,
        expected_generation=0,
        expected_policy_cid=None,
        new_policy_cid=cid,
        operation_id=f"seed-{name}",
    )
    assert result.status is GovernorStoreStatus.UPDATED
    return cid


def _promotion_auth() -> str:
    return _cid("scg047-external-promotion-authorization-board")


# ---------------------------------------------------------------------------
# Qualification builders used for promotion and artifact emission
# ---------------------------------------------------------------------------


def _live_sealer_mapping() -> dict[str, Any]:
    return dict(probe_incremental_sealer_capability().to_mapping())


def _authorized_release_qualification(
    evaluation: Mapping[str, Any],
) -> Any:
    return qualify_policy_candidate(
        evaluation,
        sealer_surface=SimpleNamespace(__name__="empty.scg047"),
        release_qualification_authorization_cid=_auth_cid(),
        release_qualification_bundle=_verification_bundle(),
        metadata={"task_id": TASK_ID, "goal_id": GOAL_ID},
    )


def _incremental_seal_qualification(
    evaluation: Mapping[str, Any],
    *,
    seal_cid: str | None = None,
) -> Any:
    surface = _released_sealer_surface()
    evidence = {"seal_cid": seal_cid or _cid("scg047-delta-seal-1"), "kind": "delta_seal"}
    return qualify_policy_candidate(
        evaluation,
        sealer_surface=surface,
        incremental_seal_evidence=evidence,
        metadata={"task_id": TASK_ID, "goal_id": GOAL_ID},
    )


# ---------------------------------------------------------------------------
# Artifact builders
# ---------------------------------------------------------------------------


def build_seal_qualification_artifact() -> dict[str, Any]:
    """Deterministic live-tree seal qualification record (truthful unavailable)."""

    live = _live_sealer_mapping()
    evaluation = _pass_evaluation()
    adapter = load_seal_adapter()

    blocked = qualify_policy_candidate(
        evaluation,
        sealer_surface=SimpleNamespace(__name__="empty.scg047"),
    )
    authorized = _authorized_release_qualification(evaluation)
    released_path = _incremental_seal_qualification(evaluation)

    # Seal under authorized path (sealer unavailable) — precise scope only.
    seal = seal_governor_run(
        evaluation,
        qualification=authorized,
        bindings=(
            {
                "role": ARTIFACT_ROLE_CONTEXT_PACK,
                "artifact_cid": _cid("scg047-context-pack-bind"),
            },
            {
                "role": ARTIFACT_ROLE_DIFFERENTIAL_REPORT,
                "artifact_cid": _cid("scg047-diff-report-bind"),
            },
            {
                "role": ARTIFACT_ROLE_CALIBRATION_PROFILE,
                "artifact_cid": _cid("scg047-calib-bind"),
            },
            {
                "role": ARTIFACT_ROLE_PROMOTION_DECISION,
                "artifact_cid": _cid("scg047-promo-decision-bind"),
            },
        ),
    )
    seal_cid = verify_governor_seal(
        seal,
        expected_evaluation_report_cid=evaluation["report_cid"],
        expected_candidate_cid=evaluation["candidate_cid"],
        expected_policy_cid=evaluation["baseline_policy_cid"],
    )

    bound_roles = sorted({b.role for b in seal.bindings})
    bound_cids = sorted({b.artifact_cid for b in seal.bindings})

    # Stale / corrupt / mismatch cases (pure qualify + seal verify).
    stale_identity = {
        "case": "mismatched_qualification_identity",
        "outcome": "fail_closed",
        "reason_codes": [REASON_IDENTITY_MISMATCH],
    }
    corrupt_seal = {
        "case": "tampered_seal_cid",
        "outcome": "fail_closed",
        "reason_codes": [REASON_STALE_SEAL],
    }
    ivp_rejected = {
        "case": "ivp_commitment_as_seal_evidence",
        "outcome": "fail_closed",
        "reason_codes": [REASON_IVP_COMMITMENT_NOT_SEALER, REASON_PROMOTION_BLOCKED],
        "promotion_allowed": False,
    }

    # Exercise pure fail-closed cases for evidence completeness.
    mismatched_eval = dict(evaluation)
    mismatched_eval["candidate_cid"] = _cid("scg047-other-candidate")
    try:
        seal_governor_run(mismatched_eval, qualification=authorized)
        raise AssertionError("expected identity mismatch to fail closed")
    except SealingError as identity_exc:
        assert REASON_IDENTITY_MISMATCH in str(identity_exc)

    tampered = seal.to_dict()
    tampered["seal_cid"] = _cid("scg047-tampered-seal")
    try:
        verify_governor_seal(tampered)
        raise AssertionError("expected tampered seal to fail closed")
    except SealingError as stale_exc:
        assert REASON_STALE_SEAL in str(stale_exc) or "stale" in str(stale_exc).lower()

    ivp_result = qualify_policy_candidate(
        evaluation,
        sealer_surface=_released_sealer_surface(),
        incremental_seal_evidence="VerificationCommitment",
    )
    assert ivp_result.promotion_allowed is False
    assert REASON_IVP_COMMITMENT_NOT_SEALER in ivp_result.blocking_reasons

    # Binding policy mismatch fails closed.
    try:
        seal_governor_run(
            evaluation,
            qualification=authorized,
            bindings=(
                SealArtifactBinding(
                    role=ARTIFACT_ROLE_CONTEXT_PACK,
                    artifact_cid=_cid("scg047-ctx-bad-policy"),
                    policy_cid=_cid("scg047-other-policy"),
                    evaluation_report_cid=evaluation["report_cid"],
                ),
            ),
        )
        raise AssertionError("expected binding policy mismatch to fail closed")
    except SealingError as bind_exc:
        bind_msg = str(bind_exc).lower()
        assert (
            "binding" in bind_msg
            or "mismatch" in bind_msg
            or REASON_BINDING_MISMATCH in str(bind_exc)
        )

    # Overclaim fails closed.
    try:
        seal_governor_run(
            evaluation,
            claims=("semantic_sufficiency",),
            qualification=authorized,
        )
        raise AssertionError("expected overclaim to fail closed")
    except SealingError as over_exc:
        assert REASON_OVERCLAIM in str(over_exc) or REASON_UNAUTHORIZED_CLAIM in str(
            over_exc
        )

    cases = [
        {
            "case_id": "live_sealer_probe",
            "sealer_available": live["available"],
            "seal_status": live["seal_status"],
            "promotion_allowed_without_independent_auth": blocked.promotion_allowed,
            "path": blocked.path,
        },
        {
            "case_id": "authorized_release_qualification_while_sealer_unavailable",
            "promotion_allowed": authorized.promotion_allowed,
            "path": authorized.path,
            "seal_status": authorized.seal_status,
            "sealer_available": authorized.sealer_available,
            "qualification_cid": authorized.qualification_cid,
            "claims": list(authorized.claims),
        },
        {
            "case_id": "injected_released_incremental_sealer",
            "promotion_allowed": released_path.promotion_allowed,
            "path": released_path.path,
            "seal_status": released_path.seal_status,
            "sealer_available": released_path.sealer_available,
            "incremental_seal_cid": released_path.incremental_seal_cid,
            "note": (
                "Injected released surface for binding verification only; "
                "live tree remains typed unavailable"
            ),
        },
        {
            "case_id": "precise_seal_scope_binding",
            "seal_cid": seal_cid,
            "seal_status": seal.seal_status,
            "is_zk": seal.is_zk,
            "claims": list(seal.claims),
            "bound_roles": bound_roles,
            "bound_artifact_cids": bound_cids,
            "qualification_path": seal.qualification_path,
            "non_claims": list(seal.identity_payload()["non_claims"]),
        },
        stale_identity,
        corrupt_seal,
        ivp_rejected,
        {
            "case": "binding_policy_mismatch",
            "outcome": "fail_closed",
            "reason_codes": [REASON_BINDING_MISMATCH],
        },
        {
            "case": "overclaim_semantic_sufficiency",
            "outcome": "fail_closed",
            "reason_codes": [REASON_OVERCLAIM, REASON_UNAUTHORIZED_CLAIM],
        },
    ]

    artifact: dict[str, Any] = {
        "schema": SEAL_QUALIFICATION_SCHEMA,
        "interface_id": SEAL_QUALIFICATION_INTERFACE,
        "evidence": SCG_INCREMENTAL_SEAL_QUALIFICATION_EVIDENCE,
        "related_evidence": [
            SCG_RELEASE_QUALIFICATION_EVIDENCE,
            SCG_SEAL_BINDING_EVIDENCE,
        ],
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "authoritative": False,
        "status": "qualified_unavailable",
        "current_tree": True,
        "proof_scope_precise": True,
        "sealer": {
            "available": live["available"],
            "adapter_id": live["adapter_id"],
            "interface_id": live["interface_id"],
            "seal_status": live["seal_status"],
            "status": live["status"],
            "is_zk": live["is_zk"],
            "is_full_or_delta_seal": live["is_full_or_delta_seal"],
            "can_be_satisfied_by_ivp_commitment": False,
            "public_module": live["public_module"],
            "reason_code": live["reason_code"],
            "diagnostic": live["diagnostic"],
            "operations": list(live.get("operations") or ()),
            "fingerprints": dict(live.get("fingerprints") or {}),
            "live_adapter_available": adapter.sealer_is_available(),
        },
        "seal_scope": {
            "status": seal.seal_status,
            "seal_cid": seal_cid,
            "sealer_interface_id": BOUNDED_CLAIM_SET_INTERFACE,
            "governor_seal_interface_id": GOVERNOR_SEAL_INTERFACE,
            "release_qualification_interface_id": RELEASE_QUALIFICATION_INTERFACE,
            "qualification_path": seal.qualification_path,
            "qualification_cid": seal.qualification_cid,
            "bounded_claims": list(DEFAULT_BOUNDED_CLAIMS),
            "claims_encoded": list(seal.claims),
            "forbidden_claim_kinds": sorted(FORBIDDEN_CLAIM_KINDS),
            "bound_artifact_cids": bound_cids,
            "bound_roles": bound_roles,
            "evaluation_report_cid": seal.evaluation_report_cid,
            "candidate_cid": seal.candidate_cid,
            "baseline_policy_cid": seal.baseline_policy_cid,
            "is_zk": seal.is_zk,
            "sealer_available": seal.sealer_available,
            "unavailable": seal.seal_status == SealStatus.UNAVAILABLE.value,
            "non_claims": list(seal.identity_payload()["non_claims"]),
        },
        "qualification_paths": {
            "incremental_seal": {
                "available_on_live_tree": False,
                "injected_surface_qualifies": released_path.promotion_allowed,
                "path_token": QualificationPath.INCREMENTAL_SEAL.value,
            },
            "authorized_release_qualification": {
                "promotion_allowed": authorized.promotion_allowed,
                "path_token": QualificationPath.AUTHORIZED_RELEASE_QUALIFICATION.value,
                "qualification_cid": authorized.qualification_cid,
                "authorization_cid": authorized.authorization_cid,
                "verification_bundle_cid": authorized.verification_bundle_cid,
                "seal_status_remains": authorized.seal_status,
            },
            "blocked_without_independent_auth": {
                "promotion_allowed": blocked.promotion_allowed,
                "path_token": QualificationPath.BLOCKED.value,
                "blocking_reasons": list(blocked.blocking_reasons),
            },
        },
        "acceptance": {
            "seal_scope_precise": True,
            "stale_corrupt_mismatched_candidate_fails": True,
            "unavailable_sealer_never_stalls_unrelated_qualification": True,
            "proof_backed_promotion_not_claimed_on_unavailable_sealer": (
                live["available"] is False and seal.is_zk is False
            ),
            "ivp_commitment_never_satisfies_sealer": True,
        },
        "cases": cases,
        "missing_evidence": (
            []
            if live["available"]
            else [
                "released_IncrementalProofSealer_public_api",
                "zk_seal_scope",
                "live_delta_or_full_checkpoint_seal",
            ]
        ),
        "notes": (
            "Live tree probe records typed sealer unavailability. Injected "
            "released-sealer cases verify binding contracts without claiming "
            "that IncrementalProofSealer is present on this tree. Authorized "
            "VerificationBundle release qualification remains the non-proof "
            "promotion gate while the sealer is unavailable."
        ),
    }
    artifact["content_id"] = _content_id(artifact)
    return artifact


def build_rollback_artifact(
    *,
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> dict[str, Any]:
    """Hermetic promote → rollback → re-promote → re-rollback qualification."""

    base = _seed_policy_head(coordination, cas, name="scg047-rb-policy-v1")
    proposed = _store_block(coordination, "scg047-rb-policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _promo_evaluation(candidate)
    qualification = _authorized_release_qualification(evaluation)
    auth = _promotion_auth()

    first_promo = promote_compression_policy(
        candidate,
        evaluation,
        auth,
        release_qualification=qualification,
        policy_repository=cas.policy,
        promotion_repository=cas.promotion,
        workspace=HERMETIC_WORKSPACE,
        operation_id="scg047-promo-1",
        promoted_policy_version="1.1.0",
    )
    assert first_promo.head_mutated is True
    assert first_promo.status == PromotionStatus.PROMOTED.value
    assert cas.current_policy(HERMETIC_WORKSPACE).policy_cid == proposed

    transitions_after_promo = len(cas.policy_transitions(HERMETIC_WORKSPACE))

    first_rollback = rollback_compression_policy(
        auth,
        target_policy_cid=base,
        policy_repository=cas.policy,
        workspace=HERMETIC_WORKSPACE,
        operation_id="scg047-rollback-1",
        current_policy_version="1.1.0",
        target_policy_version="1.0.0",
        candidate_cid=candidate.candidate_cid,
        evaluation_report_cid=evaluation["report_cid"],
    )
    assert first_rollback.status == RollbackStatus.ROLLED_BACK.value
    assert first_rollback.head_mutated is True
    assert cas.current_policy(HERMETIC_WORKSPACE).policy_cid == base
    assert len(cas.policy_transitions(HERMETIC_WORKSPACE)) == transitions_after_promo + 1

    # Re-promote from restored base, then roll back again for reproducibility.
    second_promo = promote_compression_policy(
        candidate,
        evaluation,
        auth,
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=HERMETIC_WORKSPACE,
        operation_id="scg047-promo-2",
        promoted_policy_version="1.1.0",
    )
    assert second_promo.head_mutated is True
    assert cas.current_policy(HERMETIC_WORKSPACE).policy_cid == proposed

    second_rollback = rollback_compression_policy(
        auth,
        target_policy_cid=base,
        policy_repository=cas.policy,
        workspace=HERMETIC_WORKSPACE,
        operation_id="scg047-rollback-2",
        current_policy_version="1.1.0",
        target_policy_version="1.0.0",
        candidate_cid=candidate.candidate_cid,
        evaluation_report_cid=evaluation["report_cid"],
    )
    assert second_rollback.status == RollbackStatus.ROLLED_BACK.value
    assert second_rollback.head_mutated is True
    assert cas.current_policy(HERMETIC_WORKSPACE).policy_cid == base

    # Stale candidate after concurrent race must not mutate head.
    interloper = _store_block(coordination, "scg047-rb-interloper")
    # Re-promote to establish live head, then race with interloper.
    third_promo = promote_compression_policy(
        candidate,
        evaluation,
        auth,
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=HERMETIC_WORKSPACE,
        operation_id="scg047-promo-3",
        promoted_policy_version="1.1.0",
    )
    assert third_promo.head_mutated is True
    live_after_promo = cas.current_policy(HERMETIC_WORKSPACE)
    assert live_after_promo.policy_cid == proposed

    raced = cas.compare_and_swap_policy(
        HERMETIC_WORKSPACE,
        expected_generation=live_after_promo.generation,
        expected_policy_cid=proposed,
        new_policy_cid=interloper,
        operation_id="scg047-race-winner",
    )
    assert raced.status is GovernorStoreStatus.UPDATED

    stale_candidate = _candidate(
        candidate_id="cand_scg047_stale",
        base_policy_cid=base,
        proposed_policy_cid=proposed,
        proposal_cid=_cid("scg047-proposal-stale"),
    )
    stale_eval = _promo_evaluation(stale_candidate)
    stale_qual = _authorized_release_qualification(stale_eval)
    stale_promo = promote_compression_policy(
        stale_candidate,
        stale_eval,
        auth,
        release_qualification=stale_qual,
        policy_repository=cas.policy,
        workspace=HERMETIC_WORKSPACE,
        operation_id="scg047-promo-stale",
        expected_generation=1,
        expected_policy_cid=base,
        promoted_policy_version="1.2.0",
    )
    assert stale_promo.head_mutated is False
    assert (
        REASON_STALE_CANDIDATE in stale_promo.blocking_reasons
        or REASON_CAS_CONFLICT in stale_promo.blocking_reasons
    )
    assert cas.current_policy(HERMETIC_WORKSPACE).policy_cid == interloper

    # Mismatched evaluation cannot mutate head.
    other = _candidate(
        candidate_id="cand_scg047_other",
        base_policy_cid=interloper,
        proposed_policy_cid=_store_block(coordination, "scg047-rb-other-succ"),
        proposal_cid=_cid("scg047-proposal-other"),
    )
    mismatch_eval = _promo_evaluation(other)
    # Bind evaluation of `other` to promotion of a different candidate.
    mismatch_promo = promote_compression_policy(
        candidate,
        mismatch_eval,
        auth,
        release_qualification=_authorized_release_qualification(mismatch_eval),
        policy_repository=cas.policy,
        workspace=HERMETIC_WORKSPACE,
        operation_id="scg047-promo-mismatch",
        promoted_policy_version="1.3.0",
    )
    assert mismatch_promo.head_mutated is False
    assert REASON_MISMATCHED_EVALUATION in mismatch_promo.blocking_reasons
    assert cas.current_policy(HERMETIC_WORKSPACE).policy_cid == interloper

    published = {
        row["new_root_cid"] for row in cas.policy_transitions(HERMETIC_WORKSPACE)
    }
    assert base in published
    assert proposed in published
    assert interloper in published

    first_rb_receipt = (
        None if first_rollback.receipt is None else first_rollback.receipt.to_dict()
    )
    second_rb_receipt = (
        None if second_rollback.receipt is None else second_rollback.receipt.to_dict()
    )

    # Receipt identity fields are stable across equivalent structural operations
    # (operation_id differs so receipt_ids differ; statuses match).
    assert first_rollback.status == second_rollback.status
    assert first_rollback.head_mutated is second_rollback.head_mutated
    assert first_rollback.target_policy_cid == second_rollback.target_policy_cid

    artifact: dict[str, Any] = {
        "schema": ROLLBACK_QUALIFICATION_SCHEMA,
        "interface_id": ROLLBACK_QUALIFICATION_INTERFACE,
        "evidence": SCG_ROLLBACK_EVIDENCE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "authoritative": False,
        "status": "qualified",
        "current_tree": True,
        "rollback_tested": True,
        "reproducible": True,
        "history_preserved": True,
        "workspace": HERMETIC_WORKSPACE,
        "namespace": "hermetic_durable_coordination_store",
        "policy_head": {
            "final_policy_cid": cas.current_policy(HERMETIC_WORKSPACE).policy_cid,
            "final_generation": cas.current_policy(HERMETIC_WORKSPACE).generation,
            "published_policy_cids": sorted(published),
            "transition_count": len(cas.policy_transitions(HERMETIC_WORKSPACE)),
        },
        "operations": [
            {
                "operation_id": "scg047-promo-1",
                "kind": "promote",
                "status": first_promo.status,
                "head_mutated": first_promo.head_mutated,
                "promoted_policy_cid": first_promo.promoted_policy_cid,
                "previous_policy_cid": base,
                "structural_outcome_cid": _structural_outcome_cid(
                    {
                        "operation_id": "scg047-promo-1",
                        "kind": "promote",
                        "status": first_promo.status,
                        "head_mutated": first_promo.head_mutated,
                        "promoted_policy_cid": first_promo.promoted_policy_cid,
                        "previous_policy_cid": base,
                    }
                ),
            },
            {
                "operation_id": "scg047-rollback-1",
                "kind": "rollback",
                "status": first_rollback.status,
                "head_mutated": first_rollback.head_mutated,
                "target_policy_cid": first_rollback.target_policy_cid,
                "structural_outcome_cid": _structural_outcome_cid(
                    {
                        "operation_id": "scg047-rollback-1",
                        "kind": "rollback",
                        "status": first_rollback.status,
                        "head_mutated": first_rollback.head_mutated,
                        "target_policy_cid": first_rollback.target_policy_cid,
                        "receipt_id": (
                            None
                            if first_rb_receipt is None
                            else first_rb_receipt.get("receipt_id")
                        ),
                    }
                ),
                "receipt": first_rb_receipt,
            },
            {
                "operation_id": "scg047-promo-2",
                "kind": "promote",
                "status": second_promo.status,
                "head_mutated": second_promo.head_mutated,
                "promoted_policy_cid": second_promo.promoted_policy_cid,
                "structural_outcome_cid": _structural_outcome_cid(
                    {
                        "operation_id": "scg047-promo-2",
                        "kind": "promote",
                        "status": second_promo.status,
                        "head_mutated": second_promo.head_mutated,
                        "promoted_policy_cid": second_promo.promoted_policy_cid,
                    }
                ),
            },
            {
                "operation_id": "scg047-rollback-2",
                "kind": "rollback",
                "status": second_rollback.status,
                "head_mutated": second_rollback.head_mutated,
                "target_policy_cid": second_rollback.target_policy_cid,
                "structural_outcome_cid": _structural_outcome_cid(
                    {
                        "operation_id": "scg047-rollback-2",
                        "kind": "rollback",
                        "status": second_rollback.status,
                        "head_mutated": second_rollback.head_mutated,
                        "target_policy_cid": second_rollback.target_policy_cid,
                        "receipt_id": (
                            None
                            if second_rb_receipt is None
                            else second_rb_receipt.get("receipt_id")
                        ),
                    }
                ),
                "receipt": second_rb_receipt,
            },
            {
                "operation_id": "scg047-promo-stale",
                "kind": "promote_stale_candidate",
                "status": stale_promo.status,
                "head_mutated": stale_promo.head_mutated,
                "blocking_reasons": list(stale_promo.blocking_reasons),
                "structural_outcome_cid": _structural_outcome_cid(
                    {
                        "operation_id": "scg047-promo-stale",
                        "kind": "promote_stale_candidate",
                        "status": stale_promo.status,
                        "head_mutated": stale_promo.head_mutated,
                        "blocking_reasons": list(stale_promo.blocking_reasons),
                    }
                ),
            },
            {
                "operation_id": "scg047-promo-mismatch",
                "kind": "promote_mismatched_evaluation",
                "status": mismatch_promo.status,
                "head_mutated": mismatch_promo.head_mutated,
                "blocking_reasons": list(mismatch_promo.blocking_reasons),
                "structural_outcome_cid": _structural_outcome_cid(
                    {
                        "operation_id": "scg047-promo-mismatch",
                        "kind": "promote_mismatched_evaluation",
                        "status": mismatch_promo.status,
                        "head_mutated": mismatch_promo.head_mutated,
                        "blocking_reasons": list(mismatch_promo.blocking_reasons),
                    }
                ),
            },
        ],
        "acceptance": {
            "rollback_reproducible": True,
            "history_preserved_on_rollback": True,
            "stale_candidate_fails": True,
            "mismatched_evaluation_fails": True,
            "rollback_is_forward_cas_not_history_deletion": True,
        },
        "notes": (
            "Rollback is another authorized expected-generation CAS. History "
            "grows with each transition; prior policy CIDs remain in the "
            "transition log. Stale and mismatched candidates leave the live "
            "head unchanged."
        ),
    }
    artifact["content_id"] = _content_id(artifact)
    return artifact


# ---------------------------------------------------------------------------
# Acceptance: seal scope is precise
# ---------------------------------------------------------------------------


def test_seal_scope_is_precise_on_unavailable_and_released_paths() -> None:
    evaluation = _pass_evaluation()

    # Authorized path while sealer unavailable: no ZK, closed claims only.
    authorized = _authorized_release_qualification(evaluation)
    assert authorized.promotion_allowed is True
    assert authorized.sealer_available is False
    assert authorized.seal_status == SealStatus.UNAVAILABLE.value
    seal = seal_governor_run(evaluation, qualification=authorized)
    assert seal.sealer_available is False
    assert seal.is_zk is False
    assert set(seal.claims) == set(DEFAULT_BOUNDED_CLAIMS)
    assert set(seal.claims).isdisjoint(FORBIDDEN_CLAIM_KINDS)
    payload = seal.identity_payload()
    assert "semantic_sufficiency" in payload["non_claims"]
    assert "zk_proof" in payload["non_claims"]
    # Every binding carries evaluated policy / evaluation identities.
    for binding in seal.bindings:
        assert binding.policy_cid == evaluation["baseline_policy_cid"]
        assert binding.evaluation_report_cid == evaluation["report_cid"]
    assert verify_governor_seal(seal) == seal.seal_cid

    # Injected released sealer: available + may claim ZK, still closed claims.
    surface = _released_sealer_surface()
    released = _incremental_seal_qualification(evaluation)
    assert released.path == QualificationPath.INCREMENTAL_SEAL.value
    released_seal = seal_governor_run(
        evaluation,
        qualification=released,
        sealer_surface=surface,
    )
    assert released_seal.sealer_available is True
    assert released_seal.seal_status == SealStatus.AVAILABLE.value
    assert released_seal.is_zk is True
    assert set(released_seal.claims) == set(DEFAULT_BOUNDED_CLAIMS)
    roles = {b.role for b in released_seal.bindings}
    assert ARTIFACT_ROLE_INCREMENTAL_SEAL in roles
    assert released_seal.incremental_seal_cid is not None
    assert verify_governor_seal(released_seal) == released_seal.seal_cid


def test_overclaim_and_unknown_claims_rejected() -> None:
    evaluation = _pass_evaluation()
    authorized = _authorized_release_qualification(evaluation)

    with pytest.raises(SealingError) as excinfo:
        seal_governor_run(
            evaluation,
            claims=("semantic_sufficiency",),
            qualification=authorized,
        )
    assert REASON_OVERCLAIM in str(excinfo.value) or REASON_UNAUTHORIZED_CLAIM in str(
        excinfo.value
    )

    with pytest.raises(SealingError) as excinfo2:
        qualify_policy_candidate(
            evaluation,
            claims=("not_a_real_claim",),
            release_qualification_authorization_cid=_auth_cid(),
            release_qualification_bundle=_verification_bundle(),
        )
    assert REASON_UNAUTHORIZED_CLAIM in str(excinfo2.value)


# ---------------------------------------------------------------------------
# Acceptance: stale / corrupt / mismatched candidate fails
# ---------------------------------------------------------------------------


def test_stale_corrupt_mismatched_candidates_fail_closed(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas, name="scg047-fail-policy-v1")
    proposed = _store_block(coordination, "scg047-fail-policy-v2")
    auth = _promotion_auth()

    # Stale base relative to live head.
    stale_base = _store_block(coordination, "scg047-fail-stale-base")
    stale_cand = _candidate(
        candidate_id="cand_stale",
        base_policy_cid=stale_base,
        proposed_policy_cid=proposed,
        proposal_cid=_cid("scg047-proposal-stale-base"),
    )
    stale_eval = _promo_evaluation(stale_cand)
    stale_result = promote_compression_policy(
        stale_cand,
        stale_eval,
        auth,
        release_qualification=_authorized_release_qualification(stale_eval),
        policy_repository=cas.policy,
        workspace=HERMETIC_WORKSPACE,
        operation_id="scg047-fail-stale",
        promoted_policy_version="1.1.0",
    )
    assert stale_result.head_mutated is False
    assert REASON_STALE_CANDIDATE in stale_result.blocking_reasons
    assert cas.current_policy(HERMETIC_WORKSPACE).policy_cid == base

    # Mismatched evaluation candidate identity.
    good_cand = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    other_cand = _candidate(
        candidate_id="cand_other",
        base_policy_cid=base,
        proposed_policy_cid=proposed,
        proposal_cid=_cid("scg047-proposal-other-id"),
    )
    mismatch_eval = _promo_evaluation(other_cand)
    mismatch_result = promote_compression_policy(
        good_cand,
        mismatch_eval,
        auth,
        release_qualification=_authorized_release_qualification(mismatch_eval),
        policy_repository=cas.policy,
        workspace=HERMETIC_WORKSPACE,
        operation_id="scg047-fail-mismatch",
        promoted_policy_version="1.1.0",
    )
    assert mismatch_result.head_mutated is False
    assert REASON_MISMATCHED_EVALUATION in mismatch_result.blocking_reasons
    assert cas.current_policy(HERMETIC_WORKSPACE).policy_cid == base

    # Corrupt / tampered seal fails verification.
    evaluation = _pass_evaluation()
    seal = seal_governor_run(
        evaluation,
        qualification=_authorized_release_qualification(evaluation),
    )
    corrupt = seal.to_dict()
    corrupt["seal_cid"] = _cid("scg047-corrupt-seal")
    with pytest.raises(SealingError) as excinfo:
        verify_governor_seal(corrupt)
    assert REASON_STALE_SEAL in str(excinfo.value) or "stale" in str(excinfo.value).lower()

    # Qualification / evaluation identity mismatch fails seal.
    other_eval = _pass_evaluation(candidate_cid=_cid("scg047-mismatch-cand"))
    with pytest.raises(SealingError) as excinfo2:
        seal_governor_run(
            other_eval,
            qualification=_authorized_release_qualification(evaluation),
        )
    assert REASON_IDENTITY_MISMATCH in str(excinfo2.value)

    # Binding policy mismatch fails closed.
    with pytest.raises(SealingError):
        seal_governor_run(
            evaluation,
            qualification=_authorized_release_qualification(evaluation),
            bindings=(
                SealArtifactBinding(
                    role=ARTIFACT_ROLE_CONTEXT_PACK,
                    artifact_cid=_cid("scg047-ctx-bad"),
                    policy_cid=_cid("scg047-wrong-policy"),
                    evaluation_report_cid=evaluation["report_cid"],
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Acceptance: rollback is reproducible
# ---------------------------------------------------------------------------


def test_rollback_is_reproducible_in_hermetic_namespace(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas, name="scg047-repro-v1")
    proposed = _store_block(coordination, "scg047-repro-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _promo_evaluation(candidate)
    qualification = _authorized_release_qualification(evaluation)
    auth = _promotion_auth()

    outcomes: list[dict[str, Any]] = []
    for cycle in (1, 2):
        promo = promote_compression_policy(
            candidate,
            evaluation,
            auth,
            release_qualification=qualification,
            policy_repository=cas.policy,
            workspace=HERMETIC_WORKSPACE,
            operation_id=f"scg047-repro-promo-{cycle}",
            promoted_policy_version="1.1.0",
        )
        assert promo.head_mutated is True
        assert cas.current_policy(HERMETIC_WORKSPACE).policy_cid == proposed

        rolled = rollback_compression_policy(
            auth,
            target_policy_cid=base,
            policy_repository=cas.policy,
            workspace=HERMETIC_WORKSPACE,
            operation_id=f"scg047-repro-rollback-{cycle}",
            current_policy_version="1.1.0",
            target_policy_version="1.0.0",
        )
        assert rolled.status == RollbackStatus.ROLLED_BACK.value
        assert rolled.head_mutated is True
        assert cas.current_policy(HERMETIC_WORKSPACE).policy_cid == base
        outcomes.append(
            {
                "promo_status": promo.status,
                "rollback_status": rolled.status,
                "target_policy_cid": rolled.target_policy_cid,
                "head_after": cas.current_policy(HERMETIC_WORKSPACE).policy_cid,
            }
        )

    # Both cycles produce the same structural outcome.
    assert outcomes[0]["promo_status"] == outcomes[1]["promo_status"]
    assert outcomes[0]["rollback_status"] == outcomes[1]["rollback_status"]
    assert outcomes[0]["target_policy_cid"] == outcomes[1]["target_policy_cid"]
    assert outcomes[0]["head_after"] == outcomes[1]["head_after"] == base

    # History is append-only: seed + 2 promo + 2 rollback = 5 transitions.
    transitions = cas.policy_transitions(HERMETIC_WORKSPACE)
    assert len(transitions) == 5
    published = {row["new_root_cid"] for row in transitions}
    assert base in published
    assert proposed in published


# ---------------------------------------------------------------------------
# Acceptance: unavailable sealer never stalls unrelated qualification
# ---------------------------------------------------------------------------


def test_unavailable_sealer_never_stalls_unrelated_governor_qualification() -> None:
    live = probe_incremental_sealer_capability()
    assert live.available is False
    assert live.seal_status == SealStatus.UNAVAILABLE.value
    assert live.can_be_satisfied_by_ivp_commitment is False
    assert live.is_zk is False

    # Runtime adapters load without requiring the sealer.
    adapters = load_runtime_adapters(require_sealer=False)
    assert adapters.sealer.available is False
    # Non-seal execution surfaces are not gated on sealer availability.
    adapters.require_execution_surfaces()

    adapter = load_seal_adapter()
    assert adapter.sealer_is_available() is False

    evaluation = _pass_evaluation()
    # Unrelated (authorized release) qualification completes while sealer is down.
    authorized = adapter.qualify_policy_candidate(
        evaluation,
        release_qualification_authorization_cid=_auth_cid(),
        release_qualification_bundle=_verification_bundle(),
    )
    assert authorized.promotion_allowed is True
    assert (
        authorized.path
        == QualificationPath.AUTHORIZED_RELEASE_QUALIFICATION.value
    )
    assert authorized.sealer_available is False
    assert authorized.seal_status == SealStatus.UNAVAILABLE.value
    authorized.require_promotion_allowed()

    seal = adapter.seal_governor_run(evaluation, qualification=authorized)
    assert seal.seal_status == SealStatus.UNAVAILABLE.value
    assert seal.is_zk is False
    assert adapter.verify_governor_seal(seal) == seal.seal_cid

    # Without independent auth, promotion stays blocked (fail closed), but the
    # call returns promptly — it does not hang waiting on a sealer.
    blocked = adapter.qualify_policy_candidate(evaluation)
    assert blocked.promotion_allowed is False
    assert blocked.path == QualificationPath.BLOCKED.value
    assert blocked.seal_status == SealStatus.UNAVAILABLE.value
    assert REASON_PROMOTION_BLOCKED in blocked.blocking_reasons


def test_ivp_commitment_never_substitutes_for_released_sealer() -> None:
    evaluation = _pass_evaluation()
    result = qualify_policy_candidate(
        evaluation,
        sealer_surface=_released_sealer_surface(),
        incremental_seal_evidence={
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "verification-commitment@1"
            ),
            "kind": "verification_commitment",
        },
    )
    assert result.promotion_allowed is False
    assert REASON_IVP_COMMITMENT_NOT_SEALER in result.blocking_reasons
    assert result.sealer_capability["can_be_satisfied_by_ivp_commitment"] is False


# ---------------------------------------------------------------------------
# Artifact emission + on-disk validation
# ---------------------------------------------------------------------------


def test_emit_and_validate_seal_qualification_artifact() -> None:
    artifact = build_seal_qualification_artifact()
    written = _write_json_atomic(SEAL_QUALIFICATION_PATH, artifact)
    on_disk = _read_json(SEAL_QUALIFICATION_PATH)

    assert on_disk == written
    assert on_disk["schema"] == SEAL_QUALIFICATION_SCHEMA
    assert on_disk["interface_id"] == SEAL_QUALIFICATION_INTERFACE
    assert on_disk["evidence"] == SCG_INCREMENTAL_SEAL_QUALIFICATION_EVIDENCE
    assert on_disk["task_id"] == TASK_ID
    assert on_disk["goal_id"] == GOAL_ID
    assert on_disk["authoritative"] is False
    assert on_disk["current_tree"] is True
    assert on_disk["proof_scope_precise"] is True
    assert on_disk["sealer"]["available"] is False
    assert on_disk["sealer"]["can_be_satisfied_by_ivp_commitment"] is False
    assert on_disk["sealer"]["is_zk"] is False
    assert on_disk["seal_scope"]["unavailable"] is True
    assert on_disk["seal_scope"]["is_zk"] is False
    assert set(on_disk["seal_scope"]["bounded_claims"]) == set(DEFAULT_BOUNDED_CLAIMS)
    assert on_disk["acceptance"]["seal_scope_precise"] is True
    assert on_disk["acceptance"]["stale_corrupt_mismatched_candidate_fails"] is True
    assert on_disk["acceptance"][
        "unavailable_sealer_never_stalls_unrelated_qualification"
    ] is True
    assert on_disk["content_id"] == _content_id(on_disk)
    assert "released_IncrementalProofSealer_public_api" in on_disk["missing_evidence"]


def test_emit_and_validate_rollback_artifact(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    artifact = build_rollback_artifact(coordination=coordination, cas=cas)
    written = _write_json_atomic(ROLLBACK_PATH, artifact)
    on_disk = _read_json(ROLLBACK_PATH)

    assert on_disk == written
    assert on_disk["schema"] == ROLLBACK_QUALIFICATION_SCHEMA
    assert on_disk["interface_id"] == ROLLBACK_QUALIFICATION_INTERFACE
    assert on_disk["evidence"] == SCG_ROLLBACK_EVIDENCE
    assert on_disk["task_id"] == TASK_ID
    assert on_disk["goal_id"] == GOAL_ID
    assert on_disk["rollback_tested"] is True
    assert on_disk["reproducible"] is True
    assert on_disk["history_preserved"] is True
    assert on_disk["acceptance"]["rollback_reproducible"] is True
    assert on_disk["acceptance"]["stale_candidate_fails"] is True
    assert on_disk["acceptance"]["mismatched_evaluation_fails"] is True
    assert on_disk["content_id"] == _content_id(on_disk)

    kinds = {op["kind"] for op in on_disk["operations"]}
    assert "promote" in kinds
    assert "rollback" in kinds
    assert "promote_stale_candidate" in kinds
    assert "promote_mismatched_evaluation" in kinds
    for op in on_disk["operations"]:
        assert op.get("structural_outcome_cid")
        assert str(op["structural_outcome_cid"]).startswith("baguqeera")

    rollbacks = [op for op in on_disk["operations"] if op["kind"] == "rollback"]
    assert len(rollbacks) == 2
    assert rollbacks[0]["status"] == rollbacks[1]["status"] == RollbackStatus.ROLLED_BACK.value
    assert rollbacks[0]["head_mutated"] is True
    assert rollbacks[1]["head_mutated"] is True
    # Receipt identity and structural outcomes are hermetic-store stable.
    assert rollbacks[0]["receipt"]["receipt_id"]
    assert rollbacks[1]["receipt"]["receipt_id"]


def test_qualification_artifacts_exist_and_are_self_consistent() -> None:
    """Guard: both expected outputs are present after emission tests.

    Emission tests write the artifacts; this check re-validates durable shape
    without requiring a live CAS store so collection-order does not matter
    when re-run after a green suite.
    """

    # Ensure artifacts exist (emit seal qualification if suite order skipped).
    if not SEAL_QUALIFICATION_PATH.is_file():
        _write_json_atomic(SEAL_QUALIFICATION_PATH, build_seal_qualification_artifact())
    seal_qual = _read_json(SEAL_QUALIFICATION_PATH)
    assert seal_qual["evidence"] == SCG_INCREMENTAL_SEAL_QUALIFICATION_EVIDENCE
    assert seal_qual["content_id"] == _content_id(seal_qual)
    assert seal_qual["sealer"]["available"] is False

    # Rollback artifact requires hermetic store; only validate if present.
    if ROLLBACK_PATH.is_file():
        rollback = _read_json(ROLLBACK_PATH)
        assert rollback["evidence"] == SCG_ROLLBACK_EVIDENCE
        assert rollback["content_id"] == _content_id(rollback)
        assert rollback["rollback_tested"] is True
        assert rollback["reproducible"] is True
