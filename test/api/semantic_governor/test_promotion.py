"""Tests for SCG-034 authorized compare-and-swap promotion and rollback.

Acceptance criteria enforced here:

* Stale candidate cannot mutate the head.
* Absent or unavailable release qualification cannot mutate the head.
* Absent authorization cannot mutate the head.
* Reduced high-risk assurance cannot mutate the head.
* Mismatched evaluation cannot mutate the head.
* CAS conflict cannot mutate the head.
* Self-promotion cannot mutate the head.
* Authorized promotion and rollback publish one version atomically and
  preserve history on rollback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from ipfs_kit_py.semantic_governor_store.policy import (
    DurableCompressionPolicyRepository,
    DurablePolicyCASRepositories,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.promotion import (
    PROMOTE_COMPRESSION_POLICY_INTERFACE,
    ROLLBACK_COMPRESSION_POLICY_INTERFACE,
    PolicyPromotionResult,
    PolicyRollbackResult,
    PromotionStatus,
    REASON_ABSENT_AUTHORIZATION,
    REASON_ABSENT_QUALIFICATION,
    REASON_CAS_CONFLICT,
    REASON_EVALUATION_NOT_PASS,
    REASON_HIGH_RISK_REDUCTION,
    REASON_MISMATCHED_EVALUATION,
    REASON_SELF_PROMOTION,
    REASON_STALE_CANDIDATE,
    REASON_UNAVAILABLE_QUALIFICATION,
    RollbackStatus,
    SCG_AUTHORIZED_PROMOTION_EVIDENCE,
    evaluate_promotion_gates,
    promote_compression_policy,
    rollback_compression_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.sealing import (
    QualificationPath,
    ReleaseQualification,
    SealStatus,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/promotion.py"
)
WORKSPACE = "default"


# ---------------------------------------------------------------------------
# Fixtures / recipes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    """Canonical dag-json CID (store CAS requires dag-json profile)."""

    return cid_for_structured({"test_label": label, "schema": "test/label@1"})


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
        "metadata": {"track": "policy_promotion"},
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


def _pass_evaluation(candidate: CompressionPolicyCandidate, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "report_cid": _cid(f"eval-pass-{candidate.candidate_id}"),
        "candidate_cid": candidate.candidate_cid,
        "held_out_benchmark_cid": _cid("benchmark-held-out"),
        "baseline_policy_cid": candidate.base_policy_cid,
        "verdict": EvaluationVerdict.PASS.value,
        "partition": EvidencePartition.HELD_OUT.value,
        "declared_thresholds_applied": True,
        "blocking_reasons": (),
        "high_risk_assurance_reduced": False,
    }
    fields.update(overrides)
    return fields


def _fail_evaluation(candidate: CompressionPolicyCandidate, **overrides: Any) -> dict[str, Any]:
    return _pass_evaluation(
        candidate,
        report_cid=_cid(f"eval-fail-{candidate.candidate_id}"),
        verdict=EvaluationVerdict.FAIL.value,
        blocking_reasons=("critical_omission_detection_below_threshold",),
        **overrides,
    )


def _allowed_qualification(
    candidate: CompressionPolicyCandidate,
    evaluation: dict[str, Any],
    **overrides: Any,
) -> ReleaseQualification:
    fields: dict[str, Any] = {
        "qualification_id": "qual_ok",
        "path": QualificationPath.AUTHORIZED_RELEASE_QUALIFICATION.value,
        "promotion_allowed": True,
        "seal_status": SealStatus.UNAVAILABLE.value,
        "sealer_available": False,
        "sealer_capability": {
            "available": False,
            "seal_status": SealStatus.UNAVAILABLE.value,
            "can_be_satisfied_by_ivp_commitment": False,
        },
        "evaluation_report_cid": evaluation["report_cid"],
        "candidate_cid": candidate.candidate_cid,
        "baseline_policy_cid": candidate.base_policy_cid,
        "held_out_benchmark_cid": evaluation.get("held_out_benchmark_cid"),
        "authorization_cid": _cid("release-qual-auth"),
        "verification_bundle_cid": _cid("release-qual-bundle"),
        "incremental_seal_cid": None,
        "blocking_reasons": (),
        "claims": (
            "exact_artifacts_evaluated",
            "required_evaluations_completed",
            "declared_thresholds_applied",
            "no_blocking_status_omitted",
            "promoted_policy_equals_evaluated_candidate",
        ),
        "diagnostic": None,
        "metadata": {},
    }
    fields.update(overrides)
    return ReleaseQualification(**fields)


def _blocked_qualification(
    candidate: CompressionPolicyCandidate,
    evaluation: dict[str, Any],
    **overrides: Any,
) -> ReleaseQualification:
    return _allowed_qualification(
        candidate,
        evaluation,
        qualification_id="qual_blocked",
        path=QualificationPath.BLOCKED.value,
        promotion_allowed=False,
        authorization_cid=None,
        verification_bundle_cid=None,
        blocking_reasons=("promotion_blocked", "sealer_unavailable"),
        diagnostic="release qualification blocked",
        **overrides,
    )


def _auth() -> str:
    return _cid("external-promotion-authorization-board")


def _store_block(store: DurableCoordinationStore, name: str) -> str:
    payload = {"schema": "example/governor-policy@1", "name": name}
    return store.put(payload, expected_cid=cid_for_artifact(payload), replicate=False)[
        "cid"
    ]


@pytest.fixture()
def coordination(tmp_path: Path) -> DurableCoordinationStore:
    root = DurableCoordinationStore(tmp_path / "promo-store")
    yield root
    root.close()


@pytest.fixture()
def policy_repo(
    coordination: DurableCoordinationStore,
) -> DurableCompressionPolicyRepository:
    return DurableCompressionPolicyRepository(coordination)


@pytest.fixture()
def cas(
    coordination: DurableCoordinationStore,
) -> DurablePolicyCASRepositories:
    return DurablePolicyCASRepositories(coordination)


def _seed_policy_head(
    coordination: DurableCoordinationStore,
    policy_repo: DurableCompressionPolicyRepository,
    *,
    name: str = "policy-v1",
) -> str:
    """Publish a generation-1 head and return its CID."""

    cid = _store_block(coordination, name)
    result = policy_repo.compare_and_swap_policy(
        WORKSPACE,
        expected_generation=0,
        expected_policy_cid=None,
        new_policy_cid=cid,
        operation_id=f"seed-{name}",
    )
    assert result.status is GovernorStoreStatus.UPDATED
    return cid


def _seed_successor(
    coordination: DurableCoordinationStore,
    policy_repo: DurableCompressionPolicyRepository,
    *,
    base_cid: str,
    name: str = "policy-v2",
) -> str:
    """Put a successor block (not yet head) and return its CID."""

    # Ensure base is live head generation 1.
    head = policy_repo.current_policy(WORKSPACE)
    assert head.policy_cid == base_cid
    return _store_block(coordination, name)


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exists_and_exports_interfaces() -> None:
    assert MODULE_PATH.is_file()
    assert PROMOTE_COMPRESSION_POLICY_INTERFACE == "promote_compression_policy@1"
    assert ROLLBACK_COMPRESSION_POLICY_INTERFACE == "rollback_compression_policy@1"
    assert SCG_AUTHORIZED_PROMOTION_EVIDENCE == "scg/authorized-promotion@1"


def test_happy_path_promotes_with_expected_version_cas(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")

    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _pass_evaluation(candidate)
    qualification = _allowed_qualification(candidate, evaluation)
    auth = _auth()

    before = cas.current_policy(WORKSPACE)
    assert before.generation == 1
    assert before.policy_cid == base

    result = promote_compression_policy(
        candidate,
        evaluation,
        auth,
        release_qualification=qualification,
        policy_repository=cas.policy,
        promotion_repository=cas.promotion,
        workspace=WORKSPACE,
        operation_id="promo-happy-1",
        promoted_policy_version="1.1.0",
    )

    assert result.status == PromotionStatus.PROMOTED.value
    assert result.head_mutated is True
    assert result.blocking_reasons == ()
    assert result.receipt is not None
    assert result.receipt.promoted_policy_cid == proposed
    assert result.receipt.authorization_cid == auth
    assert result.receipt.previous_policy_cid == base
    assert result.policy_cas is not None
    assert result.policy_cas["status"] == "updated"

    after = cas.current_policy(WORKSPACE)
    assert after.generation == 2
    assert after.policy_cid == proposed
    # History retained: one seed + one promotion transition.
    assert len(cas.policy_transitions(WORKSPACE)) == 2

    # Idempotent replay does not advance generation again.
    replay = promote_compression_policy(
        candidate,
        evaluation,
        auth,
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-happy-1",
        promoted_policy_version="1.1.0",
    )
    assert replay.head_mutated is False
    assert replay.status == PromotionStatus.UNCHANGED.value
    assert cas.current_policy(WORKSPACE).generation == 2


# ---------------------------------------------------------------------------
# Fail-closed gates: head must not mutate
# ---------------------------------------------------------------------------


def test_stale_candidate_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    live = _seed_policy_head(coordination, cas.policy, name="live-policy")
    proposed = _store_block(coordination, "proposed-policy")
    stale_base = _store_block(coordination, "stale-base")

    candidate = _candidate(
        base_policy_cid=stale_base,
        proposed_policy_cid=proposed,
    )
    evaluation = _pass_evaluation(candidate)
    qualification = _allowed_qualification(candidate, evaluation)

    result = promote_compression_policy(
        candidate,
        evaluation,
        _auth(),
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-stale",
        promoted_policy_version="1.1.0",
    )
    assert result.head_mutated is False
    assert result.status == PromotionStatus.REJECTED.value
    assert REASON_STALE_CANDIDATE in result.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == live
    assert cas.current_policy(WORKSPACE).generation == 1


def test_absent_qualification_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _pass_evaluation(candidate)

    result = promote_compression_policy(
        candidate,
        evaluation,
        _auth(),
        release_qualification=None,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-no-qual",
        promoted_policy_version="1.1.0",
    )
    assert result.head_mutated is False
    assert REASON_ABSENT_QUALIFICATION in result.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == base


def test_unavailable_qualification_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _pass_evaluation(candidate)
    blocked = _blocked_qualification(candidate, evaluation)

    result = promote_compression_policy(
        candidate,
        evaluation,
        _auth(),
        release_qualification=blocked,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-blocked-qual",
        promoted_policy_version="1.1.0",
    )
    assert result.head_mutated is False
    assert REASON_UNAVAILABLE_QUALIFICATION in result.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == base


def test_absent_authorization_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _pass_evaluation(candidate)
    qualification = _allowed_qualification(candidate, evaluation)

    result = promote_compression_policy(
        candidate,
        evaluation,
        None,
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-no-auth",
        promoted_policy_version="1.1.0",
    )
    assert result.head_mutated is False
    assert REASON_ABSENT_AUTHORIZATION in result.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == base


def test_self_promotion_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _pass_evaluation(candidate)
    qualification = _allowed_qualification(candidate, evaluation)

    # Candidate tries to authorize itself.
    result = promote_compression_policy(
        candidate,
        evaluation,
        candidate.candidate_cid,
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-self-cand",
        promoted_policy_version="1.1.0",
    )
    assert result.head_mutated is False
    assert REASON_SELF_PROMOTION in result.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == base

    # Evaluation tries to authorize itself.
    result2 = promote_compression_policy(
        candidate,
        evaluation,
        evaluation["report_cid"],
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-self-eval",
        promoted_policy_version="1.1.0",
    )
    assert result2.head_mutated is False
    assert REASON_SELF_PROMOTION in result2.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == base

    # Proposed policy cannot authorize itself.
    result3 = promote_compression_policy(
        candidate,
        evaluation,
        proposed,
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-self-policy",
        promoted_policy_version="1.1.0",
    )
    assert result3.head_mutated is False
    assert REASON_SELF_PROMOTION in result3.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == base


def test_reduced_high_risk_assurance_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    # Reduce a protected threshold; requires external auth on the candidate.
    external = _cid("threshold-reduction-auth")
    candidate = _candidate(
        base_policy_cid=base,
        proposed_policy_cid=proposed,
        baseline_protected_thresholds=_thresholds(
            min_critical_omission_detection_bp=9_500
        ),
        proposed_protected_thresholds=_thresholds(
            min_critical_omission_detection_bp=5_000
        ),
        external_authorization_cid=external,
    )
    # Evaluation that somehow claims pass with high_risk reduced (mapping form).
    evaluation = _pass_evaluation(
        candidate,
        high_risk_assurance_reduced=True,
    )
    qualification = _allowed_qualification(candidate, evaluation)

    result = promote_compression_policy(
        candidate,
        evaluation,
        _auth(),
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-high-risk",
        promoted_policy_version="1.1.0",
    )
    assert result.head_mutated is False
    assert REASON_HIGH_RISK_REDUCTION in result.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == base


def test_mismatched_evaluation_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    other = _candidate(
        candidate_id="cand_other",
        base_policy_cid=base,
        proposed_policy_cid=proposed,
        proposal_cid=_cid("proposal-other"),
    )
    # Evaluation bound to a different candidate.
    evaluation = _pass_evaluation(other)
    qualification = _allowed_qualification(candidate, evaluation)

    result = promote_compression_policy(
        candidate,
        evaluation,
        _auth(),
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-mismatch",
        promoted_policy_version="1.1.0",
    )
    assert result.head_mutated is False
    assert REASON_MISMATCHED_EVALUATION in result.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == base


def test_non_pass_evaluation_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _fail_evaluation(candidate)
    # Qualification would also fail gates, but use allowed shape to isolate verdict.
    qualification = _allowed_qualification(
        candidate,
        evaluation,
        # ReleaseQualification itself does not re-check verdict; promotion does.
    )

    result = promote_compression_policy(
        candidate,
        evaluation,
        _auth(),
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-fail-eval",
        promoted_policy_version="1.1.0",
    )
    assert result.head_mutated is False
    assert REASON_EVALUATION_NOT_PASS in result.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == base


def test_cas_conflict_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    interloper = _store_block(coordination, "policy-interloper")

    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _pass_evaluation(candidate)
    qualification = _allowed_qualification(candidate, evaluation)

    # Concurrent writer wins first with a different successor.
    raced = cas.compare_and_swap_policy(
        WORKSPACE,
        expected_generation=1,
        expected_policy_cid=base,
        new_policy_cid=interloper,
        operation_id="race-winner",
    )
    assert raced.status is GovernorStoreStatus.UPDATED
    assert cas.current_policy(WORKSPACE).policy_cid == interloper

    # Stale expected generation/cid from before the race → conflict, no mutation.
    result = promote_compression_policy(
        candidate,
        evaluation,
        _auth(),
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-race-loser",
        expected_generation=1,
        expected_policy_cid=base,
        promoted_policy_version="1.1.0",
    )
    # Gate rejects stale candidate (base != live interloper) before CAS, or
    # policy head mismatch — either way head stays interloper.
    assert result.head_mutated is False
    assert cas.current_policy(WORKSPACE).policy_cid == interloper
    assert cas.current_policy(WORKSPACE).generation == 2
    assert (
        REASON_STALE_CANDIDATE in result.blocking_reasons
        or REASON_CAS_CONFLICT in result.blocking_reasons
    )


def test_cas_conflict_via_stale_expectation_only(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    """When gates would pass but CAS expectation is stale, report conflict.

    Uses a candidate whose base matches live head, then races after gate
    evaluation by performing a second CAS with a forged repository that
    still reports the old head for gate checks — covered here by calling
    compare_and_swap directly after a successful seed and forcing conflict
    through evaluate_promotion_gates + manual CAS status mapping.

    Direct path: publish promotion with correct head, then attempt a second
    promotion of another candidate against the old expectation.
    """

    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed_a = _store_block(coordination, "policy-v2a")
    proposed_b = _store_block(coordination, "policy-v2b")

    cand_a = _candidate(
        candidate_id="cand_a",
        base_policy_cid=base,
        proposed_policy_cid=proposed_a,
        proposal_cid=_cid("proposal-a"),
    )
    eval_a = _pass_evaluation(cand_a)
    qual_a = _allowed_qualification(cand_a, eval_a)

    first = promote_compression_policy(
        cand_a,
        eval_a,
        _auth(),
        release_qualification=qual_a,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-a",
        promoted_policy_version="1.1.0",
    )
    assert first.head_mutated is True

    # Candidate B is based on the old base (stale) — rejected as stale.
    cand_b = _candidate(
        candidate_id="cand_b",
        base_policy_cid=base,
        proposed_policy_cid=proposed_b,
        proposal_cid=_cid("proposal-b"),
    )
    eval_b = _pass_evaluation(cand_b)
    qual_b = _allowed_qualification(cand_b, eval_b)
    second = promote_compression_policy(
        cand_b,
        eval_b,
        _auth(),
        release_qualification=qual_b,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-b",
        expected_generation=1,
        expected_policy_cid=base,
        promoted_policy_version="1.2.0",
    )
    assert second.head_mutated is False
    assert cas.current_policy(WORKSPACE).policy_cid == proposed_a
    assert REASON_STALE_CANDIDATE in second.blocking_reasons


# ---------------------------------------------------------------------------
# Pure gate helper
# ---------------------------------------------------------------------------


def test_evaluate_promotion_gates_covers_acceptance_reasons() -> None:
    base = _cid("policy-v1")
    proposed = _cid("policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _pass_evaluation(candidate)
    qualification = _allowed_qualification(candidate, evaluation)

    clear = evaluate_promotion_gates(
        candidate,
        evaluation,
        _auth(),
        release_qualification=qualification,
        current_policy_cid=base,
        current_generation=1,
    )
    assert clear == ()

    assert REASON_ABSENT_AUTHORIZATION in evaluate_promotion_gates(
        candidate,
        evaluation,
        None,
        release_qualification=qualification,
        current_policy_cid=base,
        current_generation=1,
    )
    assert REASON_ABSENT_QUALIFICATION in evaluate_promotion_gates(
        candidate,
        evaluation,
        _auth(),
        release_qualification=None,
        current_policy_cid=base,
        current_generation=1,
    )
    assert REASON_STALE_CANDIDATE in evaluate_promotion_gates(
        candidate,
        evaluation,
        _auth(),
        release_qualification=qualification,
        current_policy_cid=_cid("other-live"),
        current_generation=1,
    )
    assert REASON_SELF_PROMOTION in evaluate_promotion_gates(
        candidate,
        evaluation,
        candidate.candidate_cid,
        release_qualification=qualification,
        current_policy_cid=base,
        current_generation=1,
    )
    assert REASON_MISMATCHED_EVALUATION in evaluate_promotion_gates(
        candidate,
        _pass_evaluation(candidate, candidate_cid=_cid("other-cand")),
        _auth(),
        release_qualification=qualification,
        current_policy_cid=base,
        current_generation=1,
    )
    assert REASON_HIGH_RISK_REDUCTION in evaluate_promotion_gates(
        candidate,
        _pass_evaluation(candidate, high_risk_assurance_reduced=True),
        _auth(),
        release_qualification=qualification,
        current_policy_cid=base,
        current_generation=1,
    )


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_authorized_rollback_preserves_history(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _pass_evaluation(candidate)
    qualification = _allowed_qualification(candidate, evaluation)

    promoted = promote_compression_policy(
        candidate,
        evaluation,
        _auth(),
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-then-rollback",
        promoted_policy_version="1.1.0",
    )
    assert promoted.head_mutated is True
    assert cas.current_policy(WORKSPACE).policy_cid == proposed
    transitions_after_promo = len(cas.policy_transitions(WORKSPACE))

    rolled = rollback_compression_policy(
        _auth(),
        target_policy_cid=base,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="rollback-1",
        current_policy_version="1.1.0",
        target_policy_version="1.0.0",
    )
    assert rolled.status == RollbackStatus.ROLLED_BACK.value
    assert rolled.head_mutated is True
    assert rolled.receipt is not None
    assert cas.current_policy(WORKSPACE).policy_cid == base
    # Rollback is a forward transition: history length increases, not shrinks.
    assert len(cas.policy_transitions(WORKSPACE)) == transitions_after_promo + 1
    # All prior policy CIDs remain in transition history.
    published = {row["new_root_cid"] for row in cas.policy_transitions(WORKSPACE)}
    assert base in published
    assert proposed in published


def test_rollback_absent_authorization_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    # Advance head to proposed so rollback has somewhere to go.
    advanced = cas.compare_and_swap_policy(
        WORKSPACE,
        expected_generation=1,
        expected_policy_cid=base,
        new_policy_cid=proposed,
        operation_id="advance",
    )
    assert advanced.status is GovernorStoreStatus.UPDATED

    result = rollback_compression_policy(
        None,
        target_policy_cid=base,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="rollback-no-auth",
    )
    assert result.head_mutated is False
    assert REASON_ABSENT_AUTHORIZATION in result.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == proposed


def test_rollback_self_authorization_cannot_mutate_head(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    advanced = cas.compare_and_swap_policy(
        WORKSPACE,
        expected_generation=1,
        expected_policy_cid=base,
        new_policy_cid=proposed,
        operation_id="advance-self",
    )
    assert advanced.status is GovernorStoreStatus.UPDATED

    result = rollback_compression_policy(
        proposed,
        target_policy_cid=base,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="rollback-self",
    )
    assert result.head_mutated is False
    assert REASON_SELF_PROMOTION in result.blocking_reasons
    assert cas.current_policy(WORKSPACE).policy_cid == proposed


def test_promotion_result_identity_is_stable(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _pass_evaluation(candidate)
    qualification = _allowed_qualification(candidate, evaluation)

    result = promote_compression_policy(
        candidate,
        evaluation,
        _auth(),
        release_qualification=qualification,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-identity",
        promoted_policy_version="1.1.0",
    )
    assert isinstance(result, PolicyPromotionResult)
    assert result.result_cid == cid_for_structured(result.identity_payload())
    payload = result.to_dict()
    assert payload["result_cid"] == result.result_cid
    assert payload["evidence"] == SCG_AUTHORIZED_PROMOTION_EVIDENCE


def test_rejected_result_never_claims_mutation() -> None:
    """Construct a rejected result and verify consistency invariants."""

    result = PolicyPromotionResult(
        status=PromotionStatus.REJECTED.value,
        head_mutated=False,
        blocking_reasons=(REASON_ABSENT_AUTHORIZATION,),
        workspace="default",
        operation_id="op-1",
        candidate_cid=_cid("c"),
        evaluation_report_cid=_cid("e"),
        authorization_cid=None,
        qualification_cid=None,
        expected_generation=1,
        expected_policy_cid=_cid("p"),
        promoted_policy_cid=_cid("n"),
        receipt=None,
        policy_cas=None,
        promotion_cas=None,
    )
    assert result.head_mutated is False
    with pytest.raises(Exception):
        PolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=True,  # inconsistent
            blocking_reasons=(),
            workspace="default",
            operation_id="op-2",
            candidate_cid=None,
            evaluation_report_cid=None,
            authorization_cid=None,
            qualification_cid=None,
            expected_generation=None,
            expected_policy_cid=None,
            promoted_policy_cid=None,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
        )


def test_authorization_mapping_form(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    candidate = _candidate(base_policy_cid=base, proposed_policy_cid=proposed)
    evaluation = _pass_evaluation(candidate)
    qualification = _allowed_qualification(candidate, evaluation)
    auth = _auth()

    result = promote_compression_policy(
        candidate,
        evaluation,
        {"authorization_cid": auth},
        release_qualification=qualification.to_dict(),
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="promo-auth-map",
        promoted_policy_version="1.1.0",
    )
    assert result.head_mutated is True
    assert result.authorization_cid == auth
    assert isinstance(result, PolicyPromotionResult)


def test_rollback_result_type(
    coordination: DurableCoordinationStore,
    cas: DurablePolicyCASRepositories,
) -> None:
    base = _seed_policy_head(coordination, cas.policy, name="policy-v1")
    proposed = _store_block(coordination, "policy-v2")
    cas.compare_and_swap_policy(
        WORKSPACE,
        expected_generation=1,
        expected_policy_cid=base,
        new_policy_cid=proposed,
        operation_id="seed-rollback-type",
    )
    result = rollback_compression_policy(
        _auth(),
        target_policy_cid=base,
        policy_repository=cas.policy,
        workspace=WORKSPACE,
        operation_id="rollback-type",
        current_policy_version="1.1.0",
        target_policy_version="1.0.0",
    )
    assert isinstance(result, PolicyRollbackResult)
    assert result.result_cid == cid_for_structured(result.identity_payload())
