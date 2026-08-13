"""IPS-035: execute plans with fresh cache verification and admission."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
    EvidenceCandidate,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
    CACHE_REVERIFY_EVIDENCE,
    EVIDENCE_SUBSET,
    CachedCandidate,
    ExecutionOutcome,
    ExecutionReasonCode,
    FreshProof,
    IncrementalPlanExecutor,
    ResourcePolicy,
    execute_incremental_plan,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    ParentSealContext,
    PlanMode,
    UnitPlanKind,
    UnitPlanningInput,
    create_incremental_plan,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.process_control import (
    CancellationToken,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
)

_PARENT = ParentSealContext(
    seal_cid="sha256:" + ("aa" * 32),
    repository_state_cid="sha256:" + ("bb" * 32),
    source_root_cid="sha256:" + ("cc" * 32),
)
_OLD = "sha256:" + ("bb" * 32)
_NEW = "sha256:" + ("dd" * 32)


def _unit(unit_id: str, **overrides: object) -> UnitPlanningInput:
    payload = {
        "unit_id": unit_id,
        "preserved": True,
        "cache_key_complete": True,
        "admitted": True,
        "candidate_present": True,
    }
    payload.update(overrides)
    return UnitPlanningInput(**payload)  # type: ignore[arg-type]


def _plan():
    return create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit("unit/reuse"),
            _unit(
                "unit/reprove",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
            _unit("unit/new", preserved=False, added=True, admitted=False),
            _unit("unit/gone", preserved=False, removed=True, admitted=False),
        ),
    )


def _good_candidate(unit_id: str, digest: str) -> CachedCandidate:
    cid = "sha256:" + ("11" * 32)
    return CachedCandidate(
        unit_id=unit_id,
        expected_digest=digest,
        observed_digest=digest,
        public_input_cid=cid,
        observed_public_input_cid=cid,
        proof_object_cid=cid,
        evidence=IntegrityCommitment(
            digest=digest,
            cid=cid,
            merkle_inclusion="leaf:0",
            byte_length=32,
        ),
    )


def test_evidence_subsets() -> None:
    assert EVIDENCE_SUBSET == "ips/incremental-execution@1"
    assert CACHE_REVERIFY_EVIDENCE == "ips/cache-reverification@1"


def test_reused_and_proved_sets_cover_plan_exactly() -> None:
    digest = "sha256:" + ("ab" * 32)
    store = {"unit/reuse": _good_candidate("unit/reuse", digest)}
    result = execute_incremental_plan(
        _plan(),
        ResourcePolicy(),
        fetch=store.get,
    )
    assert result.outcome is ExecutionOutcome.COMPLETED
    assert result.succeeded is True
    assert result.may_aggregate is True
    assert result.complete_coverage is True
    assert result.mode is PlanMode.INCREMENTAL
    assert result.reused_unit_ids == ("unit/reuse",)
    assert set(result.newly_proved_unit_ids) == {"unit/reprove", "unit/new"}
    assert result.tombstoned_unit_ids == ("unit/gone",)
    assert result.rejected_unit_ids == ()
    covered = (
        set(result.reused_unit_ids)
        | set(result.newly_proved_unit_ids)
        | set(result.tombstoned_unit_ids)
    )
    assert covered == set(result.required_unit_ids)
    assert result.admissions
    assert all(item.verified is True for item in result.admissions)


def test_stale_poisoned_corrupt_and_mismatched_candidates_are_rejected() -> None:
    digest = "sha256:" + ("ab" * 32)
    cid = "sha256:" + ("11" * 32)
    cases = {
        "stale": CachedCandidate(
            "unit/reuse", digest, digest, cid, cid, stale=True
        ),
        "poisoned": CachedCandidate(
            "unit/reuse", digest, digest, cid, cid, poisoned=True
        ),
        "corrupt": CachedCandidate(
            "unit/reuse", digest, digest, cid, cid, corrupt=True
        ),
        "mismatch": CachedCandidate(
            "unit/reuse", digest, "sha256:" + ("ff" * 32), cid, cid
        ),
    }
    expected = {
        "stale": ExecutionReasonCode.STALE_CANDIDATE,
        "poisoned": ExecutionReasonCode.POISONED_CANDIDATE,
        "corrupt": ExecutionReasonCode.CORRUPT_CANDIDATE,
        "mismatch": ExecutionReasonCode.DIGEST_MISMATCH,
    }
    for name, candidate in cases.items():
        result = execute_incremental_plan(_plan(), fetch=lambda uid, c=candidate: c)
        assert result.outcome is ExecutionOutcome.REJECTED, name
        assert result.may_aggregate is False, name
        assert result.succeeded is False, name
        assert "unit/reuse" in result.rejected_unit_ids, name
        assert expected[name].value in result.reason_codes, name


def test_simulated_evidence_cannot_be_reused_or_newly_proved() -> None:
    digest = "sha256:" + ("ab" * 32)
    cid = "sha256:" + ("11" * 32)
    simulated = CachedCandidate(
        "unit/reuse", digest, digest, cid, cid, simulated=True
    )
    reused = execute_incremental_plan(_plan(), fetch=lambda uid: simulated)
    assert reused.outcome is ExecutionOutcome.REJECTED
    assert ExecutionReasonCode.SIMULATED_FORBIDDEN.value in reused.reason_codes
    assert reused.may_aggregate is False

    def prove_simulated(unit):
        return FreshProof(
            unit.unit_id,
            EvidenceCandidate(
                evidence=IntegrityCommitment(
                    digest=digest,
                    cid=cid,
                    merkle_inclusion="leaf:0",
                    byte_length=32,
                ),
                proof_system_id="integrity",
                public_input_cid=cid,
                proof_unit_id=unit.unit_id,
                expected_digest=digest,
                observed_digest=digest,
                observed_public_input_cid=cid,
                proof_mode=ProofMode.INTEGRITY_ONLY,
                terminal_status=ProofTerminalStatus.INTEGRITY_VERIFIED,
            ),
            digest,
            simulated=True,
            status="simulated",
        )

    store = {"unit/reuse": _good_candidate("unit/reuse", digest)}
    proved = execute_incremental_plan(_plan(), fetch=store.get, prove=prove_simulated)
    assert proved.outcome is ExecutionOutcome.REJECTED
    assert ExecutionReasonCode.SIMULATED_FORBIDDEN.value in proved.reason_codes
    assert proved.may_aggregate is False


def test_missing_candidate_cannot_fast_path_reuse() -> None:
    result = execute_incremental_plan(_plan(), fetch=lambda uid: None)
    assert result.outcome is ExecutionOutcome.REJECTED
    assert ExecutionReasonCode.MISSING_CANDIDATE.value in result.reason_codes
    assert result.reused_unit_ids == ()
    assert result.may_aggregate is False


def test_prior_admission_record_is_not_a_cache_fast_path() -> None:
    digest = "sha256:" + ("ab" * 32)
    stale = _good_candidate("unit/reuse", digest)
    object.__setattr__(stale, "stale", True)
    result = execute_incremental_plan(_plan(), fetch=lambda uid: stale)
    assert result.outcome is ExecutionOutcome.REJECTED
    assert result.may_aggregate is False
    for record in result.units:
        payload = record.to_canonical()
        assert payload["cache_fast_path"] is False


def test_cancellation_and_unavailable_cannot_aggregate() -> None:
    token = CancellationToken()
    token.cancel("operator")
    cancelled = IncrementalPlanExecutor(token=token).execute(_plan())
    assert cancelled.outcome is ExecutionOutcome.CANCELLED
    assert cancelled.may_aggregate is False
    assert cancelled.succeeded is False

    unavailable = IncrementalPlanExecutor(backend_available=False).execute(_plan())
    assert unavailable.outcome is ExecutionOutcome.UNAVAILABLE
    assert unavailable.may_aggregate is False

    timed = IncrementalPlanExecutor(timed_out=lambda: True).execute(_plan())
    assert timed.outcome is ExecutionOutcome.TIMEOUT
    assert timed.may_aggregate is False


def test_reject_reuse_units_are_proved_not_cached() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit(
                "unit/hint-only",
                admitted=False,
                candidate_present=True,
            ),
        ),
    )
    assert plan.units[0].kind is UnitPlanKind.REJECT_REUSE
    result = execute_incremental_plan(plan, fetch=lambda uid: None)
    assert result.outcome is ExecutionOutcome.COMPLETED
    assert result.newly_proved_unit_ids == ("unit/hint-only",)
    assert result.reused_unit_ids == ()
    assert result.may_aggregate is True


def test_result_is_deterministic() -> None:
    digest = "sha256:" + ("ab" * 32)
    store = {"unit/reuse": _good_candidate("unit/reuse", digest)}
    first = execute_incremental_plan(_plan(), fetch=store.get)
    second = execute_incremental_plan(_plan(), fetch=store.get)
    assert first.result_cid() == second.result_cid()
    assert first.to_canonical() == second.to_canonical()
