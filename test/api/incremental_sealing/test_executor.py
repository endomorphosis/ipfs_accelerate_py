"""IPS-035: execute plans with fresh cache verification and verified admission."""

from __future__ import annotations

import hashlib

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
    AdmissionPolicy,
    CacheAdmissionRecord,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
    CACHE_REVERIFY_EVIDENCE_SUBSET,
    EVIDENCE_SUBSET,
    RESULT_SCHEMA,
    CacheCandidatePayload,
    ExecutionReasonCode,
    ExecutionStatus,
    ExecutorError,
    FreshProvePayload,
    IncrementalPlanExecutor,
    UnitDisposition,
    closed_execution_reason_codes,
    closed_execution_statuses,
    closed_unit_dispositions,
    execute_incremental_plan,
    plan_prove_unit_ids,
    plan_required_unit_ids,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    ParentSealContext,
    PlanMode,
    UnitPlanKind,
    create_incremental_plan,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.process_control import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.scheduling import (
    ProofResourcePolicy,
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
_DIGEST = "sha256:" + ("ab" * 32)
_DIGEST_B = "sha256:" + ("cd" * 32)
_DIGEST_C = "sha256:" + ("ef" * 32)


def _digest_for(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _integrity(digest: str | None = None, cid: str | None = None) -> IntegrityCommitment:
    d = digest or _DIGEST
    return IntegrityCommitment(
        digest=d,
        cid=cid or _DIGEST_B,
        merkle_inclusion="leaf:0",
        byte_length=32,
    )


def _unit(unit_id: str, **overrides: object):
    payload = {
        "unit_id": unit_id,
        "preserved": True,
        "cache_key_complete": True,
        "admitted": True,
        "candidate_present": True,
    }
    payload.update(overrides)
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
        UnitPlanningInput,
    )

    return UnitPlanningInput(**payload)  # type: ignore[arg-type]


def _plan_mixed():
    return create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit("unit/reuse"),
            _unit(
                "unit/changed",
                preserved=False,
                invalidated=True,
                admitted=False,
                cache_key_complete=True,
            ),
            _unit("unit/new", preserved=False, added=True, admitted=False),
            _unit("unit/gone", preserved=False, removed=True, admitted=False),
        ),
        changed_root_cids=(_DIGEST_C,),
    )


def _candidate(
    unit_id: str,
    *,
    data: bytes | None = None,
    **overrides: object,
) -> CacheCandidatePayload:
    payload_bytes = data if data is not None else f"artifact:{unit_id}".encode()
    digest = _digest_for(payload_bytes)
    base: dict[str, object] = {
        "unit_id": unit_id,
        "cache_key": f"cache-key/{unit_id}",
        "artifact_cid": digest,
        "artifact_bytes": payload_bytes,
        "expected_digest": digest,
        "evidence": _integrity(digest=digest, cid=digest),
        "proof_system_id": "integrity",
        "public_input_cid": digest,
        "proof_object_cid": digest,
        "proof_mode": ProofMode.INTEGRITY_ONLY,
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED,
        "logical_epoch": 1,
    }
    base.update(overrides)
    return CacheCandidatePayload(**base)  # type: ignore[arg-type]


def _prove(
    unit_id: str,
    *,
    data: bytes | None = None,
    **overrides: object,
) -> FreshProvePayload:
    payload_bytes = data if data is not None else f"proof:{unit_id}".encode()
    digest = _digest_for(payload_bytes)
    base: dict[str, object] = {
        "unit_id": unit_id,
        "evidence": _integrity(digest=digest, cid=digest),
        "proof_system_id": "integrity",
        "public_input_cid": digest,
        "proof_object_cid": digest,
        "proof_mode": ProofMode.INTEGRITY_ONLY,
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED,
        "prove_status": "proved",
        "expected_digest": digest,
        "observed_digest": digest,
        "logical_epoch": 1,
    }
    base.update(overrides)
    return FreshProvePayload(**base)  # type: ignore[arg-type]


def test_evidence_subsets_and_closed_vocabularies() -> None:
    assert EVIDENCE_SUBSET == "ips/incremental-execution@1"
    assert CACHE_REVERIFY_EVIDENCE_SUBSET == "ips/cache-reverification@1"
    assert RESULT_SCHEMA.endswith("incremental-proof-result@1")
    for required in (
        "completed",
        "rejected",
        "cancelled",
        "unavailable",
        "failed",
        "incomplete",
    ):
        assert required in closed_execution_statuses()
    for required in (
        "reused",
        "newly_proved",
        "removed",
        "rejected",
        "cancelled",
        "unavailable",
        "failed",
    ):
        assert required in closed_unit_dispositions()
    reasons = closed_execution_reason_codes()
    for required in (
        "reused_after_fresh_verification",
        "newly_proved_and_admitted",
        "stale_evidence",
        "poisoned_evidence",
        "corrupt_evidence",
        "cache_key_mismatch",
        "simulated_evidence",
        "prove_cancelled",
        "prove_unavailable",
    ):
        assert required in reasons


def test_happy_path_reused_and_newly_proved_exactly_cover_requirements() -> None:
    plan = _plan_mixed()
    assert plan.mode is PlanMode.INCREMENTAL
    assert plan.reusable_unit_ids == ("unit/reuse",)
    assert "unit/changed" in plan.invalidated_unit_ids
    assert plan.added_unit_ids == ("unit/new",)
    assert plan.removed_unit_ids == ("unit/gone",)

    required = plan_required_unit_ids(plan)
    prove_ids = plan_prove_unit_ids(plan)
    assert set(required) == {"unit/reuse", "unit/changed", "unit/new"}
    assert set(prove_ids) == {"unit/changed", "unit/new"}

    admissions: list[CacheAdmissionRecord] = []
    tombstones: list[tuple[str, str]] = []

    result = execute_incremental_plan(
        plan,
        candidates={"unit/reuse": _candidate("unit/reuse")},
        prove_payloads={
            "unit/changed": _prove("unit/changed"),
            "unit/new": _prove("unit/new"),
        },
        on_admission=admissions.append,
        on_tombstone=lambda uid, reason: tombstones.append((uid, reason)),
    )

    assert result.status is ExecutionStatus.COMPLETED
    assert result.success is True
    assert result.ready_for_aggregation is True
    assert result.coverage_complete is True
    assert set(result.reused_unit_ids) | set(result.newly_proved_unit_ids) == set(
        result.required_unit_ids
    )
    assert set(result.reused_unit_ids) & set(result.newly_proved_unit_ids) == set()
    assert result.reused_unit_ids == ("unit/reuse",)
    assert set(result.newly_proved_unit_ids) == {"unit/changed", "unit/new"}
    assert result.removed_unit_ids == ("unit/gone",)
    assert result.rejected_unit_ids == ()
    assert result.cancelled_unit_ids == ()
    assert result.unavailable_unit_ids == ()
    assert len(result.admissions) == 3
    assert len(admissions) == 3
    assert all(item.verified is True for item in result.admissions)
    # Removal produces a tombstone; successful reuse/prove do not.
    assert ("unit/gone", "removed") in tombstones
    for record in result.unit_records:
        if record.disposition is UnitDisposition.REUSED:
            assert record.freshly_verified is True
            assert record.cache_admission_record is not None
            assert record.reverify_digest is not None
        if record.disposition is UnitDisposition.NEWLY_PROVED:
            assert record.freshly_verified is True
            assert record.cache_admission_record is not None
    canonical = result.to_canonical()
    assert canonical["ready_for_aggregation"] is True
    assert canonical["evidence_subset"] == EVIDENCE_SUBSET


def test_no_cache_fast_path_missing_candidate_rejects() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(_unit("unit/reuse"),),
    )
    result = execute_incremental_plan(plan, candidates={})
    assert result.ready_for_aggregation is False
    assert result.status is ExecutionStatus.REJECTED
    assert "unit/reuse" in result.rejected_unit_ids
    assert result.reused_unit_ids == ()
    record = result.unit_records[0]
    assert record.reason_code == ExecutionReasonCode.MISSING_CANDIDATE.value
    assert record.cache_admission_record is None


@pytest.mark.parametrize(
    ("flag", "reason"),
    [
        ("stale", ExecutionReasonCode.STALE_EVIDENCE),
        ("poisoned", ExecutionReasonCode.POISONED_EVIDENCE),
        ("corrupt", ExecutionReasonCode.CORRUPT_EVIDENCE),
        ("simulated", ExecutionReasonCode.SIMULATED_EVIDENCE),
        ("cache_key_mismatch", ExecutionReasonCode.CACHE_KEY_MISMATCH),
    ],
)
def test_stale_poisoned_corrupt_mismatched_simulated_rejected(
    flag: str, reason: ExecutionReasonCode
) -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(_unit("unit/bad"),),
    )
    kwargs = {flag: True}
    if flag == "corrupt":
        # Keep expected_digest wrong relative to bytes.
        data = b"poisoned-bytes"
        candidate = _candidate(
            "unit/bad",
            data=data,
            expected_digest=_DIGEST,  # mismatch with actual bytes
            corrupt=True,
        )
    else:
        candidate = _candidate("unit/bad", **kwargs)

    tombstones: list[tuple[str, str]] = []
    result = execute_incremental_plan(
        plan,
        candidates={"unit/bad": candidate},
        on_tombstone=lambda uid, r: tombstones.append((uid, r)),
    )
    assert result.ready_for_aggregation is False
    assert result.status is ExecutionStatus.REJECTED
    assert result.rejected_unit_ids == ("unit/bad",)
    assert result.unit_records[0].reason_code == reason.value
    assert result.unit_records[0].cache_admission_record is None
    assert any(uid == "unit/bad" for uid, _ in tombstones)


def test_rehash_mismatch_rejects_without_admission() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(_unit("unit/rehash"),),
    )
    candidate = _candidate(
        "unit/rehash",
        data=b"actual-bytes",
        expected_digest=_DIGEST,  # not the digest of actual-bytes
    )
    result = execute_incremental_plan(
        plan, candidates={"unit/rehash": candidate}
    )
    assert result.ready_for_aggregation is False
    assert result.unit_records[0].reason_code == ExecutionReasonCode.REHASH_FAILED.value
    assert "unit/rehash" in result.tombstones


def test_public_input_mismatch_on_reuse_rejects() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(_unit("unit/pi"),),
    )
    data = b"pi-unit"
    digest = _digest_for(data)
    candidate = _candidate(
        "unit/pi",
        data=data,
        public_input_cid=digest,
        observed_public_input_cid=_DIGEST_B,
    )
    result = execute_incremental_plan(plan, candidates={"unit/pi": candidate})
    assert result.ready_for_aggregation is False
    assert (
        result.unit_records[0].reason_code
        == ExecutionReasonCode.PUBLIC_INPUT_MISMATCH.value
    )


def test_admission_rejection_on_simulated_mode_evidence() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(_unit("unit/sim-mode"),),
    )
    data = b"sim-mode"
    candidate = _candidate(
        "unit/sim-mode",
        data=data,
        proof_mode=ProofMode.SIMULATED,
        terminal_status=ProofTerminalStatus.SIMULATED,
        # Mode alone is enough; simulated flag stays false.
        simulated=False,
    )
    result = execute_incremental_plan(
        plan, candidates={"unit/sim-mode": candidate}
    )
    assert result.ready_for_aggregation is False
    assert result.unit_records[0].reason_code == ExecutionReasonCode.SIMULATED_EVIDENCE.value


def test_fresh_prove_admission_rejection_blocks_aggregation() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit(
                "unit/bad-proof",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
    )
    # Wrong observed digest causes integrity admission failure.
    data = b"new-proof"
    digest = _digest_for(data)
    payload = _prove(
        "unit/bad-proof",
        data=data,
        expected_digest=digest,
        observed_digest=_DIGEST_B,
    )
    result = execute_incremental_plan(plan, prove_payloads={"unit/bad-proof": payload})
    assert result.ready_for_aggregation is False
    assert result.status is ExecutionStatus.REJECTED
    assert "unit/bad-proof" in result.rejected_unit_ids
    assert result.unit_records[0].reason_code == ExecutionReasonCode.ADMISSION_REJECTED.value
    assert result.admissions == ()


def test_cancellation_cannot_proceed_to_aggregation() -> None:
    plan = _plan_mixed()
    token = CancellationToken()
    token.cancel("operator")
    result = execute_incremental_plan(
        plan,
        candidates={"unit/reuse": _candidate("unit/reuse")},
        prove_payloads={
            "unit/changed": _prove("unit/changed"),
            "unit/new": _prove("unit/new"),
        },
        cancellation=token,
    )
    assert result.ready_for_aggregation is False
    assert result.status is ExecutionStatus.CANCELLED
    assert result.success is False
    assert set(result.cancelled_unit_ids) == set(result.required_unit_ids)
    assert result.reused_unit_ids == ()
    assert result.newly_proved_unit_ids == ()
    for record in result.unit_records:
        if record.unit_id in result.required_unit_ids:
            assert record.disposition is UnitDisposition.CANCELLED
            assert record.ready_for_aggregation is False
            assert record.cache_admission_record is None


def test_prove_cancelled_status_blocks_aggregation() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit(
                "unit/cancel",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
    )
    result = execute_incremental_plan(
        plan,
        prove_payloads={
            "unit/cancel": _prove("unit/cancel", prove_status="cancelled"),
        },
    )
    assert result.ready_for_aggregation is False
    assert result.status is ExecutionStatus.CANCELLED
    assert result.cancelled_unit_ids == ("unit/cancel",)
    assert result.unit_records[0].reason_code == ExecutionReasonCode.PROVE_CANCELLED.value


def test_unavailable_prove_cannot_proceed_to_aggregation() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit(
                "unit/gone-backend",
                preserved=False,
                added=True,
                admitted=False,
            ),
        ),
    )
    result = execute_incremental_plan(
        plan,
        prove_payloads={
            "unit/gone-backend": _prove(
                "unit/gone-backend", prove_status="unavailable"
            ),
        },
    )
    assert result.ready_for_aggregation is False
    assert result.status is ExecutionStatus.UNAVAILABLE
    assert result.unavailable_unit_ids == ("unit/gone-backend",)
    assert result.unit_records[0].reason_code == ExecutionReasonCode.PROVE_UNAVAILABLE.value


def test_missing_prove_payload_is_unavailable() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit(
                "unit/missing-prove",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
    )
    result = execute_incremental_plan(plan, prove_payloads={})
    assert result.ready_for_aggregation is False
    assert result.status is ExecutionStatus.UNAVAILABLE
    assert "unit/missing-prove" in result.unavailable_unit_ids


def test_resource_unavailable_blocks_aggregation() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit(
                "unit/gpu",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
    )
    result = execute_incremental_plan(
        plan,
        resource_policy=ProofResourcePolicy(max_gpu=0),
        prove_payloads={
            "unit/gpu": _prove("unit/gpu", gpu=1),
        },
    )
    assert result.ready_for_aggregation is False
    assert result.status is ExecutionStatus.UNAVAILABLE
    assert result.unavailable_unit_ids == ("unit/gpu",)
    assert (
        result.unit_records[0].reason_code
        == ExecutionReasonCode.RESOURCE_UNAVAILABLE.value
    )


def test_reject_reuse_units_are_proved_for_coverage() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit(
                "unit/hint-only",
                admitted=False,
                candidate_present=True,
                cache_key_complete=True,
            ),
            _unit("unit/ok"),
        ),
    )
    assert plan.units[0].kind is UnitPlanKind.REJECT_REUSE or any(
        u.kind is UnitPlanKind.REJECT_REUSE for u in plan.units
    )
    required = plan_required_unit_ids(plan)
    assert "unit/hint-only" in required
    assert "unit/hint-only" in plan_prove_unit_ids(plan)

    result = execute_incremental_plan(
        plan,
        candidates={"unit/ok": _candidate("unit/ok")},
        prove_payloads={"unit/hint-only": _prove("unit/hint-only")},
    )
    assert result.success is True
    assert "unit/hint-only" in result.newly_proved_unit_ids
    assert "unit/ok" in result.reused_unit_ids
    assert set(result.reused_unit_ids) | set(result.newly_proved_unit_ids) == set(
        required
    )


def test_executor_class_matches_facade() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(_unit("unit/a"),),
    )
    executor = IncrementalPlanExecutor(
        candidates={"unit/a": _candidate("unit/a")},
        admission_policy=AdmissionPolicy(),
    )
    result = executor.execute(plan)
    assert result.success is True
    assert result.plan_cid == plan.plan_cid()


def test_incomplete_plan_raises() -> None:
    with pytest.raises(ExecutorError, match="IncrementalProofPlan"):
        execute_incremental_plan("not-a-plan")  # type: ignore[arg-type]


def test_result_invariants_reject_aggregation_on_cancel() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
        IncrementalProofResult,
        UnitExecutionRecord,
    )

    with pytest.raises(ExecutorError):
        IncrementalProofResult(
            schema=RESULT_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            status=ExecutionStatus.COMPLETED,
            plan_cid="sha256:" + ("11" * 32),
            reused_unit_ids=("unit/a",),
            newly_proved_unit_ids=(),
            removed_unit_ids=(),
            rejected_unit_ids=(),
            cancelled_unit_ids=("unit/a",),
            unavailable_unit_ids=(),
            failed_unit_ids=(),
            required_unit_ids=("unit/a",),
            unit_records=(
                UnitExecutionRecord(
                    unit_id="unit/a",
                    disposition=UnitDisposition.CANCELLED,
                    reason_code=ExecutionReasonCode.PROVE_CANCELLED.value,
                    message="cancelled",
                    freshly_verified=False,
                    ready_for_aggregation=False,
                ),
            ),
            admissions=(),
            tombstones=(),
            ready_for_aggregation=True,  # illegal with cancelled units
            coverage_complete=True,
            message="bad",
        )


def test_unit_record_rejects_admission_on_cancelled() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
        UnitExecutionRecord,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
        ADMISSION_SCHEMA,
        CacheAdmissionRecord,
    )

    bogus = CacheAdmissionRecord(
        schema=ADMISSION_SCHEMA,
        proof_unit_id="unit/x",
        evidence_class="IntegrityCommitment",
        proof_system_id="integrity",
        proof_object_cid="n/a",
        public_input_cid=_DIGEST,
        verification_digest=_DIGEST,
        establishes="x",
        does_not_establish="y",
        logical_epoch=0,
        verified=True,
    )
    with pytest.raises(ExecutorError):
        UnitExecutionRecord(
            unit_id="unit/x",
            disposition=UnitDisposition.CANCELLED,
            reason_code=ExecutionReasonCode.PROVE_CANCELLED.value,
            message="cancelled",
            freshly_verified=False,
            ready_for_aggregation=False,
            cache_admission_record=bogus,
        )


def test_lookup_hooks_are_used() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit("unit/reuse"),
            _unit(
                "unit/prove",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
    )
    result = execute_incremental_plan(
        plan,
        candidate_lookup=lambda uid: _candidate(uid) if uid == "unit/reuse" else None,
        prove_lookup=lambda uid: _prove(uid) if uid == "unit/prove" else None,
    )
    assert result.success is True
    assert result.reused_unit_ids == ("unit/reuse",)
    assert result.newly_proved_unit_ids == ("unit/prove",)


def test_candidate_unit_id_mismatch_rejects() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(_unit("unit/expected"),),
    )
    mismatched = _candidate("unit/other")
    result = execute_incremental_plan(
        plan, candidates={"unit/expected": mismatched}
    )
    assert result.ready_for_aggregation is False
    assert result.status is ExecutionStatus.REJECTED
    assert "unit/expected" in result.rejected_unit_ids
    assert (
        result.unit_records[0].reason_code
        == ExecutionReasonCode.CACHE_KEY_MISMATCH.value
    )


def test_reused_and_newly_proved_are_disjoint_and_exact() -> None:
    plan = _plan_mixed()
    result = execute_incremental_plan(
        plan,
        candidates={"unit/reuse": _candidate("unit/reuse")},
        prove_payloads={
            "unit/changed": _prove("unit/changed"),
            "unit/new": _prove("unit/new"),
        },
    )
    assert result.success is True
    assert result.ready_for_aggregation is True
    reused = set(result.reused_unit_ids)
    newly = set(result.newly_proved_unit_ids)
    required = set(result.required_unit_ids)
    assert reused | newly == required
    assert reused & newly == set()
    assert result.cancelled_unit_ids == ()
    assert result.unavailable_unit_ids == ()
    # Required units that succeed must be aggregation-ready and freshly verified.
    assert all(
        record.ready_for_aggregation is True
        and record.freshly_verified is True
        and record.cache_admission_record is not None
        for record in result.unit_records
        if record.unit_id in required
    )


def test_simulated_prove_payload_cannot_aggregate() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit(
                "unit/sim-prove",
                preserved=False,
                added=True,
                admitted=False,
            ),
        ),
    )
    result = execute_incremental_plan(
        plan,
        prove_payloads={
            "unit/sim-prove": _prove(
                "unit/sim-prove",
                prove_status="simulated",
                proof_mode=ProofMode.SIMULATED,
                terminal_status=ProofTerminalStatus.SIMULATED,
            ),
        },
    )
    assert result.ready_for_aggregation is False
    assert result.status is ExecutionStatus.REJECTED
    assert result.rejected_unit_ids == ("unit/sim-prove",)
    assert (
        result.unit_records[0].reason_code
        == ExecutionReasonCode.SIMULATED_EVIDENCE.value
    )
    assert result.admissions == ()
