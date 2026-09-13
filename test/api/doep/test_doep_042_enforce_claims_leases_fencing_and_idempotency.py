"""Independent current-tree checks for DOEP-042 claim/lease/fence/idempotency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CANONICAL_TASK_STATE_MACHINE_INTERFACE,
    CommandOutcome,
    ControlPlaneContractError,
    TaskState,
    TaskStateSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    STATE_TRANSACTION_INTERFACE,
    FenceMismatchError,
    IdempotencyConflictError,
    OptimisticConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
    ACCEPTED_LEASE_STATE,
    CLAIM_LEASE_FENCE_IDEMPOTENCY_BINDING,
    QUACK_OWNER_MUTATION_INTERFACE,
    LiveOwnerClaimLease,
    OwnerMutationAuthority,
    QuackOwnerMutationEnvelopeError,
    admit_owner_mutation,
    assert_live_claim_lease_fence,
    assert_mutation_idempotency,
    claim_lease_fence_idempotency_allowed,
    mutation_request_digest,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
MUTATION_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-042.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-042.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
    "test/api/doep/test_doep_042_enforce_claims_leases_fencing_and_idempotency.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-042.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-042.json",
)
TASK_CID = "sha256:1b802b8cb08834a07cc3900c218964477e3d079d9871398e615289a425b67c1a"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}
NOW_MS = 1_700_000_000_000
SQL = "UPDATE tasks SET status = ? WHERE task_cid = ? AND revision = ?"
PARAMETERS = ["completed", TASK_CID, 7]


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _digest() -> str:
    return mutation_request_digest(sql=SQL, parameters=PARAMETERS)


def _requested(**updates: object) -> OwnerMutationAuthority:
    values: dict[str, object] = {
        "task_cid": TASK_CID,
        "claim_id": "claim:current",
        "lease_id": "lease:current",
        "claimant_did": "did:key:owner",
        "fencing_token": 1,
        "fence_epoch": 3,
        "owner_session_id": "session:owner",
        "idempotency_key": "idem:current",
        "request_digest": _digest(),
    }
    values.update(updates)
    return OwnerMutationAuthority(**values)


def _live(**updates: object) -> LiveOwnerClaimLease:
    values: dict[str, object] = {
        "task_cid": TASK_CID,
        "claim_id": "claim:current",
        "lease_id": "lease:current",
        "claimant_did": "did:key:owner",
        "fencing_token": 1,
        "fence_epoch": 3,
        "expires_at_ms": NOW_MS + 60_000,
        "state": ACCEPTED_LEASE_STATE,
        "owner_session_id": "session:owner",
        "revision": 7,
    }
    values.update(updates)
    return LiveOwnerClaimLease(**values)


def _snapshot(**updates: object) -> TaskStateSnapshot:
    values: dict[str, object] = {
        "task_cid": TASK_CID,
        "state": TaskState.IN_PROGRESS,
        "revision": 7,
        "lease_id": "lease:current",
        "fence_epoch": 3,
        "policy_cid": "policy:current",
        "repository_tree_id": "tree:current",
        "plan_cid": PLAN_CID,
        "plan_epoch": 1,
    }
    values.update(updates)
    return TaskStateSnapshot(**values)


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_claims_leases_and_fences_reject_stale_bindings() -> None:
    assert QUACK_OWNER_MUTATION_INTERFACE == "QuackOwnerMutation@1"
    assert CLAIM_LEASE_FENCE_IDEMPOTENCY_BINDING == "ClaimLeaseFenceIdempotency@1"
    assert CANONICAL_TASK_STATE_MACHINE_INTERFACE == "CanonicalTaskStateMachine@1"
    assert STATE_TRANSACTION_INTERFACE == "StateTransaction@1"
    requested = _requested()
    live = _live()
    assert assert_live_claim_lease_fence(requested, live, now_ms=NOW_MS) is live
    admitted = admit_owner_mutation(requested, live, now_ms=NOW_MS)
    assert admitted.outcome is CommandOutcome.ACCEPTED
    assert admitted.changed is True
    assert admitted.to_dict()["worker_assertion_is_authority"] is False
    assert admitted.to_dict()["consumes"]["state_transaction"] == STATE_TRANSACTION_INTERFACE
    with pytest.raises(QuackOwnerMutationEnvelopeError) as claim_exc:
        assert_live_claim_lease_fence(
            _requested(claim_id="claim:replaced"), live, now_ms=NOW_MS
        )
    assert claim_exc.value.code == "claim_mismatch"
    with pytest.raises(QuackOwnerMutationEnvelopeError) as lease_exc:
        assert_live_claim_lease_fence(
            _requested(lease_id="lease:replaced"), live, now_ms=NOW_MS
        )
    assert lease_exc.value.code == "lease_mismatch"
    with pytest.raises(QuackOwnerMutationEnvelopeError) as principal_exc:
        assert_live_claim_lease_fence(
            _requested(claimant_did="did:key:intruder"), live, now_ms=NOW_MS
        )
    assert principal_exc.value.code == "lease_mismatch"
    with pytest.raises(QuackOwnerMutationEnvelopeError) as revoked:
        assert_live_claim_lease_fence(
            requested, _live(state="released"), now_ms=NOW_MS
        )
    assert revoked.value.code == "lease_revoked"
    with pytest.raises(QuackOwnerMutationEnvelopeError) as expired:
        assert_live_claim_lease_fence(
            requested, _live(expires_at_ms=NOW_MS), now_ms=NOW_MS
        )
    assert expired.value.code == "lease_expired"
    with pytest.raises(FenceMismatchError):
        assert_live_claim_lease_fence(
            _requested(fence_epoch=4), live, now_ms=NOW_MS
        )
    with pytest.raises(FenceMismatchError):
        assert_live_claim_lease_fence(
            _requested(fencing_token=2), live, now_ms=NOW_MS
        )
    path_live = _live(path_claim_id="claim:path", path_claim_state=ACCEPTED_LEASE_STATE)
    path_requested = _requested(claim_id="claim:path")
    assert assert_live_claim_lease_fence(path_requested, path_live, now_ms=NOW_MS) is path_live
    with pytest.raises(QuackOwnerMutationEnvelopeError) as path_revoked:
        assert_live_claim_lease_fence(
            path_requested,
            _live(path_claim_id="claim:path", path_claim_state="released"),
            now_ms=NOW_MS,
        )
    assert path_revoked.value.code == "claim_mismatch"
    assert not claim_lease_fence_idempotency_allowed(
        _requested(lease_id="lease:replaced"), live, now_ms=NOW_MS
    )


def test_idempotent_replay_is_exact_digest_and_conflicts_fail_closed() -> None:
    requested = _requested()
    live = _live()
    prior = {
        "idempotency_key": requested.idempotency_key,
        "request_digest": requested.request_digest,
        "body": {"ok": True, "rowcount": 1},
    }
    replayed = admit_owner_mutation(
        requested, live, now_ms=NOW_MS, existing_idempotency=prior
    )
    assert replayed.outcome is CommandOutcome.IDEMPOTENT_REPLAY
    assert replayed.changed is False
    assert dict(replayed.replay_result or {}) == {"ok": True, "rowcount": 1}
    assert assert_mutation_idempotency(requested, None) is None
    conflicting = mutation_request_digest(
        sql=SQL, parameters=["in_progress", TASK_CID, 7]
    )
    with pytest.raises(IdempotencyConflictError):
        assert_mutation_idempotency(
            requested,
            {
                "idempotency_key": requested.idempotency_key,
                "request_digest": conflicting,
                "body": {"ok": True},
            },
        )
    with pytest.raises(IdempotencyConflictError):
        admit_owner_mutation(
            requested,
            live,
            now_ms=NOW_MS,
            existing_idempotency={
                "idempotency_key": "idem:other",
                "request_digest": requested.request_digest,
            },
        )
    expected = _snapshot()
    proposed = _snapshot(state=TaskState.COMPLETED, revision=8)
    replay_skips_cas = admit_owner_mutation(
        requested,
        live,
        now_ms=NOW_MS,
        existing_idempotency=prior,
        expected_task=expected,
        proposed_task=proposed,
    )
    assert replay_skips_cas.outcome is CommandOutcome.IDEMPOTENT_REPLAY
    accepted = admit_owner_mutation(
        requested,
        live,
        now_ms=NOW_MS,
        expected_task=expected,
        proposed_task=proposed,
    )
    assert accepted.outcome is CommandOutcome.ACCEPTED
    with pytest.raises(OptimisticConflictError):
        admit_owner_mutation(
            requested,
            live,
            now_ms=NOW_MS,
            expected_task=expected,
            proposed_task=_snapshot(
                state=TaskState.COMPLETED, revision=8, lease_id="lease:replaced"
            ),
        )
    with pytest.raises(ControlPlaneContractError):
        admit_owner_mutation(
            requested,
            live,
            now_ms=NOW_MS,
            expected_task=_snapshot(state=TaskState.CLAIMED),
            proposed_task=proposed,
        )


def test_worker_assertion_is_not_mutation_authority() -> None:
    requested = _requested()
    live = _live()
    with pytest.raises(QuackOwnerMutationEnvelopeError) as expired:
        admit_owner_mutation(
            requested,
            _live(expires_at_ms=NOW_MS - 1),
            now_ms=NOW_MS,
            worker_assertion=True,
        )
    assert expired.value.code == "lease_expired"
    with pytest.raises(FenceMismatchError):
        admit_owner_mutation(
            _requested(fence_epoch=9),
            live,
            now_ms=NOW_MS,
            worker_assertion=True,
        )
    with pytest.raises(IdempotencyConflictError):
        admit_owner_mutation(
            requested,
            live,
            now_ms=NOW_MS,
            existing_idempotency={
                "idempotency_key": requested.idempotency_key,
                "request_digest": "sha256:" + ("ab" * 32),
            },
            worker_assertion=True,
        )
    expected = _snapshot()
    proposed = _snapshot(state=TaskState.COMPLETED, revision=8)
    still_admitted = admit_owner_mutation(
        requested,
        live,
        now_ms=NOW_MS,
        expected_task=expected,
        proposed_task=proposed,
        worker_assertion=True,
    )
    assert still_admitted.outcome is CommandOutcome.ACCEPTED
    stale_current = _snapshot(revision=8)
    with pytest.raises(OptimisticConflictError):
        admit_owner_mutation(
            requested,
            live,
            now_ms=NOW_MS,
            expected_task=expected,
            proposed_task=proposed,
            current_task=stale_current,
            worker_assertion=True,
        )
    with pytest.raises(ControlPlaneContractError):
        admit_owner_mutation(
            requested,
            live,
            now_ms=NOW_MS,
            expected_task=_snapshot(state=TaskState.CLAIMED),
            proposed_task=_snapshot(state=TaskState.COMPLETED, revision=8),
            worker_assertion=True,
        )
    assert not claim_lease_fence_idempotency_allowed(
        requested,
        _live(state="released"),
        now_ms=NOW_MS,
        worker_assertion=True,
    )


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-042"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["carrier"] == "QuackOwnerMutation"
    assert manifest["canonical_extension"]["binding"] == CLAIM_LEASE_FENCE_IDEMPOTENCY_BINDING
    assert manifest["canonical_extension"]["consumes"] == [
        CANONICAL_TASK_STATE_MACHINE_INTERFACE,
        STATE_TRANSACTION_INTERFACE,
    ]
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(MUTATION_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == "pending_independent_fenced_supervisor"
