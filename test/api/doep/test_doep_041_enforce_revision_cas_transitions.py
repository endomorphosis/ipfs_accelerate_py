"""Independent current-tree checks for DOEP-041 revision/CAS enforcement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CANONICAL_TASK_STATE_MACHINE_INTERFACE,
    ControlPlaneContractError,
    TaskState,
    TaskStateSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    REVISION_CAS_TRANSITION_BINDING,
    STATE_TRANSACTION_INTERFACE,
    FenceMismatchError,
    OptimisticConflictError,
    StateTransaction,
    TransactionConflictKind,
    TransactionError,
    assert_revision_cas_transition,
    assert_task_cas_transition,
    next_cas_revision,
    revision_cas_transition_allowed,
    task_cas_transition_allowed,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
TRANSACTION_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_transactions.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-041.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-041.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_transactions.py",
    "test/api/doep/test_doep_041_enforce_revision_cas_transitions.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-041.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-041.json",
)
TASK_CID = "sha256:ffd46b657cbca7def221b13cbecbb6827fc34a3c7823414398ed5dd878dca972"
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


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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


class _FakeResult:
    def __init__(self, row: object) -> None:
        self._row = row

    def fetchone(self) -> object:
        return self._row


class _FakeConnection:
    def __init__(self, revision: int) -> None:
        self.revision = revision

    def execute(self, sql: str, parameters: object = None) -> _FakeResult:
        params = list(parameters or [])
        if "UPDATE" in sql.upper():
            expected = params[-1]
            new_revision = params[-3]
            if expected != self.revision:
                return _FakeResult(None)
            self.revision = new_revision
            return _FakeResult((new_revision,))
        return _FakeResult(None)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_revision_cas_is_strictly_monotonic_and_rejects_stale_expected() -> None:
    assert STATE_TRANSACTION_INTERFACE == "StateTransaction@1"
    assert REVISION_CAS_TRANSITION_BINDING == "RevisionCASTransition@1"
    assert next_cas_revision(7) == 8
    assert assert_revision_cas_transition(expected_revision=7, live_revision=7) == 8
    assert assert_revision_cas_transition(
        expected_revision=7, live_revision=7, next_revision=8
    ) == 8
    assert revision_cas_transition_allowed(expected_revision=7, live_revision=7)
    assert not revision_cas_transition_allowed(expected_revision=7, live_revision=8)
    with pytest.raises(OptimisticConflictError):
        assert_revision_cas_transition(expected_revision=7, live_revision=8)
    with pytest.raises(OptimisticConflictError):
        assert_revision_cas_transition(
            expected_revision=7, live_revision=7, next_revision=9
        )
    with pytest.raises(OptimisticConflictError):
        StateTransaction.assert_revision_cas_transition(
            expected_revision=7, live_revision=6
        )


def test_task_cas_consumes_canonical_state_machine_and_rejects_stale_completion() -> None:
    assert CANONICAL_TASK_STATE_MACHINE_INTERFACE == "CanonicalTaskStateMachine@1"
    ready = _snapshot(state=TaskState.READY, revision=4)
    claimed = _snapshot(state=TaskState.CLAIMED, revision=5)
    assert assert_task_cas_transition(ready, claimed) == 5
    assert task_cas_transition_allowed(ready, claimed)
    in_progress = _snapshot(state=TaskState.IN_PROGRESS, revision=7)
    completed = _snapshot(state=TaskState.COMPLETED, revision=8)
    assert assert_task_cas_transition(in_progress, completed) == 8
    unknown = _snapshot(state=TaskState.PROVIDER_OUTCOME_UNKNOWN, revision=2)
    reconciling = _snapshot(state=TaskState.RECONCILING, revision=3)
    assert assert_task_cas_transition(unknown, reconciling) == 3
    with pytest.raises(ControlPlaneContractError):
        assert_task_cas_transition(
            unknown, _snapshot(state=TaskState.RETRYING, revision=3)
        )
    with pytest.raises(ControlPlaneContractError):
        assert_task_cas_transition(
            ready, _snapshot(state=TaskState.COMPLETED, revision=5)
        )
    stale_lease = _snapshot(state=TaskState.COMPLETED, revision=8, lease_id="lease:replaced")
    assert not task_cas_transition_allowed(in_progress, stale_lease)
    with pytest.raises(OptimisticConflictError):
        assert_task_cas_transition(in_progress, stale_lease)
    with pytest.raises(FenceMismatchError):
        assert_task_cas_transition(
            in_progress,
            _snapshot(state=TaskState.COMPLETED, revision=8, fence_epoch=4),
        )
    with pytest.raises(OptimisticConflictError):
        assert_task_cas_transition(
            in_progress,
            _snapshot(state=TaskState.COMPLETED, revision=9),
        )
    live_advanced = _snapshot(state=TaskState.IN_PROGRESS, revision=8)
    with pytest.raises(OptimisticConflictError):
        assert_task_cas_transition(in_progress, completed, current=live_advanced)


def test_worker_assertion_is_not_cas_authority() -> None:
    claimed = _snapshot(state=TaskState.CLAIMED, revision=7)
    completed = _snapshot(state=TaskState.COMPLETED, revision=8)
    with pytest.raises(ControlPlaneContractError):
        assert_task_cas_transition(
            claimed, completed, worker_assertion=True
        )
    in_progress = _snapshot(state=TaskState.IN_PROGRESS, revision=7)
    stale_policy = _snapshot(
        state=TaskState.COMPLETED, revision=8, policy_cid="policy:replaced"
    )
    with pytest.raises(OptimisticConflictError):
        StateTransaction.assert_task_cas_transition(
            in_progress, stale_policy, worker_assertion=True
        )
    connection = _FakeConnection(revision=7)
    txn = StateTransaction(connection, store_id="control.duckdb")
    applied = txn.cas_task_state_transition(
        in_progress,
        _snapshot(state=TaskState.COMPLETED, revision=8),
        worker_assertion=True,
    )
    assert applied == 8
    assert connection.revision == 8
    with pytest.raises(OptimisticConflictError) as stale:
        txn.cas_task_state_transition(
            in_progress,
            _snapshot(state=TaskState.COMPLETED, revision=8),
            worker_assertion=True,
        )
    assert stale.value.kind is TransactionConflictKind.OPTIMISTIC
    with pytest.raises(TransactionError):
        assert_task_cas_transition(
            in_progress,
            _snapshot(
                task_cid="sha256:" + ("ab" * 32),
                state=TaskState.COMPLETED,
                revision=8,
            ),
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
        assert payload["task_id"] == "DOEP-041"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["carrier"] == "StateTransaction"
    assert manifest["canonical_extension"]["binding"] == STATE_TRANSACTION_INTERFACE
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(TRANSACTION_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == "pending_independent_fenced_supervisor"
