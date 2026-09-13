"""Independent current-tree checks for DOEP-044 owner-loss recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    EXTERNAL_QUACK_OWNER_INTERFACE,
    OWNER_LEASE_INTERFACE,
    OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING,
    OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_CONSUMES,
    OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_INTERFACE,
    OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_SCHEMA,
    OWNER_LOSS_OBSERVATION_SCHEMA,
    ExternalQuackOwner,
    ExternalQuackOwnerError,
    ExternalQuackOwnerNotReady,
    OwnerLease,
    OwnerLossKind,
    OwnerRecoveryOutcome,
    OwnerRestartRecovery,
    StaleOwnerError,
    _bind_external_quack_owner,
    assert_owner_restart_successor,
    assert_stale_owner_cannot_complete,
    detect_owner_loss,
    owner_lease_is_current,
    recover_from_owner_restart,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import ServerLifecycle
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CANONICAL_TASK_STATE_MACHINE_INTERFACE,
    TaskState,
    TaskStateSnapshot,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OWNER_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/runtime/external_quack_owner.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-044.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-044.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/external_quack_owner.py",
    "test/api/doep/test_doep_044_add_owner_loss_and_owner_restart_recovery.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-044.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-044.json",
)
TASK_CID = "sha256:f54f588bed8a89b5dc0e85c9cb3158aa57945328933cfb2b825f234b0e70ccfa"
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
BOARD_NAMESPACE = "agent-supervisor-direct-objective-and-event-driven-planning-v1"
SHARD_ID = "shard:doep-044"


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _lease(**updates: object) -> OwnerLease:
    values: dict[str, object] = {
        "board_namespace": BOARD_NAMESPACE,
        "server_id": "server:owner-1",
        "store_id": "store:doep-044",
        "database_uuid": "uuid:doep-044",
        "generation": 1,
        "fence_epoch": 1,
        "secret_handle": "handle:doep-044",
        "listen_uri": "quack://127.0.0.1:19495",
        "shard_id": SHARD_ID,
    }
    values.update(updates)
    return OwnerLease(**values)  # type: ignore[arg-type]


def _successor(previous: OwnerLease, **updates: object) -> OwnerLease:
    values: dict[str, object] = {
        "server_id": "server:owner-2",
        "generation": previous.generation + 1,
        "fence_epoch": previous.fence_epoch + 1,
        "listen_uri": "quack://127.0.0.1:19496",
        "secret_handle": "handle:doep-044-restart",
    }
    values.update(updates)
    return replace(previous, **values)  # type: ignore[arg-type]


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
    return TaskStateSnapshot(**values)  # type: ignore[arg-type]


class _FakeIdentity:
    def __init__(self, lease: OwnerLease) -> None:
        self.server_id = lease.server_id
        self.store_id = lease.store_id
        self.database_uuid = lease.database_uuid
        self.generation = lease.generation
        self.fence_epoch = lease.fence_epoch
        self.secret_handle = lease.secret_handle
        self.listen_uri = lease.listen_uri


class _FakeHold:
    def __init__(self, *, held: bool = True, fence_token: str = "fence:1") -> None:
        self.held = held
        self.fence_token = fence_token


class _FakeServer:
    def __init__(
        self,
        lease: OwnerLease,
        *,
        lifecycle: ServerLifecycle | None = None,
        held: bool = True,
    ) -> None:
        self.lifecycle = (
            ServerLifecycle.READY if lifecycle is None else lifecycle
        )
        self.identity = _FakeIdentity(lease)
        self._owner = _FakeHold(held=held)
        self._connection = object()


def _bind(lease: OwnerLease, *, held: bool = True) -> ExternalQuackOwner:
    return _bind_external_quack_owner(
        owner_server=_FakeServer(lease, held=held),
        board_namespace=lease.board_namespace,
        shard_id=lease.shard_id,
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_external_quack_owner_without_competing_subsystem() -> None:
    assert EXTERNAL_QUACK_OWNER_INTERFACE == "ExternalQuackOwner@1"
    assert OWNER_LEASE_INTERFACE == "ExternalQuackOwnerLease@1"
    assert OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING == (
        "OwnerLossAndOwnerRestartRecovery@1"
    )
    assert OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_INTERFACE == (
        OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING
    )
    assert OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/owner-loss-and-owner-restart-recovery@1"
    )
    assert OWNER_LOSS_OBSERVATION_SCHEMA.endswith("owner-loss-observation@1")
    assert OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_CONSUMES == (
        EXTERNAL_QUACK_OWNER_INTERFACE,
        OWNER_LEASE_INTERFACE,
        CANONICAL_TASK_STATE_MACHINE_INTERFACE,
    )
    assert ExternalQuackOwner.INTERFACE == EXTERNAL_QUACK_OWNER_INTERFACE
    assert (
        ExternalQuackOwner.OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING
        == OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING
    )
    assert (
        ExternalQuackOwner.CONSUMES_TASK_STATE_MACHINE
        == CANONICAL_TASK_STATE_MACHINE_INTERFACE
    )
    assert ExternalQuackOwner.recover_owner_restart.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner"
    )
    assert ExternalQuackOwner.observe_owner_loss.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner"
    )
    source = OWNER_PATH.read_text(encoding="utf-8")
    assert "class ExternalQuackOwner" in source
    assert "def observe_owner_loss(" in source
    assert "def recover_owner_restart(" in source
    assert "def detect_owner_loss(" in source
    assert "def recover_from_owner_restart(" in source
    assert "def assert_owner_restart_successor(" in source
    assert "def assert_stale_owner_cannot_complete(" in source
    assert "class OwnerRecoveryService" not in source
    assert "class CompetingQuackOwner" not in source
    assert "class ExternalOwnerRestartSubsystem" not in source
    assert "class OwnerLossController" not in source


def test_owner_loss_is_detected_and_cannot_complete_even_with_worker_assertion() -> None:
    previous = _lease()
    current = _successor(previous)
    assert detect_owner_loss(lease=previous, current=previous) is None
    stopped = detect_owner_loss(
        lease=previous,
        ready=False,
        held=False,
        lifecycle="stopped",
        worker_assertion=True,
    )
    assert stopped is not None
    assert stopped.kind is OwnerLossKind.PROCESS_STOPPED
    assert stopped.outcome is OwnerRecoveryOutcome.OWNER_LOST
    assert stopped.authorizes_completion is False
    assert stopped.worker_assertion_is_authority is False
    assert stopped.worker_assertion is True
    lost_hold = detect_owner_loss(lease=previous, ready=True, held=False)
    assert lost_hold is not None
    assert lost_hold.kind is OwnerLossKind.LOST_HOLD
    stale = detect_owner_loss(lease=previous, current=current, worker_assertion=True)
    assert stale is not None
    assert stale.kind is OwnerLossKind.STALE_GENERATION
    fence_only = detect_owner_loss(
        lease=previous,
        current=replace(previous, fence_epoch=2, server_id="server:owner-fence"),
    )
    assert fence_only is not None
    assert fence_only.kind is OwnerLossKind.STALE_FENCE
    snapshot = _snapshot()
    with pytest.raises(ExternalQuackOwnerNotReady) as lost:
        assert_stale_owner_cannot_complete(
            snapshot,
            lease=previous,
            current=None,
            ready=False,
            held=False,
            lifecycle="stopped",
            worker_assertion=True,
        )
    assert lost.value.reason_code == "process_stopped"
    with pytest.raises(StaleOwnerError) as rejected:
        assert_stale_owner_cannot_complete(
            snapshot,
            lease=previous,
            current=current,
            worker_assertion=True,
        )
    assert rejected.value.reason_code == "stale_owner"
    with pytest.raises(ExternalQuackOwnerError) as insufficient:
        assert_stale_owner_cannot_complete(
            _snapshot(state=TaskState.READY),
            lease=current,
            current=current,
            worker_assertion=True,
        )
    assert insufficient.value.reason_code == "worker_assertion_insufficient"
    assert (
        assert_stale_owner_cannot_complete(snapshot, lease=current, current=current)
        == current
    )


def test_owner_restart_admits_later_generation_and_rejects_invalid_failover() -> None:
    previous = _lease()
    current = _successor(previous)
    assert not owner_lease_is_current(previous, current)
    assert owner_lease_is_current(current, current)
    admitted = assert_owner_restart_successor(
        previous, current, worker_assertion=True
    )
    assert admitted == current
    recovery = recover_from_owner_restart(
        previous, current, worker_assertion=True
    )
    assert recovery.outcome is OwnerRecoveryOutcome.SUCCESSOR_ADMITTED
    assert recovery.authorizes_completion is False
    assert recovery.worker_assertion_is_authority is False
    assert recovery.current.generation > previous.generation
    assert recovery.current.fence_epoch > previous.fence_epoch
    assert recovery.current.server_id != previous.server_id
    payload = recovery.to_dict()
    assert payload["binding"] == OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING
    assert payload["carrier"] == EXTERNAL_QUACK_OWNER_INTERFACE
    with pytest.raises(StaleOwnerError) as same_generation:
        assert_owner_restart_successor(previous, previous, worker_assertion=True)
    assert same_generation.value.reason_code == "invalid_failover"
    with pytest.raises(StaleOwnerError):
        recover_from_owner_restart(
            previous,
            replace(current, store_id="store:other"),
            worker_assertion=True,
        )
    with pytest.raises(StaleOwnerError):
        OwnerRestartRecovery(
            previous=previous,
            current=replace(current, server_id=previous.server_id),
        )
    with pytest.raises(StaleOwnerError):
        assert_owner_restart_successor(
            previous,
            replace(current, generation=previous.generation, fence_epoch=9),
        )


def test_facade_recovers_lost_owner_and_rejects_stale_completion() -> None:
    previous_lease = _lease()
    first = _bind(previous_lease)
    issued = first.lease()
    assert first.assert_current(issued) == issued
    with pytest.raises(ExternalQuackOwnerError) as not_lost:
        first.observe_owner_loss(issued)
    assert not_lost.value.reason_code == "owner_not_lost"
    assert first.evidence()["owner_loss_and_owner_restart_recovery_binding"] == (
        OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING
    )
    snapshot = _snapshot()
    assert first.assert_task_may_complete(snapshot, lease=issued) == issued

    first._owner_server.lifecycle = ServerLifecycle.STOPPED  # noqa: SLF001
    first._owner_server._owner.held = False  # noqa: SLF001
    first._owner_server._connection = None  # noqa: SLF001
    with pytest.raises(ExternalQuackOwnerNotReady):
        first.lease()
    loss = first.observe_owner_loss(issued, worker_assertion=True)
    assert loss.kind is OwnerLossKind.PROCESS_STOPPED
    assert loss.outcome is OwnerRecoveryOutcome.OWNER_LOST
    assert loss.authorizes_completion is False
    with pytest.raises(ExternalQuackOwnerNotReady):
        first.assert_task_may_complete(snapshot, lease=issued, worker_assertion=True)
    with pytest.raises(ExternalQuackOwnerNotReady):
        first.assert_current(issued, worker_assertion=True)

    successor_lease = _successor(issued)
    successor = _bind(successor_lease)
    recovered = successor.recover_owner_restart(issued, worker_assertion=True)
    assert recovered.outcome is OwnerRecoveryOutcome.SUCCESSOR_ADMITTED
    assert recovered.current == successor.lease()
    with pytest.raises(StaleOwnerError) as stale:
        successor.assert_current(issued, worker_assertion=True)
    assert stale.value.reason_code == "stale_owner"
    with pytest.raises(StaleOwnerError):
        successor.assert_task_may_complete(
            snapshot, lease=issued, worker_assertion=True
        )
    assert successor.assert_task_may_complete(snapshot) == successor.lease()
    with pytest.raises(StaleOwnerError):
        successor.recover_owner_restart(successor.lease(), worker_assertion=True)


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-044"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["carrier"] == "ExternalQuackOwner"
    assert manifest["canonical_extension"]["binding"] == (
        OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING
    )
    assert manifest["canonical_extension"]["entrypoint"] == (
        "ExternalQuackOwner.recover_owner_restart"
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(OWNER_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
