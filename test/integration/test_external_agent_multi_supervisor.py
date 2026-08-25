"""EAAEF-141: three-supervisor / eight-worker in-process qualification.

Live eight-container clusters are not invoked.  The live path fails closed and
the passing evidence is the in-process contract: three supervisor identities,
eight worker leases, one exclusive write owner, a conflict-free frontier, and
rejection of a stale supervisor fence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.containers.contracts import (
    IsolationPolicy,
    ResourceBounds,
    WorkerLease,
    bind_container_execution,
)
from ipfs_accelerate_py.agent_supervisor.planning.external_frontier import (
    FrontierTask,
    select_frontier,
)
from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
    ExternalQuackOwner,
    RetiredInMemoryOwnerError,
    StaleOwnerError,
    issue_envelope,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServer,
    QuackStateServerOwnershipError,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QuackDaemonGatewayError,
)

SUPERVISORS = (
    "supervisor:analysis",
    "supervisor:implementation",
    "supervisor:verification",
)
WRITE_OWNER = "owner:exclusive-quack"
WORKER_COUNT = 8
ROLES = ("analysis", "implementation", "verification")
BOARD_NAMESPACE = "external-agent-autonomous-execution-fabric-v1"
SHARD_ID = "eaaef-141-disposable-multi-supervisor-shard"
STORE_ID = "eaaef-141-control"


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _resources() -> ResourceBounds:
    return ResourceBounds(
        cpu_millicores=1000,
        ram_mib=512,
        disk_mib=1024,
        timeout_seconds=60,
        gpu_count=0,
    )


def _issue_leases() -> tuple[WorkerLease, ...]:
    leases: list[WorkerLease] = []
    for index in range(WORKER_COUNT):
        supervisor = SUPERVISORS[index % len(SUPERVISORS)]
        bound = bind_container_execution(
            image_digest=_digest(f"image-worker-{index}"),
            worktree_id=f"worktree:worker-{index}",
            task_id=f"task:worker-{index}",
            authority_id=f"authority:{supervisor}",
            worker_id=f"worker:slot-{index}",
            resources=_resources(),
            policy=IsolationPolicy(),
            fencing_token=1,
        )
        leases.append(bound.lease)
    return tuple(leases)


def _frontier_tasks() -> tuple[FrontierTask, ...]:
    tasks = [
        FrontierTask(
            task_id=f"task:worker-{index}",
            depends_on=(),
            write_scope=(f"owned/worker-{index}.py",),
            effect_scope=(f"write-worker-{index}",),
            cpu_millicores=1000,
        )
        for index in range(WORKER_COUNT)
    ]
    return tuple(tasks)


def _server(root: Path) -> QuackStateServer:
    return build_server(
        database_path=root / "control.duckdb",
        state_dir=root / "owner",
        port=0,
        repository_id="repository:eaaef-141-test",
        store_id=STORE_ID,
        secret_handle="handle:eaaef-141-test-owner",
    )


def _owner(server: QuackStateServer) -> ExternalQuackOwner:
    owner = server.bind_external_quack_owner(
        board_namespace=BOARD_NAMESPACE,
        shard_id=SHARD_ID,
    )
    assert isinstance(owner, ExternalQuackOwner)
    return owner


def _overlay_contract(**extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-overlay-receipt@1",
        "task_id": "EAAEF-141",
        "evidence_mode": "contract_fail_closed",
        "live_runtime_invoked": False,
        "live_eight_container_qualification": False,
        "docker_workers_started": 0,
        "supervisor_identities": list(SUPERVISORS),
        "supervisor_roles": list(ROLES),
        "worker_lease_count": WORKER_COUNT,
        "exclusive_write_owner": WRITE_OWNER,
    }
    payload.update(extra)
    return payload


def test_three_supervisors_eight_leases_exclusive_owner_conflict_free_frontier(
    tmp_path: Path,
) -> None:
    assert len(SUPERVISORS) == 3
    assert tuple(ROLES) == ("analysis", "implementation", "verification")

    leases = _issue_leases()
    assert len(leases) == WORKER_COUNT
    worker_ids = [lease.worker_id for lease in leases]
    worktrees = [lease.worktree_id for lease in leases]
    tasks = [lease.task_id for lease in leases]
    assert len(set(worker_ids)) == WORKER_COUNT
    assert len(set(worktrees)) == WORKER_COUNT
    assert len(set(tasks)) == WORKER_COUNT
    assert {lease.authority_id for lease in leases} == {
        f"authority:{identity}" for identity in SUPERVISORS
    }
    for lease in leases:
        assert lease.active is True
        assert lease.fencing_token == 1
        assert lease.worker_id != lease.authority_id

    server = _server(tmp_path)
    identity = server.start()
    owner = _owner(server)
    lease = owner.lease()
    assert lease.owner_id == identity.server_id
    assert owner.operational_table_exposed is False
    assert owner.production_admitted is False

    duplicate = _server(tmp_path)
    with pytest.raises(
        QuackStateServerOwnershipError,
        match="second state-owner refused",
    ):
        duplicate.start()
    assert owner.assert_current(lease) == lease

    with pytest.raises(RetiredInMemoryOwnerError) as retired:
        issue_envelope(
            operation="put",
            key="frontier-claim",
            value={"owner": WRITE_OWNER, "workers": WORKER_COUNT},
            principal_id=WRITE_OWNER,
            idempotency_key="idem:owner-1",
        )
    assert retired.value.reason_code == "in_memory_owner_retired"
    with pytest.raises(
        QuackDaemonGatewayError,
        match=EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
    ):
        owner.daemon_gateway()

    frontier = select_frontier(_frontier_tasks(), cpu_budget=16_000)
    assert frontier["task_ids"] == [f"task:worker-{index}" for index in range(WORKER_COUNT)]

    overlapping = _frontier_tasks() + (
        FrontierTask(
            task_id="task:overlap-b",
            depends_on=(),
            write_scope=("owned/worker-0.py",),
            effect_scope=("write-worker-0",),
            cpu_millicores=1000,
        ),
    )
    serialized = select_frontier(overlapping, cpu_budget=16_000)
    overlapping_ids = {"task:worker-0", "task:overlap-b"}
    selected_overlap = [task_id for task_id in serialized["task_ids"] if task_id in overlapping_ids]
    assert len(selected_overlap) == 1

    stale = owner.lease()
    server.stop()
    successor_server = _server(tmp_path)
    successor_server.start()
    try:
        successor = _owner(successor_server)
        takeover = successor.assert_successor(stale)
        assert takeover.epoch > stale.epoch
        assert takeover.fence > stale.fence
        with pytest.raises(StaleOwnerError, match="stale owner") as err:
            successor.assert_current(stale)
        assert err.value.reason_code == "stale_owner"
    finally:
        successor_server.stop()

    payload = _overlay_contract(
        worker_ids=worker_ids,
        worktree_ids=worktrees,
        frontier_task_ids=list(frontier["task_ids"]),
        overlapping_pair_accepted=selected_overlap[0],
        overlapping_pair_rejected=sorted(overlapping_ids - {selected_overlap[0]})[0],
        stale_supervisor_fence_rejected=True,
        exclusive_write_owner_epoch=takeover.epoch,
        owner_dispatch_admitted=False,
    )
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["live_runtime_invoked"] is False
    assert payload["live_eight_container_qualification"] is False
    assert payload["docker_workers_started"] == 0
    assert payload["worker_lease_count"] == 8
    assert payload["owner_dispatch_admitted"] is False
    assert payload["stale_supervisor_fence_rejected"] is True
