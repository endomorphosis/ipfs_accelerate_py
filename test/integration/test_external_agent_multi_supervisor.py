"""EAAEF-141: three-supervisor / eight-worker source contract.

This source harness observes typed worker leases, one disposable exclusive
owner, a conflict-free frontier, and stale-fence rejection.  It does not claim
that eight live containers ran.  The separate board-receipt test requires
independent container and process observations before accepting that claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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
WORKER_COUNT = 8
ROLES = ("analysis", "implementation", "verification")
BOARD_NAMESPACE = "external-agent-autonomous-execution-fabric-v1"
SHARD_ID = "eaaef-141-disposable-multi-supervisor-shard"
STORE_ID = "eaaef-141-control"
RECEIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/architecture/external_agent_autonomous_execution_fabric/receipts/multi_supervisor.json"
)


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


def _validate_current_receipt(payload: object) -> None:
    assert isinstance(payload, Mapping)
    assert payload.get("schema") == "qualification-receipt@1"
    assert payload.get("task_id") == "EAAEF-141"
    assert payload.get("evidence_mode") != "contract_fail_closed"

    encoded = json.dumps(payload, sort_keys=True)
    assert "owner:exclusive-quack" not in encoded
    assert "in_memory_ExternalQuackOwner" not in encoded

    owner_evidence = payload.get("owner_evidence")
    assert isinstance(owner_evidence, Mapping)
    observed_server_id = str(owner_evidence.get("server_id") or "")
    assert observed_server_id.startswith("server:")
    assert owner_evidence.get("backing_owner_interface") == "QuackStateServer@1"
    assert payload.get("exclusive_write_owner") == observed_server_id

    supervisor_observations = payload.get("supervisor_observations")
    assert isinstance(supervisor_observations, list)
    assert len(supervisor_observations) == len(ROLES)
    observed_roles: set[str] = set()
    for observation in supervisor_observations:
        assert isinstance(observation, Mapping)
        observed_roles.add(str(observation.get("role") or ""))
        assert str(observation.get("process_birth_id") or "")
        assert str(observation.get("evidence_cid") or "")
    assert observed_roles == set(ROLES)

    worker_observations = payload.get("worker_observations")
    assert isinstance(worker_observations, list)
    started_workers = []
    for observation in worker_observations:
        assert isinstance(observation, Mapping)
        assert str(observation.get("worker_id") or "")
        assert str(observation.get("container_id") or "")
        assert str(observation.get("lease_cid") or "")
        assert str(observation.get("evidence_cid") or "")
        if observation.get("started") is True:
            started_workers.append(observation)
    assert len(worker_observations) == WORKER_COUNT
    assert len(started_workers) == WORKER_COUNT
    assert payload.get("worker_lease_count") == len(worker_observations)
    assert payload.get("docker_workers_started") == len(started_workers)
    assert payload.get("live_eight_container_qualification") is (
        len(started_workers) == WORKER_COUNT
    )


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
    try:
        owner = _owner(server)
        lease = owner.lease()
        assert lease.owner_id == identity.server_id
        assert owner.operational_table_exposed is False
        assert owner.production_admitted is False

        duplicate = _server(tmp_path)
        try:
            with pytest.raises(
                QuackStateServerOwnershipError,
                match="second state-owner refused",
            ):
                duplicate.start()
        finally:
            duplicate.stop()
        assert owner.assert_current(lease) == lease

        with pytest.raises(RetiredInMemoryOwnerError) as retired:
            issue_envelope(
                operation="put",
                key="frontier-claim",
                value={"owner": identity.server_id, "workers": WORKER_COUNT},
                principal_id=identity.server_id,
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
        selected_overlap = [
            task_id for task_id in serialized["task_ids"] if task_id in overlapping_ids
        ]
        assert len(selected_overlap) == 1
        stale = owner.lease()
    finally:
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


def test_board_declared_qualification_receipt_is_current() -> None:
    assert RECEIPT_PATH.is_file(), f"EAAEF-141 board-declared receipt is missing: {RECEIPT_PATH}"
    _validate_current_receipt(json.loads(RECEIPT_PATH.read_text(encoding="utf-8")))
