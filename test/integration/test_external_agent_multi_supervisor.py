"""EAAEF-141: three-supervisor / eight-worker in-process qualification.

Live eight-container clusters are not invoked.  The live path fails closed and
the passing evidence is the in-process contract: three supervisor identities,
eight worker leases, one exclusive write owner, a conflict-free frontier, and
rejection of a stale supervisor fence.
"""

from __future__ import annotations

import hashlib
import json
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
    DuplicateOwnerError,
    ExternalQuackOwner,
    StaleOwnerError,
    issue_envelope,
)


RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "multi_supervisor.json"
)

SUPERVISORS = (
    "supervisor:analysis",
    "supervisor:implementation",
    "supervisor:verification",
)
WRITE_OWNER = "owner:exclusive-quack"
WORKER_COUNT = 8
ROLES = ("analysis", "implementation", "verification")


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


def _overlay_receipt(**extra: object) -> dict[str, object]:
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
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_three_supervisors_eight_leases_exclusive_owner_conflict_free_frontier() -> None:
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

    owner = ExternalQuackOwner(WRITE_OWNER, shard_id="multi-supervisor-shard")
    lease = owner.lease()
    assert lease.owner_id == WRITE_OWNER
    assert owner.operational_table_exposed is False
    with pytest.raises(DuplicateOwnerError, match="second owner"):
        owner.claim(SUPERVISORS[0], epoch=lease.epoch)

    applied = owner.apply(
        issue_envelope(
            operation="put",
            key="frontier-claim",
            value={"owner": WRITE_OWNER, "workers": WORKER_COUNT},
            principal_id=WRITE_OWNER,
            idempotency_key="idem:owner-1",
        ),
        owner_id=WRITE_OWNER,
        epoch=lease.epoch,
    )
    assert applied["status"] == "applied"
    assert owner.get("frontier-claim")["owner"] == WRITE_OWNER

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
    takeover = owner.failover(SUPERVISORS[1])
    assert takeover.epoch == stale.epoch + 1
    assert takeover.fence == stale.fence + 1
    with pytest.raises(StaleOwnerError, match="stale owner") as err:
        owner.apply(
            issue_envelope(
                operation="put",
                key="stale-write",
                value={"status": "hijack"},
                principal_id=stale.owner_id,
                idempotency_key="idem:stale",
            ),
            owner_id=stale.owner_id,
            epoch=stale.epoch,
        )
    assert err.value.reason_code == "stale_owner"
    assert owner.get("stale-write") is None

    payload = _overlay_receipt(
        worker_ids=worker_ids,
        worktree_ids=worktrees,
        frontier_task_ids=list(frontier["task_ids"]),
        overlapping_pair_accepted=selected_overlap[0],
        overlapping_pair_rejected=sorted(overlapping_ids - {selected_overlap[0]})[0],
        stale_supervisor_fence_rejected=True,
        exclusive_write_owner_epoch=takeover.epoch,
    )
    saved = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert saved["evidence_mode"] == "contract_fail_closed"
    assert saved["live_runtime_invoked"] is False
    assert saved["live_eight_container_qualification"] is False
    assert saved["docker_workers_started"] == 0
    assert saved["worker_lease_count"] == 8
    assert payload["stale_supervisor_fence_rejected"] is True
