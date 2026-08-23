"""EAAEF-145: in-process end-to-end fault qualification.

Composes handoff → plan admit → frontier → quack owner apply → recovery into
a typed terminal outcome.  A live eight-container cluster is not required and
is not claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import ExternalHandoffAPI
from ipfs_accelerate_py.agent_supervisor.planning.external_frontier import (
    FrontierTask,
    select_frontier,
)
from ipfs_accelerate_py.agent_supervisor.planning.external_goal_contract import (
    ExternalGoalContract,
)
from ipfs_accelerate_py.agent_supervisor.planning.external_work_plan import ExternalWorkPlan
from ipfs_accelerate_py.agent_supervisor.planning.plan_admission import admit_plan
from ipfs_accelerate_py.agent_supervisor.runtime.external_control_recovery import recover
from ipfs_accelerate_py.agent_supervisor.runtime.external_fixed_point import terminate
from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    ExternalQuackOwner,
    issue_envelope,
)


RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "end_to_end.json"
)

WRITE_A = "ipfs_accelerate_py/agent_supervisor/handoff/adapters/codex.py"
WRITE_B = "ipfs_accelerate_py/agent_supervisor/handoff/adapters/claude.py"
SOURCE_ROOT = "sha256:" + "c" * 64
SEMANTIC_ROOT = "sha256:" + "d" * 64


def _goal():
    return ExternalGoalContract.compile(
        {
            "objective_id": "EAAEF-G150",
            "desired_outcomes": ("normalize export", "preserve identities"),
            "prohibited_outcomes": ("self_approve", "hidden_chain_of_thought"),
            "write_scope": (WRITE_A, WRITE_B),
            "authority_ceiling": "preview_only",
            "verification_requirements": ("focused pytest",),
            "proof_requirements": ("content identity",),
            "review_requirements": ("independent supervisor",),
            "completion_evidence": ("test receipt", "patch identity"),
            "timeout_seconds": 7200,
            "cpu_millicores": 4000,
            "ram_mib": 8192,
        }
    )


def _task(task_id: str, covers, write_scope, **overrides):
    payload = {
        "task_id": task_id,
        "covers": covers,
        "write_scope": write_scope,
        "depends_on": (),
        "timeout_seconds": 600,
        "cpu_millicores": 1000,
        "ram_mib": 1024,
    }
    payload.update(overrides)
    return payload


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_handoff_plan_frontier_owner_recovery_typed_terminal() -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(
        {
            "principal_id": "principal:operator",
            "worker_principal_id": "principal:worker",
            "reviewer_principal_id": "principal:reviewer",
            "session_id": "session:e2e",
            "repository_id": "repo:e2e",
            "objective_id": "objective:e2e",
            "idempotency_key": "idem:e2e-1",
        }
    )
    assert started.verdict == "admitted"
    assert started.run_status == "running"

    sequential = ExternalWorkPlan.decompose(
        _goal(),
        (
            _task("task-a", ("normalize export",), (WRITE_A,)),
            _task(
                "task-b",
                ("preserve identities",),
                (WRITE_B,),
                depends_on=("task-a",),
            ),
        ),
    )
    parallel = ExternalWorkPlan.decompose(
        _goal(),
        (
            _task("task-a", ("normalize export",), (WRITE_A,)),
            _task("task-b", ("preserve identities",), (WRITE_B,)),
        ),
    )
    admitted = admit_plan((sequential, parallel))
    assert admitted.to_dict()["verdict"] == "admitted"
    assert admitted.admitted.content_id == parallel.content_id

    frontier = select_frontier(
        (
            FrontierTask("task-a", (), (WRITE_A,), ("write-a",), 1000),
            FrontierTask("task-b", (), (WRITE_B,), ("write-b",), 1000),
        ),
        cpu_budget=4000,
    )
    assert set(frontier["task_ids"]) == {"task-a", "task-b"}

    owner = ExternalQuackOwner("owner:e2e", shard_id="e2e-shard")
    lease = owner.lease()
    apply_receipt = owner.apply(
        issue_envelope(
            operation="put",
            key="e2e-run",
            value={"run_id": started.run_id, "plan_id": admitted.admitted_id},
            principal_id="principal:operator",
            idempotency_key="idem:e2e-apply",
        ),
        owner_id=lease.owner_id,
        epoch=lease.epoch,
    )
    assert apply_receipt["status"] == "applied"

    recovery = recover(
        current_epoch=lease.epoch,
        backup_epoch=lease.epoch,
        duplicate=False,
        ducklake_available=False,
    )
    assert recovery["accepted_stale_write"] is False

    approved = api.approve(
        {
            "principal_id": "principal:operator",
            "worker_principal_id": "principal:worker",
            "reviewer_principal_id": "principal:reviewer",
            "run_id": started.run_id,
            "authority_id": started.authority_id,
            "session_id": "session:e2e",
        }
    )
    assert approved.run_status == "approved"

    terminal = terminate(
        goals_complete=True,
        tests_current=True,
        proofs_current=True,
        invalidations_empty=True,
        merge_queue_empty=True,
        claims_empty=True,
        source_root=SOURCE_ROOT,
        semantic_root=SEMANTIC_ROOT,
    )
    assert terminal["terminal"] == "completed"

    payload = _write_receipt(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-overlay-receipt@1",
            "task_id": "EAAEF-145",
            "evidence_mode": "contract_fail_closed",
            "live_runtime_invoked": False,
            "live_eight_container_qualification": False,
            "live_cluster_required": False,
            "stages": [
                "handoff",
                "plan_admit",
                "frontier",
                "quack_owner_apply",
                "recovery",
                "typed_terminal",
            ],
            "run_id": started.run_id,
            "admitted_plan_id": admitted.admitted_id,
            "frontier_task_ids": list(frontier["task_ids"]),
            "terminal": terminal["terminal"],
            "run_status": approved.run_status,
            "accepted_stale_write": False,
        }
    )
    saved = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert saved["evidence_mode"] == "contract_fail_closed"
    assert saved["live_runtime_invoked"] is False
    assert saved["live_eight_container_qualification"] is False
    assert saved["terminal"] == "completed"
    assert payload["stages"][-1] == "typed_terminal"
