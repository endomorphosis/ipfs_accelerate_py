"""EAAEF-085: overlapping writes/effects serialize; disjoint work may run in parallel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.planning.external_conflict_graph import (
    ConflictGraph,
)
from ipfs_accelerate_py.agent_supervisor.planning.external_frontier import (
    FrontierError,
    FrontierTask,
    select_frontier,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "parallel_frontier.json"
)
ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-qualification-artifact@1"
)
PRODUCER_ARGV = (
    "python3",
    "-m",
    "pytest",
    "-q",
    "test/integration/test_external_agent_parallel_frontier.py",
)
RECEIPT_FIELDS = {
    "artifact_cid",
    "disjoint_frontier_contract",
    "evidence_mode",
    "live_concurrency_invoked",
    "live_runtime_invoked",
    "observed_live_task_count",
    "overlap_serialized_by_contract",
    "producer_argv",
    "producer_source_cid",
    "production_qualification_claimed",
    "qualification_scope",
    "qualification_status",
    "resource_budget_contract_validated",
    "schema",
    "task_completion_claimed",
    "task_id",
    "unqualified_requirements",
}


def _producer_source_cid() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_receipt(payload: dict[str, object]) -> None:
    assert set(payload) == RECEIPT_FIELDS
    assert payload["schema"] == ARTIFACT_SCHEMA
    assert payload["task_id"] == "EAAEF-085"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["qualification_scope"] == "offline_parallel_frontier_contract_only"
    assert payload["qualification_status"] == "not_live_qualified"
    assert payload["task_completion_claimed"] is False
    assert payload["production_qualification_claimed"] is False
    assert payload["live_runtime_invoked"] is False
    assert payload["live_concurrency_invoked"] is False
    assert payload["observed_live_task_count"] == 0
    assert payload["producer_argv"] == list(PRODUCER_ARGV)
    assert payload["producer_source_cid"] == _producer_source_cid()
    unsealed = dict(payload)
    artifact_cid = unsealed.pop("artifact_cid")
    assert artifact_cid == content_identity(unsealed)


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    sealed = {
        **payload,
        "producer_argv": list(PRODUCER_ARGV),
        "producer_source_cid": _producer_source_cid(),
    }
    sealed["artifact_cid"] = content_identity(sealed)
    _validate_receipt(sealed)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sealed


def test_overlapping_writes_or_same_effects_conflict() -> None:
    overlapping_writes = (
        FrontierTask("write-a", (), ("shared.py",), ("edit",), 100),
        FrontierTask("write-b", (), ("shared.py",), ("format",), 100),
    )
    derived_writes = ConflictGraph.derive(
        overlapping_writes[0].as_scope(),
        overlapping_writes[1].as_scope(),
    )
    assert derived_writes.conflicts is True
    frontier_writes = select_frontier(overlapping_writes, cpu_budget=1000)
    assert frontier_writes["task_ids"] == ["write-a"]

    same_effects = (
        FrontierTask("fx-a", (), ("a.py",), ("merge",), 100),
        FrontierTask("fx-b", (), ("b.py",), ("merge",), 100),
    )
    derived_effects = ConflictGraph.derive(
        same_effects[0].as_scope(),
        same_effects[1].as_scope(),
    )
    assert derived_effects.conflicts is True
    frontier_effects = select_frontier(same_effects, cpu_budget=1000)
    assert frontier_effects["task_ids"] == ["fx-a"]


def test_distinct_effects_and_writes_can_be_parallel() -> None:
    tasks = (
        FrontierTask("left", (), ("left.py",), ("edit-left",), 1000),
        FrontierTask("right", (), ("right.py",), ("edit-right",), 1000),
    )
    derived = ConflictGraph.derive(tasks[0].as_scope(), tasks[1].as_scope())
    assert derived.conflicts is False
    frontier = select_frontier(tasks, cpu_budget=3000)
    assert frontier["task_ids"] == ["left", "right"]
    assert frontier["cpu_used"] == 2000


def test_cpu_budget_and_completed_deps_are_required() -> None:
    tasks = (
        FrontierTask("root", (), ("root.py",), ("edit-root",), 1500),
        FrontierTask("child", ("root",), ("child.py",), ("edit-child",), 500),
        FrontierTask("other", (), ("other.py",), ("edit-other",), 1500),
    )
    tight = select_frontier(tasks, cpu_budget=1500)
    assert "child" not in tight["task_ids"]
    assert tight["cpu_used"] <= 1500
    assert set(tight["task_ids"]) <= {"other", "root"}
    assert len(tight["task_ids"]) == 1
    after_root = select_frontier(tasks, cpu_budget=2000, completed_ids=("root",))
    assert "root" not in after_root["task_ids"]
    assert "child" in after_root["task_ids"]
    assert after_root["cpu_used"] <= 2000
    with pytest.raises(FrontierError, match="cpu_budget"):
        select_frontier(tasks, cpu_budget=0)


def test_write_offline_parallel_frontier_receipt() -> None:
    overlapping = (
        FrontierTask("write-a", (), ("shared.py",), ("edit",), 100),
        FrontierTask("write-b", (), ("shared.py",), ("format",), 100),
    )
    serialized = select_frontier(overlapping, cpu_budget=1000)
    assert serialized["task_ids"] == ["write-a"]

    disjoint = (
        FrontierTask("left", (), ("left.py",), ("edit-left",), 1000),
        FrontierTask("right", (), ("right.py",), ("edit-right",), 1000),
    )
    parallel = select_frontier(disjoint, cpu_budget=2000)
    assert parallel["task_ids"] == ["left", "right"]
    assert parallel["cpu_used"] == 2000

    receipt = _write_receipt(
        {
            "schema": ARTIFACT_SCHEMA,
            "task_id": "EAAEF-085",
            "evidence_mode": "contract_fail_closed",
            "qualification_scope": "offline_parallel_frontier_contract_only",
            "qualification_status": "not_live_qualified",
            "task_completion_claimed": False,
            "production_qualification_claimed": False,
            "live_runtime_invoked": False,
            "live_concurrency_invoked": False,
            "observed_live_task_count": 0,
            "overlap_serialized_by_contract": True,
            "disjoint_frontier_contract": ["left", "right"],
            "resource_budget_contract_validated": True,
            "unqualified_requirements": [
                "live_one-result_verification",
                "live_resource_enforcement",
                "live_stale_fence_rejection",
            ],
        }
    )
    _validate_receipt(receipt)
