from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import (
    BundleLaneSpec,
    launch_bundle_lanes,
    plan_bundle_lanes,
)
from ipfs_accelerate_py.agent_supervisor.lease_coordination import (
    DependencyNotReadyError,
    LeaseCoordinator,
    LeaseGrant,
    adapt_goal_bundle,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    build_bundle_task_payloads,
    critical_path_schedule,
    materialize_task_dependency_dag,
)


def test_materialized_dag_records_every_prerequisite_kind_with_provenance() -> None:
    tasks = [
        {
            "task_id": "BUILD-001",
            "canonical_task_cid": "cid-build",
            "goal_id": "GOAL-BUILD",
            "provides_imports": ["project.runtime"],
            "provides_interfaces": ["RuntimeAPI"],
            "outputs": ["build/runtime.json"],
            "provides_migrations": ["schema-v2"],
            "provides_validations": ["runtime-green"],
        },
        {
            "task_id": "USE-002",
            "canonical_task_cid": "cid-use",
            "parent_goal_ids": ["GOAL-BUILD"],
            "import_dependencies": ["project.runtime"],
            "interface_dependencies": ["RuntimeAPI"],
            "input_dependencies": ["build/runtime.json"],
            "migration_dependencies": ["schema-v2"],
            "validation_dependencies": ["runtime-green"],
        },
    ]

    graph = materialize_task_dependency_dag(tasks, now=10_000)

    assert {edge.kind for edge in graph.edges} == {
        "goal",
        "import",
        "interface",
        "output_input",
        "migration",
        "validation",
    }
    assert {(edge.source_task_cid, edge.target_task_cid) for edge in graph.edges} == {
        ("cid-build", "cid-use")
    }
    assert all(
        edge.provenance["field"] and edge.provenance["value"] for edge in graph.edges
    )
    assert all(edge.provenance["resolution"] for edge in graph.edges)


def test_critical_path_schedule_uses_receipts_and_all_priority_dimensions() -> None:
    tasks = [
        {
            "task_id": "A",
            "canonical_task_cid": "cid-a",
            "estimated_duration": 2,
            "created_at_ms": 1_000,
            "objective_priority": 4,
        },
        {
            "task_id": "B",
            "canonical_task_cid": "cid-b",
            "depends_on": ["A"],
            "estimated_duration": 3,
        },
        {
            "task_id": "C",
            "canonical_task_cid": "cid-c",
            "depends_on": ["B"],
            "estimated_duration": 5,
        },
        {
            "task_id": "SHORT",
            "canonical_task_cid": "cid-short",
            "estimated_duration": 1,
            "created_at_ms": 9_000,
            "objective_priority": 1,
        },
    ]
    graph = materialize_task_dependency_dag(tasks, now=11_000)
    records = {record.task_id: record for record in graph.schedule}

    assert graph.schedule[0].task_id == "A"
    assert records["A"].claimable is True
    assert records["A"].critical_path_length == 10
    assert records["A"].slack == 0
    assert records["A"].downstream_unlock_value == 2
    assert records["A"].age_seconds == 10
    assert records["A"].objective_priority == 4
    assert records["A"].score > records["SHORT"].score
    assert records["B"].claimable is False
    assert records["B"].blocking_task_cids == ["cid-a"]

    # A mutable task status is not merge proof.  Only a successful receipt
    # makes its dependent task schedulable.
    completed_status = materialize_task_dependency_dag(
        [
            {**task, "status": "completed"} if task["task_id"] == "A" else task
            for task in tasks
        ],
        now=11_000,
    )
    assert (
        next(
            record for record in completed_status.schedule if record.task_id == "B"
        ).claimable
        is False
    )
    with_receipt = critical_path_schedule(
        graph,
        merge_receipts=[{"task_cid": "cid-a", "status": "succeeded"}],
        now=11_000,
    )
    assert (
        next(record for record in with_receipt if record.task_id == "B").claimable
        is True
    )


def test_cycle_and_missing_dependency_emit_bounded_repairs_without_blocking_independent_work() -> (
    None
):
    graph = materialize_task_dependency_dag(
        [
            {"task_id": "A", "canonical_task_cid": "cid-a", "depends_on": ["B"]},
            {"task_id": "B", "canonical_task_cid": "cid-b", "depends_on": ["A"]},
            {
                "task_id": "MISSING",
                "canonical_task_cid": "cid-missing",
                "depends_on": ["UNKNOWN"],
            },
            {"task_id": "READY", "canonical_task_cid": "cid-ready"},
        ],
        max_repair_evidence=2,
        now=1,
    )

    assert len(graph.repair_evidence) <= 2
    assert {item.kind for item in graph.repair_evidence} <= {
        "missing_dependency",
        "dependency_cycle",
    }
    assert (
        next(record for record in graph.schedule if record.task_id == "READY").claimable
        is True
    )
    assert (
        next(
            record for record in graph.schedule if record.task_id == "MISSING"
        ).claimable
        is False
    )
    assert not next(
        record for record in graph.schedule if record.task_id == "A"
    ).claimable
    assert not next(
        record for record in graph.schedule if record.task_id == "B"
    ).claimable


def test_terminal_duplicate_alias_does_not_block_its_live_canonical_task() -> None:
    graph = materialize_task_dependency_dag(
        [
            {
                "task_id": "REF-084",
                "canonical_task_cid": "cid-finding",
                "status": "todo",
            },
            {
                "task_id": "REF-091",
                "canonical_task_cid": "cid-finding",
                "status": "completed",
            },
        ]
    )

    assert graph.repair_evidence == []
    assert graph.invalid_task_cids == []
    assert graph.nodes["cid-finding"].status == "todo"
    assert graph.nodes["cid-finding"].metadata["task_id_aliases"] == [
        "REF-084",
        "REF-091",
    ]
    assert graph.schedule[0].claimable is True


def test_bundle_lane_planner_prioritizes_ready_critical_path_before_applying_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = [
        {
            "bundle_key": "bundle/blocked",
            "todo_path": "blocked.md",
            "tasks": [{"task_id": "BLOCKED"}],
            "claimable": False,
            "schedule_rank": 0,
            "blocking_task_cids": ["cid-parent"],
            "dependency_task_cids": ["cid-parent"],
            "critical_path_length": 20,
            "dependency_repair_evidence": [{"kind": "missing_dependency"}],
            "profile_g": {"task_cid": "cid-blocked"},
        },
        {
            "bundle_key": "bundle/second",
            "todo_path": "second.md",
            "tasks": [{"task_id": "SECOND"}],
            "claimable": True,
            "schedule_rank": 2,
            "critical_path_length": 2,
            "profile_g": {"task_cid": "cid-second"},
        },
        {
            "bundle_key": "bundle/critical",
            "todo_path": "critical.md",
            "tasks": [{"task_id": "CRITICAL"}],
            "claimable": True,
            "schedule_rank": 1,
            "critical_path_length": 8,
            "slack": 0,
            "downstream_unlock_value": 3,
            "age_seconds": 20,
            "objective_priority": 9,
            "schedule_score": 8_030_920,
            "profile_g": {"task_cid": "cid-critical"},
        },
    ]
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.bundle_supervisor.build_bundle_task_payloads",
        lambda _path: payloads,
    )

    lanes = plan_bundle_lanes(
        bundle_index_path=tmp_path / "index.json",
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        worktree_root=tmp_path / "worktrees",
        log_dir=tmp_path / "logs",
        max_lanes=2,
    )

    assert [lane.bundle_key for lane in lanes] == ["bundle/critical", "bundle/second"]
    assert lanes[0].critical_path_length == 8
    assert lanes[0].downstream_unlock_value == 3
    assert lanes[0].schedule_score == 8_030_920


def test_bundle_lane_planner_skips_explicitly_excluded_execution_units(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps({"excluded_bundle_keys": ["bundle/excluded"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.bundle_supervisor.build_bundle_task_payloads",
        lambda _path: [
            {
                "bundle_key": "bundle/included",
                "todo_path": "included.md",
                "tasks": [{"task_id": "INCLUDED"}],
                "profile_g": {"task_cid": "cid-included"},
            },
            {
                "bundle_key": "bundle/excluded",
                "todo_path": "excluded.md",
                "tasks": [{"task_id": "EXCLUDED"}],
                "profile_g": {"task_cid": "cid-excluded"},
            },
        ],
    )

    lanes = plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        worktree_root=tmp_path / "worktrees",
        log_dir=tmp_path / "logs",
    )

    assert [lane.bundle_key for lane in lanes] == ["bundle/included"]


def test_bundle_index_enrichment_maps_member_edges_to_bundle_execution_cids(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "todo.md",
                "bundles": {
                    "bundle/child": {
                        "shard_path": "child.md",
                        "tasks": [
                            {
                                "task_id": "CHILD",
                                "canonical_task_cid": "cid-child-member",
                                "depends_on": ["ROOT"],
                                "estimated_duration": 2,
                            }
                        ],
                    },
                    "bundle/root": {
                        "shard_path": "root.md",
                        "tasks": [
                            {
                                "task_id": "ROOT",
                                "canonical_task_cid": "cid-root-member",
                                "estimated_duration": 3,
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payloads = build_bundle_task_payloads(index_path)
    by_key = {payload["bundle_key"]: payload for payload in payloads}
    root = by_key["bundle/root"]
    child = by_key["bundle/child"]

    assert [payload["bundle_key"] for payload in payloads] == [
        "bundle/root",
        "bundle/child",
    ]
    assert root["claimable"] is True
    assert child["claimable"] is False
    assert child["dependency_task_cids"] == [root["canonical_task_cid"]]
    assert child["profile_g"]["task"]["dependency_task_cids"] == [
        root["canonical_task_cid"]
    ]
    # Member-level provenance remains in the embedded graph, but the lease
    # dependency is the bundle execution identity that actually gets receipts.
    assert child["tasks"][0]["dependency_task_cids"] == ["cid-root-member"]
    assert (
        child["task_dependency_graph"]["edges"][0]["source_task_cid"]
        == "cid-root-member"
    )
    assert root["schedule_rank"] < child["schedule_rank"]

    lanes = plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        worktree_root=tmp_path / "worktrees",
        log_dir=tmp_path / "logs",
    )
    assert [lane.bundle_key for lane in lanes] == ["bundle/root", "bundle/child"]
    assert lanes[1].dependency_task_cids == [
        lanes[0].queue_payload["canonical_task_cid"]
    ]


def test_truncated_graph_repairs_still_block_every_invalid_bundle(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "bundles": {
                    f"bundle/{index:03d}": {
                        "shard_path": f"{index:03d}.md",
                        "tasks": [
                            {
                                "task_id": f"TASK-{index:03d}",
                                "depends_on": [f"UNKNOWN-{index:03d}"],
                            }
                        ],
                    }
                    for index in range(65)
                }
            }
        ),
        encoding="utf-8",
    )

    payload = next(
        item
        for item in build_bundle_task_payloads(index_path)
        if item["bundle_key"] == "bundle/064"
    )

    assert payload["claimable"] is False
    assert payload["dependency_repair_evidence"][0]["kind"] == "missing_dependency"
    assert payload["dependency_repair_evidence"][0]["provenance"][
        "evidence_truncated"
    ] is True
    with LeaseCoordinator(tmp_path / "coordination.sqlite3") as coordinator:
        registered = coordinator.register_bundle(payload, created_at_ms=1)
        with pytest.raises(DependencyNotReadyError):
            coordinator.claim(
                registered["task_cid"],
                "did:web:bounded-repair.example",
                requested_lease_ms=5_000,
                now_ms=1,
            )


def _lane(tmp_path: Path, bundle: dict[str, object]) -> BundleLaneSpec:
    key = str(bundle["bundle_key"])
    adapted = adapt_goal_bundle(bundle, created_at_ms=1_783_872_000_000)
    payload = {**bundle, "profile_g": adapted}
    return BundleLaneSpec(
        bundle_key=key,
        parallel_lane=key,
        todo_path=tmp_path / f"{key.replace('/', '-')}.md",
        state_dir=tmp_path / "state" / key.replace("/", "-"),
        worktree_root=tmp_path / "worktrees" / key.replace("/", "-"),
        state_prefix=key.replace("/", "_"),
        task_ids=[str(bundle["tasks"][0]["task_id"])],  # type: ignore[index]
        conflict_policy="",
        command=["worker", key],
        log_path=tmp_path / "logs" / f"{key.replace('/', '-')}.log",
        task_cid=str(adapted["task_cid"]),
        queue_payload=payload,
        dependency_task_cids=list(bundle.get("dependency_task_cids", [])),  # type: ignore[arg-type]
    )


def test_bundle_launcher_surfaces_dependency_evidence_and_unblocks_after_successful_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prerequisite = {
        "bundle_key": "bundle/root",
        "tasks": [{"task_id": "ROOT", "goal": "build root"}],
    }
    root_lane = _lane(tmp_path, prerequisite)
    dependent = {
        "bundle_key": "bundle/child",
        "tasks": [{"task_id": "CHILD", "goal": "use root"}],
        "dependency_task_cids": [
            root_lane.queue_payload["profile_g"]["canonical_task_cid"]
        ],  # type: ignore[index]
    }
    child_lane = _lane(tmp_path, dependent)
    starts: list[list[str]] = []

    class Process:
        pid = 4321

    def fake_popen(command, **_kwargs):
        starts.append(list(command))
        return Process()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.bundle_supervisor.subprocess.Popen",
        fake_popen,
    )
    coordination_path = tmp_path / "coordination.sqlite3"

    initial = launch_bundle_lanes(
        [child_lane, root_lane],
        repo_root=tmp_path,
        coordination_path=coordination_path,
        claimant_did="did:web:planner.example",
    )
    assert initial[0]["accepted"] is False
    assert initial[0]["code"] == "G_DEPENDENCY_NOT_READY"
    assert initial[0]["dependency_evidence"]["blocked_dependency_task_cids"]
    assert initial[1]["accepted"] is True
    assert len(starts) == 1

    root_grant = LeaseGrant(**initial[1]["lease"])
    with LeaseCoordinator(coordination_path) as coordinator:
        coordinator.receipt(
            root_grant, status="succeeded", output={"merge_commit": "abc123"}
        )

    retried = launch_bundle_lanes(
        [replace(child_lane, claimable=True)],
        repo_root=tmp_path,
        coordination_path=coordination_path,
        claimant_did="did:web:planner.example",
    )
    assert retried[0]["accepted"] is True
    assert len(starts) == 2
