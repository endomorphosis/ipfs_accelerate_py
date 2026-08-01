from __future__ import annotations

import json

from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    build_task_work_contract,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import todo_vector_index
from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
    write_todo_vector_index,
)


def test_todo_vector_refresh_reprojects_resources_into_bundle_and_dependency_graphs(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## FVT-024 Verify an optional advisor fallback

- Status: todo
- Priority: P1
- Track: formal-verification
- Bundle: objective/formal-verification/advisor
- Canonical task CID: cid-fvt-024
- Resource class: cpu-medium
- Resources: cpu-medium, prover-lean
""",
        encoding="utf-8",
    )
    bundle_index_path = repo / "bundles" / "index.json"
    bundle_index_path.parent.mkdir()
    stale_node = {
        "task_cid": "cid-fvt-024",
        "task_id": "FVT-024",
        "goal_id": "",
        "status": "todo",
        "objective_priority": 8,
        "created_at_ms": 0,
        "estimated_duration": 1,
        "metadata": {
            "task_id": "FVT-024",
            "canonical_task_cid": "cid-fvt-024",
            "resource_class": "gpu-advisor-optional",
            "resources": ["gpu-advisor-optional"],
        },
    }
    bundle_index_path.write_text(
        json.dumps(
            {
                "bundles": {
                    "objective/formal-verification/advisor": {
                        "tasks": [
                            {
                                "task_id": "FVT-024",
                                "canonical_task_cid": "cid-fvt-024",
                                "status": "todo",
                                "resource_class": "gpu-advisor-optional",
                                "resources": ["gpu-advisor-optional"],
                            }
                        ]
                    }
                },
                "dependency_dag": {
                    "nodes": {"cid-fvt-024": stale_node},
                    "edges": [],
                },
                "task_dependency_graph": {
                    "nodes": {"cid-fvt-024": stale_node},
                    "edges": [],
                },
            }
        ),
        encoding="utf-8",
    )

    vector_payload = write_todo_vector_index(
        repo_root=repo,
        todo_path=todo_path,
        index_path=repo / "bundles" / "todo_vector_index.json",
        task_header_prefix="## FVT-",
        bundle_index_path=bundle_index_path,
    )

    assert vector_payload["records"][0]["resource_class"] == "cpu-medium"
    assert vector_payload["records"][0]["resources"] == [
        "cpu-medium",
        "prover-lean",
    ]

    refreshed = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    task = refreshed["bundles"]["objective/formal-verification/advisor"]["tasks"][0]
    assert task["resource_class"] == "cpu-medium"
    assert task["resources"] == ["cpu-medium", "prover-lean"]
    assert refreshed["dependency_dag"] == refreshed["task_dependency_graph"]
    graph_node = refreshed["task_dependency_graph"]["nodes"][
        task["canonical_task_cid"]
    ]
    assert graph_node["metadata"]["resource_class"] == "cpu-medium"
    assert graph_node["metadata"]["resources"] == ["cpu-medium", "prover-lean"]


def test_todo_vector_refresh_preserves_all_bundle_contract_ids_and_timestamp(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## FVT-024 Verify an optional advisor fallback

- Status: todo
- Priority: P1
- Track: formal-verification
- Bundle: objective/formal-verification/advisor
- Goal ID: FVT-G024
- Acceptance: The advisor fallback is validated.
- Effects: emit advisor evidence
- Dependencies: FVT-023
- Outputs: src/advisor.py
- Resource class: cpu-medium

## FVT-025 Verify the advisor replay guard

- Status: todo
- Priority: P1
- Track: formal-verification
- Bundle: objective/formal-verification/advisor
- Goal ID: FVT-G025
- Acceptance: The replay guard rejects stale evidence.
- Effects: emit replay evidence
- Dependencies: FVT-024
- Outputs: src/replay.py
- Resource class: cpu-small
""",
        encoding="utf-8",
    )
    bundle_index_path = repo / "bundles" / "index.json"
    vector_index_path = repo / "bundles" / "todo_vector_index.json"
    bundle_index_path.parent.mkdir()
    bundle_index_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-01T00:00:00+00:00",
                "bundles": {
                    "objective/formal-verification/advisor": {
                        "tasks": [
                            {
                                "task_id": "FVT-024",
                                "goal_id": "FVT-G024",
                                "acceptance_subset": [
                                    "The admitted advisor has independent evidence."
                                ],
                                "preconditions": ["the advisor is installed"],
                                "token_class": "medium",
                                "status": "todo",
                            },
                            {
                                "task_id": "FVT-025",
                                "goal_id": "FVT-G025",
                                "acceptance_subset": [
                                    "The admitted replay guard is fail closed."
                                ],
                                "preconditions": ["the replay lock is current"],
                                "token_class": "small",
                                "status": "todo",
                            },
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    refreshed_at = "2026-08-01T12:34:56+00:00"
    monkeypatch.setattr(todo_vector_index, "utc_now", lambda: refreshed_at)

    returned = write_todo_vector_index(
        repo_root=repo,
        todo_path=todo_path,
        index_path=vector_index_path,
        task_header_prefix="## FVT-",
        bundle_index_path=bundle_index_path,
    )

    vector_payload = json.loads(vector_index_path.read_text(encoding="utf-8"))
    bundle_payload = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    vector_tasks = {
        record["task_id"]: record for record in vector_payload["records"]
    }
    bundle_tasks = {
        task["task_id"]: task
        for bundle in bundle_payload["bundles"].values()
        for task in bundle["tasks"]
    }

    assert set(vector_tasks) == set(bundle_tasks) == {"FVT-024", "FVT-025"}
    for task_id, vector_task in vector_tasks.items():
        bundle_task = bundle_tasks[task_id]
        assert vector_task["work_contract"] == bundle_task["work_contract"]
        assert (
            vector_task["work_contract_id"]
            == bundle_task["work_contract_id"]
        )
        assert (
            vector_task["task_work_contract"]
            == bundle_task["task_work_contract"]
        )
        assert (
            vector_task["task_work_contract_id"]
            == bundle_task["task_work_contract_id"]
        )
        assert build_task_work_contract(vector_task).verify_integrity()
        assert build_task_work_contract(bundle_task).verify_integrity()

    assert returned["generated_at"] == refreshed_at
    assert vector_payload["generated_at"] == refreshed_at
    assert bundle_payload["generated_at"] == refreshed_at
