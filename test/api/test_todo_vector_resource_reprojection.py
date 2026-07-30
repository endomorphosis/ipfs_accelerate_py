from __future__ import annotations

import json

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
