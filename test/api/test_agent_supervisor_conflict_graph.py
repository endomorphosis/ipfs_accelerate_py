from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import plan_bundle_lanes
from ipfs_accelerate_py.agent_supervisor.conflict_graph import (
    build_conflict_surface,
    materialize_task_conflict_graph,
)


def _edge(graph, left: str, right: str):
    pair = frozenset((left, right))
    return next(
        edge
        for edge in graph.edges
        if frozenset((edge.left_task_cid, edge.right_task_cid)) == pair
    )


def _decision(graph, left: str, right: str):
    pair = frozenset((left, right))
    return next(
        decision
        for decision in graph.decisions
        if frozenset((decision.left_task_cid, decision.right_task_cid)) == pair
    )


def test_conflict_surface_collects_every_predicted_and_observed_change_domain(tmp_path: Path) -> None:
    source = tmp_path / "src" / "runtime_router.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """from typing import Protocol

class RuntimeContract(Protocol):
    def dispatch(self, request): ...

class CapabilityRouter:
    def dispatch(self, request):
        return request
""",
        encoding="utf-8",
    )

    surface = build_conflict_surface(
        {
            "task_id": "TASK-A",
            "task_cid": "cid-a",
            # Outputs are predicted files too, rather than only being used as a
            # single path-root hint.
            "predicted_files": ["./src/runtime_router.py"],
            "outputs": ["src/runtime_router.py", "tests/test_runtime_router.py"],
            "ast_symbols": ["RuntimeContract", "CapabilityRouter.dispatch"],
            "provides_interfaces": ["RuntimeAPI@1"],
            "required_interfaces": ["RequestEnvelope@2"],
            "submodules": ["external/runtime", "./external/runtime"],
            "generated_artifacts": ["generated/runtime.schema.json"],
        },
        repo_root=tmp_path,
        changed_paths=["src/runtime_router.py", "docs/runtime.md"],
    )

    assert surface.task_id == "TASK-A"
    assert surface.task_cid == "cid-a"
    assert surface.files == ["src/runtime_router.py", "tests/test_runtime_router.py"]
    # Symbols declared by planning and symbols discovered from existing Python
    # outputs are one conflict domain.
    assert {
        "RuntimeContract",
        "RuntimeContract.dispatch",
        "CapabilityRouter",
        "CapabilityRouter.dispatch",
    }.issubset(surface.ast_symbols)
    assert surface.global_ast_symbols == ["CapabilityRouter.dispatch", "RuntimeContract"]
    assert surface.interfaces == ["RequestEnvelope@2", "RuntimeAPI@1"]
    assert surface.submodules == ["external/runtime"]
    assert surface.generated_artifacts == ["generated/runtime.schema.json"]
    assert surface.changed_paths == ["docs/runtime.md", "src/runtime_router.py"]
    assert json.loads(json.dumps(surface.to_dict()))["task_cid"] == "cid-a"


def test_conflict_graph_covers_all_surface_types_and_colors_only_blocking_edges(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": "BASE",
            "task_cid": "cid-base",
            "predicted_files": ["src/base.py"],
            "ast_symbols": ["Router.dispatch"],
            "interfaces": ["RuntimeAPI@1"],
            "submodules": ["external/runtime"],
            "generated_artifacts": ["generated/runtime.schema.json"],
        },
        {
            "task_id": "FILE",
            "task_cid": "cid-file",
            "predicted_files": ["src/base.py"],
        },
        {
            "task_id": "SYMBOL",
            "task_cid": "cid-symbol",
            "predicted_files": ["src/other_symbol.py"],
            "ast_symbols": ["Router.dispatch"],
        },
        {
            "task_id": "INTERFACE",
            "task_cid": "cid-interface",
            "predicted_files": ["src/other_interface.py"],
            "required_interfaces": ["RuntimeAPI@1"],
        },
        {
            "task_id": "SUBMODULE",
            "task_cid": "cid-submodule",
            "predicted_files": ["src/other_submodule.py"],
            "submodules": ["external/runtime"],
        },
        {
            "task_id": "GENERATED",
            "task_cid": "cid-generated",
            "predicted_files": ["src/other_generated.py"],
            "generated_artifacts": ["generated/runtime.schema.json"],
        },
        {
            "task_id": "ALLOWED",
            "task_cid": "cid-allowed",
            "predicted_files": ["src/base.py"],
        },
        {
            "task_id": "DISJOINT",
            "task_cid": "cid-disjoint",
            "predicted_files": ["docs/disjoint.md"],
        },
    ]

    graph = materialize_task_conflict_graph(
        tasks,
        repo_root=tmp_path,
        concurrency_overrides=[("cid-base", "cid-allowed")],
    )

    expected_reasons = {
        "cid-file": "files",
        "cid-symbol": "ast_symbols",
        "cid-interface": "interfaces",
        "cid-submodule": "submodules",
        "cid-generated": "generated_artifacts",
    }
    for other, surface_name in expected_reasons.items():
        edge = _edge(graph, "cid-base", other)
        assert edge.overlaps[surface_name]
        assert any(reason.startswith(f"{surface_name}:") for reason in edge.reasons)
        assert edge.weight > 0
        assert edge.explicitly_allowed is False

    allowed_edge = _edge(graph, "cid-base", "cid-allowed")
    assert allowed_edge.overlaps["files"] == ["src/base.py"]
    assert allowed_edge.explicitly_allowed is True

    colors = {assignment.task_cid: assignment.color for assignment in graph.assignments}
    # Blocking overlap is a graph edge for coloring; an explicit override is
    # retained for auditing but deliberately omitted from coloring constraints.
    for other in expected_reasons:
        assert colors["cid-base"] != colors[other]
    assert colors["cid-base"] == colors["cid-allowed"]
    assert colors["cid-base"] == colors["cid-disjoint"]

    # Planner explanations are exhaustive, not limited to pairs with an edge.
    assert len(graph.decisions) == len(list(combinations(tasks, 2)))
    assert _decision(graph, "cid-base", "cid-file").action == "separate"
    assert _decision(graph, "cid-base", "cid-allowed").action == "concurrent_override"
    assert _decision(graph, "cid-base", "cid-disjoint").action == "co_locate"
    assert all(decision.explanation.strip() for decision in graph.decisions)
    assert json.loads(json.dumps(graph.to_dict()))["assignments"]


def test_discovered_ast_names_are_file_scoped_across_disjoint_outputs(tmp_path: Path) -> None:
    for name in ("alpha.py", "beta.py"):
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "from datetime import datetime\n\nclass Service:\n    def __init__(self):\n        self.at = datetime.now()\n",
            encoding="utf-8",
        )

    graph = materialize_task_conflict_graph(
        [
            {"task_id": "ALPHA", "task_cid": "cid-alpha", "outputs": ["src/alpha.py"]},
            {"task_id": "BETA", "task_cid": "cid-beta", "outputs": ["src/beta.py"]},
        ],
        repo_root=tmp_path,
    )

    assert graph.edges == []
    assert _decision(graph, "cid-alpha", "cid-beta").action == "co_locate"


def test_file_isolated_bundle_policy_scopes_generated_ast_terms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = [
        {
            "bundle_key": f"bundle/{name}",
            "todo_path": f"{name}.md",
            "conflict_policy": (
                "serialize findings for the same file; allow independent file bundles "
                "to run concurrently"
            ),
            "tasks": [
                {
                    "task_id": name.upper(),
                    "outputs": [f"src/{name}.py"],
                    "ast_symbols": ["__init__", "datetime.datetime"],
                }
            ],
            "profile_g": {"task_cid": f"cid-{name}"},
        }
        for name in ("alpha", "beta")
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
    )

    assert len(lanes) == 2
    assert all(not lane.conflicting_task_ids for lane in lanes)


def test_bundle_lane_prefers_canonical_files_over_broad_planning_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = [
        {
            "bundle_key": "bundle/alpha",
            "todo_path": "alpha.md",
            "tasks": [
                {
                    "task_id": "ALPHA",
                    "files": ["src/alpha.py", "tests/test_alpha.py"],
                    "outputs": ["src/alpha.py", "tests/test_alpha.py"],
                    "predicted_files": [
                        "src/alpha.py",
                        "tests/test_alpha.py",
                        "data/agent_supervisor/discovery",
                        "docs/objectives.md",
                    ],
                }
            ],
            "profile_g": {"task_cid": "cid-alpha"},
        }
    ]
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.bundle_supervisor.build_bundle_task_payloads",
        lambda _path: payloads,
    )

    [lane] = plan_bundle_lanes(
        bundle_index_path=tmp_path / "index.json",
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        worktree_root=tmp_path / "worktrees",
        log_dir=tmp_path / "logs",
    )

    assert lane.conflict_surface["files"] == ["src/alpha.py", "tests/test_alpha.py"]
    metadata = lane.conflict_surface["metadata"]["metadata"]
    assert metadata["advisory_paths"] == [
        "data/agent_supervisor/discovery",
        "docs/objectives.md",
    ]


def test_shared_bookkeeping_paths_do_not_block_disjoint_bundle_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shared = [
        "data/agent_supervisor/discovery",
        "docs/objectives.md",
    ]
    payloads = [
        {
            "bundle_key": f"bundle/{name}",
            "todo_path": f"{name}.md",
            "tasks": [
                {
                    "task_id": name.upper(),
                    "files": [f"src/{name}.py"],
                    "outputs": [f"src/{name}.py"],
                    "predicted_files": [f"src/{name}.py", *shared],
                }
            ],
            "profile_g": {"task_cid": f"cid-{name}"},
        }
        for name in ("alpha", "beta")
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
    )
    by_key = {lane.bundle_key: lane for lane in lanes}

    assert by_key["bundle/alpha"].conflicting_task_ids == []
    assert by_key["bundle/beta"].conflicting_task_ids == []
    assert by_key["bundle/alpha"].conflict_color == by_key["bundle/beta"].conflict_color
    assert by_key["bundle/alpha"].conflict_surface["files"] == ["src/alpha.py"]
    assert by_key["bundle/beta"].conflict_surface["files"] == ["src/beta.py"]


def test_same_canonical_file_still_blocks_bundle_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = [
        {
            "bundle_key": f"bundle/{name}",
            "todo_path": f"{name}.md",
            "tasks": [
                {
                    "task_id": name.upper(),
                    "files": ["src/shared.py"],
                    "predicted_files": [
                        "src/shared.py",
                        "data/agent_supervisor/discovery",
                    ],
                }
            ],
            "profile_g": {"task_cid": f"cid-{name}"},
        }
        for name in ("alpha", "beta")
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
    )
    by_key = {lane.bundle_key: lane for lane in lanes}

    assert "bundle/beta" in by_key["bundle/alpha"].conflicting_task_ids
    assert "bundle/alpha" in by_key["bundle/beta"].conflicting_task_ids
    assert by_key["bundle/alpha"].conflict_color != by_key["bundle/beta"].conflict_color
    assert any(
        "src/shared.py" in decision["explanation"]
        for decision in by_key["bundle/alpha"].conflict_decisions
        if decision["action"] == "separate"
    )


def test_bundle_ast_symbols_are_file_local_unless_explicitly_global(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = [
        {
            "bundle_key": f"bundle/{name}",
            "todo_path": f"{name}.md",
            "tasks": [
                {
                    "task_id": name.upper(),
                    "files": [f"src/{name}.py"],
                    "ast_symbols": ["__post_init__", "_digest"],
                    **(
                        {"global_ast_symbols": ["GlobalRegistry"]}
                        if name.startswith("global-")
                        else {}
                    ),
                }
            ],
            "profile_g": {"task_cid": f"cid-{name}"},
        }
        for name in ("alpha", "beta", "global-alpha", "global-beta")
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
    )
    by_key = {lane.bundle_key: lane for lane in lanes}

    assert "bundle/beta" not in by_key["bundle/alpha"].conflicting_task_ids
    assert by_key["bundle/alpha"].conflict_surface["global_ast_symbols"] == []
    assert by_key["bundle/alpha"].conflict_surface["ast_symbols"] == [
        "__post_init__",
        "_digest",
    ]
    assert "bundle/global-beta" in by_key["bundle/global-alpha"].conflicting_task_ids


def test_non_python_path_overlap_does_not_inflate_discovered_ast_conflict(
    tmp_path: Path,
) -> None:
    for name in ("alpha.py", "beta.py"):
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "class Record:\n    def __post_init__(self):\n        return None\n",
            encoding="utf-8",
        )

    graph = materialize_task_conflict_graph(
        [
            {
                "task_id": "ALPHA",
                "task_cid": "cid-alpha",
                "outputs": ["src/alpha.py", "docs/shared.md"],
            },
            {
                "task_id": "BETA",
                "task_cid": "cid-beta",
                "outputs": ["src/beta.py", "docs/shared.md"],
            },
        ],
        repo_root=tmp_path,
    )

    edge = _edge(graph, "cid-alpha", "cid-beta")
    assert edge.overlaps["files"] == ["docs/shared.md"]
    assert "ast_symbols" not in edge.overlaps


def test_actual_diffs_and_conflict_receipts_increase_future_pair_weight(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": "TASK-A",
            "task_cid": "cid-a",
            "predicted_files": ["src/planned_shared.py"],
        },
        {
            "task_id": "TASK-B",
            "task_cid": "cid-b",
            "predicted_files": ["src/planned_shared.py"],
        },
    ]
    baseline = materialize_task_conflict_graph(tasks, repo_root=tmp_path)
    baseline_edge = _edge(baseline, "cid-a", "cid-b")

    learned = materialize_task_conflict_graph(
        tasks,
        repo_root=tmp_path,
        branch_diffs={
            "cid-a": ["src/actually_shared.py", "src/a_only.py"],
            "cid-b": ["src/actually_shared.py", "src/b_only.py"],
        },
        conflict_receipts=[
            {
                "task_cids": ["cid-a", "cid-b"],
                "status": "conflict",
                "reason": "content_conflict",
                "paths": ["src/actually_shared.py"],
                "count": 2,
            }
        ],
    )
    learned_edge = _edge(learned, "cid-a", "cid-b")
    # Learned state is an explicit planner input, so subsequent planning runs
    # retain the receipt signal even before either branch has a new diff.
    future = materialize_task_conflict_graph(
        tasks,
        repo_root=tmp_path,
        history=learned.history.to_dict(),
    )
    future_edge = _edge(future, "cid-a", "cid-b")

    assert learned.surfaces["cid-a"].changed_paths == ["src/a_only.py", "src/actually_shared.py"]
    assert learned.surfaces["cid-b"].changed_paths == ["src/actually_shared.py", "src/b_only.py"]
    assert learned_edge.weight > baseline_edge.weight
    assert future_edge.weight > baseline_edge.weight
    assert future_edge.overlaps["historical_task_pair"] == ["cid-a<->cid-b"]
    assert learned_edge.observed_weight > baseline_edge.observed_weight
    assert learned_edge.overlaps["changed_paths"] == ["src/actually_shared.py"]
    assert learned_edge.overlaps["historical_task_pair"] == ["cid-a<->cid-b"]
    decision = _decision(learned, "cid-a", "cid-b")
    assert decision.action == "separate"
    assert "src/actually_shared.py" in decision.explanation
    assert "historical_task_pair" in decision.explanation


def test_duplicate_canonical_tasks_are_coalesced_conservatively(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": "REF-064",
            "task_cid": "cid-shared",
            "predicted_files": ["src/later.py"],
            "ast_symbols": ["Later.run"],
        },
        {
            "task_id": "REF-057",
            "task_cid": "cid-shared",
            "predicted_files": ["src/earlier.py"],
            "ast_symbols": ["Earlier.run"],
        },
    ]

    graph = materialize_task_conflict_graph(
        tasks,
        repo_root=tmp_path,
        branch_diffs={
            "REF-057": ["src/observed-earlier.py"],
            "REF-064": ["src/observed-later.py"],
        },
    )

    assert list(graph.surfaces) == ["cid-shared"]
    surface = graph.surfaces["cid-shared"]
    assert surface.task_id == "REF-057"
    assert surface.files == ["src/earlier.py", "src/later.py"]
    assert surface.changed_paths == [
        "src/observed-earlier.py",
        "src/observed-later.py",
    ]
    assert {"Earlier.run", "Later.run"}.issubset(surface.ast_symbols)
    assert surface.metadata["task_id_aliases"] == ["REF-057", "REF-064"]
    assert graph.history.observation_count == 1
    assert graph.history.path_weights == {
        "src/observed-earlier.py": 1.0,
        "src/observed-later.py": 1.0,
    }


def test_bundle_lane_planner_projects_blocking_edges_but_honors_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = [
        {
            "bundle_key": "bundle/base",
            "todo_path": "base.md",
            "tasks": [{"task_id": "BASE", "outputs": ["src/shared.py"]}],
            "profile_g": {"task_cid": "cid-base"},
        },
        {
            "bundle_key": "bundle/blocked",
            "todo_path": "blocked.md",
            "tasks": [{"task_id": "BLOCKED", "outputs": ["src/shared.py"]}],
            "profile_g": {"task_cid": "cid-blocked"},
        },
        {
            "bundle_key": "bundle/allowed",
            "todo_path": "allowed.md",
            "allow_concurrent_with": ["cid-base"],
            "tasks": [{"task_id": "ALLOWED", "outputs": ["src/shared.py"]}],
            "profile_g": {"task_cid": "cid-allowed"},
        },
        {
            "bundle_key": "bundle/disjoint",
            "todo_path": "disjoint.md",
            "tasks": [{"task_id": "DISJOINT", "outputs": ["docs/disjoint.md"]}],
            "profile_g": {"task_cid": "cid-disjoint"},
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
    )
    by_key = {lane.bundle_key: lane for lane in lanes}

    assert "bundle/blocked" in by_key["bundle/base"].conflicting_task_ids
    assert "bundle/allowed" not in by_key["bundle/base"].conflicting_task_ids
    assert "bundle/disjoint" not in by_key["bundle/base"].conflicting_task_ids
    assert by_key["bundle/base"].conflict_color == by_key["bundle/allowed"].conflict_color
    assert any(
        decision["action"] == "concurrent_override"
        for decision in by_key["bundle/base"].conflict_decisions
    )


def test_bundle_lane_planner_excludes_completed_members_from_conflict_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "base.md").write_text(
        """## REF-001 Settled shared-file task

- Status: completed
- Completion: manual

## REF-002 Remaining base task

- Status: todo
- Completion: manual
""",
        encoding="utf-8",
    )
    (tmp_path / "other.md").write_text(
        """## REF-003 Other live task

- Status: todo
- Completion: manual
""",
        encoding="utf-8",
    )
    payloads = [
        {
            "bundle_key": "bundle/base",
            "todo_path": "base.md",
            "tasks": [
                {
                    "task_id": "REF-001",
                    "status": "todo",
                    "outputs": ["src/shared.py"],
                },
                {
                    "task_id": "REF-002",
                    "status": "todo",
                    "outputs": ["src/base.py"],
                },
            ],
            "profile_g": {"task_cid": "cid-base"},
        },
        {
            "bundle_key": "bundle/other",
            "todo_path": "other.md",
            "tasks": [
                {
                    "task_id": "REF-003",
                    "status": "todo",
                    "outputs": ["src/shared.py"],
                }
            ],
            "profile_g": {"task_cid": "cid-other"},
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
        task_prefix="REF-",
    )
    by_key = {lane.bundle_key: lane for lane in lanes}

    assert "bundle/other" not in by_key["bundle/base"].conflicting_task_ids
    assert by_key["bundle/base"].conflict_surface["files"] == ["src/base.py"]
