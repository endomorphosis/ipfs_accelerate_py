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
