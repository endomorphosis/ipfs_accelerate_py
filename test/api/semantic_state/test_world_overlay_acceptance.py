"""LGSWF-013 world-overlay integration acceptance."""

from __future__ import annotations

import ipfs_accelerate_py.agent_supervisor.semantic_state as semantic_state
from ipfs_accelerate_py.agent_supervisor.semantic_state import (
    SupervisorWorldView,
    build_world_snapshot,
    parse_world_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_builder import (
    current_authority_fixture,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.world_view import (
    required_query_names,
)


def test_public_imports_are_unique_and_lazy() -> None:
    import importlib
    import sys

    names = [
        "SupervisorWorldView",
        "build_world_snapshot",
        "parse_world_snapshot",
        "persist_semantic_baseline",
    ]
    assert len(set(names)) == len(names)
    prefix = "ipfs_accelerate_py.agent_supervisor.semantic_state"
    for key in list(sys.modules):
        if key == prefix or key.startswith(prefix + "."):
            sys.modules.pop(key)
    pkg = importlib.import_module(prefix)
    assert "world_snapshot_contracts" not in pkg.__dict__
    assert "world_snapshot_builder" not in pkg.__dict__
    assert "world_view" not in pkg.__dict__
    for name in names:
        assert name in pkg.__all__
        assert callable(getattr(pkg, name))


def test_current_fixture_schedules_and_stale_mismatch_fail_closed() -> None:
    admitted = build_world_snapshot(current_authority_fixture())
    assert admitted["schedulable"] is True
    snapshot = parse_world_snapshot(admitted["snapshot"])
    assert snapshot["snapshot_cid"] == admitted["snapshot"]["snapshot_cid"]

    stale = current_authority_fixture()
    stale["accepted_plan_root"] = {**stale["accepted_plan_root"], "status": "stale"}
    assert build_world_snapshot(stale)["schedulable"] is False

    mismatch = current_authority_fixture()
    mismatch["policy_root"] = {
        **mismatch["policy_root"],
        "repository_id": "other-repository",
    }
    assert build_world_snapshot(mismatch)["schedulable"] is False


def test_read_queries_cover_required_list() -> None:
    admitted = build_world_snapshot(current_authority_fixture())
    view = SupervisorWorldView(
        admitted["snapshot"],
        {"tasks": {"LGSWF-013": {"status": "ready"}}},
    )
    assert view.query_matrix() == required_query_names()
    assert view.task_state("LGSWF-013")["found"] is True
    assert view.goal_state("missing")["reason"] == "unknown-reference"
