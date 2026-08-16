"""Focused SupervisorWorldView@1 purity and query checks."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.world_view import (
    SupervisorWorldView,
    WorldViewError,
    required_query_names,
)


def _snapshot() -> dict:
    try:
        from ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_contracts import (
            example_current_snapshot,
        )

        return dict(example_current_snapshot())
    except ImportError:
        digest = "ab" * 32
        return {
            "schema": "lgswf/supervisor-world-snapshot@1",
            "interface": "SupervisorWorldSnapshot@1",
            "repository_id": "ipfs_accelerate_py",
            "components": {
                "repository": {
                    "status": "current",
                    "cid": f"sha256:{digest}",
                    "owner": "ipfs_accelerate_py",
                }
            },
            "coordination_epoch": 1,
            "fencing_epoch": 1,
            "snapshot_cid": f"sha256:{digest}",
        }


def _view() -> SupervisorWorldView:
    return SupervisorWorldView(
        _snapshot(),
        {
            "goals": {"LGSWF-G020": {"status": "open"}},
            "tasks": {"LGSWF-010": {"status": "ready"}},
            "bindings": {"LGSWF-010": {"binding_cid": "sha256:" + ("11" * 32)}},
            "dependencies": {"LGSWF-010": {"depends_on": ("LGSWF-005",)}},
            "conflicts": {"LGSWF-010": {"conflicts": ()}},
            "resources": {"LGSWF-010": {"reserved": False}},
            "claims": {"LGSWF-010": {"claimed": False}},
            "capsules": {"LGSWF-010": {"capsule_cids": ()}},
            "contracts": {"LGSWF-010": {"interface": "SupervisorWorldSnapshot@1"}},
            "obligations": {"LGSWF-010": {"open": ()}},
            "completion": {"LGSWF-010": {"accepted": False}},
            "refill": {"LGSWF-010": {"eligible": False}},
        },
    )


def test_required_queries_are_deterministic_and_typed() -> None:
    view = _view()
    again = _view()
    assert view.snapshot_cid == again.snapshot_cid
    assert view.query_matrix() == required_query_names()
    assert view.goal_state("LGSWF-G020")["found"] is True
    assert view.task_state("LGSWF-010")["status"] == "ready"
    assert view.semantic_binding("LGSWF-010")["found"] is True
    assert view.dependencies("LGSWF-010")["depends_on"] == ("LGSWF-005",)
    assert view.conflicts("LGSWF-010")["conflicts"] == ()
    assert view.resources("LGSWF-010")["reserved"] is False
    assert view.claims("LGSWF-010")["claimed"] is False
    assert view.capsules("LGSWF-010")["capsule_cids"] == ()
    assert view.contracts("LGSWF-010")["interface"] == "SupervisorWorldSnapshot@1"
    assert view.obligations("LGSWF-010")["open"] == ()
    assert view.completion_evidence("LGSWF-010")["accepted"] is False
    assert view.refill_eligibility("LGSWF-010")["eligible"] is False


def test_unknown_references_are_explicit() -> None:
    view = _view()
    missing = view.task_state("LGSWF-999")
    assert missing["found"] is False
    assert missing["reason"] == "unknown-reference"
    assert missing["snapshot_cid"] == view.snapshot_cid
    with pytest.raises(WorldViewError, match="unknown reference"):
        view.component("not-a-component")


def test_mutation_attempts_fail_and_snapshot_is_not_refreshed() -> None:
    view = _view()
    original = view.snapshot_cid
    with pytest.raises(WorldViewError, match="immutable"):
        view.snapshot_cid = "mutated"  # type: ignore[misc]
    with pytest.raises(WorldViewError, match="immutable"):
        del view._snapshot
    assert view.snapshot_cid == original
    # Injected views are frozen; mutation of the mapping proxy must fail.
    with pytest.raises((TypeError, WorldViewError)):
        view._views["tasks"]["LGSWF-010"] = {"status": "mutated"}  # type: ignore[index]
