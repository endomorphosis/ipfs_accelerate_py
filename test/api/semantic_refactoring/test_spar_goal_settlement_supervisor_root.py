"""Supervisor-admitted SPAR roots must not invalidate existing goal receipts."""

from ipfs_accelerate_py.agent_supervisor.task_sources.spar_goal_settlement import (
    RECEIPT_SCHEMA,
    observe_goal_settlement,
)


def _native(goal_cid: str, *, accepted_root_cid: str) -> dict:
    return {
        goal_cid: {
            "status": "completed",
            "body_json": {
                "completion_receipt": {
                    "schema": RECEIPT_SCHEMA,
                    "goal_cid": goal_cid,
                    "accepted_root_cid": accepted_root_cid,
                    "runtime_receipt_cid": "runtime:1",
                }
            },
        }
    }


def test_supervisor_clause_root_keeps_bootstrap_goal_receipts():
    goal_cid = "goal:g000"
    observed = observe_goal_settlement(
        native_goals=_native(goal_cid, accepted_root_cid="old-root"),
        profile={"goals": [{"goal_cid": goal_cid, "goal_alias": "SPAR-G000"}]},
        accepted_root={
            "admitted": True,
            "admission_mode": "bootstrap",
            "authority": "spar_supervisor_current_bound_clause_records",
            "accepted_root_cid": "new-root",
        },
        runtime={"admitted": True},
        kit={"admitted": True},
    )
    assert observed["admitted"] is True
    assert observed["accepted_goal_ids"] == ["SPAR-G000"]


def test_non_bootstrap_root_still_requires_matching_accepted_root_cid():
    goal_cid = "goal:g000"
    observed = observe_goal_settlement(
        native_goals=_native(goal_cid, accepted_root_cid="old-root"),
        profile={"goals": [{"goal_cid": goal_cid, "goal_alias": "SPAR-G000"}]},
        accepted_root={
            "admitted": True,
            "admission_mode": "required",
            "accepted_root_cid": "new-root",
        },
        runtime={"admitted": True},
        kit={"admitted": True},
    )
    assert observed["admitted"] is False
    assert observed["reason"] == "spar_native_goal_cas_settlement_adapter_required"
