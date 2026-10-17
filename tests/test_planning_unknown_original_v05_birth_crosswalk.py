"""Nonadmitting birth-crosswalk subchecks; no native receipt is created.

The complete custody verifier still requires its original launch/owner/terminal
CID and time joins. These isolated dictionaries exercise only the narrow typed
join that permits a genuine v05 direct-child/master proof to supply its already
recorded birth, never admission or nonreachability.
"""
from copy import deepcopy
import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment as policy


def fields():
    return ({"owner_pid": 11001, "child_pid": 11002},
            {"master_pid": 11002, "master_start_time_ticks": 22002},
            [{"pid": 11001, "start_time_ticks": 22001}, {"pid": 11002, "start_time_ticks": 22002}])


def test_missing_v05_marker_uses_existing_exact_direct_master_birth_without_rewriting_receipt():
    owner, fence, rows = fields()
    before = deepcopy((owner, fence, rows))
    policy.original_process_crosswalk(owner, fence, rows)
    assert (owner, fence, rows) == before
    assert "child_birth_marker" not in owner


@pytest.mark.parametrize("change", ["descendant_master", "wrong_child_ticks", "wrong_master_ticks",
    "bool_child_pid", "bool_master_pid", "bool_master_ticks", "missing_master_ticks", "missing_owner"])
def test_missing_marker_refuses_every_other_birth_scope(change):
    owner, fence, rows = fields()
    if change == "descendant_master":
        fence["master_pid"] = 11003
        rows.append({"pid": 11003, "start_time_ticks": 22002})
    elif change == "wrong_child_ticks": rows[1]["start_time_ticks"] += 1
    elif change == "wrong_master_ticks": fence["master_start_time_ticks"] += 1
    elif change == "bool_child_pid": owner["child_pid"] = True
    elif change == "bool_master_pid": fence["master_pid"] = True
    elif change == "bool_master_ticks": fence["master_start_time_ticks"] = True
    elif change == "missing_master_ticks": del fence["master_start_time_ticks"]
    else: rows.pop(0)
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        policy.original_process_crosswalk(owner, fence, rows)


@pytest.mark.parametrize("marker", [None, True, 22002, "wrong", "", "0", "-1", "022002", "２２００２"])
def test_present_invalid_marker_cannot_use_missing_marker_fallback(marker):
    owner, fence, rows = fields()
    owner["child_birth_marker"] = marker
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        policy.original_process_crosswalk(owner, fence, rows)


def test_explicit_child_birth_retains_descendant_master_support():
    owner, fence, rows = fields()
    owner["child_birth_marker"] = "22002"
    fence.update(master_pid=11003, master_start_time_ticks=22003)
    rows.append({"pid": 11003, "start_time_ticks": 22003})
    policy.original_process_crosswalk(owner, fence, rows)
    rows[1]["start_time_ticks"] += 1
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        policy.original_process_crosswalk(owner, fence, rows)
