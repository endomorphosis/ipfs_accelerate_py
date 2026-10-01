"""Exercise the retained callback consumer, including its lazy export imports."""

import copy

import pytest

from test.api.test_retained_callback_cooldown import fixture
from ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_callback_suffix import (
    verified_seed_predecessor,
)


@pytest.mark.parametrize(
    "mutation", ["none", "schema", "extra_field", "missing_field", "fence", "history"]
)
def test_retained_seed_predecessor_contract(mutation):
    task, history, _ = fixture()
    predecessor = copy.deepcopy(task["body"]["completion_receipt"])
    seed = copy.deepcopy(predecessor["post_merge_completion_recovery_seed"])
    if mutation == "schema":
        seed["schema"] = "wrong-schema"
    elif mutation == "extra_field":
        predecessor["unexpected"] = True
    elif mutation == "missing_field":
        predecessor.pop("queue_receipt")
    elif mutation == "fence":
        predecessor["fencing_token"] += 1
    elif mutation == "history":
        history["revisions"][-1]["status"] = "succeeded"
    assert verified_seed_predecessor(
        history,
        task_cid=task["task_cid"],
        task_alias=task["task_alias"],
        seed=seed,
        predecessor=predecessor,
    ) is (mutation == "none")
