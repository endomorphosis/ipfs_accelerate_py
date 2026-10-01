"""Ordinary reservations cannot erase transfer lineage or change home lanes."""
from copy import deepcopy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository as module


def inputs():
    task = {"task_cid": "task:ordinary", "task_alias": "T-001", "status": "ready",
            "revision": 1, "body": {"title": "Ordinary task"}}
    receipt = {"operation": "database_claim", "claimed_from_revision": 1,
               "task_prefix": "T-", "task_shard_count": 1, "task_shard_index": 0,
               "strict_task_sharding": True, "idle_lane_work_stealing": "",
               "claim_id": "claim:exact", "claim_process_attestation": {"bound": "by owner"}}
    return task, receipt


def prepare(task, receipt):
    # This hook has no independent mutation or attestation authority. The real
    # owner transaction surrounding it is exercised by the daemon feedback tests.
    return module._prepare_database_virgin_transfer_receipt_on(
        object(), task=task, previous_status=task["status"], current_revision=1,
        new_status="in_progress", receipt=receipt, now_ms=1000,
    )


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON", raising=False)


def test_ordinary_claim_is_an_exact_unmodified_receipt():
    task, receipt = inputs()
    before = deepcopy((task, receipt))
    result = prepare(task, receipt)
    assert result == receipt and result is not receipt
    assert (task, receipt) == before


@pytest.mark.parametrize("field", ["virgin_task_transfer", "virgin_task_transfer_claim_cursor",
                                   "virgin_task_transfer_request"])
@pytest.mark.parametrize("location", ["body", "prior", "next"])
def test_transfer_presence_even_null_cannot_be_laundered(field, location):
    task, receipt = inputs()
    target = {"body": task["body"], "prior": task["body"].setdefault("completion_receipt", {}),
              "next": receipt}[location]
    target[field] = None
    with pytest.raises(module.IntentRepositoryTransitionError, match="transfer"):
        prepare(task, receipt)


@pytest.mark.parametrize("field,value", [
    ("claimed_from_revision", True), ("claimed_from_revision", 2),
    ("operation", "database_attempt_admitted"), ("task_shard_count", True),
    ("task_shard_count", 0), ("task_shard_index", False), ("task_shard_index", 1),
    ("strict_task_sharding", 1), ("task_prefix", "foreign-"),
    ("idle_lane_work_stealing", "virgin-transfer"),
    ("idle_lane_work_stealing", None),
])
def test_claim_binding_cannot_be_reinterpreted(field, value):
    task, receipt = inputs()
    receipt[field] = value
    with pytest.raises(module.IntentRepositoryTransitionError):
        prepare(task, receipt)


def test_home_shard_is_enforced_without_restricting_independent_home_lanes():
    task, receipt = inputs()
    receipt["task_shard_count"] = 4
    home = module.database_task_alias_home_shard_index(task["task_alias"], 4)
    receipt["task_shard_index"] = home
    assert prepare(task, receipt) == receipt
    receipt["task_shard_index"] = (home + 1) % 4
    with pytest.raises(module.IntentRepositoryTransitionError, match="home shard"):
        prepare(task, receipt)


@pytest.mark.parametrize("raw", ["not-json", "[]", '{"claim_policy":true}',
    '{"claim_policy":{"idle_lane_work_stealing":"virgin-transfer"}}',
    '{"claim_policy":null}', '{"claim_policy":{},"claim_policy":null}'])
def test_ordinary_claim_cannot_override_configured_owner_policy(monkeypatch, raw):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON", raw)
    with pytest.raises(module.IntentRepositoryTransitionError):
        prepare(*inputs())


def test_configured_transfer_policy_cannot_be_reinterpreted_as_ordinary(monkeypatch):
    task, receipt = inputs()
    policy = {key: receipt[key] for key in ("task_prefix", "task_shard_count",
               "strict_task_sharding", "idle_lane_work_stealing")}
    policy["schema"] = module.DATABASE_CLAIM_POLICY_SCHEMA
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON", json.dumps({"claim_policy": policy}))
    with pytest.raises(module.IntentRepositoryTransitionError, match="policy is unavailable"):
        prepare(task, receipt)
