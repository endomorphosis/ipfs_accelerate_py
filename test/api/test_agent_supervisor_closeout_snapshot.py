"""Owner-transaction closeout observations retain unresolved authority facts."""

import json

import pytest

from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
    TypedStateOwnerError,
    STATUS_BOOTSTRAP_CLIENT_ID,
)


@pytest.fixture
def native(tmp_path):
    db = tmp_path / "control.duckdb"
    _install(db)
    gateway, connection = _gateway(db, tmp_path / "owner.sock")
    token = gateway.configure_status_bootstrap()
    connection.execute(
        "UPDATE tasks SET plan_cid=?, identity_json=?, body_json=?",
        [
            "plan:sealed",
            json.dumps({"repository_tree_id": "tree:sealed"}),
            json.dumps({"board_namespace": "board:sealed"}),
        ],
    )
    gateway.bind_database_status_scope(
        board_namespace="board:sealed",
        plan_root_cid="plan:sealed",
        repository_tree_id="tree:sealed",
        task_cids=["task:typed-owner"],
    )
    client = TypedStateOwnerConnection(
        socket_path=gateway.socket_path,
        token=token,
        client_id=STATUS_BOOTSTRAP_CLIENT_ID,
        process_birth_id="birth:closeout-reader",
        store_id="control.duckdb",
        status_bootstrap=True,
    )
    yield client, connection, gateway
    client.close()
    gateway.stop()
    connection.close()


def test_native_closeout_contains_goals_claims_blocks_and_exact_task_receipts(native):
    client, connection, _ = native
    connection.execute(
        "INSERT INTO task_claims VALUES ('claim:active','task:typed-owner','session:worker',1,1,'now','later',NULL,'active',0,'id:claim')"
    )
    connection.execute(
        "INSERT INTO task_blocks VALUES ('block:active','task:typed-owner','proof','proof:missing','evidence missing','now',NULL,'active')"
    )
    snapshot = client.completion_closeout_snapshot(["task:typed-owner"])
    facts = snapshot["closeout_facts"]
    relations = facts["relations"]
    assert relations["tasks"]["rows"][0]["task_cid"] == "task:typed-owner"
    assert relations["goals"]["rows"][0]["status"] == "open"
    assert relations["task_claims"]["rows"][0]["claim_id"] == "claim:active"
    assert relations["task_blocks"]["rows"][0]["blocker_id"] == "proof:missing"
    assert (
        snapshot["completion_snapshot"]["owner_identity"]["generation"]
        == client.identity["generation"]
    )
    assert (
        snapshot["completion_snapshot"]["completion_projection"]["task_states"][0][
            "status"
        ]
        == "ready"
    )
    assert snapshot["completion_authority"] is False
    assert facts["goal_contracts_evaluated"] is False
    assert facts["external_semantic_obligations_verified"] is False
    assert facts["all_relations_available"] is True
    connection.execute("UPDATE task_claims SET state='released'")
    assert (
        client.completion_closeout_snapshot(["task:typed-owner"])["closeout_facts"][
            "relations"
        ]["task_claims"]["rows"]
        == []
    )
    assert relations["task_claims"]["rows"][0]["state"] == "active"


def test_closeout_missing_authority_relation_is_unknown_and_foreign_scope_rejected(
    native,
):
    client, connection, _ = native
    connection.execute("DROP TABLE proof_obligations")
    facts = client.completion_closeout_snapshot(["task:typed-owner"])["closeout_facts"]
    assert facts["relations"]["proof_obligations"]["available"] is False
    assert facts["all_relations_available"] is False
    with pytest.raises(TypedStateOwnerError):
        client.completion_closeout_snapshot(["task:foreign"])
    connection.execute("UPDATE tasks SET plan_cid='plan:changed'")
    with pytest.raises(TypedStateOwnerError):
        client.completion_closeout_snapshot(["task:typed-owner"])


def test_closeout_relation_population_is_bounded_and_cannot_report_complete(native):
    client, connection, _ = native
    connection.execute(
        "INSERT INTO goals SELECT 'goal:'||i, 'G-'||i, '', '', i, 'Goal', 'open', '', '', 0, '{}' FROM range(1, 514) AS t(i)"
    )
    facts = client.completion_closeout_snapshot(["task:typed-owner"])["closeout_facts"]
    assert len(facts["relations"]["goals"]["rows"]) == 512
    assert facts["relations"]["goals"]["truncated"] is True
    assert facts["truncated"] is True
    assert facts["completion_authority"] is False


def test_nullable_legacy_claim_state_remains_an_unresolved_native_fact(native):
    client, connection, _ = native
    connection.execute("CREATE TABLE legacy_claims AS SELECT * FROM task_claims")
    connection.execute("DROP TABLE task_claims")
    connection.execute("ALTER TABLE legacy_claims RENAME TO task_claims")
    connection.execute(
        "INSERT INTO task_claims VALUES ('claim:unknown','task:typed-owner','session:worker',1,1,'now','later',NULL,NULL,0,'id:unknown')"
    )
    facts = client.completion_closeout_snapshot(["task:typed-owner"])["closeout_facts"]
    assert facts["relations"]["task_claims"]["rows"][0]["claim_id"] == "claim:unknown"
    assert facts["relations"]["task_claims"]["rows"][0]["state"] is None
    assert facts["completion_authority"] is False
