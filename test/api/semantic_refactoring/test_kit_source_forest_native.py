"""Real kit CAS bytes observed through the existing native SPAR read channel."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from test.api.semantic_refactoring.test_spar_closeout_profile import population
from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from ipfs_accelerate_py.agent_supervisor.task_sources import spar_closeout_profile as sp
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    STATUS_BOOTSTRAP_CLIENT_ID,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
)

OBSERVE_SOURCE = sp.observe_source


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def commit(root):
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "source",
    )
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def native_source(tmp_path, population, monkeypatch):
    checkout = Path(
        os.environ.get(
            "SPAR_KIT_NATIVE_CHECKOUT",
            Path(__file__).resolve().parents[3] / "ipfs_kit_py",
        )
    )
    leaf = Path("ipfs_kit_py/mcp_server/mcplusplus")
    if not (checkout / leaf / "duckdb_coordination_storage.py").is_file():
        pytest.fail(
            "native kit backend required; set SPAR_KIT_NATIVE_CHECKOUT to its actual checkout"
        )
    root = tmp_path / "source"
    kit = root / "ipfs_kit_py"
    (kit / leaf).mkdir(parents=True)
    for name in ("coordination_storage.py", "duckdb_coordination_storage.py"):
        shutil.copyfile(checkout / leaf / name, kit / leaf / name)
    git(kit, "init", "-q")
    kit_head = commit(kit)
    (root / "program.py").write_text("version = 1\n")
    git(root, "init", "-q")
    commit(root)
    material, facts, snapshot, _ = population
    material["nested_repositories"] = [
        {"repository": "ipfs_kit", "path": "ipfs_kit_py", "planning_revision": kit_head}
    ]
    monkeypatch.setattr(sp, "observe_source", OBSERVE_SOURCE)
    db = tmp_path / "control.duckdb"
    _install(db)
    gateway, connection = _gateway(db, tmp_path / "owner.sock")
    identity = gateway.identity
    connection.execute(
        "INSERT INTO state_servers (server_id,store_id,database_uuid,process_birth_id,listen_uri,extension_fingerprint,schema_revision,generation,started_at,status,revision) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            identity["server_id"],
            identity["store_id"],
            identity["database_uuid"],
            identity["process_birth_id"],
            "quack:127.0.0.1:7777",
            "sha256:test",
            identity["schema_revision"],
            identity["generation"],
            "now",
            "ready",
            0,
        ],
    )
    connection.execute("DELETE FROM tasks")
    connection.execute("DELETE FROM goals")
    for goal in facts["relations"]["goals"]["rows"]:
        connection.execute(
            "INSERT INTO goals (goal_cid,goal_alias,objective_id,parent_goal_cid,ordinal,title,status,created_at,updated_at,revision,body_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                goal["goal_cid"],
                goal["goal_alias"],
                "objective:spar",
                goal["parent_goal_cid"],
                1,
                goal["title"],
                goal["status"],
                "now",
                "now",
                goal["revision"],
                goal["body_json"],
            ],
        )
    for task in facts["relations"]["tasks"]["rows"]:
        connection.execute(
            "INSERT INTO tasks (task_cid,task_alias,goal_cid,plan_cid,objective_id,ordinal,status,revision,priority,created_at,updated_at,identity_json,body_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                task["task_cid"],
                task["task_alias"],
                task["goal_cid"],
                material["plan_root_cid"],
                "objective:spar",
                1,
                task["status"],
                task["revision"],
                "P0",
                "now",
                "now",
                json.dumps({"repository_tree_id": material["repository_tree_id"]}),
                json.dumps(
                    {
                        **json.loads(task["body_json"]),
                        "board_namespace": material["board_namespace"],
                    }
                ),
            ],
        )
    for receipt in snapshot["completion_projection"]["completion_receipts"]:
        connection.execute(
            "INSERT INTO completion_receipts (receipt_cid,task_cid,goal_cid,completed_at,evidence_digest,body_json) VALUES (?,?,?,?,?,?)",
            [
                receipt["receipt_cid"],
                receipt["task_cid"],
                receipt["goal_cid"],
                "now",
                receipt["evidence_digest"],
                json.dumps(receipt["body"]),
            ],
        )
    token = gateway.configure_status_bootstrap()
    profile = sp.SparCloseoutProfile(material, repository_root=str(root))
    cids = [t["task_cid"] for t in material["tasks"]]
    gateway.bind_database_status_scope(
        **{
            k: material[k]
            for k in ("board_namespace", "plan_root_cid", "repository_tree_id")
        },
        task_cids=cids,
        closeout_profile=profile,
    )
    client = TypedStateOwnerConnection(
        socket_path=gateway.socket_path,
        token=token,
        client_id=STATUS_BOOTSTRAP_CLIENT_ID,
        process_birth_id="birth:kit-profile-reader",
        store_id="control.duckdb",
        status_bootstrap=True,
    )
    try:
        yield gateway, connection, profile, client, cids, root
    finally:
        client.close()
        gateway.stop()
        connection.close()


def view(native_source):
    _, _, _, client, cids, _ = native_source
    return client.completion_closeout_snapshot(cids)["closeout_facts"][
        "completion_profile"
    ]


def test_native_producer_and_readback_admit_only_the_persistence_component(
    native_source,
):
    gateway, connection, _, client, _, _ = native_source
    before = view(native_source)
    assert before["kit_source_forest_persistence"]["admitted"] is False
    produced = gateway.publish_spar_source_forest()
    assert produced["admitted"] is True, produced
    after = view(native_source)
    kit = after["kit_source_forest_persistence"]
    assert (
        kit["admitted"] is True and kit["transition_cid"] == produced["transition_cid"]
    )
    assert kit["owner_identity"]["database_uuid"] == client.identity["database_uuid"]
    assert (
        sum(
            t["receipt"] is not None and not t["blockers"]
            for t in after["task_evidence"]
        )
        == 51
    )
    assert len(after["goal_requirements"]) == 32 and all(
        not g["accepted"] for g in after["goal_requirements"]
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM goals WHERE status='active'"
        ).fetchone()[0]
        == 32
    )
    assert (
        "kit_source_forest_cas_receipt_producer_and_admission_required"
        not in after["blockers"]
    )
    assert (
        "datasets_independent_accepted_root_producer_and_admission_required"
        in after["blockers"]
    )
    assert (
        not kit["semantic_acceptance_authority"] and not after["completion_authority"]
    )
    replay = gateway.publish_spar_source_forest()
    assert replay["idempotent_replay"] is True and replay["root_revision"] == 1
    assert view(native_source)["kit_source_forest_persistence"] == kit
    with pytest.raises(TypedStateOwnerError):
        client._request("publish_spar_source_forest")


def test_changed_current_source_invalidates_old_cas_until_real_successor(native_source):
    gateway, _, _, _, _, root = native_source
    original = gateway.publish_spar_source_forest()
    assert original["admitted"], original
    (root / "program.py").write_text("version = 2\n")
    commit(root)
    stale = view(native_source)
    assert not stale["kit_source_forest_persistence"]["admitted"]
    assert (
        "kit_source_forest_cas_receipt_producer_and_admission_required"
        in stale["blockers"]
    )
    successor = gateway.publish_spar_source_forest()
    assert successor["admitted"] and successor["root_revision"] == 2
    assert successor["transition_cid"] != original["transition_cid"]
    assert view(native_source)["kit_source_forest_persistence"]["admitted"]


def test_dirty_kit_module_hidden_by_git_flag_is_not_imported(native_source):
    gateway, _, _, _, _, root = native_source
    kit = root / "ipfs_kit_py"
    relative = "ipfs_kit_py/mcp_server/mcplusplus/duckdb_coordination_storage.py"
    marker = root.parent / "IMPORTED"
    git(kit, "update-index", "--assume-unchanged", relative)
    (kit / relative).write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    result = gateway.publish_spar_source_forest()
    assert not result["admitted"] and not marker.exists()
    assert not view(native_source)["kit_source_forest_persistence"]["admitted"]


def test_source_change_after_commit_stays_unaccepted_until_next_production_cas(
    native_source,
):
    gateway, _, profile, _, _, root = native_source
    first = gateway.publish_spar_source_forest()
    assert first["admitted"], first
    store = profile._kit_source_forest.store
    original = store.compare_and_swap_state_root

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "program.py").write_text("version = 3\n")
        commit(root)
        return result

    store.compare_and_swap_state_root = change
    result = gateway.publish_spar_source_forest()
    assert not result["admitted"]
    assert not view(native_source)["kit_source_forest_persistence"]["admitted"]
    store.compare_and_swap_state_root = original
    assert gateway.publish_spar_source_forest()["admitted"]
