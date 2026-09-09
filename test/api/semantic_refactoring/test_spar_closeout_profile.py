"""Native SPAR requirements never promote task counts or nominated reports."""

import copy
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import spar_closeout_profile as sp
from ipfs_accelerate_py.agent_supervisor.task_sources.closeout_snapshot import RELATIONS
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    COMPLETION_EVIDENCE_SCHEMA,
)


@pytest.fixture
def population(monkeypatch):
    goals = [
        {
            "goal_cid": f"goal:{i}",
            "goal_alias": f"SPAR-G{i:03}",
            "title": f"Goal {i}",
            "parent_goal_cid": "",
            "body": {"completion_contract": "independently accepted current root"},
        }
        for i in range(32)
    ]
    tasks = [
        {
            "task_cid": f"task:{i}",
            "task_alias": f"SPAR-{i:03}",
            "title": f"Task {i}",
            "goal_cid": f"goal:{i%32}",
            "dependencies": [],
            "contract_fields": {"completion_contract": "current root; required modes"},
        }
        for i in range(51)
    ]
    material = {
        "schema": sp.SCHEMA,
        "board_namespace": "semantic-preserving-autonomous-remodularization-v1",
        "bootstrap_receipt_id": "sealed:bootstrap",
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "source_identities": {},
        "completion_policy": {
            **{key: True for key in sp.POLICY_FIELDS},
            "terminal_task_id": "SPAR-050",
        },
        "goals": goals,
        "tasks": tasks,
        "goal_edges": [],
        "nested_repositories": [],
    }
    relations = {
        name: {"available": True, "truncated": False, "rows": []} for name in RELATIONS
    }
    relations["goals"]["rows"] = [
        {
            **g,
            "body_json": json.dumps({"body": g["body"]}),
            "status": "active",
            "revision": 1,
        }
        for g in goals
    ]
    receipts = []
    for task in tasks:
        control = {"test": "passed", "attempt": task["task_cid"]}
        digest = content_identity(
            {
                "task_cid": task["task_cid"],
                "revision": 3,
                "receipt": control,
                "evidence_digests": [],
            }
        )
        receipt_cid = content_identity(
            {
                "namespace": "completion-receipt",
                "task_cid": task["task_cid"],
                "revision": 3,
                "evidence_digest": digest,
            }
        )
        receipts.append(
            {
                "receipt_cid": receipt_cid,
                "task_cid": task["task_cid"],
                "goal_cid": task["goal_cid"],
                "evidence_digest": digest,
                "body": {
                    "schema": COMPLETION_EVIDENCE_SCHEMA,
                    "revision": 3,
                    "receipt": control,
                    "evidence_digests": [],
                },
            }
        )
        relations["tasks"]["rows"].append(
            {
                **task,
                "revision": 3,
                "status": "completed",
                "body_json": json.dumps(
                    {
                        **task["contract_fields"],
                        "title": task["title"],
                        "completion_receipt": control,
                    }
                ),
            }
        )
    source = {
        "available": True,
        "clean": True,
        "source_forest": {"source_forest_root": "forest:new"},
        "reports": [
            {
                "available": True,
                "path": p,
                "nomination_only": True,
                "can_authorize_completion": False,
                "authority_roots": {"repository_forest_cid": "forest:old"},
            }
            for p in sp.REPORTS
        ],
    }
    monkeypatch.setattr(sp, "observe_source", lambda *args: copy.deepcopy(source))
    snapshot = {
        "snapshot_cid": "snapshot:current",
        "completion_projection": {"completion_receipts": receipts},
    }
    return material, {"relations": relations}, snapshot, source


def evaluate(population):
    material, facts, snapshot, _ = population
    return sp.SparCloseoutProfile(material, repository_root="/unused").evaluate(
        facts, snapshot
    )


def test_all_current_task_receipts_do_not_accept_a_single_goal(population):
    result = evaluate(population)
    assert sum(t["receipt"] is not None for t in result["task_evidence"]) == 51
    assert all(not g["accepted"] for g in result["goal_requirements"])
    assert not result["completion_authority"]
    assert "final_report_current_source_forest_mismatch" in result["blockers"]
    assert (
        "datasets_independent_accepted_root_producer_and_admission_required"
        in result["blockers"]
    )
    assert result["observation_cid"] == content_identity(
        {k: v for k, v in result.items() if k != "observation_cid"}
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "task_revision",
        "receipt_body",
        "goal_contract",
        "goal_population",
        "dependency",
        "missing_relation",
    ],
)
def test_exact_native_evidence_corruption_is_a_typed_blocker(population, corruption):
    _, facts, snapshot, _ = population
    if corruption == "task_revision":
        facts["relations"]["tasks"]["rows"][0]["revision"] += 1
    elif corruption == "receipt_body":
        snapshot["completion_projection"]["completion_receipts"][0]["body"][
            "receipt"
        ] = {"test": "forged"}
    elif corruption == "goal_contract":
        facts["relations"]["goals"]["rows"][0][
            "body_json"
        ] = '{"body":{"completion_contract":"task count is sufficient"}}'
    elif corruption == "goal_population":
        facts["relations"]["goals"]["rows"].pop()
    elif corruption == "dependency":
        facts["relations"]["task_dependencies"]["rows"].append(
            {"task_cid": "task:0", "dependency_task_cid": "task:50"}
        )
    else:
        facts["relations"]["effect_claims"]["available"] = False
    result = evaluate(population)
    prefixes = {
        "task_revision": "task_current_completion_receipt_population_not_exact",
        "receipt_body": "task_current_completion_receipt_body_invalid",
        "goal_contract": "sealed_goal_contract_changed",
        "goal_population": "sealed_goal_population_changed",
        "dependency": "sealed_task_dependencies_changed",
        "missing_relation": "native_relation_unavailable_or_truncated",
    }
    assert any(b.startswith(prefixes[corruption]) for b in result["blockers"])
    assert not result["completion_authority"]


def test_self_claimed_report_acceptance_is_not_independent_authority(population):
    source = population[3]
    for report in source["reports"]:
        report.update(
            nomination_only=False,
            final_root_accepted=True,
            can_authorize_completion=True,
            authority_roots={"repository_forest_cid": "forest:new"},
        )
    result = evaluate(population)
    assert (
        "datasets_independent_accepted_root_producer_and_admission_required"
        in result["blockers"]
    )
    assert not result["completion_authority"]


def test_profile_rejects_weakened_policy_and_foreign_scope(population):
    material = population[0]
    profile = sp.SparCloseoutProfile(material, repository_root="/unused")
    binding = {
        key: material[key]
        for key in ("board_namespace", "plan_root_cid", "repository_tree_id")
    }
    binding["task_cids"] = [t["task_cid"] for t in material["tasks"]]
    profile.assert_scope(binding)
    with pytest.raises(ValueError, match="scope"):
        profile.assert_scope({**binding, "plan_root_cid": "foreign"})
    material["completion_policy"]["safety_floors_noncompensable"] = False
    with pytest.raises(ValueError, match="closed"):
        sp.SparCloseoutProfile(material, repository_root="/unused")


def test_actual_source_observation_has_deadline_and_preserves_unavailability(
    tmp_path, monkeypatch
):
    observed = []

    def timed_out(*args, **kwargs):
        observed.append(kwargs["timeout"])
        raise sp.subprocess.TimeoutExpired("git", kwargs["timeout"])

    monkeypatch.setattr(sp.subprocess, "run", timed_out)
    result = sp.observe_source(str(tmp_path), [])
    assert not result["available"] and 0 < observed[0] <= 4
    assert not result["semantic_acceptance_authority"]


def test_native_gateway_binds_profile_to_actual_owner_snapshot(tmp_path, population):
    from test.api.causal_federation.test_typed_state_owner import _gateway, _install
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TypedStateOwnerConnection,
        STATUS_BOOTSTRAP_CLIENT_ID,
    )

    db = tmp_path / "control.duckdb"
    _install(db)
    gateway, connection = _gateway(db, tmp_path / "owner.sock")
    token = gateway.configure_status_bootstrap()
    material, facts, _, _ = population
    connection.execute("DELETE FROM tasks")
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
    profile = sp.SparCloseoutProfile(material, repository_root=str(tmp_path))
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
        process_birth_id="birth:spar-profile",
        store_id="control.duckdb",
        status_bootstrap=True,
    )
    try:
        snapshot = client.completion_closeout_snapshot(cids)
        observed = snapshot["closeout_facts"]["completion_profile"]
        assert observed["profile_cid"] == profile.profile_cid
        assert (
            observed["completion_snapshot_cid"]
            == snapshot["completion_snapshot"]["snapshot_cid"]
        )
        assert not observed["completion_authority"]
        assert "sealed_goal_population_changed" in observed["blockers"]
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_launcher_rejects_original_source_hash_mismatch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from test.api.semantic_refactoring.test_bootstrap_controls import _materializer

    m = _materializer()
    board = SimpleNamespace(
        objectives_path="objectives.md",
        taskboard_path="tasks.md",
        config_path=m.ROOT / "config.json",
    )
    monkeypatch.setattr(m, "_git", lambda *args, **kw: b"altered goals")
    with pytest.raises(m.OperatorError, match="bootstrap objectives"):
        m._native_closeout_profile(
            board,
            {},
            {
                "source_head": "sealed:head",
                "source_identities": {"objectives": "sha256:original"},
            },
        )


def test_unknown_native_effect_claim_remains_blocking(population):
    population[1]["relations"]["effect_claims"]["rows"] = [
        {"state": None, "effect_id": "unclassified"}
    ]
    assert "native_unsettled_rows:effect_claims" in evaluate(population)["blockers"]
