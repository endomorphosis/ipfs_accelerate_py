"""Real native gateway/transaction tests with only decision policy replaced.

These tests qualify actual AF_UNIX grants, creator binding, physical canonical
connection, reserved evidence and append/replay behavior. They do not qualify
launch custody, native status/WAL maintenance or execution settlement.
"""
from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource, TaskSourceIntegrityError
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes, content_identity
from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment as policy


@pytest.fixture
def native(tmp_path, monkeypatch):
    path = tmp_path / "fixture.duckdb"
    repository = IntentRepository(path)
    repository.upsert_goal(goal_cid="fixture-goal", goal_alias="FIXTURE-G", title="TEST ONLY")
    repository.upsert_task(task_cid="fixture-task", task_alias="FIXTURE-T", goal_cid="fixture-goal",
                           status="in_progress", body={"objective": "TEST ONLY"})
    repository.upsert_task(task_cid="another-task", task_alias="OTHER", goal_cid="fixture-goal", status="blocked",
        body={"objective": "TEST ONLY other task"}, outputs=[{"path": "data/original.txt", "effect": "write"}],
        acceptance=["TEST ONLY original criterion"], validations=[["/usr/bin/true"]])
    repository.close()
    connection = duckdb.connect(str(path))
    source = DatabaseTaskSource(intent=IntentRepository(bound_connection=connection, install_schema=False))
    identity = {"server_id": "test-only-admin-owner", "generation": 1, "store_id": "fixture-logical-store"}
    gateway = typed.TypedStateOwnerGateway(connection=connection, socket_path=tmp_path / "owner.sock",
                                           store_id=identity["store_id"], identity=identity)
    gateway.start()
    before = {"task": policy.plain(source.get_task("fixture-task").to_dict()),
              "intent_task": policy.plain(source.intent.get_task("fixture-task")),
              "canonical_population": policy.canonical_population(source)}
    context = {"context_cid": "TEST-ONLY-transport-policy-context",
               "registered_owner_identity": identity,
               "registered_creator": {"pid": os.getpid(), "process_birth_id": typed.kernel_process_birth_id(os.getpid())}}
    original = copy.deepcopy(context)
    revision = before["task"]["revision"]
    scope = {"task_cid": "fixture-task", "context_cid": context["context_cid"],
             "expected_task_revision": revision, "abandonment_scope_cid": "TEST-ONLY-transport-scope",
             "canonical_store_id": str(path), "logical_store_id": identity["store_id"]}
    calls = []

    def scope_policy(value):
        if value != original:
            raise policy.PlanningUnknownAbandonmentError("TEST ONLY registered policy context differs")
        return dict(scope)

    def load_policy(value, *, check_closed_stores):
        scope_policy(value)
        return {"attempt": {"attempt_id": "fixture-attempt2", "claim_id": "fixture-claim2"}}

    def verify_policy(payload, *, authority_context, current_observation):
        scope_policy(authority_context)
        if payload != {"operation": policy.OPERATION, "context_cid": original["context_cid"],
                       "expected_task_revision": revision} or type(payload["expected_task_revision"]) is not int:
            raise policy.PlanningUnknownAbandonmentError("TEST ONLY ADMIN request differs")
        observed = {key: policy.plain(current_observation[key]) for key in before}
        if observed != before:
            raise policy.PlanningUnknownAbandonmentError("TEST ONLY canonical preimage changed")
        calls.append(observed)
        return {"schema": "TEST-ONLY-transport-decision@1", "task_cid": "fixture-task",
                "context_cid": original["context_cid"], **policy.DECISION}

    monkeypatch.setattr(policy, "scope", scope_policy)
    monkeypatch.setattr(policy, "load_context", load_policy)
    monkeypatch.setattr(policy, "verify_request", verify_policy)
    gateway.bind_planning_unknown_abandonment_service(expected_identity=identity, authority_context=context)
    clients = []

    def connect(token, grant):
        client = typed.TypedStateOwnerConnection(socket_path=gateway.socket_path, token=token,
            client_id=grant.client_id, process_birth_id=grant.process_birth_id, store_id=identity["store_id"])
        clients.append(client)
        return client

    def admin():
        return connect(*gateway.issue_planning_unknown_abandonment_grant())

    yield SimpleNamespace(path=path, connection=connection, source=source, gateway=gateway, identity=identity,
        context=context, original=original, scope=scope, before=before, revision=revision,
        calls=calls, connect=connect, admin=admin)
    for client in clients: client.close()
    gateway.stop()
    source.close()
    connection.close()


def decide(native, client=None):
    return (client or native.admin()).record_planning_unknown_abandonment_decision(
        native.original["context_cid"], native.revision)


def test_real_admin_rpc_appends_one_reserved_audit_and_event_with_no_task_mutation(native):
    client = native.admin()
    watermark = native.source.intent.event_watermark()
    first, replay = decide(native, client), decide(native, client)
    assert first["replayed"] is False and replay["replayed"] is True
    assert first["decision"] == replay["decision"] and first["audit_receipt"] == replay["audit_receipt"]
    assert native.source.intent.event_watermark() == watermark + 1
    assert native.connection.execute("SELECT COUNT(*) FROM evidence_nodes WHERE evidence_kind=?", [policy.EVIDENCE_KIND]).fetchone()[0] == 1
    assert policy.plain(native.source.intent.get_task("fixture-task")) == native.before["intent_task"]
    assert policy.canonical_population(native.source) == native.before["canonical_population"]
    assert first["decision"]["provider_outcome"] == "unknown"
    assert first["decision"]["callback_closed"] is False
    assert policy.OPERATION not in typed.DATABASE_TASK_COMMANDS


def test_registered_context_is_frozen_and_cannot_be_rebound(native):
    native.context["context_cid"] = "changed-by-caller"
    assert decide(native)["decision"]["context_cid"] == native.original["context_cid"]
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        native.gateway.bind_planning_unknown_abandonment_service(expected_identity=native.identity,
                                                               authority_context=native.original)


@pytest.mark.parametrize("corruption", ["parent_evidence", "event_schema", "event_owner"])
def test_native_admin_replay_refuses_corrupted_audit_lineage_or_owner(native, corruption):
    first = decide(native)
    receipt = first["audit_receipt"]
    evidence_id = receipt["subject_id"]
    if corruption == "parent_evidence":
        native.connection.execute("UPDATE evidence_nodes SET parent_evidence_id=? WHERE evidence_id=?",
                                  ["TEST-ONLY-foreign-parent", evidence_id])
    else:
        row = native.connection.execute(
            "SELECT stream_id,sequence,global_sequence,event_type,body_json FROM domain_events WHERE event_id=?",
            [receipt["event_id"]]).fetchone()
        envelope = json.loads(row[4])
        envelope["schema" if corruption == "event_schema" else "owner_id"] = "TEST-ONLY-foreign-value"
        # Recompute the content identity so this specifically checks the native
        # schema/registered ADMIN owner, rather than only a selfhash mismatch.
        changed_id = content_identity({"stream_id": row[0], "sequence": row[1],
            "global_sequence": row[2], "event_type": row[3], "body": envelope})
        native.connection.execute("UPDATE domain_events SET event_id=?,body_json=? WHERE event_id=?",
            [changed_id, canonical_json_bytes(envelope).decode("utf-8"), receipt["event_id"]])
    watermark = native.source.intent.event_watermark()
    with pytest.raises(typed.TypedStateOwnerRemoteError):
        decide(native)
    assert native.source.intent.event_watermark() == watermark
    assert native.connection.execute("SELECT COUNT(*) FROM evidence_nodes WHERE evidence_kind=?",
                                      [policy.EVIDENCE_KIND]).fetchone()[0] == 1


def test_ordinary_worker_cannot_issue_a_decision_or_forge_reserved_evidence(native):
    token, grant = native.gateway.issue_grant(client_id="ordinary-worker",
        process_birth_id=typed.kernel_process_birth_id(os.getpid()), allowed_database_task_commands=("record_evidence",))
    with pytest.raises(typed.TypedStateOwnerRemoteError):
        decide(native, native.connect(token, grant))
    with pytest.raises(TaskSourceIntegrityError):
        native.source.record_evidence(task_cid="fixture-task", evidence_kind=policy.EVIDENCE_KIND,
                                      digest="TEST-ONLY-forged", body={"completion_authority": False})
    assert native.calls == []


@pytest.mark.parametrize("change", ["client", "scope", "operations", "commands", "database_commands", "birth"])
def test_admin_grant_refuses_wider_or_foreign_authority(native, change):
    kwargs = {"client_id": typed.PLANNING_UNKNOWN_ABANDONMENT_ADMIN_CLIENT_ID,
        "process_birth_id": typed.kernel_process_birth_id(os.getpid()),
        "allowed_operations": (policy.OPERATION,),
        "entity_scopes": {"task_cid": "fixture-task", "abandonment_scope_cid": native.scope["abandonment_scope_cid"]}}
    if change == "client": kwargs["client_id"] = "ordinary-worker"
    elif change == "scope": kwargs["entity_scopes"]["task_cid"] = "another-task"
    elif change == "operations": kwargs["allowed_operations"] += (typed.HASH_OBSERVATION_SERVICE_OPERATION,)
    elif change == "commands": kwargs["allowed_command_operations"] = ("task.status.cas",)
    elif change == "database_commands": kwargs["allowed_database_task_commands"] = ("record_evidence",)
    elif change == "birth": kwargs["process_birth_id"] = "foreign-birth"
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        native.gateway.issue_grant(**kwargs)
    assert native.calls == []


@pytest.mark.parametrize("change", ["context", "revision", "bool_revision", "extra_fields"])
def test_rpc_refuses_request_outside_exact_registered_context(native, change):
    payload = {"operation": policy.OPERATION, "context_cid": native.original["context_cid"],
               "expected_task_revision": native.revision}
    if change == "context": payload["context_cid"] = "another-context"
    elif change == "revision": payload["expected_task_revision"] += 1
    elif change == "bool_revision": payload["expected_task_revision"] = True
    else: payload["caller_sql"] = "must never reach a database"
    with pytest.raises(typed.TypedStateOwnerRemoteError):
        native.admin()._request(policy.OPERATION, payload=payload)
    assert native.calls == []


@pytest.mark.parametrize("relation", ["outputs", "acceptance", "validations"])
def test_changed_normalized_other_task_contract_refuses_audit(native, relation):
    # The actual source contract is normalized; a tasks row alone is inadequate.
    # Deliberate corruption is confined to this test's disposable database.
    before = native.connection.execute("SELECT body_json,revision FROM tasks WHERE task_cid='another-task'").fetchone()
    statements = {"outputs": "UPDATE task_outputs SET path='data/changed.txt' WHERE task_cid='another-task'",
        "acceptance": "UPDATE task_acceptance SET criterion='changed criterion' WHERE task_cid='another-task'",
        "validations": "UPDATE task_validations SET argv_json='[\"/usr/bin/false\"]' WHERE task_cid='another-task'"}
    native.connection.execute(statements[relation])
    assert native.connection.execute("SELECT body_json,revision FROM tasks WHERE task_cid='another-task'").fetchone() == before
    with pytest.raises(typed.TypedStateOwnerRemoteError):
        decide(native)
    assert native.calls == []
