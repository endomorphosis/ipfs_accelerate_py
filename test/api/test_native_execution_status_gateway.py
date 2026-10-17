"""Public disposable source fixtures; no campaign/native authority is granted.

The standalone component's public fixture creates new local native tables from
actual producer DDL. This file adds a genuine typed socket and new primary store.
No original daemon handles, provider callbacks, production paths, SQL clients,
claim resets or native admission proofs are fabricated.
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import replace
from types import MappingProxyType

import pytest

from test.api.test_native_execution_status_owner import store, no_processes, digest, identity, control_task
from test.api.causal_federation.test_typed_state_owner import _install
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServer, QuackStateServerConfig, QuackStateServerControlError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_execution_status_owner import NativeExecutionStatusOwner


@pytest.fixture
def unstarted(store, tmp_path):
    db = tmp_path / "control.duckdb"
    _install(db)
    connection = open_duckdb_connection(db)
    row = connection.execute("SELECT generation,schema_revision,fence_epoch,revision,database_uuid,birth_id FROM store_generations ORDER BY generation DESC LIMIT 1").fetchone()
    fingerprint = connection.execute("SELECT value FROM control_plane_metadata WHERE key='schema_fingerprint'").fetchone()[0]
    owner_identity = dict(server_id="server:fixture-native-status", store_id="control.duckdb",
        generation=row[0],schema_revision=row[1],fence_epoch=row[2],revision=row[3],
        database_uuid=row[4],process_birth_id=row[5],schema_fingerprint=fingerprint)
    binding = replace(store["binding"], control_store_id="control.duckdb", control_schema_version=row[1])
    body = {"board_namespace":"board:sealed", **control_task(binding)["body"]}
    # Actual retained claim receipts join task CID through the primary row;
    # they do not contain a duplicate receipt.task_cid field.
    body["completion_receipt"].pop("task_cid",None)
    body["source_context"]="x"*46_000
    connection.execute("UPDATE tasks SET task_cid=?,task_alias=?,plan_cid=?,identity_json=?,body_json=?,status=?,revision=?",
        [binding.task_cid,"FIXTURE-1","plan:sealed",json.dumps({"repository_tree_id":"tree:sealed"}),
         canonical_json_bytes(body).decode(),binding.expected_control_status,binding.control_revision])
    gateway = typed.TypedStateOwnerGateway(connection=connection, socket_path=tmp_path/"owner.sock",
        store_id="control.duckdb",identity=owner_identity)
    token = gateway.configure_status_bootstrap()
    gateway.bind_database_status_scope(board_namespace="board:sealed",plan_root_cid="plan:sealed",
        repository_tree_id="tree:sealed",task_cids=[binding.task_cid])
    context = dict(store=store,db=db,connection=connection,gateway=gateway,binding=binding,status_token=token,
        control_identity={key:owner_identity[key] for key in typed._NATIVE_EXECUTION_STATUS_CONTROL_KEYS})
    try:
        yield context
    finally:
        gateway.stop()
        connection.close()


@pytest.fixture
def ready(unstarted):
    context=unstarted
    context["gateway"].bind_native_execution_status_before_start(binding=context["binding"].to_dict(),
        control_identity=context["control_identity"],deadline_monotonic=time.monotonic()+60)
    context["gateway"].start()
    return context


def grant(context, **changed):
    arguments=dict(process_birth_id=typed.kernel_process_birth_id(),peer_pid=os.getpid(),ttl_seconds=20)
    arguments.update(changed)
    return context["gateway"].issue_native_execution_status_grant(**arguments)


def client(context, token, capability):
    return typed.TypedStateOwnerConnection(socket_path=context["gateway"].socket_path,token=token,
        client_id=capability.client_id,process_birth_id=capability.process_birth_id,
        store_id=context["gateway"].store_id)


def test_genuine_socket_reads_exact_native_attempt_preserves_float_and_unknown(ready):
    store=ready["store"]
    before=digest(store["path"]),identity(store["path"])
    token,capability=grant(ready)
    assert len(ready["connection"].execute("SELECT body_json FROM tasks").fetchone()[0].encode()) > 16_384
    assert capability.allowed_operations == frozenset({typed.NATIVE_EXECUTION_STATUS_OPERATION})
    connection=client(ready,token,capability)
    try:
        result=connection.native_execution_status_snapshot(binding_cid=ready["binding"].binding_cid)
        assert result["control_join_verified"] is True
        observation=result["native_observation"]
        assert observation["attempt"]["attempt_number"] == 2
        assert observation["attempt"]["body"]["feature_score"] == 0.75
        assert observation["provider_records"][0]["result"]["numeric_diagnostic"] == 0.5
        assert observation["native_store_uuid"] is None
        assert observation["provider_effect_state"] == "unknown"
        assert observation["native_callback_authority_observed"] is False
        assert observation["coordination_authority_observed"] is False
        assert result["automatic_retry_admitted"] is False
        assert result["semantic_admission"] is False
    finally:
        connection.close()
    assert (digest(store["path"]),identity(store["path"])) == before
    row=ready["connection"].execute("SELECT status,revision FROM tasks").fetchone()
    assert (row["status"],row["revision"]) == ("in_progress",12)
    assert ready["gateway"]._native_execution_control_task(ready["binding"])["body"]["completion_receipt"]["attempt_id"] == ready["binding"].attempt_id
    assert type(ready["gateway"]._native_execution_status_owner) is NativeExecutionStatusOwner


def test_primary_status_bootstrap_never_gets_native_read(ready):
    assert typed.NATIVE_EXECUTION_STATUS_OPERATION not in typed.DATABASE_STATUS_ALLOWED_OPERATIONS
    assert typed.NATIVE_EXECUTION_STATUS_OPERATION not in typed.STATUS_BOOTSTRAP_ALLOWED_OPERATIONS
    connection=typed.TypedStateOwnerConnection(socket_path=ready["gateway"].socket_path,
        token=ready["status_token"],client_id=typed.STATUS_BOOTSTRAP_CLIENT_ID,
        process_birth_id="birth:fixture-primary-status",store_id="control.duckdb",status_bootstrap=True)
    try:
        with pytest.raises(typed.TypedStateOwnerAuthorizationError):
            connection.native_execution_status_snapshot(binding_cid=ready["binding"].binding_cid)
        with pytest.raises(typed.TypedStateOwnerRemoteError):
            connection._request(typed.NATIVE_EXECUTION_STATUS_OPERATION,payload={"binding_cid":ready["binding"].binding_cid})
    finally:
        connection.close()


@pytest.mark.parametrize("change",[
    {"client_id":"ordinary-worker"}, {"allowed_operations":("native_execution.status.snapshot","whoami_metadata")},
    {"entity_scopes":{"task_cid":"other",typed.NATIVE_EXECUTION_STATUS_SCOPE:"sha256:"+"0"*64}},
    {"process_birth_id":"birth:asserted-audit-label"}, {"tenant_id":"tenant:extra"},
])
def test_generic_issuer_cannot_widen_native_profile(ready,change):
    arguments=dict(client_id=typed.NATIVE_EXECUTION_STATUS_CLIENT_ID,process_birth_id=typed.kernel_process_birth_id(),
        allowed_operations=(typed.NATIVE_EXECUTION_STATUS_OPERATION,),peer_pid=os.getpid(),ttl_seconds=20,
        entity_scopes={"task_cid":ready["binding"].task_cid,typed.NATIVE_EXECUTION_STATUS_SCOPE:ready["binding"].binding_cid})
    arguments.update(change)
    before=len(ready["gateway"]._grants)
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        ready["gateway"].issue_grant(**arguments)
    assert len(ready["gateway"]._grants) == before


@pytest.mark.parametrize("payload",[
    {}, {"binding_cid":"sha256:"+"f"*64}, {"binding_cid":"placeholder","path":"/private"},
    {"binding_cid":"placeholder","sql":"SELECT 1"}, {"binding_cid":"placeholder","limit":1},
])
def test_rpc_rejects_unbound_paths_sql_limits_and_selectors(ready,payload,monkeypatch):
    token,capability=grant(ready)
    connection=client(ready,token,capability)
    selected={key:(ready["binding"].binding_cid if value=="placeholder" else value) for key,value in payload.items()}
    service=ready["gateway"]._native_execution_status_owner
    def no_observation():
        raise AssertionError("unauthorized payload must not query native store")
    monkeypatch.setattr(service,"observe_attempt",no_observation)
    try:
        with pytest.raises(typed.TypedStateOwnerRemoteError):
            connection._request(typed.NATIVE_EXECUTION_STATUS_OPERATION,payload=selected)
    finally:
        connection.close()


def test_no_renewal_past_component_lifetime(ready):
    _token,capability=grant(ready)
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        ready["gateway"].renew_grant(capability.grant_id,ttl_seconds=30)
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        grant(ready,ttl_seconds=61)


def test_revoked_grant_cannot_observe(ready):
    token,capability=grant(ready)
    connection=client(ready,token,capability)
    ready["gateway"].revoke_grant(capability.grant_id)
    try:
        with pytest.raises(typed.TypedStateOwnerRemoteError):
            connection.native_execution_status_snapshot(binding_cid=ready["binding"].binding_cid)
    finally:
        connection.close()


@pytest.mark.parametrize("field,value",[("database_uuid","other-uuid"),("schema_revision",999),("schema_fingerprint","other")])
def test_wrong_control_identity_refuses_before_native_open(unstarted,field,value):
    scope={**unstarted["control_identity"],field:value}
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        unstarted["gateway"].bind_native_execution_status_before_start(binding=unstarted["binding"].to_dict(),
            control_identity=scope,deadline_monotonic=time.monotonic()+60)
    assert unstarted["gateway"]._native_execution_status_owner is None


def test_control_revision_drift_never_becomes_nondispatch(ready):
    token,capability=grant(ready)
    connection=client(ready,token,capability)
    service=ready["gateway"]._native_execution_status_owner
    ready["connection"].execute("UPDATE tasks SET revision=13")
    try:
        with pytest.raises(typed.TypedStateOwnerRemoteError):
            connection.native_execution_status_snapshot(binding_cid=ready["binding"].binding_cid)
    finally:
        connection.close()
    assert service._connection is None
    assert ready["gateway"]._native_execution_status_invalidated is True
    assert ready["connection"].execute("SELECT revision FROM tasks").fetchone()[0] == 13


def test_actual_primary_generation_drift_refuses(ready):
    ready["connection"].execute("UPDATE store_generations SET database_uuid='wrong'")
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        grant(ready)


def test_stop_releases_genuine_owner_after_clients_and_never_reopens(ready):
    service=ready["gateway"]._native_execution_status_owner
    ready["gateway"].stop()
    assert service._connection is None and service._lock_fd is None and service._database_fd is None
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        grant(ready)


def test_prestart_binding_cannot_be_rebound_or_added_to_live_gateway(unstarted):
    gateway=unstarted["gateway"]
    gateway.start()
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        gateway.bind_native_execution_status_before_start(binding=unstarted["binding"].to_dict(),
            control_identity=unstarted["control_identity"],deadline_monotonic=time.monotonic()+60)


def test_quack_configuration_freezes_binding_without_opening_native_store(unstarted,tmp_path):
    server=QuackStateServer(config=QuackStateServerConfig(database_path=tmp_path/"new-only.duckdb",state_dir=tmp_path/"server-state"))
    server.configure_database_status_before_start(board_namespace="board:sealed",plan_root_cid="plan:sealed",
        repository_tree_id="tree:sealed",task_cids=[unstarted["binding"].task_cid])
    binding=unstarted["binding"].to_dict()
    server.configure_native_execution_status_before_start(binding=binding,control_identity=unstarted["control_identity"],
        deadline_monotonic=time.monotonic()+60)
    sealed=server._native_execution_status_startup_json
    binding["control_revision"]=999
    assert json.loads(sealed)["binding"]["control_revision"] == 12
    assert server._command_gateway is None
    assert not (tmp_path/"new-only.duckdb").exists()
    with pytest.raises(QuackStateServerControlError):
        server.configure_native_execution_status_before_start(binding=unstarted["binding"].to_dict(),
            control_identity=unstarted["control_identity"],deadline_monotonic=time.monotonic()+60)


@pytest.mark.parametrize("operations",[(),("whoami_metadata",),"catalog"])
def test_existing_default_and_catalog_grants_do_not_inherit_native_status(unstarted,operations):
    gateway=unstarted["gateway"]
    selected=tuple(gateway.catalog) if operations=="catalog" else operations
    token,capability=gateway.issue_grant(client_id="ordinary-fixture-worker",process_birth_id="birth:ordinary",
        allowed_operations=selected,peer_pid=os.getpid(),ttl_seconds=20)
    assert capability.authority_profile == ""
    assert capability.allowed_operations == frozenset(selected)
    assert typed.NATIVE_EXECUTION_STATUS_OPERATION not in capability.allowed_operations
    assert token in gateway._grants


def test_ordinary_reader_keeps_existing_behavior_with_native_owner_present(ready):
    token,capability=ready["gateway"].issue_grant(client_id="ordinary-fixture-reader",process_birth_id="birth:ordinary",
        allowed_operations=("whoami_metadata",),peer_pid=os.getpid(),ttl_seconds=20)
    connection=client(ready,token,capability)
    try:
        assert connection.grant["authority_profile"] == ""
        assert connection.execute_operation("whoami_metadata").fetchone() is not None
        with pytest.raises(typed.TypedStateOwnerAuthorizationError):
            connection.native_execution_status_snapshot(binding_cid=ready["binding"].binding_cid)
    finally:
        connection.close()


def test_startup_recheck_close_fault_retains_real_owner_until_shutdown(unstarted,monkeypatch):
    original_open=NativeExecutionStatusOwner.open
    original_close=NativeExecutionStatusOwner.close
    observed=[]
    close_calls=[]
    def open_and_drift(instance):
        original_open(instance)
        observed.append(instance)
        unstarted["connection"].execute("UPDATE tasks SET revision=13")
        return instance
    def first_close_refuses(instance):
        close_calls.append(instance)
        if len(close_calls)==1:
            raise RuntimeError("disposable close fault")
        return original_close(instance)
    monkeypatch.setattr(NativeExecutionStatusOwner,"open",open_and_drift)
    monkeypatch.setattr(NativeExecutionStatusOwner,"close",first_close_refuses)
    gateway=unstarted["gateway"]
    with pytest.raises(RuntimeError,match="disposable close fault"):
        gateway.bind_native_execution_status_before_start(binding=unstarted["binding"].to_dict(),
            control_identity=unstarted["control_identity"],deadline_monotonic=time.monotonic()+60)
    assert len(observed)==1
    service=observed[0]
    assert type(service) is NativeExecutionStatusOwner
    assert gateway._native_execution_status_owner is service
    assert service._connection is not None
    assert gateway._native_execution_status_invalidated is True
    gateway.stop()
    assert gateway._native_execution_status_owner is None
    assert service._connection is None and service._lock_fd is None and service._database_fd is None


def test_quack_cleanup_failure_keeps_genuine_gateway_for_retry(ready,tmp_path,monkeypatch):
    # New disposable server facet owns the actual fresh fixture gateway. No
    # historical daemon object/handle or native admission is reconstructed.
    server=QuackStateServer(config=QuackStateServerConfig(database_path=ready["db"],state_dir=tmp_path/"facet-state"))
    gateway=ready["gateway"]
    service=gateway._native_execution_status_owner
    original_close=service.close
    calls=[]
    def first_close_refuses():
        calls.append(True)
        if len(calls)==1:
            raise RuntimeError("disposable native close fault")
        return original_close()
    monkeypatch.setattr(service,"close",first_close_refuses)
    server._command_gateway=gateway
    with pytest.raises(RuntimeError,match="disposable native close fault"):
        server._stop_worker_facets()
    assert server._command_gateway is gateway
    assert gateway._native_execution_status_owner is service
    assert service._connection is not None
    server._stop_worker_facets()
    assert server._command_gateway is None
    assert service._connection is None and service._lock_fd is None and service._database_fd is None


def test_persistent_startup_cleanup_fault_retains_owner_then_supported_stop_retries(ready,tmp_path,monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import ServerLifecycle
    server=QuackStateServer(config=QuackStateServerConfig(database_path=ready["db"],state_dir=tmp_path/"emergency-state"))
    gateway=ready["gateway"]
    service=gateway._native_execution_status_owner
    original_close=service.close
    pending={"value":True}
    def close_while_pending():
        if pending["value"]:
            raise RuntimeError("disposable persistent native close fault")
        return original_close()
    monkeypatch.setattr(service,"close",close_while_pending)
    # Genuine newly created fixture handles are installed on a new local server
    # solely to exercise the existing emergency/STOP cleanup path.
    server._command_gateway=gateway
    server._connection=ready["connection"]
    server._lifecycle=ServerLifecycle.FAILED
    server._emergency_cleanup()
    assert server._command_gateway is gateway
    assert server._connection is ready["connection"]
    assert service._connection is not None
    with pytest.raises(RuntimeError,match="disposable persistent native close fault"):
        server.stop()
    assert server.lifecycle is ServerLifecycle.FAILED
    assert server._command_gateway is gateway
    assert server._connection is ready["connection"]
    pending["value"]=False
    stopped=server.stop()
    assert stopped["stopped"] is True
    assert server.lifecycle is ServerLifecycle.STOPPED
    assert server._command_gateway is None
    assert service._connection is None and service._lock_fd is None and service._database_fd is None
