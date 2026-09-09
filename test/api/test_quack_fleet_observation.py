# Retain datetime parsing compatibility with supported Python runtimes.
# ruff: noqa: UP017
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.federation import (
    fleet_observation as observation,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_fleet_observer import (
    read_native_source,
)


@pytest.fixture
def store(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_control_plane_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
        QuackStateClient,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TypedStateOwnerConnection,
        TypedStateOwnerGateway,
        build_control_plane_operation_catalog,
    )
    db = tmp_path / 'control.duckdb'
    install_control_plane_schema(db, application_version='0.0.45', tool_version='1.5.2', owner_id='test:fleet')
    seed = QuackStateClient(owner_id='test:seed', store_id='control.duckdb')
    seed.attach(db, seed_generation=True); seed.close()
    connection = open_duckdb_connection(db)
    row = connection.execute('SELECT generation, schema_revision, fence_epoch, revision, database_uuid, birth_id FROM store_generations ORDER BY generation DESC LIMIT 1').fetchone()
    identity = {'server_id': 'server:fleet-test', 'store_id': 'control.duckdb', 'generation': row[0], 'schema_revision': row[1],
                'fence_epoch': row[2], 'revision': row[3], 'database_uuid': row[4], 'process_birth_id': row[5] or 'birth:owner'}
    socket = tmp_path / 'owner.sock'
    gateway = TypedStateOwnerGateway(connection=connection, socket_path=socket, store_id='control.duckdb', identity=identity)
    gateway.start()
    token, _ = gateway.issue_grant(client_id='test:fleet', process_birth_id='birth:test', allowed_operations=tuple(build_control_plane_operation_catalog()),
                                    allowed_command_operations=(observation.OPERATION,), peer_pid=os.getpid())
    def connect(endpoint):
        return TypedStateOwnerConnection(socket_path=socket, token=token, client_id='test:fleet', process_birth_id='birth:test', store_id='control.duckdb')
    client = QuackStateClient(owner_id='test:fleet', store_id='control.duckdb', process_birth_id='birth:test', connection_factory=connect)
    client.attach('quack:127.0.0.1:7777')
    try:
        yield observation.FleetObservationStore(client), connection
    finally:
        client.close(); gateway.stop(); connection.close()


def sample(source_id='spar', available=True, now=None):
    return {'schema': observation.SCHEMA, 'source_id': source_id, 'observed_at': (now or datetime.now(timezone.utc)).isoformat(),
            'availability': 'available' if available else 'unavailable', 'completion_authority': False,
            'source_identity': {'database_uuid': 'uuid:test', 'generation': 1, 'process_birth_id': 'birth:source', 'listen_uri': 'quack:127.0.0.1:7778'},
            'native_receipt': {'authority': {'status_counts': {'completed': 2, 'blocked': 1}}} if available else {},
            'reason': '' if available else 'native_read_not_admitted'}


def test_native_quack_roundtrip_retains_admitted_history_and_staleness(store):
    projection, connection = store
    now = datetime.now(timezone.utc)
    admitted = sample(now=now)
    projection.record(admitted)
    projection.record(sample(available=False, now=now + timedelta(seconds=1)))
    projection.record(sample('aseh', now=now))
    view = projection.view(['spar', 'aseh', 'pcpr'], now=now + timedelta(seconds=2))
    assert view['sources']['spar']['available'] is False
    assert view['sources']['spar']['last_admitted'] == admitted
    assert view['sources']['aseh']['available'] is True
    assert view['sources']['pcpr']['reason'] == 'source_observation_absent'
    assert projection.view(['aseh'], now=now + timedelta(seconds=61))['sources']['aseh']['reason'] == 'source_observation_stale'
    assert connection.execute('SELECT COUNT(*) FROM tasks').fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM federation_receipts WHERE receipt_kind='fleet_source_observation'").fetchone()[0] == 3


def test_native_observation_record_is_idempotent(store):
    projection, connection = store
    value = sample()
    projection.record(value); projection.record(value)
    assert connection.execute("SELECT COUNT(*) FROM artifacts WHERE kind='fleet_source_observation'").fetchone()[0] == 1


def test_native_manifest_refuses_authority_promotion_atomically(store, monkeypatch):
    projection, connection = store
    original = projection.client.execute
    def corrupt(name, bound):
        if name == 'fleet_insert_observation_artifact':
            payload = json.loads(bound['payload_json']); payload['completion_authority'] = True
            bound = {**bound, 'payload_json': observation.canonical(payload)}
        return original(name, bound)
    monkeypatch.setattr(projection.client, 'execute', corrupt)
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
        TransactionError,
    )
    with pytest.raises(TransactionError):
        projection.record(sample())
    assert connection.execute("SELECT COUNT(*) FROM artifacts WHERE kind='fleet_source_observation'").fetchone()[0] == 0
    assert connection.execute('SELECT COUNT(*) FROM federation_receipts').fetchone()[0] == 0


@pytest.mark.parametrize('native,expected', [
    ({'status_age_seconds': 0, 'task_authority': {'available': True, 'authenticated_query': True, 'status_counts': {'completed': 2}}}, 'available'),
    ({'status_age_seconds': 0, 'task_authority': {'available': False, 'transport': 'exclusive_owner_authenticated_quack_projection'}}, 'unavailable'),
    ({'status_age_seconds': 31, 'task_authority': {'available': True, 'authenticated_query': True}}, 'unavailable'),
    ({'task_authority': {'available': True, 'authenticated_query': True}}, 'unavailable'),
    ({'task_authority': {'status_counts': {'completed': 2}}}, 'unavailable'),
    ({'task_authority': {}, 'daemon_projection': {'status_counts': {'completed': 2}}}, 'unavailable'),
])
def test_source_adapter_admits_native_authenticated_reads_only(tmp_path, native, expected):
    db, config = tmp_path / 'control.duckdb', tmp_path / 'config.json'
    db.touch(); config.write_text('{}')
    identity = {'database_uuid': 'uuid:test', 'generation': 1, 'process_birth_id': 'birth:owner', 'listen_uri': 'quack:127.0.0.1:7777', 'process_birth': {'pid': 1}}
    status = {'lifecycle': 'ready', 'identity': identity}
    adapter = SimpleNamespace(read_json=lambda path: status, birth_matches=lambda one,two: True, process_identity=lambda pid: identity['process_birth'],
                              _status_with_receipt_retry=lambda board,birth: (native, '', 1))
    board = {'id': 'spar', 'database_path': str(db), 'config_path': str(config), 'owner_status_path': str(tmp_path / 'owner.json'), 'quack_endpoint': identity['listen_uri']}
    value = read_native_source(board, adapter=adapter)
    assert value['availability'] == expected
    assert value['completion_authority'] is False
    if expected == 'unavailable': assert value['native_receipt'] == {}


def test_missing_native_database_never_calls_status(tmp_path):
    def forbidden(*args): raise AssertionError('missing native source queried')
    adapter = SimpleNamespace(read_json=forbidden)
    result = read_native_source({'id': 'pcpr', 'database_path': str(tmp_path / 'missing'), 'config_path': str(tmp_path / 'missingconfig')}, adapter=adapter)
    assert result['reason'] == 'native_configuration_or_database_missing'
    assert not (tmp_path / 'missing').exists()


def test_source_owner_dies_during_query_even_with_retained_status(tmp_path):
    db, config = tmp_path / "control.duckdb", tmp_path / "config.json"
    db.touch(); config.write_text("{}")
    identity = {"database_uuid": "uuid:test", "generation": 1, "process_birth_id": "birth:owner", "listen_uri": "quack:127.0.0.1:7777", "process_birth": {"pid": 1}}
    status = {"lifecycle": "ready", "identity": identity}
    alive = {"value": True}
    def query(board, birth):
        alive["value"] = False
        return {"status_age_seconds": 0, "task_authority": {"available": True, "authenticated_query": True}}, "", 1
    adapter = SimpleNamespace(read_json=lambda path: status, birth_matches=lambda one,two: bool(one),
                              process_identity=lambda pid: identity["process_birth"] if alive["value"] else {},
                              _status_with_receipt_retry=query)
    board = {"id": "spar", "database_path": str(db), "config_path": str(config), "owner_status_path": str(tmp_path / "owner.json"), "quack_endpoint": identity["listen_uri"]}
    value = read_native_source(board, adapter=adapter)
    assert value["availability"] == "unavailable"
    assert value["reason"] == "native_owner_changed_during_query"
    assert value["native_receipt"] == {}


@pytest.mark.parametrize("mutation,available", [
    (lambda n: None, True),
    (lambda n: n["bootstrap_broker"].update(launch_id="foreign"), False),
    (lambda n: n["handoff"]["owner_identity"].update(generation=9), False),
    (lambda n: n["task_authority"].update(updated_at="2000-01-01T00:00:00+00:00"), False),
])
def test_doep_native_monitor_requires_current_exact_broker_binding(mutation, available):
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_fleet_observer import (
        _doep_native_authority,
    )
    now = datetime.now(timezone.utc)
    identity = {"server_id": "server:test", "database_uuid": "uuid:test", "generation": 1, "process_birth_id": "birth:test", "listen_uri": "quack:127.0.0.1:7777", "process_birth": {"pid": 44}}
    live = {"schema": "ipfs_accelerate_py/agent-supervisor/doep-live-status@1", "credential_transport": "private_inherited_socket", "owner_ready": True,
            "raw_token_in_evidence": False, "launch_id": "launch:test", "monitor_pid": 44, "owner_server_id": "server:test", "plan_root_cid": "plan:test",
            "repository_tree_id": "tree:test", "updated_at": now.isoformat(), "blocked": True}
    native = {"schema": "ipfs_accelerate_py/agent-supervisor/doep-bootstrap-handoff@1", "operator_alive": True, "operator_pid": 44, "task_authority": live,
              "bootstrap_broker": {"schema": "ipfs_accelerate_py/agent-supervisor/doep-bootstrap-broker@1", "ready": True, "launch_id": "launch:test", "operator_pid": 44,
                                   "server_id": "server:test", "state_owner_process_birth_id": "birth:test"},
              "handoff": {"owner_identity": dict(identity), "launch_id": "launch:test", "operator_pid": 44, "plan_root_cid": "plan:test", "repository_tree_id": "tree:test"}}
    mutation(native)
    assert bool(_doep_native_authority(native, identity, now=now)) is available
