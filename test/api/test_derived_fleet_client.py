"""Fleet-derived clients renew native credentials without replay or fallback."""

import json
from pathlib import Path

import pytest

from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from ipfs_accelerate_py.agent_supervisor.analysis.derived_coordination import DerivedCoordinationClient
from ipfs_accelerate_py.agent_supervisor.runtime import derived_fleet_client as fleet_client
from ipfs_accelerate_py.agent_supervisor.runtime.quack_fleet_topology import DEFAULTS, SCHEMA
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_FILENAME, TypedStateOwnerError, compact_default_owner_socket_path,
)


@pytest.fixture
def managed_owner(tmp_path):
    database, state = tmp_path / "control.duckdb", tmp_path / "owner"
    state.mkdir()
    _install(database)
    socket = compact_default_owner_socket_path(state / TYPED_STATE_OWNER_SOCKET_FILENAME, identity=database)
    owners = []

    def start():
        gateway, connection = _gateway(database, socket)
        owners.append((gateway, connection))
        token = state / "derived-coordination.token"
        token.write_text(gateway.bind_derived_coordination_service())
        token.chmod(0o600)
        return gateway, connection

    deployment = tmp_path / "deployment.json"
    payload = {"schema": SCHEMA, "instances": {"derived_coordination": {
        "managed_by_fleet": True, "database_path": str(database), "state_dir": str(state),
        "database_program": {**DEFAULTS, "store_id": "control.duckdb",
            "quack_endpoint": "quack:127.0.0.1:12345", "endpoint_secret_handle": "handle:test",
            "store_generation": "1", "schema_revision": "1"}}}}
    deployment.write_text(json.dumps(payload))
    yield deployment, state, start
    for gateway, connection in owners:
        gateway.stop()
        connection.close()


def test_same_client_recovers_after_native_owner_replacement(managed_owner, monkeypatch):
    path, state, start = managed_owner
    gateway, connection = start()
    api = DerivedCoordinationClient.from_fleet_deployment(path, repository_id="repo:one", client_id="supervisor:one")
    metadata = dict(tree_id="tree:pinned", artifact_kind="proof_cache", input_digest="sha256:" + "a" * 64,
                    producer_id="test-producer", producer_revision="rev:1", parameters_digest="sha256:" + "b" * 64)
    first = api.call("record_artifact", **metadata, artifact_cid="cid:test")
    old_token = (state / "derived-coordination.token").read_text()
    gateway.stop()
    connection.close()
    # Neither loss of transport nor a credential transition may create another
    # writer or replay the publication. The caller chooses each new operation.
    with monkeypatch.context() as context:
        import duckdb
        context.setattr(duckdb, "connect", lambda *a, **kw: pytest.fail("client opened database"))
        with pytest.raises((OSError, TypedStateOwnerError)):
            api.call("lookup_artifact", **metadata)
    start()
    assert (state / "derived-coordination.token").read_text() != old_token
    assert api.call("lookup_artifact", **metadata)["result"] == first["result"]
    other = DerivedCoordinationClient.from_fleet_deployment(path, repository_id="repo:two", client_id="supervisor:two")
    assert other.call("lookup_artifact", **metadata)["result"]["artifact"] is None


def test_factory_does_not_replay_a_failed_write(tmp_path, monkeypatch):
    path, state = tmp_path / "deployment.json", tmp_path / "owner"
    state.mkdir()
    (state / "derived-coordination.token").write_text("test-credential")
    path.write_text(json.dumps({"schema": SCHEMA, "instances": {"derived_coordination": {
        "managed_by_fleet": True, "state_dir": str(state), "database_path": str(tmp_path / "db")}}}))
    calls, closed = [], []

    class Client:
        def derived_coordination(self, payload):
            calls.append(payload)
            raise TimeoutError("response lost after publication")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(fleet_client, "attach_typed_instance", lambda *a, **kw: Client())
    api = DerivedCoordinationClient.from_fleet_deployment(path, repository_id="repo:one", client_id="supervisor:one")
    with pytest.raises(TimeoutError):
        api.call("record_artifact", artifact_cid="cid:test")
    assert len(calls) == 1 and closed == [True]


@pytest.mark.parametrize("violation", ["symlink", "fifo", "oversized", "unmanaged", "relative_owner"])
def test_fleet_factory_rejects_invalid_binding_without_attachment(tmp_path, monkeypatch, violation):
    import os

    path = tmp_path / "deployment.json"
    payload = {"schema": SCHEMA, "instances": {"derived_coordination": {
        "managed_by_fleet": True, "state_dir": str(tmp_path), "database_path": str(tmp_path / "db")}}}
    if violation == "symlink":
        target = tmp_path / "target"
        target.write_text(json.dumps(payload))
        path.symlink_to(target)
    elif violation == "fifo":
        os.mkfifo(path)
    elif violation == "oversized":
        path.write_bytes(b" " * (8 * 1024 * 1024 + 1))
    else:
        if violation == "unmanaged": payload["instances"]["derived_coordination"]["managed_by_fleet"] = False
        if violation == "relative_owner": payload["instances"]["derived_coordination"]["state_dir"] = "relative"
        path.write_text(json.dumps(payload))
    monkeypatch.setattr(fleet_client, "attach_typed_instance", lambda *a, **kw: pytest.fail("invalid attachment"))
    api = DerivedCoordinationClient.from_fleet_deployment(path, repository_id="repo:one", client_id="supervisor:one")
    with pytest.raises((ValueError, OSError)):
        api.call("capabilities")


@pytest.mark.parametrize("timeout", [True, 0, -1, 31, float("inf"), float("nan")])
def test_factory_rejects_unbounded_timeouts(tmp_path, timeout):
    with pytest.raises(ValueError):
        DerivedCoordinationClient.from_fleet_deployment(tmp_path / "deployment.json", repository_id="repo:one",
            client_id="supervisor:one", timeout_seconds=timeout)


def test_factory_requires_explicit_absolute_deployment():
    with pytest.raises(ValueError):
        DerivedCoordinationClient.from_fleet_deployment(Path("deployment.json"), repository_id="repo:one", client_id="supervisor:one")
