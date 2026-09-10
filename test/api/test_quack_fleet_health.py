import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import quack_fleet_health as health


def test_gateway_failure_requires_repetition_and_bounded_backoff():
    failure = {"healthy": False, "reason": "typed_gateway_timeout"}
    first = health.recovery_decision({}, failure, now=1000)
    assert first["restart_requested"] is False
    second = health.recovery_decision(first, failure, now=1030)
    assert second["restart_requested"] is True
    second["restart_times"] = [1030]
    assert health.recovery_decision(second, failure, now=1050)["restart_requested"] is False
    second["restart_times"] = [500, 900, 1030]
    assert health.recovery_decision(second, failure, now=1500)["restart_requested"] is False
    assert health.recovery_decision(second, {"healthy": True}, now=1500)["consecutive_failures"] == 0
    incompatible = {"healthy": False, "reason": "derived_capability_mismatch", "restartable": False}
    assert health.recovery_decision(first, incompatible, now=1500)["restart_requested"] is False


def test_probe_detects_live_but_hung_gateway(tmp_path, monkeypatch):
    identity = {"process_birth": {"pid": 22}, "process_birth_id": "birth:test", "database_uuid": "uuid:test", "generation": 1}
    (tmp_path / "quack-state-server.status.json").write_text(json.dumps({"lifecycle": "ready", "identity": identity}))
    (tmp_path / "derived-coordination.token").write_text("private-test-credential")
    monkeypatch.setattr(health, 'birth_matches', lambda one,two: True)
    monkeypatch.setattr(health, 'process_identity', lambda pid: identity['process_birth'])
    def hung(*args, **kwargs):
        assert kwargs['timeout_seconds'] == 5
        raise TimeoutError('no native reply')
    monkeypatch.setattr(health, 'attach_typed_instance', hung)
    deployment = {'instances': {'derived_coordination': {'state_dir': str(tmp_path), 'database_path': str(tmp_path / 'control.duckdb')}}}
    result = health.probe_owner(deployment, 'derived_coordination')
    assert result == {'healthy': False, 'reason': 'typed_gateway_unavailable:TimeoutError'}


def test_health_cannot_target_taskboard_services():
    assert set(health.SERVICES.values()) == {'ipfs-quack-fleet-aggregate-control.service', 'ipfs-quack-fleet-derived-coordination.service'}
    with pytest.raises(ValueError, match='unknown dedicated'):
        health.probe_owner({}, 'spar')


def test_derived_health_checks_service_even_when_generation_read_succeeds(tmp_path, monkeypatch):
    identity = {"process_birth": {"pid": 22}, "process_birth_id": "birth:test",
                "database_uuid": "uuid:test", "generation": 1, "server_id": "server:test"}
    (tmp_path / "quack-state-server.status.json").write_text(json.dumps({"lifecycle": "ready", "identity": identity}))
    (tmp_path / "derived-coordination.token").write_text("private-test-credential")
    monkeypatch.setattr(health, "birth_matches", lambda one, two: True)
    monkeypatch.setattr(health, "process_identity", lambda pid: identity["process_birth"])
    closed = []

    def unavailable(payload):
        assert payload == {"operation": "list_references", "repository_id": "fleet:health", "tree_id": "fleet:health"}
        raise TimeoutError("derived service unavailable")

    client = SimpleNamespace(load_generation=lambda: SimpleNamespace(generation=1, database_uuid="uuid:test"),
                             derived_coordination=unavailable, close=lambda: closed.append(True))
    monkeypatch.setattr(health, "attach_typed_instance", lambda *args, **kwargs: client)
    deployment = {"instances": {"derived_coordination": {"state_dir": str(tmp_path), "database_path": str(tmp_path / "control.duckdb")}}}
    assert health.probe_owner(deployment, "derived_coordination") == {"healthy": False, "reason": "typed_gateway_unavailable:TimeoutError"}
    assert closed == [True]


@pytest.mark.parametrize("mismatch", [None, "artifact_schema", "artifact_kinds", "operations", "unsupported"])
def test_health_verifies_advertised_artifact_capabilities(tmp_path, monkeypatch, mismatch):
    identity = {"process_birth": {"pid": 22}, "process_birth_id": "birth:test",
                "database_uuid": "uuid:test", "generation": 1, "server_id": "server:test"}
    (tmp_path / "quack-state-server.status.json").write_text(json.dumps({"lifecycle": "ready", "identity": identity}))
    (tmp_path / "derived-coordination.token").write_text("private-test-credential")
    monkeypatch.setattr(health, "birth_matches", lambda one, two: True)
    monkeypatch.setattr(health, "process_identity", lambda pid: identity["process_birth"])
    capabilities = {"artifact_schema": "schema:test", "artifact_kinds": ["certificate"],
                    "operations": ["record_artifact", "lookup_artifact", "list_artifacts"]}
    if mismatch:
        capabilities[mismatch] = "wrong" if mismatch == "artifact_schema" else []
    calls, closed = [], []

    def derived(payload):
        calls.append(payload["operation"])
        if mismatch == "unsupported" and payload["operation"] == "capabilities":
            raise health.TypedStateOwnerRemoteError("operation_failed", "ValueError")
        return {"result": capabilities if payload["operation"] == "capabilities" else {"references": []}}

    client = SimpleNamespace(load_generation=lambda: SimpleNamespace(generation=1, database_uuid="uuid:test"),
                             session=SimpleNamespace(server_id="server:test"),
                             derived_coordination=derived, close=lambda: closed.append(True))
    monkeypatch.setattr(health, "attach_typed_instance", lambda *args, **kwargs: client)
    deployment = {"instances": {"derived_coordination": {"state_dir": str(tmp_path), "database_path": str(tmp_path / "control.duckdb")}},
                  "derived_coordination": {"artifact_schema": "schema:test", "artifact_kinds": ["certificate"]}}
    result = health.probe_owner(deployment, "derived_coordination")
    assert calls == ["list_references", "capabilities"]
    assert closed == [True]
    if mismatch:
        assert result == {"healthy": False, "reason": "derived_capability_mismatch", "restartable": False}
    else:
        assert result["healthy"] is True
        assert result["verified_artifact_kinds"] == ["certificate"]
