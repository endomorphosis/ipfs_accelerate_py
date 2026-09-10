import json
import os
import sys
import time
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
    identity = {"process_birth": {"pid": 22, "start_time_ticks": 1, "boot_id": "boot:test"}, "process_birth_id": "birth:test", "database_uuid": "uuid:test", "generation": 1}
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
    identity = {"process_birth": {"pid": 22, "start_time_ticks": 1, "boot_id": "boot:test"}, "process_birth_id": "birth:test",
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
    identity = {"process_birth": {"pid": 22, "start_time_ticks": 1, "boot_id": "boot:test"}, "process_birth_id": "birth:test",
                "database_uuid": "uuid:test", "generation": 1, "server_id": "server:test"}
    (tmp_path / "quack-state-server.status.json").write_text(json.dumps({"lifecycle": "ready", "identity": identity}))
    (tmp_path / "derived-coordination.token").write_text("private-test-credential")
    monkeypatch.setattr(health, "birth_matches", lambda one, two: True)
    monkeypatch.setattr(health, "process_identity", lambda pid: identity["process_birth"])
    monkeypatch.setattr(health, "_canonical_writer_lock", lambda *_args: {"verified": True, "held": True})
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


@pytest.mark.parametrize("role", ["aggregate_control", "derived_coordination"])
def test_actual_fleet_health_detects_lost_lock_and_new_owner_reacquisition(tmp_path, monkeypatch, role):
    """Only this test's newly created store is opened to reproduce lock loss."""
    from ipfs_accelerate_py.agent_supervisor.runtime import quack_state_server as quack
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_fleet_topology import (
        DEFAULTS,
        SCHEMA,
    )

    pytest.importorskip("duckdb")
    if quack.probe_quack_capabilities(allow_network_install=False).status.value != "compatible":
        pytest.skip("preinstalled actual Quack unavailable")
    database, state = tmp_path / "control.duckdb", tmp_path / "owner"
    deployment = {"schema": SCHEMA, "instances": {role: {
        "managed_by_fleet": True, "database_path": str(database), "state_dir": str(state),
        "database_program": {**DEFAULTS, "store_id": "disposable-fleet-control",
            "store_generation": "1", "schema_revision": "1"}}}}
    servers = []

    def start():
        server = quack.build_server(database_path=database, state_dir=state,
            repository_id=f"fleet:{role}", store_id="disposable-fleet-control",
            secret_handle="handle:disposable-health-lock", allow_legacy_board_unstall=False)
        servers.append(server)
        identity = server.start()
        deployment["instances"][role]["database_program"].update(
            quack_endpoint=identity.listen_uri, endpoint_secret_handle=identity.secret_handle,
            store_generation=str(identity.generation))
        if role == "aggregate_control":
            server.bind_fleet_observation_reads()
        else:
            server.bind_derived_coordination_service()
        return server, identity

    try:
        first, identity = start()
        healthy = health.probe_owner(deployment, role)
        assert healthy["healthy"] is True
        assert first.checkpoint()["checkpointed"] is True
        assert health.probe_owner(deployment, role)["healthy"] is True
        # Linux closes process-scoped POSIX locks even when the closed fd was
        # an unrelated reader. This is the exact old replica-copy failure.
        descriptor = os.open(database, os.O_RDONLY)
        os.close(descriptor)
        missing = health.probe_owner(deployment, role)
        assert missing == {"healthy": False, "reason": "canonical_writer_lock_missing", "restartable": True}
        once = health.recovery_decision({}, missing, now=1000)
        assert once["restart_requested"] is False
        assert health.recovery_decision(once, missing, now=1030)["restart_requested"] is True
        assert first.stop()["stopped"] is True
        _second, replacement = start()
        assert replacement.generation > identity.generation
        recovered = health.probe_owner(deployment, role)
        assert recovered["healthy"] is True and recovered["canonical_writer_lock_verified"] is True
        assert recovered["generation"] == replacement.generation
        assert health.recovery_decision(once, recovered, now=1060)["restart_requested"] is False
        with monkeypatch.context() as context:
            def unreadable():
                raise PermissionError("kernel observation unavailable")
            context.setattr(health, "_read_kernel_locks", unreadable)
            uncertain = health.probe_owner(deployment, role)
            assert uncertain == {"healthy": False, "reason": "canonical_writer_lock_observation_unavailable", "restartable": False}
            assert health.recovery_decision({"consecutive_failures": 10}, uncertain, now=1090)["restart_requested"] is False
        assert health.probe_owner(deployment, role)["healthy"] is True
    finally:
        for server in reversed(servers):
            server.stop()


@pytest.fixture
def disposable_writer(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    database = tmp_path / "control.duckdb"
    connection = duckdb.connect(str(database), config={"threads": 1})
    connection.execute("CREATE TABLE preserved (value INTEGER)")
    connection.execute("INSERT INTO preserved VALUES (7)")
    connection.execute("CHECKPOINT")
    try:
        yield database, connection, health.process_identity(os.getpid())
    finally:
        connection.close()


@pytest.mark.parametrize("failure", ["permission", "malformed", "oversized", "birth_change", "inode_change"])
def test_uncertain_kernel_lock_observation_cannot_request_restart(disposable_writer, monkeypatch, failure):
    database, writer, birth = disposable_writer
    read_locks = health._read_kernel_locks
    if failure == "permission":
        def unavailable():
            raise PermissionError("kernel observation unavailable")
        monkeypatch.setattr(health, "_read_kernel_locks", unavailable)
    elif failure == "malformed":
        monkeypatch.setattr(health, "_read_kernel_locks", lambda: "incomplete kernel lock row")
    elif failure == "oversized":
        monkeypatch.setattr(health, "_MAX_KERNEL_LOCK_BYTES", 1)
    elif failure == "birth_change":
        calls = []
        def changed(_pid):
            calls.append(True)
            return birth if len(calls) == 1 else {**birth, "start_time_ticks": birth["start_time_ticks"] + 1}
        monkeypatch.setattr(health, "process_identity", changed)
    else:
        def changed_inode():
            raw = read_locks()
            database.rename(database.with_suffix(".preserved.duckdb"))
            database.write_bytes(b"different canonical inode")
            return raw
        monkeypatch.setattr(health, "_read_kernel_locks", changed_inode)
    observation = health._canonical_writer_lock(database, birth)
    assert observation["verified"] is False
    assert "held" not in observation
    assert writer.execute("SELECT value FROM preserved").fetchone()[0] == 7
    probe = {"healthy": False, "reason": observation["reason"], "restartable": False}
    assert health.recovery_decision({"consecutive_failures": 10}, probe, now=1000)["restart_requested"] is False


def test_read_only_holder_is_not_a_canonical_writer(disposable_writer):
    database, writer, birth = disposable_writer
    writer.close()
    import duckdb
    reader = duckdb.connect(str(database), read_only=True)
    try:
        assert health._canonical_writer_lock(database, birth) == {"verified": True, "held": False}
        assert reader.execute("SELECT value FROM preserved").fetchone()[0] == 7
    finally:
        reader.close()


def test_blocked_write_request_cannot_prove_writer_custody(disposable_writer, monkeypatch):
    database, _writer, birth = disposable_writer
    info = database.stat()
    row = f"1: -> POSIX ADVISORY WRITE {birth['pid']} {os.major(info.st_dev):x}:{os.minor(info.st_dev):x}:{info.st_ino} 0 EOF\n"
    monkeypatch.setattr(health, "_read_kernel_locks", lambda: row)
    assert health._canonical_writer_lock(database, birth) == {"verified": True, "held": False}


@pytest.mark.parametrize("failure", ["malformed", "null_birth", "null_identity", "unreadable", "changed"])
def test_initial_unknown_birth_cannot_request_restart(tmp_path, monkeypatch, failure):
    birth = health.process_identity(os.getpid())
    if failure == "malformed":
        birth = {"pid": os.getpid()}
    elif failure == "null_birth":
        birth = None
    elif failure == "unreadable":
        monkeypatch.setattr(health, "process_identity", lambda _pid: {})
    elif failure == "changed":
        monkeypatch.setattr(health, "process_identity", lambda _pid: {**birth, "start_time_ticks": birth["start_time_ticks"] + 1})
    (tmp_path / "quack-state-server.status.json").write_text(json.dumps({
        "lifecycle": "ready", "identity": None if failure == "null_identity" else {"process_birth": birth}}))
    monkeypatch.setattr(health, "attach_typed_instance", lambda *args, **kwargs: pytest.fail("unverified owner attachment"))
    deployment = {"instances": {"aggregate_control": {"state_dir": str(tmp_path)}}}
    result = health.probe_owner(deployment, "aggregate_control")
    assert result == {"healthy": False, "reason": "native_owner_birth_unverified", "restartable": False}
    assert health.recovery_decision({"consecutive_failures": 10}, result, now=1000)["restart_requested"] is False


def test_proven_absent_owner_retains_existing_recovery_policy(tmp_path, monkeypatch):
    birth = {"pid": 2147483647, "start_time_ticks": 1, "boot_id": "boot:test"}
    (tmp_path / "quack-state-server.status.json").write_text(json.dumps({
        "lifecycle": "ready", "identity": {"process_birth": birth}}))
    monkeypatch.setattr(health, "process_identity", lambda _pid: {})
    original_stat = os.stat
    def absent(path, *args, **kwargs):
        if path == f"/proc/{birth['pid']}":
            raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(health.os, "stat", absent)
    result = health.probe_owner({"instances": {"aggregate_control": {"state_dir": str(tmp_path)}}}, "aggregate_control")
    assert result == {"healthy": False, "reason": "native_owner_not_ready_or_alive"}
    assert health.recovery_decision({"consecutive_failures": 1}, result, now=1000)["restart_requested"] is True


def _prepare_health_main(tmp_path, monkeypatch):
    deployment = tmp_path / "deployment.json"
    deployment.write_text("{}")
    previous = {role: {"consecutive_failures": 1, "restart_times": []} for role in health.ROLES}
    (tmp_path / "gateway-health.json").write_text(json.dumps(previous))
    monkeypatch.setattr(sys, "argv", ["quack-fleet-health", "--deployment", str(deployment), "--apply"])
    monkeypatch.setattr(health.time, "time", lambda: 1000.0)
    return lambda: json.loads((tmp_path / "gateway-health.json").read_text())


@pytest.mark.parametrize("hold_name", ["HOLD", "OPERATOR_STOP"])
def test_health_rechecks_hold_created_during_native_probe(tmp_path, monkeypatch, hold_name):
    read_result = _prepare_health_main(tmp_path, monkeypatch)
    def held_probe(_deployment, _role):
        (tmp_path / hold_name).touch()
        return {"healthy": False, "reason": "canonical_writer_lock_missing", "restartable": True}
    monkeypatch.setattr(health, "probe_owner", held_probe)
    monkeypatch.setattr(health.subprocess, "run", lambda *args, **kwargs: pytest.fail("restart crossed a new hold"))
    assert health.main() == 0
    for result in read_result().values():
        assert result["restart_requested"] is False
        assert result["restart_deferred"] == "operator_hold"
        assert result["restart_times"] == []


def test_health_rechecks_hold_before_second_owner_restart(tmp_path, monkeypatch):
    read_result = _prepare_health_main(tmp_path, monkeypatch)
    monkeypatch.setattr(health, "probe_owner", lambda *_args: {"healthy": False, "reason": "canonical_writer_lock_missing"})
    calls = []
    def restart(command, **kwargs):
        calls.append(command)
        (tmp_path / "HOLD").touch()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(health.subprocess, "run", restart)
    assert health.main() == 0
    assert calls == [["systemctl", "--user", "restart", health.SERVICES["aggregate_control"]]]
    result = read_result()
    assert result["aggregate_control"]["restart_outcome"] == "completed"
    assert result["derived_coordination"]["restart_deferred"] == "operator_hold"
    assert result["derived_coordination"]["restart_times"] == []


def test_health_persists_and_budgets_restart_timeout_without_assuming_nonexecution(tmp_path, monkeypatch):
    read_result = _prepare_health_main(tmp_path, monkeypatch)
    monkeypatch.setattr(health, "probe_owner", lambda *_args: {"healthy": False, "reason": "canonical_writer_lock_missing"})
    calls = []
    def timeout(command, **kwargs):
        calls.append(command)
        raise health.subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(health.subprocess, "run", timeout)
    assert health.main() == 0
    assert len(calls) == 2
    for result in read_result().values():
        assert result["restart_outcome"] == "timeout_unknown"
        assert result["restart_returncode"] is None
        assert result["restart_times"] == [1000.0]
        assert health.recovery_decision(result, result["probe"], now=1030)["restart_requested"] is False


@pytest.mark.parametrize("stage", ["before", "after"])
@pytest.mark.parametrize("metadata", ["malformed", "nonmapping", "unreadable", "null_identity", "nested", "fifo"])
def test_unverified_owner_metadata_never_becomes_restartable_gateway_failure(tmp_path, monkeypatch, stage, metadata):
    birth = {"pid": 22, "start_time_ticks": 1, "boot_id": "boot:test"}
    identity = {"process_birth": birth, "process_birth_id": "birth:test", "database_uuid": "uuid:test",
                "generation": 1, "server_id": "server:test"}
    path = tmp_path / "quack-state-server.status.json"
    path.write_text(json.dumps({"lifecycle": "ready", "identity": identity}))
    (tmp_path / "fleet-observation-read.token").write_text("private-disposable-test-credential")
    monkeypatch.setattr(health, "process_identity", lambda _pid: birth)
    closed = []
    def corrupt():
        if metadata == "unreadable":
            path.unlink()
            path.mkdir()
        elif metadata == "fifo":
            path.unlink()
            os.mkfifo(path)
        elif metadata == "nested":
            path.write_text("[" * 6000 + "]" * 6000)
        elif metadata == "null_identity":
            path.write_text(json.dumps({"lifecycle": "ready", "identity": None}))
        else:
            path.write_text("{" if metadata == "malformed" else "[]")
    def generation():
        if stage == "after":
            corrupt()
        return SimpleNamespace(generation=1, database_uuid="uuid:test")
    client = SimpleNamespace(load_generation=generation, session=SimpleNamespace(server_id="server:test"),
                             close=lambda: closed.append(True))
    monkeypatch.setattr(health, "attach_typed_instance", lambda *args, **kwargs: client)
    if stage == "before":
        corrupt()
    started = time.monotonic()
    result = health.probe_owner({"instances": {"aggregate_control": {
        "state_dir": str(tmp_path), "database_path": str(tmp_path / "control.duckdb")}}}, "aggregate_control")
    assert time.monotonic() - started < 2
    assert result["healthy"] is False and result["restartable"] is False
    assert result["reason"] in {"native_owner_status_unverified", "native_owner_birth_unverified"}
    assert closed == ([] if stage == "before" else [True])
    assert health.recovery_decision({"consecutive_failures": 10}, result, now=1000)["restart_requested"] is False
