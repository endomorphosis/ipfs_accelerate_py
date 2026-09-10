"""Native observation replay and private admission for a real QuackLake connector."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
from ipfs_accelerate_py.agent_supervisor.federation import (
    fleet_observation as native,
)
from ipfs_accelerate_py.agent_supervisor.federation import (
    quacklake_catalog as catalog,
)


@pytest.fixture
def lake():
    connection = duckdb.connect(":memory:")
    connection.execute("ATTACH ':memory:' AS fleet_lake")
    try:
        yield connection
    finally:
        connection.close()


def observation(source_id="spar", *, available=True, stamp="2026-09-09T16:00:00+00:00"):
    return native.validate_observation({
        "schema": native.SCHEMA,
        "source_id": source_id,
        "observed_at": stamp,
        "availability": "available" if available else "unavailable",
        "source_identity": {
            "database_uuid": "uuid:native-source", "generation": 3,
            "process_birth_id": "birth:native-source", "listen_uri": "quack:127.0.0.1:7788",
        } if available else {},
        "native_receipt": {"authority": {"status_counts": {"completed": 2, "blocked": 1}}} if available else {},
        "reason": "" if available else "native_configuration_or_database_missing",
        "completion_authority": False,
    })


def fleet_view(*samples):
    return {
        "schema": catalog.VIEW_SCHEMA, "completion_authority": False,
        "control_store_id": "fleet-aggregate-control", "control_generation": 5,
        "ducklake_history_required_for_control": False,
        "sources": {item["source_id"]: {"current": item, "available": item["availability"] == "available"} for item in samples},
    }


def count(connection):
    return connection.execute("SELECT COUNT(*) FROM fleet_lake.fleet_source_observations").fetchone()[0]


def test_real_duckdb_replay_preserves_history_without_promoting_missing_source(lake):
    admitted, missing = observation(), observation("pcpr", available=False)
    view = fleet_view(admitted, missing)
    assert catalog.project_view(lake, view) == 2
    catalog.project_view(lake, copy.deepcopy(view))
    assert count(lake) == 2
    newer = observation(stamp="2026-09-09T16:01:00+00:00")
    newer["native_receipt"]["authority"]["status_counts"]["completed"] = 3
    catalog.project_view(lake, fleet_view(newer, missing))
    assert count(lake) == 3
    rows = lake.execute("SELECT observation_cid, source_id, payload_json, completion_authority FROM fleet_lake.fleet_source_observations").fetchall()
    assert {row[0] for row in rows} == {native.observation_cid(sample) for sample in (admitted, missing, newer)}
    assert all(row[3] is False for row in rows)
    missing_payload = next(json.loads(row[2]) for row in rows if row[1] == "pcpr")
    assert missing_payload["availability"] == "unavailable"
    assert missing_payload["native_receipt"] == {}
    assert lake.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'tasks'").fetchone()[0] == 0


def test_nonascii_observation_keeps_native_cid_and_canonical_bytes(lake):
    sample = observation("aseh", available=False)
    sample["reason"] = "native source unavailable: café 東京"
    catalog.project_view(lake, fleet_view(sample))
    cid, payload = lake.execute("SELECT observation_cid, payload_json FROM fleet_lake.fleet_source_observations").fetchone()
    assert cid == native.observation_cid(sample)
    assert payload == native.canonical(sample)


def test_valid_native_payload_above_64k_remains_exportable(lake):
    sample = observation()
    sample["native_receipt"]["bounded_diagnostic"] = "x" * 70000
    native.validate_observation(sample)
    catalog.project_view(lake, fleet_view(sample))
    assert count(lake) == 1


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(completion_authority=True),
    lambda value: value.update(schema="diagnostic:watchdog-json"),
    lambda value: value["sources"]["spar"]["current"].update(source_id="other"),
    lambda value: value["sources"]["spar"]["current"].update(native_receipt={}),
    lambda value: value["sources"]["spar"]["current"].update(source_identity={}),
    lambda value: value["sources"]["spar"]["current"].update(observed_at="2026-09-09T16:00:00"),
    lambda value: value["sources"]["spar"]["current"].update(unexpected="not a closed native observation"),
])
def test_invalid_or_non_native_views_cannot_write_rows(lake, mutation):
    sample = observation()
    catalog.project_view(lake, fleet_view(sample))
    invalid = fleet_view(observation(stamp="2026-09-09T16:01:00+00:00"))
    mutation(invalid)
    with pytest.raises((ValueError, TypeError)):
        catalog.project_view(lake, invalid)
    assert count(lake) == 1


def test_real_duckdb_constraint_failure_rolls_back_entire_projection(lake):
    lake.execute("CREATE TABLE fleet_lake.fleet_source_observations (observation_cid VARCHAR, source_id VARCHAR CHECK (source_id <> 'sawm'), observed_at VARCHAR, availability VARCHAR, payload_json VARCHAR, completion_authority BOOLEAN)")
    prior = observation("pcpr", available=False)
    catalog.project_view(lake, fleet_view(prior))
    with pytest.raises(duckdb.ConstraintException):
        catalog.project_view(lake, fleet_view(observation("spar"), observation("sawm")))
    assert count(lake) == 1
    assert lake.execute("SELECT source_id FROM fleet_lake.fleet_source_observations").fetchall() == [("pcpr",)]
    catalog.project_view(lake, fleet_view(observation("doep")))
    assert count(lake) == 2


@pytest.fixture
def configured(tmp_path):
    jwt = tmp_path / "catalog.jwt"
    jwt.write_text("header.payload.signature\n")
    jwt.chmod(0o600)
    storage = tmp_path / "storage.json"
    storage.write_text(json.dumps({
        "access_key_id": "r2-access-id", "secret_access_key": "r2-secret'value",
        "session_token": "r2-session", "endpoint": "account.r2.cloudflarestorage.com",
    }))
    storage.chmod(0o600)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema": catalog.CONFIG_SCHEMA, "endpoint": "quack:catalog.example.com:443",
        "data_path": "r2://fleet-bucket/catalogs/supervisors/",
        "jwt_file": str(jwt), "storage_credentials_file": str(storage),
    }))
    return config


class FakeConnection:
    def __init__(self, *, fail_on=""):
        self.commands = []
        self.closed = False
        self.fail_on = fail_on

    def execute(self, sql):
        self.commands.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("remote operation failed: " + sql)
        return self

    def close(self):
        self.closed = True


def test_attach_uses_memory_only_scoped_native_secrets_and_installed_extensions(configured):
    config = catalog.CatalogConfig.load(configured)
    fake = FakeConnection()
    databases = []
    def connect(database):
        databases.append(database)
        return fake
    assert config.open(connect=connect) is fake
    assert databases == [":memory:"]
    assert [sql for sql in fake.commands if sql.startswith("LOAD ")] == ["LOAD quack", "LOAD ducklake", "LOAD httpfs"]
    assert not any("INSTALL" in sql or "PERSISTENT" in sql for sql in fake.commands)
    quack = next(sql for sql in fake.commands if "SECRET fleet_quacklake_catalog" in sql)
    assert "TYPE quack" in quack and "TOKEN 'header.payload.signature'" in quack
    assert "SCOPE 'quack:catalog.example.com:443'" in quack and "ENDPOINT" not in quack
    storage = next(sql for sql in fake.commands if "SECRET fleet_quacklake_r2" in sql)
    assert "SECRET 'r2-secret''value'" in storage
    assert "SESSION_TOKEN 'r2-session'" in storage
    assert "SCOPE 'r2://fleet-bucket/catalogs/supervisors/'" in storage
    assert fake.commands[-1] == "ATTACH 'ducklake:quack:catalog.example.com:443' AS fleet_lake (DATA_PATH 'r2://fleet-bucket/catalogs/supervisors/')"
    assert fake.closed is False


@pytest.mark.parametrize("fail_on", ["LOAD ducklake", "SECRET fleet_quacklake_catalog", "SECRET fleet_quacklake_r2", "ATTACH"])
def test_failed_extension_secret_or_attach_closes_connection(configured, fail_on):
    config = catalog.CatalogConfig.load(configured)
    fake = FakeConnection(fail_on=fail_on)
    with pytest.raises(RuntimeError):
        config.open(connect=lambda _database: fake)
    assert fake.closed is True


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(endpoint="quack:127.0.0.1:27841"),
    lambda value: value.update(endpoint="https://catalog.example.com/quack"),
    lambda value: value.update(endpoint="quack:catalog.example.com:443'; DROP TABLE tasks; --"),
    lambda value: value.update(data_path="r2://fleet-bucket/catalogs/supervisors/../other/"),
    lambda value: value.update(data_path="r2://fleet-bucket/catalogs/supervisors/?token=secret"),
    lambda value: value.update(data_path="s3://fleet-bucket/catalogs/supervisors/"),
    lambda value: value.update(jwt_file="relative.jwt"),
    lambda value: value.update(token="inline secret"),
])
def test_closed_config_refuses_unsafe_or_ambiguous_admission(configured, mutation):
    value = json.loads(configured.read_text())
    mutation(value)
    configured.write_text(json.dumps(value))
    with pytest.raises((ValueError, TypeError)):
        catalog.CatalogConfig.load(configured)


@pytest.mark.parametrize("mode,content", [(0o644, "token"), (0o600, ""), (0o600, "x" * 16385)])
def test_private_file_rejects_public_empty_and_oversized_credentials(tmp_path, mode, content):
    path = tmp_path / "secret"
    path.write_text(content)
    path.chmod(mode)
    with pytest.raises(ValueError):
        catalog._private_file(path)


def test_private_file_rejects_symlink_and_other_owner(tmp_path, monkeypatch):
    path = tmp_path / "secret"
    path.write_text("secret")
    path.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        catalog._private_file(link)
    actual_uid = os.getuid()
    monkeypatch.setattr(catalog.os, "getuid", lambda: actual_uid + 1)
    with pytest.raises(ValueError):
        catalog._private_file(path)


def test_private_file_refuses_fifo_without_waiting_for_writer(tmp_path):
    fifo = tmp_path / "credential.pipe"
    os.mkfifo(fifo, 0o600)
    code = "from pathlib import Path; from ipfs_accelerate_py.agent_supervisor.federation.quacklake_catalog import _private_file; _private_file(Path(__import__('sys').argv[1]))"
    result = subprocess.run([sys.executable, "-c", code, str(fifo)], cwd=Path(catalog.__file__).resolve().parents[3], timeout=5, capture_output=True)
    assert result.returncode != 0
    assert b"credential file must be private" in result.stderr


@pytest.mark.parametrize("target,value", [("jwt", "not-a-jwt"), ("storage", {"access_key_id": "id", "secret_access_key": "secret", "endpoint": "account.r2.cloudflarestorage.com.evil.example"})])
def test_credentials_are_admitted_before_opening_any_database(configured, target, value):
    config = catalog.CatalogConfig.load(configured)
    path = config.jwt_file if target == "jwt" else config.storage_credentials_file
    path.write_text(value if isinstance(value, str) else json.dumps(value))
    called = []
    with pytest.raises((ValueError, TypeError)):
        config.open(connect=lambda database: called.append(database))
    assert called == []


@pytest.mark.parametrize("fail_on", ["SECRET fleet_quacklake_catalog", "SECRET fleet_quacklake_r2"])
def test_cli_redacts_sql_credentials_when_native_attach_fails(configured, tmp_path, monkeypatch, capsys, fail_on):
    monkeypatch.setattr(sys, "argv", ["quacklake", "--config", str(configured), "--deployment", str(tmp_path / "deployment.json"), "--inventory", str(tmp_path / "inventory.json")])
    monkeypatch.setattr(catalog, "read_native_view", lambda *_args: fleet_view(observation()))
    connection = FakeConnection(fail_on=fail_on)
    monkeypatch.setattr(duckdb, "connect", lambda database: connection)
    assert catalog.main() == 1
    assert connection.closed is True
    output = capsys.readouterr()
    assert json.loads(output.out) == {"status": "unavailable", "error_type": "RuntimeError", "completion_authority": False}
    assert output.err == ""
    assert not any(value in output.out for value in ("header.payload.signature", "r2-secret", "r2-session", "CREATE SECRET"))


def test_native_view_invokes_live_read_only_cli_with_bounded_output(tmp_path, monkeypatch):
    deployment = tmp_path / "deployment.json"
    inventory = tmp_path / "inventory.json"
    script = tmp_path / "release/scripts/ops/agent_supervisor/quack_state_server.py"
    deployment.write_text(json.dumps({"schema": "ipfs_accelerate_py/agent-supervisor/quack-fleet-topology@1", "instances": {"aggregate_control": {"start_argv": [sys.executable, str(script)]}}}))
    expected = fleet_view(observation())
    def run(argv, **kwargs):
        assert argv == [sys.executable, "-P", str(script.with_name("quack_fleet_aggregate.py")), "--query", "--deployment", str(deployment), "--inventory", str(inventory)]
        assert kwargs["timeout"] == 45 and kwargs["check"] is True
        assert kwargs["stderr"] == subprocess.DEVNULL
        kwargs["stdout"].write(json.dumps(expected).encode())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    assert catalog.read_native_view(deployment, inventory) == expected


def test_absent_current_projection_never_creates_a_native_source_observation(lake):
    value = fleet_view()
    value["sources"]["pcpr"] = {"available": False, "reason": "source_observation_absent"}
    assert catalog.project_view(lake, value) == 0
    assert count(lake) == 0


def test_native_view_output_limit_is_enforced_without_accepting_partial_json(tmp_path, monkeypatch):
    deployment = tmp_path / "deployment.json"
    deployment.write_text(json.dumps({"schema": "ipfs_accelerate_py/agent-supervisor/quack-fleet-topology@1", "instances": {"aggregate_control": {"start_argv": [sys.executable, "/release/quack_state_server.py"]}}}))
    def oversized(_argv, **kwargs):
        kwargs["stdout"].seek(16 * 1024 * 1024)
        kwargs["stdout"].write(b"x")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", oversized)
    with pytest.raises(ValueError, match="export limit"):
        catalog.read_native_view(deployment, tmp_path / "inventory.json")
