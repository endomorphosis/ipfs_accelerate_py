"""Deployment isolation, native launch policy, and real typed-owner attachment."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import quack_fleet_topology as fleet

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def topology(tmp_path):
    value = json.loads((ROOT / "config/agent_supervisor_quack_fleet_topology.json").read_text())
    value["sources"] = [{"id": "spar", "database_path": str(tmp_path / "spar/control.duckdb"),
                         "database_program": {"quack_endpoint": "quack:127.0.0.1:46731", "store_id": "control.duckdb",
                                              "store_generation": "1", "schema_revision": "1", "endpoint_secret_handle": "handle:spar"}}]
    return value


def compile(value, tmp_path):
    return fleet.compile_topology(value, state_root=tmp_path / "fleet", code_root=ROOT)


def test_native_owner_launches_are_separate_and_fail_closed(topology, tmp_path):
    result = compile(topology, tmp_path)
    assert result["defaults"] == fleet.DEFAULTS
    assert len({v["database_path"] for v in result["instances"].values()}) == 3
    assert len({v["database_program"]["quack_endpoint"] for v in result["instances"].values()}) == 3
    for role in fleet.ROLES:
        argv = result["instances"][role]["start_argv"]
        assert "--deny-legacy-board-unstall" in argv
        assert "--allow-experimental" not in argv
        assert argv[-1] == "start"
        assert str(ROOT / "scripts/ops/agent_supervisor/quack_state_server.py") in argv
    assert not (tmp_path / "fleet").exists()
    assert result["aggregation"]["authoritative"] is False
    assert result["derived_coordination"]["semantic_truth_authority"] == "ipfs_datasets_py"
    assert result["derived_coordination"]["artifact_kinds"] == list(fleet.ARTIFACT_KINDS)
    assert result["derived_coordination"]["artifact_schema"] == fleet.ARTIFACT_SCHEMA
    assert result["runtime_qualified"] is False


@pytest.mark.parametrize("field,value", [("authority_mode", "embedded"), ("task_source_kind", "markdown"), ("failover_policy", "local")])
def test_default_demotion_refused(topology, tmp_path, field, value):
    topology["defaults"][field] = value
    with pytest.raises(fleet.FleetTopologyError, match="operational defaults"):
        compile(topology, tmp_path)


@pytest.mark.parametrize("mutation,pattern", [
    (lambda t: t["owners"]["aggregate_control"]["database_program"].update(quack_endpoint="quack:127.0.0.1:46731"), "separate endpoints"),
    (lambda t: t["owners"]["aggregate_control"].update(database_path="../spar/control.duckdb"), "within state root"),
    (lambda t: t["owners"]["derived_coordination"].update(state_dir="aggregate_control/owner"), "separate"),
    (lambda t: t["owners"]["derived_coordination"].update(state_dir="aggregate_control/owner/nested"), "overlap"),
    (lambda t: t["owners"]["aggregate_control"]["database_program"].update(quack_endpoint="quack:0.0.0.0:47831"), "loopback"),
    (lambda t: t["owners"]["aggregate_control"].update(isolation_receipt_path="derived_coordination/owner/receipt.json"), "exact owner"),
])
def test_owner_collisions_and_boundary_escapes_refused(topology, tmp_path, mutation, pattern):
    mutation(topology)
    with pytest.raises(ValueError, match=pattern):
        compile(topology, tmp_path)


def test_source_population_has_no_fixed_lane_limit(topology, tmp_path):
    original = topology["sources"][0]
    topology["sources"] = []
    for index in range(100):
        source = copy.deepcopy(original)
        source["id"] = f"board-{index}"
        source["database_path"] = str(tmp_path / f"board-{index}/control.duckdb")
        source["database_program"]["quack_endpoint"] = f"quack:127.0.0.1:{20000+index}"
        topology["sources"].append(source)
    assert len(compile(topology, tmp_path)["aggregation"]["sources"]) == 100


def test_inventory_retains_native_bindings_and_refuses_stale_endpoint(topology, tmp_path):
    source = topology["sources"][0]
    config = tmp_path / "board.json"
    config.write_text(json.dumps({"database_program": source["database_program"]}))
    board = {"id": "spar", "config_path": str(config), "database_path": source["database_path"],
             "quack_endpoint": source["database_program"]["quack_endpoint"]}
    Path(board["database_path"]).parent.mkdir(parents=True)
    Path(board["database_path"]).touch()
    inventory = {"schema": "ipfs_accelerate_py/taskboard-fleet-inventory@1", "boards": [board]}
    result = fleet.bind_inventory(topology, inventory)
    assert result["sources"] == topology["sources"]
    board["quack_endpoint"] = "quack:127.0.0.1:7777"
    with pytest.raises(fleet.FleetTopologyError, match="differs from sealed board"):
        fleet.bind_inventory(topology, inventory)


def test_ops_forwards_legacy_mutation_fence(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("fleet_quack_owner_ops", ROOT / "scripts/ops/agent_supervisor/quack_state_server.py")
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    from ipfs_accelerate_py.agent_supervisor.runtime import quack_state_server
    seen = {}
    monkeypatch.setattr(quack_state_server, "build_server", lambda **kwargs: seen.update(kwargs))
    args = ops.build_parser().parse_args(["--repository-root", str(tmp_path), "--database", "control.duckdb", "--state-dir", "owner", "--deny-legacy-board-unstall", "start"])
    ops._build_server(args)
    assert seen["allow_legacy_board_unstall"] is False


def test_systemd_render_preserves_literal_argument_values():
    unit = fleet.systemd_unit(["/usr/bin/python3", "/tmp/space dir/%x$FOO.py"], role="aggregate_control")
    assert '"/tmp/space dir/%%x$$FOO.py"' in unit
    assert "Restart=on-failure" in unit
    with pytest.raises(fleet.FleetTopologyError):
        fleet.systemd_unit(["/tmp/evil\nExecStart=x"], role="aggregate_control")


@pytest.mark.parametrize("role", fleet.ROLES)
def test_managed_owner_timeout_cannot_force_kill_retained_writer(role):
    unit = fleet.systemd_unit(["/usr/bin/python3", "/tmp/native-owner.py"], role=role)
    assert "\nKillMode=mixed\n" in unit
    assert "\nTimeoutStopSec=120\n" in unit
    assert "\nSendSIGKILL=no\n" in unit


def test_attach_rejects_foreign_managed_socket_before_transport(topology, tmp_path):
    result = compile(topology, tmp_path)
    with pytest.raises(fleet.FleetTopologyError, match="socket differs"):
        fleet.attach_typed_instance(result, "derived_coordination", socket_path=tmp_path / "foreign.sock",
                                    token="unused", client_id="fleet:test", process_birth_id="birth:test")


def test_real_typed_owner_attach_and_wrong_grant_fail_closed(topology, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_control_plane_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
        QuackClientTransportError,
        QuackStateClient,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TypedStateOwnerError,
        TypedStateOwnerGateway,
    )

    db = tmp_path / "source.duckdb"
    install_control_plane_schema(db, application_version="0.0.45", tool_version="1.5.2", owner_id="fleet:test")
    seed = QuackStateClient(owner_id="fleet:seed", store_id="control.duckdb")
    seed.attach(db, seed_generation=True)
    seed.close()
    connection = open_duckdb_connection(db)
    row = connection.execute("SELECT generation, schema_revision, fence_epoch, revision, database_uuid, birth_id FROM store_generations ORDER BY generation DESC LIMIT 1").fetchone()
    identity = {"server_id": "server:fleet-test", "store_id": "control.duckdb", "generation": row[0], "schema_revision": row[1],
                "fence_epoch": row[2], "revision": row[3], "database_uuid": row[4], "process_birth_id": row[5] or "birth:owner"}
    socket = tmp_path / "owner.sock"
    gateway = TypedStateOwnerGateway(connection=connection, socket_path=socket, store_id="control.duckdb", identity=identity)
    gateway.start()
    token, _ = gateway.issue_grant(client_id="fleet:test", process_birth_id="birth:test",
                                    allowed_operations=("whoami_metadata", "load_store_generation"), peer_pid=os.getpid())
    topology["sources"][0]["database_path"] = str(db)
    deployment = compile(topology, tmp_path)
    try:
        client = fleet.attach_typed_instance(deployment, "spar", socket_path=socket, token=token,
                                             client_id="fleet:test", process_birth_id="birth:test")
        assert client.load_generation().generation == row[0]
        client.close()
        with pytest.raises((QuackClientTransportError, TypedStateOwnerError)):
            fleet.attach_typed_instance(deployment, "spar", socket_path=socket, token="unissued-grant",
                                        client_id="fleet:test", process_birth_id="birth:test")
    finally:
        gateway.stop()
        connection.close()


def test_missing_source_is_registered_unavailable_without_creating_database(topology, tmp_path):
    inventory = {"schema": "ipfs_accelerate_py/taskboard-fleet-inventory@1", "boards": [{
        "id": "pcpr", "config_path": str(tmp_path / "missing.json"), "database_path": str(tmp_path / "missing.duckdb"),
        "quack_endpoint": "quack:127.0.0.1:27777"}]}
    result = compile(fleet.bind_inventory(topology, inventory), tmp_path)
    assert result["unavailable_sources"] == ["pcpr"]
    assert result["instances"]["pcpr"]["available"] is False
    assert not Path(inventory["boards"][0]["database_path"]).exists()
    with pytest.raises(fleet.FleetTopologyError, match="source is unavailable"):
        fleet.attach_typed_instance(result, "pcpr", socket_path=tmp_path / "no.sock", token="unused",
                                    client_id="fleet:test", process_birth_id="birth:test")


def test_database_program_defaults_require_real_quack_binding(topology):
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        DatabaseProgramConfig,
    )
    source = topology["sources"][0]["database_program"]
    program = DatabaseProgramConfig.from_mapping(source)
    assert program.authority_mode == "quack"
    assert program.task_source_kind == "duckdb"
    with pytest.raises(ValueError, match="endpoint_secret_handle"):
        DatabaseProgramConfig()
    with pytest.raises(ValueError, match="unsupported authority_mode"):
        DatabaseProgramConfig.from_mapping({**source, "authority_mode": ""})
