"""Separate native Quack owners for fleet history and derived coordination.

This is deployment wiring over the existing state owner and typed client. It
creates no new store, accepts no SQL, and does not promote derived AST/hash
references or DuckLake observations to task completion authority.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..analysis.derived_artifacts import ARTIFACT_KINDS, ARTIFACT_SCHEMA
from .multi_supervisor_runner import DatabaseProgramConfig

SCHEMA = "ipfs_accelerate_py/agent-supervisor/quack-fleet-topology@1"
ROLES = ("aggregate_control", "derived_coordination")
DEFAULTS = {"authority_mode": "quack", "task_source_kind": "duckdb", "failover_policy": "fail_closed"}
_ENDPOINT = re.compile(r"quack:127\.0\.0\.1:([1-9][0-9]{0,4})\Z")
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")


class FleetTopologyError(ValueError):
    """The requested deployment cannot preserve independent native owners."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise FleetTopologyError(message)


def _endpoint(value: Any) -> str:
    value = str(value or "")
    match = _ENDPOINT.fullmatch(value)
    _require(match and int(match[1]) <= 65535, "explicit loopback Quack endpoint required")
    return value


def _path(value: Any, root: Path) -> Path:
    result = (root / str(value)).resolve()
    _require(result != root and result.is_relative_to(root), "owner path must stay within state root")
    return result


def compile_topology(
    topology: Mapping[str, Any], *, state_root: Path, code_root: Path,
) -> dict[str, Any]:
    """Validate all bindings and emit executable native-owner argv without I/O writes."""
    _require(topology.get("schema") == SCHEMA, "unsupported fleet topology schema")
    _require(state_root.is_absolute() and code_root.is_absolute(), "absolute roots required")
    state_root, code_root = state_root.resolve(), code_root.resolve()
    defaults = dict(DEFAULTS)
    defaults.update(topology.get("defaults", {}))
    _require(defaults == DEFAULTS, "fleet operational defaults must be DuckDB + Quack, fail closed")
    owners = topology.get("owners")
    _require(isinstance(owners, Mapping) and set(owners) == set(ROLES), "two separate fleet owner roles required")
    sources = topology.get("sources", [])
    _require(isinstance(sources, list) and bool(sources), "at least one supervisor source required")
    _require(len(sources) <= 4096, "fleet source population exceeds the admission bound")
    bindings, endpoints, databases, store_ids, owner_dirs, source_ids = {}, set(), set(), set(), set(), set()
    unavailable_sources = []

    def reserve(binding: Mapping[str, Any], *, database: Path) -> None:
        endpoint = _endpoint(binding["quack_endpoint"])
        store_id = binding["store_id"]
        _require(endpoint not in endpoints, "instances must use separate endpoints")
        _require(database not in databases, "instances must use separate databases")
        # Store IDs need only be unique together with endpoint/database identity;
        # existing independent boards legitimately use the same logical store ID.
        identity = (endpoint, store_id)
        _require(identity not in store_ids, "duplicate store identity")
        endpoints.add(endpoint)
        databases.add(database)
        store_ids.add(identity)

    for source in sources:
        _require(isinstance(source, Mapping), "source must be an object")
        source_id = str(source.get("id", ""))
        _require(_ID.fullmatch(source_id) and source_id not in source_ids and source_id not in ROLES, "unique source id required")
        source_ids.add(source_id)
        if source.get("available") is False:
            _require(not source.get("database_program"), "unavailable source cannot claim a qualified program")
            database = Path(str(source.get("database_path", ""))).expanduser()
            _require(database.is_absolute(), "unavailable source database path must be explicit")
            endpoint = _endpoint(source.get("quack_endpoint"))
            reserve({"quack_endpoint": endpoint, "store_id": "unqualified"}, database=database.resolve())
            bindings[source_id] = {"available": False, "reason": str(source.get("reason", "source_unavailable")),
                                   "database_path": str(database.resolve()), "quack_endpoint": endpoint,
                                   "managed_by_fleet": False, "completion_authority": False}
            unavailable_sources.append(source_id)
            continue
        program = dict(defaults)
        program.update(source.get("database_program", {}))
        checked = DatabaseProgramConfig.from_mapping(program)
        _require(all(getattr(checked, k) == v for k, v in DEFAULTS.items()), "source cannot demote fleet authority")
        _require(not checked.explicit_legacy, "legacy source is not a live Quack fleet member")
        database = Path(str(source.get("database_path", ""))).expanduser()
        _require(database.is_absolute(), "source database path must be explicit and absolute")
        database = database.resolve()
        reserve(checked.to_dict(), database=database)
        bindings[source_id] = {"database_program": checked.to_dict(), "database_path": str(database), "managed_by_fleet": False}

    for role in ROLES:
        owner = owners[role]
        _require(isinstance(owner, Mapping), "owner must be an object")
        program = dict(defaults)
        program.update(owner.get("database_program", {}))
        checked = DatabaseProgramConfig.from_mapping(program)
        _require(all(getattr(checked, k) == v for k, v in DEFAULTS.items()) and not checked.explicit_legacy, "owner cannot demote fleet authority")
        database = _path(owner.get("database_path", f"{role}/control.duckdb"), state_root)
        owner_dir = _path(owner.get("state_dir", f"{role}/owner"), state_root)
        _require(owner_dir not in owner_dirs, "owner state directories must be separate")
        _require(database != owner_dir and not database.is_relative_to(owner_dir), "database must be outside owner credential directory")
        owner_dirs.add(owner_dir)
        reserve(checked.to_dict(), database=database)
        endpoint = _ENDPOINT.fullmatch(checked.quack_endpoint)
        assert endpoint is not None
        argv = [sys.executable, str(code_root / "scripts/ops/agent_supervisor/quack_state_server.py"),
                "--repository-root", str(state_root), "--database", str(database),
                "--state-dir", str(owner_dir), "--host", "127.0.0.1", "--port", endpoint[1],
                "--store-id", checked.store_id, "--repository-id", f"fleet:{role}",
                "--secret-handle", checked.endpoint_secret_handle, "--deny-legacy-board-unstall"]
        isolation = owner.get("isolation_receipt_path")
        if isolation:
            receipt = _path(isolation, state_root)
            _require(receipt.is_relative_to(owner_dir), "isolation receipt must belong to its exact owner")
            argv.extend(["--isolation-receipt-json", str(receipt)])
        if role == "derived_coordination":
            argv.append("--derived-coordination")
        argv.extend(["--json", "start"])
        bindings[role] = {"database_program": checked.to_dict(), "database_path": str(database),
                          "state_dir": str(owner_dir), "start_argv": argv, "managed_by_fleet": True}

    # Nested paths also collide: an owner may otherwise rewrite a second
    # owner's credential/status directory through its own writable root.
    for left in owner_dirs:
        for right in owner_dirs:
            _require(left == right or not left.is_relative_to(right), "owner directories cannot overlap")
        _require(all(not db.is_relative_to(left) for db in databases), "owner credentials cannot contain any instance database")
    return {"schema": SCHEMA, "defaults": defaults, "instances": bindings,
            "aggregation": {"history_interface": "DuckLakeHistoryProjection@1", "receipt_store_interface": "DuckLakeProjectionStore@1",
                            "control_instance": "aggregate_control", "sources": sorted(source_ids), "authoritative": False},
            "derived_coordination": {"instance": "derived_coordination", "source_authority": "git",
                                     "semantic_truth_authority": "ipfs_datasets_py", "payloads": ["ast_cid", "content_hash", "state_root_cid"],
                                     "artifact_kinds": list(ARTIFACT_KINDS), "artifact_schema": ARTIFACT_SCHEMA,
                                     "completion_authority": False},
            "unavailable_sources": sorted(unavailable_sources), "runtime_qualified": False}


def bind_inventory(topology: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Bind sources to existing sealed board selections; never rewrite boards."""
    _require(inventory.get("schema") == "ipfs_accelerate_py/taskboard-fleet-inventory@1", "unsupported inventory schema")
    sources = []
    for board in inventory.get("boards", []):
        config_path, database_path = Path(board["config_path"]), Path(board["database_path"])
        if not config_path.is_file() or not database_path.is_file():
            sources.append({"id": board["id"], "available": False, "database_path": str(database_path),
                            "quack_endpoint": board["quack_endpoint"], "reason": "native_configuration_or_database_missing"})
            continue
        payload = json.loads(config_path.read_text())
        program = payload.get("database_program")
        _require(isinstance(program, Mapping), f"{board['id']}: explicit native database_program required")
        _require(program.get("quack_endpoint") == board.get("quack_endpoint"), f"{board['id']}: inventory endpoint differs from sealed board")
        sources.append({"id": board["id"], "database_path": board["database_path"], "database_program": dict(program)})
    return {**topology, "sources": sources}


def systemd_unit(owner_argv: Sequence[str], *, role: str) -> str:
    """Render a persistent native owner; systemd is the sole restart authority."""
    _require(role in ROLES, "unknown owner role")
    def quote(value: str) -> str:
        # systemd expands specifiers and environment variables even in quotes.
        _require(not any(char in value for char in ("\n", "\r", "\x00")), "invalid unit argument")
        return json.dumps(value.replace("%", "%%").replace("$", "$$"))
    command = " ".join(quote(str(value)) for value in owner_argv)
    return ("[Unit]\nDescription=Quack fleet " + role + " owner\n"
            "StartLimitIntervalSec=1800\nStartLimitBurst=4\n\n[Service]\nType=simple\n"
            "ExecStart=" + command + "\nRestart=on-failure\nRestartSec=60\n"
            "KillMode=mixed\nTimeoutStopSec=120\nUMask=0077\n\n[Install]\nWantedBy=default.target\n")


def attach_typed_instance(
    deployment: Mapping[str, Any], instance_id: str, *, socket_path: Path,
    token: str, client_id: str, process_birth_id: str, derived_repository_id: str = "", fleet_observation_read: bool = False, timeout_seconds: float = 30,
) -> Any:
    """Attach the existing typed Quack client using an owner-issued grant.

    Credential issuance/admission remains the native owner's job. Transport
    failure never opens the instance's database path or retries as embedded.
    """
    from ..task_sources.quack_state_client import QuackStateClient
    from ..task_sources.typed_state_owner import TypedStateOwnerConnection

    _require(deployment.get("schema") == SCHEMA, "compiled topology required")
    instance = deployment["instances"][instance_id]
    _require(instance.get("available") is not False, "native source is unavailable")
    program = DatabaseProgramConfig.from_mapping(instance["database_program"])
    _require(program.authority_mode == "quack" and program.task_source_kind == "duckdb", "typed fleet client requires Quack")
    _endpoint(program.quack_endpoint)
    _require(socket_path.is_absolute(), "typed socket path must be absolute")
    if instance.get("managed_by_fleet"):
        from ..task_sources.typed_state_owner import (
            TYPED_STATE_OWNER_SOCKET_FILENAME,
            compact_default_owner_socket_path,
        )
        expected = compact_default_owner_socket_path(
            Path(instance["state_dir"]) / TYPED_STATE_OWNER_SOCKET_FILENAME,
            identity=Path(instance["database_path"]),
        )
        _require(socket_path.resolve() == expected.resolve(), "typed socket differs from compiled owner")

    def connect(_endpoint: Any) -> Any:
        return TypedStateOwnerConnection(socket_path=socket_path, token=token, client_id=client_id,
                                         process_birth_id=process_birth_id, store_id=program.store_id,
                                         derived_repository_id=derived_repository_id, fleet_observation_read=fleet_observation_read,
                                         timeout_seconds=timeout_seconds)

    client = QuackStateClient(owner_id=client_id, store_id=program.store_id,
                              process_birth_id=process_birth_id, connection_factory=connect)
    try:
        client.attach(program.quack_endpoint)
    except BaseException:
        client.close()
        raise
    return client


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--start-owner", choices=ROLES)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--write-unit-dir", type=Path)
    args = parser.parse_args(argv)
    topology = json.loads(args.config.read_text())
    if args.inventory:
        topology = bind_inventory(topology, json.loads(args.inventory.read_text()))
    deployment = compile_topology(topology, state_root=args.state_root, code_root=args.code_root)
    if args.write_unit_dir:
        args.write_unit_dir.mkdir(parents=True, exist_ok=True)
        for role in ROLES:
            target = args.write_unit_dir / f"ipfs-quack-fleet-{role.replace(chr(95), chr(45))}.service"
            # Existing installed units may have operator changes. Refuse to
            # overwrite; rendering to a new review directory is repeatable.
            with target.open("x") as output:
                output.write(systemd_unit(deployment["instances"][role]["start_argv"], role=role))
    if args.start_owner:
        # Exec preserves systemd's main PID and all existing native owner
        # generation, process-birth, isolation and capability admission gates.
        import os
        owner_argv = deployment["instances"][args.start_owner]["start_argv"]
        os.execv(owner_argv[0], owner_argv)
    print(json.dumps(deployment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
