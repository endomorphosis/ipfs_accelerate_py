#!/usr/bin/env python3
"""Thin operator facade for the existing SAWM supervisor authorities.

Import and ``--help`` are side-effect free.  All operational work is delegated
to the landed validators, DatabaseTaskSource, Quack owner, provider router and
configured-board scheduler; this module is not a second agent framework.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"


class OperatorError(RuntimeError):
    pass


def _load_script(relative: str, name: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OperatorError(f"cannot load sealed operator script: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OperatorError("scheduler config must be an object")
    return value


def _emit(value: Mapping[str, Any]) -> int:
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value.get("valid", True) is True else 2


def _validator(relative: str, function: str) -> dict[str, Any]:
    module = _load_script(relative, "_sawm_operator_" + function)
    return dict(getattr(module, function)(REPO_ROOT))


def _materializer():
    return _load_script("scripts/materialize_semantic_addressed_world_model_program.py", "_sawm_operator_materializer")


def _quack_args(config: Mapping[str, Any], command: str) -> list[str]:
    owner = config["quack_owner"]
    return [
        "--database", str(REPO_ROOT / owner["database_path"]),
        "--state-dir", str(REPO_ROOT / owner["state_dir"]),
        "--host", str(owner["host"]), "--port", str(owner["port"]),
        "--store-id", str(owner["store_id"]),
        "--repository-id", str(owner["repository_id"]),
        "--secret-handle", str(owner["secret_handle"]), "--json", command,
    ]


def _owner_connection(path: Path, owner: Mapping[str, Any]):
    """Open the sealed canonical writer without loading or serving Quack."""

    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnection,
        connect_duckdb_with_policy,
        exclusive_file_lock,
    )

    database = Path(path).resolve()
    lock = exclusive_file_lock(database.with_name(f".{database.name}.lock"))
    lock.__enter__()
    connection = None
    try:
        connection = connect_duckdb_with_policy(
            duckdb, database,
            configuration={"threads": 1, "memory_limit": "256MB"},
        )
        wrapped = DuckDBConnection.wrap(connection)
        wrapped.path = database
        wrapped._lock_context = lock
        return wrapped
    except BaseException:
        if connection is not None:
            connection.close()
        lock.__exit__(*sys.exc_info())
        raise


def _remote_owner_identity(
    uri: str,
    token: str,
    identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Resolve the live owner from canonical rows on the Quack replica."""

    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        _schema_fingerprint_digest,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_quack_transport_connection,
    )

    expected = dict(identity)
    required = {
        "server_id",
        "store_id",
        "database_uuid",
        "process_birth_id",
        "listen_uri",
        "extension_fingerprint",
        "schema_revision",
        "schema_fingerprint",
        "generation",
        "fence_epoch",
        "revision",
        "credential_generation",
        "secret_handle",
    }
    missing = sorted(key for key in required if expected.get(key) in (None, ""))
    if missing:
        raise OperatorError(
            "expected Quack owner identity is incomplete: " + ", ".join(missing)
        )

    connection = open_quack_transport_connection(uri, token=token)
    try:
        state_rows = connection.execute(
            "SELECT server_id, store_id, database_uuid, process_birth_id, "
            "listen_uri, extension_fingerprint, schema_revision, generation, "
            "status, revision FROM state_servers WHERE server_id = ? AND generation = ?",
            [expected["server_id"], int(expected["generation"])],
        ).fetchall()
        generation_rows = connection.execute(
            "SELECT generation, schema_revision, fence_epoch, revision, "
            "database_uuid, birth_id FROM store_generations WHERE generation = ?",
            [int(expected["generation"])],
        ).fetchall()
        credential_id = (
            f"cred:{expected['server_id']}:{int(expected['credential_generation'])}"
        )
        credential_rows = connection.execute(
            "SELECT credential_id, secret_handle, generation, purpose, revoked_at, "
            "revision FROM credentials WHERE credential_id = ?",
            [credential_id],
        ).fetchall()
        metadata_rows = connection.execute(
            "SELECT key, value FROM control_plane_metadata WHERE key IN "
            "('database_uuid', 'schema_version', 'schema_fingerprint') ORDER BY key"
        ).fetchall()
    finally:
        connection.close()

    if len(state_rows) != 1 or len(generation_rows) != 1 or len(credential_rows) != 1:
        raise OperatorError("live Quack owner identity rows are missing or ambiguous")
    state = state_rows[0]
    generation = generation_rows[0]
    credential = credential_rows[0]
    metadata = {str(row[0]): str(row[1]) for row in metadata_rows}
    normalized_metadata = {
        **metadata,
        "schema_fingerprint": _schema_fingerprint_digest(
            metadata.get("schema_fingerprint", "")
        ),
    }
    expected_state = (
        str(expected["server_id"]),
        str(expected["store_id"]),
        str(expected["database_uuid"]),
        str(expected["process_birth_id"]),
        str(expected["listen_uri"]),
        str(expected["extension_fingerprint"]),
        int(expected["schema_revision"]),
        int(expected["generation"]),
        "ready",
    )
    observed_state = (
        str(state[0]),
        str(state[1]),
        str(state[2]),
        str(state[3]),
        str(state[4]),
        str(state[5]),
        int(state[6]),
        int(state[7]),
        str(state[8]),
    )
    if observed_state != expected_state or int(state[9]) < int(expected["revision"]):
        raise OperatorError("live Quack state-server row differs from the owner identity")
    if (
        int(generation[0]),
        int(generation[1]),
        int(generation[2]),
        int(generation[3]),
        str(generation[4]),
        str(generation[5]),
    ) != (
        int(expected["generation"]),
        int(expected["schema_revision"]),
        int(expected["fence_epoch"]),
        int(expected["revision"]),
        str(expected["database_uuid"]),
        str(expected["process_birth_id"]),
    ):
        raise OperatorError("live Quack store-generation row differs from the owner identity")
    if (
        str(credential[0]),
        str(credential[1]),
        int(credential[2]),
        str(credential[3]),
        credential[4],
    ) != (
        credential_id,
        str(expected["secret_handle"]),
        int(expected["credential_generation"]),
        "quack-auth",
        None,
    ) or int(credential[5]) < int(expected["revision"]):
        raise OperatorError("live Quack credential row differs from the owner identity")
    if normalized_metadata != {
        "database_uuid": str(expected["database_uuid"]),
        "schema_fingerprint": str(expected["schema_fingerprint"]),
        "schema_version": str(int(expected["schema_revision"])),
    }:
        raise OperatorError("live Quack schema metadata differs from the owner identity")
    return MappingProxyType(
        {
            "server_id": str(state[0]),
            "store_id": str(state[1]),
            "database_uuid": str(state[2]),
            "process_birth_id": str(state[3]),
            "listen_uri": str(state[4]),
            "extension_fingerprint": str(state[5]),
            "schema_revision": int(state[6]),
            "generation": int(state[7]),
            "schema_fingerprint": normalized_metadata["schema_fingerprint"],
            "credential_generation": int(credential[2]),
            "live": True,
            "canonical_rows_verified": True,
        }
    )


class _SawmQuackTransport:
    """Serve only an atomically refreshed, read-only replica through Quack."""

    def __init__(self, owner: Mapping[str, Any]) -> None:
        self._owner = dict(owner)
        self._serve_uri = ""
        self._server_identity: dict[str, Any] = {}
        self._replica_connection = None
        self._replica_path: Path | None = None
        self._refresh_sequence = 0

    def _open_replica_connection(self, path: Path):
        import duckdb

        connection = duckdb.connect(
            str(path),
            read_only=True,
            config={
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "true",
                "enable_external_access": "true",
                "allow_unsigned_extensions": "false",
                "threads": "1",
                "memory_limit": "256MB",
            },
        )
        try:
            connection.execute("LOAD httpfs")
            connection.execute("LOAD quack")
            pin = self._owner.get("pinned_extension") or {}
            expected_path = Path(str(pin.get("path") or "")).resolve()
            observed = connection.execute(
                "SELECT install_path, extension_version FROM duckdb_extensions() "
                "WHERE extension_name = 'quack' AND installed AND loaded"
            ).fetchone()
            if observed is None or Path(str(observed[0])).resolve() != expected_path:
                raise OperatorError("loaded Quack extension path differs from the reviewed pin")
            if str(observed[1] or "") != str(pin.get("version") or ""):
                raise OperatorError("loaded Quack extension version differs from the reviewed pin")
            if hashlib.sha256(expected_path.read_bytes()).hexdigest() != str(pin.get("sha256") or ""):
                raise OperatorError("loaded Quack extension bytes differ from the reviewed pin")
            connection.execute("SET autoload_known_extensions = false")
            connection.execute("SET enable_external_access = false")
            connection.execute("SET lock_configuration = true")
            settings = connection.execute(
                "SELECT current_setting('autoinstall_known_extensions'), "
                "current_setting('autoload_known_extensions'), "
                "current_setting('enable_external_access'), "
                "current_setting('allow_unsigned_extensions'), "
                "current_setting('lock_configuration')"
            ).fetchone()
            if settings != (False, False, False, False, True):
                raise OperatorError("read-only Quack replica policy did not lock exactly")
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _copy_replica(source: Path, target: Path) -> Mapping[str, Any]:
        source = source.resolve()
        target = target.resolve()
        if source == target or source.parent != target.parent:
            raise OperatorError("Quack replica target is not a confined sibling")
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        target_fd = -1
        digest = hashlib.sha256()
        size = 0
        try:
            source_stat = os.fstat(source_fd)
            if not source_stat.st_size or source_stat.st_size > 8 * 1024**3:
                raise OperatorError("canonical store exceeds the replica copy bound")
            target_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    view = view[written:]
            if size != source_stat.st_size:
                raise OperatorError("canonical store changed during replica copy")
            os.fsync(target_fd)
            os.close(target_fd)
            target_fd = -1
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.chmod(target, 0o600)
            return {
                "authority": "non_authoritative_read_replica",
                "path": str(target),
                "source_database_path": str(source),
                "sha256": digest.hexdigest(),
                "size_bytes": size,
            }
        finally:
            os.close(source_fd)
            if target_fd >= 0:
                os.close(target_fd)
            temporary.unlink(missing_ok=True)

    def _stop_replica(self) -> None:
        connection = self._replica_connection
        if connection is None:
            return
        try:
            if self._serve_uri:
                connection.execute("SELECT * FROM quack_stop(?)", [self._serve_uri])
        finally:
            connection.close()
            self._replica_connection = None

    def refresh(self, writer) -> Mapping[str, Any]:
        database = Path(writer.path).resolve()
        replica = database.with_name(
            f"{database.stem}.read-replica{database.suffix}"
        )
        self._stop_replica()
        writer.execute("CHECKPOINT")
        observation = dict(self._copy_replica(database, replica))
        connection = self._open_replica_connection(replica)
        try:
            connection.execute(
                "SELECT * FROM quack_serve(?, token := ?, "
                "allow_other_hostname := false, disable_ssl := true)",
                [self._serve_uri, self._owner_token],
            )
        except BaseException:
            connection.close()
            raise
        self._replica_connection = connection
        self._replica_path = replica
        self._refresh_sequence += 1
        observation.update(
            {
                "refresh_sequence": self._refresh_sequence,
                "live": True,
                **self._server_identity,
            }
        )
        self._probe()
        return MappingProxyType(observation)

    def start(self, connection, *, host: str, port: int, token: str, identity):
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import listen_uri

        uri = listen_uri(host, port)
        self._serve_uri = uri
        self._owner_token = token
        self._server_identity = {
            "server_id": identity.server_id,
            "store_id": identity.store_id,
            "database_uuid": identity.database_uuid,
            "schema_revision": identity.schema_revision,
            "schema_fingerprint": identity.schema_fingerprint,
            "generation": identity.generation,
            "process_birth_id": identity.process_birth_id,
            "listen_uri": uri,
        }
        return self.refresh(connection)

    def _probe(self) -> None:
        if not self._serve_uri:
            raise OperatorError("Quack replica transport has not started")
        import duckdb

        last_error: Exception | None = None
        deadline = time.monotonic() + 3.0
        while True:
            client = duckdb.connect(
                ":memory:",
                config={
                    "autoinstall_known_extensions": "false",
                    "autoload_known_extensions": "false",
                    "allow_unsigned_extensions": "false",
                },
            )
            try:
                client.execute("LOAD quack")
                try:
                    rows = client.execute(
                        "SELECT * FROM quack_query(?, ?, token := ?, disable_ssl := true)",
                        [self._serve_uri, "SELECT count(*) FROM tasks", self._owner_token],
                    ).fetchall()
                    if len(rows) != 1:
                        raise OperatorError("Quack replica probe returned an invalid row count")
                    return
                except Exception as exc:
                    last_error = exc
            finally:
                client.close()
            if time.monotonic() >= deadline:
                raise OperatorError(
                    "authenticated Quack replica probe failed: "
                    f"{type(last_error).__name__ if last_error else 'unknown'}"
                ) from last_error
            time.sleep(0.05)

    def live_query(self, connection, *, identity, token: str):
        del connection
        if token != self._owner_token:
            raise OperatorError("Quack readiness token differs from the owner token")
        self._probe()
        return _remote_owner_identity(
            self._serve_uri,
            token,
            identity.to_dict(),
        )

    def stop(self, connection=None) -> None:
        del connection
        self._stop_replica()
        self._serve_uri = ""
        self._server_identity = {}


def _process_mutation_inbox(server: Any, *, max_requests: int = 32) -> None:
    """Service closed, atomic protocol-2 bundles on the exclusive writer."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
        mutation_binding_from_identity,
        service_mutation_inbox,
    )

    identity = server.identity
    connection = getattr(server, "_connection", None)
    vault = getattr(server, "_vault", None)
    transport = getattr(server, "transport", None)
    if identity is None or connection is None or vault is None or transport is None:
        raise OperatorError("Quack mutation inbox requires the live exclusive owner")
    token = vault.resolve(identity.secret_handle)
    inbox = Path(server.config.state_dir) / "mutations"
    service_mutation_inbox(
        connection,
        inbox=inbox,
        binding=mutation_binding_from_identity(identity),
        token=token,
        refresh_replica=lambda: transport.refresh(connection),
        max_requests=max_requests,
    )


def _serve_sawm_owner(server: Any) -> dict[str, Any]:
    stop_requested = {"value": False}

    def handle_signal(_signum: int, _frame: Any) -> None:
        stop_requested["value"] = True

    previous_int = signal.signal(signal.SIGINT, handle_signal)
    previous_term = signal.signal(signal.SIGTERM, handle_signal)
    try:
        control = server.stop_control_path()
        while server.lifecycle.value == "ready" and not stop_requested["value"]:
            if control.is_file():
                break
            _process_mutation_inbox(server, max_requests=32)
            time.sleep(0.05)
        return server.stop()
    except BaseException:
        server.stop()
        raise
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


def _start_quack(config: Mapping[str, Any]) -> int:
    owner = config["quack_owner"]
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_datasets_authoritative_operational_schema,
    )
    server = build_server(
        database_path=REPO_ROOT / owner["database_path"],
        state_dir=REPO_ROOT / owner["state_dir"], host=str(owner["host"]),
        port=int(owner["port"]), repository_id=str(owner["repository_id"]),
        store_id=str(owner["store_id"]), allow_experimental=True,
        secret_handle=str(owner["secret_handle"]),
        # Critical authority boundary: never install the generic full schema.
        migrate=install_datasets_authoritative_operational_schema,
        connection_factory=lambda path: _owner_connection(path, owner),
        transport=_SawmQuackTransport(owner),
    )
    identity = server.start()
    try:
        # State-server identity rows are published after transport.start();
        # refresh once more so readiness resolves those canonical rows through
        # the same read-only Quack replica used by schedulers.
        server.transport.refresh(server._connection)
        readiness = server.ready()
    except BaseException:
        server.stop()
        raise
    print(json.dumps({"identity": identity.to_dict(), "readiness": readiness}, indent=2, sort_keys=True))
    sys.stdout.flush()
    result = _serve_sawm_owner(server)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _live_preflight(
    config: Mapping[str, Any],
    *,
    probe_provider: bool = False,
    retire_provider_token_handoff: bool = False,
) -> dict[str, Any]:
    if probe_provider and not retire_provider_token_handoff:
        raise OperatorError(
            "provider probe requires prior retirement of the token handoff"
        )
    dependency = _validator("scripts/validate_semantic_addressed_world_model_dependencies.py", "validate_dependencies")
    board = _validator("scripts/validate_semantic_addressed_world_model_board.py", "validate_program")
    if dependency.get("valid") is not True or board.get("valid") is not True:
        raise OperatorError("sealed dependency or board validation failed")
    materializer = _materializer()
    population = materializer.build_population(REPO_ROOT)
    store = REPO_ROOT / config["database_program"]["store_id"]

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QUACK_ENDPOINT_ENV,
        QUACK_MUTATION_BINDING_ENV,
        QUACK_STORE_ID_ENV,
        QUACK_TOKEN_ENV,
        discover_live_quack_endpoint,
    )
    discovery = discover_live_quack_endpoint(store)
    expected_uri = str(config["database_program"]["quack_endpoint"])
    if not discovery.uri or discovery.uri != expected_uri or not discovery.token:
        raise OperatorError(f"exact live Quack owner unavailable: {discovery.reason}")
    try:
        owner_status = json.loads(Path(discovery.status_path).read_text(encoding="utf-8"))
        live_identity = owner_status["identity"]
        live_generation = int(live_identity["generation"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperatorError("live Quack generation is not bound by its status identity") from exc
    expected_generation = int(config["database_program"]["store_generation"])
    if live_generation != expected_generation:
        raise OperatorError(
            f"live Quack generation {live_generation} differs from the sealed "
            f"generation {expected_generation}"
        )
    remote_identity = _remote_owner_identity(
        discovery.uri,
        discovery.token,
        live_identity,
    )
    if remote_identity.get("canonical_rows_verified") is not True:
        raise OperatorError("live Quack owner canonical rows were not verified")
    # Resolve the opaque secret handle only into this coordinator environment.
    # No raw token is printed, put in argv, receipts or provider inputs.
    os.environ[QUACK_ENDPOINT_ENV] = discovery.uri
    os.environ[QUACK_STORE_ID_ENV] = str(config["database_program"]["store_id"])
    os.environ[QUACK_TOKEN_ENV] = discovery.token
    os.environ["IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"] = str(live_generation)
    mutation_binding = {
        "server_id": str(live_identity["server_id"]),
        "store_id": str(live_identity["store_id"]),
        "database_uuid": str(live_identity["database_uuid"]),
        "schema_revision": int(live_identity["schema_revision"]),
        "schema_fingerprint": str(live_identity["schema_fingerprint"]),
        "generation": live_generation,
        "process_birth_id": str(live_identity["process_birth_id"]),
        "listen_uri": str(live_identity["listen_uri"]),
        "extension_fingerprint": str(
            live_identity.get("extension_fingerprint") or "none"
        ),
    }
    os.environ[QUACK_MUTATION_BINDING_ENV] = json.dumps(
        mutation_binding, sort_keys=True, separators=(",", ":")
    )
    os.environ["IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"] = str(
        (REPO_ROOT / config["quack_owner"]["state_dir"] / "mutations").resolve()
    )
    os.environ["SAWM_QUACK_TOKEN"] = discovery.token
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    live = DatabaseTaskSource(discovery.uri, install_schema=False,
                              repository_tree_id=population["repository_tree_id"],
                              plan_root_cid=population["plan_root_cid"],
                              owner_id="sawm-r2-live-preflight")
    try:
        live_snapshot = live.snapshot().to_dict()
        if live_snapshot["task_count"] != 45 or live_snapshot["goal_count"] != 29 or live_snapshot["plan_root_cid"] != population["plan_root_cid"]:
            raise OperatorError("live Quack snapshot differs from the exact program root/counts")
        statuses: dict[str, str] = {}
        for expected in population["taskboard"]:
            observed = live.get_task(expected["task_cid"])
            if (
                observed is None
                or observed.task_alias != expected["task_id"]
                or observed.body.get("definition_cid") != expected["definition_cid"]
                or sorted(observed.dependencies) != sorted(expected["depends_on"])
                or [(item.get("effect") or {}).get("declared_path") for item in observed.outputs]
                   != [item["declared_path"] for item in expected["outputs"]]
                or [item.get("criterion") for item in observed.acceptance]
                   != [item["criterion"] for item in expected["acceptance_criteria"]]
                or [list(item.get("argv") or ()) for item in observed.validations]
                   != [[command] for command in expected["validation_commands"]]
            ):
                raise OperatorError(f"live Quack task definition conflict: {expected['task_id']}")
            statuses[observed.task_alias] = observed.status
        for expected in population["objectives"]:
            observed = live.get_goal(expected["goal_cid"])
            if observed is None or observed.get("goal_alias") != expected["goal_id"] or (observed.get("body") or {}).get("definition_cid") != expected["definition_cid"]:
                raise OperatorError(f"live Quack goal definition conflict: {expected['goal_id']}")
        if statuses.get("SAWM-000") not in {"completed", "complete", "done"}:
            raise OperatorError("live Quack authority lacks the SAWM-000 completion CAS")
    finally:
        live.close()
    store_report = {
        "valid": True, "task_count": live_snapshot["task_count"],
        "goal_count": live_snapshot["goal_count"],
        "projection_cid": live_snapshot["projection_cid"],
        "event_cursor": live_snapshot["event_cursor"],
        "store_generation": live_generation,
        "queried_through_live_quack_only": True,
        "direct_authoritative_file_opened": False,
        "statuses": statuses,
    }

    credential_report: dict[str, Any] = {
        "retired": False,
        "reason": "provider_launch_not_requested",
    }
    if retire_provider_token_handoff:
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
            retire_token_handoff,
        )

        status_path = Path(discovery.status_path).resolve()
        expected_state_dir = (
            REPO_ROOT / str(config["quack_owner"]["state_dir"])
        ).resolve()
        if status_path.parent != expected_state_dir:
            raise OperatorError(
                "live token handoff is outside the sealed owner state directory"
            )
        credential_report = retire_token_handoff(
            state_dir=expected_state_dir,
            secret_handle=str(live_identity["secret_handle"]),
            expected_token=discovery.token,
        )
        if credential_report.get("retired") is not True:
            raise OperatorError("live token handoff retirement failed closed")

    provider_report: dict[str, Any] = {
        "probed": False,
        "reason": "deferred_until_real_launch",
    }
    if probe_provider:
        provider = config["provider"]
        from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
            DatabaseProgramConfig,
            provider_subprocess_environment,
        )
        from ipfs_accelerate_py.llm_router import probe_grok_codex_agent_route_readiness

        database_program = DatabaseProgramConfig.from_mapping(
            config["database_program"]
        )
        provider_environment = provider_subprocess_environment(
            os.environ,
            program=database_program,
        )
        if discovery.token in provider_environment.values():
            raise OperatorError("provider probe environment retained owner credential")
        readiness = probe_grok_codex_agent_route_readiness(
            grok_model=str(provider["primary_model_id"]),
            codex_model=str(provider["fallback_model_id"]),
            codex_reasoning_effort=str(provider["fallback_reasoning_effort"]),
            environment=provider_environment,
        )
        failure = readiness.failure_kind.value if readiness.failure_kind is not None else ""
        provider_report = {**dataclasses.asdict(readiness), "failure_kind": failure, "probed": True}
        if not readiness.effective_provider:
            raise OperatorError(f"ordered provider route unavailable: {readiness.reason_code}")
        if readiness.effective_provider == "codex" and (
            provider["fallback_trigger"] != "primary_quota_exhausted"
            or failure != "grok_quota_exhausted"
        ):
            raise OperatorError("Codex fallback is not admitted by the reviewed quota-only trigger")
    return {
        "schema": "sawm/live-control-preflight@1", "valid": True,
        "dependency_valid": True, "board_valid": True,
        "store": store_report,
        "quack": {"uri": discovery.uri, "source": discovery.source,
                  "reason": discovery.reason, "token_present": True,
                  "live_query": True, "task_count": live_snapshot["task_count"],
                  "canonical_owner_rows_verified": True,
                  "provider_token_handoff": credential_report},
        "provider": provider_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "validate-dependencies", "validate-board", "materialize", "render", "check",
        "quack-start", "quack-status", "quack-ready", "quack-stop", "preflight", "dry-run",
    ):
        sub.add_parser(command)
    launch = sub.add_parser("launch")
    launch.add_argument("--foreground", action="store_true")
    launch.add_argument("--duration-seconds", type=float, default=float("inf"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
        config = _config(config_path)
        if args.command == "validate-dependencies":
            return _emit(_validator("scripts/validate_semantic_addressed_world_model_dependencies.py", "validate_dependencies"))
        if args.command == "validate-board":
            return _emit(_validator("scripts/validate_semantic_addressed_world_model_board.py", "validate_program"))
        if args.command in {"materialize", "render", "check"}:
            materializer = _materializer()
            if args.command == "render":
                return _emit({"valid": True, **materializer.build_population(REPO_ROOT)})
            if args.command == "check":
                population = materializer.build_population(REPO_ROOT)
                store = REPO_ROOT / config["database_program"]["store_id"]
                from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
                    discover_live_quack_endpoint,
                )
                discovery = discover_live_quack_endpoint(store)
                if discovery.uri:
                    return _emit({"action": "checked_live", **_live_preflight(config, probe_provider=False)})
                materializer._assert_committed_clean_source(REPO_ROOT, population)
                dependency = materializer._validator_report(
                    REPO_ROOT,
                    "scripts/validate_semantic_addressed_world_model_dependencies.py",
                )
                board = materializer._validator_report(
                    REPO_ROOT,
                    "scripts/validate_semantic_addressed_world_model_board.py",
                )
                validation_digest = materializer._identity(
                    {
                        "dependency": dependency,
                        "board": board,
                        "program_definition_cid": population["program_definition_cid"],
                    }
                )
                prior = materializer._verify_prior_store(
                    REPO_ROOT, config, population
                )
                verified = materializer._verify_store(
                    store,
                    population,
                    require_operator_complete=True,
                    require_migration=True,
                    migration_config=config,
                    expected_validation_digest=validation_digest,
                )
                receipt = materializer._ensure_migration_receipt(
                    REPO_ROOT,
                    store,
                    population,
                    verified,
                    validation_digest,
                )
                return _emit(
                    {
                        "valid": True,
                        "action": "checked",
                        "prior_authority": prior,
                        "receipt": receipt,
                        **verified,
                    }
                )
            return _emit(materializer.materialize(REPO_ROOT, config_path))
        if args.command == "quack-start":
            return _start_quack(config)
        if args.command in {"quack-status", "quack-ready", "quack-stop"}:
            ops = _load_script("scripts/ops/agent_supervisor/quack_state_server.py", "_sawm_landed_quack_ops")
            return int(ops.main(_quack_args(config, args.command.removeprefix("quack-"))))

        real_launch = args.command == "launch"
        live = _live_preflight(
            config,
            probe_provider=real_launch,
            retire_provider_token_handoff=real_launch,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
            main as scheduler_main,
        )
        scheduler_args = ["--repo-root", str(REPO_ROOT), "--config", str(config_path)]
        if args.command == "preflight":
            result = int(scheduler_main([*scheduler_args, "preflight"]))
        else:
            launch_args = [*scheduler_args, "launch", "--implement"]
            if args.command == "dry-run":
                launch_args.append("--dry-run")
            else:
                if args.foreground:
                    launch_args.append("--foreground")
                if math.isfinite(args.duration_seconds):
                    launch_args.extend(["--duration-seconds", str(args.duration_seconds)])
            result = int(scheduler_main(launch_args))
        if result:
            return result
        # Only secret-free preflight facts are emitted by this facade.
        print(json.dumps({"schema": "sawm/operator-delegation@1", "valid": True,
                          "command": args.command, "live_preflight": live}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        return _emit({"schema": "sawm/operator-error@1", "valid": False,
                      "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    raise SystemExit(main())
