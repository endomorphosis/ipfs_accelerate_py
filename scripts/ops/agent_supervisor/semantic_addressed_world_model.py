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
import re
import shutil
import signal
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
_M9_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m9/control.duckdb"
)
_M9_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m9/control.coordination.duckdb"
)
_M9_GENERATION = 11


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


def _active_source_repair_materialization(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the newest sealed successor authority by key presence.

    M9 is the append-only live-recovery successor to M8.  A present malformed
    M9 authority must fail closed rather than silently selecting M8.  The same
    rule applies to M8 before the historical M7 fallback.
    """

    recovery_key = "live_recovery_successor_materialization"
    successor_key = "source_repair_successor_materialization"
    historical_key = "source_repair_materialization"
    if recovery_key in config:
        recovery = config.get(recovery_key)
        required = {
            "migration_revision",
            "target_store_id",
            "target_coordination_store_id",
            "target_generation",
            "target_plan_revision",
            "target_event_watermark",
            "target_projection_cid",
            "target_coordination_projection_digest",
            "provider_strategy",
            "task_rearm",
        }
        if (
            not isinstance(recovery, Mapping)
            or any(recovery.get(key) in (None, "") for key in required)
            or not isinstance(recovery.get("provider_strategy"), Mapping)
            or not isinstance(recovery.get("task_rearm"), Mapping)
        ):
            raise OperatorError("active M9 live-recovery successor authority is invalid")
        return recovery
    if successor_key in config:
        successor = config.get(successor_key)
        if not isinstance(successor, Mapping):
            raise OperatorError("active source-only successor authority is invalid")
        return successor
    historical = config.get(historical_key)
    if not isinstance(historical, Mapping):
        raise OperatorError("source-only successor authority is unavailable")
    return historical


def _successor_materialization_configured(config: Mapping[str, Any]) -> bool:
    """Return whether any append-only successor key is present.

    Key presence is intentional: malformed newer controls must reach the
    fail-closed selector instead of falling through to historical authority.
    """

    return any(
        key in config
        for key in (
            "live_recovery_successor_materialization",
            "source_repair_successor_materialization",
            "source_repair_materialization",
        )
    )


def _validate_m9_runtime_binding(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    """Bind M9 to its closed store pair, generation, and unchanged route."""

    if "live_recovery_successor_materialization" not in config:
        return
    program = config.get("database_program")
    owner = config.get("quack_owner")
    provider = config.get("provider")
    strategy = authority.get("provider_strategy")
    rearm = authority.get("task_rearm")
    if not all(
        isinstance(value, Mapping)
        for value in (program, owner, provider, strategy, rearm)
    ):
        raise OperatorError("M9 runtime authority binding is incomplete")
    prior_route = str(strategy.get("prior_route") or "")
    if (
        authority.get("migration_revision") != "SAWM-R2-M9"
        or authority.get("target_store_id") != _M9_STORE_ID
        or authority.get("target_coordination_store_id")
        != _M9_COORDINATION_STORE_ID
        or int(authority.get("target_generation") or 0) != _M9_GENERATION
        or program.get("store_id") != _M9_STORE_ID
        or int(program.get("store_generation") or 0) != _M9_GENERATION
        or owner.get("database_path") != _M9_STORE_ID
        or owner.get("store_id") != _M9_STORE_ID
        or not prior_route
        or strategy.get("target_route") != prior_route
        or strategy.get("route_changed") is not False
        or strategy.get("capability_probe_required") is not True
        or strategy.get("provider_result_is_completion_authority") is not False
        or not str(provider.get("primary_provider_id") or "")
        or not str(provider.get("primary_model_id") or "")
        or not str(provider.get("fallback_provider_id") or "")
        or not str(provider.get("fallback_model_id") or "")
        or provider.get("fallback_trigger") != "primary_quota_exhausted"
        or provider.get("provider_results_are_completion_authority") is not False
        or rearm.get("coordination_rearm_required") is not True
        or int(rearm.get("target_coordination_event_count") or 0) != 36
        or rearm.get("historical_settlement_preserved") is not True
    ):
        raise OperatorError("M9 runtime store, generation, or provider binding differs")


def _require_m9_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require the materializer's final marker for the verified M9 pair.

    Offline startup supplies the complete ``check_materialized`` report.  Live
    preflight rechecks its immutable final marker without opening the Quack-
    owned control database directly.
    """

    if "live_recovery_successor_materialization" not in config:
        return MappingProxyType({})
    _validate_m9_runtime_binding(config, authority)
    receipt_path = REPO_ROOT / Path(_M9_STORE_ID).parent / "migration-receipt.json"
    try:
        observed, _receipt_sha256 = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M9 final pair commit marker",
        )
    except Exception as exc:
        raise OperatorError("M9 materialized final pair marker is unavailable") from exc
    if not isinstance(observed, Mapping):
        raise OperatorError("M9 materialized final pair marker is invalid")
    coordination_path = (REPO_ROOT / _M9_COORDINATION_STORE_ID).resolve()
    try:
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination_path,
                root=REPO_ROOT,
                noun="materialized M9 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M9 materialized coordination store is unavailable") from exc
    unhashed = dict(observed)
    claimed_cid = str(unhashed.pop("receipt_cid", ""))
    rearm = authority["task_rearm"]
    if (
        claimed_cid != materializer._identity(unhashed)
        or observed.get("schema") != "sawm/non-authoritative-migration-receipt@7"
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != authority["migration_revision"]
        or observed.get("database_path") != _M9_STORE_ID
        or observed.get("coordination_path") != _M9_COORDINATION_STORE_ID
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or observed.get("migration_projection_cid")
        != authority["target_projection_cid"]
        or observed.get("coordination_projection_digest")
        != authority["target_coordination_projection_digest"]
        or int(observed.get("migration_event_watermark") or 0)
        != int(authority["target_event_watermark"])
        or int(observed.get("coordination_event_count") or 0)
        != int(rearm["target_coordination_event_count"])
        or observed.get("coordination_sidecar_copied_before_rearm") is not True
        or observed.get("failed_attempt_history_preserved") is not True
        or observed.get("failed_completion_barrier_rearmed") is not True
        or observed.get("execution_sidecar_copied") is not False
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M9 materialized final pair marker differs")
    if checked is not None:
        expected_control = str((REPO_ROOT / _M9_STORE_ID).resolve())
        expected_coordination = str(coordination_path)
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != expected_control
            or checked.get("coordination_path") != expected_coordination
            or checked.get("projection_cid") != authority["target_projection_cid"]
            or checked.get("coordination_projection_digest")
            != authority["target_coordination_projection_digest"]
            or int(checked.get("event_watermark") or 0)
            != int(authority["target_event_watermark"])
            or checked.get("receipt") != observed
        ):
            raise OperatorError("M9 materializer check differs from its final pair marker")
    return MappingProxyType(dict(observed))


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
        self._extension_projection_parent: Path | None = None
        self._sealed_extension_set: Any | None = None

    @staticmethod
    def _verify_extension_source(
        name: str,
        pin: Mapping[str, Any],
    ) -> tuple[Path, Path]:
        """Verify accepted extension and metadata bytes before native LOAD."""

        if name not in {"httpfs", "quack"}:
            raise OperatorError("extension name is outside the reviewed set")
        expected_fields = {
            "path",
            "info_path",
            "version",
            "sha256",
            "size",
            "info_sha256",
            "info_size",
            "network_install_allowed",
            "unsigned_extension_allowed",
            *(
                {"service_external_access_limitation"}
                if name == "quack"
                else set()
            ),
        }
        if (
            type(pin) is not dict
            or set(pin) != expected_fields
            or pin.get("network_install_allowed") is not False
            or pin.get("unsigned_extension_allowed") is not False
            or type(pin.get("version")) is not str
            or re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}", pin["version"])
            is None
            or type(pin.get("sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) is None
            or type(pin.get("info_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", pin["info_sha256"]) is None
            or (
                name == "quack"
                and pin.get("service_external_access_limitation")
                != "canonical_writer_sealed; "
                "pinned_extension_preloaded_only_in_locked_read_only_loopback_replica"
            )
        ):
            raise OperatorError(f"{name} extension pin is noncanonical")

        def evidence(
            path_value: object,
            expected_size: object,
            *,
            maximum_size: int,
        ) -> tuple[Path, str]:
            if type(path_value) is not str or type(expected_size) is not int:
                raise OperatorError(f"{name} extension path or size is invalid")
            path = Path(path_value)
            size = expected_size
            if not path.is_absolute() or size <= 0 or size > maximum_size:
                raise OperatorError(f"{name} extension size is outside policy")
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise OperatorError(f"{name} extension source is unavailable") from exc
            if resolved != path:
                raise OperatorError(f"{name} extension source path is noncanonical")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(path, flags)
            except OSError as exc:
                raise OperatorError(f"{name} extension source is unavailable") from exc
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_uid != os.geteuid()
                    or before.st_nlink != 1
                    or before.st_size != size
                ):
                    raise OperatorError(f"{name} extension source is not stable evidence")
                digest = hashlib.sha256()
                offset = 0
                while offset < size:
                    block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
                    if not block:
                        break
                    digest.update(block)
                    offset += len(block)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            try:
                current = os.stat(path, follow_symlinks=False)
            except OSError as exc:
                raise OperatorError(f"{name} extension source changed after read") from exc

            def identity(item: os.stat_result) -> tuple[int, ...]:
                return (
                    item.st_dev,
                    item.st_ino,
                    item.st_mode,
                    item.st_uid,
                    item.st_nlink,
                    item.st_size,
                    item.st_mtime_ns,
                    item.st_ctime_ns,
                )

            if (
                offset != size
                or identity(before) != identity(after)
                or identity(after) != identity(current)
            ):
                raise OperatorError(f"{name} extension source changed while read")
            return resolved, digest.hexdigest()

        payload_path, payload_sha = evidence(
            pin.get("path"), pin.get("size"), maximum_size=64 * 1024 * 1024
        )
        info_path, info_sha = evidence(
            pin.get("info_path"),
            pin.get("info_size"),
            maximum_size=64 * 1024,
        )
        if (
            payload_path.name != f"{name}.duckdb_extension"
            or info_path != payload_path.with_name(f"{payload_path.name}.info")
            or payload_sha != pin["sha256"]
            or info_sha != pin["info_sha256"]
        ):
            raise OperatorError(f"{name} extension bytes differ from the reviewed pin")
        return payload_path, info_path

    def _ensure_extension_projection(self) -> Any:
        """Publish reviewed bytes and return the shared sealed-set custodian."""

        if self._sealed_extension_set is not None:
            self._verify_sealed_extension_projection()
            return self._sealed_extension_set
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_extension_projection import (
            ConfiguredBoardExtensionProjectionError,
            build_configured_board_extension_set_pin,
            inspect_configured_board_extension_sources,
            project_configured_board_extension_set_home,
            seal_configured_board_extension_set_home,
        )

        source_pins = {
            "httpfs": self._owner.get("pinned_httpfs_extension") or {},
            "quack": self._owner.get("pinned_extension") or {},
        }
        sources = {
            name: self._verify_extension_source(name, pin)
            for name, pin in source_pins.items()
        }
        roots = {payload.parent for payload, _info in sources.values()}
        if len(roots) != 1:
            raise OperatorError("reviewed extensions do not share one exact root")
        source_root = next(iter(roots))
        engine_version = source_root.parent.name
        platform = source_root.name
        if (
            re.fullmatch(r"v[0-9][0-9A-Za-z.+_-]{0,62}", engine_version) is None
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", platform)
            is None
        ):
            raise OperatorError("reviewed extension engine/platform root is invalid")
        pins = {
            name: inspect_configured_board_extension_sources(
                payload,
                info,
                name=name,
                engine_version=engine_version,
                platform=platform,
            )
            for name, (payload, info) in sources.items()
        }
        for name, pin in pins.items():
            accepted = source_pins[name]
            if (
                pin.payload_sha256 != "sha256:" + str(accepted["sha256"])
                or pin.payload_size != accepted["size"]
                or pin.info_sha256 != "sha256:" + str(accepted["info_sha256"])
                or pin.info_size != accepted["info_size"]
            ):
                raise OperatorError(
                    f"{name} extension projection differs from its reviewed pin"
                )
        parent = Path(
            tempfile.mkdtemp(prefix="sawm-quack-extension-projection-", dir="/tmp")
        )
        os.chmod(parent, 0o700)
        sealed = None
        try:
            regular_home = project_configured_board_extension_set_home(
                pins,
                sources=sources,
                parent=parent,
            )
            set_pin = build_configured_board_extension_set_pin(
                pins,
                versions={
                    name: str(source_pins[name]["version"])
                    for name in sorted(source_pins)
                },
            )
            sealed = seal_configured_board_extension_set_home(
                set_pin,
                regular_home,
            )
            sealed.verify()
            self._extension_projection_parent = parent
            self._sealed_extension_set = sealed
            return sealed
        except BaseException as exc:
            if sealed is not None:
                sealed.close()
            self._remove_regular_extension_projection(parent)
            if isinstance(exc, ConfiguredBoardExtensionProjectionError):
                raise OperatorError(str(exc)) from exc
            raise

    def _verify_sealed_extension_projection(self) -> None:
        """Delegate custody and byte verification to the shared authority."""

        sealed = self._sealed_extension_set
        if sealed is None:
            raise OperatorError("sealed extension projection is incomplete")
        try:
            sealed.verify()
        except (OSError, ValueError) as exc:
            raise OperatorError(str(exc)) from exc

    def _load_reviewed_extensions(self, connection: Any) -> None:
        """Load exact sealed httpfs+Quack bytes and verify DuckDB's mapping."""

        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_extension_projection import (
            ConfiguredBoardExtensionProjectionError,
        )

        sealed = self._ensure_extension_projection()
        extension_pins = {
            "httpfs": self._owner.get("pinned_httpfs_extension") or {},
            "quack": self._owner.get("pinned_extension") or {},
        }
        try:
            with sealed.load_guard():
                connection.execute("LOAD httpfs")
                connection.execute("LOAD quack")
                observed_rows = connection.execute(
                    "SELECT extension_name, install_path, extension_version "
                    "FROM duckdb_extensions() "
                    "WHERE extension_name IN ('httpfs', 'quack') "
                    "AND installed AND loaded ORDER BY extension_name"
                ).fetchall()
        except ConfiguredBoardExtensionProjectionError as exc:
            raise OperatorError(str(exc)) from exc
        observed: dict[str, tuple[Path, str]] = {}
        if len(observed_rows) != len(extension_pins):
            raise OperatorError("loaded extension set is missing or ambiguous")
        for row in observed_rows:
            if not isinstance(row, (tuple, list)) or len(row) != 3:
                raise OperatorError("loaded extension observation is malformed")
            extension_name = str(row[0] or "")
            if extension_name not in extension_pins or extension_name in observed:
                raise OperatorError("loaded extension set is missing or ambiguous")
            install_path = Path(str(row[1] or ""))
            if not install_path.is_absolute():
                raise OperatorError(
                    f"loaded {extension_name} extension path is unavailable"
                )
            observed[extension_name] = (install_path, str(row[2] or ""))
        if set(observed) != set(extension_pins):
            raise OperatorError("loaded extension set is missing or ambiguous")
        for name, pin in extension_pins.items():
            expected_path = sealed.install_paths[name]
            if observed[name] != (expected_path, pin["version"]):
                raise OperatorError(
                    f"loaded {name} extension differs from the reviewed pin"
                )

    def _open_replica_connection(self, path: Path):
        import duckdb

        sealed = self._ensure_extension_projection()
        connection = duckdb.connect(
            str(path),
            read_only=True,
            config={
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
                "enable_external_access": "true",
                "allow_unsigned_extensions": "false",
                "extension_directory": str(sealed.extension_directory),
                "threads": "1",
                "memory_limit": "256MB",
            },
        )
        try:
            self._load_reviewed_extensions(connection)
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

    @staticmethod
    def _remove_regular_extension_projection(projection_parent: Path | None) -> None:
        if projection_parent is None:
            return
        try:
            if (
                not projection_parent.is_absolute()
                or projection_parent.parent != Path("/tmp")
                or not projection_parent.name.startswith(
                    "sawm-quack-extension-projection-"
                )
            ):
                return
            for current, directories, files in os.walk(projection_parent):
                current_path = Path(current)
                os.chmod(current_path, 0o700)
                for name in directories:
                    candidate = current_path / name
                    if not candidate.is_symlink():
                        os.chmod(candidate, 0o700)
                for name in files:
                    candidate = current_path / name
                    if not candidate.is_symlink():
                        os.chmod(candidate, 0o600)
            shutil.rmtree(projection_parent)
        except OSError:
            return

    def _remove_extension_projection(self) -> None:
        sealed = self._sealed_extension_set
        projection_parent = self._extension_projection_parent
        self._sealed_extension_set = None
        self._extension_projection_parent = None
        if sealed is not None:
            sealed.close()
        self._remove_regular_extension_projection(projection_parent)

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
        sealed = self._ensure_extension_projection()
        while True:
            client = duckdb.connect(
                ":memory:",
                config={
                    "autoinstall_known_extensions": "false",
                    "autoload_known_extensions": "false",
                    "enable_external_access": "true",
                    "allow_unsigned_extensions": "false",
                    "extension_directory": str(sealed.extension_directory),
                    "lock_configuration": "true",
                },
            )
            try:
                self._load_reviewed_extensions(client)
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
        try:
            self._stop_replica()
        finally:
            self._remove_extension_projection()
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


def _validate_offline_quack_start(
    config: Mapping[str, Any],
    config_path: Path = CONFIG_PATH,
) -> Mapping[str, Any]:
    """Revalidate committed controls and the exact store before native LOAD."""

    dependency = _validator(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "validate_dependencies",
    )
    board = _validator(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "validate_program",
    )
    if dependency.get("valid") is not True or board.get("valid") is not True:
        raise OperatorError("Quack start requires valid dependency and board seals")
    materializer = _materializer()
    population = materializer.build_population(REPO_ROOT)
    materializer._assert_committed_clean_source(REPO_ROOT, population)
    if _successor_materialization_configured(config):
        active_materialization = _active_source_repair_materialization(config)
        checked = materializer.check_materialized(REPO_ROOT, config_path)
        if checked.get("valid") is not True:
            raise OperatorError(
                "current append-only successor authority does not verify"
            )
        _require_m9_final_pair_marker(
            config,
            active_materialization,
            materializer,
            checked=checked,
        )
        return MappingProxyType(
            {
                "dependency_valid": True,
                "board_valid": True,
                "prior_authority": checked["prior_authority"],
                "store": checked,
            }
        )
    validation_digest = materializer._identity(
        {
            "dependency": dependency,
            "board": board,
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    prior = materializer._verify_prior_store(REPO_ROOT, config, population)
    verified = materializer._verify_store(
        REPO_ROOT / str(config["database_program"]["store_id"]),
        population,
        require_operator_complete=True,
        require_migration=True,
        migration_config=config,
        expected_validation_digest=validation_digest,
    )
    return MappingProxyType(
        {
            "dependency_valid": True,
            "board_valid": True,
            "prior_authority": prior,
            "store": verified,
        }
    )


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
    active_source_repair = _active_source_repair_materialization(config)
    final_pair_marker = _require_m9_final_pair_marker(
        config,
        active_source_repair,
        materializer,
    )
    expected_event_cursor = int(active_source_repair["target_event_watermark"])
    expected_projection_cid = str(
        active_source_repair["target_projection_cid"]
    )
    expected_plan_revision = int(active_source_repair["target_plan_revision"])
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
        if (
            live_snapshot["task_count"] != 45
            or live_snapshot["goal_count"] != 29
            or live_snapshot["dependency_count"] != 136
            or live_snapshot["plan_count"] != 1
            or live_snapshot["event_cursor"] != expected_event_cursor
            or live_snapshot["projection_cid"] != expected_projection_cid
            or live_snapshot["plan_root_cid"] != population["plan_root_cid"]
        ):
            raise OperatorError("live Quack snapshot differs from the exact program root/counts")
        try:
            statuses, _revisions, _receipts = (
                materializer._verify_m6_task_projection(live, population)
            )
            with live.intent._connection(write=False) as connection:
                semantic_authority_digest = (
                    materializer._semantic_authority_digest_on(connection)
                )
        except (
            materializer.MigrationRequired,
            materializer.MaterializationError,
        ) as exc:
            raise OperatorError(
                f"live Quack task authority conflict: {exc}"
            ) from exc
        expected_semantic_authority_digest = active_source_repair[
            "prior_semantic_authority_digest"
        ]
        if semantic_authority_digest != expected_semantic_authority_digest:
            raise OperatorError("live Quack semantic task authority differs")
        for expected in population["objectives"]:
            observed = live.get_goal(expected["goal_cid"])
            body = observed.get("body") if isinstance(observed, Mapping) else {}
            if (
                observed is None
                or observed.get("goal_cid") != expected["goal_cid"]
                or observed.get("goal_alias") != expected["goal_id"]
                or observed.get("objective_id")
                != str(expected.get("objective_id") or "")
                or observed.get("parent_goal_cid")
                != expected["parent_goal_cid"]
                or int(observed.get("ordinal") or 0) != int(expected["ordinal"])
                or observed.get("title") != expected["title"]
                or observed.get("status") != expected["status"]
                or int(observed.get("revision") or 0) != 1
                or body.get("definition_cid") != expected["definition_cid"]
                or body.get("definition") != expected["definition"]
                or materializer._identity(body.get("definition"))
                != expected["definition_cid"]
            ):
                raise OperatorError(f"live Quack goal definition conflict: {expected['goal_id']}")
        live_plan = live.plans.get(str(population["plan_root_cid"]))
        live_plan_body = (
            live_plan.get("body") if isinstance(live_plan, Mapping) else {}
        )
        if (
            live_plan is None
            or live_plan.get("plan_cid") != population["plan_root_cid"]
            or int(live_plan.get("revision") or 0) != expected_plan_revision
            or live_plan_body.get("current_source_binding_cid")
            != population["source_binding"]["source_binding_cid"]
            or live_plan_body.get("source_migration_revision")
            != active_source_repair["migration_revision"]
        ):
            raise OperatorError(
                "live Quack plan head is not bound to the current successor source"
            )
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
    if final_pair_marker:
        store_report.update(
            {
                "coordination_path": str(
                    (REPO_ROOT / _M9_COORDINATION_STORE_ID).resolve()
                ),
                "coordination_projection_digest": final_pair_marker[
                    "coordination_projection_digest"
                ],
                "coordination_event_count": final_pair_marker[
                    "coordination_event_count"
                ],
                "materialization_receipt_cid": final_pair_marker["receipt_cid"],
                "final_pair_commit_marker_verified": True,
            }
        )

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
                if _successor_materialization_configured(config):
                    active_materialization = _active_source_repair_materialization(
                        config
                    )
                    checked = materializer.check_materialized(
                        REPO_ROOT, config_path
                    )
                    _require_m9_final_pair_marker(
                        config,
                        active_materialization,
                        materializer,
                        checked=checked,
                    )
                    return _emit(checked)
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
            _validate_offline_quack_start(config, config_path)
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
