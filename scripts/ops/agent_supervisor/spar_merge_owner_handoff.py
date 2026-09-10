"""Native fresh-origin queue startup and exact-peer credential composition.

Legacy capture admission is deliberately separate: no absent lock, empty
projection, or caller JSON can select this fresh-origin initializer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from . import spar_merge_owner as role

FRESH_PROFILE = "native-fresh-origin@1"
LEGACY_PROFILE = "native-legacy-capture@1"
BUNDLE_PROFILE = "native-owner-merge-pair@1"
ORIGIN_SCHEMA = "spar/native-fresh-queue-origin@1"
ORIGIN_TABLE = "legacy_merge_native_origins"


def _scopes(*, board, paths, amendment):
    import re

    state_slug = (
        re.sub(r"[^a-z0-9._-]+", "-", board.task_prefix.strip().lower()).strip("-")
        or "configured-board"
    )
    raw = board.payload["runtime_paths"]
    state = board.path(raw["state"])
    return [
        {
            "board_namespace": board.board_namespace,
            "config_cid": amendment.launch_config_cid,
            "plan_cid": amendment.bootstrap_plan_root_cid,
            "lane_id": str(index),
            "attempt_root": str(
                state
                / f"lane-{index}"
                / f"{state_slug}_lane_{index}_database_portal_attempts"
            ),
        }
        for index in range(board.max_lanes)
    ]


def configured_queue_root(board):
    raw = board.payload.get("runtime_paths")
    if type(raw) is not dict or type(raw.get("merge_queue")) is not str:
        raise role.SparMergeOwnerError(
            "explicit configured native merge queue root required"
        )
    root = board.path(raw["merge_queue"]).absolute()
    runtime = board.path(raw["root"]).absolute()
    if (
        root == runtime
        or root.resolve(strict=False) != root
        or not root.is_relative_to(runtime)
    ):
        raise role.SparMergeOwnerError(
            "native queue root is outside its admitted runtime"
        )
    return root


def _load_origin(database, *, repository_id, target_branch, store_id, scopes):
    # A positive observed lock veto precedes every canonical DB open. Absence
    # does not certify legacy closure; only this role's canonical origin is read.
    info = database.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        raise role.SparMergeOwnerError(
            "native queue database is not an owned regular file"
        )
    role._refuse_observed_input_locks({"database": role._file_identity(info)})
    with role.open_duckdb_connection(database) as connection:
        names = {
            r[0]
            for r in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog=current_database() AND table_schema='main'"
            ).fetchall()
        }
        if ORIGIN_TABLE not in names:
            raise role.SparMergeOwnerError(
                "existing queue has no native fresh origin; legacy capture admission required"
            )
        columns = connection.execute(
            "SELECT column_name,data_type,is_nullable FROM information_schema.columns WHERE table_catalog=current_database() AND table_schema='main' AND table_name=? ORDER BY ordinal_position",
            [ORIGIN_TABLE],
        ).fetchall()
        if [tuple(row[i] for i in range(3)) for row in columns] != [
            ("origin_cid", "VARCHAR", "NO"),
            ("origin_json", "VARCHAR", "NO"),
        ]:
            raise role.SparMergeOwnerError("native fresh origin schema differs")
        rows = connection.execute(
            "SELECT origin_cid,origin_json FROM " + ORIGIN_TABLE + " LIMIT 2"
        ).fetchall()
        if len(rows) != 1:
            raise role.SparMergeOwnerError(
                "native fresh origin is missing or ambiguous"
            )
        record = role._decode(str(rows[0][1]).encode())
        role._closed(record, {"schema", "database_uuid", "database_path", "manifest"})
        if (
            rows[0][0] != role._cid(record)
            or record["schema"] != ORIGIN_SCHEMA
            or record["database_path"] != str(database)
        ):
            raise role.SparMergeOwnerError("native fresh origin identity differs")
        metadata = role._owner_metadata(connection)
        manifest = role.validate_manifest(record["manifest"])
        if metadata is None or metadata["database_uuid"] != record["database_uuid"]:
            raise role.SparMergeOwnerError(
                "native fresh UUID differs from preserved origin"
            )
        expected = {
            "repository_id": repository_id,
            "target_branch": target_branch,
            "store_id": store_id,
            "scope_bindings": scopes,
        }
        if any(manifest[key] != value for key, value in expected.items()):
            raise role.SparMergeOwnerError(
                "current native source namespace differs from fresh origin; explicit migration required"
            )
        if (
            manifest["receipt_imports"]
            or manifest["cursor_imports"]
            or manifest["wal"] is not None
        ):
            raise role.SparMergeOwnerError("fresh origin cannot contain legacy imports")
        baseline = role.inventory(connection)
    role.verify_installed_schema(database)
    return role.PreparedQueueStore(
        database, manifest, record["database_uuid"], baseline, (), ()
    )


def prepare_fresh_native_queue(
    *,
    queue_root,
    repository_id,
    target_branch,
    store_id,
    source_commit,
    source_tree,
    scopes,
):
    """Create once under an absent directory; restarts verify canonical origin."""
    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
        _MERGE_QUEUE_SCHEMA_SQL,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        initialize_duckdb_database,
    )

    root = Path(queue_root).absolute()
    descriptor = role._open_directory(root.parent)
    os.close(descriptor)
    database = root / "merge_queue.duckdb"
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        descriptor = role._open_directory(root)
        try:
            info = os.fstat(descriptor)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise role.SparMergeOwnerError(
                    "existing native queue root is not private and owned"
                )
        finally:
            os.close(descriptor)
        if not database.is_file():
            raise role.SparMergeOwnerError(
                "existing queue namespace cannot be initialized as fresh"
            )
        return _load_origin(
            database,
            repository_id=repository_id,
            target_branch=target_branch,
            store_id=store_id,
            scopes=scopes,
        )
    # These schema routines receive a newly created private directory and no
    # SQLite source. They neither construct MergeQueue nor scan/import JSON.
    initialize_duckdb_database(
        database,
        schema_sql=_MERGE_QUEUE_SCHEMA_SQL,
        table_names=("merge_requests",),
        legacy_sqlite_path=None,
    )
    role.install_control_plane_schema(database, owner_id="spar-native-fresh-owner")
    role.verify_installed_schema(database)
    with role.open_duckdb_connection(database) as connection:
        if connection.execute("SELECT COUNT(*) FROM merge_requests").fetchone()[0] != 0:
            raise role.SparMergeOwnerError(
                "new native queue unexpectedly contains work"
            )
        metadata = role._owner_metadata(connection)
        connection.execute("CHECKPOINT")
    # This is origin metadata for a known-created empty inode, not an offline
    # capture receipt and not a claim about any other legacy queue's callbacks.
    content = database.read_bytes()
    manifest = role.validate_manifest(
        {
            "schema": role.SCHEMA,
            "database": "merge_queue.duckdb",
            "wal": None,
            "repository_id": repository_id,
            "target_branch": target_branch,
            "store_id": store_id,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "scope_bindings": scopes,
            "queue_policy": dict(role.DEFAULT_QUEUE_POLICY),
            "receipt_imports": [],
            "cursor_imports": [],
            "files": [
                {
                    "path": "merge_queue.duckdb",
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            ],
        }
    )
    origin = {
        "schema": ORIGIN_SCHEMA,
        "database_uuid": metadata["database_uuid"],
        "database_path": str(database),
        "manifest": manifest,
    }
    with role.open_duckdb_connection(database) as connection:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            "CREATE TABLE "
            + ORIGIN_TABLE
            + " (origin_cid VARCHAR PRIMARY KEY, origin_json VARCHAR NOT NULL)"
        )
        connection.execute(
            "INSERT INTO " + ORIGIN_TABLE + " VALUES (?,?)",
            [role._cid(origin), role._json(origin).decode()],
        )
        connection.commit()
        baseline = role.inventory(connection)
    return role.PreparedQueueStore(
        database, manifest, metadata["database_uuid"], baseline, (), ()
    )


@dataclass
class NativeQueueOwner:
    server: Any
    prepared: role.PreparedQueueStore

    def close(self):
        self.server.stop()


def start_native_queue_for_launch(
    *, board, paths, amendment, profile, transport=None, capability_probe=None
):
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        checkout_repository_id,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.launch_source_amendment import (
        LaunchSourceAmendment,
    )

    if profile != FRESH_PROFILE:
        raise role.SparMergeOwnerError(
            "legacy queue capture and old-consumer closure are not independently admitted"
        )
    if (
        type(amendment) is not LaunchSourceAmendment
        or amendment.board_namespace != board.board_namespace
    ):
        raise role.SparMergeOwnerError(
            "native queue requires the admitted launch amendment"
        )
    root = configured_queue_root(board)
    scopes = _scopes(board=board, paths=paths, amendment=amendment)
    prepared = prepare_fresh_native_queue(
        queue_root=root,
        repository_id=checkout_repository_id(board.repo_root),
        target_branch=str(board.payload.get("merge_target_branch") or ""),
        store_id=str(root / "merge_queue.duckdb"),
        source_commit=amendment.launch_source_head,
        source_tree=amendment.launch_repository_tree_id,
        scopes=scopes,
    )
    server = role.start_queue_owner(
        prepared,
        state_dir=root / "native-owner",
        transport=transport,
        capability_probe=capability_probe,
    )
    return NativeQueueOwner(server, prepared)


class NativeMergeBundleIssuer:
    """Controller-local composition; no remotely callable grant or migration API."""

    def __init__(self, queue_owner, *, amendment, board):
        self.owner = queue_owner
        self.amendment = amendment
        self.board = board
        self.grants = {}
        self.pending = {}

    def validate_request(self, request, *, session):
        from ipfs_accelerate_py.agent_supervisor.task_sources.owner_merge_bootstrap import (
            REQUEST_FIELDS,
            REQUEST_SCHEMA,
        )

        role._closed(request, set(REQUEST_FIELDS))
        if (
            request["schema"] != REQUEST_SCHEMA
            or type(request["request_id"]) is not str
            or len(request["request_id"]) != 32
            or any(c not in "0123456789abcdef" for c in request["request_id"])
            or request["config_cid"] != self.amendment.launch_config_cid
            or request["plan_cid"] != self.amendment.bootstrap_plan_root_cid
            or session
            not in {
                f"{self.board.board_namespace}-{i}" for i in range(self.board.max_lanes)
            }
        ):
            raise role.SparMergeOwnerError(
                "native bundle request differs from admitted source/lane"
            )
        if self.owner.server.ready().get("ready") is not True:
            raise role.SparMergeOwnerError("native queue owner is not ready")

    def issue(self, request, *, session, task_response, task_owner_identity):
        from ipfs_accelerate_py.agent_supervisor.merge.owner_merge_queue import (
            SERVICE_OPERATIONS as QUEUE_OPS,
        )
        from ipfs_accelerate_py.agent_supervisor.merge.owner_recovery_runtime import (
            SERVICE_OPERATIONS as RECOVERY_OPS,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.owner_merge_bootstrap import (
            RESPONSE_SCHEMA,
        )

        self.validate_request(request, session=session)
        owner = self.owner.server
        manifest = self.owner.prepared.manifest
        index = int(session.rsplit("-", 1)[1])
        scope = manifest["scope_bindings"][index]
        scope_id = role.recovery_scope_cid(
            store_id=manifest["store_id"],
            repository_id=manifest["repository_id"],
            target_branch=manifest["target_branch"],
            scope_binding=scope,
        )
        credentials, grants = [], []
        identity = owner.identity.to_dict()
        try:
            for operations, recovery in ((QUEUE_OPS, False), (RECOVERY_OPS, True)):
                entity = {
                    "repository_id": manifest["repository_id"],
                    "target_branch": manifest["target_branch"],
                    "consumer_id": request["client_id"],
                }
                if recovery:
                    entity["recovery_scope_cid"] = scope_id
                token, grant = owner.issue_typed_client_grant_record(
                    client_id=request["client_id"],
                    process_birth_id=request["process_birth_id"],
                    peer_pid=request["pid"],
                    allowed_operations=tuple(operations),
                    entity_scopes=entity,
                    ttl_seconds=300,
                )
                grants.append(grant.grant_id)
                credentials.append(
                    {
                        "socket_path": str(owner.typed_command_socket_path()),
                        "store_id": identity["store_id"],
                        "server_id": identity["server_id"],
                        "client_id": request["client_id"],
                        "process_birth_id": request["process_birth_id"],
                        "token": token,
                    }
                )
            if (
                owner.identity.to_dict() != identity
                or owner.ready().get("ready") is not True
            ):
                raise role.SparMergeOwnerError(
                    "native queue owner changed during bundle issue"
                )
            result = {
                "schema": RESPONSE_SCHEMA,
                "ok": True,
                "request_id": request["request_id"],
                "task": task_response,
                "task_owner_identity": task_owner_identity,
                "queue": credentials[0],
                "recovery": credentials[1],
                "queue_owner_identity": identity,
                "repository_id": manifest["repository_id"],
                "target_branch": manifest["target_branch"],
                "scope_binding": scope,
                "recovery_scope_cid": scope_id,
            }
            self.grants[session] = tuple(grants)
            self.pending[session] = (dict(request), result)
            return result
        except BaseException:
            for grant in grants:
                owner.revoke_typed_client_grant(grant)
            raise

    def revoke(self, session):
        for grant in self.grants.pop(session, ()):
            self.owner.server.revoke_typed_client_grant(grant)
        self.pending.pop(session, None)

    def renew(self, session):
        if self.owner.server.ready().get("ready") is not True:
            raise role.SparMergeOwnerError(
                "native queue owner unavailable during renewal"
            )
        for grant in self.grants.get(session, ()):
            self.owner.server.renew_typed_client_grant(grant, ttl_seconds=300)

    def replay(self, request, *, session, task_owner_identity):
        self.validate_request(request, session=session)
        prior = self.pending.get(session)
        if prior is None:
            return None
        expected, response = prior
        if request != expected:
            raise role.SparMergeOwnerError(
                "current daemon bundle cannot be replaced by another request"
            )
        if (
            response["task_owner_identity"] != task_owner_identity
            or response["queue_owner_identity"] != self.owner.server.identity.to_dict()
        ):
            raise role.SparMergeOwnerError(
                "pending native bundle owner generation changed"
            )
        # Renewal validates exact current active grants; revoked grants cannot replay.
        self.renew(session)
        return response
