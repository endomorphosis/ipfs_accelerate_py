"""Preserved offline legacy queue qualification and native retained owner role.

An offline bundle is explicit input, not proof of live consumer closure. This
module never discovers live queues, issues worker grants, imports SQLite/JSON
implicitly, settles claims, or authorizes source adoption.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import datetime
import decimal
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any
import uuid

from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    _MERGE_QUEUE_SETTLEMENT_COLUMNS,
)
from ipfs_accelerate_py.agent_supervisor.merge.owner_recovery_runtime import (
    STAGES,
    _cid,
    recovery_scope_cid,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    MigrationRunReport,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    default_control_plane_schema,
    default_causal_event_federation_schema_extension,
    install_control_plane_schema,
    load_control_plane_catalog,
    verify_installed_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)

SCHEMA = "spar/legacy-queue-offline-bundle@1"
MAX_INPUT_BYTES = 8 * 1024**3
MAX_JSON_BYTES = 4 * 1024**2
MAX_FILES = 10_256
MAX_ROWS = 1_000_000
MAX_INVENTORY_DIGEST_BYTES = 64 * 1024**2
APPEND_ONLY_OWNER_TABLES = frozenset(
    {
        "control_plane_metadata",
        "schema_migrations",
        "schema_migration_attempts",
        "schema_contracts",
        "store_generations",
        "state_servers",
        "server_epochs",
        "client_sessions",
        "capability_snapshots",
        "credentials",
        "legacy_merge_recovery_migrations",
        "legacy_merge_recovery_scopes",
        "legacy_merge_recovery_cursors",
        "legacy_merge_recovery_cursor_history",
        "legacy_merge_recovery_operations",
        "legacy_merge_recovery_leases",
        "legacy_merge_recovery_receipt_heads",
        "legacy_merge_recovery_receipt_versions",
    }
)
DEFAULT_QUEUE_POLICY = {
    "max_age_seconds": 3600,
    "max_queue_size": 100,
    "max_processing": 100,
    "max_attempts": 3,
    "max_worktree_bytes": None,
}


class SparMergeOwnerError(RuntimeError):
    """A preserved-input or exact native role boundary was not admitted."""


def _json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")


def _closed(value: Any, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise SparMergeOwnerError("closed native queue manifest fields required")
    return value


def _text(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 4096
        or any(ord(c) < 32 for c in value)
    ):
        raise SparMergeOwnerError("native queue identity is malformed")
    return value


def _relative(value: Any) -> str:
    text = _text(value)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(p in {".", ".."} for p in path.parts)
    ):
        raise SparMergeOwnerError("offline input path is not canonical relative")
    return text


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SparMergeOwnerError("duplicate JSON key in preserved evidence")
        result[key] = value
    return result


def _decode(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_JSON_BYTES:
        raise SparMergeOwnerError("preserved JSON exceeds bound")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SparMergeOwnerError("preserved JSON is malformed") from exc
    if type(value) is not dict:
        raise SparMergeOwnerError("preserved JSON must be an object")
    return value


def _open_directory(root: Path) -> int:
    absolute = Path(root).absolute()
    if ".." in absolute.parts:
        raise SparMergeOwnerError("offline directory is not canonical")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in absolute.parts[1:]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise SparMergeOwnerError("offline directory namespace is unavailable") from exc


def _open_regular(root: Path, relative: str) -> int:
    """Walk pinned directory FDs: ancestor symlinks are never followed."""
    parts = PurePosixPath(_relative(relative)).parts
    descriptor = _open_directory(root)
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        result = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=descriptor,
        )
        info = os.fstat(result)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            os.close(result)
            raise SparMergeOwnerError("offline input is not an owned regular file")
        return result
    except OSError as exc:
        raise SparMergeOwnerError("offline input namespace is unavailable") from exc
    finally:
        os.close(descriptor)


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def file_inventory(root: Path) -> dict[str, tuple]:
    """Bounded descriptor-relative inventory of the explicitly supplied bundle."""
    result = {}
    visited = 0

    def walk(descriptor, prefix, depth):
        nonlocal visited
        if depth > 64:
            raise SparMergeOwnerError("offline directory depth exceeds bound")
        before = os.fstat(descriptor)
        with os.scandir(descriptor) as entries:
            for entry in entries:
                visited += 1
                if visited > MAX_FILES * 2:
                    raise SparMergeOwnerError("offline namespace exceeds bound")
                path = _relative(prefix + entry.name)
                info = os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=descriptor,
                    )
                    try:
                        if _file_identity(os.fstat(child)) != _file_identity(info):
                            raise SparMergeOwnerError("offline directory changed")
                        walk(child, path + "/", depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid():
                    result[path] = _file_identity(info)
                    if len(result) > MAX_FILES:
                        raise SparMergeOwnerError(
                            "offline file inventory exceeds bound"
                        )
                else:
                    raise SparMergeOwnerError(
                        "offline namespace has a nonregular entry"
                    )
        if _file_identity(os.fstat(descriptor)) != _file_identity(before):
            raise SparMergeOwnerError("offline directory changed during inventory")

    descriptor = _open_directory(root)
    try:
        walk(descriptor, "", 0)
    except OSError as exc:
        raise SparMergeOwnerError("offline inventory is unavailable") from exc
    finally:
        os.close(descriptor)
    return result


def _refuse_observed_input_locks(files):
    # This is a veto, not proof that an offline copy's original consumers closed.
    # In particular, do not open/close a file locked by this same controller.
    identities = {
        (os.major(value[0]), os.minor(value[0]), value[1]) for value in files.values()
    }
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if "->" in fields:
            continue
        if len(fields) != 8:
            raise SparMergeOwnerError("kernel lock census format is unavailable")
        major, minor, inode = fields[5].split(":")
        if (int(major, 16), int(minor, 16), int(inode)) in identities:
            raise SparMergeOwnerError(
                "supplied offline input has an observed kernel lock"
            )


def copy_entry(
    root: Path, entry: Mapping[str, Any], destination: Path | None, *, digest_only=False
) -> bytes:
    """Copy a declared offline inode, or read its bounded JSON, with exact hash."""
    descriptor = _open_regular(root, entry["path"])
    target = -1
    raw = bytearray()
    try:
        before = os.fstat(descriptor)
        if before.st_size != entry["size_bytes"]:
            raise SparMergeOwnerError("offline input size differs from manifest")
        if destination is None and not digest_only and before.st_size > MAX_JSON_BYTES:
            raise SparMergeOwnerError("preserved JSON exceeds bound")
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise SparMergeOwnerError("offline input changed during copy")
            digest.update(block)
            remaining -= len(block)
            if target >= 0:
                pending = memoryview(block)
                while pending:
                    written = os.write(target, pending)
                    if written <= 0:
                        raise SparMergeOwnerError("offline clone write failed")
                    pending = pending[written:]
            elif not digest_only:
                raw.extend(block)
        if os.read(descriptor, 1) or _file_identity(
            os.fstat(descriptor)
        ) != _file_identity(before):
            raise SparMergeOwnerError("offline input identity changed during copy")
        if digest.hexdigest() != entry["sha256"]:
            raise SparMergeOwnerError("offline input digest differs from manifest")
        if target >= 0:
            os.fsync(target)
        return bytes(raw)
    finally:
        os.close(descriptor)
        if target >= 0:
            os.close(target)


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _closed(
        value,
        {
            "schema",
            "database",
            "wal",
            "files",
            "repository_id",
            "target_branch",
            "store_id",
            "source_commit",
            "source_tree",
            "scope_bindings",
            "queue_policy",
            "receipt_imports",
            "cursor_imports",
        },
    )
    if manifest["schema"] != SCHEMA or manifest["database"] != "merge_queue.duckdb":
        raise SparMergeOwnerError("offline queue schema or database name differs")
    if manifest["wal"] not in (None, "merge_queue.duckdb.wal"):
        raise SparMergeOwnerError("offline WAL must pair with its database")
    for key in ("repository_id", "target_branch", "store_id"):
        _text(manifest[key])
    for key in ("source_commit", "source_tree"):
        if type(manifest[key]) is not str or not re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})", manifest[key]
        ):
            raise SparMergeOwnerError("offline source generation is malformed")
    files = manifest["files"]
    if type(files) is not list or not 1 <= len(files) <= MAX_FILES:
        raise SparMergeOwnerError("offline file inventory exceeds bound")
    names = set()
    total = 0
    for entry in files:
        _closed(entry, {"path", "size_bytes", "sha256"})
        name = _relative(entry["path"])
        if (
            name in names
            or type(entry["size_bytes"]) is not int
            or not 0 <= entry["size_bytes"] <= MAX_INPUT_BYTES
        ):
            raise SparMergeOwnerError("offline file inventory is malformed")
        if type(entry["sha256"]) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", entry["sha256"]
        ):
            raise SparMergeOwnerError("offline file digest is malformed")
        names.add(name)
        total += entry["size_bytes"]
    if (
        total > MAX_INPUT_BYTES
        or manifest["database"] not in names
        or (("merge_queue.duckdb.wal" in names) != (manifest["wal"] is not None))
    ):
        raise SparMergeOwnerError("offline database/WAL inventory is incomplete")
    policy = _closed(manifest["queue_policy"], set(DEFAULT_QUEUE_POLICY))
    for key, item in policy.items():
        if key == "max_worktree_bytes" and item is None:
            continue
        lower = 0 if key == "max_worktree_bytes" else 1
        upper = MAX_INPUT_BYTES if key == "max_worktree_bytes" else 1_000_000
        if type(item) is not int or not lower <= item <= upper:
            raise SparMergeOwnerError("offline queue policy is invalid")
    scopes = manifest["scope_bindings"]
    if type(scopes) is not list or not 1 <= len(scopes) <= 256:
        raise SparMergeOwnerError("offline scope inventory exceeds bound")
    seen = set()
    for scope in scopes:
        _closed(
            scope,
            {"board_namespace", "config_cid", "plan_cid", "lane_id", "attempt_root"},
        )
        for item in scope.values():
            _text(item)
        if not Path(scope["attempt_root"]).is_absolute():
            raise SparMergeOwnerError("offline scope attempt root must be absolute")
        cid = recovery_scope_cid(
            store_id=manifest["store_id"],
            repository_id=manifest["repository_id"],
            target_branch=manifest["target_branch"],
            scope_binding=scope,
        )
        if cid in seen:
            raise SparMergeOwnerError("offline scope repeats")
        seen.add(cid)
    for key in ("receipt_imports", "cursor_imports"):
        if type(manifest[key]) is not list or len(manifest[key]) > MAX_FILES:
            raise SparMergeOwnerError("offline import inventory exceeds bound")
        for item in manifest[key]:
            fields = (
                {"path", "receipt_key", "revision", "receipt_cid"}
                if key == "receipt_imports"
                else {"path", "scope_cid"}
            )
            _closed(item, fields)
            if _relative(item["path"]) not in names:
                raise SparMergeOwnerError("offline import file is undeclared")
    from ipfs_accelerate_py.agent_supervisor.merge.legacy_train_imports import (
        validate_train_import_coverage,
    )
    try:
        validate_train_import_coverage(
            file_names=names,
            receipt_imports=manifest["receipt_imports"],
            cursor_imports=manifest["cursor_imports"],
        )
    except ValueError as exc:
        raise SparMergeOwnerError(str(exc)) from exc
    return _decode(_json(manifest))


def _scalar(value):
    if value is None or type(value) in (str, bool, int):
        return [type(value).__name__, value]
    if type(value) is float and math.isfinite(value):
        return ["float", value.hex()]
    if type(value) is bytes:
        return ["bytes", len(value), hashlib.sha256(value).hexdigest()]
    if type(value) in (datetime.datetime, datetime.date, datetime.time):
        return [type(value).__name__, value.isoformat()]
    if type(value) in (decimal.Decimal, uuid.UUID):
        return [type(value).__name__, str(value)]
    raise SparMergeOwnerError("preserved table has unsupported scalar type")


def inventory(connection) -> dict[str, Any]:
    """Read logical row hashes on an offline or already-retained owner handle."""
    if connection.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_catalog=current_database() AND table_schema<>'main' AND table_type='BASE TABLE'"
    ).fetchone()[0]:
        raise SparMergeOwnerError("non-main preserved table namespace is unsupported")
    tables = connection.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_catalog=current_database() AND table_schema='main' AND table_type='BASE TABLE' ORDER BY table_name"
    ).fetchall()
    if len(tables) > 1024:
        raise SparMergeOwnerError("preserved schema exceeds bound")
    result = {}
    aggregate_rows = 0
    for table in tables:
        name = table[0]
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name):
            raise SparMergeOwnerError(
                "preserved table name is outside closed namespace"
            )
        columns = connection.execute(
            "SELECT column_name,data_type,is_nullable FROM information_schema.columns WHERE table_catalog=current_database() AND table_schema='main' AND table_name=? ORDER BY ordinal_position",
            [name],
        ).fetchall()
        count = connection.execute('SELECT COUNT(*) FROM "' + name + '"').fetchone()[0]
        aggregate_rows += count
        if (
            aggregate_rows > MAX_ROWS
            or aggregate_rows * 64 > MAX_INVENTORY_DIGEST_BYTES
        ):
            raise SparMergeOwnerError("preserved row inventory exceeds bound")
        hashes = []
        for offset in range(0, count, 1024):
            page = connection.execute(
                'SELECT * FROM "' + name + '" ORDER BY ALL LIMIT 1024 OFFSET ?',
                [offset],
            ).fetchall()
            for row in page:
                hashes.append(
                    hashlib.sha256(
                        _json([_scalar(row[i]) for i in range(len(row))])
                    ).hexdigest()
                )
        if len(hashes) != count:
            raise SparMergeOwnerError("preserved row inventory changed")
        result[name] = {
            "columns": [[row[i] for i in range(len(row))] for row in columns],
            "rows": sorted(hashes),
        }
    return result


def require_preserved(before, after):
    from collections import Counter

    schema = default_control_plane_schema()
    allowed_new_tables = (
        set(schema.all_domain_tables)
        | set(schema.bookkeeping_tables)
        | set(default_causal_event_federation_schema_extension().tables)
        | APPEND_ONLY_OWNER_TABLES
    )
    for name in set(after) - set(before):
        if name not in allowed_new_tables:
            raise SparMergeOwnerError(
                "migration introduced an undeclared foreign table: " + name
            )
        if name not in APPEND_ONLY_OWNER_TABLES and after[name]["rows"]:
            raise SparMergeOwnerError("migration introduced non-owner domain rows")

    for name, old in before.items():
        new = after.get(name)
        if (
            new is None
            or new["columns"] != old["columns"]
            or (Counter(old["rows"]) - Counter(new["rows"]))
        ):
            raise SparMergeOwnerError("migration changed preserved schema or rows")
        if name not in APPEND_ONLY_OWNER_TABLES and old != new:
            raise SparMergeOwnerError(
                "migration changed canonical legacy or foreign state: " + name
            )


def _owner_metadata(connection):
    names = {
        r[0]
        for r in connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_catalog=current_database() AND table_schema='main'"
        ).fetchall()
    }
    owner_names = {
        "control_plane_metadata",
        "store_generations",
        "state_servers",
        "client_sessions",
    }
    schema = default_control_plane_schema()
    known_owner_tables = (
        set(schema.all_domain_tables)
        | set(schema.bookkeeping_tables)
        | set(default_causal_event_federation_schema_extension().tables)
    )
    if not names.intersection(known_owner_tables):
        return None
    if not owner_names.issubset(names):
        raise SparMergeOwnerError("existing owner schema is partial")
    rows = connection.execute(
        "SELECT key,value FROM control_plane_metadata LIMIT 1025"
    ).fetchall()
    metadata = {row[0]: row[1] for row in rows}
    if len(rows) > 1024 or len(metadata) != len(rows):
        raise SparMergeOwnerError(
            "existing owner metadata is duplicate or exceeds bound"
        )
    try:
        if str(uuid.UUID(metadata["database_uuid"])) != metadata["database_uuid"]:
            raise ValueError("noncanonical UUID")
        revision = int(metadata["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SparMergeOwnerError("existing owner identity is malformed") from exc
    if revision < 1 or not metadata.get("schema_fingerprint"):
        raise SparMergeOwnerError("existing owner schema identity is incomplete")
    generations = connection.execute(
        "SELECT generation,fence_epoch,database_uuid FROM store_generations ORDER BY generation LIMIT 1000001"
    ).fetchall()
    if len(generations) > MAX_ROWS or any(
        type(g) is not int
        or g < 1
        or type(f) is not int
        or f < 1
        or ident != metadata["database_uuid"]
        for g, f, ident in (tuple(row[i] for i in range(3)) for row in generations)
    ):
        raise SparMergeOwnerError("existing owner generation history is malformed")
    return metadata


@dataclass(frozen=True)
class PreparedQueueStore:
    database_path: Path
    manifest: Mapping[str, Any]
    database_uuid: str
    preserved_inventory: Mapping[str, Any]
    receipt_imports: tuple[Mapping[str, Any], ...]
    cursor_imports: tuple[Mapping[str, Any], ...]

    @property
    def manifest_cid(self):
        return "sha256:" + hashlib.sha256(_json(dict(self.manifest))).hexdigest()


def prepare_offline_clone(
    *, offline_root: Path, destination: Path, manifest: Mapping[str, Any]
) -> PreparedQueueStore:
    """Transform only a newly created clone; never certify live custody."""
    manifest = validate_manifest(manifest)
    destination = Path(destination).absolute()
    descriptor = _open_directory(destination.parent)
    os.close(descriptor)
    original_files = file_inventory(offline_root)
    _refuse_observed_input_locks(original_files)
    if set(original_files) != {entry["path"] for entry in manifest["files"]}:
        raise SparMergeOwnerError(
            "offline file inventory is incomplete or has extra files"
        )
    destination.mkdir(parents=False, exist_ok=False, mode=0o700)
    entries = {entry["path"]: entry for entry in manifest["files"]}
    for entry in entries.values():
        _refuse_observed_input_locks(original_files)
        copy_entry(offline_root, entry, destination / entry["path"])
    receipts = []
    receipt_heads = {}
    for spec in manifest["receipt_imports"]:
        receipts.append(
            {k: spec[k] for k in ("receipt_key", "revision", "receipt_cid")}
            | {
                "receipt": _decode(
                    copy_entry(destination, entries[spec["path"]], None)
                ),
            }
        )
        receipt = receipts[-1]
        key = _text(receipt["receipt_key"])
        if (
            type(receipt["revision"]) is not int
            or receipt["revision"] != receipt_heads.get(key, 0) + 1
            or receipt["receipt_cid"] != _cid(receipt["receipt"])
        ):
            raise SparMergeOwnerError("preserved receipt history or content differs")
        receipt_heads[key] = receipt["revision"]
    scopes = {
        recovery_scope_cid(
            store_id=manifest["store_id"],
            repository_id=manifest["repository_id"],
            target_branch=manifest["target_branch"],
            scope_binding=s,
        ): s
        for s in manifest["scope_bindings"]
    }
    cursors = []
    seen_cursors = set()
    for spec in manifest["cursor_imports"]:
        scope = scopes.get(spec["scope_cid"])
        body = _decode(copy_entry(destination, entries[spec["path"]], None))
        _closed(
            body,
            {
                "schema",
                "target_repository_id",
                "target_branch",
                "attempt_root",
                "cursors",
                "state_id",
            },
        )
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
            _POST_MERGE_RECOVERY_CURSOR_SCHEMA,
            _canonical_json,
        )
        from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
            content_identity,
        )

        expected_path = (
            None
            if scope is None
            else (
                "train/post-merge-recovery-cursors/"
                + hashlib.sha256(
                    _canonical_json(
                        {
                            "target_repository_id": manifest["repository_id"],
                            "target_branch": manifest["target_branch"],
                            "attempt_root": scope["attempt_root"],
                        }
                    )
                ).hexdigest()
                + ".json"
            )
        )
        if (
            scope is None
            or spec["path"] != expected_path
            or body["schema"] != _POST_MERGE_RECOVERY_CURSOR_SCHEMA
            or body["target_repository_id"] != manifest["repository_id"]
            or body["target_branch"] != manifest["target_branch"]
            or body["attempt_root"] != scope["attempt_root"]
            or type(body["cursors"]) is not dict
            or set(body["cursors"]) != set(STAGES)
            or any(
                type(value) is not str or len(value) > 4096
                for value in body["cursors"].values()
            )
            or spec["scope_cid"] in seen_cursors
            or body["state_id"]
            != content_identity({k: v for k, v in body.items() if k != "state_id"})
        ):
            raise SparMergeOwnerError("preserved cursor binding is invalid")
        seen_cursors.add(spec["scope_cid"])
        cursors.append(
            {
                "scope_cid": spec["scope_cid"],
                "cursors": body["cursors"],
                "state_cid": _cid(body["cursors"]),
            }
        )
    # Validate the exact backend migration envelope before any owner is born.
    _cid(
        {
            "repository_id": manifest["repository_id"],
            "target_branch": manifest["target_branch"],
            "scope_bindings": manifest["scope_bindings"],
            "receipt_imports": receipts,
            "cursor_imports": cursors,
        }
    )
    for entry in entries.values():
        _refuse_observed_input_locks(original_files)
        copy_entry(offline_root, entry, None, digest_only=True)
    if file_inventory(offline_root) != original_files:
        raise SparMergeOwnerError("offline input changed during clone preparation")
    database = destination / manifest["database"]
    with open_duckdb_connection(database) as connection:
        before = inventory(connection)
        for name, columns in _MERGE_QUEUE_SETTLEMENT_COLUMNS.items():
            if name not in before or before[name]["columns"] != [
                list(row) for row in columns
            ]:
                raise SparMergeOwnerError("exact existing legacy schema is required")
        old_meta = _owner_metadata(connection)
        connection.execute("CHECKPOINT")
    if old_meta is None:
        install_control_plane_schema(database, owner_id="spar-offline-queue-migration")
    verify_installed_schema(database)
    with open_duckdb_connection(database) as connection:
        after = inventory(connection)
        require_preserved(before, after)
        metadata = _owner_metadata(connection)
        if metadata is None or (old_meta is not None and metadata != old_meta):
            raise SparMergeOwnerError("existing owner metadata changed")
    return PreparedQueueStore(
        database,
        manifest,
        metadata["database_uuid"],
        before,
        tuple(receipts),
        tuple(cursors),
    )


def start_queue_owner(
    prepared: PreparedQueueStore,
    *,
    state_dir: Path,
    transport=None,
    capability_probe=None,
):
    """Start the distinct retained role; still no worker or execution admission."""
    if type(prepared) is not PreparedQueueStore:
        raise SparMergeOwnerError("prepared queue role input is required")
    manifest = validate_manifest(dict(prepared.manifest))
    startup_inventory = None

    def verify(path):
        nonlocal startup_inventory
        checked = verify_installed_schema(path)
        with open_duckdb_connection(path) as connection:
            metadata = _owner_metadata(connection)
            if metadata is None or metadata["database_uuid"] != prepared.database_uuid:
                raise SparMergeOwnerError("queue owner UUID changed")
            startup_inventory = inventory(connection)
        return MigrationRunReport(
            from_version=int(metadata["schema_version"]),
            to_version=int(metadata["schema_version"]),
            receipts=(),
            schema_fingerprint=checked["schema_fingerprint"],
            catalog_fingerprint=load_control_plane_catalog().fingerprint(),
            changed=False,
        )

    server = build_server(
        database_path=prepared.database_path,
        state_dir=state_dir,
        repository_id=manifest["repository_id"],
        store_id=manifest["store_id"],
        port=0,
        allow_legacy_board_unstall=False,
        migrate=verify,
        transport=transport,
        capability_probe=capability_probe,
    )
    try:
        server.start()
        policy = dict(manifest["queue_policy"])
        scope = {key: manifest[key] for key in ("repository_id", "target_branch")}
        server.bind_legacy_merge_queue_service(**scope, **policy)
        server.provision_legacy_merge_recovery_schema(
            **scope,
            migration_id="spar-offline:" + prepared.manifest_cid,
            scope_bindings=manifest["scope_bindings"],
            receipt_imports=prepared.receipt_imports,
            cursor_imports=prepared.cursor_imports,
        )
        server.bind_legacy_merge_recovery_service(**scope)
        with server._owner_transaction_lock:
            require_preserved(startup_inventory, inventory(server._connection))
        server.ready()
        return server
    except BaseException:
        server.stop()
        raise


def _writer_lock_held(database: Path) -> bool:
    """Observe kernel custody without opening/closing the canonical database."""
    info = database.lstat()
    if not stat.S_ISREG(info.st_mode):
        return False
    identity = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if (
            len(fields) == 8
            and fields[1:5] == ["POSIX", "ADVISORY", "WRITE", str(os.getpid())]
            and fields[6:] == ["0", "EOF"]
        ):
            major, minor, inode = fields[5].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) == identity:
                return True
    return False


def _require_independent_writer_excluded(database: Path) -> None:
    import duckdb

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
sys.path.insert(0, sys.argv[2])
import duckdb
try:
    connection = duckdb.connect(sys.argv[1], config={'threads': 1})
except duckdb.IOException as error:
    sys.exit(69 if 'lock' in str(error).lower() else 2)
else:
    connection.close()
    sys.exit(0)
""",
            str(database),
            str(Path(duckdb.__file__).resolve().parent.parent),
        ],
        env={"PATH": os.defpath, "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=15,
    )
    if result.returncode != 69:
        raise SparMergeOwnerError("disposable owner did not exclude another writer")


def _observe_role_reads(server, prepared):
    """Issue local, read-only exact-peer grants solely for disposable rehearsal."""
    from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
        process_birth_id,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.owner_recovery_runtime import (
        OwnerRecoveryRuntimeClient,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TypedStateOwnerConnection,
    )

    manifest = prepared.manifest
    count = 0
    for index, scope in enumerate(manifest["scope_bindings"]):
        consumer = "spar-offline-observer:" + str(index)
        birth = process_birth_id(current_process_birth())
        scope_id = recovery_scope_cid(
            store_id=manifest["store_id"],
            repository_id=manifest["repository_id"],
            target_branch=manifest["target_branch"],
            scope_binding=scope,
        )
        token, grant = server.issue_typed_client_grant_record(
            client_id=consumer,
            process_birth_id=birth,
            peer_pid=os.getpid(),
            ttl_seconds=60,
            allowed_operations=(
                "legacy.merge_recovery.describe_scope",
                "legacy.merge_recovery.load_cursors",
                "legacy.merge_recovery.get_receipt",
            ),
            entity_scopes={
                "repository_id": manifest["repository_id"],
                "target_branch": manifest["target_branch"],
                "consumer_id": consumer,
                "recovery_scope_cid": scope_id,
            },
        )
        connection = None
        try:
            connection = TypedStateOwnerConnection(
                socket_path=server.typed_command_socket_path(),
                token=token,
                client_id=consumer,
                process_birth_id=birth,
                store_id=manifest["store_id"],
            )
            api = OwnerRecoveryRuntimeClient(
                connection,
                repository_id=manifest["repository_id"],
                target_branch=manifest["target_branch"],
                consumer_id=consumer,
                recovery_scope_cid=scope_id,
            )
            if api.describe_scope() != scope:
                raise SparMergeOwnerError("disposable role scope changed")
            api.load_cursors()  # Native service validates head and complete immutable history.
            for receipt in prepared.receipt_imports:
                observed = api.get_receipt(
                    receipt["receipt_key"], revision=receipt["revision"]
                )
                if (
                    observed is None
                    or observed["receipt_cid"] != receipt["receipt_cid"]
                    or observed["receipt"] != receipt["receipt"]
                ):
                    raise SparMergeOwnerError(
                        "preserved receipt was not served exactly"
                    )
            count += 1
        finally:
            if connection is not None:
                connection.close()
            server.revoke_typed_client_grant(grant.grant_id)
    return count


def qualify_offline_bundle(
    *,
    offline_root: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    transport_factory=None,
    capability_probe=None,
) -> dict[str, Any]:
    """Two disposable generations; never a live closure or cutover receipt.

    The supplied source IDs remain declared inputs. The native launcher must
    independently admit its current source/config/plan and old consumer closure
    before using a qualified store. No observer grant can settle queue work.
    """
    result = {
        "schema": "spar/legacy-queue-offline-qualification@1",
        "qualified": False,
        "live_custody_qualified": False,
        "completion_authority": False,
        "source_admission": False,
        "cycles": [],
        "transport": "real_quack"
        if transport_factory is None
        else "injected_test_transport",
    }
    try:
        prepared = prepare_offline_clone(
            offline_root=offline_root, destination=destination, manifest=manifest
        )
        result.update(
            manifest_cid=prepared.manifest_cid,
            database_uuid=prepared.database_uuid,
            declared_source_commit=prepared.manifest["source_commit"],
            declared_source_tree=prepared.manifest["source_tree"],
        )
        previous_identity = None
        for cycle in (1, 2):
            server = None
            evidence = {"cycle": cycle, "closed": False}
            result["cycles"].append(evidence)
            try:
                server = start_queue_owner(
                    prepared,
                    state_dir=destination / "qualification-owner",
                    transport=None
                    if transport_factory is None
                    else transport_factory(),
                    capability_probe=capability_probe,
                )
                identity = server.identity
                if previous_identity is not None and (
                    identity.database_uuid != previous_identity.database_uuid
                    or identity.generation != previous_identity.generation + 1
                ):
                    raise SparMergeOwnerError("disposable successor generation differs")
                previous_identity = identity
                if not _writer_lock_held(prepared.database_path):
                    raise SparMergeOwnerError("disposable startup writer lock missing")
                _require_independent_writer_excluded(prepared.database_path)
                evidence["scopes_read"] = _observe_role_reads(server, prepared)
                with server._owner_transaction_lock:
                    require_preserved(
                        prepared.preserved_inventory, inventory(server._connection)
                    )
                if server.checkpoint().get(
                    "checkpointed"
                ) is not True or not _writer_lock_held(prepared.database_path):
                    raise SparMergeOwnerError(
                        "disposable checkpoint writer lock missing"
                    )
                _require_independent_writer_excluded(prepared.database_path)
                evidence.update(
                    generation=identity.generation,
                    owner_identity=identity.to_dict(),
                    preserved=True,
                    startup_and_checkpoint_writer_lock=True,
                )
            finally:
                if server is not None:
                    server.stop()
                    if _writer_lock_held(prepared.database_path):
                        raise SparMergeOwnerError(
                            "disposable owner writer lock did not close"
                        )
                    evidence["closed"] = True
        for entry in prepared.manifest["files"]:
            copy_entry(offline_root, entry, None, digest_only=True)
        if set(file_inventory(offline_root)) != {
            entry["path"] for entry in prepared.manifest["files"]
        }:
            raise SparMergeOwnerError("offline input inventory changed after rehearsal")
        result["qualified"] = True
    except Exception as exc:
        result["qualified"] = False
        result["error_type"] = type(exc).__name__
        result["reason"] = "offline queue rehearsal refused; no live admission"
    return result
