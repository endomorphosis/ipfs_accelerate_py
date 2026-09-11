"""Produce native import mappings from a preserved legacy queue copy.

This is an offline producer, not a process-close or capture admission API.
It opens only its own new copy, never the supplied database. Every input byte
is preserved before inspection. Current request coordinates establish known
native receipt keys; filenames alone cannot establish their original keys.
Unknown callback outcomes and signing material are copied, never replaced or
interpreted as acceptance. Legacy production launch remains independently held.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from types import SimpleNamespace

from . import spar_merge_owner as role
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    MERGE_TARGET_BINDING_SCHEMA,
    _MERGE_QUEUE_SETTLEMENT_COLUMNS,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain

SCHEMA = "spar/native-legacy-import-plan@1"
CONTEXT_FIELDS = {
    "repository_id",
    "target_branch",
    "store_id",
    "source_commit",
    "source_tree",
    "scope_bindings",
    "queue_policy",
}


def _entries(root, identities):
    result = []
    total = 0
    for name, expected in sorted(identities.items()):
        descriptor = role._open_regular(root, name)
        try:
            if role._file_identity(os.fstat(descriptor)) != expected:
                raise role.SparMergeOwnerError("captured input identity changed")
            total += expected[2]
            if total > role.MAX_INPUT_BYTES:
                raise role.SparMergeOwnerError("captured input exceeds byte bound")
            digest = hashlib.sha256()
            remaining = expected[2]
            while remaining:
                block = os.read(descriptor, min(1024 * 1024, remaining))
                if not block:
                    raise role.SparMergeOwnerError("captured input was truncated")
                digest.update(block)
                remaining -= len(block)
            if (
                os.read(descriptor, 1)
                or role._file_identity(os.fstat(descriptor)) != expected
            ):
                raise role.SparMergeOwnerError("captured input identity changed")
            result.append(
                {"path": name, "size_bytes": expected[2], "sha256": digest.hexdigest()}
            )
        finally:
            os.close(descriptor)
    return result


def _receipt_imports(connection, root, entries, context):
    columns = [row[0] for row in _MERGE_QUEUE_SETTLEMENT_COLUMNS["merge_requests"]]
    rows = connection.execute(
        "SELECT * FROM merge_requests ORDER BY request_id LIMIT ?", [role.MAX_ROWS + 1]
    ).fetchall()
    if len(rows) > role.MAX_ROWS:
        raise role.SparMergeOwnerError("native receipt population exceeds bound")
    requests = {}
    for row in rows:
        item = {name: row[index] for index, name in enumerate(columns)}
        key = item["request_id"]
        if (
            type(key) is not str
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key)
            or key in requests
        ):
            raise role.SparMergeOwnerError("native request identity is ambiguous")
        requests[key] = item
    train = SimpleNamespace(
        queue=SimpleNamespace(
            target_repository_id=context["repository_id"],
            target_branch=context["target_branch"],
        )
    )
    imports = []
    for path, entry in sorted(entries.items()):
        if not path.startswith("train/receipts/") or not path.endswith(".json"):
            continue
        body = role._decode(role.copy_entry(root, entry, None))
        request = requests.get(body.get("request_id"))
        if request is None:
            raise role.SparMergeOwnerError("canonical receipt has no preserved request")
        metadata = role._decode(request["metadata_json"].encode())
        if (
            metadata.get("target_binding_schema") != MERGE_TARGET_BINDING_SCHEMA
            or metadata.get("target_repository_id") != context["repository_id"]
            or metadata.get("target_branch") != context["target_branch"]
        ):
            raise role.SparMergeOwnerError(
                "canonical receipt request target is unbound"
            )
        canonical = (
            request["canonical_task_key"]
            or request["canonical_task_id"]
            or request["task_id"]
        )
        candidate = request["commit_sha"]
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", candidate or ""):
            raise role.SparMergeOwnerError(
                "native candidate requires its full captured object ID"
            )
        for field, expected in {
            "request_id": request["request_id"],
            "task_id": request["task_id"],
            "canonical_task_id": canonical,
            "commit_sha": candidate,
            "target_branch": context["target_branch"],
        }.items():
            if body.get(field) != expected:
                raise role.SparMergeOwnerError(
                    "canonical receipt differs from preserved request"
                )
        # The primary key comes from the native source's exact forward function.
        # The quarantine variant comes from _finish_failure's literal producer.
        # No arbitrary reverse mapping of lossy receipt filenames is permitted.
        primary = MergeTrain._dedupe_key(train, canonical, candidate)
        candidates = [primary, "quarantine-" + request["request_id"]]
        matches = [
            key for key in candidates if path == "train/receipts/" + key + ".json"
        ]
        if len(matches) != 1:
            raise role.SparMergeOwnerError(
                "receipt has no unambiguous native key producer"
            )
        imports.append(
            {
                "path": path,
                "receipt_key": matches[0],
                "revision": 1,
                "receipt_cid": role._cid(body),
            }
        )
    ledger = "train/distributed-publications.json"
    if ledger in entries:
        body = role._decode(role.copy_entry(root, entries[ledger], None))
        imports.append(
            {
                "path": ledger,
                "receipt_key": "distributed-publications",
                "revision": 1,
                "receipt_cid": role._cid(body),
            }
        )
    return imports


def _cursor_imports(root, entries, context):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        _POST_MERGE_RECOVERY_CURSOR_SCHEMA,
        _canonical_json,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )

    by_path = {}
    for scope in context["scope_bindings"]:
        binding = {
            "target_repository_id": context["repository_id"],
            "target_branch": context["target_branch"],
            "attempt_root": scope["attempt_root"],
        }
        path = (
            "train/post-merge-recovery-cursors/"
            + hashlib.sha256(_canonical_json(binding)).hexdigest()
            + ".json"
        )
        if path in by_path:
            raise role.SparMergeOwnerError("captured cursor scope is ambiguous")
        by_path[path] = (scope, binding)
    result = []
    for path, entry in sorted(entries.items()):
        if not path.startswith(
            "train/post-merge-recovery-cursors/"
        ) or not path.endswith(".json"):
            continue
        if path not in by_path:
            raise role.SparMergeOwnerError(
                "captured cursor has no current native scope"
            )
        scope, binding = by_path[path]
        body = role._decode(role.copy_entry(root, entry, None))
        role._closed(
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
        if (
            body["schema"] != _POST_MERGE_RECOVERY_CURSOR_SCHEMA
            or any(body[k] != v for k, v in binding.items())
            or type(body["cursors"]) is not dict
            or set(body["cursors"]) != set(role.STAGES)
            or any(
                type(v) is not str or len(v) > 4096 for v in body["cursors"].values()
            )
            or body["state_id"]
            != content_identity({k: v for k, v in body.items() if k != "state_id"})
        ):
            raise role.SparMergeOwnerError(
                "captured cursor differs from native binding"
            )
        result.append(
            {
                "path": path,
                "scope_cid": role.recovery_scope_cid(
                    store_id=context["store_id"],
                    repository_id=context["repository_id"],
                    target_branch=context["target_branch"],
                    scope_binding=scope,
                ),
            }
        )
    return result


def produce_offline_import_plan(
    *, offline_root: Path, destination: Path, context: dict
) -> dict:
    """Copy an explicit offline bundle and derive imports without changing it.

    The caller must independently establish capture coherence and stop/callback
    custody. Source IDs remain declared inputs. This result cannot select a live
    legacy launch, issue grants, release claims, or mint acceptance signatures.
    Revision one denotes first import into the new recovery store; it does not
    reconstruct any historical versions the old filesystem no longer contains.
    """
    role._closed(context, CONTEXT_FIELDS)
    # Validate all context fields using an empty disposable manifest before I/O.
    role.validate_manifest(
        {
            **context,
            "schema": role.SCHEMA,
            "database": "merge_queue.duckdb",
            "wal": None,
            "receipt_imports": [],
            "cursor_imports": [],
            "files": [
                {"path": "merge_queue.duckdb", "size_bytes": 0, "sha256": "0" * 64}
            ],
        }
    )
    offline_root, destination = (
        Path(offline_root).absolute(),
        Path(destination).absolute(),
    )
    if destination == offline_root or destination.is_relative_to(offline_root):
        raise role.SparMergeOwnerError("offline inspection destination overlaps input")
    descriptor = role._open_directory(destination.parent)
    os.close(descriptor)
    identities = role.file_inventory(offline_root)
    role._refuse_observed_input_locks(identities)
    entries = _entries(offline_root, identities)
    if "merge_queue.duckdb" not in identities:
        raise role.SparMergeOwnerError("captured canonical database is missing")
    destination.mkdir(mode=0o700)
    for entry in entries:
        role._refuse_observed_input_locks(identities)
        role.copy_entry(offline_root, entry, destination / entry["path"])
    if role.file_inventory(offline_root) != identities:
        raise role.SparMergeOwnerError("captured namespace changed while copying")
    for entry in entries:
        role.copy_entry(offline_root, entry, None, digest_only=True)
    by_path = {e["path"]: e for e in entries}
    # Only this new private copy is opened. WAL replay cannot alter the input.
    with role.open_duckdb_connection(
        destination / "merge_queue.duckdb", prefer_quack=False
    ) as connection:
        baseline = role.inventory(connection)
        for table, columns in _MERGE_QUEUE_SETTLEMENT_COLUMNS.items():
            if table not in baseline or baseline[table]["columns"] != [
                list(c) for c in columns
            ]:
                raise role.SparMergeOwnerError("captured native queue schema differs")
        receipts = _receipt_imports(connection, destination, by_path, context)
        cursors = _cursor_imports(destination, by_path, context)
        if role.inventory(connection) != baseline:
            raise role.SparMergeOwnerError(
                "import mapping changed preserved logical state"
            )
    role._refuse_observed_input_locks(identities)
    if role.file_inventory(offline_root) != identities:
        raise role.SparMergeOwnerError("captured namespace changed during inspection")
    manifest = role.validate_manifest(
        {
            **context,
            "schema": role.SCHEMA,
            "database": "merge_queue.duckdb",
            "wal": "merge_queue.duckdb.wal"
            if "merge_queue.duckdb.wal" in identities
            else None,
            "files": entries,
            "receipt_imports": receipts,
            "cursor_imports": cursors,
        }
    )
    return {
        "schema": SCHEMA,
        "manifest": manifest,
        "manifest_cid": role._cid(manifest),
        "preserved_inventory": baseline,
        "capture_coherent": False,
        "consumer_closed": False,
        "callback_settled": False,
        "source_admitted": False,
        "completion_authority": False,
    }
