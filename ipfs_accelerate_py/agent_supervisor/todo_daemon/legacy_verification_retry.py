"""Evidence for a fresh operator retry after a legacy verification timeout.

Older Portal attempts retained a worktree without a candidate fingerprint.
They cannot enter retained-candidate recovery. A completed provider lifecycle
can instead support a new bounded attempt, preserving the old workspace and
requiring fresh Portal validation. This module supplies evidence only; the
operator must retain native maintenance custody and use the existing scoped
Quack blocked-retry command. Ordinary daemon grants remain insufficient.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

from .database_portal_bridge import (
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
    _bounded_file,
    _strict_json_bytes,
    verify_database_portal_attempt_projection_identity,
)

REASON = "implementation_protected_path_verification_lock_timeout"
SCHEMA = "ipfs_accelerate_py/agent-supervisor/legacy-verification-fresh-retry-evidence@1"
_MAX_EVENTS_BYTES = 8 * 1024 * 1024


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def inspect_legacy_verification_retry(
    task_row: Mapping[str, Any], *, task_projection: Path,
    allowed_attempt_root: Path, retained_worktree_root: Path,
    expected_task_revision: int,
) -> dict[str, Any]:
    """Verify one finished, uncommitted lifecycle without changing its history.

    This deliberately rejects multiple implementations, queued candidates and
    post-terminal activity. Those cases need their own merge/lifecycle recovery;
    a fresh attempt must not hide an outstanding predecessor effect.
    """
    alias, task_cid = task_row.get("task_alias"), task_row.get("task_cid")
    if (not isinstance(alias, str) or not re.fullmatch(r"[A-Z][A-Z0-9]*-[0-9]+", alias)
            or not isinstance(task_cid, str) or not task_cid
            or type(expected_task_revision) is not int or expected_task_revision < 2
            or task_row.get("revision") != expected_task_revision
            or task_row.get("status") != "blocked"):
        raise DatabasePortalBridgeError("legacy verification retry requires an exact blocked task revision")
    body_raw = task_row.get("body_json")
    if not isinstance(body_raw, str) or len(body_raw.encode()) > _MAX_EVENTS_BYTES:
        raise DatabasePortalBridgeError("legacy verification retry task body is unavailable")
    body = _strict_json_bytes(body_raw.encode(), noun="legacy verification retry task body")
    terminal = body.get("completion_receipt") if isinstance(body, Mapping) else None
    if (not isinstance(terminal, Mapping)
            or terminal.get("operation") != "database_portal_terminal_failure"
            or terminal.get("reason") != REASON or terminal.get("retryable") is not False
            or terminal.get("execution_phase") != "failed"
            or terminal.get("control_expected_status") != "in_progress"
            or type(terminal.get("control_expected_revision")) is not int
            or terminal.get("control_expected_revision") != expected_task_revision - 1):
        raise DatabasePortalBridgeError("legacy verification retry terminal lineage differs")
    identity = verify_database_portal_attempt_projection_identity(
        task_projection, expected_task_alias=alias, expected_task_cid=task_cid,
        allowed_root=allowed_attempt_root,
    )
    for field in ("attempt_id", "claim_id", "lease_id", "owner_session_id",
                  "attempt_number", "fencing_token", "fence_epoch"):
        if terminal.get(field) != identity[field] or type(terminal.get(field)) is not type(identity[field]):
            raise DatabasePortalBridgeError("legacy verification retry differs from its immutable attempt")
    if (identity["task_revision"] != expected_task_revision - 1
            or identity["goal_cid"] != task_row.get("goal_cid")
            or identity["plan_cid"] != task_row.get("plan_cid")):
        raise DatabasePortalBridgeError("legacy verification retry task contract binding differs")
    events_path = Path(identity["projection_path"]).with_name("portal-events.jsonl")
    event_bytes = _bounded_file(events_path, limit=_MAX_EVENTS_BYTES)
    events = DatabasePortalExecutionBridge._verified_event_chain(
        SimpleNamespace(events=events_path), payload=event_bytes,
    )
    lifecycle_types = ("implementation_started", REASON,
                       "protected_path_verification_deferred_worktree_retained", "implementation_finished")
    lifecycle = [event for event in events if event.get("type") in lifecycle_types]
    if [event.get("type") for event in lifecycle] != list(lifecycle_types):
        raise DatabasePortalBridgeError("legacy verification retry has no single completed provider lifecycle")
    started, timeout, retained, finished = lifecycle
    for event in lifecycle:
        if (event.get("task_id") != alias
                or event.get("canonical_task_cid") != identity["portal_canonical_task_cid"]
                or event.get("canonical_task_key") != identity["portal_canonical_task_key"]
                or type(event.get("attempt")) is not int or event["attempt"] != 1):
            raise DatabasePortalBridgeError("legacy verification retry event identity differs")
    if (retained.get("previous_event_id") != timeout["event_id"]
            or finished.get("previous_event_id") != retained["event_id"]
            or any(event.get("type") != "daemon_pass"
                   for event in events[finished["sequence"]:])
            or any(event.get("type", "").startswith(("merge_", "task_completed", "worktree_reconciliation"))
                   for event in events)):
        raise DatabasePortalBridgeError("legacy verification retry has unsettled merge or terminal history")
    lock = timeout.get("lock")
    if (timeout.get("reason") != REASON or not isinstance(lock, Mapping)
            or lock.get("acquired") is not False or lock.get("reason") != "lock_exists"):
        raise DatabasePortalBridgeError("legacy verification retry is not an unacquired verification lock")
    for event in (retained, finished):
        cleanup, commit = event.get("cleanup_result"), event.get("commit_result")
        if (not isinstance(cleanup, Mapping) or cleanup.get("retained") is not True
                or cleanup.get("cleaned") is not False
                or cleanup.get("reason") != "verification_deferred_checkout_lease_active"
                or not isinstance(commit, Mapping) or commit.get("committed") is not False
                or commit.get("reason") != "verification_deferred_checkout_lease_active"
                or event.get("implementation_commit") not in (None, "")
                or event.get("retained_candidate_receipt") is not None):
            raise DatabasePortalBridgeError("legacy verification retry did not retain an unfingerprinted uncommitted workspace")
    merge, board = finished.get("merge_result"), finished.get("board_completion")
    preservation = finished.get("failed_preservation_result")
    if (finished.get("reason") != REASON or type(finished.get("returncode")) is not int
            or finished["returncode"] != 1 or finished.get("provider_dispatched") is not True
            or finished.get("attempt_consumed") is not False or finished.get("deferred") is not True
            or not isinstance(merge, Mapping) or merge.get("merged") is not False
            or merge.get("reason") != "not_attempted" or merge.get("queued") is True
            or not isinstance(board, Mapping) or board.get("complete") is not False
            or board.get("pending_merge") is not False
            or not isinstance(preservation, Mapping) or preservation.get("retained") is not True
            or preservation.get("preserved") is not False
            or preservation.get("retained_candidate_receipt") is not None
            or any(finished.get(key) is not None
                   for key in ("termination_result", "timeout_result", "exception_result"))):
        raise DatabasePortalBridgeError("legacy verification retry provider finish is not an uncommitted verification timeout")
    workspace = finished.get("worktree_path")
    branch, baseline = finished.get("branch"), finished.get("baseline_ref")
    if (not isinstance(workspace, str) or not workspace or len(workspace.encode()) > 4096
            or not isinstance(branch, str) or not branch.startswith("implementation/")
            or not isinstance(baseline, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", baseline) is None
            or any(event.get("worktree_path") != workspace for event in (started, retained))
            or timeout.get("workspace_path") != workspace
            or any(event.get("branch") != branch for event in (started, retained))
            or started.get("baseline_ref") != baseline):
        raise DatabasePortalBridgeError("legacy verification retry retained workspace binding differs")
    path = Path(workspace)
    root = retained_worktree_root.resolve(strict=True)
    if (not path.is_absolute() or path.is_symlink() or not path.is_dir()
            or path.resolve(strict=True).parent != root):
        raise DatabasePortalBridgeError("legacy verification retry retained workspace is outside its owned pool")
    if verify_database_portal_attempt_projection_identity(
        task_projection, expected_task_alias=alias, expected_task_cid=task_cid,
        allowed_root=allowed_attempt_root,
    ) != identity or _bounded_file(events_path, limit=_MAX_EVENTS_BYTES) != event_bytes:
        raise DatabasePortalBridgeError("legacy verification retry history changed during observation")
    material = {
        "schema": SCHEMA, "task_alias": alias, "task_cid": task_cid,
        "expected_task_revision": expected_task_revision,
        "task_body_digest": _digest(body), "terminal_receipt_digest": _digest(terminal),
        "projection_identity": identity, "events_path": str(events_path),
        "events_digest": "sha256:" + hashlib.sha256(event_bytes).hexdigest(),
        "event_ids": [event["event_id"] for event in lifecycle],
        "retained_worktree": str(path), "retained_branch": branch, "retained_baseline": baseline,
        "historical_candidate_fingerprint": "unavailable", "retained_candidate_admitted": False,
        "fresh_attempt_number": terminal["attempt_number"] + 1,
        "attempt_refunded": False, "require_fresh_portal_revalidation": True,
        "retry_authorized": False,
    }
    return {**material, "evidence_id": _digest(material)}


@contextmanager
def hold_legacy_retry_queue_absence(
    *, queue_dir: Path, target_repository_id: str, target_branch: str,
    evidence: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Keep the native queue guard through an operator's scoped Quack CAS.

    Any queue history naming this task, including completed or quarantined
    candidates, requires reconciliation before using this fresh-attempt path.
    The existing file-backed queue is observed in place, never initialized or
    migrated to a competing store. A task-owned Quack token cannot own it.
    """
    from ..merge.merge_queue import (
        _settlement_regular_file_identity,
        hold_merge_queue_settlement,
        connect_duckdb_with_policy,
    )
    import duckdb

    material = dict(evidence)
    if material.pop("evidence_id", None) != _digest(material) or material.get("schema") != SCHEMA:
        raise DatabasePortalBridgeError("legacy retry queue guard requires verified evidence")
    alias, task_cid = material["task_alias"], material["task_cid"]
    portal_cid = material["projection_identity"]["portal_canonical_task_cid"]
    with hold_merge_queue_settlement(
        queue_dir, target_repository_id=target_repository_id, target_branch=target_branch,
    ) as settlement:
        if settlement["row_count"] > 8192:
            raise DatabasePortalBridgeError("legacy retry queue observation exceeds its bound")
        database = Path(settlement["database"]["path"])
        before = _settlement_regular_file_identity(database, label="database")
        if before != {key: settlement["database"][key] for key in before}:
            raise DatabasePortalBridgeError("legacy retry queue identity differs from its settlement")
        connection = connect_duckdb_with_policy(duckdb, database, read_only=True)
        try:
            rows = connection.execute("""
                SELECT CASE WHEN octet_length(encode(task_id)) <= 1024 THEN task_id END,
                    CASE WHEN octet_length(encode(canonical_task_id)) <= 1024 THEN canonical_task_id END,
                    CASE WHEN octet_length(encode(completion_bindings)) <= 65536
                         THEN completion_bindings END
                FROM (SELECT request_id, task_id, canonical_task_id,
                    CAST(json_extract(metadata_json, '$.completion_task_cids') AS VARCHAR) AS completion_bindings
                    FROM merge_requests)
                ORDER BY request_id LIMIT 8193
            """).fetchall()
        finally:
            connection.close()
        if (len(rows) != settlement["row_count"]
                or _settlement_regular_file_identity(database, label="database") != before):
            raise DatabasePortalBridgeError("legacy retry queue changed during observation")
        for task_alias, canonical_cid, bindings_raw in rows:
            if (not isinstance(task_alias, str) or not task_alias
                    or not isinstance(canonical_cid, str) or not canonical_cid
                    or not isinstance(bindings_raw, str) or len(bindings_raw.encode()) > 65536):
                raise DatabasePortalBridgeError("legacy retry queue task bindings are unavailable")
            bindings = _strict_json_bytes(bindings_raw.encode(), noun="legacy retry queue bindings")
            if (not isinstance(bindings, dict) or not bindings
                    or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                           for k, v in bindings.items())
                    or bindings.get(task_alias) != canonical_cid):
                raise DatabasePortalBridgeError("legacy retry queue task bindings are malformed")
            if (task_alias == alias or canonical_cid in {task_cid, portal_cid}
                    or alias in bindings or set(bindings.values()).intersection({task_cid, portal_cid})):
                raise DatabasePortalBridgeError("legacy verification retry has existing task merge history")
        result = {"schema": SCHEMA + "/queue-absence", "task_alias": alias, "task_cid": task_cid,
                  "queue_settlement": settlement, "matching_queue_rows": 0,
                  "evidence_id": evidence["evidence_id"], "guard_retained": True,
                  "retry_authorized": False}
        yield {**result, "receipt_id": _digest(result)}
