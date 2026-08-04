#!/usr/bin/env python3
"""Perform the exact, fail-closed FVT-086 -> FVT-101 board migration.

This is intentionally a one-purpose operator migration, not a generic task
supersession mechanism.  FVT-086 and FVT-101 have different evidence
contracts.  The former is retained as blocked historical work, while only the
three reviewed scheduling dependencies are rebound to the successfully merged
FVT-101 repair.

The migration requires a stopped supervisor and a durable FVT-101 completion
receipt, snapshots every mutable projection, holds the repository-wide
protected-path maintenance lease, regenerates artifacts through the existing
writers, validates JSON/DuckDB/runtime parity, and restores the exact snapshot
on any ordinary failure.  A durable journal supports explicit recovery after
process or host interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(DEFAULT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (  # noqa: E402
    rehydrate_task_work_contract_projection,
)
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (  # noqa: E402
    PROTECTED_PATH_MAINTENANCE_LOCK_NAME,
    CheckoutMaintenanceLease,
    checkout_mutation_lock_path,
)
from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (  # noqa: E402
    bundle_member_completion_receipts,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    build_bundle_task_payloads,
    write_bundle_shards,
)
from ipfs_accelerate_py.agent_supervisor.runtime.artifact_store import (  # noqa: E402
    write_bundle_index_artifact,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (  # noqa: E402
    write_todo_vector_index,
)

MIGRATION_SCHEMA = (
    "formal-verification-tactician/fvt086-to-fvt101-migration@1"
)
JOURNAL_SCHEMA = (
    "formal-verification-tactician/fvt086-to-fvt101-journal@1"
)
SOURCE_TASK_ID = "FVT-086"
REPLACEMENT_TASK_ID = "FVT-101"
DIRECT_DEPENDENT_TASK_IDS = ("FVT-088", "FVT-092", "FVT-099")
TASK_CIDS = {
    "FVT-086": "baguqeeram465zjadscydj5mjdh2se33udmqskmu2tfe6nco4lhbhv3emqkya",
    "FVT-088": "baguqeeraej7hdfylgv2ifgn3pgxnnh5kd3ipqalo7nkyhcj4d3fpun7le5ua",
    "FVT-092": "baguqeeraikp3hrazxplac5niv4ikngrb4zuehqjbjad3qkmpbwcrflbs466q",
    "FVT-099": "baguqeeraziap5lduyvpkekg37cbkrh2cpzgzowvuhw7whlhggv4ks5wmhbha",
    "FVT-101": "baguqeera2wbd6pcboahukx3xvm3kp3hrumumex27qhkaquhokp5t7xtlsbma",
}
BOARD_RELATIVE = Path(
    "docs/architecture/formal_verification_tactician_readiness.todo.md"
)
OBJECTIVE_RELATIVE = Path(
    "docs/architecture/formal_verification_tactician_readiness.objectives.md"
)
BUNDLE_DIR_RELATIVE = Path(
    "data/agent_supervisor/formal_verification_tactician_readiness/bundles"
)
LIVE_RELATIVE = Path(
    "data/agent_supervisor/formal_verification_tactician_readiness/live"
)
INDEX_RELATIVE = BUNDLE_DIR_RELATIVE / "index.json"
INDEX_DATABASE_RELATIVE = BUNDLE_DIR_RELATIVE / "index.duckdb"
VECTOR_RELATIVE = BUNDLE_DIR_RELATIVE / "todo_vector_index.json"
DATASET_DIR_RELATIVE = Path(
    "data/agent_supervisor/formal_verification_tactician_readiness/datasets"
)
VECTOR_DATASET_ID = "fvt-todo-vector-index"
VECTOR_DATASET_JSONL_RELATIVE = (
    DATASET_DIR_RELATIVE / f"{VECTOR_DATASET_ID}.jsonl"
)
VECTOR_DATASET_MANIFEST_RELATIVE = (
    DATASET_DIR_RELATIVE / f"{VECTOR_DATASET_ID}.manifest.json"
)
VECTOR_DATASET_PARQUET_RELATIVE = (
    DATASET_DIR_RELATIVE / f"{VECTOR_DATASET_ID}.parquet"
)
RECEIPT_RELATIVE = Path(
    "docs/architecture/"
    "formal_verification_fvt086_to_fvt101_migration_receipt.json"
)
SHARD_RELATIVES = {
    "FVT-086": BUNDLE_DIR_RELATIVE
    / "formal-verification-tactician-secpal-live-toolchain.todo.md",
    "FVT-088": BUNDLE_DIR_RELATIVE
    / "formal-verification-tactician-end-to-end-assurance.todo.md",
    "FVT-092": BUNDLE_DIR_RELATIVE
    / "formal-verification-tactician-secpal-operator-compatibility.todo.md",
    "FVT-099": BUNDLE_DIR_RELATIVE
    / "formal-verification-tactician-production-authorization-replacement.todo.md",
}
SNAPSHOT_RELATIVES = (
    BOARD_RELATIVE,
    *SHARD_RELATIVES.values(),
    INDEX_RELATIVE,
    INDEX_DATABASE_RELATIVE,
    VECTOR_RELATIVE,
    VECTOR_DATASET_JSONL_RELATIVE,
    VECTOR_DATASET_MANIFEST_RELATIVE,
    VECTOR_DATASET_PARQUET_RELATIVE,
    RECEIPT_RELATIVE,
)
TERMINAL_JOURNAL_PHASES = frozenset(
    {"completed", "recovered", "rolled_back"}
)
TERMINAL_LEASE_STATES = frozenset(
    {"cancelled", "completed", "failed", "released"}
)


class MigrationError(RuntimeError):
    """Raised when a migration invariant is not proved."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _repository_relative_path(repo_root: Path, value: Any) -> str:
    """Render repository-owned evidence without checkout-specific prefixes."""

    path = Path(str(value or ""))
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _portable_preflight_evidence(
    repo_root: Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Detach and normalize paths embedded in durable receipt evidence."""

    portable = json.loads(json.dumps(dict(evidence)))
    completion = portable.get("fvt101_completion_receipt")
    if isinstance(completion, dict) and completion.get("event_path"):
        completion["event_path"] = _repository_relative_path(
            repo_root, completion["event_path"]
        )
    return portable


def _portable_snapshot_evidence(
    repo_root: Path, snapshot: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Keep recovery paths local while making receipt snapshots portable."""

    portable: list[dict[str, Any]] = []
    for raw_record in snapshot:
        record = dict(raw_record)
        if record.get("backup_path"):
            record["backup_path"] = _repository_relative_path(
                repo_root, record["backup_path"]
            )
        portable.append(record)
    return portable


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary_path, path.stat().st_mode & 0o777)
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def task_block(text: str, task_id: str) -> str:
    """Return the unique Markdown block for ``task_id``."""

    matches = list(
        re.finditer(
            rf"^## {re.escape(task_id)}(?:\s|$).*?(?=^## |\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
    )
    if len(matches) != 1:
        raise MigrationError(
            f"expected one {task_id} block, observed {len(matches)}"
        )
    return matches[0].group(0).rstrip() + "\n"


def replace_task_block(text: str, task_id: str, replacement: str) -> str:
    """Replace the unique Markdown block for ``task_id``."""

    matches = list(
        re.finditer(
            rf"^## {re.escape(task_id)}(?:\s|$).*?(?=^## |\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
    )
    if len(matches) != 1:
        raise MigrationError(
            f"expected one {task_id} block, observed {len(matches)}"
        )
    match = matches[0]
    trailing = re.search(r"(\n*)\Z", match.group(0))
    suffix = trailing.group(1) if trailing is not None else ""
    rendered = replacement.rstrip("\n") + suffix
    return text[: match.start()] + rendered + text[match.end() :]


def _replace_scalar_field(
    block: str,
    field: str,
    expected: str,
    replacement: str,
) -> str:
    pattern = rf"^- {re.escape(field)}: (.+)$"
    matches = re.findall(pattern, block, flags=re.MULTILINE)
    if matches != [expected]:
        raise MigrationError(
            f"{SOURCE_TASK_ID} {field!r} expected {expected!r}, "
            f"observed {matches!r}"
        )
    return re.sub(
        pattern,
        f"- {field}: {replacement}",
        block,
        count=1,
        flags=re.MULTILINE,
    )


def retire_source_block(block: str) -> str:
    """Retire FVT-086 without manufacturing completion authority."""

    if not block.startswith(f"## {SOURCE_TASK_ID} "):
        raise MigrationError("source retirement received the wrong task block")
    result = _replace_scalar_field(block, "Status", "todo", "blocked")
    result = _replace_scalar_field(
        result, "Completion", "manual", "superseded:FVT-101"
    )
    result = _replace_scalar_field(
        result, "Is schedulable", "true", "false"
    )
    result = _replace_scalar_field(result, "Review only", "false", "true")
    result = _replace_scalar_field(
        result, "Completion authority", "local", "none"
    )
    anchor = "- Completion: superseded:FVT-101\n"
    if result.count(anchor) != 1:
        raise MigrationError("source completion anchor is ambiguous")
    result = result.replace(
        anchor,
        anchor
        + "- Superseded by: FVT-101\n"
        + "- Supersession completion authority: none\n"
        + "- Historical task: true\n",
        1,
    )
    return result


def rewire_dependent_block(block: str, task_id: str) -> str:
    """Replace exactly one FVT-086 scheduling dependency with FVT-101."""

    if task_id not in DIRECT_DEPENDENT_TASK_IDS:
        raise MigrationError(f"unreviewed dependent task: {task_id}")
    match = re.search(r"^- Depends on: (.+)$", block, flags=re.MULTILINE)
    if match is None:
        raise MigrationError(f"{task_id} has no Depends on field")
    dependencies = [item.strip() for item in match.group(1).split(",")]
    if dependencies.count(SOURCE_TASK_ID) != 1:
        raise MigrationError(
            f"{task_id} must depend on {SOURCE_TASK_ID} exactly once"
        )
    if REPLACEMENT_TASK_ID in dependencies:
        raise MigrationError(
            f"{task_id} already contains {REPLACEMENT_TASK_ID}"
        )
    rewritten = [
        REPLACEMENT_TASK_ID if item == SOURCE_TASK_ID else item
        for item in dependencies
    ]
    replacement = "- Depends on: " + ", ".join(rewritten)
    return block[: match.start()] + replacement + block[match.end() :]


def rewrite_markdown_documents(
    board_text: str,
    shard_texts: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    """Return the exact aggregate and shard migration projection."""

    expected_ids = {SOURCE_TASK_ID, *DIRECT_DEPENDENT_TASK_IDS}
    if set(shard_texts) != expected_ids:
        raise MigrationError("shard task set does not match reviewed migration")

    dependency_users = {
        match.group(1)
        for match in re.finditer(
            r"^## (FVT-[0-9]+).*?(?=^## |\Z)",
            board_text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if re.search(
            rf"^- Depends on: .*\b{re.escape(SOURCE_TASK_ID)}\b",
            match.group(0),
            flags=re.MULTILINE,
        )
    }
    if dependency_users != set(DIRECT_DEPENDENT_TASK_IDS):
        raise MigrationError(
            "direct FVT-086 dependency closure changed: "
            + ",".join(sorted(dependency_users))
        )

    rewritten_blocks = {
        SOURCE_TASK_ID: retire_source_block(
            task_block(board_text, SOURCE_TASK_ID)
        )
    }
    for task_id in DIRECT_DEPENDENT_TASK_IDS:
        rewritten_blocks[task_id] = rewire_dependent_block(
            task_block(board_text, task_id), task_id
        )

    rewritten_board = board_text
    rewritten_shards: dict[str, str] = {}
    for task_id, block in rewritten_blocks.items():
        current_shard_block = task_block(shard_texts[task_id], task_id)
        if current_shard_block.strip() != task_block(
            board_text, task_id
        ).strip():
            raise MigrationError(
                f"aggregate/shard input parity failed for {task_id}"
            )
        rewritten_board = replace_task_block(
            rewritten_board, task_id, block
        )
        rewritten_shards[task_id] = replace_task_block(
            shard_texts[task_id], task_id, block
        )
    return rewritten_board, rewritten_shards


def _bundle_tasks(index_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bundles = index_payload.get("bundles")
    if not isinstance(bundles, Mapping):
        raise MigrationError("bundle index has no bundles mapping")
    result: dict[str, dict[str, Any]] = {}
    for bundle in bundles.values():
        if not isinstance(bundle, Mapping):
            continue
        for raw_task in bundle.get("tasks") or []:
            if not isinstance(raw_task, dict):
                continue
            task_id = str(raw_task.get("task_id") or "")
            if not task_id:
                continue
            if task_id in result:
                raise MigrationError(f"duplicate indexed task {task_id}")
            result[task_id] = raw_task
    return result


def _replace_dependency_values(
    values: Any, *, task_id: str, field: str
) -> list[str]:
    if not isinstance(values, list):
        raise MigrationError(f"{task_id} {field} is not a list")
    normalized = [str(item) for item in values]
    if normalized.count(SOURCE_TASK_ID) != 1:
        raise MigrationError(
            f"{task_id} {field} must contain {SOURCE_TASK_ID} exactly once"
        )
    if REPLACEMENT_TASK_ID in normalized:
        raise MigrationError(
            f"{task_id} {field} already contains {REPLACEMENT_TASK_ID}"
        )
    return [
        REPLACEMENT_TASK_ID if value == SOURCE_TASK_ID else value
        for value in normalized
    ]


def rewrite_bundle_index(
    index_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Rewrite the exact index rows and invalidate three work contracts."""

    tasks = _bundle_tasks(index_payload)
    for task_id, expected_cid in TASK_CIDS.items():
        task = tasks.get(task_id)
        if task is None:
            raise MigrationError(f"bundle index is missing {task_id}")
        if str(task.get("canonical_task_cid") or "") != expected_cid:
            raise MigrationError(f"canonical CID mismatch for {task_id}")

    source = tasks[SOURCE_TASK_ID]
    expected_source = {
        "status": "todo",
        "is_schedulable": True,
        "review_only": False,
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            raise MigrationError(
                f"source index {field} expected {expected!r}, "
                f"observed {source.get(field)!r}"
            )
    if source.get("completion_authority") not in ("", "local"):
        raise MigrationError(
            "source index completion_authority expected an empty legacy "
            f"projection or 'local', observed {source.get('completion_authority')!r}"
        )
    source.update(
        {
            "status": "blocked",
            "is_schedulable": False,
            "review_only": True,
            "completion_authority": "none",
            "superseded_by": REPLACEMENT_TASK_ID,
            "supersession_completion_authority": "none",
            "historical_task": True,
        }
    )
    surface = source.get("conflict_surface")
    if isinstance(surface, dict):
        surface["completion_authority"] = "none"

    before: dict[str, dict[str, Any]] = {}
    for task_id in DIRECT_DEPENDENT_TASK_IDS:
        task = tasks[task_id]
        if task.get("validation_receipts") not in (None, []):
            raise MigrationError(
                f"{task_id} already carries validation receipts"
            )
        before[task_id] = {
            "canonical_task_cid": task["canonical_task_cid"],
            "canonical_task_key": task.get("canonical_task_key", ""),
            "depends_on": list(task.get("depends_on") or []),
            "work_contract_id": str(task.get("work_contract_id") or ""),
            "task_work_contract_id": str(
                task.get("task_work_contract_id") or ""
            ),
        }
        if not before[task_id]["work_contract_id"] or not before[task_id][
            "task_work_contract_id"
        ]:
            raise MigrationError(f"{task_id} has no admitted work contract")
        for field in (
            "depends_on",
            "dependency_task_ids",
            "dependency_task_cids",
        ):
            task[field] = _replace_dependency_values(
                task.get(field), task_id=task_id, field=field
            )
        for field in (
            "work_contract",
            "work_contract_id",
            "task_work_contract",
            "task_work_contract_id",
        ):
            task.pop(field, None)
        nested = task.get("conflict_surface")
        if isinstance(nested, dict):
            for field in (
                "work_contract",
                "work_contract_id",
                "task_work_contract",
                "task_work_contract_id",
            ):
                nested.pop(field, None)
    return index_payload, before


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _process_command_line(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(
            b"\0", b" "
        ).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _maintenance_owner_is_active(metadata: Mapping[str, Any]) -> bool:
    if metadata.get("protected_recovery_required") is True:
        return True
    try:
        pid = int(metadata.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    return _process_is_running(pid)


def _coordination_evidence(
    coordination_path: Path,
) -> dict[str, Any]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - production dependency
        raise MigrationError("DuckDB is required for coordination audit") from exc

    connection = duckdb.connect(str(coordination_path), read_only=True)
    try:
        active_leases = connection.execute(
            "SELECT task_cid, state, attempt, expires_at_ms "
            "FROM leases WHERE lower(state) NOT IN "
            "('cancelled','completed','failed','released')"
        ).fetchall()
        if active_leases:
            raise MigrationError(
                f"coordination has active leases: {active_leases[:5]!r}"
            )
        direct_rows = connection.execute(
            "SELECT task_cid, task_id FROM tasks "
            "WHERE task_id IN ('FVT-088','FVT-092','FVT-099')"
        ).fetchall()
        direct_task_cids = [str(row[0]) for row in direct_rows]
        direct_lease_rows: list[tuple[Any, ...]] = []
        direct_receipt_rows: list[tuple[Any, ...]] = []
        if direct_task_cids:
            placeholders = ",".join("?" for _ in direct_task_cids)
            direct_lease_rows = connection.execute(
                "SELECT task_cid, state, attempt FROM leases WHERE task_cid "
                f"IN ({placeholders})",
                direct_task_cids,
            ).fetchall()
            direct_receipt_rows = connection.execute(
                "SELECT receipt_cid, task_cid, "
                "json_extract_string(payload_json, '$.status') "
                f"FROM receipts WHERE task_cid IN ({placeholders})",
                direct_task_cids,
            ).fetchall()
        if direct_lease_rows or direct_receipt_rows:
            raise MigrationError(
                "direct dependents already have coordination attempts or "
                "receipts"
            )

        replacement_alias = connection.execute(
            "SELECT task_cid FROM task_aliases WHERE alias_task_cid = ?",
            [TASK_CIDS[REPLACEMENT_TASK_ID]],
        ).fetchone()
        if replacement_alias is None:
            raise MigrationError("FVT-101 has no coordination alias")
        replacement_receipts = connection.execute(
            "SELECT receipt_cid, "
            "json_extract_string(payload_json, '$.status') "
            "FROM receipts WHERE task_cid = ?",
            [replacement_alias[0]],
        ).fetchall()
        succeeded = [
            row
            for row in replacement_receipts
            if str(row[1] or "").casefold() == "succeeded"
        ]
        if len(succeeded) != 1:
            raise MigrationError(
                "FVT-101 requires exactly one successful coordination receipt"
            )
        source_alias = connection.execute(
            "SELECT task_cid FROM task_aliases WHERE alias_task_cid = ?",
            [TASK_CIDS[SOURCE_TASK_ID]],
        ).fetchone()
        source_receipts: list[tuple[Any, ...]] = []
        if source_alias is not None:
            source_receipts = connection.execute(
                "SELECT receipt_cid, "
                "json_extract_string(payload_json, '$.status') "
                "FROM receipts WHERE task_cid = ?",
                [source_alias[0]],
            ).fetchall()
        if any(
            str(row[1] or "").casefold() == "succeeded"
            for row in source_receipts
        ):
            raise MigrationError("FVT-086 unexpectedly has a success receipt")
        return {
            "active_lease_count": 0,
            "direct_task_count": len(direct_rows),
            "direct_attempt_or_receipt_count": 0,
            "fvt101_profile_task_cid": str(replacement_alias[0]),
            "fvt101_coordination_receipt_cid": str(succeeded[0][0]),
            "fvt086_non_success_receipts": [
                {"receipt_cid": str(row[0]), "status": str(row[1] or "")}
                for row in source_receipts
            ],
        }
    finally:
        connection.close()


def preflight(repo_root: Path) -> dict[str, Any]:
    """Prove the stopped, receipt-bound migration preconditions."""

    live_root = repo_root / LIVE_RELATIVE
    manifest_path = live_root / "lane_manifest.json"
    coordination_path = live_root / "coordination.duckdb"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("scheduler_state") != "stopped":
        raise MigrationError("bundle supervisor is not stopped")
    for field in ("active_worker_count", "running_count", "started_count"):
        if int(manifest.get(field) or 0) != 0:
            raise MigrationError(f"bundle supervisor {field} is nonzero")
    active_pids = [int(value) for value in manifest.get("active_worker_pids") or []]
    if any(_process_is_running(pid) for pid in active_pids):
        raise MigrationError("bundle supervisor still has live worker PIDs")
    supervisor_pid = int(manifest.get("supervisor_pid") or 0)
    if supervisor_pid and _process_is_running(supervisor_pid):
        command_line = _process_command_line(supervisor_pid)
        if str(live_root) in command_line:
            raise MigrationError("bundle supervisor PID is still active")

    receipts = bundle_member_completion_receipts(live_root)
    replacement_receipt = receipts.get(TASK_CIDS[REPLACEMENT_TASK_ID])
    if not isinstance(replacement_receipt, Mapping):
        raise MigrationError("FVT-101 durable completion receipt is missing")
    if (
        replacement_receipt.get("task_id") != REPLACEMENT_TASK_ID
        or replacement_receipt.get("status") != "succeeded"
    ):
        raise MigrationError("FVT-101 durable receipt is not successful")
    for task_id in (SOURCE_TASK_ID, *DIRECT_DEPENDENT_TASK_IDS):
        if TASK_CIDS[task_id] in receipts:
            raise MigrationError(
                f"{task_id} already has a durable completion receipt"
            )
    return {
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "manifest_generated_at": str(manifest.get("generated_at") or ""),
        "scheduler_state": "stopped",
        "active_worker_count": 0,
        "fvt101_completion_receipt": dict(replacement_receipt),
        "coordination": _coordination_evidence(coordination_path),
    }


def _snapshot(
    repo_root: Path,
    migration_root: Path,
    relatives: Sequence[Path] = SNAPSHOT_RELATIVES,
) -> list[dict[str, Any]]:
    backup_root = migration_root / "backup"
    records: list[dict[str, Any]] = []
    for relative in relatives:
        source = repo_root / relative
        record: dict[str, Any] = {
            "path": relative.as_posix(),
            "existed": source.exists(),
        }
        if source.exists():
            if source.is_symlink() or not source.is_file():
                raise MigrationError(f"snapshot target is not a regular file: {relative}")
            destination = backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            record.update(
                {
                    "backup_path": str(destination),
                    "size": source.stat().st_size,
                    "sha256": _sha256_path(source),
                }
            )
        records.append(record)
    return records


def _restore_snapshot(repo_root: Path, snapshot: Sequence[Mapping[str, Any]]) -> None:
    for record in snapshot:
        relative = Path(str(record.get("path") or ""))
        target = repo_root / relative
        if record.get("existed") is True:
            backup = Path(str(record.get("backup_path") or ""))
            if not backup.is_file():
                raise MigrationError(f"snapshot backup is missing: {backup}")
            if _sha256_path(backup) != str(record.get("sha256") or ""):
                raise MigrationError(f"snapshot backup digest mismatch: {backup}")
            _atomic_write_bytes(target, backup.read_bytes())
        else:
            target.unlink(missing_ok=True)


def _artifact_hashes(repo_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in SNAPSHOT_RELATIVES[:-1]:
        path = repo_root / relative
        artifact: dict[str, Any] = {"exists": path.exists()}
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise MigrationError(
                    f"output artifact is not a regular file: {relative}"
                )
            artifact.update(
                {
                    "sha256": _sha256_path(path),
                    "size": path.stat().st_size,
                }
            )
        result[relative.as_posix()] = artifact
    return result


def _git_identity(repo_root: Path) -> dict[str, str]:
    def output(*arguments: str, cwd: Path = repo_root) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=cwd, text=True, capture_output=True
        )
        if result.returncode != 0:
            raise MigrationError(result.stderr.strip() or "git command failed")
        return result.stdout.strip()

    return {
        "root_head": output("rev-parse", "HEAD"),
        "root_branch": output("branch", "--show-current"),
        "ipfs_datasets_py_head": output(
            "rev-parse", "HEAD", cwd=repo_root / "ipfs_datasets_py"
        ),
    }


def _validate_duckdb_parity(
    index_path: Path,
    tasks: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - production dependency
        raise MigrationError("DuckDB is required for projection validation") from exc
    connection = duckdb.connect(str(index_path.with_suffix(".duckdb")), read_only=True)
    try:
        rows = connection.execute(
            "SELECT task_id, payload_json FROM bundle_tasks "
            "WHERE task_id IN ('FVT-086','FVT-088','FVT-092','FVT-099','FVT-101')"
        ).fetchall()
        database_tasks = {
            str(task_id): json.loads(payload_json)
            for task_id, payload_json in rows
        }
        if set(database_tasks) != set(TASK_CIDS):
            raise MigrationError("DuckDB relevant-task projection is incomplete")
        fields = (
            "canonical_task_cid",
            "canonical_task_key",
            "status",
            "is_schedulable",
            "review_only",
            "completion_authority",
            "depends_on",
            "dependency_task_ids",
            "dependency_task_cids",
            "work_contract_id",
            "task_work_contract_id",
        )
        for task_id in TASK_CIDS:
            expected = tasks[task_id]
            observed = database_tasks[task_id]
            for field in fields:
                if _canonical_json(expected.get(field)) != _canonical_json(
                    observed.get(field)
                ):
                    raise MigrationError(
                        f"JSON/DuckDB parity failed for {task_id}.{field}"
                    )
        stale_dependencies = connection.execute(
            "SELECT task_id, dependency_kind, dependency_id "
            "FROM bundle_task_dependencies "
            "WHERE task_id IN ('FVT-088','FVT-092','FVT-099') "
            "AND dependency_id = 'FVT-086'"
        ).fetchall()
        if stale_dependencies:
            raise MigrationError("DuckDB retains executable FVT-086 dependencies")
        for task_id in DIRECT_DEPENDENT_TASK_IDS:
            count = connection.execute(
                "SELECT count(*) FROM bundle_task_dependencies "
                "WHERE task_id = ? AND dependency_id = 'FVT-101'",
                [task_id],
            ).fetchone()[0]
            if int(count) < 2:
                raise MigrationError(
                    f"DuckDB did not project FVT-101 for {task_id}"
                )
        open_source = connection.execute(
            "SELECT count(*) FROM open_bundle_tasks WHERE task_id = 'FVT-086'"
        ).fetchone()[0]
        if int(open_source) != 0:
            raise MigrationError("FVT-086 remains in DuckDB open task view")
    finally:
        connection.close()


def validate_final_projection(
    repo_root: Path,
    *,
    before: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate every migrated projection and execution overlay."""

    board_text = (repo_root / BOARD_RELATIVE).read_text(encoding="utf-8")
    source_block = task_block(board_text, SOURCE_TASK_ID)
    required_source_lines = (
        "- Status: blocked",
        "- Completion: superseded:FVT-101",
        "- Is schedulable: false",
        "- Review only: true",
        "- Completion authority: none",
        "- Superseded by: FVT-101",
        "- Supersession completion authority: none",
        "- Historical task: true",
    )
    for line in required_source_lines:
        if source_block.count(line) != 1:
            raise MigrationError(f"source retirement line is missing: {line}")
    for task_id, shard_relative in SHARD_RELATIVES.items():
        shard_text = (repo_root / shard_relative).read_text(encoding="utf-8")
        if task_block(shard_text, task_id).strip() != task_block(
            board_text, task_id
        ).strip():
            raise MigrationError(f"aggregate/shard parity failed for {task_id}")
    for task_id in DIRECT_DEPENDENT_TASK_IDS:
        block = task_block(board_text, task_id)
        depends = re.search(r"^- Depends on: (.+)$", block, re.MULTILINE)
        if depends is None:
            raise MigrationError(f"{task_id} lost its dependency field")
        values = [item.strip() for item in depends.group(1).split(",")]
        if SOURCE_TASK_ID in values or values.count(REPLACEMENT_TASK_ID) != 1:
            raise MigrationError(f"Markdown dependency migration failed for {task_id}")

    index_path = repo_root / INDEX_RELATIVE
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    tasks = _bundle_tasks(index_payload)
    source = tasks[SOURCE_TASK_ID]
    if (
        source.get("status") != "blocked"
        or source.get("is_schedulable") is not False
        or source.get("review_only") is not True
        or source.get("completion_authority") != "none"
        or source.get("superseded_by") != REPLACEMENT_TASK_ID
    ):
        raise MigrationError("indexed FVT-086 retirement is incomplete")

    contract_rotations: dict[str, dict[str, Any]] = {}
    for task_id, expected_cid in TASK_CIDS.items():
        task = tasks.get(task_id)
        if task is None or task.get("canonical_task_cid") != expected_cid:
            raise MigrationError(f"canonical identity changed for {task_id}")
        rehydrate_task_work_contract_projection(task)
        if task_id not in DIRECT_DEPENDENT_TASK_IDS:
            continue
        for field in (
            "depends_on",
            "dependency_task_ids",
            "dependency_task_cids",
        ):
            values = [str(item) for item in task.get(field) or []]
            if SOURCE_TASK_ID in values or values.count(REPLACEMENT_TASK_ID) != 1:
                raise MigrationError(
                    f"indexed dependency migration failed for {task_id}.{field}"
                )
        current = {
            "canonical_task_cid": task["canonical_task_cid"],
            "canonical_task_key": task.get("canonical_task_key", ""),
            "depends_on": list(task.get("depends_on") or []),
            "work_contract_id": str(task.get("work_contract_id") or ""),
            "task_work_contract_id": str(
                task.get("task_work_contract_id") or ""
            ),
        }
        if before is not None:
            previous = before[task_id]
            if current["canonical_task_cid"] != previous["canonical_task_cid"]:
                raise MigrationError(f"canonical CID rotated for {task_id}")
            if current["canonical_task_key"] != previous["canonical_task_key"]:
                raise MigrationError(f"canonical key rotated for {task_id}")
            if current["work_contract_id"] == previous["work_contract_id"]:
                raise MigrationError(f"work contract did not rotate for {task_id}")
            if current["task_work_contract_id"] == previous[
                "task_work_contract_id"
            ]:
                raise MigrationError(
                    f"task work contract did not rotate for {task_id}"
                )
            current["previous_work_contract_id"] = previous[
                "work_contract_id"
            ]
            current["previous_task_work_contract_id"] = previous[
                "task_work_contract_id"
            ]
            current["previous_depends_on"] = previous["depends_on"]
        contract_rotations[task_id] = current

    vector_payload = json.loads(
        (repo_root / VECTOR_RELATIVE).read_text(encoding="utf-8")
    )
    if vector_payload.get("task_header_prefix") != "## FVT-":
        raise MigrationError("todo vector task-header contract drifted")
    dataset_artifact = vector_payload.get("dataset_artifact")
    if not isinstance(dataset_artifact, Mapping):
        raise MigrationError("todo vector dataset artifact is missing")
    if (
        dataset_artifact.get("dataset_id") != VECTOR_DATASET_ID
        or int(dataset_artifact.get("row_count") or 0) != 101
    ):
        raise MigrationError("todo vector dataset artifact is inconsistent")
    vector_tasks = {
        str(item.get("task_id") or ""): item
        for item in vector_payload.get("records") or []
        if isinstance(item, Mapping)
    }
    dataset_jsonl_path = repo_root / VECTOR_DATASET_JSONL_RELATIVE
    dataset_manifest_path = repo_root / VECTOR_DATASET_MANIFEST_RELATIVE
    for field, expected_path in (
        ("jsonl_path", dataset_jsonl_path),
        ("manifest_path", dataset_manifest_path),
    ):
        observed_path = Path(str(dataset_artifact.get(field) or ""))
        if observed_path.resolve() != expected_path.resolve():
            raise MigrationError(
                f"todo vector dataset {field} escapes its managed path"
            )
    dataset_tasks: dict[str, Mapping[str, Any]] = {}
    with dataset_jsonl_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or "")
            if not task_id or task_id in dataset_tasks:
                raise MigrationError(
                    "todo vector dataset has an invalid or duplicate task "
                    f"at line {line_number}"
                )
            dataset_tasks[task_id] = row
    if set(dataset_tasks) != set(vector_tasks):
        raise MigrationError("todo vector dataset task set drifted")
    dataset_fields = (
        "status",
        "canonical_task_cid",
        "canonical_task_key",
        "dependency_task_cids",
        "work_contract_id",
        "task_work_contract_id",
    )
    for task_id, vector_record in vector_tasks.items():
        dataset_record = dataset_tasks[task_id]
        for field in dataset_fields:
            if _canonical_json(dataset_record.get(field)) != _canonical_json(
                vector_record.get(field)
            ):
                raise MigrationError(
                    f"todo vector dataset parity failed for {task_id}.{field}"
                )
    dataset_manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    if (
        dataset_manifest.get("dataset_id") != VECTOR_DATASET_ID
        or int(dataset_manifest.get("row_count") or 0) != len(vector_tasks)
    ):
        raise MigrationError("todo vector dataset manifest drifted")
    if vector_tasks.get(SOURCE_TASK_ID, {}).get("status") != "blocked":
        raise MigrationError("todo vector retains open FVT-086 status")
    for task_id in DIRECT_DEPENDENT_TASK_IDS:
        record = vector_tasks.get(task_id)
        if not isinstance(record, Mapping):
            raise MigrationError(f"todo vector is missing {task_id}")
        dependencies = [
            str(item) for item in record.get("dependency_task_cids") or []
        ]
        if SOURCE_TASK_ID in dependencies or dependencies.count(
            REPLACEMENT_TASK_ID
        ) != 1:
            raise MigrationError(f"todo vector dependency failed for {task_id}")
        if record.get("work_contract_id") != tasks[task_id].get(
            "work_contract_id"
        ):
            raise MigrationError(f"vector/index contract parity failed for {task_id}")

    graph_edges = index_payload.get("task_dependency_graph", {}).get(
        "edges", []
    )
    for task_id in DIRECT_DEPENDENT_TASK_IDS:
        target_cid = TASK_CIDS[task_id]
        if any(
            edge.get("source_task_cid") == TASK_CIDS[SOURCE_TASK_ID]
            and edge.get("target_task_cid") == target_cid
            for edge in graph_edges
            if isinstance(edge, Mapping)
        ):
            raise MigrationError(f"dependency graph retains FVT-086 -> {task_id}")
        if not any(
            edge.get("source_task_cid") == TASK_CIDS[REPLACEMENT_TASK_ID]
            and edge.get("target_task_cid") == target_cid
            for edge in graph_edges
            if isinstance(edge, Mapping)
        ):
            raise MigrationError(f"dependency graph lacks FVT-101 -> {task_id}")

    _validate_duckdb_parity(index_path, tasks)
    live_root = repo_root / LIVE_RELATIVE
    receipts = bundle_member_completion_receipts(live_root)
    payloads = build_bundle_task_payloads(
        index_path,
        merge_receipts=receipts,
    )
    task_runtime: dict[str, Mapping[str, Any]] = {}
    execution_slice_ids: set[str] = set()
    for payload in payloads:
        execution_slice_ids.update(
            str(item) for item in payload.get("execution_slice_task_ids") or []
        )
        for task in payload.get("tasks") or []:
            if isinstance(task, Mapping) and task.get("task_id"):
                task_runtime[str(task["task_id"])] = task
    if SOURCE_TASK_ID in execution_slice_ids:
        raise MigrationError("FVT-086 remains in an execution slice")
    for task_id in DIRECT_DEPENDENT_TASK_IDS:
        blockers = set(task_runtime[task_id].get("blocking_task_cids") or [])
        if TASK_CIDS[SOURCE_TASK_ID] in blockers:
            raise MigrationError(f"FVT-086 still blocks {task_id}")
        if TASK_CIDS[REPLACEMENT_TASK_ID] in blockers:
            raise MigrationError(
                f"successful FVT-101 receipt did not unblock {task_id}"
            )
    if task_runtime["FVT-092"].get("claimable") is not True:
        raise MigrationError("FVT-092 is not claimable after FVT-101 completion")
    return {
        "aggregate_shard_parity": True,
        "json_duckdb_parity": True,
        "vector_index_parity": True,
        "canonical_task_identities_preserved": True,
        "work_contracts_rotated": contract_rotations,
        "fvt086_absent_from_execution_slices": True,
        "fvt092_claimable": True,
        "runtime_execution_slice_task_ids": sorted(execution_slice_ids),
    }


def _writer_pass(repo_root: Path) -> None:
    board_path = repo_root / BOARD_RELATIVE
    bundle_dir = repo_root / BUNDLE_DIR_RELATIVE
    index_path = repo_root / INDEX_RELATIVE
    vector_path = repo_root / VECTOR_RELATIVE
    objective_path = repo_root / OBJECTIVE_RELATIVE
    write_todo_vector_index(
        repo_root=repo_root,
        todo_path=board_path,
        index_path=vector_path,
        task_header_prefix="## FVT-",
        objective_path=objective_path,
        bundle_index_path=index_path,
    )
    write_bundle_shards(
        bundle_dir=bundle_dir,
        repo_root=repo_root,
        todo_path=board_path,
        records=(),
    )
    # The vector writer is the final projection owner for admitted work
    # contracts and the compact query references used by the supervisor.
    write_todo_vector_index(
        repo_root=repo_root,
        todo_path=board_path,
        index_path=vector_path,
        task_header_prefix="## FVT-",
        objective_path=objective_path,
        bundle_index_path=index_path,
        dataset_dir=repo_root / DATASET_DIR_RELATIVE,
        dataset_id=VECTOR_DATASET_ID,
        persist_dataset=True,
    )


def _journal_phase(
    journal_path: Path,
    journal: dict[str, Any],
    phase: str,
    **extra: Any,
) -> None:
    journal.update(extra)
    journal["phase"] = phase
    journal["updated_at"] = _utc_now()
    _atomic_write_json(journal_path, journal)


def apply_migration(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    preflight_evidence = preflight(repo_root)
    migration_id = (
        "fvt086-to-fvt101-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    migrations_root = repo_root / LIVE_RELATIVE / "migrations" / "fvt086-to-fvt101"
    latest_path = migrations_root / "latest.json"
    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        latest_journal_path = Path(str(latest.get("journal_path") or ""))
        if latest_journal_path.is_file():
            latest_journal = json.loads(
                latest_journal_path.read_text(encoding="utf-8")
            )
            if latest_journal.get("phase") not in TERMINAL_JOURNAL_PHASES:
                raise MigrationError(
                    "an incomplete migration journal requires --recover"
                )
    migration_root = migrations_root / migration_id
    migration_root.mkdir(parents=True, exist_ok=False)
    journal_path = migration_root / "journal.json"
    snapshot = _snapshot(repo_root, migration_root)
    journal: dict[str, Any] = {
        "schema": JOURNAL_SCHEMA,
        "migration_id": migration_id,
        "repo_root": str(repo_root),
        "created_at": _utc_now(),
        "phase": "snapshotted",
        "snapshot": snapshot,
        "preflight": preflight_evidence,
    }
    _atomic_write_json(journal_path, journal)
    _atomic_write_json(
        latest_path,
        {"migration_id": migration_id, "journal_path": str(journal_path)},
    )

    lock_path = checkout_mutation_lock_path(
        repo_root, lock_name=PROTECTED_PATH_MAINTENANCE_LOCK_NAME
    )
    lease = CheckoutMaintenanceLease(
        lock_path=lock_path,
        metadata={
            "kind": "implementation-protected-maintenance",
            "lease_role": "fvt086_to_fvt101_migration",
            "pid": os.getpid(),
            "repo_root": str(repo_root),
            "migration_id": migration_id,
            "started_at": _utc_now(),
        },
        max_hold_seconds=300.0,
    )
    try:
        with lease.exclusive_section(
            owner_is_active=_maintenance_owner_is_active
        ) as lease_timing:
            # Re-prove quiescence while the shared protected-path fence is held.
            preflight(repo_root)
            board_path = repo_root / BOARD_RELATIVE
            board_text = board_path.read_text(encoding="utf-8")
            shard_texts = {
                task_id: (repo_root / relative).read_text(encoding="utf-8")
                for task_id, relative in SHARD_RELATIVES.items()
            }
            rewritten_board, rewritten_shards = rewrite_markdown_documents(
                board_text, shard_texts
            )
            _atomic_write_bytes(board_path, rewritten_board.encode("utf-8"))
            for task_id, rendered in rewritten_shards.items():
                _atomic_write_bytes(
                    repo_root / SHARD_RELATIVES[task_id],
                    rendered.encode("utf-8"),
                )
            _journal_phase(journal_path, journal, "markdown_published")

            index_path = repo_root / INDEX_RELATIVE
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            rewritten_index, before = rewrite_bundle_index(index_payload)
            write_bundle_index_artifact(index_path, rewritten_index)
            _journal_phase(
                journal_path,
                journal,
                "index_seed_published",
                before=before,
            )
            _writer_pass(repo_root)
            _journal_phase(journal_path, journal, "projections_regenerated")
            validation = validate_final_projection(repo_root, before=before)
            _journal_phase(
                journal_path,
                journal,
                "validated",
                validation=validation,
            )
            output_hashes = _artifact_hashes(repo_root)
            receipt_preflight = _portable_preflight_evidence(
                repo_root, preflight_evidence
            )
            receipt_snapshot = _portable_snapshot_evidence(
                repo_root, snapshot
            )
            receipt = {
                "schema": MIGRATION_SCHEMA,
                "migration_id": migration_id,
                "applied_at": _utc_now(),
                "source_task": {
                    "task_id": SOURCE_TASK_ID,
                    "canonical_task_cid": TASK_CIDS[SOURCE_TASK_ID],
                    "disposition": "blocked_historical",
                    "superseded_by": REPLACEMENT_TASK_ID,
                    "completion_authority": "none",
                },
                "replacement_task": {
                    "task_id": REPLACEMENT_TASK_ID,
                    "canonical_task_cid": TASK_CIDS[REPLACEMENT_TASK_ID],
                    "completion_receipt": receipt_preflight[
                        "fvt101_completion_receipt"
                    ],
                    "coordination_receipt_cid": preflight_evidence[
                        "coordination"
                    ]["fvt101_coordination_receipt_cid"],
                },
                "direct_dependents": validation["work_contracts_rotated"],
                "preflight": receipt_preflight,
                "validation": validation,
                "input_snapshot": receipt_snapshot,
                "output_artifacts": output_hashes,
                "journal_path": str(journal_path.relative_to(repo_root)),
                "repository": _git_identity(repo_root),
                "authority_limits": [
                    "FVT-086 supersession is not completion authority",
                    "historical discovery receipts remain immutable provenance",
                    "FVT-G219 still requires a supported production SecPAL artifact, runtime, and arbitrary-policy query interface",
                    "FVT-G232 still requires signed legal, IP, security, and deployment approval",
                ],
            }
            _atomic_write_json(repo_root / RECEIPT_RELATIVE, receipt)
            _journal_phase(
                journal_path,
                journal,
                "completed",
                receipt_path=RECEIPT_RELATIVE.as_posix(),
                receipt_sha256=_sha256_path(repo_root / RECEIPT_RELATIVE),
                lease_timing=lease_timing,
            )
            return receipt
    except BaseException as exc:
        try:
            _restore_snapshot(repo_root, snapshot)
        except BaseException as rollback_exc:
            _journal_phase(
                journal_path,
                journal,
                "rollback_failed",
                error=f"{type(exc).__name__}: {exc}",
                rollback_error=(
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                ),
            )
            raise MigrationError(
                "migration failed and automatic rollback failed; use the "
                f"journal at {journal_path}"
            ) from rollback_exc
        _journal_phase(
            journal_path,
            journal,
            "rolled_back",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def recover_migration(repo_root: Path, journal_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    journal_path = journal_path.resolve(strict=True)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if journal.get("schema") != JOURNAL_SCHEMA:
        raise MigrationError("recovery journal schema is not recognized")
    if Path(str(journal.get("repo_root") or "")).resolve() != repo_root:
        raise MigrationError("recovery journal belongs to another repository")
    if journal.get("phase") in {"completed", "recovered"}:
        raise MigrationError("completed migration must not be recovered")
    preflight(repo_root)
    lock_path = checkout_mutation_lock_path(
        repo_root, lock_name=PROTECTED_PATH_MAINTENANCE_LOCK_NAME
    )
    lease = CheckoutMaintenanceLease(
        lock_path=lock_path,
        metadata={
            "kind": "implementation-protected-maintenance",
            "lease_role": "fvt086_to_fvt101_recovery",
            "pid": os.getpid(),
            "repo_root": str(repo_root),
            "started_at": _utc_now(),
        },
        max_hold_seconds=120.0,
    )
    with lease.exclusive_section(owner_is_active=_maintenance_owner_is_active):
        _restore_snapshot(repo_root, journal.get("snapshot") or [])
        _journal_phase(journal_path, journal, "recovered")
    return {
        "recovered": True,
        "journal_path": str(journal_path),
        "migration_id": str(journal.get("migration_id") or ""),
    }


def _previous_dependents_from_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_dependents = receipt.get("direct_dependents")
    if not isinstance(raw_dependents, Mapping) or set(raw_dependents) != set(
        DIRECT_DEPENDENT_TASK_IDS
    ):
        raise MigrationError("migration receipt dependent set is invalid")
    before: dict[str, dict[str, Any]] = {}
    for task_id in DIRECT_DEPENDENT_TASK_IDS:
        values = raw_dependents.get(task_id)
        if not isinstance(values, Mapping):
            raise MigrationError(f"migration receipt is missing {task_id}")
        before[task_id] = {
            "canonical_task_cid": str(
                values.get("canonical_task_cid") or ""
            ),
            "canonical_task_key": str(
                values.get("canonical_task_key") or ""
            ),
            "depends_on": list(values.get("previous_depends_on") or []),
            "work_contract_id": str(
                values.get("previous_work_contract_id") or ""
            ),
            "task_work_contract_id": str(
                values.get("previous_task_work_contract_id") or ""
            ),
        }
        if any(
            not before[task_id][field]
            for field in (
                "canonical_task_cid",
                "canonical_task_key",
                "depends_on",
                "work_contract_id",
                "task_work_contract_id",
            )
        ):
            raise MigrationError(
                f"migration receipt has incomplete prior evidence for {task_id}"
            )
    return before


def refresh_migrated_projections(repo_root: Path) -> dict[str, Any]:
    """Regenerate a completed migration with its exact writer contracts."""

    repo_root = repo_root.resolve(strict=True)
    preflight_evidence = preflight(repo_root)
    receipt_path = repo_root / RECEIPT_RELATIVE
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != MIGRATION_SCHEMA:
        raise MigrationError("migration receipt schema is not recognized")
    before = _previous_dependents_from_receipt(receipt)
    source = task_block(
        (repo_root / BOARD_RELATIVE).read_text(encoding="utf-8"),
        SOURCE_TASK_ID,
    )
    if "- Completion: superseded:FVT-101" not in source:
        raise MigrationError("projection refresh requires a migrated board")

    journal_relative = Path(str(receipt.get("journal_path") or ""))
    if journal_relative.is_absolute():
        raise MigrationError("migration receipt journal path is not portable")
    journal_path = (repo_root / journal_relative).resolve()
    try:
        journal_path.relative_to(repo_root)
    except ValueError as exc:
        raise MigrationError("migration journal escapes the repository") from exc
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if (
        journal.get("schema") != JOURNAL_SCHEMA
        or journal.get("phase") != "completed"
        or journal.get("migration_id") != receipt.get("migration_id")
    ):
        raise MigrationError("completed migration journal is inconsistent")

    refresh_id = (
        "fvt086-to-fvt101-projection-refresh-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    refresh_root = journal_path.parent / refresh_id
    refresh_root.mkdir(parents=True, exist_ok=False)
    refresh_journal_path = refresh_root / "journal.json"
    refresh_snapshot = _snapshot(
        repo_root,
        refresh_root,
        (*SNAPSHOT_RELATIVES, journal_relative),
    )
    refresh_journal: dict[str, Any] = {
        "schema": JOURNAL_SCHEMA,
        "migration_id": refresh_id,
        "parent_migration_id": str(receipt.get("migration_id") or ""),
        "repo_root": str(repo_root),
        "created_at": _utc_now(),
        "phase": "snapshotted",
        "snapshot": refresh_snapshot,
        "preflight": preflight_evidence,
    }
    _atomic_write_json(refresh_journal_path, refresh_journal)

    lock_path = checkout_mutation_lock_path(
        repo_root, lock_name=PROTECTED_PATH_MAINTENANCE_LOCK_NAME
    )
    lease = CheckoutMaintenanceLease(
        lock_path=lock_path,
        metadata={
            "kind": "implementation-protected-maintenance",
            "lease_role": "fvt086_to_fvt101_projection_refresh",
            "pid": os.getpid(),
            "repo_root": str(repo_root),
            "migration_id": str(receipt.get("migration_id") or ""),
            "started_at": _utc_now(),
        },
        max_hold_seconds=300.0,
    )
    try:
        with lease.exclusive_section(
            owner_is_active=_maintenance_owner_is_active
        ) as lease_timing:
            preflight(repo_root)
            _writer_pass(repo_root)
            validation = validate_final_projection(repo_root, before=before)
            portable_preflight = _portable_preflight_evidence(
                repo_root, preflight_evidence
            )
            output_artifacts = _artifact_hashes(repo_root)
            receipt["preflight"] = portable_preflight
            replacement = receipt.get("replacement_task")
            if not isinstance(replacement, dict):
                raise MigrationError(
                    "migration receipt replacement task is invalid"
                )
            replacement["completion_receipt"] = portable_preflight[
                "fvt101_completion_receipt"
            ]
            receipt["validation"] = validation
            receipt["output_artifacts"] = output_artifacts
            receipt["projection_refreshed_at"] = _utc_now()
            receipt["projection_refresh"] = {
                "refresh_id": refresh_id,
                "journal_path": str(
                    refresh_journal_path.relative_to(repo_root)
                ),
                "input_snapshot": _portable_snapshot_evidence(
                    repo_root, refresh_snapshot
                ),
                "output_artifacts": output_artifacts,
            }
            _atomic_write_json(receipt_path, receipt)
            receipt_sha256 = _sha256_path(receipt_path)
            _journal_phase(
                journal_path,
                journal,
                "completed",
                validation=validation,
                receipt_sha256=receipt_sha256,
                projection_refresh_lease_timing=lease_timing,
            )
            _journal_phase(
                refresh_journal_path,
                refresh_journal,
                "completed",
                validation=validation,
                receipt_path=RECEIPT_RELATIVE.as_posix(),
                receipt_sha256=receipt_sha256,
                lease_timing=lease_timing,
            )
        return {
            "state": "refreshed",
            "migration_id": str(receipt.get("migration_id") or ""),
            "refresh_id": refresh_id,
            "receipt_path": RECEIPT_RELATIVE.as_posix(),
            "receipt_sha256": receipt_sha256,
            "validation": validation,
        }
    except BaseException as exc:
        try:
            _restore_snapshot(repo_root, refresh_snapshot)
        except BaseException as rollback_exc:
            _journal_phase(
                refresh_journal_path,
                refresh_journal,
                "rollback_failed",
                error=f"{type(exc).__name__}: {exc}",
                rollback_error=(
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                ),
            )
            raise MigrationError(
                "projection refresh failed and rollback failed; use the "
                f"journal at {refresh_journal_path}"
            ) from rollback_exc
        _journal_phase(
            refresh_journal_path,
            refresh_journal,
            "rolled_back",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def check_migration(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    preflight_evidence = preflight(repo_root)
    source = task_block(
        (repo_root / BOARD_RELATIVE).read_text(encoding="utf-8"),
        SOURCE_TASK_ID,
    )
    if "- Status: blocked" in source:
        receipt_path = repo_root / RECEIPT_RELATIVE
        before: Mapping[str, Mapping[str, Any]] | None = None
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            before = _previous_dependents_from_receipt(receipt)
        validation = validate_final_projection(repo_root, before=before)
        return {
            "state": "migrated",
            "preflight": preflight_evidence,
            "validation": validation,
        }
    board_text = (repo_root / BOARD_RELATIVE).read_text(encoding="utf-8")
    shard_texts = {
        task_id: (repo_root / relative).read_text(encoding="utf-8")
        for task_id, relative in SHARD_RELATIVES.items()
    }
    rewritten_board, _ = rewrite_markdown_documents(board_text, shard_texts)
    if rewritten_board == board_text:
        raise MigrationError("migration check produced no reviewed change")
    index_payload = json.loads(
        (repo_root / INDEX_RELATIVE).read_text(encoding="utf-8")
    )
    # Operate on a detached JSON copy during a check.
    rewrite_bundle_index(json.loads(json.dumps(index_payload)))
    return {
        "state": "ready_to_apply",
        "preflight": preflight_evidence,
        "direct_dependents": list(DIRECT_DEPENDENT_TASK_IDS),
        "source_task": SOURCE_TASK_ID,
        "replacement_task": REPLACEMENT_TASK_ID,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=DEFAULT_REPO_ROOT
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--recover", type=Path)
    action.add_argument("--refresh-projections", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.recover is not None:
            result = recover_migration(args.repo_root, args.recover)
        elif args.refresh_projections:
            result = refresh_migrated_projections(args.repo_root)
        elif args.apply:
            result = apply_migration(args.repo_root)
        else:
            result = check_migration(args.repo_root)
    except (MigrationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
