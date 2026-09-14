"""SPAR three-lane runtime and merge-queue settlement, never goal acceptance.

The producer holds the configured merge queue and re-reads lane sidecars. A WAL,
live pid, active claim, or missing lane remains a typed non-admission. Observe
and hold paths never checkpoint or delete sidecar state. Owner closeout may
CHECKPOINT leftover WALs only on lanes with no live pid.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..merge.merge_queue import (
    MergeQueueIntegrityError,
    hold_merge_queue_settlement,
    read_merge_queue_settlement,
)
from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.duckdb_state import connect_duckdb_with_policy

SCHEMA = "ipfs_accelerate_py/agent-supervisor/spar-runtime-settlement@1"
CONFIG_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "semantic-preserving-autonomous-remodularization.scheduler_config@1"
)
PROGRAM = "semantic-preserving-autonomous-remodularization-v1"
BOARD = "semantic-preserving-autonomous-remodularization-v1"
SCHEDULER_RELATIVE = (
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json"
)
MISSING = "runtime_lane_and_merge_queue_settlement_receipt_required"
WAL_OUTSTANDING = "runtime_lane_outstanding_wal"
LIVE_PROCESS = "runtime_lane_process_live"
LANE_MISSING = "runtime_lane_artifact_missing"
CHECKPOINT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/spar-stopped-sidecar-checkpoint@1"
)
EXPECTED_LANES = (
    {"index": 0, "name": "spar-lane-0", "strict_shard_remainder": 0},
    {"index": 1, "name": "spar-lane-1", "strict_shard_remainder": 1},
    {"index": 2, "name": "spar-lane-2", "strict_shard_remainder": 2},
)
_TERMINAL_ATTEMPTS = frozenset(
    {"succeeded", "completed", "failed", "cancelled", "aborted", "released"}
)
_LIVE_PIDS = ("supervisor.pid", "managed_daemon.pid")


def _deferred(reason: str = MISSING, **extra: Any) -> dict[str, Any]:
    result = {
        "schema": SCHEMA,
        "admitted": False,
        "settled": False,
        "authority": "spar_runtime_settlement",
        "completion_authority": False,
        "semantic_acceptance_authority": False,
        "reason": reason,
    }
    result.update(extra)
    return result


def _lane_prefix(index: int) -> str:
    return f"spar_lane_{index}"


def lane_paths(state_path: Path, index: int) -> dict[str, Path]:
    directory = state_path / f"lane-{index}"
    prefix = _lane_prefix(index)
    coordination = directory / f"{prefix}_database_coordination.duckdb"
    execution = directory / f"{prefix}_database_execution.duckdb"
    return {
        "directory": directory,
        "coordination": coordination,
        "execution": execution,
        "coordination_wal": Path(str(coordination) + ".wal"),
        "execution_wal": Path(str(execution) + ".wal"),
        "coordination_lock": coordination.with_name(f".{coordination.name}.lock"),
        "execution_lock": execution.with_name(f".{execution.name}.lock"),
        "execution_writer_lock": execution.with_name(f".{execution.name}.writer.lock"),
        "supervisor_pid": directory / f"{prefix}_supervisor.pid",
        "daemon_pid": directory / f"{prefix}_managed_daemon.pid",
    }


def read_spar_runtime_profile(repository_root: str | Path) -> dict[str, Any]:
    """Extract the closed three-lane SPAR settlement profile from scheduler JSON."""
    root = Path(repository_root)
    path = root / SCHEDULER_RELATIVE
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("SPAR scheduler config is unavailable")
    config = json.loads(path.read_text())
    lanes = config.get("lanes")
    runtime_paths = config.get("runtime_paths") or {}
    if (
        not isinstance(config, dict)
        or config.get("schema") != CONFIG_SCHEMA
        or config.get("program_identifier") != PROGRAM
        or config.get("board_namespace") != BOARD
        or config.get("max_lanes") != 3
        or not isinstance(lanes, list)
        or len(lanes) != 3
        or not isinstance(runtime_paths, dict)
        or not runtime_paths.get("state")
        or not runtime_paths.get("merge_queue")
        or not config.get("merge_target_branch")
    ):
        raise ValueError("SPAR runtime settlement profile is not the closed v1 contract")
    observed = []
    for expected, row in zip(EXPECTED_LANES, lanes):
        if not isinstance(row, dict) or any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("SPAR lane population differs from the sealed three-lane contract")
        observed.append(dict(expected))
    return {
        "schema": SCHEMA,
        "board_namespace": BOARD,
        "program_identifier": PROGRAM,
        "merge_target_branch": str(config["merge_target_branch"]),
        "store_id": str((config.get("database_program") or {}).get("store_id") or ""),
        "state_path": str((root / runtime_paths["state"]).resolve()),
        "merge_queue_path": str((root / runtime_paths["merge_queue"]).resolve()),
        "lanes": observed,
    }


def _live_pid(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        text = path.read_text().strip()
        pid = int(text.splitlines()[0])
        return pid > 1 and os.path.exists(f"/proc/{pid}")
    except (OSError, ValueError, IndexError):
        return True


def checkpoint_stopped_lane_sidecars(repository_root: str | Path) -> dict[str, Any]:
    """CHECKPOINT leftover WALs on lanes with no live pid. Never unlink WAL files.

    Observe/hold remain read-only. Live supervisor or daemon pids are skipped so
    a drained SPAR closeout can absorb SIGTERM leftover WALs without deleting
    sidecar state.
    """
    try:
        profile = read_spar_runtime_profile(repository_root)
    except Exception as exc:  # noqa: BLE001 - closeout must fail closed
        return {
            "schema": CHECKPOINT_SCHEMA,
            "attempted": False,
            "completion_authority": False,
            "semantic_acceptance_authority": False,
            "reason": MISSING,
            "error_class": type(exc).__name__,
            "error": str(exc)[:256],
            "remaining_wal": [],
            "settled": False,
        }
    import duckdb

    lanes: list[dict[str, Any]] = []
    remaining: list[str] = []
    state_path = Path(profile["state_path"])
    for spec in profile["lanes"]:
        index = spec["index"]
        paths = lane_paths(state_path, index)
        live = _live_pid(paths["supervisor_pid"]) or _live_pid(paths["daemon_pid"])
        row: dict[str, Any] = {
            "index": index,
            "live": live,
            "checkpointed": [],
            "skipped": [],
            "errors": [],
        }
        for kind in ("coordination", "execution"):
            wal = paths[f"{kind}_wal"]
            database = paths[kind]
            if not wal.exists():
                continue
            label = f"lane-{index}-{kind}"
            if live:
                row["skipped"].append(kind)
                remaining.append(label)
                continue
            if database.is_symlink() or not database.is_file():
                row["errors"].append({"kind": kind, "error_class": "missing_database"})
                remaining.append(label)
                continue
            try:
                connection = connect_duckdb_with_policy(
                    duckdb, database, read_only=False
                )
                try:
                    connection.execute("CHECKPOINT")
                finally:
                    connection.close()
                if wal.exists():
                    row["errors"].append(
                        {"kind": kind, "error_class": "wal_remained_after_checkpoint"}
                    )
                    remaining.append(label)
                else:
                    row["checkpointed"].append(kind)
            except Exception as exc:  # noqa: BLE001 - leftover WAL stays a blocker
                row["errors"].append(
                    {
                        "kind": kind,
                        "error_class": type(exc).__name__,
                        "error": str(exc)[:256],
                    }
                )
                remaining.append(label)
        lanes.append(row)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "attempted": True,
        "completion_authority": False,
        "semantic_acceptance_authority": False,
        "lanes": lanes,
        "remaining_wal": remaining,
        "settled": not remaining,
    }


def _count_active(connection: Any, sql: str) -> int:
    row = connection.execute(sql).fetchone()
    return int(row[0] if row else 0)


def _sidecar_active(path: Path, kind: str) -> int:
    import duckdb

    connection = connect_duckdb_with_policy(duckdb, path, read_only=True)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        if kind == "coordination":
            required = {"task_claims", "fenced_leases", "resource_claims", "maintenance_leases"}
            if not required.issubset(tables):
                raise ValueError("coordination sidecar schema is incomplete")
            return (
                _count_active(connection, "SELECT COUNT(*) FROM task_claims WHERE released_at_ms IS NULL")
                + _count_active(
                    connection,
                    "SELECT COUNT(*) FROM fenced_leases WHERE state NOT IN ('released', 'expired')",
                )
                + _count_active(
                    connection,
                    "SELECT COUNT(*) FROM resource_claims WHERE state NOT IN ('released', 'expired')",
                )
                + _count_active(
                    connection,
                    "SELECT COUNT(*) FROM maintenance_leases WHERE released_at_ms IS NULL",
                )
            )
        required = {"database_task_attempts"}
        if not required.issubset(tables):
            raise ValueError("execution sidecar schema is incomplete")
        allowed = ", ".join("'" + status + "'" for status in sorted(_TERMINAL_ATTEMPTS))
        return _count_active(
            connection,
            "SELECT COUNT(*) FROM database_task_attempts "
            f"WHERE finished_at_ms IS NULL AND status NOT IN ({allowed})",
        )
    finally:
        connection.close()


class RuntimeSettlementBlocked(ValueError):
    """Typed SPAR runtime non-admission; never completion authority."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _lane_snapshot(paths: Mapping[str, Path], index: int) -> dict[str, Any]:
    for key in ("directory", "coordination", "execution"):
        path = paths[key]
        if path.is_symlink() or not path.exists():
            raise RuntimeSettlementBlocked(
                LANE_MISSING, f"SPAR lane {index} {key} is missing"
            )
    if paths["coordination_wal"].exists() or paths["execution_wal"].exists():
        raise RuntimeSettlementBlocked(
            WAL_OUTSTANDING, f"SPAR lane {index} sidecar has an outstanding WAL"
        )
    if _live_pid(paths["supervisor_pid"]) or _live_pid(paths["daemon_pid"]):
        raise RuntimeSettlementBlocked(
            LIVE_PROCESS, f"SPAR lane {index} still has a live process"
        )
    coordination_active = _sidecar_active(paths["coordination"], "coordination")
    execution_active = _sidecar_active(paths["execution"], "execution")
    return {
        "index": index,
        "name": EXPECTED_LANES[index]["name"],
        "coordination_active": coordination_active,
        "execution_active": execution_active,
        "active_count": coordination_active + execution_active,
    }


def observe_spar_runtime_settlement(
    repository_root: str | Path,
    *,
    owner_identity: Mapping[str, Any],
    target_repository_id: str,
) -> dict[str, Any]:
    """Read-only SPAR settlement observation; a later CAS still requires hold()."""
    try:
        profile = read_spar_runtime_profile(repository_root)
        lanes = []
        active = 0
        for spec in profile["lanes"]:
            snapshot = _lane_snapshot(lane_paths(Path(profile["state_path"]), spec["index"]), spec["index"])
            lanes.append(snapshot)
            active += snapshot["active_count"]
        merge = read_merge_queue_settlement(
            profile["merge_queue_path"],
            target_repository_id=target_repository_id,
            target_branch=profile["merge_target_branch"],
            lock_timeout_seconds=0.05,
        )
        settled = active == 0 and merge.get("settled") is True
        receipt = {
            "schema": SCHEMA,
            "admitted": settled,
            "settled": settled,
            "authority": "spar_runtime_settlement",
            "completion_authority": False,
            "semantic_acceptance_authority": False,
            "owner_generation": owner_identity.get("generation"),
            "store_id": owner_identity.get("store_id") or profile["store_id"],
            "target": {
                "repository_id": target_repository_id,
                "branch": profile["merge_target_branch"],
            },
            "lanes": lanes,
            "active_count": active,
            "merge_queue": {
                "settled": merge.get("settled") is True,
                "active_count": merge.get("active_count"),
                "receipt_cid": merge.get("receipt_cid"),
            },
        }
        if not settled:
            receipt["reason"] = MISSING
        receipt["receipt_cid"] = content_identity(
            {key: value for key, value in receipt.items() if key != "receipt_cid"}
        )
        return receipt
    except Exception as exc:  # noqa: BLE001 - settlement failure is a typed blocker
        reason = getattr(exc, "reason", None)
        if reason not in {MISSING, WAL_OUTSTANDING, LIVE_PROCESS, LANE_MISSING}:
            reason = MISSING
        return _deferred(reason, error_class=type(exc).__name__, error=str(exc)[:256])


@contextmanager
def hold_spar_runtime_settlement(
    repository_root: str | Path,
    *,
    owner_identity: Mapping[str, Any],
    target_repository_id: str,
) -> Iterator[dict[str, Any]]:
    """Hold the merge-queue writer lock while the caller performs owner CAS."""
    try:
        profile = read_spar_runtime_profile(repository_root)
    except Exception as exc:  # noqa: BLE001
        yield _deferred(error_class=type(exc).__name__)
        return
    try:
        lanes = []
        active = 0
        for spec in profile["lanes"]:
            snapshot = _lane_snapshot(lane_paths(Path(profile["state_path"]), spec["index"]), spec["index"])
            lanes.append(snapshot)
            active += snapshot["active_count"]
        with hold_merge_queue_settlement(
            profile["merge_queue_path"],
            target_repository_id=target_repository_id,
            target_branch=profile["merge_target_branch"],
            lock_timeout_seconds=0.05,
        ) as merge:
            settled = active == 0 and merge.get("settled") is True
            receipt = {
                "schema": SCHEMA,
                "admitted": settled,
                "settled": settled,
                "authority": "spar_runtime_settlement",
                "completion_authority": False,
                "semantic_acceptance_authority": False,
                "held": True,
                "owner_generation": owner_identity.get("generation"),
                "store_id": owner_identity.get("store_id") or profile["store_id"],
                "target": {
                    "repository_id": target_repository_id,
                    "branch": profile["merge_target_branch"],
                },
                "lanes": lanes,
                "active_count": active,
                "merge_queue": {
                    "settled": merge.get("settled") is True,
                    "active_count": merge.get("active_count"),
                    "receipt_cid": merge.get("receipt_cid"),
                },
            }
            if not settled:
                receipt["reason"] = MISSING
            receipt["receipt_cid"] = content_identity(
                {key: value for key, value in receipt.items() if key != "receipt_cid"}
            )
            yield receipt
    except (MergeQueueIntegrityError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", None)
        if reason not in {MISSING, WAL_OUTSTANDING, LIVE_PROCESS, LANE_MISSING}:
            reason = MISSING
        yield _deferred(reason, error_class=type(exc).__name__, error=str(exc)[:256])
