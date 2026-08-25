"""Fail-closed runtime settlement and terminal checkpoint gates for VRIF.

The canonical task and goal authority remains the Quack-owned control store.
The settlement APIs read only the four lane-local coordination/execution
sidecars and configured merge queue; they never construct or repair runtime
state.  The one explicit mutation API is the terminal-only sidecar checkpoint:
after the master has proven every process tree fenced, it takes the same
existing master/lane locks, requires exact zero-active stores, and asks DuckDB
to checkpoint committed WALs without changing logical rows or unlinking files.
The settlement context manager remains the completion-CAS authorization
boundary and retains every lock until the caller's compare-and-swap finishes.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Final

from ..merge import database_coordination as _coordination
from ..merge.merge_queue import (
    MERGE_QUEUE_SETTLEMENT_SCHEMA,
    MERGE_TARGET_BINDING_SCHEMA,
    MergeQueueIntegrityError,
    hold_merge_queue_settlement,
)
from ..task_sources.duckdb_state import connect_duckdb_with_policy

VRIF_RUNTIME_SETTLEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-runtime-settlement@1"
)
VRIF_COORDINATION_SETTLEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-coordination-settlement@1"
)
VRIF_EXECUTION_SETTLEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-execution-settlement@1"
)
VRIF_MERGE_QUEUE_VERIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-merge-queue-verification@1"
)
VRIF_LANE_CROSS_STORE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-lane-cross-store-binding@1"
)
VRIF_RUNTIME_SETTLEMENT_CONFIG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-runtime-settlement-config@1"
)
VRIF_RETIRED_COORDINATION_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-retired-coordination-snapshot@1"
)
VRIF_RETIRED_COORDINATION_LINEAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-retired-coordination-lineage@1"
)
VRIF_RUNTIME_SETTLEMENT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-runtime-settlement-binding@1"
)
VRIF_TERMINAL_SIDECAR_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-terminal-sidecar-checkpoint@1"
)
VRIF_CONFIG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "verified-residual-intelligence-foundry.scheduler_config@1"
)
VRIF_PROGRAM_IDENTIFIER: Final[str] = (
    "agent-supervisor-verified-residual-intelligence-foundry-v1"
)
VRIF_TASK_PREFIX: Final[str] = "VRIF-"
VRIF_SCHEDULER_CONFIG_RELATIVE_PATH: Final[str] = (
    "config/agent_supervisor_residual_intelligence_scheduler.json"
)
VRIF_STATE_RELATIVE_PATH: Final[str] = (
    "data/agent_supervisor/residual_intelligence_foundry/state"
)
VRIF_MERGE_QUEUE_RELATIVE_PATH: Final[str] = (
    "data/agent_supervisor/residual_intelligence_foundry/merge-queue"
)
VRIF_CONTROL_STORE_ID: Final[str] = (
    "data/agent_supervisor/residual_intelligence_foundry/control.duckdb"
)
VRIF_CONTROL_STORE_GENERATION: Final[str] = "vrif-v1"
VRIF_CONTROL_SCHEMA_REVISION: Final[str] = "1"
VRIF_TARGET_REPOSITORY_ID: Final[str] = (
    "repository:baguqeeraul4vqj7wze6dfjxogue57aadnvwrzw55527c2kfafyiyvuoaw2ca"
)

_CONFIG_MAX_BYTES = 2 * 1024 * 1024
_JSON_MAX_BYTES = 262_144
_MERGE_QUEUE_METADATA_MAX_BYTES = 64 * 1_024
_HISTORY_ROW_BOUND = 8_192
_RETIRED_STORE_MAX_BYTES = 64 * 1024 * 1024
_RETIRED_COORDINATION_LANE_INDEXES: Final[tuple[int, ...]] = (2, 3)
_MAX_RETIRED_COORDINATION_SNAPSHOTS = len(
    _RETIRED_COORDINATION_LANE_INDEXES
)
_MAX_ACTIVE_IDS = 1_024
_MAX_LOCK_TIMEOUT_SECONDS = 1.0
_PID_MAX_BYTES = 32
_PID_PAYLOAD = re.compile(rb"[1-9][0-9]*\n")
_PROCESS_INSTANCE = re.compile(r"process:[0-9a-f]{24}")
_SHA256_CID = re.compile(r"sha256:[0-9a-f]{64}")
_CONTROL_CONTENT_CID = re.compile(r"b[a-z2-7]{20,}")

_EXPECTED_LANES: Final[tuple[dict[str, Any], ...]] = (
    {
        "index": 0,
        "name": "vrif-lane-0",
        "strict_shard_remainder": 0,
        "initial_task_ids": ["VRIF-009"],
        "initial_focus": "deterministic-baselines",
    },
    {
        "index": 1,
        "name": "vrif-lane-1",
        "strict_shard_remainder": 1,
        "initial_task_ids": ["VRIF-010"],
        "initial_focus": "expert-specifications",
    },
    {
        "index": 2,
        "name": "vrif-lane-2",
        "strict_shard_remainder": 2,
        "initial_task_ids": ["VRIF-011"],
        "initial_focus": "calibration-and-abstention",
    },
    {
        "index": 3,
        "name": "vrif-lane-3",
        "strict_shard_remainder": 3,
        "initial_task_ids": ["VRIF-012"],
        "initial_focus": "ood-and-boundaries",
    },
)

_EXPECTED_DATABASE_PROGRAM: Final[dict[str, Any]] = {
    "authority_mode": "quack",
    "task_source_kind": "duckdb",
    "endpoint_secret_handle": "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
    "quack_endpoint": "quack:127.0.0.1:41327",
    "store_id": VRIF_CONTROL_STORE_ID,
    "store_generation": VRIF_CONTROL_STORE_GENERATION,
    "schema_revision": VRIF_CONTROL_SCHEMA_REVISION,
    "event_store_path": "data/agent_supervisor/residual_intelligence_foundry/events",
    "runtime_registry_path": "data/agent_supervisor/residual_intelligence_foundry/registry",
    "worktree_root": "data/agent_supervisor/residual_intelligence_foundry/worktrees",
    "export_profile": "vrif-v1",
    "failover_policy": "fail_closed",
    "explicit_legacy": False,
}

_LEASE_STATES: Final[tuple[str, ...]] = (
    "accepted",
    "released",
    "expired",
    "superseded",
    "completed",
)
_TASK_CLAIM_STATES: Final[tuple[str, ...]] = (
    "accepted",
    "released",
    "expired",
    "completed",
)
_RESOURCE_CLAIM_STATES: Final[tuple[str, ...]] = (
    "accepted",
    "released",
    "expired",
)
_ATTEMPT_STATES: Final[tuple[str, ...]] = (
    "running",
    "succeeded",
    "released",
    "expired",
)
_EXECUTION_STATES: Final[tuple[str, ...]] = (
    "running",
    "succeeded",
    "failed",
    "blocked",
)
_EXECUTION_PHASES: Final[tuple[str, ...]] = (
    "claimed",
    "context",
    "provider",
    "effect",
    "validation",
    "complete",
    "failed",
    "blocked",
)
_COMPLETION_STATES: Final[tuple[str, ...]] = ("prepared", "succeeded")
_LEASE_KINDS: Final[tuple[str, ...]] = (
    "task",
    "resource",
    "path",
    "merge",
    "maintenance",
    "provider_capacity",
    "prover_capacity",
)
_LEASE_MODES: Final[tuple[str, ...]] = ("exclusive", "shared")
_LEASE_EVENT_TYPES: Final[tuple[str, ...]] = (
    "acquired",
    "expired",
    "prepared_completion_aborted",
    "prepared_completion_recovered",
    "promoted_completion_reconciled",
    "protected_resource_write",
    "protected_task_write",
    "protected_write",
    "released",
    "renewed",
    "task_claim_settled",
    "task_claimed",
    "task_completion_prepared",
    "task_completion_promoted",
    _coordination.CONTROL_READY_FRONTIER_RECONCILIATION_EVENT,
    _coordination.TASK_COMPLETION_REARM_EVENT,
)
_TASK_COMPLETION_REARM_EVENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_cid",
        "claim_id",
        "attempt_id",
        "prior_attempt_number",
        "lease_id",
        "fencing_token",
        "fence_epoch",
        "previous_control_revision",
        "previous_control_status",
        "control_revision",
        "control_status",
        "control_cas_receipt_cid",
        "control_cas_receipt_digest",
        "completion_digest",
        "ready",
    }
)
_CONTROL_READY_FRONTIER_RECONCILIATION_EVENT_FIELDS: Final[frozenset[str]] = (
    frozenset(
        {
            "schema",
            "task_cid",
            "task_alias",
            "control_task_status",
            "control_task_revision",
            "control_snapshot_revision",
            "control_inventory_task_count",
            "control_inventory_projection_digest",
            "ready_frontier_task_count",
            "ready_frontier_projection_digest",
            "control_observation_digest",
            "direction",
            "ready_before",
            "ready_after",
            "task_completion_count",
            "task_claim_count",
            "task_attempt_count",
            "task_history_preserved",
            "lease_id",
            "fencing_token",
            "fence_epoch",
            "receipt_cid",
        }
    )
)
_CONTROL_READY_FRONTIER_RECONCILIATION_LEASE_FIELDS: Final[frozenset[str]] = (
    frozenset(
        {
            "schema",
            "task_cid",
            "task_alias",
            "control_snapshot_revision",
            "control_inventory_projection_digest",
            "ready_frontier_projection_digest",
            "control_observation_digest",
            "ready_after",
        }
    )
)
_CONTROL_READY_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {"proposed", "admitted", "pending", "ready", "todo", "queued", "retrying"}
)
_CONTROL_TERMINAL_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "completed",
        "complete",
        "done",
        "skipped",
        "cancelled",
        "failed",
        "quarantined",
        "rejected",
    }
)
_MERGE_QUEUE_STATES: Final[tuple[str, ...]] = (
    "pending",
    "processing",
    "completed",
    "quarantined",
    "cancelled",
)
_MERGE_QUEUE_ACTIVE_STATES: Final[tuple[str, ...]] = ("pending", "processing")

_COORDINATION_TABLES: Final[tuple[str, ...]] = tuple(
    sorted(_coordination._COORDINATION_REQUIRED_COLUMNS)
)
_COORDINATION_JSON_TABLES: Final[tuple[tuple[str, str], ...]] = (
    ("coordination_tasks", "task_cid"),
    ("task_completions", "task_cid"),
    ("fenced_leases", "lease_id"),
    ("lease_events", "event_id"),
    ("task_claims", "claim_id"),
    ("resource_claims", "claim_id"),
    ("maintenance_leases", "lease_id"),
)

_EXECUTION_COLUMNS: Final[dict[str, tuple[tuple[str, str, str], ...]]] = {
    "attempt_phases": (
        ("attempt_id", "VARCHAR", "NO"),
        ("phase", "VARCHAR", "NO"),
        ("committed_at_ms", "BIGINT", "NO"),
        ("fencing_token", "BIGINT", "NO"),
        ("fence_epoch", "BIGINT", "NO"),
        ("revision", "BIGINT", "NO"),
        ("body_json", "VARCHAR", "NO"),
    ),
    "daemon_execution_events": (
        ("event_id", "VARCHAR", "NO"),
        ("attempt_id", "VARCHAR", "NO"),
        ("task_cid", "VARCHAR", "NO"),
        ("event_type", "VARCHAR", "NO"),
        ("recorded_at_ms", "BIGINT", "NO"),
        ("body_json", "VARCHAR", "NO"),
    ),
    "daemon_execution_metadata": (
        ("key", "VARCHAR", "NO"),
        ("value", "VARCHAR", "NO"),
    ),
    "database_task_attempts": (
        ("attempt_id", "VARCHAR", "NO"),
        ("claim_id", "VARCHAR", "NO"),
        ("task_cid", "VARCHAR", "NO"),
        ("task_alias", "VARCHAR", "NO"),
        ("attempt_number", "BIGINT", "NO"),
        ("owner_session_id", "VARCHAR", "NO"),
        ("fencing_token", "BIGINT", "NO"),
        ("fence_epoch", "BIGINT", "NO"),
        ("lease_id", "VARCHAR", "NO"),
        ("committed_phase", "VARCHAR", "NO"),
        ("status", "VARCHAR", "NO"),
        ("started_at_ms", "BIGINT", "NO"),
        ("finished_at_ms", "BIGINT", "YES"),
        ("revision", "BIGINT", "NO"),
        ("body_json", "VARCHAR", "NO"),
    ),
    "effect_claims": (
        ("effect_id", "VARCHAR", "NO"),
        ("attempt_id", "VARCHAR", "NO"),
        ("task_cid", "VARCHAR", "NO"),
        ("effect_key", "VARCHAR", "NO"),
        ("idempotency_key", "VARCHAR", "NO"),
        ("owner_session_id", "VARCHAR", "NO"),
        ("recorded_at_ms", "BIGINT", "NO"),
        ("result_json", "VARCHAR", "NO"),
    ),
    "provider_invocations": (
        ("invocation_id", "VARCHAR", "NO"),
        ("attempt_id", "VARCHAR", "NO"),
        ("task_cid", "VARCHAR", "NO"),
        ("idempotency_key", "VARCHAR", "NO"),
        ("owner_session_id", "VARCHAR", "NO"),
        ("recorded_at_ms", "BIGINT", "NO"),
        ("result_json", "VARCHAR", "NO"),
    ),
}
_EXECUTION_INDEXES: Final[frozenset[str]] = frozenset(
    {
        "database_task_attempts_task_idx",
        "database_task_attempts_owner_idx",
        "database_task_attempts_claim_idx",
    }
)
_EXECUTION_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "interface",
        "schema",
        "authority_mode",
        "logical_owner_session_id",
        "process_instance_id",
        "state_schema_revision",
        "control_schema_profile_id",
        "control_schema_fingerprint",
    }
)


class VRIFRuntimeSettlementError(RuntimeError):
    """The local VRIF runtime cannot prove one immutable settlement snapshot."""


def _canonical_bytes(value: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json(raw: bytes | str, *, label: str, max_bytes: int) -> Any:
    encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
    if not encoded or len(encoded) > max_bytes:
        raise VRIFRuntimeSettlementError(f"{label} exceeds its strict JSON bound")

    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r}")

    try:
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise VRIFRuntimeSettlementError(f"{label} is not strict JSON") from exc


def _file_identity(
    path: Path,
    *,
    label: str,
    require_nonempty: bool = False,
) -> dict[str, Any]:
    try:
        details = path.lstat()
    except OSError as exc:
        raise VRIFRuntimeSettlementError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or int(details.st_nlink) != 1
    ):
        raise VRIFRuntimeSettlementError(
            f"{label} must be a single-link regular file"
        )
    if require_nonempty and int(details.st_size) <= 0:
        raise VRIFRuntimeSettlementError(f"{label} must not be empty")
    return {
        "path": str(path.absolute()),
        "device": int(details.st_dev),
        "inode": int(details.st_ino),
        "mode": int(stat.S_IMODE(details.st_mode)),
        "link_count": int(details.st_nlink),
        "uid": int(details.st_uid),
        "size_bytes": int(details.st_size),
        "modified_ns": int(details.st_mtime_ns),
        "changed_ns": int(details.st_ctime_ns),
    }


def _optional_wal_identity(database_path: Path) -> dict[str, Any]:
    wal_path = Path(str(database_path) + ".wal")
    try:
        wal_path.lstat()
    except FileNotFoundError:
        return {"path": str(wal_path.absolute()), "state": "absent"}
    except OSError as exc:
        raise VRIFRuntimeSettlementError("sidecar WAL is unreadable") from exc
    database = _file_identity(
        database_path,
        label="sidecar database",
        require_nonempty=True,
    )
    wal = _file_identity(wal_path, label="sidecar WAL", require_nonempty=True)
    if wal["uid"] != database["uid"]:
        raise VRIFRuntimeSettlementError(
            "sidecar WAL owner differs from its database owner"
        )
    return {"path": str(wal_path.absolute()), "state": "present", "file": wal}


def _store_identity(database_path: Path) -> dict[str, Any]:
    return {
        "database": _file_identity(
            database_path,
            label="sidecar database",
            require_nonempty=True,
        ),
        "wal": _optional_wal_identity(database_path),
    }


def _require_store_unchanged(
    database_path: Path,
    expected: Mapping[str, Any],
) -> None:
    if _store_identity(database_path) != dict(expected):
        raise VRIFRuntimeSettlementError(
            "sidecar database or WAL identity changed while settlement was guarded"
        )


def _bounded_lock_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("lock_timeout_seconds must be a finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < 0
        or result > _MAX_LOCK_TIMEOUT_SECONDS
    ):
        raise ValueError(
            "lock_timeout_seconds must be between 0 and "
            f"{_MAX_LOCK_TIMEOUT_SECONDS:g} seconds"
        )
    return result


@contextmanager
def _hold_existing_lock(
    path: Path,
    *,
    label: str,
    timeout_seconds: float,
) -> Iterator[dict[str, Any]]:
    expected = _file_identity(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VRIFRuntimeSettlementError(f"{label} could not be opened") from exc
    acquired = False
    deadline = time.monotonic() + timeout_seconds
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_dev) != expected["device"]
            or int(opened.st_ino) != expected["inode"]
        ):
            raise VRIFRuntimeSettlementError(f"{label} identity changed")
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise VRIFRuntimeSettlementError(f"{label} is busy") from None
                time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))
        try:
            yield expected
        finally:
            if _file_identity(path, label=label) != expected:
                raise VRIFRuntimeSettlementError(
                    f"{label} identity changed while guarded"
                )
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _absolute_lexical(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_real_directory(path: Path, *, label: str) -> None:
    try:
        details = path.lstat()
    except OSError as exc:
        raise VRIFRuntimeSettlementError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise VRIFRuntimeSettlementError(
            f"{label} must be an existing non-symlink directory"
        )


def _contained_existing_path(
    repository_root: Path,
    supplied: Path | str,
    *,
    label: str,
    kind: str,
) -> Path:
    raw = Path(supplied)
    if "\x00" in os.fspath(supplied) or ".." in raw.parts:
        raise VRIFRuntimeSettlementError(f"{label} contains traversal")
    candidate = _absolute_lexical(raw if raw.is_absolute() else repository_root / raw)
    try:
        relative = candidate.relative_to(repository_root)
    except ValueError as exc:
        raise VRIFRuntimeSettlementError(
            f"{label} escapes repository_root"
        ) from exc
    current = repository_root
    for ordinal, part in enumerate(relative.parts):
        current = current / part
        try:
            details = current.lstat()
        except OSError as exc:
            raise VRIFRuntimeSettlementError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(details.st_mode):
            raise VRIFRuntimeSettlementError(f"{label} contains a symbolic link")
        if ordinal < len(relative.parts) - 1 and not stat.S_ISDIR(details.st_mode):
            raise VRIFRuntimeSettlementError(
                f"{label} contains a non-directory component"
            )
    final = candidate.lstat()
    if kind == "directory" and not stat.S_ISDIR(final.st_mode):
        raise VRIFRuntimeSettlementError(f"{label} must be a directory")
    if kind == "file" and (
        not stat.S_ISREG(final.st_mode) or int(final.st_nlink) != 1
    ):
        raise VRIFRuntimeSettlementError(
            f"{label} must be a single-link regular file"
        )
    return candidate


def _read_stable_regular_bytes(
    path: Path,
    *,
    expected: Mapping[str, Any],
    label: str,
    max_bytes: int,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VRIFRuntimeSettlementError(f"{label} is unreadable") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_dev) != expected["device"]
            or int(opened.st_ino) != expected["inode"]
            or int(opened.st_size) != expected["size_bytes"]
        ):
            raise VRIFRuntimeSettlementError(
                f"{label} identity changed during read"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if (
        len(raw) != expected["size_bytes"]
        or len(raw) > max_bytes
        or _file_identity(path, label=label) != dict(expected)
    ):
        raise VRIFRuntimeSettlementError(
            f"{label} changed or exceeds its read bound"
        )
    return raw


def _parse_retired_coordination_config(
    decoded: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    raw = decoded.get("runtime_settlement")
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema",
        "retired_coordination_snapshots",
    }:
        raise VRIFRuntimeSettlementError(
            "VRIF runtime settlement config has a noncanonical field set"
        )
    if raw.get("schema") != VRIF_RUNTIME_SETTLEMENT_CONFIG_SCHEMA:
        raise VRIFRuntimeSettlementError(
            "VRIF runtime settlement config schema differs"
        )
    entries = raw.get("retired_coordination_snapshots")
    if (
        not isinstance(entries, list)
        or len(entries) > _MAX_RETIRED_COORDINATION_SNAPSHOTS
    ):
        raise VRIFRuntimeSettlementError(
            "VRIF retired coordination inventory exceeds its exact bound"
        )
    normalized: list[dict[str, Any]] = []
    seen_attempt_ids: set[str] = set()
    seen_lane_indexes: set[int] = set()
    for ordinal, item in enumerate(entries):
        if not isinstance(item, Mapping) or set(item) != {
            "schema",
            "lane_index",
            "database_path",
            "database_size_bytes",
            "database_sha256",
            "wal_path",
            "wal_size_bytes",
            "wal_sha256",
            "terminal_execution_attempt_ids",
            "terminal_execution_attempt_ids_cid",
        }:
            raise VRIFRuntimeSettlementError(
                f"retired coordination snapshot {ordinal} has a noncanonical field set"
            )
        lane_index = item.get("lane_index")
        if (
            type(lane_index) is not int
            or lane_index not in _RETIRED_COORDINATION_LANE_INDEXES
        ):
            raise VRIFRuntimeSettlementError(
                "VRIF retired coordination lane is outside the exact admission set"
            )
        if lane_index in seen_lane_indexes:
            raise VRIFRuntimeSettlementError(
                "VRIF retired coordination lane is repeated"
            )
        seen_lane_indexes.add(lane_index)
        if item.get("schema") != VRIF_RETIRED_COORDINATION_SNAPSHOT_SCHEMA:
            raise VRIFRuntimeSettlementError(
                "retired coordination snapshot schema differs"
            )
        database_relative = item.get("database_path")
        wal_relative = item.get("wal_path")
        if (
            type(database_relative) is not str
            or type(wal_relative) is not str
            or not database_relative
            or not wal_relative
            or len(database_relative.encode("utf-8")) > 4_096
            or len(wal_relative.encode("utf-8")) > 4_096
            or Path(database_relative).is_absolute()
            or Path(wal_relative).is_absolute()
            or ".." in Path(database_relative).parts
            or ".." in Path(wal_relative).parts
            or wal_relative != database_relative + ".wal"
        ):
            raise VRIFRuntimeSettlementError(
                "retired coordination snapshot paths differ"
            )
        database_parts = Path(database_relative).parts
        required_prefix = Path(
            VRIF_STATE_RELATIVE_PATH, "sidecar-quarantine"
        ).parts
        if (
            database_parts[: len(required_prefix)] != required_prefix
            or len(database_parts) < len(required_prefix) + 3
            or database_parts[-2] != f"lane-{lane_index}"
            or database_parts[-1]
            != "quack-lane-control.coordination.duckdb"
        ):
            raise VRIFRuntimeSettlementError(
                "retired coordination snapshot is outside the exact quarantine profile"
            )
        database_path = _contained_existing_path(
            repository_root,
            database_relative,
            label=f"retired coordination snapshot {ordinal} database",
            kind="file",
        )
        _contained_existing_path(
            repository_root,
            wal_relative,
            label=f"retired coordination snapshot {ordinal} WAL",
            kind="file",
        )
        policy_lock_path = database_path.with_name(f".{database_path.name}.lock")
        _contained_existing_path(
            repository_root,
            policy_lock_path,
            label=f"retired coordination snapshot {ordinal} policy lock",
            kind="file",
        )
        database_size = item.get("database_size_bytes")
        wal_size = item.get("wal_size_bytes")
        if (
            type(database_size) is not int
            or type(wal_size) is not int
            or database_size <= 0
            or wal_size <= 0
            or database_size > _RETIRED_STORE_MAX_BYTES
            or wal_size > _RETIRED_STORE_MAX_BYTES
        ):
            raise VRIFRuntimeSettlementError(
                "retired coordination snapshot size binding differs"
            )
        database_cid = item.get("database_sha256")
        wal_cid = item.get("wal_sha256")
        if (
            type(database_cid) is not str
            or type(wal_cid) is not str
            or _SHA256_CID.fullmatch(database_cid) is None
            or _SHA256_CID.fullmatch(wal_cid) is None
        ):
            raise VRIFRuntimeSettlementError(
                "retired coordination snapshot content binding differs"
            )
        attempt_ids = item.get("terminal_execution_attempt_ids")
        if (
            not isinstance(attempt_ids, list)
            or not attempt_ids
            or len(attempt_ids) > _HISTORY_ROW_BOUND
            or any(
                type(attempt_id) is not str
                or not attempt_id.startswith("attempt:")
                or len(attempt_id.encode("utf-8")) > 512
                for attempt_id in attempt_ids
            )
            or attempt_ids != sorted(set(attempt_ids))
            or seen_attempt_ids.intersection(attempt_ids)
            or item.get("terminal_execution_attempt_ids_cid")
            != _content_id(attempt_ids)
        ):
            raise VRIFRuntimeSettlementError(
                "retired coordination snapshot attempt inventory differs"
            )
        seen_attempt_ids.update(attempt_ids)
        normalized.append(
            {
                "schema": VRIF_RETIRED_COORDINATION_SNAPSHOT_SCHEMA,
                "lane_index": lane_index,
                "database_path": database_relative,
                "database_size_bytes": database_size,
                "database_sha256": database_cid,
                "wal_path": wal_relative,
                "wal_size_bytes": wal_size,
                "wal_sha256": wal_cid,
                "terminal_execution_attempt_ids": list(attempt_ids),
                "terminal_execution_attempt_ids_cid": _content_id(attempt_ids),
            }
        )
    if normalized != sorted(
        normalized,
        key=lambda entry: (entry["lane_index"], entry["database_path"]),
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination snapshot order differs"
        )
    return {
        "schema": VRIF_RUNTIME_SETTLEMENT_CONFIG_SCHEMA,
        "retired_coordination_snapshots": normalized,
    }


def _read_config(
    config_path: Path | str,
    *,
    repository_root: Path,
    target_branch: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _contained_existing_path(
        repository_root,
        config_path,
        label="VRIF scheduler config",
        kind="file",
    )
    initial = _file_identity(
        path,
        label="VRIF scheduler config",
        require_nonempty=True,
    )
    if initial["size_bytes"] > _CONFIG_MAX_BYTES:
        raise VRIFRuntimeSettlementError("VRIF scheduler config exceeds read bound")
    raw = _read_stable_regular_bytes(
        path,
        expected=initial,
        label="VRIF scheduler config",
        max_bytes=_CONFIG_MAX_BYTES,
    )
    decoded = _strict_json(
        raw,
        label="VRIF scheduler config",
        max_bytes=_CONFIG_MAX_BYTES,
    )
    if not isinstance(decoded, dict):
        raise VRIFRuntimeSettlementError("VRIF scheduler config must be an object")

    expected_scalars = {
        "schema": VRIF_CONFIG_SCHEMA,
        "program_identifier": VRIF_PROGRAM_IDENTIFIER,
        "board_namespace": VRIF_PROGRAM_IDENTIFIER,
        "task_prefix": VRIF_TASK_PREFIX,
        "max_lanes": 4,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
        "objective_goal_refinement_enabled": False,
        "reconciliation_guardrail_enabled": False,
        "merge_target_branch": target_branch,
    }
    if any(decoded.get(key) != value for key, value in expected_scalars.items()):
        raise VRIFRuntimeSettlementError(
            "VRIF scheduler config profile identity differs"
        )
    if (
        type(decoded.get("max_lanes")) is not int
        or decoded.get("exit_when_all_tracks_terminal") is not True
        or decoded.get("objective_refill_enabled") is not False
        or decoded.get("codebase_refill_enabled") is not False
        or decoded.get("objective_goal_refinement_enabled") is not False
        or decoded.get("reconciliation_guardrail_enabled") is not False
    ):
        raise VRIFRuntimeSettlementError(
            "VRIF scheduler config scalar types differ"
        )
    if decoded.get("database_program") != _EXPECTED_DATABASE_PROGRAM:
        raise VRIFRuntimeSettlementError(
            "VRIF scheduler Quack store identity differs"
        )
    if decoded["database_program"].get("explicit_legacy") is not False:
        raise VRIFRuntimeSettlementError(
            "VRIF scheduler Quack store identity differs"
        )
    if decoded.get("lanes") != list(_EXPECTED_LANES):
        raise VRIFRuntimeSettlementError(
            "VRIF scheduler requires the exact ordered four-lane profile"
        )
    for index, lane in enumerate(decoded["lanes"]):
        if (
            type(lane.get("index")) is not int
            or type(lane.get("strict_shard_remainder")) is not int
            or lane["index"] != index
            or lane["strict_shard_remainder"] != index
        ):
            raise VRIFRuntimeSettlementError(
                "VRIF scheduler lane identity types differ"
            )
    runtime_settlement = _parse_retired_coordination_config(
        decoded,
        repository_root=repository_root,
    )
    runtime_paths = decoded.get("runtime_paths")
    if not isinstance(runtime_paths, dict):
        raise VRIFRuntimeSettlementError("VRIF runtime_paths must be an object")
    if (
        runtime_paths.get("state") != VRIF_STATE_RELATIVE_PATH
        or runtime_paths.get("merge_queue") != VRIF_MERGE_QUEUE_RELATIVE_PATH
        or runtime_paths.get("generated_runtime_artifacts_are_completion_authority")
        is not False
    ):
        raise VRIFRuntimeSettlementError(
            "VRIF state or merge queue runtime path differs"
        )
    state_path = _contained_existing_path(
        repository_root,
        runtime_paths["state"],
        label="VRIF runtime state",
        kind="directory",
    )
    merge_queue_path = _contained_existing_path(
        repository_root,
        runtime_paths["merge_queue"],
        label="VRIF merge queue",
        kind="directory",
    )
    profile = {
        "config_path": str(path),
        "config_cid": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "config_file": initial,
        "repository_root": str(repository_root),
        "config_schema": VRIF_CONFIG_SCHEMA,
        "program_identifier": VRIF_PROGRAM_IDENTIFIER,
        "board_namespace": VRIF_PROGRAM_IDENTIFIER,
        "task_prefix": VRIF_TASK_PREFIX,
        "max_lanes": 4,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
        "objective_goal_refinement_enabled": False,
        "reconciliation_guardrail_enabled": False,
        "state_path": str(state_path),
        "merge_queue_path": str(merge_queue_path),
        "database_program": dict(_EXPECTED_DATABASE_PROGRAM),
        "lanes": [dict(lane) for lane in _EXPECTED_LANES],
        "runtime_settlement": runtime_settlement,
    }
    profile["profile_cid"] = _content_id(profile)
    return decoded, profile


def _require_config_unchanged(profile: Mapping[str, Any]) -> None:
    path = Path(str(profile["config_path"]))
    expected = profile["config_file"]
    observed = _file_identity(
        path,
        label="VRIF scheduler config",
        require_nonempty=True,
    )
    if observed != expected:
        raise VRIFRuntimeSettlementError(
            "VRIF scheduler config changed while settlement was guarded"
        )
    raw = _read_stable_regular_bytes(
        path,
        expected=expected,
        label="VRIF scheduler config",
        max_bytes=_CONFIG_MAX_BYTES,
    )
    if "sha256:" + hashlib.sha256(raw).hexdigest() != profile["config_cid"]:
        raise VRIFRuntimeSettlementError(
            "VRIF scheduler config content changed while guarded"
        )


def _pid_observation(path: Path, *, label: str) -> dict[str, Any]:
    try:
        path.lstat()
    except FileNotFoundError:
        return {"path": str(path.absolute()), "state": "absent"}
    except OSError as exc:
        raise VRIFRuntimeSettlementError(f"{label} is unreadable") from exc
    identity = _file_identity(path, label=label, require_nonempty=True)
    if identity["uid"] != os.geteuid():
        raise VRIFRuntimeSettlementError(
            f"{label} is not an owner-owned PID projection"
        )
    if identity["size_bytes"] > _PID_MAX_BYTES:
        raise VRIFRuntimeSettlementError(f"{label} exceeds the PID read bound")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            int(opened.st_dev) != identity["device"]
            or int(opened.st_ino) != identity["inode"]
        ):
            raise VRIFRuntimeSettlementError(f"{label} identity changed")
        payload = os.read(descriptor, _PID_MAX_BYTES + 1)
    except OSError as exc:
        raise VRIFRuntimeSettlementError(f"{label} could not be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        _file_identity(path, label=label) != identity
        or _PID_PAYLOAD.fullmatch(payload) is None
    ):
        raise VRIFRuntimeSettlementError(
            f"{label} is malformed or changed during read"
        )
    pid = int(payload[:-1].decode("ascii"))
    try:
        os.kill(pid, 0)
    except OverflowError as exc:
        raise VRIFRuntimeSettlementError(f"{label} liveness is unknown") from exc
    except ProcessLookupError as exc:
        if exc.errno != errno.ESRCH:
            raise VRIFRuntimeSettlementError(f"{label} liveness is unknown") from exc
    except (PermissionError, OSError) as exc:
        raise VRIFRuntimeSettlementError(f"{label} liveness is unknown") from exc
    else:
        raise VRIFRuntimeSettlementError(
            f"{label} names a live or reused process"
        )
    return {
        "path": str(path.absolute()),
        "state": "stale_dead",
        "pid": pid,
        "content_cid": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "file": identity,
    }


def _lifecycle_observations(state_path: Path) -> dict[str, Any]:
    master = _pid_observation(
        state_path / "configured-board-master.pid",
        label="VRIF master PID marker",
    )
    lanes = [
        {
            "index": index,
            "observation": _pid_observation(
                state_path / f"lane-{index}" / f"vrif_lane_{index}_supervisor.pid",
                label=f"VRIF lane {index} supervisor PID marker",
            ),
        }
        for index in range(4)
    ]
    return {"master_pid": master, "lane_supervisor_pids": lanes}


def _status_counts(
    connection: Any,
    *,
    table: str,
    column: str,
    allowed: Sequence[str],
) -> dict[str, int]:
    counts = {value: 0 for value in allowed}
    rows = connection.execute(
        f"""
        SELECT {column}, count(*)
        FROM {table}
        GROUP BY {column}
        ORDER BY {column}
        LIMIT ?
        """,
        [len(allowed) + 1],
    ).fetchall()
    if len(rows) > len(allowed):
        raise VRIFRuntimeSettlementError(f"{table} has an unknown state")
    for raw_state, raw_count in rows:
        state_value = str(raw_state or "")
        if state_value not in counts or int(raw_count) < 0:
            raise VRIFRuntimeSettlementError(f"{table} has an unknown state")
        counts[state_value] = int(raw_count)
    return counts


def _closed_values(
    connection: Any,
    *,
    table: str,
    column: str,
    allowed: Sequence[str],
) -> None:
    rows = connection.execute(
        f"SELECT DISTINCT {column} FROM {table} ORDER BY {column} LIMIT ?",
        [len(allowed) + 1],
    ).fetchall()
    if any(str(row[0] or "") not in allowed for row in rows):
        raise VRIFRuntimeSettlementError(
            f"{table}.{column} contains an unknown value"
        )


def _row_counts(connection: Any, tables: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for table in tables:
        row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
        if row is None or int(row[0]) < 0:
            raise VRIFRuntimeSettlementError(f"{table} row count is unavailable")
        result[table] = int(row[0])
    return result


def _require_unique_rows(
    connection: Any,
    *,
    table: str,
    columns: Sequence[str],
) -> None:
    grouped = ", ".join(columns)
    row = connection.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT {grouped}, count(*) AS duplicate_count
            FROM {table}
            GROUP BY {grouped}
            HAVING count(*) <> 1
        )
        """
    ).fetchone()
    if row is None or int(row[0]) != 0:
        raise VRIFRuntimeSettlementError(
            f"{table} contains duplicate producer identities"
        )


def _bounded_json_rows(
    connection: Any,
    *,
    table: str,
    identity_column: str,
    json_column: str,
) -> list[tuple[str, dict[str, Any]]]:
    rows = connection.execute(
        f"""
        SELECT
            CASE WHEN octet_length(encode({identity_column})) <= 512
                 THEN {identity_column} END,
            octet_length(encode({identity_column})),
            CASE WHEN octet_length(encode({json_column})) <= ?
                 THEN {json_column} END,
            octet_length(encode({json_column}))
        FROM {table}
        ORDER BY {identity_column}
        LIMIT ?
        """,
        [_JSON_MAX_BYTES, _HISTORY_ROW_BOUND + 1],
    ).fetchall()
    if len(rows) > _HISTORY_ROW_BOUND:
        raise VRIFRuntimeSettlementError(f"{table} exceeds the history read bound")
    decoded: list[tuple[str, dict[str, Any]]] = []
    for identity, identity_size, raw_json, json_size in rows:
        if (
            identity is None
            or int(identity_size) <= 0
            or int(identity_size) > 512
            or raw_json is None
            or int(json_size) <= 0
            or int(json_size) > _JSON_MAX_BYTES
        ):
            raise VRIFRuntimeSettlementError(f"{table} contains an oversized row")
        body = _strict_json(
            str(raw_json),
            label=f"{table} JSON for {identity}",
            max_bytes=_JSON_MAX_BYTES,
        )
        if not isinstance(body, dict):
            raise VRIFRuntimeSettlementError(
                f"{table} JSON for {identity} must be an object"
            )
        decoded.append((str(identity), body))
    return decoded


def _logical_owner_id(
    *,
    logical_database_path: Path,
    coordination_path: Path,
    execution_path: Path,
) -> str:
    payload = "\n".join(
        str(path.absolute())
        for path in (
            logical_database_path,
            coordination_path,
            execution_path,
        )
    ).encode("utf-8")
    return f"embedded-store:{hashlib.sha256(payload).hexdigest()[:32]}"


def _exact_nonempty_text(value: Any) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and "\x00" not in value
    )


def _control_receipt_cid(value: Any) -> bool:
    return bool(
        type(value) is str
        and (
            _SHA256_CID.fullmatch(value) is not None
            or _CONTROL_CONTENT_CID.fullmatch(value) is not None
        )
    )


def _coordination_records_by_id(
    connection: Any,
    *,
    table: str,
    columns: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        f"SELECT {', '.join(columns)} FROM {table} ORDER BY {columns[0]}"
    ).fetchall()
    return {
        str(row[0] or ""): dict(zip(columns, row, strict=True))
        for row in rows
    }


def _validate_task_completion_rearm_events(
    connection: Any,
    *,
    json_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Validate typed completion-rearm history, not only its event name."""

    events = connection.execute(
        """
        SELECT event_id, lease_id, scope_key, fencing_token, fence_epoch
        FROM lease_events
        WHERE event_type = ?
        ORDER BY event_id
        LIMIT ?
        """,
        [
            _coordination.TASK_COMPLETION_REARM_EVENT,
            _HISTORY_ROW_BOUND + 1,
        ],
    ).fetchall()
    if len(events) > _HISTORY_ROW_BOUND:
        raise VRIFRuntimeSettlementError(
            "coordination task-completion rearm history exceeds its bound"
        )
    if not events:
        return

    leases = _coordination_records_by_id(
        connection,
        table="fenced_leases",
        columns=(
            "lease_id",
            "lease_kind",
            "scope_key",
            "scope",
            "mode",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "state",
            "task_cid",
            "claim_id",
            "attempt_id",
            "attempt_number",
        ),
    )
    claims = _coordination_records_by_id(
        connection,
        table="task_claims",
        columns=(
            "claim_id",
            "task_cid",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "state",
            "attempt_id",
            "attempt_number",
            "lease_id",
        ),
    )
    attempts = _coordination_records_by_id(
        connection,
        table="task_attempts",
        columns=(
            "attempt_id",
            "task_cid",
            "attempt_number",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "status",
        ),
    )
    task_cids = {
        str(row[0] or "")
        for row in connection.execute(
            "SELECT task_cid FROM coordination_tasks ORDER BY task_cid"
        ).fetchall()
    }
    seen_control_revisions: set[tuple[str, int]] = set()
    for event_id_raw, event_lease_id_raw, scope_key_raw, token_raw, epoch_raw in events:
        event_id = str(event_id_raw or "")
        event_lease_id = str(event_lease_id_raw or "")
        body = json_rows["lease_events"].get(event_id)
        if body is None or set(body) != _TASK_COMPLETION_REARM_EVENT_FIELDS:
            raise VRIFRuntimeSettlementError(
                "coordination task-completion rearm event body differs"
            )
        task_cid = body.get("task_cid")
        claim_id = body.get("claim_id")
        attempt_id = body.get("attempt_id")
        lease_id = body.get("lease_id")
        prior_attempt_number = body.get("prior_attempt_number")
        token = body.get("fencing_token")
        epoch = body.get("fence_epoch")
        previous_revision = body.get("previous_control_revision")
        control_revision = body.get("control_revision")
        text_fields = (
            task_cid,
            claim_id,
            attempt_id,
            lease_id,
            body.get("previous_control_status"),
            body.get("control_status"),
        )
        integer_fields = (
            prior_attempt_number,
            token,
            epoch,
            previous_revision,
            control_revision,
        )
        if (
            body.get("schema") != _coordination.TASK_COMPLETION_REARM_SCHEMA
            or any(not _exact_nonempty_text(value) for value in text_fields)
            or any(type(value) is not int or value <= 0 for value in integer_fields)
            or body.get("previous_control_status")
            not in {"completed", "complete", "done"}
            or body.get("control_status") != "retrying"
            or control_revision != previous_revision + 1
            or not _control_receipt_cid(body.get("control_cas_receipt_cid"))
            or type(body.get("control_cas_receipt_digest")) is not str
            or _SHA256_CID.fullmatch(body["control_cas_receipt_digest"]) is None
            or type(body.get("completion_digest")) is not str
            or _SHA256_CID.fullmatch(body["completion_digest"]) is None
            or type(body.get("ready")) is not bool
            or task_cid not in task_cids
        ):
            raise VRIFRuntimeSettlementError(
                "coordination task-completion rearm event is malformed"
            )
        revision_identity = (task_cid, control_revision)
        if revision_identity in seen_control_revisions:
            raise VRIFRuntimeSettlementError(
                "coordination task-completion rearm revision is duplicated"
            )
        seen_control_revisions.add(revision_identity)

        expected_scope_key = _coordination.exclusive_scope_key(
            lease_kind=_coordination.LeaseKind.TASK,
            scope=task_cid,
            task_cid=task_cid,
        )
        lease = leases.get(event_lease_id)
        claim = claims.get(claim_id)
        attempt = attempts.get(attempt_id)
        terminal_state = lease.get("state") if lease is not None else ""
        expected_claim_state = (
            terminal_state if terminal_state in {"released", "completed"} else ""
        )
        if (
            event_lease_id != lease_id
            or str(scope_key_raw or "") != expected_scope_key
            or type(token_raw) is not int
            or type(epoch_raw) is not int
            or token_raw != token
            or epoch_raw != epoch
            or lease is None
            or lease.get("lease_kind") != "task"
            or lease.get("scope_key") != expected_scope_key
            or lease.get("scope") != task_cid
            or lease.get("mode") != "exclusive"
            or terminal_state not in {"released", "completed"}
            or lease.get("task_cid") != task_cid
            or lease.get("claim_id") != claim_id
            or lease.get("attempt_id") != attempt_id
            or lease.get("attempt_number") != prior_attempt_number
            or lease.get("fencing_token") != token
            or lease.get("fence_epoch") != epoch
            or claim is None
            or claim.get("task_cid") != task_cid
            or claim.get("owner_session_id") != lease.get("owner_session_id")
            or claim.get("fencing_token") != token
            or claim.get("fence_epoch") != epoch
            or claim.get("state") != expected_claim_state
            or claim.get("attempt_id") != attempt_id
            or claim.get("attempt_number") != prior_attempt_number
            or claim.get("lease_id") != lease_id
            or attempt is None
            or attempt.get("task_cid") != task_cid
            or attempt.get("attempt_number") != prior_attempt_number
            or attempt.get("owner_session_id") != lease.get("owner_session_id")
            or attempt.get("fencing_token") != token
            or attempt.get("fence_epoch") != epoch
            or attempt.get("status") != "succeeded"
        ):
            raise VRIFRuntimeSettlementError(
                "coordination task-completion rearm event is fence-unbound"
            )


def _validate_control_ready_frontier_reconciliation_events(
    connection: Any,
    *,
    json_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Validate exact bidirectional control-frontier synchronization receipts."""

    event_columns = (
        "event_id",
        "lease_id",
        "scope_key",
        "event_type",
        "fencing_token",
        "fence_epoch",
        "observed_at_ms",
    )
    all_events = connection.execute(
        f"SELECT {', '.join(event_columns)} FROM lease_events ORDER BY event_id"
    ).fetchall()
    event_records = [dict(zip(event_columns, row, strict=True)) for row in all_events]
    events = [
        event
        for event in event_records
        if event["event_type"]
        == _coordination.CONTROL_READY_FRONTIER_RECONCILIATION_EVENT
    ]
    if not events:
        return

    leases = _coordination_records_by_id(
        connection,
        table="fenced_leases",
        columns=(
            "lease_id",
            "lease_kind",
            "scope_key",
            "scope",
            "mode",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "acquired_at_ms",
            "expires_at_ms",
            "state",
            "revision",
            "task_cid",
            "worktree_id",
            "resource_kind",
            "resource_id",
            "repository_id",
            "path",
            "claim_id",
            "attempt_id",
            "attempt_number",
            "idempotency_key",
        ),
    )
    maintenance = _coordination_records_by_id(
        connection,
        table="maintenance_leases",
        columns=(
            "lease_id",
            "scope",
            "owner_session_id",
            "process_birth_id",
            "fencing_token",
            "fence_epoch",
            "acquired_at_ms",
            "expires_at_ms",
            "released_at_ms",
            "state",
            "revision",
        ),
    )
    tasks = _coordination_records_by_id(
        connection,
        table="coordination_tasks",
        columns=("task_cid", "task_id"),
    )
    current_completion_counts = {
        str(task_cid or ""): int(count or 0)
        for task_cid, count in connection.execute(
            "SELECT task_cid, count(*) FROM task_completions GROUP BY task_cid"
        ).fetchall()
    }
    current_claim_counts = {
        str(task_cid or ""): int(count or 0)
        for task_cid, count in connection.execute(
            "SELECT task_cid, count(*) FROM task_claims GROUP BY task_cid"
        ).fetchall()
    }
    current_attempt_counts = {
        str(task_cid or ""): int(count or 0)
        for task_cid, count in connection.execute(
            "SELECT task_cid, count(*) FROM task_attempts GROUP BY task_cid"
        ).fetchall()
    }
    events_by_lease: dict[str, list[dict[str, Any]]] = {}
    for event in event_records:
        events_by_lease.setdefault(str(event["lease_id"] or ""), []).append(event)

    seen_observations: set[tuple[str, str]] = set()
    seen_receipts: set[str] = set()
    for event in events:
        event_id = str(event["event_id"] or "")
        event_lease_id = str(event["lease_id"] or "")
        body = json_rows["lease_events"].get(event_id)
        if (
            body is None
            or set(body)
            != _CONTROL_READY_FRONTIER_RECONCILIATION_EVENT_FIELDS
        ):
            raise VRIFRuntimeSettlementError(
                "coordination control-ready-frontier event body differs"
            )
        task_cid = body.get("task_cid")
        task_alias = body.get("task_alias")
        task_status = body.get("control_task_status")
        direction = body.get("direction")
        lease_id = body.get("lease_id")
        snapshot_revision = body.get("control_snapshot_revision")
        inventory_count = body.get("control_inventory_task_count")
        ready_count = body.get("ready_frontier_task_count")
        completion_count = body.get("task_completion_count")
        claim_count = body.get("task_claim_count")
        attempt_count = body.get("task_attempt_count")
        token = body.get("fencing_token")
        epoch = body.get("fence_epoch")
        control_task_revision = body.get("control_task_revision")
        nonnegative_integers = (
            inventory_count,
            ready_count,
            completion_count,
            claim_count,
            attempt_count,
        )
        positive_integers = (
            snapshot_revision,
            token,
            epoch,
            control_task_revision,
        )
        digest_fields = (
            body.get("control_inventory_projection_digest"),
            body.get("ready_frontier_projection_digest"),
            body.get("control_observation_digest"),
            body.get("receipt_cid"),
        )
        direction_valid = (
            direction == "demote"
            and body.get("ready_before") is True
            and body.get("ready_after") is False
            and task_status in _CONTROL_TERMINAL_TASK_STATUSES
            and ready_count < inventory_count
        ) or (
            direction == "promote"
            and body.get("ready_before") is False
            and body.get("ready_after") is True
            and task_status in _CONTROL_READY_TASK_STATUSES
            and completion_count == 0
            and ready_count > 0
        )
        if (
            body.get("schema")
            != _coordination.CONTROL_READY_FRONTIER_RECONCILIATION_SCHEMA
            or any(
                not _exact_nonempty_text(value)
                for value in (task_cid, task_alias, task_status, direction, lease_id)
            )
            or any(
                type(value) is not int or value < 0
                for value in nonnegative_integers
            )
            or any(type(value) is not int or value <= 0 for value in positive_integers)
            or inventory_count < 1
            or inventory_count > 1_000
            or ready_count > inventory_count
            or completion_count > 1
            or claim_count != attempt_count
            or body.get("task_history_preserved") is not True
            or not direction_valid
            or any(
                type(value) is not str or _SHA256_CID.fullmatch(value) is None
                for value in digest_fields
            )
            or task_cid not in tasks
            or tasks[task_cid].get("task_id") != task_alias
            or claim_count > current_claim_counts.get(task_cid, 0)
            or attempt_count > current_attempt_counts.get(task_cid, 0)
            or (
                completion_count > current_completion_counts.get(task_cid, 0)
                and not any(
                    other.get("event_type")
                    == _coordination.TASK_COMPLETION_REARM_EVENT
                    and (
                        rearm_body := json_rows["lease_events"].get(
                            str(other.get("event_id") or ""), {}
                        )
                    ).get("task_cid")
                    == task_cid
                    and type(rearm_body.get("previous_control_revision")) is int
                    and rearm_body["previous_control_revision"]
                    >= control_task_revision
                    for other in event_records
                )
            )
        ):
            raise VRIFRuntimeSettlementError(
                "coordination control-ready-frontier event is malformed"
            )

        expected_observation_digest = _content_id(
            {
                "task_cid": task_cid,
                "control_snapshot_revision": snapshot_revision,
                "control_inventory_projection_digest": body[
                    "control_inventory_projection_digest"
                ],
                "ready_frontier_projection_digest": body[
                    "ready_frontier_projection_digest"
                ],
            }
        )
        receipt_payload = dict(body)
        stored_receipt = str(receipt_payload.pop("receipt_cid"))
        if (
            body["control_observation_digest"] != expected_observation_digest
            or stored_receipt != _content_id(receipt_payload)
        ):
            raise VRIFRuntimeSettlementError(
                "coordination control-ready-frontier event digest differs"
            )
        observation_identity = (task_cid, expected_observation_digest)
        if observation_identity in seen_observations or stored_receipt in seen_receipts:
            raise VRIFRuntimeSettlementError(
                "coordination control-ready-frontier receipt is duplicated"
            )
        seen_observations.add(observation_identity)
        seen_receipts.add(stored_receipt)

        maintenance_scope = f"control-ready-frontier:{task_cid}"
        expected_scope_key = _coordination.exclusive_scope_key(
            lease_kind=_coordination.LeaseKind.MAINTENANCE,
            scope=maintenance_scope,
        )
        expected_lease_body = {
            "schema": _coordination.CONTROL_READY_FRONTIER_RECONCILIATION_SCHEMA,
            "task_cid": task_cid,
            "task_alias": task_alias,
            "control_snapshot_revision": snapshot_revision,
            "control_inventory_projection_digest": body[
                "control_inventory_projection_digest"
            ],
            "ready_frontier_projection_digest": body[
                "ready_frontier_projection_digest"
            ],
            "control_observation_digest": body["control_observation_digest"],
            "ready_after": body["ready_after"],
        }
        lease = leases.get(event_lease_id)
        maintenance_lease = maintenance.get(event_lease_id)
        lease_body = json_rows["fenced_leases"].get(event_lease_id)
        maintenance_body = json_rows["maintenance_leases"].get(event_lease_id)
        if (
            event_lease_id != lease_id
            or event.get("scope_key") != expected_scope_key
            or event.get("fencing_token") != token
            or event.get("fence_epoch") != epoch
            or lease is None
            or lease.get("lease_kind") != "maintenance"
            or lease.get("scope_key") != expected_scope_key
            or lease.get("scope") != maintenance_scope
            or lease.get("mode") != "exclusive"
            or not _exact_nonempty_text(lease.get("owner_session_id"))
            or lease.get("fencing_token") != token
            or lease.get("fence_epoch") != epoch
            or lease.get("state") != "released"
            or lease.get("revision") != 2
            or lease.get("task_cid") != ""
            or lease.get("worktree_id") != ""
            or lease.get("resource_kind") != ""
            or lease.get("resource_id") != ""
            or lease.get("repository_id") != ""
            or lease.get("path") != ""
            or lease.get("claim_id") != ""
            or lease.get("attempt_id") != ""
            or lease.get("attempt_number") != 0
            or lease.get("idempotency_key")
            != f"control-ready-frontier:{body['control_observation_digest']}"
            or lease_body != expected_lease_body
            or set(lease_body or {})
            != _CONTROL_READY_FRONTIER_RECONCILIATION_LEASE_FIELDS
            or maintenance_lease is None
            or maintenance_lease.get("scope") != maintenance_scope
            or maintenance_lease.get("owner_session_id")
            != lease.get("owner_session_id")
            or maintenance_lease.get("process_birth_id") != ""
            or maintenance_lease.get("fencing_token") != token
            or maintenance_lease.get("fence_epoch") != epoch
            or maintenance_lease.get("acquired_at_ms")
            != lease.get("acquired_at_ms")
            or maintenance_lease.get("acquired_at_ms")
            != event.get("observed_at_ms")
            or maintenance_lease.get("expires_at_ms") != lease.get("expires_at_ms")
            or maintenance_lease.get("released_at_ms") is None
            or maintenance_lease.get("released_at_ms")
            != event.get("observed_at_ms")
            or maintenance_lease.get("state") != "released"
            or maintenance_lease.get("revision") != 2
            or maintenance_body != expected_lease_body
            or set(maintenance_body or {})
            != _CONTROL_READY_FRONTIER_RECONCILIATION_LEASE_FIELDS
        ):
            raise VRIFRuntimeSettlementError(
                "coordination control-ready-frontier event is maintenance-unbound"
            )

        related_events = events_by_lease.get(event_lease_id, [])
        related_types = [str(item.get("event_type") or "") for item in related_events]
        acquired = [item for item in related_events if item.get("event_type") == "acquired"]
        released = [item for item in related_events if item.get("event_type") == "released"]
        reconciled = [
            item
            for item in related_events
            if item.get("event_type")
            == _coordination.CONTROL_READY_FRONTIER_RECONCILIATION_EVENT
        ]
        acquired_body = (
            json_rows["lease_events"].get(str(acquired[0]["event_id"] or ""))
            if len(acquired) == 1
            else None
        )
        released_body = (
            json_rows["lease_events"].get(str(released[0]["event_id"] or ""))
            if len(released) == 1
            else None
        )
        if (
            sorted(related_types)
            != sorted(
                [
                    "acquired",
                    _coordination.CONTROL_READY_FRONTIER_RECONCILIATION_EVENT,
                    "released",
                ]
            )
            or len(acquired) != 1
            or len(released) != 1
            or len(reconciled) != 1
            or acquired_body
            != {
                "owner_session_id": lease["owner_session_id"],
                "mode": "exclusive",
            }
            or released_body
            != {
                "reason": _coordination.CONTROL_READY_FRONTIER_RECONCILIATION_EVENT
            }
            or acquired[0].get("scope_key") != expected_scope_key
            or released[0].get("scope_key") != expected_scope_key
            or acquired[0].get("fencing_token") != token
            or released[0].get("fencing_token") != token
            or acquired[0].get("fence_epoch") != epoch
            or released[0].get("fence_epoch") != epoch
            or acquired[0].get("observed_at_ms") != event.get("observed_at_ms")
            or released[0].get("observed_at_ms") != event.get("observed_at_ms")
        ):
            raise VRIFRuntimeSettlementError(
                "coordination control-ready-frontier event lineage differs"
            )


def _read_coordination_snapshot(
    database_path: Path,
    *,
    store_identity: Mapping[str, Any],
    max_active_ids: int,
) -> dict[str, Any]:
    connection: Any | None = None
    transaction_open = False
    try:
        import duckdb

        connection = connect_duckdb_with_policy(
            duckdb,
            database_path,
            read_only=True,
        )
        connection.execute("BEGIN TRANSACTION")
        transaction_open = True
        _coordination._validate_coordination_authority(connection)

        nullable_columns = {
            ("task_claims", "released_at_ms"),
            ("task_attempts", "finished_at_ms"),
            ("maintenance_leases", "released_at_ms"),
        }
        observed_nullability = connection.execute(
            """
            SELECT table_name, column_name, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
        expected_nullability = [
            (
                table,
                column,
                "YES" if (table, column) in nullable_columns else "NO",
            )
            for table in sorted(_coordination._COORDINATION_REQUIRED_COLUMNS)
            for column, _data_type in _coordination._COORDINATION_REQUIRED_COLUMNS[
                table
            ]
        ]
        if [tuple(str(item) for item in row) for row in observed_nullability] != expected_nullability:
            raise VRIFRuntimeSettlementError(
                "coordination sidecar column nullability inventory differs"
            )

        unique_identities = {
            "coordination_metadata": ("key",),
            "coordination_tasks": ("task_cid",),
            "task_dependencies": ("task_cid", "dependency_task_cid"),
            "task_completions": ("task_cid",),
            "fenced_leases": ("lease_id",),
            "token_history": ("scope_key", "fencing_token", "fence_epoch"),
            "lease_events": ("event_id",),
            "task_claims": ("claim_id",),
            "task_attempts": ("attempt_id",),
            "resource_claims": ("claim_id",),
            "maintenance_leases": ("lease_id",),
        }
        for table, columns in unique_identities.items():
            _require_unique_rows(connection, table=table, columns=columns)
        _require_unique_rows(
            connection,
            table="task_attempts",
            columns=("task_cid", "attempt_number"),
        )

        row_counts = _row_counts(connection, _COORDINATION_TABLES)
        status_counts = {
            "fenced_leases": _status_counts(
                connection,
                table="fenced_leases",
                column="state",
                allowed=_LEASE_STATES,
            ),
            "task_claims": _status_counts(
                connection,
                table="task_claims",
                column="state",
                allowed=_TASK_CLAIM_STATES,
            ),
            "task_attempts": _status_counts(
                connection,
                table="task_attempts",
                column="status",
                allowed=_ATTEMPT_STATES,
            ),
            "resource_claims": _status_counts(
                connection,
                table="resource_claims",
                column="state",
                allowed=_RESOURCE_CLAIM_STATES,
            ),
            "maintenance_leases": _status_counts(
                connection,
                table="maintenance_leases",
                column="state",
                allowed=_RESOURCE_CLAIM_STATES,
            ),
            "task_completions": _status_counts(
                connection,
                table="task_completions",
                column="status",
                allowed=_COMPLETION_STATES,
            ),
        }
        _closed_values(
            connection,
            table="fenced_leases",
            column="lease_kind",
            allowed=_LEASE_KINDS,
        )
        _closed_values(
            connection,
            table="fenced_leases",
            column="mode",
            allowed=_LEASE_MODES,
        )
        _closed_values(
            connection,
            table="resource_claims",
            column="mode",
            allowed=_LEASE_MODES,
        )
        _closed_values(
            connection,
            table="lease_events",
            column="event_type",
            allowed=_LEASE_EVENT_TYPES,
        )

        json_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for table, identity_column in _COORDINATION_JSON_TABLES:
            json_rows[table] = {
                identity: body
                for identity, body in _bounded_json_rows(
                    connection,
                    table=table,
                    identity_column=identity_column,
                    json_column="body_json",
                )
            }

        basic_integrity_queries = (
            """
            SELECT count(*) FROM coordination_tasks
            WHERE task_cid = '' OR task_id = '' OR registered_at_ms < 0
            """,
            """
            SELECT count(*) FROM task_dependencies
            WHERE task_cid = '' OR dependency_task_cid = ''
               OR task_cid = dependency_task_cid
            """,
            """
            SELECT count(*) FROM fenced_leases
            WHERE lease_id = '' OR lease_kind = '' OR scope_key = '' OR scope = ''
               OR owner_session_id = '' OR fencing_token <= 0 OR fence_epoch <= 0
               OR acquired_at_ms < 0 OR expires_at_ms < acquired_at_ms
               OR revision <= 0
            """,
            """
            SELECT count(*) FROM token_history
            WHERE scope_key = '' OR fencing_token <= 0 OR fence_epoch <= 0
               OR recorded_at_ms < 0
            """,
            """
            SELECT count(*) FROM lease_events
            WHERE event_id = '' OR lease_id = '' OR scope_key = ''
               OR event_type = '' OR fencing_token <= 0 OR fence_epoch <= 0
               OR observed_at_ms < 0
            """,
            """
            SELECT count(*) FROM task_claims
            WHERE claim_id = '' OR task_cid = '' OR owner_session_id = ''
               OR fencing_token <= 0 OR fence_epoch <= 0 OR claimed_at_ms < 0
               OR expires_at_ms < claimed_at_ms OR revision <= 0
               OR attempt_id = '' OR attempt_number <= 0 OR lease_id = ''
            """,
            """
            SELECT count(*) FROM task_attempts
            WHERE attempt_id = '' OR task_cid = '' OR attempt_number <= 0
               OR owner_session_id = '' OR fencing_token <= 0 OR fence_epoch <= 0
               OR started_at_ms < 0 OR revision <= 0
            """,
            """
            SELECT count(*) FROM resource_claims
            WHERE claim_id = '' OR resource_kind = '' OR resource_id = ''
               OR owner_session_id = '' OR fencing_token <= 0 OR fence_epoch <= 0
               OR acquired_at_ms < 0 OR expires_at_ms < acquired_at_ms
               OR revision <= 0 OR lease_id = ''
            """,
            """
            SELECT count(*) FROM maintenance_leases
            WHERE lease_id = '' OR scope = '' OR owner_session_id = ''
               OR fencing_token <= 0 OR fence_epoch <= 0 OR acquired_at_ms < 0
               OR expires_at_ms < acquired_at_ms OR revision <= 0
            """,
        )
        for query in basic_integrity_queries:
            row = connection.execute(query).fetchone()
            if row is None or int(row[0]) != 0:
                raise VRIFRuntimeSettlementError(
                    "coordination authority contains malformed producer identity"
                )
        dependency_orphans = connection.execute(
            """
            SELECT count(*)
            FROM task_dependencies AS dependency
            LEFT JOIN coordination_tasks AS task
              ON task.task_cid = dependency.task_cid
            LEFT JOIN coordination_tasks AS prerequisite
              ON prerequisite.task_cid = dependency.dependency_task_cid
            WHERE task.task_cid IS NULL OR prerequisite.task_cid IS NULL
            """
        ).fetchone()
        if dependency_orphans is None or int(dependency_orphans[0]) != 0:
            raise VRIFRuntimeSettlementError(
                "coordination task dependency endpoint is orphaned"
            )

        orphan_row = connection.execute(
            """
            SELECT count(*)
            FROM task_claims AS claim
            LEFT JOIN fenced_leases AS lease
              ON lease.lease_id = claim.lease_id
            LEFT JOIN task_attempts AS attempt
              ON attempt.attempt_id = claim.attempt_id
            WHERE lease.lease_id IS NULL OR attempt.attempt_id IS NULL
               OR lease.claim_id <> claim.claim_id
               OR lease.task_cid <> claim.task_cid
               OR lease.attempt_id <> claim.attempt_id
               OR lease.owner_session_id <> claim.owner_session_id
               OR lease.fencing_token <> claim.fencing_token
               OR lease.fence_epoch <> claim.fence_epoch
               OR lease.lease_kind <> 'task'
               OR lease.mode <> 'exclusive'
               OR lease.scope <> claim.task_cid
               OR lease.attempt_number <> claim.attempt_number
               OR lease.expires_at_ms <> claim.expires_at_ms
               OR attempt.task_cid <> claim.task_cid
               OR attempt.owner_session_id <> claim.owner_session_id
               OR attempt.fencing_token <> claim.fencing_token
               OR attempt.fence_epoch <> claim.fence_epoch
               OR attempt.attempt_number <> claim.attempt_number
            """
        ).fetchone()
        resource_orphans = connection.execute(
            """
            SELECT count(*)
            FROM resource_claims AS claim
            LEFT JOIN fenced_leases AS lease
              ON lease.lease_id = claim.lease_id
            WHERE lease.lease_id IS NULL
               OR lease.claim_id <> claim.claim_id
               OR lease.owner_session_id <> claim.owner_session_id
               OR lease.fencing_token <> claim.fencing_token
               OR lease.fence_epoch <> claim.fence_epoch
               OR lease.resource_kind <> claim.resource_kind
               OR lease.resource_id <> claim.resource_id
               OR lease.mode <> claim.mode
               OR lease.expires_at_ms <> claim.expires_at_ms
            """
        ).fetchone()
        maintenance_orphans = connection.execute(
            """
            SELECT count(*)
            FROM maintenance_leases AS item
            LEFT JOIN fenced_leases AS lease
              ON lease.lease_id = item.lease_id
            WHERE lease.lease_id IS NULL
               OR lease.owner_session_id <> item.owner_session_id
               OR lease.fencing_token <> item.fencing_token
               OR lease.fence_epoch <> item.fence_epoch
               OR lease.lease_kind <> 'maintenance'
               OR lease.mode <> 'exclusive'
               OR lease.scope <> item.scope
               OR lease.expires_at_ms <> item.expires_at_ms
            """
        ).fetchone()
        if any(
            row is None or int(row[0]) != 0
            for row in (orphan_row, resource_orphans, maintenance_orphans)
        ):
            raise VRIFRuntimeSettlementError(
                "coordination claim/lease/attempt projections are orphaned or unbound"
            )

        token_or_event_mismatches = connection.execute(
            """
            SELECT
                (
                    SELECT count(*)
                    FROM fenced_leases AS lease
                    LEFT JOIN token_history AS token
                      ON token.scope_key = lease.scope_key
                     AND token.fencing_token = lease.fencing_token
                     AND token.fence_epoch = lease.fence_epoch
                    WHERE token.scope_key IS NULL
                )
                +
                (
                    SELECT count(*)
                    FROM token_history AS token
                    LEFT JOIN fenced_leases AS lease
                      ON lease.scope_key = token.scope_key
                     AND lease.fencing_token = token.fencing_token
                     AND lease.fence_epoch = token.fence_epoch
                    WHERE lease.lease_id IS NULL
                )
                +
                (
                    SELECT count(*)
                    FROM lease_events AS event
                    LEFT JOIN fenced_leases AS lease
                      ON lease.lease_id = event.lease_id
                    WHERE lease.lease_id IS NULL
                       OR event.scope_key <> lease.scope_key
                       OR event.fencing_token <> lease.fencing_token
                       OR event.fence_epoch <> lease.fence_epoch
                       OR event.observed_at_ms < lease.acquired_at_ms
                )
                +
                (
                    SELECT count(*)
                    FROM fenced_leases AS lease
                    LEFT JOIN lease_events AS event
                      ON event.lease_id = lease.lease_id
                    WHERE event.event_id IS NULL
                )
            """
        ).fetchone()
        if token_or_event_mismatches is None or int(token_or_event_mismatches[0]) != 0:
            raise VRIFRuntimeSettlementError(
                "coordination fence token or event lineage differs"
            )
        task_claim_event_rows = connection.execute(
            """
            SELECT claim.claim_id, claim.attempt_id, claim.attempt_number,
                   event.event_id
            FROM task_claims AS claim
            JOIN lease_events AS event
              ON event.lease_id = claim.lease_id
             AND event.event_type = 'task_claimed'
            ORDER BY claim.attempt_id, event.event_id
            LIMIT ?
            """,
            [_HISTORY_ROW_BOUND + 1],
        ).fetchall()
        if len(task_claim_event_rows) != row_counts["task_claims"]:
            raise VRIFRuntimeSettlementError(
                "coordination task claim event lineage differs"
            )
        seen_task_claims: set[str] = set()
        for claim_id, attempt_id, attempt_number, event_id in task_claim_event_rows:
            claim_text = str(claim_id or "")
            event_body = json_rows["lease_events"].get(str(event_id or ""))
            if (
                claim_text in seen_task_claims
                or event_body
                != {
                    "claim_id": claim_text,
                    "attempt_id": str(attempt_id or ""),
                    "attempt_number": int(attempt_number or 0),
                }
            ):
                raise VRIFRuntimeSettlementError(
                    "coordination task claim event body differs"
                )
            seen_task_claims.add(claim_text)

        task_state_mismatches = connection.execute(
            """
            SELECT count(*)
            FROM task_claims AS claim
            JOIN fenced_leases AS lease ON lease.lease_id = claim.lease_id
            JOIN task_attempts AS attempt
              ON attempt.attempt_id = claim.attempt_id
            WHERE NOT (
                    (lease.state = 'accepted'
                     AND claim.state = 'accepted'
                     AND attempt.status = 'running')
                 OR (lease.state = 'released'
                     AND claim.state = 'released'
                     AND attempt.status IN ('released', 'succeeded'))
                 OR (lease.state = 'expired'
                     AND claim.state = 'expired'
                     AND attempt.status = 'expired')
                 OR (lease.state = 'completed'
                     AND claim.state = 'completed'
                     AND attempt.status = 'succeeded')
                 OR (lease.state = 'superseded'
                     AND claim.state = 'expired'
                     AND attempt.status = 'expired')
                )
               OR (
                    claim.state IN ('accepted', 'expired')
                    AND claim.released_at_ms IS NOT NULL
                )
               OR (
                    claim.state IN ('released', 'completed')
                    AND claim.released_at_ms IS NULL
                )
               OR (
                    claim.released_at_ms IS NOT NULL
                    AND claim.released_at_ms < 0
                )
               OR (
                    attempt.status = 'running'
                    AND attempt.finished_at_ms IS NOT NULL
                )
               OR (
                    attempt.status <> 'running'
                    AND attempt.finished_at_ms IS NULL
                )
               OR (
                    attempt.finished_at_ms IS NOT NULL
                    AND attempt.finished_at_ms < 0
                )
            """
        ).fetchone()
        resource_state_mismatches = connection.execute(
            """
            SELECT count(*)
            FROM resource_claims AS claim
            JOIN fenced_leases AS lease ON lease.lease_id = claim.lease_id
            WHERE NOT (
                    (lease.state = 'accepted' AND claim.state = 'accepted')
                 OR (lease.state = 'released' AND claim.state = 'released')
                 OR (lease.state = 'expired' AND claim.state = 'expired')
                 OR (lease.state = 'superseded' AND claim.state = 'expired')
                )
            """
        ).fetchone()
        maintenance_state_mismatches = connection.execute(
            """
            SELECT count(*)
            FROM maintenance_leases AS item
            JOIN fenced_leases AS lease ON lease.lease_id = item.lease_id
            WHERE NOT (
                    (lease.state = 'accepted' AND item.state = 'accepted')
                 OR (lease.state = 'released' AND item.state = 'released')
                 OR (lease.state = 'expired' AND item.state = 'expired')
                 OR (lease.state = 'superseded' AND item.state = 'expired')
                )
               OR (
                    item.state IN ('accepted', 'expired')
                    AND item.released_at_ms IS NOT NULL
                )
               OR (
                    item.state = 'released'
                    AND item.released_at_ms IS NULL
                )
               OR (
                    item.released_at_ms IS NOT NULL
                    AND item.released_at_ms < 0
                )
            """
        ).fetchone()
        if any(
            row is None or int(row[0]) != 0
            for row in (
                task_state_mismatches,
                resource_state_mismatches,
                maintenance_state_mismatches,
            )
        ):
            raise VRIFRuntimeSettlementError(
                "coordination authority state or terminal markers disagree"
            )

        _validate_task_completion_rearm_events(
            connection,
            json_rows=json_rows,
        )
        _validate_control_ready_frontier_reconciliation_events(
            connection,
            json_rows=json_rows,
        )

        barrier_rows = connection.execute(
            """
            SELECT
                completion.task_cid,
                completion.status,
                claim.claim_id,
                claim.task_cid,
                claim.owner_session_id,
                claim.fencing_token,
                claim.fence_epoch,
                claim.attempt_id,
                claim.attempt_number,
                claim.lease_id,
                claim.state,
                lease.state,
                attempt.status,
                completion.completed_at_ms,
                task.task_cid,
                task.ready
            FROM task_completions AS completion
            LEFT JOIN coordination_tasks AS task
              ON task.task_cid = completion.task_cid
            LEFT JOIN task_claims AS claim
              ON claim.claim_id = json_extract_string(
                  completion.body_json, '$.claim_id'
              )
            LEFT JOIN fenced_leases AS lease
              ON lease.lease_id = json_extract_string(
                  completion.body_json, '$.lease_id'
              )
            LEFT JOIN task_attempts AS attempt
              ON attempt.attempt_id = json_extract_string(
                  completion.body_json, '$.attempt_id'
              )
            ORDER BY completion.task_cid
            LIMIT ?
            """,
            [_HISTORY_ROW_BOUND + 1],
        ).fetchall()
        if len(barrier_rows) > _HISTORY_ROW_BOUND:
            raise VRIFRuntimeSettlementError(
                "coordination completion barriers exceed the history bound"
            )
        unresolved_barriers: list[str] = []
        for row in barrier_rows:
            task_cid = str(row[0] or "")
            completion_status = str(row[1] or "")
            body = json_rows["task_completions"].get(task_cid)
            if body is None:
                raise VRIFRuntimeSettlementError(
                    "coordination completion barrier body is unavailable"
                )
            is_preparation_barrier = (
                body.get("schema")
                == _coordination.TASK_COMPLETION_PREPARATION_SCHEMA
            )
            if completion_status == "succeeded" and not is_preparation_barrier:
                if (
                    not task_cid
                    or str(row[14] or "") != task_cid
                    or row[15] is not False
                    or int(row[13] or 0) < 0
                ):
                    raise VRIFRuntimeSettlementError(
                        "ordinary task completion is unbound or still ready"
                    )
                continue
            if completion_status == "prepared" and not is_preparation_barrier:
                raise VRIFRuntimeSettlementError(
                    "prepared completion is missing its authority identity"
                )
            expected = {
                "task_cid": task_cid,
                "claim_id": str(row[2] or ""),
                "owner_session_id": str(row[4] or ""),
                "fencing_token": int(row[5] or 0),
                "fence_epoch": int(row[6] or 0),
                "attempt_id": str(row[7] or ""),
                "attempt_number": int(row[8] or 0),
                "lease_id": str(row[9] or ""),
            }
            preparation_payload = dict(body)
            supplied_preparation_digest = preparation_payload.pop(
                "preparation_digest", None
            )
            preparation_payload.pop("control_completion", None)
            preparation_payload.pop("cross_store_guard", None)
            expected_preparation_digest = _content_id(preparation_payload)
            control_expected_revision = body.get("control_expected_revision")
            control_expected_status = body.get("control_expected_status")
            evidence_digest = body.get("evidence_digest")
            prepared_at_ms = body.get("prepared_at_ms")
            if (
                not is_preparation_barrier
                or any(body.get(key) != value for key, value in expected.items())
                or str(row[3] or "") != task_cid
                or not all(expected[key] for key in ("claim_id", "attempt_id", "lease_id"))
                or expected["fencing_token"] <= 0
                or expected["fence_epoch"] <= 0
                or expected["attempt_number"] <= 0
                or int(row[13] or 0) < 0
                or supplied_preparation_digest != expected_preparation_digest
                or type(control_expected_revision) is not int
                or control_expected_revision <= 0
                or type(control_expected_status) is not str
                or not control_expected_status.strip()
                or control_expected_status != control_expected_status.strip()
                or "\x00" in control_expected_status
                or type(evidence_digest) is not str
                or not evidence_digest.strip()
                or evidence_digest != evidence_digest.strip()
                or "\x00" in evidence_digest
                or type(prepared_at_ms) is not int
                or prepared_at_ms < 0
            ):
                raise VRIFRuntimeSettlementError(
                    "coordination completion barrier identity is malformed or orphaned"
                )
            claim_state = str(row[10] or "")
            lease_state = str(row[11] or "")
            attempt_status = str(row[12] or "")
            if claim_state != lease_state:
                raise VRIFRuntimeSettlementError(
                    "coordination completion barrier claim and lease disagree"
                )
            if claim_state == "accepted":
                expected_attempt = "running"
            elif claim_state == "expired":
                expected_attempt = "expired"
            elif claim_state in {"released", "completed"}:
                expected_attempt = "succeeded"
            else:
                raise VRIFRuntimeSettlementError(
                    "coordination completion barrier has an invalid lease state"
                )
            if attempt_status != expected_attempt:
                raise VRIFRuntimeSettlementError(
                    "coordination completion barrier attempt state disagrees"
                )
            if completion_status == "prepared":
                if claim_state not in {"accepted", "expired"}:
                    raise VRIFRuntimeSettlementError(
                        "prepared completion barrier has terminal authority"
                    )
                unresolved_barriers.append(task_cid)
            elif completion_status == "succeeded":
                if claim_state in {"accepted", "expired"}:
                    unresolved_barriers.append(task_cid)
            else:
                raise VRIFRuntimeSettlementError(
                    "coordination completion barrier has an unknown status"
                )

        ready_rows = connection.execute(
            """
            SELECT CASE WHEN octet_length(encode(task_cid)) <= 512
                        THEN task_cid END
            FROM coordination_tasks
            WHERE ready = TRUE
            ORDER BY task_cid
            LIMIT ?
            """,
            [max_active_ids + 1],
        ).fetchall()
        if any(row[0] is None for row in ready_rows):
            raise VRIFRuntimeSettlementError(
                "coordination ready task identity is malformed"
            )
        ready_task_ids = [str(row[0]) for row in ready_rows]
        active_counts = {
            "ready_coordination_tasks": len(ready_task_ids),
            "accepted_fenced_leases": status_counts["fenced_leases"]["accepted"],
            "accepted_task_claims": status_counts["task_claims"]["accepted"],
            "running_task_attempts": status_counts["task_attempts"]["running"],
            "accepted_resource_claims": status_counts["resource_claims"]["accepted"],
            "accepted_maintenance_leases": status_counts["maintenance_leases"]["accepted"],
            "unresolved_completion_barriers": len(unresolved_barriers),
        }
        expected_active = sum(active_counts.values())
        if expected_active > max_active_ids:
            raise VRIFRuntimeSettlementError(
                "coordination active identity bound was exceeded"
            )
        active_rows = connection.execute(
            """
            SELECT kind, identity
            FROM (
                SELECT 'fenced_lease' AS kind, lease_id AS identity
                FROM fenced_leases WHERE state = 'accepted'
                UNION ALL
                SELECT 'coordination_task', task_cid
                FROM coordination_tasks WHERE ready = TRUE
                UNION ALL
                SELECT 'task_claim', claim_id
                FROM task_claims WHERE state = 'accepted'
                UNION ALL
                SELECT 'task_attempt', attempt_id
                FROM task_attempts WHERE status = 'running'
                UNION ALL
                SELECT 'resource_claim', claim_id
                FROM resource_claims WHERE state = 'accepted'
                UNION ALL
                SELECT 'maintenance_lease', lease_id
                FROM maintenance_leases WHERE state = 'accepted'
            )
            ORDER BY kind, identity
            LIMIT ?
            """,
            [max_active_ids + 1],
        ).fetchall()
        active_ids = [f"{row[0]}:{row[1]}" for row in active_rows]
        active_ids.extend(f"completion_barrier:{item}" for item in unresolved_barriers)
        active_ids.sort()
        if len(active_ids) != expected_active:
            raise VRIFRuntimeSettlementError(
                "coordination active counts and identities disagree"
            )

        connection.execute("COMMIT")
        transaction_open = False
    except BaseException as exc:
        if connection is not None and transaction_open:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        if isinstance(exc, VRIFRuntimeSettlementError):
            raise
        raise VRIFRuntimeSettlementError(
            "coordination sidecar is unreadable or malformed"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    _require_store_unchanged(database_path, store_identity)
    metadata = {
        "interface": _coordination.DATABASE_COORDINATOR_INTERFACE,
        "schema": _coordination.DATABASE_COORDINATION_SCHEMA,
    }
    snapshot: dict[str, Any] = {
        "schema": VRIF_COORDINATION_SETTLEMENT_SCHEMA,
        "store": dict(store_identity),
        "metadata": metadata,
        "metadata_cid": _content_id(metadata),
        "row_counts": row_counts,
        "status_counts": status_counts,
        "active_counts": active_counts,
        "active_count": len(active_ids),
        "active_ids": active_ids,
    }
    snapshot["snapshot_cid"] = _content_id(snapshot)
    return snapshot


def _validate_execution_schema(connection: Any) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position
        """
    ).fetchall()
    observed: dict[str, list[tuple[str, str, str]]] = {}
    for table, column, data_type, nullable in rows:
        observed.setdefault(str(table), []).append(
            (str(column), str(data_type).upper(), str(nullable).upper())
        )
    if set(observed) != set(_EXECUTION_COLUMNS) or any(
        tuple(observed.get(table, ())) != expected
        for table, expected in _EXECUTION_COLUMNS.items()
    ):
        raise VRIFRuntimeSettlementError(
            "execution sidecar table or ordered-column inventory differs"
        )
    index_rows = connection.execute(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE schema_name = 'main'
        ORDER BY index_name
        """
    ).fetchall()
    if {str(row[0]) for row in index_rows} != _EXECUTION_INDEXES:
        raise VRIFRuntimeSettlementError("execution sidecar index inventory differs")
    metadata_rows = connection.execute(
        """
        SELECT
            CASE WHEN octet_length(encode(key)) <= 128 THEN key END,
            CASE WHEN octet_length(encode(value)) <= 4096 THEN value END
        FROM daemon_execution_metadata
        ORDER BY key
        LIMIT ?
        """,
        [len(_EXECUTION_METADATA_KEYS) + 1],
    ).fetchall()
    if (
        len(metadata_rows) != len(_EXECUTION_METADATA_KEYS)
        or any(row[0] is None or row[1] is None for row in metadata_rows)
    ):
        raise VRIFRuntimeSettlementError(
            "execution sidecar metadata inventory is malformed"
        )
    metadata = {str(key): str(value) for key, value in metadata_rows}
    if set(metadata) != _EXECUTION_METADATA_KEYS:
        raise VRIFRuntimeSettlementError(
            "execution sidecar metadata inventory differs"
        )
    if (
        metadata["interface"] != "DatabaseImplementationDaemon@1"
        or metadata["schema"]
        != "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1"
        or metadata["authority_mode"] != "quack"
        or metadata["state_schema_revision"] != VRIF_CONTROL_SCHEMA_REVISION
        or _PROCESS_INSTANCE.fullmatch(metadata["process_instance_id"]) is None
        or metadata["control_schema_profile_id"] != ""
        or metadata["control_schema_fingerprint"] != ""
    ):
        raise VRIFRuntimeSettlementError(
            "execution sidecar metadata binding differs"
        )
    return metadata


def _read_execution_snapshot(
    database_path: Path,
    *,
    store_identity: Mapping[str, Any],
    expected_owner_session_id: str,
    max_active_ids: int,
) -> dict[str, Any]:
    connection: Any | None = None
    transaction_open = False
    try:
        import duckdb

        connection = connect_duckdb_with_policy(
            duckdb,
            database_path,
            read_only=True,
        )
        connection.execute("BEGIN TRANSACTION")
        transaction_open = True
        metadata = _validate_execution_schema(connection)
        for table, columns in {
            "daemon_execution_metadata": ("key",),
            "database_task_attempts": ("attempt_id",),
            "attempt_phases": ("attempt_id", "phase"),
            "provider_invocations": ("invocation_id",),
            "effect_claims": ("effect_id",),
            "daemon_execution_events": ("event_id",),
        }.items():
            _require_unique_rows(connection, table=table, columns=columns)
        if metadata["logical_owner_session_id"] != expected_owner_session_id:
            raise VRIFRuntimeSettlementError(
                "execution sidecar logical owner identity differs"
            )
        row_counts = _row_counts(connection, tuple(sorted(_EXECUTION_COLUMNS)))
        if any(count > _HISTORY_ROW_BOUND for count in row_counts.values()):
            raise VRIFRuntimeSettlementError(
                "execution sidecar exceeds the bounded history profile"
            )
        status_counts = _status_counts(
            connection,
            table="database_task_attempts",
            column="status",
            allowed=_EXECUTION_STATES,
        )
        phase_counts = _status_counts(
            connection,
            table="attempt_phases",
            column="phase",
            allowed=_EXECUTION_PHASES,
        )

        attempt_bodies = dict(
            _bounded_json_rows(
                connection,
                table="database_task_attempts",
                identity_column="attempt_id",
                json_column="body_json",
            )
        )
        phase_bodies = dict(
            _bounded_json_rows(
                connection,
                table="attempt_phases",
                identity_column="attempt_id || ':' || phase",
                json_column="body_json",
            )
        )
        provider_bodies = dict(
            _bounded_json_rows(
                connection,
                table="provider_invocations",
                identity_column="invocation_id",
                json_column="result_json",
            )
        )
        effect_bodies = dict(
            _bounded_json_rows(
                connection,
                table="effect_claims",
                identity_column="effect_id",
                json_column="result_json",
            )
        )
        _bounded_json_rows(
            connection,
            table="daemon_execution_events",
            identity_column="event_id",
            json_column="body_json",
        )

        attempt_rows = connection.execute(
            """
            SELECT attempt_id, claim_id, task_cid, attempt_number,
                   owner_session_id, fencing_token, fence_epoch, lease_id,
                   committed_phase, status, started_at_ms, finished_at_ms,
                   revision
            FROM database_task_attempts
            ORDER BY attempt_id
            LIMIT ?
            """,
            [_HISTORY_ROW_BOUND + 1],
        ).fetchall()
        attempts: dict[str, dict[str, Any]] = {}
        for row in attempt_rows:
            attempt_id = str(row[0] or "")
            attempt = {
                "attempt_id": attempt_id,
                "claim_id": str(row[1] or ""),
                "task_cid": str(row[2] or ""),
                "attempt_number": int(row[3] or 0),
                "owner_session_id": str(row[4] or ""),
                "fencing_token": int(row[5] or 0),
                "fence_epoch": int(row[6] or 0),
                "lease_id": str(row[7] or ""),
                "committed_phase": str(row[8] or ""),
                "status": str(row[9] or ""),
                "started_at_ms": int(row[10] or 0),
                "finished_at_ms": None if row[11] is None else int(row[11]),
                "revision": int(row[12] or 0),
            }
            if (
                not attempt_id
                or any(
                    not attempt[name]
                    for name in (
                        "claim_id",
                        "task_cid",
                        "owner_session_id",
                        "lease_id",
                    )
                )
                or attempt["owner_session_id"] != expected_owner_session_id
                or attempt["attempt_number"] <= 0
                or attempt["fencing_token"] <= 0
                or attempt["fence_epoch"] <= 0
                or attempt["started_at_ms"] < 0
                or attempt["revision"] <= 0
                or attempt_id not in attempt_bodies
            ):
                raise VRIFRuntimeSettlementError(
                    "execution attempt identity is malformed or unbound"
                )
            status_value = attempt["status"]
            phase_value = attempt["committed_phase"]
            finished = attempt["finished_at_ms"]
            consistent = (
                status_value == "running"
                and phase_value
                in {"claimed", "context", "provider", "effect", "validation"}
                and finished is None
            ) or (
                status_value == "succeeded"
                and phase_value == "complete"
                and finished is not None
            ) or (
                status_value == "failed"
                and phase_value == "failed"
                and finished is not None
            ) or (
                status_value == "blocked"
                and phase_value == "blocked"
                and finished is not None
            )
            if not consistent or (finished is not None and finished < 0):
                raise VRIFRuntimeSettlementError(
                    "execution attempt status, phase, and finish fields disagree"
                )
            attempts[attempt_id] = attempt

        phase_rows = connection.execute(
            """
            SELECT attempt_id, phase, committed_at_ms, fencing_token,
                   fence_epoch, revision
            FROM attempt_phases
            ORDER BY attempt_id, phase
            LIMIT ?
            """,
            [_HISTORY_ROW_BOUND + 1],
        ).fetchall()
        phases_by_attempt: dict[str, set[str]] = {}
        for row in phase_rows:
            attempt_id = str(row[0] or "")
            phase = str(row[1] or "")
            attempt = attempts.get(attempt_id)
            if (
                attempt is None
                or phase not in _EXECUTION_PHASES
                or f"{attempt_id}:{phase}" not in phase_bodies
                or int(row[2] or 0) < 0
                or int(row[3] or 0) != attempt["fencing_token"]
                or int(row[4] or 0) != attempt["fence_epoch"]
                or int(row[5] or 0) <= 0
                or int(row[5]) > attempt["revision"]
            ):
                raise VRIFRuntimeSettlementError(
                    "execution phase is orphaned or fence-unbound"
                )
            phases_by_attempt.setdefault(attempt_id, set()).add(phase)
        for attempt_id, attempt in attempts.items():
            phases = phases_by_attempt.get(attempt_id, set())
            if "claimed" not in phases or attempt["committed_phase"] not in phases:
                raise VRIFRuntimeSettlementError(
                    "execution attempt is missing its claimed or committed phase"
                )

        provider_rows = connection.execute(
            """
            SELECT invocation_id, attempt_id, task_cid, idempotency_key,
                   owner_session_id, recorded_at_ms
            FROM provider_invocations
            ORDER BY invocation_id
            LIMIT ?
            """,
            [_HISTORY_ROW_BOUND + 1],
        ).fetchall()
        providers: list[tuple[str, str]] = []
        for row in provider_rows:
            invocation_id = str(row[0] or "")
            attempt_id = str(row[1] or "")
            attempt = attempts.get(attempt_id)
            if (
                not invocation_id
                or attempt is None
                or invocation_id not in provider_bodies
                or str(row[2] or "") != attempt["task_cid"]
                or not str(row[3] or "")
                or str(row[4] or "") != attempt["owner_session_id"]
                or int(row[5] or 0) < 0
            ):
                raise VRIFRuntimeSettlementError(
                    "execution provider invocation is orphaned or unbound"
                )
            providers.append((invocation_id, attempt_id))

        effect_rows = connection.execute(
            """
            SELECT effect_id, attempt_id, task_cid, effect_key,
                   idempotency_key, owner_session_id, recorded_at_ms
            FROM effect_claims
            ORDER BY effect_id
            LIMIT ?
            """,
            [_HISTORY_ROW_BOUND + 1],
        ).fetchall()
        effects: list[tuple[str, str]] = []
        for row in effect_rows:
            effect_id = str(row[0] or "")
            attempt_id = str(row[1] or "")
            attempt = attempts.get(attempt_id)
            if (
                not effect_id
                or attempt is None
                or effect_id not in effect_bodies
                or str(row[2] or "") != attempt["task_cid"]
                or not str(row[3] or "")
                or not str(row[4] or "")
                or str(row[5] or "") != attempt["owner_session_id"]
                or int(row[6] or 0) < 0
            ):
                raise VRIFRuntimeSettlementError(
                    "execution effect claim is orphaned or unbound"
                )
            effects.append((effect_id, attempt_id))

        event_orphans = connection.execute(
            """
            SELECT count(*)
            FROM daemon_execution_events AS event
            LEFT JOIN database_task_attempts AS attempt
              ON attempt.attempt_id = event.attempt_id
            WHERE event.attempt_id <> ''
              AND (
                  attempt.attempt_id IS NULL
                  OR event.task_cid <> attempt.task_cid
              )
            """
        ).fetchone()
        if event_orphans is None or int(event_orphans[0]) != 0:
            raise VRIFRuntimeSettlementError(
                "execution event is orphaned or task-unbound"
            )

        running = {
            attempt_id
            for attempt_id, attempt in attempts.items()
            if attempt["status"] == "running"
        }
        active_ids = [f"execution_attempt:{item}" for item in sorted(running)]
        active_phase_ids = sorted(
            f"execution_phase:{attempt_id}:{phase}"
            for attempt_id, phases in phases_by_attempt.items()
            if attempt_id in running
            for phase in phases
        )
        active_provider_ids = sorted(
            f"provider_invocation:{identity}"
            for identity, attempt_id in providers
            if attempt_id in running
        )
        active_effect_ids = sorted(
            f"effect_claim:{identity}"
            for identity, attempt_id in effects
            if attempt_id in running
        )
        active_ids.extend(active_phase_ids)
        active_ids.extend(active_provider_ids)
        active_ids.extend(active_effect_ids)
        active_ids.sort()
        if len(active_ids) > max_active_ids:
            raise VRIFRuntimeSettlementError(
                "execution active identity bound was exceeded"
            )
        active_counts = {
            "running_attempts": len(running),
            "running_phase_rows": len(active_phase_ids),
            "running_provider_invocations": len(active_provider_ids),
            "running_effect_claims": len(active_effect_ids),
        }
        if sum(active_counts.values()) != len(active_ids):
            raise VRIFRuntimeSettlementError(
                "execution active counts and identities disagree"
            )

        connection.execute("COMMIT")
        transaction_open = False
    except BaseException as exc:
        if connection is not None and transaction_open:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        if isinstance(exc, VRIFRuntimeSettlementError):
            raise
        raise VRIFRuntimeSettlementError(
            "execution sidecar is unreadable or malformed"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    _require_store_unchanged(database_path, store_identity)
    snapshot: dict[str, Any] = {
        "schema": VRIF_EXECUTION_SETTLEMENT_SCHEMA,
        "store": dict(store_identity),
        "metadata": metadata,
        "metadata_cid": _content_id(metadata),
        "row_counts": row_counts,
        "status_counts": status_counts,
        "phase_counts": phase_counts,
        "active_counts": active_counts,
        "active_count": len(active_ids),
        "active_ids": active_ids,
    }
    snapshot["snapshot_cid"] = _content_id(snapshot)
    return snapshot


def _configured_retired_store_identity(
    entry: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[Path, dict[str, Any]]:
    database_path = repository_root / str(entry["database_path"])
    wal_path = repository_root / str(entry["wal_path"])
    store = _store_identity(database_path)
    wal = store["wal"]
    if (
        store["database"]["size_bytes"] != entry["database_size_bytes"]
        or not isinstance(wal, Mapping)
        or wal.get("state") != "present"
        or wal.get("file", {}).get("size_bytes") != entry["wal_size_bytes"]
        or wal.get("path") != str(wal_path.absolute())
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination snapshot file identity differs from config"
        )
    database_raw = _read_stable_regular_bytes(
        database_path,
        expected=store["database"],
        label="retired coordination database",
        max_bytes=_RETIRED_STORE_MAX_BYTES,
    )
    wal_raw = _read_stable_regular_bytes(
        wal_path,
        expected=wal["file"],
        label="retired coordination WAL",
        max_bytes=_RETIRED_STORE_MAX_BYTES,
    )
    if (
        "sha256:" + hashlib.sha256(database_raw).hexdigest()
        != entry["database_sha256"]
        or "sha256:" + hashlib.sha256(wal_raw).hexdigest()
        != entry["wal_sha256"]
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination snapshot content differs from config"
        )
    return database_path, store


def _read_coordination_lineage_rows(
    database_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    connection: Any | None = None
    transaction_open = False
    try:
        import duckdb

        connection = connect_duckdb_with_policy(
            duckdb,
            database_path,
            read_only=True,
        )
        connection.execute("BEGIN TRANSACTION")
        transaction_open = True
        rows = connection.execute(
            """
            SELECT claim.attempt_id, claim.claim_id, claim.task_cid,
                   claim.attempt_number, claim.owner_session_id,
                   claim.fencing_token, claim.fence_epoch, claim.lease_id,
                   claim.state, attempt.status, lease.state
            FROM task_claims AS claim
            JOIN task_attempts AS attempt
              ON attempt.attempt_id = claim.attempt_id
            JOIN fenced_leases AS lease ON lease.lease_id = claim.lease_id
            ORDER BY claim.attempt_id
            LIMIT ?
            """,
            [_HISTORY_ROW_BOUND + 1],
        ).fetchall()
        ready_rows = connection.execute(
            """
            SELECT task_cid, task_id
            FROM coordination_tasks
            WHERE ready = TRUE
            ORDER BY task_cid, task_id
            LIMIT ?
            """,
            [_HISTORY_ROW_BOUND + 1],
        ).fetchall()
        connection.execute("COMMIT")
        transaction_open = False
    except BaseException as exc:
        if connection is not None and transaction_open:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        if isinstance(exc, VRIFRuntimeSettlementError):
            raise
        raise VRIFRuntimeSettlementError(
            "coordination lineage snapshot is unreadable"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    if len(rows) > _HISTORY_ROW_BOUND or len(ready_rows) > _HISTORY_ROW_BOUND:
        raise VRIFRuntimeSettlementError(
            "coordination lineage snapshot exceeds its history bound"
        )
    bindings = [
        {
            "attempt_id": str(row[0]),
            "claim_id": str(row[1]),
            "task_cid": str(row[2]),
            "attempt_number": int(row[3]),
            "owner_session_id": str(row[4]),
            "fencing_token": int(row[5]),
            "fence_epoch": int(row[6]),
            "lease_id": str(row[7]),
            "coordination_claim_state": str(row[8]),
            "coordination_attempt_status": str(row[9]),
            "coordination_lease_state": str(row[10]),
        }
        for row in rows
    ]
    ready_tasks = [
        {"task_cid": str(row[0]), "task_id": str(row[1])}
        for row in ready_rows
    ]
    if (
        [row["attempt_id"] for row in bindings]
        != sorted({row["attempt_id"] for row in bindings})
        or any(
            not row["task_cid"] or not row["task_id"]
            for row in ready_tasks
        )
        or ready_tasks
        != sorted(ready_tasks, key=lambda row: (row["task_cid"], row["task_id"]))
    ):
        raise VRIFRuntimeSettlementError(
            "coordination lineage identity order differs"
        )
    return bindings, ready_tasks


def _read_retired_coordination_lineage(
    entry: Mapping[str, Any],
    *,
    config_ordinal: int,
    repository_root: Path,
    policy_lock: Mapping[str, Any],
    max_active_ids: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], tuple[Path, dict[str, Any]]]:
    database_path, store = _configured_retired_store_identity(
        entry,
        repository_root=repository_root,
    )
    coordination = _read_coordination_snapshot(
        database_path,
        store_identity=store,
        max_active_ids=max(max_active_ids, _HISTORY_ROW_BOUND),
    )
    active_counts = coordination["active_counts"]
    if any(
        count != 0
        for name, count in active_counts.items()
        if name != "ready_coordination_tasks"
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination snapshot retains active authority"
        )
    authority_bindings, ready_tasks = _read_coordination_lineage_rows(database_path)
    terminal_triples = {
        ("released", "released", "released"),
        ("released", "succeeded", "released"),
        ("expired", "expired", "expired"),
        ("expired", "expired", "superseded"),
        ("completed", "succeeded", "completed"),
    }
    if any(
        (
            row["coordination_claim_state"],
            row["coordination_attempt_status"],
            row["coordination_lease_state"],
        )
        not in terminal_triples
        for row in authority_bindings
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination snapshot contains nonterminal authority"
        )
    authority_by_attempt = {
        row["attempt_id"]: row for row in authority_bindings
    }
    configured_attempt_ids = list(entry["terminal_execution_attempt_ids"])
    if any(attempt_id not in authority_by_attempt for attempt_id in configured_attempt_ids):
        raise VRIFRuntimeSettlementError(
            "retired coordination snapshot lacks configured terminal authority"
        )
    ready_task_cids = [row["task_cid"] for row in ready_tasks]
    expected_ready_ids = sorted(
        f"coordination_task:{task_cid}" for task_cid in ready_task_cids
    )
    if (
        ready_task_cids != sorted(set(ready_task_cids))
        or coordination["active_count"] != len(ready_tasks)
        or coordination["active_ids"] != expected_ready_ids
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination ready-task lineage differs"
        )
    lineage: dict[str, Any] = {
        "schema": VRIF_RETIRED_COORDINATION_LINEAGE_SCHEMA,
        "config_ordinal": config_ordinal,
        "lane_index": entry["lane_index"],
        "database_path": str(database_path.absolute()),
        "wal_path": str(Path(str(database_path) + ".wal").absolute()),
        "configured_content": {
            "database_size_bytes": entry["database_size_bytes"],
            "database_sha256": entry["database_sha256"],
            "wal_size_bytes": entry["wal_size_bytes"],
            "wal_sha256": entry["wal_sha256"],
        },
        "policy_lock": dict(policy_lock),
        "coordination": coordination,
        "historical_ready_tasks": ready_tasks,
        "historical_ready_tasks_cid": _content_id(ready_tasks),
        "authority_bindings": authority_bindings,
        "authority_binding_count": len(authority_bindings),
        "authority_bindings_cid": _content_id(authority_bindings),
        "admitted_terminal_execution_attempt_ids": configured_attempt_ids,
        "admitted_terminal_execution_attempt_ids_cid": _content_id(
            configured_attempt_ids
        ),
    }
    return lineage, authority_by_attempt, (database_path, store)


def _read_execution_authority_rows(
    execution_path: Path,
) -> list[dict[str, Any]]:
    connection: Any | None = None
    transaction_open = False
    try:
        import duckdb

        connection = connect_duckdb_with_policy(
            duckdb,
            execution_path,
            read_only=True,
        )
        connection.execute("BEGIN TRANSACTION")
        transaction_open = True
        rows = connection.execute(
            """
            SELECT attempt_id, claim_id, task_cid, attempt_number,
                   owner_session_id, fencing_token, fence_epoch, lease_id,
                   status
            FROM database_task_attempts
            ORDER BY attempt_id
            LIMIT ?
            """,
            [_HISTORY_ROW_BOUND + 1],
        ).fetchall()
        connection.execute("COMMIT")
        transaction_open = False
    except BaseException as exc:
        if connection is not None and transaction_open:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        if isinstance(exc, VRIFRuntimeSettlementError):
            raise
        raise VRIFRuntimeSettlementError(
            "execution authority snapshot is unreadable"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    if len(rows) > _HISTORY_ROW_BOUND:
        raise VRIFRuntimeSettlementError(
            "execution authority snapshot exceeds its history bound"
        )
    return [
        {
            "attempt_id": str(row[0]),
            "claim_id": str(row[1]),
            "task_cid": str(row[2]),
            "attempt_number": int(row[3]),
            "owner_session_id": str(row[4]),
            "fencing_token": int(row[5]),
            "fence_epoch": int(row[6]),
            "lease_id": str(row[7]),
            "execution_status": str(row[8]),
        }
        for row in rows
    ]


def _read_lane_cross_store_binding(
    coordination_path: Path,
    execution_path: Path,
    *,
    retired_lineages: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]
    ] = (),
) -> dict[str, Any]:
    """Bind each execution attempt to current or exact configured retired authority."""

    current_rows, _ready = _read_coordination_lineage_rows(coordination_path)
    execution_rows = _read_execution_authority_rows(execution_path)
    current_by_attempt = {row["attempt_id"]: row for row in current_rows}
    retired_by_attempt: dict[str, tuple[int, Mapping[str, Any]]] = {}
    configured_retired_ids: list[str] = []
    for ordinal, (lineage, authority_rows) in enumerate(retired_lineages):
        admitted = list(lineage["admitted_terminal_execution_attempt_ids"])
        configured_retired_ids.extend(admitted)
        for attempt_id, row in authority_rows.items():
            if attempt_id in retired_by_attempt:
                raise VRIFRuntimeSettlementError(
                    "retired coordination snapshots contain duplicate authority"
                )
            retired_by_attempt[attempt_id] = (ordinal, row)
    overlap = set(current_by_attempt).intersection(retired_by_attempt)
    if overlap:
        raise VRIFRuntimeSettlementError(
            "current and retired coordination authority overlap"
        )
    configured_retired_ids = sorted(configured_retired_ids)
    if configured_retired_ids != sorted(set(configured_retired_ids)):
        raise VRIFRuntimeSettlementError(
            "retired coordination admission inventory overlaps"
        )
    allowed_current_statuses = {
        "running": {"running", "expired"},
        "succeeded": {"running", "expired", "succeeded"},
        "failed": {"running", "released", "expired"},
        "blocked": {"running", "released", "expired"},
    }
    bindings: list[dict[str, Any]] = []
    retired_matched_ids: list[str] = []
    for execution_row in execution_rows:
        execution_identity = (
            execution_row["attempt_id"],
            execution_row["claim_id"],
            execution_row["task_cid"],
            execution_row["attempt_number"],
            execution_row["owner_session_id"],
            execution_row["fencing_token"],
            execution_row["fence_epoch"],
            execution_row["lease_id"],
        )
        attempt_id = execution_identity[0]
        execution_status = execution_row["execution_status"]
        authority_source = "current"
        retired_ordinal = -1
        coordination_row = current_by_attempt.get(attempt_id)
        if coordination_row is None:
            retired = retired_by_attempt.get(attempt_id)
            if (
                retired is None
                or attempt_id not in configured_retired_ids
                or execution_status == "running"
            ):
                raise VRIFRuntimeSettlementError(
                    "execution attempt is missing exact coordination authority"
                )
            retired_ordinal, coordination_row = retired
            authority_source = "retired"
            retired_matched_ids.append(attempt_id)
        coordination_identity = tuple(
            coordination_row[name]
            for name in (
                "attempt_id",
                "claim_id",
                "task_cid",
                "attempt_number",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "lease_id",
            )
        )
        coordination_attempt_status = str(
            coordination_row["coordination_attempt_status"]
        )
        claim_state = str(coordination_row["coordination_claim_state"])
        lease_state = str(coordination_row["coordination_lease_state"])
        if authority_source == "current":
            state_valid = (
                coordination_attempt_status
                in allowed_current_statuses.get(execution_status, set())
                and not (
                    coordination_attempt_status == "succeeded"
                    and execution_status != "succeeded"
                )
            )
        elif execution_status == "succeeded":
            state_valid = (
                coordination_attempt_status == "succeeded"
                and (claim_state, lease_state)
                in {("released", "released"), ("completed", "completed")}
            )
        else:
            state_valid = (
                execution_status in {"failed", "blocked"}
                and (
                    (claim_state, coordination_attempt_status, lease_state)
                    in {
                        ("released", "released", "released"),
                        ("expired", "expired", "expired"),
                        ("expired", "expired", "superseded"),
                    }
                )
            )
        if execution_identity != coordination_identity or not state_valid:
            raise VRIFRuntimeSettlementError(
                "execution attempt differs from its coordination authority"
            )
        bindings.append(
            {
                "attempt_id": attempt_id,
                "claim_id": execution_identity[1],
                "task_cid": execution_identity[2],
                "attempt_number": execution_identity[3],
                "owner_session_id": execution_identity[4],
                "fencing_token": execution_identity[5],
                "fence_epoch": execution_identity[6],
                "lease_id": execution_identity[7],
                "execution_status": execution_status,
                "coordination_claim_state": claim_state,
                "coordination_attempt_status": coordination_attempt_status,
                "coordination_lease_state": lease_state,
                "authority_source": authority_source,
                "retired_snapshot_ordinal": retired_ordinal,
            }
        )
    retired_matched_ids.sort()
    if retired_matched_ids != configured_retired_ids:
        raise VRIFRuntimeSettlementError(
            "current unmatched execution set differs from retired admission config"
        )
    current_matched_count = len(bindings) - len(retired_matched_ids)
    snapshot: dict[str, Any] = {
        "schema": VRIF_LANE_CROSS_STORE_BINDING_SCHEMA,
        "coordination_task_claim_count": len(current_rows),
        "retired_coordination_task_claim_count": sum(
            len(authority_rows) for _lineage, authority_rows in retired_lineages
        ),
        "execution_attempt_count": len(execution_rows),
        "current_matched_execution_attempt_count": current_matched_count,
        "retired_matched_execution_attempt_count": len(retired_matched_ids),
        "matched_execution_attempt_count": len(bindings),
        "retired_matched_execution_attempt_ids": retired_matched_ids,
        "retired_matched_execution_attempt_ids_cid": _content_id(
            retired_matched_ids
        ),
        "current_coordination_authority_bindings": current_rows,
        "current_coordination_authority_bindings_cid": _content_id(current_rows),
        "execution_authority_rows": execution_rows,
        "execution_authority_rows_cid": _content_id(execution_rows),
        "bindings": bindings,
        "bindings_cid": _content_id(bindings),
    }
    snapshot["snapshot_cid"] = _content_id(snapshot)
    return snapshot


def _lane_runtime_paths(state_path: Path, index: int) -> dict[str, Path]:
    lane_directory = state_path / f"lane-{index}"
    _require_real_directory(lane_directory, label=f"VRIF lane {index} directory")
    logical = lane_directory / "quack-lane-control.duckdb"
    coordination = lane_directory / "quack-lane-control.coordination.duckdb"
    execution = lane_directory / "quack-lane-control.execution.duckdb"
    return {
        "directory": lane_directory,
        "logical": logical,
        "coordination": coordination,
        "execution": execution,
        "writer_lock": execution.with_name(f".{execution.name}.writer.lock"),
        "coordination_lock": coordination.with_name(f".{coordination.name}.lock"),
        "execution_lock": execution.with_name(f".{execution.name}.lock"),
    }


def _require_current_master_pid_projection(
    path: Path,
    *,
    expected_pid: int,
) -> dict[str, Any]:
    """Bind terminal maintenance to the still-live owning master process."""

    if (
        type(expected_pid) is not int
        or expected_pid <= 0
        or expected_pid != os.getpid()
    ):
        raise VRIFRuntimeSettlementError(
            "terminal checkpoint master PID is not the current process"
        )
    identity = _file_identity(
        path,
        label="VRIF terminal checkpoint master PID marker",
        require_nonempty=True,
    )
    expected_payload = f"{expected_pid}\n".encode("ascii")
    if (
        identity["uid"] != os.geteuid()
        or identity["mode"] != 0o600
        or identity["size_bytes"] != len(expected_payload)
    ):
        raise VRIFRuntimeSettlementError(
            "terminal checkpoint master PID marker ownership differs"
        )
    payload = _read_stable_regular_bytes(
        path,
        expected=identity,
        label="VRIF terminal checkpoint master PID marker",
        max_bytes=_PID_MAX_BYTES,
    )
    if payload != expected_payload:
        raise VRIFRuntimeSettlementError(
            "terminal checkpoint master PID marker identity differs"
        )
    return identity


def _checkpoint_vrif_sidecar(database_path: Path) -> dict[str, Any]:
    """Checkpoint one already-locked local sidecar without deleting its WAL."""

    before = _store_identity(database_path)
    connection: Any | None = None
    try:
        import duckdb

        connection = connect_duckdb_with_policy(duckdb, database_path)
        connection.execute("CHECKPOINT")
    except Exception as exc:
        if isinstance(exc, VRIFRuntimeSettlementError):
            raise
        raise VRIFRuntimeSettlementError(
            f"terminal sidecar checkpoint failed: {database_path.name}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()

    after_database = _file_identity(
        database_path,
        label="checkpointed sidecar database",
        require_nonempty=True,
    )
    stable_identity_fields = ("path", "device", "inode", "mode", "link_count", "uid")
    if any(
        after_database[field] != before["database"][field]
        for field in stable_identity_fields
    ):
        raise VRIFRuntimeSettlementError(
            "checkpointed sidecar database identity changed"
        )
    wal_after = _optional_wal_identity(database_path)
    if wal_after["state"] != "absent":
        raise VRIFRuntimeSettlementError(
            "terminal sidecar WAL remained after checkpoint"
        )
    return {
        "database_path": str(database_path),
        "wal_before": before["wal"]["state"],
        "wal_after": "absent",
        "checkpoint_executed": True,
    }


def checkpoint_vrif_terminal_sidecars(
    config_path: Path | str,
    *,
    repository_root: Path | str,
    target_branch: str,
    master_pid_path: Path | str,
    expected_master_pid: int,
    max_active_ids: int = 256,
    lock_timeout_seconds: float = 1.0,
) -> dict[str, Any]:
    """Checkpoint exact zero-active VRIF sidecars after every lane is fenced.

    This is a bounded terminal-shutdown operation, not a repair primitive.  It
    takes the same master, lifetime-writer, and policy locks as runtime
    settlement, validates all current sidecars through read-only settlement
    readers, and only then issues DuckDB's ``CHECKPOINT``.  Logical rows are
    never changed and WAL files are never unlinked by Python.
    """

    timeout = _validate_public_inputs(
        target_repository_id=VRIF_TARGET_REPOSITORY_ID,
        target_branch=target_branch,
        owner_generation=1,
        max_active_ids=max_active_ids,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    root = _absolute_lexical(repository_root)
    _require_real_directory(root, label="repository_root")
    _config, profile = _read_config(
        config_path,
        repository_root=root,
        target_branch=target_branch,
    )
    state_path = Path(profile["state_path"])
    expected_master_path = state_path / "configured-board-master.pid"
    supplied_master_path = _absolute_lexical(master_pid_path)
    if supplied_master_path != expected_master_path:
        raise VRIFRuntimeSettlementError(
            "terminal checkpoint master PID path differs from the VRIF profile"
        )
    _contained_existing_path(
        root,
        supplied_master_path,
        label="VRIF terminal checkpoint master PID marker",
        kind="file",
    )
    master_fence_path = state_path / ".configured-board-master.pid.update.lock"
    lane_paths = [_lane_runtime_paths(state_path, index) for index in range(4)]

    with _hold_existing_lock(
        master_fence_path,
        label="VRIF master launch fence",
        timeout_seconds=timeout,
    ):
        _require_config_unchanged(profile)
        master_identity = _require_current_master_pid_projection(
            supplied_master_path,
            expected_pid=expected_master_pid,
        )
        with ExitStack() as lane_locks:
            for index, paths in enumerate(lane_paths):
                lane_locks.enter_context(
                    _hold_existing_lock(
                        paths["writer_lock"],
                        label=(
                            f"VRIF lane {index} execution lifetime writer lock"
                        ),
                        timeout_seconds=timeout,
                    )
                )
            policy_specs = sorted(
                (
                    (paths["coordination_lock"], index, "coordination")
                    for index, paths in enumerate(lane_paths)
                ),
                key=lambda item: str(item[0]),
            ) + sorted(
                (
                    (paths["execution_lock"], index, "execution")
                    for index, paths in enumerate(lane_paths)
                ),
                key=lambda item: str(item[0]),
            )
            # Preserve the settlement lock order globally, not by lock kind.
            policy_specs.sort(key=lambda item: str(item[0]))
            for lock_path, index, kind in policy_specs:
                lane_locks.enter_context(
                    _hold_existing_lock(
                        lock_path,
                        label=f"VRIF lane {index} {kind} policy lock",
                        timeout_seconds=timeout,
                    )
                )

            lane_pid_observations = [
                _pid_observation(
                    state_path
                    / f"lane-{index}"
                    / f"vrif_lane_{index}_supervisor.pid",
                    label=f"VRIF lane {index} supervisor PID marker",
                )
                for index in range(4)
            ]
            active_coordination = 0
            active_execution = 0
            for paths in lane_paths:
                coordination_store = _store_identity(paths["coordination"])
                execution_store = _store_identity(paths["execution"])
                owner_id = _logical_owner_id(
                    logical_database_path=paths["logical"],
                    coordination_path=paths["logical"],
                    execution_path=paths["execution"],
                )
                coordination = _read_coordination_snapshot(
                    paths["coordination"],
                    store_identity=coordination_store,
                    max_active_ids=max_active_ids,
                )
                execution = _read_execution_snapshot(
                    paths["execution"],
                    store_identity=execution_store,
                    expected_owner_session_id=owner_id,
                    max_active_ids=max_active_ids,
                )
                active_coordination += int(coordination["active_count"])
                active_execution += int(execution["active_count"])
            if active_coordination != 0 or active_execution != 0:
                raise VRIFRuntimeSettlementError(
                    "terminal sidecar checkpoint requires exact zero-active stores"
                )

            checkpointed_lanes: list[dict[str, Any]] = []
            for index, paths in enumerate(lane_paths):
                checkpointed_lanes.append(
                    {
                        "index": index,
                        "coordination": _checkpoint_vrif_sidecar(
                            paths["coordination"]
                        ),
                        "execution": _checkpoint_vrif_sidecar(paths["execution"]),
                    }
                )

            for index, paths in enumerate(lane_paths):
                owner_id = _logical_owner_id(
                    logical_database_path=paths["logical"],
                    coordination_path=paths["logical"],
                    execution_path=paths["execution"],
                )
                coordination_store = _store_identity(paths["coordination"])
                execution_store = _store_identity(paths["execution"])
                if (
                    coordination_store["wal"]["state"] != "absent"
                    or execution_store["wal"]["state"] != "absent"
                    or _read_coordination_snapshot(
                        paths["coordination"],
                        store_identity=coordination_store,
                        max_active_ids=max_active_ids,
                    )["active_count"]
                    != 0
                    or _read_execution_snapshot(
                        paths["execution"],
                        store_identity=execution_store,
                        expected_owner_session_id=owner_id,
                        max_active_ids=max_active_ids,
                    )["active_count"]
                    != 0
                ):
                    raise VRIFRuntimeSettlementError(
                        f"VRIF lane {index} changed during terminal checkpoint"
                    )

            _require_config_unchanged(profile)
            if (
                _require_current_master_pid_projection(
                    supplied_master_path,
                    expected_pid=expected_master_pid,
                )
                != master_identity
                or [
                    _pid_observation(
                        state_path
                        / f"lane-{index}"
                        / f"vrif_lane_{index}_supervisor.pid",
                        label=f"VRIF lane {index} supervisor PID marker",
                    )
                    for index in range(4)
                ]
                != lane_pid_observations
            ):
                raise VRIFRuntimeSettlementError(
                    "VRIF lifecycle state changed during terminal checkpoint"
                )

            receipt: dict[str, Any] = {
                "schema": VRIF_TERMINAL_SIDECAR_CHECKPOINT_SCHEMA,
                "config_cid": profile["config_cid"],
                "master_pid": expected_master_pid,
                "active_counts": {
                    "coordination": active_coordination,
                    "execution": active_execution,
                    "total": active_coordination + active_execution,
                },
                "lane_count": len(checkpointed_lanes),
                "lanes": checkpointed_lanes,
            }
            receipt["receipt_cid"] = _content_id(receipt)
            return receipt


def _validate_public_inputs(
    *,
    target_repository_id: str,
    target_branch: str,
    owner_generation: int,
    max_active_ids: int,
    lock_timeout_seconds: float,
) -> float:
    if (
        type(target_repository_id) is not str
        or not target_repository_id
        or target_repository_id != target_repository_id.strip()
        or len(target_repository_id.encode("utf-8")) > 512
        or any(ord(character) < 0x20 for character in target_repository_id)
        or ".." in target_repository_id
        or "/" in target_repository_id
        or "\\" in target_repository_id
        or target_repository_id != VRIF_TARGET_REPOSITORY_ID
    ):
        raise ValueError(
            "target_repository_id must be one bounded canonical repository identity"
        )
    if (
        not isinstance(target_branch, str)
        or not target_branch
        or target_branch != target_branch.strip()
        or "\x00" in target_branch
    ):
        raise ValueError("target_branch must be an exact non-empty string")
    if type(owner_generation) is not int or owner_generation <= 0:
        raise ValueError("owner_generation must be a positive integer")
    if (
        type(max_active_ids) is not int
        or max_active_ids < 1
        or max_active_ids > _MAX_ACTIVE_IDS
    ):
        raise ValueError(
            f"max_active_ids must be an integer between 1 and {_MAX_ACTIVE_IDS}"
        )
    return _bounded_lock_timeout(lock_timeout_seconds)


def _add_snapshot_cid(value: dict[str, Any]) -> dict[str, Any]:
    value["snapshot_cid"] = _content_id(value)
    return value


def _validate_section_cid(value: Mapping[str, Any], *, label: str) -> None:
    supplied = value.get("snapshot_cid")
    content = dict(value)
    content.pop("snapshot_cid", None)
    if supplied != _content_id(content):
        raise VRIFRuntimeSettlementError(f"{label} snapshot CID differs")


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise VRIFRuntimeSettlementError(f"{label} has a noncanonical field set")
    return value


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise VRIFRuntimeSettlementError(f"{label} must be a nonnegative integer")
    return value


def _require_finite_nonnegative_float(value: Any, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise VRIFRuntimeSettlementError(
            f"{label} must be a finite nonnegative floating-point value"
        )
    return value


def _validate_bounded_text(
    value: Any,
    *,
    label: str,
    max_bytes: int,
    allow_empty: bool = False,
) -> str:
    if (
        type(value) is not str
        or (not allow_empty and not value)
        or len(value.encode("utf-8")) > max_bytes
        or "\x00" in value
    ):
        raise VRIFRuntimeSettlementError(f"{label} is not a bounded exact string")
    return value


def _validate_file_receipt(
    value: Any,
    *,
    expected_path: Path,
    label: str,
    require_nonempty: bool = False,
) -> Mapping[str, Any]:
    file_row = _require_exact_keys(
        value,
        {
            "path",
            "device",
            "inode",
            "mode",
            "link_count",
            "uid",
            "size_bytes",
            "modified_ns",
            "changed_ns",
        },
        label=label,
    )
    if file_row["path"] != str(expected_path.absolute()):
        raise VRIFRuntimeSettlementError(f"{label} path differs")
    for key in (
        "device",
        "inode",
        "mode",
        "link_count",
        "uid",
        "size_bytes",
        "modified_ns",
        "changed_ns",
    ):
        _require_nonnegative_int(file_row[key], label=f"{label}.{key}")
    if (
        file_row["inode"] <= 0
        or file_row["mode"] > 0o7777
        or file_row["link_count"] != 1
        or (require_nonempty and file_row["size_bytes"] <= 0)
    ):
        raise VRIFRuntimeSettlementError(f"{label} identity is malformed")
    return file_row


def _validate_sidecar_store(
    value: Any,
    *,
    expected_database_path: Path,
    label: str,
) -> None:
    store = _require_exact_keys(value, {"database", "wal"}, label=label)
    database = _validate_file_receipt(
        store["database"],
        expected_path=expected_database_path,
        label=f"{label} database",
        require_nonempty=True,
    )
    expected_wal_path = Path(str(expected_database_path) + ".wal")
    wal = store["wal"]
    if not isinstance(wal, Mapping) or wal.get("path") != str(
        expected_wal_path.absolute()
    ):
        raise VRIFRuntimeSettlementError(f"{label} WAL path differs")
    if wal.get("state") == "absent":
        _require_exact_keys(wal, {"path", "state"}, label=f"{label} WAL")
        return
    wal_row = _require_exact_keys(
        wal,
        {"path", "state", "file"},
        label=f"{label} WAL",
    )
    if wal_row["state"] != "present":
        raise VRIFRuntimeSettlementError(f"{label} WAL state is malformed")
    wal_file = _validate_file_receipt(
        wal_row["file"],
        expected_path=expected_wal_path,
        label=f"{label} WAL file",
        require_nonempty=True,
    )
    if wal_file["uid"] != database["uid"]:
        raise VRIFRuntimeSettlementError(f"{label} WAL owner differs")


def _validate_count_map(
    value: Any,
    expected_keys: Sequence[str],
    *,
    label: str,
    maximum: int | None = None,
) -> dict[str, int]:
    counts = _require_exact_keys(value, set(expected_keys), label=label)
    result: dict[str, int] = {}
    for key in expected_keys:
        count = _require_nonnegative_int(counts[key], label=f"{label}.{key}")
        if maximum is not None and count > maximum:
            raise VRIFRuntimeSettlementError(f"{label}.{key} exceeds its bound")
        result[key] = count
    return result


def _validate_identifier_list(
    value: Any,
    *,
    label: str,
    maximum: int,
    max_item_bytes: int = 1_024,
) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise VRIFRuntimeSettlementError(f"{label} exceeds its identity bound")
    identifiers = [
        _validate_bounded_text(
            item,
            label=f"{label} item",
            max_bytes=max_item_bytes,
        )
        for item in value
    ]
    if identifiers != sorted(set(identifiers)):
        raise VRIFRuntimeSettlementError(
            f"{label} must contain sorted unique identities"
        )
    return identifiers


def _validate_prefixed_identity_counts(
    identifiers: Sequence[str],
    expected: Mapping[str, int],
    *,
    label: str,
) -> None:
    observed = {prefix: 0 for prefix in expected}
    for identity in identifiers:
        matches = [prefix for prefix in expected if identity.startswith(prefix)]
        if len(matches) != 1 or len(identity) == len(matches[0]):
            raise VRIFRuntimeSettlementError(f"{label} contains an unknown identity")
        observed[matches[0]] += 1
    if observed != dict(expected):
        raise VRIFRuntimeSettlementError(f"{label} counts disagree")


def _validate_queue_database_identity(
    value: Any,
    *,
    expected_path: Path,
) -> Mapping[str, Any]:
    identity = _require_exact_keys(
        value,
        {"path", "device", "inode", "size_bytes", "modified_ns", "changed_ns"},
        label="merge queue database identity",
    )
    if identity["path"] != str(expected_path.absolute()):
        raise VRIFRuntimeSettlementError("merge queue database path differs")
    for key in ("device", "inode", "size_bytes", "modified_ns", "changed_ns"):
        _require_nonnegative_int(
            identity[key], label=f"merge queue database identity.{key}"
        )
    if identity["inode"] <= 0 or identity["size_bytes"] <= 0:
        raise VRIFRuntimeSettlementError("merge queue database identity is malformed")
    return identity


def _validate_merge_queue_receipt(
    value: Any,
    *,
    expected_queue_path: Path,
    target_repository_id: str,
    target_branch: str,
    max_active_ids: int,
) -> tuple[Mapping[str, Any], int, list[str]]:
    queue = _require_exact_keys(
        value,
        {
            "schema",
            "settled",
            "target",
            "database",
            "store",
            "row_count",
            "max_updated_at",
            "max_claim_generation",
            "status_counts",
            "active_count",
            "active_request_ids",
            "snapshot_cid",
            "receipt_cid",
        },
        label="merge queue settlement receipt",
    )
    if queue["schema"] != MERGE_QUEUE_SETTLEMENT_SCHEMA:
        raise VRIFRuntimeSettlementError("merge queue settlement schema differs")
    if type(queue["settled"]) is not bool:
        raise VRIFRuntimeSettlementError("merge queue settled marker is malformed")
    target = _require_exact_keys(
        queue["target"],
        {"binding_schema", "repository_id", "branch"},
        label="merge queue target",
    )
    if target != {
        "binding_schema": MERGE_TARGET_BINDING_SCHEMA,
        "repository_id": target_repository_id,
        "branch": target_branch,
    }:
        raise VRIFRuntimeSettlementError("merge queue target differs")
    _validate_queue_database_identity(
        queue["database"],
        expected_path=expected_queue_path / "merge_queue.duckdb",
    )
    store = _require_exact_keys(
        queue["store"],
        {"metadata_cid", "metadata_rows", "store_id", "generation"},
        label="merge queue store identity",
    )
    if (
        type(store["metadata_cid"]) is not str
        or _SHA256_CID.fullmatch(store["metadata_cid"]) is None
    ):
        raise VRIFRuntimeSettlementError("merge queue metadata CID is malformed")
    metadata_rows = _require_nonnegative_int(
        store["metadata_rows"], label="merge queue metadata row count"
    )
    if metadata_rows > 256:
        raise VRIFRuntimeSettlementError("merge queue metadata row count exceeds bound")
    for key in ("store_id", "generation"):
        if store[key] is not None:
            _validate_bounded_text(
                store[key], label=f"merge queue {key}", max_bytes=4_096
            )
    row_count = _require_nonnegative_int(
        queue["row_count"], label="merge queue row count"
    )
    if row_count > _HISTORY_ROW_BOUND:
        raise VRIFRuntimeSettlementError("merge queue row count exceeds bound")
    _require_finite_nonnegative_float(
        queue["max_updated_at"], label="merge queue max_updated_at"
    )
    _require_nonnegative_int(
        queue["max_claim_generation"],
        label="merge queue max_claim_generation",
    )
    status_counts = _validate_count_map(
        queue["status_counts"],
        _MERGE_QUEUE_STATES,
        label="merge queue status counts",
    )
    if sum(status_counts.values()) != row_count:
        raise VRIFRuntimeSettlementError("merge queue status counts disagree")
    active_ids = _validate_identifier_list(
        queue["active_request_ids"],
        label="merge queue active request identities",
        maximum=max_active_ids,
        max_item_bytes=512,
    )
    active_count = _require_nonnegative_int(
        queue["active_count"], label="merge queue active count"
    )
    expected_active = sum(status_counts[state] for state in _MERGE_QUEUE_ACTIVE_STATES)
    if (
        active_count != expected_active
        or active_count != len(active_ids)
        or queue["settled"] is not (active_count == 0)
    ):
        raise VRIFRuntimeSettlementError("merge queue active state disagrees")
    if (
        type(queue["snapshot_cid"]) is not str
        or _SHA256_CID.fullmatch(queue["snapshot_cid"]) is None
    ):
        raise VRIFRuntimeSettlementError("merge queue snapshot CID is malformed")
    receipt_content = dict(queue)
    supplied_receipt_cid = receipt_content.pop("receipt_cid")
    if supplied_receipt_cid != _content_id(receipt_content):
        raise VRIFRuntimeSettlementError("merge queue receipt CID differs")
    return queue, active_count, active_ids


def _queue_snapshot_material(
    queue: Mapping[str, Any],
    active_requests: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    store = queue["store"]
    return {
        "database": queue["database"],
        "store_metadata_cid": store["metadata_cid"],
        "store_metadata_rows": store["metadata_rows"],
        "store_id": store["store_id"],
        "store_generation": store["generation"],
        "row_count": queue["row_count"],
        "max_updated_at": queue["max_updated_at"],
        "max_claim_generation": queue["max_claim_generation"],
        "status_counts": queue["status_counts"],
        "active_requests": list(active_requests),
    }


def _read_merge_queue_verification(
    queue_path: Path,
    *,
    queue_receipt: Mapping[str, Any],
    target_repository_id: str,
    target_branch: str,
    max_active_ids: int,
) -> dict[str, Any]:
    """Verify row markers and the hidden queue snapshot under its held lock."""

    queue, expected_active, expected_ids = _validate_merge_queue_receipt(
        queue_receipt,
        expected_queue_path=queue_path,
        target_repository_id=target_repository_id,
        target_branch=target_branch,
        max_active_ids=max_active_ids,
    )
    database_path = queue_path / "merge_queue.duckdb"
    connection: Any | None = None
    transaction_open = False
    try:
        import duckdb

        connection = connect_duckdb_with_policy(
            duckdb,
            database_path,
            read_only=True,
        )
        connection.execute("BEGIN TRANSACTION")
        transaction_open = True
        queue_rows = connection.execute(
            """
            SELECT
                CASE WHEN octet_length(encode(request_id)) <= 512
                     THEN request_id END,
                status,
                CASE WHEN octet_length(encode(metadata_json)) <= ?
                     THEN metadata_json END,
                finished_at
            FROM merge_requests
            ORDER BY request_id
            LIMIT ?
            """,
            [_MERGE_QUEUE_METADATA_MAX_BYTES, _HISTORY_ROW_BOUND + 1],
        ).fetchall()
        connection.execute("COMMIT")
        transaction_open = False
    except BaseException as exc:
        if connection is not None and transaction_open:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        if isinstance(exc, VRIFRuntimeSettlementError):
            raise
        raise VRIFRuntimeSettlementError(
            "merge queue marker verification is unavailable"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    if len(queue_rows) > _HISTORY_ROW_BOUND or len(queue_rows) != queue["row_count"]:
        raise VRIFRuntimeSettlementError(
            "merge queue row inventory exceeds the aggregate verification bound"
        )
    verified_rows: list[dict[str, Any]] = []
    active_rows: list[tuple[str, str]] = []
    verified_status_counts = {state: 0 for state in _MERGE_QUEUE_STATES}
    for request_id, status_value, metadata_raw, finished_raw in queue_rows:
        status_text = str(status_value)
        if (
            request_id is None
            or metadata_raw is None
            or status_text not in _MERGE_QUEUE_STATES
            or isinstance(finished_raw, bool)
            or not isinstance(finished_raw, (int, float))
        ):
            raise VRIFRuntimeSettlementError(
                "merge queue row is malformed during aggregate verification"
            )
        finished_at = float(finished_raw)
        marker_is_valid = (
            status_text in _MERGE_QUEUE_ACTIVE_STATES
            and math.isfinite(finished_at)
            and finished_at == 0.0
        ) or (
            status_text not in _MERGE_QUEUE_ACTIVE_STATES
            and math.isfinite(finished_at)
            and finished_at > 0.0
        )
        if not marker_is_valid:
            raise VRIFRuntimeSettlementError(
                "merge queue active or terminal finished_at marker disagrees"
            )
        metadata = _strict_json(
            str(metadata_raw),
            label=f"merge queue metadata for {request_id}",
            max_bytes=_MERGE_QUEUE_METADATA_MAX_BYTES,
        )
        if (
            not isinstance(metadata, dict)
            or metadata.get("target_binding_schema") != MERGE_TARGET_BINDING_SCHEMA
            or metadata.get("target_repository_id") != target_repository_id
            or metadata.get("target_branch") != target_branch
        ):
            raise VRIFRuntimeSettlementError(
                "merge queue row target binding is malformed or foreign"
            )
        request_text = str(request_id)
        verified_status_counts[status_text] += 1
        verified_rows.append(
            {
                "request_id": request_text,
                "status": status_text,
                "finished_at": finished_at,
            }
        )
        if status_text in _MERGE_QUEUE_ACTIVE_STATES:
            active_rows.append((request_text, status_text))
    verified_request_ids = [row["request_id"] for row in verified_rows]
    if (
        verified_request_ids != sorted(set(verified_request_ids))
        or verified_status_counts != queue["status_counts"]
    ):
        raise VRIFRuntimeSettlementError(
            "merge queue verified row identities or status counts disagree"
        )
    if len(active_rows) != expected_active:
        raise VRIFRuntimeSettlementError(
            "merge queue active rows changed or exceed the aggregate bound"
        )
    active_requests: list[dict[str, str]] = []
    for request_id, status_value in active_rows:
        active_requests.append(
            {"request_id": request_id, "status": status_value}
        )
    if [row["request_id"] for row in active_requests] != expected_ids:
        raise VRIFRuntimeSettlementError(
            "merge queue active identities differ during marker verification"
        )
    observed_status_counts = {state: 0 for state in _MERGE_QUEUE_ACTIVE_STATES}
    for row in active_requests:
        observed_status_counts[row["status"]] += 1
    if any(
        observed_status_counts[state] != queue["status_counts"][state]
        for state in _MERGE_QUEUE_ACTIVE_STATES
    ):
        raise VRIFRuntimeSettlementError(
            "merge queue active status assignment differs"
        )
    queue_snapshot = _queue_snapshot_material(queue, active_requests)
    if queue["snapshot_cid"] != _content_id(queue_snapshot):
        raise VRIFRuntimeSettlementError("merge queue hidden snapshot CID differs")
    verification: dict[str, Any] = {
        "schema": VRIF_MERGE_QUEUE_VERIFICATION_SCHEMA,
        "finished_at_markers_consistent": True,
        "target_bindings_consistent": True,
        "verified_row_count": len(verified_rows),
        "verified_rows_cid": _content_id(verified_rows),
        "queue_snapshot_cid": queue["snapshot_cid"],
        "active_requests": active_requests,
    }
    verification["snapshot_cid"] = _content_id(verification)
    return verification


def _validate_coordination_receipt(
    value: Any,
    *,
    expected_database_path: Path,
    max_active_ids: int,
    label: str,
) -> tuple[int, list[str]]:
    snapshot = _require_exact_keys(
        value,
        {
            "schema",
            "store",
            "metadata",
            "metadata_cid",
            "row_counts",
            "status_counts",
            "active_counts",
            "active_count",
            "active_ids",
            "snapshot_cid",
        },
        label=label,
    )
    if snapshot["schema"] != VRIF_COORDINATION_SETTLEMENT_SCHEMA:
        raise VRIFRuntimeSettlementError(f"{label} schema differs")
    _validate_sidecar_store(
        snapshot["store"],
        expected_database_path=expected_database_path,
        label=f"{label} store",
    )
    expected_metadata = {
        "interface": _coordination.DATABASE_COORDINATOR_INTERFACE,
        "schema": _coordination.DATABASE_COORDINATION_SCHEMA,
    }
    if (
        snapshot["metadata"] != expected_metadata
        or snapshot["metadata_cid"] != _content_id(expected_metadata)
    ):
        raise VRIFRuntimeSettlementError(f"{label} metadata differs")
    row_counts = _validate_count_map(
        snapshot["row_counts"],
        _COORDINATION_TABLES,
        label=f"{label} row counts",
        maximum=_HISTORY_ROW_BOUND,
    )
    if row_counts["coordination_metadata"] != 2:
        raise VRIFRuntimeSettlementError(f"{label} metadata row count differs")
    status_specs: dict[str, Sequence[str]] = {
        "fenced_leases": _LEASE_STATES,
        "task_claims": _TASK_CLAIM_STATES,
        "task_attempts": _ATTEMPT_STATES,
        "resource_claims": _RESOURCE_CLAIM_STATES,
        "maintenance_leases": _RESOURCE_CLAIM_STATES,
        "task_completions": _COMPLETION_STATES,
    }
    status_inventory = _require_exact_keys(
        snapshot["status_counts"],
        set(status_specs),
        label=f"{label} status inventory",
    )
    status_counts: dict[str, dict[str, int]] = {}
    for table, allowed in status_specs.items():
        counts = _validate_count_map(
            status_inventory[table],
            allowed,
            label=f"{label} {table} status counts",
            maximum=_HISTORY_ROW_BOUND,
        )
        if sum(counts.values()) != row_counts[table]:
            raise VRIFRuntimeSettlementError(
                f"{label} {table} status counts disagree"
            )
        status_counts[table] = counts
    active_names = (
        "ready_coordination_tasks",
        "accepted_fenced_leases",
        "accepted_task_claims",
        "running_task_attempts",
        "accepted_resource_claims",
        "accepted_maintenance_leases",
        "unresolved_completion_barriers",
    )
    active_counts = _validate_count_map(
        snapshot["active_counts"],
        active_names,
        label=f"{label} active counts",
        maximum=max_active_ids,
    )
    expected_from_status = {
        "accepted_fenced_leases": status_counts["fenced_leases"]["accepted"],
        "accepted_task_claims": status_counts["task_claims"]["accepted"],
        "running_task_attempts": status_counts["task_attempts"]["running"],
        "accepted_resource_claims": status_counts["resource_claims"]["accepted"],
        "accepted_maintenance_leases": status_counts["maintenance_leases"][
            "accepted"
        ],
    }
    if any(active_counts[name] != count for name, count in expected_from_status.items()):
        raise VRIFRuntimeSettlementError(f"{label} active status counts disagree")
    if active_counts["ready_coordination_tasks"] > row_counts["coordination_tasks"]:
        raise VRIFRuntimeSettlementError(f"{label} ready task count differs")
    if (
        active_counts["unresolved_completion_barriers"]
        < status_counts["task_completions"]["prepared"]
        or active_counts["unresolved_completion_barriers"]
        > row_counts["task_completions"]
    ):
        raise VRIFRuntimeSettlementError(f"{label} completion barrier count differs")
    active_ids = _validate_identifier_list(
        snapshot["active_ids"],
        label=f"{label} active identities",
        maximum=max_active_ids,
    )
    active_count = _require_nonnegative_int(
        snapshot["active_count"], label=f"{label} active count"
    )
    if active_count != sum(active_counts.values()) or active_count != len(active_ids):
        raise VRIFRuntimeSettlementError(f"{label} active counts disagree")
    _validate_prefixed_identity_counts(
        active_ids,
        {
            "coordination_task:": active_counts["ready_coordination_tasks"],
            "fenced_lease:": active_counts["accepted_fenced_leases"],
            "task_claim:": active_counts["accepted_task_claims"],
            "task_attempt:": active_counts["running_task_attempts"],
            "resource_claim:": active_counts["accepted_resource_claims"],
            "maintenance_lease:": active_counts[
                "accepted_maintenance_leases"
            ],
            "completion_barrier:": active_counts[
                "unresolved_completion_barriers"
            ],
        },
        label=f"{label} active identities",
    )
    _validate_section_cid(snapshot, label=label)
    return active_count, active_ids


def _validate_execution_receipt(
    value: Any,
    *,
    expected_database_path: Path,
    expected_owner_session_id: str,
    max_active_ids: int,
    label: str,
) -> tuple[int, list[str]]:
    snapshot = _require_exact_keys(
        value,
        {
            "schema",
            "store",
            "metadata",
            "metadata_cid",
            "row_counts",
            "status_counts",
            "phase_counts",
            "active_counts",
            "active_count",
            "active_ids",
            "snapshot_cid",
        },
        label=label,
    )
    if snapshot["schema"] != VRIF_EXECUTION_SETTLEMENT_SCHEMA:
        raise VRIFRuntimeSettlementError(f"{label} schema differs")
    _validate_sidecar_store(
        snapshot["store"],
        expected_database_path=expected_database_path,
        label=f"{label} store",
    )
    metadata = _require_exact_keys(
        snapshot["metadata"],
        set(_EXECUTION_METADATA_KEYS),
        label=f"{label} metadata",
    )
    if (
        metadata["interface"] != "DatabaseImplementationDaemon@1"
        or metadata["schema"]
        != "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1"
        or metadata["authority_mode"] != "quack"
        or metadata["logical_owner_session_id"] != expected_owner_session_id
        or type(metadata["process_instance_id"]) is not str
        or _PROCESS_INSTANCE.fullmatch(metadata["process_instance_id"]) is None
        or metadata["state_schema_revision"] != VRIF_CONTROL_SCHEMA_REVISION
        or metadata["control_schema_profile_id"] != ""
        or metadata["control_schema_fingerprint"] != ""
        or snapshot["metadata_cid"] != _content_id(metadata)
    ):
        raise VRIFRuntimeSettlementError(f"{label} metadata binding differs")
    row_counts = _validate_count_map(
        snapshot["row_counts"],
        tuple(sorted(_EXECUTION_COLUMNS)),
        label=f"{label} row counts",
        maximum=_HISTORY_ROW_BOUND,
    )
    if row_counts["daemon_execution_metadata"] != len(_EXECUTION_METADATA_KEYS):
        raise VRIFRuntimeSettlementError(f"{label} metadata row count differs")
    status_counts = _validate_count_map(
        snapshot["status_counts"],
        _EXECUTION_STATES,
        label=f"{label} status counts",
        maximum=_HISTORY_ROW_BOUND,
    )
    phase_counts = _validate_count_map(
        snapshot["phase_counts"],
        _EXECUTION_PHASES,
        label=f"{label} phase counts",
        maximum=_HISTORY_ROW_BOUND,
    )
    if (
        sum(status_counts.values()) != row_counts["database_task_attempts"]
        or sum(phase_counts.values()) != row_counts["attempt_phases"]
    ):
        raise VRIFRuntimeSettlementError(f"{label} row status counts disagree")
    active_names = (
        "running_attempts",
        "running_phase_rows",
        "running_provider_invocations",
        "running_effect_claims",
    )
    active_counts = _validate_count_map(
        snapshot["active_counts"],
        active_names,
        label=f"{label} active counts",
        maximum=max_active_ids,
    )
    if (
        active_counts["running_attempts"] != status_counts["running"]
        or active_counts["running_phase_rows"] > row_counts["attempt_phases"]
        or active_counts["running_provider_invocations"]
        > row_counts["provider_invocations"]
        or active_counts["running_effect_claims"] > row_counts["effect_claims"]
        or (
            active_counts["running_attempts"] > 0
            and active_counts["running_phase_rows"]
            < active_counts["running_attempts"]
        )
        or (
            active_counts["running_attempts"] == 0
            and any(active_counts[name] for name in active_names[1:])
        )
    ):
        raise VRIFRuntimeSettlementError(f"{label} active row counts disagree")
    active_ids = _validate_identifier_list(
        snapshot["active_ids"],
        label=f"{label} active identities",
        maximum=max_active_ids,
    )
    active_count = _require_nonnegative_int(
        snapshot["active_count"], label=f"{label} active count"
    )
    if active_count != sum(active_counts.values()) or active_count != len(active_ids):
        raise VRIFRuntimeSettlementError(f"{label} active counts disagree")
    _validate_prefixed_identity_counts(
        active_ids,
        {
            "execution_attempt:": active_counts["running_attempts"],
            "execution_phase:": active_counts["running_phase_rows"],
            "provider_invocation:": active_counts[
                "running_provider_invocations"
            ],
            "effect_claim:": active_counts["running_effect_claims"],
        },
        label=f"{label} active identities",
    )
    _validate_section_cid(snapshot, label=label)
    return active_count, active_ids


def _validate_retired_config_profile(
    value: Any,
    *,
    repository_root: Path,
) -> list[Mapping[str, Any]]:
    profile = _require_exact_keys(
        value,
        {"schema", "retired_coordination_snapshots"},
        label="runtime retired coordination config",
    )
    if profile["schema"] != VRIF_RUNTIME_SETTLEMENT_CONFIG_SCHEMA:
        raise VRIFRuntimeSettlementError(
            "runtime retired coordination config schema differs"
        )
    entries = profile["retired_coordination_snapshots"]
    if (
        not isinstance(entries, list)
        or len(entries) > _MAX_RETIRED_COORDINATION_SNAPSHOTS
    ):
        raise VRIFRuntimeSettlementError(
            "runtime retired coordination config inventory differs"
        )
    normalized: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    seen_lane_indexes: set[int] = set()
    for ordinal, entry in enumerate(entries):
        item = _require_exact_keys(
            entry,
            {
                "schema",
                "lane_index",
                "database_path",
                "database_size_bytes",
                "database_sha256",
                "wal_path",
                "wal_size_bytes",
                "wal_sha256",
                "terminal_execution_attempt_ids",
                "terminal_execution_attempt_ids_cid",
            },
            label=f"runtime retired coordination config {ordinal}",
        )
        database_relative = _validate_bounded_text(
            item["database_path"],
            label=f"runtime retired coordination config {ordinal} database path",
            max_bytes=4_096,
        )
        wal_relative = _validate_bounded_text(
            item["wal_path"],
            label=f"runtime retired coordination config {ordinal} WAL path",
            max_bytes=4_096,
        )
        database_parts = Path(database_relative).parts
        prefix = Path(VRIF_STATE_RELATIVE_PATH, "sidecar-quarantine").parts
        lane_index = item["lane_index"]
        if (
            item["schema"] != VRIF_RETIRED_COORDINATION_SNAPSHOT_SCHEMA
            or type(lane_index) is not int
            or lane_index not in _RETIRED_COORDINATION_LANE_INDEXES
            or Path(database_relative).is_absolute()
            or Path(wal_relative).is_absolute()
            or ".." in database_parts
            or ".." in Path(wal_relative).parts
            or wal_relative != database_relative + ".wal"
            or database_parts[: len(prefix)] != prefix
            or len(database_parts) < len(prefix) + 3
            or database_parts[-2] != f"lane-{lane_index}"
            or database_parts[-1]
            != "quack-lane-control.coordination.duckdb"
        ):
            raise VRIFRuntimeSettlementError(
                "runtime retired coordination config path or lane differs"
            )
        if lane_index in seen_lane_indexes:
            raise VRIFRuntimeSettlementError(
                "runtime retired coordination config lane repeats"
            )
        seen_lane_indexes.add(lane_index)
        database_path = _absolute_lexical(repository_root / database_relative)
        wal_path = _absolute_lexical(repository_root / wal_relative)
        try:
            database_path.relative_to(repository_root)
            wal_path.relative_to(repository_root)
        except ValueError as exc:
            raise VRIFRuntimeSettlementError(
                "runtime retired coordination config escapes repository root"
            ) from exc
        for name in ("database_size_bytes", "wal_size_bytes"):
            size = _require_nonnegative_int(
                item[name],
                label=f"runtime retired coordination config {ordinal}.{name}",
            )
            if size <= 0 or size > _RETIRED_STORE_MAX_BYTES:
                raise VRIFRuntimeSettlementError(
                    "runtime retired coordination config size differs"
                )
        for name in ("database_sha256", "wal_sha256"):
            if (
                type(item[name]) is not str
                or _SHA256_CID.fullmatch(item[name]) is None
            ):
                raise VRIFRuntimeSettlementError(
                    "runtime retired coordination config CID differs"
                )
        attempt_ids = _validate_identifier_list(
            item["terminal_execution_attempt_ids"],
            label=(
                f"runtime retired coordination config {ordinal} terminal attempts"
            ),
            maximum=_HISTORY_ROW_BOUND,
            max_item_bytes=512,
        )
        if (
            not attempt_ids
            or any(not attempt_id.startswith("attempt:") for attempt_id in attempt_ids)
            or seen_ids.intersection(attempt_ids)
            or item["terminal_execution_attempt_ids_cid"]
            != _content_id(attempt_ids)
        ):
            raise VRIFRuntimeSettlementError(
                "runtime retired coordination config attempt inventory differs"
            )
        seen_ids.update(attempt_ids)
        normalized.append(item)
    if list(normalized) != sorted(
        normalized,
        key=lambda entry: (entry["lane_index"], entry["database_path"]),
    ):
        raise VRIFRuntimeSettlementError(
            "runtime retired coordination config order differs"
        )
    return normalized


def _validate_coordination_authority_bindings(
    value: Any,
    *,
    label: str,
    terminal_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(value, list) or len(value) > _HISTORY_ROW_BOUND:
        raise VRIFRuntimeSettlementError(f"{label} exceeds its history bound")
    rows: list[dict[str, Any]] = []
    exact_keys = {
        "attempt_id",
        "claim_id",
        "task_cid",
        "attempt_number",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "lease_id",
        "coordination_claim_state",
        "coordination_attempt_status",
        "coordination_lease_state",
    }
    terminal_triples = {
        ("released", "released", "released"),
        ("released", "succeeded", "released"),
        ("expired", "expired", "expired"),
        ("expired", "expired", "superseded"),
        ("completed", "succeeded", "completed"),
    }
    producer_triples = terminal_triples | {
        ("accepted", "running", "accepted"),
    }
    for ordinal, row in enumerate(value):
        item = _require_exact_keys(
            row,
            exact_keys,
            label=f"{label} row {ordinal}",
        )
        normalized = {
            name: _validate_bounded_text(
                item[name],
                label=f"{label} row {ordinal}.{name}",
                max_bytes=1_024,
            )
            for name in (
                "attempt_id",
                "claim_id",
                "task_cid",
                "owner_session_id",
                "lease_id",
            )
        }
        for name in ("attempt_number", "fencing_token", "fence_epoch"):
            normalized[name] = _require_nonnegative_int(
                item[name], label=f"{label} row {ordinal}.{name}"
            )
            if normalized[name] <= 0:
                raise VRIFRuntimeSettlementError(
                    f"{label} row {ordinal}.{name} must be positive"
                )
        for name, allowed in (
            ("coordination_claim_state", _TASK_CLAIM_STATES),
            ("coordination_attempt_status", _ATTEMPT_STATES),
            ("coordination_lease_state", _LEASE_STATES),
        ):
            state_value = item[name]
            if type(state_value) is not str or state_value not in allowed:
                raise VRIFRuntimeSettlementError(
                    f"{label} row {ordinal}.{name} differs"
                )
            normalized[name] = state_value
        triple = (
            normalized["coordination_claim_state"],
            normalized["coordination_attempt_status"],
            normalized["coordination_lease_state"],
        )
        if triple not in producer_triples or (
            terminal_only and triple not in terminal_triples
        ):
            raise VRIFRuntimeSettlementError(
                f"{label} row {ordinal} is not terminal producer authority"
            )
        rows.append(normalized)
    if [row["attempt_id"] for row in rows] != sorted(
        {row["attempt_id"] for row in rows}
    ):
        raise VRIFRuntimeSettlementError(
            f"{label} must contain sorted unique attempts"
        )
    return rows, {row["attempt_id"]: row for row in rows}


def _validate_retired_coordination_lineage(
    value: Any,
    *,
    entry: Mapping[str, Any],
    config_ordinal: int,
    repository_root: Path,
) -> tuple[Mapping[str, Any], dict[str, dict[str, Any]], list[str]]:
    lineage = _require_exact_keys(
        value,
        {
            "schema",
            "config_ordinal",
            "lane_index",
            "database_path",
            "wal_path",
            "configured_content",
            "policy_lock",
            "coordination",
            "historical_ready_tasks",
            "historical_ready_tasks_cid",
            "authority_bindings",
            "authority_binding_count",
            "authority_bindings_cid",
            "admitted_terminal_execution_attempt_ids",
            "admitted_terminal_execution_attempt_ids_cid",
            "snapshot_cid",
        },
        label=f"retired coordination lineage {config_ordinal}",
    )
    database_path = repository_root / entry["database_path"]
    wal_path = repository_root / entry["wal_path"]
    policy_lock_path = database_path.with_name(f".{database_path.name}.lock")
    if (
        lineage["schema"] != VRIF_RETIRED_COORDINATION_LINEAGE_SCHEMA
        or type(lineage["config_ordinal"]) is not int
        or lineage["config_ordinal"] != config_ordinal
        or type(lineage["lane_index"]) is not int
        or lineage["lane_index"] != entry["lane_index"]
        or lineage["database_path"] != str(database_path.absolute())
        or lineage["wal_path"] != str(wal_path.absolute())
        or lineage["configured_content"]
        != {
            "database_size_bytes": entry["database_size_bytes"],
            "database_sha256": entry["database_sha256"],
            "wal_size_bytes": entry["wal_size_bytes"],
            "wal_sha256": entry["wal_sha256"],
        }
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination lineage identity differs"
        )
    _validate_file_receipt(
        lineage["policy_lock"],
        expected_path=policy_lock_path,
        label=f"retired coordination lineage {config_ordinal} policy lock",
    )
    coordination_active, coordination_ids = _validate_coordination_receipt(
        lineage["coordination"],
        expected_database_path=database_path,
        max_active_ids=_HISTORY_ROW_BOUND,
        label=f"retired coordination lineage {config_ordinal} snapshot",
    )
    store = lineage["coordination"]["store"]
    if (
        store["database"]["size_bytes"] != entry["database_size_bytes"]
        or store["wal"].get("state") != "present"
        or store["wal"].get("file", {}).get("size_bytes")
        != entry["wal_size_bytes"]
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination lineage file sizes differ"
        )
    active_counts = lineage["coordination"]["active_counts"]
    if any(
        count != 0
        for name, count in active_counts.items()
        if name != "ready_coordination_tasks"
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination lineage retains active authority"
        )
    ready_value = lineage["historical_ready_tasks"]
    if not isinstance(ready_value, list) or len(ready_value) > _HISTORY_ROW_BOUND:
        raise VRIFRuntimeSettlementError(
            "retired coordination ready task inventory exceeds its bound"
        )
    ready_tasks: list[dict[str, str]] = []
    for ordinal, row in enumerate(ready_value):
        item = _require_exact_keys(
            row,
            {"task_cid", "task_id"},
            label=f"retired coordination ready task {ordinal}",
        )
        ready_tasks.append(
            {
                "task_cid": _validate_bounded_text(
                    item["task_cid"],
                    label=f"retired coordination ready task {ordinal} CID",
                    max_bytes=1_024,
                ),
                "task_id": _validate_bounded_text(
                    item["task_id"],
                    label=f"retired coordination ready task {ordinal} alias",
                    max_bytes=512,
                ),
            }
        )
    if (
        ready_tasks
        != sorted(ready_tasks, key=lambda row: (row["task_cid"], row["task_id"]))
        or len({row["task_cid"] for row in ready_tasks}) != len(ready_tasks)
        or lineage["historical_ready_tasks_cid"] != _content_id(ready_tasks)
        or coordination_active != len(ready_tasks)
        or coordination_ids
        != sorted(f"coordination_task:{row['task_cid']}" for row in ready_tasks)
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination ready task binding differs"
        )
    authority_rows, authority_by_attempt = (
        _validate_coordination_authority_bindings(
            lineage["authority_bindings"],
            label=f"retired coordination lineage {config_ordinal} authority",
            terminal_only=True,
        )
    )
    admitted_ids = _validate_identifier_list(
        lineage["admitted_terminal_execution_attempt_ids"],
        label=f"retired coordination lineage {config_ordinal} admitted attempts",
        maximum=_HISTORY_ROW_BOUND,
        max_item_bytes=512,
    )
    if (
        lineage["authority_binding_count"] != len(authority_rows)
        or len(authority_rows)
        != lineage["coordination"]["row_counts"]["task_claims"]
        or lineage["authority_bindings_cid"] != _content_id(authority_rows)
        or admitted_ids != entry["terminal_execution_attempt_ids"]
        or any(attempt_id not in authority_by_attempt for attempt_id in admitted_ids)
        or lineage["admitted_terminal_execution_attempt_ids_cid"]
        != _content_id(admitted_ids)
    ):
        raise VRIFRuntimeSettlementError(
            "retired coordination lineage authority binding differs"
        )
    _validate_section_cid(
        lineage,
        label=f"retired coordination lineage {config_ordinal}",
    )
    return lineage, authority_by_attempt, [
        row["task_cid"] for row in ready_tasks
    ]


def _validate_execution_authority_rows(
    value: Any,
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(value, list) or len(value) > _HISTORY_ROW_BOUND:
        raise VRIFRuntimeSettlementError(f"{label} exceeds its history bound")
    rows: list[dict[str, Any]] = []
    exact_keys = {
        "attempt_id",
        "claim_id",
        "task_cid",
        "attempt_number",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "lease_id",
        "execution_status",
    }
    for ordinal, row in enumerate(value):
        item = _require_exact_keys(
            row,
            exact_keys,
            label=f"{label} row {ordinal}",
        )
        normalized: dict[str, Any] = {
            name: _validate_bounded_text(
                item[name],
                label=f"{label} row {ordinal}.{name}",
                max_bytes=1_024,
            )
            for name in (
                "attempt_id",
                "claim_id",
                "task_cid",
                "owner_session_id",
                "lease_id",
            )
        }
        for name in ("attempt_number", "fencing_token", "fence_epoch"):
            normalized[name] = _require_nonnegative_int(
                item[name], label=f"{label} row {ordinal}.{name}"
            )
            if normalized[name] <= 0:
                raise VRIFRuntimeSettlementError(
                    f"{label} row {ordinal}.{name} must be positive"
                )
        if (
            type(item["execution_status"]) is not str
            or item["execution_status"] not in _EXECUTION_STATES
        ):
            raise VRIFRuntimeSettlementError(
                f"{label} row {ordinal} status differs"
            )
        normalized["execution_status"] = item["execution_status"]
        rows.append(normalized)
    if [row["attempt_id"] for row in rows] != sorted(
        {row["attempt_id"] for row in rows}
    ):
        raise VRIFRuntimeSettlementError(
            f"{label} must contain sorted unique attempts"
        )
    return rows, {row["attempt_id"]: row for row in rows}


def _validate_cross_store_binding(
    value: Any,
    *,
    coordination_task_claim_count: int,
    execution_attempt_count: int,
    retired_lineages: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]
    ],
    label: str,
) -> None:
    binding = _require_exact_keys(
        value,
        {
            "schema",
            "coordination_task_claim_count",
            "retired_coordination_task_claim_count",
            "execution_attempt_count",
            "current_matched_execution_attempt_count",
            "retired_matched_execution_attempt_count",
            "matched_execution_attempt_count",
            "retired_matched_execution_attempt_ids",
            "retired_matched_execution_attempt_ids_cid",
            "current_coordination_authority_bindings",
            "current_coordination_authority_bindings_cid",
            "execution_authority_rows",
            "execution_authority_rows_cid",
            "bindings",
            "bindings_cid",
            "snapshot_cid",
        },
        label=label,
    )
    for key in (
        "coordination_task_claim_count",
        "retired_coordination_task_claim_count",
        "execution_attempt_count",
        "current_matched_execution_attempt_count",
        "retired_matched_execution_attempt_count",
        "matched_execution_attempt_count",
    ):
        _require_nonnegative_int(binding[key], label=f"{label}.{key}")
    retired_authorities: dict[tuple[int, str], Mapping[str, Any]] = {}
    expected_retired_ids: list[str] = []
    retired_claim_count = 0
    for ordinal, (lineage, authorities) in enumerate(retired_lineages):
        retired_claim_count += len(authorities)
        admitted = list(lineage["admitted_terminal_execution_attempt_ids"])
        expected_retired_ids.extend(admitted)
        for attempt_id, authority in authorities.items():
            retired_authorities[(ordinal, attempt_id)] = authority
    expected_retired_ids.sort()
    retired_ids = _validate_identifier_list(
        binding["retired_matched_execution_attempt_ids"],
        label=f"{label} retired matched attempts",
        maximum=_HISTORY_ROW_BOUND,
        max_item_bytes=512,
    )
    current_authority_rows, current_authority_by_attempt = (
        _validate_coordination_authority_bindings(
            binding["current_coordination_authority_bindings"],
            label=f"{label} current coordination authority",
            terminal_only=False,
        )
    )
    execution_authority_rows, execution_authority_by_attempt = (
        _validate_execution_authority_rows(
            binding["execution_authority_rows"],
            label=f"{label} execution authority",
        )
    )
    bindings_value = binding["bindings"]
    if not isinstance(bindings_value, list) or len(bindings_value) > _HISTORY_ROW_BOUND:
        raise VRIFRuntimeSettlementError(f"{label} bindings exceed their bound")
    normalized_bindings: list[dict[str, Any]] = []
    observed_retired_ids: list[str] = []
    for ordinal, row in enumerate(bindings_value):
        item = _require_exact_keys(
            row,
            {
                "attempt_id",
                "claim_id",
                "task_cid",
                "attempt_number",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "lease_id",
                "execution_status",
                "coordination_claim_state",
                "coordination_attempt_status",
                "coordination_lease_state",
                "authority_source",
                "retired_snapshot_ordinal",
            },
            label=f"{label} binding {ordinal}",
        )
        authority_projection, _by_id = _validate_coordination_authority_bindings(
            [
                {
                    key: item[key]
                    for key in (
                        "attempt_id",
                        "claim_id",
                        "task_cid",
                        "attempt_number",
                        "owner_session_id",
                        "fencing_token",
                        "fence_epoch",
                        "lease_id",
                        "coordination_claim_state",
                        "coordination_attempt_status",
                        "coordination_lease_state",
                    )
                }
            ],
            label=f"{label} binding {ordinal} authority",
            terminal_only=item["authority_source"] == "retired",
        )
        execution_status = item["execution_status"]
        if type(execution_status) is not str or execution_status not in _EXECUTION_STATES:
            raise VRIFRuntimeSettlementError(
                f"{label} binding {ordinal} execution status differs"
            )
        source = item["authority_source"]
        retired_ordinal = item["retired_snapshot_ordinal"]
        if source == "current":
            if type(retired_ordinal) is not int or retired_ordinal != -1:
                raise VRIFRuntimeSettlementError(
                    f"{label} current binding source differs"
                )
            attempt_id = authority_projection[0]["attempt_id"]
            if current_authority_by_attempt.get(attempt_id) != authority_projection[0]:
                raise VRIFRuntimeSettlementError(
                    f"{label} current binding authority differs"
                )
        elif source == "retired":
            if (
                type(retired_ordinal) is not int
                or retired_ordinal < 0
                or retired_ordinal >= len(retired_lineages)
                or execution_status == "running"
            ):
                raise VRIFRuntimeSettlementError(
                    f"{label} retired binding source differs"
                )
            attempt_id = authority_projection[0]["attempt_id"]
            expected_authority = retired_authorities.get(
                (retired_ordinal, attempt_id)
            )
            if expected_authority != authority_projection[0]:
                raise VRIFRuntimeSettlementError(
                    f"{label} retired binding authority differs"
                )
            observed_retired_ids.append(attempt_id)
        else:
            raise VRIFRuntimeSettlementError(
                f"{label} binding authority source differs"
            )
        execution_projection = {
            key: item[key]
            for key in (
                "attempt_id",
                "claim_id",
                "task_cid",
                "attempt_number",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "lease_id",
                "execution_status",
            )
        }
        if (
            execution_authority_by_attempt.get(item["attempt_id"])
            != execution_projection
        ):
            raise VRIFRuntimeSettlementError(
                f"{label} execution binding authority differs"
            )
        authority = authority_projection[0]
        coordination_status = authority["coordination_attempt_status"]
        claim_state = authority["coordination_claim_state"]
        lease_state = authority["coordination_lease_state"]
        if source == "current":
            allowed = {
                "running": {"running", "expired"},
                "succeeded": {"running", "expired", "succeeded"},
                "failed": {"running", "released", "expired"},
                "blocked": {"running", "released", "expired"},
            }
            state_valid = (
                coordination_status in allowed[execution_status]
                and not (
                    coordination_status == "succeeded"
                    and execution_status != "succeeded"
                )
            )
        elif execution_status == "succeeded":
            state_valid = (
                coordination_status == "succeeded"
                and (claim_state, lease_state)
                in {("released", "released"), ("completed", "completed")}
            )
        else:
            state_valid = (
                execution_status in {"failed", "blocked"}
                and (claim_state, coordination_status, lease_state)
                in {
                    ("released", "released", "released"),
                    ("expired", "expired", "expired"),
                    ("expired", "expired", "superseded"),
                }
            )
        if not state_valid:
            raise VRIFRuntimeSettlementError(
                f"{label} binding status projection differs"
            )
        normalized_bindings.append(dict(item))
    if [row["attempt_id"] for row in normalized_bindings] != sorted(
        {row["attempt_id"] for row in normalized_bindings}
    ):
        raise VRIFRuntimeSettlementError(f"{label} binding order differs")
    observed_retired_ids.sort()
    if (
        binding["schema"] != VRIF_LANE_CROSS_STORE_BINDING_SCHEMA
        or binding["coordination_task_claim_count"]
        != coordination_task_claim_count
        or binding["retired_coordination_task_claim_count"]
        != retired_claim_count
        or len(current_authority_rows) != coordination_task_claim_count
        or binding["current_coordination_authority_bindings_cid"]
        != _content_id(current_authority_rows)
        or len(execution_authority_rows) != execution_attempt_count
        or binding["execution_authority_rows_cid"]
        != _content_id(execution_authority_rows)
        or binding["execution_attempt_count"] != execution_attempt_count
        or binding["matched_execution_attempt_count"] != execution_attempt_count
        or binding["current_matched_execution_attempt_count"]
        + binding["retired_matched_execution_attempt_count"]
        != execution_attempt_count
        or binding["retired_matched_execution_attempt_count"]
        != len(retired_ids)
        or binding["current_matched_execution_attempt_count"]
        > coordination_task_claim_count
        or retired_ids != expected_retired_ids
        or retired_ids != observed_retired_ids
        or binding["retired_matched_execution_attempt_ids_cid"]
        != _content_id(retired_ids)
        or binding["bindings_cid"] != _content_id(normalized_bindings)
    ):
        raise VRIFRuntimeSettlementError(f"{label} differs")
    _validate_section_cid(binding, label=label)


def _validate_merge_queue_verification(
    value: Any,
    *,
    queue: Mapping[str, Any],
    active_request_ids: Sequence[str],
) -> None:
    verification = _require_exact_keys(
        value,
        {
            "schema",
            "finished_at_markers_consistent",
            "target_bindings_consistent",
            "verified_row_count",
            "verified_rows_cid",
            "queue_snapshot_cid",
            "active_requests",
            "snapshot_cid",
        },
        label="merge queue aggregate verification",
    )
    if (
        verification["schema"] != VRIF_MERGE_QUEUE_VERIFICATION_SCHEMA
        or verification["finished_at_markers_consistent"] is not True
        or verification["target_bindings_consistent"] is not True
        or type(verification["verified_row_count"]) is not int
        or verification["verified_row_count"] != queue["row_count"]
        or type(verification["verified_rows_cid"]) is not str
        or _SHA256_CID.fullmatch(verification["verified_rows_cid"]) is None
        or verification["queue_snapshot_cid"] != queue["snapshot_cid"]
    ):
        raise VRIFRuntimeSettlementError("merge queue aggregate verification differs")
    rows = verification["active_requests"]
    if not isinstance(rows, list) or len(rows) != len(active_request_ids):
        raise VRIFRuntimeSettlementError("merge queue verified active rows differ")
    active_requests: list[dict[str, str]] = []
    for ordinal, row in enumerate(rows):
        item = _require_exact_keys(
            row,
            {"request_id", "status"},
            label=f"merge queue verified active row {ordinal}",
        )
        request_id = _validate_bounded_text(
            item["request_id"],
            label=f"merge queue verified request {ordinal}",
            max_bytes=512,
        )
        if item["status"] not in _MERGE_QUEUE_ACTIVE_STATES:
            raise VRIFRuntimeSettlementError(
                "merge queue verified active status is unknown"
            )
        active_requests.append(
            {"request_id": request_id, "status": item["status"]}
        )
    if [row["request_id"] for row in active_requests] != list(active_request_ids):
        raise VRIFRuntimeSettlementError("merge queue verified active IDs differ")
    observed = {state: 0 for state in _MERGE_QUEUE_ACTIVE_STATES}
    for row in active_requests:
        observed[row["status"]] += 1
    if any(observed[state] != queue["status_counts"][state] for state in observed):
        raise VRIFRuntimeSettlementError(
            "merge queue verified active status counts differ"
        )
    if queue["snapshot_cid"] != _content_id(
        _queue_snapshot_material(queue, active_requests)
    ):
        raise VRIFRuntimeSettlementError("merge queue snapshot CID differs")
    _validate_section_cid(verification, label="merge queue aggregate verification")


def _validate_pid_receipt(
    value: Any,
    *,
    expected_path: Path,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or value.get("path") != str(expected_path):
        raise VRIFRuntimeSettlementError(f"{label} path differs")
    state_value = value.get("state")
    if state_value == "absent":
        _require_exact_keys(value, {"path", "state"}, label=label)
        return
    _require_exact_keys(
        value,
        {"path", "state", "pid", "content_cid", "file"},
        label=label,
    )
    if (
        state_value != "stale_dead"
        or type(value.get("pid")) is not int
        or value["pid"] <= 0
        or type(value.get("content_cid")) is not str
        or _SHA256_CID.fullmatch(value["content_cid"]) is None
    ):
        raise VRIFRuntimeSettlementError(f"{label} observation is malformed")
    _validate_file_receipt(
        value["file"],
        expected_path=expected_path,
        label=f"{label} file",
        require_nonempty=True,
    )


def validate_vrif_runtime_settlement_receipt(
    receipt: Mapping[str, Any],
    *,
    target_repository_id: str,
    target_branch: str,
    owner_generation: int,
) -> dict[str, Any]:
    """Purely validate and normalize one aggregate settlement receipt."""
    if not isinstance(receipt, Mapping):
        raise VRIFRuntimeSettlementError("runtime settlement receipt must be an object")
    try:
        normalized = json.loads(_canonical_bytes(receipt).decode("utf-8"))
    except (TypeError, ValueError, OverflowError, UnicodeError, json.JSONDecodeError) as exc:
        raise VRIFRuntimeSettlementError(
            "runtime settlement receipt is not canonical JSON"
        ) from exc
    raw_max_active_ids = normalized.get("max_active_ids")
    if type(raw_max_active_ids) is not int:
        raise VRIFRuntimeSettlementError(
            "runtime settlement max_active_ids must be an exact integer"
        )
    _validate_public_inputs(
        target_repository_id=target_repository_id,
        target_branch=target_branch,
        owner_generation=owner_generation,
        max_active_ids=raw_max_active_ids,
        lock_timeout_seconds=0.0,
    )
    top = _require_exact_keys(
        normalized,
        {
            "schema",
            "settled",
            "max_active_ids",
            "target",
            "owner_generation",
            "profile",
            "lifecycle",
            "lanes",
            "merge_queue",
            "merge_queue_verification",
            "active_counts",
            "active_ids",
            "snapshot_cid",
            "receipt_cid",
        },
        label="runtime settlement receipt",
    )
    if (
        top["schema"] != VRIF_RUNTIME_SETTLEMENT_SCHEMA
        or type(top["settled"]) is not bool
        or type(top["owner_generation"]) is not int
        or top["owner_generation"] != owner_generation
    ):
        raise VRIFRuntimeSettlementError(
            "runtime settlement schema, state, or owner generation differs"
        )
    target = _require_exact_keys(
        top["target"],
        {"binding_schema", "repository_id", "branch"},
        label="runtime settlement target",
    )
    if target != {
        "binding_schema": MERGE_TARGET_BINDING_SCHEMA,
        "repository_id": target_repository_id,
        "branch": target_branch,
    }:
        raise VRIFRuntimeSettlementError("runtime settlement target differs")

    profile = _require_exact_keys(
        top["profile"],
        {
            "config_path",
            "config_cid",
            "config_file",
            "repository_root",
            "config_schema",
            "program_identifier",
            "board_namespace",
            "task_prefix",
            "max_lanes",
            "exit_when_all_tracks_terminal",
            "objective_refill_enabled",
            "codebase_refill_enabled",
            "objective_goal_refinement_enabled",
            "reconciliation_guardrail_enabled",
            "state_path",
            "merge_queue_path",
            "database_program",
            "lanes",
            "runtime_settlement",
            "profile_cid",
        },
        label="runtime settlement profile",
    )
    profile_content = dict(profile)
    profile_cid = profile_content.pop("profile_cid")
    repository_root_text = _validate_bounded_text(
        profile["repository_root"],
        label="runtime settlement repository root",
        max_bytes=4_096,
    )
    repository_root = Path(repository_root_text)
    if (
        not repository_root.is_absolute()
        or str(_absolute_lexical(repository_root)) != repository_root_text
    ):
        raise VRIFRuntimeSettlementError(
            "runtime settlement repository root is not canonical"
        )
    state_path = repository_root / VRIF_STATE_RELATIVE_PATH
    merge_queue_path = repository_root / VRIF_MERGE_QUEUE_RELATIVE_PATH
    config_path_text = _validate_bounded_text(
        profile["config_path"],
        label="runtime settlement config path",
        max_bytes=4_096,
    )
    config_path = Path(config_path_text)
    try:
        config_path.relative_to(repository_root)
    except ValueError as exc:
        raise VRIFRuntimeSettlementError(
            "runtime settlement config path escapes repository root"
        ) from exc
    if not config_path.is_absolute() or str(_absolute_lexical(config_path)) != config_path_text:
        raise VRIFRuntimeSettlementError(
            "runtime settlement config path is not canonical"
        )
    _validate_file_receipt(
        profile["config_file"],
        expected_path=config_path,
        label="runtime settlement config file",
        require_nonempty=True,
    )
    if (
        profile_cid != _content_id(profile_content)
        or profile["config_schema"] != VRIF_CONFIG_SCHEMA
        or profile["program_identifier"] != VRIF_PROGRAM_IDENTIFIER
        or profile["board_namespace"] != VRIF_PROGRAM_IDENTIFIER
        or profile["task_prefix"] != VRIF_TASK_PREFIX
        or type(profile["max_lanes"]) is not int
        or profile["max_lanes"] != 4
        or profile["exit_when_all_tracks_terminal"] is not True
        or profile["objective_refill_enabled"] is not False
        or profile["codebase_refill_enabled"] is not False
        or profile["objective_goal_refinement_enabled"] is not False
        or profile["reconciliation_guardrail_enabled"] is not False
        or profile["state_path"] != str(state_path)
        or profile["merge_queue_path"] != str(merge_queue_path)
        or profile["database_program"] != _EXPECTED_DATABASE_PROGRAM
        or not isinstance(profile["database_program"], Mapping)
        or profile["database_program"].get("explicit_legacy") is not False
        or profile["lanes"] != list(_EXPECTED_LANES)
        or type(profile["config_cid"]) is not str
        or _SHA256_CID.fullmatch(profile["config_cid"]) is None
    ):
        raise VRIFRuntimeSettlementError("runtime settlement profile differs")
    if not isinstance(profile["lanes"], list):
        raise VRIFRuntimeSettlementError("runtime settlement lane profile differs")
    for index, lane_identity in enumerate(profile["lanes"]):
        if (
            not isinstance(lane_identity, Mapping)
            or type(lane_identity.get("index")) is not int
            or type(lane_identity.get("strict_shard_remainder")) is not int
            or lane_identity.get("index") != index
            or lane_identity.get("strict_shard_remainder") != index
        ):
            raise VRIFRuntimeSettlementError("runtime settlement lane profile differs")
    retired_config_entries = _validate_retired_config_profile(
        profile["runtime_settlement"],
        repository_root=repository_root,
    )

    lifecycle = _require_exact_keys(
        top["lifecycle"],
        {"master_launch_fence", "master_pid", "lane_supervisor_pids"},
        label="runtime lifecycle settlement",
    )
    _validate_file_receipt(
        lifecycle["master_launch_fence"],
        expected_path=state_path / ".configured-board-master.pid.update.lock",
        label="master launch fence",
    )
    _validate_pid_receipt(
        lifecycle["master_pid"],
        expected_path=state_path / "configured-board-master.pid",
        label="master PID observation",
    )
    lane_pid_rows = lifecycle["lane_supervisor_pids"]
    if not isinstance(lane_pid_rows, list) or len(lane_pid_rows) != 4:
        raise VRIFRuntimeSettlementError("lane PID observation inventory differs")
    for index, row in enumerate(lane_pid_rows):
        row_map = _require_exact_keys(
            row,
            {"index", "observation"},
            label=f"lane {index} PID observation",
        )
        if type(row_map["index"]) is not int or row_map["index"] != index:
            raise VRIFRuntimeSettlementError("lane PID observation order differs")
        _validate_pid_receipt(
            row_map["observation"],
            expected_path=(
                state_path
                / f"lane-{index}"
                / f"vrif_lane_{index}_supervisor.pid"
            ),
            label=f"lane {index} PID observation",
        )

    lane_rows = top["lanes"]
    if not isinstance(lane_rows, list) or len(lane_rows) != 4:
        raise VRIFRuntimeSettlementError("runtime lane snapshot inventory differs")
    derived_ids: list[str] = []
    coordination_active = 0
    execution_active = 0
    for index, lane in enumerate(lane_rows):
        lane_map = _require_exact_keys(
            lane,
            {
                "index",
                "identity",
                "directory",
                "logical_owner_session_id",
                "locks",
                "coordination",
                "execution",
                "retired_coordination_lineage",
                "cross_store_binding",
                "snapshot_cid",
            },
            label=f"runtime lane {index}",
        )
        _validate_section_cid(lane_map, label=f"runtime lane {index}")
        lane_directory = state_path / f"lane-{index}"
        logical = lane_directory / "quack-lane-control.duckdb"
        coordination_path = lane_directory / "quack-lane-control.coordination.duckdb"
        execution_path = lane_directory / "quack-lane-control.execution.duckdb"
        expected_owner = _logical_owner_id(
            logical_database_path=logical,
            coordination_path=logical,
            execution_path=execution_path,
        )
        if (
            type(lane_map["index"]) is not int
            or lane_map["index"] != index
            or lane_map["identity"] != _EXPECTED_LANES[index]
            or lane_map["directory"] != str(lane_directory)
            or lane_map["logical_owner_session_id"] != expected_owner
        ):
            raise VRIFRuntimeSettlementError(
                f"runtime lane {index} identity or path differs"
            )
        lane_identity = lane_map["identity"]
        if (
            not isinstance(lane_identity, Mapping)
            or type(lane_identity.get("index")) is not int
            or type(lane_identity.get("strict_shard_remainder")) is not int
        ):
            raise VRIFRuntimeSettlementError(
                f"runtime lane {index} identity types differ"
            )
        locks = lane_map["locks"]
        expected_lock_paths = {
            "writer": execution_path.with_name(f".{execution_path.name}.writer.lock"),
            "coordination_policy": coordination_path.with_name(
                f".{coordination_path.name}.lock"
            ),
            "execution_policy": execution_path.with_name(
                f".{execution_path.name}.lock"
            ),
        }
        locks = _require_exact_keys(
            locks,
            set(expected_lock_paths),
            label=f"runtime lane {index} locks",
        )
        for name, path in expected_lock_paths.items():
            _validate_file_receipt(
                locks[name],
                expected_path=path,
                label=f"runtime lane {index} {name} lock",
            )
        coordination_count, coordination_ids = _validate_coordination_receipt(
            lane_map["coordination"],
            expected_database_path=coordination_path,
            max_active_ids=top["max_active_ids"],
            label=f"runtime lane {index} coordination",
        )
        execution_count, execution_ids = _validate_execution_receipt(
            lane_map["execution"],
            expected_database_path=execution_path,
            expected_owner_session_id=expected_owner,
            max_active_ids=top["max_active_ids"],
            label=f"runtime lane {index} execution",
        )
        lane_retired_values = lane_map["retired_coordination_lineage"]
        if not isinstance(lane_retired_values, list):
            raise VRIFRuntimeSettlementError(
                f"runtime lane {index} retired lineage inventory differs"
            )
        expected_retired_entries = [
            (ordinal, entry)
            for ordinal, entry in enumerate(retired_config_entries)
            if entry["lane_index"] == index
        ]
        if len(lane_retired_values) != len(expected_retired_entries):
            raise VRIFRuntimeSettlementError(
                f"runtime lane {index} retired lineage inventory differs"
            )
        validated_retired_lineages: list[
            tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]
        ] = []
        for lineage_value, (config_ordinal, entry) in zip(
            lane_retired_values,
            expected_retired_entries,
            strict=True,
        ):
            lineage, authority_by_attempt, _ready_task_cids = (
                _validate_retired_coordination_lineage(
                    lineage_value,
                    entry=entry,
                    config_ordinal=config_ordinal,
                    repository_root=repository_root,
                )
            )
            validated_retired_lineages.append(
                (lineage, authority_by_attempt)
            )
        _validate_cross_store_binding(
            lane_map["cross_store_binding"],
            coordination_task_claim_count=lane_map["coordination"]["row_counts"][
                "task_claims"
            ],
            execution_attempt_count=lane_map["execution"]["row_counts"][
                "database_task_attempts"
            ],
            retired_lineages=validated_retired_lineages,
            label=f"runtime lane {index} cross-store binding",
        )
        coordination_active += coordination_count
        execution_active += execution_count
        derived_ids.extend(
            f"lane-{index}:coordination:{item}"
            for item in coordination_ids
        )
        derived_ids.extend(
            f"lane-{index}:execution:{item}"
            for item in execution_ids
        )

    queue, queue_active, queue_ids = _validate_merge_queue_receipt(
        top["merge_queue"],
        expected_queue_path=merge_queue_path,
        target_repository_id=target_repository_id,
        target_branch=target_branch,
        max_active_ids=top["max_active_ids"],
    )
    _validate_merge_queue_verification(
        top["merge_queue_verification"],
        queue=queue,
        active_request_ids=queue_ids,
    )
    derived_ids.extend(
        f"merge_queue:{item}" for item in queue_ids
    )
    derived_ids.sort()
    derived_ids = _validate_identifier_list(
        derived_ids,
        label="runtime aggregate active identities",
        maximum=top["max_active_ids"],
        max_item_bytes=2_048,
    )
    active_counts = _require_exact_keys(
        top["active_counts"],
        {"coordination", "execution", "merge_queue", "total"},
        label="runtime active counts",
    )
    expected_counts = {
        "coordination": coordination_active,
        "execution": execution_active,
        "merge_queue": queue_active,
        "total": coordination_active + execution_active + queue_active,
    }
    for key in ("coordination", "execution", "merge_queue", "total"):
        _require_nonnegative_int(
            active_counts[key], label=f"runtime active counts.{key}"
        )
    if (
        active_counts != expected_counts
        or top["active_ids"] != derived_ids
        or len(derived_ids) != expected_counts["total"]
        or top["settled"] is not (expected_counts["total"] == 0)
        or queue["settled"] is not (queue_active == 0)
    ):
        raise VRIFRuntimeSettlementError(
            "runtime settlement active state is internally inconsistent"
        )
    snapshot_content = dict(top)
    supplied_receipt_cid = snapshot_content.pop("receipt_cid")
    supplied_snapshot_cid = snapshot_content.pop("snapshot_cid")
    if supplied_snapshot_cid != _content_id(snapshot_content):
        raise VRIFRuntimeSettlementError("runtime settlement snapshot CID differs")
    receipt_content = dict(top)
    receipt_content.pop("receipt_cid")
    if supplied_receipt_cid != _content_id(receipt_content):
        raise VRIFRuntimeSettlementError("runtime settlement receipt CID differs")
    return normalized


def vrif_runtime_settlement_binding(
    receipt: Mapping[str, Any],
    *,
    target_repository_id: str,
    target_branch: str,
    owner_generation: int,
) -> dict[str, Any]:
    """Return the closed, zero-active binding consumed by the root-goal CAS."""

    normalized = validate_vrif_runtime_settlement_receipt(
        receipt,
        target_repository_id=target_repository_id,
        target_branch=target_branch,
        owner_generation=owner_generation,
    )
    zero_counts = {
        "coordination": 0,
        "execution": 0,
        "merge_queue": 0,
        "total": 0,
    }
    if (
        normalized["settled"] is not True
        or normalized["active_counts"] != zero_counts
        or normalized["active_ids"]
        or normalized["merge_queue"]["settled"] is not True
    ):
        raise VRIFRuntimeSettlementError(
            "runtime settlement binding requires an exact zero-active receipt"
        )
    retired_ready_task_cids = sorted(
        {
            task["task_cid"]
            for lane in normalized["lanes"]
            for lineage in lane["retired_coordination_lineage"]
            for task in lineage["historical_ready_tasks"]
        }
    )
    binding: dict[str, Any] = {
        "schema": VRIF_RUNTIME_SETTLEMENT_BINDING_SCHEMA,
        "settled": True,
        "receipt_cid": normalized["receipt_cid"],
        "snapshot_cid": normalized["snapshot_cid"],
        "owner_generation": owner_generation,
        "target": dict(normalized["target"]),
        "config_cid": normalized["profile"]["config_cid"],
        "profile_cid": normalized["profile"]["profile_cid"],
        "lane_snapshot_cids": [
            lane["snapshot_cid"] for lane in normalized["lanes"]
        ],
        "merge_queue_receipt_cid": normalized["merge_queue"]["receipt_cid"],
        "merge_queue_snapshot_cid": normalized["merge_queue"]["snapshot_cid"],
        "active_counts": zero_counts,
        "retired_ready_task_cids": retired_ready_task_cids,
    }
    binding["binding_id"] = _content_id(binding)
    return binding


@contextmanager
def hold_vrif_runtime_settlement(
    config_path: Path | str,
    *,
    repository_root: Path | str,
    target_repository_id: str,
    target_branch: str,
    owner_generation: int,
    max_active_ids: int = 256,
    lock_timeout_seconds: float = 0.05,
) -> Iterator[dict[str, Any]]:
    """Yield one immutable aggregate runtime snapshot through the caller CAS."""

    timeout = _validate_public_inputs(
        target_repository_id=target_repository_id,
        target_branch=target_branch,
        owner_generation=owner_generation,
        max_active_ids=max_active_ids,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    root = _absolute_lexical(repository_root)
    _require_real_directory(root, label="repository_root")
    _config, profile = _read_config(
        config_path,
        repository_root=root,
        target_branch=target_branch,
    )
    state_path = Path(profile["state_path"])
    merge_queue_path = Path(profile["merge_queue_path"])
    retired_entries = profile["runtime_settlement"][
        "retired_coordination_snapshots"
    ]
    master_fence_path = state_path / ".configured-board-master.pid.update.lock"
    lane_paths = [_lane_runtime_paths(state_path, index) for index in range(4)]

    with _hold_existing_lock(
        master_fence_path,
        label="VRIF master launch fence",
        timeout_seconds=timeout,
    ) as master_fence:
        _require_config_unchanged(profile)
        lifecycle_observations = _lifecycle_observations(state_path)
        lifecycle = {
            "master_launch_fence": master_fence,
            **lifecycle_observations,
        }
        with ExitStack() as lane_locks:
            held_locks: list[dict[str, dict[str, Any]]] = [dict() for _ in range(4)]
            held_retired_locks: dict[int, dict[str, Any]] = {}
            for index, paths in enumerate(lane_paths):
                held_locks[index]["writer"] = lane_locks.enter_context(
                    _hold_existing_lock(
                        paths["writer_lock"],
                        label=f"VRIF lane {index} execution lifetime writer lock",
                        timeout_seconds=timeout,
                    )
                )
            policy_specs: list[tuple[Path, str, int, str]] = []
            for index, paths in enumerate(lane_paths):
                policy_specs.extend(
                    (
                        (
                            paths["coordination_lock"],
                            "lane",
                            index,
                            "coordination_policy",
                        ),
                        (
                            paths["execution_lock"],
                            "lane",
                            index,
                            "execution_policy",
                        ),
                    )
                )
            for ordinal, entry in enumerate(retired_entries):
                retired_database = root / entry["database_path"]
                policy_specs.append(
                    (
                        retired_database.with_name(
                            f".{retired_database.name}.lock"
                        ),
                        "retired",
                        ordinal,
                        "coordination_policy",
                    )
                )
            for path, kind, index, name in sorted(
                policy_specs,
                key=lambda item: str(item[0]),
            ):
                lock_receipt = lane_locks.enter_context(
                    _hold_existing_lock(
                        path,
                        label=(
                            f"VRIF lane {index} {name} lock"
                            if kind == "lane"
                            else f"VRIF retired coordination {index} policy lock"
                        ),
                        timeout_seconds=timeout,
                    )
                )
                if kind == "lane":
                    held_locks[index][name] = lock_receipt
                else:
                    held_retired_locks[index] = lock_receipt

            lane_snapshots: list[dict[str, Any]] = []
            store_guards: list[tuple[Path, dict[str, Any]]] = []
            retired_store_guards: list[
                tuple[Mapping[str, Any], Path, dict[str, Any]]
            ] = []
            retired_by_lane: dict[
                int,
                list[tuple[dict[str, Any], dict[str, dict[str, Any]]]],
            ] = {index: [] for index in range(4)}
            for ordinal, entry in enumerate(retired_entries):
                lineage, authorities, retired_store_guard = (
                    _read_retired_coordination_lineage(
                        entry,
                        config_ordinal=ordinal,
                        repository_root=root,
                        policy_lock=held_retired_locks[ordinal],
                        max_active_ids=max_active_ids,
                    )
                )
                retired_by_lane[entry["lane_index"]].append(
                    (lineage, authorities)
                )
                retired_store_guards.append(
                    (entry, retired_store_guard[0], retired_store_guard[1])
                )
            active_ids: list[str] = []
            coordination_active = 0
            execution_active = 0
            for index, paths in enumerate(lane_paths):
                coordination_store = _store_identity(paths["coordination"])
                execution_store = _store_identity(paths["execution"])
                store_guards.extend(
                    (
                        (paths["coordination"], coordination_store),
                        (paths["execution"], execution_store),
                    )
                )
                owner_id = _logical_owner_id(
                    logical_database_path=paths["logical"],
                    coordination_path=paths["logical"],
                    execution_path=paths["execution"],
                )
                coordination = _read_coordination_snapshot(
                    paths["coordination"],
                    store_identity=coordination_store,
                    max_active_ids=max_active_ids,
                )
                execution = _read_execution_snapshot(
                    paths["execution"],
                    store_identity=execution_store,
                    expected_owner_session_id=owner_id,
                    max_active_ids=max_active_ids,
                )
                cross_store_binding = _read_lane_cross_store_binding(
                    paths["coordination"],
                    paths["execution"],
                    retired_lineages=retired_by_lane[index],
                )
                coordination_active += coordination["active_count"]
                execution_active += execution["active_count"]
                active_ids.extend(
                    f"lane-{index}:coordination:{item}"
                    for item in coordination["active_ids"]
                )
                active_ids.extend(
                    f"lane-{index}:execution:{item}"
                    for item in execution["active_ids"]
                )
                lane_snapshot = {
                    "index": index,
                    "identity": dict(_EXPECTED_LANES[index]),
                    "directory": str(paths["directory"]),
                    "logical_owner_session_id": owner_id,
                    "locks": held_locks[index],
                    "coordination": coordination,
                    "execution": execution,
                    "retired_coordination_lineage": [
                        _add_snapshot_cid(lineage)
                        for lineage, _authorities in retired_by_lane[index]
                    ],
                    "cross_store_binding": cross_store_binding,
                }
                lane_snapshots.append(_add_snapshot_cid(lane_snapshot))

            try:
                with hold_merge_queue_settlement(
                    merge_queue_path,
                    target_repository_id=target_repository_id,
                    target_branch=target_branch,
                    max_active_ids=max_active_ids,
                    lock_timeout_seconds=timeout,
                ) as queue_receipt:
                    queue_verification = _read_merge_queue_verification(
                        merge_queue_path,
                        queue_receipt=queue_receipt,
                        target_repository_id=target_repository_id,
                        target_branch=target_branch,
                        max_active_ids=max_active_ids,
                    )
                    active_ids.extend(
                        f"merge_queue:{item}"
                        for item in queue_receipt["active_request_ids"]
                    )
                    active_ids.sort()
                    if len(active_ids) > max_active_ids:
                        raise VRIFRuntimeSettlementError(
                            "aggregate runtime active identity bound was exceeded"
                        )
                    active_counts = {
                        "coordination": coordination_active,
                        "execution": execution_active,
                        "merge_queue": int(queue_receipt["active_count"]),
                        "total": len(active_ids),
                    }
                    if sum(
                        active_counts[key]
                        for key in ("coordination", "execution", "merge_queue")
                    ) != active_counts["total"]:
                        raise VRIFRuntimeSettlementError(
                            "aggregate runtime active counts disagree"
                        )
                    receipt: dict[str, Any] = {
                        "schema": VRIF_RUNTIME_SETTLEMENT_SCHEMA,
                        "settled": not active_ids,
                        "max_active_ids": max_active_ids,
                        "target": {
                            "binding_schema": MERGE_TARGET_BINDING_SCHEMA,
                            "repository_id": target_repository_id,
                            "branch": target_branch,
                        },
                        "owner_generation": owner_generation,
                        "profile": profile,
                        "lifecycle": lifecycle,
                        "lanes": lane_snapshots,
                        "merge_queue": queue_receipt,
                        "merge_queue_verification": queue_verification,
                        "active_counts": active_counts,
                        "active_ids": active_ids,
                    }
                    receipt["snapshot_cid"] = _content_id(receipt)
                    receipt["receipt_cid"] = _content_id(receipt)
                    normalized = validate_vrif_runtime_settlement_receipt(
                        receipt,
                        target_repository_id=target_repository_id,
                        target_branch=target_branch,
                        owner_generation=owner_generation,
                    )
                    try:
                        yield normalized
                    finally:
                        _require_config_unchanged(profile)
                        if _lifecycle_observations(state_path) != lifecycle_observations:
                            raise VRIFRuntimeSettlementError(
                                "VRIF lifecycle state changed while guarded"
                            )
                        for database_path, expected_store in store_guards:
                            _require_store_unchanged(database_path, expected_store)
                        for entry, database_path, expected_store in retired_store_guards:
                            observed_path, observed_store = (
                                _configured_retired_store_identity(
                                    entry,
                                    repository_root=root,
                                )
                            )
                            if (
                                observed_path != database_path
                                or observed_store != expected_store
                            ):
                                raise VRIFRuntimeSettlementError(
                                    "retired coordination snapshot changed while guarded"
                                )
            except MergeQueueIntegrityError as exc:
                raise VRIFRuntimeSettlementError(
                    "VRIF merge queue settlement is unavailable"
                ) from exc


def read_vrif_runtime_settlement(
    config_path: Path | str,
    *,
    repository_root: Path | str,
    target_repository_id: str,
    target_branch: str,
    owner_generation: int,
    max_active_ids: int = 256,
    lock_timeout_seconds: float = 0.05,
) -> dict[str, Any]:
    """Return a convenience snapshot; use ``hold_*`` to authorize a CAS."""

    with hold_vrif_runtime_settlement(
        config_path,
        repository_root=repository_root,
        target_repository_id=target_repository_id,
        target_branch=target_branch,
        owner_generation=owner_generation,
        max_active_ids=max_active_ids,
        lock_timeout_seconds=lock_timeout_seconds,
    ) as receipt:
        return receipt


__all__ = [
    "VRIF_RUNTIME_SETTLEMENT_BINDING_SCHEMA",
    "VRIF_RUNTIME_SETTLEMENT_CONFIG_SCHEMA",
    "VRIF_RUNTIME_SETTLEMENT_SCHEMA",
    "VRIF_SCHEDULER_CONFIG_RELATIVE_PATH",
    "VRIF_TERMINAL_SIDECAR_CHECKPOINT_SCHEMA",
    "VRIFRuntimeSettlementError",
    "checkpoint_vrif_terminal_sidecars",
    "hold_vrif_runtime_settlement",
    "read_vrif_runtime_settlement",
    "validate_vrif_runtime_settlement_receipt",
    "vrif_runtime_settlement_binding",
]
