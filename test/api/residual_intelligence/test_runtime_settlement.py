from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    TASK_COMPLETION_PREPARATION_SCHEMA,
    DatabaseCoordinator,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.runtime.vrif_runtime_settlement import (
    VRIF_RUNTIME_SETTLEMENT_BINDING_SCHEMA,
    VRIFRuntimeSettlementError,
    hold_vrif_runtime_settlement,
    read_vrif_runtime_settlement,
    validate_vrif_runtime_settlement_receipt,
    vrif_runtime_settlement_binding,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_execution_schema import (
    install_database_execution_schema,
)

_TARGET_REPOSITORY_ID = (
    "repository:baguqeeraul4vqj7wze6dfjxogue57aadnvwrzw55527c2kfafyiyvuoaw2ca"
)
_TARGET_BRANCH = "codex/verified-residual-intelligence-foundry-v1"
_STATE_RELATIVE = "data/agent_supervisor/residual_intelligence_foundry/state"
_QUEUE_RELATIVE = "data/agent_supervisor/residual_intelligence_foundry/merge-queue"
_LANES = [
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
]


@dataclass
class RuntimeFixture:
    root: Path
    config_path: Path
    state_path: Path
    queue_path: Path
    queue: MergeQueue
    coordination_paths: list[Path]
    execution_paths: list[Path]
    owner_ids: list[str]


@dataclass
class RetiredRolloverFixture:
    lane_index: int
    attempt_id: str
    database_path: Path
    wal_path: Path
    policy_lock_path: Path
    ready_task_cid: str


def _owner_id(lane_directory: Path) -> str:
    logical = lane_directory / "quack-lane-control.duckdb"
    execution = lane_directory / "quack-lane-control.execution.duckdb"
    payload = "\n".join(
        str(path.absolute()) for path in (logical, logical, execution)
    ).encode("utf-8")
    return f"embedded-store:{hashlib.sha256(payload).hexdigest()[:32]}"


def _config() -> dict[str, Any]:
    return {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "verified-residual-intelligence-foundry.scheduler_config@1"
        ),
        "program_identifier": (
            "agent-supervisor-verified-residual-intelligence-foundry-v1"
        ),
        "board_namespace": (
            "agent-supervisor-verified-residual-intelligence-foundry-v1"
        ),
        "task_prefix": "VRIF-",
        "merge_target_branch": _TARGET_BRANCH,
        "max_lanes": 4,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
        "objective_goal_refinement_enabled": False,
        "reconciliation_guardrail_enabled": False,
        "database_program": {
            "authority_mode": "quack",
            "task_source_kind": "duckdb",
            "endpoint_secret_handle": "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
            "quack_endpoint": "quack:127.0.0.1:41327",
            "store_id": (
                "data/agent_supervisor/residual_intelligence_foundry/control.duckdb"
            ),
            "store_generation": "vrif-v1",
            "schema_revision": "1",
            "event_store_path": (
                "data/agent_supervisor/residual_intelligence_foundry/events"
            ),
            "runtime_registry_path": (
                "data/agent_supervisor/residual_intelligence_foundry/registry"
            ),
            "worktree_root": (
                "data/agent_supervisor/residual_intelligence_foundry/worktrees"
            ),
            "export_profile": "vrif-v1",
            "failover_policy": "fail_closed",
            "explicit_legacy": False,
        },
        "runtime_paths": {
            "state": _STATE_RELATIVE,
            "merge_queue": _QUEUE_RELATIVE,
            "generated_runtime_artifacts_are_completion_authority": False,
        },
        "runtime_settlement": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "vrif-runtime-settlement-config@1"
            ),
            "retired_coordination_snapshots": [],
        },
        "lanes": list(_LANES),
    }


def _write_config(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _runtime(tmp_path: Path) -> RuntimeFixture:
    root = tmp_path / "repository"
    root.mkdir()
    config_path = root / "config" / "scheduler.json"
    _write_config(config_path, _config())
    state_path = root / _STATE_RELATIVE
    queue_path = root / _QUEUE_RELATIVE
    state_path.mkdir(parents=True)
    queue_path.mkdir(parents=True)
    (state_path / ".configured-board-master.pid.update.lock").touch()
    coordination_paths: list[Path] = []
    execution_paths: list[Path] = []
    owner_ids: list[str] = []
    for index in range(4):
        lane = state_path / f"lane-{index}"
        lane.mkdir()
        coordination = lane / "quack-lane-control.coordination.duckdb"
        execution = lane / "quack-lane-control.execution.duckdb"
        coordinator = DatabaseCoordinator(coordination).open()
        coordinator.close()
        owner_id = _owner_id(lane)
        install_database_execution_schema(
            execution,
            metadata={
                "authority_mode": "quack",
                "logical_owner_session_id": owner_id,
                "process_instance_id": f"process:{index + 1:024x}",
                "state_schema_revision": "1",
                "control_schema_profile_id": "bootstrap",
                "control_schema_fingerprint": "bootstrap",
            },
        )
        with open_duckdb_connection(execution) as connection:
            connection.execute(
                """
                UPDATE daemon_execution_metadata
                SET value = ''
                WHERE key IN (
                    'control_schema_profile_id',
                    'control_schema_fingerprint'
                )
                """
            )
        execution.with_name(f".{execution.name}.writer.lock").touch()
        coordination_paths.append(coordination)
        execution_paths.append(execution)
        owner_ids.append(owner_id)
    queue = MergeQueue(
        queue_path,
        target_repository_id=_TARGET_REPOSITORY_ID,
        target_branch=_TARGET_BRANCH,
        require_target_binding=True,
    )
    return RuntimeFixture(
        root=root,
        config_path=config_path,
        state_path=state_path,
        queue_path=queue_path,
        queue=queue,
        coordination_paths=coordination_paths,
        execution_paths=execution_paths,
        owner_ids=owner_ids,
    )


def _read(runtime: RuntimeFixture, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "repository_root": runtime.root,
        "target_repository_id": _TARGET_REPOSITORY_ID,
        "target_branch": _TARGET_BRANCH,
        "owner_generation": 7,
        "lock_timeout_seconds": 0.0,
    }
    arguments.update(overrides)
    return read_vrif_runtime_settlement(runtime.config_path, **arguments)


def _filesystem_snapshot(root: Path) -> dict[str, tuple[Any, ...]]:
    result: dict[str, tuple[Any, ...]] = {}
    for path in sorted(root.rglob("*")):
        details = path.lstat()
        result[str(path.relative_to(root))] = (
            details.st_mode,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
            path.read_bytes() if path.is_file() else None,
        )
    return result


def _content_id(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _readdress_section(value: dict[str, Any], cid_field: str = "snapshot_cid") -> None:
    material = dict(value)
    material.pop(cid_field, None)
    value[cid_field] = _content_id(material)


def _readdress_runtime_receipt(receipt: dict[str, Any]) -> None:
    material = dict(receipt)
    material.pop("receipt_cid", None)
    material.pop("snapshot_cid", None)
    receipt["snapshot_cid"] = _content_id(material)
    receipt_material = dict(receipt)
    receipt_material.pop("receipt_cid", None)
    receipt["receipt_cid"] = _content_id(receipt_material)


def _readdress_queue_receipt(receipt: dict[str, Any]) -> None:
    queue = receipt["merge_queue"]
    verification = receipt["merge_queue_verification"]
    queue["snapshot_cid"] = _content_id(
        {
            "database": queue["database"],
            "store_metadata_cid": queue["store"]["metadata_cid"],
            "store_metadata_rows": queue["store"]["metadata_rows"],
            "store_id": queue["store"]["store_id"],
            "store_generation": queue["store"]["generation"],
            "row_count": queue["row_count"],
            "max_updated_at": queue["max_updated_at"],
            "max_claim_generation": queue["max_claim_generation"],
            "status_counts": queue["status_counts"],
            "active_requests": verification["active_requests"],
        }
    )
    queue_material = dict(queue)
    queue_material.pop("receipt_cid", None)
    queue["receipt_cid"] = _content_id(queue_material)
    verification["queue_snapshot_cid"] = queue["snapshot_cid"]
    _readdress_section(verification)
    _readdress_runtime_receipt(receipt)


def _insert_execution_attempt(
    runtime: RuntimeFixture,
    *,
    lane: int = 0,
    status: str = "running",
    phase: str = "claimed",
    finished_at_ms: int | None = None,
) -> str:
    attempt_id = f"attempt:test:{lane}:{status}"
    claim_id = f"claim:{lane}"
    task_cid = f"task:{lane}"
    lease_id = f"lease:{lane}"
    if status == "succeeded":
        authority_state = "released"
        authority_attempt_status = "succeeded"
        released_at_ms: int | None = 2
        authority_finished_at_ms: int | None = 2
    elif status in {"failed", "blocked"}:
        authority_state = "released"
        authority_attempt_status = "released"
        released_at_ms = 2
        authority_finished_at_ms = 2
    else:
        authority_state = "accepted"
        authority_attempt_status = "running"
        released_at_ms = None
        authority_finished_at_ms = None
    with open_duckdb_connection(runtime.coordination_paths[lane]) as connection:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            """
            INSERT INTO coordination_tasks(
                task_cid, task_id, registered_at_ms, ready
            ) VALUES (?, ?, 1, FALSE)
            """,
            [task_cid, f"VRIF-{lane + 9:03d}"],
        )
        connection.execute(
            """
            INSERT INTO fenced_leases(
                lease_id, lease_kind, scope_key, scope, mode,
                owner_session_id, fencing_token, fence_epoch,
                acquired_at_ms, expires_at_ms, state, revision,
                task_cid, claim_id, attempt_id, attempt_number
            ) VALUES (?, 'task', ?, ?, 'exclusive', ?, 1, 1,
                      1, 100, ?, 1, ?, ?, ?, 1)
            """,
            [
                lease_id,
                f"task:{task_cid}",
                task_cid,
                runtime.owner_ids[lane],
                authority_state,
                task_cid,
                claim_id,
                attempt_id,
            ],
        )
        connection.execute(
            """
            INSERT INTO task_claims(
                claim_id, task_cid, owner_session_id, fencing_token,
                fence_epoch, claimed_at_ms, expires_at_ms, released_at_ms,
                state, revision, attempt_id, attempt_number, lease_id
            ) VALUES (?, ?, ?, 1, 1, 1, 100, ?, ?, 1, ?, 1, ?)
            """,
            [
                claim_id,
                task_cid,
                runtime.owner_ids[lane],
                released_at_ms,
                authority_state,
                attempt_id,
                lease_id,
            ],
        )
        connection.execute(
            """
            INSERT INTO task_attempts(
                attempt_id, task_cid, attempt_number, owner_session_id,
                fencing_token, fence_epoch, started_at_ms, finished_at_ms,
                status, revision
            ) VALUES (?, ?, 1, ?, 1, 1, 1, ?, ?, 1)
            """,
            [
                attempt_id,
                task_cid,
                runtime.owner_ids[lane],
                authority_finished_at_ms,
                authority_attempt_status,
            ],
        )
        connection.execute(
            """
            INSERT INTO token_history(
                scope_key, fencing_token, fence_epoch, recorded_at_ms
            ) VALUES (?, 1, 1, 1)
            """,
            [f"task:{task_cid}"],
        )
        connection.execute(
            """
            INSERT INTO lease_events(
                event_id, lease_id, scope_key, event_type, fencing_token,
                fence_epoch, observed_at_ms, body_json
            ) VALUES (?, ?, ?, 'task_claimed', 1, 1, 1, ?)
            """,
            [
                f"lease-event:{lane}",
                lease_id,
                f"task:{task_cid}",
                json.dumps(
                    {
                        "claim_id": claim_id,
                        "attempt_id": attempt_id,
                        "attempt_number": 1,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ],
        )
        connection.commit()
    with open_duckdb_connection(runtime.execution_paths[lane]) as connection:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            """
            INSERT INTO database_task_attempts(
                attempt_id, claim_id, task_cid, task_alias, attempt_number,
                owner_session_id, fencing_token, fence_epoch, lease_id,
                committed_phase, status, started_at_ms, finished_at_ms,
                revision, body_json
            ) VALUES (?, ?, ?, '', 1, ?, 1, 1, ?, ?, ?, 1, ?, 1, '{}')
            """,
            [
                attempt_id,
                claim_id,
                task_cid,
                runtime.owner_ids[lane],
                lease_id,
                phase,
                status,
                finished_at_ms,
            ],
        )
        connection.execute(
            """
            INSERT INTO attempt_phases(
                attempt_id, phase, committed_at_ms, fencing_token,
                fence_epoch, revision, body_json
            ) VALUES (?, ?, 1, 1, 1, 1, '{}')
            """,
            [attempt_id, phase],
        )
        if phase != "claimed":
            connection.execute(
                """
                INSERT INTO attempt_phases(
                    attempt_id, phase, committed_at_ms, fencing_token,
                    fence_epoch, revision, body_json
                ) VALUES (?, 'claimed', 0, 1, 1, 1, '{}')
                """,
                [attempt_id],
            )
        connection.commit()
    return attempt_id


def _file_cid(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_retired_rollover(
    runtime: RuntimeFixture,
    *,
    lane: int = 2,
) -> RetiredRolloverFixture:
    if lane not in {2, 3}:
        raise ValueError("test retirement lane must be 2 or 3")
    attempt_id = _insert_execution_attempt(
        runtime,
        lane=lane,
        status="succeeded",
        phase="complete",
        finished_at_ms=2,
    )
    ready_task_alias = f"VRIF-{lane + 11:03d}"
    ready_task_cid = f"task:retired-ready:{ready_task_alias.lower()}"
    coordination = runtime.coordination_paths[lane]
    with open_duckdb_connection(coordination) as connection:
        connection.execute("PRAGMA disable_checkpoint_on_shutdown")
        connection.execute(
            """
            INSERT INTO coordination_tasks(
                task_cid, task_id, registered_at_ms, ready, body_json
            ) VALUES (?, ?, 3, TRUE, '{}')
            """,
            [ready_task_cid, ready_task_alias],
        )
    wal = Path(str(coordination) + ".wal")
    assert wal.is_file() and wal.stat().st_size > 0
    archive_directory = (
        runtime.state_path
        / "sidecar-quarantine"
        / "test-rollover"
        / f"lane-{lane}"
    )
    archive_directory.mkdir(parents=True)
    archive_database = archive_directory / coordination.name
    archive_wal = Path(str(archive_database) + ".wal")
    coordination.rename(archive_database)
    wal.rename(archive_wal)
    archive_policy_lock = archive_database.with_name(
        f".{archive_database.name}.lock"
    )
    archive_policy_lock.touch()

    coordinator = DatabaseCoordinator(coordination).open()
    coordinator.close()
    config = json.loads(runtime.config_path.read_text(encoding="utf-8"))
    attempt_ids = [attempt_id]
    entries = config["runtime_settlement"]["retired_coordination_snapshots"]
    entries.append(
        {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "vrif-retired-coordination-snapshot@1"
            ),
            "lane_index": lane,
            "database_path": str(archive_database.relative_to(runtime.root)),
            "database_size_bytes": archive_database.stat().st_size,
            "database_sha256": _file_cid(archive_database),
            "wal_path": str(archive_wal.relative_to(runtime.root)),
            "wal_size_bytes": archive_wal.stat().st_size,
            "wal_sha256": _file_cid(archive_wal),
            "terminal_execution_attempt_ids": attempt_ids,
            "terminal_execution_attempt_ids_cid": _content_id(attempt_ids),
        }
    )
    entries.sort(key=lambda entry: (entry["lane_index"], entry["database_path"]))
    _write_config(runtime.config_path, config)
    return RetiredRolloverFixture(
        lane_index=lane,
        attempt_id=attempt_id,
        database_path=archive_database,
        wal_path=archive_wal,
        policy_lock_path=archive_policy_lock,
        ready_task_cid=ready_task_cid,
    )


def _refresh_retired_config_hashes(
    runtime: RuntimeFixture,
    retired: RetiredRolloverFixture,
) -> None:
    config = json.loads(runtime.config_path.read_text(encoding="utf-8"))
    entry = config["runtime_settlement"]["retired_coordination_snapshots"][0]
    entry["database_size_bytes"] = retired.database_path.stat().st_size
    entry["database_sha256"] = _file_cid(retired.database_path)
    entry["wal_size_bytes"] = retired.wal_path.stat().st_size
    entry["wal_sha256"] = _file_cid(retired.wal_path)
    _write_config(runtime.config_path, config)


def _insert_preparation_completion(
    runtime: RuntimeFixture,
    *,
    mutation: str | None = None,
) -> None:
    _insert_execution_attempt(
        runtime,
        status="succeeded",
        phase="complete",
        finished_at_ms=2,
    )
    body: dict[str, Any] = {
        "schema": TASK_COMPLETION_PREPARATION_SCHEMA,
        "task_cid": "task:0",
        "claim_id": "claim:0",
        "owner_session_id": runtime.owner_ids[0],
        "fencing_token": 1,
        "fence_epoch": 1,
        "attempt_id": "attempt:test:0:succeeded",
        "attempt_number": 1,
        "lease_id": "lease:0",
        "control_expected_revision": 1,
        "control_expected_status": "completed",
        "evidence_digest": "sha256:evidence",
        "prepared_at_ms": 1,
    }
    body["preparation_digest"] = _content_id(body)
    if mutation == "preparation_digest":
        body["preparation_digest"] = "sha256:" + "0" * 64
    elif mutation is not None:
        field, replacement = {
            "control_expected_revision": ("control_expected_revision", 0),
            "control_expected_status": ("control_expected_status", ""),
            "evidence_digest": ("evidence_digest", ""),
            "prepared_at_ms": ("prepared_at_ms", -1),
        }[mutation]
        body[field] = replacement
        material = dict(body)
        material.pop("preparation_digest")
        body["preparation_digest"] = _content_id(material)
    with open_duckdb_connection(runtime.coordination_paths[0]) as connection:
        connection.execute(
            """
            INSERT INTO task_completions(
                task_cid, completed_at_ms, status, body_json
            ) VALUES ('task:0', 2, 'succeeded', ?)
            """,
            [json.dumps(body, sort_keys=True, separators=(",", ":"))],
        )


def _enqueue_final(runtime: RuntimeFixture, ordinal: int = 0) -> str:
    request = runtime.queue.enqueue(
        branch_name=f"candidate/final-{ordinal}",
        task_id="VRIF-032",
        canonical_task_id=f"task:vrif-032:{ordinal}",
        commit_sha=f"{ordinal + 1:040x}",
        metadata={"kind": "final-task-race"},
    )
    return request.request_id


def test_runtime_settlement_is_deterministic_content_addressed_and_read_only(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    stale_pid_path = (
        runtime.state_path / "lane-0" / "vrif_lane_0_supervisor.pid"
    )
    stale_pid_path.write_text("2147483647\n", encoding="ascii")
    before = _filesystem_snapshot(runtime.root)

    first = _read(runtime)
    second = _read(runtime)

    assert _filesystem_snapshot(runtime.root) == before
    assert first == second
    assert first["settled"] is True
    assert first["active_counts"] == {
        "coordination": 0,
        "execution": 0,
        "merge_queue": 0,
        "total": 0,
    }
    assert first["active_ids"] == []
    assert first["owner_generation"] == 7
    assert len(first["lanes"]) == 4
    assert [lane["index"] for lane in first["lanes"]] == [0, 1, 2, 3]
    assert (
        first["lifecycle"]["lane_supervisor_pids"][0]["observation"]["state"]
        == "stale_dead"
    )
    assert first["receipt_cid"].startswith("sha256:")
    assert first["snapshot_cid"].startswith("sha256:")
    assert validate_vrif_runtime_settlement_receipt(
        first,
        target_repository_id=_TARGET_REPOSITORY_ID,
        target_branch=_TARGET_BRANCH,
        owner_generation=7,
    ) == first


def test_runtime_settlement_reports_active_final_task_without_false_success(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    request_id = _enqueue_final(runtime)

    receipt = _read(runtime)

    assert receipt["settled"] is False
    assert receipt["merge_queue"]["settled"] is False
    assert receipt["active_counts"] == {
        "coordination": 0,
        "execution": 0,
        "merge_queue": 1,
        "total": 1,
    }
    assert receipt["active_ids"] == [f"merge_queue:{request_id}"]


def test_runtime_settlement_reports_running_execution_attempt(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    attempt_id = _insert_execution_attempt(runtime)

    receipt = _read(runtime)

    assert receipt["settled"] is False
    assert receipt["active_counts"]["coordination"] == 3
    assert receipt["active_counts"]["execution"] == 2
    assert receipt["active_ids"] == [
        "lane-0:coordination:fenced_lease:lease:0",
        "lane-0:coordination:task_attempt:" + attempt_id,
        "lane-0:coordination:task_claim:claim:0",
        f"lane-0:execution:execution_attempt:{attempt_id}",
        f"lane-0:execution:execution_phase:{attempt_id}:claimed",
    ]


def test_ready_unclaimed_coordination_task_prevents_false_settlement(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    coordinator = DatabaseCoordinator(runtime.coordination_paths[2]).open()
    try:
        coordinator.register_task(
            task_cid="task:ready",
            task_id="VRIF-032",
            now_ms=1,
        )
    finally:
        coordinator.close()

    receipt = _read(runtime)

    assert receipt["settled"] is False
    assert receipt["active_counts"]["coordination"] == 1
    assert receipt["active_ids"] == [
        "lane-2:coordination:coordination_task:task:ready"
    ]


def test_ordinary_public_completion_is_terminal_and_bound_to_registry(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    coordinator = DatabaseCoordinator(runtime.coordination_paths[1]).open()
    try:
        coordinator.register_task(
            task_cid="task:bootstrap-complete",
            task_id="VRIF-008",
            now_ms=1,
        )
        coordinator.mark_task_complete(
            "task:bootstrap-complete",
            body={"source": "bootstrap"},
            now_ms=2,
        )
    finally:
        coordinator.close()

    receipt = _read(runtime)

    assert receipt["settled"] is True
    assert receipt["active_counts"]["coordination"] == 0


def test_coordination_dependency_endpoints_must_exist(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    with open_duckdb_connection(runtime.coordination_paths[0]) as connection:
        connection.execute(
            "INSERT INTO task_dependencies VALUES ('task:missing', 'dep:missing')"
        )

    with pytest.raises(VRIFRuntimeSettlementError, match="dependency endpoint"):
        _read(runtime)


def test_runtime_settlement_rejects_busy_execution_lifetime_writer(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    execution = runtime.execution_paths[2]
    writer_lock = execution.with_name(f".{execution.name}.writer.lock")
    descriptor = os.open(writer_lock, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(VRIFRuntimeSettlementError, match="writer lock is busy"):
            _read(runtime)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_live_supervisor_marker_blocks_otherwise_idle_runtime(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    marker = runtime.state_path / "lane-1" / "vrif_lane_1_supervisor.pid"
    marker.write_text(f"{os.getpid()}\n", encoding="ascii")

    with pytest.raises(VRIFRuntimeSettlementError, match="live or reused"):
        _read(runtime)


def test_runtime_guard_excludes_master_launch_and_final_queue_writer(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    master_lock = runtime.state_path / ".configured-board-master.pid.update.lock"
    master_started = threading.Event()
    master_entered = threading.Event()
    queue_started = threading.Event()
    queue_finished = threading.Event()
    failures: list[BaseException] = []

    def competing_master() -> None:
        descriptor = os.open(master_lock, os.O_RDONLY)
        master_started.set()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            master_entered.set()
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            os.close(descriptor)

    def competing_queue_writer() -> None:
        queue_started.set()
        try:
            _enqueue_final(runtime, 1)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            queue_finished.set()

    master = threading.Thread(target=competing_master, daemon=True)
    queue_writer = threading.Thread(target=competing_queue_writer, daemon=True)
    with hold_vrif_runtime_settlement(
        runtime.config_path,
        repository_root=runtime.root,
        target_repository_id=_TARGET_REPOSITORY_ID,
        target_branch=_TARGET_BRANCH,
        owner_generation=7,
        lock_timeout_seconds=0.0,
    ) as receipt:
        assert receipt["settled"] is True
        master.start()
        queue_writer.start()
        assert master_started.wait(timeout=1.0)
        assert queue_started.wait(timeout=1.0)
        assert master_entered.wait(timeout=0.1) is False
        assert queue_finished.wait(timeout=0.1) is False

    master.join(timeout=5.0)
    queue_writer.join(timeout=5.0)
    assert master.is_alive() is False
    assert queue_writer.is_alive() is False
    assert master_entered.is_set()
    assert queue_finished.is_set()
    assert failures == []


def _leave_committed_wal(database_path: Path) -> Path:
    script = """
import os
import sys
import duckdb
path = sys.argv[1]
connection = duckdb.connect(path)
connection.execute(
    \"INSERT INTO daemon_execution_events VALUES "
    "('event:wal', '', '', 'wal_test', 1, '{}')\"
)
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(database_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    wal_path = Path(str(database_path) + ".wal")
    assert wal_path.is_file() and wal_path.stat().st_size > 0
    return wal_path


def test_optional_committed_sidecar_wal_is_bound_and_read_only(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    wal_path = _leave_committed_wal(runtime.execution_paths[0])
    before = _filesystem_snapshot(runtime.root)

    receipt = _read(runtime)

    assert receipt["settled"] is True
    wal = receipt["lanes"][0]["execution"]["store"]["wal"]
    assert wal["state"] == "present"
    assert wal["file"]["path"] == str(wal_path)
    assert _filesystem_snapshot(runtime.root) == before


def test_sidecar_wal_identity_change_during_guard_fails_closed(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    wal_path = _leave_committed_wal(runtime.execution_paths[0])

    with pytest.raises(VRIFRuntimeSettlementError, match="WAL identity changed"):
        with hold_vrif_runtime_settlement(
            runtime.config_path,
            repository_root=runtime.root,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
            lock_timeout_seconds=0.0,
        ):
            os.utime(wal_path, ns=(wal_path.stat().st_atime_ns, 1))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update({"lanes": value["lanes"][:3]}),
        lambda value: value.update({"lanes": list(reversed(value["lanes"]))}),
        lambda value: value.update({"max_lanes": 3}),
        lambda value: value.update({"exit_when_all_tracks_terminal": False}),
        lambda value: value.update({"objective_refill_enabled": True}),
        lambda value: value.update({"codebase_refill_enabled": True}),
        lambda value: value.update({"objective_goal_refinement_enabled": True}),
        lambda value: value.update({"reconciliation_guardrail_enabled": True}),
        lambda value: value["database_program"].update(
            {"store_generation": "foreign-generation"}
        ),
    ),
    ids=(
        "lane-omission",
        "lane-order",
        "lane-count",
        "terminal-exit-disabled",
        "objective-refill-enabled",
        "codebase-refill-enabled",
        "goal-refinement-enabled",
        "reconciliation-enabled",
        "store-generation",
    ),
)
def test_runtime_settlement_rejects_foreign_config_profile(
    tmp_path: Path,
    mutation,
) -> None:
    runtime = _runtime(tmp_path)
    value = _config()
    mutation(value)
    _write_config(runtime.config_path, value)

    with pytest.raises(VRIFRuntimeSettlementError, match="profile|identity|store"):
        _read(runtime)


def test_runtime_settlement_rejects_target_and_owner_generation_mismatch(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)

    with pytest.raises(VRIFRuntimeSettlementError, match="profile identity"):
        _read(runtime, target_branch="foreign")
    with pytest.raises(ValueError, match="owner_generation"):
        _read(runtime, owner_generation=0)
    with pytest.raises(ValueError, match="target_repository_id"):
        _read(runtime, target_repository_id="repository:foreign")


def test_runtime_settlement_rejects_symlinked_runtime_path(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    real_state = runtime.state_path.with_name("state-real")
    runtime.state_path.rename(real_state)
    runtime.state_path.symlink_to(real_state, target_is_directory=True)

    with pytest.raises(VRIFRuntimeSettlementError, match="symbolic link"):
        _read(runtime)


def test_missing_lane_store_fails_without_recreation(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    missing = runtime.coordination_paths[3]
    missing.unlink()
    before = _filesystem_snapshot(runtime.root)

    with pytest.raises(VRIFRuntimeSettlementError, match="database.*unavailable"):
        _read(runtime)

    assert missing.exists() is False
    assert _filesystem_snapshot(runtime.root) == before


def test_runtime_settlement_rejects_execution_schema_tamper(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    with open_duckdb_connection(runtime.execution_paths[0]) as connection:
        connection.execute(
            "ALTER TABLE effect_claims ADD COLUMN unexpected VARCHAR"
        )

    with pytest.raises(VRIFRuntimeSettlementError, match="ordered-column"):
        _read(runtime)


def test_runtime_settlement_rejects_orphan_provider_invocation(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    with open_duckdb_connection(runtime.execution_paths[1]) as connection:
        connection.execute(
            """
            INSERT INTO provider_invocations(
                invocation_id, attempt_id, task_cid, idempotency_key,
                owner_session_id, recorded_at_ms, result_json
            ) VALUES ('provider:orphan', 'attempt:missing', 'task:missing',
                      'provider:key', ?, 1, '{}')
            """,
            [runtime.owner_ids[1]],
        )

    with pytest.raises(VRIFRuntimeSettlementError, match="orphaned or unbound"):
        _read(runtime)


def test_runtime_settlement_rejects_unknown_execution_state(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _insert_execution_attempt(runtime, status="paused")

    with pytest.raises(VRIFRuntimeSettlementError, match="unknown state"):
        _read(runtime)


def test_runtime_settlement_receipt_validator_rejects_tamper(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    receipt = _read(runtime)
    receipt["owner_generation"] = 8

    with pytest.raises(VRIFRuntimeSettlementError, match="owner generation"):
        validate_vrif_runtime_settlement_receipt(
            receipt,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
        )


def test_runtime_settlement_receipt_validator_rejects_noninteger_bound(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    receipt = _read(runtime)
    receipt["max_active_ids"] = 256.0
    _readdress_runtime_receipt(receipt)

    with pytest.raises(VRIFRuntimeSettlementError, match="exact integer"):
        validate_vrif_runtime_settlement_receipt(
            receipt,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
        )


def test_runtime_settlement_receipt_validator_rejects_balanced_negative_forgery(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    receipt = _read(runtime)
    lane = receipt["lanes"][0]
    coordination = lane["coordination"]
    execution = lane["execution"]
    coordination["active_count"] = -1
    coordination["active_counts"]["accepted_fenced_leases"] = -1
    execution["active_count"] = 1
    execution["active_counts"]["running_attempts"] = 1
    _readdress_section(coordination)
    _readdress_section(execution)
    _readdress_section(lane)
    _readdress_runtime_receipt(receipt)

    with pytest.raises(VRIFRuntimeSettlementError, match="nonnegative"):
        validate_vrif_runtime_settlement_receipt(
            receipt,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
        )


def test_runtime_settlement_receipt_validator_rejects_nested_field_injection(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    receipt = _read(runtime)
    lane = receipt["lanes"][0]
    lane["coordination"]["unexpected"] = "forged"
    _readdress_section(lane["coordination"])
    _readdress_section(lane)
    _readdress_runtime_receipt(receipt)

    with pytest.raises(VRIFRuntimeSettlementError, match="noncanonical field set"):
        validate_vrif_runtime_settlement_receipt(
            receipt,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
        )


def test_runtime_settlement_receipt_validator_rejects_negative_queue_count(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    receipt = _read(runtime)
    receipt["merge_queue"]["row_count"] = -1
    receipt["merge_queue"]["status_counts"]["completed"] = -1
    receipt["merge_queue_verification"]["verified_row_count"] = -1
    _readdress_queue_receipt(receipt)

    with pytest.raises(VRIFRuntimeSettlementError, match="nonnegative"):
        validate_vrif_runtime_settlement_receipt(
            receipt,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
        )


def test_runtime_settlement_receipt_validator_rejects_queue_field_injection(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    receipt = _read(runtime)
    receipt["merge_queue"]["store"]["unexpected"] = "forged"
    _readdress_queue_receipt(receipt)

    with pytest.raises(VRIFRuntimeSettlementError, match="noncanonical field set"):
        validate_vrif_runtime_settlement_receipt(
            receipt,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
        )


def test_active_merge_request_with_finished_marker_fails_closed(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    request_id = _enqueue_final(runtime)
    with runtime.queue._connect() as connection:
        connection.execute(
            "UPDATE merge_requests SET finished_at = 1 WHERE request_id = ?",
            [request_id],
        )

    with pytest.raises(VRIFRuntimeSettlementError, match="finished_at marker"):
        _read(runtime)


def test_terminal_merge_request_without_finished_marker_fails_closed(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _enqueue_final(runtime)
    claimed = runtime.queue.dequeue("test-consumer")
    assert claimed is not None
    runtime.queue.complete(claimed)
    with runtime.queue._connect() as connection:
        connection.execute(
            "UPDATE merge_requests SET finished_at = 0 WHERE request_id = ?",
            [claimed.request_id],
        )

    with pytest.raises(VRIFRuntimeSettlementError, match="finished_at marker"):
        _read(runtime)


def test_terminal_merge_request_with_foreign_target_fails_closed(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _enqueue_final(runtime)
    claimed = runtime.queue.dequeue("test-consumer")
    assert claimed is not None
    runtime.queue.complete(claimed)
    with runtime.queue._connect() as connection:
        row = connection.execute(
            "SELECT metadata_json FROM merge_requests WHERE request_id = ?",
            [claimed.request_id],
        ).fetchone()
        assert row is not None
        metadata = json.loads(row[0])
        metadata["target_repository_id"] = "repository:foreign"
        connection.execute(
            "UPDATE merge_requests SET metadata_json = ? WHERE request_id = ?",
            [json.dumps(metadata, sort_keys=True), claimed.request_id],
        )

    with pytest.raises(VRIFRuntimeSettlementError, match="target binding"):
        _read(runtime)


def test_terminal_execution_attempt_must_match_coordination_authority(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    attempt_id = _insert_execution_attempt(
        runtime,
        status="succeeded",
        phase="complete",
        finished_at_ms=2,
    )
    with open_duckdb_connection(runtime.execution_paths[0]) as connection:
        connection.execute(
            "UPDATE database_task_attempts SET claim_id = 'claim:foreign' "
            "WHERE attempt_id = ?",
            [attempt_id],
        )

    with pytest.raises(VRIFRuntimeSettlementError, match="coordination authority"):
        _read(runtime)


def test_prepared_completion_requires_preparation_identity_body(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _insert_execution_attempt(
        runtime,
        status="succeeded",
        phase="complete",
        finished_at_ms=2,
    )
    with open_duckdb_connection(runtime.coordination_paths[0]) as connection:
        connection.execute(
            "INSERT INTO task_completions VALUES ('task:0', 2, 'prepared', '{}')"
        )

    with pytest.raises(VRIFRuntimeSettlementError, match="authority identity"):
        _read(runtime)


@pytest.mark.parametrize(
    "repository_id",
    ("repository:../foreign", "repository:foreign/path", "repository:bad\x01id"),
)
def test_runtime_settlement_rejects_noncanonical_repository_identity(
    tmp_path: Path,
    repository_id: str,
) -> None:
    runtime = _runtime(tmp_path)

    with pytest.raises(ValueError, match="canonical repository identity"):
        _read(runtime, target_repository_id=repository_id)


@pytest.mark.parametrize("lane_index", [2, 3])
def test_retired_coordination_rollover_is_exact_deterministic_and_bound(
    tmp_path: Path,
    lane_index: int,
) -> None:
    runtime = _runtime(tmp_path)
    retired = _configure_retired_rollover(runtime, lane=lane_index)
    before = _filesystem_snapshot(runtime.root)

    first = _read(runtime)
    second = _read(runtime)

    assert _filesystem_snapshot(runtime.root) == before
    assert first == second
    assert first["settled"] is True
    lane = first["lanes"][lane_index]
    lineage = lane["retired_coordination_lineage"]
    assert len(lineage) == 1
    assert lineage[0]["admitted_terminal_execution_attempt_ids"] == [
        retired.attempt_id
    ]
    assert lineage[0]["historical_ready_tasks"] == [
        {
            "task_cid": retired.ready_task_cid,
            "task_id": f"VRIF-{lane_index + 11:03d}",
        }
    ]
    assert lane["cross_store_binding"][
        "retired_matched_execution_attempt_ids"
    ] == [retired.attempt_id]
    assert validate_vrif_runtime_settlement_receipt(
        first,
        target_repository_id=_TARGET_REPOSITORY_ID,
        target_branch=_TARGET_BRANCH,
        owner_generation=7,
    ) == first
    binding = vrif_runtime_settlement_binding(
        first,
        target_repository_id=_TARGET_REPOSITORY_ID,
        target_branch=_TARGET_BRANCH,
        owner_generation=7,
    )
    assert binding["schema"] == VRIF_RUNTIME_SETTLEMENT_BINDING_SCHEMA
    assert binding["settled"] is True
    assert binding["retired_ready_task_cids"] == [retired.ready_task_cid]
    assert binding["lane_snapshot_cids"] == [
        item["snapshot_cid"] for item in first["lanes"]
    ]
    material = dict(binding)
    binding_id = material.pop("binding_id")
    assert binding_id == _content_id(material)


def test_lane_two_and_three_retirements_are_jointly_admitted(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    retired = [
        _configure_retired_rollover(runtime, lane=lane_index)
        for lane_index in (2, 3)
    ]

    receipt = _read(runtime)

    assert receipt["settled"] is True
    for item in retired:
        lane = receipt["lanes"][item.lane_index]
        assert lane["retired_coordination_lineage"][0][
            "admitted_terminal_execution_attempt_ids"
        ] == [item.attempt_id]
        assert lane["cross_store_binding"][
            "retired_matched_execution_attempt_ids"
        ] == [item.attempt_id]


def test_retired_coordination_rejects_unadmitted_or_repeated_lane(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _configure_retired_rollover(runtime, lane=2)
    _configure_retired_rollover(runtime, lane=3)
    config = json.loads(runtime.config_path.read_text(encoding="utf-8"))
    entries = config["runtime_settlement"]["retired_coordination_snapshots"]

    changed = json.loads(json.dumps(config))
    changed["runtime_settlement"]["retired_coordination_snapshots"][1][
        "lane_index"
    ] = 1
    _write_config(runtime.config_path, changed)
    with pytest.raises(VRIFRuntimeSettlementError, match="admission set"):
        _read(runtime)

    entries[1]["lane_index"] = 2
    entries[1]["database_path"] = entries[1]["database_path"].replace(
        "lane-3", "lane-2"
    )
    entries[1]["wal_path"] = entries[1]["wal_path"].replace(
        "lane-3", "lane-2"
    )
    _write_config(runtime.config_path, config)
    with pytest.raises(VRIFRuntimeSettlementError, match="repeated"):
        _read(runtime)


@pytest.mark.parametrize(
    "mutation",
    ["database-hash", "database-path", "wal-hash", "wal-path"],
)
def test_retired_coordination_config_tamper_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    runtime = _runtime(tmp_path)
    _configure_retired_rollover(runtime)
    config = json.loads(runtime.config_path.read_text(encoding="utf-8"))
    entry = config["runtime_settlement"]["retired_coordination_snapshots"][0]
    if mutation == "database-hash":
        entry["database_sha256"] = "sha256:" + "0" * 64
    elif mutation == "database-path":
        entry["database_path"] = entry["database_path"].replace(
            "test-rollover", "missing-rollover"
        )
    elif mutation == "wal-hash":
        entry["wal_sha256"] = "sha256:" + "0" * 64
    else:
        entry["wal_path"] = entry["database_path"]
    _write_config(runtime.config_path, config)

    with pytest.raises(VRIFRuntimeSettlementError):
        _read(runtime)


def test_retired_coordination_active_authority_is_never_admitted(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    retired = _configure_retired_rollover(runtime)
    with open_duckdb_connection(retired.database_path) as connection:
        connection.execute("PRAGMA disable_checkpoint_on_shutdown")
        connection.execute(
            "UPDATE fenced_leases SET state = 'accepted' WHERE attempt_id = ?",
            [retired.attempt_id],
        )
        connection.execute(
            """
            UPDATE task_claims
            SET state = 'accepted', released_at_ms = NULL
            WHERE attempt_id = ?
            """,
            [retired.attempt_id],
        )
        connection.execute(
            """
            UPDATE task_attempts
            SET status = 'running', finished_at_ms = NULL
            WHERE attempt_id = ?
            """,
            [retired.attempt_id],
        )
    _refresh_retired_config_hashes(runtime, retired)

    with pytest.raises(VRIFRuntimeSettlementError, match="active authority"):
        _read(runtime)


def test_retired_coordination_identity_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    retired = _configure_retired_rollover(runtime)
    with open_duckdb_connection(runtime.execution_paths[2]) as connection:
        connection.execute(
            """
            UPDATE database_task_attempts
            SET claim_id = 'claim:foreign'
            WHERE attempt_id = ?
            """,
            [retired.attempt_id],
        )

    with pytest.raises(VRIFRuntimeSettlementError, match="coordination authority"):
        _read(runtime)


def test_current_and_retired_coordination_overlap_fails_closed(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    retired = _configure_retired_rollover(runtime)
    import duckdb

    connection = duckdb.connect(str(runtime.coordination_paths[2]))
    try:
        archive = str(retired.database_path).replace("'", "''")
        connection.execute(f"ATTACH '{archive}' AS retired (READ_ONLY)")
        for table, predicate in (
            ("coordination_tasks", "task_cid = 'task:2'"),
            ("fenced_leases", f"attempt_id = '{retired.attempt_id}'"),
            ("token_history", "scope_key = 'task:task:2'"),
            ("lease_events", "lease_id = 'lease:2'"),
            ("task_claims", f"attempt_id = '{retired.attempt_id}'"),
            ("task_attempts", f"attempt_id = '{retired.attempt_id}'"),
        ):
            connection.execute(
                f"INSERT INTO {table} SELECT * FROM retired.{table} WHERE {predicate}"
            )
        connection.execute("DETACH retired")
    finally:
        connection.close()

    with pytest.raises(VRIFRuntimeSettlementError, match="overlap"):
        _read(runtime)


def test_running_execution_cannot_use_retired_coordination_authority(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    retired = _configure_retired_rollover(runtime)
    with open_duckdb_connection(runtime.execution_paths[2]) as connection:
        connection.execute(
            """
            UPDATE database_task_attempts
            SET status = 'running', committed_phase = 'claimed',
                finished_at_ms = NULL
            WHERE attempt_id = ?
            """,
            [retired.attempt_id],
        )

    with pytest.raises(VRIFRuntimeSettlementError, match="coordination authority"):
        _read(runtime)


def test_binding_helper_rejects_well_formed_active_runtime(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _enqueue_final(runtime)
    receipt = _read(runtime)

    with pytest.raises(VRIFRuntimeSettlementError, match="zero-active"):
        vrif_runtime_settlement_binding(
            receipt,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
        )


def test_huge_pid_marker_uses_typed_fail_closed_error(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    marker = runtime.state_path / "configured-board-master.pid"
    marker.write_text("9999999999999999999999999999999\n", encoding="ascii")

    with pytest.raises(VRIFRuntimeSettlementError, match="liveness is unknown"):
        _read(runtime)


def test_retired_policy_lock_is_held_through_guarded_callback(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    retired = _configure_retired_rollover(runtime)
    descriptor = os.open(retired.policy_lock_path, os.O_RDONLY)
    try:
        with hold_vrif_runtime_settlement(
            runtime.config_path,
            repository_root=runtime.root,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
            lock_timeout_seconds=0.0,
        ):
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def test_retired_wal_identity_change_during_guard_fails_closed(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    retired = _configure_retired_rollover(runtime)

    with pytest.raises(VRIFRuntimeSettlementError, match="changed while guarded"):
        with hold_vrif_runtime_settlement(
            runtime.config_path,
            repository_root=runtime.root,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
            lock_timeout_seconds=0.0,
        ):
            details = retired.wal_path.stat()
            os.utime(
                retired.wal_path,
                ns=(details.st_atime_ns, details.st_mtime_ns + 1),
            )


def test_valid_preparation_completion_digest_is_accepted(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _insert_preparation_completion(runtime)

    receipt = _read(runtime)

    assert receipt["settled"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        "preparation_digest",
        "control_expected_revision",
        "control_expected_status",
        "evidence_digest",
        "prepared_at_ms",
    ),
)
def test_preparation_completion_authority_fields_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    runtime = _runtime(tmp_path)
    _insert_preparation_completion(runtime, mutation=mutation)

    with pytest.raises(VRIFRuntimeSettlementError, match="identity is malformed"):
        _read(runtime)


def test_pure_validator_rejects_hidden_prepared_completion(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    forged = _read(runtime)
    lane = forged["lanes"][0]
    coordination = lane["coordination"]
    coordination["row_counts"]["task_completions"] = 1
    coordination["status_counts"]["task_completions"]["prepared"] = 1
    _readdress_section(coordination)
    _readdress_section(lane)
    _readdress_runtime_receipt(forged)

    with pytest.raises(VRIFRuntimeSettlementError, match="barrier count differs"):
        validate_vrif_runtime_settlement_receipt(
            forged,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
        )


def test_pure_validator_rejects_readdressed_current_binding_forgery(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _insert_execution_attempt(
        runtime,
        status="succeeded",
        phase="complete",
        finished_at_ms=2,
    )
    forged = _read(runtime)
    lane = forged["lanes"][0]
    cross = lane["cross_store_binding"]
    cross["bindings"][0]["claim_id"] = "claim:forged"
    cross["bindings_cid"] = _content_id(cross["bindings"])
    _readdress_section(cross)
    _readdress_section(lane)
    _readdress_runtime_receipt(forged)

    with pytest.raises(VRIFRuntimeSettlementError, match="binding authority"):
        validate_vrif_runtime_settlement_receipt(
            forged,
            target_repository_id=_TARGET_REPOSITORY_ID,
            target_branch=_TARGET_BRANCH,
            owner_generation=7,
        )
