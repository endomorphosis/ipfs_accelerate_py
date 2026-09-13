"""Independent current-tree checks for DOEP-033 materialized replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    CursorReplayError,
    EventCursor,
)
from ipfs_accelerate_py.agent_supervisor.runtime.database_event_log import (
    CONSUMER_CHECKPOINT_INTERFACE,
    DATABASE_EVENT_LOG_INTERFACE,
    EVENT_CURSOR_INTERFACE,
    IDEMPOTENT_EVENT_CONSUMPTION_BINDING,
    MATERIALIZED_STATE_RECOVERY_SCHEMA,
    MATERIALIZED_STATE_REPLAY_BINDING,
    MATERIALIZED_STATE_REPLAY_CONSUMES,
    MATERIALIZED_STATE_REPLAY_INTERFACE,
    MATERIALIZED_STATE_REPLAY_SCHEMA,
    MATERIALIZED_STATE_SNAPSHOT_SCHEMA,
    DatabaseEventLog,
    DatabaseEventLogConflictError,
    DatabaseEventLogError,
    DatabaseEventLogIntegrityError,
    MaterializedRecoveryOutcome,
    MaterializedSnapshotStatus,
    MaterializedStateRecovery,
    MaterializedStateSnapshot,
    duckdb_available,
    open_database_event_log,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
EVENT_LOG_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/runtime/database_event_log.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-033.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-033.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/database_event_log.py",
    "test/api/doep/test_doep_033_add_materialized_state_replay_and_recovery.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-033.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-033.json",
)
TASK_CID = "sha256:9f276040e34160f78b651fff755ea63ae12ddc6fd97ec6a47f938e836be6307c"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}

requires_duckdb = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DatabaseEventLog hermetic tests",
)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _open(tmp_path: Path) -> DatabaseEventLog:
    return open_database_event_log(tmp_path / "events.duckdb")


def _count_reducer(
    state: Mapping[str, object], event: Mapping[str, object]
) -> dict[str, object]:
    seen = list(state.get("seen") or [])
    ordinal = int(event["n"])
    seen.append(ordinal)
    return {"seen": seen, "count": len(seen), "last": ordinal}


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_database_event_log_without_competing_subsystem() -> None:
    assert DATABASE_EVENT_LOG_INTERFACE == "DatabaseEventLog@1"
    assert EVENT_CURSOR_INTERFACE == "EventCursor@1"
    assert CONSUMER_CHECKPOINT_INTERFACE == "ConsumerCheckpoint@1"
    assert MATERIALIZED_STATE_REPLAY_BINDING == "MaterializedStateReplay@1"
    assert MATERIALIZED_STATE_REPLAY_INTERFACE == MATERIALIZED_STATE_REPLAY_BINDING
    assert MATERIALIZED_STATE_REPLAY_SCHEMA == MATERIALIZED_STATE_RECOVERY_SCHEMA
    assert MATERIALIZED_STATE_SNAPSHOT_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/materialized-state-snapshot@1"
    )
    assert MATERIALIZED_STATE_RECOVERY_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/materialized-state-recovery@1"
    )
    assert MATERIALIZED_STATE_REPLAY_CONSUMES == (
        DATABASE_EVENT_LOG_INTERFACE,
        EVENT_CURSOR_INTERFACE,
        CONSUMER_CHECKPOINT_INTERFACE,
        IDEMPOTENT_EVENT_CONSUMPTION_BINDING,
    )
    assert DatabaseEventLog.INTERFACE == DATABASE_EVENT_LOG_INTERFACE
    assert (
        DatabaseEventLog.MATERIALIZED_STATE_REPLAY_BINDING
        == MATERIALIZED_STATE_REPLAY_BINDING
    )
    assert DatabaseEventLog.recover_materialized_state.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.database_event_log"
    )
    assert DatabaseEventLog.materialize.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.database_event_log"
    )
    source = EVENT_LOG_PATH.read_text(encoding="utf-8")
    assert "class DatabaseEventLog" in source
    assert "def recover_materialized_state(" in source
    assert "def materialize(" in source
    assert "def save_materialized_snapshot(" in source
    assert "class MaterializedStateStore" not in source
    assert "class CompetingMaterializedState" not in source
    assert "class SnapshotEventBus" not in source


@requires_duckdb
def test_materialize_folds_events_and_idempotent_recovery_replays_only_suffix(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as log:
        written = [
            log.append_event("tick", {"n": index}) for index in range(1, 5)
        ]
        first = log.materialize(
            "projection:board",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
            limit=2,
        )
        assert isinstance(first, MaterializedStateRecovery)
        assert first.outcome is MaterializedRecoveryOutcome.RECOVERED
        assert first.recovered_from_snapshot is False
        assert first.state["seen"] == [1, 2]
        assert first.state["count"] == 2
        assert first.applied_event_ids == (written[0].event_id, written[1].event_id)
        assert first.cursor.position == 2
        assert first.has_more is True
        assert first.snapshot is not None
        assert first.snapshot.status is MaterializedSnapshotStatus.COMMITTED
        assert first.to_dict()["worker_assertion_is_authority"] is False
        assert first.to_dict()["binding"] == MATERIALIZED_STATE_REPLAY_BINDING
        assert first.to_dict()["carrier"] == DATABASE_EVENT_LOG_INTERFACE
        assert first.checkpoint is not None
        assert first.checkpoint.cursor.position == 2

        rest = log.recover_materialized_state(
            "projection:board",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
        )
        assert rest.recovered_from_snapshot is True
        assert rest.applied_event_ids == (written[2].event_id, written[3].event_id)
        assert rest.state["seen"] == [1, 2, 3, 4]
        assert rest.cursor.position == 4
        assert rest.has_more is False
        idle = log.recover_materialized_state(
            "projection:board", reducer=_count_reducer
        )
        assert idle.outcome is MaterializedRecoveryOutcome.IDEMPOTENT_REPLAY
        assert idle.applied_event_ids == ()
        assert idle.state["seen"] == [1, 2, 3, 4]
        assert idle.recovered_from_snapshot is True
        loaded = log.load_materialized_snapshot("projection:board")
        assert loaded is not None
        assert loaded.cursor.position == 4
        assert dict(loaded.state) == dict(idle.state)


@requires_duckdb
def test_pending_snapshot_is_ignored_and_recovery_replays_from_origin(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as log:
        written = [log.append_event("tick", {"n": index}) for index in range(1, 4)]
        fake_state = {"seen": [99], "count": 1, "last": 99}
        pending = MaterializedStateSnapshot(
            projection_id="projection:crash",
            cursor=EventCursor(
                stream_id=written[0].stream_id,
                snapshot_id=log.snapshot_id,
                position=written[0].sequence,
                last_event_id=written[0].event_id,
            ),
            state=fake_state,
            status=MaterializedSnapshotStatus.PENDING,
        )
        with log._lock:
            log._upsert_snapshot_unlocked(log._require(), pending)
        assert log.load_materialized_snapshot("projection:crash") is None
        recovered = log.recover_materialized_state(
            "projection:crash",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
        )
        assert recovered.recovered_from_snapshot is False
        assert recovered.state["seen"] == [1, 2, 3]
        assert recovered.applied_event_ids == tuple(
            event.event_id for event in written
        )
        committed = log.load_materialized_snapshot("projection:crash")
        assert committed is not None
        assert committed.status is MaterializedSnapshotStatus.COMMITTED
        assert dict(committed.state) != fake_state


@requires_duckdb
def test_snapshot_survives_reopen_and_recovery_is_stable(tmp_path: Path) -> None:
    database = tmp_path / "durable.duckdb"
    with open_database_event_log(database) as log:
        for index in range(1, 4):
            log.append_event("persist", {"n": index})
        page = log.materialize(
            "projection:durable",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
            limit=2,
        )
        assert page.cursor.position == 2
        token = page.cursor.to_token()
        decoded = EventCursor.from_token(token)
        assert decoded.position == 2
        assert log._cursors_equivalent(decoded, page.cursor)

    with open_database_event_log(database) as log:
        recovered = log.load_materialized_snapshot("projection:durable")
        assert recovered is not None
        assert recovered.cursor.position == 2
        rest = log.recover_materialized_state(
            "projection:durable", reducer=_count_reducer
        )
        assert rest.recovered_from_snapshot is True
        assert rest.state["seen"] == [1, 2, 3]
        assert rest.cursor.position == 3
        again = log.recover_materialized_state(
            "projection:durable", reducer=_count_reducer
        )
        assert again.outcome is MaterializedRecoveryOutcome.IDEMPOTENT_REPLAY
        assert dict(again.state) == dict(rest.state)


@requires_duckdb
def test_stale_snapshot_after_retention_fails_closed(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        for index in range(1, 6):
            log.append_event("tick", {"n": index})
        mid = log.materialize(
            "projection:retain",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
            limit=2,
        )
        assert mid.cursor.position == 2
        log.apply_retention(retain_recent=2)
        with pytest.raises(CursorReplayError):
            log.recover_materialized_state(
                "projection:retain", reducer=_count_reducer
            )
        fresh = log.recover_materialized_state(
            "projection:fresh",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
        )
        assert fresh.state["seen"] == [4, 5]
        assert fresh.recovered_from_snapshot is False


@requires_duckdb
def test_snapshot_cannot_rewind_or_fork_at_the_same_cursor(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        log.append_event("one", {"n": 1})
        log.append_event("two", {"n": 2})
        page = log.materialize(
            "projection:strict",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
            limit=1,
        )
        with pytest.raises(DatabaseEventLogConflictError):
            log.save_materialized_snapshot(
                "projection:strict",
                {"seen": [], "count": 0},
                log.initial_cursor(),
            )
        with pytest.raises(DatabaseEventLogConflictError):
            log.save_materialized_snapshot(
                "projection:strict",
                {"seen": [1, 99], "count": 2, "last": 99},
                page.cursor,
            )
        same = log.save_materialized_snapshot(
            "projection:strict", dict(page.state), page.cursor
        )
        assert same.snapshot_cid == page.snapshot.snapshot_cid


@requires_duckdb
def test_independent_projections_have_independent_snapshots(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        for index in range(1, 4):
            log.append_event("tick", {"n": index})
        left = log.materialize(
            "projection:left",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
            limit=10,
        )
        right = log.materialize(
            "projection:right",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
            limit=1,
        )
        assert left.state["seen"] == [1, 2, 3]
        assert right.state["seen"] == [1]
        assert left.cursor.position == 3
        assert right.cursor.position == 1
        assert log.load_materialized_snapshot("projection:left").cursor.position == 3
        assert log.load_materialized_snapshot("projection:right").cursor.position == 1
        with pytest.raises(DatabaseEventLogConflictError):
            log.recover_materialized_state(
                "projection:left",
                reducer=_count_reducer,
                stream_id="stream:other",
            )


@requires_duckdb
def test_reducer_failure_does_not_commit_a_snapshot(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        for index in range(1, 4):
            log.append_event("step", {"n": index})

        def flaky(
            state: Mapping[str, object], event: Mapping[str, object]
        ) -> dict[str, object]:
            folded = _count_reducer(state, event)
            if int(event["n"]) == 2:
                raise RuntimeError("transient reducer failure")
            return folded

        with pytest.raises(RuntimeError, match="transient reducer failure"):
            log.materialize(
                "projection:retry",
                reducer=flaky,
                initial_state={"seen": [], "count": 0},
            )
        assert log.load_materialized_snapshot("projection:retry") is None
        recovered = log.recover_materialized_state(
            "projection:retry",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
        )
        assert recovered.state["seen"] == [1, 2, 3]
        assert recovered.recovered_from_snapshot is False


@requires_duckdb
def test_worker_assertion_cannot_skip_suffix_replay(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        for index in range(1, 4):
            log.append_event("gated", {"n": index})
        first = log.materialize(
            "projection:gated",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
            limit=1,
            worker_assertion=True,
        )
        assert first.state["seen"] == [1]
        assert first.has_more is True
        assert first.to_dict()["worker_assertion_is_authority"] is False
        second = log.recover_materialized_state(
            "projection:gated",
            reducer=_count_reducer,
            worker_assertion=True,
        )
        assert second.state["seen"] == [1, 2, 3]
        assert second.applied_event_ids
        assert second.to_dict()["worker_assertion_is_authority"] is False


@requires_duckdb
def test_tampered_snapshot_digest_fails_closed(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        log.append_event("tick", {"n": 1})
        log.materialize(
            "projection:integrity",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
        )
        with log._lock:
            log._require().execute(
                """
                UPDATE materialized_state_snapshots
                SET state_json = ?, status = ?
                WHERE projection_id = ? AND status = ?
                """,
                [
                    json.dumps({"seen": [99], "count": 1, "last": 99}),
                    MaterializedSnapshotStatus.COMMITTED.value,
                    "projection:integrity",
                    MaterializedSnapshotStatus.COMMITTED.value,
                ],
            )
            log._commit_if_idle(log._require())
        with pytest.raises(DatabaseEventLogIntegrityError):
            log.load_materialized_snapshot("projection:integrity")


@requires_duckdb
def test_empty_stream_recovers_initial_state_without_a_reducer(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as log:
        recovered = log.recover_materialized_state(
            "projection:empty",
            initial_state={"seen": [], "count": 0},
        )
        assert recovered.outcome is MaterializedRecoveryOutcome.IDEMPOTENT_REPLAY
        assert recovered.state == {"seen": [], "count": 0}
        assert recovered.applied_event_ids == ()
        assert recovered.cursor.position == 0
        log.append_event("tick", {"n": 1})
        with pytest.raises(DatabaseEventLogError, match="reducer is required"):
            log.recover_materialized_state("projection:empty")


@requires_duckdb
def test_snapshot_is_not_authority_over_the_event_log(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        events = [
            log.append_event("tick", {"n": index}) for index in range(1, 3)
        ]
        log.materialize(
            "projection:hint",
            reducer=_count_reducer,
            initial_state={"seen": [], "count": 0},
        )
        snapshot = log.load_materialized_snapshot("projection:hint")
        assert snapshot is not None
        assert snapshot.to_dict()["authoritative"] is False
        physical = log.poll(log.initial_cursor(), limit=10)
        assert [item["event_id"] for item in physical.events] == [
            event.event_id for event in events
        ]


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-033"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["plan_epoch"] == 1
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["carrier"] == "DatabaseEventLog"
    assert (
        manifest["canonical_extension"]["binding"]
        == MATERIALIZED_STATE_REPLAY_BINDING
    )
    assert (
        manifest["canonical_extension"]["entrypoint"]
        == "DatabaseEventLog.recover_materialized_state"
    )
    assert manifest["canonical_extension"]["consumes"] == list(
        MATERIALIZED_STATE_REPLAY_CONSUMES
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(EVENT_LOG_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert (
        receipt["required_evidence"]["verifier_admission"]
        == "pending_independent_fenced_supervisor"
    )
    assert receipt["title"] == "Add materialized-state replay and recovery"
