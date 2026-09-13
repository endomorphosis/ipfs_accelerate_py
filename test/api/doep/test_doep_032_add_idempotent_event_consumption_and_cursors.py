"""Independent current-tree checks for DOEP-032 idempotent consumption."""

from __future__ import annotations

import hashlib
import json
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
    IDEMPOTENT_EVENT_CONSUMPTION_CONSUMES,
    IDEMPOTENT_EVENT_CONSUMPTION_INTERFACE,
    IDEMPOTENT_EVENT_CONSUMPTION_SCHEMA,
    ConsumptionOutcome,
    ConsumptionStatus,
    ConsumerCheckpoint,
    DatabaseEventLog,
    DatabaseEventLogConflictError,
    EventConsumptionPage,
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
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-032.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-032.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/database_event_log.py",
    "test/api/doep/test_doep_032_add_idempotent_event_consumption_and_cursors.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-032.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-032.json",
)
TASK_CID = "sha256:1a35178e898e95d5bb71fc6549d7c3730a8a436390ddfe668e51abf26a0b71e6"
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


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_database_event_log_without_competing_subsystem() -> None:
    assert DATABASE_EVENT_LOG_INTERFACE == "DatabaseEventLog@1"
    assert EVENT_CURSOR_INTERFACE == "EventCursor@1"
    assert CONSUMER_CHECKPOINT_INTERFACE == "ConsumerCheckpoint@1"
    assert IDEMPOTENT_EVENT_CONSUMPTION_BINDING == "IdempotentEventConsumption@1"
    assert IDEMPOTENT_EVENT_CONSUMPTION_INTERFACE == IDEMPOTENT_EVENT_CONSUMPTION_BINDING
    assert IDEMPOTENT_EVENT_CONSUMPTION_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/idempotent-event-consumption@1"
    )
    assert IDEMPOTENT_EVENT_CONSUMPTION_CONSUMES == (
        DATABASE_EVENT_LOG_INTERFACE,
        EVENT_CURSOR_INTERFACE,
        CONSUMER_CHECKPOINT_INTERFACE,
    )
    assert DatabaseEventLog.INTERFACE == DATABASE_EVENT_LOG_INTERFACE
    assert (
        DatabaseEventLog.IDEMPOTENT_EVENT_CONSUMPTION_BINDING
        == IDEMPOTENT_EVENT_CONSUMPTION_BINDING
    )
    assert DatabaseEventLog.consume.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.database_event_log"
    )
    source = EVENT_LOG_PATH.read_text(encoding="utf-8")
    assert "class DatabaseEventLog" in source
    assert "def consume(" in source
    assert "class CompetingEventConsumption" not in source
    assert "class IdempotentEventBus" not in source


@requires_duckdb
def test_consume_applies_once_and_replays_are_idempotent(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        written = [
            log.append_event("runtime_wake", {"ordinal": index})
            for index in range(1, 5)
        ]
        seen: list[int] = []

        def handler(event: dict[str, object]) -> None:
            seen.append(int(event["ordinal"]))

        first = log.consume("consumer:worker-a", limit=2, handler=handler)
        assert isinstance(first, EventConsumptionPage)
        assert first.applied_event_ids == (written[0].event_id, written[1].event_id)
        assert first.replayed_event_ids == ()
        assert seen == [1, 2]
        assert first.next_cursor.position == 2
        assert first.has_more is True
        assert first.to_dict()["worker_assertion_is_authority"] is False
        assert first.to_dict()["binding"] == IDEMPOTENT_EVENT_CONSUMPTION_BINDING
        assert first.to_dict()["carrier"] == DATABASE_EVENT_LOG_INTERFACE
        assert isinstance(first.checkpoint, ConsumerCheckpoint)
        assert log.event_consumed("consumer:worker-a", written[0].event_id)
        assert not log.event_consumed("consumer:worker-a", written[2].event_id)

        rest = log.consume(
            "consumer:worker-a",
            limit=10,
            handler=handler,
            cursor=first.next_cursor,
        )
        assert seen == [1, 2, 3, 4]
        assert rest.applied_event_ids == (written[2].event_id, written[3].event_id)
        assert rest.has_more is False
        empty = log.consume("consumer:worker-a", limit=10, handler=handler)
        assert empty.records == ()
        assert empty.next_cursor.position == 4
        assert seen == [1, 2, 3, 4]
        assert log.consumed_event_ids("consumer:worker-a") == tuple(
            event.event_id for event in written
        )
        # Physical poll remains at-least-once; logical consume does not re-apply.
        physical = log.poll(log.initial_cursor(), limit=10)
        assert [item["ordinal"] for item in physical.events] == [1, 2, 3, 4]


@requires_duckdb
def test_redelivery_of_applied_identity_is_idempotent_replay(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        written = [
            log.append_event("step", {"n": index}) for index in range(1, 4)
        ]
        log.consume("consumer:crash-window", limit=1)
        # Applied row committed, checkpoint still at event 1: crash between
        # consume-record and cursor persist.
        with log._lock:
            log._upsert_consumed_unlocked(
                log._require(),
                consumer_id="consumer:crash-window",
                event_id=written[1].event_id,
                stream_id=written[1].stream_id,
                sequence=written[1].sequence,
                status=ConsumptionStatus.APPLIED,
                applied_at=written[1].recorded_at,
            )
        seen: list[int] = []
        page = log.consume(
            "consumer:crash-window",
            limit=10,
            handler=lambda event: seen.append(int(event["n"])),
        )
        assert page.replayed_event_ids == (written[1].event_id,)
        assert page.applied_event_ids == (written[2].event_id,)
        assert seen == [3]
        assert all(
            record.outcome is ConsumptionOutcome.IDEMPOTENT_REPLAY
            for record in page.records
            if record.event_id == written[1].event_id
        )
        assert log.consumer_cursor("consumer:crash-window").position == 3


@requires_duckdb
def test_independent_consumers_have_independent_cursors(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        events = [
            log.append_event("tick", {"n": index}) for index in range(1, 4)
        ]
        left_seen: list[int] = []
        right_seen: list[int] = []
        left = log.consume(
            "consumer:left",
            limit=10,
            handler=lambda event: left_seen.append(int(event["n"])),
        )
        right = log.consume(
            "consumer:right",
            limit=1,
            handler=lambda event: right_seen.append(int(event["n"])),
        )
        assert left_seen == [1, 2, 3]
        assert right_seen == [1]
        assert left.next_cursor.position == 3
        assert right.next_cursor.position == 1
        assert log.consumer_cursor("consumer:left").last_event_id == events[-1].event_id
        assert log.consumer_cursor("consumer:right").last_event_id == events[0].event_id
        with pytest.raises(DatabaseEventLogConflictError):
            log.consume("consumer:left", stream_id="stream:other")


@requires_duckdb
def test_handler_failure_retries_pending_without_skipping(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        for index in range(1, 4):
            log.append_event("step", {"n": index})
        seen: list[int] = []

        def flaky(event: dict[str, object]) -> None:
            ordinal = int(event["n"])
            seen.append(ordinal)
            if ordinal == 2 and seen.count(2) == 1:
                raise RuntimeError("transient handler failure")

        with pytest.raises(RuntimeError, match="transient handler failure"):
            log.consume("consumer:retry", limit=10, handler=flaky)
        assert seen == [1, 2]
        assert log.consumed_event_ids("consumer:retry") == (
            log.poll(log.initial_cursor(), limit=1).events[0]["event_id"],
        )
        checkpoint = log.load_consumer_checkpoint("consumer:retry")
        assert checkpoint is not None
        assert checkpoint.cursor.position == 1

        recovered = log.consume("consumer:retry", limit=10, handler=flaky)
        assert seen == [1, 2, 2, 3]
        assert recovered.replayed_event_ids == ()
        assert len(recovered.applied_event_ids) == 2
        assert log.consumer_cursor("consumer:retry").position == 3


@requires_duckdb
def test_duplicate_append_does_not_double_consume(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        event = log.append_event("wake", {"ordinal": 1})
        again = log.append_event(
            "wake",
            {"ordinal": 1},
            event_id=event.event_id,
            recorded_at=event.recorded_at,
        )
        assert again.event_id == event.event_id
        seen: list[str] = []
        page = log.consume(
            "consumer:coalesce",
            limit=10,
            handler=lambda item: seen.append(str(item["event_id"])),
        )
        assert seen == [event.event_id]
        assert page.applied_event_ids == (event.event_id,)
        replay = log.consume(
            "consumer:coalesce",
            limit=10,
            handler=lambda item: seen.append(str(item["event_id"])),
        )
        assert seen == [event.event_id]
        assert replay.records == ()


@requires_duckdb
def test_checkpoint_survives_reopen_and_cursor_tokens_round_trip(
    tmp_path: Path,
) -> None:
    database = tmp_path / "durable.duckdb"
    with open_database_event_log(database) as log:
        for index in range(1, 4):
            log.append_event("persist", {"n": index})
        page = log.consume("consumer:durable", limit=2)
        token = page.next_cursor.to_token()
        decoded = EventCursor.from_token(token)
        assert decoded.position == 2
        assert log._cursors_equivalent(decoded, page.next_cursor)

    with open_database_event_log(database) as log:
        recovered = log.load_consumer_checkpoint("consumer:durable")
        assert recovered is not None
        assert recovered.cursor.position == 2
        rest = log.consume(
            "consumer:durable", limit=10, cursor=recovered.cursor
        )
        assert [item["n"] for item in log.poll(log.initial_cursor(), limit=10).events] == [
            1,
            2,
            3,
        ]
        assert rest.applied_event_ids
        assert rest.next_cursor.position == 3


@requires_duckdb
def test_stale_cursor_after_retention_fails_closed(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        for index in range(1, 6):
            log.append_event("tick", {"n": index})
        mid = log.consume("consumer:retain", limit=2)
        assert mid.next_cursor.position == 2
        log.apply_retention(retain_recent=2)
        with pytest.raises(CursorReplayError):
            log.consume("consumer:retain", limit=10)
        fresh = log.consume("consumer:fresh", limit=10)
        assert [record.sequence for record in fresh.records] == [4, 5]


@requires_duckdb
def test_supplied_cursor_must_match_durable_checkpoint(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        log.append_event("one", {"v": 1})
        log.append_event("two", {"v": 2})
        page = log.consume("consumer:strict", limit=1)
        foreign = log.initial_cursor()
        with pytest.raises(DatabaseEventLogConflictError):
            log.consume("consumer:strict", cursor=foreign)
        matched = log.consume("consumer:strict", cursor=page.next_cursor, limit=10)
        assert matched.applied_event_ids
        with pytest.raises(DatabaseEventLogConflictError):
            log.save_consumer_checkpoint("consumer:strict", log.initial_cursor())


@requires_duckdb
def test_worker_assertion_cannot_bypass_consumption_records(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        event = log.append_event("gated", {"n": 1})
        seen: list[int] = []
        first = log.consume(
            "consumer:gated",
            handler=lambda item: seen.append(int(item["n"])),
            worker_assertion=True,
        )
        assert seen == [1]
        assert first.to_dict()["worker_assertion_is_authority"] is False
        second = log.consume(
            "consumer:gated",
            handler=lambda item: seen.append(int(item["n"])),
            worker_assertion=True,
        )
        assert seen == [1]
        assert log.event_consumed("consumer:gated", event.event_id)
        assert second.records == ()
        assert seen == [1]


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-032"
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
        == IDEMPOTENT_EVENT_CONSUMPTION_BINDING
    )
    assert manifest["canonical_extension"]["entrypoint"] == "DatabaseEventLog.consume"
    assert manifest["canonical_extension"]["consumes"] == list(
        IDEMPOTENT_EVENT_CONSUMPTION_CONSUMES
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
    assert receipt["title"] == "Add idempotent event consumption and cursors"
