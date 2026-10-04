"""Independent current-tree checks for DOEP-032 event consumption."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.database_event_log import (
    CONSUMER_CHECKPOINT_INTERFACE,
    DATABASE_EVENT_LOG_INTERFACE,
    EVENT_CONSUMPTION_INTERFACE,
    EVENT_CONSUMPTION_SCHEMA,
    EVENT_CURSOR_INTERFACE,
    DatabaseEventLog,
    DatabaseEventLogConflictError,
    DatabaseEventLogError,
    EventConsumption,
    duckdb_available,
    open_database_event_log,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "runtime"
    / "database_event_log.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-032.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-032.json"
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

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DatabaseEventLog hermetic DOEP-032 tests",
)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _open(tmp_path: Path) -> DatabaseEventLog:
    return open_database_event_log(tmp_path / "doep-032-events.duckdb")


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_interface_identities_extend_canonical_event_log() -> None:
    assert DATABASE_EVENT_LOG_INTERFACE == "DatabaseEventLog@1"
    assert EVENT_CURSOR_INTERFACE == "EventCursor@1"
    assert CONSUMER_CHECKPOINT_INTERFACE == "ConsumerCheckpoint@1"
    assert EVENT_CONSUMPTION_INTERFACE == "EventConsumption@1"
    assert EVENT_CONSUMPTION_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/event-consumption@1"
    )
    assert DatabaseEventLog.INTERFACE == DATABASE_EVENT_LOG_INTERFACE


def test_consume_is_idempotent_and_advances_consumer_cursors(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        first = log.append_event("task.started", {"ordinal": 1})
        second = log.append_event("task.finished", {"ordinal": 2})

        page = log.poll_consumer("consumer:doep-032")
        assert [item["ordinal"] for item in page.events] == [1, 2]

        consumed = log.consume_event(
            "consumer:doep-032",
            first.event_id,
            operation_id="event-consumption:doep-032:1",
        )
        assert consumed["status"] == "consumed"
        assert consumed["reason_code"] == "applied"
        assert consumed["local_durable"] is True
        assert consumed["schema"] == EVENT_CONSUMPTION_SCHEMA
        assert consumed["interface"] == EVENT_CONSUMPTION_INTERFACE
        assert consumed["sequence"] == 1
        assert consumed["cursor"]["position"] == 1
        assert consumed["cursor"]["last_event_id"] == first.event_id

        replay = log.consume_canonical_event(
            "consumer:doep-032",
            first.event_id,
            operation_id="event-consumption:doep-032:1",
        )
        assert replay["status"] == "unchanged"
        assert replay["reason_code"] == "idempotent_replay"
        assert replay["consumption_id"] == consumed["consumption_id"]
        assert replay["cursor"]["position"] == 1

        second_consumed = log.consume_event(
            "consumer:doep-032",
            second.event_id,
            operation_id="event-consumption:doep-032:2",
        )
        assert second_consumed["status"] == "consumed"
        assert second_consumed["cursor"]["position"] == 2

        checkpoint = log.load_consumer_checkpoint("consumer:doep-032")
        assert checkpoint is not None
        assert checkpoint.cursor.position == 2
        assert checkpoint.cursor.last_event_id == second.event_id

        empty = log.poll_consumer("consumer:doep-032")
        assert list(empty.events) == []
        assert empty.next_cursor.position == 2

        assert log.has_consumed("consumer:doep-032", first.event_id) is True
        loaded = log.get_event_consumption(
            "consumer:doep-032", event_id=first.event_id
        )
        assert isinstance(loaded, EventConsumption)
        assert loaded.operation_id == "event-consumption:doep-032:1"
        assert log.consumed_events("consumer:doep-032") == [
            {
                "consumption_id": consumed["consumption_id"],
                "operation_id": "event-consumption:doep-032:1",
                "event_id": first.event_id,
                "stream_id": first.stream_id,
                "sequence": 1,
                "consumed_at": consumed["consumed_at"],
                "cursor": consumed["cursor"],
            },
            {
                "consumption_id": second_consumed["consumption_id"],
                "operation_id": "event-consumption:doep-032:2",
                "event_id": second.event_id,
                "stream_id": second.stream_id,
                "sequence": 2,
                "consumed_at": second_consumed["consumed_at"],
                "cursor": second_consumed["cursor"],
            },
        ]


def test_rejected_reuse_does_not_double_consume(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        first = log.append_event("wake", {"ordinal": 1})
        second = log.append_event("wake", {"ordinal": 2})
        applied = log.consume_event(
            "consumer:doep-032",
            first.event_id,
            operation_id="event-consumption:doep-032:1",
        )

        operation_conflict = log.consume_event(
            "consumer:doep-032",
            second.event_id,
            operation_id="event-consumption:doep-032:1",
        )
        event_conflict = log.consume_event(
            "consumer:doep-032",
            first.event_id,
            operation_id="event-consumption:doep-032:2",
        )

        assert operation_conflict["status"] == "conflict"
        assert operation_conflict["reason_code"] == "operation_id_reused"
        assert operation_conflict["consumption_id"] == applied["consumption_id"]
        assert event_conflict["status"] == "conflict"
        assert event_conflict["reason_code"] == "event_id_reused"
        assert event_conflict["consumption_id"] == applied["consumption_id"]
        assert [row["event_id"] for row in log.consumed_events("consumer:doep-032")] == [
            first.event_id
        ]
        checkpoint = log.load_consumer_checkpoint("consumer:doep-032")
        assert checkpoint is not None
        assert checkpoint.cursor.position == 1


def test_at_least_once_redelivery_remains_logically_once(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        written = [
            log.append_event("runtime_wake", {"ordinal": index})
            for index in range(1, 4)
        ]
        for index, event in enumerate(written, start=1):
            result = log.consume_event(
                "consumer:worker-a",
                event.event_id,
                operation_id=f"event-consumption:doep-032:{index}",
            )
            assert result["status"] == "consumed"

        # Redelivery from the initial cursor still sees physical events, but
        # logical consumption stays exactly once under the same operation keys.
        redelivered = log.poll(log.initial_cursor(), limit=10)
        assert [item["ordinal"] for item in redelivered.events] == [1, 2, 3]
        for index, event in enumerate(written, start=1):
            replay = log.consume_event(
                "consumer:worker-a",
                event.event_id,
                operation_id=f"event-consumption:doep-032:{index}",
            )
            assert replay["status"] == "unchanged"
            assert replay["reason_code"] == "idempotent_replay"

        assert len(log.consumed_events("consumer:worker-a")) == 3
        assert log.poll_consumer("consumer:worker-a").events == ()


def test_missing_event_and_stream_mismatch_fail_closed(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        default_event = log.append_event("default", {"n": 1})
        other = log.append_event("other", {"n": 1}, stream_id="stream:other")
        log.consume_event(
            "consumer:doep-032",
            default_event.event_id,
            operation_id="event-consumption:doep-032:default",
        )
        with pytest.raises(DatabaseEventLogError):
            log.consume_event(
                "consumer:doep-032",
                "event:missing",
                operation_id="event-consumption:doep-032:missing",
            )
        with pytest.raises(DatabaseEventLogConflictError):
            log.consume_event(
                "consumer:doep-032",
                other.event_id,
                operation_id="event-consumption:doep-032:other",
            )
        with pytest.raises(DatabaseEventLogConflictError):
            log.poll_consumer("consumer:doep-032", stream_id="stream:other")


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)

    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-032"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["board_namespace"] == (
            "agent-supervisor-direct-objective-and-event-driven-planning-v1"
        )
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True

    assert manifest["title"] == "Add idempotent event consumption and cursors"
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["validation_profile"] == "doep-validation/DOEP-PLAN-V5/DOEP-032@1"
    extension = manifest["canonical_extension"]
    assert extension["identifier"] == EVENT_CONSUMPTION_INTERFACE
    assert extension["module"] == (
        "ipfs_accelerate_py.agent_supervisor.runtime.database_event_log"
    )
    assert extension["entrypoint"] == "DatabaseEventLog.consume_event"
    assert extension["carrier"] == "EventConsumption"
    assert extension["binding"] == "ConsumerCheckpoint@1"
    assert extension["idempotency_semantics"] == (
        "operation_id_and_event_id_exact_replay"
    )
    assert extension["delivery_semantics"] == "at_least_once_physical"
    assert extension["logical_semantics"] == "exactly_once_per_consumer_event"
    assert extension["competing_subsystem_created"] is False
    assert extension["completion_authority"] is False

    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {
        path: True for path in OWNER_RELATIVE_OUTPUTS
    }
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(PRIMARY_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
    assert (
        receipt["required_evidence"]["source_commit_tree_gitlinks"]
        == BASE_REPOSITORIES
    )
    assert receipt["required_evidence"]["test_proof_results"]["validation_command"] == [
        "python3",
        "-m",
        "pytest",
        "test/api/doep/test_doep_032_add_idempotent_event_consumption_and_cursors.py",
        "-q",
    ]
    assert receipt["candidate_status"] == "implemented"
    assert receipt["supervisor_acceptance"]["completion_authoritative"] is False
    assert receipt["supervisor_acceptance"]["state"] == (
        "pending_independent_fenced_supervisor"
    )
