"""Recorded source selection plus isolated native callback replay checks.

The fixture is an excerpt, not a complete causal chain or task authority.
Git artifact readers are substituted here; the merge-train tests exercise
the same callback with real isolated Git commits and event files.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_completion_events import (
    retained_synchronous_reconciliation,
)


def captured():
    return json.loads(
        (
            Path(__file__).parent / "fixtures/synchronous_append_quarantine.json"
        ).read_text()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "",
        "duplicate_finish",
        "duplicate_source",
        "duplicate_receipt",
        "foreign_attempt",
        "foreign_key",
        "missing_key",
        "foreign_stream",
        "foreign_namespace",
        "foreign_source_identity",
        "foreign_request",
        "conflicting_request",
        "wrong_source",
        "out_of_order",
        "failed_finish",
        "consumed_unknown",
        "provider_unknown",
        "not_queued",
        "wrong_reason",
    ],
)
def test_recorded_synchronous_handoff_is_unambiguous(mutation):
    fixture = captured()
    events = fixture["events"]
    enqueue, source, receipt, finish = events
    if mutation == "duplicate_finish":
        events.append(copy.deepcopy(finish))
    elif mutation == "duplicate_source":
        events.insert(1, copy.deepcopy(source))
    elif mutation == "duplicate_receipt":
        events.insert(2, copy.deepcopy(receipt))
    elif mutation == "foreign_attempt":
        finish["attempt"] += 1
    elif mutation == "foreign_key":
        finish["canonical_task_key"] = "foreign"
    elif mutation == "missing_key":
        for event in (source, receipt, finish):
            event.pop("canonical_task_key")
    elif mutation == "foreign_stream":
        finish["stream_id"] = "foreign"
    elif mutation == "foreign_namespace":
        finish["board_namespace"] = "foreign"
    elif mutation == "foreign_source_identity":
        finish["task_source_identity"] = {"foreign": True}
    elif mutation == "foreign_request":
        finish["merge_result"]["request_id"] = "foreign"
    elif mutation == "conflicting_request":
        finish["request_id"] = "foreign"
    elif mutation == "wrong_source":
        receipt["completion_source_event_id"] = enqueue["event_id"]
    elif mutation == "out_of_order":
        events[:] = [enqueue, source, finish, receipt]
    elif mutation == "failed_finish":
        finish["returncode"] = 1
    elif mutation == "consumed_unknown":
        finish["attempt_consumed"] = "unknown"
    elif mutation == "provider_unknown":
        finish["provider_dispatched"] = "unknown"
    elif mutation == "not_queued":
        finish["merge_result"]["queued"] = False
    elif mutation == "wrong_reason":
        finish["merge_result"]["reason"] = "unverified"
    result = retained_synchronous_reconciliation(
        events,
        request_id=fixture["request"]["request_id"],
        queued_confirmation=finish,
    )
    assert (result is not None) is (not mutation)
    if result is not None:
        assert result == (source, receipt)


@pytest.mark.parametrize(
    "mutation",
    [
        "",
        "wrong_proof",
        "wrong_output",
        "wrong_receipt_id",
        "duplicate_receipt",
        "duplicate_finish",
        "foreign_attempt",
        "wrong_projection",
        "missing_key",
    ],
)
def test_callback_replays_recorded_source_without_rewriting_events(mutation):
    fixture = captured()
    request = SimpleNamespace(**fixture["request"])
    events = fixture["events"]
    _enqueue, source, receipt, finish = events
    proof = copy.deepcopy(receipt["integration_commit_proof"])
    invariant = copy.deepcopy(receipt["post_merge_declared_output_invariant"])
    members = copy.deepcopy(
        receipt["completion_receipt_evidence"]["completion_receipts"]
    )
    if mutation == "wrong_proof":
        receipt["integration_commit_proof"]["integration_commit"] = "a" * 40
    elif mutation == "wrong_output":
        receipt["post_merge_declared_output_invariant"]["checks"][0][
            "repository_ref"
        ] = ("a" * 40)
    elif mutation == "wrong_receipt_id":
        receipt["completion_receipt_evidence"]["receipt_id"] = "foreign"
    elif mutation == "duplicate_receipt":
        events.insert(2, copy.deepcopy(receipt))
    elif mutation == "duplicate_finish":
        events.append(copy.deepcopy(finish))
    elif mutation == "foreign_attempt":
        finish["attempt"] += 1
    elif mutation == "wrong_projection":
        source["merge_queue_synchronous_source"]["source_projection_id"] = "foreign"
    elif mutation == "missing_key":
        receipt.pop("canonical_task_key")
    before = copy.deepcopy(events)
    daemon = object.__new__(PortalImplementationDaemon)
    daemon._main_branch_name = lambda: request.metadata["target_branch"]
    daemon._iter_merge_lifecycle_events = lambda: events
    daemon._record_event = lambda *args: pytest.fail(
        "replay attempted to append an event"
    )
    daemon._task_source_identity_record = lambda: None
    daemon._task_identity_by_display_id = {}
    daemon._completion_tasks_for_declared_output_gate = lambda *args: ([], None)
    daemon._immutable_integration_commit = lambda *args, **kwargs: proof
    daemon._declared_output_tracking_invariant = lambda *args, **kwargs: invariant
    result = daemon._record_merge_queue_callback_reconciliation(
        request=request,
        task=SimpleNamespace(task_id=request.task_id),
        metadata=request.metadata,
        implementation_commit=request.commit_sha,
        integration_commit=proof["integration_commit"],
        integration_commit_proof=proof,
        completion_task_cids=request.metadata["completion_task_cids"],
        completion_receipts=members,
        declared_output_invariant=invariant,
    )
    assert result["recorded"] is (not mutation), result
    assert events == before
    if not mutation:
        assert result == {
            "recorded": True,
            "replayed": True,
            "event_id": receipt["event_id"],
        }
