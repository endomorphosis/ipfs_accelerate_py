"""Independent current-tree checks for DOEP-034 coalescing and causal order."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    CAUSAL_ORDERING_BINDING,
    EVENT_COALESCING_AND_CAUSAL_ORDERING_BINDING,
    EVENT_COALESCING_AND_CAUSAL_ORDERING_CONSUMES,
    EVENT_COALESCING_AND_CAUSAL_ORDERING_INTERFACE,
    EVENT_COALESCING_BINDING,
    EVENT_CURSOR_INTERFACE,
    EVENT_LOG_INTERFACE,
    JSONL_EVENT_LOG_INTERFACE,
    CausalOrderError,
    CoalescedEventPage,
    EventCoalescingDecision,
    EventCoalescingMode,
    append_jsonl_event,
    assert_causal_order,
    event_coalescing_forbidden,
    happens_before,
    initial_event_cursor,
    plan_event_coalescing,
    read_coalesced_event_page,
    read_jsonl_event_page,
    read_jsonl_events,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
EVENT_LOG_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/event_log.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-034.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-034.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/event_log.py",
    "test/api/doep/test_doep_034_add_event_coalescing_and_causal_ordering.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-034.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-034.json",
)
TASK_CID = "sha256:d57bb00c7e2d3013e213bc94c1796515f77b49038e5337cec0d015918515a831"
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


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_event_log_without_competing_subsystem() -> None:
    assert EVENT_LOG_INTERFACE == "EventLog@1"
    assert JSONL_EVENT_LOG_INTERFACE == EVENT_LOG_INTERFACE
    assert EVENT_CURSOR_INTERFACE == "EventCursor@1"
    assert EVENT_COALESCING_BINDING == "EventCoalescing@1"
    assert CAUSAL_ORDERING_BINDING == "CausalOrdering@1"
    assert EVENT_COALESCING_AND_CAUSAL_ORDERING_BINDING == (
        "EventCoalescingAndCausalOrdering@1"
    )
    assert (
        EVENT_COALESCING_AND_CAUSAL_ORDERING_INTERFACE
        == EVENT_COALESCING_AND_CAUSAL_ORDERING_BINDING
    )
    assert EVENT_COALESCING_AND_CAUSAL_ORDERING_CONSUMES == (
        EVENT_LOG_INTERFACE,
        EVENT_CURSOR_INTERFACE,
    )
    assert append_jsonl_event.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.event_log"
    )
    assert read_coalesced_event_page.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.event_log"
    )
    assert plan_event_coalescing.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.event_log"
    )
    assert assert_causal_order.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.event_log"
    )
    source = EVENT_LOG_PATH.read_text(encoding="utf-8")
    assert "def append_jsonl_event(" in source
    assert "def read_coalesced_event_page(" in source
    assert "def plan_event_coalescing(" in source
    assert "def assert_causal_order(" in source
    assert "class CompetingEventLog" not in source
    assert "class CausalEventBus" not in source
    assert "class CoalescingEventStore" not in source
    assert "class CompetingCausalLog" not in source


def test_same_key_latest_generation_coalesces_without_rewriting_the_log(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    first = append_jsonl_event(
        path, "runtime_wake", {"generation": 1}, coalescing_key="wake:board"
    )
    second = append_jsonl_event(
        path, "runtime_wake", {"generation": 2}, coalescing_key="wake:board"
    )
    third = append_jsonl_event(
        path, "runtime_wake", {"generation": 3}, coalescing_key="wake:board"
    )
    other = append_jsonl_event(
        path, "runtime_wake", {"generation": 1}, coalescing_key="wake:other"
    )
    page = read_coalesced_event_page(path, initial_event_cursor(path), limit=10)
    assert isinstance(page, CoalescedEventPage)
    assert [event["event_id"] for event in page.events] == [
        third["event_id"],
        other["event_id"],
    ]
    wake_board = page.decisions[0]
    assert isinstance(wake_board, EventCoalescingDecision)
    assert wake_board.mode is EventCoalescingMode.LATEST_GENERATION
    assert wake_board.input_event_ids == (
        first["event_id"],
        second["event_id"],
        third["event_id"],
    )
    assert wake_board.representative_event_id == third["event_id"]
    assert page.consumed_event_ids == (
        first["event_id"],
        second["event_id"],
        third["event_id"],
        other["event_id"],
    )
    assert page.next_cursor.position == 4
    assert page.has_more is False
    assert page.to_dict()["worker_assertion_is_authority"] is False
    assert page.to_dict()["authoritative"] is False
    assert page.to_dict()["binding"] == EVENT_COALESCING_AND_CAUSAL_ORDERING_BINDING
    assert page.to_dict()["carrier"] == EVENT_LOG_INTERFACE
    physical = read_jsonl_event_page(path, initial_event_cursor(path), limit=10)
    assert [event["event_id"] for event in physical.events] == [
        first["event_id"],
        second["event_id"],
        third["event_id"],
        other["event_id"],
    ]
    assert [event["generation"] for event in read_jsonl_events(path)] == [1, 2, 3, 1]


def test_head_identity_coalesces_without_advancing_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = append_jsonl_event(
        path, "provider_capacity", {"level": 4}, coalescing_key="gpu:0"
    )
    again = append_jsonl_event(
        path, "provider_capacity", {"level": 4}, coalescing_key="gpu:0"
    )
    assert again["event_id"] == first["event_id"]
    assert again["sequence"] == 1
    newer = append_jsonl_event(
        path, "provider_capacity", {"level": 7}, coalescing_key="gpu:0"
    )
    assert newer["event_id"] != first["event_id"]
    assert newer["sequence"] == 2
    assert len(read_jsonl_events(path)) == 2
    page = read_coalesced_event_page(path, initial_event_cursor(path), limit=10)
    assert [event["event_id"] for event in page.events] == [newer["event_id"]]
    assert page.decisions[0].mode is EventCoalescingMode.LATEST_GENERATION
    assert page.consumed_event_ids == (first["event_id"], newer["event_id"])


def test_safety_events_are_never_coalesced(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = append_jsonl_event(
        path, "lease_expiring", {"lease_id": "lease:1"}, coalescing_key="lease:1"
    )
    second = append_jsonl_event(
        path, "lease_expiring", {"lease_id": "lease:1"}, coalescing_key="lease:1"
    )
    proof = append_jsonl_event(
        path, "proof_completed", {"proof_id": "proof:1"}, coalescing_key="proof:1"
    )
    receipt = append_jsonl_event(
        path,
        "refill_scan_receipt",
        {"artifact": "receipt:1"},
        coalescing_key="receipt:1",
    )
    assert event_coalescing_forbidden(first) is True
    assert event_coalescing_forbidden("decision_runtime_semantic_change") is True
    assert first["coalescing_forbidden"] is True
    assert first["coalescing_key"] == ""
    again = append_jsonl_event(
        path, "lease_expiring", {"lease_id": "lease:1"}, coalescing_key="lease:1"
    )
    assert again["event_id"] != first["event_id"]
    page = read_coalesced_event_page(path, initial_event_cursor(path), limit=10)
    assert [event["event_id"] for event in page.events] == [
        first["event_id"],
        second["event_id"],
        proof["event_id"],
        receipt["event_id"],
        again["event_id"],
    ]
    assert all(decision.mode is EventCoalescingMode.NONE for decision in page.decisions)
    assert all(len(decision.input_event_ids) == 1 for decision in page.decisions)


def test_causal_parents_must_exist_and_order_the_linear_extension(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    root = append_jsonl_event(path, "task.created", {"task_id": "T-1"})
    child = append_jsonl_event(
        path,
        "task.queued",
        {"task_id": "T-1"},
        causal_parent_ids=(root["event_id"],),
    )
    grandchild = append_jsonl_event(
        path,
        "task.started",
        {"task_id": "T-1"},
        causal_parent_ids=(child["event_id"], root["event_id"]),
    )
    sibling = append_jsonl_event(
        path,
        "task.note",
        {"task_id": "T-1"},
        causal_parent_ids=(root["event_id"],),
    )
    assert child["causal_parent_ids"] == [root["event_id"]]
    assert grandchild["previous_event_id"] == child["event_id"]
    assert happens_before(root, grandchild, (root, child, grandchild, sibling))
    assert happens_before(child, grandchild, (root, child, grandchild, sibling))
    assert not happens_before(sibling, grandchild, (root, child, grandchild, sibling))
    assert not happens_before(grandchild, root, (root, child, grandchild, sibling))
    page = read_coalesced_event_page(path, initial_event_cursor(path), limit=10)
    assert [event["event_id"] for event in page.events] == [
        root["event_id"],
        child["event_id"],
        grandchild["event_id"],
        sibling["event_id"],
    ]
    with pytest.raises(CausalOrderError, match="missing from the event log"):
        append_jsonl_event(
            path,
            "task.orphaned",
            {"task_id": "T-1"},
            causal_parent_ids=("sha256:" + "a" * 64,),
        )


def test_reordered_or_cyclic_causal_parents_fail_closed() -> None:
    first = {
        "type": "tick",
        "event_id": "sha256:one",
        "sequence": 1,
        "previous_event_id": "",
        "causal_parent_ids": [],
    }
    second = {
        "type": "tick",
        "event_id": "sha256:two",
        "sequence": 2,
        "previous_event_id": "sha256:one",
        "causal_parent_ids": ["sha256:one"],
    }
    assert_causal_order((first, second))
    with pytest.raises(
        CausalOrderError, match="physical predecessor|missing or reordered"
    ):
        assert_causal_order((second, first))
    later_parent = {
        "type": "tick",
        "event_id": "sha256:two",
        "sequence": 2,
        "previous_event_id": "sha256:one",
        "causal_parent_ids": ["sha256:three"],
    }
    trailing = {
        "type": "tick",
        "event_id": "sha256:three",
        "sequence": 3,
        "previous_event_id": "sha256:two",
        "causal_parent_ids": ["sha256:two"],
    }
    with pytest.raises(CausalOrderError, match="missing or reordered"):
        assert_causal_order((first, later_parent, trailing))
    cyclic = {
        "type": "tick",
        "event_id": "sha256:two",
        "sequence": 2,
        "previous_event_id": "sha256:one",
        "causal_parent_ids": ["sha256:two"],
    }
    with pytest.raises(CausalOrderError, match="cannot list itself"):
        assert_causal_order((first, cyclic))
    broken_chain = {
        "type": "tick",
        "event_id": "sha256:two",
        "sequence": 2,
        "previous_event_id": "sha256:missing",
        "causal_parent_ids": ["sha256:one"],
    }
    with pytest.raises(CausalOrderError, match="physical predecessor"):
        assert_causal_order((first, broken_chain))


def test_coalesced_cursor_advances_by_physical_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    written = [
        append_jsonl_event(
            path, "runtime_wake", {"n": index}, coalescing_key="wake:shared"
        )
        for index in range(1, 5)
    ]
    first = read_coalesced_event_page(path, initial_event_cursor(path), limit=2)
    assert [event["event_id"] for event in first.events] == [written[1]["event_id"]]
    assert first.consumed_event_ids == (written[0]["event_id"], written[1]["event_id"])
    assert first.next_cursor.position == 2
    assert first.has_more is True
    rest = read_coalesced_event_page(path, first.next_cursor, limit=10)
    assert rest.consumed_event_ids == (written[2]["event_id"], written[3]["event_id"])
    assert rest.next_cursor.position == 4
    assert rest.has_more is False
    idle = read_coalesced_event_page(path, rest.next_cursor, limit=10)
    assert idle.events == ()
    assert idle.consumed_event_ids == ()
    assert idle.next_cursor.position == 4


def test_worker_assertion_cannot_skip_causal_or_coalescing_checks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    first = append_jsonl_event(
        path, "runtime_wake", {"n": 1}, coalescing_key="wake:gated"
    )
    second = append_jsonl_event(
        path, "runtime_wake", {"n": 2}, coalescing_key="wake:gated"
    )
    page = read_coalesced_event_page(
        path,
        initial_event_cursor(path),
        limit=10,
        worker_assertion=True,
    )
    assert [event["event_id"] for event in page.events] == [second["event_id"]]
    assert page.consumed_event_ids == (first["event_id"], second["event_id"])
    assert page.to_dict()["worker_assertion_is_authority"] is False
    payload = page.decisions[0].to_dict()
    assert payload["worker_assertion_is_authority"] is False
    assert payload["binding"] == EVENT_COALESCING_BINDING


def test_default_causal_parent_is_the_physical_predecessor(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = append_jsonl_event(path, "tick", {"n": 1})
    second = append_jsonl_event(path, "tick", {"n": 2})
    assert first["causal_parent_ids"] == []
    assert second["causal_parent_ids"] == [first["event_id"]]
    assert second["previous_event_id"] == first["event_id"]
    page = read_coalesced_event_page(path, initial_event_cursor(path), limit=10)
    assert happens_before(first, second, page.events)


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-034"
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
    assert manifest["canonical_extension"]["carrier"] == "EventLog"
    assert (
        manifest["canonical_extension"]["binding"]
        == EVENT_COALESCING_AND_CAUSAL_ORDERING_BINDING
    )
    assert manifest["canonical_extension"]["entrypoint"] == "read_coalesced_event_page"
    assert manifest["canonical_extension"]["consumes"] == list(
        EVENT_COALESCING_AND_CAUSAL_ORDERING_CONSUMES
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
    assert receipt["title"] == "Add event coalescing and causal ordering"
