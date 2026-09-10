"""Contract tests for the descriptive candidate supervisor transition table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_state_model import (
    CANONICAL_SUPERVISOR_EVENTS,
    CANONICAL_SUPERVISOR_STATES,
    CANONICAL_TERMINAL_STATES,
    REQUIRED_LEGACY_STATUSES,
    REQUIRED_STATE_TRANSITION_INVARIANTS,
    STATE_TRANSITION_TABLE_SCHEMA,
    StateTransitionTable,
    StateTransitionTableError,
    load_state_transition_table,
)


_ROOT = Path(__file__).resolve().parents[4]
_SCHEMA_PATH = _ROOT / "ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition.schema.json"
_TABLE_PATH = _ROOT / "ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition_table.json"


def _payload() -> dict[str, object]:
    return json.loads(_TABLE_PATH.read_text(encoding="utf-8"))


def _transition(payload: dict[str, object], transition_id: str) -> dict[str, object]:
    transitions = payload["transitions"]
    assert isinstance(transitions, list)
    return next(item for item in transitions if item["id"] == transition_id)


def test_closed_transition_table_covers_every_required_state_event_and_invariant() -> None:
    table = load_state_transition_table()
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$id"] == STATE_TRANSITION_TABLE_SCHEMA
    assert set(table.states) == set(CANONICAL_SUPERVISOR_STATES)
    assert set(table.events) == set(CANONICAL_SUPERVISOR_EVENTS)
    assert set(table.terminal_states) == set(CANONICAL_TERMINAL_STATES)
    assert set(table.invariants) >= set(REQUIRED_STATE_TRANSITION_INVARIANTS)
    assert {row.event for row in table.transitions} == set(table.events)
    assert {row.target_state for row in table.transitions} | {
        source for row in table.transitions for source in row.source_states
    } == set(table.states)
    assert table.authority == {
        "mode": "descriptive_only",
        "mutation_authority": "IntentRepository",
    }


def test_legacy_status_projection_is_complete_deterministic_and_read_only() -> None:
    table = load_state_transition_table()

    assert set(table.legacy_status_mappings) == set(REQUIRED_LEGACY_STATUSES)
    for legacy_status, expected_state in table.legacy_status_mappings.items():
        assert table.map_legacy_status(legacy_status) == expected_state
        assert table.map_legacy_status(f"  {legacy_status.upper()}  ") == expected_state

    assert table.map_legacy_status("unstarted") == "ready"
    assert table.map_legacy_status("skipped") == "terminal_succeeded"

    with pytest.raises(StateTransitionTableError, match="unmapped legacy status"):
        table.map_legacy_status("invented_status")


def test_forbidden_terminal_and_unknown_outcome_edges_fail_closed() -> None:
    payload = _payload()
    _transition(payload, "reconcile_unknown")["event"] = "retry"
    with pytest.raises(StateTransitionTableError, match="unknown provider outcomes cannot retry"):
        StateTransitionTable.from_dict(payload)
    payload = _payload()
    _transition(payload, "terminal_succeeded")["event"] = "merge"
    with pytest.raises(StateTransitionTableError, match="only terminalization"):
        StateTransitionTable.from_dict(payload)

    payload = _payload()
    _transition(payload, "terminal_succeeded")["source_states"] = ["terminal_failed"]
    with pytest.raises(StateTransitionTableError, match="leaves a terminal state"):
        StateTransitionTable.from_dict(payload)

    payload = _payload()
    _transition(payload, "terminal_failed_validation")["guards"].remove(
        "single_terminalization"
    )
    with pytest.raises(StateTransitionTableError, match="single-terminalization guard"):
        StateTransitionTable.from_dict(payload)

    payload = _payload()
    _transition(payload, "reconcile_unknown")["target_state"] = "retry_pending"
    with pytest.raises(StateTransitionTableError, match="cannot leave reconciliation"):
        StateTransitionTable.from_dict(payload)


def test_policy_authorization_is_bound_to_event_time_and_changes_are_nonretroactive() -> None:
    table = load_state_transition_table()
    dispatch = next(row for row in table.transitions if row.transition_id == "dispatch_claimed")
    policy_change_rows = [row for row in table.transitions if row.event == "policy_change"]

    assert "policy_authorized_at_event_time" in dispatch.guards
    assert policy_change_rows
    assert all(row.source_states == (row.target_state,) for row in policy_change_rows)
    assert all("policy_change_is_nonretroactive" in row.guards for row in policy_change_rows)

    payload = _payload()
    _transition(payload, "dispatch_claimed")["guards"].remove("policy_authorized_at_event_time")
    with pytest.raises(StateTransitionTableError, match="event-time policy guard"):
        StateTransitionTable.from_dict(payload)


def test_unknown_provider_outcome_must_reconcile_before_a_retry_can_be_resolved() -> None:
    table = load_state_transition_table()
    reconcile = next(
        row
        for row in table.transitions_for("reconciliation", "provider_outcome_unknown")
        if row.target_state == "reconciliation_pending"
    )

    assert "reconciliation_observation_required" in reconcile.guards
    assert table.transitions_for("retry", "provider_outcome_unknown") == ()
    assert (
        table.resolve(
            "reconciliation",
            "provider_outcome_unknown",
            satisfied_guards=reconcile.guards,
        )
        == reconcile
    )


def test_failed_validation_cannot_merge_to_success_and_success_requires_current_receipts() -> None:
    table = load_state_transition_table()
    success = next(row for row in table.transitions if row.target_state == "terminal_succeeded")

    assert not table.transitions_for("merge", "validation_failed")
    assert {"current_validation_evidence", "success_receipt_present", "single_terminalization"} <= set(
        success.guards
    )

    payload = _payload()
    _transition(payload, "terminal_succeeded")["guards"].remove("current_validation_evidence")
    with pytest.raises(StateTransitionTableError, match="terminal success requires current validation"):
        StateTransitionTable.from_dict(payload)

    payload = _payload()
    _transition(payload, "terminal_succeeded")["source_states"] = ["validation_failed"]
    with pytest.raises(StateTransitionTableError, match="failed validation cannot merge to success"):
        StateTransitionTable.from_dict(payload)
