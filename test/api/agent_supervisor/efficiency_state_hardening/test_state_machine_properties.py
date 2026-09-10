"""Bounded executable qualification for the descriptive transition contract.

The production transition table is deliberately declarative.  This corpus
therefore checks its closed vocabulary and semantic guard profile without
pretending that a bounded exploration is an unbounded proof.  Each negative
seed changes one fact only and carries a one-step counterexample.
"""

from __future__ import annotations

import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Callable

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_state_model import (
    CANONICAL_SUPERVISOR_EVENTS,
    CANONICAL_SUPERVISOR_STATES,
    REQUIRED_STATE_TRANSITION_INVARIANTS,
    StateTransition,
    StateTransitionTable,
    StateTransitionTableError,
    load_state_transition_table,
)


_ROOT = Path(__file__).resolve().parents[4]
_TABLE_PATH = _ROOT / "ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition_table.json"
_QUALIFICATION_PATH = (
    _ROOT
    / "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory"
    / "state_machine_qualification.json"
)
_EFFECTFUL_ROWS = frozenset(
    {
        "dispatch_claimed",
        "provider_start_dispatch",
        "provider_start_running",
        "provider_complete",
        "provider_timeout",
        "provider_connection_loss",
        "merge_start",
    }
)
_FENCED_ROWS = frozenset(
    {
        "claim_ready",
        "dispatch_claimed",
        "fence_takeover_reconciled",
        "lease_expiry_active",
        "merge_conflict",
        "merge_pending",
        "merge_start",
        "provider_complete",
        "provider_start_dispatch",
        "provider_start_running",
        "rescue_validation_failure",
        "retry_rescue",
        "validation_failed",
        "validation_start",
        "validation_succeeded",
    }
)
_EFFECT_PRESERVING_ROWS = frozenset(
    {
        "fence_takeover_reconciled",
        "lease_expiry_active",
        "merge_conflict",
        "owner_loss_active",
        "provider_connection_loss",
        "provider_timeout",
        "reconcile_observed_completion",
        "reconcile_retry",
        "reconcile_unknown",
        "rescue_validation_failure",
        "retry_rescue",
        "terminal_compensated",
        "terminal_quarantined",
    }
)


class BoundedInvariantViolation(AssertionError):
    """A one-transition witness produced by the bounded semantic checker."""

    def __init__(self, invariant: str, transition_id: str, detail: str) -> None:
        self.invariant = invariant
        self.transition_id = transition_id
        self.trace = (transition_id,)
        super().__init__(f"{invariant}: {transition_id}: {detail}")


def _payload() -> dict[str, Any]:
    return json.loads(_TABLE_PATH.read_text(encoding="utf-8"))


def _rows(table: StateTransitionTable) -> dict[str, StateTransition]:
    return {row.transition_id: row for row in table.transitions}


def _require_guard(
    rows: dict[str, StateTransition], invariant: str, transition_id: str, guard: str
) -> None:
    row = rows.get(transition_id)
    if row is None:
        raise BoundedInvariantViolation(invariant, transition_id, "required row is absent")
    if guard not in row.guards:
        raise BoundedInvariantViolation(invariant, transition_id, f"missing {guard}")


def _require_rows(
    rows: dict[str, StateTransition], invariant: str, identifiers: frozenset[str], guard: str
) -> None:
    for transition_id in sorted(identifiers):
        _require_guard(rows, invariant, transition_id, guard)


def _check_semantic_invariants(table: StateTransitionTable) -> None:
    """Evaluate the finite guard vocabulary and emit minimal witnesses."""

    rows = _rows(table)
    _require_guard(rows, "single_authoritative_owner", "claim_ready", "single_authoritative_owner")
    _require_rows(rows, "revision_cas", frozenset(rows), "revision_cas_matches")
    _require_rows(rows, "event_materialized_state_reconciliation", frozenset(rows), "transition_service_only")
    _require_rows(rows, "idempotency_key_for_effects", _EFFECTFUL_ROWS, "idempotency_key_bound")
    _require_rows(rows, "fresh_lease_and_fence", _FENCED_ROWS, "fresh_lease_and_fence")
    _require_rows(rows, "external_effect_preservation", _EFFECT_PRESERVING_ROWS, "external_effect_preserved")
    _require_guard(
        rows, "deterministic_owner_restart", "owner_restart_reconciles", "durable_state_reconstructed"
    )

    unknown_rows = [
        row for row in rows.values() if "provider_outcome_unknown" in row.source_states
    ]
    if len(unknown_rows) != 1:
        raise BoundedInvariantViolation(
            "unknown_outcome_reconciliation", "provider_outcome_unknown", "unknown outcome has non-unique exit"
        )
    unknown = unknown_rows[0]
    if unknown.event != "reconciliation" or unknown.target_state != "reconciliation_pending":
        raise BoundedInvariantViolation(
            "unknown_outcome_reconciliation", unknown.transition_id, "unknown outcome bypasses reconciliation"
        )
    _require_guard(
        rows, "unknown_outcome_reconciliation", unknown.transition_id, "reconciliation_observation_required"
    )

    for row in rows.values():
        if row.target_state in table.terminal_states:
            _require_guard(rows, "single_terminalization", row.transition_id, "single_terminalization")
    _require_guard(rows, "current_validation_before_merge", "merge_pending", "current_validation_evidence")
    _require_guard(rows, "current_validation_before_merge", "merge_start", "current_validation_evidence")
    _require_guard(rows, "receipt_required_success", "terminal_succeeded", "success_receipt_present")
    _require_guard(rows, "receipt_required_success", "terminal_succeeded", "merge_receipt_present")

    for row in rows.values():
        if "validation_failed" in row.source_states and row.target_state == "terminal_succeeded":
            raise BoundedInvariantViolation(
                "failed_validation_cannot_merge_to_success", row.transition_id, "failed validation reaches success"
            )
    for row in rows.values():
        if row.event == "policy_change":
            if row.source_states != (row.target_state,):
                raise BoundedInvariantViolation(
                    "nonretroactive_policy_changes", row.transition_id, "policy change changes state"
                )
            _require_guard(rows, "nonretroactive_policy_changes", row.transition_id, "policy_change_is_nonretroactive")
        else:
            _require_guard(rows, "nonretroactive_policy_changes", row.transition_id, "policy_authorized_at_event_time")


def _seed(payload: dict[str, Any], transition_id: str, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    candidate = deepcopy(payload)
    transition = next(item for item in candidate["transitions"] if item["id"] == transition_id)
    mutate(transition)
    return candidate


def _remove_guard(guard: str) -> Callable[[dict[str, Any]], None]:
    return lambda transition: transition["guards"].remove(guard)


_SEEDS: tuple[tuple[str, str, Callable[[dict[str, Any]], None]], ...] = (
    ("single_authoritative_owner", "claim_ready", _remove_guard("single_authoritative_owner")),
    ("revision_cas", "claim_ready", _remove_guard("revision_cas_matches")),
    ("idempotency_key_for_effects", "dispatch_claimed", _remove_guard("idempotency_key_bound")),
    ("fresh_lease_and_fence", "merge_start", _remove_guard("fresh_lease_and_fence")),
    ("unknown_outcome_reconciliation", "reconcile_unknown", lambda row: row.__setitem__("event", "retry")),
    ("single_terminalization", "terminal_succeeded", _remove_guard("single_terminalization")),
    ("current_validation_before_merge", "merge_start", _remove_guard("current_validation_evidence")),
    ("external_effect_preservation", "provider_timeout", _remove_guard("external_effect_preserved")),
    ("deterministic_owner_restart", "owner_restart_reconciles", _remove_guard("durable_state_reconstructed")),
    ("event_materialized_state_reconciliation", "claim_ready", _remove_guard("transition_service_only")),
    ("receipt_required_success", "terminal_succeeded", _remove_guard("success_receipt_present")),
    ("failed_validation_cannot_merge_to_success", "terminal_succeeded", lambda row: row.__setitem__("source_states", ["validation_failed"])),
    ("nonretroactive_policy_changes", "policy_change_ready", _remove_guard("policy_change_is_nonretroactive")),
)


def _assert_rejected(
    invariant: str, transition_id: str, candidate: dict[str, Any]
) -> StateTransitionTableError | BoundedInvariantViolation:
    try:
        table = StateTransitionTable.from_dict(candidate)
        _check_semantic_invariants(table)
    except (StateTransitionTableError, BoundedInvariantViolation) as exc:
        return exc
    raise AssertionError(f"seed for {invariant} at {transition_id} escaped qualification")


def _changed_transition_ids(candidate: dict[str, Any]) -> list[str]:
    baseline = {item["id"]: item for item in _payload()["transitions"]}
    return [
        item["id"]
        for item in candidate["transitions"]
        if item != baseline[item["id"]]
    ]


def test_bounded_corpus_covers_every_state_event_invariant_and_transition() -> None:
    table = load_state_transition_table()
    _check_semantic_invariants(table)

    assert set(table.states) == set(CANONICAL_SUPERVISOR_STATES)
    assert set(table.events) == set(CANONICAL_SUPERVISOR_EVENTS)
    assert set(table.invariants) >= set(REQUIRED_STATE_TRANSITION_INVARIANTS)
    assert {row.event for row in table.transitions} == set(table.events)
    assert {state for row in table.transitions for state in (*row.source_states, row.target_state)} == set(table.states)

    # One-step exploration proves each declared edge resolves only with its
    # exact guards.  Two-step exploration covers every composable hand-off.
    for row in table.transitions:
        assert table.resolve(row.event, row.source_states[0], satisfied_guards=row.guards) == row
        with pytest.raises(StateTransitionTableError, match="exactly one row"):
            table.resolve(row.event, row.source_states[0])
    composable = 0
    for first, second in product(table.transitions, repeat=2):
        if first.target_state in second.source_states:
            composable += 1
            assert table.resolve(second.event, first.target_state, satisfied_guards=second.guards) == second
    assert composable > 0


@pytest.mark.parametrize(("invariant", "transition_id", "mutate"), _SEEDS, ids=[seed[0] for seed in _SEEDS])
def test_each_seeded_violation_is_rejected_with_a_minimal_reproducible_counterexample(
    invariant: str, transition_id: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    assert invariant in REQUIRED_STATE_TRANSITION_INVARIANTS
    candidate = _seed(_payload(), transition_id, mutate)
    rejection = _assert_rejected(invariant, transition_id, candidate)
    assert _changed_transition_ids(candidate) == [transition_id]

    # Recreating the seed from disk must produce the same one-transition
    # witness; no random generation or external prover is involved.
    replay = _seed(_payload(), transition_id, mutate)
    assert replay == candidate
    assert _changed_transition_ids(replay) == [transition_id]
    assert str(rejection)

    qualification = json.loads(_QUALIFICATION_PATH.read_text(encoding="utf-8"))
    witness = qualification["seeded_violations"][invariant]
    assert witness["transition_id"] == transition_id
    assert witness["minimal_trace"] == [transition_id]


def test_machine_readable_qualification_is_complete_and_truthful_about_optional_datasets_vectors() -> None:
    qualification = json.loads(_QUALIFICATION_PATH.read_text(encoding="utf-8"))
    coverage = qualification["coverage"]
    assert set(coverage["states"]) == set(CANONICAL_SUPERVISOR_STATES)
    assert set(coverage["events"]) == set(CANONICAL_SUPERVISOR_EVENTS)
    assert set(coverage["invariants"]) == set(REQUIRED_STATE_TRANSITION_INVARIANTS)
    assert set(qualification["seeded_violations"]) == set(REQUIRED_STATE_TRANSITION_INVARIANTS)
    assert qualification["bounds"] == {"max_trace_length": 2, "model_tasks": 1}

    vectors = qualification["datasets_formal_vectors"]
    assert vectors["mode"] == "consume_when_installed"
    for relative_path in vectors["discovery_paths"]:
        candidate = _ROOT / relative_path
        if candidate.is_dir():
            for vector in sorted(candidate.glob("*.json")):
                loaded = json.loads(vector.read_text(encoding="utf-8"))
                assert isinstance(loaded, (dict, list))
