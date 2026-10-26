"""Nonadmitting unit checks for UNKNOWN disposition policy.

These tests neither create native receipts/grants nor open databases. The
context fixtures stop at a trapped custody boundary: filesystem/source and
custody seams are explicitly mocked, so reaching that boundary proves only
structural validation. Genuine gateway, three-store, WAL and crash closeout
qualification must run separately under the original native finite owner.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    planning_unknown_abandonment as policy,
    planning_unknown_abandonment_settlement as settlement,
)


def identity(number=2):
    return dict(task_cid="task:nonadmitting-policy-fixture",
        attempt_id=f"attempt:nonadmitting-policy-{number}", attempt_number=number,
        claim_id=f"claim:nonadmitting-policy-{number}",
        lease_id=f"lease:nonadmitting-policy-{number}",
        owner_session_id=f"owner:nonadmitting-policy-{number}",
        fencing_token=number, fence_epoch=number)


def unknown(ident):
    result = {key: ident[key] for key in (
        "task_cid", "attempt_id", "claim_id", "lease_id", "fencing_token", "fence_epoch")}
    result.update(schema="database-portal-callback-intent@1",
        callback_state="started_outcome_unknown",
        provider_effect_state="unknown_may_have_started")
    # Deliberately retain legal noncanonical whitespace in the opaque raw row.
    return dict(invocation_id=f"provider:nonadmitting-policy-{ident['attempt_number']}",
        attempt_id=ident["attempt_id"], task_cid=ident["task_cid"],
        idempotency_key="provider:" + ident["attempt_id"],
        owner_session_id=ident["owner_session_id"], recorded_at_ms=1700000000000,
        result_json=json.dumps(result, indent=2))


def attempt_record(number=2, *, failed=False):
    result = identity(number)
    result.update(schema="nonadmitting-unit-attempt", task_alias="LIR-9001",
        committed_phase="failed" if failed else "context",
        status="failed" if failed else "running", revision=3 if failed else 2,
        started_at_ms=1700000000000, finished_at_ms=1700000000300 if failed else None,
        body={"fixture_only": True})
    return result


def pin(path, raw=b""):
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def seal(context):
    context["context_cid"] = policy.content_identity(
        {key: value for key, value in context.items() if key != "context_cid"})


class ReachedUnadmittedCustodyBoundary(RuntimeError):
    """The unit fixture deliberately supplies no native custody authority."""


@pytest.fixture
def structural_context(tmp_path):
    stores = {}
    for role in ("canonical", "execution", "coordination"):
        placeholder = tmp_path / (role + ".placeholder")
        placeholder.write_bytes(b"")  # Regular placeholders, never DuckDB files.
        stores[role] = str(placeholder)
    current, predecessor = attempt_record(), attempt_record(1, failed=True)
    task = dict(task_cid=current["task_cid"], status="in_progress", revision=12,
        body={"board_namespace": "nonadmitting-policy-fixture"})
    snapshot = dict(schema=policy.SNAPSHOT_SCHEMA, task=task,
        intent_task={**deepcopy(task), "identity": {"fixture_only": True},
                     "created_at": "2020-01-01T00:00:00Z"},
        canonical_population={}, attempt=current,
        phases=[{"phase": "claimed"}, {"phase": "context"}],
        providers=[unknown(current)], effects=[],
        coordination={"claim": {}, "lease": {}, "attempt": {},
            "previous_attempts": [{key: predecessor[key] for key in (
                "task_cid", "attempt_id", "attempt_number", "owner_session_id",
                "fencing_token", "fence_epoch")} ]},
        historical_unknowns=[dict(row=unknown(predecessor), attempt=predecessor,
            store_pins=[pin(tmp_path / "unopened-predecessor.placeholder")])])
    source_root = Path(policy.__file__).resolve().parents[1]
    required_sources = (
        "todo_daemon/planning_unknown_abandonment.py",
        "todo_daemon/planning_unknown_abandonment_settlement.py",
        "todo_daemon/implementation_daemon.py", "task_sources/typed_state_owner.py",
        "task_sources/database_task_source.py", "task_sources/intent_repository.py",
        "task_sources/control_plane_contracts.py", "task_sources/duckdb_state.py",
        "merge/database_coordination.py", "runtime/quack_state_server.py",
        "todo_daemon/native_execution_status_owner.py",
        "todo_daemon/native_execution_wal_maintenance.py",
    )
    context = dict(schema=policy.AUTHORITY_SCHEMA, context_cid="",
        task_cid=current["task_cid"], namespace="nonadmitting-policy-fixture",
        expected_task_revision=12, logical_store_id="nonadmitting-policy-store",
        stores=stores, snapshot_pin=pin(tmp_path / "unopened-snapshot.json"),
        source_pins=[pin(source_root / name) for name in required_sources],
        immutable_pins=[pin(tmp_path / "unopened-immutable.placeholder")],
        closed_store_pins={role: pin(stores[role]) for role in ("execution", "coordination")},
        stopped_processes=[],  # The real check is explicitly mocked, never admitted.
        operator_decision={**policy.DECISION, "reason": "Nonadmitting policy fixture",
            "unknown_effect_risk_accepted": True},
        registered_owner_identity={"server_id": "nonadmitting-fixture",
            "store_id": "nonadmitting-policy-store"},
        registered_creator={"pid": 12345, "process_birth_id": "nonadmitting-fixture"},
        custody_artifacts={})  # No native receipt is forged to satisfy custody.
    seal(context)
    return context, snapshot


def structural_call(context, snapshot):
    """Run only through structural checks, with a hard trap before custody."""
    raw = json.dumps(snapshot, sort_keys=True).encode()
    expected = deepcopy(context)
    expected["snapshot_pin"] = pin(context["snapshot_pin"]["path"], raw)
    seal(expected)

    def fixture_read(selected):
        return raw if selected["path"] == expected["snapshot_pin"]["path"] else b""

    with patch.object(policy, "read_pin", side_effect=fixture_read), \
         patch.object(policy, "stopped_processes"), \
         patch.object(policy, "verify_custody", side_effect=ReachedUnadmittedCustodyBoundary):
        return policy.load_context(expected, check_closed_stores=False)


def test_exact_unknown_validation_preserves_opaque_result_bytes():
    ident = identity()
    row = unknown(ident)
    original = deepcopy(row)
    policy.unknown_row(row, task_cid=ident["task_cid"], identity=ident)
    assert row == original
    assert row["result_json"].startswith("{\n")


@pytest.mark.parametrize("field,value", [
    ("owner_session_id", "owner:other"),
    ("idempotency_key", "provider:other-attempt"),
    ("attempt_id", "attempt:other"),
    ("task_cid", "task:other"),
])
def test_unknown_raw_identity_mismatches_refuse(field, value):
    ident = identity()
    row = unknown(ident)
    row[field] = value
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        policy.unknown_row(row, task_cid=ident["task_cid"], identity=ident)


@pytest.mark.parametrize("field,value", [
    ("attempt_id", "attempt:other"), ("task_cid", "task:other"),
    ("claim_id", "claim:other"), ("lease_id", "lease:other"),
    ("fencing_token", 99), ("fence_epoch", 99), ("fencing_token", True), ("fence_epoch", True),
    ("schema", "database-portal-callback-intent@2"),
    ("callback_state", "accepted"), ("provider_effect_state", "not_dispatched"),
])
def test_unknown_decoded_result_mismatches_refuse(field, value):
    ident = identity()
    row = unknown(ident)
    result = json.loads(row["result_json"])
    result[field] = value
    row["result_json"] = json.dumps(result)
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        policy.unknown_row(row, task_cid=ident["task_cid"], identity=ident)


@pytest.mark.parametrize("change", ["missing_claim", "extra_accepted", "duplicate_key"])
def test_unknown_result_requires_exact_closed_json_keys(change):
    ident = identity()
    row = unknown(ident)
    result = json.loads(row["result_json"])
    if change == "missing_claim":
        del result["claim_id"]
        row["result_json"] = json.dumps(result)
    elif change == "extra_accepted":
        row["result_json"] = json.dumps({**result, "accepted": True})
    else:
        row["result_json"] = row["result_json"].rstrip()[:-1] + ',"claim_id":"claim:other"}'
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        policy.unknown_row(row, task_cid=ident["task_cid"], identity=ident)


def test_complete_structural_preimage_reaches_only_unadmitted_custody_boundary(structural_context):
    context, snapshot = structural_context
    with pytest.raises(ReachedUnadmittedCustodyBoundary):
        structural_call(context, snapshot)


@pytest.mark.parametrize("change", ["omit_coordinator", "omit_unknown", "extra_predecessor", "wrong_ordinal", "wrong_fence"])
def test_incomplete_or_inconsistent_predecessor_inventory_refuses_before_custody(structural_context, change):
    context, snapshot = structural_context
    if change == "omit_coordinator":
        snapshot["coordination"]["previous_attempts"] = []
    elif change == "omit_unknown":
        snapshot["historical_unknowns"] = []
    elif change == "extra_predecessor":
        snapshot["coordination"]["previous_attempts"] *= 2
    elif change == "wrong_ordinal":
        snapshot["coordination"]["previous_attempts"][0]["attempt_number"] = 7
    else:
        snapshot["coordination"]["previous_attempts"][0]["fencing_token"] = 7
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        structural_call(context, snapshot)


def test_context_identity_change_refuses_without_reading_any_pin(structural_context):
    context, _snapshot = structural_context
    context["operator_decision"]["reason"] = "Changed after context was sealed"
    with patch.object(policy, "read_pin", side_effect=AssertionError("must refuse before pin reads")):
        with pytest.raises(policy.PlanningUnknownAbandonmentError, match="context identity"):
            policy.load_context(context, check_closed_stores=False)


@pytest.mark.parametrize("field,value", [
    ("unknown_effect_risk_accepted", False), ("provider_outcome", "accepted"),
    ("callback_closed", True), ("remote_cancellation_proved", True),
    ("nonreachability_asserted", True), ("automatic_retry_admitted", True),
    ("completion_authority", True),
])
def test_context_cannot_claim_a_settled_or_effect_free_outcome(structural_context, field, value):
    context, snapshot = structural_context
    context["operator_decision"][field] = value
    with pytest.raises(policy.PlanningUnknownAbandonmentError, match="disposition/risk"):
        structural_call(context, snapshot)


def test_opaque_finite_native_json_hash_does_not_widen_typed_content_ids():
    native = {"body": {"feature_score": 0.75}, "finished_at_ms": None}
    expected = hashlib.sha256(json.dumps(native, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert policy.digest(native) == "sha256:" + expected
    with pytest.raises(ValueError):
        policy.content_identity(native)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_opaque_native_hash_rejects_nonfinite_values(value):
    with pytest.raises(ValueError):
        policy.digest({"body": {"feature_score": value}})


@pytest.mark.parametrize("raw", ['{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
def test_nonfinite_native_json_cannot_reach_a_persisted_digest(raw):
    with pytest.raises(ValueError):
        policy.digest(policy.decode(raw))


def test_audit_body_remains_a_decision_without_completion_or_nonreachability(structural_context):
    context, snapshot = structural_context
    before = deepcopy(snapshot)
    body = policy.audit_body(context, snapshot)
    assert all(body[key] == value for key, value in policy.DECISION.items())
    assert body["recovered_execution_verified_by_this_audit"] is False
    assert body["observation_attestation"] == "exact_native_preimage_registered_by_host_administrator"
    assert body["unknown_provider_rows_digest"] == policy.digest(snapshot["providers"])
    assert body["historical_unknown_rows_digest"] == policy.digest(snapshot["historical_unknowns"])
    assert snapshot == before


def test_audit_evidence_identity_binds_attempt_claim_and_original_revision():
    fields = dict(task_cid=identity()["task_cid"], attempt_id=identity()["attempt_id"],
        claim_id=identity()["claim_id"], control_revision=12)
    expected = policy.evidence_id(**fields)
    assert policy.evidence_id(**fields) == expected
    for key, value in (("attempt_id", "attempt:other"), ("claim_id", "claim:other"), ("control_revision", 13)):
        assert policy.evidence_id(**{**fields, key: value}) != expected


@pytest.mark.parametrize("status,revision", [("retrying", 14), ("in_progress", 15), ("blocked", 14)])
def test_stale_after_rearm_is_refused_before_population_or_control_mutation(status, revision):
    """Exercise the actual guard with explicit unadmitted adapter seams."""
    original = attempt_record()
    failed = attempt_record(failed=True)
    predecessor = attempt_record(1, failed=True)
    receipt = {"settlement_id": "nonadmitting-unit-receipt"}
    phases = [dict(phase="claimed", revision=1, committed_at_ms=1700000000000,
        fencing_token=2, fence_epoch=2, body={}),
        dict(phase="context", revision=2, committed_at_ms=1700000000100,
        fencing_token=2, fence_epoch=2, body={})]
    task_original = dict(task_cid=original["task_cid"], status="in_progress", revision=12, body={})
    intent_original = {**deepcopy(task_original), "identity": {"fixture_only": True},
        "created_at": "2020-01-01T00:00:00Z", "updated_at": "2020-01-01T00:00:00Z"}
    snapshot = dict(attempt=original, providers=[unknown(original)], effects=[], phases=phases,
        task=task_original, intent_task=intent_original,
        coordination={"previous_attempts": [predecessor]})
    audit = {"context": {"stores": {}, "expected_task_revision": 12}, "snapshot": snapshot}

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def fetchall(self):
            return self.rows

    class ExecutionRead:
        def execute(self, sql, values):
            assert sql.startswith("SELECT effect_id FROM effect_claims")
            return Result([])

    class CoordinationRead:
        def execute(self, sql, values):
            assert sql.startswith("SELECT attempt_id,attempt_number FROM task_attempts")
            return Result([(predecessor["attempt_id"], 1), (original["attempt_id"], 2)])

    def record(values, **attributes):
        return SimpleNamespace(**attributes, to_dict=lambda: deepcopy(values))

    coordinator = SimpleNamespace(
        get_task_claim=lambda _key: record(identity(), state="released"),
        get_lease=lambda _key: record(identity(), state="released"),
        get_task_attempt=lambda key: record(predecessor) if key == predecessor["attempt_id"]
            else record({k: v for k, v in identity().items() if k not in {"claim_id", "lease_id"}}, status="failed"),
        _coordinator_context=lambda: (ExitStack(), SimpleNamespace(_require=lambda: CoordinationRead())))
    changed_task = {**deepcopy(task_original), "status": status, "revision": revision}
    changed_intent = {**deepcopy(intent_original), "status": status, "revision": revision}
    task = SimpleNamespace(**changed_task, to_dict=lambda: deepcopy(changed_task))
    phase_failed = dict(phase="failed", revision=3, committed_at_ms=failed["finished_at_ms"],
        fencing_token=2, fence_epoch=2, body=receipt)
    daemon = SimpleNamespace(coordinator=coordinator,
        task_source=SimpleNamespace(get_task=lambda _key: task,
            intent=SimpleNamespace(get_task=lambda _key: deepcopy(changed_intent))),
        phase_history=lambda _key: deepcopy([*phases, phase_failed]),
        _require_connection=lambda: ExecutionRead())
    attempt = SimpleNamespace(**failed, to_dict=lambda: deepcopy(failed))
    with patch.object(settlement, "native_stores", return_value={}), \
         patch.object(settlement, "provider_rows", return_value=deepcopy(snapshot["providers"])), \
         patch.object(policy, "canonical_population", side_effect=AssertionError("stale control must refuse first")):
        with pytest.raises(policy.PlanningUnknownAbandonmentError, match="canonical"):
            settlement.verify_current(daemon, attempt, audit, receipt)
