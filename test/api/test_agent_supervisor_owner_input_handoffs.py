"""Normal-import, temporary-DuckDB controls for owner-input coordination.

An observation requests reconsideration. It supplies no legal, execution,
review, or completion authority.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository as intent_module
from ipfs_accelerate_py.agent_supervisor.task_sources import database_task_source as database_module
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import duckdb_available
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentEventType, IntentReceipt, IntentRepository, IntentRepositoryConflictError,
    IntentRepositoryError, OWNER_INPUT_WAIT_KIND, OWNER_INPUT_OBSERVATION_KIND,
    missing_current_evidence_on, open_intent_repository, task_authority_spec_cid,
)

pytestmark = pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
TASK = "task:needs-owner-input"
OTHER_TASK = "task:independent"
OWNER = "owner:legal-cohort"
INPUT_KEY = "legal_cohort"
MISSING = ("original_owner_selection_receipt",)
DIGEST_A, DIGEST_B, DIGEST_C = (
    "sha256:" + hashlib.sha256(value).hexdigest() for value in (b"input-a", b"input-b", b"input-c")
)
BODY = {
    "completion": "manual",
    "admitted": False, "formalized": False, "gold_or_teacher_admission_granted": False,
    "resource_budget": {"max_calls": 0},
    "source_revision": "fictional-world-law-fixture-v1",
}
ACCEPTANCE = [{"criterion": "Authenticated owner input remains required.", "evidence_kind": "validation"}]


def _task(repo, cid=TASK, *, status="ready", body=None, acceptance=None, expected_revision=None):
    return repo.upsert_task(
        task_cid=cid, task_alias="INPUT" if cid == TASK else "INDEPENDENT",
        goal_cid="goal:owner-input", plan_cid="plan:owner-input",
        objective_id="objective:owner-input", status=status,
        body=deepcopy(BODY if body is None else body), identity={"fixture": cid},
        dependencies=[], acceptance=deepcopy(ACCEPTANCE if acceptance is None else acceptance),
        expected_revision=expected_revision,
    )


@pytest.fixture
def handoffs(tmp_path: Path, monkeypatch):
    clock = [1_900_000_000_000]
    monkeypatch.setattr(
        intent_module, "_utc_iso",
        lambda: datetime.fromtimestamp(clock[0] / 1000, tz=timezone.utc)
        .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    path = tmp_path / "control.duckdb"
    repo = open_intent_repository(
        path, owner_id="owner:input-test", evidence_freshness_seconds=60,
        clock_ms=lambda: clock[0],
    )
    repo.upsert_objective(
        objective_id="objective:owner-input", objective_alias="owner-input", title="Owner input"
    )
    repo.upsert_goal(
        goal_cid="goal:owner-input", goal_alias="owner-input-goal", title="Owner input",
        objective_id="objective:owner-input",
    )
    repo.upsert_plan(
        plan_cid="plan:owner-input", plan_alias="owner-input-plan", status="active",
        goal_cid="goal:owner-input",
    )
    _task(repo)
    _task(repo, OTHER_TASK)
    fixture = SimpleNamespace(repo=repo, source=DatabaseTaskSource(intent=repo), clock=clock, path=path)
    try:
        yield fixture
    finally:
        repo.close()


def _request(fixture, **changes):
    args = {
        "task_cid": TASK, "owner_id": OWNER, "input_key": INPUT_KEY,
        "missing_fields": MISSING, "observed_input_digest": DIGEST_A,
        "expected_revision": fixture.repo.get_task(TASK)["revision"],
    }
    args.update(changes)
    return fixture.source.request_owner_input(**args)


def _evidence(fixture, digest=DIGEST_B, *, task_cid=TASK, body_changes=None):
    body = {
        "task_authority_spec_cid": task_authority_spec_cid(fixture.repo.get_task(TASK)),
        "owner_id": OWNER, "input_key": INPUT_KEY,
    }
    body.update(body_changes or {})
    return fixture.repo.record_evidence(
        task_cid=task_cid, evidence_kind="owner_input_observation", digest=digest, body=body,
    )


def _observe(fixture, request, evidence, *, digest=DIGEST_B, **changes):
    args = {
        "task_cid": TASK, "request_id": request.subject_id, "owner_id": OWNER,
        "observed_input_digest": digest, "observation_evidence_id": evidence.subject_id,
    }
    args.update(changes)
    return fixture.source.observe_owner_input(**args)


def _view(fixture, *, owner_id=OWNER, limit=100, after_event_cursor=0):
    return tuple(dict(row) for row in fixture.source.owner_input_handoffs(
        owner_id=owner_id, limit=limit, after_event_cursor=after_event_cursor,
    ))


def _blocks(repo):
    with repo._connection(write=False) as connection:
        rows = connection.execute(
            "SELECT block_id, task_cid, blocker_kind, blocker_id, reason, created_at, cleared_at, state "
            "FROM task_blocks ORDER BY block_id"
        ).fetchall()
    return tuple(tuple(row[index] for index in range(8)) for row in rows)


def _state(fixture):
    return (
        deepcopy(dict(fixture.repo.get_task(TASK))),
        deepcopy(dict(fixture.repo.get_task(OTHER_TASK))),
        _blocks(fixture.repo), fixture.repo.event_watermark(), _view(fixture),
    )


def _assert_preserved(before, after):
    for key in ("identity", "dependencies", "acceptance", "outputs", "validations"):
        assert after[key] == before[key]
    for key, value in BODY.items():
        assert after["body"][key] == value


def test_request_blocks_one_task_and_exposes_owner_filtered_readonly_view(handoffs):
    before = deepcopy(dict(handoffs.repo.get_task(TASK)))
    independent = deepcopy(dict(handoffs.repo.get_task(OTHER_TASK)))
    request = _request(handoffs)
    assert request.changed is True
    task = handoffs.repo.get_task(TASK)
    assert task["status"] == "blocked"
    _assert_preserved(before, task)
    assert dict(handoffs.repo.get_task(OTHER_TASK)) == independent
    assert OTHER_TASK in {row["task_cid"] for row in handoffs.repo.select_ready_tasks()}
    assert TASK not in {row["task_cid"] for row in handoffs.repo.select_ready_tasks()}
    rows = _view(handoffs)
    assert len(rows) == 1
    assert rows[0]["request_id"] == request.subject_id
    assert rows[0]["owner_id"] == OWNER
    assert rows[0]["input_key"] == INPUT_KEY
    assert tuple(rows[0]["missing_fields"]) == MISSING
    assert rows[0]["observed_input_digest"] == DIGEST_A
    assert rows[0]["reconsideration_requested"] is False
    assert rows[0]["event_cursor"] == request.global_sequence
    snapshot = _state(handoffs)
    assert _view(handoffs, owner_id="owner:unrelated") == ()
    assert _view(handoffs, limit=1) == rows
    assert _state(handoffs) == snapshot


def test_request_exact_repeat_uses_same_identity_without_event_or_revision_churn(handoffs):
    request = _request(handoffs)
    before = _state(handoffs)
    handoffs.clock[0] += 1_000
    repeated = _request(handoffs)  # Uses the refreshed task revision.
    assert repeated.subject_id == request.subject_id
    assert repeated.changed is False
    assert _state(handoffs) == before


def test_request_stale_cas_refuses_without_creating_handoff(handoffs):
    stale_revision = handoffs.repo.get_task(TASK)["revision"]
    handoffs.repo.block_task(
        task_cid=TASK, blocker_kind="operator", blocker_id="separate-hold", reason="Retain hold"
    )
    before = _state(handoffs)
    with pytest.raises(IntentRepositoryConflictError):
        _request(handoffs, expected_revision=stale_revision)
    assert _state(handoffs) == before


@pytest.mark.parametrize("status", ["completed", "cancelled", "claimed", "in_progress"])
def test_request_refuses_terminal_or_claimed_tasks(handoffs, status):
    current = handoffs.repo.get_task(TASK)
    _task(handoffs.repo, status=status, expected_revision=current["revision"])
    before = _state(handoffs)
    with pytest.raises(IntentRepositoryError):
        _request(handoffs)
    assert _state(handoffs) == before


def test_unchanged_digest_has_no_notification_and_changed_digest_requests_reconsideration(handoffs):
    request = _request(handoffs)
    baseline = _evidence(handoffs, DIGEST_A)
    before = _state(handoffs)
    unchanged = _observe(handoffs, request, baseline, digest=DIGEST_A)
    assert unchanged.changed is False
    assert _state(handoffs) == before
    evidence = _evidence(handoffs)
    task_before = deepcopy(dict(handoffs.repo.get_task(TASK)))
    cursor = handoffs.repo.event_watermark()
    observed = _observe(handoffs, request, evidence)
    assert observed.changed is True
    assert observed.event_type == IntentEventType.EVIDENCE_RECORDED.value
    assert observed.details["reconsideration_requested"] is True
    assert observed.details["owner_id"] == OWNER
    assert observed.details["request_id"] == request.subject_id
    assert observed.global_sequence == cursor + 1
    task_after = handoffs.repo.get_task(TASK)
    assert task_after["status"] == "blocked"
    _assert_preserved(task_before, task_after)
    row = _view(handoffs)[0]
    assert row["observed_input_digest"] == DIGEST_B
    assert row["reconsideration_requested"] is True
    assert row["event_cursor"] == observed.global_sequence
    assert TASK not in {row["task_cid"] for row in handoffs.repo.select_ready_tasks()}


def test_current_input_deduplicates_but_a_to_b_to_a_notifies_each_change(handoffs):
    request = _request(handoffs)
    evidence_b = _evidence(handoffs)
    first = _observe(handoffs, request, evidence_b)
    before = _state(handoffs)
    assert _observe(handoffs, request, evidence_b).changed is False
    assert _state(handoffs) == before
    evidence_a = _evidence(handoffs, DIGEST_A)
    second = _observe(handoffs, request, evidence_a, digest=DIGEST_A)
    assert second.changed is True
    assert second.global_sequence > first.global_sequence
    assert _view(handoffs)[0]["observed_input_digest"] == DIGEST_A
    before = _state(handoffs)
    assert _observe(handoffs, request, evidence_a, digest=DIGEST_A).changed is False
    assert _state(handoffs) == before


@pytest.mark.parametrize(
    "invalid",
    ["missing", "wrong_digest", "wrong_task", "wrong_owner", "wrong_key", "wrong_spec", "future", "stale"],
)
def test_observation_requires_fresh_same_task_digest_and_exact_owner_binding(handoffs, invalid):
    request = _request(handoffs)
    body_changes = {
        "wrong_owner": {"owner_id": "owner:other"},
        "wrong_key": {"input_key": "other-input"},
        "wrong_spec": {"task_authority_spec_cid": "sha256:" + "f" * 64},
    }.get(invalid)
    original_clock = handoffs.clock[0]
    if invalid == "future":
        handoffs.clock[0] += 1_000
    evidence = _evidence(
        handoffs,
        DIGEST_C if invalid == "wrong_digest" else DIGEST_B,
        task_cid=OTHER_TASK if invalid == "wrong_task" else TASK,
        body_changes=body_changes,
    )
    if invalid == "future":
        handoffs.clock[0] = original_clock
    elif invalid == "stale":
        handoffs.clock[0] += 61_000
    changes = {"observation_evidence_id": "evidence:missing"} if invalid == "missing" else {}
    before = _state(handoffs)
    with pytest.raises((IntentRepositoryError, KeyError)):
        _observe(handoffs, request, evidence, **changes)
    assert _state(handoffs) == before


@pytest.mark.parametrize("invalid", ["owner", "task", "request"])
def test_observation_refuses_foreign_request_owner_or_task(handoffs, invalid):
    request = _request(handoffs)
    evidence = _evidence(handoffs)
    changes = {
        "owner": {"owner_id": "owner:other"},
        "task": {"task_cid": OTHER_TASK},
        "request": {"request_id": "request:missing"},
    }[invalid]
    before = _state(handoffs)
    with pytest.raises((IntentRepositoryError, KeyError)):
        _observe(handoffs, request, evidence, **changes)
    assert _state(handoffs) == before


def test_observation_refuses_request_for_stale_task_spec(handoffs):
    request = _request(handoffs)
    current = handoffs.repo.get_task(TASK)
    changed_body = {**dict(current["body"]), "source_revision": "fictional-world-law-fixture-v2"}
    _task(handoffs.repo, status="blocked", body=changed_body, expected_revision=current["revision"])
    evidence = _evidence(handoffs)  # New-spec evidence cannot revive the old request.
    before = _state(handoffs)
    with pytest.raises(IntentRepositoryError):
        _observe(handoffs, request, evidence)
    assert _state(handoffs) == before


@pytest.mark.parametrize("status", ["completed", "claimed"])
def test_observation_refuses_terminal_or_claimed_task_without_notification(handoffs, status):
    request = _request(handoffs)
    current = handoffs.repo.get_task(TASK)
    _task(handoffs.repo, status=status, expected_revision=current["revision"])
    evidence = _evidence(handoffs)
    before = _state(handoffs)
    with pytest.raises(IntentRepositoryError):
        _observe(handoffs, request, evidence)
    assert _state(handoffs) == before


@pytest.mark.parametrize("operation", ["request", "observe"])
def test_handoff_event_failure_rolls_back_all_mutations(handoffs, monkeypatch, operation):
    if operation == "observe":
        request = _request(handoffs)
        evidence = _evidence(handoffs)
    before = _state(handoffs)
    original = IntentRepository._append_event

    def fail_after_append(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("injected handoff event failure")

    with monkeypatch.context() as patch:
        patch.setattr(IntentRepository, "_append_event", fail_after_append)
        with pytest.raises(RuntimeError, match="injected handoff event failure"):
            if operation == "request":
                _request(handoffs)
            else:
                _observe(handoffs, request, evidence)
    assert _state(handoffs) == before


def test_handoff_state_survives_restart_and_admitted_event_rebuild(handoffs):
    request = _request(handoffs)
    _observe(handoffs, request, _evidence(handoffs))
    before = _state(handoffs)
    handoffs.repo.close()
    with open_intent_repository(
        handoffs.path, owner_id="owner:input-test", evidence_freshness_seconds=60,
        clock_ms=lambda: handoffs.clock[0],
    ) as reopened:
        handoffs.repo = reopened
        handoffs.source = DatabaseTaskSource(intent=reopened)
        assert _state(handoffs) == before
        projection = reopened.snapshot().projection_cid
        reopened.rebuild_projections_from_events()
        assert _state(handoffs) == before
        assert reopened.snapshot().projection_cid == projection


def test_owner_view_returns_active_requests_only(handoffs):
    _request(handoffs)
    assert len(_view(handoffs)) == 1
    handoffs.repo.unblock_task(task_cid=TASK)
    assert _view(handoffs) == ()


def test_generic_block_exact_repeat_is_idempotent_with_refreshed_revision(handoffs):
    args = {"task_cid": TASK, "blocker_kind": "operator", "blocker_id": "hold:one", "reason": "Keep hold"}
    first = handoffs.repo.block_task(**args, expected_revision=handoffs.repo.get_task(TASK)["revision"])
    before = _state(handoffs)
    handoffs.clock[0] += 1_000
    repeated = handoffs.repo.block_task(**args, expected_revision=handoffs.repo.get_task(TASK)["revision"])
    assert repeated.subject_id == first.subject_id
    assert repeated.changed is False
    assert _state(handoffs) == before


def test_selective_unblock_retains_other_blocks_and_missing_block_refuses(handoffs):
    first = handoffs.repo.block_task(
        task_cid=TASK, blocker_kind="operator", blocker_id="hold:first", reason="First hold"
    )
    second = handoffs.repo.block_task(
        task_cid=TASK, blocker_kind="operator", blocker_id="hold:second", reason="Second hold"
    )
    handoffs.repo.unblock_task(
        task_cid=TASK, block_id=first.subject_id,
        expected_revision=handoffs.repo.get_task(TASK)["revision"],
    )
    assert handoffs.repo.get_task(TASK)["status"] == "blocked"
    active = {row[0] for row in _blocks(handoffs.repo) if row[7] == "active"}
    assert active == {second.subject_id}
    assert TASK not in {row["task_cid"] for row in handoffs.repo.select_ready_tasks()}
    before = _state(handoffs)
    with pytest.raises((IntentRepositoryError, KeyError)):
        handoffs.repo.unblock_task(
            task_cid=TASK, block_id="block:missing",
            expected_revision=handoffs.repo.get_task(TASK)["revision"],
        )
    assert _state(handoffs) == before


def test_packet_required_blocked_task_still_requires_native_packet_after_observation(handoffs):
    current = handoffs.repo.get_task(TASK)
    _task(
        handoffs.repo, status="blocked", body={**BODY, "execution_packet_required": True},
        expected_revision=current["revision"],
    )
    request = _request(handoffs)
    _observe(handoffs, request, _evidence(handoffs))
    task = handoffs.repo.get_task(TASK)
    assert task["status"] == "blocked"
    assert task["body"]["execution_packet_required"] is True
    before = _state(handoffs)
    with pytest.raises(IntentRepositoryConflictError, match="packet_current_native_contract_required"):
        handoffs.repo.cas_task_status(
            task_cid=TASK, expected_revision=task["revision"], new_status="in_progress",
            receipt={"operation": "fictional_packet_gate_control"},
        )
    assert _state(handoffs) == before


@pytest.mark.parametrize("acceptance", [[], [{"criterion": "An input note is not acceptance."}]])
@pytest.mark.parametrize("coordination_rows", ["wait", "observation"])
def test_coordination_rows_never_satisfy_completion_after_explicit_block_clear(
    handoffs, acceptance, coordination_rows,
):
    current = handoffs.repo.get_task(TASK)
    _task(handoffs.repo, acceptance=acceptance, expected_revision=current["revision"])
    request = _request(handoffs)
    if coordination_rows == "observation":
        evidence = _evidence(handoffs)
        handoffs.clock[0] += 59_000
        _observe(handoffs, request, evidence)
        # Source evidence expires naturally; the coordination note stays fresh.
        handoffs.clock[0] += 2_000
    handoffs.repo.unblock_task(task_cid=TASK)
    assert handoffs.repo.get_task(TASK)["status"] == "ready"
    current_evidence = handoffs.repo.current_evidence_for_task(TASK)
    kinds = {row["evidence_kind"] for row in current_evidence}
    assert kinds
    assert kinds <= {OWNER_INPUT_WAIT_KIND, OWNER_INPUT_OBSERVATION_KIND}
    satisfied, missing = handoffs.repo.required_evidence_satisfied(TASK)
    assert satisfied is False
    assert missing
    digests = [row["digest"] for row in current_evidence]
    before = _state(handoffs)
    with handoffs.repo._connection(write=False) as connection:
        for supplied in (None, digests):
            assert missing_current_evidence_on(
                connection, TASK, evidence_digests=supplied,
                now_ms=handoffs.clock[0], evidence_freshness_seconds=60,
            )
    for supplied in (None, digests):
        with pytest.raises(IntentRepositoryError):
            handoffs.repo.cas_task_status(
                task_cid=TASK, expected_revision=handoffs.repo.get_task(TASK)["revision"],
                new_status="completed", receipt={"operation": "fictional_completion_control"},
                evidence_digests=supplied,
            )
        assert _state(handoffs) == before


def test_bound_owner_requires_active_transaction_and_retains_outer_rollback(handoffs):
    before = _state(handoffs)
    with handoffs.repo._connection(write=False) as connection:
        borrowed = IntentRepository(
            bound_connection=connection, clock_ms=lambda: handoffs.clock[0],
            evidence_freshness_seconds=60,
        )
        bound_fixture = SimpleNamespace(repo=borrowed, source=DatabaseTaskSource(intent=borrowed))
        with pytest.raises(IntentRepositoryError, match="active owner transaction"):
            _request(bound_fixture)
    assert _state(handoffs) == before
    with pytest.raises(RuntimeError, match="outer owner rollback"):
        with handoffs.repo._connection(write=True) as connection:
            borrowed = IntentRepository(
                bound_connection=connection, clock_ms=lambda: handoffs.clock[0],
                evidence_freshness_seconds=60,
            )
            bound_fixture = SimpleNamespace(repo=borrowed, source=DatabaseTaskSource(intent=borrowed))
            assert _request(bound_fixture).changed is True
            raise RuntimeError("outer owner rollback")
    assert _state(handoffs) == before
    with handoffs.repo._connection(write=True) as connection:
        borrowed = IntentRepository(
            bound_connection=connection, clock_ms=lambda: handoffs.clock[0],
            evidence_freshness_seconds=60,
        )
        bound_fixture = SimpleNamespace(repo=borrowed, source=DatabaseTaskSource(intent=borrowed))
        request = _request(bound_fixture)
    assert _view(handoffs)[0]["request_id"] == request.subject_id
    assert handoffs.repo.get_task(TASK)["status"] == "blocked"


@pytest.mark.parametrize("kind", [OWNER_INPUT_WAIT_KIND, OWNER_INPUT_OBSERVATION_KIND])
@pytest.mark.parametrize("route", ["repository", "source"])
def test_generic_evidence_api_cannot_mutate_owner_input_coordination(handoffs, kind, route):
    body = {
        "owner_id": OWNER, "input_key": INPUT_KEY, "missing_fields": list(MISSING),
        "observed_input_digest": DIGEST_A,
        "expected_revision": handoffs.repo.get_task(TASK)["revision"],
    }
    if kind == OWNER_INPUT_OBSERVATION_KIND:
        request = _request(handoffs)
        evidence = _evidence(handoffs)
        body = {
            "owner_id": OWNER, "request_id": request.subject_id,
            "observed_input_digest": DIGEST_B, "observation_evidence_id": evidence.subject_id,
        }
    target = handoffs.repo if route == "repository" else handoffs.source
    before = _state(handoffs)
    with pytest.raises(IntentRepositoryError):
        target.record_evidence(task_cid=TASK, evidence_kind=kind, digest=content_identity(body), body=body)
    assert _state(handoffs) == before


def test_whitespace_owner_and_key_are_canonical_retrievable_and_observable(handoffs):
    request = _request(handoffs, owner_id=f" {OWNER} ", input_key=f" {INPUT_KEY} ")
    rows = _view(handoffs)
    assert len(rows) == 1
    assert rows[0]["owner_id"] == OWNER
    assert rows[0]["input_key"] == INPUT_KEY
    assert rows[0]["request_id"] == request.subject_id
    before = _state(handoffs)
    canonical_repeat = _request(handoffs)
    assert canonical_repeat.subject_id == request.subject_id
    assert canonical_repeat.changed is False
    assert _state(handoffs) == before
    observed = _observe(handoffs, request, _evidence(handoffs))
    assert observed.changed is True
    assert observed.details["owner_id"] == OWNER
    assert _view(handoffs)[0]["observed_input_digest"] == DIGEST_B


def test_owner_handoff_cursor_drains_requests_then_delivers_earlier_request_update(handoffs):
    requests = [
        _request(handoffs, input_key=key)
        for key in (INPUT_KEY, "origin_and_decoder", "resources_and_cost")
    ]
    cursor = 0
    delivered = []
    for _ in requests:
        rows = _view(handoffs, limit=1, after_event_cursor=cursor)
        assert len(rows) == 1
        assert rows[0]["event_cursor"] > cursor
        cursor = rows[0]["event_cursor"]
        delivered.append(rows[0]["request_id"])
    assert delivered == [request.subject_id for request in requests]
    assert _view(handoffs, limit=1, after_event_cursor=cursor) == ()
    observed = _observe(handoffs, requests[0], _evidence(handoffs))
    updated = _view(handoffs, limit=1, after_event_cursor=cursor)
    assert len(updated) == 1
    assert updated[0]["request_id"] == requests[0].subject_id
    assert updated[0]["reconsideration_requested"] is True
    assert updated[0]["observed_input_digest"] == DIGEST_B
    assert updated[0]["event_cursor"] == observed.global_sequence > cursor
    assert _view(handoffs, limit=1, after_event_cursor=observed.global_sequence) == ()


@pytest.mark.parametrize("operation", ["request", "observe"])
def test_remote_public_handoffs_use_separate_closed_owner_command(monkeypatch, operation):
    source = DatabaseTaskSource.__new__(DatabaseTaskSource)
    source._intent = SimpleNamespace(uses_quack_transport=True)
    expected_body = (
        {"owner_id": OWNER, "input_key": INPUT_KEY, "missing_fields": list(MISSING),
         "observed_input_digest": DIGEST_A, "expected_revision": 7}
        if operation == "request" else
        {"request_id": "request:remote", "owner_id": OWNER,
         "observed_input_digest": DIGEST_B, "observation_evidence_id": "evidence:remote"}
    )
    kind = OWNER_INPUT_WAIT_KIND if operation == "request" else OWNER_INPUT_OBSERVATION_KIND
    receipt = IntentReceipt(
        event_id="event:remote-handoff", event_type=IntentEventType.EVIDENCE_RECORDED.value,
        global_sequence=17, recorded_at="2030-03-17T17:46:40+00:00",
        subject_id="request:remote", revision=8, changed=True,
        details={"request_id": "request:remote", "owner_id": OWNER,
                 "reconsideration_requested": operation == "observe"},
    )
    calls = []

    def submit(command, payload):
        calls.append((command, payload))
        return receipt.to_dict()

    def generic_evidence_is_not_coordination(**kwargs):
        raise AssertionError("generic evidence authority must not coordinate owner input")

    monkeypatch.setattr(database_module, "submit_quack_owner_command", submit)
    monkeypatch.setattr(source, "record_evidence", generic_evidence_is_not_coordination)
    if operation == "request":
        observed = source.request_owner_input(
            task_cid=TASK, owner_id=OWNER, input_key=INPUT_KEY, missing_fields=MISSING,
            observed_input_digest=DIGEST_A, expected_revision=7,
        )
    else:
        observed = source.observe_owner_input(
            task_cid=TASK, request_id="request:remote", owner_id=OWNER,
            observed_input_digest=DIGEST_B, observation_evidence_id="evidence:remote",
        )
    assert observed.to_dict() == receipt.to_dict()
    assert database_module.QUACK_OWNER_COMMAND_COORDINATE_OWNER_INPUT == "coordinate_owner_input_v1"
    assert calls == [(
        database_module.QUACK_OWNER_COMMAND_COORDINATE_OWNER_INPUT,
        {"task_cid": TASK, "evidence_kind": kind, "body": expected_body},
    )]
