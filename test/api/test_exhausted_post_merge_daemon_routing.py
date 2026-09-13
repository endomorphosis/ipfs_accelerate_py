"""Public recovery routing retains physical proof and a real task fence.

Git/Portal qualification is supplied at its existing verified boundary; its
native evidence tests are separate. These tests retain the actual historical
and physical suffix classifiers and an actual disposable DuckDB coordinator.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinator,
    DatabaseCoordinationError,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
    TaskExecutionRouteBinding,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_callback_suffix import (
    IDENTITY,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _callback_integration_recovery_evidence,
)
from test.api.test_retained_callback_preflight_suffix import physical_fixture


class OwnerBoundaryReached(RuntimeError):
    pass


def _remap(value, substitutions):
    if isinstance(value, dict):
        return {key: _remap(item, substitutions) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap(item, substitutions) for item in value]
    if isinstance(value, str):
        for old, new in substitutions.items():
            value = value.replace(old, new)
    return value


@contextmanager
def _routing_fixture(tmp_path, monkeypatch, *, typed=True):
    daemon, task, old_attempts, old_phases, _ = physical_fixture()
    original_history = daemon.task_source.task_revision_history_projection(
        task.task_cid
    )
    now = {"ms": 1_789_320_000_000}
    daemon._now_ms = lambda: now["ms"]
    daemon.require_real_execution = True
    coordinator = DatabaseCoordinator(
        tmp_path / "coordination.duckdb",
        clock_ms=lambda: now["ms"],
        default_lease_ms=5000,
    )
    coordinator.open()
    daemon._coordinator = coordinator
    coordinator.register_task(task_cid=task.task_cid, task_id=task.task_alias)
    substitutions, claims, coordination = {}, [], []
    for old in old_attempts.values():
        claim = coordinator.claim_task(
            task_cid=task.task_cid, owner_session_id=old.owner_session_id
        )
        now["ms"] += 5001
        coordinator.expire_task_claim(claim)
        claims.append(claim)
        identity = claim.to_dict()
        for key in IDENTITY:
            if type(getattr(old, key)) is str:
                substitutions[getattr(old, key)] = identity[key]
            else:
                assert getattr(old, key) == identity[key]
        coordination.append(
            daemon._reconcile_failed_attempt_coordination(
                SimpleNamespace(
                    **{key: identity[key] for key in IDENTITY},
                    task_cid=task.task_cid,
                )
            )
        )
    history = _remap(original_history, substitutions)
    attempts = {}
    phases = {}
    for old in old_attempts.values():
        values = _remap(vars(old), substitutions)
        values["body"] = {}
        attempt = SimpleNamespace(**values)
        attempts[attempt.attempt_id] = attempt
        phases[attempt.attempt_id] = _remap(old_phases[old.attempt_id], substitutions)
    source, middle, current = list(attempts.values())
    rows = history["revisions"]
    for index in (3, 4):
        rows[index]["body"]["completion_receipt"]["coordination"] = coordination[0]
    rows[7]["body"]["completion_receipt"]["coordination"] = coordination[1]
    rows[10]["body"]["completion_receipt"]["coordination"] = coordination[2]
    budget = rows[10]["body"]["completion_receipt"]["retry_budget"]
    budget.pop("observation_id")
    budget["observation_id"] = daemon._database_portal_evidence_digest(budget)
    history.pop("projection_cid")
    history["projection_cid"] = content_identity(history)
    task.body = copy.deepcopy(rows[-1]["body"])
    daemon.get_attempt = attempts.get
    daemon.phase_history = phases.__getitem__
    daemon._latest_failed_attempts = lambda: list(attempts.values())
    daemon._typed_deferral_budget_observation = lambda attempt: copy.deepcopy(budget)
    daemon._terminal_coordination_reproduces_read_only = DatabaseImplementationDaemon._terminal_coordination_reproduces_read_only.__get__(
        daemon
    )
    calls, mint_calls = [], []
    seal = object()
    daemon._DatabaseImplementationDaemon__post_merge_queue_authority = seal
    source_adapter = SimpleNamespace(
        get=lambda cid: task if cid == task.task_cid else None,
        task_revision_history_projection=lambda cid: copy.deepcopy(history),
        validate_execution_route_binding=lambda binding, **kwargs: TaskExecutionRouteBinding.from_dict(
            binding
        ).to_dict(),
    )
    embedded = None
    if not typed:
        database = tmp_path / "embedded-control.duckdb"
        initial = DatabaseTaskSource(database)
        initial.materialize(
            {
                "repository_tree_id": "tree:routing",
                "plan_root_cid": "plan:routing",
                "goals": [
                    {
                        "goal_cid": "goal:routing",
                        "goal_alias": "ROUTING",
                        "title": "Routing",
                    }
                ],
                "tasks": [
                    {
                        "task_cid": task.task_cid,
                        "task_id": task.task_alias,
                        "goal_cid": "goal:routing",
                        "status": "ready",
                    }
                ],
            }
        )
        initial.close()
        connection = open_duckdb_connection(database)
        try:
            connection.execute(
                "UPDATE tasks SET revision=?, status=?, body_json=? WHERE task_cid=?",
                [
                    task.revision,
                    task.status,
                    canonical_json_bytes(task.body).decode(),
                    task.task_cid,
                ],
            )
        finally:
            connection.close()
        embedded = DatabaseTaskSource(database, _post_merge_queue_authority=seal)

        def mint(**kwargs):
            assert coordinator._fenced_callback_active is True
            mint_calls.append(kwargs)
            return embedded._mint_post_merge_queue_admission(**kwargs)

        def guarded(**kwargs):
            assert coordinator._fenced_callback_active is True
            assert kwargs.get("_post_merge_recovery_admission") is not None
            calls.append(kwargs)
            return embedded.record_queue_backoff_and_cas_status(**kwargs)

        source_adapter._mint_post_merge_queue_admission = mint
        source_adapter.record_queue_backoff_and_cas_status = guarded
    else:

        def typed_owner(**kwargs):
            assert coordinator._fenced_callback_active is True
            assert "_post_merge_recovery_admission" not in kwargs
            calls.append(kwargs)
            raise OwnerBoundaryReached("verified typed owner command boundary")

        source_adapter.recover_exhausted_post_merge_retry = typed_owner
        source_adapter.record_queue_backoff_and_cas_status = (
            lambda **kwargs: pytest.fail("generic typed rearm reached")
        )
        source_adapter._mint_post_merge_queue_admission = lambda **kwargs: pytest.fail(
            "process-local seal crossed the typed route"
        )
    daemon._task_source = source_adapter
    # Git/Portal proof is independently exercised by native append recovery and
    # typed-owner tests. Keep its checked output fixed while testing routing.
    monkeypatch.setattr(
        daemon,
        "_verified_post_merge_callback_integration_receipt",
        lambda raw, **kwargs: dict(raw),
    )
    evidence = _callback_integration_recovery_evidence(daemon, source)
    context = daemon._retained_callback_suffix_context(task)
    assert context is not None and context["preflight_exhaustion"] is True
    try:
        yield SimpleNamespace(
            daemon=daemon,
            task=task,
            history=history,
            attempts=attempts,
            phases=phases,
            source=source,
            middle=middle,
            current=current,
            coordinator=coordinator,
            calls=calls,
            mint_calls=mint_calls,
            evidence=evidence,
            embedded=embedded,
            seal=seal,
            now=now,
        )
    finally:
        if embedded is not None:
            embedded.close()
        coordinator.close()


def test_typed_recovery_reaches_owner_only_inside_current_real_fence(
    tmp_path, monkeypatch
):
    with _routing_fixture(tmp_path, monkeypatch) as fixture:
        with pytest.raises(OwnerBoundaryReached):
            fixture.daemon.recover_blocked_post_merge_declared_outputs(fixture.evidence)
        assert len(fixture.calls) == 1 and fixture.mint_calls == []
        arguments = fixture.calls[0]
        assert arguments["expected_revision"] == 11
        assert (
            arguments["expected_control_receipt"]["attempt_id"]
            == fixture.current.attempt_id
        )
        assert arguments["receipt"]["attempt_id"] == fixture.source.attempt_id
        assert (
            arguments["receipt"]["post_merge_completion_recovery_seed"][
                "recovery_control_revision"
            ]
            == 11
        )
        assert (
            fixture.coordinator.get_task_claim(fixture.current.claim_id).state.value
            == "expired"
        )


def test_embedded_recovery_preserves_real_sealed_admission(tmp_path, monkeypatch):
    with _routing_fixture(tmp_path, monkeypatch, typed=False) as fixture:
        result = fixture.daemon.recover_blocked_post_merge_declared_outputs(
            fixture.evidence
        )
        assert result["recovered"] is True and result["changed"] is True
        assert len(fixture.mint_calls) == 1 and len(fixture.calls) == 1
        assert fixture.mint_calls[0]["_portable_authority"] is fixture.seal
        assert fixture.embedded.get(fixture.task.task_cid).status == "retrying"
        assert fixture.embedded.get(fixture.task.task_cid).revision == 12


@pytest.mark.parametrize(
    "mutation",
    [
        "history_gap",
        "history_semantics",
        "physical_provider",
        "physical_current_missing",
        "bad_evidence",
        "manual_task",
        "physical_change_inside_fence",
        "new_fence_before_callback",
    ],
)
def test_denied_recovery_never_calls_typed_owner(tmp_path, monkeypatch, mutation):
    with _routing_fixture(tmp_path, monkeypatch) as fixture:
        d = fixture.daemon
        if mutation == "history_gap":
            fixture.history["revisions"].pop(5)
        elif mutation == "history_semantics":
            fixture.history["revisions"][7]["body"]["title"] = "changed"
        elif mutation == "physical_provider":
            fixture.phases[fixture.current.attempt_id][1]["phase"] = "provider"
        elif mutation == "physical_current_missing":
            fixture.attempts.pop(fixture.current.attempt_id)
        elif mutation == "bad_evidence":
            fixture.evidence["source_claim_id"] = "claim:foreign"
        elif mutation == "manual_task":
            fixture.task.body["completion"] = "manual"
        elif mutation == "physical_change_inside_fence":
            original = fixture.coordinator.execute_with_task_fence

            def execute(claim, callback, **kwargs):
                def changed():
                    fixture.phases[fixture.current.attempt_id][1]["phase"] = "provider"
                    return callback()

                return original(claim, changed, **kwargs)

            monkeypatch.setattr(fixture.coordinator, "execute_with_task_fence", execute)
        elif mutation == "new_fence_before_callback":
            original = fixture.coordinator.execute_with_task_fence

            def execute(claim, callback, **kwargs):
                fixture.coordinator.claim_task(
                    task_cid=fixture.task.task_cid,
                    owner_session_id=fixture.current.owner_session_id,
                )
                return original(claim, callback, **kwargs)

            monkeypatch.setattr(fixture.coordinator, "execute_with_task_fence", execute)
        if mutation.startswith("history_"):
            fixture.history.pop("projection_cid")
            fixture.history["projection_cid"] = content_identity(fixture.history)
        with pytest.raises(
            (
                DatabaseImplementationAuthorityError,
                DatabaseImplementationConflictError,
                DatabaseCoordinationError,
            )
        ):
            d.recover_blocked_post_merge_declared_outputs(fixture.evidence)
        assert fixture.calls == [] and fixture.mint_calls == []
