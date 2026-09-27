"""Offline native-control contracts; providers are injected, no model or prover."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import database_task_source as sources
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepositoryBoundsError, IntentRepositoryConflictError,
    IntentRepositoryIntegrityError, open_intent_repository,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population


def _seed(repo, *, count=1002, parked_prefix=1001):
    """Synthetic committed projections in one bounded transaction, not admissions."""
    repo.upsert_objective(objective_id="objective:claim-test", objective_alias="CLAIM-TEST", title="Claim tests")
    for cid, status in (("goal:parked", "analysis_inconclusive"), ("goal:open", "open")):
        repo.upsert_goal(goal_cid=cid, goal_alias=cid, objective_id="objective:claim-test", title=cid, status=status)
    with repo._connection(write=True) as connection:
        connection.execute(
            "INSERT INTO tasks (task_cid,task_alias,goal_cid,ordinal,status,revision) "
            "SELECT 'task:selector:' || i, 'SEL-' || i, "
            "CASE WHEN i < ? THEN 'goal:parked' ELSE 'goal:open' END, i, 'ready', 1 "
            "FROM range(?) x(i)", [parked_prefix, count],
        )


def _park(repo, goal_cid):
    goal = repo.get_goal(goal_cid)
    repo.cas_goal_status(goal_cid=goal_cid, expected_revision=goal["revision"],
                         new_status="analysis_inconclusive", receipt={"admitted": False})


@pytest.mark.parametrize("status", ["proposed", "admitted", "pending", "ready", "todo", "queued", "retrying"])
def test_all_native_ready_statuses_under_parked_goals_are_unclaimable(tmp_path, status):
    with open_intent_repository(tmp_path / "control.duckdb") as repo:
        _seed(repo, count=2, parked_prefix=1)
        with repo._connection(write=True) as connection:
            connection.execute("UPDATE tasks SET status = ?", [status])
        assert [row["task_cid"] for row in repo.select_ready_tasks(limit=1)] == ["task:selector:1"]
        assert repo.parked_ready_task_cids(limit=1) == ("task:selector:0",)
        before = repo.snapshot()
        with pytest.raises(IntentRepositoryConflictError, match="analysis_inconclusive"):
            repo.cas_task_status(task_cid="task:selector:0", expected_revision=1,
                                 new_status="in_progress", receipt={"operation": "database_claim"})
        after = repo.snapshot()
        assert (after.projection_cid, after.event_watermark) == (before.projection_cid, before.event_watermark)
        assert repo.get_task("task:selector:0")["status"] == status


def test_positive_selection_and_parking_apply_before_the_1000_row_limit(tmp_path):
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        _seed(source._intent)
        target = "task:selector:1001"
        assert source.ready_tasks(limit=1).tasks[0].task_cid == target
        assert source.ready_tasks(limit=1, task_aliases=("SEL-1001",)).tasks[0].task_cid == target
        assert source.ready_tasks(limit=1, task_cids=(target,), task_prefix="SEL-1001").tasks[0].task_cid == target
        assert source.ready_tasks(limit=1, task_cids=(target,), task_prefix="OTHER-").tasks == ()
        assert source.ready_tasks(limit=1, task_aliases=("SEL-0",)).tasks == ()
        assert source.parked_ready_task_cids(limit=1, task_aliases=("SEL-0",)) == ("task:selector:0",)


def test_positive_id_alias_union_keeps_native_dependency_and_block_rules(tmp_path):
    with open_intent_repository(tmp_path / "control.duckdb") as repo:
        _seed(repo, count=3, parked_prefix=0)
        with repo._connection(write=True) as connection:
            connection.execute("INSERT INTO task_dependencies VALUES (?, ?, ?)",
                               ["task:selector:1", "task:selector:0", "requires"])
            connection.execute(
                "INSERT INTO task_blocks (block_id,task_cid,blocker_kind,blocker_id,reason,created_at,state) "
                "VALUES ('block:test','task:selector:2','review','review:test','held','','active')")
        assert [r["task_cid"] for r in repo.select_ready_tasks(
            task_cids=("task:selector:0", "task:selector:1"), task_aliases=("SEL-2",), limit=3,
        )] == ["task:selector:0"]


@pytest.mark.parametrize("value", ["task:selector:1", True, ["x"] * 1001])
def test_selection_filters_reject_unbounded_or_ambiguous_collections(tmp_path, value):
    with open_intent_repository(tmp_path / "control.duckdb") as repo:
        with pytest.raises(IntentRepositoryBoundsError):
            repo.select_ready_tasks(task_cids=value)


def test_same_status_unused_claim_still_checks_its_current_goal(tmp_path):
    with open_intent_repository(tmp_path / "control.duckdb") as repo:
        _seed(repo, count=1, parked_prefix=0)
        receipt = {"operation": "database_claim", "claim_id": "claim:unused"}
        repo.cas_task_status(task_cid="task:selector:0", expected_revision=1,
                             new_status="in_progress", receipt=receipt)
        task = repo.get_task("task:selector:0")
        _park(repo, "goal:open")
        with pytest.raises(IntentRepositoryConflictError, match="analysis_inconclusive"):
            repo.cas_task_status(task_cid=task["task_cid"], expected_revision=task["revision"],
                                 new_status="in_progress", receipt=receipt,
                                 expected_control_receipt=receipt)
        assert repo.get_task(task["task_cid"])["body"]["completion_receipt"] == receipt


def test_missing_goal_is_not_claim_authority(tmp_path):
    with open_intent_repository(tmp_path / "control.duckdb") as repo:
        _seed(repo, count=1, parked_prefix=0)
        with repo._connection(write=True) as connection:
            connection.execute("UPDATE tasks SET goal_cid = 'goal:absent'")
        assert repo.select_ready_tasks() == ()
        with pytest.raises(IntentRepositoryIntegrityError, match="owning goal"):
            repo.cas_task_status(task_cid="task:selector:0", expected_revision=1,
                                 new_status="in_progress", receipt={"operation": "database_claim"})


def test_protocol_two_forwards_the_receipt_and_real_repository_rejects_it(tmp_path, monkeypatch):
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        _seed(source._intent, count=1, parked_prefix=0)
        stored = {"operation": "database_claim", "claim_id": "claim:exact"}
        source.compare_and_set_status("task:selector:0", 1, "in_progress", stored)
        original = source._intent

        class ProtocolTwoView:
            uses_quack_transport = True

            def __getattr__(self, name):
                return getattr(original, name)

        source._intent = ProtocolTwoView()
        monkeypatch.setattr(sources, "_mutation_transport_ready", lambda: True)
        mirrors = []

        def mirror(result, record_kind, subject_ref):
            mirrors.append((result, record_kind, subject_ref))
            return result

        monkeypatch.setattr(sources, "_mirror_status_transition", mirror)
        try:
            task = original.get_task("task:selector:0")
            with pytest.raises(IntentRepositoryConflictError, match="control receipt CAS is stale"):
                source.compare_and_set_status(task["task_cid"], task["revision"], "retrying",
                    {"operation": "next"}, expected_control_receipt={"operation": "foreign"})
            assert original.get_task(task["task_cid"])["status"] == "in_progress"
            assert mirrors == []
            result = source.compare_and_set_status(task["task_cid"], task["revision"], "retrying",
                {"operation": "next"}, expected_control_receipt=stored)
            assert result.task.status == "retrying"
            assert mirrors == [(result, "status_cas_record", task["task_cid"])]
        finally:
            source._intent = original


def test_older_owner_command_also_preserves_expected_control_receipt(monkeypatch):
    source = object.__new__(DatabaseTaskSource)
    source._intent = SimpleNamespace(uses_quack_transport=True)
    monkeypatch.setattr(sources, "_mutation_transport_ready", lambda: False)
    expected = {"operation": "claim", "claim_id": "claim:one"}
    seen = []

    class Captured(Exception):
        pass

    def submit(command, payload):
        seen.append(payload)
        raise Captured

    monkeypatch.setattr(sources, "submit_quack_owner_command", submit)
    with pytest.raises(Captured):
        source.compare_and_set_status("task:one", 1, "retrying", {}, expected_control_receipt=expected)
    assert seen[0]["expected_control_receipt"] == expected


def test_stale_coordination_rows_cannot_escape_positive_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "MAX_QUERY_LIMIT", 2)
    monkeypatch.setattr(native, "TASK_SOURCE_QUERY_LIMIT", 2)
    with _open_daemon(tmp_path, session="owner:slice") as daemon:
        daemon.materialize_population(_population(4))
        # This off-slice row lies beyond the source prefix and is already in
        # coordination from another prior sync. It must not become a fallback.
        daemon.coordinator.register_task(task_cid="task:cid:003", task_id="DQP-T003")
        daemon.execution_slice_task_cids = frozenset({"task:cid:004"})
        claim = daemon.claim_next()
        assert claim is not None and claim.task_cid == "task:cid:004"
        assert daemon.task_source.get("task:cid:003").status == "ready"


def test_goal_park_race_withdraws_unused_claim_without_attempt_or_provider(tmp_path, monkeypatch):
    calls = []
    with _open_daemon(tmp_path, session="owner:race", provider_calls=calls) as daemon:
        daemon.materialize_population(_population(1))
        compare = daemon.task_source.compare_and_set_status
        captured = []

        def park_then_compare(*args, **kwargs):
            if kwargs.get("status") == "in_progress":
                captured.append(dict(kwargs["receipt"]))
                _park(daemon.task_source._intent, "goal:cid:root")
            return compare(*args, **kwargs)

        monkeypatch.setattr(daemon.task_source, "compare_and_set_status", park_then_compare)
        with pytest.raises(Exception, match="analysis_inconclusive"):
            daemon.claim_next()
        assert len(captured) == 1 and calls == []
        assert daemon.get_attempt(captured[0]["attempt_id"]) is None
        assert daemon.task_source.get("task:cid:001").status == "ready"
        lease = daemon.coordinator.get_task_claim(captured[0]["claim_id"])
        assert lease.state.value == "released"


@pytest.mark.parametrize("ineligible", ["slice", "parked", "manual"])
def test_recovered_unused_lease_is_withdrawn_when_no_longer_eligible(tmp_path, ineligible):
    with _open_daemon(tmp_path, session="owner:recovery") as daemon:
        daemon.materialize_population(_population(1))
        daemon.sync_ready_tasks_into_coordination()
        lease = daemon.coordinator.claim_ready_task(owner_session_id=daemon.owner_session_id)
        assert lease is not None and daemon.get_attempt(lease.attempt_id) is None
        if ineligible == "slice":
            daemon.execution_slice_task_cids = frozenset({"task:absent"})
        elif ineligible == "parked":
            _park(daemon.task_source._intent, "goal:cid:root")
        else:
            with daemon.task_source._intent._connection(write=True) as connection:
                connection.execute("UPDATE tasks SET body_json = '{\"review_only\":true}'")
        assert daemon.claim_next() is None
        assert daemon.coordinator.get_task_claim(lease.claim_id).state.value == "released"
        assert daemon.get_attempt(lease.attempt_id) is None


def _router_effect(daemon, *, already_recorded, effect_override=None):
    claim = daemon.claim_next()
    assert claim is not None
    current = daemon.commit_phase(claim, "context", body={"fixture": "injected router"})
    provider = {"status": "router_proposal", "accepted": False, "wrote_compiler": False,
                "imported": False, "admitted": False, "formalized": False,
                "router_called": not already_recorded, "proposal_sha256": "a" * 64,
                "proposal_keys": ["parser"],
                "reason": "proposal_already_recorded" if already_recorded else ""}
    current, _, _ = daemon.run_provider(current, provider_fn=lambda _attempt: provider)
    effect = {
        "status": "refused", "applied": False, "effect": "router_proposal_not_applied",
        "proposal_sha256": "a" * 64, "admitted": False, "formalized": False,
        "wrote_compiler": False, "imported": False,
    }
    # The router-negative branch creates its own no-effect record and never
    # invokes arbitrary effect callbacks, including in production mode.
    current, recorded, _ = daemon.run_effect(current, provider)
    assert recorded == effect
    if effect_override:
        effect.update(effect_override)
        connection = daemon._require_connection()
        connection.execute("UPDATE effect_claims SET result_json = ? WHERE attempt_id = ?",
                           [json.dumps(effect), current.attempt_id])
        connection.execute("UPDATE attempt_phases SET body_json = ? WHERE attempt_id = ? AND phase = 'effect'",
                           [json.dumps({"idempotency_key": f"effect:{current.attempt_id}", "result": effect}),
                            current.attempt_id])
    return current


@pytest.mark.parametrize("boundary", ["after_task_cas", "after_release", "after_goal_park"])
def test_router_refusal_reconciles_after_each_response_loss_boundary(tmp_path, monkeypatch, boundary):
    from ipfs_datasets_py.logic.autoformal import supervisor_loop

    calls = []
    daemon = _open_daemon(tmp_path, session="owner:router", provider_calls=calls)
    try:
        daemon.materialize_population(_population(1))
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute("UPDATE goals SET body_json = '{\"kind\":\"subgoal\"}'")
        current = _router_effect(daemon, already_recorded=True)
        target, name = {
            "after_task_cas": (daemon.task_source, "compare_and_set_status"),
            "after_release": (daemon, "_release_unadmitted_new_claim"),
            "after_goal_park": (supervisor_loop, "mark_span_subgoal_review"),
        }[boundary]
        original = getattr(target, name)

        def committed_then_lost(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected lost reply")

        with monkeypatch.context() as local:
            local.setattr(target, name, committed_then_lost)
            with pytest.raises(RuntimeError, match="injected lost reply"):
                daemon.resume_attempt(current.attempt_id)
        daemon.close()
        daemon = _open_daemon(tmp_path, session="owner:router", provider_calls=calls)
        result = daemon.resume_attempt(current.attempt_id)
        assert result["reason"] == "router_proposal_not_applied"
        assert result["requeued"] is False and result["resumed"] is False
        assert result["admitted"] is result["formalized"] is result["wrote_compiler"] is False
        assert daemon.coordinator.get_task_claim(current.claim_id).state.value == "released"
        assert daemon.task_source.get("task:cid:001").status == "blocked"
        assert daemon.task_source.get_goal("goal:cid:root")["status"] == "analysis_inconclusive"
        assert calls == []
        assert daemon.get_attempt(current.attempt_id).committed_phase != "complete"
    finally:
        daemon.close()


def test_router_reconciliation_rejects_superseding_receipt(tmp_path, monkeypatch):
    with _open_daemon(tmp_path, session="owner:foreign") as daemon:
        daemon.materialize_population(_population(1))
        current = _router_effect(daemon, already_recorded=False)
        def stop_before_release(*_args, **_kwargs):
            raise RuntimeError("injected before lease release")
        with monkeypatch.context() as local:
            local.setattr(daemon, "_release_unadmitted_new_claim", stop_before_release)
            with pytest.raises(RuntimeError, match="before lease release"):
                daemon.return_unapplied_router_proposal(current)
        task = daemon.task_source.get(current.task_cid)
        daemon.task_source.compare_and_set_status(task.task_cid, task.revision, "blocked",
                                                  {"operation": "operator_review", "admitted": False})
        with pytest.raises(native.DatabaseImplementationConflictError, match="current claim"):
            daemon.resume_attempt(current.attempt_id)
        assert daemon.task_source.get(task.task_cid).body["completion_receipt"]["operation"] == "operator_review"


@pytest.mark.parametrize("override", [{"proposal_sha256": "b" * 64}, {"imported": 0}])
def test_router_reconciliation_needs_exact_committed_refusal(tmp_path, override):
    with _open_daemon(tmp_path, session="owner:proof") as daemon:
        daemon.materialize_population(_population(1))
        current = _router_effect(daemon, already_recorded=False, effect_override=override)
        before = daemon.task_source.get(current.task_cid)
        with pytest.raises(native.DatabaseImplementationConflictError, match="exact committed refusal"):
            daemon.resume_attempt(current.attempt_id)
        after = daemon.task_source.get(current.task_cid)
        assert after.revision == before.revision and after.status == "in_progress"
        assert daemon.coordinator.get_task_claim(current.claim_id).state.value == "accepted"


def test_parked_canonical_orphan_is_requeued_without_admitting_a_new_attempt(tmp_path):
    with _open_daemon(tmp_path, session="owner:orphan") as daemon:
        daemon.materialize_population(_population(1))
        daemon.sync_ready_tasks_into_coordination()
        lease = daemon.coordinator.claim_ready_task(owner_session_id=daemon.owner_session_id)
        task = daemon.task_source.get(lease.task_cid)
        daemon.task_source.compare_and_set_status(task.task_cid, task.revision, "in_progress",
                                                  daemon._database_claim_receipt(lease))
        _park(daemon.task_source._intent, task.goal_cid)
        assert daemon.claim_next() is None
        assert daemon.coordinator.get_task_claim(lease.claim_id).state.value == "released"
        assert daemon.get_attempt(lease.attempt_id) is None
        assert daemon.claim_next() is None
        assert daemon.task_source.get(task.task_cid).status == "retrying"
        assert daemon.get_attempt(lease.attempt_id) is None


def test_previously_admitted_attempt_recovery_is_not_reclassified_as_new_claim(tmp_path):
    calls = []
    with _open_daemon(tmp_path, session="owner:admitted", provider_calls=calls) as daemon:
        daemon.materialize_population(_population(1))
        current = daemon.claim_next()
        _park(daemon.task_source._intent, "goal:cid:root")
        # An admitted execution obligation retains its existing explicit resume
        # semantics; this new-claim rule does not silently cancel it.
        result = daemon.resume_attempt(current.attempt_id)
        assert result["resumed"] is True
        assert calls == [current.task_cid]


def test_native_shard_selection_reaches_candidate_after_1000_other_lane_rows(tmp_path):
    aliases = [f"part:LANE-{i}" for i in range(6000)]
    lane = lambda alias: int(hashlib.sha256(alias.rsplit(":", 1)[-1].encode()).hexdigest()[:8], 16) % 2
    wrong = [alias for alias in aliases if lane(alias) == 1][:1001]
    desired = next(alias for alias in aliases if lane(alias) == 0)
    with _open_daemon(tmp_path, session="owner:lane", task_shard_count=2,
                      task_shard_index=0, strict_task_sharding=True) as daemon:
        _seed(daemon.task_source._intent, count=1002, parked_prefix=0)
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.executemany("UPDATE tasks SET task_alias = ? WHERE ordinal = ?",
                                   [(alias, i) for i, alias in enumerate([*wrong, desired])])
        page = daemon.task_source.ready_tasks(limit=1, task_shard_count=2, task_shard_index=0)
        assert [task.task_cid for task in page.tasks] == ["task:selector:1001"]
        registered = daemon.sync_ready_tasks_into_coordination()
        assert registered == ["task:selector:1001"]
        current = daemon.claim_next()
        assert current is not None and current.task_cid == "task:selector:1001"
        assert current.task_alias == desired


@pytest.mark.parametrize("kwargs", [
    {"task_shard_count": True}, {"task_shard_count": 0},
    {"task_shard_count": 2, "task_shard_index": 2}, {"task_shard_index": False},
])
def test_native_shard_filters_are_strict(tmp_path, kwargs):
    with open_intent_repository(tmp_path / "control.duckdb") as repo:
        with pytest.raises(IntentRepositoryBoundsError, match="selection shard"):
            repo.select_ready_tasks(**kwargs)


@pytest.mark.parametrize("tamper", ["foreign_owner", "forged_attempt", "provider_owner", "phase_fence"])
def test_released_refusal_recovery_requires_durable_current_owner_and_receipts(tmp_path, monkeypatch, tamper):
    with _open_daemon(tmp_path, session="owner:receipt") as daemon:
        daemon.materialize_population(_population(1))
        current = _router_effect(daemon, already_recorded=False)
        release = daemon._release_unadmitted_new_claim
        def release_then_lose(*args, **kwargs):
            release(*args, **kwargs)
            raise RuntimeError("injected released reply loss")
        with monkeypatch.context() as local:
            local.setattr(daemon, "_release_unadmitted_new_claim", release_then_lose)
            with pytest.raises(RuntimeError, match="released reply loss"):
                daemon.return_unapplied_router_proposal(current)
        before = daemon.task_source.get(current.task_cid)
        if tamper == "foreign_owner":
            daemon.owner_session_id = "owner:other"
        elif tamper == "forged_attempt":
            current = replace(current, claim_id="claim:forged")
        elif tamper == "provider_owner":
            daemon._require_connection().execute(
                "UPDATE provider_invocations SET owner_session_id = 'owner:other' WHERE attempt_id = ?",
                [current.attempt_id])
        else:
            daemon._require_connection().execute(
                "UPDATE attempt_phases SET fencing_token = fencing_token + 1 "
                "WHERE attempt_id = ? AND phase = 'effect'", [current.attempt_id])
        with pytest.raises((native.DatabaseImplementationAuthorityError, native.DatabaseImplementationConflictError)):
            daemon.resume_attempt(current)
        after = daemon.task_source.get(current.task_cid)
        assert after.status == before.status and after.revision == before.revision
        assert daemon.coordinator.get_task_claim(before.body["completion_receipt"]["claim_id"]).state.value == "released"


def _proposal(*, recorded=False):
    return {
        "status": "router_proposal", "accepted": False,
        "wrote_compiler": False, "imported": False, "admitted": False, "formalized": False,
        "router_called": not recorded, "proposal_sha256": "a" * 64, "proposal_keys": ["parser"],
        "reason": "proposal_already_recorded" if recorded else "",
    }


@pytest.mark.parametrize("recorded", [False, True])
def test_production_refusal_records_negative_evidence_without_effect_or_validation(tmp_path, monkeypatch, recorded):
    # Production acceptance policy is exercised with an explicitly injected
    # router response; this does not qualify a native provider or model.
    calls = []
    with _open_daemon(tmp_path, session="owner:production-refusal") as daemon:
        daemon.require_real_execution = True
        daemon.materialize_population(_population(1))
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute("UPDATE goals SET body_json = '{\"kind\":\"subgoal\"}'")
        current = daemon.claim_next()

        def provider(_attempt):
            calls.append("provider")
            return _proposal(recorded=recorded)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("negative evidence must not execute an effect or validation")

        daemon._effect_fn = forbidden
        result = daemon.resume_attempt(current, provider_fn=provider, effect_fn=forbidden, validation_fn=forbidden)
        assert calls == ["provider"]
        assert result["reason"] == "router_proposal_not_applied" and result["resumed"] is False
        assert result["admitted"] is result["formalized"] is result["wrote_compiler"] is False
        assert result["task_status"] == ("blocked" if recorded else "ready")
        stored = daemon.get_attempt(current.attempt_id)
        assert stored.committed_phase == "blocked" and stored.status == "blocked"
        assert {row["phase"] for row in daemon.phase_history(current.attempt_id)} == {"claimed", "context", "provider", "effect", "blocked"}
        assert daemon.coordinator.get_task_claim(current.claim_id).state.value == "released"
        again = daemon.resume_attempt(current.attempt_id, provider_fn=forbidden, effect_fn=forbidden, validation_fn=forbidden)
        assert again["resumed"] is False and again["reason"] == "attempt_blocked" and calls == ["provider"]


@pytest.mark.parametrize("change", [
    {"status": "noop"}, {"accepted": 0}, {"wrote_compiler": True},
    {"proposal_sha256": "not-a-digest"}, {"applied": True},
])
def test_production_refusal_exception_does_not_accept_generic_or_malformed_results(tmp_path, change):
    with _open_daemon(tmp_path, session="owner:reject") as daemon:
        daemon.require_real_execution = True
        daemon.materialize_population(_population(1))
        current = daemon.claim_next()
        current = daemon.commit_phase(current, "context", body={"injected": True})
        proposal = {**_proposal(), **change}
        with pytest.raises(native.DatabaseImplementationAuthorityError, match="not accepted real-execution evidence"):
            daemon.run_provider(current, provider_fn=lambda _attempt: proposal)
        assert daemon.provider_invocation_recorded(current.attempt_id, idempotency_key=f"provider:{current.attempt_id}") is None
        assert daemon.get_attempt(current.attempt_id).committed_phase == "context"


@pytest.mark.parametrize("boundary", ["effect_callback", "validate", "complete"])
def test_negative_provider_cannot_enter_generic_success_paths(tmp_path, boundary):
    with _open_daemon(tmp_path, session="owner:closed-negative") as daemon:
        daemon.require_real_execution = True
        daemon.materialize_population(_population(1))
        current = daemon.claim_next()
        current = daemon.commit_phase(current, "context", body={"injected": True})
        proposal = _proposal()
        current, _, _ = daemon.run_provider(current, provider_fn=lambda _attempt: proposal)
        if boundary == "effect_callback":
            with pytest.raises(native.DatabaseImplementationAuthorityError, match="effect callback"):
                daemon.run_effect(current, proposal, effect_fn=lambda *_args: {"status": "applied"})
        else:
            current, _, _ = daemon.run_effect(current, proposal)
            with pytest.raises(native.DatabaseImplementationAuthorityError, match="cannot validate or complete"):
                if boundary == "validate":
                    daemon.commit_phase(current, "validation", body={"outcome": "passed", "evidence_digest": "sha256:" + "b" * 64})
                else:
                    daemon.complete_attempt(current, validation_result={"outcome": "passed", "evidence_digest": "sha256:" + "b" * 64})
        assert daemon.get_attempt(current.attempt_id).committed_phase in {"provider", "effect"}
        assert daemon.coordinator.get_task_claim(current.claim_id).state.value == "accepted"
        assert daemon.task_source.get(current.task_cid).status == "in_progress"


def test_automatic_selection_reaches_candidate_beyond_1000_manual_rows(tmp_path):
    with _open_daemon(tmp_path, session="owner:automatic") as daemon:
        _seed(daemon.task_source._intent, count=1002, parked_prefix=0)
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute("UPDATE tasks SET body_json = '{\"completion\":\"manual\"}' WHERE ordinal < 1001")
        assert daemon.task_source.ready_tasks(limit=1).tasks[0].task_cid == "task:selector:0"
        assert daemon.task_source.ready_tasks(limit=1, automatic_only=True).tasks[0].task_cid == "task:selector:1001"
        assert daemon.sync_ready_tasks_into_coordination() == ["task:selector:1001"]
        current = daemon.claim_next()
        assert current is not None and current.task_cid == "task:selector:1001"


@pytest.mark.parametrize("body,forbidden", [
    ({"completion": "manual"}, True), ({"completion": {"mode": "manual"}}, True),
    ({"completion": {"kind": "manual"}}, True), ({"review_only": True}, True),
    ({"review only": "yes"}, True), ({"is_schedulable": False}, True),
    ({"is schedulable": "off"}, True), ({"completion": "auto", "review_only": False}, False),
    ({"review_only": False, "review only": True}, False),
    ({"is_schedulable": True, "is schedulable": False}, False),
])
def test_automatic_selection_preserves_existing_body_exclusion_semantics(tmp_path, body, forbidden):
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        _seed(source._intent, count=1, parked_prefix=0)
        with source._intent._connection(write=True) as connection:
            connection.execute("UPDATE tasks SET body_json = ?", [json.dumps(body)])
        task = source.get("task:selector:0")
        assert native.DatabaseImplementationDaemon._automatic_claim_forbidden(task) is forbidden
        assert bool(source.ready_tasks(limit=1, automatic_only=True).tasks) is (not forbidden)
        assert len(source.ready_tasks(limit=1).tasks) == 1
        with pytest.raises(IntentRepositoryBoundsError, match="automatic_only"):
            source.ready_tasks(automatic_only=1)


def test_negative_provider_requires_canonical_idempotency_key_before_recording(tmp_path):
    with _open_daemon(tmp_path, session="owner:negative-key") as daemon:
        daemon.require_real_execution = True
        daemon.materialize_population(_population(1))
        current = daemon.claim_next()
        current = daemon.commit_phase(current, "context", body={"injected": True})
        with pytest.raises(native.DatabaseImplementationAuthorityError, match="canonical provider key"):
            daemon.run_provider(current, idempotency_key="alternate",
                                provider_fn=lambda _attempt: _proposal())
        assert daemon.provider_invocation_recorded(current.attempt_id, idempotency_key="alternate") is None
        assert daemon.get_attempt(current.attempt_id).committed_phase == "context"


def test_released_refusal_is_reconciled_before_generic_terminal_claim_recovery(tmp_path, monkeypatch):
    with _open_daemon(tmp_path, session="owner:refusal-sweep") as daemon:
        daemon.require_real_execution = True
        daemon.materialize_population(_population(1))
        current = _router_effect(daemon, already_recorded=False)
        release = daemon._release_unadmitted_new_claim
        def release_then_lose(*args, **kwargs):
            release(*args, **kwargs)
            raise RuntimeError("injected released reply loss")
        with monkeypatch.context() as local:
            local.setattr(daemon, "_release_unadmitted_new_claim", release_then_lose)
            with pytest.raises(RuntimeError, match="released reply loss"):
                daemon.resume_attempt(current)
        assert daemon.get_attempt(current.attempt_id).status == "running"
        outcomes = daemon.reconcile_expired_running_attempts()
        assert len(outcomes) == 1 and outcomes[0]["reason"] == "router_proposal_not_applied"
        assert daemon.get_attempt(current.attempt_id).status == "blocked"
        assert daemon.list_running_attempts() == []
        assert daemon.reconcile_expired_running_attempts() == []
        assert daemon.task_source.get(current.task_cid).status == "ready"
        assert daemon.coordinator.get_task_claim(current.claim_id).state.value == "released"


@pytest.mark.parametrize("factory_name", [
    "build_portal_implementation_daemon_from_args",
    "build_database_implementation_daemon_from_args",
])
def test_database_factories_forward_bounded_execution_slice(tmp_path, monkeypatch, factory_name):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon_runner as runner

    captured = {}
    sentinel = object()

    def construct(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(native, "DatabaseImplementationDaemon", construct)
    monkeypatch.setattr(runner, "bind_database_portal_execution_from_args", lambda *args, **kwargs: None)
    parsed = SimpleNamespace(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        state_dir=tmp_path / "state", state_prefix="slice-test",
        authority_mode="embedded", task_source_kind="duckdb",
        execution_slice_task_id=["TASK-selected"],
        execution_slice_task_cid=["task:selected"],
        implement=False,
    )
    factory = getattr(runner, factory_name)
    if factory_name == "build_portal_implementation_daemon_from_args":
        daemon, _ = factory(parsed, repo_root=tmp_path)
    else:
        daemon = factory(parsed)
    assert daemon is sentinel
    assert captured["execution_slice_task_ids"] == ("TASK-selected",)
    assert captured["execution_slice_task_cids"] == ("task:selected",)
