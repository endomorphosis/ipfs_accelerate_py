"""Real private stores: unknown SAWM-016 cannot starve independent SAWM-023."""

import hashlib
import json

import pytest
from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.runtime import native_dispatch_drain as drain
from ipfs_accelerate_py.agent_supervisor.todo_daemon import expired_attempt_custody
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population


def snapshot(daemon, attempt, artifact):
    value = {
        "task": daemon.task_source.get(attempt.task_cid).to_dict(),
        "attempt": daemon.get_attempt(attempt.attempt_id).to_dict(),
        "phases": daemon.phase_history(attempt.attempt_id),
        "claim": daemon.coordinator.get_task_claim(attempt.claim_id).to_dict(),
        "lease": daemon.coordinator.get_lease(attempt.lease_id).to_dict(),
        "coordination_attempt": daemon.coordinator.get_task_attempt(attempt.attempt_id).to_dict(),
        "counts": daemon._attempt_execution_evidence_counts(attempt.attempt_id),
        "artifact": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "artifact_stat": list(artifact.stat()),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def setup(tmp_path, *, expire=True):
    now = {"ms": 1000}
    calls = []
    old_file = tmp_path / "retained-016.py"
    old_file.write_bytes(b"# Opaque historical candidate; unknown callback outcome.\n")
    new_file = tmp_path / "independent-023.py"

    def provider(attempt):
        assert attempt.task_alias == "SAWM-023"
        calls.append(attempt.task_cid)
        new_file.write_bytes(b"VALUE = 23\n")
        return {"status": "ok", "accepted": True}

    daemon = _open_daemon(
        tmp_path, session="session:sawm-fairness", provider_fn=provider,
        lease_ms=5000, clock_ms=lambda: now["ms"], task_shard_count=4,
        task_shard_index=3, strict_task_sharding=True, task_prefix="SAWM",
    )
    daemon.require_real_execution = True

    def validate(_attempt, _result):
        body = new_file.read_bytes()
        compile(body, str(new_file), "exec")
        assert body == b"VALUE = 23\n"
        return {"outcome": "passed", "evidence_digest": "sha256:" + hashlib.sha256(body).hexdigest()}

    daemon._validation_fn = validate
    population = _population(3)
    for task, alias in zip(population["tasks"], ("SAWM-016", "SAWM-023", "SAWM-020")):
        task["task_id"] = alias
    population["tasks"][2]["depends_on"] = ["SAWM-016"]
    daemon.materialize_population(population)
    attempt = daemon.claim_next()
    assert attempt.task_alias == "SAWM-016"
    attempt = daemon.commit_phase(attempt, "context")
    now["ms"] = 7000
    if expire:
        daemon.coordinator.expire_task_claim(
            daemon.coordinator.get_task_claim(attempt.claim_id), now_ms=now["ms"],
        )
    return daemon, attempt, now, calls, old_file


def test_independent_same_shard_completes_without_touching_unknown_attempt(tmp_path, monkeypatch):
    daemon, attempt, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    observed = []
    original = expired_attempt_custody.guard_generic_retirement

    def record(*args, **kwargs):
        observed.append(args[1].attempt_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(expired_attempt_custody, "guard_generic_retirement", record)
    try:
        result = daemon.run_once()
        assert result["active_task_id"] == "SAWM-023"
        assert result["implementation_result"]["status"] == "succeeded"
        assert calls == ["task:cid:002"]
        assert result["retained_attempts"][0]["retry_authorized"] is False
        assert result["retained_attempts"][0]["attempt_consumed"] == "unknown"
        assert snapshot(daemon, attempt, artifact) == before
        again = daemon.run_once()
        assert again["implementation_result"] is None
        assert again["retained_attempt_settlement_required"] is True
        assert observed == [attempt.attempt_id, attempt.attempt_id]
        assert calls == ["task:cid:002"]
        assert snapshot(daemon, attempt, artifact) == before
        assert daemon.task_source.get("task:cid:003").status == "ready"
        assert "task:cid:003" not in {t.task_cid for t in daemon.task_source.ready_tasks().tasks}
    finally:
        daemon.close()


def test_elapsed_but_unexpired_claim_is_not_swept_to_enable_fairness(tmp_path):
    daemon, attempt, _, calls, artifact = setup(tmp_path, expire=False)
    before = snapshot(daemon, attempt, artifact)
    try:
        result = daemon.run_once()
        assert result["reason"] == "production_attempt_settlement_required"
        assert calls == []
        assert snapshot(daemon, attempt, artifact) == before
    finally:
        daemon.close()


@pytest.mark.parametrize("permitted", [True, False])
def test_native_pause_boundary_reports_old_custody_and_never_preclaim(tmp_path, permitted):
    daemon, attempt, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    packets = []
    client = drain.NativeDispatchClient.__new__(drain.NativeDispatchClient)

    def exchange(operation, body):
        packets.append((operation, body))
        return {"new_dispatch_permitted": permitted, "reason": "fixture_pause"}

    client.exchange = exchange
    daemon._native_dispatch_control = client
    try:
        result = daemon.run_once()
        assert bool(calls) is permitted
        assert snapshot(daemon, attempt, artifact) == before
        assert not any(body.get("phase") == "preclaim" for _, body in packets)
        observations = [body["observation"] for op, body in packets if op == "custody_boundary"]
        assert len(observations) >= 3  # reconciliation, claim gate, final custody
        assert all(item["attempt"]["attempt_id"] == attempt.attempt_id for item in observations)
        assert all(item["callback_outcome"] == "unknown" for item in observations)
        assert result["retained_attempts"]
    finally:
        daemon.close()


def test_unavailable_dispatch_observation_does_not_starve_independent_ready_task(tmp_path):
    daemon, attempt, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    client = drain.NativeDispatchClient.__new__(drain.NativeDispatchClient)
    client.before_independent_claim = lambda *_: {
        "new_dispatch_permitted": False,
        "reason": "retained_attempt_dispatch_observation_unavailable",
    }
    client.before_claim = lambda: {
        "new_dispatch_permitted": False,
        "reason": "native_dispatch_observation_unavailable",
    }
    daemon._native_dispatch_control = client
    try:
        result = daemon.run_once()
        assert result["active_task_id"] == "SAWM-023"
        assert result["implementation_result"]["status"] == "succeeded"
        assert calls == ["task:cid:002"]
        assert snapshot(daemon, attempt, artifact) == before
        assert result["retained_attempts"][0]["attempt_consumed"] == "unknown"
        assert daemon._last_native_dispatch_boundary["reason"] == (
            "retained_attempt_observation_unverified_independent_claim"
        )
    finally:
        daemon.close()


@pytest.mark.parametrize("damage", ["missing_native_method", "unreadable_custody"])
def test_missing_peer_observation_falls_back_to_ordinary_permitted_claim(tmp_path, monkeypatch, damage):
    daemon, attempt, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    client = drain.NativeDispatchClient.__new__(drain.NativeDispatchClient)
    client.exchange = lambda *_: {"new_dispatch_permitted": True, "reason": "fixture_pause"}
    daemon._native_dispatch_control = client
    if damage == "missing_native_method":
        client.before_independent_claim = None
    else:
        from ipfs_accelerate_py.agent_supervisor.runtime import attempt_custody_observation as obs
        def unavailable(*_):
            raise obs.AttemptObservationUnavailable()
        monkeypatch.setattr(obs, "observe_attempt", unavailable)
    try:
        result = daemon.run_once()
        assert result["active_task_id"] == "SAWM-023"
        assert calls == ["task:cid:002"]
        assert snapshot(daemon, attempt, artifact) == before
    finally:
        daemon.close()


@pytest.mark.parametrize("damage", ["missing_claim", "changed_claim", "changed_task_receipt", "boolean_claim_receipt", "oversized_snapshot"])
def test_unverified_retention_never_opens_new_provider(tmp_path, monkeypatch, damage):
    daemon, attempt, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    try:
        if damage in {"missing_claim", "changed_claim"}:
            get = daemon.coordinator.get_task_claim
            monkeypatch.setattr(daemon.coordinator, "get_task_claim", lambda claim_id:
                (None if damage == "missing_claim" else replace(get(claim_id), revision=get(claim_id).revision + 1))
                if claim_id == attempt.claim_id else get(claim_id))
            if damage == "changed_claim":
                # Change between the exact retention snapshot and new selection.
                reads = []
                def drifting(claim_id):
                    value = get(claim_id)
                    reads.append(claim_id)
                    return replace(value, revision=value.revision + 1) if len(reads) >= 3 else value
                monkeypatch.setattr(daemon.coordinator, "get_task_claim", drifting)
        elif damage in {"changed_task_receipt", "boolean_claim_receipt", "oversized_snapshot"}:
            get = daemon.task_source.get
            def damaged_task(cid):
                value = get(cid)
                if cid != attempt.task_cid:
                    return value
                body = dict(value.body)
                if damage == "oversized_snapshot":
                    body["opaque_fixture"] = "x" * (1024 * 1024)
                elif damage == "boolean_claim_receipt":
                    body["completion_receipt"] = {**body["completion_receipt"], "fencing_token": True}
                else:
                    body["completion_receipt"] = {}
                return replace(value, body=body)
            monkeypatch.setattr(daemon.task_source, "get", damaged_task)
        result = daemon.run_once()
        assert result.get("implementation_result") is None
        assert calls == []
        monkeypatch.undo()
        assert snapshot(daemon, attempt, artifact) == before
    finally:
        daemon.close()


def test_direct_reconciliation_still_refuses_and_two_retained_obligations_are_bounded(tmp_path):
    daemon, attempt, now, calls, artifact = setup(tmp_path)
    try:
        with pytest.raises(expired_attempt_custody.ExpiredExecutionCustodyPending):
            daemon.reconcile_expired_running_attempts()
        second = daemon.commit_phase(daemon.claim_next(exclude_task_cids=(attempt.task_cid,)), "context")
        now["ms"] = 14000
        daemon.coordinator.expire_task_claim(daemon.coordinator.get_task_claim(second.claim_id), now_ms=now["ms"])
        before = snapshot(daemon, attempt, artifact)
        second_before = snapshot(daemon, second, artifact)
        result = daemon.run_once()
        assert result["reason"] == "production_attempt_settlement_required"
        assert calls == []
        assert snapshot(daemon, attempt, artifact) == before
        assert snapshot(daemon, second, artifact) == second_before
    finally:
        daemon.close()


def test_reconciliation_rejects_an_arbitrary_retention_callback(tmp_path):
    daemon, attempt, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    from types import SimpleNamespace
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationAuthorityError
    try:
        with pytest.raises(DatabaseImplementationAuthorityError, match="invalid retained-attempt"):
            daemon.reconcile_expired_running_attempts(retained=SimpleNamespace(retain=lambda *_: True))
        assert calls == []
        assert snapshot(daemon, attempt, artifact) == before
    finally:
        daemon.close()


def test_normal_exact_claim_guard_remains_required(tmp_path, monkeypatch):
    daemon, attempt, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    checked = []
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import DatabaseCoordinationStaleFenceError
    original = daemon._protect_new_claim
    def stale_claim(claim):
        checked.append(claim.task_cid)
        # The ordinary database guard must compare this incoming claim with
        # the actual stored lease/attempt, even after fairness selected 023.
        return original(replace(claim, fencing_token=claim.fencing_token + 1))
    monkeypatch.setattr(daemon, "_protect_new_claim", stale_claim)
    try:
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            daemon.run_once()
        assert checked == ["task:cid:002"]
        assert calls == []
        assert snapshot(daemon, attempt, artifact) == before
    finally:
        daemon.close()


def test_operator_stop_rearms_blocked_peer_without_settlement_phase(tmp_path):
    from types import SimpleNamespace
    daemon, old, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, old, artifact)
    peer = daemon.commit_phase(
        daemon.claim_next(exclude_task_cids=(old.task_cid,)), "context"
    )
    assert peer.task_alias == "SAWM-023"
    state = tmp_path / "portal-task-state.json"
    state.write_text(json.dumps({"last_implementation_returncode": -15}))
    daemon._provider_fn = SimpleNamespace(
        __self__=SimpleNamespace(_paths=lambda _attempt: SimpleNamespace(state=state))
    )
    task = daemon.task_source.get(peer.task_cid)
    daemon._cas_task_status_database(
        peer.task_cid,
        expected_revision=int(task.revision),
        new_status="blocked",
        receipt={
            "operation": "database_claim",
            "attempt_id": peer.attempt_id,
            "claim_id": peer.claim_id,
        },
    )
    assert daemon.task_source.get(peer.task_cid).status == "blocked"
    try:
        rearms = daemon.reconcile_recoverable_portal_failure_rearms()
        assert rearms
        assert rearms[0]["reason"] == "operator_session_stop_unsettled_independent_attempt"
        assert rearms[0]["task_alias"] == "SAWM-023"
        assert daemon.task_source.get(peer.task_cid).status == "retrying"
        assert snapshot(daemon, old, artifact) == before
        assert calls == []
    finally:
        daemon.close()


def test_operator_stop_rearms_from_projection_when_execution_attempt_missing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    daemon, old, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, old, artifact)
    peer = daemon.commit_phase(
        daemon.claim_next(exclude_task_cids=(old.task_cid,)), "context"
    )
    attempts = tmp_path / "attempts" / "deadbeef"
    attempts.mkdir(parents=True)
    (attempts / "portal-task-state.json").write_text(
        json.dumps(
            {
                "last_implementation_returncode": -15,
                "last_implementation_task_id": "SAWM-023",
            }
        )
    )
    daemon._provider_fn = SimpleNamespace(
        __self__=SimpleNamespace(
            _paths=lambda _attempt: SimpleNamespace(state=attempts / "missing.json"),
            attempt_root=tmp_path / "attempts",
        )
    )
    original = daemon.get_attempt

    def hidden(attempt_id):
        if attempt_id == peer.attempt_id:
            return None
        return original(attempt_id)

    monkeypatch.setattr(daemon, "get_attempt", hidden)
    task = daemon.task_source.get(peer.task_cid)
    daemon._cas_task_status_database(
        peer.task_cid,
        expected_revision=int(task.revision),
        new_status="blocked",
        receipt={"operation": "database_claim", "attempt_id": peer.attempt_id},
    )
    try:
        rearms = daemon.reconcile_recoverable_portal_failure_rearms()
        assert rearms
        assert rearms[0]["reason"] == "operator_session_stop_unsettled_independent_attempt"
        assert daemon.task_source.get(peer.task_cid).status == "retrying"
        assert snapshot(daemon, old, artifact) == before
        assert calls == []
    finally:
        daemon.close()


def test_operator_stop_projection_reads_sigterm_returncode(tmp_path):
    from types import SimpleNamespace
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    state = tmp_path / "portal-task-state.json"
    state.write_text(json.dumps({"last_implementation_returncode": -15}))
    daemon = DatabaseImplementationDaemon.__new__(DatabaseImplementationDaemon)
    daemon._provider_fn = SimpleNamespace(
        __self__=SimpleNamespace(_paths=lambda _attempt: SimpleNamespace(state=state))
    )
    assert daemon._attempt_operator_stop_projection(SimpleNamespace()) is True
    state.write_text(json.dumps({"last_implementation_returncode": 1}))
    assert daemon._attempt_operator_stop_projection(SimpleNamespace()) is False


def test_operator_session_stop_does_not_block_independent_peer(tmp_path, monkeypatch):
    daemon, attempt, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        DatabasePortalBridgeError,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        operator_session_stop_failure,
    )

    assert operator_session_stop_failure(
        DatabasePortalBridgeError(
            "IO Error: Failed to send message: IO Error: Could not connect to server "
            "error for HTTP POST to 'http://127.0.0.1:24070/quack'"
        )
    )
    peer = daemon.commit_phase(
        daemon.claim_next(exclude_task_cids=(attempt.task_cid,)), "context"
    )
    assert peer.task_alias == "SAWM-023"

    def boom(_current):
        raise DatabasePortalBridgeError(
            "IO Error: Failed to send message: IO Error: Could not connect to server "
            "error for HTTP POST to 'http://127.0.0.1:24070/quack'"
        )

    monkeypatch.setattr(daemon, "resume_attempt", boom)
    try:
        result = daemon._resume_attempt_without_process_crash(peer)
        assert result["status"] == "running"
        assert result["reason"] == "operator_session_stop_unsettled_independent_attempt"
        assert daemon.task_source.get(peer.task_cid).status != "blocked"
        assert snapshot(daemon, attempt, artifact) == before
        assert calls == []
    finally:
        daemon.close()


def test_retention_read_allows_a_replaced_daemon_session(tmp_path):
    daemon, attempt, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_attempt_fairness import RetainedAttemptFairness
    original = daemon.owner_session_id
    daemon.owner_session_id = "session:replaced-owner"
    try:
        retained = RetainedAttemptFairness(daemon)
        assert retained._read(attempt) is not None
        assert snapshot(daemon, attempt, artifact) == before
        assert calls == []
    finally:
        daemon.owner_session_id = original
        daemon.close()


def test_restart_reconstructs_exclusion_from_durable_authorities(tmp_path):
    daemon, attempt, now, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    provider, validator = daemon._provider_fn, daemon._validation_fn
    daemon.close()
    reopened = _open_daemon(
        tmp_path, session="session:sawm-fairness", provider_fn=provider,
        lease_ms=5000, clock_ms=lambda: now["ms"], task_shard_count=4,
        task_shard_index=3, strict_task_sharding=True, task_prefix="SAWM",
    )
    reopened.require_real_execution = True
    reopened._validation_fn = validator
    try:
        assert reopened.run_once()["active_task_id"] == "SAWM-023"
        assert calls == ["task:cid:002"]
        assert snapshot(reopened, attempt, artifact) == before
    finally:
        reopened.close()


def test_retained_claim_pause_uses_real_owner_peer_and_preserves_unknown(tmp_path):
    from test.api.test_sawm_native_dispatch_drain import _peer_native, _lane, _public
    from ipfs_accelerate_py.agent_supervisor.runtime.attempt_custody_observation import observe_attempt

    owner_root, task_root = tmp_path / "owner", tmp_path / "tasks"
    owner_root.mkdir()
    task_root.mkdir()
    daemon, attempt, _, _, _ = setup(task_root)
    try:
        value = observe_attempt(daemon, attempt)
        with _peer_native(owner_root) as native:
            assert _lane(native, {"observation": value})["new_dispatch_permitted"] is True
            _public(native, "request")
            native.roster()
            assert _lane(native, {"observation": value})["new_dispatch_permitted"] is False
            state = _public(native)["state"]
            assert state["dispatch_pause_observed"] is False
            assert state["callback_custody_known"] is False
            assert state["terminal_custody"] == "unknown"
            assert state["lanes"][0]["attempt_observation"] == value
            assert state["lanes"][0]["pause_epoch_acknowledged"] is False
            native.owner_pipe.send("state")
            assert native.owner_pipe.recv() == [(17,)]
    finally:
        daemon.close()
