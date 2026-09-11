"""Actual native Doctor effect scope and unknown-outcome custody."""
from pathlib import Path
import hashlib
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.doctor_worktree_adapter import DoctorExactEdit
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import DatabaseCoordinationError
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_doctor_callback import (
    NativeDoctorCallback, DoctorCallbackDenied, PROFILE_KEY, CLOSED,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.doctor_interruption_custody import (
    DoctorInterruptionCustody,
)
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population


def setup(tmp_path, *, callback_budget=3):
    repo = tmp_path / "repository"
    repo.mkdir()
    for args in [("init", "-b", "main"), ("config", "user.name", "Test"), ("config", "user.email", "test@example.invalid")]:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    body = b"VALUE = 'baseline'\n"
    (repo / "value.py").write_bytes(body)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=repo, check=True, capture_output=True)
    callback = NativeDoctorCallback(
        repository_root=repo, state_root=tmp_path / "doctor",
        edits=[DoctorExactEdit("value.py", "sha256:" + hashlib.sha256(body).hexdigest(), b"VALUE = 'candidate'\n")],
        max_attempts=callback_budget,
    )
    now = {"ms": 1000}
    daemon = _open_daemon(tmp_path, provider_fn=callback, lease_ms=5000, clock_ms=lambda: now["ms"])
    daemon.require_real_execution = True
    daemon.materialize_population(_population(1))
    attempt = daemon.commit_phase(daemon.claim_next(), "context")
    return daemon, attempt, callback, now


def test_real_declared_callback_preserves_exact_candidate_after_outer_lease_loss(tmp_path, monkeypatch):
    daemon, attempt, callback, now = setup(tmp_path)
    original_record = daemon._record_event

    def lose_lease(event_type, **kwargs):
        result = original_record(event_type, **kwargs)
        if event_type == CLOSED:
            now["ms"] = 7000
        return result

    monkeypatch.setattr(daemon, "_record_event", lose_lease)
    try:
        with pytest.raises(DatabaseCoordinationError):
            daemon.run_provider(attempt)
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict()
        with DoctorInterruptionCustody(daemon, attempt, callback) as custody:
            observed = custody.require_current()
            assert observed["callback_outcome"] == "unknown"
            assert observed["settlement_authority"] is False
            workspace = Path(custody.started["worktree_path"])
            assert (workspace / "value.py").read_text() == "VALUE = 'candidate'\n"
        assert (callback.adapter.repository_root / "value.py").read_text() == "VALUE = 'baseline'\n"
    finally:
        daemon.close()


def test_recovery_profile_cannot_be_attached_after_claim(tmp_path):
    daemon, attempt, callback, _ = setup(tmp_path)
    try:
        from dataclasses import replace
        retrospective = replace(attempt, body={key:value for key,value in attempt.body.items() if key != PROFILE_KEY})
        with pytest.raises(DoctorCallbackDenied, match="retrospectively"):
            daemon.run_provider(retrospective)
    finally:
        daemon.close()


def test_declared_profile_rejects_arbitrary_callback_override(tmp_path):
    daemon, attempt, _, _ = setup(tmp_path)
    try:
        with pytest.raises(DoctorCallbackDenied, match="another callback"):
            daemon.run_provider(attempt, provider_fn=lambda _: pytest.fail("override ran"))
    finally:
        daemon.close()


def test_separate_native_reservation_and_continuation_preserve_unknown_history(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.unresolved_interruption import (
        reserve, admit_continuation,
    )
    daemon, attempt, callback, now = setup(tmp_path)
    original_record = daemon._record_event

    def lose_lease(event_type, **kwargs):
        result = original_record(event_type, **kwargs)
        if event_type == CLOSED:
            now["ms"] = 7000
        return result

    monkeypatch.setattr(daemon, "_record_event", lose_lease)
    try:
        with pytest.raises(DatabaseCoordinationError):
            daemon.run_provider(attempt)
        before = daemon.get_attempt(attempt.attempt_id).to_dict()
        phases = daemon.phase_history(attempt.attempt_id)
        reserved = reserve(daemon, attempt, callback)
        assert reserve(daemon, attempt, callback) == reserved
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == before
        assert daemon.phase_history(attempt.attempt_id) == phases
        admitted = admit_continuation(daemon, attempt, callback)
        assert admitted["callback_outcome"] == "unknown"
        assert admitted["settlement_authority"] is False
        assert admit_continuation(daemon, attempt, callback) == admitted
        assert daemon.task_source.get(attempt.task_cid).status == "retrying"
        assert daemon.list_running_attempts() == []
        next_attempt = daemon.claim_next()
        assert next_attempt.attempt_number == attempt.attempt_number + 1
        assert next_attempt.fencing_token > attempt.fencing_token
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == before
        assert daemon.phase_history(attempt.attempt_id) == phases
        assert daemon.list_running_attempts() == [next_attempt]
        with pytest.raises(DatabaseCoordinationError):
            daemon.run_provider(attempt)
    finally:
        daemon.close()


def make_unknown(daemon, attempt, now, monkeypatch):
    original_record = daemon._record_event

    def close_then_lose(event_type, **kwargs):
        result = original_record(event_type, **kwargs)
        if event_type == CLOSED:
            now["ms"] = 7000
        return result

    monkeypatch.setattr(daemon, "_record_event", close_then_lose)
    with pytest.raises(DatabaseCoordinationError):
        daemon.run_provider(attempt)


@pytest.mark.parametrize("boundary", ["reserve_record", "admit_record"])
def test_response_loss_after_native_cas_resumes_without_replaying_callback(tmp_path, monkeypatch, boundary):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import unresolved_interruption as recovery
    daemon, attempt, callback, now = setup(tmp_path)
    try:
        make_unknown(daemon, attempt, now, monkeypatch)
        real_record = recovery._record
        failing_kind = recovery.RESERVED if boundary == "reserve_record" else recovery.ADMITTED
        fired = []

        def fail_once(daemon, kind, attempt, body):
            if kind == failing_kind and not fired:
                fired.append(kind)
                raise OSError("audit unavailable after committed CAS")
            return real_record(daemon, kind, attempt, body)

        monkeypatch.setattr(recovery, "_record", fail_once)
        if boundary == "reserve_record":
            with pytest.raises(OSError):
                recovery.reserve(daemon, attempt, callback)
            assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        else:
            recovery.reserve(daemon, attempt, callback)
            with pytest.raises(OSError):
                recovery.admit_continuation(daemon, attempt, callback)
            assert daemon.task_source.get(attempt.task_cid).status == "retrying"
            assert daemon.list_running_attempts() == [attempt]
        if boundary == "reserve_record":
            recovery.reserve(daemon, attempt, callback)
        recovery.admit_continuation(daemon, attempt, callback)
        assert daemon.list_running_attempts() == []
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict()
    finally:
        daemon.close()


@pytest.mark.parametrize("damage", ["candidate", "checkpoint", "source", "closed_observation"])
def test_changed_native_recovery_inputs_refuse_without_task_mutation(tmp_path, monkeypatch, damage):
    import json
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.unresolved_interruption import reserve
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.doctor_interruption_custody import observations
    daemon, attempt, callback, now = setup(tmp_path)
    try:
        make_unknown(daemon, attempt, now, monkeypatch)
        started, _ = observations(daemon, attempt)
        if damage == "candidate":
            (Path(started["worktree_path"]) / "value.py").write_text("different candidate")
        elif damage == "checkpoint":
            (Path(started["session_dir"]) / "checkpoint" / "00000000.blob").write_text("different baseline")
        elif damage == "source":
            subprocess.run(["git", "commit", "--allow-empty", "-m", "new source"], cwd=callback.adapter.repository_root, check=True, capture_output=True)
        else:
            connection = daemon._require_connection()
            row = connection.execute("SELECT body_json FROM daemon_execution_events WHERE event_type=?", [CLOSED]).fetchone()
            changed = json.loads(row[0]); changed["observed_candidate"]["tree_cid"] = "forged"
            connection.execute("UPDATE daemon_execution_events SET body_json=? WHERE event_type=?", [json.dumps(changed), CLOSED])
        task_before = daemon.task_source.get(attempt.task_cid).to_dict()
        with pytest.raises(Exception):
            reserve(daemon, attempt, callback)
        assert daemon.task_source.get(attempt.task_cid).to_dict() == task_before
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict()
    finally:
        daemon.close()


def test_missing_positive_callback_closure_stays_unknown(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.unresolved_interruption import reserve
    daemon, attempt, callback, now = setup(tmp_path)
    original_record = daemon._record_event

    def lose_closed_observation(event_type, **kwargs):
        if event_type == CLOSED:
            now["ms"] = 7000
            raise OSError("closure evidence not persisted")
        return original_record(event_type, **kwargs)

    monkeypatch.setattr(daemon, "_record_event", lose_closed_observation)
    try:
        with pytest.raises((DatabaseCoordinationError, OSError)):
            daemon.run_provider(attempt)
        before = daemon.task_source.get(attempt.task_cid).to_dict()
        with pytest.raises(DoctorCallbackDenied, match="started and closed"):
            reserve(daemon, attempt, callback)
        assert daemon.task_source.get(attempt.task_cid).to_dict() == before
        assert daemon.run_once()["retry_authorized"] is False
    finally:
        daemon.close()


def test_native_supervisor_reconciles_only_declared_profile_and_runs_distinct_callback(tmp_path, monkeypatch):
    daemon, attempt, callback, now = setup(tmp_path)
    try:
        make_unknown(daemon, attempt, now, monkeypatch)
        before = daemon.get_attempt(attempt.attempt_id).to_dict()
        # Observe real native selection without the unrelated fixture's fake
        # default effect/validation callbacks granting task completion.
        selected = []
        def stop_after_native_claim(next_attempt):
            selected.append(next_attempt)
            return {"status": "selected_for_test"}
        monkeypatch.setattr(daemon, "_resume_attempt_without_process_crash", stop_after_native_claim)
        daemon.run_once()
        assert len(selected) == 1
        next_attempt = selected[0]
        assert next_attempt.attempt_number == attempt.attempt_number + 1
        assert next_attempt.fencing_token > attempt.fencing_token
        next_attempt = daemon.commit_phase(next_attempt, "context")
        current, result, duplicate = daemon.run_provider(next_attempt)
        assert result["status"] == "native_doctor_candidate_prepared"
        assert duplicate is False
        assert current.attempt_id == next_attempt.attempt_id
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == before
        rows = daemon._require_connection().execute(
            "SELECT attempt_id FROM daemon_execution_events WHERE event_type=? ORDER BY attempt_id", [CLOSED],
        ).fetchall()
        assert {row[0] for row in rows} == {attempt.attempt_id, next_attempt.attempt_id}
    finally:
        daemon.close()


def test_actual_competing_native_writer_lock_denies_recovery(tmp_path, monkeypatch):
    import select
    import sys
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.unresolved_interruption import reserve
    daemon, attempt, callback, now = setup(tmp_path)
    try:
        make_unknown(daemon, attempt, now, monkeypatch)
        lock_name = hashlib.sha256(str(callback.adapter.repository_root).encode()).hexdigest()
        lock_path = callback.adapter.state_root / "locks" / (lock_name + ".lock")
        child = subprocess.Popen(
            [sys.executable, "-c", "import fcntl,sys; f=open(sys.argv[1],'r+b'); fcntl.flock(f,fcntl.LOCK_EX); print('held',flush=True); sys.stdin.read()", str(lock_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            assert select.select([child.stdout], [], [], 5)[0]
            assert child.stdout.readline().strip() == "held"
            before = daemon.task_source.get(attempt.task_cid).to_dict()
            with pytest.raises(BlockingIOError):
                reserve(daemon, attempt, callback)
            assert daemon.task_source.get(attempt.task_cid).to_dict() == before
        finally:
            child.communicate(timeout=5)
    finally:
        daemon.close()


def test_replaced_native_lock_name_never_rebinds_existing_custody(tmp_path, monkeypatch):
    daemon, attempt, callback, now = setup(tmp_path)
    try:
        make_unknown(daemon, attempt, now, monkeypatch)
        with DoctorInterruptionCustody(daemon, attempt, callback) as held:
            original = held.lock_path.with_suffix(".original")
            held.lock_path.rename(original)
            held.lock_path.write_bytes(b"")
            with pytest.raises(DoctorCallbackDenied, match="lock path"):
                held.require_current()
    finally:
        daemon.close()


@pytest.mark.parametrize("boundary", ["reserved_control_cas", "admitted_control_cas", "admitted_execution_record"])
def test_real_peer_cannot_claim_until_all_native_admission_records_are_durable(tmp_path, monkeypatch, boundary):
    import json
    import sys
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import unresolved_interruption as recovery
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import open_process_serialized_database_coordinator
    daemon, attempt, callback, now = setup(tmp_path)
    try:
        make_unknown(daemon, attempt, now, monkeypatch)
        daemon.coordinator.close()
        daemon._coordinator = open_process_serialized_database_coordinator(
            tmp_path / "coordination.duckdb", clock_ms=lambda: now["ms"],
        )
        real_record = recovery._record
        real_admit = daemon.coordinator.admit_unresolved_interruption
        peer_observations = []

        def peer_must_be_blocked():
            code = "\n".join([
                "import json,sys",
                "from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import open_process_serialized_database_coordinator, DatabaseCoordinationNotReadyError",
                "coordinator=open_process_serialized_database_coordinator(sys.argv[1])",
                "try:",
                " coordinator.claim_task(task_cid=sys.argv[2],owner_session_id='independent-peer',now_ms=7000)",
                "except DatabaseCoordinationNotReadyError as exc:",
                " print(json.dumps(exc.evidence))",
                "else:",
                " raise AssertionError('peer advanced the native fence during an incomplete admission')",
                "finally:",
                " coordinator.close()",
            ])
            child = subprocess.run([sys.executable, "-c", code, str(tmp_path / "coordination.duckdb"), attempt.task_cid], check=False, capture_output=True, text=True, timeout=10)
            assert child.returncode == 0, child.stderr
            observed = json.loads(child.stdout)
            assert {item["kind"] for item in observed["repair_evidence"]} == {"unresolved_interruption_pending"}
            peer_observations.append(observed)

        def boundary_record(daemon, kind, attempt, body):
            if (boundary == "reserved_control_cas" and kind == recovery.RESERVED) or (boundary == "admitted_control_cas" and kind == recovery.ADMITTED):
                peer_must_be_blocked()
            return real_record(daemon, kind, attempt, body)

        def boundary_admit(*args, **kwargs):
            if boundary == "admitted_execution_record":
                peer_must_be_blocked()
            return real_admit(*args, **kwargs)

        monkeypatch.setattr(recovery, "_record", boundary_record)
        monkeypatch.setattr(daemon.coordinator, "admit_unresolved_interruption", boundary_admit)
        recovery.reserve(daemon, attempt, callback)
        recovery.admit_continuation(daemon, attempt, callback)
        assert len(peer_observations) == 1
        next_attempt = daemon.claim_next()
        assert next_attempt.attempt_number == attempt.attempt_number + 1
    finally:
        daemon.close()


def test_native_barrier_completion_response_loss_remains_idempotent_after_next_claim(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import unresolved_interruption as recovery
    daemon, attempt, callback, now = setup(tmp_path)
    try:
        make_unknown(daemon, attempt, now, monkeypatch)
        recovery.reserve(daemon, attempt, callback)
        real_admit = daemon.coordinator.admit_unresolved_interruption
        def lost_response(*args, **kwargs):
            real_admit(*args, **kwargs)
            raise OSError("native committed response lost")
        monkeypatch.setattr(daemon.coordinator, "admit_unresolved_interruption", lost_response)
        with pytest.raises(OSError):
            recovery.admit_continuation(daemon, attempt, callback)
        assert daemon.list_running_attempts() == []
        next_attempt = daemon.claim_next()
        assert next_attempt.attempt_number == attempt.attempt_number + 1
        receipt = recovery.admit_continuation(daemon, attempt, callback)
        assert receipt["callback_outcome"] == "unknown"
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict()
    finally:
        daemon.close()


def test_forged_admission_dictionary_cannot_clear_actual_native_barrier(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import unresolved_interruption as recovery
    daemon, attempt, callback, now = setup(tmp_path)
    try:
        make_unknown(daemon, attempt, now, monkeypatch)
        reserved = recovery.reserve(daemon, attempt, callback)
        forged = recovery._body(
            recovery.ADMITTED, attempt, reservation_receipt_id=reserved["receipt_id"],
            prior_control_revision=reserved["resulting_control_revision"],
            resulting_control_revision=reserved["resulting_control_revision"] + 1,
            retry_authorized=True, prior_evidence_reused=False,
        )
        claimed_task = daemon.task_source.get(attempt.task_cid).to_dict()
        claimed_task.update(status="retrying", revision=forged["resulting_control_revision"])
        claimed_task["body"]["completion_receipt"] = forged
        with pytest.raises(DatabaseCoordinationError, match="retained native"):
            daemon.coordinator.admit_unresolved_interruption(
                attempt.to_dict(), reservation_receipt_id=reserved["receipt_id"],
                admission={"admission_receipt": forged, "control_task": claimed_task}, now_ms=7000,
            )
        assert daemon.coordinator.get_unresolved_interruption(attempt.to_dict(), reserved["receipt_id"])["admitted"] is None
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
    finally:
        daemon.close()


@pytest.mark.parametrize("damage", ["closed_custody", "missing_execution_record", "changed_control"])
def test_native_admission_rechecks_actual_custody_and_both_stores(tmp_path, monkeypatch, damage):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import unresolved_interruption as recovery
    daemon, attempt, callback, now = setup(tmp_path)
    try:
        make_unknown(daemon, attempt, now, monkeypatch)
        reserved = recovery.reserve(daemon, attempt, callback)
        real_admit = daemon.coordinator.admit_unresolved_interruption

        def changed_boundary(*args, **kwargs):
            admission = kwargs["admission"]
            if damage == "closed_custody":
                admission.custody.close()
            elif damage == "missing_execution_record":
                daemon._require_connection().execute("DELETE FROM daemon_execution_events WHERE event_type=?", [recovery.ADMITTED])
            else:
                task = daemon.task_source.get(attempt.task_cid)
                daemon._cas_task_status_database(attempt.task_cid, expected_revision=task.revision, new_status="blocked", receipt={"operator": "different"})
            return real_admit(*args, **kwargs)

        monkeypatch.setattr(daemon.coordinator, "admit_unresolved_interruption", changed_boundary)
        with pytest.raises(DoctorCallbackDenied):
            recovery.admit_continuation(daemon, attempt, callback)
        assert daemon.coordinator.get_unresolved_interruption(attempt.to_dict(), reserved["receipt_id"])["admitted"] is None
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict()
    finally:
        daemon.close()


@pytest.mark.parametrize("budget_source", ["daemon", "declared_callback"])
def test_spent_attempt_budget_is_not_refunded_by_continuation(tmp_path, monkeypatch, budget_source):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import unresolved_interruption as recovery
    daemon, attempt, callback, now = setup(tmp_path, callback_budget=1 if budget_source == "declared_callback" else 3)
    try:
        if budget_source == "daemon":
            daemon.max_task_attempts = 1
        make_unknown(daemon, attempt, now, monkeypatch)
        recovery.reserve(daemon, attempt, callback)
        with pytest.raises(DoctorCallbackDenied, match="spent attempt budget"):
            recovery.admit_continuation(daemon, attempt, callback)
        assert daemon.claim_next() is None
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict()
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
    finally:
        daemon.close()


@pytest.mark.parametrize("stage", ["closed_unknown", "reserve_cas", "admit_cas", "native_admission"])
def test_actual_process_death_resumes_native_stage_without_replaying_old_callback(tmp_path, stage):
    import json
    import sys
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import unresolved_interruption as recovery
    code = "\n".join([
        "import json,os,sys,pytest",
        "from pathlib import Path",
        "from test.api.test_native_doctor_interruption import setup,make_unknown",
        "from ipfs_accelerate_py.agent_supervisor.todo_daemon import unresolved_interruption as r",
        "root=Path(sys.argv[1]); stage=sys.argv[2]",
        "daemon,attempt,callback,now=setup(root)",
        "m=pytest.MonkeyPatch(); make_unknown(daemon,attempt,now,m)",
        "daemon._require_connection().execute('CREATE TABLE opaque_retained (key VARCHAR, value BLOB)')",
        "daemon._require_connection().execute('INSERT INTO opaque_retained VALUES (?,?)', ['unknown', b'\\x00\\xffuninterpreted'])",
        "(root/'original-attempt.json').write_text(json.dumps(attempt.to_dict()))",
        "if stage=='closed_unknown': os._exit(0)",
        "original=r._record",
        "def stop_at_boundary(daemon,kind,attempt,body):",
        " if (stage=='reserve_cas' and kind==r.RESERVED) or (stage=='admit_cas' and kind==r.ADMITTED): os._exit(0)",
        " return original(daemon,kind,attempt,body)",
        "r._record=stop_at_boundary",
        "r.reserve(daemon,attempt,callback)",
        "r.admit_continuation(daemon,attempt,callback)",
        "os._exit(0)",
    ])
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), stage], capture_output=True, text=True, timeout=20)
    assert child.returncode == 0, child.stderr
    original = json.loads((tmp_path / "original-attempt.json").read_text())
    body = b"VALUE = 'baseline'\n"
    callback = NativeDoctorCallback(
        repository_root=tmp_path / "repository", state_root=tmp_path / "doctor",
        edits=[DoctorExactEdit("value.py", "sha256:" + hashlib.sha256(body).hexdigest(), b"VALUE = 'candidate'\n")],
    )
    daemon = _open_daemon(tmp_path, session=original["owner_session_id"], provider_fn=callback, lease_ms=5000, clock_ms=lambda: 7000)
    daemon.require_real_execution = True
    try:
        attempt = daemon.get_attempt(original["attempt_id"])
        assert attempt.to_dict() == original
        if recovery._read(daemon, recovery.RESERVED, attempt) is None:
            recovery.reserve(daemon, attempt, callback)
        receipt = recovery.admit_continuation(daemon, attempt, callback)
        assert receipt["callback_outcome"] == "unknown"
        next_attempt = daemon.claim_next()
        assert next_attempt.attempt_number == attempt.attempt_number + 1
        assert next_attempt.fencing_token > attempt.fencing_token
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == original
        row = daemon._require_connection().execute("SELECT value FROM opaque_retained WHERE key='unknown'").fetchone()
        assert row[0] == b"\x00\xffuninterpreted"
        events = daemon._require_connection().execute("SELECT count(*) FROM daemon_execution_events WHERE event_type=? AND attempt_id=?", [CLOSED, attempt.attempt_id]).fetchone()
        assert events[0] == 1
    finally:
        daemon.close()


@pytest.mark.parametrize("damage", ["invalid_reservation_id", "boolean_fence", "invalid_admission_id", "crossed_preparation"])
def test_native_claimability_refuses_malformed_or_crossed_barrier_history(tmp_path, monkeypatch, damage):
    import json
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import unresolved_interruption as recovery
    from ipfs_accelerate_py.agent_supervisor.merge import unresolved_interruption_barrier as barrier
    daemon, attempt, callback, now = setup(tmp_path)
    try:
        make_unknown(daemon, attempt, now, monkeypatch)
        recovery.reserve(daemon, attempt, callback)
        recovery.admit_continuation(daemon, attempt, callback)
        connection = daemon.coordinator._require()
        kind = barrier.PREPARED if damage in {"invalid_reservation_id", "boolean_fence"} else barrier.ADMITTED
        row = connection.execute("SELECT event_id,body_json FROM lease_events WHERE event_type=?", [kind]).fetchone()
        body = json.loads(row[1])
        if damage == "invalid_reservation_id":
            body["reservation_receipt_id"] = "sha256:" + "z" * 64
        elif damage == "boolean_fence":
            body["identity"]["fence_epoch"] = True
        elif damage == "invalid_admission_id":
            body["admission_receipt_id"] = "not-a-receipt"
        else:
            body["control_revision"] += 1
            body["admission_control_revision"] += 1
        original = {key:value for key,value in body.items() if key not in {"barrier_id", "record_id", "admission_receipt_id", "admission_control_revision"}}
        original["operation"] = barrier.PREPARED
        body["barrier_id"] = barrier._identity(original)
        if kind == barrier.ADMITTED:
            body["record_id"] = barrier._identity({key:value for key,value in body.items() if key != "record_id"})
        connection.execute("UPDATE lease_events SET body_json=? WHERE event_id=?", [json.dumps(body), row[0]])
        with pytest.raises(DatabaseCoordinationError):
            daemon.coordinator.claim_task(task_cid=attempt.task_cid, owner_session_id="peer", now_ms=7000)
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict()
    finally:
        daemon.close()


def test_current_owner_recovery_waits_for_native_outer_callback_return(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import unresolved_interruption as recovery
    daemon, attempt, callback, now = setup(tmp_path)
    closed, release, tried, recovered, entered = (threading.Event() for _ in range(5))
    original_heartbeat = daemon._run_with_attempt_heartbeat
    original_require = recovery._require_unknown

    def observed_recovery_entry(*args):
        entered.set()
        return original_require(*args)

    def paused_outer_return(attempt, execute):
        def after_actual_callback():
            result = execute()
            # The real Doctor callback has returned and released its writer
            # lock. Pause before outer acceptance, where recovery must still
            # wait for this native daemon dispatch to leave its critical section.
            now["ms"] = 7000
            closed.set()
            assert release.wait(10)
            return result
        return original_heartbeat(attempt, after_actual_callback)

    def run_recovery():
        tried.set()
        result = recovery.reserve(daemon, attempt, callback)
        recovered.set()
        return result

    monkeypatch.setattr(daemon, "_run_with_attempt_heartbeat", paused_outer_return)
    monkeypatch.setattr(recovery, "_require_unknown", observed_recovery_entry)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            provider = pool.submit(daemon.run_provider, attempt)
            assert closed.wait(10)
            recovering = pool.submit(run_recovery)
            assert tried.wait(5)
            try:
                assert not entered.wait(0.15)
                assert not recovered.is_set()
                assert not recovering.done()
            finally:
                release.set()
            with pytest.raises(DatabaseCoordinationError):
                provider.result(timeout=10)
            assert recovering.result(timeout=10)["callback_outcome"] == "unknown"
            assert recovered.is_set()
    finally:
        release.set()
        daemon.close()
