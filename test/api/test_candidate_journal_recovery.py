"""Real typed attempt admission and native journal replay; no invented settlement.

Signed route/DurableProviderAttemptCAS, Git rescue, native lifecycle files,
owner-issued typed grants and callback CAS are real. Protected Docker absence,
source capsule admission and the TCP Quack transport use the existing doubles.
"""

import json
from contextlib import contextmanager

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import (
    worktree_lifecycle_delete_journal as journal,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    candidate_rejection_closure as closure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as impl,
)
from test.api import test_candidate_rejection_closure as fixtures
from test.api.test_database_attempt_diagnostic_feedback import prompt_daemon


class InterruptedProcess(BaseException):
    pass


@contextmanager
def admitted(tmp_path, monkeypatch, *, omit_post=False):
    route_fixture = fixtures.routes._reviewed_route(tmp_path)
    state = {"calls": []}

    def factory(paths, alias):
        daemon = prompt_daemon(monkeypatch)
        daemon.repo_root = route_fixture[0]
        daemon.bind_launch_task_execution_route = lambda _: None
        daemon.close_event_runtime = lambda: None

        def run():
            state["paths"] = paths
            task = impl.parse_task_text(
                paths.task_projection.read_text(),
                path=paths.task_projection,
                task_header_prefix=f"## {alias}",
            )[0]
            state["calls"].append(task.task_id)
            return fixtures._preserved_candidate_pass(
                tmp_path,
                monkeypatch,
                bridge=bridge,
                paths=paths,
                task=task,
                route_fixture=route_fixture,
                omit_post=omit_post,
                journal=True,
            )

        daemon.run_once = run
        return daemon

    with fixtures._typed_outer(
        tmp_path, monkeypatch, repository=route_fixture[0], factory=factory, tick=True
    ) as objects:
        outer, bridge, source = objects
        state.update(
            outer=outer, bridge=bridge, source=source, repository=route_fixture[0]
        )
        yield state


def interrupt_after(fault, boundary):
    publish, move = journal._publish, journal._move_without_replace

    def interrupted_publish(directory, name, value):
        result = publish(directory, name, value)
        if boundary in {"prepared", "committed"} and name.endswith(f".{boundary}.json"):
            raise InterruptedProcess(boundary)
        return result

    def interrupted_move(source_directory, source, target_directory, target):
        result = move(source_directory, source, target_directory, target)
        if boundary == target.rsplit(".", 1)[-1]:
            raise InterruptedProcess(boundary)
        return result

    fault.setattr(journal, "_publish", interrupted_publish)
    fault.setattr(journal, "_move_without_replace", interrupted_move)


def callback(state, attempt):
    return state["outer"].provider_invocation_recorded(
        attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
    )


def assert_unknown(state, attempt, original):
    assert callback(state, attempt) == original
    assert original["callback_state"] == "started_outcome_unknown"
    assert (
        state["outer"].coordinator.get_task_claim(attempt.claim_id).state.value
        == "accepted"
    )
    assert len(state["calls"]) == 1


@pytest.mark.parametrize(
    "boundary",
    ["prepared", "record", "index", "committed", "post_event", "public_tick"],
)
def test_actual_typed_journal_recovery_admits_successor_without_provider_replay(
    tmp_path, monkeypatch, boundary
):
    with admitted(tmp_path, monkeypatch, omit_post=boundary == "post_event") as state:
        outer, bridge = state["outer"], state["bridge"]
        attempt = outer.claim_next()
        with monkeypatch.context() as fault:
            interrupt_after(fault, "record" if boundary == "public_tick" else boundary)
            with pytest.raises(
                (InterruptedProcess, closure.CandidateClosureObservationUnknown)
            ):
                outer._resume_attempt_without_process_crash(attempt)
        original = callback(state, attempt)
        assert_unknown(state, attempt, original)
        before_events = state["paths"].events.read_bytes()
        kinds = [row["type"] for row in bridge._verified_event_chain(state["paths"])]
        assert "implementation_finished" not in kinds
        assert "failed_validation_worktree_preserved" not in kinds
        assert closure.RELEASED_EVENT not in kinds
        # Read-only verification cannot complete an interrupted native operation.
        with monkeypatch.context() as readonly:
            readonly.setattr(
                journal.os, "fsync", lambda *_: pytest.fail("observer wrote")
            )
            observed = bridge.verify_candidate_rejection_closure(attempt)
        assert (observed is not None) == (boundary in {"committed", "post_event"})
        if observed is None:
            assert list((state["repository"] / ".git").rglob("*.committed.json")) == []
        result = (
            outer.run_once()["implementation_result"]
            if boundary == "public_tick"
            else outer._resume_attempt_without_process_crash(
                outer.get_attempt(attempt.attempt_id)
            )
        )
        assert result.get("status") == "failed", result
        receipt = callback(state, attempt)
        assert receipt["schema"] == closure.CALLBACK_SCHEMA
        assert receipt["closure"]["schema"] == closure.RECOVERED_CLOSURE_SCHEMA
        assert receipt["failure_fingerprint"] == original["failure_fingerprint"]
        assert state["paths"].events.read_bytes() == before_events
        assert (
            bridge.verify_candidate_rejection_closure(attempt, receipt["closure"])
            == receipt["closure"]
        )
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
        )
        outer.reconcile_terminal_retry_states()
        assert callback(state, attempt) == receipt
        successor = outer.claim_next()
        assert (
            successor is not None
            and successor.attempt_number == attempt.attempt_number + 1
        )
        assert len(state["calls"]) == 1


def test_actual_future_journal_normal_finish_remains_verifiable(tmp_path, monkeypatch):
    with admitted(tmp_path, monkeypatch) as state:
        attempt = state["outer"].claim_next()
        state["outer"]._resume_attempt_without_process_crash(attempt)
        receipt = callback(state, attempt)
        assert receipt["schema"] == closure.CALLBACK_SCHEMA
        assert receipt["closure"]["schema"] == closure.RECOVERED_CLOSURE_SCHEMA
        assert (
            state["bridge"].verify_candidate_rejection_closure(
                attempt, receipt["closure"]
            )
            == receipt["closure"]
        )
        assert len(state["calls"]) == 1


@pytest.mark.parametrize(
    "fault_kind",
    [
        "no_prepared",
        "missing_record",
        "foreign_canonical",
        "foreign_handoff",
        "changed_projection",
        "missing_rescue",
        "cleanup_unavailable",
        "lease_renewal",
        "directory_fsync",
        "publication_binding",
        "foreign_admission",
    ],
)
def test_actual_admitted_replay_failure_keeps_original_callback_and_claim(
    tmp_path, monkeypatch, fault_kind
):
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        WorktreeLifecycleStore,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        candidate_journal_recovery,
    )

    with admitted(tmp_path, monkeypatch) as state:
        outer, bridge = state["outer"], state["bridge"]
        attempt = outer.claim_next()
        with monkeypatch.context() as fault:
            if fault_kind == "no_prepared":
                original_publish = journal._publish

                def before_prepare(directory, name, value):
                    if name.endswith(".prepared.json"):
                        raise InterruptedProcess("before prepared")
                    return original_publish(directory, name, value)

                fault.setattr(journal, "_publish", before_prepare)
            else:
                interrupt_after(
                    fault, "record" if fault_kind == "missing_record" else "prepared"
                )
            with pytest.raises(InterruptedProcess):
                outer._resume_attempt_without_process_crash(attempt)
        original = callback(state, attempt)
        assert_unknown(state, attempt, original)
        store = WorktreeLifecycleStore(repo_root=state["repository"])
        directory = store.store_dir / journal.JOURNAL_DIR
        events = bridge._verified_event_chain(state["paths"])
        native = next(
            e["closure_terminal"] for e in events if e["type"] == closure.TERMINAL_EVENT
        )
        with monkeypatch.context() as fault:
            if fault_kind == "missing_record":
                next(directory.glob("*.record")).unlink()
            elif fault_kind == "foreign_canonical":
                path = store.workspace_path_for(native["terminal"]["workspace_path"])
                raw = path.read_bytes()
                path.unlink()
                path.write_bytes(raw)
            elif fault_kind == "foreign_handoff":
                path = next(directory.glob("*.prepared.json"))
                payload = json.loads(path.read_text())
                payload["binding"]["handoff_receipt_id"] = "sha256:" + "9" * 64
                path.write_text(
                    json.dumps(
                        journal._seal(
                            {k: v for k, v in payload.items() if k != "receipt_id"}
                        )
                    )
                )
            elif fault_kind == "changed_projection":
                with state["paths"].task_projection.open("a") as stream:
                    stream.write("\nUnexpected changed task contract\n")
            elif fault_kind == "missing_rescue":
                fixtures.routes._git(
                    state["repository"], "branch", "-D", native["rescue_branch"]
                )
            elif fault_kind == "cleanup_unavailable":
                fault.setattr(
                    candidate_journal_recovery,
                    "observe_provider_cleanup",
                    lambda **_: None,
                )
            elif fault_kind == "lease_renewal":

                def expired(*_args, **_kwargs):
                    raise impl.DatabaseImplementationAuthorityError(
                        "exact lease renewal unavailable"
                    )

                fault.setattr(outer, "_renew_attempt_lease", expired)
            elif fault_kind == "directory_fsync":

                def fail_sync(*_):
                    raise OSError("injected durability failure")

                fault.setattr(journal.os, "fsync", fail_sync)
            elif fault_kind == "publication_binding":
                publish = journal._publish

                def remove_commit(directory_fd, name, value):
                    result = publish(directory_fd, name, value)
                    if name.endswith(".committed.json"):
                        journal.os.unlink(name, dir_fd=directory_fd)
                    return result

                fault.setattr(journal, "_publish", remove_commit)
            elif fault_kind == "foreign_admission":
                with pytest.raises(
                    impl.DatabaseImplementationAuthorityError, match="bound attempt"
                ):
                    bridge.recover_candidate_rejection_closure(
                        attempt, admitted_daemon=object()
                    )
                assert not list(directory.glob("*.committed.json"))
                assert_unknown(state, attempt, original)
                return
            with pytest.raises(closure.CandidateClosureObservationUnknown):
                outer.run_provider(outer.get_attempt(attempt.attempt_id))
        assert_unknown(state, attempt, original)
        # In particular no callback completion, task retry or claim release has
        # been manufactured merely from absence or a changed byte-valid record.
        assert state["source"].get_task(attempt.task_cid).status == "in_progress"


def test_visible_commit_requires_mutating_durability_reassertion_before_callback_cas(
    tmp_path, monkeypatch
):
    with admitted(tmp_path, monkeypatch, omit_post=True) as state:
        outer, bridge = state["outer"], state["bridge"]
        attempt = outer.claim_next()
        with pytest.raises(closure.CandidateClosureObservationUnknown):
            outer._resume_attempt_without_process_crash(attempt)
        original = callback(state, attempt)
        expected = bridge.verify_candidate_rejection_closure(attempt)
        assert expected is not None
        events = state["paths"].events.read_bytes()
        with monkeypatch.context() as fault:

            def unavailable(*_):
                raise OSError("retry durability unavailable")

            fault.setattr(journal.os, "fsync", unavailable)
            with pytest.raises(closure.CandidateClosureObservationUnknown):
                outer.run_provider(outer.get_attempt(attempt.attempt_id))
        assert_unknown(state, attempt, original)
        result = outer._resume_attempt_without_process_crash(
            outer.get_attempt(attempt.attempt_id)
        )
        assert result["status"] == "failed"
        receipt = callback(state, attempt)
        assert receipt["closure"] == expected
        assert state["paths"].events.read_bytes() == events
        assert outer.claim_next() is not None
        assert len(state["calls"]) == 1


@pytest.mark.parametrize("malformed", [None, [], "invalid", True])
def test_malformed_nested_validation_through_public_unknown_callback_entry(
    tmp_path, monkeypatch, malformed
):
    with admitted(tmp_path, monkeypatch, omit_post=True) as state:
        outer, bridge = state["outer"], state["bridge"]
        attempt = outer.claim_next()
        append = impl.append_jsonl_event

        def malformed_terminal(path, kind, payload, **kwargs):
            if kind == closure.TERMINAL_EVENT:
                payload = json.loads(json.dumps(payload))
                native = payload["closure_terminal"]
                disposition = native["disposition"]
                disposition["validation"] = malformed
                native["disposition"] = closure.sealed(
                    {k: v for k, v in disposition.items() if k != "receipt_id"}
                )
                payload["closure_terminal"] = closure.sealed(
                    {k: v for k, v in native.items() if k != "receipt_id"}
                )
            return append(path, kind, payload, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(impl, "append_jsonl_event", malformed_terminal)
            with pytest.raises(closure.CandidateClosureObservationUnknown):
                outer._resume_attempt_without_process_crash(attempt)
        original = callback(state, attempt)
        # The event chain itself is valid; nested evidence is not. Public
        # run_provider must retain the original unknown intent, not leak an
        # AttributeError into ordinary outer failure cleanup.
        assert bridge._verified_event_chain(state["paths"])
        with pytest.raises(closure.CandidateClosureObservationUnknown):
            outer.run_provider(outer.get_attempt(attempt.attempt_id))
        assert bridge.verify_candidate_rejection_closure(attempt) is None
        assert_unknown(state, attempt, original)


@pytest.mark.parametrize("after_commit", [False, True])
def test_callback_publication_uncertainty_replays_exact_receipt_without_provider(
    tmp_path, monkeypatch, after_commit
):
    with admitted(tmp_path, monkeypatch, omit_post=True) as state:
        outer = state["outer"]
        attempt = outer.claim_next()
        with pytest.raises(closure.CandidateClosureObservationUnknown):
            outer._resume_attempt_without_process_crash(attempt)
        original = callback(state, attempt)
        publish = outer._commit_candidate_callback_closure

        def response_lost(*args, **kwargs):
            if after_commit:
                publish(*args, **kwargs)
            raise OSError("callback CAS reply unavailable")

        with monkeypatch.context() as fault:
            fault.setattr(outer, "_commit_candidate_callback_closure", response_lost)
            with pytest.raises(
                closure.CandidateClosureObservationUnknown, match="publication"
            ):
                outer._resume_attempt_without_process_crash(
                    outer.get_attempt(attempt.attempt_id)
                )
        captured = callback(state, attempt)
        assert captured["failure_fingerprint"] == original["failure_fingerprint"]
        assert (captured["schema"] == closure.CALLBACK_SCHEMA) is after_commit
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        )
        assert state["source"].get_task(attempt.task_cid).status == "in_progress"
        result = outer._resume_attempt_without_process_crash(
            outer.get_attempt(attempt.attempt_id)
        )
        assert result["status"] == "failed"
        closed = callback(state, attempt)
        if after_commit:
            assert closed == captured
        assert outer.claim_next() is not None
        assert len(state["calls"]) == 1


def test_real_later_finish_events_do_not_change_prefix_journal_receipt(
    tmp_path, monkeypatch
):
    with admitted(tmp_path, monkeypatch) as state:
        outer, bridge = state["outer"], state["bridge"]
        attempt = outer.claim_next()
        append = impl.append_jsonl_event
        observed = []

        def observe_before_later_events(path, kind, payload, **kwargs):
            result = append(path, kind, payload, **kwargs)
            if kind == closure.RELEASED_EVENT:
                before = bridge._verified_event_chain(state["paths"])
                assert not any(e["type"] == "implementation_finished" for e in before)
                observed.append(bridge.verify_candidate_rejection_closure(attempt))
            return result

        with monkeypatch.context() as fault:
            fault.setattr(impl, "append_jsonl_event", observe_before_later_events)
            outer._resume_attempt_without_process_crash(attempt)
        receipt = callback(state, attempt)
        assert observed == [receipt["closure"]]
        assert observed[0] is not None
        assert any(
            e["type"] == "implementation_finished"
            for e in bridge._verified_event_chain(state["paths"])
        )
        assert (
            bridge.verify_candidate_rejection_closure(attempt, observed[0])
            == observed[0]
        )
        assert len(state["calls"]) == 1


def test_expired_real_claim_cannot_resume_prepared_native_deletion(
    tmp_path, monkeypatch
):
    import time

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinationExpiredError,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        WorktreeLifecycleStore,
    )

    with admitted(tmp_path, monkeypatch) as state:
        outer, bridge = state["outer"], state["bridge"]
        attempt = outer.claim_next()
        with monkeypatch.context() as fault:
            interrupt_after(fault, "prepared")
            with pytest.raises(InterruptedProcess):
                outer._resume_attempt_without_process_crash(attempt)
        original = callback(state, attempt)
        store = WorktreeLifecycleStore(repo_root=state["repository"])
        native = next(
            e["closure_terminal"]
            for e in bridge._verified_event_chain(state["paths"])
            if e["type"] == closure.TERMINAL_EVENT
        )
        record_path = store.workspace_path_for(native["terminal"]["workspace_path"])
        before = (record_path.stat().st_ino, record_path.read_bytes())
        claim = outer.coordinator.get_task_claim(attempt.claim_id)
        time.sleep(
            max(0, (claim.expires_at_ms - int(time.time() * 1000)) / 1000) + 0.15
        )
        with pytest.raises(DatabaseCoordinationExpiredError):
            bridge.recover_candidate_rejection_closure(attempt, admitted_daemon=outer)
        assert (record_path.stat().st_ino, record_path.read_bytes()) == before
        assert callback(state, attempt) == original
        assert not list(
            (store.store_dir / journal.JOURNAL_DIR).glob("*.committed.json")
        )
        assert len(state["calls"]) == 1


@pytest.mark.parametrize("closed", [False, True])
def test_public_tick_failed_journal_proof_preserves_task_claim_and_callback(
    tmp_path, monkeypatch, closed
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        candidate_journal_recovery,
    )

    with admitted(tmp_path, monkeypatch, omit_post=closed) as state:
        outer = state["outer"]
        attempt = outer.claim_next()
        with monkeypatch.context() as fault:
            if not closed:
                interrupt_after(fault, "prepared")
            with pytest.raises(
                (InterruptedProcess, closure.CandidateClosureObservationUnknown)
            ):
                outer._resume_attempt_without_process_crash(attempt)
        if closed:
            publish = outer._commit_candidate_callback_closure

            def lose_reply(*args, **kwargs):
                publish(*args, **kwargs)
                raise OSError("callback CAS reply lost")

            with monkeypatch.context() as fault:
                fault.setattr(outer, "_commit_candidate_callback_closure", lose_reply)
                with pytest.raises(closure.CandidateClosureObservationUnknown):
                    outer._resume_attempt_without_process_crash(
                        outer.get_attempt(attempt.attempt_id)
                    )
        original = callback(state, attempt)
        task = state["source"].get_task(attempt.task_cid)
        expected_task = (task.revision, task.status, dict(task.body))
        with monkeypatch.context() as fault:
            if closed:
                fault.setattr(
                    candidate_journal_recovery,
                    "observe_provider_cleanup",
                    lambda **_: None,
                )
            else:
                sync = journal.os.fsync

                def fail_journal_sync(fd):
                    if journal.JOURNAL_DIR in journal.os.readlink(
                        f"/proc/self/fd/{fd}"
                    ):
                        raise OSError("journal durability unavailable")
                    return sync(fd)

                fault.setattr(journal.os, "fsync", fail_journal_sync)
            with pytest.raises(closure.CandidateClosureObservationUnknown):
                outer.run_once()
        assert callback(state, attempt) == original
        task = state["source"].get_task(attempt.task_cid)
        assert (task.revision, task.status, dict(task.body)) == expected_task
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        )
        assert len(state["calls"]) == 1
        # Restore proof, then ordinary public tick can complete exact replay.
        result = outer.run_once()
        assert result["implementation_result"]["status"] == "failed"
        assert outer.claim_next() is not None
        assert len(state["calls"]) == 1


@pytest.mark.parametrize(
    "fault_kind,malformed",
    [
        ("unreadable_events", None),
        ("malformed_original_intent", {"schema": "malformed-fixture-intent"}),
        ("malformed_original_intent", None),
        ("malformed_original_intent", True),
    ],
)
def test_public_tick_unavailable_classification_and_malformed_callback_retain_custody(
    tmp_path, monkeypatch, fault_kind, malformed
):
    with admitted(tmp_path, monkeypatch, omit_post=True) as state:
        outer, bridge = state["outer"], state["bridge"]
        attempt = outer.claim_next()
        with pytest.raises(closure.CandidateClosureObservationUnknown):
            outer._resume_attempt_without_process_crash(attempt)
        if fault_kind == "malformed_original_intent":
            publish = outer._commit_candidate_callback_closure

            def lose_reply(*args, **kwargs):
                publish(*args, **kwargs)
                raise OSError("callback CAS reply lost")

            with monkeypatch.context() as fault:
                fault.setattr(outer, "_commit_candidate_callback_closure", lose_reply)
                with pytest.raises(closure.CandidateClosureObservationUnknown):
                    outer._resume_attempt_without_process_crash(
                        outer.get_attempt(attempt.attempt_id)
                    )
        original = callback(state, attempt)
        corrupted = None
        if fault_kind == "malformed_original_intent":
            corrupted = dict(original)
            corrupted["original_intent"] = malformed
            corrupted = closure.sealed(
                {k: v for k, v in corrupted.items() if k != "receipt_id"}
            )
            # Corruption injection into this disposable execution database only.
            changed = (
                outer._require_connection()
                .execute(
                    "UPDATE provider_invocations SET result_json = ? WHERE attempt_id = ? AND result_json = ? RETURNING invocation_id",
                    [
                        impl._database_daemon_json(corrupted),
                        attempt.attempt_id,
                        impl._database_daemon_json(original),
                    ],
                )
                .fetchone()
            )
            assert changed is not None
        task = state["source"].get_task(attempt.task_cid)
        expected_task = (task.revision, task.status, dict(task.body))
        with monkeypatch.context() as fault:
            if fault_kind == "unreadable_events":

                def unreadable(*_):
                    raise OSError("exact attempt event stream unavailable")

                fault.setattr(bridge, "_verified_event_chain", unreadable)
            with pytest.raises(closure.CandidateClosureObservationUnknown):
                outer.run_once()
        assert callback(state, attempt) == (corrupted or original)
        task = state["source"].get_task(attempt.task_cid)
        assert (task.revision, task.status, dict(task.body)) == expected_task
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        )
        assert len(state["calls"]) == 1
