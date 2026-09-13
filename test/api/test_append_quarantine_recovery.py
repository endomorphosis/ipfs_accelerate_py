"""Native queue/callback settlement after a retained append-verification failure."""

import json
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_PORTAL_COMPLETION_CALLBACK_BINDING_INVALID_REASON,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.append_reconciliation_recovery import (
    REVIVAL_REASON,
    append_quarantine_lineage,
)
from test.api.test_agent_supervisor_merge_train import (
    _DatabaseProjectionTaskSource,
    _database_projection_attempt,
    _database_projection_daemon,
    _database_projection_record,
    _git,
    _repo,
)


def native_append_quarantine(
    tmp_path,
    monkeypatch,
    *,
    source_attempt=None,
    source_record=None,
    task_source=None,
    nested_output=False,
    attempt_inside_repository=False,
):
    repo = _repo(tmp_path)
    output_path = "external/child/base.txt" if nested_output else "base.txt"
    submodule_paths = ("external/child",) if nested_output else ()
    if nested_output:
        monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
        seed_root = tmp_path / "child-seed"
        seed_root.mkdir()
        seed = _repo(seed_root)
        origin = tmp_path / "child-origin.git"
        _git(repo, "clone", "--bare", str(seed), str(origin))
        _git(repo, "submodule", "add", str(origin), "external/child")
        _git(repo, "commit", "-am", "add declared output submodule")
        child = repo / "external/child"
        _git(child, "config", "user.name", "Append Recovery Test")
        _git(child, "config", "user.email", "append@example.invalid")
    attempt = source_attempt or _database_projection_attempt(
        attempt_id="attempt:append-verification",
        claim_id="claim:append-verification",
        task_cid="task:cid:ref-040",
        attempt_number=1,
    )
    record = source_record or _database_projection_record(revision=2)
    if source_record is None:
        record.status = "blocked"
        record.outputs = ({"path": output_path},)
        record.validations = (
            {
                "argv": [
                    "python3",
                    "-c",
                    f"from pathlib import Path; assert Path({output_path!r}).read_text() == 'verified candidate\\n'",
                ]
            },
        )
    attempt_root = (
        repo if attempt_inside_repository else tmp_path
    ) / "append_database_portal_attempts"
    if attempt_inside_repository:
        with (repo / ".git/info/exclude").open("a") as stream:
            stream.write("\n/append_database_portal_attempts/\n")
    daemon, paths, binding = _database_projection_daemon(
        repo=repo,
        attempt_root=attempt_root,
        merge_queue_dir=tmp_path / "queue",
        attempt=attempt,
        record=record,
    )
    daemon.worktree_submodule_paths = submodule_paths
    baseline = _git(repo, "rev-parse", "HEAD")
    branch = "implementation/append-verification"
    _git(repo, "switch", "-c", branch)
    (repo / output_path).write_text("verified candidate\n")
    if nested_output:
        _git(child, "commit", "-am", "candidate declared output")
        _git(child, "push", "origin", "HEAD:refs/heads/main")
    _git(repo, "add", "external/child" if nested_output else "base.txt")
    _git(repo, "commit", "-m", "candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", daemon.resolved_merge_target_branch)
    if nested_output:
        _git(repo, "submodule", "update", "--init")
    [task] = daemon._load_tasks()
    daemon._task_identity_by_display_id[task.task_id] = daemon._identity_for_task(task)
    request, queued = daemon._enqueue_merge_candidate(
        branch_name=branch,
        implementation_commit=candidate,
        baseline_ref=baseline,
        worktree_path=None,
        task=task,
        attempt=1,
        changed_submodule_paths=submodule_paths,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
            "selection": {"scope": "pre_merge"},
        },
    )
    original = daemon._record_merge_queue_callback_reconciliation

    def fail_after_append(**kwargs):
        result = original(**kwargs)
        assert result["recorded"] is True and result["replayed"] is False
        return {
            "recorded": False,
            "reason": "merge_queue_reconciliation_append_unverified",
            "reconciliation_count": 1,
        }

    monkeypatch.setattr(
        daemon, "_record_merge_queue_callback_reconciliation", fail_after_append
    )
    train = MergeTrain(
        repo,
        daemon.merge_queue,
        target_branch=daemon.resolved_merge_target_branch,
        merge_callback=daemon._merge_train_callback,
    )
    result = train.run_once()
    assert result["status"] == "quarantined", result
    assert result["reason"] == "merge_queue_reconciliation_append_unverified"
    monkeypatch.setattr(daemon, "_record_merge_queue_callback_reconciliation", original)
    daemon._record_event(
        "implementation_finished",
        {
            "task_id": task.task_id,
            "canonical_task_cid": request.canonical_task_id,
            "attempt": 1,
            "returncode": 0,
            "attempt_consumed": True,
            "provider_dispatched": True,
            "branch": branch,
            "baseline_ref": baseline,
            "implementation_commit": candidate,
            "validation_result": {"attempted": True, "passed": True, "returncode": 0},
            "merge_result": dict(queued),
            "board_completion": {
                "complete": False,
                "pending_merge": True,
                "reason": "merge_queued_awaiting_integration",
            },
        },
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=task_source or _DatabaseProjectionTaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda *_: daemon,
        repository_root=repo,
        merge_queue=daemon.merge_queue,
        merge_target_branch=daemon.resolved_merge_target_branch,
        task_header_prefix="## REF-",
        worktree_submodule_paths=submodule_paths,
    )
    return SimpleNamespace(
        repo=repo,
        daemon=daemon,
        train=train,
        paths=paths,
        binding=binding,
        request=request,
        bridge=bridge,
        record=record,
    )


def test_native_append_quarantine_is_selected_without_effects(tmp_path, monkeypatch):
    fixture = native_append_quarantine(tmp_path, monkeypatch)
    request = fixture.daemon.merge_queue.get(fixture.request.request_id)
    events_before = fixture.paths.events.read_bytes()
    assert fixture.bridge._owned_post_merge_recovery_projection(request) is None
    projection = fixture.bridge._owned_post_merge_maintenance_projection(
        request, train=fixture.train
    )
    assert projection is not None
    assert fixture.daemon.merge_queue.get(request.request_id).status == "quarantined"
    assert fixture.paths.events.read_bytes() == events_before
    fixture.record.status = "in_progress"
    assert (
        fixture.bridge._owned_post_merge_maintenance_projection(
            request, train=fixture.train
        )
        is None
    )


@pytest.mark.parametrize("projection_already_completed", [False, True])
def test_native_append_quarantine_settlement_retains_original_receipt(
    tmp_path, monkeypatch, projection_already_completed
):
    fixture = native_append_quarantine(tmp_path, monkeypatch)
    daemon, train = fixture.daemon, fixture.train
    if projection_already_completed:
        updated = daemon._mark_task_completed_in_todo(
            fixture.request.task_id,
            expected_task_cids={
                fixture.request.task_id: fixture.request.canonical_task_id
            },
        )
        assert updated["updated"] is True
    before = fixture.paths.events.read_bytes()
    [historical] = [
        json.loads(line)
        for line in before.decode().splitlines()
        if json.loads(line).get("type") == "merge_reconciled"
    ]
    _git(fixture.repo, "commit", "--allow-empty", "-m", "advance current target")

    @contextmanager
    def processor(selected_train):
        previous = selected_train._portal_projection_invalid_metadata_already_on_target
        selected_train._portal_projection_invalid_metadata_already_on_target = (
            lambda _: False
        )
        try:
            yield
        finally:
            selected_train._portal_projection_invalid_metadata_already_on_target = (
                previous
            )

    result = train.recover_one_integrated_quarantine(
        request_id=fixture.request.request_id,
        processor_context=processor,
        request_filter=lambda r: r.request_id == fixture.request.request_id,
    )
    assert result["status"] == "already_merged", result
    completed = daemon.merge_queue.get(fixture.request.request_id)
    assert completed.status == "completed"
    assert fixture.paths.events.read_bytes().startswith(before)
    events = daemon._iter_merge_lifecycle_events()
    reconciliations = [e for e in events if e["type"] == "merge_reconciled"]
    assert reconciliations == [historical]
    projection = SimpleNamespace(paths=fixture.paths, binding=fixture.binding)
    evidence = fixture.bridge._callback_integration_source_evidence(
        completed, projection, train=train
    )
    assert evidence is not None, {
        "settlement": result,
        "events": [(e["type"], e.get("reason")) for e in events],
    }


@pytest.mark.parametrize("crash_before_task_cas", [False, True])
@pytest.mark.parametrize("nested_output", [False, True])
@pytest.mark.parametrize("advance_target", [False, True])
@pytest.mark.parametrize(
    "projection_already_completed", [False, "single_task", "merged_status_repair"]
)
def test_public_append_recovery_validates_current_target_and_replays_settlement(
    tmp_path,
    monkeypatch,
    crash_before_task_cas,
    nested_output,
    advance_target,
    projection_already_completed,
):
    fixture = native_append_quarantine(
        tmp_path,
        monkeypatch,
        nested_output=nested_output,
        attempt_inside_repository=projection_already_completed,
    )
    daemon, bridge = fixture.daemon, fixture.bridge
    if projection_already_completed:
        result = daemon._mark_tasks_completed_in_todo(
            [fixture.request.task_id],
            primary_task_id=fixture.request.task_id,
            completion_reason=projection_already_completed,
            expected_task_cids={
                fixture.request.task_id: fixture.request.canonical_task_id
            },
        )
        assert result["updated"] is True
        if projection_already_completed == "merged_status_repair":
            daemon._record_event(
                "task_completed",
                {
                    "task_id": fixture.request.task_id,
                    "reason": "task_became_completed",
                    "completion_receipt_repair": False,
                },
            )
            daemon._record_event(
                "daemon_pass",
                {
                    "active_task_id": "",
                    "completed_count": 1,
                    "ready_count": 0,
                    "selection_idle_reason": "database_pending_merge_reconciliation",
                },
            )
    if advance_target:
        if nested_output:
            child = fixture.repo / "external/child"
            _git(child, "commit", "--allow-empty", "-m", "advance child target")
            _git(child, "push", "origin", "HEAD:refs/heads/main")
            _git(fixture.repo, "add", "external/child")
        _git(fixture.repo, "commit", "--allow-empty", "-m", "advance target")
    authority = object.__new__(DatabaseImplementationDaemon)
    authority._merge_queue = daemon.merge_queue
    authority._merge_repo_root = fixture.repo
    authority._merge_target_branch = daemon.resolved_merge_target_branch
    authority._merge_portal_attempt_root = bridge.attempt_root
    authority._callback_requalification_setup_audit_paths = (
        bridge.worktree_submodule_paths
    )
    authorizations, qualifications, validations, transactions = [], [], [], []
    original_validation = daemon._run_validation_commands

    def observe_validation(*args, **kwargs):
        result = original_validation(*args, **kwargs)
        validations.append(result)
        return result

    monkeypatch.setattr(daemon, "_run_validation_commands", observe_validation)
    original_transaction = daemon._run_checkout_mutation_transaction

    def observe_transaction(**kwargs):
        result = original_transaction(**kwargs)
        transactions.append(result)
        return result

    monkeypatch.setattr(
        daemon, "_run_checkout_mutation_transaction", observe_transaction
    )

    def preauthorize(source):
        authorizations.append(dict(source))
        result = {**source, "authorized": True, "task_status": "blocked"}
        return {
            **result,
            "authorization_id": authority._database_portal_evidence_digest(result),
        }

    def recover(evidence):
        acquired, _ = fixture.train.run_under_consumer_lease(lambda: None)
        assert acquired is False
        qualification = authority._verified_post_merge_callback_integration_receipt(
            evidence["callback_requalification_receipt"],
            recovery_evidence=evidence,
        )
        qualifications.append(qualification)
        if crash_before_task_cas and len(qualifications) == 1:
            raise RuntimeError("fixture crash before task CAS")
        return {
            "attempted": True,
            "recovered": True,
            "changed": True,
            "status": "retrying",
            "write_count": 2,
        }

    boundary = SimpleNamespace(
        preauthorize_post_merge_declared_output_recovery=preauthorize,
        recover_blocked_post_merge_declared_outputs=recover,
        _database_portal_evidence_digest=authority._database_portal_evidence_digest,
    )
    before = fixture.paths.events.read_bytes()
    if crash_before_task_cas:
        with pytest.raises(RuntimeError, match="fixture crash before task CAS"):
            bridge.recover_post_merge_declared_outputs(boundary)
        assert daemon.merge_queue.get(fixture.request.request_id).status == "completed"
        if advance_target:
            if nested_output:
                child = fixture.repo / "external/child"
                _git(child, "commit", "--allow-empty", "-m", "advance settled child")
                _git(child, "push", "origin", "HEAD:refs/heads/main")
                _git(fixture.repo, "add", "external/child")
            _git(
                fixture.repo,
                "commit",
                "--allow-empty",
                "-m",
                "advance after settlement",
            )
            assert (
                _git(fixture.repo, "rev-parse", "HEAD")
                != qualifications[0]["current_target_commit"]
            )
    result = bridge.recover_post_merge_declared_outputs(boundary)
    assert result is not None and result["recovered"] is True, json.dumps(
        {"result": result, "validations": validations, "transactions": transactions},
        indent=2,
    )
    assert authorizations and qualifications and validations
    if crash_before_task_cas and advance_target:
        assert qualifications[-1]["current_target_commit"] == _git(
            fixture.repo, "rev-parse", "HEAD"
        )
        assert (
            qualifications[-1]["current_target_commit"]
            != qualifications[0]["current_target_commit"]
        )
    assert all(result["passed"] is True for result in validations)
    assert fixture.paths.events.read_bytes().startswith(before)
    assert (
        len(
            [
                e
                for e in daemon._iter_merge_lifecycle_events()
                if e["type"] == "merge_reconciled"
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "failure_reason",
        "failure_count",
        "attempt",
        "completion",
        "revivals_type",
        "second_revival",
        "live_claim",
        "accepted",
        "quarantine_identity",
        "quarantine_time",
        "boolean_count",
        "boolean_receipt_count",
        "integer_recorded",
        "max_attempts",
    ],
)
def test_recorded_append_quarantine_lineage_rejects_changed_contract(mutation):
    raw = json.loads(
        (
            Path(__file__).parent / "fixtures/synchronous_append_quarantine.json"
        ).read_text()
    )["request"]
    request = SimpleNamespace(**raw)
    q = request.metadata["quarantine"]
    if mutation == "failure_reason":
        request.failure_reason = "arbitrary"
    elif mutation == "failure_count":
        request.failure_count = 2
    elif mutation == "attempt":
        request.attempt = 2
    elif mutation == "completion":
        request.metadata["completion"] = {}
    elif mutation == "revivals_type":
        request.metadata["revivals"] = {}
    elif mutation == "second_revival":
        request.metadata["revivals"] = [{"reason": "arbitrary"}]
    elif mutation == "live_claim":
        request.claim_token = "other-live-consumer"
    elif mutation == "accepted":
        q["accepted"] = True
    elif mutation == "quarantine_identity":
        q["canonical_task_id"] = "foreign"
    elif mutation == "quarantine_time":
        q["finished_at"] = float("nan")
    elif mutation == "boolean_count":
        q["failure_count"] = True
    elif mutation == "boolean_receipt_count":
        q["merge_result"]["merge_reconciliation_receipt"]["reconciliation_count"] = True
    elif mutation == "integer_recorded":
        q["merge_result"]["merge_reconciliation_receipt"]["recorded"] = 0
    elif mutation == "max_attempts":
        q["max_attempts"] = 4
    assert append_quarantine_lineage(request, max_attempts=3) is (mutation is None)


@pytest.mark.parametrize(
    "mutation",
    [
        "preauthorization",
        "event_chain",
        "later_execution",
        "duplicate_source",
        "receipt_missing",
        "receipt_symlink",
        "receipt_directory_symlink",
        "receipt_content",
    ],
)
def test_public_append_recovery_rejects_before_queue_or_task_effects(
    tmp_path, monkeypatch, mutation
):
    fixture = native_append_quarantine(tmp_path, monkeypatch)
    calls = []
    receipt = fixture.train._receipt_path("quarantine-" + fixture.request.request_id)
    if mutation == "event_chain":
        lines = fixture.paths.events.read_text().splitlines()
        event = json.loads(lines[-1])
        event["attempt"] += 1
        lines[-1] = json.dumps(event)
        fixture.paths.events.write_text("\n".join(lines) + "\n")
    elif mutation == "later_execution":
        fixture.daemon._record_event(
            "implementation_started", {"task_id": fixture.request.task_id, "attempt": 2}
        )
    elif mutation == "duplicate_source":
        original = next(
            e
            for e in fixture.daemon._iter_merge_lifecycle_events()
            if e.get("reason") == "merge_queue_synchronous_source_projected"
        )
        fixture.daemon._record_event(
            original["type"],
            {
                k: v
                for k, v in original.items()
                if k
                not in {
                    "type",
                    "event_id",
                    "timestamp",
                    "stream_id",
                    "sequence",
                    "previous_event_id",
                }
            },
        )
    elif mutation == "receipt_missing":
        receipt.unlink()
    elif mutation == "receipt_symlink":
        moved = tmp_path / "outside-receipt.json"
        receipt.rename(moved)
        receipt.symlink_to(moved)
    elif mutation == "receipt_directory_symlink":
        directory = fixture.train.receipt_dir
        moved = tmp_path / "outside-receipts"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
    elif mutation == "receipt_content":
        value = json.loads(receipt.read_text())
        value["reason"] = "foreign"
        receipt.write_text(json.dumps(value))

    def preauthorize(source):
        calls.append("preauthorize")
        if mutation == "preauthorization":
            raise DatabaseImplementationConflictError(
                "fixture source attempt superseded"
            )
        pytest.fail("invalid append source reached canonical preauthorization")

    boundary = SimpleNamespace(
        preauthorize_post_merge_declared_output_recovery=preauthorize,
        recover_blocked_post_merge_declared_outputs=lambda _: pytest.fail(
            "invalid source reached task CAS"
        ),
        _database_portal_evidence_digest=DatabaseImplementationDaemon._database_portal_evidence_digest,
    )
    before = fixture.paths.events.read_bytes()
    fixture.bridge.portal_factory = lambda *_: pytest.fail(
        "invalid source constructed a Portal daemon"
    )
    assert fixture.bridge.recover_post_merge_declared_outputs(boundary) is None
    assert (
        fixture.daemon.merge_queue.get(fixture.request.request_id).status
        == "quarantined"
    )
    assert fixture.paths.events.read_bytes() == before
    assert fixture.record.status == "blocked"
    assert calls == (["preauthorize"] if mutation == "preauthorization" else [])


@pytest.mark.parametrize("abandoned_claims", [1, 2, 3])
def test_native_append_claim_recovery_is_bounded(
    tmp_path, monkeypatch, abandoned_claims
):
    fixture = native_append_quarantine(tmp_path, monkeypatch)
    queue, train = fixture.daemon.merge_queue, fixture.train
    with train._consumer_lease() as acquired:
        assert acquired
        revived = queue.revive_quarantined(
            fixture.request, reason=REVIVAL_REASON, reset_failures=True
        )
        assert append_quarantine_lineage(revived, max_attempts=3)
        for number in range(abandoned_claims):
            claimed = queue.claim_pending_request(
                fixture.request.request_id, consumer_id=train.owner_id
            )
            assert append_quarantine_lineage(claimed, max_attempts=3)
            # Model an exited consumer under the same native exclusive lease.
            assert queue.recover_abandoned_train_claims() == 1
    current = queue.get(fixture.request.request_id)
    assert current.failure_count == abandoned_claims
    assert append_quarantine_lineage(current, max_attempts=3) is (abandoned_claims < 3)
    authorizations, recovered = [], []

    def preauthorize(source):
        authorizations.append(source)
        value = {**source, "authorized": True, "task_status": "blocked"}
        return {
            **value,
            "authorization_id": DatabaseImplementationDaemon._database_portal_evidence_digest(
                value
            ),
        }

    def recover(evidence):
        recovered.append(evidence)
        return {"recovered": True, "write_count": 0}

    boundary = SimpleNamespace(
        preauthorize_post_merge_declared_output_recovery=preauthorize,
        recover_blocked_post_merge_declared_outputs=recover,
        _database_portal_evidence_digest=DatabaseImplementationDaemon._database_portal_evidence_digest,
    )
    result = fixture.bridge.recover_post_merge_declared_outputs(boundary)
    after = queue.get(fixture.request.request_id)
    assert len(after.metadata["revivals"]) == 1
    if abandoned_claims < 3:
        assert result and result["recovered"] is True
        assert after.status == "completed" and len(recovered) == 1
        assert (
            len(
                [
                    e
                    for e in fixture.daemon._iter_merge_lifecycle_events()
                    if e["type"] == "merge_reconciled"
                ]
            )
            == 1
        )
    else:
        assert result is None and after.status == "quarantined"
        assert not authorizations and not recovered


@pytest.mark.parametrize("lose_cas_response", [False, True])
@pytest.mark.parametrize(
    "terminal_reason",
    [
        "post_merge_declared_outputs_missing",
        DATABASE_PORTAL_COMPLETION_CALLBACK_BINDING_INVALID_REASON,
    ],
)
@pytest.mark.parametrize("projection_already_completed", [False, True])
def test_native_append_recovery_rearms_real_database_task_once(
    tmp_path,
    monkeypatch,
    lose_cas_response,
    terminal_reason,
    projection_already_completed,
):
    from test.api.test_agent_supervisor_database_implementation_daemon import (
        _open_daemon,
        _population,
    )

    owner = _open_daemon(
        tmp_path / "owner",
        provider_fn=lambda _: pytest.fail("recovery dispatched a provider"),
    )
    try:
        population = _population(1)
        population["tasks"][0].update(
            task_id="REF-040",
            completion="auto",
            track="implementation",
            outputs=[{"path": "base.txt"}],
            validations=[
                {
                    "argv": [
                        "python3",
                        "-c",
                        "from pathlib import Path; assert Path('base.txt').read_text() == 'verified candidate\\n'",
                    ]
                }
            ],
        )
        owner.materialize_population(population)
        attempt = owner.claim_next()
        assert attempt is not None
        record = owner.task_source.get(attempt.task_cid)
        fixture = native_append_quarantine(
            tmp_path,
            monkeypatch,
            source_attempt=attempt,
            source_record=record,
            task_source=owner.task_source,
            attempt_inside_repository=projection_already_completed,
        )
        if projection_already_completed:
            update = fixture.daemon._mark_tasks_completed_in_todo(
                [fixture.request.task_id],
                primary_task_id=fixture.request.task_id,
                completion_reason="merged_status_repair",
                expected_task_cids={
                    fixture.request.task_id: fixture.request.canonical_task_id
                },
            )
            assert update["updated"] is True
            fixture.daemon._record_event(
                "task_completed",
                {
                    "task_id": fixture.request.task_id,
                    "reason": "task_became_completed",
                    "completion_receipt_repair": False,
                },
            )
            fixture.daemon._record_event(
                "daemon_pass",
                {
                    "active_task_id": "",
                    "completed_count": 1,
                    "ready_count": 0,
                    "selection_idle_reason": "database_pending_merge_reconciliation",
                },
            )
        failed = owner.commit_phase(attempt, "context")
        failed = owner.commit_phase(
            failed,
            "failed",
            body={
                "reason": terminal_reason,
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        owner._persist_terminal_portal_failure(
            failed,
            reason=terminal_reason,
            coordination_evidence=owner._reconcile_failed_attempt_coordination(failed),
        )
        before = owner.task_source.get(attempt.task_cid)
        assert before.status == "blocked"
        owner._merge_queue = fixture.daemon.merge_queue
        owner._merge_repo_root = fixture.repo
        owner._merge_target_branch = fixture.daemon.resolved_merge_target_branch
        owner._merge_portal_attempt_root = fixture.bridge.attempt_root
        cas_calls, evidence_calls = [], []
        cas_name = (
            "recover_post_merge_retry"
            if terminal_reason
            == DATABASE_PORTAL_COMPLETION_CALLBACK_BINDING_INVALID_REASON
            and callable(getattr(owner.task_source, "recover_post_merge_retry", None))
            else "record_queue_backoff_and_cas_status"
        )
        original_cas = getattr(owner.task_source, cas_name)
        original_recover = owner.recover_blocked_post_merge_declared_outputs

        def observe_cas(**kwargs):
            cas_calls.append(kwargs)
            result = original_cas(**kwargs)
            if lose_cas_response:
                raise RuntimeError("fixture committed CAS response lost")
            return result

        def observe_recover(evidence):
            evidence_calls.append(evidence)
            return original_recover(evidence)

        monkeypatch.setattr(owner.task_source, cas_name, observe_cas)
        monkeypatch.setattr(
            owner, "recover_blocked_post_merge_declared_outputs", observe_recover
        )
        if lose_cas_response:
            with pytest.raises(
                RuntimeError, match="fixture committed CAS response lost"
            ):
                fixture.bridge.recover_post_merge_declared_outputs(owner)
            result = original_recover(evidence_calls[0])
        else:
            result = fixture.bridge.recover_post_merge_declared_outputs(owner)
        assert result and result["recovered"] is True, result
        after = owner.task_source.get(attempt.task_cid)
        assert after.status == "retrying" and after.revision == before.revision + 1
        assert len(cas_calls) == 1 and len(evidence_calls) == 1
        assert (
            cas_calls[0]["expected_control_receipt"]
            == before.body["completion_receipt"]
        )
        replay = original_recover(evidence_calls[0])
        assert replay["recovered"] is True and replay["changed"] is False
        assert replay["write_count"] == 0 and len(cas_calls) == 1
        assert owner.list_running_attempts() == []
    finally:
        owner.close()
