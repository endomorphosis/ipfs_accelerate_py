"""Native owner recovery admits retained candidates without blind provider retry."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    probe_quack_capabilities,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
)
from test.api.test_agent_supervisor_quack_owner_mutation import (
    _legacy_orphan_database_path,
    _seed_legacy_orphan_landed_attempt,
    _typed_owner_completion_state,
    _typed_task_source,
)


@pytest.mark.parametrize(
    "case",
    [
        "exact",
        "alive",
        "unknown",
        "foreign_lane",
        "wrong_attempt",
        "stale_revision",
        "changed_receipt",
        "missing_history",
        "no_candidate",
        "bad_candidate",
        "cas_race",
    ],
)
def test_native_dead_admitted_candidate_rearm(tmp_path, monkeypatch, case):
    database = _legacy_orphan_database_path(tmp_path)
    client_id = "database-implementation-daemon:retained-candidate-lane"
    fixture = _seed_legacy_orphan_landed_attempt(
        database, client_id=client_id, source_status="in_progress"
    )
    server = build_server(
        database_path=database,
        state_dir=fixture["state_root"] / "candidate-owner",
        store_id="candidate-owner-v1",
        repository_id="repository:test",
        repository_root=fixture["repository_root"],
        transport=FakeQuackTransport(),
        capability_probe=lambda **kwargs: probe_quack_capabilities(),
        owner_liveness_probe=lambda birth: module.OwnerLiveness.DEAD,
    )
    identity = server.start()
    source = _typed_task_source(
        server,
        identity,
        monkeypatch,
        client_id=client_id,
        allowed_command_operations=(
            "task.status.cas.receipt",
            "task.retry.cooldown.record",
        ),
    )
    try:
        before = source.get_task("task:test")
        source.quarantine_dead_admitted_provider_outcome_unknown(
            before.task_cid,
            expected_task_revision=before.revision,
            expected_control_receipt=fixture["source_receipt"],
        )
        task = source.get_task("task:test")
        receipt = task.body["completion_receipt"]
        attempt = DatabaseTaskAttempt(
            task_cid=task.task_cid,
            task_alias=task.task_alias,
            **{
                k: receipt[k]
                for k in (
                    "attempt_id",
                    "claim_id",
                    "lease_id",
                    "owner_session_id",
                    "attempt_number",
                    "fencing_token",
                    "fence_epoch",
                )
            },
            committed_phase=module.ATTEMPT_PHASE_FAILED,
            status="failed",
            started_at_ms=0,
        )
        daemon = object.__new__(DatabaseImplementationDaemon)
        daemon.open = lambda: daemon
        daemon._task_source = source
        # The legacy fixture predates launch-policy sealing. Keep its exact
        # admitted route fixed while exercising the real native CAS protocol.
        route = fixture["admitted_receipt"]["execution_route_binding"]
        source._execution_route_policy = SimpleNamespace()

        def validate_fixture_route(value, *, task, allow_claim_revision=False):
            assert allow_claim_revision is True
            assert dict(value) == route
            assert (
                dict(task.body["completion_receipt"]["execution_route_binding"])
                == route
            )
            return dict(route)

        monkeypatch.setattr(
            source, "validate_execution_route_binding", validate_fixture_route
        )
        daemon.max_task_attempts = 1
        daemon.get_attempt = lambda attempt_id: (
            attempt if attempt_id == attempt.attempt_id else None
        )
        daemon._coordinator = SimpleNamespace(get_task_claim=lambda claim_id: None)
        recorded = []
        daemon._record_event = lambda *args, **kwargs: recorded.append((args, kwargs))
        liveness = {
            "alive": module.OwnerLiveness.ALIVE,
            "unknown": module.OwnerLiveness.UNKNOWN,
            "foreign_lane": None,
        }.get(case, module.OwnerLiveness.DEAD)
        # Exercise the production birth observer; only the OS liveness seam is replaced.
        monkeypatch.setattr(module, "owner_liveness", lambda birth: liveness)
        if case == "foreign_lane":
            monkeypatch.setattr(
                daemon,
                "_typed_claim_process_attestation",
                lambda: {**source.claim_process_attestation(), "client_id": "foreign"},
            )
        seed = {
            "schema": "ipfs_accelerate_py/agent-supervisor/database-portal-post-commit-candidate-recovery@1",
            "disposition": "retry_exact_post_commit_candidate",
            "reason": "process_lost_before_merge_queue_publication",
            "task_cid": attempt.task_cid,
            "task_alias": attempt.task_alias,
            **{
                k: receipt[k]
                for k in (
                    "attempt_id",
                    "claim_id",
                    "lease_id",
                    "attempt_number",
                    "fencing_token",
                    "fence_epoch",
                )
            },
            "source_task_revision": before.revision,
            "portal_attempt": 1,
            "baseline_commit": "a" * 40,
            "implementation_commit": "b" * 40,
            "preserved_commit": "b" * 40,
            "rescue_branch": "implementation/retained",
            "original_branch": "implementation/retained",
            "original_worktree_path": str(tmp_path / "retained"),
            "source_workspace_disposition": "candidate_head",
            "source_workspace_observed_head": "b" * 40,
            "source_workspace_observed_tree": "c" * 40,
            "source_workspace_observed_branch": "implementation/retained",
            "final_tree": "c" * 40,
            "candidate_fingerprint": "sha256:" + "1" * 64,
            "binding_id": "sha256:" + "2" * 64,
            "events_digest": "sha256:" + "3" * 64,
            "event_stream_id": "stream:retained",
            "implementation_started_event_id": "sha256:" + "4" * 64,
            "pre_commit_handoff_event_id": "sha256:" + "5" * 64,
            "post_commit_handoff_event_id": "sha256:" + "6" * 64,
            "attempt_consumed": True,
            "provider_dispatched": True,
            "completion_authoritative": False,
            "merge_attempted": False,
        }
        seed["receipt_id"] = module._database_daemon_evidence_digest(seed)
        recovered = []

        def recover(value):
            recovered.append(value.attempt_id)
            if case == "no_candidate":
                return {
                    "schema": "ipfs_accelerate_py/agent-supervisor/database-portal-callback-no-effect-recovery@1"
                }
            if case == "bad_candidate":
                return {**seed, "implementation_commit": "d" * 40}
            if case == "cas_race":
                # A fresh native history observation replaces the caller's stale view.
                source.compare_and_set_status(
                    task.task_cid,
                    task.revision,
                    "blocked",
                    {"operation": "test-concurrent-block"},
                )
            return seed

        daemon._post_commit_candidate_recovery_fn = recover
        if case == "wrong_attempt":
            attempt = replace(attempt, claim_id="claim:other")
        if case == "stale_revision":
            task = replace(task, revision=task.revision - 1)
        if case == "changed_receipt":
            body = dict(task.body)
            body["completion_receipt"] = {
                **dict(task.body["completion_receipt"]),
                "retry_suppressed": False,
            }
            task = replace(task, body=body)
        if case == "missing_history":
            monkeypatch.setattr(
                source,
                "task_revision_history_projection",
                lambda task_cid: {"revisions": []},
            )
        native_before = _typed_owner_completion_state(server._connection)
        if case in {"no_candidate", "bad_candidate"}:
            result = daemon._reopen_unimplemented_unknown_callback_task(task)
            assert result["reopened"] is False
            assert result["reason"] == "post_commit_recovery_evidence_rejected"
        else:
            result = daemon._reopen_unimplemented_unknown_callback_task(task)
            if case == "exact":
                assert result["reason"] == "exact_post_commit_candidate_rearmed"
                assert result["provider_dispatched"] is False
                assert result["attempt_consumed"] is False
                updated = source.get_task(task.task_cid)
                assert updated.status == "retrying"
                assert updated.revision == task.revision + 1
                assert (
                    updated.body["completion_receipt"][
                        "post_commit_candidate_recovery_seed"
                    ]
                    == seed
                )
                assert source.get_queue_entry(task.task_cid) is not None
                assert (
                    daemon._reopen_unimplemented_unknown_callback_task(updated) is None
                )
            else:
                assert not result["reopened"]
        if case not in {"exact", "cas_race"}:
            assert _typed_owner_completion_state(server._connection) == native_before
        if case not in {"exact", "no_candidate", "bad_candidate", "cas_race"}:
            assert recovered == []
        assert (
            server._connection.execute(
                "SELECT COUNT(*) FROM completion_receipts"
            ).fetchone()[0]
            == 0
        )
    finally:
        source.close()
        server.stop()
