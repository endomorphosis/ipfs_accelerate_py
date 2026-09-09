"""Exact receiver failures use retained callback reconciliation, never generic retry."""

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_PORTAL_COMPLETION_SOURCE_KEY_MISMATCH_REASON as REASON,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
    DatabaseImplementationConflictError,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _callback_integration_recovery_evidence,
    _open_daemon,
    _population,
)


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "foreign_claim",
        "foreign_task",
        "absent_history",
        "foreign_history",
        "active_attempt",
        "foreign_reason",
    ],
)
def test_exact_terminal_rearms_only_retained_callback(tmp_path, monkeypatch, mutation):
    calls = []
    daemon = _open_daemon(tmp_path, provider_calls=calls, effect_calls=calls)
    try:
        daemon.materialize_population(_population(1))
        source = daemon.claim_next()
        source = daemon.commit_phase(source, "context")
        if mutation == "active_attempt":
            evidence = _callback_integration_recovery_evidence(daemon, source)
            monkeypatch.setattr(
                daemon,
                "_verified_post_merge_callback_integration_receipt",
                lambda raw, **kwargs: dict(raw),
            )
            with pytest.raises(
                (
                    DatabaseImplementationAuthorityError,
                    DatabaseImplementationConflictError,
                )
            ):
                daemon.recover_blocked_post_merge_declared_outputs(evidence)
            assert daemon.list_running_attempts()
            assert calls == []
            return
        reason = "foreign terminal" if mutation == "foreign_reason" else REASON
        source = daemon.commit_phase(
            source,
            "failed",
            body={
                "reason": reason,
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        daemon._persist_terminal_portal_failure(
            source,
            reason=reason,
            coordination_evidence=daemon._reconcile_failed_attempt_coordination(source),
        )
        evidence = _callback_integration_recovery_evidence(daemon, source)
        # Git/event receipt verification has independent real-chain/negative tests.
        # This test retains real task/history/coordination and queue CAS authority.
        monkeypatch.setattr(
            daemon,
            "_verified_post_merge_callback_integration_receipt",
            lambda raw, **kwargs: dict(raw),
        )
        before = daemon.task_source.get(source.task_cid)
        if mutation == "none":
            # Actual generic recovery selection must leave this terminal alone.
            daemon._persist_task_retry_state = lambda *a, **k: pytest.fail(
                "generic handshake retry"
            )
            daemon.reconcile_terminal_portal_failures()
            assert daemon.task_source.get(source.task_cid).status == "blocked"
            from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
                DatabasePortalBridgeError,
                DatabasePortalExecutionBridge,
            )

            bridge = object.__new__(DatabasePortalExecutionBridge)
            bridge.task_source = daemon.task_source
            bridge.implementation_timeout = 1
            bridge._record_for_attempt = lambda *args: before
            bridge._execution_route_binding = lambda **kwargs: pytest.fail(
                "provider dispatch setup reached"
            )
            with pytest.raises(
                DatabasePortalBridgeError, match="requires exact source seed"
            ):
                bridge.run_provider(source)
        assert before.status == "blocked"
        assert not daemon._is_post_merge_declared_outputs_missing_terminal(
            source, before
        )
        assert daemon._is_portal_completion_evaluated_baseline_missing_terminal(
            source, before
        ) == (mutation != "foreign_reason")
        if mutation == "foreign_claim":
            evidence["source_claim_id"] = "claim:foreign"
        elif mutation == "foreign_task":
            evidence["task_cid"] = "task:foreign"
        elif mutation in ("absent_history", "foreign_history"):
            original = daemon.task_source.task_revision_history_projection

            def history(cid):
                from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
                    content_identity,
                )

                value = dict(original(cid))
                rows = value["revisions"]
                if mutation == "absent_history":
                    value["revisions"] = [
                        row for row in rows if row["revision"] != before.revision
                    ]
                else:
                    value["revisions"] = [
                        {
                            **row,
                            "body": {
                                **row["body"],
                                "completion_receipt": {
                                    **row["body"].get("completion_receipt", {}),
                                    "claim_id": "claim:foreign",
                                },
                            },
                        }
                        if row["revision"] == before.revision
                        else row
                        for row in rows
                    ]
                value.pop("projection_cid")
                value["projection_cid"] = content_identity(value)
                return value

            monkeypatch.setattr(
                daemon.task_source, "task_revision_history_projection", history
            )
        evidence["evidence_id"] = daemon._database_portal_evidence_digest(
            {k: v for k, v in evidence.items() if k != "evidence_id"}
        )
        if mutation != "none":
            with pytest.raises(
                (
                    DatabaseImplementationAuthorityError,
                    DatabaseImplementationConflictError,
                )
            ):
                daemon.recover_blocked_post_merge_declared_outputs(evidence)
        else:
            assert daemon.post_merge_completion_recovery_task_cids() == (
                source.task_cid,
            )
            result = daemon.recover_blocked_post_merge_declared_outputs(evidence)
            assert result["recovered"] and result["changed"]
            current = daemon.task_source.get(source.task_cid)
            assert current.status == "retrying"
            seed = current.body["completion_receipt"][
                "post_merge_completion_recovery_seed"
            ]
            assert seed["terminal_reason"] == REASON
            assert seed["attempt_id"] == source.attempt_id
            assert seed["claim_id"] == source.claim_id
            assert seed["candidate_commit"] == evidence["candidate_commit"]
            assert seed["qualification_kind"] == "callback_integration"
            assert (
                daemon._post_merge_completion_terminal_receipt_from_history(
                    attempt=source, seed=seed
                )
                == before.body["completion_receipt"]
            )
        assert calls == []
    finally:
        daemon.close()
