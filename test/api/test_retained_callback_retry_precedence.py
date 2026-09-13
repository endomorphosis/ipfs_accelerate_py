"""Retained callback denials cannot become unseeded ordinary provider work."""
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import TASK_REVISION_HISTORY_PROJECTION_SCHEMA
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalBridgeError, DatabasePortalExecutionBridge
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population

CALLBACK_REASON = "Portal callback reconciliation binding is invalid"
KEY_REASON = "Portal completion source canonical task key mismatches"


@pytest.mark.parametrize("dedicated_callback", [False, True])
def test_rejected_callback_stays_for_dedicated_recovery(tmp_path, dedicated_callback):
    calls = []
    def provider(attempt):
        calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError(CALLBACK_REASON)
    daemon = _open_daemon(tmp_path, session="session:callback-denial",
                          provider_fn=provider, max_task_attempts=4)
    if dedicated_callback:
        daemon._post_merge_recovery_fn = lambda: None
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        attempt = daemon.get_attempt(result["attempt_id"])
        task = daemon.task_source.get(attempt.task_cid)
        assert task.status == "blocked"
        assert task.body["completion_receipt"]["reason"] == CALLBACK_REASON
        assert daemon.reconcile_terminal_portal_failures() == []
        current = daemon.task_source.get(attempt.task_cid)
        assert current.revision == task.revision
        assert current.body == task.body
        daemon.run_once()
        assert calls == [attempt.attempt_id]
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
    finally:
        daemon.close()


@pytest.mark.parametrize("reason", [CALLBACK_REASON, KEY_REASON])
@pytest.mark.parametrize("mutation", ["missing", "foreign_identity", "foreign_task", "wrong_reason", None])
def test_dispatch_requires_exact_retained_source_seed(reason, mutation):
    source = {"operation": "database_portal_terminal_failure", "reason": reason,
              "attempt_id": "attempt:original", "claim_id": "claim:original",
              "lease_id": "lease:original", "owner_session_id": "owner:original",
              "attempt_number": 1, "fencing_token": 1, "fence_epoch": 1}
    history = {"schema": TASK_REVISION_HISTORY_PROJECTION_SCHEMA, "task_cid": "task:one",
               "revisions": [{"revision": 4, "status": "blocked", "body": {"completion_receipt": source}}]}
    history["projection_cid"] = content_identity(history)
    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.task_source = SimpleNamespace(task_revision_history_projection=lambda cid: history)
    seed = {k: source[k] for k in ("attempt_id", "claim_id", "lease_id", "owner_session_id", "attempt_number", "fencing_token", "fence_epoch")}
    seed.update(terminal_reason=reason, task_cid="task:one")
    if mutation == "foreign_identity": seed["claim_id"] = "claim:other"
    if mutation == "foreign_task": seed["task_cid"] = "task:other"
    if mutation == "wrong_reason": seed["terminal_reason"] = KEY_REASON if reason == CALLBACK_REASON else CALLBACK_REASON
    record = SimpleNamespace(body={"completion_receipt": {} if mutation == "missing" else {"post_merge_completion_recovery_seed": seed}})
    attempt = SimpleNamespace(task_cid="task:one")
    if mutation:
        with pytest.raises(DatabasePortalBridgeError, match="exact source seed before dispatch"):
            bridge._require_retained_completion_key_seed(attempt=attempt, record=record)
    else:
        # This clears only a denial guard; full seed/queue/CAS admission follows.
        bridge._require_retained_completion_key_seed(attempt=attempt, record=record)
