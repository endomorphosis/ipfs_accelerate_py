from types import SimpleNamespace
import hashlib
import json
import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_SCHEMA, TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_OPERATION, TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_FIELD

@pytest.mark.parametrize("tamper", ["", "history", "budget", "source", "task_revision", "obligation"])
def test_operator_rearm_suppresses_only_exact_historical_budget(tamper):
    daemon = object.__new__(DatabaseImplementationDaemon)
    attempt = SimpleNamespace(attempt_id="attempt:79", claim_id="claim:79", attempt_number=79, task_cid="task:x")
    budget = {"exhausted": True, "observation_id": "observed"}
    terminal = {"attempt_id": attempt.attempt_id, "retry_budget": budget}
    digest = "sha256:" + hashlib.sha256(json.dumps(terminal,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    receipt = {"schema": TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_SCHEMA, "operation": TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_OPERATION, "terminal_operation": "database_portal_typed_deferral_budget_exhausted", "attempt_id": attempt.attempt_id, "claim_id": attempt.claim_id, "attempt_number": 79, "fresh_attempt_number": 80, "recovered_from_revision": 28, "attempt_refunded": False, "source_completion_receipt_id": digest, "operator_handoff_receipt_id": "operator", "sidecar_evidence_id": "sidecar"}
    requirement = {"source_completion_receipt_id": digest, "operator_handoff_receipt_id": "operator", "sidecar_evidence_id": "sidecar"}
    task = SimpleNamespace(revision=29, body={"completion_receipt": receipt, TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_FIELD: requirement})
    history = [{"revision": 28, "status": "blocked", "body": {"completion_receipt": terminal}}]
    daemon._requires_fresh_portal_revalidation=lambda t: tamper != "obligation"
    daemon._task_revision_history_for_recovery=lambda cid: history
    if tamper == "history": history[0]["status"]="completed"
    if tamper == "budget": budget={"exhausted":True,"observation_id":"different"}
    if tamper == "source": receipt["source_completion_receipt_id"]="foreign"
    if tamper == "task_revision": task.revision=30
    assert daemon._operator_typed_deferral_recovery_is_admitted(attempt,task,budget) is (not tamper)
