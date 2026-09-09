from copy import deepcopy
from types import SimpleNamespace
import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.callback_terminal import is_expired_callback_quarantine_phase


@pytest.mark.parametrize("drift", [None, "claim", "task", "evidence", "preparation", "revision", "fence", "finished", "extra_phase", "missing_phase"])
def test_expired_callback_terminal_requires_exact_reconciliation(drift):
    attempt = SimpleNamespace(attempt_id="attempt:1", claim_id="claim:1", task_cid="task:1",
        status="failed", committed_phase="failed", revision=3, fencing_token=7,
        fence_epoch=2, finished_at_ms=7000)
    terminal = {"phase": "failed", "revision": 3, "fencing_token": 7, "fence_epoch": 2,
        "committed_at_ms": 7000, "body": {"cross_store_reconciled": True,
        "preparation_digest": "", "reconciliation": {"attempt_id": "attempt:1",
        "claim_id": "claim:1", "task_cid": "task:1", "disposition": "quarantined",
        "effect_evidence_reused": False, "provider_evidence_reused": False,
        "provider_invocation_receipt_present": True, "reason": "portal_neutral_failure",
        "retry_required": False, "status": "failed"}}}
    phases = [{"phase": "claimed"}, {"phase": "context"}, deepcopy(terminal)]
    row = phases[-1]
    if drift == "claim": row["body"]["reconciliation"]["claim_id"] = "claim:other"
    if drift == "task": row["body"]["reconciliation"]["task_cid"] = "task:other"
    if drift == "evidence": row["body"]["reconciliation"]["effect_evidence_reused"] = True
    if drift == "preparation": row["body"]["preparation_digest"] = "sha256:other"
    if drift == "revision": row["revision"] += 1
    if drift == "fence": row["fencing_token"] += 1
    if drift == "finished": row["committed_at_ms"] += 1
    if drift == "extra_phase": phases.insert(2, {"phase": "effect"})
    if drift == "missing_phase": phases.pop(1)
    assert is_expired_callback_quarantine_phase(attempt, phases) is (drift is None)
