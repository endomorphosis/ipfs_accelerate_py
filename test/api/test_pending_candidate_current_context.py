"""Current intent binds identities even with historical Native empty-cache events."""
from dataclasses import asdict
import os
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import pending_task_claim_recovery as recovery
from test.api.test_pending_task_claim_recovery import pending

pytestmark = pytest.mark.parametrize("pending", ["candidate"], indirect=True)


def no_enrichment(daemon, monkeypatch, *, damage=None):
    daemon._task_identity_by_display_id.clear()
    actual = daemon._record_event
    if os.environ.get("TEST_ACTUAL_LEGACY_NATIVE") == "1":
        path = Path(native.__file__).resolve(); root = path.parents[3]
        assert path.read_bytes() == subprocess.check_output(["git", "show",
            "6af2ffd429eb0902f66064a55ac76f18c54498d2:" + path.relative_to(root).as_posix()], cwd=root)
        if damage is None: return
    def projection(kind, value, **kwargs):
        if kind == "implementation_pending_candidate_preserved":
            value = {k: v for k, v in value.items() if k not in
                ("canonical_task_cid", "canonical_task_key", "board_namespace")}
            evidence = dict(value["pending_candidate_recovery"])
            if damage in {"canonical_task_cid", "canonical_task_key", "board_namespace", "operation_id", "candidate_manifest_sha256"}:
                evidence[damage] = "foreign"
            elif damage == "false_flag": evidence["completion_authority"] = True
            elif damage == "missing": evidence.pop("canonical_task_cid")
            elif damage == "extra": evidence["unreviewed_authority"] = False
            elif damage == "contradictory_top": value["canonical_task_cid"] = "foreign"
            value["pending_candidate_recovery"] = evidence
        return actual(kind, value, **kwargs)
    monkeypatch.setattr(daemon, "_record_event", projection)


def test_current_closed_intent_handles_unenriched_native_checkpoint_without_bridge(pending, monkeypatch):
    daemon, task, workspace, claim = pending
    no_enrichment(daemon, monkeypatch)
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    result = recovery.reconcile_recovery(daemon, request=request)
    assert result["reconciled"] and not result["blocked"] and not claim.exists() and not workspace.exists(), result
    assert all(result[k] is False for k in recovery._FALSE)
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    event = next(row for row in daemon._iter_events() if row["type"] == "implementation_pending_candidate_preserved")
    assert not any(k in event for k in ("canonical_task_cid", "canonical_task_key", "board_namespace"))
    base = request["base"]; evidence = event["pending_candidate_recovery"]
    assert evidence["operation_id"] == "sha256:" + recovery._sha(recovery._bytes(request))
    assert evidence["canonical_task_cid"] == base["task_cid"]
    assert evidence["canonical_task_key"] == base["task_key"] and evidence["board_namespace"] == base["board_namespace"]
    assert recovery.reconcile_recovery(daemon, request=recovery.review_recovery(daemon, task_id=task.task_id)) == result


@pytest.mark.parametrize("damage", ["canonical_task_cid", "canonical_task_key", "board_namespace", "operation_id",
    "candidate_manifest_sha256", "false_flag", "missing", "extra", "contradictory_top"])
def test_absent_top_identity_never_authorizes_malformed_inner_context(pending, monkeypatch, damage):
    daemon, task, workspace, claim = pending
    no_enrichment(daemon, monkeypatch, damage=damage)
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    result = recovery.reconcile_recovery(daemon, request=request)
    assert result["blocked"] and not result["reconciled"] and claim.exists(), result
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    rows = [row for row in daemon._iter_events() if row["type"] == "implementation_pending_candidate_preserved"]
    assert len(rows) == 1
    preserved = rows[0]
    assert recovery._git(daemon.repo_root, "rev-parse", preserved["rescue_branch"]) == preserved["preserved_commit"]
    assert recovery._candidate_prior(recovery._candidate_receipt_path(daemon, task.task_id))["phase"] == "prepared"


def test_later_proposal_for_same_attempt_cannot_rebind_preservation(pending, monkeypatch):
    daemon, task, _, claim = pending
    daemon._task_identity_by_display_id.clear()
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    actual = daemon._record_event
    def superseding(kind, value, **kwargs):
        if kind == "implementation_pending_candidate_preserved":
            actual("implementation_proposal_validated", {"task_id": task.task_id,
                "canonical_task_cid": request["base"]["task_cid"], "canonical_task_key": request["base"]["task_key"],
                "attempt": 1, "accepted": True, "attempted": True, "reason_codes": [], "proposal_id": "f" * 64})
        return actual(kind, value, **kwargs)
    monkeypatch.setattr(daemon, "_record_event", superseding)
    result = recovery.reconcile_recovery(daemon, request=request)
    assert result["blocked"] and result["reason"] == "candidate_proposal_superseded" and claim.exists(), result
