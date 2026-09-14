"""Already-terminal legacy conflicts retain custody without callback replay."""

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import owner_task_quarantine as local
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.owner_task_quarantine import (
    QuarantineDenied,
)
from test.api.test_owner_task_quarantine_custody import legacy_finalizer_conflict


@pytest.mark.parametrize("field,value", [
    ("database_disposition", "superseded_attempt_revoked"),
    ("intended_database_disposition", "terminalized_for_retry"),
    ("database_attempt_status", "superseded"),
    ("database_attempt_phase", "context"),
    ("terminal_reconciliation_evidence_id", "unrelated-evidence"),
    ("prepared_reconciliation_receipt_id", "sha256:" + "0" * 64),
    ("trigger", "unrelated-recovery"),
    ("reconciled_at", "2020-01-01T00:00:00+00:00"),
    ("blocked", 0),  # Type-sensitive equality: False and 0 differ.
    ("terminal_provider_evidence", True),
])
def test_self_identified_terminal_receipt_cannot_change_quarantine_basis(
    tmp_path, monkeypatch, field, value,
):
    """Valid receipt hashing alone never establishes the retained conflict."""
    _, daemon, _, attempt, paths = legacy_finalizer_conflict(
        tmp_path, monkeypatch, terminal=True,
        change_payload=lambda payload: payload.update({field: value}),
    )
    try:
        before = local.execution_snapshot(daemon._require_connection(), attempt.task_cid)
        files = {p.name: p.read_bytes() for p in paths.reconciliation.iterdir()}
        with pytest.raises((QuarantineDenied, DatabasePortalBridgeError)):
            local.capture(daemon, attempt)
        assert local.execution_snapshot(daemon._require_connection(), attempt.task_cid) == before
        assert {p.name: p.read_bytes() for p in paths.reconciliation.iterdir()} == files
    finally:
        daemon.close()


@pytest.mark.parametrize("damage", ["missing", "changed", "symlink"])
def test_terminal_receipt_must_remain_available_and_exact(tmp_path, monkeypatch, damage):
    _, daemon, _, attempt, paths = legacy_finalizer_conflict(
        tmp_path, monkeypatch, terminal=True,
    )
    try:
        before = local.capture(daemon, attempt)
        saga = daemon._database_portal_terminal_reconciliation_saga(attempt)
        receipt = paths.reconciliation / (saga["receipt_id"][7:] + ".json")
        original = receipt.read_bytes()
        if damage == "missing":
            receipt.unlink()
        elif damage == "symlink":
            preserved = tmp_path / "foreign-receipt.json"
            preserved.write_bytes(original)
            receipt.unlink()
            receipt.symlink_to(preserved)
        else:
            body = json.loads(original)
            body["database_disposition"] = "superseded_attempt_revoked"
            receipt.write_text(json.dumps(body))
        with pytest.raises(DatabasePortalBridgeError):
            local.capture(daemon, attempt)
        assert local.execution_snapshot(daemon._require_connection(), attempt.task_cid) == before["execution"]
        assert daemon.get_attempt(attempt.attempt_id) == attempt
    finally:
        daemon.close()
