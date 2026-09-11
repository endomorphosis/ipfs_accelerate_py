"""A blocked local recovery must leave another lane's real writer available."""
from pathlib import Path
import subprocess
import sys

import pytest
import duckdb

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon_module
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
from test.api.test_agent_supervisor_pctdd005_successor_retry_v3 import (
    _lease_scoped_reconciliation_supervisor,
    _retained_controller_cleanup,
)


def _peer(root):
    return DatabaseImplementationDaemon(
        database_path=root / "control.duckdb",
        coordination_path=root / "coordination.duckdb",
        execution_path=root / "execution.duckdb",
        owner_session_id="peer-lane",
        authority_mode="embedded",
        task_source_kind="duckdb",
    )


def _independent_peer_writer(root):
    source = Path(__file__).resolve().parents[2]
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.insert(1, sys.argv[3])
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
p = Path(sys.argv[2])
d = DatabaseImplementationDaemon(database_path=p/'control.duckdb',
    coordination_path=p/'coordination.duckdb', execution_path=p/'execution.duckdb',
    owner_session_id='independent-peer-writer', authority_mode='embedded', task_source_kind='duckdb')
d.close()
"""
    duckdb_site = Path(duckdb.__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, "-I", "-c", code, str(source), str(root), str(duckdb_site)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("local_result", [
    {"reconciled": False, "blocked": True, "reason": "unknown_callback"},
    {"reconciled": False, "blocked": False, "reason": "continuation_required"},
    {"reconciled": False, "blocked": False, "repair_batch_pending": True},
    {"reconciled": True, "blocked": False, "continuation_required": True,
     "reason": "database_portal_post_cas_transitions_reconciled"},
    {"reconciled": True, "blocked": False, "continuation_required": True,
     "safe_to_restart": False},
])
def test_local_recovery_does_not_reserve_peer_writer(tmp_path, monkeypatch, local_result):
    peer_root = tmp_path / "peer"
    _peer(peer_root).close()
    supervisor, program, events, daemons, _ = _lease_scoped_reconciliation_supervisor(tmp_path, monkeypatch)
    opened = []

    def bind(**_kwargs):
        peer = _peer(peer_root)
        opened.append(peer)
        return [peer]

    def local(_self, **_kwargs):
        _independent_peer_writer(peer_root)
        return dict(local_result)

    monkeypatch.setattr(supervisor, "_bind_retained_recovery_lane_attempt_authorities", bind)
    monkeypatch.setattr(daemon_module.DatabaseImplementationDaemon,
                        "reconcile_quiesced_database_portal_attempts", local)
    monkeypatch.setattr(daemon_module.DatabaseImplementationDaemon,
                        "_database_portal_terminal_repair_cursor", lambda _self: {}, raising=False)
    result = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program, owner_fence_held=True, managed_daemon_launch_lock_held=True,
        managed_daemon_cleanup=_retained_controller_cleanup())
    assert opened == []
    assert result["reconciled"] is local_result["reconciled"]
    if local_result.get("continuation_required"):
        assert result == local_result
    assert not any(event in events for event in ("pctdd005_orphan", "retained_orphan"))
    _independent_peer_writer(peer_root)


@pytest.mark.parametrize("fail_at", [None, "retained_occurrence"])
def test_admitted_recovery_opens_peer_only_after_local_pass_and_releases_it(tmp_path, monkeypatch, fail_at):
    peer_root = tmp_path / "peer"
    _peer(peer_root).close()
    supervisor, program, events, daemons, _ = _lease_scoped_reconciliation_supervisor(
        tmp_path, monkeypatch, fail_at=fail_at)
    opened = []
    local_complete = []

    def local(_self, **_kwargs):
        _independent_peer_writer(peer_root)
        local_complete.append(True)
        return {"reconciled": True, "blocked": False}

    def bind(**_kwargs):
        assert local_complete == [True]
        peer = _peer(peer_root)
        opened.append(peer)
        return [peer]

    monkeypatch.setattr(supervisor, "_bind_retained_recovery_lane_attempt_authorities", bind)
    monkeypatch.setattr(daemon_module.DatabaseImplementationDaemon,
                        "reconcile_quiesced_database_portal_attempts", local)
    result = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program, owner_fence_held=True, managed_daemon_launch_lock_held=True,
        managed_daemon_cleanup=_retained_controller_cleanup())
    assert len(opened) == 1
    assert result["reconciled"] is (fail_at is None)
    assert opened[0]._connection is None
    assert opened[0]._embedded_writer_lock_handle is None
    _independent_peer_writer(peer_root)
