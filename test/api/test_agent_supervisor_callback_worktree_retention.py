"""Older native runtimes retain callback evidence without a cleanup verifier."""
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import PortalImplementationSupervisor


def test_bound_portal_does_not_enter_peer_cleanup(tmp_path):
    workspace = tmp_path / "unknown-callback"
    workspace.mkdir()
    source = workspace / "candidate.py"
    source.write_text("unsettled effect evidence\n")
    daemon = object.__new__(PortalImplementationDaemon)
    daemon._database_attempt_authority = {"database_attempt_id": "attempt:one"}
    # Any access to pool/ancestry settings would fail: the guard precedes them.
    result = daemon._cleanup_already_merged_worktrees()
    assert result["reason"] == "canonical_cleanup_requires_guarded_runtime"
    assert result["removed_count"] == 0
    assert source.read_text() == "unsettled effect evidence\n"


def test_database_supervisor_retains_until_guarded_runtime(tmp_path):
    workspace = tmp_path / "unknown-callback"
    workspace.mkdir()
    source = workspace / "candidate.py"
    source.write_text("unsettled effect evidence\n")
    supervisor = object.__new__(PortalImplementationSupervisor)
    supervisor.config = SimpleNamespace(database_program=object())
    # No lease or Git method is available; database cleanup must stop first.
    result = supervisor.cleanup_backlogged_worktrees()
    assert result["reason"] == "canonical_cleanup_requires_guarded_runtime"
    assert result["removed_count"] == 0
    assert source.read_text() == "unsettled effect evidence\n"
