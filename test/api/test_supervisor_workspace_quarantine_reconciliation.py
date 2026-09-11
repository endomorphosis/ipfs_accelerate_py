"""Independent normal-native cleanup route must retain a frozen workspace."""
import threading

import pytest

from test.api.test_agent_supervisor_callback_worktree_retention import completed_workspace as _completed_workspace
from test.api.test_agent_supervisor_reconciliation_auto_unblock import _git
from ipfs_accelerate_py.agent_supervisor.merge import workspace_quarantine as q
from ipfs_accelerate_py.agent_supervisor.merge.quarantine_validation import QuarantineDenied


completed_workspace = _completed_workspace


def test_nested_supervisor_rescue_preserves_parent_frozen_workspace(tmp_path):
    from test.api.test_workspace_root_quarantine import nested_repository
    from test.api.test_agent_supervisor_reconciliation_auto_unblock import _supervisor

    repo, root, module = nested_repository(tmp_path)
    workspace = root / "nested-native-workspace"
    _git(module, "worktree", "add", "-b", "attempt/nested-retained", str(workspace), "HEAD")
    supervisor = _supervisor(module, worktree_root=root)
    (workspace / "README").write_bytes(b"retained unknown nested callback output\n")

    def preimage():
        return (
            _git(workspace, "branch", "--show-current"),
            _git(workspace, "rev-parse", "HEAD"),
            _git(workspace, "status", "--porcelain"),
            (workspace / "README").read_bytes(),
        )

    branch, head, status, _ = before = preimage()
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        supervisor._rescue_dirty_worktree(
            workspace, branch=branch, head=head, target_ref="HEAD",
            status_lines=status.splitlines(), reason="retained callback custody",
        )
    assert preimage() == before
    assert q.verify(repo, root) == frozen

def test_completed_rescue_prune_cannot_remove_workspace_inside_frozen_root(completed_workspace):
    case = completed_workspace
    head = _git(case.repo, 'rev-parse', case.branch)
    frozen = q.freeze(case.repo, case.workspace.parent, expected=q.census(case.repo, case.workspace.parent))
    try:
        result = case.supervisor._prune_completed_leftover_worktree(case.workspace, case.branch, expected_head=head)
    except QuarantineDenied:
        result = {'removed': False}
    assert not result.get('removed'), result
    assert case.workspace.is_dir()
    assert q.verify(case.repo, case.workspace.parent) == frozen


def test_dirty_rescue_cannot_change_frozen_branch_or_content(completed_workspace):
    case = completed_workspace
    pending = case.workspace / "unresolved-callback.txt"
    pending.write_bytes(b"retained unpublished work\n")
    head = _git(case.workspace, "rev-parse", "HEAD")
    branch = _git(case.workspace, "branch", "--show-current")
    status = _git(case.workspace, "status", "--short").splitlines()
    q.freeze(case.repo, case.workspace.parent,
             expected=q.census(case.repo, case.workspace.parent))
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        case.supervisor._rescue_dirty_worktree(
            case.workspace, branch=branch, head=head, target_ref="HEAD",
            status_lines=status, reason="recovery_probe",
        )
    assert _git(case.workspace, "branch", "--show-current") == branch
    assert _git(case.workspace, "rev-parse", "HEAD") == head
    assert _git(case.workspace, "status", "--short").splitlines() == status
    assert pending.read_bytes() == b"retained unpublished work\n"


def test_supervisor_guard_holds_custody_through_nonpooled_mutation(completed_workspace):
    case = completed_workspace
    before = q.census(case.repo, case.workspace.parent)
    finished = threading.Event()
    outcomes = []

    def freeze():
        try:
            outcomes.append(q.freeze(case.repo, case.workspace.parent, expected=before))
        except BaseException as error:
            outcomes.append(error)
        finally:
            finished.set()

    with case.supervisor._pooled_worktree_mutation_guard(
        case.workspace, expected_branch=case.branch, operation="custody_probe",
    ) as admission:
        assert admission["allowed"] is True
        worker = threading.Thread(target=freeze)
        worker.start()
        assert not finished.wait(0.05)
    worker.join(timeout=5)
    assert finished.is_set() and len(outcomes) == 1
    assert isinstance(outcomes[0], dict), outcomes
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        with case.supervisor._pooled_worktree_mutation_guard(
            case.workspace, expected_branch=case.branch, operation="late_custody_probe",
        ):
            pytest.fail("late supervisor mutation entered")
