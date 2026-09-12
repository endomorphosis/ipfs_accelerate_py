"""Native journal recovery binds physical Git identity and exact worktree scope."""

import copy

import pytest
import test_agent_supervisor_implementation_protected_paths as native

from ipfs_accelerate_py.agent_supervisor.merge import checkout_lock as locks
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as module,
)


def journal(tmp_path):
    supervisor, repo, todo = native._generated_protected_supervisor(tmp_path)

    def interrupted():
        todo.write_text("# Tasks\n\n## EX-002 Restart recovery\n")
        return ["EX-002"]

    with pytest.raises(RuntimeError, match="protected_generated_outputs_dirty"):
        supervisor._run_generated_board_producer(
            producer="actual-native-journal-test",
            commit_outputs=True,
            callback=interrupted,
        )
    lease = locks.read_checkout_mutation_lease(supervisor._repo_merge_lock_path())
    assert lease is not None
    assert lease.metadata["repo_root"] == ""
    assert lease.metadata["worktree_root"] == str(repo.resolve())
    assert lease.metadata["repository_id"]
    dead = locks.update_checkout_mutation_lease(
        lease, {**lease.metadata, "pid": 2_147_483_647}
    )
    assert dead is not None
    return module.PortalImplementationSupervisor(supervisor.config), repo, todo, dead


@pytest.mark.parametrize("legacy", [False, True])
def test_native_journal_recovery_uses_scoped_identity_from_any_cwd(
    tmp_path, monkeypatch, legacy
):
    supervisor, repo, todo, lease = journal(tmp_path)
    if legacy:
        payload = dict(lease.metadata)
        payload.pop("worktree_root")
        payload.pop("repository_id")
        payload["repo_root"] = str(repo.resolve())
        lease = locks.update_checkout_mutation_lease(lease, payload)
        assert lease is not None
    monkeypatch.chdir(tmp_path)
    result = supervisor._recover_retained_generated_checkout_lease()
    assert result["recovered"] is True
    assert result["adoption"]["adopted"] is True
    assert result["retained_lease"] is False
    assert not lease.lock_path.exists()
    assert native._git(repo, "status", "--porcelain", "--", todo.name) == ""
    assert (
        native._git(repo, "log", "-1", "--pretty=%ae")
        == locks.BACKLOG_REFINERY_AUTHOR_EMAIL
    )


@pytest.mark.parametrize(
    "mismatch",
    [
        "worktree",
        "sibling",
        "repository",
        "common_git",
        "unknown_git",
        "missing_worktree",
        "legacy_contradiction",
        "guard",
        "intent",
        "paths",
        "unknown_matcher",
    ],
)
def test_native_journal_rejects_changed_authority_and_retains_all_evidence(
    tmp_path,
    monkeypatch,
    mismatch,
):
    supervisor, repo, todo, lease = journal(tmp_path)
    payload = copy.deepcopy(dict(lease.metadata))
    if mismatch in {"worktree", "repository", "common_git"}:
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        native._git(foreign, "init")
        foreign_identity = locks.checkout_lock_metadata(kind="merge", repo_root=foreign)
        if mismatch == "worktree":
            payload["worktree_root"] = str(foreign)
        elif mismatch == "repository":
            payload["repository_id"] = foreign_identity["repository_id"]
        else:
            payload["worktree_root"] = str(foreign)
            payload["repository_id"] = ""
    elif mismatch == "sibling":
        sibling = tmp_path / "sibling"
        native._git(repo, "worktree", "add", "-b", "sibling", str(sibling))
        payload["worktree_root"] = str(sibling)
        assert locks.checkout_lock_repository_matches(payload, repo) is True
    elif mismatch == "unknown_git":
        payload["worktree_root"] = str(tmp_path / "unavailable-git-worktree")
        payload["repository_id"] = ""
        assert locks.checkout_lock_repository_matches(payload, repo) is None
    elif mismatch == "missing_worktree":
        payload["worktree_root"] = ""
    elif mismatch == "legacy_contradiction":
        payload["repo_root"] = str(tmp_path / "wrong-legacy-worktree")
    elif mismatch == "guard":
        payload["protected_release_guard"]["protected_paths"] = ["foreign.md"]
    elif mismatch == "intent":
        payload["protected_recovery_intent"]["producer"] = "replacement"
    elif mismatch == "paths":
        payload["protected_paths"] = ["foreign.md"]
    elif mismatch == "unknown_matcher":
        monkeypatch.setattr(module, "checkout_lock_repository_matches", lambda *_: None)
    replaced = locks.update_checkout_mutation_lease(lease, payload)
    assert replaced is not None
    before = replaced.lock_path.read_bytes()
    todo_before = todo.read_bytes()
    head = native._git(repo, "rev-parse", "HEAD")
    # The old empty-repo_root consumer accidentally accepted unrelated
    # authority when a wrapper's cwd happened to equal its configured root.
    monkeypatch.chdir(repo)
    result = supervisor._recover_retained_generated_checkout_lease()
    assert result["blocked"] is True
    assert result["adopted"] is False
    assert result["retained_lease"] is True
    assert replaced.lock_path.read_bytes() == before
    assert todo.read_bytes() == todo_before
    assert native._git(repo, "rev-parse", "HEAD") == head


def test_foreign_daemon_owned_journal_stays_pending_without_adoption(tmp_path):
    supervisor, _repo, todo, lease = journal(tmp_path)
    foreign = locks.update_checkout_mutation_lease(
        lease, {**lease.metadata, "protected_recovery_owner": "implementation_daemon"}
    )
    assert foreign is not None
    before = foreign.lock_path.read_bytes()
    todo_before = todo.read_bytes()
    result = supervisor._recover_retained_generated_checkout_lease()
    assert result["blocked"] is False and result["pending"] is True
    assert result["adopted"] is False and result["recovered"] is False
    assert result["reason"] == "daemon_protected_checkout_recovery_pending"
    assert foreign.lock_path.read_bytes() == before
    assert todo.read_bytes() == todo_before
