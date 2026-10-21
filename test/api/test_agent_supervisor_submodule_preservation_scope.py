"""Preservation must not manufacture commits from read-only dependency dirt."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from test.api.test_agent_supervisor_nested_submodule_scope import (
    _daemon,
    _git,
    _seed_parent_with_submodule,
    _task,
)


def _add_nested(parent: Path, child: Path, tmp_path: Path) -> Path:
    source = tmp_path / "nested-source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.name", "Test User")
    _git(source, "config", "user.email", "test@example.invalid")
    (source / "base.txt").write_text("nested base\n", encoding="utf-8")
    _git(source, "add", "base.txt")
    _git(source, "commit", "-m", "nested base")
    _git(child, "-c", "protocol.file.allow=always", "submodule", "add", str(source), "nested")
    _git(child, "commit", "-am", "add nested dependency")
    _git(parent, "commit", "-am", "record nested dependency")
    nested = child / "nested"
    _git(nested, "config", "user.name", "Test User")
    _git(nested, "config", "user.email", "test@example.invalid")
    return nested


def _snapshot(repo: Path) -> tuple[str, bytes, str, dict[str, str]]:
    index = Path(_git(repo, "rev-parse", "--git-path", "index"))
    if not index.is_absolute():
        index = repo / index
    files = {}
    for path in repo.rglob("*"):
        if not path.is_file() or ".git" in path.relative_to(repo).parts:
            continue
        files[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return (
        _git(repo, "rev-parse", "HEAD"),
        index.read_bytes(),
        _git(repo, "--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"),
        files,
    )


def _make_drift(repo: Path, kind: str) -> None:
    if kind == "untracked":
        (repo / "unknown.py").write_bytes(b"")
    elif kind == "source":
        (repo / "base.txt").write_text("provider changed source\n", encoding="utf-8")
    elif kind == "index":
        (repo / "base.txt").write_text("index-only evidence\n", encoding="utf-8")
        _git(repo, "add", "base.txt")
        (repo / "base.txt").write_text("base\n", encoding="utf-8")
    elif kind == "head":
        (repo / "committed.txt").write_text("provider commit\n", encoding="utf-8")
        _git(repo, "add", "committed.txt")
        _git(repo, "commit", "-m", "provider child change")
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize("declared_only", [False, True])
@pytest.mark.parametrize("kind", ["source", "index", "head", "untracked"])
def test_unowned_dependency_refusal_preserves_head_index_and_source(tmp_path: Path, kind: str, declared_only: bool) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    task = _task()
    task.outputs[:] = ["root.txt"]
    (parent / "root.txt").write_text("task-owned root\n", encoding="utf-8")
    _make_drift(child, kind)
    before = (_snapshot(parent), _snapshot(child))

    daemon = _daemon(parent, tmp_path)
    if declared_only:
        daemon.worktree_submodule_paths = ()
        task.metadata["submodules"] = "libs/child"
    result = daemon._commit_worktree_changes_unchecked(parent, task, 1)

    assert result["committed"] is False
    assert result["reason"] == "submodule_scope_refused"
    assert result["retain_worktree"] is True
    assert (_snapshot(parent), _snapshot(child)) == before


@pytest.mark.parametrize("kind", ["source", "index", "head", "untracked"])
def test_unknown_nested_dependency_is_never_native_committed(tmp_path: Path, kind: str) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    nested = _add_nested(parent, child, tmp_path)
    task = _task()
    (child / "allowed.txt").write_text("task-owned child\n", encoding="utf-8")
    _make_drift(nested, kind)
    # A local ignore policy must not hide a changed nested checkout.
    _git(child, "config", "submodule.nested.ignore", "all")
    before = (_snapshot(parent), _snapshot(child), _snapshot(nested))

    result = _daemon(parent, tmp_path)._commit_worktree_changes_unchecked(parent, task, 1)

    assert result["committed"] is False
    assert result["reason"] == "submodule_scope_refused"
    assert (_snapshot(parent), _snapshot(child), _snapshot(nested)) == before


def test_entire_forest_is_preflighted_before_owned_sibling_is_staged(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    second = parent / "libs" / "second"
    _git(parent, "-c", "protocol.file.allow=always", "submodule", "add", str(tmp_path / "child-source"), "libs/second")
    _git(parent, "commit", "-am", "second dependency")
    (child / "allowed.txt").write_text("owned\n", encoding="utf-8")
    (second / "unknown.py").write_text("unowned\n", encoding="utf-8")
    daemon = _daemon(parent, tmp_path)
    daemon.worktree_submodule_paths = ("libs/child", "libs/second")
    before = (_snapshot(parent), _snapshot(child), _snapshot(second))

    result = daemon._commit_worktree_changes_unchecked(parent, _task(), 1)

    assert result["reason"] == "submodule_scope_refused"
    assert (_snapshot(parent), _snapshot(child), _snapshot(second)) == before


def test_owned_child_rejects_undeclared_empty_sibling_without_staging(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    (child / "allowed.txt").write_text("owned\n", encoding="utf-8")
    (child / "undeclared.py").write_bytes(b"")
    before = (_snapshot(parent), _snapshot(child))

    result = _daemon(parent, tmp_path)._commit_worktree_changes_unchecked(parent, _task(), 1)

    assert result["reason"] == "submodule_scope_refused"
    assert (_snapshot(parent), _snapshot(child)) == before


def test_only_explicit_owned_child_paths_are_committed(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    (child / "allowed.txt").write_text("authorized\n", encoding="utf-8")
    before = _git(child, "rev-parse", "HEAD")

    result = _daemon(parent, tmp_path)._commit_worktree_changes_unchecked(parent, _task(), 1)

    assert result["committed"] is True, result
    assert _git(child, "rev-parse", "HEAD") != before
    assert _git(child, "diff", "--name-only", before, "HEAD") == "allowed.txt"
    assert _git(parent, "status", "--porcelain", "--ignore-submodules=none") == ""


def test_explicit_owned_nested_paths_are_committed_recursively(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    nested = _add_nested(parent, child, tmp_path)
    task = _task()
    task.outputs[:] = ["libs/child/nested/allowed.txt"]
    (nested / "allowed.txt").write_text("authorized\n", encoding="utf-8")
    before = _git(nested, "rev-parse", "HEAD")
    child_before = _git(child, "rev-parse", "HEAD")

    results = _daemon(parent, tmp_path)._commit_worktree_submodule_changes(parent, task, 1)

    assert results[0]["committed"] is True, results
    assert results[0]["nested_submodule_results"][0]["committed"] is True
    assert _git(nested, "diff", "--name-only", before, "HEAD") == "allowed.txt"
    assert _git(child, "diff", "--name-only", child_before, "HEAD") == "nested"
    assert _git(child, "status", "--porcelain", "--ignore-submodules=none") == ""


def test_failed_validation_retains_unowned_evidence_without_rescue_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    task = _task()
    task.outputs[:] = ["root.txt"]
    (parent / "root.txt").write_text("task-owned\n", encoding="utf-8")
    _make_drift(child, "source")
    daemon = _daemon(parent, tmp_path)
    monkeypatch.setattr(daemon, "_commit_worktree_changes", daemon._commit_worktree_changes_unchecked)
    monkeypatch.setattr(daemon, "_drop_unchanged_seeded_worktree_context", lambda *a, **kw: [])
    def forbidden_cleanup(*args, **kwargs):
        raise AssertionError("refused dependency evidence must not be cleaned")
    monkeypatch.setattr(daemon, "_cleanup_merged_worktree", forbidden_cleanup)
    before = (_snapshot(parent), _snapshot(child))

    result = daemon._preserve_interrupted_worktree(
        parent, "implementation/synthetic", task, 1,
        evidence={"passed": False}, rescue_suffix="failed-validation",
        event_type="synthetic_preservation", evidence_field="validation",
    )

    assert result["preserved"] is False
    assert result["retained"] is True
    assert result["rescue_branch"] == ""
    assert (_snapshot(parent), _snapshot(child)) == before


def test_restore_does_not_claim_success_for_dirty_nested_checkout(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    nested = _add_nested(parent, child, tmp_path)
    _make_drift(nested, "head")
    nested_head = _git(nested, "rev-parse", "HEAD")
    baseline = _git(parent, "rev-parse", "HEAD")

    restored = _daemon(parent, tmp_path)._force_checkout_gitlink_at_baseline(parent, "libs/child", baseline_ref=baseline)

    assert restored is False
    assert _git(nested, "rev-parse", "HEAD") == nested_head
    assert _git(child, "status", "--porcelain", "--ignore-submodules=none")


def test_unowned_parent_index_gitlink_is_retained_without_reset(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    recorded = _git(child, "rev-parse", "HEAD")
    _make_drift(child, "head")
    other = _git(child, "rev-parse", "HEAD")
    _git(parent, "update-index", "--cacheinfo", "160000", other, "libs/child")
    _git(child, "checkout", recorded)
    task = _task()
    task.outputs[:] = ["root.txt"]
    (parent / "root.txt").write_text("owned\n", encoding="utf-8")
    before = (_snapshot(parent), _snapshot(child))

    result = _daemon(parent, tmp_path)._commit_worktree_changes_unchecked(parent, task, 1)

    assert result["reason"] == "submodule_scope_refused"
    assert (_snapshot(parent), _snapshot(child)) == before


def test_clean_read_dependency_is_not_committed_with_owned_root_output(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    task = _task()
    task.outputs[:] = ["root.txt"]
    (parent / "root.txt").write_text("owned\n", encoding="utf-8")
    child_before = _snapshot(child)

    result = _daemon(parent, tmp_path)._commit_worktree_changes_unchecked(parent, task, 1)

    assert result["committed"] is True, result
    assert _snapshot(child) == child_before
    assert not result.get("submodule_results")


@pytest.mark.parametrize("parent_already_committed", [False, True])
def test_owned_provider_commits_keep_their_original_identity(tmp_path: Path, parent_already_committed: bool) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    baseline = _git(parent, "rev-parse", "HEAD")
    (child / "allowed.txt").write_text("provider-owned\n", encoding="utf-8")
    _git(child, "add", "allowed.txt")
    _git(child, "commit", "-m", "provider child")
    child_commit = _git(child, "rev-parse", "HEAD")
    if parent_already_committed:
        _git(parent, "add", "libs/child")
        _git(parent, "commit", "-m", "provider handoff")
    parent_before = _git(parent, "rev-parse", "HEAD")

    result = _daemon(parent, tmp_path)._commit_worktree_changes_unchecked(parent, _task(), 1, baseline_ref=baseline)

    assert result["committed"] is True, result
    assert _git(child, "rev-parse", "HEAD") == child_commit
    assert _git(parent, "rev-parse", "HEAD:libs/child") == child_commit
    if parent_already_committed:
        assert result["commit"] == parent_before
        assert result["reason"] == "existing_commit"
    else:
        assert result["submodule_results"][0]["reason"] == "existing_commit"


def test_recursive_owned_commit_preserves_protected_generated_bytes(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    nested = _add_nested(parent, child, tmp_path)
    task = _task()
    task.outputs[:] = ["libs/child/nested/base.txt"]
    for target in (child, nested):
        _git(target, "config", "core.excludesfile", str(tmp_path / "excludes"))
        for relative in (".pytest_cache/sentinel", "test-results/generated.txt"):
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic artifact")
    (tmp_path / "excludes").write_text(".pytest_cache/\ntest-results/\n", encoding="utf-8")
    (nested / "base.txt").write_text("authorized change\n", encoding="utf-8")
    daemon = _daemon(parent, tmp_path)
    daemon.implementation_protected_paths = (
        "libs/child/.pytest_cache/sentinel", "libs/child/nested/.pytest_cache/sentinel",
    )

    results = daemon._commit_worktree_submodule_changes(parent, task, 1)

    assert results[0]["committed"] is True, results
    for target in (child, nested):
        assert (target / ".pytest_cache/sentinel").read_bytes() == b"synthetic artifact"
        assert not (target / "test-results/generated.txt").exists()


def test_direct_recursive_preservation_also_refuses_unowned_nested_changes(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    nested = _add_nested(parent, child, tmp_path)
    _make_drift(nested, "source")
    before = (_snapshot(parent), _snapshot(child), _snapshot(nested))

    results = _daemon(parent, tmp_path)._commit_nested_submodule_changes(child, _task(), 1, parent_relative="libs/child")

    assert results and all(result["reason"] == "submodule_scope_refused" for result in results)
    assert (_snapshot(parent), _snapshot(child), _snapshot(nested)) == before


def test_local_restore_certifies_exact_clean_checkout(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    baseline = _git(parent, "rev-parse", "HEAD")
    expected = _git(child, "rev-parse", "HEAD")
    _make_drift(child, "head")

    assert _daemon(parent, tmp_path)._force_checkout_gitlink_at_baseline(parent, "libs/child", baseline_ref=baseline) is True
    assert _git(child, "rev-parse", "HEAD") == expected
    assert _git(parent, "status", "--porcelain", "--ignore-submodules=none") == ""


def test_missing_local_restore_object_never_fetches_or_initializes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    parent, child = _seed_parent_with_submodule(tmp_path)
    _git(parent, "update-index", "--cacheinfo", "160000", "f" * 40, "libs/child")
    _git(parent, "commit", "-m", "synthetic unavailable child ref")
    baseline = _git(parent, "rev-parse", "HEAD")
    child_head = _git(child, "rev-parse", "HEAD")
    original_run = subprocess.run
    def local_only(argv, **kwargs):
        if argv and argv[0] == "git":
            assert not any(token in {"fetch", "clone", "update"} for token in argv[1:])
        return original_run(argv, **kwargs)
    monkeypatch.setattr(subprocess, "run", local_only)

    assert _daemon(parent, tmp_path)._force_checkout_gitlink_at_baseline(parent, "libs/child", baseline_ref=baseline) is False
    assert _git(child, "rev-parse", "HEAD") == child_head


def test_post_validation_nested_drift_keeps_original_candidate_rejected(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    nested = _add_nested(parent, child, tmp_path)
    baseline = _git(parent, "rev-parse", "HEAD")
    task = _task()
    task.outputs[:] = ["root.txt"]
    (parent / "root.txt").write_text("validated candidate\n", encoding="utf-8")
    daemon = _daemon(parent, tmp_path)
    proposal = daemon._validate_implementation_patch(parent, task, baseline_ref=baseline)
    assert proposal.accepted is True
    _make_drift(nested, "head")
    child_head = _git(child, "rev-parse", "HEAD")
    nested_head = _git(nested, "rev-parse", "HEAD")

    result = daemon._verify_post_validation_candidate_binding(
        parent, task, baseline_ref=baseline, proposal_validation=proposal,
        validation_result={"attempted": True, "passed": True, "returncode": 0, "results": []},
    )

    assert result["passed"] is False
    assert result["reason"] == "candidate_changed_during_validation"
    assert result["candidate_binding"]["verified"] is False
    assert _git(child, "rev-parse", "HEAD") == child_head
    assert _git(nested, "rev-parse", "HEAD") == nested_head


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_uninitialized_gitlink_contents_are_refused_and_never_restored_over(tmp_path: Path, kind: str) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    _git(parent, "submodule", "deinit", "-f", "libs/child")
    outside = tmp_path / "outside-sentinel"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"outside evidence")
    if kind == "directory":
        (child / "unknown.py").write_bytes(b"partial clone evidence")
        # Git cannot report these bytes: an empty status is insufficient.
        assert _git(parent, "status", "--porcelain", "--ignore-submodules=none") == ""
    elif kind == "file":
        child.rmdir()
        child.write_bytes(b"replacement evidence")
    else:
        child.rmdir()
        child.symlink_to(outside, target_is_directory=True)
    task = _task()
    task.outputs[:] = ["root.txt"]
    (parent / "root.txt").write_text("owned\n", encoding="utf-8")
    daemon = _daemon(parent, tmp_path)
    before = _snapshot(parent)
    baseline = _git(parent, "rev-parse", "HEAD")

    result = daemon._commit_worktree_changes_unchecked(parent, task, 1)

    assert result["reason"] == "submodule_scope_refused"
    assert result["retain_worktree"] is True
    assert _snapshot(parent) == before
    assert daemon._force_checkout_gitlink_at_baseline(parent, "libs/child", baseline_ref=baseline) is False
    assert _snapshot(parent) == before
    assert sentinel.read_bytes() == b"outside evidence"


def test_known_empty_deferred_gitlink_can_remain_uninitialized(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    _git(parent, "submodule", "deinit", "-f", "libs/child")
    task = _task()
    task.outputs[:] = ["root.txt"]
    (parent / "root.txt").write_text("owned\n", encoding="utf-8")
    daemon = _daemon(parent, tmp_path)

    result = daemon._commit_worktree_changes_unchecked(parent, task, 1)

    assert result["committed"] is True, result
    assert child.is_dir() and list(child.iterdir()) == []
    assert daemon._force_checkout_gitlink_at_baseline(parent, "libs/child", baseline_ref="HEAD") is True
    assert list(child.iterdir()) == []


@pytest.mark.parametrize("operation", ["commit", "restore"])
def test_ancestor_directory_alias_is_refused_before_outside_git_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    import subprocess
    parent, original_child = _seed_parent_with_submodule(tmp_path)
    daemon = _daemon(parent, tmp_path)
    baseline = _git(parent, "rev-parse", "HEAD")
    _git(parent, "submodule", "deinit", "-f", "libs/child")
    original_child.rmdir()
    (parent / "libs").rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _git(outside, "clone", str(tmp_path / "child-source"), "child")
    child = outside / "child"
    (child / "base.txt").write_bytes(b"outside owner modified source\n")
    (parent / "libs").symlink_to(outside, target_is_directory=True)
    assert not (parent / "libs/child").is_symlink()
    task = _task()
    task.outputs[:] = ["libs/child/base.txt"]
    before = (_snapshot(parent), _snapshot(child))
    original_run = subprocess.run
    def reject_outside_git(argv, **kwargs):
        cwd = kwargs.get("cwd")
        if argv and argv[0] == "git" and cwd is not None:
            assert not Path(cwd).resolve().is_relative_to(outside), "aliased child must be refused before outside Git access"
        return original_run(argv, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(subprocess, "run", reject_outside_git)
        if operation == "commit":
            result = daemon._commit_worktree_changes_unchecked(parent, task, 1)
            assert result["committed"] is False
            assert result["reason"] == "submodule_scope_refused"
            assert result["retain_worktree"] is True
        else:
            assert daemon._force_checkout_gitlink_at_baseline(parent, "libs/child", baseline_ref=baseline) is False

    assert (_snapshot(parent), _snapshot(child)) == before
