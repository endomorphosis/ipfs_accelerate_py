"""Real Git integration tests for publication holds, ordering, and retry safety."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/rescue/fleet_completion.py"
SPEC = importlib.util.spec_from_file_location("fleet_completion_under_test", SOURCE)
fleet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fleet)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()


def commit(root: Path, content: str, name: str = "feature.txt") -> str:
    (root / name).write_text(content)
    git(root, "add", name)
    git(root, "commit", "-m", content)
    return git(root, "rev-parse", "HEAD")


def repository(tmp_path: Path, name: str = "repo") -> tuple[Path, Path]:
    remote = tmp_path / f"{name}.git"
    root = tmp_path / name
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    git(tmp_path, "clone", str(remote), str(root))
    git(root, "config", "user.name", "Fleet tests")
    git(root, "config", "user.email", "fleet@example.invalid")
    commit(root, "baseline", "baseline.txt")
    git(root, "push", "origin", "HEAD:main")
    git(root, "switch", "-c", "accepted")
    return root, remote


def manifest(tmp_path: Path, roots: dict[str, Path]) -> tuple[dict, Path]:
    evidence_path = tmp_path / "completion-evidence.json"
    evidence_path.write_text(json.dumps({
        "authoritative": True, "complete": True, "board_id": "BOARD",
        "active_claims": 0, "pending_merges": 0, "blocking_obligations": 0,
        "source_heads": {name: git(root, "rev-parse", "HEAD") for name, root in roots.items()},
    }))
    return {
        "schema": fleet.SCHEMA, "board_id": "BOARD",
        "completion_gate": {
            "cwd": str(tmp_path),
            "argv": [sys.executable, "-c", "import pathlib; print(pathlib.Path(__import__('sys').argv[1]).read_text())", str(evidence_path)],
        },
        "repositories": [{
            "id": name, "root": str(root), "source_ref": "refs/heads/accepted",
            "validation": [{"argv": [sys.executable, "-c", "from pathlib import Path; assert Path('baseline.txt').read_text() == 'baseline'"]}],
            "dependencies": [],
        } for name, root in roots.items()],
    }, evidence_path


@pytest.fixture(autouse=True)
def isolated_git_environment(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")


def local_publication(monkeypatch):
    monkeypatch.setattr(fleet, "_github_origin", lambda root: git(root, "remote", "get-url", "origin"))
    # Local bare repositories stand in for GitHub's normal PR merge service.
    monkeypatch.setattr(fleet, "_merge_reviewed_pull_request",
                        lambda root, remote, candidate: git(root, "push", remote, f"{candidate}:refs/heads/main"))


def test_publishes_accepted_source_and_retries_without_duplicate_commit(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    source = commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    local_publication(monkeypatch)
    first = fleet.publish_completed_board(config, tmp_path / "state")
    assert first["status"] == "published", first
    main = git(remote, "rev-parse", "main")
    assert git(remote, "show", "main:feature.txt") == "accepted feature"
    assert git(root, "rev-parse", "HEAD") == source
    assert git(root, "branch", "--show-current") == "accepted"
    second = fleet.publish_completed_board(config, tmp_path / "state")
    assert second["status"] == "published", second
    assert second["repositories"][0]["status"] == "already_published"
    assert git(remote, "rev-parse", "main") == main


@pytest.mark.parametrize("patch,expected", [
    ({"complete": False}, "authoritative successful completion"),
    ({"authoritative": False}, "authoritative successful completion"),
    ({"active_claims": 1}, "active_claims=0"),
    ({"pending_merges": False}, "pending_merges=0"),
    ({"blocking_obligations": 2}, "blocking_obligations=0"),
    ({"source_heads": {"repo": "0" * 40}}, "exact accepted source heads"),
    ({"board_id": "WRONG"}, "identity mismatch"),
])
def test_incomplete_or_stale_gate_never_publishes(tmp_path, monkeypatch, patch, expected):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    commit(root, "accepted feature")
    config, evidence = manifest(tmp_path, {"repo": root})
    evidence.write_text(json.dumps({**json.loads(evidence.read_text()), **patch}))
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert expected in result["reason"]
    assert git(remote, "rev-parse", "main") == baseline
    assert not (tmp_path / "state/integrations").exists()


def test_dirty_source_is_preserved_and_not_published(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    config, _ = manifest(tmp_path, {"repo": root})
    (root / "baseline.txt").write_text("user work")
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert "dirty" in result["reason"]
    assert (root / "baseline.txt").read_text() == "user work"
    assert git(remote, "show", "main:baseline.txt") == "baseline"


def test_conflict_holds_and_keeps_isolated_checkout(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    git(root, "switch", "main")
    commit(root, "main change", "baseline.txt")
    git(root, "push", "origin", "main")
    baseline = git(remote, "rev-parse", "main")
    git(root, "switch", "accepted")
    source = commit(root, "accepted conflicting change", "baseline.txt")
    config, _ = manifest(tmp_path, {"repo": root})
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held", result
    assert "merge conflict" in result["reason"]
    integration = Path(result["repositories"][0]["integration_worktree"])
    assert git(integration, "diff", "--name-only", "--diff-filter=U") == "baseline.txt"
    assert git(root, "rev-parse", "HEAD") == source
    assert git(remote, "rev-parse", "main") == baseline


def test_failed_validation_never_pushes(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    config["repositories"][0]["validation"] = [{"argv": [sys.executable, "-c", "raise SystemExit(7)"]}]
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert "exit code 7" in result["reason"]
    assert git(remote, "rev-parse", "main") == baseline


def test_nested_repository_published_before_parent_gitlink(tmp_path, monkeypatch):
    child, child_remote = repository(tmp_path, "child")
    child_source = commit(child, "child feature")
    parent, parent_remote = repository(tmp_path, "parent")
    git(parent, "submodule", "add", str(child_remote), "nested")
    git(parent / "nested", "fetch", str(child), "accepted")
    git(parent / "nested", "checkout", child_source)
    git(parent, "add", ".gitmodules", "nested")
    git(parent, "commit", "-m", "accepted nested feature")
    config, _ = manifest(tmp_path, {"parent": parent, "child": child})
    config["repositories"][0]["dependencies"] = [{"repository": "child", "path": "nested"}]
    config["repositories"][0]["validation"].append({"argv": [sys.executable, "-c", "from pathlib import Path; assert Path('nested/feature.txt').read_text() == 'child feature'"]})
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "published", result
    assert [r["id"] for r in result["repositories"]] == ["child", "parent"]
    published_child = git(child_remote, "rev-parse", "main")
    assert published_child != child_source
    assert fleet._gitlinks(parent_remote, "main") == {"nested": published_child}
    assert git(parent / "nested", "rev-parse", "HEAD") == child_source


def test_undeclared_changed_gitlink_holds(tmp_path, monkeypatch):
    child, child_remote = repository(tmp_path, "child")
    parent, parent_remote = repository(tmp_path, "parent")
    baseline = git(parent_remote, "rev-parse", "main")
    git(parent, "submodule", "add", str(child_remote), "nested")
    git(parent, "commit", "-am", "new undeclared dependency")
    config, _ = manifest(tmp_path, {"parent": parent})
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert "changed gitlinks lack" in result["reason"]
    assert git(parent_remote, "rev-parse", "main") == baseline


def test_origin_resolution_follows_local_clone_without_mutation(tmp_path):
    root, remote = repository(tmp_path)
    git(remote, "remote", "add", "origin", "https://github.com/endomorphosis/ipfs_accelerate_py.git")
    assert fleet._github_origin(root) == "https://github.com/endomorphosis/ipfs_accelerate_py.git"
    assert git(root, "remote", "get-url", "origin") == str(remote)
    git(remote, "remote", "set-url", "origin", str(root))
    with pytest.raises(fleet.PublicationHold, match="cycle"):
        fleet._github_origin(root)


def test_manifest_requires_validation_and_rejects_dependency_cycles(tmp_path):
    root, _ = repository(tmp_path)
    config, _ = manifest(tmp_path, {"repo": root})
    config["repositories"][0]["validation"] = []
    with pytest.raises(fleet.PublicationHold, match="validation is required"):
        fleet._ordered_repositories(config)
    config["repositories"][0]["validation"] = [{"argv": ["true"]}]
    config["repositories"][0]["dependencies"] = [{"repository": "repo", "path": "self"}]
    with pytest.raises(fleet.PublicationHold, match="cycle"):
        fleet._ordered_repositories(config)


def test_remote_advancement_during_validation_is_preserved(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    source = commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    local_publication(monkeypatch)
    gate = fleet._completion_gate
    calls = 0
    concurrent = None

    def gate_with_concurrent_push(manifest, heads):
        nonlocal calls, concurrent
        calls += 1
        if calls == 2:
            tree = git(root, "rev-parse", baseline + "^{tree}")
            concurrent = git(root, "commit-tree", tree, "-p", baseline, "-m", "concurrent origin main update")
            git(root, "push", "origin", f"{concurrent}:refs/heads/main")
        return gate(manifest, heads)

    monkeypatch.setattr(fleet, "_completion_gate", gate_with_concurrent_push)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held", result
    assert "advanced during validation" in result["reason"]
    assert git(remote, "rev-parse", "main") == concurrent
    assert git(root, "rev-parse", "HEAD") == source


def test_validation_cannot_replace_commit_then_publish(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    config["repositories"][0]["validation"] = [
        {"argv": ["git", "commit", "--allow-empty", "-m", "unexpected validation commit"]},
    ]
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held", result
    assert "validation changed the integration commit" in result["reason"]
    assert git(remote, "rev-parse", "main") == baseline


def test_timeout_kills_descendants_even_when_they_redirect_output(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    child_script = (
        "import os,signal,time,pathlib; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(__import__('sys').argv[1]).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    parent_script = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], "
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(60)"
    )
    with pytest.raises(fleet.PublicationHold, match="timed out"):
        fleet._run([sys.executable, "-c", parent_script, child_script, str(child_pid_path)], tmp_path, timeout=.5)
    child_pid = child_pid_path.read_text()
    stat = Path("/proc") / child_pid / "stat"
    for _ in range(50):
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(.02)
    else:
        pytest.fail("timed-out command left a live descendant")


def github_pr(monkeypatch, *, patch=None, checks_fail=False, second_patch=None, listed=True):
    candidate = "a" * 40
    calls = []
    views = 0
    ready = {"number": 7, "state": "OPEN", "isDraft": False, "baseRefName": "main",
             "headRefOid": candidate, "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN",
             "reviewDecision": "APPROVED"}

    def run(argv, root, *args):
        nonlocal views
        calls.append(argv)
        if argv[:3] == ["gh", "pr", "list"]:
            return json.dumps([{"number": 7, "headRefOid": candidate}] if listed else [])
        if argv[:3] == ["gh", "pr", "view"]:
            views += 1
            return json.dumps({**ready, **(patch or {}), **(second_patch or {} if views == 2 else {})})
        if argv[:3] == ["gh", "pr", "checks"] and checks_fail:
            raise fleet.PublicationHold("hosted check unavailable due to billing")
        return ""

    monkeypatch.setattr(fleet, "_run", run)
    return candidate, calls


def test_github_publication_merges_exact_reviewed_head_without_direct_main_push(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch)
    fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", candidate)
    pushes = [argv for argv in calls if argv[:2] == ["git", "push"]]
    assert pushes == [["git", "push", "https://github.com/owner/repo.git",
                       f"{candidate}:refs/heads/fleet-publication/{candidate}"]]
    assert ["gh", "pr", "checks", "7", "--repo", "owner/repo", "--required"] in calls
    assert calls[-1] == ["gh", "pr", "merge", "7", "--repo", "owner/repo", "--merge",
                         "--match-head-commit", candidate]
    assert not any("--admin" in argv or "--auto" in argv for argv in calls)


@pytest.mark.parametrize("patch", [
    {"mergeStateStatus": "BLOCKED"}, {"mergeStateStatus": "UNKNOWN"},
    {"mergeStateStatus": "BEHIND"}, {"mergeable": "CONFLICTING"},
    {"reviewDecision": "REVIEW_REQUIRED"}, {"reviewDecision": "CHANGES_REQUESTED"},
    {"reviewDecision": None}, {"headRefOid": "b" * 40}, {"isDraft": True},
    {"baseRefName": "other"}, {"state": "CLOSED"},
])
def test_unready_github_pr_cannot_use_account_bypass_rights(tmp_path, monkeypatch, patch):
    candidate, calls = github_pr(monkeypatch, patch=patch)
    with pytest.raises(fleet.PublicationHold, match="not ready"):
        fleet._merge_reviewed_pull_request(tmp_path, "git@github.com:owner/repo.git", candidate)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)


def test_failed_or_unavailable_hosted_checks_hold_despite_local_validation(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch, checks_fail=True)
    with pytest.raises(fleet.PublicationHold, match="checks are unsuccessful or unavailable"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo", candidate)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)


def test_pr_head_movement_during_checks_refuses_merge(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch, second_patch={"headRefOid": "b" * 40})
    with pytest.raises(fleet.PublicationHold, match="not ready"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo", candidate)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)


def test_new_pr_waits_for_hosted_checks(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch, listed=False)
    with pytest.raises(fleet.PublicationHold, match="pull request created"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo", candidate)
    assert any(argv[:3] == ["gh", "pr", "create"] for argv in calls)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)


def test_pending_pr_retry_reuses_validated_commit_and_checkout(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    local_publication(monkeypatch)
    merge = fleet._merge_reviewed_pull_request
    attempted = []

    def pending(integration, remote, candidate):
        attempted.append((integration, candidate))
        if len(attempted) == 1:
            raise fleet.PublicationHold("required checks pending")
        merge(integration, remote, candidate)

    monkeypatch.setattr(fleet, "_merge_reviewed_pull_request", pending)
    first = fleet.publish_completed_board(config, tmp_path / "state")
    assert first["status"] == "held"
    second = fleet.publish_completed_board(config, tmp_path / "state")
    assert second["status"] == "published", second
    assert attempted[0] == attempted[1]
    assert len(list((tmp_path / "state/integrations").iterdir())) == 1


@pytest.mark.parametrize("mutation", ["dirty", "head", "record"])
def test_pending_pr_retry_refuses_changed_retained_candidate(tmp_path, monkeypatch, mutation):
    root, _ = repository(tmp_path)
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    local_publication(monkeypatch)
    calls = []

    def pending(*args):
        calls.append(args)
        raise fleet.PublicationHold("required checks pending")

    monkeypatch.setattr(fleet, "_merge_reviewed_pull_request", pending)
    first = fleet.publish_completed_board(config, tmp_path / "state")
    integration = Path(first["repositories"][0]["integration_worktree"])
    if mutation == "dirty":
        (integration / "baseline.txt").write_text("unexpected changes")
    elif mutation == "head":
        commit(integration, "unexpected commit")
    else:
        record = next((tmp_path / "state").glob("candidate-*.json"))
        value = json.loads(record.read_text())
        value["integration"] = str(root)
        record.write_text(json.dumps(value))
    second = fleet.publish_completed_board(config, tmp_path / "state")
    assert second["status"] == "held"
    assert len(calls) == 1
