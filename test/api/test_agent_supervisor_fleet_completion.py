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
