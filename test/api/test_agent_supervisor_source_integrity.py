"""Actual Git worktrees and live probe fences for mutable control-plane code."""

import subprocess

import pytest
from test.api.test_agent_supervisor_live_board_probe import board as board_fixture
from test.api.test_agent_supervisor_live_board_probe import probe

board = board_fixture


def git(root, *args):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
                          check=True, capture_output=True, text=True).stdout


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source with spaces"
    root.mkdir()
    git(root, "init", "-q")
    (root / "runtime").mkdir()
    (root / "runtime/[driver].py").write_text("safe = True\n")
    (root / "task.py").write_text("task = True\n")
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "qualified source")
    return root, {"source_integrity_paths": [{"repository": str(root), "paths": ["runtime"]}]}


def test_clean_source_and_unrelated_task_edits_are_allowed(source):
    root, config = source
    (root / "task.py").write_text("ordinary task edit\n")
    result = probe._source_integrity(config)
    assert result["configured"] and result["valid"]
    assert result["scope"] == "configured_control_plane_cleanliness"


@pytest.mark.parametrize("kind", ["unstaged", "staged", "untracked", "deleted"])
def test_real_runtime_mutations_are_refused(source, kind):
    root, config = source
    target = root / "runtime/[driver].py"
    if kind == "deleted":
        target.unlink()
    elif kind == "untracked":
        (root / "runtime/new.py").write_text("unqualified = True\n")
    else:
        target.write_text("unsafe = True\n")
        if kind == "staged":
            git(root, "add", ".")
    result = probe._source_integrity(config)
    assert result["valid"] is False
    assert result["reason"] == "configured_control_plane_dirty"
    assert "unsafe" not in str(result)


def test_literal_paths_and_absent_tracked_paths(source):
    root, config = source
    config["source_integrity_paths"][0]["paths"] = ["runtime/[driver].py"]
    assert probe._source_integrity(config)["valid"]
    config["source_integrity_paths"][0]["paths"] = ["runtime/absent.py"]
    assert probe._source_integrity(config)["reason"] == "source_paths_not_tracked"


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_index_flags_cannot_hide_runtime_edits(source, flag):
    root, config = source
    git(root, "update-index", flag, "runtime/[driver].py")
    (root / "runtime/[driver].py").write_text("unsafe hidden bytes\n")
    assert not git(root, "status", "--porcelain", "--", "runtime")
    assert probe._source_integrity(config)["reason"] == "source_index_not_verifiable"


def test_truncated_index_cannot_hide_a_flag_outside_the_bound(source, monkeypatch):
    _, config = source
    monkeypatch.setattr(probe, "MAX_SOURCE_INDEX_BYTES", 5)
    assert probe._source_integrity(config)["valid"] is False


def test_tracked_symlink_cannot_hide_external_runtime_edits(source, tmp_path):
    root, config = source
    external = tmp_path / "external.py"
    external.write_text("safe = True\n")
    (root / "runtime/linked.py").symlink_to(external)
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "link")
    external.write_text("unsafe = True\n")
    assert not git(root, "status", "--porcelain", "--", "runtime")
    assert probe._source_integrity(config)["reason"] == "source_index_not_verifiable"


def test_nested_nonrepository_cannot_borrow_parent_git_authority(source):
    root, config = source
    config["source_integrity_paths"][0]["repository"] = str(root / "runtime")
    assert probe._source_integrity(config)["reason"] == "source_repository_unavailable"


@pytest.mark.parametrize("paths", [["../runtime"], ["/tmp/runtime"], [".git/config"], [], ["."], ["././"]])
def test_bad_scopes_fail_closed_without_git(source, monkeypatch, paths):
    _, config = source
    config["source_integrity_paths"][0]["paths"] = paths
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: pytest.fail("invalid scope invoked Git"))
    assert probe._source_integrity(config)["valid"] is False


def test_timed_out_git_is_inconclusive(source, monkeypatch):
    _, config = source
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 5)
    monkeypatch.setattr(probe.subprocess, "run", timeout)
    assert probe._source_integrity(config)["reason"] == "source_integrity_unavailable"


def test_dirty_source_skips_native_code_and_cannot_ensure(board, source, monkeypatch):
    config, identities, _ = board
    root, integrity = source
    config.update(integrity)
    (root / "runtime/[driver].py").write_text("unsafe = True\n")
    monkeypatch.setattr(probe, "_status_command", lambda _: pytest.fail("must not import dirty native code"))
    for live in (True, False):
        if not live:
            identities.clear()
        result = probe.observe_board(config, now=1000)
        assert result["health"] == "degraded"
        assert "source_integrity_not_verified" in result["reason_codes"]
        assert result["completion_candidate"] is False
        assert "recovery_action" not in result


def test_runtime_changed_during_native_read_cannot_appear_recovered(board, source, monkeypatch):
    config, _, _ = board
    root, integrity = source
    config.update(integrity)
    def changed(_):
        (root / "runtime/[driver].py").write_text("changed during read\n")
        return {"task_authority": {"status_counts": {"completed": 67}, "task_count": 67,
                                    "authenticated_query": True}}, ""
    monkeypatch.setattr(probe, "_status_command", changed)
    result = probe.observe_board(config, now=1000)
    assert result["health"] == "degraded"
    assert result["details"]["source_integrity"]["valid"] is False
    assert result["completion_candidate"] is False
