"""Active-storage loss is visible without granting cleanup or recovery authority."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_watchdog as fleet
from ipfs_accelerate_py.agent_supervisor.rescue import storage_diagnostics as storage


@pytest.fixture
def linked_worktree(tmp_path):
    root = tmp_path / "active-board"
    root.mkdir()
    common = tmp_path / "legacy-name" / ".git"
    git_dir = common / "worktrees" / "active-board"
    git_dir.mkdir(parents=True)
    (common / "objects").mkdir()
    (root / ".git").write_text("gitdir: ../legacy-name/.git/worktrees/active-board\n")
    (git_dir / "commondir").write_text("../..\n")
    (git_dir / "gitdir").write_text(str(root / ".git") + "\n")
    (git_dir / "HEAD").write_text("ref: refs/heads/active-board\n")
    return root, git_dir, common


def test_linked_worktree_resolves_external_metadata(linked_worktree):
    root, git_dir, common = linked_worktree
    observation = storage.observe_storage({"git_worktrees": [str(root)]})
    result = observation["git_worktrees"][0]
    assert observation["status"] == "healthy"
    assert result["git_dir"] == str(git_dir)
    assert result["common_dir"] == str(common)


def test_deleted_shared_metadata_is_detected_while_worktree_survives(linked_worktree):
    root, git_dir, common = linked_worktree
    common.rename(common.with_name("preserved-metadata"))
    observation = storage.observe_storage({"git_worktrees": [str(root)]})
    result = observation["git_worktrees"][0]
    assert root.is_dir() and (root / ".git").is_file()
    assert result["reason_codes"] == ["git_directory_missing"]
    assert result["git_dir"] == str(git_dir)
    assert observation["handling"] == "diagnostic_only"
    assert not common.exists()


@pytest.mark.parametrize("component,expected", [
    ("HEAD", "git_head_missing"),
    ("gitdir", "git_worktree_registration_missing"),
])
def test_incomplete_linked_registration_is_detected(linked_worktree, component, expected):
    root, git_dir, _ = linked_worktree
    (git_dir / component).unlink()
    result = storage.observe_storage({"git_worktrees": [str(root)]})
    assert result["reason_codes"] == [expected]


def test_wrong_registration_and_missing_common_directory(linked_worktree):
    root, git_dir, _ = linked_worktree
    (git_dir / "gitdir").write_text(str(root / "wrong-marker"))
    (git_dir / "commondir").write_text("/nonexistent-storage-diagnostic-common")
    result = storage.observe_storage({"git_worktrees": [str(root)]})
    assert set(result["reason_codes"]) == {
        "git_common_directory_missing", "git_worktree_registration_mismatch"}


@pytest.mark.parametrize("pointer", ["fifo", "oversized", "malformed"])
def test_bad_pointer_is_bounded_and_never_echoes_contents(tmp_path, pointer):
    marker = tmp_path / ".git"
    secret = "NOT_FOR_DIAGNOSTIC_OUTPUT"
    if pointer == "fifo":
        os.mkfifo(marker)
    else:
        marker.write_text(secret * 300 if pointer == "oversized" else secret)
    result = storage.observe_storage({"git_worktrees": [str(tmp_path)]})
    assert result["reason_codes"] == ["git_metadata_unavailable"]
    assert secret not in json.dumps(result)
    assert marker.exists()


def test_plain_and_submodule_git_directories_are_supported(tmp_path):
    root = tmp_path / "normal"
    git_dir = root / ".git"
    (git_dir / "objects").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    submodule = tmp_path / "submodule"
    submodule.mkdir()
    (submodule / ".git").write_text("gitdir: ../normal/.git\n")
    result = storage.observe_storage({"git_worktrees": [str(root), str(submodule)]})
    assert result["status"] == "healthy"


def _sample(monkeypatch, *, available=10, blocks=1000, fragment=4096):
    monkeypatch.setattr(storage.os, "statvfs", lambda path: SimpleNamespace(
        f_bavail=available, f_bfree=500, f_blocks=blocks, f_frsize=fragment))


def test_low_space_uses_available_blocks_and_reports_exact_bytes_and_percent(tmp_path, monkeypatch):
    _sample(monkeypatch)
    result = storage.observe_storage({"filesystems": [{"path": str(tmp_path),
        "min_available_bytes": 50000, "min_available_percent": 2}]})
    sample = result["filesystems"][0]
    assert sample["available_bytes"] == 40960
    assert sample["capacity_bytes"] == 4096000
    assert sample["available_percent"] == 1.0
    assert sample["reason_codes"] == [
        "filesystem_available_bytes_low", "filesystem_available_percent_low"]


@pytest.mark.parametrize("minimum_bytes,minimum_percent,reasons", [
    (40960, 1, []),
    (40961, 1, ["filesystem_available_bytes_low"]),
    (0, 1.000001, ["filesystem_available_percent_low"]),
])
def test_space_thresholds_are_independent_and_strict(tmp_path, monkeypatch, minimum_bytes,
                                                    minimum_percent, reasons):
    _sample(monkeypatch)
    result = storage.observe_storage({"filesystems": [{"path": str(tmp_path),
        "min_available_bytes": minimum_bytes, "min_available_percent": minimum_percent}]})
    assert result["filesystems"][0]["reason_codes"] == reasons


def test_unavailable_filesystem_has_no_invented_capacity_or_raw_error(tmp_path, monkeypatch):
    def fail(path):
        raise PermissionError(13, "PRIVATE_ERROR_TEXT")
    monkeypatch.setattr(storage.os, "statvfs", fail)
    result = storage.observe_storage({"filesystems": [{"path": str(tmp_path)}]})
    sample = result["filesystems"][0]
    assert sample["reason_codes"] == ["filesystem_sample_unavailable"]
    assert "available_bytes" not in sample
    assert sample["errno"] == 13
    assert "PRIVATE_ERROR_TEXT" not in json.dumps(result)


@pytest.mark.parametrize("config", [
    {"git_worktrees": ["relative"]},
    {"filesystems": [{"path": "/tmp", "min_available_bytes": True}]},
    {"filesystems": [{"path": "/tmp", "min_available_percent": float("nan")}]},
    {"filesystems": [{"path": "/tmp", "min_available_percent": 101}]},
    {"cleanup_argv": ["rm"]},
])
def test_invalid_storage_policy_is_rejected(config):
    with pytest.raises(ValueError):
        storage.validate_storage_checks(config)


def test_storage_fault_is_diagnostic_only_and_resolution_is_persisted(tmp_path, monkeypatch):
    _sample(monkeypatch)
    board = {"id": "spar", "cwd": str(tmp_path), "probe": {"argv": ["probe"]},
             "repair": {"argv": ["repair"]}, "ensure": {"argv": ["ensure"]},
             "storage_checks": {"filesystems": [{"path": str(tmp_path), "min_available_percent": 2}]}}
    observation = {"board_id": "spar", "health": "healthy", "complete": False,
                   "busy": True, "progress_token": "unchanged", "reason_codes": []}
    calls = []
    def runner(spec, **kwargs):
        calls.append(spec["argv"])
        assert spec["argv"] == ["probe"]
        return {"returncode": 0, "stdout": json.dumps(observation)}
    state_root = tmp_path / "watchdog"
    first = fleet.tick_board(board, state_root, apply=True, runner=runner, now=100)
    assert first["health"] == "healthy" and first["planned_action"] == ""
    assert first["observation"]["reason_codes"] == []
    assert first["attempts"] == first["ensure_attempts"] == 0
    incident_path = state_root / "spar/storage-incident.json"
    assert json.loads(incident_path.read_text())["action"] == "diagnostic_only"
    _sample(monkeypatch, available=30)
    second = fleet.tick_board(board, state_root, apply=True, runner=runner, now=200)
    assert second["last_progress_at"] == first["last_progress_at"]
    assert json.loads(incident_path.read_text())["status"] == "healthy"
    assert len(calls) == 2


def test_storage_observation_preserves_holds_and_completion_gate(tmp_path, monkeypatch):
    _sample(monkeypatch)
    hold = tmp_path / "HOLD"
    hold.touch()
    board = {"id": "spar", "cwd": str(tmp_path), "probe": {"argv": ["probe"]},
             "publication": {"board_id": "spar"}, "hold_files": [str(hold)],
             "storage_checks": {"git_worktrees": [str(tmp_path)]}}
    observation = {"board_id": "spar", "health": "healthy", "complete": False,
                   "completion_candidate": True, "progress_token": "terminal", "reason_codes": []}
    def runner(spec, **kwargs):
        return {"returncode": 0, "stdout": json.dumps(observation)}
    root = tmp_path / "watchdog"
    first = fleet.tick_board(board, root, apply=False, runner=runner, now=100)
    assert first["health"] == "operator_hold" and first["planned_action"] == ""
    assert (root / "spar/storage-incident.json").is_file()
    hold.unlink()
    second = fleet.tick_board(board, root, apply=False, runner=runner, now=101)
    assert second["planned_action"] == "publish"
    assert second["observation"]["complete"] is False
    assert second["observation"]["completion_candidate"] is True
