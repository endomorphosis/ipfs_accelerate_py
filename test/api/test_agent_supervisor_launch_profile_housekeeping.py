"""Tests for automatic launch-profile merge-base housekeeping."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.launch_profile_housekeeping import (
    HousekeepError,
    apply_merge_base_housekeeping,
    plan_merge_base_housekeeping,
    update_pinned_base_constant,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "housekeep@example.test")
    _git(repo, "config", "user.name", "Housekeep Test")
    (repo / "README").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    # Simulate origin/main tracking local main
    _git(repo, "branch", "-M", "main")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    return repo


def _write_profile(
    repo: Path,
    *,
    pin: str,
    merge_target: str = "agent/example",
) -> Path:
    path = repo / "profile.json"
    path.write_text(
        json.dumps(
            {
                "merge_target_branch": merge_target,
                "merge_target_creation": {
                    "required_before_worker_start": True,
                    "base_ref": "origin/main",
                    "expected_base_commit": pin,
                    "require_clean_recursive_tree": True,
                    "fast_forward_merges_only": True,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _advance_main(repo: Path, message: str) -> str:
    readme = repo / "README"
    readme.write_text(readme.read_text(encoding="utf-8") + message + "\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", message)
    tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", tip)
    return tip


def test_unchanged_when_pin_matches_base_ref(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    pin = _git(repo, "rev-parse", "origin/main")
    profile_path = _write_profile(repo, pin=pin)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    plan = plan_merge_base_housekeeping(repo, profile)
    assert plan.safe is True
    assert plan.action == "unchanged"
    assert plan.old_pin == pin
    assert plan.new_pin == pin


def test_repin_and_ff_merge_target_when_main_advances(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    old_pin = _git(repo, "rev-parse", "origin/main")
    _git(repo, "branch", "agent/example", old_pin)
    profile_path = _write_profile(repo, pin=old_pin)
    companion = repo / "validate_plan.py"
    companion.write_text(
        f'PINNED_BASE_COMMIT = "{old_pin}"\n',
        encoding="utf-8",
    )

    new_pin = _advance_main(repo, "advance")
    assert new_pin != old_pin

    receipt = apply_merge_base_housekeeping(
        repo,
        profile_path,
        write=True,
        companion_pin_paths=[companion],
        receipt_path=repo / "receipt.json",
        update_merge_target=True,
    )
    assert receipt["applied"] is True
    assert receipt["action"] == "repin_and_ff_merge_target"
    assert receipt["old_pin"] == old_pin
    assert receipt["new_pin"] == new_pin

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["merge_target_creation"]["expected_base_commit"] == new_pin
    assert f'PINNED_BASE_COMMIT = "{new_pin}"' in companion.read_text(encoding="utf-8")
    assert _git(repo, "rev-parse", "agent/example") == new_pin
    assert (repo / "receipt.json").is_file()


def test_dry_run_does_not_mutate(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    old_pin = _git(repo, "rev-parse", "origin/main")
    profile_path = _write_profile(repo, pin=old_pin)
    _advance_main(repo, "advance")

    receipt = apply_merge_base_housekeeping(
        repo,
        profile_path,
        write=False,
        dry_run=True,
    )
    assert receipt["applied"] is False
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["merge_target_creation"]["expected_base_commit"] == old_pin


def test_refuses_non_ff_base_ref(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    # Two divergent histories: pin on main tip; origin/main points at side tip.
    _git(repo, "checkout", "-b", "side")
    (repo / "SIDE").write_text("side\n", encoding="utf-8")
    _git(repo, "add", "SIDE")
    _git(repo, "commit", "-m", "side")
    side = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    main_tip = _advance_main(repo, "main-advance")
    profile_path = _write_profile(repo, pin=main_tip)
    _git(repo, "update-ref", "refs/remotes/origin/main", side)

    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", main_tip, side],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ancestor.returncode != 0

    with pytest.raises(HousekeepError):
        apply_merge_base_housekeeping(repo, profile_path, write=True, fail_on_unsafe=True)

    plan = plan_merge_base_housekeeping(
        repo, json.loads(profile_path.read_text(encoding="utf-8"))
    )
    assert plan.safe is False
    assert plan.action == "fail"


def test_leave_ahead_merge_target(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    pin = _git(repo, "rev-parse", "origin/main")
    _git(repo, "branch", "agent/example", pin)
    # Advance agent branch only
    _git(repo, "checkout", "agent/example")
    (repo / "AGENT").write_text("work\n", encoding="utf-8")
    _git(repo, "add", "AGENT")
    _git(repo, "commit", "-m", "agent work")
    agent_tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")

    profile_path = _write_profile(repo, pin=pin)
    receipt = apply_merge_base_housekeeping(
        repo,
        profile_path,
        write=True,
        update_merge_target=True,
    )
    assert receipt["action"] == "unchanged"
    assert receipt["merge_target_action"] == "leave_ahead"
    assert _git(repo, "rev-parse", "agent/example") == agent_tip


def test_update_pinned_base_constant_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text('PINNED_BASE_COMMIT = "abc1234"\nother = 1\n', encoding="utf-8")
    assert update_pinned_base_constant(path, "deadbeefcafebabe") is True
    text = path.read_text(encoding="utf-8")
    assert 'PINNED_BASE_COMMIT = "deadbeefcafebabe"' in text
    assert update_pinned_base_constant(path, "deadbeefcafebabe") is False
