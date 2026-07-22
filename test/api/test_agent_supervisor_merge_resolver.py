from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.merge_resolver import (
    active_merge_matches_payload,
    latest_failed_merge_event,
    resolver_payload,
    validate_resolved_paths,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )


def test_latest_failed_merge_ignores_failure_superseded_by_success() -> None:
    failure = {
        "type": "merge_reconciled",
        "task_id": "REF-001",
        "resolved": False,
        "merge_result": {
            "attempted": True,
            "merged": False,
            "branch": "implementation/ref-001",
            "target_branch": "main",
            "reason": "content_conflict",
        },
    }
    success = {
        "type": "implementation_finished",
        "task_id": "REF-001",
        "merge_result": {
            "attempted": True,
            "merged": True,
            "branch": "implementation/ref-001",
            "target_branch": "main",
        },
    }

    assert latest_failed_merge_event([failure, success]) is None
    assert latest_failed_merge_event([failure, success], task_id="REF-001") is None


def test_resolver_payload_targets_recorded_merge_workspace(tmp_path: Path) -> None:
    fallback_repo = tmp_path / "fallback"
    merge_workspace = tmp_path / "merge-workspace"
    for repo in (fallback_repo, merge_workspace):
        repo.mkdir()
        _git(repo, "init")
        _git(repo, "checkout", "-b", "main")

    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "type": "merge_finished",
                "task_id": "REF-002",
                "attempted": True,
                "merged": False,
                "branch": "implementation/ref-002",
                "target_branch": "main",
                "reason": "content_conflict",
                "implementation_commit": "abc123",
                "main_worktree_path": str(merge_workspace),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = resolver_payload(events_path=events_path, repo_root=fallback_repo)

    assert payload["found"] is True
    assert payload["repo_root"] == str(merge_workspace.resolve())
    assert payload["merge_in_progress"] is False
    assert payload["implementation_commit"] == "abc123"
    assert len(payload["conflict_fingerprint"]) == 64
    assert f"Repository: {merge_workspace.resolve()}" in payload["prompt"]


def test_active_merge_must_match_recorded_branch_identity(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.email", "agent@example.com")
    _git(repo, "config", "user.name", "Agent")
    target = repo / "target.py"
    target.write_text("value = 'base'\n", encoding="utf-8")
    _git(repo, "add", "target.py")
    _git(repo, "commit", "-m", "base")

    _git(repo, "checkout", "-b", "implementation/ref-expected")
    target.write_text("value = 'expected'\n", encoding="utf-8")
    _git(repo, "commit", "-am", "expected")
    expected_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _git(repo, "checkout", "main")
    target.write_text("value = 'main'\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    _git(repo, "checkout", "-b", "implementation/ref-other")
    (repo / "other.txt").write_text("other\n", encoding="utf-8")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-m", "other")
    _git(repo, "checkout", "main")

    merge = subprocess.run(
        ["git", "merge", "--no-ff", "implementation/ref-expected"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert merge.returncode != 0

    matching = active_merge_matches_payload(
        {
            "branch": "implementation/ref-expected",
            "implementation_commit": expected_commit,
        },
        repo,
    )
    mismatched = active_merge_matches_payload(
        {"branch": "implementation/ref-other"},
        repo,
    )

    assert matching["matches"] is True
    assert matching["reason"] == "matched"
    assert mismatched["matches"] is False
    assert mismatched["reason"] == "active_merge_identity_mismatch"


def test_resolution_validation_rejects_markers_and_invalid_python(tmp_path: Path) -> None:
    clean = tmp_path / "clean.py"
    marked = tmp_path / "marked.py"
    invalid = tmp_path / "invalid.py"
    clean.write_text("value = 1\n", encoding="utf-8")
    marked.write_text("<<<<<<< HEAD\nvalue = 1\n=======\nvalue = 2\n>>>>>>> branch\n", encoding="utf-8")
    invalid.write_text("def broken(\n", encoding="utf-8")

    result = validate_resolved_paths(tmp_path, ["clean.py", "marked.py", "invalid.py"])

    assert result["valid"] is False
    assert {item["path"] for item in result["marker_findings"]} == {"marked.py"}
    assert {item["path"] for item in result["syntax_errors"]} == {"marked.py", "invalid.py"}
    assert validate_resolved_paths(tmp_path, ["clean.py"])["valid"] is True


def test_resolution_validation_expands_changed_files_in_nested_git_repo(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    _git(nested, "init")
    _git(nested, "config", "user.email", "agent@example.com")
    _git(nested, "config", "user.name", "Agent")
    (nested / "broken.py").write_text("def broken(\n", encoding="utf-8")
    _git(nested, "add", "broken.py")
    _git(nested, "commit", "-m", "invalid resolution")

    result = validate_resolved_paths(tmp_path, ["nested"])

    assert result["valid"] is False
    assert result["expanded_paths"] == ["nested/broken.py"]
    assert result["syntax_errors"][0]["path"] == "nested/broken.py"
