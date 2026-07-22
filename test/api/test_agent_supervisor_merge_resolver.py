from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.merge_resolver import (
    latest_failed_merge_event,
    resolver_payload,
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
    assert len(payload["conflict_fingerprint"]) == 64
    assert f"Repository: {merge_workspace.resolve()}" in payload["prompt"]
