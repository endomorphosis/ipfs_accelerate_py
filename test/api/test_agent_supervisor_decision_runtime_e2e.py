from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DEFAULT_IMPLEMENTATION_PROPOSAL_OUTPUT_BYTES,
    DEFAULT_IMPLEMENTATION_PROPOSAL_PATCH_BYTES,
    MAX_IMPLEMENTATION_PROPOSAL_MATERIALIZED_BYTES,
    MAX_IMPLEMENTATION_PROPOSAL_SERIALIZED_BYTES,
    PortalImplementationDaemon,
    PortalTask,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_daemon_accepts_bounded_local_sources_when_raw_patch_is_small(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    baseline = ("# locally observed source\n" * 31_000) + "VALUE = 1\n"
    paths = (repo / "runtime_a.py", repo / "runtime_b.py")
    for path in paths:
        path.write_text(baseline, encoding="utf-8")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "supervisor@example.invalid")
    _git(repo, "config", "user.name", "Supervisor Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    baseline_ref = _git(repo, "rev-parse", "HEAD")
    for index, path in enumerate(paths, start=2):
        path.write_text(
            baseline.replace("VALUE = 1", f"VALUE = {index}"),
            encoding="utf-8",
        )

    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_pool_enabled=False,
    )
    task = PortalTask(
        task_id="ASI-141",
        title="Repair the retry-budget validation blocker",
        status="todo",
        completion="manual",
        priority="P1",
        track="ops",
        outputs=[path.name for path in paths],
        validation=["python -m pytest"],
        acceptance="Keep raw patches bounded while validating local sources.",
    )

    result = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline_ref,
    )

    assert result.accepted
    assert result.policy.max_patch_bytes > DEFAULT_IMPLEMENTATION_PROPOSAL_PATCH_BYTES
    assert result.policy.max_output_bytes > DEFAULT_IMPLEMENTATION_PROPOSAL_OUTPUT_BYTES
    assert result.policy.max_patch_bytes <= MAX_IMPLEMENTATION_PROPOSAL_MATERIALIZED_BYTES
    assert result.policy.max_output_bytes <= MAX_IMPLEMENTATION_PROPOSAL_SERIALIZED_BYTES


def test_daemon_does_not_expand_limits_for_an_oversized_raw_patch() -> None:
    proposal = SimpleNamespace(
        candidate_diff=(),
        patch_text="x" * (DEFAULT_IMPLEMENTATION_PROPOSAL_PATCH_BYTES + 1),
        to_dict=lambda: {},
    )

    assert PortalImplementationDaemon._proposal_local_envelope_limits(
        proposal
    ) == {
        "max_patch_bytes": DEFAULT_IMPLEMENTATION_PROPOSAL_PATCH_BYTES,
        "max_output_bytes": DEFAULT_IMPLEMENTATION_PROPOSAL_OUTPUT_BYTES,
    }
