"""Run the production implementation daemon against an isolated benchmark repo.

Task boards must first be generated with objectives.objective_daemon. This is
an explicit legacy-Markdown profile of the real daemon, not an imitation loop.
The native daemon owns worktrees, implementation dispatch, tests and merging.
"""
import argparse
import logging
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    repo, state = args.repository.resolve(), args.state_root.resolve()
    if not (repo / ".git").is_dir() or not (repo / "tasks.todo.md").is_file():
        parser.error("expected a standalone benchmark Git repo with a generated task board")
    if repo == Path(__file__).resolve().parents[3]:
        parser.error("refusing to use the supervisor source checkout as the benchmark target")
    # Provider children switch cwd to the worktree. A relative PYTHONPATH would
    # then lose the supervisor package, even though the parent imported it.
    os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
        run_configured_portal_implementation_daemon,
    )
    argv = [
        "--todo-path", str(repo / "tasks.todo.md"),
        "--task-source-kind", "legacy-markdown", "--explicit-legacy-task-source",
        "--state-dir", str(state), "--state-prefix", "benchmark",
        "--task-prefix", "## BENCH-", "--board-namespace", "full-daemon-benchmark",
        "--implement", "--implementation-timeout", "180", "--max-task-attempts", "2",
        "--worktree-root", str(state / "worktrees"), "--merge-target-branch", "main",
        "--retain-worktree-artifacts", "--merged-worktree-cleanup-max", "0",
        "--implementation-protected-path", "test_operations.py",
        "--implementation-protected-path", "README.md",
        "--objective-path", str(repo / "objectives.md"),
        "--objective-bundle-dir", str(repo / ".runtime/bundles"),
        "--objective-scan-max-findings", "0", "--codebase-scan-max-findings", "0",
        "--interval", "5", "--log-level", "INFO",
    ]
    if args.once:
        argv.append("--once")
    run_configured_portal_implementation_daemon(
        argv, repo_root=repo, logger=logging.getLogger("benchmark.full_daemon"),
    )


if __name__ == "__main__":
    main()
