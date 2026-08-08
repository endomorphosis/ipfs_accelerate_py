from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    MERGE_TARGET_BINDING_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_validation import (
    build_post_merge_validation_evidence,
    verify_post_merge_validation_evidence,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_scheduler import (
    ValidationScheduler,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _runtime(
    tmp_path: Path,
    *,
    scheduler: Any,
) -> tuple[TodoImplementationDaemon, Path, PortalTask, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "checkout", "-qb", "main")
    _git(repo, "config", "user.name", "Post-merge Runtime Test")
    _git(repo, "config", "user.email", "post-merge@example.invalid")
    (repo / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "baseline")
    commit = _git(repo, "rev-parse", "HEAD")
    tree_id = f"git-tree:{_git(repo, 'rev-parse', 'HEAD^{tree}')}"

    state_dir = tmp_path / "state"
    todo_path = tmp_path / "tasks.todo.md"
    todo_path.write_text(
        "## PMV-001 Validate landed commit\n\n"
        "- Status: todo\n"
        "- Completion: manual\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        merge_target_branch="main",
        validation_scheduler=scheduler,
        validation_cache_dir=state_dir / "validation-cache",
        worktree_root=tmp_path / "worktrees",
        worktree_pool_enabled=False,
        merge_queue_dir=state_dir / "merge-queue",
        worktree_submodule_paths=(),
    )
    task = PortalTask(
        task_id="PMV-001",
        title="Validate landed commit",
        status="todo",
        completion="manual",
        priority="P0",
        track="quality",
        validation=["git diff --check"],
        metadata={"Provider role": "deterministic-only"},
    )
    return daemon, repo, task, commit, tree_id


class _RecordingScheduler:
    def __init__(
        self,
        during_run: Callable[[Path], None] | None = None,
    ) -> None:
        self.during_run = during_run
        self.calls: list[dict[str, Any]] = []

    def run(self, commands: Any, **kwargs: Any) -> dict[str, Any]:
        command_specs = tuple(commands)
        workspace = Path(kwargs["workspace_path"])
        self.calls.append(
            {
                "commands": command_specs,
                "workspace": workspace,
                **kwargs,
            }
        )
        if self.during_run is not None:
            self.during_run(workspace)
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [
                {
                    "command": command_specs[0].command,
                    "returncode": 0,
                    "stage": "cheap",
                    "output": "passed\n",
                }
            ],
            "selection": {
                "scope": kwargs["scope"],
                "changed_files": [],
                "escalated": True,
            },
            "elapsed_seconds": 0.125,
        }


def test_validation_runner_binds_uncached_post_merge_scope_and_target(
    tmp_path: Path,
) -> None:
    scheduler = _RecordingScheduler()
    daemon, repo, task, commit, _tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )

    result = daemon._run_validation_commands(
        repo,
        task,
        tmp_path / "post-merge.log",
        scope="post_merge",
        target_commit=commit,
    )

    assert result["passed"] is True
    assert result["validation_scope"] == "post_merge"
    assert result["target_commit"] == commit
    assert result["validated_commit"] == commit
    call = scheduler.calls[0]
    assert call["scope"] == "post_merge"
    assert call["target_commit"] == commit
    assert call["require_full_validation"] is True
    assert all(spec.cacheable is False for spec in call["commands"])


def test_exact_post_merge_runtime_builds_receipt_in_detached_clean_worktree(
    tmp_path: Path,
) -> None:
    scheduler = _RecordingScheduler()
    daemon, repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is True
    assert evidence["attempted"] is True
    assert evidence["target_commit"] == commit
    assert evidence["validated_commit"] == commit
    assert evidence["repository_tree_id"] == tree_id
    assert evidence["validation_scope"] == "post_merge"
    fence = evidence["validation_result"]["fence"]
    for stage in ("before", "after"):
        assert fence[stage]["tracked_checkout_exact"] is True
        assert fence[stage]["tracked_checkout_proof"]["passed"] is True
        assert fence[stage]["tracked_checkout_proof"]["tracked_entry_count"] == 1
    assert verify_post_merge_validation_evidence(
        evidence,
        expected_task_id=task.task_id,
        expected_target_commit=commit,
        expected_repository_tree_id=tree_id,
    ) == (True, ())
    workspace = scheduler.calls[0]["workspace"]
    assert workspace != repo
    assert not workspace.exists()
    assert str(workspace) not in _git(repo, "worktree", "list", "--porcelain")
    assert evidence["validation_result"]["elapsed_seconds"] == "0.125"


def test_exact_post_merge_runtime_unregisters_partial_worktree_add(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _RecordingScheduler()
    daemon, repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    real_run = subprocess.run
    partial_workspace: Path | None = None
    remove_calls: list[Path] = []

    def fail_after_registering_worktree(arguments, *args, **kwargs):
        nonlocal partial_workspace
        result = real_run(arguments, *args, **kwargs)
        command = [str(item) for item in arguments]
        if command[:3] == ["git", "worktree", "add"]:
            partial_workspace = Path(command[-2])
            assert result.returncode == 0, result.stderr
            return subprocess.CompletedProcess(
                result.args,
                1,
                result.stdout,
                "seeded failure after worktree registration",
            )
        if command[:3] == ["git", "worktree", "remove"]:
            remove_calls.append(Path(command[-1]))
        return result

    monkeypatch.setattr(
        implementation_daemon_module.subprocess,
        "run",
        fail_after_registering_worktree,
    )

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is False
    assert evidence["validation_result"]["reason"] == (
        "post_merge_validation_worktree_add_failed"
    )
    assert partial_workspace is not None
    assert remove_calls == [partial_workspace]
    assert partial_workspace.exists() is False
    assert str(partial_workspace) not in _git(
        repo,
        "worktree",
        "list",
        "--porcelain",
    )


def test_exact_post_merge_runtime_executes_every_outer_namespace_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = ValidationScheduler(max_workers=1, resource_budget=1)
    daemon, repo, _task, _commit, _tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    dependency = tmp_path / "ipfs-accelerate"
    dependency.mkdir()
    _git(dependency, "init", "-qb", "main")
    _git(dependency, "config", "user.name", "Outer Namespace Test")
    _git(
        dependency,
        "config",
        "user.email",
        "outer-namespace@example.invalid",
    )
    module_path = (
        dependency
        / "ipfs_accelerate_py/agent_supervisor"
        / "analysis/deterministic_repair_forest.py"
    )
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        """from __future__ import annotations

import json
import sys
from pathlib import Path

MARKER = \"outer-namespace-loaded\"


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != \"validate\":
        return 2
    payload = json.loads(Path(sys.argv[2]).read_text(encoding=\"utf-8\"))
    return 0 if payload == {\"forest\": \"current\"} else 1


if __name__ == \"__main__\":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    test_path = (
        dependency / "test/api/test_agent_supervisor_dcr_forest.py"
    )
    test_path.parent.mkdir(parents=True)
    test_path.write_text(
        """from external.ipfs_accelerate.ipfs_accelerate_py.agent_supervisor.analysis import deterministic_repair_forest


def test_outer_namespace_is_importable() -> None:
    assert deterministic_repair_forest.MARKER == \"outer-namespace-loaded\"
        """,
        encoding="utf-8",
    )
    (dependency / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n",
        encoding="utf-8",
    )
    _git(dependency, "add", ".")
    _git(dependency, "commit", "-qm", "seed forest validator")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(dependency),
        "external/ipfs_accelerate",
    )
    forest_path = (
        repo / "data/agent_supervisor/deterministic_contract_repair/forest.json"
    )
    forest_path.parent.mkdir(parents=True)
    forest_path.write_text('{"forest": "current"}\n', encoding="utf-8")
    _git(repo, "add", "data")
    _git(repo, "commit", "-qm", "seed outer namespace validation")
    commit = _git(repo, "rev-parse", "HEAD")
    tree_id = f"git-tree:{_git(repo, 'rev-parse', 'HEAD^{tree}')}"
    commands = (
        "python3 -m pytest -q "
        "external/ipfs_accelerate/test/api/"
        "test_agent_supervisor_dcr_forest.py",
        "python3 -m external.ipfs_accelerate.ipfs_accelerate_py."
        "agent_supervisor.analysis.deterministic_repair_forest validate "
        "data/agent_supervisor/deterministic_contract_repair/forest.json",
    )
    task = PortalTask(
        task_id="PMV-001",
        title="Replay the complete DCR validation plan",
        status="todo",
        completion="manual",
        priority="P0",
        track="quality",
        validation=list(commands),
        metadata={"Provider role": "deterministic-only"},
    )
    monkeypatch.setattr(
        daemon,
        "_with_worktree_validation_pythonpath",
        lambda *_args, **_kwargs: pytest.fail(
            "post-merge commands must use checkout-root package namespaces"
        ),
    )

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is True, evidence.get("validation_result", {}).get(
        "workspace_error"
    )
    results = evidence["validation_result"]["results"]
    assert [result["command"] for result in results] == list(commands)
    assert [result["returncode"] for result in results] == [0, 0]
    assert all("PYTHONPATH=" not in result["command"] for result in results)


def test_exact_post_merge_runtime_materializes_local_nested_submodules_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grandchild = tmp_path / "grandchild"
    grandchild.mkdir()
    _git(grandchild, "init", "-qb", "main")
    _git(grandchild, "config", "user.name", "Nested Validation Test")
    _git(
        grandchild,
        "config",
        "user.email",
        "nested-validation@example.invalid",
    )
    (grandchild / "proof.txt").write_text(
        "nested proof\n",
        encoding="utf-8",
    )
    _git(grandchild, "add", "proof.txt")
    _git(grandchild, "commit", "-qm", "grandchild proof")

    child = tmp_path / "child"
    child.mkdir()
    _git(child, "init", "-qb", "main")
    _git(child, "config", "user.name", "Nested Validation Test")
    _git(
        child,
        "config",
        "user.email",
        "nested-validation@example.invalid",
    )
    (child / "child.txt").write_text("child\n", encoding="utf-8")
    _git(child, "add", "child.txt")
    _git(child, "commit", "-qm", "child base")
    _git(
        child,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(grandchild),
        "nested/docs",
    )
    _git(child, "commit", "-qam", "add nested docs")

    inspected: dict[str, Path] = {}

    def inspect_nested_checkout(workspace: Path) -> None:
        inspected["workspace"] = workspace
        assert (workspace / "libs/child/nested/docs/proof.txt").read_text(
            encoding="utf-8"
        ) == "nested proof\n"
        assert _git(workspace / "libs/child", "rev-parse", "HEAD") == _git(
            workspace,
            "rev-parse",
            "HEAD:libs/child",
        )
        assert _git(
            workspace / "libs/child/nested/docs",
            "rev-parse",
            "HEAD",
        ) == _git(
            workspace / "libs/child",
            "rev-parse",
            "HEAD:nested/docs",
        )

    scheduler = _RecordingScheduler(inspect_nested_checkout)
    daemon, repo, _task, _commit, _tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "libs/child",
    )
    _git(repo, "commit", "-qam", "add nested child")
    commit = _git(repo, "rev-parse", "HEAD")
    tree_id = f"git-tree:{_git(repo, 'rev-parse', 'HEAD^{tree}')}"

    # Mirror the provider shape: the primary checkout has the top-level
    # dependency but intentionally leaves its nested docs gitlink absent,
    # while a separate local root worktree has the exact nested objects.
    provider_workspace = tmp_path / "provider-worktree"
    _git(
        repo,
        "worktree",
        "add",
        "--detach",
        str(provider_workspace),
        commit,
    )
    _git(
        provider_workspace,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
        "--",
        "libs/child",
    )
    nested_source = provider_workspace / "libs/child/nested/docs"
    assert not (repo / "libs/child/nested/docs/.git").exists()
    assert (nested_source / "proof.txt").read_text(encoding="utf-8") == (
        "nested proof\n"
    )
    task = PortalTask(
        task_id="PMV-001",
        title="Validate nested local dependencies",
        status="todo",
        completion="manual",
        priority="P0",
        track="quality",
        validation=["test -f libs/child/nested/docs/proof.txt"],
        metadata={
            "Provider role": "deterministic-only",
            "submodules": "libs/child",
        },
    )
    real_run = implementation_daemon_module.subprocess.run

    def reject_network_git(arguments: Any, *args: Any, **kwargs: Any):
        command = [str(item) for item in arguments]
        forbidden = (
            command[:2] in (["git", "fetch"], ["git", "clone"])
            or command[:3] == ["git", "submodule", "update"]
        )
        assert not forbidden, command
        return real_run(arguments, *args, **kwargs)

    monkeypatch.setattr(
        implementation_daemon_module.subprocess,
        "run",
        reject_network_git,
    )

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is True
    assert evidence["validation_result"]["offline_submodule_paths"] == [
        "libs/child"
    ]
    after_proof = evidence["validation_result"]["fence"]["after"][
        "tracked_checkout_proof"
    ]
    assert after_proof["passed"] is True
    assert after_proof["materialized_gitlink_count"] == 2
    assert {
        item["repository"] for item in after_proof["repositories"]
    } == {".", "libs/child", "libs/child/nested/docs"}
    workspace = inspected["workspace"]
    assert not workspace.exists()
    assert str(workspace / "libs/child") not in _git(
        repo / "libs/child",
        "worktree",
        "list",
        "--porcelain",
    )
    assert str(workspace / "libs/child/nested/docs") not in _git(
        nested_source,
        "worktree",
        "list",
        "--porcelain",
    )


def test_exact_post_merge_runtime_allows_clean_unmaterialized_gitlink(
    tmp_path: Path,
) -> None:
    child = tmp_path / "optional-child"
    child.mkdir()
    _git(child, "init", "-qb", "main")
    _git(child, "config", "user.name", "Optional Dependency Test")
    _git(
        child,
        "config",
        "user.email",
        "optional-dependency@example.invalid",
    )
    (child / "optional.txt").write_text("optional\n", encoding="utf-8")
    _git(child, "add", "optional.txt")
    _git(child, "commit", "-qm", "optional dependency")

    scheduler = _RecordingScheduler()
    daemon, repo, task, _commit, _tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "libs/optional",
    )
    _git(repo, "commit", "-qam", "add optional dependency")
    commit = _git(repo, "rev-parse", "HEAD")
    tree_id = f"git-tree:{_git(repo, 'rev-parse', 'HEAD^{tree}')}"

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is True
    proof = evidence["validation_result"]["fence"]["after"][
        "tracked_checkout_proof"
    ]
    assert proof["passed"] is True
    assert proof["materialized_gitlink_count"] == 0
    assert {item["repository"] for item in proof["repositories"]} == {"."}
    assert not scheduler.calls[0]["workspace"].exists()


@pytest.mark.parametrize("fence", ("target", "workspace"))
def test_exact_post_merge_runtime_rejects_changed_fence(
    tmp_path: Path,
    fence: str,
) -> None:
    future_commit = ""

    def mutate(workspace: Path) -> None:
        if fence == "target":
            _git(
                workspace,
                "update-ref",
                "refs/heads/main",
                future_commit,
            )
        else:
            (workspace / "tracked.txt").write_text(
                "changed during validation\n",
                encoding="utf-8",
            )

    scheduler = _RecordingScheduler(mutate)
    daemon, repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    if fence == "target":
        _git(repo, "checkout", "-qb", "future")
        (repo / "future.txt").write_text("future\n", encoding="utf-8")
        _git(repo, "add", "future.txt")
        _git(repo, "commit", "-qm", "future")
        future_commit = _git(repo, "rev-parse", "HEAD")
        _git(repo, "checkout", "-q", "main")

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is False
    assert evidence["stale"] is True
    nested = evidence["validation_result"]
    expected_reason = (
        "post_merge_validation_target_changed_after_execution"
        if fence == "target"
        else "post_merge_validation_workspace_dirty_after_execution"
    )
    assert nested["reason"] == expected_reason
    verified, reasons = verify_post_merge_validation_evidence(evidence)
    assert verified is True
    assert reasons == ()


@pytest.mark.parametrize(
    ("index_flag", "expected_tag"),
    (
        ("--assume-unchanged", "h"),
        ("--skip-worktree", "S"),
    ),
)
def test_exact_post_merge_runtime_rejects_hidden_tracked_mutation(
    tmp_path: Path,
    index_flag: str,
    expected_tag: str,
) -> None:
    def conceal_mutation(workspace: Path) -> None:
        if index_flag == "--skip-worktree":
            _git(workspace, "update-index", index_flag, "tracked.txt")
        (workspace / "tracked.txt").write_text(
            "validation-time mutation\n",
            encoding="utf-8",
        )
        if index_flag == "--assume-unchanged":
            _git(workspace, "update-index", index_flag, "tracked.txt")
        assert _git(
            workspace,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ) == ""
        assert _git(
            workspace,
            "ls-files",
            "-v",
            "--",
            "tracked.txt",
        ).startswith(f"{expected_tag} ")

    scheduler = _RecordingScheduler(conceal_mutation)
    daemon, _repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is False
    assert evidence["stale"] is True
    validation_result = evidence["validation_result"]
    assert validation_result["reason"] == (
        "post_merge_validation_workspace_integrity_failed_after_execution"
    )
    after = validation_result["fence"]["after"]
    assert after["workspace_clean"] is True
    assert after["tracked_checkout_exact"] is False
    proof = after["tracked_checkout_proof"]
    assert proof["reason"] == (
        "post_merge_tracked_index_flags_forbidden"
    )
    assert proof["failure_repository"] == "."
    assert proof["forbidden_index_flags"] == [
        {"path": "tracked.txt", "flag": expected_tag}
    ]


@pytest.mark.parametrize(
    "concealment",
    ("stat_cache", "ignored_filemode"),
)
def test_exact_post_merge_runtime_hashes_tracked_bytes_and_modes(
    tmp_path: Path,
    concealment: str,
) -> None:
    def conceal_without_index_flag(workspace: Path) -> None:
        tracked = workspace / "tracked.txt"
        if concealment == "stat_cache":
            stable_ns = 1_000_000_000_000_000_000
            os.utime(tracked, ns=(stable_ns, stable_ns))
            _git(workspace, "update-index", "--refresh", "tracked.txt")
            _git(workspace, "config", "core.trustctime", "false")
            tracked.write_text("mutated!\n", encoding="utf-8")
            os.utime(tracked, ns=(stable_ns, stable_ns))
        else:
            _git(workspace, "config", "core.filemode", "false")
            tracked.chmod(tracked.stat().st_mode | 0o100)
        assert _git(workspace, "status", "--porcelain") == ""
        assert _git(
            workspace,
            "ls-files",
            "-v",
            "--",
            "tracked.txt",
        ).startswith("H ")

    scheduler = _RecordingScheduler(conceal_without_index_flag)
    daemon, _repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is False
    validation_result = evidence["validation_result"]
    assert validation_result["reason"] == (
        "post_merge_validation_workspace_integrity_failed_after_execution"
    )
    after = validation_result["fence"]["after"]
    assert after["workspace_clean"] is True
    proof = after["tracked_checkout_proof"]
    assert proof["reason"] == "post_merge_tracked_worktree_mismatch"
    assert proof["failure_repository"] == "."
    assert proof["failure_path"] == "tracked.txt"
    if concealment == "ignored_filemode":
        assert proof["path_error"] == "tracked executable mode mismatch"
    else:
        assert proof["expected_object_id"] != proof["observed_object_id"]


@pytest.mark.parametrize(
    ("target_mode", "expected_pass"),
    (
        pytest.param(
            0o755,
            False,
            id="100755-to-0654-rejected",
        ),
        pytest.param(
            0o644,
            True,
            id="100644-to-0654-equivalent",
        ),
    ),
)
def test_exact_post_merge_runtime_uses_owner_execute_for_git_mode(
    tmp_path: Path,
    target_mode: int,
    expected_pass: bool,
) -> None:
    def apply_group_only_execute(workspace: Path) -> None:
        _git(workspace, "config", "core.filemode", "false")
        tracked = workspace / "tracked.txt"
        tracked.chmod(0o654)
        assert tracked.stat().st_mode & 0o777 == 0o654
        assert _git(workspace, "status", "--porcelain") == ""
        assert _git(
            workspace,
            "ls-files",
            "-v",
            "--",
            "tracked.txt",
        ).startswith("H ")

    scheduler = _RecordingScheduler(apply_group_only_execute)
    daemon, repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    if target_mode == 0o755:
        (repo / "tracked.txt").chmod(target_mode)
        _git(repo, "add", "tracked.txt")
        _git(repo, "commit", "-qm", "make tracked file executable")
        commit = _git(repo, "rev-parse", "HEAD")
        tree_id = f"git-tree:{_git(repo, 'rev-parse', 'HEAD^{tree}')}"
    assert _git(
        repo,
        "ls-tree",
        commit,
        "--",
        "tracked.txt",
    ).split()[0] == f"100{target_mode:o}"

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is expected_pass
    validation_result = evidence["validation_result"]
    if expected_pass:
        assert validation_result["fence"]["after"][
            "tracked_checkout_exact"
        ] is True
        assert verify_post_merge_validation_evidence(
            evidence,
            expected_task_id=task.task_id,
            expected_target_commit=commit,
            expected_repository_tree_id=tree_id,
        ) == (True, ())
    else:
        assert validation_result["reason"] == (
            "post_merge_validation_workspace_integrity_failed_after_execution"
        )
        proof = validation_result["fence"]["after"][
            "tracked_checkout_proof"
        ]
        assert proof["reason"] == "post_merge_tracked_worktree_mismatch"
        assert proof["failure_path"] == "tracked.txt"
        assert proof["path_error"] == "tracked executable mode mismatch"


def test_exact_post_merge_runtime_rejects_hidden_nested_submodule_mutation(
    tmp_path: Path,
) -> None:
    child = tmp_path / "hidden-child"
    child.mkdir()
    _git(child, "init", "-qb", "main")
    _git(child, "config", "user.name", "Nested Integrity Test")
    _git(
        child,
        "config",
        "user.email",
        "nested-integrity@example.invalid",
    )
    (child / "child.txt").write_text("child baseline\n", encoding="utf-8")
    _git(child, "add", "child.txt")
    _git(child, "commit", "-qm", "child baseline")

    def conceal_nested_mutation(workspace: Path) -> None:
        nested = workspace / "libs/child"
        (nested / "child.txt").write_text(
            "hidden nested mutation\n",
            encoding="utf-8",
        )
        _git(
            nested,
            "update-index",
            "--assume-unchanged",
            "child.txt",
        )
        assert _git(nested, "status", "--porcelain") == ""
        assert _git(workspace, "status", "--porcelain") == ""

    scheduler = _RecordingScheduler(conceal_nested_mutation)
    daemon, repo, _task, _commit, _tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "libs/child",
    )
    _git(repo, "commit", "-qam", "add integrity child")
    commit = _git(repo, "rev-parse", "HEAD")
    tree_id = f"git-tree:{_git(repo, 'rev-parse', 'HEAD^{tree}')}"
    task = PortalTask(
        task_id="PMV-001",
        title="Validate nested tracked checkout integrity",
        status="todo",
        completion="manual",
        priority="P0",
        track="quality",
        validation=["test -f libs/child/child.txt"],
        metadata={
            "Provider role": "deterministic-only",
            "submodules": "libs/child",
        },
    )

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is False
    validation_result = evidence["validation_result"]
    assert validation_result["reason"] == (
        "post_merge_validation_workspace_integrity_failed_after_execution"
    )
    after = validation_result["fence"]["after"]
    assert after["workspace_clean"] is True
    proof = after["tracked_checkout_proof"]
    assert proof["reason"] == (
        "post_merge_tracked_index_flags_forbidden"
    )
    assert proof["failure_repository"] == "libs/child"
    assert proof["forbidden_index_flags"] == [
        {"path": "child.txt", "flag": "h"}
    ]
    assert not scheduler.calls[0]["workspace"].exists()


def test_merge_callback_uses_fresh_post_merge_receipt_for_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _RecordingScheduler()
    daemon, _repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    evidence = build_post_merge_validation_evidence(
        task_id=task.task_id,
        target_commit=commit,
        repository_tree_id=tree_id,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
        },
    )
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        daemon,
        "_reject_protected_merge_candidate",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_rehydrate_merge_request_branch",
        lambda **_kwargs: {"ready": True, "rehydrated": False},
    )
    monkeypatch.setattr(
        daemon,
        "_merge_branch_to_main",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "merged": True,
            "returncode": 0,
            "merge_commit": commit,
        },
    )
    monkeypatch.setattr(
        daemon,
        "_validate_exact_post_merge_commit",
        lambda selected_task, **kwargs: observed.update(
            {"validated_task": selected_task, "validation_kwargs": kwargs}
        )
        or evidence,
    )
    monkeypatch.setattr(
        daemon,
        "_mark_task_completed_in_todo",
        lambda task_id, **kwargs: observed.update(
            {"completed_task_id": task_id, "completion_kwargs": kwargs}
        )
        or {
            "updated": True,
            "task_id": task_id,
            "reason": "updated",
        },
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_completion",
        lambda *_args, **_kwargs: None,
    )
    request = SimpleNamespace(
        branch_name="implementation/pmv-001",
        commit_sha=commit,
        task_id=task.task_id,
        priority=task.priority,
        attempt=1,
        target_repository_id=daemon.merge_target_repository_id,
        target_branch=daemon.resolved_merge_target_branch,
        metadata={
            "target_binding_schema": MERGE_TARGET_BINDING_SCHEMA,
            "target_repository_id": daemon.merge_target_repository_id,
            "target_branch": daemon.resolved_merge_target_branch,
            "validation_proof": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "selection": {"scope": "pre_merge"},
            },
            "task": {
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status,
                "completion": task.completion,
                "priority": task.priority,
                "track": task.track,
                "validation": list(task.validation),
                "metadata": dict(task.metadata),
            },
        },
    )

    result = daemon._merge_train_callback(request)

    assert result["merged"] is True
    assert result["post_merge_validation"] == evidence
    assert observed["validation_kwargs"] == {
        "target_commit": commit,
        "repository_tree_id": tree_id,
    }
    assert observed["completed_task_id"] == task.task_id
    assert observed["completion_kwargs"]["expected_target_commit"] == commit
    assert result["todo_update_result"]["updated"] is True


def test_reconciliation_reruns_exact_validation_instead_of_replaying_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _RecordingScheduler()
    daemon, _repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    fresh = build_post_merge_validation_evidence(
        task_id=task.task_id,
        target_commit=commit,
        repository_tree_id=tree_id,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
        },
    )
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        daemon,
        "_validate_exact_post_merge_commit",
        lambda selected_task, **kwargs: observed.update(
            {"validated_task": selected_task, "validation_kwargs": kwargs}
        )
        or fresh,
    )
    stale_event = {
        "task_id": task.task_id,
        "merge_result": {
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "validation_scope": "pre_merge",
            }
        },
    }

    result = daemon._run_reconciled_post_merge_completion_gate(
        task,
        target_commit=commit,
        target_branch="main",
        prerequisites_passed=True,
    )

    assert stale_event["merge_result"]["validation_result"] != fresh
    assert result["validation"]["passed"] is True
    assert result["validation"]["evidence"] == fresh
    assert observed["validation_kwargs"] == {
        "target_commit": commit,
        "repository_tree_id": tree_id,
    }
