from __future__ import annotations

import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
    task_declared_output_paths,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"git {' '.join(args)} failed in {repo}:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    return result.stdout.strip()


def _init_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Invariant Test")
    _git(path, "config", "user.email", "invariant@example.invalid")
    return path


def _init_parent_with_submodule(tmp_path: Path) -> tuple[Path, Path]:
    child = _init_repo(tmp_path / "child")
    (child / "base.txt").write_text("base\n", encoding="utf-8")
    _git(child, "add", "base.txt")
    _git(child, "commit", "-m", "child base")

    repo = _init_repo(tmp_path / "repo")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "libs/child",
    )
    _git(repo, "commit", "-am", "add child submodule")
    return repo, repo / "libs" / "child"


def _daemon(
    repo: Path,
    *,
    worktree_submodule_paths: tuple[str, ...] = (),
    task_header_prefix: str = "TODO-",
    implementation_protected_paths: tuple[str, ...] = (),
) -> TodoImplementationDaemon:
    state_dir = repo.parent / f".{repo.name}-declared-output-invariant-state"
    return TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix=task_header_prefix,
        worktree_submodule_paths=worktree_submodule_paths,
        worktree_pool_enabled=False,
        implementation_protected_paths=implementation_protected_paths,
    )


def _task(task_id: str, output: str) -> PortalTask:
    return PortalTask(
        task_id=task_id,
        title=f"Produce {output}",
        status="todo",
        completion="manual",
        priority="P0",
        track="verification",
        outputs=[output],
    )


def _proposal_task(task_id: str, *outputs: str) -> PortalTask:
    return PortalTask(
        task_id=task_id,
        title=f"Produce {', '.join(outputs)}",
        status="todo",
        completion="manual",
        priority="P0",
        track="verification",
        outputs=list(outputs),
        validation=["python -m pytest"],
    )


def _null_merge_scenario(
    tmp_path: Path,
    *,
    output: str = "deliverable.txt",
    undeclared_candidate_output: str = "",
    executable_output: bool = False,
) -> tuple[Path, PortalTask, str, str, str, str]:
    """Build the exact two-parent, first-parent-tree loss shape."""

    repo = _init_repo(tmp_path / "repo")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-b", "implementation/out-repair")
    candidate_output = repo / output
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_output.write_text("candidate\n", encoding="utf-8")
    if executable_output:
        candidate_output.chmod(0o755)
    candidate_paths = [output]
    if undeclared_candidate_output:
        undeclared_output = repo / undeclared_candidate_output
        undeclared_output.parent.mkdir(parents=True, exist_ok=True)
        undeclared_output.write_text("undeclared\n", encoding="utf-8")
        candidate_paths.append(undeclared_candidate_output)
    _git(repo, "add", *candidate_paths)
    _git(repo, "commit", "-m", "OUT-REPAIR: add declared output")
    candidate = _git(repo, "rev-parse", "HEAD")
    candidate_tree = _git(repo, "rev-parse", f"{candidate}^{{tree}}")

    _git(repo, "checkout", "main")
    baseline_tree = _git(repo, "rev-parse", f"{baseline}^{{tree}}")
    null_merge = _git(
        repo,
        "commit-tree",
        baseline_tree,
        "-p",
        baseline,
        "-p",
        candidate,
        "-m",
        "discard candidate tree",
    )
    _git(repo, "update-ref", "refs/heads/main", null_merge, baseline)

    task = PortalTask(
        task_id="OUT-REPAIR",
        title="Recover the declared output",
        status="todo",
        completion="manual",
        priority="P0",
        track="verification",
        outputs=[output],
        validation=[
            "python -c \"assert(__import__('pathlib')."
            f"Path('{output}').read_bytes()=="
            "b'candidate'+bytes([10]))\""
        ],
    )
    return repo, task, baseline, candidate, candidate_tree, null_merge


def test_declared_output_tracking_invariant_accepts_tracked_root_file(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "add tracked output")

    result = _daemon(repo)._declared_output_tracking_invariant(
        [_task("OUT-001", "tracked.txt")],
        workspace_path=repo,
    )

    assert result["passed"] is True
    assert result["mode"] == "workspace_index"
    assert result["missing_outputs"] == []
    assert result["untracked_outputs"] == []
    assert result["checks"][0]["reason"] == "declared_output_tracked"
    assert result["checks"][0]["repository"] == "."


def test_declared_output_tracking_invariant_rejects_ignored_untracked_root_file(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore generated output")
    (repo / "ignored.txt").write_text("present but ignored\n", encoding="utf-8")

    result = _daemon(repo)._declared_output_tracking_invariant(
        [_task("OUT-002", "ignored.txt")],
        workspace_path=repo,
    )

    assert result["passed"] is False
    assert result["missing_outputs"] == []
    assert result["untracked_outputs"] == [
        {"task_id": "OUT-002", "path": "ignored.txt"}
    ]
    assert result["checks"][0]["exists"] is True
    assert result["checks"][0]["tracked"] is False
    assert result["checks"][0]["reason"] == "declared_output_untracked"


def test_declared_output_tracking_invariant_rejects_repository_dot(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")

    result = _daemon(repo)._declared_output_tracking_invariant(
        [_task("OUT-DOT", ".")],
        repository_ref=_git(repo, "rev-parse", "HEAD"),
    )

    assert result["passed"] is False
    assert result["unsafe_outputs"] == [
        {"task_id": "OUT-DOT", "path": "."}
    ]
    assert result["checks"][0]["reason"] == "declared_output_path_unsafe"


def test_declared_output_tracking_invariant_binds_exact_tree_not_dirty_index(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    target_commit = _git(repo, "rev-parse", "HEAD")

    dirty_output = repo / "staged-after-target.txt"
    dirty_output.write_text("not in the target tree\n", encoding="utf-8")
    _git(repo, "add", dirty_output.name)

    result = _daemon(repo)._declared_output_tracking_invariant(
        [_task("OUT-003", dirty_output.name)],
        repository_ref=target_commit,
    )

    assert dirty_output.exists()
    assert result["passed"] is False
    assert result["mode"] == "repository_tree"
    assert result["repository_ref"] == target_commit
    assert result["missing_outputs"] == [
        {"task_id": "OUT-003", "path": dirty_output.name}
    ]
    assert result["untracked_outputs"] == []
    assert result["checks"][0]["exists"] is False
    assert result["checks"][0]["tracked"] is False
    assert result["checks"][0]["repository_ref"] == target_commit


def test_declared_output_tracking_invariant_checks_recorded_submodule_tree(
    tmp_path: Path,
) -> None:
    child = _init_repo(tmp_path / "child")
    (child / "tracked-proof.txt").write_text("proof\n", encoding="utf-8")
    _git(child, "add", "tracked-proof.txt")
    _git(child, "commit", "-m", "add tracked proof")

    repo = _init_repo(tmp_path / "repo")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "libs/child",
    )
    _git(repo, "commit", "-am", "add child submodule")
    target_commit = _git(repo, "rev-parse", "HEAD")
    recorded_child_commit = _git(
        repo,
        "rev-parse",
        f"{target_commit}:libs/child",
    )

    dirty_missing = repo / "libs" / "child" / "missing-proof.txt"
    dirty_missing.write_text("not recorded by the gitlink\n", encoding="utf-8")
    daemon = _daemon(repo, worktree_submodule_paths=("libs/child",))

    tracked = daemon._declared_output_tracking_invariant(
        [_task("OUT-004", "libs/child/tracked-proof.txt")],
        repository_ref=target_commit,
    )
    missing = daemon._declared_output_tracking_invariant(
        [_task("OUT-005", "libs/child/missing-proof.txt")],
        repository_ref=target_commit,
    )

    assert tracked["passed"] is True
    assert tracked["checks"][0]["repository"] == "libs/child"
    assert tracked["checks"][0]["tracked_path"] == "tracked-proof.txt"
    assert tracked["checks"][0]["repository_ref"] == recorded_child_commit
    assert tracked["checks"][0]["reason"] == "declared_output_tracked"

    assert dirty_missing.exists()
    assert missing["passed"] is False
    assert missing["missing_outputs"] == [
        {
            "task_id": "OUT-005",
            "path": "libs/child/missing-proof.txt",
        }
    ]
    assert missing["checks"][0]["repository"] == "libs/child"
    assert missing["checks"][0]["repository_ref"] == recorded_child_commit
    assert missing["checks"][0]["exists"] is False
    assert missing["checks"][0]["reason"] == "declared_output_missing"


def test_proposal_expected_outputs_use_managed_child_ignore_and_index(
    tmp_path: Path,
) -> None:
    repo, child = _init_parent_with_submodule(tmp_path)
    (repo / ".gitignore").write_text("core\n*.json\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "parent-only ignore rules")
    baseline = _git(repo, "rev-parse", "HEAD")

    contract = child / "core" / "host_contracts.py"
    contract.parent.mkdir()
    contract.write_text("CONTRACT_VERSION = 1\n", encoding="utf-8")
    manifest = child / "fixtures" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    _git(child, "add", "core/host_contracts.py", "fixtures/manifest.json")
    _git(child, "commit", "-m", "commit nested expected outputs")
    outputs = (
        "libs/child/core/host_contracts.py",
        "libs/child/fixtures/manifest.json",
    )
    task = _proposal_task("OUT-SUBMODULE-OWNER", *outputs)
    daemon = _daemon(repo, worktree_submodule_paths=("libs/child",))

    for relative in outputs:
        assert subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", relative],
            cwd=repo,
            check=False,
        ).returncode == 0
        assert subprocess.run(
            [
                "git",
                "check-ignore",
                "--no-index",
                "--quiet",
                "--",
                relative.removeprefix("libs/child/"),
            ],
            cwd=child,
            check=False,
        ).returncode == 1

    preflight = daemon._prepare_proposal_expected_outputs(
        repo,
        task,
        baseline_ref=baseline,
        scope_paths=outputs,
    )
    checks = {item["path"]: item for item in preflight["checks"]}
    assert preflight["staged_paths"] == []
    assert all(checks[path]["repository"] == "libs/child" for path in outputs)
    assert all(checks[path]["git_owner_valid"] is True for path in outputs)
    assert all(checks[path]["ignored"] is False for path in outputs)
    assert all(checks[path]["issue"] == "" for path in outputs)

    proposal = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline,
    )

    assert proposal.accepted is True
    assert proposal.proposal.changed_paths == outputs
    assert _git(child, "diff", "--cached", "--name-only") == ""


def test_proposal_force_stages_only_exact_managed_child_ignored_output(
    tmp_path: Path,
) -> None:
    repo, child = _init_parent_with_submodule(tmp_path)
    (child / ".gitignore").write_text("artifacts/*.json\n", encoding="utf-8")
    _git(child, "add", ".gitignore")
    _git(child, "commit", "-m", "ignore child artifacts")
    _git(repo, "add", "libs/child")
    _git(repo, "commit", "-m", "record child ignore policy")
    baseline = _git(repo, "rev-parse", "HEAD")

    deliverable = child / "artifacts" / "proof.json"
    deliverable.parent.mkdir()
    deliverable.write_text('{"proved": true}\n', encoding="utf-8")
    unrelated = child / "artifacts" / "unrelated.json"
    unrelated.write_text('{"unrelated": true}\n', encoding="utf-8")
    output = "libs/child/artifacts/proof.json"
    daemon = _daemon(repo, worktree_submodule_paths=("libs/child",))

    proposal = daemon._validate_implementation_patch(
        repo,
        _proposal_task("OUT-SUBMODULE-IGNORED", output),
        baseline_ref=baseline,
    )

    assert proposal.accepted is True
    assert proposal.proposal.changed_paths == (output,)
    assert _git(child, "diff", "--cached", "--name-only") == (
        "artifacts/proof.json"
    )
    assert _git(repo, "ls-files", "--", output) == ""


def test_proposal_does_not_force_stage_unmanaged_child_output(
    tmp_path: Path,
) -> None:
    repo, child = _init_parent_with_submodule(tmp_path)
    (child / ".gitignore").write_text("artifacts/*.json\n", encoding="utf-8")
    _git(child, "add", ".gitignore")
    _git(child, "commit", "-m", "ignore child artifacts")
    _git(repo, "add", "libs/child")
    _git(repo, "commit", "-m", "record child ignore policy")
    baseline = _git(repo, "rev-parse", "HEAD")
    deliverable = child / "artifacts" / "proof.json"
    deliverable.parent.mkdir()
    deliverable.write_text('{"proved": true}\n', encoding="utf-8")

    proposal = _daemon(repo)._validate_implementation_patch(
        repo,
        _proposal_task(
            "OUT-SUBMODULE-UNMANAGED",
            "libs/child/artifacts/proof.json",
        ),
        baseline_ref=baseline,
    )

    assert proposal.accepted is False
    assert "expected_output_ignored_or_unstaged" in {
        finding.code.value for finding in proposal.findings
    }
    assert _git(child, "diff", "--cached", "--name-only") == ""


def test_proposal_and_commit_force_stage_only_exact_ignored_declared_output(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("*.json\n", encoding="utf-8")
    (repo / "implementation.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "implementation.py")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "deliverable.json").write_text(
        '{"certified": true}\n',
        encoding="utf-8",
    )
    (repo / "unrelated.json").write_text(
        '{"must_not_be_staged": true}\n',
        encoding="utf-8",
    )
    daemon = _daemon(repo)
    task = _proposal_task(
        "OUT-006",
        "implementation.py",
        "deliverable.json",
    )

    proposal = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline,
    )

    assert proposal.accepted is True
    assert proposal.proposal.changed_paths == (
        "deliverable.json",
        "implementation.py",
    )
    assert _git(repo, "diff", "--cached", "--name-only") == "deliverable.json"

    result = daemon._commit_worktree_changes(
        repo,
        task,
        1,
        baseline_ref=baseline,
    )

    assert result["committed"] is True
    candidate = result["commit"]
    assert candidate != baseline
    assert _git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        candidate,
    ).splitlines() == ["deliverable.json", "implementation.py"]
    assert _git(repo, "show", f"{candidate}:deliverable.json") == (
        '{"certified": true}'
    )
    assert _git(repo, "show", f"{candidate}:implementation.py") == "VALUE = 1"
    assert _git(
        repo,
        "ls-tree",
        "--name-only",
        candidate,
        "--",
        "unrelated.json",
    ) == ""


def test_proposal_and_commit_reject_missing_declared_output_with_stable_reason(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "implementation.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repo, "add", "implementation.py")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    daemon = _daemon(repo)
    task = _proposal_task(
        "OUT-006-MISSING",
        "implementation.py",
        "missing.json",
    )

    proposal = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline,
    )

    assert proposal.accepted is False
    assert {
        finding.code.value for finding in proposal.findings
    } == {"expected_output_ignored_or_unstaged"}
    assert [finding.path for finding in proposal.findings] == ["missing.json"]
    assert _git(repo, "diff", "--cached", "--name-only") == ""

    result = daemon._commit_worktree_changes(
        repo,
        task,
        1,
        baseline_ref=baseline,
    )

    assert result["committed"] is False
    assert result["reason"] == "expected_output_ignored_or_unstaged"
    assert result["declared_output_invariant"]["missing_outputs"] == [
        {"task_id": "OUT-006-MISSING", "path": "missing.json"}
    ]
    assert _git(repo, "rev-parse", "HEAD") == baseline


def test_proposal_and_commit_never_force_stage_protected_ignored_output(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("*.json\n", encoding="utf-8")
    (repo / "implementation.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "implementation.py")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "protected.json").write_text(
        '{"protected": true}\n',
        encoding="utf-8",
    )
    daemon = _daemon(
        repo,
        implementation_protected_paths=("protected.json",),
    )
    task = _proposal_task(
        "OUT-006-PROTECTED",
        "implementation.py",
        "protected.json",
    )

    proposal = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline,
    )

    assert proposal.accepted is False
    assert {
        finding.code.value for finding in proposal.findings
    } == {"expected_output_ignored_or_unstaged"}
    assert [finding.path for finding in proposal.findings] == [
        "protected.json"
    ]
    assert _git(repo, "diff", "--cached", "--name-only") == ""

    result = daemon._commit_worktree_changes(
        repo,
        task,
        1,
        baseline_ref=baseline,
    )

    assert result["committed"] is False
    assert result["reason"] == "expected_output_ignored_or_unstaged"
    assert result["declared_output_invariant"]["untracked_outputs"] == [
        {"task_id": "OUT-006-PROTECTED", "path": "protected.json"}
    ]
    assert _git(repo, "rev-parse", "HEAD") == baseline


def test_merge_callback_skips_completion_when_ignored_output_is_not_in_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Tasks

## OUT-007 Require the merged deliverable

- Status: todo
- Completion: manual
- Priority: P0
- Track: verification
- Outputs: deliverable.json
- Acceptance: The declared JSON is present in the merged tree.
""",
        encoding="utf-8",
    )
    (repo / ".gitignore").write_text("*.json\n", encoding="utf-8")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "base.txt", "todo.md")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")

    branch = "implementation/out-007"
    _git(repo, "checkout", "-b", branch)
    (repo / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "deliverable.json").write_text(
        '{"present_only_in_checkout": true}\n',
        encoding="utf-8",
    )
    _git(repo, "add", "implementation.py")
    _git(repo, "commit", "-m", "OUT-007: implementation without artifact")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")

    daemon = _daemon(repo, task_header_prefix="OUT-")
    task = daemon._load_tasks()[0]
    request, _ = daemon._enqueue_merge_candidate(
        branch_name=branch,
        implementation_commit=candidate,
        baseline_ref=baseline,
        worktree_path=None,
        task=task,
        attempt=1,
    )

    def integrate_candidate(selected_branch, *_args, **_kwargs):
        _git(repo, "merge", "--ff-only", selected_branch)
        integration_commit = _git(repo, "rev-parse", "HEAD")
        _git(repo, "add", "-f", "deliverable.json")
        _git(repo, "commit", "-m", "unrelated later artifact")
        return {
            "merged": True,
            "returncode": 0,
            "merge_commit": integration_commit,
        }

    monkeypatch.setattr(
        daemon,
        "_merge_branch_to_main",
        integrate_candidate,
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_completion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completion must not run")
        ),
    )

    result = daemon._merge_train_callback(request)

    assert result["merged"] is False
    assert result["reason"] == "post_merge_declared_outputs_missing"
    assert result["completion_skipped"] is True
    assert result["integration_occurred"] is True
    assert result["post_merge_declared_output_invariant"][
        "missing_outputs"
    ] == [{"task_id": "OUT-007", "path": "deliverable.json"}]
    assert _git(repo, "merge-base", "--is-ancestor", candidate, "main") == ""
    assert _git(repo, "cat-file", "-e", "main:deliverable.json") == ""
    assert "- Status: todo" in todo_path.read_text(encoding="utf-8")


def test_exact_null_merge_declared_output_recovery_adopts_identical_untracked(
    tmp_path: Path,
) -> None:
    output = "artifacts/deliverable.txt"
    (
        repo,
        task,
        baseline,
        candidate,
        candidate_tree,
        null_merge,
    ) = _null_merge_scenario(tmp_path, output=output)
    untracked_output = repo / output
    untracked_output.parent.mkdir(parents=True, exist_ok=True)
    untracked_output.write_bytes(
        subprocess.run(
            ["git", "show", f"{candidate}:{output}"],
            cwd=repo,
            capture_output=True,
            check=True,
        ).stdout
    )
    candidate_blob = _git(repo, "rev-parse", f"{candidate}:{output}")
    assert _git(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ) == f"?? {output}"

    daemon = _daemon(repo, task_header_prefix="OUT-")
    result = daemon._repair_post_merge_declared_outputs_locked(
        [task],
        primary_task=task,
        attempt=3,
        candidate_commit=candidate,
        candidate_tree=candidate_tree,
        baseline_ref=baseline,
        target_branch="main",
        target_commit=null_merge,
    )

    repair_commit = _git(repo, "rev-parse", "main")
    assert result["passed"] is True, result.get("reason")
    assert result["reason"] == "post_merge_declared_outputs_repaired"
    assert result["attempted"] is True
    assert result["failed_integration_commit"] == null_merge
    assert result["repair_commit"] == repair_commit
    assert repair_commit != null_merge
    assert _git(repo, "rev-list", "--count", f"{null_merge}..main") == "1"
    assert _git(repo, "rev-parse", f"{repair_commit}^") == null_merge
    assert _git(repo, "diff", "--name-only", null_merge, repair_commit) == output
    assert _git(repo, "rev-parse", f"{repair_commit}:{output}") == candidate_blob
    assert _git(repo, "ls-files", "--error-unmatch", "--", output) == output
    assert untracked_output.read_text(encoding="utf-8") == "candidate\n"
    assert _git(repo, "status", "--porcelain") == ""

    assert len(result["validation"]) == 1
    validation = result["validation"][0]
    assert validation["task_id"] == task.task_id
    assert validation["result"]["attempted"] is True
    assert validation["result"]["passed"] is True
    assert result["staged_declared_output_invariant"]["passed"] is True
    assert result["repaired_declared_output_invariant"]["passed"] is True
    assert daemon._declared_output_tracking_invariant(
        [task],
        repository_ref=repair_commit,
    )["passed"] is True
    assert result["receipt"]["candidate_commit"] == candidate
    assert result["receipt"]["repair_parent_commit"] == null_merge
    assert result["receipt"]["entries"] == [
        {
            "path": output,
            "mode": "100644",
            "object_type": "blob",
            "object_id": candidate_blob,
        }
    ]


def test_null_merge_recovery_rejects_divergent_untracked_without_mutation(
    tmp_path: Path,
) -> None:
    output = "deliverable.txt"
    (
        repo,
        task,
        baseline,
        candidate,
        candidate_tree,
        null_merge,
    ) = _null_merge_scenario(tmp_path, output=output)
    untracked_output = repo / output
    untracked_output.write_text("operator-owned divergence\n", encoding="utf-8")
    head_before = _git(repo, "rev-parse", "HEAD")
    tree_before = _git(repo, "write-tree")
    status_before = _git(repo, "status", "--porcelain=v1")
    content_before = untracked_output.read_bytes()

    result = _daemon(
        repo,
        task_header_prefix="OUT-",
    )._repair_post_merge_declared_outputs_locked(
        [task],
        primary_task=task,
        attempt=4,
        candidate_commit=candidate,
        candidate_tree=candidate_tree,
        baseline_ref=baseline,
        target_branch="main",
        target_commit=null_merge,
    )

    assert result["passed"] is False
    assert result["attempted"] is False
    assert result["reason"] == "repair_declared_output_content_conflict"
    assert [item["path"] for item in result["mismatched_files"]] == [output]
    assert _git(repo, "rev-parse", "HEAD") == head_before == null_merge
    assert _git(repo, "write-tree") == tree_before
    assert _git(repo, "status", "--porcelain=v1") == status_before
    assert untracked_output.read_bytes() == content_before
    assert _git(repo, "ls-files", "--", output) == ""


def test_null_merge_recovery_rejects_incomplete_candidate_replay(
    tmp_path: Path,
) -> None:
    output = "deliverable.txt"
    (
        repo,
        task,
        baseline,
        candidate,
        candidate_tree,
        null_merge,
    ) = _null_merge_scenario(
        tmp_path,
        output=output,
        undeclared_candidate_output="implementation.py",
    )
    (repo / output).write_text("candidate\n", encoding="utf-8")
    head_before = _git(repo, "rev-parse", "HEAD")
    status_before = _git(repo, "status", "--porcelain=v1")

    result = _daemon(
        repo,
        task_header_prefix="OUT-",
    )._repair_post_merge_declared_outputs_locked(
        [task],
        primary_task=task,
        attempt=5,
        candidate_commit=candidate,
        candidate_tree=candidate_tree,
        baseline_ref=baseline,
        target_branch="main",
        target_commit=null_merge,
    )

    assert result["passed"] is False
    assert result["attempted"] is False
    assert result["reason"] == (
        "repair_candidate_delta_not_exact_declared_additions"
    )
    assert result["candidate_delta_paths"] == [
        output,
        "implementation.py",
    ]
    assert result["expected_paths"] == [output]
    assert _git(repo, "rev-parse", "HEAD") == head_before == null_merge
    assert _git(repo, "status", "--porcelain=v1") == status_before
    assert _git(repo, "ls-files", "--", output) == ""


def test_null_merge_recovery_rejects_validation_staged_blob_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = "deliverable.txt"
    (
        repo,
        task,
        baseline,
        candidate,
        candidate_tree,
        null_merge,
    ) = _null_merge_scenario(tmp_path, output=output)
    (repo / output).write_text("candidate\n", encoding="utf-8")
    daemon = _daemon(repo, task_header_prefix="OUT-")

    def tamper_and_stage(workspace, *_args, **_kwargs):
        (workspace / output).write_text("tampered\n", encoding="utf-8")
        _git(workspace, "add", output)
        return {"attempted": True, "passed": True, "returncode": 0, "results": []}

    monkeypatch.setattr(
        daemon,
        "_run_validation_commands",
        tamper_and_stage,
    )
    result = daemon._repair_post_merge_declared_outputs_locked(
        [task],
        primary_task=task,
        attempt=6,
        candidate_commit=candidate,
        candidate_tree=candidate_tree,
        baseline_ref=baseline,
        target_branch="main",
        target_commit=null_merge,
    )

    assert result["passed"] is False
    assert result["reason"] == (
        "repair_validation_mutated_disposable_tree"
    )
    assert result["rollback"]["restored"] is True
    assert _git(repo, "rev-parse", "HEAD") == null_merge
    assert _git(repo, "status", "--porcelain=v1") == f"?? {output}"
    assert (repo / output).read_text(encoding="utf-8") == "candidate\n"


def test_null_merge_recovery_exception_restores_executable_untracked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = "repair-tool"
    (
        repo,
        task,
        baseline,
        candidate,
        candidate_tree,
        null_merge,
    ) = _null_merge_scenario(
        tmp_path,
        output=output,
        executable_output=True,
    )
    candidate_bytes = subprocess.run(
        ["git", "show", f"{candidate}:{output}"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    untracked_output = repo / output
    untracked_output.write_bytes(candidate_bytes)
    untracked_output.chmod(0o755)
    daemon = _daemon(repo, task_header_prefix="OUT-")

    def raise_during_validation(workspace, *_args, **_kwargs):
        (workspace / "base.txt").write_text(
            "validation mutation\n",
            encoding="utf-8",
        )
        (workspace / "validation.tmp").write_text(
            "validation byproduct\n",
            encoding="utf-8",
        )
        raise RuntimeError("synthetic validation infrastructure failure")

    monkeypatch.setattr(
        daemon,
        "_run_validation_commands",
        raise_during_validation,
    )
    result = daemon._repair_post_merge_declared_outputs_locked(
        [task],
        primary_task=task,
        attempt=7,
        candidate_commit=candidate,
        candidate_tree=candidate_tree,
        baseline_ref=baseline,
        target_branch="main",
        target_commit=null_merge,
    )

    assert result["passed"] is False
    assert result["reason"] == "repair_internal_error"
    assert result["error_class"] == "RuntimeError"
    assert result["rollback"]["restored"] is True
    assert _git(repo, "rev-parse", "HEAD") == null_merge
    assert _git(repo, "status", "--porcelain=v1") == f"?? {output}"
    assert untracked_output.read_bytes() == candidate_bytes
    assert untracked_output.stat().st_mode & 0o777 == 0o755
    assert (repo / "base.txt").read_text(encoding="utf-8") == "base\n"
    assert not (repo / "validation.tmp").exists()


def test_repair_validation_terminal_only_for_admitted_command_failure() -> None:
    admitted_failure = {
        "validation": [
            {
                "result": {
                    "attempted": True,
                    "results": [
                        {
                            "returncode": 1,
                            "timed_out": False,
                            "infrastructure_failure": False,
                        }
                    ],
                }
            }
        ]
    }
    timed_out = {
        "validation": [
            {
                "result": {
                    "attempted": True,
                    "results": [
                        {
                            "returncode": 124,
                            "timed_out": True,
                            "infrastructure_failure": False,
                        }
                    ],
                }
            }
        ]
    }
    infrastructure_unavailable = {
        "validation": [
            {
                "result": {
                    "attempted": True,
                    "results": [
                        {
                            "returncode": 75,
                            "timed_out": False,
                            "infrastructure_failure": True,
                        }
                    ],
                }
            }
        ]
    }

    classify = (
        TodoImplementationDaemon
        ._post_merge_repair_validation_rejection_admitted
    )
    assert classify(admitted_failure) is True
    assert classify(timed_out) is False
    assert classify(infrastructure_unavailable) is False


def test_historical_completion_requires_bound_immutable_tree(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Tasks

## OUT-008 Preserve exact completion evidence

- Status: todo
- Completion: manual
- Priority: P0
- Track: verification
- Outputs: evidence.txt
- Acceptance: Historical completion is bound to an immutable tree.
""",
        encoding="utf-8",
    )
    (repo / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    _git(repo, "add", "todo.md", "evidence.txt")
    _git(repo, "commit", "-m", "add exact completion evidence")
    integration_commit = _git(repo, "rev-parse", "HEAD")

    daemon = _daemon(repo, task_header_prefix="OUT-")
    task = daemon._load_tasks()[0]
    task_cid = daemon._identity_for_task(task).canonical_task_cid
    event = {
        "type": "implementation_finished",
        "task_id": task.task_id,
        "task_cid": task_cid,
        "implementation_commit": integration_commit,
        "merge_result": {
            "merged": True,
            "merge_commit": integration_commit,
            "completion_task_cids": {task.task_id: task_cid},
        },
    }
    daemon._iter_events = lambda: [event]  # type: ignore[method-assign]

    assert daemon._successfully_merged_task_ids() == {"OUT-008"}

    event["merge_result"].pop("merge_commit")
    assert daemon._successfully_merged_task_ids() == set()


def test_reconciliation_rejects_stale_task_revision_before_merge(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = _daemon(repo, task_header_prefix="OUT-")
    task = _task("OUT-009", "evidence.txt")
    event = {
        "task_id": task.task_id,
        "task_cid": "stale-task-cid",
        "attempt": 1,
        "branch": "implementation/out-009",
        "implementation_commit": "candidate-commit",
    }
    merge_attempts: list[str] = []
    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._load_tasks = lambda: [task]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._merge_branch_to_main = lambda branch, *_args, **_kwargs: merge_attempts.append(branch)  # type: ignore[method-assign]

    result = daemon._reconcile_failed_merges()

    assert merge_attempts == []
    assert result[0]["reason"] == "reconciliation_task_revision_unavailable"
    assert (
        result[0]["completion_binding_error"]["reason"]
        == "completion_task_revision_changed"
    )


def test_immutable_integration_commit_resolves_moving_ref_once(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "candidate.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _git(repo, "commit", "-m", "candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    daemon = _daemon(repo)

    proof = daemon._immutable_integration_commit(
        {"merge_commit": "main"},
        implementation_commit=candidate,
        target_branch="main",
    )

    assert proof["passed"] is True
    assert proof["integration_ref"] == "main"
    assert proof["integration_commit"] == candidate

    (repo / "later.txt").write_text("later\n", encoding="utf-8")
    _git(repo, "add", "later.txt")
    _git(repo, "commit", "-m", "advance moving ref")

    assert _git(repo, "rev-parse", "main") != proof["integration_commit"]
    exact_tree = daemon._declared_output_tracking_invariant(
        [_task("OUT-010", "later.txt")],
        repository_ref=proof["integration_commit"],
    )
    assert exact_tree["passed"] is False
    assert exact_tree["missing_outputs"] == [
        {"task_id": "OUT-010", "path": "later.txt"}
    ]


def test_reconciliation_stays_unresolved_when_completion_revision_changes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = _daemon(repo, task_header_prefix="OUT-")
    task = _task("OUT-011", "evidence.txt")
    task_cid = daemon._identity_for_task(task).canonical_task_cid
    event = {
        "task_id": task.task_id,
        "task_cid": task_cid,
        "attempt": 1,
        "branch": "implementation/out-011",
        "implementation_commit": "candidate-commit",
    }
    observed_completion_cids: dict[str, str] = {}

    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._load_tasks = lambda: [task]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._git_ref_exists = lambda ref: True  # type: ignore[method-assign]
    daemon._merge_branch_to_main = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "merged": True,
        "merge_commit": "integration-commit",
    }
    daemon._immutable_integration_commit = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "passed": True,
        "integration_commit": "integration-commit",
        "reasons": [],
    }
    daemon._declared_output_tracking_invariant = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "passed": True,
        "repository_ref": "integration-commit",
    }
    daemon._cleanup_merged_worktree = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "cleaned": True,
    }

    def reject_revised_completion(
        _task_arg: PortalTask,
        _completion_tasks: list[PortalTask],
        completion_task_cids: dict[str, str],
        *,
        validation_evidence: dict[str, object] | None = None,
    ) -> dict[str, object]:
        assert validation_evidence is None
        observed_completion_cids.update(completion_task_cids)
        return {
            "updated": False,
            "reason": "completion_task_revision_changed",
        }

    daemon._mark_reconciled_completion_in_todo = (  # type: ignore[method-assign]
        reject_revised_completion
    )

    result = daemon._reconcile_failed_merges()

    assert observed_completion_cids == {task.task_id: task_cid}
    assert result[0]["merge_result"]["merged"] is True
    assert result[0]["resolved"] is False
    assert result[0]["reason"] == "completion_persistence_failed"
    assert result[0]["completion_persistence"]["passed"] is False


def test_formal_verification_json_deliverables_are_not_ignored() -> None:
    repo = Path(__file__).resolve().parents[2]
    deliverables = [
        "config/formal_verification_toolchains.lock.json",
        "docs/architecture/formal_verification_live_example_report.json",
        "docs/architecture/formal_verification_readiness_baseline.json",
        "docs/architecture/formal_verification_tactician_benchmark.json",
        "docs/architecture/formal_verification_tactician_readiness_completion_receipt.json",
        "docs/architecture/formal_verification_toolchain_certificate.json",
    ]

    for deliverable in deliverables:
        result = subprocess.run(
            [
                "git",
                "check-ignore",
                "--no-index",
                "--quiet",
                "--",
                deliverable,
            ],
            cwd=repo,
            check=False,
        )
        assert result.returncode == 1, f"{deliverable} is still ignored"


def test_expected_output_soft_skips_optional_declared_outside_predicted_files(
    tmp_path: Path,
) -> None:
    """Broader Outputs beyond Predicted files must not hard-fail preflight.

    CIG re-enable tasks often declare submodule/probe paths as Outputs while
    Predicted files only list the test/Makefile write set. Missing optional
    outputs must soft-skip instead of expected_output_absent_from_proposal.
    """

    repo = _init_repo(tmp_path / "repo")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_suite.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (repo / "Makefile").write_text("test:\n\tpytest\n", encoding="utf-8")
    _git(repo, "add", "tests/test_suite.py", "Makefile")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "tests" / "test_suite.py").write_text(
        "def test_ok():\n    assert True\n# re-enabled\n",
        encoding="utf-8",
    )
    daemon = _daemon(repo)
    task = PortalTask(
        task_id="OUT-010-OPTIONAL",
        title="Re-enable suite",
        status="todo",
        completion="manual",
        priority="P0",
        track="ci",
        outputs=[
            "tests/test_suite.py",
            "Makefile",
            "swissknife/src/services/missing-bridge.ts",
        ],
        validation=["python -m pytest"],
        metadata={
            "predicted files": "tests/test_suite.py",
        },
    )

    preflight = daemon._prepare_proposal_expected_outputs(
        repo,
        task,
        baseline_ref=baseline,
        scope_paths=tuple(task.outputs),
    )
    checks = {item["path"]: item for item in preflight["checks"]}
    assert checks["swissknife/src/services/missing-bridge.ts"]["exists"] is False
    assert checks["swissknife/src/services/missing-bridge.ts"][
        "optional_declared_output"
    ] is True
    assert checks["swissknife/src/services/missing-bridge.ts"][
        "needs_candidate"
    ] is False
    assert checks["swissknife/src/services/missing-bridge.ts"]["issue"] == ""
    assert checks["tests/test_suite.py"]["optional_declared_output"] is False
    assert checks["tests/test_suite.py"]["exists"] is True

    issues = daemon._proposal_expected_output_issues(
        preflight,
        changed_paths=["tests/test_suite.py"],
    )
    assert issues == ()


def test_expected_output_soft_skips_unpopulated_submodule_paths(
    tmp_path: Path,
) -> None:
    """Unpopulated submodule Outputs must soft-skip when not materialized."""

    repo = _init_repo(tmp_path / "repo")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_suite.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (repo / ".gitmodules").write_text(
        '[submodule "swissknife"]\n\tpath = swissknife\n\turl = ./swissknife.git\n',
        encoding="utf-8",
    )
    _git(repo, "add", "tests/test_suite.py", ".gitmodules")
    _git(repo, "commit", "-m", "base with submodule declaration")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "tests" / "test_suite.py").write_text(
        "def test_ok():\n    assert True\n# touch\n",
        encoding="utf-8",
    )
    # swissknife/ is declared but never checked out → unpopulated.
    daemon = _daemon(repo)
    task = _proposal_task(
        "OUT-011-SUBMODULE",
        "tests/test_suite.py",
        "swissknife/src/services/missing-bridge.ts",
    )

    preflight = daemon._prepare_proposal_expected_outputs(
        repo,
        task,
        baseline_ref=baseline,
        scope_paths=tuple(task.outputs),
    )
    checks = {item["path"]: item for item in preflight["checks"]}
    bridge = checks["swissknife/src/services/missing-bridge.ts"]
    assert bridge["exists"] is False
    assert bridge["submodule_root"] == "swissknife"
    assert bridge["submodule_unpopulated"] is True
    assert bridge["needs_candidate"] is False
    assert bridge["issue"] == ""

    issues = daemon._proposal_expected_output_issues(
        preflight,
        changed_paths=["tests/test_suite.py"],
    )
    assert issues == ()


def test_expected_output_still_requires_hard_predicted_missing_output(
    tmp_path: Path,
) -> None:
    """Predicted-file Outputs that are missing still hard-fail preflight."""

    repo = _init_repo(tmp_path / "repo")
    (repo / "present.py").write_text("X = 0\n", encoding="utf-8")
    _git(repo, "add", "present.py")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "present.py").write_text("X = 1\n", encoding="utf-8")
    daemon = _daemon(repo)
    task = PortalTask(
        task_id="OUT-012-HARD",
        title="Must write predicted path",
        status="todo",
        completion="manual",
        priority="P0",
        track="ci",
        outputs=["present.py", "missing_predicted.py"],
        validation=["python -m pytest"],
        metadata={"predicted files": "present.py, missing_predicted.py"},
    )

    preflight = daemon._prepare_proposal_expected_outputs(
        repo,
        task,
        baseline_ref=baseline,
        scope_paths=tuple(task.outputs),
    )
    checks = {item["path"]: item for item in preflight["checks"]}
    assert checks["missing_predicted.py"]["optional_declared_output"] is False
    assert checks["missing_predicted.py"]["issue"] == "expected_output_missing"

    issues = daemon._proposal_expected_output_issues(
        preflight,
        changed_paths=["present.py"],
    )
    assert any(
        issue["path"] == "missing_predicted.py"
        and issue["reason"] == "expected_output_missing"
        for issue in issues
    )


def test_task_declared_output_paths_skips_absolute_host_outputs() -> None:
    task = PortalTask(
        task_id="PGIR-115",
        title="Retry-budget card with host discovery evidence",
        status="todo",
        completion="manual",
        priority="P1",
        track="ops",
        outputs=[
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
            "/home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/note.md",
            "C:/Windows/Temp/host.md",
            "../escape.md",
        ],
    )

    assert task_declared_output_paths(task) == (
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    )
