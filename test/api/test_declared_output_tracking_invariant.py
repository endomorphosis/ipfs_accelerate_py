from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Mapping

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
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
    observed_expected_target: list[str] = []

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
    daemon._run_reconciled_post_merge_completion_gate = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: {
            "validation": {
                "passed": True,
                "target_commit": "integration-commit",
                "repository_tree_id": "git-tree:integration-tree",
                "evidence": {},
                "reasons": [],
            },
            "cleanup_result": {"cleaned": True},
            "cleanup_cleaned": True,
        }
    )

    def reject_revised_completion(
        _task_arg: PortalTask,
        _completion_tasks: list[PortalTask],
        completion_task_cids: dict[str, str],
        *,
        expected_target_commit: str,
        validation_evidence: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        del validation_evidence
        observed_completion_cids.update(completion_task_cids)
        observed_expected_target.append(expected_target_commit)
        return {
            "updated": False,
            "reason": "completion_task_revision_changed",
        }

    daemon._mark_reconciled_completion_in_todo = (  # type: ignore[method-assign]
        reject_revised_completion
    )

    result = daemon._reconcile_failed_merges()

    assert observed_completion_cids == {task.task_id: task_cid}
    assert observed_expected_target == ["integration-commit"]
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
