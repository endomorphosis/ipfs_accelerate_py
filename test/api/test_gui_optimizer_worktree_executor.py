"""VGO-050: isolated Git worktree executor tests.

Acceptance coverage:

* temporary-repository fixtures apply only the admitted patch
* rejected and interrupted proposals leave the canonical branch unchanged
* lease/fence failures create no checkout
* undeclared files discovered after apply fail closed and are cleaned up
* browser paths, broad roots, destructive reset, and command strings reject
* host git argv is fixed; the resulting diff matches the admitted scope
* nothing is promoted without a later acceptance step
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.patch_scope import (
    GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
    GUI_IMPROVEMENT_PROPOSAL_SCHEMA,
    GUI_PATCH_SCOPE_GATE_INTERFACE,
    PatchScopeReasonCode,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.worktree_executor import (
    ALLOWED_GIT_VERBS,
    ApplicationDisposition,
    BROAD_ROOTS,
    CleanupState,
    FORBIDDEN_GIT_VERBS,
    GUI_ISOLATED_WORKTREE_EXECUTOR_INTERFACE,
    GUI_PATCH_APPLICATION_RECEIPT_INTERFACE,
    GuiIsolatedWorktreeExecutor,
    GuiPatchApplicationReceipt,
    GuiWorktreeApplyRequest,
    GuiWorktreeExecutorError,
    HOST_GIT_EXECUTABLE,
    HostGitRunner,
    ISOLATED_BRANCH_PREFIX,
    WorktreeExecutorReasonCode,
    default_isolated_worktree_executor,
    sealed_git_environment,
)

IN_SCOPE = "swissknife/web/js/apps/agent-supervisor.js"
OTHER_APP = "swissknife/web/js/apps/legal-assistant.js"
ORIGINAL = "export const label = 'old';\n"
UPDATED = "export const label = 'accessible';\n"


pytestmark = pytest.mark.skipif(
    not (
        Path(HOST_GIT_EXECUTABLE).is_file()
        and os.access(HOST_GIT_EXECUTABLE, os.X_OK)
    ),
    reason="authoritative validation environment requires /usr/bin/git",
)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = sealed_git_environment()
    env.update(
        {
            "GIT_AUTHOR_NAME": "vgo-test",
            "GIT_AUTHOR_EMAIL": "vgo-test@example.invalid",
            "GIT_COMMITTER_NAME": "vgo-test",
            "GIT_COMMITTER_EMAIL": "vgo-test@example.invalid",
        }
    )
    completed = subprocess.run(
        [HOST_GIT_EXECUTABLE, "-c", "core.hooksPath=/dev/null", *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed: {completed.stderr or completed.stdout}"
        )
    return completed


def _init_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    worktrees = tmp_path / "worktrees"
    repo.mkdir()
    worktrees.mkdir()
    target = repo / "swissknife" / "web" / "js" / "apps"
    target.mkdir(parents=True)
    (target / "agent-supervisor.js").write_text(ORIGINAL, encoding="utf-8")
    init = _git(repo, "init", "-b", "main", check=False)
    if init.returncode != 0:
        _git(repo, "init")
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(repo, "add", IN_SCOPE)
    _git(repo, "commit", "-m", "baseline")
    return repo, worktrees


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "--verify", "HEAD").stdout.strip()


def _branch(repo: Path) -> str:
    return _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def _porcelain(repo: Path) -> str:
    return _git(repo, "status", "--porcelain=v1", "-uall").stdout


def _file_text(repo: Path, relative: str) -> str:
    return (repo / relative).read_text(encoding="utf-8")


def _proposal(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "proposal_id": "proposal:label-form",
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
        "objective": "Ensure the goal form has an accessible name.",
        "intended_file_paths": [IN_SCOPE],
        "intended_component_ids": ["comp:goal-form"],
        "acceptance_criteria": ["Goal input has one accessible name."],
        "expected_test_ids": ["test:goal-form-a11y"],
        "expected_screenshot_ids": ["screenshot:keyboard-desktop"],
        "state_effect_ids": ["state:ready"],
        "visual_effect_summary": "Adds the declared visible label.",
        "route_kind": "deterministic_transform",
        "context_pack_id": "pack:label-form",
        "decision": "pending",
        "analysis_classification": "exact",
        "verification_status": "unverified",
        "interface": GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
        "schema_version": GUI_IMPROVEMENT_PROPOSAL_SCHEMA,
    }
    payload.update(overrides)
    return payload


def _observation(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "touched_component_ids": ["comp:goal-form"],
        "touched_state_effect_ids": ["state:ready"],
        "touched_test_ids": [],
        "touched_screenshot_ids": [],
        "application_ids": ["app:agent-supervisor"],
        "action_binding_ids": [],
        "action_contract_evidence": [],
        "visual_effect_observed": True,
        "unresolved_paths": [],
    }
    payload.update(overrides)
    return payload


def _invalidation(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "plan_id": "invalidate:label-form",
        "change_set_id": "changeset:label-form",
        "reasons": ["component_changed"],
        "affected_component_ids": ["comp:goal-form"],
        "affected_scenario_ids": ["scenario:keyboard-only"],
        "affected_check_ids": ["check:direct-tests"],
        "fallback_triggered": False,
        "fallback_explanation": "",
        "interface": "UiInvalidationPlan@1",
        "schema_version": "ui-invalidation-plan/v1",
        "confidence": "exact",
    }
    payload.update(overrides)
    return payload


def _modify_diff(*, new_text: str = UPDATED, path: str = IN_SCOPE) -> str:
    old_count = ORIGINAL.count("\n") or 1
    new_count = new_text.count("\n") or 1
    return (
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -1,{old_count} +1,{new_count} @@\n"
        f"-{ORIGINAL.rstrip(chr(10))}\n"
        f"+{new_text.rstrip(chr(10))}\n"
    )


def _undeclared_diff() -> str:
    return (
        f"--- a/{IN_SCOPE}\n"
        f"+++ b/{IN_SCOPE}\n"
        "@@ -1,1 +1,1 @@\n"
        f"-{ORIGINAL.rstrip(chr(10))}\n"
        f"+{UPDATED.rstrip(chr(10))}\n"
        f"--- /dev/null\n"
        f"+++ b/{OTHER_APP}\n"
        "@@ -0,0 +1,1 @@\n"
        "+export const leaked = true;\n"
    )


def _request(
    repo: Path,
    worktrees: Path,
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "repository_path": str(repo),
        "worktree_parent": str(worktrees),
        "source_revision": _head(repo),
        "proposal": _proposal(),
        "diff_text": _modify_diff(),
        "observation": _observation(),
        "invalidation": _invalidation(),
        "task_id": "VGO-050",
        "attempt": 1,
        "lane_id": "vgo-lane-1",
    }
    payload.update(overrides)
    return payload


def _assert_canonical_untouched(repo: Path, revision: str, branch: str = "main") -> None:
    assert _branch(repo) == branch
    assert _head(repo) == revision
    assert _porcelain(repo) == ""
    assert _file_text(repo, IN_SCOPE) == ORIGINAL


class _FailApplyRunner(HostGitRunner):
    def run(self, argv, *, cwd, input_text=None):  # type: ignore[no-untyped-def]
        if argv and argv[0] == "apply":
            from ipfs_accelerate_py.agent_supervisor.gui_optimizer.worktree_executor import (
                HostGitResult,
            )

            return HostGitResult(
                argv=(HOST_GIT_EXECUTABLE, *tuple(argv)),
                returncode=1,
                stdout="",
                stderr="injected apply failure",
            )
        return super().run(argv, cwd=cwd, input_text=input_text)


# ---------------------------------------------------------------------------
# Package / interface surface
# ---------------------------------------------------------------------------


def test_executor_exports_declared_interfaces() -> None:
    executor = default_isolated_worktree_executor()
    assert executor.interface == GUI_ISOLATED_WORKTREE_EXECUTOR_INTERFACE
    assert GUI_ISOLATED_WORKTREE_EXECUTOR_INTERFACE == "GuiIsolatedWorktreeExecutor@1"
    assert GUI_PATCH_APPLICATION_RECEIPT_INTERFACE == "GuiPatchApplicationReceipt@1"
    assert executor.scope_gate.interface == GUI_PATCH_SCOPE_GATE_INTERFACE
    assert HOST_GIT_EXECUTABLE == "/usr/bin/git"
    assert "reset" in FORBIDDEN_GIT_VERBS
    assert "checkout" in FORBIDDEN_GIT_VERBS
    assert "apply" in ALLOWED_GIT_VERBS
    assert "worktree" in ALLOWED_GIT_VERBS
    assert "/" in BROAD_ROOTS
    assert "/tmp" in BROAD_ROOTS
    assert ISOLATED_BRANCH_PREFIX == "vgo/isolated/"
    env = sealed_git_environment()
    assert env["PATH"] == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert "GIT_DIR" not in env


def test_host_git_runner_rejects_destructive_reset(tmp_path: Path) -> None:
    runner = HostGitRunner()
    cwd = tmp_path
    cwd.mkdir(exist_ok=True)
    with pytest.raises(GuiWorktreeExecutorError) as exc:
        runner.validate_argv(("reset", "--hard", "HEAD"), cwd=cwd)
    assert (
        exc.value.reason_code
        == WorktreeExecutorReasonCode.DESTRUCTIVE_RESET_FORBIDDEN.value
    )
    with pytest.raises(GuiWorktreeExecutorError):
        runner.validate_argv(("checkout", "main"), cwd=cwd)
    with pytest.raises(GuiWorktreeExecutorError):
        runner.validate_argv(("apply", "--unsafe-paths"), cwd=cwd)
    with pytest.raises(GuiWorktreeExecutorError):
        runner.validate_argv(("branch", "-D", "main"), cwd=cwd)


def test_closed_request_rejects_command_and_browser_fields() -> None:
    with pytest.raises(GuiWorktreeExecutorError) as command:
        GuiWorktreeApplyRequest.from_mapping(
            {
                "repository_path": "/tmp/repo",
                "worktree_parent": "/tmp/worktrees",
                "proposal": _proposal(),
                "diff_text": _modify_diff(),
                "command": "git apply --unsafe-paths",
            }
        )
    assert (
        command.value.reason_code
        == WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value
    )
    with pytest.raises(GuiWorktreeExecutorError) as reset:
        GuiWorktreeApplyRequest.from_mapping(
            {
                "repository_path": "/tmp/repo",
                "worktree_parent": "/tmp/worktrees",
                "proposal": _proposal(),
                "diff_text": _modify_diff(),
                "destructive_reset": True,
            }
        )
    assert (
        reset.value.reason_code
        == WorktreeExecutorReasonCode.DESTRUCTIVE_RESET_FORBIDDEN.value
    )
    with pytest.raises(GuiWorktreeExecutorError) as browser:
        GuiWorktreeApplyRequest.from_mapping(
            {
                "repository_path": "/tmp/repo",
                "worktree_parent": "/tmp/worktrees",
                "proposal": _proposal(),
                "diff_text": _modify_diff(),
                "host_path": "C:\\\\Users\\\\browser",
            }
        )
    assert (
        browser.value.reason_code
        == WorktreeExecutorReasonCode.BROWSER_PATH_FORBIDDEN.value
    )


# ---------------------------------------------------------------------------
# Successful isolated apply
# ---------------------------------------------------------------------------


def test_admitted_patch_applies_only_in_isolated_worktree(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    executor = default_isolated_worktree_executor()
    receipt = executor.apply(_request(repo, worktrees))
    assert receipt.disposition is ApplicationDisposition.APPLIED
    assert receipt.applied is True
    assert receipt.promoted is False
    assert receipt.cleanup_state is CleanupState.RETAINED
    assert receipt.source_revision == before
    assert receipt.parent_revision == before
    assert receipt.canonical_branch == "main"
    assert receipt.isolated_branch.startswith(ISOLATED_BRANCH_PREFIX)
    assert receipt.observed_paths == (IN_SCOPE,)
    assert receipt.admitted_paths == (IN_SCOPE,)
    assert IN_SCOPE in receipt.observed_diff
    assert receipt.patch_digest.startswith("sha256:")
    assert WorktreeExecutorReasonCode.NOT_PROMOTED.value in receipt.reason_codes
    worktree = Path(receipt.worktree_path)
    assert worktree.is_dir()
    assert worktree.parent.resolve() == worktrees.resolve()
    assert _file_text(worktree, IN_SCOPE) == UPDATED
    _assert_canonical_untouched(repo, before)
    discarded = executor.discard(receipt)
    assert discarded.cleanup_state is CleanupState.REMOVED
    assert discarded.promoted is False
    assert not worktree.exists()
    _assert_canonical_untouched(repo, before)
    branches = _git(repo, "branch").stdout
    assert receipt.isolated_branch not in branches


def test_apply_request_records_exact_source_revision(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    receipt = default_isolated_worktree_executor().apply_request(
        _request(repo, worktrees, source_revision=before)
    )
    assert receipt.applied is True
    assert receipt.source_revision == before
    payload = receipt.to_dict()
    assert payload["promoted"] is False
    assert payload["interface"] == GUI_PATCH_APPLICATION_RECEIPT_INTERFACE
    assert payload["source_revision"] == before


# ---------------------------------------------------------------------------
# Rejected / interrupted proposals cannot mutate the canonical branch
# ---------------------------------------------------------------------------


def test_rejected_undeclared_patch_leaves_canonical_untouched(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    receipt = default_isolated_worktree_executor().apply(
        _request(
            repo,
            worktrees,
            diff_text=_undeclared_diff(),
            proposal=_proposal(intended_file_paths=[IN_SCOPE]),
        )
    )
    assert receipt.disposition is ApplicationDisposition.REJECTED
    assert receipt.applied is False
    assert receipt.promoted is False
    assert receipt.cleanup_state is CleanupState.NEVER_CREATED
    assert receipt.worktree_path == ""
    assert WorktreeExecutorReasonCode.SCOPE_REJECTED.value in receipt.reason_codes
    assert PatchScopeReasonCode.UNDECLARED_FILE.value in receipt.reason_codes
    assert list(worktrees.iterdir()) == []
    _assert_canonical_untouched(repo, before)


def test_review_gated_proposal_does_not_create_worktree(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    credential_diff = (
        f"--- a/{IN_SCOPE}\n"
        f"+++ b/{IN_SCOPE}\n"
        "@@ -1,1 +1,2 @@\n"
        f" {ORIGINAL.rstrip(chr(10))}\n"
        "+const config = { password: 'secret', authorization: 'bearer x' };\n"
    )
    receipt = default_isolated_worktree_executor().apply(
        _request(repo, worktrees, diff_text=credential_diff)
    )
    assert receipt.applied is False
    assert receipt.cleanup_state is CleanupState.NEVER_CREATED
    assert (
        WorktreeExecutorReasonCode.SCOPE_REQUIRES_REVIEW.value in receipt.reason_codes
        or WorktreeExecutorReasonCode.SCOPE_REJECTED.value in receipt.reason_codes
    )
    _assert_canonical_untouched(repo, before)


def test_interrupted_apply_removes_worktree_and_preserves_branch(
    tmp_path: Path,
) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    executor = GuiIsolatedWorktreeExecutor(git_runner=_FailApplyRunner())
    receipt = executor.apply(_request(repo, worktrees))
    assert receipt.disposition is ApplicationDisposition.INTERRUPTED
    assert receipt.applied is False
    assert receipt.promoted is False
    assert receipt.cleanup_state is CleanupState.REMOVED
    assert WorktreeExecutorReasonCode.INTERRUPTED.value in receipt.reason_codes
    assert list(worktrees.iterdir()) == []
    assert receipt.isolated_branch not in _git(repo, "branch").stdout
    _assert_canonical_untouched(repo, before)


def test_source_revision_mismatch_does_not_create_worktree(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    fake = "ab" * 20
    receipt = default_isolated_worktree_executor().apply(
        _request(repo, worktrees, source_revision=fake)
    )
    assert receipt.applied is False
    assert (
        WorktreeExecutorReasonCode.SOURCE_REVISION_MISMATCH.value
        in receipt.reason_codes
    )
    assert receipt.cleanup_state is CleanupState.NEVER_CREATED
    _assert_canonical_untouched(repo, before)


# ---------------------------------------------------------------------------
# Lease / fence failures
# ---------------------------------------------------------------------------


def test_duplicate_task_attempt_is_fenced_without_second_worktree(
    tmp_path: Path,
) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    executor = default_isolated_worktree_executor()
    first = executor.apply(_request(repo, worktrees, task_id="VGO-050", attempt=1))
    assert first.applied is True
    second = executor.apply(_request(repo, worktrees, task_id="VGO-050", attempt=1))
    assert second.disposition is ApplicationDisposition.FENCED
    assert second.applied is False
    assert (
        WorktreeExecutorReasonCode.LEASE_FENCE_FAILURE.value in second.reason_codes
    )
    assert second.cleanup_state is CleanupState.NEVER_CREATED
    remaining = [path for path in worktrees.iterdir() if path.is_dir()]
    assert remaining == [Path(first.worktree_path)]
    _assert_canonical_untouched(repo, before)
    executor.discard(first)
    _assert_canonical_untouched(repo, before)


def test_discard_with_foreign_lease_fails_closed(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    executor = default_isolated_worktree_executor()
    receipt = executor.apply(_request(repo, worktrees))
    assert receipt.applied is True
    forged = GuiPatchApplicationReceipt(
        disposition=receipt.disposition,
        reason_codes=receipt.reason_codes,
        applied=True,
        repository_path=receipt.repository_path,
        worktree_path=receipt.worktree_path,
        worktree_parent=receipt.worktree_parent,
        isolated_branch=receipt.isolated_branch,
        canonical_branch=receipt.canonical_branch,
        source_revision=receipt.source_revision,
        parent_revision=receipt.parent_revision,
        observed_diff=receipt.observed_diff,
        observed_paths=receipt.observed_paths,
        admitted_paths=receipt.admitted_paths,
        cleanup_state=receipt.cleanup_state,
        lease_id="not-the-owner-lease",
        fence=receipt.fence,
        lifecycle_state=receipt.lifecycle_state,
        proposal_id=receipt.proposal_id,
        patch_digest=receipt.patch_digest,
        observed_diff_digest=receipt.observed_diff_digest,
        scope_decision=dict(receipt.scope_decision),
        message=receipt.message,
        details=dict(receipt.details),
    )
    blocked = executor.discard(forged)
    assert (
        WorktreeExecutorReasonCode.LEASE_FENCE_FAILURE.value in blocked.reason_codes
    )
    assert Path(receipt.worktree_path).is_dir()
    stale_fence = GuiPatchApplicationReceipt(
        disposition=receipt.disposition,
        reason_codes=receipt.reason_codes,
        applied=True,
        repository_path=receipt.repository_path,
        worktree_path=receipt.worktree_path,
        worktree_parent=receipt.worktree_parent,
        isolated_branch=receipt.isolated_branch,
        canonical_branch=receipt.canonical_branch,
        source_revision=receipt.source_revision,
        parent_revision=receipt.parent_revision,
        observed_diff=receipt.observed_diff,
        observed_paths=receipt.observed_paths,
        admitted_paths=receipt.admitted_paths,
        cleanup_state=receipt.cleanup_state,
        lease_id=receipt.lease_id,
        fence=max(1, receipt.fence - 1),
        lifecycle_state=receipt.lifecycle_state,
        proposal_id=receipt.proposal_id,
        patch_digest=receipt.patch_digest,
        observed_diff_digest=receipt.observed_diff_digest,
        scope_decision=dict(receipt.scope_decision),
        message=receipt.message,
        details=dict(receipt.details),
    )
    fenced = executor.discard(stale_fence)
    assert (
        WorktreeExecutorReasonCode.LEASE_FENCE_FAILURE.value in fenced.reason_codes
    )
    assert Path(receipt.worktree_path).is_dir()
    _assert_canonical_untouched(repo, before)
    executor.discard(receipt)


# ---------------------------------------------------------------------------
# Post-apply undeclared-file recheck
# ---------------------------------------------------------------------------


def test_undeclared_file_post_apply_recheck_cleans_worktree(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    executor = default_isolated_worktree_executor()
    receipt = executor.apply(_request(repo, worktrees))
    assert receipt.applied is True
    leaked = Path(receipt.worktree_path) / OTHER_APP
    leaked.parent.mkdir(parents=True, exist_ok=True)
    leaked.write_text("export const leaked = true;\n", encoding="utf-8")
    rechecked = executor.recheck(receipt)
    assert rechecked.disposition is ApplicationDisposition.REJECTED
    assert rechecked.applied is False
    assert rechecked.promoted is False
    assert rechecked.cleanup_state is CleanupState.REMOVED
    assert (
        WorktreeExecutorReasonCode.UNDECLARED_FILE_POST_APPLY.value
        in rechecked.reason_codes
    )
    assert OTHER_APP in rechecked.observed_paths
    assert not Path(receipt.worktree_path).exists()
    _assert_canonical_untouched(repo, before)


# ---------------------------------------------------------------------------
# Host path / command fences
# ---------------------------------------------------------------------------


def test_browser_uri_repository_is_rejected(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    receipt = default_isolated_worktree_executor().apply(
        _request(repo, worktrees, repository_path=f"file://{repo}")
    )
    assert receipt.applied is False
    assert (
        WorktreeExecutorReasonCode.BROWSER_PATH_FORBIDDEN.value
        in receipt.reason_codes
    )
    _assert_canonical_untouched(repo, before)


def test_broad_root_worktree_parent_is_rejected(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    receipt = default_isolated_worktree_executor().apply(
        _request(repo, worktrees, worktree_parent="/")
    )
    assert receipt.applied is False
    assert (
        WorktreeExecutorReasonCode.BROAD_ROOT_FORBIDDEN.value in receipt.reason_codes
    )
    _assert_canonical_untouched(repo, before)


def test_tmp_root_is_rejected_as_broad_parent(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    receipt = default_isolated_worktree_executor().apply(
        _request(repo, worktrees, worktree_parent="/tmp")
    )
    assert receipt.applied is False
    assert (
        WorktreeExecutorReasonCode.BROAD_ROOT_FORBIDDEN.value in receipt.reason_codes
    )
    _assert_canonical_untouched(repo, before)


def test_command_like_repository_path_is_rejected(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    receipt = default_isolated_worktree_executor().apply(
        _request(repo, worktrees, repository_path="git apply; rm -rf /")
    )
    assert receipt.applied is False
    assert (
        WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value
        in receipt.reason_codes
    )
    _assert_canonical_untouched(repo, before)


def test_worktree_parent_inside_repo_is_rejected(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    nested = repo / "nested-worktrees"
    nested.mkdir()
    receipt = default_isolated_worktree_executor().apply(
        _request(repo, worktrees, worktree_parent=str(nested))
    )
    assert receipt.applied is False
    assert (
        WorktreeExecutorReasonCode.WORKTREE_PARENT_INVALID.value
        in receipt.reason_codes
    )
    _assert_canonical_untouched(repo, before)


def test_unknown_request_field_rejects_before_git(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    payload = _request(repo, worktrees)
    payload["shell"] = "bash -lc 'git reset --hard'"
    receipt = default_isolated_worktree_executor().apply(payload)
    assert receipt.applied is False
    assert (
        WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value
        in receipt.reason_codes
    )
    _assert_canonical_untouched(repo, before)


def test_malformed_diff_never_creates_worktree(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    receipt = default_isolated_worktree_executor().apply(
        _request(repo, worktrees, diff_text="not a unified diff")
    )
    assert receipt.applied is False
    assert receipt.cleanup_state is CleanupState.NEVER_CREATED
    _assert_canonical_untouched(repo, before)


def test_executor_rejects_non_gate_scope_injection() -> None:
    with pytest.raises(GuiWorktreeExecutorError):
        GuiIsolatedWorktreeExecutor(scope_gate=object())  # type: ignore[arg-type]
    with pytest.raises(GuiWorktreeExecutorError):
        HostGitRunner(executable="/usr/local/bin/git")


def test_receipt_cannot_claim_promotion() -> None:
    receipt = GuiPatchApplicationReceipt(
        disposition=ApplicationDisposition.APPLIED,
        reason_codes=(WorktreeExecutorReasonCode.APPLIED.value,),
        applied=True,
        promoted=True,
    )
    assert receipt.promoted is False
    assert receipt.to_dict()["promoted"] is False


def test_resulting_diff_matches_admitted_scope_only(tmp_path: Path) -> None:
    repo, worktrees = _init_repo(tmp_path)
    executor = default_isolated_worktree_executor()
    receipt = executor.apply(_request(repo, worktrees))
    assert receipt.applied is True
    assert tuple(receipt.observed_paths) == tuple(receipt.admitted_paths)
    assert receipt.admitted_paths == (IN_SCOPE,)
    # The isolated file changed; the canonical file did not.
    assert _file_text(Path(receipt.worktree_path), IN_SCOPE) == UPDATED
    assert _file_text(Path(receipt.repository_path), IN_SCOPE) == ORIGINAL
    executor.discard(receipt)


def test_git_executable_is_not_world_writable() -> None:
    mode = Path(HOST_GIT_EXECUTABLE).stat().st_mode
    assert not mode & stat.S_IWOTH
    assert os.access(HOST_GIT_EXECUTABLE, os.X_OK)
