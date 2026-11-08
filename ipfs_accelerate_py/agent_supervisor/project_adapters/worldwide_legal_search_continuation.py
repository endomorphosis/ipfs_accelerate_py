"""Admit an exact retained candidate through Native's no-provider recovery.

An operator request names bytes, never completion.  The live database owner
must supply its genuine superseded-attempt authority method; Native still
owns proposal admission, validators, the merge train and completion CAS.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

SCHEMA = "wwl.native-candidate-continuation@1"
_FIELDS = frozenset({"schema", "run_id", "task_id", "database_task_cid",
    "predecessor_attempt_id", "predecessor_attempt_number", "predecessor_binding_id",
    "predecessor_projection_immutable_digest", "next_attempt_number", "task_contract_id",
    "repository_root", "worktree_path", "branch_name", "baseline_commit",
    "candidate_commit", "candidate_tree", "changed_files", "receipt_id"})


def identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _need(value: Any, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "--no-optional-locks", "--no-replace-objects", *args],
        cwd=root, text=True, capture_output=True,
        timeout=180 if args[:2] == ("worktree", "add") else 60, check=False)
    _need(result.returncode == 0, "candidate_git_observation_failed")
    return result.stdout.strip()


def task_contract_id(task: Any) -> str:
    """Bind the actual task contract, excluding attempt-local DB metadata."""
    execution_metadata = {"database attempt id", "database claim id", "database attempt number",
        "database lease id", "database owner session id", "database fencing token",
        "database fence epoch", "completion receipt"}
    return identity({"task_id": task.task_id, "title": task.title,
        "outputs": list(task.outputs), "validation": list(task.validation),
        "acceptance": task.acceptance, "board_namespace": task.board_namespace,
        "completion": task.completion, "priority": task.priority, "track": task.track,
        "metadata": {key: value for key, value in task.metadata.items()
            if key.strip().lower().replace("_", " ") not in execution_metadata}})


def candidate_manifest(worktree: Path, baseline: str, candidate: str) -> dict[str, Any]:
    """Read exact committed root regular-file outputs; no staging or mutation."""
    _need(re.fullmatch(r"[0-9a-f]{40}", baseline) and
        re.fullmatch(r"[0-9a-f]{40}", candidate), "candidate_commit_identity_invalid")
    _need(_git(worktree, "rev-parse", "HEAD") == candidate, "candidate_head_changed")
    _need(not _git(worktree, "status", "--porcelain", "--untracked-files=all",
        "--ignore-submodules=none"), "candidate_worktree_not_clean")
    _need(baseline != candidate, "candidate_delta_required")
    _git(worktree, "merge-base", "--is-ancestor", baseline, candidate)
    paths = _git(worktree, "diff", "--name-only", "--no-renames", "-z", baseline,
        candidate, "--").split("\0")
    paths = sorted(path for path in paths if path)
    _need(0 < len(paths) <= 64, "candidate_file_count_outside_bound")
    result: dict[str, Any] = {}
    total = 0
    for relative in paths:
        path = Path(relative)
        _need(not path.is_absolute() and all(part not in ("", ".", "..", ".git")
            for part in path.parts) and "\n" not in relative and "\r" not in relative,
            "candidate_path_unsafe")
        target = worktree / path
        _need(target.resolve(strict=True) == target and target.is_file(),
            "candidate_requires_root_regular_files")
        entry = _git(worktree, "ls-tree", candidate, "--", relative).split("\t")
        _need(len(entry) == 2 and entry[1] == relative, "candidate_tree_entry_invalid")
        mode, kind, _blob = entry[0].split()
        _need(mode in {"100644", "100755"} and kind == "blob", "candidate_mode_unsafe")
        raw = target.read_bytes()
        total += len(raw)
        _need(len(raw) <= 4 * 1024 * 1024 and total <= 16 * 1024 * 1024,
            "candidate_bytes_outside_bound")
        result[relative] = {"mode": mode, "sha256": hashlib.sha256(raw).hexdigest()}
    return result


def build_candidate_request(*, run_id: str, task: Any, predecessor_binding: Mapping[str, Any],
    repository_root: Path, worktree_path: Path, branch_name: str,
    baseline_commit: str, candidate_commit: str, predecessor_attempt_number: int) -> dict[str, Any]:
    """Create a reviewable request from already committed bytes, read-only."""
    repo, worktree = repository_root.resolve(strict=True), worktree_path.resolve(strict=True)
    _need(_git(worktree, "symbolic-ref", "--short", "HEAD") == branch_name,
        "candidate_branch_changed")
    body = {"schema": SCHEMA, "run_id": run_id, "task_id": task.task_id,
        "database_task_cid": predecessor_binding["task_cid"],
        "predecessor_attempt_id": predecessor_binding["attempt_id"],
        "predecessor_attempt_number": predecessor_attempt_number,
        "predecessor_binding_id": predecessor_binding["binding_id"],
        "predecessor_projection_immutable_digest": predecessor_binding["projection_immutable_digest"],
        "next_attempt_number": predecessor_attempt_number + 1, "task_contract_id": task_contract_id(task),
        "repository_root": str(repo), "worktree_path": str(worktree), "branch_name": branch_name,
        "baseline_commit": baseline_commit, "candidate_commit": candidate_commit,
        "candidate_tree": _git(worktree, "rev-parse", candidate_commit + "^{tree}"),
        "changed_files": candidate_manifest(worktree, baseline_commit, candidate_commit)}
    return {**body, "receipt_id": identity(body)}


@dataclass(frozen=True)
class CandidateContinuationAdmission:
    request: Mapping[str, Any]
    predecessor_binding: Mapping[str, Any]
    prior_attempt_authority: Callable[..., Mapping[str, Any]]


def reconcile_candidate(*, portal_daemon: Any, attempt: Any, paths: Any,
    binding: Mapping[str, Any], admission: CandidateContinuationAdmission) -> Mapping[str, Any]:
    """Use genuine native custody and validators; return no completion shortcut."""
    from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon, PortalImplementationDaemon
    from ..todo_daemon.database_portal_bridge import (
        DatabasePortalExecutionBridge, _attempt_key, _accepted_source_events,
    )
    _need(type(admission) is CandidateContinuationAdmission, "native_candidate_admission_required")
    authority = admission.prior_attempt_authority
    _need(type(getattr(authority, "__self__", None)) is DatabaseImplementationDaemon and
        getattr(authority, "__func__", None) is DatabaseImplementationDaemon.authorize_superseded_portal_attempt_binding,
        "genuine_native_attempt_authority_required")
    _need(type(portal_daemon) is PortalImplementationDaemon, "canonical_native_portal_required")
    request, prior = dict(admission.request), dict(admission.predecessor_binding)
    _need(set(request) == _FIELDS and request.get("schema") == SCHEMA and
        request.get("receipt_id") == identity({k: v for k, v in request.items() if k != "receipt_id"}),
        "candidate_request_identity_invalid")
    _need(type(request["predecessor_attempt_number"]) is int and
        type(request["next_attempt_number"]) is int and
        1 <= request["predecessor_attempt_number"] < request["next_attempt_number"] ==
        request["predecessor_attempt_number"] + 1 == attempt.attempt_number,
        "candidate_next_attempt_changed")
    expected = {"task_id": attempt.task_alias, "database_task_cid": attempt.task_cid,
        "predecessor_attempt_id": prior.get("attempt_id"), "predecessor_binding_id": prior.get("binding_id"),
        "predecessor_projection_immutable_digest": prior.get("projection_immutable_digest")}
    _need(all(request[key] == value for key, value in expected.items()) and
        prior.get("task_cid") == attempt.task_cid and prior.get("task_alias") == attempt.task_alias,
        "candidate_predecessor_join_changed")
    _need(re.fullmatch(r"WWL-[0-9]{3}", str(request["task_id"])) and
        type(request["run_id"]) is str and request["run_id"] == authority.__self__.owner_session_id,
        "candidate_campaign_identity_invalid")
    # Native rechecks its actual live DB lease, task/control binding and the
    # predecessor's terminal execution/coordination fences. No request flag
    # may stand in for this capability.
    live_proof = dict(authority(attempt, dict(binding), prior))
    _need(live_proof.get("prior_attempt_number") == request["predecessor_attempt_number"] and
        live_proof.get("prior_execution_status") == "failed", "candidate_predecessor_not_failed")
    repo, worktree = Path(request["repository_root"]), Path(request["worktree_path"])
    # The retained source is read-only and may precede repository-specific
    # routing. Its native owner admits the campaign pool; Git identity below
    # still binds it to this exact repository. Fresh validation checkouts use
    # only Portal's narrower execution pool.
    _need(authority.__self__.execution_state_dir is not None, "candidate_native_execution_state_required")
    source_pool = (Path(authority.__self__.execution_state_dir).resolve(strict=True).parent / "worktrees").resolve(strict=True)
    execution_pool = portal_daemon.worktree_root.resolve(strict=True)
    _need(repo.resolve(strict=True) == repo == portal_daemon.repo_root.resolve(strict=True) and
        worktree.resolve(strict=True) == worktree and worktree != repo and
        execution_pool.is_relative_to(source_pool) and worktree.is_relative_to(source_pool),
        "candidate_workspace_outside_custody")
    _need(_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir") ==
        _git(worktree, "rev-parse", "--path-format=absolute", "--git-common-dir"),
        "candidate_repository_changed")
    _need(paths.task_projection == portal_daemon.todo_path and paths.state == portal_daemon.state_path and
        paths.events == portal_daemon.events_path, "candidate_portal_paths_changed")
    projection = DatabasePortalExecutionBridge._verify_projection(paths, binding)
    from ..todo_daemon.implementation_daemon import parse_task_text
    tasks = parse_task_text(projection, path=paths.task_projection, task_header_prefix=f"## {attempt.task_alias}")
    _need(len(tasks) == 1 and tasks[0].task_id == attempt.task_alias and
        tasks[0].status.lower() not in {"completed", "done", "complete", "cancelled", "failed"},
        "candidate_exact_pending_task_required")
    task = tasks[0]
    _need(task_contract_id(task) == request["task_contract_id"], "candidate_task_contract_changed")
    prior_root = paths.root.parent / _attempt_key(request["predecessor_attempt_id"])
    _need(prior_root.resolve(strict=True) == prior_root, "candidate_predecessor_directory_changed")
    _need(DatabasePortalExecutionBridge._strict_binding(prior_root / "database-attempt-binding.json") == prior,
        "candidate_predecessor_binding_changed")
    prior_paths = type(paths)(root=prior_root, task_projection=prior_root / "task-projection.runtime.todo.md",
        binding=prior_root / "database-attempt-binding.json", state=prior_root / "portal-task-state.json",
        strategy=prior_root / "portal-strategy.json", events=prior_root / "portal-events.jsonl",
        implementation_logs=prior_root / "implementation-logs")
    old_projection = DatabasePortalExecutionBridge._verify_projection(prior_paths, prior)
    old_tasks = parse_task_text(old_projection, path=prior_paths.task_projection, task_header_prefix=f"## {attempt.task_alias}")
    _need(len(old_tasks) == 1 and task_contract_id(old_tasks[0]) == request["task_contract_id"],
        "candidate_original_task_contract_changed")
    bound = getattr(portal_daemon, "_database_attempt_authority", {}) or {}
    _need(bound.get("database_attempt_id") == attempt.attempt_id and
        bound.get("database_binding_id") == binding["binding_id"] and
        bound.get("database_claim_id") == attempt.claim_id, "candidate_portal_attempt_unbound")
    _need(_git(worktree, "symbolic-ref", "--short", "HEAD") == request["branch_name"] and
        _git(worktree, "rev-parse", request["candidate_commit"] + "^{tree}") == request["candidate_tree"] and
        candidate_manifest(worktree, request["baseline_commit"], request["candidate_commit"]) == request["changed_files"],
        "candidate_committed_bytes_changed")
    allowed = set(task.outputs)
    _need(set(request["changed_files"]) <= allowed, "candidate_paths_outside_declared_outputs")
    # Read retained bytes only. A new cleanup-visible native lease owns a
    # fresh checkout at the immutable commit; no old workspace is adopted and
    # absence of historical lifecycle/process records grants no authority.
    fresh_id = hashlib.sha256((attempt.attempt_id + request["receipt_id"]).encode()).hexdigest()[:24]
    fresh = portal_daemon.worktree_root.resolve(strict=True) / ("workspace_continuation_" + fresh_id)
    # Keep the normal campaign branch grammar: real workspace validators and
    # lifecycle cleanup both consume it. The request-bound hash remains unique.
    branch = ("implementation/" + attempt.task_alias.lower() + "-" + fresh_id
              + "-attempt-1-" + str(attempt.started_at_ms // 1000))
    if fresh.exists() or fresh.is_symlink() or _git(repo, "branch", "--list", branch):
        # A checkout can outlive a killed Git command. Never reset, remove or
        # adopt that partial tree. Only Native's exact dead-owner CAS may retire
        # its PREPARING claim, followed by one separate request-bound retry.
        previous = portal_daemon.worktree_lifecycle.load_workspace(fresh)
        _need(previous is not None, "candidate_fresh_workspace_already_exists")
        _need(previous.state.value in {"preparing", "terminal"}
            and previous.task_id == task.task_id and previous.canonical_task_cid == bound["canonical_task_cid"]
            and previous.attempt == portal_daemon._worktree_lifecycle_attempt(task, 1)
            and previous.branch == branch and Path(previous.state_dir) == paths.root.resolve(strict=True)
            and Path(previous.repo_root) == repo, "candidate_interrupted_checkout_custody_unverified")
        if previous.state.value == "preparing":
            previous = portal_daemon.worktree_lifecycle.reclaim_dead_owner_for_controlled_restart(
                fresh, expected_state_dir=paths.root,
                reason="candidate_checkout_interrupted_before_execution")
        _need(previous is not None and previous.is_terminal
            and previous.terminal_reason in {"candidate_checkout_interrupted_before_execution", "controlled_restart_dead_owner"},
            "candidate_interrupted_checkout_owner_not_fenced")
        events = _accepted_source_events(paths.events)[0] if paths.events.exists() else []
        _need(not any(event.get("type") == "candidate_continuation_workspace_prepared"
            and event.get("database_attempt_id") == attempt.attempt_id for event in events),
            "candidate_checkout_already_entered_validation")
        fresh_id = hashlib.sha256((fresh_id + ":checkout-retry-1").encode()).hexdigest()[:24]
        fresh = portal_daemon.worktree_root.resolve(strict=True) / ("workspace_continuation_" + fresh_id)
        branch = ("implementation/" + attempt.task_alias.lower() + "-" + fresh_id
                  + "-attempt-1-" + str(attempt.started_at_ms // 1000))
    _need(not fresh.exists() and not fresh.is_symlink() and not _git(repo, "branch", "--list", branch),
        "candidate_fresh_workspace_already_exists")
    record = portal_daemon.worktree_lifecycle.begin_preparing(task_id=task.task_id,
        canonical_task_cid=bound["canonical_task_cid"], attempt=portal_daemon._worktree_lifecycle_attempt(task, 1),
        lane_id=portal_daemon._worktree_lifecycle_lane_id(), workspace_path=fresh, branch=branch,
        merge_target=portal_daemon.resolved_merge_target_branch, state_dir=str(paths.root.resolve(strict=True)),
        allow_replace_stale=False)
    portal_daemon._active_worktree_lifecycle = record
    authority(attempt, dict(binding), prior)
    lease, _reason, _cleared, _waited = portal_daemon._acquire_checkout_mutation_lease(
        task_id=task.task_id, attempt=1, branch=branch, operation="candidate_continuation_checkout",
        timeout_seconds=0, preserve_existing=True)
    _need(lease is not None, "candidate_native_checkout_lease_unavailable")
    try:
        _git(repo, "worktree", "add", "-b", branch, str(fresh), request["candidate_commit"])
    finally:
        _need(portal_daemon._release_checkout_mutation_lease(lease), "candidate_native_checkout_lease_changed")
    record = portal_daemon.worktree_lifecycle.mark_active(fresh, lease_id=record.lease_id, expected_fence=record.fence)
    portal_daemon._active_worktree_lifecycle = record
    _need(candidate_manifest(fresh, request["baseline_commit"], request["candidate_commit"]) == request["changed_files"] and
        candidate_manifest(worktree, request["baseline_commit"], request["candidate_commit"]) == request["changed_files"],
        "candidate_snapshot_bytes_changed")
    authority(attempt, dict(binding), prior)
    portal_daemon._record_event("candidate_continuation_workspace_prepared", {
        "task_id": task.task_id, "database_attempt_id": attempt.attempt_id,
        "candidate_admission_receipt_id": request["receipt_id"], "source_worktree": str(worktree),
        "worktree_path": str(fresh), "branch": branch, "candidate_commit": request["candidate_commit"],
        "candidate_tree": request["candidate_tree"], "lifecycle_record_id": record.record_id,
        "provider_dispatched": False, "attempt_consumed": False})
    result = portal_daemon.reconcile_validated_worktree_candidate(worktree_path=fresh,
        branch_name=branch, task=task, baseline_ref=request["baseline_commit"],
        candidate_commit=request["candidate_commit"], changed_submodule_paths=(), recovery_key=request["receipt_id"])
    if not result.get("merge_result", {}).get("merged") and not result.get("merge_result", {}).get("queued"):
        current_record = portal_daemon.worktree_lifecycle.load_workspace(fresh)
        if current_record is not None and current_record.is_nonterminal:
            portal_daemon._finalize_exact_worktree_lifecycle(current_record,
                reason="candidate_continuation_validation_failed")
    # Reconciliation has no provider phase. Only a genuinely terminal private
    # projection permits the normal receipt-repair pass to run afterwards.
    projection = DatabasePortalExecutionBridge._verify_projection(paths, binding)
    finished = parse_task_text(projection, path=paths.task_projection, task_header_prefix=f"## {attempt.task_alias}")
    completion_pass: Mapping[str, Any] = {}
    if len(finished) == 1 and finished[0].status.lower() in {"completed", "complete", "done"}:
        _need(result.get("returncode") == 0 and result.get("merge_result", {}).get("merged") is True and
            result.get("provider_dispatched") is False and result.get("attempt_consumed") is False,
            "candidate_native_merge_proof_missing")
        completion_pass = portal_daemon.run_once()
        _need(DatabasePortalExecutionBridge._has_completion_event(paths, attempt.task_alias),
            "candidate_native_completion_event_missing")
    authority(attempt, dict(binding), prior)
    return {"implementation_result": dict(result), "completion_pass": dict(completion_pass),
        "candidate_admission_receipt_id": request["receipt_id"], "provider_dispatched": False,
        "continuation_workspace": str(fresh), "continuation_branch": branch}
