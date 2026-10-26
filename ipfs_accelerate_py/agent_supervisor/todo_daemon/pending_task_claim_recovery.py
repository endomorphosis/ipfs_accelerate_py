"""Explicit custody release for an unchanged, stopped ordinary pending attempt.

This API abandons a dead dispatch lease. It does not settle the provider,
complete a task, change consumed ordinals, or authorize another dispatch.
The caller must retain its stopped-launcher, checkout and queue guards for the
entire review/reconcile operation; those application locks are not SDK-owned.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
import importlib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess

from .current_incident_adjudication import (
    _FALSE, _baseline_unchanged, _bytes, _event_rows, _fresh_container_quiescence, _need,
    _object, _ordinary_implementation_lease_proof, _read, _sha, _state_hash,
)

SCHEMA = "ordinary-pending-task-claim-recovery@1"
DIRECTORY = "pending-task-claim-recoveries"
_OUTCOMES = {"implementation_finished", "implementation_terminated", "implementation_timeout"}
CANDIDATE_SCHEMA = "ordinary-preserved-pending-candidate-recovery@1"


def runtime_sources() -> dict:
    """Pin actual imported Native support, including cross-version recovery use.

    A reviewed standalone helper can be loaded beside an older running SDK.
    Its neighboring files are not evidence of the Native code actually used.
    """
    names = (".current_incident_adjudication", ".implementation_daemon",
             ".portal_task_state_control_plane", "..merge.worktree_lifecycle", "..merge.checkout_lock",
             "..merge.merge_queue", "..merge.workspace_quarantine", "..runtime.grok_cli_runner",
             "..proof.code_proof_obligations", "..validation.proposal_validation")
    modules = {name: importlib.import_module(name, __package__) for name in names}
    paths = {name: Path(module.__file__).resolve() for name, module in modules.items()}
    paths["pending_task_claim_recovery"] = Path(__file__).resolve()
    return {name: {"path": str(path), "sha256": _sha(_read(path))} for name, path in paths.items()}


def runtime_revision() -> str:
    return "sha256:" + _sha(_bytes(runtime_sources()))


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(["git", "--no-replace-objects", *args], cwd=workspace,
                            capture_output=True, text=True, timeout=15, check=False)
    _need(result.returncode == 0, "pending_claim_git_read_failed")
    return result.stdout.strip()


def _pin(path: Path) -> dict:
    raw = _read(path)
    info = path.lstat()
    return {"path": str(path), "bytes": len(raw), "sha256": _sha(raw),
            "identity": {key: getattr(info, key) for key in
                ("st_dev", "st_ino", "st_mode", "st_uid", "st_nlink", "st_mtime_ns", "st_ctime_ns")}}


def _queue(daemon) -> dict:
    from ..merge.merge_queue import MergeQueue, _read_merge_queue_settlement_under_lock
    _need(type(daemon.merge_queue) is MergeQueue, "ordinary_merge_inventory_required")
    # Application recovery retains the real writer guard. The usual instance
    # inventory opens a writer and would reacquire that same flock. Native's
    # operator reader opens only the existing store in a readonly transaction.
    receipt = _read_merge_queue_settlement_under_lock(daemon.merge_queue.queue_dir,
        target_repository_id=daemon.merge_target_repository_id,
        target_branch=daemon.resolved_merge_target_branch)
    _need(receipt.get("settled") is True and receipt.get("status_counts", {}).get("quarantined") == 0,
          "independent_merge_custody")
    return receipt


def _ordinary(daemon):
    from . import implementation_daemon as native
    _need(type(daemon) is native.PortalImplementationDaemon, "real_native_portal_required")
    _need(daemon.implement is False and daemon.task_source is None
          and daemon.production_provider_policy is None
          and daemon.production_owner_handoff_factory is None
          and getattr(daemon, "_database_attempt_authority", None) is None
          and not daemon.manual_completion_authority_task_ids
          and not daemon.manual_completion_authority_required_task_ids
          and not daemon.manual_completion_authority_revalidation_only,
          "nonimplementing_ordinary_portal_required")
    _need(not native.automatic_retry_rearm_enabled(), "strict_attempt_policy_required")
    _need(daemon.state_path.is_absolute() and daemon.state_path.resolve() == daemon.state_path,
          "aliased_state_path")
    info = daemon.state_path.parent.stat()
    _need(info.st_uid == os.getuid() and not info.st_mode & 0o077,
          "private_state_directory_required")
    return native


def _terminal_maintenance_owner(daemon, native, owner) -> dict:
    """Allow a dead maintenance birth or this exact recovery process only.

    A terminal record owns no running implementation. A controlled restart
    may have just written it in this same public recovery call; that caller
    need not exit before retiring the original, separately proven-dead claim.
    PID reuse is not accepted as provenance for a maintenance record.
    """
    from ..merge.worktree_lifecycle import read_process_birth
    _need(owner.pid > 0 and owner.start_time_ticks > 0 and bool(owner.boot_id),
          "terminal_maintenance_birth_invalid")
    proc = daemon.worktree_lifecycle.proc_root
    observed = read_process_birth(owner.pid, proc_root=proc)
    liveness = native.owner_liveness(owner, proc_root=proc)
    current = native.current_process_birth(proc_root=proc)
    if observed is None:
        _need(liveness is native.OwnerLiveness.DEAD, "terminal_maintenance_owner_unknown")
        source = "dead_controlled_restart_maintenance_owner"
    else:
        _need(liveness is native.OwnerLiveness.ALIVE and owner == observed == current
              and current.pid == os.getpid(), "terminal_maintenance_owner_foreign_or_reused")
        source = "exact_current_recovery_process"
    return {"source": source, "owner": owner.to_dict()}


def _build(daemon, task_id: str) -> dict:
    native = _ordinary(daemon)
    tasks = [task for task in daemon._load_tasks() if task.task_id == task_id]
    _need(len(tasks) == 1, "current_task_not_unique")
    task = tasks[0]
    identity = daemon._identity_for_task(task)
    state = native.PortalTaskState.load(daemon.state_path)
    attempt = state.implementation_attempts.get(task_id)
    _need(type(attempt) is int and attempt > 0
          and state.implementation_attempts_by_cid.get(identity.canonical_task_cid) == attempt,
          "consumed_ordinal_mismatch")
    events = _event_rows(_read(daemon.events_path))
    starts = [row for row in events if row.get("type") == "implementation_started"
              and row.get("canonical_task_cid") == identity.canonical_task_cid]
    _need(starts and starts[-1].get("attempt") == attempt, "latest_exact_start_required")
    start = starts[-1]
    workspace = Path(start["worktree_path"])
    life = daemon.worktree_lifecycle.load_workspace(workspace)
    _need(life is not None, "terminal_lifecycle_required")
    directory = daemon.state_path.parent
    paths = {"claim": daemon._implementation_task_claim_path(task_id, canonical_task_cid=identity.canonical_task_cid),
             "lifecycle": daemon.worktree_lifecycle.workspace_path_for(workspace),
             "lifecycle_index": daemon.worktree_lifecycle.task_index_path_for(
                 canonical_task_cid=identity.canonical_task_cid, task_id=task_id, attempt=life.attempt),
             "strategy": daemon.strategy_path, "launcher": directory / "launcher.json",
             "shutdown": directory / "container_shutdown.json", "board": daemon.todo_path,
             "events": daemon.events_path}
    return {"schema": SCHEMA, "runtime_revision": runtime_revision(), "runtime_sources": runtime_sources(),
            "repo_root": str(daemon.repo_root.resolve()), "repo_head": _git(daemon.repo_root, "rev-parse", "HEAD"),
            "state_path": str(daemon.state_path), "state_sha256": _state_hash(daemon),
            "task_id": task_id, "task_cid": identity.canonical_task_cid,
            "task_key": identity.canonical_task_key, "board_namespace": identity.board_namespace,
            "task_header_prefix": daemon.task_header_prefix, "attempt": attempt,
            "workspace": str(workspace), "baseline_ref": start["baseline_ref"],
            "start_event_id": start["event_id"],
            "merge_queue": _queue(daemon),
            "implementation_lease_sha256": (_sha(_read(daemon._implementation_lock_path()))
                if os.path.lexists(daemon._implementation_lock_path()) else ""),
            "pins": {name: _pin(path) for name, path in paths.items()}}


def _validated(daemon, request: dict, prior: dict | None, *, candidate: dict | None = None) -> dict:
    native = _ordinary(daemon)
    keys = {"schema", "runtime_revision", "runtime_sources", "repo_root", "repo_head", "state_path", "state_sha256",
            "task_id", "task_cid", "task_key", "board_namespace", "task_header_prefix", "attempt",
            "workspace", "baseline_ref", "start_event_id", "implementation_lease_sha256", "merge_queue", "pins"}
    _need(set(request) == keys and request["schema"] == SCHEMA, "pending_claim_request_shape_changed")
    _need(request["runtime_revision"] == runtime_revision() and request["runtime_sources"] == runtime_sources(),
          "pending_claim_runtime_changed")
    _need(request["repo_root"] == str(daemon.repo_root.resolve())
          and request["state_path"] == str(daemon.state_path)
          and request["task_header_prefix"] == daemon.task_header_prefix
          and _git(daemon.repo_root, "rev-parse", "HEAD") == request["repo_head"], "pending_claim_source_changed")
    _need(_state_hash(daemon) == request["state_sha256"], "pending_claim_state_changed")
    _need(_queue(daemon) == request["merge_queue"], "pending_claim_merge_queue_changed")
    state = native.PortalTaskState.load(daemon.state_path)
    active = ("implementation_in_progress", "active_task_id", "active_task_cid", "active_attempt",
              "active_task_key", "active_phase", "active_phase_started_at", "active_worktree_path",
              "active_branch", "active_log_path", "active_task_started_at", "active_provider_runner")
    _need(not any(getattr(state, key, None) for key in active), "pending_claim_lane_not_quiescent")
    task_id, cid, attempt = request["task_id"], request["task_cid"], request["attempt"]
    _need(type(attempt) is int and attempt > 0
          and state.implementation_attempts.get(task_id) == state.implementation_attempts_by_cid.get(cid) == attempt,
          "pending_claim_ordinal_changed")
    _need(state.last_implementation_task_id == task_id and state.last_implementation_task_cid == cid,
          "pending_claim_last_attempt_changed")
    tasks = [task for task in daemon._load_tasks() if task.task_id == task_id and daemon._canonical_ref(task) == cid]
    _need(len(tasks) == 1 and native.normalize_status(tasks[0].status) in {"todo", "in_progress", "ready"},
          "pending_current_task_required")
    task = tasks[0]; identity = daemon._identity_for_task(task)
    _need(identity.canonical_task_key == request["task_key"] and identity.board_namespace == request["board_namespace"],
          "pending_claim_task_identity_changed")
    refs = {task_id, cid, identity.canonical_task_key}
    _need(not any(str(v).strip().lower() in {"true", "yes", "1"} for k, v in task.metadata.items()
          if str(k).lower().replace("_", " ").strip() in {"hold", "manual hold"}), "manual_hold_present")
    for key in ("waiting_task_ids", "blocked_task_ids", "external_reserved_task_ids", "completed_task_ids"):
        values = getattr(state, key)
        _need(type(values) is list and all(type(v) is str for v in values) and not refs.intersection(values),
              "independent_native_hold")
    _need(all(key not in state.task_statuses or native.normalize_status(state.task_statuses[key])
          in {"todo", "in_progress", "ready"} for key in refs), "independent_status_hold")
    _need(type(state.protected_implementation_attempts) is dict and not any(
          key in refs or type(value) is not dict or value.get("task_id") in refs
          or value.get("task_revision_cid") in refs for key, value in state.protected_implementation_attempts.items()),
          "protected_attempt_hold")
    _need(daemon.max_task_attempts <= 0 or attempt < daemon.max_task_attempts, "retry_budget_exhausted")
    _need(daemon._retained_provider_capacity_deferral() is None, "capacity_hold_present")
    for path in (daemon._implementation_protected_active_snapshot_path(), daemon._implementation_protected_incident_path(),
                 daemon._protected_path_maintenance_lock_path()):
        _need(not os.path.lexists(path), "protected_hold_present")
    pins = request["pins"]
    _need(type(pins) is dict and set(pins) == {"claim", "lifecycle", "lifecycle_index", "strategy", "launcher", "shutdown", "board", "events"},
          "pending_claim_pin_set_changed")
    values = {}
    for name, pin in pins.items():
        _need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256", "identity"}, "pending_claim_pin_shape_changed")
        path = Path(pin["path"])
        if name == "claim" and not os.path.lexists(path):
            _need(prior is not None and prior.get("phase") in {"prepared", "released"}, "claim_missing_without_prepared_receipt")
            raw = base64.b64decode(prior["claim_raw_base64"], validate=True)
        else:
            raw = _read(path)
            info = path.lstat()
            expected_identity = pin["identity"]
            _need(type(expected_identity) is dict and set(expected_identity) == {
                "st_dev", "st_ino", "st_mode", "st_uid", "st_nlink", "st_mtime_ns", "st_ctime_ns"}, "pending_claim_identity_shape_changed")
            fields = set(expected_identity) - ({"st_mtime_ns", "st_ctime_ns"} if name == "events" else set())
            _need(all(getattr(info, key) == expected_identity[key] for key in fields), "pending_claim_pinned_" + name + "_identity_changed")
        prefix = raw[:pin["bytes"]] if name == "events" else raw
        _need(len(prefix) == pin["bytes"] and _sha(prefix) == pin["sha256"], "pending_claim_pinned_" + name + "_changed")
        values[name] = _event_rows(raw) if name == "events" else (_object(raw) if name != "board" else raw)
    workspace = Path(request["workspace"])
    _need(workspace.is_absolute() and workspace.resolve() == workspace and workspace != daemon.repo_root.resolve()
          and workspace.is_relative_to(daemon.worktree_root.resolve()), "pending_claim_workspace_scope_changed")
    life = daemon.worktree_lifecycle.load_workspace(workspace)
    _need(life is not None and life.to_dict() == values["lifecycle"] and life.is_terminal
          and life.record_id == life.compute_record_id(), "exact_terminal_lifecycle_required")
    expected = {"claim": daemon._implementation_task_claim_path(task_id, canonical_task_cid=cid),
                "lifecycle": daemon.worktree_lifecycle.workspace_path_for(workspace),
                "lifecycle_index": daemon.worktree_lifecycle.task_index_path_for(canonical_task_cid=cid, task_id=task_id, attempt=life.attempt),
                "strategy": daemon.strategy_path, "board": daemon.todo_path, "events": daemon.events_path,
                "launcher": daemon.state_path.parent / "launcher.json", "shutdown": daemon.state_path.parent / "container_shutdown.json"}
    _need(all(Path(pins[key]["path"]) == path for key, path in expected.items()), "foreign_pending_claim_authority_path")
    _need(values["lifecycle_index"] == daemon.worktree_lifecycle._task_index_payload(life), "lifecycle_task_index_changed")
    _need(life.task_id == task_id and life.canonical_task_cid == cid
          and life.attempt == daemon._worktree_lifecycle_attempt_for_identity(task_id=task_id, canonical_task_cid=cid, portal_attempt=attempt)
          and life.workspace_path == str(workspace) and life.repo_root == str(daemon.repo_root.resolve())
          and life.state_dir == str(daemon.state_path.parent), "pending_claim_lifecycle_scope_changed")
    _need(daemon.worktree_lifecycle.load_quarantine(workspace) is None and not daemon.worktree_lifecycle.quarantines_for_task_attempt(
          canonical_task_cid=cid, task_id=task_id, attempt=life.attempt), "lifecycle_quarantine_present")
    rows = values["events"]
    starts = [row for row in rows if row.get("type") == "implementation_started" and row.get("canonical_task_cid") == cid]
    _need(starts and starts[-1].get("event_id") == request["start_event_id"]
          and sum(row.get("attempt") == attempt for row in starts) == 1, "latest_exact_start_required")
    start = starts[-1]
    _need(start.get("task_id") == task_id and start.get("attempt") == attempt
          and start.get("canonical_task_key") == request["task_key"] and start.get("board_namespace") == request["board_namespace"]
          and start.get("worktree_path") == str(workspace) and start.get("branch") == life.branch
          and start.get("baseline_ref") == request["baseline_ref"], "pending_claim_start_scope_changed")
    _need(not any(row.get("type") in _OUTCOMES and row.get("canonical_task_cid") == cid
          and row.get("attempt") == attempt for row in rows), "attempt_already_has_outcome")
    claim, launcher, shutdown, strategy = (values[key] for key in ("claim", "launcher", "shutdown", "strategy"))
    _need(claim.get("kind") == native.IMPLEMENTATION_TASK_CLAIM_LOCK_KIND
          and claim.get("lease_role") == native.IMPLEMENTATION_DISPATCH_INTENT_LEASE_ROLE
          and claim.get("task_id") == task_id and claim.get("canonical_task_cid") == cid
          and claim.get("canonical_task_key") == request["task_key"] and claim.get("board_namespace") == request["board_namespace"]
          and type(claim.get("attempt")) is int and claim["attempt"] == attempt
          and type(claim.get("lease_id")) is str and bool(claim["lease_id"])
          and claim.get("state_path") == str(daemon.state_path) and claim.get("state_dir") == str(daemon.state_path.parent)
          and claim.get("repository_id") == daemon.merge_target_repository_id
          and claim.get("task_shard_count") == daemon.task_shard_count and claim.get("task_shard_index") == daemon.task_shard_index
          and claim.get("dispatch_authority_cid") == daemon._implementation_dispatch_authority_cid(claim)
          and not native.implementation_task_claim_protected_fence_paths(claim), "pending_claim_identity_changed")
    _need(claim.get("started_at") == state.last_implementation_started_at, "claim_start_identity_changed")
    birth = native.ProcessBirthIdentity.from_dict(claim.get("owner_process_birth"))
    launch = launcher.get("identity", {})
    launch_birth = native.ProcessBirthIdentity(pid=launch.get("pid", 0), start_time_ticks=launch.get("start_ticks", 0), boot_id=launch.get("boot_id", ""))
    for owner in (birth, launch_birth):
        _need(owner.pid > 0 and owner.start_time_ticks > 0 and owner.boot_id
              and native.owner_liveness(owner, proc_root=daemon.worktree_lifecycle.proc_root) is native.OwnerLiveness.DEAD,
              "pending_claim_owner_live_or_unknown")
    maintenance_owner = _terminal_maintenance_owner(daemon, native, life.owner)
    _need(claim.get("pid") == birth.pid and birth.boot_id == launch_birth.boot_id
          and (birth.pid == launch_birth.pid and birth.start_time_ticks == launch_birth.start_time_ticks
               or birth.parent_pid == launch_birth.pid and birth.start_time_ticks >= launch_birth.start_time_ticks),
          "claim_not_in_stopped_launcher")
    _need(launcher.get("status") == "stopped" and launcher.get("repo_root") == str(daemon.repo_root.resolve())
          and launcher.get("state_dir") == str(daemon.state_path.parent)
          and launcher.get("board_path") == str(daemon.todo_path) and type(launcher.get("owner_token")) is str
          and bool(launcher["owner_token"]) and launcher.get("plan_id"), "stopped_launcher_binding_required")
    _need(shutdown.get("status") == "stopped" and shutdown.get("quiesced") is True
          and shutdown.get("reason") == "owned_container_closure_verified"
          and shutdown.get("repo_root") == str(daemon.repo_root.resolve()) and shutdown.get("state_dir") == str(daemon.state_path.parent)
          and shutdown.get("plan_id") == launcher["plan_id"] and shutdown.get("started_at") == launcher.get("started_at")
          and shutdown.get("owner_token_sha256") == _sha(launcher["owner_token"].encode())
          and all(shutdown.get(key) is False for key in _FALSE if key != "new_attempt_authority"),
          "bound_owned_container_closure_required")
    # Controlled restart may replace the lifecycle owner with a maintenance
    # process outside the launch tree. Its exact terminal tuple must be in the
    # authenticated native shutdown history; it need not impersonate the old daemon.
    original = start.get("worktree_lifecycle", {})
    _need(original.get("state") == "active" and type(original.get("fence")) is int
          and original["fence"] + (2 if candidate is not None else 1) == life.fence
          and original.get("lease_id") != life.lease_id
          and life.terminal_reason in {"controlled_restart_dead_owner", "controlled_shutdown_quiesced_owner"}
          and life.lane_id == f"{daemon.state_path.parent}:{daemon.task_shard_index}/{daemon.task_shard_count}:{birth.pid}",
          "controlled_terminal_transition_unproved")
    cleanups = [row for row in rows if row.get("type") == "implementation_shutdown_reconciliation_blocked"
                and row.get("sequence", 0) > start["sequence"] and row.get("task_id") == task_id
                and row.get("attempt") == attempt]
    _need(any(row.get("reason") == "task_claim_reconciliation_blocked"
          and row.get("task_claim_reconciliation", {}).get("canonical_task_cid") == cid
          and row.get("task_claim_reconciliation", {}).get("reason") == "canonical_task_not_terminal"
          and all(row.get("worktree_lifecycle_reconciliation", {}).get(key) == value for key, value in {
              "record_id": life.record_id, "fence": life.fence, "owner_pid": life.owner.pid,
              "workspace_path": str(workspace), "state": "terminal", "canonical_task_cid": cid}.items())
          for row in cleanups), "terminal_shutdown_history_unproved")
    _need(any(row.get("type") == "implementation_protected_path_snapshot_reconciled"
          and row.get("task_id") == task_id and row.get("attempt") == attempt
          and row.get("sequence", 0) > start["sequence"] and row.get("blocked") is False
          and row.get("reason") == "crash_reconciliation_unchanged"
          and row.get("workspace_path") == str(workspace) for row in rows), "unchanged_protected_reconciliation_required")
    for key in ("blocked_tasks", "deprioritized_tasks", "manual_holds", "manual_hold_tasks", "blocked_task_reasons", "independent_blockers"):
        _need(not strategy.get(key), "independent_strategy_hold")
    _need(not strategy.get("autonomous_unstall_quarantines"), "autonomous_quarantine_present")
    lease_path = daemon._implementation_lock_path()
    lease = _object(_read(lease_path)) if os.path.lexists(lease_path) else None
    _need((_sha(_read(lease_path)) if lease is not None else "") == request["implementation_lease_sha256"], "implementation_lease_changed")
    lease_proof = _ordinary_implementation_lease_proof(daemon, request, lease, claim, birth)
    _need(not daemon._implementation_process_active(start), "implementation_process_still_active")
    _fresh_container_quiescence(daemon, workspace)
    if candidate is None:
        _baseline_unchanged(task, workspace, request["baseline_ref"], native)
        _need(_git(workspace, "rev-parse", "HEAD") == _git(workspace, "rev-parse", request["baseline_ref"] + "^{commit}")
              and not _git(workspace, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=all"),
              "pending_workspace_not_clean_baseline")
    else:
        _need(_candidate_source(daemon, request) == candidate, "pending_candidate_source_changed")
    return {"claim_raw_base64": base64.b64encode(_bytes(claim)).decode(), "claim": claim,
            "implementation_lease_proof": lease_proof, "maintenance_owner_proof": maintenance_owner,
            "task_status": native.normalize_status(task.status)}


def review(daemon, *, task_id: str) -> dict:
    """Read and fully prove one stopped pending claim, without publishing files."""
    request = _build(daemon, task_id)
    _validated(daemon, request, None)
    return request


def reconcile(daemon, *, request: Mapping) -> dict:
    """CAS-release one exact dead claim; callers retain application closure locks."""
    from . import implementation_daemon as native
    result = {"reconciled": False, "blocked": True, "provider_outcome": "unknown", **_FALSE}
    try:
        _need(isinstance(request, Mapping), "pending_claim_request_required")
        request = deepcopy(dict(request))
        _ordinary(daemon)
        operation = "sha256:" + _sha(_bytes(request))
        directory = daemon.state_path.parent / DIRECTORY
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
        _need(directory.resolve() == directory and stat.S_ISDIR(info.st_mode)
              and info.st_uid == os.getuid() and not info.st_mode & 0o077, "unsafe_pending_claim_receipt_directory")
        path = directory / (_sha(_bytes(request)) + ".json")
        claim_path = daemon._implementation_task_claim_path(request["task_id"], canonical_task_cid=request["task_cid"])
        prior = _object(_read(path)) if os.path.lexists(path) else None
        if prior is not None:
            _need(prior.get("request") == request and prior.get("operation_id") == operation
                  and prior.get("receipt_id") == "sha256:" + _sha(_bytes({k:v for k,v in prior.items() if k != "receipt_id"})),
                  "pending_claim_receipt_corrupt")
        # Validate untrusted request paths before acquiring guards (which may
        # create their adjacent lock files). Mutation locks use derived paths.
        _validated(daemon, request, prior)
        workspace = Path(request["workspace"])
        with ExitStack() as stack:
            for lock in (daemon._implementation_lock_path(), directory / "recovery.lock", claim_path, daemon.strategy_path,
                         daemon.worktree_lifecycle.task_index_path_for(canonical_task_cid=request["task_cid"],
                             task_id=request["task_id"], attempt=daemon.worktree_lifecycle.load_workspace(workspace).attempt),
                         daemon.worktree_lifecycle.workspace_path_for(workspace)):
                stack.enter_context(native.serialized_lock_update(lock))
            prior = _object(_read(path)) if os.path.lexists(path) else None
            if prior is not None:
                _need(prior.get("request") == request and prior.get("operation_id") == operation
                      and prior.get("receipt_id") == "sha256:" + _sha(_bytes({k:v for k,v in prior.items() if k != "receipt_id"})),
                      "pending_claim_receipt_corrupt")
            checked = _validated(daemon, request, prior)
            body = {"schema": SCHEMA, "request": request, "operation_id": operation,
                    "claim_raw_base64": base64.b64encode(_read(claim_path)).decode() if claim_path.exists() else prior["claim_raw_base64"],
                    "implementation_lease_proof": checked["implementation_lease_proof"],
                    "maintenance_owner_proof": checked["maintenance_owner_proof"],
                    "provider_outcome": "unknown", "task_status": checked["task_status"], **_FALSE}
            def publish(phase):
                value = {**body, "phase": phase}
                value["receipt_id"] = "sha256:" + _sha(_bytes(value))
                daemon._write_private_task_claim_release_receipt(path, value)
                return value
            if prior is not None and prior.get("phase") == "released":
                _need(not os.path.lexists(claim_path), "released_pending_claim_reappeared")
                return {**result, "reconciled": True, "blocked": False, "reason": "pending_claim_already_released",
                        "receipt_path": str(path), "receipt_id": prior["receipt_id"], "counters_preserved": True, "task_status_unchanged": True}
            if prior is None:
                prior = publish("prepared")
            _need(prior.get("phase") == "prepared", "pending_claim_receipt_phase_invalid")
            _validated(daemon, request, prior)
            if os.path.lexists(claim_path):
                claim_path.unlink()
                fd = os.open(claim_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
            _validated(daemon, request, prior)
            released = publish("released")
            daemon._record_event("implementation_pending_task_claim_released", {
                "task_id": request["task_id"], "canonical_task_cid": request["task_cid"],
                "attempt": request["attempt"], "operation_id": operation, "receipt_id": released["receipt_id"], **_FALSE})
            return {**result, "reconciled": True, "blocked": False, "reason": "quiesced_pending_task_claim_released",
                    "receipt_path": str(path), "receipt_id": released["receipt_id"], "counters_preserved": True, "task_status_unchanged": True}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        return {**result, "reason": str(exc) if type(exc) is ValueError else type(exc).__name__}


def _candidate_dependencies(workspace: Path) -> dict:
    """Pin initialized dependencies; root-only custody cannot discard their edits."""
    result = {}
    def visit(root, prefix="", depth=0):
        _need(depth <= 8 and len(result) <= 64, "candidate_dependency_inventory_limit")
        for row in _git(root, "ls-files", "--stage", "-z").split("\0"):
            if not row: continue
            metadata, relative = row.split("\t", 1)
            mode, commit, stage = metadata.split()
            if mode != "160000": continue
            path = root / relative
            _need(stage == "0" and path.resolve() == path and not path.is_symlink(), "candidate_dependency_alias")
            if not path.exists(): continue
            _need(path.is_dir(), "candidate_dependency_not_directory")
            if not (path / ".git").exists():
                _need(not any(path.iterdir()), "candidate_unmanaged_dependency_contents")
                continue
            _need(_git(path, "rev-parse", "--show-toplevel") == str(path)
                  and _git(path, "rev-parse", "HEAD") == commit, "candidate_dependency_revision_changed")
            _need(not _git(path, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=all"),
                  "candidate_dependency_source_changed")
            result[prefix + relative] = commit
            visit(path, prefix + relative + "/", depth + 1)
    visit(workspace)
    return result


def _candidate_proposal_id(daemon, task, workspace: Path, baseline: str) -> str:
    """Rebuild Native's source-bound proposal without admission or mutation.

    This deliberately admits root regular files only. Submodule expansion,
    staging and scope adjudication remain the normal later Native gates.
    """
    from . import implementation_daemon as native
    from ..proof.code_proof_obligations import collect_git_candidate_diff
    from ..validation.proposal_validation import (
        ImplementationProposal, ProposalOperation, ProposalRisk, ProposalValidationStep,
    )
    authority = daemon._proposal_authority_values(task, workspace_path=workspace, baseline_ref=baseline)
    entries = tuple(collect_git_candidate_diff(workspace, base_revision=baseline, include_untracked=True))
    changed = tuple(sorted({path for entry in entries for path in (entry.old_path, entry.new_path) if path}))
    scope = daemon._proposal_scope_paths(task)
    rationale = tuple(sorted({f"task:{task.task_id}",
        *((f"acceptance:{task.task_id}",) if task.acceptance else ()),
        *(f"output:{path}" for path in scope)}))
    operations = tuple(ProposalOperation(operation=entry.change_kind.value, path=entry.path,
        old_path=entry.old_path if entry.change_kind.value in {"rename", "copy"} else "",
        rationale_refs=(f"output:{entry.path}" if f"output:{entry.path}" in rationale else f"task:{task.task_id}",))
        for entry in entries)
    commands = []
    for raw in task.validation:
        command = native.rewrite_validation_command(str(raw))
        if daemon._task_uses_typed_local_execution(task):
            command, _ = daemon._normalize_validation_command(command)
        commands.append(ProposalValidationStep(command=tuple(shlex.split(command)),
            rationale_refs=(f"task:{task.task_id}",)))
    nonce = hashlib.sha256(json.dumps({"task_id": task.task_id,
        "tree": authority["repository_tree_id"], "context": authority["context_id"],
        "changed_paths": changed, "operations": [(e.change_kind.value, e.old_path, e.new_path) for e in entries]},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ImplementationProposal(**authority, candidate_diff=entries, declared_paths=changed,
        operations=operations, rationale_references=rationale, validation_plan=tuple(commands),
        risks=(ProposalRisk(risk="candidate behavior may regress task-owned interfaces",
            mitigation="run the task-declared validation plan before merge"),),
        authority_claims={**authority, "proof_authoritative": False, "code_proof_authoritative": False,
            "completion_authoritative": False, "completed": False},
        patch_text=daemon._proposal_patch_text(workspace, baseline, entries), replay_nonce=nonce,
        proposal_version="2").proposal_id


def _candidate_source(daemon, base: dict) -> dict:
    """Pin only already-staged declared regular outputs; never stage in review."""
    from . import implementation_daemon as native
    task = next(task for task in daemon._load_tasks() if task.task_id == base["task_id"])
    workspace, baseline = Path(base["workspace"]), base["baseline_ref"]
    _need(_git(workspace, "rev-parse", "HEAD") == baseline, "candidate_requires_original_baseline_head")
    dependencies = _candidate_dependencies(workspace)
    rows = _git(workspace, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=all").splitlines()
    allowed = set(task.outputs)
    _need(allowed and rows, "nonempty_declared_candidate_required")
    changed = []
    for row in rows:
        _need(len(row) >= 4 and row[0] in {"A", "M"} and row[1] == " "
              and row[3:] in allowed, "candidate_scope_or_unstaged_change")
        changed.append(row[3:])
    _need(len(changed) == len(set(changed)), "candidate_duplicate_path")
    files = {}
    for relative in sorted(allowed):
        _need(native.PortalImplementationDaemon._repo_relative_path_safe(relative), "candidate_unsafe_output_path")
        path = workspace / relative
        _need(path.resolve(strict=True) == path and path.is_file(), "candidate_output_alias_or_missing")
        raw = _read(path)
        entries = _git(workspace, "ls-files", "--stage", "--", relative).splitlines()
        _need(len(entries) == 1, "candidate_output_not_tracked")
        metadata, name = entries[0].split("\t", 1)
        mode, blob, stage = metadata.split()
        _need(name == relative and mode in {"100644", "100755"} and stage == "0", "candidate_output_mode_changed")
        indexed = subprocess.run(["git", "--no-replace-objects", "show", ":" + relative], cwd=workspace,
                                 capture_output=True, timeout=15, check=False)
        _need(indexed.returncode == 0 and indexed.stdout == raw, "candidate_index_working_bytes_differ")
        files[relative] = {"sha256": _sha(raw), "bytes": len(raw), "mode": mode, "blob": blob}
    rows = _event_rows(_read(daemon.events_path))
    start = next(r for r in rows if r["event_id"] == base["start_event_id"])
    _need(_git(workspace, "symbolic-ref", "--short", "HEAD") == start.get("branch"),
          "candidate_workspace_branch_changed")
    accepted = [row for row in rows if row.get("task_id") == base["task_id"]
                and row.get("canonical_task_cid") == base["task_cid"]
                and row.get("sequence", 0) > next(r["sequence"] for r in rows if r["event_id"] == base["start_event_id"])
                and row.get("type") in {"implementation_proposal_validated", "implementation_proposal_receipt_reused",
                                         "implementation_proposal_rejected"}]
    _need(accepted and accepted[-1].get("accepted") is True and accepted[-1].get("attempted") is True
          and not accepted[-1].get("reason_codes") and accepted[-1].get("repository_tree_id") == baseline
          and accepted[-1].get("canonical_task_key") == base["task_key"]
          and all(re.fullmatch(r"[a-f0-9]{64}", str(accepted[-1].get(key)))
                  for key in ("proposal_id", "receipt_id", "policy_id")), "latest_accepted_candidate_proposal_required")
    _need(_candidate_proposal_id(daemon, task, workspace, baseline) == accepted[-1]["proposal_id"],
          "accepted_proposal_candidate_changed")
    patch = _git(workspace, "diff", "--cached", "--binary", "--full-index", baseline, "--")
    _need(bool(patch), "nonempty_candidate_patch_required")
    return {"files": files, "changed_paths": sorted(changed), "patch_sha256": _sha(patch.encode()),
            "accepted_event_id": accepted[-1]["event_id"], "proposal_id": accepted[-1].get("proposal_id"),
            "baseline_ref": baseline, "dependencies": dependencies}


def _candidate_receipt_path(daemon, task_id: str) -> Path:
    native = _ordinary(daemon)
    tasks = [task for task in daemon._load_tasks() if task.task_id == task_id]
    _need(len(tasks) == 1, "current_task_not_unique")
    identity = daemon._identity_for_task(tasks[0])
    state = native.PortalTaskState.load(daemon.state_path)
    attempt = state.implementation_attempts.get(task_id)
    _need(type(attempt) is int and attempt > 0, "consumed_ordinal_mismatch")
    key = _sha(_bytes({"task_cid": identity.canonical_task_cid, "attempt": attempt}))
    return daemon.state_path.parent / DIRECTORY / ("candidate-" + key + ".json")


def _candidate_prior(path: Path, request: dict | None = None) -> dict | None:
    if not os.path.lexists(path):
        return None
    prior = _object(_read(path))
    _need(prior.get("schema") == CANDIDATE_SCHEMA
          and prior.get("receipt_id") == "sha256:" + _sha(_bytes({k:v for k,v in prior.items() if k != "receipt_id"}))
          and (request is None or prior.get("request") == request), "candidate_receipt_corrupt")
    _need(prior.get("phase") in {"prepared", "preserved", "retired", "released"}
          and prior.get("operation_id") == "sha256:" + _sha(_bytes(prior["request"]))
          and all(prior.get(key) is False for key in _FALSE), "candidate_receipt_corrupt")
    return prior


def _candidate_stable(daemon, request: dict, prior: dict, *, released=False) -> None:
    """Check the authorities which preservation is never allowed to change."""
    native = _ordinary(daemon); base = request["base"]
    _need(set(request) == {"schema", "base", "candidate"} and request["schema"] == CANDIDATE_SCHEMA
          and base["schema"] == SCHEMA and base["repo_root"] == str(daemon.repo_root.resolve())
          and base["state_path"] == str(daemon.state_path) and base["task_header_prefix"] == daemon.task_header_prefix,
          "candidate_request_scope_changed")
    _need(base["runtime_revision"] == runtime_revision() and base["runtime_sources"] == runtime_sources(),
          "pending_claim_runtime_changed")
    _need(_state_hash(daemon) == base["state_sha256"] and _queue(daemon) == base["merge_queue"],
          "candidate_state_or_queue_changed")
    _need(_git(daemon.repo_root, "rev-parse", "HEAD") == base["repo_head"], "pending_claim_source_changed")
    _need(daemon.max_task_attempts <= 0 or base["attempt"] < daemon.max_task_attempts, "retry_budget_exhausted")
    task = [t for t in daemon._load_tasks() if t.task_id == base["task_id"] and daemon._canonical_ref(t) == base["task_cid"]]
    _need(len(task) == 1 and native.normalize_status(task[0].status) in {"todo", "ready", "in_progress"},
          "pending_current_task_required")
    identity = daemon._identity_for_task(task[0])
    _need(identity.canonical_task_key == base["task_key"] and identity.board_namespace == base["board_namespace"],
          "pending_claim_task_identity_changed")
    _need(daemon._retained_provider_capacity_deferral() is None, "capacity_hold_present")
    for name in ("strategy", "launcher", "shutdown", "board"):
        pin = base["pins"][name]
        _need(_pin(Path(pin["path"])) == pin, "candidate_stable_" + name + "_changed")
    claim = Path(base["pins"]["claim"]["path"])
    if os.path.lexists(claim):
        _need(not released and _read(claim) == base64.b64decode(prior["claim_raw_base64"], validate=True)
              and _pin(claim) == base["pins"]["claim"], "candidate_claim_changed")
    else:
        _need(prior.get("phase") in {"preserved", "retired", "released"}, "candidate_claim_missing_before_preservation")
    rows = _event_rows(_read(daemon.events_path)); pin = base["pins"]["events"]
    _need(_sha(_read(daemon.events_path)[:pin["bytes"]]) == pin["sha256"], "candidate_history_changed")
    info = daemon.events_path.lstat()
    _need(all(getattr(info, key) == value for key, value in pin["identity"].items()
              if key not in {"st_mtime_ns", "st_ctime_ns"}), "candidate_history_identity_changed")
    starts = [row for row in rows if row.get("type") == "implementation_started"
              and row.get("canonical_task_cid") == base["task_cid"]]
    _need(starts and starts[-1]["event_id"] == base["start_event_id"] and not any(
        row.get("type") in _OUTCOMES and row.get("canonical_task_cid") == base["task_cid"]
        and row.get("attempt") == base["attempt"] for row in rows), "candidate_attempt_history_changed")
    if prior["phase"] != "prepared":
        _need(prior.get("preserved_candidate") == _preserved_history(daemon, request, prior["operation_id"],
              _candidate_receipt_path(daemon, base["task_id"])), "candidate_preservation_history_changed")
    launch = _object(_read(Path(base["pins"]["launcher"]["path"])))["identity"]
    launch_birth = native.ProcessBirthIdentity(pid=launch["pid"], start_time_ticks=launch["start_ticks"], boot_id=launch["boot_id"])
    for owner in (native.ProcessBirthIdentity.from_dict(prior["claim"]["owner_process_birth"]), launch_birth):
        _need(native.owner_liveness(owner, proc_root=daemon.worktree_lifecycle.proc_root) is native.OwnerLiveness.DEAD,
              "pending_claim_owner_live_or_unknown")
    _terminal_maintenance_owner(daemon, native, native.ProcessBirthIdentity.from_dict(prior["terminal_lifecycle"]["owner"]))
    workspace = Path(base["workspace"])
    life = daemon.worktree_lifecycle.load_workspace(workspace)
    if life is not None:
        _need(life.to_dict() == prior["terminal_lifecycle"], "candidate_lifecycle_changed")
        _need(_pin(daemon.worktree_lifecycle.task_index_path_for(canonical_task_cid=base["task_cid"],
                  task_id=base["task_id"], attempt=life.attempt)) == base["pins"]["lifecycle_index"],
              "candidate_lifecycle_index_changed")
    else:
        _need(prior["phase"] != "prepared" or _preserved_history(daemon, request, prior["operation_id"],
              _candidate_receipt_path(daemon, base["task_id"])) is not None, "candidate_lifecycle_missing")
        _need(not daemon.worktree_lifecycle.task_index_path_for(canonical_task_cid=base["task_cid"],
              task_id=base["task_id"], attempt=prior["terminal_lifecycle"]["attempt"]).exists(),
              "candidate_lifecycle_index_changed")
    _need(daemon.worktree_lifecycle.load_quarantine(workspace) is None
          and not daemon.worktree_lifecycle.quarantines_for_task_attempt(canonical_task_cid=base["task_cid"],
              task_id=base["task_id"], attempt=prior["terminal_lifecycle"]["attempt"]), "lifecycle_quarantine_present")
    for protected in (daemon._implementation_protected_active_snapshot_path(), daemon._implementation_protected_incident_path(),
                      daemon._protected_path_maintenance_lock_path()):
        _need(not os.path.lexists(protected), "protected_hold_present")
    lease_path = daemon._implementation_lock_path()
    lease_sha = _sha(_read(lease_path)) if os.path.lexists(lease_path) else ""
    if lease_sha != base["implementation_lease_sha256"]:
        _need(released and not lease_sha and any(row.get("type") == "implementation_lock_cleared"
            and row.get("lock_path") == str(lease_path) and row.get("lock_owner_pid") == prior["claim"]["pid"]
            and row.get("task_id") == base["task_id"] for row in rows), "implementation_lease_changed")
    _need(not daemon._implementation_process_active(next(row for row in rows if row["event_id"] == base["start_event_id"])),
          "implementation_process_still_active")
    _fresh_container_quiescence(daemon, Path(base["workspace"]))


def _candidate_checkout(daemon, request: dict, prior: dict, *, preserved: dict | None = None) -> None:
    """A prior checkpoint never authorizes cleanup of a reused checkout/ref."""
    workspace = Path(request["base"]["workspace"])
    branch = prior["terminal_lifecycle"]["branch"]
    if workspace.exists():
        _need(workspace.resolve() == workspace and _git(workspace, "symbolic-ref", "--short", "HEAD") == branch,
              "candidate_workspace_reused")
        _need(_candidate_dependencies(workspace) == request["candidate"]["dependencies"],
              "candidate_dependencies_changed")
        head = _git(workspace, "rev-parse", "HEAD")
        if preserved is None and head == request["base"]["baseline_ref"]:
            _need(_candidate_source(daemon, request["base"]) == request["candidate"], "pending_candidate_source_changed")
        else:
            _candidate_commit(daemon, request, preserved or {"commit": head})
            _need(head == (preserved or {"commit": head})["commit"]
                  and not _git(workspace, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=all"),
                  "candidate_workspace_changed_after_preservation")
    else:
        _need(preserved is not None, "candidate_workspace_missing_without_preservation")
    if daemon._git_ref_exists(branch):
        expected = preserved["commit"] if preserved else (_git(workspace, "rev-parse", "HEAD"))
        _need(_git(daemon.repo_root, "rev-parse", branch) == expected, "candidate_branch_reused")


def _candidate_commit(daemon, request: dict, preserved: dict) -> None:
    base, candidate = request["base"], request["candidate"]
    commit = preserved["commit"]
    if preserved.get("rescue_ref"):
        _need(_git(daemon.repo_root, "rev-parse", preserved["rescue_ref"]) == commit, "candidate_rescue_ref_changed")
    _need(_git(daemon.repo_root, "rev-list", "--parents", "-n", "1", commit).split() == [commit, base["baseline_ref"]],
          "candidate_commit_parent_changed")
    patch = _git(daemon.repo_root, "diff", "--binary", "--full-index", base["baseline_ref"], commit, "--")
    _need(_sha(patch.encode()) == candidate["patch_sha256"], "candidate_commit_patch_changed")
    for relative, pin in candidate["files"].items():
        result = subprocess.run(["git", "--no-replace-objects", "show", commit + ":" + relative], cwd=daemon.repo_root,
                                capture_output=True, timeout=15, check=False)
        _need(result.returncode == 0 and _sha(result.stdout) == pin["sha256"], "candidate_commit_output_changed")


def _preserved_history(daemon, request, operation, path):
    base = request["base"]
    native = _ordinary(daemon)
    _need(path == _candidate_receipt_path(daemon, base["task_id"]), "candidate_receipt_scope_changed")
    prior = _candidate_prior(path, request)
    _need(prior is not None and prior["operation_id"] == operation == "sha256:" + _sha(_bytes(request)),
          "candidate_prepared_operation_required")
    identities = {"canonical_task_cid": base["task_cid"], "canonical_task_key": base["task_key"],
                  "board_namespace": base["board_namespace"]}
    task = [task for task in daemon._load_tasks() if task.task_id == base["task_id"]]
    _need(len(task) == 1 and native.normalize_status(task[0].status) in {"todo", "in_progress", "ready"}
          and _state_hash(daemon) == base["state_sha256"], "candidate_current_task_required")
    identity = daemon._identity_for_task(task[0])
    _need(identity.canonical_task_cid == base["task_cid"] and identity.canonical_task_key == base["task_key"]
          and identity.board_namespace == base["board_namespace"], "candidate_current_identity_changed")
    evidence = {"operation_id": operation, "candidate_manifest_sha256": _sha(_bytes(request["candidate"])),
                **identities, **_FALSE}
    rows = _event_rows(_read(daemon.events_path))
    matches = [row for row in rows if row.get("type") == "implementation_pending_candidate_preserved"
        and row.get("pending_candidate_recovery") == evidence
        and all(row["pending_candidate_recovery"].get(key) is False for key in _FALSE)
        and row.get("task_id") == base["task_id"] and type(row.get("attempt")) is int
        and row["attempt"] == base["attempt"]
        and (not any(key in row for key in identities) or all(row.get(key) == value for key, value in identities.items()))]
    if not matches: return None
    _need(len(matches) == 1, "candidate_checkpoint_ambiguous")
    row = matches[-1]
    start = next(row for row in rows if row["event_id"] == base["start_event_id"])
    decisions = [row for row in rows if row.get("type") in {"implementation_proposal_validated",
        "implementation_proposal_receipt_reused", "implementation_proposal_rejected"}
        and row.get("task_id") == base["task_id"] and row.get("canonical_task_cid") == base["task_cid"]
        and row.get("sequence", 0) > start["sequence"]]
    _need(decisions and decisions[-1]["event_id"] == request["candidate"]["accepted_event_id"],
          "candidate_proposal_superseded")
    expected_ref = daemon._interrupted_worktree_rescue_branch_name(start["branch"],
        "pending-candidate-" + operation.removeprefix("sha256:")[:20])
    _need(row.get("branch") == start["branch"] and row.get("worktree_path") == base["workspace"]
          and row.get("rescue_branch") == expected_ref, "native_preservation_scope_changed")
    value = {"commit": row["preserved_commit"], "rescue_ref": row["rescue_branch"],
        "baseline_ref": base["baseline_ref"], "source_root": str(daemon.repo_root),
        "original_workspace": base["workspace"], "original_branch": row["branch"],
        "task_id": base["task_id"], "task_cid": base["task_cid"], "task_key": base["task_key"],
        "attempt": base["attempt"], "source_manifest_sha256": _sha(_bytes(request["candidate"])),
        "preservation_receipt_path": str(path), "preservation_event_id": row["event_id"]}
    _need(row.get("preserved") is True and row.get("pending_candidate_recovery", {}).get("candidate_manifest_sha256")
          == value["source_manifest_sha256"], "native_preservation_event_changed")
    _candidate_commit(daemon, request, value)
    return value


def review_candidate(daemon, *, task_id: str) -> dict:
    """Review staged provider code without granting it validation authority."""
    path = _candidate_receipt_path(daemon, task_id); prior = _candidate_prior(path)
    if prior is not None:
        request = prior["request"]
        _candidate_stable(daemon, request, prior, released=prior["phase"] == "released")
        if prior.get("preserved_candidate"):
            _candidate_commit(daemon, request, prior["preserved_candidate"])
        elif _preserved_history(daemon, request, prior["operation_id"], path) is None:
            _candidate_checkout(daemon, request, prior)
        return request
    base = _build(daemon, task_id); candidate = _candidate_source(daemon, base)
    _validated(daemon, base, None, candidate=candidate)
    return {"schema": CANDIDATE_SCHEMA, "base": base, "candidate": candidate}


def reconcile_candidate(daemon, *, request: Mapping) -> dict:
    """Preserve exact code with Native, retire its old fence, then CAS-release."""
    result = {"reconciled": False, "blocked": True, "provider_outcome": "unknown", **_FALSE}
    try:
        _need(isinstance(request, Mapping), "candidate_request_required")
        request = deepcopy(dict(request)); _need(set(request) == {"schema", "base", "candidate"}
            and request["schema"] == CANDIDATE_SCHEMA, "candidate_request_shape_changed")
        base = request["base"]; native = _ordinary(daemon)
        path = _candidate_receipt_path(daemon, base["task_id"])
        prior = _candidate_prior(path, request)
        if prior is None:
            checked = _validated(daemon, base, None, candidate=request["candidate"])
        path.parent.mkdir(mode=0o700, exist_ok=True)
        _need(path.parent.resolve() == path.parent and path.parent.stat().st_uid == os.getuid()
              and not path.parent.stat().st_mode & 0o077, "unsafe_pending_claim_receipt_directory")
        claim_path = Path(base["pins"]["claim"]["path"])
        operation = "sha256:" + _sha(_bytes(request))
        with ExitStack() as stack:
            # Native preservation/finalization own their lifecycle locks. Do
            # not retain those here and deadlock its existing CAS operations.
            for lock in (daemon._implementation_lock_path(), path.parent / "recovery.lock", claim_path, daemon.strategy_path):
                stack.enter_context(native.serialized_lock_update(lock))
            prior = _candidate_prior(path, request)
            if prior is None:
                checked = _validated(daemon, base, None, candidate=request["candidate"])
                branch = _object(_read(Path(base["pins"]["lifecycle"]["path"]))) ["branch"]
                rescue = daemon._interrupted_worktree_rescue_branch_name(branch,
                    "pending-candidate-" + operation.removeprefix("sha256:")[:20])
                _need(not daemon._git_ref_exists(rescue), "candidate_rescue_ref_already_exists")
                body = {"schema": CANDIDATE_SCHEMA, "request": request, "operation_id": operation,
                        "claim_raw_base64": base64.b64encode(_read(claim_path)).decode(), "claim": checked["claim"],
                        "terminal_lifecycle": _object(_read(Path(base["pins"]["lifecycle"]["path"]))), **_FALSE}
            else: body = {k:v for k,v in prior.items() if k not in {"phase", "receipt_id"}}
            def publish(phase):
                value = {**body, "phase": phase}; value["receipt_id"] = "sha256:" + _sha(_bytes(value))
                daemon._write_private_task_claim_release_receipt(path, value); return value
            if prior is None: prior = publish("prepared")
            _candidate_stable(daemon, request, prior, released=prior["phase"] == "released")
            if prior["phase"] == "prepared":
                history = _preserved_history(daemon, request, operation, path)
                if history is None:
                    _candidate_checkout(daemon, request, prior)
                    rescue = daemon._interrupted_worktree_rescue_branch_name(prior["terminal_lifecycle"]["branch"],
                        "pending-candidate-" + operation.removeprefix("sha256:")[:20])
                    if daemon._git_ref_exists(rescue):
                        head = _git(Path(base["workspace"]), "rev-parse", "HEAD")
                        _candidate_commit(daemon, request, {"commit": head, "rescue_ref": rescue})
                    task = next(t for t in daemon._load_tasks() if t.task_id == base["task_id"])
                    # Native itself commits, checkpoints its rescue ref/event,
                    # and cleans the old checkout. No replacement task claim
                    # or source reset is synthesized by this recovery API.
                    preserved = daemon._preserve_interrupted_worktree(Path(base["workspace"]),
                        prior["terminal_lifecycle"]["branch"], task, base["attempt"],
                        evidence={"operation_id": operation, "candidate_manifest_sha256": _sha(_bytes(request["candidate"])),
                            "canonical_task_cid": base["task_cid"], "canonical_task_key": base["task_key"],
                            "board_namespace": base["board_namespace"], **_FALSE},
                        rescue_suffix="pending-candidate-" + operation.removeprefix("sha256:")[:20],
                        event_type="implementation_pending_candidate_preserved", evidence_field="pending_candidate_recovery",
                        baseline_ref=base["baseline_ref"], retain_on_failure=True)
                    _need(preserved.get("preserved") is True, "native_candidate_preservation_failed")
                    history = _preserved_history(daemon, request, operation, path)
                _need(history is not None, "native_preservation_event_required")
                body["preserved_candidate"] = history
                prior = publish("preserved")
            _candidate_commit(daemon, request, prior["preserved_candidate"])
            _candidate_stable(daemon, request, prior, released=prior["phase"] == "released")
            if prior["phase"] == "preserved":
                _candidate_checkout(daemon, request, prior, preserved=prior["preserved_candidate"])
                if Path(base["workspace"]).exists() or daemon._worktree_path_registered_in_repo(daemon.repo_root, Path(base["workspace"])):
                    cleanup = daemon._cleanup_merged_worktree(Path(base["workspace"]), prior["terminal_lifecycle"]["branch"])
                    _need(cleanup.get("cleaned") is True, "candidate_checkout_cleanup_unproved")
                _need(not Path(base["workspace"]).exists()
                      and not daemon._worktree_path_registered_in_repo(daemon.repo_root, Path(base["workspace"]))
                      and not daemon._git_ref_exists(prior["terminal_lifecycle"]["branch"]),
                      "candidate_checkout_cleanup_incomplete")
                life = native.WorkspaceLifecycleRecord.from_dict(prior["terminal_lifecycle"])
                existing = daemon.worktree_lifecycle.load_workspace(base["workspace"])
                if existing is not None:
                    _need(existing == life, "candidate_lifecycle_changed")
                    finalized = daemon._finalize_exact_worktree_lifecycle(life, reason="pending_candidate_preserved")
                    _need(finalized.get("finalized") is True, "candidate_lifecycle_retirement_unproved")
                _need(not daemon.worktree_lifecycle.workspace_path_for(base["workspace"]).exists()
                    and not daemon.worktree_lifecycle.task_index_path_for(canonical_task_cid=base["task_cid"],
                        task_id=base["task_id"], attempt=life.attempt).exists(), "candidate_lifecycle_retirement_incomplete")
                prior = publish("retired")
            _candidate_stable(daemon, request, prior, released=prior["phase"] == "released")
            if prior["phase"] == "retired":
                if claim_path.exists():
                    claim_path.unlink()
                    descriptor = os.open(claim_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try: os.fsync(descriptor)
                    finally: os.close(descriptor)
                prior = publish("released")
                daemon._record_event("implementation_preserved_pending_candidate_claim_released", {
                    "task_id": base["task_id"], "canonical_task_cid": base["task_cid"], "attempt": base["attempt"],
                    "operation_id": operation, "receipt_id": prior["receipt_id"], **_FALSE})
            _need(prior["phase"] == "released", "candidate_receipt_phase_invalid")
            _candidate_stable(daemon, request, prior, released=True)
            return {**result, "reconciled": True, "blocked": False, "reason": "preserved_candidate_pending_task_claim_released",
                    "receipt_path": str(path), "receipt_id": prior["receipt_id"], "preserved_candidate": prior["preserved_candidate"],
                    "counters_preserved": True, "task_status_unchanged": True}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration, subprocess.SubprocessError) as exc:
        return {**result, "reason": str(exc) if type(exc) is ValueError else type(exc).__name__}


def review_recovery(daemon, *, task_id: str) -> dict:
    """Prefer a durable preserved-candidate replay before ordinary no-claim cleanup."""
    path = _candidate_receipt_path(daemon, task_id)
    if os.path.lexists(path): return review_candidate(daemon, task_id=task_id)
    base = _build(daemon, task_id)
    if _git(Path(base["workspace"]), "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=all"):
        return review_candidate(daemon, task_id=task_id)
    return review(daemon, task_id=task_id)


def reconcile_recovery(daemon, *, request: Mapping) -> dict:
    """Dispatch only the closed request schemas produced by the public reviewer."""
    if isinstance(request, Mapping) and request.get("schema") == CANDIDATE_SCHEMA:
        return reconcile_candidate(daemon, request=request)
    return reconcile(daemon, request=request)
