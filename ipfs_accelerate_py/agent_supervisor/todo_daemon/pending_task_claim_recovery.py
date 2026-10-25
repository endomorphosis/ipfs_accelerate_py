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
import os
from pathlib import Path
import stat
import subprocess

from .current_incident_adjudication import (
    _FALSE, _baseline_unchanged, _bytes, _event_rows, _fresh_container_quiescence, _need,
    _object, _ordinary_implementation_lease_proof, _read, _sha, _state_hash,
)

SCHEMA = "ordinary-pending-task-claim-recovery@1"
DIRECTORY = "pending-task-claim-recoveries"
_OUTCOMES = {"implementation_finished", "implementation_terminated", "implementation_timeout"}


def runtime_sources() -> dict:
    """Pin actual imported Native support, including cross-version recovery use.

    A reviewed standalone helper can be loaded beside an older running SDK.
    Its neighboring files are not evidence of the Native code actually used.
    """
    names = (".current_incident_adjudication", ".implementation_daemon",
             ".portal_task_state_control_plane", "..merge.worktree_lifecycle", "..merge.checkout_lock",
             "..merge.merge_queue", "..merge.workspace_quarantine", "..runtime.grok_cli_runner")
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


def _validated(daemon, request: dict, prior: dict | None) -> dict:
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
    for owner in (birth, launch_birth, life.owner):
        _need(owner.pid > 0 and owner.start_time_ticks > 0 and owner.boot_id
              and native.owner_liveness(owner, proc_root=daemon.worktree_lifecycle.proc_root) is native.OwnerLiveness.DEAD,
              "pending_claim_owner_live_or_unknown")
    _need(claim.get("pid") == birth.pid and birth.boot_id == launch_birth.boot_id
          and life.owner.boot_id == launch_birth.boot_id
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
          and original["fence"] + 1 == life.fence and original.get("lease_id") != life.lease_id
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
    _baseline_unchanged(task, workspace, request["baseline_ref"], native)
    _need(_git(workspace, "rev-parse", "HEAD") == _git(workspace, "rev-parse", request["baseline_ref"] + "^{commit}")
          and not _git(workspace, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=all"),
          "pending_workspace_not_clean_baseline")
    return {"claim_raw_base64": base64.b64encode(_bytes(claim)).decode(), "claim": claim,
            "implementation_lease_proof": lease_proof, "task_status": native.normalize_status(task.status)}


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
