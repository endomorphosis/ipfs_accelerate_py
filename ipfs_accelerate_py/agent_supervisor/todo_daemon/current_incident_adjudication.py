"""Explicit, source-bound abandonment of one quiesced ordinary incident.

This is operator custody cleanup, never provider-outcome or task-completion
proof. Original incident/quarantine history and consumed ordinals are retained.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time
from typing import Any

SCHEMA = "ordinary-current-incident-adjudication@1"
DIRECTORY = "ordinary-incident-adjudications"
MAX_BYTES = 64 * 1024 * 1024
_FALSE = {"completion_authority": False, "attempt_refunded": False,
          "provider_callback_settled": False, "new_attempt_authority": False}


def _need(value: Any, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    _need(type(value) is dict, "object_required")
    return value


def _read(path: Path) -> bytes:
    _need(path.is_absolute() and path.resolve() == path, "aliased_evidence_path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
              and before.st_nlink == 1 and not before.st_mode & 0o002
              and before.st_size <= MAX_BYTES, "unsafe_evidence_file")
        with os.fdopen(os.dup(fd), "rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        after = path.lstat()
        _need(len(raw) <= MAX_BYTES and all(getattr(before, key) == getattr(after, key)
              for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
              "evidence_changed")
        return raw
    finally:
        os.close(fd)


def runtime_revision() -> str:
    root = Path(__file__).resolve().parent
    paths = tuple(root / name for name in ("current_incident_adjudication.py",
        "implementation_daemon.py", "implementation_supervisor.py",
        "portal_task_state_control_plane.py", "supervisor_loop.py", "supervisor_runtime.py",
        "../merge/worktree_lifecycle.py", "../merge/workspace_quarantine.py",
        "../rescue/supervisor_watchdog.py", "../rescue/recovery_diagnostics.py"))
    return "sha256:" + _sha(_bytes({str(p.relative_to(root)): _sha(p.read_bytes()) for p in paths}))


def _state_hash(daemon) -> str:
    from .portal_task_state_control_plane import read_task_state_payload
    value = read_task_state_payload(daemon.state_path)
    _need(type(value) is dict, "durable_state_unavailable")
    return _sha(_bytes(value))


def _pins(daemon, task_id: str, incident_cid: str) -> tuple[dict, dict]:
    from . import implementation_daemon as native
    _need(type(daemon) is native.PortalImplementationDaemon, "real_native_portal_required")
    _need(daemon.state_path.is_absolute() and daemon.state_path.resolve() == daemon.state_path, "aliased_state_path")
    state_dir = daemon.state_path.parent
    tasks = [t for t in daemon._load_tasks() if t.task_id == task_id]
    _need(len(tasks) == 1, "current_task_not_unique")
    task = tasks[0]
    cid = daemon._canonical_ref(task)
    state = native.PortalTaskState.load(daemon.state_path)
    count = state.implementation_attempts.get(task_id, 0)
    _need(type(count) is int and count > 0 and state.implementation_attempts_by_cid.get(cid) == count,
          "consumed_ordinal_mismatch")
    starts = [e for e in daemon._iter_merge_lifecycle_events() if e.get("type") == "implementation_started"
              and e.get("canonical_task_cid") == cid and e.get("attempt") == count]
    _need(len(starts) == 1, "exact_start_required")
    start = starts[0]
    workspace = Path(start["worktree_path"])
    prefix = daemon.state_path.name.removesuffix("_task_state.json")
    incident_path = state_dir / f"{prefix}_autonomous_unstall" / "autonomous-unstall-state.json"
    paths = {"claim": daemon._implementation_task_claim_path(task_id, canonical_task_cid=cid),
        "lifecycle": daemon.worktree_lifecycle.workspace_path_for(workspace),
        "strategy": daemon.strategy_path, "incident": incident_path,
        "launcher": state_dir / "launcher.json", "shutdown": state_dir / "container_shutdown.json",
        "board": daemon.todo_path, "events": daemon.events_path,
        "supervisor_events": state_dir / f"{prefix}_supervisor_events.jsonl", "log": Path(start["log_path"])}
    request = {"schema": SCHEMA, "task_id": task_id, "task_cid": cid, "attempt": count,
        "repo_root": str(daemon.repo_root.resolve()), "state_path": str(daemon.state_path.resolve()),
        "task_key": daemon._identity_for_task(task).canonical_task_key,
        "board_namespace": daemon._identity_for_task(task).board_namespace,
        "task_header_prefix": daemon.task_header_prefix,
        "incident_cid": incident_cid, "runtime_revision": runtime_revision(),
        "state_sha256": _state_hash(daemon), "start_event_id": start["event_id"],
        "workspace": str(workspace), "baseline_ref": start["baseline_ref"],
        "pins": {name: {"path": str(path), "bytes": len(raw), "sha256": _sha(raw)}
                 for name, path in paths.items() for raw in (_read(path),)}}
    return request, {"task": task, "state": state, "start": start}


def review(daemon, *, task_id: str, incident_cid: str) -> dict:
    """Build a reviewable preimage; this does not release or write anything."""
    request, _ = _pins(daemon, task_id, incident_cid)
    return request


def _journal_path(state_dir: Path, incident_cid: str) -> Path:
    return state_dir / DIRECTORY / (_sha(incident_cid.encode()) + ".json")


def _event_rows(raw: bytes) -> list[dict]:
    _need(not raw or raw.endswith(b"\n"), "partial_event_history")
    rows = [_object(line) for line in raw.splitlines() if line.strip()]
    previous = ""
    sequence = 0
    for row in rows:
        _need(type(row.get("sequence")) is int and row["sequence"] == sequence + 1,
              "event_sequence_changed")
        body = {k: v for k, v in row.items() if k != "event_id"}
        _need(row.get("event_id") == "sha256:" + _sha(_bytes(body))
              and row.get("previous_event_id", "") == previous, "event_custody_changed")
        sequence = row["sequence"]
        previous = row["event_id"]
    return rows


def _validated(daemon, request: dict, journal: dict | None) -> dict:
    from . import implementation_daemon as native
    _need(type(daemon) is native.PortalImplementationDaemon, "real_native_portal_required")
    _need(set(request) == {"schema", "task_id", "task_cid", "attempt", "incident_cid",
          "runtime_revision", "state_sha256", "start_event_id", "workspace", "baseline_ref", "pins",
          "repo_root", "state_path", "task_key", "board_namespace", "task_header_prefix"}
          and request["schema"] == SCHEMA, "request_shape_changed")
    _need(daemon.task_source is None and daemon.production_provider_policy is None
          and daemon.production_owner_handoff_factory is None
          and getattr(daemon, "_database_attempt_authority", None) is None
          and not daemon.manual_completion_authority_task_ids
          and not daemon.manual_completion_authority_required_task_ids
          and not daemon.manual_completion_authority_revalidation_only,
          "ordinary_pending_claim_required")
    _need(not native.automatic_retry_rearm_enabled(), "strict_attempt_policy_required")
    _need(request["runtime_revision"] == runtime_revision(), "runtime_source_changed")
    _need(daemon.state_path.is_absolute() and daemon.state_path.resolve() == daemon.state_path, "aliased_state_path")
    state_dir = daemon.state_path.parent
    _need(request["repo_root"] == str(daemon.repo_root.resolve())
          and request["state_path"] == str(daemon.state_path.resolve())
          and request["task_header_prefix"] == daemon.task_header_prefix, "foreign_request_scope")
    _need(state_dir.stat().st_uid == os.getuid() and not state_dir.stat().st_mode & 0o077,
          "private_state_directory_required")
    task_id, cid, attempt = request["task_id"], request["task_cid"], request["attempt"]
    _need(type(attempt) is int and attempt > 0, "invalid_ordinal")
    state = native.PortalTaskState.load(daemon.state_path)
    _need(_state_hash(daemon) == request["state_sha256"], "state_preimage_changed")
    _need(not any((state.implementation_in_progress, state.active_task_id, state.active_task_cid,
          state.active_attempt, state.active_worktree_path, state.active_branch,
          state.active_task_key, state.active_phase, state.active_phase_started_at,
          state.active_log_path, state.active_task_started_at,
          getattr(state, "active_provider_runner", {}))), "lane_not_quiescent")
    _need(state.implementation_attempts.get(task_id) == state.implementation_attempts_by_cid.get(cid) == attempt,
          "consumed_ordinal_changed")
    _need(state.last_implementation_task_id == task_id and state.last_implementation_task_cid == cid,
          "last_attempt_identity_changed")
    task = next((t for t in daemon._load_tasks() if t.task_id == task_id and daemon._canonical_ref(t) == cid), None)
    _need(task is not None and native.normalize_status(task.status) in {"todo", "in_progress"},
          "pending_current_task_required")
    _need(not any(str(v).lower().strip() in {"1", "true", "yes"} for k, v in task.metadata.items()
          if str(k).lower().replace("_", " ").strip() in {"manual hold", "hold"}), "manual_hold_present")
    identity = daemon._identity_for_task(task)
    _need(identity.canonical_task_key == request["task_key"]
          and identity.board_namespace == request["board_namespace"], "task_identity_changed")
    refs = {task_id, cid, identity.canonical_task_key}
    for field in ("waiting_task_ids", "blocked_task_ids", "external_reserved_task_ids", "completed_task_ids"):
        values = getattr(state, field)
        _need(type(values) is list and all(type(v) is str for v in values)
              and not refs.intersection(values), "independent_native_hold")
    _need(type(state.task_statuses) is dict and all(
          key not in state.task_statuses or native.normalize_status(state.task_statuses[key]) in {"todo", "in_progress", "ready"}
          for key in refs), "independent_task_status_hold")
    protected = state.protected_implementation_attempts
    _need(type(protected) is dict and not any(key in refs or type(v) is not dict
          or v.get("task_id") in refs or v.get("task_revision_cid") in refs
          for key, v in protected.items()), "protected_attempt_hold")
    _need(daemon._retained_provider_capacity_deferral() is None, "capacity_hold_present")
    _need(not os.path.lexists(daemon._protected_path_maintenance_lock_path()), "protected_maintenance_hold")
    _need(daemon.max_task_attempts <= 0 or attempt < daemon.max_task_attempts, "retry_budget_exhausted")
    lock_path = daemon._implementation_lock_path()
    if os.path.lexists(lock_path):
        lease = _object(_read(lock_path))
        lease_birth = native.ProcessBirthIdentity.from_dict(lease.get("owner_process_birth"))
        _need(lease.get("kind") == "implementation" and lease_birth.pid == lease.get("pid")
              and lease_birth.start_time_ticks > 0
              and native.owner_liveness(lease_birth, proc_root=daemon.worktree_lifecycle.proc_root)
                  is native.OwnerLiveness.DEAD, "implementation_lease_live_or_unknown")
    pins = request["pins"]
    _need(set(pins) == {"claim", "lifecycle", "strategy", "incident", "launcher", "shutdown", "board", "events", "supervisor_events", "log"},
          "pin_set_changed")
    values = {}
    for name, pin in pins.items():
        _need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}, "pin_shape_changed")
        path = Path(pin["path"])
        if name == "claim" and not path.exists():
            _need(journal is not None and journal.get("phase") in {"prepared", "released"}, "claim_missing_without_journal")
            raw = base64.b64decode(journal["claim_raw_base64"], validate=True)
            _need(len(raw) == pin["bytes"] and _sha(raw) == pin["sha256"]
                  and _object(raw) == journal["claim"], "saved_claim_changed")
            values[name] = _object(raw)
            continue
        raw = _read(path)
        if name == "strategy" and journal is not None and _sha(raw) == journal.get("after_strategy_sha256"):
            before_raw = base64.b64decode(journal["strategy_raw_base64"], validate=True)
            _need(len(before_raw) == pin["bytes"] and _sha(before_raw) == pin["sha256"]
                  and _object(before_raw) == journal["before_strategy"], "saved_strategy_changed")
            expected_after = _after_strategy(_object(before_raw), request, _journal_path(state_dir, request["incident_cid"]))
            _need(_object(raw) == expected_after and _sha(raw) == _sha(_pretty(expected_after)), "strategy_cas_changed")
            values[name] = _object(before_raw)
            continue
        prefix = raw[:pin["bytes"]] if name in {"events", "supervisor_events"} else raw
        _need(len(prefix) == pin["bytes"] and _sha(prefix) == pin["sha256"], "pinned_" + name + "_changed")
        values[name] = _event_rows(raw) if name in {"events", "supervisor_events"} else (
            _object(raw) if name in {"claim", "lifecycle", "strategy", "incident", "launcher", "shutdown"} else raw)
    workspace = Path(request["workspace"])
    expected_claim = daemon._implementation_task_claim_path(task_id, canonical_task_cid=cid)
    _need(expected_claim.is_absolute() and expected_claim.resolve() == expected_claim, "aliased_claim_path")
    _need(Path(pins["claim"]["path"]) == expected_claim
          and Path(pins["lifecycle"]["path"]) == daemon.worktree_lifecycle.workspace_path_for(workspace).resolve()
          and Path(pins["strategy"]["path"]) == daemon.strategy_path.resolve()
          and Path(pins["board"]["path"]) == daemon.todo_path.resolve()
          and Path(pins["events"]["path"]) == daemon.events_path.resolve(), "foreign_authority_path")
    prefix = daemon.state_path.name.removesuffix("_task_state.json")
    _need(all(Path(pins[name]["path"]) == state_dir / filename for name, filename in (
        ("launcher", "launcher.json"), ("shutdown", "container_shutdown.json"),
        ("supervisor_events", f"{prefix}_supervisor_events.jsonl"))), "foreign_lane_evidence")
    _need(Path(pins["incident"]["path"]) == state_dir / f"{prefix}_autonomous_unstall" / "autonomous-unstall-state.json",
          "foreign_incident_state")
    starts = [e for e in values["events"] if e.get("type") == "implementation_started" and e.get("canonical_task_cid") == cid]
    _need(starts and starts[-1].get("event_id") == request["start_event_id"]
          and starts[-1].get("attempt") == attempt and starts[-1].get("worktree_path") == str(workspace)
          and starts[-1].get("baseline_ref") == request["baseline_ref"]
          and starts[-1].get("log_path") == pins["log"]["path"], "newer_or_different_attempt")
    _need(not any(e.get("type") == "implementation_finished" and e.get("canonical_task_cid") == cid
          and e.get("attempt") == attempt for e in values["events"]), "attempt_already_finished")
    claim, life, launcher, shutdown = (values[k] for k in ("claim", "lifecycle", "launcher", "shutdown"))
    _need(claim.get("kind") == native.IMPLEMENTATION_TASK_CLAIM_LOCK_KIND
          and claim.get("task_id") == task_id and claim.get("canonical_task_cid") == cid
          and claim.get("canonical_task_key") == request["task_key"]
          and claim.get("board_namespace") == request["board_namespace"]
          and claim.get("lease_role") == native.IMPLEMENTATION_DISPATCH_INTENT_LEASE_ROLE
          and type(claim.get("lease_id")) is str and bool(claim["lease_id"])
          and claim.get("task_shard_count") == daemon.task_shard_count
          and claim.get("task_shard_index") == daemon.task_shard_index
          and claim.get("attempt") == attempt and claim.get("state_path") == str(daemon.state_path.resolve())
          and claim.get("state_dir") == str(state_dir) and claim.get("repository_id") == daemon.merge_target_repository_id
          and claim.get("dispatch_authority_cid") == daemon._implementation_dispatch_authority_cid(claim), "claim_identity_changed")
    _need(not native.implementation_task_claim_protected_fence_paths(claim)
          and not daemon._implementation_protected_active_snapshot_path().exists(), "protected_hold_present")
    _need(life.get("state") == "terminal" and life.get("task_id") == task_id
          and life.get("canonical_task_cid") == cid and life.get("attempt") == daemon._worktree_lifecycle_attempt_for_identity(
              task_id=task_id, canonical_task_cid=cid, portal_attempt=attempt)
          and life.get("workspace_path") == str(workspace) and life.get("repo_root") == str(daemon.repo_root.resolve())
          and life.get("state_dir") == str(state_dir) and life.get("branch") == starts[-1].get("branch"),
          "exact_terminal_lifecycle_required")
    _need(daemon.worktree_lifecycle.load_quarantine(workspace) is None
          and not daemon.worktree_lifecycle.quarantines_for_task_attempt(
              canonical_task_cid=cid, task_id=task_id, attempt=life["attempt"]), "lifecycle_quarantine_present")
    loaded_life = daemon.worktree_lifecycle.load_workspace(workspace)
    _need(loaded_life is not None and loaded_life.to_dict() == life, "lifecycle_record_changed")
    birth = launcher.get("identity", {})
    _need(launcher.get("status") == "stopped" and launcher.get("repo_root") == str(daemon.repo_root.resolve())
          and launcher.get("state_dir") == str(state_dir) and isinstance(launcher.get("owner_token"), str)
          and launcher["owner_token"] and type(birth.get("pid")) is int and birth["pid"] > 0
          and type(birth.get("start_ticks")) is int and birth["start_ticks"] > 0 and birth.get("boot_id"),
          "stopped_launcher_identity_required")
    launch_owner = native.ProcessBirthIdentity(pid=birth["pid"], start_time_ticks=birth["start_ticks"], boot_id=birth["boot_id"])
    owners = [launch_owner, native.ProcessBirthIdentity.from_dict(claim.get("owner_process_birth")),
              native.ProcessBirthIdentity.from_dict(life.get("owner"))]
    for owner in owners:
        _need(owner.pid > 0 and owner.start_time_ticks > 0 and owner.boot_id == launch_owner.boot_id
              and native.owner_liveness(owner, proc_root=daemon.worktree_lifecycle.proc_root) is native.OwnerLiveness.DEAD,
              "owner_live_or_unknown")
        _need((owner.pid == launch_owner.pid and owner.start_time_ticks == launch_owner.start_time_ticks)
              or owner.parent_pid == launch_owner.pid and owner.start_time_ticks >= launch_owner.start_time_ticks,
              "owner_not_in_stopped_launch")
    _need(claim.get("pid") == owners[1].pid, "claim_birth_mismatch")
    _need(shutdown.get("status") == "stopped" and shutdown.get("quiesced") is True
          and shutdown.get("reason") == "owned_container_closure_verified"
          and shutdown.get("repo_root") == str(daemon.repo_root.resolve()) and shutdown.get("state_dir") == str(state_dir)
          and shutdown.get("plan_id") == launcher.get("plan_id") and launcher.get("plan_id")
          and shutdown.get("started_at") == launcher.get("started_at")
          and shutdown.get("owner_token_sha256") == _sha(launcher["owner_token"].encode())
          and all(shutdown.get(k) is False for k in ("attempt_refunded", "completion_authority", "provider_callback_settled")),
          "bound_owned_container_closure_required")
    _need(not daemon._implementation_process_active(starts[-1]), "runner_container_or_recent_log_active")
    _fresh_container_quiescence(daemon, workspace)
    _baseline_unchanged(task, workspace, request["baseline_ref"], native)
    strategy, incident = values["strategy"], values["incident"]
    record = incident.get("incidents", {}).get(request["incident_cid"])
    _need(type(record) is dict and record.get("phase") == "quarantined"
          and record.get("incident_cid") == request["incident_cid"]
          and record.get("completion_authority") is False, "original_incident_not_quarantined")
    quarantines = strategy.get("autonomous_unstall_quarantines", [])
    affected = [q for q in quarantines if isinstance(q, dict) and q.get("task_id") == task_id]
    _need(len(affected) == 1 and affected[0].get("incident_cid") == request["incident_cid"], "independent_quarantine_present")
    scope = affected[0]
    _need(scope.get("reason") == record.get("reason") and scope.get("target_ids") == record.get("target_ids"),
          "quarantine_scope_changed")
    proof = [e for e in values["supervisor_events"] if e.get("type") == "autonomous_unstall_scope_quarantined"
             and e.get("incident_cid") == request["incident_cid"] and e.get("task_id") == task_id]
    _need(len(proof) == 1 and proof[0].get("reason") == scope.get("reason")
          and proof[0].get("target_ids") == scope.get("target_ids"), "generated_block_provenance_missing")
    rewrites = [e for e in values["supervisor_events"] if e.get("type") == "strategy_rewrite"
                and e.get("active_task_id") == task_id and e["sequence"] < proof[0]["sequence"]]
    _need(bool(rewrites) and all(e.get("blocked_active_task") is False for e in rewrites)
          and strategy.get("last_rewrite_reason") == "autonomous unstall quarantine: " + scope["reason"],
          "independent_strategy_rewrite_or_unproved_origin")
    _need(all(type(strategy.get(k)) is list and strategy[k].count(task_id) == 1 for k in ("blocked_tasks", "deprioritized_tasks")),
          "generated_strategy_block_not_exact")
    _need(not any(strategy.get(k) for k in ("manual_holds", "manual_hold_tasks", "blocked_task_reasons", "independent_blockers")),
          "independent_or_manual_blocker_present")
    return {"claim": claim, "before_strategy": strategy,
            "claim_raw_base64": (journal["claim_raw_base64"] if journal is not None
                else base64.b64encode(_read(expected_claim)).decode("ascii")),
            "strategy_raw_base64": (journal["strategy_raw_base64"] if journal is not None
                else base64.b64encode(_read(daemon.strategy_path)).decode("ascii")), "quarantine": scope,
            "incident_record": record, "owner_birth": birth, "state": state}


def _pretty(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _after_strategy(before: dict, request: dict, path: Path) -> dict:
    after = deepcopy(before)
    for key in ("blocked_tasks", "deprioritized_tasks"):
        after[key] = [v for v in after[key] if v != request["task_id"]]
    after.setdefault("ordinary_incident_adjudications", {})[request["incident_cid"]] = {
        "operation_id": "sha256:" + _sha(_bytes(request)), "receipt_path": str(path),
        "task_id": request["task_id"], "task_cid": request["task_cid"], "attempt": request["attempt"]}
    return after


def _fresh_container_quiescence(daemon, workspace: Path) -> None:
    # Inspect mount metadata only; Docker Config.Env and command arguments may
    # contain credentials. Unrelated campaigns are outside this operation.
    from ..runtime.grok_cli_runner import _docker_isolation_binary
    binary = _docker_isolation_binary()
    _need(bool(binary), "trusted_docker_inventory_unavailable")
    deadline = time.monotonic() + 15.0
    result = subprocess.run([binary, "ps", "--format", "{{.ID}}"],
                            capture_output=True, text=True, timeout=15, check=False)
    identifiers = result.stdout.splitlines()
    _need(result.returncode == 0 and len(identifiers) <= 16
          and all(v and all(c in "0123456789abcdef" for c in v) for v in identifiers),
          "container_inventory_unknown")
    roots = (workspace.resolve(), daemon.state_path.parent.resolve())
    for identifier in identifiers:
        remaining = deadline - time.monotonic()
        _need(remaining > 0, "container_inventory_deadline")
        inspected = subprocess.run([binary, "inspect", "--format", "{{json .Mounts}}", identifier],
                                   capture_output=True, text=True, timeout=remaining, check=False)
        _need(inspected.returncode == 0, "container_inventory_unknown")
        mounts = json.loads(inspected.stdout)
        _need(type(mounts) is list and all(type(v) is dict for v in mounts), "container_inventory_unknown")
        for mount in mounts:
            source = mount.get("Source")
            _need(type(source) is str and source.startswith("/"), "container_inventory_unknown")
            candidate = Path(source).resolve()
            _need(not any(candidate == root or root in candidate.parents or candidate in root.parents
                          for root in roots), "owned_container_still_running")


def _baseline_unchanged(task, workspace: Path, baseline_ref: str, native) -> None:
    def git(*args):
        result = subprocess.run(["git", "--no-replace-objects", *args], cwd=workspace,
                                capture_output=True, timeout=15, check=False)
        _need(result.returncode == 0, "baseline_git_read_failed")
        return result.stdout
    commit = git("rev-parse", "--verify", baseline_ref + "^{commit}").decode().strip()
    _need(commit == baseline_ref and git("rev-parse", "HEAD").decode().strip() == commit,
          "baseline_commit_changed")
    git("cat-file", "-e", commit + "^{tree}")
    outputs = native.task_declared_output_paths(task)
    _need(bool(outputs), "declared_outputs_required")
    for target in outputs:
        relative = Path(target)
        _need(not relative.is_absolute() and ".." not in relative.parts, "invalid_output_path")
        file = workspace / relative
        entry = git("ls-tree", "-z", commit, "--", target)
        if not entry:
            _need(not os.path.lexists(file), "declared_baseline_changed")
        else:
            entries = entry.split(b"\0")
            _need(len(entries) == 2 and entries[-1] == b"", "baseline_tree_shape_changed")
            metadata, name = entries[0].split(b"\t", 1)
            mode, kind, oid = metadata.split(b" ")
            _need(kind == b"blob" and mode in {b"100644", b"100755"}
                  and name.decode() == target, "baseline_entry_not_regular_blob")
            _need(file.is_file() and not file.is_symlink()
                  and _read(file) == git("cat-file", "blob", oid.decode()), "declared_baseline_changed")


def adjudicate(daemon, *, request: Mapping[str, Any]) -> dict:
    """Retire one exact dead pending claim; no provider/callback/status authority.

    The caller must hold its application's stopped-launcher/runner guards for
    the complete call. Native implementation/claim/strategy guards below stop
    dispatch publication; they cannot own a separate application's launcher.
    """
    from . import implementation_daemon as native
    request = deepcopy(dict(request))
    result = {"reconciled": False, "blocked": True, "provider_outcome": "unknown", **_FALSE}
    try:
        _need(type(daemon) is native.PortalImplementationDaemon, "real_native_portal_required")
        _need(daemon.state_path.is_absolute() and daemon.state_path.resolve() == daemon.state_path, "aliased_state_path")
        path = _journal_path(daemon.state_path.parent, str(request.get("incident_cid", "")))
        directory = path.parent
        if not directory.exists():
            directory.mkdir(mode=0o700)
        info = directory.lstat()
        _need(stat.S_ISDIR(info.st_mode) and directory.resolve() == directory and info.st_uid == os.getuid()
              and not info.st_mode & 0o077, "unsafe_adjudication_directory")
        claim_path = daemon._implementation_task_claim_path(
            request["task_id"], canonical_task_cid=request["task_cid"])
        _need(claim_path.is_absolute() and claim_path.resolve() == claim_path, "aliased_claim_path")
        with ExitStack() as stack:
            for lock in (daemon._implementation_lock_path(), directory / "adjudication.lock", claim_path, daemon.strategy_path):
                stack.enter_context(native.serialized_lock_update(lock))
            prior = _object(_read(path)) if os.path.lexists(path) else None
            operation_id = "sha256:" + _sha(_bytes(request))
            if prior is not None:
                _need(prior.get("operation_id") == operation_id and prior.get("request") == request
                      and prior.get("receipt_id") == "sha256:" + _sha(_bytes({k:v for k,v in prior.items() if k != "receipt_id"})),
                      "journal_corrupt_or_different_operation")
            checked = _validated(daemon, request, prior)
            before = checked["before_strategy"]
            after = _after_strategy(before, request, path)
            after_raw = _pretty(after)
            body = {"schema": SCHEMA, "operation_id": operation_id, "request": request,
                    "claim": checked["claim"], "claim_raw_base64": checked["claim_raw_base64"],
                    "strategy_raw_base64": checked["strategy_raw_base64"],
                    "before_strategy": before, "after_strategy_sha256": _sha(after_raw),
                    "quarantine": checked["quarantine"], "incident_record": checked["incident_record"],
                    "phase": "prepared", "provider_outcome": "unknown", **_FALSE}
            def publish(value):
                value = {**value, "receipt_id": "sha256:" + _sha(_bytes(value))}
                _need(len(_pretty(value)) <= MAX_BYTES, "journal_bound_exceeded")
                daemon._write_private_task_claim_release_receipt(path, value)
                return value
            if prior is None:
                prior = publish(body)
            if prior.get("phase") == "released":
                _need(not os.path.lexists(claim_path), "released_claim_reappeared")
                return {**result, "reconciled": True, "blocked": False, "reason": "incident_already_adjudicated",
                        "receipt_path": str(path), "receipt_id": prior["receipt_id"]}
            _need(prior.get("phase") == "prepared", "incomplete_journal_phase")
            _validated(daemon, request, prior)
            if claim_path.exists():
                claim_path.unlink()
                fd = os.open(claim_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            _validated(daemon, request, prior)
            daemon._write_private_task_claim_release_receipt(daemon.strategy_path, after)
            _validated(daemon, request, prior)
            released = publish({**body, "phase": "released"})
            daemon._record_event("ordinary_incident_administratively_abandoned", {
                "task_id": request["task_id"], "canonical_task_cid": request["task_cid"],
                "attempt": request["attempt"], "incident_cid": request["incident_cid"],
                "receipt_id": released["receipt_id"], "provider_outcome": "unknown", **_FALSE})
            return {**result, "reconciled": True, "blocked": False, "reason": "quiesced_incident_adjudicated",
                    "receipt_path": str(path), "receipt_id": released["receipt_id"]}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        return {**result, "reason": str(exc) if type(exc) is ValueError else type(exc).__name__}


def settlement_valid(*, repo_root: Path, state_path: Path, incident_cid: str, quarantine: Mapping) -> bool:
    """Only the exact old incident is settled; later incidents retain their deny."""
    try:
        from . import implementation_daemon as native
        path = _journal_path(state_path.parent.resolve(), incident_cid)
        _need(path.parent.resolve() == path.parent and path.parent.stat().st_uid == os.getuid()
              and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
              and stat.S_IMODE(path.lstat().st_mode) == 0o600, "settlement_not_private")
        receipt = _object(_read(path))
        request = receipt["request"]
        _need(receipt.get("schema") == SCHEMA and receipt.get("phase") == "released"
              and receipt.get("receipt_id") == "sha256:" + _sha(_bytes({k:v for k,v in receipt.items() if k != "receipt_id"}))
              and receipt.get("operation_id") == "sha256:" + _sha(_bytes(request))
              and request.get("runtime_revision") == runtime_revision()
              and request.get("repo_root") == str(repo_root.resolve())
              and request.get("state_path") == str(state_path.resolve())
              and request.get("incident_cid") == incident_cid and receipt.get("quarantine") == dict(quarantine)
              and request.get("task_id") == quarantine.get("task_id")
              and type(request.get("attempt")) is int and request["attempt"] > 0
              and receipt.get("provider_outcome") == "unknown"
              and all(receipt.get(k) is False for k in _FALSE), "settlement_not_applicable")
        pins = request["pins"]
        before_raw = base64.b64decode(receipt["strategy_raw_base64"], validate=True)
        claim_raw = base64.b64decode(receipt["claim_raw_base64"], validate=True)
        _need(len(before_raw) == pins["strategy"]["bytes"] and _sha(before_raw) == pins["strategy"]["sha256"]
              and _object(before_raw) == receipt["before_strategy"]
              and len(claim_raw) == pins["claim"]["bytes"] and _sha(claim_raw) == pins["claim"]["sha256"]
              and _object(claim_raw) == receipt["claim"]
              and Path(pins["strategy"]["path"]).parent == state_path.parent.resolve(), "settlement_preimage_changed")
        current_strategy = _object(_read(Path(pins["strategy"]["path"])))
        marker = _after_strategy(receipt["before_strategy"], request, path)["ordinary_incident_adjudications"][incident_cid]
        _need(current_strategy.get("ordinary_incident_adjudications", {}).get(incident_cid) == marker,
              "settlement_strategy_marker_changed")
        claim = receipt["claim"]
        _need(claim.get("canonical_task_cid") == request["task_cid"] and claim.get("task_id") == request["task_id"]
              and claim.get("attempt") == request["attempt"] and claim.get("state_path") == str(state_path.resolve()),
              "settlement_claim_identity_changed")
        incident = _object(_read(Path(pins["incident"]["path"])))
        _need(incident.get("incidents", {}).get(incident_cid) == receipt["incident_record"], "incident_record_changed")
        board = Path(pins["board"]["path"])
        tasks = native.parse_task_text(_read(board).decode(), path=board,
                                       task_header_prefix=request["task_header_prefix"])
        matching = [t for t in tasks if t.task_id == request["task_id"]]
        _need(len(matching) == 1 and matching[0].canonical_task_cid == request["task_cid"]
              and matching[0].canonical_task_key == request["task_key"]
              and matching[0].board_namespace == request["board_namespace"], "settlement_current_task_changed")
        state = native.PortalTaskState.load(state_path)
        claim_path = Path(pins["claim"]["path"])
        if os.path.lexists(claim_path):
            current_claim = _object(_read(claim_path))
            ordinal = current_claim.get("attempt")
            _need(type(ordinal) is int and ordinal > request["attempt"]
                  and current_claim.get("task_id") == request["task_id"]
                  and current_claim.get("canonical_task_cid") == request["task_cid"]
                  and current_claim.get("canonical_task_key") == request["task_key"]
                  and current_claim.get("state_path") == request["state_path"]
                  and current_claim.get("dispatch_authority_cid") == native.PortalImplementationDaemon._implementation_dispatch_authority_cid(current_claim)
                  and current_claim.get("board_namespace") == request["board_namespace"]
                  and current_claim.get("repository_id") == claim.get("repository_id")
                  and current_claim.get("kind") == native.IMPLEMENTATION_TASK_CLAIM_LOCK_KIND
                  and current_claim.get("lease_role") == native.IMPLEMENTATION_DISPATCH_INTENT_LEASE_ROLE
                  and state.implementation_attempts.get(request["task_id"]) == state.implementation_attempts_by_cid.get(request["task_cid"])
                  and ordinal in {state.implementation_attempts.get(request["task_id"]),
                                  state.implementation_attempts.get(request["task_id"], 0) + 1},
                  "new_claim_not_a_distinct_later_attempt")
        _need(state.implementation_attempts.get(request["task_id"], 0) >= request["attempt"]
              and state.implementation_attempts_by_cid.get(request["task_cid"], 0) >= request["attempt"],
              "settlement_accounting_changed")
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def effective_strategy(*, repo_root: Path, state_path: Path, strategy: Mapping) -> dict:
    """A missing final settlement can only add a block, never remove one."""
    projected = deepcopy(dict(strategy))
    for q in projected.get("autonomous_unstall_quarantines", ()):
        if not isinstance(q, Mapping) or not q.get("task_id"):
            continue
        if settlement_valid(repo_root=repo_root, state_path=state_path,
                            incident_cid=str(q.get("incident_cid", "")), quarantine=q):
            continue
        for key in ("blocked_tasks", "deprioritized_tasks"):
            if q["task_id"] not in projected.setdefault(key, []):
                projected[key].append(q["task_id"])
    return projected
