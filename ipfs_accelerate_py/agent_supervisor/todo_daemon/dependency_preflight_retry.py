"""Reconcile a finished Native dependency refusal without replaying unknown work.

Only the sealed projection, Native state and terminal workspace lifecycle can
prove this narrow pre-provider boundary. Generic exceptions and provider JSON
never grant retry rights. A retry keeps the original database attempt, deadline
and paid history, and enters dispatch intent again before calling Native.
"""
from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

REASON = "validation_project_dependency_preflight_failed"
SCHEMA = "database-native-dependency-wait@1"
MAX_RETRIES = 8
IDENTITY = ("attempt_id", "task_cid", "owner_session_id", "claim_id", "lease_id", "fencing_token", "fence_epoch")


def observe(bridge, attempt):
    """Read an exact, fully cleaned Native no-dispatch refusal, or return None."""
    from .database_portal_bridge import DatabasePortalExecutionBridge, _accepted_source_events, _sha256_bytes
    from .implementation_daemon import PortalImplementationDaemon
    from ..runtime.event_log import _event_identity

    if type(bridge) is not DatabasePortalExecutionBridge:
        return None
    paths = bridge._paths(attempt)
    if not paths.binding.is_file() or not paths.events.is_file():
        return None
    daemon = None
    try:
        bridge._seal_attempt_directory(paths, attempt_id=attempt.attempt_id, create=False)
        record = bridge._record_for_attempt(bridge.task_source, attempt)
        paths, binding = bridge._ensure_attempt_projection(attempt, record)
        events, digest = _accepted_source_events(paths.events)
        # Do not even construct an ordinary unknown provider's factory.
        finished = [e for e in events if e.get("type") == "implementation_finished"]
        if not finished or any(e.get("type") in {"implementation_started", "merge_started", "merge_finished", "task_completed"}
                               for e in events):
            return None
        if any(e.get("reason") != REASON or e.get("provider_dispatched") is not False
               or e.get("attempt_consumed") is not False or e.get("provider_call_allowed") is not False
               for e in finished):
            return None
        previous = ""
        for sequence, event in enumerate(events, 1):
            if (event.get("sequence") != sequence or event.get("previous_event_id") != previous
                    or event.get("event_id") != _event_identity(event)):
                return None
            previous = event["event_id"]
        last = finished[-1]
        local_attempt = last.get("attempt")
        if (type(local_attempt) is not int or not 1 <= local_attempt < 65536
                or last.get("task_id") != binding["task_alias"]
                or last.get("deferred") is not True
                or last.get("lifecycle_finalize", {}).get("finalized") is not True
                or last.get("cleanup_result", {}).get("removed_worktree") is not True
                or last.get("implementation_commit") not in (None, "")
                or last.get("merge_result", {}).get("merged") is not False
                or last.get("merge_result", {}).get("queued", False) is not False):
            return None
        gate = [e for e in events if e.get("type") == "validation_project_dependency_preflight_failed"
                and e.get("task_id") == last["task_id"] and e.get("attempt") == local_attempt]
        if (not gate or gate[-1].get("dependency_preflight", {}).get("passed") is not False
                or gate[-1].get("sequence", 0) >= last["sequence"]):
            return None
        daemon = bridge.portal_factory(paths, binding["task_alias"])
        if (type(daemon) is not PortalImplementationDaemon
                or daemon.state_path.resolve() != paths.state.resolve()
                or daemon.events_path.resolve() != paths.events.resolve()):
            return None
        authority = daemon.bind_database_attempt_authority(task_id=binding["task_alias"],
            database_task_cid=attempt.task_cid, database_attempt_id=attempt.attempt_id,
            database_claim_id=attempt.claim_id, database_attempt_number=attempt.attempt_number,
            database_binding_id=binding["binding_id"])
        if any(e.get("canonical_task_cid") not in (None, "", authority["canonical_task_cid"]) for e in events):
            return None
        workspace = Path(last["worktree_path"])
        if (not workspace.is_absolute() or workspace.is_symlink() or workspace.exists()
                or not workspace.is_relative_to(daemon.worktree_root.resolve())):
            return None
        lifecycle = daemon.worktree_lifecycle.load_workspace(workspace)
        if lifecycle is None:
            # Successful Native pool cleanup compare-deletes its terminal
            # lifecycle. Join that durable cleanup event instead of inventing
            # a record or treating a missing directory as non-dispatch proof.
            cleaned = [e for e in events if e.get("type") == "cleanup_finished"
                and e.get("worktree_path") == str(workspace) and e.get("branch") == last.get("branch")
                and gate[-1]["sequence"] < e["sequence"] < last["sequence"]]
            if (not cleaned or cleaned[-1].get("cleaned") is not True
                    or cleaned[-1].get("removed_worktree") is not True
                    or cleaned[-1].get("deleted_branch") is not True
                    or cleaned[-1].get("lifecycle_finalize", {}).get("finalized") is not True
                    or cleaned[-1]["lifecycle_finalize"].get("fence") != last["lifecycle_finalize"].get("fence")
                    or daemon.worktree_lifecycle.load_task_attempt(canonical_task_cid=authority["canonical_task_cid"],
                        task_id=binding["task_alias"], attempt=authority["worktree_lifecycle_attempt_prefix"] + local_attempt) is not None):
                return None
            branch = subprocess.run(['git', 'show-ref', '--verify', '--quiet', 'refs/heads/' + last['branch']],
                cwd=daemon.repo_root, capture_output=True, timeout=30)
            if branch.returncode != 1:
                return None
            lifecycle_id = cleaned[-1]["event_id"]
        elif (not lifecycle.is_terminal
                or lifecycle.task_id != binding["task_alias"]
                or lifecycle.canonical_task_cid != authority["canonical_task_cid"]
                or lifecycle.attempt != authority["worktree_lifecycle_attempt_prefix"] + local_attempt
                or lifecycle.branch != last.get("branch")
                or lifecycle.state_dir != str(paths.root.resolve())
                or lifecycle.repo_root != str(daemon.repo_root.resolve())):
            return None
        else:
            lifecycle_id = lifecycle.record_id
        state_bytes, target = bridge._native_provider_state(paths)
        state = json.loads(state_bytes)
        if (state.get("implementation_in_progress") is not False
                or state.get("active_provider_runner") != {} or state.get("active_attempt") != 0
                or any(state.get(k) not in (None, "") for k in
                       ("active_task_id", "active_task_cid", "active_branch", "active_worktree_path", "active_log_path"))
                or state.get("last_implementation_branch") != last.get("branch")
                or state.get("last_implementation_worktree_path") != str(workspace)
                or state.get("last_implementation_commit") not in (None, "")
                or state.get("last_implementation_returncode") != last.get("returncode")):
            return None
        backoff = last.get("backoff_seconds")
        timeout = daemon.implementation_timeout
        if (type(backoff) is not int or not 1 <= backoff <= 1800
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            return None
        finished_ms = int(datetime.fromisoformat(last["timestamp"]).timestamp() * 1000)
        proof = dict(binding_id=binding["binding_id"], finished_event_id=last["event_id"],
            events_digest=digest, state_digest=_sha256_bytes(state_bytes), state_target_digest=target,
            canonical_task_cid=authority["canonical_task_cid"], lifecycle_record_id=lifecycle_id,
            retry_not_before_ms=finished_ms + backoff * 1000,
            retry_deadline_ms=attempt.started_at_ms + int(timeout * 1000),
            provider_dispatched=False, attempt_consumed=False, completion_authority=False)
        # Reject any concurrent mutation between Native observation and return.
        if (_accepted_source_events(paths.events)[1] != digest
                or bridge._native_provider_state(paths) != (state_bytes, target)):
            return None
        return proof
    except Exception:
        return None
    finally:
        if type(daemon) is PortalImplementationDaemon:
            daemon.close_event_runtime()


def settle(owner, attempt, bridge, prior, key, *, previous=None):
    """CAS only an actual unknown intent to a bounded, proven no-dispatch wait."""
    from .implementation_daemon import _database_daemon_json, content_identity

    proof = observe(bridge, attempt)
    if proof is None:
        return False
    owner._protect_attempt_write(attempt)
    owner._require_provider_admission(attempt)
    expected = dict(schema="database-portal-callback-intent@1", callback_state="started_outcome_unknown",
        provider_effect_state="unknown_may_have_started", **{k: getattr(attempt, k) for k in IDENTITY if k != "owner_session_id"})
    if (key != f"provider:{attempt.attempt_id}" or prior != expected
            or attempt.committed_phase not in {"claimed", "context"}
            or owner.effect_claim_recorded(attempt.attempt_id, idempotency_key=f"effect:{attempt.attempt_id}") is not None):
        return False
    receipt = dict(schema=SCHEMA, **{k: getattr(attempt, k) for k in IDENTITY},
        callback_state="not_dispatched", provider_effect_state="not_started", reason=REASON,
        retry_count=(previous["retry_count"] if previous else 0) + 1,
        execution_phase=attempt.committed_phase, execution_revision=attempt.revision, native=proof)
    if previous:
        receipt["native"]["retry_deadline_ms"] = min(proof["retry_deadline_ms"], previous["native"]["retry_deadline_ms"])
    receipt["receipt_id"] = content_identity(receipt)
    changed = owner._require_connection().execute(
        "UPDATE provider_invocations SET result_json = ?, recorded_at_ms = ? WHERE attempt_id = ? "
        "AND idempotency_key = ? AND owner_session_id = ? AND task_cid = ? AND result_json = ? RETURNING invocation_id",
        [_database_daemon_json(receipt), owner._now_ms(), attempt.attempt_id, key,
         owner.owner_session_id, attempt.task_cid, _database_daemon_json(dict(prior))]).fetchone()
    if changed is None:
        return False
    owner._record_event("native_dependency_no_dispatch_observed", attempt_id=attempt.attempt_id,
        task_cid=attempt.task_cid, body=receipt)
    return True


def ready(owner, attempt, receipt, key, callback):
    """Verify original custody and re-prove Native's boundary before retry."""
    from .database_portal_bridge import DatabasePortalBridgeDeferred, DatabasePortalExecutionBridge
    from .implementation_daemon import DatabaseImplementationAuthorityError, content_identity

    owner._protect_attempt_write(attempt)
    owner._require_provider_admission(attempt)
    fields = {"schema", *IDENTITY, "callback_state", "provider_effect_state", "reason", "retry_count",
              "execution_phase", "execution_revision", "native", "receipt_id"}
    if (type(receipt) is not dict or set(receipt) != fields or receipt.get("schema") != SCHEMA
            or receipt.get("receipt_id") != content_identity({k: v for k, v in receipt.items() if k != "receipt_id"})
            or any(type(receipt.get(k)) is not type(getattr(attempt, k)) or receipt[k] != getattr(attempt, k) for k in IDENTITY)
            or receipt.get("callback_state") != "not_dispatched" or receipt.get("provider_effect_state") != "not_started"
            or receipt.get("reason") != REASON or key != f"provider:{attempt.attempt_id}"
            or type(receipt.get("retry_count")) is not int or not 1 <= receipt["retry_count"] <= MAX_RETRIES
            or receipt.get("execution_phase") != attempt.committed_phase
            or receipt.get("execution_revision") != attempt.revision
            or attempt.committed_phase not in {"claimed", "context"}
            or owner.effect_claim_recorded(attempt.attempt_id, idempotency_key=f"effect:{attempt.attempt_id}") is not None):
        raise DatabaseImplementationAuthorityError("native dependency retry custody differs")
    native = receipt["native"]
    native_fields = {"binding_id", "finished_event_id", "events_digest", "state_digest", "state_target_digest",
        "canonical_task_cid", "lifecycle_record_id", "retry_not_before_ms", "retry_deadline_ms",
        "provider_dispatched", "attempt_consumed", "completion_authority"}
    if (type(native) is not dict or set(native) != native_fields
            or any(native.get(k) is not False for k in ("provider_dispatched", "attempt_consumed", "completion_authority"))
            or any(type(native.get(k)) is not int or native[k] <= 0 for k in ("retry_not_before_ms", "retry_deadline_ms"))
            or any(type(native.get(k)) is not str or not native[k] for k in native_fields - {
                "provider_dispatched", "attempt_consumed", "completion_authority", "retry_not_before_ms", "retry_deadline_ms"})):
        raise DatabaseImplementationAuthorityError("native dependency retry proof differs")
    now = owner._now_ms()
    if now >= native["retry_deadline_ms"] or receipt["retry_count"] >= MAX_RETRIES:
        raise DatabasePortalBridgeDeferred("validation_dependency_repair_required", backoff_seconds=1800,
            result={"dependency_wait": receipt, "original_deadline_or_retry_bound_reached": True})
    if now < native["retry_not_before_ms"]:
        raise DatabasePortalBridgeDeferred(REASON, backoff_seconds=min(1800, max(1, math.ceil((native["retry_not_before_ms"] - now) / 1000))),
            result={"dependency_wait": receipt})
    official = owner._resolve_native_portal_provider_callback(callback, attempt)
    bridge = getattr(official, "__self__", None)
    if type(bridge) is not DatabasePortalExecutionBridge or getattr(official, "__func__", None) is not DatabasePortalExecutionBridge.run_provider:
        raise DatabaseImplementationAuthorityError("native dependency retry requires its official Portal bridge")
    proof = observe(bridge, attempt)
    if proof is None or any(proof[k] != native[k] for k in ("binding_id", "finished_event_id", "canonical_task_cid", "lifecycle_record_id")):
        raise DatabaseImplementationAuthorityError("native dependency refusal changed before retry")
