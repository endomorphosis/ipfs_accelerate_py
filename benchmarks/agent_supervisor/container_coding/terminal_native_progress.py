"""Bounded, body-free observations for the single-task benchmark lifecycle.

These observations can stop this disposable run; they cannot settle a callback,
release a claim, retry a provider, or make a task complete. Only canonical task
status retains completion authority. In particular, an idle heartbeat alone is
not evidence of either failure or provider non-dispatch.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import stat

HEARTBEAT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/database-daemon-pass-heartbeat@1"
MAX_HEARTBEAT_BYTES = 65536
MAX_TRANSITIONS = 32
POST_START_BOOTSTRAP_ERROR_LIMIT = 3
TERMINAL_STATUSES = frozenset({"completed", "failed", "blocked", "cancelled"})
TASK_STATUSES = TERMINAL_STATUSES | {"ready", "pending", "in_progress", "retrying"}
IDLE_REASONS = frozenset({
    "", "no_ready_tasks", "no_shard_selectable_ready_tasks",
    "no_eligible_ready_tasks_after_selection_filters",
    "all_selectable_ready_tasks_deprioritized_as_off_mission",
    "all_selectable_ready_tasks_reached_max_task_attempts",
    "expired_attempt_settlement_unavailable", "unsettled_portal_failure_quarantine",
    "completion_evidence_unavailable", "task_fence_evidence_unavailable",
    "claim_withdrawn_control_not_dispatchable", "provider_capacity_backoff",
    "native_dispatch_observation_unavailable",
})
# Both indicate that automatic work is exhausted/held. Two advancing passes of
# the same process generation and unchanged task revision are required. Other
# deferrals may recover on a subsequent ordinary reconciliation pass.
QUIESCENT_STOP_REASONS = frozenset({
    "all_selectable_ready_tasks_reached_max_task_attempts",
    "unsettled_portal_failure_quarantine",
})
RECEIPT_OPERATIONS = frozenset({
    "database_claim", "database_task_claim_failure", "database_portal_retry",
    "database_portal_retry_budget_exhausted", "database_task_completion",
})
REJECTION_REASONS = frozenset({
    "proposal_gate_failed", "proposal_validation_failed",
    "no_change_completion_not_allowed", "incomplete_expected_outputs",
    "expected_output_ignored_or_unstaged", "empty_or_no_change",
    "empty_patch_reserved_for_no_change_gate", "no_changes",
    "declared_validation_failed", "validation_command_failed",
    "worktree_lifecycle_claim_exists", "terminal_portal_bridge_error",
})


def _number(value, *, maximum=2**63 - 1):
    return type(value) is int and 0 <= value <= maximum


def _identity(value):
    # Hash opaque identifiers: a malicious code-shaped value is not exported.
    if type(value) is str and 0 < len(value.encode()) <= 1024:
        return hashlib.sha256(value.encode()).hexdigest()
    return None


def _closed(value, allowed):
    return value if type(value) is str and value in allowed else "unrecognized"


def _bootstrap_counts(value):
    # The driver supplies the native runtime's closed diagnostic projection.
    # Counts observe bootstrap attempts, not their cause or task correctness.
    if (type(value) is dict
            and value.get("schema") == "admitted-native-startup-observation@1"
            # Saturated counters cannot establish absence of later progress.
            and all(_number(value.get(key), maximum=65534) for key in
                    ("bootstrap_receipt_count", "bootstrap_error_count"))
            and value["bootstrap_receipt_count"] > 0):
        return value["bootstrap_receipt_count"], value["bootstrap_error_count"]
    return None


def project_task(task, *, task_cid: str) -> dict:
    if getattr(task, "task_cid", None) != task_cid:
        return {"availability": "task_identity_mismatch"}
    revision = getattr(task, "revision", None)
    if not _number(revision):
        return {"availability": "malformed_task_revision"}
    result = {"availability": "observed", "task_cid": task_cid,
              "status": _closed(getattr(task, "status", None), TASK_STATUSES), "revision": revision}
    body = getattr(task, "body", None)
    receipt = body.get("completion_receipt") if isinstance(body, Mapping) else None
    if not isinstance(receipt, Mapping):
        return result
    projected = {"operation": _closed(receipt.get("operation"), RECEIPT_OPERATIONS)}
    for key in ("attempt_id", "claim_id", "lease_id", "owner_session_id", "settlement_id",
                "failure_payload_digest", "execution_route_policy_id"):
        digest = _identity(receipt.get(key))
        if digest is not None:
            projected[key + "_sha256"] = digest
    for key in ("attempt_number", "fencing_token", "fence_epoch", "execution_revision",
                "control_expected_revision", "max_task_attempts", "provider_invocation_count",
                "effect_claim_count", "execution_finished_at_ms"):
        if _number(receipt.get(key)):
            projected[key] = receipt[key]
    for key in ("automatic_retry_admitted", "attempt_consumed", "provider_dispatched"):
        if type(receipt.get(key)) is bool:
            projected[key] = receipt[key]
    for key in ("reason", "failure_kind"):
        if key in receipt:
            projected[key] = _closed(receipt[key], REJECTION_REASONS)
    codes = receipt.get("finding_codes", receipt.get("reason_codes"))
    if type(codes) is list and len(codes) <= 16:
        # Use the native closed taxonomy. No proposal/diagnostic body is read.
        from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import ProposalFindingCode
        allowed = {item.value for item in ProposalFindingCode} | {"validation_channel_tampering_forbidden"}
        projected["finding_codes"] = sorted({code for code in codes if type(code) is str and code in allowed})
        projected["unrecognized_finding_count"] = sum(type(code) is not str or code not in allowed for code in codes)
    result["completion_receipt"] = projected
    return result


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate heartbeat field")
        result[key] = value
    return result


def read_heartbeat(state: Path) -> dict:
    """Read one fixed owner-written advisory file, without logs or discovery."""
    path = state / "run/admitted_database_daemon_pass_heartbeat.json"
    try:
        if path.resolve(strict=True) != path:
            return {"availability": "unavailable"}
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_HEARTBEAT_BYTES:
                return {"availability": "unavailable"}
            raw = stream.read(MAX_HEARTBEAT_BYTES + 1)
            after = os.fstat(stream.fileno())
        if (len(raw) > MAX_HEARTBEAT_BYTES
                or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            return {"availability": "unavailable"}
        value = json.loads(raw, object_pairs_hook=_strict_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite heartbeat")))
        if (type(value) is not dict or value.get("schema") != HEARTBEAT_SCHEMA
                or not _number(value.get("sequence")) or value["sequence"] < 1
                or type(value.get("process_birth")) is not dict
                or not _number(value.get("write_count"))
                or type(value.get("unchanged")) is not bool
                or any(type(value.get(name)) is not str or len(value[name]) > 1024
                       for name in ("active_task_id", "claimed_task_cid"))):
            return {"availability": "unavailable"}
        birth = value["process_birth"]
        # No path, command line or credential enters the generation identity.
        if (not _number(birth.get("pid")) or birth["pid"] < 1
                or not _number(birth.get("start_time_ticks")) or birth["start_time_ticks"] < 1
                or _identity(birth.get("boot_id")) is None
                or _identity(value.get("process_instance_id")) is None
                or _identity(value.get("owner_session_id")) is None):
            return {"availability": "unavailable"}
        generation = [birth["pid"], birth["start_time_ticks"], birth["boot_id"],
                      value["process_instance_id"], value["owner_session_id"]]
        return {"availability": "observed", "schema": HEARTBEAT_SCHEMA,
            "sequence": value["sequence"], "write_count": value["write_count"],
            "unchanged": value["unchanged"],
            "active_task_present": bool(value.get("active_task_id")),
            "claimed_task_present": bool(value.get("claimed_task_cid")),
            "selection_idle_reason": _closed(value.get("selection_idle_reason"), IDLE_REASONS),
            "generation_sha256": hashlib.sha256(json.dumps(generation).encode()).hexdigest()}
    except FileNotFoundError:
        return {"availability": "missing"}
    except (OSError, ValueError, TypeError, RecursionError):
        return {"availability": "unavailable"}


class NativeProgress:
    def __init__(self, *, started: float):
        self.started = started
        self.transitions = deque(maxlen=MAX_TRANSITIONS)
        self.previous = None
        self.quiescent = None
        self.bootstrap_enabled = False
        self.bootstrap_counts = None
        self.bootstrap_error_base = None
        self.bootstrap_anchor = None
        self.bootstrap_confirmation = None
        self.report = {"schema": "terminal-native-progress@1", "samples": 0,
            "transition_count": 0, "transitions_omitted": 0, "transitions": [],
            "completion_authority": False, "settlement_authority": False, "retry_authority": False}

    def begin_post_start(self, *, startup=None):
        """Arm only after the driver's successful native START observation."""
        self.bootstrap_enabled = True
        self._reset_bootstrap(_bootstrap_counts(startup))
        self.report["post_start_bootstrap_guard"] = {
            "new_error_threshold": POST_START_BOOTSTRAP_ERROR_LIMIT,
            "unchanged_confirmation_required": True, "root_cause_verified": False}

    def _reset_bootstrap(self, counts, anchor=None):
        self.bootstrap_counts = counts
        self.bootstrap_error_base = None if counts is None else counts[1]
        self.bootstrap_anchor = anchor
        self.bootstrap_confirmation = None

    def _sample_bootstrap(self, *, startup, task_row, heartbeat):
        if not self.bootstrap_enabled:
            return None, None
        counts = _bootstrap_counts(startup)
        if (counts is None or task_row.get("availability") != "observed"
                or task_row.get("status") not in {"ready", "pending", "in_progress", "retrying"}
                or heartbeat.get("availability") not in {"observed", "missing"}):
            self._reset_bootstrap(None)
            return {"availability": "unavailable"}, None
        # Any native task/heartbeat change conservatively starts a new window.
        # A missing heartbeat is allowed; a malformed/unreadable one abstains.
        anchor = (task_row["status"], task_row["revision"], json.dumps(heartbeat, sort_keys=True))
        previous = self.bootstrap_counts
        reset = (previous is None or counts[0] != previous[0] or counts[1] < previous[1]
                 or (self.bootstrap_anchor is not None and self.bootstrap_anchor != anchor))
        if reset:
            self._reset_bootstrap(counts, anchor)
        self.bootstrap_anchor = anchor
        self.bootstrap_counts = counts
        delta = counts[1] - self.bootstrap_error_base
        observation = {"availability": "observed", "receipt_count": counts[0],
            "error_count": counts[1], "new_errors_without_native_progress": delta}
        confirmation = (counts, anchor)
        stop = None
        if delta >= POST_START_BOOTSTRAP_ERROR_LIMIT:
            # Confirm on a subsequent unchanged sample: independent native
            # counter reads are cooperative observations, not an atomic snapshot.
            if self.bootstrap_confirmation == confirmation:
                stop = "repeated_post_start_bootstrap_failure"
            self.bootstrap_confirmation = confirmation
        else:
            self.bootstrap_confirmation = None
        observation["failure_window_confirmed"] = stop is not None
        return observation, stop

    def sample(self, *, task, task_cid: str, state: Path, now: float, startup=None) -> str | None:
        task_row = project_task(task, task_cid=task_cid)
        heartbeat = read_heartbeat(state)
        elapsed = max(0., now - self.started)
        if not math.isfinite(elapsed):
            raise ValueError("finite observation time required")
        self.report["samples"] += 1
        row = {"seconds": elapsed, "task": task_row, "heartbeat": heartbeat}
        bootstrap, bootstrap_stop = self._sample_bootstrap(
            startup=startup, task_row=task_row, heartbeat=heartbeat)
        if bootstrap is not None:
            row["bootstrap"] = bootstrap
        self.report["latest"] = row
        # Heartbeat sequence advancement alone does not flood the transcript.
        signature = json.dumps([task_row, {k: v for k, v in heartbeat.items() if k != "sequence"}, bootstrap], sort_keys=True)
        if signature != self.previous:
            self.previous = signature
            self.transitions.append(row)
            self.report["transition_count"] += 1
            self.report["transitions"] = list(self.transitions)
            self.report["transitions_omitted"] = self.report["transition_count"] - len(self.transitions)
        status = task_row.get("status")
        stop = "native_task_" + status if status in TERMINAL_STATUSES else None
        reason = heartbeat.get("selection_idle_reason")
        if not stop and task_row.get("availability") == "observed":
            if reason == "expired_attempt_settlement_unavailable":
                stop = reason  # Preserve the existing driver's explicit stop.
            elif (reason in QUIESCENT_STOP_REASONS and status in {"ready", "in_progress", "retrying"}
                    and heartbeat.get("active_task_present") is False
                    and heartbeat.get("claimed_task_present") is False
                    and heartbeat.get("unchanged") is True and heartbeat.get("write_count") == 0):
                current = (reason, heartbeat["generation_sha256"], task_row["revision"], heartbeat["sequence"])
                if self.quiescent is not None and current[:3] == self.quiescent[:3] and current[3] > self.quiescent[3]:
                    stop = reason
                self.quiescent = current
            else:
                self.quiescent = None
        if stop is None:
            stop = bootstrap_stop
        if stop:
            self.report["stop_reason"] = stop
        return stop

    def finish(self, report: dict) -> None:
        """Keep shutdown/provider outcome metadata even when observation fails."""
        for name in ("start", "stop"):
            value = report.get(name)
            if type(value) is dict:
                result = {"status": _closed(value.get("status"), {
                    "succeeded", "failed", "denied", "conflict", "not_found", "cancelled", "unavailable"})}
                for key in ("request_id", "audit_receipt_id"):
                    digest = _identity(value.get(key))
                    if digest is not None:
                        result[key + "_sha256"] = digest
                self.report[name] = result
        invocations = report.get("provider_invocations", [])
        selected = []
        for item in invocations[:16]:
            if type(item) is not dict:
                continue
            row = {"phase": _closed(item.get("phase"), {"planning", "coding"}),
                   "status": _closed(item.get("status"), {"provider_returned", "failed"})}
            for name in ("timeout_seconds", "router_calls"):
                if _number(item.get(name), maximum=86400):
                    row[name] = item[name]
            seconds = item.get("seconds")
            if type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= 86400:
                row["seconds"] = seconds
            digest = _identity(item.get("invocation_id"))
            if digest is not None:
                row["invocation_id_sha256"] = digest
            usage = item.get("usage")
            if type(usage) is dict:
                if type(usage.get("timed_out")) is bool:
                    row["timed_out"] = usage["timed_out"]
                if type(usage.get("exit_code")) is int and -(2**31) <= usage["exit_code"] < 2**31:
                    row["exit_code"] = usage["exit_code"]
            selected.append(row)
        self.report["provider_outcomes"] = selected
        self.report["provider_outcomes_omitted"] = max(0, len(invocations) - 16)
        if "stop_reason" not in self.report:
            # TimeoutError also comes from preparation, START, or a native
            # operation. Its class alone cannot identify the work watchdog.
            self.report["stop_reason"] = ("timeout_observed_without_terminal_state" if report.get("error", {}).get("type") == "TimeoutError"
                                          else "run_ended_without_terminal_observation")
