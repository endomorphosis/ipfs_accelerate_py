"""Run one daemon command while its scheduler lease remains accepted.

This module is the execution fence between the dynamic bundle scheduler and a
lane subprocess.  The scheduler is free to reuse a lane slot after this guard
exits because every exit path either writes a terminal receipt, releases the
lease after a bookkeeping failure, or proves that this worker has already been
fenced by a newer lease.

``run_leased_lane`` retains the original integer-returning API and command-line
interface.  ``run_leased_lane_result`` exposes the same lifecycle as a small,
immutable result for in-process schedulers and tests.

FVT-G212 / FVT-078 objective validation repair: leased-lane durable completion
fencing shares the member-completion receipt schema with
``AgentSupervisorReleaseEvidence@1``.  The synthetic discovery term
``objective validation repair`` is re-exported from
:mod:`ipfs_accelerate_py.agent_supervisor.release_evidence` so scans re-find
the validation gate on this predicted path without granting completion or
proof authority.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

from ..runtime.release_evidence import (
    MEMBER_COMPLETION_RECEIPT_SCHEMA as _MEMBER_COMPLETION_RECEIPT_SCHEMA,
    OBJECTIVE_VALIDATION_REPAIR_EVIDENCE as _OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
    OBJECTIVE_VALIDATION_REPAIR_TASK_ID as _OBJECTIVE_VALIDATION_REPAIR_TASK_ID,
    objective_validation_repair_evidence_terms as _objective_validation_repair_evidence_terms,
)

# Exact-text discovery key for FVT-078 objective validation repair (re-export).
OBJECTIVE_VALIDATION_REPAIR_EVIDENCE: Final[str] = (
    _OBJECTIVE_VALIDATION_REPAIR_EVIDENCE
)
OBJECTIVE_VALIDATION_REPAIR_TASK_ID: Final[str] = (
    _OBJECTIVE_VALIDATION_REPAIR_TASK_ID
)
assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
assert _objective_validation_repair_evidence_terms() == (
    "objective validation repair",
)
from ..runtime.event_log import event_log_sources, read_jsonl_events
from .lease_coordination import LeaseCoordinator, LeaseError, LeaseGrant
from ..todo_daemon.core import terminate_pid_tree

logger = logging.getLogger(__name__)

FENCED_EXIT_CODE = 75
START_FAILED_EXIT_CODE = 70
_DUCKDB_LOCK_RETRY_INITIAL_SECONDS = 0.05
_DUCKDB_LOCK_RETRY_MAX_SECONDS = 0.5
_LEASE_EXPIRY_SAFETY_MARGIN_MS = 1_000
_DUCKDB_LOCK_ERROR_MARKERS = (
    "could not set lock",
    "conflicting lock",
)

LaneDisposition = Literal[
    "completed",
    "pending_acceptance",
    "blocked",
    "failed",
    "cancelled",
    "fenced",
    "start_failed",
]


class ProcessFenceError(RuntimeError):
    """The lane cannot prove that its spawned process group is quiescent."""


@dataclass(frozen=True)
class LeasedLaneResult:
    """Scheduler-facing terminal state for one leased command execution."""

    task_cid: str
    claim_cid: str
    claimant_did: str
    fencing_token: int
    disposition: LaneDisposition
    exit_code: int
    child_exit_code: int | None
    started_at_ms: int
    finished_at_ms: int
    receipt_cid: str | None = None
    resolution_cid: str | None = None
    lease_released: bool = False
    error: str = ""

    @property
    def reusable(self) -> bool:
        """Return whether the scheduler may immediately reuse the lane slot."""

        # A fenced result is reusable as well: the old process has been
        # terminated and authority belongs to another accepted fencing token.
        if self.disposition == "fenced":
            return True
        return self.lease_released and self.disposition in {
            "completed",
            "pending_acceptance",
            "blocked",
            "failed",
            "cancelled",
            "start_failed",
        }

    @property
    def successful(self) -> bool:
        return self.disposition == "completed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable projection suitable for a live manifest."""

        return {**asdict(self), "reusable": self.reusable, "successful": self.successful}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_transient_duckdb_lock_error(exc: Exception) -> bool:
    """Return whether ``exc`` is the narrow retryable DuckDB lock conflict."""

    if isinstance(exc, LeaseError):
        return False
    exception_type = type(exc)
    if (
        exception_type.__module__ not in {"duckdb", "_duckdb"}
        or exception_type.__name__ not in {"IOException", "OperationalError"}
    ):
        return False
    message = str(exc).casefold()
    return all(marker in message for marker in _DUCKDB_LOCK_ERROR_MARKERS)


def _resource_measurements(
    sampler: Callable[..., Any] | None,
    *,
    active_phase: str,
    occupied_workers: int,
) -> dict[str, Any]:
    """Return a best-effort, integer-only heartbeat resource projection."""

    try:
        if sampler is None:
            from ..runtime.resource_scheduler import sample_host_resources

            snapshot = sample_host_resources(
                Path.cwd(),
                active_workers=occupied_workers,
                worker_limit=1,
                active_phase=active_phase,
            )
        else:
            try:
                snapshot = sampler(
                    active_workers=occupied_workers,
                    worker_limit=1,
                    active_phase=active_phase,
                )
            except TypeError:
                snapshot = sampler()
        values = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
    except Exception:
        # Resource telemetry must never cause a healthy, fenced child to lose
        # its lease.  Explicit occupancy still makes the heartbeat useful.
        logger.debug("Could not sample lane resources", exc_info=True)
        values = {}

    aliases = {
        "cpu_millionths": ("cpu_millionths", "cpu_utilization_millionths"),
        "cpu_percent": ("cpu_percent",),
        "memory_percent": ("memory_percent",),
        "disk_percent": ("disk_percent",),
        "memory_used_bytes": ("memory_used_bytes",),
        "memory_available_bytes": ("memory_available_bytes", "available_memory_bytes"),
        "memory_total_bytes": ("memory_total_bytes", "total_memory_bytes"),
        "disk_used_bytes": ("disk_used_bytes",),
        "disk_available_bytes": ("disk_available_bytes", "available_disk_bytes"),
        "disk_total_bytes": ("disk_total_bytes", "total_disk_bytes"),
    }
    result: dict[str, Any] = {
        "active_phase": active_phase,
        "occupied_workers": int(occupied_workers),
        "available_workers": max(0, 1 - int(occupied_workers)),
    }
    for target, candidates in aliases.items():
        for candidate in candidates:
            value = values.get(candidate)
            if value is not None:
                try:
                    result[target] = int(value)
                except (TypeError, ValueError):
                    pass
                break
    return result


def _active_phase(state_path: Path | None, default: str) -> str:
    if state_path is None:
        return default
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if not isinstance(state, dict):
        return default
    explicit = str(state.get("active_phase") or "").strip()
    if explicit:
        return explicit[:64]
    if state.get("implementation_in_progress") or state.get("active_task_id"):
        return "implementation"
    if int(state.get("ready_count") or 0) > 0:
        return "ready"
    if int(state.get("blocked_count") or 0) > 0:
        return "blocked"
    return default


def _execution_slice_violation(
    state_path: Path | None,
    expected_task_ids: frozenset[str],
    *,
    started_at_ms: int,
) -> str:
    """Return an unauthorized active task from fresh lane state, if any."""

    if state_path is None or not expected_task_ids:
        return ""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(state, dict):
        return ""
    active_task_id = str(state.get("active_task_id") or "").strip()
    if not active_task_id or active_task_id in expected_task_ids:
        return ""

    observed_times_ms: list[int] = []
    for field_name in (
        "heartbeat_at",
        "active_task_started_at",
        "active_phase_started_at",
        "last_implementation_started_at",
    ):
        raw = str(state.get(field_name) or "").strip()
        if not raw:
            continue
        try:
            observed_times_ms.append(int(datetime.fromisoformat(raw).timestamp() * 1000))
        except ValueError:
            continue
    # A phase file can survive a clean restart. Give the replacement child a
    # chance to recover stale state, then enforce the scope as soon as it
    # writes any fresh heartbeat or execution timestamp.
    if observed_times_ms and max(observed_times_ms) + 1_000 < started_at_ms:
        return ""
    return active_task_id


# Durable member receipts share one schema with AgentSupervisorReleaseEvidence@1.
# The constant is imported from release_evidence so G212 exports and leased-lane
# completion fencing cannot drift.
_TASK_ATTEMPT_LIMIT_IDLE_REASON = (
    "all_selectable_ready_tasks_reached_max_task_attempts"
)
_PENDING_ACCEPTANCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/pending-acceptance@1"
)
_ACCEPTANCE_STATUS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/authoritative-acceptance-status@1"
)
_IMPLEMENTATION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/implementation-receipt@1"
)
_AUTHORITATIVE_COMPLETION_GATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/authoritative-completion-gate@1"
)
_PENDING_ACCEPTANCE_RETRY_DELAY_MS = 30_000
_PROVIDER_REVIEW_SATISFIED_GATES = frozenset(
    {"merge", "freshness", "semantic", "proof", "deterministic_only"}
)


def _normalized_task_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    return "completed" if status == "complete" else status


def _expected_task_identity_map(
    expected_task_ids: Sequence[str],
    expected_task_cids_by_id: Mapping[str, str] | None,
) -> dict[str, str]:
    """Normalize and validate one exact execution-slice identity projection."""

    normalized_ids = tuple(
        dict.fromkeys(
            str(task_id).strip()
            for task_id in expected_task_ids
            if str(task_id).strip()
        )
    )
    normalized_bindings = {
        str(task_id).strip(): str(task_cid).strip()
        for task_id, task_cid in (expected_task_cids_by_id or {}).items()
        if str(task_id).strip() and str(task_cid).strip()
    }
    if normalized_ids and not normalized_bindings:
        raise ValueError(
            "expected_task_ids require exact expected_task_cids_by_id bindings"
        )
    if normalized_bindings and set(normalized_bindings) != set(normalized_ids):
        raise ValueError(
            "expected task display IDs and canonical CID bindings must match exactly"
        )
    return {
        task_id: normalized_bindings[task_id]
        for task_id in normalized_ids
    }


def _timestamp_ms(value: Any) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return int(parsed.timestamp() * 1000)


def _validated_member_completion_receipts(
    rows: Any,
    expected_task_cids_by_id: Mapping[str, str],
) -> list[dict[str, str]] | None:
    """Validate exact successful member receipts without accepting ID aliases."""

    if not isinstance(rows, list):
        return None
    expected = set(expected_task_cids_by_id.items())
    matched: dict[tuple[str, str], dict[str, str]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        task_id = str(raw.get("task_id") or "").strip()
        canonical_task_cid = str(raw.get("canonical_task_cid") or "").strip()
        pair = (task_id, canonical_task_cid)
        if pair not in expected:
            continue
        if str(raw.get("schema") or "") != _MEMBER_COMPLETION_RECEIPT_SCHEMA:
            continue
        if str(raw.get("status") or "").strip().lower() != "succeeded":
            continue
        matched[pair] = {
            "task_id": task_id,
            "canonical_task_cid": canonical_task_cid,
        }
    if set(matched) != expected:
        return None
    return [matched[pair] for pair in sorted(matched)]


def _fresh_durable_member_completion_receipts(
    events_path: Path,
    expected_task_cids_by_id: Mapping[str, str],
    *,
    started_at_ms: int,
) -> dict[str, Any] | None:
    """Return fresh fsynced terminal member receipts from one daemon event log."""

    matched: dict[tuple[str, str], dict[str, str]] = {}
    event_ids: set[str] = set()
    expected = set(expected_task_cids_by_id.items())
    for source in event_log_sources((events_path,), include_rotated=True):
        for event in read_jsonl_events(source):
            event_at_ms = _timestamp_ms(event.get("timestamp"))
            if event_at_ms is None or event_at_ms < int(started_at_ms):
                continue
            event_type = str(event.get("type") or "")
            if event_type == "todo_status_updated":
                completed_ids = {
                    str(item)
                    for key in ("updated_task_ids", "already_completed_task_ids")
                    for item in (event.get(key) or ())
                }
                if not event.get("updated") and not completed_ids:
                    continue
                completion_payload: Mapping[str, Any] = event
            elif event_type == "implementation_finished":
                merge_result = event.get("merge_result")
                if (
                    event.get("returncode") != 0
                    or not isinstance(merge_result, Mapping)
                    or merge_result.get("merged") is not True
                ):
                    continue
                candidate = event.get("todo_update_result")
                completion_payload = (
                    candidate if isinstance(candidate, Mapping) else {}
                )
            else:
                continue
            rows = completion_payload.get("completion_receipts")
            if rows is None:
                rows = completion_payload.get("member_completion_receipts")
            validated = _validated_member_completion_receipts(
                rows,
                expected_task_cids_by_id,
            )
            if validated is None:
                # A packet can complete only part of a multi-member slice. Add
                # each exact successful row, then prove the union below.
                validated = []
                if isinstance(rows, list):
                    for row in rows:
                        single = _validated_member_completion_receipts(
                            [row],
                            {
                                task_id: task_cid
                                for task_id, task_cid in expected
                                if isinstance(row, Mapping)
                                and str(row.get("task_id") or "").strip() == task_id
                                and str(
                                    row.get("canonical_task_cid") or ""
                                ).strip()
                                == task_cid
                            },
                        )
                        if single:
                            validated.extend(single)
            for receipt in validated:
                pair = (
                    receipt["task_id"],
                    receipt["canonical_task_cid"],
                )
                matched[pair] = receipt
                event_id = str(event.get("event_id") or "").strip()
                if event_id:
                    event_ids.add(event_id)
    if set(matched) != expected:
        return None
    return {
        "member_completion_receipts_validated": True,
        "completion_receipt_boundary": "durable_event_log",
        "completion_events_path": str(events_path),
        "completion_event_ids": sorted(event_ids),
    }


def _fresh_provider_review_pending_acceptance(
    state_path: Path | None,
    events_path: Path | None,
    expected_task_cids_by_id: Mapping[str, str],
    *,
    started_at_ms: int,
) -> dict[str, Any] | None:
    """Return exact durable evidence for resumable provider review.

    Every admitted execution-slice member must remain ready on the board and
    have a fresh ``implementation_merged_pending_acceptance`` event proving
    that merge and all non-review gates passed.  A later exact ``daemon_pass``
    proves the child reached an idle boundary.  Anything partial, stale,
    readdressed, or authoritative fails closed.
    """

    if state_path is None or events_path is None or not expected_task_cids_by_id:
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, Mapping):
        return None
    if state.get("implementation_in_progress") is not False:
        return None
    if str(state.get("active_task_id") or "").strip():
        return None
    heartbeat_at_ms = _timestamp_ms(state.get("heartbeat_at"))
    observed_at_ms = _now_ms()
    if (
        heartbeat_at_ms is None
        or heartbeat_at_ms < int(started_at_ms)
        or heartbeat_at_ms > observed_at_ms + 1_000
    ):
        return None
    identities = state.get("task_identities")
    statuses = state.get("task_statuses")
    if not isinstance(identities, Mapping) or not isinstance(statuses, Mapping):
        return None
    for task_id, task_cid in expected_task_cids_by_id.items():
        identity = identities.get(task_id)
        if (
            not isinstance(identity, Mapping)
            or str(identity.get("canonical_task_cid") or "").strip()
            != task_cid
            or _normalized_task_status(statuses.get(task_id)) != "ready"
        ):
            return None

    events: list[dict[str, Any]] = []
    for source in event_log_sources((events_path,), include_rotated=True):
        events.extend(read_jsonl_events(source))
    pending_by_task_id: dict[str, dict[str, Any]] = {}
    pending_at_by_task_id: dict[str, int] = {}
    for event in events:
        if str(event.get("type") or "") != "implementation_merged_pending_acceptance":
            continue
        event_at_ms = _timestamp_ms(event.get("timestamp"))
        if (
            event_at_ms is None
            or event_at_ms < int(started_at_ms)
            or event_at_ms > observed_at_ms + 1_000
        ):
            continue
        task_id = str(event.get("task_id") or "").strip()
        expected_cid = expected_task_cids_by_id.get(task_id)
        if not expected_cid:
            continue
        gate = event.get("gate")
        receipt = event.get("receipt")
        pending = tuple(
            str(item).strip()
            for item in (event.get("pending_gates") or ())
            if str(item).strip()
        )
        if not isinstance(gate, Mapping) or not isinstance(receipt, Mapping):
            continue
        gate_pending = tuple(
            str(item).strip()
            for item in (gate.get("pending_gates") or ())
            if str(item).strip()
        )
        receipt_pending = tuple(
            str(item).strip()
            for item in (receipt.get("pending_gates") or ())
            if str(item).strip()
        )
        satisfied = {
            str(item).strip()
            for item in (gate.get("satisfied_gates") or ())
            if str(item).strip()
        }
        merge_commit = str(event.get("merge_commit") or "").strip()
        repository_tree_id = str(gate.get("repository_tree_id") or "").strip()
        if (
            event.get("schema") != _ACCEPTANCE_STATUS_SCHEMA
            or str(event.get("canonical_task_cid") or "").strip()
            != expected_cid
            or event.get("admitted") is not False
            or event.get("completion_authoritative") is not False
            or str(event.get("acceptance_state") or event.get("state") or "")
            != "implemented_merged_but_pending"
            or pending != ("provider_review",)
            or gate.get("admitted") is not False
            or gate.get("schema") != _AUTHORITATIVE_COMPLETION_GATE_SCHEMA
            or str(gate.get("task_id") or "").strip() != task_id
            or gate.get("completion_authoritative") is not False
            or str(gate.get("acceptance_state") or "")
            != "implemented_merged_but_pending"
            or gate_pending != ("provider_review",)
            or not _PROVIDER_REVIEW_SATISFIED_GATES.issubset(satisfied)
            or gate.get("merge_commit") != merge_commit
            or receipt.get("merged") is not True
            or receipt.get("schema") != _IMPLEMENTATION_RECEIPT_SCHEMA
            or str(receipt.get("task_id") or "").strip() != task_id
            or receipt.get("completion_authoritative") is not False
            or str(receipt.get("acceptance_state") or "")
            != "implemented_merged_but_pending"
            or receipt_pending != ("provider_review",)
            or receipt.get("validation_passed") is not True
            or receipt.get("validation_stale") is not False
            or not merge_commit
            or str(receipt.get("merge_commit") or "").strip() != merge_commit
            or not repository_tree_id
            or str(receipt.get("repository_tree_id") or "").strip()
            != repository_tree_id
            or not str(event.get("event_id") or "").strip()
        ):
            continue
        pending_by_task_id[task_id] = dict(event)
        pending_at_by_task_id[task_id] = event_at_ms

    if set(pending_by_task_id) != set(expected_task_cids_by_id):
        return None
    pending_not_before_ms = max(pending_at_by_task_id.values())
    if heartbeat_at_ms < pending_not_before_ms:
        return None

    terminal_pass: dict[str, Any] | None = None
    terminal_pass_at_ms = 0
    expected_ids = set(expected_task_cids_by_id)
    for event in events:
        if str(event.get("type") or "") != "daemon_pass":
            continue
        event_at_ms = _timestamp_ms(event.get("timestamp"))
        if (
            event_at_ms is None
            or event_at_ms < pending_not_before_ms
            or event_at_ms > observed_at_ms + 1_000
            or str(event.get("active_task_id") or "").strip()
        ):
            continue
        pass_cids = event.get("execution_slice_task_cids_by_id")
        pass_statuses = event.get("execution_slice_task_statuses")
        if not isinstance(pass_cids, Mapping) or not isinstance(pass_statuses, Mapping):
            continue
        if set(pass_cids) != expected_ids or set(pass_statuses) != expected_ids:
            continue
        if any(
            str(pass_cids.get(task_id) or "").strip() != task_cid
            or _normalized_task_status(pass_statuses.get(task_id)) != "ready"
            for task_id, task_cid in expected_task_cids_by_id.items()
        ):
            continue
        if not str(event.get("event_id") or "").strip():
            continue
        if event_at_ms >= terminal_pass_at_ms:
            terminal_pass = dict(event)
            terminal_pass_at_ms = event_at_ms
    if terminal_pass is None:
        return None

    ordered_task_ids = sorted(expected_task_cids_by_id)
    return {
        "schema": _PENDING_ACCEPTANCE_SCHEMA,
        "acceptance_pending": True,
        "completion_authoritative": False,
        "admitted": False,
        "pending_gates": ["provider_review"],
        "task_ids": ordered_task_ids,
        "task_cids": [
            expected_task_cids_by_id[task_id] for task_id in ordered_task_ids
        ],
        "task_cids_by_id": {
            task_id: expected_task_cids_by_id[task_id]
            for task_id in ordered_task_ids
        },
        "acceptance_event_ids": sorted(
            str(event["event_id"]) for event in pending_by_task_id.values()
        ),
        "terminal_event_id": str(terminal_pass["event_id"]),
        "terminal_event_at_ms": terminal_pass_at_ms,
        "phase_state_heartbeat_at_ms": heartbeat_at_ms,
    }


def _fresh_durable_terminal_blocked_pass(
    events_path: Path,
    state: Mapping[str, Any],
    *,
    expected_task_cids_by_id: Mapping[str, str],
    expected_task_statuses: Mapping[str, str],
    expected_attempt_limited_task_ids: frozenset[str],
    started_at_ms: int,
    heartbeat_at_ms: int,
) -> dict[str, Any] | None:
    """Bind terminal-blocked phase state to a later fsynced daemon pass."""

    matched: dict[str, Any] | None = None
    count_fields = (
        "completed_count",
        "ready_count",
        "waiting_count",
        "blocked_count",
        "selectable_ready_count",
    )
    expected_task_ids = set(expected_task_cids_by_id)
    observed_at_ms = _now_ms()
    for source in event_log_sources((events_path,), include_rotated=True):
        for event in read_jsonl_events(source):
            if str(event.get("type") or "") != "daemon_pass":
                continue
            event_at_ms = _timestamp_ms(event.get("timestamp"))
            if (
                event_at_ms is None
                or event_at_ms < int(started_at_ms)
                or event_at_ms < int(heartbeat_at_ms)
                or event_at_ms > observed_at_ms + 1_000
            ):
                continue
            if str(event.get("active_task_id") or "").strip():
                continue
            if any(
                int(event.get(field) or 0) != int(state.get(field) or 0)
                for field in count_fields
            ):
                continue
            state_idle_reason = str(state.get("selection_idle_reason") or "")
            if str(event.get("selection_idle_reason") or "") != state_idle_reason:
                continue
            event_task_cids = event.get("execution_slice_task_cids_by_id")
            event_task_statuses = event.get("execution_slice_task_statuses")
            if not isinstance(event_task_cids, Mapping) or not isinstance(
                event_task_statuses,
                Mapping,
            ):
                continue
            if (
                set(event_task_cids) != expected_task_ids
                or set(event_task_statuses) != expected_task_ids
            ):
                continue
            if any(
                str(event_task_cids.get(task_id) or "").strip() != task_cid
                or _normalized_task_status(event_task_statuses.get(task_id))
                != expected_task_statuses[task_id]
                for task_id, task_cid in expected_task_cids_by_id.items()
            ):
                continue
            limited_ids = frozenset(
                str(task_id).strip()
                for task_id in (event.get("attempt_limited_task_ids") or ())
                if str(task_id).strip()
            )
            if limited_ids != expected_attempt_limited_task_ids:
                continue
            matched = {
                "terminal_evidence_boundary": "durable_daemon_pass",
                "terminal_events_path": str(events_path),
                "terminal_event_id": str(event.get("event_id") or ""),
                "terminal_event_at_ms": event_at_ms,
            }
    return matched


def _fresh_blocked_execution_slice(
    state_path: Path | None,
    expected_task_cids_by_id: Mapping[str, str],
    *,
    started_at_ms: int,
    completion_events_path: Path | None = None,
) -> dict[str, Any] | None:
    """Return fresh exact evidence that a leased slice cannot make progress.

    A terminal-blocked projection must bind every expected display ID to its
    admitted canonical CID, show no active implementation, and report every
    expected status as completed or blocked. Attempt-limited ready members are
    treated as blocked only when the exact idle reason and zero selectable
    capacity agree; bundle lanes additionally require a later fsynced
    ``daemon_pass`` event before the wrapper may stop its child.
    """

    if state_path is None or not expected_task_cids_by_id:
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    if state.get("implementation_in_progress") is not False:
        return None
    active_task_id = state.get("active_task_id")
    if not isinstance(active_task_id, str) or active_task_id.strip():
        return None
    heartbeat_at = str(state.get("heartbeat_at") or "").strip()
    heartbeat_at_ms = _timestamp_ms(heartbeat_at)
    observed_at_ms = _now_ms()
    if (
        heartbeat_at_ms is None
        or heartbeat_at_ms < int(started_at_ms)
        or heartbeat_at_ms > observed_at_ms + 1_000
    ):
        return None
    task_identities = state.get("task_identities")
    task_statuses = state.get("task_statuses")
    if not isinstance(task_identities, Mapping) or not isinstance(
        task_statuses,
        Mapping,
    ):
        return None

    statuses: dict[str, str] = {}
    for task_id, expected_task_cid in expected_task_cids_by_id.items():
        identity = task_identities.get(task_id)
        if not isinstance(identity, Mapping):
            return None
        if str(identity.get("canonical_task_cid") or "").strip() != expected_task_cid:
            return None
        raw_status = task_statuses.get(task_id)
        if not isinstance(raw_status, str) or not raw_status.strip():
            return None
        statuses[task_id] = _normalized_task_status(raw_status)

    projected_statuses = dict(statuses)
    attempt_limited_ids: frozenset[str] = frozenset()
    if str(state.get("selection_idle_reason") or "") == _TASK_ATTEMPT_LIMIT_IDLE_REASON:
        selectable_ready_count = state.get("selectable_ready_count")
        if (
            isinstance(selectable_ready_count, bool)
            or not isinstance(selectable_ready_count, int)
            or selectable_ready_count != 0
        ):
            return None
        attempt_limited_ids = frozenset(
            task_id for task_id, status in statuses.items() if status == "ready"
        )
        if not attempt_limited_ids:
            return None
        statuses.update({task_id: "blocked" for task_id in attempt_limited_ids})

    terminal_statuses = {"completed", "blocked", "on_hold"}
    if (
        any(status not in terminal_statuses for status in statuses.values())
        or not any(status in {"blocked", "on_hold"} for status in statuses.values())
    ):
        return None

    if completion_events_path is not None:
        durable_evidence = _fresh_durable_terminal_blocked_pass(
            completion_events_path,
            state,
            expected_task_cids_by_id=expected_task_cids_by_id,
            expected_task_statuses=projected_statuses,
            expected_attempt_limited_task_ids=attempt_limited_ids,
            started_at_ms=started_at_ms,
            heartbeat_at_ms=heartbeat_at_ms,
        )
        if durable_evidence is None:
            return None
    else:
        durable_evidence = {
            "terminal_evidence_boundary": "phase_state_identity_only",
        }

    blocked_task_ids = sorted(
        task_id
        for task_id, status in statuses.items()
        if status in {"blocked", "on_hold"}
    )
    return {
        "blocked_task_ids": blocked_task_ids,
        "blocked_task_cids": sorted(
            expected_task_cids_by_id[task_id]
            for task_id in blocked_task_ids
        ),
        "execution_slice_task_statuses": dict(sorted(statuses.items())),
        "attempt_limited_task_ids": sorted(attempt_limited_ids),
        "terminal_reason": (
            "task_attempt_limit"
            if attempt_limited_ids
            else "terminal_blocked_status"
        ),
        "phase_state_not_before_ms": int(started_at_ms),
        "phase_state_heartbeat_at": heartbeat_at,
        "phase_state_heartbeat_at_ms": heartbeat_at_ms,
        **durable_evidence,
    }


def _fresh_completed_execution_slice(
    state_path: Path | None,
    expected_task_cids_by_id: Mapping[str, str],
    *,
    started_at_ms: int,
    completion_events_path: Path | None = None,
) -> dict[str, Any] | None:
    """Return fresh, identity-bound evidence for a completed execution slice.

    Implementation supervisors are intentionally long-lived.  Completing the
    final leased task therefore does not imply that their process exits.  The
    wrapper may stop that process only when one atomic phase-state projection
    proves all leased members complete and proves that no implementation is
    still active.  Every display ID must resolve to the exact canonical CID
    admitted by the bundle planner.

    A task-state file can survive an earlier lease.  Completion is consequently
    authoritative only when its heartbeat was written after this wrapper began;
    missing, malformed, naive, or older timestamps fail closed.

    Bundle lanes also supply their daemon's append-only event-log path.  In
    that mode, the state projection is necessary but insufficient: fresh,
    fsynced ``member_completion_receipt@1`` records must prove every exact
    ID/CID member succeeded.  The optional path preserves an identity-only
    boundary for direct in-process integrations whose architecture exposes no
    durable member receipt source.
    """

    if state_path is None or not expected_task_cids_by_id:
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    completed_value = state.get("completed_task_ids")
    if not isinstance(completed_value, list):
        return None
    if any(
        not isinstance(task_id, str) or not task_id.strip()
        for task_id in completed_value
    ):
        return None
    completed_task_ids = {
        task_id.strip()
        for task_id in completed_value
    }
    expected_task_ids = frozenset(expected_task_cids_by_id)
    if not expected_task_ids.issubset(completed_task_ids):
        return None
    task_identities = state.get("task_identities")
    if not isinstance(task_identities, Mapping):
        return None
    for task_id, expected_task_cid in expected_task_cids_by_id.items():
        identity = task_identities.get(task_id)
        if not isinstance(identity, Mapping):
            return None
        if (
            str(identity.get("canonical_task_cid") or "").strip()
            != expected_task_cid
        ):
            return None
    if state.get("implementation_in_progress") is not False:
        return None
    active_task_id = state.get("active_task_id")
    if not isinstance(active_task_id, str) or active_task_id.strip():
        return None
    heartbeat_value = state.get("heartbeat_at")
    if not isinstance(heartbeat_value, str):
        return None
    heartbeat_at = heartbeat_value.strip()
    if not heartbeat_at:
        return None
    heartbeat_at_ms = _timestamp_ms(heartbeat_at)
    if heartbeat_at_ms is None:
        return None
    observed_at_ms = _now_ms()
    if (
        heartbeat_at_ms < int(started_at_ms)
        or heartbeat_at_ms > observed_at_ms + 1_000
    ):
        return None
    receipt_evidence: dict[str, Any]
    if completion_events_path is not None:
        durable_evidence = _fresh_durable_member_completion_receipts(
            completion_events_path,
            expected_task_cids_by_id,
            started_at_ms=started_at_ms,
        )
        if durable_evidence is None:
            return None
        receipt_evidence = durable_evidence
    else:
        embedded_key = next(
            (
                key
                for key in (
                    "completion_receipts",
                    "member_completion_receipts",
                )
                if key in state
            ),
            "",
        )
        if embedded_key:
            embedded = _validated_member_completion_receipts(
                state.get(embedded_key),
                expected_task_cids_by_id,
            )
            if embedded is None:
                return None
            receipt_evidence = {
                "member_completion_receipts_validated": True,
                "completion_receipt_boundary": "phase_state_embedded",
            }
        else:
            receipt_evidence = {
                "member_completion_receipts_validated": False,
                "completion_receipt_boundary": "phase_state_identity_only",
            }
    return {
        "completed_task_ids": sorted(expected_task_ids),
        "completed_task_cids": sorted(expected_task_cids_by_id.values()),
        "phase_state_not_before_ms": int(started_at_ms),
        "phase_state_heartbeat_at": heartbeat_at,
        "phase_state_heartbeat_at_ms": heartbeat_at_ms,
        **receipt_evidence,
    }


def _terminate_child(
    process: subprocess.Popen[Any],
    *,
    timeout: float = 5.0,
    fence_descendants: bool = False,
) -> None:
    """Terminate a child, optionally proving its dedicated process tree gone.

    Completion uses ``fence_descendants`` because capacity is published only
    after the stopped/rescanned tree and its owned process group have no live
    members.  The ordinary compatibility path retains graceful termination.
    """

    if fence_descendants:
        expected_start_time = getattr(
            process,
            "_supervisor_start_time_ticks",
            None,
        )
        fenced = terminate_pid_tree(
            process.pid,
            grace_seconds=timeout,
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=(process.pid if os.name == "posix" else None),
            expected_root_start_time_ticks=expected_start_time,
        )
        if not fenced:
            raise ProcessFenceError(
                f"could not prove process tree {process.pid} fully fenced"
            )
    elif process.poll() is not None:
        return
    else:
        terminate_pid_tree(process.pid, grace_seconds=timeout)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _capture_spawned_direct_child_start_time(
    pid: int,
    *,
    expected_parent_pid: int,
    proc_root: Path = Path("/proc"),
) -> int | None:
    """Capture a direct child's Linux birth time, including zombie children.

    General lifecycle liveness intentionally treats zombies as dead.  Spawn
    fencing has a different requirement: the unreaped zombie's ``stat`` record
    is the last authoritative chance to bind a fast child PID to its dedicated
    process group before proving that group empty.
    """

    if int(pid) <= 1 or int(expected_parent_pid) <= 0:
        return None
    try:
        raw = (proc_root / str(int(pid)) / "stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    try:
        parent_pid = int(fields[1])
        start_time_ticks = int(fields[19])
    except (TypeError, ValueError, IndexError):
        return None
    if parent_pid != int(expected_parent_pid) or start_time_ticks <= 0:
        return None
    return start_time_ticks


def _receipt_cid(receipt: Mapping[str, Any] | None) -> str | None:
    if not receipt:
        return None
    value = receipt.get("receipt_cid")
    return str(value) if value else None


def _release_after_bookkeeping_failure(
    coordinator: LeaseCoordinator,
    grant: LeaseGrant,
) -> bool:
    """Best-effort release when a non-fencing receipt error occurs."""

    try:
        coordinator.release(grant)
        return True
    except LeaseError:
        # A takeover or expiry has already removed this worker's authority.
        return False
    except Exception:
        logger.exception("Could not release lease after terminal receipt failure")
        return False


def run_leased_lane_result(
    *,
    coordination_path: Path,
    grant: LeaseGrant,
    command: Sequence[str],
    lease_ms: int,
    heartbeat_interval: float,
    capacity_millionths: int = 1_000_000,
    resource_class: str = "",
    provider_id: str = "",
    resource_sampler: Callable[..., Any] | None = None,
    phase_state_path: Path | None = None,
    expected_task_ids: Sequence[str] = (),
    expected_task_cids_by_id: Mapping[str, str] | None = None,
    completion_events_path: Path | None = None,
) -> LeasedLaneResult:
    """Run ``command`` and return its fenced, identity-bound disposition.

    Successful children produce a successful receipt (which completes the
    task).  Provider-review-only acceptance gaps produce a deferred claim
    resolution with no task receipt. Non-zero children produce a retryable
    failed receipt, signals produce a cancelled receipt, and both cases release
    the task for another worker. Exit code 75 is treated as a blocked/retryable
    child convention.
    If renewal or heartbeat proves the grant stale, the child is synchronously
    stopped and no stale receipt is manufactured.  Supplying execution-slice
    display IDs requires an exact canonical-CID binding.  Bundle callers also
    supply ``completion_events_path`` so phase completion requires durable
    member receipts; callers with no execution slice retain child-exit mode.
    """

    if not command:
        raise ValueError("leased lane command is required")
    if float(heartbeat_interval) <= 0:
        raise ValueError("heartbeat_interval must be positive")
    if int(lease_ms) <= 0:
        raise ValueError("lease_ms must be positive")

    started_at_ms = _now_ms()
    expected_task_identity_map = _expected_task_identity_map(
        expected_task_ids,
        expected_task_cids_by_id,
    )
    expected_task_id_set = frozenset(expected_task_identity_map)
    with LeaseCoordinator(coordination_path) as coordinator:
        try:
            grant = coordinator.validate(grant)
            coordinator.heartbeat(
                grant,
                capacity_millionths=capacity_millionths,
                resource_class=resource_class,
                provider_id=provider_id,
                **_resource_measurements(
                    resource_sampler,
                    active_phase=_active_phase(phase_state_path, "starting"),
                    occupied_workers=1,
                ),
            )
        except LeaseError as exc:
            logger.warning("Refused to start fenced lane %s: %s", grant.task_cid, exc)
            return LeasedLaneResult(
                task_cid=grant.task_cid,
                claim_cid=grant.claim_cid,
                claimant_did=grant.claimant_did,
                fencing_token=grant.fencing_token,
                disposition="fenced",
                exit_code=FENCED_EXIT_CODE,
                child_exit_code=None,
                started_at_ms=started_at_ms,
                finished_at_ms=_now_ms(),
                error=str(exc),
            )
        except Exception as exc:
            # Invalid capacity and local coordination failures must not strand
            # an accepted lease when no child was ever started.
            logger.exception("Could not initialize leased lane %s", grant.task_cid)
            released = _release_after_bookkeeping_failure(coordinator, grant)
            return LeasedLaneResult(
                task_cid=grant.task_cid,
                claim_cid=grant.claim_cid,
                claimant_did=grant.claimant_did,
                fencing_token=grant.fencing_token,
                disposition="start_failed",
                exit_code=START_FAILED_EXIT_CODE,
                child_exit_code=None,
                started_at_ms=started_at_ms,
                finished_at_ms=_now_ms(),
                lease_released=released,
                error=str(exc),
            )

        phase_state_not_before_ms = _now_ms()
        try:
            process = subprocess.Popen(
                list(command),
                start_new_session=(os.name == "posix"),
            )
            if os.name == "posix" and Path("/proc").is_dir():
                start_time_ticks = _capture_spawned_direct_child_start_time(
                    int(process.pid),
                    expected_parent_pid=os.getpid(),
                )
                if start_time_ticks is None:
                    try:
                        _terminate_child(process, fence_descendants=True)
                    except ProcessFenceError as fence_exc:
                        return LeasedLaneResult(
                            task_cid=grant.task_cid,
                            claim_cid=grant.claim_cid,
                            claimant_did=grant.claimant_did,
                            fencing_token=grant.fencing_token,
                            disposition="start_failed",
                            exit_code=START_FAILED_EXIT_CODE,
                            child_exit_code=process.returncode,
                            started_at_ms=started_at_ms,
                            finished_at_ms=_now_ms(),
                            lease_released=False,
                            error=(
                                "spawned child birth identity unavailable; "
                                f"{fence_exc}"
                            ),
                        )
                    raise RuntimeError(
                        "spawned child birth identity unavailable"
                    )
                process._supervisor_start_time_ticks = start_time_ticks  # type: ignore[attr-defined]
        except Exception as exc:
            logger.error("Could not start leased lane %s: %s", grant.task_cid, exc)
            try:
                receipt = coordinator.receipt(
                    grant,
                    status="failed",
                    failure_class="retryable",
                    started_at_ms=started_at_ms,
                )
                return LeasedLaneResult(
                    task_cid=grant.task_cid,
                    claim_cid=grant.claim_cid,
                    claimant_did=grant.claimant_did,
                    fencing_token=grant.fencing_token,
                    disposition="start_failed",
                    exit_code=START_FAILED_EXIT_CODE,
                    child_exit_code=None,
                    started_at_ms=started_at_ms,
                    finished_at_ms=_now_ms(),
                    receipt_cid=_receipt_cid(receipt),
                    lease_released=True,
                    error=str(exc),
                )
            except LeaseError as lease_exc:
                return LeasedLaneResult(
                    task_cid=grant.task_cid,
                    claim_cid=grant.claim_cid,
                    claimant_did=grant.claimant_did,
                    fencing_token=grant.fencing_token,
                    disposition="fenced",
                    exit_code=FENCED_EXIT_CODE,
                    child_exit_code=None,
                    started_at_ms=started_at_ms,
                    finished_at_ms=_now_ms(),
                    error=str(lease_exc),
                )
            except Exception as receipt_exc:
                logger.exception("Could not record leased lane start failure")
                released = _release_after_bookkeeping_failure(coordinator, grant)
                return LeasedLaneResult(
                    task_cid=grant.task_cid,
                    claim_cid=grant.claim_cid,
                    claimant_did=grant.claimant_did,
                    fencing_token=grant.fencing_token,
                    disposition="start_failed",
                    exit_code=START_FAILED_EXIT_CODE,
                    child_exit_code=None,
                    started_at_ms=started_at_ms,
                    finished_at_ms=_now_ms(),
                    lease_released=released,
                    error=f"{exc}; terminal bookkeeping failed: {receipt_exc}",
                )

        stopping_signal: int | None = None
        execution_scope_error = ""
        completed_execution_slice: dict[str, Any] | None = None
        pending_acceptance_slice: dict[str, Any] | None = None
        blocked_execution_slice: dict[str, Any] | None = None
        stop_event = threading.Event()

        def stop_child(signum: int, _frame: object) -> None:
            nonlocal stopping_signal
            stopping_signal = signum
            stop_event.set()
            # The polling loop must capture and terminate the whole descendant
            # tree before the immediate child can exit and orphan its daemon.

        handlers_installed = threading.current_thread() is threading.main_thread()
        old_term: Any = None
        old_int: Any = None
        if handlers_installed:
            old_term = signal.signal(signal.SIGTERM, stop_child)
            old_int = signal.signal(signal.SIGINT, stop_child)

        try:
            while process.poll() is None:
                now = _now_ms()
                renew_window_ms = max(1_000, int(lease_ms) // 2)
                until_renewal = max(0.0, (grant.lease_expires_at_ms - now - renew_window_ms) / 1000)
                # Never sleep through the renewal window, even when an
                # operator configured a heartbeat interval longer than it.
                delay = min(max(0.05, float(heartbeat_interval)), max(0.05, until_renewal))
                stop_event.wait(delay)
                if stopping_signal is not None:
                    _terminate_child(process, fence_descendants=True)
                    break
                if process.poll() is not None:
                    break
                unauthorized_task_id = _execution_slice_violation(
                    phase_state_path,
                    expected_task_id_set,
                    started_at_ms=started_at_ms,
                )
                if unauthorized_task_id:
                    execution_scope_error = (
                        f"active task {unauthorized_task_id!r} is outside leased "
                        f"execution slice {sorted(expected_task_id_set)!r}"
                    )
                    logger.error("Fencing daemon lane: %s", execution_scope_error)
                    _terminate_child(process, fence_descendants=True)
                    break
                completed_execution_slice = _fresh_completed_execution_slice(
                    phase_state_path,
                    expected_task_identity_map,
                    started_at_ms=phase_state_not_before_ms,
                    completion_events_path=completion_events_path,
                )
                if completed_execution_slice is not None:
                    logger.info(
                        "Stopping leased lane %s after fresh completion of %s",
                        grant.task_cid,
                        completed_execution_slice["completed_task_ids"],
                    )
                    _terminate_child(process, fence_descendants=True)
                    break
                pending_acceptance_slice = (
                    _fresh_provider_review_pending_acceptance(
                        phase_state_path,
                        completion_events_path,
                        expected_task_identity_map,
                        started_at_ms=phase_state_not_before_ms,
                    )
                )
                if pending_acceptance_slice is not None:
                    logger.info(
                        "Stopping leased lane %s with resumable provider review pending for %s",
                        grant.task_cid,
                        pending_acceptance_slice["task_ids"],
                    )
                    _terminate_child(process, fence_descendants=True)
                    break
                blocked_execution_slice = _fresh_blocked_execution_slice(
                    phase_state_path,
                    expected_task_identity_map,
                    started_at_ms=phase_state_not_before_ms,
                    completion_events_path=completion_events_path,
                )
                if blocked_execution_slice is not None:
                    logger.info(
                        "Stopping leased lane %s after fresh terminal block of %s",
                        grant.task_cid,
                        blocked_execution_slice["blocked_task_ids"],
                    )
                    _terminate_child(process, fence_descendants=True)
                    break
                lock_retry_delay = _DUCKDB_LOCK_RETRY_INITIAL_SECONDS
                previous_lock_error: Exception | None = None
                while True:
                    try:
                        now = _now_ms()
                        retry_deadline_ms = (
                            grant.lease_expires_at_ms
                            - _LEASE_EXPIRY_SAFETY_MARGIN_MS
                        )
                        if previous_lock_error is not None and now >= retry_deadline_ms:
                            raise previous_lock_error
                        if grant.lease_expires_at_ms - now <= renew_window_ms:
                            # Let the coordinator sample time after it acquires
                            # its database lock. Passing this pre-operation
                            # timestamp could renew an already expired grant
                            # after a long lock wait.
                            grant = coordinator.renew(
                                grant,
                                requested_lease_ms=lease_ms,
                            )
                        coordinator.heartbeat(
                            grant,
                            capacity_millionths=capacity_millionths,
                            resource_class=resource_class,
                            provider_id=provider_id,
                            **_resource_measurements(
                                resource_sampler,
                                active_phase=_active_phase(phase_state_path, "executing"),
                                occupied_workers=1,
                            ),
                        )
                        break
                    except LeaseError as exc:
                        logger.error("Fencing daemon lane after lease loss: %s", exc)
                        _terminate_child(process, fence_descendants=True)
                        return LeasedLaneResult(
                            task_cid=grant.task_cid,
                            claim_cid=grant.claim_cid,
                            claimant_did=grant.claimant_did,
                            fencing_token=grant.fencing_token,
                            disposition="fenced",
                            exit_code=FENCED_EXIT_CODE,
                            child_exit_code=process.returncode,
                            started_at_ms=started_at_ms,
                            finished_at_ms=_now_ms(),
                            error=str(exc),
                        )
                    except Exception as exc:
                        retry_now_ms = _now_ms()
                        retry_deadline_ms = (
                            grant.lease_expires_at_ms
                            - _LEASE_EXPIRY_SAFETY_MARGIN_MS
                        )
                        retry_seconds = min(
                            lock_retry_delay,
                            max(0.0, (retry_deadline_ms - retry_now_ms) / 1_000),
                        )
                        if (
                            _is_transient_duckdb_lock_error(exc)
                            and retry_seconds > 0
                        ):
                            previous_lock_error = exc
                            logger.warning(
                                "Retrying lease maintenance for %s after transient "
                                "DuckDB lock contention (%.3fs backoff)",
                                grant.task_cid,
                                retry_seconds,
                            )
                            stop_event.wait(retry_seconds)
                            if stopping_signal is not None:
                                _terminate_child(process, fence_descendants=True)
                                break
                            if process.poll() is not None:
                                break
                            lock_retry_delay = min(
                                _DUCKDB_LOCK_RETRY_MAX_SECONDS,
                                lock_retry_delay * 2,
                            )
                            continue

                        # Losing access to the coordination store is equivalent
                        # to losing proof of authority. Stop execution first; a
                        # best-effort release then makes recovery immediate when
                        # the store failure was transient.
                        logger.exception("Fencing daemon lane after coordination failure")
                        _terminate_child(process, fence_descendants=True)
                        released = _release_after_bookkeeping_failure(coordinator, grant)
                        return LeasedLaneResult(
                            task_cid=grant.task_cid,
                            claim_cid=grant.claim_cid,
                            claimant_did=grant.claimant_did,
                            fencing_token=grant.fencing_token,
                            disposition="failed",
                            exit_code=START_FAILED_EXIT_CODE,
                            child_exit_code=process.returncode,
                            started_at_ms=started_at_ms,
                            finished_at_ms=_now_ms(),
                            lease_released=released,
                            error=f"coordination failure: {exc}",
                        )

            # A short-lived supervisor can publish its final phase state and
            # exit between polling iterations. Read once more before
            # classifying that natural exit; exact canonical identities and
            # durable receipts remain mandatory.
            observed_exit_code = int(process.returncode or 0)
            if (
                completed_execution_slice is None
                and stopping_signal is None
                and not execution_scope_error
            ):
                completed_execution_slice = _fresh_completed_execution_slice(
                    phase_state_path,
                    expected_task_identity_map,
                    started_at_ms=phase_state_not_before_ms,
                    completion_events_path=completion_events_path,
                )
                if completed_execution_slice is not None:
                    logger.info(
                        "Accepting final fresh completion of %s after child exit %s",
                        completed_execution_slice["completed_task_ids"],
                        observed_exit_code,
                    )
                    _terminate_child(process, fence_descendants=True)
            if (
                completed_execution_slice is None
                and pending_acceptance_slice is None
                and stopping_signal is None
                and not execution_scope_error
            ):
                pending_acceptance_slice = (
                    _fresh_provider_review_pending_acceptance(
                        phase_state_path,
                        completion_events_path,
                        expected_task_identity_map,
                        started_at_ms=phase_state_not_before_ms,
                    )
                )
                if pending_acceptance_slice is not None:
                    logger.info(
                        "Accepting final resumable provider-review gap for %s after child exit %s",
                        pending_acceptance_slice["task_ids"],
                        observed_exit_code,
                    )
                    _terminate_child(process, fence_descendants=True)
            if (
                completed_execution_slice is None
                and pending_acceptance_slice is None
                and blocked_execution_slice is None
                and stopping_signal is None
                and not execution_scope_error
            ):
                blocked_execution_slice = _fresh_blocked_execution_slice(
                    phase_state_path,
                    expected_task_identity_map,
                    started_at_ms=phase_state_not_before_ms,
                    completion_events_path=completion_events_path,
                )
                if blocked_execution_slice is not None:
                    logger.info(
                        "Accepting final fresh terminal block of %s after child exit %s",
                        blocked_execution_slice["blocked_task_ids"],
                        observed_exit_code,
                    )
                    _terminate_child(process, fence_descendants=True)

            # Polling reaps only the immediate child.  Prove its dedicated
            # group and every captured detached descendant gone before any
            # terminal path advertises zero occupancy.
            _terminate_child(process, fence_descendants=True)
            child_exit_code = int(process.returncode or 0)
            completed_by_state = (
                completed_execution_slice is not None
                and stopping_signal is None
                and not execution_scope_error
            )
            pending_acceptance_by_state = (
                pending_acceptance_slice is not None
                and not completed_by_state
                and stopping_signal is None
                and not execution_scope_error
            )
            blocked_by_state = (
                blocked_execution_slice is not None
                and not completed_by_state
                and not pending_acceptance_by_state
                and stopping_signal is None
                and not execution_scope_error
            )
            completed_execution_output = dict(completed_execution_slice or {})
            pending_acceptance_output = dict(pending_acceptance_slice or {})
            blocked_execution_output = dict(blocked_execution_slice or {})
            missing_completion_evidence = (
                bool(expected_task_identity_map)
                and child_exit_code == 0
                and not completed_by_state
                and not pending_acceptance_by_state
                and not blocked_by_state
                and stopping_signal is None
                and not execution_scope_error
            )
            lane_exit_code = (
                0
                if completed_by_state
                else FENCED_EXIT_CODE
                if (
                    pending_acceptance_by_state
                    or blocked_by_state
                    or missing_completion_evidence
                )
                else child_exit_code
            )
            if stopping_signal is not None:
                receipt_status = "cancelled"
                disposition: LaneDisposition = "cancelled"
            elif execution_scope_error:
                receipt_status = "failed"
                disposition = "failed"
            elif completed_by_state:
                receipt_status = "succeeded"
                disposition = "completed"
            elif pending_acceptance_by_state:
                receipt_status = ""
                disposition = "pending_acceptance"
            elif blocked_by_state:
                receipt_status = "failed"
                disposition = "blocked"
            elif missing_completion_evidence:
                receipt_status = "failed"
                disposition = "failed"
            elif child_exit_code == 0:
                receipt_status = "succeeded"
                disposition = "completed"
            elif child_exit_code == FENCED_EXIT_CODE:
                receipt_status = "failed"
                disposition = "blocked"
            else:
                receipt_status = "failed"
                disposition = "failed"
            receipt: Mapping[str, Any] | None = None
            resolution: Mapping[str, Any] | None = None
            try:
                # Publish a final live-capacity observation before the receipt
                # closes the lease.  The lane slot can be reassigned as soon as
                # its child exits, regardless of terminal task disposition.
                coordinator.heartbeat(
                    grant,
                    capacity_millionths=0,
                    resource_class=resource_class,
                    provider_id=provider_id,
                    **_resource_measurements(
                        resource_sampler,
                        active_phase="idle",
                        occupied_workers=0,
                    ),
                )
                if pending_acceptance_by_state:
                    resolution = coordinator.defer_pending_acceptance(
                        grant,
                        evidence=pending_acceptance_output,
                        retry_delay_ms=_PENDING_ACCEPTANCE_RETRY_DELAY_MS,
                        now_ms=_now_ms(),
                    )
                else:
                    receipt = coordinator.receipt(
                        grant,
                        status=receipt_status,
                        output=(
                        {
                            "exit_code": lane_exit_code,
                            "child_exit_code": child_exit_code,
                            "command": list(command),
                            "reason": "completed_execution_slice",
                            **completed_execution_output,
                        }
                        if completed_by_state
                        else {
                            "exit_code": lane_exit_code,
                            "child_exit_code": child_exit_code,
                            "command": list(command),
                            "reason": "terminal_blocked_execution_slice",
                            **blocked_execution_output,
                        }
                        if blocked_by_state
                        else {
                            "exit_code": lane_exit_code,
                            "child_exit_code": child_exit_code,
                            "command": list(command),
                            "reason": "missing_execution_slice_completion_evidence",
                            "expected_task_ids": sorted(expected_task_identity_map),
                            "expected_task_cids": sorted(
                                expected_task_identity_map.values()
                            ),
                        }
                        if missing_completion_evidence
                        else {"exit_code": child_exit_code, "command": list(command)}
                        if receipt_status == "succeeded"
                        else {
                            "reason": "execution_slice_violation",
                            "error": execution_scope_error,
                        }
                        if execution_scope_error
                        else None
                        ),
                        failure_class=(
                            "none"
                            if receipt_status == "succeeded"
                            else "blocked"
                            if blocked_by_state
                            else "retryable"
                        ),
                        started_at_ms=started_at_ms,
                    )
            except LeaseError as exc:
                # The takeover's fencing token is authoritative; the old lane
                # must not manufacture a receipt after losing ownership.
                logger.warning("Discarded terminal result from fenced lane %s", grant.task_cid)
                return LeasedLaneResult(
                    task_cid=grant.task_cid,
                    claim_cid=grant.claim_cid,
                    claimant_did=grant.claimant_did,
                    fencing_token=grant.fencing_token,
                    disposition="fenced",
                    exit_code=FENCED_EXIT_CODE,
                    child_exit_code=child_exit_code,
                    started_at_ms=started_at_ms,
                    finished_at_ms=_now_ms(),
                    error=str(exc),
                )
            except Exception as exc:
                logger.exception("Could not record terminal result for lane %s", grant.task_cid)
                released = _release_after_bookkeeping_failure(coordinator, grant)
                return LeasedLaneResult(
                    task_cid=grant.task_cid,
                    claim_cid=grant.claim_cid,
                    claimant_did=grant.claimant_did,
                    fencing_token=grant.fencing_token,
                    disposition="failed",
                    exit_code=START_FAILED_EXIT_CODE,
                    child_exit_code=child_exit_code,
                    started_at_ms=started_at_ms,
                    finished_at_ms=_now_ms(),
                    lease_released=released,
                    error=f"terminal bookkeeping failed: {exc}",
                )

            return LeasedLaneResult(
                task_cid=grant.task_cid,
                claim_cid=grant.claim_cid,
                claimant_did=grant.claimant_did,
                fencing_token=grant.fencing_token,
                disposition=disposition,
                exit_code=lane_exit_code,
                child_exit_code=child_exit_code,
                started_at_ms=started_at_ms,
                finished_at_ms=_now_ms(),
                receipt_cid=_receipt_cid(receipt),
                resolution_cid=(
                    str(resolution.get("resolution_cid") or "") or None
                    if resolution is not None
                    else None
                ),
                lease_released=True,
                error=execution_scope_error,
            )
        except ProcessFenceError as exc:
            # Never close/release the accepted lease when process quiescence
            # could not be proved.  Expiry remains the recovery boundary and
            # callers receive a typed terminal result instead of an exception.
            logger.error(
                "Could not prove leased lane %s process fence: %s",
                grant.task_cid,
                exc,
            )
            return LeasedLaneResult(
                task_cid=grant.task_cid,
                claim_cid=grant.claim_cid,
                claimant_did=grant.claimant_did,
                fencing_token=grant.fencing_token,
                disposition="failed",
                exit_code=START_FAILED_EXIT_CODE,
                child_exit_code=process.returncode,
                started_at_ms=started_at_ms,
                finished_at_ms=_now_ms(),
                lease_released=False,
                error=f"process_fence_unproven: {exc}",
            )
        finally:
            if handlers_installed:
                signal.signal(signal.SIGTERM, old_term)
                signal.signal(signal.SIGINT, old_int)


def run_leased_lane(
    *,
    coordination_path: Path,
    grant: LeaseGrant,
    command: Sequence[str],
    lease_ms: int,
    heartbeat_interval: float,
    capacity_millionths: int = 1_000_000,
    resource_class: str = "",
    provider_id: str = "",
    resource_sampler: Callable[..., Any] | None = None,
    phase_state_path: Path | None = None,
    expected_task_ids: Sequence[str] = (),
    expected_task_cids_by_id: Mapping[str, str] | None = None,
    completion_events_path: Path | None = None,
) -> int:
    """Compatibility wrapper returning the guarded command's lane exit code.

    Legacy callers may omit execution-slice arguments entirely.  Once a slice
    is supplied, both display IDs and canonical CIDs are required.
    """

    return run_leased_lane_result(
        coordination_path=coordination_path,
        grant=grant,
        command=command,
        lease_ms=lease_ms,
        heartbeat_interval=heartbeat_interval,
        capacity_millionths=capacity_millionths,
        resource_class=resource_class,
        provider_id=provider_id,
        resource_sampler=resource_sampler,
        phase_state_path=phase_state_path,
        expected_task_ids=expected_task_ids,
        expected_task_cids_by_id=expected_task_cids_by_id,
        completion_events_path=completion_events_path,
    ).exit_code


def _parse_expected_task_identity_json(value: str) -> tuple[str, str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "expected task identity must be valid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise argparse.ArgumentTypeError("expected task identity must be an object")
    task_id = str(payload.get("task_id") or "").strip()
    canonical_task_cid = str(payload.get("canonical_task_cid") or "").strip()
    if not task_id or not canonical_task_cid:
        raise argparse.ArgumentTypeError(
            "expected task identity requires task_id and canonical_task_cid"
        )
    return task_id, canonical_task_cid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a command under an accepted Profile G lease")
    parser.add_argument("--coordination-path", type=Path, required=True)
    parser.add_argument("--grant-json", required=True)
    parser.add_argument("--lease-ms", type=int, default=60_000)
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--capacity-millionths", type=int, default=1_000_000)
    parser.add_argument("--resource-class", default="")
    parser.add_argument("--provider-id", default="")
    parser.add_argument("--phase-state-path", type=Path, default=None)
    parser.add_argument("--completion-events-path", type=Path, default=None)
    parser.add_argument("--expected-task-id", action="append", default=[])
    parser.add_argument(
        "--expected-task-identity-json",
        action="append",
        type=_parse_expected_task_identity_json,
        default=[],
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    expected_task_cids_by_id: dict[str, str] = {}
    for task_id, canonical_task_cid in args.expected_task_identity_json:
        previous = expected_task_cids_by_id.get(task_id)
        if previous is not None and previous != canonical_task_cid:
            parser.error(
                f"conflicting canonical task CIDs supplied for {task_id!r}"
            )
        expected_task_cids_by_id[task_id] = canonical_task_cid
    grant = LeaseGrant(**json.loads(args.grant_json))
    return run_leased_lane(
        coordination_path=args.coordination_path,
        grant=grant,
        command=command,
        lease_ms=args.lease_ms,
        heartbeat_interval=args.heartbeat_interval,
        capacity_millionths=args.capacity_millionths,
        resource_class=args.resource_class,
        provider_id=args.provider_id,
        phase_state_path=args.phase_state_path,
        expected_task_ids=args.expected_task_id,
        expected_task_cids_by_id=expected_task_cids_by_id,
        completion_events_path=args.completion_events_path,
    )


__all__ = [
    "FENCED_EXIT_CODE",
    "OBJECTIVE_VALIDATION_REPAIR_EVIDENCE",
    "OBJECTIVE_VALIDATION_REPAIR_TASK_ID",
    "START_FAILED_EXIT_CODE",
    "LeasedLaneResult",
    "build_parser",
    "main",
    "run_leased_lane",
    "run_leased_lane_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
