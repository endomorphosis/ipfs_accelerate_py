"""Plan and launch per-bundle todo daemon lanes."""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import logging
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..core.conflict_graph import materialize_task_conflict_graph
from ..merge.lease_coordination import (
    DistributedLaneDispatch,
    ImmutableLaneInputArtifact,
    LeaseCoordinator,
    LeaseError,
    RemoteLaneResult,
    WorkerCapabilityReceipt,
    WorkerEnvironmentReceipt,
)
from ..runtime.artifact_store import (
    BUNDLE_INDEX_KIND,
    read_artifact_fields,
    write_scheduler_manifest_artifact,
)
from ..runtime.event_log import event_log_sources, read_jsonl_events
from ..runtime.provider_capacity_snapshot import (
    DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS,
    DUAL_REVIEW_PROVIDER_ID,
    DUAL_REVIEW_REQUIRED_CAPABILITIES,
    load_provider_capacity_snapshot,
    synthesize_dual_review_provider_capacity,
)
from ..runtime.resource_scheduler import (
    ADAPTIVE_STAGES,
    AdmissionDecision,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ResourceAdmissionLease,
    ResourcePolicy,
    ResourceScheduler,
    ResourceScheduleSnapshot,
    normalize_adaptive_stage,
    sample_host_resources,
)
from ..runtime.scheduler_metrics import (
    SchedulerSnapshot,
    scheduler_snapshot,
    scheduler_state_events,
    write_scheduler_snapshot,
)
from ..todo_daemon.implementation_timeout import (
    effective_implementation_hard_timeout,
)
from ..todo_daemon.legacy_landed_attestation import LegacyLandedReviewAuthority
from ..todo_daemon.legacy_landed_review import (
    LegacyLandedReviewPolicy,
    load_legacy_landed_review_policy,
)
from ..todo_daemon.production_provider_attestation import (
    DEFAULT_PRODUCTION_PROVIDER_REVIEW_KEY_NAME,
)
from ..todo_daemon.production_provider_cli import (
    DEFAULT_CONTEXT_BUDGET_TOKENS as DEFAULT_PRODUCTION_CONTEXT_BUDGET_TOKENS,
)
from ..todo_daemon.production_provider_cli import (
    DEFAULT_PROVIDER_TIMEOUT_SECONDS as DEFAULT_PRODUCTION_PROVIDER_TIMEOUT_SECONDS,
)
from ..todo_daemon.production_provider_cli import (
    PRODUCTION_CLI_POLICY_NAME,
    ProductionCLIProviderPolicy,
)
from ..todo_daemon.supervisor import active_codex_exec_workers
from .bundle_optimizer import BundleOptimizationPolicy, optimize_task_bundles
from .objective_graph import (
    DEFAULT_TASK_PREFIX,
    build_bundle_task_payloads,
    profile_g_safe_planning_value,
    repo_relative_path,
    safe_bundle_key,
    utc_now,
)

logger = logging.getLogger(__name__)

COORDINATION_COMPACTION_INTERVAL_CYCLES = 10
COORDINATION_COMPACTION_MIN_BYTES = 64 * 1024 * 1024
SCHEDULER_GC_INTERVAL_CYCLES = 10
_TASK_ATTEMPT_LIMIT_IDLE_REASON = (
    "all_selectable_ready_tasks_reached_max_task_attempts"
)
BUNDLE_TASKBOARD_INPUT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.bundle_taskboard_input@1"
)
INTERNAL_EXECUTION_AUTHORITY = "agent-supervisor/v1"
DISTRIBUTED_LANE_REQUIREMENT_ID = "314703454108352614663943447510592855908"
DISTRIBUTED_LANE_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/distributed-lane-evidence@1"
)
DISTRIBUTED_LANE_PUBLICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/distributed-lane-publication@1"
)
_DISTRIBUTED_LANE_EVIDENCE_SEAL = object()

_MANIFEST_REFERENCED_BUNDLE_FIELDS = frozenset(
    {
        "conflict_graph",
        "conflict_planning_decisions",
        "dependency_dag",
        "task_conflict_graph",
        "task_dependency_graph",
    }
)

_MANIFEST_MEMBER_TASK_FIELDS = frozenset(
    {
        "blocking_task_cids",
        "canonical_task_cid",
        "depends_on",
        "dependency_task_cids",
        "goal_id",
        "parent_goal_id",
        "priority",
        "status",
        "subgoal_id",
        "task_cid",
        "task_id",
        "title",
    }
)

_MANIFEST_PROFILE_G_REFERENCE_FIELDS = frozenset(
    {
        "canonical_task_cid",
        "canonical_task_key",
        "dependency_repair_evidence",
        "dependency_repair_evidence_count",
        "goal_cid",
        "plan_branch_cid",
        "selection_cid",
        "subgoal_cid",
        "task_cid",
        "task_spec_cid",
    }
)


def bundle_member_completion_event_sources(state_root: Path) -> tuple[Path, ...]:
    """Return active and rotated member-completion logs in stable order."""

    base_paths: set[Path] = set()
    for pattern in ("*_events.jsonl*", "*/state/*_events.jsonl*"):
        for candidate in state_root.glob(pattern):
            name = candidate.name
            if name.endswith("_events.jsonl"):
                pass
            elif ".jsonl.rotated-" in name:
                name = name.split(".rotated-", 1)[0]
                candidate = candidate.with_name(name)
            else:
                continue
            base_paths.add(candidate)
    return tuple(event_log_sources(sorted(base_paths), include_rotated=True))


def bundle_member_completion_source_revision(
    state_root: Path,
) -> tuple[tuple[str, int, int], ...]:
    """Return the dynamic source revision used to fence receipt-plan caching."""

    revisions: list[tuple[str, int, int]] = []
    for source in bundle_member_completion_event_sources(state_root):
        try:
            stat = source.stat()
        except OSError:
            continue
        revisions.append((str(source), stat.st_mtime_ns, stat.st_size))
    return tuple(revisions)


def _legacy_completed_member_identities(
    todo_path: Any,
    task_ids: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Recover explicit canonical identities from a legacy generated board."""

    path = Path(str(todo_path or ""))
    if not str(todo_path or "").strip() or path.suffix.lower() not in {".md", ".markdown"}:
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    requested = {str(task_id) for task_id in task_ids if str(task_id)}
    identities: dict[str, dict[str, str]] = {}
    current_task_id = ""
    current: dict[str, str] = {}

    def flush() -> None:
        if (
            current_task_id in requested
            and current.get("canonical_task_cid")
            and current.get("canonical_task_key")
        ):
            identities[current_task_id] = {
                "task_id": current_task_id,
                "canonical_task_cid": current["canonical_task_cid"],
                "canonical_task_key": current["canonical_task_key"],
                "board_namespace": current.get("board_namespace", path.name),
            }

    for line in [*lines, "## "]:
        if line.startswith("## "):
            flush()
            header = line[3:].strip()
            current_task_id = header.split(" ", 1)[0] if header else ""
            current = {}
            continue
        if current_task_id not in requested:
            continue
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        if normalized_key in {
            "canonical_task_cid",
            "canonical_task_key",
            "board_namespace",
        }:
            current[normalized_key] = value.strip()
    return identities


def bundle_member_completion_receipts(state_root: Path) -> dict[str, dict[str, Any]]:
    """Return successful member-task receipts keyed by canonical task identity.

    Reviewed bundle shards are immutable inputs and bundle boards are mutable
    operational projections, so their current status is not a durable
    completion authority.  The implementation daemon writes operational copies
    and emits terminal events after a successful merge; retain those receipts
    so a source board can promote the matching canonical task even after a
    shard is regenerated without rewriting the reviewed shard.

    New events carry one canonical ``completion_receipts`` entry per member.
    A successful merge event is only terminal when it also carries the
    authoritative acceptance packet that performed that board mutation.  A
    raw ``returncode == 0``/``merged == true`` pair is integration evidence,
    not completion authority, and must not drain a regenerated execution
    lane.
    Legacy packet-aggregate events only carried the primary canonical CID plus
    ``updated_task_ids``/``already_completed_task_ids``.  For those events,
    explicit canonical identities are recovered from the generated board named
    by the terminal event.  A display ID alone never manufactures a receipt.
    """

    receipts: dict[str, dict[str, Any]] = {}
    for source in bundle_member_completion_event_sources(state_root):
        for event in read_jsonl_events(source):
            event_type = str(event.get("type") or "")
            task_id = str(event.get("task_id") or "")
            acceptance_result = (
                event.get("acceptance_result")
                if isinstance(event.get("acceptance_result"), Mapping)
                else {}
            )
            todo_update = (
                acceptance_result.get("todo_update_result")
                if isinstance(acceptance_result.get("todo_update_result"), Mapping)
                else {}
            )
            completed = False
            if event_type == "todo_status_updated":
                completed_ids = {
                    str(item)
                    for key in ("updated_task_ids", "already_completed_task_ids")
                    for item in (event.get(key) or [])
                }
                completed = bool(event.get("updated")) or bool(task_id and task_id in completed_ids)
            elif event_type == "implementation_finished":
                merge_result = event.get("merge_result")
                authoritative_member_receipts = (
                    todo_update.get("completion_receipts")
                    or todo_update.get("member_completion_receipts")
                )
                authoritative_completed_ids = {
                    str(item)
                    for key in ("updated_task_ids", "already_completed_task_ids")
                    for item in (todo_update.get(key) or [])
                    if str(item)
                }
                task_has_canonical_receipt = any(
                    isinstance(item, Mapping)
                    and str(item.get("task_id") or "") == task_id
                    and bool(str(item.get("canonical_task_cid") or ""))
                    for item in (
                        authoritative_member_receipts
                        if isinstance(authoritative_member_receipts, list)
                        else []
                    )
                )
                completed = (
                    event.get("returncode") == 0
                    and isinstance(merge_result, dict)
                    and merge_result.get("merged") is True
                    and acceptance_result.get("authoritatively_completed") is True
                    and acceptance_result.get("completion_authoritative") is True
                    and not acceptance_result.get("pending_gates")
                    and task_id in authoritative_completed_ids
                    and task_has_canonical_receipt
                )
            if not completed:
                continue

            if event_type == "todo_status_updated":
                todo_update = event
            completion_payload = event if event_type == "todo_status_updated" else todo_update
            raw_member_receipts = completion_payload.get(
                "completion_receipts"
            ) or completion_payload.get("member_completion_receipts")
            member_task_ids = list(
                dict.fromkeys(
                    [
                        task_id,
                        *(
                            str(item)
                            for key in (
                                "updated_task_ids",
                                "already_completed_task_ids",
                            )
                            for item in (completion_payload.get(key) or [])
                            if str(item)
                        ),
                    ]
                )
            )
            explicit_by_task_id = {
                str(item.get("task_id") or ""): dict(item)
                for item in raw_member_receipts
                if isinstance(item, Mapping)
                and str(item.get("task_id") or "") in member_task_ids
            } if isinstance(raw_member_receipts, list) else {}
            # Backward-compatible recovery for terminal aggregate events
            # written before per-member canonical receipts were included. It
            # also safely fills a partial new event after a transient board
            # read failure at write time.
            legacy_identities = _legacy_completed_member_identities(
                completion_payload.get("path"),
                member_task_ids,
            )
            member_receipts: list[dict[str, Any]] = []
            for member_task_id in member_task_ids:
                member = explicit_by_task_id.get(member_task_id, {})
                if not str(member.get("canonical_task_cid") or ""):
                    if (
                        member_task_id == task_id
                        and str(event.get("canonical_task_cid") or "")
                    ):
                        member = {
                            "task_id": member_task_id,
                            "canonical_task_cid": str(
                                event.get("canonical_task_cid") or ""
                            ),
                            "canonical_task_key": str(
                                event.get("canonical_task_key") or ""
                            ),
                            "board_namespace": str(
                                event.get("board_namespace") or ""
                            ),
                        }
                    else:
                        member = legacy_identities.get(member_task_id, {})
                if member:
                    member_receipts.append(member)

            for member in member_receipts:
                member_task_id = str(member.get("task_id") or "")
                canonical_task_cid = str(member.get("canonical_task_cid") or "")
                if not canonical_task_cid:
                    continue
                receipt = {
                    "canonical_task_cid": canonical_task_cid,
                    "canonical_task_key": str(member.get("canonical_task_key") or ""),
                    "task_id": member_task_id,
                    "board_namespace": str(member.get("board_namespace") or ""),
                    "status": "succeeded",
                    "timestamp": str(event.get("timestamp") or ""),
                    "event_type": event_type,
                    "event_path": str(source),
                }
                previous = receipts.get(canonical_task_cid)
                if previous is None or receipt["timestamp"] >= previous["timestamp"]:
                    receipts[canonical_task_cid] = receipt
    return receipts


LEGACY_ADOPTION_BARRIER_REASON = "legacy_adoption_incomplete"
LEGACY_ADOPTION_BARRIER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.legacy_adoption_barrier@1"
)


def _legacy_adoption_planning_receipts(
    policy: LegacyLandedReviewPolicy,
    completion_receipts: Mapping[str, Any],
) -> dict[str, Any]:
    """Exclude malformed exact-policy receipts before planner overlay."""

    expected = {
        task.canonical_task_cid: task.task_id for task in policy.tasks
    }
    admitted: dict[str, Any] = {}
    for receipt_cid, receipt in completion_receipts.items():
        task_id = expected.get(str(receipt_cid))
        if task_id is None:
            admitted[str(receipt_cid)] = receipt
            continue
        if not isinstance(receipt, Mapping) or (
            str(receipt.get("task_id") or "") != task_id
            or str(receipt.get("canonical_task_cid") or "")
            != str(receipt_cid)
            or str(receipt.get("status") or "").strip().casefold()
            != "succeeded"
        ):
            continue
        admitted[str(receipt_cid)] = receipt
    return admitted


def _legacy_adoption_barrier_payloads(
    payloads: Sequence[dict[str, Any]],
    *,
    policy: LegacyLandedReviewPolicy,
    completion_receipts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Constrain every execution slice until all exact policy CIDs complete.

    This runs on the same stable durable-receipt snapshot used to plan bundle
    lanes.  It therefore precedes coordinator registration, resource/provider
    admission, lease claims, expected-ID binding, and child process launch.
    Mutable board status and transient live-lane manifests are never accepted
    as migration completion authority.
    """

    expected = {
        task.task_id: task.canonical_task_cid for task in policy.tasks
    }
    if len(expected) != 8 or len(expected) != len(policy.tasks):
        raise ValueError(
            "legacy adoption policy must contain eight unique task identities"
        )
    inventory: dict[str, list[str]] = {
        task_id: [] for task_id in expected
    }
    for payload in payloads:
        for task in _mapping_list(payload.get("tasks")):
            task_id = str(task.get("task_id") or "").strip()
            if task_id not in expected:
                continue
            inventory[task_id].append(
                str(
                    task.get("canonical_task_cid")
                    or task.get("task_cid")
                    or ""
                ).strip()
            )
    invalid_inventory = [
        task_id
        for task_id, expected_cid in expected.items()
        if inventory[task_id] != [expected_cid]
    ]
    if invalid_inventory:
        raise ValueError(
            "legacy adoption exact task inventory is invalid: "
            + ", ".join(invalid_inventory)
        )

    completed: list[str] = []
    invalid_receipts: list[str] = []
    for task_id, task_cid in expected.items():
        receipt = completion_receipts.get(task_cid)
        if receipt is None:
            continue
        if not isinstance(receipt, Mapping) or (
            str(receipt.get("task_id") or "") != task_id
            or str(receipt.get("canonical_task_cid") or "") != task_cid
            or str(receipt.get("status") or "").strip().casefold()
            != "succeeded"
        ):
            invalid_receipts.append(task_id)
            continue
        completed.append(task_id)
    completed_set = set(completed)
    remaining = [
        task_id for task_id in expected if task_id not in completed_set
    ]
    remaining_set = set(remaining)
    barrier = {
        "schema": LEGACY_ADOPTION_BARRIER_SCHEMA,
        "active": bool(remaining),
        "reason": (
            LEGACY_ADOPTION_BARRIER_REASON
            if remaining
            else "legacy_adoption_complete"
        ),
        "policy_id": policy.policy_id,
        "completion_authority": (
            "durable_exact_member_completion_receipts"
        ),
        "policy_task_ids": list(expected),
        "policy_task_cids": [expected[task_id] for task_id in expected],
        "completed_task_ids": completed,
        "remaining_task_ids": remaining,
        "remaining_task_cids": [expected[task_id] for task_id in remaining],
        "invalid_receipt_task_ids": invalid_receipts,
    }
    projected: list[dict[str, Any]] = []
    for original in payloads:
        payload = dict(original)
        payload["legacy_adoption_barrier"] = dict(barrier)
        if not remaining:
            tasks = _mapping_list(payload.get("tasks"))
            authorized = _execution_slice_members(payload, tasks)
            has_policy_members = any(
                str(task.get("task_id") or "") in expected
                for task in tasks
            )
            if has_policy_members and not authorized:
                # Receipt-drained policy lanes are retained for planning
                # visibility but can never register as fresh executable work,
                # including when bundle optimization is explicitly disabled.
                payload["claimable"] = False
            projected.append(payload)
            continue
        tasks = _mapping_list(payload.get("tasks"))
        authorized = _execution_slice_members(payload, tasks)
        pending = [
            task
            for task in authorized
            if (
                str(task.get("task_id") or "") in remaining_set
                and expected.get(str(task.get("task_id") or ""))
                == str(
                    task.get("canonical_task_cid")
                    or task.get("task_cid")
                    or ""
                )
            )
        ]
        payload["execution_slice_task_ids"] = [
            str(task.get("task_id") or "") for task in pending
        ]
        payload["execution_slice_task_cids"] = [
            str(
                task.get("canonical_task_cid")
                or task.get("task_cid")
                or ""
            )
            for task in pending
        ]
        if not pending:
            payload["claimable"] = False
            payload["blocked_reason"] = LEGACY_ADOPTION_BARRIER_REASON
        projected.append(payload)
    return projected


@dataclass(frozen=True)
class BundleLaneSpec:
    """One isolated daemon/supervisor lane for an objective bundle shard.

    ``expected_task_cids_by_id`` binds the mutable display IDs used by the
    implementation daemon to the canonical member identities admitted by the
    bundle planner.  Local execution wrappers must carry that binding across
    the process boundary before phase state can prove a slice complete.
    """

    bundle_key: str
    parallel_lane: str
    todo_path: Path
    state_dir: Path
    worktree_root: Path
    state_prefix: str
    task_ids: list[str]
    conflict_policy: str
    command: list[str]
    log_path: Path
    runtime_todo_path: Path | None = None
    source_todo_sha256: str = ""
    source_todo: str = ""
    task_cid: str = ""
    goal_cid: str = ""
    subgoal_cid: str = ""
    queue_payload: dict[str, Any] | None = None
    schedule_rank: int | None = None
    claimable: bool = True
    dependency_task_cids: list[str] = field(default_factory=list)
    blocking_task_cids: list[str] = field(default_factory=list)
    critical_path_length: int = 0
    slack: int = 0
    downstream_unlock_value: int = 0
    age_seconds: int = 0
    objective_priority: int = 0
    schedule_score: int = 0
    dependency_repair_evidence: list[dict[str, Any]] = field(default_factory=list)
    conflict_color: int | None = None
    conflicting_task_ids: list[str] = field(default_factory=list)
    conflict_decisions: list[dict[str, Any]] = field(default_factory=list)
    conflict_surface: dict[str, Any] = field(default_factory=dict)
    resource_class: str = "cpu-small"
    resource_stage: str = "analysis"
    required_capabilities: list[str] = field(default_factory=list)
    llm_provider: str = ""
    required_context_tokens: int = 0
    token_budget: int = 0
    max_provider_latency_ms: int = 0
    memory_bytes: int = 0
    gpu_memory_bytes: int = 0
    disk_bytes: int = 0
    process_slots: int = 1
    implementation_max_timeout: float = 1800.0
    optimizer_bundle_cid: str = ""
    optimizer_policy_id: str = ""
    optimizer_execution_wave: int = 0
    optimization_metrics: dict[str, int] = field(default_factory=dict)
    planner_comparison: dict[str, Any] = field(default_factory=dict)
    packet_aggregates: list[dict[str, Any]] = field(default_factory=list)
    expected_task_cids_by_id: dict[str, str] = field(default_factory=dict)

    def to_dict(self, *, repo_root: Path | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for definition in fields(self):
            value = getattr(self, definition.name)
            if isinstance(value, dict):
                value = dict(value)
            elif isinstance(value, list):
                value = list(value)
            payload[definition.name] = value
        for key in (
            "todo_path",
            "runtime_todo_path",
            "state_dir",
            "worktree_root",
            "log_path",
        ):
            if payload[key] is None:
                continue
            path = Path(payload[key])
            payload[key] = (
                repo_relative_path(repo_root, path)
                if repo_root is not None
                else str(path)
            )
        return payload


def _distributed_lane_digest(value: Mapping[str, Any]) -> str:
    """Return the stable digest used by distributed receipts and publications."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _evidence_rows_by_task(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        task_cid = str(
            row.get("canonical_task_cid")
            or row.get("task_cid")
            or row.get("canonical_task_id")
            or ""
        ).strip()
        if task_cid:
            indexed[task_cid] = dict(row)
    return indexed


def _distributed_lane_evidence_failures(
    *,
    repository_tree: str,
    task_population: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    resources: Sequence[Mapping[str, Any]],
    ownership: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    terminal_results: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Re-evaluate whether evidence covers every canonical task exactly once."""

    def positive_integer(value: Any) -> bool:
        try:
            return not isinstance(value, bool) and int(value) > 0
        except (TypeError, ValueError):
            return False

    failures: list[str] = []
    if not str(repository_tree).strip():
        failures.append("repository_tree_missing")
    population = _evidence_rows_by_task(task_population)
    if not population:
        failures.append("task_population_missing")
    if len(population) != len(task_population):
        failures.append("task_population_not_canonical")
    surface_rows = {
        "effects": effects,
        "resources": resources,
        "ownership": ownership,
        "validation": validation,
        "terminal_results": terminal_results,
    }
    surfaces = {
        name: _evidence_rows_by_task(rows)
        for name, rows in surface_rows.items()
    }
    population_cids = set(population)
    for name, indexed in surfaces.items():
        rows = surface_rows[name]
        if set(indexed) != population_cids or len(indexed) != len(rows):
            failures.append(f"{name}_coverage_incomplete")
    for task_cid in sorted(population_cids):
        effect = surfaces["effects"].get(task_cid, {})
        resource = surfaces["resources"].get(task_cid, {})
        owner = surfaces["ownership"].get(task_cid, {})
        checked = surfaces["validation"].get(task_cid, {})
        terminal = surfaces["terminal_results"].get(task_cid, {})
        if not effect.get("paths") and not effect.get("symbols") and not effect.get(
            "interfaces"
        ):
            failures.append(f"effect_binding_missing:{task_cid}")
        if not str(resource.get("resource_class") or "").strip():
            failures.append(f"resource_binding_missing:{task_cid}")
        if (
            not str(owner.get("claim_cid") or "").strip()
            or not positive_integer(
                owner.get("logical_epoch") or owner.get("fencing_epoch")
            )
            or not positive_integer(owner.get("fencing_token"))
            or not str(owner.get("input_artifact_id") or owner.get("artifact_id") or "").strip()
            or not str(owner.get("capability_receipt_id") or "").strip()
            or not str(owner.get("environment_receipt_id") or "").strip()
        ):
            failures.append(f"ownership_binding_missing:{task_cid}")
        if checked.get("passed") is not True or not (
            checked.get("receipt_ids") or checked.get("validation_receipt_ids")
        ):
            failures.append(f"validation_binding_missing:{task_cid}")
        if terminal.get("accepted") is not True or not str(
            terminal.get("candidate_commit")
            or terminal.get("accepted_commit")
            or terminal.get("target_commit")
            or ""
        ).strip():
            failures.append(f"terminal_result_missing:{task_cid}")
    return tuple(dict.fromkeys(failures))


@dataclass(frozen=True)
class DistributedLaneEvidenceReceipt:
    """Authoritative, repository-tree-bound evidence for distributed lanes.

    Remote completion messages are deliberately insufficient.  Authority is
    exposed only by a receipt constructed by the local acceptance boundary and
    only when every canonical task has exact effects, resources, lease
    ownership, validation receipts, and a merge-gated terminal result.
    """

    repository_tree: str
    task_population: tuple[dict[str, Any], ...]
    effects: tuple[dict[str, Any], ...]
    resources: tuple[dict[str, Any], ...]
    ownership: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    terminal_results: tuple[dict[str, Any], ...]
    failure_codes: tuple[str, ...]
    content_id: str
    schema: str = DISTRIBUTED_LANE_EVIDENCE_SCHEMA
    requirement_id: str = DISTRIBUTED_LANE_REQUIREMENT_ID
    _producer_seal: object | None = field(default=None, compare=False, repr=False)

    def _content(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "requirement_id": self.requirement_id,
            "repository_tree": self.repository_tree,
            "task_population": [dict(item) for item in self.task_population],
            "effects": [dict(item) for item in self.effects],
            "resources": [dict(item) for item in self.resources],
            "ownership": [dict(item) for item in self.ownership],
            "validation": [dict(item) for item in self.validation],
            "terminal_results": [dict(item) for item in self.terminal_results],
            "failure_codes": list(self.failure_codes),
        }

    def verify_integrity(self) -> bool:
        return self.content_id == _distributed_lane_digest(self._content())

    def proved_requirement_ids_for(
        self, repository_tree: str
    ) -> tuple[str, ...]:
        failures = _distributed_lane_evidence_failures(
            repository_tree=repository_tree,
            task_population=self.task_population,
            effects=self.effects,
            resources=self.resources,
            ownership=self.ownership,
            validation=self.validation,
            terminal_results=self.terminal_results,
        )
        if (
            self._producer_seal is _DISTRIBUTED_LANE_EVIDENCE_SEAL
            and self.schema == DISTRIBUTED_LANE_EVIDENCE_SCHEMA
            and self.requirement_id == DISTRIBUTED_LANE_REQUIREMENT_ID
            and self.repository_tree == str(repository_tree).strip()
            and not self.failure_codes
            and not failures
            and self.verify_integrity()
        ):
            return (DISTRIBUTED_LANE_REQUIREMENT_ID,)
        return ()

    def to_dict(self) -> dict[str, Any]:
        payload = self._content()
        proved = self.proved_requirement_ids_for(self.repository_tree)
        payload.update(
            {
                "content_id": self.content_id,
                "passed": bool(proved),
                "proved_requirement_ids": list(proved),
                "source_tier": "post-merge-validation",
            }
        )
        return payload

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "DistributedLaneEvidenceReceipt":
        """Restore a diagnostic projection without minting local authority."""

        return cls(
            repository_tree=str(value.get("repository_tree") or ""),
            task_population=tuple(
                dict(item)
                for item in value.get("task_population", ())
                if isinstance(item, Mapping)
            ),
            effects=tuple(
                dict(item)
                for item in value.get("effects", ())
                if isinstance(item, Mapping)
            ),
            resources=tuple(
                dict(item)
                for item in value.get("resources", ())
                if isinstance(item, Mapping)
            ),
            ownership=tuple(
                dict(item)
                for item in value.get("ownership", ())
                if isinstance(item, Mapping)
            ),
            validation=tuple(
                dict(item)
                for item in value.get("validation", ())
                if isinstance(item, Mapping)
            ),
            terminal_results=tuple(
                dict(item)
                for item in value.get("terminal_results", ())
                if isinstance(item, Mapping)
            ),
            failure_codes=tuple(
                str(item) for item in value.get("failure_codes", ())
            ),
            content_id=str(value.get("content_id") or ""),
            schema=str(value.get("schema") or ""),
            requirement_id=str(value.get("requirement_id") or ""),
        )


def evaluate_distributed_lane_evidence(
    *,
    repository_tree: str,
    task_population: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    resources: Sequence[Mapping[str, Any]],
    ownership: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    terminal_results: Sequence[Mapping[str, Any]],
) -> DistributedLaneEvidenceReceipt:
    """Construct local evidence after merge and post-merge validation."""

    def detached(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        payload = json.loads(
            json.dumps(
                [dict(item) for item in rows],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return tuple(payload)

    normalized = {
        "repository_tree": str(repository_tree).strip(),
        "task_population": detached(task_population),
        "effects": detached(effects),
        "resources": detached(resources),
        "ownership": detached(ownership),
        "validation": detached(validation),
        "terminal_results": detached(terminal_results),
    }
    failures = _distributed_lane_evidence_failures(**normalized)
    receipt = DistributedLaneEvidenceReceipt(
        **normalized,
        failure_codes=failures,
        content_id="",
        _producer_seal=_DISTRIBUTED_LANE_EVIDENCE_SEAL,
    )
    return replace(receipt, content_id=_distributed_lane_digest(receipt._content()))


def _taskboard_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def bundle_taskboard_input_binding_path(lane: BundleLaneSpec) -> Path:
    """Return the durable source-to-runtime taskboard binding for one lane."""

    return lane.state_dir / f"{lane.state_prefix}_taskboard_input.json"


def stale_bundle_lane_input_binding(
    lane: BundleLaneSpec,
    *,
    repo_root: Path,
) -> dict[str, str] | None:
    """Diagnose a planned lane whose immutable runtime input is from an older plan.

    Runtime taskboards are deliberately mutable after materialization because
    the implementation daemon records task status there. Their accompanying
    input binding is immutable, however. If a later plan points the same lane
    state directory at different source bytes, launching it would only fail
    inside :func:`materialize_bundle_lane_taskboard` after a coordination lease
    and resource reservation had already been acquired.

    This read-only preflight recognizes that one actionable mismatch without
    rewriting either the binding or the runtime taskboard. Other malformed
    binding conditions continue through the existing fail-closed materializer.
    Callers that can safely rematerialize should use
    :func:`refresh_stale_bundle_lane_input_binding`.
    """

    planned_digest = str(lane.source_todo_sha256 or "").strip().lower()
    if len(planned_digest) != 64:
        return None
    binding_path = bundle_taskboard_input_binding_path(lane)
    try:
        existing_binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(existing_binding, dict):
        return None
    bound_digest = str(
        existing_binding.get("source_todo_sha256") or ""
    ).strip().lower()
    if not bound_digest or bound_digest == planned_digest:
        return None
    return {
        "reason": "stale_input_binding",
        "binding_path": repo_relative_path(repo_root, binding_path),
        "bound_source_todo_sha256": bound_digest,
        "planned_source_todo_sha256": planned_digest,
    }


def refresh_stale_bundle_lane_input_binding(
    lane: BundleLaneSpec,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    """Archive a stale binding/runtime pair and rematerialize from the plan.

    Used when the source board advanced (e.g. operator or soft-complete status
    flips) while the lane state directory still points at the previous digest.
    Safe only when the lane has no live worker holding the runtime board.
    Returns ``None`` when the binding is not stale; otherwise a report of the
    refresh (``refreshed`` true on success).
    """

    diagnosis = stale_bundle_lane_input_binding(lane, repo_root=repo_root)
    if diagnosis is None:
        return None
    binding_path = bundle_taskboard_input_binding_path(lane)
    runtime_path = lane.runtime_todo_path
    stamp = utc_now().replace(":", "").replace("+", "_")
    archived: list[str] = []
    try:
        if binding_path.is_file():
            archive = binding_path.with_name(f"{binding_path.name}.stale-{stamp}")
            os.replace(binding_path, archive)
            archived.append(str(archive))
        if runtime_path is not None and Path(runtime_path).is_file():
            runtime = Path(runtime_path)
            archive = runtime.with_name(f"{runtime.name}.stale-{stamp}")
            os.replace(runtime, archive)
            archived.append(str(archive))
        binding = materialize_bundle_lane_taskboard(lane, repo_root=repo_root)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Failed to refresh stale input binding for %s: %s",
            lane.bundle_key,
            exc,
        )
        return {
            **diagnosis,
            "refreshed": False,
            "error": str(exc)[-1000:],
            "archived": archived,
        }
    logger.info(
        "Refreshed stale input binding for %s (bound %s → planned %s)",
        lane.bundle_key,
        diagnosis.get("bound_source_todo_sha256", "")[:12],
        diagnosis.get("planned_source_todo_sha256", "")[:12],
    )
    return {
        **diagnosis,
        "refreshed": True,
        "archived": archived,
        "binding": binding,
    }


def _write_bytes_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def materialize_bundle_lane_taskboard(
    lane: BundleLaneSpec,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Copy one digest-bound source shard into lane-owned operational state."""

    runtime_path = lane.runtime_todo_path
    expected_digest = str(lane.source_todo_sha256 or "").strip().lower()
    if runtime_path is None:
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} has no operational taskboard path"
        )
    if len(expected_digest) != 64:
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} has no valid source taskboard digest"
        )
    source_path = lane.todo_path.resolve()
    runtime_path = runtime_path.resolve()
    state_dir = lane.state_dir.resolve()
    try:
        runtime_path.relative_to(state_dir)
    except ValueError as exc:
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} runtime taskboard must be inside its state directory"
        ) from exc
    if runtime_path == source_path:
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} runtime taskboard must not replace its source"
        )
    try:
        content = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} source taskboard is unavailable: {source_path}"
        ) from exc
    observed_digest = hashlib.sha256(content).hexdigest()
    if observed_digest != expected_digest:
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} source taskboard digest changed "
            f"after planning: expected {expected_digest}, observed {observed_digest}"
        )

    binding_path = bundle_taskboard_input_binding_path(lane)
    binding_source_path = repo_relative_path(repo_root, source_path)
    binding_runtime_path = repo_relative_path(repo_root, runtime_path)
    try:
        existing_binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        existing_binding = None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} taskboard input binding is invalid"
        ) from exc
    if existing_binding is not None and not isinstance(existing_binding, dict):
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} taskboard input binding is invalid"
        )
    if runtime_path.exists() and isinstance(existing_binding, dict):
        expected_binding = {
            "schema": BUNDLE_TASKBOARD_INPUT_SCHEMA,
            "bundle_key": lane.bundle_key,
            "source_todo_path": binding_source_path,
            "source_todo_sha256": expected_digest,
            "runtime_todo_path": binding_runtime_path,
            "runtime_initial_sha256": expected_digest,
        }
        mismatched = [
            key
            for key, value in expected_binding.items()
            if existing_binding.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                f"bundle lane {lane.bundle_key!r} runtime taskboard is bound "
                f"to different input fields: {', '.join(mismatched)}"
            )
        return {
            **existing_binding,
            "materialized": False,
            "reused": True,
            "runtime_current_sha256": _taskboard_sha256(runtime_path),
        }
    if existing_binding is not None and not runtime_path.exists():
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} runtime taskboard is missing "
            "for its existing input binding"
        )
    if runtime_path.exists() and _taskboard_sha256(runtime_path) != expected_digest:
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} has an unbound modified runtime taskboard"
        )

    _write_bytes_atomically(runtime_path, content)
    runtime_digest = _taskboard_sha256(runtime_path)
    if runtime_digest != expected_digest:
        raise OSError(
            f"bundle lane {lane.bundle_key!r} runtime taskboard copy failed digest verification"
        )
    if _taskboard_sha256(source_path) != expected_digest:
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} source taskboard changed during materialization"
        )
    binding = {
        "schema": BUNDLE_TASKBOARD_INPUT_SCHEMA,
        "bundle_key": lane.bundle_key,
        "source_todo_path": binding_source_path,
        "source_todo_sha256": expected_digest,
        "runtime_todo_path": binding_runtime_path,
        "runtime_initial_sha256": runtime_digest,
        "materialized_at": utc_now(),
        "materialized": True,
    }
    _write_bytes_atomically(
        binding_path,
        (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return binding


def immutable_lane_input_artifact(
    lane: BundleLaneSpec,
    *,
    repository_id: str,
    created_at_ms: int | None = None,
) -> ImmutableLaneInputArtifact:
    """Build the exact, content-addressed payload available to any worker.

    The source taskboard bytes are included instead of a checkout-relative
    pathname.  A remote worker therefore cannot observe a later local rewrite,
    and local fallback consumes the same artifact as remote execution.
    """

    expected_digest = str(lane.source_todo_sha256 or "").strip().lower()
    try:
        source_bytes = lane.todo_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"distributed lane source is unavailable: {lane.todo_path}"
        ) from exc
    observed_digest = hashlib.sha256(source_bytes).hexdigest()
    if len(expected_digest) != 64 or observed_digest != expected_digest:
        raise ValueError(
            "distributed lane source digest differs from its planned immutable input"
        )
    payload = {
        "bundle_key": lane.bundle_key,
        "parallel_lane": lane.parallel_lane,
        "task_ids": list(lane.task_ids),
        "task_cid": lane.task_cid,
        "goal_cid": lane.goal_cid,
        "subgoal_cid": lane.subgoal_cid,
        "source_todo_sha256": observed_digest,
        "source_todo_base64": base64.b64encode(source_bytes).decode("ascii"),
        "command": list(lane.command),
        "required_capabilities": sorted(set(lane.required_capabilities)),
        "resource_class": lane.resource_class,
        "resource_stage": lane.resource_stage,
        "required_context_tokens": lane.required_context_tokens,
        "token_budget": lane.token_budget,
        "memory_bytes": lane.memory_bytes,
        "gpu_memory_bytes": lane.gpu_memory_bytes,
        "disk_bytes": lane.disk_bytes,
        "process_slots": lane.process_slots,
        "queue_payload": dict(lane.queue_payload or {}),
    }
    return ImmutableLaneInputArtifact(
        repository_id=repository_id,
        task_cid=lane.task_cid,
        payload=payload,
        created_at_ms=(
            int(time.time() * 1000)
            if created_at_ms is None
            else int(created_at_ms)
        ),
    )


@dataclass(frozen=True)
class DistributedLaneWorker:
    """One optional remote target and its exact expiring receipts."""

    worker_id: str
    capability_receipt: WorkerCapabilityReceipt
    environment_receipt: WorkerEnvironmentReceipt
    execute: Callable[
        [DistributedLaneDispatch, ImmutableLaneInputArtifact, threading.Event],
        RemoteLaneResult | Mapping[str, Any],
    ]
    cancel: Callable[[DistributedLaneDispatch, str], Any] | None = None

    def validate_for(
        self,
        lane: BundleLaneSpec,
        *,
        now_ms: int,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        if (
            not self.worker_id
            or self.capability_receipt.worker_id != self.worker_id
            or self.environment_receipt.worker_id != self.worker_id
        ):
            failures.append("foreign_worker_receipt")
        if (
            self.environment_receipt.capability_receipt_id
            != self.capability_receipt.receipt_id
        ):
            failures.append("capability_environment_binding_mismatch")
        try:
            self.capability_receipt.validate_at(now_ms)
        except ValueError:
            failures.append("capability_receipt_expired")
        try:
            self.environment_receipt.validate_at(now_ms)
        except ValueError:
            failures.append("environment_receipt_expired")
        missing = sorted(
            set(lane.required_capabilities)
            - set(self.capability_receipt.capabilities)
        )
        if missing:
            failures.append("required_capability_missing")
        required_environment = (
            lane.queue_payload.get("required_environment")
            if isinstance(lane.queue_payload, Mapping)
            else None
        )
        if isinstance(required_environment, Mapping):
            attributes = self.environment_receipt.attributes
            if any(attributes.get(key) != value for key, value in required_environment.items()):
                failures.append("environment_mismatch")
        if not callable(self.execute):
            failures.append("worker_executor_missing")
        return tuple(dict.fromkeys(failures))


@dataclass(frozen=True)
class DistributedLaneExecution:
    """Terminal projection of one remote or deterministic local attempt."""

    request_id: str
    execution_mode: str
    worker_id: str
    artifact_id: str
    task_cid: str
    disposition: str
    publication: Mapping[str, Any] = field(default_factory=dict)
    merge_result: Mapping[str, Any] = field(default_factory=dict)
    quarantine: Mapping[str, Any] = field(default_factory=dict)
    fallback_reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.disposition in {"accepted", "duplicate"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "execution_mode": self.execution_mode,
            "worker_id": self.worker_id,
            "artifact_id": self.artifact_id,
            "task_cid": self.task_cid,
            "disposition": self.disposition,
            "accepted": self.accepted,
            "publication": dict(self.publication),
            "merge_result": dict(self.merge_result),
            "quarantine": dict(self.quarantine),
            "fallback_reason": self.fallback_reason,
        }


class RemoteLaneUnavailable(RuntimeError):
    """A worker could not start or finish and local fallback may proceed."""


class DistributedLaneDispatcher:
    """Dispatch one fenced lane remotely, or deterministically execute locally.

    The coordinator remains the authority for dispatch and publication.  This
    class owns only deterministic worker selection, heartbeat/cancellation
    propagation, and the hand-off of an accepted publication to the merge
    train.  An ambiguous remote failure is never re-executed locally in the
    same fencing epoch.
    """

    def __init__(
        self,
        coordinator: LeaseCoordinator,
        *,
        repository_id: str,
        local_executor: Callable[
            [DistributedLaneDispatch, ImmutableLaneInputArtifact, threading.Event],
            RemoteLaneResult | Mapping[str, Any],
        ],
        remote_workers: Sequence[DistributedLaneWorker] = (),
        merge_submit: Callable[[Any], Mapping[str, Any]] | Any | None = None,
        clock_ms: Callable[[], int] | None = None,
        heartbeat_interval: float = 5.0,
        heartbeat_capacity_millionths: int = 1_000_000,
    ) -> None:
        if not str(repository_id).strip():
            raise ValueError("repository_id must be non-empty")
        if not callable(local_executor):
            raise ValueError("local_executor must be callable")
        if float(heartbeat_interval) <= 0:
            raise ValueError("heartbeat_interval must be positive")
        capacity = int(heartbeat_capacity_millionths)
        if not 0 <= capacity <= 1_000_000:
            raise ValueError(
                "heartbeat_capacity_millionths must be in [0, 1000000]"
            )
        self.coordinator = coordinator
        self.repository_id = str(repository_id).strip()
        self.local_executor = local_executor
        self.merge_submit = merge_submit
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.heartbeat_interval = float(heartbeat_interval)
        self.heartbeat_capacity_millionths = capacity
        self._lock = threading.RLock()
        self._workers: dict[str, DistributedLaneWorker] = {}
        self.set_remote_workers(remote_workers)

    def set_remote_workers(
        self, workers: Sequence[DistributedLaneWorker]
    ) -> None:
        """Atomically replace advertisements; duplicate worker IDs fail closed."""

        registered: dict[str, DistributedLaneWorker] = {}
        for worker in workers:
            if not isinstance(worker, DistributedLaneWorker):
                raise TypeError("remote workers must be DistributedLaneWorker values")
            if worker.worker_id in registered:
                raise ValueError(f"duplicate distributed worker id: {worker.worker_id}")
            registered[worker.worker_id] = worker
        with self._lock:
            self._workers = registered

    def _eligible_workers(
        self,
        lane: BundleLaneSpec,
        *,
        now_ms: int,
        lease_expires_at_ms: int | None = None,
    ) -> tuple[DistributedLaneWorker, ...]:
        with self._lock:
            workers = tuple(self._workers.values())
        return tuple(
            sorted(
                (
                    worker
                    for worker in workers
                    if not worker.validate_for(lane, now_ms=now_ms)
                    and (
                        lease_expires_at_ms is None
                        or (
                            worker.capability_receipt.expires_at_ms
                            >= lease_expires_at_ms
                            and worker.environment_receipt.expires_at_ms
                            >= lease_expires_at_ms
                        )
                    )
                ),
                key=lambda worker: (
                    worker.worker_id,
                    worker.capability_receipt.receipt_id,
                    worker.environment_receipt.receipt_id,
                ),
            )
        )

    def _local_worker(
        self, lane: BundleLaneSpec
    ) -> DistributedLaneWorker:
        """Return deterministic local receipts for the same artifact contract."""

        capabilities = tuple(
            sorted(set(lane.required_capabilities) | {"local-execution"})
        )
        capability = WorkerCapabilityReceipt(
            worker_id="local",
            capabilities=capabilities,
            issued_at_ms=0,
            expires_at_ms=9_223_372_036_854_775_807,
            capability_revision="local-fallback@1",
            metadata={"repository_id": self.repository_id},
        )
        required_environment = (
            dict(lane.queue_payload.get("required_environment") or {})
            if isinstance(lane.queue_payload, Mapping)
            and isinstance(lane.queue_payload.get("required_environment"), Mapping)
            else {}
        )
        environment = WorkerEnvironmentReceipt(
            worker_id="local",
            environment_id=f"local:{self.repository_id}",
            capability_receipt_id=capability.receipt_id,
            issued_at_ms=0,
            expires_at_ms=9_223_372_036_854_775_807,
            attributes=required_environment,
        )
        return DistributedLaneWorker(
            worker_id="local",
            capability_receipt=capability,
            environment_receipt=environment,
            execute=self.local_executor,
        )

    def _current_worker(
        self, selected: DistributedLaneWorker
    ) -> DistributedLaneWorker:
        if selected.worker_id == "local":
            return selected
        with self._lock:
            return self._workers.get(selected.worker_id, selected)

    @staticmethod
    def _normalize_result(
        raw: RemoteLaneResult | Mapping[str, Any],
        *,
        dispatch: DistributedLaneDispatch,
        artifact: ImmutableLaneInputArtifact,
        worker: DistributedLaneWorker,
        now_ms: int,
    ) -> RemoteLaneResult | Mapping[str, Any]:
        if isinstance(raw, RemoteLaneResult):
            return raw
        if not isinstance(raw, Mapping):
            # Preserve malformed foreign input for the coordinator's
            # quarantine boundary instead of manufacturing a valid result.
            return {"malformed_result_type": type(raw).__name__}
        value = dict(raw)
        value.setdefault("repository_id", dispatch.repository_id)
        value.setdefault("worker_id", worker.worker_id)
        value.setdefault("task_cid", dispatch.task_cid)
        value.setdefault("artifact_id", artifact.artifact_id)
        value.setdefault(
            "capability_receipt_id", worker.capability_receipt.receipt_id
        )
        value.setdefault(
            "environment_receipt_id", worker.environment_receipt.receipt_id
        )
        value.setdefault("claim_cid", dispatch.grant.claim_cid)
        value.setdefault("logical_epoch", dispatch.logical_epoch)
        value.setdefault("fencing_token", dispatch.fencing_token)
        value.setdefault("created_at_ms", now_ms)
        value.setdefault("output", {})
        try:
            return RemoteLaneResult.from_dict(value)
        except (TypeError, ValueError):
            # The lease boundary records the exact malformed projection and
            # reason; do not throw it away by failing early here.
            return value

    @staticmethod
    def _publication_envelope(
        result: RemoteLaneResult,
        dispatch: DistributedLaneDispatch,
    ) -> dict[str, Any]:
        envelope = {
            "schema": DISTRIBUTED_LANE_PUBLICATION_SCHEMA,
            "repository_id": result.repository_id,
            "request_id": result.request_id,
            "publication_id": result.publication_id,
            "worker_id": result.worker_id,
            "task_cid": result.task_cid,
            "artifact_id": result.artifact_id,
            "candidate_commit": result.candidate_commit,
            "capability_receipt_id": result.capability_receipt_id,
            "environment_receipt_id": result.environment_receipt_id,
            "claimant": dispatch.grant.claimant_did,
            "claim_cid": result.claim_cid,
            "logical_epoch": result.logical_epoch,
            "fencing_epoch": result.logical_epoch,
            "fencing_token": result.fencing_token,
            "cancelled": result.cancelled,
        }
        envelope["digest"] = _distributed_lane_digest(envelope)
        return envelope

    @staticmethod
    def _disposition_name(value: Mapping[str, Any]) -> str:
        if value.get("duplicate") is True:
            return "duplicate"
        if value.get("quarantined") is True:
            return "quarantined"
        if value.get("cancelled") is True:
            return "cancelled"
        raw = str(
            value.get("disposition")
            or value.get("status")
            or value.get("decision")
            or ""
        ).strip().lower()
        if value.get("accepted") is True and raw not in {"duplicate", "cancelled"}:
            return "accepted"
        if raw in {"published", "succeeded", "ok"}:
            return "accepted"
        if raw in {"duplicate", "already_published", "already_completed"}:
            return "duplicate"
        if raw in {"cancelled", "canceled"}:
            return "cancelled"
        if raw in {"quarantined", "quarantine", "rejected", "invalid", "stale"}:
            return "quarantined"
        return raw or "rejected"

    def _submit_merge(
        self,
        lane: BundleLaneSpec,
        result: RemoteLaneResult,
        envelope: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.merge_submit is None:
            return {"accepted": True, "queued": False, "reason": "merge_submit_absent"}
        from ..merge.merge_queue import MergeRequest

        priority = str(
            (lane.queue_payload or {}).get("priority")
            or f"P{min(3, max(0, lane.objective_priority))}"
        ).upper()
        if priority not in {"P0", "P1", "P2", "P3"}:
            priority = "P2"
        branch_name = str(result.output.get("branch_name") or "").strip()
        if not branch_name:
            raise ValueError(
                "distributed merge candidate must identify its source branch"
            )
        request = MergeRequest(
            request_id=result.publication_id,
            branch_name=branch_name,
            task_id=lane.task_ids[0] if lane.task_ids else lane.task_cid,
            priority=priority,
            lane_id=lane.parallel_lane or result.worker_id,
            enqueued_at=self._clock_ms() / 1000,
            metadata={
                "candidate_commit": result.candidate_commit,
                "canonical_task_id": lane.task_cid,
                "distributed_publication": dict(envelope),
            },
            commit_sha=result.candidate_commit,
            canonical_task_id=lane.task_cid,
        )
        submitter = self.merge_submit
        if (
            hasattr(submitter, "queue")
            and hasattr(submitter, "run_once")
            and hasattr(submitter.queue, "enqueue")
        ):
            submitter.queue.enqueue(
                branch_name=request.branch_name,
                task_id=request.task_id,
                priority=request.priority,
                lane_id=request.lane_id,
                metadata=request.metadata,
                commit_sha=request.commit_sha,
                canonical_task_id=request.canonical_task_id,
            )
            response = submitter.run_once()
        elif hasattr(submitter, "admit_distributed_publication"):
            # Lightweight adapters may own queue claiming externally while
            # exposing the merge train's admission spelling directly.
            response = submitter.admit_distributed_publication(request)
        else:
            response = submitter(request)
        if not isinstance(response, Mapping):
            raise TypeError("merge submission must return a mapping")
        return dict(response)

    def execute(
        self,
        lane: BundleLaneSpec,
        grant: Any,
        *,
        cancel_event: threading.Event | None = None,
    ) -> DistributedLaneExecution:
        """Execute one lease without allowing duplicate local/remote work."""

        cancellation = cancel_event or threading.Event()
        artifact = immutable_lane_input_artifact(
            lane,
            repository_id=self.repository_id,
            created_at_ms=self._clock_ms(),
        )
        eligible = self._eligible_workers(
            lane,
            now_ms=self._clock_ms(),
            lease_expires_at_ms=int(grant.lease_expires_at_ms),
        )
        worker = eligible[0] if eligible else self._local_worker(lane)
        mode = "remote" if eligible else "local"
        fallback_reason = "" if eligible else "remote_capacity_unavailable"
        dispatch = self.coordinator.dispatch_remote(
            grant,
            input_artifact=artifact,
            capability_receipt=worker.capability_receipt,
            environment_receipt=worker.environment_receipt,
            worker_id=worker.worker_id,
            repository_id=self.repository_id,
            required_capabilities=tuple(lane.required_capabilities),
            now_ms=self._clock_ms(),
        )
        request_id = dispatch.dispatch_cid
        if cancellation.is_set():
            cancelled = self.coordinator.cancel_remote(
                dispatch,
                reason="caller_cancelled_before_execution",
                now_ms=self._clock_ms(),
            )
            if worker.cancel is not None:
                worker.cancel(cancelled, "caller_cancelled_before_execution")
            return DistributedLaneExecution(
                request_id=request_id,
                execution_mode=mode,
                worker_id=worker.worker_id,
                artifact_id=artifact.artifact_id,
                task_cid=lane.task_cid,
                disposition="cancelled",
                fallback_reason=fallback_reason,
            )

        heartbeat_stop = threading.Event()
        heartbeat_failures: list[BaseException] = []

        def heartbeat() -> None:
            nonlocal dispatch
            while not heartbeat_stop.wait(self.heartbeat_interval):
                if cancellation.is_set():
                    try:
                        self.coordinator.cancel_remote(
                            dispatch,
                            reason="caller_cancelled",
                            now_ms=self._clock_ms(),
                        )
                    except BaseException as exc:
                        heartbeat_failures.append(exc)
                    return
                try:
                    dispatch = self.coordinator.heartbeat_remote(
                        dispatch,
                        phase="executing",
                        capacity_millionths=self.heartbeat_capacity_millionths,
                        now_ms=self._clock_ms(),
                    )
                except BaseException as exc:
                    heartbeat_failures.append(exc)
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"distributed-lane-heartbeat-{lane.task_cid[-12:]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            raw_result = worker.execute(dispatch, artifact, cancellation)
        except BaseException as exc:
            heartbeat_stop.set()
            heartbeat_thread.join()
            try:
                updated = self.coordinator.cancel_remote(
                    dispatch,
                    reason=f"worker_execution_failed:{type(exc).__name__}",
                    now_ms=self._clock_ms(),
                )
                if worker.cancel is not None:
                    worker.cancel(updated, "worker_execution_failed")
            except LeaseError:
                pass
            # Once a remote dispatch was visible, local retry in the same epoch
            # would create ambiguous duplicate work.  The scheduler may reclaim
            # it only through a later fencing epoch.
            return DistributedLaneExecution(
                request_id=request_id,
                execution_mode=mode,
                worker_id=worker.worker_id,
                artifact_id=artifact.artifact_id,
                task_cid=lane.task_cid,
                disposition="worker_lost",
                quarantine={"error": type(exc).__name__, "message": str(exc)},
                fallback_reason=fallback_reason,
            )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join()

        normalized = self._normalize_result(
            raw_result,
            dispatch=dispatch,
            artifact=artifact,
            worker=worker,
            now_ms=self._clock_ms(),
        )
        if cancellation.is_set():
            try:
                dispatch = self.coordinator.cancel_remote(
                    dispatch,
                    reason="caller_cancelled",
                    now_ms=self._clock_ms(),
                )
            except LeaseError:
                pass
            if worker.cancel is not None:
                worker.cancel(dispatch, "caller_cancelled")
        current_worker = self._current_worker(worker)
        publication = self.coordinator.publish_remote_result(
            dispatch,
            normalized,
            current_capability_receipt=current_worker.capability_receipt,
            current_environment_receipt=current_worker.environment_receipt,
            now_ms=self._clock_ms(),
        )
        publication_payload = (
            dict(publication) if isinstance(publication, Mapping) else {}
        )
        disposition = self._disposition_name(publication_payload)
        merge_result: dict[str, Any] = {}
        if (
            disposition == "accepted"
            and isinstance(normalized, RemoteLaneResult)
            and normalized.status == "succeeded"
            and not normalized.cancelled
        ):
            envelope = self._publication_envelope(normalized, dispatch)
            merge_result = self._submit_merge(lane, normalized, envelope)
            try:
                finalized = self.coordinator.finalize_remote_result(
                    dispatch,
                    normalized,
                    merge_result=merge_result,
                    now_ms=self._clock_ms(),
                )
            except (LeaseError, ValueError) as exc:
                merge_result["distributed_finalization"] = {
                    "finalized": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                disposition = "merge_rejected"
            else:
                merge_result["distributed_finalization"] = dict(finalized)
                if finalized.get("finalized") is not True:
                    disposition = "merge_rejected"
        return DistributedLaneExecution(
            request_id=(
                normalized.request_id
                if isinstance(normalized, RemoteLaneResult)
                else request_id
            ),
            execution_mode=mode,
            worker_id=worker.worker_id,
            artifact_id=artifact.artifact_id,
            task_cid=lane.task_cid,
            disposition=disposition,
            publication=publication_payload,
            merge_result=merge_result,
            quarantine=(
                dict(publication_payload.get("quarantine") or {})
                if isinstance(publication_payload.get("quarantine"), Mapping)
                else {}
            ),
            fallback_reason=fallback_reason,
        )


# Supervisor is a compatibility-friendly name for orchestration callers.
DistributedLaneSupervisor = DistributedLaneDispatcher


def _compact_bundle_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep live manifests bounded while retaining the planning source."""

    compact = {
        key: value
        for key, value in payload.items()
        if key not in _MANIFEST_REFERENCED_BUNDLE_FIELDS
    }
    omitted = sorted(_MANIFEST_REFERENCED_BUNDLE_FIELDS.intersection(payload))
    tasks = compact.get("tasks")
    if isinstance(tasks, list):
        compact["tasks"] = [
            {
                key: value
                for key, value in task.items()
                if key in _MANIFEST_MEMBER_TASK_FIELDS
            }
            for task in tasks
            if isinstance(task, dict)
        ]
    profile_g = compact.get("profile_g")
    if isinstance(profile_g, dict):
        compact["profile_g"] = {
            key: value
            for key, value in profile_g.items()
            if key in _MANIFEST_PROFILE_G_REFERENCE_FIELDS
        }
    existing_reference = (
        dict(compact.get("planning_evidence_ref") or {})
        if isinstance(compact.get("planning_evidence_ref"), dict)
        else {}
    )
    if omitted or existing_reference:
        index_path = str(payload.get("objective_bundle_index") or "")
        compact["planning_evidence_ref"] = {
            **existing_reference,
            "bundle_index": index_path,
            "bundle_index_duckdb": str(Path(index_path).with_suffix(".duckdb")) if index_path else "",
            "bundle_key": str(payload.get("bundle_key") or ""),
            "omitted_fields": sorted(_MANIFEST_REFERENCED_BUNDLE_FIELDS),
            "bundle_table": "bundles",
            "task_table": "bundle_tasks",
        }
    return compact


def _compact_task_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    bundle = compact.get("bundle")
    if isinstance(bundle, dict):
        compact["bundle"] = _compact_bundle_manifest_payload(bundle)
    return compact


def _lane_manifest_payload(lane: BundleLaneSpec, *, repo_root: Path) -> dict[str, Any]:
    payload = lane.to_dict(repo_root=repo_root)
    queue_payload = payload.get("queue_payload")
    if isinstance(queue_payload, dict):
        payload["queue_payload"] = _compact_bundle_manifest_payload(queue_payload)
    return payload


def _lane_database_payload(lane: BundleLaneSpec, *, repo_root: Path) -> dict[str, Any]:
    """Keep live rows bounded and reference complete planning evidence by key."""

    return _lane_manifest_payload(lane, repo_root=repo_root)


@dataclass
class RunningBundleLane:
    """Live scheduler ownership for one subprocess and its fenced lease."""

    spec: BundleLaneSpec
    grant: Any
    handle: Any
    started_at: str
    resource_lease: ResourceAdmissionLease | None = None
    resource_stage_started_at: str = ""

    def __post_init__(self) -> None:
        if not self.resource_stage_started_at:
            self.resource_stage_started_at = self.started_at

    @property
    def pid(self) -> int | None:
        value = getattr(self.handle, "pid", None)
        return int(value) if isinstance(value, int) else None

    def to_dict(self, *, repo_root: Path) -> dict[str, Any]:
        payload = _lane_manifest_payload(self.spec, repo_root=repo_root)
        payload.update(
            {
                "state": "running",
                "pid": self.pid,
                "started_at": self.started_at,
                "resource_stage_started_at": self.resource_stage_started_at,
                "lease": self.grant.to_dict(),
                "resource_lease": (
                    {
                        "lease_id": self.resource_lease.lease_id,
                        "lane_id": self.resource_lease.lane_id,
                        "stage": self.resource_lease.requirement.stage,
                        "resource_class": self.resource_lease.resource_class,
                        "resource_pool": self.resource_lease.resource_pool,
                        "provider_id": self.resource_lease.provider_id,
                        "acquired_at_ms": self.resource_lease.acquired_at_ms,
                    }
                    if self.resource_lease is not None
                    else None
                ),
            }
        )
        return payload

    def to_database_dict(self, *, repo_root: Path) -> dict[str, Any]:
        payload = _lane_database_payload(self.spec, repo_root=repo_root)
        payload.update(
            {
                "state": "running",
                "pid": self.pid,
                "started_at": self.started_at,
                "resource_stage_started_at": self.resource_stage_started_at,
                "lease": self.grant.to_dict(),
                "resource_lease": (
                    self.resource_lease.to_dict()
                    if self.resource_lease is not None
                    else None
                ),
            }
        )
        return payload


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def lane_state_prefix(bundle_key: str) -> str:
    return f"agent_{safe_bundle_key(bundle_key).replace('-', '_')}"


def _execution_slice_task_cids_by_id(
    payload: Mapping[str, Any],
    execution_tasks: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[str],
    task_cids: Sequence[str],
) -> dict[str, str]:
    """Return the exact display-ID to canonical-CID execution-slice binding.

    Member rows are the only planner surface which contains both identities.
    Optimizer work contracts independently bind the canonical CIDs; when those
    contracts are present, this function cross-checks them instead of assuming
    that two separately ordered ID/CID lists align.
    """

    selected_ids = tuple(
        dict.fromkeys(str(task_id).strip() for task_id in task_ids if str(task_id).strip())
    )
    selected_cids = {
        str(task_cid).strip()
        for task_cid in task_cids
        if str(task_cid).strip()
    }
    candidates: dict[str, str] = {}
    for task in execution_tasks:
        task_id = str(task.get("task_id") or "").strip()
        task_cid = str(
            task.get("canonical_task_cid") or task.get("task_cid") or ""
        ).strip()
        if not task_id or not task_cid:
            continue
        previous = candidates.get(task_id)
        if previous is not None and previous != task_cid:
            raise ValueError(
                f"execution slice task {task_id!r} has conflicting canonical CIDs"
            )
        candidates[task_id] = task_cid

    missing_ids = [task_id for task_id in selected_ids if task_id not in candidates]
    if missing_ids:
        raise ValueError(
            "execution slice lacks canonical task identities for "
            f"{missing_ids!r}"
        )
    bindings = {task_id: candidates[task_id] for task_id in selected_ids}
    bound_cids = set(bindings.values())
    if selected_cids and bound_cids != selected_cids:
        raise ValueError(
            "execution slice ID/CID projections disagree: "
            f"bound={sorted(bound_cids)!r}, declared={sorted(selected_cids)!r}"
        )

    optimization = (
        payload.get("bundle_optimization")
        if isinstance(payload.get("bundle_optimization"), Mapping)
        else {}
    )
    optimized_bundle = (
        optimization.get("bundle")
        if isinstance(optimization.get("bundle"), Mapping)
        else {}
    )
    raw_contracts = payload.get("task_work_contracts")
    if not isinstance(raw_contracts, (list, tuple)):
        raw_contracts = optimized_bundle.get("task_work_contracts")
    contract_cids = {
        str(contract.get("canonical_task_cid") or "").strip()
        for contract in (raw_contracts or ())
        if isinstance(contract, Mapping)
        and str(contract.get("canonical_task_cid") or "").strip()
    }
    if contract_cids and not bound_cids.issubset(contract_cids):
        raise ValueError(
            "execution slice canonical identities are not bound by its "
            "task work contracts"
        )
    return bindings


def _schedule_int(payload: dict[str, Any], key: str, default: int = 0) -> int:
    """Read integer scheduler metadata without trusting generated JSON types."""

    value = payload.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _schedule_bool(payload: dict[str, Any], key: str, default: bool = True) -> bool:
    value = payload.get(key, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "no", "0", "blocked"}:
            return False
        if normalized in {"true", "yes", "1", "ready"}:
            return True
    return bool(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _lane_schedule_key(lane: BundleLaneSpec) -> tuple[int, int, str]:
    """Put ready lanes in critical-path order while retaining blocked lanes.

    ``claimable`` in an index is a planning snapshot.  It controls only the
    ordering here; :class:`LeaseCoordinator` remains the authority at claim
    time because prerequisite receipts can arrive after the index is written.
    """

    rank = lane.schedule_rank if lane.schedule_rank is not None else sys.maxsize
    return (0 if lane.claimable else 1, rank, lane.bundle_key)


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _execution_slice_members(
    payload: Mapping[str, Any],
    tasks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restrict planning surfaces to the current ready-member slice."""

    if (
        "execution_slice_task_cids" not in payload
        and "execution_slice_task_ids" not in payload
    ):
        return [dict(task) for task in tasks]
    selected_cids = set(_string_list(payload.get("execution_slice_task_cids")))
    selected_ids = set(_string_list(payload.get("execution_slice_task_ids")))
    return [
        dict(task)
        for task in tasks
        if str(task.get("canonical_task_cid") or task.get("task_cid") or "") in selected_cids
        or str(task.get("task_id") or "") in selected_ids
    ]


def _apply_runtime_task_exclusions(
    payloads: Sequence[dict[str, Any]],
    *,
    excluded_task_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Fence selected ready members without changing reviewed bundle inputs.

    Task exclusions are an operator control for one supervisor run, not a task
    completion signal. Keep every source member and work contract attached as
    dependency/provenance evidence, but narrow the authorized execution slice.
    A bundle whose entire ready slice is excluded is omitted instead of being
    registered with coordination, so the fenced work cannot consume a retry.

    Embedded Profile-G artifacts bind the original execution slice. Detach
    that aggregate identity whenever the slice changes and retain a compact
    provenance reference; registration will derive a fresh slice identity from
    the remaining canonical member identities.
    """

    excluded = {
        str(task_id).strip()
        for task_id in excluded_task_ids
        if str(task_id).strip()
    }
    if not excluded:
        return list(payloads)

    projected_payloads: list[dict[str, Any]] = []
    for original in payloads:
        payload = dict(original)
        execution_tasks = _execution_slice_members(
            payload,
            _mapping_list(payload.get("tasks")),
        )
        excluded_members = [
            task
            for task in execution_tasks
            if str(task.get("task_id") or "").strip() in excluded
        ]
        if not excluded_members:
            projected_payloads.append(payload)
            continue

        retained_members = [
            task
            for task in execution_tasks
            if str(task.get("task_id") or "").strip() not in excluded
        ]
        if not retained_members:
            continue

        source_profile = (
            dict(payload.get("profile_g") or {})
            if isinstance(payload.get("profile_g"), Mapping)
            else {}
        )
        if source_profile:
            payload["source_profile_g_ref"] = {
                key: str(source_profile.get(key) or "")
                for key in (
                    "goal_cid",
                    "subgoal_cid",
                    "plan_branch_cid",
                    "selection_cid",
                    "task_cid",
                    "task_spec_cid",
                )
                if source_profile.get(key)
            }
            payload.pop("profile_g", None)

        payload["execution_slice_task_ids"] = [
            str(task.get("task_id") or "").strip()
            for task in retained_members
            if str(task.get("task_id") or "").strip()
        ]
        payload["execution_slice_task_cids"] = [
            str(
                task.get("canonical_task_cid")
                or task.get("task_cid")
                or ""
            ).strip()
            for task in retained_members
            if str(
                task.get("canonical_task_cid")
                or task.get("task_cid")
                or ""
            ).strip()
        ]
        payload["runtime_excluded_task_ids"] = sorted(
            {
                str(task.get("task_id") or "").strip()
                for task in excluded_members
                if str(task.get("task_id") or "").strip()
            }
        )
        projected_payloads.append(
            dict(profile_g_safe_planning_value(payload))
        )
    return projected_payloads


def _first_nonempty(payloads: Sequence[dict[str, Any]], *keys: str) -> Any:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _resource_lane_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize planner/router resource metadata without requiring one schema."""

    profile = payload.get("profile_g") if isinstance(payload.get("profile_g"), dict) else {}
    task_spec = profile.get("task") if isinstance(profile.get("task"), dict) else {}
    selection = profile.get("selection") if isinstance(profile.get("selection"), dict) else {}
    tasks = _execution_slice_members(payload, _mapping_list(payload.get("tasks")))
    # Member requirements precede the adapter's compatibility defaults. This
    # prevents a synthesized ``cpu-small`` TaskSpec from masking an explicit
    # GPU/resource request on the underlying work item.
    sources = [payload, *tasks, task_spec, selection]
    capabilities: list[str] = []
    for source in sources:
        for key in ("required_capabilities", "capabilities", "required_tools"):
            value = source.get(key)
            if isinstance(value, str):
                candidates = value.split(",")
            elif isinstance(value, (list, tuple, set, frozenset)):
                candidates = value
            else:
                continue
            capabilities.extend(str(item).strip() for item in candidates if str(item).strip())

    def maximum(*keys: str) -> int:
        values = [_schedule_int(source, key) for source in sources for key in keys]
        return max(values, default=0)

    return {
        "resource_stage": str(
            _first_nonempty(
                sources,
                "resource_stage",
                "supervisor_stage",
                "stage",
                "active_phase",
            )
            or "analysis"
        ),
        "resource_class": str(
            _first_nonempty(sources, "resource_class", "worker_resource_class") or "cpu-small"
        ),
        "required_capabilities": list(dict.fromkeys(capabilities)),
        "llm_provider": str(
            _first_nonempty(
                sources,
                "llm_provider",
                "provider_id",
                "provider",
                "effective_provider_name",
            )
            or ""
        ),
        "required_context_tokens": maximum(
            "required_context_tokens",
            "context_tokens",
            "compact_context_tokens",
            "estimated_compact_context_tokens",
            "prompt_tokens",
        ),
        "token_budget": maximum(
            "token_budget",
            "estimated_tokens",
            "max_tokens",
            "max_new_tokens",
        ),
        "max_provider_latency_ms": maximum(
            "max_provider_latency_ms",
            "max_latency_ms",
            "latency_budget_ms",
        ),
        "memory_bytes": maximum(
            "memory_bytes",
            "required_memory_bytes",
            "estimated_memory_bytes",
        ),
        "gpu_memory_bytes": maximum(
            "gpu_memory_bytes",
            "required_gpu_memory_bytes",
            "vram_bytes",
            "required_vram_bytes",
        ),
        "disk_bytes": maximum(
            "disk_bytes",
            "required_disk_bytes",
            "estimated_disk_bytes",
        ),
        "process_slots": max(
            1,
            maximum("process_slots", "required_process_slots", "worker_slots"),
        ),
    }


def _execution_slice_implementation_max_timeout(
    payload: Mapping[str, Any],
    *,
    default_timeout: float,
) -> float:
    """Return the largest hard timeout authorized by the execution slice.

    The implementation daemon still receives ``default_timeout`` as its idle
    and ordinary-task policy. This separate maximum sizes the parent
    supervisor watchdog so a task-specific hard timeout cannot be interrupted
    early. Exact per-task limits remain in the digest-bound taskboard and are
    enforced by ``PortalImplementationDaemon``.
    """

    baseline = effective_implementation_hard_timeout(
        {},
        configured_timeout=default_timeout,
    ).seconds
    effective: list[float] = []
    tasks = _execution_slice_members(
        payload,
        _mapping_list(payload.get("tasks")),
    )
    for task in tasks:
        effective.append(
            effective_implementation_hard_timeout(
                task,
                configured_timeout=default_timeout,
                task_id=str(task.get("task_id") or "<unknown task>"),
            ).seconds
        )
    return max(effective, default=baseline)


def _dual_review_resource_fields(
    resource_fields: Mapping[str, Any],
    *,
    context_budget_tokens: int,
) -> dict[str, Any]:
    """Bind one model-backed lane to both independent review providers."""

    result = dict(resource_fields)
    capabilities = [
        *(
            str(item)
            for item in result.get("required_capabilities", ())
            if str(item).strip()
        ),
        *DUAL_REVIEW_REQUIRED_CAPABILITIES,
    ]
    result.update(
        {
            "required_capabilities": list(dict.fromkeys(capabilities)),
            "llm_provider": DUAL_REVIEW_PROVIDER_ID,
            "required_context_tokens": max(
                int(result.get("required_context_tokens") or 0),
                max(1, int(context_budget_tokens)),
            ),
            # Both reviewed routes currently cap generated responses at 4096
            # tokens. The synthetic pair reserves this value against the
            # smaller of the two live token budgets.
            "token_budget": max(
                int(result.get("token_budget") or 0),
                4_096,
            ),
        }
    )
    return result


_TYPED_LOCAL_PROVIDER_ROLES = frozenset(
    {
        "deterministic-only",
        "operator-only",
    }
)
_MODEL_ASSISTED_PROVIDER_ROLES = frozenset(
    {
        "grok",
        "grok-implement",
        "grok-draft",
        "codex",
        "codex-implement",
        "codex-draft",
        "codex-review",
        "codex_independent_review",
    }
)


def _task_provider_roles(task: Mapping[str, Any]) -> frozenset[str]:
    """Read exactly the child daemon's authoritative provider-role field."""

    nested = task.get("metadata")
    if not isinstance(nested, Mapping):
        return frozenset()
    normalized = {
        str(raw_key).strip().lower().replace("_", " "): str(raw_value).strip()
        for raw_key, raw_value in nested.items()
    }
    raw_roles = normalized.get("provider role", "")
    return frozenset(
        item.strip().lower()
        for item in raw_roles.replace(";", ",").split(",")
        if item.strip()
    )


def _production_lane_requires_dual_review(
    tasks: Sequence[Mapping[str, Any]],
    resource_fields: Mapping[str, Any],
) -> bool:
    """Classify model-backed work without fencing typed-local-only lanes."""

    role_sets = [_task_provider_roles(task) for task in tasks]
    if role_sets and all(
        len(roles) == 1 and next(iter(roles)) in _TYPED_LOCAL_PROVIDER_ROLES
        for roles in role_sets
    ):
        return False
    if any(roles.intersection(_MODEL_ASSISTED_PROVIDER_ROLES) for roles in role_sets):
        return True
    if (
        str(resource_fields.get("llm_provider") or "").strip()
        or int(resource_fields.get("required_context_tokens") or 0) > 0
        or int(resource_fields.get("token_budget") or 0) > 0
        or any(
            str(item).strip().lower().startswith("llm:")
            for item in resource_fields.get("required_capabilities", ())
        )
    ):
        return True
    # An unclassified implementation task uses the daemon's configured
    # production provider route. Only an explicit typed-local contract may
    # bypass that route and its capacity reservation.
    return True


_TERMINAL_CONFLICT_TASK_STATUSES = frozenset(
    {"complete", "completed", "done", "merged", "success", "succeeded"}
)


def _live_bundle_conflict_members(
    payload: dict[str, Any],
    *,
    repo_root: Path,
    task_prefix: str,
) -> list[dict[str, Any]]:
    """Exclude settled members from the bundle's prospective edit surface."""

    tasks = _execution_slice_members(payload, _mapping_list(payload.get("tasks")))
    todo_path_text = str(payload.get("todo_path") or "").strip()
    live_statuses: dict[str, str] = {}
    if todo_path_text:
        from ..todo_daemon.implementation_daemon import parse_task_file

        try:
            portal_tasks = parse_task_file(
                resolve_repo_path(repo_root, todo_path_text),
                task_prefix,
            )
        except OSError:
            portal_tasks = []
        live_statuses = {
            str(task.task_id): str(task.status or "").strip().lower()
            for task in portal_tasks
            if task.task_id
        }

    live: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        status = live_statuses.get(task_id, str(task.get("status") or "").strip().lower())
        if status in _TERMINAL_CONFLICT_TASK_STATUSES:
            continue
        live.append(task)
    return live


def _bundle_conflict_task(
    payload: dict[str, Any],
    *,
    tasks: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project all member work onto the execution unit that owns a lane."""

    members = _mapping_list(payload.get("tasks")) if tasks is None else [dict(task) for task in tasks]

    def member_values(*keys: str) -> list[str]:
        values: list[str] = []
        for source in [payload, *members]:
            for key in keys:
                raw = source.get(key)
                if isinstance(raw, str):
                    candidates = [part.strip() for part in raw.split(",")]
                elif isinstance(raw, (list, tuple, set, frozenset)):
                    candidates = [str(part).strip() for part in raw]
                else:
                    candidates = []
                values.extend(item for item in candidates if item)
        return list(dict.fromkeys(values))

    def source_values(source: Mapping[str, Any], *keys: str) -> list[str]:
        values: list[str] = []
        for key in keys:
            raw = source.get(key)
            if isinstance(raw, str):
                candidates = [part.strip() for part in raw.split(",")]
            elif isinstance(raw, (list, tuple, set, frozenset)):
                candidates = [str(part).strip() for part in raw]
            else:
                candidates = []
            values.extend(item for item in candidates if item)
        return list(dict.fromkeys(values))

    # Objective bundles retain broad evidence/control-plane outputs alongside
    # the narrower code and test files that an implementation may mutate.
    # Unioning every alias made a shared discovery directory or objective
    # document serialize otherwise independent lanes.  Select the most precise
    # mutation declaration per live member, retaining Outputs as a compatibility
    # fallback for legacy task payloads.
    mutation_sources: Sequence[Mapping[str, Any]] = members or [payload]
    mutation_files: list[str] = []
    declared_path_candidates: list[str] = []
    for source in mutation_sources:
        files = source_values(source, "files")
        predicted = source_values(
            source,
            "predicted_files",
            "predicted_paths",
            "affected_files",
        )
        legacy_outputs = source_values(source, "outputs", "requested_outputs")
        selected = files or predicted or legacy_outputs
        mutation_files.extend(selected)
        declared_path_candidates.extend([*files, *predicted, *legacy_outputs])
    mutation_files = list(dict.fromkeys(mutation_files))
    mutation_file_set = set(mutation_files)
    advisory_paths = [
        path
        for path in dict.fromkeys(declared_path_candidates)
        if path not in mutation_file_set
    ]

    bundle_key = str(payload.get("bundle_key") or "objective/general")
    profile_g = payload.get("profile_g") if isinstance(payload.get("profile_g"), dict) else {}
    conflict_policy = str(payload.get("conflict_policy") or "")
    ast_symbols = member_values("ast_symbols", "symbols")
    ast_symbol_scopes = {item.lower() for item in member_values("ast_symbol_scope")}
    global_ast_symbols = member_values("global_ast_symbols")
    if ast_symbol_scopes & {"global", "project", "repository"}:
        global_ast_symbols = list(dict.fromkeys([*global_ast_symbols, *ast_symbols]))
    return {
        "task_id": bundle_key,
        "task_cid": str(profile_g.get("task_cid") or payload.get("task_cid") or ""),
        "files": mutation_files,
        "changed_paths": member_values("changed_paths", "actual_changed_paths"),
        "ast_symbols": ast_symbols,
        # Bundle AST findings describe definitions in their predicted files by
        # default.  Cross-file semantic conflicts must be declared explicitly
        # through global_ast_symbols, a global scope, or an interface surface.
        "global_ast_symbols": global_ast_symbols,
        "ast_query": ", ".join(member_values("ast_query")),
        "interfaces": member_values(
            "interfaces", "interface_contracts", "provides_interfaces", "requires_interfaces",
            "required_interfaces", "interface_dependencies", "public_interfaces",
        ),
        "submodules": member_values("submodules", "submodule_paths", "interoperability_pair", "gitlinks"),
        "generated_artifacts": member_values(
            "generated_artifacts", "generated_outputs", "generated_paths", "artifacts"
        ),
        "allow_concurrent_with": member_values("allow_concurrent_with", "concurrency_overrides"),
        "metadata": {
            "bundle_key": bundle_key,
            "member_task_ids": [
                str(task.get("task_id")) for task in members if task.get("task_id")
            ],
            "conflict_policy": conflict_policy,
            "advisory_paths": advisory_paths,
        },
    }


def _excluded_bundle_keys(bundle_index_path: Path) -> set[str]:
    """Return execution units retained only as dependency metadata."""

    try:
        payload = read_artifact_fields(
            bundle_index_path,
            ("excluded_bundle_keys",),
            kind=BUNDLE_INDEX_KIND,
        )
    except (OSError, ValueError, RuntimeError):
        return set()
    value = payload.get("excluded_bundle_keys")
    if isinstance(value, dict):
        return {str(key) for key, excluded in value.items() if excluded and str(key).strip()}
    return set(_string_list(value))


def _conflict_graph_inputs(bundle_index_path: Path) -> dict[str, Any]:
    """Load optional learned-conflict inputs without making them mandatory."""

    try:
        payload = read_artifact_fields(
            bundle_index_path,
            (
                "branch_diffs",
                "conflict_history",
                "conflict_receipts",
                "conflict_weight_history",
                "concurrency_overrides",
            ),
            kind=BUNDLE_INDEX_KIND,
        )
    except (OSError, ValueError, RuntimeError):
        return {}
    history = payload.get("conflict_history")
    if history is None:
        history = payload.get("conflict_weight_history")
    return {
        "branch_diffs": payload.get("branch_diffs"),
        "conflict_receipts": payload.get("conflict_receipts"),
        "concurrency_overrides": payload.get("concurrency_overrides"),
        "history": history,
    }


def _graph_payload(graph: Any) -> dict[str, Any]:
    to_dict = getattr(graph, "to_dict", None)
    payload = to_dict() if callable(to_dict) else graph
    return dict(payload) if isinstance(payload, dict) else {}


def _bundle_conflict_annotations(
    payloads: Sequence[dict[str, Any]],
    *,
    bundle_index_path: Path,
    repo_root: Path,
    task_prefix: str,
) -> dict[str, dict[str, Any]]:
    """Return graph color, surface, edges, and reasons keyed by bundle."""

    if not payloads:
        return {}
    inputs = {key: value for key, value in _conflict_graph_inputs(bundle_index_path).items() if value is not None}
    conflict_members = [
        _live_bundle_conflict_members(
            payload,
            repo_root=repo_root,
            task_prefix=task_prefix,
        )
        for payload in payloads
    ]
    conflict_tasks = [
        _bundle_conflict_task(payload, tasks=members)
        for payload, members in zip(payloads, conflict_members)
    ]
    aliases: dict[str, str] = {}
    for conflict_task, members in zip(conflict_tasks, conflict_members):
        conflict_cid = str(conflict_task.get("task_cid") or conflict_task.get("task_id") or "")
        aliases[str(conflict_task.get("task_id") or "")] = conflict_cid
        aliases[conflict_cid] = conflict_cid
        for member in members:
            for key in ("task_id", "task_cid", "canonical_task_cid"):
                if member.get(key):
                    aliases[str(member[key])] = conflict_cid

    # Overrides are normally declared against member task identities, while a
    # lane executes their whole bundle.  Translate them to execution-unit CIDs
    # before coloring so explicit safe-concurrency declarations remain valid.
    for conflict_task in conflict_tasks:
        translated = [
            aliases.get(str(item), str(item))
            for item in (conflict_task.get("allow_concurrent_with") or [])
        ]
        conflict_task["allow_concurrent_with"] = list(dict.fromkeys(translated))
    history = inputs.get("history")
    if isinstance(history, dict):
        translated_history = dict(history)
        pair_weights = history.get("pair_weights")
        if isinstance(pair_weights, dict):
            translated_pairs: dict[str, float] = {}
            for pair, weight in pair_weights.items():
                parts = str(pair).split("\0", 1)
                if len(parts) != 2:
                    continue
                translated_pair = "\0".join(
                    sorted((aliases.get(parts[0], parts[0]), aliases.get(parts[1], parts[1])))
                )
                try:
                    translated_pairs[translated_pair] = translated_pairs.get(translated_pair, 0.0) + float(weight)
                except (TypeError, ValueError):
                    continue
            translated_history["pair_weights"] = translated_pairs
        inputs["history"] = translated_history
    diffs = inputs.get("branch_diffs")
    if isinstance(diffs, dict):
        by_conflict_cid = {
            str(task.get("task_cid") or task.get("task_id") or ""): task
            for task in conflict_tasks
        }
        translated_diffs: dict[str, list[str]] = {}
        for identity, value in diffs.items():
            if isinstance(value, dict):
                value = value.get("changed_paths") or value.get("paths") or value.get("files") or []
            paths = [str(item) for item in value] if isinstance(value, (list, tuple, set, frozenset)) else []
            target_cid = aliases.get(str(identity), str(identity))
            translated_diffs.setdefault(target_cid, []).extend(paths)
            target = by_conflict_cid.get(target_cid)
            if target is not None:
                target["changed_paths"] = list(
                    dict.fromkeys([*(target.get("changed_paths") or []), *paths])
                )
        inputs["branch_diffs"] = translated_diffs
    receipts = inputs.get("conflict_receipts")
    if isinstance(receipts, dict):
        if set(receipts) & {"left_task_cid", "source_task_cid", "task_cids", "status"}:
            receipts = [receipts]
        else:
            normalized_receipts: list[dict[str, Any]] = []
            for pair, raw_receipt in receipts.items():
                if not isinstance(raw_receipt, dict):
                    continue
                item = dict(raw_receipt)
                if not item.get("task_cids"):
                    parts = str(pair).replace("<->", "\0").replace("::", "\0").split("\0", 1)
                    if len(parts) == 2:
                        item["task_cids"] = parts
                normalized_receipts.append(item)
            receipts = normalized_receipts
    if isinstance(receipts, (list, tuple)):
        translated_receipts: list[dict[str, Any]] = []
        for receipt in receipts:
            if not isinstance(receipt, dict):
                continue
            item = dict(receipt)
            for key in (
                "left_task_cid", "source_task_cid", "task_cid", "left_task_id",
                "right_task_cid", "target_task_cid", "other_task_cid", "right_task_id",
            ):
                if item.get(key):
                    item[key] = aliases.get(str(item[key]), str(item[key]))
            task_cids = item.get("task_cids") or item.get("tasks")
            if isinstance(task_cids, (list, tuple)):
                item["task_cids"] = [aliases.get(str(value), str(value)) for value in task_cids]
            translated_receipts.append(item)
        inputs["conflict_receipts"] = translated_receipts
    overrides = inputs.get("concurrency_overrides")
    if isinstance(overrides, dict):
        if set(overrides) & {"left", "left_task_cid", "task", "right", "right_task_cid", "with"}:
            overrides = [overrides]
        else:
            normalized_overrides: list[Any] = []
            for pair, allowed in overrides.items():
                if not allowed:
                    continue
                parts = str(pair).replace("<->", "\0").replace("::", "\0").split("\0", 1)
                if len(parts) == 2:
                    normalized_overrides.append(tuple(parts))
            overrides = normalized_overrides
    if isinstance(overrides, (list, tuple, set, frozenset)):
        translated_overrides: list[Any] = []
        for override in overrides:
            if isinstance(override, dict):
                item = dict(override)
                for key in ("left", "left_task_cid", "task"):
                    if item.get(key):
                        item[key] = aliases.get(str(item[key]), str(item[key]))
                for key in ("right", "right_task_cid", "with"):
                    if item.get(key):
                        item[key] = aliases.get(str(item[key]), str(item[key]))
                translated_overrides.append(item)
            elif isinstance(override, (list, tuple)) and len(override) == 2:
                translated_overrides.append(
                    (aliases.get(str(override[0]), str(override[0])), aliases.get(str(override[1]), str(override[1])))
                )
            else:
                translated_overrides.append(override)
        inputs["concurrency_overrides"] = translated_overrides
    graph = materialize_task_conflict_graph(
        conflict_tasks,
        repo_root=repo_root,
        **inputs,
    )
    serialized = _graph_payload(graph)
    surfaces = serialized.get("surfaces") if isinstance(serialized.get("surfaces"), dict) else {}
    assignments = serialized.get("assignments") if isinstance(serialized.get("assignments"), list) else []
    decisions = serialized.get("decisions") if isinstance(serialized.get("decisions"), list) else []
    edges = serialized.get("edges") if isinstance(serialized.get("edges"), list) else []
    colors: dict[str, int] = {}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        task_id = str(assignment.get("task_cid") or assignment.get("task_id") or assignment.get("node") or "")
        value = assignment.get("lane_color", assignment.get("color"))
        try:
            if task_id and value is not None:
                colors[task_id] = int(value)
        except (TypeError, ValueError):
            continue
    # Some graph serializers expose the inverse color -> task list mapping.
    lanes = serialized.get("lanes") if isinstance(serialized.get("lanes"), dict) else {}
    for color, task_ids in lanes.items():
        try:
            parsed_color = int(color)
        except (TypeError, ValueError):
            continue
        if isinstance(task_ids, list):
            for task_id in task_ids:
                colors.setdefault(str(task_id), parsed_color)

    annotations: dict[str, dict[str, Any]] = {}
    for payload, conflict_task in zip(payloads, conflict_tasks):
        bundle_key = str(payload.get("bundle_key") or "objective/general")
        conflict_key = str(conflict_task.get("task_cid") or bundle_key)
        peers: set[str] = set()
        relevant_decisions: list[dict[str, Any]] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            if edge.get("blocks_concurrency") is False or edge.get("explicitly_allowed") is True:
                continue
            left = str(
                edge.get("left_task_cid") or edge.get("source") or edge.get("left")
                or edge.get("task_a") or edge.get("source_task_id") or ""
            )
            right = str(
                edge.get("right_task_cid") or edge.get("target") or edge.get("right")
                or edge.get("task_b") or edge.get("target_task_id") or ""
            )
            if left == conflict_key and right:
                peer = next(
                    (
                        str(task.get("task_id"))
                        for task in conflict_tasks
                        if str(task.get("task_cid") or task.get("task_id")) == right
                    ),
                    right,
                )
                peers.add(peer)
            elif right == conflict_key and left:
                peer = next(
                    (
                        str(task.get("task_id"))
                        for task in conflict_tasks
                        if str(task.get("task_cid") or task.get("task_id")) == left
                    ),
                    left,
                )
                peers.add(peer)
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            ids = {
                str(decision.get(key) or "")
                for key in (
                    "task_id", "task_cid", "source", "target", "left", "right", "task_a", "task_b",
                    "left_task_cid", "right_task_cid",
                )
            }
            if conflict_key in ids or bundle_key in ids:
                relevant_decisions.append(dict(decision))
        annotations[bundle_key] = {
            "conflict_color": colors.get(conflict_key),
            "conflicting_task_ids": sorted(peers),
            "conflict_decisions": relevant_decisions,
            "conflict_surface": (
                dict(surfaces.get(conflict_key) or {})
                if isinstance(surfaces.get(conflict_key), dict)
                else {}
            ),
        }
    return annotations


def implementation_supervisor_command(
    *,
    todo_path: Path,
    state_dir: Path,
    worktree_root: Path,
    state_prefix: str,
    task_prefix: str,
    implement: bool,
    daemon_interval: float,
    stale_seconds: float,
    check_interval: float,
    watchdog_startup_grace_seconds: float | None,
    max_restarts: int,
    implementation_timeout: float,
    implementation_max_timeout: float | None = None,
    implementation_log_stall_seconds: float | None = None,
    max_task_attempts: int = 0,
    implementation_command: str = "",
    production_provider_policy: str = "",
    production_provider_context_budget_tokens: int = 0,
    production_provider_timeout_seconds: float = 0.0,
    production_provider_review_authority_key_path: Path | None = None,
    production_provider_launch_authority_receipt_path: Path | None = None,
    production_provider_launch_authority_receipt_content_id: str = "",
    legacy_landed_review_policy_path: Path | None = None,
    legacy_landed_review_key_path: Path | None = None,
    merge_target_branch: str = "",
    llm_merge_resolver_command: str = "",
    llm_merge_resolver_timeout_seconds: float | None = None,
    merge_reconciliation_max_merges: int | None = None,
    generated_dirty_repair_enabled: bool = False,
    generated_dirty_repair_commit_subject: str = "",
    generated_dirty_repair_include_submodule_gitlinks: bool = False,
    generated_dirty_repair_max_paths: int | None = None,
    generated_dirty_repair_stale_lock_seconds: float | None = None,
    generated_dirty_repair_paths: Sequence[Path | str] = (),
    worktree_submodule_paths: Sequence[str] = (),
    assumed_completed_task_ids: Sequence[str] = (),
    execution_slice_task_ids: Sequence[str] = (),
    execution_slice_task_cids: Sequence[str] = (),
    log_level: str = "INFO",
) -> list[str]:
    if bool(production_provider_launch_authority_receipt_path) != bool(
        production_provider_launch_authority_receipt_content_id
    ):
        raise ValueError(
            "production provider launch authority path/content identity are required together"
        )
    command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor",
        "--todo-path",
        str(todo_path),
        "--state-dir",
        str(state_dir),
        "--state-prefix",
        state_prefix,
        "--task-prefix",
        task_prefix,
        "--worktree-root",
        str(worktree_root),
        "--daemon-interval",
        str(daemon_interval),
        "--stale-seconds",
        str(stale_seconds),
        "--check-interval",
        str(check_interval),
        "--max-restarts",
        str(max_restarts),
        "--max-task-attempts",
        str(max(0, int(max_task_attempts))),
        "--implementation-timeout",
        str(implementation_timeout),
        "--log-level",
        log_level,
        # Bundle workers receive a digest-bound operational copy. Keep them
        # execution-only so refill/repair code cannot revise reviewed inputs
        # or create shard-local task IDs.
        "--no-retry-budget-guardrail",
        "--no-dependency-guardrail",
        "--no-reconciliation-guardrail",
        "--no-objective-task-janitor",
        "--no-objective-goal-migration",
    ]
    if implementation_max_timeout is not None:
        command.extend(
            [
                "--implementation-max-timeout",
                str(implementation_max_timeout),
            ]
        )
    if watchdog_startup_grace_seconds is not None:
        command.extend(
            [
                "--watchdog-startup-grace-seconds",
                str(watchdog_startup_grace_seconds),
            ]
        )
    for relative in dict.fromkeys(str(path).strip().strip("/") for path in worktree_submodule_paths):
        if relative:
            command.extend(["--worktree-submodule-path", relative])
    command.append("--implement" if implement else "--no-implement")
    if implementation_command:
        command.extend(["--implementation-command", implementation_command])
    if production_provider_policy:
        review_authority_key_path = (
            production_provider_review_authority_key_path
            or state_dir.parent / DEFAULT_PRODUCTION_PROVIDER_REVIEW_KEY_NAME
        )
        command.extend(
            [
                "--production-provider-policy",
                production_provider_policy,
                "--production-provider-context-budget-tokens",
                str(int(production_provider_context_budget_tokens)),
                "--production-provider-timeout-seconds",
                str(
                    float(
                        production_provider_timeout_seconds
                        or DEFAULT_PRODUCTION_PROVIDER_TIMEOUT_SECONDS
                    )
                ),
                "--production-provider-review-authority-key-path",
                str(review_authority_key_path),
            ]
        )
        if production_provider_launch_authority_receipt_path is not None:
            command.extend(
                [
                    "--production-provider-launch-authority-receipt-path",
                    str(production_provider_launch_authority_receipt_path),
                    "--production-provider-launch-authority-receipt-content-id",
                    str(production_provider_launch_authority_receipt_content_id),
                ]
            )
    if (legacy_landed_review_policy_path is None) != (
        legacy_landed_review_key_path is None
    ):
        raise ValueError(
            "legacy landed review requires both explicit policy and key paths"
        )
    legacy_review_enabled = (
        legacy_landed_review_policy_path is not None
        and legacy_landed_review_key_path is not None
    )
    if legacy_review_enabled:
        # Exact historical review can spend the full bounded implementation
        # timeout inside provider calls without emitting child-log output.
        # Keep the overall timeout authoritative while preventing the default
        # 300-second log-stall watchdog from killing healthy legacy review.
        legacy_log_stall_seconds = max(300.0, float(implementation_timeout))
        implementation_log_stall_seconds = max(
            legacy_log_stall_seconds,
            float(implementation_log_stall_seconds or 0.0),
        )
    if implementation_log_stall_seconds is not None:
        command.extend(
            [
                "--implementation-log-stall-seconds",
                str(implementation_log_stall_seconds),
            ]
        )
    if legacy_review_enabled:
        command.extend(
            [
                "--legacy-landed-review-policy-path",
                str(legacy_landed_review_policy_path),
                "--legacy-landed-review-key-path",
                str(legacy_landed_review_key_path),
            ]
        )
    if merge_target_branch:
        command.extend(["--merge-target-branch", merge_target_branch])
    if llm_merge_resolver_command:
        command.extend(["--llm-merge-resolver-command", llm_merge_resolver_command])
    if llm_merge_resolver_timeout_seconds is not None:
        command.extend(["--llm-merge-resolver-timeout-seconds", str(llm_merge_resolver_timeout_seconds)])
    if merge_reconciliation_max_merges is not None:
        command.extend(["--merge-reconciliation-max-merges", str(merge_reconciliation_max_merges)])
    if generated_dirty_repair_commit_subject:
        command.extend(["--generated-dirty-commit-subject", generated_dirty_repair_commit_subject])
    if not generated_dirty_repair_include_submodule_gitlinks:
        command.append("--no-generated-dirty-submodule-gitlinks")
    if generated_dirty_repair_max_paths is not None:
        command.extend(["--generated-dirty-max-paths", str(generated_dirty_repair_max_paths)])
    if generated_dirty_repair_stale_lock_seconds is not None:
        command.extend(["--generated-dirty-stale-lock-seconds", str(generated_dirty_repair_stale_lock_seconds)])
    for path in dict.fromkeys(Path(path) for path in generated_dirty_repair_paths):
        command.extend(["--generated-dirty-path", str(path)])
    for task_id in dict.fromkeys(str(task_id).strip() for task_id in assumed_completed_task_ids):
        if task_id:
            command.extend(["--assume-completed-task-id", task_id])
    for task_id in dict.fromkeys(str(task_id).strip() for task_id in execution_slice_task_ids):
        if task_id:
            command.extend(["--execution-slice-task-id", task_id])
    for task_cid in dict.fromkeys(str(task_cid).strip() for task_cid in execution_slice_task_cids):
        if task_cid:
            command.extend(["--execution-slice-task-cid", task_cid])
    return command


def optimize_bundle_payloads(
    payloads: Sequence[dict[str, Any]],
    *,
    policy: BundleOptimizationPolicy | None = None,
) -> list[dict[str, Any]]:
    """Split current-planner payloads into optimized canonical execution units.

    The current planner remains the baseline and source of authority metadata.
    Optimization only runs when every live member has the canonical identity
    produced by task admission; legacy or partially migrated payloads pass
    through unchanged.  Full member rows remain attached for completion and
    dependency reasoning, while execution-slice fields identify the exact work
    assigned to each optimized lane.
    """

    selected_policy = policy or BundleOptimizationPolicy()
    optimized_payloads: list[dict[str, Any]] = []
    for original in payloads:
        payload = dict(original)
        tasks = _mapping_list(payload.get("tasks"))
        has_execution_slice = (
            "execution_slice_task_cids" in payload
            or "execution_slice_task_ids" in payload
        )
        completed_cids = set(
            _string_list(payload.get("completed_member_task_cids"))
        )
        completed_ids = set(
            _string_list(payload.get("completed_member_task_ids"))
        )
        candidate_tasks = _execution_slice_members(payload, tasks)
        live_tasks = [
            task
            for task in candidate_tasks
            if str(task.get("status") or "").strip().casefold()
            not in _TERMINAL_CONFLICT_TASK_STATUSES
            and str(
                task.get("canonical_task_cid") or task.get("task_cid") or ""
            )
            not in completed_cids
            and str(task.get("task_id") or "") not in completed_ids
        ]
        # The dependency planner's execution slice and durable completion
        # overlay are authoritative inputs to optimization.  Source taskboards
        # are immutable and may still label receipt-completed members ``todo``;
        # optimizing every raw task would otherwise resurrect completed work
        # and admit deferred dependencies.  Normalize the slice even when
        # optimization later passes through due to missing legacy identities.
        if has_execution_slice or completed_cids or completed_ids:
            payload["execution_slice_task_cids"] = [
                str(
                    task.get("canonical_task_cid")
                    or task.get("task_cid")
                    or ""
                )
                for task in live_tasks
                if str(
                    task.get("canonical_task_cid")
                    or task.get("task_cid")
                    or ""
                )
            ]
            payload["execution_slice_task_ids"] = [
                str(task.get("task_id") or "")
                for task in live_tasks
                if str(task.get("task_id") or "")
            ]
        if not live_tasks:
            payload["claimable"] = False
            optimized_payloads.append(payload)
            continue
        if any(
            not str(
                task.get("canonical_task_cid") or task.get("task_cid") or ""
            ).strip()
            or not str(task.get("canonical_task_key") or "").strip()
            for task in live_tasks
        ):
            payload["bundle_optimization"] = {
                "applied": False,
                "reason": "canonical_task_identity_required",
            }
            optimized_payloads.append(payload)
            continue

        normalized: list[dict[str, Any]] = []
        for task in live_tasks:
            member = dict(task)
            for key in (
                "goal_id",
                "merge_family",
                "merge_fate",
                "resource_class",
                "provider_batch_key",
                "provider_id",
                "provider_route",
                "model_id",
                "provider_operation",
                "provider_context_limit",
                "provider_policy_digest",
                "provider_generation_digest",
                "estimated_context_tokens",
                "context_paths",
                "validation_commands",
            ):
                if member.get(key) in (None, "", [], {}):
                    fallback = payload.get(key)
                    if fallback not in (None, "", [], {}):
                        member[key] = fallback
            if not member.get("validation_commands") and member.get("validation"):
                member["validation_commands"] = member["validation"]
            if not member.get("predicted_paths"):
                member["predicted_paths"] = (
                    member.get("predicted_files")
                    or member.get("outputs")
                    or member.get("files")
                    or []
                )
            if not member.get("dependencies"):
                member["dependencies"] = (
                    member.get("dependency_task_cids")
                    or member.get("depends_on")
                    or member.get("graph_parents")
                    or []
                )
            normalized.append(member)

        try:
            result = optimize_task_bundles(
                normalized,
                policy=selected_policy,
                current_planner_bundles=[
                    {
                        "task_cids": [
                            str(
                                task.get("canonical_task_cid")
                                or task.get("task_cid")
                            )
                            for task in live_tasks
                        ],
                        "execution_wave": max(
                            _schedule_int(payload, "optimizer_execution_wave"),
                            _schedule_int(payload, "dependency_depth"),
                            _schedule_int(payload, "graph_depth"),
                        ),
                    }
                ],
            )
        except (TypeError, ValueError) as exc:
            payload["bundle_optimization"] = {
                "applied": False,
                "reason": "invalid_optimizer_input",
                "error": str(exc),
            }
            optimized_payloads.append(payload)
            continue

        task_by_cid = {
            str(task.get("canonical_task_cid") or task.get("task_cid")): task
            for task in live_tasks
        }
        base_key = str(payload.get("bundle_key") or "objective/general")
        result_projection = result.to_dict()
        for bundle in result.bundles:
            projected = dict(payload)
            if len(result.bundles) > 1:
                source_profile = (
                    dict(projected.get("profile_g") or {})
                    if isinstance(projected.get("profile_g"), Mapping)
                    else {}
                )
                if source_profile:
                    projected["source_profile_g_ref"] = {
                        key: str(source_profile.get(key) or "")
                        for key in (
                            "goal_cid",
                            "subgoal_cid",
                            "plan_branch_cid",
                            "selection_cid",
                            "task_cid",
                            "task_spec_cid",
                        )
                        if source_profile.get(key)
                    }
                # One immutable Profile-G TaskSpec cannot identify multiple
                # execution slices.  Let the lease adapter derive a distinct
                # content-addressed chain for each optimized slice.
                projected.pop("profile_g", None)
                projected["bundle_key"] = (
                    f"{base_key}/optimized/{bundle.bundle_cid[-12:]}"
                )
                projected["parallel_lane"] = (
                    f"{str(payload.get('parallel_lane') or base_key)}/"
                    f"{bundle.bundle_cid[-12:]}"
                )
            ids = [
                str(task_by_cid[cid].get("task_id") or "")
                for cid in bundle.task_cids
                if cid in task_by_cid
            ]
            projected["execution_slice_task_cids"] = list(bundle.task_cids)
            projected["execution_slice_task_ids"] = [
                task_id for task_id in ids if task_id
            ]
            projected["dependency_task_cids"] = sorted(
                set(_string_list(payload.get("dependency_task_cids")))
                | set(bundle.dependency_task_cids)
            )
            projected["optimizer_bundle_cid"] = bundle.bundle_cid
            projected["optimizer_policy_id"] = result.policy_id
            projected["optimizer_execution_wave"] = bundle.execution_wave
            projected["bundle_optimization"] = {
                "applied": True,
                "bundle": bundle.to_dict(),
                "metrics": dict(result.metrics),
                "comparison": result.comparison.to_dict(),
                "packet_aggregates": [
                    aggregate.to_dict()
                    for aggregate in result.packet_aggregates
                    if aggregate.aggregate_task_cid in bundle.task_cids
                ],
                "result_schema": result_projection["schema"],
            }
            projected["critical_path_length"] = max(
                _schedule_int(payload, "critical_path_length"),
                int(result.metrics.get("critical_path_wave_count", 0))
                - bundle.execution_wave,
            )
            projected["schedule_rank"] = (
                bundle.execution_wave * 1_000_000
                + _schedule_int(payload, "schedule_rank")
            )
            optimized_payloads.append(projected)
    return optimized_payloads


def plan_bundle_lanes(
    *,
    bundle_index_path: Path,
    repo_root: Path,
    state_root: Path,
    worktree_root: Path,
    log_dir: Path,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    implement: bool = False,
    daemon_interval: float = 300.0,
    stale_seconds: float = 1800.0,
    check_interval: float = 60.0,
    watchdog_startup_grace_seconds: float | None = None,
    max_restarts: int = 10,
    implementation_timeout: float = 1800.0,
    implementation_max_timeout: float | None = None,
    implementation_log_stall_seconds: float | None = None,
    max_task_attempts: int = 0,
    implementation_command: str = "",
    production_provider_policy: str = "",
    production_provider_context_budget_tokens: int = 0,
    production_provider_timeout_seconds: float = 0.0,
    production_provider_review_authority_key_path: Path | None = None,
    production_provider_launch_authority_receipt_path: Path | None = None,
    production_provider_launch_authority_receipt_content_id: str = "",
    legacy_landed_review_policy_path: Path | None = None,
    legacy_landed_review_key_path: Path | None = None,
    merge_target_branch: str = "",
    llm_merge_resolver_command: str = "",
    llm_merge_resolver_timeout_seconds: float | None = None,
    merge_reconciliation_max_merges: int | None = None,
    generated_dirty_repair_enabled: bool = False,
    generated_dirty_repair_commit_subject: str = "",
    generated_dirty_repair_include_submodule_gitlinks: bool = False,
    generated_dirty_repair_max_paths: int | None = None,
    generated_dirty_repair_stale_lock_seconds: float | None = None,
    generated_dirty_repair_paths: Sequence[Path | str] = (),
    worktree_submodule_paths: Sequence[str] = (),
    log_level: str = "INFO",
    max_lanes: int | None = None,
    completion_receipts: Mapping[str, Any] | None = None,
    optimize_bundles: bool = True,
    bundle_optimization_policy: BundleOptimizationPolicy | None = None,
    excluded_bundle_keys: Sequence[str] = (),
    excluded_task_ids: Sequence[str] = (),
) -> list[BundleLaneSpec]:
    """Return one isolated supervisor command for each objective bundle."""

    lanes: list[BundleLaneSpec] = []
    if (
        production_provider_policy
        and production_provider_review_authority_key_path is None
    ):
        production_provider_review_authority_key_path = (
            state_root / DEFAULT_PRODUCTION_PROVIDER_REVIEW_KEY_NAME
        )
    legacy_review_policy: LegacyLandedReviewPolicy | None = None
    legacy_review_task_ids: frozenset[str] = frozenset()
    legacy_review_context_tokens = 0
    if legacy_landed_review_policy_path is not None:
        legacy_review_policy = load_legacy_landed_review_policy(
            legacy_landed_review_policy_path
        )
        if legacy_review_policy.enabled:
            legacy_review_task_ids = frozenset(
                item.task_id for item in legacy_review_policy.tasks
            )
            legacy_review_context_tokens = int(
                legacy_review_policy.max_leaf_tokens
            )
    if completion_receipts is None:
        completion_receipts = bundle_member_completion_receipts(state_root)
    planning_completion_receipts: Mapping[str, Any] = completion_receipts
    if legacy_review_policy is not None and legacy_review_policy.enabled:
        planning_completion_receipts = _legacy_adoption_planning_receipts(
            legacy_review_policy,
            completion_receipts,
        )
    if planning_completion_receipts:
        bundle_payloads = build_bundle_task_payloads(
            bundle_index_path,
            merge_receipts=planning_completion_receipts,
            max_attempts=max_task_attempts,
        )
    elif max_task_attempts:
        # Preserve the legacy single-argument injection point for the default
        # unlimited policy while forwarding every finite supervisor ceiling
        # into the immutable Profile-G TaskSpec.
        bundle_payloads = build_bundle_task_payloads(
            bundle_index_path,
            max_attempts=max_task_attempts,
        )
    else:
        # Keep the legacy single-argument call path for integrations which
        # inject a planner and have no durable receipt overlay to apply.
        bundle_payloads = build_bundle_task_payloads(bundle_index_path)
    if legacy_review_policy is not None and legacy_review_policy.enabled:
        bundle_payloads = _legacy_adoption_barrier_payloads(
            bundle_payloads,
            policy=legacy_review_policy,
            completion_receipts=completion_receipts,
        )
    globally_completed_task_ids = {
        str(task.get("task_id") or "")
        for payload in bundle_payloads
        for task in _mapping_list(payload.get("tasks"))
        if str(task.get("status") or "").strip().lower()
        in _TERMINAL_CONFLICT_TASK_STATUSES
        and str(task.get("task_id") or "")
    }
    globally_completed_task_ids.update(
        task_id
        for payload in bundle_payloads
        for task_id in _string_list(payload.get("completed_member_task_ids"))
    )
    excluded_bundle_keys = {
        *_excluded_bundle_keys(bundle_index_path),
        *(
            str(bundle_key).strip()
            for bundle_key in excluded_bundle_keys
            if str(bundle_key).strip()
        ),
    }
    bundle_payloads = [
        payload
        for payload in bundle_payloads
        if str(payload.get("bundle_key") or "objective/general") not in excluded_bundle_keys
        and payload.get("is_schedulable") is not False
        and payload.get("review_only") is not True
    ]
    bundle_payloads = _apply_runtime_task_exclusions(
        bundle_payloads,
        excluded_task_ids=excluded_task_ids,
    )
    if optimize_bundles:
        bundle_payloads = optimize_bundle_payloads(
            bundle_payloads,
            policy=bundle_optimization_policy,
        )
    conflict_annotations = _bundle_conflict_annotations(
        bundle_payloads,
        bundle_index_path=bundle_index_path,
        repo_root=repo_root,
        task_prefix=task_prefix,
    )
    for payload in bundle_payloads:
        bundle_key = str(payload.get("bundle_key") or "objective/general")
        conflict_annotation = conflict_annotations.get(bundle_key, {})
        safe_key = safe_bundle_key(bundle_key)
        todo_path = resolve_repo_path(repo_root, str(payload.get("todo_path") or ""))
        state_dir = state_root / safe_key / "state"
        runtime_todo_path = (
            state_dir / f"{lane_state_prefix(bundle_key)}_runtime.todo.md"
        )
        source_todo_sha256 = _taskboard_sha256(todo_path)
        lane_worktree_root = worktree_root / safe_key
        log_path = log_dir / f"{safe_key}.log"
        state_prefix = lane_state_prefix(bundle_key)
        execution_tasks = _execution_slice_members(
            payload,
            _mapping_list(payload.get("tasks")),
        )
        assumed_completed_task_ids = sorted(
            {
                dependency_id
                for task in execution_tasks
                for dependency_id in _string_list(task.get("depends_on"))
                if dependency_id in globally_completed_task_ids
            }
        )
        task_ids = (
            _string_list(payload.get("execution_slice_task_ids"))
            if "execution_slice_task_ids" in payload
            else [
                str(item.get("task_id"))
                for item in payload.get("tasks", [])
                if isinstance(item, dict) and item.get("task_id")
            ]
        )
        task_cids = (
            _string_list(payload.get("execution_slice_task_cids"))
            if "execution_slice_task_cids" in payload
            else [
                str(item.get("canonical_task_cid") or item.get("task_cid"))
                for item in execution_tasks
                if str(item.get("canonical_task_cid") or item.get("task_cid") or "")
            ]
        )
        expected_task_cids_by_id = _execution_slice_task_cids_by_id(
            payload,
            execution_tasks,
            task_ids=task_ids,
            task_cids=task_cids,
        )
        profile_g = payload.get("profile_g") if isinstance(payload.get("profile_g"), dict) else {}
        resource_fields = _resource_lane_fields(payload)
        slice_implementation_max_timeout = (
            _execution_slice_implementation_max_timeout(
                payload,
                default_timeout=implementation_timeout,
            )
        )
        if implementation_max_timeout is not None:
            if (
                isinstance(implementation_max_timeout, bool)
                or not isinstance(implementation_max_timeout, (int, float))
                or not math.isfinite(float(implementation_max_timeout))
                or float(implementation_max_timeout) <= 0
            ):
                raise ValueError(
                    "implementation_max_timeout must be finite and positive"
                )
            slice_implementation_max_timeout = max(
                slice_implementation_max_timeout,
                float(implementation_max_timeout),
            )
        legacy_dual_review = bool(
            legacy_review_task_ids.intersection(task_ids)
        )
        production_dual_review = bool(
            production_provider_policy
            and _production_lane_requires_dual_review(
                execution_tasks,
                resource_fields,
            )
        )
        if implement and (production_dual_review or legacy_dual_review):
            dual_review_context_tokens = (
                int(production_provider_context_budget_tokens)
                or DEFAULT_PRODUCTION_CONTEXT_BUDGET_TOKENS
                if production_provider_policy
                else 0
            )
            if legacy_dual_review:
                dual_review_context_tokens = max(
                    dual_review_context_tokens,
                    legacy_review_context_tokens,
                )
            resource_fields = _dual_review_resource_fields(
                resource_fields,
                context_budget_tokens=dual_review_context_tokens,
            )
        command = implementation_supervisor_command(
            todo_path=runtime_todo_path,
            state_dir=state_dir,
            worktree_root=lane_worktree_root,
            state_prefix=state_prefix,
            task_prefix=task_prefix,
            implement=implement,
            daemon_interval=daemon_interval,
            stale_seconds=stale_seconds,
            check_interval=check_interval,
            watchdog_startup_grace_seconds=watchdog_startup_grace_seconds,
            max_restarts=max_restarts,
            implementation_timeout=implementation_timeout,
            implementation_max_timeout=(
                slice_implementation_max_timeout
                if slice_implementation_max_timeout
                > float(implementation_timeout)
                else None
            ),
            implementation_log_stall_seconds=(
                implementation_log_stall_seconds
            ),
            max_task_attempts=max_task_attempts,
            implementation_command=implementation_command,
            production_provider_policy=production_provider_policy,
            production_provider_context_budget_tokens=(
                production_provider_context_budget_tokens
            ),
            production_provider_timeout_seconds=(
                production_provider_timeout_seconds
            ),
            production_provider_review_authority_key_path=(
                production_provider_review_authority_key_path
            ),
            production_provider_launch_authority_receipt_path=(
                production_provider_launch_authority_receipt_path
            ),
            production_provider_launch_authority_receipt_content_id=(
                production_provider_launch_authority_receipt_content_id
            ),
            legacy_landed_review_policy_path=legacy_landed_review_policy_path,
            legacy_landed_review_key_path=legacy_landed_review_key_path,
            merge_target_branch=merge_target_branch,
            llm_merge_resolver_command=llm_merge_resolver_command,
            llm_merge_resolver_timeout_seconds=llm_merge_resolver_timeout_seconds,
            merge_reconciliation_max_merges=merge_reconciliation_max_merges,
            generated_dirty_repair_enabled=generated_dirty_repair_enabled,
            generated_dirty_repair_commit_subject=generated_dirty_repair_commit_subject,
            generated_dirty_repair_include_submodule_gitlinks=generated_dirty_repair_include_submodule_gitlinks,
            generated_dirty_repair_max_paths=generated_dirty_repair_max_paths,
            generated_dirty_repair_stale_lock_seconds=generated_dirty_repair_stale_lock_seconds,
            generated_dirty_repair_paths=generated_dirty_repair_paths,
            worktree_submodule_paths=worktree_submodule_paths,
            assumed_completed_task_ids=assumed_completed_task_ids,
            execution_slice_task_ids=task_ids,
            execution_slice_task_cids=task_cids,
            log_level=log_level,
        )
        lanes.append(
            BundleLaneSpec(
                bundle_key=bundle_key,
                parallel_lane=str(payload.get("parallel_lane") or bundle_key),
                todo_path=todo_path,
                state_dir=state_dir,
                worktree_root=lane_worktree_root,
                state_prefix=state_prefix,
                task_ids=task_ids,
                conflict_policy=str(payload.get("conflict_policy") or ""),
                command=command,
                log_path=log_path,
                expected_task_cids_by_id=expected_task_cids_by_id,
                runtime_todo_path=runtime_todo_path,
                source_todo_sha256=source_todo_sha256,
                source_todo=str(payload.get("source_todo") or ""),
                task_cid=str(profile_g.get("task_cid") or ""),
                goal_cid=str(profile_g.get("goal_cid") or ""),
                subgoal_cid=str(profile_g.get("subgoal_cid") or ""),
                queue_payload=dict(payload),
                schedule_rank=(_schedule_int(payload, "schedule_rank") if payload.get("schedule_rank") is not None else None),
                claimable=_schedule_bool(payload, "claimable"),
                dependency_task_cids=_string_list(payload.get("dependency_task_cids")),
                blocking_task_cids=_string_list(payload.get("blocking_task_cids")),
                critical_path_length=_schedule_int(payload, "critical_path_length"),
                slack=_schedule_int(payload, "slack"),
                downstream_unlock_value=_schedule_int(payload, "downstream_unlock_value"),
                age_seconds=_schedule_int(payload, "age_seconds"),
                objective_priority=_schedule_int(payload, "objective_priority"),
                schedule_score=_schedule_int(payload, "schedule_score"),
                dependency_repair_evidence=[dict(item) for item in (payload.get("dependency_repair_evidence") or []) if isinstance(item, dict)],
                conflict_color=conflict_annotation.get("conflict_color"),
                conflicting_task_ids=_string_list(conflict_annotation.get("conflicting_task_ids")),
                conflict_decisions=_mapping_list(conflict_annotation.get("conflict_decisions")),
                conflict_surface=dict(conflict_annotation.get("conflict_surface") or {}),
                optimizer_bundle_cid=str(payload.get("optimizer_bundle_cid") or ""),
                optimizer_policy_id=str(payload.get("optimizer_policy_id") or ""),
                optimizer_execution_wave=_schedule_int(
                    payload, "optimizer_execution_wave"
                ),
                optimization_metrics=dict(
                    (
                        payload.get("bundle_optimization") or {}
                    ).get("metrics")
                    or {}
                )
                if isinstance(payload.get("bundle_optimization"), Mapping)
                else {},
                planner_comparison=dict(
                    (
                        payload.get("bundle_optimization") or {}
                    ).get("comparison")
                    or {}
                )
                if isinstance(payload.get("bundle_optimization"), Mapping)
                else {},
                packet_aggregates=[
                    dict(item)
                    for item in (
                        (
                            payload.get("bundle_optimization") or {}
                        ).get("packet_aggregates")
                        or []
                    )
                    if isinstance(item, Mapping)
                ]
                if isinstance(payload.get("bundle_optimization"), Mapping)
                else [],
                implementation_max_timeout=(
                    slice_implementation_max_timeout
                ),
                **resource_fields,
            )
        )
    lanes.sort(key=_lane_schedule_key)
    return lanes[:max_lanes] if max_lanes is not None else lanes


def _lane_launch_policy_error(lane: BundleLaneSpec) -> str:
    """Return why a lane lacks a concrete, schedulable execution slice."""

    payload = lane.queue_payload
    if not isinstance(payload, dict):
        return "missing queue payload"
    execution_authority = str(
        payload.get("execution_authority") or INTERNAL_EXECUTION_AUTHORITY
    ).strip()
    if execution_authority != INTERNAL_EXECUTION_AUTHORITY:
        return (
            "bundle requires external execution authority "
            f"{execution_authority!r}"
        )
    if payload.get("is_schedulable") is not True:
        return "bundle is not schedulable"
    if payload.get("review_only") is not False:
        return "bundle is review-only"
    if not lane.task_ids:
        return "lane has no execution task ids"

    has_execution_slice = (
        "execution_slice_task_cids" in payload
        or "execution_slice_task_ids" in payload
    )
    execution_cids = _string_list(payload.get("execution_slice_task_cids"))
    execution_ids = _string_list(payload.get("execution_slice_task_ids"))
    if not has_execution_slice or not (execution_cids or execution_ids):
        return "bundle has no authorized execution slice"

    execution_tasks = _execution_slice_members(
        payload,
        _mapping_list(payload.get("tasks")),
    )
    if not execution_tasks:
        return "authorized execution slice has no task records"
    execution_task_ids = {
        str(task.get("task_id") or "")
        for task in execution_tasks
        if str(task.get("task_id") or "")
    }
    if not set(lane.task_ids).issubset(execution_task_ids):
        return "lane task ids are outside the authorized execution slice"
    for task in execution_tasks:
        task_execution_authority = str(
            task.get("execution_authority") or INTERNAL_EXECUTION_AUTHORITY
        ).strip()
        if task_execution_authority != INTERNAL_EXECUTION_AUTHORITY:
            return (
                "execution slice requires external execution authority "
                f"{task_execution_authority!r}"
            )
        if not _schedule_bool(task, "is_schedulable", True):
            return "execution slice contains a non-schedulable task"
        if _schedule_bool(task, "review_only", False):
            return "execution slice contains a review-only task"
        status = str(task.get("status") or "todo").strip().lower()
        if status in {
            "blocked",
            "on_hold",
            "complete",
            "completed",
            "done",
            "merged",
            "passed",
            "success",
            "succeeded",
            "verified_complete",
        }:
            return f"execution slice contains terminal task status {status}"
    return ""


def _legacy_adoption_lane_blocked(lane: BundleLaneSpec) -> bool:
    payload = lane.queue_payload
    barrier = (
        payload.get("legacy_adoption_barrier")
        if isinstance(payload, Mapping)
        and isinstance(payload.get("legacy_adoption_barrier"), Mapping)
        else {}
    )
    return bool(
        barrier.get("active") is True
        and payload.get("blocked_reason")
        == LEGACY_ADOPTION_BARRIER_REASON
    )


def _receipt_drained_execution_slice(lane: BundleLaneSpec) -> bool:
    """Return whether durable member receipts drained the entire lane slice."""

    payload = lane.queue_payload
    if (
        lane.task_ids
        or not isinstance(payload, dict)
        or payload.get("claimable") is not False
        or payload.get("external_active_member_fence") is True
        or "execution_slice_task_cids" not in payload
        or "execution_slice_task_ids" not in payload
        or _string_list(payload.get("execution_slice_task_cids"))
        or _string_list(payload.get("execution_slice_task_ids"))
    ):
        return False
    completed_ids = set(_string_list(payload.get("completed_member_task_ids")))
    completed_cids = set(_string_list(payload.get("completed_member_task_cids")))
    tasks = _mapping_list(payload.get("tasks"))
    if not tasks or not (completed_ids or completed_cids):
        return False
    return all(
        str(task.get("task_id") or "") in completed_ids
        or str(task.get("canonical_task_cid") or task.get("task_cid") or "")
        in completed_cids
        for task in tasks
    )


def launch_bundle_lanes(
    lanes: Sequence[BundleLaneSpec],
    *,
    repo_root: Path,
    coordination_path: Path | None = None,
    claimant_did: str = "did:web:ipfs-accelerate.local",
    lease_ms: int = 60_000,
    heartbeat_interval: float = 5.0,
    capacity_millionths: int = 1_000_000,
) -> list[dict[str, Any]]:
    """Claim and launch lane supervisors under accepted, fenced leases."""

    policy_errors = {
        id(lane): _lane_launch_policy_error(lane)
        for lane in lanes
    }
    stale_input_bindings: dict[int, dict[str, Any] | None] = {}
    for lane in lanes:
        diagnosis = stale_bundle_lane_input_binding(
            lane,
            repo_root=repo_root,
        )
        if diagnosis is None:
            stale_input_bindings[id(lane)] = None
            continue
        refreshed = refresh_stale_bundle_lane_input_binding(
            lane,
            repo_root=repo_root,
        )
        if refreshed and refreshed.get("refreshed"):
            logger.info(
                "Auto-refreshed stale taskboard binding for %s before launch",
                lane.bundle_key,
            )
            stale_input_bindings[id(lane)] = None
            continue
        if refreshed is not None and not refreshed.get("refreshed"):
            diagnosis = {
                **diagnosis,
                "refresh_error": str(refreshed.get("error") or "refresh_failed"),
            }
        stale_input_bindings[id(lane)] = diagnosis
    if lanes and all(policy_errors[id(lane)] for lane in lanes):
        return [
            {
                "bundle_key": lane.bundle_key,
                "accepted": False,
                "error": policy_errors[id(lane)],
                "code": "G_EXECUTION_POLICY_DENIED",
            }
            for lane in lanes
        ]

    results: list[dict[str, Any]] = []
    active_lanes: list[BundleLaneSpec] = []
    path = coordination_path or default_state_root(repo_root) / "coordination.duckdb"
    with LeaseCoordinator(path) as coordinator:
        for lane in lanes:
            policy_error = policy_errors[id(lane)]
            if policy_error:
                results.append(
                    {
                        "bundle_key": lane.bundle_key,
                        "accepted": False,
                        "error": policy_error,
                        "code": "G_EXECUTION_POLICY_DENIED",
                    }
                )
                continue
            stale_input_binding = stale_input_bindings[id(lane)]
            if stale_input_binding is not None:
                results.append(
                    {
                        "bundle_key": lane.bundle_key,
                        "accepted": False,
                        "error": (
                            "immutable runtime taskboard input is bound to a "
                            "different planned source digest"
                        ),
                        "code": "G_STALE_INPUT_BINDING",
                        **stale_input_binding,
                    }
                )
                continue
            blockers = [active.bundle_key for active in active_lanes if _lanes_conflict(lane, active)]
            if blockers:
                results.append(
                    {
                        "bundle_key": lane.bundle_key,
                        "accepted": False,
                        "error": "conflicts with an active lane",
                        "code": "G_LANE_CONFLICT",
                        "blocking_bundle_keys": blockers,
                        "conflict_color": lane.conflict_color,
                        "decisions": lane.conflict_decisions,
                    }
                )
                continue
            assert lane.queue_payload is not None
            adapted = coordinator.register_bundle(lane.queue_payload)
            try:
                grant = coordinator.claim(adapted["task_cid"], claimant_did, requested_lease_ms=lease_ms)
            except LeaseError as exc:
                rejected: dict[str, Any] = {
                    "bundle_key": lane.bundle_key,
                    "accepted": False,
                    "error": str(exc),
                    "code": exc.code,
                }
                evidence = getattr(exc, "evidence", None)
                if isinstance(evidence, dict):
                    rejected["dependency_evidence"] = dict(evidence)
                results.append(rejected)
                continue
            try:
                process, guarded_command, pid_path = _spawn_accepted_lane(
                    lane,
                    grant,
                    repo_root=repo_root,
                    coordination_path=path,
                    lease_ms=lease_ms,
                    heartbeat_interval=heartbeat_interval,
                    capacity_millionths=capacity_millionths,
                )
            except Exception:
                coordinator.release(grant, reason="launch failed")
                raise
            results.append(
                {
                    "bundle_key": lane.bundle_key, "accepted": True, "pid": process.pid,
                    "pid_path": repo_relative_path(repo_root, pid_path), "log_path": repo_relative_path(repo_root, lane.log_path),
                    "command": guarded_command, "lease": grant.to_dict(),
                }
            )
            active_lanes.append(lane)
    return results


def _allow_concurrent_peer_ids(lane: BundleLaneSpec) -> set[str]:
    """Return bundle keys / task ids this lane explicitly allows concurrent with."""

    peers: set[str] = set()
    payload = lane.queue_payload if isinstance(lane.queue_payload, dict) else {}
    raw_values: list[Any] = []
    for key in ("allow_concurrent_with", "concurrency_overrides"):
        value = payload.get(key)
        if isinstance(value, (list, tuple, set, frozenset)):
            raw_values.extend(value)
        elif isinstance(value, str) and value.strip():
            raw_values.append(value)
    for task in payload.get("tasks") or ():
        if not isinstance(task, Mapping):
            continue
        for key in ("allow_concurrent_with", "concurrency_overrides"):
            value = task.get(key)
            if isinstance(value, (list, tuple, set, frozenset)):
                raw_values.extend(value)
            elif isinstance(value, str) and value.strip():
                raw_values.append(value)
    surface = lane.conflict_surface if isinstance(lane.conflict_surface, Mapping) else {}
    surface_allowed = surface.get("allow_concurrent_with")
    if isinstance(surface_allowed, (list, tuple, set, frozenset)):
        raw_values.extend(surface_allowed)
    for item in raw_values:
        if isinstance(item, Mapping):
            for key in (
                "bundle_key",
                "task_id",
                "right",
                "left",
                "with",
                "peer",
            ):
                text = str(item.get(key) or "").strip()
                if text:
                    peers.add(text)
            continue
        text = str(item or "").strip()
        if text:
            peers.add(text)
    return peers


def _lanes_allow_concurrent(left: BundleLaneSpec, right: BundleLaneSpec) -> bool:
    """Return whether either lane declares the other safe to co-schedule."""

    left_peers = _allow_concurrent_peer_ids(left)
    right_peers = _allow_concurrent_peer_ids(right)
    if right.bundle_key in left_peers or left.bundle_key in right_peers:
        return True
    if any(task_id in left_peers for task_id in right.task_ids):
        return True
    if any(task_id in right_peers for task_id in left.task_ids):
        return True
    return False


def _lanes_conflict(left: BundleLaneSpec, right: BundleLaneSpec) -> bool:
    """Return whether graph edges prohibit two currently active lanes."""

    if _lanes_allow_concurrent(left, right):
        return False
    return (
        right.bundle_key in left.conflicting_task_ids
        or left.bundle_key in right.conflicting_task_ids
    )


def _lane_conflict_manifest(lanes: Sequence[BundleLaneSpec]) -> dict[str, Any]:
    """Serialize the bundle projection with one explanation per decision."""

    unique_decisions: dict[str, dict[str, Any]] = {}
    edge_pairs: set[tuple[str, str]] = set()
    color_lanes: dict[str, list[str]] = {}
    surfaces: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        if lane.conflict_color is not None:
            color_lanes.setdefault(str(lane.conflict_color), []).append(lane.bundle_key)
        if lane.conflict_surface:
            surfaces[lane.bundle_key] = dict(lane.conflict_surface)
        for peer in lane.conflicting_task_ids:
            edge_pairs.add(tuple(sorted((lane.bundle_key, peer))))
        for decision in lane.conflict_decisions:
            key = json.dumps(decision, sort_keys=True, default=str)
            unique_decisions.setdefault(key, dict(decision))
    return {
        "color_count": len(color_lanes),
        "assignments": [
            {"task_id": lane.bundle_key, "lane_color": lane.conflict_color}
            for lane in lanes
            if lane.conflict_color is not None
        ],
        "lanes": {color: sorted(task_ids) for color, task_ids in sorted(color_lanes.items())},
        "surfaces": surfaces,
        "edges": [
            {"left_task_id": left, "right_task_id": right, "blocks_concurrency": True}
            for left, right in sorted(edge_pairs)
        ],
        "decisions": list(unique_decisions.values()),
    }


def _spawn_accepted_lane(
    lane: BundleLaneSpec,
    grant: Any,
    *,
    repo_root: Path,
    coordination_path: Path,
    lease_ms: int,
    heartbeat_interval: float,
    capacity_millionths: int,
) -> tuple[subprocess.Popen[bytes], list[str], Path]:
    """Start one already-claimed lane with its exact member identity fence."""

    expected_task_ids = tuple(
        dict.fromkeys(
            str(task_id).strip()
            for task_id in lane.task_ids
            if str(task_id).strip()
        )
    )
    expected_task_cids_by_id = {
        str(task_id).strip(): str(task_cid).strip()
        for task_id, task_cid in lane.expected_task_cids_by_id.items()
        if str(task_id).strip() and str(task_cid).strip()
    }
    if set(expected_task_cids_by_id) != set(expected_task_ids):
        raise ValueError(
            f"bundle lane {lane.bundle_key!r} lacks an exact canonical "
            "identity for every execution-slice task"
        )
    lane.state_dir.mkdir(parents=True, exist_ok=True)
    lane.worktree_root.mkdir(parents=True, exist_ok=True)
    lane.log_path.parent.mkdir(parents=True, exist_ok=True)
    materialize_bundle_lane_taskboard(lane, repo_root=repo_root)
    guarded_command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.merge.leased_lane",
        "--coordination-path",
        str(coordination_path),
        "--grant-json",
        json.dumps(grant.to_dict(), sort_keys=True),
        "--lease-ms",
        str(lease_ms),
        "--heartbeat-interval",
        str(heartbeat_interval),
        "--capacity-millionths",
        str(capacity_millionths),
        "--resource-class",
        lane.resource_class,
        "--provider-id",
        lane.llm_provider,
        "--phase-state-path",
        str(lane.state_dir / f"{lane.state_prefix}_task_state.json"),
        "--completion-events-path",
        str(lane.state_dir / f"{lane.state_prefix}_events.jsonl"),
    ]
    for task_id in expected_task_ids:
        guarded_command.extend(["--expected-task-id", task_id])
        guarded_command.extend(
            [
                "--expected-task-identity-json",
                json.dumps(
                    {
                        "task_id": task_id,
                        "canonical_task_cid": expected_task_cids_by_id[task_id],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    guarded_command.extend(["--", *lane.command])
    env = os.environ.copy()
    env["PYTHONPATH"] = _bundle_lane_pythonpath(
        repo_root,
        existing=env.get("PYTHONPATH", ""),
    )
    handle = lane.log_path.open("ab")
    try:
        process = subprocess.Popen(
            guarded_command,
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        handle.close()
    pid_path = lane.state_dir / f"{lane.state_prefix}_bundle_supervisor.pid"
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return process, guarded_command, pid_path


def _bundle_lane_pythonpath(repo_root: Path, *, existing: str = "") -> str:
    """Keep child lanes on the same agent-supervisor implementation as the parent."""

    # objectives/bundle_supervisor.py -> agent_supervisor -> ipfs_accelerate_py -> install/repo root
    runtime_package_root = Path(__file__).resolve().parents[3]
    legacy_package_root = repo_root / "ipfs_datasets_py" / "ipfs_accelerate_py"
    entries = [str(runtime_package_root)]
    if legacy_package_root.exists():
        entries.append(str(legacy_package_root.resolve()))
    entries.extend(item for item in existing.split(os.pathsep) if item)
    return os.pathsep.join(dict.fromkeys(entries))


def check_lane_health(
    lanes: Sequence[BundleLaneSpec],
    *,
    repo_root: Path,
    coordination_path: Path | None = None,
    claimant_did: str = "did:web:ipfs-accelerate.local",
) -> list[dict[str, Any]]:
    """Check health of all launched lanes and restart any dead ones.

    Returns a list of health reports, one per lane. Dead lanes are
    automatically relaunched so the bundle supervisor can run indefinitely.
    """
    import os

    reports: list[dict[str, Any]] = []
    for lane in lanes:
        pid_path = lane.state_dir / f"{lane.state_prefix}_bundle_supervisor.pid"
        report: dict[str, Any] = {"bundle_key": lane.bundle_key, "alive": False, "restarted": False}

        if not pid_path.exists():
            report["reason"] = "no_pid_file"
        else:
            try:
                pid = int(pid_path.read_text().strip())
                # Check if process is alive
                os.kill(pid, 0)
                report["alive"] = True
                report["pid"] = pid
            except (ValueError, ProcessLookupError, PermissionError):
                report["reason"] = "process_dead"
            except OSError:
                report["reason"] = "check_failed"

        if not report["alive"]:
            # Restart the dead lane
            try:
                started = launch_bundle_lanes([lane], repo_root=repo_root, coordination_path=coordination_path, claimant_did=claimant_did)
                if started and started[0].get("accepted"):
                    report["restarted"] = True
                    report["new_pid"] = started[0]["pid"]
                    logger.info("Restarted dead lane %s with PID %d", lane.bundle_key, started[0]["pid"])
                else:
                    report["restart_error"] = str(started[0].get("error") if started else "lease not accepted")
            except OSError as exc:
                report["restart_error"] = str(exc)
                logger.error("Failed to restart lane %s: %s", lane.bundle_key, exc)

        reports.append(report)
    return reports


def write_bundle_lane_manifest(
    *,
    manifest_path: Path,
    repo_root: Path,
    bundle_index_path: Path,
    lanes: Sequence[BundleLaneSpec],
    started: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    detailed_lanes = [_lane_database_payload(lane, repo_root=repo_root) for lane in lanes]
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor",
        "generated_at": utc_now(),
        "repo_root": str(repo_root),
        "bundle_index_path": repo_relative_path(repo_root, bundle_index_path),
        "planned_count": len(lanes),
        "claimable_count": sum(lane.claimable for lane in lanes),
        "blocked_count": sum(not lane.claimable for lane in lanes),
        "started_count": len(started),
        "critical_path": [lane.bundle_key for lane in lanes if lane.claimable],
        "conflict_graph": _lane_conflict_manifest(lanes),
        "lanes": [_lane_manifest_payload(lane, repo_root=repo_root) for lane in lanes],
        "started": list(started),
    }
    database_payload = {**payload, "lanes": detailed_lanes}
    return write_scheduler_manifest_artifact(
        manifest_path,
        payload,
        database_payload=database_payload,
    )


def default_state_root(repo_root: Path) -> Path:
    return repo_root / "data" / "agent_supervisor" / "bundle_lanes"


class DynamicBundleScheduler:
    """Persistent, capacity-bounded reconciler for objective bundle workers.

    The bundle index is an input stream, not a launch snapshot.  Every
    reconciliation rereads it, durably registers all discovered work, reaps
    lanes that no longer execute, and claims enough ready work to fill the
    configured capacity. DuckDB leases remain the sole execution authority;
    the in-memory process table is only a live-process projection.
    """

    def __init__(
        self,
        *,
        bundle_index_path: Path,
        repo_root: Path,
        state_root: Path | None = None,
        worktree_root: Path | None = None,
        log_dir: Path | None = None,
        manifest_path: Path | None = None,
        metrics_path: Path | None = None,
        coordination_path: Path | None = None,
        max_lanes: int = 1,
        claimant_did: str = "did:web:ipfs-accelerate.local",
        lease_ms: int = 60_000,
        heartbeat_interval: float = 5.0,
        capacity_millionths: int = 1_000_000,
        poll_interval: float = 5.0,
        launcher: Callable[[BundleLaneSpec, Any], Any] | None = None,
        process_alive: Callable[[Any], bool] | None = None,
        lane_disposition: Callable[[BundleLaneSpec], str | bool] | None = None,
        resource_scheduler: ResourceScheduler | None = None,
        host_resource_source: Callable[..., HostResourceSnapshot | dict[str, Any]] | None = None,
        provider_capacity_source: Callable[..., Any] | None = None,
        provider_capacity_path: Path | None = None,
        provider_capacity_max_age_ms: int = DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS,
        external_task_state_paths: Sequence[Path | str] = (),
        bundle_index_refresh_command: str = "",
        bundle_index_refresh_timeout_seconds: float = 60.0,
        bundle_index_refresher: Callable[[], Any] | None = None,
        resource_policy: ResourcePolicy | dict[str, Any] | None = None,
        **lane_options: Any,
    ) -> None:
        if int(max_lanes) < 1:
            raise ValueError("max_lanes must be at least 1")
        self.repo_root = Path(repo_root).resolve()
        self.bundle_index_path = Path(bundle_index_path).resolve()
        self.state_root = Path(state_root or default_state_root(self.repo_root)).resolve()
        self.worktree_root = Path(worktree_root or self.state_root / "worktrees").resolve()
        self.log_dir = Path(log_dir or self.state_root / "logs").resolve()
        self.manifest_path = Path(manifest_path or self.state_root / "bundle_lanes.json").resolve()
        self.metrics_path = Path(metrics_path or self.state_root / "scheduler_metrics.json").resolve()
        self.decision_metrics_path = self.metrics_path.with_name("scheduler_decision_metrics.json")
        self.coordination_path = Path(
            coordination_path or self.state_root / "coordination.duckdb"
        ).resolve()
        self.max_lanes = int(max_lanes)
        self.claimant_did = str(claimant_did)
        self.lease_ms = int(lease_ms)
        self.heartbeat_interval = float(heartbeat_interval)
        self.capacity_millionths = int(capacity_millionths)
        self.poll_interval = max(0.0, float(poll_interval))
        self.lane_options = dict(lane_options)
        self._private_dual_review_capacity_required = bool(
            str(self.lane_options.get("production_provider_policy") or "").strip()
        )
        legacy_policy_path = self.lane_options.get(
            "legacy_landed_review_policy_path"
        )
        if legacy_policy_path is not None:
            legacy_policy = load_legacy_landed_review_policy(
                Path(legacy_policy_path)
            )
            self._private_dual_review_capacity_required = bool(
                self._private_dual_review_capacity_required
                or legacy_policy.enabled
            )
        self._launcher = launcher or self._default_launcher
        self._process_alive = process_alive or self._default_process_alive
        self._lane_disposition = lane_disposition or self._default_lane_disposition
        if resource_scheduler is not None:
            self.resource_scheduler = resource_scheduler
        else:
            policy_values = dict(resource_policy or {}) if isinstance(resource_policy, dict) else None
            if policy_values is not None:
                policy_values["max_lanes"] = self.max_lanes
                policy_values.setdefault("adaptive_enabled", False)
                policy = ResourcePolicy.from_mapping(policy_values)
            elif isinstance(resource_policy, ResourcePolicy):
                policy = replace(resource_policy, max_lanes=self.max_lanes)
            else:
                policy = ResourcePolicy(
                    max_lanes=self.max_lanes,
                    adaptive_enabled=False,
                )
            self.resource_scheduler = ResourceScheduler(policy)
        self._host_resource_source = host_resource_source or sample_host_resources
        self._provider_capacity_source = provider_capacity_source
        self.provider_capacity_path = Path(provider_capacity_path).resolve() if provider_capacity_path else None
        if (
            isinstance(provider_capacity_max_age_ms, bool)
            or not isinstance(provider_capacity_max_age_ms, int)
            or provider_capacity_max_age_ms <= 0
        ):
            raise ValueError(
                "provider_capacity_max_age_ms must be a positive integer"
            )
        self.provider_capacity_max_age_ms = provider_capacity_max_age_ms
        self.external_task_state_paths = tuple(
            Path(path).resolve() for path in external_task_state_paths
        )
        self.bundle_index_refresh_command = str(
            bundle_index_refresh_command or ""
        ).strip()
        self.bundle_index_refresh_timeout_seconds = float(
            bundle_index_refresh_timeout_seconds
        )
        if (
            not math.isfinite(self.bundle_index_refresh_timeout_seconds)
            or self.bundle_index_refresh_timeout_seconds <= 0
        ):
            raise ValueError(
                "bundle_index_refresh_timeout_seconds must be positive"
            )
        self._bundle_index_refresher = bundle_index_refresher
        self._running: dict[str, RunningBundleLane] = {}
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._cycle = 0
        self._last_discovery_error = ""
        self._last_coordination_compaction: dict[str, Any] = {}
        self._last_scheduler_snapshot: SchedulerSnapshot | None = None
        self._last_resource_snapshot: ResourceScheduleSnapshot | None = None
        self._event_source_cache: dict[Path, tuple[int, int, list[dict[str, Any]]]] = {}
        self._last_bundle_index_refresh_source_revision: (
            tuple[tuple[str, int, int], ...] | None
        ) = None
        self._plan_cache: tuple[
            tuple[Path, ...],
            tuple[tuple[str, int, int], ...],
            tuple[tuple[str, int, int], ...],
            tuple[BundleLaneSpec, ...],
        ] | None = None

    @property
    def running_count(self) -> int:
        return len(self._running)

    def _sample_host_resources(
        self,
        *,
        active_workers: int | None = None,
    ) -> HostResourceSnapshot | dict[str, Any]:
        source = self._host_resource_source
        accounted_active_workers = (
            len(self._running)
            if active_workers is None
            else max(0, int(active_workers))
        )
        try:
            return source(
                self.state_root,
                active_workers=accounted_active_workers,
                worker_limit=self.max_lanes,
                active_phase="scheduler",
            )
        except TypeError:
            return source()

    def _provider_capacities(self, coordinator: LeaseCoordinator) -> Any:
        """Read injected/file/fenced-heartbeat provider telemetry in that order."""

        raw_capacities: Any = ()
        try:
            if self._provider_capacity_source is not None:
                try:
                    raw_capacities = self._provider_capacity_source()
                except TypeError:
                    raw_capacities = self._provider_capacity_source(self)
            else:
                configured_path = self.provider_capacity_path
                if (
                    configured_path is None
                    and not self._private_dual_review_capacity_required
                ):
                    env_path = os.environ.get(
                        "IPFS_ACCELERATE_LLM_ROUTER_CAPACITY_PATH", ""
                    ).strip()
                    if env_path:
                        configured_path = Path(env_path)
                if configured_path is not None:
                    if self._private_dual_review_capacity_required:
                        raw_capacities = load_provider_capacity_snapshot(
                            configured_path,
                            max_age_ms=self.provider_capacity_max_age_ms,
                        )
                    else:
                        # Retain the pre-production generic llm_router file
                        # contract. Only dual-review policy elevates this path
                        # to the private, fresh snapshot authority above.
                        payload = json.loads(
                            configured_path.read_text(encoding="utf-8")
                        )
                        raw_capacities = (
                            payload.get("providers", payload)
                            if isinstance(payload, dict)
                            else payload
                        )
                elif self._private_dual_review_capacity_required:
                    # Production/legacy independent review accepts only an
                    # explicitly named private snapshot (or an injected API
                    # source). Ambient unsigned JSON and unrelated worker
                    # heartbeats cannot rescue a missing authority file.
                    raw_capacities = ()
                else:
                    env_json = os.environ.get(
                        "IPFS_ACCELERATE_LLM_ROUTER_CAPACITY_JSON", ""
                    ).strip()
                    if env_json:
                        payload = json.loads(env_json)
                        raw_capacities = (
                            payload.get("providers", payload)
                            if isinstance(payload, dict)
                            else payload
                        )
                    else:
                        advertised: list[dict[str, Any]] = []
                        for heartbeat in coordinator.latest_heartbeats():
                            capacity = heartbeat.get("provider_capacity")
                            if not isinstance(capacity, dict):
                                continue
                            item = dict(capacity)
                            item.setdefault(
                                "provider_id", heartbeat.get("provider_id")
                            )
                            item.setdefault(
                                "observed_at_ms",
                                heartbeat.get("created_at_ms"),
                            )
                            advertised.append(item)
                        raw_capacities = advertised
        except Exception as exc:
            # Provider-dependent reviewed lanes still receive an explicit
            # unhealthy pair below. This makes every telemetry failure close
            # admission before a coordination claim is attempted.
            logger.warning("Could not read provider capacity: %s", exc)
            raw_capacities = ()
        if not self._private_dual_review_capacity_required:
            return raw_capacities
        return synthesize_dual_review_provider_capacity(
            raw_capacities,
            max_age_ms=self.provider_capacity_max_age_ms,
        )

    @staticmethod
    def _lane_resource_requirement(lane: BundleLaneSpec) -> LaneResourceRequirements:
        payload = lane.to_dict()
        payload.update(
            {
                "lane_id": lane.task_cid or lane.bundle_key,
                "stage": lane.resource_stage,
                "provider_id": lane.llm_provider,
                "context_tokens": lane.required_context_tokens,
                "max_provider_latency_ms": lane.max_provider_latency_ms,
                "memory_bytes": lane.memory_bytes,
                "gpu_memory_bytes": lane.gpu_memory_bytes,
                "disk_bytes": lane.disk_bytes,
                "process_slots": lane.process_slots,
                "critical_path_length": lane.critical_path_length,
                "downstream_unlock_value": lane.downstream_unlock_value,
                "queue_age_ms": max(0, lane.age_seconds) * 1_000,
                "merge_age_ms": (
                    max(0, lane.age_seconds) * 1_000
                    if normalize_adaptive_stage(lane.resource_stage) == "merge"
                    else 0
                ),
                "priority": lane.objective_priority,
            }
        )
        return LaneResourceRequirements.from_mapping(payload)

    @staticmethod
    def _resource_stage_for_phase(phase: Any) -> str:
        """Translate durable daemon phases into adaptive resource pools.

        The implementation daemon predates the resource scheduler and uses
        operational gerunds and queue names in its heartbeat.  Keep that
        compatibility translation at this boundary so the scheduler only
        receives canonical pool identities.  Empty, idle, and unknown phases
        intentionally return an empty string: they must not silently move a
        live lease out of its last observed working stage.
        """

        raw = str(phase or "").strip().lower()
        raw = raw.replace(" ", "_").replace("-", "_")
        aliases = {
            "implementing": "inference",
            "implementation": "inference",
            "validating": "validation",
            "merge_queue": "merge",
            "merge_reconciliation": "merge",
            "reconciliation": "merge",
            "merge_resolver": "inference",
            "resolving": "inference",
            "cleanup": "persistence",
            "objective_refill": "analysis",
            "codebase_refill": "analysis",
        }
        normalized = normalize_adaptive_stage(aliases.get(raw, raw))
        return normalized if normalized in set(ADAPTIVE_STAGES) - {"execution"} else ""

    def _transition_running_resource_stages(
        self,
        *,
        host: HostResourceSnapshot | Mapping[str, Any],
        providers: Any,
    ) -> None:
        """Atomically follow durable child-stage heartbeat transitions.

        A running bundle owns one process slot throughout its lifetime, but
        provider/GPU/disk and per-stage capacity must follow the work it is
        actually performing. Failed transitions retain the prior reservation
        and are retried on the next reconciliation cycle.
        """

        canonical_stages = set(ADAPTIVE_STAGES)
        for task_cid, running in list(self._running.items()):
            lease = running.resource_lease
            if lease is None:
                continue
            event = self._lane_phase_event(running.spec)
            observed = self._resource_stage_for_phase(event.get("phase"))
            if observed not in canonical_stages:
                continue
            if observed == lease.requirement.stage:
                continue
            next_spec = replace(running.spec, resource_stage=observed)
            next_requirement = self._lane_resource_requirement(next_spec)
            decision, next_lease = self.resource_scheduler.transition(
                lease,
                next_requirement,
                host=host,
                providers=providers,
                path=self.state_root,
            )
            if not decision.admitted or next_lease is None:
                continue
            self.resource_scheduler.record_stage_completion(
                lease.requirement.stage,
                duration_ms=self._running_stage_duration_ms(running),
                accepted=False,
            )
            running.spec = next_spec
            running.resource_lease = next_lease
            running.resource_stage_started_at = utc_now()

    def _plan_source_revision(self, paths: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
        """Return the cheap source revision that makes a lane plan reusable."""

        revisions: list[tuple[str, int, int]] = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                revisions.append((str(path), -1, -1))
            else:
                revisions.append((str(path), stat.st_mtime_ns, stat.st_size))
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        revisions.append((f"git:{head.stdout.strip() if head.returncode == 0 else ''}", 0, 0))
        return tuple(revisions)

    def _external_active_task_ids(self) -> set[str]:
        active_task_ids: set[str] = set()
        for path in self.external_task_state_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            active_task_id = str(payload.get("active_task_id") or "").strip()
            if active_task_id and (
                payload.get("implementation_in_progress")
                or str(payload.get("active_phase") or "").strip()
            ):
                active_task_ids.add(active_task_id)
        return active_task_ids

    @staticmethod
    def _fence_external_active_members(
        lane: BundleLaneSpec,
        active_task_ids: set[str],
    ) -> BundleLaneSpec:
        payload = dict(lane.queue_payload)
        active_tasks = [
            task
            for task in _mapping_list(payload.get("tasks"))
            if str(task.get("task_id") or "") in active_task_ids
        ]
        if not active_tasks:
            return lane
        active_ids = {
            str(task.get("task_id") or "")
            for task in active_tasks
            if str(task.get("task_id") or "")
        }
        active_cids = {
            str(task.get("canonical_task_cid") or task.get("task_cid") or "")
            for task in active_tasks
            if str(task.get("canonical_task_cid") or task.get("task_cid") or "")
        }
        payload.update(
            {
                "claimable": False,
                "active_member_task_ids": sorted(
                    set(_string_list(payload.get("active_member_task_ids"))) | active_ids
                ),
                "active_member_task_cids": sorted(
                    set(_string_list(payload.get("active_member_task_cids"))) | active_cids
                ),
                "execution_slice_task_ids": [],
                "execution_slice_task_cids": [],
                "external_active_member_fence": True,
            }
        )
        return replace(
            lane,
            task_ids=[],
            claimable=False,
            queue_payload=payload,
        )

    def _plan(self) -> list[BundleLaneSpec]:
        base_lanes: list[BundleLaneSpec]
        receipt_revision = bundle_member_completion_source_revision(self.state_root)
        if self._plan_cache is not None:
            (
                cached_paths,
                cached_revision,
                cached_receipt_revision,
                cached_lanes,
            ) = self._plan_cache
            if (
                self._plan_source_revision(cached_paths) == cached_revision
                and receipt_revision == cached_receipt_revision
            ):
                base_lanes = list(cached_lanes)
            else:
                self._plan_cache = None
        if self._plan_cache is None:
            allowed = {
                "task_prefix", "excluded_bundle_keys", "excluded_task_ids",
                "implement", "daemon_interval", "stale_seconds",
                "check_interval", "max_restarts", "max_task_attempts",
                "implementation_timeout", "implementation_max_timeout",
                "implementation_log_stall_seconds",
                "implementation_command", "production_provider_policy",
                "production_provider_context_budget_tokens",
                "production_provider_timeout_seconds",
                "production_provider_review_authority_key_path",
                "production_provider_launch_authority_receipt_path",
                "production_provider_launch_authority_receipt_content_id",
                "legacy_landed_review_policy_path",
                "legacy_landed_review_key_path",
                "llm_merge_resolver_command",
                "llm_merge_resolver_timeout_seconds", "merge_target_branch",
                "merge_reconciliation_max_merges",
                "generated_dirty_repair_enabled", "generated_dirty_repair_commit_subject",
                "generated_dirty_repair_include_submodule_gitlinks",
                "generated_dirty_repair_max_paths", "generated_dirty_repair_stale_lock_seconds",
                "generated_dirty_repair_paths",
                "worktree_submodule_paths", "log_level",
                "optimize_bundles", "bundle_optimization_policy",
            }
            options = {key: value for key, value in self.lane_options.items() if key in allowed}
            # Bind the planned slice and its receipt evidence to the same
            # stable log revision.  If a completion arrives during the read,
            # retry instead of caching stale work under the newer revision.
            for _attempt in range(3):
                receipt_revision = bundle_member_completion_source_revision(
                    self.state_root
                )
                completion_receipts = bundle_member_completion_receipts(
                    self.state_root
                )
                base_lanes = plan_bundle_lanes(
                    bundle_index_path=self.bundle_index_path,
                    repo_root=self.repo_root,
                    state_root=self.state_root,
                    worktree_root=self.worktree_root,
                    log_dir=self.log_dir,
                    max_lanes=None,
                    completion_receipts=completion_receipts,
                    **options,
                )
                observed_revision = bundle_member_completion_source_revision(
                    self.state_root
                )
                if receipt_revision == observed_revision:
                    receipt_revision = observed_revision
                    break
            else:
                # Avoid pinning an unstable snapshot.  It will be re-read on
                # the next scheduler cycle even if the log is continuously
                # active.
                receipt_revision = (("<unstable>", -1, -1),)
            source_paths = tuple(
                sorted(
                    {
                        self.bundle_index_path,
                        self.bundle_index_path.with_suffix(".duckdb"),
                        *(lane.todo_path for lane in base_lanes),
                    },
                    key=str,
                )
            )
            self._plan_cache = (
                source_paths,
                self._plan_source_revision(source_paths),
                receipt_revision,
                tuple(base_lanes),
            )

        external_active_task_ids = self._external_active_task_ids()
        if not external_active_task_ids:
            return base_lanes
        return [
            self._fence_external_active_members(lane, external_active_task_ids)
            for lane in base_lanes
        ]

    def _refresh_bundle_index_if_needed(self) -> bool:
        """Refresh a derived index before applying changed merge evidence."""

        if (
            self._bundle_index_refresher is None
            and not self.bundle_index_refresh_command
        ):
            return False
        receipt_revision = list(
            bundle_member_completion_source_revision(self.state_root)
        )
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        source_revision = tuple(
            [
                *receipt_revision,
                (
                    f"git:{head.stdout.strip() if head.returncode == 0 else ''}",
                    0,
                    0,
                ),
            ]
        )
        if (
            self._last_bundle_index_refresh_source_revision is not None
            and source_revision
            == self._last_bundle_index_refresh_source_revision
        ):
            return False
        try:
            if self._bundle_index_refresher is not None:
                self._bundle_index_refresher()
            else:
                command = shlex.split(self.bundle_index_refresh_command)
                if not command:
                    raise ValueError("bundle-index refresh command is empty")
                completed = subprocess.run(
                    command,
                    cwd=self.repo_root,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=self.bundle_index_refresh_timeout_seconds,
                    check=False,
                )
                if completed.returncode != 0:
                    error = str(completed.stderr or "").strip()[-2_000:]
                    raise ValueError(
                        "bundle-index refresh command failed with exit code "
                        f"{completed.returncode}"
                        + (f": {error}" if error else "")
                    )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                "bundle-index refresh command timed out after "
                f"{self.bundle_index_refresh_timeout_seconds:.3f}s"
            ) from exc
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(
                f"bundle-index refresh failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not self.bundle_index_path.is_file():
            raise ValueError(
                "bundle-index refresh did not produce the configured index"
            )
        self._last_bundle_index_refresh_source_revision = source_revision
        self._plan_cache = None
        return True

    @staticmethod
    def _default_process_alive(handle: Any) -> bool:
        poll = getattr(handle, "poll", None)
        if callable(poll):
            return poll() is None
        return bool(getattr(handle, "alive", False))

    def _default_lane_disposition(self, lane: BundleLaneSpec) -> str:
        """Project a settled execution slice or shard board to a disposition."""

        operational_todo_path = (
            lane.runtime_todo_path
            if lane.runtime_todo_path is not None and lane.runtime_todo_path.exists()
            else lane.todo_path
        )
        try:
            markdown = operational_todo_path.read_text(encoding="utf-8")
        except OSError:
            markdown = ""
        from ..todo_daemon.implementation_daemon import (
            TASK_ATTEMPT_LIMIT_IDLE_REASON,
            parse_task_file,
        )

        task_prefix = str(self.lane_options.get("task_prefix") or DEFAULT_TASK_PREFIX)
        portal_tasks = (
            parse_task_file(operational_todo_path, task_prefix) if markdown else []
        )
        portal_task_ids = {str(task.task_id) for task in portal_tasks}

        state_path = lane.state_dir / f"{lane.state_prefix}_task_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if isinstance(state, dict):
            task_count = _schedule_int(state, "task_count")
            completed_count = _schedule_int(state, "completed_count")
            blocked_count = _schedule_int(state, "blocked_count")
            ready_count = _schedule_int(state, "ready_count")
            waiting_count = _schedule_int(state, "waiting_count")
            active = bool(state.get("implementation_in_progress") or state.get("active_task_id"))
            state_task_ids = {
                str(task_id)
                for task_id in (state.get("task_identities") or {})
            }
            try:
                state_mtime_ns = state_path.stat().st_mtime_ns
                board_mtime_ns = operational_todo_path.stat().st_mtime_ns
            except OSError:
                state_covers_current_board = True
            else:
                state_covers_current_board = state_mtime_ns > board_mtime_ns
                if state_mtime_ns == board_mtime_ns:
                    projected_statuses = state.get("task_statuses")
                    if isinstance(projected_statuses, dict):
                        terminal_aliases = {
                            "complete": "completed",
                            "completed": "completed",
                            "blocked": "blocked",
                            "on_hold": "blocked",
                        }
                        state_covers_current_board = all(
                            terminal_aliases.get(str(task.status).strip().lower())
                            not in {"completed", "blocked"}
                            or terminal_aliases.get(
                                str(projected_statuses.get(task.task_id) or "").strip().lower()
                            )
                            == terminal_aliases.get(str(task.status).strip().lower())
                            for task in portal_tasks
                        )
                    else:
                        state_covers_current_board = True
            state_matches_board = not portal_tasks or (
                state_task_ids == portal_task_ids
                if state_task_ids
                else task_count == len(portal_tasks)
            )
            state_matches_board = state_matches_board and state_covers_current_board
            raw_statuses = state.get("task_statuses")
            statuses = {
                str(task_id): str(status).strip().lower()
                for task_id, status in (
                    raw_statuses.items() if isinstance(raw_statuses, dict) else ()
                )
            }
            board_statuses = {
                str(task.task_id): str(task.status).strip().lower()
                for task in portal_tasks
            }
            execution_statuses = [
                statuses.get(task_id, board_statuses.get(task_id, ""))
                for task_id in lane.task_ids
            ]
            # Authoritative shard board open work always wins over a newer
            # runtime/portal projection that still says completed. Lease
            # requeue leaves runtime todos + task_state completed while the
            # bundle shard remains todo; treating the runtime snapshot as
            # terminal permanently starves relaunch even when capacity is free.
            shard_open = False
            try:
                if lane.todo_path.is_file():
                    from ..todo_daemon.implementation_daemon import parse_task_file as _parse_shard

                    shard_prefix = str(
                        self.lane_options.get("task_prefix") or DEFAULT_TASK_PREFIX
                    )
                    shard_tasks = _parse_shard(lane.todo_path, shard_prefix)
                    selected = {str(task_id) for task_id in lane.task_ids}
                    shard_open = any(
                        str(task.task_id) in selected
                        and str(task.status).strip().lower()
                        not in {"complete", "completed", "blocked", "on_hold", "done"}
                        for task in shard_tasks
                    )
            except OSError:
                shard_open = False
            board_has_open_work = shard_open or any(
                board_statuses.get(str(task_id), "todo")
                not in {"complete", "completed", "blocked", "on_hold", "done"}
                for task_id in lane.task_ids
            )
            if board_has_open_work and not active:
                return ""
            if (
                state_matches_board
                and not active
                and str(state.get("selection_idle_reason") or "")
                == TASK_ATTEMPT_LIMIT_IDLE_REASON
            ):
                # An ordinary empty queue remains persistent. This exact reason
                # means every selectable member exhausted its bounded retry
                # budget, so after the wrapper settles and exits the scheduler
                # must not launch the unchanged execution slice again.
                return "blocked"
            if state_matches_board and not active and execution_statuses:
                if all(status in {"complete", "completed"} for status in execution_statuses):
                    return "completed"
                if all(
                    status in {"complete", "completed", "blocked", "on_hold"}
                    for status in execution_statuses
                ):
                    return "blocked"
            if state_matches_board and task_count > 0 and not active and waiting_count == 0:
                if completed_count >= task_count:
                    return "completed"
                if completed_count + blocked_count >= task_count:
                    return "blocked"
            if state_matches_board and task_count > 0 and not active and ready_count == 0:
                completed_ids = {
                    task.task_id
                    for task in portal_tasks
                    if statuses.get(task.task_id, task.status) in {"complete", "completed"}
                }
                blocked_ids = {
                    task.task_id
                    for task in portal_tasks
                    if statuses.get(task.task_id, task.status) == "blocked"
                }
                # Waiting descendants of an internally blocked task can never
                # become ready in this lane. Propagate that closure so the
                # bundle releases its lease instead of retaining an idle worker.
                changed = True
                while changed:
                    changed = False
                    for task in portal_tasks:
                        if task.task_id in completed_ids or task.task_id in blocked_ids:
                            continue
                        internal_dependencies = portal_task_ids.intersection(task.depends_on)
                        if internal_dependencies.intersection(blocked_ids):
                            blocked_ids.add(task.task_id)
                            changed = True
                if portal_task_ids and portal_task_ids.issubset(completed_ids | blocked_ids):
                    return "blocked"

        if not markdown:
            return ""
        if portal_tasks:
            if all(task.status == "completed" for task in portal_tasks):
                return "completed"
            if all(task.status in {"completed", "blocked"} for task in portal_tasks):
                return "blocked"

        from ..todo_daemon.engine import parse_markdown_tasks

        tasks = parse_markdown_tasks(markdown)
        if not tasks:
            return ""
        if all(task.status == "complete" for task in tasks):
            return "completed"
        if all(task.status in {"complete", "blocked"} for task in tasks):
            return "blocked"
        return ""

    def _authoritative_lane_has_open_work(self, lane: BundleLaneSpec) -> bool:
        """Return whether the shard explicitly reopens this execution slice."""

        if not lane.todo_path.exists() or not lane.task_ids:
            return False
        from ..todo_daemon.implementation_daemon import parse_task_file

        task_prefix = str(self.lane_options.get("task_prefix") or DEFAULT_TASK_PREFIX)
        try:
            tasks = parse_task_file(lane.todo_path, task_prefix)
        except OSError:
            return False
        selected_ids = set(lane.task_ids)
        selected = [
            task
            for task in tasks
            if str(task.task_id) in selected_ids
        ]
        return bool(selected) and any(
            str(task.status).strip().lower() not in {"complete", "completed", "blocked", "on_hold"}
            for task in selected
        )

    @staticmethod
    def _reopen_portal_task_state_for_board(lane: BundleLaneSpec) -> bool:
        """Reset stale portal completion so a reopened board can relaunch.

        Also clears repair-budget deferrals and burned attempt counters for
        residual open members. Infrastructure failures (broken Grok runner
        import, etc.) can exhaust ``implementation_max_repair_rounds`` without
        landing a fix; without this reset the leased lane idles forever while
        holding bundle capacity.

        Returns True when the portal projection was rewritten.
        """

        if not lane.task_ids:
            return False
        state_path = lane.state_dir / f"{lane.state_prefix}_task_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(state, dict):
            return False
        statuses = state.get("task_statuses")
        if not isinstance(statuses, dict):
            statuses = {}
        changed = False
        completed_ids = {
            str(item)
            for item in (state.get("completed_task_ids") or [])
            if str(item).strip()
        }
        open_ids: list[str] = []
        for task_id in lane.task_ids:
            key = str(task_id)
            current = str(statuses.get(key) or "").strip().lower()
            if current in {"complete", "completed"}:
                statuses[key] = "ready"
                completed_ids.discard(key)
                open_ids.append(key)
                changed = True
            elif current in {"ready", "todo", "pending", "open", ""}:
                open_ids.append(key)
                if key not in statuses or current in {"", "todo", "pending", "open"}:
                    statuses[key] = "ready"
                    changed = True
        idle_reason = str(state.get("selection_idle_reason") or "")
        deferred_repair = idle_reason.startswith("implementation_retry_deferred:")
        if deferred_repair and open_ids:
            state["selection_idle_reason"] = ""
            changed = True
        attempts = state.get("implementation_attempts")
        if not isinstance(attempts, dict):
            attempts = {}
        attempts_by_cid = state.get("implementation_attempts_by_cid")
        if not isinstance(attempts_by_cid, dict):
            attempts_by_cid = {}
        if open_ids and (
            deferred_repair
            or any(int(attempts.get(tid, 0) or 0) > 0 for tid in open_ids)
        ):
            for tid in open_ids:
                if tid in attempts:
                    attempts.pop(tid, None)
                    changed = True
            if deferred_repair:
                attempts_by_cid = {}
                changed = True
            state["implementation_attempts"] = attempts
            state["implementation_attempts_by_cid"] = attempts_by_cid
        if not changed:
            return False
        state["task_statuses"] = statuses
        state["completed_task_ids"] = sorted(completed_ids)
        state["completed_count"] = len(completed_ids)
        ready_ids = [
            str(task_id)
            for task_id, status in statuses.items()
            if str(status).strip().lower() in {"ready", "todo", "pending", "open"}
        ]
        if open_ids:
            ready_ids = list(dict.fromkeys([*open_ids, *ready_ids]))
        state["ready_count"] = len(ready_ids)
        state["eligible_ready_count"] = len(ready_ids)
        state["eligible_ready_task_ids"] = ready_ids
        state["selectable_ready_count"] = len(ready_ids)
        state["selectable_ready_task_ids"] = list(ready_ids)
        state["selection_idle_reason"] = ""
        state["implementation_in_progress"] = False
        state["active_task_id"] = ""
        state["active_task_cid"] = ""
        try:
            lane.state_dir.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            return False
        # Failure cooldowns (up to hours) would still starve residual
        # redispatch after an infrastructure fix. Clear queue backoff for
        # residual open members so auto-finish does not wait out exponential
        # selection cooldowns burned by prior NameError/probe failures.
        if open_ids:
            queue_path = lane.state_dir / "task_queue.json"
            try:
                queue = json.loads(queue_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                queue = None
            if isinstance(queue, dict) and isinstance(queue.get("entries"), dict):
                open_set = set(open_ids)
                queue_changed = False
                for entry in queue["entries"].values():
                    if not isinstance(entry, dict):
                        continue
                    tid = str(entry.get("task_id") or "")
                    if tid not in open_set:
                        continue
                    if (
                        float(entry.get("cooldown_until") or 0) > 0
                        or int(entry.get("consecutive_failures") or 0) > 0
                        or int(entry.get("selection_penalty") or 0) > 0
                    ):
                        entry["cooldown_until"] = 0
                        entry["consecutive_failures"] = 0
                        entry["selection_penalty"] = 0
                        entry["notes"] = "bundle_board_reopened_cooldown_cleared"
                        queue_changed = True
                if queue_changed:
                    try:
                        queue_path.write_text(
                            json.dumps(queue, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    except OSError:
                        pass
        return True

    def _reopen_runtime_todo_statuses(self, lane: BundleLaneSpec) -> bool:
        """Rewrite runtime/shard todo Status:completed → todo for open residual work."""

        if not lane.task_ids:
            return False
        paths: list[Path] = []
        if lane.todo_path:
            paths.append(Path(lane.todo_path))
        runtime_todo = lane.state_dir / f"{lane.state_prefix}_runtime.todo.md"
        paths.append(runtime_todo)
        # Also reopen operational portal copy if present.
        for candidate in lane.state_dir.glob("*runtime*.todo.md"):
            paths.append(candidate)
        selected = {str(task_id) for task_id in lane.task_ids}
        changed_any = False
        for path in paths:
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            original = text
            for task_id in selected:
                # Only flip Status under the matching task header.
                pattern = (
                    rf"(^## {re.escape(task_id)}\b.*?\n(?:.*\n)*?^- Status:\s*)"
                    rf"(complete|completed)\s*$"
                )
                text, count = re.subn(
                    pattern,
                    r"\1todo",
                    text,
                    count=1,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                if count:
                    changed_any = True
            if text != original:
                try:
                    path.write_text(text, encoding="utf-8")
                except OSError:
                    continue
        return changed_any

    @staticmethod
    def _receipt_backed_attempt_limit_disposition(
        lane: BundleLaneSpec,
        projection: Mapping[str, Any],
    ) -> str:
        """Return ``blocked`` for an idle slice already fenced by its wrapper.

        The implementation daemon and the bundle coordinator have independent
        attempt budgets.  Once a scoped wrapper publishes a durable blocked
        receipt because the daemon exhausted its task attempts, later
        coordination attempts may consume their remaining budget without
        spawning the same exhausted daemon again.  Requiring both the receipt
        and exact idle state keeps an ordinary transiently idle worker alive.
        """

        release_reason = str(projection.get("release_reason") or "")
        if not (
            release_reason.startswith("receipt:")
            and release_reason.endswith(":blocked")
        ):
            return ""
        if not lane.task_ids:
            return ""
        state_path = lane.state_dir / f"{lane.state_prefix}_task_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        if not isinstance(state, Mapping):
            return ""
        if (
            str(state.get("selection_idle_reason") or "")
            != _TASK_ATTEMPT_LIMIT_IDLE_REASON
            or state.get("implementation_in_progress") is not False
            or str(state.get("active_task_id") or "").strip()
            or state.get("selectable_ready_count") != 0
        ):
            return ""
        statuses = state.get("task_statuses")
        if not isinstance(statuses, Mapping):
            return ""
        expected_ids = {
            str(task_id).strip()
            for task_id in lane.task_ids
            if str(task_id).strip()
        }
        terminal_or_limited = {"ready", "complete", "completed", "blocked", "on_hold"}
        if not expected_ids or any(
            str(statuses.get(task_id) or "").strip().lower()
            not in terminal_or_limited
            for task_id in expected_ids
        ):
            return ""
        return "blocked"

    def _disposition(self, lane: BundleLaneSpec) -> str:
        value = self._lane_disposition(lane)
        if value is True:
            return "completed"
        if value is False or value is None:
            return ""
        normalized = str(value).strip().lower()
        return normalized if normalized in {"completed", "blocked"} else ""

    @staticmethod
    def _terminate_handle(handle: Any, *, grace_seconds: float = 5.0) -> None:
        """Stop a live child and always collect its exit status when possible."""

        poll = getattr(handle, "poll", None)
        try:
            if callable(poll):
                alive = poll() is None
            elif hasattr(handle, "alive"):
                alive = bool(handle.alive)
            else:
                alive = getattr(handle, "returncode", None) is None
        except OSError:
            alive = False

        terminate = getattr(handle, "terminate", None)
        if alive and callable(terminate):
            try:
                terminate()
            except OSError:
                pass

        wait = getattr(handle, "wait", None)
        if not callable(wait):
            return
        timeout = max(0.0, float(grace_seconds))
        try:
            wait(timeout=timeout)
            return
        except OSError:
            return
        except subprocess.TimeoutExpired:
            pass

        kill = getattr(handle, "kill", None)
        if callable(kill):
            try:
                kill()
            except OSError:
                pass
        try:
            wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            pass

    @staticmethod
    def _settle_grant(
        coordinator: LeaseCoordinator,
        grant: Any,
        *,
        disposition: str,
    ) -> None:
        if disposition == "completed":
            coordinator.receipt(
                grant,
                status="succeeded",
                output={"reason": "bundle board drained"},
            )
        else:
            coordinator.receipt(
                grant,
                status="failed",
                failure_class="blocked",
            )

    @staticmethod
    def _running_stage_duration_ms(running: RunningBundleLane) -> int:
        """Return a non-negative duration for the lane's current resource stage."""

        try:
            started = datetime.fromisoformat(
                (
                    running.resource_stage_started_at
                    or running.started_at
                ).strip().replace("Z", "+00:00")
            )
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
        except (AttributeError, TypeError, ValueError):
            return 0
        return max(
            0,
            int((datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds() * 1_000),
        )

    def _reconcile_untracked_terminal_leases(
        self,
        coordinator: LeaseCoordinator,
        lanes: Sequence[BundleLaneSpec],
    ) -> dict[str, dict[str, str]]:
        """Defer untracked accepted leases to their fenced wrapper or expiry.

        ``_running`` is process-local, so after restart this scheduler cannot
        prove that a predecessor wrapper has exited or fenced its descendants.
        Neither mutable terminal board state nor a durable member receipt is
        sufficient authority to publish a Profile-G terminal receipt.  The
        predecessor ``leased_lane`` remains the only successful-settlement
        authority; blocked work likewise remains accepted until that wrapper
        settles, its lease expires, or an operator performs fenced recovery.
        """

        del coordinator, lanes
        return {}

    def _default_launcher(self, lane: BundleLaneSpec, grant: Any) -> subprocess.Popen[bytes]:
        process, _command, _pid_path = _spawn_accepted_lane(
            lane,
            grant,
            repo_root=self.repo_root,
            coordination_path=self.coordination_path,
            lease_ms=self.lease_ms,
            heartbeat_interval=self.heartbeat_interval,
            capacity_millionths=self.capacity_millionths,
        )
        return process

    @staticmethod
    def _reap_exited_handle(handle: Any) -> bool:
        """Collect an already-exited wrapper without sending it a signal."""

        poll = getattr(handle, "poll", None)
        try:
            if callable(poll) and poll() is None:
                return False
            if not callable(poll) and hasattr(handle, "alive") and bool(handle.alive):
                return False
        except OSError:
            return False
        wait = getattr(handle, "wait", None)
        if not callable(wait):
            return True
        try:
            wait(timeout=0)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return True

    def _reap(self, coordinator: LeaseCoordinator) -> list[str]:
        """Reap wrappers only after their fenced execution boundary exits.

        The leased wrapper is the settlement authority for live work: it binds
        exact durable evidence, fences descendants, publishes its receipt, and
        exits.  Mutable task-board projections must not let the scheduler
        manufacture that receipt or release capacity prematurely.
        """

        reaped: list[str] = []
        for task_cid, running in list(self._running.items()):
            try:
                alive = bool(self._process_alive(running.handle))
            except (OSError, RuntimeError):
                alive = False
            if alive:
                continue

            # ``process_alive`` has independently observed wrapper exit. Reap
            # only its already-terminal status; never terminate or force-kill
            # here because that could bypass the wrapper's descendant fence.
            if not self._reap_exited_handle(running.handle):
                continue
            receipts = coordinator.list_receipts(task_cid)
            terminal_status = (
                str(receipts[-1].get("receipt", {}).get("status") or "")
                if receipts
                else ""
            )
            # The leased-lane wrapper normally publishes a receipt first. A
            # crashed wrapper does not, so explicitly release its still-current
            # grant only after wrapper exit makes reuse safe.
            try:
                if coordinator.active_lease(task_cid) is not None:
                    coordinator.release(running.grant, reason="worker drained or exited")
            except LeaseError:
                pass
            if running.resource_lease is not None:
                self.resource_scheduler.release(running.resource_lease)
                self.resource_scheduler.record_stage_completion(
                    running.resource_lease.requirement.stage,
                    duration_ms=self._running_stage_duration_ms(running),
                    accepted=terminal_status == "succeeded",
                    cancelled=terminal_status in {"", "cancelled"},
                )
            del self._running[task_cid]
            reaped.append(task_cid)
        return reaped

    @staticmethod
    def _projection_state(item: dict[str, Any]) -> str:
        state = str(item.get("state") or item.get("lease_state") or "ready")
        if state in {"released", "expired", "pending", "registered"}:
            return "ready"
        if state in {"complete", "completed", "succeeded"}:
            return "completed"
        return state

    @staticmethod
    def _lane_phase_event(lane: BundleLaneSpec) -> dict[str, Any]:
        """Read the durable lane heartbeat as one current-state event."""

        state_path = lane.state_dir / f"{lane.state_prefix}_task_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state_observed = isinstance(state, dict)
        except (OSError, json.JSONDecodeError):
            state = {}
            state_observed = False
        if not isinstance(state, dict):
            state = {}
        phase = str(state.get("active_phase") or "")
        if not phase:
            if state.get("implementation_in_progress") or state.get("active_task_id"):
                phase = "active"
            elif int(state.get("blocked_count") or 0) > 0 and int(state.get("ready_count") or 0) == 0:
                phase = "blocked"
            elif int(state.get("ready_count") or 0) > 0:
                phase = "ready"
            else:
                phase = "idle"
        return {
            "type": "lane_heartbeat",
            "timestamp": str(state.get("heartbeat_at") or utc_now()),
            "phase": phase,
            "task_id": str(state.get("active_task_id") or ""),
            "canonical_task_cid": str(state.get("active_task_cid") or lane.task_cid),
            "state_observed": state_observed,
        }

    def _read_scheduler_events(self, path: Path) -> list[dict[str, Any]]:
        """Read active/rotated lane logs once per observed file revision."""

        events: list[dict[str, Any]] = []
        for source in event_log_sources((path,), include_rotated=True):
            try:
                stat = source.stat()
                revision = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
            cached = self._event_source_cache.get(source)
            if cached is None or cached[:2] != revision:
                source_events = read_jsonl_events(source)
                self._event_source_cache[source] = (*revision, source_events)
            else:
                source_events = cached[2]
            events.extend(dict(event) for event in source_events)
        return events

    def _build_scheduler_snapshot(
        self,
        lanes: Sequence[BundleLaneSpec],
        task_projection: Sequence[dict[str, Any]],
    ) -> SchedulerSnapshot:
        """Join lane logs, heartbeats, and lease events through one reducer."""

        events: list[dict[str, Any]] = []
        by_task_cid = {lane.task_cid: lane for lane in lanes if lane.task_cid}
        for lane in lanes:
            defaults = {
                "goal_cid": lane.goal_cid,
                "subgoal_cid": lane.subgoal_cid,
                "task_cid": lane.task_cid,
                "lane_id": lane.parallel_lane or lane.bundle_key,
                "provider_id": self.claimant_did,
                "bundle_key": lane.bundle_key,
            }
            for suffix in ("_events.jsonl", "_supervisor_events.jsonl"):
                path = lane.state_dir / f"{lane.state_prefix}{suffix}"
                for raw in self._read_scheduler_events(path):
                    event = dict(raw)
                    for key, value in defaults.items():
                        if value:
                            event.setdefault(key, value)
                    events.append(event)
            heartbeat = self._lane_phase_event(lane)
            for key, value in defaults.items():
                if value:
                    heartbeat.setdefault(key, value)
            events.append(heartbeat)

        running_ids = set(self._running)
        projected: list[dict[str, Any]] = []
        for raw in task_projection:
            item = dict(raw)
            task_cid = str(item.get("task_cid") or "")
            lane = by_task_cid.get(task_cid)
            state = self._projection_state(item)
            if (
                state == "ready"
                and lane is not None
                and lane.queue_payload.get("external_active_member_fence")
            ):
                state = "blocked"
                item.setdefault("blocked_reason", "external_active_member")
            if state == "accepted" and task_cid not in running_ids:
                state = "blocked"
            elif state == "accepted":
                state = "active"
            item["state"] = state
            if lane is not None:
                item.setdefault("goal_cid", lane.goal_cid)
                item.setdefault("subgoal_cid", lane.subgoal_cid)
                item.setdefault("lane_id", lane.parallel_lane or lane.bundle_key)
                item.setdefault("provider_id", self.claimant_did)
            projected.append(item)
        # Lease state is emitted last at one timestamp so it is the authority
        # over historical lifecycle events.  A following lane heartbeat can
        # refine active into validation/merge/resolver.
        projection_timestamp = utc_now()
        events.extend(scheduler_state_events(projected, timestamp=projection_timestamp))
        for task_cid, running in sorted(self._running.items()):
            lane_event = self._lane_phase_event(running.spec)
            lane_event["timestamp"] = projection_timestamp
            if not lane_event.pop("state_observed", False):
                lane_event["phase"] = "active"
            lane_event.update(
                {
                    "goal_cid": running.spec.goal_cid,
                    "subgoal_cid": running.spec.subgoal_cid,
                    "task_cid": task_cid,
                    "lane_id": running.spec.parallel_lane or running.spec.bundle_key,
                    "provider_id": self.claimant_did,
                }
            )
            events.append(lane_event)
        if self._last_resource_snapshot is not None:
            events.append(
                {
                    "type": "resource_schedule_observed",
                    "timestamp": projection_timestamp,
                    "resource_schedule": self._last_resource_snapshot.to_dict(),
                }
            )
        return scheduler_snapshot(events, now=projection_timestamp)

    def _write_live_manifest(
        self,
        *,
        discovered: Sequence[BundleLaneSpec],
        task_projection: Sequence[dict[str, Any]],
        launched: Sequence[str],
        reaped: Sequence[str],
        reconciled: Sequence[str] = (),
        scheduler_state: SchedulerSnapshot | None = None,
        decision_snapshot: SchedulerSnapshot | None = None,
        decisions: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        running_ids = set(self._running)
        lanes_by_task_cid = {
            lane.task_cid: lane for lane in discovered if lane.task_cid
        }
        lanes_by_bundle_key = {lane.bundle_key: lane for lane in discovered}
        normalized: list[dict[str, Any]] = []
        for raw in task_projection:
            detailed = dict(raw)
            detailed["state"] = self._projection_state(detailed)
            lane = lanes_by_task_cid.get(str(detailed.get("task_cid") or ""))
            if lane is None:
                lane = lanes_by_bundle_key.get(str(detailed.get("bundle_key") or ""))
            if (
                detailed["state"] == "ready"
                and lane is not None
                and lane.queue_payload.get("external_active_member_fence")
            ):
                detailed["state"] = "blocked"
                detailed.setdefault("blocked_reason", "external_active_member")
            normalized.append(_compact_task_manifest_payload(detailed))
        ready = [item for item in normalized if item["state"] == "ready"]
        completed = [item for item in normalized if item["state"] == "completed"]
        blocked = [
            item
            for item in normalized
            if item["state"] == "blocked"
            or (
                item["state"] == "accepted"
                and str(item.get("task_cid") or "") not in running_ids
            )
        ]
        active_lanes = [
            running.to_dict(repo_root=self.repo_root)
            for _task_cid, running in sorted(self._running.items())
        ]
        detailed_active_lanes = [
            running.to_database_dict(repo_root=self.repo_root)
            for _task_cid, running in sorted(self._running.items())
        ]
        active_worker_pids = sorted(
            {
                int(worker["pid"])
                for running in self._running.values()
                for worker in active_codex_exec_workers(getattr(running.handle, "pid", None))
                if worker.get("pid") is not None
            }
        )
        current_snapshot = scheduler_state or self._build_scheduler_snapshot(discovered, normalized)
        decision_state = decision_snapshot or current_snapshot
        self._last_scheduler_snapshot = current_snapshot
        snapshot_payload = current_snapshot.to_dict()
        decision_snapshot_payload = decision_state.to_dict()
        resource_snapshot_payload = (
            self._last_resource_snapshot.to_dict()
            if self._last_resource_snapshot is not None
            else None
        )
        if resource_snapshot_payload is not None:
            # Per-lane admission evidence is already retained beside the
            # scheduler decision. Keep the authoritative live resource view
            # compact enough for long-running manifests.
            resource_snapshot_payload.pop("decisions", None)
            resource_snapshot_payload.pop("admitted_lane_ids", None)
            resource_snapshot_payload.pop("policy", None)
            resource_snapshot_payload.pop("observed_at_ms", None)
            resource_snapshot_payload.pop("configured_max_lanes", None)
            resource_snapshot_payload.pop("available_slots", None)
        payload: dict[str, Any] = {
            "schema": "ipfs_accelerate_py.agent_supervisor.dynamic_bundle_scheduler@1",
            "generated_at": utc_now(),
            "authoritative": True,
            "scheduler_state": "stopping" if self._stop_event.is_set() else "running",
            "cycle": self._cycle,
            "repo_root": str(self.repo_root),
            "bundle_index_path": repo_relative_path(self.repo_root, self.bundle_index_path),
            "coordination_path": repo_relative_path(self.repo_root, self.coordination_path),
            "coordination_compaction": dict(
                self._last_coordination_compaction
            ),
            "capacity": self.max_lanes,
            "effective_capacity": (
                self._last_resource_snapshot.effective_slots
                if self._last_resource_snapshot is not None
                else self.max_lanes
            ),
            "available_worker_capacity": (
                self._last_resource_snapshot.available_slots
                if self._last_resource_snapshot is not None
                else max(0, self.max_lanes - len(active_lanes))
            ),
            "planned_count": len(discovered),
            "started_count": len(active_lanes),
            "running_count": len(active_lanes),
            "active_worker_count": len(active_worker_pids),
            "active_worker_pids": active_worker_pids,
            "ready_count": len(ready),
            "blocked_count": len(blocked),
            "completed_count": len(completed),
            "counts": {
                "active": len(active_lanes),
                "ready": len(ready),
                "blocked": len(blocked),
                "completed": len(completed),
                "capacity": self.max_lanes,
                "effective_capacity": (
                    self._last_resource_snapshot.effective_slots
                    if self._last_resource_snapshot is not None
                    else self.max_lanes
                ),
            },
            "scheduler_snapshot": snapshot_payload,
            "scheduler_snapshot_id": current_snapshot.snapshot_id,
            "scheduler_metrics_path": repo_relative_path(self.repo_root, self.metrics_path),
            "scheduler_decision_snapshot_id": decision_state.snapshot_id,
            "scheduler_decision_snapshot": {
                "snapshot_id": decision_state.snapshot_id,
                "schema": decision_snapshot_payload.get("schema"),
                "generated_at": decision_snapshot_payload.get("generated_at"),
                "phase_counts": decision_snapshot_payload.get("phase_counts", {}),
                "source_event_count": decision_snapshot_payload.get("source_event_count", 0),
                "metrics_path": repo_relative_path(self.repo_root, self.decision_metrics_path),
            },
            "scheduler_decisions": [dict(item) for item in decisions],
            "resource_schedule": resource_snapshot_payload,
            "backpressure_reasons": (
                list(self._last_resource_snapshot.backpressure_reasons)
                if self._last_resource_snapshot is not None
                else []
            ),
            "conflict_graph": _lane_conflict_manifest(discovered),
            "lanes": active_lanes,
            "ready": ready,
            "blocked": blocked,
            "completed": completed,
            "tasks": normalized,
            "launched_task_cids": list(launched),
            "reaped_task_cids": list(reaped),
            "reconciled_task_cids": list(reconciled),
        }
        if self._last_discovery_error:
            payload["discovery_error"] = self._last_discovery_error
        database_payload = {
            **payload,
            "lanes": detailed_active_lanes,
            "tasks": normalized,
        }
        payload = write_scheduler_manifest_artifact(
            self.manifest_path,
            payload,
            database_payload=database_payload,
        )
        write_scheduler_snapshot(self.metrics_path, current_snapshot)
        write_scheduler_snapshot(self.decision_metrics_path, decision_state)
        return payload

    def reconcile_once(self) -> dict[str, Any]:
        """Perform one atomic discovery/reap/claim/fill/projection cycle."""

        with self._lock:
            self._cycle += 1
            try:
                self._refresh_bundle_index_if_needed()
                discovered = self._plan()
                self._last_discovery_error = ""
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                discovered = []
                self._last_discovery_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Bundle discovery failed; retaining live lanes: %s", exc)

            launched: list[str] = []
            with LeaseCoordinator(self.coordination_path) as coordinator:
                # Reap and recover before registration.  Registration refreshes
                # mutable discovery metadata; doing it first could overwrite
                # the accepted predecessor's execution-slice payload when the
                # newly planned slice happens to have the same canonical CID.
                reaped = self._reap(coordinator)
                reconciled = self._reconcile_untracked_terminal_leases(
                    coordinator,
                    discovered,
                )
                accepted_by_task_cid = {
                    str(item.get("task_cid") or ""): item
                    for item in coordinator.list_tasks()
                    if self._projection_state(item) == "accepted"
                    and str(item.get("task_cid") or "")
                }
                accounted_worker_task_cids = set(self._running)
                accounted_worker_task_cids.update(
                    task_cid
                    for task_cid, accepted in accepted_by_task_cid.items()
                    if str(accepted.get("claimant_did") or "")
                    == self.claimant_did
                )
                accounted_active_workers = len(accounted_worker_task_cids)
                # Receipt overlays intentionally leave fully drained bundle
                # records in discovery so they remain visible to planning, but
                # those records must never be registered, claimed, or launched.
                # Other non-launchable records (for example an external active
                # fence) remain registered so their blocked state is observable.
                registered: list[BundleLaneSpec] = []
                receipt_drained_completion_task_cids: set[str] = set()
                for lane in (
                    item
                    for item in discovered
                    if item.queue_payload
                    and not (
                        _receipt_drained_execution_slice(item)
                        and self._disposition(item) != "completed"
                    )
                ):
                    receipt_drained = _receipt_drained_execution_slice(lane)
                    accepted = accepted_by_task_cid.get(lane.task_cid)
                    if accepted is not None:
                        # Preserve the immutable payload of work that is still
                        # executing under its accepted fencing token.
                        registered.append(
                            replace(
                                lane,
                                goal_cid=str(accepted.get("goal_cid") or lane.goal_cid),
                                subgoal_cid=str(
                                    accepted.get("subgoal_cid") or lane.subgoal_cid
                                ),
                            )
                        )
                        continue
                    adapted = coordinator.register_bundle(lane.queue_payload)
                    registered_lane = replace(
                        lane,
                        task_cid=str(adapted["task_cid"]),
                        goal_cid=str(adapted["goal_cid"]),
                        subgoal_cid=str(adapted["subgoal_cid"]),
                    )
                    registered.append(registered_lane)
                    if receipt_drained and self._disposition(registered_lane) == "completed":
                        receipt_drained_completion_task_cids.add(
                            registered_lane.task_cid
                        )

                for lane in registered:
                    # Live workers can still idle after a repair-budget burn
                    # (for example a broken provider runner). Reset their
                    # portal attempt budget while they hold the lease so the
                    # daemon can redispatch without waiting for process exit.
                    if lane.task_cid in self._running:
                        if self._authoritative_lane_has_open_work(lane):
                            if self._reopen_portal_task_state_for_board(lane):
                                logger.info(
                                    "Reset deferred repair budget for active lane %s",
                                    lane.bundle_key,
                                )
                            self._reopen_runtime_todo_statuses(lane)
                        continue
                    # Board reopen must run *before* disposition. Runtime
                    # portal todos often still say completed after residual
                    # requeue; disposition then short-circuits and never
                    # relaunches install lanes that the shard still marks todo.
                    if self._authoritative_lane_has_open_work(lane):
                        coordinator.requeue_completed(
                            lane.task_cid,
                            reason="bundle_board_reopened",
                        )
                        # Attempt budgets burned by supervisor restarts must
                        # also clear so residual FVT install/packaging lanes
                        # become claimable again without operator SQL.
                        coordinator.requeue_exhausted_blocked(
                            lane.task_cid,
                            reason="bundle_board_reopened",
                        )
                        self._reopen_portal_task_state_for_board(lane)
                        self._reopen_runtime_todo_statuses(lane)
                    disposition = self._disposition(lane)
                    if disposition:
                        if (
                            disposition == "completed"
                            and lane.task_cid
                            in receipt_drained_completion_task_cids
                        ):
                            coordinator.requeue_exhausted_blocked(
                                lane.task_cid,
                                reason="receipt_drained_completion",
                            )
                        # Even with a completed disposition, keep residual
                        # board-open lanes launchable after the reopen above.
                        if not self._authoritative_lane_has_open_work(lane):
                            continue
                    current_projection = coordinator.task_state(lane.task_cid) or {}
                    if (
                        self._authoritative_lane_has_open_work(lane)
                        and str(
                            current_projection.get("state")
                            or current_projection.get("lease_state")
                            or ""
                        )
                        == "completed"
                    ):
                        coordinator.requeue_completed(
                            lane.task_cid,
                            reason="bundle_board_reopened_stale_completion",
                        )
                        self._reopen_portal_task_state_for_board(lane)
                        self._reopen_runtime_todo_statuses(lane)
                        current_projection = (
                            coordinator.task_state(lane.task_cid) or {}
                        )
                    if not self._receipt_backed_attempt_limit_disposition(
                        lane,
                        current_projection,
                    ):
                        coordinator.requeue_exhausted_blocked(
                            lane.task_cid,
                            reason="bundle_board_reopened",
                        )
                current_task_cids = {
                    *(lane.task_cid for lane in registered),
                    *self._running.keys(),
                    *reconciled.keys(),
                }
                decision_projection = coordinator.list_tasks(
                    task_cids=current_task_cids,
                    include_claimability=True,
                )
                ready_input_binding_task_cids = {
                    str(item.get("task_cid") or "")
                    for item in decision_projection
                    if self._projection_state(item) == "ready"
                }
                stale_input_bindings: dict[str, dict[str, Any]] = {}
                for lane in registered:
                    if lane.task_cid not in ready_input_binding_task_cids:
                        continue
                    # Skip rematerialization while a worker is already holding
                    # the runtime board; still surface the diagnosis so the
                    # lane is not re-launched under a mismatched plan digest.
                    if lane.task_cid in self._running:
                        diagnosis = stale_bundle_lane_input_binding(
                            lane,
                            repo_root=self.repo_root,
                        )
                        if diagnosis is not None:
                            stale_input_bindings[lane.task_cid] = diagnosis
                        continue
                    diagnosis = stale_bundle_lane_input_binding(
                        lane,
                        repo_root=self.repo_root,
                    )
                    if diagnosis is None:
                        continue
                    refreshed = refresh_stale_bundle_lane_input_binding(
                        lane,
                        repo_root=self.repo_root,
                    )
                    if refreshed and refreshed.get("refreshed"):
                        logger.info(
                            "Auto-refreshed stale taskboard binding for %s during reconcile",
                            lane.bundle_key,
                        )
                        continue
                    if refreshed is not None and not refreshed.get("refreshed"):
                        diagnosis = {
                            **diagnosis,
                            "refresh_error": str(
                                refreshed.get("error") or "refresh_failed"
                            ),
                        }
                    stale_input_bindings[lane.task_cid] = diagnosis
                for item in decision_projection:
                    task_cid = str(item.get("task_cid") or "")
                    diagnosis = stale_input_bindings.get(task_cid)
                    if diagnosis is not None:
                        item["state"] = "blocked"
                        item["blocked_reason"] = "stale_input_binding"
                        item["stale_input_binding"] = dict(diagnosis)
                decision_projection_by_task_cid = {
                    str(item.get("task_cid") or ""): item
                    for item in decision_projection
                    if str(item.get("task_cid") or "")
                }
                decision_snapshot = self._build_scheduler_snapshot(registered, decision_projection)
                registered_by_task_cid = {
                    lane.task_cid: lane for lane in registered
                }
                snapshot_ready = {
                    str(item.get("task_cid") or "")
                    for item in decision_projection
                    if self._projection_state(item) == "ready"
                    and (
                        registered_by_task_cid.get(str(item.get("task_cid") or "")) is None
                        or not registered_by_task_cid[
                            str(item.get("task_cid") or "")
                        ].queue_payload.get("external_active_member_fence")
                    )
                }
                decisions: list[dict[str, Any]] = [
                    {
                        "task_cid": task_cid,
                        "bundle_key": result["bundle_key"],
                        "decision": "settled",
                        "reason": (
                            f"reconciled_terminal_{result['disposition']}"
                        ),
                        "recovery": "untracked_accepted_lease",
                        "snapshot_id": decision_snapshot.snapshot_id,
                    }
                    for task_cid, result in reconciled.items()
                ]
                running_by_bundle_key = {
                    running.spec.bundle_key: running
                    for running in self._running.values()
                }
                dispositions = {
                    lane.task_cid: (
                        self._disposition(lane)
                        or self._receipt_backed_attempt_limit_disposition(
                            lane,
                            decision_projection_by_task_cid.get(lane.task_cid, {}),
                        )
                    )
                    for lane in registered
                    if lane.task_cid not in self._running and lane.task_cid in snapshot_ready
                    and lane.bundle_key not in running_by_bundle_key
                }
                resource_candidates = [
                    lane
                    for lane in registered
                    if lane.task_cid not in self._running
                    and lane.bundle_key not in running_by_bundle_key
                    and lane.task_cid in snapshot_ready
                    and not dispositions.get(lane.task_cid)
                    and not _legacy_adoption_lane_blocked(lane)
                    and not any(
                        _lanes_conflict(lane, running.spec)
                        for running in self._running.values()
                    )
                ]
                try:
                    host_resources = self._sample_host_resources(
                        active_workers=accounted_active_workers,
                    )
                except Exception:
                    logger.exception("Host resource sampling failed; retaining configured bounds")
                    host_resources = HostResourceSnapshot(
                        active_workers=accounted_active_workers,
                        worker_limit=self.max_lanes,
                        available_worker_capacity=max(
                            0,
                            self.max_lanes - accounted_active_workers,
                        ),
                    )
                try:
                    provider_capacities = self._provider_capacities(coordinator)
                except Exception:
                    logger.exception("Provider capacity sampling failed")
                    provider_capacities = ()
                self._transition_running_resource_stages(
                    host=host_resources,
                    providers=provider_capacities,
                )
                resource_requirements = {
                    lane.task_cid: self._lane_resource_requirement(lane)
                    for lane in resource_candidates
                }
                # Evaluate the complete ready set once. Adaptive fairness may
                # reorder decisions, so all consumers address decisions by
                # canonical lane identity rather than positional coincidence.
                candidate_schedule = self.resource_scheduler.schedule(
                    resource_requirements.values(),
                    host=host_resources,
                    providers=provider_capacities,
                    path=self.state_root,
                    active_workers=accounted_active_workers,
                )
                confirmed_requirements: list[LaneResourceRequirements] = []
                resource_cycle_decisions: list[AdmissionDecision] = []
                admission_slot_reopened = False
                candidate_rank = {
                    decision.lane_id: index
                    for index, decision in enumerate(candidate_schedule.decisions)
                }
                ordered_registered = [
                    lane
                    for _index, lane in sorted(
                        enumerate(registered),
                        key=lambda item: (
                            (
                                0,
                                candidate_rank[
                                    resource_requirements[item[1].task_cid].lane_id
                                ],
                            )
                            if item[1].task_cid in resource_requirements
                            else (1, item[0])
                        ),
                    )
                ]
                for lane in ordered_registered:
                    if lane.task_cid in self._running:
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "retained",
                            "reason": "already_active",
                            "snapshot_id": decision_snapshot.snapshot_id,
                        })
                        continue
                    if lane.task_cid in reconciled:
                        continue
                    if _legacy_adoption_lane_blocked(lane):
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "deferred",
                            "reason": LEGACY_ADOPTION_BARRIER_REASON,
                            "snapshot_id": decision_snapshot.snapshot_id,
                            "legacy_adoption_barrier": dict(
                                lane.queue_payload.get(
                                    "legacy_adoption_barrier"
                                )
                                or {}
                            ),
                        })
                        continue
                    stale_input_binding = stale_input_bindings.get(lane.task_cid)
                    if stale_input_binding is not None:
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "deferred",
                            **stale_input_binding,
                            "snapshot_id": decision_snapshot.snapshot_id,
                        })
                        continue
                    scope_owner = running_by_bundle_key.get(lane.bundle_key)
                    if scope_owner is not None:
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "deferred",
                            "reason": "bundle_key_active",
                            "blocking_task_cid": scope_owner.spec.task_cid,
                            "snapshot_id": decision_snapshot.snapshot_id,
                        })
                        continue
                    if lane.task_cid not in snapshot_ready:
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "deferred",
                            "reason": "snapshot_not_ready",
                            "snapshot_id": decision_snapshot.snapshot_id,
                        })
                        continue
                    if any(_lanes_conflict(lane, running.spec) for running in self._running.values()):
                        # A color is a reusable placement hint, not a lock.
                        # Only an actual graph edge against a live lane blocks
                        # admission, and a later reconciliation retries it.
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "deferred",
                            "reason": "conflict",
                            "snapshot_id": decision_snapshot.snapshot_id,
                        })
                        continue
                    disposition = dispositions.get(lane.task_cid, "")
                    if disposition:
                        grant = coordinator.claim_ready(
                            self.claimant_did,
                            requested_lease_ms=self.lease_ms,
                            eligible_task_cids=(lane.task_cid,),
                        )
                        if grant is not None:
                            try:
                                self._settle_grant(coordinator, grant, disposition=disposition)
                            except LeaseError:
                                pass
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "settled" if grant is not None else "deferred",
                            "reason": disposition if grant is not None else "lease_unavailable",
                            "snapshot_id": decision_snapshot.snapshot_id,
                        })
                        continue
                    requirement = resource_requirements.get(lane.task_cid)
                    admission = (
                        candidate_schedule.decision_for(requirement.lane_id)
                        if requirement is not None
                        else None
                    )
                    resource_decision_index = len(resource_cycle_decisions)
                    if admission is not None:
                        resource_cycle_decisions.append(admission)
                    resource_lease: ResourceAdmissionLease | None = None
                    if (
                        requirement is not None
                        and admission_slot_reopened
                        and (admission is None or not admission.admitted)
                    ):
                        # A coordination lease race after the batch preview
                        # returns its resource slot immediately. Recheck later
                        # candidates so a single lost claim cannot leave the
                        # pool idle until the next polling cycle.
                        admission, resource_lease = self.resource_scheduler.acquire(
                            requirement,
                            host=host_resources,
                            providers=provider_capacities,
                            path=self.state_root,
                        )
                        if resource_decision_index == len(resource_cycle_decisions):
                            resource_cycle_decisions.append(admission)
                        else:
                            resource_cycle_decisions[resource_decision_index] = admission
                    if admission is None or not admission.admitted:
                        evidence = admission.to_dict() if admission is not None else {}
                        reasons = list(admission.reasons) if admission is not None else ["resource_capacity"]
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "deferred",
                            "reason": reasons[0],
                            "backpressure_reasons": reasons,
                            "resource_admission": evidence,
                            "snapshot_id": decision_snapshot.snapshot_id,
                        })
                        continue
                    # A schedule is an explainable preview; acquire is the
                    # atomic reservation boundary. It rechecks live active
                    # leases so concurrent supervisor stages cannot over-admit.
                    if resource_lease is None:
                        live_admission, resource_lease = self.resource_scheduler.acquire(
                            requirement,
                            host=host_resources,
                            providers=provider_capacities,
                            path=self.state_root,
                        )
                        if resource_lease is None:
                            admission_slot_reopened = True
                            resource_cycle_decisions[-1] = live_admission
                            decisions.append({
                                "task_cid": lane.task_cid,
                                "bundle_key": lane.bundle_key,
                                "decision": "deferred",
                                "reason": live_admission.reason or "resource_capacity",
                                "backpressure_reasons": list(live_admission.reasons),
                                "resource_admission": live_admission.to_dict(),
                                "snapshot_id": decision_snapshot.snapshot_id,
                            })
                            continue
                        admission = live_admission
                        resource_cycle_decisions[-1] = admission
                    grant = coordinator.claim_ready(
                        self.claimant_did,
                        requested_lease_ms=self.lease_ms,
                        eligible_task_cids=(lane.task_cid,),
                    )
                    if grant is None:
                        self.resource_scheduler.release(resource_lease)
                        admission_slot_reopened = True
                        resource_cycle_decisions[-1] = replace(
                            admission,
                            admitted=False,
                            reasons=("lease_unavailable",),
                            reserved_quota_units=0,
                            reserved_tokens=0,
                        )
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "deferred",
                            "reason": "lease_unavailable",
                            "resource_admission": resource_cycle_decisions[-1].to_dict(),
                            "snapshot_id": decision_snapshot.snapshot_id,
                        })
                        continue
                    if admission.provider_id and admission.provider_id != lane.llm_provider:
                        lane = replace(lane, llm_provider=admission.provider_id)
                    try:
                        handle = self._launcher(lane, grant)
                    except Exception:
                        self.resource_scheduler.release(resource_lease)
                        admission_slot_reopened = True
                        try:
                            coordinator.release(grant, reason="launch failed")
                        except LeaseError:
                            pass
                        resource_cycle_decisions[-1] = replace(
                            admission,
                            admitted=False,
                            reasons=("launch_failed",),
                            reserved_quota_units=0,
                            reserved_tokens=0,
                        )
                        logger.exception("Failed to launch bundle lane %s", lane.bundle_key)
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "deferred",
                            "reason": "launch_failed",
                            "resource_admission": resource_cycle_decisions[-1].to_dict(),
                            "snapshot_id": decision_snapshot.snapshot_id,
                        })
                        continue
                    self._running[lane.task_cid] = RunningBundleLane(
                        spec=lane,
                        grant=grant,
                        handle=handle,
                        started_at=utc_now(),
                        resource_lease=resource_lease,
                    )
                    confirmed_requirements.append(requirement)
                    launched.append(lane.task_cid)
                    decisions.append({
                        "task_cid": lane.task_cid,
                        "bundle_key": lane.bundle_key,
                        "decision": "launched",
                        "reason": "ready_capacity",
                        "resource_admission": admission.to_dict(),
                        "snapshot_id": decision_snapshot.snapshot_id,
                    })

                resource_backpressure = tuple(
                    dict.fromkeys(
                        reason
                        for decision in resource_cycle_decisions
                        if not decision.admitted
                        for reason in decision.reasons
                    )
                )
                resource_backpressure_counts = {
                    reason: sum(
                        1
                        for decision in resource_cycle_decisions
                        if not decision.admitted and reason in decision.reasons
                    )
                    for reason in resource_backpressure
                }
                self._last_resource_snapshot = replace(
                    candidate_schedule,
                    decisions=tuple(resource_cycle_decisions),
                    admitted_count=len(confirmed_requirements),
                    backpressure_reasons=resource_backpressure,
                    adaptive_metrics=self.resource_scheduler.metrics_snapshot(
                        observed_at_ms=candidate_schedule.observed_at_ms,
                    ),
                    active_lease_count=len(self.resource_scheduler.active_leases),
                    backpressure_counts=resource_backpressure_counts,
                )
                current_task_cids = {
                    *(lane.task_cid for lane in registered),
                    *self._running.keys(),
                    *reconciled.keys(),
                }
                projection = coordinator.list_tasks(
                    task_cids=current_task_cids,
                    include_claimability=True,
                )
                for item in projection:
                    task_cid = str(item.get("task_cid") or "")
                    diagnosis = stale_input_bindings.get(task_cid)
                    if (
                        diagnosis is not None
                        and self._projection_state(item) == "ready"
                    ):
                        item["state"] = "blocked"
                        item["blocked_reason"] = "stale_input_binding"
                        item["stale_input_binding"] = dict(diagnosis)
                current_snapshot = self._build_scheduler_snapshot(registered, projection)
                if (
                    self._cycle % COORDINATION_COMPACTION_INTERVAL_CYCLES == 0
                    and self.coordination_path.stat().st_size
                    >= COORDINATION_COMPACTION_MIN_BYTES
                ):
                    try:
                        self._last_coordination_compaction = coordinator.compact()
                    except Exception:
                        logger.exception(
                            "Could not compact DuckDB coordination store"
                        )
            return self._write_live_manifest(
                discovered=discovered,
                task_projection=projection,
                launched=launched,
                reaped=reaped,
                reconciled=tuple(reconciled),
                scheduler_state=current_snapshot,
                decision_snapshot=decision_snapshot,
                decisions=decisions,
            )

    def run(
        self,
        *,
        max_cycles: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Reconcile until stopped; an empty queue never terminates the pool."""

        external_stop = stop_event
        cycles = 0
        payload: dict[str, Any] = {}
        if max_cycles is not None and int(max_cycles) <= 0:
            return payload
        install_handlers = max_cycles is None and threading.current_thread() is threading.main_thread()
        previous_term: Any = None
        previous_int: Any = None
        run_error: BaseException | None = None

        def request_stop(_signum: int, _frame: object) -> None:
            self._stop_event.set()

        if install_handlers:
            previous_term = signal.signal(signal.SIGTERM, request_stop)
            previous_int = signal.signal(signal.SIGINT, request_stop)
        try:
            while not self._stop_event.is_set() and not (external_stop and external_stop.is_set()):
                payload = self.reconcile_once()
                if cycles % SCHEDULER_GC_INTERVAL_CYCLES == 0:
                    gc.collect()
                cycles += 1
                if max_cycles is not None and cycles >= int(max_cycles):
                    break
                wait_for = max(0.01, self.poll_interval)
                if self._stop_event.wait(wait_for):
                    break
                if external_stop and external_stop.is_set():
                    break
        except BaseException as exc:
            run_error = exc
            self._stop_event.set()
            raise
        finally:
            if install_handlers:
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)
            if self._stop_event.is_set() or (external_stop and external_stop.is_set()):
                try:
                    payload = self.stop()
                except BaseException:
                    if run_error is None:
                        raise
                    logger.exception(
                        "Scheduler cleanup failed while preserving the original run error"
                    )
        return payload

    def stop(self, *, grace_seconds: float = 5.0) -> dict[str, Any]:
        """Stop owned processes and release only leases still fenced to them."""

        self._stop_event.set()
        with self._lock:
            stopped_running = list(self._running.items())
            stopped_task_cids = set(self._running)
            for _task_cid, running in stopped_running:
                self._terminate_handle(running.handle, grace_seconds=grace_seconds)
                if running.resource_lease is not None:
                    self.resource_scheduler.cancel(
                        running.resource_lease,
                        reason="scheduler_stopped",
                    )
            self._running.clear()

            with LeaseCoordinator(self.coordination_path) as coordinator:
                for task_cid, running in stopped_running:
                    try:
                        if coordinator.active_lease(task_cid) is not None:
                            coordinator.release(
                                running.grant,
                                reason="scheduler stopped",
                            )
                    except LeaseError:
                        pass
                try:
                    discovered = self._plan()
                except (OSError, ValueError, json.JSONDecodeError):
                    discovered = []
                current_task_cids = {
                    lane.task_cid
                    for lane in discovered
                    if lane.task_cid
                }
                current_task_cids.update(stopped_task_cids)
                projection = (
                    coordinator.list_tasks(
                        task_cids=current_task_cids,
                        include_claimability=True,
                    )
                    if current_task_cids
                    else []
                )
        return self._write_live_manifest(
            discovered=discovered,
            task_projection=projection,
            launched=(),
            reaped=(),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or launch isolated daemon lanes for objective bundle shards")
    parser.add_argument("--bundle-index-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--state-root", type=Path, default=None)
    parser.add_argument("--worktree-root", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_PREFIX)
    parser.add_argument(
        "--exclude-bundle-key",
        action="append",
        default=[],
        help=(
            "Exact bundle key to omit from this supervisor run; repeat the "
            "option to fence multiple bundles without rewriting the index"
        ),
    )
    parser.add_argument(
        "--exclude-task-id",
        action="append",
        default=[],
        help=(
            "Exact task ID to omit from this supervisor run; repeat the option "
            "to fence members without rewriting their shared bundle or taskboard"
        ),
    )
    parser.add_argument("--start", action="store_true", help="Launch the planned lane supervisors")
    parser.add_argument("--max-lanes", type=int, default=1, help="Maximum concurrent leased workers")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument(
        "--bundle-index-refresh-command",
        default="",
        help=(
            "Optional argv string that atomically refreshes a derived bundle "
            "index before planning against a changed completion-receipt stream"
        ),
    )
    parser.add_argument(
        "--bundle-index-refresh-timeout-seconds",
        type=float,
        default=60.0,
        help="Positive timeout for the optional bundle-index refresh command",
    )
    parser.add_argument("--once", action="store_true", help="Run one reconciliation cycle and exit")
    implement_group = parser.add_mutually_exclusive_group()
    implement_group.add_argument("--implement", dest="implement", action="store_true")
    implement_group.add_argument("--no-implement", dest="implement", action="store_false")
    parser.set_defaults(implement=False)
    parser.add_argument("--daemon-interval", type=float, default=300.0)
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--check-interval", type=float, default=60.0)
    parser.add_argument("--watchdog-startup-grace-seconds", type=float, default=None)
    parser.add_argument("--max-restarts", type=int, default=0)
    parser.add_argument(
        "--max-task-attempts",
        type=int,
        default=0,
        help=(
            "Maximum implementation attempts per canonical task identity in each lane. "
            "Zero disables the limit."
        ),
    )
    parser.add_argument("--implementation-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--implementation-max-timeout",
        type=float,
        default=None,
        help=(
            "Maximum task-specific implementation hard timeout across each "
            "lane. This extends only the child supervisor watchdog; "
            "--implementation-timeout remains the daemon's ordinary policy."
        ),
    )
    parser.add_argument(
        "--implementation-log-stall-seconds",
        type=float,
        default=None,
        help=(
            "Recycle an active implementation attempt after this many "
            "seconds without supervisor-observed log output; <=0 disables."
        ),
    )
    parser.add_argument("--implementation-command", default="")
    parser.add_argument(
        "--production-provider-policy",
        default="",
        choices=("", PRODUCTION_CLI_POLICY_NAME),
        help=(
            "Operator policy overlay for the typed Grok implementation and "
            "independent Codex review route; task metadata and CIDs are unchanged."
        ),
    )
    parser.add_argument(
        "--production-provider-context-budget-tokens",
        type=int,
        default=0,
        help=(
            "Positive provider packet context budget. Zero uses the policy "
            f"default ({DEFAULT_PRODUCTION_CONTEXT_BUDGET_TOKENS})."
        ),
    )
    parser.add_argument(
        "--production-provider-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Bounded per-provider timeout for the production route. "
            f"Zero selects the policy default ({DEFAULT_PRODUCTION_PROVIDER_TIMEOUT_SECONDS:g}s)."
        ),
    )
    parser.add_argument(
        "--production-provider-review-authority-key-path",
        type=Path,
        default=None,
        help=(
            "Shared operator-controlled Ed25519 private-key file for every "
            "bundle lane. Defaults beneath --state-root."
        ),
    )
    parser.add_argument(
        "--production-provider-launch-authority-receipt-path",
        type=Path,
        default=None,
        help="Exact operator-owned admitted control-launch receipt.",
    )
    parser.add_argument(
        "--production-provider-launch-authority-receipt-content-id",
        default="",
        help="Exact sha256 content identity of the admitted control-launch receipt.",
    )
    parser.add_argument(
        "--legacy-landed-review-policy-path",
        type=Path,
        default=None,
        help=(
            "Explicit operator-owned LegacyLandedReviewPolicy@2 JSON. Omitted "
            "by default; supplying a path does not enable a disabled policy."
        ),
    )
    parser.add_argument(
        "--legacy-landed-review-key-path",
        type=Path,
        default=None,
        help=(
            "Explicit mode-0600 Ed25519 key bound by the legacy policy. No "
            "key is inferred or generated by this migration path."
        ),
    )
    parser.add_argument(
        "--merge-target-branch",
        default="",
        help=(
            "Existing branch that receives each isolated lane merge. When omitted, "
            "the child supervisor retains its main/master/current-branch fallback."
        ),
    )
    parser.add_argument("--llm-merge-resolver-command", default="")
    parser.add_argument("--llm-merge-resolver-timeout-seconds", type=float, default=None)
    parser.add_argument("--merge-reconciliation-max-merges", type=int, default=None)
    parser.add_argument("--auto-commit-generated-dirty", dest="generated_dirty_repair_enabled", action="store_true")
    parser.set_defaults(generated_dirty_repair_enabled=False)
    parser.add_argument("--generated-dirty-commit-subject", default="Agent: commit generated supervisor outputs")
    parser.add_argument(
        "--no-generated-dirty-submodule-gitlinks",
        dest="generated_dirty_repair_include_submodule_gitlinks",
        action="store_false",
    )
    parser.set_defaults(generated_dirty_repair_include_submodule_gitlinks=False)
    parser.add_argument("--generated-dirty-max-paths", type=int, default=200)
    parser.add_argument("--generated-dirty-stale-lock-seconds", type=float, default=300.0)
    parser.add_argument("--generated-dirty-path", type=Path, action="append", default=[])
    parser.add_argument(
        "--worktree-submodule-path",
        action="append",
        default=[],
        help="Repeatable nested submodule path to prepare, commit, merge, and clean in every lane.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--coordination-path", type=Path, default=None)
    parser.add_argument("--claimant-did", default="did:web:ipfs-accelerate.local")
    parser.add_argument("--lease-ms", type=int, default=60_000)
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--capacity-millionths", type=int, default=1_000_000)
    parser.add_argument("--max-cpu-percent", type=int, default=90)
    parser.add_argument("--max-memory-percent", type=int, default=90)
    parser.add_argument("--max-disk-percent", type=int, default=95)
    parser.add_argument(
        "--max-cpu-proof-concurrency",
        type=int,
        default=0,
        help=(
            "Maximum concurrent CPU proof/validation pool slots. "
            "Zero defaults to --max-lanes so multi-lane programs are not "
            "starved by the one-slot lease-budget default."
        ),
    )
    parser.add_argument(
        "--max-model-concurrency",
        type=int,
        default=0,
        help="Maximum concurrent model/inference pool slots (0 => --max-lanes).",
    )
    parser.add_argument(
        "--max-artifact-concurrency",
        type=int,
        default=0,
        help="Maximum concurrent artifact/I/O pool slots (0 => --max-lanes).",
    )
    parser.add_argument("--minimum-memory-available-bytes", type=int, default=0)
    parser.add_argument("--minimum-disk-available-bytes", type=int, default=0)
    parser.add_argument("--maximum-provider-latency-ms", type=int, default=120_000)
    parser.add_argument("--provider-quota-reserve", type=int, default=0)
    parser.add_argument("--provider-token-reserve", type=int, default=0)
    parser.add_argument("--provider-capacity-path", type=Path, default=None)
    parser.add_argument(
        "--provider-capacity-max-age-ms",
        type=int,
        default=DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS,
        help=(
            "Hard freshness TTL for both Grok and Codex capacity samples; "
            "missing or older samples close dual-review admission"
        ),
    )
    parser.add_argument(
        "--external-task-state-path",
        type=Path,
        action="append",
        default=[],
        help="Repeatable serial scheduler state file whose active task fences matching bundles.",
    )
    parser.add_argument(
        "--allow-missing-provider-telemetry",
        action="store_true",
        help="Allow provider-dependent lanes when no capacity monitor is available",
    )
    return parser


def run_bundle_supervisor(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    state_root = (args.state_root or default_state_root(repo_root)).resolve()
    worktree_root = (args.worktree_root or state_root / "worktrees").resolve()
    log_dir = (args.log_dir or state_root / "logs").resolve()
    manifest_path = (args.manifest_path or state_root / "bundle_lanes.json").resolve()
    bundle_index_path = args.bundle_index_path.resolve()
    production_provider_policy = str(
        getattr(args, "production_provider_policy", "") or ""
    ).strip()
    raw_production_budget = int(
        getattr(args, "production_provider_context_budget_tokens", 0) or 0
    )
    raw_production_timeout = float(
        getattr(args, "production_provider_timeout_seconds", 0.0) or 0.0
    )
    raw_review_authority_key_path = getattr(
        args, "production_provider_review_authority_key_path", None
    )
    raw_launch_authority_path = getattr(
        args, "production_provider_launch_authority_receipt_path", None
    )
    raw_launch_authority_content_id = str(
        getattr(
            args,
            "production_provider_launch_authority_receipt_content_id",
            "",
        )
        or ""
    ).strip()
    if (
        raw_production_budget
        or raw_production_timeout
        or raw_review_authority_key_path is not None
        or raw_launch_authority_path is not None
        or raw_launch_authority_content_id
    ) and not production_provider_policy:
        raise ValueError(
            "production provider bounds/review authority require a production "
            "provider policy"
        )
    production_provider_context_budget_tokens = 0
    production_provider_timeout_seconds = 0.0
    production_provider_review_authority_key_path = None
    production_provider_launch_authority_receipt_path = None
    production_provider_launch_authority_receipt_content_id = ""
    if bool(raw_launch_authority_path) != bool(raw_launch_authority_content_id):
        raise ValueError(
            "production provider launch authority path/content identity are required together"
        )
    if production_provider_policy:
        production_policy = ProductionCLIProviderPolicy(
            name=production_provider_policy,
            context_budget_tokens=(
                raw_production_budget
                or DEFAULT_PRODUCTION_CONTEXT_BUDGET_TOKENS
            ),
            provider_timeout_seconds=(
                raw_production_timeout
                or DEFAULT_PRODUCTION_PROVIDER_TIMEOUT_SECONDS
            ),
        )
        production_provider_context_budget_tokens = (
            production_policy.context_budget_tokens
        )
        production_provider_timeout_seconds = float(
            production_policy.provider_timeout_seconds
        )
        production_provider_review_authority_key_path = Path(
            raw_review_authority_key_path
            or state_root / DEFAULT_PRODUCTION_PROVIDER_REVIEW_KEY_NAME
        )
        if raw_launch_authority_path is not None:
            if not re.fullmatch(
                r"sha256:[0-9a-f]{64}", raw_launch_authority_content_id
            ):
                raise ValueError(
                    "production provider launch authority content identity is malformed"
                )
            production_provider_launch_authority_receipt_path = Path(
                raw_launch_authority_path
            )
            production_provider_launch_authority_receipt_content_id = (
                raw_launch_authority_content_id
            )
    legacy_landed_review_policy_path = getattr(
        args, "legacy_landed_review_policy_path", None
    )
    legacy_landed_review_key_path = getattr(
        args, "legacy_landed_review_key_path", None
    )
    if (legacy_landed_review_policy_path is None) != (
        legacy_landed_review_key_path is None
    ):
        raise ValueError(
            "legacy landed review requires both explicit policy and key paths"
        )
    if (
        legacy_landed_review_policy_path is not None
        and legacy_landed_review_key_path is not None
    ):
        legacy_policy = load_legacy_landed_review_policy(
            legacy_landed_review_policy_path
        )
        legacy_authority = LegacyLandedReviewAuthority.from_private_key_path(
            legacy_landed_review_key_path
        )
        if legacy_policy.issuer_key_id != legacy_authority.issuer_key_id:
            raise ValueError("legacy landed review policy/key binding is invalid")
    lane_options = dict(
        task_prefix=args.task_prefix,
        excluded_bundle_keys=tuple(
            str(bundle_key).strip()
            for bundle_key in (getattr(args, "exclude_bundle_key", ()) or ())
            if str(bundle_key).strip()
        ),
        excluded_task_ids=tuple(
            str(task_id).strip()
            for task_id in (getattr(args, "exclude_task_id", ()) or ())
            if str(task_id).strip()
        ),
        implement=args.implement,
        daemon_interval=args.daemon_interval,
        stale_seconds=args.stale_seconds,
        check_interval=args.check_interval,
        watchdog_startup_grace_seconds=args.watchdog_startup_grace_seconds,
        max_restarts=args.max_restarts,
        max_task_attempts=max(0, int(getattr(args, "max_task_attempts", 0))),
        implementation_timeout=args.implementation_timeout,
        implementation_max_timeout=getattr(
            args,
            "implementation_max_timeout",
            None,
        ),
        implementation_log_stall_seconds=getattr(
            args,
            "implementation_log_stall_seconds",
            None,
        ),
        implementation_command=args.implementation_command,
        production_provider_policy=production_provider_policy,
        production_provider_context_budget_tokens=(
            production_provider_context_budget_tokens
        ),
        production_provider_timeout_seconds=production_provider_timeout_seconds,
        production_provider_review_authority_key_path=(
            production_provider_review_authority_key_path
        ),
        production_provider_launch_authority_receipt_path=(
            production_provider_launch_authority_receipt_path
        ),
        production_provider_launch_authority_receipt_content_id=(
            production_provider_launch_authority_receipt_content_id
        ),
        legacy_landed_review_policy_path=legacy_landed_review_policy_path,
        legacy_landed_review_key_path=legacy_landed_review_key_path,
        merge_target_branch=args.merge_target_branch,
        llm_merge_resolver_command=args.llm_merge_resolver_command,
        llm_merge_resolver_timeout_seconds=args.llm_merge_resolver_timeout_seconds,
        merge_reconciliation_max_merges=args.merge_reconciliation_max_merges,
        generated_dirty_repair_enabled=args.generated_dirty_repair_enabled,
        generated_dirty_repair_commit_subject=args.generated_dirty_commit_subject,
        generated_dirty_repair_include_submodule_gitlinks=args.generated_dirty_repair_include_submodule_gitlinks,
        generated_dirty_repair_max_paths=args.generated_dirty_max_paths,
        generated_dirty_repair_stale_lock_seconds=args.generated_dirty_stale_lock_seconds,
        generated_dirty_repair_paths=tuple(args.generated_dirty_path or ()),
        worktree_submodule_paths=tuple(args.worktree_submodule_path or ()),
        log_level=args.log_level,
    )
    if args.start:
        scheduler = DynamicBundleScheduler(
            bundle_index_path=bundle_index_path,
            repo_root=repo_root,
            state_root=state_root,
            worktree_root=worktree_root,
            log_dir=log_dir,
            manifest_path=manifest_path,
            metrics_path=getattr(args, "metrics_path", None),
            coordination_path=getattr(args, "coordination_path", None),
            max_lanes=getattr(args, "max_lanes", 1) or 1,
            claimant_did=getattr(args, "claimant_did", "did:web:ipfs-accelerate.local"),
            lease_ms=getattr(args, "lease_ms", 60_000),
            heartbeat_interval=getattr(args, "heartbeat_interval", 5.0),
            capacity_millionths=getattr(args, "capacity_millionths", 1_000_000),
            poll_interval=getattr(args, "poll_interval", 5.0),
            provider_capacity_path=getattr(args, "provider_capacity_path", None),
            provider_capacity_max_age_ms=getattr(
                args,
                "provider_capacity_max_age_ms",
                DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS,
            ),
            external_task_state_paths=tuple(
                getattr(args, "external_task_state_path", ()) or ()
            ),
            bundle_index_refresh_command=getattr(
                args,
                "bundle_index_refresh_command",
                "",
            ),
            bundle_index_refresh_timeout_seconds=getattr(
                args,
                "bundle_index_refresh_timeout_seconds",
                60.0,
            ),
            resource_policy={
                "max_lanes": getattr(args, "max_lanes", 1) or 1,
                # Scale CPU/model/artifact pools with the configured lane
                # budget so multi-lane FVT programs are not capped at one
                # concurrent validation/proof slot by the zero defaults.
                "max_cpu_proof_concurrency": int(
                    getattr(args, "max_cpu_proof_concurrency", 0)
                    or getattr(args, "max_lanes", 1)
                    or 1
                ),
                "max_model_concurrency": int(
                    getattr(args, "max_model_concurrency", 0)
                    or getattr(args, "max_lanes", 1)
                    or 1
                ),
                "max_artifact_concurrency": int(
                    getattr(args, "max_artifact_concurrency", 0)
                    or getattr(args, "max_lanes", 1)
                    or 1
                ),
                # Adaptive contraction was reducing multi-lane FVT programs to a
                # single analysis slot under shared-host load, starving ready
                # work. Keep deterministic stage/pool caps equal to max-lanes.
                "adaptive_enabled": False,
                "stage_concurrency_limits": {
                    "analysis": int(getattr(args, "max_lanes", 1) or 1),
                    "inference": int(getattr(args, "max_lanes", 1) or 1),
                    "validation": int(getattr(args, "max_lanes", 1) or 1),
                    "proof": int(getattr(args, "max_lanes", 1) or 1),
                    "merge": max(1, min(2, int(getattr(args, "max_lanes", 1) or 1))),
                    "persistence": int(getattr(args, "max_lanes", 1) or 1),
                },
                "max_cpu_percent": getattr(args, "max_cpu_percent", 90),
                "max_memory_percent": getattr(args, "max_memory_percent", 90),
                "max_disk_percent": getattr(args, "max_disk_percent", 95),
                "minimum_memory_available_bytes": getattr(args, "minimum_memory_available_bytes", 0),
                "minimum_disk_available_bytes": getattr(args, "minimum_disk_available_bytes", 0),
                "maximum_provider_latency_ms": getattr(args, "maximum_provider_latency_ms", 120_000),
                "provider_quota_reserve": getattr(args, "provider_quota_reserve", 0),
                "provider_token_reserve": getattr(args, "provider_token_reserve", 0),
                "require_provider_telemetry": not getattr(args, "allow_missing_provider_telemetry", False),
            },
            **lane_options,
        )
        return scheduler.run(max_cycles=1 if getattr(args, "once", False) else None)

    lanes = plan_bundle_lanes(
        bundle_index_path=bundle_index_path,
        repo_root=repo_root,
        state_root=state_root,
        worktree_root=worktree_root,
        log_dir=log_dir,
        max_lanes=None,
        **lane_options,
    )
    payload = write_bundle_lane_manifest(
        manifest_path=manifest_path,
        repo_root=repo_root,
        bundle_index_path=bundle_index_path,
        lanes=lanes,
        started=[],
    )
    logger.info("Planned %s bundle lanes; started 0", len(lanes))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    payload = run_bundle_supervisor(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
