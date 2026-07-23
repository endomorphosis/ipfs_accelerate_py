"""Plan and launch per-bundle todo daemon lanes."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable

from .artifact_store import (
    BUNDLE_INDEX_KIND,
    read_artifact_fields,
    write_scheduler_manifest_artifact,
)
from .conflict_graph import materialize_task_conflict_graph
from .lease_coordination import LeaseCoordinator, LeaseError
from .objective_graph import (
    DEFAULT_TASK_PREFIX,
    build_bundle_task_payloads,
    repo_relative_path,
    safe_bundle_key,
    utc_now,
)
from .event_log import event_log_sources, read_jsonl_events
from .scheduler_metrics import (
    SchedulerSnapshot,
    scheduler_snapshot,
    scheduler_state_events,
    write_scheduler_snapshot,
)
from .resource_scheduler import (
    AdmissionDecision,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ResourcePolicy,
    ResourceScheduleSnapshot,
    ResourceScheduler,
    sample_host_resources,
)
from .todo_daemon.supervisor import active_codex_exec_workers

logger = logging.getLogger(__name__)

COORDINATION_COMPACTION_INTERVAL_CYCLES = 10
COORDINATION_COMPACTION_MIN_BYTES = 64 * 1024 * 1024
SCHEDULER_GC_INTERVAL_CYCLES = 10

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


def bundle_member_completion_receipts(state_root: Path) -> dict[str, dict[str, Any]]:
    """Return successful member-task receipts keyed by canonical task CID.

    Bundle boards are mutable projections, so their current status is not a
    durable completion authority.  The implementation daemon emits terminal
    events after a successful merge; retain those receipts so a source board
    can promote the matching canonical task even after a shard is regenerated.
    """

    event_paths = sorted(
        {
            *state_root.glob("*_events.jsonl"),
            *state_root.glob("*/state/*_events.jsonl"),
        }
    )
    receipts: dict[str, dict[str, Any]] = {}
    for source in event_log_sources(event_paths, include_rotated=True):
        for event in read_jsonl_events(source):
            event_type = str(event.get("type") or "")
            task_id = str(event.get("task_id") or "")
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
                completed = (
                    event.get("returncode") == 0
                    and isinstance(merge_result, dict)
                    and merge_result.get("merged") is True
                )
            if not completed:
                continue
            canonical_task_cid = str(event.get("canonical_task_cid") or "")
            if not canonical_task_cid:
                continue
            receipt = {
                "canonical_task_cid": canonical_task_cid,
                "canonical_task_key": str(event.get("canonical_task_key") or ""),
                "task_id": task_id,
                "timestamp": str(event.get("timestamp") or ""),
                "event_type": event_type,
                "event_path": str(source),
            }
            previous = receipts.get(canonical_task_cid)
            if previous is None or receipt["timestamp"] >= previous["timestamp"]:
                receipts[canonical_task_cid] = receipt
    return receipts


@dataclass(frozen=True)
class BundleLaneSpec:
    """One isolated daemon/supervisor lane for an objective bundle shard."""

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
    required_capabilities: list[str] = field(default_factory=list)
    llm_provider: str = ""
    required_context_tokens: int = 0
    token_budget: int = 0
    max_provider_latency_ms: int = 0

    def to_dict(self, *, repo_root: Path | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for definition in fields(self):
            value = getattr(self, definition.name)
            if isinstance(value, dict):
                value = dict(value)
            elif isinstance(value, list):
                value = list(value)
            payload[definition.name] = value
        for key in ("todo_path", "state_dir", "worktree_root", "log_path"):
            path = Path(payload[key])
            payload[key] = repo_relative_path(repo_root, path) if repo_root is not None else str(path)
        return payload


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
                "lease": self.grant.to_dict(),
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
                "lease": self.grant.to_dict(),
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
    }


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
        from .todo_daemon.implementation_daemon import parse_task_file

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

    bundle_key = str(payload.get("bundle_key") or "objective/general")
    profile_g = payload.get("profile_g") if isinstance(payload.get("profile_g"), dict) else {}
    conflict_policy = str(payload.get("conflict_policy") or "")
    ast_symbols = member_values("ast_symbols", "symbols")
    ast_symbol_scopes = {item.lower() for item in member_values("ast_symbol_scope")}
    file_scoped_ast = bool(ast_symbol_scopes & {"file", "path", "local"}) or (
        "allow independent file bundles" in conflict_policy.lower()
    )
    return {
        "task_id": bundle_key,
        "task_cid": str(profile_g.get("task_cid") or payload.get("task_cid") or ""),
        "outputs": member_values("outputs", "files", "predicted_files"),
        "changed_paths": member_values("changed_paths", "actual_changed_paths"),
        "ast_symbols": ast_symbols,
        "global_ast_symbols": [] if file_scoped_ast else ast_symbols,
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
    max_restarts: int,
    implementation_timeout: float,
    implementation_command: str = "",
    llm_merge_resolver_command: str = "",
    llm_merge_resolver_timeout_seconds: float | None = None,
    merge_target_branch: str = "",
    merge_reconciliation_max_merges: int | None = None,
    generated_dirty_repair_enabled: bool = False,
    generated_dirty_repair_commit_subject: str = "",
    generated_dirty_repair_include_submodule_gitlinks: bool = False,
    generated_dirty_repair_max_paths: int | None = None,
    generated_dirty_repair_stale_lock_seconds: float | None = None,
    generated_dirty_repair_paths: Sequence[Path | str] = (),
    worktree_submodule_paths: Sequence[str] = (),
    assumed_completed_task_ids: Sequence[str] = (),
    log_level: str = "INFO",
) -> list[str]:
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
        "--implementation-timeout",
        str(implementation_timeout),
        "--log-level",
        log_level,
        # Bundle boards are projections of the canonical taskboard. Keep lane
        # workers execution-only so they cannot create shard-local task IDs or
        # mistake valid cross-bundle dependencies for missing prerequisites.
        "--no-retry-budget-guardrail",
        "--no-dependency-guardrail",
        "--no-reconciliation-guardrail",
    ]
    for relative in dict.fromkeys(str(path).strip().strip("/") for path in worktree_submodule_paths):
        if relative:
            command.extend(["--worktree-submodule-path", relative])
    command.append("--implement" if implement else "--no-implement")
    if implementation_command:
        command.extend(["--implementation-command", implementation_command])
    if llm_merge_resolver_command:
        command.extend(["--llm-merge-resolver-command", llm_merge_resolver_command])
    if llm_merge_resolver_timeout_seconds is not None:
        command.extend(["--llm-merge-resolver-timeout-seconds", str(llm_merge_resolver_timeout_seconds)])
    if merge_target_branch:
        command.extend(["--merge-target-branch", merge_target_branch])
    if merge_reconciliation_max_merges is not None:
        command.extend(["--merge-reconciliation-max-merges", str(merge_reconciliation_max_merges)])
    if generated_dirty_repair_enabled:
        command.append("--auto-commit-generated-dirty")
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
    return command


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
    max_restarts: int = 10,
    implementation_timeout: float = 1800.0,
    implementation_command: str = "",
    llm_merge_resolver_command: str = "",
    llm_merge_resolver_timeout_seconds: float | None = None,
    merge_target_branch: str = "",
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
) -> list[BundleLaneSpec]:
    """Return one isolated supervisor command for each objective bundle."""

    lanes: list[BundleLaneSpec] = []
    bundle_payloads = build_bundle_task_payloads(bundle_index_path)
    globally_completed_task_ids = {
        str(task.get("task_id") or "")
        for payload in bundle_payloads
        for task in _mapping_list(payload.get("tasks"))
        if str(task.get("status") or "").strip().lower()
        in _TERMINAL_CONFLICT_TASK_STATUSES
        and str(task.get("task_id") or "")
    }
    excluded_bundle_keys = _excluded_bundle_keys(bundle_index_path)
    bundle_payloads = [
        payload
        for payload in bundle_payloads
        if str(payload.get("bundle_key") or "objective/general") not in excluded_bundle_keys
    ]
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
        profile_g = payload.get("profile_g") if isinstance(payload.get("profile_g"), dict) else {}
        resource_fields = _resource_lane_fields(payload)
        command = implementation_supervisor_command(
            todo_path=todo_path,
            state_dir=state_dir,
            worktree_root=lane_worktree_root,
            state_prefix=state_prefix,
            task_prefix=task_prefix,
            implement=implement,
            daemon_interval=daemon_interval,
            stale_seconds=stale_seconds,
            check_interval=check_interval,
            max_restarts=max_restarts,
            implementation_timeout=implementation_timeout,
            implementation_command=implementation_command,
            llm_merge_resolver_command=llm_merge_resolver_command,
            llm_merge_resolver_timeout_seconds=llm_merge_resolver_timeout_seconds,
            merge_target_branch=merge_target_branch,
            merge_reconciliation_max_merges=merge_reconciliation_max_merges,
            generated_dirty_repair_enabled=generated_dirty_repair_enabled,
            generated_dirty_repair_commit_subject=generated_dirty_repair_commit_subject,
            generated_dirty_repair_include_submodule_gitlinks=generated_dirty_repair_include_submodule_gitlinks,
            generated_dirty_repair_max_paths=generated_dirty_repair_max_paths,
            generated_dirty_repair_stale_lock_seconds=generated_dirty_repair_stale_lock_seconds,
            generated_dirty_repair_paths=generated_dirty_repair_paths,
            worktree_submodule_paths=worktree_submodule_paths,
            assumed_completed_task_ids=assumed_completed_task_ids,
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
                **resource_fields,
            )
        )
    lanes.sort(key=_lane_schedule_key)
    return lanes[:max_lanes] if max_lanes is not None else lanes


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

    results: list[dict[str, Any]] = []
    active_lanes: list[BundleLaneSpec] = []
    path = coordination_path or default_state_root(repo_root) / "coordination.duckdb"
    with LeaseCoordinator(path) as coordinator:
        for lane in lanes:
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
            if not lane.queue_payload:
                results.append({"bundle_key": lane.bundle_key, "accepted": False, "error": "missing queue payload"})
                continue
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


def _lanes_conflict(left: BundleLaneSpec, right: BundleLaneSpec) -> bool:
    """Return whether graph edges prohibit two currently active lanes."""

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
    """Start one already-claimed lane without opening a second claim race."""

    lane.state_dir.mkdir(parents=True, exist_ok=True)
    lane.worktree_root.mkdir(parents=True, exist_ok=True)
    lane.log_path.parent.mkdir(parents=True, exist_ok=True)
    guarded_command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.leased_lane",
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
        "--",
        *lane.command,
    ]
    env = os.environ.copy()
    package_root = repo_root / "ipfs_datasets_py" / "ipfs_accelerate_py"
    if package_root.exists():
        env["PYTHONPATH"] = str(package_root) + os.pathsep + env.get("PYTHONPATH", "")
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
        "schema": "ipfs_accelerate_py.agent_supervisor.bundle_supervisor",
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
        external_task_state_paths: Sequence[Path | str] = (),
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
        self._launcher = launcher or self._default_launcher
        self._process_alive = process_alive or self._default_process_alive
        self._lane_disposition = lane_disposition or self._default_lane_disposition
        if resource_scheduler is not None:
            self.resource_scheduler = resource_scheduler
        else:
            policy_values = dict(resource_policy or {}) if isinstance(resource_policy, dict) else None
            if policy_values is not None:
                policy_values["max_lanes"] = self.max_lanes
                policy = ResourcePolicy.from_mapping(policy_values)
            elif isinstance(resource_policy, ResourcePolicy):
                policy = replace(resource_policy, max_lanes=self.max_lanes)
            else:
                policy = ResourcePolicy(max_lanes=self.max_lanes)
            self.resource_scheduler = ResourceScheduler(policy)
        self._host_resource_source = host_resource_source or sample_host_resources
        self._provider_capacity_source = provider_capacity_source
        self.provider_capacity_path = Path(provider_capacity_path).resolve() if provider_capacity_path else None
        self.external_task_state_paths = tuple(
            Path(path).resolve() for path in external_task_state_paths
        )
        self._running: dict[str, RunningBundleLane] = {}
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._cycle = 0
        self._last_discovery_error = ""
        self._last_coordination_compaction: dict[str, Any] = {}
        self._last_scheduler_snapshot: SchedulerSnapshot | None = None
        self._last_resource_snapshot: ResourceScheduleSnapshot | None = None
        self._event_source_cache: dict[Path, tuple[int, int, list[dict[str, Any]]]] = {}
        self._plan_cache: tuple[
            tuple[Path, ...],
            tuple[tuple[str, int, int], ...],
            tuple[BundleLaneSpec, ...],
        ] | None = None

    @property
    def running_count(self) -> int:
        return len(self._running)

    def _sample_host_resources(self) -> HostResourceSnapshot | dict[str, Any]:
        source = self._host_resource_source
        try:
            return source(
                self.state_root,
                active_workers=len(self._running),
                worker_limit=self.max_lanes,
                active_phase="scheduler",
            )
        except TypeError:
            return source()

    def _provider_capacities(self, coordinator: LeaseCoordinator) -> Any:
        """Read injected/file/fenced-heartbeat provider telemetry in that order."""

        if self._provider_capacity_source is not None:
            try:
                return self._provider_capacity_source()
            except TypeError:
                return self._provider_capacity_source(self)

        configured_path = self.provider_capacity_path
        if configured_path is None:
            env_path = os.environ.get("IPFS_ACCELERATE_LLM_ROUTER_CAPACITY_PATH", "").strip()
            if env_path:
                configured_path = Path(env_path)
        if configured_path is not None:
            try:
                payload = json.loads(configured_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("providers"), (dict, list)):
                    return payload["providers"]
                return payload
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read provider capacity %s: %s", configured_path, exc)
                return ()

        env_json = os.environ.get("IPFS_ACCELERATE_LLM_ROUTER_CAPACITY_JSON", "").strip()
        if env_json:
            try:
                payload = json.loads(env_json)
                return payload.get("providers", payload) if isinstance(payload, dict) else payload
            except json.JSONDecodeError as exc:
                logger.warning("Invalid IPFS_ACCELERATE_LLM_ROUTER_CAPACITY_JSON: %s", exc)
                return ()

        advertised: list[dict[str, Any]] = []
        for heartbeat in coordinator.latest_heartbeats():
            capacity = heartbeat.get("provider_capacity")
            if not isinstance(capacity, dict):
                continue
            item = dict(capacity)
            item.setdefault("provider_id", heartbeat.get("provider_id"))
            advertised.append(item)
        return advertised

    @staticmethod
    def _lane_resource_requirement(lane: BundleLaneSpec) -> LaneResourceRequirements:
        payload = lane.to_dict()
        payload.update(
            {
                "lane_id": lane.task_cid or lane.bundle_key,
                "provider_id": lane.llm_provider,
                "context_tokens": lane.required_context_tokens,
                "max_provider_latency_ms": lane.max_provider_latency_ms,
            }
        )
        return LaneResourceRequirements.from_mapping(payload)

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
        if self._plan_cache is not None:
            cached_paths, cached_revision, cached_lanes = self._plan_cache
            if self._plan_source_revision(cached_paths) == cached_revision:
                base_lanes = list(cached_lanes)
            else:
                self._plan_cache = None
        if self._plan_cache is None:
            allowed = {
                "task_prefix", "implement", "daemon_interval", "stale_seconds",
                "check_interval", "max_restarts", "implementation_timeout",
                "implementation_command", "llm_merge_resolver_command",
                "llm_merge_resolver_timeout_seconds", "merge_target_branch",
                "merge_reconciliation_max_merges",
                "generated_dirty_repair_enabled", "generated_dirty_repair_commit_subject",
                "generated_dirty_repair_include_submodule_gitlinks",
                "generated_dirty_repair_max_paths", "generated_dirty_repair_stale_lock_seconds",
                "generated_dirty_repair_paths",
                "worktree_submodule_paths", "log_level",
            }
            options = {key: value for key, value in self.lane_options.items() if key in allowed}
            base_lanes = plan_bundle_lanes(
                bundle_index_path=self.bundle_index_path,
                repo_root=self.repo_root,
                state_root=self.state_root,
                worktree_root=self.worktree_root,
                log_dir=self.log_dir,
                max_lanes=None,
                **options,
            )
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
                tuple(base_lanes),
            )

        external_active_task_ids = self._external_active_task_ids()
        if not external_active_task_ids:
            return base_lanes
        return [
            self._fence_external_active_members(lane, external_active_task_ids)
            for lane in base_lanes
        ]

    @staticmethod
    def _default_process_alive(handle: Any) -> bool:
        poll = getattr(handle, "poll", None)
        if callable(poll):
            return poll() is None
        return bool(getattr(handle, "alive", False))

    def _default_lane_disposition(self, lane: BundleLaneSpec) -> str:
        """Project a settled execution slice or shard board to a disposition."""

        try:
            markdown = lane.todo_path.read_text(encoding="utf-8")
        except OSError:
            markdown = ""
        from .todo_daemon.implementation_daemon import parse_task_file

        task_prefix = str(self.lane_options.get("task_prefix") or DEFAULT_TASK_PREFIX)
        portal_tasks = parse_task_file(lane.todo_path, task_prefix) if markdown else []
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
                board_mtime_ns = lane.todo_path.stat().st_mtime_ns
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

        from .todo_daemon.engine import parse_markdown_tasks

        tasks = parse_markdown_tasks(markdown)
        if not tasks:
            return ""
        if all(task.status == "complete" for task in tasks):
            return "completed"
        if all(task.status in {"complete", "blocked"} for task in tasks):
            return "blocked"
        return ""

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

    def _reap(self, coordinator: LeaseCoordinator) -> list[str]:
        reaped: list[str] = []
        for task_cid, running in list(self._running.items()):
            try:
                alive = bool(self._process_alive(running.handle))
            except (OSError, RuntimeError):
                alive = False
            disposition = self._disposition(running.spec) if alive else ""
            if alive and not disposition:
                continue
            if disposition:
                try:
                    self._settle_grant(coordinator, running.grant, disposition=disposition)
                except LeaseError:
                    pass
            self._terminate_handle(running.handle)
            # The leased-lane wrapper normally publishes a receipt first.  A
            # crashed wrapper does not, so explicitly release its still-current
            # grant and make the lane immediately reclaimable.
            try:
                if coordinator.active_lease(task_cid) is not None:
                    coordinator.release(running.grant, reason="worker drained or exited")
            except LeaseError:
                pass
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
            if state == "ready" and lane is not None and not lane.claimable:
                state = "blocked"
                item.setdefault("blocked_reason", "planner_not_claimable")
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
        return scheduler_snapshot(events, now=projection_timestamp)

    def _write_live_manifest(
        self,
        *,
        discovered: Sequence[BundleLaneSpec],
        task_projection: Sequence[dict[str, Any]],
        launched: Sequence[str],
        reaped: Sequence[str],
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
            if detailed["state"] == "ready" and lane is not None and not lane.claimable:
                detailed["state"] = "blocked"
                detailed.setdefault("blocked_reason", "planner_not_claimable")
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
                discovered = self._plan()
                self._last_discovery_error = ""
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                discovered = []
                self._last_discovery_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Bundle discovery failed; retaining live lanes: %s", exc)

            launched: list[str] = []
            with LeaseCoordinator(self.coordination_path) as coordinator:
                registered: list[BundleLaneSpec] = []
                registration_lanes = [
                    lane for lane in discovered if lane.queue_payload
                ]
                adapted_bundles = coordinator.register_bundles(
                    lane.queue_payload for lane in registration_lanes
                )
                for lane, adapted in zip(
                    registration_lanes,
                    adapted_bundles,
                ):
                    registered.append(
                        replace(
                            lane,
                            task_cid=str(adapted["task_cid"]),
                            goal_cid=str(adapted["goal_cid"]),
                            subgoal_cid=str(adapted["subgoal_cid"]),
                        )
                    )

                reaped = self._reap(coordinator)
                for lane in registered:
                    if lane.task_cid in self._running or self._disposition(lane):
                        continue
                    coordinator.requeue_exhausted_blocked(
                        lane.task_cid,
                        reason="bundle_board_reopened",
                    )
                current_task_cids = {
                    *(lane.task_cid for lane in registered),
                    *self._running.keys(),
                }
                decision_projection = coordinator.list_tasks(
                    task_cids=current_task_cids,
                    include_claimability=True,
                )
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
                        or registered_by_task_cid[str(item.get("task_cid") or "")].claimable
                    )
                }
                decisions: list[dict[str, Any]] = []
                running_by_bundle_key = {
                    running.spec.bundle_key: running
                    for running in self._running.values()
                }
                dispositions = {
                    lane.task_cid: self._disposition(lane)
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
                    and not any(
                        _lanes_conflict(lane, running.spec)
                        for running in self._running.values()
                    )
                ]
                try:
                    host_resources = self._sample_host_resources()
                except Exception:
                    logger.exception("Host resource sampling failed; retaining configured bounds")
                    host_resources = HostResourceSnapshot(
                        active_workers=len(self._running),
                        worker_limit=self.max_lanes,
                        available_worker_capacity=max(0, self.max_lanes - len(self._running)),
                    )
                try:
                    provider_capacities = self._provider_capacities(coordinator)
                except Exception:
                    logger.exception("Provider capacity sampling failed")
                    provider_capacities = ()
                resource_requirements = {
                    lane.task_cid: self._lane_resource_requirement(lane)
                    for lane in resource_candidates
                }
                confirmed_requirements: list[LaneResourceRequirements] = []
                resource_cycle_decisions: list[AdmissionDecision] = []
                for lane in registered:
                    if lane.task_cid in self._running:
                        decisions.append({
                            "task_cid": lane.task_cid,
                            "bundle_key": lane.bundle_key,
                            "decision": "retained",
                            "reason": "already_active",
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
                    if requirement is None:
                        admission = None
                    else:
                        candidate_schedule = self.resource_scheduler.schedule(
                            [*confirmed_requirements, requirement],
                            host=host_resources,
                            providers=provider_capacities,
                            path=self.state_root,
                            active_workers=len(self._running),
                        )
                        admission = candidate_schedule.decisions[-1]
                        resource_cycle_decisions.append(admission)
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
                    grant = coordinator.claim_ready(
                        self.claimant_did,
                        requested_lease_ms=self.lease_ms,
                        eligible_task_cids=(lane.task_cid,),
                    )
                    if grant is None:
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

                resource_snapshot = self.resource_scheduler.schedule(
                    confirmed_requirements,
                    host=host_resources,
                    providers=provider_capacities,
                    path=self.state_root,
                    active_workers=len(self._running) - len(confirmed_requirements),
                )
                resource_backpressure = tuple(
                    dict.fromkeys(
                        reason
                        for decision in resource_cycle_decisions
                        if not decision.admitted
                        for reason in decision.reasons
                    )
                )
                self._last_resource_snapshot = replace(
                    resource_snapshot,
                    decisions=tuple(resource_cycle_decisions),
                    admitted_count=len(confirmed_requirements),
                    backpressure_reasons=resource_backpressure,
                )
                current_task_cids = {
                    *(lane.task_cid for lane in registered),
                    *self._running.keys(),
                }
                projection = coordinator.list_tasks(
                    task_cids=current_task_cids,
                    include_claimability=True,
                )
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
        except BaseException:
            self._stop_event.set()
            raise
        finally:
            if install_handlers:
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)
            if self._stop_event.is_set() or (external_stop and external_stop.is_set()):
                payload = self.stop()
        return payload

    def stop(self, *, grace_seconds: float = 5.0) -> dict[str, Any]:
        """Stop owned processes and release only leases still fenced to them."""

        self._stop_event.set()
        with self._lock, LeaseCoordinator(self.coordination_path) as coordinator:
            for task_cid, running in list(self._running.items()):
                self._terminate_handle(running.handle, grace_seconds=grace_seconds)
                try:
                    if coordinator.active_lease(task_cid) is not None:
                        coordinator.release(running.grant, reason="scheduler stopped")
                except LeaseError:
                    pass
            self._running.clear()
            projection = coordinator.list_tasks()
        try:
            discovered = self._plan()
        except (OSError, ValueError, json.JSONDecodeError):
            discovered = []
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
    parser.add_argument("--start", action="store_true", help="Launch the planned lane supervisors")
    parser.add_argument("--max-lanes", type=int, default=1, help="Maximum concurrent leased workers")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="Run one reconciliation cycle and exit")
    implement_group = parser.add_mutually_exclusive_group()
    implement_group.add_argument("--implement", dest="implement", action="store_true")
    implement_group.add_argument("--no-implement", dest="implement", action="store_false")
    parser.set_defaults(implement=False)
    parser.add_argument("--daemon-interval", type=float, default=300.0)
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--check-interval", type=float, default=60.0)
    parser.add_argument("--max-restarts", type=int, default=0)
    parser.add_argument("--implementation-timeout", type=float, default=1800.0)
    parser.add_argument("--implementation-command", default="")
    parser.add_argument("--llm-merge-resolver-command", default="")
    parser.add_argument("--llm-merge-resolver-timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--merge-target-branch",
        default="",
        help=(
            "Branch that receives isolated implementation merges. The branch "
            "is forwarded to every bundle lane."
        ),
    )
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
    parser.add_argument("--minimum-memory-available-bytes", type=int, default=0)
    parser.add_argument("--minimum-disk-available-bytes", type=int, default=0)
    parser.add_argument("--maximum-provider-latency-ms", type=int, default=120_000)
    parser.add_argument("--provider-quota-reserve", type=int, default=0)
    parser.add_argument("--provider-token-reserve", type=int, default=0)
    parser.add_argument("--provider-capacity-path", type=Path, default=None)
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
    lane_options = dict(
        task_prefix=args.task_prefix,
        implement=args.implement,
        daemon_interval=args.daemon_interval,
        stale_seconds=args.stale_seconds,
        check_interval=args.check_interval,
        max_restarts=args.max_restarts,
        implementation_timeout=args.implementation_timeout,
        implementation_command=args.implementation_command,
        llm_merge_resolver_command=args.llm_merge_resolver_command,
        llm_merge_resolver_timeout_seconds=args.llm_merge_resolver_timeout_seconds,
        merge_target_branch=getattr(args, "merge_target_branch", ""),
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
            external_task_state_paths=tuple(
                getattr(args, "external_task_state_path", ()) or ()
            ),
            resource_policy={
                "max_lanes": getattr(args, "max_lanes", 1) or 1,
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
