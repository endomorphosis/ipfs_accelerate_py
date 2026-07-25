"""Deterministic, conflict-aware optimization of already-admitted tasks.

Task admission and bundle planning are deliberately separate boundaries.
``task_quality`` decides whether work is coherent enough to enter the board;
this module accepts canonical admitted tasks and chooses execution bundles.  It
never splits, coalesces, or semantically deduplicates tasks.

The optimizer preserves dependency width, reuses context and validation work,
separates conflicting edits, and carries canonical task identities through
every result projection.  Packet completion is an independent fail-closed
operation: packet, cluster, context, and merge-family membership are scheduling
hints, never completion authority.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Final

from .conflict_graph import (
    ConflictEdge,
    ConflictWaveProjection,
    TaskConflictGraph,
    materialize_task_conflict_graph,
    project_conflict_free_wave,
)
from .task_identity import canonical_content_cid, canonical_json_bytes


PACKET_COMPLETION_BINDING_REQUIREMENT_ID: Final = (
    "187052702852200236079602798955260586139"
)
CRITICAL_PATH_WIDTH_REQUIREMENT_ID: Final = (
    "061582446926920746660485801841658333166"
)
BUNDLE_OPTIMIZER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/bundle-optimization@1"
)
PACKET_COMPLETION_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/packet-completion-binding-evidence@1"
)
CRITICAL_PATH_WIDTH_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/critical-path-width-evidence@1"
)
_PACKET_COMPLETION_EVIDENCE_SEAL: Final = object()
_CRITICAL_PATH_WIDTH_EVIDENCE_SEAL: Final = object()


def _strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = value.split(",")
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        values = value
    else:
        values = (value,)
    return tuple(
        sorted(
            {
                " ".join(str(item).strip().split())
                for item in values
                if str(item).strip()
            }
        )
    )


def _paths(value: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                path
                for item in _strings(value)
                for path in [item.replace("\\", "/").removeprefix("./").rstrip("/")]
                if path
            }
        )
    )


def _value(source: Mapping[str, Any], *names: str, default: Any = "") -> Any:
    normalized = {
        str(key).strip().casefold().replace("_", " "): value
        for key, value in source.items()
    }
    for name in names:
        value = normalized.get(name.casefold().replace("_", " "))
        if value not in (None, "", [], ()):
            return value
    return default


def _task_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError("bundle tasks must be mappings or expose to_dict()")


@dataclass(frozen=True)
class BundleOptimizationPolicy:
    """Bounded deterministic bundle-planning policy."""

    max_tasks_per_bundle: int = 4
    max_context_tokens_per_bundle: int = 32_768
    allow_internal_conflicts: bool = False
    require_affinity: bool = True
    context_weight: int = 5
    validation_weight: int = 4
    merge_locality_weight: int = 3
    goal_weight: int = 2
    evidence_weight: int = 4
    resource_weight: int = 1
    provider_batch_weight: int = 4
    packet_weight: int = 6
    require_resource_compatibility: bool = True
    require_provider_compatibility: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_tasks_per_bundle",
            "max_context_tokens_per_bundle",
            "context_weight",
            "validation_weight",
            "merge_locality_weight",
            "goal_weight",
            "evidence_weight",
            "resource_weight",
            "provider_batch_weight",
            "packet_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_tasks_per_bundle < 1:
            raise ValueError("max_tasks_per_bundle must be positive")

    @property
    def policy_id(self) -> str:
        digest = hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()
        return f"bundle-optimization-policy/v1/{digest}"


@dataclass(frozen=True)
class _CanonicalTask:
    task_id: str
    canonical_task_key: str
    canonical_task_cid: str
    semantic_identity: str
    goal_id: str
    dependency_depth: int
    goal_packet_key: str
    goal_packet_role: str
    merge_family: str
    merge_fate: str
    context_paths: tuple[str, ...]
    evidence_keys: tuple[str, ...]
    validation_commands: tuple[str, ...]
    outputs: tuple[str, ...]
    predicted_paths: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    dependencies: tuple[str, ...]
    conflict_keys: tuple[str, ...]
    resource_class: str
    provider_batch_key: str
    estimated_context_tokens: int
    work_item_count: int
    status: str
    completion_task_bindings: tuple[str, ...]
    payload: Mapping[str, Any] = field(compare=False, repr=False)

    @classmethod
    def from_value(cls, value: Any) -> "_CanonicalTask":
        payload = _task_mapping(value)
        task_id = str(_value(payload, "task id", "display task id") or "").strip()
        task_key = str(_value(payload, "canonical task key") or "").strip()
        task_cid = str(
            _value(payload, "canonical task cid", "task cid") or ""
        ).strip()
        if not task_key or not task_cid:
            raise ValueError(
                "bundle optimization requires canonical_task_key and "
                "canonical_task_cid from task admission"
            )
        try:
            context_tokens = int(
                _value(payload, "estimated context tokens", default=0) or 0
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("estimated_context_tokens must be an integer") from exc
        if context_tokens < 0:
            raise ValueError("estimated_context_tokens must be non-negative")
        try:
            dependency_depth = int(
                _value(payload, "dependency depth", "graph depth", default=0) or 0
            )
            work_item_count = int(_value(payload, "work item count", default=1) or 1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "dependency_depth and work_item_count must be integers"
            ) from exc
        if dependency_depth < 0 or work_item_count < 1:
            raise ValueError(
                "dependency_depth must be non-negative and work_item_count positive"
            )
        outputs = _paths(_value(payload, "outputs", "files"))
        predicted_paths = _paths(
            _value(payload, "predicted paths", "predicted files")
        )
        normalized_payload = dict(payload)
        normalized_payload.update(
            {
                "task_id": task_id,
                "canonical_task_key": task_key,
                "canonical_task_cid": task_cid,
                "task_cid": task_cid,
                "outputs": list(outputs),
                "predicted_paths": list(predicted_paths),
            }
        )
        return cls(
            task_id=task_id,
            canonical_task_key=task_key,
            canonical_task_cid=task_cid,
            semantic_identity=str(
                _value(
                    payload,
                    "canonical semantic identity",
                    "semantic identity",
                )
                or ""
            ).strip(),
            goal_id=str(_value(payload, "goal id") or "").strip(),
            dependency_depth=dependency_depth,
            goal_packet_key=str(_value(payload, "goal packet key", "goal packet") or "").strip(),
            goal_packet_role=str(_value(payload, "goal packet role") or "").strip(),
            merge_family=str(_value(payload, "merge family") or "").strip(),
            merge_fate=str(_value(payload, "merge fate") or "").strip(),
            context_paths=_paths(
                _value(payload, "context paths", "context keys", "context files")
            ),
            evidence_keys=_strings(
                _value(
                    payload,
                    "evidence subset",
                    "evidence keys",
                    "missing evidence",
                    "provenance cids",
                )
            ),
            validation_commands=_strings(
                _value(payload, "validation commands", "validation")
            ),
            outputs=outputs,
            predicted_paths=predicted_paths,
            predicted_symbols=_strings(
                _value(payload, "predicted symbols", "ast symbols")
            ),
            dependencies=_strings(
                _value(
                    payload,
                    "dependency task cids",
                    "dependencies",
                    "depends on",
                    "graph parents",
                )
            ),
            conflict_keys=_strings(_value(payload, "conflicts", "conflict keys")),
            resource_class=str(_value(payload, "resource class") or "").strip(),
            provider_batch_key=_provider_batch_key(payload),
            estimated_context_tokens=context_tokens,
            work_item_count=work_item_count,
            status=str(_value(payload, "status") or "").strip().casefold(),
            completion_task_bindings=_strings(
                _value(payload, "completion task bindings")
            ),
            payload=normalized_payload,
        )

    @property
    def is_packet_aggregate(self) -> bool:
        return self.goal_packet_role.casefold() == "packet_aggregate" or str(
            _value(self.payload, "candidate kind", "merge role") or ""
        ).casefold() in {"goal_packet_aggregate", "packet_aggregate"}

    @property
    def is_completed(self) -> bool:
        return self.status in {
            "complete",
            "completed",
            "done",
            "merged",
            "success",
            "succeeded",
        }


def _provider_batch_key(payload: Mapping[str, Any]) -> str:
    """Return a complete provider compatibility key, or an explicit key.

    Provider batching is safe only when every compatibility dimension matches.
    A partial provider description is still useful as a separation boundary,
    so missing dimensions are represented explicitly instead of being dropped.
    """

    explicit = _value(
        payload,
        "provider batch key",
        "provider compatibility key",
        "batch compatibility digest",
    )
    if explicit not in (None, "", [], ()):
        if isinstance(explicit, Mapping):
            return hashlib.sha256(canonical_json_bytes(dict(explicit))).hexdigest()
        return str(explicit).strip()
    provider_id = str(
        _value(payload, "provider id", "llm provider", "provider") or ""
    ).strip()
    if not provider_id:
        return ""
    material = {
        "provider_id": provider_id,
        "route": str(_value(payload, "provider route", "route") or ""),
        "model": str(_value(payload, "model id", "model") or ""),
        "operation": str(
            _value(payload, "provider operation", "operation id", "operation")
            or ""
        ),
        "context_limit": str(
            _value(payload, "provider context limit", "context limit") or ""
        ),
        "policy_digest": str(
            _value(payload, "provider policy digest", "policy digest") or ""
        ),
        "generation_digest": str(
            _value(
                payload,
                "provider generation digest",
                "generation digest",
            )
            or ""
        ),
    }
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def _canonical_tasks(values: Any) -> tuple[_CanonicalTask, ...]:
    # The admission result is accepted explicitly; its rejected candidates can
    # never leak into optimization.
    if hasattr(values, "accepted") and not isinstance(values, (list, tuple)):
        values = values.accepted
    tasks = tuple(_CanonicalTask.from_value(value) for value in values)
    cids = [task.canonical_task_cid for task in tasks]
    keys = [task.canonical_task_key for task in tasks]
    if len(cids) != len(set(cids)):
        raise ValueError("canonical_task_cid values must be unique")
    if len(keys) != len(set(keys)):
        raise ValueError("canonical_task_key values must be unique")
    return tuple(sorted(tasks, key=lambda item: item.canonical_task_cid))


def _resolve_dependencies(
    tasks: Sequence[_CanonicalTask],
) -> dict[str, set[str]]:
    by_cid = {task.canonical_task_cid: task for task in tasks}
    aliases = {
        alias: task.canonical_task_cid
        for task in tasks
        for alias in (
            task.task_id,
            task.canonical_task_key,
            task.canonical_task_cid,
        )
        if alias
    }
    dependencies: dict[str, set[str]] = {}
    for task in tasks:
        resolved = {
            aliases[value]
            for value in task.dependencies
            if value in aliases and aliases[value] != task.canonical_task_cid
        }
        dependencies[task.canonical_task_cid] = resolved & set(by_cid)
    return dependencies


def _dependency_waves(
    tasks: Sequence[_CanonicalTask],
) -> tuple[dict[str, int], dict[str, set[str]]]:
    dependencies = _resolve_dependencies(tasks)
    minimum_waves = {
        task.canonical_task_cid: task.dependency_depth for task in tasks
    }
    return _waves_from_dependencies(
        dependencies, minimum_waves=minimum_waves
    ), dependencies


def _waves_from_dependencies(
    dependencies: Mapping[str, set[str]],
    *,
    minimum_waves: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Return longest-path waves for one finite acyclic prerequisite graph."""

    waves: dict[str, int] = {}
    remaining = set(dependencies)
    while remaining:
        ready = sorted(
            cid for cid in remaining if dependencies[cid].issubset(waves)
        )
        if not ready:
            raise ValueError(
                "bundle optimization requires an acyclic canonical dependency graph"
            )
        for cid in ready:
            waves[cid] = max(
                int((minimum_waves or {}).get(cid, 0)),
                max((waves[parent] + 1 for parent in dependencies[cid]), default=0),
            )
            remaining.remove(cid)
    return waves


def _conflict_pairs(
    graph: TaskConflictGraph,
    tasks: Sequence[_CanonicalTask],
) -> tuple[set[frozenset[str]], dict[frozenset[str], ConflictEdge]]:
    edges = {
        frozenset((edge.left_task_cid, edge.right_task_cid)): edge
        for edge in graph.edges
        if edge.blocks_concurrency
    }
    # Explicit conflict keys are a supervisor contract even if they do not map
    # to a file or AST surface.
    for index, left in enumerate(tasks):
        for right in tasks[index + 1 :]:
            shared = set(left.conflict_keys) & set(right.conflict_keys)
            if not shared:
                continue
            pair = frozenset((left.canonical_task_cid, right.canonical_task_cid))
            edges.setdefault(
                pair,
                ConflictEdge(
                    left_task_cid=min(pair),
                    right_task_cid=max(pair),
                    weight=float(len(shared)),
                    reasons=[f"conflict_key: {item}" for item in sorted(shared)],
                    overlaps={"conflict_keys": sorted(shared)},
                ),
            )
    return set(edges), edges


def _affinity(
    left: _CanonicalTask,
    right: _CanonicalTask,
    policy: BundleOptimizationPolicy,
) -> int:
    context = len(set(left.context_paths) & set(right.context_paths))
    evidence = len(set(left.evidence_keys) & set(right.evidence_keys))
    validation = len(
        set(left.validation_commands) & set(right.validation_commands)
    )
    merge = int(
        bool(left.merge_family or left.merge_fate)
        and (left.merge_family or left.merge_fate)
        == (right.merge_family or right.merge_fate)
    )
    goal = int(bool(left.goal_id) and left.goal_id == right.goal_id)
    resource = int(
        bool(left.resource_class) and left.resource_class == right.resource_class
    )
    provider = int(
        bool(left.provider_batch_key)
        and left.provider_batch_key == right.provider_batch_key
    )
    packet = int(
        bool(left.goal_packet_key)
        and left.goal_packet_key == right.goal_packet_key
        and (
            left.is_packet_aggregate
            or right.is_packet_aggregate
            or left.canonical_task_cid in right.completion_task_bindings
            or right.canonical_task_cid in left.completion_task_bindings
        )
    )
    return (
        policy.context_weight * context
        + policy.evidence_weight * evidence
        + policy.validation_weight * validation
        + policy.merge_locality_weight * merge
        + policy.goal_weight * goal
        + policy.resource_weight * resource
        + policy.provider_batch_weight * provider
        + policy.packet_weight * packet
    )


def _compatible(
    left: _CanonicalTask,
    right: _CanonicalTask,
    policy: BundleOptimizationPolicy,
) -> bool:
    if (
        policy.require_resource_compatibility
        and left.resource_class != right.resource_class
        and (left.resource_class or right.resource_class)
    ):
        return False
    if (
        policy.require_provider_compatibility
        and left.provider_batch_key != right.provider_batch_key
        and (left.provider_batch_key or right.provider_batch_key)
    ):
        return False
    return True


def _conflict_color_order(
    task_cids: Iterable[str],
    conflict_pairs: set[frozenset[str]],
) -> dict[str, int]:
    """Color one independent dependency wave without destroying graph width."""

    nodes = set(task_cids)
    pairs = (
        tuple(sorted(pair))
        for pair in conflict_pairs
        if pair.issubset(nodes)
    )
    return dict(
        project_conflict_free_wave(nodes, pairs).color_by_task_cid
    )


@dataclass(frozen=True)
class PacketAggregateProjection:
    """Canonical packet aggregate and the exact siblings it covers."""

    aggregate_task_cid: str
    aggregate_canonical_task_key: str
    goal_packet_key: str
    covered_sibling_task_cids: tuple[str, ...]
    covered_sibling_canonical_task_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_task_cid": self.aggregate_task_cid,
            "aggregate_canonical_task_key": self.aggregate_canonical_task_key,
            "goal_packet_key": self.goal_packet_key,
            "covered_sibling_task_cids": list(self.covered_sibling_task_cids),
            "covered_sibling_canonical_task_keys": list(
                self.covered_sibling_canonical_task_keys
            ),
        }


@dataclass(frozen=True)
class OptimizedTaskBundle:
    """One identity-preserving execution bundle."""

    bundle_cid: str
    task_cids: tuple[str, ...]
    canonical_task_keys: tuple[str, ...]
    display_task_ids: tuple[str, ...]
    execution_wave: int
    goal_ids: tuple[str, ...]
    validation_commands: tuple[str, ...]
    shared_validation_commands: tuple[str, ...]
    context_paths: tuple[str, ...]
    shared_context_paths: tuple[str, ...]
    evidence_keys: tuple[str, ...]
    shared_evidence_keys: tuple[str, ...]
    resource_classes: tuple[str, ...]
    provider_batch_keys: tuple[str, ...]
    merge_localities: tuple[str, ...]
    packet_aggregate_task_cids: tuple[str, ...]
    covered_sibling_task_cids: tuple[str, ...]
    dependency_task_cids: tuple[str, ...]
    conflict_weight: int
    estimated_context_tokens: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "task_cids",
            "canonical_task_keys",
            "display_task_ids",
            "goal_ids",
            "validation_commands",
            "shared_validation_commands",
            "context_paths",
            "shared_context_paths",
            "evidence_keys",
            "shared_evidence_keys",
            "resource_classes",
            "provider_batch_keys",
            "merge_localities",
            "packet_aggregate_task_cids",
            "covered_sibling_task_cids",
            "dependency_task_cids",
        ):
            result[name] = list(result[name])
        return result


@dataclass(frozen=True)
class BundlePlanComparison:
    """Paired quality/cost metrics for current and optimized planners."""

    current_metrics: Mapping[str, int]
    optimized_metrics: Mapping[str, int]
    deltas: Mapping[str, int]
    improvements: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_planner": dict(sorted(self.current_metrics.items())),
            "optimized_planner": dict(sorted(self.optimized_metrics.items())),
            "deltas": dict(sorted(self.deltas.items())),
            "improvements": dict(sorted(self.improvements.items())),
        }


@dataclass(frozen=True)
class BundleOptimizationResult:
    policy_id: str
    bundles: tuple[OptimizedTaskBundle, ...]
    task_count: int
    execution_width_by_wave: Mapping[int, int]
    metrics: Mapping[str, int]
    packet_aggregates: tuple[PacketAggregateProjection, ...]
    comparison: BundlePlanComparison
    conflict_graph: Mapping[str, Any] = field(compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BUNDLE_OPTIMIZER_SCHEMA,
            "policy_id": self.policy_id,
            "task_count": self.task_count,
            "bundle_count": len(self.bundles),
            "bundles": [bundle.to_dict() for bundle in self.bundles],
            "execution_width_by_wave": {
                str(key): value
                for key, value in sorted(self.execution_width_by_wave.items())
            },
            "metrics": dict(sorted(self.metrics.items())),
            "packet_aggregates": [
                aggregate.to_dict() for aggregate in self.packet_aggregates
            ],
            "comparison": self.comparison.to_dict(),
            "conflict_graph": dict(self.conflict_graph),
        }


_LOWER_IS_BETTER_METRICS: Final = frozenset(
    {
        "model_call_count",
        "model_calls_per_work_item_millionths",
        "critical_path_wave_count",
        "merge_conflict_rate_millionths",
        "blocking_conflict_count",
        "internal_blocking_conflict_count",
    }
)


def compare_bundle_plan_metrics(
    current_metrics: Mapping[str, int],
    optimized_metrics: Mapping[str, int],
) -> BundlePlanComparison:
    """Compare paired planner metrics using explicit metric direction."""

    keys = sorted(set(current_metrics) | set(optimized_metrics))
    current = {key: int(current_metrics.get(key, 0)) for key in keys}
    optimized = {key: int(optimized_metrics.get(key, 0)) for key in keys}
    deltas = {key: optimized[key] - current[key] for key in keys}
    improvements = {
        key: (
            current[key] - optimized[key]
            if key in _LOWER_IS_BETTER_METRICS
            else optimized[key] - current[key]
        )
        for key in keys
    }
    return BundlePlanComparison(
        current_metrics=current,
        optimized_metrics=optimized,
        deltas=deltas,
        improvements=improvements,
    )


def _packet_aggregate_projections(
    tasks: Sequence[_CanonicalTask],
) -> tuple[PacketAggregateProjection, ...]:
    if not any(task.is_packet_aggregate for task in tasks):
        return ()
    by_cid = {task.canonical_task_cid: task for task in tasks}
    alias_owners: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        for alias in (
            task.canonical_task_cid,
            task.canonical_task_key,
            task.semantic_identity,
        ):
            if alias:
                alias_owners[alias].add(task.canonical_task_cid)
    ambiguous = sorted(
        alias for alias, owners in alias_owners.items() if len(owners) != 1
    )
    if ambiguous:
        raise ValueError(
            "packet aggregate identity aliases must be globally unique: "
            + ", ".join(ambiguous)
        )
    aliases = {
        alias: next(iter(owners)) for alias, owners in alias_owners.items()
    }
    projections: list[PacketAggregateProjection] = []
    for aggregate in tasks:
        if not aggregate.is_packet_aggregate:
            continue
        unresolved = sorted(
            binding
            for binding in aggregate.completion_task_bindings
            if binding not in aliases
        )
        if unresolved:
            raise ValueError(
                "packet aggregate completion bindings reference unknown "
                f"canonical tasks: {', '.join(unresolved)}"
            )
        covered_cids = tuple(
            sorted(
                {
                    aliases[binding]
                    for binding in aggregate.completion_task_bindings
                    if aliases[binding] != aggregate.canonical_task_cid
                }
            )
        )
        if any(
            not aggregate.goal_packet_key
            or by_cid[cid].goal_packet_key != aggregate.goal_packet_key
            for cid in covered_cids
        ):
            raise ValueError(
                "packet aggregate completion bindings must remain in one "
                "explicit goal_packet_key"
            )
        projections.append(
            PacketAggregateProjection(
                aggregate_task_cid=aggregate.canonical_task_cid,
                aggregate_canonical_task_key=aggregate.canonical_task_key,
                goal_packet_key=aggregate.goal_packet_key,
                covered_sibling_task_cids=covered_cids,
                covered_sibling_canonical_task_keys=tuple(
                    by_cid[cid].canonical_task_key for cid in covered_cids
                ),
            )
        )
    return tuple(
        sorted(projections, key=lambda item: item.aggregate_task_cid)
    )


def _plan_metrics(
    *,
    tasks: Sequence[_CanonicalTask],
    groups: Sequence[tuple[int, Sequence[_CanonicalTask]]],
    conflict_pairs: set[frozenset[str]],
) -> dict[str, int]:
    raw_context_references = sum(len(task.context_paths) for task in tasks)
    materialized_context_references = sum(
        len({path for task in group for path in task.context_paths})
        for _, group in groups
    )
    raw_validation_references = sum(
        len(task.validation_commands) for task in tasks
    )
    materialized_validation_references = sum(
        len(
            {
                command
                for task in group
                for command in task.validation_commands
            }
        )
        for _, group in groups
    )
    task_wave = {
        task.canonical_task_cid: wave for wave, group in groups for task in group
    }
    concurrent_conflicts = sum(
        1
        for pair in conflict_pairs
        if len({task_wave[cid] for cid in pair if cid in task_wave}) == 1
    )
    completed_bundles = sum(
        1 for _, group in groups if group and all(task.is_completed for task in group)
    )
    task_aliases = {
        alias: task.canonical_task_cid
        for task in tasks
        for alias in (
            task.canonical_task_cid,
            task.canonical_task_key,
            task.semantic_identity,
        )
        if alias
    }
    covered_sibling_cids = {
        task_aliases[binding]
        for task in tasks
        if task.is_packet_aggregate
        for binding in task.completion_task_bindings
        if binding in task_aliases
        and task_aliases[binding] != task.canonical_task_cid
    }
    accepted_work_items = sum(
        task.work_item_count
        for task in tasks
        if task.canonical_task_cid not in covered_sibling_cids
    )
    bundle_count = len(groups)
    return {
        "accepted_task_count": len(tasks),
        "accepted_work_item_count": accepted_work_items,
        "covered_sibling_task_count": len(covered_sibling_cids),
        "model_call_count": bundle_count,
        "model_calls_per_work_item_millionths": (
            bundle_count * 1_000_000 // accepted_work_items
            if accepted_work_items
            else 0
        ),
        "context_reuse_millionths": (
            (raw_context_references - materialized_context_references)
            * 1_000_000
            // raw_context_references
            if raw_context_references
            else 0
        ),
        "validation_reuse_millionths": (
            (raw_validation_references - materialized_validation_references)
            * 1_000_000
            // raw_validation_references
            if raw_validation_references
            else 0
        ),
        "critical_path_wave_count": max(task_wave.values(), default=-1) + 1,
        "blocking_conflict_count": len(conflict_pairs),
        "internal_blocking_conflict_count": sum(
            1
            for _, group in groups
            for pair in conflict_pairs
            if pair.issubset(
                {task.canonical_task_cid for task in group}
            )
        ),
        "merge_conflict_rate_millionths": (
            concurrent_conflicts * 1_000_000 // len(conflict_pairs)
            if conflict_pairs
            else 0
        ),
        "completed_bundle_count": completed_bundles,
        "bundle_completion_millionths": (
            completed_bundles * 1_000_000 // bundle_count
            if bundle_count
            else 0
        ),
    }


def optimize_task_bundles(
    admitted_tasks: Any,
    *,
    policy: BundleOptimizationPolicy | None = None,
    current_planner_bundles: Sequence[Any] | None = None,
) -> BundleOptimizationResult:
    """Optimize canonical admitted tasks without changing task admission.

    ``current_planner_bundles`` optionally supplies the exact baseline grouping
    as mappings/objects with ``task_cids`` and ``execution_wave`` fields (or as
    sequences of canonical task CIDs).  When omitted, the legacy one-task-call
    plan is used.
    """

    selected = policy or BundleOptimizationPolicy()
    tasks = _canonical_tasks(admitted_tasks)
    if not tasks:
        metrics = _plan_metrics(tasks=(), groups=(), conflict_pairs=set())
        comparison = compare_bundle_plan_metrics(metrics, metrics)
        return BundleOptimizationResult(
            policy_id=selected.policy_id,
            bundles=(),
            task_count=0,
            execution_width_by_wave={},
            metrics=metrics,
            packet_aggregates=(),
            comparison=comparison,
            conflict_graph={},
        )
    packet_aggregates = _packet_aggregate_projections(tasks)
    waves, dependencies = _dependency_waves(tasks)
    graph_tasks: list[dict[str, Any]] = []
    for task in tasks:
        graph_task = dict(task.payload)
        if task.conflict_keys:
            graph_task["interfaces"] = [
                *_strings(_value(graph_task, "interfaces")),
                *[
                    f"bundle-conflict-key:{key}"
                    for key in task.conflict_keys
                ],
            ]
        graph_tasks.append(graph_task)
    graph = materialize_task_conflict_graph(graph_tasks, max_lanes=None)
    conflict_pairs, edge_by_pair = _conflict_pairs(graph, tasks)

    # Turn each blocking conflict into a stable one-way prerequisite.  Conflicts
    # across dependency depths point forward.  Conflicts within one independent
    # wave use a deterministic graph coloring, so path-shaped conflict graphs
    # retain the width of their independent endpoints instead of becoming a
    # needless lexical chain.
    serialized_dependencies = {
        cid: set(prerequisites) for cid, prerequisites in dependencies.items()
    }
    colors_by_cid: dict[str, int] = {}
    for wave in sorted(set(waves.values())):
        wave_cids = [cid for cid, value in waves.items() if value == wave]
        colors_by_cid.update(
            _conflict_color_order(wave_cids, conflict_pairs)
        )
    for pair in conflict_pairs:
        left, right = sorted(pair)
        if waves[left] != waves[right]:
            left, right = sorted(pair, key=lambda cid: (waves[cid], cid))
        else:
            left, right = sorted(
                pair, key=lambda cid: (colors_by_cid[cid], cid)
            )
        serialized_dependencies[right].add(left)
    effective_wave = _waves_from_dependencies(
        serialized_dependencies,
        minimum_waves={
            task.canonical_task_cid: task.dependency_depth for task in tasks
        },
    )
    by_wave: dict[int, list[_CanonicalTask]] = defaultdict(list)
    for task in tasks:
        by_wave[effective_wave[task.canonical_task_cid]].append(task)

    groups: list[tuple[int, list[_CanonicalTask]]] = []
    for wave, wave_tasks in sorted(by_wave.items()):
        remaining = {
            task.canonical_task_cid: task for task in wave_tasks
        }
        while remaining:
            def seed_rank(item: _CanonicalTask) -> tuple[int, int, str]:
                compatible_scores = [
                    _affinity(item, peer, selected)
                    for peer in remaining.values()
                    if peer.canonical_task_cid != item.canonical_task_cid
                    and _compatible(item, peer, selected)
                    and frozenset(
                        (item.canonical_task_cid, peer.canonical_task_cid)
                    )
                    not in conflict_pairs
                ]
                return (
                    -max(compatible_scores, default=0),
                    -sum(score > 0 for score in compatible_scores),
                    item.canonical_task_cid,
                )

            seed_cid = min(remaining.values(), key=seed_rank).canonical_task_cid
            group = [remaining.pop(seed_cid)]
            while remaining and len(group) < selected.max_tasks_per_bundle:
                ranked: list[tuple[int, str, _CanonicalTask]] = []
                for candidate in remaining.values():
                    if any(
                        not _compatible(candidate, member, selected)
                        for member in group
                    ):
                        continue
                    if not selected.allow_internal_conflicts and any(
                        frozenset(
                            (candidate.canonical_task_cid, member.canonical_task_cid)
                        )
                        in conflict_pairs
                        for member in group
                    ):
                        continue
                    if (
                        sum(
                            item.estimated_context_tokens
                            for item in (*group, candidate)
                        )
                        > selected.max_context_tokens_per_bundle
                    ):
                        continue
                    score = sum(
                        _affinity(candidate, member, selected) for member in group
                    )
                    ranked.append((-score, candidate.canonical_task_cid, candidate))
                if not ranked:
                    break
                score, candidate_cid, candidate = min(ranked)
                if selected.require_affinity and -score <= 0:
                    break
                group.append(candidate)
                remaining.pop(candidate_cid)
            groups.append((wave, sorted(group, key=lambda item: item.canonical_task_cid)))

    bundles: list[OptimizedTaskBundle] = []
    for wave, group in groups:
        task_cids = tuple(item.canonical_task_cid for item in group)
        task_cid_set = set(task_cids)
        keys_by_cid = {
            item.canonical_task_cid: item.canonical_task_key for item in group
        }
        ids_by_cid = {item.canonical_task_cid: item.task_id for item in group}
        contexts = tuple(sorted({path for item in group for path in item.context_paths}))
        context_sets = [set(item.context_paths) for item in group]
        shared_contexts = (
            tuple(sorted(context_sets[0].intersection(*context_sets[1:])))
            if len(context_sets) > 1
            else ()
        )
        evidence = tuple(
            sorted({key for item in group for key in item.evidence_keys})
        )
        evidence_sets = [set(item.evidence_keys) for item in group]
        shared_evidence = (
            tuple(sorted(evidence_sets[0].intersection(*evidence_sets[1:])))
            if len(evidence_sets) > 1
            else ()
        )
        validation_sets = [
            set(item.validation_commands)
            for item in group
        ]
        shared_validations = (
            tuple(
                sorted(
                    validation_sets[0].intersection(*validation_sets[1:])
                )
            )
            if len(validation_sets) > 1
            else ()
        )
        group_aggregate_cids = tuple(
            item.canonical_task_cid for item in group if item.is_packet_aggregate
        )
        group_covered_cids = tuple(
            sorted(
                {
                    covered
                    for projection in packet_aggregates
                    if projection.aggregate_task_cid in group_aggregate_cids
                    for covered in projection.covered_sibling_task_cids
                }
            )
        )
        incident_edges = {
            pair: edge
            for pair, edge in edge_by_pair.items()
            if pair & task_cid_set
        }
        conflict_weight = int(
            round(sum(edge.weight for edge in incident_edges.values()) * 1_000_000)
        )
        material = {
            "schema": BUNDLE_OPTIMIZER_SCHEMA,
            "policy_id": selected.policy_id,
            "execution_wave": wave,
            "task_cids": list(task_cids),
        }
        bundles.append(
            OptimizedTaskBundle(
                bundle_cid=canonical_content_cid(material),
                task_cids=task_cids,
                canonical_task_keys=tuple(keys_by_cid[cid] for cid in task_cids),
                display_task_ids=tuple(ids_by_cid[cid] for cid in task_cids),
                execution_wave=wave,
                goal_ids=tuple(
                    sorted({item.goal_id for item in group if item.goal_id})
                ),
                validation_commands=tuple(
                    sorted(
                        {
                            command
                            for item in group
                            for command in item.validation_commands
                        }
                    )
                ),
                shared_validation_commands=shared_validations,
                context_paths=contexts,
                shared_context_paths=shared_contexts,
                evidence_keys=evidence,
                shared_evidence_keys=shared_evidence,
                resource_classes=tuple(
                    sorted(
                        {
                            item.resource_class
                            for item in group
                            if item.resource_class
                        }
                    )
                ),
                provider_batch_keys=tuple(
                    sorted(
                        {
                            item.provider_batch_key
                            for item in group
                            if item.provider_batch_key
                        }
                    )
                ),
                merge_localities=tuple(
                    sorted(
                        {
                            item.merge_family or item.merge_fate
                            for item in group
                            if item.merge_family or item.merge_fate
                        }
                    )
                ),
                packet_aggregate_task_cids=group_aggregate_cids,
                covered_sibling_task_cids=group_covered_cids,
                dependency_task_cids=tuple(
                    sorted(
                        {
                            dependency
                            for item in group
                            for dependency in serialized_dependencies[
                                item.canonical_task_cid
                            ]
                            if dependency not in task_cid_set
                        }
                    )
                ),
                conflict_weight=conflict_weight,
                estimated_context_tokens=sum(
                    item.estimated_context_tokens for item in group
                ),
            )
        )
    bundles.sort(key=lambda item: (item.execution_wave, item.bundle_cid))
    width: dict[int, int] = defaultdict(int)
    for bundle in bundles:
        width[bundle.execution_wave] += 1
    optimized_groups = [
        (
            bundle.execution_wave,
            [task for task in tasks if task.canonical_task_cid in bundle.task_cids],
        )
        for bundle in bundles
    ]
    if current_planner_bundles is None:
        current_groups = [
            (waves[task.canonical_task_cid], [task]) for task in tasks
        ]
    else:
        by_cid = {task.canonical_task_cid: task for task in tasks}
        current_groups = []
        seen_current: set[str] = set()
        for raw_bundle in current_planner_bundles:
            if isinstance(raw_bundle, Mapping):
                raw_cids = raw_bundle.get("task_cids") or ()
                wave = int(raw_bundle.get("execution_wave") or 0)
            elif hasattr(raw_bundle, "task_cids"):
                raw_cids = getattr(raw_bundle, "task_cids")
                wave = int(getattr(raw_bundle, "execution_wave", 0) or 0)
            else:
                raw_cids = raw_bundle
                wave = 0
            if isinstance(raw_cids, str):
                raw_cids = (raw_cids,)
            cids = tuple(str(cid) for cid in raw_cids)
            unknown = sorted(set(cids) - set(by_cid))
            duplicate = sorted(set(cids) & seen_current)
            if unknown or duplicate or not cids:
                raise ValueError(
                    "current planner bundles must be a non-empty exact "
                    "partition of canonical task CIDs"
                )
            seen_current.update(cids)
            current_groups.append((wave, [by_cid[cid] for cid in cids]))
        if seen_current != set(by_cid):
            raise ValueError(
                "current planner bundles must cover every admitted canonical task"
            )
    metrics = _plan_metrics(
        tasks=tasks,
        groups=optimized_groups,
        conflict_pairs=conflict_pairs,
    )
    current_metrics = _plan_metrics(
        tasks=tasks,
        groups=current_groups,
        conflict_pairs=conflict_pairs,
    )
    comparison = compare_bundle_plan_metrics(current_metrics, metrics)
    graph_payload = graph.to_dict()
    existing_pairs = {
        frozenset(
            (
                str(edge.get("left_task_cid") or ""),
                str(edge.get("right_task_cid") or ""),
            )
        )
        for edge in graph_payload.get("edges", [])
        if isinstance(edge, Mapping)
    }
    graph_payload["edges"] = [
        *graph_payload.get("edges", []),
        *[
            edge.to_dict()
            for pair, edge in sorted(
                edge_by_pair.items(), key=lambda item: sorted(item[0])
            )
            if pair not in existing_pairs
        ],
    ]
    return BundleOptimizationResult(
        policy_id=selected.policy_id,
        bundles=tuple(bundles),
        task_count=len(tasks),
        execution_width_by_wave=dict(width),
        metrics=metrics,
        packet_aggregates=packet_aggregates,
        comparison=comparison,
        conflict_graph=graph_payload,
    )


def _critical_path_width_material(
    *,
    tasks: Sequence[_CanonicalTask],
    plan: BundleOptimizationResult,
    repository_tree: str,
) -> dict[str, Any]:
    base_waves, dependencies = _dependency_waves(tasks)
    graph_tasks: list[dict[str, Any]] = []
    for task in tasks:
        graph_task = dict(task.payload)
        if task.conflict_keys:
            graph_task["interfaces"] = [
                *_strings(_value(graph_task, "interfaces")),
                *[
                    f"bundle-conflict-key:{key}"
                    for key in task.conflict_keys
                ],
            ]
        graph_tasks.append(graph_task)
    graph = materialize_task_conflict_graph(graph_tasks, max_lanes=None)
    conflict_pairs, _edges = _conflict_pairs(graph, tasks)

    projections: list[ConflictWaveProjection] = []
    colors_by_cid: dict[str, int] = {}
    for wave in sorted(set(base_waves.values())):
        wave_cids = [
            cid for cid, assigned_wave in base_waves.items()
            if assigned_wave == wave
        ]
        projection = project_conflict_free_wave(
            wave_cids,
            (
                tuple(sorted(pair))
                for pair in conflict_pairs
                if pair.issubset(set(wave_cids))
            ),
            dependency_wave=wave,
        )
        projections.append(projection)
        colors_by_cid.update(projection.color_by_task_cid)

    serialized_dependencies = {
        cid: set(prerequisites) for cid, prerequisites in dependencies.items()
    }
    for pair in conflict_pairs:
        left, right = sorted(pair)
        if base_waves[left] != base_waves[right]:
            left, right = sorted(
                pair, key=lambda cid: (base_waves[cid], cid)
            )
        else:
            left, right = sorted(
                pair, key=lambda cid: (colors_by_cid[cid], cid)
            )
        serialized_dependencies[right].add(left)
    effective_waves = _waves_from_dependencies(
        serialized_dependencies,
        minimum_waves={
            task.canonical_task_cid: task.dependency_depth for task in tasks
        },
    )
    planned_task_waves = {
        cid: bundle.execution_wave
        for bundle in plan.bundles
        for cid in bundle.task_cids
    }
    return {
        "schema": CRITICAL_PATH_WIDTH_EVIDENCE_SCHEMA,
        "requirement_id": CRITICAL_PATH_WIDTH_REQUIREMENT_ID,
        "repository_tree": str(repository_tree),
        "policy_id": plan.policy_id,
        "task_population": [
            {
                "canonical_task_cid": task.canonical_task_cid,
                "canonical_task_key": task.canonical_task_key,
                "semantic_identity": task.semantic_identity,
                "display_task_id": task.task_id,
            }
            for task in tasks
        ],
        "dependency_task_cids": {
            cid: sorted(prerequisites)
            for cid, prerequisites in sorted(dependencies.items())
        },
        "base_dependency_waves": dict(sorted(base_waves.items())),
        "blocking_conflict_pairs": [
            list(sorted(pair))
            for pair in sorted(conflict_pairs, key=lambda value: tuple(sorted(value)))
        ],
        "conflict_wave_projections": [
            projection.to_dict() for projection in projections
        ],
        "serialized_dependencies": {
            cid: sorted(prerequisites)
            for cid, prerequisites in sorted(serialized_dependencies.items())
        },
        "effective_task_waves": dict(sorted(effective_waves.items())),
        "planned_task_waves": dict(sorted(planned_task_waves.items())),
        "bundle_population": [
            {
                "bundle_cid": bundle.bundle_cid,
                "execution_wave": bundle.execution_wave,
                "task_cids": list(bundle.task_cids),
            }
            for bundle in plan.bundles
        ],
        "independent_width_by_dependency_wave": {
            str(projection.dependency_wave): projection.independent_width
            for projection in projections
        },
    }


def _critical_path_width_qualifies(material: Mapping[str, Any]) -> bool:
    """Validate exact conflict serialization and a preserved width witness."""

    try:
        if not str(material.get("repository_tree") or "").strip():
            return False
        if not str(material.get("policy_id") or "").strip():
            return False
        population = material.get("task_population")
        if not isinstance(population, list) or len(population) < 2:
            return False
        cids = [
            str(item.get("canonical_task_cid") or "")
            for item in population
            if isinstance(item, Mapping)
            and str(item.get("canonical_task_cid") or "")
            and str(item.get("canonical_task_key") or "")
        ]
        if len(cids) != len(population) or len(set(cids)) != len(cids):
            return False
        cid_set = set(cids)

        def dependency_map(name: str) -> dict[str, set[str]]:
            raw = material.get(name)
            if not isinstance(raw, Mapping) or set(raw) != cid_set:
                raise ValueError(f"{name} must cover the canonical population")
            result = {
                str(cid): {
                    str(value) for value in values if str(value)
                }
                for cid, values in raw.items()
                if isinstance(values, Sequence)
                and not isinstance(values, (str, bytes, bytearray))
            }
            if set(result) != cid_set or any(
                not prerequisites.issubset(cid_set - {cid})
                for cid, prerequisites in result.items()
            ):
                raise ValueError(f"{name} contains an invalid dependency")
            return result

        dependencies = dependency_map("dependency_task_cids")
        serialized = dependency_map("serialized_dependencies")
        base_waves = {
            str(cid): int(wave)
            for cid, wave in dict(
                material.get("base_dependency_waves") or {}
            ).items()
        }
        effective_waves = {
            str(cid): int(wave)
            for cid, wave in dict(
                material.get("effective_task_waves") or {}
            ).items()
        }
        planned_waves = {
            str(cid): int(wave)
            for cid, wave in dict(
                material.get("planned_task_waves") or {}
            ).items()
        }
        if (
            set(base_waves) != cid_set
            or set(effective_waves) != cid_set
            or planned_waves != effective_waves
        ):
            return False
        if any(
            base_waves[parent] >= base_waves[child]
            or effective_waves[parent] >= effective_waves[child]
            for child, parents in dependencies.items()
            for parent in parents
        ):
            return False
        if any(
            effective_waves[parent] >= effective_waves[child]
            for child, parents in serialized.items()
            for parent in parents
        ):
            return False

        raw_pairs = material.get("blocking_conflict_pairs")
        if not isinstance(raw_pairs, list):
            return False
        conflict_pairs = {
            tuple(str(value) for value in pair)
            for pair in raw_pairs
            if isinstance(pair, list)
            and len(pair) == 2
            and pair[0] != pair[1]
            and set(str(value) for value in pair).issubset(cid_set)
        }
        if len(conflict_pairs) != len(raw_pairs):
            return False
        if any(
            effective_waves[left] == effective_waves[right]
            for left, right in conflict_pairs
        ):
            return False

        projections = material.get("conflict_wave_projections")
        if not isinstance(projections, list) or not projections:
            return False
        projected_cids: set[str] = set()
        projected_colors: dict[str, int] = {}
        preserved_width_witness = False
        expected_width_by_wave: dict[str, int] = {}
        for raw_projection in projections:
            if not isinstance(raw_projection, Mapping):
                return False
            wave = int(raw_projection.get("dependency_wave") or 0)
            members = tuple(
                str(value) for value in raw_projection.get("task_cids", [])
            )
            if (
                not members
                or len(set(members)) != len(members)
                or not set(members).issubset(cid_set)
                or any(base_waves[cid] != wave for cid in members)
            ):
                return False
            if projected_cids.intersection(members):
                return False
            lanes = raw_projection.get("independent_lanes")
            colors = raw_projection.get("color_by_task_cid")
            if not isinstance(lanes, list) or not isinstance(colors, Mapping):
                return False
            flat_lanes = [
                str(cid) for lane in lanes for cid in lane
                if isinstance(lane, list)
            ]
            if sorted(flat_lanes) != sorted(members):
                return False
            independent_width = max(
                (len(lane) for lane in lanes if isinstance(lane, list)),
                default=0,
            )
            if int(raw_projection.get("independent_width") or 0) != independent_width:
                return False
            expected_width_by_wave[str(wave)] = independent_width
            widest_lane_preserved = independent_width <= 1
            for color, lane in enumerate(lanes):
                if not isinstance(lane, list):
                    return False
                lane_set = {str(cid) for cid in lane}
                if any(set(pair).issubset(lane_set) for pair in conflict_pairs):
                    return False
                if len(lane_set) >= 2 and len(
                    {effective_waves[cid] for cid in lane_set}
                ) == 1:
                    preserved_width_witness = True
                    if len(lane_set) == independent_width:
                        widest_lane_preserved = True
                for cid in lane_set:
                    if int(colors.get(cid, -1)) != color:
                        return False
                    projected_colors[cid] = color
            if not widest_lane_preserved:
                return False
            projected_cids.update(members)
        if projected_cids != cid_set or not preserved_width_witness:
            return False
        recorded_widths = {
            str(wave): int(width)
            for wave, width in dict(
                material.get("independent_width_by_dependency_wave") or {}
            ).items()
        }
        if recorded_widths != expected_width_by_wave:
            return False

        expected_serialized = {
            cid: set(parents) for cid, parents in dependencies.items()
        }
        for left, right in conflict_pairs:
            if base_waves[left] != base_waves[right]:
                before, after = sorted(
                    (left, right), key=lambda cid: (base_waves[cid], cid)
                )
            else:
                before, after = sorted(
                    (left, right), key=lambda cid: (projected_colors[cid], cid)
                )
            expected_serialized[after].add(before)
        if serialized != expected_serialized:
            return False

        bundles = material.get("bundle_population")
        if not isinstance(bundles, list) or not bundles:
            return False
        bundled_cids: list[str] = []
        for bundle in bundles:
            if not isinstance(bundle, Mapping):
                return False
            wave = int(bundle.get("execution_wave") or 0)
            members = [str(value) for value in bundle.get("task_cids", [])]
            if not members or any(planned_waves[cid] != wave for cid in members):
                return False
            if any(set(pair).issubset(set(members)) for pair in conflict_pairs):
                return False
            bundled_cids.extend(members)
        return sorted(bundled_cids) == sorted(cids) and len(bundled_cids) == len(
            set(bundled_cids)
        )
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class CriticalPathWidthEvidence:
    """Content-addressed proof that bundling retains conflict-free width."""

    repository_tree: str
    policy_id: str
    task_population: tuple[Mapping[str, Any], ...]
    dependency_task_cids: Mapping[str, tuple[str, ...]]
    base_dependency_waves: Mapping[str, int]
    blocking_conflict_pairs: tuple[tuple[str, str], ...]
    conflict_wave_projections: tuple[Mapping[str, Any], ...]
    serialized_dependencies: Mapping[str, tuple[str, ...]]
    effective_task_waves: Mapping[str, int]
    planned_task_waves: Mapping[str, int]
    bundle_population: tuple[Mapping[str, Any], ...]
    independent_width_by_dependency_wave: Mapping[str, int]
    evidence_id: str
    integrity_digest: str
    _producer_seal: Any = field(default=None, compare=False, repr=False)

    @classmethod
    def create(
        cls,
        admitted_tasks: Any,
        *,
        policy: BundleOptimizationPolicy | None = None,
        current_planner_bundles: Sequence[Any] | None = None,
        repository_tree: str = "in-memory",
    ) -> "CriticalPathWidthEvidence":
        """Run the optimizer and bind its complete dependency/conflict witness."""

        selected = policy or BundleOptimizationPolicy()
        tasks = _canonical_tasks(admitted_tasks)
        plan = optimize_task_bundles(
            [task.payload for task in tasks],
            policy=selected,
            current_planner_bundles=current_planner_bundles,
        )
        material = _critical_path_width_material(
            tasks=tasks,
            plan=plan,
            repository_tree=repository_tree,
        )
        digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        return cls._from_material(
            material,
            evidence_id=canonical_content_cid(material),
            integrity_digest=digest,
            producer_seal=_CRITICAL_PATH_WIDTH_EVIDENCE_SEAL,
        )

    @classmethod
    def _from_material(
        cls,
        material: Mapping[str, Any],
        *,
        evidence_id: str,
        integrity_digest: str,
        producer_seal: Any,
    ) -> "CriticalPathWidthEvidence":
        return cls(
            repository_tree=str(material.get("repository_tree") or ""),
            policy_id=str(material.get("policy_id") or ""),
            task_population=tuple(
                dict(item) for item in material.get("task_population", [])
            ),
            dependency_task_cids={
                str(cid): tuple(str(value) for value in values)
                for cid, values in dict(
                    material.get("dependency_task_cids") or {}
                ).items()
            },
            base_dependency_waves={
                str(cid): int(wave)
                for cid, wave in dict(
                    material.get("base_dependency_waves") or {}
                ).items()
            },
            blocking_conflict_pairs=tuple(
                tuple(str(value) for value in pair)
                for pair in material.get("blocking_conflict_pairs", [])
            ),
            conflict_wave_projections=tuple(
                dict(item)
                for item in material.get("conflict_wave_projections", [])
            ),
            serialized_dependencies={
                str(cid): tuple(str(value) for value in values)
                for cid, values in dict(
                    material.get("serialized_dependencies") or {}
                ).items()
            },
            effective_task_waves={
                str(cid): int(wave)
                for cid, wave in dict(
                    material.get("effective_task_waves") or {}
                ).items()
            },
            planned_task_waves={
                str(cid): int(wave)
                for cid, wave in dict(
                    material.get("planned_task_waves") or {}
                ).items()
            },
            bundle_population=tuple(
                dict(item) for item in material.get("bundle_population", [])
            ),
            independent_width_by_dependency_wave={
                str(wave): int(width)
                for wave, width in dict(
                    material.get("independent_width_by_dependency_wave") or {}
                ).items()
            },
            evidence_id=str(evidence_id),
            integrity_digest=str(integrity_digest),
            _producer_seal=producer_seal,
        )

    def _material(self) -> dict[str, Any]:
        return {
            "schema": CRITICAL_PATH_WIDTH_EVIDENCE_SCHEMA,
            "requirement_id": CRITICAL_PATH_WIDTH_REQUIREMENT_ID,
            "repository_tree": self.repository_tree,
            "policy_id": self.policy_id,
            "task_population": [dict(item) for item in self.task_population],
            "dependency_task_cids": {
                cid: list(values)
                for cid, values in sorted(self.dependency_task_cids.items())
            },
            "base_dependency_waves": dict(
                sorted(self.base_dependency_waves.items())
            ),
            "blocking_conflict_pairs": [
                list(pair) for pair in self.blocking_conflict_pairs
            ],
            "conflict_wave_projections": [
                dict(item) for item in self.conflict_wave_projections
            ],
            "serialized_dependencies": {
                cid: list(values)
                for cid, values in sorted(self.serialized_dependencies.items())
            },
            "effective_task_waves": dict(
                sorted(self.effective_task_waves.items())
            ),
            "planned_task_waves": dict(sorted(self.planned_task_waves.items())),
            "bundle_population": [
                dict(item) for item in self.bundle_population
            ],
            "independent_width_by_dependency_wave": dict(
                sorted(self.independent_width_by_dependency_wave.items())
            ),
        }

    def verify_integrity(self) -> bool:
        material = self._material()
        return (
            self.integrity_digest
            == hashlib.sha256(canonical_json_bytes(material)).hexdigest()
            and self.evidence_id == canonical_content_cid(material)
        )

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        if (
            self._producer_seal is _CRITICAL_PATH_WIDTH_EVIDENCE_SEAL
            and self.verify_integrity()
            and _critical_path_width_qualifies(self._material())
        ):
            return (CRITICAL_PATH_WIDTH_REQUIREMENT_ID,)
        return ()

    def to_dict(self) -> dict[str, Any]:
        material = self._material()
        qualifies = bool(self.proved_requirement_ids)
        material.update(
            {
                "evidence_id": self.evidence_id,
                "integrity_digest": self.integrity_digest,
                "proved_requirement_ids": list(self.proved_requirement_ids),
                "status": "passed" if qualifies else "diagnostic",
                "complete": qualifies,
                "coverage_complete": qualifies,
                "source_tier": "validation",
            }
        )
        return material

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CriticalPathWidthEvidence":
        evidence = cls._from_material(
            payload,
            evidence_id=str(payload.get("evidence_id") or ""),
            integrity_digest=str(payload.get("integrity_digest") or ""),
            producer_seal=None,
        )
        if not evidence.verify_integrity():
            raise ValueError("critical-path width evidence digest mismatch")
        return evidence


def prove_critical_path_width(
    admitted_tasks: Any,
    *,
    policy: BundleOptimizationPolicy | None = None,
    current_planner_bundles: Sequence[Any] | None = None,
    repository_tree: str = "in-memory",
) -> CriticalPathWidthEvidence:
    """Produce a canonical conflict-aware critical-path width receipt."""

    return CriticalPathWidthEvidence.create(
        admitted_tasks,
        policy=policy,
        current_planner_bundles=current_planner_bundles,
        repository_tree=repository_tree,
    )


def _completion_material(
    *,
    tasks: Sequence[_CanonicalTask],
    completed_task_cids: Sequence[str],
    propagated_task_cids: Sequence[str],
    repository_tree: str,
    policy_id: str,
) -> dict[str, Any]:
    return {
        "schema": PACKET_COMPLETION_EVIDENCE_SCHEMA,
        "requirement_id": PACKET_COMPLETION_BINDING_REQUIREMENT_ID,
        "repository_tree": repository_tree,
        "policy_id": policy_id,
        "task_population": [
            {
                "canonical_task_cid": task.canonical_task_cid,
                "canonical_task_key": task.canonical_task_key,
                "display_task_id": task.task_id,
                "goal_packet_key": task.goal_packet_key,
                "goal_packet_role": task.goal_packet_role,
                "completion_task_bindings": list(task.completion_task_bindings),
            }
            for task in tasks
        ],
        "completed_task_cids": sorted(set(completed_task_cids)),
        "propagated_task_cids": sorted(set(propagated_task_cids)),
    }


def _completion_qualifies(material: Mapping[str, Any]) -> bool:
    population = material.get("task_population")
    if not isinstance(population, list) or not population:
        return False
    by_cid = {
        str(item.get("canonical_task_cid") or ""): item
        for item in population
        if isinstance(item, Mapping)
        and str(item.get("canonical_task_cid") or "")
        and str(item.get("canonical_task_key") or "")
    }
    if len(by_cid) != len(population):
        return False
    completed = {
        str(value) for value in material.get("completed_task_cids", []) if str(value)
    }
    propagated = {
        str(value)
        for value in material.get("propagated_task_cids", [])
        if str(value)
    }
    if not completed or not completed.issubset(by_cid) or not propagated.issubset(by_cid):
        return False
    expected: set[str] = set()
    aggregate_seen = False
    for cid in completed:
        task = by_cid[cid]
        is_aggregate = str(task.get("goal_packet_role") or "").casefold() == "packet_aggregate"
        bindings = {
            str(value)
            for value in task.get("completion_task_bindings", [])
            if str(value)
        }
        if not is_aggregate:
            continue
        aggregate_seen = True
        if cid in bindings or not bindings.issubset(by_cid):
            return False
        packet_key = str(task.get("goal_packet_key") or "")
        if any(
            not packet_key
            or str(by_cid[bound].get("goal_packet_key") or "") != packet_key
            for bound in bindings
        ):
            return False
        expected.update(bindings)
    return aggregate_seen and expected == propagated and not (propagated & completed)


@dataclass(frozen=True)
class PacketCompletionBindingEvidence:
    """Content-addressed proof of exact packet completion propagation."""

    repository_tree: str
    policy_id: str
    task_population: tuple[Mapping[str, Any], ...]
    completed_task_cids: tuple[str, ...]
    propagated_task_cids: tuple[str, ...]
    evidence_id: str
    integrity_digest: str
    _producer_seal: Any = field(
        default=None, compare=False, repr=False
    )

    @classmethod
    def create(
        cls,
        tasks: Sequence[Any],
        *,
        completed_task_cids: Sequence[str],
        propagated_task_cids: Sequence[str],
        repository_tree: str = "in-memory",
        policy_id: str = "bundle-optimization-policy/default",
    ) -> "PacketCompletionBindingEvidence":
        canonical = _canonical_tasks(tasks)
        material = _completion_material(
            tasks=canonical,
            completed_task_cids=completed_task_cids,
            propagated_task_cids=propagated_task_cids,
            repository_tree=str(repository_tree),
            policy_id=str(policy_id),
        )
        digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        return cls(
            repository_tree=str(repository_tree),
            policy_id=str(policy_id),
            task_population=tuple(material["task_population"]),
            completed_task_cids=tuple(material["completed_task_cids"]),
            propagated_task_cids=tuple(material["propagated_task_cids"]),
            evidence_id=canonical_content_cid(material),
            integrity_digest=digest,
            _producer_seal=_PACKET_COMPLETION_EVIDENCE_SEAL,
        )

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        if (
            self._producer_seal is _PACKET_COMPLETION_EVIDENCE_SEAL
            and self.verify_integrity()
            and _completion_qualifies(self._material())
        ):
            return (PACKET_COMPLETION_BINDING_REQUIREMENT_ID,)
        return ()

    def _material(self) -> dict[str, Any]:
        return {
            "schema": PACKET_COMPLETION_EVIDENCE_SCHEMA,
            "requirement_id": PACKET_COMPLETION_BINDING_REQUIREMENT_ID,
            "repository_tree": self.repository_tree,
            "policy_id": self.policy_id,
            "task_population": [dict(item) for item in self.task_population],
            "completed_task_cids": list(self.completed_task_cids),
            "propagated_task_cids": list(self.propagated_task_cids),
        }

    def verify_integrity(self) -> bool:
        material = self._material()
        digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        return (
            self.integrity_digest == digest
            and self.evidence_id == canonical_content_cid(material)
        )

    def to_dict(self) -> dict[str, Any]:
        material = self._material()
        qualifies = bool(self.proved_requirement_ids)
        material.update(
            {
                "evidence_id": self.evidence_id,
                "integrity_digest": self.integrity_digest,
                "proved_requirement_ids": list(self.proved_requirement_ids),
                "status": "passed" if qualifies else "diagnostic",
                "complete": qualifies,
                "coverage_complete": qualifies,
                "source_tier": "validation",
            }
        )
        return material

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PacketCompletionBindingEvidence":
        """Validate a serialized receipt without granting producer authority.

        Content integrity is reproducible by any caller; producer provenance is
        not.  A deserialized lookalike therefore remains diagnostic even when
        its digest and structural completion claim are valid.
        """

        receipt = cls(
            repository_tree=str(payload.get("repository_tree") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            task_population=tuple(
                dict(item)
                for item in payload.get("task_population", [])
                if isinstance(item, Mapping)
            ),
            completed_task_cids=tuple(
                str(value) for value in payload.get("completed_task_cids", [])
            ),
            propagated_task_cids=tuple(
                str(value) for value in payload.get("propagated_task_cids", [])
            ),
            evidence_id=str(payload.get("evidence_id") or ""),
            integrity_digest=str(payload.get("integrity_digest") or ""),
            _producer_seal=None,
        )
        if not receipt.verify_integrity():
            raise ValueError("packet completion evidence digest mismatch")
        return receipt


@dataclass(frozen=True)
class PacketCompletionResult:
    completed_task_cids: tuple[str, ...]
    propagated_task_cids: tuple[str, ...]
    evidence: PacketCompletionBindingEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_task_cids": list(self.completed_task_cids),
            "propagated_task_cids": list(self.propagated_task_cids),
            "evidence": self.evidence.to_dict(),
        }


def propagate_goal_packet_completion(
    tasks: Sequence[Any],
    *,
    completed_task_cids: Sequence[str],
    repository_tree: str = "in-memory",
    policy_id: str = "bundle-optimization-policy/default",
) -> PacketCompletionResult:
    """Propagate completion to exactly explicitly bound canonical siblings."""

    canonical = _canonical_tasks(tasks)
    by_cid = {task.canonical_task_cid: task for task in canonical}
    completed = tuple(sorted(set(str(value) for value in completed_task_cids if str(value))))
    unknown = set(completed) - set(by_cid)
    if unknown:
        raise ValueError(
            "completed canonical task CIDs are not in the task population: "
            + ", ".join(sorted(unknown))
        )
    propagated: set[str] = set()
    for cid in completed:
        task = by_cid[cid]
        if not task.is_packet_aggregate:
            continue
        bindings = set(task.completion_task_bindings)
        unknown_bindings = bindings - set(by_cid)
        if unknown_bindings:
            raise ValueError(
                "completion bindings reference unknown canonical task CIDs: "
                + ", ".join(sorted(unknown_bindings))
            )
        propagated.update(bindings)
    propagated.difference_update(completed)
    propagated_tuple = tuple(sorted(propagated))
    evidence = PacketCompletionBindingEvidence.create(
        [task.payload for task in canonical],
        completed_task_cids=completed,
        propagated_task_cids=propagated_tuple,
        repository_tree=repository_tree,
        policy_id=policy_id,
    )
    # Fail closed if packet or cross-packet constraints invalidate the binding.
    if propagated_tuple and not evidence.proved_requirement_ids:
        raise ValueError("packet completion binding is not exact and canonical")
    return PacketCompletionResult(
        completed_task_cids=tuple(sorted(set(completed) | propagated)),
        propagated_task_cids=propagated_tuple,
        evidence=evidence,
    )


__all__ = [
    "BUNDLE_OPTIMIZER_SCHEMA",
    "CRITICAL_PATH_WIDTH_EVIDENCE_SCHEMA",
    "CRITICAL_PATH_WIDTH_REQUIREMENT_ID",
    "PACKET_COMPLETION_BINDING_REQUIREMENT_ID",
    "PACKET_COMPLETION_EVIDENCE_SCHEMA",
    "BundleOptimizationPolicy",
    "BundleOptimizationResult",
    "BundlePlanComparison",
    "CriticalPathWidthEvidence",
    "OptimizedTaskBundle",
    "PacketAggregateProjection",
    "PacketCompletionBindingEvidence",
    "PacketCompletionResult",
    "compare_bundle_plan_metrics",
    "optimize_task_bundles",
    "prove_critical_path_width",
    "propagate_goal_packet_completion",
]
