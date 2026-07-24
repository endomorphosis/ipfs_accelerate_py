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

from .conflict_graph import ConflictEdge, TaskConflictGraph, materialize_task_conflict_graph
from .task_identity import canonical_content_cid, canonical_json_bytes


PACKET_COMPLETION_BINDING_REQUIREMENT_ID: Final = (
    "187052702852200236079602798955260586139"
)
BUNDLE_OPTIMIZER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/bundle-optimization@1"
)
PACKET_COMPLETION_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/packet-completion-binding-evidence@1"
)
_PACKET_COMPLETION_EVIDENCE_SEAL: Final = object()


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
    resource_weight: int = 1

    def __post_init__(self) -> None:
        for name in (
            "max_tasks_per_bundle",
            "max_context_tokens_per_bundle",
            "context_weight",
            "validation_weight",
            "merge_locality_weight",
            "goal_weight",
            "resource_weight",
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
    goal_id: str
    goal_packet_key: str
    goal_packet_role: str
    merge_family: str
    merge_fate: str
    context_paths: tuple[str, ...]
    validation_commands: tuple[str, ...]
    outputs: tuple[str, ...]
    predicted_paths: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    dependencies: tuple[str, ...]
    conflict_keys: tuple[str, ...]
    resource_class: str
    estimated_context_tokens: int
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
            goal_id=str(_value(payload, "goal id") or "").strip(),
            goal_packet_key=str(_value(payload, "goal packet key", "goal packet") or "").strip(),
            goal_packet_role=str(_value(payload, "goal packet role") or "").strip(),
            merge_family=str(_value(payload, "merge family") or "").strip(),
            merge_fate=str(_value(payload, "merge fate") or "").strip(),
            context_paths=_paths(
                _value(payload, "context paths", "context keys", "context files")
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
                _value(payload, "dependency task cids", "dependencies", "depends on")
            ),
            conflict_keys=_strings(_value(payload, "conflicts", "conflict keys")),
            resource_class=str(_value(payload, "resource class") or "").strip(),
            estimated_context_tokens=context_tokens,
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
    return _waves_from_dependencies(dependencies), dependencies


def _waves_from_dependencies(
    dependencies: Mapping[str, set[str]],
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
            waves[cid] = (
                max((waves[parent] + 1 for parent in dependencies[cid]), default=0)
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
    return (
        policy.context_weight * context
        + policy.validation_weight * validation
        + policy.merge_locality_weight * merge
        + policy.goal_weight * goal
        + policy.resource_weight * resource
    )


@dataclass(frozen=True)
class OptimizedTaskBundle:
    """One identity-preserving execution bundle."""

    bundle_cid: str
    task_cids: tuple[str, ...]
    canonical_task_keys: tuple[str, ...]
    display_task_ids: tuple[str, ...]
    execution_wave: int
    validation_commands: tuple[str, ...]
    context_paths: tuple[str, ...]
    shared_context_paths: tuple[str, ...]
    resource_classes: tuple[str, ...]
    dependency_task_cids: tuple[str, ...]
    conflict_weight: int
    estimated_context_tokens: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "task_cids",
            "canonical_task_keys",
            "display_task_ids",
            "validation_commands",
            "context_paths",
            "shared_context_paths",
            "resource_classes",
            "dependency_task_cids",
        ):
            result[name] = list(result[name])
        return result


@dataclass(frozen=True)
class BundleOptimizationResult:
    policy_id: str
    bundles: tuple[OptimizedTaskBundle, ...]
    task_count: int
    execution_width_by_wave: Mapping[int, int]
    metrics: Mapping[str, int]
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
            "conflict_graph": dict(self.conflict_graph),
        }


def optimize_task_bundles(
    admitted_tasks: Any,
    *,
    policy: BundleOptimizationPolicy | None = None,
) -> BundleOptimizationResult:
    """Optimize canonical admitted tasks without changing task admission."""

    selected = policy or BundleOptimizationPolicy()
    tasks = _canonical_tasks(admitted_tasks)
    if not tasks:
        return BundleOptimizationResult(
            policy_id=selected.policy_id,
            bundles=(),
            task_count=0,
            execution_width_by_wave={},
            metrics={
                "accepted_work_item_count": 0,
                "model_call_count": 0,
                "model_calls_per_work_item_millionths": 0,
                "context_reuse_millionths": 0,
                "validation_reuse_millionths": 0,
                "critical_path_wave_count": 0,
                "blocking_conflict_count": 0,
                "internal_blocking_conflict_count": 0,
            },
            conflict_graph={},
        )
    waves, dependencies = _dependency_waves(tasks)
    graph = materialize_task_conflict_graph(
        [task.payload for task in tasks],
        max_lanes=None,
    )
    conflict_pairs, edge_by_pair = _conflict_pairs(graph, tasks)

    # Turn each blocking conflict into a stable one-way prerequisite.  Ordering
    # by the original dependency wave prevents a conflict edge from pointing
    # backwards through the DAG; the CID tie-break is deterministic within one
    # independent wave.  Recomputing longest paths preserves every dependency
    # while serializing only the conflicting pairs.
    serialized_dependencies = {
        cid: set(prerequisites) for cid, prerequisites in dependencies.items()
    }
    for pair in conflict_pairs:
        left, right = sorted(
            pair, key=lambda cid: (waves[cid], cid)
        )
        serialized_dependencies[right].add(left)
    effective_wave = _waves_from_dependencies(serialized_dependencies)
    by_wave: dict[int, list[_CanonicalTask]] = defaultdict(list)
    for task in tasks:
        by_wave[effective_wave[task.canonical_task_cid]].append(task)

    groups: list[tuple[int, list[_CanonicalTask]]] = []
    for wave, wave_tasks in sorted(by_wave.items()):
        remaining = {
            task.canonical_task_cid: task for task in wave_tasks
        }
        while remaining:
            seed_cid = min(remaining)
            group = [remaining.pop(seed_cid)]
            while remaining and len(group) < selected.max_tasks_per_bundle:
                ranked: list[tuple[int, str, _CanonicalTask]] = []
                for candidate in remaining.values():
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
        context_sets = [set(item.context_paths) for item in group if item.context_paths]
        shared_contexts = (
            tuple(sorted(context_sets[0].intersection(*context_sets[1:])))
            if context_sets
            else ()
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
                validation_commands=tuple(
                    sorted(
                        {
                            command
                            for item in group
                            for command in item.validation_commands
                        }
                    )
                ),
                context_paths=contexts,
                shared_context_paths=shared_contexts,
                resource_classes=tuple(
                    sorted(
                        {
                            item.resource_class
                            for item in group
                            if item.resource_class
                        }
                    )
                ),
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
        width[bundle.execution_wave] += len(bundle.task_cids)
    raw_context_references = sum(len(task.context_paths) for task in tasks)
    unique_context_references = len(
        {path for task in tasks for path in task.context_paths}
    )
    raw_validation_references = sum(
        len(task.validation_commands) for task in tasks
    )
    unique_validation_references = len(
        {command for task in tasks for command in task.validation_commands}
    )
    metrics = {
        "accepted_work_item_count": len(tasks),
        "model_call_count": len(bundles),
        "model_calls_per_work_item_millionths": (
            len(bundles) * 1_000_000 // len(tasks)
        ),
        "context_reuse_millionths": (
            (raw_context_references - unique_context_references) * 1_000_000
            // raw_context_references
            if raw_context_references
            else 0
        ),
        "validation_reuse_millionths": (
            (raw_validation_references - unique_validation_references)
            * 1_000_000
            // raw_validation_references
            if raw_validation_references
            else 0
        ),
        "critical_path_wave_count": max(width, default=-1) + 1,
        "blocking_conflict_count": len(conflict_pairs),
        "internal_blocking_conflict_count": sum(
            1
            for bundle in bundles
            for pair in conflict_pairs
            if pair.issubset(bundle.task_cids)
        ),
    }
    return BundleOptimizationResult(
        policy_id=selected.policy_id,
        bundles=tuple(bundles),
        task_count=len(tasks),
        execution_width_by_wave=dict(width),
        metrics=metrics,
        conflict_graph=graph.to_dict(),
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
    "PACKET_COMPLETION_BINDING_REQUIREMENT_ID",
    "PACKET_COMPLETION_EVIDENCE_SCHEMA",
    "BundleOptimizationPolicy",
    "BundleOptimizationResult",
    "OptimizedTaskBundle",
    "PacketCompletionBindingEvidence",
    "PacketCompletionResult",
    "optimize_task_bundles",
    "propagate_goal_packet_completion",
]
