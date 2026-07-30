"""Bounded, replay-safe backlog refill from symbolic finding receipts.

The finding ledger is the authority for evidence admission.  This module is a
small deterministic planner: when open work is below a threshold it consumes
fresh ledger receipts, binds actionable findings to an exact goal family, and
emits goal/task packets.  It does not mutate the ledger or a task board.

Ambiguous, stale, rejected, unbound, or overly broad evidence is retained in
the returned diagnostics but can never create executable work.  All identities
are derived from semantic inputs so replaying a receipt is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import posixpath
import time
from typing import Any, Mapping, Protocol, Sequence

from .contract_findings import (
    AppendOutcome,
    AppendReceipt,
    ContractFindingRecord,
    FindingAdmissionState,
)


SYMBOLIC_FINDING_REFILL_VERSION = 1
SYMBOLIC_FINDING_REFILL_INTERFACE = "vfs/symbolic-finding-refill@1"
SYMBOLIC_REFILL_EPOCH_SCHEMA = "vfs/symbolic-refill-epoch@1"
REFILL_IDEMPOTENCY_SCHEMA = "vfs/refill-idempotency@1"

DEFAULT_REFILL_THRESHOLD = 4
DEFAULT_OPEN_WORK_CEILING = 12
DEFAULT_MAX_FINDINGS_PER_PASS = 8
DEFAULT_MAX_SURPLUS_PER_GOAL = 2
DEFAULT_MAX_CHILDREN = 3
DEFAULT_MAX_GOAL_DEPTH = 4
DEFAULT_COOLDOWN_SECONDS = 900
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_OUTPUT_PATHS = 8

# A refill proposal is neither completion evidence nor permission to repair.
REFILL_AUTHORIZES_COMPLETION = False
REFILL_AUTHORIZES_EXECUTION = False


class SymbolicFindingRefillError(ValueError):
    """Base error for invalid refill inputs."""


class RefillBindingError(SymbolicFindingRefillError):
    """Raised when repository, tree, policy, or forest binding is incomplete."""


class RefillAncestryError(SymbolicFindingRefillError):
    """Raised when a supplied goal has invalid or excessive ancestry."""


class RefillReason(str, Enum):
    REFILLED = "refilled"
    THRESHOLD_SATISFIED = "threshold_satisfied"
    OPEN_WORK_CEILING = "open_work_ceiling"
    COOLDOWN = "cooldown"
    NO_FRESH_RECEIPTS = "no_fresh_receipts"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    HEALTHY_EXHAUSTED = "healthy_exhausted"


class FindingDisposition(str, Enum):
    MATERIALIZED = "materialized"
    REVIEW_MATERIALIZED = "review_materialized"
    REPLAY = "replay"
    UNCHANGED_BACKOFF = "unchanged_backoff"
    STALE_RECEIPT = "stale_receipt"
    MISSING_FINDING = "missing_finding"
    REJECTED = "rejected"
    STALE = "stale"
    AMBIGUOUS = "ambiguous"
    UNACTIONABLE = "unactionable"
    UNBOUND = "unbound"
    IMPRECISE_SCOPE = "imprecise_scope"
    CHILD_LIMIT = "child_limit"
    DEPTH_LIMIT = "depth_limit"
    SURPLUS_LIMIT = "surplus_limit"
    OPEN_WORK_LIMIT = "open_work_limit"
    DEPENDENCY_REJECTED = "dependency_rejected"
    DEPENDENCY_CYCLE = "dependency_cycle"
    HEALTHY_EXHAUSTED = "healthy_exhausted"


class TaskKind(str, Enum):
    REPAIR = "repair"
    UNBLOCK_REVIEW = "unblock_review"


class FindingLedgerReader(Protocol):
    """Minimal ledger interface needed by the planner."""

    def get(self, finding_cid: str) -> ContractFindingRecord | None:
        ...


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SymbolicFindingRefillError(f"{name} is required")
    return text


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _unique(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    normalized = posixpath.normpath(raw)
    return "" if normalized == "." else normalized


def _safe_output_paths(
    record: ContractFindingRecord,
    *,
    roots: Sequence[str],
    maximum: int,
) -> tuple[str, ...]:
    candidates: list[str] = []
    for value in record.remediation_scope:
        candidate = _path(value)
        if candidate and (
            "/" in candidate
            or candidate.endswith((".py", ".js", ".ts", ".tsx", ".json", ".md"))
        ):
            candidates.append(candidate)
    for step in record.call_slice.steps:
        candidate = _path(step.path)
        if candidate:
            candidates.append(candidate)
    outputs = _unique(candidates)
    if not outputs or len(outputs) > maximum:
        return ()
    normalized_roots = tuple(_path(root).rstrip("/") for root in roots if _path(root))
    for output in outputs:
        if (
            output.startswith("/")
            or output == ".."
            or output.startswith("../")
            or "/../" in f"/{output}/"
        ):
            return ()
        if normalized_roots and not any(
            output == root or output.startswith(root + "/")
            for root in normalized_roots
        ):
            return ()
    return outputs


@dataclass(frozen=True)
class RefillBinding:
    """Exact authority coordinates copied onto every emitted packet."""

    repository_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    objective_forest_id: str
    objective_forest_revision: str
    refinement_goal_id: str

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "objective_forest_id",
            "objective_forest_revision",
            "refinement_goal_id",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))

    @property
    def binding_id(self) -> str:
        return _stable_id("refill-binding", self.to_record())

    def to_record(self) -> dict[str, str]:
        return {
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "objective_forest_id": self.objective_forest_id,
            "objective_forest_revision": self.objective_forest_revision,
            "refinement_goal_id": self.refinement_goal_id,
        }


@dataclass(frozen=True)
class RefillGoal:
    """An existing or newly proposed goal with explicit ancestry."""

    goal_id: str
    title: str
    root_cause_family: str = ""
    semantic_key: str = ""
    parent_goal_id: str = ""
    ancestor_goal_ids: tuple[str, ...] = ()
    depth: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))
        object.__setattr__(self, "title", _required_text(self.title, "title"))
        object.__setattr__(
            self, "root_cause_family", str(self.root_cause_family or "").strip()
        )
        object.__setattr__(self, "semantic_key", str(self.semantic_key or "").strip())
        object.__setattr__(
            self, "parent_goal_id", str(self.parent_goal_id or "").strip()
        )
        ancestors = tuple(str(value).strip() for value in self.ancestor_goal_ids)
        if any(not value for value in ancestors) or len(set(ancestors)) != len(ancestors):
            raise RefillAncestryError("goal ancestry must be non-empty and acyclic")
        object.__setattr__(self, "ancestor_goal_ids", ancestors)
        if not isinstance(self.depth, int) or self.depth < 0:
            raise RefillAncestryError("goal depth must be a non-negative integer")
        if self.depth != len(ancestors):
            raise RefillAncestryError("goal depth must equal ancestry length")
        if self.depth == 0:
            if self.parent_goal_id:
                raise RefillAncestryError("a root goal cannot have a parent")
        elif not self.parent_goal_id or ancestors[-1] != self.parent_goal_id:
            raise RefillAncestryError("goal parent must be the final ancestor")
        if self.goal_id in ancestors:
            raise RefillAncestryError("goal cannot be its own ancestor")

    def to_record(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "root_cause_family": self.root_cause_family,
            "semantic_key": self.semantic_key,
            "parent_goal_id": self.parent_goal_id,
            "ancestor_goal_ids": self.ancestor_goal_ids,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class RefillTask:
    """A bounded board task proposal."""

    task_id: str
    semantic_key: str
    finding_semantic_key: str
    kind: TaskKind
    goal_id: str
    parent_goal_id: str
    ancestor_goal_ids: tuple[str, ...]
    root_cause_family: str
    finding_cids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    depends_on: tuple[str, ...]
    output_paths: tuple[str, ...]
    semantic_effects: tuple[str, ...]
    validation_commands: tuple[str, ...]
    repository_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    objective_forest_id: str
    objective_forest_revision: str
    status: str = "open"
    attempts: int = 0
    write_authorized: bool = False

    @property
    def open(self) -> bool:
        return self.status.casefold() in {
            "open",
            "ready",
            "pending",
            "in_progress",
            "blocked",
        }

    def to_record(self) -> dict[str, Any]:
        result = dict(vars(self))
        result["kind"] = self.kind.value
        result["open"] = self.open
        return result


@dataclass(frozen=True)
class RefillDiagnostic:
    """Retained evidence that did not create executable work."""

    receipt_id: str
    finding_cid: str
    semantic_key: str
    disposition: FindingDisposition
    reasons: tuple[str, ...] = ()
    retained: bool = True


@dataclass(frozen=True)
class HealthyExhaustionReceipt:
    """Conclusive, fully bound proof that the finding source is exhausted."""

    repository_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    objective_forest_id: str
    objective_forest_revision: str
    conclusive: bool
    healthy: bool
    coverage_complete: bool
    evidence_cids: tuple[str, ...] = ()

    @property
    def receipt_id(self) -> str:
        return _stable_id(
            "healthy-exhaustion",
            {
                "repository_id": self.repository_id,
                "tree_id": self.tree_id,
                "policy_id": self.policy_id,
                "policy_revision": self.policy_revision,
                "objective_forest_id": self.objective_forest_id,
                "objective_forest_revision": self.objective_forest_revision,
                "conclusive": self.conclusive,
                "healthy": self.healthy,
                "coverage_complete": self.coverage_complete,
                "evidence_cids": _unique(self.evidence_cids),
            },
        )

    def matches(self, binding: RefillBinding) -> bool:
        return (
            self.conclusive
            and self.healthy
            and self.coverage_complete
            and self.repository_id == binding.repository_id
            and self.tree_id == binding.tree_id
            and self.policy_id == binding.policy_id
            and self.policy_revision == binding.policy_revision
            and self.objective_forest_id == binding.objective_forest_id
            and self.objective_forest_revision == binding.objective_forest_revision
            and bool(self.evidence_cids)
        )


@dataclass(frozen=True)
class RefillState:
    """Caller-persisted replay and cooldown state."""

    last_sequence: int = -1
    last_refill_epoch: int = 0
    next_allowed_epoch: int = 0
    seen_receipt_ids: tuple[str, ...] = ()
    semantic_task_ids: tuple[tuple[str, str], ...] = ()
    diagnostic_states: tuple[tuple[str, str, int], ...] = ()
    review_task_ids: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.last_sequence < -1:
            raise SymbolicFindingRefillError("last_sequence cannot be below -1")
        object.__setattr__(self, "seen_receipt_ids", _unique(self.seen_receipt_ids))
        for name in ("semantic_task_ids", "review_task_ids"):
            pairs = tuple(sorted((str(key), str(value)) for key, value in getattr(self, name)))
            if len({key for key, _ in pairs}) != len(pairs):
                raise SymbolicFindingRefillError(f"{name} contains duplicate keys")
            object.__setattr__(self, name, pairs)
        diagnostic_states = tuple(
            sorted((str(key), str(signature), int(count)) for key, signature, count in self.diagnostic_states)
        )
        if any(count < 1 for _, _, count in diagnostic_states):
            raise SymbolicFindingRefillError("diagnostic counts must be positive")
        object.__setattr__(self, "diagnostic_states", diagnostic_states)


@dataclass(frozen=True)
class SymbolicFindingRefillPolicy:
    refill_threshold: int = DEFAULT_REFILL_THRESHOLD
    open_work_ceiling: int = DEFAULT_OPEN_WORK_CEILING
    max_findings_per_pass: int = DEFAULT_MAX_FINDINGS_PER_PASS
    max_surplus_per_goal: int = DEFAULT_MAX_SURPLUS_PER_GOAL
    max_children: int = DEFAULT_MAX_CHILDREN
    max_goal_depth: int = DEFAULT_MAX_GOAL_DEPTH
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_output_paths: int = DEFAULT_MAX_OUTPUT_PATHS
    output_roots: tuple[str, ...] = ("ipfs_accelerate_py", "test")
    validation_commands: tuple[str, ...] = (
        "python -m pytest test/api/test_agent_supervisor_symbolic_finding_refill.py -q",
    )

    def __post_init__(self) -> None:
        positive = (
            "refill_threshold",
            "open_work_ceiling",
            "max_findings_per_pass",
            "max_surplus_per_goal",
            "max_children",
            "max_goal_depth",
            "max_retries",
            "max_output_paths",
        )
        if any(not isinstance(getattr(self, name), int) or getattr(self, name) < 1 for name in positive):
            raise SymbolicFindingRefillError("refill policy bounds must be positive integers")
        if self.refill_threshold > self.open_work_ceiling:
            raise SymbolicFindingRefillError("refill threshold exceeds open-work ceiling")
        if self.max_children > 3 or self.max_goal_depth > 4:
            raise SymbolicFindingRefillError("goal refinement exceeds hard safety ceiling")
        if self.max_findings_per_pass > 8 or self.max_surplus_per_goal > 2:
            raise SymbolicFindingRefillError("refill breadth exceeds hard safety ceiling")
        if not isinstance(self.cooldown_seconds, int) or self.cooldown_seconds < 0:
            raise SymbolicFindingRefillError("cooldown_seconds must be non-negative")
        object.__setattr__(self, "output_roots", _unique(self.output_roots))
        object.__setattr__(
            self, "validation_commands", _unique(self.validation_commands)
        )


@dataclass(frozen=True)
class RefillOutcome:
    reason: RefillReason
    binding: RefillBinding
    new_goals: tuple[RefillGoal, ...] = ()
    new_tasks: tuple[RefillTask, ...] = ()
    diagnostics: tuple[RefillDiagnostic, ...] = ()
    processed_receipt_ids: tuple[str, ...] = ()
    state: RefillState = field(default_factory=RefillState)
    open_work_before: int = 0
    refill_epoch_id: str = ""
    idempotency_id: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.new_goals or self.new_tasks)


@dataclass
class _Candidate:
    receipt: AppendReceipt
    finding: ContractFindingRecord
    goal: RefillGoal
    task_kind: TaskKind
    outputs: tuple[str, ...]
    dependency_keys: tuple[str, ...]
    task_id: str
    task_semantic_key: str


def _goal_from_any(value: RefillGoal | Mapping[str, Any]) -> RefillGoal:
    if isinstance(value, RefillGoal):
        return value
    return RefillGoal(
        goal_id=value.get("goal_id", ""),
        title=value.get("title", value.get("goal_id", "")),
        root_cause_family=value.get("root_cause_family", ""),
        semantic_key=value.get("semantic_key", ""),
        parent_goal_id=value.get("parent_goal_id", ""),
        ancestor_goal_ids=tuple(value.get("ancestor_goal_ids") or ()),
        depth=int(value.get("depth", len(value.get("ancestor_goal_ids") or ()))),
    )


def _task_open(value: RefillTask | Mapping[str, Any]) -> bool:
    if isinstance(value, RefillTask):
        return value.open
    return str(value.get("status", "open")).casefold() in {
        "open", "ready", "pending", "in_progress", "blocked"
    }


def _task_value(value: RefillTask | Mapping[str, Any], name: str, default: Any = "") -> Any:
    return getattr(value, name, default) if not isinstance(value, Mapping) else value.get(name, default)


def _finding_bound(record: ContractFindingRecord, binding: RefillBinding) -> bool:
    return (
        record.repositories == (binding.repository_id,)
        and record.tree_id == binding.tree_id
        and record.policy_revision == binding.policy_revision
    )


def _validate_goal_forest(
    goals: Sequence[RefillGoal], policy: SymbolicFindingRefillPolicy
) -> None:
    by_id = {goal.goal_id: goal for goal in goals}
    if len(by_id) != len(goals):
        raise RefillAncestryError("goal forest contains duplicate goal ids")
    child_counts: dict[str, int] = {}
    for goal in goals:
        if goal.depth > policy.max_goal_depth:
            raise RefillAncestryError("goal forest exceeds maximum depth")
        if goal.depth == 0:
            continue
        parent = by_id.get(goal.parent_goal_id)
        if parent is None:
            raise RefillAncestryError("goal parent is absent from the goal forest")
        expected_ancestors = (*parent.ancestor_goal_ids, parent.goal_id)
        if goal.ancestor_goal_ids != expected_ancestors:
            raise RefillAncestryError("goal ancestry does not match its parent chain")
        child_counts[parent.goal_id] = child_counts.get(parent.goal_id, 0) + 1
    if any(count > policy.max_children for count in child_counts.values()):
        raise RefillAncestryError("goal forest exceeds the maximum child count")


def _diagnostic_signature(
    disposition: FindingDisposition, reasons: Sequence[str]
) -> str:
    return _stable_id(
        "refill-diagnostic",
        {"disposition": disposition.value, "reasons": sorted(reasons)},
    )


def _is_ambiguous(record: ContractFindingRecord) -> bool:
    return _enum_text(record.status) in {
        "ambiguous", "suspected", "unsupported", "inconclusive"
    }


def _is_stale(record: ContractFindingRecord) -> bool:
    return _enum_text(record.freshness) == "stale" or _enum_text(record.status) == "stale"


def _goal_for_family(
    family: str,
    goals: Sequence[RefillGoal],
    proposed: list[RefillGoal],
    binding: RefillBinding,
    policy: SymbolicFindingRefillPolicy,
) -> tuple[RefillGoal | None, FindingDisposition | None]:
    matches = sorted(
        (
            goal
            for goal in (*goals, *proposed)
            if goal.root_cause_family == family
        ),
        key=lambda goal: goal.goal_id,
    )
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, FindingDisposition.AMBIGUOUS
    parent = next(
        (goal for goal in goals if goal.goal_id == binding.refinement_goal_id),
        None,
    )
    if parent is None:
        raise RefillAncestryError("refinement_goal_id is absent from the goal forest")
    child_count = sum(
        goal.parent_goal_id == parent.goal_id for goal in (*goals, *proposed)
    )
    if child_count >= policy.max_children:
        return None, FindingDisposition.CHILD_LIMIT
    depth = parent.depth + 1
    if depth > policy.max_goal_depth:
        return None, FindingDisposition.DEPTH_LIMIT
    semantic_key = _stable_id(
        "refill-goal-semantic",
        {
            "family": family,
            "parent_goal_id": parent.goal_id,
            "binding_id": binding.binding_id,
        },
    )
    child = RefillGoal(
        goal_id=_stable_id("goal", {"semantic_key": semantic_key}),
        title=f"Repair symbolic finding family: {family}",
        root_cause_family=family,
        semantic_key=semantic_key,
        parent_goal_id=parent.goal_id,
        ancestor_goal_ids=(*parent.ancestor_goal_ids, parent.goal_id),
        depth=depth,
    )
    proposed.append(child)
    return child, None


def _topological_candidates(
    candidates: Sequence[_Candidate],
    existing_by_finding_key: Mapping[str, str],
) -> tuple[list[_Candidate], dict[str, FindingDisposition]]:
    by_key = {candidate.finding.semantic_key_id: candidate for candidate in candidates}
    rejected: dict[str, FindingDisposition] = {}
    indegree: dict[str, int] = {}
    edges: dict[str, set[str]] = {key: set() for key in by_key}
    for key, candidate in by_key.items():
        deps = set(candidate.dependency_keys)
        if key in deps or any(dep not in by_key and dep not in existing_by_finding_key for dep in deps):
            rejected[key] = FindingDisposition.DEPENDENCY_REJECTED
            continue
        candidate_deps = {dep for dep in deps if dep in by_key}
        indegree[key] = len(candidate_deps)
        for dep in candidate_deps:
            edges[dep].add(key)
    ready = sorted(key for key, degree in indegree.items() if degree == 0 and key not in rejected)
    ordered: list[_Candidate] = []
    while ready:
        key = ready.pop(0)
        ordered.append(by_key[key])
        for dependent in sorted(edges[key]):
            if dependent in rejected:
                continue
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    emitted = {candidate.finding.semantic_key_id for candidate in ordered}
    for key in by_key:
        if key not in emitted and key not in rejected:
            rejected[key] = FindingDisposition.DEPENDENCY_CYCLE
    return ordered, rejected


def refill_symbolic_findings(
    *,
    receipts: Sequence[AppendReceipt],
    ledger: FindingLedgerReader,
    binding: RefillBinding,
    goals: Sequence[RefillGoal | Mapping[str, Any]],
    tasks: Sequence[RefillTask | Mapping[str, Any]] = (),
    state: RefillState | None = None,
    policy: SymbolicFindingRefillPolicy | None = None,
    now_epoch: int | None = None,
    dependencies: Mapping[str, Sequence[str]] | None = None,
    healthy_exhaustion: HealthyExhaustionReceipt | None = None,
) -> RefillOutcome:
    """Plan one bounded refill pass.

    ``dependencies`` is keyed by finding semantic key (or finding CID), and its
    values are prerequisite finding semantic keys.  Missing prerequisites and
    cycles reject the affected candidates rather than weakening the DAG.
    """

    state = state or RefillState()
    policy = policy or SymbolicFindingRefillPolicy()
    now = int(time.time()) if now_epoch is None else int(now_epoch)
    normalized_goals = tuple(_goal_from_any(goal) for goal in goals)
    _validate_goal_forest(normalized_goals, policy)
    if not any(goal.goal_id == binding.refinement_goal_id for goal in normalized_goals):
        raise RefillAncestryError("binding refinement goal is absent")

    open_work = sum(_task_open(task) for task in tasks)
    empty = dict(
        binding=binding,
        state=state,
        open_work_before=open_work,
    )
    if open_work >= policy.open_work_ceiling:
        return RefillOutcome(reason=RefillReason.OPEN_WORK_CEILING, **empty)
    if open_work >= policy.refill_threshold:
        return RefillOutcome(reason=RefillReason.THRESHOLD_SATISFIED, **empty)
    if now < state.next_allowed_epoch:
        return RefillOutcome(reason=RefillReason.COOLDOWN, **empty)

    semantic_task_ids = dict(state.semantic_task_ids)
    review_task_ids = dict(state.review_task_ids)
    diagnostic_states = {
        key: (signature, count)
        for key, signature, count in state.diagnostic_states
    }
    exhausted_finding_keys: set[str] = set()
    for task in tasks:
        finding_key = str(_task_value(task, "finding_semantic_key", "") or "")
        status = str(_task_value(task, "status", "") or "").casefold()
        attempts = int(_task_value(task, "attempts", 0) or 0)
        if (
            finding_key
            and status in {"failed", "retry_exhausted"}
            and attempts >= policy.max_retries
        ):
            exhausted_finding_keys.add(finding_key)
        task_kind = _enum_text(_task_value(task, "kind", "")).casefold()
        task_id = str(_task_value(task, "task_id", "") or "")
        if finding_key and task_id and task_kind == TaskKind.UNBLOCK_REVIEW.value:
            review_task_ids.setdefault(finding_key, task_id)

    ordered_receipts = sorted(receipts, key=lambda item: (item.sequence, item.receipt_id))
    seen = set(state.seen_receipt_ids)
    fresh_receipts = [
        receipt
        for receipt in ordered_receipts
        if receipt.receipt_id not in seen and receipt.sequence > state.last_sequence
    ]
    # Retry exhaustion normally occurs after the admitted receipt was consumed
    # by an earlier refill.  That receipt remains the provenance for the single
    # bounded review task and must not be hidden by replay filtering.
    exhausted_receipts = [
        receipt
        for receipt in ordered_receipts
        if receipt.semantic_key_id in exhausted_finding_keys
        and receipt.semantic_key_id not in review_task_ids
    ]

    if (
        not fresh_receipts
        and not exhausted_receipts
        and healthy_exhaustion
        and healthy_exhaustion.matches(binding)
        and healthy_exhaustion.receipt_id not in seen
    ):
        seen = _unique((*state.seen_receipt_ids, healthy_exhaustion.receipt_id))
        next_state = RefillState(
            last_sequence=state.last_sequence,
            last_refill_epoch=now,
            next_allowed_epoch=now + policy.cooldown_seconds,
            seen_receipt_ids=seen,
            semantic_task_ids=state.semantic_task_ids,
            diagnostic_states=state.diagnostic_states,
            review_task_ids=state.review_task_ids,
        )
        diagnostic = RefillDiagnostic(
            receipt_id=healthy_exhaustion.receipt_id,
            finding_cid="",
            semantic_key="",
            disposition=FindingDisposition.HEALTHY_EXHAUSTED,
            reasons=("conclusive_healthy_exhaustion",),
        )
        return _finish(
            reason=RefillReason.HEALTHY_EXHAUSTED,
            binding=binding,
            state=next_state,
            diagnostics=(diagnostic,),
            processed=(healthy_exhaustion.receipt_id,),
            open_work=open_work,
            now=now,
        )

    # Always admit fresh evidence ahead of an arbitrarily long replay prefix.
    # When there is no fresh receipt, select old receipts so repeated unchanged
    # diagnostics can be retained with deterministic exponential backoff.
    selected = (
        fresh_receipts[: policy.max_findings_per_pass]
        if fresh_receipts
        else (
            exhausted_receipts[: policy.max_findings_per_pass]
            if exhausted_receipts
            else ordered_receipts[: policy.max_findings_per_pass]
        )
    )
    if not selected:
        return RefillOutcome(reason=RefillReason.NO_FRESH_RECEIPTS, **empty)

    existing_by_finding_key = {
        key: task_id
        for key, task_id in semantic_task_ids.items()
        if key not in exhausted_finding_keys
    }
    for task in tasks:
        finding_key = str(_task_value(task, "finding_semantic_key", "") or "")
        task_id = str(_task_value(task, "task_id", "") or "")
        if finding_key and task_id and finding_key not in exhausted_finding_keys:
            existing_by_finding_key[finding_key] = task_id

    diagnostics: list[RefillDiagnostic] = []
    proposed_goals: list[RefillGoal] = []
    candidates: list[_Candidate] = []
    processed: list[str] = []
    max_sequence = state.last_sequence
    baseline_remaining = max(0, policy.refill_threshold - open_work)
    surplus_by_goal: dict[str, int] = {}
    dependency_map = dependencies or {}

    def retain(
        receipt: AppendReceipt,
        semantic_key: str,
        disposition: FindingDisposition,
        *reasons: str,
    ) -> None:
        nonlocal diagnostics
        signature = _diagnostic_signature(disposition, reasons)
        previous = diagnostic_states.get(semantic_key or receipt.receipt_id)
        effective = disposition
        count = 1
        if previous and previous[0] == signature:
            count = previous[1] + 1
            effective = FindingDisposition.UNCHANGED_BACKOFF
        diagnostic_states[semantic_key or receipt.receipt_id] = (signature, count)
        diagnostics.append(
            RefillDiagnostic(
                receipt_id=receipt.receipt_id,
                finding_cid=receipt.finding_cid,
                semantic_key=semantic_key,
                disposition=effective,
                reasons=tuple(reasons),
            )
        )

    for receipt in selected:
        receipt_id = receipt.receipt_id
        semantic_key = receipt.semantic_key_id
        retry_review = (
            semantic_key in exhausted_finding_keys
            and semantic_key not in review_task_ids
        )
        if receipt_id in seen and not retry_review:
            retain(receipt, semantic_key, FindingDisposition.REPLAY, "receipt_replay")
            continue
        processed.append(receipt_id)
        seen.add(receipt_id)
        max_sequence = max(max_sequence, receipt.sequence)
        if receipt.sequence <= state.last_sequence and not retry_review:
            retain(receipt, semantic_key, FindingDisposition.STALE_RECEIPT, "sequence_not_fresh")
            continue
        if receipt.admission is FindingAdmissionState.STALE:
            retain(receipt, semantic_key, FindingDisposition.STALE, "ledger_marked_stale")
            continue
        if (
            not receipt.stored
            or receipt.outcome not in {AppendOutcome.STORED, AppendOutcome.SUPERSEDED_PRIOR}
            or receipt.admission
            not in {FindingAdmissionState.ADMITTED, FindingAdmissionState.SUPERSEDED}
        ):
            retain(receipt, semantic_key, FindingDisposition.REJECTED, "ledger_not_admitted")
            continue
        finding = ledger.get(receipt.finding_cid)
        if finding is None:
            retain(receipt, semantic_key, FindingDisposition.MISSING_FINDING, "finding_not_in_ledger")
            continue
        semantic_key = finding.semantic_key_id
        if receipt.semantic_key_id and receipt.semantic_key_id != semantic_key:
            retain(receipt, semantic_key, FindingDisposition.REJECTED, "receipt_semantic_key_mismatch")
            continue
        exhausted = semantic_key in exhausted_finding_keys
        if semantic_key in existing_by_finding_key and not exhausted:
            retain(receipt, semantic_key, FindingDisposition.REPLAY, "semantic_task_exists")
            continue
        if _is_stale(finding):
            retain(receipt, semantic_key, FindingDisposition.STALE, "stale_evidence")
            continue
        if _is_ambiguous(finding):
            retain(receipt, semantic_key, FindingDisposition.AMBIGUOUS, "evidence_not_conclusive")
            continue
        if not finding.actionable:
            retain(receipt, semantic_key, FindingDisposition.UNACTIONABLE, "finding_not_actionable")
            continue
        if not _finding_bound(finding, binding):
            retain(receipt, semantic_key, FindingDisposition.UNBOUND, "repository_tree_or_policy_mismatch")
            continue
        outputs = _safe_output_paths(
            finding, roots=policy.output_roots, maximum=policy.max_output_paths
        )
        if not outputs:
            retain(receipt, semantic_key, FindingDisposition.IMPRECISE_SCOPE, "output_scope_not_bounded")
            continue
        goal, rejected = _goal_for_family(
            finding.root_cause_family,
            normalized_goals,
            proposed_goals,
            binding,
            policy,
        )
        if goal is None:
            retain(
                receipt,
                semantic_key,
                rejected or FindingDisposition.CHILD_LIMIT,
                "goal_refinement_rejected",
            )
            continue
        if open_work + len(candidates) >= policy.open_work_ceiling:
            retain(receipt, semantic_key, FindingDisposition.OPEN_WORK_LIMIT, "open_work_ceiling")
            continue
        if baseline_remaining:
            baseline_remaining -= 1
        else:
            surplus = surplus_by_goal.get(goal.goal_id, 0)
            if surplus >= policy.max_surplus_per_goal:
                retain(receipt, semantic_key, FindingDisposition.SURPLUS_LIMIT, "per_goal_surplus_limit")
                continue
            surplus_by_goal[goal.goal_id] = surplus + 1

        kind = TaskKind.UNBLOCK_REVIEW if exhausted else TaskKind.REPAIR
        if exhausted and semantic_key in review_task_ids:
            retain(receipt, semantic_key, FindingDisposition.REPLAY, "bounded_review_exists")
            continue
        task_semantic_key = _stable_id(
            "refill-task-semantic",
            {
                "finding_semantic_key": semantic_key,
                "goal_id": goal.goal_id,
                "kind": kind.value,
                "binding_id": binding.binding_id,
            },
        )
        task_id = _stable_id("task", {"semantic_key": task_semantic_key})
        deps = dependency_map.get(semantic_key, dependency_map.get(finding.finding_cid, ()))
        candidates.append(
            _Candidate(
                receipt=receipt,
                finding=finding,
                goal=goal,
                task_kind=kind,
                outputs=outputs,
                dependency_keys=_unique(tuple(deps)),
                task_id=task_id,
                task_semantic_key=task_semantic_key,
            )
        )

    ordered_candidates, dependency_rejections = _topological_candidates(
        candidates, existing_by_finding_key
    )
    for candidate in candidates:
        disposition = dependency_rejections.get(candidate.finding.semantic_key_id)
        if disposition:
            retain(
                candidate.receipt,
                candidate.finding.semantic_key_id,
                disposition,
                "dependency_dag_rejected",
            )

    emitted_ids: dict[str, str] = dict(existing_by_finding_key)
    new_tasks: list[RefillTask] = []
    for candidate in ordered_candidates:
        dependency_ids = tuple(
            emitted_ids[key] for key in candidate.dependency_keys if key in emitted_ids
        )
        finding = candidate.finding
        effects = _unique(
            (
                f"restore:{finding.root_cause_family}",
                f"satisfy_expected:{finding.expected_contract_cid}",
                f"align_observed:{finding.observed_contract_cid}",
                *(f"preserve_interface:{value}" for value in finding.interfaces),
            )
        )
        task = RefillTask(
            task_id=candidate.task_id,
            semantic_key=candidate.task_semantic_key,
            finding_semantic_key=finding.semantic_key_id,
            kind=candidate.task_kind,
            goal_id=candidate.goal.goal_id,
            parent_goal_id=candidate.goal.parent_goal_id,
            ancestor_goal_ids=candidate.goal.ancestor_goal_ids,
            root_cause_family=finding.root_cause_family,
            finding_cids=(finding.finding_cid,),
            receipt_ids=(candidate.receipt.receipt_id,),
            depends_on=dependency_ids,
            output_paths=candidate.outputs,
            semantic_effects=effects,
            validation_commands=policy.validation_commands,
            repository_id=binding.repository_id,
            tree_id=binding.tree_id,
            policy_id=binding.policy_id,
            policy_revision=binding.policy_revision,
            objective_forest_id=binding.objective_forest_id,
            objective_forest_revision=binding.objective_forest_revision,
        )
        new_tasks.append(task)
        emitted_ids[finding.semantic_key_id] = task.task_id
        semantic_task_ids[finding.semantic_key_id] = task.task_id
        if task.kind is TaskKind.UNBLOCK_REVIEW:
            review_task_ids[finding.semantic_key_id] = task.task_id
        diagnostics.append(
            RefillDiagnostic(
                receipt_id=candidate.receipt.receipt_id,
                finding_cid=finding.finding_cid,
                semantic_key=finding.semantic_key_id,
                disposition=(
                    FindingDisposition.REVIEW_MATERIALIZED
                    if task.kind is TaskKind.UNBLOCK_REVIEW
                    else FindingDisposition.MATERIALIZED
                ),
                reasons=(),
            )
        )

    used_goal_ids = {task.goal_id for task in new_tasks}
    proposed_goals = [goal for goal in proposed_goals if goal.goal_id in used_goal_ids]
    max_backoff_count = max(
        (count for _, count in diagnostic_states.values()), default=1
    )
    backoff_factor = 2 ** min(max_backoff_count - 1, 5)
    next_state = RefillState(
        last_sequence=max_sequence,
        last_refill_epoch=now,
        next_allowed_epoch=now + policy.cooldown_seconds * backoff_factor,
        seen_receipt_ids=tuple(seen),
        semantic_task_ids=tuple(semantic_task_ids.items()),
        diagnostic_states=tuple(
            (key, signature, count)
            for key, (signature, count) in diagnostic_states.items()
        ),
        review_task_ids=tuple(review_task_ids.items()),
    )
    if new_tasks:
        reason = RefillReason.REFILLED
    elif diagnostics:
        reason = RefillReason.DIAGNOSTICS_ONLY
    else:
        reason = RefillReason.NO_FRESH_RECEIPTS
    return _finish(
        reason=reason,
        binding=binding,
        state=next_state,
        goals=tuple(proposed_goals),
        tasks=tuple(new_tasks),
        diagnostics=tuple(diagnostics),
        processed=tuple(processed),
        open_work=open_work,
        now=now,
    )


def _finish(
    *,
    reason: RefillReason,
    binding: RefillBinding,
    state: RefillState,
    goals: tuple[RefillGoal, ...] = (),
    tasks: tuple[RefillTask, ...] = (),
    diagnostics: tuple[RefillDiagnostic, ...] = (),
    processed: tuple[str, ...] = (),
    open_work: int,
    now: int,
) -> RefillOutcome:
    material = {
        "schema": SYMBOLIC_REFILL_EPOCH_SCHEMA,
        "binding_id": binding.binding_id,
        "now_epoch": now,
        "processed_receipt_ids": sorted(processed),
        "goal_ids": sorted(goal.goal_id for goal in goals),
        "task_ids": sorted(task.task_id for task in tasks),
        "reason": reason.value,
    }
    return RefillOutcome(
        reason=reason,
        binding=binding,
        new_goals=goals,
        new_tasks=tasks,
        diagnostics=diagnostics,
        processed_receipt_ids=processed,
        state=state,
        open_work_before=open_work,
        refill_epoch_id=_stable_id("refill-epoch", material),
        idempotency_id=_stable_id(
            "refill-idempotency",
            {
                "schema": REFILL_IDEMPOTENCY_SCHEMA,
                "binding_id": binding.binding_id,
                "receipt_ids": sorted(processed),
                "task_ids": sorted(task.task_id for task in tasks),
            },
        ),
    )


class SymbolicFindingRefiller:
    """Reusable policy wrapper for :func:`refill_symbolic_findings`."""

    def __init__(self, policy: SymbolicFindingRefillPolicy | None = None) -> None:
        self.policy = policy or SymbolicFindingRefillPolicy()

    def refill(self, **kwargs: Any) -> RefillOutcome:
        return refill_symbolic_findings(policy=self.policy, **kwargs)


# Concise compatibility alias for callers that treat refill as a pure planner.
plan_symbolic_finding_refill = refill_symbolic_findings


__all__ = [
    "DEFAULT_COOLDOWN_SECONDS",
    "DEFAULT_MAX_CHILDREN",
    "DEFAULT_MAX_FINDINGS_PER_PASS",
    "DEFAULT_MAX_GOAL_DEPTH",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_SURPLUS_PER_GOAL",
    "DEFAULT_OPEN_WORK_CEILING",
    "DEFAULT_REFILL_THRESHOLD",
    "FindingDisposition",
    "HealthyExhaustionReceipt",
    "REFILL_AUTHORIZES_COMPLETION",
    "REFILL_AUTHORIZES_EXECUTION",
    "REFILL_IDEMPOTENCY_SCHEMA",
    "RefillAncestryError",
    "RefillBinding",
    "RefillBindingError",
    "RefillDiagnostic",
    "RefillGoal",
    "RefillOutcome",
    "RefillReason",
    "RefillState",
    "RefillTask",
    "SYMBOLIC_FINDING_REFILL_INTERFACE",
    "SYMBOLIC_FINDING_REFILL_VERSION",
    "SYMBOLIC_REFILL_EPOCH_SCHEMA",
    "SymbolicFindingRefillError",
    "SymbolicFindingRefillPolicy",
    "SymbolicFindingRefiller",
    "TaskKind",
    "plan_symbolic_finding_refill",
    "refill_symbolic_findings",
]
