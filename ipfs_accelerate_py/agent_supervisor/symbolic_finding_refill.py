"""Bounded, replay-safe backlog refill from symbolic finding receipts.

The finding ledger is the authority for evidence admission.  This module is a
small deterministic planner: when open work is below a threshold it consumes
fresh ledger receipts, binds actionable findings to an exact goal family, and
emits goal/task packets.  It does not mutate the ledger or a task board.

Ambiguous, stale, rejected, unbound, or overly broad evidence is retained in
the returned diagnostics but can never create executable work.  All identities
are derived from semantic inputs so replaying a receipt is a no-op.

Evidence schemas are ``vfs/symbolic-refill-epoch@1`` and
``vfs/refill-idempotency@1`` (goal_packet/autonomous_refill).

Objective-heap ownership for the autonomous-refill packet:

* VFS-G160 / VFS-080 prove ``vfs/symbolic-refill-epoch@1``
* VFS-G161 / VFS-083 prove ``vfs/refill-idempotency@1``
* VFS-G120 remains the parent refill goal; proposals never authorize execution
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Mapping, Protocol, Sequence

from .contract_findings import (
    AppendOutcome,
    AppendReceipt,
    ContractFindingRecord,
    FindingAdmissionState,
)

SYMBOLIC_FINDING_REFILL_VERSION = 1
SYMBOLIC_FINDING_REFILL_INTERFACE = "vfs/symbolic-finding-refill@1"
SYMBOLIC_REFILL_EPOCH_SCHEMA: Final = "vfs/symbolic-refill-epoch@1"
REFILL_IDEMPOTENCY_SCHEMA: Final = "vfs/refill-idempotency@1"
# Domain evidence identities (alias schemas for objective-heap discovery).
SYMBOLIC_REFILL_EPOCH_EVIDENCE: Final = SYMBOLIC_REFILL_EPOCH_SCHEMA
REFILL_IDEMPOTENCY_EVIDENCE: Final = REFILL_IDEMPOTENCY_SCHEMA
SYMBOLIC_REFILL_EVIDENCE_SCHEMAS: Final[tuple[str, ...]] = (
    SYMBOLIC_REFILL_EPOCH_SCHEMA,
    REFILL_IDEMPOTENCY_SCHEMA,
)

# ---------------------------------------------------------------------------
# Objective-heap discovery anchors (VFS-G160 / VFS-G161 packet)
# goal_packet/autonomous_refill/ipfs_accelerate_py/767f3cfd52ba
# Labels never enter task_id / epoch_id / idempotency_id digests.
# ---------------------------------------------------------------------------
OBJECTIVE_PARENT_GOAL_ID: Final = "VFS-G120"
OBJECTIVE_GOAL_G160_ID: Final = "VFS-G160"
OBJECTIVE_GOAL_G161_ID: Final = "VFS-G161"
OBJECTIVE_TASK_G160_ID: Final = "VFS-080"
OBJECTIVE_TASK_G161_ID: Final = "VFS-083"
OBJECTIVE_TASK_PACKET_ID: Final = "VFS-079"
OBJECTIVE_PACKET_GOAL_IDS: Final[tuple[str, ...]] = (
    OBJECTIVE_GOAL_G160_ID,
    OBJECTIVE_GOAL_G161_ID,
)
OBJECTIVE_DOMAIN_EVIDENCE_TERMS: Final[tuple[str, ...]] = (
    SYMBOLIC_REFILL_EPOCH_EVIDENCE,
    REFILL_IDEMPOTENCY_EVIDENCE,
)
OBJECTIVE_PACKET_EVIDENCE_TERMS: Final[tuple[str, ...]] = (
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS
)
SYMBOLIC_REFILL_EPOCH_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-refill-epoch-claim@1"
)
REFILL_IDEMPOTENCY_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/refill-idempotency-claim@1"
)
AUTONOMOUS_REFILL_PACKET_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/autonomous-refill-packet-claim@1"
)
SYMBOLIC_REFILL_EPOCH_INVARIANTS: Final[tuple[str, ...]] = (
    "only fresh admitted findings produce work",
    "existing goal families are reused under exact binding",
    "new children are bounded by breadth/depth/open-work/cooldown",
    "each decision carries content-addressed prior and result state",
    "wall-clock observation changes epoch id without changing task identity",
    "refill proposals never authorize execution or completion",
)
REFILL_IDEMPOTENCY_INVARIANTS: Final[tuple[str, ...]] = (
    "replay of the same operation is a no-op",
    "unchanged failures back off without re-emitting tasks",
    "taskboard restoration preserves semantic task identity",
    "idempotency id excludes observation time and emitted task ids",
    "conclusive healthy exhaustion creates no busywork",
)

# Keep exact-text discovery anchors aligned with the objective heap.
assert SYMBOLIC_REFILL_EPOCH_SCHEMA == "vfs/symbolic-refill-epoch@1"
assert REFILL_IDEMPOTENCY_SCHEMA == "vfs/refill-idempotency@1"
assert SYMBOLIC_REFILL_EPOCH_EVIDENCE == "vfs/symbolic-refill-epoch@1"
assert REFILL_IDEMPOTENCY_EVIDENCE == "vfs/refill-idempotency@1"
assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G120"
assert OBJECTIVE_GOAL_G160_ID == "VFS-G160"
assert OBJECTIVE_GOAL_G161_ID == "VFS-G161"
assert OBJECTIVE_TASK_G160_ID == "VFS-080"
assert OBJECTIVE_TASK_G161_ID == "VFS-083"
assert OBJECTIVE_TASK_PACKET_ID == "VFS-079"
assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == (
    "vfs/symbolic-refill-epoch@1",
    "vfs/refill-idempotency@1",
)
assert OBJECTIVE_PACKET_EVIDENCE_TERMS == OBJECTIVE_DOMAIN_EVIDENCE_TERMS

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
            output.startswith(("/", "../"))
            or output == ".."
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


def _state_id(state: RefillState) -> str:
    """Content identity for caller-persisted refill state."""

    return _stable_id(
        "refill-state",
        {
            "last_sequence": state.last_sequence,
            "last_refill_epoch": state.last_refill_epoch,
            "next_allowed_epoch": state.next_allowed_epoch,
            "seen_receipt_ids": state.seen_receipt_ids,
            "semantic_task_ids": state.semantic_task_ids,
            "diagnostic_states": state.diagnostic_states,
            "review_task_ids": state.review_task_ids,
        },
    )


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
class SymbolicRefillEpochEvidence:
    """Exact, content-addressed account of one refill planner decision.

    The receipt distinguishes the observed epoch from the idempotent operation
    below.  Wall-clock time therefore cannot change a task identity, while the
    supervisor can still prove which persisted state and objective-heap
    revision produced a particular decision.
    """

    binding: RefillBinding
    reason: RefillReason
    observed_at_epoch: int
    prior_state_id: str
    result_state_id: str
    input_receipt_ids: tuple[str, ...] = ()
    fresh_receipt_ids: tuple[str, ...] = ()
    processed_receipt_ids: tuple[str, ...] = ()
    emitted_goal_ids: tuple[str, ...] = ()
    emitted_task_ids: tuple[str, ...] = ()
    diagnostic_dispositions: tuple[str, ...] = ()
    open_work_before: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.observed_at_epoch, int) or self.observed_at_epoch < 0:
            raise SymbolicFindingRefillError(
                "observed_at_epoch must be a non-negative integer"
            )
        if not isinstance(self.open_work_before, int) or self.open_work_before < 0:
            raise SymbolicFindingRefillError(
                "open_work_before must be a non-negative integer"
            )
        for name in (
            "prior_state_id",
            "result_state_id",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        for name in (
            "input_receipt_ids",
            "fresh_receipt_ids",
            "processed_receipt_ids",
            "emitted_goal_ids",
            "emitted_task_ids",
            "diagnostic_dispositions",
        ):
            object.__setattr__(self, name, _unique(getattr(self, name)))
        if not set(self.fresh_receipt_ids).issubset(self.input_receipt_ids):
            raise SymbolicFindingRefillError(
                "fresh receipt ids must be included in input receipt ids"
            )
        if not set(self.processed_receipt_ids).issubset(self.input_receipt_ids):
            raise SymbolicFindingRefillError(
                "processed receipt ids must be included in input receipt ids"
            )

    @property
    def changed(self) -> bool:
        return bool(self.emitted_goal_ids or self.emitted_task_ids)

    @property
    def epoch_id(self) -> str:
        return _stable_id("refill-epoch", self._identity_record())

    def _identity_record(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_REFILL_EPOCH_SCHEMA,
            "binding": self.binding.to_record(),
            "reason": self.reason.value,
            "observed_at_epoch": self.observed_at_epoch,
            "prior_state_id": self.prior_state_id,
            "result_state_id": self.result_state_id,
            "input_receipt_ids": self.input_receipt_ids,
            "fresh_receipt_ids": self.fresh_receipt_ids,
            "processed_receipt_ids": self.processed_receipt_ids,
            "emitted_goal_ids": self.emitted_goal_ids,
            "emitted_task_ids": self.emitted_task_ids,
            "diagnostic_dispositions": self.diagnostic_dispositions,
            "open_work_before": self.open_work_before,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self._identity_record(),
            "epoch_id": self.epoch_id,
            "binding_id": self.binding.binding_id,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class RefillIdempotencyEvidence:
    """Replay witness for the stable work identity of a refill operation."""

    binding: RefillBinding
    epoch_id: str
    operation_receipt_ids: tuple[str, ...] = ()
    emitted_task_ids: tuple[str, ...] = ()
    resolved_task_ids: tuple[str, ...] = ()
    replay_receipt_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch_id", _required_text(self.epoch_id, "epoch_id"))
        for name in (
            "operation_receipt_ids",
            "emitted_task_ids",
            "resolved_task_ids",
            "replay_receipt_ids",
        ):
            object.__setattr__(self, name, _unique(getattr(self, name)))
        if not set(self.replay_receipt_ids).issubset(self.operation_receipt_ids):
            raise SymbolicFindingRefillError(
                "replay receipt ids must be included in operation receipt ids"
            )
        if not set(self.emitted_task_ids).issubset(self.resolved_task_ids):
            raise SymbolicFindingRefillError(
                "emitted task ids must be included in resolved task ids"
            )

    @property
    def idempotency_id(self) -> str:
        # Output task IDs and observation time are deliberately excluded.  The
        # same evidence bound to the same objective revision is one operation,
        # whether this is the materializing pass or a persisted-state replay.
        return _stable_id(
            "refill-idempotency",
            {
                "schema": REFILL_IDEMPOTENCY_SCHEMA,
                "binding_id": self.binding.binding_id,
                "operation_receipt_ids": self.operation_receipt_ids,
            },
        )

    @property
    def replay_noop(self) -> bool:
        return bool(self.replay_receipt_ids) and not self.emitted_task_ids

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": REFILL_IDEMPOTENCY_SCHEMA,
            "idempotency_id": self.idempotency_id,
            "epoch_id": self.epoch_id,
            "binding": self.binding.to_record(),
            "binding_id": self.binding.binding_id,
            "operation_receipt_ids": self.operation_receipt_ids,
            "emitted_task_ids": self.emitted_task_ids,
            "resolved_task_ids": self.resolved_task_ids,
            "replay_receipt_ids": self.replay_receipt_ids,
            "replay_noop": self.replay_noop,
        }


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
    epoch_evidence: SymbolicRefillEpochEvidence | None = None
    idempotency_evidence: RefillIdempotencyEvidence | None = None

    @property
    def changed(self) -> bool:
        return bool(self.new_goals or self.new_tasks)

    @property
    def evidence_methods(self) -> tuple[str, ...]:
        """Schemas directly evidenced by this decision."""

        if self.epoch_evidence is None or self.idempotency_evidence is None:
            return ()
        return SYMBOLIC_REFILL_EVIDENCE_SCHEMAS

    def evidence_records(self) -> tuple[dict[str, Any], ...]:
        if self.epoch_evidence is None or self.idempotency_evidence is None:
            return ()
        return (
            self.epoch_evidence.to_record(),
            self.idempotency_evidence.to_record(),
        )


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

    ordered_receipts = tuple(
        sorted(receipts, key=lambda item: (item.sequence, item.receipt_id))
    )
    seen = set(state.seen_receipt_ids)
    fresh_receipts = tuple(
        receipt
        for receipt in ordered_receipts
        if receipt.receipt_id not in seen and receipt.sequence > state.last_sequence
    )
    gate_receipts = (
        fresh_receipts[: policy.max_findings_per_pass]
        or ordered_receipts[: policy.max_findings_per_pass]
    )
    open_work = sum(_task_open(task) for task in tasks)
    if open_work >= policy.open_work_ceiling:
        return _finish(
            reason=RefillReason.OPEN_WORK_CEILING,
            binding=binding,
            prior_state=state,
            state=state,
            input_receipts=gate_receipts,
            fresh_receipts=tuple(
                receipt for receipt in gate_receipts if receipt in fresh_receipts
            ),
            open_work=open_work,
            now=now,
        )
    if open_work >= policy.refill_threshold:
        return _finish(
            reason=RefillReason.THRESHOLD_SATISFIED,
            binding=binding,
            prior_state=state,
            state=state,
            input_receipts=gate_receipts,
            fresh_receipts=tuple(
                receipt for receipt in gate_receipts if receipt in fresh_receipts
            ),
            open_work=open_work,
            now=now,
        )
    if now < state.next_allowed_epoch:
        return _finish(
            reason=RefillReason.COOLDOWN,
            binding=binding,
            prior_state=state,
            state=state,
            input_receipts=gate_receipts,
            fresh_receipts=tuple(
                receipt for receipt in gate_receipts if receipt in fresh_receipts
            ),
            open_work=open_work,
            now=now,
        )

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
            prior_state=state,
            state=next_state,
            diagnostics=(diagnostic,),
            processed=(healthy_exhaustion.receipt_id,),
            input_receipt_ids=(healthy_exhaustion.receipt_id,),
            fresh_receipt_ids=(healthy_exhaustion.receipt_id,),
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
        return _finish(
            reason=RefillReason.NO_FRESH_RECEIPTS,
            binding=binding,
            prior_state=state,
            state=state,
            open_work=open_work,
            now=now,
        )

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
            # Import the authoritative taskboard identity into the next replay
            # state.  The objective heap and supervisor-fed backlog then share
            # one semantic-task mapping even when the caller restored a task
            # before restoring planner state.
            semantic_task_ids.setdefault(finding_key, task_id)

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
        prior_state=state,
        state=next_state,
        goals=tuple(proposed_goals),
        tasks=tuple(new_tasks),
        diagnostics=tuple(diagnostics),
        processed=tuple(processed),
        input_receipts=selected,
        fresh_receipts=tuple(
            receipt for receipt in selected if receipt in fresh_receipts
        ),
        open_work=open_work,
        now=now,
    )


def _finish(
    *,
    reason: RefillReason,
    binding: RefillBinding,
    prior_state: RefillState,
    state: RefillState,
    goals: tuple[RefillGoal, ...] = (),
    tasks: tuple[RefillTask, ...] = (),
    diagnostics: tuple[RefillDiagnostic, ...] = (),
    processed: tuple[str, ...] = (),
    input_receipts: Sequence[AppendReceipt] = (),
    fresh_receipts: Sequence[AppendReceipt] = (),
    input_receipt_ids: tuple[str, ...] = (),
    fresh_receipt_ids: tuple[str, ...] = (),
    open_work: int,
    now: int,
) -> RefillOutcome:
    operation_receipt_ids = _unique(
        (*input_receipt_ids, *(receipt.receipt_id for receipt in input_receipts))
    )
    operation_fresh_ids = _unique(
        (*fresh_receipt_ids, *(receipt.receipt_id for receipt in fresh_receipts))
    )
    operation_semantic_keys = _unique(
        tuple(receipt.semantic_key_id for receipt in input_receipts)
    )
    diagnostic_dispositions = tuple(
        diagnostic.disposition.value for diagnostic in diagnostics
    )
    epoch_evidence = SymbolicRefillEpochEvidence(
        binding=binding,
        reason=reason,
        observed_at_epoch=now,
        prior_state_id=_state_id(prior_state),
        result_state_id=_state_id(state),
        input_receipt_ids=operation_receipt_ids,
        fresh_receipt_ids=operation_fresh_ids,
        processed_receipt_ids=processed,
        emitted_goal_ids=tuple(goal.goal_id for goal in goals),
        emitted_task_ids=tuple(task.task_id for task in tasks),
        diagnostic_dispositions=diagnostic_dispositions,
        open_work_before=open_work,
    )
    task_ids_by_finding = dict(state.semantic_task_ids)
    resolved_task_ids = _unique(
        tuple(
            task_ids_by_finding[key]
            for key in operation_semantic_keys
            if key in task_ids_by_finding
        )
    )
    emitted_task_ids = tuple(task.task_id for task in tasks)
    resolved_task_ids = _unique((*resolved_task_ids, *emitted_task_ids))
    replay_receipt_ids = _unique(
        tuple(
            diagnostic.receipt_id
            for diagnostic in diagnostics
            if diagnostic.disposition
            in {
                FindingDisposition.REPLAY,
                FindingDisposition.UNCHANGED_BACKOFF,
            }
        )
    )
    idempotency_evidence = RefillIdempotencyEvidence(
        binding=binding,
        epoch_id=epoch_evidence.epoch_id,
        operation_receipt_ids=operation_receipt_ids,
        emitted_task_ids=emitted_task_ids,
        resolved_task_ids=resolved_task_ids,
        replay_receipt_ids=replay_receipt_ids,
    )
    return RefillOutcome(
        reason=reason,
        binding=binding,
        new_goals=goals,
        new_tasks=tasks,
        diagnostics=diagnostics,
        processed_receipt_ids=processed,
        state=state,
        open_work_before=open_work,
        refill_epoch_id=epoch_evidence.epoch_id,
        idempotency_id=idempotency_evidence.idempotency_id,
        epoch_evidence=epoch_evidence,
        idempotency_evidence=idempotency_evidence,
    )


class SymbolicFindingRefiller:
    """Reusable policy wrapper for :func:`refill_symbolic_findings`."""

    def __init__(self, policy: SymbolicFindingRefillPolicy | None = None) -> None:
        self.policy = policy or SymbolicFindingRefillPolicy()

    def refill(self, **kwargs: Any) -> RefillOutcome:
        return refill_symbolic_findings(policy=self.policy, **kwargs)


@dataclass(frozen=True)
class SupervisorBacklogSnapshot:
    """Exact objective-heap/taskboard coordinates consumed by one refill."""

    binding: RefillBinding
    goals: tuple[RefillGoal, ...]
    tasks: tuple[RefillTask | Mapping[str, Any], ...] = ()
    state: RefillState = field(default_factory=RefillState)

    def __post_init__(self) -> None:
        object.__setattr__(self, "goals", tuple(self.goals))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        matching_roots = tuple(
            goal
            for goal in self.goals
            if goal.goal_id == self.binding.refinement_goal_id
        )
        if len(matching_roots) != 1:
            raise SymbolicFindingRefillError(
                "backlog snapshot must contain its exact refinement goal once"
            )

    @property
    def evidence_methods(self) -> tuple[str, ...]:
        return SYMBOLIC_REFILL_EVIDENCE_SCHEMAS


class BacklogRefinery:
    """Adapt a supervisor-fed objective snapshot to a bounded proposal."""

    def __init__(
        self,
        ledger: FindingLedgerReader,
        policy: SymbolicFindingRefillPolicy | None = None,
    ) -> None:
        self.ledger = ledger
        self.policy = policy or SymbolicFindingRefillPolicy()

    def refill(
        self,
        snapshot: SupervisorBacklogSnapshot,
        receipts: Sequence[AppendReceipt],
        *,
        now_epoch: int | None = None,
        dependencies: Mapping[str, Sequence[str]] | None = None,
        healthy_exhaustion: HealthyExhaustionReceipt | None = None,
    ) -> RefillOutcome:
        """Propose work against exactly ``snapshot`` without mutating it."""

        outcome = refill_symbolic_findings(
            receipts=receipts,
            ledger=self.ledger,
            binding=snapshot.binding,
            goals=snapshot.goals,
            tasks=snapshot.tasks,
            state=snapshot.state,
            policy=self.policy,
            now_epoch=now_epoch,
            dependencies=dependencies,
            healthy_exhaustion=healthy_exhaustion,
        )
        if outcome.evidence_methods != snapshot.evidence_methods:
            raise SymbolicFindingRefillError(
                "refill proposal is missing required objective evidence"
            )
        return outcome


# ---------------------------------------------------------------------------
# Objective evidence discovery + prove claims (VFS-G160 / VFS-G161)
# ---------------------------------------------------------------------------


def symbolic_refill_epoch_evidence() -> str:
    """Return the closed ``vfs/symbolic-refill-epoch@1`` evidence term."""

    return SYMBOLIC_REFILL_EPOCH_EVIDENCE


def refill_idempotency_evidence() -> str:
    """Return the closed ``vfs/refill-idempotency@1`` evidence term."""

    return REFILL_IDEMPOTENCY_EVIDENCE


def symbolic_refill_epoch_evidence_terms() -> tuple[str, ...]:
    """Return the VFS-G160 domain evidence surface for discovery scanners.

    Exact identity: ``vfs/symbolic-refill-epoch@1``.  Authored only by
    :class:`SymbolicRefillEpochEvidence` and :func:`prove_symbolic_refill_epoch`.
    """

    return (SYMBOLIC_REFILL_EPOCH_EVIDENCE,)


def refill_idempotency_evidence_terms() -> tuple[str, ...]:
    """Return the VFS-G161 domain evidence surface for discovery scanners.

    Exact identity: ``vfs/refill-idempotency@1``.  Authored only by
    :class:`RefillIdempotencyEvidence` and :func:`prove_refill_idempotency`.
    """

    return (REFILL_IDEMPOTENCY_EVIDENCE,)


def covered_evidence_terms() -> tuple[str, ...]:
    """Return domain objective evidence terms this refill surface proves.

    Covers ``vfs/symbolic-refill-epoch@1`` (VFS-G160) and
    ``vfs/refill-idempotency@1`` (VFS-G161) for the autonomous-refill
    goal packet.  Goal/task labels stay metadata and never enter epoch or
    task content identities.
    """

    return OBJECTIVE_DOMAIN_EVIDENCE_TERMS


def packet_evidence_terms() -> tuple[str, ...]:
    """Return the closed autonomous-refill packet evidence set."""

    return OBJECTIVE_PACKET_EVIDENCE_TERMS


def all_covered_evidence_terms() -> tuple[str, ...]:
    """Alias of :func:`covered_evidence_terms` for cross-module discovery."""

    return covered_evidence_terms()


def verify_symbolic_refill_epoch(outcome: RefillOutcome) -> bool:
    """Return True when ``outcome`` carries a well-formed epoch receipt."""

    if not isinstance(outcome, RefillOutcome):
        return False
    epoch = outcome.epoch_evidence
    if epoch is None:
        return False
    try:
        record = epoch.to_record()
    except (TypeError, ValueError, SymbolicFindingRefillError):
        return False
    return (
        record.get("schema") == SYMBOLIC_REFILL_EPOCH_SCHEMA
        and record.get("epoch_id") == epoch.epoch_id == outcome.refill_epoch_id
        and record.get("binding") == outcome.binding.to_record()
        and record.get("binding_id") == outcome.binding.binding_id
        and isinstance(record.get("prior_state_id"), str)
        and bool(record.get("prior_state_id"))
        and isinstance(record.get("result_state_id"), str)
        and bool(record.get("result_state_id"))
        and record.get("reason") == outcome.reason.value
        and record.get("open_work_before") == outcome.open_work_before
        and tuple(record.get("emitted_task_ids") or ())
        == tuple(task.task_id for task in outcome.new_tasks)
        and tuple(record.get("emitted_goal_ids") or ())
        == tuple(goal.goal_id for goal in outcome.new_goals)
        and bool(record.get("changed")) is bool(outcome.changed)
        and not REFILL_AUTHORIZES_EXECUTION
        and not REFILL_AUTHORIZES_COMPLETION
    )


def verify_refill_idempotency(outcome: RefillOutcome) -> bool:
    """Return True when ``outcome`` carries a well-formed idempotency receipt."""

    if not isinstance(outcome, RefillOutcome):
        return False
    evidence = outcome.idempotency_evidence
    if evidence is None:
        return False
    try:
        record = evidence.to_record()
    except (TypeError, ValueError, SymbolicFindingRefillError):
        return False
    return (
        record.get("schema") == REFILL_IDEMPOTENCY_SCHEMA
        and record.get("idempotency_id")
        == evidence.idempotency_id
        == outcome.idempotency_id
        and record.get("epoch_id") == outcome.refill_epoch_id
        and record.get("binding_id") == outcome.binding.binding_id
        and set(record.get("emitted_task_ids") or ()).issubset(
            set(record.get("resolved_task_ids") or ())
        )
        and set(record.get("replay_receipt_ids") or ()).issubset(
            set(record.get("operation_receipt_ids") or ())
        )
        and bool(record.get("replay_noop")) is bool(evidence.replay_noop)
    )


def symbolic_refill_epoch_acceptance_dimensions(
    outcome: RefillOutcome,
    *,
    policy: SymbolicFindingRefillPolicy | None = None,
) -> dict[str, bool]:
    """Map VFS-G160 / parent VFS-G120 acceptance onto one epoch receipt.

    Parent frozen-root acceptance for ``vfs/symbolic-refill-epoch@1``:

    * only fresh admitted findings produce work;
    * existing goal families are reused under exact binding;
    * new children are bounded by breadth/depth/open-work/cooldown;
    * each decision is content-addressed against prior and result state;
    * wall-clock observation cannot rewrite task identity;
    * proposals never authorize execution or completion.
    """

    policy = policy or SymbolicFindingRefillPolicy()
    epoch = outcome.epoch_evidence
    if epoch is None:
        return {
            "evidence_identity": False,
            "fresh_admitted_only": False,
            "goal_family_reuse_or_bounded_child": False,
            "breadth_depth_open_work_cooldown": False,
            "prior_and_result_state_tracked": False,
            "epoch_distinct_from_task_identity": False,
            "binding_identity": False,
            "non_authoritative": False,
            "outcome_verified": False,
        }
    bound_tasks = all(
        task.repository_id == outcome.binding.repository_id
        and task.tree_id == outcome.binding.tree_id
        and task.policy_id == outcome.binding.policy_id
        and task.policy_revision == outcome.binding.policy_revision
        and task.objective_forest_id == outcome.binding.objective_forest_id
        and task.objective_forest_revision
        == outcome.binding.objective_forest_revision
        and not task.write_authorized
        for task in outcome.new_tasks
    )
    bound_goals = all(
        goal.depth <= policy.max_goal_depth
        and (
            not goal.parent_goal_id
            or goal.parent_goal_id in goal.ancestor_goal_ids
        )
        for goal in outcome.new_goals
    )
    gate_only = outcome.reason in {
        RefillReason.THRESHOLD_SATISFIED,
        RefillReason.OPEN_WORK_CEILING,
        RefillReason.COOLDOWN,
    }
    materializes_from_fresh = bool(epoch.fresh_receipt_ids)
    materializes_bounded_review = any(
        diagnostic.disposition is FindingDisposition.REVIEW_MATERIALIZED
        for diagnostic in outcome.diagnostics
    )
    # Work requires either a fresh admitted receipt or a single bounded review
    # task whose provenance is an already-consumed exhausted repair receipt.
    no_work_without_fresh = (
        not outcome.new_tasks
        or materializes_from_fresh
        or materializes_bounded_review
    )
    breadth_ok = (
        len(outcome.new_tasks) <= policy.max_findings_per_pass
        and len(outcome.new_goals) <= policy.max_children
        and (
            outcome.open_work_before < policy.open_work_ceiling
            or gate_only
            or not outcome.changed
        )
    )
    return {
        "evidence_identity": (
            epoch.to_record().get("schema") == SYMBOLIC_REFILL_EPOCH_SCHEMA
            and epoch.epoch_id == outcome.refill_epoch_id
        ),
        "fresh_admitted_only": no_work_without_fresh,
        "goal_family_reuse_or_bounded_child": bound_goals
        and (
            not outcome.new_goals
            or all(
                goal.root_cause_family and goal.depth >= 1
                for goal in outcome.new_goals
            )
        ),
        "breadth_depth_open_work_cooldown": breadth_ok
        and policy.max_children <= 3
        and policy.max_goal_depth <= 4
        and policy.max_findings_per_pass <= 8
        and policy.max_surplus_per_goal <= 2,
        "prior_and_result_state_tracked": bool(
            epoch.prior_state_id and epoch.result_state_id
        ),
        "epoch_distinct_from_task_identity": (
            bool(outcome.refill_epoch_id)
            and (
                not outcome.idempotency_id
                or outcome.refill_epoch_id != outcome.idempotency_id
            )
        ),
        "binding_identity": bound_tasks
        and epoch.binding.binding_id == outcome.binding.binding_id,
        "non_authoritative": (
            not REFILL_AUTHORIZES_EXECUTION
            and not REFILL_AUTHORIZES_COMPLETION
            and all(not task.write_authorized for task in outcome.new_tasks)
        ),
        "outcome_verified": verify_symbolic_refill_epoch(outcome),
    }


def refill_idempotency_acceptance_dimensions(
    outcome: RefillOutcome,
) -> dict[str, bool]:
    """Map VFS-G161 acceptance criteria onto one idempotency receipt."""

    evidence = outcome.idempotency_evidence
    if evidence is None:
        return {
            "evidence_identity": False,
            "stable_operation_id": False,
            "replay_noop_when_replayed": False,
            "resolved_covers_emitted": False,
            "wall_clock_excluded": False,
            "non_authoritative": False,
            "outcome_verified": False,
        }
    record = evidence.to_record()
    return {
        "evidence_identity": record.get("schema") == REFILL_IDEMPOTENCY_SCHEMA
        and evidence.idempotency_id == outcome.idempotency_id,
        "stable_operation_id": bool(evidence.idempotency_id)
        and evidence.idempotency_id.startswith("refill-idempotency:"),
        "replay_noop_when_replayed": (
            not evidence.replay_receipt_ids or evidence.replay_noop
        ),
        "resolved_covers_emitted": set(evidence.emitted_task_ids).issubset(
            set(evidence.resolved_task_ids)
        ),
        "wall_clock_excluded": (
            # Identity payload deliberately omits observation time; emitted
            # task ids are recorded for diagnostics but do not feed the
            # stable operation id (see RefillIdempotencyEvidence.idempotency_id).
            "observed_at_epoch" not in record
            and evidence.idempotency_id
            == _stable_id(
                "refill-idempotency",
                {
                    "schema": REFILL_IDEMPOTENCY_SCHEMA,
                    "binding_id": outcome.binding.binding_id,
                    "operation_receipt_ids": evidence.operation_receipt_ids,
                },
            )
        ),
        "non_authoritative": (
            not REFILL_AUTHORIZES_EXECUTION
            and not REFILL_AUTHORIZES_COMPLETION
        ),
        "outcome_verified": verify_refill_idempotency(outcome),
    }


def prove_symbolic_refill_epoch(
    outcome: RefillOutcome,
    *,
    policy: SymbolicFindingRefillPolicy | None = None,
) -> dict[str, Any]:
    """Emit a portable ``vfs/symbolic-refill-epoch@1`` evidence claim (VFS-G160).

    Proves one content-addressed refill decision for goal_packet/autonomous_refill:

    * only fresh admitted findings produce work under the bound forest revision;
    * existing families are reused; new children stay within hard ceilings;
    * prior and result state ids record the exact planner transition;
    * epoch id is observation-sensitive while task identity remains stable;
    * proposals never authorize execution or completion.
    """

    if not isinstance(outcome, RefillOutcome):
        raise TypeError("outcome must be a RefillOutcome")
    if outcome.epoch_evidence is None:
        raise SymbolicFindingRefillError(
            "refill outcome is missing vfs/symbolic-refill-epoch@1 evidence"
        )
    dimensions = symbolic_refill_epoch_acceptance_dimensions(
        outcome, policy=policy
    )
    verified = verify_symbolic_refill_epoch(outcome)
    epoch = outcome.epoch_evidence
    satisfied = verified and all(dimensions.values())
    return {
        "schema": SYMBOLIC_REFILL_EPOCH_CLAIM_SCHEMA,
        "evidence": SYMBOLIC_REFILL_EPOCH_EVIDENCE,
        "evidence_terms": list(symbolic_refill_epoch_evidence_terms()),
        "requirement_id": SYMBOLIC_REFILL_EPOCH_EVIDENCE,
        "goal_id": OBJECTIVE_GOAL_G160_ID,
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "task_id": OBJECTIVE_TASK_G160_ID,
        "packet_task_id": OBJECTIVE_TASK_PACKET_ID,
        "packet_goal_ids": list(OBJECTIVE_PACKET_GOAL_IDS),
        "epoch_id": epoch.epoch_id,
        "binding_id": outcome.binding.binding_id,
        "binding": outcome.binding.to_record(),
        "reason": outcome.reason.value,
        "prior_state_id": epoch.prior_state_id,
        "result_state_id": epoch.result_state_id,
        "fresh_receipt_ids": list(epoch.fresh_receipt_ids),
        "processed_receipt_ids": list(epoch.processed_receipt_ids),
        "emitted_goal_ids": list(epoch.emitted_goal_ids),
        "emitted_task_ids": list(epoch.emitted_task_ids),
        "open_work_before": epoch.open_work_before,
        "changed": epoch.changed,
        "epoch_record": epoch.to_record(),
        "acceptance_dimensions": dimensions,
        "invariants": list(SYMBOLIC_REFILL_EPOCH_INVARIANTS),
        "verified": verified,
        "satisfied": satisfied,
        "write_authorized": False,
        "authorizes_execution": REFILL_AUTHORIZES_EXECUTION,
        "authorizes_completion": REFILL_AUTHORIZES_COMPLETION,
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


def prove_refill_idempotency(outcome: RefillOutcome) -> dict[str, Any]:
    """Emit a portable ``vfs/refill-idempotency@1`` evidence claim (VFS-G161).

    Proves replay-safe work identity for goal_packet/autonomous_refill:

    * the same binding and operation receipts yield one stable idempotency id;
    * replay of consumed receipts is a no-op with ``replay_noop``;
    * resolved task ids cover every emitted id;
    * observation time never feeds the operation identity.
    """

    if not isinstance(outcome, RefillOutcome):
        raise TypeError("outcome must be a RefillOutcome")
    if outcome.idempotency_evidence is None:
        raise SymbolicFindingRefillError(
            "refill outcome is missing vfs/refill-idempotency@1 evidence"
        )
    dimensions = refill_idempotency_acceptance_dimensions(outcome)
    verified = verify_refill_idempotency(outcome)
    evidence = outcome.idempotency_evidence
    satisfied = verified and all(dimensions.values())
    return {
        "schema": REFILL_IDEMPOTENCY_CLAIM_SCHEMA,
        "evidence": REFILL_IDEMPOTENCY_EVIDENCE,
        "evidence_terms": list(refill_idempotency_evidence_terms()),
        "requirement_id": REFILL_IDEMPOTENCY_EVIDENCE,
        "goal_id": OBJECTIVE_GOAL_G161_ID,
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "task_id": OBJECTIVE_TASK_G161_ID,
        "packet_task_id": OBJECTIVE_TASK_PACKET_ID,
        "packet_goal_ids": list(OBJECTIVE_PACKET_GOAL_IDS),
        "idempotency_id": evidence.idempotency_id,
        "epoch_id": evidence.epoch_id,
        "binding_id": outcome.binding.binding_id,
        "operation_receipt_ids": list(evidence.operation_receipt_ids),
        "emitted_task_ids": list(evidence.emitted_task_ids),
        "resolved_task_ids": list(evidence.resolved_task_ids),
        "replay_receipt_ids": list(evidence.replay_receipt_ids),
        "replay_noop": evidence.replay_noop,
        "idempotency_record": evidence.to_record(),
        "acceptance_dimensions": dimensions,
        "invariants": list(REFILL_IDEMPOTENCY_INVARIANTS),
        "verified": verified,
        "satisfied": satisfied,
        "authorizes_execution": REFILL_AUTHORIZES_EXECUTION,
        "authorizes_completion": REFILL_AUTHORIZES_COMPLETION,
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


def prove_autonomous_refill_packet(
    outcome: RefillOutcome,
    *,
    policy: SymbolicFindingRefillPolicy | None = None,
) -> dict[str, Any]:
    """Emit the full VFS-G160 + VFS-G161 evidence set for the refill packet.

    Covers both ``vfs/symbolic-refill-epoch@1`` and
    ``vfs/refill-idempotency@1`` in one cohesive claim for
    goal_packet/autonomous_refill.  Never grants execution or completion
    authority.
    """

    if not isinstance(outcome, RefillOutcome):
        raise TypeError("outcome must be a RefillOutcome")
    epoch_claim = prove_symbolic_refill_epoch(outcome, policy=policy)
    idempotency_claim = prove_refill_idempotency(outcome)
    satisfied = bool(epoch_claim.get("satisfied")) and bool(
        idempotency_claim.get("satisfied")
    )
    return {
        "schema": AUTONOMOUS_REFILL_PACKET_CLAIM_SCHEMA,
        "evidence_terms": list(packet_evidence_terms()),
        "all_evidence_terms": list(OBJECTIVE_DOMAIN_EVIDENCE_TERMS),
        "goal_ids": list(OBJECTIVE_PACKET_GOAL_IDS),
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "task_ids": [OBJECTIVE_TASK_G160_ID, OBJECTIVE_TASK_G161_ID],
        "packet_task_id": OBJECTIVE_TASK_PACKET_ID,
        "symbolic_refill_epoch": epoch_claim,
        "refill_idempotency": idempotency_claim,
        "satisfied": satisfied,
        "authorizes_execution": REFILL_AUTHORIZES_EXECUTION,
        "authorizes_completion": REFILL_AUTHORIZES_COMPLETION,
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


# Concise compatibility alias for callers that treat refill as a pure planner.
plan_symbolic_finding_refill = refill_symbolic_findings


__all__ = [
    "AUTONOMOUS_REFILL_PACKET_CLAIM_SCHEMA",
    "DEFAULT_COOLDOWN_SECONDS",
    "DEFAULT_MAX_CHILDREN",
    "DEFAULT_MAX_FINDINGS_PER_PASS",
    "DEFAULT_MAX_GOAL_DEPTH",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_SURPLUS_PER_GOAL",
    "DEFAULT_OPEN_WORK_CEILING",
    "DEFAULT_REFILL_THRESHOLD",
    "OBJECTIVE_DOMAIN_EVIDENCE_TERMS",
    "OBJECTIVE_GOAL_G160_ID",
    "OBJECTIVE_GOAL_G161_ID",
    "OBJECTIVE_PACKET_EVIDENCE_TERMS",
    "OBJECTIVE_PACKET_GOAL_IDS",
    "OBJECTIVE_PARENT_GOAL_ID",
    "OBJECTIVE_TASK_G160_ID",
    "OBJECTIVE_TASK_G161_ID",
    "OBJECTIVE_TASK_PACKET_ID",
    "REFILL_AUTHORIZES_COMPLETION",
    "REFILL_AUTHORIZES_EXECUTION",
    "REFILL_IDEMPOTENCY_CLAIM_SCHEMA",
    "REFILL_IDEMPOTENCY_EVIDENCE",
    "REFILL_IDEMPOTENCY_INVARIANTS",
    "REFILL_IDEMPOTENCY_SCHEMA",
    "SYMBOLIC_FINDING_REFILL_INTERFACE",
    "SYMBOLIC_FINDING_REFILL_VERSION",
    "SYMBOLIC_REFILL_EPOCH_CLAIM_SCHEMA",
    "SYMBOLIC_REFILL_EPOCH_EVIDENCE",
    "SYMBOLIC_REFILL_EPOCH_INVARIANTS",
    "SYMBOLIC_REFILL_EPOCH_SCHEMA",
    "SYMBOLIC_REFILL_EVIDENCE_SCHEMAS",
    "BacklogRefinery",
    "FindingDisposition",
    "HealthyExhaustionReceipt",
    "RefillAncestryError",
    "RefillBinding",
    "RefillBindingError",
    "RefillDiagnostic",
    "RefillGoal",
    "RefillIdempotencyEvidence",
    "RefillOutcome",
    "RefillReason",
    "RefillState",
    "RefillTask",
    "SupervisorBacklogSnapshot",
    "SymbolicFindingRefillError",
    "SymbolicFindingRefillPolicy",
    "SymbolicFindingRefiller",
    "SymbolicRefillEpochEvidence",
    "TaskKind",
    "all_covered_evidence_terms",
    "covered_evidence_terms",
    "packet_evidence_terms",
    "plan_symbolic_finding_refill",
    "prove_autonomous_refill_packet",
    "prove_refill_idempotency",
    "prove_symbolic_refill_epoch",
    "refill_idempotency_acceptance_dimensions",
    "refill_idempotency_evidence",
    "refill_idempotency_evidence_terms",
    "refill_symbolic_findings",
    "symbolic_refill_epoch_acceptance_dimensions",
    "symbolic_refill_epoch_evidence",
    "symbolic_refill_epoch_evidence_terms",
    "verify_refill_idempotency",
    "verify_symbolic_refill_epoch",
]
