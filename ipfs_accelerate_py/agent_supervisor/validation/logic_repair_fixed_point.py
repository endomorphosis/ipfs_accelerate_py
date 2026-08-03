"""Post-edit joint program+logic fixed-point validation (LPR-018).

After each provisional transaction commit the validator:

1. rebuilds repository/AST/vector/KG/call/dependency/schema/value graphs and
   tombstones;
2. recomputes delta/closure/consumer ledger;
3. regenerates corpus/goals/gaps/Tactician plan and Hammer/native-goal/
   countermodel receipts for changed or introduced clauses;
4. revalidates every original and newly resolved caller, second-order change,
   chosen value/behavior/placement, and policy tool.

Completion requires no unresolved mandatory consumer, open required frontier,
unplanned breaking delta, new required logic gap, stale prediction, or failed
validation.  Exact per-iteration logic evidence is attached to the existing
:class:`PropagationCompletionReceipt` via
:class:`LogicFixedPointEvidenceAttachment` (never a replacement receipt).

Finalize only after residual-free success.  Bound exhaustion, incompleteness,
or failure after a provisional commit triggers compensating rollback to the
checkpoint.  Partial SCC/packet completion can never close the task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ..analysis.change_propagation_contracts import (
    AtomicPropagationPlan,
    CompletionDisposition,
    PropagationAuthorityRoots,
    PropagationCompletionReceipt,
    PropagationTransaction,
    TransactionState,
)
from ..analysis.program_logic_prediction_contracts import (
    FixedPointAttachmentDisposition,
    LogicFixedPointEvidenceAttachment,
    ProgramLogicAuthorityRoots,
)
from ..planning.change_propagation_transaction import (
    ChangePropagationTransaction,
    PropagationCheckpoint,
    PropagationRollbackReceipt,
    TransactionExecutionReport,
)
from ..proof.change_propagation_edit_packet import ChangePropagationEditPacket
from ..proof.formal_verification_contracts import content_identity
from .change_propagation_validation import (
    DEFAULT_FIXED_POINT_BOUND,
    MAX_ITERATIONS,
    CandidatePropagationEvidence,
    ChangePropagationValidationError,
    ChangePropagationValidator,
    PropagationValidationOutcome,
    PropagationValidationReason,
)


LOGIC_REPAIR_FIXED_POINT_INTERFACE: Final[str] = "LogicRepairFixedPointValidator@1"
LOGIC_REPAIR_FIXED_POINT_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-fixed-point-report@1"
)
LOGIC_REPAIR_ITERATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-iteration-receipt@1"
)
PROPAGATION_FINALIZE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/propagation-finalize-receipt@1"
)
COMPENSATING_ROLLBACK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/compensating-rollback-receipt@1"
)
LOGIC_REBUILD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair/rebuild@1"
)
LOGIC_REPLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair/replan@1"
)
LOGIC_REPROVE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair/reprove@1"
)
LOGIC_CONSUMER_REVALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair/consumer-revalidation@1"
)

PRODUCER_ID: Final[str] = "logic-repair-fixed-point@1"
MAX_REASON_CODES: Final[int] = 64
MAX_IDS: Final[int] = 1_024
MAX_TEXT_BYTES: Final[int] = 4_096
DEFAULT_LOGIC_FIXED_POINT_BOUND: Final[int] = DEFAULT_FIXED_POINT_BOUND


class LogicRepairFixedPointError(ValueError):
    """Joint program+logic fixed-point validation failed closed."""


class LogicFixedPointReason(str, Enum):
    """Stable, machine-readable logic fixed-point failure codes."""

    MALFORMED_INPUT = "malformed_input"
    PROGRAM_FIXED_POINT_INCOMPLETE = "program_fixed_point_incomplete"
    TRANSACTION_NOT_PROVISIONAL = "transaction_not_provisional"
    ROOT_DRIFT = "root_drift"
    STALE_CANDIDATE_TREE = "stale_candidate_tree"
    REBUILD_INCOMPLETE = "rebuild_incomplete"
    TOMBSTONE_MISSING = "tombstone_missing"
    DELTA_RECOMPUTE_FAILED = "delta_recompute_failed"
    UNPLANNED_BREAKING_DELTA = "unplanned_breaking_delta"
    CLOSURE_RECOMPUTE_FAILED = "closure_recompute_failed"
    UNCOVERED_FRONTIER = "uncovered_frontier"
    UNRESOLVED_MANDATORY_CONSUMER = "unresolved_mandatory_consumer"
    NEW_RESOLVED_CONSUMER_OPEN = "new_resolved_consumer_open"
    CORPUS_REGENERATION_FAILED = "corpus_regeneration_failed"
    GOAL_REGENERATION_FAILED = "goal_regeneration_failed"
    TACTICIAN_PLAN_STALE = "tactician_plan_stale"
    HAMMER_RECEIPT_MISSING = "hammer_receipt_missing"
    NATIVE_GOAL_STALE = "native_goal_stale"
    COUNTERMODEL_STALE = "countermodel_stale"
    PREDICTION_STALE = "prediction_stale"
    NEW_REQUIRED_LOGIC_GAP = "new_required_logic_gap"
    VALUE_BEHAVIOR_FAILED = "value_behavior_failed"
    PLACEMENT_FAILED = "placement_failed"
    POLICY_TOOL_FAILED = "policy_tool_failed"
    BOUND_EXHAUSTED = "fixed_point_bound_exhausted"
    FIXED_POINT_NOT_REACHED = "fixed_point_not_reached"
    PARTIAL_SCC_FORBIDDEN = "partial_scc_completion_forbidden"
    PARTIAL_PACKET_FORBIDDEN = "partial_packet_completion_forbidden"
    FINALIZE_REQUIRED = "finalize_required"
    ROLLBACK_REQUIRED = "compensating_rollback_required"
    ROLLBACK_FAILED = "compensating_rollback_failed"
    MISSING_LOGIC_EVIDENCE = "missing_logic_evidence"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"


class LogicFixedPointStage(str, Enum):
    """Ordered stages of one joint program+logic fixed-point iteration."""

    PROGRAM_VALIDATION = "program_validation"
    REBUILD = "rebuild"
    DELTA_CLOSURE = "delta_closure"
    CONSUMER_LEDGER = "consumer_ledger"
    CORPUS_GOALS = "corpus_goals"
    TACTICIAN_PLAN = "tactician_plan"
    HAMMER_REPROVE = "hammer_reprove"
    PREDICTION_REVALIDATE = "prediction_revalidate"
    VALUE_BEHAVIOR_PLACEMENT = "value_behavior_placement"
    POLICY_TOOLS = "policy_tools"
    FIXED_POINT = "fixed_point"
    FINALIZE = "finalize"
    COMPENSATING_ROLLBACK = "compensating_rollback"


class LogicStageDisposition(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    CONTINUE = "continue"
    ROLLED_BACK = "rolled_back"
    FINALIZED = "finalized"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise LogicRepairFixedPointError(f"{name} must be a compact identifier")
    text = value.strip()
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise LogicRepairFixedPointError(f"{name} exceeds text bound")
    return text


def _ids(
    values: Sequence[str],
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_IDS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise LogicRepairFixedPointError(f"{name} must be an identifier sequence")
    result = tuple(
        sorted(
            {
                value.strip()
                for value in values
                if isinstance(value, str) and value.strip()
            }
        )
    )
    for item in result:
        if any(char.isspace() for char in item):
            raise LogicRepairFixedPointError(f"{name} must contain compact identifiers")
    if required and not result:
        raise LogicRepairFixedPointError(f"{name} must not be empty")
    if len(result) > maximum:
        raise LogicRepairFixedPointError(f"{name} exceeds item bound")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise LogicRepairFixedPointError(f"{name} must be a boolean")
    return value


def _bounded_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_ITERATIONS,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LogicRepairFixedPointError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise LogicRepairFixedPointError(f"{name} out of bounds [{minimum}, {maximum}]")
    return value


# ---------------------------------------------------------------------------
# Stage evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicRebuildEvidence:
    """Proof that program graphs/indexes/tombstones were rebuilt post-edit."""

    candidate_tree_id: str
    repository_index_id: str
    ast_index_id: str
    vector_row_ids: tuple[str, ...]
    kg_node_ids: tuple[str, ...]
    call_graph_id: str
    dependency_graph_id: str
    schema_graph_id: str
    value_graph_id: str
    tombstone_ids: tuple[str, ...]
    clean_rebuild_equivalent: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        for name in (
            "repository_index_id",
            "ast_index_id",
            "call_graph_id",
            "dependency_graph_id",
            "schema_graph_id",
            "value_graph_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "vector_row_ids", _ids(self.vector_row_ids, "vector_row_ids")
        )
        object.__setattr__(self, "kg_node_ids", _ids(self.kg_node_ids, "kg_node_ids"))
        object.__setattr__(
            self, "tombstone_ids", _ids(self.tombstone_ids, "tombstone_ids")
        )
        object.__setattr__(
            self,
            "clean_rebuild_equivalent",
            _bool(self.clean_rebuild_equivalent, "clean_rebuild_equivalent"),
        )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": LOGIC_REBUILD_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "repository_index_id": self.repository_index_id,
            "ast_index_id": self.ast_index_id,
            "vector_row_ids": list(self.vector_row_ids),
            "kg_node_ids": list(self.kg_node_ids),
            "call_graph_id": self.call_graph_id,
            "dependency_graph_id": self.dependency_graph_id,
            "schema_graph_id": self.schema_graph_id,
            "value_graph_id": self.value_graph_id,
            "tombstone_ids": list(self.tombstone_ids),
            "clean_rebuild_equivalent": self.clean_rebuild_equivalent,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class LogicReplanEvidence:
    """Regenerated corpus, goals, gaps, and Tactician plan for changed clauses."""

    candidate_tree_id: str
    corpus_root_id: str
    goal_root_ids: tuple[str, ...]
    gap_ids: tuple[str, ...]
    required_gap_ids: tuple[str, ...]
    new_required_gap_ids: tuple[str, ...]
    tactician_plan_id: str
    plan_current: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        object.__setattr__(
            self, "corpus_root_id", _identifier(self.corpus_root_id, "corpus_root_id")
        )
        object.__setattr__(
            self, "goal_root_ids", _ids(self.goal_root_ids, "goal_root_ids")
        )
        object.__setattr__(self, "gap_ids", _ids(self.gap_ids, "gap_ids"))
        object.__setattr__(
            self, "required_gap_ids", _ids(self.required_gap_ids, "required_gap_ids")
        )
        object.__setattr__(
            self,
            "new_required_gap_ids",
            _ids(self.new_required_gap_ids, "new_required_gap_ids"),
        )
        object.__setattr__(
            self,
            "tactician_plan_id",
            _identifier(self.tactician_plan_id, "tactician_plan_id"),
        )
        object.__setattr__(self, "plan_current", _bool(self.plan_current, "plan_current"))
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )
        if self.plan_current and self.new_required_gap_ids:
            raise LogicRepairFixedPointError(
                "current tactician plan forbids new required logic gaps"
            )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": LOGIC_REPLAN_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "corpus_root_id": self.corpus_root_id,
            "goal_root_ids": list(self.goal_root_ids),
            "gap_ids": list(self.gap_ids),
            "required_gap_ids": list(self.required_gap_ids),
            "new_required_gap_ids": list(self.new_required_gap_ids),
            "tactician_plan_id": self.tactician_plan_id,
            "plan_current": self.plan_current,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class LogicReproveEvidence:
    """Hammer, native-goal, countermodel, and prediction receipts for clauses."""

    candidate_tree_id: str
    hammer_receipt_ids: tuple[str, ...]
    native_goal_binding_ids: tuple[str, ...]
    countermodel_receipt_ids: tuple[str, ...]
    prediction_receipt_ids: tuple[str, ...]
    stale_prediction_ids: tuple[str, ...]
    failed_reconstruction_ids: tuple[str, ...]
    all_promoted_clauses_current: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        for name in (
            "hammer_receipt_ids",
            "native_goal_binding_ids",
            "countermodel_receipt_ids",
            "prediction_receipt_ids",
            "stale_prediction_ids",
            "failed_reconstruction_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self,
            "all_promoted_clauses_current",
            _bool(
                self.all_promoted_clauses_current, "all_promoted_clauses_current"
            ),
        )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )
        if self.all_promoted_clauses_current and (
            self.stale_prediction_ids or self.failed_reconstruction_ids
        ):
            raise LogicRepairFixedPointError(
                "current promoted clauses forbid stale predictions or failed reconstructions"
            )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": LOGIC_REPROVE_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "hammer_receipt_ids": list(self.hammer_receipt_ids),
            "native_goal_binding_ids": list(self.native_goal_binding_ids),
            "countermodel_receipt_ids": list(self.countermodel_receipt_ids),
            "prediction_receipt_ids": list(self.prediction_receipt_ids),
            "stale_prediction_ids": list(self.stale_prediction_ids),
            "failed_reconstruction_ids": list(self.failed_reconstruction_ids),
            "all_promoted_clauses_current": self.all_promoted_clauses_current,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class LogicConsumerRevalidationEvidence:
    """Original and second-order consumer coverage after re-resolution."""

    candidate_tree_id: str
    original_consumer_ids: tuple[str, ...]
    discharged_original_ids: tuple[str, ...]
    newly_resolved_consumer_ids: tuple[str, ...]
    discharged_new_consumer_ids: tuple[str, ...]
    unresolved_mandatory_ids: tuple[str, ...]
    open_required_frontier_ids: tuple[str, ...]
    second_order_consumer_ids: tuple[str, ...]
    value_choice_ids: tuple[str, ...]
    behavior_choice_ids: tuple[str, ...]
    placement_choice_ids: tuple[str, ...]
    failed_value_behavior_placement_ids: tuple[str, ...]
    policy_tool_receipt_ids: tuple[str, ...]
    failed_policy_tool_ids: tuple[str, ...]
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        for name in (
            "original_consumer_ids",
            "discharged_original_ids",
            "newly_resolved_consumer_ids",
            "discharged_new_consumer_ids",
            "unresolved_mandatory_ids",
            "open_required_frontier_ids",
            "second_order_consumer_ids",
            "value_choice_ids",
            "behavior_choice_ids",
            "placement_choice_ids",
            "failed_value_behavior_placement_ids",
            "policy_tool_receipt_ids",
            "failed_policy_tool_ids",
        ):
            required = name == "original_consumer_ids"
            object.__setattr__(
                self, name, _ids(getattr(self, name), name, required=required)
            )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    @property
    def complete(self) -> bool:
        discharged = set(self.discharged_new_consumer_ids) | set(
            self.discharged_original_ids
        )
        return (
            not self.unresolved_mandatory_ids
            and not self.open_required_frontier_ids
            and not self.failed_value_behavior_placement_ids
            and not self.failed_policy_tool_ids
            and set(self.original_consumer_ids).issubset(self.discharged_original_ids)
            and set(self.newly_resolved_consumer_ids).issubset(
                self.discharged_new_consumer_ids
            )
            and set(self.second_order_consumer_ids).issubset(discharged)
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": LOGIC_CONSUMER_REVALIDATION_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "original_consumer_ids": list(self.original_consumer_ids),
            "discharged_original_ids": list(self.discharged_original_ids),
            "newly_resolved_consumer_ids": list(self.newly_resolved_consumer_ids),
            "discharged_new_consumer_ids": list(self.discharged_new_consumer_ids),
            "unresolved_mandatory_ids": list(self.unresolved_mandatory_ids),
            "open_required_frontier_ids": list(self.open_required_frontier_ids),
            "second_order_consumer_ids": list(self.second_order_consumer_ids),
            "value_choice_ids": list(self.value_choice_ids),
            "behavior_choice_ids": list(self.behavior_choice_ids),
            "placement_choice_ids": list(self.placement_choice_ids),
            "failed_value_behavior_placement_ids": list(
                self.failed_value_behavior_placement_ids
            ),
            "policy_tool_receipt_ids": list(self.policy_tool_receipt_ids),
            "failed_policy_tool_ids": list(self.failed_policy_tool_ids),
            "complete": self.complete,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class LogicRepairIterationReceipt:
    """Exact per-iteration logic evidence for one fixed-point round."""

    iteration: int
    rebuild: LogicRebuildEvidence
    replan: LogicReplanEvidence
    reprove: LogicReproveEvidence
    consumer_revalidation: LogicConsumerRevalidationEvidence
    unplanned_breaking_delta_ids: tuple[str, ...] = ()
    residual_logic_gap_ids: tuple[str, ...] = ()
    unsupported_logic_gap_ids: tuple[str, ...] = ()
    requires_another_iteration: bool = False
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "iteration", _bounded_int(self.iteration, "iteration", minimum=1)
        )
        for name, expected in (
            ("rebuild", LogicRebuildEvidence),
            ("replan", LogicReplanEvidence),
            ("reprove", LogicReproveEvidence),
            ("consumer_revalidation", LogicConsumerRevalidationEvidence),
        ):
            value = getattr(self, name)
            if not isinstance(value, expected):
                raise LogicRepairFixedPointError(f"{name} must be {expected.__name__}")
        object.__setattr__(
            self,
            "unplanned_breaking_delta_ids",
            _ids(self.unplanned_breaking_delta_ids, "unplanned_breaking_delta_ids"),
        )
        object.__setattr__(
            self,
            "residual_logic_gap_ids",
            _ids(self.residual_logic_gap_ids, "residual_logic_gap_ids"),
        )
        object.__setattr__(
            self,
            "unsupported_logic_gap_ids",
            _ids(self.unsupported_logic_gap_ids, "unsupported_logic_gap_ids"),
        )
        object.__setattr__(
            self,
            "requires_another_iteration",
            _bool(self.requires_another_iteration, "requires_another_iteration"),
        )
        residual = bool(
            self.unplanned_breaking_delta_ids
            or self.residual_logic_gap_ids
            or self.requires_another_iteration
            or not self.consumer_revalidation.complete
            or self.replan.new_required_gap_ids
            or not self.reprove.all_promoted_clauses_current
        )
        if self.requires_another_iteration and not residual:
            # Requires flag without residual is allowed only when explicitly
            # requesting another round due to second-order discovery already
            # folded into residual fields; otherwise fail closed.
            if not (
                self.consumer_revalidation.newly_resolved_consumer_ids
                or self.consumer_revalidation.second_order_consumer_ids
            ):
                raise LogicRepairFixedPointError(
                    "requires_another_iteration needs residual logic impacts"
                )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": LOGIC_REPAIR_ITERATION_RECEIPT_SCHEMA,
            "iteration": self.iteration,
            "rebuild": self.rebuild.to_dict(),
            "replan": self.replan.to_dict(),
            "reprove": self.reprove.to_dict(),
            "consumer_revalidation": self.consumer_revalidation.to_dict(),
            "unplanned_breaking_delta_ids": list(self.unplanned_breaking_delta_ids),
            "residual_logic_gap_ids": list(self.residual_logic_gap_ids),
            "unsupported_logic_gap_ids": list(self.unsupported_logic_gap_ids),
            "requires_another_iteration": self.requires_another_iteration,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class CandidateLogicRepairEvidence:
    """Multi-iteration logic evidence bundle for joint fixed-point validation."""

    candidate_tree_id: str
    logic_roots: ProgramLogicAuthorityRoots
    iterations: tuple[LogicRepairIterationReceipt, ...]
    expected_tombstone_ids: tuple[str, ...] = ()
    program_evidence: CandidatePropagationEvidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        if not isinstance(self.logic_roots, ProgramLogicAuthorityRoots):
            raise LogicRepairFixedPointError(
                "logic_roots must be ProgramLogicAuthorityRoots"
            )
        if (
            not isinstance(self.iterations, Sequence)
            or not self.iterations
            or not all(
                isinstance(item, LogicRepairIterationReceipt) for item in self.iterations
            )
        ):
            raise LogicRepairFixedPointError(
                "iterations must be a non-empty LogicRepairIterationReceipt sequence"
            )
        if len(self.iterations) > MAX_ITERATIONS:
            raise LogicRepairFixedPointError("iterations exceed policy bound")
        object.__setattr__(self, "iterations", tuple(self.iterations))
        object.__setattr__(
            self,
            "expected_tombstone_ids",
            _ids(self.expected_tombstone_ids, "expected_tombstone_ids"),
        )
        if self.program_evidence is not None and not isinstance(
            self.program_evidence, CandidatePropagationEvidence
        ):
            raise LogicRepairFixedPointError(
                "program_evidence must be CandidatePropagationEvidence when supplied"
            )
        if self.logic_roots.tree_id != self.candidate_tree_id:
            raise LogicRepairFixedPointError(
                "logic authority tree_id must match candidate_tree_id"
            )


@dataclass(frozen=True)
class PropagationFinalizeReceipt:
    """Finalize a provisional commit only after residual-free fixed point."""

    roots: PropagationAuthorityRoots
    finalize_id: str
    transaction_id: str
    plan_id: str
    completion_id: str
    checkpoint_id: str
    iteration_count: int
    fixed_point_receipt_id: str
    diagnostic_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise LogicRepairFixedPointError(
                "finalize roots must be PropagationAuthorityRoots"
            )
        for name in (
            "finalize_id",
            "transaction_id",
            "plan_id",
            "completion_id",
            "checkpoint_id",
            "fixed_point_receipt_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "iteration_count",
            _bounded_int(self.iteration_count, "iteration_count", minimum=1),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_FINALIZE_RECEIPT_SCHEMA,
            "interface": LOGIC_REPAIR_FIXED_POINT_INTERFACE,
            "roots": self.roots.to_dict(),
            "finalize_id": self.finalize_id,
            "transaction_id": self.transaction_id,
            "plan_id": self.plan_id,
            "completion_id": self.completion_id,
            "checkpoint_id": self.checkpoint_id,
            "iteration_count": self.iteration_count,
            "fixed_point_receipt_id": self.fixed_point_receipt_id,
            "diagnostic_refs": list(self.diagnostic_refs),
            "partial_merge_allowed": False,
            "provider_success_is_not_completion": True,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class CompensatingRollbackReceipt:
    """Compensating rollback after provisional-commit fixed-point failure."""

    roots: PropagationAuthorityRoots
    rollback_id: str
    transaction_id: str
    plan_id: str
    checkpoint_id: str
    restored: bool
    reason_codes: tuple[str, ...]
    iteration_count: int = 0
    diagnostic_refs: tuple[str, ...] = ()
    underlying_rollback_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise LogicRepairFixedPointError(
                "compensating rollback roots must be PropagationAuthorityRoots"
            )
        for name in ("rollback_id", "transaction_id", "plan_id", "checkpoint_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "restored", _bool(self.restored, "restored"))
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", required=True, maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "iteration_count",
            _bounded_int(self.iteration_count, "iteration_count", minimum=0),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs"),
        )
        if self.underlying_rollback_id:
            object.__setattr__(
                self,
                "underlying_rollback_id",
                _identifier(self.underlying_rollback_id, "underlying_rollback_id"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPENSATING_ROLLBACK_RECEIPT_SCHEMA,
            "interface": LOGIC_REPAIR_FIXED_POINT_INTERFACE,
            "roots": self.roots.to_dict(),
            "rollback_id": self.rollback_id,
            "transaction_id": self.transaction_id,
            "plan_id": self.plan_id,
            "checkpoint_id": self.checkpoint_id,
            "restored": self.restored,
            "reason_codes": list(self.reason_codes),
            "iteration_count": self.iteration_count,
            "diagnostic_refs": list(self.diagnostic_refs),
            "underlying_rollback_id": self.underlying_rollback_id,
            "partial_merge_allowed": False,
            "task_closed": False,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class LogicStageResult:
    stage: LogicFixedPointStage
    disposition: LogicStageDisposition
    reason_codes: tuple[str, ...] = ()
    receipt_id: str = ""
    iteration: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage",
            LogicFixedPointStage(self.stage)
            if not isinstance(self.stage, LogicFixedPointStage)
            else self.stage,
        )
        object.__setattr__(
            self,
            "disposition",
            LogicStageDisposition(self.disposition)
            if not isinstance(self.disposition, LogicStageDisposition)
            else self.disposition,
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        if self.receipt_id:
            object.__setattr__(
                self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
            )
        object.__setattr__(
            self,
            "iteration",
            _bounded_int(self.iteration, "iteration", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "receipt_id": self.receipt_id,
            "iteration": self.iteration,
        }


@dataclass(frozen=True)
class LogicRepairFixedPointReport:
    """Ordered joint fixed-point report; success is not merge authority alone."""

    plan_id: str
    transaction_id: str
    candidate_tree_id: str
    roots: PropagationAuthorityRoots
    stages: tuple[LogicStageResult, ...]
    reason_codes: tuple[str, ...]
    iteration_count: int
    complete: bool
    iteration_receipts: tuple[LogicRepairIterationReceipt, ...] = ()
    program_complete: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "transaction_id", _identifier(self.transaction_id, "transaction_id")
        )
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise LogicRepairFixedPointError(
                "report roots must be PropagationAuthorityRoots"
            )
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "iteration_count",
            _bounded_int(self.iteration_count, "iteration_count", minimum=0),
        )
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        object.__setattr__(
            self, "program_complete", _bool(self.program_complete, "program_complete")
        )
        object.__setattr__(self, "iteration_receipts", tuple(self.iteration_receipts))
        if self.complete and self.reason_codes:
            raise LogicRepairFixedPointError(
                "a complete report cannot carry failure reason codes"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LOGIC_REPAIR_FIXED_POINT_REPORT_SCHEMA,
            "interface": LOGIC_REPAIR_FIXED_POINT_INTERFACE,
            "plan_id": self.plan_id,
            "transaction_id": self.transaction_id,
            "candidate_tree_id": self.candidate_tree_id,
            "roots": self.roots.to_dict(),
            "stages": [item.to_dict() for item in self.stages],
            "reason_codes": list(self.reason_codes),
            "iteration_count": self.iteration_count,
            "complete": self.complete,
            "program_complete": self.program_complete,
            "iteration_receipt_ids": [item.receipt_id for item in self.iteration_receipts],
            "partial_merge_allowed": False,
            "provider_success_is_not_completion": True,
        }

    @property
    def report_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class LogicRepairFixedPointOutcome:
    """Joint fixed-point outcome: completion + logic attachment or rollback."""

    report: LogicRepairFixedPointReport
    completion: PropagationCompletionReceipt | None = None
    logic_attachment: LogicFixedPointEvidenceAttachment | None = None
    finalize: PropagationFinalizeReceipt | None = None
    compensating_rollback: CompensatingRollbackReceipt | None = None
    program_outcome: PropagationValidationOutcome | None = None
    rolled_back: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.report, LogicRepairFixedPointReport):
            raise LogicRepairFixedPointError(
                "outcome requires a LogicRepairFixedPointReport"
            )
        if self.completion is not None and not isinstance(
            self.completion, PropagationCompletionReceipt
        ):
            raise LogicRepairFixedPointError(
                "completion must be PropagationCompletionReceipt@1"
            )
        if self.logic_attachment is not None and not isinstance(
            self.logic_attachment, LogicFixedPointEvidenceAttachment
        ):
            raise LogicRepairFixedPointError(
                "logic_attachment must be LogicFixedPointEvidenceAttachment@1"
            )
        if self.finalize is not None and not isinstance(
            self.finalize, PropagationFinalizeReceipt
        ):
            raise LogicRepairFixedPointError(
                "finalize must be PropagationFinalizeReceipt"
            )
        if self.compensating_rollback is not None and not isinstance(
            self.compensating_rollback, CompensatingRollbackReceipt
        ):
            raise LogicRepairFixedPointError(
                "compensating_rollback must be CompensatingRollbackReceipt"
            )
        object.__setattr__(self, "rolled_back", _bool(self.rolled_back, "rolled_back"))
        if self.report.complete:
            if self.completion is None or self.completion.disposition is not (
                CompletionDisposition.COMPLETE
            ):
                raise LogicRepairFixedPointError(
                    "complete report requires COMPLETE PropagationCompletionReceipt"
                )
            if self.logic_attachment is None:
                raise LogicRepairFixedPointError(
                    "complete report requires LogicFixedPointEvidenceAttachment"
                )
            if self.finalize is None:
                raise LogicRepairFixedPointError(
                    "complete report requires PropagationFinalizeReceipt"
                )
            if self.compensating_rollback is not None or self.rolled_back:
                raise LogicRepairFixedPointError(
                    "complete report cannot carry compensating rollback"
                )
        if self.rolled_back and self.compensating_rollback is None:
            raise LogicRepairFixedPointError(
                "rolled_back outcome requires CompensatingRollbackReceipt"
            )
        if self.finalize is not None and self.compensating_rollback is not None:
            raise LogicRepairFixedPointError(
                "finalize and compensating rollback are mutually exclusive"
            )

    @property
    def complete(self) -> bool:
        return (
            self.report.complete
            and self.completion is not None
            and self.logic_attachment is not None
            and self.finalize is not None
            and not self.rolled_back
        )

    def require_complete(self) -> PropagationCompletionReceipt:
        if not self.complete or self.completion is None:
            reasons = ", ".join(self.report.reason_codes) or "incomplete"
            raise LogicRepairFixedPointError(
                "logic repair fixed-point validation rejected: " + reasons
            )
        return self.completion

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "rolled_back": self.rolled_back,
            "report": self.report.to_dict(),
            "completion_id": self.completion.completion_id if self.completion else "",
            "logic_attachment_id": (
                self.logic_attachment.attachment_id if self.logic_attachment else ""
            ),
            "finalize_id": self.finalize.finalize_id if self.finalize else "",
            "compensating_rollback_id": (
                self.compensating_rollback.rollback_id
                if self.compensating_rollback
                else ""
            ),
            "partial_merge_allowed": False,
            "provider_success_is_not_completion": True,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@dataclass
class LogicRepairFixedPointValidator:
    """Orchestrate joint program+logic fixed-point after provisional commit.

    Always invokes :class:`ChangePropagationValidator` for the program plane.
    When :class:`CandidateLogicRepairEvidence` is supplied, validates rebuild /
    replan / reprove / consumer revalidation per iteration, attaches
    :class:`LogicFixedPointEvidenceAttachment` to the existing completion, and
    emits finalize or compensating-rollback receipts.  Partial SCC/packet
    completion never yields COMPLETE.
    """

    INTERFACE: Final[str] = LOGIC_REPAIR_FIXED_POINT_INTERFACE

    program_validator: ChangePropagationValidator = field(
        default_factory=ChangePropagationValidator
    )
    fixed_point_bound: int = DEFAULT_LOGIC_FIXED_POINT_BOUND
    require_logic_evidence: bool = False

    def __post_init__(self) -> None:
        self.fixed_point_bound = _bounded_int(
            self.fixed_point_bound,
            "fixed_point_bound",
            minimum=1,
            maximum=MAX_ITERATIONS,
        )
        if not isinstance(self.require_logic_evidence, bool):
            raise LogicRepairFixedPointError("require_logic_evidence must be a boolean")
        if not isinstance(self.program_validator, ChangePropagationValidator):
            raise LogicRepairFixedPointError(
                "program_validator must be ChangePropagationValidator"
            )

    def validate(
        self,
        plan: AtomicPropagationPlan,
        transaction: PropagationTransaction,
        *,
        program_evidence: CandidatePropagationEvidence,
        logic_evidence: CandidateLogicRepairEvidence | None = None,
        packet: ChangePropagationEditPacket | None = None,
        execution_report: TransactionExecutionReport | None = None,
        fixed_point_bound: int | None = None,
        restore_adapter: Any | None = None,
        checkpoint: PropagationCheckpoint | None = None,
        contract_repair_completion_id: str = "",
    ) -> LogicRepairFixedPointOutcome:
        """Run program fixed-point then optional logic fixed-point (fail-closed)."""

        stages: list[LogicStageResult] = []
        reasons: set[str] = set()

        typed = (
            isinstance(plan, AtomicPropagationPlan)
            and isinstance(transaction, PropagationTransaction)
            and isinstance(program_evidence, CandidatePropagationEvidence)
        )
        if not typed:
            return self._malformed_outcome()

        bound = self.fixed_point_bound
        if fixed_point_bound is not None:
            bound = _bounded_int(
                fixed_point_bound, "fixed_point_bound", minimum=1, maximum=MAX_ITERATIONS
            )

        # Provisional commit: transaction must be COMMITTED (candidate tree
        # mutated) before fixed-point validation may finalize or roll back.
        if transaction.state is not TransactionState.COMMITTED:
            reasons.add(LogicFixedPointReason.TRANSACTION_NOT_PROVISIONAL.value)
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.PROGRAM_VALIDATION,
                    LogicStageDisposition.FAILED,
                    tuple(sorted(reasons)),
                )
            )
            return self._incomplete(
                plan,
                transaction,
                stages,
                reasons,
                iteration_count=0,
                program_complete=False,
            )

        # Partial packet/SCC completion cannot close the task.
        if execution_report is not None:
            if not isinstance(execution_report, TransactionExecutionReport):
                reasons.add(LogicFixedPointReason.MALFORMED_INPUT.value)
            else:
                if execution_report.partial_merge_allowed:
                    reasons.add(LogicFixedPointReason.PARTIAL_SCC_FORBIDDEN.value)
                if not execution_report.committed:
                    reasons.add(LogicFixedPointReason.TRANSACTION_NOT_PROVISIONAL.value)
                if set(execution_report.transaction.completed_step_ids) != {
                    step.step_id for step in plan.steps
                }:
                    reasons.add(LogicFixedPointReason.PARTIAL_PACKET_FORBIDDEN.value)

        if packet is not None and not isinstance(packet, ChangePropagationEditPacket):
            reasons.add(LogicFixedPointReason.MALFORMED_INPUT.value)

        if reasons:
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.PROGRAM_VALIDATION,
                    LogicStageDisposition.FAILED,
                    tuple(sorted(reasons)),
                )
            )
            return self._rollback_or_incomplete(
                plan,
                transaction,
                stages,
                reasons,
                iteration_count=0,
                program_complete=False,
                restore_adapter=restore_adapter,
                checkpoint=checkpoint or (
                    execution_report.checkpoint if execution_report else None
                ),
                execution_report=execution_report,
            )

        # --- Program plane (existing RPR fixed-point; never weakened) ---
        program_outcome = self.program_validator.validate(
            plan,
            transaction,
            evidence=program_evidence,
            packet=packet,
            execution_report=execution_report,
            fixed_point_bound=bound,
        )
        program_complete = bool(program_outcome.complete)
        stages.append(
            LogicStageResult(
                LogicFixedPointStage.PROGRAM_VALIDATION,
                LogicStageDisposition.PASSED
                if program_complete
                else LogicStageDisposition.FAILED,
                ()
                if program_complete
                else tuple(program_outcome.report.reason_codes)
                or (LogicFixedPointReason.PROGRAM_FIXED_POINT_INCOMPLETE.value,),
                (
                    program_outcome.completion.completion_id
                    if program_outcome.completion is not None
                    else ""
                ),
            )
        )
        if not program_complete:
            reasons.add(LogicFixedPointReason.PROGRAM_FIXED_POINT_INCOMPLETE.value)
            reasons.update(program_outcome.report.reason_codes)
            # Map common program residuals into logic reason vocabulary.
            for code in program_outcome.report.reason_codes:
                if code == PropagationValidationReason.BOUND_EXHAUSTED.value:
                    reasons.add(LogicFixedPointReason.BOUND_EXHAUSTED.value)
                elif code == PropagationValidationReason.UNCOVERED_FRONTIER.value:
                    reasons.add(LogicFixedPointReason.UNCOVERED_FRONTIER.value)
                elif code == PropagationValidationReason.UNRESOLVED_MANDATORY.value:
                    reasons.add(LogicFixedPointReason.UNRESOLVED_MANDATORY_CONSUMER.value)
                elif code == PropagationValidationReason.UNPLANNED_BREAKING_DELTA.value:
                    reasons.add(LogicFixedPointReason.UNPLANNED_BREAKING_DELTA.value)
            return self._rollback_or_incomplete(
                plan,
                transaction,
                stages,
                reasons,
                iteration_count=program_outcome.report.iteration_count,
                program_complete=False,
                program_outcome=program_outcome,
                completion=program_outcome.completion,
                restore_adapter=restore_adapter,
                checkpoint=checkpoint or (
                    execution_report.checkpoint if execution_report else None
                ),
                execution_report=execution_report,
            )

        assert program_outcome.completion is not None

        # Program-only path: logic evidence optional unless required.
        if logic_evidence is None:
            if self.require_logic_evidence:
                reasons.add(LogicFixedPointReason.MISSING_LOGIC_EVIDENCE.value)
                return self._rollback_or_incomplete(
                    plan,
                    transaction,
                    stages,
                    reasons,
                    iteration_count=program_outcome.report.iteration_count,
                    program_complete=True,
                    program_outcome=program_outcome,
                    completion=program_outcome.completion,
                    restore_adapter=restore_adapter,
                    checkpoint=checkpoint or (
                        execution_report.checkpoint if execution_report else None
                    ),
                    execution_report=execution_report,
                )
            # Backward-compatible: program fixed-point alone is not a *logic*
            # attachment, but pipeline may still treat program completion as
            # authoritative when logic plane is disabled.  Emit a non-attached
            # incomplete-for-logic report with program completion available.
            report = LogicRepairFixedPointReport(
                plan_id=plan.plan_id,
                transaction_id=transaction.transaction_id,
                candidate_tree_id=plan.roots.candidate_tree_id,
                roots=plan.roots,
                stages=tuple(stages),
                reason_codes=(),
                iteration_count=program_outcome.report.iteration_count,
                complete=False,
                program_complete=True,
            )
            return LogicRepairFixedPointOutcome(
                report=report,
                completion=program_outcome.completion,
                logic_attachment=None,
                finalize=None,
                compensating_rollback=None,
                program_outcome=program_outcome,
                rolled_back=False,
            )

        if not isinstance(logic_evidence, CandidateLogicRepairEvidence):
            reasons.add(LogicFixedPointReason.MALFORMED_INPUT.value)
            return self._rollback_or_incomplete(
                plan,
                transaction,
                stages,
                reasons,
                iteration_count=program_outcome.report.iteration_count,
                program_complete=True,
                program_outcome=program_outcome,
                completion=program_outcome.completion,
                restore_adapter=restore_adapter,
                checkpoint=checkpoint or (
                    execution_report.checkpoint if execution_report else None
                ),
                execution_report=execution_report,
            )

        if logic_evidence.candidate_tree_id != plan.roots.candidate_tree_id:
            reasons.add(LogicFixedPointReason.STALE_CANDIDATE_TREE.value)

        # --- Logic plane iterations ---
        reached = False
        iteration_count = 0
        accepted_receipts: list[LogicRepairIterationReceipt] = []
        residual_gaps: list[str] = []
        unsupported_gaps: list[str] = []
        last_goal_roots: tuple[str, ...] = ()
        last_corpus: tuple[str, ...] = ()
        last_tactician: tuple[str, ...] = ()
        last_hammer: tuple[str, ...] = ()
        last_predictions: tuple[str, ...] = ()
        last_original_coverage: tuple[str, ...] = ()
        last_second_order_coverage: tuple[str, ...] = ()

        for iter_receipt in logic_evidence.iterations:
            iteration_count = iter_receipt.iteration
            if iteration_count > bound:
                reasons.add(LogicFixedPointReason.BOUND_EXHAUSTED.value)
                break

            iter_reasons: list[str] = []

            # Rebuild
            rebuild = iter_receipt.rebuild
            if rebuild.candidate_tree_id != plan.roots.candidate_tree_id:
                iter_reasons.append(LogicFixedPointReason.STALE_CANDIDATE_TREE.value)
            if not rebuild.clean_rebuild_equivalent:
                iter_reasons.append(LogicFixedPointReason.REBUILD_INCOMPLETE.value)
            if not (
                rebuild.vector_row_ids
                and rebuild.kg_node_ids
                and rebuild.call_graph_id
                and rebuild.dependency_graph_id
                and rebuild.schema_graph_id
                and rebuild.value_graph_id
            ):
                iter_reasons.append(LogicFixedPointReason.REBUILD_INCOMPLETE.value)
            if logic_evidence.expected_tombstone_ids and not set(
                logic_evidence.expected_tombstone_ids
            ).issubset(rebuild.tombstone_ids):
                iter_reasons.append(LogicFixedPointReason.TOMBSTONE_MISSING.value)
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.REBUILD,
                    LogicStageDisposition.PASSED
                    if not iter_reasons
                    else LogicStageDisposition.FAILED,
                    tuple(sorted(set(iter_reasons))),
                    rebuild.receipt_id,
                    iteration_count,
                )
            )

            # Delta / closure residuals carried on the iteration receipt
            delta_reasons: list[str] = []
            if iter_receipt.unplanned_breaking_delta_ids:
                delta_reasons.append(LogicFixedPointReason.UNPLANNED_BREAKING_DELTA.value)
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.DELTA_CLOSURE,
                    LogicStageDisposition.PASSED
                    if not delta_reasons
                    else LogicStageDisposition.FAILED,
                    tuple(sorted(set(delta_reasons))),
                    iter_receipt.receipt_id,
                    iteration_count,
                )
            )
            iter_reasons.extend(delta_reasons)

            # Consumer ledger
            consumers = iter_receipt.consumer_revalidation
            cons_reasons: list[str] = []
            if consumers.candidate_tree_id != plan.roots.candidate_tree_id:
                cons_reasons.append(LogicFixedPointReason.STALE_CANDIDATE_TREE.value)
            if consumers.unresolved_mandatory_ids:
                cons_reasons.append(
                    LogicFixedPointReason.UNRESOLVED_MANDATORY_CONSUMER.value
                )
            if consumers.open_required_frontier_ids:
                cons_reasons.append(LogicFixedPointReason.UNCOVERED_FRONTIER.value)
            open_new = set(consumers.newly_resolved_consumer_ids) - set(
                consumers.discharged_new_consumer_ids
            )
            if open_new:
                cons_reasons.append(LogicFixedPointReason.NEW_RESOLVED_CONSUMER_OPEN.value)
            if not set(consumers.original_consumer_ids).issubset(
                consumers.discharged_original_ids
            ):
                cons_reasons.append(
                    LogicFixedPointReason.UNRESOLVED_MANDATORY_CONSUMER.value
                )
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.CONSUMER_LEDGER,
                    LogicStageDisposition.PASSED
                    if not cons_reasons
                    else LogicStageDisposition.FAILED,
                    tuple(sorted(set(cons_reasons))),
                    consumers.receipt_id,
                    iteration_count,
                )
            )
            iter_reasons.extend(cons_reasons)

            # Corpus / goals / gaps / Tactician
            replan = iter_receipt.replan
            plan_reasons: list[str] = []
            if replan.candidate_tree_id != plan.roots.candidate_tree_id:
                plan_reasons.append(LogicFixedPointReason.STALE_CANDIDATE_TREE.value)
            if not replan.corpus_root_id:
                plan_reasons.append(LogicFixedPointReason.CORPUS_REGENERATION_FAILED.value)
            if not replan.goal_root_ids:
                plan_reasons.append(LogicFixedPointReason.GOAL_REGENERATION_FAILED.value)
            if not replan.plan_current:
                plan_reasons.append(LogicFixedPointReason.TACTICIAN_PLAN_STALE.value)
            if replan.new_required_gap_ids:
                plan_reasons.append(LogicFixedPointReason.NEW_REQUIRED_LOGIC_GAP.value)
                residual_gaps = list(
                    dict.fromkeys(residual_gaps + list(replan.new_required_gap_ids))
                )
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.CORPUS_GOALS,
                    LogicStageDisposition.PASSED
                    if not plan_reasons
                    else LogicStageDisposition.FAILED,
                    tuple(sorted(set(plan_reasons))),
                    replan.receipt_id,
                    iteration_count,
                )
            )
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.TACTICIAN_PLAN,
                    LogicStageDisposition.PASSED
                    if replan.plan_current and not replan.new_required_gap_ids
                    else LogicStageDisposition.FAILED,
                    tuple(sorted(set(plan_reasons))),
                    replan.tactician_plan_id,
                    iteration_count,
                )
            )
            iter_reasons.extend(plan_reasons)

            # Hammer / native / countermodel / prediction
            reprove = iter_receipt.reprove
            prove_reasons: list[str] = []
            if reprove.candidate_tree_id != plan.roots.candidate_tree_id:
                prove_reasons.append(LogicFixedPointReason.STALE_CANDIDATE_TREE.value)
            if not reprove.hammer_receipt_ids:
                prove_reasons.append(LogicFixedPointReason.HAMMER_RECEIPT_MISSING.value)
            if not reprove.native_goal_binding_ids:
                prove_reasons.append(LogicFixedPointReason.NATIVE_GOAL_STALE.value)
            if reprove.stale_prediction_ids:
                prove_reasons.append(LogicFixedPointReason.PREDICTION_STALE.value)
            if reprove.failed_reconstruction_ids or not reprove.all_promoted_clauses_current:
                prove_reasons.append(LogicFixedPointReason.HAMMER_RECEIPT_MISSING.value)
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.HAMMER_REPROVE,
                    LogicStageDisposition.PASSED
                    if not prove_reasons
                    else LogicStageDisposition.FAILED,
                    tuple(sorted(set(prove_reasons))),
                    reprove.receipt_id,
                    iteration_count,
                )
            )
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.PREDICTION_REVALIDATE,
                    LogicStageDisposition.PASSED
                    if not reprove.stale_prediction_ids
                    else LogicStageDisposition.FAILED,
                    (LogicFixedPointReason.PREDICTION_STALE.value,)
                    if reprove.stale_prediction_ids
                    else (),
                    reprove.receipt_id,
                    iteration_count,
                )
            )
            iter_reasons.extend(prove_reasons)

            # Value / behavior / placement + policy tools
            vb_reasons: list[str] = []
            if consumers.failed_value_behavior_placement_ids:
                vb_reasons.append(LogicFixedPointReason.VALUE_BEHAVIOR_FAILED.value)
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.VALUE_BEHAVIOR_PLACEMENT,
                    LogicStageDisposition.PASSED
                    if not vb_reasons
                    else LogicStageDisposition.FAILED,
                    tuple(sorted(set(vb_reasons))),
                    consumers.receipt_id,
                    iteration_count,
                )
            )
            tool_reasons: list[str] = []
            if consumers.failed_policy_tool_ids:
                tool_reasons.append(LogicFixedPointReason.POLICY_TOOL_FAILED.value)
            stages.append(
                LogicStageResult(
                    LogicFixedPointStage.POLICY_TOOLS,
                    LogicStageDisposition.PASSED
                    if not tool_reasons
                    else LogicStageDisposition.FAILED,
                    tuple(sorted(set(tool_reasons))),
                    consumers.receipt_id,
                    iteration_count,
                )
            )
            iter_reasons.extend(vb_reasons)
            iter_reasons.extend(tool_reasons)

            # Residuals are per-iteration: a later iteration may clear them.
            residual_gaps = list(iter_receipt.residual_logic_gap_ids)
            unsupported_gaps = list(iter_receipt.unsupported_logic_gap_ids)

            last_goal_roots = replan.goal_root_ids
            last_corpus = (replan.corpus_root_id,)
            last_tactician = (replan.tactician_plan_id,)
            last_hammer = reprove.hammer_receipt_ids
            last_predictions = reprove.prediction_receipt_ids
            last_original_coverage = consumers.discharged_original_ids
            last_second_order_coverage = tuple(
                sorted(
                    set(consumers.discharged_new_consumer_ids)
                    | set(consumers.second_order_consumer_ids)
                )
            )

            if iter_reasons:
                reasons.update(iter_reasons)
                accepted_receipts.append(iter_receipt)
                break

            accepted_receipts.append(iter_receipt)

            if iter_receipt.requires_another_iteration or residual_gaps:
                stages.append(
                    LogicStageResult(
                        LogicFixedPointStage.FIXED_POINT,
                        LogicStageDisposition.CONTINUE,
                        (LogicFixedPointReason.FIXED_POINT_NOT_REACHED.value,),
                        iter_receipt.receipt_id,
                        iteration_count,
                    )
                )
                continue

            # Residual-free iteration.
            if (
                not residual_gaps
                and not unsupported_gaps
                and consumers.complete
                and replan.plan_current
                and not replan.new_required_gap_ids
                and reprove.all_promoted_clauses_current
                and not iter_receipt.unplanned_breaking_delta_ids
            ):
                reached = True
                stages.append(
                    LogicStageResult(
                        LogicFixedPointStage.FIXED_POINT,
                        LogicStageDisposition.PASSED,
                        (),
                        f"logic-fixed-point:iter-{iteration_count}",
                        iteration_count,
                    )
                )
                break

            reasons.add(LogicFixedPointReason.FIXED_POINT_NOT_REACHED.value)
            break
        else:
            if not reached:
                if iteration_count >= bound:
                    reasons.add(LogicFixedPointReason.BOUND_EXHAUSTED.value)
                else:
                    reasons.add(LogicFixedPointReason.FIXED_POINT_NOT_REACHED.value)

        if not reached and iteration_count >= bound:
            reasons.add(LogicFixedPointReason.BOUND_EXHAUSTED.value)

        if reasons or not reached:
            return self._rollback_or_incomplete(
                plan,
                transaction,
                stages,
                reasons
                or {LogicFixedPointReason.FIXED_POINT_NOT_REACHED.value},
                iteration_count=iteration_count,
                program_complete=True,
                program_outcome=program_outcome,
                completion=program_outcome.completion,
                iteration_receipts=tuple(accepted_receipts),
                residual_gaps=residual_gaps,
                unsupported_gaps=unsupported_gaps,
                logic_roots=logic_evidence.logic_roots,
                restore_adapter=restore_adapter,
                checkpoint=checkpoint or (
                    execution_report.checkpoint if execution_report else None
                ),
                execution_report=execution_report,
                goal_roots=last_goal_roots,
                corpus_roots=last_corpus,
                tactician=last_tactician,
                hammer=last_hammer,
                predictions=last_predictions,
                original_coverage=last_original_coverage,
                second_order_coverage=last_second_order_coverage,
            )

        # --- Finalize ---
        completion = program_outcome.completion
        fixed_point_id = (
            completion.fixed_point_receipt.receipt_id
            if completion.fixed_point_receipt is not None
            else content_identity(
                {
                    "schema": "logic-fixed-point",
                    "plan_id": plan.plan_id,
                    "iteration": iteration_count,
                }
            )
        )
        checkpoint_id = transaction.checkpoint_id or (
            execution_report.checkpoint.checkpoint_id
            if execution_report is not None
            else "checkpoint:missing"
        )
        finalize_preimage = {
            "schema": PROPAGATION_FINALIZE_RECEIPT_SCHEMA,
            "plan_id": plan.plan_id,
            "transaction_id": transaction.transaction_id,
            "completion_id": completion.completion_id,
            "checkpoint_id": checkpoint_id,
            "iteration_count": max(iteration_count, 1),
            "fixed_point_receipt_id": fixed_point_id,
        }
        finalize = PropagationFinalizeReceipt(
            roots=plan.roots,
            finalize_id=content_identity(finalize_preimage),
            transaction_id=transaction.transaction_id,
            plan_id=plan.plan_id,
            completion_id=completion.completion_id,
            checkpoint_id=checkpoint_id,
            iteration_count=max(iteration_count, 1),
            fixed_point_receipt_id=fixed_point_id,
            diagnostic_refs=tuple(item.receipt_id for item in accepted_receipts),
        )
        stages.append(
            LogicStageResult(
                LogicFixedPointStage.FINALIZE,
                LogicStageDisposition.FINALIZED,
                (),
                finalize.finalize_id,
                iteration_count,
            )
        )

        completion_id = completion.completion_id
        if contract_repair_completion_id:
            completion_id = _identifier(
                contract_repair_completion_id, "contract_repair_completion_id"
            )

        attachment = LogicFixedPointEvidenceAttachment(
            roots=logic_evidence.logic_roots,
            attachment_id=content_identity(
                {
                    "schema": "logic-fp-attachment",
                    "completion_id": completion_id,
                    "finalize_id": finalize.finalize_id,
                    "iterations": [item.receipt_id for item in accepted_receipts],
                }
            ),
            completion_receipt_id=completion_id,
            disposition=FixedPointAttachmentDisposition.ATTACHED,
            iteration_count=max(iteration_count, 1),
            goal_root_ids=last_goal_roots,
            corpus_root_ids=last_corpus,
            tactician_plan_ids=last_tactician,
            hammer_receipt_ids=last_hammer,
            prediction_receipt_ids=last_predictions,
            original_consumer_coverage_ids=last_original_coverage,
            second_order_consumer_coverage_ids=last_second_order_coverage,
            residual_logic_gap_ids=(),
            unsupported_logic_gap_ids=(),
            finalize_receipt_id=finalize.finalize_id,
            compensating_rollback_receipt_id="",
            replaces_completion=False,
            invalidation_refs=plan.invalidation_refs
            or (plan.roots.candidate_tree_id,),
        )

        report = LogicRepairFixedPointReport(
            plan_id=plan.plan_id,
            transaction_id=transaction.transaction_id,
            candidate_tree_id=plan.roots.candidate_tree_id,
            roots=plan.roots,
            stages=tuple(stages),
            reason_codes=(),
            iteration_count=iteration_count,
            complete=True,
            iteration_receipts=tuple(accepted_receipts),
            program_complete=True,
        )
        return LogicRepairFixedPointOutcome(
            report=report,
            completion=completion,
            logic_attachment=attachment,
            finalize=finalize,
            compensating_rollback=None,
            program_outcome=program_outcome,
            rolled_back=False,
        )

    def require_complete(self, *args: Any, **kwargs: Any) -> PropagationCompletionReceipt:
        return self.validate(*args, **kwargs).require_complete()

    def validate_contract_repair_via_propagation(
        self,
        plan: AtomicPropagationPlan,
        transaction: PropagationTransaction,
        *,
        program_evidence: CandidatePropagationEvidence,
        logic_evidence: CandidateLogicRepairEvidence,
        packet: ChangePropagationEditPacket | None = None,
        execution_report: TransactionExecutionReport | None = None,
        fixed_point_bound: int | None = None,
        restore_adapter: Any | None = None,
        checkpoint: PropagationCheckpoint | None = None,
        contract_repair_completion_id: str = "",
    ) -> LogicRepairFixedPointOutcome:
        """Route broken-contract repair through atomic propagation after admission.

        Requires logic evidence (always).  Same finalize/compensating-rollback
        protocol as :meth:`validate`.
        """

        return self.validate(
            plan,
            transaction,
            program_evidence=program_evidence,
            logic_evidence=logic_evidence,
            packet=packet,
            execution_report=execution_report,
            fixed_point_bound=fixed_point_bound,
            restore_adapter=restore_adapter,
            checkpoint=checkpoint,
            contract_repair_completion_id=contract_repair_completion_id,
        )

    # --- internals ---

    def _malformed_outcome(self) -> LogicRepairFixedPointOutcome:
        roots = PropagationAuthorityRoots(
            repository_id="repository:invalid",
            base_forest_id="forest:invalid",
            base_tree_id="tree:invalid-base",
            base_overlay_id="overlay:invalid-base",
            candidate_forest_id="forest:invalid-cand",
            candidate_tree_id="tree:invalid-cand",
            candidate_overlay_id="overlay:invalid-cand",
            graph_id="graph:invalid",
            index_id="index:invalid",
            model_id="model:invalid",
            config_id="config:invalid",
            translator_id="translator:invalid",
            toolchain_id="toolchain:invalid",
            policy_id="policy:invalid",
        )
        report = LogicRepairFixedPointReport(
            plan_id="plan:invalid",
            transaction_id="txn:invalid",
            candidate_tree_id="tree:invalid-cand",
            roots=roots,
            stages=(
                LogicStageResult(
                    LogicFixedPointStage.PROGRAM_VALIDATION,
                    LogicStageDisposition.FAILED,
                    (LogicFixedPointReason.MALFORMED_INPUT.value,),
                ),
            ),
            reason_codes=(LogicFixedPointReason.MALFORMED_INPUT.value,),
            iteration_count=0,
            complete=False,
        )
        return LogicRepairFixedPointOutcome(report=report)

    def _incomplete(
        self,
        plan: AtomicPropagationPlan,
        transaction: PropagationTransaction,
        stages: Sequence[LogicStageResult],
        reasons: set[str],
        *,
        iteration_count: int,
        program_complete: bool,
        program_outcome: PropagationValidationOutcome | None = None,
        completion: PropagationCompletionReceipt | None = None,
        iteration_receipts: tuple[LogicRepairIterationReceipt, ...] = (),
    ) -> LogicRepairFixedPointOutcome:
        report = LogicRepairFixedPointReport(
            plan_id=plan.plan_id,
            transaction_id=transaction.transaction_id,
            candidate_tree_id=plan.roots.candidate_tree_id,
            roots=plan.roots,
            stages=tuple(stages),
            reason_codes=tuple(sorted(reasons)),
            iteration_count=iteration_count,
            complete=False,
            iteration_receipts=iteration_receipts,
            program_complete=program_complete,
        )
        return LogicRepairFixedPointOutcome(
            report=report,
            completion=completion,
            program_outcome=program_outcome,
            rolled_back=False,
        )

    def _rollback_or_incomplete(
        self,
        plan: AtomicPropagationPlan,
        transaction: PropagationTransaction,
        stages: list[LogicStageResult],
        reasons: set[str],
        *,
        iteration_count: int,
        program_complete: bool,
        program_outcome: PropagationValidationOutcome | None = None,
        completion: PropagationCompletionReceipt | None = None,
        iteration_receipts: tuple[LogicRepairIterationReceipt, ...] = (),
        residual_gaps: Sequence[str] = (),
        unsupported_gaps: Sequence[str] = (),
        logic_roots: ProgramLogicAuthorityRoots | None = None,
        restore_adapter: Any | None = None,
        checkpoint: PropagationCheckpoint | None = None,
        execution_report: TransactionExecutionReport | None = None,
        goal_roots: Sequence[str] = (),
        corpus_roots: Sequence[str] = (),
        tactician: Sequence[str] = (),
        hammer: Sequence[str] = (),
        predictions: Sequence[str] = (),
        original_coverage: Sequence[str] = (),
        second_order_coverage: Sequence[str] = (),
    ) -> LogicRepairFixedPointOutcome:
        """After provisional commit, failure/incompleteness triggers compensating rollback."""

        # Prefer checkpoint from execution report when not supplied.
        if checkpoint is None and execution_report is not None:
            checkpoint = execution_report.checkpoint

        compensating: CompensatingRollbackReceipt | None = None
        rolled_back = False
        attachment: LogicFixedPointEvidenceAttachment | None = None

        if (
            transaction.state is TransactionState.COMMITTED
            and checkpoint is not None
        ):
            txn_engine = ChangePropagationTransaction(
                restore_adapter=restore_adapter or (lambda _cp: True),
            )
            try:
                rollback = txn_engine.compensating_rollback(
                    plan=plan,
                    transaction=transaction,
                    checkpoint=checkpoint,
                    reason_codes=tuple(sorted(reasons)),
                    diagnostic_refs=tuple(
                        item.receipt_id for item in iteration_receipts
                    ),
                )
                compensating = CompensatingRollbackReceipt(
                    roots=plan.roots,
                    rollback_id=content_identity(
                        {
                            "schema": COMPENSATING_ROLLBACK_RECEIPT_SCHEMA,
                            "transaction_id": transaction.transaction_id,
                            "checkpoint_id": checkpoint.checkpoint_id,
                            "reasons": sorted(reasons),
                            "underlying": rollback.rollback_id
                            if isinstance(rollback, PropagationRollbackReceipt)
                            else "",
                        }
                    ),
                    transaction_id=transaction.transaction_id,
                    plan_id=plan.plan_id,
                    checkpoint_id=checkpoint.checkpoint_id,
                    restored=bool(
                        getattr(rollback, "restored", True)
                        if rollback is not None
                        else True
                    ),
                    reason_codes=tuple(sorted(reasons)),
                    iteration_count=iteration_count,
                    diagnostic_refs=tuple(
                        item.receipt_id for item in iteration_receipts
                    ),
                    underlying_rollback_id=(
                        rollback.rollback_id
                        if isinstance(rollback, PropagationRollbackReceipt)
                        else ""
                    ),
                )
                rolled_back = compensating.restored
                if not compensating.restored:
                    reasons.add(LogicFixedPointReason.ROLLBACK_FAILED.value)
                stages.append(
                    LogicStageResult(
                        LogicFixedPointStage.COMPENSATING_ROLLBACK,
                        LogicStageDisposition.ROLLED_BACK
                        if rolled_back
                        else LogicStageDisposition.FAILED,
                        tuple(sorted(reasons)),
                        compensating.rollback_id,
                        iteration_count,
                    )
                )
            except Exception:
                reasons.add(LogicFixedPointReason.ROLLBACK_FAILED.value)
                stages.append(
                    LogicStageResult(
                        LogicFixedPointStage.COMPENSATING_ROLLBACK,
                        LogicStageDisposition.FAILED,
                        (LogicFixedPointReason.ROLLBACK_FAILED.value,),
                        "",
                        iteration_count,
                    )
                )

        if logic_roots is not None and completion is not None:
            disposition = (
                FixedPointAttachmentDisposition.ROLLED_BACK
                if rolled_back and compensating is not None
                else FixedPointAttachmentDisposition.INCOMPLETE
                if not residual_gaps and not unsupported_gaps
                else FixedPointAttachmentDisposition.RESIDUAL
            )
            try:
                attachment = LogicFixedPointEvidenceAttachment(
                    roots=logic_roots,
                    attachment_id=content_identity(
                        {
                            "schema": "logic-fp-attachment-fail",
                            "completion_id": completion.completion_id,
                            "reasons": sorted(reasons),
                            "iterations": [item.receipt_id for item in iteration_receipts],
                        }
                    ),
                    completion_receipt_id=completion.completion_id,
                    disposition=disposition,
                    iteration_count=max(iteration_count, 1),
                    goal_root_ids=tuple(goal_roots),
                    corpus_root_ids=tuple(corpus_roots),
                    tactician_plan_ids=tuple(tactician),
                    hammer_receipt_ids=tuple(hammer),
                    prediction_receipt_ids=tuple(predictions),
                    original_consumer_coverage_ids=tuple(original_coverage),
                    second_order_consumer_coverage_ids=tuple(second_order_coverage),
                    residual_logic_gap_ids=tuple(residual_gaps)
                    if disposition is FixedPointAttachmentDisposition.RESIDUAL
                    else (),
                    unsupported_logic_gap_ids=tuple(unsupported_gaps)
                    if disposition is FixedPointAttachmentDisposition.RESIDUAL
                    else (),
                    finalize_receipt_id="",
                    compensating_rollback_receipt_id=(
                        compensating.rollback_id if compensating is not None else ""
                    ),
                    replaces_completion=False,
                    invalidation_refs=plan.invalidation_refs
                    or (plan.roots.candidate_tree_id,),
                )
                # RESIDUAL requires residual/unsupported gaps; if disposition
                # is residual but both empty, fall back to incomplete.
                if (
                    disposition is FixedPointAttachmentDisposition.RESIDUAL
                    and not residual_gaps
                    and not unsupported_gaps
                ):
                    attachment = None
            except Exception:
                attachment = None

        report = LogicRepairFixedPointReport(
            plan_id=plan.plan_id,
            transaction_id=transaction.transaction_id,
            candidate_tree_id=plan.roots.candidate_tree_id,
            roots=plan.roots,
            stages=tuple(stages),
            reason_codes=tuple(sorted(reasons)),
            iteration_count=iteration_count,
            complete=False,
            iteration_receipts=iteration_receipts,
            program_complete=program_complete,
        )
        return LogicRepairFixedPointOutcome(
            report=report,
            completion=completion,
            logic_attachment=attachment,
            finalize=None,
            compensating_rollback=compensating,
            program_outcome=program_outcome,
            rolled_back=rolled_back,
        )


def validate_logic_repair_fixed_point(
    plan: AtomicPropagationPlan,
    transaction: PropagationTransaction,
    *,
    program_evidence: CandidatePropagationEvidence,
    logic_evidence: CandidateLogicRepairEvidence | None = None,
    packet: ChangePropagationEditPacket | None = None,
    fixed_point_bound: int | None = None,
    require_logic_evidence: bool = False,
) -> LogicRepairFixedPointOutcome:
    """Module entry point matching :meth:`LogicRepairFixedPointValidator.validate`."""

    return LogicRepairFixedPointValidator(
        require_logic_evidence=require_logic_evidence
    ).validate(
        plan,
        transaction,
        program_evidence=program_evidence,
        logic_evidence=logic_evidence,
        packet=packet,
        fixed_point_bound=fixed_point_bound,
    )


def daemon_require_logic_fixed_point(
    plan: Any,
    transaction: Any,
    *,
    program_evidence: Any,
    logic_evidence: Any = None,
    packet: Any = None,
    execution_report: Any = None,
    fixed_point_bound: int | None = None,
    restore_adapter: Any = None,
    checkpoint: Any = None,
    require_logic_evidence: bool = False,
) -> Any:
    """Daemon-facing fail-closed joint fixed-point gate."""

    outcome = LogicRepairFixedPointValidator(
        require_logic_evidence=require_logic_evidence
    ).validate(
        plan,
        transaction,
        program_evidence=program_evidence,
        logic_evidence=logic_evidence,
        packet=packet,
        execution_report=execution_report,
        fixed_point_bound=fixed_point_bound,
        restore_adapter=restore_adapter,
        checkpoint=checkpoint,
    )
    if logic_evidence is not None:
        return outcome.require_complete()
    # Program-only path: require program completion without logic attachment.
    if outcome.completion is None or outcome.completion.disposition is not (
        CompletionDisposition.COMPLETE
    ):
        reasons = ", ".join(outcome.report.reason_codes) or "incomplete"
        raise LogicRepairFixedPointError(
            "logic repair fixed-point validation rejected: " + reasons
        )
    return outcome.completion


__all__ = [
    "LOGIC_REPAIR_FIXED_POINT_INTERFACE",
    "LOGIC_REPAIR_FIXED_POINT_REPORT_SCHEMA",
    "LOGIC_REPAIR_ITERATION_RECEIPT_SCHEMA",
    "PROPAGATION_FINALIZE_RECEIPT_SCHEMA",
    "COMPENSATING_ROLLBACK_RECEIPT_SCHEMA",
    "PRODUCER_ID",
    "DEFAULT_LOGIC_FIXED_POINT_BOUND",
    "CandidateLogicRepairEvidence",
    "CompensatingRollbackReceipt",
    "LogicConsumerRevalidationEvidence",
    "LogicFixedPointReason",
    "LogicFixedPointStage",
    "LogicRebuildEvidence",
    "LogicRepairFixedPointError",
    "LogicRepairFixedPointOutcome",
    "LogicRepairFixedPointReport",
    "LogicRepairFixedPointValidator",
    "LogicRepairIterationReceipt",
    "LogicReplanEvidence",
    "LogicReproveEvidence",
    "LogicStageDisposition",
    "LogicStageResult",
    "PropagationFinalizeReceipt",
    "daemon_require_logic_fixed_point",
    "validate_logic_repair_fixed_point",
    # Canonical re-exports
    "LogicFixedPointEvidenceAttachment",
    "FixedPointAttachmentDisposition",
]
