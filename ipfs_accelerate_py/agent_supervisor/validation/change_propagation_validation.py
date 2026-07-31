"""Fixed-point post-mutation completion gate for change propagation.

A committed transaction is not completion.  Completion requires the candidate
tree to repeatedly:

1. rebuild repository, AST, vector, and graph rows and tombstones;
2. re-extract the base/candidate semantic delta;
3. re-resolve calls, data/value flow, constructors, schemas, and wiring;
4. recompute the reverse impact closure and unknown frontier;
5. verify every original consumer obligation is discharged exactly once;
6. discover second-order impacts / new deltas and iterate to a policy bound;
7. reconstruct original and introduced logic obligations;
8. run type/schema/effect/capability/resource/memory tools and
   dependency-complete tests without accepting weakened or deleted checks;
9. emit a candidate-tree-bound :class:`PropagationCompletionReceipt`.

Canonical RPR-022 records (:class:`AtomicPropagationPlan`,
:class:`PropagationCompletionReceipt`, :class:`FixedPointReceipt`) are
imported and returned — never redefined.  Bound exhaustion, skipped required
tools, and weakened/deleted checks fail closed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from ..analysis.change_propagation_contracts import (
    AtomicPropagationPlan,
    CompletionDisposition,
    ConsumerDisposition,
    FixedPointReceipt,
    PlanDisposition,
    PropagationAuthorityRoots,
    PropagationCompletionReceipt,
    PropagationTransaction,
    TransactionState,
)
from ..planning.change_propagation_transaction import (
    TransactionExecutionReport,
    TransactionLease,
)
from ..proof.change_propagation_edit_packet import ChangePropagationEditPacket
from ..proof.formal_verification_contracts import content_identity
from .contract_repair_validation import (
    DEFAULT_POLICY_REQUIRED_TOOLS,
    POLICY_TOOL_FAMILIES,
    ImpactedTestEvidence,
    IntegrityEvidence,
    PolicyToolEvidence,
    ToolGateResult,
    build_passing_tool_evidence,
)


CHANGE_PROPAGATION_VALIDATOR_INTERFACE: Final[str] = "ChangePropagationValidator@1"
PROPAGATION_VALIDATION_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/validation-report@1"
)
PROPAGATION_INDEX_REBUILD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/index-rebuild@1"
)
PROPAGATION_DELTA_REEXTRACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/delta-reextract@1"
)
PROPAGATION_RESOLUTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/resolution@1"
)
PROPAGATION_CLOSURE_RECOMPUTE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/closure-recompute@1"
)
PROPAGATION_CONSUMER_DISCHARGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/consumer-discharge@1"
)
PROPAGATION_SECOND_ORDER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/second-order@1"
)
PROPAGATION_PROOF_RECONSTRUCTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/proof-reconstruction@1"
)

PRODUCER_ID: Final[str] = "change-propagation-validation@1"

MAX_PATHS: Final[int] = 1_024
MAX_IDS: Final[int] = 1_024
MAX_REASON_CODES: Final[int] = 64
MAX_ITERATIONS: Final[int] = 32
DEFAULT_FIXED_POINT_BOUND: Final[int] = 8
MAX_TEXT_BYTES: Final[int] = 4_096


class ChangePropagationValidationError(ValueError):
    """The candidate tree failed the fixed-point completion gate."""


class PropagationValidationReason(str, Enum):
    """Stable, machine-readable fixed-point failure codes."""

    MALFORMED_INPUT = "malformed_input"
    PLAN_NOT_ADMITTED = "plan_not_admitted"
    TRANSACTION_NOT_COMMITTED = "transaction_not_committed"
    ROOT_DRIFT = "root_drift"
    STALE_CANDIDATE_TREE = "stale_candidate_tree"
    PLAN_PACKET_MISMATCH = "plan_packet_mismatch"
    INDEX_REBUILD_INCOMPLETE = "index_rebuild_incomplete"
    TOMBSTONE_MISSING = "tombstone_missing"
    DELTA_REEXTRACT_FAILED = "delta_reextract_failed"
    UNPLANNED_BREAKING_DELTA = "unplanned_breaking_delta"
    RESOLUTION_INCOMPLETE = "resolution_incomplete"
    CLOSURE_INCOMPLETE = "closure_incomplete"
    UNCOVERED_FRONTIER = "uncovered_frontier"
    CONSUMER_NOT_DISCHARGED = "consumer_not_discharged"
    CONSUMER_DOUBLE_DISCHARGE = "consumer_double_discharge"
    UNRESOLVED_MANDATORY = "unresolved_mandatory"
    OMITTED_DEPENDENT = "omitted_dependent"
    SECOND_ORDER_RESIDUAL = "second_order_residual"
    PROOF_RECONSTRUCTION_FAILED = "proof_reconstruction_failed"
    SKIPPED_REQUIRED_TOOL = "skipped_required_tool"
    TOOL_FAILED = "tool_failed"
    FOCUSED_TEST_FAILED = "focused_test_failed"
    IMPACTED_TEST_FAILED = "impacted_test_failed"
    IMPACTED_TEST_OMITTED = "impacted_test_omitted"
    TEST_DELETED = "test_deleted"
    TEST_WEAKENED = "test_weakened"
    CHECKER_DELETED = "checker_deleted"
    CHECKER_WEAKENED = "checker_weakened"
    CONTRACT_DELETED = "contract_deleted"
    CONTRACT_WEAKENED = "contract_weakened"
    BOUND_EXHAUSTED = "bound_exhausted"
    FIXED_POINT_NOT_REACHED = "fixed_point_not_reached"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    PARTIAL_COMPLETION_FORBIDDEN = "partial_completion_forbidden"


class ValidationStage(str, Enum):
    """Ordered stages of one fixed-point iteration."""

    INDEX_REBUILD = "index_rebuild"
    DELTA_REEXTRACT = "delta_reextract"
    RESOLUTION = "resolution"
    CLOSURE_RECOMPUTE = "closure_recompute"
    CONSUMER_DISCHARGE = "consumer_discharge"
    SECOND_ORDER = "second_order"
    PROOF_RECONSTRUCTION = "proof_reconstruction"
    POLICY_TOOLS = "policy_tools"
    IMPACTED_TESTS = "impacted_tests"
    INTEGRITY = "integrity"
    FIXED_POINT = "fixed_point"
    COMPLETION = "completion"


class StageDisposition(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CONTINUE = "continue"  # residual found; another iteration required


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise ChangePropagationValidationError(f"{name} must be a compact identifier")
    text = value.strip()
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ChangePropagationValidationError(f"{name} exceeds text bound")
    return text


def _paths(values: Sequence[str], name: str, *, required: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationValidationError(f"{name} must be a path sequence")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ChangePropagationValidationError(f"{name} contains an invalid path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise ChangePropagationValidationError(f"{name} contains an escaped path")
        result.add(path.as_posix())
    if required and not result:
        raise ChangePropagationValidationError(f"{name} must not be empty")
    if len(result) > MAX_PATHS:
        raise ChangePropagationValidationError(f"{name} exceeds path bound")
    return tuple(sorted(result))


def _ids(
    values: Sequence[str],
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_IDS,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationValidationError(f"{name} must be an identifier sequence")
    if preserve_order:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            item = value.strip()
            if any(char.isspace() for char in item):
                raise ChangePropagationValidationError(
                    f"{name} must contain compact identifiers"
                )
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        result = tuple(ordered)
    else:
        result = tuple(
            sorted(
                {
                    value.strip()
                    for value in values
                    if isinstance(value, str) and value.strip()
                }
            )
        )
    if required and not result:
        raise ChangePropagationValidationError(f"{name} must not be empty")
    if len(result) > maximum:
        raise ChangePropagationValidationError(f"{name} exceeds item bound")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ChangePropagationValidationError(f"{name} must be a boolean")
    return value


def _bounded_int(value: Any, name: str, *, minimum: int = 0, maximum: int = MAX_ITERATIONS) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChangePropagationValidationError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ChangePropagationValidationError(f"{name} out of bounds [{minimum}, {maximum}]")
    return value


# ---------------------------------------------------------------------------
# Stage evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagationIndexRebuildEvidence:
    """Proof that repository/AST/vector/graph rows and tombstones were rebuilt."""

    candidate_tree_id: str
    index_id: str
    graph_id: str
    rebuilt_source_paths: tuple[str, ...]
    rebuilt_ast_paths: tuple[str, ...]
    rebuilt_vector_row_ids: tuple[str, ...]
    rebuilt_graph_node_ids: tuple[str, ...]
    tombstone_ids: tuple[str, ...]
    affected_paths: tuple[str, ...]
    clean_rebuild_equivalent: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id")
        )
        object.__setattr__(self, "index_id", _identifier(self.index_id, "index_id"))
        object.__setattr__(self, "graph_id", _identifier(self.graph_id, "graph_id"))
        object.__setattr__(
            self, "rebuilt_source_paths", _paths(self.rebuilt_source_paths, "rebuilt_source_paths")
        )
        object.__setattr__(
            self, "rebuilt_ast_paths", _paths(self.rebuilt_ast_paths, "rebuilt_ast_paths")
        )
        object.__setattr__(
            self,
            "rebuilt_vector_row_ids",
            _ids(self.rebuilt_vector_row_ids, "rebuilt_vector_row_ids"),
        )
        object.__setattr__(
            self,
            "rebuilt_graph_node_ids",
            _ids(self.rebuilt_graph_node_ids, "rebuilt_graph_node_ids"),
        )
        object.__setattr__(
            self, "tombstone_ids", _ids(self.tombstone_ids, "tombstone_ids", required=False)
        )
        object.__setattr__(self, "affected_paths", _paths(self.affected_paths, "affected_paths"))
        object.__setattr__(
            self,
            "clean_rebuild_equivalent",
            _bool(self.clean_rebuild_equivalent, "clean_rebuild_equivalent"),
        )
        if not set(self.affected_paths).issubset(self.rebuilt_source_paths):
            raise ChangePropagationValidationError(
                "index rebuild must cover every affected source path"
            )
        if not set(self.affected_paths).issubset(self.rebuilt_ast_paths):
            raise ChangePropagationValidationError(
                "index rebuild must cover every affected AST path"
            )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROPAGATION_INDEX_REBUILD_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "index_id": self.index_id,
            "graph_id": self.graph_id,
            "rebuilt_source_paths": list(self.rebuilt_source_paths),
            "rebuilt_ast_paths": list(self.rebuilt_ast_paths),
            "rebuilt_vector_row_ids": list(self.rebuilt_vector_row_ids),
            "rebuilt_graph_node_ids": list(self.rebuilt_graph_node_ids),
            "tombstone_ids": list(self.tombstone_ids),
            "affected_paths": list(self.affected_paths),
            "clean_rebuild_equivalent": self.clean_rebuild_equivalent,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class DeltaReextractEvidence:
    """Proof that the base/candidate semantic delta was re-extracted."""

    candidate_tree_id: str
    original_delta_id: str
    reextracted_delta_id: str
    breaking_delta_ids: tuple[str, ...]
    unplanned_breaking_delta_ids: tuple[str, ...]
    extraction_receipt_id: str
    matches_plan_delta: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id")
        )
        object.__setattr__(
            self, "original_delta_id", _identifier(self.original_delta_id, "original_delta_id")
        )
        object.__setattr__(
            self,
            "reextracted_delta_id",
            _identifier(self.reextracted_delta_id, "reextracted_delta_id"),
        )
        object.__setattr__(
            self,
            "breaking_delta_ids",
            _ids(self.breaking_delta_ids, "breaking_delta_ids", required=False),
        )
        object.__setattr__(
            self,
            "unplanned_breaking_delta_ids",
            _ids(
                self.unplanned_breaking_delta_ids,
                "unplanned_breaking_delta_ids",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "extraction_receipt_id",
            _identifier(self.extraction_receipt_id, "extraction_receipt_id"),
        )
        object.__setattr__(
            self, "matches_plan_delta", _bool(self.matches_plan_delta, "matches_plan_delta")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_DELTA_REEXTRACT_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "original_delta_id": self.original_delta_id,
            "reextracted_delta_id": self.reextracted_delta_id,
            "breaking_delta_ids": list(self.breaking_delta_ids),
            "unplanned_breaking_delta_ids": list(self.unplanned_breaking_delta_ids),
            "extraction_receipt_id": self.extraction_receipt_id,
            "matches_plan_delta": self.matches_plan_delta,
        }


@dataclass(frozen=True)
class ResolutionEvidence:
    """Proof that calls/data/schema/wiring re-resolve on the candidate tree."""

    candidate_tree_id: str
    resolved_call_ids: tuple[str, ...]
    resolved_data_flow_ids: tuple[str, ...]
    resolved_schema_ids: tuple[str, ...]
    resolved_wiring_ids: tuple[str, ...]
    unresolved_ids: tuple[str, ...]
    resolution_receipt_id: str
    complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id")
        )
        for name in (
            "resolved_call_ids",
            "resolved_data_flow_ids",
            "resolved_schema_ids",
            "resolved_wiring_ids",
            "unresolved_ids",
        ):
            object.__setattr__(
                self,
                name,
                _ids(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self,
            "resolution_receipt_id",
            _identifier(self.resolution_receipt_id, "resolution_receipt_id"),
        )
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        if self.complete and self.unresolved_ids:
            raise ChangePropagationValidationError(
                "complete resolution forbids unresolved ids"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_RESOLUTION_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "resolved_call_ids": list(self.resolved_call_ids),
            "resolved_data_flow_ids": list(self.resolved_data_flow_ids),
            "resolved_schema_ids": list(self.resolved_schema_ids),
            "resolved_wiring_ids": list(self.resolved_wiring_ids),
            "unresolved_ids": list(self.unresolved_ids),
            "resolution_receipt_id": self.resolution_receipt_id,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class ClosureRecomputeEvidence:
    """Proof that reverse impact closure and frontier were recomputed."""

    candidate_tree_id: str
    original_closure_id: str
    recomputed_closure_id: str
    consumer_ids: tuple[str, ...]
    mandatory_consumer_ids: tuple[str, ...]
    frontier_node_ids: tuple[str, ...]
    required_frontier_ids: tuple[str, ...]
    uncovered_frontier_ids: tuple[str, ...]
    complete: bool
    receipt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id")
        )
        object.__setattr__(
            self, "original_closure_id", _identifier(self.original_closure_id, "original_closure_id")
        )
        object.__setattr__(
            self,
            "recomputed_closure_id",
            _identifier(self.recomputed_closure_id, "recomputed_closure_id"),
        )
        for name in (
            "consumer_ids",
            "mandatory_consumer_ids",
            "frontier_node_ids",
            "required_frontier_ids",
            "uncovered_frontier_ids",
        ):
            object.__setattr__(
                self, name, _ids(getattr(self, name), name, required=False)
            )
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        object.__setattr__(self, "receipt_id", _identifier(self.receipt_id, "receipt_id"))
        if self.complete and self.uncovered_frontier_ids:
            raise ChangePropagationValidationError(
                "complete closure forbids uncovered required frontier"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_CLOSURE_RECOMPUTE_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "original_closure_id": self.original_closure_id,
            "recomputed_closure_id": self.recomputed_closure_id,
            "consumer_ids": list(self.consumer_ids),
            "mandatory_consumer_ids": list(self.mandatory_consumer_ids),
            "frontier_node_ids": list(self.frontier_node_ids),
            "required_frontier_ids": list(self.required_frontier_ids),
            "uncovered_frontier_ids": list(self.uncovered_frontier_ids),
            "complete": self.complete,
            "receipt_id": self.receipt_id,
        }


@dataclass(frozen=True)
class ConsumerDischargeEvidence:
    """Proof that each original consumer obligation is discharged exactly once."""

    candidate_tree_id: str
    original_obligation_ids: tuple[str, ...]
    discharged_obligation_ids: tuple[str, ...]
    unresolved_mandatory_ids: tuple[str, ...]
    omitted_dependent_ids: tuple[str, ...]
    double_discharged_ids: tuple[str, ...]
    receipt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id")
        )
        object.__setattr__(
            self,
            "original_obligation_ids",
            _ids(self.original_obligation_ids, "original_obligation_ids", required=True),
        )
        for name in (
            "discharged_obligation_ids",
            "unresolved_mandatory_ids",
            "omitted_dependent_ids",
            "double_discharged_ids",
        ):
            object.__setattr__(
                self, name, _ids(getattr(self, name), name, required=False)
            )
        object.__setattr__(self, "receipt_id", _identifier(self.receipt_id, "receipt_id"))

    @property
    def complete(self) -> bool:
        return (
            not self.unresolved_mandatory_ids
            and not self.omitted_dependent_ids
            and not self.double_discharged_ids
            and set(self.original_obligation_ids).issubset(self.discharged_obligation_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_CONSUMER_DISCHARGE_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "original_obligation_ids": list(self.original_obligation_ids),
            "discharged_obligation_ids": list(self.discharged_obligation_ids),
            "unresolved_mandatory_ids": list(self.unresolved_mandatory_ids),
            "omitted_dependent_ids": list(self.omitted_dependent_ids),
            "double_discharged_ids": list(self.double_discharged_ids),
            "receipt_id": self.receipt_id,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class SecondOrderImpactEvidence:
    """New deltas/consumers discovered after the mutation (second-order impacts)."""

    candidate_tree_id: str
    new_delta_ids: tuple[str, ...]
    new_consumer_ids: tuple[str, ...]
    residual_frontier_ids: tuple[str, ...]
    requires_another_iteration: bool
    receipt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id")
        )
        for name in ("new_delta_ids", "new_consumer_ids", "residual_frontier_ids"):
            object.__setattr__(
                self, name, _ids(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "requires_another_iteration",
            _bool(self.requires_another_iteration, "requires_another_iteration"),
        )
        object.__setattr__(self, "receipt_id", _identifier(self.receipt_id, "receipt_id"))
        has_residual = bool(
            self.new_delta_ids or self.new_consumer_ids or self.residual_frontier_ids
        )
        if self.requires_another_iteration and not has_residual:
            raise ChangePropagationValidationError(
                "requires_another_iteration needs residual second-order impacts"
            )
        if has_residual and not self.requires_another_iteration:
            # Residual without requesting another iteration is incomplete evidence.
            raise ChangePropagationValidationError(
                "residual second-order impacts require another iteration"
            )

    @property
    def is_fixed(self) -> bool:
        return not self.requires_another_iteration

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_SECOND_ORDER_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "new_delta_ids": list(self.new_delta_ids),
            "new_consumer_ids": list(self.new_consumer_ids),
            "residual_frontier_ids": list(self.residual_frontier_ids),
            "requires_another_iteration": self.requires_another_iteration,
            "receipt_id": self.receipt_id,
        }


@dataclass(frozen=True)
class ProofReconstructionEvidence:
    """Reconstructed original and introduced proof obligations."""

    candidate_tree_id: str
    original_proof_refs: tuple[str, ...]
    reconstructed_proof_refs: tuple[str, ...]
    introduced_proof_refs: tuple[str, ...]
    failed_proof_refs: tuple[str, ...]
    all_mandatory_reconstructed: bool
    receipt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id")
        )
        for name in (
            "original_proof_refs",
            "reconstructed_proof_refs",
            "introduced_proof_refs",
            "failed_proof_refs",
        ):
            required = name == "original_proof_refs"
            object.__setattr__(
                self, name, _ids(getattr(self, name), name, required=required)
            )
        object.__setattr__(
            self,
            "all_mandatory_reconstructed",
            _bool(self.all_mandatory_reconstructed, "all_mandatory_reconstructed"),
        )
        object.__setattr__(self, "receipt_id", _identifier(self.receipt_id, "receipt_id"))
        if self.all_mandatory_reconstructed and self.failed_proof_refs:
            raise ChangePropagationValidationError(
                "all_mandatory_reconstructed forbids failed proof refs"
            )
        if self.all_mandatory_reconstructed and not set(self.original_proof_refs).issubset(
            self.reconstructed_proof_refs
        ):
            raise ChangePropagationValidationError(
                "all_mandatory_reconstructed requires every original proof reconstructed"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_PROOF_RECONSTRUCTION_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "original_proof_refs": list(self.original_proof_refs),
            "reconstructed_proof_refs": list(self.reconstructed_proof_refs),
            "introduced_proof_refs": list(self.introduced_proof_refs),
            "failed_proof_refs": list(self.failed_proof_refs),
            "all_mandatory_reconstructed": self.all_mandatory_reconstructed,
            "receipt_id": self.receipt_id,
        }


@dataclass(frozen=True)
class StageResult:
    stage: ValidationStage
    disposition: StageDisposition
    reason_codes: tuple[str, ...] = ()
    evidence_id: str = ""
    iteration: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage",
            ValidationStage(self.stage)
            if not isinstance(self.stage, ValidationStage)
            else self.stage,
        )
        object.__setattr__(
            self,
            "disposition",
            StageDisposition(self.disposition)
            if not isinstance(self.disposition, StageDisposition)
            else self.disposition,
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        if self.evidence_id:
            object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        else:
            object.__setattr__(self, "evidence_id", "")
        object.__setattr__(
            self, "iteration", _bounded_int(self.iteration, "iteration", minimum=1)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "evidence_id": self.evidence_id,
            "iteration": self.iteration,
        }


@dataclass(frozen=True)
class FixedPointIterationEvidence:
    """Structured evidence for one fixed-point iteration."""

    iteration: int
    index_rebuild: PropagationIndexRebuildEvidence
    delta_reextract: DeltaReextractEvidence
    resolution: ResolutionEvidence
    closure_recompute: ClosureRecomputeEvidence
    consumer_discharge: ConsumerDischargeEvidence
    second_order: SecondOrderImpactEvidence
    proof_reconstruction: ProofReconstructionEvidence
    policy_tools: PolicyToolEvidence
    impacted_tests: ImpactedTestEvidence
    integrity: IntegrityEvidence

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "iteration", _bounded_int(self.iteration, "iteration", minimum=1)
        )
        for name, expected in (
            ("index_rebuild", PropagationIndexRebuildEvidence),
            ("delta_reextract", DeltaReextractEvidence),
            ("resolution", ResolutionEvidence),
            ("closure_recompute", ClosureRecomputeEvidence),
            ("consumer_discharge", ConsumerDischargeEvidence),
            ("second_order", SecondOrderImpactEvidence),
            ("proof_reconstruction", ProofReconstructionEvidence),
            ("policy_tools", PolicyToolEvidence),
            ("impacted_tests", ImpactedTestEvidence),
            ("integrity", IntegrityEvidence),
        ):
            value = getattr(self, name)
            if not isinstance(value, expected):
                raise ChangePropagationValidationError(
                    f"{name} must be {expected.__name__}"
                )


@dataclass(frozen=True)
class CandidatePropagationEvidence:
    """Complete evidence for fixed-point validation (one or more iterations)."""

    candidate_tree_id: str
    iterations: tuple[FixedPointIterationEvidence, ...]
    expected_tombstone_ids: tuple[str, ...] = ()
    expected_deleted_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id")
        )
        if (
            not isinstance(self.iterations, Sequence)
            or not self.iterations
            or not all(isinstance(item, FixedPointIterationEvidence) for item in self.iterations)
        ):
            raise ChangePropagationValidationError(
                "iterations must be a non-empty FixedPointIterationEvidence sequence"
            )
        if len(self.iterations) > MAX_ITERATIONS:
            raise ChangePropagationValidationError("iterations exceed policy bound")
        object.__setattr__(self, "iterations", tuple(self.iterations))
        object.__setattr__(
            self,
            "expected_tombstone_ids",
            _ids(self.expected_tombstone_ids, "expected_tombstone_ids", required=False),
        )
        object.__setattr__(
            self,
            "expected_deleted_paths",
            _paths(self.expected_deleted_paths, "expected_deleted_paths"),
        )


@dataclass(frozen=True)
class PropagationValidationReport:
    """Ordered multi-iteration gate report; success is not completion authority."""

    plan_id: str
    transaction_id: str
    candidate_tree_id: str
    roots: PropagationAuthorityRoots
    stages: tuple[StageResult, ...]
    reason_codes: tuple[str, ...]
    iteration_count: int
    complete: bool
    fixed_point_receipt: FixedPointReceipt | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "transaction_id", _identifier(self.transaction_id, "transaction_id")
        )
        object.__setattr__(
            self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id")
        )
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise ChangePropagationValidationError(
                "report roots must be PropagationAuthorityRoots"
            )
        if not isinstance(self.stages, Sequence) or not all(
            isinstance(item, StageResult) for item in self.stages
        ):
            raise ChangePropagationValidationError("stages must be StageResult values")
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
        if self.fixed_point_receipt is not None and not isinstance(
            self.fixed_point_receipt, FixedPointReceipt
        ):
            raise ChangePropagationValidationError(
                "fixed_point_receipt must be the canonical FixedPointReceipt@1"
            )
        if self.complete and self.reason_codes:
            raise ChangePropagationValidationError(
                "a complete report cannot carry failure reason codes"
            )
        if self.complete and (
            self.fixed_point_receipt is None or not self.fixed_point_receipt.is_fixed_point
        ):
            raise ChangePropagationValidationError(
                "a complete report requires a residual-free fixed-point receipt"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_VALIDATION_REPORT_SCHEMA,
            "interface": CHANGE_PROPAGATION_VALIDATOR_INTERFACE,
            "plan_id": self.plan_id,
            "transaction_id": self.transaction_id,
            "candidate_tree_id": self.candidate_tree_id,
            "roots": self.roots.to_dict(),
            "stages": [item.to_dict() for item in self.stages],
            "reason_codes": list(self.reason_codes),
            "iteration_count": self.iteration_count,
            "complete": self.complete,
            "fixed_point_receipt": (
                self.fixed_point_receipt.to_dict() if self.fixed_point_receipt else None
            ),
            "provider_success_is_not_completion": True,
        }

    @property
    def report_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "report_id": self.report_id}


@dataclass(frozen=True)
class PropagationValidationOutcome:
    """Either a completion receipt or a failed/incomplete validation report."""

    report: PropagationValidationReport
    completion: PropagationCompletionReceipt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.report, PropagationValidationReport):
            raise ChangePropagationValidationError(
                "outcome requires a PropagationValidationReport"
            )
        if self.completion is not None and not isinstance(
            self.completion, PropagationCompletionReceipt
        ):
            raise ChangePropagationValidationError(
                "completion must be the canonical PropagationCompletionReceipt@1"
            )
        if self.completion is not None:
            if self.report.complete:
                if self.completion.disposition is not CompletionDisposition.COMPLETE:
                    raise ChangePropagationValidationError(
                        "complete report requires COMPLETE disposition receipt"
                    )
            elif self.completion.disposition is CompletionDisposition.COMPLETE:
                raise ChangePropagationValidationError(
                    "incomplete report cannot carry a COMPLETE disposition receipt"
                )

    @property
    def complete(self) -> bool:
        return self.completion is not None and self.report.complete

    def require_complete(self) -> PropagationCompletionReceipt:
        if self.completion is None or not self.report.complete:
            reasons = ", ".join(self.report.reason_codes) or "incomplete"
            raise ChangePropagationValidationError(
                "change propagation fixed-point validation rejected: " + reasons
            )
        return self.completion


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@dataclass
class ChangePropagationValidator:
    """Orchestrate fixed-point re-index / re-diff / re-resolve / re-prove gates.

    ``validate`` is pure over structured :class:`CandidatePropagationEvidence`
    and returns a report plus an optional canonical
    :class:`PropagationCompletionReceipt`.  Bound exhaustion is incomplete, not
    success.  Partial completion is never emitted as COMPLETE.
    """

    INTERFACE: Final[str] = CHANGE_PROPAGATION_VALIDATOR_INTERFACE

    default_required_tool_families: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_POLICY_REQUIRED_TOOLS
    )
    fixed_point_bound: int = DEFAULT_FIXED_POINT_BOUND

    def __post_init__(self) -> None:
        self.fixed_point_bound = _bounded_int(
            self.fixed_point_bound, "fixed_point_bound", minimum=1, maximum=MAX_ITERATIONS
        )

    def validate(
        self,
        plan: AtomicPropagationPlan,
        transaction: PropagationTransaction,
        *,
        evidence: CandidatePropagationEvidence,
        packet: ChangePropagationEditPacket | None = None,
        execution_report: TransactionExecutionReport | None = None,
        required_tool_families: Sequence[str] | None = None,
        fixed_point_bound: int | None = None,
    ) -> PropagationValidationOutcome:
        """Run the full multi-iteration fixed-point gate (fail-closed)."""

        stages: list[StageResult] = []
        reasons: set[str] = set()

        typed = (
            isinstance(plan, AtomicPropagationPlan)
            and isinstance(transaction, PropagationTransaction)
            and isinstance(evidence, CandidatePropagationEvidence)
        )
        if not typed:
            return self._malformed_outcome()

        bound = self.fixed_point_bound
        if fixed_point_bound is not None:
            bound = _bounded_int(
                fixed_point_bound, "fixed_point_bound", minimum=1, maximum=MAX_ITERATIONS
            )

        tools = _ids(
            tuple(required_tool_families or self.default_required_tool_families),
            "required_tool_families",
            required=True,
            maximum=len(POLICY_TOOL_FAMILIES) + 8,
        )

        # --- Binding ---
        if plan.disposition is not PlanDisposition.ADMITTED:
            reasons.add(PropagationValidationReason.PLAN_NOT_ADMITTED.value)
        if transaction.state is not TransactionState.COMMITTED:
            reasons.add(PropagationValidationReason.TRANSACTION_NOT_COMMITTED.value)
        if transaction.plan_id != plan.plan_id:
            reasons.add(PropagationValidationReason.PLAN_PACKET_MISMATCH.value)
        if transaction.roots != plan.roots:
            reasons.add(PropagationValidationReason.ROOT_DRIFT.value)
        if evidence.candidate_tree_id != plan.roots.candidate_tree_id:
            reasons.add(PropagationValidationReason.STALE_CANDIDATE_TREE.value)
        if packet is not None:
            if not isinstance(packet, ChangePropagationEditPacket):
                reasons.add(PropagationValidationReason.MALFORMED_INPUT.value)
            elif packet.plan_id != plan.plan_id or packet.roots != plan.roots:
                reasons.add(PropagationValidationReason.PLAN_PACKET_MISMATCH.value)
        if execution_report is not None:
            if not execution_report.committed:
                reasons.add(PropagationValidationReason.TRANSACTION_NOT_COMMITTED.value)
            if execution_report.transaction.transaction_id != transaction.transaction_id:
                reasons.add(PropagationValidationReason.PLAN_PACKET_MISMATCH.value)

        if reasons:
            report = PropagationValidationReport(
                plan_id=plan.plan_id,
                transaction_id=transaction.transaction_id,
                candidate_tree_id=plan.roots.candidate_tree_id,
                roots=plan.roots,
                stages=(
                    StageResult(
                        ValidationStage.COMPLETION,
                        StageDisposition.FAILED,
                        tuple(sorted(reasons)),
                    ),
                ),
                reason_codes=tuple(sorted(reasons)),
                iteration_count=0,
                complete=False,
            )
            return PropagationValidationOutcome(
                report=report,
                completion=self._incomplete_completion(
                    plan, transaction, reasons, residual_consumers=(), residual_frontier=(), residual_deltas=()
                ),
            )

        original_obligation_ids = tuple(
            item.obligation_id
            for item in plan.obligations
            if item.disposition
            in {
                ConsumerDisposition.MIGRATE,
                ConsumerDisposition.ADAPTER,
            }
        )
        if not original_obligation_ids:
            # All consumers non-migrate still need identity discharge of every obligation.
            original_obligation_ids = tuple(item.obligation_id for item in plan.obligations)

        residual_deltas: list[str] = []
        residual_consumers: list[str] = []
        residual_frontier: list[str] = []
        last_discharged: tuple[str, ...] = ()
        last_proof_refs: tuple[str, ...] = ()
        last_validation_refs: list[str] = []
        reached_fixed_point = False
        iteration_count = 0

        for iteration_evidence in evidence.iterations:
            iteration_count = iteration_evidence.iteration
            if iteration_count > bound:
                reasons.add(PropagationValidationReason.BOUND_EXHAUSTED.value)
                break

            # Each iteration re-evaluates residuals from its own stage evidence.
            residual_deltas = []
            residual_consumers = []
            residual_frontier = []

            # Bind every stage evidence to the candidate tree.
            for stage_ev in (
                iteration_evidence.index_rebuild,
                iteration_evidence.delta_reextract,
                iteration_evidence.resolution,
                iteration_evidence.closure_recompute,
                iteration_evidence.consumer_discharge,
                iteration_evidence.second_order,
                iteration_evidence.proof_reconstruction,
                iteration_evidence.policy_tools,
                iteration_evidence.impacted_tests,
                iteration_evidence.integrity,
            ):
                tree_id = getattr(stage_ev, "candidate_tree_id", "")
                if tree_id != plan.roots.candidate_tree_id:
                    reasons.add(PropagationValidationReason.STALE_CANDIDATE_TREE.value)

            # --- Index rebuild ---
            idx = iteration_evidence.index_rebuild
            index_reasons: list[str] = []
            write_paths = set(plan.permitted_write_paths)
            if not write_paths.issubset(idx.affected_paths) and not write_paths.issubset(
                idx.rebuilt_source_paths
            ):
                index_reasons.append(
                    PropagationValidationReason.INDEX_REBUILD_INCOMPLETE.value
                )
            if not idx.clean_rebuild_equivalent:
                index_reasons.append(
                    PropagationValidationReason.INDEX_REBUILD_INCOMPLETE.value
                )
            if not idx.rebuilt_vector_row_ids or not idx.rebuilt_graph_node_ids:
                index_reasons.append(
                    PropagationValidationReason.INDEX_REBUILD_INCOMPLETE.value
                )
            if evidence.expected_tombstone_ids and not set(
                evidence.expected_tombstone_ids
            ).issubset(idx.tombstone_ids):
                index_reasons.append(PropagationValidationReason.TOMBSTONE_MISSING.value)
            if evidence.expected_deleted_paths and not idx.tombstone_ids:
                index_reasons.append(PropagationValidationReason.TOMBSTONE_MISSING.value)
            stages.append(
                StageResult(
                    ValidationStage.INDEX_REBUILD,
                    StageDisposition.PASSED if not index_reasons else StageDisposition.FAILED,
                    tuple(sorted(set(index_reasons))),
                    idx.receipt_id,
                    iteration_count,
                )
            )
            reasons.update(index_reasons)

            # --- Delta re-extract ---
            delta = iteration_evidence.delta_reextract
            delta_reasons: list[str] = []
            if delta.original_delta_id != plan.delta_id:
                delta_reasons.append(
                    PropagationValidationReason.DELTA_REEXTRACT_FAILED.value
                )
            if not delta.matches_plan_delta and not delta.unplanned_breaking_delta_ids:
                # Re-extract may differ only when new unplanned breaking deltas are declared.
                delta_reasons.append(
                    PropagationValidationReason.DELTA_REEXTRACT_FAILED.value
                )
            if delta.unplanned_breaking_delta_ids:
                delta_reasons.append(
                    PropagationValidationReason.UNPLANNED_BREAKING_DELTA.value
                )
                residual_deltas = list(delta.unplanned_breaking_delta_ids)
            stages.append(
                StageResult(
                    ValidationStage.DELTA_REEXTRACT,
                    StageDisposition.PASSED if not delta_reasons else StageDisposition.FAILED,
                    tuple(sorted(set(delta_reasons))),
                    delta.extraction_receipt_id,
                    iteration_count,
                )
            )
            reasons.update(delta_reasons)

            # --- Resolution ---
            resolution = iteration_evidence.resolution
            res_reasons: list[str] = []
            if not resolution.complete or resolution.unresolved_ids:
                res_reasons.append(PropagationValidationReason.RESOLUTION_INCOMPLETE.value)
            stages.append(
                StageResult(
                    ValidationStage.RESOLUTION,
                    StageDisposition.PASSED if not res_reasons else StageDisposition.FAILED,
                    tuple(sorted(set(res_reasons))),
                    resolution.resolution_receipt_id,
                    iteration_count,
                )
            )
            reasons.update(res_reasons)

            # --- Closure recompute ---
            closure = iteration_evidence.closure_recompute
            closure_reasons: list[str] = []
            if closure.original_closure_id != plan.impact_closure_id:
                closure_reasons.append(PropagationValidationReason.CLOSURE_INCOMPLETE.value)
            if not closure.complete:
                closure_reasons.append(PropagationValidationReason.CLOSURE_INCOMPLETE.value)
            if closure.uncovered_frontier_ids:
                closure_reasons.append(PropagationValidationReason.UNCOVERED_FRONTIER.value)
                residual_frontier = list(closure.uncovered_frontier_ids)
            stages.append(
                StageResult(
                    ValidationStage.CLOSURE_RECOMPUTE,
                    StageDisposition.PASSED if not closure_reasons else StageDisposition.FAILED,
                    tuple(sorted(set(closure_reasons))),
                    closure.receipt_id,
                    iteration_count,
                )
            )
            reasons.update(closure_reasons)

            # --- Consumer discharge (exactly once) ---
            discharge = iteration_evidence.consumer_discharge
            disc_reasons: list[str] = []
            if set(discharge.original_obligation_ids) != set(original_obligation_ids):
                # Evidence must acknowledge the exact original obligation set.
                if not set(original_obligation_ids).issubset(
                    discharge.original_obligation_ids
                ):
                    disc_reasons.append(
                        PropagationValidationReason.CONSUMER_NOT_DISCHARGED.value
                    )
            if discharge.double_discharged_ids:
                disc_reasons.append(
                    PropagationValidationReason.CONSUMER_DOUBLE_DISCHARGE.value
                )
            if discharge.unresolved_mandatory_ids:
                disc_reasons.append(PropagationValidationReason.UNRESOLVED_MANDATORY.value)
                residual_consumers = list(discharge.unresolved_mandatory_ids)
            if discharge.omitted_dependent_ids:
                disc_reasons.append(PropagationValidationReason.OMITTED_DEPENDENT.value)
            if not set(original_obligation_ids).issubset(
                discharge.discharged_obligation_ids
            ):
                disc_reasons.append(
                    PropagationValidationReason.CONSUMER_NOT_DISCHARGED.value
                )
            stages.append(
                StageResult(
                    ValidationStage.CONSUMER_DISCHARGE,
                    StageDisposition.PASSED if not disc_reasons else StageDisposition.FAILED,
                    tuple(sorted(set(disc_reasons))),
                    discharge.receipt_id,
                    iteration_count,
                )
            )
            reasons.update(disc_reasons)
            last_discharged = discharge.discharged_obligation_ids

            # --- Second-order impacts ---
            second = iteration_evidence.second_order
            second_reasons: list[str] = []
            if second.requires_another_iteration:
                second_reasons.append(
                    PropagationValidationReason.SECOND_ORDER_RESIDUAL.value
                )
                residual_deltas = list(
                    dict.fromkeys(residual_deltas + list(second.new_delta_ids))
                )
                residual_consumers = list(
                    dict.fromkeys(residual_consumers + list(second.new_consumer_ids))
                )
                residual_frontier = list(
                    dict.fromkeys(residual_frontier + list(second.residual_frontier_ids))
                )
            stages.append(
                StageResult(
                    ValidationStage.SECOND_ORDER,
                    StageDisposition.CONTINUE
                    if second.requires_another_iteration
                    else StageDisposition.PASSED,
                    tuple(sorted(set(second_reasons))),
                    second.receipt_id,
                    iteration_count,
                )
            )
            # Second-order residual is not a hard failure if another iteration follows
            # within the bound; track for fixed-point decision.
            if second.requires_another_iteration:
                # Do not add to terminal reasons yet — only if no further iterations.
                pass

            # --- Proof reconstruction ---
            proofs = iteration_evidence.proof_reconstruction
            proof_reasons: list[str] = []
            if not set(plan.proof_refs).issubset(proofs.original_proof_refs) and not set(
                plan.proof_refs
            ).issubset(proofs.reconstructed_proof_refs):
                proof_reasons.append(
                    PropagationValidationReason.PROOF_RECONSTRUCTION_FAILED.value
                )
            if not proofs.all_mandatory_reconstructed or proofs.failed_proof_refs:
                proof_reasons.append(
                    PropagationValidationReason.PROOF_RECONSTRUCTION_FAILED.value
                )
            stages.append(
                StageResult(
                    ValidationStage.PROOF_RECONSTRUCTION,
                    StageDisposition.PASSED if not proof_reasons else StageDisposition.FAILED,
                    tuple(sorted(set(proof_reasons))),
                    proofs.receipt_id,
                    iteration_count,
                )
            )
            reasons.update(proof_reasons)
            last_proof_refs = proofs.reconstructed_proof_refs

            # --- Policy tools ---
            policy = iteration_evidence.policy_tools
            tool_reasons: list[str] = []
            if policy.policy_id not in {plan.roots.policy_id}:
                tool_reasons.append(PropagationValidationReason.ROOT_DRIFT.value)
            if set(tools) - set(policy.required_families):
                tool_reasons.append(PropagationValidationReason.SKIPPED_REQUIRED_TOOL.value)
            by_family = {item.family: item for item in policy.results}
            for family in tools:
                result = by_family.get(family)
                if result is None:
                    tool_reasons.append(
                        PropagationValidationReason.SKIPPED_REQUIRED_TOOL.value
                    )
                    continue
                if result.required and (result.skipped or not result.executed):
                    tool_reasons.append(
                        PropagationValidationReason.SKIPPED_REQUIRED_TOOL.value
                    )
                elif result.required and not result.passed:
                    tool_reasons.append(PropagationValidationReason.TOOL_FAILED.value)
            stages.append(
                StageResult(
                    ValidationStage.POLICY_TOOLS,
                    StageDisposition.PASSED if not tool_reasons else StageDisposition.FAILED,
                    tuple(sorted(set(tool_reasons))),
                    content_identity(policy.to_dict()),
                    iteration_count,
                )
            )
            reasons.update(tool_reasons)
            last_validation_refs.append(content_identity(policy.to_dict()))

            # --- Impacted tests ---
            tests = iteration_evidence.impacted_tests
            test_reasons: list[str] = []
            if tests.failed_test_ids:
                if set(tests.failed_test_ids) & set(tests.focused_test_ids):
                    test_reasons.append(
                        PropagationValidationReason.FOCUSED_TEST_FAILED.value
                    )
                if set(tests.failed_test_ids) & set(tests.impacted_test_ids):
                    test_reasons.append(
                        PropagationValidationReason.IMPACTED_TEST_FAILED.value
                    )
                if not test_reasons:
                    test_reasons.append(
                        PropagationValidationReason.IMPACTED_TEST_FAILED.value
                    )
            if tests.omitted_dependant_ids or not tests.dependency_complete:
                test_reasons.append(
                    PropagationValidationReason.IMPACTED_TEST_OMITTED.value
                )
            required_tests = (
                set(tests.focused_test_ids)
                | set(tests.impacted_test_ids)
                | set(tests.required_dependant_ids)
            )
            if required_tests - set(tests.executed_test_ids):
                test_reasons.append(
                    PropagationValidationReason.IMPACTED_TEST_OMITTED.value
                )
            if required_tests - set(tests.passed_test_ids):
                if PropagationValidationReason.FOCUSED_TEST_FAILED.value not in test_reasons:
                    test_reasons.append(
                        PropagationValidationReason.IMPACTED_TEST_FAILED.value
                    )
            stages.append(
                StageResult(
                    ValidationStage.IMPACTED_TESTS,
                    StageDisposition.PASSED if not test_reasons else StageDisposition.FAILED,
                    tuple(sorted(set(test_reasons))),
                    content_identity(tests.to_dict()),
                    iteration_count,
                )
            )
            reasons.update(test_reasons)
            last_validation_refs.append(content_identity(tests.to_dict()))

            # --- Integrity (anti-weakening) ---
            integrity = iteration_evidence.integrity
            integ_reasons: list[str] = []
            if integrity.contracts_deleted:
                integ_reasons.append(PropagationValidationReason.CONTRACT_DELETED.value)
            if integrity.contracts_weakened:
                integ_reasons.append(PropagationValidationReason.CONTRACT_WEAKENED.value)
            if integrity.tests_deleted:
                integ_reasons.append(PropagationValidationReason.TEST_DELETED.value)
            if integrity.tests_weakened:
                integ_reasons.append(PropagationValidationReason.TEST_WEAKENED.value)
            if integrity.checkers_deleted:
                integ_reasons.append(PropagationValidationReason.CHECKER_DELETED.value)
            if integrity.checkers_weakened:
                integ_reasons.append(PropagationValidationReason.CHECKER_WEAKENED.value)
            if not integrity.original_finding_closed and integrity.findings_suppressed:
                integ_reasons.append(PropagationValidationReason.CONTRACT_WEAKENED.value)
            if not integrity.clean:
                if not integ_reasons:
                    integ_reasons.append(
                        PropagationValidationReason.CONTRACT_WEAKENED.value
                    )
            stages.append(
                StageResult(
                    ValidationStage.INTEGRITY,
                    StageDisposition.PASSED if not integ_reasons else StageDisposition.FAILED,
                    tuple(sorted(set(integ_reasons))),
                    content_identity(integrity.to_dict()),
                    iteration_count,
                )
            )
            reasons.update(integ_reasons)

            # Terminal hard failures on this iteration stop the loop.
            hard = reasons - {
                PropagationValidationReason.SECOND_ORDER_RESIDUAL.value,
            }
            # second_order is tracked separately
            if hard:
                break

            if second.requires_another_iteration:
                # Continue to next provided iteration evidence.
                residual_present = True
            else:
                residual_present = bool(
                    residual_deltas or residual_consumers or residual_frontier
                )
                if not residual_present and not disc_reasons and not delta_reasons:
                    reached_fixed_point = True
                    stages.append(
                        StageResult(
                            ValidationStage.FIXED_POINT,
                            StageDisposition.PASSED,
                            (),
                            f"fixed-point:iter-{iteration_count}",
                            iteration_count,
                        )
                    )
                    break
                # Residuals without second-order flag: fail.
                if residual_present:
                    reasons.add(PropagationValidationReason.FIXED_POINT_NOT_REACHED.value)
                    break
        else:
            # Exhausted provided iterations without fixed point.
            if not reached_fixed_point:
                if iteration_count >= bound:
                    reasons.add(PropagationValidationReason.BOUND_EXHAUSTED.value)
                else:
                    # Last iteration still had second-order residual and no more evidence.
                    last = evidence.iterations[-1]
                    if last.second_order.requires_another_iteration:
                        if iteration_count >= bound:
                            reasons.add(PropagationValidationReason.BOUND_EXHAUSTED.value)
                        else:
                            reasons.add(
                                PropagationValidationReason.FIXED_POINT_NOT_REACHED.value
                            )
                            reasons.add(
                                PropagationValidationReason.SECOND_ORDER_RESIDUAL.value
                            )
                    elif not reasons:
                        reasons.add(
                            PropagationValidationReason.FIXED_POINT_NOT_REACHED.value
                        )

        # Bound check when more iterations would be needed.
        if not reached_fixed_point and iteration_count >= bound:
            reasons.add(PropagationValidationReason.BOUND_EXHAUSTED.value)

        if reasons:
            # Never emit COMPLETE with residuals or failures.
            report = PropagationValidationReport(
                plan_id=plan.plan_id,
                transaction_id=transaction.transaction_id,
                candidate_tree_id=plan.roots.candidate_tree_id,
                roots=plan.roots,
                stages=tuple(stages)
                or (
                    StageResult(
                        ValidationStage.COMPLETION,
                        StageDisposition.FAILED,
                        tuple(sorted(reasons)),
                    ),
                ),
                reason_codes=tuple(sorted(reasons)),
                iteration_count=iteration_count,
                complete=False,
            )
            disposition = (
                CompletionDisposition.INCOMPLETE
                if PropagationValidationReason.BOUND_EXHAUSTED.value in reasons
                or PropagationValidationReason.FIXED_POINT_NOT_REACHED.value in reasons
                or PropagationValidationReason.SECOND_ORDER_RESIDUAL.value in reasons
                else CompletionDisposition.FAILED
            )
            completion = PropagationCompletionReceipt(
                roots=plan.roots,
                completion_id=content_identity(
                    {
                        "schema": "completion-failed",
                        "plan_id": plan.plan_id,
                        "transaction_id": transaction.transaction_id,
                        "reasons": sorted(reasons),
                    }
                ),
                plan_id=plan.plan_id,
                transaction_id=transaction.transaction_id,
                disposition=disposition,
                fixed_point_receipt=None,
                discharged_obligation_ids=last_discharged,
                unresolved_mandatory_ids=tuple(sorted(set(residual_consumers))),
                omitted_dependent_ids=(),
                uncovered_frontier_ids=tuple(sorted(set(residual_frontier))),
                unplanned_breaking_delta_ids=tuple(sorted(set(residual_deltas))),
                proof_refs=last_proof_refs,
                validation_refs=tuple(sorted(set(last_validation_refs))),
                invalidation_refs=plan.invalidation_refs,
            )
            # Ensure non-complete has residual diagnostics when no residuals listed.
            if (
                not completion.unresolved_mandatory_ids
                and not completion.omitted_dependent_ids
                and not completion.uncovered_frontier_ids
                and not completion.unplanned_breaking_delta_ids
                and disposition
                not in {CompletionDisposition.ABSTAINED, CompletionDisposition.FAILED}
            ):
                completion = PropagationCompletionReceipt(
                    roots=plan.roots,
                    completion_id=completion.completion_id,
                    plan_id=plan.plan_id,
                    transaction_id=transaction.transaction_id,
                    disposition=CompletionDisposition.FAILED,
                    fixed_point_receipt=None,
                    discharged_obligation_ids=last_discharged,
                    unresolved_mandatory_ids=(),
                    omitted_dependent_ids=(),
                    uncovered_frontier_ids=(),
                    unplanned_breaking_delta_ids=(),
                    proof_refs=last_proof_refs,
                    validation_refs=tuple(sorted(set(last_validation_refs))),
                    invalidation_refs=plan.invalidation_refs,
                )
            return PropagationValidationOutcome(report=report, completion=completion)

        # Success path: residual-free fixed point.
        if not reached_fixed_point or not last_discharged:
            reasons.add(PropagationValidationReason.FIXED_POINT_NOT_REACHED.value)
            report = PropagationValidationReport(
                plan_id=plan.plan_id,
                transaction_id=transaction.transaction_id,
                candidate_tree_id=plan.roots.candidate_tree_id,
                roots=plan.roots,
                stages=tuple(stages),
                reason_codes=tuple(sorted(reasons)),
                iteration_count=iteration_count,
                complete=False,
            )
            return PropagationValidationOutcome(
                report=report,
                completion=self._incomplete_completion(
                    plan,
                    transaction,
                    reasons,
                    residual_consumers=residual_consumers,
                    residual_frontier=residual_frontier,
                    residual_deltas=residual_deltas,
                ),
            )

        fixed = FixedPointReceipt(
            roots=plan.roots,
            receipt_id=content_identity(
                {
                    "schema": "fixed-point",
                    "plan_id": plan.plan_id,
                    "iteration_count": max(iteration_count, 1),
                    "transaction_id": transaction.transaction_id,
                }
            ),
            plan_id=plan.plan_id,
            iteration_count=max(iteration_count, 1),
            residual_delta_ids=(),
            residual_consumer_ids=(),
            residual_frontier_ids=(),
            proof_refs=last_proof_refs or plan.proof_refs,
            validation_refs=tuple(sorted(set(last_validation_refs)))
            or ("validation:fixed-point",),
        )
        stages.append(
            StageResult(
                ValidationStage.COMPLETION,
                StageDisposition.PASSED,
                (),
                fixed.receipt_id,
                iteration_count,
            )
        )
        report = PropagationValidationReport(
            plan_id=plan.plan_id,
            transaction_id=transaction.transaction_id,
            candidate_tree_id=plan.roots.candidate_tree_id,
            roots=plan.roots,
            stages=tuple(stages),
            reason_codes=(),
            iteration_count=iteration_count,
            complete=True,
            fixed_point_receipt=fixed,
        )
        completion = PropagationCompletionReceipt(
            roots=plan.roots,
            completion_id=content_identity(
                {
                    "schema": "completion-ok",
                    "plan_id": plan.plan_id,
                    "transaction_id": transaction.transaction_id,
                    "fixed_point": fixed.receipt_id,
                    "discharged": list(last_discharged),
                }
            ),
            plan_id=plan.plan_id,
            transaction_id=transaction.transaction_id,
            disposition=CompletionDisposition.COMPLETE,
            fixed_point_receipt=fixed,
            discharged_obligation_ids=last_discharged,
            unresolved_mandatory_ids=(),
            omitted_dependent_ids=(),
            uncovered_frontier_ids=(),
            unplanned_breaking_delta_ids=(),
            proof_refs=last_proof_refs or plan.proof_refs,
            validation_refs=tuple(sorted(set(last_validation_refs)))
            or ("validation:fixed-point",),
            invalidation_refs=plan.invalidation_refs,
        )
        return PropagationValidationOutcome(report=report, completion=completion)

    def require_complete(self, *args: Any, **kwargs: Any) -> PropagationCompletionReceipt:
        return self.validate(*args, **kwargs).require_complete()

    def is_complete(self, *args: Any, **kwargs: Any) -> bool:
        return self.validate(*args, **kwargs).complete

    def _malformed_outcome(self) -> PropagationValidationOutcome:
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
        report = PropagationValidationReport(
            plan_id="plan:invalid",
            transaction_id="txn:invalid",
            candidate_tree_id="tree:invalid-cand",
            roots=roots,
            stages=(
                StageResult(
                    ValidationStage.COMPLETION,
                    StageDisposition.FAILED,
                    (PropagationValidationReason.MALFORMED_INPUT.value,),
                ),
            ),
            reason_codes=(PropagationValidationReason.MALFORMED_INPUT.value,),
            iteration_count=0,
            complete=False,
        )
        return PropagationValidationOutcome(report=report, completion=None)

    def _incomplete_completion(
        self,
        plan: AtomicPropagationPlan,
        transaction: PropagationTransaction,
        reasons: set[str] | Sequence[str],
        *,
        residual_consumers: Sequence[str],
        residual_frontier: Sequence[str],
        residual_deltas: Sequence[str],
    ) -> PropagationCompletionReceipt:
        reason_list = sorted(set(reasons))
        return PropagationCompletionReceipt(
            roots=plan.roots,
            completion_id=content_identity(
                {
                    "schema": "completion-incomplete",
                    "plan_id": plan.plan_id,
                    "transaction_id": transaction.transaction_id,
                    "reasons": reason_list,
                }
            ),
            plan_id=plan.plan_id,
            transaction_id=transaction.transaction_id,
            disposition=CompletionDisposition.FAILED
            if reason_list
            else CompletionDisposition.INCOMPLETE,
            fixed_point_receipt=None,
            discharged_obligation_ids=(),
            unresolved_mandatory_ids=tuple(sorted(set(residual_consumers))),
            omitted_dependent_ids=(),
            uncovered_frontier_ids=tuple(sorted(set(residual_frontier))),
            unplanned_breaking_delta_ids=tuple(sorted(set(residual_deltas))),
            proof_refs=(),
            validation_refs=(),
            invalidation_refs=plan.invalidation_refs,
        )


def validate_change_propagation(
    plan: AtomicPropagationPlan,
    transaction: PropagationTransaction,
    *,
    evidence: CandidatePropagationEvidence,
    packet: ChangePropagationEditPacket | None = None,
    fixed_point_bound: int | None = None,
) -> PropagationValidationOutcome:
    """Module entry point matching :meth:`ChangePropagationValidator.validate`."""

    return ChangePropagationValidator().validate(
        plan,
        transaction,
        evidence=evidence,
        packet=packet,
        fixed_point_bound=fixed_point_bound,
    )


def validate_change_propagation_with_logic_fixed_point(
    plan: AtomicPropagationPlan,
    transaction: PropagationTransaction,
    *,
    evidence: CandidatePropagationEvidence,
    logic_evidence: Any = None,
    packet: ChangePropagationEditPacket | None = None,
    execution_report: TransactionExecutionReport | None = None,
    fixed_point_bound: int | None = None,
    restore_adapter: Any = None,
    checkpoint: Any = None,
    require_logic_evidence: bool = False,
) -> Any:
    """Joint program+logic fixed-point via :class:`LogicRepairFixedPointValidator`.

    Extends the program-only :func:`validate_change_propagation` path without
    weakening any legacy fixed-point condition.  When ``logic_evidence`` is
    supplied, per-iteration logic rebuild/replan/reprove evidence is validated
    and attached to the existing completion receipt.
    """

    from .logic_repair_fixed_point import LogicRepairFixedPointValidator

    return LogicRepairFixedPointValidator(
        require_logic_evidence=require_logic_evidence
    ).validate(
        plan,
        transaction,
        program_evidence=evidence,
        logic_evidence=logic_evidence,
        packet=packet,
        execution_report=execution_report,
        fixed_point_bound=fixed_point_bound,
        restore_adapter=restore_adapter,
        checkpoint=checkpoint,
    )


__all__ = [
    "CHANGE_PROPAGATION_VALIDATOR_INTERFACE",
    "DEFAULT_FIXED_POINT_BOUND",
    "DEFAULT_POLICY_REQUIRED_TOOLS",
    "MAX_ITERATIONS",
    "POLICY_TOOL_FAMILIES",
    "PRODUCER_ID",
    "PROPAGATION_VALIDATION_REPORT_SCHEMA",
    "CandidatePropagationEvidence",
    "ChangePropagationValidationError",
    "ChangePropagationValidator",
    "ClosureRecomputeEvidence",
    "ConsumerDischargeEvidence",
    "DeltaReextractEvidence",
    "FixedPointIterationEvidence",
    "ImpactedTestEvidence",
    "IntegrityEvidence",
    "PolicyToolEvidence",
    "ProofReconstructionEvidence",
    "PropagationIndexRebuildEvidence",
    "PropagationValidationOutcome",
    "PropagationValidationReason",
    "PropagationValidationReport",
    "ResolutionEvidence",
    "SecondOrderImpactEvidence",
    "StageDisposition",
    "StageResult",
    "ToolGateResult",
    "ValidationStage",
    "build_passing_tool_evidence",
    "validate_change_propagation",
    "validate_change_propagation_with_logic_fixed_point",
    # Canonical re-exports
    "AtomicPropagationPlan",
    "CompletionDisposition",
    "FixedPointReceipt",
    "PropagationCompletionReceipt",
    "PropagationTransaction",
]
