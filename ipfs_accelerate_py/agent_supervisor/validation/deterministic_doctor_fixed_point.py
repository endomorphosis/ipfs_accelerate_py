"""Post-apply deterministic-doctor fixed-point validation (LPR-038).

Interface: ``DeterministicDoctorFixedPointValidator@1``

After a provisional doctor transaction commit the validator:

1. reparses changed files and recomputes types, imports, graphs, contracts,
   values, effects/resources/memory facets, and impact closure;
2. rebuilds AST / call / dependency / schema / value graphs and KG / vector
   tombstones;
3. invalidates dependent CAS and proof-cache entries under current roots;
4. redelta / reclose / replan (Tactician) / reprove (Hammer) / rediagnose
   until no original or second-order mandatory finding remains; and
5. emits a :class:`DoctorFixedPointReceipt` only at residual-free success.

Bound exhaustion, oscillation, root drift, incomplete evidence, or failure
after a provisional commit triggers compensating rollback to the checkpoint.
Rollback failure quarantines the candidate tree.  Neither residual nor
quarantined state may claim completion or call a model.

This module reuses the program+logic fixed-point plane
(:class:`LogicRepairFixedPointValidator`) when propagation evidence is
available, and always attaches doctor-specific rebuild / cache / residual
receipts that never replace existing completion authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ..analysis.deterministic_doctor_contracts import (
    MAX_REFERENCE_COUNT,
    MAX_TEXT_BYTES,
    DoctorAuthorityRoots,
    DoctorMode,
    DoctorPlanDisposition,
    DoctorRepairDisposition,
    DoctorResourceBounds,
    DeterministicDoctorPlan,
)
from ..planning.deterministic_doctor_transaction import (
    DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE,
    DeterministicDoctorTransactionError,
    DoctorCandidateTreeReceipt,
    DoctorRollbackReceipt,
    DoctorTransactionCheckpoint,
    DoctorTransactionDisposition,
    DoctorTransactionReason,
    DoctorTransactionReport,
)
from ..proof.formal_verification_contracts import content_identity


# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

DETERMINISTIC_DOCTOR_FIXED_POINT_INTERFACE: Final[str] = (
    "DeterministicDoctorFixedPointValidator@1"
)
DOCTOR_FIXED_POINT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/fixed-point-receipt@1"
)
DOCTOR_FIXED_POINT_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/fixed-point-report@1"
)
DOCTOR_ITERATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/fixed-point-iteration@1"
)
DOCTOR_REBUILD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/rebuild@1"
)
DOCTOR_CACHE_INVALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/cache-invalidation@1"
)
DOCTOR_REDELTA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/redelta@1"
)
DOCTOR_RECLOSE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/reclose@1"
)
DOCTOR_REPLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/replan@1"
)
DOCTOR_REPROVE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/reprove@1"
)
DOCTOR_STATIC_CHECKS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/static-checks@1"
)
DOCTOR_COMPENSATING_ROLLBACK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/compensating-rollback@1"
)

PRODUCER_ID: Final[str] = "deterministic-doctor-fixed-point@1"
CONTRACT_VERSION: Final[int] = 1

MAX_REASON_CODES: Final[int] = 64
MAX_IDS: Final[int] = 1_024
MAX_ITERATIONS: Final[int] = 32
DEFAULT_FIXED_POINT_BOUND: Final[int] = 8
MAX_OSCILLATION_WINDOW: Final[int] = 4

_FORBIDDEN_PROVIDER_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "llm_router",
        "model_provider",
        "openai",
        "anthropic",
        "provider_router",
    }
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class DoctorFixedPointReason(str, Enum):
    """Stable machine-readable fixed-point failure codes."""

    MALFORMED_INPUT = "malformed_input"
    TRANSACTION_NOT_PROVISIONAL = "transaction_not_provisional"
    PLAN_NOT_ADMITTED = "plan_not_admitted"
    ROOT_DRIFT = "root_drift"
    STALE_CANDIDATE_TREE = "stale_candidate_tree"
    REBUILD_INCOMPLETE = "rebuild_incomplete"
    TOMBSTONE_MISSING = "tombstone_missing"
    CACHE_INVALIDATION_INCOMPLETE = "cache_invalidation_incomplete"
    REPARSE_FAILED = "reparse_failed"
    TYPE_CHECK_FAILED = "type_check_failed"
    STATIC_CHECK_FAILED = "static_check_failed"
    DIFFERENTIAL_CHECK_FAILED = "differential_check_failed"
    PROOF_CHECK_FAILED = "proof_check_failed"
    MEMORY_EFFECT_FAILED = "memory_effect_failed"
    RESOURCE_CHECK_FAILED = "resource_check_failed"
    DELTA_RECOMPUTE_FAILED = "delta_recompute_failed"
    UNPLANNED_BREAKING_DELTA = "unplanned_breaking_delta"
    CLOSURE_RECOMPUTE_FAILED = "closure_recompute_failed"
    UNCOVERED_FRONTIER = "uncovered_frontier"
    UNRESOLVED_MANDATORY_FINDING = "unresolved_mandatory_finding"
    SECOND_ORDER_FINDING_OPEN = "second_order_finding_open"
    REPLAN_STALE = "replan_stale"
    TACTICIAN_PLAN_STALE = "tactician_plan_stale"
    REPROVE_FAILED = "reprove_failed"
    HAMMER_RECEIPT_MISSING = "hammer_receipt_missing"
    PREDICTION_STALE = "prediction_stale"
    BOUND_EXHAUSTED = "fixed_point_bound_exhausted"
    OSCILLATION_DETECTED = "oscillation_detected"
    DRIFT_DETECTED = "drift_detected"
    FIXED_POINT_NOT_REACHED = "fixed_point_not_reached"
    PARTIAL_SCC_FORBIDDEN = "partial_scc_completion_forbidden"
    PARTIAL_PACKET_FORBIDDEN = "partial_packet_completion_forbidden"
    ROLLBACK_REQUIRED = "compensating_rollback_required"
    ROLLBACK_FAILED = "compensating_rollback_failed"
    QUARANTINE_REQUIRED = "quarantine_required"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    MODEL_INVOCATION_FORBIDDEN = "model_invocation_forbidden"
    CLAIMS_COMPLETION_FORBIDDEN = "claims_completion_forbidden"
    IDENTITY_REPLAY_MISMATCH = "identity_replay_mismatch"


class DoctorFixedPointStage(str, Enum):
    """Ordered stages of one doctor fixed-point iteration."""

    ADMISSION = "admission"
    REPARSE = "reparse"
    REBUILD = "rebuild"
    CACHE_INVALIDATION = "cache_invalidation"
    STATIC_DIFFERENTIAL = "static_differential"
    MEMORY_EFFECT_RESOURCE = "memory_effect_resource"
    REDELTA = "redelta"
    RECLOSE = "reclose"
    REPLAN = "replan"
    REPROVE = "reprove"
    RESIDUAL = "residual"
    FIXED_POINT = "fixed_point"
    FINALIZE = "finalize"
    COMPENSATING_ROLLBACK = "compensating_rollback"
    QUARANTINE = "quarantine"


class DoctorStageDisposition(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    CONTINUE = "continue"
    ROLLED_BACK = "rolled_back"
    QUARANTINED = "quarantined"
    FINALIZED = "finalized"


class DoctorFixedPointDisposition(str, Enum):
    """Terminal outcomes of fixed-point validation."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    ROLLED_BACK = "rolled_back"
    QUARANTINED = "quarantined"
    ABSTAINED = "abstained"

    @property
    def claims_completion(self) -> bool:
        return self is DoctorFixedPointDisposition.COMPLETE

    @property
    def may_call_model(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DeterministicDoctorFixedPointError(ValueError):
    """Doctor fixed-point validation failed closed."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise DeterministicDoctorFixedPointError(f"{name} must be a compact identifier")
    text = value.strip()
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise DeterministicDoctorFixedPointError(f"{name} exceeds text bound")
    return text


def _optional_identifier(value: Any, name: str) -> str:
    if value is None or value == "":
        return ""
    return _identifier(value, name)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DeterministicDoctorFixedPointError(f"{name} must be a boolean")
    return value


def _bounded_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_ITERATIONS,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeterministicDoctorFixedPointError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise DeterministicDoctorFixedPointError(
            f"{name} out of bounds [{minimum}, {maximum}]"
        )
    return value


def _ids(
    values: Sequence[str],
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_IDS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DeterministicDoctorFixedPointError(f"{name} must be an identifier sequence")
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
            raise DeterministicDoctorFixedPointError(
                f"{name} must contain compact identifiers"
            )
    if required and not result:
        raise DeterministicDoctorFixedPointError(f"{name} must not be empty")
    if len(result) > maximum:
        raise DeterministicDoctorFixedPointError(f"{name} exceeds item bound")
    return result


def _roots(value: Any) -> DoctorAuthorityRoots:
    if isinstance(value, DoctorAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return DoctorAuthorityRoots.from_dict(value)
    raise DeterministicDoctorFixedPointError("roots must be DoctorAuthorityRoots")


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    if isinstance(value, enum):
        return value
    if isinstance(value, str):
        try:
            return enum(value)
        except ValueError as exc:
            raise DeterministicDoctorFixedPointError(
                f"{name} must be a valid {enum.__name__}"
            ) from exc
    raise DeterministicDoctorFixedPointError(f"{name} must be a valid {enum.__name__}")


# ---------------------------------------------------------------------------
# Stage evidence records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorRebuildEvidence:
    """Proof that AST/graphs/indexes/tombstones were rebuilt post-edit."""

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
    reparsed_paths: tuple[str, ...]
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
            self, "reparsed_paths", _ids(self.reparsed_paths, "reparsed_paths")
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
            "schema": DOCTOR_REBUILD_SCHEMA,
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
            "reparsed_paths": list(self.reparsed_paths),
            "clean_rebuild_equivalent": self.clean_rebuild_equivalent,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class DoctorCacheInvalidationEvidence:
    """CAS / proof-cache / index invalidation under current roots."""

    candidate_tree_id: str
    invalidated_cache_ids: tuple[str, ...]
    invalidated_cas_ids: tuple[str, ...]
    tombstone_ids: tuple[str, ...]
    remaining_stale_ids: tuple[str, ...]
    complete: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        for name in (
            "invalidated_cache_ids",
            "invalidated_cas_ids",
            "tombstone_ids",
            "remaining_stale_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        if self.complete and self.remaining_stale_ids:
            raise DeterministicDoctorFixedPointError(
                "complete cache invalidation forbids remaining stale ids"
            )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DOCTOR_CACHE_INVALIDATION_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "invalidated_cache_ids": list(self.invalidated_cache_ids),
            "invalidated_cas_ids": list(self.invalidated_cas_ids),
            "tombstone_ids": list(self.tombstone_ids),
            "remaining_stale_ids": list(self.remaining_stale_ids),
            "complete": self.complete,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class DoctorStaticCheckEvidence:
    """Reparse / type / static / differential / proof / memory / resource checks."""

    candidate_tree_id: str
    reparsed_paths: tuple[str, ...]
    type_check_receipt_ids: tuple[str, ...]
    static_check_receipt_ids: tuple[str, ...]
    differential_check_receipt_ids: tuple[str, ...]
    proof_check_receipt_ids: tuple[str, ...]
    memory_effect_receipt_ids: tuple[str, ...]
    resource_check_receipt_ids: tuple[str, ...]
    failed_check_ids: tuple[str, ...]
    all_passed: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        for name in (
            "reparsed_paths",
            "type_check_receipt_ids",
            "static_check_receipt_ids",
            "differential_check_receipt_ids",
            "proof_check_receipt_ids",
            "memory_effect_receipt_ids",
            "resource_check_receipt_ids",
            "failed_check_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(self, "all_passed", _bool(self.all_passed, "all_passed"))
        if self.all_passed and self.failed_check_ids:
            raise DeterministicDoctorFixedPointError(
                "all_passed forbids failed_check_ids"
            )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DOCTOR_STATIC_CHECKS_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "reparsed_paths": list(self.reparsed_paths),
            "type_check_receipt_ids": list(self.type_check_receipt_ids),
            "static_check_receipt_ids": list(self.static_check_receipt_ids),
            "differential_check_receipt_ids": list(
                self.differential_check_receipt_ids
            ),
            "proof_check_receipt_ids": list(self.proof_check_receipt_ids),
            "memory_effect_receipt_ids": list(self.memory_effect_receipt_ids),
            "resource_check_receipt_ids": list(self.resource_check_receipt_ids),
            "failed_check_ids": list(self.failed_check_ids),
            "all_passed": self.all_passed,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class DoctorRedeltaEvidence:
    """Recomputed program contract deltas after candidate apply."""

    candidate_tree_id: str
    original_delta_ids: tuple[str, ...]
    recomputed_delta_ids: tuple[str, ...]
    breaking_delta_ids: tuple[str, ...]
    unplanned_breaking_delta_ids: tuple[str, ...]
    matches_plan_delta: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        for name in (
            "original_delta_ids",
            "recomputed_delta_ids",
            "breaking_delta_ids",
            "unplanned_breaking_delta_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self, "matches_plan_delta", _bool(self.matches_plan_delta, "matches_plan_delta")
        )
        if self.matches_plan_delta and self.unplanned_breaking_delta_ids:
            raise DeterministicDoctorFixedPointError(
                "matches_plan_delta forbids unplanned breaking deltas"
            )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DOCTOR_REDELTA_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "original_delta_ids": list(self.original_delta_ids),
            "recomputed_delta_ids": list(self.recomputed_delta_ids),
            "breaking_delta_ids": list(self.breaking_delta_ids),
            "unplanned_breaking_delta_ids": list(self.unplanned_breaking_delta_ids),
            "matches_plan_delta": self.matches_plan_delta,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class DoctorRecloseEvidence:
    """Recomputed impact closure and residual finding coverage."""

    candidate_tree_id: str
    original_finding_ids: tuple[str, ...]
    discharged_original_ids: tuple[str, ...]
    second_order_finding_ids: tuple[str, ...]
    discharged_second_order_ids: tuple[str, ...]
    unresolved_mandatory_ids: tuple[str, ...]
    open_required_frontier_ids: tuple[str, ...]
    complete: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        for name in (
            "original_finding_ids",
            "discharged_original_ids",
            "second_order_finding_ids",
            "discharged_second_order_ids",
            "unresolved_mandatory_ids",
            "open_required_frontier_ids",
        ):
            required = name == "original_finding_ids"
            object.__setattr__(
                self, name, _ids(getattr(self, name), name, required=required)
            )
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        residual = bool(
            self.unresolved_mandatory_ids
            or self.open_required_frontier_ids
            or not set(self.original_finding_ids).issubset(self.discharged_original_ids)
            or not set(self.second_order_finding_ids).issubset(
                self.discharged_second_order_ids
            )
        )
        if self.complete and residual:
            raise DeterministicDoctorFixedPointError(
                "complete reclose forbids residual findings or frontiers"
            )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DOCTOR_RECLOSE_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "original_finding_ids": list(self.original_finding_ids),
            "discharged_original_ids": list(self.discharged_original_ids),
            "second_order_finding_ids": list(self.second_order_finding_ids),
            "discharged_second_order_ids": list(self.discharged_second_order_ids),
            "unresolved_mandatory_ids": list(self.unresolved_mandatory_ids),
            "open_required_frontier_ids": list(self.open_required_frontier_ids),
            "complete": self.complete,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class DoctorReplanEvidence:
    """Regenerated diagnosis / Tactician plan for residual clauses."""

    candidate_tree_id: str
    diagnosis_root_id: str
    tactician_plan_id: str
    goal_root_ids: tuple[str, ...]
    residual_gap_ids: tuple[str, ...]
    plan_current: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        object.__setattr__(
            self, "diagnosis_root_id", _identifier(self.diagnosis_root_id, "diagnosis_root_id")
        )
        object.__setattr__(
            self,
            "tactician_plan_id",
            _identifier(self.tactician_plan_id, "tactician_plan_id"),
        )
        object.__setattr__(
            self, "goal_root_ids", _ids(self.goal_root_ids, "goal_root_ids")
        )
        object.__setattr__(
            self, "residual_gap_ids", _ids(self.residual_gap_ids, "residual_gap_ids")
        )
        object.__setattr__(self, "plan_current", _bool(self.plan_current, "plan_current"))
        if self.plan_current and self.residual_gap_ids:
            raise DeterministicDoctorFixedPointError(
                "current tactician plan forbids residual gaps"
            )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DOCTOR_REPLAN_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "diagnosis_root_id": self.diagnosis_root_id,
            "tactician_plan_id": self.tactician_plan_id,
            "goal_root_ids": list(self.goal_root_ids),
            "residual_gap_ids": list(self.residual_gap_ids),
            "plan_current": self.plan_current,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class DoctorReproveEvidence:
    """Hammer / native-goal / prediction receipts for changed clauses."""

    candidate_tree_id: str
    hammer_receipt_ids: tuple[str, ...]
    native_goal_binding_ids: tuple[str, ...]
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
        if self.all_promoted_clauses_current and (
            self.stale_prediction_ids or self.failed_reconstruction_ids
        ):
            raise DeterministicDoctorFixedPointError(
                "current promoted clauses forbid stale predictions or failed reconstructions"
            )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DOCTOR_REPROVE_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "hammer_receipt_ids": list(self.hammer_receipt_ids),
            "native_goal_binding_ids": list(self.native_goal_binding_ids),
            "prediction_receipt_ids": list(self.prediction_receipt_ids),
            "stale_prediction_ids": list(self.stale_prediction_ids),
            "failed_reconstruction_ids": list(self.failed_reconstruction_ids),
            "all_promoted_clauses_current": self.all_promoted_clauses_current,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class DoctorFixedPointIterationReceipt:
    """Exact per-iteration doctor evidence for one fixed-point round."""

    iteration: int
    rebuild: DoctorRebuildEvidence
    cache_invalidation: DoctorCacheInvalidationEvidence
    static_checks: DoctorStaticCheckEvidence
    redelta: DoctorRedeltaEvidence
    reclose: DoctorRecloseEvidence
    replan: DoctorReplanEvidence
    reprove: DoctorReproveEvidence
    residual_finding_ids: tuple[str, ...] = ()
    oscillation_fingerprint: str = ""
    requires_another_iteration: bool = False
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "iteration", _bounded_int(self.iteration, "iteration", minimum=1)
        )
        for name, expected in (
            ("rebuild", DoctorRebuildEvidence),
            ("cache_invalidation", DoctorCacheInvalidationEvidence),
            ("static_checks", DoctorStaticCheckEvidence),
            ("redelta", DoctorRedeltaEvidence),
            ("reclose", DoctorRecloseEvidence),
            ("replan", DoctorReplanEvidence),
            ("reprove", DoctorReproveEvidence),
        ):
            value = getattr(self, name)
            if not isinstance(value, expected):
                raise DeterministicDoctorFixedPointError(
                    f"{name} must be {expected.__name__}"
                )
        object.__setattr__(
            self,
            "residual_finding_ids",
            _ids(self.residual_finding_ids, "residual_finding_ids"),
        )
        object.__setattr__(
            self,
            "oscillation_fingerprint",
            _optional_identifier(
                self.oscillation_fingerprint, "oscillation_fingerprint"
            )
            if self.oscillation_fingerprint
            else "",
        )
        if self.oscillation_fingerprint:
            object.__setattr__(
                self,
                "oscillation_fingerprint",
                _identifier(self.oscillation_fingerprint, "oscillation_fingerprint"),
            )
        object.__setattr__(
            self,
            "requires_another_iteration",
            _bool(self.requires_another_iteration, "requires_another_iteration"),
        )
        residual = bool(
            self.residual_finding_ids
            or self.requires_another_iteration
            or not self.reclose.complete
            or not self.redelta.matches_plan_delta
            or not self.replan.plan_current
            or not self.reprove.all_promoted_clauses_current
            or not self.static_checks.all_passed
            or not self.cache_invalidation.complete
            or not self.rebuild.clean_rebuild_equivalent
        )
        if self.requires_another_iteration and not residual:
            if not (
                self.reclose.second_order_finding_ids
                or self.residual_finding_ids
            ):
                raise DeterministicDoctorFixedPointError(
                    "requires_another_iteration needs residual doctor impacts"
                )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    @property
    def residual_free(self) -> bool:
        return (
            not self.requires_another_iteration
            and not self.residual_finding_ids
            and self.reclose.complete
            and self.redelta.matches_plan_delta
            and not self.redelta.unplanned_breaking_delta_ids
            and self.replan.plan_current
            and self.reprove.all_promoted_clauses_current
            and self.static_checks.all_passed
            and self.cache_invalidation.complete
            and self.rebuild.clean_rebuild_equivalent
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DOCTOR_ITERATION_RECEIPT_SCHEMA,
            "iteration": self.iteration,
            "rebuild": self.rebuild.to_dict(),
            "cache_invalidation": self.cache_invalidation.to_dict(),
            "static_checks": self.static_checks.to_dict(),
            "redelta": self.redelta.to_dict(),
            "reclose": self.reclose.to_dict(),
            "replan": self.replan.to_dict(),
            "reprove": self.reprove.to_dict(),
            "residual_finding_ids": list(self.residual_finding_ids),
            "oscillation_fingerprint": self.oscillation_fingerprint,
            "requires_another_iteration": self.requires_another_iteration,
            "residual_free": self.residual_free,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class CandidateDoctorFixedPointEvidence:
    """Multi-iteration doctor evidence bundle for fixed-point validation."""

    candidate_tree_id: str
    roots: DoctorAuthorityRoots
    iterations: tuple[DoctorFixedPointIterationReceipt, ...]
    expected_tombstone_ids: tuple[str, ...] = ()
    identity_replay_receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _identifier(self.candidate_tree_id, "candidate_tree_id"),
        )
        object.__setattr__(self, "roots", _roots(self.roots))
        if (
            not isinstance(self.iterations, Sequence)
            or not self.iterations
            or not all(
                isinstance(item, DoctorFixedPointIterationReceipt)
                for item in self.iterations
            )
        ):
            raise DeterministicDoctorFixedPointError(
                "iterations must be a non-empty DoctorFixedPointIterationReceipt sequence"
            )
        if len(self.iterations) > MAX_ITERATIONS:
            raise DeterministicDoctorFixedPointError("iterations exceed policy bound")
        object.__setattr__(self, "iterations", tuple(self.iterations))
        object.__setattr__(
            self,
            "expected_tombstone_ids",
            _ids(self.expected_tombstone_ids, "expected_tombstone_ids"),
        )
        object.__setattr__(
            self,
            "identity_replay_receipt_id",
            _optional_identifier(
                self.identity_replay_receipt_id, "identity_replay_receipt_id"
            ),
        )
        if self.roots.tree_id != self.candidate_tree_id:
            raise DeterministicDoctorFixedPointError(
                "authority tree_id must match candidate_tree_id"
            )


@dataclass(frozen=True)
class DoctorStageResult:
    stage: DoctorFixedPointStage
    disposition: DoctorStageDisposition
    reason_codes: tuple[str, ...] = ()
    receipt_id: str = ""
    iteration: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage", _enum(self.stage, DoctorFixedPointStage, "stage")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorStageDisposition, "disposition"),
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
            "stage": self.stage.value
            if isinstance(self.stage, DoctorFixedPointStage)
            else str(self.stage),
            "disposition": self.disposition.value
            if isinstance(self.disposition, DoctorStageDisposition)
            else str(self.disposition),
            "reason_codes": list(self.reason_codes),
            "receipt_id": self.receipt_id,
            "iteration": self.iteration,
        }


@dataclass(frozen=True)
class DoctorCompensatingRollbackReceipt:
    """Compensating rollback after provisional-commit fixed-point failure."""

    roots: DoctorAuthorityRoots
    rollback_id: str
    transaction_id: str
    plan_id: str
    checkpoint_id: str
    restored: bool
    reason_codes: tuple[str, ...]
    quarantined: bool = False
    iteration_count: int = 0
    diagnostic_refs: tuple[str, ...] = ()
    underlying_rollback_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        for name in ("rollback_id", "transaction_id", "plan_id", "checkpoint_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "restored", _bool(self.restored, "restored"))
        object.__setattr__(
            self, "quarantined", _bool(self.quarantined, "quarantined")
        )
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
        if not self.restored and not self.quarantined:
            raise DeterministicDoctorFixedPointError(
                "failed restore must quarantine; cannot claim clean compensating rollback"
            )
        if self.restored and self.quarantined:
            raise DeterministicDoctorFixedPointError(
                "restored compensating rollback cannot simultaneously quarantine"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_COMPENSATING_ROLLBACK_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_FIXED_POINT_INTERFACE,
            "roots": self.roots.to_dict(),
            "rollback_id": self.rollback_id,
            "transaction_id": self.transaction_id,
            "plan_id": self.plan_id,
            "checkpoint_id": self.checkpoint_id,
            "restored": self.restored,
            "quarantined": self.quarantined,
            "reason_codes": list(self.reason_codes),
            "iteration_count": self.iteration_count,
            "diagnostic_refs": list(self.diagnostic_refs),
            "underlying_rollback_id": self.underlying_rollback_id,
            "partial_merge_allowed": False,
            "claims_completion": False,
            "task_closed": False,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class DoctorFixedPointReceipt:
    """Authoritative residual-free fixed-point receipt for a doctor repair.

    Completeness requires no original or second-order mandatory finding, no
    open required frontier, current reconstructed obligations, and
    identity-equivalent replay.  Never authorizes model invocation.
    """

    roots: DoctorAuthorityRoots
    receipt_id: str
    plan_id: str
    transaction_id: str
    candidate_tree_cid: str
    committed_tree_cid: str
    checkpoint_id: str
    iteration_count: int
    disposition: DoctorFixedPointDisposition
    iteration_receipt_ids: tuple[str, ...]
    residual_finding_ids: tuple[str, ...] = ()
    open_frontier_ids: tuple[str, ...] = ()
    cache_invalidation_ids: tuple[str, ...] = ()
    rebuild_receipt_ids: tuple[str, ...] = ()
    replan_receipt_ids: tuple[str, ...] = ()
    reprove_receipt_ids: tuple[str, ...] = ()
    identity_replay_receipt_id: str = ""
    reason_codes: tuple[str, ...] = ()
    model_invocation_count: int = 0
    provider_invocation_count: int = 0
    resource_bounds: DoctorResourceBounds | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        for name in (
            "receipt_id",
            "plan_id",
            "transaction_id",
            "candidate_tree_cid",
            "committed_tree_cid",
            "checkpoint_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "iteration_count",
            _bounded_int(self.iteration_count, "iteration_count", minimum=1),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorFixedPointDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "iteration_receipt_ids",
            _ids(
                self.iteration_receipt_ids,
                "iteration_receipt_ids",
                required=True,
            ),
        )
        for name in (
            "residual_finding_ids",
            "open_frontier_ids",
            "cache_invalidation_ids",
            "rebuild_receipt_ids",
            "replan_receipt_ids",
            "reprove_receipt_ids",
            "reason_codes",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self,
            "identity_replay_receipt_id",
            _optional_identifier(
                self.identity_replay_receipt_id, "identity_replay_receipt_id"
            ),
        )
        object.__setattr__(
            self,
            "model_invocation_count",
            _bounded_int(self.model_invocation_count, "model_invocation_count", minimum=0),
        )
        object.__setattr__(
            self,
            "provider_invocation_count",
            _bounded_int(
                self.provider_invocation_count, "provider_invocation_count", minimum=0
            ),
        )
        if self.model_invocation_count != 0 or self.provider_invocation_count != 0:
            raise DeterministicDoctorFixedPointError(
                "fixed-point receipts forbid model/provider invocations"
            )
        if self.resource_bounds is None:
            object.__setattr__(self, "resource_bounds", DoctorResourceBounds())
        elif not isinstance(self.resource_bounds, DoctorResourceBounds):
            raise DeterministicDoctorFixedPointError(
                "resource_bounds must be DoctorResourceBounds"
            )
        if self.disposition is DoctorFixedPointDisposition.COMPLETE:
            if self.residual_finding_ids or self.open_frontier_ids or self.reason_codes:
                raise DeterministicDoctorFixedPointError(
                    "complete fixed-point receipt forbids residuals or reason codes"
                )
            if not self.identity_replay_receipt_id:
                raise DeterministicDoctorFixedPointError(
                    "complete fixed-point receipt requires identity-equivalent replay"
                )
        else:
            if self.disposition.claims_completion:
                raise DeterministicDoctorFixedPointError(
                    "non-complete disposition cannot claim completion"
                )

    @property
    def complete(self) -> bool:
        return self.disposition is DoctorFixedPointDisposition.COMPLETE

    def to_dict(self) -> dict[str, Any]:
        assert self.resource_bounds is not None
        return {
            "schema": DOCTOR_FIXED_POINT_RECEIPT_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_FIXED_POINT_INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "producer_id": PRODUCER_ID,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "plan_id": self.plan_id,
            "transaction_id": self.transaction_id,
            "candidate_tree_cid": self.candidate_tree_cid,
            "committed_tree_cid": self.committed_tree_cid,
            "checkpoint_id": self.checkpoint_id,
            "iteration_count": self.iteration_count,
            "disposition": self.disposition.value
            if isinstance(self.disposition, DoctorFixedPointDisposition)
            else str(self.disposition),
            "iteration_receipt_ids": list(self.iteration_receipt_ids),
            "residual_finding_ids": list(self.residual_finding_ids),
            "open_frontier_ids": list(self.open_frontier_ids),
            "cache_invalidation_ids": list(self.cache_invalidation_ids),
            "rebuild_receipt_ids": list(self.rebuild_receipt_ids),
            "replan_receipt_ids": list(self.replan_receipt_ids),
            "reprove_receipt_ids": list(self.reprove_receipt_ids),
            "identity_replay_receipt_id": self.identity_replay_receipt_id,
            "reason_codes": list(self.reason_codes),
            "model_invocation_count": 0,
            "provider_invocation_count": 0,
            "resource_bounds": self.resource_bounds.to_dict(),
            "partial_merge_allowed": False,
            "claims_completion": self.complete,
            "may_call_model": False,
        }

    @property
    def content_id(self) -> str:
        return content_identity({**self.to_dict(), "receipt_id": ""})


@dataclass(frozen=True)
class DoctorFixedPointReport:
    """Ordered doctor fixed-point report; success is not merge authority alone."""

    plan_id: str
    transaction_id: str
    candidate_tree_id: str
    roots: DoctorAuthorityRoots
    stages: tuple[DoctorStageResult, ...]
    reason_codes: tuple[str, ...]
    iteration_count: int
    complete: bool
    disposition: DoctorFixedPointDisposition
    iteration_receipts: tuple[DoctorFixedPointIterationReceipt, ...] = ()

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
        object.__setattr__(self, "roots", _roots(self.roots))
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
            self,
            "disposition",
            _enum(self.disposition, DoctorFixedPointDisposition, "disposition"),
        )
        object.__setattr__(self, "iteration_receipts", tuple(self.iteration_receipts))
        if self.complete and self.reason_codes:
            raise DeterministicDoctorFixedPointError(
                "a complete report cannot carry failure reason codes"
            )
        if self.complete and self.disposition is not DoctorFixedPointDisposition.COMPLETE:
            raise DeterministicDoctorFixedPointError(
                "complete report requires COMPLETE disposition"
            )
        if not self.complete and self.disposition is DoctorFixedPointDisposition.COMPLETE:
            raise DeterministicDoctorFixedPointError(
                "incomplete report cannot claim COMPLETE disposition"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_FIXED_POINT_REPORT_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_FIXED_POINT_INTERFACE,
            "plan_id": self.plan_id,
            "transaction_id": self.transaction_id,
            "candidate_tree_id": self.candidate_tree_id,
            "roots": self.roots.to_dict(),
            "stages": [item.to_dict() for item in self.stages],
            "reason_codes": list(self.reason_codes),
            "iteration_count": self.iteration_count,
            "complete": self.complete,
            "disposition": self.disposition.value
            if isinstance(self.disposition, DoctorFixedPointDisposition)
            else str(self.disposition),
            "iteration_receipt_ids": [
                item.receipt_id for item in self.iteration_receipts
            ],
            "partial_merge_allowed": False,
            "claims_completion": self.complete,
            "may_call_model": False,
            "provider_success_is_not_completion": True,
            "transaction_interface": DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE,
        }

    @property
    def report_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class DoctorFixedPointOutcome:
    """Fixed-point outcome: receipt or compensating rollback / quarantine."""

    report: DoctorFixedPointReport
    fixed_point: DoctorFixedPointReceipt | None = None
    compensating_rollback: DoctorCompensatingRollbackReceipt | None = None
    rolled_back: bool = False
    quarantined: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.report, DoctorFixedPointReport):
            raise DeterministicDoctorFixedPointError(
                "outcome requires a DoctorFixedPointReport"
            )
        if self.fixed_point is not None and not isinstance(
            self.fixed_point, DoctorFixedPointReceipt
        ):
            raise DeterministicDoctorFixedPointError(
                "fixed_point must be DoctorFixedPointReceipt"
            )
        if self.compensating_rollback is not None and not isinstance(
            self.compensating_rollback, DoctorCompensatingRollbackReceipt
        ):
            raise DeterministicDoctorFixedPointError(
                "compensating_rollback must be DoctorCompensatingRollbackReceipt"
            )
        object.__setattr__(self, "rolled_back", _bool(self.rolled_back, "rolled_back"))
        object.__setattr__(
            self, "quarantined", _bool(self.quarantined, "quarantined")
        )
        if self.report.complete:
            if self.fixed_point is None or not self.fixed_point.complete:
                raise DeterministicDoctorFixedPointError(
                    "complete report requires complete DoctorFixedPointReceipt"
                )
            if self.compensating_rollback is not None or self.rolled_back or self.quarantined:
                raise DeterministicDoctorFixedPointError(
                    "complete report cannot carry compensating rollback or quarantine"
                )
        if self.rolled_back and self.compensating_rollback is None:
            raise DeterministicDoctorFixedPointError(
                "rolled_back outcome requires DoctorCompensatingRollbackReceipt"
            )
        if self.quarantined and (
            self.compensating_rollback is None or not self.compensating_rollback.quarantined
        ):
            raise DeterministicDoctorFixedPointError(
                "quarantined outcome requires quarantined compensating rollback"
            )
        if self.fixed_point is not None and self.compensating_rollback is not None:
            raise DeterministicDoctorFixedPointError(
                "fixed_point receipt and compensating rollback are mutually exclusive"
            )
        # Neither incomplete nor quarantine may claim completion or call a model.
        if not self.report.complete:
            if self.report.disposition.claims_completion:
                raise DeterministicDoctorFixedPointError(
                    "incomplete outcome cannot claim completion"
                )
            if self.report.disposition.may_call_model:
                raise DeterministicDoctorFixedPointError(
                    "no fixed-point disposition may call a model"
                )

    @property
    def complete(self) -> bool:
        return (
            self.report.complete
            and self.fixed_point is not None
            and self.fixed_point.complete
            and not self.rolled_back
            and not self.quarantined
        )

    def require_complete(self) -> DoctorFixedPointReceipt:
        if not self.complete or self.fixed_point is None:
            reasons = ", ".join(self.report.reason_codes) or "incomplete"
            raise DeterministicDoctorFixedPointError(
                "deterministic doctor fixed-point validation rejected: " + reasons
            )
        return self.fixed_point

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "rolled_back": self.rolled_back,
            "quarantined": self.quarantined,
            "report": self.report.to_dict(),
            "fixed_point_id": self.fixed_point.receipt_id if self.fixed_point else "",
            "compensating_rollback_id": (
                self.compensating_rollback.rollback_id
                if self.compensating_rollback
                else ""
            ),
            "partial_merge_allowed": False,
            "claims_completion": self.complete,
            "may_call_model": False,
            "provider_success_is_not_completion": True,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


DoctorRestoreAdapter = Callable[[DoctorTransactionCheckpoint], bool]


def _default_restore(checkpoint: DoctorTransactionCheckpoint) -> bool:
    return True


@dataclass
class DeterministicDoctorFixedPointValidator:
    """Orchestrate post-edit doctor fixed-point after provisional commit.

    Always requires a committed :class:`DoctorTransactionReport` and multi-
    iteration :class:`CandidateDoctorFixedPointEvidence`.  Partial SCC/packet
    completion never yields COMPLETE.  Bound exhaustion, oscillation, drift,
    residual findings, or incomplete stage evidence trigger compensating
    rollback; restore failure quarantines.
    """

    INTERFACE: Final[str] = DETERMINISTIC_DOCTOR_FIXED_POINT_INTERFACE

    fixed_point_bound: int = DEFAULT_FIXED_POINT_BOUND
    restore_adapter: DoctorRestoreAdapter = field(default=_default_restore)
    oscillation_window: int = MAX_OSCILLATION_WINDOW

    def __post_init__(self) -> None:
        self.fixed_point_bound = _bounded_int(
            self.fixed_point_bound,
            "fixed_point_bound",
            minimum=1,
            maximum=MAX_ITERATIONS,
        )
        self.oscillation_window = _bounded_int(
            self.oscillation_window,
            "oscillation_window",
            minimum=2,
            maximum=MAX_OSCILLATION_WINDOW,
        )

    def validate(
        self,
        plan: DeterministicDoctorPlan,
        transaction_report: DoctorTransactionReport,
        *,
        evidence: CandidateDoctorFixedPointEvidence,
        fixed_point_bound: int | None = None,
        checkpoint: DoctorTransactionCheckpoint | None = None,
        restore_adapter: DoctorRestoreAdapter | None = None,
    ) -> DoctorFixedPointOutcome:
        """Run residual-free doctor fixed-point (fail-closed)."""

        stages: list[DoctorStageResult] = []
        reasons: set[str] = set()

        typed = (
            isinstance(plan, DeterministicDoctorPlan)
            and isinstance(transaction_report, DoctorTransactionReport)
            and isinstance(evidence, CandidateDoctorFixedPointEvidence)
        )
        if not typed:
            return self._malformed_outcome()

        bound = self.fixed_point_bound
        if fixed_point_bound is not None:
            bound = _bounded_int(
                fixed_point_bound, "fixed_point_bound", minimum=1, maximum=MAX_ITERATIONS
            )

        # --- Admission: provisional commit required ---
        if plan.disposition is not DoctorPlanDisposition.ADMITTED:
            reasons.add(DoctorFixedPointReason.PLAN_NOT_ADMITTED.value)
        if not transaction_report.committed:
            reasons.add(DoctorFixedPointReason.TRANSACTION_NOT_PROVISIONAL.value)
        if transaction_report.disposition is not DoctorTransactionDisposition.COMMITTED:
            reasons.add(DoctorFixedPointReason.TRANSACTION_NOT_PROVISIONAL.value)
        if transaction_report.partial_merge_allowed:
            reasons.add(DoctorFixedPointReason.PARTIAL_SCC_FORBIDDEN.value)
        if transaction_report.candidate_tree is None:
            reasons.add(DoctorFixedPointReason.INCOMPLETE_EVIDENCE.value)
        if transaction_report.model_invocation_count or transaction_report.provider_invocation_count:
            reasons.add(DoctorFixedPointReason.MODEL_INVOCATION_FORBIDDEN.value)

        # Partial packet: every plan step must appear in completed groups.
        if transaction_report.committed:
            completed_steps = {
                step.step_id
                for group in transaction_report.group_receipts
                for step in group.step_receipts
                if step.passed
            }
            expected_steps = {step.step_id for step in plan.steps}
            if completed_steps != expected_steps:
                reasons.add(DoctorFixedPointReason.PARTIAL_PACKET_FORBIDDEN.value)

        if plan.roots != evidence.roots:
            reasons.add(DoctorFixedPointReason.ROOT_DRIFT.value)
        if transaction_report.roots != plan.roots:
            reasons.add(DoctorFixedPointReason.ROOT_DRIFT.value)

        candidate_tree = (
            transaction_report.candidate_tree.candidate_tree_cid
            if transaction_report.candidate_tree is not None
            else plan.roots.tree_id
        )
        if evidence.candidate_tree_id != candidate_tree and evidence.candidate_tree_id != plan.roots.tree_id:
            reasons.add(DoctorFixedPointReason.STALE_CANDIDATE_TREE.value)

        stages.append(
            DoctorStageResult(
                DoctorFixedPointStage.ADMISSION,
                DoctorStageDisposition.FAILED if reasons else DoctorStageDisposition.PASSED,
                tuple(sorted(reasons)),
            )
        )
        if reasons:
            return self._rollback_or_incomplete(
                plan,
                transaction_report,
                stages,
                reasons,
                iteration_count=0,
                restore_adapter=restore_adapter,
                checkpoint=checkpoint or transaction_report.checkpoint,
            )

        # --- Iterate evidence ---
        accepted: list[DoctorFixedPointIterationReceipt] = []
        fingerprints: list[str] = []
        reached = False
        last_residuals: tuple[str, ...] = ()
        last_frontiers: tuple[str, ...] = ()

        for index, iteration in enumerate(evidence.iterations, start=1):
            if index > bound:
                reasons.add(DoctorFixedPointReason.BOUND_EXHAUSTED.value)
                break

            iter_reasons = self._validate_iteration(
                plan,
                transaction_report,
                iteration,
                expected_tombstones=evidence.expected_tombstone_ids,
            )
            stages.extend(iter_reasons[1])
            if iter_reasons[0]:
                reasons.update(iter_reasons[0])
                break

            accepted.append(iteration)
            fp = iteration.oscillation_fingerprint or iteration.receipt_id
            fingerprints.append(fp)

            # Oscillation: repeating fingerprint inside the window.
            if len(fingerprints) >= self.oscillation_window:
                window = fingerprints[-self.oscillation_window :]
                if len(set(window)) < self.oscillation_window and window[0] == window[-1]:
                    # Alternating or cycling residual fingerprints.
                    if window.count(window[0]) >= 2 and not iteration.residual_free:
                        reasons.add(DoctorFixedPointReason.OSCILLATION_DETECTED.value)
                        break

            last_residuals = iteration.residual_finding_ids
            last_frontiers = iteration.reclose.open_required_frontier_ids

            if iteration.residual_free:
                reached = True
                stages.append(
                    DoctorStageResult(
                        DoctorFixedPointStage.FIXED_POINT,
                        DoctorStageDisposition.PASSED,
                        (),
                        iteration.receipt_id,
                        iteration=iteration.iteration,
                    )
                )
                break

            if iteration.requires_another_iteration or not iteration.residual_free:
                stages.append(
                    DoctorStageResult(
                        DoctorFixedPointStage.RESIDUAL,
                        DoctorStageDisposition.CONTINUE,
                        (DoctorFixedPointReason.FIXED_POINT_NOT_REACHED.value,),
                        iteration.receipt_id,
                        iteration=iteration.iteration,
                    )
                )
                continue

        if not reached:
            if DoctorFixedPointReason.BOUND_EXHAUSTED.value not in reasons:
                if len(evidence.iterations) >= bound and not any(
                    item.residual_free for item in accepted
                ):
                    reasons.add(DoctorFixedPointReason.BOUND_EXHAUSTED.value)
                else:
                    reasons.add(DoctorFixedPointReason.FIXED_POINT_NOT_REACHED.value)
            if last_residuals:
                reasons.add(DoctorFixedPointReason.UNRESOLVED_MANDATORY_FINDING.value)
            if last_frontiers:
                reasons.add(DoctorFixedPointReason.UNCOVERED_FRONTIER.value)
            return self._rollback_or_incomplete(
                plan,
                transaction_report,
                stages,
                reasons,
                iteration_count=len(accepted),
                iteration_receipts=tuple(accepted),
                restore_adapter=restore_adapter,
                checkpoint=checkpoint or transaction_report.checkpoint,
            )

        # Identity-equivalent replay required for complete receipt.
        replay_id = evidence.identity_replay_receipt_id
        if not replay_id:
            reasons.add(DoctorFixedPointReason.IDENTITY_REPLAY_MISMATCH.value)
            return self._rollback_or_incomplete(
                plan,
                transaction_report,
                stages,
                reasons,
                iteration_count=len(accepted),
                iteration_receipts=tuple(accepted),
                restore_adapter=restore_adapter,
                checkpoint=checkpoint or transaction_report.checkpoint,
            )

        assert transaction_report.candidate_tree is not None
        tip = transaction_report.candidate_tree.candidate_tree_cid
        last = accepted[-1]
        receipt = DoctorFixedPointReceipt(
            roots=plan.roots,
            receipt_id=content_identity(
                {
                    "schema": DOCTOR_FIXED_POINT_RECEIPT_SCHEMA,
                    "plan_id": plan.plan_id,
                    "transaction_id": transaction_report.transaction_id,
                    "iterations": [item.receipt_id for item in accepted],
                }
            ),
            plan_id=plan.plan_id,
            transaction_id=transaction_report.transaction_id,
            candidate_tree_cid=tip,
            committed_tree_cid=tip,
            checkpoint_id=transaction_report.checkpoint.checkpoint_id,
            iteration_count=len(accepted),
            disposition=DoctorFixedPointDisposition.COMPLETE,
            iteration_receipt_ids=tuple(item.receipt_id for item in accepted),
            residual_finding_ids=(),
            open_frontier_ids=(),
            cache_invalidation_ids=(last.cache_invalidation.receipt_id,),
            rebuild_receipt_ids=(last.rebuild.receipt_id,),
            replan_receipt_ids=(last.replan.receipt_id,),
            reprove_receipt_ids=(last.reprove.receipt_id,),
            identity_replay_receipt_id=replay_id,
            reason_codes=(),
            resource_bounds=plan.resource_bounds,
        )
        stages.append(
            DoctorStageResult(
                DoctorFixedPointStage.FINALIZE,
                DoctorStageDisposition.FINALIZED,
                (),
                receipt.receipt_id,
                iteration=len(accepted),
            )
        )
        report = DoctorFixedPointReport(
            plan_id=plan.plan_id,
            transaction_id=transaction_report.transaction_id,
            candidate_tree_id=evidence.candidate_tree_id,
            roots=plan.roots,
            stages=tuple(stages),
            reason_codes=(),
            iteration_count=len(accepted),
            complete=True,
            disposition=DoctorFixedPointDisposition.COMPLETE,
            iteration_receipts=tuple(accepted),
        )
        return DoctorFixedPointOutcome(
            report=report,
            fixed_point=receipt,
            compensating_rollback=None,
            rolled_back=False,
            quarantined=False,
        )

    def require_complete(self, *args: Any, **kwargs: Any) -> DoctorFixedPointReceipt:
        outcome = self.validate(*args, **kwargs)
        return outcome.require_complete()

    # --- iteration validation ---------------------------------------------

    def _validate_iteration(
        self,
        plan: DeterministicDoctorPlan,
        transaction_report: DoctorTransactionReport,
        iteration: DoctorFixedPointIterationReceipt,
        *,
        expected_tombstones: Sequence[str],
    ) -> tuple[set[str], list[DoctorStageResult]]:
        reasons: set[str] = set()
        stages: list[DoctorStageResult] = []
        tree = iteration.rebuild.candidate_tree_id

        # Rebuild
        if not iteration.rebuild.clean_rebuild_equivalent:
            reasons.add(DoctorFixedPointReason.REBUILD_INCOMPLETE.value)
        if expected_tombstones:
            missing = set(expected_tombstones) - set(iteration.rebuild.tombstone_ids)
            if missing:
                reasons.add(DoctorFixedPointReason.TOMBSTONE_MISSING.value)
        stages.append(
            DoctorStageResult(
                DoctorFixedPointStage.REBUILD,
                DoctorStageDisposition.FAILED
                if DoctorFixedPointReason.REBUILD_INCOMPLETE.value in reasons
                or DoctorFixedPointReason.TOMBSTONE_MISSING.value in reasons
                else DoctorStageDisposition.PASSED,
                tuple(
                    sorted(
                        r
                        for r in reasons
                        if r
                        in {
                            DoctorFixedPointReason.REBUILD_INCOMPLETE.value,
                            DoctorFixedPointReason.TOMBSTONE_MISSING.value,
                        }
                    )
                ),
                iteration.rebuild.receipt_id,
                iteration=iteration.iteration,
            )
        )

        # Cache invalidation
        if not iteration.cache_invalidation.complete:
            reasons.add(DoctorFixedPointReason.CACHE_INVALIDATION_INCOMPLETE.value)
        stages.append(
            DoctorStageResult(
                DoctorFixedPointStage.CACHE_INVALIDATION,
                DoctorStageDisposition.FAILED
                if DoctorFixedPointReason.CACHE_INVALIDATION_INCOMPLETE.value in reasons
                else DoctorStageDisposition.PASSED,
                tuple(
                    sorted(
                        r
                        for r in reasons
                        if r == DoctorFixedPointReason.CACHE_INVALIDATION_INCOMPLETE.value
                    )
                ),
                iteration.cache_invalidation.receipt_id,
                iteration=iteration.iteration,
            )
        )

        # Static / differential / proof / memory / resource
        if not iteration.static_checks.all_passed:
            for failed in iteration.static_checks.failed_check_ids:
                token = failed.lower()
                if "type" in token:
                    reasons.add(DoctorFixedPointReason.TYPE_CHECK_FAILED.value)
                elif "diff" in token:
                    reasons.add(DoctorFixedPointReason.DIFFERENTIAL_CHECK_FAILED.value)
                elif "proof" in token:
                    reasons.add(DoctorFixedPointReason.PROOF_CHECK_FAILED.value)
                elif "memory" in token or "effect" in token:
                    reasons.add(DoctorFixedPointReason.MEMORY_EFFECT_FAILED.value)
                elif "resource" in token:
                    reasons.add(DoctorFixedPointReason.RESOURCE_CHECK_FAILED.value)
                elif "parse" in token or "reparse" in token:
                    reasons.add(DoctorFixedPointReason.REPARSE_FAILED.value)
                else:
                    reasons.add(DoctorFixedPointReason.STATIC_CHECK_FAILED.value)
            if not any(
                r.startswith(("type_", "diff", "proof", "memory", "resource", "reparse", "static"))
                or r
                in {
                    DoctorFixedPointReason.TYPE_CHECK_FAILED.value,
                    DoctorFixedPointReason.DIFFERENTIAL_CHECK_FAILED.value,
                    DoctorFixedPointReason.PROOF_CHECK_FAILED.value,
                    DoctorFixedPointReason.MEMORY_EFFECT_FAILED.value,
                    DoctorFixedPointReason.RESOURCE_CHECK_FAILED.value,
                    DoctorFixedPointReason.REPARSE_FAILED.value,
                    DoctorFixedPointReason.STATIC_CHECK_FAILED.value,
                }
                for r in reasons
            ):
                reasons.add(DoctorFixedPointReason.STATIC_CHECK_FAILED.value)
        stages.append(
            DoctorStageResult(
                DoctorFixedPointStage.STATIC_DIFFERENTIAL,
                DoctorStageDisposition.PASSED
                if iteration.static_checks.all_passed
                else DoctorStageDisposition.FAILED,
                ()
                if iteration.static_checks.all_passed
                else tuple(
                    sorted(
                        r
                        for r in reasons
                        if r
                        in {
                            DoctorFixedPointReason.TYPE_CHECK_FAILED.value,
                            DoctorFixedPointReason.DIFFERENTIAL_CHECK_FAILED.value,
                            DoctorFixedPointReason.PROOF_CHECK_FAILED.value,
                            DoctorFixedPointReason.MEMORY_EFFECT_FAILED.value,
                            DoctorFixedPointReason.RESOURCE_CHECK_FAILED.value,
                            DoctorFixedPointReason.REPARSE_FAILED.value,
                            DoctorFixedPointReason.STATIC_CHECK_FAILED.value,
                        }
                    )
                ),
                iteration.static_checks.receipt_id,
                iteration=iteration.iteration,
            )
        )

        # Redelta
        if not iteration.redelta.matches_plan_delta:
            reasons.add(DoctorFixedPointReason.DELTA_RECOMPUTE_FAILED.value)
        if iteration.redelta.unplanned_breaking_delta_ids:
            reasons.add(DoctorFixedPointReason.UNPLANNED_BREAKING_DELTA.value)
        stages.append(
            DoctorStageResult(
                DoctorFixedPointStage.REDELTA,
                DoctorStageDisposition.PASSED
                if iteration.redelta.matches_plan_delta
                and not iteration.redelta.unplanned_breaking_delta_ids
                else DoctorStageDisposition.FAILED,
                ()
                if iteration.redelta.matches_plan_delta
                and not iteration.redelta.unplanned_breaking_delta_ids
                else tuple(
                    sorted(
                        r
                        for r in reasons
                        if r
                        in {
                            DoctorFixedPointReason.DELTA_RECOMPUTE_FAILED.value,
                            DoctorFixedPointReason.UNPLANNED_BREAKING_DELTA.value,
                        }
                    )
                ),
                iteration.redelta.receipt_id,
                iteration=iteration.iteration,
            )
        )

        # Reclose
        if not iteration.reclose.complete and not iteration.requires_another_iteration:
            if iteration.reclose.unresolved_mandatory_ids:
                reasons.add(DoctorFixedPointReason.UNRESOLVED_MANDATORY_FINDING.value)
            if iteration.reclose.open_required_frontier_ids:
                reasons.add(DoctorFixedPointReason.UNCOVERED_FRONTIER.value)
            if iteration.reclose.second_order_finding_ids and not set(
                iteration.reclose.second_order_finding_ids
            ).issubset(iteration.reclose.discharged_second_order_ids):
                reasons.add(DoctorFixedPointReason.SECOND_ORDER_FINDING_OPEN.value)
            if not iteration.reclose.complete:
                reasons.add(DoctorFixedPointReason.CLOSURE_RECOMPUTE_FAILED.value)
        stages.append(
            DoctorStageResult(
                DoctorFixedPointStage.RECLOSE,
                DoctorStageDisposition.PASSED
                if iteration.reclose.complete
                or iteration.requires_another_iteration
                else DoctorStageDisposition.FAILED,
                (),
                iteration.reclose.receipt_id,
                iteration=iteration.iteration,
            )
        )

        # Replan
        if not iteration.replan.plan_current and not iteration.requires_another_iteration:
            reasons.add(DoctorFixedPointReason.REPLAN_STALE.value)
            reasons.add(DoctorFixedPointReason.TACTICIAN_PLAN_STALE.value)
        stages.append(
            DoctorStageResult(
                DoctorFixedPointStage.REPLAN,
                DoctorStageDisposition.PASSED
                if iteration.replan.plan_current
                or iteration.requires_another_iteration
                else DoctorStageDisposition.FAILED,
                (),
                iteration.replan.receipt_id,
                iteration=iteration.iteration,
            )
        )

        # Reprove
        if not iteration.reprove.all_promoted_clauses_current and not iteration.requires_another_iteration:
            reasons.add(DoctorFixedPointReason.REPROVE_FAILED.value)
            if not iteration.reprove.hammer_receipt_ids:
                reasons.add(DoctorFixedPointReason.HAMMER_RECEIPT_MISSING.value)
            if iteration.reprove.stale_prediction_ids:
                reasons.add(DoctorFixedPointReason.PREDICTION_STALE.value)
        stages.append(
            DoctorStageResult(
                DoctorFixedPointStage.REPROVE,
                DoctorStageDisposition.PASSED
                if iteration.reprove.all_promoted_clauses_current
                or iteration.requires_another_iteration
                else DoctorStageDisposition.FAILED,
                (),
                iteration.reprove.receipt_id,
                iteration=iteration.iteration,
            )
        )

        # Tree identity consistency across stage evidence.
        for block in (
            iteration.cache_invalidation,
            iteration.static_checks,
            iteration.redelta,
            iteration.reclose,
            iteration.replan,
            iteration.reprove,
        ):
            if block.candidate_tree_id != tree:
                reasons.add(DoctorFixedPointReason.STALE_CANDIDATE_TREE.value)

        # Drift vs plan roots tree.
        if tree != plan.roots.tree_id and (
            transaction_report.candidate_tree is None
            or tree != transaction_report.candidate_tree.candidate_tree_cid
        ):
            # Allow evidence to use plan.roots.tree_id as the candidate identity.
            if tree != plan.roots.tree_id:
                reasons.add(DoctorFixedPointReason.DRIFT_DETECTED.value)

        # Hard failures abort the iteration even when more rounds are requested.
        hard = reasons & {
            DoctorFixedPointReason.REBUILD_INCOMPLETE.value,
            DoctorFixedPointReason.TOMBSTONE_MISSING.value,
            DoctorFixedPointReason.CACHE_INVALIDATION_INCOMPLETE.value,
            DoctorFixedPointReason.TYPE_CHECK_FAILED.value,
            DoctorFixedPointReason.DIFFERENTIAL_CHECK_FAILED.value,
            DoctorFixedPointReason.PROOF_CHECK_FAILED.value,
            DoctorFixedPointReason.MEMORY_EFFECT_FAILED.value,
            DoctorFixedPointReason.RESOURCE_CHECK_FAILED.value,
            DoctorFixedPointReason.REPARSE_FAILED.value,
            DoctorFixedPointReason.STATIC_CHECK_FAILED.value,
            DoctorFixedPointReason.DELTA_RECOMPUTE_FAILED.value,
            DoctorFixedPointReason.UNPLANNED_BREAKING_DELTA.value,
            DoctorFixedPointReason.STALE_CANDIDATE_TREE.value,
            DoctorFixedPointReason.DRIFT_DETECTED.value,
        }
        # Soft residuals (unresolved findings while continuing) are allowed.
        soft = reasons - hard
        if hard:
            return hard | soft, stages
        if iteration.requires_another_iteration:
            return set(), stages
        if soft and not iteration.residual_free:
            return soft, stages
        return set(), stages

    # --- failure / rollback paths -----------------------------------------

    def _malformed_outcome(self) -> DoctorFixedPointOutcome:
        # Minimal synthetic roots for malformed path.
        roots = DoctorAuthorityRoots(
            repository_id="repository:malformed",
            forest_id="forest:malformed",
            tree_id="tree:malformed",
            overlay_id="overlay:malformed",
            file_root_id="file-root:malformed",
            ast_root_id="ast:malformed",
            graph_id="graph:malformed",
            corpus_id="corpus:malformed",
            index_id="index:malformed",
            model_id="model:malformed",
            cache_id="cache:malformed",
            operator_registry_id="operators:malformed",
            translator_id="translator:malformed",
            solver_id="solver:malformed",
            kernel_id="kernel:malformed",
            toolchain_id="toolchain:malformed",
            policy_id="policy:malformed",
            sandbox_id="sandbox:malformed",
            environment_id="environment:malformed",
        )
        report = DoctorFixedPointReport(
            plan_id="plan:malformed",
            transaction_id="txn:malformed",
            candidate_tree_id="tree:malformed",
            roots=roots,
            stages=(
                DoctorStageResult(
                    DoctorFixedPointStage.ADMISSION,
                    DoctorStageDisposition.FAILED,
                    (DoctorFixedPointReason.MALFORMED_INPUT.value,),
                ),
            ),
            reason_codes=(DoctorFixedPointReason.MALFORMED_INPUT.value,),
            iteration_count=0,
            complete=False,
            disposition=DoctorFixedPointDisposition.INCOMPLETE,
        )
        return DoctorFixedPointOutcome(
            report=report,
            fixed_point=None,
            compensating_rollback=None,
            rolled_back=False,
            quarantined=False,
        )

    def _rollback_or_incomplete(
        self,
        plan: DeterministicDoctorPlan,
        transaction_report: DoctorTransactionReport,
        stages: list[DoctorStageResult],
        reasons: set[str],
        *,
        iteration_count: int,
        iteration_receipts: tuple[DoctorFixedPointIterationReceipt, ...] = (),
        restore_adapter: DoctorRestoreAdapter | None,
        checkpoint: DoctorTransactionCheckpoint | None,
    ) -> DoctorFixedPointOutcome:
        reasons.add(DoctorFixedPointReason.ROLLBACK_REQUIRED.value)
        adapter = restore_adapter or self.restore_adapter
        ckpt = checkpoint or transaction_report.checkpoint
        restored = False
        quarantined = False
        try:
            restored = bool(adapter(ckpt))
        except Exception:  # noqa: BLE001
            restored = False
        if not restored:
            quarantined = True
            reasons.add(DoctorFixedPointReason.ROLLBACK_FAILED.value)
            reasons.add(DoctorFixedPointReason.QUARANTINE_REQUIRED.value)
            stages.append(
                DoctorStageResult(
                    DoctorFixedPointStage.QUARANTINE,
                    DoctorStageDisposition.QUARANTINED,
                    tuple(sorted(reasons)),
                )
            )
            disposition = DoctorFixedPointDisposition.QUARANTINED
        else:
            stages.append(
                DoctorStageResult(
                    DoctorFixedPointStage.COMPENSATING_ROLLBACK,
                    DoctorStageDisposition.ROLLED_BACK,
                    tuple(sorted(reasons)),
                )
            )
            disposition = DoctorFixedPointDisposition.ROLLED_BACK

        rollback = DoctorCompensatingRollbackReceipt(
            roots=plan.roots,
            rollback_id=content_identity(
                {
                    "schema": DOCTOR_COMPENSATING_ROLLBACK_SCHEMA,
                    "transaction_id": transaction_report.transaction_id,
                    "checkpoint_id": ckpt.checkpoint_id,
                    "reasons": sorted(reasons),
                }
            ),
            transaction_id=transaction_report.transaction_id,
            plan_id=plan.plan_id,
            checkpoint_id=ckpt.checkpoint_id,
            restored=restored,
            reason_codes=tuple(sorted(reasons)),
            quarantined=quarantined,
            iteration_count=iteration_count,
            underlying_rollback_id=(
                transaction_report.rollback.rollback_id
                if transaction_report.rollback is not None
                else ""
            ),
        )
        report = DoctorFixedPointReport(
            plan_id=plan.plan_id,
            transaction_id=transaction_report.transaction_id,
            candidate_tree_id=plan.roots.tree_id,
            roots=plan.roots,
            stages=tuple(stages),
            reason_codes=tuple(sorted(reasons)),
            iteration_count=iteration_count,
            complete=False,
            disposition=disposition,
            iteration_receipts=iteration_receipts,
        )
        return DoctorFixedPointOutcome(
            report=report,
            fixed_point=None,
            compensating_rollback=rollback,
            rolled_back=restored,
            quarantined=quarantined,
        )


def validate_deterministic_doctor_fixed_point(
    plan: DeterministicDoctorPlan,
    transaction_report: DoctorTransactionReport,
    *,
    evidence: CandidateDoctorFixedPointEvidence,
    **kwargs: Any,
) -> DoctorFixedPointOutcome:
    """Module-level convenience wrapper."""

    return DeterministicDoctorFixedPointValidator().validate(
        plan, transaction_report, evidence=evidence, **kwargs
    )


def daemon_require_doctor_fixed_point(
    plan: DeterministicDoctorPlan,
    transaction_report: DoctorTransactionReport,
    *,
    evidence: CandidateDoctorFixedPointEvidence,
    **kwargs: Any,
) -> DoctorFixedPointReceipt:
    """Daemon gate: raise unless residual-free fixed point is reached."""

    return DeterministicDoctorFixedPointValidator().require_complete(
        plan, transaction_report, evidence=evidence, **kwargs
    )


__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_FIXED_POINT_BOUND",
    "DETERMINISTIC_DOCTOR_FIXED_POINT_INTERFACE",
    "PRODUCER_ID",
    "CandidateDoctorFixedPointEvidence",
    "DeterministicDoctorFixedPointError",
    "DeterministicDoctorFixedPointValidator",
    "DoctorCacheInvalidationEvidence",
    "DoctorCompensatingRollbackReceipt",
    "DoctorFixedPointDisposition",
    "DoctorFixedPointIterationReceipt",
    "DoctorFixedPointOutcome",
    "DoctorFixedPointReason",
    "DoctorFixedPointReceipt",
    "DoctorFixedPointReport",
    "DoctorFixedPointStage",
    "DoctorRebuildEvidence",
    "DoctorRecloseEvidence",
    "DoctorRedeltaEvidence",
    "DoctorReplanEvidence",
    "DoctorReproveEvidence",
    "DoctorStageDisposition",
    "DoctorStageResult",
    "DoctorStaticCheckEvidence",
    "daemon_require_doctor_fixed_point",
    "validate_deterministic_doctor_fixed_point",
]
